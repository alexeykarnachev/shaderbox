"""CPU-script engine (feature 041) — pure, no GL. A SimpleNamespace stands in for
moderngl.Uniform (the coercion/shape logic is GL-free; the GL write reaching the GPU is verified
in test_script_engine_gl.py). Covers the stateful-class contract: state accumulates, state resets
on edit + manual reset (VALUE back to baseline), export-instance isolation (live state never poisons
the export), the typed outputs (bare scalar / Vec*/Array/Text) coercing, shape-mismatch + compile
errors as data at the user line, the (path, mtime) cache, and scoped determinism (ctx.t-pure same,
integrator diverges by design).
"""

import dataclasses
import time
import types
from pathlib import Path

import pytest

from shaderbox.scripting import (
    EXPORT_MOUSE,
    EngineContext,
    MouseState,
    ScriptEngine,
    brain_stub_for,
    is_scriptable,
    parse_script_filename,
    per_uniform_filename,
    stub_for,
    uniform_tag,
)

_GL_FLOAT = 0x1406
_GL_UNSIGNED_INT = 0x1405
_GL_INT = 0x1404
_GL_INT_VEC2 = 0x8B53


def _u(
    name: str, dim: int = 1, n: int = 1, gl_type: int = _GL_FLOAT
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        name=name, dimension=dim, array_length=n, gl_type=gl_type, value=0.0
    )


class _FakeNode:
    # The EngineNode slice: a uniform_values dict + get_active_uniforms().
    def __init__(self, uniforms: list[types.SimpleNamespace]) -> None:
        self.uniform_values: dict[str, object] = {}
        self._uniforms = uniforms

    def get_active_uniforms(self) -> list[types.SimpleNamespace]:
        return self._uniforms


def _write(tmp: Path, name: str, body: str, tag: str = "float") -> Path:
    # Write a per-uniform script under the 047 tagged scheme `u_<name>__<tag>.py`. `tag` defaults to
    # "float" (the common scalar case); a test with a vec/array/text uniform passes the matching tag.
    scripts_dir = tmp / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    path = scripts_dir / f"{name}__{tag}.py"
    path.write_text(body, encoding="utf-8")
    return path


def _write_brain(tmp: Path, body: str) -> Path:
    # The node-brain file (feature 044): nodes/<id>/scripts/script.py — one class driving many
    # uniforms via a dict return. The dot keeps it out of the u_*__*.py glob.
    scripts_dir = tmp / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    path = scripts_dir / "script.py"
    path.write_text(body, encoding="utf-8")
    return path


def _all_active(tmp: Path) -> set[str]:
    # Every script file on disk is active (047) — the default for tests that don't exercise the
    # active/inactive flag itself. A test of inactive behavior passes a restricted set instead.
    scripts_dir = tmp / "scripts"
    if not scripts_dir.is_dir():
        return set()
    return {p.name for p in scripts_dir.glob("*.py")}


def _reload(eng: ScriptEngine, name: str, tmp: Path, node: _FakeNode) -> None:
    # reload with every on-disk script activated (047) — the test default.
    eng.reload(name, tmp / "scripts", node, _all_active(tmp))


def _ctx(t: float, dt: float = 1 / 60, frame: int = 0) -> EngineContext:
    return EngineContext(t=t, dt=dt, frame=frame)


def _engine(tmp: Path, node: _FakeNode, name: str = "n0") -> ScriptEngine:
    eng = ScriptEngine()
    _reload(eng, name, tmp, node)
    return eng


# A scalar class body returning a bare float each frame.
_SCALAR = (
    "import math\n"
    "class Behavior(ScriptBehavior):\n"
    "    def update(self, ctx: Ctx) -> float:\n"
    "        return 0.5 + 0.3 * math.sin(ctx.t)\n"
)
# A stateful integrator — only possible with per-instance state.
_INTEGRATOR = (
    "class Behavior(ScriptBehavior):\n"
    "    def __init__(self) -> None:\n"
    "        self.v = 0.0\n"
    "    def update(self, ctx: Ctx) -> float:\n"
    "        self.v += ctx.dt\n"
    "        return self.v\n"
)


# ---- basic compute + output types ----


def test_scalar_body(tmp_path: Path) -> None:
    _write(tmp_path, "u_wave", _SCALAR)
    node = _FakeNode([_u("u_wave")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert abs(node.uniform_values["u_wave"] - 0.5) < 1e-9


def test_vec2_output(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "u_off",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> Vec2:\n"
        "        return Vec2(0.3, 0.7)\n",
        tag="vec2",
    )
    node = _FakeNode([_u("u_off", dim=2)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_off"] == (0.3, 0.7)


def test_vec3_output(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "u_color",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> Vec3:\n"
        "        return Vec3(0.1, 0.2, 0.3)\n",
        tag="vec3",
    )
    node = _FakeNode([_u("u_color", dim=3)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_color"] == (0.1, 0.2, 0.3)


def test_array_output(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "u_vals",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> Array:\n"
        "        return Array([1.0, 2.0, 3.0, 4.0])\n",
        tag="array",
    )
    node = _FakeNode([_u("u_vals", dim=1, n=4)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert tuple(node.uniform_values["u_vals"]) == (1.0, 2.0, 3.0, 4.0)


def test_array_wrong_length_freezes(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "u_vals",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> Array:\n"
        "        return Array([1.0, 2.0])\n",  # 2 for a float[4]
        tag="array",
    )
    node = _FakeNode([_u("u_vals", dim=1, n=4)])
    node.uniform_values["u_vals"] = (0.0, 0.0, 0.0, 0.0)
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert tuple(node.uniform_values["u_vals"]) == (0.0, 0.0, 0.0, 0.0)  # frozen
    assert eng.errors[("n0", "u_vals")].kind == "runtime"


def test_text_output(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "u_text",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> Text:\n"
        '        return Text("Hi")\n',
        tag="text",
    )
    node = _FakeNode([_u("u_text", dim=1, n=8, gl_type=_GL_UNSIGNED_INT)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    value = node.uniform_values["u_text"]
    assert value[0] == ord("H") and value[1] == ord("i")
    assert len(value) == 8  # null-padded to the cap


# ---- stateful contract ----


def test_state_accumulates(tmp_path: Path) -> None:
    _write(tmp_path, "u_x", _INTEGRATOR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    for i in range(5):
        eng.tick("n0", node, _ctx(i / 60, dt=1.0, frame=i))
    assert node.uniform_values["u_x"] == 5.0  # 5 ticks of dt=1.0 accumulated on self.v


def test_state_resets_on_edit(tmp_path: Path) -> None:
    # Accumulate, then edit the file -> a recompile makes a fresh instance -> state back to baseline.
    path = _write(tmp_path, "u_x", _INTEGRATOR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    for i in range(3):
        eng.tick("n0", node, _ctx(0.0, dt=1.0, frame=i))
    assert node.uniform_values["u_x"] == 3.0

    time.sleep(0.01)
    path.write_text(_INTEGRATOR + "        # edited\n", encoding="utf-8")
    _reload(eng, "n0", tmp_path, node)
    eng.tick("n0", node, _ctx(0.0, dt=1.0, frame=0))
    assert node.uniform_values["u_x"] == 1.0  # fresh instance: self.v was 0, +1 dt


def test_manual_reset_clears_state(tmp_path: Path) -> None:
    _write(tmp_path, "u_x", _INTEGRATOR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    for i in range(4):
        eng.tick("n0", node, _ctx(0.0, dt=1.0, frame=i))
    assert node.uniform_values["u_x"] == 4.0
    eng.reset("n0", "u_x")
    eng.tick("n0", node, _ctx(0.0, dt=1.0, frame=0))
    assert node.uniform_values["u_x"] == 1.0  # state cleared, +1 dt


def test_export_instance_isolated_from_live(tmp_path: Path) -> None:
    # The headline contract: accumulate on the LIVE instance, then a fresh export set ticks from a
    # clean __init__ — the export value must NOT inherit the live-accumulated state. (Pure-CPU.)
    _write(tmp_path, "u_x", _INTEGRATOR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    for i in range(10):
        eng.tick("n0", node, _ctx(0.0, dt=1.0, frame=i))
    live_value = node.uniform_values["u_x"]
    assert live_value == 10.0

    fresh = eng.fresh_behaviors_for("n0")
    export_node = _FakeNode([_u("u_x")])
    eng.tick_behaviors("n0", export_node, _ctx(0.0, dt=1.0, frame=0), fresh)
    assert export_node.uniform_values["u_x"] == 1.0  # cold start, NOT the live 10.0
    assert export_node.uniform_values["u_x"] != live_value


# ---- errors as data ----


def test_syntax_error_is_compile_error_at_user_line(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "u_x",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> float:\n"
        "        return 0.5 +\n",  # incomplete expression on line 3
    )
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    err = eng.errors[("n0", "u_x")]
    assert err.kind == "compile" and err.line == 3


def test_no_subclass_is_compile_error(tmp_path: Path) -> None:
    _write(tmp_path, "u_x", "x = 1\n")  # no ScriptBehavior subclass at all
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    assert eng.errors[("n0", "u_x")].kind == "compile"


def test_no_update_override_is_compile_error(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "u_x",
        "class Behavior(ScriptBehavior):\n    pass\n",  # never overrides update
    )
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    assert eng.errors[("n0", "u_x")].kind == "compile"


def test_raising_init_is_compile_error(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "u_x",
        "class Behavior(ScriptBehavior):\n"
        "    def __init__(self) -> None:\n"
        "        raise ValueError('boom')\n"
        "    def update(self, ctx: Ctx) -> float:\n"
        "        return 0.0\n",
    )
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    assert eng.errors[("n0", "u_x")].kind == "compile"


def test_reset_recovers_a_once_failing_init(tmp_path: Path) -> None:
    # A raising __init__ freezes (compile error); after the cause clears, reset() must re-instantiate
    # AND clear the stale error so the binding unfreezes (else it stays frozen forever). A class var
    # raises on the FIRST construct only — the second (reset) build succeeds.
    body = (
        "class Behavior(ScriptBehavior):\n"
        "    _seen = False\n"
        "    def __init__(self) -> None:\n"
        "        if not Behavior._seen:\n"
        "            Behavior._seen = True\n"
        "            raise ValueError('boom')\n"
        "    def update(self, ctx: Ctx) -> float:\n"
        "        return 0.7\n"
    )
    _write(tmp_path, "u_x", body)
    node = _FakeNode([_u("u_x")])
    node.uniform_values["u_x"] = 0.0
    eng = _engine(tmp_path, node)
    assert eng.errors[("n0", "u_x")].kind == "compile"  # first __init__ raised

    eng.reset("n0", "u_x")  # second construct succeeds
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_x"] == 0.7  # unfrozen
    assert ("n0", "u_x") not in eng.errors  # stale error cleared


def test_user_raised_builtin_exception_is_its_real_error(tmp_path: Path) -> None:
    # A user `raise ValueError(...)` surfaces as its real error (real builtins are in scope).
    _write(
        tmp_path,
        "u_x",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> float:\n"
        "        raise ValueError('nope')\n",
    )
    node = _FakeNode([_u("u_x")])
    node.uniform_values["u_x"] = 0.0
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    err = eng.errors[("n0", "u_x")]
    assert err.kind == "runtime" and "ValueError" in err.message


def test_runtime_error_freezes_last_good(tmp_path: Path) -> None:
    path = _write(tmp_path, "u_x", _SCALAR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    good = node.uniform_values["u_x"]

    time.sleep(0.01)
    path.write_text(
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> float:\n"
        "        return 1.0 / 0.0\n",  # ZeroDivisionError at runtime
        encoding="utf-8",
    )
    _reload(eng, "n0", tmp_path, node)
    eng.tick("n0", node, _ctx(0.1))
    assert node.uniform_values["u_x"] == good  # frozen at last-good
    assert eng.errors[("n0", "u_x")].kind == "runtime"


def test_shape_mismatch_freezes_and_records(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "u_x",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> Vec3:\n"
        "        return Vec3(1.0, 2.0, 3.0)\n",  # vec3 into a scalar uniform
    )
    node = _FakeNode([_u("u_x")])
    node.uniform_values["u_x"] = 0.0
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_x"] == 0.0  # frozen
    assert eng.errors[("n0", "u_x")].kind == "runtime"


def test_none_return_freezes(tmp_path: Path) -> None:
    # A body that forgets to return -> None -> coercion rejects -> freeze, not a silent hold.
    _write(
        tmp_path,
        "u_x",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> float:\n"
        "        pass\n",
    )
    node = _FakeNode([_u("u_x")])
    node.uniform_values["u_x"] = 0.0
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_x"] == 0.0
    assert eng.errors[("n0", "u_x")].kind == "runtime"


def test_orphan_script_warns_once_and_is_ignored(tmp_path: Path) -> None:
    _write(tmp_path, "u_ghost", _SCALAR)  # no such uniform on the node
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    assert "u_ghost" not in eng.script_driven_uniforms("n0")
    # warn-once dedup: the orphan is recorded so a per-frame reload doesn't re-log it.
    assert eng._nodes["n0"].warned == {"u_ghost"}
    _reload(eng, "n0", tmp_path, node)  # second poll
    assert eng._nodes["n0"].warned == {"u_ghost"}  # unchanged, not re-warned


# ---- scoped determinism ----


def test_t_pure_script_is_deterministic(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "u_x",
        "import math\n"
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> float:\n"
        "        return math.sin(ctx.t)\n",
    )
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(1.234, dt=1 / 30))
    a = node.uniform_values["u_x"]
    eng.tick("n0", node, _ctx(1.234, dt=1 / 120))  # different dt, same t
    assert node.uniform_values["u_x"] == a


def test_integrator_diverges_by_design(tmp_path: Path) -> None:
    # A self-reading nonlinear integrator is path-dependent: the SAME elapsed time reached via
    # different dt yields different values (live variable-dt vs export fixed-dt). Documented as
    # expected (determinism is scoped to ctx.t-pure scripts), not a violation. Reads self.prev now
    # (the old ctx.uniforms snapshot is gone — state lives in the instance).
    body = (
        "class Behavior(ScriptBehavior):\n"
        "    def __init__(self) -> None:\n"
        "        self.prev = 1.0\n"
        "    def update(self, ctx: Ctx) -> float:\n"
        "        self.prev = self.prev + self.prev * ctx.dt\n"
        "        return self.prev\n"
    )
    _write(tmp_path, "u_x", body)
    node_a = _FakeNode([_u("u_x")])
    node_b = _FakeNode([_u("u_x")])
    eng = ScriptEngine()
    _reload(eng, "a", tmp_path, node_a)
    _reload(eng, "b", tmp_path, node_b)

    for _i in range(2):  # two steps of dt=0.5 (variable-dt live path) over 1.0s
        eng.tick("a", node_a, EngineContext(t=0.0, dt=0.5, frame=0))
    eng.tick("b", node_b, EngineContext(t=0.0, dt=1.0, frame=0))  # one step of dt=1.0

    # 1*(1.5)^2 = 2.25 vs 1*(1+1) = 2.0 — divergent by design.
    assert node_a.uniform_values["u_x"] != node_b.uniform_values["u_x"]


# ---- (path, mtime) cache ----


def test_cache_no_recompile_when_mtime_unchanged(tmp_path: Path) -> None:
    _write(tmp_path, "u_x", _SCALAR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    first = eng._nodes["n0"].behaviors["u_x"]
    _reload(eng, "n0", tmp_path, node)  # nothing changed
    assert eng._nodes["n0"].behaviors["u_x"] is first  # same object, not recompiled


def test_cache_recompiles_on_mtime_change(tmp_path: Path) -> None:
    path = _write(tmp_path, "u_x", _SCALAR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    first = eng._nodes["n0"].behaviors["u_x"]
    time.sleep(0.01)
    path.write_text(_SCALAR + "        # changed\n", encoding="utf-8")
    _reload(eng, "n0", tmp_path, node)
    assert eng._nodes["n0"].behaviors["u_x"] is not first  # recompiled


def test_removed_script_drops_binding(tmp_path: Path) -> None:
    path = _write(tmp_path, "u_x", _SCALAR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    assert "u_x" in eng.script_driven_uniforms("n0")
    path.unlink()
    _reload(eng, "n0", tmp_path, node)
    assert "u_x" not in eng.script_driven_uniforms("n0")


# ---- is_scriptable + stub_for ----


def test_is_scriptable_gate() -> None:
    assert is_scriptable(_u("u_x"))
    assert is_scriptable(_u("u_v", dim=3))
    assert not is_scriptable(_u("u_tex", gl_type=0x8B5E))  # GL_SAMPLER_2D
    assert not is_scriptable(object())  # no shape attrs


def test_stub_for_each_kind_compiles_and_runs(tmp_path: Path) -> None:
    # A freshly-generated stub for each kind compiles, instantiates, and returns a coercion-valid
    # value (so a new script runs immediately, not as an error). Drive it through the engine.
    cases = [
        ("u_s", _u("u_s")),
        ("u_v2", _u("u_v2", dim=2)),
        ("u_v3", _u("u_v3", dim=3)),
        ("u_v4", _u("u_v4", dim=4)),
        ("u_arr", _u("u_arr", dim=1, n=4)),
        ("u_txt", _u("u_txt", dim=1, n=8, gl_type=_GL_UNSIGNED_INT)),
    ]
    for name, uniform in cases:
        body = stub_for(uniform)  # type: ignore[arg-type]
        _write(tmp_path, name, body, tag=uniform_tag(uniform))  # type: ignore[arg-type]
        node = _FakeNode([uniform])
        eng = ScriptEngine()
        _reload(eng, name, tmp_path, node)
        assert (name, name) not in eng.errors, f"stub for {name} failed to compile"
        eng.tick(name, node, _ctx(0.0))
        assert (name, name) not in eng.errors, f"stub for {name} errored on tick"
        assert name in node.uniform_values


# ---- integer-uniform coercion (review-swarm finding: float-into-int silently failed) ----


def test_int_scalar_output_rounds_to_int(tmp_path: Path) -> None:
    # A script returning a float for an int/uint uniform must round to int (moderngl rejects a float
    # write). is_scriptable + the coercion now handle GL_INT/GL_UNSIGNED_INT.
    _write(
        tmp_path,
        "u_n",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> int:\n"
        "        return 2.7\n",
        tag="int",
    )
    node = _FakeNode([_u("u_n", gl_type=_GL_INT)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    val = node.uniform_values["u_n"]
    assert val == 3 and isinstance(val, int)
    assert ("n0", "u_n") not in eng.errors


def test_uint_scalar_output_rounds_to_int(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "u_count",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> int:\n"
        "        return 3.9\n",
        tag="int",
    )
    node = _FakeNode([_u("u_count", gl_type=_GL_UNSIGNED_INT)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    val = node.uniform_values["u_count"]
    assert val == 4 and isinstance(val, int)


def test_ivec2_output_rounds_components(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "u_iv",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> Vec2:\n"
        "        return Vec2(1.4, 2.6)\n",
        tag="vec2",
    )
    node = _FakeNode([_u("u_iv", dim=2, gl_type=_GL_INT_VEC2)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_iv"] == (1, 3)  # round(1.4)=1, round(2.6)=3
    assert all(isinstance(v, int) for v in node.uniform_values["u_iv"])


def test_int_array_rounds_and_exact_length(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "u_arr",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> Array:\n"
        "        return Array([1.4, 2.6, 3.5])\n",
        tag="array",
    )
    node = _FakeNode([_u("u_arr", dim=1, n=3, gl_type=_GL_INT)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert tuple(node.uniform_values["u_arr"]) == (1, 3, 4)  # round-half-to-even


def test_stub_for_int_scalar_returns_int(tmp_path: Path) -> None:
    body = stub_for(_u("u_n", gl_type=_GL_INT))  # type: ignore[arg-type]
    assert "-> int" in body and "return 0\n" in body
    _write(tmp_path, "u_n", body, tag="int")
    node = _FakeNode([_u("u_n", gl_type=_GL_INT)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_n"] == 0
    assert ("n0", "u_n") not in eng.errors


# ---- vecN[M] array (review-swarm: zero coverage of the nested-tuple chunking) ----


def test_vec2_array_chunks_into_rows(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "u_pts",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> Array:\n"
        "        return Array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])\n",  # vec2[3]
        tag="array",
    )
    node = _FakeNode([_u("u_pts", dim=2, n=3)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_pts"] == [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]


def test_vec2_array_wrong_flat_length_freezes(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "u_pts",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> Array:\n"
        "        return Array([0.0, 1.0, 2.0])\n",  # 3 for a vec2[3] (needs 6)
        tag="array",
    )
    node = _FakeNode([_u("u_pts", dim=2, n=3)])
    node.uniform_values["u_pts"] = [(0.0, 0.0)] * 3
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_pts"] == [(0.0, 0.0)] * 3  # frozen
    assert eng.errors[("n0", "u_pts")].kind == "runtime"


# ---- namespace: super + containers (review-swarm: the engine's own idiom NameError'd) ----


def test_super_init_works(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "u_x",
        "class Behavior(ScriptBehavior):\n"
        "    def __init__(self) -> None:\n"
        "        super().__init__()\n"
        "        self.buf = []\n"
        "    def update(self, ctx: Ctx) -> float:\n"
        "        self.buf.append(ctx.t)\n"
        "        return sum(self.buf) / len(self.buf)\n",
    )
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(2.0))
    assert ("n0", "u_x") not in eng.errors  # super + list + sum all resolved
    assert node.uniform_values["u_x"] == 2.0


# ---- arity validation (review-swarm: def update(ctx) forgot self) ----


def test_update_missing_self_is_compile_error(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "u_x",
        "class Behavior(ScriptBehavior):\n"
        "    def update(ctx) -> float:\n"  # forgot self
        "        return 0.5\n",
    )
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    err = eng.errors[("n0", "u_x")]
    assert err.kind == "compile" and "self" in err.message


# ---- error-line recovery (review-swarm: runtime errors recorded line=-1) ----


def test_runtime_error_records_user_line(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "u_x",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> float:\n"
        "        x = 1\n"
        "        return 1.0 / 0.0\n",  # line 4
    )
    node = _FakeNode([_u("u_x")])
    node.uniform_values["u_x"] = 0.0
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    err = eng.errors[("n0", "u_x")]
    assert err.kind == "runtime" and err.line == 4


def test_runtime_error_in_helper_records_deepest_user_line(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "u_x",
        "class Behavior(ScriptBehavior):\n"
        "    def _bad(self):\n"
        "        return 1.0 / 0.0\n"  # line 3 — the deepest user frame
        "    def update(self, ctx: Ctx) -> float:\n"
        "        return self._bad()\n",
    )
    node = _FakeNode([_u("u_x")])
    node.uniform_values["u_x"] = 0.0
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert eng.errors[("n0", "u_x")].line == 3


# ---- reload robustness (review-swarm: read_text could crash the frame loop) ----


def test_non_utf8_file_does_not_crash_reload(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    # Tagged name (047) so it matches the glob + is actually read — the read-guard only fires if the
    # file reaches read_text(encoding="utf-8"); an untagged name would skip the read entirely and the
    # assert would pass vacuously.
    (scripts_dir / "u_x__float.py").write_bytes(b"\xff\xfe not utf-8")
    node = _FakeNode([_u("u_x")])
    eng = ScriptEngine()
    eng.reload(
        "n0", scripts_dir, node, _all_active(scripts_dir.parent)
    )  # must NOT raise
    assert "u_x" not in eng.script_driven_uniforms("n0")


# ---- engine-driven reject (review-swarm: a u_time.py would silently no-op) ----


def test_engine_driven_uniform_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "u_time", _SCALAR)
    node = _FakeNode([_u("u_time")])
    eng = ScriptEngine(engine_driven=frozenset({"u_time"}))
    _reload(eng, "n0", tmp_path, node)
    assert "u_time" not in eng.script_driven_uniforms("n0")


# ---- inactive-uniform reclaim (review-swarm: stale binding when uniform removed) ----


def test_uniform_gone_inactive_drops_binding(tmp_path: Path) -> None:
    # u_x removed from the shader while u_keep remains active (a non-empty active set, so this is a
    # genuine removal, NOT a transient program-invalidation where active is briefly empty).
    _write(tmp_path, "u_x", _SCALAR)
    node = _FakeNode([_u("u_x"), _u("u_keep")])
    eng = _engine(tmp_path, node)
    assert "u_x" in eng.script_driven_uniforms("n0")
    node._uniforms = [_u("u_keep")]  # u_x removed; u_keep still active
    _reload(eng, "n0", tmp_path, node)
    assert "u_x" not in eng.script_driven_uniforms("n0")


def test_empty_active_set_keeps_binding_no_false_orphan(tmp_path: Path) -> None:
    # An empty active set = the program is mid-invalidation (a lib edit dropped it); a live binding
    # must NOT be flagged a false orphan and dropped — it returns when the program recompiles.
    _write(tmp_path, "u_x", _SCALAR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    assert "u_x" in eng.script_driven_uniforms("n0")
    node._uniforms = []  # program invalidated this frame
    _reload(eng, "n0", tmp_path, node)
    assert "u_x" in eng.script_driven_uniforms("n0")  # kept, not falsely orphaned


# ---- drop_node (review-swarm: defined but never called → leak) ----


def test_drop_node_clears_state_and_errors(tmp_path: Path) -> None:
    _write(
        tmp_path, "u_x", "class Behavior(ScriptBehavior):\n    pass\n"
    )  # compile err
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    assert ("n0", "u_x") in eng.errors
    eng.drop_node("n0")
    assert ("n0", "u_x") not in eng.errors
    assert eng.script_driven_uniforms("n0") == set()


# ---- export-error isolation (review-swarm: export ticked the shared errors dict) ----


def test_export_tick_does_not_touch_live_errors(tmp_path: Path) -> None:
    # A live binding has a recorded shape error; ticking a FRESH export set (a clean script) must
    # NOT clear the live error — the export's errors sink is a throwaway.
    _write(
        tmp_path,
        "u_x",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> Vec3:\n"
        "        return Vec3(1.0, 2.0, 3.0)\n",  # vec3 into a scalar -> live error
    )
    node = _FakeNode([_u("u_x")])
    node.uniform_values["u_x"] = 0.0
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert ("n0", "u_x") in eng.errors  # live error recorded
    fresh = eng.fresh_behaviors_for("n0")
    export_node = _FakeNode([_u("u_x")])
    export_node.uniform_values["u_x"] = 0.0
    eng.tick_behaviors("n0", export_node, _ctx(0.0), fresh)
    assert ("n0", "u_x") in eng.errors  # live error UNTOUCHED by the export tick


# ---- ctx is frozen (review-swarm: docstring said read-only, wasn't enforced) ----


def test_ctx_is_immutable() -> None:
    ctx = EngineContext(t=1.0, dt=0.1, frame=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.t = 2.0  # type: ignore[misc]


# ---- scriptless-node tick is a no-op (review-swarm S1: don't build the active dict) ----


def test_scriptless_node_tick_skips_active_build(tmp_path: Path) -> None:
    # A node with no scripts must NOT call get_active_uniforms() each tick (the empty-behaviors
    # early-out). Count the calls via a spy.
    calls = {"n": 0}

    class _SpyNode(_FakeNode):
        def get_active_uniforms(self) -> list[types.SimpleNamespace]:
            calls["n"] += 1
            return self._uniforms

    (tmp_path / "scripts").mkdir(exist_ok=True)  # empty scripts dir
    node = _SpyNode([_u("u_x"), _u("u_y")])
    eng = _engine(tmp_path, node)  # reload() reads active uniforms once (unavoidable)
    calls["n"] = 0  # count only the tick path
    for i in range(10):
        eng.tick("n0", node, _ctx(i / 60))
    assert (
        calls["n"] == 0
    )  # no per-tick active-uniform dict build for a scriptless node


# ---- perf sanity (not a gate) ----


def test_tick_is_cheap(tmp_path: Path) -> None:
    _write(tmp_path, "u_x", _SCALAR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    start = time.perf_counter()
    for i in range(1000):
        eng.tick("n0", node, _ctx(i / 60))
    per_tick_us = (time.perf_counter() - start) / 1000 * 1e6
    assert per_tick_us < 500  # generous ceiling


# ============================================================================
# Node-brain script (feature 044): one stateful class drives MANY uniforms via a dict return.
# Per-uniform is the 1-entry case of the same value-map pipeline. These cover the brain-only
# decisions (per-key vs behavior-level freeze, the unknown-key skip, the conflict apply-order,
# the dynamic driven set) + the cardinality-agnostic invariants parametrized over both paths.
# ============================================================================

# A brain integrator that drives TWO uniforms from ONE accumulator (the headline goal — the
# physics-copy-paste is gone). Mirrors _INTEGRATOR's accumulation on a per-uniform binding.
_BRAIN_INTEGRATOR = (
    "class Behavior(ScriptBehavior):\n"
    "    def __init__(self) -> None:\n"
    "        self.v = 0.0\n"
    "    def update(self, ctx: Ctx) -> dict:\n"
    "        self.v += ctx.dt\n"
    "        return {'u_x': self.v, 'u_y': self.v * 2.0}\n"
)


def test_brain_drives_multiple_uniforms_one_tick(tmp_path: Path) -> None:
    _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_a': 0.25, 'u_b': 0.75}\n",
    )
    node = _FakeNode([_u("u_a"), _u("u_b")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_a"] == 0.25
    assert node.uniform_values["u_b"] == 0.75


def test_brain_per_key_shape_mismatch_freezes_only_that_key(tmp_path: Path) -> None:
    # The is-it-native falsifier: a per-KEY coercion mismatch freezes ONLY that key; siblings still
    # write (matches the per-uniform path's granular freeze, not a behavior-level all-or-nothing).
    _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_a': 0.4, 'u_b': Vec3(1.0, 2.0, 3.0)}\n",  # vec3 into a scalar
    )
    node = _FakeNode([_u("u_a"), _u("u_b")])
    node.uniform_values["u_b"] = 0.0
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_a"] == 0.4  # sibling wrote
    assert node.uniform_values["u_b"] == 0.0  # frozen
    assert (
        eng.errors[("n0", "u_b")].kind == "runtime"
    )  # per-KEY error, not the sentinel
    assert ("n0", "script.py") not in eng.errors


def test_brain_non_dict_return_is_clean_error_under_sentinel(tmp_path: Path) -> None:
    # A brain that returns a non-dict is a behavior-level failure under the sentinel key — not a crash.
    _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return 0.5\n",  # a bare float, not a dict
    )
    node = _FakeNode([_u("u_a")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))  # must NOT raise
    err = eng.errors[("n0", "script.py")]
    assert err.kind == "runtime"
    assert ("n0", "u_a") not in eng.errors


def test_brain_raw_exception_freezes_all_and_records_user_line(tmp_path: Path) -> None:
    # The decision-11 falsifier: a raw update() exception freezes EVERY name the brain drove last
    # frame (one object = one coherent state) AND records the error under the sentinel with the
    # CORRECT user line (fails if _user_error_line still hardcodes <u:name> for the brain).
    path = _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_x': 0.3, 'u_y': 0.6}\n",
    )
    node = _FakeNode([_u("u_x"), _u("u_y")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))  # establish last-good for both
    assert node.uniform_values["u_x"] == 0.3 and node.uniform_values["u_y"] == 0.6

    time.sleep(0.01)
    path.write_text(
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        x = 1\n"
        "        raise ValueError('boom')\n",  # line 4
        encoding="utf-8",
    )
    _reload(eng, "n0", tmp_path, node)
    eng.tick("n0", node, _ctx(0.1))
    assert node.uniform_values["u_x"] == 0.3  # both frozen together
    assert node.uniform_values["u_y"] == 0.6
    err = eng.errors[("n0", "script.py")]
    assert err.kind == "runtime" and "ValueError" in err.message
    assert err.line == 4  # the real user line, NOT -1


def test_brain_unknown_key_warns_skips_and_records_soft_error(
    tmp_path: Path,
) -> None:
    # A key naming no active scriptable uniform is warn-once + SKIPPED (never a None write —
    # decision 5) AND records a SOFT ScriptError under (node, name) so the UI surfaces the typo
    # (NOT loguru-only). The bad key must NOT claim ownership in script_driven_uniforms.
    _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_a': 0.5, 'u_ghost': 0.9}\n",  # u_ghost not on the node
    )
    node = _FakeNode([_u("u_a")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_a"] == 0.5
    assert "u_ghost" not in node.uniform_values  # NOT written as None
    err = eng.errors[("n0", "u_ghost")]
    assert err.kind == "runtime" and "orphan" in err.message  # soft error surfaced
    assert "u_ghost" not in eng.script_driven_uniforms(
        "n0"
    )  # bad key claims no ownership
    assert "u_ghost" in eng._nodes["n0"].warned
    eng.tick("n0", node, _ctx(0.1))  # second tick: no re-warn (warn-once)
    assert eng._nodes["n0"].warned == {"u_ghost"}


def test_brain_unknown_key_error_clears_when_key_fixed(tmp_path: Path) -> None:
    # The typo'd-key soft error is a zombie-free: once the bad key stops being returned (the user
    # fixes the typo), its (node, name) error is cleared on the next tick (decision 8, generalized).
    path = _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_a': 0.5, 'u_ghost': 0.9}\n",
    )
    node = _FakeNode([_u("u_a")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert ("n0", "u_ghost") in eng.errors
    time.sleep(0.01)
    path.write_text(  # typo fixed: u_ghost gone
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_a': 0.5}\n",
        encoding="utf-8",
    )
    _reload(eng, "n0", tmp_path, node)
    eng.tick("n0", node, _ctx(0.1))
    assert ("n0", "u_ghost") not in eng.errors  # zombie cleared


def test_brain_engine_owned_key_dropped_silently(tmp_path: Path) -> None:
    # Feature 047 decision 5: a brain naming an engine-owned uniform (u_time) SILENTLY drops the key
    # — no false ownership AND no soft error (the renderer owns that slot; a brain can't be expected
    # to avoid naming it). The renderer's u_time value is untouched.
    _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_a': 0.5, 'u_time': 9.0}\n",
    )
    node = _FakeNode([_u("u_a"), _u("u_time")])
    node.uniform_values["u_time"] = 1.23  # the renderer's value
    eng = ScriptEngine(engine_driven=frozenset({"u_time"}))
    _reload(eng, "n0", tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_a"] == 0.5
    assert "u_time" not in eng.script_driven_uniforms("n0")  # no false ownership
    assert ("n0", "u_time") not in eng.errors  # silently dropped, no soft error
    assert node.uniform_values["u_time"] == 1.23  # renderer value untouched


def test_brain_nan_freezes_and_records(tmp_path: Path) -> None:
    # The NaN/Inf fix: a non-finite value is no longer written silently (a black-frame footgun that
    # also poisons last-good) — it freezes the uniform at last-good + records a runtime ScriptError,
    # folding into the normal frozen-uniform path.
    path = _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_a': 0.3}\n",
    )
    node = _FakeNode([_u("u_a")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_a"] == 0.3  # good last value
    time.sleep(0.01)
    path.write_text(
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_a': float('inf')}\n",  # non-finite
        encoding="utf-8",
    )
    _reload(eng, "n0", tmp_path, node)
    eng.tick("n0", node, _ctx(0.1))
    assert node.uniform_values["u_a"] == 0.3  # frozen at last-good, NOT inf
    err = eng.errors[("n0", "u_a")]
    assert err.kind == "runtime" and "finite" in err.message


def test_per_uniform_nan_freezes(tmp_path: Path) -> None:
    # The NaN guard is in the shared coerce_one, so it covers the per-uniform path too.
    _write(
        tmp_path,
        "u_x",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> float:\n"
        "        return float('nan')\n",
    )
    node = _FakeNode([_u("u_x")])
    node.uniform_values["u_x"] = 0.0
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_x"] == 0.0  # frozen
    assert eng.errors[("n0", "u_x")].kind == "runtime"


def test_import_math_works(tmp_path: Path) -> None:
    # A script is plain Python (feature 045): `import math` succeeds and `math.*` drives the value —
    # no curated vocab, no _no_import shim.
    _write(
        tmp_path,
        "u_x",
        "import math\n"
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> float:\n"
        "        return math.cos(0.0)\n",  # 1.0
    )
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert ("n0", "u_x") not in eng.errors
    assert abs(node.uniform_values["u_x"] - 1.0) < 1e-9


def test_chr_ord_available_for_codepoint_text(tmp_path: Path) -> None:
    # chr/ord are real builtins so text can be built by codepoint.
    _write(
        tmp_path,
        "u_t",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> Text:\n"
        "        return Text(chr(65) + chr(66))\n",  # 'AB'
        tag="text",
    )
    node = _FakeNode([_u("u_t", dim=1, n=8, gl_type=_GL_UNSIGNED_INT)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    val = node.uniform_values["u_t"]
    assert val[0] == ord("A") and val[1] == ord("B")
    assert ("n0", "u_t") not in eng.errors


def test_array_nested_tuples_gives_flatten_hint(tmp_path: Path) -> None:
    # The vecN[M] footgun (a list of tuples) raises a clear flatten hint, not the cryptic
    # "float() argument must be ... not 'tuple'". A raise in update() is caught as a runtime error.
    _write(
        tmp_path,
        "u_pts",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> Array:\n"
        "        return Array([(0.0, 1.0), (2.0, 3.0)])\n",  # nested, wrong
        tag="array",
    )
    node = _FakeNode([_u("u_pts", dim=2, n=2)])
    node.uniform_values["u_pts"] = [(0.0, 0.0), (0.0, 0.0)]
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    err = eng.errors[("n0", "u_pts")]
    assert err.kind == "runtime"
    assert "flattened" in err.message and "float()" not in err.message


def test_conflict_per_uniform_overrides_brain(tmp_path: Path) -> None:
    # The conflict rule (decision 7): when u_x.py AND script.py both target u_x, BOTH run but the
    # per-uniform value wins (two-pass write order: brain first, u_x.py last). Written so a naive
    # insertion-order apply would invert (the brain file is created AFTER u_x.py here).
    _write(tmp_path, "u_x", _SCALAR)  # u_x.py: returns 0.5 at t=0
    _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_x': 9.9, 'u_y': 0.2}\n",
    )
    node = _FakeNode([_u("u_x"), _u("u_y")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert (
        abs(node.uniform_values["u_x"] - 0.5) < 1e-9
    )  # u_x.py wins, NOT the brain's 9.9
    assert node.uniform_values["u_y"] == 0.2  # brain owns its own slot


def test_conflict_broken_per_uniform_yields_to_brain(tmp_path: Path) -> None:
    # The conflict-freeze-fallback decision: when a u_x.py that conflicts with a brain on u_x is
    # BROKEN, the slot shows the brain's live value (a broken override lets the base behavior show
    # through), NOT a freeze that clobbers it — and the per-uniform error is still recorded.
    _write(  # u_x.py raises every tick
        tmp_path,
        "u_x",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> float:\n"
        "        raise ValueError('boom')\n",
    )
    _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def __init__(self) -> None:\n"
        "        self.v = 0.0\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        self.v += 0.1\n"
        "        return {'u_x': self.v}\n",
    )
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert (
        abs(node.uniform_values["u_x"] - 0.1) < 1e-9
    )  # brain's live value, not frozen
    eng.tick("n0", node, _ctx(0.1))
    assert (
        abs(node.uniform_values["u_x"] - 0.2) < 1e-9
    )  # brain still advances, not stuck
    assert eng.errors[("n0", "u_x")].kind == "runtime"  # the u_x.py error IS surfaced


def test_conflict_broken_per_uniform_yields_even_after_success(tmp_path: Path) -> None:
    # The determinism point: the outcome must NOT depend on whether u_x.py ever succeeded. A per-
    # uniform that worked, then breaks, still yields the slot to the brain (no stale-last-good freeze).
    path = _write(  # u_x.py works at first
        tmp_path,
        "u_x",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> float:\n"
        "        return 0.77\n",
    )
    _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def __init__(self) -> None:\n"
        "        self.v = 0.0\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        self.v += 0.5\n"
        "        return {'u_x': self.v}\n",
    )
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_x"] == 0.77  # per-uniform wins while healthy
    time.sleep(0.01)
    path.write_text(  # now u_x.py breaks
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> float:\n"
        "        raise ValueError('boom')\n",
        encoding="utf-8",
    )
    _reload(eng, "n0", tmp_path, node)
    eng.tick("n0", node, _ctx(0.1))
    assert (
        abs(node.uniform_values["u_x"] - 1.0) < 1e-9
    )  # brain shows through, NOT frozen at 0.77


def test_brain_script_driven_uniforms_reports_driven_names(tmp_path: Path) -> None:
    # script_driven_uniforms returns the brain's driven uniform names (NOT "script.py"); partial
    # (empty) before the first tick since the driven set is only known after a tick (decision 10).
    _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_a': 0.1, 'u_b': 0.2}\n",
    )
    node = _FakeNode([_u("u_a"), _u("u_b")])
    eng = _engine(tmp_path, node)
    assert eng.script_driven_uniforms("n0") == set()  # cold start: no keys yet
    eng.tick("n0", node, _ctx(0.0))
    assert eng.script_driven_uniforms("n0") == {"u_a", "u_b"}
    assert "script.py" not in eng.script_driven_uniforms("n0")  # never the sentinel


def test_brain_partial_dict_leaves_others_default(tmp_path: Path) -> None:
    # The dict is NON-exhaustive: a brain drives 2 of 3, the 3rd keeps its seeded value.
    _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_a': 0.1, 'u_b': 0.2}\n",
    )
    node = _FakeNode([_u("u_a"), _u("u_b"), _u("u_c")])
    node.uniform_values["u_c"] = 0.77  # its slider/default
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_a"] == 0.1
    assert node.uniform_values["u_b"] == 0.2
    assert node.uniform_values["u_c"] == 0.77  # untouched


def test_brain_omitted_key_keeps_last_value_and_clears_error(tmp_path: Path) -> None:
    # A key driven on frame N then omitted on frame N+1 keeps its last WRITTEN value (NOT frozen-as-
    # error) AND any stale error for it is cleared — no zombie error (decision 8).
    path = _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_a': 0.4, 'u_b': Vec3(1.0, 2.0, 3.0)}\n",  # u_b errors (vec3->scalar)
    )
    node = _FakeNode([_u("u_a"), _u("u_b")])
    node.uniform_values["u_b"] = 0.0
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert eng.errors[("n0", "u_b")].kind == "runtime"  # u_b errored

    time.sleep(0.01)
    path.write_text(  # now the brain only drives u_a (u_b omitted)
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_a': 0.9}\n",
        encoding="utf-8",
    )
    _reload(eng, "n0", tmp_path, node)
    eng.tick("n0", node, _ctx(0.1))
    assert node.uniform_values["u_a"] == 0.9
    assert node.uniform_values["u_b"] == 0.0  # last value kept (not corrupted)
    assert ("n0", "u_b") not in eng.errors  # zombie error cleared
    assert eng.script_driven_uniforms("n0") == {"u_a"}  # u_b no longer driven


def test_brain_compile_error_under_sentinel(tmp_path: Path) -> None:
    # A raising __init__ declares ZERO keys (it never ticks) yet must surface a compile error under
    # the sentinel — the gap the per-uniform compile tests don't transfer (no uniform name to key on).
    _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def __init__(self) -> None:\n"
        "        raise ValueError('boom')\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {}\n",
    )
    node = _FakeNode([_u("u_a")])
    eng = _engine(tmp_path, node)
    assert eng.errors[("n0", "script.py")].kind == "compile"


def test_brain_removed_clears_per_key_errors(tmp_path: Path) -> None:
    # On brain removal the drop loop must clear the per-KEY errors of its driven uniforms (a
    # coercion-failed key records under (node_id, name), not the sentinel) — no zombie error.
    path = _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_a': 0.4, 'u_b': Vec3(1.0, 2.0, 3.0)}\n",  # u_b errors
    )
    node = _FakeNode([_u("u_a"), _u("u_b")])
    node.uniform_values["u_b"] = 0.0
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert eng.errors[("n0", "u_b")].kind == "runtime"  # per-key error recorded
    path.unlink()
    _reload(eng, "n0", tmp_path, node)
    assert ("n0", "u_b") not in eng.errors  # cleared on removal, not a zombie
    assert ("n0", "script.py") not in eng.errors


def test_brain_export_tick_does_not_warn_into_live(tmp_path: Path) -> None:
    # The export tick path's warn-once set is a throwaway — an unknown brain key seen ONLY during an
    # export must NOT leak into the live `warned` set (the structural-isolation contract, decision 11).
    _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_a': 0.5, 'u_ghost': 0.9}\n",  # u_ghost not on the node
    )
    node = _FakeNode([_u("u_a")])
    eng = _engine(tmp_path, node)
    fresh = eng.fresh_behaviors_for("n0")
    eng.tick_behaviors("n0", node, _ctx(0.0), fresh)
    assert (
        eng._nodes["n0"].warned == set()
    )  # live warn set UNTOUCHED by the export tick


def test_brain_removed_clears_driven_set(tmp_path: Path) -> None:
    path = _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_a': 0.1}\n",
    )
    node = _FakeNode([_u("u_a")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert eng.script_driven_uniforms("n0") == {"u_a"}
    path.unlink()
    _reload(eng, "n0", tmp_path, node)
    assert eng.script_driven_uniforms("n0") == set()


# ---- cardinality-agnostic invariants: SAME behavior, both paths ----

_PARAM_PER_UNIFORM = ("per_uniform", _INTEGRATOR, "u_x")
_PARAM_NODE_BRAIN = ("node_brain", _BRAIN_INTEGRATOR, "u_x")


def _setup_path(tmp: Path, kind: str, body: str) -> "tuple[Path, _FakeNode]":
    if kind == "per_uniform":
        path = _write(tmp, "u_x", body)
        node = _FakeNode([_u("u_x")])
    else:
        path = _write_brain(tmp, body)
        node = _FakeNode([_u("u_x"), _u("u_y")])
    return path, node


@pytest.mark.parametrize(
    ("kind", "body", "read"),
    [_PARAM_PER_UNIFORM, _PARAM_NODE_BRAIN],
    ids=["per_uniform", "node_brain"],
)
def test_state_accumulates_both_paths(
    tmp_path: Path, kind: str, body: str, read: str
) -> None:
    _, node = _setup_path(tmp_path, kind, body)
    eng = _engine(tmp_path, node)
    for i in range(5):
        eng.tick("n0", node, _ctx(0.0, dt=1.0, frame=i))
    assert node.uniform_values[read] == 5.0  # 5 ticks of dt=1.0 on the ONE accumulator


@pytest.mark.parametrize(
    ("kind", "body", "read"),
    [_PARAM_PER_UNIFORM, _PARAM_NODE_BRAIN],
    ids=["per_uniform", "node_brain"],
)
def test_reset_clears_state_both_paths(
    tmp_path: Path, kind: str, body: str, read: str
) -> None:
    _, node = _setup_path(tmp_path, kind, body)
    eng = _engine(tmp_path, node)
    for i in range(4):
        eng.tick("n0", node, _ctx(0.0, dt=1.0, frame=i))
    assert node.uniform_values[read] == 4.0
    key = "u_x" if kind == "per_uniform" else "script.py"
    eng.reset("n0", key)
    eng.tick("n0", node, _ctx(0.0, dt=1.0, frame=0))
    assert node.uniform_values[read] == 1.0  # fresh instance, +1 dt


@pytest.mark.parametrize(
    ("kind", "body", "read"),
    [_PARAM_PER_UNIFORM, _PARAM_NODE_BRAIN],
    ids=["per_uniform", "node_brain"],
)
def test_recompile_on_edit_resets_both_paths(
    tmp_path: Path, kind: str, body: str, read: str
) -> None:
    path, node = _setup_path(tmp_path, kind, body)
    eng = _engine(tmp_path, node)
    for i in range(3):
        eng.tick("n0", node, _ctx(0.0, dt=1.0, frame=i))
    assert node.uniform_values[read] == 3.0
    time.sleep(0.01)
    path.write_text(body + "        # edited\n", encoding="utf-8")
    _reload(eng, "n0", tmp_path, node)
    eng.tick("n0", node, _ctx(0.0, dt=1.0, frame=0))
    assert node.uniform_values[read] == 1.0  # fresh instance: +1 dt


@pytest.mark.parametrize(
    ("kind", "body", "read"),
    [_PARAM_PER_UNIFORM, _PARAM_NODE_BRAIN],
    ids=["per_uniform", "node_brain"],
)
def test_export_isolation_interleave_both_paths(
    tmp_path: Path, kind: str, body: str, read: str
) -> None:
    # The export-isolation falsifier: warm the LIVE instance, then a FRESH export set ticks cold.
    # Assert the EXTRACTED per-key value is cold (a dict is truthy and would mask a coercion fail —
    # we read node.uniform_values[read], not the raw run() return).
    _, node = _setup_path(tmp_path, kind, body)
    eng = _engine(tmp_path, node)
    for i in range(10):
        eng.tick("n0", node, _ctx(0.0, dt=1.0, frame=i))
    live = node.uniform_values[read]
    assert live == 10.0

    fresh = eng.fresh_behaviors_for("n0")
    export_node = _FakeNode([_u("u_x"), _u("u_y")])
    eng.tick_behaviors("n0", export_node, _ctx(0.0, dt=1.0, frame=0), fresh)
    assert export_node.uniform_values[read] == 1.0  # cold start, NOT the live 10.0
    assert export_node.uniform_values[read] != live


# ---- feature 042: brain_stub_for / _stub_kind / ctx.mouse ----


def test_brain_stub_for_compiles_and_runs(tmp_path: Path) -> None:
    # The load-bearing trap (044 OOS / 041 decision 12): the brain stub must annotate bare `-> dict`,
    # NEVER `dict[str, Any]` (Any is absent from the exec globals -> permanent compile-freeze). The
    # only honest check is to compile + tick the emitted text in the real engine and assert no error.

    uniforms = [_u("u_a"), _u("u_b", dim=3), _u("u_n", n=4)]
    body = brain_stub_for(uniforms)
    _write_brain(tmp_path, body)
    node = _FakeNode(uniforms)
    eng = _engine(tmp_path, node)
    assert eng.errors == {}  # the brain compiled clean (no sentinel error)
    eng.tick("n0", node, _ctx(0.0))
    assert eng.errors == {}  # and ran clean (no per-key coercion error)
    # Every seeded uniform got a coercion-valid value (the stub defaults coerce).
    assert "u_a" in node.uniform_values
    assert len(node.uniform_values["u_b"]) == 3


def test_brain_stub_for_empty_when_no_scriptable(tmp_path: Path) -> None:
    # A node whose only uniform is a sampler (not scriptable) still emits a valid empty-dict brain.

    _GL_SAMPLER_2D = 0x8B5E
    sampler = _u("u_tex", gl_type=_GL_SAMPLER_2D)
    body = brain_stub_for([sampler])
    _write_brain(tmp_path, body)
    node = _FakeNode([sampler])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert eng.errors == {}  # an empty-{} brain compiles + runs, drives nothing


def test_brain_stub_for_skips_non_scriptable(tmp_path: Path) -> None:
    # brain_stub_for pre-lists ONLY the scriptable uniforms — a sampler is omitted from the dict.

    _GL_SAMPLER_2D = 0x8B5E
    body = brain_stub_for([_u("u_ok"), _u("u_tex", gl_type=_GL_SAMPLER_2D)])
    assert "u_ok" in body
    assert "u_tex" not in body


def test_ctx_mouse_default_is_export_center() -> None:
    # EngineContext.mouse defaults to the fixed export value (center) so the bare-clock construct
    # sites compile AND an export with no live mouse is deterministic.

    assert MouseState(0.5, 0.5) == EXPORT_MOUSE
    assert _ctx(0.0).mouse == EXPORT_MOUSE


def test_ctx_mouse_reaches_script(tmp_path: Path) -> None:
    # A script reads ctx.mouse.x/.y; the live tick threads the passed-in mouse, an export tick
    # (no mouse) freezes at center.

    body = (
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> Vec2:\n"
        "        return Vec2(ctx.mouse.x, ctx.mouse.y)\n"
    )
    _write(tmp_path, "u_p", body, tag="vec2")
    node = _FakeNode([_u("u_p", dim=2)])
    eng = _engine(tmp_path, node)

    eng.tick(
        "n0", node, EngineContext(t=0.0, dt=1 / 60, frame=0, mouse=MouseState(0.2, 0.7))
    )
    assert tuple(node.uniform_values["u_p"]) == pytest.approx((0.2, 0.7))

    # An export tick (fresh instances, default-mouse ctx) freezes at center.
    fresh = eng.fresh_behaviors_for("n0")
    export_node = _FakeNode([_u("u_p", dim=2)])
    eng.tick_behaviors(
        "n0", export_node, EngineContext(t=0.0, dt=1 / 60, frame=0), fresh
    )
    assert tuple(export_node.uniform_values["u_p"]) == pytest.approx((0.5, 0.5))


def test_brain_status_reports_soft_errors(tmp_path: Path) -> None:
    # brain_status surfaces the sentinel-clear case: a driven count + a homeless soft-key error for a
    # typo'd key (names no real uniform). The real-uniform key it drives is NOT in soft_errors.
    body = (
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_real': 1.0, 'u_typo': 2.0}\n"
    )
    _write_brain(tmp_path, body)
    node = _FakeNode([_u("u_real")])  # u_typo names no active uniform
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    status = eng.brain_status("n0")
    assert status is not None
    assert status.sentinel_error is None
    assert status.driven_count == 1  # u_real
    soft_keys = [k for k, _ in status.soft_errors]
    assert "u_typo" in soft_keys
    assert "u_real" not in soft_keys


def test_brain_status_none_without_brain(tmp_path: Path) -> None:
    _write(tmp_path, "u_x", _SCALAR.replace("u_wave", "u_x"))
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert eng.brain_status("n0") is None


# ---- inactive (not in active_scripts) script state (feature 047) ----


def test_inactive_script_skips_per_uniform_binding(tmp_path: Path) -> None:
    # A file absent from active_scripts means inactive: never bound, so its uniform stays under
    # manual control (the dict value the test sets is untouched).
    _write(tmp_path, "u_x", _SCALAR.replace("u_wave", "u_x"))
    node = _FakeNode([_u("u_x")])
    node.uniform_values["u_x"] = 0.123  # manual value
    eng = ScriptEngine()
    eng.reload("n0", tmp_path / "scripts", node, set())  # nothing active
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_x"] == 0.123  # not driven
    assert eng.script_driven_uniforms("n0") == set()


def test_activating_an_inactive_binding_rebinds_on_reload(tmp_path: Path) -> None:
    # Adding the file to active_scripts re-binds on the next reload — the script drives again.
    _write(tmp_path, "u_x", _SCALAR.replace("u_wave", "u_x"))
    node = _FakeNode([_u("u_x")])
    eng = ScriptEngine()
    eng.reload("n0", tmp_path / "scripts", node, set())
    eng.tick("n0", node, _ctx(0.0))
    assert eng.script_driven_uniforms("n0") == set()  # inactive
    eng.reload("n0", tmp_path / "scripts", node, {"u_x__float.py"})
    eng.tick("n0", node, _ctx(0.0))
    assert abs(node.uniform_values["u_x"] - 0.5) < 1e-9  # driven again


def test_deactivating_a_live_binding_drops_it(tmp_path: Path) -> None:
    # A binding removed from active_scripts between reloads is dropped — it stops ticking and its
    # uniform returns to manual, like a vanished file.
    _write(tmp_path, "u_x", _SCALAR.replace("u_wave", "u_x"))
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)  # active by default
    eng.tick("n0", node, _ctx(0.0))
    assert "u_x" in eng.script_driven_uniforms("n0")
    eng.reload("n0", tmp_path / "scripts", node, set())  # deactivated
    assert eng.script_driven_uniforms("n0") == set()


def test_inactive_node_brain_skips_binding(tmp_path: Path) -> None:
    # The node-brain honors active_scripts too: script.py absent from the set means no brain binding.
    _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_x': 0.7}\n",
    )
    node = _FakeNode([_u("u_x")])
    node.uniform_values["u_x"] = 0.2
    eng = ScriptEngine()
    eng.reload("n0", tmp_path / "scripts", node, set())  # brain inactive
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_x"] == 0.2  # brain inactive
    assert eng.brain_status("n0") is None


def test_inactive_binding_excluded_from_export_set(tmp_path: Path) -> None:
    # Export isolation honors inactivity by construction: a never-bound script is not in the cached
    # source dict, so fresh_behaviors_for can't recompile it.
    _write(tmp_path, "u_x", _SCALAR.replace("u_wave", "u_x"))
    node = _FakeNode([_u("u_x")])
    eng = ScriptEngine()
    eng.reload("n0", tmp_path / "scripts", node, set())
    assert eng.fresh_behaviors_for("n0") == {}


def test_fresh_stub_compiles_and_runs_under_stripped_globals(tmp_path: Path) -> None:
    # A freshly-generated stub (045 — `import math`, no curated vocab) must compile + run with no
    # error, returning its coercion-valid default.
    _write(tmp_path, "u_x", stub_for(_u("u_x")))
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert ("n0", "u_x") not in eng.errors
    assert node.uniform_values["u_x"] == 0.0


# ---- (name, type) filename key — the 047 retype/rename invariants (F14 core) ----


def test_retype_drops_old_tag_binding_without_corruption(tmp_path: Path) -> None:
    # The headline F14 invariant: a script written for vec2 u_x, when the live uniform becomes vec3,
    # must DROP the binding (tag mismatch) — NO coercion error, the value left under manual control,
    # the vec2 file untouched on disk for a retype-back. (Neuter the tag check in engine._binding_reject
    # and this goes red — it's the only guard against the pre-047 retype-corruption class.)
    _write(
        tmp_path,
        "u_x",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> Vec2:\n"
        "        return Vec2(1.0, 2.0)\n",
        tag="vec2",
    )
    vec2_node = _FakeNode([_u("u_x", dim=2)])
    eng = _engine(tmp_path, vec2_node)
    eng.tick("n0", vec2_node, _ctx(0.0))
    assert vec2_node.uniform_values["u_x"] == (1.0, 2.0)  # bound

    # Retype to vec3: same name, different tag -> looks for u_x__vec3.py (absent) -> drop.
    vec3_node = _FakeNode([_u("u_x", dim=3)])
    vec3_node.uniform_values["u_x"] = (0.0, 0.0, 0.0)  # manual value
    eng.reload("n0", tmp_path / "scripts", vec3_node, {"u_x__vec2.py"})
    eng.tick("n0", vec3_node, _ctx(0.0))
    assert "u_x" not in eng.script_driven_uniforms("n0")  # binding dropped
    assert ("n0", "u_x") not in eng.errors  # NO coercion error
    assert vec3_node.uniform_values["u_x"] == (0.0, 0.0, 0.0)  # manual value untouched
    assert (tmp_path / "scripts" / "u_x__vec2.py").is_file()  # old-tag file safe


def test_retype_back_rebinds_original_script(tmp_path: Path) -> None:
    # Retyping back to vec2 finds u_x__vec2.py still on disk and rebinds it (lossless + reversible).
    _write(
        tmp_path,
        "u_x",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> Vec2:\n"
        "        return Vec2(1.0, 2.0)\n",
        tag="vec2",
    )
    eng = ScriptEngine()
    # vec3 (no binding) -> vec2 (rebinds).
    eng.reload(
        "n0", tmp_path / "scripts", _FakeNode([_u("u_x", dim=3)]), {"u_x__vec2.py"}
    )
    assert "u_x" not in eng.script_driven_uniforms("n0")
    vec2_node = _FakeNode([_u("u_x", dim=2)])
    eng.reload("n0", tmp_path / "scripts", vec2_node, {"u_x__vec2.py"})
    eng.tick("n0", vec2_node, _ctx(0.0))
    assert vec2_node.uniform_values["u_x"] == (1.0, 2.0)  # rebound, code intact


def test_delete_readd_rebinds(tmp_path: Path) -> None:
    # Deleting u_x (script strands) then re-adding it rebinds the original script.
    _write(tmp_path, "u_x", _SCALAR.replace("u_wave", "u_x"))
    eng = ScriptEngine()
    # u_x absent (only u_keep live): the script is an orphan, dropped.
    eng.reload("n0", tmp_path / "scripts", _FakeNode([_u("u_keep")]), {"u_x__float.py"})
    assert "u_x" not in eng.script_driven_uniforms("n0")
    # Re-add u_x: the stranded u_x__float.py rebinds.
    readd = _FakeNode([_u("u_x"), _u("u_keep")])
    eng.reload("n0", tmp_path / "scripts", readd, {"u_x__float.py"})
    eng.tick("n0", readd, _ctx(0.0))
    assert "u_x" in eng.script_driven_uniforms("n0")


def test_parse_script_filename_round_trip_and_edges() -> None:
    # parse_script_filename is the discovery matcher's name/tag splitter (047); per_uniform_filename
    # is its inverse. The LAST `__` separates name from tag, so a name with its own underscores
    # round-trips. Boundary/malformed names return None (not a crash, not a mis-split).
    assert parse_script_filename("u_x__vec2.py") == ("u_x", "vec2")
    # A name containing `__` must round-trip (rfind, not find).
    u = types.SimpleNamespace(
        name="u_a__b", dimension=2, array_length=1, gl_type=_GL_FLOAT
    )
    fn = per_uniform_filename(u)
    assert fn == "u_a__b__vec2.py"
    assert parse_script_filename(fn) == ("u_a__b", "vec2")
    # Malformed / boundary names -> None.
    assert parse_script_filename("script.py") is None  # no `__`
    assert parse_script_filename("u_x.py") is None  # untagged (pre-047)
    assert parse_script_filename("__vec2.py") is None  # empty name
    assert parse_script_filename("u_x__.py") is None  # empty tag
    assert parse_script_filename("u_x__vec2.txt") is None  # not .py
