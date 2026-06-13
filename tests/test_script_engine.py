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

from shaderbox.scripting import EngineContext, ScriptEngine, is_scriptable, stub_for

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


def _write(tmp: Path, name: str, body: str) -> Path:
    scripts_dir = tmp / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    path = scripts_dir / f"{name}.py"
    path.write_text(body, encoding="utf-8")
    return path


def _write_brain(tmp: Path, body: str) -> Path:
    # The node-brain file (feature 044): nodes/<id>/scripts/script.py — one class driving many
    # uniforms via a dict return. The dot keeps it out of the u_*.py glob.
    scripts_dir = tmp / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    path = scripts_dir / "script.py"
    path.write_text(body, encoding="utf-8")
    return path


def _ctx(t: float, dt: float = 1 / 60, frame: int = 0) -> EngineContext:
    return EngineContext(t=t, dt=dt, frame=frame)


def _engine(tmp: Path, node: _FakeNode, name: str = "n0") -> ScriptEngine:
    eng = ScriptEngine()
    eng.reload(name, tmp / "scripts", node)
    return eng


# A scalar class body returning a bare float each frame.
_SCALAR = (
    "class Behavior(ScriptBehavior):\n"
    "    def update(self, ctx: Ctx) -> float:\n"
    "        return 0.5 + 0.3 * sin(ctx.t)\n"
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
    eng.reload("n0", tmp_path / "scripts", node)
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
    # A user `raise ValueError(...)` must surface as its real error, not a NameError from the
    # curated namespace (the common builtin exceptions are seeded).
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
    eng.reload("n0", tmp_path / "scripts", node)
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
    eng.reload("n0", tmp_path / "scripts", node)  # second poll
    assert eng._nodes["n0"].warned == {"u_ghost"}  # unchanged, not re-warned


# ---- scoped determinism ----


def test_t_pure_script_is_deterministic(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "u_x",
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> float:\n"
        "        return sin(ctx.t)\n",
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
    eng.reload("a", tmp_path / "scripts", node_a)
    eng.reload("b", tmp_path / "scripts", node_b)

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
    eng.reload("n0", tmp_path / "scripts", node)  # nothing changed
    assert eng._nodes["n0"].behaviors["u_x"] is first  # same object, not recompiled


def test_cache_recompiles_on_mtime_change(tmp_path: Path) -> None:
    path = _write(tmp_path, "u_x", _SCALAR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    first = eng._nodes["n0"].behaviors["u_x"]
    time.sleep(0.01)
    path.write_text(_SCALAR + "        # changed\n", encoding="utf-8")
    eng.reload("n0", tmp_path / "scripts", node)
    assert eng._nodes["n0"].behaviors["u_x"] is not first  # recompiled


def test_removed_script_drops_binding(tmp_path: Path) -> None:
    path = _write(tmp_path, "u_x", _SCALAR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    assert "u_x" in eng.script_driven_uniforms("n0")
    path.unlink()
    eng.reload("n0", tmp_path / "scripts", node)
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
        _write(tmp_path, name, body)
        node = _FakeNode([uniform])
        eng = ScriptEngine()
        eng.reload(name, tmp_path / "scripts", node)
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
    )
    node = _FakeNode([_u("u_arr", dim=1, n=3, gl_type=_GL_INT)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert tuple(node.uniform_values["u_arr"]) == (1, 3, 4)  # round-half-to-even


def test_stub_for_int_scalar_returns_int(tmp_path: Path) -> None:
    body = stub_for(_u("u_n", gl_type=_GL_INT))  # type: ignore[arg-type]
    assert "-> int" in body and "return 0\n" in body
    _write(tmp_path, "u_n", body)
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
    (scripts_dir / "u_x.py").write_bytes(b"\xff\xfe not utf-8")
    node = _FakeNode([_u("u_x")])
    eng = ScriptEngine()
    eng.reload("n0", scripts_dir, node)  # must NOT raise
    assert "u_x" not in eng.script_driven_uniforms("n0")


# ---- engine-driven reject (review-swarm: a u_time.py would silently no-op) ----


def test_engine_driven_uniform_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "u_time", _SCALAR)
    node = _FakeNode([_u("u_time")])
    eng = ScriptEngine(engine_driven=frozenset({"u_time"}))
    eng.reload("n0", tmp_path / "scripts", node)
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
    eng.reload("n0", tmp_path / "scripts", node)
    assert "u_x" not in eng.script_driven_uniforms("n0")


def test_empty_active_set_keeps_binding_no_false_orphan(tmp_path: Path) -> None:
    # An empty active set = the program is mid-invalidation (a lib edit dropped it); a live binding
    # must NOT be flagged a false orphan and dropped — it returns when the program recompiles.
    _write(tmp_path, "u_x", _SCALAR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    assert "u_x" in eng.script_driven_uniforms("n0")
    node._uniforms = []  # program invalidated this frame
    eng.reload("n0", tmp_path / "scripts", node)
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
    eng.reload("n0", tmp_path / "scripts", node)
    eng.tick("n0", node, _ctx(0.1))
    assert node.uniform_values["u_x"] == 0.3  # both frozen together
    assert node.uniform_values["u_y"] == 0.6
    err = eng.errors[("n0", "script.py")]
    assert err.kind == "runtime" and "ValueError" in err.message
    assert err.line == 4  # the real user line, NOT -1


def test_brain_unknown_key_warns_once_and_skips_no_none_pollution(
    tmp_path: Path,
) -> None:
    # A key naming no active scriptable uniform is warn-once + SKIPPED — never a None write
    # (decision 5: frozen is None for a never-bound name, and writing it would pollute).
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
    assert ("n0", "u_ghost") not in eng.errors
    assert "u_ghost" in eng._nodes["n0"].warned
    eng.tick("n0", node, _ctx(0.1))  # second tick: no re-warn (warn-once)
    assert eng._nodes["n0"].warned == {"u_ghost"}


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
    eng.reload("n0", tmp_path / "scripts", node)
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
    eng.reload("n0", tmp_path / "scripts", node)
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
    eng.reload("n0", tmp_path / "scripts", node)
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
    eng.reload("n0", tmp_path / "scripts", node)
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
