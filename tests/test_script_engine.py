"""CPU-script engine (feature 041, redesigned by 048 to ONE script per node) — pure, no GL. A
SimpleNamespace stands in for moderngl.Uniform (the coercion/shape logic is GL-free; the GL write
reaching the GPU is verified in test_script_engine_gl.py). Covers the single-node-brain contract:
state accumulates, state resets on edit + manual reset, export-instance isolation (live state never
poisons the export, the live instance is not poisoned), the typed outputs (bare scalar / Vec*/Array/
Text) coercing, errors-as-data at the user line, the (path, mtime) cache, scoped determinism, the
soft (node,name) skip for orphan/sampler keys, the silent engine-owned-key drop, brain_status, and
the 048 play/stop model (a stopped uniform's WRITE is skipped while the brain keeps ticking; export
always plays).
"""

import dataclasses
import time
import types
from pathlib import Path

import pytest

from shaderbox.scripting import (
    EngineContext,
    ScriptEngine,
    brain_stub_for,
    is_scriptable,
)

_GL_FLOAT = 0x1406
_GL_UNSIGNED_INT = 0x1405
_GL_INT = 0x1404
_GL_INT_VEC2 = 0x8B53
_GL_SAMPLER_2D = 0x8B5E


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


def _write_brain(tmp: Path, body: str) -> Path:
    # The node-brain file (048): nodes/<id>/scripts/script.py — ONE class driving many uniforms
    # via a dict return. There is no per-uniform file anymore; this is the only script on a node.
    scripts_dir = tmp / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    path = scripts_dir / "script.py"
    path.write_text(body, encoding="utf-8")
    return path


def _ctx(t: float, dt: float = 1 / 60, frame: int = 0) -> EngineContext:
    return EngineContext(t=t, dt=dt, frame=frame)


def _engine(tmp: Path, node: _FakeNode, node_id: str = "n0") -> ScriptEngine:
    eng = ScriptEngine()
    eng.reload(node_id, tmp / "scripts", node)
    return eng


def _brain(*, update_body: str, init_body: str = "") -> str:
    # Assemble a brain class body. `update_body` is the (already-indented-by-8) body of `update`;
    # `init_body` (indented-by-8) is an optional __init__ body.
    head = "class Behavior(ScriptBehavior):\n"
    init = f"    def __init__(self) -> None:\n{init_body}" if init_body else ""
    return f"{head}{init}    def update(self, ctx: Ctx) -> dict:\n{update_body}"


# A brain returning a single bare float — exercises bare-scalar coercion.
_SCALAR = _brain(update_body="        return {'u_x': 0.5}\n")
# A stateful integrator on ONE uniform — only possible with per-instance self.* state.
_INTEGRATOR = _brain(
    init_body="        self.v = 0.0\n",
    update_body="        self.v += ctx.dt\n        return {'u_x': self.v}\n",
)
# A two-uniform integrator: one accumulator drives both u_x and u_y (the headline 048 goal).
_TWO_INTEGRATOR = _brain(
    init_body="        self.v = 0.0\n",
    update_body=(
        "        self.v += ctx.dt\n"
        "        return {'u_x': self.v, 'u_y': self.v * 2.0}\n"
    ),
)


# ---- output types coerce (falsifier: a wrong-shape write or a None hold) ----


def test_scalar_output_coerces(tmp_path: Path) -> None:
    _write_brain(tmp_path, _SCALAR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    # Falsifier: a bare float not coerced/written -> KeyError or != 0.5.
    assert abs(node.uniform_values["u_x"] - 0.5) < 1e-9


def test_vec2_output_coerces(tmp_path: Path) -> None:
    _write_brain(
        tmp_path,
        _brain(update_body="        return {'u_off': Vec2(0.3, 0.7)}\n"),
    )
    node = _FakeNode([_u("u_off", dim=2)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    # Falsifier: a Vec2 not shaped to a 2-tuple.
    assert node.uniform_values["u_off"] == (0.3, 0.7)


def test_vec3_output_coerces(tmp_path: Path) -> None:
    _write_brain(
        tmp_path,
        _brain(update_body="        return {'u_color': Vec3(0.1, 0.2, 0.3)}\n"),
    )
    node = _FakeNode([_u("u_color", dim=3)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    # Falsifier: a Vec3 not shaped to a 3-tuple.
    assert node.uniform_values["u_color"] == (0.1, 0.2, 0.3)


def test_vec4_output_coerces(tmp_path: Path) -> None:
    _write_brain(
        tmp_path,
        _brain(update_body="        return {'u_q': Vec4(0.1, 0.2, 0.3, 0.4)}\n"),
    )
    node = _FakeNode([_u("u_q", dim=4)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    # Falsifier: a Vec4 not shaped to a 4-tuple.
    assert node.uniform_values["u_q"] == (0.1, 0.2, 0.3, 0.4)


def test_array_output_coerces(tmp_path: Path) -> None:
    _write_brain(
        tmp_path,
        _brain(update_body="        return {'u_vals': Array([1.0, 2.0, 3.0, 4.0])}\n"),
    )
    node = _FakeNode([_u("u_vals", dim=1, n=4)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    # Falsifier: an Array not coerced to the float[4] sequence.
    assert tuple(node.uniform_values["u_vals"]) == (1.0, 2.0, 3.0, 4.0)


def test_text_output_coerces(tmp_path: Path) -> None:
    _write_brain(
        tmp_path,
        _brain(update_body="        return {'u_text': Text(\"Hi\")}\n"),
    )
    node = _FakeNode([_u("u_text", dim=1, n=8, gl_type=_GL_UNSIGNED_INT)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    value = node.uniform_values["u_text"]
    # Falsifier: a Text not codepoint-encoded + null-padded to the uint[8] cap.
    assert value[0] == ord("H") and value[1] == ord("i")
    assert len(value) == 8


def test_vec2_array_chunks_into_rows(tmp_path: Path) -> None:
    _write_brain(
        tmp_path,
        _brain(
            update_body="        return {'u_pts': Array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])}\n"
        ),  # vec2[3]
    )
    node = _FakeNode([_u("u_pts", dim=2, n=3)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    # Falsifier: the flat list not chunked into dim-tuples.
    assert node.uniform_values["u_pts"] == [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]


# ---- integer-uniform coercion (a float must round; moderngl rejects a float into an int) ----


def test_int_scalar_rounds(tmp_path: Path) -> None:
    _write_brain(tmp_path, _brain(update_body="        return {'u_n': 2.7}\n"))
    node = _FakeNode([_u("u_n", gl_type=_GL_INT)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    val = node.uniform_values["u_n"]
    # Falsifier: a float written into an int uniform (or no round).
    assert val == 3 and isinstance(val, int)
    assert ("n0", "u_n") not in eng.errors


def test_uint_scalar_rounds(tmp_path: Path) -> None:
    _write_brain(tmp_path, _brain(update_body="        return {'u_count': 3.9}\n"))
    node = _FakeNode([_u("u_count", gl_type=_GL_UNSIGNED_INT)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    val = node.uniform_values["u_count"]
    # Falsifier: a float into a uint uniform.
    assert val == 4 and isinstance(val, int)


def test_ivec2_rounds_components(tmp_path: Path) -> None:
    _write_brain(
        tmp_path,
        _brain(update_body="        return {'u_iv': Vec2(1.4, 2.6)}\n"),
    )
    node = _FakeNode([_u("u_iv", dim=2, gl_type=_GL_INT_VEC2)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    # Falsifier: ivec components not rounded to ints.
    assert node.uniform_values["u_iv"] == (1, 3)
    assert all(isinstance(v, int) for v in node.uniform_values["u_iv"])


def test_int_array_rounds_each(tmp_path: Path) -> None:
    _write_brain(
        tmp_path,
        _brain(update_body="        return {'u_arr': Array([1.4, 2.6, 3.5])}\n"),
    )
    node = _FakeNode([_u("u_arr", dim=1, n=3, gl_type=_GL_INT)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    # Falsifier: an int[] not rounded element-wise (round-half-to-even).
    assert tuple(node.uniform_values["u_arr"]) == (1, 3, 4)


# ---- stateful contract (falsifier: self.* not persisted across frames) ----


def test_state_accumulates(tmp_path: Path) -> None:
    _write_brain(tmp_path, _INTEGRATOR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    for i in range(5):
        eng.tick("n0", node, _ctx(i / 60, dt=1.0, frame=i))
    # Falsifier: != 5.0 means self.v reset each frame (no instance persistence).
    assert node.uniform_values["u_x"] == 5.0


def test_state_resets_on_edit(tmp_path: Path) -> None:
    # Accumulate, then edit the file -> mtime change -> a recompile makes a FRESH instance ->
    # state back to baseline. Falsifier: u_x stays at the accumulated 3.0 (no fresh instance).
    path = _write_brain(tmp_path, _INTEGRATOR)
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
    # reset(node_id) re-runs __init__ on the live brain (no recompile). Falsifier: u_x stays at
    # the accumulated 4.0 (reset didn't re-instantiate).
    _write_brain(tmp_path, _INTEGRATOR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    for i in range(4):
        eng.tick("n0", node, _ctx(0.0, dt=1.0, frame=i))
    assert node.uniform_values["u_x"] == 4.0
    eng.reset("n0")
    eng.tick("n0", node, _ctx(0.0, dt=1.0, frame=0))
    assert node.uniform_values["u_x"] == 1.0


# ---- export isolation (falsifier: the export sees live state, or live state is poisoned) ----


def test_export_instance_isolated_from_live(tmp_path: Path) -> None:
    # Accumulate on the LIVE instance, then a FRESH export instance ticks from a clean __init__ —
    # the export value must NOT inherit live state. Falsifier: export == live (== 10.0).
    _write_brain(tmp_path, _INTEGRATOR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    for i in range(10):
        eng.tick("n0", node, _ctx(0.0, dt=1.0, frame=i))
    live_value = node.uniform_values["u_x"]
    assert live_value == 10.0

    fresh = eng.fresh_behavior_for("n0")
    assert fresh is not None
    export_node = _FakeNode([_u("u_x")])
    eng.tick_export("n0", export_node, _ctx(0.0, dt=1.0, frame=0), fresh)
    assert export_node.uniform_values["u_x"] == 1.0  # cold start, NOT the live 10.0
    assert export_node.uniform_values["u_x"] != live_value


def test_export_does_not_poison_live_instance(tmp_path: Path) -> None:
    # The mirror guarantee: ticking the export instance must NOT advance the LIVE one. Tick the
    # export several times; the live instance keeps its own state. Falsifier: the live value jumps
    # after the export ticks (a shared instance).
    _write_brain(tmp_path, _INTEGRATOR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    for i in range(3):
        eng.tick("n0", node, _ctx(0.0, dt=1.0, frame=i))
    assert node.uniform_values["u_x"] == 3.0

    fresh = eng.fresh_behavior_for("n0")
    assert fresh is not None
    export_node = _FakeNode([_u("u_x")])
    for _i in range(50):
        eng.tick_export("n0", export_node, _ctx(0.0, dt=1.0, frame=0), fresh)

    eng.tick("n0", node, _ctx(0.0, dt=1.0, frame=3))
    assert (
        node.uniform_values["u_x"] == 4.0
    )  # live advanced +1 from 3, untouched by the export


def test_export_tick_does_not_touch_live_errors(tmp_path: Path) -> None:
    # A live binding has a recorded shape error; ticking a FRESH export instance (which writes to a
    # throwaway errors sink) must NOT clear the live error. Falsifier: the live error vanishes.
    _write_brain(
        tmp_path,
        _brain(
            update_body="        return {'u_x': Vec3(1.0, 2.0, 3.0)}\n"
        ),  # vec3->scalar
    )
    node = _FakeNode([_u("u_x")])
    node.uniform_values["u_x"] = 0.0
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert ("n0", "u_x") in eng.errors

    fresh = eng.fresh_behavior_for("n0")
    assert fresh is not None
    export_node = _FakeNode([_u("u_x")])
    export_node.uniform_values["u_x"] = 0.0
    eng.tick_export("n0", export_node, _ctx(0.0), fresh)
    assert ("n0", "u_x") in eng.errors  # live error UNTOUCHED


# ---- errors as data (a broken brain never raises into the tick) ----


def test_compile_error_keys_on_sentinel(tmp_path: Path) -> None:
    _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {\n",  # unterminated dict
    )
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    err = eng.errors[("n0", "script.py")]
    # Falsifier: not a compile error keyed on the sentinel.
    assert err.kind == "compile"


def test_no_subclass_is_compile_error(tmp_path: Path) -> None:
    _write_brain(tmp_path, "x = 1\n")  # no ScriptBehavior subclass at all
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    assert eng.errors[("n0", "script.py")].kind == "compile"


def test_wrong_import_of_injected_type_steers_to_scripting_module(
    tmp_path: Path,
) -> None:
    # A wrong `from shaderbox import ScriptBehavior` (the module names the symbol but the raw
    # ImportError points at the wrong module + never names shaderbox.scripting) must append the
    # canonical-import steer, so the agent self-corrects instead of grepping fruitlessly (043 dogfood).
    _write_brain(
        tmp_path,
        "from shaderbox import ScriptBehavior\n"
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx):\n        return {}\n",
    )
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    msg = eng.errors[("n0", "script.py")].message
    assert "shaderbox.scripting" in msg
    assert "engine injects them" in msg


def test_unrelated_import_error_does_not_get_the_steer(tmp_path: Path) -> None:
    # The steer must NOT false-fire on an import unrelated to the injected scripting types.
    _write_brain(
        tmp_path,
        "from os import notathing\n"
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx):\n        return {}\n",
    )
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    assert "shaderbox.scripting" not in eng.errors[("n0", "script.py")].message


def test_no_update_override_is_compile_error(tmp_path: Path) -> None:
    _write_brain(tmp_path, "class Behavior(ScriptBehavior):\n    pass\n")
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    assert eng.errors[("n0", "script.py")].kind == "compile"


def test_update_missing_self_is_compile_error(tmp_path: Path) -> None:
    _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(ctx) -> dict:\n"  # forgot self
        "        return {}\n",
    )
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    err = eng.errors[("n0", "script.py")]
    assert err.kind == "compile" and "self" in err.message


def test_raising_init_is_compile_error(tmp_path: Path) -> None:
    _write_brain(
        tmp_path,
        _brain(
            init_body="        raise ValueError('boom')\n",
            update_body="        return {}\n",
        ),
    )
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    # Falsifier: a raising __init__ surfaces as anything but a frozen compile error.
    assert eng.errors[("n0", "script.py")].kind == "compile"


def test_reset_recovers_a_once_failing_init(tmp_path: Path) -> None:
    # A raising __init__ freezes; after the cause clears, reset() must re-instantiate AND clear the
    # stale sentinel error so the brain unfreezes. A class var raises on the FIRST construct only.
    body = (
        "class Behavior(ScriptBehavior):\n"
        "    _seen = False\n"
        "    def __init__(self) -> None:\n"
        "        if not Behavior._seen:\n"
        "            Behavior._seen = True\n"
        "            raise ValueError('boom')\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_x': 0.7}\n"
    )
    _write_brain(tmp_path, body)
    node = _FakeNode([_u("u_x")])
    node.uniform_values["u_x"] = 0.0
    eng = _engine(tmp_path, node)
    assert eng.errors[("n0", "script.py")].kind == "compile"  # first __init__ raised

    eng.reset("n0")  # second construct succeeds
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_x"] == 0.7  # unfrozen
    assert ("n0", "script.py") not in eng.errors  # stale sentinel cleared


def test_raw_runtime_throw_freezes_all_at_last_good(tmp_path: Path) -> None:
    # A raw update() exception freezes EVERY name the brain drove last frame (one object = one
    # coherent state) AND records under the sentinel at the CORRECT user line. Falsifier: a name
    # advances past last-good, or the error isn't keyed on the sentinel at the user line.
    path = _write_brain(
        tmp_path,
        _brain(update_body="        return {'u_x': 0.3, 'u_y': 0.6}\n"),
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
    assert node.uniform_values["u_x"] == 0.3 and node.uniform_values["u_y"] == 0.6
    err = eng.errors[("n0", "script.py")]
    assert err.kind == "runtime" and "ValueError" in err.message
    assert err.line == 4  # the real user line, NOT -1


def test_runtime_error_records_deepest_user_line(tmp_path: Path) -> None:
    _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def _bad(self):\n"
        "        return 1.0 / 0.0\n"  # line 3 — the deepest user frame
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_x': self._bad()}\n",
    )
    node = _FakeNode([_u("u_x")])
    node.uniform_values["u_x"] = 0.0
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    # Falsifier: the recorded line isn't the deepest user frame (3), e.g. -1.
    assert eng.errors[("n0", "script.py")].line == 3


def test_user_raised_builtin_exception_keeps_its_real_error(tmp_path: Path) -> None:
    # A user `raise ValueError(...)` surfaces as its real error (real builtins are in scope).
    _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        raise ValueError('nope')\n",
    )
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    err = eng.errors[("n0", "script.py")]
    assert err.kind == "runtime" and "ValueError" in err.message


def test_non_dict_return_is_clean_sentinel_error(tmp_path: Path) -> None:
    # A brain that returns a non-dict is a behavior-level failure under the sentinel — not a crash.
    # Falsifier: tick raises, or the error keys per-uniform instead of the sentinel.
    _write_brain(tmp_path, _brain(update_body="        return 0.5\n"))  # a bare float
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))  # must NOT raise
    err = eng.errors[("n0", "script.py")]
    assert err.kind == "runtime"
    assert ("n0", "u_x") not in eng.errors


def test_none_value_freezes(tmp_path: Path) -> None:
    # A key mapped to None -> coercion rejects -> freeze at last-good, not a silent hold.
    path = _write_brain(tmp_path, _brain(update_body="        return {'u_x': 0.4}\n"))
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_x"] == 0.4

    time.sleep(0.01)
    path.write_text(
        _brain(update_body="        return {'u_x': None}\n"), encoding="utf-8"
    )
    eng.reload("n0", tmp_path / "scripts", node)
    eng.tick("n0", node, _ctx(0.1))
    assert node.uniform_values["u_x"] == 0.4  # frozen at last-good
    assert eng.errors[("n0", "u_x")].kind == "runtime"


def test_per_key_shape_mismatch_freezes_only_that_key(tmp_path: Path) -> None:
    # A per-KEY coercion mismatch freezes ONLY that key; siblings still write. Falsifier: the
    # sibling u_a is also frozen, or the error keys on the sentinel instead of (node, u_b).
    _write_brain(
        tmp_path,
        _brain(
            update_body="        return {'u_a': 0.4, 'u_b': Vec3(1.0, 2.0, 3.0)}\n"
        ),  # vec3 into a scalar
    )
    node = _FakeNode([_u("u_a"), _u("u_b")])
    node.uniform_values["u_b"] = 0.0
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_a"] == 0.4  # sibling wrote
    assert node.uniform_values["u_b"] == 0.0  # frozen
    assert eng.errors[("n0", "u_b")].kind == "runtime"
    assert ("n0", "script.py") not in eng.errors  # NOT the sentinel


def test_array_wrong_length_freezes(tmp_path: Path) -> None:
    _write_brain(
        tmp_path,
        _brain(
            update_body="        return {'u_vals': Array([1.0, 2.0])}\n"
        ),  # 2 for float[4]
    )
    node = _FakeNode([_u("u_vals", dim=1, n=4)])
    node.uniform_values["u_vals"] = (0.0, 0.0, 0.0, 0.0)
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert tuple(node.uniform_values["u_vals"]) == (0.0, 0.0, 0.0, 0.0)  # frozen
    assert eng.errors[("n0", "u_vals")].kind == "runtime"


def test_nan_inf_freezes_and_records(tmp_path: Path) -> None:
    # A non-finite value is no longer written silently (a black-frame footgun that also poisons
    # last-good) — it freezes at last-good + records a runtime ScriptError. Falsifier: inf is
    # written, or no error recorded.
    path = _write_brain(tmp_path, _brain(update_body="        return {'u_x': 0.3}\n"))
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_x"] == 0.3

    time.sleep(0.01)
    path.write_text(
        _brain(update_body="        return {'u_x': float('inf')}\n"), encoding="utf-8"
    )
    eng.reload("n0", tmp_path / "scripts", node)
    eng.tick("n0", node, _ctx(0.1))
    assert node.uniform_values["u_x"] == 0.3  # frozen at last-good, NOT inf
    err = eng.errors[("n0", "u_x")]
    assert err.kind == "runtime" and "finite" in err.message


def test_array_nested_tuples_gives_flatten_hint(tmp_path: Path) -> None:
    # The vecN[M] footgun (a list of tuples) surfaces a clear flatten hint, not the cryptic
    # "float() argument must be ... not 'tuple'". NOTE (brain change vs the 044 per-uniform world):
    # `Array(...)` raises its TypeError inside `update()` BEFORE the dict returns, so the whole brain
    # freezes at the SENTINEL key — not a per-key (node, u_pts) error. The hint is still surfaced.
    # Falsifier: no flatten hint, or the cryptic float() message leaks.
    _write_brain(
        tmp_path,
        _brain(
            update_body="        return {'u_pts': Array([(0.0, 1.0), (2.0, 3.0)])}\n"
        ),  # nested, wrong
    )
    node = _FakeNode([_u("u_pts", dim=2, n=2)])
    node.uniform_values["u_pts"] = [(0.0, 0.0), (0.0, 0.0)]
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_pts"] == [(0.0, 0.0), (0.0, 0.0)]  # frozen
    err = eng.errors[("n0", "script.py")]
    assert err.kind == "runtime"
    assert "flattened" in err.message and "float()" not in err.message


# ---- soft (node,name) errors: orphan/typo + sampler/block keys skipped, not driven ----


def test_orphan_key_warns_skips_and_records_soft_error(tmp_path: Path) -> None:
    # A key naming no active scriptable uniform is warn-once + SKIPPED (never a None write) AND
    # records a SOFT ScriptError under (node, name). It must NOT claim ownership in
    # script_driven_uniforms. Falsifier: u_ghost written, or driven, or no soft error.
    _write_brain(
        tmp_path,
        _brain(update_body="        return {'u_a': 0.5, 'u_ghost': 0.9}\n"),
    )
    node = _FakeNode([_u("u_a")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_a"] == 0.5
    assert "u_ghost" not in node.uniform_values  # NOT written as None
    err = eng.errors[("n0", "u_ghost")]
    assert err.kind == "runtime" and "orphan" in err.message
    assert "u_ghost" not in eng.script_driven_uniforms("n0")  # claims no ownership
    assert "u_ghost" in eng._nodes["n0"].warned
    eng.tick("n0", node, _ctx(0.1))  # second tick: warn-once
    assert eng._nodes["n0"].warned == {"u_ghost"}


def test_sampler_key_records_soft_error_and_is_skipped(tmp_path: Path) -> None:
    # A key naming a sampler (non-scriptable) records a soft (node,name) error + is skipped (not
    # driven). Falsifier: the sampler is driven, or no soft error.
    _write_brain(
        tmp_path,
        _brain(update_body="        return {'u_a': 0.5, 'u_tex': 0.1}\n"),
    )
    node = _FakeNode([_u("u_a"), _u("u_tex", gl_type=_GL_SAMPLER_2D)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_a"] == 0.5
    err = eng.errors[("n0", "u_tex")]
    assert err.kind == "runtime" and "sampler" in err.message
    assert "u_tex" not in eng.script_driven_uniforms("n0")


def test_orphan_key_error_clears_when_key_fixed(tmp_path: Path) -> None:
    # Once the bad key stops being returned (the user fixes the typo), its (node, name) soft error
    # is cleared on the next tick. Falsifier: the zombie error persists.
    path = _write_brain(
        tmp_path,
        _brain(update_body="        return {'u_a': 0.5, 'u_ghost': 0.9}\n"),
    )
    node = _FakeNode([_u("u_a")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert ("n0", "u_ghost") in eng.errors

    time.sleep(0.01)
    path.write_text(
        _brain(update_body="        return {'u_a': 0.5}\n"), encoding="utf-8"
    )
    eng.reload("n0", tmp_path / "scripts", node)
    eng.tick("n0", node, _ctx(0.1))
    assert ("n0", "u_ghost") not in eng.errors  # zombie cleared


# ---- engine-owned key dropped SILENTLY (no error, not driven) ----


def test_engine_owned_key_dropped_silently(tmp_path: Path) -> None:
    # A brain naming an engine-owned uniform (u_time) SILENTLY drops the key — no false ownership
    # AND no soft error (the renderer owns that slot). Falsifier: u_time is driven, a soft error
    # appears, or the renderer's value is overwritten.
    _write_brain(
        tmp_path,
        _brain(update_body="        return {'u_a': 0.5, 'u_time': 9.0}\n"),
    )
    node = _FakeNode([_u("u_a"), _u("u_time")])
    node.uniform_values["u_time"] = 1.23  # the renderer's value
    eng = ScriptEngine(engine_driven=frozenset({"u_time"}))
    eng.reload("n0", tmp_path / "scripts", node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_a"] == 0.5
    assert "u_time" not in eng.script_driven_uniforms("n0")  # no false ownership
    assert ("n0", "u_time") not in eng.errors  # silently dropped
    assert node.uniform_values["u_time"] == 1.23  # renderer value untouched


# ---- (path, mtime) cache ----


def test_cache_no_recompile_when_mtime_unchanged(tmp_path: Path) -> None:
    _write_brain(tmp_path, _SCALAR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    first = eng._nodes["n0"].brain
    eng.reload("n0", tmp_path / "scripts", node)  # nothing changed
    # Falsifier: a fresh brain object means an unnecessary recompile.
    assert eng._nodes["n0"].brain is first


def test_cache_recompiles_on_mtime_change(tmp_path: Path) -> None:
    path = _write_brain(tmp_path, _SCALAR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    first = eng._nodes["n0"].brain
    time.sleep(0.01)
    path.write_text(_SCALAR + "        # changed\n", encoding="utf-8")
    eng.reload("n0", tmp_path / "scripts", node)
    assert eng._nodes["n0"].brain is not first  # recompiled


# ---- binding by existence (the file IS the binding; no active flag) ----


def test_brain_binds_by_existence(tmp_path: Path) -> None:
    _write_brain(tmp_path, _SCALAR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    # Falsifier: the brain didn't bind despite script.py existing on disk.
    assert eng.has_script("n0")
    assert "u_x" in eng.script_driven_uniforms("n0")


def test_removed_script_drops_brain(tmp_path: Path) -> None:
    path = _write_brain(tmp_path, _SCALAR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert "u_x" in eng.script_driven_uniforms("n0")
    path.unlink()
    eng.reload("n0", tmp_path / "scripts", node)
    assert not eng.has_script("n0")
    assert "u_x" not in eng.script_driven_uniforms("n0")


def test_no_script_file_is_no_op_tick(tmp_path: Path) -> None:
    # A node dir with no script.py: reload binds nothing, tick is a no-op. Falsifier: tick raises
    # or invents a driven uniform.
    (tmp_path / "scripts").mkdir(exist_ok=True)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    assert not eng.has_script("n0")
    eng.tick("n0", node, _ctx(0.0))  # must NOT raise
    assert eng.script_driven_uniforms("n0") == set()


# ---- scoped determinism ----


def test_t_pure_script_is_deterministic(tmp_path: Path) -> None:
    # A ctx.t-pure update is identical across dt (the scoped-determinism guarantee). Falsifier: a
    # different dt at the same t yields a different value.
    _write_brain(
        tmp_path,
        "import math\n"
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_x': math.sin(ctx.t)}\n",
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
    # expected (determinism is scoped to ctx.t-pure scripts), not a violation.
    body = (
        "class Behavior(ScriptBehavior):\n"
        "    def __init__(self) -> None:\n"
        "        self.prev = 1.0\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        self.prev = self.prev + self.prev * ctx.dt\n"
        "        return {'u_x': self.prev}\n"
    )
    _write_brain(tmp_path, body)
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


# ---- brain_status (sentinel + soft errors + driven_count) ----


def test_brain_status_reflects_driven_count(tmp_path: Path) -> None:
    _write_brain(tmp_path, _TWO_INTEGRATOR)
    node = _FakeNode([_u("u_x"), _u("u_y")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0, dt=1.0, frame=0))
    status = eng.brain_status("n0")
    assert status is not None
    # Falsifier: a wrong driven_count, or a phantom sentinel/soft error on a clean brain.
    assert status.driven_count == 2
    assert status.sentinel_error is None
    assert status.soft_errors == []


def test_brain_status_reflects_sentinel_and_soft_errors(tmp_path: Path) -> None:
    # A brain with an orphan key: driven_count counts only the real driven uniform; the orphan
    # surfaces in soft_errors. Falsifier: the orphan inflates driven_count or is missing from soft.
    _write_brain(
        tmp_path,
        _brain(update_body="        return {'u_a': 0.5, 'u_ghost': 0.9}\n"),
    )
    node = _FakeNode([_u("u_a")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    status = eng.brain_status("n0")
    assert status is not None
    assert status.driven_count == 1  # only u_a
    assert status.sentinel_error is None
    assert [name for name, _ in status.soft_errors] == ["u_ghost"]


def test_brain_status_none_without_script(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir(exist_ok=True)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    assert eng.brain_status("n0") is None


# ---- drop_node clears state ----


def test_drop_node_clears_state_and_errors(tmp_path: Path) -> None:
    _write_brain(tmp_path, "class Behavior(ScriptBehavior):\n    pass\n")  # compile err
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    assert ("n0", "script.py") in eng.errors
    eng.drop_node("n0")
    assert ("n0", "script.py") not in eng.errors
    assert eng.script_driven_uniforms("n0") == set()
    assert not eng.has_script("n0")


# ---- reload robustness (read_text must not crash the frame loop) ----


def test_non_utf8_file_does_not_crash_reload(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "script.py").write_bytes(b"\xff\xfe not utf-8")
    node = _FakeNode([_u("u_x")])
    eng = ScriptEngine()
    eng.reload("n0", scripts_dir, node)  # must NOT raise
    assert "u_x" not in eng.script_driven_uniforms("n0")


def test_unreadable_rewrite_mid_edit_keeps_cached_brain(tmp_path: Path) -> None:
    # A reload that races a half-saved / non-UTF8 rewrite at a changed mtime keeps the prior brain
    # rather than crashing the frame loop. Falsifier: reload raises, or the cached brain is dropped.
    path = _write_brain(tmp_path, _SCALAR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    first = eng._nodes["n0"].brain
    assert first is not None
    time.sleep(0.01)
    path.write_bytes(b"\xff\xfe still here but unreadable")
    eng.reload("n0", tmp_path / "scripts", node)  # must NOT raise
    assert eng._nodes["n0"].brain is first  # cached brain kept


# ---- namespace: imports, super, builtins (the engine's own idioms resolve) ----


def test_import_math_works(tmp_path: Path) -> None:
    _write_brain(
        tmp_path,
        "import math\n"
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_x': math.cos(0.0)}\n",  # 1.0
    )
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert ("n0", "u_x") not in eng.errors
    assert abs(node.uniform_values["u_x"] - 1.0) < 1e-9


def test_explicit_import_line_resolves(tmp_path: Path) -> None:
    # 048 decision 8: the stub emits a real `from shaderbox.scripting import ...`; it must RESOLVE
    # inside the exec'd script. Falsifier: the import raises an opaque compile-freeze.
    _write_brain(
        tmp_path,
        "from shaderbox.scripting import ScriptBehavior, Ctx, Vec2\n"
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_off': Vec2(0.1, 0.2)}\n",
    )
    node = _FakeNode([_u("u_off", dim=2)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert ("n0", "script.py") not in eng.errors
    assert node.uniform_values["u_off"] == (0.1, 0.2)


def test_super_and_containers_resolve(tmp_path: Path) -> None:
    _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def __init__(self) -> None:\n"
        "        super().__init__()\n"
        "        self.buf = []\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        self.buf.append(ctx.t)\n"
        "        return {'u_x': sum(self.buf) / len(self.buf)}\n",
    )
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(2.0))
    assert ("n0", "script.py") not in eng.errors  # super + list + sum resolved
    assert node.uniform_values["u_x"] == 2.0


def test_chr_ord_available_for_codepoint_text(tmp_path: Path) -> None:
    _write_brain(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_t': Text(chr(65) + chr(66))}\n",  # 'AB'
    )
    node = _FakeNode([_u("u_t", dim=1, n=8, gl_type=_GL_UNSIGNED_INT)])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    val = node.uniform_values["u_t"]
    assert val[0] == ord("A") and val[1] == ord("B")
    assert ("n0", "u_t") not in eng.errors


# ---- is_scriptable + brain_stub_for ----


def test_is_scriptable_gate() -> None:
    assert is_scriptable(_u("u_x"))
    assert is_scriptable(_u("u_v", dim=3))
    assert not is_scriptable(_u("u_tex", gl_type=_GL_SAMPLER_2D))
    assert not is_scriptable(object())  # no shape attrs


def test_brain_stub_compiles_runs_and_drives_nothing_by_default(tmp_path: Path) -> None:
    # The 048 stub: an empty-dict default (a fresh script drives nothing) with commented examples.
    # It must compile + run without error AND drive no uniform. Falsifier: the stub errors, or it
    # drives a uniform by default (a non-empty live body).
    uniforms = [
        _u("u_s"),
        _u("u_v2", dim=2),
        _u("u_v3", dim=3),
        _u("u_v4", dim=4),
        _u("u_arr", dim=1, n=4),
        _u("u_txt", dim=1, n=8, gl_type=_GL_UNSIGNED_INT),
    ]
    body = brain_stub_for(uniforms)  # type: ignore[arg-type]
    _write_brain(tmp_path, body)
    node = _FakeNode(uniforms)
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0))
    assert not any(k[0] == "n0" for k in eng.errors), "fresh stub errored"
    assert eng.script_driven_uniforms("n0") == set()  # empty-dict default
    assert node.uniform_values == {}


def test_brain_stub_emits_explicit_import_for_referenced_types(tmp_path: Path) -> None:
    # The stub's import line names ScriptBehavior + Ctx always, plus only the output types the
    # node's uniforms reference. Falsifier: a referenced type missing, or an unreferenced one emitted.
    body = brain_stub_for([_u("u_v3", dim=3)])  # type: ignore[arg-type]
    assert "from shaderbox.scripting import" in body
    assert "ScriptBehavior" in body and "Ctx" in body and "Vec3" in body
    assert "Vec2" not in body and "Text" not in body  # not referenced


# ---- ctx is frozen ----


def test_ctx_is_immutable() -> None:
    ctx = EngineContext(t=1.0, dt=0.1, frame=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.t = 2.0  # type: ignore[misc]


# ---- perf sanity (not a gate) ----


def test_tick_is_cheap(tmp_path: Path) -> None:
    _write_brain(tmp_path, _SCALAR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    start = time.perf_counter()
    for i in range(1000):
        eng.tick("n0", node, _ctx(i / 60))
    per_tick_us = (time.perf_counter() - start) / 1000 * 1e6
    assert per_tick_us < 500  # generous ceiling


# ============================================================================
# Play/stop (048 decisions 4-7): a STOPPED uniform's WRITE is skipped, but the brain still ticks
# (state advances + the name stays driven); the export path forwards NO stopped set (always plays).
# ============================================================================


def test_stopped_uniform_write_skipped_but_still_driven_and_sibling_advances(
    tmp_path: Path,
) -> None:
    # The core stopped-skip canary (decision 4/5): tick with stopped={'u_x'} ->
    #   - u_x is NOT written (stays the pre-tick manual value),
    #   - u_x IS still in script_driven_uniforms (last_driven — keeps its PLAY button),
    #   - the brain's OTHER driven uniform u_y IS written (the brain ticked, it just skipped one write).
    # Falsifier: u_x changed, OR u_x absent from driven, OR u_y not advanced.
    _write_brain(tmp_path, _TWO_INTEGRATOR)
    node = _FakeNode([_u("u_x"), _u("u_y")])
    node.uniform_values["u_x"] = 0.42  # the user's frozen manual value
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0, dt=1.0, frame=0), stopped=frozenset({"u_x"}))
    assert (
        node.uniform_values["u_x"] == 0.42
    )  # frozen at the manual value, NOT self.v (==1.0)
    assert "u_x" in eng.script_driven_uniforms("n0")  # still driven
    assert (
        node.uniform_values["u_y"] == 2.0
    )  # sibling advanced (self.v*2 with self.v==1.0)


def test_stopped_brain_keeps_ticking_then_resumes_advanced(tmp_path: Path) -> None:
    # Decision 5: the brain keeps TICKING while a uniform is stopped, so on resume the value jumps
    # to the value the integrator reached WHILE stopped — NOT the value at stop time. Falsifier: the
    # resumed value equals the stop-instant value (the brain stopped advancing).
    _write_brain(tmp_path, _INTEGRATOR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0, dt=1.0, frame=0))
    assert node.uniform_values["u_x"] == 1.0  # value at the moment we stop

    node.uniform_values["u_x"] = 99.0  # the user's manual value while stopped
    for i in range(1, 4):  # three stopped ticks: self.v advances 2,3,4 but no write
        eng.tick("n0", node, _ctx(0.0, dt=1.0, frame=i), stopped=frozenset({"u_x"}))
    assert node.uniform_values["u_x"] == 99.0  # never written while stopped

    eng.tick("n0", node, _ctx(0.0, dt=1.0, frame=4))  # resume (not stopped)
    # self.v advanced to 5 across the stop window (1 initial + 4 ticks of dt=1.0).
    assert node.uniform_values["u_x"] == 5.0  # advanced state, NOT the 1.0 stop value


def test_export_always_plays_a_live_stopped_uniform(tmp_path: Path) -> None:
    # Decision 5 + export-isolation: tick_export forwards NO stopped set, so an export writes the
    # SCRIPT value even for a uniform stopped in the live preview. Falsifier: the export freezes the
    # stopped manual value (tick_export leaked the live stopped set).
    _write_brain(tmp_path, _INTEGRATOR)
    node = _FakeNode([_u("u_x")])
    eng = _engine(tmp_path, node)
    eng.tick("n0", node, _ctx(0.0, dt=1.0, frame=0))
    node.uniform_values["u_x"] = 99.0
    eng.tick("n0", node, _ctx(0.0, dt=1.0, frame=1), stopped=frozenset({"u_x"}))
    assert node.uniform_values["u_x"] == 99.0  # frozen live

    fresh = eng.fresh_behavior_for("n0")
    assert fresh is not None
    export_node = _FakeNode([_u("u_x")])
    eng.tick_export("n0", export_node, _ctx(0.0, dt=1.0, frame=0), fresh)
    # The export plays: a fresh instance ticks once -> self.v == 1.0, WRITTEN (not the frozen 99.0).
    assert export_node.uniform_values["u_x"] == 1.0
