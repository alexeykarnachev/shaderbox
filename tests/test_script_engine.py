"""CPU-script engine (feature 041, redesigned by 048 to ONE script per document) — pure, no GL. A
SimpleNamespace stands in for moderngl.Uniform (the coercion/shape logic is GL-free; the GL write
reaching the GPU is verified in test_script_engine_gl.py). Covers the single-document script contract:
state accumulates, state resets on edit + manual reset, export-instance isolation (live state never
poisons the export, the live instance is not poisoned), the typed outputs (bare scalar / Vec*/Array/
Text) coercing, errors-as-data at the user line, the (path, mtime) cache, scoped determinism, the
soft (document,name) skip for orphan/sampler keys, the silent engine-owned-key drop, script_status, and
the 048 play/stop model (a stopped uniform's WRITE is skipped while the script keeps ticking; export
always plays).
"""

import ast
import time
import types
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from shaderbox.core import Pass
from shaderbox.scripting import (
    EngineContext,
    ScriptEngine,
    StoppedKey,
    coerce_one,
    is_scriptable,
    script_stub_for,
)
from shaderbox.scripting.behavior import _RuntimeScriptError
from shaderbox.scripting.errors import ScriptError

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


class _FakePass:
    # The ScriptPass slice: a uniform_values dict + get_active_uniforms() + script_ready.
    def __init__(
        self, uniforms: list[types.SimpleNamespace], *, ready: bool = True
    ) -> None:
        self.uniform_values: dict[str, object] = {}
        self.script_ready = ready
        self._uniforms = uniforms

    def get_active_uniforms(self) -> list[types.SimpleNamespace]:
        return self._uniforms


class _FakeDocument:
    # The ScriptTarget slice: passes by name. A bare uniform list builds the one-pass document most
    # tests want (069's broadcast rule means their `{"u_x": ...}` scripts need no edit); a dict of
    # name -> uniforms builds a multi-pass one.
    def __init__(
        self,
        uniforms: list[types.SimpleNamespace] | dict[str, list[types.SimpleNamespace]],
    ) -> None:
        by_pass = {"main": uniforms} if isinstance(uniforms, list) else uniforms
        self.passes: dict[str, _FakePass] = {
            name: _FakePass(us) for name, us in by_pass.items()
        }

    @property
    def uniform_values(self) -> dict[str, object]:
        # The one-pass shorthand every single-pass test reads through.
        return self.passes["main"].uniform_values


def _write_script(tmp: Path, body: str) -> Path:
    # The document script file (048): documents/<id>/scripts/script.py — ONE class driving many uniforms
    # via a dict return. There is no per-uniform file anymore; this is the only script on a document.
    scripts_dir = tmp / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    path = scripts_dir / "script.py"
    path.write_text(body, encoding="utf-8")
    return path


def _ctx(t: float, dt: float = 1 / 60, frame: int = 0) -> EngineContext:
    return EngineContext(t=t, dt=dt, frame=frame)


def _engine(
    tmp: Path, document: _FakeDocument, document_id: str = "n0"
) -> ScriptEngine:
    eng = ScriptEngine()
    eng.reload(document_id, tmp / "scripts", document)
    return eng


def _script(*, update_body: str, init_body: str = "") -> str:
    # Assemble a script class body. `update_body` is the (already-indented-by-8) body of `update`;
    # `init_body` (indented-by-8) is an optional __init__ body.
    head = "class Behavior(ScriptBehavior):\n"
    init = f"    def __init__(self) -> None:\n{init_body}" if init_body else ""
    return f"{head}{init}    def update(self, ctx: Ctx) -> dict:\n{update_body}"


# A script returning a single bare float — exercises bare-scalar coercion.
_SCALAR = _script(update_body="        return {'u_x': 0.5}\n")
# A stateful integrator on ONE uniform — only possible with per-instance self.* state.
_INTEGRATOR = _script(
    init_body="        self.v = 0.0\n",
    update_body="        self.v += ctx.dt\n        return {'u_x': self.v}\n",
)
# A two-uniform integrator: one accumulator drives both u_x and u_y (the headline 048 goal).
_TWO_INTEGRATOR = _script(
    init_body="        self.v = 0.0\n",
    update_body=(
        "        self.v += ctx.dt\n"
        "        return {'u_x': self.v, 'u_y': self.v * 2.0}\n"
    ),
)


# ---- output types coerce (falsifier: a wrong-shape write or a None hold) ----


def test_scalar_output_coerces(tmp_path: Path) -> None:
    _write_script(tmp_path, _SCALAR)
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    # Falsifier: a bare float not coerced/written -> KeyError or != 0.5.
    assert abs(document.uniform_values["u_x"] - 0.5) < 1e-9


def test_vec2_output_coerces(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        _script(update_body="        return {'u_off': [0.3, 0.7]}\n"),
    )
    document = _FakeDocument([_u("u_off", dim=2)])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    # Falsifier: a Vec2 not shaped to a 2-tuple.
    assert document.uniform_values["u_off"] == (0.3, 0.7)


def test_vec3_output_coerces(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        _script(update_body="        return {'u_color': [0.1, 0.2, 0.3]}\n"),
    )
    document = _FakeDocument([_u("u_color", dim=3)])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    # Falsifier: a Vec3 not shaped to a 3-tuple.
    assert document.uniform_values["u_color"] == (0.1, 0.2, 0.3)


def test_vec4_output_coerces(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        _script(update_body="        return {'u_q': [0.1, 0.2, 0.3, 0.4]}\n"),
    )
    document = _FakeDocument([_u("u_q", dim=4)])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    # Falsifier: a Vec4 not shaped to a 4-tuple.
    assert document.uniform_values["u_q"] == (0.1, 0.2, 0.3, 0.4)


def test_array_output_coerces(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        _script(update_body="        return {'u_vals': [1.0, 2.0, 3.0, 4.0]}\n"),
    )
    document = _FakeDocument([_u("u_vals", dim=1, n=4)])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    # Falsifier: an Array not coerced to the float[4] sequence.
    assert tuple(document.uniform_values["u_vals"]) == (1.0, 2.0, 3.0, 4.0)


def test_text_output_coerces(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        _script(update_body="        return {'u_text': \"Hi\"}\n"),
    )
    document = _FakeDocument([_u("u_text", dim=1, n=8, gl_type=_GL_UNSIGNED_INT)])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    value = document.uniform_values["u_text"]
    # Falsifier: a Text not codepoint-encoded + null-padded to the uint[8] cap.
    assert value[0] == ord("H") and value[1] == ord("i")
    assert len(value) == 8


def test_vec2_array_chunks_into_rows(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        _script(
            update_body="        return {'u_pts': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}\n"
        ),  # vec2[3]
    )
    document = _FakeDocument([_u("u_pts", dim=2, n=3)])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    # Falsifier: the flat list not chunked into dim-tuples.
    assert document.uniform_values["u_pts"] == [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]


# ---- integer-uniform coercion (a float must round; moderngl rejects a float into an int) ----


def test_int_scalar_rounds(tmp_path: Path) -> None:
    _write_script(tmp_path, _script(update_body="        return {'u_n': 2.7}\n"))
    document = _FakeDocument([_u("u_n", gl_type=_GL_INT)])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    val = document.uniform_values["u_n"]
    # Falsifier: a float written into an int uniform (or no round).
    assert val == 3 and isinstance(val, int)
    assert ("n0", "main", "u_n") not in eng.errors


def test_uint_scalar_rounds(tmp_path: Path) -> None:
    _write_script(tmp_path, _script(update_body="        return {'u_count': 3.9}\n"))
    document = _FakeDocument([_u("u_count", gl_type=_GL_UNSIGNED_INT)])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    val = document.uniform_values["u_count"]
    # Falsifier: a float into a uint uniform.
    assert val == 4 and isinstance(val, int)


def test_ivec2_rounds_components(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        _script(update_body="        return {'u_iv': [1.4, 2.6]}\n"),
    )
    document = _FakeDocument([_u("u_iv", dim=2, gl_type=_GL_INT_VEC2)])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    # Falsifier: ivec components not rounded to ints.
    assert document.uniform_values["u_iv"] == (1, 3)
    assert all(isinstance(v, int) for v in document.uniform_values["u_iv"])


def test_int_array_rounds_each(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        _script(update_body="        return {'u_arr': [1.4, 2.6, 3.5]}\n"),
    )
    document = _FakeDocument([_u("u_arr", dim=1, n=3, gl_type=_GL_INT)])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    # Falsifier: an int[] not rounded element-wise (round-half-to-even).
    assert tuple(document.uniform_values["u_arr"]) == (1, 3, 4)


# ---- stateful contract (falsifier: self.* not persisted across frames) ----


def test_state_accumulates(tmp_path: Path) -> None:
    _write_script(tmp_path, _INTEGRATOR)
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    for i in range(5):
        eng.tick("n0", document, _ctx(i / 60, dt=1.0, frame=i))
    # Falsifier: != 5.0 means self.v reset each frame (no instance persistence).
    assert document.uniform_values["u_x"] == 5.0


def test_state_resets_on_edit(tmp_path: Path) -> None:
    # Accumulate, then edit the file -> mtime change -> a recompile makes a FRESH instance ->
    # state back to baseline. Falsifier: u_x stays at the accumulated 3.0 (no fresh instance).
    path = _write_script(tmp_path, _INTEGRATOR)
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    for i in range(3):
        eng.tick("n0", document, _ctx(0.0, dt=1.0, frame=i))
    assert document.uniform_values["u_x"] == 3.0

    time.sleep(0.01)
    path.write_text(_INTEGRATOR + "        # edited\n", encoding="utf-8")
    eng.reload("n0", tmp_path / "scripts", document)
    eng.tick("n0", document, _ctx(0.0, dt=1.0, frame=0))
    assert document.uniform_values["u_x"] == 1.0  # fresh instance: self.v was 0, +1 dt


def test_manual_reset_clears_state(tmp_path: Path) -> None:
    # reset(document_id) re-runs __init__ on the live script (no recompile). Falsifier: u_x stays at
    # the accumulated 4.0 (reset didn't re-instantiate).
    _write_script(tmp_path, _INTEGRATOR)
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    for i in range(4):
        eng.tick("n0", document, _ctx(0.0, dt=1.0, frame=i))
    assert document.uniform_values["u_x"] == 4.0
    eng.reset("n0")
    eng.tick("n0", document, _ctx(0.0, dt=1.0, frame=0))
    assert document.uniform_values["u_x"] == 1.0


# ---- export isolation (falsifier: the export sees live state, or live state is poisoned) ----


def test_export_instance_isolated_from_live(tmp_path: Path) -> None:
    # Accumulate on the LIVE instance, then a FRESH export instance ticks from a clean __init__ —
    # the export value must NOT inherit live state. Falsifier: export == live (== 10.0).
    _write_script(tmp_path, _INTEGRATOR)
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    for i in range(10):
        eng.tick("n0", document, _ctx(0.0, dt=1.0, frame=i))
    live_value = document.uniform_values["u_x"]
    assert live_value == 10.0

    fresh = eng.fresh_behavior_for("n0")
    assert fresh is not None
    export_document = _FakeDocument([_u("u_x")])
    eng.tick_export("n0", export_document, _ctx(0.0, dt=1.0, frame=0), fresh)
    assert export_document.uniform_values["u_x"] == 1.0  # cold start, NOT the live 10.0
    assert export_document.uniform_values["u_x"] != live_value


def test_export_does_not_poison_live_instance(tmp_path: Path) -> None:
    # The mirror guarantee: ticking the export instance must NOT advance the LIVE one. Tick the
    # export several times; the live instance keeps its own state. Falsifier: the live value jumps
    # after the export ticks (a shared instance).
    _write_script(tmp_path, _INTEGRATOR)
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    for i in range(3):
        eng.tick("n0", document, _ctx(0.0, dt=1.0, frame=i))
    assert document.uniform_values["u_x"] == 3.0

    fresh = eng.fresh_behavior_for("n0")
    assert fresh is not None
    export_document = _FakeDocument([_u("u_x")])
    for _i in range(50):
        eng.tick_export("n0", export_document, _ctx(0.0, dt=1.0, frame=0), fresh)

    eng.tick("n0", document, _ctx(0.0, dt=1.0, frame=3))
    assert (
        document.uniform_values["u_x"] == 4.0
    )  # live advanced +1 from 3, untouched by the export


def test_export_tick_does_not_touch_live_errors(tmp_path: Path) -> None:
    # A live binding has a recorded shape error; ticking a FRESH export instance (which writes to a
    # throwaway errors sink) must NOT clear the live error. Falsifier: the live error vanishes.
    _write_script(
        tmp_path,
        _script(
            update_body="        return {'u_x': [1.0, 2.0, 3.0]}\n"
        ),  # vec3->scalar
    )
    document = _FakeDocument([_u("u_x")])
    document.uniform_values["u_x"] = 0.0
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    assert ("n0", "main", "u_x") in eng.errors

    fresh = eng.fresh_behavior_for("n0")
    assert fresh is not None
    export_document = _FakeDocument([_u("u_x")])
    export_document.uniform_values["u_x"] = 0.0
    eng.tick_export("n0", export_document, _ctx(0.0), fresh)
    assert ("n0", "main", "u_x") in eng.errors  # live error UNTOUCHED


# ---- errors as data (a broken script never raises into the tick) ----


def test_compile_error_keys_on_sentinel(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {\n",  # unterminated dict
    )
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    err = eng.errors[("n0", "", "script.py")]
    # Falsifier: not a compile error keyed on the sentinel.
    assert err.kind == "compile"


def test_no_subclass_is_compile_error(tmp_path: Path) -> None:
    _write_script(tmp_path, "x = 1\n")  # no ScriptBehavior subclass at all
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    assert eng.errors[("n0", "", "script.py")].kind == "compile"


def test_a_wrong_import_names_the_right_module(tmp_path: Path) -> None:
    # A wrong `from shaderbox import ScriptBehavior` must name `shaderbox.scripting`, so the
    # reader self-corrects instead of grepping fruitlessly (043 dogfood). The import gate's own
    # message does this now; before the gate it was the appended steer.
    _write_script(
        tmp_path,
        "from shaderbox import ScriptBehavior\n"
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx):\n        return {}\n",
    )
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    msg = eng.errors[("n0", "", "script.py")].message
    assert "shaderbox.scripting" in msg


def test_a_missing_import_of_a_scripting_type_names_the_import_line(
    tmp_path: Path,
) -> None:
    # The other half: a script that USES `Vec3` without importing it. The engine injects the
    # names as a fallback, so this only surfaces where the injection cannot help — an eager
    # annotation on a name the user misspelled. Falsifier: drop the hint and the message is a
    # bare NameError with no route to the import line.
    from shaderbox.scripting.behavior import _import_hint

    assert "from shaderbox.scripting import Ctx" in _import_hint(
        NameError("name 'Ctx' is not defined", name="Ctx")
    )
    assert _import_hint(ImportError("No module named 'numpy'")) == ""


def test_unrelated_import_error_does_not_get_the_steer(tmp_path: Path) -> None:
    # The steer must NOT false-fire on an import unrelated to the injected scripting types.
    _write_script(
        tmp_path,
        "from os import notathing\n"
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx):\n        return {}\n",
    )
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    assert "shaderbox.scripting" not in eng.errors[("n0", "", "script.py")].message


def test_no_update_override_is_compile_error(tmp_path: Path) -> None:
    _write_script(tmp_path, "class Behavior(ScriptBehavior):\n    pass\n")
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    assert eng.errors[("n0", "", "script.py")].kind == "compile"


def test_update_missing_self_is_compile_error(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(ctx) -> dict:\n"  # forgot self
        "        return {}\n",
    )
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    err = eng.errors[("n0", "", "script.py")]
    assert err.kind == "compile" and "self" in err.message


def test_raising_init_is_compile_error(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        _script(
            init_body="        raise ValueError('boom')\n",
            update_body="        return {}\n",
        ),
    )
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    # Falsifier: a raising __init__ surfaces as anything but a frozen compile error.
    assert eng.errors[("n0", "", "script.py")].kind == "compile"


def test_reset_recovers_a_once_failing_init(tmp_path: Path) -> None:
    # A raising __init__ freezes; after the cause clears, reset() must re-instantiate AND clear the
    # stale sentinel error so the script unfreezes. A class var raises on the FIRST construct only.
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
    _write_script(tmp_path, body)
    document = _FakeDocument([_u("u_x")])
    document.uniform_values["u_x"] = 0.0
    eng = _engine(tmp_path, document)
    assert (
        eng.errors[("n0", "", "script.py")].kind == "compile"
    )  # first __init__ raised

    eng.reset("n0")  # second construct succeeds
    eng.tick("n0", document, _ctx(0.0))
    assert document.uniform_values["u_x"] == 0.7  # unfrozen
    assert ("n0", "", "script.py") not in eng.errors  # stale sentinel cleared


def test_raw_runtime_throw_freezes_all_at_last_good(tmp_path: Path) -> None:
    # A raw update() exception freezes EVERY name the script drove last frame (one object = one
    # coherent state) AND records under the sentinel at the CORRECT user line. Falsifier: a name
    # advances past last-good, or the error isn't keyed on the sentinel at the user line.
    path = _write_script(
        tmp_path,
        _script(update_body="        return {'u_x': 0.3, 'u_y': 0.6}\n"),
    )
    document = _FakeDocument([_u("u_x"), _u("u_y")])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))  # establish last-good for both
    assert (
        document.uniform_values["u_x"] == 0.3 and document.uniform_values["u_y"] == 0.6
    )

    time.sleep(0.01)
    path.write_text(
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        x = 1\n"
        "        raise ValueError('boom')\n",  # line 4
        encoding="utf-8",
    )
    eng.reload("n0", tmp_path / "scripts", document)
    eng.tick("n0", document, _ctx(0.1))
    assert (
        document.uniform_values["u_x"] == 0.3 and document.uniform_values["u_y"] == 0.6
    )
    err = eng.errors[("n0", "", "script.py")]
    assert err.kind == "runtime" and "ValueError" in err.message
    assert err.line == 4  # the real user line, NOT -1


def test_runtime_error_records_deepest_user_line(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def _bad(self):\n"
        "        return 1.0 / 0.0\n"  # line 3 — the deepest user frame
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_x': self._bad()}\n",
    )
    document = _FakeDocument([_u("u_x")])
    document.uniform_values["u_x"] = 0.0
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    # Falsifier: the recorded line isn't the deepest user frame (3), e.g. -1.
    assert eng.errors[("n0", "", "script.py")].line == 3


def test_user_raised_builtin_exception_keeps_its_real_error(tmp_path: Path) -> None:
    # A user `raise ValueError(...)` surfaces as its real error (real builtins are in scope).
    _write_script(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        raise ValueError('nope')\n",
    )
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    err = eng.errors[("n0", "", "script.py")]
    assert err.kind == "runtime" and "ValueError" in err.message


def test_non_dict_return_is_clean_sentinel_error(tmp_path: Path) -> None:
    # A script that returns a non-dict is a behavior-level failure under the sentinel — not a crash.
    # Falsifier: tick raises, or the error keys per-uniform instead of the sentinel.
    _write_script(tmp_path, _script(update_body="        return 0.5\n"))  # a bare float
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))  # must NOT raise
    err = eng.errors[("n0", "", "script.py")]
    assert err.kind == "runtime"
    assert ("n0", "main", "u_x") not in eng.errors


def test_a_none_value_hands_the_uniform_back_to_the_user(tmp_path: Path) -> None:
    # The stub, the copilot's API block and 059's spec all tell the author that a key mapped to
    # None "stays MANUAL". It did not: None reached coercion, failed as "not a number" and still
    # counted as driven, so the panel showed a red row for the one gesture that means "leave this
    # one to me". The value is unchanged either way; what changes is that the state is now clean.
    # Falsifier: let None through to coercion again and the error row comes back.
    path = _write_script(tmp_path, _script(update_body="        return {'u_x': 0.4}\n"))
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    assert document.uniform_values["u_x"] == 0.4

    time.sleep(0.01)
    path.write_text(
        _script(update_body="        return {'u_x': None}\n"), encoding="utf-8"
    )
    eng.reload("n0", tmp_path / "scripts", document)
    eng.tick("n0", document, _ctx(0.1))
    assert document.uniform_values["u_x"] == 0.4  # the last value stands
    assert ("n0", "main", "u_x") not in eng.errors
    assert ("main", "u_x") not in eng.script_driven_uniforms("n0")


def test_per_key_shape_mismatch_freezes_only_that_key(tmp_path: Path) -> None:
    # A per-KEY coercion mismatch freezes ONLY that key; siblings still write. Falsifier: the
    # sibling u_a is also frozen, or the error keys on the sentinel instead of (document, u_b).
    _write_script(
        tmp_path,
        _script(
            update_body="        return {'u_a': 0.4, 'u_b': [1.0, 2.0, 3.0]}\n"
        ),  # vec3 into a scalar
    )
    document = _FakeDocument([_u("u_a"), _u("u_b")])
    document.uniform_values["u_b"] = 0.0
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    assert document.uniform_values["u_a"] == 0.4  # sibling wrote
    assert document.uniform_values["u_b"] == 0.0  # frozen
    assert eng.errors[("n0", "main", "u_b")].kind == "runtime"
    assert ("n0", "", "script.py") not in eng.errors  # NOT the sentinel


def test_array_wrong_length_freezes(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        _script(
            update_body="        return {'u_vals': [1.0, 2.0]}\n"
        ),  # 2 for float[4]
    )
    document = _FakeDocument([_u("u_vals", dim=1, n=4)])
    document.uniform_values["u_vals"] = (0.0, 0.0, 0.0, 0.0)
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    assert tuple(document.uniform_values["u_vals"]) == (0.0, 0.0, 0.0, 0.0)  # frozen
    assert eng.errors[("n0", "main", "u_vals")].kind == "runtime"


def test_nan_inf_freezes_and_records(tmp_path: Path) -> None:
    # A non-finite value is no longer written silently (a black-frame footgun that also poisons
    # last-good) — it freezes at last-good + records a runtime ScriptError. Falsifier: inf is
    # written, or no error recorded.
    path = _write_script(tmp_path, _script(update_body="        return {'u_x': 0.3}\n"))
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    assert document.uniform_values["u_x"] == 0.3

    time.sleep(0.01)
    path.write_text(
        _script(update_body="        return {'u_x': float('inf')}\n"), encoding="utf-8"
    )
    eng.reload("n0", tmp_path / "scripts", document)
    eng.tick("n0", document, _ctx(0.1))
    assert document.uniform_values["u_x"] == 0.3  # frozen at last-good, NOT inf
    err = eng.errors[("n0", "main", "u_x")]
    assert err.kind == "runtime" and "finite" in err.message


def test_array_accepts_nested_vec_rows(tmp_path: Path) -> None:
    # Feature 054: `Array` now AUTO-FLATTENS a list of Vec/tuple rows (the natural sim form) instead
    # of raising the old flatten-hint TypeError -- a `vec2[2]` built as `[(x0,y0),(x1,y1)]` drives
    # the uniform correctly. Falsifier: the uniform freezes / the value isn't the flattened rows.
    _write_script(
        tmp_path,
        _script(
            update_body="        return {'u_pts': [(0.0, 1.0), (2.0, 3.0)]}\n"
        ),  # nested rows -> auto-flattened
    )
    document = _FakeDocument([_u("u_pts", dim=2, n=2)])
    document.uniform_values["u_pts"] = [(0.0, 0.0), (0.0, 0.0)]
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    assert document.uniform_values["u_pts"] == [
        (0.0, 1.0),
        (2.0, 3.0),
    ]  # driven from the rows
    assert (
        "n0",
        "script.py",
    ) not in eng.errors  # no error -- nested rows are accepted now


# ---- soft (document,pass,name) errors: orphan/typo + sampler/block keys skipped, not driven ----


def test_a_key_no_pass_declares_is_skipped_silently(tmp_path: Path) -> None:
    # 079 D5: writing the script before the shader declares the uniform is a normal authoring step,
    # so a BARE key naming no active uniform on ANY pass is SKIPPED with NO error row — and still
    # claims no ownership in script_driven_uniforms. Falsifier: u_ghost written, or driven, or an
    # error row appears for it.
    _write_script(
        tmp_path,
        _script(update_body="        return {'u_a': 0.5, 'u_ghost': 0.9}\n"),
    )
    document = _FakeDocument([_u("u_a")])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    assert document.uniform_values["u_a"] == 0.5
    assert "u_ghost" not in document.uniform_values  # NOT written as None
    assert not [key for key in eng.errors if key[2] == "u_ghost"]
    assert ("", "u_ghost") not in eng.script_driven_uniforms(
        "n0"
    )  # claims no ownership


def test_a_bare_sampler_key_errors_like_a_pass_block_one(tmp_path: Path) -> None:
    # 079 D5 keeps sampler/block keys errors, and the BROADCAST path lost that: `_binds` answers
    # falsy for "not declared" and for "declared but a sampler" alike, so a bare sampler key went
    # silent while the same key inside a pass block errored. Falsifier: collapse the two again and
    # the bare form reports no error.
    _write_script(tmp_path, _script(update_body="        return {'u_tex': 1.0}\n"))
    document = _FakeDocument([_u("u_a"), _u("u_tex", gl_type=_GL_SAMPLER_2D)])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))

    error = eng.errors[("n0", "main", "u_tex")]
    assert "sampler" in error.message and error.pass_name == "main"
    assert ("main", "u_tex") not in eng.script_driven_uniforms("n0")


def test_a_pass_block_key_that_pass_does_not_declare_is_skipped_silently(
    tmp_path: Path,
) -> None:
    # 079 D5 through the block phase: the pass compiles and simply has no such uniform yet.
    # Falsifier: an error row, or the key written.
    _write_script(
        tmp_path,
        _script(update_body="        return {'main': {'u_a': 0.5, 'u_ghost': 0.9}}\n"),
    )
    document = _FakeDocument([_u("u_a")])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    assert document.uniform_values["u_a"] == 0.5
    assert "u_ghost" not in document.uniform_values
    assert not [key for key in eng.errors if key[2] == "u_ghost"]
    assert ("main", "u_ghost") not in eng.script_driven_uniforms("n0")


def test_sampler_key_records_soft_error_and_is_skipped(tmp_path: Path) -> None:
    # A key naming a sampler (non-scriptable) inside a PASS BLOCK records a soft (document, pass,
    # name) error naming the pass + is skipped (not driven). Falsifier: the sampler is driven, or no
    # soft error.
    _write_script(
        tmp_path,
        _script(update_body="        return {'u_a': 0.5, 'main': {'u_tex': 0.1}}\n"),
    )
    document = _FakeDocument([_u("u_a"), _u("u_tex", gl_type=_GL_SAMPLER_2D)])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    assert document.uniform_values["u_a"] == 0.5
    err = eng.errors[("n0", "main", "u_tex")]
    assert err.kind == "runtime" and "sampler" in err.message
    assert err.pass_name == "main"
    assert ("main", "u_tex") not in eng.script_driven_uniforms("n0")


def test_a_soft_error_persists_while_the_script_keeps_producing_it(
    tmp_path: Path,
) -> None:
    # The other half of the stale-clear: a row must survive every tick the script keeps earning
    # it. A first attempt at clearing the zombie row asked whether the key was REWRITTEN this
    # tick, which a re-recorded error is not — so the row oscillated on and off frame by frame.
    # Falsifier: clear on anything other than "the tick did not write it" and this alternates.
    _write_script(
        tmp_path, _script(update_body="        return {'blur': {'u_x': 1.0}}\n")
    )
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    for frame in range(4):
        eng.tick("n0", document, _ctx(frame / 60.0))
        assert ("n0", "", "blur") in eng.errors, f"the row vanished on tick {frame + 1}"


def test_a_pass_name_error_clears_when_the_key_becomes_a_bare_one(
    tmp_path: Path,
) -> None:
    # The pass-free slot `(document_id, "", key)` holds two namespaces: a bare uniform name no
    # pass declares, and a block naming a pass that does not exist. Rewriting `{"blur": {...}}`
    # as `{"blur": 1.0}` touched the pair from the BARE path, which suppressed the stale-clear,
    # while never writing the slot to overwrite it — so the "no pass named 'blur'" error stood
    # forever, on the strip and in the copilot's probe. Falsifier: clear on "touched this tick"
    # again rather than on "re-recorded this tick", and the error survives the rewrite.
    path = _write_script(
        tmp_path, _script(update_body="        return {'blur': {'u_x': 1.0}}\n")
    )
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    assert ("n0", "", "blur") in eng.errors

    time.sleep(0.01)
    path.write_text(
        _script(update_body="        return {'blur': 1.0}\n"), encoding="utf-8"
    )
    eng.reload("n0", tmp_path / "scripts", document)
    eng.tick("n0", document, _ctx(0.1))
    assert ("n0", "", "blur") not in eng.errors, (
        "the pass-name error outlived the pass block"
    )


def test_a_bad_key_error_clears_when_the_key_is_fixed(tmp_path: Path) -> None:
    # Once the bad key stops being returned (the user fixes the typo), its soft error is cleared on
    # the next tick. Falsifier: the zombie error persists.
    path = _write_script(
        tmp_path,
        _script(update_body="        return {'u_a': 0.5, 'main': {'u_tex': 0.1}}\n"),
    )
    document = _FakeDocument([_u("u_a"), _u("u_tex", gl_type=_GL_SAMPLER_2D)])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    assert ("n0", "main", "u_tex") in eng.errors

    time.sleep(0.01)
    path.write_text(
        _script(update_body="        return {'u_a': 0.5}\n"), encoding="utf-8"
    )
    eng.reload("n0", tmp_path / "scripts", document)
    eng.tick("n0", document, _ctx(0.1))
    assert ("n0", "main", "u_tex") not in eng.errors  # zombie cleared


# ---- engine-owned key dropped SILENTLY (no error, not driven) ----


def test_engine_owned_key_dropped_silently(tmp_path: Path) -> None:
    # A script naming an engine-owned uniform (u_time) SILENTLY drops the key — no false ownership
    # AND no soft error (the renderer owns that slot). Falsifier: u_time is driven, a soft error
    # appears, or the renderer's value is overwritten.
    _write_script(
        tmp_path,
        _script(update_body="        return {'u_a': 0.5, 'u_time': 9.0}\n"),
    )
    document = _FakeDocument([_u("u_a"), _u("u_time")])
    document.uniform_values["u_time"] = 1.23  # the renderer's value
    eng = ScriptEngine(engine_driven=frozenset({"u_time"}))
    eng.reload("n0", tmp_path / "scripts", document)
    eng.tick("n0", document, _ctx(0.0))
    assert document.uniform_values["u_a"] == 0.5
    assert ("main", "u_time") not in eng.script_driven_uniforms(
        "n0"
    )  # no false ownership
    assert ("n0", "main", "u_time") not in eng.errors  # silently dropped
    assert document.uniform_values["u_time"] == 1.23  # renderer value untouched


# ---- (path, mtime) cache ----


def test_cache_no_recompile_when_mtime_unchanged(tmp_path: Path) -> None:
    _write_script(tmp_path, _SCALAR)
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    first = eng._documents["n0"].behavior
    eng.reload("n0", tmp_path / "scripts", document)  # nothing changed
    # Falsifier: a fresh script object means an unnecessary recompile.
    assert eng._documents["n0"].behavior is first


def test_cache_recompiles_on_mtime_change(tmp_path: Path) -> None:
    path = _write_script(tmp_path, _SCALAR)
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    first = eng._documents["n0"].behavior
    time.sleep(0.01)
    path.write_text(_SCALAR + "        # changed\n", encoding="utf-8")
    eng.reload("n0", tmp_path / "scripts", document)
    assert eng._documents["n0"].behavior is not first  # recompiled


# ---- binding by existence (the file IS the binding; no active flag) ----


def test_script_binds_by_existence(tmp_path: Path) -> None:
    _write_script(tmp_path, _SCALAR)
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    # Falsifier: the script didn't bind despite script.py existing on disk.
    assert eng.has_script("n0")
    assert ("main", "u_x") in eng.script_driven_uniforms("n0")


def test_removed_script_drops_script(tmp_path: Path) -> None:
    path = _write_script(tmp_path, _SCALAR)
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    assert ("main", "u_x") in eng.script_driven_uniforms("n0")
    path.unlink()
    eng.reload("n0", tmp_path / "scripts", document)
    assert not eng.has_script("n0")
    assert ("main", "u_x") not in eng.script_driven_uniforms("n0")


def test_no_script_file_is_no_op_tick(tmp_path: Path) -> None:
    # A document dir with no script.py: reload binds nothing, tick is a no-op. Falsifier: tick raises
    # or invents a driven uniform.
    (tmp_path / "scripts").mkdir(exist_ok=True)
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    assert not eng.has_script("n0")
    eng.tick("n0", document, _ctx(0.0))  # must NOT raise
    assert eng.script_driven_uniforms("n0") == set()


# ---- scoped determinism ----


def test_t_pure_script_is_deterministic(tmp_path: Path) -> None:
    # A ctx.t-pure update is identical across dt (the scoped-determinism guarantee). Falsifier: a
    # different dt at the same t yields a different value.
    _write_script(
        tmp_path,
        "import math\n"
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_x': math.sin(ctx.t)}\n",
    )
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(1.234, dt=1 / 30))
    a = document.uniform_values["u_x"]
    eng.tick("n0", document, _ctx(1.234, dt=1 / 120))  # different dt, same t
    assert document.uniform_values["u_x"] == a


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
    _write_script(tmp_path, body)
    document_a = _FakeDocument([_u("u_x")])
    document_b = _FakeDocument([_u("u_x")])
    eng = ScriptEngine()
    eng.reload("a", tmp_path / "scripts", document_a)
    eng.reload("b", tmp_path / "scripts", document_b)

    for _i in range(2):  # two steps of dt=0.5 (variable-dt live path) over 1.0s
        eng.tick("a", document_a, EngineContext(t=0.0, dt=0.5, frame=0))
    eng.tick(
        "b", document_b, EngineContext(t=0.0, dt=1.0, frame=0)
    )  # one step of dt=1.0

    # 1*(1.5)^2 = 2.25 vs 1*(1+1) = 2.0 — divergent by design.
    assert document_a.uniform_values["u_x"] != document_b.uniform_values["u_x"]


# ---- script_status (sentinel + soft errors + driven_count) ----


def test_script_status_reflects_driven_count(tmp_path: Path) -> None:
    _write_script(tmp_path, _TWO_INTEGRATOR)
    document = _FakeDocument([_u("u_x"), _u("u_y")])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0, dt=1.0, frame=0))
    status = eng.script_status("n0")
    assert status is not None
    # Falsifier: a wrong driven_count, or a phantom sentinel/soft error on a clean script.
    assert status.driven_count == 2
    assert status.sentinel_error is None
    assert status.soft_errors == []


def test_script_status_reflects_sentinel_and_soft_errors(tmp_path: Path) -> None:
    # A script with a sampler key: driven_count counts only the real driven uniform; the sampler
    # surfaces in soft_errors. Falsifier: the bad key inflates driven_count or is missing from soft.
    _write_script(
        tmp_path,
        _script(update_body="        return {'u_a': 0.5, 'main': {'u_tex': 0.1}}\n"),
    )
    document = _FakeDocument([_u("u_a"), _u("u_tex", gl_type=_GL_SAMPLER_2D)])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    status = eng.script_status("n0")
    assert status is not None
    assert status.driven_count == 1  # only u_a
    assert status.sentinel_error is None
    assert [(p, name) for p, name, _ in status.soft_errors] == [("main", "u_tex")]


def test_script_status_none_without_script(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir(exist_ok=True)
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    assert eng.script_status("n0") is None


# ---- drop_document clears state ----


def test_drop_document_clears_state_and_errors(tmp_path: Path) -> None:
    _write_script(
        tmp_path, "class Behavior(ScriptBehavior):\n    pass\n"
    )  # compile err
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    assert ("n0", "", "script.py") in eng.errors
    eng.drop_document("n0")
    assert ("n0", "", "script.py") not in eng.errors
    assert eng.script_driven_uniforms("n0") == set()
    assert not eng.has_script("n0")


# ---- reload robustness (read_text must not crash the frame loop) ----


def test_non_utf8_file_does_not_crash_reload(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "script.py").write_bytes(b"\xff\xfe not utf-8")
    document = _FakeDocument([_u("u_x")])
    eng = ScriptEngine()
    eng.reload("n0", scripts_dir, document)  # must NOT raise
    assert ("main", "u_x") not in eng.script_driven_uniforms("n0")


def test_unreadable_rewrite_mid_edit_keeps_cached_script(tmp_path: Path) -> None:
    # A reload that races a half-saved / non-UTF8 rewrite at a changed mtime keeps the prior script
    # rather than crashing the frame loop. Falsifier: reload raises, or the cached script is dropped.
    path = _write_script(tmp_path, _SCALAR)
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    first = eng._documents["n0"].behavior
    assert first is not None
    time.sleep(0.01)
    path.write_bytes(b"\xff\xfe still here but unreadable")
    eng.reload("n0", tmp_path / "scripts", document)  # must NOT raise
    assert eng._documents["n0"].behavior is first  # cached script kept


# ---- namespace: imports, super, builtins (the engine's own idioms resolve) ----


def test_import_math_works(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        "import math\n"
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_x': math.cos(0.0)}\n",  # 1.0
    )
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    assert ("n0", "main", "u_x") not in eng.errors
    assert abs(document.uniform_values["u_x"] - 1.0) < 1e-9


def test_explicit_import_line_resolves(tmp_path: Path) -> None:
    # 048 decision 8: the stub emits a real `from shaderbox.scripting import ...`; it must RESOLVE
    # inside the exec'd script. Falsifier: the import raises an opaque compile-freeze.
    _write_script(
        tmp_path,
        "from shaderbox.scripting import ScriptBehavior, Ctx\n"
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_off': [0.1, 0.2]}\n",
    )
    document = _FakeDocument([_u("u_off", dim=2)])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    assert ("n0", "", "script.py") not in eng.errors
    assert document.uniform_values["u_off"] == (0.1, 0.2)


def test_super_and_containers_resolve(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def __init__(self) -> None:\n"
        "        super().__init__()\n"
        "        self.buf = []\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        self.buf.append(ctx.t)\n"
        "        return {'u_x': sum(self.buf) / len(self.buf)}\n",
    )
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(2.0))
    assert ("n0", "", "script.py") not in eng.errors  # super + list + sum resolved
    assert document.uniform_values["u_x"] == 2.0


def test_chr_ord_available_for_codepoint_text(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_t': chr(65) + chr(66)}\n",  # 'AB'
    )
    document = _FakeDocument([_u("u_t", dim=1, n=8, gl_type=_GL_UNSIGNED_INT)])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    val = document.uniform_values["u_t"]
    assert val[0] == ord("A") and val[1] == ord("B")
    assert ("n0", "main", "u_t") not in eng.errors


# ---- is_scriptable + script_stub_for ----


def test_is_scriptable_gate() -> None:
    assert is_scriptable(_u("u_x"))
    assert is_scriptable(_u("u_v", dim=3))
    assert not is_scriptable(_u("u_tex", gl_type=_GL_SAMPLER_2D))
    assert not is_scriptable(object())  # no shape attrs


def test_script_stub_compiles_runs_and_drives_nothing_by_default(
    tmp_path: Path,
) -> None:
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
    body = script_stub_for({"main": uniforms})
    _write_script(tmp_path, body)
    document = _FakeDocument(uniforms)
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    assert not any(k[0] == "n0" for k in eng.errors), "fresh stub errored"
    assert eng.script_driven_uniforms("n0") == set()  # empty-dict default
    assert document.uniform_values == {}


def test_the_stub_imports_the_whole_scripting_surface(tmp_path: Path) -> None:
    # The importable surface is two names, whatever the document declares: a script returns plain
    # Python, so there are no value types to import. Falsifier: emit a value type in the import
    # line and it names something a script cannot import.
    body = script_stub_for({"main": [_u("u_v3", dim=3), _u("u_a", n=4)]})
    import_line = next(
        line for line in body.splitlines() if "from shaderbox.scripting import" in line
    )
    assert import_line == "from shaderbox.scripting import ScriptBehavior, Ctx"


# ---- ctx is frozen ----


def test_stopped_uniform_write_skipped_but_still_driven_and_sibling_advances(
    tmp_path: Path,
) -> None:
    # The core stopped-skip canary (decision 4/5): tick with stopped={'u_x'} ->
    #   - u_x is NOT written (stays the pre-tick manual value),
    #   - u_x IS still in script_driven_uniforms (last_driven — keeps its PLAY button),
    #   - the script's OTHER driven uniform u_y IS written (the script ticked, it just skipped one write).
    # Falsifier: u_x changed, OR u_x absent from driven, OR u_y not advanced.
    _write_script(tmp_path, _TWO_INTEGRATOR)
    document = _FakeDocument([_u("u_x"), _u("u_y")])
    document.uniform_values["u_x"] = 0.42  # the user's frozen manual value
    eng = _engine(tmp_path, document)
    eng.tick(
        "n0",
        document,
        _ctx(0.0, dt=1.0, frame=0),
        stopped=frozenset({StoppedKey(pass_name="main", name="u_x")}),
    )
    assert (
        document.uniform_values["u_x"] == 0.42
    )  # frozen at the manual value, NOT self.v (==1.0)
    assert ("main", "u_x") in eng.script_driven_uniforms("n0")  # still driven
    assert (
        document.uniform_values["u_y"] == 2.0
    )  # sibling advanced (self.v*2 with self.v==1.0)


def test_stopped_script_keeps_ticking_then_resumes_advanced(tmp_path: Path) -> None:
    # Decision 5: the script keeps TICKING while a uniform is stopped, so on resume the value jumps
    # to the value the integrator reached WHILE stopped — NOT the value at stop time. Falsifier: the
    # resumed value equals the stop-instant value (the script stopped advancing).
    _write_script(tmp_path, _INTEGRATOR)
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0, dt=1.0, frame=0))
    assert document.uniform_values["u_x"] == 1.0  # value at the moment we stop

    document.uniform_values["u_x"] = 99.0  # the user's manual value while stopped
    for i in range(1, 4):  # three stopped ticks: self.v advances 2,3,4 but no write
        eng.tick(
            "n0",
            document,
            _ctx(0.0, dt=1.0, frame=i),
            stopped=frozenset({StoppedKey(pass_name="main", name="u_x")}),
        )
    assert document.uniform_values["u_x"] == 99.0  # never written while stopped

    eng.tick("n0", document, _ctx(0.0, dt=1.0, frame=4))  # resume (not stopped)
    # self.v advanced to 5 across the stop window (1 initial + 4 ticks of dt=1.0).
    assert (
        document.uniform_values["u_x"] == 5.0
    )  # advanced state, NOT the 1.0 stop value


def test_export_always_plays_a_live_stopped_uniform(tmp_path: Path) -> None:
    # Decision 5 + export-isolation: tick_export forwards NO stopped set, so an export writes the
    # SCRIPT value even for a uniform stopped in the live preview. Falsifier: the export freezes the
    # stopped manual value (tick_export leaked the live stopped set).
    _write_script(tmp_path, _INTEGRATOR)
    document = _FakeDocument([_u("u_x")])
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0, dt=1.0, frame=0))
    document.uniform_values["u_x"] = 99.0
    eng.tick(
        "n0",
        document,
        _ctx(0.0, dt=1.0, frame=1),
        stopped=frozenset({StoppedKey(pass_name="main", name="u_x")}),
    )
    assert document.uniform_values["u_x"] == 99.0  # frozen live

    fresh = eng.fresh_behavior_for("n0")
    assert fresh is not None
    export_document = _FakeDocument([_u("u_x")])
    eng.tick_export("n0", export_document, _ctx(0.0, dt=1.0, frame=0), fresh)
    # The export plays: a fresh instance ticks once -> self.v == 1.0, WRITTEN (not the frozen 99.0).
    assert export_document.uniform_values["u_x"] == 1.0


# ---- 069 D3: routing across passes ----


def _two_pass() -> _FakeDocument:
    # `paint` declares u_a AND u_b; `composite` declares u_a only — the shape every routing case
    # below needs: a name on both passes, a name on one, and a name on neither.
    return _FakeDocument(
        {
            "paint": [_u("u_a"), _u("u_b")],
            "composite": [_u("u_a")],
        }
    )


@pytest.mark.parametrize(
    ("returned", "written", "error_key", "error_fragment"),
    [
        # bare key, BOTH passes declare it -> both written, no error
        ("{'u_a': 1.0}", {"paint": {"u_a": 1.0}, "composite": {"u_a": 1.0}}, None, ""),
        # bare key, ONE pass declares it -> that pass only; the sibling is untouched
        ("{'u_b': 1.0}", {"paint": {"u_b": 1.0}, "composite": {}}, None, ""),
        # bare key, NO pass declares it -> nothing written, NO error (079 D5: a normal
        # authoring step, the shader has yet to declare it)
        ("{'u_z': 1.0}", {"paint": {}, "composite": {}}, None, ""),
        # pass block, declared there -> that pass only
        (
            "{'paint': {'u_a': 1.0}}",
            {"paint": {"u_a": 1.0}, "composite": {}},
            None,
            "",
        ),
        # pass block, NOT declared on that pass -> nothing written, NO error (079 D5)
        ("{'composite': {'u_b': 1.0}}", {"paint": {}, "composite": {}}, None, ""),
        # pass block naming NO pass -> nothing written, an error listing the real passes
        (
            "{'nope': {'u_a': 1.0}}",
            {"paint": {}, "composite": {}},
            ("n0", "", "nope"),
            "no pass named 'nope' in this document (passes: composite, paint)",
        ),
    ],
)
def test_the_routing_table(
    tmp_path: Path,
    returned: str,
    written: dict[str, dict[str, float]],
    error_key: tuple[str, str, str] | None,
    error_fragment: str,
) -> None:
    # The (bare, nested) x (declared in one pass, in two, in none) matrix (069 D3; the undeclared
    # rows carry 079 D5's silent skip). Falsifier: route every key to one pass (the pre-069
    # output-only behaviour) — rows 1/2/4 go red on the write side, rows 3/5 on the no-error side
    # and row 6 on the error side, so no single wrong implementation passes the table.
    _write_script(tmp_path, _script(update_body=f"        return {returned}\n"))
    document = _two_pass()
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    for pass_name, expected in written.items():
        assert document.passes[pass_name].uniform_values == expected
    if error_key is None:
        assert not any(k[0] == "n0" for k in eng.errors)
    else:
        assert error_fragment in eng.errors[error_key].message


def test_a_pass_block_beats_a_broadcast_on_that_pass(tmp_path: Path) -> None:
    # Specific over general, and INDEPENDENT of the author's insertion order — that is what the
    # two-phase (broadcasts, then blocks) shape buys. Falsifier: collapse the phases into one loop
    # over raw.items(); the second half goes red because insertion order then decides the winner.
    for returned in (
        "{'u_a': 1.0, 'paint': {'u_a': 2.0}}",
        "{'paint': {'u_a': 2.0}, 'u_a': 1.0}",
    ):
        _write_script(tmp_path, _script(update_body=f"        return {returned}\n"))
        document = _two_pass()
        eng = _engine(tmp_path, document)
        eng.tick("n0", document, _ctx(0.0))
        assert document.passes["paint"].uniform_values["u_a"] == 2.0
        assert document.passes["composite"].uniform_values["u_a"] == 1.0


def test_an_unknown_pass_names_the_pass_and_lists_the_real_ones(tmp_path: Path) -> None:
    # One key's failure must not cost its siblings (the freeze-granularity rule). Falsifier: raise on
    # an unknown pass instead of recording — the sibling half goes red (nothing is driven) and the
    # message half with it.
    _write_script(
        tmp_path,
        _script(update_body="        return {'nope': {'u_a': 1.0}, 'u_b': 3.0}\n"),
    )
    document = _two_pass()
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    message = eng.errors[("n0", "", "nope")].message
    assert "'nope'" in message and "composite" in message and "paint" in message
    assert (
        document.passes["paint"].uniform_values["u_b"] == 3.0
    )  # the sibling still drove


def test_a_not_yet_compiled_pass_is_held_not_errored(tmp_path: Path) -> None:
    # A pass that has NEVER attempted a compile is skipped for the tick with NO error — else the
    # first frame of a multi-pass document sprays orphan errors that clear a frame later (066 D1 is
    # what forbids compiling one from inside the tick). Falsifier: treat a not-ready pass as absent;
    # the no-error assertion goes red.
    _write_script(
        tmp_path, _script(update_body="        return {'paint': {'u_a': 1.0}}\n")
    )
    document = _FakeDocument({"paint": [_u("u_a")]})
    document.passes["paint"].script_ready = False
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    assert not any(k[0] == "n0" for k in eng.errors)
    assert document.passes["paint"].uniform_values == {}

    document.passes["paint"].script_ready = True
    eng.tick("n0", document, _ctx(0.1))
    assert document.passes["paint"].uniform_values["u_a"] == 1.0
    assert not any(k[0] == "n0" for k in eng.errors)


def test_coerce_one_rejects_a_dict(tmp_path: Path) -> None:
    # The invariant D3's value-type dispatch rests on, made a CHECKED property of the coercion atom
    # rather than an inspection of normalize_output. Falsifier: delete the isinstance(value, dict)
    # branch — the direct call goes red immediately.
    with pytest.raises(_RuntimeScriptError) as excinfo:
        coerce_one({"x": 1}, _u("u_a"), "u_a")
    assert "PASS BLOCK" in excinfo.value.error.message

    # And through the engine: one level too deep freezes that key at last-good and records the
    # grammar error, rather than an unhelpful shape hint about a float.
    path = _write_script(tmp_path, _script(update_body="        return {'u_a': 1.0}\n"))
    document = _FakeDocument({"paint": [_u("u_a")]})
    eng = _engine(tmp_path, document)
    eng.tick("n0", document, _ctx(0.0))
    time.sleep(0.01)
    path.write_text(
        _script(update_body="        return {'paint': {'u_a': {'x': 0.5}}}\n"),
        encoding="utf-8",
    )
    eng.reload("n0", tmp_path / "scripts", document)
    eng.tick("n0", document, _ctx(0.1))
    assert "PASS BLOCK" in eng.errors[("n0", "paint", "u_a")].message
    assert document.passes["paint"].uniform_values["u_a"] == 1.0  # frozen at last-good


def test_the_stub_has_one_block_per_pass() -> None:
    # One commented block per pass, each listing that pass's own scriptable uniforms, plus the
    # bare-key rule. Falsifier: emit only the first pass's block — the composite/u_b assertions go
    # red; the ast.parse half falsifies emitting comment text that is not valid Python.
    body = script_stub_for(
        {
            "paint": [_u("u_a"), _u("u_b")],
            "composite": [_u("u_a")],
            "empty": [],
        }
    )
    # DOUBLE quotes, matching the bare-key example in the same stub and the design note's snippet.
    # Falsifier: emit `!r` (single quotes) — the stub would be quoted two ways in one block.
    assert '"paint": {' in body
    assert '"composite": {' in body
    assert '"empty": {' in body
    assert "(no scriptable uniforms)" in body
    assert "EVERY pass declaring it" in body
    paint_block = body.split('"paint": {')[1].split('"composite": {')[0]
    composite_block = body.split('"composite": {')[1].split('"empty": {')[0]
    assert "u_b" in paint_block and "u_b" not in composite_block

    ast.parse(body)  # the comments are comments, not broken code
    namespace: dict[str, object] = {}
    exec(compile(body, "stub", "exec"), namespace)
    behavior_class: Any = namespace["Behavior"]
    behavior = behavior_class()
    assert behavior.update(_ctx(0.0)) == {}


def test_a_document_with_no_scriptable_uniforms_keeps_the_bare_return() -> None:
    # Falsifier: emit a block per pass unconditionally — an empty document would grow comment blocks
    # naming nothing.
    body = script_stub_for({"paint": [], "composite": []})
    assert "return {}" in body


def test_a_bad_key_reaches_the_strip_and_not_the_console(tmp_path: Path) -> None:
    # #29's own last sentence: a bad key becomes a VISIBLE strip error, not a console line. The sink
    # must be a real loguru one — this repo logs through loguru, which does not propagate into the
    # stdlib tree pytest's caplog reads, so a caplog assertion would pass vacuously with the
    # logger.warning still in place. Falsifier: restore the logger.warning; the first half goes red.
    records: list[str] = []
    handle = logger.add(lambda m: records.append(str(m)), level="WARNING")
    try:
        _write_script(
            tmp_path,
            _script(
                update_body="        return {'u_a': 0.5, 'main': {'u_tex': 0.1}}\n"
            ),
        )
        document = _FakeDocument([_u("u_a"), _u("u_tex", gl_type=_GL_SAMPLER_2D)])
        eng = _engine(tmp_path, document)
        eng.tick("n0", document, _ctx(0.0))
    finally:
        logger.remove(handle)
    assert not [r for r in records if "shaderbox.scripting.engine" in r]
    status = eng.script_status("n0")
    assert status is not None
    assert [(p, name) for p, name, _ in status.soft_errors] == [("main", "u_tex")]


def test_a_stopped_pair_freezes_only_that_pass(tmp_path: Path) -> None:
    # The defect the pass-qualified stop set exists to prevent. Falsifier: key the stopped set by
    # NAME — paint freezes too and the first assertion goes red.
    _write_script(tmp_path, _INTEGRATOR.replace("u_x", "u_a"))
    document = _two_pass()
    eng = _engine(tmp_path, document)
    stopped = frozenset({StoppedKey(pass_name="composite", name="u_a")})
    eng.tick("n0", document, _ctx(0.0, dt=1.0, frame=0), stopped=stopped)
    document.passes["composite"].uniform_values["u_a"] = 99.0
    eng.tick("n0", document, _ctx(1.0, dt=1.0, frame=1), stopped=stopped)

    assert document.passes["paint"].uniform_values["u_a"] == 2.0  # advanced
    assert (
        document.passes["composite"].uniform_values["u_a"] == 99.0
    )  # manual value held
    # BOTH keep their play/stop button: a stopped pair still counts as driven.
    driven = eng.script_driven_uniforms("n0")
    assert ("paint", "u_a") in driven and ("composite", "u_a") in driven


def test_a_broadcast_is_held_for_a_not_yet_compiled_pass(tmp_path: Path) -> None:
    # The hold is the BROADCAST's too, not only the pass block's (069 round 3). A never-rendered
    # two-pass document must produce NO strip row on frame 0 and drive on frame 1 — "absent because
    # it has not compiled yet" and "absent because no pass declares it" are different answers.
    # Falsifier: drop the not_ready check in the broadcast phase; frame 0 records
    # `no pass declares 'u_a'` and the strip shows the orphan the hold exists to prevent.
    _write_script(tmp_path, _script(update_body="        return {'u_a': 1.0}\n"))
    document = _FakeDocument({"paint": [_u("u_a")], "composite": [_u("u_a")]})
    for render_pass in document.passes.values():
        render_pass.script_ready = False
    eng = _engine(tmp_path, document)

    eng.tick("n0", document, _ctx(0.0))
    assert not any(k[0] == "n0" for k in eng.errors)
    status = eng.script_status("n0")
    assert status is not None and status.soft_errors == []
    assert eng.script_driven_uniforms("n0") == set()
    assert document.passes["paint"].uniform_values == {}

    for render_pass in document.passes.values():
        render_pass.script_ready = True
    eng.tick("n0", document, _ctx(0.1))
    assert eng.script_driven_uniforms("n0") == {("paint", "u_a"), ("composite", "u_a")}
    assert not any(k[0] == "n0" for k in eng.errors)


def test_an_undeclared_key_stays_silent_whatever_the_passes_are_doing(
    tmp_path: Path,
) -> None:
    # 079 D5 across every reason a pass can fail to declare the key: a compile that failed (ready
    # but declaring nothing), and a pass that compiles and simply does not have it. Neither is the
    # script's defect — a broken pass surfaces its own compile error on its shader tab. Falsifier:
    # restore either orphan row; both halves go red.
    _write_script(tmp_path, _script(update_body="        return {'u_wave': 1.0}\n"))
    broken = _FakeDocument({"main": []})  # ready, declaring nothing = a failed compile
    eng = _engine(tmp_path, broken)
    eng.tick("n0", broken, _ctx(0.0))
    assert not any(k[0] == "n0" for k in eng.errors)

    healthy = _FakeDocument({"paint": [_u("u_a")], "composite": [_u("u_a")]})
    eng2 = _engine(tmp_path, healthy)
    eng2.tick("n0", healthy, _ctx(0.0))
    assert not any(k[0] == "n0" for k in eng2.errors)
    status = eng2.script_status("n0")
    assert status is not None and status.soft_errors == []


# ---- Pass.script_ready's truth table, GL-free ----


def _ready(program: object | None, error_raw: str) -> bool:
    # `Pass.script_ready` bound onto a light stub (the __get__ idiom the backend tests already use):
    # the property reads only these two attributes, so no GL context is needed. A GL-gated pin would
    # leave the expression unverified on a display-less box, which is where the inverted form shipped
    # green during the spec's own round 2.
    stub = types.SimpleNamespace(
        program=program, compile_unit=types.SimpleNamespace(error_raw=error_raw)
    )
    return Pass.script_ready.fget(stub)


def test_script_ready_matches_its_truth_table() -> None:
    # The wave spec's three rows, asserted on the expression itself. `script_ready` is the NEGATION
    # of the guard `get_active_uniforms` tests before compiling, and writing that guard verbatim
    # inverts the member — which is exactly what round 2 caught in the spec.
    # Falsifier: invert to `program is None and not error_raw`; all three rows go red.
    assert (
        _ready(None, "") is False
    )  # never attempted -> HELD, no compile from the tick
    assert (
        _ready(None, "boom") is True
    )  # compile FAILED -> ready-but-empty, orphan path
    assert _ready(object(), "") is True  # compiled -> routes normally


def test_script_ready_never_compiles() -> None:
    # It is a pure read: a stub whose attributes raise if anything tries to build a program proves
    # the property triggers no compile, which is what makes 066 D1 hold by construction rather than
    # by an engine-side exception handler. Falsifier: have the property call get_active_uniforms.
    calls: list[str] = []

    class _Probe:
        program = None
        compile_unit = types.SimpleNamespace(error_raw="")

        def compile(self) -> None:
            calls.append("compile")

        def get_active_uniforms(self) -> list[object]:
            calls.append("get_active_uniforms")
            return []

    assert Pass.script_ready.fget(_Probe()) is False
    assert calls == []


# ---- the script's import surface: `shaderbox.scripting`'s user types, and nothing else ----


def _run_script(tmp_path: Path, body: str) -> ScriptError | None:
    _write_script(tmp_path, body)
    document = _FakeDocument([_u("u_a")])
    eng = _engine(tmp_path, document)
    return eng.errors.get(("n0", "", "script.py"))


def test_a_script_imports_the_stdlib_and_the_scripting_types(tmp_path: Path) -> None:
    # The gate narrows ONE package path; everything else a script could import before still
    # imports. Falsifier: gate the root builtins instead and `import math` goes red.
    error = _run_script(
        tmp_path,
        "import math\n"
        "import json\n"
        "from shaderbox.scripting import ScriptBehavior, Ctx, MouseState\n"
        "\n"
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_a': math.sin(ctx.t)}\n",
    )
    assert error is None, error


def test_a_script_cannot_import_the_app_or_its_modules(tmp_path: Path) -> None:
    # `shaderbox.app`, `shaderbox.core` and `ProjectSession` were all reachable from a script.
    # They are the app's machinery, not the scripting interface. Falsifier: drop the import gate
    # and every one of these compiles clean.
    for module in (
        "shaderbox",
        "shaderbox.app",
        "shaderbox.core",
        "shaderbox.project_session",
        "shaderbox.scripting.engine",
    ):
        error = _run_script(
            tmp_path,
            f"import {module}\n"
            "from shaderbox.scripting import ScriptBehavior, Ctx\n"
            "\n"
            "class Behavior(ScriptBehavior):\n"
            "    def update(self, ctx: Ctx) -> dict:\n"
            "        return {}\n",
        )
        assert error is not None, f"a script imported {module}"
        assert "shaderbox.scripting" in error.message, error.message


def test_the_scripting_package_offers_a_script_only_its_user_types(
    tmp_path: Path,
) -> None:
    # The package holds the engine, the probe and the stub generator beside the user's types.
    # A script gets the types. Falsifier: gate the module path alone and `ScriptEngine` imports.
    for name in ("ScriptEngine", "ScriptProbe", "PythonBehavior", "script_stub_for"):
        error = _run_script(
            tmp_path,
            f"from shaderbox.scripting import ScriptBehavior, Ctx, {name}\n"
            "\n"
            "class Behavior(ScriptBehavior):\n"
            "    def update(self, ctx: Ctx) -> dict:\n"
            "        return {}\n",
        )
        assert error is not None, f"a script imported {name}"
        assert name in error.message, error.message


def test_importing_the_module_itself_says_to_import_the_names(tmp_path: Path) -> None:
    # `import shaderbox.scripting` would bind the whole module, engine included. Falsifier: allow
    # a bare import and the module object is a script's back door to everything in it.
    error = _run_script(
        tmp_path,
        "import shaderbox.scripting\n"
        "from shaderbox.scripting import ScriptBehavior, Ctx\n"
        "\n"
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {}\n",
    )
    assert error is not None
    assert "import the names" in error.message, error.message
