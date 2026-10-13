"""A script drives a sampler2D (feature 110).

The gates of `ai_docs/features/110_script_textures/01_spec.md ## Verification`, numbered as
there. Fixtures render once before the first tick, so the pass is compiled and no key is
`held_uncompiled`. Every field differs from anything else the sampler could bind: the wired
pass draws `_SRC_COLOUR`, an unfilled sampler reads black, and fields hold no black texel
unless a gate says so.
"""

import json
import shutil
import subprocess
import types
from pathlib import Path
from typing import Any, get_args

import moderngl
import numpy as np
import pytest
from imgui_bundle import imgui
from OpenGL import GL
from PIL import Image as PILImage

from shaderbox.constants import RESOURCES_DIR
from shaderbox.copilot.backend import (
    _format_uniforms,
    _motion_verdict,
    _texture_facts_text,
)
from shaderbox.core import Pass
from shaderbox.document import Document
from shaderbox.graph_canvas.adapter import _PIN_BY_KIND, flat_view
from shaderbox.intel.script import glsl_type_of_value
from shaderbox.media import MediaDetails, MediaWithTexture
from shaderbox.pass_graph import NoSource, PassSource, PortKind, node_ports
from shaderbox.paths import DOCUMENT_SCRIPT_BASENAME, pass_name_of
from shaderbox.script_texture import MAX_SCRIPT_TEXTURE_EXTENT, REFUSED_TEXTURE
from shaderbox.scripting import ScriptContext, ScriptEngine, StoppedKey
from shaderbox.scripting.engine import is_script_texture, is_scriptable
from shaderbox.ui_models import UIUniform, load_document_from_dir
from shaderbox.uniform_coerce import VALUE_GL_TYPES, coerce_uniform_value
from shaderbox.widgets import uniform as uniform_widget
from shaderbox.widgets.graph_state import ports_of
from shaderbox.widgets.pass_list import _reads

_SRC_COLOUR = (0.25, 0.5, 0.75)

_SRC = """#version 460 core
out vec4 fs_color;
void main() { fs_color = vec4(0.25, 0.5, 0.75, 1.0); }
"""

_HEAD = (
    "import numpy as np\n"
    "from shaderbox.scripting import ScriptBehavior, ScriptContext, native\n\n"
)


def _view(sampler: str) -> str:
    # Each texel of the canvas reads the field texel at the same place: a canvas the size of
    # the field reads every texel, row 0 at the bottom on both sides.
    return (
        "#version 460 core\n"
        f"uniform sampler2D {sampler};\n"
        "out vec4 fs_color;\n"
        f"void main() {{ fs_color = texelFetch({sampler}, ivec2(gl_FragCoord.xy), 0); }}\n"
    )


def _write(
    root: Path,
    update_body: str,
    *,
    sampler: str = "u_field",
    with_src: bool = False,
    init_body: str = "",
) -> Path:
    (root / "passes").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "passes" / "view.frag.glsl").write_text(_view(sampler))
    passes: dict[str, Any] = {"view": {"target": {"dtype": "f4"}}}
    if with_src:
        (root / "passes" / "src.frag.glsl").write_text(_SRC)
        passes["src"] = {"target": {"dtype": "f4"}}
    init = f"    def __init__(self) -> None:\n{init_body}" if init_body else ""
    (root / "scripts" / "script.py").write_text(
        _HEAD
        + "class Behavior(ScriptBehavior):\n"
        + init
        + "    def update(self, context: ScriptContext) -> dict:\n"
        + update_body
    )
    (root / "graph.json").write_text(
        json.dumps({"version": 2, "output": "view", "passes": passes})
    )
    (root / "document.json").write_text('{"uniforms": {}}')
    return root


def _document(
    tmp_path: Path,
    gl_ctx: moderngl.Context,
    update_body: str,
    *,
    size: tuple[int, int] = (8, 8),
    **kwargs: Any,
) -> tuple[Document, ScriptEngine]:
    root = _write(tmp_path / "doc", update_body, **kwargs)
    document, _ = Document.load_from_dir(root, gl=gl_ctx, canvas_size=size)
    document.render()
    engine = ScriptEngine()
    engine.reload("doc", root / "scripts")
    return document, engine


def _tick(
    engine: ScriptEngine,
    document: Document,
    frame: int,
    stopped: frozenset[StoppedKey] = frozenset(),
) -> None:
    engine.tick(
        "doc", document, ScriptContext(t=frame / 60, dt=1 / 60, frame=frame), stopped
    )


def _drawn(document: Document, width: int, height: int) -> np.ndarray:
    document.render()
    canvas = document.passes["view"].canvas
    data = canvas.fbo.read(viewport=(0, 0, width, height), components=3, dtype="f4")
    return np.frombuffer(data, dtype="f4").reshape(height, width, 3)


def _field_expr(height: int, width: int, channels: int, scale: float = 100.0) -> str:
    # Every texel distinct and non-zero, so a transpose, a flip or black reads differently.
    count = height * width * channels
    return (
        f"((np.arange({count}, dtype=np.float32) + 1.0) / {scale})"
        f".reshape({height}, {width}, {channels})"
    )


def _field(height: int, width: int, channels: int, scale: float = 100.0) -> np.ndarray:
    count = height * width * channels
    return ((np.arange(count, dtype=np.float32) + 1.0) / np.float32(scale)).reshape(
        height, width, channels
    )


def _is_src(pixels: np.ndarray) -> bool:
    return bool(np.allclose(pixels, _SRC_COLOUR))


# ---- 1. round trip, both key forms -------------------------------------------------------


@pytest.mark.parametrize("form", ["block", "bare"])
def test_a_field_reaches_the_pass_exactly(
    gl_ctx: moderngl.Context, tmp_path: Path, form: str
) -> None:
    returned = "{'view': {'u_field': f}}" if form == "block" else "{'u_field': f}"
    document, engine = _document(
        tmp_path,
        gl_ctx,
        f"        f = {_field_expr(5, 7, 3)}\n        return {returned}\n",
        size=(7, 5),
    )
    _tick(engine, document, 0)
    assert np.array_equal(_drawn(document, 7, 5), _field(5, 7, 3))


# ---- 2. mixed broadcast ------------------------------------------------------------------


def test_a_bare_key_drives_each_pass_by_its_own_type(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    root = _write(
        tmp_path / "doc",
        f"        return {{'u_x': {_field_expr(4, 4, 3)}}}\n",
        sampler="u_x",
    )
    (root / "passes" / "num.frag.glsl").write_text(
        "#version 460 core\nuniform float u_x;\nout vec4 fs_color;\n"
        "void main() { fs_color = vec4(u_x); }\n"
    )
    graph = json.loads((root / "graph.json").read_text())
    graph["passes"]["num"] = {}
    (root / "graph.json").write_text(json.dumps(graph))
    document, _ = Document.load_from_dir(root, gl=gl_ctx, canvas_size=(4, 4))
    document.render()
    document.passes[
        "num"
    ].get_active_uniforms()  # off the output chain: compile it by hand
    engine = ScriptEngine()
    engine.reload("doc", root / "scripts")
    _tick(engine, document, 0)
    assert np.array_equal(_drawn(document, 4, 4), _field(4, 4, 3))
    assert ("doc", "num", "u_x") in engine.errors
    assert ("doc", "view", "u_x") not in engine.errors


# ---- 3. normalized u1 --------------------------------------------------------------------


def test_a_uint8_field_reads_normalized(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _document(
        tmp_path,
        gl_ctx,
        "        return {'view': {'u_field': np.array([[0, 128, 255]], np.uint8)}}\n",
        size=(3, 1),
    )
    _tick(engine, document, 0)
    red = _drawn(document, 3, 1)[0, :, 0]
    assert np.allclose(red, [0.0, 128 / 255, 1.0])


# ---- 4. the script outranks inputs -------------------------------------------------------


def test_a_field_outranks_the_texture_the_wiring_binds(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _document(
        tmp_path,
        gl_ctx,
        f"        return {{'view': {{'u_field': {_field_expr(4, 4, 3)}}}}}\n",
        size=(4, 4),
    )
    _tick(engine, document, 0)
    view = document.passes["view"]
    other = gl_ctx.texture(
        (4, 4), 3, data=np.full((4, 4, 3), 0.9, "f4").tobytes(), dtype="f4"
    )
    view.render(inputs={"u_field": other})
    data = view.canvas.fbo.read(viewport=(0, 0, 4, 4), components=3, dtype="f4")
    assert np.array_equal(np.frombuffer(data, "f4").reshape(4, 4, 3), _field(4, 4, 3))
    other.release()


# ---- 5. absent returns the source --------------------------------------------------------

_FIRST_FRAME_ONLY = (
    "        if context.frame == 0:\n"
    f"            return {{'view': {{'u_src': {_field_expr(4, 4, 3)}}}}}\n"
    "        return {}\n"
)


def test_an_omitted_field_hands_the_sampler_back_to_its_wire(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _document(
        tmp_path, gl_ctx, _FIRST_FRAME_ONLY, sampler="u_src", with_src=True, size=(4, 4)
    )
    _tick(engine, document, 0)
    assert np.array_equal(_drawn(document, 4, 4), _field(4, 4, 3))
    _tick(engine, document, 1)
    assert _is_src(_drawn(document, 4, 4))


# ---- 6. the source is never written ------------------------------------------------------


def test_a_fed_sampler_saves_exactly_as_before(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # UNWIRED (`NoSource`): `Pass.render` writes a value back only for a sampler the wiring
    # does not fill, so a wired one would pass this with the write-back broken.
    root = _write(
        tmp_path / "doc",
        f"        return {{'view': {{'u_field': {_field_expr(4, 4, 3)}}}}}\n",
    )
    ui_document = load_document_from_dir(root)
    document = ui_document.document
    document.passes["view"].uniform_values["u_field"] = NoSource()
    document.render()
    ui_document.save(tmp_path, dir_name="before")

    engine = ScriptEngine()
    engine.reload("doc", root / "scripts")
    for frame in range(3):
        _tick(engine, document, frame)
        document.render()
    assert document.passes["view"].script_textures  # the field did reach the draw
    ui_document.save(tmp_path, dir_name="after")

    before, after = tmp_path / "before", tmp_path / "after"
    assert (after / "document.json").read_bytes() == (
        before / "document.json"
    ).read_bytes()
    listing = sorted(
        p.relative_to(after).as_posix() for p in after.rglob("*") if p.is_file()
    )
    assert listing == sorted(
        p.relative_to(before).as_posix() for p in before.rglob("*") if p.is_file()
    )


# ---- 7. stop binds the source ------------------------------------------------------------


def test_a_stopped_field_hands_the_sampler_back_to_its_wire(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _document(
        tmp_path,
        gl_ctx,
        f"        return {{'view': {{'u_src': {_field_expr(4, 4, 3)}}}}}\n",
        sampler="u_src",
        with_src=True,
        size=(4, 4),
    )
    stopped = frozenset({StoppedKey(pass_name="view", name="u_src")})
    _tick(engine, document, 0, stopped)
    assert _is_src(_drawn(document, 4, 4))
    assert ("view", "u_src") in engine.script_driven_uniforms("doc")


# ---- 8/9. a behavior failure holds the slot and touches no source ------------------------

_THEN_RAISE = (
    "        if context.frame >= 1:\n"
    "            raise RuntimeError('boom')\n"
    f"        return {{'view': {{'u_field': {_field_expr(4, 4, 3)}}}}}\n"
)


def test_a_raise_keeps_the_last_field_on_screen(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _document(tmp_path, gl_ctx, _THEN_RAISE, size=(4, 4))
    _tick(engine, document, 0)
    _drawn(document, 4, 4)
    _tick(engine, document, 1)
    assert engine.errors.get(("doc", "", "script.py")) is not None
    assert np.array_equal(_drawn(document, 4, 4), _field(4, 4, 3))


def test_a_raise_never_restores_a_field_into_the_source(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _document(tmp_path, gl_ctx, _THEN_RAISE, size=(4, 4))
    view = document.passes["view"]
    source = view.uniform_values["u_field"]
    _tick(engine, document, 0)
    _drawn(document, 4, 4)
    _tick(engine, document, 1)
    _drawn(document, 4, 4)
    assert view.uniform_values["u_field"] is source

    # The probe, on an engine with no live failure recorded (a live one answers with it):
    # frames 6 and 12 sample after the raise at frame 1.
    probe_engine = ScriptEngine()
    probe_engine.reload("doc", tmp_path / "doc" / "scripts")
    probe = probe_engine.dry_run("doc", document, (0.0, 0.5, 1.0), 12)
    assert probe.runtime_error is not None
    assert all(("view", "u_field") not in values for _, values in probe.samples)


# ---- 10. removal clears ------------------------------------------------------------------


def test_removing_the_script_clears_what_it_fed(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _document(
        tmp_path,
        gl_ctx,
        f"        return {{'view': {{'u_src': {_field_expr(4, 4, 3)}}}}}\n",
        sampler="u_src",
        with_src=True,
        size=(4, 4),
    )
    _tick(engine, document, 0)
    assert not _is_src(_drawn(document, 4, 4))
    document.passes["view"].pending_instances = {"pos": np.zeros((1, 2), "f4")}

    (tmp_path / "doc" / "scripts" / "script.py").unlink()
    engine.reload("doc", tmp_path / "doc" / "scripts")
    _tick(engine, document, 1)
    assert _is_src(_drawn(document, 4, 4))
    assert document.passes["view"].pending_instances == {}
    assert engine.script_driven_uniforms("doc") == set()


# ---- 11/12/13. refusal -------------------------------------------------------------------

_GOOD_THEN_F8 = (
    "        if context.frame == 0:\n"
    f"            return {{'view': {{'u_src': {_field_expr(4, 4, 3)}}}}}\n"
    "        return {'view': {'u_src': np.ones((4, 4, 3))}}\n"
)


def test_a_refused_field_holds_the_last_good_one(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _document(
        tmp_path, gl_ctx, _GOOD_THEN_F8, sampler="u_src", with_src=True, size=(4, 4)
    )
    _tick(engine, document, 0)
    _drawn(document, 4, 4)
    _tick(engine, document, 1)
    assert document.passes["view"].pending_textures["u_src"] is REFUSED_TEXTURE
    assert np.array_equal(_drawn(document, 4, 4), _field(4, 4, 3))


def test_a_refused_field_warns_on_the_strip_and_stays_driven(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _document(
        tmp_path, gl_ctx, _GOOD_THEN_F8, sampler="u_src", with_src=True, size=(4, 4)
    )
    _tick(engine, document, 1)
    error = engine.errors[("doc", "view", "u_src")]
    assert "float64" in error.message and "astype" in error.message
    status = engine.script_status("doc")
    assert status is not None
    assert [(p, n) for p, n, _ in status.soft_errors] == [("view", "u_src")]
    assert ("view", "u_src") in engine.script_driven_uniforms("doc")


def test_a_field_refused_on_the_first_frame_reads_black(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _document(
        tmp_path, gl_ctx, _GOOD_THEN_F8, sampler="u_src", with_src=True, size=(4, 4)
    )
    _tick(engine, document, 1)
    assert np.array_equal(_drawn(document, 4, 4), np.zeros((4, 4, 3), "f4"))


# ---- 14. reallocation --------------------------------------------------------------------


def test_a_field_whose_layout_changes_gets_a_new_texture(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # Pairs of equal byte size, so a texture kept by size alone takes the write silently.
    document, engine = _document(
        tmp_path,
        gl_ctx,
        "        return {'view': {'u_field': self.fields[context.frame]}}\n",
        init_body=(
            f"        self.fields = [{_field_expr(32, 64, 3, 1e4)}, "
            f"{_field_expr(64, 32, 3, 1e4)}, "
            f"{_field_expr(16, 16, 1, 1e4)}, "
            "(np.arange(16 * 16 * 4) % 250 + 1).astype(np.uint8).reshape(16, 16, 4)]\n"
        ),
        size=(64, 64),
    )
    expected = [
        _field(32, 64, 3, 1e4),
        _field(64, 32, 3, 1e4),
        np.repeat(_field(16, 16, 1, 1e4), 3, axis=2) * np.array([1, 0, 0], "f4"),
        ((np.arange(16 * 16 * 4) % 250 + 1).reshape(16, 16, 4)[..., :3] / 255.0).astype(
            "f4"
        ),
    ]
    for frame, want in enumerate(expected):
        _tick(engine, document, frame)
        height, width = want.shape[:2]
        got = _drawn(document, 64, 64)[:height, :width]
        assert np.allclose(got, want, atol=1e-6), f"frame {frame}"


# ---- 15. uncopied ------------------------------------------------------------------------


def test_the_slot_holds_the_script_s_own_array(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # The script returns ONE array object every frame; an engine copy hands the slot a new
    # object each tick.
    document, engine = _document(
        tmp_path,
        gl_ctx,
        "        return {'view': {'u_field': self.f}}\n",
        init_body=f"        self.f = {_field_expr(4, 4, 3)}\n",
        size=(4, 4),
    )
    _tick(engine, document, 0)
    first = document.passes["view"].pending_textures["u_field"]
    _tick(engine, document, 1)
    assert document.passes["view"].pending_textures["u_field"] is first


_STAMP_C = r"""
#include "shaderbox_native.h"
SB_LIBRARY()
SB_EXPORT int32_t stamp(sb_call *c) {
    sb_arg *b = sb_buf(c, "field");
    if (!b || b->dtype != SB_F32) return sb_fail(c, "field must be f4");
    float v = (float)sb_num(c, "v", 0.0);
    int64_t n = 1;
    for (int32_t d = 0; d < b->ndim; d++) n *= b->shape[d];
    for (int64_t i = 0; i < n; i++) ((float *)b->data)[i] = v;
    return 0;
}
"""


@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler")
def test_a_native_buffer_reaches_the_pass_uncopied(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _document(
        tmp_path,
        gl_ctx,
        "        self.lib.call('stamp', field=self.buf, v=0.1 * (context.frame + 1))\n"
        "        return {'view': {'u_field': self.buf}}\n",
        init_body=(
            "        self.lib = native('stamp')\n"
            "        self.buf = self.lib.buffer((4, 4, 3), 'f4')\n"
        ),
        size=(4, 4),
    )
    scripts = tmp_path / "doc" / "scripts"
    (scripts / "stamp.c").write_text(_STAMP_C)
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            f"-I{RESOURCES_DIR / 'native'}",
            "-o",
            str(scripts / "stamp.so"),
            str(scripts / "stamp.c"),
        ],
        check=True,
    )
    engine.reload("doc", scripts)  # the library exists now; re-run __init__ against it
    engine.reset("doc")
    try:
        _tick(engine, document, 0)
        first = document.passes["view"].pending_textures["u_field"]
        assert np.allclose(_drawn(document, 4, 4), 0.1)
        _tick(engine, document, 1)
        assert document.passes["view"].pending_textures["u_field"] is first
        assert np.allclose(_drawn(document, 4, 4), 0.2)
    finally:
        engine.release_all()


# ---- 16. wrap ----------------------------------------------------------------------------


def test_a_field_texture_clamps_at_its_edge(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # Filter is not gated: moderngl's default is already linear, so no break could show.
    document, engine = _document(
        tmp_path,
        gl_ctx,
        f"        return {{'view': {{'u_field': {_field_expr(4, 4, 3)}}}}}\n",
        size=(4, 4),
    )
    _tick(engine, document, 0)
    _drawn(document, 4, 4)
    texture = document.passes["view"].script_textures["u_field"]
    assert texture.repeat_x is False and texture.repeat_y is False


# ---- 17/18/19. the probe -----------------------------------------------------------------

_A_B_A = (
    "        a = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)\n"
    "        b = np.random.default_rng(0).permutation(a.ravel()).reshape(32, 32)\n"
    "        return {'view': {'u_field': b if context.frame == 1 else a}}\n"
)


def test_a_dry_run_leaves_the_live_slot_alone(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _document(tmp_path, gl_ctx, _A_B_A, size=(32, 32))
    _tick(engine, document, 0)
    live = document.passes["view"].pending_textures
    engine.dry_run("doc", document, (0.0, 0.5, 1.0), 12)
    assert document.passes["view"].pending_textures is live


def test_a_field_that_moves_inside_a_held_range_reads_as_moving(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # B is a permutation of A, so min, max and mean all hold; only the content moved. 32x32
    # keeps the digest's stride at 1, so it sees every texel.
    document, engine = _document(tmp_path, gl_ctx, _A_B_A, size=(32, 32))
    probe = engine.dry_run("doc", document, (0.0, 1 / 12, 2 / 12), 12)
    first, second = (
        samples[("view", "u_field")] for _, samples in probe.texture_samples[:2]
    )
    assert (first.minimum, first.maximum, first.mean) == (
        second.minimum,
        second.maximum,
        second.mean,
    )
    facts = _texture_facts_text(probe)
    assert "float32 (32, 32)" in facts and "range [0, 1.02e+03]" in facts
    assert "view.u_field: field CHANGES across t (ANIMATING)" in facts


def test_a_constant_field_reads_as_static(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _document(
        tmp_path,
        gl_ctx,
        "        return {'view': {'u_field': np.ones((4, 4), np.float32)}}\n",
        size=(4, 4),
    )
    probe = engine.dry_run("doc", document, (0.0, 0.5, 1.0), 12)
    assert "field UNCHANGED across t (STATIC)" in _texture_facts_text(probe)


def test_a_texture_only_script_is_not_called_static_or_idle(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _document(tmp_path, gl_ctx, _A_B_A, size=(32, 32))
    probe = engine.dry_run("doc", document, (0.0, 1 / 12, 2 / 12), 12)
    verdict = _motion_verdict(probe, "", 1e-4)
    assert "texture facts" in verdict
    assert "UNCHANGED (STATIC)" not in verdict and "Nothing animates" not in verdict


# ---- 20/21. one predicate ----------------------------------------------------------------

_GL_SAMPLER_2D = 0x8B5E


def _uniform(name: str, gl_type: int, dim: int = 1) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        name=name, dimension=dim, array_length=1, gl_type=gl_type, value=0.0
    )


class _FakePass:
    def __init__(self, uniforms: list[types.SimpleNamespace]) -> None:
        self.uniform_values: dict[str, object] = {}
        self.script_ready = True
        self._uniforms = uniforms

    def get_active_uniforms(self) -> list[types.SimpleNamespace]:
        return self._uniforms


def _lands_as_texture(tmp_path: Path, value_expr: str) -> bool:
    scripts = tmp_path / value_expr.replace(" ", "")[:40].replace("/", "_")
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "script.py").write_text(
        _HEAD
        + f"V = {value_expr}\n"
        + "class Behavior(ScriptBehavior):\n"
        + "    def update(self, context):\n"
        + "        return {'main': {'u_tex': V}}\n"
    )
    document = types.SimpleNamespace(
        passes={"main": _FakePass([_uniform("u_tex", _GL_SAMPLER_2D)])}
    )
    engine = ScriptEngine()
    engine.reload("n0", scripts)
    engine.tick("n0", document, ScriptContext(t=0.0, dt=1 / 60, frame=0))
    landed = document.passes["main"].pending_textures.get("u_tex")
    return landed is not None and landed is not REFUSED_TEXTURE


_TEXTURE_CASES = [
    "np.zeros((4, 4), np.float32)",
    "np.zeros((4, 4, 1), np.float32)",
    "np.zeros((4, 4, 2), np.float32)",
    "np.zeros((4, 4, 3), np.float32)",
    "np.zeros((4, 4, 4), np.float32)",
    "np.zeros((4, 4, 3), np.uint8)",
    "np.zeros((4, 4))",
    "np.zeros((4, 4), np.int32)",
    "np.zeros((4, 4), '>f4')",
    "np.zeros((2, 4, 4, 1), np.float32)",
    "np.zeros((4, 4, 5), np.float32)",
    "np.zeros((4, 8), np.float32)[:, ::2]",
    "np.zeros((0, 4), np.float32)",
    f"np.zeros((1, {MAX_SCRIPT_TEXTURE_EXTENT + 1}), np.float32)",
    "np.zeros(3, np.float32)",
]


def test_completion_offers_a_sampler_exactly_when_a_tick_lands_the_field(
    tmp_path: Path,
) -> None:
    for case in _TEXTURE_CASES:
        value = eval(case, {"np": np})
        offered = glsl_type_of_value(value) == "sampler2D"
        assert offered == _lands_as_texture(tmp_path, case), case
    assert _lands_as_texture(tmp_path, _TEXTURE_CASES[5])  # the u1 case does land


def test_completion_and_coercion_agree_on_a_one_dimensional_array() -> None:
    vec_types = {
        1: GL.GL_FLOAT,
        2: GL.GL_FLOAT_VEC2,
        3: GL.GL_FLOAT_VEC3,
        4: GL.GL_FLOAT_VEC4,
    }
    names = {1: "float", 2: "vec2", 3: "vec3", 4: "vec4"}
    for length in range(1, 5):
        uniform = _uniform("u_v", int(vec_types[length]), dim=length)
        numbers = [float(i) for i in range(length)]
        for value in (numbers, np.array(numbers, np.float32)):
            lands = coerce_uniform_value(value, uniform) is not None
            assert lands == (glsl_type_of_value(value) == names[length]), (
                length,
                value,
            )
    assert coerce_uniform_value(
        np.array([1.0, 2.0], np.float32), _uniform("u", int(GL.GL_FLOAT_VEC2), 2)
    ) == (1.0, 2.0)


# ---- 22. only a single sampler2D is drivable ---------------------------------------------

_VALUE_DECLS = [
    ("float", "x"),
    ("vec2", "x.x"),
    ("vec3", "x.x"),
    ("vec4", "x.x"),
    ("int", "float(x)"),
    ("ivec2", "float(x.x)"),
    ("ivec3", "float(x.x)"),
    ("ivec4", "float(x.x)"),
    ("uint", "float(x)"),
    ("uvec2", "float(x.x)"),
    ("uvec3", "float(x.x)"),
    ("uvec4", "float(x.x)"),
    ("bool", "float(x)"),
    ("bvec2", "float(x.x)"),
    ("bvec3", "float(x.x)"),
    ("bvec4", "float(x.x)"),
    ("mat2", "x[0][0]"),
    ("mat3", "x[0][0]"),
    ("mat4", "x[0][0]"),
    ("mat2x3", "x[0][0]"),
    ("mat2x4", "x[0][0]"),
    ("mat3x2", "x[0][0]"),
    ("mat3x4", "x[0][0]"),
    ("mat4x2", "x[0][0]"),
    ("mat4x3", "x[0][0]"),
    ("double", "float(x)"),
    ("dvec2", "float(x.x)"),
    ("dvec3", "float(x.x)"),
    ("dvec4", "float(x.x)"),
    ("dmat2", "float(x[0][0])"),
    ("dmat3", "float(x[0][0])"),
    ("dmat4", "float(x[0][0])"),
    ("dmat2x3", "float(x[0][0])"),
    ("dmat2x4", "float(x[0][0])"),
    ("dmat3x2", "float(x[0][0])"),
    ("dmat3x4", "float(x[0][0])"),
    ("dmat4x2", "float(x[0][0])"),
    ("dmat4x3", "float(x[0][0])"),
]


def _active(
    gl_ctx: moderngl.Context, decls: list[tuple[str, str, str]]
) -> dict[str, Any]:
    body = "\n".join(f"uniform {t} {n};" for t, n, _ in decls)
    use = " + ".join(expr for _, _, expr in decls)
    program = gl_ctx.program(
        vertex_shader="#version 460 core\nin vec2 p;\nvoid main() { gl_Position = vec4(p, 0, 1); }\n",
        fragment_shader=(
            f"#version 460 core\n{body}\nout vec4 fs_color;\n"
            f"void main() {{ fs_color = vec4({use}); }}\n"
        ),
    )
    return {name: program[name] for name in program if name.startswith("u_")}


def test_every_value_type_is_scriptable(gl_ctx: moderngl.Context) -> None:
    decls = [
        (t, f"u_{i}", expr.replace("x", f"u_{i}", 1))
        for i, (t, expr) in enumerate(_VALUE_DECLS)
    ]
    active = _active(gl_ctx, decls)
    assert len(active) == len(decls)
    for t, name, _ in decls:
        assert is_scriptable(active[name]), t


def test_only_a_single_sampler2D_takes_a_field(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    decls = [
        ("sampler3D", "u_vol", "texture(u_vol, vec3(0)).x"),
        ("samplerCube", "u_cube", "texture(u_cube, vec3(1)).x"),
        ("isampler2D", "u_int", "float(texture(u_int, vec2(0)).x)"),
        (
            "sampler2D",
            "u_pair[2]",
            "texture(u_pair[0], vec2(0)).x + texture(u_pair[1], vec2(0)).x",
        ),
        ("sampler2D", "u_one", "texture(u_one, vec2(0)).x"),
    ]
    active = _active(gl_ctx, decls)
    for name in ("u_vol", "u_cube", "u_int", "u_pair"):
        assert not is_scriptable(active[name]) and not is_script_texture(
            active[name]
        ), name
    assert is_script_texture(active["u_one"])

    sampler_constants = {
        int(getattr(GL, n))
        for n in dir(GL)
        if n.startswith("GL_")
        and ("SAMPLER" in n or "IMAGE" in n)
        and isinstance(getattr(GL, n), int)
    }
    assert not sampler_constants & VALUE_GL_TYPES

    for value in ("1.0", "np.zeros((4, 4), np.float32)"):
        scripts = tmp_path / f"s{len(value)}"
        scripts.mkdir()
        (scripts / "script.py").write_text(
            _HEAD
            + "class Behavior(ScriptBehavior):\n    def update(self, context):\n"
            + f"        return {{'u_vol': {value}, 'u_cube': {value}, 'u_int': {value}, "
            f"'u_pair': {value}}}\n"
        )
        fake = _FakePass(list(active.values()))
        document = types.SimpleNamespace(passes={"main": fake})
        engine = ScriptEngine()
        engine.reload("n0", scripts)
        engine.tick("n0", document, ScriptContext(t=0.0, dt=1 / 60, frame=0))
        for name in ("u_vol", "u_cube", "u_int", "u_pair"):
            assert (
                "not a value a script can drive"
                in engine.errors[("n0", "main", name)].message
            )
        assert fake.uniform_values == {} and getattr(fake, "pending_textures", {}) == {}


# ---- 23. pins and ports ------------------------------------------------------------------


def test_every_port_kind_has_a_pin() -> None:
    assert set(get_args(PortKind)) <= set(_PIN_BY_KIND)


def test_a_fed_sampler_is_a_script_port_that_keeps_its_wire(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _document(
        tmp_path,
        gl_ctx,
        f"        return {{'view': {{'u_src': {_field_expr(4, 4, 3)}}}}}\n",
        sampler="u_src",
        with_src=True,
        size=(4, 4),
    )
    _tick(engine, document, 0)
    ports = ports_of(document, document.effective_wiring())
    assert [(p.sampler, p.kind, p.source) for p in ports["view"]] == [
        ("u_src", "script", "src")
    ]
    view = flat_view(["src", "view"], ports, {})
    assert [(edge.sampler) for edge in view.edges] == ["u_src"]


def test_a_fed_feedback_sampler_has_a_port_and_no_edge() -> None:
    ports = node_ports(["u_prev"], {}, {"u_prev": "view"}, "view", {"u_prev"})
    assert [(p.kind, p.source) for p in ports] == [("script", None)]
    view = flat_view(["view"], {"view": ports}, {})
    assert view.edges == ()


# ---- 27. strip chips ---------------------------------------------------------------------


def test_a_producer_keeps_its_chip_while_an_unfed_sampler_reads_it() -> None:
    wiring = {"view": {"u_a": "src", "u_b": "src"}}
    assert _reads("view", wiring, ["src", "view"], {"u_a"}) == ["src"]
    assert _reads("view", wiring, ["src", "view"], {"u_a", "u_b"}) == []


# ---- 28. headless driven set -------------------------------------------------------------


def test_a_dry_run_alone_marks_a_fed_sampler_driven(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _document(
        tmp_path,
        gl_ctx,
        f"        return {{'view': {{'u_field': {_field_expr(4, 4, 3)}}}}}\n",
        size=(4, 4),
    )
    engine.dry_run("doc", document, (0.0, 0.5), 12)
    assert ("view", "u_field") in engine.script_driven_uniforms("doc")


# ---- 29. copilot cell --------------------------------------------------------------------


def test_the_copilot_reads_a_fed_sampler_as_the_script_s(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # Through the working-set formatter, with the sampler in the driven set exactly as the
    # backend passes it: the cell has to win over the "driven by script.py" marker to be seen.
    document, engine = _document(
        tmp_path,
        gl_ctx,
        f"        return {{'view': {{'u_src': {_field_expr(4, 4, 3)}}}}}\n",
        sampler="u_src",
        with_src=True,
        size=(4, 4),
    )
    _tick(engine, document, 0)
    _drawn(document, 4, 4)
    rows = _format_uniforms(
        document.passes["view"], {"u_src"}, document.effective_wiring()["view"]
    )
    assert "u_src sampler2D <- (script field 4x4, 3 x float32)" in rows


def test_the_copilot_reads_a_first_frame_refusal_as_black(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _document(
        tmp_path, gl_ctx, _GOOD_THEN_F8, sampler="u_src", with_src=True, size=(4, 4)
    )
    _tick(engine, document, 1)
    _drawn(document, 4, 4)
    rows = _format_uniforms(
        document.passes["view"], {"u_src"}, document.effective_wiring()["view"]
    )
    assert "u_src sampler2D <- (script field REFUSED this frame; reads BLACK)" in rows


# ---- 30. release -------------------------------------------------------------------------


def test_releasing_a_pass_frees_its_field_textures(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _document(
        tmp_path,
        gl_ctx,
        f"        return {{'view': {{'u_field': {_field_expr(4, 4, 3)}}}}}\n",
        size=(4, 4),
    )
    _tick(engine, document, 0)
    _drawn(document, 4, 4)
    view: Pass = document.passes["view"]
    held = list(view.script_textures.values())
    assert held
    view.release()
    assert view.script_textures == {}
    assert all(type(t.mglo).__name__ == "InvalidObject" for t in held)


# ---- 24/25/26. the app: picks while playing, export --------------------------------------

_APP_VIEW = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_field;
out vec4 fs_color;
void main() { fs_color = texture(u_field, vs_uv); }
"""


def _app_fed(app: Any) -> tuple[str, Any, str]:
    # The current document's output pass samples `u_field`, and a script feeds it a constant
    # 0.75 red field: the row plays.
    document_id = app.current_document_id
    document = app.ui_documents[document_id].document
    render_pass = document.render_pass
    render_pass.release_program(_APP_VIEW)
    render_pass.source.path.write_text(_APP_VIEW)
    render_pass.compile()
    render_pass.seed_uniform_values()
    pass_name = pass_name_of(render_pass.source.path)
    scripts_dir = app.session.paths.scripts_dir_for(document_id)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / DOCUMENT_SCRIPT_BASENAME).write_text(
        _HEAD
        + "class Behavior(ScriptBehavior):\n    def update(self, context):\n"
        + "        f = np.zeros((2, 2, 3), np.float32)\n        f[..., 0] = 0.75\n"
        + f"        return {{'{pass_name}': {{'u_field': f}}}}\n"
    )
    app.session.reload_scripts()
    app.session.tick([document_id], 0.0, 1 / 60)
    assert app.session.uniform_is_driven(document_id, pass_name, "u_field")
    return document_id, render_pass, pass_name


def _draw_row_picking(
    app: Any, monkeypatch: pytest.MonkeyPatch, render_pass: Any, pick: str
) -> None:
    def spy(id_: str, current: Any, groups: Any, width: float) -> int | None:
        labels = [label for _, rows in groups for label, _ in rows]
        return labels.index(pick) if id_.startswith("##source_") else None

    monkeypatch.setattr(uniform_widget, "grouped_combo", spy)
    ui_uniform = next(
        UIUniform.from_uniform(u)
        for u in render_pass.get_active_uniforms()
        if u.name == "u_field"
    )
    imgui.new_frame()
    imgui.begin("rig")
    uniform_widget.draw_ui_uniform(app, ui_uniform)
    imgui.end()
    imgui.end_frame()


def test_picking_none_on_a_playing_sampler_stops_it(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_id, render_pass, pass_name = _app_fed(app)
    # The row must show something other than `none`, or picking it is no change at all.
    render_pass.uniform_values["u_field"] = PassSource(pass_name)
    assert not app.session.is_uniform_stopped(document_id, pass_name, "u_field")
    _draw_row_picking(app, monkeypatch, render_pass, "none")
    assert app.session.is_uniform_stopped(document_id, pass_name, "u_field")
    assert isinstance(render_pass.uniform_values["u_field"], NoSource)


def test_picking_a_file_on_a_playing_sampler_binds_it(
    app: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    document_id, render_pass, pass_name = _app_fed(app)
    picture = tmp_path / "pick.png"
    PILImage.new("RGB", (2, 2), (10, 20, 30)).save(picture)
    monkeypatch.setattr(uniform_widget, "_pick_media_file", lambda: picture)
    _draw_row_picking(app, monkeypatch, render_pass, "file...")
    assert isinstance(render_pass.uniform_values["u_field"], MediaWithTexture)
    assert app.session.is_uniform_stopped(document_id, pass_name, "u_field")


def test_an_exported_frame_carries_the_field(app: Any, tmp_path: Path) -> None:
    document_id, render_pass, _ = _app_fed(app)
    document = app.ui_documents[document_id].document
    out = tmp_path / "exported.png"
    details = MediaDetails(is_video=False, duration=1.0)
    details.file_details.path = str(out)
    width, height = render_pass.canvas.texture.size
    details.resolution_details.width = width
    details.resolution_details.height = height
    render_pass.pending_textures = {}  # only the export's own tick can feed it now
    document.render_media(details)
    red = PILImage.open(out).convert("RGBA").getpixel((0, 0))[0]
    assert abs(red - round(0.75 * 255)) <= 1, red


# ---- post-implementation review: the fixes it asked for ----------------------------------


def _fake_engine(
    tmp_path: Path, update_body: str
) -> tuple[ScriptEngine, Any, list[str]]:
    scripts = tmp_path / "fake_scripts"
    scripts.mkdir()
    (scripts / "script.py").write_text(
        _HEAD
        + "class Behavior(ScriptBehavior):\n    def update(self, context):\n"
        + update_body
    )
    warnings: list[str] = []
    engine = ScriptEngine(
        on_key_warning=lambda doc, p, n, reason, message: warnings.append(reason)
    )
    engine.reload("n0", scripts)
    document = types.SimpleNamespace(
        passes={
            "main": _FakePass(
                [_uniform("u_a", int(GL.GL_FLOAT)), _uniform("u_tex", _GL_SAMPLER_2D)]
            )
        }
    )
    return engine, document, warnings


def test_a_fault_sent_now_and_then_warns_once(tmp_path: Path) -> None:
    # The same typo returned every other frame is one fault, not one toast per return.
    engine, document, warnings = _fake_engine(
        tmp_path,
        "        return {'u_ghost': 1.0} if context.frame % 2 == 0 else {}\n",
    )
    for frame in range(10):
        engine.tick("n0", document, ScriptContext(t=frame / 60, dt=1 / 60, frame=frame))
    assert warnings == ["no_such_uniform"]


def test_a_field_that_lands_and_fails_again_warns_again(tmp_path: Path) -> None:
    engine, document, warnings = _fake_engine(
        tmp_path,
        "        bad = context.frame in (1, 2, 3, 5)\n"
        "        f = np.zeros((2, 2)) if bad else np.zeros((2, 2), np.float32)\n"
        "        return {'main': {'u_tex': f}}\n",
    )
    for frame in range(6):
        engine.tick("n0", document, ScriptContext(t=frame / 60, dt=1 / 60, frame=frame))
    assert warnings == ["bad_texture", "bad_texture"]


def test_an_empty_script_reads_as_driving_nothing(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _document(tmp_path, gl_ctx, "        return {}\n", size=(4, 4))
    probe = engine.dry_run("doc", document, (0.0, 0.5), 12)
    verdict = _motion_verdict(probe, "", 1e-4)
    assert "Nothing animates" in verdict and "facts below" not in verdict


def test_a_field_sent_early_is_still_reported(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _document(
        tmp_path,
        gl_ctx,
        "        if context.t < 0.9:\n"
        f"            return {{'view': {{'u_field': {_field_expr(4, 4, 3)}}}}}\n"
        "        return {}\n",
        size=(4, 4),
    )
    probe = engine.dry_run("doc", document, (0.0, 0.5, 1.0), 12)
    facts = _texture_facts_text(probe)
    assert "view.u_field: field float32 (4, 4, 3)" in facts
    assert "field UNCHANGED across t (STATIC)" in facts


def test_a_good_block_field_clears_the_bare_key_s_refusal(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _document(
        tmp_path,
        gl_ctx,
        "        return {'u_field': np.zeros((4, 4)), "
        f"'view': {{'u_field': {_field_expr(4, 4, 3)}}}}}\n",
        size=(4, 4),
    )
    _tick(engine, document, 0)
    assert np.array_equal(_drawn(document, 4, 4), _field(4, 4, 3))
    status = engine.script_status("doc")
    assert status is not None and status.soft_errors == []


def test_a_field_that_stopped_cannot_return_as_a_held_frame(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # Fed, then dropped, then refused: the refusal reads black, not the frame from before the
    # gap -- that frame belongs to a field the script had stopped sending.
    document, engine = _document(
        tmp_path,
        gl_ctx,
        "        if context.frame == 0:\n"
        f"            return {{'view': {{'u_field': {_field_expr(4, 4, 3)}}}}}\n"
        "        if context.frame == 1:\n"
        "            return {}\n"
        "        return {'view': {'u_field': np.zeros((4, 4))}}\n",
        size=(4, 4),
    )
    for frame in range(3):
        _tick(engine, document, frame)
        drawn = _drawn(document, 4, 4)
    assert np.array_equal(drawn, np.zeros((4, 4, 3), "f4"))


def test_a_sampler_that_leaves_the_program_releases_its_field(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # The prune must run on a draw that reaches no sampler at all, or an edit that removes the
    # sampler keeps the old frame alive for a later refusal to show.
    document, engine = _document(
        tmp_path,
        gl_ctx,
        "        if context.frame == 0:\n"
        f"            return {{'view': {{'u_field': {_field_expr(4, 4, 3)}}}}}\n"
        "        return {}\n",
        size=(4, 4),
    )
    _tick(engine, document, 0)
    _drawn(document, 4, 4)
    view = document.passes["view"]
    assert view.script_textures
    view.release_program(_SRC)
    view.compile()
    _tick(engine, document, 1)
    _drawn(document, 4, 4)
    assert view.script_textures == {}
