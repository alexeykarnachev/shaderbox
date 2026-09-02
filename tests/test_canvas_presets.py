"""The Document tab's canvas presets menu (069 W-A).

What the menu offers and where each entry comes from: the squares W-H's first tutorial step
depends on, the named video shapes single-homed in `render_shape.py`, and any bound texture's
size across ALL passes. Building the list forces no compile (066 D1) and no entry can be
silently altered by the clamp on its way through `_apply_canvas_size`.

Needs a real GL context, like every other module that builds a Document.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import moderngl
import pytest
from PIL import Image as PILImage

from shaderbox.core import Pass
from shaderbox.document import DEFAULT_PASS_NAME, Document
from shaderbox.media import Image
from shaderbox.pass_graph import PassEntry, PassGraph, clamp_canvas_size
from shaderbox.paths import shader_lib_root
from shaderbox.render_preset import resolve_dims
from shaderbox.render_shape import MENU_SHAPES, SHAPE_TABLE, RenderShape, shape_to_preset
from shaderbox.shader_lib import ShaderLibIndex, set_active
from shaderbox.tabs.document import _canvas_presets
from shaderbox.ui_models import UIDocument

_PLAIN = """#version 460 core
in vec2 vs_uv;
out vec4 fs_color;
void main() { fs_color = vec4(1.0, 0.0, 0.0, 1.0); }
"""

_SAMPLES = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_image;
out vec4 fs_color;
void main() { fs_color = texture(u_image, vs_uv); }
"""


@pytest.fixture(scope="module")
def gl_ctx() -> Iterator[moderngl.Context]:
    os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.6")
    os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "460")
    try:
        ctx = moderngl.create_standalone_context()
    except Exception as e:
        pytest.skip(f"no standalone GL context available: {e}")
    set_active(ShaderLibIndex.build(shader_lib_root()))
    yield ctx
    ctx.release()


def _document(
    gl: moderngl.Context,
    sources: dict[str, str],
    graph: PassGraph,
    size: tuple[int, int] = (8, 8),
    *,
    compile_passes: bool = True,
) -> Document:
    doc = Document(gl=gl, canvas_size=size)
    doc.passes[DEFAULT_PASS_NAME].release()
    doc.passes = {}
    for name, src in sources.items():
        entry = graph.passes.get(name, PassEntry())
        render_pass = Pass(gl=gl, canvas_size=size, target=entry.target)
        render_pass.release_program(src)
        if compile_passes:
            render_pass.compile()
            assert render_pass.compile_unit.errors == [], (
                f"{name}: {render_pass.compile_unit.errors}"
            )
        doc.passes[name] = render_pass
    doc.graph = graph
    return doc


def _presets(doc: Document) -> list[tuple[str, tuple[int, int]]]:
    return _canvas_presets(UIDocument(document=doc))


def _one_pass(gl: moderngl.Context, size: tuple[int, int] = (8, 8)) -> Document:
    return _document(gl, {DEFAULT_PASS_NAME: _PLAIN}, PassGraph(), size=size)


def test_the_square_presets_include_512(gl_ctx: moderngl.Context) -> None:
    # W-H's "Before you start" step names 512 x 512; drop 512 and that step is unperformable.
    doc = _one_pass(gl_ctx)
    presets = _presets(doc)
    assert ("512x512 (1:1)", (512, 512)) in presets
    sizes = [size for _, size in presets]
    for n in (256, 1024, 2048):
        assert (n, n) in sizes
    doc.release()


def test_every_preset_survives_the_clamp(gl_ctx: moderngl.Context) -> None:
    # A preset outside [16, 4096] would show one number and set another.
    doc = _one_pass(gl_ctx)
    for label, size in _presets(doc):
        assert clamp_canvas_size(size) == size, label
    doc.release()


def test_the_video_shapes_come_from_the_shape_table(gl_ctx: moderngl.Context) -> None:
    # Single-homed in render_shape.py: labels AND dims are recomputed here from the table, so a
    # hand-rolled literal that drifts from `longest_edge` goes red before the menu disagrees
    # with the Share tab. The (1, 1) source size is itself the assertion that FIXED_ASPECT
    # ignores it -- the menu passes the real canvas and must get the same answer.
    doc = _one_pass(gl_ctx)
    presets = _presets(doc)
    by_label = dict(presets)
    for shape in MENU_SHAPES:
        label = SHAPE_TABLE[shape].menu_label
        if shape is RenderShape.NATIVE:
            assert label not in by_label
            continue
        expected = resolve_dims(
            shape_to_preset(
                shape, is_video=False, fps=None, container=None, duration_max=None
            ),
            (1, 1),
        )
        assert by_label[label] == expected, label
        assert [lbl for lbl, _ in presets].count(label) == 1, label
    doc.release()


def test_a_bound_texture_is_offered_and_the_default_image_is_not(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # The seeded placeholder is not a choice the user made; a texture bound on a NON-output
    # pass is, and the old combo could not see it.
    doc = _document(
        gl_ctx,
        {"src": _PLAIN, "out": _SAMPLES},
        PassGraph(output="out", passes={"out": PassEntry()}),
    )
    doc.render(u_time=0.0)
    seeded = doc.passes["out"].uniform_values["u_image"]
    assert seeded is not None
    assert not any("u_image" in label for label, _ in _presets(doc)), (
        "the shipped default image is not a canvas preset"
    )

    path = tmp_path / "bound.png"
    PILImage.new("RGBA", (1234, 567), (10, 20, 30, 255)).save(path)
    doc.passes["src"].uniform_values["u_bound"] = Image(path)
    presets = _presets(doc)
    assert (1234, 567) in [size for _, size in presets]
    assert any("u_bound" in label for label, _ in presets)
    doc.release()


def test_building_the_presets_compiles_nothing(gl_ctx: moderngl.Context) -> None:
    # get_active_uniforms() compiles a never-attempted pass (066 D1); on the Document tab that
    # is the whole graph on the first frame, invisible because it renders correctly.
    doc = _document(
        gl_ctx,
        {"a": _PLAIN, "b": _PLAIN, "c": _PLAIN},
        PassGraph(output="c"),
        compile_passes=False,
    )
    for render_pass in doc.passes.values():
        assert render_pass.program is None
    _presets(doc)
    for render_pass in doc.passes.values():
        assert render_pass.program is None
    doc.release()


def test_no_preset_duplicates_the_current_size(gl_ctx: moderngl.Context) -> None:
    # Picking the current size is a no-op the early return swallows, which reads as a dead item.
    doc = _one_pass(gl_ctx, size=(512, 512))
    assert (512, 512) not in [size for _, size in _presets(doc)]
    doc.release()
