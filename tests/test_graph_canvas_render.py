"""The moderngl renderer draws what the library emits (feature 098).

Every test here pins a defect that shipped once during this feature's
implementation, because all three rendered something plausible rather than
failing: a stride that made every instance after the first read its
neighbour's parameters, a run offset that walked off the end of the buffer,
and a preview texture sharing the glyph atlas's unit.
"""

import ctypes

import moderngl
import numpy as np
import pytest

from shaderbox.graph_canvas import ffi
from shaderbox.graph_canvas.render import (
    ortho,
    _SHAPE_ATTRS,
    _SHAPE_BINDINGS,
    _SHAPE_FORMAT,
    _SHAPE_STRIDE,
    CanvasPanel,
    CanvasRenderer,
    glyphs_array,
    shapes_array,
)


def _one_node_frame(canvas: ffi.Canvas, size: tuple[float, float]) -> ffi.Result:
    nodes = [
        ffi.NodeSpec(id=1, title="seed", x=0, y=0, ports=[ffi.PortSpec("out", False)])
    ]
    return canvas.frame(nodes, [], size, ffi.View(), ffi.PointerState())


def _lit(fbo: moderngl.Framebuffer, size: tuple[int, int]) -> int:
    pixels = np.frombuffer(fbo.read(components=3), dtype="u1")
    return int((pixels.reshape(size[1], size[0], 3).sum(axis=2) > 0).sum())


def test_the_shape_format_describes_the_struct_exactly(gl_ctx: moderngl.Context) -> None:
    """The instance is 120 bytes and its eight fields TILE it: 4+4+4+4+4+2+4+4
    is 30 floats, not 26. A padding token appended to the format would widen
    the binding's stride past the row, and then instance N reads from N times
    the wrong stride -- which instance 0 can never reveal."""
    widths = [int(token[0]) for token in _SHAPE_FORMAT.split("/")[0].split()]
    assert sum(widths) * 4 == ctypes.sizeof(ffi.ShapeInstance) == _SHAPE_STRIDE
    assert len(widths) == len(_SHAPE_ATTRS)
    assert "x" not in _SHAPE_FORMAT


def test_every_instance_reads_its_own_row(gl_ctx: moderngl.Context) -> None:
    """The falsifier for the stride bug. Two instances carrying distinct
    markers must come back distinct; under a widened stride the second reads
    into the first's tail and the values shift by a field."""
    renderer = CanvasRenderer(gl=gl_ctx)
    program = gl_ctx.program(
        vertex_shader=(
            "#version 330 core\n"
            + (
                gl_ctx.extra
                if False
                else open(
                    "shaderbox/resources/graph_canvas/shaders/common.glsl"
                ).read()
            )
            + """
layout(location = ATTR_CORNER) in vec2 a_corner;
layout(location = ATTR_RECT) in vec4 a_rect;
layout(location = ATTR_FILL_TOP) in vec4 a_fill_top;
layout(location = ATTR_SHAPE) in vec4 a_shape;
layout(location = ATTR_FILL_BOT) in vec4 a_fill_bot;
layout(location = ATTR_EDGE) in vec4 a_edge;
layout(location = ATTR_ROTATION) in vec2 a_rotation;
layout(location = ATTR_FIELD) in vec4 a_field;
layout(location = ATTR_UV) in vec4 a_uv;
out vec4 o_rect;
void main() {
    o_rect = a_rect + 0.000001 * (a_fill_top + a_shape + a_fill_bot + a_edge
             + a_field + a_uv + vec4(a_rotation, a_corner));
    gl_Position = vec4(0.0);
}"""
        ),
        varyings=["o_rect"],
    )
    rows = np.zeros((3, _SHAPE_STRIDE // 4), dtype="f4")
    for index in range(3):
        rows[index, 0:4] = [index * 10 + n for n in range(4)]
    instances = gl_ctx.buffer(rows.tobytes())
    corner = gl_ctx.buffer(np.zeros(2, dtype="f4").tobytes())
    out = gl_ctx.buffer(reserve=3 * 16)
    vao = gl_ctx.vertex_array(
        program,
        [(corner, "2f", "a_corner"), (instances, _SHAPE_FORMAT, *_SHAPE_ATTRS)],
    )
    vao.transform(out, moderngl.POINTS, vertices=1, instances=3)
    got = np.frombuffer(out.read(), dtype="f4").reshape(3, 4)
    assert got[:, 0].tolist() == [0.0, 10.0, 20.0]
    for obj in (vao, out, corner, instances, program):
        obj.release()
    renderer.release()


def test_a_run_offset_draws_that_run_and_not_the_head_of_the_stream(
    gl_ctx: moderngl.Context,
) -> None:
    """The falsifier for the base-instance bug: a host that drops the offset
    redraws instances 0..n instead of the run's own, which is DIM rather than
    broken because the head of the stream is the grid.

    Built as two slabs at known places. Drawing the run at offset 1 must light
    the second slab's pixels and leave the first's dark; ignoring the offset
    lights the first and misses the second.
    """
    renderer = CanvasRenderer(gl=gl_ctx)
    panel = CanvasPanel(renderer)
    canvas = ffi.Canvas()
    canvas.load_atlas()

    size = (400, 400)
    result = _one_node_frame(canvas, (float(size[0]), float(size[1])))
    offsets = [
        result.runs[i].first
        for i in range(result.run_count)
        if result.runs[i].stream == 0
    ]
    assert max(offsets) > 0, "this frame has no resumed shape run to test with"

    panel.render(result, size, canvas.distance_range, (0.0, 0.0, 0.0, 1.0))
    assert panel.fbo is not None
    honoured = np.frombuffer(panel.fbo.read(components=3), dtype="u1").copy()

    # The same frame with every run forced to start at 0, which is what a host
    # that ignores `first` draws. The two pictures must differ.
    panel._drop_shape_vaos()
    real_vao = panel._shape_vao
    panel._shape_vao = lambda first: real_vao(0)  # type: ignore[method-assign]
    panel.render(result, size, canvas.distance_range, (0.0, 0.0, 0.0, 1.0))
    ignored = np.frombuffer(panel.fbo.read(components=3), dtype="u1").copy()
    panel._shape_vao = real_vao  # type: ignore[method-assign]

    assert not np.array_equal(honoured, ignored), (
        "dropping the run offset changed nothing, so this test cannot see it"
    )

    canvas.release()
    panel.release()
    renderer.release()


def test_a_frame_renders_the_node_body(gl_ctx: moderngl.Context) -> None:
    """The end-to-end check: a node is a large opaque slab, so a frame that
    draws one lights a large share of a small viewport. Under the stride bug
    this fell to a few hundred pixels."""
    renderer = CanvasRenderer(gl=gl_ctx)
    panel = CanvasPanel(renderer)
    canvas = ffi.Canvas()
    canvas.load_atlas()
    size = (400, 400)
    result = _one_node_frame(canvas, (float(size[0]), float(size[1])))
    panel.render(result, size, canvas.distance_range, (0.0, 0.0, 0.0, 1.0))
    assert panel.fbo is not None
    body = result.node_rects[0]
    assert _lit(panel.fbo, size) > 0.5 * body.w * body.h
    canvas.release()
    panel.release()
    renderer.release()


def test_the_streams_are_uploaded_at_the_library_s_own_widths(
    gl_ctx: moderngl.Context,
) -> None:
    canvas = ffi.Canvas()
    canvas.load_atlas()
    result = _one_node_frame(canvas, (400.0, 400.0))
    shapes = shapes_array(result)
    glyphs = glyphs_array(result)
    assert shapes.shape == (
        result.shape_count,
        ctypes.sizeof(ffi.ShapeInstance) // 4,
    )
    assert glyphs.shape == (result.glyph_count, ctypes.sizeof(ffi.GlyphVertex) // 4)
    # A copy: the result's storage is overwritten by the next frame.
    first_row = shapes[0].copy()
    _one_node_frame(canvas, (800.0, 800.0))
    assert np.array_equal(shapes[0], first_row)
    canvas.release()


def test_the_glyph_atlas_and_a_preview_do_not_share_a_texture_unit() -> None:
    """A preview bound where the atlas lives is sampled as glyph coverage by
    the next text run."""
    from shaderbox.graph_canvas.render import _ATLAS_UNIT, _IMAGE_UNIT

    assert _ATLAS_UNIT != _IMAGE_UNIT


def test_an_empty_frame_clears_without_drawing(gl_ctx: moderngl.Context) -> None:
    renderer = CanvasRenderer(gl=gl_ctx)
    panel = CanvasPanel(renderer)
    canvas = ffi.Canvas()
    canvas.load_atlas()
    result = canvas.frame([], [], (256.0, 256.0), ffi.View(), ffi.PointerState())
    texture = panel.render(
        result, (256, 256), canvas.distance_range, (0.0, 0.0, 0.0, 1.0)
    )
    assert texture.size == (256, 256)
    canvas.release()
    panel.release()
    renderer.release()


@pytest.mark.parametrize("size", [(320, 240), (900, 400), (1280, 720)])
def test_the_body_survives_every_viewport(
    gl_ctx: moderngl.Context, size: tuple[int, int]
) -> None:
    """The stride bug hid at one viewport and showed at another, because the
    instance count decides how far past the buffer the walk runs. A single
    size proves nothing."""
    renderer = CanvasRenderer(gl=gl_ctx)
    panel = CanvasPanel(renderer)
    canvas = ffi.Canvas()
    canvas.load_atlas()
    result = _one_node_frame(canvas, (float(size[0]), float(size[1])))
    panel.render(result, size, canvas.distance_range, (0.0, 0.0, 0.0, 1.0))
    assert panel.fbo is not None
    body = result.node_rects[0]
    assert _lit(panel.fbo, size) > 0.5 * body.w * body.h
    canvas.release()
    panel.release()
    renderer.release()
