"""The Steps surface: the engine's read-only view, and the float-target tonemap.

Two seams, tested apart from the panel that uses them:

- `Node.step_views()` is the UI's whole window onto the chain. Keeping the ping-pong
  pair and the program map private behind it is what lets either side be rewritten.
- `StepPreview` decides HOW a float step is shown. A raw blit of a target holding 7.0
  throws away most of the detail, and every step worth debugging is float.
"""

from pathlib import Path

import moderngl
import numpy as np
import pytest

from shaderbox.core import Node
from shaderbox.media import texture_to_rgba8
from shaderbox.paths import shader_lib_root
from shaderbox.shader_lib import ShaderLibIndex, set_active
from shaderbox.shader_source import ShaderSource
from shaderbox.step_preview import StepPreview

_CHAIN = (
    "#version 330\n"
    "out vec4 f_color;\n"
    "in vec2 vs_uv;\n"
    "uniform sampler2D u_a;      // step, f4\n"
    "uniform sampler2D u_half;   // step, scale: 0.5, f2, nearest, repeat\n"
    "uniform sampler2D u_loop;   // step, f4, persist\n"
    "void step_a(out vec4 o) { o = vec4(1.0); }\n"
    "void step_half(out vec4 o) { o = texture(u_a, vs_uv); }\n"
    "void step_loop(out vec4 o) { o = texture(u_loop, vs_uv) + vec4(0.1); }\n"
    "void main() { f_color = texture(u_half, vs_uv) + texture(u_loop, vs_uv); }\n"
)


@pytest.fixture(scope="module")
def gl() -> moderngl.Context:
    ctx = moderngl.create_standalone_context(require=460)
    set_active(ShaderLibIndex.build(shader_lib_root()))
    return ctx


def _node(gl: moderngl.Context, tmp_path: Path, text: str) -> Node:
    path = tmp_path / "n.frag.glsl"
    path.write_text(text, encoding="utf-8")
    node = Node(gl=gl, source=ShaderSource.load(path), canvas_size=(32, 32))
    node.compile()
    return node


def test_a_step_free_node_has_no_views(gl: moderngl.Context, tmp_path: Path) -> None:
    # The section must not render at all for the overwhelming majority of nodes -- not
    # even an empty header.
    node = _node(
        gl, tmp_path, "#version 330\nout vec4 f;\nvoid main() { f = vec4(1.0); }\n"
    )
    assert node.step_views() == []


def test_views_carry_every_fact_the_panel_shows(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    node = _node(gl, tmp_path, _CHAIN)
    assert node.compile_unit.errors == [], node.compile_unit.errors
    node.render(u_time=0.0)
    views = {v.name: v for v in node.step_views()}

    assert views["half"].size == (16, 16)
    assert views["half"].dtype == "f2"
    assert views["half"].filter_linear is False
    assert views["half"].wrap is True
    assert views["half"].reads == ["a"]
    assert views["half"].sampler == "u_half"

    assert views["loop"].persist is True
    assert views["loop"].reads_self is True
    assert views["a"].reads == []
    # `main()` reads half and loop, not `a`.
    assert views["half"].read_by_output is True
    assert views["a"].read_by_output is False


def test_views_are_in_evaluation_order(gl: moderngl.Context, tmp_path: Path) -> None:
    node = _node(gl, tmp_path, _CHAIN)
    views = node.step_views()
    assert [v.name for v in views] == node.step_plan.order
    assert [v.order_index for v in views] == list(range(len(views)))


def test_views_report_a_size_before_the_first_render(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # Targets are allocated on the first render, but the panel draws before that on the
    # frame a node is selected -- the row must still show a real size, not 0x0.
    node = _node(gl, tmp_path, _CHAIN)
    views = {v.name: v for v in node.step_views()}
    assert views["half"].size == (16, 16)
    assert views["half"].texture_glo is None


def test_an_eight_bit_texture_is_passed_through_untouched(
    gl: moderngl.Context,
) -> None:
    # The final composite and any f1 step must look exactly as they do today; routing
    # them through the tonemap would change every existing node.
    source = gl.texture((8, 8), 4, dtype="f1")
    assert StepPreview(gl).texture_for(source) is source


def test_a_float_target_keeps_detail_a_raw_blit_would_lose(
    gl: moderngl.Context,
) -> None:
    # The R7 x R9 gap: a cascade level holds values far above 1.0, so a straight blit
    # crushes the bright end to a few levels. 063 measured f2 reaching 7.0 where f1
    # saturated at 255 on the first pass -- this is that range.
    width = height = 32
    payload = np.zeros((height, width, 4), dtype=np.float32)
    for x in range(width):
        payload[:, x, :3] = 8.0 * x / (width - 1)
    payload[:, :, 3] = 1.0
    source = gl.texture((width, height), 4, data=payload.tobytes(), dtype="f4")

    raw = np.frombuffer(source.read(), dtype=np.float32).reshape(height, width, 4)
    raw_levels = len(np.unique((np.clip(raw[:, :, 0], 0, 1) * 255).astype(np.uint8)))
    toned = texture_to_rgba8(StepPreview(gl).texture_for(source))
    toned_levels = len(np.unique(toned[:, :, 0]))

    assert toned_levels > raw_levels * 3
    # And the bright end must not be a flat white plateau: the last column still reads
    # brighter than the middle, rather than both sitting at 255.
    assert int(toned[:, -1, 0].mean()) > int(toned[:, width // 2, 0].mean())


def test_the_preview_reuses_one_canvas_across_calls(gl: moderngl.Context) -> None:
    preview = StepPreview(gl)
    a = gl.texture((8, 8), 4, dtype="f2")
    first = preview.texture_for(a)
    second = preview.texture_for(a)
    assert first is second, "a new canvas per frame would churn GL objects"


def test_the_preview_follows_a_size_change(gl: moderngl.Context) -> None:
    preview = StepPreview(gl)
    small = preview.texture_for(gl.texture((8, 8), 4, dtype="f2"))
    assert small.size == (8, 8)
    large = preview.texture_for(gl.texture((16, 16), 4, dtype="f2"))
    assert large.size == (16, 16)


def test_the_preview_allocates_nothing_until_a_float_step_is_viewed(
    gl: moderngl.Context,
) -> None:
    preview = StepPreview(gl)
    preview.texture_for(gl.texture((8, 8), 4, dtype="f1"))
    assert preview._program is None, "an 8-bit-only session must pay nothing"
    preview.texture_for(gl.texture((8, 8), 4, dtype="f2"))
    assert preview._program is not None
    preview.release()
    assert preview._program is None
