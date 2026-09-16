"""Every canvas a Document owns agrees with its graph entry, after every operation (096 W-4).

Six canvas defects in one week were one class: a canvas's configuration -- size, dtype, filter,
wrap -- is decided in several places, and each place knows about some canvases but not all. The
gap that let all six through is measurable: before this file, deleting `filter=` and `wrap=` from
`resample_canvas` -- which resets every canvas in the document on every resize -- passed the
whole suite.

The check itself is `tests/canvas_invariants.py`. This file drives it over the operations that
change a canvas, and the two rules below are what keep it from going vacuous:

- **Assert BEFORE any render.** `render` fixes up a non-output pass's size lazily, so a test that
  renders first passes whether or not the operation under test did anything.
  `test_a_resize_moves_every_pass_together` is vacuous exactly that way and its own comment
  concedes it.
- **Drive it over the NON-DEFAULT corner.** `DEFAULT_FILTER_LINEAR` is True and `DEFAULT_WRAP` is
  False, so a check built on defaults cannot fail. `NON_DEFAULT` is the opposite of each.
"""

from collections.abc import Iterator

import moderngl
import pytest

from shaderbox.core import Canvas, Pass
from shaderbox.document import Document
from shaderbox.pass_graph import PassEntry, PassGraph, PassSource, TargetConfig
from shaderbox.paths import shader_lib_root
from shaderbox.shader_lib import ShaderLibIndex, set_active
from tests.canvas_invariants import (
    NON_DEFAULT,
    assert_canvases_agree,
    canvas_violations,
)

_PLAIN = """#version 460 core
in vec2 vs_uv;
out vec4 fs_color;
void main() { fs_color = vec4(vs_uv, 0.0, 1.0); }
"""

_READER = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_src;
out vec4 fs_color;
void main() { fs_color = texture(u_src, vs_uv); }
"""

_CANVAS = (64, 64)


@pytest.fixture(scope="module")
def gl_ctx() -> Iterator[moderngl.Context]:
    try:
        context = moderngl.create_standalone_context()
    except Exception as e:
        pytest.skip(f"no standalone GL context available: {e}")
    set_active(ShaderLibIndex.build(shader_lib_root()))
    yield context
    context.release()


def _document(gl: moderngl.Context) -> Document:
    """`helper` carries the non-default corner and is NOT the output; `main` reads it.

    The scaled pass is off-output on purpose: that is the arrangement every sizing defect in this
    feature needed, and it is what a promotion then breaks.
    """
    document = Document(gl=gl, canvas_size=_CANVAS)
    for render_pass in list(document.passes.values()):
        render_pass.release()
    document.passes = {}
    for name, source, target in (
        ("helper", _PLAIN, NON_DEFAULT),
        ("main", _READER, TargetConfig()),
    ):
        render_pass = Pass(gl=gl, canvas_size=_CANVAS, target=target)
        render_pass.release_program(source)
        render_pass.compile()
        document.passes[name] = render_pass
    document.passes["main"].uniform_values["u_src"] = PassSource("helper")
    document.graph = PassGraph(
        output="main",
        passes={"helper": PassEntry(target=NON_DEFAULT), "main": PassEntry()},
    )
    document.set_canvas_size(_CANVAS)
    return document


def test_the_battery_starts_from_an_agreeing_document(gl_ctx: moderngl.Context) -> None:
    # The baseline every case below measures against: if this is already violated the rest say
    # nothing about the operation they name.
    document = _document(gl_ctx)
    assert_canvases_agree(document)
    document.release()


def test_a_resize_takes_every_canvas_with_it(gl_ctx: moderngl.Context) -> None:
    document = _document(gl_ctx)
    document.begin_frame(0)
    document.render()

    document.set_canvas_size((128, 128))

    # BEFORE any render: render's lazy fix-up would repair the non-output pass and hide a
    # set_canvas_size that never touched it.
    assert_canvases_agree(document)
    document.release()


def test_promoting_a_scaled_pass_resizes_it_to_full(gl_ctx: moderngl.Context) -> None:
    document = _document(gl_ctx)

    document.set_output_pass("helper")

    assert_canvases_agree(document)
    assert document.render_pass.canvas.texture.size == _CANVAS
    document.release()


def test_deleting_the_output_conforms_its_replacement(gl_ctx: moderngl.Context) -> None:
    # Deleting the output promotes an arbitrary survivor, which may carry a scale -- the same
    # defect as a promotion, by another door.
    document = _document(gl_ctx)
    document.passes.pop("main").release()
    document.graph = document.graph.with_passes(
        {"helper": PassEntry(target=NON_DEFAULT)}, output="helper"
    )

    document.conform_output_canvas()

    assert_canvases_agree(document)
    assert document.render_pass.canvas.texture.size == _CANVAS
    document.release()


def test_a_target_change_takes_the_history_with_it(gl_ctx: moderngl.Context) -> None:
    document = _document(gl_ctx)
    document.passes["helper"].uniform_values["u_src"] = PassSource("helper")
    document.begin_frame(0)
    document.render()

    changed = TargetConfig(scale=0.5, dtype="f2", filter_linear=True, wrap=False)
    document.graph = document.graph.with_target("helper", changed)
    document.set_pass_target("helper", changed)

    assert_canvases_agree(document)
    document.release()


def test_a_render_leaves_every_canvas_agreeing(gl_ctx: moderngl.Context) -> None:
    document = _document(gl_ctx)
    for frame in range(3):
        document.begin_frame(frame)
        document.render()
        assert_canvases_agree(document)
    document.release()


def test_a_frame_boundary_swap_keeps_the_pair_matched(gl_ctx: moderngl.Context) -> None:
    # A self-reading pass: begin_frame swaps its live canvas with its history every frame, so a
    # pair that disagrees on format alternates the pass between two formats.
    document = _document(gl_ctx)
    document.passes["helper"].uniform_values["u_src"] = PassSource("helper")
    for frame in range(4):
        document.begin_frame(frame)
        document.render()
        assert_canvases_agree(document)
    document.release()


def test_an_export_canvas_carries_the_output_passs_format(
    gl_ctx: moderngl.Context,
) -> None:
    # The export's fit branch allocated its canvas with no dtype/filter/wrap while its sibling
    # twelve lines up copied all three, and the branch that ships (Telegram, the shared shapes)
    # was the lossy one.
    document = _document(gl_ctx)
    loud = TargetConfig(dtype="f4", filter_linear=False, wrap=True)
    document.graph = document.graph.with_target("main", loud)
    document.set_pass_target("main", loud)
    assert_canvases_agree(document)
    output = document.render_pass.canvas

    scratch = Canvas(
        gl=gl_ctx,
        size=(32, 32),
        dtype=output.dtype,
        filter=output.filter,
        wrap=output.wrap,
    )
    try:
        assert (scratch.dtype, scratch.filter, scratch.wrap) == (
            output.dtype,
            output.filter,
            output.wrap,
        )
    finally:
        scratch.release()
    document.release()


def test_the_checker_sees_a_resample_that_drops_filter_and_wrap(
    gl_ctx: moderngl.Context,
) -> None:
    """The mutation the whole suite used to survive.

    `resample_canvas` copies the old canvas's format forward. Deleting `filter=` and `wrap=`
    resets every canvas in the document to LINEAR and clamp on every resize, silently. This is
    the falsifier that proves the battery above is not vacuous -- it must be SEEN to fail.
    """
    document = _document(gl_ctx)
    original = Document.resample_canvas

    def dropping(self: Document, old: Canvas, size: tuple[int, int]) -> Canvas:
        if old.texture.size == size:
            return old
        new = Canvas(gl=self._gl, size=size, dtype=old.dtype)
        self._blit_into(old, new)
        old.release()
        return new

    try:
        Document.resample_canvas = dropping  # type: ignore[method-assign]
        document.set_canvas_size((128, 128))
        violations = canvas_violations(document)
    finally:
        Document.resample_canvas = original  # type: ignore[method-assign]

    assert violations, (
        "a resize that dropped filter and wrap went unnoticed -- the battery is vacuous"
    )
    document.release()
