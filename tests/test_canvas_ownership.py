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

import json
from collections.abc import Iterator
from pathlib import Path

import imageio.v3 as iio
import moderngl
import pytest

from shaderbox.core import Canvas, Pass
from shaderbox.document import Document
from shaderbox.media import FileDetails, MediaDetails, ResolutionDetails
from shaderbox.pass_graph import (
    MIN_CANVAS_PX,
    PassEntry,
    PassGraph,
    PassSource,
    TargetConfig,
    clamp_canvas_size,
)
from shaderbox.paths import (
    DOCUMENT_JSON_BASENAME,
    GRAPH_JSON_BASENAME,
    PASSES_DIR_NAME,
    pass_shader_name,
    shader_lib_root,
)
from shaderbox.popups.pass_settings import displayed_target_size
from shaderbox.render_preset import FitPolicy, RenderPreset, ResolutionPolicy
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


def test_a_document_built_below_the_floor_is_clamped_like_a_resize(
    gl_ctx: moderngl.Context,
) -> None:
    # `__init__` and `set_canvas_size` are the field's only two writers and the comment above it
    # says both normalize. Only the second one did: a Document built below MIN_CANVAS_PX kept the
    # under-floor size until the first resize, at which point every canvas jumped to a size
    # nobody asked for. `load_from_dir` passes the caller's size straight through.
    document = Document(gl=gl_ctx, canvas_size=(8, 8))

    assert document.canvas_size == clamp_canvas_size((8, 8))
    assert document.canvas_size == (MIN_CANVAS_PX, MIN_CANVAS_PX)
    document.release()


def test_adding_and_renaming_a_pass_keeps_every_canvas_agreeing(
    gl_ctx: moderngl.Context,
) -> None:
    # Both are named in W-4's operation battery. They hold for a structural reason rather than by
    # a rule either one applies -- a new pass is built at the document's size with a default
    # target, and a rename re-keys the same Pass object -- so this pins that reason rather than
    # leaving the two operations untested on the assumption it stays true.
    document = _document(gl_ctx)

    fresh = Pass(gl=gl_ctx, canvas_size=_CANVAS, target=TargetConfig())
    fresh.release_program(_PLAIN)
    fresh.compile()
    document.passes["added"] = fresh
    document.graph = document.graph.with_passes(
        {**document.graph.passes, "added": PassEntry()}
    )
    assert_canvases_agree(document)

    # Rename the OUTPUT, the case where the size rule's answer depends on the name.
    renamed = document.passes.pop("main")
    document.passes["shown"] = renamed
    entries = {
        name: entry for name, entry in document.graph.passes.items() if name != "main"
    }
    document.graph = document.graph.with_passes(
        {**entries, "shown": PassEntry()}, output="shown"
    )

    assert_canvases_agree(document)
    assert document.render_pass.canvas.texture.size == _CANVAS
    document.release()


def test_a_scale_change_reaches_an_off_chain_passs_canvas(
    gl_ctx: moderngl.Context,
) -> None:
    """The size rule's INPUT can move, and the canvas has to follow it.

    `Pass.set_target` keeps the canvas's size by decision, so the only other writer is `render`'s
    lazy fix-up -- which a pass outside the output chain never reaches. Without the resize in
    `set_pass_target` the canvas kept its old size forever while the graph and the settings modal
    both reported the new one.
    """
    document = _document(gl_ctx)
    # `off` is read by nothing, so no render will ever visit it.
    off = Pass(gl=gl_ctx, canvas_size=_CANVAS, target=TargetConfig())
    off.release_program(_PLAIN)
    off.compile()
    document.passes["off"] = off
    document.graph = document.graph.with_passes(
        {**document.graph.passes, "off": PassEntry()}
    )
    document.set_canvas_size(_CANVAS)
    assert_canvases_agree(document)

    scaled = TargetConfig(scale=0.5)
    document.graph = document.graph.with_target("off", scaled)
    document.set_pass_target("off", scaled)

    assert_canvases_agree(document)
    assert document.passes["off"].canvas.texture.size == (32, 32)
    document.release()


def test_demoting_the_output_applies_the_scale_it_was_ignoring(
    gl_ctx: moderngl.Context,
) -> None:
    # The mirror of the promotion case: the output ignores its own scale, so the pass LEAVING the
    # role has been full-size and that scale applies again the moment it is not the output.
    document = _document(gl_ctx)
    document.set_output_pass("helper")
    assert_canvases_agree(document)

    document.set_output_pass("main")

    assert_canvases_agree(document)
    assert document.passes["helper"].canvas.texture.size == (32, 32)
    document.release()


def test_a_render_repairs_a_pass_born_at_the_wrong_size(
    gl_ctx: moderngl.Context,
) -> None:
    """`render`'s lazy fix-up, driven through the one state that actually needs it.

    A pass IN the output chain whose canvas disagrees with its entry: nothing else has conformed
    it, so the fix-up branch is the only thing that can. Asserting after a render on an already
    conformed document exercises nothing -- a canary raised inside that branch was never reached
    by the earlier shape of this test.
    """
    document = _document(gl_ctx)
    # Put `helper` back at full size behind the document's back, as a birth would.
    document.passes["helper"].canvas.set_size(_CANVAS)
    assert canvas_violations(document), (
        "the setup did not create the disagreement it needs"
    )

    document.begin_frame(0)
    document.render()

    assert_canvases_agree(document)
    assert document.passes["helper"].canvas.texture.size == (32, 32)
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
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    """The canvas `render_media` ALLOCATES, on the branch that ships.

    `FitPolicy.RENDER_AT_TARGET` is what `render_shape.py` and `exporters/telegram.py` take,
    and its canvas is handed to the output pass as its draw target -- so the canvas's format IS
    the pass's format for the whole export, and the loss is at WRITE time. Inspecting the output
    file finds nothing, since the readback tonemaps a float target to 8-bit either way.

    The canvas is captured from inside the real call rather than rebuilt here: a test that
    constructs its own copy asserts a fact about `Canvas.__init__` and passes with the defect
    present -- measured, F1 reintroduced verbatim survived exactly that shape.
    """
    document = _document(gl_ctx)
    loud = TargetConfig(dtype="f4", filter_linear=False, wrap=True)
    document.graph = document.graph.with_target("main", loud)
    document.set_pass_target("main", loud)
    output = document.render_pass.canvas

    seen: dict[str, object] = {}
    original = Document._render_media_into

    def capture(self: Document, details: MediaDetails, canvas: Canvas) -> MediaDetails:
        seen["dtype"] = canvas.dtype
        seen["filter"] = canvas.filter
        seen["wrap"] = canvas.wrap
        return details

    details = MediaDetails(
        is_video=False,
        file_details=FileDetails(path=str(tmp_path / "out.png"), size=0),
        resolution_details=ResolutionDetails(width=32, height=32),
    )
    preset = RenderPreset(
        resolution_policy=ResolutionPolicy.FIXED_DIMS,
        target_w=32,
        target_h=32,
        fit=FitPolicy.RENDER_AT_TARGET,
    )
    try:
        Document._render_media_into = capture  # type: ignore[method-assign]
        document.render_media(details, preset)
    finally:
        Document._render_media_into = original  # type: ignore[method-assign]

    assert seen == {
        "dtype": output.dtype,
        "filter": output.filter,
        "wrap": output.wrap,
    }, "the export's fit branch did not carry the output pass's format"
    document.release()


def test_a_self_reading_output_exports_a_chain_that_advances(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    """The frozen export (F6), judged on the DECODED VIDEO.

    An output pass reading its own previous frame advanced correctly in the viewer and exported
    a video where every frame held the same accumulated value. The last iteration drew into the
    caller's canvas, which left the pass's own canvas unwritten, so the frame-boundary swap had
    nothing to advance.

    The decoding is load-bearing and is why this is not a canvas assertion: within a single frame
    the external canvas holds a CORRECT value, and comparing it against a no-external-canvas
    reference is what led one investigation agent to report this defect as sound. The failure is
    only visible ACROSS frames, in the file that was written.

    Both iteration counts: the chain is frozen at every N, not only at N=1.
    """
    accumulate = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_prev;
out vec4 fs_color;
void main() { fs_color = texture(u_prev, vs_uv) + vec4(0.05, 0.0, 0.0, 1.0); }
"""
    for iterations in (1, 2):
        document = Document(gl=gl_ctx, canvas_size=(32, 32))
        for render_pass in list(document.passes.values()):
            render_pass.release()
        document.passes = {}
        accumulator = Pass(gl=gl_ctx, canvas_size=(32, 32), target=TargetConfig())
        accumulator.release_program(accumulate)
        accumulator.compile()
        document.passes["acc"] = accumulator
        accumulator.uniform_values["u_prev"] = PassSource("acc")
        document.graph = PassGraph(
            output="acc", passes={"acc": PassEntry(iterations=iterations)}
        )

        out = tmp_path / f"acc_{iterations}.mp4"
        document.render_media(
            MediaDetails(
                is_video=True,
                file_details=FileDetails(path=str(out), size=0),
                resolution_details=ResolutionDetails(width=32, height=32),
                fps=10,
                duration=0.6,
            )
        )
        reds = [int(frame[:, :, 0].max()) for frame in iio.imread(out)]
        assert len(set(reds)) > 1, (
            f"iterations={iterations}: every exported frame is {reds[0]} -- "
            f"the feedback chain never advanced ({reds})"
        )
        assert reds == sorted(reds), f"iterations={iterations}: not climbing ({reds})"
        document.release()


def test_a_pass_file_with_no_graph_entry_matches_the_entry_it_gets(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # `load_from_dir` backfills the graph with PassEntry() for a file with no entry, whose target
    # is f2 -- while the canvas was built from None and took Canvas's own f1. The panel then
    # showed f2 over an f1 canvas, and a save wrote a graph its own reload rejects.
    document_dir = tmp_path / "doc"
    (document_dir / PASSES_DIR_NAME).mkdir(parents=True)
    (document_dir / PASSES_DIR_NAME / pass_shader_name("solo")).write_text(_PLAIN)
    (document_dir / GRAPH_JSON_BASENAME).write_text(
        json.dumps({"output": "solo", "passes": {}})
    )
    (document_dir / DOCUMENT_JSON_BASENAME).write_text(
        json.dumps({"uniforms": {}, "ui_state": {}})
    )

    document, _ = Document.load_from_dir(document_dir, gl=gl_ctx, canvas_size=_CANVAS)

    assert_canvases_agree(document)
    entry = document.graph.passes["solo"]
    assert document.passes["solo"].canvas.dtype == entry.target.dtype
    document.release()


def test_the_settings_label_shows_the_size_the_canvas_has(
    gl_ctx: moderngl.Context,
) -> None:
    # The fifth copy of the sizing rule, and the only one that was wrong: it applied `scale` with
    # no output exemption, so an output pass carrying a stored scale was shown a size its texture
    # did not have. Display-only, and still the rule this feature exists to keep in one shape.
    scaled = TargetConfig(scale=0.5)

    assert displayed_target_size(scaled, (256, 256), is_output=False) == (128, 128)
    assert displayed_target_size(scaled, (256, 256), is_output=True) == (256, 256)


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
