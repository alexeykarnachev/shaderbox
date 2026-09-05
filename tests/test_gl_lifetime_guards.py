"""Guards over release/reuse decisions that no test observed.

Each of these survived a mutation with the whole suite green: delete the line and 700+ tests
still pass while a GL object leaks every frame, or a revert restores the wrong file. They are
cheap to write and the thing they protect is invisible until it is expensive — a leak shows up
as the app slowly eating VRAM, not as a failure anyone can point at.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import moderngl
import pytest

from shaderbox.copilot.checkpoint import TurnCheckpoint
from shaderbox.core import Canvas, Pass
from shaderbox.document import Document
from shaderbox.pass_graph import PassEntry, PassGraph, TargetConfig
from shaderbox.paths import DOCUMENT_SCRIPT_BASENAME, shader_lib_root
from shaderbox.shader_lib import ShaderLibIndex, set_active

_EXAMPLE = (
    Path(__file__).resolve().parent.parent
    / "shaderbox/resources/document_examples/53724dbd-8efb-4c09-8c7d-28d626a066e7"
)


@pytest.fixture(scope="module")
def gl() -> moderngl.Context:
    context = moderngl.create_standalone_context(require=460)
    set_active(ShaderLibIndex.build(shader_lib_root()))
    return context


def _released(obj: object) -> bool:
    """True when the GL object has been freed.

    `.glo` keeps its stale integer id after release, so it cannot be the probe. moderngl
    swaps the underlying handle for an `InvalidObject` instead, and that holds for every
    object type (texture, framebuffer, buffer, program, vao) — unlike raising, which only
    some of them do.
    """
    return type(getattr(obj, "mglo", None)).__name__ == "InvalidObject"


def test_document_release_frees_the_canvas(gl: moderngl.Context) -> None:
    # The one test naming Document.release checks only the uniform-values half, so dropping
    # `self.canvas.release()` leaked a texture + FBO per reload with the suite still green.
    document, _ = Document.load_from_dir(_EXAMPLE)
    texture = document.render_pass.canvas.texture
    fbo = document.render_pass.canvas.fbo

    document.release()

    assert _released(texture), "the canvas texture outlived Document.release()"
    assert _released(fbo), "the canvas framebuffer outlived Document.release()"


def test_invalidate_frees_the_program_and_its_buffers(gl: moderngl.Context) -> None:
    # invalidate() runs on every hot-reload and every lib change; leaking here leaks per edit.
    document, _ = Document.load_from_dir(_EXAMPLE)
    document.render()
    assert document.render_pass.program is not None
    program, vbo, vao = (
        document.render_pass.program,
        document.render_pass.vbo,
        document.render_pass.vao,
    )

    document.render_pass.invalidate()

    assert (
        document.render_pass.program is None
        and document.render_pass.vbo is None
        and document.render_pass.vao is None
    )
    assert _released(program), "the GL program outlived invalidate()"
    for name, obj in (("vbo", vbo), ("vao", vao)):
        if obj is not None:
            assert _released(obj), f"the {name} outlived invalidate()"


def test_canvas_resize_frees_the_old_texture(gl: moderngl.Context) -> None:
    canvas = Canvas(size=(64, 64))
    old_texture, old_fbo = canvas.texture, canvas.fbo

    assert canvas.set_size((128, 128))

    assert canvas.texture is not old_texture
    assert _released(old_texture), "the pre-resize texture leaked"
    assert _released(old_fbo), "the pre-resize framebuffer leaked"


def test_canvas_resize_to_the_same_size_is_a_no_op(gl: moderngl.Context) -> None:
    # set_size is called every frame with the current size; without the guard that is a
    # texture+FBO reallocation per frame, which no test would otherwise notice.
    canvas = Canvas(size=(64, 64))
    texture = canvas.texture

    assert canvas.set_size((64, 64)) is False

    assert canvas.texture is texture, "a same-size resize reallocated the canvas"


def test_snapshot_script_keeps_the_pre_turn_bytes(tmp_path: Path) -> None:
    # First-touch-wins: a second edit in the same turn must NOT overwrite the snapshot, or
    # Revert restores the copilot's own mid-turn draft over the user's pre-turn script —
    # reporting success while losing exactly what the user asked to get back.
    scripts_dir = tmp_path / "document" / "scripts"
    scripts_dir.mkdir(parents=True)
    script = scripts_dir / DOCUMENT_SCRIPT_BASENAME
    script.write_text("PRE-TURN\n")

    checkpoint = TurnCheckpoint(turn_id="t1", root=tmp_path / "checkpoints")
    checkpoint.snapshot_script("document", script)
    script.write_text("MID-TURN DRAFT\n")
    checkpoint.snapshot_script("document", script)

    snapshot = checkpoint.turn_dir / "document" / "scripts" / DOCUMENT_SCRIPT_BASENAME
    assert snapshot.read_text() == "PRE-TURN\n", (
        "the second snapshot overwrote the pre-turn bytes — revert would restore the draft"
    )


def test_snapshot_script_ignores_a_missing_script(tmp_path: Path) -> None:
    checkpoint = TurnCheckpoint(turn_id="t1", root=tmp_path / "checkpoints")
    checkpoint.snapshot_script(
        "document", tmp_path / "absent" / DOCUMENT_SCRIPT_BASENAME
    )
    assert not (checkpoint.turn_dir / "document").exists()


def test_document_release_frees_uniform_held_media(gl: moderngl.Context) -> None:
    # The 060 fix, pinned from the other side: the uniform values own textures/captures, and
    # every reload releases the document.
    document, _ = Document.load_from_dir(_EXAMPLE)
    # A texture is PUT there rather than hoped for. The shipped example this loads holds only
    # float uniforms, so the release loop below ran zero times and the test passed no matter
    # what `Pass.release()` did -- found by a sweep, and it is the reason the assertion count is
    # asserted too.
    document.render_pass.uniform_values["u_probe"] = gl.texture((2, 2), 4)
    held = [
        v
        for v in document.render_pass.uniform_values.values()
        if isinstance(v, moderngl.Texture)
    ]
    assert held, "nothing texture-shaped to release -- the check would be vacuous"

    document.release()

    assert document.render_pass.uniform_values == {}
    for texture in held:
        assert _released(texture), "a uniform-held texture outlived Document.release()"


# --- feedback-history lifecycle (068 machinery, found by the repo-wide sweep) ---


def _feedback_document(gl: moderngl.Context) -> Document:
    # One self-feeding pass, which is what allocates a feedback history at all.
    src = (
        "#version 460 core\nin vec2 vs_uv;\nuniform sampler2D u_prev;\nout vec4 fs_color;\n"
        "void main(){ fs_color = texture(u_prev, vs_uv) + vec4(0.1, 0.0, 0.0, 1.0); }\n"
    )
    doc = Document(gl=gl, canvas_size=(8, 8))
    for existing in list(doc.passes.values()):
        existing.release()
    doc.passes = {}
    render_pass = Pass(gl=gl, canvas_size=(8, 8), target=TargetConfig(dtype="f1"))
    render_pass.release_program(src)
    render_pass.compile()
    doc.passes["fb"] = render_pass
    doc.graph = PassGraph(
        output="fb",
        passes={"fb": PassEntry(target=TargetConfig(dtype="f1"))},
    )
    doc.begin_frame(0)
    doc.render()
    return doc


def test_dropping_a_pass_releases_its_feedback_history(gl: moderngl.Context) -> None:
    # The history is keyed by NAME and owned by the Document, so releasing the Pass does not
    # release it. Falsifier: remove the drop_feedback call in delete_pass and the canvas stays
    # in the dict, reachable by nothing, for the life of the document.
    doc = _feedback_document(gl)
    assert "fb" in doc._feedback
    doc.drop_feedback("fb")
    assert "fb" not in doc._feedback
    doc.release()


def test_a_target_format_change_never_leaves_the_pair_disagreeing(
    gl: moderngl.Context,
) -> None:
    # The one that corrupts rather than leaks. A target change reallocates the pass's LIVE
    # canvas; `begin_frame` then SWAPS it into the history, so the pair ends up holding one
    # canvas of each format and the pass samples its own previous frame through the wrong one --
    # no error, no crash, wrong numbers. The invariant is that the two never disagree, which is
    # what a shader reading `u_prev` depends on. Asserting a specific dtype would be asserting
    # which side of the swap won, which is not the property that matters.
    #
    # Falsifier: drop the target_generation check in `_feedback_canvas` and the pair splits
    # f1/f2 on the frame after the change.
    doc = _feedback_document(gl)
    assert doc._feedback["fb"].dtype == doc.passes["fb"].canvas.dtype

    doc.passes["fb"].set_target(TargetConfig(dtype="f2"))
    for frame in range(1, 4):
        doc.begin_frame(frame)
        doc.render()
        assert doc._feedback["fb"].dtype == doc.passes["fb"].canvas.dtype, (
            f"frame {frame}: history {doc._feedback['fb'].dtype} vs live "
            f"{doc.passes['fb'].canvas.dtype} -- the pass reads its history through the "
            f"wrong format"
        )
    doc.release()


@pytest.fixture
def stale_default_context() -> Iterator[moderngl.Context]:
    # create_standalone_context installs its context as moderngl's process-wide default; a
    # later App would inherit that wrapper. Requested BEFORE `app` in a test's signature so it
    # runs first.
    stale = moderngl.create_standalone_context(require=460)
    assert moderngl.get_context() is stale
    yield stale
    stale.release()


def test_the_app_rebinds_the_default_context_to_its_window(
    stale_default_context: moderngl.Context, app: Any
) -> None:
    # moderngl binds its process-wide default context ONCE, to whatever GL context is current at
    # the first get_context(). A standalone fixture context that took that role leaves a wrapper
    # every later App would inherit: module-order-only failures (an export rendered garbage after
    # test_document_graph + test_default_wiring). App.__init__ therefore re-initialises the default
    # to its own window. Falsifier: drop that init_context() call -- `after` is then the very
    # object the stale fixture handed out.
    after = moderngl.get_context()
    assert after is not stale_default_context
    assert app.ui_documents  # the App built its documents on the rebound context
