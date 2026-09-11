"""An Auto resize keeps the picture, on the live canvas AND on every feedback history (090 D4).

The trap this exists for is silent by construction. `Canvas.set_size` is release-then-allocate,
so resizing the LIVE canvas blanks it; the next `_swap_feedback` trades that blank into the
history, and a self-reading pass samples black one frame later -- a document that looks right on
the resize frame and wrong on the one after. Resampling the history alone has the same outcome,
which is why both halves are asserted here.

Read through `media.texture_to_rgba8`, never `texture.read()[0]`: a graph entry's default target
is `f2`, where a raw byte read returns half of one float16 channel as a plausible small integer.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import moderngl
import pytest

from shaderbox.media import texture_to_rgba8
from shaderbox.paths import (
    DOCUMENT_JSON_BASENAME,
    GRAPH_JSON_BASENAME,
    PASSES_DIR_NAME,
    pass_shader_name,
    shader_lib_root,
)
from shaderbox.shader_lib import ShaderLibIndex, set_active
from shaderbox.ui_models import UIDocument, load_document_from_dir

# A pass that reads itself and draws a NON-UNIFORM picture: bright in one corner, dark in the
# other. A uniform fill cannot see a 1:1 corner copy (which is what `copy_framebuffer` does
# between differently-sized framebuffers -- measured), because every texel already agrees.
_TRAIL = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_prev;
out vec4 fs_color;
void main() {
    float seeded = max(texture(u_prev, vs_uv).r, vs_uv.x * vs_uv.y);
    fs_color = vec4(seeded, 0.0, 0.0, 1.0);
}
"""

# The OUTPUT, and it feeds back too (`u_prev` reads the pass's own previous frame): the output
# canvas is the one `set_canvas_size` used to blank, and its history is what the next swap
# trades that blank into -- the exact pair D4 exists for.
_SHOW = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_trail;
uniform sampler2D u_prev;
out vec4 fs_color;
void main() {
    fs_color = vec4(max(texture(u_trail, vs_uv).r, texture(u_prev, vs_uv).r), 0.0, 0.0, 1.0);
}
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


def _write_document(dir: Path) -> Path:
    """`trail` feeds back at scale 0.5 and is not the output; `show` is, and feeds back too."""
    passes = dir / PASSES_DIR_NAME
    passes.mkdir(parents=True, exist_ok=True)
    (passes / pass_shader_name("trail")).write_text(_TRAIL)
    (passes / pass_shader_name("show")).write_text(_SHOW)
    (dir / GRAPH_JSON_BASENAME).write_text(
        json.dumps(
            {
                "output": "show",
                "passes": {
                    "trail": {"target": {"scale": 0.5}},
                    "show": {"target": {}},
                },
            }
        )
    )
    (dir / DOCUMENT_JSON_BASENAME).write_text(
        json.dumps(
            {
                "uniforms": {},
                # FIXED, so the document's live size is the pair this fixture names rather
                # than whatever region the viewer would have handed an Auto document (090
                # revision 1). The resample under test is the same either way.
                "ui_state": {"resolution_mode": "fixed", "resolution": list(_CANVAS)},
            }
        )
    )
    return dir


def _loaded(tmp_path: Path) -> UIDocument:
    ui_document = load_document_from_dir(_write_document(tmp_path / "doc"))
    for frame in range(4):
        ui_document.document.begin_frame(frame)
        ui_document.document.render(u_time=float(frame))
    return ui_document


def _far_corner(texture: moderngl.Texture) -> int:
    # The texel a 1:1 corner copy leaves black: the source's bright end, at the far edge of the
    # destination. A mean over the whole image passes under that bug; this does not.
    image = texture_to_rgba8(texture)
    height, width = image.shape[0], image.shape[1]
    return int(image[height - 1, width - 1, 0])


class _TextureCounts:
    """Textures the context handed out against the ones handed back.

    ALLOCATIONS against RELEASES, not a live total: the textures the document already held when
    counting started were created before the wrapper existed, so counting only what it sees
    would score their replacement as growth. A resize that allocates and releases in step is what
    the counts agreeing says.
    """

    def __init__(self) -> None:
        self.allocated: int = 0
        self.released: int = 0


def _count_textures(
    monkeypatch: pytest.MonkeyPatch, gl: moderngl.Context
) -> _TextureCounts:
    counts = _TextureCounts()
    real_texture = gl.texture

    def counted(*args: Any, **kwargs: Any) -> moderngl.Texture:
        texture = real_texture(*args, **kwargs)
        counts.allocated += 1
        real_release = texture.release

        def release() -> None:
            counts.released += 1
            real_release()

        monkeypatch.setattr(texture, "release", release)
        return texture

    monkeypatch.setattr(gl, "texture", counted)
    return counts


def _assert_resized(document: Any, size: tuple[int, int]) -> None:
    """Immediately after a resize: every canvas is at its own target size, and none went blank.

    Checked here rather than after the next render, because the render hides both halves -- the
    shader re-draws its picture from `vs_uv` whatever it read, and `_feedback_canvas` repairs a
    history whose size disagrees with its pass.
    """
    scale = {"trail": 0.5, "show": 1.0}
    assert document.render_pass.canvas.texture.size == size
    assert _far_corner(document.render_pass.canvas.texture) > 8, (
        f"the live output canvas came back blank at {size}"
    )
    for name in document.feedback_passes():
        history = document._feedback[name]
        wanted = (round(size[0] * scale[name]), round(size[1] * scale[name]))
        assert history.texture.size == wanted, (
            f"'{name}' history is {history.texture.size}, not its own target {wanted}"
        )
        assert _far_corner(history.texture) > 8, (
            f"'{name}' history came back blank at {size}"
        )


def test_a_resize_keeps_the_live_canvas_and_every_history(
    gl_ctx: moderngl.Context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The picture survives a resize on BOTH canvases, and no texture is leaked or freed early.

    Falsifiers, each applied at `Document.set_canvas_size` / `resample_canvas`: resample only
    the history and the live canvas comes back blank, so the next swap trades a blank into the
    history and the far corner reads black; release the old canvas BEFORE allocating the new one
    and the blit samples a freed texture; skip the release and the live count climbs per resize;
    swap the one-quad draw for `copy_framebuffer` and the far corner reads black while a mean
    check still passes; resample the scaled pass to the DOCUMENT size and its history comes back
    at the wrong dimensions.
    """
    ui_document = _loaded(tmp_path)
    document = ui_document.document
    assert set(document.feedback_passes()) == {"trail", "show"}

    # The document holds four textures across the two passes and their two histories, and every
    # one of them is replaced by every resize, so the counts move together or not at all.
    held = len(document.passes) + len(document.feedback_passes())
    sizes = ((96, 96), (48, 48), (80, 80), (32, 32), (128, 128), (64, 64))
    counter = _count_textures(monkeypatch, gl_ctx)
    for size in sizes:
        document.set_canvas_size(size)
        # Asserted on the RESIZE frame, before the re-render: the next render re-draws the
        # picture from `vs_uv` and `_feedback_canvas` fixes any wrong history size, so a
        # check after it passes whether or not the resample did anything.
        _assert_resized(document, size)
        document.begin_frame(document._frame + 1)
        document.render(u_time=1.0)
    # Every allocation is matched by a release, EXCEPT the four the document already held when
    # the counter started: those were created before the wrapper existed, so their release is
    # invisible to it while their replacements are counted.
    assert counter.allocated - counter.released == held, (
        f"{counter.allocated} textures allocated against {counter.released} released across "
        f"{len(sizes)} resizes of {held} textures -- a resize leaks one or frees one it still "
        "needs"
    )
    assert counter.allocated >= held * len(sizes), (
        f"only {counter.allocated} allocations across {len(sizes)} resizes of {held} "
        "textures -- something is not being resized at all"
    )

    # The live canvas carries its picture: a blank one reads 0 in every channel.
    assert _far_corner(document.render_pass.canvas.texture) > 8, (
        "the live output canvas came back blank after the resize"
    )
    # ... and so does each history, which is what a self-reading pass samples next frame.
    for name in document.feedback_passes():
        history = document._feedback[name]
        assert _far_corner(history.texture) > 8, f"'{name}' history came back blank"

    # A non-output pass's history sizes from its OWN target, never the document's: a full-size
    # history on a scale=0.5 pass becomes a full-size live canvas at the next swap.
    assert document._feedback["trail"].texture.size == (32, 32)
    assert document._feedback["show"].texture.size == (64, 64)
    document.release()


def test_a_seeded_history_at_another_size_is_resampled_not_dropped(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # Under Auto the size a save wrote is whatever the panel was then, so the load's size match
    # stops being a rejection and becomes the resample target (090 D4). Falsifier: keep the
    # strict size match and a document saved at one panel size opens black at another.
    ui_document = _loaded(tmp_path)
    document_dir = tmp_path / "doc"
    ui_document.save(document_dir.parent, dir_name=document_dir.name)
    ui_document.document.release()

    meta = json.loads((document_dir / DOCUMENT_JSON_BASENAME).read_text())
    assert meta["feedback"], "the save wrote no feedback frame to seed from"
    # Reopen at a DIFFERENT resolution, which is what an Auto document does across a restart.
    meta["ui_state"]["resolution"] = [96, 96]
    (document_dir / DOCUMENT_JSON_BASENAME).write_text(json.dumps(meta))

    reopened = load_document_from_dir(document_dir)
    assert set(reopened.document.feedback_passes()) == {"trail", "show"}
    assert reopened.document._feedback["trail"].texture.size == (48, 48)
    assert reopened.document._feedback["show"].texture.size == (96, 96)
    for name in reopened.document.feedback_passes():
        assert _far_corner(reopened.document._feedback[name].texture) > 8, (
            f"'{name}' seeded black instead of being resampled"
        )
    reopened.document.release()


def test_a_dtype_mismatch_is_still_refused(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # Size is rescalable; FORMAT is not. A stored `f1` frame written into an `f2` target is not a
    # picture at another size, it is other bytes. Falsifier: resample a dtype mismatch too and a
    # history loads as noise with no warning.
    ui_document = _loaded(tmp_path)
    document_dir = tmp_path / "doc"
    ui_document.save(document_dir.parent, dir_name=document_dir.name)
    ui_document.document.release()

    meta = json.loads((document_dir / DOCUMENT_JSON_BASENAME).read_text())
    for row in meta["feedback"].values():
        row["dtype"] = "f4"
    (document_dir / DOCUMENT_JSON_BASENAME).write_text(json.dumps(meta))

    reopened = load_document_from_dir(document_dir)
    assert reopened.document.feedback_passes() == []
    reopened.document.release()


def test_an_auto_document_opens_at_its_aspect_not_at_its_stored_pair(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    """A loaded Auto document is shaped by its ASPECT before any frame has drawn.

    The stored pair is what a switch to Fixed seeds from and nothing else, so opening at it
    would render the document at a size its mode does not use -- and a pair left over from an
    older shape would render it at the WRONG shape until the first tick corrected it.
    Falsifier: pass `ui_state.resolution` as the loader's size whatever the mode, and the
    aspect assertion below goes red while the pair one passes.
    """
    from shaderbox.render_shape import ResolutionMode, aspect_of, fit_to_aspect
    from shaderbox.ui_models import INITIAL_AUTO_REGION, load_document_from_dir

    document_dir = tmp_path / "doc"
    (document_dir / "passes").mkdir(parents=True)
    (document_dir / "passes" / "main.frag.glsl").write_text(
        "#version 460 core\nin vec2 vs_uv;\nout vec4 fs_color;\n"
        "void main() { fs_color = vec4(vs_uv, 0.0, 1.0); }\n"
    )
    # A 16:9 document whose stored pair is SQUARE: the two disagree on purpose, so only one of
    # them can be the size it opens at.
    (document_dir / "document.json").write_text(
        json.dumps(
            {
                "uniforms": {},
                "ui_state": {
                    "resolution_mode": "auto",
                    "aspect": [16, 9],
                    "resolution": [512, 512],
                },
            }
        )
    )

    ui_document = load_document_from_dir(document_dir)
    document = ui_document.document
    assert document.resolution_mode is ResolutionMode.AUTO
    assert document.aspect == (16, 9)
    # The pair is carried, untouched, for the day it switches to Fixed.
    assert document.resolution == (512, 512)
    # ... and the live canvas is its ASPECT fitted to the nominal region, never the pair.
    assert aspect_of(document.canvas_size) == (16, 9), (
        f"a 16:9 Auto document opened at {document.canvas_size}, which is "
        f"{aspect_of(document.canvas_size)}"
    )
    assert document.canvas_size == fit_to_aspect(INITIAL_AUTO_REGION, (16, 9))
    document.release()
