"""GL-backed tests for the share-tab render glue.

These need a real OpenGL context. We build a headless standalone moderngl
context; if the environment can't provide one (no GL driver on a CI runner),
the whole module skips rather than failing.
"""

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import moderngl
import pytest
from PIL import Image as PILImage

from shaderbox.core import Pass
from shaderbox.document import DEFAULT_PASS_NAME, Document
from shaderbox.media import MediaDetails
from shaderbox.pass_graph import PassEntry, PassGraph, PassSource, TargetConfig
from shaderbox.render_job import render_for
from shaderbox.render_preset import (
    FitPolicy,
    RenderPreset,
    ResolutionPolicy,
    resolve_dims,
)
from shaderbox.render_shape import (
    RenderShape,
    ResolutionMode,
    aspect_of,
    shape_to_preset,
)
from shaderbox.ui_models import UIDocument


@pytest.fixture(scope="module")
def gl_ctx() -> Iterator[moderngl.Context]:
    try:
        context = moderngl.create_standalone_context()
    except Exception as e:
        pytest.skip(f"no standalone GL context available: {e}")
    yield context
    context.release()


@pytest.fixture
def document(gl_ctx: moderngl.Context) -> Iterator[Document]:
    n = Document(gl=gl_ctx)
    n.render()  # warm-up: compile the default program
    yield n
    # release_program ends in a raw PyOpenGL glUseProgram(0) that has no bound
    # context under a moderngl *standalone* context (fine in-app on glfw). The
    # standalone context's own teardown reclaims the GL objects regardless.
    with contextlib.suppress(Exception):
        n.release()


def _image_details(document: Document, path: Path) -> MediaDetails:
    details = MediaDetails(is_video=False, duration=1.0)
    details.file_details.path = str(path)
    w, h = document.render_pass.canvas.texture.size
    details.resolution_details.width = w
    details.resolution_details.height = h
    return details


def test_render_media_preset_none_byte_identical(
    document: Document, tmp_path: Path
) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    document.render_media(_image_details(document, a), preset=None)
    document.render_media(_image_details(document, b), preset=None)
    assert a.exists() and b.exists()
    assert a.read_bytes() == b.read_bytes()


def test_render_for_mints_and_artifact_exists(
    document: Document, tmp_path: Path
) -> None:
    # Outlet presets render at a resolved target (RENDER_AT_TARGET) — that path
    # fills resolution_details via resolve_dims, unlike a bare SCALE_DISTORT preset.
    preset = RenderPreset(
        is_video=False,
        container=".png",
        resolution_policy=ResolutionPolicy.LONGEST_EDGE,
        longest_edge=64,
        fit=FitPolicy.RENDER_AT_TARGET,
    )
    artifact = render_for(document, preset, duration=1.0, scratch_dir=tmp_path)
    assert artifact is not None
    assert artifact.path.exists()
    assert artifact.path.parent == tmp_path
    assert not artifact.is_video


def test_render_for_respects_longest_edge(document: Document, tmp_path: Path) -> None:
    preset = RenderPreset(
        is_video=False,
        container=".png",
        resolution_policy=ResolutionPolicy.LONGEST_EDGE,
        longest_edge=32,
        fit=FitPolicy.RENDER_AT_TARGET,
    )
    artifact = render_for(document, preset, duration=1.0, scratch_dir=tmp_path)
    assert artifact is not None
    # Default canvas is 64x64; longest_edge=32 halves it (16-aligned → 32).
    assert max(artifact.size) <= 32


_MAGENTA = """#version 460 core
in vec2 vs_uv;
out vec4 fs_color;
void main() { fs_color = vec4(1.0, 0.0, 1.0, 1.0); }
"""


def test_an_off_size_export_renders_the_shader_not_a_blank(
    document: Document, tmp_path: Path
) -> None:
    # A RENDER_AT_TARGET preset draws into a SCRATCH canvas, so the document must hand that
    # canvas down to its pass. Asserting only the size passes on a fully transparent image:
    # the resize runs either way, and a blank export is the exact shape of a dropped canvas.
    document.render_pass.release_program(_MAGENTA)
    document.render_pass.compile()
    assert document.render_pass.compile_unit.errors == []
    preset = RenderPreset(
        is_video=False,
        container=".png",
        resolution_policy=ResolutionPolicy.LONGEST_EDGE,
        longest_edge=32,
        fit=FitPolicy.RENDER_AT_TARGET,
    )
    artifact = render_for(document, preset, duration=1.0, scratch_dir=tmp_path)
    assert artifact is not None
    pixel = PILImage.open(artifact.path).convert("RGBA").getpixel((0, 0))
    assert pixel == (255, 0, 255, 255)


def test_render_for_cleans_up_on_render_failure(
    document: Document, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_: object, **__: object) -> MediaDetails:
        raise RuntimeError("render blew up")

    monkeypatch.setattr(document, "render_media", boom)
    before = set(tmp_path.iterdir())
    artifact = render_for(
        document, RenderPreset(is_video=False, container=".png"), 1.0, tmp_path
    )
    assert artifact is None
    # No partial file left behind.
    assert set(tmp_path.iterdir()) == before


# ---------------------------------------------------------------------------
# 090 D5 (revision 1) -- an export resolves its source from `export_source_size`: the stored
# pair under Fixed, the LIVE canvas under Auto, where the document carries no pair. The Render
# tab's own W x H still decides the file on the `preset=None` path.
# ---------------------------------------------------------------------------


_SOURCE_SIZES: list[tuple[int, int]] = []


def _last_source_size() -> tuple[int, int]:
    return _SOURCE_SIZES[-1]


@pytest.fixture(autouse=True)
def _record_source_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Record the canvas each export actually rendered into.

    The size the frame was DRAWN at is the thing D5 moves; the file's own size is a resize
    downstream of it, and on the `preset=None` path it is the Render tab's number either way.
    """
    _SOURCE_SIZES.clear()
    real = Document._render_media_into

    def spy(self: Document, details: MediaDetails, canvas: Any) -> MediaDetails:
        _SOURCE_SIZES.append(canvas.texture.size)
        return real(self, details, canvas)

    monkeypatch.setattr(Document, "_render_media_into", spy)


def _auto_document(gl: moderngl.Context, size: tuple[int, int]) -> Document:
    """An Auto document at a live size: it stores the ASPECT, and the size is the viewer's."""
    doc = Document(gl=gl, canvas_size=size)
    doc.resolution_mode = ResolutionMode.AUTO
    doc.aspect = aspect_of(size)
    doc.render()
    return doc


@pytest.mark.parametrize("shape", [None, "native", "wide"])
def test_an_auto_export_renders_at_the_live_size(
    gl_ctx: moderngl.Context, tmp_path: Path, shape: str | None
) -> None:
    # Under Auto the document stores NO size (090 revision 1), so `RenderShape.NATIVE` and
    # `resolve_dims`'s FREE fall-through mean the LIVE canvas -- what the user is looking at.
    # Falsifier: resolve from anything else (a stored pair, a constant) and the None and NATIVE
    # cases stop tracking the size the viewer put the document at.
    document = _auto_document(gl_ctx, (640, 480))
    document.set_canvas_size((320, 180))  # what the viewer put it at
    assert document.render_pass.canvas.texture.size == (320, 180)

    preset = (
        None
        if shape is None
        else shape_to_preset(
            RenderShape.NATIVE if shape == "native" else RenderShape.WIDE_720,
            is_video=False,
            fps=None,
            container=".png",
            duration_max=None,
        )
    )
    path = tmp_path / f"{shape}.png"
    details = MediaDetails(is_video=False, duration=1.0)
    details.file_details.path = str(path)
    details.resolution_details.width, details.resolution_details.height = (640, 480)
    document.render_media(details, preset)

    # The Render tab's own W x H still decides the FILE on the `preset=None` path, as it always
    # has; a shape preset decides it from the shape. NATIVE's 320x180 comes back 16-ALIGNED,
    # which `resolve_dims` has always done for codec compatibility and which this revision does
    # not change.
    assert (
        PILImage.open(path).size
        == {
            None: (640, 480),
            "native": (320, 192),
            "wide": (1280, 720),
        }[shape]
    )
    # The SOURCE the frame was drawn at, which is what the revision moves: the live canvas on
    # both free paths, the shape's own size on the fixed-aspect one. `_render_media_into`
    # records the canvas it was handed.
    assert (
        _last_source_size()
        == {
            None: (320, 180),
            "native": (320, 192),
            "wide": (1280, 720),
        }[shape]
    )
    document.release()


def test_the_publish_gate_resolves_from_the_same_source_the_export_does(
    gl_ctx: moderngl.Context,
) -> None:
    # The gate answers "would a render now produce this artifact?", so it must resolve from the
    # size a render WOULD use. Under Auto that is the live canvas, so an artifact rendered at
    # the old size stops matching once the viewer resizes -- correct, and why the Render button
    # sits beside it. A FIXED document's gate is immune to the viewer entirely. Falsifier:
    # resolve the gate from anything but `export_source_size` and one of the two halves flips.
    from shaderbox.exporters.base import RenderedArtifact
    from shaderbox.exporters.youtube import YouTubeExporter

    document = _auto_document(gl_ctx, (1920, 1080))
    ui_document = UIDocument(document=document)
    exporter = YouTubeExporter()
    exporter._render_state.shape = RenderShape.NATIVE
    expected = resolve_dims(exporter.render_preset(), document.export_source_size())
    artifact = RenderedArtifact(
        path=Path("x.mp4"), is_video=True, duration=4.0, size=expected
    )
    assert exporter._artifact_matches_shape(artifact, ui_document)

    # Auto: the source moved, so the artifact is genuinely stale.
    document.set_canvas_size((320, 180))
    assert not exporter._artifact_matches_shape(artifact, ui_document)

    # Fixed: the stored pair is the source, and no resize can move it.
    document.resolution_mode = ResolutionMode.FIXED
    document.resolution = (1920, 1080)
    fixed_artifact = RenderedArtifact(
        path=Path("y.mp4"),
        is_video=True,
        duration=4.0,
        size=resolve_dims(exporter.render_preset(), document.export_source_size()),
    )
    assert exporter._artifact_matches_shape(fixed_artifact, ui_document)
    document.set_canvas_size((640, 360))
    assert exporter._artifact_matches_shape(fixed_artifact, ui_document), (
        "a live resize disarmed publish on a FIXED document, whose source cannot move"
    )
    document.release()


def test_the_render_tabs_own_size_still_lands_on_disk(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # D5 moved the SOURCE size to `resolution` and left `resolution_details` alone: the Render
    # tab's W x H is still what the file comes out at, through the PIL resize. Falsifier:
    # overwrite `resolution_details` from `resolution` and this exports 1920x1080. Note
    # `test_render_media_preset_none_byte_identical` above passes either way, so no existing
    # gate catches it.
    document = _auto_document(gl_ctx, (1920, 1080))
    path = tmp_path / "tab_size.png"
    details = MediaDetails(is_video=False, duration=1.0)
    details.file_details.path = str(path)
    details.resolution_details.width, details.resolution_details.height = (640, 480)
    document.render_media(details, preset=None)
    assert PILImage.open(path).size == (640, 480)
    document.release()


_FEEDBACK_TRAIL = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_prev;
out vec4 fs_color;
void main() {
    fs_color = vec4(max(texture(u_prev, vs_uv).r, 1.0 - vs_uv.y), 0.0, 1.0, 1.0);
}
"""

_FEEDBACK_SHOW = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_trail;
out vec4 fs_color;
void main() { fs_color = vec4(texture(u_trail, vs_uv).r, 0.0, 1.0, 1.0); }
"""


def test_a_scaled_feedback_pass_survives_an_off_size_export(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # The untested territory D5 names: non-output passes size from `target_size(canvas_size)`
    # while the OUTPUT frame is captured at the export size, so a scale=0.5 feedback pass inside
    # an off-size export is the case where the two could disagree. Falsifier: size the feedback
    # canvas from the export target and the sampler reads a mismatched texture, which shows up
    # as a blank or a wrongly-scaled contribution.
    document = Document(gl=gl_ctx, canvas_size=(64, 64))
    document.resolution = (64, 64)
    document.passes[DEFAULT_PASS_NAME].release()
    document.passes = {}
    for name, source in (("trail", _FEEDBACK_TRAIL), ("show", _FEEDBACK_SHOW)):
        render_pass = Pass(
            gl=gl_ctx,
            canvas_size=(64, 64),
            target=TargetConfig(scale=0.5) if name == "trail" else TargetConfig(),
        )
        render_pass.release_program(source)
        render_pass.compile()
        assert render_pass.compile_unit.errors == [], render_pass.compile_unit.errors
        document.passes[name] = render_pass
    document.graph = PassGraph(
        output="show",
        passes={
            "trail": PassEntry(target=TargetConfig(scale=0.5)),
            "show": PassEntry(),
        },
    )
    document.passes["show"].uniform_values["u_trail"] = PassSource("trail")

    preset = RenderPreset(
        is_video=False,
        container=".png",
        resolution_policy=ResolutionPolicy.LONGEST_EDGE,
        longest_edge=32,
        fit=FitPolicy.RENDER_AT_TARGET,
    )
    artifact = render_for(document, preset, duration=1.0, scratch_dir=tmp_path)
    assert artifact is not None
    image = PILImage.open(artifact.path).convert("RGBA")
    assert image.size == (32, 32)
    # The trail's own contribution: a gradient in the red channel, bright at one end of the
    # image and dark at the other. A dropped or mis-sized feedback canvas leaves it flat.
    near = image.getpixel((16, 30))[0]
    far = image.getpixel((16, 1))[0]
    assert near > far + 32, f"the feedback pass contributed nothing: {near} vs {far}"
    # And it drew at the DOCUMENT's scale, not the export target's: a non-output pass sizes
    # from `target_size(canvas_size)` while the OUTPUT frame is captured at the export size.
    # The picture cannot see this on its own -- a smooth gradient resampled 16 -> 32 by the
    # sampler looks the same -- so the size is the assertion. Falsifier: size the pass from
    # the export canvas and the trail comes back at 16x16, a quarter of the pixels it owes.
    assert document.passes["trail"].canvas.texture.size == (32, 32)
    assert document.passes["show"].canvas.texture.size == (64, 64)
    document.release()
