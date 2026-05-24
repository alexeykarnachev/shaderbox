"""GL-backed tests for the share-tab render glue.

These need a real OpenGL context. We build a headless standalone moderngl
context; if the environment can't provide one (no GL driver on a CI runner),
the whole module skips rather than failing.
"""

import contextlib
from collections.abc import Iterator
from pathlib import Path

import moderngl
import pytest

from shaderbox.core import Node
from shaderbox.media import MediaDetails
from shaderbox.render_preset import FitPolicy, RenderPreset, ResolutionPolicy
from shaderbox.tabs.share_state import render_for


@pytest.fixture(scope="module")
def gl_ctx() -> Iterator[moderngl.Context]:
    try:
        ctx = moderngl.create_standalone_context()
    except Exception as e:
        pytest.skip(f"no standalone GL context available: {e}")
    yield ctx
    ctx.release()


@pytest.fixture
def node(gl_ctx: moderngl.Context) -> Iterator[Node]:
    n = Node(gl=gl_ctx)
    n.render()  # warm-up: compile the default program
    yield n
    # release_program ends in a raw PyOpenGL glUseProgram(0) that has no bound
    # context under a moderngl *standalone* context (fine in-app on glfw). The
    # standalone context's own teardown reclaims the GL objects regardless.
    with contextlib.suppress(Exception):
        n.release()


def _image_details(node: Node, path: Path) -> MediaDetails:
    details = MediaDetails(is_video=False, duration=1.0)
    details.file_details.path = str(path)
    w, h = node.canvas.texture.size
    details.resolution_details.width = w
    details.resolution_details.height = h
    return details


def test_render_media_preset_none_byte_identical(node: Node, tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    node.render_media(_image_details(node, a), preset=None)
    node.render_media(_image_details(node, b), preset=None)
    assert a.exists() and b.exists()
    assert a.read_bytes() == b.read_bytes()


def test_render_for_mints_and_artifact_exists(node: Node, tmp_path: Path) -> None:
    # Outlet presets render at a resolved target (RENDER_AT_TARGET) — that path
    # fills resolution_details via resolve_dims, unlike a bare SCALE_DISTORT preset.
    preset = RenderPreset(
        is_video=False,
        container=".png",
        resolution_policy=ResolutionPolicy.LONGEST_EDGE,
        longest_edge=64,
        fit=FitPolicy.RENDER_AT_TARGET,
    )
    artifact = render_for(node, preset, duration=1.0, scratch_dir=tmp_path)
    assert artifact is not None
    assert artifact.path.exists()
    assert artifact.path.parent == tmp_path
    assert not artifact.is_video


def test_render_for_respects_longest_edge(node: Node, tmp_path: Path) -> None:
    preset = RenderPreset(
        is_video=False,
        container=".png",
        resolution_policy=ResolutionPolicy.LONGEST_EDGE,
        longest_edge=32,
        fit=FitPolicy.RENDER_AT_TARGET,
    )
    artifact = render_for(node, preset, duration=1.0, scratch_dir=tmp_path)
    assert artifact is not None
    # Default canvas is 64x64; longest_edge=32 halves it (16-aligned → 32).
    assert max(artifact.size) <= 32


def test_render_for_cleans_up_on_render_failure(
    node: Node, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_: object, **__: object) -> MediaDetails:
        raise RuntimeError("render blew up")

    monkeypatch.setattr(node, "render_media", boom)
    before = set(tmp_path.iterdir())
    artifact = render_for(
        node, RenderPreset(is_video=False, container=".png"), 1.0, tmp_path
    )
    assert artifact is None
    # No partial file left behind.
    assert set(tmp_path.iterdir()) == before
