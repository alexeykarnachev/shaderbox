"""Regenerate the store and README screenshots from the real app.

Runs the app in a hidden window against a throwaway project seeded with the shipped examples,
opens each shot's document and view, lets it render for a while and saves the frame the app
was about to present. Nothing is staged by hand, so a release re-takes every shot with one
command after the UI changes.

Usage: `uv run python scripts/screenshots.py [OUT_DIR] [--only NAME ...]`
(default OUT_DIR: docs/screenshots). Needs a display with hardware GL.
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import glfw
import moderngl
import numpy as np
from PIL import Image

from shaderbox.app import App
from shaderbox.constants import DOCUMENT_EXAMPLES_DIR, EXAMPLE_ORDER
from shaderbox.ui import update_and_draw
from shaderbox.ui_regions import DocumentTab

WIDTH = 1920
HEIGHT = 1080
# Long enough for every pass of every example to compile through the first-render sweep
# and for the animated ones to leave their first frame.
WARMUP_FRAMES = 180

UV_MANGO, ENTITY_FLOCK, _, _, FIRE, NIGHT_CITY, RADIANCE_CASCADES = EXAMPLE_ORDER


@dataclass(frozen=True)
class Shot:
    name: str
    document_id: str
    setup: Callable[[App, str], None]


def _uniforms(app: App, document_id: str) -> None:
    app.ensure_shader_tab(
        document_id, next(iter(app.ui_documents[document_id].document.passes))
    )
    app.focus_document_tab(DocumentTab.UNIFORMS)


def _graph(app: App, document_id: str) -> None:
    app.open_graph_for(document_id, focus_editor=True)
    app.focus_document_tab(DocumentTab.DOCUMENT)


def _script(app: App, document_id: str) -> None:
    app.open_script_for(document_id, focus_editor=True)
    app.focus_document_tab(DocumentTab.UNIFORMS)


SHOTS: tuple[Shot, ...] = (
    Shot("editor_uniforms", NIGHT_CITY, _uniforms),
    Shot("pass_graph", RADIANCE_CASCADES, _graph),
    Shot("script_instancing", ENTITY_FLOCK, _script),
    Shot("fire", FIRE, _uniforms),
)


def _seed(root: Path) -> Path:
    project = root / "examples"
    documents = project / "documents"
    documents.mkdir(parents=True)
    for document_id in EXAMPLE_ORDER:
        shutil.copytree(DOCUMENT_EXAMPLES_DIR / document_id, documents / document_id)
    return project


def _frame(app: App) -> None:
    # The pacing and the FPS the canvas chip shows, measured as `ui.run` measures them.
    start = glfw.get_time()
    update_and_draw(app)
    time.sleep(
        max(0.0, 1.0 / app.app_state.global_target_fps - (glfw.get_time() - start))
    )
    fps = 1.0 / (glfw.get_time() - start)
    app.global_fps = (
        fps if app.global_fps <= 0.0 else 0.95 * app.global_fps + 0.05 * fps
    )


def _take(app: App, shot: Shot, out_dir: Path) -> Path:
    captured: list[Image.Image] = []
    present = glfw.swap_buffers

    def capture_then_present(window: object) -> None:
        if not captured:
            # The hidden window's default framebuffer keeps the monitor's size; the frame
            # is the window's corner of it.
            width, height = glfw.get_framebuffer_size(window)
            data = moderngl.get_context().screen.read(
                viewport=(0, 0, width, height), components=3
            )
            pixels = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
            captured.append(Image.fromarray(pixels[::-1].copy()))
            # Reading the default framebuffer sets GL_INVALID_OPERATION (moderngl selects a
            # colour attachment it does not have); left pending, the app's next checked GL
            # call raises it as its own.
            moderngl.get_context().clear_errors()
        present(window)

    app.set_current_document_id(shot.document_id)
    shot.setup(app, shot.document_id)
    for _ in range(WARMUP_FRAMES):
        _frame(app)
    glfw.swap_buffers = capture_then_present
    try:
        update_and_draw(app)
    finally:
        glfw.swap_buffers = present
    path = out_dir / f"{shot.name}.png"
    captured[0].save(path, optimize=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the store and README screenshots from the real app."
    )
    parser.add_argument("out_dir", nargs="?", default="docs/screenshots", type=Path)
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="shaderbox-shots-data-") as data_dir:
        # A fresh data dir: the user's own settings (keymap, theme, layout) stay out of the shots.
        os.environ["SHADERBOX_DATA_DIR"] = data_dir
        # One app for every shot: imgui's font texture belongs to the first app's context, so
        # a second App in the same process draws with a texture it does not own.
        app = App(project_dir=_seed(Path(data_dir)), headless=True)
        glfw.set_window_size(app.window, WIDTH, HEIGHT)
        # A fresh data dir is a first run, which opens the keyboard cheatsheet over the panel.
        app.app_state.show_cheatsheet = False
        for shot in SHOTS:
            if args.only and shot.name not in args.only:
                continue
            print(_take(app, shot, out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
