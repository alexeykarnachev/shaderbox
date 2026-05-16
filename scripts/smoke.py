"""Headless smoke test — runs ~200 frames of update_and_draw against projects/dev/
in an invisible glfw window and asserts no exception + a few invariants.

Catches import errors, callback dispatch failures, popup state-machine crashes,
released-texture binding errors. Doesn't catch visual bugs.

Usage: `uv run python scripts/smoke.py` (exit 0 on success, non-zero on failure).
"""

import sys
from pathlib import Path

import glfw
from loguru import logger
from platformdirs import user_data_dir

from shaderbox.app import App
from shaderbox.ui import update_and_draw

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEV_PROJECT_DIR: Path = REPO_ROOT / "projects" / "dev"
PROJECT_DIR_POINTER: Path = Path(user_data_dir("shaderbox")) / "project_dir"
N_FRAMES: int = 200


def _check_invariants(app: App, frame_idx: int) -> None:
    assert not (
        app.is_node_creator_open and app.is_settings_open
    ), f"frame {frame_idx}: both popups open (mutex broken)"
    assert app.current_node_id == "" or app.current_node_id in app.ui_nodes, (
        f"frame {frame_idx}: current_node_id={app.current_node_id!r} not in "
        f"ui_nodes={list(app.ui_nodes.keys())}"
    )


def main() -> int:
    if not DEV_PROJECT_DIR.exists():
        logger.error(f"Fixture project not found: {DEV_PROJECT_DIR}")
        return 1

    saved_pointer: str | None = (
        PROJECT_DIR_POINTER.read_text() if PROJECT_DIR_POINTER.exists() else None
    )

    glfw.init()
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

    try:
        app = App(project_dir=DEV_PROJECT_DIR)
        for frame_idx in range(N_FRAMES):
            update_and_draw(app)
            _check_invariants(app, frame_idx)
        app.release()
        logger.info(f"smoke: OK ({N_FRAMES} frames, {len(app.ui_nodes)} nodes)")
        return 0
    except Exception as e:
        logger.exception(f"smoke: FAIL — {e}")
        return 1
    finally:
        if saved_pointer is not None:
            PROJECT_DIR_POINTER.write_text(saved_pointer)


if __name__ == "__main__":
    sys.exit(main())
