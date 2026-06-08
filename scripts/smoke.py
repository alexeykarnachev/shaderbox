"""Headless smoke test — runs ~200 frames of update_and_draw against projects/dev/
in an invisible glfw window and asserts no exception + a few invariants.

Catches import errors, callback dispatch failures, popup state-machine crashes,
released-texture binding errors. Doesn't catch visual bugs.

Usage: `uv run python scripts/smoke.py` (exit 0 on success, non-zero on failure).
"""

import sys
from pathlib import Path

import glfw
from imgui_bundle import imgui
from loguru import logger
from platformdirs import user_data_dir

from shaderbox.app import App, PopupState
from shaderbox.commands import ActiveRegion, NodeTab
from shaderbox.logging_setup import configure_logging
from shaderbox.ui import update_and_draw

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEV_PROJECT_DIR: Path = REPO_ROOT / "projects" / "dev"
PROJECT_DIR_POINTER: Path = Path(user_data_dir("shaderbox")) / "project_dir"
N_FRAMES: int = 200


def _check_invariants(app: App, frame_idx: int) -> None:
    # The "at most one modal popup open" mutex is now structural — popup_state is a single
    # PopupState value, so two modals can't be open at once by construction (feature 023).
    assert isinstance(
        app.popup_state, PopupState
    ), f"frame {frame_idx}: popup_state is not a PopupState ({app.popup_state!r})"
    assert app.current_node_id == "" or app.current_node_id in app.ui_nodes, (
        f"frame {frame_idx}: current_node_id={app.current_node_id!r} not in "
        f"ui_nodes={list(app.ui_nodes.keys())}"
    )
    # Feature 018: the registry must be populated + dispatched every frame (the
    # cheatsheet overlay draws here too, exercising its no-assert path headlessly).
    assert app.effective_bindings, f"frame {frame_idx}: effective_bindings empty"
    # Feature 019: nav focus model stays in valid enum states.
    assert (
        app.active_region in ActiveRegion
    ), f"frame {frame_idx}: bad active_region={app.active_region!r}"
    assert (
        app.active_node_tab in NodeTab
    ), f"frame {frame_idx}: bad active_node_tab={app.active_node_tab!r}"


def main() -> int:
    configure_logging()
    if not DEV_PROJECT_DIR.exists():
        logger.error(f"Fixture project not found: {DEV_PROJECT_DIR}")
        return 1

    saved_pointer: str | None = (
        PROJECT_DIR_POINTER.read_text() if PROJECT_DIR_POINTER.exists() else None
    )

    glfw.init()

    try:
        app = App(project_dir=DEV_PROJECT_DIR, headless=True)
        # Feature 019: nav_enable_keyboard is set in __init__, before any frame —
        # check it here (get_io() reads are frame-context-sensitive mid-loop).
        assert (
            imgui.get_io().config_flags & imgui.ConfigFlags_.nav_enable_keyboard
        ), "nav_enable_keyboard not set"
        for frame_idx in range(N_FRAMES):
            update_and_draw(app)
            _check_invariants(app, frame_idx)
            # Exercise the region-cycle + tab-jump wiring (a callback throw surfaces
            # via the except below); nav *behavior* is un-headless-able.
            if frame_idx == 50:
                app.cycle_region()
            if frame_idx == 60:
                app.focus_node_tab(NodeTab.RENDER)
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
