"""Headless smoke test — runs ~200 frames of update_and_draw against a THROWAWAY tmp project
(seeded with the shipped template nodes; never the tracked projects/dev sandbox) in an invisible
glfw window and asserts no exception + a few invariants.

Catches import errors, callback dispatch failures, popup state-machine crashes,
released-texture binding errors. Doesn't catch visual bugs.

Usage: `uv run python scripts/smoke.py` (exit 0 on success, non-zero on failure).
"""

import shutil
import sys
import tempfile
from pathlib import Path

import glfw
from imgui_bundle import imgui
from loguru import logger

from shaderbox.app import App, PopupState
from shaderbox.commands import ActiveRegion, NodeTab
from shaderbox.constants import RESOURCES_DIR
from shaderbox.logging_setup import configure_logging
from shaderbox.ui import update_and_draw

N_FRAMES: int = 200


def _has_gpu_window() -> bool:
    # The GUI smoke drives a full glfw window backed by hardware GL. On a display-less box
    # (a Pi over SSH, a CI runner with no GPU) glfw can't create that window — skip the smoke
    # loudly instead of crashing. Probe by actually trying: init + a hidden window. See
    # ai_docs/todo.md "headless GL".
    if not glfw.init():
        return False
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(64, 64, "probe", None, None)
    if window is None:
        return False
    glfw.destroy_window(window)
    return True


_TEMPLATE_IDS: list[str] = [
    "53724dbd-8efb-4c09-8c7d-28d626a066e7",  # UV Mango
    "73ea2431-13f6-41e4-b923-04d846b678b0",  # Media Input
    "f90f5ff9-29c6-4bcf-aee7-090f20542353",  # Text Rendering
]


def _seed_tmp_project(root: Path) -> Path:
    # A throwaway project seeded with the shipped template nodes — smoke must never read or
    # mutate the tracked projects/dev sandbox.
    project = root / "project"
    nodes = project / "nodes"
    nodes.mkdir(parents=True)
    for tid in _TEMPLATE_IDS:
        shutil.copytree(RESOURCES_DIR / "node_templates" / tid, nodes / tid)
    return project


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

    if not _has_gpu_window():
        logger.warning(
            "smoke: SKIPPED — no GPU window available (display-less box / no hardware GL). "
            "The GUI smoke needs a real glfw window; run it on a machine with a display."
        )
        return 0

    with tempfile.TemporaryDirectory(prefix="shaderbox-smoke-") as tmp:
        project = _seed_tmp_project(Path(tmp))
        try:
            app = App(project_dir=project, headless=True)
            if app.ui_nodes:
                app.set_current_node_id(next(iter(app.ui_nodes)))
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


if __name__ == "__main__":
    sys.exit(main())
