"""Headless smoke test — runs ~200 frames of update_and_draw against a THROWAWAY tmp project
(seeded with the shipped template nodes; never the tracked projects/dev sandbox) in an invisible
glfw window and asserts no exception + a few invariants.

Catches import errors, callback dispatch failures, popup state-machine crashes,
released-texture binding errors. Doesn't catch visual bugs.

Usage: `uv run python scripts/smoke.py` (exit 0 on success, non-zero on failure).
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import glfw
from imgui_bundle import imgui
from loguru import logger

from shaderbox.app import App, PopupState
from shaderbox.commands import ActiveRegion, NodeTab
from shaderbox.constants import NODE_TEMPLATES_DIR, TEMPLATE_ORDER
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


_BRAIN_SHADER = """#version 460 core
in vec2 vs_uv;
uniform float u_a;
uniform float u_b;
out vec4 fs_color;
void main() { fs_color = vec4(u_a, u_b, 0.0, 1.0); }
"""

_BRAIN_NODE_JSON = {
    "canvas_size": [256, 256],
    "uniforms": {},
    "ui_state": {
        "ui_name": "Node Brain",
        "description": "smoke: a node-brain driving many uniforms",
    },
}

# The seeded brain: TWO stateful integrators (u_a, u_b both accumulate) + a typo'd homeless key
# (u_typo) so the brain's drive/skip/soft-error paths all run under smoke. Both keys integrate (NOT
# ctx.mouse, which is frozen at 0.5 headless) so the stopped-skip canary is falsifiable: a stopped u_a
# must STAY frozen while the un-stopped u_b keeps ADVANCING.
_BRAIN_SCRIPT = (
    "from shaderbox.scripting import ScriptBehavior, Ctx\n\n"
    "class Behavior(ScriptBehavior):\n"
    "    def __init__(self) -> None:\n"
    "        self.a = 0.0\n"
    "        self.b = 0.0\n"
    "    def update(self, ctx: Ctx) -> dict:\n"
    "        self.a += ctx.dt\n"
    "        self.b += ctx.dt * 2.0\n"
    "        return {'u_a': self.a, 'u_b': self.b, 'u_typo': 1.0}\n"
)


def _seed_tmp_project(root: Path) -> Path:
    # A throwaway project seeded with the shipped template nodes — smoke must never read or
    # mutate the tracked projects/dev sandbox.
    project = root / "project"
    nodes = project / "nodes"
    nodes.mkdir(parents=True)
    for tid in TEMPLATE_ORDER:
        shutil.copytree(NODE_TEMPLATES_DIR / tid, nodes / tid)

    # A node-brain node (048 — one script per node): the engine ticks it every frame, so 200 clean
    # frames prove the App-with-a-scripted-node loop doesn't crash. (Engine-correctness — values/
    # freeze/determinism/play-stop — is the pure-CPU unit test's job; smoke proves the App loop +
    # the binding + the stopped-skip wire.)
    brain = nodes / "brain"
    brain.mkdir()
    (brain / "shader.frag.glsl").write_text(_BRAIN_SHADER, encoding="utf-8")
    (brain / "node.json").write_text(json.dumps(_BRAIN_NODE_JSON), encoding="utf-8")
    brain_scripts = brain / "scripts"
    brain_scripts.mkdir()
    (brain_scripts / "script.py").write_text(_BRAIN_SCRIPT, encoding="utf-8")
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
            # Decision-15 regression canary (048): _init opens the restored current node's shader tab
            # (it used to stay blank until a node switch). A non-empty project must have a tab now.
            assert app.editor_tabs, (
                "smoke: editor_tabs empty after _init — the restored current node's shader tab "
                "was not opened (the load->ensure_shader_tab wire is missing)"
            )
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
                # Open the New Node modal for a stretch so its draw path (grid + desc slot +
                # action row sizing) is exercised — it never opens on its own in the loop.
                if frame_idx == 70:
                    app.popup_state = PopupState.NODE_CREATOR
                if frame_idx == 90:
                    app.popup_state = PopupState.CLOSED
                # Cold-start copilot gate over an open Settings modal: the chat is open with no key
                # (gate path) and the focus-pending latch is set, the exact state that re-grabbed
                # focus every frame and dismissed the modal. The render must not crash; the
                # focus-guard regression itself is asserted in tests/test_copilot_focus.py (a frame
                # render can't read is_popup_open between frames without segfaulting). /imgui-ui §8.
                if frame_idx == 100:
                    app.is_copilot_open = True
                    app.integrations_store.copilot.openrouter_key = ""
                    app.focus_copilot()
                if frame_idx == 102:
                    app.open_settings()
                if frame_idx == 113:
                    app.popup_state = PopupState.CLOSED
                    app.is_copilot_open = False
            # Canary (048): the brain must have BOUND + ticked (binding is by `script.py` existence).
            engine = app.session.script_engine
            driven = engine.script_driven_uniforms("brain")
            assert "u_a" in driven and "u_b" in driven, (
                f"smoke: the node brain did not bind/tick (driven={driven}) — script.py wasn't "
                "discovered/bound"
            )
            # Stopped-skip wire (048): stop u_a, tick once; its WRITE must be skipped (the manual
            # value sticks) while u_a stays driven AND the un-stopped u_b still advances. A dead
            # `stopped` wire would keep writing u_a and green-wash the play/stop model.
            brain_node = app.ui_nodes["brain"].node
            brain_node.uniform_values["u_a"] = -999.0
            b_before = brain_node.uniform_values["u_b"]
            app.session.set_uniform_stopped("brain", "u_a", True)
            app.session.tick(["brain"], t=1.0, dt=0.5, frame=999)
            assert (
                brain_node.uniform_values["u_a"] == -999.0
            ), "smoke: a stopped uniform was overwritten — the tick(stopped=) skip is unwired"
            assert (
                "u_a" in engine.script_driven_uniforms("brain")
            ), "smoke: a stopped uniform fell out of the driven set (its play button would vanish)"
            assert brain_node.uniform_values["u_b"] != b_before, (
                "smoke: the un-stopped u_b did not advance while u_a was stopped — the stop is "
                "freezing the whole brain, not just the one uniform"
            )
            app.release()
            logger.info(f"smoke: OK ({N_FRAMES} frames, {len(app.ui_nodes)} nodes)")
            return 0
        except Exception as e:
            logger.exception(f"smoke: FAIL — {e}")
            return 1


if __name__ == "__main__":
    sys.exit(main())
