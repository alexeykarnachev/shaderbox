"""Headless smoke test — runs ~200 frames of update_and_draw against a THROWAWAY tmp project
(seeded with the shipped example documents; never the tracked projects/dev sandbox) in an invisible
glfw window and asserts no exception + a few invariants.

Catches import errors, callback dispatch failures, popup state-machine crashes,
released-texture binding errors. Doesn't catch visual bugs.

Usage: `uv run python scripts/smoke.py` (exit 0 on success, non-zero on failure).

A display-less box cannot run the smoke at all, and that is not a failure — the skip path exits 0,
so `build.sh` and a direct `make smoke` keep their two-outcome contract. A caller that must tell a
skip apart from a pass (`make gates` does) sets `SHADERBOX_SMOKE_SKIP_EXIT` to the code it wants the
skip to return instead.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import glfw
from imgui_bundle import imgui
from loguru import logger

from shaderbox.app import App, PopupState
from shaderbox.constants import DOCUMENT_EXAMPLES_DIR, EXAMPLE_ORDER
from shaderbox.document import Document
from shaderbox.editor.ffi import Kind as EditorKind
from shaderbox.help_content import help_sections
from shaderbox.logging_setup import configure_logging
from shaderbox.pass_graph import PassEntry, PassGraph
from shaderbox.paths import PASSES_DIR_NAME, pass_shader_name
from shaderbox.ui import update_and_draw
from shaderbox.ui_regions import DocumentTab

N_FRAMES: int = 200


def _skip_exit_code() -> int:
    # Default 0: a display-less box has not failed anything, and every direct caller
    # (`make smoke`, `build.sh`) reads exit 0 as "nothing wrong here". A caller that
    # must score a skip apart from a pass sets SHADERBOX_SMOKE_SKIP_EXIT.
    raw: str = os.environ.get("SHADERBOX_SMOKE_SKIP_EXIT", "0")
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            f"smoke: SHADERBOX_SMOKE_SKIP_EXIT={raw!r} is not an integer; skipping with 0"
        )
        return 0


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


_SCRIPT_SHADER = """#version 460 core
in vec2 vs_uv;
uniform float u_a;
uniform float u_b;
out vec4 fs_color;
void main() { fs_color = vec4(u_a, u_b, 0.0, 1.0); }
"""

_SCRIPT_NODE_JSON = {
    "canvas_size": [256, 256],
    "uniforms": {},
    "ui_state": {
        "ui_name": "Scripted Document",
        "description": "smoke: a document script driving many uniforms",
    },
}

# The seeded script: TWO stateful integrators (u_a, u_b both accumulate) + a typo'd homeless key
# (u_typo) so the script's drive/skip/soft-error paths all run under smoke. Both keys integrate (NOT
# ctx.mouse, which is frozen at 0.5 headless) so the stopped-skip canary is falsifiable: a stopped u_a
# must STAY frozen while the un-stopped u_b keeps ADVANCING.
_SCRIPT_SOURCE = (
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
    # A throwaway project seeded with the shipped example documents — smoke must never read or
    # mutate the tracked projects/dev sandbox.
    project = root / "project"
    documents = project / "documents"
    documents.mkdir(parents=True)
    for tid in EXAMPLE_ORDER:
        shutil.copytree(DOCUMENT_EXAMPLES_DIR / tid, documents / tid)

    # A document script document (048 — one script per document): the engine ticks it every frame, so 200 clean
    # frames prove the App-with-a-scripted-document loop doesn't crash. (Engine-correctness — values/
    # freeze/determinism/play-stop — is the pure-CPU unit test's job; smoke proves the App loop +
    # the binding + the stopped-skip wire.)
    script_document = documents / "script_document"
    (script_document / PASSES_DIR_NAME).mkdir(parents=True)
    (script_document / PASSES_DIR_NAME / pass_shader_name("main")).write_text(
        _SCRIPT_SHADER, encoding="utf-8"
    )
    (script_document / "document.json").write_text(
        json.dumps(_SCRIPT_NODE_JSON), encoding="utf-8"
    )
    script_dir = script_document / "scripts"
    script_dir.mkdir()
    (script_dir / "script.py").write_text(_SCRIPT_SOURCE, encoding="utf-8")
    return project


def _check_invariants(app: App, frame_idx: int) -> None:
    # The "at most one modal popup open" mutex is now structural — popup_state is a single
    # PopupState value, so two modals can't be open at once by construction (feature 023).
    assert isinstance(app.popup_state, PopupState), (
        f"frame {frame_idx}: popup_state is not a PopupState ({app.popup_state!r})"
    )
    assert (
        app.current_document_id == "" or app.current_document_id in app.ui_documents
    ), (
        f"frame {frame_idx}: current_document_id={app.current_document_id!r} not in "
        f"ui_documents={list(app.ui_documents.keys())}"
    )
    # Feature 018: the registry must be populated + dispatched every frame (the
    # cheatsheet overlay draws here too, exercising its no-assert path headlessly).
    assert app.effective_bindings, f"frame {frame_idx}: effective_bindings empty"
    assert app.active_document_tab in DocumentTab, (
        f"frame {frame_idx}: bad active_document_tab={app.active_document_tab!r}"
    )


_ACCUMULATE_SRC = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_prev;
out vec4 fs_color;
void main() { fs_color = vec4(texture(u_prev, vs_uv).r + 0.02, 0.0, 0.0, 1.0); }
"""


def _arm_feedback_canary(app: App) -> Document:
    """Turn one document into a feedback document so the frame loop's swap has an observable.

    A feedback pass advances once per FRAME, but the loop renders the current document twice
    (preview canvas, then its own), so a swap tied to the render call would run at the wrong
    rate — and a still image looks identical either way. This gives the rate a number.
    """
    document_id = next(iter(app.ui_documents))
    document = app.ui_documents[document_id].document
    name = next(iter(document.passes))
    document.passes[name].release_program(_ACCUMULATE_SRC)
    document.passes[name].compile()
    assert document.passes[name].compile_unit.errors == [], (
        f"smoke: the feedback canary shader did not compile: "
        f"{document.passes[name].compile_unit.errors}"
    )
    document.graph = PassGraph(
        output=name, passes={name: PassEntry(inputs={"u_prev": name})}
    )
    document.reset_feedback()
    app.set_current_document_id(document_id)
    return document


def main() -> int:
    configure_logging()

    if not _has_gpu_window():
        logger.warning(
            "smoke: SKIPPED — no GPU window available (display-less box / no hardware GL). "
            "The GUI smoke needs a real glfw window; run it on a machine with a display."
        )
        return _skip_exit_code()

    with tempfile.TemporaryDirectory(prefix="shaderbox-smoke-") as tmp:
        project = _seed_tmp_project(Path(tmp))
        try:
            app = App(project_dir=project, headless=True)
            # An explicit-dir App is never a first run: the examples browser must NOT
            # auto-open (feature 051 — the auto-open is gated on project_dir=None).
            assert app.popup_state == PopupState.CLOSED, (
                f"popup auto-opened for an explicit-dir App ({app.popup_state!r})"
            )
            # Decision-15 regression canary (048): _init opens the restored current document's shader tab
            # (it used to stay blank until a document switch). A non-empty project must have a tab now.
            assert app.editor_tabs, (
                "smoke: editor_tabs empty after _init — the restored current document's shader tab "
                "was not opened (the load->ensure_shader_tab wire is missing)"
            )
            if app.ui_documents:
                app.set_current_document_id(next(iter(app.ui_documents)))
            # 069 W-E: nav is OFF app-wide (D4). Checked here for the same reason the
            # old assertion was: get_io() reads are frame-context-sensitive mid-loop.
            assert not (
                imgui.get_io().config_flags & imgui.ConfigFlags_.nav_enable_keyboard
            ), "nav_enable_keyboard is set; D4 removed app-wide nav"
            feedback_document = _arm_feedback_canary(app)
            canary_id = app.current_document_id
            for frame_idx in range(N_FRAMES):
                update_and_draw(app)
                _check_invariants(app, frame_idx)
                # Exercise the region-cycle + tab-jump wiring (a callback throw surfaces
                # via the except below); nav *behavior* is un-headless-able.
                # The pass list's non-default draw paths: a second pass, the settings modal
                # (its input combos + target controls), and both inline inputs. None of them
                # draw by default, and none can be screenshotted on this box, so the frame
                # loop executing them IS the check.
                if frame_idx == 20:
                    app.session.add_pass(app.current_document_id, "smoke_pass")
                if frame_idx == 25:
                    app.open_pass_settings("smoke_pass")
                if frame_idx == 32:
                    # The tile's delete-✕ wash, which only draws while armed.
                    app.pass_delete_armed = "smoke_pass"
                if frame_idx == 35:
                    app.pass_delete_armed = ""
                    app.pass_add.open(
                        app.session.paths.passes_dir_for(app.current_document_id)
                    )
                if frame_idx == 40:
                    app.pass_add.close()
                    # Deleting the pass while its settings modal is open exercises the modal's
                    # missing-pass close branch on the next frame.
                    assert (
                        app.session.delete_pass(app.current_document_id, "smoke_pass")
                        == ""
                    ), "smoke: the pass list's own pass could not be deleted"
                # The five-tile strip of a genuinely multi-pass document — the shape the panel is
                # for, and the one a single-pass fixture never draws.
                if frame_idx == 42:
                    multi = next(
                        (
                            i
                            for i, u in app.ui_documents.items()
                            if len(u.document.passes) > 1
                        ),
                        "",
                    )
                    assert multi, "smoke: no multi-pass document to draw the strip with"
                    app.set_current_document_id(multi)
                    app.open_pass_settings(
                        sorted(app.ui_documents[multi].document.passes)[0]
                    )
                if frame_idx == 48:
                    # Back to the feedback canary's document, whose accumulation the tail asserts.
                    app.popup_state = PopupState.CLOSED
                    app.pass_settings_name = ""
                    app.set_current_document_id(canary_id)
                if frame_idx == 60:
                    app.focus_document_tab(DocumentTab.RENDER)
                # Open the Examples browser for a stretch so its draw path (grid + desc slot +
                # action row sizing) is exercised — it never opens on its own in the loop.
                if frame_idx == 70:
                    app.popup_state = PopupState.EXAMPLES
                if frame_idx == 90:
                    app.popup_state = PopupState.CLOSED
                # Help goes through the REAL opener (not a popup_state poke) so the
                # section reset is exercised, not just the draw path.
                if frame_idx == 120:
                    app.open_help()
                    assert app.help_section == help_sections()[0].key, (
                        f"open_help did not reset the section ({app.help_section!r})"
                    )
                if frame_idx == 135:
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
            # Canary (048): the script must have BOUND + ticked (binding is by `script.py` existence).
            engine = app.session.script_engine
            driven = engine.script_driven_uniforms("script_document")
            assert "u_a" in driven and "u_b" in driven, (
                f"smoke: the document script did not bind/tick (driven={driven}) — script.py wasn't "
                "discovered/bound"
            )
            # Stopped-skip wire (048): stop u_a, tick once; its WRITE must be skipped (the manual
            # value sticks) while u_a stays driven AND the un-stopped u_b still advances. A dead
            # `stopped` wire would keep writing u_a and green-wash the play/stop model.
            script_document_obj = app.ui_documents["script_document"].document
            script_document_obj.render_pass.uniform_values["u_a"] = -999.0
            b_before = script_document_obj.render_pass.uniform_values["u_b"]
            app.session.set_uniform_stopped("script_document", "u_a", True)
            app.session.tick(["script_document"], t=1.0, dt=0.5, frame=999)
            assert script_document_obj.render_pass.uniform_values["u_a"] == -999.0, (
                "smoke: a stopped uniform was overwritten — the tick(stopped=) skip is unwired"
            )
            assert "u_a" in engine.script_driven_uniforms("script_document"), (
                "smoke: a stopped uniform fell out of the driven set (its play button would vanish)"
            )
            assert script_document_obj.render_pass.uniform_values["u_b"] != b_before, (
                "smoke: the un-stopped u_b did not advance while u_a was stopped — the stop is "
                "freezing the whole script, not just the one uniform"
            )
            # The feedback canary: N frames of +0.02 on an 8-bit target reach ~255 well before
            # the loop ends, so the check is that it advanced AT ALL and did not stall at one
            # step — a missing begin_frame freezes it at 5.
            canary_red = feedback_document.render_pass.canvas.texture.read()[0]
            assert canary_red > 32, (
                f"smoke: the feedback pass advanced to {canary_red} over {N_FRAMES} frames — "
                "the frame loop is not calling Document.begin_frame"
            )
            # The editor drew (feature 067): the active session's last layout produced
            # primitives INCLUDING at least one atlas Glyph (a missing atlas load
            # yields only Missing_Glyph kinds — a background-only frame is not text),
            # and the redraw gate fired at least once.
            editor_session = app.get_current_session_if_exists()
            assert editor_session is not None, "smoke: no editor session after the loop"
            prims = editor_session.editor.prims_list()
            assert prims, "smoke: the editor layout produced no primitives"
            glyphs = sum(1 for p in prims if p.kind == int(EditorKind.GLYPH))
            assert glyphs > 0, (
                "smoke: the editor layout produced no Glyph primitives — shader text "
                "is not reaching the atlas (atlas load or language wiring broken)"
            )
            assert 1 <= app.editor_redraw_count < N_FRAMES, (
                f"smoke: editor_redraw_count={app.editor_redraw_count} over "
                f"{N_FRAMES} frames — 0 means the panel never drew; >= N_FRAMES "
                "means the gate is stuck open and re-renders every frame"
            )
            app.release()
            logger.info(
                f"smoke: OK ({N_FRAMES} frames, {len(app.ui_documents)} documents)"
            )
            return 0
        except Exception as e:
            logger.exception(f"smoke: FAIL — {e}")
            return 1


if __name__ == "__main__":
    sys.exit(main())
