from imgui_bundle import imgui, imgui_ctx

from shaderbox.app import App
from shaderbox.copilot.state import CopilotLayout
from shaderbox.theme import SIZE
from shaderbox.ui_primitives import caption_text, ghost_button, unconnected_gate

# The copilot chat — a floating top-level window (NOT a tab/region). Drawn from
# update_and_draw after the main window closes, like the cheatsheet, so it floats on
# top. no_nav_focus keeps it out of imgui's Ctrl+Tab window-switcher (does NOT block
# click-to-focus — verified in imgui source); no_collapse so it can't be collapsed into
# an unrecoverable thin title-bar strip (the collapse state persists in imgui.ini).
#
# SCAFFOLD: the transcript / input / streaming / tool-call rendering is the capability
# wave. This wave stands up the window, the layout presets, the focus wiring, and the
# not-configured gate.

_WINDOW_FLAGS = imgui.WindowFlags_.no_nav_focus | imgui.WindowFlags_.no_collapse
_NEXT_LAYOUT: dict[CopilotLayout, CopilotLayout] = {
    CopilotLayout.CORNER: CopilotLayout.BOTTOM_STRIP,
    CopilotLayout.BOTTOM_STRIP: CopilotLayout.FREE,
    CopilotLayout.FREE: CopilotLayout.CORNER,
}


def _apply_layout(app: App) -> None:
    # Anchor to the editor child's screen rect (the coding area), NOT the glfw window —
    # the chat belongs over the editor column. CORNER / BOTTOM_STRIP force pos+size every
    # frame (fixed presets). FREE seeds once and lets imgui.ini persist the user's drag.
    ex, ey, ew, eh = app.editor_rect
    margin = float(SIZE.COPILOT_MARGIN)
    if app.copilot_layout == CopilotLayout.CORNER:
        w, h = float(SIZE.COPILOT_W), float(SIZE.COPILOT_H)
        imgui.set_next_window_pos((ex + ew - w - margin, ey + eh - h - margin))
        imgui.set_next_window_size((w, h))
    elif app.copilot_layout == CopilotLayout.BOTTOM_STRIP:
        h = float(SIZE.COPILOT_STRIP_H)
        imgui.set_next_window_pos((ex + margin, ey + eh - h - margin))
        imgui.set_next_window_size((ew - 2.0 * margin, h))
    else:
        imgui.set_next_window_size(
            (float(SIZE.COPILOT_W), float(SIZE.COPILOT_H)),
            imgui.Cond_.first_use_ever,
        )


def draw(app: App) -> None:
    if not app.is_copilot_open:
        app.copilot_focused = False
        return

    if app.copilot_focus_pending:
        imgui.set_next_window_focus()

    _apply_layout(app)
    with imgui_ctx.begin("Copilot", flags=_WINDOW_FLAGS) as window:
        if not window.expanded:
            app.copilot_focused = False
            app.copilot_focus_pending = False
            return

        app.copilot_focused = imgui.is_window_focused(imgui.FocusedFlags_.child_windows)
        # No explicit editor-defocus here: when the chat becomes the focused top-level
        # window imgui makes IT the NavWindow, so the editor's own is_window_focused
        # reads False and it yields the caret on its own (the editor only re-grabs focus
        # on an explicit one-shot, never unconditionally). Arming editor_defocus_requested
        # per-frame would be fatal — code.py consumes it via set_window_focus(None), which
        # clears the GLOBAL NavWindow and would steal the chat's own focus. The region
        # outlines are suppressed while copilot_focused (ui.py / node_grid.py) so none lies.
        # Esc-defocus (the one place we DO programmatically drop the chat's focus):
        if app.copilot_defocus_requested:
            imgui.set_window_focus(None)
            app.copilot_defocus_requested = False
            app.copilot_focused = False

        _draw_top_bar(app)

        if not app.integrations_store.copilot.openrouter_key:
            unconnected_gate(
                not_connected_msg="Copilot is not set up.",
                hint="Add your OpenRouter API key in Settings to enable the copilot.",
                action_label="Open Settings",
                on_action=app.open_settings,
            )
        else:
            caption_text("Copilot chat — coming in a later wave.")

    app.copilot_focus_pending = False


def _draw_top_bar(app: App) -> None:
    if ghost_button(f"Layout: {app.copilot_layout.value}"):
        app.copilot_layout = _NEXT_LAYOUT[app.copilot_layout]
    imgui.same_line()
    if ghost_button("Close"):
        app.is_copilot_open = False
    imgui.separator()
