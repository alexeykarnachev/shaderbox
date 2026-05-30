from imgui_bundle import imgui, imgui_ctx

from shaderbox.app import App
from shaderbox.copilot.state import CopilotLayout
from shaderbox.theme import SIZE
from shaderbox.ui_primitives import (
    active_region_outline,
    caption_text,
    ghost_button,
    unconnected_gate,
)

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
        app.copilot_hovered = False
        return

    if app.copilot_focus_pending:
        imgui.set_next_window_focus()

    _apply_layout(app)
    with imgui_ctx.begin("Copilot", flags=_WINDOW_FLAGS) as window:
        if not window.expanded:
            app.copilot_focused = False
            app.copilot_hovered = False
            app.copilot_focus_pending = False
            return

        app.copilot_focused = imgui.is_window_focused(imgui.FocusedFlags_.child_windows)
        # Clear the focus-pending one-shot only once the focus actually TOOK. A bar-button
        # click races the chat's set_next_window_focus (the click focuses the bar/main
        # window the same frame), so a single attempt can lose; re-asserting each frame
        # until copilot_focused makes it robust (settles in 1-2 frames).
        if app.copilot_focus_pending and app.copilot_focused:
            app.copilot_focus_pending = False
        # Mouse over the chat -> neutralize the editor's direct-mouse read (code.py) so a
        # drag inside the chat can't select editor text beneath it.
        app.copilot_hovered = imgui.is_window_hovered(imgui.HoveredFlags_.child_windows)
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

        # The accent outline when the chat owns focus — the same active-region cue the
        # editor/grid/panel use, so "focused" reads consistently across the app. Not
        # while a modal is open (the outline is on the foreground draw list, immune to
        # window clip, so it would render over the popup).
        if app.copilot_focused and not app.any_popup_open():
            active_region_outline()

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


def _draw_top_bar(app: App) -> None:
    if ghost_button(f"Layout: {app.copilot_layout.value}"):
        app.copilot_layout = _NEXT_LAYOUT[app.copilot_layout]
    imgui.same_line()
    if ghost_button("Close"):
        app.is_copilot_open = False
    imgui.separator()
