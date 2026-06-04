import contextlib
from pathlib import Path

import pyperclip
from imgui_bundle import imgui, imgui_ctx

from shaderbox.app import App
from shaderbox.copilot.gate import GateKind
from shaderbox.copilot.state import CopilotLayout, Message, ResultWidget
from shaderbox.copilot.text_render import sanitize_display
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_primitives import (
    active_region_outline,
    caption_text,
    ghost_button,
    labeled_text_input,
    open_path_button,
    open_url_button,
    primary_button,
    unconnected_gate,
)
from shaderbox.util import open_in_file_manager

# The copilot chat — a floating top-level window (NOT a tab/region). Drawn from
# update_and_draw after the main window closes, like the cheatsheet, so it floats on
# top. no_nav_focus keeps it out of imgui's Ctrl+Tab window-switcher (does NOT block
# click-to-focus — verified in imgui source); no_collapse so it can't be collapsed into
# an unrecoverable thin title-bar strip (the collapse state persists in imgui.ini).
# Renders the transcript by message role — incl. the pending_action confirm gate + its
# Recover affordance (feature 020·17) — plus the streaming/input row.

# no_move is layout-conditional (added per-layout in _apply_layout). The fixed presets
# (CORNER/BOTTOM_STRIP) force pos+size every frame, so no_move only stops a stray drag from
# fighting that snap; FREE is user-positioned, so it must stay draggable from the title bar.
_WINDOW_FLAGS = imgui.WindowFlags_.no_nav_focus | imgui.WindowFlags_.no_collapse
_NEXT_LAYOUT: dict[CopilotLayout, CopilotLayout] = {
    CopilotLayout.CORNER: CopilotLayout.BOTTOM_STRIP,
    CopilotLayout.BOTTOM_STRIP: CopilotLayout.FREE,
    CopilotLayout.FREE: CopilotLayout.CORNER,
}


def _apply_layout(app: App) -> int:
    # Anchor to the editor child's screen rect (the coding area), NOT the glfw window —
    # the chat belongs over the editor column. CORNER / BOTTOM_STRIP force pos+size every
    # frame (fixed presets) and add no_move so a drag can't fight the per-frame snap. FREE
    # seeds once, stays draggable, and lets imgui.ini persist the user's position.
    ex, ey, ew, eh = app.editor_rect
    margin = float(SIZE.COPILOT_MARGIN)
    if app.copilot_layout == CopilotLayout.CORNER:
        w, h = float(SIZE.COPILOT_W), float(SIZE.COPILOT_H)
        imgui.set_next_window_pos((ex + ew - w - margin, ey + eh - h - margin))
        imgui.set_next_window_size((w, h))
        return imgui.WindowFlags_.no_move
    if app.copilot_layout == CopilotLayout.BOTTOM_STRIP:
        h = float(SIZE.COPILOT_STRIP_H)
        imgui.set_next_window_pos((ex + margin, ey + eh - h - margin))
        imgui.set_next_window_size((ew - 2.0 * margin, h))
        return imgui.WindowFlags_.no_move
    imgui.set_next_window_size(
        (float(SIZE.COPILOT_W), float(SIZE.COPILOT_H)),
        imgui.Cond_.first_use_ever,
    )
    return 0


def draw(app: App) -> None:
    if not app.is_copilot_open:
        app.copilot_focused = False
        app.copilot_hovered = False
        return

    if app.copilot_focus_pending:
        imgui.set_next_window_focus()

    flags = _WINDOW_FLAGS | _apply_layout(app)
    with imgui_ctx.begin("Copilot", flags=flags) as window:
        if not window.expanded:
            app.copilot_focused = False
            app.copilot_hovered = False
            app.copilot_focus_pending = False
            return

        app.copilot_focused = imgui.is_window_focused(imgui.FocusedFlags_.child_windows)
        # Re-assert the focus request each frame until it lands — a bar-button click
        # focuses the bar the same frame and races set_next_window_focus, so one attempt
        # can lose (settles in 1-2 frames).
        if app.copilot_focus_pending and app.copilot_focused:
            app.copilot_focus_pending = False
        # Neutralize the editor's direct io.mouse_down read (code.py) while the mouse is
        # over the chat OR dragging/resizing it — a resize moves the cursor outside the
        # window, which is_window_hovered alone misses. The editor doesn't need an
        # explicit defocus: focusing the chat makes it the NavWindow, so the editor's own
        # is_window_focused reads False and it yields the caret.
        hovered = imgui.is_window_hovered(imgui.HoveredFlags_.child_windows)
        dragging = app.copilot_focused and imgui.is_mouse_down(imgui.MouseButton_.left)
        app.copilot_hovered = hovered or dragging
        # Esc requests a programmatic defocus (set_window_focus(None)).
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
            _draw_transcript(app)


def _draw_transcript(app: App) -> None:
    state = app.copilot.state
    input_h = imgui.get_frame_height_with_spacing() + float(SPACE.SM)
    avail = imgui.get_content_region_avail()
    if imgui.begin_child("##copilot_history", size=(0.0, avail.y - input_h)):
        for i, msg in enumerate(state.messages):
            _draw_message(app, msg, i)
        if state.streaming_text:
            _draw_message(app, Message(role="assistant", text=state.streaming_text), -1)
        if state.in_flight and not state.streaming_text:
            # The transient status line (D1) in place of a bare caption — the latest per-tool
            # phrase while a turn runs, falling back to "thinking..." before the first status.
            caption_text(
                sanitize_display(state.status) if state.status else "thinking..."
            )
        if state.in_flight:
            imgui.set_scroll_here_y(1.0)
    imgui.end_child()

    in_flight = state.in_flight
    if app.copilot_focus_pending:
        imgui.set_keyboard_focus_here()
    imgui.set_next_item_width(-1.0 if in_flight else _send_button_offset())
    submitted, app.copilot_input = imgui.input_text(
        "##copilot_input",
        app.copilot_input,
        flags=imgui.InputTextFlags_.enter_returns_true,
    )
    if not in_flight:
        imgui.same_line()
        if (primary_button("Send") or submitted) and app.copilot_input.strip():
            app.copilot_send(app.copilot_input)
            app.copilot_input = ""
            imgui.set_keyboard_focus_here(-1)
    else:
        imgui.same_line()
        if ghost_button("Stop"):
            app.copilot.cancel_turn()


def _draw_message(app: App, msg: Message, idx: int) -> None:
    # Sanitize at DRAW so a glyph the font can't render never reaches the atlas (D2). The
    # committed Message text is already ASCII (materialize-time sanitize) — this is idempotent;
    # the live streaming preview (raw streaming_text) is sanitized HERE, tear-safe on the full
    # accumulated string rather than per-delta.
    text = sanitize_display(msg.text)
    role = msg.role
    if role == "user":
        imgui.text_colored(COLOR.ACCENT_PRIMARY, "you")
        imgui.text_wrapped(text)
        _copy_affordance(text, idx)
    elif role == "assistant":
        imgui.text_wrapped(text)
        _copy_affordance(text, idx)
    elif role == "tool_status":
        caption_text(text)
        if msg.result_widget is not None:
            _draw_result_widget(msg.result_widget, idx)
    elif role == "error":
        imgui.text_colored(COLOR.STATE_ERROR, text)
    elif role == "pending_action":
        _draw_pending_action(app, msg, idx)
    imgui.separator()


def _copy_affordance(text: str, idx: int) -> None:
    # A copy-to-clipboard for a chat message (feature 020·23 D7): prose isn't natively selectable in
    # imgui, so this gets the text OUT. A small always-visible ghost (no hover-jitter); copies the
    # whole message.
    if ghost_button(f"Copy##msg_copy_{idx}"):
        with contextlib.suppress(pyperclip.PyperclipException):
            pyperclip.copy(text)


def _draw_result_widget(widget: ResultWidget, idx: int) -> None:
    # A first-class result button the engine renders from a tool's structured outcome (feature 020·21).
    # Open-only: a url opens the browser, a path opens the file manager; neither copies. An unknown
    # kind draws nothing (fail-soft for a future kind loaded on an older build). The ##_{idx} id keeps
    # two transcript widgets from colliding.
    label = sanitize_display(widget.label)
    if widget.kind == "open_url":
        open_url_button(label, widget.target, id_=f"##rw_url_{idx}")
    elif widget.kind == "open_path":
        open_path_button(
            label,
            widget.target,
            lambda p: open_in_file_manager(Path(p)),
            id_=f"##rw_path_{idx}",
        )


def _draw_pending_action(app: App, msg: Message, idx: int) -> None:
    imgui.text_wrapped(sanitize_display(msg.text))
    if not msg.resolved:
        if msg.gate_kind is GateKind.CREDENTIAL:
            _draw_credential_input(app, msg, idx)
        elif msg.gate_kind is GateKind.CONFIG:
            _draw_config_panel(app, msg, idx)
        else:
            if primary_button(f"Yes##gate_yes_{idx}"):
                app.copilot.answer_gate(approved=True)
            imgui.same_line()
            if ghost_button(f"No##gate_no_{idx}"):
                app.copilot.answer_gate(approved=False)
        return
    recover = msg.recover
    if recover is None:
        return
    if recover.done:
        # done True + the node back in play = recovered; gone = the trash was cleared.
        caption_text(
            "Recovered" if recover.node_id in app.ui_nodes else "No longer recoverable"
        )
        return
    # Disabled while a turn runs: a recover mutates ui_nodes/current-node, which the
    # in-flight turn owns (matches the Clear button's in_flight guard).
    imgui.begin_disabled(app.copilot.state.in_flight)
    if ghost_button(f"Recover##gate_recover_{idx}"):
        app.recover_deleted_node(msg)
    imgui.end_disabled()


def _draw_config_panel(app: App, msg: Message, idx: int) -> None:
    # An inline integration-setup gate (feature 020·21): render the exporter's EXISTING draw_config_ui()
    # verbatim (same widgets as Settings — instruction + paste client_secret + Connect), plus a Cancel
    # button absent in Settings. The exporter's own _progress_queue is drained by the global
    # share_tab.update pump each frame, so we DON'T pump here (a second drain would steal events). When
    # the exporter reaches connected the gate auto-resolves approved=True (the worker's connect await
    # then sees it); Cancel resolves approved=False and the agent explains it couldn't proceed.
    exporter = app.exporter_registry.get(msg.gate_integration)
    if exporter is None:
        if ghost_button(f"Cancel##gate_cfg_cancel_{idx}"):
            app.copilot.answer_gate(approved=False)
        return
    if exporter.is_connected():
        app.copilot.answer_gate(approved=True)
        return
    exporter.draw_config_ui()
    imgui.dummy(imgui.ImVec2(0, float(SPACE.SM)))
    if ghost_button(f"Cancel##gate_cfg_cancel_{idx}"):
        app.copilot.answer_gate(approved=False)


def _draw_credential_input(app: App, msg: Message, idx: int) -> None:
    # A masked secret input for a CREDENTIAL gate (feature 020·19). The typed value lives only
    # in msg.gate_input (UI-only, never persisted); Save hands it to answer_gate_credential,
    # which redacts the card echo. Cancel = a plain decline.
    msg.gate_input = labeled_text_input(
        f"##gate_secret_{idx}",
        msg.gate_input,
        float(SIZE.SHARE_PREVIEW_W),
        password=True,
    )
    if primary_button(f"Save##gate_save_{idx}"):
        app.copilot.answer_gate_credential(msg.gate_input)
    imgui.same_line()
    if ghost_button(f"Cancel##gate_cancel_{idx}"):
        app.copilot.answer_gate(approved=False)


def _send_button_offset() -> float:
    return -(float(SIZE.BTN_SM_W) + float(SPACE.SM))


def _draw_top_bar(app: App) -> None:
    if ghost_button(f"Layout: {app.copilot_layout.value}"):
        app.copilot_layout = _NEXT_LAYOUT[app.copilot_layout]
    imgui.same_line()
    # Clear archives the conversation then resets to an empty chat (feature 022). Disabled
    # mid-turn so it can't bypass the in_flight gate the reset relies on.
    imgui.begin_disabled(app.copilot.state.in_flight)
    if ghost_button("Clear"):
        app.copilot_clear_chat()
    imgui.end_disabled()
    imgui.same_line()
    if ghost_button("Close"):
        app.is_copilot_open = False
    imgui.separator()
