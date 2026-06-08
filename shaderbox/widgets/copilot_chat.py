import contextlib
from pathlib import Path

import pyperclip
from imgui_bundle import imgui, imgui_ctx

from shaderbox.app import App
from shaderbox.copilot.gate import GateKind
from shaderbox.copilot.sanitize import sanitize_display
from shaderbox.copilot.state import (
    CopilotLayout,
    Message,
    ResultWidget,
    context_gauge_readout,
)
from shaderbox.theme import COLOR, SIZE, SPACE, fade
from shaderbox.ui_primitives import (
    active_region_outline,
    caption_text,
    copy_icon_button,
    danger_button,
    gauge_bar,
    ghost_button,
    labeled_text_input,
    layout_icon_button,
    message_bubble,
    open_path_button,
    open_url_button,
    primary_button,
    unconnected_gate,
)
from shaderbox.util import open_in_file_manager

# Floating top-level window (NOT a tab/region), drawn after the main window so it floats on
# top. no_nav_focus keeps it out of Ctrl+Tab (does NOT block click-to-focus); no_collapse
# stops it collapsing into an unrecoverable title-bar strip. no_nav_inputs stops the nav
# outline on the programmatically-focused input (still typeable; /imgui-ui §8).
# no_move is layout-conditional (added in _apply_layout): the fixed presets force pos+size
# every frame so no_move just stops a drag fighting the snap; FREE must stay draggable.
_WINDOW_FLAGS = (
    imgui.WindowFlags_.no_nav_focus
    | imgui.WindowFlags_.no_collapse
    | imgui.WindowFlags_.no_nav_inputs
    # The feed (an inner child) owns the only scrollbar; the window itself never scrolls — its
    # content (header + feed + splitter + input) is sized to exactly fit, never overflow.
    | imgui.WindowFlags_.no_scrollbar
    | imgui.WindowFlags_.no_scroll_with_mouse
)


def _apply_layout(app: App) -> int:
    # Anchor to the editor child's screen rect, NOT the glfw window. CORNER / BOTTOM_STRIP
    # force pos+size every frame and add no_move; FREE seeds once and stays draggable.
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
    entering_free = app.copilot_prev_layout != CopilotLayout.FREE
    if entering_free and app.copilot_free_rect is not None:
        fx, fy, fw, fh = app.copilot_free_rect
        imgui.set_next_window_pos((fx, fy), imgui.Cond_.always)
        imgui.set_next_window_size((fw, fh), imgui.Cond_.always)
    else:
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

    # Hold focus on the chat while a turn runs: the editor steals focus when the copilot
    # creates/switches the current node (TextEditor first-render grab, /imgui-ui §8), which
    # would route a keystroke into the editor. Re-assert every frame, not just the one-shot.
    if app.copilot_focus_pending or app.copilot.state.in_flight:
        imgui.set_next_window_focus()

    flags = _WINDOW_FLAGS | _apply_layout(app)
    # Floor the width so the header row (icon + usage bars + Clear/Close) can't be narrowed
    # until the right-aligned cluster overlaps the bars. Height floor is nominal.
    min_w: float = float(
        SIZE.BTN_SM_H + SIZE.USAGE_BARS_W + 2 * SIZE.BTN_SM_W + 4 * SPACE.LG
    )
    imgui.set_next_window_size_constraints((min_w, 120.0), (1.0e4, 1.0e4))
    with imgui_ctx.begin("Copilot", flags=flags) as window:
        if app.copilot_layout == CopilotLayout.FREE:
            pos = imgui.get_window_pos()
            size = imgui.get_window_size()
            app.copilot_free_rect = (pos.x, pos.y, size.x, size.y)
        app.copilot_prev_layout = app.copilot_layout

        if not window.expanded:
            app.copilot_focused = False
            app.copilot_hovered = False
            app.copilot_focus_pending = False
            return

        app.copilot_focused = imgui.is_window_focused(imgui.FocusedFlags_.child_windows)
        # Neutralize the editor's direct io.mouse_down read (code.py) while hovering OR
        # dragging the chat — a resize moves the cursor outside, which is_window_hovered
        # alone misses. No explicit editor defocus needed: focusing the chat makes it the
        # NavWindow, so the editor's is_window_focused reads False and yields the caret.
        hovered = imgui.is_window_hovered(imgui.HoveredFlags_.child_windows)
        dragging = app.copilot_focused and imgui.is_mouse_down(imgui.MouseButton_.left)
        app.copilot_hovered = hovered or dragging
        if app.copilot_defocus_requested:
            imgui.set_window_focus(None)
            app.copilot_defocus_requested = False
            app.copilot_focused = False

        # Focus outline on the foreground list so it covers the title bar (the content clip
        # would cut it). Skipped under a modal — foreground would render over the popup.
        if app.copilot_focused and not app.any_popup_open():
            active_region_outline(foreground=True)

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


_MIN_INPUT_H: float = 40.0
_MIN_FEED_H: float = 60.0
# Core divider thickness; the actual hit-band also absorbs the item-spacing on each side (see
# _draw_input_splitter) so the whole feed->input gap is one hit-rect — no dead zone to flicker
# the resize cursor — with the grab line centred in it.
_SPLITTER_CORE_H: float = 4.0


def _splitter_band_h() -> float:
    # The full feed->input gap the splitter owns: core + the item-spacing above AND below it.
    return _SPLITTER_CORE_H + 2.0 * imgui.get_style().item_spacing.y


def _draw_input_splitter(app: App, avail_y: float) -> None:
    # The feed/input divider: drag to trade height between the feed (above) and the input
    # (below). The input keeps its set height on window resize; the FEED flexes. Mirrors the
    # editor splitter idiom in ui.py (imgui has no built-in sibling-splitter widget). The button
    # is pulled UP over the spacing after the feed and spans the whole gap (core + both spacings),
    # so the hit-rect runs feed-edge to input-edge (no dead zone) and the line is the true centre.
    spacing_y = imgui.get_style().item_spacing.y
    pos = imgui.get_cursor_pos()
    imgui.set_cursor_pos((pos.x, pos.y - spacing_y))
    imgui.invisible_button(
        "##copilot_input_splitter", imgui.ImVec2(-1.0, _splitter_band_h())
    )
    if imgui.is_item_hovered() or imgui.is_item_active():
        app.want_cursor = app.resize_ns_cursor
    if imgui.is_item_active():
        dy = imgui.get_io().mouse_delta.y
        if dy:  # drag down -> smaller input; drag up -> larger input
            hi = max(_MIN_INPUT_H, avail_y - _MIN_FEED_H)
            app.app_state.copilot_input_h = max(
                _MIN_INPUT_H, min(hi, app.app_state.copilot_input_h - dy)
            )
    rmin, rmax = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    cx, cy = (rmin.x + rmax.x) * 0.5, (rmin.y + rmax.y) * 0.5
    imgui.get_window_draw_list().add_line(
        (cx - 12.0, cy),
        (cx + 12.0, cy),
        imgui.color_convert_float4_to_u32(COLOR.BORDER),
        1.0,
    )


def _draw_transcript(app: App) -> None:
    state = app.copilot.state
    avail_y = imgui.get_content_region_avail().y
    # Input keeps its stored height (clamped); the FEED takes the rest -> the input never gets
    # hidden by a window resize, and the splitter trades height between the two.
    input_h = max(
        _MIN_INPUT_H, min(app.app_state.copilot_input_h, avail_y - _MIN_FEED_H)
    )
    # Feed fills everything EXCEPT the reserved footer (splitter band + input). A NEGATIVE child
    # height = "available minus this" — self-adjusting, so the column can't overflow the window
    # (no second, window-level scrollbar) regardless of spacing accounting.
    footer_h = _splitter_band_h() + input_h
    if imgui.begin_child("##copilot_history", size=(0.0, -footer_h)):
        # Stick-to-bottom: read the scroll position BEFORE this frame's content is laid out (it
        # reflects last frame's layout). If the view was at the bottom, re-pin after drawing; if
        # the user scrolled up even slightly, stop auto-scrolling so they can read while a turn
        # streams. On first draw scroll_max is 0 -> at-bottom -> opens pinned to the bottom.
        stick = imgui.get_scroll_y() >= imgui.get_scroll_max_y() - 1.0
        for i, msg in enumerate(state.messages):
            _draw_message(app, msg, i)
        if state.streaming_text:
            _draw_message(app, Message(role="assistant", text=state.streaming_text), -1)
        if state.in_flight and not state.streaming_text:
            # Latest per-tool status phrase; "thinking..." before the first status.
            caption_text(
                sanitize_display(state.status) if state.status else "thinking..."
            )
        if stick:
            imgui.set_scroll_here_y(1.0)
    imgui.end_child()

    _draw_input_splitter(app, avail_y)

    in_flight = state.in_flight
    # One-shot, never every frame (/imgui-ui §7.5).
    if app.copilot_focus_pending:
        imgui.set_keyboard_focus_here(0)
        app.copilot_focus_pending = False

    # ONE layout for both states — same input box + same trailing button slot. Only what
    # changes by state changes: the input is frozen mid-turn, and the slot is Send (idle) vs
    # Stop (working). The geometry is identical, so the row never shifts between modes.
    btn_w: float = float(SIZE.BTN_SM_W)
    imgui.begin_disabled(in_flight)
    submitted, app.copilot_input = imgui.input_text_multiline(
        "##copilot_input",
        app.copilot_input,
        imgui.ImVec2(_send_button_offset(), input_h),
        flags=imgui.InputTextFlags_.enter_returns_true
        | imgui.InputTextFlags_.ctrl_enter_for_new_line
        | imgui.InputTextFlags_.word_wrap,
    )
    imgui.end_disabled()
    if submitted:
        app.copilot_focus_pending = True
    imgui.same_line()
    if in_flight:
        if ghost_button("Stop", width=btn_w):
            app.copilot.cancel_turn()
    elif (
        primary_button("Send", width=btn_w) or submitted
    ) and app.copilot_input.strip():
        app.copilot_send(app.copilot_input)
        app.copilot_input = ""


def _draw_message(app: App, msg: Message, idx: int) -> None:
    # Sanitize at DRAW so an unrenderable glyph never reaches the atlas. Idempotent on
    # committed (already-ASCII) text; on the live streaming preview it runs on the full
    # accumulated string (tear-safe), not per-delta.
    text = sanitize_display(msg.text)
    role = msg.role
    if role == "user":
        _draw_bubble(
            "you", COLOR.ACCENT_PRIMARY, fade(COLOR.ACCENT_PRIMARY, 0.08), text, idx
        )
    elif role == "assistant":
        _draw_bubble("copilot", COLOR.STATE_INFO, COLOR.BG_SURFACE, text, idx)
    elif role == "tool_status":
        caption_text(text)
        if msg.result_widget is not None:
            _draw_result_widget(msg.result_widget, idx)
        imgui.dummy(imgui.ImVec2(0, float(SPACE.SM)))
    elif role == "error":
        imgui.text_colored(COLOR.STATE_ERROR, text)
        imgui.dummy(imgui.ImVec2(0, float(SPACE.SM)))
    elif role == "pending_action":
        _draw_pending_action(app, msg, idx)
        imgui.dummy(imgui.ImVec2(0, float(SPACE.SM)))


def _draw_bubble(
    name: str,
    name_color: tuple[float, float, float, float],
    bg: tuple[float, float, float, float],
    text: str,
    idx: int,
) -> None:
    # A chat bubble: a colored name header + wrapped text, with a corner copy glyph pinned
    # top-right (imgui prose isn't selectable, so copy is the affordance). The bubble gap, not
    # a separator, divides messages.
    side: float = float(SIZE.ROW_HEIGHT)
    with message_bubble(f"##bubble_{idx}", bg) as origin:
        imgui.text_colored(name_color, name)
        imgui.text_wrapped(text)
        # Copy glyph at the bubble's top-right inner corner; allow_overlap so it wins the click.
        avail_w = imgui.get_content_region_avail().x + float(SPACE.MD)
        imgui.set_next_item_allow_overlap()
        imgui.set_cursor_screen_pos((origin.x + avail_w - side, origin.y))
        if copy_icon_button(f"copy_{idx}", side):
            with contextlib.suppress(pyperclip.PyperclipException):
                pyperclip.copy(text)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Copy")
    imgui.dummy(imgui.ImVec2(0, float(SPACE.SM)))


def _draw_result_widget(widget: ResultWidget, idx: int) -> None:
    # Open-only result button. An unknown kind draws nothing (fail-soft for a kind a
    # newer build emits). The ##_{idx} id keeps transcript widgets from colliding.
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
        # Node back in ui_nodes = recovered; gone = the trash was cleared.
        caption_text(
            "Recovered" if recover.node_id in app.ui_nodes else "No longer recoverable"
        )
        return
    # Disabled while a turn runs: a recover mutates ui_nodes, which the in-flight turn owns.
    imgui.begin_disabled(app.copilot.state.in_flight)
    if ghost_button(f"Recover##gate_recover_{idx}"):
        app.recover_deleted_node(msg)
    imgui.end_disabled()


def _draw_config_panel(app: App, msg: Message, idx: int) -> None:
    # Inline integration-setup gate: render the exporter's own draw_config_ui() plus a Cancel.
    # Do NOT pump its _progress_queue here — the global share_tab.update pump drains it; a
    # second drain steals events. Connected auto-resolves approved=True; Cancel resolves False.
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
    # Masked secret input. gate_input is UI-only, never persisted; answer_gate_credential
    # redacts the card echo.
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
    # Reserve the button width + the exact gap same_line() inserts (style item_spacing.x), so
    # the trailing button lands flush against the right content edge with no leftover sliver.
    return -(float(SIZE.BTN_SM_W) + imgui.get_style().item_spacing.x)


def _draw_top_bar(app: App) -> None:
    # Row: [layout icon] [context gauge ............] [Clear][Close]. One arithmetic owner so the
    # gauge width and the right-aligned cluster x can't drift apart.
    content_w: float = imgui.get_content_region_avail().x
    icon_side: float = float(SIZE.BTN_SM_H)
    btn_pad: float = 2.0 * float(SPACE.MD)
    clear_w: float = imgui.calc_text_size("Clear").x + btn_pad
    close_w: float = imgui.calc_text_size("Close").x + btn_pad
    cluster_w: float = clear_w + float(SPACE.SM) + close_w
    cluster_x: float = content_w - cluster_w
    # Leave a clear breathing gap between the gauge and the Clear button (SPACE.LG), beyond the
    # icon's same_line gap on the left.
    gauge_w: float = max(
        float(SIZE.USAGE_BARS_W),
        cluster_x - icon_side - float(SPACE.MD) - float(SPACE.LG),
    )

    if layout_icon_button("copilot_layout", app.copilot_layout.variant, icon_side):
        app.cycle_copilot_layout()
    if imgui.is_item_hovered():
        imgui.set_tooltip(f"Layout: {app.copilot_layout.value}")

    imgui.same_line()
    fraction, tooltip = context_gauge_readout(
        app.copilot.state.last_turn, app.copilot.state.session_cost_usd
    )
    gauge_bar("copilot_context", fraction, tooltip, gauge_w)

    imgui.same_line(cluster_x)
    # Disabled mid-turn so it can't bypass the in_flight gate the reset relies on.
    imgui.begin_disabled(app.copilot.state.in_flight)
    if danger_button("Clear", width=clear_w):
        app.copilot_clear_chat()
    imgui.end_disabled()
    imgui.same_line()
    if ghost_button("Close", width=close_w):
        app.is_copilot_open = False
    imgui.separator()
