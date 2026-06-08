from pathlib import Path

from imgui_bundle import imgui
from imgui_bundle import imgui_color_text_edit as text_edit

from shaderbox.app import App
from shaderbox.editor_types import HoverMark, JumpRequest
from shaderbox.shader_errors import ShaderError
from shaderbox.theme import COLOR, EDITOR_UNFOCUSED_ALPHA, SIZE, SPACE, fade
from shaderbox.ui_primitives import draw_copyable_text
from shaderbox.util import format_auto_value

_MAX_ERROR_ROWS = 3


def _apply_markers(
    editor: text_edit.TextEditor,
    errors: list[ShaderError],
    hover: HoverMark | None,
    current_path: Path,
) -> None:
    editor.clear_markers()
    # Markers are line fills — translucent or they hide the glyphs. Only errors whose
    # `path` matches the open file mark its gutter.
    err_color = imgui.color_convert_float4_to_u32(fade(COLOR.STATE_ERROR, 0.35))
    for err in errors:
        if err.line >= 0 and err.path == current_path:
            editor.add_marker(err.line, err_color, err_color, err.message, err.message)
    if hover is not None and hover.path == current_path:
        accent = imgui.color_convert_float4_to_u32(fade(COLOR.ACCENT_PRIMARY, 0.15))
        editor.add_marker(hover.line, accent, accent, "", "")


def _consume_jump(app: App, editor: text_edit.TextEditor, current_path: Path) -> bool:
    req = app.editor_jump_request
    if req is None:
        return False
    # A request for a different file is stale (one editor only); clear it.
    if req.path != current_path:
        app.editor_jump_request = None
        return False
    app.editor_jump_request = None
    editor.set_cursor(text_edit.TextEditor.DocPos(req.line, req.column))
    editor.select_line(req.line)
    editor.scroll_to_line(req.line, text_edit.TextEditor.Scroll.align_middle)
    editor.set_focus()
    return True


def _draw_error_strip(
    app: App, errors: list[ShaderError], height: float, current_path: Path
) -> None:
    imgui.push_style_color(imgui.Col_.child_bg, COLOR.BG_SURFACE)
    if imgui.begin_child("##shader_errors", size=(0.0, height)):
        n = len(errors)
        if n > 1:
            imgui.text_colored(COLOR.FG_DIM, f"{n} errors  (F8: next)")
        for i, err in enumerate(errors[:_MAX_ERROR_ROWS]):
            label = (
                err.message
                if err.line < 0
                else f"Line {err.line + 1}  ·  {err.message}"
            )
            imgui.push_style_color(imgui.Col_.text, COLOR.STATE_ERROR)
            clicked = imgui.selectable(f"{label}##err{i}", False)[0]
            imgui.pop_style_color(1)
            if clicked and err.line >= 0:
                app.editor_jump_request = JumpRequest(err.path, err.line, 0)
        extra = n - _MAX_ERROR_ROWS
        if extra > 0:
            imgui.text_colored(COLOR.FG_DIM, f"+{extra} more")
    imgui.end_child()
    imgui.pop_style_color(1)


def draw_chrome(app: App) -> None:
    # The editor's status chrome — file path + dirty/compiled state + nav links.
    if not (ui_node := app.ui_nodes.get(app.current_node_id)):
        imgui.text_colored(COLOR.FG_DIM, "No node selected")
        return
    current_path = app.current_editor_path
    if current_path is None:
        imgui.text_colored(COLOR.FG_DIM, "No file open")
        return
    is_lib = current_path != ui_node.node.source.path

    if is_lib:
        imgui.text_colored(COLOR.FG_DIM, str(current_path.name))
        imgui.same_line(spacing=float(SPACE.LG))
        if imgui.button("< back to node", size=(SIZE.BTN_SM_W * 1.5, 0)):
            app.show_node_editor()
    else:
        full_file_path = ui_node.node.source.path
        local_file_path = full_file_path.relative_to(app.project_dir)
        if draw_copyable_text(str(local_file_path), copy_value=str(full_file_path)):
            app.notifications.push("Copied to clipboard!")
        if app.is_current_editor_dirty():
            imgui.same_line()
            imgui.text_colored(COLOR.STATE_WARN, "(unsaved)")
        elif not ui_node.node.compile_unit.error_raw:
            imgui.same_line()
            imgui.text_colored(COLOR.STATE_OK, "compiled")
        imgui.same_line(spacing=float(SPACE.LG))
        if imgui.button("Open dir", size=(SIZE.BTN_SM_W, 0)):
            app.open_current_node_dir()
        if app.last_shader_lib_path is not None and app.last_shader_lib_path.exists():
            imgui.same_line()
            if imgui.button(
                f"back to {app.last_shader_lib_path.name} >",
                size=(SIZE.BTN_SM_W * 1.8, 0),
            ):
                app.open_shader_lib_file(app.last_shader_lib_path)


def draw(app: App) -> None:
    app.code_hovered_uniform = ""
    if not (ui_node := app.ui_nodes.get(app.current_node_id)):
        app.editor_focused = False
        return

    # The editor shows the node's shader OR a lib file the user opened.
    current_path = app.current_editor_path
    if current_path is None:
        app.editor_focused = False
        return
    is_lib = current_path != ui_node.node.source.path

    # render() FPEs while a popup is open (conventions.md ## Known quirks) — skip drawing it.
    if app.any_popup_open():
        app.editor_focused = False
        return

    session = app.editor_sessions.get(current_path)
    if session is None:
        session = (
            app.open_shader_lib_file(current_path)
            if is_lib
            else app.get_session(ui_node.node.source)
        )
    editor = session.editor
    settings = app.app_state.editor_settings

    # Markers come from the active node's compile unit, filtered by path in _apply_markers.
    errors = ui_node.node.compile_unit.errors
    _apply_markers(editor, errors, app.editor_hover_line, current_path)
    app.editor_hover_line = None
    strip_height = 0.0
    if errors:
        n = len(errors)
        # capped rows + "+N more" line + "N errors" header
        rows = min(n, _MAX_ERROR_ROWS) + (n > _MAX_ERROR_ROWS) + (n > 1)
        # Measure in the strip's own font (font_12), not the ambient UI font.
        imgui.push_font(app.font_12, app.font_12.legacy_size)
        strip_height = (
            rows * imgui.get_text_line_height_with_spacing()
            + 2.0 * imgui.get_style().window_padding.y
        )
        imgui.pop_font()

    imgui.push_font(app.font_14, float(settings.font_size))

    editor_pos = imgui.get_cursor_screen_pos()
    editor_size = imgui.get_content_region_avail()
    editor_size.y = max(0.0, editor_size.y - strip_height)
    editor_max = imgui.ImVec2(
        editor_pos.x + editor_size.x, editor_pos.y + editor_size.y
    )
    hovering = imgui.is_mouse_hovering_rect(editor_pos, editor_max)

    # Consume the Ctrl+scroll wheel BEFORE render() so the editor doesn't also scroll on it
    io = imgui.get_io()
    if hovering and io.key_ctrl and io.mouse_wheel != 0.0:
        new_size = settings.font_size + int(io.mouse_wheel)
        settings.font_size = max(8, min(48, new_size))
        io.mouse_wheel = 0.0

    # Jump/focus requests latch for the upcoming render(); must run before it.
    jumped = _consume_jump(app, editor, current_path)

    focus_requested = app.editor_focus_requested
    if focus_requested:
        editor.set_focus()
        app.editor_focus_requested = False
        app.editor_was_ever_focused = True

    # Dim the pane while unfocused. editor_focused is last frame's value (set after render).
    dim = not app.editor_focused and not jumped and not focus_requested
    if dim:
        imgui.push_style_var(imgui.StyleVar_.alpha, EDITOR_UNFOCUSED_ALPHA)

    # Lock read-only during a copilot turn. Set every frame — the active session can change.
    # Programmatic set_text is unaffected by read-only.
    editor.set_read_only_enabled(app.copilot_turn_active)

    # TextEditor reads io.mouse_down[0] directly, bypassing imgui's hit-test (begin_disabled
    # / z-order do nothing) — force it False for this render so it can't select under a
    # splitter drag or the copilot chat, then restore before the splitter reads it.
    if app.splitter_dragging or app.copilot_hovered:
        prev_down = bool(io.mouse_down[0])
        io.mouse_down[0] = False
        editor.render("##glsl_editor", size=editor_size)
        io.mouse_down[0] = prev_down
    else:
        editor.render("##glsl_editor", size=editor_size)

    if dim:
        imgui.pop_style_var()

    # No is-focused getter on the editor — read imgui's child-window focus instead.
    app.editor_focused = imgui.is_window_focused(imgui.FocusedFlags_.child_windows)
    if app.editor_focused:
        # Sticky: stays True across popups/menus until an explicit defocus.
        app.editor_was_ever_focused = True

    # A freshly-rendered editor auto-grabs focus, so clear it AFTER render, not before.
    if app.editor_defocus_requested:
        imgui.set_window_focus(None)
        app.editor_defocus_requested = False
        app.editor_focused = False
        app.editor_was_ever_focused = False

    # Drive the glfw cursor directly — imgui cursors are no-op here (conventions.md ## Known
    # quirks). is_window_hovered respects popup-blocking (is_mouse_hovering_rect doesn't). Also
    # NOT over the editor when the floating chat is hovered: it owns its own cursor (the splitter's
    # resize cursor), and the editor's rect-test reads "hovered" THROUGH the chat — so without this
    # the two fight every frame and the cursor blinks (this also suppresses the glyph tooltip there).
    cursor_over_editor = (
        hovering
        and imgui.is_window_hovered(imgui.HoveredFlags_.child_windows)
        and not app.copilot_hovered
    )
    if cursor_over_editor:
        app.want_cursor = app.ibeam_cursor

    # Cursor-following tooltip for words that are live uniforms; also lights up the panel row.
    if cursor_over_editor and editor.is_mouse_pos_over_glyph(imgui.get_mouse_pos()):
        word = editor.get_word_at_mouse_pos(imgui.get_mouse_pos())
        if word in ui_node.node.uniform_values:
            value = ui_node.node.uniform_values[word]
            imgui.set_tooltip(f"{word}: {format_auto_value(value)}")
            app.code_hovered_uniform = word

    imgui.pop_font()

    if errors:
        imgui.push_font(app.font_12, app.font_12.legacy_size)
        _draw_error_strip(app, errors, strip_height, current_path)
        imgui.pop_font()
