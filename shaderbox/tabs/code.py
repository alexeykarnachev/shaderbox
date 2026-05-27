import glfw
from imgui_bundle import imgui
from imgui_bundle import imgui_color_text_edit as text_edit

from shaderbox.app import App
from shaderbox.theme import COLOR, EDITOR_UNFOCUSED_ALPHA, SIZE, SPACE, fade
from shaderbox.ui_primitives import draw_copyable_text
from shaderbox.util import ShaderError, format_auto_value, parse_shader_errors

_MAX_ERROR_ROWS = 3


def _apply_markers(
    editor: text_edit.TextEditor, errors: list[ShaderError], hover_line: int | None
) -> None:
    editor.clear_markers()
    # The marker colours are line fills, so use translucent washes — an opaque fill
    # hides the glyphs underneath.
    err_color = imgui.color_convert_float4_to_u32(fade(COLOR.STATE_ERROR, 0.35))
    for err in errors:
        if err.line >= 0:
            editor.add_marker(err.line, err_color, err_color, err.message, err.message)
    if hover_line is not None:
        accent = imgui.color_convert_float4_to_u32(fade(COLOR.ACCENT_PRIMARY, 0.15))
        editor.add_marker(hover_line, accent, accent, "", "")


def _consume_jump(app: App, editor: text_edit.TextEditor) -> bool:
    if app.editor_jump_request is None:
        return False
    line, index = app.editor_jump_request
    app.editor_jump_request = None
    editor.set_cursor(text_edit.TextEditor.DocPos(line, index))
    editor.select_line(line)
    editor.scroll_to_line(line, text_edit.TextEditor.Scroll.align_middle)
    editor.set_focus()
    return True


def _draw_error_strip(app: App, errors: list[ShaderError], height: float) -> None:
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
                app.editor_jump_request = (err.line, 0)
        extra = n - _MAX_ERROR_ROWS
        if extra > 0:
            imgui.text_colored(COLOR.FG_DIM, f"+{extra} more")
    imgui.end_child()
    imgui.pop_style_color(1)


def draw(app: App) -> None:
    app.code_hovered_uniform = ""
    if not (ui_node := app.ui_nodes.get(app.current_node_id)):
        glfw.set_cursor(app.window, None)
        app.editor_focused = False
        imgui.text_colored(COLOR.FG_DIM, "No node selected")
        return

    full_file_path = app.nodes_dir / ui_node.id / "shader.frag.glsl"
    local_file_path = full_file_path.relative_to(app.project_dir)

    if draw_copyable_text(str(local_file_path), copy_value=str(full_file_path)):
        app.notifications.push("Copied to clipboard!")

    if app.is_current_editor_dirty():
        imgui.same_line()
        imgui.text_colored(COLOR.STATE_WARN, "(unsaved)")
    elif not ui_node.node.shader_error:
        imgui.same_line()
        imgui.text_colored(COLOR.STATE_OK, "compiled")

    imgui.same_line(spacing=float(SPACE.LG))
    if imgui.button("Open dir", size=(SIZE.BTN_SM_W, 0)):
        app.open_current_node_dir()

    imgui.spacing()

    # The editor's render() FPEs while a popup is open (conventions.md ## Known quirks),
    # so it's simply not drawn then — it reappears when the popup closes.
    if app.any_popup_open():
        glfw.set_cursor(app.window, None)
        app.editor_focused = False
        return

    editor = app.get_editor(ui_node.id)
    settings = app.app_state.editor_settings

    errors = (
        parse_shader_errors(ui_node.node.shader_error)
        if ui_node.node.shader_error
        else []
    )
    _apply_markers(editor, errors, app.editor_hover_line)
    app.editor_hover_line = None
    strip_height = 0.0
    if errors:
        n = len(errors)
        # error rows (capped) + the "+N more" line + the "N errors" count header
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

    # set_cursor/select_line/scroll latch a request the upcoming render() executes;
    # a jump also focuses the editor so the highlighted line shows un-dimmed.
    jumped = _consume_jump(app, editor)

    # Dim the whole pane while the editor lacks focus. app.editor_focused is last
    # frame's value (computed below, after render) — a one-frame lag on transitions.
    dim = not app.editor_focused and not jumped
    if dim:
        imgui.push_style_var(imgui.StyleVar_.alpha, EDITOR_UNFOCUSED_ALPHA)

    editor.render("##glsl_editor", size=editor_size)

    if dim:
        imgui.pop_style_var()

    # The editor exposes no is-focused getter, so read imgui's real focus state for
    # this pane (the editor renders its own focusable child window). hotkeys.py reads
    # app.editor_focused to suppress arrow nav while the caret is active.
    app.editor_focused = imgui.is_window_focused(imgui.FocusedFlags_.child_windows)

    # Esc / arrow-nav request defocus (hotkeys.py, app.select_next_current_node).
    # A freshly-rendered editor auto-grabs focus, so the focus must be cleared AFTER
    # render — clearing before is undone by the editor's own first-render focus grab.
    if app.editor_defocus_requested:
        imgui.set_window_focus(None)
        app.editor_defocus_requested = False
        app.editor_focused = False

    # glfw cursor driven directly — editor isn't a hoverable imgui item and imgui cursors are no-op here (conventions.md ## Known quirks).
    # is_window_hovered respects popup-blocking (is_mouse_hovering_rect doesn't), so a menu over the editor keeps the arrow.
    cursor_over_editor = hovering and imgui.is_window_hovered(
        imgui.HoveredFlags_.child_windows
    )
    glfw.set_cursor(app.window, app.ibeam_cursor if cursor_over_editor else None)

    # A passive cursor-following tooltip (not the editor's own popup, which would
    # occlude the code beneath it) — only for words that are live uniforms. The
    # hovered uniform also lights up its panel row (the panel draws after this).
    if cursor_over_editor and editor.is_mouse_pos_over_glyph(imgui.get_mouse_pos()):
        word = editor.get_word_at_mouse_pos(imgui.get_mouse_pos())
        if word in ui_node.node.uniform_values:
            value = ui_node.node.uniform_values[word]
            imgui.set_tooltip(f"{word}: {format_auto_value(value)}")
            app.code_hovered_uniform = word

    imgui.pop_font()

    if errors:
        imgui.push_font(app.font_12, app.font_12.legacy_size)
        _draw_error_strip(app, errors, strip_height)
        imgui.pop_font()
