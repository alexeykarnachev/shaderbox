import glfw
from imgui_bundle import imgui, imgui_ctx

from shaderbox.app import App
from shaderbox.theme import COLOR, SIZE, SPACE


def draw(app: App) -> None:
    if not (ui_node := app.ui_nodes.get(app.current_node_id)):
        glfw.set_cursor(app.window, None)
        imgui.text_colored(COLOR.FG_DIM, "No node selected")
        return

    full_file_path = app.nodes_dir / ui_node.id / "shader.frag.glsl"
    local_file_path = full_file_path.relative_to(app.project_dir)

    imgui.push_style_color(imgui.Col_.text, COLOR.FG_DIM)
    imgui.text(str(local_file_path))
    imgui.pop_style_color()

    if app.is_current_editor_dirty():
        imgui.same_line()
        imgui.text_colored(COLOR.STATE_WARN, "(unsaved)")

    imgui.same_line(spacing=float(SPACE.LG))
    if imgui.button("Open dir", size=(SIZE.BTN_SM_W, 0)):
        app.open_current_node_dir()

    imgui.same_line()
    if imgui.button("Options", size=(SIZE.BTN_SM_W, 0)):
        app.open_editor_settings()

    imgui.spacing()

    editor = app.get_editor(ui_node.id)
    settings = app.app_state.editor_settings
    imgui.push_font(app.font_14, float(settings.font_size))

    # render() FPEs while a popup is open (conventions.md ## Known quirks) — show a read-only snapshot instead
    if app.any_popup_open():
        glfw.set_cursor(app.window, None)
        _draw_code_snapshot(editor.get_text())
    else:
        editor_pos = imgui.get_cursor_screen_pos()
        editor_size = imgui.get_content_region_avail()
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

        editor.render("##glsl_editor", size=editor_size)

        # glfw cursor driven directly — editor isn't a hoverable imgui item and imgui cursors are no-op here (conventions.md ## Known quirks)
        glfw.set_cursor(app.window, app.ibeam_cursor if hovering else None)

    imgui.pop_font()


def _draw_code_snapshot(source: str) -> None:
    with imgui_ctx.begin_child("code_snapshot", size=imgui.get_content_region_avail()):
        imgui.push_style_color(imgui.Col_.text, COLOR.FG_MUTED)
        imgui.text_unformatted(source)
        imgui.pop_style_color()
