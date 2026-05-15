import glfw
import imgui

from shaderbox.app import App


def process_hotkeys(app: App) -> None:
    glfw.poll_events()
    app.imgui_renderer.process_inputs()
    io = imgui.get_io()

    if io.key_ctrl and imgui.is_key_pressed(ord("O")):
        app.open_project()
    if io.key_ctrl and imgui.is_key_pressed(ord("S")):
        app.save()
    if io.key_ctrl and imgui.is_key_pressed(ord("E")):
        app.edit_current_node_fs_file()
    if io.key_ctrl and imgui.is_key_pressed(ord("D")):
        app.delete_current_node()
    if io.key_ctrl and imgui.is_key_pressed(ord("N")):
        app.open_node_creator()

    if io.key_alt and imgui.is_key_pressed(ord("S")):
        app.open_settings()

    if imgui.is_key_pressed(glfw.KEY_ESCAPE, repeat=False):
        if not app.any_popup_open():
            glfw.set_window_should_close(app.window, True)
        app.is_node_creator_open = False
        app.is_settings_open = False

    if not imgui.is_any_item_active():
        if not app.any_popup_open():
            if imgui.is_key_pressed(glfw.KEY_LEFT, repeat=True):
                app.select_next_current_node(-1)
            if imgui.is_key_pressed(glfw.KEY_RIGHT, repeat=True):
                app.select_next_current_node(+1)
        if app.is_node_creator_open:
            if imgui.is_key_pressed(glfw.KEY_LEFT, repeat=True):
                app.select_next_template(-1)
            if imgui.is_key_pressed(glfw.KEY_RIGHT, repeat=True):
                app.select_next_template(+1)
            if imgui.is_key_pressed(glfw.KEY_ENTER, repeat=False):
                app.create_node_from_selected_template()
                app.is_node_creator_open = False
