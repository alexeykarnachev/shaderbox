import glfw
from imgui_bundle import imgui

from shaderbox.app import App


def process_hotkeys(app: App) -> None:
    glfw.poll_events()
    app.imgui_renderer.process_inputs()
    io = imgui.get_io()

    if io.key_ctrl and imgui.is_key_pressed(imgui.Key.o):
        app.open_project()
    if io.key_ctrl and imgui.is_key_pressed(imgui.Key.s):
        app.save()
    if io.key_ctrl and imgui.is_key_pressed(imgui.Key.d):
        app.delete_current_node()
    if io.key_ctrl and imgui.is_key_pressed(imgui.Key.n):
        app.open_node_creator()

    if io.key_alt and imgui.is_key_pressed(imgui.Key.s):
        app.open_settings()

    if imgui.is_key_pressed(imgui.Key.escape, repeat=False):
        if not app.any_popup_open():
            glfw.set_window_should_close(app.window, True)
        was_editor_settings_open = app.is_editor_settings_open
        app.is_node_creator_open = False
        app.is_settings_open = False
        app.is_editor_settings_open = False
        if was_editor_settings_open:
            app.apply_editor_settings()

    if not imgui.is_any_item_active():
        if not app.any_popup_open():
            if imgui.is_key_pressed(imgui.Key.left_arrow, repeat=True):
                app.select_next_current_node(-1)
            if imgui.is_key_pressed(imgui.Key.right_arrow, repeat=True):
                app.select_next_current_node(+1)
        if app.is_node_creator_open:
            if imgui.is_key_pressed(imgui.Key.left_arrow, repeat=True):
                app.select_next_template(-1)
            if imgui.is_key_pressed(imgui.Key.right_arrow, repeat=True):
                app.select_next_template(+1)
            if imgui.is_key_pressed(imgui.Key.enter, repeat=False):
                app.create_node_from_selected_template()
                app.is_node_creator_open = False
