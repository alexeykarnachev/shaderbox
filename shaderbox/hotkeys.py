import glfw
from imgui_bundle import imgui

from shaderbox.app import App, JumpRequest
from shaderbox.util import next_error_line


def _jump_to_next_error(app: App) -> None:
    session = app.get_current_session()
    if session is None:
        return
    ui_node = app.ui_nodes[app.current_node_id]
    errors = ui_node.node.compile_unit.errors
    if not errors:
        return
    caret = session.editor.get_current_cursor_position().line
    line = next_error_line(errors, caret)
    if line is not None:
        app.editor_jump_request = JumpRequest(session.source.path, line, 0)


def process_hotkeys(app: App) -> None:
    glfw.poll_events()
    app.imgui_renderer.process_inputs()
    io = imgui.get_io()

    if imgui.is_key_pressed(imgui.Key.f8, repeat=False) and not app.any_popup_open():
        _jump_to_next_error(app)

    if io.key_ctrl and imgui.is_key_pressed(imgui.Key.o):
        app.open_project()
    if io.key_ctrl and imgui.is_key_pressed(imgui.Key.p):
        app.open_lib_picker()
    if io.key_ctrl and imgui.is_key_pressed(imgui.Key.s):
        app.save()
    if io.key_ctrl and imgui.is_key_pressed(imgui.Key.d):
        app.delete_current_node()
    if io.key_ctrl and imgui.is_key_pressed(imgui.Key.n):
        app.open_node_creator()
    if io.key_ctrl and imgui.is_key_pressed(imgui.Key.q):
        glfw.set_window_should_close(app.window, True)

    if io.key_alt and imgui.is_key_pressed(imgui.Key.s):
        app.open_settings()

    if imgui.is_key_pressed(imgui.Key.escape, repeat=False):
        # ESC returns the app to its default state: close any popup, drop editor
        # focus. Never quits the app.
        # Settings holds the editor options — push them when it closes (apply-on-close
        # avoids the modal-open FPE, conventions.md ## Known quirks).
        was_settings_open = app.is_settings_open
        app.is_node_creator_open = False
        app.is_settings_open = False
        app.editor_defocus_requested = True
        if was_settings_open:
            app.apply_editor_settings()

    # Arrow nav is suppressed while the code editor has keyboard focus (set in
    # tabs/code.py after render), so arrows move the editor cursor, not the node.
    if not app.editor_focused:
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
