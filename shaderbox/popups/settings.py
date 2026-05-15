import webbrowser

import imgui

from shaderbox.app import App

_LABEL = "Settings##popup"


def draw_settings(app: App) -> None:
    if not app.is_settings_open:
        return

    if not imgui.is_popup_open(_LABEL):
        imgui.open_popup(_LABEL)

    if imgui.begin_popup_modal(_LABEL).opened:
        if not _draw_body(app):
            app.is_settings_open = False
            imgui.close_current_popup()
        imgui.end_popup()


def _draw_body(app: App) -> bool:
    app.app_state.global_target_fps = imgui.drag_int(
        "Global target FPS",
        app.app_state.global_target_fps,
        min_value=30,
        max_value=240,
    )[1]

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    width = imgui.get_content_region_available_width()
    app.app_state.text_editor_cmd = imgui.input_text(
        "Text editor cmd", app.app_state.text_editor_cmd
    )[1]

    imgui.same_line()
    imgui.set_cursor_pos_x(width)
    imgui.text_colored("?", *(0.5, 0.5, 0.5))
    if imgui.is_item_hovered():
        imgui.begin_tooltip()
        imgui.text(
            "This command will be executed when you click 'Edit code' (CTRL+E)"
        )
        imgui.text(
            "The '{file_path}' placeholder will be replaced with the actual shader file path."
        )
        imgui.spacing()
        imgui.text("Examples:")
        imgui.text("Linux:")
        imgui.text("  code {file_path}")
        imgui.text("  gnome-terminal -- nvim {file_path}")
        imgui.text("  gnome-terminal -- nano {file_path}")
        imgui.spacing()
        imgui.text("Windows:")
        imgui.text("  code {file_path}")
        imgui.text("  notepad++ {file_path}")
        imgui.text("  cmd /c start nvim {file_path}")
        imgui.new_line()
        imgui.text(
            "Alternatively, you can click the 'Open dir' button in the editor"
        )
        imgui.text(
            "to open the node directory and manually open the shader.frag.glsl file."
        )
        imgui.spacing()
        imgui.end_tooltip()

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    app.app_state.modelbox_url = imgui.input_text(
        "ModelBox url", app.app_state.modelbox_url
    )[1]

    if imgui.button("Install from GitHub##modelbox"):
        url = "https://github.com/alexeykarnachev/modelbox"
        webbrowser.open(url)

    imgui.same_line()
    imgui.set_cursor_pos_x(width)
    imgui.text_colored("?", *(0.5, 0.5, 0.5))
    if imgui.is_item_hovered():
        imgui.begin_tooltip()
        imgui.text(
            "ModelBox is a service which provides AI-powered image processing models."
        )
        imgui.text("You can apply these models to your image and video uniforms.")
        imgui.text(
            "Click 'Install from GitHub' button to check the installation instruction."
        )
        imgui.end_tooltip()

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    imgui.new_line()

    is_keep_opened: bool = True
    if imgui.button("Close", width=80):
        is_keep_opened = False

    return is_keep_opened
