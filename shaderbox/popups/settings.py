from imgui_bundle import imgui, imgui_ctx

from shaderbox.app import App
from shaderbox.theme import COLOR, SIZE

_LABEL = "Settings##popup"


def draw_settings(app: App) -> None:
    if not app.is_settings_open:
        return

    if not imgui.is_popup_open(_LABEL):
        imgui.open_popup(_LABEL)

    with imgui_ctx.begin_popup_modal(_LABEL) as popup:
        if popup.visible and not _draw_body(app):
            app.is_settings_open = False
            imgui.close_current_popup()


def _draw_body(app: App) -> bool:
    app.app_state.global_target_fps = imgui.drag_int(
        "Global target FPS",
        app.app_state.global_target_fps,
        v_min=30,
        v_max=240,
    )[1]

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    width = imgui.get_content_region_avail().x
    app.app_state.text_editor_cmd = imgui.input_text(
        "Text editor cmd", app.app_state.text_editor_cmd
    )[1]

    imgui.same_line()
    imgui.set_cursor_pos_x(width)
    imgui.text_colored(COLOR.FG_DIM, "?")
    if imgui.is_item_hovered():
        with imgui_ctx.begin_tooltip():
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

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    imgui.new_line()

    is_keep_opened: bool = True
    if imgui.button("Close", size=(SIZE.BTN_SM_W, 0)):
        is_keep_opened = False

    return is_keep_opened
