from imgui_bundle import imgui, imgui_ctx

from shaderbox.app import App
from shaderbox.theme import SIZE

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

    is_keep_opened: bool = True
    if imgui.button("Close", size=(SIZE.BTN_SM_W, 0)):
        is_keep_opened = False

    return is_keep_opened
