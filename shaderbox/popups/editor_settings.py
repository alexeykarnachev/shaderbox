from imgui_bundle import imgui, imgui_ctx

from shaderbox.app import App
from shaderbox.theme import SIZE

_LABEL = "Editor options##popup"


def draw_editor_settings(app: App) -> None:
    if not app.is_editor_settings_open:
        return

    if not imgui.is_popup_open(_LABEL):
        imgui.open_popup(_LABEL)

    with imgui_ctx.begin_popup_modal(_LABEL) as popup:
        if popup.visible and not _draw_body(app):
            # Apply ONLY on close. Calling the editor's set_*() methods while this
            # modal is open corrupts its glyph metrics and FPE-crashes the editor's
            # next render() (imgui 1.92 dynamic-atlas interaction). So we mutate the
            # plain settings model live in the popup, but push it to the TextEditor
            # instances only once, here, after the modal is dismissed.
            app.apply_editor_settings()
            app.is_editor_settings_open = False
            imgui.close_current_popup()


def _draw_body(app: App) -> bool:
    settings = app.app_state.editor_settings

    settings.show_whitespace = imgui.checkbox(
        "Show whitespace", settings.show_whitespace
    )[1]
    settings.show_line_numbers = imgui.checkbox(
        "Show line numbers", settings.show_line_numbers
    )[1]
    settings.show_matching_brackets = imgui.checkbox(
        "Highlight matching brackets", settings.show_matching_brackets
    )[1]

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    settings.font_size = imgui.drag_int(
        "Font size", settings.font_size, v_min=8, v_max=48
    )[1]
    settings.tab_size = imgui.drag_int(
        "Tab size", settings.tab_size, v_min=1, v_max=8
    )[1]
    settings.line_spacing = imgui.drag_float(
        "Line spacing", settings.line_spacing, v_min=1.0, v_max=2.0, v_speed=0.01
    )[1]

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    is_keep_opened = True
    if imgui.button("Close", size=(SIZE.BTN_SM_W, 0)):
        is_keep_opened = False

    return is_keep_opened
