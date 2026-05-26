from imgui_bundle import imgui, imgui_ctx

from shaderbox.app import App
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_primitives import ghost_button, label_row

_LABEL = "Settings##popup"


def draw_settings(app: App) -> None:
    if not app.is_settings_open:
        return

    if not imgui.is_popup_open(_LABEL):
        imgui.open_popup(_LABEL)

    imgui.set_next_window_size((float(SIZE.SETTINGS_W), 0.0), imgui.Cond_.appearing)
    with imgui_ctx.begin_popup_modal(_LABEL) as popup:
        if popup.visible and not _draw_body(app):
            # Editor settings apply on close only — set_*() while the modal is open
            # FPE-crashes the editor (conventions.md ## Known quirks).
            app.apply_editor_settings()
            app.is_settings_open = False
            imgui.close_current_popup()


def _draw_body(app: App) -> bool:
    ctrl_w = float(SIZE.SETTINGS_CTRL_W)
    label_w = float(SIZE.SETTINGS_LABEL_W)

    imgui.separator_text("General")
    label_row(app.font_12, "Target FPS", ctrl_w, label_w)
    app.app_state.global_target_fps = imgui.drag_int(
        "##global_target_fps", app.app_state.global_target_fps, v_min=30, v_max=240
    )[1]

    imgui.dummy((0.0, SPACE.MD))
    imgui.separator_text("Editor")

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

    imgui.dummy((0.0, SPACE.SM))
    label_row(app.font_12, "Font size", ctrl_w, label_w)
    settings.font_size = imgui.drag_int(
        "##font_size", settings.font_size, v_min=8, v_max=48
    )[1]
    label_row(app.font_12, "Tab size", ctrl_w, label_w)
    settings.tab_size = imgui.drag_int(
        "##tab_size", settings.tab_size, v_min=1, v_max=8
    )[1]
    label_row(app.font_12, "Line spacing", ctrl_w, label_w)
    settings.line_spacing = imgui.drag_float(
        "##line_spacing", settings.line_spacing, v_min=1.0, v_max=2.0, v_speed=0.01
    )[1]

    imgui.dummy((0.0, SPACE.MD))
    imgui.separator_text("Integrations")

    for exporter in app.exporter_registry.all():
        if exporter.is_available:
            if imgui.tree_node_ex(
                exporter.display_name, imgui.TreeNodeFlags_.default_open
            ):
                exporter.draw_config_ui()
                imgui.tree_pop()
        else:
            imgui.text_colored(
                COLOR.FG_DIM, f"{exporter.display_name} — {exporter.unavailable_reason}"
            )

    imgui.dummy((0.0, SPACE.MD))

    is_keep_opened: bool = True
    if ghost_button("Close", width=float(SIZE.BTN_SM_W)):
        is_keep_opened = False

    return is_keep_opened
