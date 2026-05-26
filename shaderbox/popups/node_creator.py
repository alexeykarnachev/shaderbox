from imgui_bundle import imgui, imgui_ctx

from shaderbox.app import App
from shaderbox.theme import COLOR, SIZE, SPACE

_LABEL = "New node##popup"


def draw_node_creator(app: App) -> None:
    if not app.is_node_creator_open:
        return

    if not imgui.is_popup_open(_LABEL):
        imgui.open_popup(_LABEL)

    with imgui_ctx.begin_popup_modal(_LABEL) as popup:
        if popup.visible and not _draw_body(app):
            app.is_node_creator_open = False
            imgui.close_current_popup()


def _draw_body(app: App) -> bool:
    imgui.text("Select template:")

    is_template_selected = False

    preview_size = SIZE.THUMB_LG
    available_width = imgui.get_content_region_avail().x
    n_cols = max(1, int(available_width // (preview_size + SPACE.SM)))

    for i, ui_node_template in enumerate(app.ui_node_templates.values()):
        if ui_node_template.id == app.app_state.selected_node_template_id:
            border_color: tuple[float, float, float, float] | None = (
                COLOR.ACCENT_PRIMARY
            )
            is_template_selected = True
        else:
            border_color = None

        if ui_node_template.draw_preview_button(border_color, preview_size).clicked:
            app.app_state.selected_node_template_id = ui_node_template.id
            app.app_state.new_node_name = ui_node_template.ui_state.ui_name

        if (i + 1) % n_cols != 0 and i != len(app.ui_node_templates) - 1:
            imgui.same_line()
        else:
            imgui.spacing()

    imgui.new_line()

    is_keep_opened: bool = True
    if imgui.button("Create", size=(SIZE.BTN_SM_W, 0)) and is_template_selected:
        app.create_node_from_selected_template()
        is_keep_opened = False

    imgui.same_line()
    is_keep_opened &= not imgui.button("Cancel", size=(SIZE.BTN_SM_W, 0))

    return is_keep_opened
