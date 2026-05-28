from imgui_bundle import imgui

from shaderbox.app import App
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_primitives import ghost_button, modal_window, primary_button

_LABEL = "New node##popup"
_POPUP_W = 490.0
_POPUP_H = 530.0


def draw_node_creator(app: App) -> None:
    if not app.is_node_creator_open:
        return
    with modal_window(_LABEL, (_POPUP_W, _POPUP_H)) as visible:
        if not visible:
            return
        if not _draw_body(app):
            app.is_node_creator_open = False
            imgui.close_current_popup()


def _draw_body(app: App) -> bool:
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

    imgui.dummy((0.0, float(SPACE.MD)))

    # Enter on a selected template commits (matching the picker's Enter→Insert);
    # the modal's own Esc auto-closes via imgui.
    enter_create = is_template_selected and imgui.is_key_pressed(
        imgui.Key.enter, repeat=False
    )

    keep_open = True
    imgui.begin_disabled(not is_template_selected)
    create_clicked = primary_button("Create")
    imgui.end_disabled()
    if (create_clicked or enter_create) and is_template_selected:
        app.create_node_from_selected_template()
        keep_open = False
    imgui.same_line()
    if ghost_button("Cancel"):
        keep_open = False
    return keep_open
