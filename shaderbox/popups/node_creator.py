from pathlib import Path

from imgui_bundle import imgui

from shaderbox.app import App, PopupState
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_primitives import (
    caption_text,
    ghost_button,
    labeled_multiline_input,
    modal_window,
    primary_button,
)
from shaderbox.widgets.node_grid import draw_node_preview_button

_LABEL = "New node##popup"
_GRID_COLS = 3
_GRID_MAX_ROWS = 5  # scrollbar appears past this
# Fixed-height so a growing grid never pushes the description slot off the modal.
_DESC_SLOT_H = 132.0
_DESC_EDIT_H = 60.0


def _cell_h() -> float:
    # The preview_cell is a square image (THUMB_LG) plus one footer line for the name.
    return float(SIZE.THUMB_LG) + imgui.get_text_line_height_with_spacing()


def _grid_dims(app: App) -> tuple[float, float]:
    # (grid child height, modal client width) for a fixed 3-col grid capped at 5 rows. Derived from
    # the live style so it stays correct across the DPI-scaled spacing tokens.
    style = imgui.get_style()
    grid_w = _GRID_COLS * float(SIZE.THUMB_LG) + (_GRID_COLS - 1) * style.item_spacing.x
    n = len(app.ui_node_templates)
    rows = max(1, -(-n // _GRID_COLS))  # ceil
    shown = min(rows, _GRID_MAX_ROWS)
    grid_h = (
        shown * _cell_h()
        + (shown - 1) * style.item_spacing.y
        + 2.0 * style.window_padding.y
    )
    # The child reserves the vertical scrollbar's width when it actually scrolls (rows > cap).
    scrollbar = style.scrollbar_size if rows > _GRID_MAX_ROWS else 0.0
    modal_w = grid_w + scrollbar + 2.0 * style.window_padding.x
    return grid_h, modal_w


def draw_node_creator(app: App) -> None:
    if app.popup_state != PopupState.NODE_CREATOR:
        return
    style = imgui.get_style()
    grid_h, modal_w = _grid_dims(app)
    frame_h = imgui.get_frame_height()
    # set_next_window_size sets the WINDOW rect, so the height must include the chrome the content
    # region sits inside: the title bar (== frame_h) + top & bottom window padding. The body itself
    # is grid + desc slot + action row (frame_h) with two inter-block item_spacing.y gaps.
    body_h = grid_h + _DESC_SLOT_H + frame_h + 2.0 * style.item_spacing.y
    modal_h = body_h + frame_h + 2.0 * style.window_padding.y
    flags = (
        imgui.WindowFlags_.no_resize
        | imgui.WindowFlags_.no_scrollbar
        | imgui.WindowFlags_.no_scroll_with_mouse
    )
    with modal_window(_LABEL, (modal_w, modal_h), flags=flags, fixed_size=True) as vis:
        if not vis:
            return
        if not _draw_body(app, grid_h):
            app.popup_state = PopupState.CLOSED
            app.template_desc_input.close()
            imgui.close_current_popup()


def _draw_body(app: App, grid_h: float) -> bool:
    selected = app.app_state.selected_node_template_id
    if imgui.begin_child("##template_grid", size=(0.0, grid_h)):
        selected = _draw_grid(app)
    imgui.end_child()
    is_template_selected = selected in app.ui_node_templates

    # A selection change must close any open editor — else it saves template A's text to template B.
    if (
        app.template_desc_input.is_open
        and app.template_desc_input.target != _template_dir(app, selected)
    ):
        app.template_desc_input.close()

    desc_focused = _draw_description_slot(app, selected)

    # Enter commits — suppressed while the editor is focused, else newline-in-editor creates a node.
    enter_create = (
        is_template_selected
        and not desc_focused
        and imgui.is_key_pressed(imgui.Key.enter, repeat=False)
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


def _draw_grid(app: App) -> str:
    selected_id = app.app_state.selected_node_template_id
    preview_size = SIZE.THUMB_LG
    for i, ui_node_template in enumerate(app.ui_node_templates.values()):
        border = COLOR.SELECT if ui_node_template.id == selected_id else None
        if draw_node_preview_button(
            ui_node_template, border, preview_size, nav_flatten=True
        ).clicked:
            app.app_state.selected_node_template_id = ui_node_template.id
            selected_id = ui_node_template.id
        if (i + 1) % _GRID_COLS != 0 and i != len(app.ui_node_templates) - 1:
            imgui.same_line()
        else:
            imgui.spacing()
    return selected_id


def _template_dir(app: App, template_id: str) -> Path | None:
    if not template_id:
        return None
    return app.node_templates_dir / template_id


def _draw_description_slot(app: App, selected: str) -> bool:
    # Returns True if the editor has keyboard focus, so the outer Enter is suppressed.
    focused = False
    if imgui.begin_child("##template_desc", size=(0.0, _DESC_SLOT_H)):
        template_dir = _template_dir(app, selected)
        if template_dir is None:
            caption_text("Pick a template to start from.")
        elif app.template_desc_input.is_open:
            focused = _draw_description_editor(app, selected)
        else:
            imgui.push_text_wrap_pos(
                0.0
            )  # imgui.text clips long strings; wrap at the edge (/imgui-ui)
            desc = app.template_description(selected)
            imgui.text(desc if desc else "(no description)")
            imgui.pop_text_wrap_pos()
            imgui.dummy((0.0, float(SPACE.SM)))
            if ghost_button("Edit description"):
                app.template_desc_input.open(template_dir, desc)
    imgui.end_child()
    return focused


def _draw_description_editor(app: App, selected: str) -> bool:
    inp = app.template_desc_input
    if inp.needs_focus:
        imgui.set_keyboard_focus_here()
        inp.needs_focus = False
    width = imgui.get_content_region_avail().x
    inp.buf = labeled_multiline_input(
        "Description (saved as you type)", inp.buf, width, _DESC_EDIT_H
    )
    focused = imgui.is_item_focused()
    app.set_template_description(selected, inp.buf)  # on-change persist
    if ghost_button("Done"):
        inp.close()
    return focused
