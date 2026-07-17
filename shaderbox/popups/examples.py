from imgui_bundle import imgui

from shaderbox.app import App, PopupState
from shaderbox.theme import COLOR, SIZE
from shaderbox.ui_primitives import (
    caption_text,
    ghost_button,
    modal_window,
    primary_button,
)
from shaderbox.widgets.node_grid import draw_node_preview_button

_LABEL = "Examples##popup"
_GRID_COLS = 3
_GRID_MAX_ROWS = 5  # scrollbar appears past this
# Fixed-height so a growing grid never pushes the description slot off the modal.
_DESC_SLOT_H = 132.0


def _cell_h() -> float:
    # The preview_cell is a square image (THUMB_LG) plus one footer line for the name.
    return float(SIZE.THUMB_LG) + imgui.get_text_line_height_with_spacing()


def _grid_dims(app: App) -> tuple[float, float]:
    # (grid child height, modal client width) for a fixed 3-col grid capped at 5 rows. Derived from
    # the live style so it stays correct across the DPI-scaled spacing tokens.
    style = imgui.get_style()
    grid_w = _GRID_COLS * float(SIZE.THUMB_LG) + (_GRID_COLS - 1) * style.item_spacing.x
    n = len(app.ui_node_examples)
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


def draw_examples(app: App) -> None:
    if app.popup_state != PopupState.EXAMPLES:
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
            imgui.close_current_popup()


def _draw_body(app: App, grid_h: float) -> bool:
    selected = app.app_state.selected_example_id
    if imgui.begin_child("##example_grid", size=(0.0, grid_h)):
        selected = _draw_grid(app)
    imgui.end_child()
    is_selected = selected in app.ui_node_examples

    _draw_description_slot(app, selected)

    enter_open = is_selected and imgui.is_key_pressed(imgui.Key.enter, repeat=False)

    keep_open = True
    imgui.begin_disabled(not is_selected)
    open_clicked = primary_button("Open a copy")
    imgui.end_disabled()
    if (open_clicked or enter_open) and is_selected:
        app.create_node_from_example(selected)
        keep_open = False
    imgui.same_line()
    if ghost_button("Close"):
        keep_open = False
    return keep_open


def _draw_grid(app: App) -> str:
    selected_id = app.app_state.selected_example_id
    preview_size = SIZE.THUMB_LG
    for i, ui_node_example in enumerate(app.ui_node_examples.values()):
        border = COLOR.SELECT if ui_node_example.id == selected_id else None
        if draw_node_preview_button(
            ui_node_example, border, preview_size, nav_flatten=True
        ).clicked:
            app.app_state.selected_example_id = ui_node_example.id
            selected_id = ui_node_example.id
        if (i + 1) % _GRID_COLS != 0 and i != len(app.ui_node_examples) - 1:
            imgui.same_line()
        else:
            imgui.spacing()
    return selected_id


def _draw_description_slot(app: App, selected: str) -> None:
    if imgui.begin_child("##example_desc", size=(0.0, _DESC_SLOT_H)):
        if selected not in app.ui_node_examples:
            caption_text("Pick an example to read about it; open a copy to dig in.")
        else:
            imgui.push_text_wrap_pos(
                0.0
            )  # imgui.text clips long strings; wrap at the edge (/imgui-ui)
            desc = app.example_description(selected)
            imgui.text(desc if desc else "(no description)")
            imgui.pop_text_wrap_pos()
    imgui.end_child()
