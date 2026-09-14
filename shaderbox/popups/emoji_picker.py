from imgui_bundle import imgui, imgui_ctx

from shaderbox.app import App, ModalId
from shaderbox.popups import Modal
from shaderbox.popups.emoji_data import EmojiEntry, EmojiGroup, load_emoji_groups
from shaderbox.theme import COLOR
from shaderbox.ui_primitives import modal_footer, modal_footer_height, standard_button

_LABEL = "Emoji##picker"
_GRID_COLS = 12
_CELL = 34.0
_POPUP_W = 600.0
_POPUP_H = 560.0


def _draw_body(app: App) -> bool:
    keep_open: bool = True

    _, app.emoji_picker_query = imgui.input_text("Search", app.emoji_picker_query)

    query: str = app.emoji_picker_query.strip().lower()
    groups: list[EmojiGroup] = load_emoji_groups()

    # The grid is its own bordered scroll child, so it takes the content height directly
    # rather than nesting inside `modal_content`; the footer's room is the same number the
    # primitive reserves.
    avail = imgui.get_content_region_avail()
    scroll_h = max(80.0, avail.y - modal_footer_height())
    any_match = False
    with imgui_ctx.begin_child(
        "emoji_scroll",
        size=imgui.ImVec2(0.0, scroll_h),
        child_flags=imgui.ChildFlags_.borders,
    ):
        imgui.push_font(app.font_emoji, app.font_emoji.legacy_size)
        for group in groups:
            matches: list[EmojiEntry] = [
                e for e in group.entries if not query or query in e.name.lower()
            ]
            if not matches:
                continue
            any_match = True
            imgui.pop_font()
            imgui.separator_text(group.name)
            imgui.push_font(app.font_emoji, app.font_emoji.legacy_size)
            for col, entry in enumerate(matches):
                if imgui.button(
                    f"{entry.char}##{group.name}_{col}_{entry.name}",
                    size=(_CELL, _CELL),
                ):
                    _pick(app, entry.char)
                    keep_open = False
                if imgui.is_item_hovered():
                    imgui.pop_font()
                    imgui.set_tooltip(entry.name)
                    imgui.push_font(app.font_emoji, app.font_emoji.legacy_size)
                if (col + 1) % _GRID_COLS != 0:
                    imgui.same_line()
            imgui.new_line()
        imgui.pop_font()
        if not any_match:
            imgui.text_colored(COLOR.FG_DIM, "(no matches)")

    with modal_footer():
        if standard_button("Close"):
            keep_open = False
    return keep_open


def _pick(app: App, char: str) -> None:
    if app.emoji_pick_target is not None:
        app.emoji_pick_target(char)


MODAL = Modal(
    id=ModalId.EMOJI_PICKER,
    label=_LABEL,
    size=lambda app: (_POPUP_W, _POPUP_H),
    body=_draw_body,
    on_close=lambda app: app.close_emoji_picker(),
)
