from imgui_bundle import imgui, imgui_ctx

from shaderbox.app import App
from shaderbox.popups.emoji_data import EmojiEntry, EmojiGroup, load_emoji_groups
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_primitives import ghost_button, modal_window

_LABEL = "Emoji##picker"
_GRID_COLS = 12
_CELL = 34.0
_POPUP_W = 520.0
_POPUP_H = 480.0


def draw_emoji_picker(app: App) -> None:
    if not app.is_emoji_picker_open:
        return
    with modal_window(_LABEL, (_POPUP_W, _POPUP_H)) as visible:
        if not visible:
            return
        if not _draw_body(app):
            app.is_emoji_picker_open = False
            app.emoji_pick_target = None
            imgui.close_current_popup()


def _draw_body(app: App) -> bool:
    keep_open: bool = True

    _, app.emoji_picker_query = imgui.input_text("Search", app.emoji_picker_query)

    query: str = app.emoji_picker_query.strip().lower()
    groups: list[EmojiGroup] = load_emoji_groups()

    # Reserve room at the bottom for the action row (Close); the scroll child
    # takes the remaining height.
    avail = imgui.get_content_region_avail()
    scroll_h = max(80.0, avail.y - SIZE.BTN_SM_H - float(SPACE.MD) * 2.0)
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

    imgui.dummy((0.0, float(SPACE.MD)))
    if ghost_button("Close"):
        keep_open = False
    return keep_open


def _pick(app: App, char: str) -> None:
    if app.emoji_pick_target is not None:
        app.emoji_pick_target(char)
