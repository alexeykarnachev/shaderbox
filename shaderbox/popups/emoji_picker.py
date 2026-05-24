from imgui_bundle import imgui, imgui_ctx

from shaderbox.app import App
from shaderbox.emoji_data import EmojiEntry, EmojiGroup, load_emoji_groups

_LABEL = "Emoji##picker"
_GRID_COLS = 12
_CELL = 34.0
_POPUP_W = 520.0
_POPUP_H = 480.0


def draw_emoji_picker(app: App) -> None:
    if not app.is_emoji_picker_open:
        return

    if not imgui.is_popup_open(_LABEL):
        imgui.open_popup(_LABEL)

    imgui.set_next_window_size(imgui.ImVec2(_POPUP_W, _POPUP_H), imgui.Cond_.appearing)
    with imgui_ctx.begin_popup_modal(_LABEL) as popup:
        if not popup.visible:
            return
        if not _draw_body(app):
            app.is_emoji_picker_open = False
            imgui.close_current_popup()


def _draw_body(app: App) -> bool:
    keep_open: bool = True

    _, app.emoji_picker_query = imgui.input_text("Search", app.emoji_picker_query)
    if imgui.button("Close"):
        keep_open = False

    query: str = app.emoji_picker_query.strip().lower()
    groups: list[EmojiGroup] = load_emoji_groups()

    with imgui_ctx.begin_child("emoji_scroll", child_flags=imgui.ChildFlags_.borders):
        imgui.push_font(app.font_emoji, app.font_emoji.legacy_size)
        for group in groups:
            matches: list[EmojiEntry] = [
                e for e in group.entries if not query or query in e.name.lower()
            ]
            if not matches:
                continue
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

    return keep_open


def _pick(app: App, char: str) -> None:
    if app.emoji_pick_target is not None:
        app.emoji_pick_target(char)
