"""Help browser (F1) — the shader contract, natively rendered (feature 055).

Section list on the left, prose + an insertable GLSL snippet on the right; the lib-picker shape.
"""

from imgui_bundle import imgui

from shaderbox.app import App, PopupState
from shaderbox.help_content import HelpSection, help_sections
from shaderbox.theme import COLOR, SPACE
from shaderbox.ui_primitives import (
    ghost_button,
    markdown_text,
    modal_window,
    primary_button,
)

_LABEL = "Help##help"
_POPUP_W = 900.0
_POPUP_H = 640.0
_LIST_W = 200.0


def draw_help(app: App) -> None:
    if app.popup_state != PopupState.HELP:
        return
    with modal_window(_LABEL, (_POPUP_W, _POPUP_H)) as visible:
        if not visible:
            return
        if not _draw_body(app):
            app.popup_state = PopupState.CLOSED
            imgui.close_current_popup()


def _current_section(app: App) -> HelpSection:
    sections = help_sections()
    for section in sections:
        if section.key == app.help_section:
            return section
    # An unknown key (a renamed section, a harness that set popup_state directly) falls back
    # rather than indexing into nothing.
    return sections[0]


def _draw_body(app: App) -> bool:
    keep_open = True
    section = _current_section(app)

    list_h = -imgui.get_frame_height_with_spacing()
    if imgui.begin_child("##help_sections", size=(_LIST_W, list_h)):
        for entry in help_sections():
            if imgui.selectable(entry.title, entry.key == section.key)[0]:
                app.help_section = entry.key
    imgui.end_child()

    imgui.same_line()

    if imgui.begin_child("##help_content", size=(0.0, list_h)):
        imgui.push_font(app.font_18, app.font_18.legacy_size)
        imgui.text_colored(COLOR.FG_PRIMARY, section.title)
        imgui.pop_font()
        imgui.dummy((0.0, float(SPACE.SM)))
        markdown_text(section.body, app.font_14_bold)
        if section.snippet:
            imgui.dummy((0.0, float(SPACE.MD)))
            # Fenced so markdown_text takes its code-block path — an unfenced snippet would hit the
            # no-markers fast path and render as prose.
            markdown_text(f"```\n{section.snippet}\n```", app.font_14_bold)
    imgui.end_child()

    if imgui.is_key_pressed(imgui.Key.escape, repeat=False):
        keep_open = False

    # A display-only snippet (the shortcuts table) gets no button at all — the affordance exists
    # only where inserting is meaningful.
    if section.snippet and section.insertable:
        target_ok = _insert_target_ok(app)
        imgui.begin_disabled(not target_ok)
        inserted = primary_button("Insert at caret")
        imgui.end_disabled()
        if not target_ok and imgui.is_item_hovered(
            imgui.HoveredFlags_.allow_when_disabled
        ):
            imgui.set_tooltip(
                "Open a node's shader and click into the editor first (so the caret is positioned)"
            )
        if inserted and app.insert_text_at_caret(section.snippet):
            keep_open = False
        imgui.same_line()
    if ghost_button("Close"):
        keep_open = False
    return keep_open


def _insert_target_ok(app: App) -> bool:
    # A GLSL block belongs in a node's shader — not a lib file and not a script.py, both of which
    # the editor can also have open (the picker's bare-name insert is harmless anywhere; this isn't).
    # The tab's own `kind` is the semantic answer; a filename test would re-derive it.
    tab = app.active_tab
    return app.editor_was_ever_focused and tab is not None and tab.kind == "shader"
