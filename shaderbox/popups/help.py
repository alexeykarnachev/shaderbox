"""Documentation (F1) — the shader contract, natively rendered (feature 055).

Section list on the left, prose + a GLSL snippet on the right; the lib-picker shape. A reading
surface: nothing here writes into a buffer.
"""

from imgui_bundle import imgui

from shaderbox.app import App, ModalId
from shaderbox.help_content import HelpSection, help_sections
from shaderbox.popups import Modal
from shaderbox.theme import COLOR, SPACE
from shaderbox.ui_primitives import (
    markdown_text,
    modal_content,
    modal_footer,
    standard_button,
)

_LABEL = "Documentation##help"
_POPUP_W = 900.0
_POPUP_H = 640.0
_LIST_W = 200.0


def _current_section(app: App) -> HelpSection:
    sections = help_sections()
    for section in sections:
        if section.key == app.help_section:
            return section
    # An unknown key (a renamed section, a harness that set `app.modal` directly) falls back
    # rather than indexing into nothing.
    return sections[0]


def _draw_body(app: App) -> bool:
    keep_open = True
    section = _current_section(app)

    with modal_content():
        if imgui.begin_child("##help_sections", size=(_LIST_W, 0.0)):
            for entry in help_sections():
                if imgui.selectable(entry.title, entry.key == section.key)[0]:
                    app.help_section = entry.key
        imgui.end_child()

        imgui.same_line()

        if imgui.begin_child("##help_content", size=(0.0, 0.0)):
            imgui.push_font(app.font_18, app.font_18.legacy_size)
            imgui.text_colored(COLOR.FG_PRIMARY, section.title)
            imgui.pop_font()
            imgui.dummy((0.0, float(SPACE.SM)))
            markdown_text(section.body, app.font_14_bold)
            if section.snippet:
                imgui.dummy((0.0, float(SPACE.MD)))
                # Fenced so markdown_text takes its code-block path — an unfenced snippet would
                # hit the no-markers fast path and render as prose.
                markdown_text(f"```\n{section.snippet}\n```", app.font_14_bold)
        imgui.end_child()

    if imgui.is_key_pressed(imgui.Key.escape, repeat=False):
        keep_open = False

    with modal_footer():
        if standard_button("Close"):
            keep_open = False
    return keep_open


MODAL = Modal(
    id=ModalId.HELP,
    label=_LABEL,
    size=lambda app: (_POPUP_W, _POPUP_H),
    body=_draw_body,
)
