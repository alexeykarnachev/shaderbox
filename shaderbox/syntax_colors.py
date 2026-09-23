"""How a KIND of name is coloured: the editor's syntax slots and the host's own lists.

Split from `theme.py` so the palette can be imported by a host that has
neither the editor nor the symbol taxonomy -- `theme` is colours and
`apply_theme`, this is what consumes them. The colours themselves stay
there; nothing here defines one.
"""

from shaderbox.editor import ffi as editor_ffi
from shaderbox.intel.symbols import SymbolKind
from shaderbox.theme import COLOR, fade


def kind_color(kind: SymbolKind) -> tuple[float, float, float, float]:
    """The color a kind of name has on a host surface (a source list, a note), the same one
    its syntax slot draws in the popup and the text (078 D2): the enum test pins that a
    slotted kind's color IS its slot's palette entry. Language words and types read as the
    lexer colors them; the engine's uniforms are blue-ish, the script's green-ish, a pass
    sampler aqua; a kind with no slot is a plain identifier."""
    return _KIND_COLOR[kind]


_KIND_COLOR: dict[SymbolKind, tuple[float, float, float, float]] = {
    SymbolKind.GLSL_KEYWORD: COLOR.SYN_KEYWORD,
    SymbolKind.GLSL_TYPE: COLOR.SYN_KEYWORD,
    SymbolKind.GLSL_BUILTIN: COLOR.SYN_BUILTIN,
    SymbolKind.GLSL_VARIABLE: COLOR.SYN_BUILTIN,
    SymbolKind.LIB_FUNCTION: COLOR.SYN_BUILTIN,
    SymbolKind.ENGINE_UNIFORM: COLOR.SYN_UNIFORM,
    SymbolKind.PASS_UNIFORM: COLOR.SYN_IDENT,
    SymbolKind.PASS_SAMPLER: COLOR.SYN_PASS_SAMPLER,
    SymbolKind.WIRABLE_SAMPLER: COLOR.SYN_PASS_SAMPLER,
    SymbolKind.SCRIPT_UNIFORM: COLOR.SYN_SCRIPT_UNIFORM,
    SymbolKind.BUFFER_SYMBOL: COLOR.SYN_IDENT,
    SymbolKind.OUTPUT_VARIABLE: COLOR.SYN_OUTPUT,
    SymbolKind.PY_KEYWORD: COLOR.SYN_KEYWORD,
    SymbolKind.PY_BUILTIN: COLOR.SYN_BUILTIN,
    SymbolKind.PY_API: COLOR.SYN_KEYWORD,
    SymbolKind.PY_MEMBER: COLOR.SYN_IDENT,
    SymbolKind.PY_LOCAL: COLOR.SYN_IDENT,
    SymbolKind.GLSL_MEMBER: COLOR.SYN_IDENT,
}


# The library's syntax slot a host-classified identifier draws in (078 D2, D12): the GLSL lexer
# emits 1 (keywords and types), 4 (numbers), 6 (builtins and functions); a host class fills
# only the identifiers the lexer left plain. Engine uniforms take slot 7, pass samplers 8;
# script uniforms and library functions share the builtin green (6); the fragment output takes
# 9, orange (079 D11). CLASS numbers are contiguous; the SLOTS they draw in are addressed by
# name (`SYNTAX_8` is 25). 0 is no class: the word stays the lexer's.
# Every kind names its slot; 0 is the plain text color. The TEXT feed pushes only the host
# classes (`GlslIndex.classes`), so the language kinds here color popup rows alone.
_KIND_SLOT: dict[SymbolKind, int] = {
    SymbolKind.GLSL_KEYWORD: 1,
    SymbolKind.GLSL_TYPE: 1,
    SymbolKind.GLSL_BUILTIN: 6,
    SymbolKind.GLSL_VARIABLE: 6,
    SymbolKind.LIB_FUNCTION: 6,
    SymbolKind.ENGINE_UNIFORM: 7,
    SymbolKind.PASS_UNIFORM: 0,
    SymbolKind.PASS_SAMPLER: 8,
    SymbolKind.WIRABLE_SAMPLER: 8,
    SymbolKind.SCRIPT_UNIFORM: 6,
    SymbolKind.BUFFER_SYMBOL: 0,
    SymbolKind.OUTPUT_VARIABLE: 9,
    SymbolKind.PY_KEYWORD: 1,
    SymbolKind.PY_BUILTIN: 6,
    SymbolKind.PY_API: 1,
    SymbolKind.PY_MEMBER: 0,
    SymbolKind.PY_LOCAL: 0,
    SymbolKind.GLSL_MEMBER: 0,
}


def kind_slot(kind: SymbolKind) -> int:
    return _KIND_SLOT[kind]


def editor_palette() -> dict["editor_ffi.Slot", tuple[float, float, float, float]]:
    """The gruvbox palette in libeditor theme slots (feature 067). Applied at
    editor-session creation; the syntax slots follow the lexer's token classes
    (1 keyword, 2 string, 3 comment, 4 number, 5 operator, 6 builtin)."""
    slot = editor_ffi.Slot
    return {
        slot.BACKGROUND: COLOR.BG_SURFACE,
        slot.TEXT: COLOR.SYN_IDENT,
        # Reverse-video block caret: the quad is opaque and the glyph under it is
        # emitted in CARET_TEXT, the panel ground, so it reads cut out of the block.
        slot.CARET: COLOR.ACCENT_PRIMARY,
        slot.CARET_INSERT: COLOR.ACCENT_ACTIVE,
        slot.CARET_TEXT: COLOR.BG_SURFACE,
        slot.SELECTION: fade(COLOR.SELECT, 0.35),
        slot.GUTTER_TEXT: COLOR.FG_DIM,
        slot.GUTTER_CURRENT: COLOR.FG_SECONDARY,
        slot.FILLER: COLOR.FG_DIM,
        # One step off the editor ground (BG_SURFACE), so the status row reads as
        # a band rather than as text on an unbroken field.
        slot.STATUS_BG: COLOR.BG_APP,
        slot.STATUS_TEXT: COLOR.FG_SECONDARY,
        slot.STATUS_ACCENT: COLOR.ACCENT_PRIMARY,
        slot.POPUP_PANEL: COLOR.BG_POPUP,
        # The selected row is a TEXT color, not a fill: it must be the BRIGHTER of the two,
        # so the unselected rows recede and the accent picks out the row Enter takes.
        slot.POPUP_TEXT: COLOR.FG_MUTED,
        slot.POPUP_SELECTED: COLOR.ACCENT_PRIMARY,
        slot.SYNTAX_1: COLOR.SYN_KEYWORD,
        slot.SYNTAX_2: COLOR.SYN_STRING,
        slot.SYNTAX_3: COLOR.SYN_COMMENT,
        slot.SYNTAX_4: COLOR.SYN_NUMBER,
        slot.SYNTAX_5: COLOR.SYN_OP,
        slot.SYNTAX_6: COLOR.SYN_BUILTIN,
        slot.SYNTAX_7: COLOR.SYN_UNIFORM,
        slot.SYNTAX_8: COLOR.SYN_PASS_SAMPLER,
        slot.SYNTAX_9: COLOR.SYN_OUTPUT,
        slot.WHITESPACE: fade(COLOR.FG_DIM, 0.5),
        slot.BRACKET_MATCH: fade(COLOR.ACCENT_PRIMARY, 0.25),
        # Drawn over the glyphs, so translucent; a different hue from the bracket box.
        slot.SEARCH_MATCH: fade(COLOR.ACCENT_ACTIVE, 0.30),
    }
