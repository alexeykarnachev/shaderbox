"""How a KIND of name is coloured: the editor's syntax slots and the host's own lists.

Split from `theme.py` so the palette can be imported by a host that has
neither the editor nor the symbol taxonomy -- `theme` is colours and
`apply_theme`, this is what consumes them. The colours themselves stay
there; nothing here defines one.
"""

from shaderbox.editor import ffi as editor_ffi
from shaderbox.intel.symbols import SymbolKind
from shaderbox.theme import COLOR, ROLE_COLOR, SYNTAX_ROLES, SyntaxRole, fade

# What each kind of name IS. The ONLY hand-written table in the colour system, and the only
# one about MEANING -- it names no colour and no slot, so re-theming never touches it.
#
# `theme.ROLE_COLOR` turns a role into a colour; `_ROLE_CLASS` below turns it into the
# syntax class the editor library draws it in. A kind added here without a role fails the
# enum gate before a frame draws it.
_KIND_ROLE: dict[SymbolKind, SyntaxRole] = {
    SymbolKind.GLSL_KEYWORD: "keyword",
    SymbolKind.GLSL_TYPE: "type",
    SymbolKind.GLSL_BUILTIN: "builtin",
    SymbolKind.GLSL_VARIABLE: "builtin",
    SymbolKind.LIB_FUNCTION: "builtin",
    SymbolKind.ENGINE_UNIFORM: "engine_uniform",
    SymbolKind.PASS_UNIFORM: "ident",
    SymbolKind.PASS_SAMPLER: "pass_sampler",
    SymbolKind.WIRABLE_SAMPLER: "pass_sampler",
    SymbolKind.SCRIPT_UNIFORM: "script_uniform",
    SymbolKind.BUFFER_SYMBOL: "ident",
    SymbolKind.OUTPUT_VARIABLE: "output",
    SymbolKind.GLSL_MEMBER: "member",
    SymbolKind.PY_KEYWORD: "keyword",
    SymbolKind.PY_BUILTIN: "builtin",
    # The engine PROVIDES these names (`ScriptContext`, `Vec3`), which is what `builtin`
    # means -- the same argument that puts `self` there rather than on `keyword`.
    SymbolKind.PY_API: "builtin",
    SymbolKind.PY_MEMBER: "member",
    SymbolKind.PY_LOCAL: "ident",
    SymbolKind.PY_SELF: "builtin",
    SymbolKind.PY_DUNDER: "builtin",
    SymbolKind.PY_CLASS: "declaration_type",
    SymbolKind.PY_DEFINITION: "declaration_function",
    SymbolKind.PY_DECORATOR: "decorator",
    SymbolKind.PY_ANNOTATION: "type",
}


def kind_role(kind: SymbolKind) -> SyntaxRole:
    return _KIND_ROLE[kind]


def kind_color(kind: SymbolKind) -> tuple[float, float, float, float]:
    """The colour a kind draws in, on every surface that shows code.

    Derived from its role, so the editor text, the completion popup, the uniform panel and
    the graph canvas cannot disagree -- they all arrive here. Nothing in this module names
    a colour; `theme.ROLE_COLOR` is the one place a role becomes one.
    """
    return ROLE_COLOR[_KIND_ROLE[kind]]


# Which syntax class the library draws each role in. DERIVED from the colours: roles
# sharing a colour share a class, and a role whose colour the lexer already emits reuses
# the lexer's own class rather than spending a host one.
#
# The lexer owns 1-6 and emits those six colours itself; 7-15 are the host's. Classes are
# GLOBAL -- one class means one thing in every buffer -- which is possible because the
# library carries fifteen, and was not when it carried nine.
_LEXER_CLASS_ROLE: dict[int, SyntaxRole] = {
    1: "keyword",
    2: "string",
    3: "comment",
    4: "number",
    5: "operator",
    6: "builtin",
}


def _build_role_class() -> dict[SyntaxRole, int]:
    by_colour: dict[tuple[float, float, float, float], int] = {
        ROLE_COLOR[role]: cls for cls, role in _LEXER_CLASS_ROLE.items()
    }
    assigned: dict[SyntaxRole, int] = {}
    host_class = 7
    for role in SYNTAX_ROLES:
        colour = ROLE_COLOR[role]
        # `ident` is the editor's plain text: class 0 means "leave it to the lexer", which
        # is both correct and free.
        if colour == ROLE_COLOR["ident"]:
            assigned[role] = 0
            continue
        if colour in by_colour:
            assigned[role] = by_colour[colour]
            continue
        by_colour[colour] = host_class
        assigned[role] = host_class
        host_class += 1
    return assigned


_ROLE_CLASS: dict[SyntaxRole, int] = _build_role_class()

# The library refuses a class past its ceiling rather than clamping it, so a role that
# overflowed would fail at the call -- but it would fail at RUNTIME, on whichever buffer
# first showed that kind. Asserting here moves it to import.
_MAX_CLASS = sum(1 for slot in editor_ffi.Slot if slot.name.startswith("SYNTAX_"))
assert max(_ROLE_CLASS.values()) <= _MAX_CLASS, (
    f"roles need {max(_ROLE_CLASS.values())} syntax classes, the library has {_MAX_CLASS}"
)


def kind_slot(kind: SymbolKind) -> int:
    """The syntax class the library draws this kind in. 0 means "leave it to the lexer".

    Derived from the kind's role, so a kind's colour and the class it is pushed as cannot
    disagree -- which they did before this table existed, when both were hand-written.
    """
    return _ROLE_CLASS[_KIND_ROLE[kind]]


def editor_palette() -> dict["editor_ffi.Slot", tuple[float, float, float, float]]:
    """The palette the editor draws with: chrome from `theme.py`, syntax from the roles.

    ONE palette, no language argument. A syntax class means the same thing in every
    buffer, which the library's fifteen classes make possible -- with nine it did not fit,
    and slots 7/8/9 meant different things in a shader and in a script.

    The syntax entries are GENERATED from `_ROLE_CLASS`, so a role's colour reaches the
    screen without anyone restating it here. Classes the roles do not claim keep the plain
    text colour, which is what the library defaults them to anyway.
    """
    palette = _base_palette()
    # The lexer's own six classes FIRST, each from the role it emits: the lexer pushes
    # class 2 for a string whatever `_ROLE_CLASS` decided, so a class it emits must carry
    # its colour even when a role shares that colour and was assigned elsewhere. Without
    # this, `string` and `builtin` both resolving to class 6 left class 2 unset and every
    # string drew as plain text.
    for cls, role in _LEXER_CLASS_ROLE.items():
        palette[getattr(editor_ffi.Slot, f"SYNTAX_{cls}")] = ROLE_COLOR[role]
    for role, cls in _ROLE_CLASS.items():
        if cls:
            palette[getattr(editor_ffi.Slot, f"SYNTAX_{cls}")] = ROLE_COLOR[role]
    return palette


def _base_palette() -> dict["editor_ffi.Slot", tuple[float, float, float, float]]:
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
        slot.WHITESPACE: fade(COLOR.FG_DIM, 0.5),
        slot.BRACKET_MATCH: fade(COLOR.ACCENT_PRIMARY, 0.25),
        # Drawn over the glyphs, so translucent; a different hue from the bracket box.
        slot.SEARCH_MATCH: fade(COLOR.ACCENT_ACTIVE, 0.30),
    }
