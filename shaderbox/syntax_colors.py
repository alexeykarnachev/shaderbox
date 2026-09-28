"""What a KIND of name IS, in the vocabulary editors already speak.

The one hand-written table here maps each `SymbolKind` to a TREESITTER CAPTURE
name -- `@variable.parameter`, `@constructor`, `@type`. It names no colour and
no syntax class, so re-theming never touches it.

The colours come from a theme file (`shaderbox/resources/themes/*.theme`),
which is a palette plus a link graph in the format vim colorschemes use. That
is what makes "these are not the colours my editor uses" a question with an
answer: both sides name the same captures, so the two files can be read side
by side.
"""

from shaderbox.editor import ffi as editor_ffi
from shaderbox.intel.symbols import SymbolKind
from shaderbox.theme import COLOR, fade
from shaderbox.theme_file import (
    DEFAULT_THEME,
    Theme,
    ThemeError,
    load_theme,
)

# What each kind of name IS, as a treesitter capture. The ONLY hand-written table in the
# colour system, and the only one about MEANING. A kind added without a capture fails the
# enum gate before a frame draws it.
#
# The captures with no treesitter equivalent (`@uniform.*`, `@output`) are facts about a
# SHADER document rather than about a language; the theme file defines them alongside the
# rest and they resolve the same way.
_KIND_CAPTURE: dict[SymbolKind, str] = {
    SymbolKind.GLSL_KEYWORD: "@keyword",
    SymbolKind.GLSL_TYPE: "@type",
    SymbolKind.GLSL_BUILTIN: "@function.builtin",
    SymbolKind.GLSL_VARIABLE: "@variable.builtin",
    SymbolKind.LIB_FUNCTION: "@function",
    SymbolKind.ENGINE_UNIFORM: "@uniform.engine",
    SymbolKind.PASS_UNIFORM: "@variable",
    SymbolKind.PASS_SAMPLER: "@uniform.sampler",
    SymbolKind.WIRABLE_SAMPLER: "@uniform.sampler",
    SymbolKind.SCRIPT_UNIFORM: "@uniform.script",
    SymbolKind.BUFFER_SYMBOL: "@variable",
    SymbolKind.OUTPUT_VARIABLE: "@output",
    SymbolKind.GLSL_MEMBER: "@variable.member",
    SymbolKind.PY_KEYWORD: "@keyword",
    SymbolKind.PY_BUILTIN: "@function.builtin",
    SymbolKind.PY_API: "@type",
    SymbolKind.PY_MEMBER: "@variable.member",
    SymbolKind.PY_LOCAL: "@variable",
    SymbolKind.PY_PARAMETER: "@variable.parameter",
    SymbolKind.PY_SELF: "@variable.builtin",
    SymbolKind.PY_DUNDER: "@function.builtin",
    SymbolKind.PY_CLASS: "@type",
    SymbolKind.PY_CONSTRUCTOR: "@constructor",
    SymbolKind.PY_DEFINITION: "@function",
    SymbolKind.PY_DECORATOR: "@attribute",
    SymbolKind.PY_ANNOTATION: "@type",
}

# The lexer inside the editor library owns classes 1-6 and emits those six itself, so each
# must carry the capture the lexer means by it whatever the derivation below decides.
_LEXER_CLASS_CAPTURE: dict[int, str] = {
    1: "@keyword",
    2: "@string",
    3: "@comment",
    4: "@number",
    5: "@operator",
    6: "@function.builtin",
}

_THEME: Theme = load_theme()


def active_theme() -> Theme:
    return _THEME


def set_theme(name: str) -> None:
    """Switch the app to another theme file, rebuilding everything derived from it.

    The class assignment is derived from the COLOURS, so it cannot be kept across a
    switch: two captures sharing a hue share a class, and which captures those are is the
    new theme's decision. Rebuilding is what keeps `kind_slot` and `editor_palette`
    agreeing, and the budget assert runs again on the new theme's numbers.

    Callers must re-push `editor_palette()` to every live editor; a handle keeps the
    palette it was given (`app.retheme_editors`).
    """
    global _THEME, _CAPTURE_CLASS, _PLAIN_CLASS
    theme = load_theme(name)
    # Build against the new theme BEFORE binding it, so a theme whose budget overflows
    # raises with the app still on the old one rather than half-switched.
    _THEME = theme
    try:
        _CAPTURE_CLASS, _PLAIN_CLASS = _build_capture_class()
        _assert_class_budget()
        # Build the palette too: a chrome key naming no slot must fail at the SWITCH, not
        # on whichever later repaint first touches it.
        editor_palette()
    except Exception:
        _THEME = load_theme(DEFAULT_THEME)
        _CAPTURE_CLASS, _PLAIN_CLASS = _build_capture_class()
        raise


def kind_capture(kind: SymbolKind) -> str:
    return _KIND_CAPTURE[kind]


def kind_color(kind: SymbolKind) -> tuple[float, float, float, float]:
    """The colour a kind draws in, on every surface that shows code.

    Derived from its capture, so the editor text, the completion popup, the uniform panel
    and the graph canvas cannot disagree -- they all arrive here.
    """
    return _THEME.capture(_KIND_CAPTURE[kind])


def _build_capture_class() -> tuple[dict[str, int], int]:
    """Which syntax class the library draws each capture in. DERIVED from the colours.

    Captures sharing a colour share a class, and one whose colour the lexer already emits
    reuses the lexer's class rather than spending a host one. Classes are GLOBAL -- one
    class means one thing in every buffer.

    Returns the table and the class reserved for PLAIN text, which the popup needs by
    number because 0 does not mean the same thing there.
    """
    by_colour: dict[tuple[float, float, float, float], int] = {
        _THEME.capture(capture): cls for cls, capture in _LEXER_CLASS_CAPTURE.items()
    }
    plain = _THEME.capture("@variable")
    assigned: dict[str, int] = {}
    host_class = 7
    for capture in sorted(set(_KIND_CAPTURE.values())):
        colour = _THEME.capture(capture)
        # Class 0 means "leave it to the lexer", which for plain text is both correct and
        # free -- the library's TEXT slot already carries this colour.
        if colour == plain:
            assigned[capture] = 0
            continue
        if colour in by_colour:
            assigned[capture] = by_colour[colour]
            continue
        by_colour[colour] = host_class
        assigned[capture] = host_class
        host_class += 1
    # A real class holding the plain colour, for the surfaces where 0 means something
    # else. It reuses one the lexer already emits if the colour matches, and spends a host
    # class only when it does not.
    plain_class = by_colour.get(plain, host_class)
    return assigned, plain_class


_CAPTURE_CLASS, _PLAIN_CLASS = _build_capture_class()

# The library refuses a class past its ceiling rather than clamping it, so an overflow
# would fail at RUNTIME on whichever buffer first showed that kind. Asserting here moves it
# to import.
_MAX_CLASS = sum(1 for slot in editor_ffi.Slot if slot.name.startswith("SYNTAX_"))


# `_PLAIN_CLASS` is counted too: it is usually the HIGHEST class, being spent last, so an
# assert over the capture table alone leaves the one most likely to overflow unchecked --
# a theme needing every class plus a distinct plain colour passed, then failed inside
# `editor_palette` with an AttributeError naming a slot instead of the budget.
def _assert_class_budget() -> None:
    needed = max(max(_CAPTURE_CLASS.values()), _PLAIN_CLASS)
    assert needed <= _MAX_CLASS, (
        f"captures need {needed} syntax classes, the library has {_MAX_CLASS}"
    )


_assert_class_budget()


def kind_slot(kind: SymbolKind) -> int:
    """The syntax class the library draws this kind in. 0 means "leave it to the lexer".

    Derived from the kind's capture, so a kind's colour and the class it is pushed as
    cannot disagree -- which they did when both were hand-written.

    For the BUFFER, where class 0 falls through to the library's TEXT slot and that slot
    carries `@variable`'s colour. The completion popup resolves 0 to its own plain-text
    colour instead, so it asks `popup_slot` rather than this.
    """
    return _CAPTURE_CLASS[_KIND_CAPTURE[kind]]


def popup_slot(kind: SymbolKind) -> int:
    """The syntax class the COMPLETION POPUP draws this kind in.

    Class 0 means a different thing in each surface: in the buffer it falls through to the
    library's TEXT slot, which carries `@variable`'s colour, while in the popup it means
    the popup's own plain text -- a dimmer colour chosen so unselected rows recede. A kind
    resolving to `@variable` therefore drew `#ebdbb2` in the buffer and `#a89984` in the
    popup, two colours for one symbol.

    Naming the class explicitly is what makes them agree; the popup then draws every kind
    in the colour the theme file asks for.
    """
    return _CAPTURE_CLASS[_KIND_CAPTURE[kind]] or _PLAIN_CLASS


def editor_palette() -> dict["editor_ffi.Slot", tuple[float, float, float, float]]:
    """The palette the editor draws with: chrome from `theme.py`, syntax from the theme file.

    ONE palette, no language argument: a syntax class means the same thing in every buffer.

    The syntax entries are GENERATED, so a capture's colour reaches the screen without
    anyone restating it here.
    """
    palette = _base_palette()
    # The theme's own chrome, before the syntax classes: a theme that names none keeps the
    # app's tokens, and one that does gets to decide what its text sits on.
    _themed_chrome(palette)
    # The lexer's own six FIRST: it pushes class 2 for a string whatever the derivation
    # decided, so a class it emits must carry its colour even when another capture shares
    # that colour and was assigned elsewhere.
    for cls, capture in _LEXER_CLASS_CAPTURE.items():
        palette[getattr(editor_ffi.Slot, f"SYNTAX_{cls}")] = _THEME.capture(capture)
    for capture, cls in _CAPTURE_CLASS.items():
        if cls:
            palette[getattr(editor_ffi.Slot, f"SYNTAX_{cls}")] = _THEME.capture(capture)
    # The plain class the popup pushes by number. Unset, a popup row naming it would draw
    # in whatever that slot defaulted to.
    palette[getattr(editor_ffi.Slot, f"SYNTAX_{_PLAIN_CLASS}")] = _THEME.capture(
        "@variable"
    )
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


# The alpha each translucent slot is drawn at, kept here rather than in the theme file:
# it is a property of how the editor LAYERS these (a wash over glyphs, a box behind them),
# not a colour a theme chooses.
_CHROME_ALPHA: dict[str, float] = {
    "selection": 0.35,
    "whitespace": 0.5,
    "bracket_match": 0.25,
    "search_match": 0.30,
}


def _themed_chrome(
    palette: dict["editor_ffi.Slot", tuple[float, float, float, float]],
) -> None:
    """Overlay the theme file's own `[chrome]`, for a theme the app's tokens cannot serve.

    The dark theme names none and keeps `theme.py`'s chrome, which it was built to match.
    A LIGHT theme has to carry its own: the app's ground is near-black, and light
    foregrounds on it measured 1.54 contrast where 4.5 is the floor -- selectable and
    unreadable. A theme is only fully a theme if it can say what it sits on.
    """
    for key, colour in _THEME.chrome.items():
        slot_name = key.upper()
        if not hasattr(editor_ffi.Slot, slot_name):
            raise ThemeError(
                f"{_THEME.name}: [chrome] names {key!r}, which is not an editor slot"
            )
        alpha = _CHROME_ALPHA.get(key)
        palette[getattr(editor_ffi.Slot, slot_name)] = (
            colour if alpha is None else fade(colour, alpha)
        )
