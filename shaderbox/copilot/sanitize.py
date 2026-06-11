"""ASCII transliteration of copilot text for the imgui chat font (feature 020 - 20 D2).

The chat font (AnonymousPro) is loaded through imgui-bundle 1.92's dynamic atlas. It has
em-dash / ellipsis / smart quotes / Latin-1 accents, but NOT arrows (U+2192 etc.) - and an
LLM freely emits all of these, so an unrenderable glyph shows as the atlas fallback box. This
leaf maps the common offenders to ASCII, passes Cyrillic through (the font carries it), and
replaces any still-unrenderable char with '?', so no glyph the font can't draw ever reaches
the screen.

Applied at three boundaries (session.py history-commit + Message-materialize, copilot_chat.py
draw) to CONTENT text only - never tool_calls / arguments / a GLSL payload. Idempotent.

Leaf: no imgui, no App, no copilot-package imports.
"""

# The substitution table - the single source of truth (a recurring missing-glyph is added here).
# Keyed by CODEPOINT (not the literal glyph) so the table source stays ASCII: the glyphs it maps
# are exactly the "ambiguous unicode" a literal key would trip the linter (RUF001) on.
_SUBSTITUTIONS: dict[str, str] = {
    chr(0x2192): "->",  # rightwards arrow
    chr(0x2190): "<-",  # leftwards arrow
    chr(0x2194): "<->",  # left-right arrow
    chr(0x21D2): "=>",  # rightwards double arrow
    chr(0x2014): "--",  # em-dash
    chr(0x2013): "-",  # en-dash
    chr(0x2026): "...",  # horizontal ellipsis
    chr(0x2022): "*",  # bullet
    chr(0x00B7): ".",  # middle dot
    chr(0x2018): "'",  # left single quote
    chr(0x2019): "'",  # right single quote
    chr(0x201C): '"',  # left double quote
    chr(0x201D): '"',  # right double quote
    chr(0x00A0): " ",  # non-breaking space
    chr(0x00D7): "x",  # multiplication sign (LLMs use it for dimensions, e.g. 512x512)
}

# ASCII (incl. tab/newline) renders verbatim; everything else must be transliterated or marked.
_ASCII_MAX = 0x7F
# Cyrillic renders natively (AnonymousPro carries U+0400-U+04FF) — pass it through so
# Russian chat isn't mangled to '?'.
_CYRILLIC_RANGE = (0x0400, 0x04FF)


def _renderable(cp: int) -> bool:
    return cp <= _ASCII_MAX or _CYRILLIC_RANGE[0] <= cp <= _CYRILLIC_RANGE[1]


def sanitize_display(text: str) -> str:
    # Map known glyphs to ASCII; any remaining char the font can't draw becomes '?' so it
    # shows "something was here" instead of a blank box. Idempotent: every output char is
    # itself renderable, so a re-run is a no-op.
    if text.isascii():
        return text
    out: list[str] = []
    for ch in text:
        if _renderable(ord(ch)):
            out.append(ch)
        elif ch in _SUBSTITUTIONS:
            out.append(_SUBSTITUTIONS[ch])
        else:
            out.append("?")
    return "".join(out)
