"""Shipped editor text carries no typographic punctuation — the editor's atlas has no glyph.

The vendored MTSDF atlas (feature 067) rasterizes ASCII, so an em-dash or a curly quote pasted
into a shipped shader or the script stub draws as a white box. That is a visible defect, not a
style preference.

Scoped to the punctuation a writer's copy-paste introduces, NOT to non-ASCII generally: the
glyph table documents the Cyrillic range it renders (`А-Я/Ё`), where the characters ARE the
subject and spelling them in ASCII would make the sentence false. Prose that only reaches imgui
(labels, tooltips) is a different atlas and is not covered here.
"""

from pathlib import Path

from shaderbox.paths import PASS_SHADER_SUFFIX
from shaderbox.scripting.engine import script_stub_for

_RESOURCES = Path(__file__).resolve().parent.parent / "shaderbox" / "resources"

# The characters an editor-bound file keeps tripping over: a writer's quotes and dashes
# arriving through copy-paste. Mapped to the ASCII the repo uses instead, so a failure names
# the fix rather than only the offence.
_ASCII_FOR = {
    "—": "--",
    "–": "-",
    "…": "...",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "•": "*",
}


def _offenders(text: str) -> list[str]:
    return sorted(set(text) & _ASCII_FOR.keys())


def _editor_bound_files() -> list[Path]:
    # What the editor can open from the shipped tree: the shader library and every example
    # document's passes and script. glyphs.glsl is generated (scripts/gen_glyphs.py) and is
    # covered too — a regenerated table must not reintroduce one either.
    files = sorted((_RESOURCES / "shader_lib").rglob("*.glsl"))
    files += sorted((_RESOURCES / "document_examples").rglob(f"*{PASS_SHADER_SUFFIX}"))
    files += sorted((_RESOURCES / "document_examples").rglob("*.py"))
    return files


def test_the_shipped_editor_files_are_ascii() -> None:
    assert _editor_bound_files(), "found no shipped editor files — the globs went stale"
    bad: dict[str, list[str]] = {}
    for path in _editor_bound_files():
        found = _offenders(path.read_text(encoding="utf-8"))
        if found:
            bad[str(path.relative_to(_RESOURCES))] = found
    hint = {c: _ASCII_FOR[c] for v in bad.values() for c in v}
    assert not bad, f"typographic punctuation in shipped editor text: {bad}; use: {hint}"


def test_the_script_stub_is_ascii() -> None:
    # The stub is generated per document, so it never passes through the file check above.
    found = _offenders(script_stub_for([]))
    hint = {c: _ASCII_FOR[c] for c in found}
    assert not found, f"typographic punctuation in the script stub: {found}; use: {hint}"
