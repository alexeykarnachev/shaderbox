"""No app chord means two things (069 W-E D7).

Every GLOBAL `CommandSpec` chord is absent from both editor keymaps' chord lists, read out
of the vendored docs rather than out of a list retyped here — a retyped list stops tracking
the artifact, and a re-vendor that grows a keymap must turn this red rather than silently
ship a chord with two owners. The audit that decided each cell is
`ai_docs/features/069_tutorial_walk_findings/02_keybindings.md`.
"""

import re
from pathlib import Path

from imgui_bundle import imgui

from shaderbox.commands import (
    COMMAND_SPECS,
    CommandId,
    CommandScope,
    chord_to_str,
)
from shaderbox.hotkeys import _RESERVED_CHORDS

_DOCS = Path("shaderbox/resources/editor")
_VIM_DOC = _DOCS / "vim_coverage.md"
_STD_DOC = _DOCS / "standard_keymap.md"

# Vim writes a chord two ways: `CTRL-X` in the motion sections (vim's own :help spelling)
# and `<C-x>` in the scrolling and word sections. Both are matched; the two notations do
# not overlap, so either one going unparsed drops real chords.
_VIM_KEY = re.compile(r"`(?:CTRL-([A-Za-z])|<C-([A-Za-z]|Left|Right|Home|End)>)`")
_STD_KEY = re.compile(r"`((?:Ctrl|Shift|Alt)(?:\+(?:Ctrl|Shift|Alt))*\+[A-Za-z]+)`")

_MODS = {
    "Ctrl": imgui.Key.mod_ctrl,
    "Shift": imgui.Key.mod_shift,
    "Alt": imgui.Key.mod_alt,
}
_NAMED = {
    "Left": imgui.Key.left_arrow,
    "Right": imgui.Key.right_arrow,
    "Home": imgui.Key.home,
    "End": imgui.Key.end,
    "Space": imgui.Key.space,
    "Tab": imgui.Key.tab,
    "Backspace": imgui.Key.backspace,
    "Delete": imgui.Key.delete,
}

# Floors, not equalities, against the 16 and 13 measured at the vendored VERSION: a
# re-vendor that ADDS a chord must fail the disjointness assertion, not the parse.
_VIM_FLOOR = 14
_STD_FLOOR = 12


def _to_chord(mods: list[str], key: str) -> int:
    imgui_key = _NAMED[key] if key in _NAMED else getattr(imgui.Key, key.lower())
    chord = int(imgui_key)
    for mod in mods:
        chord |= int(_MODS[mod])
    return chord


def _vim_chords(text: str) -> set[int]:
    # The chords live in checklist items; a `[ ]` row counts, because a chord the keymap
    # has declared and not yet built is not free either.
    out: set[int] = set()
    for line in text.splitlines():
        if not line.startswith(("- [x]", "- [ ]")):
            continue
        for match in _VIM_KEY.finditer(line):
            out.add(_to_chord(["Ctrl"], match.group(1) or match.group(2)))
    return out


def _standard_chords(text: str) -> set[int]:
    # The FIRST cell of a table row only: the doc's closing paragraph names Ctrl+X/C/V as
    # chords the editor explicitly does NOT own.
    out: set[int] = set()
    for line in text.splitlines():
        if not line.startswith("| `"):
            continue
        for match in _STD_KEY.finditer(line.split("|")[1]):
            *mods, key = match.group(1).split("+")
            out.add(_to_chord(mods, key))
    return out


def _owned() -> set[int]:
    return _vim_chords(_VIM_DOC.read_text()) | _standard_chords(_STD_DOC.read_text())


def test_the_vim_doc_still_parses() -> None:
    chords = _vim_chords(_VIM_DOC.read_text())
    assert len(chords) >= _VIM_FLOOR, (
        f"vim_coverage.md parsed {len(chords)} chords; format changed?"
    )
    assert _to_chord(["Ctrl"], "d") in chords, "the half-page scroll row stopped parsing"


def test_the_standard_doc_still_parses() -> None:
    chords = _standard_chords(_STD_DOC.read_text())
    assert len(chords) >= _STD_FLOOR, (
        f"standard_keymap.md parsed {len(chords)} chords; format changed?"
    )
    assert _to_chord(["Ctrl"], "a") in chords, "the select-all row stopped parsing"


def test_no_global_app_chord_belongs_to_either_keymap() -> None:
    owned = _owned()
    clashes = [
        f"{spec.id.value} on {chord_to_str(spec.default_chord)}"
        for spec in COMMAND_SPECS
        if spec.scope == CommandScope.GLOBAL and spec.default_chord in owned
    ]
    assert not clashes, (
        "these app chords are owned by a focused editor under at least one keymap; "
        "move them to the Alt or F-key tier (ai_docs/features/069_tutorial_walk_findings/"
        f"02_keybindings.md): {clashes}"
    )


def test_the_only_scoped_chord_a_keymap_owns_is_the_copilot_layout() -> None:
    # So a future spec cannot dodge the assertion above by declaring itself EDITOR-scoped.
    owned = _owned()
    excused = {
        spec.id
        for spec in COMMAND_SPECS
        if spec.scope != CommandScope.GLOBAL and spec.default_chord in owned
    }
    assert excused == {CommandId.CYCLE_COPILOT_LAYOUT}


# Ctrl+W is in neither keymap's list; the host set carries `w` because the HOST implements
# insert-mode word-delete on it (hotkeys._delete_word_back), which the ownership rule does
# not govern. It is the one letter exempt from the subset assertion, named rather than
# quietly widening the check.
_HOST_OWNED = frozenset("w")


def test_every_host_reserved_letter_is_a_vim_chord() -> None:
    # The host's set may be SMALLER than the keymap's list (it approximates only some of
    # it) but never larger: a letter the doc does not name is the host inventing a binding.
    chords = _vim_chords(_VIM_DOC.read_text())
    invented = [
        letter
        for letter in sorted(_RESERVED_CHORDS["vim"] - _HOST_OWNED)
        if _to_chord(["Ctrl"], letter) not in chords
    ]
    assert not invented, (
        f"_RESERVED_CHORDS['vim'] names Ctrl chords vim_coverage.md does not: {invented}"
    )
    assert not _RESERVED_CHORDS["standard"], (
        "standard consumes every chord it owns inside ed_key, so the host approximates none"
    )
