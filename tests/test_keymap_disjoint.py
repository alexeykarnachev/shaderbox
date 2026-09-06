"""No app chord means two things (069 W-E D7).

Every GLOBAL `CommandSpec` chord is absent from both editor keymaps' chord lists, read out
of the vendored docs rather than out of a list retyped here — a retyped list stops tracking
the artifact, and a re-vendor that grows a keymap must turn this red rather than silently
ship a chord with two owners. The audit that decided each cell is
`ai_docs/features/069_tutorial_walk_findings/02_keybindings.md`.
"""

import re
from pathlib import Path

import pytest
from imgui_bundle import imgui

from shaderbox import commands
from shaderbox.commands import (
    COMMAND_SPECS,
    DEFAULT_LEADER,
    LEADER_BINDINGS,
    SPEC_BY_ID,
    CommandId,
    CommandScope,
    chord_to_str,
    leader_command,
)
from shaderbox.editor.ffi import Editor, KeyCode
from shaderbox.hotkeys import _RESERVED_CHORDS, _collect_binding

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
    assert _to_chord(["Ctrl"], "d") in chords, (
        "the half-page scroll row stopped parsing"
    )


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


def test_every_leader_binding_names_a_real_command() -> None:
    # The id crossing the ABI is the table's INDEX, so a reordered or trimmed table
    # silently re-points a registered binding at a different command. Both directions:
    # every row names a command that exists, and `leader_command` round-trips the index.
    for index, (key, command_id) in enumerate(LEADER_BINDINGS):
        assert command_id in SPEC_BY_ID, command_id
        assert leader_command(index) == command_id
        assert len(key) == 1, key


def test_the_index_resolves_row_by_row_and_not_by_luck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`leader_command` maps each index to ITS row, driven over a table long enough to tell
    the mappings apart.

    While `LEADER_BINDINGS` holds one row every index resolves to the same command, so an
    off-by-one inside the function passes the round-trip above -- that assertion reads as if
    it covers the indexing and cannot until a second binding lands. Patching the table drives
    the real function over a domain that discriminates, so the coverage is the function's
    rather than today's data's.
    """
    monkeypatch.setattr(
        commands,
        "LEADER_BINDINGS",
        [
            ("f", CommandId.FORMAT_BUFFER),
            ("s", CommandId.SAVE),
            ("q", CommandId.QUIT),
        ],
    )
    assert [commands.leader_command(i) for i in (-1, 0, 1, 2, 3)] == [
        None,
        CommandId.FORMAT_BUFFER,
        CommandId.SAVE,
        CommandId.QUIT,
        None,
    ]


def test_an_id_outside_the_table_names_nothing() -> None:
    # The drain hands us whatever the library reports; an id we never registered must
    # resolve to None rather than index into the table's tail.
    assert leader_command(len(LEADER_BINDINGS)) is None
    assert leader_command(-1) is None


def test_the_leader_sequence_reaches_the_command_and_the_chord_still_does() -> None:
    # The maintainer asked for `<leader>f` ALONGSIDE Ctrl+Shift+I, not instead of it.
    # Driven through the real editor: arm the leader, send the key, drain.
    editor = Editor("abc def\n")
    editor.set_leader(DEFAULT_LEADER)
    for index, (key, _) in enumerate(LEADER_BINDINGS):
        editor.bind(key, index, leader=True)
    editor.key(KeyCode.CHAR, 0, DEFAULT_LEADER)
    editor.key(KeyCode.CHAR, 0, LEADER_BINDINGS[0][0])
    fired = editor.take_binding()
    # The first row's id is 0, which is a VALID id and also falsy: a drain written as
    # `take_binding() or default` resolves it to the default and the binding silently does
    # nothing. `_serve_leader_binding` tests `is None` for this reason; so does this.
    assert fired == 0
    assert leader_command(fired) == LEADER_BINDINGS[0][1]
    # The registry chord is untouched by any of this.
    assert SPEC_BY_ID[CommandId.FORMAT_BUFFER].default_chord != 0


def test_the_leader_never_fires_where_a_command_is_in_flight() -> None:
    # The guards the library promises (editor da8a850), pinned HERE because this host is
    # what breaks if they regress: an operand and an operator each consume the space
    # before the binding table is consulted, and a bare space moves nothing.
    # `r<Space>` is the one that fails when the guard is placed above the awaiting
    # branches rather than below them -- `d<Space>` keeps passing there.
    editor = Editor("abc def\n")
    for keys, expected in (
        ("", "abc def"),
        ("r", " bc def"),
        ("d", "bc def"),
    ):
        editor.set_text("abc def\n")
        editor.set_leader(DEFAULT_LEADER)
        editor.bind(LEADER_BINDINGS[0][0], 0, leader=True)
        for ch in keys:
            editor.key(KeyCode.CHAR, 0, ch)
        editor.key(KeyCode.CHAR, 0, " ")
        assert editor.get_text().split("\n")[0] == expected, keys
        assert editor.take_binding() is None, keys


def test_two_sequences_in_one_feed_both_survive() -> None:
    # The ABI slot holds ONE and a completing key OVERWRITES it during the feed, so where
    # the drain sits decides whether a sequence is lost. Measured (editor 09e3e59): an armed
    # leader survives a drain, so a leader pressed at the end of one frame completes at the
    # start of the next -- three keys across two frames, ordinary typing. Draining once
    # after the feed, even in a loop, yields only the SECOND id; draining beside each key
    # yields both. Broken (move `_collect_binding` out of the loop), this returns [7].
    editor = Editor("abc def\n")
    editor.set_leader(DEFAULT_LEADER)
    editor.bind("f", 0, leader=True)
    editor.bind("q", 7, leader=True)
    fired: list[int] = []
    for ch in (DEFAULT_LEADER, "f", DEFAULT_LEADER, "q"):
        _collect_binding(editor, fired)
        editor.key(KeyCode.CHAR, 0, ch)
    _collect_binding(editor, fired)
    assert fired == [0, 7]


def test_no_binding_claims_a_chord_the_host_serves_on_false() -> None:
    """A registered chord returns TRUE from `ed_key`, so it never reaches the host's
    unconsumed-key fallbacks.

    `_drain_editor_input` treats an unconsumed key as its own: `_handle_reserved_chord`
    approximates vim's insert-mode chords (Ctrl+U, Ctrl+W, ...) and `_is_lookup_key` serves
    `K`. Since editor e6ddfbc a host binding SHADOWS the built-in and reports consumed --
    the point of it -- so a binding registered with Ctrl over one of those letters would
    take the key away from the fallback that implements it, silently.

    A LEADER row is not that shape: it is reached through the leader prefix, not as a bare
    chord, so `f` here does not collide with the host's Ctrl+F. The check is on how a row
    is REGISTERED, which is what `_apply_editor_settings_to` passes to `ed_bind`.
    """
    registered = [(key, 0, True) for key, _ in LEADER_BINDINGS]
    for key, mods, leader in registered:
        if leader or mods == 0:
            continue
        assert key not in _RESERVED_CHORDS["vim"], (
            f"Ctrl+{key} is registered as a bare chord, so `ed_key` now consumes it and "
            "the host approximation in `_handle_reserved_chord` stops running"
        )
        assert key != "K", "K is served by the host's lookup path on an unconsumed key"


def test_the_registration_shape_is_the_one_the_app_actually_uses() -> None:
    # The test above reasons about (key, mods, leader) triples; this pins that the app
    # registers exactly that shape, so the two cannot drift apart.
    source = Path("shaderbox/app.py").read_text()
    assert "editor.bind(key, index, leader=True)" in source
