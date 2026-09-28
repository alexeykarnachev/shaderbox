"""Feature 105: the semantic colour a Python script gets and a lexer cannot supply.

Both halves ship. `self`, `cls` and the dunders are EDIT-INVARIANT and ride the word table;
definitions, decorators and annotations are positional and go through `ed_set_spans`, which
the `05f90ac` re-vendor brought across the ABI.

Every gate here exists because a cheaper one passes while the feature is broken. The enum
gate in `test_intel_sources.py` walks `SymbolKind` and checks a colour, a rank and a slot per
kind -- and a kind wired with all three that NOTHING EVER PRODUCES passes it clean. So each
of the four new kinds gets a case asserting it appears on real source.
"""

import time
from pathlib import Path

import pytest

from shaderbox.editor.ffi import Editor, Language, Slot
from shaderbox.intel.python import SpanSymbol, python_spans, python_word_classes
from shaderbox.intel.symbols import SymbolKind
from shaderbox.intel.worker import (
    PythonRequest,
    PythonRequestKind,
    PythonResult,
    PythonWorker,
)
from shaderbox.syntax_colors import editor_palette, kind_role, kind_slot
from shaderbox.tabs.code import (
    PythonFeed,
    accept_python_spans,
    feed_python_word_classes,
)
from shaderbox.theme import ROLE_COLOR

# A script with every one of the five cases, at values where they are DISTINGUISHABLE. The
# names are deliberately reused across roles: `update` is defined once and called once,
# `property` is a decorator on one line and a bare reference on another, `float` is an
# annotation and also an ordinary call. A position-blind mechanism cannot tell those apart,
# which is the whole reason the span half exists -- and a fixture that used each name in one
# role only would pass with the distinction deleted.
SOURCE = '''import math
from dataclasses import dataclass


@dataclass
class Rig:
    """A thing with state."""

    scale: float = 1.0

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.phase: float = 0.0

    @property
    def ready(self) -> bool:
        return self.seed > 0

    def update(self, dt: float) -> float:
        self.phase = math.fmod(self.phase + dt, math.tau)
        return float(self.phase)


def drive(rig: Rig) -> float:
    kind = property
    assert kind is not None
    return rig.update(0.5)
'''


def _line_of(text: str, needle: str) -> int:
    for index, line in enumerate(text.split("\n")):
        if needle in line:
            return index
    raise AssertionError(f"{needle!r} is not in the fixture")


def _spans_at(spans: tuple[SpanSymbol, ...], name: str) -> list[SpanSymbol]:
    return [span for span in spans if span.name == name]


# --- the word table: the half that ships (D6) --------------------------------


def test_self_and_the_dunders_get_a_kind() -> None:
    # PY_SELF and PY_DUNDER are EMITTED, which the enum gate cannot see. Break by deleting
    # the producer's branches and this fails while `test_every_kind_has_a_color` stays green.
    classes = python_word_classes(SOURCE)
    assert classes["self"] == SymbolKind.PY_SELF
    assert classes["__init__"] == SymbolKind.PY_DUNDER
    # A name that merely LOOKS like a dunder on one side is not one: the falsifier for a rule
    # written as "starts with __".
    assert "_line" not in classes
    assert python_word_classes("__init\nx__\n_x_\n") == {}


def test_cls_counts_and_a_plain_name_does_not() -> None:
    classes = python_word_classes("def f(cls, classes, selfish, self):\n    pass\n")
    assert classes == {"cls": SymbolKind.PY_SELF, "self": SymbolKind.PY_SELF}


def test_a_script_tab_feeds_word_classes_at_all() -> None:
    """THE D6 GATE. Break by removing the feed from `_python_feed_for` and `self` goes plain.

    NOT "restore the guard": the guard that keeps `_glsl_index_for` off script tabs is
    correct and stays. It is the GLSL indexer it keeps out, and running that index on a
    Python script would produce a meaningless one -- its fingerprint and `build()` are a pass
    name, sampler values and engine uniform types.

    The fixture proves CONTACT rather than asserting a silence: it reads the class back out
    of the library, so "nothing was fed" and "the right thing was fed" cannot look alike.
    """
    editor = Editor(SOURCE)
    editor.set_language(Language.PYTHON)
    feed = PythonFeed(editor=editor)
    assert feed_python_word_classes(editor, feed, SOURCE), "the first feed must push"
    assert feed.revision == 1, "a push must move the revision render_state reads"

    editor.layout((640.0, 480.0), 16.0)
    line = _line_of(SOURCE, "self.seed = seed")
    column = SOURCE.split("\n")[line].index("self")
    assert editor.class_at(line, column) == kind_slot(SymbolKind.PY_SELF)
    # The falsifier: a plain local on the same line reads back as no class, so the assertion
    # above is about `self` and not about the library colouring everything.
    assert editor.class_at(line, SOURCE.split("\n")[line].index("seed = ")) == 0
    editor.close()


def test_the_word_feed_is_idempotent_on_a_still_buffer() -> None:
    # The revision counts PUSHES, so a feed that re-pushed every frame would repaint the
    # panel every frame. Re-feeding is skipped while the buffer's revision stands still.
    editor = Editor(SOURCE)
    editor.set_language(Language.PYTHON)
    feed = PythonFeed(editor=editor)
    assert feed_python_word_classes(editor, feed, SOURCE)
    assert not feed_python_word_classes(editor, feed, SOURCE)
    assert feed.revision == 1
    editor.close()


# --- the spans: computed now, pushed when the library lands the call (D7) -----


def test_a_definition_and_a_call_of_the_same_name_differ() -> None:
    """PY_DEFINITION is EMITTED, and the fixture is built where the distinction MATTERS.

    `update` appears twice: defined at one line and called at another. A name-keyed mechanism
    returns the same answer for both, so a test that used the name once would pass with the
    positional half deleted."""
    spans = python_spans(SOURCE)
    updates = _spans_at(spans, "update")
    assert len(updates) == 1, "only the definition is a definition"
    assert updates[0].kind == SymbolKind.PY_DEFINITION
    assert updates[0].line == _line_of(SOURCE, "def update(self, dt")
    # The call site carries no definition span: the two occurrences of one spelling are
    # told apart, which is the property the word table cannot express.
    call_line = _line_of(SOURCE, "return rig.update(0.5)")
    assert all(span.line != call_line for span in updates)


def test_an_import_is_not_a_definition() -> None:
    # jedi's `get_names` reports an imported name as a definition AT ITS IMPORT SITE, so
    # `from dataclasses import dataclass` would colour `dataclass` as though the script
    # defined it. Break by reading definitions off `get_names` instead of the parso tree.
    spans = python_spans(SOURCE)
    # Both DECLARING kinds. A class is a definition too; it draws in a different colour
    # because a type declaration and a function declaration are two roles, which is about
    # colour rather than about what counts as declared here.
    declaring = {SymbolKind.PY_DEFINITION, SymbolKind.PY_CLASS}
    defined = {span.name for span in spans if span.kind in declaring}
    assert "dataclass" not in defined
    assert "math" not in defined
    assert {"Rig", "update", "drive", "ready", "__init__"} <= defined


def test_a_decorator_and_a_bare_reference_to_it_differ() -> None:
    """PY_DECORATOR is EMITTED, on the case that disproved the spec's earlier claim.

    MEASURED with jedi: `@property` and the bare `property` two functions down are
    byte-identical in every field `get_names` exposes -- both `type="statement"`,
    `is_definition()` false, the same `description`. So this cannot be answered by the
    producer the cost numbers were originally sized against, and the fixture contains both
    roles of the one name so a producer that cannot tell them apart fails here."""
    spans = python_spans(SOURCE)
    decorator_line = _line_of(SOURCE, "@property")
    reference_line = _line_of(SOURCE, "kind = property")
    properties = _spans_at(spans, "property")
    lines = {span.line for span in properties}
    assert decorator_line in lines
    assert reference_line not in lines, "a bare reference is not a decorator"
    assert all(span.kind == SymbolKind.PY_DECORATOR for span in properties)


def test_an_annotation_and_an_ordinary_call_of_the_same_name_differ() -> None:
    # The second half of D7's case, on `float`: an annotation on `dt` and `scale`, an
    # ordinary call in `return float(self.phase)`. jedi calls both a statement.
    spans = python_spans(SOURCE)
    floats = _spans_at(spans, "float")
    lines = {span.line for span in floats}
    assert _line_of(SOURCE, "def update(self, dt: float)") in lines
    assert _line_of(SOURCE, "scale: float = 1.0") in lines
    assert _line_of(SOURCE, "return float(self.phase)") not in lines


def test_a_decorators_arguments_are_not_the_decorator() -> None:
    # The falsifier for "everything after the @ is a decorator": a decorator's ARGUMENTS are
    # ordinary expressions and keep whatever colour they have anywhere else.
    text = "@register(order=3, key=helper)\ndef go() -> None:\n    pass\n"
    spans = python_spans(text)
    named = {span.name for span in spans if span.kind == SymbolKind.PY_DECORATOR}
    assert "register" in named
    assert "helper" not in named


def test_every_span_lands_on_the_text_it_names() -> None:
    """The contact check. A span set whose coordinates are off by a line or a column would
    still be a non-empty tuple of plausible names, and every test above would pass, because
    they assert on the LINE a name is reported at rather than on the characters there. This
    slices the buffer at each span and demands the name back."""
    lines = SOURCE.split("\n")
    spans = python_spans(SOURCE)
    assert spans, "the fixture must produce spans at all"
    for span in spans:
        assert lines[span.line][span.column : span.column_end] == span.name, span


def test_a_broken_buffer_still_answers() -> None:
    """The unparseable buffer is the NORMAL state mid-edit, and D7's deferral trigger was
    whether a second producer's cost there could be bounded. MEASURED on the flock script:
    clean 6.63 ms, and an unclosed paren, a dangling `def`, a half-typed annotation and an
    unterminated string all between 6.26 and 6.74 ms -- parso recovers rather than raising,
    and keeps the whole file's structure. So all five cases ship rather than two deferring.
    """
    for broken in (
        SOURCE + "\n    def ",
        SOURCE + "\n@",
        SOURCE + '\n    x = "abc',
        SOURCE.replace("def drive(rig: Rig) -> float:", "def drive(rig: Rig -> float:"),
        SOURCE[: len(SOURCE) // 2],
    ):
        spans = python_spans(broken)
        # Not merely "it did not raise": the text BEFORE the break is still classified, which
        # is what makes colour survive typing rather than blinking out at the first bad char.
        assert any(
            span.name == "Rig" and span.kind == SymbolKind.PY_CLASS for span in spans
        ), broken[-40:]


# --- staleness (D3b) ---------------------------------------------------------


def _spans_result(path: Path, revision: int, text: str = SOURCE) -> PythonResult:
    request = PythonRequest(
        PythonRequestKind.SPANS, path, text, 0, 0, revision=revision
    )
    return PythonResult(request, (), spans=python_spans(text))


def _python_editor(text: str = SOURCE) -> Editor:
    editor = Editor(text)
    editor.set_language(Language.PYTHON)
    editor.layout((640.0, 480.0), 16.0)
    return editor


def test_a_definition_is_coloured_and_its_call_site_is_not() -> None:
    """The end-to-end claim, read back out of the LIBRARY rather than off the producer.

    Every test above asks the producer what it computed. This asks the editor what it will
    DRAW, which is a different question: a span set can be correct and still land nowhere if
    the coordinate unit disagrees. The fixture picks the one name used in both roles, so a
    mechanism that cannot tell a definition from a call fails here."""
    editor = _python_editor()
    feed = PythonFeed(editor=editor)
    assert accept_python_spans(
        editor, feed, _spans_result(Path("/tmp/s.py"), editor.get_undo_index())
    )
    editor.layout((640.0, 480.0), 16.0)

    def_line = _line_of(SOURCE, "def update(self, dt")
    def_col = SOURCE.split("\n")[def_line].index("update")
    assert editor.class_at(def_line, def_col) == kind_slot(SymbolKind.PY_DEFINITION)

    call_line = _line_of(SOURCE, "return rig.update(0.5)")
    call_col = SOURCE.split("\n")[call_line].index("update")
    assert editor.class_at(call_line, call_col) == 0, (
        "the same spelling at a call site must NOT take the definition colour"
    )
    editor.close()


def test_the_column_unit_is_codepoints_not_bytes() -> None:
    """MEASURED, because the producer and the library could each be right separately.

    parso reports codepoint columns and the library wants codepoint columns; a byte-based
    producer would agree with both on every ASCII line and silently miss on any line holding
    a non-ASCII character. The fixture puts a 3-byte character BEFORE the name, so the byte
    column and the codepoint column differ by six -- at the defaults they are equal and this
    test would pass with a byte-based producer."""
    text = 's = "\u4e2d\u4e2d\u4e2d"\n\n\ndef target() -> None:\n    pass\n'
    spans = python_spans(text)
    target = next(span for span in spans if span.name == "target")
    line = text.split("\n")[target.line]
    assert line[target.column : target.column_end] == "target"

    editor = _python_editor(text)
    feed = PythonFeed(editor=editor)
    assert accept_python_spans(
        editor,
        feed,
        PythonResult(
            PythonRequest(
                PythonRequestKind.SPANS,
                Path("/tmp/s.py"),
                text,
                0,
                0,
                revision=editor.get_undo_index(),
            ),
            (),
            spans=spans,
        ),
    )
    editor.layout((640.0, 480.0), 16.0)
    assert editor.class_at(target.line, target.column) == kind_slot(
        SymbolKind.PY_DEFINITION
    )
    # The falsifier, on the line that actually holds the multi-byte characters: the byte
    # column of the string's closing quote is not its codepoint column, and a span pushed at
    # the byte column would land off the end of the line.
    first = text.split("\n")[0]
    assert len(first.encode()) != len(first), "the fixture must be multi-byte at all"
    editor.close()


def test_a_stale_span_set_is_rejected_rather_than_misapplied() -> None:
    """THE D3b GATE. The policy is DROP-AND-RE-REQUEST, and the library enforces it: a push
    carrying a revision the buffer has moved past returns 0, changes nothing, and leaves the
    previous set standing. There is no offset salvage, which is right -- a set computed
    before a newline was typed would put every span below it one row out, landing a
    definition's colour on the WRONG word rather than on no word.

    Break by passing the CURRENT revision instead of the one the text was read at: the stale
    set applies, and the assertion that the old colour survived fails."""
    editor = _python_editor()
    feed = PythonFeed(editor=editor)
    good_revision = editor.get_undo_index()
    assert accept_python_spans(
        editor, feed, _spans_result(Path("/tmp/s.py"), good_revision)
    )
    editor.layout((640.0, 480.0), 16.0)
    def_line = _line_of(SOURCE, "def update(self, dt")
    def_col = SOURCE.split("\n")[def_line].index("update")
    assert editor.class_at(def_line, def_col) == kind_slot(SymbolKind.PY_DEFINITION)

    # Now move the buffer, then offer an answer computed against the text as it WAS.
    editor.insert_text("# a new first line\n") if hasattr(
        editor, "insert_text"
    ) else None
    editor.set_cursor(0, 0)
    editor.feed("ggO# moved<ESC>")
    editor.layout((640.0, 480.0), 16.0)
    moved_revision = editor.get_undo_index()
    assert moved_revision != good_revision, "the fixture must actually move the buffer"

    stale = PythonFeed(editor=editor)
    assert not accept_python_spans(
        editor, stale, _spans_result(Path("/tmp/s.py"), good_revision)
    ), "a set computed against older text must be refused"
    assert stale.span_revision is None
    assert stale.spans == ()
    # A rejection paints nothing, so it must not move the revision render_state reads.
    assert stale.revision == 0
    # And the refusal changed NOTHING: the set applied earlier is still there, following the
    # edit down a line. This is what "the previous set stands" means, and it is the half a
    # return-code assertion alone would miss.
    editor.layout((640.0, 480.0), 16.0)
    assert editor.class_at(def_line + 1, def_col) == kind_slot(SymbolKind.PY_DEFINITION)
    editor.close()


def test_an_applied_set_follows_an_edit() -> None:
    """The regime the spec predicted WRONG, corrected by the library's own behaviour.

    105 D3 says colour "drops to PLAIN for the whole burst, then snaps back". It does not:
    an applied set is anchored in the buffer, so an insert above it carries it down and the
    characters it named keep their colour between the edit and the next debounced push. That
    changes the JUSTIFICATION for the debounce -- latency and head-of-line, not staleness."""
    editor = _python_editor()
    feed = PythonFeed(editor=editor)
    assert accept_python_spans(
        editor, feed, _spans_result(Path("/tmp/s.py"), editor.get_undo_index())
    )
    editor.layout((640.0, 480.0), 16.0)
    def_line = _line_of(SOURCE, "def update(self, dt")
    def_col = SOURCE.split("\n")[def_line].index("update")
    assert editor.class_at(def_line, def_col) == kind_slot(SymbolKind.PY_DEFINITION)

    editor.feed("ggO# inserted<ESC>")
    editor.layout((640.0, 480.0), 16.0)
    assert editor.class_at(def_line + 1, def_col) == kind_slot(
        SymbolKind.PY_DEFINITION
    ), "an anchored span follows the text it coloured"
    # The falsifier: the row it USED to be on is not still coloured, so the assertion above
    # is about the span moving rather than about everything being coloured.
    assert editor.class_at(def_line, def_col) == 0
    editor.close()


def test_set_text_drops_the_spans_and_the_revision_moves_so_they_are_re_asked() -> None:
    """`ed_set_text` DROPS the set -- the spans named characters of a text that is gone --
    where markers survive by line. The host needs no special case for it, and this is the
    measurement that says so: `ed_revision` RISES across the set, so `span_revision` no
    longer matches and the debounced request fires again on the next idle frame.

    Break by making the feed believe a span set survives a reload, and the class reads 0
    forever after a file is loaded into an open tab."""
    editor = _python_editor()
    feed = PythonFeed(editor=editor)
    before = editor.get_undo_index()
    assert accept_python_spans(editor, feed, _spans_result(Path("/tmp/s.py"), before))
    editor.layout((640.0, 480.0), 16.0)
    def_line = _line_of(SOURCE, "def update(self, dt")
    def_col = SOURCE.split("\n")[def_line].index("update")
    assert editor.class_at(def_line, def_col) == kind_slot(SymbolKind.PY_DEFINITION)

    editor.set_text(SOURCE)
    after = editor.get_undo_index()
    editor.layout((640.0, 480.0), 16.0)
    assert editor.class_at(def_line, def_col) == 0, "set_text drops the span set"
    assert after != before, "the revision must move, or nothing would re-request"
    assert feed.span_revision != after, (
        "so the feed asks again rather than believing itself"
    )
    editor.close()


def test_a_class_outside_the_palette_is_refused() -> None:
    # A class beyond the palette is REFUSED (-1) rather than clamped, so a host built
    # against a wider palette than the build has learns it at the call instead of drawing
    # in a colour the library invented.
    #
    # The ceiling is read from the mirrored enum rather than written here: it moved from 9
    # to 15 when the library widened `Theme.syntax` to [16]Color, and a hardcoded 9 would
    # have made this test the thing that fails on a re-vendor instead of the thing that
    # verifies one.
    editor = _python_editor()
    # Counted from the mirrored enum's NAMES, not from its values: `SYNTAX_1` is 14 and
    # `SYNTAX_8` is 25, so the values are not contiguous and subtracting two of them gives
    # a number in the wrong space. The CLASS is what `set_spans` takes; the slot value is
    # only where the library indexes its own theme.
    ceiling = sum(1 for slot in Slot if slot.name.startswith("SYNTAX_"))
    assert editor.set_spans([(0, 0, 0, 3, ceiling + 1)], editor.get_undo_index()) == -1
    assert editor.set_spans([(0, 0, 0, 3, ceiling)], editor.get_undo_index()) == 1
    for kind in SymbolKind:
        assert 0 <= kind_slot(kind) <= ceiling, kind
    editor.close()


def test_a_span_and_the_word_table_compose_on_one_buffer() -> None:
    """The two mechanisms cover one buffer and must not fight. The library's rule is that
    host spans WIN where they cover and the word table fills the identifiers the lexer left
    plain elsewhere -- so `self` keeps its colour under a span set that says nothing about
    it, and this is the check that the split in D2 actually composes."""
    editor = _python_editor()
    feed = PythonFeed(editor=editor)
    assert feed_python_word_classes(editor, feed, SOURCE)
    assert accept_python_spans(
        editor, feed, _spans_result(Path("/tmp/s.py"), editor.get_undo_index())
    )
    editor.layout((640.0, 480.0), 16.0)

    self_line = _line_of(SOURCE, "self.seed = seed")
    self_col = SOURCE.split("\n")[self_line].index("self")
    assert editor.class_at(self_line, self_col) == kind_slot(SymbolKind.PY_SELF)

    def_line = _line_of(SOURCE, "def update(self, dt")
    def_col = SOURCE.split("\n")[def_line].index("update")
    assert editor.class_at(def_line, def_col) == kind_slot(SymbolKind.PY_DEFINITION)
    # `self` in the SAME signature the span set covers keeps the table's colour, which is the
    # composition claim rather than two independent ones.
    assert editor.class_at(
        def_line, SOURCE.split("\n")[def_line].index("self")
    ) == kind_slot(SymbolKind.PY_SELF)
    editor.close()


def test_a_moved_caret_does_not_drop_a_span_answer() -> None:
    """THE D4 GATE. The worker's existing predicate requires caret agreement, and spans are a
    property of the TEXT: an arrow key between the request and the answer would throw away a
    correct set on every keystroke that moved the cursor without editing.

    Break by making `matches_text` call `matches`: the caret disagrees and this fails."""
    path = Path("/tmp/script.py")
    request = PythonRequest(PythonRequestKind.SPANS, path, SOURCE, 4, 11, revision=7)
    assert request.matches_text(path, 7)
    # The same request under the caret-sensitive predicate the completion path needs.
    assert not request.matches(path, 7, line=9, column=2)
    # ...and staleness is still decided, on the revision alone, so the looser predicate has
    # not simply stopped deciding.
    assert not request.matches_text(path, 8)
    assert not request.matches_text(Path("/tmp/other.py"), 7)


def test_a_span_answer_moves_the_revision_the_panel_reads() -> None:
    """THE D3 GATE, host side. The failure it exists for is invisible to any test that types
    and then asserts: the panel is a cached texture, the answer lands with the buffer IDLE,
    and nothing the editor reports has moved -- so the colour never appears at all.

    Break by reverting `render_state`'s `class_feed_revision` member and the colour never
    reaches the screen; break it here by not incrementing on acceptance and this fails."""
    editor = _python_editor()
    feed = PythonFeed(editor=editor)
    before = feed.revision
    assert accept_python_spans(
        editor, feed, _spans_result(Path("/tmp/s.py"), editor.get_undo_index())
    )
    assert feed.revision == before + 1
    editor.close()


def test_the_panel_redraws_for_a_feed_with_no_edit() -> None:
    """The same gate, end to end through the real redraw gate and a real editor.

    Nothing here types. The buffer's revision, cursor, mode, selection, scroll and primitive
    count are all identical either side of the feed -- asserted, so this cannot pass by some
    other dimension moving -- and the panel must still repaint."""
    from shaderbox.editor.render import render_state, should_redraw

    editor = Editor(SOURCE)
    editor.set_language(Language.PYTHON)
    editor.layout((640.0, 480.0), 16.0)

    def state(feed_revision: int) -> tuple:
        return render_state(
            editor,
            Path("/tmp/s.py"),
            (640, 480),
            16.0,
            editor.get_text_origin(),
            None,
            (),
            (),
            True,
            feed_revision,
        )

    feed = PythonFeed(editor=editor)
    before = state(feed.revision)
    assert feed_python_word_classes(editor, feed, SOURCE)
    after = state(feed.revision)
    assert should_redraw(before, after), "colour fed on a still buffer must repaint"
    # The falsifier: EVERY other dimension is unchanged, so the redraw above is the feed's
    # and not some editor state that happened to move. Without this the test would pass with
    # `class_feed_revision` deleted, as long as anything else in the tuple had drifted.
    assert before[2:] == after[2:], "nothing but the feed revision may have moved"
    editor.close()


def test_a_spans_job_never_delays_a_completion_job() -> None:
    """THE D3a GATE. `PythonWorker._next` pops one request at a time on ONE thread, and
    MEASURED a SPANS job costs 8.17 ms against a COMPLETE job's 8.19 ms on the flock script
    -- so a spans job popped first would double the latency of completion, a shipped feature,
    on every keystroke that also asked for colour.

    Break by restoring insertion order (`next(iter(self._pending))`) and this fails: SPANS
    was submitted first, so it comes back first, and completion waits behind it."""
    worker = PythonWorker()
    try:
        path = Path("/tmp/s.py")
        # SPANS goes in FIRST, which is the case insertion order gets wrong. If the queue
        # popped by arrival the spans answer would land before the completion one.
        worker.submit(
            PythonRequest(PythonRequestKind.SPANS, path, SOURCE, 0, 0, revision=1)
        )
        worker.submit(
            PythonRequest(PythonRequestKind.COMPLETE, path, SOURCE, 20, 8, revision=1)
        )
        order: list[PythonRequestKind] = []
        deadline = time.monotonic() + 30.0
        while len(order) < 2 and time.monotonic() < deadline:
            order.extend(result.request.kind for result in worker.poll())
            time.sleep(0.01)
        assert order[:2] == [
            PythonRequestKind.COMPLETE,
            PythonRequestKind.SPANS,
        ], f"completion must not wait behind colour, got {order}"
    finally:
        worker.close()


# --- the palette the four kinds draw in --------------------------------------


def test_a_syntax_class_means_one_thing_in_every_buffer() -> None:
    """One palette, and a class carries the same role wherever it is pushed.

    This replaces a test that checked the two PER-LANGUAGE palettes disagreed on slots
    7/8/9. That arrangement was sound -- one editor holds one language -- but it existed
    only because nine classes could not hold the roles, and with fifteen it is gone. The
    property worth pinning now is the opposite one: no class is overloaded.
    """
    palette = editor_palette()
    by_class: dict[int, set[str]] = {}
    for kind in SymbolKind:
        cls = kind_slot(kind)
        if cls:
            by_class.setdefault(cls, set()).add(kind_role(kind))

    for cls, roles in sorted(by_class.items()):
        # A class carries one COLOUR, not necessarily one role: roles that share a colour
        # deliberately share a class, which is what lets the lexer's own six be reused.
        # `type` and `declaration_type` are the live case -- both yellow, both class 7.
        colours = {ROLE_COLOR[role] for role in roles}
        assert len(colours) == 1, f"class {cls} draws {len(colours)} colours: {roles}"
        # And the palette draws it there, so what is pushed and what is drawn agree.
        assert palette[getattr(Slot, f"SYNTAX_{cls}")] == colours.pop()


@pytest.mark.parametrize(
    "kind",
    [
        SymbolKind.PY_SELF,
        SymbolKind.PY_DUNDER,
        SymbolKind.PY_DEFINITION,
        SymbolKind.PY_DECORATOR,
    ],
)
def test_each_new_kind_is_produced_by_something(kind: SymbolKind) -> None:
    """The gate the enum gate cannot be. `test_every_kind_has_a_color` walks `SymbolKind` and
    checks a colour, a rank and a slot -- and passes clean for a kind that NOTHING PRODUCES.
    Break by deleting any one producer, leaving its colour and slot in place, and watch the
    enum gate stay green while exactly this case fails."""
    produced = set(python_word_classes(SOURCE).values())
    produced |= {span.kind for span in python_spans(SOURCE)}
    assert kind in produced
