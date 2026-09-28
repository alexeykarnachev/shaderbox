"""Help content is GENERATED from the code it documents (feature 055), so these pin the generators
rather than the prose: a new engine uniform or a new command category must not ship undocumented.
GL-free (no App, no imgui)."""

import re

from shaderbox.commands import CATEGORY_ORDER, COMMAND_SPECS, CommandId, chord_to_str
from shaderbox.core import ENGINE_UNIFORM_TYPES
from shaderbox.help_content import (
    ENGINE_UNIFORM_DOCS,
    documented_instanced_names,
    help_sections,
    user_facing_engine_uniforms,
)
from shaderbox.instanced import ENGINE_INTERNAL_NAMES, USER_FACING_NAMES


def test_engine_uniform_docs_cover_every_user_facing_builtin() -> None:
    # The wire that keeps the panel honest: add a builtin to ENGINE_DRIVEN_UNIFORMS without a doc
    # entry and this fails instead of silently shipping incomplete help.
    assert set(ENGINE_UNIFORM_DOCS) == user_facing_engine_uniforms()


def test_engine_uniform_section_lists_each_uniform() -> None:
    section = next(s for s in help_sections() if s.key == "engine_uniforms")
    for name in ENGINE_UNIFORM_DOCS:
        assert f"uniform {ENGINE_UNIFORM_TYPES[name]} {name};" in section.snippet


def test_every_user_facing_instanced_name_is_documented() -> None:
    # 104 D4: `instanced.USER_FACING_NAMES` is the partition the engine grew so this gate could
    # exist at all -- documenting `sb_instanced` (ENGINE_INTERNAL_NAMES) would be the opposite
    # of that decision. `documented_instanced_names()` reads back what help_content actually
    # covers, so a name added to USER_FACING_NAMES without prose fails here rather than
    # shipping silently missing.
    assert documented_instanced_names() == USER_FACING_NAMES


def test_engine_internal_instanced_names_are_absent_from_every_section() -> None:
    # The other half of D4: sb_instanced and a_corner must NEVER appear, so the partition
    # cannot erode by someone helpfully "completing" the vocabulary section later.
    #
    # Contact proof for the silence: help_sections() is asserted non-trivial (title, snippet
    # and a body of real length present) BEFORE the absence check runs, so a fixture that
    # returned nothing at all -- e.g. a broken import short-circuiting to an empty list --
    # cannot read as "correctly absent". A search for the wrong string and a search that
    # never ran must not produce the same green.
    sections = help_sections()
    assert len(sections) >= 5
    joined_body = "\n".join(s.body for s in sections)
    joined_snippet = "\n".join(s.snippet for s in sections)
    assert len(joined_body) > 500  # real prose was read, not an empty scaffold
    for name in ENGINE_INTERNAL_NAMES:
        assert name not in joined_body, name
        assert name not in joined_snippet, name


def test_sections_are_well_formed() -> None:
    sections = help_sections()
    assert len(sections) >= 5  # an empty list would IndexError at open_help
    keys = [s.key for s in sections]
    assert len(keys) == len(set(keys))  # the modal indexes by key
    for s in sections:
        assert s.key and s.title and s.body


def test_shortcuts_section_covers_every_populated_category() -> None:
    section = next(s for s in help_sections() if s.key == "shortcuts")
    for category in CATEGORY_ORDER:
        if any(s.category == category and s.default_chord for s in COMMAND_SPECS):
            assert category.value in section.snippet
    help_spec = next(s for s in COMMAND_SPECS if s.id is CommandId.HELP)
    assert chord_to_str(help_spec.default_chord) in section.snippet


def test_shortcuts_section_lists_every_bound_command() -> None:
    # The category-level assertion above passes for a new command in an already-populated
    # category, so a bound chord could ship undocumented. This pins each spec individually.
    section = next(s for s in help_sections() if s.key == "shortcuts")
    for spec in COMMAND_SPECS:
        if not spec.default_chord:
            continue
        assert spec.label in section.snippet, spec.label
        assert chord_to_str(spec.default_chord) in section.snippet, spec.label


# A backticked chord in hand-written prose: `Ctrl+P`, `Alt+/`, `F8`, `Ctrl+Shift+N`.
_PROSE_CHORD = re.compile(r"`((?:Ctrl|Alt|Shift)\+[^`]+|F[0-9]{1,2})`")


def test_no_help_prose_quotes_a_chord_the_table_does_not_bind() -> None:
    # The generated shortcuts table follows COMMAND_SPECS for free, so a chord move updates it
    # silently — but a chord typed into a section BODY does not move with it, and the user
    # reads that body. 069 W-E shipped `Ctrl+P` for the library one commit after the chord
    # became Alt+L. Every hand-written chord must be one the table currently binds.
    bound = {
        chord_to_str(spec.default_chord) for spec in COMMAND_SPECS if spec.default_chord
    }
    stale: list[str] = []
    for section in help_sections():
        for quoted in _PROSE_CHORD.findall(section.body):
            if quoted not in bound:
                stale.append(f"{section.key}: {quoted}")
    assert stale == [], (
        f"help prose names chords no CommandSpec binds (bound: {sorted(bound)}): {stale}"
    )
