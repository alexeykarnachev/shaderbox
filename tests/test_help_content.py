"""Help content is GENERATED from the code it documents (feature 055), so these pin the generators
rather than the prose: a new engine uniform or a new command category must not ship undocumented.
GL-free (no App, no imgui)."""

import re

from shaderbox.commands import CATEGORY_ORDER, COMMAND_SPECS, chord_to_str
from shaderbox.help_content import (
    ENGINE_UNIFORM_DOCS,
    help_sections,
    user_facing_engine_uniforms,
)


def test_engine_uniform_docs_cover_every_user_facing_builtin() -> None:
    # The wire that keeps the panel honest: add a builtin to ENGINE_DRIVEN_UNIFORMS without a doc
    # entry and this fails instead of silently shipping incomplete help.
    assert set(ENGINE_UNIFORM_DOCS) == user_facing_engine_uniforms()


def test_engine_uniform_section_lists_each_uniform() -> None:
    section = next(s for s in help_sections() if s.key == "engine_uniforms")
    for name, (glsl_type, _doc) in ENGINE_UNIFORM_DOCS.items():
        assert f"uniform {glsl_type} {name};" in section.snippet


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
    help_spec = next(s for s in COMMAND_SPECS if s.label == "Help")
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
