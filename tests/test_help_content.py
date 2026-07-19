"""Help content is GENERATED from the code it documents (feature 055), so these pin the generators
rather than the prose: a new engine uniform or a new command category must not ship undocumented.
GL-free (no App, no imgui)."""

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
