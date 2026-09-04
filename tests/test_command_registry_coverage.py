"""Every command reaches every surface that lists commands (078). A command added to the enum
without a spec, a handler, or a help line is the gap the maintainer asked to be gated: the
quick help pane and the Help panel's shortcuts both read the registry, so the gate is that the
registry itself is complete."""

from typing import Any

from shaderbox.commands import COMMAND_SPECS, SPEC_BY_ID, CommandId
from shaderbox.help_content import _shortcuts_section


def test_every_command_id_has_a_spec() -> None:
    assert set(SPEC_BY_ID) == set(CommandId)


def test_every_command_id_has_a_handler(app: Any) -> None:
    assert set(app.command_callbacks) == set(CommandId)


def test_every_bound_spec_reaches_the_help_shortcuts() -> None:
    snippet = _shortcuts_section().snippet
    for spec in COMMAND_SPECS:
        if spec.default_chord:
            assert spec.label in snippet, f"{spec.id} is bound but not in the help list"
