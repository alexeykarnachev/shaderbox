"""Next / previous pass (073 W-D): the strip's drawn order, wrapping, and the rule W3-3 set
for clicks -- the walk moves the OUTPUT and opens no shader tab."""

from typing import Any

from shaderbox.commands import CommandId
from shaderbox.pass_graph import step_in_order
from shaderbox.paths import pass_name_of


def test_step_wraps_at_both_ends() -> None:
    order = ["a", "b", "c"]
    assert step_in_order(order, "c", 1) == "a"
    assert step_in_order(order, "a", -1) == "c"
    assert step_in_order(order, "b", 1) == "c"


def test_step_from_an_unknown_current_lands_on_the_first_tile() -> None:
    assert step_in_order(["a", "b"], "zzz", 1) == "a"
    assert step_in_order([], "a", 1) is None


def test_the_commands_walk_the_strip_and_open_no_tab(
    app: Any, monkeypatch: Any
) -> None:
    # Falsifier: point the two callbacks back at `ensure_shader_tab` and `opened` fills.
    document_id = app.current_document_id
    assert app.session.add_pass(document_id, "bright") == ""
    document = app.ui_documents[document_id].document
    order = sorted(document.passes)
    assert len(order) == 2

    opened: list[str] = []
    monkeypatch.setattr(
        app,
        "ensure_shader_tab",
        lambda doc_id, name="", focus_editor=False: opened.append(name),
    )

    app.editor_focused = True
    start = document.graph.output
    app.command_callbacks[CommandId.NEXT_PASS]()
    assert document.graph.output == step_in_order(order, start, 1)

    app.editor_focused = False
    app.command_callbacks[CommandId.PREV_PASS]()
    assert document.graph.output == start

    assert opened == [], "the pass walk opened a shader tab"


def test_open_shader_takes_the_worked_on_pass_not_the_output(app: Any) -> None:
    """A shader belongs to a PASS, so `Open shader` resolves the way `Pass settings` does:
    through `panel_pass`, the pass being worked on.

    Falsifier: point the callback back at a bare `ensure_shader_tab(document_id)` and the tab
    lands on the output instead.
    """
    document_id = app.current_document_id
    assert app.session.add_pass(document_id, "bright") == ""
    document = app.ui_documents[document_id].document
    output = document.graph.output
    worked_on = next(name for name in document.passes if name != output)

    app.set_panel_pass(document_id, worked_on)
    app.command_callbacks[CommandId.OPEN_SHADER]()

    active = app.active_tab
    assert active is not None
    assert active.path == document.passes[worked_on].source.path, (
        "Open shader landed on the output, not the worked-on pass"
    )
    # The tab now carries the pick, so the panel stays put without the explicit override.
    assert pass_name_of(app.panel_pass(document_id).source.path) == worked_on


def test_open_shader_reaches_a_pass_while_another_shader_is_open(app: Any) -> None:
    """With a shader tab already open, picking another pass and firing `Open shader` must
    reach the picked pass.

    `panel_pass` falls through to the ACTIVE TAB when no pass was picked, so a pick that
    cleared the record left the command resolving to the shader already on screen -- it
    looked like the hotkey did nothing. Falsifier: clear the pick in `choose_output` again.
    """
    document_id = app.current_document_id
    assert app.session.add_pass(document_id, "bright") == ""
    document = app.ui_documents[document_id].document
    output = document.graph.output
    other = next(name for name in document.passes if name != output)

    app.ensure_shader_tab(document_id, output, focus_editor=True)
    app.choose_output(document_id, other)
    app.command_callbacks[CommandId.OPEN_SHADER]()

    active = app.active_tab
    assert active is not None
    assert active.path == document.passes[other].source.path, (
        "Open shader stayed on the shader that was already open"
    )
