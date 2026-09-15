"""Next / previous pass (073 W-D): the strip's drawn order, wrapping, and the rule W3-3 set
for clicks -- the walk moves the OUTPUT and opens no shader tab."""

from typing import Any

from shaderbox.commands import CommandId
from shaderbox.pass_graph import step_in_order


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
