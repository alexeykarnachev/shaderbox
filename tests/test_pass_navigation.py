"""Next / previous pass (073 W-D): the strip's drawn order, wrapping, and the focus rule --
the editor keeps keyboard focus when it had it and is left alone when it did not."""

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


def test_the_commands_walk_the_strip_and_mirror_the_focus(
    app: Any, monkeypatch: Any
) -> None:
    document_id = app.current_document_id
    assert app.session.add_pass(document_id, "bright") == ""
    document = app.ui_documents[document_id].document
    order = sorted(document.passes)
    assert len(order) == 2

    picked: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        app,
        "ensure_shader_tab",
        lambda doc_id, name="", focus_editor=False: picked.append((name, focus_editor)),
    )

    app.editor_focused = True
    start = document.graph.output
    app.command_callbacks[CommandId.NEXT_PASS]()
    assert picked[-1] == (step_in_order(order, start, 1), True)
    assert document.graph.output == picked[-1][0]

    app.editor_focused = False
    app.command_callbacks[CommandId.PREV_PASS]()
    assert picked[-1] == (start, False)
    assert document.graph.output == start
