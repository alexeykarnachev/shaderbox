"""The graph canvas's gesture verbs on `App` (092 D12-D16), driven headlessly.

Every write the canvas makes goes through one of these, so each refusal is asserted here
without a window: the cycle drop, the media drop, the unwire, the drag commit, Group and
Dissolve. The widget itself is pinned to have no session write of its own.
"""

from pathlib import Path
from typing import Any
from unittest import mock

from shaderbox.pass_graph import NoSource, PassSource
from shaderbox.widgets import pass_graph
from shaderbox.widgets.graph_state import NodeDrag

_SAMPLER = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_src;
out vec4 fs_color;
void main() { fs_color = texture(u_src, vs_uv); }
"""


def _chain(app: Any) -> tuple[str, Any]:
    """`a -> b -> c` on the current document, each reading the one before through `u_src`."""
    document_id = app.current_document_id
    document = app.ui_documents[document_id].document
    for name in ("a", "b", "c"):
        assert app.session.add_pass(document_id, name) == ""
    for name in ("b", "c"):
        document.passes[name].release_program(_SAMPLER)
        document.passes[name].compile()
    assert app.drop_wire(document_id, "a", "b", "u_src") == ""
    assert app.drop_wire(document_id, "b", "c", "u_src") == ""
    return document_id, document


def test_drop_wire_writes_the_read_and_refuses_the_loop(app: Any) -> None:
    document_id, document = _chain(app)
    assert document.passes["c"].uniform_values["u_src"] == PassSource("b")
    document.passes["a"].release_program(_SAMPLER)
    document.passes["a"].compile()
    with mock.patch.object(
        app.session, "set_sampler_source", wraps=app.session.set_sampler_source
    ) as write:
        refusal = app.drop_wire(document_id, "c", "a", "u_src")
    # Falsifier: refuse only when the culprit is an endpoint -- `plan_passes` names `a` for a
    # drop whose endpoints are `c` and `a`, and the whole-list check is what refuses it.
    assert "passes form a cycle" in refusal
    assert write.call_count == 0, "a refused drop wrote"
    assert "u_src" not in document.passes["a"].uniform_values or not isinstance(
        document.passes["a"].uniform_values["u_src"], PassSource
    )
    # Feedback is allowed: a node's output into its own port.
    assert app.drop_wire(document_id, "c", "c", "u_src") == ""
    assert document.passes["c"].uniform_values["u_src"] == PassSource("c")


def test_drop_wire_refuses_a_media_bound_port_and_keeps_the_texture(app: Any) -> None:
    document_id, document = _chain(app)
    bound = document.gl.texture((2, 2), 4)
    document.passes["c"].uniform_values["u_src"] = bound
    with mock.patch.object(
        app.session, "set_sampler_source", wraps=app.session.set_sampler_source
    ) as write:
        refusal = app.drop_wire(document_id, "a", "c", "u_src")
    # Falsifier: call `set_sampler_source` unconditionally, which runs `try_to_release` on
    # the bound texture -- the one irreversible write a canvas gesture could make.
    assert "bound to media" in refusal
    assert write.call_count == 0
    assert document.passes["c"].uniform_values["u_src"] is bound
    assert bound.glo != 0


def test_unwire_writes_black_by_decision(app: Any) -> None:
    document_id, document = _chain(app)
    assert app.unwire(document_id, "c", "u_src") == ""
    assert document.passes["c"].uniform_values["u_src"] == NoSource()


def test_commit_node_drag_writes_once_and_only_the_moved_passes(
    app: Any, monkeypatch: Any
) -> None:
    document_id, document = _chain(app)
    view = app.graph_view_for(document_id)
    view.node_drag = NodeDrag(origin={"a": (0.0, 0.0), "gone": (5.0, 5.0)})
    view.node_drag.update(10.0, 20.0)
    with mock.patch.object(
        app.session, "save_ui_document", wraps=app.session.save_ui_document
    ) as saves:
        app.commit_node_drag(document_id)
    assert saves.call_count == 1
    assert document.graph.passes["a"].position == (10.0, 20.0)
    assert "gone" not in document.graph.passes
    assert view.node_drag is None
    # A commit with no drag in flight writes nothing.
    with mock.patch.object(
        app.session, "save_ui_document", wraps=app.session.save_ui_document
    ) as saves:
        app.commit_node_drag(document_id)
    assert saves.call_count == 0


def test_group_selection_and_dissolve_are_one_write_each(app: Any) -> None:
    document_id, document = _chain(app)
    view = app.graph_view_for(document_id)
    view.selection = {"a", "b"}
    with mock.patch.object(
        app.session, "save_ui_document", wraps=app.session.save_ui_document
    ) as saves:
        assert app.group_selection(document_id, "pair") == ""
    assert saves.call_count == 1
    assert document.graph.passes["a"].group == "pair"
    assert document.graph.passes["b"].group == "pair"
    assert document.graph.passes["c"].group == ""
    # Grouping a box with a plain pass rewrites the members to the new label (flat labels).
    view.selection = {"a", "b", "c"}
    assert app.group_selection(document_id, "trio") == ""
    assert {document.graph.passes[n].group for n in ("a", "b", "c")} == {"trio"}
    # A name a pass carries is refused and nothing changes.
    assert (
        app.group_selection(document_id, "a")
        == "a pass and a group cannot share a name"
    )
    assert {document.graph.passes[n].group for n in ("a", "b", "c")} == {"trio"}
    with mock.patch.object(
        app.session, "save_ui_document", wraps=app.session.save_ui_document
    ) as saves:
        assert app.dissolve_group(document_id, "trio") == ""
    assert saves.call_count == 1
    assert all(document.graph.passes[n].group == "" for n in ("a", "b", "c"))


def test_the_widget_makes_no_session_write_of_its_own() -> None:
    # 092 D12: every write goes through an App verb, so its refusal is testable here.
    source = Path(pass_graph.__file__).read_text(encoding="utf-8")
    for forbidden in ("set_sampler_source", "set_pass_positions", "set_pass_groups"):
        assert forbidden not in source, forbidden
