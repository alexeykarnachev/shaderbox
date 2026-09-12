"""The graph canvas's gesture verbs on `App` (092 D12-D16), driven headlessly.

Every write the canvas makes goes through one of these, so each refusal is asserted here
without a window: the cycle drop, the media drop, the unwire, the drag commit, Group and
Dissolve. The widget itself is pinned to have no session write of its own.
"""

from pathlib import Path
from typing import Any
from unittest import mock

from imgui_bundle import imgui

from shaderbox.pass_graph import NoSource, PassSource
from shaderbox.ui import update_and_draw
from shaderbox.ui_regions import PassesView
from shaderbox.widgets import pass_graph, pass_list
from shaderbox.widgets.graph_state import NodeDrag, WireDrag

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


def test_arrange_graph_saves_once_and_leaves_no_pass_unplaced(app: Any) -> None:
    document_id, document = _chain(app)
    with mock.patch.object(
        app.session, "save_ui_document", wraps=app.session.save_ui_document
    ) as saves:
        app.arrange_graph(document_id)
    assert saves.call_count == 1
    assert all(entry.position is not None for entry in document.graph.passes.values())
    assert app.graph_view_for(document_id).fitted is False


def test_dissolve_of_nothing_and_leave_group_write_as_they_should(app: Any) -> None:
    document_id, document = _chain(app)
    with mock.patch.object(
        app.session, "save_ui_document", wraps=app.session.save_ui_document
    ) as saves:
        assert app.dissolve_group(document_id, "") == ""
        assert app.dissolve_group(document_id, "nope") == ""
    assert saves.call_count == 0, "a dissolve of nothing saved"
    assert app.session.set_pass_group(document_id, "a", "g") == ""
    assert app.leave_group(document_id, "a") == ""
    assert document.graph.passes["a"].group == ""


def test_the_widget_makes_no_session_write_of_its_own() -> None:
    # 092 D12: every write goes through an App verb, so its refusal is testable here. The
    # strip's menu is shared with the canvas, so it is held to the same rule.
    for module in (pass_graph, pass_list):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "set_sampler_source",
            "set_pass_positions",
            "set_pass_groups",
            "set_pass_group(",
        ):
            assert forbidden not in source, (module.__name__, forbidden)


# ----------------------------------------------------------------
# The one gesture the verbs cannot cover: a drop lands where the port is DRAWN. Driven through
# the real frame loop with synthetic mouse input, so the gate sees the hover that feeds `_drop`.


def _frames(app: Any, n: int) -> None:
    for _ in range(n):
        update_and_draw(app)


def test_a_wire_dropped_on_a_drawn_port_writes_that_port(app: Any) -> None:
    # Falsifier: `set_next_item_allow_overlap()` on the port rects (round 2's regression) --
    # the flag costs the LAST rung its own hover, `drop_target` never sets, and the drop
    # writes nothing at any zoom.
    document_id, document = _chain(app)
    app.app_state.passes_view = PassesView.GRAPH
    _frames(app, 4)
    view = app.graph_view_for(document_id)
    assert ("c", "u_src") in view.port_rects, sorted(view.port_rects)
    x0, y0, x1, y1 = view.port_rects[("c", "u_src")]
    io = imgui.get_io()
    # Press on empty canvas (its bottom-right corner, inside the child: a press outside any
    # window is owned by the application and imgui reports no hover for the rest of the drag),
    # then carry a wire from `a` over the port.
    cx1, cy1 = view.canvas_rect[2], view.canvas_rect[3]
    io.add_mouse_pos_event(cx1 - 6.0, cy1 - 6.0)
    _frames(app, 2)
    io.add_mouse_button_event(0, True)
    _frames(app, 2)
    view.wire_drag = WireDrag(producer="a", start=(0.0, 0.0))
    io.add_mouse_pos_event((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    _frames(app, 3)
    assert view.wire_drag is not None, "the wire in flight was cancelled mid-drag"
    io.add_mouse_button_event(0, False)
    _frames(app, 2)
    assert document.passes["c"].uniform_values["u_src"] == PassSource("a")
    assert view.wire_drag is None
    app.app_state.passes_view = PassesView.STRIP
    _frames(app, 1)


def test_a_gesture_cannot_start_during_a_copilot_turn(app: Any) -> None:
    # Round 2's regression: the turn's cancel ran, then the same frame's press re-armed the
    # drag, and the release after the turn committed a position no gesture asked for.
    document_id, document = _chain(app)
    app.app_state.passes_view = PassesView.GRAPH
    _frames(app, 4)
    view = app.graph_view_for(document_id)
    x0, y0, x1, y1 = view.port_rects[("b", "u_src")]
    io = imgui.get_io()
    # The frame loop reconciles `copilot_turn_active` from the session each frame, so the
    # turn is simulated where it lives.
    app.copilot.state.in_flight = True
    io.add_mouse_pos_event((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    _frames(app, 2)
    io.add_mouse_button_event(0, True)
    _frames(app, 2)
    io.add_mouse_pos_event(x1 + 60.0, y1 + 60.0)
    _frames(app, 3)
    assert view.wire_drag is None and view.node_drag is None
    app.copilot.state.in_flight = False
    io.add_mouse_button_event(0, False)
    _frames(app, 2)
    assert document.passes["b"].uniform_values["u_src"] == PassSource("a")
    assert all(entry.position is None for entry in document.graph.passes.values())
    app.app_state.passes_view = PassesView.STRIP
    _frames(app, 1)
