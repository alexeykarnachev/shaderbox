"""The graph canvas's gesture verbs on `App` (092 D12-D16), driven headlessly.

Every write the canvas makes goes through one of these, so each refusal is asserted here
without a window: the cycle drop, the media drop, the unwire, the drag commit, Group and
Dissolve. The widget itself is pinned to have no session write of its own.
"""

from dataclasses import fields
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from imgui_bundle import imgui

from shaderbox.pass_graph import NoSource, PassSource
from shaderbox.ui import update_and_draw
from shaderbox.widgets import pass_graph, pass_list
from shaderbox.widgets.graph_state import (
    GraphViewState,
    NodeDrag,
    WireDrag,
    node_size,
)

# The imgui font atlas is process-global, so every frame-driving module owns a worker
# (`pyproject.toml`).
pytestmark = pytest.mark.xdist_group("gl_frames_graph_view")

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
    app.open_graph_for(document_id)
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
    app.close_editor_for_path(app.paths.graph_json_for(document_id))
    _frames(app, 1)


def test_a_press_that_spans_a_copilot_turn_never_becomes_a_gesture(app: Any) -> None:
    # A wire started BEFORE the turn: the turn cancels it, and when the turn ends the button
    # is still down and the dot's item still active, so the start branch would rebuild the
    # gesture from a press nobody made after the turn. Falsifier: drop the `press_blocked`
    # latch -- the release after the turn drops the wire on whatever is under the cursor.
    # The wire starts at an OUTPUT dot: an input pin is not a drag source (093 W2-4).
    document_id, document = _chain(app)
    app.open_graph_for(document_id)
    _frames(app, 4)
    view = app.graph_view_for(document_id)
    x0, y0, x1, y1 = view.out_rects[("p:a", 0)]
    io = imgui.get_io()
    io.add_mouse_pos_event((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    _frames(app, 2)
    io.add_mouse_button_event(0, True)
    _frames(app, 2)
    io.add_mouse_pos_event(x1 + 60.0, y1 + 60.0)
    _frames(app, 3)
    assert view.wire_drag is not None, "the grab never started"
    # The frame loop reconciles `copilot_turn_active` from the session each frame, so the
    # turn is simulated where it lives.
    app.copilot.state.in_flight = True
    _frames(app, 3)
    assert view.wire_drag is None and view.node_drag is None, "the turn did not cancel"
    app.copilot.state.in_flight = False
    _frames(app, 3)  # the turn is over, the button is still down
    assert view.wire_drag is None and view.node_drag is None, "the press re-armed"
    io.add_mouse_pos_event(x1 + 120.0, y1 + 90.0)
    _frames(app, 2)
    io.add_mouse_button_event(0, False)
    _frames(app, 2)
    assert document.passes["b"].uniform_values["u_src"] == PassSource("a")
    assert all(entry.position is None for entry in document.graph.passes.values())
    _let_the_double_click_lapse(app)
    # The latch clears with the button: the next press is a gesture again.
    io.add_mouse_pos_event((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    _frames(app, 2)
    io.add_mouse_button_event(0, True)
    _frames(app, 2)
    io.add_mouse_pos_event(x1 + 60.0, y1 + 60.0)
    _frames(app, 3)
    assert view.wire_drag is not None
    io.add_mouse_button_event(0, False)
    _frames(app, 2)
    app.close_editor_for_path(app.paths.graph_json_for(document_id))
    _frames(app, 1)


# ----------------------------------------------------------------
# The canvas's own gestures (093), driven through the real frame loop. Three mechanics every
# test here obeys, each measured on this imgui build: a key event queued in the same batch as
# a mouse-button event reaches `is_key_pressed` one frame LATER than the button, so a `_frames`
# call separates them; a move and a release arriving in one frame read as a click, so they are
# separate frames too; and a test that needs the canvas to have drawn asserts `canvas_rect`
# first, since every geometry field it then reads is written only on a sized frame.


def _open_graph(app: Any, document_id: str) -> Any:
    app.open_graph_for(document_id)
    _frames(app, 4)
    view = app.graph_view_for(document_id)
    assert view.canvas_rect != (0.0, 0.0, 0.0, 0.0), "the canvas never drew"
    return view


def _park(app: Any, point: tuple[float, float], frames: int = 3) -> None:
    imgui.get_io().add_mouse_pos_event(point[0], point[1])
    _frames(app, frames)


def _click_at(app: Any, point: tuple[float, float]) -> None:
    _park(app, point)
    imgui.get_io().add_mouse_button_event(0, True)
    _frames(app, 2)
    imgui.get_io().add_mouse_button_event(0, False)
    _frames(app, 2)


def _let_the_double_click_lapse(app: Any) -> None:
    """Run frames until imgui's double-click window has closed.

    A test's two presses land microseconds apart on almost the same pixel, which imgui reads
    as ONE double-click -- a real hand aiming from a wire to its badge never does. Frames are
    free here, and the alternative (moving the mouse far away between presses) would change
    what the gesture under test is."""
    deadline = imgui.get_io().mouse_double_click_time + 0.05
    elapsed = 0.0
    while elapsed < deadline:
        elapsed += imgui.get_io().delta_time
        _frames(app, 1)


def _press_key(app: Any, key: Any) -> None:
    imgui.get_io().add_key_event(key, True)
    _frames(app, 2)
    imgui.get_io().add_key_event(key, False)
    _frames(app, 1)


def test_a_wire_is_selected_by_a_click_and_deleted_by_the_key(app: Any) -> None:
    # G5: the whole point of finding 1 -- a wire can be reached at all. Falsifier: read the
    # Delete key at the TOP of `_draw_canvas`, where on the release frame `is_any_item_active`
    # is still True (measured), and the key is dead on exactly the frame a user who just
    # clicked the wire presses it.
    document_id, document = _chain(app)
    view = _open_graph(app, document_id)
    assert ("c", "u_src") in view.wire_mids, sorted(view.wire_mids)
    _click_at(app, view.wire_mids[("c", "u_src")])
    assert view.selected_wire == ("c", "u_src")
    assert view.selection == set()
    with mock.patch.object(
        app.session, "set_sampler_source", wraps=app.session.set_sampler_source
    ) as write:
        _press_key(app, imgui.Key.delete)
    assert write.call_count == 1, [c.args for c in write.call_args_list]
    assert write.call_args.args[1:] == (document_id, "c", "u_src", NoSource())[1:] or (
        write.call_args.args[0],
        write.call_args.args[1],
        write.call_args.args[2],
    ) == (document_id, "c", "u_src")
    assert document.passes["c"].uniform_values["u_src"] == NoSource()
    _close_graph(app, document_id)


def _close_graph(app: Any, document_id: str) -> None:
    app.close_editor_for_path(app.paths.graph_json_for(document_id))
    _frames(app, 1)


def test_the_wires_own_badge_unwires_it_and_the_press_is_nothing_else(app: Any) -> None:
    """G5/S6: the badge is not an imgui item -- an earlier item that declares no overlap beats
    a later one, and a short wire's midpoint lands inside its consumer port's box -- so it is
    hand hit-tested on the press and latches `press_blocked`.

    Break to try: clear the latch at the TOP of the frame as 092 did -- the covered case below
    both unwires the sampler AND, on the release, chooses the covering node as the output.
    """
    document_id, document = _chain(app)
    view = _open_graph(app, document_id)
    _click_at(app, view.wire_mids[("b", "u_src")])
    assert view.selected_wire == ("b", "u_src")
    selection_before = set(view.selection)
    assert view.x_rect is not None
    center = (
        (view.x_rect[0] + view.x_rect[2]) / 2.0,
        (view.x_rect[1] + view.x_rect[3]) / 2.0,
    )
    _let_the_double_click_lapse(app)
    with (
        mock.patch.object(
            app.session, "set_sampler_source", wraps=app.session.set_sampler_source
        ) as write,
        mock.patch.object(
            app.session, "set_output_pass", wraps=app.session.set_output_pass
        ) as output,
    ):
        _park(app, center)
        imgui.get_io().add_mouse_button_event(0, True)
        _frames(app, 2)
        assert view.band_anchor is None and view.node_drag is None
        imgui.get_io().add_mouse_button_event(0, False)
        _frames(app, 2)
    assert write.call_count == 1, [c.args for c in write.call_args_list]
    assert output.call_count == 0, "the press became an output choice as well"
    assert view.band_anchor is None and view.node_drag is None
    assert view.selection == selection_before
    assert document.passes["b"].uniform_values["u_src"] == NoSource()
    _close_graph(app, document_id)


def test_the_badge_wins_over_a_card_that_covers_the_wire(app: Any) -> None:
    # The normal case under G15: a wire runs UNDER a node. The card is shifted toward the
    # producer so its body covers the midpoint but not the port the wire ends at -- centered on
    # the midpoint it would cover both. Break to try: the same latch clear as above.
    document_id, document = _chain(app)
    view = _open_graph(app, document_id)
    _click_at(app, view.wire_mids[("b", "u_src")])
    assert view.selected_wire == ("b", "u_src")
    xf_mid = view.wire_mids[("b", "u_src")]
    canvas_mid = (
        (xf_mid[0] - view.canvas_rect[0]) / view.zoom + view.pan[0],
        (xf_mid[1] - view.canvas_rect[1]) / view.zoom + view.pan[1],
    )
    size = node_size(1, False)
    app.session.set_pass_positions(
        document_id,
        {
            "c": (
                canvas_mid[0] - size[0] / 2.0 - 50.0,
                canvas_mid[1] - size[1] / 2.0,
            )
        },
    )
    _frames(app, 3)
    assert view.x_rect is not None
    center = (
        (view.x_rect[0] + view.x_rect[2]) / 2.0,
        (view.x_rect[1] + view.x_rect[3]) / 2.0,
    )
    _let_the_double_click_lapse(app)
    with (
        mock.patch.object(
            app.session, "set_sampler_source", wraps=app.session.set_sampler_source
        ) as write,
        mock.patch.object(
            app.session, "set_output_pass", wraps=app.session.set_output_pass
        ) as output,
    ):
        _park(app, center)
        imgui.get_io().add_mouse_button_event(0, True)
        _frames(app, 2)
        imgui.get_io().add_mouse_button_event(0, False)
        _frames(app, 2)
    assert write.call_count == 1, [c.args for c in write.call_args_list]
    assert output.call_count == 0, (
        "the release-frame node click fired through the latch"
    )
    assert document.passes["b"].uniform_values["u_src"] == NoSource()
    _close_graph(app, document_id)


def test_delete_is_refused_while_a_press_is_held_on_the_canvas(app: Any) -> None:
    # S5, a behavior pin: the clause it exercises is `hovered` -- `is_window_hovered` is False
    # for the whole duration of a held press (measured), which is why the gate needs no
    # separate "a gesture is in flight" clause.
    document_id, document = _chain(app)
    view = _open_graph(app, document_id)
    _click_at(app, view.wire_mids[("c", "u_src")])
    assert view.selected_wire == ("c", "u_src")
    empty = (view.canvas_rect[2] - 6.0, view.canvas_rect[3] - 6.0)
    with mock.patch.object(
        app.session, "set_sampler_source", wraps=app.session.set_sampler_source
    ) as write:
        _park(app, empty)
        imgui.get_io().add_mouse_button_event(0, True)
        _frames(app, 2)
        _press_key(app, imgui.Key.delete)
        assert write.call_count == 0, "Delete fired under a held press"
        imgui.get_io().add_mouse_button_event(0, False)
        _frames(app, 2)
    # The press on empty canvas cleared the selection, so re-select before the positive half.
    _click_at(app, view.wire_mids[("c", "u_src")])
    assert view.selected_wire == ("c", "u_src")
    with mock.patch.object(
        app.session, "set_sampler_source", wraps=app.session.set_sampler_source
    ) as write:
        _press_key(app, imgui.Key.delete)
    assert write.call_count == 1
    assert document.passes["c"].uniform_values["u_src"] == NoSource()
    _close_graph(app, document_id)


def test_delete_typed_into_the_group_prompt_is_refused(app: Any) -> None:
    # S5, a behavior pin on the same `hovered` clause: the prompt is a plain `begin_popup`, so
    # `any_popup_open()` is False for it while `is_window_hovered(child_windows)` is already
    # False (the binding's own doc: "not blocked by a popup/modal").
    document_id, document = _chain(app)
    view = _open_graph(app, document_id)
    _click_at(app, view.wire_mids[("c", "u_src")])
    assert view.selected_wire == ("c", "u_src")
    # A one-shot the first frame consumes; re-asserting it would reopen the popup each frame.
    view.group_prompt = True
    _frames(app, 3)
    with mock.patch.object(
        app.session, "set_sampler_source", wraps=app.session.set_sampler_source
    ) as write:
        _press_key(app, imgui.Key.delete)
    assert write.call_count == 0
    assert document.passes["c"].uniform_values["u_src"] == PassSource("b")
    _close_graph(app, document_id)


def test_delete_is_refused_during_a_copilot_turn(app: Any) -> None:
    # S5's `not blocked` clause: a turn freezes every canvas write, and a key is no exception.
    document_id, document = _chain(app)
    view = _open_graph(app, document_id)
    _click_at(app, view.wire_mids[("c", "u_src")])
    assert view.selected_wire == ("c", "u_src")
    app.copilot.state.in_flight = True
    _frames(app, 3)
    with mock.patch.object(
        app.session, "set_sampler_source", wraps=app.session.set_sampler_source
    ) as write:
        _press_key(app, imgui.Key.delete)
    assert write.call_count == 0
    app.copilot.state.in_flight = False
    _frames(app, 3)
    assert document.passes["c"].uniform_values["u_src"] == PassSource("b")
    _close_graph(app, document_id)


def _hover_fields(view: Any) -> tuple[Any, Any, Any, Any]:
    return (view.hovered_port, view.hovered_out, view.hovered_node, view.hovered_wire)


def test_exactly_one_thing_is_hovered_and_the_rungs_are_in_order(app: Any) -> None:
    """G6: port, then node, then wire, then background -- each rung short-circuiting the rest,
    so a cue never says two things at once.

    Break to try: swap the node and wire rungs, and (d) flips. The port-over-node rung is NOT
    breakable from the resolution chain: imgui gives the overlap to the port button submitted
    last, so at a dot the node's own hover already reads False and the chain has nothing left
    to decide (measured -- swapping those two rungs leaves every row green, and so does
    dropping the node button's `set_next_item_allow_overlap`). What (a) pins is the
    submission chain's OUTCOME, which is what the cue is drawn from.
    """
    document_id, document = _chain(app)
    view = _open_graph(app, document_id)
    x0, y0, x1, y1 = view.port_rects[("c", "u_src")]

    # (a) a port dot outranks the node body it sits on. Aimed a pixel INSIDE the card rather
    # than at the dot's center, which sits on the card's left edge: on the boundary the node
    # button does not contain the point, and the rung would not be exercised at all.
    _park(app, ((x0 + x1) / 2.0 + 2.0, (y0 + y1) / 2.0))
    port, out, node, wire = _hover_fields(view)
    assert port is not None, "the port rung did not fire"
    assert (out, node, wire) == (None, None, None)

    # (b) a node body, away from every dot and every wire.
    picture = pass_graph._build_view(document, "", {})
    node_c = picture.nodes["p:c"]
    body = (
        node_c.pos[0] + node_c.size[0] * 0.75,
        node_c.pos[1] + node_c.size[1] * 0.2,
    )
    _park(
        app,
        (
            view.canvas_rect[0] + (body[0] - view.pan[0]) * view.zoom,
            view.canvas_rect[1] + (body[1] - view.pan[1]) * view.zoom,
        ),
    )
    port, out, node, wire = _hover_fields(view)
    assert node == "p:c", (node, port, out, wire)
    assert (port, out, wire) == (None, None, None)

    # (c) a wire in open canvas.
    _park(app, view.wire_mids[("b", "u_src")])
    port, out, node, wire = _hover_fields(view)
    assert wire == ("b", "u_src"), (wire, node, port, out)
    assert (port, out, node) == (None, None, None)

    # (d) a node body COVERING that wire: the node wins, the wire goes quiet.
    mid = view.wire_mids[("b", "u_src")]
    canvas_mid = (
        (mid[0] - view.canvas_rect[0]) / view.zoom + view.pan[0],
        (mid[1] - view.canvas_rect[1]) / view.zoom + view.pan[1],
    )
    size = node_size(1, False)
    app.session.set_pass_positions(
        document_id,
        {"c": (canvas_mid[0] - size[0] / 2.0 - 50.0, canvas_mid[1] - size[1] / 2.0)},
    )
    _frames(app, 3)
    _park(app, view.wire_mids[("b", "u_src")])
    port, out, node, wire = _hover_fields(view)
    assert node == "p:c", (node, wire)
    assert wire is None, "the wire rung fired under a node body"

    # (e) off the canvas: every field written, every one None.
    _park(app, (view.canvas_rect[0] - 40.0, view.canvas_rect[1] - 40.0))
    assert _hover_fields(view) == (None, None, None, None)
    _close_graph(app, document_id)


def test_the_hover_fields_are_exactly_the_four_that_are_written(app: Any) -> None:
    # S3: a fifth `hovered_` field nobody wires would read stale forever, and the frame loop
    # gives no sign of it. Falsifier: add one -- this row names it.
    names = {f.name for f in fields(GraphViewState) if f.name.startswith("hovered_")}
    assert names == {"hovered_node", "hovered_port", "hovered_out", "hovered_wire"}


def test_a_wire_selection_and_a_node_selection_are_exclusive(app: Any) -> None:
    # S4: one Delete, one unambiguous target. Falsifier: leave `selected_wire` alone in
    # `_click` and a node click leaves both selections live.
    document_id, document = _chain(app)
    view = _open_graph(app, document_id)
    _click_at(app, view.wire_mids[("c", "u_src")])
    assert view.selected_wire == ("c", "u_src")

    picture = pass_graph._build_view(document, "", {})
    node_a = picture.nodes["p:a"]
    center = (
        node_a.pos[0] + node_a.size[0] * 0.5,
        node_a.pos[1] + node_a.size[1] * 0.25,
    )
    _click_at(
        app,
        (
            view.canvas_rect[0] + (center[0] - view.pan[0]) * view.zoom,
            view.canvas_rect[1] + (center[1] - view.pan[1]) * view.zoom,
        ),
    )
    assert view.selected_wire is None
    assert view.selection == {"a"}

    # And a rubber band's release clears a wire selection too.
    _click_at(app, view.wire_mids[("c", "u_src")])
    assert view.selected_wire == ("c", "u_src")
    empty = (view.canvas_rect[2] - 8.0, view.canvas_rect[3] - 8.0)
    _park(app, empty)
    imgui.get_io().add_mouse_button_event(0, True)
    _frames(app, 2)
    _park(app, (empty[0] - 60.0, empty[1] - 60.0))
    imgui.get_io().add_mouse_button_event(0, False)
    _frames(app, 2)
    assert view.selected_wire is None
    _close_graph(app, document_id)


def _drag_node(app: Any, view: Any, start: tuple[float, float], dx: float) -> None:
    # The mouse is walked back to `start` with the button UP first: a second drag begun from
    # wherever the previous one ended would carry that offset into imgui's drag delta and the
    # two cases would not measure what they name.
    _park(app, start)
    imgui.get_io().add_mouse_button_event(0, True)
    _frames(app, 2)
    # The move and the release in SEPARATE frames: in one frame imgui resets the drag before
    # the frame runs and the pair reads as a click (measured).
    _park(app, (start[0] + dx, start[1]))
    imgui.get_io().add_mouse_button_event(0, False)
    _frames(app, 3)


def test_three_pixels_is_a_click_and_five_is_a_drag(app: Any) -> None:
    """G13/S15: the lock is what separates "choose this output" from "move this card", and a
    click must not evict the graph tab it was made on.

    Break to try: omit `lock_threshold` at the node-body site -- the 5px case falls back to
    imgui's 6px default and stays a click (measured today: 5px a click, 8px a drag).
    """
    document_id, document = _chain(app)
    view = _open_graph(app, document_id)
    picture = pass_graph._build_view(document, "", {})
    node_a = picture.nodes["p:a"]
    body = (
        node_a.pos[0] + node_a.size[0] * 0.5,
        node_a.pos[1] + node_a.size[1] * 0.2,
    )
    start = (
        view.canvas_rect[0] + (body[0] - view.pan[0]) * view.zoom,
        view.canvas_rect[1] + (body[1] - view.pan[1]) * view.zoom,
    )

    with (
        mock.patch.object(
            app.session, "set_output_pass", wraps=app.session.set_output_pass
        ) as output,
        mock.patch.object(
            app.session, "set_pass_positions", wraps=app.session.set_pass_positions
        ) as placed,
    ):
        _drag_node(app, view, start, 3.0)
    assert output.call_count == 1, "3px did not read as a click"
    assert placed.call_count == 0, "3px wrote a position"
    assert app.active_tab is not None and app.active_tab.kind == "graph", (
        "the click opened a shader tab and evicted the canvas it was made on"
    )

    # A fresh node, so the 3px case's own position write cannot decide this one.
    node_b = pass_graph._build_view(document, "", {}).nodes["p:b"]
    body_b = (
        node_b.pos[0] + node_b.size[0] * 0.5,
        node_b.pos[1] + node_b.size[1] * 0.2,
    )
    start_b = (
        view.canvas_rect[0] + (body_b[0] - view.pan[0]) * view.zoom,
        view.canvas_rect[1] + (body_b[1] - view.pan[1]) * view.zoom,
    )
    with (
        mock.patch.object(
            app.session, "set_output_pass", wraps=app.session.set_output_pass
        ) as output,
        mock.patch.object(
            app.session, "set_pass_positions", wraps=app.session.set_pass_positions
        ) as placed,
    ):
        _drag_node(app, view, start_b, 5.0)
    assert placed.call_count == 1, "5px did not read as a drag"
    assert output.call_count == 0, "5px chose an output as well"
    _close_graph(app, document_id)


def test_a_double_click_opens_the_passs_shader_tab(app: Any) -> None:
    # S15: the deliberate "open this pass" gesture keeps `pick_pass`, so the pane switches on
    # purpose rather than on every click. Falsifier: point `_double_click` at `choose_output`.
    document_id, document = _chain(app)
    view = _open_graph(app, document_id)
    picture = pass_graph._build_view(document, "", {})
    node_a = picture.nodes["p:a"]
    body = (
        node_a.pos[0] + node_a.size[0] * 0.5,
        node_a.pos[1] + node_a.size[1] * 0.2,
    )
    point = (
        view.canvas_rect[0] + (body[0] - view.pan[0]) * view.zoom,
        view.canvas_rect[1] + (body[1] - view.pan[1]) * view.zoom,
    )
    _park(app, point)
    for _ in range(2):
        imgui.get_io().add_mouse_button_event(0, True)
        _frames(app, 1)
        imgui.get_io().add_mouse_button_event(0, False)
        _frames(app, 1)
    _frames(app, 2)
    tab = app.active_tab
    assert tab is not None and tab.kind == "shader", tab
    assert tab.path == document.passes["a"].source.path


def test_the_selected_card_draws_and_hit_tests_last(app: Any) -> None:
    # S7: bring-to-front is one sorted list, used by the draw loop and the button loop alike,
    # so the card that paints on top is the one whose button wins the overlap. Falsifier: sort
    # only the draw loop -- the picture and the hit test disagree about which card is on top.
    document_id, _document = _chain(app)
    view = _open_graph(app, document_id)
    view.selection = {"a"}
    _frames(app, 2)
    assert view.node_order[-1] == "p:a", view.node_order
    _close_graph(app, document_id)


def test_the_card_in_flight_outranks_a_merely_selected_one(app: Any) -> None:
    # G12: the key is ascending, so its LAST component dominates -- with `is_selected` there, a
    # still card that happens to be selected drew and hit-tested over the card under the hand.
    # Falsifier: put `selected` last again and `node_order[-1]` is the selected `b`, not `a`.
    document_id, document = _chain(app)
    view = _open_graph(app, document_id)
    picture = pass_graph._build_view(document, "", {})
    node_a = picture.nodes["p:a"]
    body = (
        node_a.pos[0] + node_a.size[0] * 0.5,
        node_a.pos[1] + node_a.size[1] * 0.2,
    )
    start = (
        view.canvas_rect[0] + (body[0] - view.pan[0]) * view.zoom,
        view.canvas_rect[1] + (body[1] - view.pan[1]) * view.zoom,
    )
    # `b` selected and still, `a` under the hand: a real drag, held open across the assertion.
    view.selection = {"b"}
    _park(app, start)
    imgui.get_io().add_mouse_button_event(0, True)
    _frames(app, 2)
    _park(app, (start[0] + 30.0, start[1] + 10.0))
    assert view.node_drag is not None, "the drag never started"
    assert view.selection == {"b"}, view.selection
    assert view.node_order[-1] == "p:a", view.node_order
    imgui.get_io().add_mouse_button_event(0, False)
    _frames(app, 3)
    _close_graph(app, document_id)


def test_the_cursor_follows_the_gesture(app: Any) -> None:
    """G7: three gestures, three cursors, and nothing at rest.

    The widget REQUESTS into the single owner and `ui.py` applies once per frame on change; a
    raw `glfw.set_cursor` per surface flickers on X11. `want_cursor` is reset every frame
    after being applied, so what a test reads is `cur_cursor`, on a frame where the canvas
    provably drew. Falsifier: drop a request and its gesture leaves the arrow up.

    The pan is driven Alt+left rather than middle-drag: both take the same `panning` branch,
    and imgui does not activate an `invisible_button` on a synthetic middle press.
    """
    document_id, _document = _chain(app)
    view = _open_graph(app, document_id)
    assert view.canvas_rect != (0.0, 0.0, 0.0, 0.0)
    assert app.cur_cursor is None, "something requested a cursor at rest"
    io = imgui.get_io()
    center = (
        (view.canvas_rect[0] + view.canvas_rect[2]) / 2.0,
        (view.canvas_rect[1] + view.canvas_rect[3]) / 2.0,
    )

    _park(app, center)
    io.add_key_event(imgui.Key.mod_alt, True)
    _frames(app, 2)
    io.add_mouse_button_event(0, True)
    _frames(app, 2)
    _park(app, (center[0] + 30.0, center[1] + 20.0), frames=2)
    assert app.cur_cursor is app.hand_cursor, "a pan left the arrow up"
    io.add_mouse_button_event(0, False)
    io.add_key_event(imgui.Key.mod_alt, False)
    _frames(app, 3)
    assert view.canvas_rect != (0.0, 0.0, 0.0, 0.0)
    assert app.cur_cursor is None, "the cursor survived the gesture"

    # A wire in flight asks for the crosshair instead, driven as a user drives it: a drag off
    # an OUTPUT dot, the only end a wire can leave from (093 W2-4). A `wire_drag` merely
    # ASSIGNED would be cancelled at the top of the next frame, which is the guard that keeps
    # a gesture whose press the canvas never saw from writing.
    x0, y0, x1, y1 = view.out_rects[("p:a", 0)]
    _park(app, ((x0 + x1) / 2.0, (y0 + y1) / 2.0))
    io.add_mouse_button_event(0, True)
    _frames(app, 2)
    _park(app, ((x0 + x1) / 2.0 + 60.0, (y0 + y1) / 2.0 + 40.0), frames=2)
    assert view.wire_drag is not None, "the drag never started"
    assert app.cur_cursor is app.crosshair_cursor
    io.add_mouse_button_event(0, False)
    _frames(app, 3)
    assert app.cur_cursor is None
    _close_graph(app, document_id)


def test_an_input_pin_moves_the_node_and_never_carries_its_wire(app: Any) -> None:
    """W2-4: a press on a FILLED input port is a node drag, not a wire grab.

    The gesture the references converge on -- drag the wire off its input, a ghost following
    the cursor, release to detach -- is the one the maintainer rejected: the ghost does not say
    that letting go removes the read. So the pin is not a drag source at all, and a wire leaves
    only by its own ✕ or the Delete key.

    Falsifier: restore the `port.kind == "wired"` branch -- `wire_drag` is set instead of
    `node_drag`, and the release on empty canvas unwires `c.u_src`.
    """
    document_id, document = _chain(app)
    view = _open_graph(app, document_id)
    x0, y0, x1, y1 = view.port_rects[("c", "u_src")]
    start = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    with mock.patch.object(
        app.session, "set_sampler_source", wraps=app.session.set_sampler_source
    ) as write:
        _park(app, start)
        imgui.get_io().add_mouse_button_event(0, True)
        _frames(app, 2)
        _park(app, (start[0] + 6.0, start[1] + 6.0))
        assert view.node_drag is not None, "the press did not move the node"
        assert view.wire_drag is None, "the input pin carried its wire away"
        imgui.get_io().add_mouse_button_event(0, False)
        _frames(app, 3)
    assert write.call_count == 0, [c.args for c in write.call_args_list]
    assert document.passes["c"].uniform_values["u_src"] == PassSource("b")
    _close_graph(app, document_id)


def test_a_wire_dropped_on_a_filled_port_overwrites_its_source(app: Any) -> None:
    # W2-4: with the pin no longer a drag source, replacing a read is a drop from an output
    # onto the filled port -- `drop_wire` is the one write and it overwrites. Falsifier: refuse
    # a drop on a filled port and a wire could only ever be replaced by unwiring it first.
    document_id, document = _chain(app)
    view = _open_graph(app, document_id)
    assert document.passes["c"].uniform_values["u_src"] == PassSource("b")
    x0, y0, x1, y1 = view.out_rects[("p:a", 0)]
    _park(app, ((x0 + x1) / 2.0, (y0 + y1) / 2.0))
    imgui.get_io().add_mouse_button_event(0, True)
    _frames(app, 2)
    px0, py0, px1, py1 = view.port_rects[("c", "u_src")]
    _park(app, ((px0 + px1) / 2.0, (py0 + py1) / 2.0))
    assert view.wire_drag is not None, "the drag never started"
    imgui.get_io().add_mouse_button_event(0, False)
    _frames(app, 3)
    assert document.passes["c"].uniform_values["u_src"] == PassSource("a")
    _close_graph(app, document_id)
