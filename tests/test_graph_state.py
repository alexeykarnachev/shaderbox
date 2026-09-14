"""The graph canvas's pure gesture pieces (092 D3, D13; 093 S9): the drag writes only on
commit, a scope no pass carries falls back to the root, and the wire's whole geometry --
the cusp-proof curve, the hit test, the state precedence -- is decided without imgui."""

import ast
import inspect
import itertools
import math
from pathlib import Path

from shaderbox.theme import SIZE
from shaderbox.widgets import pass_graph
from shaderbox.widgets.graph_state import (
    NodeDrag,
    Position,
    WireState,
    bezier_point,
    delete_allowed,
    group_names_in_order,
    node_size,
    revalidated_scope,
    revalidated_wire,
    wire_hit,
    wire_hit_threshold,
    wire_points,
    wire_state,
)


def test_a_drag_writes_nothing_until_commit_and_then_every_moved_name_once() -> None:
    # Falsifier: return positions from `update` (a per-frame write, sixty saves a second).
    drag = NodeDrag(origin={"a": (0.0, 0.0), "b": (10.0, 5.0)})
    assert drag.update(3.0, 4.0) is None
    assert drag.update(1.0, 1.0) is None
    committed = drag.commit()
    assert committed == {"a": (4.0, 5.0), "b": (14.0, 10.0)}
    assert list(committed) == ["a", "b"]


def test_the_snap_offset_rides_on_top_of_the_raw_delta_and_never_corrects_it() -> None:
    # Falsifier: fold the snap into `delta` -- a node held at a guide then absorbs every
    # later mouse move and never leaves it (the post-implementation review's blocker).
    drag = NodeDrag(origin={"a": (0.0, 0.0)})
    drag.update(10.0, 0.0)
    drag.snap = (-3.0, 0.0)
    assert drag.raw()["a"] == (10.0, 0.0)
    assert drag.current()["a"] == (7.0, 0.0)
    assert drag.commit()["a"] == (7.0, 0.0)
    drag.update(10.0, 0.0)
    drag.snap = (0.0, 0.0)
    assert drag.current()["a"] == (20.0, 0.0)


def test_the_moving_picture_reads_the_same_positions_the_commit_writes() -> None:
    drag = NodeDrag(origin={"a": (2.0, 2.0)})
    drag.update(-2.0, 0.5)
    assert drag.current() == drag.commit()


def test_a_scope_no_pass_carries_falls_back_to_the_root() -> None:
    assert revalidated_scope("bloom", {"bloom", "fx"}) == "bloom"
    assert revalidated_scope("gone", {"bloom"}) == ""
    assert revalidated_scope("", {"bloom"}) == ""


def test_group_names_follow_their_first_members_order() -> None:
    order = ["scene", "b1", "g1", "b2", "final"]
    groups = {"b1": "bloom", "b2": "bloom", "g1": "grade"}
    assert group_names_in_order(order, groups) == ["bloom", "grade"]


def test_a_node_grows_one_row_per_port_and_a_box_is_wider() -> None:
    w0, h0 = node_size(0, False)
    w2, h2 = node_size(2, False)
    assert w0 == w2 == float(SIZE.GRAPH_NODE_W)
    assert h2 == h0 + 4.0 + 2 * SIZE.GRAPH_PORT_ROW
    assert node_size(0, True)[0] == w0 + SIZE.GRAPH_BOX_EXTRA_W


# ---- the wire (093) -------------------------------------------------------------------------


def _true_point(
    points: tuple[Position, Position, Position, Position], t: float
) -> Position:
    return bezier_point(*points, t)


def test_no_pair_of_endpoints_folds_the_curve() -> None:
    # G1: a fold needs `2 * offset <= dx`, and the offset is a max of two non-negative terms,
    # so it is unreachable while dx < 0. Falsifier: sign the offset by `dx` (xyflow's shape) --
    # the backward cases stop producing `cp0.x > cp1.x` and the S-curve becomes a cusp.
    runs = [-800.0, -320.0, -48.0, -1.0, 0.0, 1.0, 48.0, 320.0, 800.0]
    for zoom in (0.25, 1.0, 2.5):
        for dx, dy in itertools.product(runs, runs):
            p0, cp0, cp1, p3 = wire_points((0.0, 0.0), (dx, dy), zoom)
            offset = cp0[0] - p0[0]
            assert offset >= 0.0, (dx, dy, zoom, offset)
            assert math.isclose(p3[0] - cp1[0], offset), (dx, dy, zoom)
            if dx < 0.0:
                assert cp0[0] > cp1[0], (dx, dy, zoom)


def test_the_curve_is_continuous_across_the_bus_boundary_it_replaced() -> None:
    # G1: the old bus switched TOPOLOGY at a threshold, so a wire jumped tens of pixels as its
    # producer crossed it. One formula has no boundary. The bound is the endpoint's own 1px
    # step plus what the offset may move with it. Falsifier: restore a `if dx < k` branch.
    bound = 1.0 + 2.0 * SIZE.GRAPH_WIRE_BOW
    previous = wire_points((0.0, 0.0), (-48.0, 40.0), 1.0)
    worst = 0.0
    for dx in range(-47, 49):
        current = wire_points((0.0, 0.0), (float(dx), 40.0), 1.0)
        for before, after in zip(previous, current, strict=True):
            worst = max(worst, math.dist(before, after))
        previous = current
    assert worst < bound, f"a control point jumped {worst}px in one pixel of dx"


def test_the_hit_threshold_is_floored_in_screen_pixels() -> None:
    # G4: the drawn stroke has a 1px screen floor, so its reach needs one too -- a 1.5px reach
    # for a 1px line is not clickable. Break to try: drop the `max(GRAPH_WIRE_HIT_FLOOR, ...)`
    # and `wire_hit_threshold(0.25)` reads 0.75.
    assert wire_hit_threshold(0.25) == 6.0
    assert wire_hit_threshold(1.0) == 6.0
    assert wire_hit_threshold(2.5) == 7.5
    # And it stays under the port's own floor, so a port always outranks a wire on overlap.
    assert SIZE.GRAPH_WIRE_HIT_FLOOR < SIZE.GRAPH_HIT_MIN


def test_the_hit_test_answers_the_distance_under_the_threshold_and_none_over_it() -> (
    None
):
    # G4. Falsifier: compare against the segment ENDPOINTS rather than the segments, and a
    # point beside a long span reads far away.
    points = wire_points((0.0, 0.0), (-400.0, 120.0), 1.0)
    threshold = wire_hit_threshold(1.0)
    segs = SIZE.GRAPH_WIRE_HIT_SEGS
    on_curve = _true_point(points, 0.5)
    distance = wire_hit(on_curve, points, threshold, segs)
    assert distance is not None and distance < 0.5
    # Straight DOWN from the endpoint, perpendicular to the horizontal tangent there and
    # clear of the S's other arm, so the step from a hit to a miss is the threshold alone.
    tip = _true_point(points, 1.0)
    near = (tip[0], tip[1] + threshold - 1.0)
    assert wire_hit(near, points, threshold, segs) is not None
    far = (tip[0], tip[1] + threshold + 1.0)
    assert wire_hit(far, points, threshold, segs) is None
    # The bounding-box reject: a point nowhere near the curve costs one rect test.
    assert wire_hit((5000.0, 5000.0), points, threshold, segs) is None


def test_the_flattening_misses_no_real_hit_on_a_long_backward_curve() -> None:
    # A regression guard, not a pin on 24: it discriminates a count of 6 or below (worst error
    # 9.5px there against the 6px threshold), so it catches a catastrophic count.
    points = wire_points((0.0, 0.0), (-400.0, 120.0), 1.0)
    threshold = wire_hit_threshold(1.0)
    segs = SIZE.GRAPH_WIRE_HIT_SEGS
    for i in range(200):
        on_curve = _true_point(points, i / 199.0)
        assert wire_hit(on_curve, points, threshold, segs) is not None, i


def test_an_error_wire_stays_red_however_it_is_touched() -> None:
    # G18: `node.error` already outranks `selected` on a border; a wire follows the same chain.
    # Falsifier: put `selected` first -- the first row below flips.
    for on_cycle, selected, hovered, dim in itertools.product((False, True), repeat=4):
        state = wire_state(on_cycle, selected, hovered, dim)
        if on_cycle:
            assert state is WireState.ERROR
        elif selected:
            assert state is WireState.SELECTED
        elif hovered:
            assert state is WireState.HOVERED
        elif dim:
            assert state is WireState.DIM
        else:
            assert state is WireState.NORMAL


def test_a_wire_selection_no_drawn_edge_carries_clears() -> None:
    # S2. Falsifier: keep the selection across a rebuild -- an unwired pair would stay the
    # Delete key's target and the next press would write over whatever took its place.
    drawn = {("c", "u_src"), ("b", "u_src")}
    assert revalidated_wire(("c", "u_src"), drawn) == ("c", "u_src")
    assert revalidated_wire(("c", "u_other"), drawn) is None
    assert revalidated_wire(None, drawn) is None


def test_the_delete_gate_is_true_on_exactly_one_of_its_thirty_two_states() -> None:
    # S5, over the whole domain. The `any_item_active` clause's live scenario (a text input
    # active in another window while the mouse rests over the canvas) is the maintainer's
    # check -- the fixture draws no such input. Falsifier: drop any one clause.
    for combo in itertools.product((False, True), repeat=5):
        pressed, hovered, active, blocked, has_wire = combo
        assert delete_allowed(*combo) == (
            pressed and hovered and not active and not blocked and has_wire
        ), combo


def test_nothing_on_the_canvas_changes_size_on_hover() -> None:
    # G6: a highlight changes COLOR, never size (/imgui-ui §3). A node's size is a function of
    # its port count alone and a wire's reach of the zoom alone, so neither can grow on hover.
    assert list(inspect.signature(node_size).parameters) == ["port_count", "box"]
    assert list(inspect.signature(wire_hit_threshold).parameters) == ["zoom"]


def _widget_source() -> str:
    return Path(pass_graph.__file__).read_text(encoding="utf-8")


def test_the_loop_the_bus_and_the_module_constants_are_gone() -> None:
    # G3, G8, S12: each of these was a token or a constant the redesign removed, and a
    # leftover reader is what would quietly keep the old geometry alive.
    for token in (
        "GRAPH_LOOP_RISE",
        "GRAPH_LOOP_REACH",
        "GRAPH_BUS_STEP",
        "GRAPH_BUS_CLEAR",
        "GRAPH_MIN_H",
    ):
        assert not hasattr(SIZE, token), token
    source = _widget_source()
    for symbol in ("_draw_self_loop", "bus_y", "_MIN_DIRECT_DX", "_BEZIER_BOW"):
        assert symbol not in source, symbol
    # G7: the widget REQUESTS a cursor into the single owner; it never pokes glfw itself.
    assert "glfw" not in source


def _drag_call_sites() -> list[tuple[str, int, list[str]]]:
    """Every `is_mouse_dragging` / `get_mouse_drag_delta` CALL in the widget, with the source
    of each argument.

    Walked as an `ast`, not scanned as text: a text window around the call is satisfied by a
    comment that merely names the token, so the gate would pass on a site that passes nothing.
    """
    tree = ast.parse(_widget_source())
    sites: list[tuple[str, int, list[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else None
        if name not in ("is_mouse_dragging", "get_mouse_drag_delta"):
            continue
        arguments = [ast.unparse(a) for a in node.args]
        arguments += [ast.unparse(k.value) for k in node.keywords]
        sites.append((name, node.lineno, arguments))
    return sites


def test_the_drag_lock_is_passed_at_every_site() -> None:
    # S13: imgui's global 6px default is tuned for buttons, and one site left on it is a
    # gesture that behaves unlike its four neighbors. Falsifier: omit it anywhere -- or write
    # it in a COMMENT beside a bare call, which the text-window version of this gate accepted.
    sites = _drag_call_sites()
    assert len(sites) == 5, [(n, ln) for n, ln, _ in sites]
    for name, lineno, arguments in sites:
        assert any("GRAPH_DRAG_LOCK_PX" in a for a in arguments), (
            name,
            lineno,
            arguments,
        )


def test_the_five_channels_carry_the_layers_in_order() -> None:
    """G12: paint order follows the CHANNEL INDEX rather than the call order, so the index a
    layer is assigned is the whole layering decision -- halos under strokes under nodes under
    the wire in flight under the overlays.

    Falsifier: swap two of the five constants and the pairs below flip. The split and the
    merge are counted because an unmatched either leaves the canvas' draw list broken.
    """
    source = _widget_source()
    assert source.count("channels_split(5)") == 1
    assert source.count("channels_merge()") == 1
    layers = [
        pass_graph._CH_HALO,
        pass_graph._CH_WIRE,
        pass_graph._CH_NODE,
        pass_graph._CH_INFLIGHT,
        pass_graph._CH_OVERLAY,
    ]
    assert layers == [0, 1, 2, 3, 4], layers
    # Every channel is actually used, and nothing reaches for a sixth the split never made.
    for name in ("HALO", "WIRE", "NODE", "INFLIGHT", "OVERLAY"):
        assert f"channels_set_current(_CH_{name})" in source, name
    assert "channels_set_current(5" not in source
