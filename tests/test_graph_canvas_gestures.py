"""The canvas's gestures, driven through the library headlessly (feature 098).

The old canvas published its hit rects on the view state and the tests aimed
at those. The library owns the hit-testing now, so a test aims by ASKING it:
`node_rects[i].hover_attribute` names the port under the pointer and
`flags` bit 0 marks the hovered node. Probing for the point is not a
convenience — the pin's grab area straddles the node's edge, so a point
computed from the rect alone lands on the body and starts a node drag, which
looks exactly like the wire gesture not existing.

These drive `gc_frame` directly rather than through imgui, which is what makes
them run with no window while still exercising the real gesture machine.
"""

import pytest

from shaderbox.graph_canvas import ffi
from shaderbox.graph_canvas.adapter import (
    Activated,
    BodyRow,
    Clicked,
    Moved,
    Packed,
    PickerRequested,
    Unwired,
    ValueEdited,
    Wired,
    flat_view,
    pack_nodes,
    pass_key,
    read_events,
)
from shaderbox.graph_canvas.panel import pointer_is_claimed
from shaderbox.pass_graph import Port

DOWN = int(ffi.Pointer.DOWN)
PRESSED = int(ffi.Pointer.PRESSED)
DOUBLE = int(ffi.Pointer.DOUBLE)
EXTEND = int(ffi.Pointer.EXTEND)
SIZE = (900.0, 640.0)


def _canvas() -> ffi.Canvas:
    canvas = ffi.Canvas()
    canvas.load_atlas()
    return canvas


def _chain(wired: bool = False) -> Packed:
    """`a -> b`, with `b`'s one sampler either reading `a` or reading nothing."""
    port = Port("u_src", "wired", "a") if wired else Port("u_src", "unfilled")
    return pack_nodes(
        flat_view(
            ["a", "b"], {"a": [], "b": [port]}, {"a": (0.0, 0.0), "b": (300.0, 0.0)}
        ),
        {},
        output=pass_key("b"),
    )


def _rect(
    canvas: ffi.Canvas, packed: Packed, node: int
) -> tuple[float, float, float, float]:
    """One node's screen rect, COPIED OUT: everything in the result points into
    the handle's own storage and the next frame overwrites all of it."""
    result = canvas.frame(
        packed.nodes, packed.edges, SIZE, ffi.View(), ffi.PointerState()
    )
    box = result.node_rects[node]
    return (box.x, box.y, box.w, box.h)


def _pin(
    canvas: ffi.Canvas, packed: Packed, node: int, attribute: int, output: bool
) -> tuple[float, float]:
    """One pin's point, from the library (`gc_pin_point`).

    This replaced three brute-force probes that pressed candidate points until
    one produced the event. They existed because a pin is not where its rect
    or its hover band says: the grab straddles the node's edge and the hover
    answers across the whole row, so a point computed from either lands on the
    body and starts a node drag.
    """
    canvas.frame(packed.nodes, packed.edges, SIZE, ffi.View(), ffi.PointerState())
    point = canvas.pin_point(node, attribute, output=output)
    assert point is not None, f"node {node} has no attribute {attribute}"
    return point


def _drive(
    canvas: ffi.Canvas, packed: Packed, points: list[ffi.PointerState]
) -> list[object]:
    """Push a gesture frame by frame, carrying the view the library returns."""
    view = ffi.View()
    events: list[object] = []
    for pointer in points:
        result = canvas.frame(packed.nodes, packed.edges, SIZE, view, pointer)
        view = ffi.View(result.pan_x, result.pan_y, result.zoom)
        events.extend(read_events(result, packed))
    return events


def test_a_wire_dragged_from_an_output_onto_an_input_lands_there() -> None:
    canvas = _canvas()
    packed = _chain()
    source = _pin(canvas, packed, 0, 0, output=True)
    target = _pin(canvas, packed, 1, 0, output=False)

    events = _drive(
        canvas,
        packed,
        [
            ffi.PointerState(x=source[0], y=source[1], flags=DOWN | PRESSED),
            ffi.PointerState(x=(source[0] + target[0]) / 2, y=source[1], flags=DOWN),
            ffi.PointerState(x=target[0], y=target[1], flags=DOWN),
            ffi.PointerState(x=target[0], y=target[1], flags=0),
        ],
    )
    assert Wired("a", "b", "u_src") in events
    canvas.release()


def test_a_wire_dropped_on_empty_canvas_writes_nothing() -> None:
    """A drag abandoned over the background is a no-op, not a refusal."""
    canvas = _canvas()
    packed = _chain()
    source = _pin(canvas, packed, 0, 0, output=True)

    events = _drive(
        canvas,
        packed,
        [
            ffi.PointerState(x=source[0], y=source[1], flags=DOWN | PRESSED),
            ffi.PointerState(x=source[0], y=600.0, flags=DOWN),
            ffi.PointerState(x=source[0], y=600.0, flags=0),
        ],
    )
    assert not any(isinstance(e, Wired) for e in events)
    canvas.release()


def test_grabbing_a_connected_input_reports_the_wire_gone_on_the_press() -> None:
    """`Edge_Removed` fires on the PRESS, not the release: a drag abandoned
    over empty canvas never reaches a release that would mention it, so a host
    hearing about it later would keep a wire the library has already dropped."""
    canvas = _canvas()
    packed = _chain(wired=True)
    target = _pin(canvas, packed, 1, 0, output=False)

    events = _drive(
        canvas,
        packed,
        [ffi.PointerState(x=target[0], y=target[1], flags=DOWN | PRESSED)],
    )
    assert Unwired("b", "u_src") in events
    canvas.release()


def test_the_active_flag_marks_the_pressed_node_and_not_a_wire_end() -> None:
    """`NODE_RECT_ACTIVE` answers "this node is in play", never "these are
    the wire's endpoints" -- and a RE-DRAG is the case that separates them.

    Grabbing a connected input picks the existing wire up: the anchor moves
    to the far output and the loose end follows the pointer, while the flag
    stays on the input that was pressed -- a node the dragged wire no longer
    touches. Measured: press b's input, flag on b, anchor on a.

    The fresh-drag case cannot see this. There the pressed node IS the
    wire's end, so both readings agree and a fixture with only that case
    passes a library that swapped one for the other. The endpoints come from
    the `Edge_Removed` the press emits, which the test above pins.
    """
    canvas = _canvas()
    packed = _chain(wired=True)
    target = _pin(canvas, packed, 1, 0, output=False)

    result = canvas.frame(
        packed.nodes,
        packed.edges,
        SIZE,
        ffi.View(),
        ffi.PointerState(x=target[0], y=target[1], flags=DOWN | PRESSED),
    )
    active = {
        packed.name_of(i)
        for i in range(result.rect_count)
        if result.node_rects[i].flags & ffi.NODE_RECT_ACTIVE
    }
    assert active == {"b"}, f"the pressed node is not the one marked active: {active}"
    canvas.release()


def test_a_press_on_the_body_moves_the_node_and_reports_canvas_space() -> None:
    canvas = _canvas()
    packed = _chain()
    x0, y0, w, h = _rect(canvas, packed, 0)
    centre = (x0 + w / 2, y0 + h / 3)

    events = _drive(
        canvas,
        packed,
        [
            ffi.PointerState(x=centre[0], y=centre[1], flags=DOWN | PRESSED),
            ffi.PointerState(x=centre[0] + 60.0, y=centre[1] + 30.0, flags=DOWN),
            ffi.PointerState(x=centre[0] + 60.0, y=centre[1] + 30.0, flags=0),
        ],
    )
    moves = [e for e in events if isinstance(e, Moved)]
    assert moves and moves[-1].name == "a"
    # Canvas space at zoom 1 with no pan: the node followed the pointer.
    assert moves[-1].x == pytest.approx(60.0, abs=1.0)
    assert moves[-1].y == pytest.approx(30.0, abs=1.0)
    canvas.release()


def test_a_press_and_release_without_travel_is_a_click() -> None:
    canvas = _canvas()
    packed = _chain()
    x0, y0, w, h = _rect(canvas, packed, 1)
    point = (x0 + w / 2, y0 + h / 3)

    events = _drive(
        canvas,
        packed,
        [
            ffi.PointerState(x=point[0], y=point[1], flags=DOWN | PRESSED),
            ffi.PointerState(x=point[0], y=point[1], flags=0),
        ],
    )
    assert Clicked(pass_key("b"), "b", False) in events
    canvas.release()


def test_a_click_carrying_extend_says_so() -> None:
    """Shift is how a selection grows, and the library reports the modifier
    from the PRESS rather than from the frame the release happened on."""
    canvas = _canvas()
    packed = _chain()
    x0, y0, w, h = _rect(canvas, packed, 1)
    point = (x0 + w / 2, y0 + h / 3)

    events = _drive(
        canvas,
        packed,
        [
            ffi.PointerState(x=point[0], y=point[1], flags=DOWN | PRESSED | EXTEND),
            ffi.PointerState(x=point[0], y=point[1], flags=EXTEND),
        ],
    )
    assert Clicked(pass_key("b"), "b", True) in events
    canvas.release()


def test_a_second_click_inside_the_hosts_own_interval_activates() -> None:
    """The library does not time the double click: the interval belongs to the
    platform, and one timed here would disagree with every other double click
    on the machine. The host says so with a flag."""
    canvas = _canvas()
    packed = _chain()
    x0, y0, w, h = _rect(canvas, packed, 1)
    point = (x0 + w / 2, y0 + h / 3)

    events = _drive(
        canvas,
        packed,
        [
            ffi.PointerState(x=point[0], y=point[1], flags=DOWN | PRESSED),
            ffi.PointerState(x=point[0], y=point[1], flags=0),
            ffi.PointerState(x=point[0], y=point[1], flags=DOWN | PRESSED | DOUBLE),
            ffi.PointerState(x=point[0], y=point[1], flags=DOUBLE),
        ],
    )
    assert Activated(pass_key("b"), "b") in events
    canvas.release()


def test_a_ghost_swallows_nothing() -> None:
    """A node that refuses every gesture must let the press fall through,
    which is the difference between a node that is not interactive and a
    hole in the canvas.

    Read from the LIBRARY's raw events, not from `read_events`. The adapter
    also drops a ghost's events -- belt and braces its own docstring calls
    not a live path -- so a test reading the resolved list passes with
    `accepts` set to 0, the exact flag this names. Measured: with the flag
    dropped the library emits `NODE_MOVED` and the adapter guard hides it.
    """
    canvas = _canvas()
    packed = pack_nodes(
        flat_view(
            ["a", "b"],
            {"a": [], "b": [Port("u_src", "unfilled")]},
            {"a": (0.0, 0.0), "b": (300.0, 0.0)},
            frozenset({"a"}),
        ),
        {},
        output=pass_key("b"),
    )
    x0, y0, w, h = _rect(canvas, packed, 0)
    point = (x0 + w / 2, y0 + h / 3)

    kinds: list[int] = []
    for pointer in (
        ffi.PointerState(x=point[0], y=point[1], flags=DOWN | PRESSED),
        ffi.PointerState(x=point[0] + 60.0, y=point[1], flags=DOWN),
        ffi.PointerState(x=point[0] + 60.0, y=point[1], flags=0),
    ):
        result = canvas.frame(packed.nodes, packed.edges, SIZE, ffi.View(), pointer)
        kinds += [
            int(result.events[i].kind)
            for i in range(result.event_count)
            if result.events[i].node == 0
        ]
    assert not kinds, f"the library acted on a ghost: event kinds {kinds} on node 0"
    canvas.release()


def test_the_pointer_reads_as_claimed_over_a_node_and_free_over_the_background() -> (
    None
):
    """How a host knows not to treat a press as its own.

    `Result.flags` bit 0 alone does NOT answer this: it is measured low while
    the pointer sits on a node body, because it is computed from the hover's
    port field and a body hover leaves that unset. `pointer_is_claimed` reads
    the per-node flags too. The falsifier is the body hover, so that is what
    this aims at -- a point that also hits a port passes either way.
    """
    canvas = _canvas()
    packed = _chain()
    x0, y0, w, h = _rect(canvas, packed, 0)

    over = canvas.frame(
        packed.nodes,
        packed.edges,
        SIZE,
        ffi.View(),
        ffi.PointerState(x=x0 + w / 2, y=y0 + h / 3),
    )
    assert over.node_rects[0].hover_attribute == -1, "aim at the body, not a port"
    assert pointer_is_claimed(over)

    away = canvas.frame(
        packed.nodes, packed.edges, SIZE, ffi.View(), ffi.PointerState(x=10.0, y=620.0)
    )
    assert not pointer_is_claimed(away)
    canvas.release()


def test_a_wheel_notch_zooms_about_the_pointer() -> None:
    """The library owns the view; the host stores what comes back."""
    canvas = _canvas()
    packed = _chain()
    # The result points into the handle's own storage and the NEXT frame
    # overwrites all of it, so the value is copied out before the next call
    # rather than compared against a live pointer.
    first = canvas.frame(
        packed.nodes, packed.edges, SIZE, ffi.View(), ffi.PointerState()
    )
    before = ffi.View(first.pan_x, first.pan_y, first.zoom)
    second = canvas.frame(
        packed.nodes,
        packed.edges,
        SIZE,
        before,
        ffi.PointerState(x=450.0, y=320.0, wheel=1.0),
    )
    assert second.zoom > before.zoom
    canvas.release()


def test_each_pin_on_a_node_gets_its_own_point_in_pushed_order() -> None:
    """The falsifier for `pin_point` ignoring the attribute index.

    A GESTURE cannot decide this and it is worth being explicit about why: a
    wire only has to start on some pin of the right node, so an ignored index
    and a shifted one both connect and both report the wire landing. The
    gesture proves a point is ON a pin; it does not prove WHICH pin, and the
    first claim looks exactly like the second.

    What decides it is walking every pin and requiring distinct points in the
    order they were pushed. Ignoring the index collapses them to one point;
    shifting by one collapses them to n-1 and reorders the rest.
    """
    canvas = _canvas()
    labels = ["u_x", "u_y", "u_z"]
    packed = pack_nodes(
        flat_view(
            ["a", "b"],
            {"a": [], "b": [Port(label, "unfilled") for label in labels]},
            {"a": (0.0, 0.0), "b": (300.0, 0.0)},
        ),
        {},
        output=pass_key("b"),
    )
    canvas.frame(packed.nodes, packed.edges, SIZE, ffi.View(), ffi.PointerState())

    points = [canvas.pin_point(1, index, output=False) for index in range(len(labels))]
    assert all(p is not None for p in points), points
    assert len(set(points)) == len(labels), f"pins share a point: {points}"
    # In pushed order, down the node's input edge.
    ys = [p[1] for p in points if p is not None]
    assert ys == sorted(ys), f"pins are not in pushed order: {points}"
    # The output is on the far edge, past the inputs in the same flat array.
    out = canvas.pin_point(1, len(labels), output=True)
    assert out is not None and out[0] > points[0][0]
    assert canvas.pin_point(1, len(labels) + 1, output=False) is None
    canvas.release()


def test_a_wire_lands_on_the_sampler_it_was_aimed_at_not_the_first_one() -> None:
    """End to end, with a consumer that has THREE inputs.

    The single-input fixture cannot see this: aim, resolve and report all
    agree on index 0 whatever the code does with the index. Aiming at the
    third sampler and requiring its NAME back exercises the whole chain --
    `pin_point`'s index, the library's hit test, and the adapter's mapping
    from `to_attr` back to a sampler.
    """
    canvas = _canvas()
    labels = ["u_x", "u_y", "u_z"]
    packed = pack_nodes(
        flat_view(
            ["a", "b"],
            {"a": [], "b": [Port(label, "unfilled") for label in labels]},
            {"a": (0.0, 0.0), "b": (300.0, 0.0)},
        ),
        {},
        output=pass_key("b"),
    )
    source = _pin(canvas, packed, 0, 0, output=True)
    target = _pin(canvas, packed, 1, 2, output=False)

    events = _drive(
        canvas,
        packed,
        [
            ffi.PointerState(x=source[0], y=source[1], flags=DOWN | PRESSED),
            ffi.PointerState(x=(source[0] + target[0]) / 2, y=source[1], flags=DOWN),
            ffi.PointerState(x=target[0], y=target[1], flags=DOWN),
            ffi.PointerState(x=target[0], y=target[1], flags=0),
        ],
    )
    assert Wired("a", "b", "u_z") in events, events
    canvas.release()


def _drag(point: tuple[float, float]) -> list[ffi.PointerState]:
    """A press, two frames of travel, a release."""
    return [
        ffi.PointerState(x=point[0], y=point[1], flags=DOWN | PRESSED),
        ffi.PointerState(x=point[0] + 18, y=point[1], flags=DOWN),
        ffi.PointerState(x=point[0] + 36, y=point[1], flags=DOWN),
        ffi.PointerState(x=point[0] + 36, y=point[1], flags=0),
    ]


def _point_dragging(packed: Packed, node: int) -> tuple[float, float]:
    """A point where a drag moves a DRAG field's value.

    `ValueEdited` arrives on the frame the value moves, so the whole drag
    is read rather than only what follows the release.
    """
    canvas = _canvas()
    try:
        box = _rect(canvas, packed, node)
        x = box[0] + box[2] * 0.5
        for step in range(0, int(box[3]), 4):
            y = box[1] + step + 0.5
            canvas.frame(
                packed.nodes, packed.edges, SIZE, ffi.View(), ffi.PointerState()
            )
            if any(
                isinstance(e, ValueEdited)
                for e in _drive(canvas, packed, _drag((x, y)))
            ):
                return (x, y)
    finally:
        canvas.release()
    raise AssertionError(f"no point on node {node} drags a field")


def _click(point: tuple[float, float]) -> list[ffi.PointerState]:
    """A full click. The press alone produces nothing: the library answers a
    widget on the RELEASE, so a down-only fixture reports no event and reads
    exactly like the feature being absent."""
    return [
        ffi.PointerState(x=point[0], y=point[1], flags=DOWN | PRESSED),
        ffi.PointerState(x=point[0], y=point[1], flags=0),
    ]


def _swatch_node() -> Packed:
    """One node with a colour row and a drag row, so a press can be shown to
    fire for the swatch and not for every widget."""
    return pack_nodes(
        flat_view(["p"], {"p": []}, {"p": (0.0, 0.0)}),
        {},
        output="",
        body={
            "p": [
                BodyRow(
                    "u_line_color", (1.0, 0.5, 0.25), None, editable=True, swatch=True
                ),
                BodyRow("u_gain", (0.5,), None, editable=True),
            ]
        },
    )


def _point_hitting(packed: Packed, node: int, want: type) -> tuple[float, float]:
    """A point where a full click on `node` produces `want`.

    Probed by DRIVING the gesture rather than by reading
    `hover_attribute`, which answers across the whole row while a widget's
    hit rect is inset -- the row's reported point is a pixel and a half
    above the swatch, so aiming by it lands on the body and reports a node
    click, reading exactly like the event not existing.

    One canvas with an IDLE frame between candidates. A gesture left in
    flight would turn the next press into a drag and move the node under
    the probe; a canvas per candidate also prevents that, and costs 32ms
    each -- most of what this test used to spend.
    """
    canvas = _canvas()
    try:
        box = _rect(canvas, packed, node)
        x = box[0] + box[2] * 0.5
        for step in range(0, int(box[3]), 4):
            y = box[1] + step + 0.5
            canvas.frame(
                packed.nodes, packed.edges, SIZE, ffi.View(), ffi.PointerState()
            )
            if any(isinstance(e, want) for e in _drive(canvas, packed, _click((x, y)))):
                return (x, y)
    finally:
        canvas.release()
    raise AssertionError(f"no point on node {node} produces {want.__name__}")


def test_pressing_a_colour_swatch_asks_the_host_for_a_picker() -> None:
    """The library draws the swatch and reports the press; the picker is the
    host's. Without the event a colour row is a swatch that cannot be
    edited, which is worse than the drag fields it replaced.

    The DRAG row is the comparison: if the event fired for any press, both
    rows would produce one and the check would pass on code that cannot
    tell a swatch from a field.
    """
    packed = _swatch_node()
    swatch = _point_hitting(packed, 0, PickerRequested)
    # Probed the same way rather than offset from the swatch. An offset
    # picked off one observed layout landed on the node BODY, where a
    # click yields nothing and the silence below read as "correctly
    # silent" -- so that half of this gate passed without ever reaching a
    # drag field. The drag is what proves contact: a point that yields
    # `ValueEdited` is inside the widget, and nothing else is.
    field = _point_dragging(packed, 0)

    asked = [
        e
        for e in _drive(_canvas(), packed, _click(swatch))
        if isinstance(e, PickerRequested)
    ]
    assert len(asked) == 1, f"a press on the swatch asked for no picker: {asked}"
    assert asked[0].pass_name == "p" and asked[0].uniform == "u_line_color", (
        f"the picker was asked for the wrong row: {asked[0]}"
    )

    # Silence is the assertion here, so the point has to be shown to have
    # ARRIVED: an empty result otherwise means "correctly silent" and
    # "never landed on it" equally well.
    on_field = _drive(_canvas(), packed, _click(field))
    assert not [e for e in on_field if isinstance(e, PickerRequested)], (
        f"a press on a DRAG field also asked for a picker: {on_field}"
    )
    moved = [
        e for e in _drive(_canvas(), packed, _drag(field)) if isinstance(e, ValueEdited)
    ]
    assert moved and moved[0].uniform == "u_gain", (
        "the point meant to prove the event is SELECTIVE never reached a "
        f"drag field, so its silence proved nothing: {moved}"
    )
