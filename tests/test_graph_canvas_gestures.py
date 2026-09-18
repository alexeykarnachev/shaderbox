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
    Clicked,
    Moved,
    Packed,
    Unwired,
    Wired,
    pack_nodes,
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
        ["a", "b"],
        {"a": [], "b": [port]},
        {"a": (0.0, 0.0), "b": (300.0, 0.0)},
        {},
        output="b",
    )


def _rect(canvas: ffi.Canvas, packed: Packed, node: int) -> tuple[float, float, float, float]:
    """One node's screen rect, COPIED OUT: everything in the result points into
    the handle's own storage and the next frame overwrites all of it."""
    result = canvas.frame(
        packed.nodes, packed.edges, SIZE, ffi.View(), ffi.PointerState()
    )
    box = result.node_rects[node]
    return (box.x, box.y, box.w, box.h)


def _wire_start(canvas: ffi.Canvas, packed: Packed, node: int) -> tuple[float, float]:
    """A point on `node`'s output pin that actually STARTS a wire.

    Found by trying, not by asking where the hover is: the two differ. The
    pin's grab area overhangs the node's edge so a press there is not claimed
    by the body, while `hover_attribute` is only reported inside the node --
    measured, the grab sits at y 178 and the hover reports at y 170 on the
    same pin. A point taken from the hover starts a NODE DRAG, which is
    exactly what the wire gesture failing looks like.
    """
    x0, y0, w, h = _rect(canvas, packed, node)
    for x in range(int(x0 + w) - 12, int(x0 + w) + 9, 2):
        for y in range(int(y0 + h) - 40, int(y0 + h) + 1, 2):
            probe = ffi.Canvas()
            view = ffi.View()
            probe.frame(
                packed.nodes, packed.edges, SIZE, view,
                ffi.PointerState(x=float(x), y=float(y), flags=DOWN | PRESSED),
            )
            after = probe.frame(
                packed.nodes, packed.edges, SIZE, view,
                ffi.PointerState(x=float(x) + 50.0, y=float(y), flags=DOWN),
            )
            moved = any(isinstance(e, Moved) for e in read_events(after, packed))
            probe.release()
            if not moved:
                return (float(x), float(y))
    raise AssertionError(f"no point on node {node}'s output starts a wire")


def _wire_drop(
    canvas: ffi.Canvas, packed: Packed, source: tuple[float, float], node: int
) -> tuple[float, float]:
    """A point where a wire from `source` actually LANDS on `node`'s input.

    Probed by completing the gesture, for the same reason `_wire_start` is:
    the grab area and the hover-reporting area are not the same rectangle, and
    a drop aimed by hover silently falls on the background.
    """
    x0, y0, _w, h = _rect(canvas, packed, node)
    for x in range(int(x0) - 14, int(x0) + 15, 2):
        for y in range(int(y0 + h) - 50, int(y0 + h) + 1, 2):
            probe = ffi.Canvas()
            landed = _drive(
                probe,
                packed,
                [
                    ffi.PointerState(x=source[0], y=source[1], flags=DOWN | PRESSED),
                    ffi.PointerState(
                        x=(source[0] + x) / 2, y=source[1], flags=DOWN
                    ),
                    ffi.PointerState(x=float(x), y=float(y), flags=DOWN),
                    ffi.PointerState(x=float(x), y=float(y), flags=0),
                ],
            )
            probe.release()
            if any(isinstance(e, Wired) for e in landed):
                return (float(x), float(y))
    raise AssertionError(f"no point lands a wire on node {node}")


def _connected_input(
    canvas: ffi.Canvas, packed: Packed, node: int
) -> tuple[float, float]:
    """A point whose PRESS picks up the wire into `node`'s input.

    Probed, and the range reaches OUTSIDE the node: the input pin's grab area
    overhangs the left edge (measured at x = 290 for a node at x = 300), which
    is exactly the overhang that lets a pin on the edge be grabbed without the
    body claiming the press.
    """
    x0, y0, _w, h = _rect(canvas, packed, node)
    # Bottom-up: the pins sit in the node's lower third, and each sample costs
    # a fresh handle because the press mutates the library's own copy.
    for y in range(int(y0 + h), int(y0), -2):
        for x in range(int(x0) - 16, int(x0) + 30, 2):
            # No atlas: the probe hit-tests, and loading the metrics per
            # sample is what made this scan take seconds. Geometry does not
            # depend on it -- without one the library simply emits no text.
            probe = ffi.Canvas()
            events = _drive(
                probe,
                packed,
                [ffi.PointerState(x=float(x), y=float(y), flags=DOWN | PRESSED)],
            )
            probe.release()
            if any(isinstance(e, Unwired) for e in events):
                return (float(x), float(y))
    raise AssertionError(f"no press picks up the wire into node {node}")


def _input_pin(
    canvas: ffi.Canvas, packed: Packed, node: int, attribute: int
) -> tuple[float, float]:
    """A point that reports `attribute` hovered on `node`."""
    x0, y0, w, h = _rect(canvas, packed, node)
    for x in range(int(x0) + 2, int(x0 + w), 2):
        for y in range(int(y0), int(y0 + h) + 1, 2):
            probe = canvas.frame(
                packed.nodes,
                packed.edges,
                SIZE,
                ffi.View(),
                ffi.PointerState(x=float(x), y=float(y)),
            )
            if probe.node_rects[node].hover_attribute == attribute:
                return (float(x), float(y))
    raise AssertionError(f"node {node} never reported attribute {attribute}")


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
    source = _wire_start(canvas, packed, 0)
    target = _wire_drop(canvas, packed, source, 1)

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
    source = _wire_start(canvas, packed, 0)

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
    target = _connected_input(canvas, packed, 1)

    events = _drive(
        canvas,
        packed,
        [ffi.PointerState(x=target[0], y=target[1], flags=DOWN | PRESSED)],
    )
    assert Unwired("b", "u_src") in events
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
    assert Clicked("b", False) in events
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
    assert Clicked("b", True) in events
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
            ffi.PointerState(
                x=point[0], y=point[1], flags=DOWN | PRESSED | DOUBLE
            ),
            ffi.PointerState(x=point[0], y=point[1], flags=DOUBLE),
        ],
    )
    assert Activated("b") in events
    canvas.release()


def test_a_ghost_swallows_nothing() -> None:
    """A node that refuses every gesture must let the press fall through, which
    is the difference between a node that is not interactive and a hole."""
    canvas = _canvas()
    packed = pack_nodes(
        ["a", "b"],
        {"a": [], "b": [Port("u_src", "unfilled")]},
        {"a": (0.0, 0.0), "b": (300.0, 0.0)},
        {},
        output="b",
        ghosts=frozenset({"a"}),
    )
    x0, y0, w, h = _rect(canvas, packed, 0)
    point = (x0 + w / 2, y0 + h / 3)

    events = _drive(
        canvas,
        packed,
        [
            ffi.PointerState(x=point[0], y=point[1], flags=DOWN | PRESSED),
            ffi.PointerState(x=point[0] + 60.0, y=point[1], flags=DOWN),
            ffi.PointerState(x=point[0] + 60.0, y=point[1], flags=0),
        ],
    )
    assert not any(isinstance(e, (Moved, Clicked)) and e.name == "a" for e in events)
    canvas.release()


def test_the_pointer_reads_as_claimed_over_a_node_and_free_over_the_background() -> None:
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
