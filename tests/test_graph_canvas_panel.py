"""The panel drives the library and hands back a texture (feature 098).

The framing and the view round-trip are what a headless test can decide; the
pointer translation needs an imgui frame and is covered by the widget's own
tests.
"""

import moderngl
import numpy as np
import pytest

from shaderbox.graph_canvas import ffi
from shaderbox.graph_canvas.adapter import (
    BodyRow,
    Moved,
    flat_view,
    pack_nodes,
    pass_key,
)
from shaderbox.graph_canvas.panel import (
    GraphCanvasState,
    frame_all,
    pointer_flags,
    refusal_text,
    render_to_texture,
)
from shaderbox.graph_canvas.render import CanvasRenderer
from shaderbox.pass_graph import Port
from shaderbox.widgets.pass_graph import canvas_theme

# A canvas point the library does NOT claim, so a press there begins a pan.
# Load-bearing rather than arbitrary: a pan is decided at the press, so a
# fixture aiming at a node gets no pan at all -- correctly.
_EMPTY = (40.0, 560.0)


def _packed() -> object:
    order = ["a", "b", "c"]
    ports = {
        "a": [],
        "b": [Port("u_src", "wired", "a")],
        "c": [Port("u_in", "wired", "b")],
    }
    positions = {"a": (0.0, 0.0), "b": (900.0, 400.0), "c": (1800.0, -400.0)}
    return pack_nodes(flat_view(order, ports, positions), {}, output=pass_key("c"))


def test_framing_puts_every_node_on_screen() -> None:
    """A sign error in the pan leaves the graph a thousand pixels off to one
    side and raises nothing, so the check is that the rects land inside."""
    canvas = ffi.Canvas()
    canvas.load_atlas()
    packed = _packed()
    size = (1280.0, 720.0)
    view = frame_all(canvas, packed, size, ffi.PointerState())

    result = canvas.frame(packed.nodes, packed.edges, size, view, ffi.PointerState())
    for i in range(result.rect_count):
        rect = result.node_rects[i]
        assert rect.x + rect.w > 0 and rect.x < size[0]
        assert rect.y + rect.h > 0 and rect.y < size[1]
    canvas.release()


def test_an_empty_graph_frames_without_dividing_by_a_zero_span() -> None:
    canvas = ffi.Canvas()
    canvas.load_atlas()
    packed = pack_nodes(flat_view([], {}, {}), {}, output="")
    view = frame_all(canvas, packed, (800.0, 600.0), ffi.PointerState())
    assert view.zoom > 0.0
    canvas.release()


def test_the_view_the_library_returns_is_carried_into_the_next_frame(
    gl_ctx: moderngl.Context,
) -> None:
    """The library owns the ZOOM; a host that pushes its own back every frame
    fights the gesture instead of continuing it. (The pan is the host's --
    see the pan test below.)"""
    renderer = CanvasRenderer(gl=gl_ctx)
    state = GraphCanvasState()
    packed = _packed()
    pointer = ffi.PointerState()

    render_to_texture(state, renderer, packed, (640, 480), (0, 0, 0, 1), pointer)
    first = state.view
    assert state.fitted

    # A wheel notch over the canvas is the library's to apply.
    zooming = ffi.PointerState(x=320.0, y=240.0, wheel=1.0)
    render_to_texture(state, renderer, packed, (640, 480), (0, 0, 0, 1), zooming)
    assert state.view.zoom != first.zoom

    state.release()
    renderer.release()


def test_the_panel_returns_a_texture_of_the_size_it_was_asked_for(
    gl_ctx: moderngl.Context,
) -> None:
    renderer = CanvasRenderer(gl=gl_ctx)
    state = GraphCanvasState()
    texture, events, claimed = render_to_texture(
        state, renderer, _packed(), (512, 384), (0, 0, 0, 1), ffi.PointerState()
    )
    assert texture.size == (512, 384)
    assert events == []
    assert claimed is False
    state.release()
    renderer.release()


def test_releasing_twice_is_safe(gl_ctx: moderngl.Context) -> None:
    """A document closing mid-frame must not leave a dangling handle, and the
    second release must be a no-op rather than a double free."""
    renderer = CanvasRenderer(gl=gl_ctx)
    state = GraphCanvasState()
    render_to_texture(
        state, renderer, _packed(), (256, 256), (0, 0, 0, 1), ffi.PointerState()
    )
    state.release()
    state.release()
    assert state.canvas is None and state.panel is None
    renderer.release()


@pytest.mark.parametrize(
    "code",
    [
        int(ffi.ConnectError.CYCLE),
        int(ffi.ConnectError.INPUT_TAKEN),
        int(ffi.ConnectError.SELF),
    ],
)
def test_every_refusal_the_library_names_reads_as_a_sentence(code: int) -> None:
    assert refusal_text(code) and not refusal_text(code).startswith("refused (")


def test_an_unknown_refusal_is_a_newer_library_and_not_an_error() -> None:
    """Enums are appended, never renumbered, so a code with no name here means
    the library moved ahead — which is not a reason to raise."""
    assert "999" in refusal_text(999)


def test_every_refusal_code_the_binding_declares_has_a_sentence() -> None:
    """The domain is enumerated from the enum rather than written out beside
    it, so a member added to one and not the other fails here."""
    for error in ffi.ConnectError:
        if error is ffi.ConnectError.NONE:
            continue
        assert not refusal_text(int(error)).startswith("refused ("), error.name


def test_a_view_moving_drag_pans_at_the_right_rate(gl_ctx: moderngl.Context) -> None:
    """The library does NOT pan -- it zooms from the wheel and uses
    `view_moving` only to suppress hover -- so panning is the host's.

    The RATE is asserted, not merely the direction: `pan` is canvas space, so
    a 120px drag moves it 120/zoom, and the fixture's zoom is not 1. An
    earlier version asserted `> 0` and stayed green with the division
    dropped, which would pan at the wrong speed under the hand and error
    nowhere.

    The press must land where the library does NOT claim the pointer, because
    a pan is decided at the press now. A fixture aiming at a node gets no pan
    at all -- correctly -- so `_EMPTY` is load-bearing rather than arbitrary.
    """
    renderer = CanvasRenderer(gl=gl_ctx)
    packed = _packed()
    travel = 120.0

    def drag(flags: int) -> tuple[float, float]:
        state = GraphCanvasState()
        render_to_texture(
            state,
            renderer,
            packed,
            (800, 600),
            (0, 0, 0, 1),
            ffi.PointerState(x=_EMPTY[0], y=_EMPTY[1]),
        )
        before = state.view.pan_x
        for index, offset in enumerate((0.0, travel / 2, travel)):
            carried = flags | (int(ffi.Pointer.PRESSED) if index == 0 and flags else 0)
            render_to_texture(
                state,
                renderer,
                packed,
                (800, 600),
                (0, 0, 0, 1),
                ffi.PointerState(x=_EMPTY[0] - offset, y=_EMPTY[1], flags=carried),
            )
        moved = state.view.pan_x - before
        zoom = state.view.zoom
        state.release()
        return moved, zoom

    down = int(ffi.Pointer.DOWN)
    panned, zoom = drag(down | int(ffi.Pointer.VIEW_MOVING))
    assert panned == pytest.approx(travel / zoom, rel=0.05), (
        f"panned {panned} at zoom {zoom}; expected {travel / zoom}"
    )

    assert drag(down)[0] == 0.0, "a drag without view_moving panned"
    assert drag(0)[0] == 0.0, "a pointer with no button down panned"
    renderer.release()


def test_a_press_the_library_claims_never_becomes_a_pan(
    gl_ctx: moderngl.Context,
) -> None:
    """A press that lands in the same frame the pointer arrives on a node.

    `VIEW_MOVING` is computed from the PREVIOUS frame's claim, so on a fast
    mouse move the flag says background while the press is really on a node.
    Deciding the pan per frame from that flag panned by the whole cursor jump
    -- measured at 820x535 px -- and the node never dragged. The press now
    asks the library first, once, and the answer holds until release.
    """
    renderer = CanvasRenderer(gl=gl_ctx)
    packed = _packed()
    state = GraphCanvasState()
    render_to_texture(
        state,
        renderer,
        packed,
        (800, 600),
        (0, 0, 0, 1),
        ffi.PointerState(x=_EMPTY[0], y=_EMPTY[1]),
    )
    probe = state.canvas.frame(
        packed.nodes, packed.edges, (800.0, 600.0), state.view, ffi.PointerState()
    )
    box = probe.node_rects[0]
    on_node = (box.x + box.w / 2, box.y + box.h / 3)

    before = state.view.pan_x
    moving = int(ffi.Pointer.DOWN) | int(ffi.Pointer.VIEW_MOVING)
    events: list[object] = []
    for index, step in enumerate((0.0, 40.0, 80.0)):
        carried = moving | (int(ffi.Pointer.PRESSED) if index == 0 else 0)
        _texture, frame_events, _claimed = render_to_texture(
            state,
            renderer,
            packed,
            (800, 600),
            (0, 0, 0, 1),
            ffi.PointerState(x=on_node[0] + step, y=on_node[1] + step, flags=carried),
        )
        events.extend(frame_events)

    assert state.view.pan_x == pytest.approx(before), (
        f"a press on a node panned the canvas by {state.view.pan_x - before}"
    )
    assert any(isinstance(event, Moved) for event in events), "the node did not drag"
    state.release()
    renderer.release()


def test_a_pan_does_not_jump_on_the_first_frame(gl_ctx: moderngl.Context) -> None:
    """There is no previous pointer to take a delta from on the frame the
    canvas appears, and treating its absence as the origin would slam the view
    by the pointer's full distance from (0, 0).

    The FIRST frame has to carry the pan flag: a fixture that settles the view
    first has already recorded a pointer, so the guard it is meant to test is
    never reached and the check passes with the guard deleted.

    Both states press the button, because framing is deferred while one is
    down -- a state that presses and one that does not take different paths
    through the view, and then the comparison measures the framing rather
    than the pan it names.
    """
    renderer = CanvasRenderer(gl=gl_ctx)
    packed = _packed()
    moving = int(ffi.Pointer.DOWN) | int(ffi.Pointer.VIEW_MOVING)

    # Cold: the very first frame this state ever sees is a pan, from a pointer
    # far from the origin.
    cold = GraphCanvasState()
    render_to_texture(
        cold,
        renderer,
        packed,
        (800, 600),
        (0, 0, 0, 1),
        ffi.PointerState(x=700.0, y=560.0, flags=moving | int(ffi.Pointer.PRESSED)),
    )
    panned_cold = cold.view

    # The same press with NO pan flag: the one difference between the two.
    still = GraphCanvasState()
    render_to_texture(
        still,
        renderer,
        packed,
        (800, 600),
        (0, 0, 0, 1),
        ffi.PointerState(
            x=700.0,
            y=560.0,
            flags=int(ffi.Pointer.DOWN) | int(ffi.Pointer.PRESSED),
        ),
    )
    assert panned_cold.pan_x == still.view.pan_x, (
        "the first frame panned by the pointer's distance from the origin"
    )
    assert panned_cold.pan_y == still.view.pan_y

    # And the SECOND frame of the same drag does pan, so the guard is a
    # one-frame delay rather than a suppression.
    render_to_texture(
        cold,
        renderer,
        packed,
        (800, 600),
        (0, 0, 0, 1),
        ffi.PointerState(x=580.0, y=560.0, flags=moving),
    )
    assert cold.view.pan_x != panned_cold.pan_x

    cold.release()
    still.release()
    renderer.release()


def test_a_button_the_canvas_is_not_under_never_reaches_the_library() -> None:
    """A click on a context-menu item drawn OVER the canvas is not a node
    click, and a drag elsewhere in the app is not a pan.

    This is the sharpest bug the review found: a `Clicked` runs
    `App.choose_output`, which writes the document -- so every use of the
    node context menu persisted a change to the graph. Coordinates are
    canvas-local, so a pointer outside the region still lands on a node.

    `holding` is what makes the gate safe for a real drag: a gesture the
    canvas BEGAN keeps its buttons while the pointer wanders off, and one
    that began elsewhere never arrives. Checking "the button is down"
    instead lets the menu click straight back in, which is the hole the
    first version of this gate had.
    """
    flags = pointer_flags(hovered=False, left_down=True, left_click=True)
    assert flags & int(ffi.Pointer.DOWN) == 0, "an off-canvas press reached the library"
    assert flags & int(ffi.Pointer.PRESSED) == 0

    # The same press, from a gesture this canvas started: it must arrive.
    carried = pointer_flags(hovered=False, left_down=True, holding=True)
    assert carried & int(ffi.Pointer.DOWN), "an in-flight drag lost its button"

    # And over the canvas, everything arrives as normal.
    over = pointer_flags(hovered=True, left_down=True, left_click=True)
    assert over & int(ffi.Pointer.DOWN) and over & int(ffi.Pointer.PRESSED)


def test_a_right_click_mid_drag_does_not_also_open_a_menu() -> None:
    """The library answers the secondary button BEFORE it looks at the
    gesture in flight, so a right-click during a drag fired a Context_Menu
    and went on dragging the node underneath it -- two gestures at once, each
    individually correct, which is why nothing looked broken.

    The gate is `holding`, the same flag the menu-click fix turns on: a
    gesture this canvas began suppresses the menu until the button is up.
    """
    mid_drag = pointer_flags(
        hovered=True, left_down=True, right_click=True, holding=True
    )
    assert mid_drag & int(ffi.Pointer.ALT_PRESSED) == 0, (
        "a right-click mid-drag still asked for a menu"
    )
    assert mid_drag & int(ffi.Pointer.DOWN), "the drag lost its button"

    # With nothing in flight the menu opens as it always did.
    idle = pointer_flags(hovered=True, right_click=True)
    assert idle & int(ffi.Pointer.ALT_PRESSED), "a plain right-click opened no menu"


def test_a_popup_over_the_canvas_takes_the_pointer() -> None:
    """`CLAIMED` is how a host says "this frame's pointer is mine". The
    library answers it by hovering nothing, starting nothing, and DROPPING a
    gesture already in flight -- which is what a menu opening over a
    half-made wire needs.

    Distinct from the `holding` gate beside it: that one stops an off-canvas
    press from reaching the library at all, and this one covers the press
    that IS over the canvas, under a popup drawn on top of it.
    """
    claimed = pointer_flags(hovered=True, left_down=True, host_claimed=True)
    assert claimed & int(ffi.Pointer.CLAIMED), "the popup did not claim the pointer"
    # Sent on rather than swallowed: the library needs the press to know a
    # gesture was dropped rather than merely paused.
    assert claimed & int(ffi.Pointer.DOWN)

    free = pointer_flags(hovered=True, left_down=True)
    assert free & int(ffi.Pointer.CLAIMED) == 0


def test_framing_waits_for_the_button_to_come_up(gl_ctx: moderngl.Context) -> None:
    """A frame-all mid-press moves the camera under the pointer: its canvas
    position jumps while its screen position has not moved, and the library
    reads that as travel. A click then becomes a drag, and a node the user
    only selected is moved and saved.

    The fixture presses at a point far from the origin, because at the
    default view a framing and its absence return the same numbers and the
    guard cannot be seen.
    """
    renderer = CanvasRenderer(gl=gl_ctx)
    packed = _packed()
    down = int(ffi.Pointer.DOWN) | int(ffi.Pointer.PRESSED)

    held = GraphCanvasState()
    render_to_texture(
        held,
        renderer,
        packed,
        (800, 600),
        (0, 0, 0, 1),
        ffi.PointerState(x=700.0, y=560.0, flags=down),
    )
    assert not held.fitted, "the view was framed while the button was down"
    assert held.view == ffi.View(), "the camera moved under a press"

    # The release frames it, so the request is delayed and not discarded.
    render_to_texture(
        held,
        renderer,
        packed,
        (800, 600),
        (0, 0, 0, 1),
        ffi.PointerState(x=700.0, y=560.0),
    )
    assert held.fitted
    assert held.view != ffi.View(), "the deferred framing never happened"

    held.release()
    renderer.release()


def test_a_hovered_row_lights_up(gl_ctx: moderngl.Context) -> None:
    """The library eases a per-attribute hover and `dt` is what advances it.
    Without a frame time every ease holds at zero and the canvas answers
    "which node" while never answering "which row".

    The aim is resolved AFTER the view has settled, and that is the whole
    difficulty. The canvas frames the graph on its first frame, so a row's
    screen position before framing is not where it ends up -- a coordinate
    is valid only for the frame state it was measured in. Measuring the
    sweep under one view and the pixels under another reports zero every
    time, which is what three wrong findings on this bug all were.

    The pointer also JIGGLES by a hundredth of a pixel: a hover reached
    while nothing moves leaves the stale hover and the current one in
    agreement, so an ordering bug between resolve and update cannot show.
    """
    renderer = CanvasRenderer(gl=gl_ctx)
    # ONE node, not the shared three-node fixture: that one spreads its
    # nodes 1800px apart, so framing zooms far enough out that a row is
    # under a pixel tall and its highlight has nowhere to land. A fixture
    # has to be built where the thing under test is visible.
    packed = pack_nodes(
        flat_view(["n"], {"n": [Port("u_src", "unfilled")]}, {"n": (0.0, 0.0)}),
        {},
        output=pass_key("n"),
        body={"n": [BodyRow("u_gain", (1.0,), None, editable=True)]},
    )
    size = (500, 420)

    def frames(state: GraphCanvasState, x: float, y: float, count: int) -> None:
        for i in range(count):
            render_to_texture(
                state,
                renderer,
                packed,
                size,
                (0.0, 0.0, 0.0, 1.0),
                ffi.PointerState(x=x + (i % 2) * 0.01, y=y),
                dt=1.0 / 60.0,
            )

    # Settle the view, then ask the library where a row IS under it.
    settle = GraphCanvasState()
    frames(settle, -1e6, -1e6, 5)
    assert settle.canvas is not None
    probe = settle.canvas.frame(
        packed.nodes,
        packed.edges,
        (float(size[0]), float(size[1])),
        settle.view,
        ffi.PointerState(x=-1e6, y=-1e6),
        dt=1.0 / 60.0,
    )
    rect = probe.node_rects[0]
    centre_x = rect.x + rect.w * 0.5
    band: list[float] = []
    for step in range(int(rect.h)):
        y = rect.y + step + 0.5
        swept = settle.canvas.frame(
            packed.nodes,
            packed.edges,
            (float(size[0]), float(size[1])),
            settle.view,
            ffi.PointerState(x=centre_x, y=y),
            dt=1.0 / 60.0,
        )
        if swept.node_rects[0].hover_attribute == 0:
            band.append(y)
    settle.release()
    assert band, "no row reported a hover, so this test cannot see one light up"
    row_y = (band[0] + band[-1]) / 2.0

    def pixels(x: float, y: float) -> np.ndarray:
        state = GraphCanvasState()
        frames(state, x, y, 40)
        assert state.panel is not None and state.panel.fbo is not None
        out = np.frombuffer(state.panel.fbo.read(components=3), dtype="u1").copy()
        state.release()
        return out

    away = pixels(-1e6, -1e6)
    on = pixels(centre_x, row_y)
    changed = int((np.abs(away.astype(int) - on.astype(int)) > 0).sum())
    assert changed > 0, (
        "a hovered row changed no pixel, so either the frame time is not "
        "reaching the library or the row was not actually under the pointer"
    )

    # And WITHOUT a frame time nothing moves, which is what says the ease is
    # what lit it rather than some other per-frame difference.
    still = GraphCanvasState()
    for i in range(40):
        render_to_texture(
            still,
            renderer,
            packed,
            size,
            (0.0, 0.0, 0.0, 1.0),
            ffi.PointerState(x=centre_x + (i % 2) * 0.01, y=row_y),
            dt=0.0,
        )
    assert still.panel is not None and still.panel.fbo is not None
    frozen = np.frombuffer(still.panel.fbo.read(components=3), dtype="u1").copy()
    still.release()
    assert int((np.abs(away.astype(int) - frozen.astype(int)) > 0).sum()) == 0, (
        "the row lit up with no time passing, so the highlight is not the ease"
    )
    renderer.release()


def test_the_state_border_is_not_eaten_by_the_title_band(
    gl_ctx: moderngl.Context,
) -> None:
    """The border runs the whole way round, the header band included.

    The band is drawn AFTER the body's outline and inset by the bevel
    (`pal.bevel_width * z`) rather than by the outline's own weight, which
    is `hairline(style, z) * border_scale * state_width`. Those are
    unrelated quantities, so the band clears the border only by luck of
    the numbers, and `state_width` is what spends that luck: at the
    shipped theme the top edge measured 5px of border against 6px on the
    bottom.

    Compared TOP against BOTTOM rather than against a constant. The two
    edges of one node in one frame differ in exactly one thing -- whether
    the header band is painted over them -- so a thickness that is merely
    unexpected fails nothing here, while the band eating a pixel fails.
    A constant would have to be retuned with the theme and would pass on
    a build that thinned every edge equally.

    The OUTPUT node, because `border_scale` 1.6 is the case that loses a
    pixel first; a plain node holds on until the view is zoomed out.
    """
    renderer = CanvasRenderer(gl=gl_ctx)
    packed = pack_nodes(
        flat_view(["n"], {"n": [Port("u_src", "unfilled")]}, {"n": (0.0, 0.0)}),
        {},
        output=pass_key("n"),
        selected=frozenset({pass_key("n")}),
        body={"n": [BodyRow("u_gain", (1.0,), None, editable=True)]},
    )
    size = (500, 420)
    theme = canvas_theme()
    state = GraphCanvasState()
    for _ in range(40):
        render_to_texture(
            state,
            renderer,
            packed,
            size,
            (0.0, 0.0, 0.0, 1.0),
            ffi.PointerState(x=-1e6, y=-1e6),
            theme=theme,
            dt=1.0 / 60.0,
        )
    assert state.panel is not None and state.panel.fbo is not None
    assert state.canvas is not None
    image = np.frombuffer(state.panel.fbo.read(components=3), dtype="u1").reshape(
        size[1], size[0], 3
    )
    probe = state.canvas.frame(
        packed.nodes,
        packed.edges,
        (float(size[0]), float(size[1])),
        state.view,
        ffi.PointerState(x=-1e6, y=-1e6),
        theme=theme,
        dt=1.0 / 60.0,
    )
    rect = probe.node_rects[0]
    want = np.array([c * 255.0 for c in tuple(theme.state_selected)[:3]])

    def runs(x: float, from_top: bool) -> int:
        """How many pixels of border colour run inward from one edge."""
        depth = 0
        for step in range(24):
            y = rect.y + step if from_top else rect.y + rect.h - 1 - step
            row = round(size[1] - 1 - y)
            if not 0 <= row < size[1]:
                break
            pixel = image[row, round(x)].astype(int)
            if not bool(np.all(np.abs(pixel - want) < 60)):
                break
            depth = step + 1
        return depth

    columns = [rect.x + rect.w * f for f in (0.3, 0.5, 0.7)]
    top = min(runs(x, True) for x in columns)
    bottom = min(runs(x, False) for x in columns)
    assert bottom > 0, (
        "the bottom edge carries no border colour, so this compares nothing "
        "-- the fixture never reached a drawn state border"
    )
    assert top >= bottom, (
        f"the state border is {top}px deep under the title band and {bottom}px "
        "on the bottom edge, so the header is painted over its inner side"
    )
    state.release()
    renderer.release()
