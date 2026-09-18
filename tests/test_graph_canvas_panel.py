"""The panel drives the library and hands back a texture (feature 098).

The framing and the view round-trip are what a headless test can decide; the
pointer translation needs an imgui frame and is covered by the widget's own
tests.
"""

import moderngl
import pytest

from shaderbox.graph_canvas import ffi
from shaderbox.graph_canvas.adapter import pack_nodes
from shaderbox.graph_canvas.panel import (
    GraphCanvasState,
    frame_all,
    refusal_text,
    render_to_texture,
)
from shaderbox.graph_canvas.render import CanvasRenderer
from shaderbox.pass_graph import Port


def _packed() -> object:
    order = ["a", "b", "c"]
    ports = {
        "a": [],
        "b": [Port("u_src", "wired", "a")],
        "c": [Port("u_in", "wired", "b")],
    }
    positions = {"a": (0.0, 0.0), "b": (900.0, 400.0), "c": (1800.0, -400.0)}
    return pack_nodes(order, ports, positions, {}, output="c")


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
    packed = pack_nodes([], {}, {}, {}, output="")
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


def test_a_view_moving_drag_pans_and_a_plain_move_does_not(
    gl_ctx: moderngl.Context,
) -> None:
    """The library does NOT pan -- it zooms from the wheel and uses
    `view_moving` only to suppress hover -- so panning is the host's, as it is
    in the library's own demo. This went missing entirely in the switchover:
    every gesture test drove wires and nodes, and nothing dragged the canvas.

    `pan` is a canvas-space position that is SUBTRACTED, so dragging right
    moves `pan` left and the distance is divided by the zoom.
    """
    renderer = CanvasRenderer(gl=gl_ctx)
    packed = _packed()

    def drag(flags: int) -> float:
        state = GraphCanvasState()
        render_to_texture(
            state,
            renderer,
            packed,
            (800, 600),
            (0, 0, 0, 1),
            ffi.PointerState(x=700.0, y=560.0),
        )
        before = state.view.pan_x
        for x in (700.0, 640.0, 580.0):
            render_to_texture(
                state,
                renderer,
                packed,
                (800, 600),
                (0, 0, 0, 1),
                ffi.PointerState(x=x, y=560.0, flags=flags),
            )
        moved = state.view.pan_x - before
        state.release()
        return moved

    down = int(ffi.Pointer.DOWN)
    panned = drag(down | int(ffi.Pointer.VIEW_MOVING))
    # Dragging 120px LEFT at zoom z raises pan.x by 120/z.
    assert panned > 0.0, f"a view-moving drag did not pan: {panned}"

    assert drag(down) == 0.0, "a drag without view_moving panned"
    assert drag(0) == 0.0, "a pointer with no button down panned"
    renderer.release()


def test_a_pan_does_not_jump_on_the_first_frame(gl_ctx: moderngl.Context) -> None:
    """There is no previous pointer to take a delta from on the frame the
    canvas appears, and treating its absence as the origin would slam the view
    by the pointer's full distance from (0, 0).

    The FIRST frame has to carry the pan flag: a fixture that settles the view
    first has already recorded a pointer, so the guard it is meant to test is
    never reached and the check passes with the guard deleted.
    """
    renderer = CanvasRenderer(gl=gl_ctx)
    packed = _packed()
    moving = int(ffi.Pointer.DOWN) | int(ffi.Pointer.VIEW_MOVING)
    far = ffi.PointerState(x=700.0, y=560.0, flags=moving)

    # Cold: the very first frame this state ever sees is a pan, from a pointer
    # far from the origin.
    cold = GraphCanvasState()
    render_to_texture(cold, renderer, packed, (800, 600), (0, 0, 0, 1), far)
    panned_cold = cold.view

    # The same view, fitted with no pan flag at all.
    still = GraphCanvasState()
    render_to_texture(
        still,
        renderer,
        packed,
        (800, 600),
        (0, 0, 0, 1),
        ffi.PointerState(x=700.0, y=560.0),
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
