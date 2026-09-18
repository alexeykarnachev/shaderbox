"""The graph canvas as an imgui-hosted panel (feature 098).

The seam between the library and the app: it feeds imgui's pointer into
`gc_frame`, renders the result into an FBO, presents that FBO with
`imgui.image`, and turns the events back into `App` verb calls.

imgui still owns the window, the tab row and the menus this feature; what it
no longer owns is the canvas interior, which is now the library's geometry
drawn by moderngl.

The pointer comes from imgui's io rather than from glfw, because imgui already
owns the window's input and knows whether the canvas region is hovered. Going
around it while it still owns the window is how a gesture gets delivered twice.
"""

from dataclasses import dataclass, field

import moderngl
from imgui_bundle import imgui

from shaderbox.graph_canvas import ffi
from shaderbox.graph_canvas.adapter import GraphEvent, Packed, read_events
from shaderbox.graph_canvas.render import CanvasPanel, CanvasRenderer

# The library's own reading of the refusal codes, for a toast that names the
# rule rather than a number.
_REFUSALS: dict[int, str] = {
    int(ffi.ConnectError.MISSING): "no such port",
    int(ffi.ConnectError.SAME_SIDE): "two inputs or two outputs",
    int(ffi.ConnectError.SELF): "a pass cannot read itself here",
    int(ffi.ConnectError.SAME_NODE): "both ends on one pass",
    int(ffi.ConnectError.INPUT_TAKEN): "that input already has a wire",
    int(ffi.ConnectError.DUPLICATE): "that wire is already there",
    int(ffi.ConnectError.CYCLE): "that read would close a loop",
}


def refusal_text(code: int) -> str:
    """An unknown code is a NEWER library, not a broken one, so it reads as
    unknown rather than raising."""
    return _REFUSALS.get(code, f"refused (code {code})")


def pointer_from_io(
    canvas_origin: tuple[float, float], hovered: bool, cancelled: bool = False
) -> ffi.PointerState:
    """imgui's mouse this frame, in the canvas's own coordinates.

    `cancelled` is how the pointer is taken back mid-gesture -- a view switch,
    a modal, a copilot turn. Without it a button release and a pre-empted
    gesture look identical, and a wire in flight commits wherever the cursor
    happened to be.

    The double click is imgui's, and so the platform's: the library does not
    time one, because an interval decided there would disagree with every
    other double click on the machine.
    """
    io = imgui.get_io()
    flags = int(ffi.Pointer.CANCELLED) if cancelled else 0
    if imgui.is_mouse_down(imgui.MouseButton_.left):
        flags |= int(ffi.Pointer.DOWN)
    if imgui.is_mouse_clicked(imgui.MouseButton_.left):
        flags |= int(ffi.Pointer.PRESSED)
    if imgui.is_mouse_double_clicked(imgui.MouseButton_.left):
        flags |= int(ffi.Pointer.DOUBLE)
    if io.key_shift:
        flags |= int(ffi.Pointer.EXTEND)
    if io.key_ctrl:
        flags |= int(ffi.Pointer.FINE)
    if io.key_alt:
        flags |= int(ffi.Pointer.ALT_PRESSED)
    if imgui.is_mouse_down(imgui.MouseButton_.middle) or (
        io.key_alt and imgui.is_mouse_down(imgui.MouseButton_.left)
    ):
        flags |= int(ffi.Pointer.VIEW_MOVING)
    return ffi.PointerState(
        x=io.mouse_pos.x - canvas_origin[0],
        y=io.mouse_pos.y - canvas_origin[1],
        wheel=io.mouse_wheel if hovered else 0.0,
        flags=flags,
    )


@dataclass
class GraphCanvasState:
    """One document's live canvas: the library handle, its panel, and the view.

    Transient — nothing here is persisted. A pass's POSITION is persisted, and
    it reaches disk only through the App verb a drag's release calls, which is
    what keeps "one write per gesture" true now that the library owns the drag
    itself.
    """

    canvas: ffi.Canvas | None = None
    panel: CanvasPanel | None = None
    view: ffi.View = field(default_factory=ffi.View)
    packed: Packed | None = None
    # A drag reports every frame; the positions are written once, on release.
    dragging: dict[str, tuple[float, float]] = field(default_factory=dict)
    fitted: bool = False

    def ensure(self, renderer: CanvasRenderer) -> tuple[ffi.Canvas, CanvasPanel]:
        if self.canvas is None:
            self.canvas = ffi.Canvas()
            self.canvas.load_atlas()
        if self.panel is None:
            self.panel = CanvasPanel(renderer)
        return self.canvas, self.panel

    def release(self) -> None:
        if self.panel is not None:
            self.panel.release()
            self.panel = None
        if self.canvas is not None:
            self.canvas.release()
            self.canvas = None


def frame_all(
    canvas: ffi.Canvas,
    packed: Packed,
    size: tuple[float, float],
    pointer: ffi.PointerState,
) -> ffi.View:
    """The view that puts the whole graph on screen.

    Solved from the equation rather than nudged: a point lands at
    `screen = origin + (canvas_point - pan) * zoom`, so framing means solving
    that for `pan`. The node rects come back in SCREEN space, which is what
    makes the first pass measurable at all.
    """
    probe = canvas.frame(packed.nodes, packed.edges, size, ffi.View(), pointer)
    if probe.rect_count == 0:
        return ffi.View()
    rects = [probe.node_rects[i] for i in range(probe.rect_count)]
    lo_x = min(r.x for r in rects)
    hi_x = max(r.x + r.w for r in rects)
    lo_y = min(r.y for r in rects)
    hi_y = max(r.y + r.h for r in rects)
    span_x = max(hi_x - lo_x, 1.0)
    span_y = max(hi_y - lo_y, 1.0)
    zoom = max(min(size[0] / (span_x * 1.2), size[1] / (span_y * 1.2), 1.5), 0.2)
    return ffi.View(
        pan_x=(lo_x + hi_x) / 2 - (size[0] / 2) / zoom,
        pan_y=(lo_y + hi_y) / 2 - (size[1] / 2) / zoom,
        zoom=zoom,
    )


def render_to_texture(
    state: GraphCanvasState,
    renderer: CanvasRenderer,
    packed: Packed,
    size: tuple[int, int],
    clear_color: tuple[float, float, float, float],
    pointer: ffi.PointerState,
) -> tuple[moderngl.Texture, list[GraphEvent], bool]:
    """Push one frame and draw it. Returns the texture, the events, and whether
    the library claimed the pointer — a host must not treat a claimed press as
    its own."""
    canvas, panel = state.ensure(renderer)
    if not state.fitted:
        state.view = frame_all(canvas, packed, (float(size[0]), float(size[1])), pointer)
        state.fitted = True

    result = canvas.frame(
        packed.nodes,
        packed.edges,
        (float(size[0]), float(size[1])),
        state.view,
        pointer,
    )
    # The library pans and zooms itself; storing what it returns is what makes
    # the next frame continue the gesture rather than fight it.
    state.view = ffi.View(result.pan_x, result.pan_y, result.zoom)
    state.packed = packed

    events = read_events(result, packed)
    texture = panel.render(result, size, canvas.distance_range, clear_color)
    claimed = bool(result.flags & ffi.RESULT_POINTER_CLAIMED)
    return texture, events, claimed
