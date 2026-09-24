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

from collections.abc import Sequence
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


def pointer_flags(
    hovered: bool,
    left_down: bool = False,
    left_click: bool = False,
    right_click: bool = False,
    double: bool = False,
    shift: bool = False,
    ctrl: bool = False,
    middle: bool = False,
    alt: bool = False,
    claimed: bool = False,
    holding: bool = False,
    cancelled: bool = False,
    host_claimed: bool = False,
) -> int:
    """The pointer flags for one frame, from plain booleans.

    Split out of `pointer_from_io` so the gate is testable without an imgui
    frame: the io read is three lines and the RULE is the part that has been
    wrong twice.

    `host_claimed` is the host saying the pointer is ITS this frame -- a popup
    is up, a modal is open. It is sent on rather than swallowed here, because
    the library answers it by dropping a gesture in flight, which is the
    behaviour a menu opening over a half-made wire needs.
    """
    flags = int(ffi.Pointer.CANCELLED) if cancelled else 0
    if host_claimed:
        flags |= int(ffi.Pointer.CLAIMED)
    if not hovered and not holding:
        return flags
    if left_down:
        flags |= int(ffi.Pointer.DOWN)
    if left_click:
        flags |= int(ffi.Pointer.PRESSED)
    if double:
        flags |= int(ffi.Pointer.DOUBLE)
    if shift:
        flags |= int(ffi.Pointer.EXTEND)
    if ctrl:
        flags |= int(ffi.Pointer.FINE)
    # `ALT_PRESSED` is the library's name for THE SECONDARY BUTTON going
    # down, not for the Alt key -- its own comment says so, and it answers
    # the flag with a Context_Menu event. Mapping the Alt key onto it opened
    # the menu whenever Alt was held, which costs the user alt-tab.
    #
    # Held OFF while a gesture of this canvas is in flight: the library
    # answers the flag before it looks at the gesture, so a right-click
    # mid-drag opened a menu over a node that went on being dragged
    # underneath it -- two gestures, each individually correct.
    if right_click and not holding:
        flags |= int(ffi.Pointer.ALT_PRESSED)
    if middle or (alt and left_down) or (left_down and not claimed):
        flags |= int(ffi.Pointer.VIEW_MOVING)
    return flags


def pointer_from_io(
    canvas_origin: tuple[float, float],
    hovered: bool,
    cancelled: bool = False,
    claimed: bool = False,
    holding: bool = False,
    host_claimed: bool = False,
) -> ffi.PointerState:
    """imgui's mouse this frame, in the canvas's own coordinates.

    The flag RULE lives in `pointer_flags`, which takes plain booleans and is
    the part that has been wrong twice; this reads io and applies it.

    The double click is imgui's, and so the platform's: the library does not
    time one, because an interval decided there would disagree with every
    other double click on the machine.
    """
    io = imgui.get_io()
    flags = pointer_flags(
        hovered=hovered,
        left_down=imgui.is_mouse_down(imgui.MouseButton_.left),
        left_click=imgui.is_mouse_clicked(imgui.MouseButton_.left),
        right_click=imgui.is_mouse_clicked(imgui.MouseButton_.right),
        double=imgui.is_mouse_double_clicked(imgui.MouseButton_.left),
        shift=io.key_shift,
        ctrl=io.key_ctrl,
        alt=io.key_alt,
        middle=imgui.is_mouse_down(imgui.MouseButton_.middle),
        claimed=claimed,
        holding=holding,
        cancelled=cancelled,
        host_claimed=host_claimed,
    )
    return ffi.PointerState(
        x=io.mouse_pos.x - canvas_origin[0],
        y=io.mouse_pos.y - canvas_origin[1],
        wheel=io.mouse_wheel if hovered else 0.0,
        flags=flags,
    )


def result_claims_pointer(
    canvas: ffi.Canvas,
    packed: Packed,
    size: tuple[int, int],
    view: ffi.View,
    pointer: ffi.PointerState,
) -> bool:
    """Whether the library wants THIS frame's pointer, asked before the frame.

    A probe rather than last frame's answer, because the press that starts a
    gesture arrives in the same frame its hit is resolved. It pushes a
    pointer with no buttons, so it resolves hover and starts nothing; the
    real frame follows immediately and overwrites the result either way.

    """
    probe = canvas.frame(
        packed.nodes,
        packed.edges,
        (float(size[0]), float(size[1])),
        view,
        ffi.PointerState(x=pointer.x, y=pointer.y),
    )
    return pointer_is_claimed(probe)


def _hovered_key(result: ffi.Result, packed: Packed) -> str:
    """The canvas node the pointer is over, by KEY, or `""`.

    Read from the PER-NODE flag rather than from the result's own claim bit:
    that one says a press would be the library's -- deliberately wider than
    any node -- and this one says which node the pointer is on.

    The KEY rather than the name, because the highlight is keyed by it: at
    the root a group's box is one node and no pass carries its name.
    """
    for index in range(result.rect_count):
        if result.node_rects[index].flags & ffi.NODE_RECT_HOVERED:
            return packed.key_of(index) or ""
    return ""


def pointer_is_claimed(result: ffi.Result) -> bool:
    """Whether the library has the pointer, so the host must not also act on it.

    A press the host also treats as its own starts two gestures at once -- a
    node drag and a background marquee -- and each is individually correct, so
    nothing looks broken.

    The result's own bit is the WHOLE answer, and deliberately wider than any
    rect a host can test: a pin's grab area overhangs its node, so a press
    the library will answer with a wire can land outside every rect in
    `node_rects`. The per-node HOVERED bit stays narrow and still means "the
    pointer is on this node", which is what the highlight wants.
    """
    return bool(result.flags & ffi.RESULT_POINTER_CLAIMED)


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
    # A BOX has no position of its own -- it is drawn at its members' corner --
    # so a drag on one is applied as a DELTA. The anchor is where the library
    # first reported it this gesture, and the members' own starting positions
    # are frozen beside it: reading them live would compound the delta every
    # frame, since the drag writes them back through `dragging`.
    box_anchors: dict[str, tuple[float, float]] = field(default_factory=dict)
    box_members: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)
    # The node the last context-menu event named, so the popup that opens on
    # one frame still knows what it is about on the next. CLEARED when the
    # popup closes: left standing it names a pass that may since have been
    # deleted or renamed, and the next menu would open about the wrong thing.
    menu_node: str = ""
    # The group, when the menu was opened on a BOX. At the root a group is a
    # node and no pass carries its name, so the node alone cannot say.
    menu_group: str = ""
    # The body row whose picker is open, as (pass, uniform). The library
    # draws the swatch and reports the press; the editor itself is the
    # host's, so this is what the popup is about between frames. Cleared
    # when it closes, for the same reason `menu_node` is.
    picker_row: tuple[str, str] | None = None
    # Last frame's pointer, for the pan delta. `None` until the first frame,
    # so a pan that begins on the frame the canvas appears has no delta to
    # apply rather than a jump from the origin.
    last_pointer: tuple[float, float] | None = None
    # Whether the library held the pointer on the PREVIOUS frame, which is what
    # decides if a plain left-drag pans. A press is delivered on the same frame
    # its hit is resolved, so this frame's answer is not available in time.
    claimed: bool = False
    # The node the library reported hovered on the PREVIOUS frame, by KEY. The
    # hover is resolved inside `gc_frame`, so a highlight packed from it is one
    # frame late -- the same latency the imgui canvas had, for the same reason,
    # and invisible at any rate a hand can outrun.
    hovered: str = ""
    # Whether the CURRENT press is a pan. Decided once, when the button goes
    # down, and held until it comes up.
    panning: bool = False
    # A gesture this canvas began is in flight. Set on a press the canvas
    # received, cleared when the button comes up -- so a press that started
    # on someone else's widget never reaches the library at all.
    holding: bool = False
    # Where the canvas sits in the window this frame. The library answers in
    # canvas-local coordinates, so anything the host draws OVER the canvas --
    # a tooltip, a menu, a test aiming a click -- needs this to get back.
    origin: tuple[float, float] = (0.0, 0.0)
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
    theme: ffi.Theme | None = None,
    dt: float = 0.0,
    text: Sequence[int] = (),
    keys: Sequence[int] = (),
) -> tuple[moderngl.Texture, list[GraphEvent], bool]:
    """Push one frame and draw it. Returns the texture, the events, and whether
    the library claimed the pointer — a host must not treat a claimed press as
    its own."""
    canvas, panel = state.ensure(renderer)
    # Framing moves the camera under the pointer, so it waits for the button
    # to come up. Mid-press the pointer's canvas position jumps with the view
    # while its screen position has not moved, and the library reads that as
    # travel: a click becomes a drag, and a node the user only selected is
    # moved and saved. The request is a latch, not an edge, so nothing is
    # lost by deferring it a frame.
    if not state.fitted and not (pointer.flags & int(ffi.Pointer.DOWN)):
        state.view = frame_all(
            canvas, packed, (float(size[0]), float(size[1])), pointer
        )
        state.fitted = True

    # The library ZOOMS itself (from `wheel`) and does not pan: `view_moving`
    # only tells it to suppress hover. Panning is the host's, as it is in the
    # library's own demo, and it is applied BEFORE the frame so the picture
    # and the hit-testing agree about where the pointer is this frame.
    #
    # `pan` is a canvas-space position that is SUBTRACTED, so dragging the
    # canvas right moves `pan` LEFT, and the distance is divided by the zoom.
    #
    # A pan is a GESTURE with a beginning, not a per-frame decision. Deciding
    # it each frame from `VIEW_MOVING` alone was wrong in the case that
    # matters: the flag is computed from LAST frame's claim, so a press that
    # lands in the same frame the pointer arrives on a node -- any fast mouse
    # move -- reads as a background press and pans by the whole cursor jump
    # (measured: 820x535 px) while the node never drags at all.
    #
    # So the press decides, once, and the decision holds until release: the
    # library gets first refusal on the frame the button goes down, and a pan
    # only starts where it declined.
    down = bool(pointer.flags & int(ffi.Pointer.DOWN))
    pressed = bool(pointer.flags & int(ffi.Pointer.PRESSED))
    if pressed:
        state.panning = bool(pointer.flags & int(ffi.Pointer.VIEW_MOVING)) and not (
            result_claims_pointer(canvas, packed, size, state.view, pointer)
        )
    elif not down:
        state.panning = False

    if state.panning and down and state.last_pointer is not None:
        dx = pointer.x - state.last_pointer[0]
        dy = pointer.y - state.last_pointer[1]
        zoom = max(state.view.zoom, 1e-6)
        state.view = ffi.View(
            state.view.pan_x - dx / zoom, state.view.pan_y - dy / zoom, state.view.zoom
        )
    state.last_pointer = (pointer.x, pointer.y)

    result = canvas.frame(
        packed.nodes,
        packed.edges,
        (float(size[0]), float(size[1])),
        state.view,
        pointer,
        # Sent EVERY frame rather than latched. `gc_frame` treats a null
        # theme as KEEP, so pushing it once is cheaper -- and it means a
        # palette the host recomputes, after an accent swap say, never
        # reaches the canvas again for the life of the handle. The saving was
        # one struct copy per frame, against a theme that could not change.
        theme=theme,
        dt=dt,
        text=text,
        keys=keys,
    )
    # The zoom the library applied comes back on the result; storing it is what
    # makes the next frame continue the gesture rather than fight it.
    state.view = ffi.View(result.pan_x, result.pan_y, result.zoom)
    state.packed = packed

    events = read_events(result, packed)
    state.hovered = _hovered_key(result, packed)
    texture = panel.render(result, size, canvas.distance_range, clear_color)
    # Latched while the button is held: a drag that began on a node must keep
    # counting as the library's for its whole life, or the frame the pointer
    # wanders onto empty canvas starts panning underneath it.
    if pointer.flags & int(ffi.Pointer.PRESSED):
        state.holding = True
    if not (pointer.flags & int(ffi.Pointer.DOWN)):
        state.claimed = pointer_is_claimed(result)
        state.holding = False
    return texture, events, state.claimed
