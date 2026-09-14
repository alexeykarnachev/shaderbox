"""The graph canvas's per-document state and the pure pieces of its gestures (092 D2, D13).

Transient: nothing here is persisted, and nothing off-draw writes it. What IS persisted -- a
pass's position -- reaches disk only through `ProjectSession.set_pass_positions`, and the
drag's state machine below is what makes "one save per gesture" a fact a test can assert:
`update` returns nothing to write, `commit` is the only thing that does.
"""

import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum, auto

from shaderbox.document import Document, sampler_names
from shaderbox.pass_graph import Port, Wiring, node_ports
from shaderbox.theme import SIZE

Position = tuple[float, float]
# A wire's identity anywhere on the canvas: the consumer pass and the sampler the wire
# terminates at. A sampler has one source (072), so the pair names one wire in the whole
# document, and it is what `App.unwire` takes.
WireId = tuple[str, str]


@dataclass
class NodeDrag:
    """A press-and-move over one or more nodes: the names, where each started, the raw
    canvas-space delta the mouse accumulated, and the snap offset the guides add on top.

    `delta` is never corrected, so a node leaves a guide as soon as the cursor does; `snap`
    is recomputed from `raw()` every frame. The moving picture reads `current()`; the release
    writes `commit()`.
    """

    origin: dict[str, Position]
    delta: Position = (0.0, 0.0)
    snap: Position = (0.0, 0.0)

    def update(self, dx: float, dy: float) -> None:
        self.delta = (self.delta[0] + dx, self.delta[1] + dy)

    def raw(self) -> dict[str, Position]:
        return {
            name: (x + self.delta[0], y + self.delta[1])
            for name, (x, y) in self.origin.items()
        }

    def current(self) -> dict[str, Position]:
        return {
            name: (x + self.snap[0], y + self.snap[1])
            for name, (x, y) in self.raw().items()
        }

    def commit(self) -> dict[str, Position]:
        return self.current()


@dataclass
class WireDrag:
    """A wire in flight: from a node's output dot (`producer`), or grabbed off a filled input
    port (`grabbed` = the consumer and its sampler, whose current source the wire carries).
    `start` is the canvas point the wire is drawn from."""

    producer: str
    start: Position
    grabbed: tuple[str, str] | None = None


@dataclass
class GraphViewState:
    pan: Position = (0.0, 0.0)
    zoom: float = 1.0
    # "" is the root; a group name is that group's tab. Revalidated every frame.
    scope: str = ""
    selection: set[str] = field(default_factory=set)
    # One-shot: the first canvas frame at a nonzero size fits the view; a scope change
    # clears it so the new scope fits once too.
    fitted: bool = False
    node_drag: NodeDrag | None = None
    wire_drag: WireDrag | None = None
    # The snap guides the current drag aligned to, in canvas units: ("v", x) or ("h", y).
    guides: list[tuple[str, float]] = field(default_factory=list)
    # The Group... name prompt: open, and its buffer.
    group_prompt: bool = False
    group_name: str = ""
    # The rubber band's press point, in screen space, while one is being dragged.
    band_anchor: Position | None = None
    # Where each input port's hit rect landed on screen this frame, keyed by (pass, sampler):
    # rebuilt every draw, so a headless test can aim a drop where a user would.
    port_rects: dict[tuple[str, str], tuple[float, float, float, float]] = field(
        default_factory=dict
    )
    # The canvas child's screen rect this frame, for the same reason.
    canvas_rect: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    # A press the copilot turn saw held down -- or the one the mid-curve unwire badge
    # consumed -- may not become any other gesture; the latch clears at the END of the frame
    # the button came up on, so the release-frame node click is refused too (093 S6).
    press_blocked: bool = False
    # Last frame's exclusive hover (093 G6, S3), read at draw time because the picture is
    # drawn before the hit rects: a node key, a (node key, slot) pair per dot kind, and the
    # wire's own identity. Every field is written every frame, `None` included.
    hovered_node: str | None = None
    hovered_port: tuple[str, int] | None = None
    hovered_out: tuple[str, int] | None = None
    hovered_wire: WireId | None = None
    # The selected wire, revalidated against the drawn edges every frame. Exclusive with
    # `selection`, so one Delete has one target (093 S4).
    selected_wire: WireId | None = None
    # The selected wire's unwire badge on screen this frame, or None: hand hit-tested on the
    # press, never an imgui item (093 S6).
    x_rect: tuple[float, float, float, float] | None = None
    # The node keys in draw AND hit-test order this frame, selected and dragged last (093 S7).
    node_order: list[str] = field(default_factory=list)
    # Each drawn wire's screen-space midpoint this frame, so a headless test can aim a click
    # where a user would.
    wire_mids: dict[WireId, Position] = field(default_factory=dict)


def revalidated_scope(scope: str, groups: Collection[str]) -> str:
    """The scope to draw this frame: the requested one while some pass still carries it,
    else the root (the last member can leave from inside the tab)."""
    return scope if scope in groups else ""


def node_size(port_count: int, box: bool) -> tuple[float, float]:
    """A node's canvas-space size at zoom 1: the picture, the name, and one row per port."""
    width = float(SIZE.GRAPH_NODE_W + (SIZE.GRAPH_BOX_EXTRA_W if box else 0))
    height = float(
        SIZE.GRAPH_PAD + SIZE.GRAPH_THUMB + SIZE.GRAPH_NAME_H + SIZE.GRAPH_PAD
    )
    if port_count:
        height += SIZE.GRAPH_PORT_TOP + port_count * SIZE.GRAPH_PORT_ROW
    return width, height


def ports_of(document: Document, wiring: Wiring) -> dict[str, list[Port]]:
    """Every pass's input ports (092 D1), from its compiled program and its wiring row."""
    return {
        name: node_ports(
            sampler_names(render_pass),
            render_pass.uniform_values,
            wiring.get(name, {}),
            name,
        )
        for name, render_pass in document.passes.items()
    }


def node_sizes(ports: Mapping[str, Sequence[Port]]) -> dict[str, tuple[float, float]]:
    """Every pass's node size, one place for the layout and Arrange to agree on."""
    return {name: node_size(len(port_list), False) for name, port_list in ports.items()}


def group_names_in_order(order: Iterable[str], groups: dict[str, str]) -> list[str]:
    """Every group name once, by its first member's place in `order` (the tab row's order)."""
    seen: list[str] = []
    for name in order:
        group = groups.get(name, "")
        if group and group not in seen:
            seen.append(group)
    return seen


# ---- the wire's pure geometry (093 S9) -------------------------------------------------------


def wire_points(
    a: Position, b: Position, zoom: float
) -> tuple[Position, Position, Position, Position]:
    """One cubic bezier's four points for a wire between two screen points (093 G1).

    The control offset is non-negative for every pair of endpoints, so a fold (`2 * offset
    <= dx`) is unreachable while `dx < 0` -- a backward wire is the S-curve the same two
    lines produce, with no branch on the sign of `dx` and no second curve family.
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    offset = max(
        SIZE.GRAPH_WIRE_MIN_OFF * zoom, SIZE.GRAPH_WIRE_BOW * math.hypot(dx, dy)
    )
    return a, (a[0] + offset, a[1]), (b[0] - offset, b[1]), b


def bezier_point(
    p0: Position, cp0: Position, cp1: Position, p3: Position, t: float
) -> Position:
    u = 1.0 - t
    w0 = u * u * u
    w1 = 3.0 * u * u * t
    w2 = 3.0 * u * t * t
    w3 = t * t * t
    return (
        w0 * p0[0] + w1 * cp0[0] + w2 * cp1[0] + w3 * p3[0],
        w0 * p0[1] + w1 * cp0[1] + w2 * cp1[1] + w3 * p3[1],
    )


def wire_hit_threshold(zoom: float) -> float:
    """How far from a wire, in SCREEN pixels, a click still lands on it (093 G4).

    The floor is what makes a wire clickable at low zoom: the stroke itself has a 1px screen
    floor, so a threshold that only scaled would give a 1px line a 1.5px reach.
    """
    return max(float(SIZE.GRAPH_WIRE_HIT_FLOOR), SIZE.GRAPH_WIRE_W * 2.0 * zoom)


def _point_segment_distance(p: Position, a: Position, b: Position) -> float:
    ax, ay = b[0] - a[0], b[1] - a[1]
    length_sq = ax * ax + ay * ay
    if length_sq <= 0.0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / length_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(p[0] - (a[0] + t * ax), p[1] - (a[1] + t * ay))


def wire_hit(
    mouse: Position,
    points: tuple[Position, Position, Position, Position],
    threshold: float,
    segs: int,
) -> float | None:
    """The mouse's distance to the flattened curve, or `None` when it is over `threshold`.

    A bounding-box reject expanded by the threshold comes first, so a canvas of wires costs
    one rect test each before any of them is flattened.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if not (
        min(xs) - threshold <= mouse[0] <= max(xs) + threshold
        and min(ys) - threshold <= mouse[1] <= max(ys) + threshold
    ):
        return None
    flat = [bezier_point(*points, i / segs) for i in range(segs + 1)]
    best = min(
        _point_segment_distance(mouse, flat[i], flat[i + 1]) for i in range(segs)
    )
    return best if best <= threshold else None


class WireState(StrEnum):
    # What a wire's stroke says about it, in precedence order (093 G18): a cycle-error wire
    # stays red even while hovered, exactly as `node.error` outranks `selected` on a border.
    ERROR = auto()
    SELECTED = auto()
    HOVERED = auto()
    DIM = auto()
    NORMAL = auto()


def wire_state(on_cycle: bool, selected: bool, hovered: bool, dim: bool) -> WireState:
    if on_cycle:
        return WireState.ERROR
    if selected:
        return WireState.SELECTED
    if hovered:
        return WireState.HOVERED
    return WireState.DIM if dim else WireState.NORMAL


def revalidated_wire(
    selected: WireId | None, drawn: Collection[WireId]
) -> WireId | None:
    """The wire selection to keep this frame: the requested one while a drawn wire carries
    it, else nothing (an unwire, a pass delete or a scope change retires it)."""
    if selected is None or selected not in drawn:
        return None
    return selected


def delete_allowed(
    pressed: bool,
    hovered: bool,
    any_item_active: bool,
    blocked: bool,
    has_wire: bool,
) -> bool:
    """Whether a Delete/Backspace press unwires the selected wire (093 S5).

    `hovered` alone refuses a key typed into the group prompt or during a held press (both
    measured); `any_item_active` is what refuses a text input active in ANOTHER window while
    the mouse rests over the canvas.
    """
    return pressed and hovered and not any_item_active and not blocked and has_wire
