"""The graph canvas's per-document state and the pure pieces of its gestures (092 D2, D13).

Transient: nothing here is persisted, and nothing off-draw writes it. What IS persisted -- a
pass's position -- reaches disk only through `ProjectSession.set_pass_positions`, and the
drag's state machine below is what makes "one save per gesture" a fact a test can assert:
`update` returns nothing to write, `commit` is the only thing that does.
"""

from collections.abc import Collection, Iterable
from dataclasses import dataclass, field

from shaderbox.theme import SIZE

Position = tuple[float, float]


@dataclass
class NodeDrag:
    """A press-and-move over one or more nodes: the names, where each started, and the
    accumulated canvas-space delta. The moving picture reads `current()`; the release writes
    `commit()`."""

    origin: dict[str, Position]
    delta: Position = (0.0, 0.0)

    def update(self, dx: float, dy: float) -> None:
        self.delta = (self.delta[0] + dx, self.delta[1] + dy)

    def current(self) -> dict[str, Position]:
        return {
            name: (x + self.delta[0], y + self.delta[1])
            for name, (x, y) in self.origin.items()
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
    # The compile seam ran for this document (092 D1): every pass has ports thereafter.
    compiled: bool = False
    node_drag: NodeDrag | None = None
    wire_drag: WireDrag | None = None
    # The snap guides the current drag aligned to, in canvas units: ("v", x) or ("h", y).
    guides: list[tuple[str, float]] = field(default_factory=list)
    # The Group... name prompt: open, and its buffer.
    group_prompt: bool = False
    group_name: str = ""
    # The rubber band's press point, in screen space, while one is being dragged.
    band_anchor: Position | None = None


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
        height += 4.0 + port_count * SIZE.GRAPH_PORT_ROW
    return width, height


def group_names_in_order(order: Iterable[str], groups: dict[str, str]) -> list[str]:
    """Every group name once, by its first member's place in `order` (the tab row's order)."""
    seen: list[str] = []
    for name in order:
        group = groups.get(name, "")
        if group and group not in seen:
            seen.append(group)
    return seen
