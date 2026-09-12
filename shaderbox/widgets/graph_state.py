"""The graph canvas's per-document state and the pure pieces of its gestures (092 D2, D13).

Transient: nothing here is persisted, and nothing off-draw writes it. What IS persisted -- a
pass's position -- reaches disk only through `ProjectSession.set_pass_positions`, and the
drag's state machine below is what makes "one save per gesture" a fact a test can assert:
`update` returns nothing to write, `commit` is the only thing that does.
"""

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from shaderbox.document import Document, sampler_names
from shaderbox.pass_graph import Port, Wiring, node_ports
from shaderbox.theme import SIZE

Position = tuple[float, float]


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
