"""shaderbox's document mapped onto graph_canvas's node model (feature 098).

The shaderbox-specific half of the binding: `ffi.py` and `render.py` know the C
ABI and the geometry and nothing else, and this module is the only one that
mentions a pass, a sampler or a wiring. Keeping the split is what makes the
other two liftable into another project.

Identity crosses by ID and never by index. A node's id is a hash of its pass
NAME and an edge's of the (consumer, sampler) pair the document already uses to
name a wire, so an event names a thing in the document rather than a position
in whatever array this frame happened to pack. The library carries the u64 and
never interprets it.
"""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from shaderbox.graph_canvas.ffi import (
    EdgeSpec,
    EventKind,
    Gesture,
    NodeSpec,
    PinFill,
    PinShape,
    PortSpec,
    PreviewFit,
    Result,
)
from shaderbox.pass_graph import Port

# A pass name and a wire id are different namespaces; the tag keeps a pass
# called "a" from colliding with a wire whose pair stringifies the same way.
_NODE_TAG: bytes = b"node:"
_EDGE_TAG: bytes = b"edge:"


def _hash_id(tag: bytes, *parts: str) -> int:
    """A stable u64 for a host-side name. Never zero, which the library's own
    examples use as "no id"."""
    digest = hashlib.blake2b(
        tag + b"\x00".join(p.encode() for p in parts), digest_size=8
    )
    return int.from_bytes(digest.digest(), "big") | 1


def node_id(pass_name: str) -> int:
    return _hash_id(_NODE_TAG, pass_name)


def edge_id(consumer: str, sampler: str) -> int:
    return _hash_id(_EDGE_TAG, consumer, sampler)


# `Port.kind` decides how the pin reads, the way the hand-drawn canvas did:
# a wired port is solid, an unfilled one is an outline, a port explicitly set to
# nothing gets the cored outline, and a media-bound one is a square. The three
# axes are independent, so shaderbox composes them rather than asking the
# library for a named state it would have to define.
_PIN_BY_KIND: dict[str, tuple[PinShape, PinFill]] = {
    "wired": (PinShape.DOT, PinFill.FILLED),
    "unfilled": (PinShape.DOT, PinFill.HOLLOW),
    "none": (PinShape.DOT, PinFill.CORED),
    "media": (PinShape.SQUARE, PinFill.FILLED),
}


RGBA = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class NodePalette:
    """The three colours the node's chrome needs, handed in by the caller.

    The adapter knows the document and the library; it does not know the
    theme, which imports imgui. Passing the tokens keeps this layer free of
    the UI without hard-coding a look.
    """

    hover: RGBA
    select: RGBA
    engine_uniform: RGBA


# What a caller that passes no palette gets: white for both highlights, which
# is visible against every node fill and belongs to no theme.
_NEUTRAL_PALETTE: NodePalette = NodePalette(
    hover=(1.0, 1.0, 1.0, 1.0),
    select=(1.0, 1.0, 1.0, 1.0),
    engine_uniform=(1.0, 1.0, 1.0, 1.0),
)


def _border_of(
    name: str,
    hovered: str,
    selected: frozenset[str],
    ghost: bool,
    colors: "NodePalette",
) -> RGBA | None:
    """The border colour, in precedence order: selected, then hovered, then
    the output's accent, then the library's own default.

    Selection outranks hover so a hovered selected node still reads as
    selected; a ghost takes none of them, because it refuses every gesture and
    a highlight would promise an interaction it will not honour.
    """
    if ghost:
        return None
    if name in selected:
        return colors.select
    if name == hovered:
        return colors.hover
    return None


def _border_scale_of(
    name: str, output: str, hovered: str, selected: frozenset[str]
) -> float:
    """Hover and selection THICKEN the border; nothing changes the node's own
    size, so a highlight never shifts what is under the cursor."""
    if name in selected:
        return 2.2
    if name == hovered:
        return 1.8
    return 1.6 if name == output else 1.0


@dataclass(frozen=True, slots=True)
class NodeView:
    """One node as the adapter packed it, kept so an event can be resolved.

    `ports` is this node's attribute list in the order it was packed, which is
    what an edge's `to_attr` indexes. The library counts inputs and outputs
    together in declaration order, so the output's slot is `len(inputs)`.
    """

    name: str
    inputs: tuple[str, ...]
    is_box: bool


@dataclass(frozen=True, slots=True)
class Packed:
    """One frame's nodes and edges, plus what is needed to read the events back."""

    nodes: tuple[NodeSpec, ...]
    edges: tuple[EdgeSpec, ...]
    views: tuple[NodeView, ...]

    def view_of(self, index: int) -> NodeView | None:
        if 0 <= index < len(self.views):
            return self.views[index]
        return None

    def name_of(self, index: int) -> str | None:
        view = self.view_of(index)
        return None if view is None else view.name


def pack_nodes(
    order: Sequence[str],
    ports: Mapping[str, Sequence[Port]],
    positions: Mapping[str, tuple[float, float]],
    previews: Mapping[str, tuple[int, int, int]],
    output: str,
    ghosts: frozenset[str] = frozenset(),
    engine: Mapping[str, Sequence[str]] | None = None,
    hovered: str = "",
    selected: frozenset[str] = frozenset(),
    palette: NodePalette | None = None,
) -> Packed:
    """Turn the document's passes into one frame's nodes and edges.

    `previews` maps a pass to (texture name, width, height) — the real GL
    texture the pass renders into, which the library samples for the node's
    picture. A pass with no compiled canvas is left out of the mapping and
    draws without one.

    `ghosts` are passes shown for context outside the current scope: faded,
    dashed, and refusing every gesture, so a press falls through to the canvas
    rather than being swallowed by something that does nothing.

    `engine` names each pass's engine-driven uniforms (`u_time` and its
    siblings). They are CONTROL rows -- a label in the node's body with no pin
    -- because the engine writes them and no wire can: giving them a pin would
    offer a connection the document cannot express.

    `hovered` and `selected` decide the border. Hover has to be visible
    without moving anything, so it recolours and thickens the border rather
    than resizing the node; selection outranks it, since a hovered selected
    node should still read as selected.

    The nodes and their attributes are built in ONE pass, because each node
    names a contiguous run of the flat attribute array. The edges are resolved
    against the SAME ordering, so a wire's endpoints index what was packed
    rather than what the document happens to iterate.
    """
    colors = palette or _NEUTRAL_PALETTE
    index_of: dict[str, int] = {name: i for i, name in enumerate(order)}
    # Where each node's single output landed in its own attribute list.
    output_slot: dict[str, int] = {}
    nodes: list[NodeSpec] = []
    views: list[NodeView] = []

    for name in order:
        node_ports_ = list(ports.get(name, ()))
        specs: list[PortSpec] = []
        for port in node_ports_:
            shape, fill = _PIN_BY_KIND.get(port.kind, (PinShape.DOT, PinFill.UNSET))
            specs.append(
                PortSpec(
                    label=port.sampler, is_input=True, pin_shape=shape, pin_fill=fill
                )
            )
        output_slot[name] = len(specs)
        specs.append(PortSpec(label="out", is_input=False))

        # An engine uniform gets a PIN that refuses connection rather than a
        # control row, and the reason is a library limit worth knowing: a
        # control carries no pin, and an attribute has no label colour, so a
        # control cannot be tinted at all. The pin is the only coloured thing
        # on the row. It is drawn hollow and square -- the shape no wirable
        # port uses -- and the node refuses the wire gesture anyway where
        # every port is one of these.
        for label in (engine or {}).get(name, ()):
            specs.append(
                PortSpec(
                    label=label,
                    is_input=True,
                    pin_shape=PinShape.SQUARE,
                    pin_fill=PinFill.HOLLOW,
                    color=colors.engine_uniform,
                )
            )

        tex, width, height = previews.get(name, (0, 0, 0))
        ghost: bool = name in ghosts
        nodes.append(
            NodeSpec(
                id=node_id(name),
                title=name,
                x=positions.get(name, (0.0, 0.0))[0],
                y=positions.get(name, (0.0, 0.0))[1],
                ports=specs,
                preview_tex=tex,
                preview_w=width,
                preview_h=height,
                preview_aspect=(width / height) if width and height else 1.0,
                preview_fit=PreviewFit.CONTAIN,
                fade=0.6 if ghost else 0.0,
                dashed=ghost,
                border=_border_of(name, hovered, selected, ghost, colors),
                border_scale=_border_scale_of(name, output, hovered, selected),
                accepts=int(Gesture.NONE) if ghost else 0,
            )
        )
        views.append(
            NodeView(
                name=name,
                inputs=tuple(p.sampler for p in node_ports_),
                is_box=False,
            )
        )

    edges: list[EdgeSpec] = []
    for consumer_index, name in enumerate(order):
        for slot, port in enumerate(ports.get(name, ())):
            if port.kind != "wired" or port.source is None:
                continue
            producer = index_of.get(port.source)
            if producer is None:
                continue
            # Read from the packed list rather than recomputed as "the input
            # count". The two agree today -- the output is emitted straight
            # after the samplers and the engine rows follow it -- so this
            # fixes no live bug; it removes an arithmetic that silently
            # depends on that emission order, in a function that now appends
            # two kinds of row after the output.
            producer_out = output_slot.get(port.source)
            if producer_out is None:
                continue
            edges.append(
                EdgeSpec(
                    id=edge_id(name, port.sampler),
                    from_node=producer,
                    from_attr=producer_out,
                    to_node=consumer_index,
                    to_attr=slot,
                )
            )

    return Packed(tuple(nodes), tuple(edges), tuple(views))


@dataclass(frozen=True, slots=True)
class Moved:
    name: str
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Clicked:
    name: str
    extend: bool


@dataclass(frozen=True, slots=True)
class Activated:
    name: str


@dataclass(frozen=True, slots=True)
class MenuRequested:
    """`name` is None for the canvas's own menu. `(x, y)` is in SCREEN space."""

    name: str | None
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Wired:
    """A wire the user landed: `consumer` reads `producer` through `sampler`."""

    producer: str
    consumer: str
    sampler: str


@dataclass(frozen=True, slots=True)
class Unwired:
    consumer: str
    sampler: str


@dataclass(frozen=True, slots=True)
class Refused:
    """A wire the library rejected, and by which of its rules."""

    reason: int


GraphEvent = Moved | Clicked | Activated | MenuRequested | Wired | Unwired | Refused


def read_events(result: Result, packed: Packed) -> list[GraphEvent]:
    """This frame's events, resolved from indices back to pass names.

    An event about a node the frame no longer holds is dropped rather than
    guessed at: the library reports against the array it was handed, and a
    host that renamed or deleted between frames would otherwise act on
    whatever now sits at that index.
    """
    events: list[GraphEvent] = []
    for i in range(result.event_count):
        event = result.events[i]
        kind = event.kind
        if kind == EventKind.NODE_MOVED:
            name = packed.name_of(event.node)
            if name is not None:
                events.append(Moved(name, event.x, event.y))
        elif kind == EventKind.NODE_CLICKED:
            name = packed.name_of(event.node)
            if name is not None:
                events.append(Clicked(name, bool(event.extend)))
        elif kind == EventKind.NODE_ACTIVATED:
            name = packed.name_of(event.node)
            if name is not None:
                events.append(Activated(name))
        elif kind == EventKind.CONTEXT_MENU:
            over = packed.name_of(event.node) if event.node >= 0 else None
            events.append(MenuRequested(over, event.x, event.y))
        elif kind == EventKind.EDGE_ADDED:
            wire = _resolve_wire(packed, event.from_node, event.to_node, event.to_attr)
            if wire is not None:
                events.append(wire)
        elif kind == EventKind.EDGE_REMOVED:
            consumer = packed.view_of(event.to_node)
            if consumer is not None and 0 <= event.to_attr < len(consumer.inputs):
                events.append(Unwired(consumer.name, consumer.inputs[event.to_attr]))
        elif kind == EventKind.EDGE_REFUSED:
            events.append(Refused(event.error))
    return events


def _resolve_wire(
    packed: Packed, from_node: int, to_node: int, to_attr: int
) -> Wired | None:
    producer = packed.view_of(from_node)
    consumer = packed.view_of(to_node)
    if producer is None or consumer is None:
        return None
    if not (0 <= to_attr < len(consumer.inputs)):
        return None
    return Wired(producer.name, consumer.name, consumer.inputs[to_attr])
