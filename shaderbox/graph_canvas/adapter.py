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
    _RGBA,
    EdgeSpec,
    EventKind,
    Gesture,
    NodeSpec,
    PinFill,
    PinShape,
    PortSpec,
    PreviewFit,
    Result,
    Theme,
    Widget,
    default_theme,
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


def theme_from(
    canvas: RGBA,
    surface: RGBA,
    grid: RGBA,
    border: RGBA,
    text: RGBA,
    text_dim: RGBA,
    text_bright: RGBA,
    accent: RGBA,
    pin: RGBA,
    wire_outline: RGBA,
    wire_invalid: RGBA,
    port_input: RGBA,
    port_output: RGBA,
    port_both: RGBA,
    control: RGBA,
) -> Theme:
    """The library's palette with a host's colours over it.

    INHERITED, not built: the shading scalars -- how far a depth level
    lifts, how a chamfer catches light, how a shadow falls -- are tuned
    against a dark canvas, which shaderbox also is, and a theme constructed
    from zero sets every one of them to 0 and flattens the canvas. Taking
    them from `default_theme()` also means an upstream retune arrives here
    for free, where copied constants would silently fight it.

    The three ROLE colours are what a row's background is drawn from, so
    they are what says at a glance whether a row takes a wire in, sends one
    out, or does both. They take three separate arguments because they carry
    three different meanings: collapsing them onto one colour does not
    "unify" the palette, it deletes the distinction -- and pointing all
    three at a background grey, which this did for one commit, paints every
    row the colour of the thing behind it and turns the whole canvas into
    grey slabs.

    `surface` is the node BODY against `canvas` behind it. It must be
    LIGHTER: the library's shading lifts a node off its background, and a
    surface darker than the canvas makes every node a hole instead.

    `pin` colours the PIN DOT and nothing else. A WIRE takes the role of the
    port it leaves -- the library's `wire_color` is `shade(role, wire_lift)`
    -- so one signal keeps one colour end to end, and no argument here sets
    it. This parameter was called `wire` and named a thing it does not reach.

    `wire_outline` is the dark run UNDER a wire's core, and it has to be
    darker than everything the wire crosses: a wire runs over nodes, over the
    canvas and over other wires, so it cannot borrow contrast from any one of
    them. A mid-grey border colour here inverts it into a light halo.
    """
    theme = default_theme()
    theme.canvas = _RGBA(*canvas)
    theme.surface = _RGBA(*surface)
    theme.grid = _RGBA(*grid)
    theme.border = _RGBA(*border)
    theme.text = _RGBA(*text)
    theme.text_dim = _RGBA(*text_dim)
    theme.text_bright = _RGBA(*text_bright)
    theme.accent = _RGBA(*accent)
    theme.input = _RGBA(*port_input)
    theme.output = _RGBA(*port_output)
    theme.both = _RGBA(*port_both)
    theme.pin = _RGBA(*pin)
    theme.control = _RGBA(*control)
    theme.wire_outline = _RGBA(*wire_outline)
    theme.wire_invalid = _RGBA(*wire_invalid)
    return theme


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


def _widget_for(value: tuple[float, ...]) -> Widget:
    """What draws a read-only engine value.

    Measured: a `LABEL` renders `value_text_component(value, 0)` and nothing
    else, so a two-component value loses its second half silently. A `DRAG`
    renders one field per component and, with `read_only`, never takes the
    pointer -- which is the library's own stated reason for that flag.
    """
    if not value:
        return Widget.NONE
    return Widget.LABEL if len(value) == 1 else Widget.DRAG


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
class CanvasNode:
    """One node the canvas will draw, already resolved against the scope.

    A node is no longer one pass: at the root a group collapses into a BOX
    carrying its boundary ports, and inside a group's tab the passes it reads
    from and is read by appear as GHOSTS, so an edge never runs off the edge
    of the picture into nothing.

    `key` is the node's identity on the canvas and `name` is what it is
    called. They differ for every node that is not a plain pass: a box is
    keyed by its group, and a pass that both feeds and reads a group appears
    twice, once in each ghost column, so the pass name alone cannot identify
    a node.

    `owners` names the pass behind each input slot -- itself for a pass, the
    member for a box's boundary port. It is what turns a wire the user drew
    on a box into an `App.unwire` on the member that actually holds it.
    """

    key: str
    name: str
    pos: tuple[float, float]
    ports: tuple[Port, ...]
    labels: tuple[str, ...]
    owners: tuple[str, ...]
    # The member each output dot stands for, top to bottom. A pass has one,
    # its own; a box has one per member the outside reads, plus the bundle.
    outputs: tuple[str, ...]
    # The pass whose picture this node shows -- itself, or a box's bundle.
    preview: str
    is_box: bool = False
    is_ghost: bool = False
    group: str = ""
    members: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CanvasEdge:
    """One wire between two canvas nodes, by node key and slot.

    `owner` and `sampler` are the wire's identity in the DOCUMENT -- the pass
    that holds the read and the sampler it terminates at -- which is what
    survives a box collapsing several passes into one node.
    """

    src_key: str
    src_slot: int
    dst_key: str
    dst_slot: int
    owner: str
    sampler: str


@dataclass(frozen=True, slots=True)
class ScopedView:
    nodes: tuple[CanvasNode, ...]
    edges: tuple[CanvasEdge, ...]


@dataclass(frozen=True, slots=True)
class NodeView:
    """One node as the adapter packed it, kept so an event can be resolved.

    `inputs` is this node's input samplers in the order they were packed,
    which is what an edge's `to_attr` indexes, and `owners` is the pass each
    of those slots belongs to. The two differ only on a BOX, whose slots are
    its members' -- and that difference is the whole reason a wire dropped on
    a box reaches the right pass.

    `outputs` is the same for the other side: the pass each output dot stands
    for, so a wire drawn FROM a box names the member the outside actually
    reads rather than the group.
    """

    key: str
    name: str
    inputs: tuple[str, ...]
    owners: tuple[str, ...]
    outputs: tuple[str, ...]
    is_box: bool = False
    is_ghost: bool = False
    group: str = ""
    members: tuple[str, ...] = ()

    def owner_of(self, slot: int) -> str | None:
        """The pass holding the read at input `slot`."""
        if 0 <= slot < len(self.owners):
            return self.owners[slot]
        return None

    def sampler_of(self, slot: int) -> str | None:
        if 0 <= slot < len(self.inputs):
            return self.inputs[slot]
        return None

    def output_of(self, attr: int) -> str | None:
        """The pass an OUTPUT dot stands for, by its index in the node's flat
        attribute list. The outputs are emitted straight after the inputs, so
        the dot's own position is that index minus the input count."""
        slot = attr - len(self.inputs)
        if 0 <= slot < len(self.outputs):
            return self.outputs[slot]
        return None


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

    def key_of(self, index: int) -> str | None:
        """The node's identity on the canvas, which a hover and a selection
        are keyed by. Not the pass name: a box is one node standing for
        several passes, and a ghost is a second node for a pass already
        drawn."""
        view = self.view_of(index)
        return None if view is None else view.key


def pass_key(name: str) -> str:
    """The canvas key of the node standing for one pass.

    A function rather than an f-string at each call site: `hovered`,
    `selected` and `output` are all keyed by it, and a node is not always a
    pass, so the namespace has to be explicit somewhere.
    """
    return f"p:{name}"


def pass_node(
    name: str,
    pos: tuple[float, float] = (0.0, 0.0),
    ports: Sequence[Port] = (),
    ghost: bool = False,
) -> CanvasNode:
    """One plain node standing for one pass: its own ports, its own picture.

    The shape every node had before scopes existed, and still the shape of
    every node that is not a box. A GHOST is the same node drawn for context
    outside the current scope -- faded, dashed, refusing every gesture.
    """
    return CanvasNode(
        key=pass_key(name),
        name=name,
        pos=pos,
        ports=tuple(ports),
        labels=tuple(p.sampler for p in ports),
        owners=tuple(name for _ in ports),
        outputs=(name,),
        preview=name,
        is_ghost=ghost,
    )


def flat_view(
    order: Sequence[str],
    ports: Mapping[str, Sequence[Port]],
    positions: Mapping[str, tuple[float, float]],
    ghosts: frozenset[str] = frozenset(),
) -> ScopedView:
    """Every pass as its own node, with the wiring between them.

    The whole graph with nothing collapsed and nothing hidden -- which is
    what a host with no grouping of its own wants, and what shaderbox's root
    scope reduces to when the document has no groups.
    """
    nodes = tuple(
        pass_node(
            name, positions.get(name, (0.0, 0.0)), ports.get(name, ()), name in ghosts
        )
        for name in order
    )
    known = set(order)
    edges: list[CanvasEdge] = []
    for name in order:
        for slot, port in enumerate(ports.get(name, ())):
            if port.kind != "wired" or port.source is None:
                continue
            if port.source not in known or port.source == name:
                continue
            edges.append(
                CanvasEdge(
                    pass_key(port.source), 0, pass_key(name), slot, name, port.sampler
                )
            )
    return ScopedView(nodes, tuple(edges))


def pack_nodes(
    view: ScopedView,
    previews: Mapping[str, tuple[int, int, int]],
    output: str,
    engine: Mapping[str, Sequence[tuple[str, tuple[float, ...]]]] | None = None,
    hovered: str = "",
    selected: frozenset[str] = frozenset(),
    palette: NodePalette | None = None,
) -> Packed:
    """Turn one scope's resolved nodes into a frame the library can draw.

    `view` has already answered every question about the DOCUMENT -- which
    passes this scope shows, what a group looks like collapsed, where the
    ghosts stand. What is left here is the mapping onto the library's model,
    which is the same work for a pass, a box and a ghost.

    `previews` maps a PASS to (texture name, width, height) — the real GL
    texture it renders into, which the library samples for the node's
    picture. A node asks for its own `preview` pass, which is the node itself
    for a pass and the bundle member for a box.

    A GHOST is faded, dashed, and refuses every gesture, so a press falls
    through to the canvas rather than being swallowed by something that does
    nothing.

    `engine` gives each pass's engine-driven uniforms as (name, value)
    pairs -- `u_time` and its siblings. They are CONTROL rows carrying a
    read-only widget, which is the case the library's `read_only` field says
    it exists for: the engine writes them and no wire can, so a pin would
    offer a connection the document cannot express, and an editable widget
    would offer an edit the engine overwrites next frame. A box shows its
    bundle's, since that is the picture it is already showing.

    `hovered` and `selected` decide the border, and both are keyed by NODE
    KEY rather than by pass name: at the root a group's box is one node and
    its members are none, so a pass name would highlight nothing.

    The nodes and their attributes are built in ONE pass, because each node
    names a contiguous run of the flat attribute array. The edges are
    resolved against the SAME ordering, so a wire's endpoints index what was
    packed rather than what the document happens to iterate.
    """
    colors = palette or _NEUTRAL_PALETTE
    index_of: dict[str, int] = {node.key: i for i, node in enumerate(view.nodes)}
    # Where each node's outputs START in its own attribute list; a box has
    # several, one per member the outside reads.
    output_base: dict[str, int] = {}
    nodes: list[NodeSpec] = []
    views: list[NodeView] = []

    for node in view.nodes:
        specs: list[PortSpec] = []
        for port, label in zip(node.ports, node.labels, strict=True):
            shape, fill = _PIN_BY_KIND.get(port.kind, (PinShape.DOT, PinFill.UNSET))
            specs.append(
                PortSpec(label=label, is_input=True, pin_shape=shape, pin_fill=fill)
            )
        output_base[node.key] = len(specs)
        for member in node.outputs:
            # A box names each dot after the member it stands for; a pass has
            # one dot and the node's own title already says whose it is.
            specs.append(
                PortSpec(label=member if node.is_box else "out", is_input=False)
            )

        # An engine uniform is a CONTROL, which is what the library's own
        # model calls a row the user cannot wire. That is not only semantics:
        # the row's background comes from its KIND, so a control is drawn in
        # the neutral control tone where an input is drawn green -- measured,
        # (0.33, 0.34, 0.42) against (0.29, 0.45, 0.35). Packing these as
        # inputs made a builtin look like a free port, which is exactly what
        # they are not.
        for label, value in (engine or {}).get(node.preview, ()):
            specs.append(
                PortSpec(
                    label=label,
                    # A LABEL draws component 0 ONLY, so a vec2 like
                    # `u_resolution` would show half of itself. A read-only
                    # DRAG draws one field per component and takes no input,
                    # which is the same display without the lie.
                    widget=_widget_for(value),
                    value=value,
                    read_only=True,
                    # NO pin. `control` alone carries neither side bit, and
                    # the library draws a pin for Input or Output only.
                    #
                    # The pin was not merely misleading: a wire dropped on it
                    # produced a real `Edge_Added` that `read_events` then
                    # discarded, because an engine row has no sampler to
                    # resolve. So the wire followed the pointer and vanished
                    # on release with no refusal shown -- the canvas appeared
                    # to ignore the user.
                    is_input=False,
                    control=True,
                    color=colors.engine_uniform,
                )
            )

        tex, width, height = previews.get(node.preview, (0, 0, 0))
        nodes.append(
            NodeSpec(
                id=node_id(node.key),
                title=node.name,
                x=node.pos[0],
                y=node.pos[1],
                ports=specs,
                preview_tex=tex,
                preview_w=width,
                preview_h=height,
                preview_aspect=(width / height) if width and height else 1.0,
                preview_fit=PreviewFit.CONTAIN,
                fade=0.6 if node.is_ghost else 0.0,
                dashed=node.is_ghost,
                border=_border_of(node.key, hovered, selected, node.is_ghost, colors),
                border_scale=_border_scale_of(node.key, output, hovered, selected),
                accepts=int(Gesture.NONE) if node.is_ghost else 0,
            )
        )
        views.append(
            NodeView(
                key=node.key,
                name=node.name,
                inputs=tuple(p.sampler for p in node.ports),
                owners=node.owners,
                outputs=node.outputs,
                is_box=node.is_box,
                is_ghost=node.is_ghost,
                group=node.group,
                members=node.members,
            )
        )

    edges: list[EdgeSpec] = []
    for edge in view.edges:
        producer = index_of.get(edge.src_key)
        consumer = index_of.get(edge.dst_key)
        if producer is None or consumer is None:
            continue
        base = output_base.get(edge.src_key)
        if base is None:
            continue
        edges.append(
            EdgeSpec(
                id=edge_id(edge.owner, edge.sampler),
                from_node=producer,
                from_attr=base + edge.src_slot,
                to_node=consumer,
                to_attr=edge.dst_slot,
            )
        )

    return Packed(tuple(nodes), tuple(edges), tuple(views))


@dataclass(frozen=True, slots=True)
class Moved:
    """A node dragged. `key` is the canvas node, `name` what it is called.

    Both, because a move is written back per PASS and a box moves several at
    once: the host needs the key to know it was a box, and the members to
    know which passes to write.
    """

    key: str
    name: str
    x: float
    y: float
    members: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Clicked:
    """`members` is empty for a pass and the group's passes for a box, so a
    click on a box selects what it stands for rather than a pass that does
    not exist by that name."""

    key: str
    name: str
    extend: bool
    is_box: bool = False
    members: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Activated:
    """A double-click. On a box this is "open the group"; on a pass it is the
    library's own activate, which shaderbox does not otherwise use."""

    key: str
    name: str
    is_box: bool = False
    group: str = ""


@dataclass(frozen=True, slots=True)
class MenuRequested:
    """The node the pointer was over, or `None` for the canvas's own menu.

    `(x, y)` is in SCREEN space. `is_box` and `group` are carried so the host
    can pick the box's menu without looking the node up again -- at the root
    a group IS a node, and there is no pass by that name to look up.
    """

    key: str | None
    name: str | None
    x: float
    y: float
    is_box: bool = False
    group: str = ""


@dataclass(frozen=True, slots=True)
class Wired:
    """A wire the user landed: `consumer` reads `producer` through `sampler`.

    Both ends are PASS names, resolved through the node's owners: a wire
    dropped on a box lands on the member that actually holds the read, and
    one drawn from a box comes from the member the outside reads.
    """

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
    """This frame's events, resolved from indices back to the document.

    An event about a node the frame no longer holds is dropped rather than
    guessed at: the library reports against the array it was handed, and a
    host that renamed or deleted between frames would otherwise act on
    whatever now sits at that index.

    A GHOST is packed refusing every gesture, so the library reports nothing
    about one; the guard here is belt and braces rather than a live path.
    """
    events: list[GraphEvent] = []
    for i in range(result.event_count):
        event = result.events[i]
        kind = event.kind
        node = packed.view_of(event.node) if event.node >= 0 else None
        if kind == EventKind.NODE_MOVED:
            if node is not None and not node.is_ghost:
                events.append(
                    Moved(node.key, node.name, event.x, event.y, node.members)
                )
        elif kind == EventKind.NODE_CLICKED:
            if node is not None and not node.is_ghost:
                events.append(
                    Clicked(
                        node.key,
                        node.name,
                        bool(event.extend),
                        node.is_box,
                        node.members,
                    )
                )
        elif kind == EventKind.NODE_ACTIVATED:
            if node is not None and not node.is_ghost:
                events.append(Activated(node.key, node.name, node.is_box, node.group))
        elif kind == EventKind.CONTEXT_MENU:
            if node is None:
                events.append(MenuRequested(None, None, event.x, event.y))
            elif not node.is_ghost:
                events.append(
                    MenuRequested(
                        node.key, node.name, event.x, event.y, node.is_box, node.group
                    )
                )
        elif kind == EventKind.EDGE_ADDED:
            wire = _resolve_wire(
                packed,
                event.from_node,
                event.from_attr,
                event.to_node,
                event.to_attr,
            )
            if wire is not None:
                events.append(wire)
        elif kind == EventKind.EDGE_REMOVED:
            consumer = packed.view_of(event.to_node)
            if consumer is not None:
                owner = consumer.owner_of(event.to_attr)
                sampler = consumer.sampler_of(event.to_attr)
                if owner is not None and sampler is not None:
                    events.append(Unwired(owner, sampler))
        elif kind == EventKind.EDGE_REFUSED:
            events.append(Refused(event.error))
    return events


def _resolve_wire(
    packed: Packed, from_node: int, from_attr: int, to_node: int, to_attr: int
) -> Wired | None:
    """Both ends as PASS names.

    The producer is read from the output DOT the wire left rather than from
    the node: a box has one dot per member the outside reads, so the node's
    name is the group and only the dot says which pass.
    """
    producer = packed.view_of(from_node)
    consumer = packed.view_of(to_node)
    if producer is None or consumer is None:
        return None
    if producer.is_ghost or consumer.is_ghost:
        return None
    owner = consumer.owner_of(to_attr)
    sampler = consumer.sampler_of(to_attr)
    source = producer.output_of(from_attr)
    if owner is None or sampler is None or source is None:
        return None
    return Wired(source, owner, sampler)
