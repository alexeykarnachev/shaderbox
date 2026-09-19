"""The graph canvas's per-document state and the pure pieces of its gestures (092 D2, D13).

Transient: nothing here is persisted, and nothing off-draw writes it. What IS persisted -- a
pass's position -- reaches disk only through `ProjectSession.set_pass_positions`, and the
drag's state machine below is what makes "one save per gesture" a fact a test can assert:
`update` returns nothing to write, `commit` is the only thing that does.
"""

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace

from shaderbox.document import Document, sampler_names
from shaderbox.graph_canvas.adapter import (
    CanvasEdge,
    CanvasNode,
    ScopedView,
    pass_key,
    pass_node,
)
from shaderbox.pass_graph import Port, Wiring, group_boundary, node_ports
from shaderbox.theme import SIZE
from shaderbox.ui_primitives import InlineInput

Position = tuple[float, float]
# A wire's identity anywhere on the canvas: the consumer pass and the sampler the wire
# terminates at. A sampler has one source (072), so the pair names one wire in the whole
# document, and it is what `App.unwire` takes.
WireId = tuple[str, str]


@dataclass
class GraphViewState:
    """What the graph canvas keeps between frames, beyond the library's own.

    The library owns the geometry, the hover and the in-flight gesture (098);
    of the camera it owns only the zoom, since panning is the host's. What is
    left here is what shaderbox decides: which scope the tabs are showing,
    which passes are selected, and the Group prompt. The hit rects, the hover
    fields and the drag machines this held for the imgui canvas are gone with
    it.
    """

    # "" is the root; a group name is that group's tab. Revalidated every frame.
    scope: str = ""
    selection: set[str] = field(default_factory=set)
    # One-shot: the first canvas frame at a nonzero size fits the view; a scope change
    # clears it so the new scope fits once too.
    fitted: bool = False
    # The Group name prompt: the shared inline input, whose `target` is unused here.
    group_input: InlineInput = field(default_factory=InlineInput)


def revalidated_scope(scope: str, groups: Collection[str]) -> str:
    """The scope to draw this frame: the requested one while some pass still carries it,
    else the root (the last member can leave from inside the tab)."""
    return scope if scope in groups else ""


def node_size(port_count: int, box: bool) -> tuple[float, float]:
    """A node's canvas-space size at zoom 1: the picture, the name, and one row per port."""
    width = float(SIZE.GRAPH_NODE_W + (SIZE.GRAPH_BOX_EXTRA_W if box else 0))
    height = float(
        SIZE.GRAPH_THUMB_INSET + SIZE.GRAPH_THUMB + SIZE.GRAPH_NAME_H + SIZE.GRAPH_PAD
    )
    if port_count:
        height += (
            SIZE.GRAPH_PORT_TOP
            + port_count * SIZE.GRAPH_PORT_ROW
            + SIZE.GRAPH_PORT_BOTTOM
        )
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


# ---- the scoped picture (092 D4, D5) --------------------------------------------------------


def _ghost_column(
    names: Sequence[str],
    ports: Mapping[str, Sequence[Port]],
    sizes: Mapping[str, tuple[float, float]],
    prefix: str,
    edge: float,
    right_aligned: bool,
    top: float,
) -> list[CanvasNode]:
    """One column of ghosts, stacked downward from `top`.

    Stacked rather than drawn at their own positions: two out-of-scope passes
    can sit anywhere relative to each other, including on top of one another,
    and a ghost exists to show WHERE a wire goes rather than where the pass
    lives.

    `edge` is the column's boundary and `right_aligned` says which side of it
    the nodes hang from -- a feeder column ends at the members' left edge, so
    its nodes are placed by their right side and widths differ.
    """
    out: list[CanvasNode] = []
    y = top
    for name in names:
        port_list = tuple(ports.get(name, ()))
        width, height = sizes.get(name, node_size(len(port_list), False))
        x = edge - width if right_aligned else edge
        # The ghost's KEY carries the column, because a pass that both feeds
        # and reads the group is drawn in both and two nodes cannot share one.
        ghost = pass_node(name, (x, y), port_list, ghost=True)
        out.append(replace(ghost, key=f"{prefix}{name}"))
        y += height + float(SIZE.GRAPH_GAP_Y)
    return out


def _group_scope(
    scope: str,
    order: Sequence[str],
    groups: Mapping[str, str],
    ports: Mapping[str, Sequence[Port]],
    sizes: Mapping[str, tuple[float, float]],
    positions: Mapping[str, Position],
    wiring: Wiring,
) -> ScopedView:
    """Inside one group's tab: its members, plus a ghost per outside neighbour.

    A feeder ghost goes to the LEFT of every member and a reader ghost to the
    right, so no wire runs backward through the picture. A pass that both
    feeds and reads the group is drawn in both columns (092 D4) -- one node
    per direction, which is what keeps the reading left-to-right.
    """
    members = [name for name in order if groups.get(name, "") == scope]
    inside = set(members)
    nodes: list[CanvasNode] = [
        pass_node(name, positions.get(name, (0.0, 0.0)), ports.get(name, ()))
        for name in members
    ]
    left = min((positions[m][0] for m in members if m in positions), default=0.0)
    right = max(
        (
            positions[m][0] + sizes.get(m, (0.0, 0.0))[0]
            for m in members
            if m in positions
        ),
        default=0.0,
    )
    top = min((positions[m][1] for m in members if m in positions), default=0.0)
    rank = {name: i for i, name in enumerate(order)}
    feeders = sorted(
        {
            source
            for m in members
            for source in wiring.get(m, {}).values()
            if source not in inside
        },
        key=lambda n: rank.get(n, 0),
    )
    readers = sorted(
        {
            name
            for name in order
            if name not in inside
            and any(s in inside for s in wiring.get(name, {}).values())
        },
        key=lambda n: rank.get(n, 0),
    )
    gap = float(SIZE.GRAPH_GAP_X)
    nodes += _ghost_column(feeders, ports, sizes, "g:in:", left - gap, True, top)
    nodes += _ghost_column(readers, ports, sizes, "g:out:", right + gap, False, top)

    edges: list[CanvasEdge] = []
    for name in members:
        for slot, port in enumerate(ports.get(name, ())):
            source = port.source
            if source is None:
                continue
            src = pass_key(source) if source in inside else f"g:in:{source}"
            edges.append(CanvasEdge(src, 0, pass_key(name), slot, name, port.sampler))
    for name in readers:
        for slot, port in enumerate(ports.get(name, ())):
            if port.source in inside:
                edges.append(
                    CanvasEdge(
                        pass_key(port.source),
                        0,
                        f"g:out:{name}",
                        slot,
                        name,
                        port.sampler,
                    )
                )
    return ScopedView(tuple(nodes), tuple(edges))


def _root_scope(
    order: Sequence[str],
    groups: Mapping[str, str],
    ports: Mapping[str, Sequence[Port]],
    positions: Mapping[str, Position],
    wiring: Wiring,
    output: str,
) -> ScopedView:
    """The root tab: every ungrouped pass, and one BOX per group.

    A box stands in for its members: its input ports are the boundary's --
    every member slot reading outside the group or nothing -- and its outputs
    are the members the outside reads, plus the bundle. So the root shows the
    document's shape without showing the inside of anything.
    """
    nodes: list[CanvasNode] = [
        pass_node(name, positions.get(name, (0.0, 0.0)), ports.get(name, ()))
        for name in order
        if not groups.get(name, "")
    ]
    box_of: dict[str, str] = {
        name: f"b:{groups[name]}" for name in order if groups.get(name, "")
    }
    # Where a wire lands ON a box: which boundary slot carries this member's
    # sampler, and which output dot stands for this member.
    box_slots: dict[str, dict[tuple[str, str], int]] = {}
    box_outs: dict[str, dict[str, int]] = {}
    for group in group_names_in_order(order, dict(groups)):
        members = [name for name in order if groups.get(name, "") == group]
        boundary = group_boundary(members, ports, wiring, output)
        key = f"b:{group}"
        placed = [positions[m] for m in members if m in positions]
        x = min((p[0] for p in placed), default=0.0)
        y = min((p[1] for p in placed), default=0.0)
        box_slots[key] = {
            (bp.member, bp.port.sampler): i for i, bp in enumerate(boundary.inputs)
        }
        box_outs[key] = {m: i for i, m in enumerate(boundary.outputs)}
        nodes.append(
            CanvasNode(
                key=key,
                name=group,
                pos=(x, y),
                ports=tuple(bp.port for bp in boundary.inputs),
                labels=tuple(bp.label for bp in boundary.inputs),
                owners=tuple(bp.member for bp in boundary.inputs),
                outputs=tuple(boundary.outputs),
                preview=boundary.bundle,
                is_box=True,
                group=group,
                members=tuple(members),
            )
        )

    edges: list[CanvasEdge] = []
    for name in order:
        for slot, port in enumerate(ports.get(name, ())):
            source = port.source
            if source is None or source == name:
                continue
            if source in box_of:
                src_key = box_of[source]
                src_slot = box_outs[src_key].get(source)
                if src_slot is None:
                    continue
            else:
                src_key, src_slot = pass_key(source), 0
            if name in box_of:
                dst_key = box_of[name]
                dst_slot = box_slots[dst_key].get((name, port.sampler))
                if dst_slot is None:
                    # Both ends inside one box: an edge the root does not draw.
                    continue
            else:
                dst_key, dst_slot = pass_key(name), slot
            if src_key == dst_key:
                continue
            edges.append(
                CanvasEdge(src_key, src_slot, dst_key, dst_slot, name, port.sampler)
            )
    return ScopedView(tuple(nodes), tuple(edges))


def scoped_view(
    scope: str,
    order: Sequence[str],
    groups: Mapping[str, str],
    ports: Mapping[str, Sequence[Port]],
    positions: Mapping[str, Position],
    wiring: Wiring,
    output: str,
) -> ScopedView:
    """What the canvas draws for one scope: `""` is the root, else a group's tab.

    The scope is the whole reason the tab row exists, and it is resolved HERE
    rather than in the adapter: which passes are visible, what a group looks
    like from outside and what the outside looks like from within a group are
    all questions about shaderbox's document, and the adapter's job starts
    once they are answered.
    """
    sizes = {name: node_size(len(ports.get(name, ())), False) for name in order}
    if scope:
        return _group_scope(scope, order, groups, ports, sizes, positions, wiring)
    return _root_scope(order, groups, ports, positions, wiring, output)
