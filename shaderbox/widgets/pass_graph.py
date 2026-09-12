"""The graph canvas (092): a document's passes as nodes on a pannable, zoomable draw list.

A node is a pass's live picture, its name, one input port per sampler its compiled program
declares (`pass_graph.node_ports`) and an output dot; a wire is a read from the effective
wiring. At the root every group is one BOX whose ports are the group's boundary edges
(`pass_graph.group_boundary`); a group's own tab shows its members with the outside passes
they touch as GHOSTS. Nothing here is a second source of truth: positions are the pass entry's
(or the rank layout's, for a pass never placed), edges are the wiring, ports are the program.

Drawn on one `ImDrawList` inside one child, hit-tested with `invisible_button`s: the canvas
background first, then every node, each level declaring `set_next_item_allow_overlap()` so the
later item wins (imgui gives an overlapping hit to the EARLIEST item unless it allows it). A
context menu anchors with `begin_popup_context_item(None)` -- an explicit id on one shared
child fires on a right-click anywhere in it. Every write a gesture makes goes through an
`App` verb, never a session call from here, so each refusal is testable without a window.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Literal

from imgui_bundle import imgui

from shaderbox.app import App
from shaderbox.core import Pass
from shaderbox.document import Document
from shaderbox.pass_graph import (
    PassEntry,
    Port,
    Wiring,
    cycle_edges,
    evaluation_order,
    graph_ranks,
    group_boundary,
    plan_passes,
    rank_layout,
    strip_order,
)
from shaderbox.project_session import compile_pending_passes
from shaderbox.theme import COLOR, SIZE, SPACE, fade, group_tint
from shaderbox.ui_primitives import (
    context_menu_style,
    primary_button,
    standard_button,
    text_tab_row,
)
from shaderbox.widgets.graph_state import (
    GraphViewState,
    NodeDrag,
    WireDrag,
    group_names_in_order,
    node_size,
    node_sizes,
    ports_of,
    revalidated_scope,
)
from shaderbox.widgets.pass_list import pass_menu_items

NodeKind = Literal["pass", "box", "ghost"]

_CYCLE_PREFIX = "passes form a cycle"
# The root tab's label when the document's name is empty or collides with a group's.
_ROOT_LABELS = ("document", "root", "all")
_FIT_MARGIN = float(SPACE.LG)
_ZOOM_STEP = 1.1
# A consumer at least this far right of its producer (canvas units) gets a plain curve;
# closer or leftward, the wire rides the bus. One constant, read in one coordinate space.
_MIN_DIRECT_DX = 24.0
# The bezier's horizontal control offset, as a fraction of the run.
_BEZIER_BOW = 0.45
# The port dot's inner shapes, as fractions of its radius: the NoSource centre, the prev
# inner ring, the media square's half side.
_NONE_CORE = 0.45
_PREV_INNER = 0.5
_MEDIA_HALF = 0.8


@dataclass(frozen=True)
class _Node:
    key: str
    name: str
    kind: NodeKind
    pos: tuple[float, float]
    size: tuple[float, float]
    ports: tuple[Port, ...]
    labels: tuple[str, ...]
    # Output dots, top to bottom: a pass has one (its name); a box one per member the outside
    # reads plus the bundle output, the hollow ones being unread.
    outputs: tuple[tuple[str, bool], ...]
    # The pass each input slot belongs to: the node itself, or a box's member.
    owners: tuple[str, ...]
    texture_glo: int | None
    texture_size: tuple[int, int]
    group: str = ""
    runs: int = 1
    error: bool = False
    uncompiled: bool = False
    stale: bool = False
    members: tuple[str, ...] = ()
    bundle: str = ""


@dataclass(frozen=True)
class _Edge:
    src_key: str
    src_slot: int
    dst_key: str
    dst_slot: int
    on_cycle: bool
    dim: bool
    span: int


@dataclass
class _View:
    nodes: dict[str, _Node] = field(default_factory=dict)
    edges: list[_Edge] = field(default_factory=list)
    positions: dict[str, tuple[float, float]] = field(default_factory=dict)


# ---- the derived picture ------------------------------------------------------------------


def _positions(
    document: Document,
    wiring: Wiring,
    groups: dict[str, str],
    sizes: dict[str, tuple[float, float]],
    overrides: Mapping[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Place every pass on the canvas without writing anything.

    A stored position wins; a pass never placed takes the rank layout's (092 D6); a drag in
    flight overrides both through `overrides`.
    """
    entries = document.graph.passes
    placed = {
        name: entries[name].position
        for name in document.passes
        if name in entries and entries[name].position is not None
    }
    stored: dict[str, tuple[float, float]] = {
        name: (position[0], position[1])
        for name, position in placed.items()
        if position is not None
    }
    unplaced = [name for name in document.passes if name not in stored]
    laid = rank_layout(
        wiring,
        unplaced,
        groups,
        sizes,
        stored,
        float(SIZE.GRAPH_GAP_X),
        float(SIZE.GRAPH_GAP_Y),
    )
    return {**stored, **laid, **overrides}


def _pass_node(
    key: str,
    name: str,
    kind: NodeKind,
    render_pass: Pass,
    entry: PassEntry,
    pos: tuple[float, float],
    ports: Sequence[Port],
    stale: bool,
) -> _Node:
    return _Node(
        key=key,
        name=name,
        kind=kind,
        pos=pos,
        size=node_size(len(ports), False),
        ports=tuple(ports),
        labels=tuple("prev" if p.kind == "prev" else p.sampler for p in ports),
        outputs=((name, False),),
        owners=tuple(name for _ in ports),
        texture_glo=render_pass.canvas.texture.glo,
        texture_size=render_pass.canvas.texture.size,
        group=entry.group,
        runs=entry.iterations,
        error=bool(render_pass.compile_unit.errors),
        uncompiled=render_pass.program is None and not render_pass.compile_unit.errors,
        stale=stale,
    )


def _build_view(
    document: Document, scope: str, overrides: Mapping[str, tuple[float, float]]
) -> _View:
    """The nodes and edges to draw for `scope` ("" = the root), from the flat document."""
    wiring = document.effective_wiring()
    order = strip_order(document.passes, wiring)
    entries = document.graph.passes
    groups = {name: entries.get(name, PassEntry()).group for name in order}
    ports = ports_of(document, wiring)
    sizes = node_sizes(ports)
    positions = _positions(document, wiring, groups, sizes, overrides)
    output = document.graph.output
    live = (
        set(evaluation_order(wiring, output)) or {output}
        if output in document.passes
        else set(document.passes)
    )
    errors = plan_passes(wiring)[1]
    cycle_pairs = cycle_edges(errors)
    culprits = {e.pass_name for e in errors if e.message.startswith(_CYCLE_PREFIX)}
    ranks = graph_ranks(wiring)
    view = _View(positions=positions)

    def pass_key(name: str) -> str:
        return f"p:{name}"

    if scope:
        members = [name for name in order if groups[name] == scope]
        inside = set(members)
        for name in members:
            view.nodes[pass_key(name)] = _pass_node(
                pass_key(name),
                name,
                "pass",
                document.passes[name],
                entries.get(name, PassEntry()),
                positions[name],
                ports[name],
                name not in live,
            )
        # Ghosts: a feeder to the left of the members, a reader to the right; a pass that does
        # both is drawn twice, so no edge runs backward through the members (092 D4).
        left = min((positions[m][0] for m in members), default=0.0)
        right = max((positions[m][0] + sizes[m][0] for m in members), default=0.0)
        feeders = sorted(
            {
                source
                for m in members
                for source in wiring.get(m, {}).values()
                if source not in inside
            },
            key=order.index,
        )
        readers = sorted(
            {
                name
                for name in order
                if name not in inside
                and any(s in inside for s in wiring.get(name, {}).values())
            },
            key=order.index,
        )
        # Each ghost column is stacked from the members' top, so two ghosts cannot share a
        # rect whatever their own positions are.
        top = min((positions[m][1] for m in members), default=0.0)
        y = top
        for name in feeders:
            key = f"g:in:{name}"
            x = left - sizes[name][0] - float(SIZE.GRAPH_GAP_X)
            view.nodes[key] = _pass_node(
                key,
                name,
                "ghost",
                document.passes[name],
                entries.get(name, PassEntry()),
                (x, y),
                ports[name],
                name not in live,
            )
            y += sizes[name][1] + float(SIZE.GRAPH_GAP_Y)
        y = top
        for name in readers:
            key = f"g:out:{name}"
            x = right + float(SIZE.GRAPH_GAP_X)
            view.nodes[key] = _pass_node(
                key,
                name,
                "ghost",
                document.passes[name],
                entries.get(name, PassEntry()),
                (x, y),
                ports[name],
                name not in live,
            )
            y += sizes[name][1] + float(SIZE.GRAPH_GAP_Y)
        # Edges: into members from members or feeder ghosts; out of members into reader ghosts.
        for name in members:
            for slot, port in enumerate(ports[name]):
                source = port.source
                if source is None or source == name:
                    continue  # a self-read is the loop drawn on the node
                src_key = pass_key(source) if source in inside else f"g:in:{source}"
                view.edges.append(
                    _Edge(
                        src_key,
                        0,
                        pass_key(name),
                        slot,
                        (source, name) in cycle_pairs,
                        name not in live or source not in live,
                        ranks[name] - ranks[source],
                    )
                )
        for name in readers:
            for slot, port in enumerate(ports[name]):
                if port.source in inside:
                    view.edges.append(
                        _Edge(
                            pass_key(port.source),
                            0,
                            f"g:out:{name}",
                            slot,
                            (port.source, name) in cycle_pairs,
                            name not in live or port.source not in live,
                            ranks[name] - ranks[port.source],
                        )
                    )
        return view

    # The root: ungrouped passes as nodes, one box per group.
    box_of: dict[str, str] = {}
    for name in order:
        if groups[name]:
            box_of[name] = f"b:{groups[name]}"
    for name in order:
        if groups[name]:
            continue
        view.nodes[pass_key(name)] = _pass_node(
            pass_key(name),
            name,
            "pass",
            document.passes[name],
            entries.get(name, PassEntry()),
            positions[name],
            ports[name],
            name not in live,
        )
    box_slots: dict[str, dict[tuple[str, str], int]] = {}
    box_outs: dict[str, dict[str, int]] = {}
    for group in group_names_in_order(order, groups):
        members = [name for name in order if groups[name] == group]
        boundary = group_boundary(members, ports, wiring, output)
        key = f"b:{group}"
        x = min(positions[m][0] for m in members)
        y = min(positions[m][1] for m in members)
        bundle = boundary.bundle
        read_outside = {
            source
            for reader, row in wiring.items()
            if groups.get(reader, "") != group
            for source in row.values()
            if source in members
        }
        outputs = tuple((m, m not in read_outside) for m in boundary.outputs)
        box_ports = tuple(bp.port for bp in boundary.inputs)
        box_slots[key] = {
            (bp.member, bp.port.sampler): i for i, bp in enumerate(boundary.inputs)
        }
        box_outs[key] = {m: i for i, (m, _) in enumerate(outputs)}
        render_pass = document.passes[bundle]
        view.nodes[key] = _Node(
            key=key,
            name=group,
            kind="box",
            pos=(x, y),
            size=node_size(len(box_ports), True),
            ports=box_ports,
            labels=tuple(bp.label for bp in boundary.inputs),
            outputs=outputs,
            owners=tuple(bp.member for bp in boundary.inputs),
            texture_glo=render_pass.canvas.texture.glo,
            texture_size=render_pass.canvas.texture.size,
            group=group,
            error=any(document.passes[m].compile_unit.errors for m in members)
            or any(m in culprits for m in members),
            stale=all(m not in live for m in members),
            members=tuple(members),
            bundle=bundle,
        )

    def src_of(source: str) -> tuple[str, int] | None:
        if source in box_of:
            key = box_of[source]
            slot = box_outs[key].get(source)
            return (key, slot) if slot is not None else None
        return (pass_key(source), 0)

    for name in order:
        for port in ports[name]:
            source = port.source
            if source is None or source == name:
                continue
            src = src_of(source)
            if src is None:
                continue
            if groups[name]:
                key = box_of[name]
                slot = box_slots[key].get((name, port.sampler))
                if slot is None:
                    continue  # an edge inside the box
                dst = (key, slot)
            else:
                dst = (pass_key(name), ports[name].index(port))
            if src[0] == dst[0]:
                continue
            view.edges.append(
                _Edge(
                    src[0],
                    src[1],
                    dst[0],
                    dst[1],
                    (source, name) in cycle_pairs,
                    name not in live or source not in live,
                    ranks[name] - ranks[source],
                )
            )
    return view


# ---- geometry ------------------------------------------------------------------------------


@dataclass
class _Xf:
    origin: imgui.ImVec2
    pan: tuple[float, float]
    zoom: float

    def to_screen(self, p: tuple[float, float]) -> tuple[float, float]:
        return (
            self.origin.x + (p[0] - self.pan[0]) * self.zoom,
            self.origin.y + (p[1] - self.pan[1]) * self.zoom,
        )

    def to_canvas(self, s: tuple[float, float]) -> tuple[float, float]:
        return (
            (s[0] - self.origin.x) / self.zoom + self.pan[0],
            (s[1] - self.origin.y) / self.zoom + self.pan[1],
        )


def _thumb_rect(node: _Node) -> tuple[float, float, float, float]:
    x = node.pos[0] + (node.size[0] - SIZE.GRAPH_THUMB) / 2.0
    y = node.pos[1] + SIZE.GRAPH_PAD
    return x, y, x + SIZE.GRAPH_THUMB, y + SIZE.GRAPH_THUMB


def _port_point(node: _Node, slot: int) -> tuple[float, float]:
    y0 = (
        node.pos[1]
        + SIZE.GRAPH_PAD
        + SIZE.GRAPH_THUMB
        + SIZE.GRAPH_NAME_H
        + SIZE.GRAPH_PAD
    )
    return (
        node.pos[0],
        y0
        + SIZE.GRAPH_PORT_TOP
        + slot * SIZE.GRAPH_PORT_ROW
        + SIZE.GRAPH_PORT_ROW / 2.0,
    )


def _out_point(node: _Node, slot: int) -> tuple[float, float]:
    _, y0, _, y1 = _thumb_rect(node)
    if len(node.outputs) <= 1:
        return node.pos[0] + node.size[0], (y0 + y1) / 2.0
    step = (y1 - y0) / (len(node.outputs) + 1)
    return node.pos[0] + node.size[0], y0 + step * (slot + 1)


def _bbox(nodes: Sequence[_Node]) -> tuple[float, float, float, float]:
    xs0 = [n.pos[0] for n in nodes]
    ys0 = [n.pos[1] for n in nodes]
    xs1 = [n.pos[0] + n.size[0] for n in nodes]
    ys1 = [n.pos[1] + n.size[1] for n in nodes]
    return min(xs0), min(ys0), max(xs1), max(ys1)


def _fit(view: GraphViewState, nodes: Sequence[_Node], avail: imgui.ImVec2) -> None:
    if not nodes or avail.x <= 0 or avail.y <= 0:
        return
    x0, y0, x1, y1 = _bbox(nodes)
    w = x1 - x0 + 2 * _FIT_MARGIN
    h = y1 - y0 + 2 * _FIT_MARGIN
    zoom = min(1.0, avail.x / w, avail.y / h)
    zoom = max(SIZE.GRAPH_ZOOM_MIN, min(SIZE.GRAPH_ZOOM_MAX, zoom))
    view.zoom = zoom
    view.pan = (
        x0 - _FIT_MARGIN - (avail.x / zoom - w) / 2.0,
        y0 - _FIT_MARGIN - (avail.y / zoom - h) / 2.0,
    )
    view.fitted = True


# ---- drawing -------------------------------------------------------------------------------


def _u32(color: tuple[float, float, float, float]) -> int:
    return imgui.color_convert_float4_to_u32(color)


def _dashed_rect(
    dl: imgui.ImDrawList,
    p0: tuple[float, float],
    p1: tuple[float, float],
    col: int,
    dash: float,
    thickness: float,
) -> None:
    corners = [p0, (p1[0], p0[1]), p1, (p0[0], p1[1]), p0]
    for a, b in pairwise(corners):
        length = abs(b[0] - a[0]) + abs(b[1] - a[1])
        if length <= 0:
            continue
        ux = (b[0] - a[0]) / length
        uy = (b[1] - a[1]) / length
        t = 0.0
        while t < length:
            end = min(t + dash, length)
            dl.add_line(
                (a[0] + ux * t, a[1] + uy * t),
                (a[0] + ux * end, a[1] + uy * end),
                col,
                thickness,
            )
            t += 2 * dash


def _draw_wire(
    dl: imgui.ImDrawList,
    xf: _Xf,
    a: tuple[float, float],
    b: tuple[float, float],
    bus_y: float,
    col: int,
) -> None:
    z = xf.zoom
    p0 = xf.to_screen(a)
    p3 = xf.to_screen(b)
    thickness = max(1.0, SIZE.GRAPH_WIRE_W * z)
    dx = p3[0] - p0[0]
    if dx >= _MIN_DIRECT_DX * z and bus_y <= 0:
        c = max(_MIN_DIRECT_DX * z, dx * _BEZIER_BOW)
        dl.add_bezier_cubic(
            p0, (p0[0] + c, p0[1]), (p3[0] - c, p3[1]), p3, col, thickness
        )
        return
    gx = SIZE.GRAPH_GAP_X / 2.0 * z
    by = xf.to_screen((0.0, bus_y))[1]
    m0 = (p0[0] + gx + _MIN_DIRECT_DX * z, by)
    m1 = (p3[0] - gx - _MIN_DIRECT_DX * z, by)
    dl.add_bezier_cubic(p0, (p0[0] + gx, p0[1]), (p0[0] + gx, by), m0, col, thickness)
    dl.add_line(m0, m1, col, thickness)
    dl.add_bezier_cubic(m1, (p3[0] - gx, by), (p3[0] - gx, p3[1]), p3, col, thickness)


def _draw_self_loop(
    dl: imgui.ImDrawList, xf: _Xf, node: _Node, slot: int, col: int
) -> None:
    z = xf.zoom
    out = xf.to_screen(_out_point(node, 0))
    port = xf.to_screen(_port_point(node, slot))
    top = xf.to_screen(node.pos)[1] - 12 * z
    dl.add_bezier_cubic(
        out,
        (out[0] + 28 * z, top),
        (port[0] - 28 * z, top),
        port,
        col,
        max(1.0, SIZE.GRAPH_WIRE_W * z),
    )


def _draw_port_dot(
    dl: imgui.ImDrawList, center: tuple[float, float], kind: str, r: float, col: int
) -> None:
    ring = SIZE.GRAPH_PORT_RING_W
    if kind == "wired":
        dl.add_circle_filled(center, r, col)
    elif kind == "none":
        dl.add_circle(center, r, col, 0, ring)
        dl.add_circle_filled(center, r * _NONE_CORE, col)
    elif kind == "prev":
        dl.add_circle(center, r, col, 0, ring)
        dl.add_circle(center, r * _PREV_INNER, col, 0, ring)
    elif kind == "media":
        half = r * _MEDIA_HALF
        dl.add_rect_filled(
            (center[0] - half, center[1] - half),
            (center[0] + half, center[1] + half),
            col,
        )
    else:
        dl.add_circle(center, r, col, 0, ring)


_BADGE_PAD = 3.0
_BADGE_H = 12.0
_BADGE_INSET = 2.0


def _draw_badge(
    dl: imgui.ImDrawList,
    corner: tuple[float, float],
    right_aligned: bool,
    label: str,
    z: float,
    bg: int,
    fg: int,
) -> None:
    """A small pill with a word on it at a picture's corner, in the current font."""
    w = imgui.calc_text_size(label).x + 2 * _BADGE_PAD * z
    x0 = (
        corner[0] - _BADGE_INSET * z - w
        if right_aligned
        else corner[0] + _BADGE_INSET * z
    )
    y0 = corner[1] + _BADGE_INSET * z
    dl.add_rect_filled((x0, y0), (x0 + w, y0 + _BADGE_H * z), bg, _BADGE_PAD * z)
    dl.add_text(
        (x0 + _BADGE_PAD * z, y0 + (_BADGE_H * z - imgui.get_font_size()) / 2),
        fg,
        label,
    )


def _draw_node(
    app: App,
    dl: imgui.ImDrawList,
    xf: _Xf,
    node: _Node,
    is_output: bool,
    selected: bool,
) -> None:
    z = xf.zoom
    p0 = xf.to_screen(node.pos)
    p1 = xf.to_screen((node.pos[0] + node.size[0], node.pos[1] + node.size[1]))
    alpha = COLOR.GRAPH_GHOST_ALPHA if node.kind == "ghost" else 1.0
    tint = group_tint(node.group) if node.kind == "box" else None
    rounding = SIZE.GRAPH_ROUNDING * z
    dl.add_rect_filled(p0, p1, _u32(fade(COLOR.BG_SURFACE, alpha)), rounding)
    if tint is not None:
        dl.add_rect_filled(
            p0, p1, _u32((*tint[:3], COLOR.GROUP_FILL_ALPHA * alpha)), rounding
        )
    if node.error:
        border = COLOR.STATE_ERROR
    elif is_output:
        border = COLOR.ACCENT_PRIMARY
    elif selected:
        border = COLOR.SELECT
    elif tint is not None:
        border = tint
    else:
        border = COLOR.BORDER
    border_col = _u32(fade(border, alpha))
    thickness = 1.5 if (node.error or is_output or selected) else 1.0
    if node.kind == "ghost" or node.uncompiled:
        _dashed_rect(dl, p0, p1, border_col, SIZE.GRAPH_DASH * z, thickness)
    else:
        dl.add_rect(p0, p1, border_col, rounding, thickness)

    # The picture: the pass's own live target, scaled by imgui -- no second render.
    tx0, ty0, tx1, ty1 = _thumb_rect(node)
    s0 = xf.to_screen((tx0, ty0))
    s1 = xf.to_screen((tx1, ty1))
    picture_alpha = alpha * (
        COLOR.GRAPH_STALE_ALPHA if node.error or node.stale else 1.0
    )
    if node.texture_glo is not None and min(node.texture_size) > 0:
        tw, th = node.texture_size
        scale = min((s1[0] - s0[0]) / tw, (s1[1] - s0[1]) / th)
        dw, dh = tw * scale, th * scale
        ix = s0[0] + ((s1[0] - s0[0]) - dw) / 2
        iy = s0[1] + ((s1[1] - s0[1]) - dh) / 2
        dl.add_image_rounded(
            imgui.ImTextureRef(node.texture_glo),
            (ix, iy),
            (ix + dw, iy + dh),
            (0, 1),
            (1, 0),
            _u32(fade(COLOR.WHITE, picture_alpha)),
            SIZE.GRAPH_THUMB_ROUNDING * z,
        )

    # The name, centred under the picture.
    font = app.font_14_bold if not node.stale else app.font_14
    imgui.push_font(font, max(4.0, font.legacy_size * z))
    name_color = (
        tint if tint is not None else COLOR.FG_DORMANT if node.stale else COLOR.FG_TITLE
    )
    text_size = imgui.calc_text_size(node.name)
    name_y = s1[1] + (SIZE.GRAPH_NAME_H * z - text_size.y) / 2.0
    dl.add_text(
        ((p0[0] + p1[0] - text_size.x) / 2.0, name_y),
        _u32(fade(name_color, alpha)),
        node.name,
    )
    imgui.pop_font()

    imgui.push_font(app.font_12, max(4.0, app.font_12.legacy_size * z))
    # Badges on the picture: the run count top-right, a box's member count top-left.
    badge_bg = _u32(fade(COLOR.BG_FRAME, alpha))
    badge_fg = _u32(fade(COLOR.FG_MUTED, alpha))
    if node.runs > 1 and node.kind != "ghost":
        _draw_badge(dl, (s1[0], s0[1]), True, f"x{node.runs}", z, badge_bg, badge_fg)
    if node.kind == "box":
        _draw_badge(
            dl,
            (s0[0], s0[1]),
            False,
            f"{len(node.members)} passes",
            z,
            badge_bg,
            badge_fg,
        )

    # Ports: a dot on the left edge and the label beside it, one row each.
    r = SIZE.GRAPH_PORT_R * z
    port_col = _u32(fade(COLOR.FG_MUTED, alpha))
    label_col = _u32(fade(COLOR.FG_MUTED, alpha))
    for slot, port in enumerate(node.ports):
        center = xf.to_screen(_port_point(node, slot))
        _draw_port_dot(dl, center, port.kind, r, port_col)
        label = node.labels[slot]
        dl.add_text(
            (center[0] + 2 * r + 2 * z, center[1] - imgui.get_font_size() / 2.0),
            label_col,
            label,
        )
    # Output dots on the right of the picture; a box's unread ones hollow.
    out_col = _u32(fade(COLOR.FG_SECONDARY, alpha))
    for slot, (_member, hollow) in enumerate(node.outputs):
        center = xf.to_screen(_out_point(node, slot))
        if hollow:
            dl.add_circle(center, r, out_col, 0, 1.2)
        else:
            dl.add_circle_filled(center, r, out_col)
    imgui.pop_font()


# ---- the widget ----------------------------------------------------------------------------


def _tab_row(
    app: App, document_id: str, view: GraphViewState, groups: list[str]
) -> None:
    ui_document = app.ui_documents[document_id]
    # The row keys and answers by NAME, so the root's label is made distinct from every
    # group's before it is drawn; the click then maps back without ambiguity.
    root_label = ui_document.ui_state.ui_name.strip()
    if not root_label or root_label in groups:
        root_label = next(label for label in _ROOT_LABELS if label not in groups)
    labels = [root_label, *groups]
    scopes = ["", *groups]
    active = labels[scopes.index(view.scope)] if view.scope in scopes else root_label
    clicked = text_tab_row("graph_scope", labels, active)
    if clicked is not None:
        index = labels.index(clicked)
        if scopes[index] != view.scope:
            view.scope = scopes[index]
            view.fitted = False


def _canvas_menu(app: App, document_id: str, view: GraphViewState) -> None:
    with context_menu_style():
        if imgui.begin_popup("##graph_canvas_menu"):
            if imgui.menu_item_simple("Add pass"):
                app.open_add_pass()
            if imgui.menu_item_simple("Import..."):
                app.open_import_passes()
            imgui.separator()
            if imgui.menu_item_simple("Fit"):
                view.fitted = False
            if imgui.menu_item_simple("Arrange"):
                app.arrange_graph(document_id)
            imgui.end_popup()


def draw(app: App, document_id: str, height: float) -> None:
    """The graph canvas for one document: the tab row, then a child `height` tall with the
    picture. The caller sizes it; the widget measures no sibling."""
    ui_document = app.ui_documents.get(document_id)
    if ui_document is None:
        return
    document = ui_document.document
    view = app.graph_view_for(document_id)
    # Ports come from the compiled program (092 D1): the seam 091 uses before it plans. A
    # no-op once every pass has been attempted, so calling it per frame costs nothing.
    compile_pending_passes(document)

    entries = document.graph.passes
    wiring = document.effective_wiring()
    order = strip_order(document.passes, wiring)
    groups = {name: entries.get(name, PassEntry()).group for name in order}
    group_names = group_names_in_order(order, groups)
    view.scope = revalidated_scope(view.scope, set(group_names))
    view.selection &= set(document.passes)

    imgui.begin_disabled(app.copilot_turn_active)
    _tab_row(app, document_id, view, group_names)

    imgui.push_style_color(imgui.Col_.child_bg, COLOR.BG_APP)
    child_open = imgui.begin_child(
        "##pass_graph",
        size=imgui.ImVec2(0.0, height),
        child_flags=imgui.ChildFlags_.borders,
        window_flags=imgui.WindowFlags_.no_scrollbar
        | imgui.WindowFlags_.no_scroll_with_mouse,
    )
    imgui.pop_style_color(1)
    if child_open:
        _draw_canvas(app, document_id, document, view, wiring, groups)
    imgui.end_child()
    imgui.end_disabled()


def _draw_canvas(
    app: App,
    document_id: str,
    document: Document,
    view: GraphViewState,
    wiring: Wiring,
    groups: dict[str, str],
) -> None:
    origin = imgui.get_cursor_screen_pos()
    avail = imgui.get_content_region_avail()
    io = imgui.get_io()
    # A gesture whose release the canvas did not see (the view switched, a modal covered it,
    # a copilot turn began) is cancelled, never resumed: a stray later click must not write.
    released_elsewhere = not imgui.is_mouse_down(
        imgui.MouseButton_.left
    ) and not imgui.is_mouse_released(imgui.MouseButton_.left)
    if released_elsewhere or app.copilot_turn_active:
        view.node_drag = None
        view.wire_drag = None
        view.band_anchor = None
        view.guides = []
    overrides = view.node_drag.current() if view.node_drag is not None else {}
    picture = _build_view(document, view.scope, overrides)
    nodes = list(picture.nodes.values())
    if not view.fitted:
        _fit(view, nodes, avail)
    xf = _Xf(origin, view.pan, view.zoom)
    dl = imgui.get_window_draw_list()
    output = document.graph.output

    # ---- wheel zoom about the cursor, read before the transform is used ----
    hovered = imgui.is_window_hovered(imgui.HoveredFlags_.child_windows)
    if hovered and io.mouse_wheel != 0.0:
        mouse = (io.mouse_pos.x, io.mouse_pos.y)
        under = xf.to_canvas(mouse)
        zoom = view.zoom * (_ZOOM_STEP**io.mouse_wheel)
        zoom = max(SIZE.GRAPH_ZOOM_MIN, min(SIZE.GRAPH_ZOOM_MAX, zoom))
        view.zoom = zoom
        view.pan = (
            under[0] - (mouse[0] - origin.x) / zoom,
            under[1] - (mouse[1] - origin.y) / zoom,
        )
        xf = _Xf(origin, view.pan, view.zoom)

    # ---- the picture: wires under nodes ----
    dl.channels_split(2)
    dl.channels_set_current(0)
    edge_col = _u32(COLOR.GRAPH_EDGE)
    dim_col = _u32(fade(COLOR.GRAPH_EDGE, COLOR.GRAPH_DIM_ALPHA))
    err_col = _u32(COLOR.STATE_ERROR)
    bottom = max((n.pos[1] + n.size[1] for n in nodes), default=0.0)
    for edge in picture.edges:
        src = picture.nodes[edge.src_key]
        dst = picture.nodes[edge.dst_key]
        a = _out_point(src, edge.src_slot)
        b = _port_point(dst, edge.dst_slot)
        col = err_col if edge.on_cycle else dim_col if edge.dim else edge_col
        backward = b[0] < a[0] + _MIN_DIRECT_DX
        bus = (
            bottom
            + SIZE.GRAPH_BUS_CLEAR
            + max(0, edge.span - 2) * SIZE.GRAPH_BUS_STEP
            + (SIZE.GRAPH_BUS_STEP if backward else 0)
        )
        _draw_wire(dl, xf, a, b, bus if (backward or edge.span > 1) else 0.0, col)
    for node in nodes:
        for slot, port in enumerate(node.ports):
            if port.kind == "prev":
                _draw_self_loop(dl, xf, node, slot, dim_col if node.stale else edge_col)
    dl.channels_set_current(1)
    for node in nodes:
        is_output = (
            node.name == output if node.kind != "box" else output in node.members
        )
        selected = (node.kind != "box" and node.name in view.selection) or (
            node.kind == "box" and bool(set(node.members) & view.selection)
        )
        _draw_node(app, dl, xf, node, is_output, selected)
    dl.channels_merge()

    # ---- hit testing: the background first, every node after, each allowing overlap ----
    imgui.set_cursor_screen_pos(origin)
    imgui.set_next_item_allow_overlap()
    imgui.invisible_button(
        "##graph_bg",
        imgui.ImVec2(max(1.0, avail.x), max(1.0, avail.y)),
        imgui.ButtonFlags_.mouse_button_left | imgui.ButtonFlags_.mouse_button_middle,
    )
    bg_hovered = imgui.is_item_hovered()
    bg_active = imgui.is_item_active()
    panning = bg_active and (
        imgui.is_mouse_down(imgui.MouseButton_.middle) or io.key_alt
    )
    if panning:
        view.pan = (
            view.pan[0] - io.mouse_delta.x / view.zoom,
            view.pan[1] - io.mouse_delta.y / view.zoom,
        )
    if imgui.is_item_clicked(imgui.MouseButton_.left) and not io.key_shift:
        view.selection.clear()
    # The rubber band: a left-drag on empty canvas that is not a pan (092 D14).
    if (
        bg_active
        and not panning
        and imgui.is_mouse_dragging(imgui.MouseButton_.left)
        and view.band_anchor is None
    ):
        delta = imgui.get_mouse_drag_delta(imgui.MouseButton_.left)
        view.band_anchor = (io.mouse_pos.x - delta.x, io.mouse_pos.y - delta.y)

    node_hovered = False
    drop_target: tuple[str, str, str] | None = None  # (owner, sampler, kind)
    for node in nodes:
        p0 = xf.to_screen(node.pos)
        p1 = xf.to_screen((node.pos[0] + node.size[0], node.pos[1] + node.size[1]))
        imgui.set_cursor_screen_pos(p0)
        imgui.set_next_item_allow_overlap()
        imgui.invisible_button(
            f"##gnode_{node.key}",
            imgui.ImVec2(max(1.0, p1[0] - p0[0]), max(1.0, p1[1] - p0[1])),
        )
        if imgui.is_item_hovered():
            node_hovered = True
            if imgui.is_mouse_double_clicked(imgui.MouseButton_.left):
                _double_click(app, document_id, view, node)
            elif imgui.is_item_clicked(imgui.MouseButton_.left):
                _click(app, document_id, view, node, io.key_shift)
        if (
            imgui.is_item_active()
            and imgui.is_mouse_dragging(imgui.MouseButton_.left)
            and view.node_drag is None
            and view.wire_drag is None
            and node.kind != "ghost"
        ):
            names = _drag_names(view, node)
            view.node_drag = NodeDrag(
                origin={
                    n: picture.positions[n] for n in names if n in picture.positions
                }
            )
        _node_menu(app, document_id, view, node)
        # Ports: the dots are drag sources, the input dots drop targets too. The hit box has
        # a screen-pixel floor for a human's aim but never exceeds half the row pitch, so two
        # rows cannot share a press; the same box answers the press and the hover.
        hit = min(
            max(SIZE.GRAPH_PORT_R * view.zoom, float(SIZE.GRAPH_HIT_MIN)),
            SIZE.GRAPH_PORT_ROW * view.zoom / 2.0,
        )
        for slot, port in enumerate(node.ports):
            center = xf.to_screen(_port_point(node, slot))
            imgui.set_cursor_screen_pos((center[0] - hit, center[1] - hit))
            imgui.set_next_item_allow_overlap()
            imgui.invisible_button(
                f"##gport_{node.key}_{slot}", imgui.ImVec2(2 * hit, 2 * hit)
            )
            owner = node.owners[slot]
            if imgui.is_item_hovered() and view.wire_drag is not None:
                drop_target = (owner, port.sampler, port.kind)
            pressed = (
                imgui.is_item_active()
                and imgui.is_mouse_dragging(imgui.MouseButton_.left)
                and view.wire_drag is None
                and view.node_drag is None
            )
            if pressed and port.kind == "wired" and port.source is not None:
                view.wire_drag = WireDrag(
                    producer=port.source,
                    start=_port_point(node, slot),
                    grabbed=(owner, port.sampler),
                )
            elif pressed and node.kind != "ghost":
                # An unfilled port is not a wire to grab: the press moves the node.
                names = _drag_names(view, node)
                view.node_drag = NodeDrag(
                    origin={
                        n: picture.positions[n] for n in names if n in picture.positions
                    }
                )
        out_hit = hit
        if len(node.outputs) > 1:
            step = SIZE.GRAPH_THUMB * view.zoom / (len(node.outputs) + 1)
            out_hit = min(hit, step / 2.0)
        for slot, (member, _hollow) in enumerate(node.outputs):
            if node.kind == "ghost":
                continue
            center = xf.to_screen(_out_point(node, slot))
            imgui.set_cursor_screen_pos((center[0] - out_hit, center[1] - out_hit))
            imgui.set_next_item_allow_overlap()
            imgui.invisible_button(
                f"##gout_{node.key}_{slot}", imgui.ImVec2(2 * out_hit, 2 * out_hit)
            )
            if (
                imgui.is_item_active()
                and imgui.is_mouse_dragging(imgui.MouseButton_.left)
                and view.wire_drag is None
                and view.node_drag is None
            ):
                view.wire_drag = WireDrag(producer=member, start=_out_point(node, slot))

    # ---- the drag in flight: move, snap, and commit on release ----
    if view.node_drag is not None:
        view.node_drag.update(
            io.mouse_delta.x / view.zoom, io.mouse_delta.y / view.zoom
        )
        view.guides = _snap(view, picture, nodes)
        if imgui.is_mouse_released(imgui.MouseButton_.left):
            app.commit_node_drag(document_id)
    if view.wire_drag is not None:
        wire = view.wire_drag
        start = xf.to_screen(wire.start)
        end = (io.mouse_pos.x, io.mouse_pos.y)
        c = max(_MIN_DIRECT_DX * view.zoom, abs(end[0] - start[0]) * _BEZIER_BOW)
        dl.add_bezier_cubic(
            start,
            (start[0] + c, start[1]),
            (end[0] - c, end[1]),
            end,
            _u32(COLOR.ACCENT_PRIMARY),
            max(1.0, 1.5 * view.zoom),
        )
        if imgui.is_mouse_released(imgui.MouseButton_.left):
            _drop(app, document_id, wire, drop_target)
            view.wire_drag = None
    if view.band_anchor is not None:
        a = view.band_anchor
        b = (io.mouse_pos.x, io.mouse_pos.y)
        lo = (min(a[0], b[0]), min(a[1], b[1]))
        hi = (max(a[0], b[0]), max(a[1], b[1]))
        dl.add_rect_filled(
            lo, hi, _u32(fade(COLOR.SELECT, COLOR.GRAPH_BAND_FILL_ALPHA))
        )
        dl.add_rect(
            lo, hi, _u32(fade(COLOR.SELECT, COLOR.GRAPH_BAND_EDGE_ALPHA)), 0.0, 1.0
        )
        if imgui.is_mouse_released(imgui.MouseButton_.left):
            picked: set[str] = set()
            for node in nodes:
                n0 = xf.to_screen(node.pos)
                n1 = xf.to_screen(
                    (node.pos[0] + node.size[0], node.pos[1] + node.size[1])
                )
                if (
                    n1[0] >= lo[0]
                    and n0[0] <= hi[0]
                    and n1[1] >= lo[1]
                    and n0[1] <= hi[1]
                ):
                    picked |= set(node.members) if node.kind == "box" else {node.name}
            view.selection = (view.selection | picked) if io.key_shift else picked
            view.band_anchor = None
    for kind, value in view.guides:
        if kind == "v":
            x = xf.to_screen((value, 0.0))[0]
            dl.add_line(
                (x, origin.y),
                (x, origin.y + avail.y),
                _u32(fade(COLOR.SELECT, COLOR.GRAPH_GUIDE_ALPHA)),
            )
        else:
            y = xf.to_screen((0.0, value))[1]
            dl.add_line(
                (origin.x, y),
                (origin.x + avail.x, y),
                _u32(fade(COLOR.SELECT, COLOR.GRAPH_GUIDE_ALPHA)),
            )

    if (
        bg_hovered
        and not node_hovered
        and imgui.is_mouse_released(imgui.MouseButton_.right)
    ):
        imgui.open_popup("##graph_canvas_menu")
    _canvas_menu(app, document_id, view)
    _group_prompt(app, document_id, view)


def _drag_names(view: GraphViewState, node: _Node) -> list[str]:
    if node.kind == "box":
        return list(node.members)
    if node.name in view.selection:
        return sorted(view.selection)
    return [node.name]


def _snap(
    view: GraphViewState, picture: _View, nodes: Sequence[_Node]
) -> list[tuple[str, float]]:
    """Set the drag's snap offset and return the guides to draw (092 D13).

    The offset aligns the first dragged node's left or top edge to a still node's when within
    the snap distance, computed fresh from the raw drag each frame; the raw delta is never
    corrected, so the node leaves the guide as soon as the cursor does.
    """
    drag = view.node_drag
    if drag is None:
        return []
    drag.snap = (0.0, 0.0)
    moving = set(drag.origin)
    threshold = SIZE.GRAPH_SNAP_PX / view.zoom
    primary = next(iter(drag.origin))
    raw = drag.raw().get(primary)
    if raw is None:
        return []
    still = [n for n in nodes if n.kind != "box" and n.name not in moving] + [
        n for n in nodes if n.kind == "box" and not (set(n.members) & moving)
    ]
    guides: list[tuple[str, float]] = []
    snap_x = 0.0
    snap_y = 0.0
    best_x = min(((abs(n.pos[0] - raw[0]), n.pos[0]) for n in still), default=None)
    if best_x is not None and best_x[0] <= threshold:
        snap_x = best_x[1] - raw[0]
        guides.append(("v", best_x[1]))
    best_y = min(((abs(n.pos[1] - raw[1]), n.pos[1]) for n in still), default=None)
    if best_y is not None and best_y[0] <= threshold:
        snap_y = best_y[1] - raw[1]
        guides.append(("h", best_y[1]))
    drag.snap = (snap_x, snap_y)
    return guides


def _drop(
    app: App,
    document_id: str,
    wire: WireDrag,
    target: tuple[str, str, str] | None,
) -> None:
    if target is not None:
        owner, sampler, _kind = target
        if wire.grabbed == (owner, sampler):
            return  # put back where it was grabbed
        error = app.drop_wire(document_id, wire.producer, owner, sampler)
        if error:
            app.notifications.push(error)
            return
        if wire.grabbed is not None:
            error = app.unwire(document_id, *wire.grabbed)
            if error:
                app.notifications.push(error)
        return
    if wire.grabbed is not None:
        error = app.unwire(document_id, *wire.grabbed)
        if error:
            app.notifications.push(error)


def _click(
    app: App, document_id: str, view: GraphViewState, node: _Node, extend: bool
) -> None:
    if node.kind == "ghost":
        view.scope = ""
        view.fitted = False
        view.selection = {node.name}
        return
    names = set(node.members) if node.kind == "box" else {node.name}
    if extend:
        view.selection ^= names
        return
    view.selection = names
    app.pick_pass(
        document_id,
        node.bundle if node.kind == "box" else node.name,
        focus_editor=False,
    )


def _double_click(
    app: App, document_id: str, view: GraphViewState, node: _Node
) -> None:
    if node.kind == "box":
        view.scope = node.group
        view.fitted = False
        view.selection = set(node.members)
        return
    if node.kind == "ghost":
        _click(app, document_id, view, node, False)
        return
    app.pick_pass(document_id, node.name, focus_editor=True)


def _node_menu(app: App, document_id: str, view: GraphViewState, node: _Node) -> None:
    # Anchored to the node's button just submitted: the None id is what keeps a right-click
    # elsewhere on the shared child from opening this node's menu.
    with context_menu_style():
        if imgui.begin_popup_context_item(None):
            if node.kind == "box":
                if imgui.menu_item_simple("Open"):
                    view.scope = node.group
                    view.fitted = False
                if imgui.menu_item_simple("Dissolve"):
                    app.dissolve_group(document_id, node.group)
            else:
                pass_menu_items(app, document_id, node.name)
                if node.kind == "pass" and imgui.menu_item_simple("Group..."):
                    if node.name not in view.selection:
                        view.selection = {node.name}
                    view.group_prompt = True
                    view.group_name = ""
            imgui.end_popup()


def _group_prompt(app: App, document_id: str, view: GraphViewState) -> None:
    """The name a Group... asks for (092 D14): a small popup, Enter or Create commits."""
    if view.group_prompt:
        imgui.open_popup("##graph_group")
        view.group_prompt = False
    if not imgui.begin_popup("##graph_group"):
        return
    if imgui.is_window_appearing():
        imgui.set_keyboard_focus_here()
    imgui.set_next_item_width(float(SIZE.NAME_INPUT_W))
    entered, view.group_name = imgui.input_text(
        "##graph_group_name",
        view.group_name,
        imgui.InputTextFlags_.enter_returns_true,
    )
    imgui.same_line()
    committed = primary_button("Create") or entered
    name = view.group_name.strip()
    # A blank name would mean "no group" to the verb, which is Dissolve, not Create.
    if (
        committed
        and name
        and view.selection
        and app.group_selection(document_id, name) == ""
    ):
        imgui.close_current_popup()
    imgui.same_line()
    if standard_button("Cancel"):
        imgui.close_current_popup()
    imgui.end_popup()
