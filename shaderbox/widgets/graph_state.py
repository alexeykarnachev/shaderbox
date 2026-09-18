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
from shaderbox.ui_primitives import InlineInput

Position = tuple[float, float]
# A wire's identity anywhere on the canvas: the consumer pass and the sampler the wire
# terminates at. A sampler has one source (072), so the pair names one wire in the whole
# document, and it is what `App.unwire` takes.
WireId = tuple[str, str]


@dataclass
class GraphViewState:
    """What the graph canvas keeps between frames, beyond the library's own.

    The library owns the geometry, the hover, the in-flight gesture and the
    camera (098), so what is left here is what shaderbox decides: which scope
    the tabs are showing, which passes are selected, and the Group prompt. The
    hit rects, the hover fields and the drag machines this held for the imgui
    canvas are gone with it.
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


# ---- the wire's pure geometry (093 S9) -------------------------------------------------------
