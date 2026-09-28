"""104 D3/D8: the graph canvas marks an instanced pass in its node TITLE.

104 D2a is explicit that this is not the imgui badge item 7 is: `graph_canvas/render.py`'s
`Node` struct is a fixed-layout ctypes binding to a vendored, compiled `libgraph_canvas.so`
this repo has no source for, so a new struct field is not buildable here. The title string
follows the library's own existing precedent -- the OUTPUT pass's `->` mark -- costing
neither an FFI change nor a rebuild.

GL-free: `pack_nodes` returns plain `NodeSpec` dataclasses, so the title is asserted
directly with no draw.
"""

from shaderbox.graph_canvas.adapter import flat_view, pack_nodes, pass_key
from shaderbox.pass_graph import Port


def _view(names: list[str]) -> object:
    ports: dict[str, list[Port]] = {name: [] for name in names}
    positions = {name: (float(i) * 200.0, 0.0) for i, name in enumerate(names)}
    return flat_view(names, ports, positions)


def test_an_instanced_node_is_marked_and_a_fullscreen_node_beside_it_is_not() -> None:
    # A real contrast pair: one frame, two nodes, differing ONLY in whether their pass_key
    # is in `instanced` -- not two separate fixtures that could differ for any other reason.
    view = _view(["swarm", "blur"])
    packed = pack_nodes(
        view,
        {},
        output="",
        instanced=frozenset({pass_key("swarm")}),
    )
    titles = {packed.name_of(i): spec.title for i, spec in enumerate(packed.nodes)}
    swarm_title = titles["swarm"]
    blur_title = titles["blur"]
    assert "swarm" in swarm_title and swarm_title != "swarm", (
        "the instanced node's title carries no mark"
    )
    # Contact proof for the "is not marked" half: the fullscreen node's own name must be
    # present verbatim (proving its title really was built from real data -- an empty
    # string a broken fixture would also produce is not "correctly absent").
    assert blur_title == "blur", "the fullscreen node's title changed contact"
    assert len(titles) == 2


def test_a_box_carries_no_instanced_mark() -> None:
    # `instanced` is keyed like `failing` -- by pass_key, never a box key -- because a box
    # collapses several passes and "is THIS box instanced" has no single answer.
    view = _view(["swarm"])
    packed = pack_nodes(
        view, {}, output="", instanced=frozenset({"g:some-group"})
    )
    assert packed.nodes[0].title == "swarm"
