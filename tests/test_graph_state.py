"""The graph canvas's pure pieces, decided without imgui (092 D3; 098).

What is left here after the library took the drawing and the gestures: which
scope the tabs show, what a group is called, and how big a node is. The wire
geometry, the hit test and the drag machine went with the renderer that owned
them -- `test_graph_canvas_gestures.py` drives those through the library now.
"""

from shaderbox.theme import SIZE
from shaderbox.widgets.graph_state import (
    group_names_in_order,
    node_size,
    revalidated_scope,
)


def test_a_scope_no_pass_carries_falls_back_to_the_root() -> None:
    assert revalidated_scope("bloom", {"bloom", "fx"}) == "bloom"
    assert revalidated_scope("gone", {"bloom"}) == ""
    assert revalidated_scope("", {"bloom"}) == ""


def test_group_names_follow_their_first_members_order() -> None:
    order = ["scene", "b1", "g1", "b2", "final"]
    groups = {"b1": "bloom", "b2": "bloom", "g1": "grade"}
    assert group_names_in_order(order, groups) == ["bloom", "grade"]


def test_a_node_grows_one_row_per_port_and_a_box_is_wider() -> None:
    # A card with ports pays BOTH pads (093 W2-6): the gap above the first row and the one
    # under the last, so its label clears the border by what its left inset gives it.
    # Falsifier: drop `GRAPH_PORT_BOTTOM` from `node_size` and the last label sits ~3px off it.
    w0, h0 = node_size(0, False)
    w2, h2 = node_size(2, False)
    assert w0 == w2 == float(SIZE.GRAPH_NODE_W)
    assert h2 == h0 + SIZE.GRAPH_PORT_TOP + 2 * SIZE.GRAPH_PORT_ROW + (
        SIZE.GRAPH_PORT_BOTTOM
    )
    assert node_size(0, True)[0] == w0 + SIZE.GRAPH_BOX_EXTRA_W
    # A card with NO ports pays neither.
    assert h0 == float(
        SIZE.GRAPH_THUMB_INSET + SIZE.GRAPH_THUMB + SIZE.GRAPH_NAME_H + SIZE.GRAPH_PAD
    )
