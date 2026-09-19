"""The graph canvas's pure pieces, decided without imgui (092 D3; 098).

What is left here after the library took the drawing and the gestures: which
scope the tabs show, what a group is called, and how big a node is. The wire
geometry, the hit test and the drag machine went with the renderer that owned
them -- `test_graph_canvas_gestures.py` drives those through the library now.
"""

import pytest

from shaderbox.graph_canvas import ffi
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


def test_a_node_is_sized_the_way_the_library_will_draw_it() -> None:
    """`node_size` feeds `rank_layout`, which places a node the LIBRARY then
    draws -- so the two have to agree or the auto-layout packs a graph into
    less room than it takes.

    They did not. `node_size` was built from shaderbox's own tokens,
    written for the hand-drawn canvas and never retuned: measured against
    the library they were 24px narrow and 33px short at every port count.
    The test that stood here asserted the token formula, so it pinned the
    disagreement rather than catching it.

    Compared against `gc_node_size` -- the library's own answer for a node
    it has actually laid out -- because that is the number the drawing
    uses. A BOX is the one exception: it is wider by shaderbox's own extra,
    since the library has no idea a node stands for a group.
    """
    canvas = ffi.Canvas()
    canvas.load_atlas()
    # From ONE port up: a node with no ports has no port section, so its
    # height sits below the linear run rather than on it, and `node_size`
    # models the run. Nothing lays out a portless pass -- every pass has at
    # least its own output -- so the discontinuity costs nothing.
    for count in (1, 2, 3):
        ports = [ffi.PortSpec(f"u_{i}", True) for i in range(count)]
        node = ffi.NodeSpec(id=1, title="n", x=0, y=0, ports=ports)
        canvas.frame([node], [], (700.0, 600.0), ffi.View(), ffi.PointerState())
        drawn = canvas.node_size(0)
        assert drawn is not None
        assert node_size(count, False) == pytest.approx(drawn), (
            f"at {count} ports the layout sizes a node {node_size(count, False)} "
            f"while the library draws it {drawn}"
        )
    canvas.release()
    # A box is wider by shaderbox's own extra and otherwise the same.
    plain_w, plain_h = node_size(2, False)
    box_w, box_h = node_size(2, True)
    assert box_w == plain_w + SIZE.GRAPH_BOX_EXTRA_W
    assert box_h == plain_h
