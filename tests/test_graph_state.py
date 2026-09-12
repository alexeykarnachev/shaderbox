"""The graph canvas's pure gesture pieces (092 D3, D13): the drag writes only on commit, and
a scope no pass carries falls back to the root."""

from shaderbox.theme import SIZE
from shaderbox.widgets.graph_state import (
    NodeDrag,
    group_names_in_order,
    node_size,
    revalidated_scope,
)


def test_a_drag_writes_nothing_until_commit_and_then_every_moved_name_once() -> None:
    # Falsifier: return positions from `update` (a per-frame write, sixty saves a second).
    drag = NodeDrag(origin={"a": (0.0, 0.0), "b": (10.0, 5.0)})
    assert drag.update(3.0, 4.0) is None
    assert drag.update(1.0, 1.0) is None
    committed = drag.commit()
    assert committed == {"a": (4.0, 5.0), "b": (14.0, 10.0)}
    assert list(committed) == ["a", "b"]


def test_the_snap_offset_rides_on_top_of_the_raw_delta_and_never_corrects_it() -> None:
    # Falsifier: fold the snap into `delta` -- a node held at a guide then absorbs every
    # later mouse move and never leaves it (the post-implementation review's blocker).
    drag = NodeDrag(origin={"a": (0.0, 0.0)})
    drag.update(10.0, 0.0)
    drag.snap = (-3.0, 0.0)
    assert drag.raw()["a"] == (10.0, 0.0)
    assert drag.current()["a"] == (7.0, 0.0)
    assert drag.commit()["a"] == (7.0, 0.0)
    drag.update(10.0, 0.0)
    drag.snap = (0.0, 0.0)
    assert drag.current()["a"] == (20.0, 0.0)


def test_the_moving_picture_reads_the_same_positions_the_commit_writes() -> None:
    drag = NodeDrag(origin={"a": (2.0, 2.0)})
    drag.update(-2.0, 0.5)
    assert drag.current() == drag.commit()


def test_a_scope_no_pass_carries_falls_back_to_the_root() -> None:
    assert revalidated_scope("bloom", {"bloom", "fx"}) == "bloom"
    assert revalidated_scope("gone", {"bloom"}) == ""
    assert revalidated_scope("", {"bloom"}) == ""


def test_group_names_follow_their_first_members_order() -> None:
    order = ["scene", "b1", "g1", "b2", "final"]
    groups = {"b1": "bloom", "b2": "bloom", "g1": "grade"}
    assert group_names_in_order(order, groups) == ["bloom", "grade"]


def test_a_node_grows_one_row_per_port_and_a_box_is_wider() -> None:
    w0, h0 = node_size(0, False)
    w2, h2 = node_size(2, False)
    assert w0 == w2 == float(SIZE.GRAPH_NODE_W)
    assert h2 == h0 + 4.0 + 2 * SIZE.GRAPH_PORT_ROW
    assert node_size(0, True)[0] == w0 + SIZE.GRAPH_BOX_EXTRA_W
