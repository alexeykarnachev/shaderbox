"""Feature 087 W-A: how many pass tiles the strip puts on one row.

The count is the strip's "not overflow the right border" guarantee, so it is arithmetic
rather than a drawing detail: `preview_cell` is a child window exactly one tile wide, so a
tile that fits the count fits the pixels.
"""

from shaderbox.pass_graph import group_runs
from shaderbox.theme import SIZE, SPACE
from shaderbox.widgets.pass_list import tiles_per_row

_TILE: float = float(SIZE.PASS_TILE)
_GAP: float = float(SPACE.MD)


def test_the_last_tile_is_charged_no_trailing_gap() -> None:
    # 520 is exactly three tiles and two gaps. Falsifier: the old `avail // (tile + gap)`
    # form charges the third tile a gap it does not use and answers 2.
    assert tiles_per_row(520.0, _TILE, _GAP) == 3
    assert tiles_per_row(519.0, _TILE, _GAP) == 2
    assert tiles_per_row(100.0, _TILE, _GAP) == 1, "one tile always fits"


def test_no_width_in_the_panel_range_overflows() -> None:
    # Falsifier: the over-counting `(avail + tile) // (tile + gap)` form first exceeds its
    # own width at 184 (n = 2, a row of 344).
    for avail in range(100, 1201):
        n = tiles_per_row(float(avail), _TILE, _GAP)
        row = n * _TILE + (n - 1) * _GAP
        assert row <= avail or n == 1, f"{n} tiles span {row} in {avail}"


def test_group_runs_cut_by_adjacency_not_by_name() -> None:
    # 091 D7. Falsifier: a `defaultdict(list)` keyed by group name merges the split run.
    groups = {"a": "", "b": "g", "c": "g", "d": "", "e": "g"}
    assert group_runs(["a", "b", "c", "d", "e"], groups) == [
        ["a"],
        ["b", "c"],
        ["d"],
        ["e"],
    ]
    assert group_runs(["b", "c"], groups) == [["b", "c"]]
    assert group_runs(["a", "d"], groups) == [["a"], ["d"]]
