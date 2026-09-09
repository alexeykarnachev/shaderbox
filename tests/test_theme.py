"""The load bands: which state hue a measurement gets for its share of the frame budget
(feature 089 D7).

GL-free -- `load_color` is a pure function of a ratio, and the plan it feeds is pure too.
The bands are pinned at their EDGES, because a band is exactly its two thresholds and a
test in the middle of one passes whichever way a threshold moved.
"""

from shaderbox.profiling import FrameProfile, Span
from shaderbox.theme import COLOR, load_color
from shaderbox.ui_primitives import profile_rows_plan


def test_the_bands_meet_at_their_thresholds() -> None:
    # Falsifier: move either literal in `theme.py` and one of the four flips.
    assert load_color(0.49) is COLOR.STATE_OK
    assert load_color(0.5) is COLOR.STATE_WARN
    assert load_color(0.99) is COLOR.STATE_WARN
    assert load_color(1.0) is COLOR.STATE_ERROR


def test_a_row_over_the_budget_carries_the_error_hue() -> None:
    # The band reaches the panel through the plan, not only through the function: a span
    # at 1.2x the budget is red on the row a reader sees. Falsifier: the plan colors its
    # measured rows FG_MUTED, or divides by something other than the budget.
    root = Span("frame", cpu_ms=20.0, children=[Span("pass:slow", cpu_ms=20.0)])
    rows = profile_rows_plan(
        FrameProfile(root, 0, complete=True), fps=50, target_fps=60
    )
    slow = next(row for row in rows if row.name == "pass:slow")
    assert slow.color is COLOR.STATE_ERROR, (
        f"a span at 20 ms against a 16.7 ms budget drew {slow.color}"
    )


def test_the_static_rows_carry_no_state_hue() -> None:
    # Color on this panel means "measured against the budget"; `budget`, `fps` and `target`
    # are settings, so they stay muted. Falsifier: color them by anything.
    rows = profile_rows_plan(None, fps=60, target_fps=60)
    assert [row.name for row in rows] == ["budget", "fps", "target"]
    assert all(row.color is COLOR.FG_MUTED for row in rows)
