"""The load bands: which state hue a measurement gets for its share of the frame budget
(feature 089 D7), and the THROTTLE bands beside them (090 D9c).

GL-free -- both are pure functions of a ratio, and the plan they feed is pure too.
The bands are pinned at their EDGES, because a band is exactly its two thresholds and a
test in the middle of one passes whichever way a threshold moved.

The two functions have deliberately different knees, which is the whole reason the second
exists: `load_color` is red at 1.0, and 1.0 is where a converged throttled document sits.
"""

from shaderbox.profiling import FrameProfile, Span
from shaderbox.render_plan import RenderPlan
from shaderbox.theme import _ACCENTS, COLOR, group_tint, load_color, throttle_color
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


def test_throttle_color_reads_a_converged_document_as_healthy() -> None:
    # `throttle_color` exists because `load_color`'s knees are WRONG for a throttled document's
    # row (090 D9c, closing correctness F10): `load_color` turns red at ratio 1.0, which is
    # exactly where a converged document sits -- at its allowance, which is the state the
    # throttle aims for. Falsifier: `return load_color(share_ratio)` and the first assertion
    # goes red, reintroducing F10 verbatim.
    assert throttle_color(1.0, False) is COLOR.STATE_OK
    assert throttle_color(1.2, False) is COLOR.STATE_WARN
    assert throttle_color(1.6, False) is COLOR.STATE_ERROR


def test_a_missed_frame_reddens_a_document_inside_its_allowance() -> None:
    # The clause the spec singles out: a document within its share while the UI frame still
    # misses its target is what the reader must see. It is also the half a later edit is most
    # likely to drop, since every share-only case passes without it. Falsifier: drop
    # `frame_over_budget` from the condition and this returns STATE_OK.
    assert throttle_color(0.2, True) is COLOR.STATE_ERROR


def test_a_throttled_row_takes_the_throttle_bands_not_the_load_bands() -> None:
    # The band reaches the panel through the plan: a converged document (share == budget) draws
    # GREEN on the row a reader sees, which is the whole claim F10 found false. Falsifier: color
    # `_document_row`'s throttled branch with `load_color` and this row is red.
    root = Span("frame", cpu_ms=10.0)
    root.children.append(Span("document:aaa", cpu_ms=50.0, gpu_ms=50.0))
    plan = RenderPlan(
        intervals={"aaa": 6},
        phases={"aaa": 0},
        # 50 ms x 10 fps = 0.5 of wall time, which IS the 0.5 budget: ratio 1.0 exactly.
        document_fps={"aaa": 10.0},
    )
    rows = profile_rows_plan(
        FrameProfile(root, 0, complete=True),
        fps=60,
        target_fps=60,
        plan=plan,
        titles={"aaa": "Converged"},
        budget=0.5,
    )
    row = next(r for r in rows if r.name == "Converged")
    assert row.color is COLOR.STATE_OK, (
        f"a document exactly at its allowance drew {row.color}"
    )


def test_a_closed_documents_row_falls_back_to_a_short_handle() -> None:
    # The profile is two frames late, so a closed document's span outlives its title entry; the
    # row shows a short handle rather than a 36-character uuid that clips the number.
    root = Span("frame", cpu_ms=10.0)
    root.children.append(
        Span("document:77a84d27-2e5b-406d-8011-ee1cb1a9587c", cpu_ms=4.0, gpu_ms=4.0)
    )
    names = [
        row.name
        for row in profile_rows_plan(
            FrameProfile(root, 0, complete=True), 60, 60, titles={}
        )
    ]
    assert "77a84d27" in names
    assert "77a84d27-2e5b-406d-8011-ee1cb1a9587c" not in names


def test_group_tints_are_stable_and_collide_with_nothing() -> None:
    # 091 D8. (a) pinned by VALUE at crc32 indices: `hash()` is salted per process, so under
    # it the pin is red on essentially every run rather than flaky green.
    assert group_tint("bloom") is COLOR.GROUP_TINTS[3]
    assert group_tint("radiance") is COLOR.GROUP_TINTS[2]
    assert group_tint("fx") is COLOR.GROUP_TINTS[0]
    # (b) disjoint from every outline and chip these tiles carry, and no duplicate. Falsifier:
    # put `purple_b` (COLOR.SELECT) in the tuple -- a check over accent primaries and state
    # hues alone lets it through.
    excluded = {primary for primary, _active, _alpha in _ACCENTS.values()} | {
        COLOR.STATE_OK,
        COLOR.STATE_WARN,
        COLOR.STATE_ERROR,
        COLOR.STATE_INFO,
        COLOR.SELECT,
        COLOR.TAG,
        COLOR.FAVS,
        # 092: a wire is drawn against a box border and beside a STATE_ERROR wire.
        # Falsifier: `GRAPH_EDGE = COLOR.GROUP_TINTS[2]` imported clean before it joined.
        COLOR.GRAPH_EDGE,
    }
    assert not set(COLOR.GROUP_TINTS) & excluded
    assert len(set(COLOR.GROUP_TINTS)) == len(COLOR.GROUP_TINTS)
    assert COLOR.GRAPH_EDGE != COLOR.STATE_ERROR
