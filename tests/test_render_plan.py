"""The render plan's arithmetic and the Auto-size damping (090 D3, D6).

GL-free and imgui-free, because `render_plan.py` is. The worked examples in the spec's *The
plan function* table ARE this file's first test: each row is the input, the exact interval and
the exact phase, so a change to the rule has to change a number the spec also carries.

Each test names the falsifier it was born against, since a throttle that returns plausible
intervals is exactly the shape a passing suite cannot distinguish from a broken one.
"""

import pytest

from shaderbox.render_plan import (
    AUTO_RESIZE_STABLE_FRAMES,
    INTERVAL_HYSTERESIS_FRAMES,
    MAX_INTERVAL,
    AutoSizeState,
    CostRecord,
    ThrottleState,
    apply_damping,
    auto_canvas_size,
    plan_render_set,
)

_PERIOD = 16.7  # ms, the spec's frame period for every worked example
_BUDGET = 0.5


def _converged(
    costs: dict[str, CostRecord],
    current: str | None,
    displayed: list[str],
    budget: float = _BUDGET,
    enabled: bool = True,
):
    """The plan after the hysteresis has settled: the same input, long enough to stop moving.

    The window is what makes a `k` take effect at all, so a single call reports the interval
    the states STARTED at rather than the one the costs ask for.
    """
    states: dict[str, ThrottleState] = {}
    plan = None
    for _ in range(INTERVAL_HYSTERESIS_FRAMES * 3):
        plan = plan_render_set(
            costs, current, displayed, budget, _PERIOD, states, enabled
        )
    assert plan is not None
    return plan


def _flat(cost_ms: float, ids: list[str]) -> dict[str, CostRecord]:
    return {name: CostRecord(gpu_ms=cost_ms, cpu_ms=0.0) for name in ids}


# ---------------------------------------------------------------------------
# V1 -- the spec's nine worked examples, each on its exact interval and phase
# ---------------------------------------------------------------------------


def test_a_current_document_inside_the_budget_renders_every_frame() -> None:
    # Example 1: 6 ms <= 8.35 ms of budget.
    plan = _converged(_flat(6.0, ["c"]), "c", ["c"])
    assert plan.intervals == {"c": 1}
    assert plan.phases == {"c": 0}
    assert plan.document_fps["c"] == pytest.approx(59.88, abs=0.05)


def test_a_hundred_millisecond_current_document_lands_on_twelve() -> None:
    # Example 2: ceil(100 / 8.35) = ceil(11.976) = 12. Falsifier: `ceil` -> `floor` returns 11,
    # which is an interval whose cost does NOT fit the budget -- the whole point of the ceiling.
    plan = _converged(_flat(100.0, ["c"]), "c", ["c"])
    assert plan.intervals == {"c": 12}
    assert plan.document_fps["c"] == pytest.approx(4.99, abs=0.02)


def test_the_current_document_takes_the_budget_before_the_previews_do() -> None:
    # Example 3: current 40 ms -> k = 5 using 8.0 ms, leaving 0.35 ms; three 5 ms previews share
    # it at f = 1.4 -> k = 43. Falsifier: split the budget BEFORE the current document takes its
    # share and the previews come back at k = 5, twelve times too often.
    costs = _flat(5.0, ["a", "b", "d"]) | _flat(40.0, ["c"])
    plan = _converged(costs, "c", ["c", "a", "b", "d"])
    assert plan.intervals == {"c": 5, "a": 43, "b": 43, "d": 43}


def test_twenty_cheap_previews_share_the_remainder() -> None:
    # Example 4: current 2 ms -> k = 1, remainder 6.35 ms; 20 x 1 ms at f = 19.05 -> k = 3.
    previews = [f"p{i}" for i in range(20)]
    costs = _flat(1.0, previews) | _flat(2.0, ["c"])
    plan = _converged(costs, "c", ["c", *previews])
    assert plan.intervals["c"] == 1
    assert {plan.intervals[name] for name in previews} == {3}


def test_a_fixed_full_resolution_preview_is_made_affordable_by_the_remainder() -> None:
    # Example 5: a Fixed 100 ms document shown as a thumbnail still costs 100 ms (D10), and the
    # remainder rule is what makes it affordable -- k = 23, so it renders rarely.
    costs = _flat(4.0, ["c"]) | _flat(100.0, ["f"])
    plan = _converged(costs, "c", ["c", "f"])
    assert plan.intervals == {"c": 1, "f": 23}


def test_the_throttle_off_returns_todays_set_exactly() -> None:
    # Example 6. Falsifier: ignore the flag and the previews come back at 43.
    costs = _flat(5.0, ["a", "b", "d"]) | _flat(40.0, ["c"])
    plan = _converged(costs, "c", ["c", "a", "b", "d"], enabled=False)
    assert set(plan.intervals.values()) == {1}
    assert set(plan.phases.values()) == {0}


def test_a_full_budget_gives_the_current_document_the_whole_frame() -> None:
    # Example 7: budget 1.0 -> 16.7 ms; ceil(40 / 16.7) = ceil(2.395) = 3.
    plan = _converged(_flat(40.0, ["c"]), "c", ["c"], budget=1.0)
    assert plan.intervals == {"c": 3}
    assert plan.document_fps["c"] == pytest.approx(19.96, abs=0.05)


def test_an_unmeasured_set_renders_every_frame() -> None:
    # Example 8: the first frames, before any cost has been recorded (D7: no record -> k = 1).
    # Falsifier: treat a missing record as an infinite cost and frame 0 throttles everything.
    plan = _converged({}, "c", ["c", "a", "b"])
    assert set(plan.intervals.values()) == {1}


def test_the_interval_cap_binds_every_document_not_only_the_zero_fps_case() -> None:
    # Example 9: ten 5 ms previews behind a 40 ms current document compute f = 0.42, whose
    # uncapped interval is round(60 / 0.42) = 143 -- a tile refreshing once every 2.4 s.
    # Falsifier: clamp only the f = 0 case and this returns 143.
    previews = [f"p{i}" for i in range(10)]
    costs = _flat(5.0, previews) | _flat(40.0, ["c"])
    plan = _converged(costs, "c", ["c", *previews])
    assert plan.intervals["c"] == 5
    assert {plan.intervals[name] for name in previews} == {MAX_INTERVAL}
    assert plan.document_fps["p0"] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# V2 -- hysteresis
# ---------------------------------------------------------------------------


def test_a_cost_hovering_at_the_boundary_never_moves_the_interval() -> None:
    # 8.0 and 8.6 ms sit either side of the 8.35 ms budget, so the computed k alternates 1, 2,
    # 1, 2 ... and the LIVE interval must never move. Falsifier: apply the candidate at once and
    # the document's rate flickers every frame.
    states: dict[str, ThrottleState] = {}
    seen: set[int] = set()
    for frame in range(20):
        cost = 8.0 if frame % 2 == 0 else 8.6
        plan = plan_render_set(
            _flat(cost, ["c"]), "c", ["c"], _BUDGET, _PERIOD, states, True
        )
        seen.add(plan.intervals["c"])
    assert seen == {1}, f"the interval moved on a hovering cost: {seen}"


def test_a_steady_cost_moves_the_interval_on_exactly_the_fourth_frame() -> None:
    # The window is the point: a `k` may not act on a number the ring has not finished
    # reporting (088 D2's two-frame read lag). Falsifier: window of 1 and it moves on frame 1.
    states: dict[str, ThrottleState] = {}
    moved_on: list[int] = []
    for frame in range(1, 10):
        plan = plan_render_set(
            _flat(100.0, ["c"]), "c", ["c"], _BUDGET, _PERIOD, states, True
        )
        if plan.intervals["c"] != 1:
            moved_on.append(frame)
    assert moved_on and moved_on[0] == INTERVAL_HYSTERESIS_FRAMES


# ---------------------------------------------------------------------------
# V2a -- same-k documents land on different frames
# ---------------------------------------------------------------------------


def test_three_same_interval_previews_take_three_different_frames() -> None:
    # Without a phase offset every preview at k = 43 renders on the same frame_idx, so one
    # frame in 43 carries all three hitches and 42 carry none -- the opposite of what spacing
    # them out is for. Falsifier: drop the phase and `admitted` holds one frame with three.
    costs = _flat(5.0, ["a", "b", "d"]) | _flat(40.0, ["c"])
    plan = _converged(costs, "c", ["c", "a", "b", "d"])
    interval = plan.intervals["a"]
    assert {plan.phases[name] for name in ("a", "b", "d")} == {1, 2, 3}
    per_frame: dict[int, list[str]] = {}
    for frame in range(interval * 2):
        for name in ("a", "b", "d"):
            if (frame + plan.phases[name]) % interval == 0:
                per_frame.setdefault(frame, []).append(name)
    assert per_frame, "no preview ever rendered"
    assert all(len(names) == 1 for names in per_frame.values()), per_frame


# ---------------------------------------------------------------------------
# V3 -- the damping, pure
# ---------------------------------------------------------------------------


def _drag_ramp() -> list[tuple[int, int]]:
    """A window drag: 764 -> 940 px in ~3 px steps, at the viewer's 4:3 aspect."""
    return [(w, round(w * 3 / 4)) for w in range(764, 941, 3)]


def test_a_drag_ramp_reallocates_a_handful_of_times_not_every_frame() -> None:
    # Falsifier: drop the dead band and every one of the ~60 steps reallocates the canvas and
    # every feedback history with it.
    state = AutoSizeState()
    current = (764, 573)
    applied: list[tuple[int, int]] = []
    for requested in _drag_ramp():
        answer = apply_damping(state, requested, current)
        if answer is not None:
            applied.append(answer)
            current = answer
    assert 0 < len(applied) <= 4, (
        f"{len(applied)} reallocations across the ramp: {applied}"
    )


def test_a_size_held_still_lands_even_inside_the_dead_band() -> None:
    # The drag ends a few pixels off the last applied size, which the dead band alone would
    # never apply -- so the document would render at the wrong size for as long as it is shown.
    # Falsifier: drop the stability clause and `answers` stays all None.
    state = AutoSizeState()
    current = (800, 600)
    requested = (810, 608)  # inside 5 % on both axes
    answers = [
        apply_damping(state, requested, current)
        for _ in range(AUTO_RESIZE_STABLE_FRAMES)
    ]
    assert answers[:-1] == [None] * (AUTO_RESIZE_STABLE_FRAMES - 1)
    assert answers[-1] == requested


def test_a_size_past_the_dead_band_applies_at_once() -> None:
    # A real resize is not a jitter: waiting eight frames to follow it is visible lag.
    state = AutoSizeState()
    assert apply_damping(state, (1200, 900), (800, 600)) == (1200, 900)


def test_an_unchanged_request_is_never_reapplied() -> None:
    # The canvas is already there; answering with it would resample every frame forever.
    state = AutoSizeState()
    for _ in range(AUTO_RESIZE_STABLE_FRAMES * 2):
        assert apply_damping(state, (800, 600), (800, 600)) is None


# ---------------------------------------------------------------------------
# The Auto size itself
# ---------------------------------------------------------------------------


def test_a_document_displayed_nowhere_keeps_its_size() -> None:
    # `None` is what a document shown in no region answers, and it must not resize to anything.
    assert auto_canvas_size(None, 4 / 3, (800, 600)) == (800, 600)


def test_the_auto_size_follows_the_region_on_its_constrained_axis() -> None:
    # The region carries the document's own aspect once the cell has fitted it, so both axes
    # agree; where they do not, the smaller one decides and the other is re-derived from the
    # STORED aspect, which is what stops the shape drifting a rounding step per frame.
    assert auto_canvas_size((400, 300), 4 / 3, (800, 600)) == (400, 300)
    assert auto_canvas_size((400, 400), 4 / 3, (800, 600)) == (400, 300)
    assert auto_canvas_size((400, 100), 4 / 3, (800, 600)) == (133, 100)
