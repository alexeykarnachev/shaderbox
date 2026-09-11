"""How often each displayed document renders, and what size an Auto one renders at (090).

Leaf: `math` and `dataclasses` only. No GL, no imgui, no `App`, no `Document` — the policy as
pure data, the shape `profile_rows_plan` set (088 D5), so every rule here is testable without a
window.

`plan_render_set` is the whole throttle rule. The current document draws on the shared GPU
budget first; every other displayed document shares what is left at one common fps, and each
lands on its own phase within its interval so same-`k` documents do not pile onto one frame.
`auto_canvas_size` and `apply_damping` are the Auto-resolution half: the size the display asks
for, and whether that request is worth a reallocation yet.
"""

import math
from dataclasses import dataclass, field

# The largest interval any document is thrown to. Ten 5 ms previews behind a 40 ms current
# document compute k = 143 uncapped, which refreshes a tile once every 2.4 s and reads as
# frozen; 60 is one refresh a second at 60 fps.
MAX_INTERVAL: int = 60

# Frames a recomputed interval must hold before it is applied. The cost input is read two
# frames late (088 D2), so a shorter window would flip `k` on a number the ring has not
# finished reporting; four frames is <= 66 ms of reaction at 60 fps.
INTERVAL_HYSTERESIS_FRAMES: int = 4

# An Auto resize below this fraction of the live size is not worth a reallocation: the cost
# difference at the viewer's size is under a tenth of a millisecond.
AUTO_RESIZE_DEAD_BAND: float = 0.05

# ... unless the same size has been asked for this many frames running. A window drag emits a
# new size every frame, so the stability clause cannot fire mid-drag; ~130 ms at 60 fps.
AUTO_RESIZE_STABLE_FRAMES: int = 8


@dataclass(frozen=True)
class CostRecord:
    """What one document cost the frame that measured it, two frames back (088 D2).

    The policy reads `gpu_ms` alone; `cpu_ms` rides along so a CPU throttle later is a policy
    change rather than new plumbing.
    """

    gpu_ms: float
    cpu_ms: float


@dataclass
class ThrottleState:
    """One document's live interval and the hysteresis counter behind it. Ephemeral."""

    interval: int = 1
    candidate: int = 1
    agreeing_frames: int = 0


@dataclass
class AutoSizeState:
    """What an Auto document's display last asked for, and for how long. Ephemeral."""

    requested: tuple[int, int] | None = None
    stable_frames: int = 0


@dataclass(frozen=True)
class RenderPlan:
    """Which frames each displayed document renders on.

    `intervals[id] = k`: it renders when `(frame_idx + phases[id]) % k == 0`.
    `document_fps[id]` is the rate that works out to.
    """

    intervals: dict[str, int] = field(default_factory=dict)
    phases: dict[str, int] = field(default_factory=dict)
    document_fps: dict[str, float] = field(default_factory=dict)


def auto_canvas_size(
    displayed: tuple[int, int] | None, aspect: float, previous: tuple[int, int]
) -> tuple[int, int]:
    """The size an Auto document wants, given the largest region showing it.

    `aspect` is the STORED resolution's, never the live canvas's — reading the live one closes
    a loop on itself. A document displayed nowhere keeps `previous`.
    """
    if displayed is None:
        return previous
    width, height = displayed
    if width <= 0 or height <= 0:
        return previous
    # The drawn region already carries the document's aspect (the cell fits the texture into
    # it), so the constrained axis is whichever the region ran out of first.
    if width / height >= aspect:
        return (max(1, round(height * aspect)), max(1, round(height)))
    return (max(1, round(width)), max(1, round(width / aspect)))


def apply_damping(
    state: AutoSizeState, requested: tuple[int, int], current: tuple[int, int]
) -> tuple[int, int] | None:
    """The size to apply this frame, or None to leave the canvas where it is.

    A request past the dead band applies at once; one inside it applies only once it has been
    asked for `AUTO_RESIZE_STABLE_FRAMES` frames running, so a window drag reallocates a
    handful of times rather than every frame and still lands on the size the drag ended at.
    `state` carries the counter and is mutated in place.
    """
    if requested != state.requested:
        state.requested = requested
        state.stable_frames = 1
    else:
        state.stable_frames += 1

    if requested == current:
        return None

    if _past_dead_band(requested, current):
        state.stable_frames = 0
        return requested
    if state.stable_frames >= AUTO_RESIZE_STABLE_FRAMES:
        state.stable_frames = 0
        return requested
    return None


def _past_dead_band(requested: tuple[int, int], current: tuple[int, int]) -> bool:
    for want, have in zip(requested, current, strict=True):
        if have <= 0:
            return True
        if abs(want - have) / have > AUTO_RESIZE_DEAD_BAND:
            return True
    return False


def plan_render_set(
    costs: dict[str, CostRecord],
    current: str | None,
    displayed: list[str],
    budget: float,
    frame_period_ms: float,
    states: dict[str, ThrottleState],
    enabled: bool,
) -> RenderPlan:
    """Each displayed document's render interval, under one shared GPU budget.

    `displayed` is the frame's render set in its own order; `current` is the document the user
    is looking at when it is in that set. The current document takes the budget first, at
    `k = ceil(cost / (budget x period))`; the others share the remainder at one common fps.
    `states` is mutated in place — the hysteresis counters are the only side effect, and they
    are ephemeral.

    `enabled=False` answers today's set exactly: every interval 1, every phase 0.
    """
    target_fps: float = 1e3 / frame_period_ms if frame_period_ms > 0 else 0.0
    if not enabled:
        for state in states.values():
            state.interval = 1
            state.candidate = 1
            state.agreeing_frames = 0
        return RenderPlan(
            intervals=dict.fromkeys(displayed, 1),
            phases=dict.fromkeys(displayed, 0),
            document_fps=dict.fromkeys(displayed, target_fps),
        )

    budget_ms: float = budget * frame_period_ms
    current_cost: float = _cost_of(costs, current) if current is not None else 0.0
    wanted: dict[str, int] = {}

    current_interval: int = 1
    if current is not None:
        current_interval = _clamp_interval(
            math.ceil(current_cost / budget_ms) if current_cost > budget_ms else 1
        )
        wanted[current] = current_interval

    others: list[str] = [
        document_id for document_id in displayed if document_id != current
    ]
    remainder_ms: float = max(0.0, budget_ms - current_cost / current_interval)
    others_cost: float = sum(_cost_of(costs, document_id) for document_id in others)
    if others_cost <= remainder_ms or others_cost <= 0.0:
        for document_id in others:
            wanted[document_id] = 1
    else:
        # The largest common fps at which every other document's cost fits the remainder of
        # each second: sum(cost) x f <= remainder_ms x target_fps.
        common_fps: float = (remainder_ms * target_fps) / others_cost
        interval = _clamp_interval(
            round(target_fps / common_fps) if common_fps > 0 else 0
        )
        for document_id in others:
            wanted[document_id] = interval

    intervals: dict[str, int] = {}
    phases: dict[str, int] = {}
    fps: dict[str, float] = {}
    for index, document_id in enumerate(displayed):
        state = states.setdefault(document_id, ThrottleState())
        interval = _settle(state, wanted.get(document_id, 1))
        intervals[document_id] = interval
        phases[document_id] = index % interval
        fps[document_id] = target_fps / interval
    return RenderPlan(intervals=intervals, phases=phases, document_fps=fps)


def _cost_of(costs: dict[str, CostRecord], document_id: str) -> float:
    record = costs.get(document_id)
    return record.gpu_ms if record is not None else 0.0


def _clamp_interval(value: int) -> int:
    return min(MAX_INTERVAL, max(1, value))


def _settle(state: ThrottleState, candidate: int) -> int:
    """Move `state.interval` to `candidate` only once it has agreed for the whole window."""
    if candidate == state.interval:
        state.candidate = candidate
        state.agreeing_frames = 0
        return state.interval
    if candidate != state.candidate:
        state.candidate = candidate
        state.agreeing_frames = 1
    else:
        state.agreeing_frames += 1
    if state.agreeing_frames >= INTERVAL_HYSTERESIS_FRAMES:
        state.interval = candidate
        state.agreeing_frames = 0
    return state.interval
