"""What happened when a pass drew — the seam three producers write and three surfaces read.

Feature 100's engine judged its own population and could only RETURN, so nothing read a
verdict: a population reaching a pass with no `flat in` was accepted and dropped forever, a
pass with no population drew fullscreen with no signal, zero entities cleared the canvas in
silence, and `_instances_error` was written by the draw and read by nothing. The research
(101, I1-I7) found EIGHT reachable draw states where three had been written down.

The states are not all decided in the same place, which is why this is a TYPE rather than a
return type on `Pass.render` (102 D4):

    producer                              states it decides
    `Pass.render` / `_upload_instances`   drew-N, fullscreen, empty, refused(why)
    `Pass.compile`                        compile-failed, drawing-stale-after-failed-recompile
    `scripting/engine.py`                 population-to-a-pass-with-no-fields

A return value from the draw alone can express at most five of the eight.

It is PUBLIC because three consumers read it: the per-frame diagnostic surface
(`Document.instanced_outcomes`), the copilot's `probe_render` facts (103 D6), and the
per-pass live count on the uniform panel (104 D3).
"""

from dataclasses import dataclass
from typing import Literal, get_args

# What a pass's draw did this frame. Eight members for the eight reachable states the
# research enumerated by RUNNING each one and reading the frame, not by reasoning about the
# code -- three had been written down before that.
#
# `drew` and `empty` are deliberately distinct though both are "the population was
# accepted": zero entities clears the canvas, which looks like a broken shader and is the
# state I3 names. `fullscreen` and `no_fields` are likewise distinct though both draw the
# quad: the first is a pass that never declared `flat in` and is drawing correctly, the
# second is a population the engine ACCEPTED and dropped, which is I1.
InstancedState = Literal[
    "drew",
    "empty",
    "fullscreen",
    "refused",
    "no_fields",
    "compile_failed",
    "stale_program",
    "not_compiled",
]
INSTANCED_STATES: tuple[InstancedState, ...] = get_args(InstancedState)

# The states that mean something is wrong and someone should be told. `fullscreen` and
# `drew` are the two ordinary outcomes; `not_compiled` is frame one of every document.
# A gate walks `INSTANCED_STATES` and asserts every member is in exactly one tier, so a
# ninth state cannot be added without deciding which it is.
HEALTHY_INSTANCED_STATES: frozenset[InstancedState] = frozenset(
    {"drew", "fullscreen", "not_compiled"}
)


@dataclass(frozen=True)
class InstancedOutcome:
    """One pass's draw outcome for one frame.

    `count` is the instance count for `drew` and 0 for `empty`; None wherever no draw
    happened or the pass is not instanced, so a reader cannot mistake "not applicable" for
    "zero entities" -- which is exactly the distinction I3 exists for.

    `detail` carries the reason for `refused` and the compiler's message for
    `compile_failed`. It is free text because its two producers have nothing in common;
    nothing branches on it.
    """

    pass_name: str
    state: InstancedState
    count: int | None = None
    detail: str = ""

    @property
    def is_healthy(self) -> bool:
        return self.state in HEALTHY_INSTANCED_STATES

    def describe(self) -> str:
        """One line for a diagnostic surface. The count is in the text for `drew` and
        `empty` because a surface showing only the state loses the number 104 D3 needs."""
        if self.state == "drew":
            return f"{self.pass_name}: drew {self.count} entities"
        if self.state == "empty":
            return f"{self.pass_name}: population is empty -- nothing drawn"
        if self.state == "fullscreen":
            return f"{self.pass_name}: full-screen quad"
        if self.state == "refused":
            return f"{self.pass_name}: population refused -- {self.detail}"
        if self.state == "no_fields":
            return (
                f"{self.pass_name}: got a population but declares no `flat in` fields -- "
                f"it was dropped"
            )
        if self.state == "compile_failed":
            return f"{self.pass_name}: compile failed -- {self.detail}"
        if self.state == "stale_program":
            return (
                f"{self.pass_name}: drawing the PREVIOUS program -- the current source "
                f"failed to compile"
            )
        return f"{self.pass_name}: not compiled yet"
