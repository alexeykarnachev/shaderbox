"""The copilot brakes react to their config value.

A knob with a reader but no falsifier is indistinguishable from a dead one: the suite stays
green whatever it is set to, so a later change to its semantics has nothing to catch it.
These drive each brake's reader rather than merely reading the field back — a test that
builds its input FROM the cap can only prove the reader agrees with itself.
"""

import numpy as np
import pytest

from shaderbox.copilot.backend import _MOTION_EPS
from shaderbox.copilot.config import COPILOT_ENGINE
from shaderbox.copilot.tools.base import GatePolicy
from shaderbox.copilot.tools.registry import build_registry
from tests._caps import minimal_caps


def test_bulk_gate_policy_has_no_subscribers() -> None:
    # bulk_gate_threshold is unfalsifiable because NOTHING reaches it: no tool declares
    # GatePolicy.BULK, so requires_gate's BULK branch is unreachable in the live registry.
    # If a tool ever adopts BULK this goes red — and the threshold then needs a real
    # falsifier (assert the gate flips either side of the configured count).
    policies = {d.gate_policy for d in build_registry(minimal_caps()).definitions()}
    assert GatePolicy.BULK not in policies, (
        "a tool now uses GatePolicy.BULK — bulk_gate_threshold became reachable and needs "
        "a test that drives the gate across the threshold"
    )


def test_the_bulk_branch_reacts_to_the_threshold_when_reached() -> None:
    # The reader itself, exercised directly so the brake is covered even while no tool opts
    # in. Both sides of the bound, so a threshold change moves the verdict.
    registry = build_registry(minimal_caps())
    definition = registry.definitions()[0]
    object.__setattr__(definition, "gate_policy", GatePolicy.BULK)
    try:
        # The counts are LITERAL, not derived from the config: a test that sizes its own
        # input from the cap moves with the cap and can only prove the reader agrees with
        # itself. These pin the intended number, so changing it is a deliberate edit here.
        assert COPILOT_ENGINE.bulk_gate_threshold == 5
        assert not registry.requires_gate(definition.name, {"ids": ["x"] * 5})
        assert registry.requires_gate(definition.name, {"ids": ["x"] * 6})
    finally:
        object.__setattr__(definition, "gate_policy", GatePolicy.NONE)


@pytest.mark.parametrize(
    "mean_abs_diff,expect_static",
    [(0.0, True), (15.75, False)],
)
def test_motion_eps_separates_static_from_animating(
    mean_abs_diff: float, expect_static: bool
) -> None:
    # The measured diffs from the two real probe call sites straddle the threshold, so both
    # verdicts are naturally producible and a moved _MOTION_EPS flips one of them.
    assert (mean_abs_diff < _MOTION_EPS) is expect_static


def test_motion_eps_is_a_usable_threshold() -> None:
    # A negative or huge epsilon would make one verdict unreachable — the frame-pair probe
    # would report ANIMATES (or STATIC) for every shader ever rendered.
    assert 0.0 < _MOTION_EPS < 255.0
    identical = np.zeros(16, dtype=np.int16)
    assert float(np.mean(np.abs(identical - identical))) < _MOTION_EPS
