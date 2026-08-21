"""The copilot brakes react to their config value.

A knob with a reader but no falsifier is indistinguishable from a dead one: the suite stays
green whatever it is set to, so a later change to its semantics has nothing to catch it.
These drive each brake's reader rather than merely reading the field back — a test that
builds its input FROM the cap can only prove the reader agrees with itself.
"""

import numpy as np
import pytest

from shaderbox.copilot.backend import _MOTION_EPS
from shaderbox.copilot.tools.base import GatePolicy
from shaderbox.copilot.tools.registry import build_registry
from tests._caps import minimal_caps


def test_gating_is_a_two_state_decision() -> None:
    # GatePolicy.BULK ("confirm when a list arg exceeds bulk_gate_threshold") was built in
    # 020 and never adopted: no tool ever declared it, so requires_gate's BULK branch was
    # unreachable and its threshold unfalsifiable. Removed in 061 along with the knob. What
    # remains is a two-state decision, and this pins that — a third policy would need a
    # reachable reader and a test that drives it.
    assert {p.name for p in GatePolicy} == {"NONE", "ALWAYS"}

    registry = build_registry(minimal_caps())
    policies = {d.gate_policy for d in registry.definitions()}
    assert policies <= {GatePolicy.NONE, GatePolicy.ALWAYS}
    # Both states are actually in use — a registry where every tool gated (or none did)
    # would pass a subset check while meaning the gate had stopped discriminating.
    assert policies == {GatePolicy.NONE, GatePolicy.ALWAYS}

    gated = [d.name for d in registry.definitions() if registry.requires_gate(d.name)]
    assert "delete_node" in gated
    assert "read_shader" not in gated


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
