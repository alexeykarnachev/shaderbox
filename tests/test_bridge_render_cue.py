"""The deferred-op "Rendering..." cue timing (feature 020·20 D3, fixed). A deferred render op must
hold one frame (cue paints), then run on the next frame with the cue STILL pending (so it covers the
frozen frame), then clear. The earlier bug cleared _render_pending at the top of the run frame, so the
freeze frame painted no cue. Pure: drain() is driven by hand, no GL / no worker thread."""

from shaderbox.copilot.bridge import CopilotBridge, MainThreadOp


def test_deferred_op_holds_one_frame_then_runs_with_cue_still_up() -> None:
    bridge = CopilotBridge()
    ran: list[str] = []
    op = MainThreadOp(fn=lambda: ran.append("render"), defer=True)
    bridge._ops.put(op)  # type: ignore[attr-defined]  # the worker normally enqueues via run_on_main

    # Frame A (hold): the op must NOT run yet, but the cue must be pending so the overlay paints.
    bridge.drain()
    assert ran == [], "deferred op ran on the hold frame — the cue never gets a frame to paint"
    assert bridge.render_pending() is True, "cue not pending on the hold frame"

    # Frame B (run): the op runs (freezes), and the cue must STILL be pending so it covers the
    # frozen frame at the bottom of this same frame.
    bridge.drain()
    assert ran == ["render"], "deferred op did not run on the frame after the hold"
    assert op.done.is_set() is True, "worker was never unblocked"
    assert bridge.render_pending() is True, (
        "cue cleared on the run frame — the freeze frame would paint no cue (the original bug)"
    )

    # Frame C: nothing left — the cue clears.
    bridge.drain()
    assert bridge.render_pending() is False, "cue stuck on after the render finished"


def test_non_deferred_op_runs_immediately_no_cue() -> None:
    bridge = CopilotBridge()
    ran: list[str] = []
    op = MainThreadOp(fn=lambda: ran.append("x"), defer=False)
    bridge._ops.put(op)  # type: ignore[attr-defined]  # the worker normally enqueues via run_on_main

    bridge.drain()
    assert ran == ["x"], "a non-deferred op must run on its first drain"
    assert op.done.is_set() is True
    assert bridge.render_pending() is False, "a non-deferred op must never raise the render cue"
