"""The deferred-op "Rendering..." cue timing (feature 020·20 D3). A deferred render op is PARKED by
drain() (cue pending, op not run), then run by run_deferred_render() at the post-swap + gl.finish
firing point so the cue covers the frozen frame, then cleared. This is the same firing point the
Render/Share tabs' RenderDefer uses — one place all render encodes freeze the loop, after the cue is
provably on the glass. Pure: drain() / run_deferred_render() are driven by hand, no GL / no worker."""

from shaderbox.copilot.bridge import CopilotBridge, MainThreadOp


def test_deferred_op_is_parked_then_runs_at_the_post_swap_point() -> None:
    bridge = CopilotBridge()
    ran: list[str] = []
    op = MainThreadOp(fn=lambda: ran.append("render"), defer=True)
    bridge._ops.put(op)  # type: ignore[attr-defined]  # the worker normally enqueues via run_on_main

    # drain() (top of frame): the op must be PARKED, not run — so the cue gets the frame to paint.
    bridge.drain()
    assert ran == [], "deferred op ran in drain() — the cue never gets a frame to paint"
    assert bridge.render_pending() is True, "cue not pending after the op was parked"

    # run_deferred_render() (post-swap, after gl.finish): the op runs (freezes) with the cue STILL
    # pending so it covers the frozen frame; then it clears.
    bridge.run_deferred_render()
    assert ran == ["render"], "parked op did not run at the post-swap firing point"
    assert op.done.is_set() is True, "worker was never unblocked"
    assert bridge.render_pending() is False, "cue stuck on after the render finished"


def test_non_deferred_op_runs_immediately_no_cue() -> None:
    bridge = CopilotBridge()
    ran: list[str] = []
    op = MainThreadOp(fn=lambda: ran.append("x"), defer=False)
    bridge._ops.put(op)  # type: ignore[attr-defined]  # the worker normally enqueues via run_on_main

    bridge.drain()
    assert ran == ["x"], "a non-deferred op must run on its first drain"
    assert op.done.is_set() is True
    assert bridge.render_pending() is False, (
        "a non-deferred op must never raise the render cue"
    )


def test_cancel_all_releases_a_parked_render_op() -> None:
    # A parked render op lives outside _ops; cancel must release its blocked worker or a join past
    # it deadlocks on shutdown.
    bridge = CopilotBridge()
    op = MainThreadOp(fn=lambda: None, defer=True)
    bridge._ops.put(op)  # type: ignore[attr-defined]
    bridge.drain()
    assert bridge.render_pending() is True

    bridge.cancel_all()
    assert op.done.is_set() is True, (
        "parked op's worker was never released by cancel_all"
    )
    assert op.error is not None, "cancelled parked op must carry the cancel error"
    assert bridge.render_pending() is False, "cancel must clear the parked render"
