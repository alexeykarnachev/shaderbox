import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.errors import CopilotCancelled, CopilotToolError

# Worker->main marshalling seam for GL/main-thread WORK (run a closure, return its result).
# Sibling of `gate.py`, the same blocking round-trip for USER answers (confirm/credential).


@dataclass
class MainThreadOp:
    # A unit of GL/main-thread work. The worker fills `fn`, pushes it, and blocks
    # on `done`; the main thread runs `fn`, stores `result`/`error`, sets `done`.
    fn: Callable[[], Any]
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None
    # `defer` parks the op for the post-swap firing point (run_deferred_render) instead of running
    # it inline in drain(), so its encode freezes the loop at the SAME place the Render/Share tabs'
    # RenderDefer fires — after a swap + gl.finish, the one point a "Rendering..." cue is provably
    # presented before the freeze. done.wait's clock runs from enqueue, so the ~1-frame park is
    # covered by the long render_op_timeout_s.
    defer: bool = False


class CopilotBridge:
    """Worker -> main-thread synchronous round-trip. GL has thread-affinity (only the
    main thread owns the context); a worker-thread tool sometimes needs a GL result
    mid-turn. The worker hands a closure over and blocks; the frame loop drains it once
    per frame and unblocks the worker with the result.
    """

    def __init__(self) -> None:
        self._ops: queue.Queue[MainThreadOp] = queue.Queue(maxsize=64)
        self._shutdown: threading.Event = threading.Event()
        # A `defer=True` op parked by drain() for run_deferred_render() to fire post-swap. One slot
        # is enough — the worker is single-threaded and blocks on this op, so nothing queues behind
        # it. The UI paints the "Rendering..." cue while it's set (render_pending()).
        self._deferred_render: MainThreadOp | None = None

    def reopen(self) -> None:
        # MAIN THREAD. Clears a `_shutdown` latched by a reusable cancel_all() so a reused
        # bridge serves again. A real shutdown (reusable=False) is NOT cleared here.
        self._shutdown.clear()

    def run_on_main(
        self, fn: Callable[[], Any], timeout: float | None = None, defer: bool = False
    ) -> Any:
        # WORKER THREAD. Enqueue + block until the main thread runs it. `timeout` overrides the
        # default for ops that legitimately freeze the loop longer (a video encode). `defer` parks
        # the op for the post-swap firing point so a "Rendering..." cue paints before the freeze.
        if self._shutdown.is_set():
            raise CopilotCancelled("copilot shutting down")
        op = MainThreadOp(fn=fn, defer=defer)
        self._ops.put(op)
        wait_s = timeout if timeout is not None else COPILOT_CONFIG.bridge_op_timeout_s
        if not op.done.wait(wait_s):
            raise CopilotToolError("main-thread op timed out")
        if op.error is not None:
            raise op.error
        return op.result

    def drain(self, max_ops: int = 8) -> None:
        # MAIN THREAD, top of frame. Bounded so it never starves the frame. A raising op sets
        # op.error (re-raised on the worker) and the frame continues — a buggy tool can't crash it.
        # A `defer=True` op is PARKED (not run) for run_deferred_render() to fire post-swap; draining
        # stops there since the single-threaded worker blocks on it and queues nothing behind it.
        for _ in range(max_ops):
            try:
                op = self._ops.get_nowait()
            except queue.Empty:
                return
            if op.defer:
                self._deferred_render = op
                return
            try:
                op.result = op.fn()
            except Exception as e:
                # A raising GL op must not crash the frame loop; the worker re-raises it.
                op.error = e
            finally:
                op.done.set()

    def run_deferred_render(self) -> None:
        # MAIN THREAD, post-swap. The caller has drawn the cue, swapped, and gl.finish'd it onto the
        # glass; this runs the parked render op — the freeze — with the cue provably presented. The
        # swap+finish IS the guarantee (not a frame count), so this fires the same frame drain()
        # parked it. Cleared after, so the next frame paints no cue (render_pending goes False).
        op = self._deferred_render
        if op is None:
            return
        self._deferred_render = None
        try:
            op.result = op.fn()
        except Exception as e:
            op.error = e
        finally:
            op.done.set()

    def render_pending(self) -> bool:
        # MAIN THREAD: True while a render op is parked — the UI draws the cue over the next frame,
        # which run_deferred_render() then freezes.
        return self._deferred_render is not None

    def cancel_all(self, *, reusable: bool = False) -> None:
        # Release any worker blocked in run_on_main so a join can't deadlock. `reusable` leaves
        # the bridge able to serve again after a reopen(); the default latches it shut permanently.
        if not reusable:
            self._shutdown.set()
        parked = self._deferred_render
        self._deferred_render = None
        if parked is not None:
            # A parked render op lives outside _ops; release its blocked worker too, else a join
            # past it deadlocks.
            parked.error = CopilotCancelled("copilot shutting down")
            parked.done.set()
        while True:
            try:
                op = self._ops.get_nowait()
            except queue.Empty:
                break
            op.error = CopilotCancelled("copilot shutting down")
            op.done.set()
        logger.debug("Copilot bridge cancelled")
