import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.errors import CopilotCancelled, CopilotToolError


@dataclass
class MainThreadOp:
    # A unit of GL/main-thread work. The worker fills `fn`, pushes it, and blocks
    # on `done`; the main thread runs `fn`, stores `result`/`error`, sets `done`.
    fn: Callable[[], Any]
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None
    # `defer` holds the op back ONE frame in drain() so a "Rendering..." modal paints before the
    # freeze. One frame only — done.wait's clock runs from enqueue, a longer hold eats the timeout.
    # `_held` records the single allowed deferral was spent.
    defer: bool = False
    _held: bool = False


class CopilotBridge:
    """Worker -> main-thread synchronous round-trip. GL has thread-affinity (only the
    main thread owns the context); a worker-thread tool sometimes needs a GL result
    mid-turn. The worker hands a closure over and blocks; the frame loop drains it once
    per frame and unblocks the worker with the result.
    """

    def __init__(self) -> None:
        self._ops: queue.Queue[MainThreadOp] = queue.Queue(maxsize=64)
        self._shutdown: threading.Event = threading.Event()
        # True across the hold frame AND the frame that runs the deferred render op — the UI paints
        # the "Rendering..." cue off this. The next empty drain clears it.
        self._render_pending: bool = False

    def reopen(self) -> None:
        # MAIN THREAD. Clears a `_shutdown` latched by a reusable cancel_all() so a reused
        # bridge serves again. A real shutdown (reusable=False) is NOT cleared here.
        self._shutdown.clear()

    def run_on_main(
        self, fn: Callable[[], Any], timeout: float | None = None, defer: bool = False
    ) -> Any:
        # WORKER THREAD. Enqueue + block until the main thread runs it. `timeout` overrides the
        # default for ops that legitimately freeze the loop longer (a video encode). `defer` holds
        # the op one frame so a "Rendering..." modal paints before the freeze.
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
        # MAIN THREAD, once per frame. Bounded so it never starves the frame. A raising op sets
        # op.error (re-raised on the worker) and the frame continues — a buggy tool can't crash it.
        for _ in range(max_ops):
            try:
                op = self._ops.get_nowait()
            except queue.Empty:
                # Nothing ran this frame — the encode has returned; clear the cue.
                self._render_pending = False
                return
            if op.defer and not op._held:
                # First sight of a render op: hold it ONE frame so the "Rendering..." cue paints,
                # then stop draining so it runs at the top of the NEXT frame. Safe to return — the
                # worker is single-threaded and blocks on this op, so nothing queues behind it.
                op._held = True
                self._render_pending = True
                self._ops.put(op)
                return
            try:
                op.result = op.fn()
            except Exception as e:
                # A raising GL op must not crash the frame loop; the worker re-raises it.
                op.error = e
            finally:
                op.done.set()
            if op.defer:
                # The held render just ran. Stop draining (keeping _render_pending True) so the cue
                # paints over this freeze frame; without the early return an Empty get would clear
                # the flag in this same frame. The next frame's empty drain clears it.
                return

    def render_pending(self) -> bool:
        # MAIN THREAD: True while a render op is held/running — the UI draws the modal.
        return self._render_pending

    def cancel_all(self, *, reusable: bool = False) -> None:
        # Release any worker blocked in run_on_main so a join can't deadlock. `reusable` leaves
        # the bridge able to serve again after a reopen(); the default latches it shut permanently.
        if not reusable:
            self._shutdown.set()
        while True:
            try:
                op = self._ops.get_nowait()
            except queue.Empty:
                break
            op.error = CopilotCancelled("copilot shutting down")
            op.done.set()
        logger.debug("Copilot bridge cancelled")
