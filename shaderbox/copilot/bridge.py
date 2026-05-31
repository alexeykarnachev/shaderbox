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


class CopilotBridge:
    """Worker -> main-thread synchronous round-trip. GL has thread-affinity (only
    the main thread owns the context), but a worker-thread tool sometimes needs a
    GL result mid-turn. The worker hands a closure over and blocks; the frame loop
    drains it once per frame and unblocks the worker with the result.

    The one mechanism the exporters never needed (they marshal output one-way,
    GL-free worker). See ai_docs/features/020_copilot_agent/01_threading_architecture.md.
    """

    def __init__(self) -> None:
        self._ops: queue.Queue[MainThreadOp] = queue.Queue(maxsize=64)
        self._shutdown: threading.Event = threading.Event()

    def reopen(self) -> None:
        # MAIN THREAD, at the start of a turn. Clears a `_shutdown` latched by a prior
        # reusable cancel_all() (a project switch / Stop) so a reused bridge serves
        # again. A real shutdown (release(), reusable=False) is NOT cleared here.
        self._shutdown.clear()

    def run_on_main(self, fn: Callable[[], Any]) -> Any:
        # Called ON THE WORKER THREAD. Enqueue + block until the main thread runs it.
        if self._shutdown.is_set():
            raise CopilotCancelled("copilot shutting down")
        op = MainThreadOp(fn=fn)
        self._ops.put(op)
        if not op.done.wait(COPILOT_CONFIG.bridge_op_timeout_s):
            raise CopilotToolError("main-thread op timed out")
        if op.error is not None:
            raise op.error
        return op.result

    def drain(self, max_ops: int = 8) -> None:
        # Called ON THE MAIN THREAD, once per frame. Bounded so it never starves
        # the frame. A raising op sets op.error (re-raised on the worker) and the
        # frame continues — a buggy tool cannot crash the loop.
        for _ in range(max_ops):
            try:
                op = self._ops.get_nowait()
            except queue.Empty:
                return
            try:
                op.result = op.fn()
            except Exception as e:
                # A raising GL op must not crash the frame loop; the worker re-raises it.
                op.error = e
            finally:
                op.done.set()

    def cancel_all(self, *, reusable: bool = False) -> None:
        # Release any worker blocked in run_on_main so a join can't deadlock. `reusable`
        # (project switch / Stop) leaves the bridge able to serve again after a reopen();
        # the default (release() at shutdown) latches it shut permanently.
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
