import queue
import threading

from loguru import logger

from shaderbox.copilot.bridge import CopilotBridge
from shaderbox.copilot.capabilities import CopilotCapabilities
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.llm.openrouter import OpenRouterLLMClient
from shaderbox.copilot.state import ChatState
from shaderbox.copilot.tools.registry import build_registry

# The package composition root. App owns ONE of these (constructed before the first
# _init, which calls release()) and drives it: bridge.drain() + pump_events() per
# frame, enqueue_turn() on send, release() at shutdown. The worker thread is spawned
# lazily on the first turn (don't start a thread for a user who never opens the chat).


class CopilotSession:
    def __init__(
        self,
        caps: CopilotCapabilities,
        client: OpenRouterLLMClient,
    ) -> None:
        self.caps = caps
        self.client = client
        self.registry = build_registry(caps)
        self.bridge = CopilotBridge()
        self.state = ChatState()
        self._turn_queue: queue.Queue[str] = queue.Queue()
        self._events: queue.Queue[object] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._cancel = threading.Event()

    def pump_events(self) -> None:
        # MAIN THREAD, per frame. Drain worker events into self.state (single-writer:
        # only this method writes state). The event-apply body lands with the agent loop.
        while True:
            try:
                self._events.get_nowait()
            except queue.Empty:
                return

    def drain_bridge(self) -> None:
        # MAIN THREAD, per frame, early (pre-render) so a recompiled node renders this
        # frame. Wrapped by the caller; bridge.drain isolates each op anyway.
        self.bridge.drain()

    def reset_conversation(self) -> None:
        # Project switch: drop the transcript. The caps closures read live app.*, so no
        # reconstruction is needed (skeleton 10 §4).
        self._cancel.set()
        self.state = ChatState()
        self._cancel = threading.Event()

    def release(self) -> None:
        # MAIN THREAD, at shutdown — called at the TOP of App.release(), before the node
        # release, so a queued GL op never runs against half-released nodes. Safe when no
        # worker was ever spawned: cancel_all on an empty queue + join on a None thread
        # are both no-ops.
        self._cancel.set()
        self.bridge.cancel_all()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=COPILOT_CONFIG.worker_join_timeout_s)
            if self._worker.is_alive():
                logger.warning("Copilot worker did not exit in time; abandoning")
