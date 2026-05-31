import queue
import threading

from loguru import logger

from shaderbox.copilot.agent import (
    AgentCancelled,
    AgentError,
    AgentEvent,
    AgentStatus,
    AgentTextDelta,
    AgentToolCard,
    AgentTurnDone,
    run_turn,
)
from shaderbox.copilot.bridge import CopilotBridge
from shaderbox.copilot.capabilities import CopilotCapabilities
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.context import build_context
from shaderbox.copilot.errors import CopilotConfigError
from shaderbox.copilot.gate import GateChannel
from shaderbox.copilot.llm.api import LLMMessage
from shaderbox.copilot.llm.openrouter import OpenRouterLLMClient
from shaderbox.copilot.state import ChatState, Message
from shaderbox.copilot.tools.registry import build_registry

# The package composition root. App owns ONE of these (constructed before the first
# _init, which calls release()) and drives it: bridge.drain() + pump_events() per
# frame, enqueue_turn() on send, release() at shutdown. The worker thread is spawned
# lazily on the first turn (don't start a thread for a user who never opens the chat).

_SHUTDOWN = object()  # sentinel on the turn queue to unblock the worker's get() at exit


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
        self.gate = GateChannel()
        self.state = ChatState()
        # Conversation memory across turns. Owned + mutated ONLY by the worker thread
        # (read at turn start, appended at turn end) — never by the UI. Slice 1 keeps it
        # in memory; per-project persistence is a later slice.
        self.history: list[LLMMessage] = []
        self._turn_queue: queue.Queue[str | object] = queue.Queue()
        self._events: queue.Queue[AgentEvent] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._cancel = threading.Event()
        self._released = (
            False  # True only after release() — gates the shutdown sentinel
        )

    # ---- main thread: enqueue + drain ----

    def enqueue_turn(self, user_text: str) -> None:
        # MAIN THREAD (on Send). The editor flush + lock happens App-side BEFORE this
        # call (App.copilot_send) so the worker's first read sees consistent state.
        self.state.messages.append(Message(role="user", text=user_text))
        self.state.streaming_text = ""
        self.state.in_flight = True
        # Clear a `_shutdown` the bridge may have latched from the first _init's release()
        # (App._init tears down prior state before reuse) — else every GL tool this turn
        # raises "copilot shutting down".
        self.bridge.reopen()
        self._ensure_worker()
        self._turn_queue.put(user_text)
        logger.info("copilot enqueue_turn: queued + worker ensured")

    def cancel_turn(self) -> None:
        # MAIN THREAD (Stop button). The loop checks cancel.is_set() between steps; the
        # gate's cancel_all releases a worker blocked on a pending widget. reusable: the
        # channel must stay live for the next turn.
        self._cancel.set()
        self.gate.cancel_all(reusable=True)

    def pump_events(self) -> None:
        # MAIN THREAD, per frame. Drain worker events into self.state — the ONLY writer
        # of state (single-writer invariant, skeleton Seam-E).
        while True:
            try:
                ev = self._events.get_nowait()
            except queue.Empty:
                return
            self._apply_event(ev)

    def _apply_event(self, ev: AgentEvent) -> None:
        match ev:
            case AgentTextDelta():
                self.state.streaming_text += ev.text
            case AgentStatus():
                pass  # per-tool cards (below) carry the durable line; status is transient
            case AgentToolCard():
                verb = "ok" if ev.ok else "failed"
                self.state.messages.append(
                    Message(role="tool_status", text=f"{ev.name}: {verb}")
                )
            case AgentError():
                self.state.messages.append(Message(role="error", text=ev.message))
                self._finish_turn()
            case AgentTurnDone():
                if self.state.streaming_text:
                    self.state.messages.append(
                        Message(role="assistant", text=self.state.streaming_text)
                    )
                self.state.streaming_text = ""
                self.state.usage.add(ev.usage)
                self._finish_turn()
            case AgentCancelled():
                self.state.streaming_text = ""
                self._finish_turn()

    def _finish_turn(self) -> None:
        self.state.in_flight = False

    def drain_bridge(self) -> None:
        # MAIN THREAD, per frame, early (pre-render) so a recompiled node renders this
        # frame. Wrapped by the caller; bridge.drain isolates each op anyway.
        self.bridge.drain()

    # ---- worker thread ----

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._run_worker, name="copilot-worker", daemon=False
        )
        self._worker.start()

    def _run_worker(self) -> None:
        logger.info("copilot worker thread started")
        while True:
            item = self._turn_queue.get()
            if item is _SHUTDOWN:
                if self._released:
                    logger.info("copilot worker thread exiting (shutdown)")
                    return
                # A stale sentinel from a prior release() on this REUSED session — the
                # app is still live (not released), so this worker must keep serving.
                # Swallow it instead of dying, or the next turn is stranded.
                logger.info("copilot worker: ignoring stale shutdown sentinel")
                continue
            assert isinstance(item, str)
            self._cancel.clear()  # a fresh turn starts uncancelled
            self._run_one_turn(item)

    def _run_one_turn(self, user_text: str) -> None:
        context = build_context(self.caps)
        assistant_text = ""
        try:
            for ev in run_turn(
                self.client,
                self.registry,
                COPILOT_CONFIG,
                context,
                self.history,
                user_text,
                self.gate,
                self._cancel,
            ):
                if isinstance(ev, AgentTextDelta):
                    assistant_text += ev.text
                self._events.put(ev)
        except CopilotConfigError as e:
            logger.warning(f"Copilot turn aborted: {e}")
            self._events.put(AgentError(str(e)))
            return
        except Exception:
            logger.exception("Copilot turn failed")
            self._events.put(AgentError("copilot turn failed (see logs)"))
            return
        # Commit the turn to history (worker is the sole owner) so the next turn carries
        # context. The current shader source is NOT stored here — it is re-fetched live
        # each turn via get_current_shader (§B1).
        self.history.append(LLMMessage(role="user", content=user_text))
        self.history.append(
            LLMMessage(role="assistant", content=assistant_text or None)
        )

    # ---- lifecycle ----

    def reset_conversation(self) -> None:
        # Project switch: drop the transcript + history. The caps closures read live
        # app.*, so no reconstruction is needed (skeleton 10 §4). reusable: the same
        # session serves the next project, so the gate must NOT latch shut.
        self._cancel.set()
        self.gate.cancel_all(reusable=True)
        # Drain any queued turns / a stale shutdown sentinel left by a prior release()
        # on this reused session, so nothing strands the next turn.
        while True:
            try:
                self._turn_queue.get_nowait()
            except queue.Empty:
                break
        self.state = ChatState()
        self.history = []
        self._cancel = threading.Event()

    def release(self) -> None:
        # MAIN THREAD, at shutdown — called at the TOP of App.release(), before the node
        # release, so a queued GL op never runs against half-released nodes. Safe when no
        # worker was ever spawned: the sentinel + cancel_all + join on a None thread are
        # all no-ops.
        self._released = True
        self._cancel.set()
        self.bridge.cancel_all()
        self.gate.cancel_all()
        self._turn_queue.put(_SHUTDOWN)  # unblock worker.get() so join can return
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=COPILOT_CONFIG.worker_join_timeout_s)
            if self._worker.is_alive():
                logger.warning("Copilot worker did not exit in time; abandoning")
