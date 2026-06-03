import queue
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from loguru import logger

from shaderbox.copilot.agent import (
    AgentCancelled,
    AgentError,
    AgentEvent,
    AgentGateOpened,
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
from shaderbox.copilot.gate import GateChannel, GateResponse
from shaderbox.copilot.llm.api import LLMMessage
from shaderbox.copilot.llm.openrouter import OpenRouterLLMClient
from shaderbox.copilot.persistence import ConversationStore
from shaderbox.copilot.state import ChatState, Message, RecoverInfo
from shaderbox.copilot.tools.registry import build_registry
from shaderbox.copilot.trace import TraceLog, new_trace_log


def _trace_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")


def _slugify(name: str) -> str:
    # Project-dir name -> a filesystem-safe trace-filename slug (the trace file is
    # copilot_<slug>_<stamp>.transcript). Keep alnum/dash/underscore, collapse the rest.
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in name).strip("-")
    return safe or "project"


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
        get_project_slug: Callable[[], str],
    ) -> None:
        self.caps = caps
        self.client = client
        self._get_project_slug = get_project_slug
        self.registry = build_registry(caps)
        self.bridge = CopilotBridge()
        self.gate = GateChannel()
        self.state = ChatState()
        # Conversation memory across turns. Owned + mutated ONLY by the worker thread
        # (read at turn start, appended at turn end) — never by the UI. Persisted
        # per-project (feature 022) via App, which reads it at a quiescent point.
        self.history: list[LLMMessage] = []
        self._turn_queue: queue.Queue[str | object] = queue.Queue()
        self._events: queue.Queue[AgentEvent] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._cancel = threading.Event()
        # Set ONLY by reset_conversation / release (project teardown), never by the Stop
        # button: tells a worker finishing an aborted turn to DROP its history commit so it
        # doesn't land on the next project's freshly-reset history (feature 022). A user
        # Stop (cancel_turn) does NOT set this, so a stopped turn's partial reply still
        # commits to THIS conversation's context.
        self._drop_turn = threading.Event()
        self._released = (
            False  # True only after release() — gates the shutdown sentinel
        )
        # Full-transcript sink (prompts / responses / tool calls, in full) — a dedicated
        # per-session file, separate from the regular log stream (trace.py). Lazily
        # opened; a fresh file per project switch (reset_conversation).
        self.trace = self._new_trace()

    def _new_trace(self) -> TraceLog:
        return new_trace_log(_slugify(self._get_project_slug()), _trace_stamp())

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
        logger.debug("copilot enqueue_turn: queued + worker ensured")

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
            case AgentGateOpened():
                # Dequeue the GatePending (keeps _pending in lockstep with _current) and
                # materialize the confirm card. The UI draws Yes/No from this Message; the
                # answer flows back via answer_gate -> gate.answer (feature 020·17 §7.1).
                self.gate.take_pending()
                self.state.messages.append(
                    Message(role="pending_action", text=ev.request.prompt)
                )
            case AgentToolCard():
                verb = "ok" if ev.ok else "failed"
                self.state.messages.append(
                    Message(role="tool_status", text=f"{ev.name}: {verb}")
                )
                # A successful delete attaches the Recover affordance to its (resolved-Yes)
                # confirm card — the gated tool's card always trails the open pending_action.
                if ev.name == "delete_node" and ev.ok and ev.payload is not None:
                    self._attach_recover(ev.payload)
            case AgentError():
                self.state.messages.append(Message(role="error", text=ev.message))
                self._resolve_open_gate_card("cancelled")
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
                self._resolve_open_gate_card("cancelled")
                self._finish_turn()

    def _open_gate_card(self) -> Message | None:
        # The trailing pending_action Message still awaiting an answer (resolved==False).
        # There is only ever one open gate (_current), so the last unresolved card is it.
        for msg in reversed(self.state.messages):
            if msg.role == "pending_action" and not msg.resolved:
                return msg
        return None

    def _resolve_open_gate_card(self, outcome: str) -> None:
        # A cancel/error mid-gate must mark the open confirm card resolved, or it renders
        # live Yes/No buttons forever (feature 020·17). outcome = the terminal label.
        card = self._open_gate_card()
        if card is not None:
            card.resolved = True
            card.text = f"{card.text}\n({outcome})"

    def _attach_recover(self, payload: dict) -> None:
        # Attach RecoverInfo to the resolved-Yes delete card (the trailing resolved
        # pending_action with no recover yet). node_id/trash_name come from the tool payload.
        node_id = str(payload.get("node_id", ""))
        trash_name = str(payload.get("trash_name", ""))
        if not node_id or not trash_name:
            return
        for msg in reversed(self.state.messages):
            if msg.role == "pending_action" and msg.resolved and msg.recover is None:
                msg.recover = RecoverInfo(
                    node_id=node_id,
                    node_name=str(payload.get("deleted_name", "")),
                    trash_name=trash_name,
                )
                return

    def answer_gate(self, approved: bool) -> None:
        # MAIN THREAD (Yes/No click). Mark the open confirm card resolved + unblock the worker.
        card = self._open_gate_card()
        if card is not None:
            card.resolved = True
            card.text = f"{card.text}\nYou chose: {'Yes' if approved else 'No'}"
        self.gate.answer(
            GateResponse(approved=approved, option="Yes" if approved else "No")
        )

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
        logger.debug("copilot worker thread started")
        while True:
            item = self._turn_queue.get()
            if item is _SHUTDOWN:
                if self._released:
                    logger.debug("copilot worker thread exiting (shutdown)")
                    return
                # A stale sentinel from a prior release() on this REUSED session — the
                # app is still live (not released), so this worker must keep serving.
                # Swallow it instead of dying, or the next turn is stranded.
                logger.debug("copilot worker: ignoring stale shutdown sentinel")
                continue
            assert isinstance(item, str)
            self._cancel.clear()  # a fresh turn starts uncancelled
            self._drop_turn.clear()  # ...and committable (teardown re-sets it)
            self._run_one_turn(item)

    def _run_one_turn(self, user_text: str) -> None:
        context = build_context(self.caps)
        assistant_text = ""
        error_text = ""
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
                self.trace,
            ):
                if isinstance(ev, AgentTextDelta):
                    assistant_text += ev.text
                elif isinstance(ev, AgentError):
                    error_text = ev.message
                self._events.put(ev)
        except CopilotConfigError as e:
            logger.warning(f"Copilot turn aborted: {e}")
            self._events.put(AgentError(str(e)))
            return
        except Exception:
            logger.exception("Copilot turn failed")
            self._events.put(AgentError("copilot turn failed (see logs)"))
            return
        # A turn aborted by project teardown (reset_conversation / release set _drop_turn)
        # must NOT commit — run_turn returns NORMALLY on cancel, so without this the
        # in-flight turn's tail would land on the next project's freshly-reset history
        # (feature 022 / the trace-bleed deferral's orphaned-append half). A user Stop does
        # NOT set _drop_turn, so its partial reply still commits to this conversation.
        if self._drop_turn.is_set():
            return
        # Commit the turn to history (worker is the sole owner) so the next turn carries
        # context. The current shader source is NOT stored here — it is re-fetched live
        # each turn via get_current_shader (§B1). A turn that ended in an AgentError (a
        # giveup / cutoff) carries that note as the assistant's last word, so the next
        # turn's context shows the agent stopped instead of an empty assistant turn.
        self.history.append(LLMMessage(role="user", content=user_text))
        self.history.append(
            LLMMessage(role="assistant", content=assistant_text or error_text or None)
        )

    # ---- lifecycle ----

    def reset_conversation(self) -> None:
        # Project switch / clear: drop the transcript + history. The caps closures read
        # live app.*, so no reconstruction is needed (skeleton 10 §4). reusable: the same
        # session serves the next project, so the gate must NOT latch shut.
        # INVARIANT (feature 022): callers gate this on `not in_flight` (open_project +
        # the clear button both refuse while a turn runs), so the worker is idle here and
        # no join is needed. Warn if that invariant is ever violated.
        if self.state.in_flight:
            logger.warning(
                "reset_conversation called mid-turn — caller bypassed the in_flight gate"
            )
        self._drop_turn.set()  # a worker finishing an aborted turn must not commit it
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
        # A fresh transcript for the project we are switching INTO — its name carries
        # the new project's slug, and the prior project's transcript is left closed
        # on disk (retention prunes it later).
        self.trace.close()
        self.trace = self._new_trace()

    def save_conversation(self, path: Path) -> None:
        # MAIN THREAD, at a quiescent point (not in_flight): snapshot state + history to
        # disk (feature 022). Both arrays are consistent here — the worker is idle.
        self.pump_events()  # drain any terminal event so the last turn's bubble + usage land
        store = ConversationStore.from_runtime(self.state, self.history)
        store.save(path)

    def load_conversation(self, store: ConversationStore) -> None:
        # MAIN THREAD, after reset_conversation (fresh session): restore the parsed store
        # into state + history + usage (feature 022). Builds on reset's clean slate.
        self.state.messages = store.to_messages()
        self.state.usage = store.to_usage()
        self.history = store.to_history()
        if self.state.messages or self.history:
            self.trace.event(
                "conversation_loaded",
                messages=len(self.state.messages),
                history=len(self.history),
            )

    def release(self) -> None:
        # MAIN THREAD, at shutdown — called at the TOP of App.release(), before the node
        # release, so a queued GL op never runs against half-released nodes. Safe when no
        # worker was ever spawned: the sentinel + cancel_all + join on a None thread are
        # all no-ops.
        self._released = True
        self._drop_turn.set()  # abandon any in-flight turn's commit at teardown
        self._cancel.set()
        self.bridge.cancel_all()
        self.gate.cancel_all()
        self._turn_queue.put(_SHUTDOWN)  # unblock worker.get() so join can return
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=COPILOT_CONFIG.worker_join_timeout_s)
            if self._worker.is_alive():
                logger.warning("Copilot worker did not exit in time; abandoning")
        self.trace.close()
