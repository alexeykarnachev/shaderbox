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
    TurnSummary,
    run_turn,
)
from shaderbox.copilot.bridge import CopilotBridge
from shaderbox.copilot.capabilities import CopilotCapabilities
from shaderbox.copilot.checkpoint import CheckpointStore
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.errors import CopilotConfigError
from shaderbox.copilot.gate import GateChannel, GateResponse
from shaderbox.copilot.llm.api import LLMMessage
from shaderbox.copilot.llm.openrouter import OpenRouterLLMClient
from shaderbox.copilot.persistence import ConversationStore
from shaderbox.copilot.prompt import render_working_set
from shaderbox.copilot.prompt_context import build_context
from shaderbox.copilot.sanitize import sanitize_display
from shaderbox.copilot.state import (
    ChatState,
    Message,
    RecoverInfo,
    StepRecord,
)
from shaderbox.copilot.tools.base import mask_secret
from shaderbox.copilot.tools.registry import build_registry
from shaderbox.copilot.trace import TraceLog, new_trace_log


def _trace_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")


def _slugify(name: str) -> str:
    # Project-dir name -> filesystem-safe trace-filename slug. Keep alnum/dash/underscore.
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in name).strip("-")
    return safe or "project"


def _first_line(text: str, limit: int = 80) -> str:
    # The user-text excerpt shown in the revert confirm modal + notice.
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line[:limit] + ("..." if len(line) > limit else "")


def _tool_card_outcome(ev: AgentToolCard) -> str:
    # The terse, human consequence after the verb. Failures are just "failed" (the agent holds
    # the reason). A compile result (mutating shader tools, create_node, read_shader) collapses
    # to clean / N-errors; grep shows its hit count. Otherwise no outcome (the verb stands alone).
    if not ev.ok:
        return "failed"
    payload = ev.payload or {}
    if "errors" in payload:
        n = len(payload.get("errors", []))
        return f"{n} compile error{'s' if n != 1 else ''}" if n else "compiled clean"
    if "hits" in payload:
        n = int(payload.get("hits", 0))
        return f"{n} match{'es' if n != 1 else ''}"
    return ""


def _tool_card_line(ev: AgentToolCard, verb: str) -> str:
    outcome = _tool_card_outcome(ev)
    return f"{verb}  -  {outcome}" if outcome else verb


# Composition root. App owns ONE; drives it bridge.drain() + pump_events() per frame,
# enqueue_turn() on send, release() at shutdown. Worker thread spawned lazily on first turn.

_SHUTDOWN = object()  # sentinel on the turn queue to unblock the worker's get() at exit


class CopilotSession:
    def __init__(
        self,
        caps: CopilotCapabilities,
        client: OpenRouterLLMClient,
        get_project_slug: Callable[[], str],
        get_checkpoints_root: Callable[[], Path],
    ) -> None:
        self.caps = caps
        self.client = client
        self._get_project_slug = get_project_slug
        self._get_checkpoints_root = get_checkpoints_root
        self.registry = build_registry(caps)
        self.bridge = CopilotBridge()
        self.gate = GateChannel()
        self.state = ChatState()
        # Per-turn rollback checkpoints (feature 020·30). Built lazily: the project dir isn't
        # known at session construction (App sets it later), and reset_conversation rebuilds it
        # for the project we switch INTO. None until first accessed via `checkpoints`.
        self._checkpoints: CheckpointStore | None = None
        self._turn_seq = 0
        # Conversation memory. Owned + mutated ONLY by the worker thread (read at turn
        # start, appended at turn end) — never by the UI.
        self.history: list[LLMMessage] = []
        self._turn_queue: queue.Queue[str | object] = queue.Queue()
        self._events: queue.Queue[AgentEvent] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._cancel = threading.Event()
        # Set ONLY by reset_conversation / release (teardown), never by Stop: tells a worker
        # finishing an aborted turn to DROP its history commit so it doesn't land on the next
        # project's reset history. A user Stop still commits its partial reply to THIS context.
        self._drop_turn = threading.Event()
        self._released = (
            False  # True only after release() — gates the shutdown sentinel
        )
        # Full-transcript sink, per-session file separate from the log stream. Fresh file
        # per project switch (reset_conversation).
        self.trace = self._new_trace()

    def _new_trace(self) -> TraceLog:
        return new_trace_log(_slugify(self._get_project_slug()), _trace_stamp())

    @property
    def checkpoints(self) -> CheckpointStore:
        # Lazy first build (rehydrates the active project's persisted checkpoints); rebuilt on a
        # project switch by reset_conversation.
        if self._checkpoints is None:
            self._checkpoints = CheckpointStore(self._get_checkpoints_root())
        return self._checkpoints

    # ---- main thread: enqueue + drain ----

    def enqueue_turn(self, user_text: str) -> None:
        # MAIN THREAD (on Send). The editor flush + lock happens App-side BEFORE this call so
        # the worker's first read sees consistent state.
        self._turn_seq += 1
        turn_id = f"turn_{self._turn_seq:04d}_{_trace_stamp()}"
        self.checkpoints.open(turn_id, _first_line(user_text))
        self.state.messages.append(
            Message(role="user", text=user_text, turn_id=turn_id)
        )
        # The turn's compact snippet: tool cards append a StepRecord (the square bar), the terminal
        # event writes snippet_stats (the mini-stat line). One per turn, trailing its user message.
        self.state.messages.append(Message(role="turn_snippet"))
        self.state.streaming_text = ""
        self.state.in_flight = True
        # Clear a `_shutdown` the bridge + gate may have latched from a prior release() on this
        # reused session — else every GL tool raises "copilot shutting down" and gate.ask()
        # returns cancelled immediately.
        self.bridge.reopen()
        self.gate.reopen()
        self._ensure_worker()
        self._turn_queue.put(user_text)
        logger.debug("copilot enqueue_turn: queued + worker ensured")

    def cancel_turn(self) -> None:
        # MAIN THREAD (Stop button). The loop checks cancel.is_set() between steps; cancel_all
        # releases a worker blocked on a pending widget. reusable: channel stays live next turn.
        self._cancel.set()
        self.gate.cancel_all(reusable=True)

    def is_cancelled(self) -> bool:
        # ANY THREAD. Reads the CURRENT _cancel — reset_conversation REPLACES the event, so a
        # captured reference would go stale.
        return self._cancel.is_set()

    def pump_events(self) -> None:
        # MAIN THREAD, per frame. Drain worker events into self.state — the ONLY writer of
        # state (single-writer invariant).
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
                # Transient in-flight status line; the durable per-tool card lands below.
                self.state.status = ev.text
            case AgentGateOpened():
                # Dequeue the GatePending (keeps _pending in lockstep with _current) and
                # materialize the gate card. gate_kind picks the widget (CONFIRM Yes/No vs
                # CREDENTIAL masked input); the answer flows back via answer_gate /
                # answer_gate_credential -> gate.answer.
                self.gate.take_pending()
                self.state.messages.append(
                    Message(
                        role="pending_action",
                        text=ev.request.prompt,
                        gate_kind=ev.request.kind,
                        gate_integration=ev.request.secret_field,
                    )
                )
            case AgentToolCard():
                # Each step folds into the turn snippet's square bar (one StepRecord = one square).
                # The verbose tool `result` is the AGENT's feedback (already in its history). A tool
                # that surfaces an ACTIONABLE widget (render path / publish URL) keeps its own visible
                # result line — those are rare + worth a click, not noise to collapse.
                snippet = self._current_snippet()
                if snippet is not None:
                    snippet.steps.append(StepRecord(name=ev.name, ok=ev.ok))
                if ev.widget is not None:
                    self.state.messages.append(
                        Message(
                            role="tool_status",
                            text=_tool_card_line(ev, self.registry.label_for(ev.name)),
                            result_widget=ev.widget,
                        )
                    )
                # A successful delete attaches the Recover affordance to its (resolved-Yes)
                # confirm card — the gated tool's card always trails the open pending_action.
                if ev.name == "delete_node" and ev.ok and ev.payload is not None:
                    self._attach_recover(ev.payload)
            case AgentError():
                self.state.messages.append(Message(role="error", text=ev.message))
                self._resolve_open_gate_card("cancelled")
                if ev.stats is not None:
                    self.state.last_turn = ev.stats
                    self.state.session_cost_usd += ev.stats.cost_usd
                    snippet = self._current_snippet()
                    if snippet is not None:
                        snippet.snippet_stats = ev.stats
                self._finish_turn()
            case AgentTurnDone():
                if self.state.streaming_text:
                    self.state.messages.append(
                        Message(
                            role="assistant",
                            text=sanitize_display(self.state.streaming_text),
                        )
                    )
                self.state.streaming_text = ""
                if ev.stats is not None:
                    self.state.last_turn = ev.stats
                    self.state.session_cost_usd += ev.stats.cost_usd
                    snippet = self._current_snippet()
                    if snippet is not None:
                        snippet.snippet_stats = ev.stats
                self._finish_turn()
            case AgentCancelled():
                self.state.streaming_text = ""
                self._resolve_open_gate_card("cancelled")
                if ev.stats is not None:
                    self.state.last_turn = ev.stats
                    self.state.session_cost_usd += ev.stats.cost_usd
                self._finish_turn()

    def _current_snippet(self) -> Message | None:
        # This turn's snippet — the trailing turn_snippet message (one per turn, appended at
        # enqueue). None defensively (e.g. a torn event stream with no enqueue).
        for msg in reversed(self.state.messages):
            if msg.role == "turn_snippet":
                return msg
        return None

    def _open_gate_card(self) -> Message | None:
        # The trailing unresolved pending_action. Only ever one open gate, so the last is it.
        for msg in reversed(self.state.messages):
            if msg.role == "pending_action" and not msg.resolved:
                return msg
        return None

    def _resolve_open_gate_card(self, outcome: str) -> None:
        # A cancel/error mid-gate must mark the open confirm card resolved, or it renders live
        # Yes/No buttons forever. outcome = the terminal label.
        card = self._open_gate_card()
        if card is not None:
            card.resolved = True
            card.text = f"{card.text}\n({outcome})"

    def _attach_recover(self, payload: dict) -> None:
        # Attach RecoverInfo to the trailing resolved pending_action with no recover yet.
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
        # MAIN THREAD (Yes/No click). Mark the open card resolved + unblock the worker.
        card = self._open_gate_card()
        if card is not None:
            card.resolved = True
            card.text = f"{card.text}\nYou chose: {'Yes' if approved else 'No'}"
        self.gate.answer(
            GateResponse(approved=approved, option="Yes" if approved else "No")
        )

    def answer_gate_credential(self, secret: str) -> None:
        # MAIN THREAD (Save on a CREDENTIAL gate). The secret rides GateResponse back to the
        # worker; the card text gets only a REDACTED echo, so the persisted card + the trace
        # never hold the full secret.
        card = self._open_gate_card()
        if card is not None:
            card.resolved = True
            card.gate_input = ""  # drop the typed buffer immediately
            card.text = f"{card.text}\nYou provided: {mask_secret(secret)}"
        self.gate.answer(GateResponse(approved=True, secret=secret))

    def _finish_turn(self) -> None:
        self.state.in_flight = False
        self.state.status = ""  # clear the transient in-flight status line

    def drain_bridge(self) -> None:
        # MAIN THREAD, per frame, early (pre-render) so a recompiled node renders this frame.
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
                # Stale sentinel from a prior release() on this REUSED session — app still
                # live, so keep serving or the next turn is stranded.
                logger.debug("copilot worker: ignoring stale shutdown sentinel")
                continue
            assert isinstance(item, str)
            self._cancel.clear()  # a fresh turn starts uncancelled
            self._drop_turn.clear()  # ...and committable (teardown re-sets it)
            self._run_one_turn(item)

    def _run_one_turn(self, user_text: str) -> None:
        context = build_context(self.caps)
        error_text = ""
        # The terminal event flips in_flight (which gates save_conversation). Buffer it and
        # emit it ONLY after the history commit below, so a same-frame save can't observe a
        # history short by the just-finished turn.
        terminal: AgentEvent | None = None
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
                scratchpad_render=lambda: render_working_set(
                    self.caps.read_working_set()
                ),
                batch_begin=self.caps.batch_begin,
            ):
                if isinstance(ev, AgentError):
                    error_text = ev.message
                if isinstance(ev, AgentTurnDone | AgentError | AgentCancelled):
                    terminal = ev
                    break  # the generator yields nothing after a terminal event
                self._events.put(ev)
        except CopilotConfigError as e:
            logger.warning(f"Copilot turn aborted: {e}")
            error_text = str(e)
            terminal = AgentError(
                str(e)
            )  # no run_turn messages -> empty tail (default)
        except Exception:
            logger.exception("Copilot turn failed")
            error_text = "copilot turn failed (see logs)"
            terminal = AgentError(error_text)
        # The engine-derived NL turn-summary rides the terminal event; empty default on the
        # except fallbacks. _commit_turn renders it into one assistant history message.
        summary = getattr(terminal, "summary", TurnSummary())
        self._commit_turn(user_text, summary, error_text)
        if terminal is not None:
            self._events.put(terminal)

    def _commit_turn(
        self, user_text: str, summary: TurnSummary, error_text: str
    ) -> None:
        # Commit to history (worker is sole owner) as NATURAL LANGUAGE ONLY: the user message +
        # ONE assistant message from the turn-summary. No tool messages — the source the agent
        # read is re-fetched live each turn, never persisted. `summary` is built engine-side in
        # run_turn and rides the terminal event (see agent.TurnSummary); _render_summary renders it.
        #
        # A turn aborted by teardown (_drop_turn set) must NOT commit — run_turn returns NORMALLY
        # on cancel, so it would otherwise land on the next project's reset history. A user Stop
        # does NOT set _drop_turn.
        if self._drop_turn.is_set():
            return
        self.history.append(LLMMessage(role="user", content=user_text))
        self.history.append(
            LLMMessage(
                role="assistant", content=self._render_summary(summary, error_text)
            )
        )

    def _render_summary(self, summary: TurnSummary, error_text: str) -> str | None:
        # The NL assistant message persisted for a turn: reply prose (sanitized ASCII) + a terse
        # action ledger + the nodes touched, so the next turn can resolve "it" / not re-publish.
        # None only when a turn produced nothing — keeps the user/assistant pairing.
        parts: list[str] = []
        reply = summary.reply or error_text
        if reply:
            parts.append(sanitize_display(reply))
        if summary.ledger:
            parts.append("(this turn: " + "; ".join(summary.ledger) + ")")
        if summary.nodes:
            parts.append("(nodes: " + ", ".join(summary.nodes) + ")")
        return "\n".join(parts) if parts else None

    # ---- lifecycle ----

    def reset_conversation(self) -> None:
        # Project switch / clear: drop the transcript + history. reusable: the same session
        # serves the next project, so the gate must NOT latch shut.
        # INVARIANT: callers gate this on `not in_flight`, so the worker is idle here and no
        # join is needed. Warn if that invariant is ever violated.
        if self.state.in_flight:
            logger.warning(
                "reset_conversation called mid-turn — caller bypassed the in_flight gate"
            )
        self._drop_turn.set()  # a worker finishing an aborted turn must not commit it
        self._cancel.set()
        self.gate.cancel_all(reusable=True)
        # Drain queued turns / a stale shutdown sentinel from a prior release(), so nothing
        # strands the next turn.
        while True:
            try:
                self._turn_queue.get_nowait()
            except queue.Empty:
                break
        self.state = ChatState()
        self.history = []
        self._cancel = threading.Event()
        # Rebuild the checkpoint store for the project we switch INTO (rehydrates ITS persisted
        # checkpoints); the outgoing project's snapshots stay on its own disk. A Clear deletes
        # them explicitly App-side BEFORE this call (clear_checkpoints). Lazily rebuilt next access.
        self._checkpoints = None
        # Fresh transcript for the project we switch INTO; the prior one is left closed on
        # disk (retention prunes it later).
        self.trace.close()
        self.trace = self._new_trace()

    def seal_checkpoint(self) -> None:
        # MAIN THREAD, at turn-done (the ui.py copilot_turn_active True->False transition):
        # finalize the active checkpoint + persist its index, then prune any checkpoint whose
        # user Message is gone (retention, decision 14).
        self.checkpoints.seal()
        live = {m.turn_id for m in self.state.messages if m.turn_id}
        self.checkpoints.prune_to(live)

    def clear_checkpoints(self) -> None:
        # Clear-the-chat: delete every snapshot outright (decision 4), no archive.
        self.checkpoints.clear()

    def note_revert(self, text: str) -> None:
        # MAIN THREAD, idle (Revert is gated on not-in-flight, decision 12): record a revert as a
        # plain NL message in BOTH the transcript and the worker's replay history, so the agent's
        # next turn sees that its earlier edits were undone (keeps history coherent with disk).
        self.state.messages.append(Message(role="tool_status", text=text))
        self.history.append(LLMMessage(role="assistant", content=text))

    def save_conversation(self, path: Path) -> None:
        # MAIN THREAD, at a quiescent point (not in_flight): snapshot state + history to disk.
        # Both arrays are consistent here — the worker is idle.
        self.pump_events()  # drain any terminal event so the last turn's bubble + usage land
        store = ConversationStore.from_runtime(self.state, self.history)
        store.save(path)

    def load_conversation(self, store: ConversationStore) -> None:
        # MAIN THREAD, after reset_conversation (fresh session): restore the parsed store into
        # state + history + usage.
        self.state.messages = store.to_messages()
        self.state.session_cost_usd = store.to_session_cost()
        self.state.last_turn = store.to_last_turn()
        self.history = store.to_history()
        if self.state.messages or self.history:
            self.trace.event(
                "conversation_loaded",
                messages=len(self.state.messages),
                history=len(self.history),
            )

    def release(self) -> None:
        # MAIN THREAD, at shutdown — called BEFORE the node release, so a queued GL op never
        # runs against half-released nodes. Safe when no worker was spawned: sentinel +
        # cancel_all + join on a None thread are all no-ops.
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
