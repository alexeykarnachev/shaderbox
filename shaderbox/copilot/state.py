from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Literal

from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.gate import GateKind

# Chat render-state. Written ONLY by session.pump_events on the main thread, read ONLY by
# the UI on the main thread -> single-writer, no lock. The worker bridges its writes via events.

MessageRole = Literal["user", "assistant", "tool_status", "error", "pending_action"]


class CopilotLayout(StrEnum):
    CORNER = auto()  # bottom-right fixed box
    BOTTOM_STRIP = auto()  # full-width strip along the bottom
    FREE = auto()  # user-moved/resized; position persisted via imgui.ini

    def next(self) -> "CopilotLayout":
        order = list(CopilotLayout)
        return order[(order.index(self) + 1) % len(order)]

    @property
    def variant(self) -> int:
        # The drawn-glyph index for layout_icon_button (which stays feature-agnostic).
        return list(CopilotLayout).index(self)


@dataclass(frozen=True)
class RecoverInfo:
    # Backs a resolved-Yes delete card's Recover button. trash_name is the dir-NAME under the
    # project trash (NOT an absolute path), re-anchored via App.trash_dir at click time.
    node_id: str
    node_name: str
    trash_name: str
    done: bool = False


ResultWidgetKind = Literal["open_url", "open_path"]


@dataclass(frozen=True)
class ResultWidget:
    # A button a tool surfaces in its result; the raw target never enters LLM context. open_url ->
    # browser-open, open_path -> file-manager. Empty target = degraded result, no button drawn.
    # NON-BLOCKING — unlike a gate, the agent does not wait for a click.
    kind: ResultWidgetKind
    label: str
    target: str


@dataclass
class Message:
    role: MessageRole
    text: str = ""
    # role == "pending_action": agent loop is blocked on the gate; resolved flips True on answer/cancel.
    resolved: bool = False
    recover: RecoverInfo | None = None  # undo affordance on a resolved-Yes delete card
    # role == "user": the turn id this message initiated, keying its rollback checkpoint. "" = no
    # checkpoint (a read-only turn, or a message that predates the feature). Backs the Revert glyph.
    turn_id: str = ""
    # Gate widget on a pending_action card: CONFIRM = Yes/No, CREDENTIAL = masked input. gate_input
    # is the UI-only typed-secret buffer — NEVER persisted/read by session (leaves via GateResponse.secret).
    gate_kind: GateKind = GateKind.CONFIRM
    gate_input: str = ""
    # CONFIG gate: integration whose draw_config_ui the card renders. UI-only + transient like
    # gate_input — never persists (its turn stays in_flight, so save_conversation can't run).
    gate_integration: str = ""
    # role == "tool_status": result widget (a button) from a tool's outcome. None = plain status line.
    # Non-blocking, unlike a gate.
    result_widget: ResultWidget | None = None


@dataclass(frozen=True)
class TurnStats:
    # Last completed turn, for the header context gauge + its tooltip.
    # context_tokens = the FIRST iteration's input size = system + project + accumulated history +
    #   working set, BEFORE within-turn tool churn (which is discarded at turn end). This is the
    #   standing-context "how full / when to compact" signal, NOT the per-turn peak.
    # reply_tokens = output tokens summed across the turn's iterations (how much the model wrote).
    # cost_usd = this turn's total charged cost.
    context_tokens: int = 0
    reply_tokens: int = 0
    cost_usd: float = 0.0


def context_gauge_readout(
    last_turn: "TurnStats | None", session_cost_usd: float
) -> tuple[float, str]:
    """The header context gauge: (fill_fraction, tooltip). fill = standing context vs the input
    budget — the 'how full / when to compact' signal. The tooltip carries the secondary numbers
    (last reply, last turn cost, session cost) that are info, not fullness."""
    budget = COPILOT_CONFIG.max_input_tokens
    if last_turn is None:
        return 0.0, "No turn yet."
    fill = min(1.0, last_turn.context_tokens / budget) if budget else 0.0
    pct = round(fill * 100)
    return fill, (
        f"Context: {last_turn.context_tokens} / {budget} tok ({pct}%)\n"
        f"Last reply: {last_turn.reply_tokens} tok\n"
        f"Last turn cost: ${last_turn.cost_usd:.4f}\n"
        f"Session cost: ${session_cost_usd:.4f}"
    )


@dataclass
class ChatState:
    messages: list[Message] = field(default_factory=list)
    streaming_text: str = ""  # the in-progress assistant message, grows per token
    in_flight: bool = False  # a turn is running (gates Send, shows Stop)
    # Transient in-flight status phrase from AgentStatus; shown in place of "thinking", cleared on
    # turn end. Not persisted and not a durable Message.
    status: str = ""
    # Running cost across the whole conversation (persisted; shown in the gauge tooltip).
    session_cost_usd: float = 0.0
    # Last completed turn's stats; drives the header context gauge. Persisted (ConversationStore v7),
    # restored on load, reset by Clear.
    last_turn: TurnStats | None = None
