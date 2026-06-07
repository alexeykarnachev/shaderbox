from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Literal

from shaderbox.copilot.gate import GateKind
from shaderbox.copilot.llm.api import LLMUsage

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


@dataclass
class SessionUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, u: LLMUsage) -> None:
        self.input_tokens += u.input_tokens
        self.output_tokens += u.output_tokens
        self.cost_usd += u.cost_usd


@dataclass
class ChatState:
    messages: list[Message] = field(default_factory=list)
    streaming_text: str = ""  # the in-progress assistant message, grows per token
    in_flight: bool = False  # a turn is running (gates Send, shows Stop)
    # Transient in-flight status phrase from AgentStatus; shown in place of "thinking", cleared on
    # turn end. Not persisted and not a durable Message.
    status: str = ""
    usage: SessionUsage = field(default_factory=SessionUsage)
    # Last completed turn's usage (input = its first-iteration context size); drives the header
    # usage bars. Transient (not persisted), reset with the conversation.
    last_turn_usage: LLMUsage | None = None
