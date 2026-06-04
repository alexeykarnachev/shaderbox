from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Literal

from shaderbox.copilot.gate import GateKind
from shaderbox.copilot.llm.api import LLMUsage

# Seam E: the chat render-state. THREAD-SAFETY: written ONLY by session.pump_events on
# the main thread (draining CopilotEvents), read ONLY by the UI on the main thread ->
# single-writer, no lock. The worker NEVER writes this directly; its own working usage
# rollup is bridged in via events, not a shared field.

MessageRole = Literal["user", "assistant", "tool_status", "error", "pending_action"]


class CopilotLayout(StrEnum):
    # The chat floating-window placement presets, cycled by a top-bar button.
    CORNER = auto()  # bottom-right fixed box
    BOTTOM_STRIP = auto()  # full-width strip along the bottom
    FREE = auto()  # user-moved/resized; position persisted via imgui.ini


@dataclass(frozen=True)
class RecoverInfo:
    # Carried by a resolved-Yes delete card so it can offer a Recover button (feature 020·17).
    # trash_name is the dir-NAME under the project trash (NOT an absolute path — the project dir
    # is relocatable), re-anchored via App.trash_dir at click time. done = the one-shot consumed.
    node_id: str
    node_name: str
    trash_name: str
    done: bool = False


ResultWidgetKind = Literal["open_url", "open_path"]


@dataclass(frozen=True)
class ResultWidget:
    # A first-class chat affordance a tool surfaces in its RESULT (feature 020·21): the engine renders
    # it (a button), the agent only learns the FACT a widget was shown — the raw target never enters
    # LLM context. open_url -> a browser-open button (publish links); open_path -> a file-manager button
    # (render output, an absolute path). target is empty only on a degraded result (then no button is
    # drawn). NON-BLOCKING — unlike a gate, the agent does not wait for a click.
    kind: ResultWidgetKind
    label: str
    target: str


@dataclass
class Message:
    role: MessageRole
    text: str = ""
    # For role == "pending_action": the agent loop is blocked on the gate until the user
    # answers (feature 020·17). resolved flips True once the UI answers (or a cancel resolves it).
    resolved: bool = False
    # Set on a resolved-Yes delete card: the undo affordance (feature 020·17).
    recover: RecoverInfo | None = None
    # The gate widget to draw on a pending_action card (feature 020·19): CONFIRM = Yes/No,
    # CREDENTIAL = a masked secret input. gate_input is the UI-only typed-secret buffer — NEVER
    # persisted, NEVER read by session/trace (the secret leaves only via GateResponse.secret).
    gate_kind: GateKind = GateKind.CONFIRM
    gate_input: str = ""
    # For a CONFIG gate (feature 020·21): the integration whose draw_config_ui the card renders
    # ("youtube"). UI-only + transient like gate_input — a CONFIG card never persists (its turn stays
    # in_flight, so save_conversation can't run while it's open).
    gate_integration: str = ""
    # For role == "tool_status": a first-class result widget (a button) the engine renders from a
    # tool's structured outcome (feature 020·21). None = a plain status line. Parallel to recover, NOT
    # a reuse of gate_kind (a result widget is non-blocking; a gate blocks the worker).
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
    # The latest transient in-flight status phrase (feature 020·20 D1): set from AgentStatus,
    # shown in place of the bare "thinking" caption while a turn runs, cleared on turn end. NOT
    # persisted (transient) and NOT a durable Message — the tool cards carry the durable line.
    status: str = ""
    usage: SessionUsage = field(default_factory=SessionUsage)
