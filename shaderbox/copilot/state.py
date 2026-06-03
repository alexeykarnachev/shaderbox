from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Literal

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


@dataclass
class Message:
    role: MessageRole
    text: str = ""
    # For role == "pending_action": the agent loop is blocked on the gate until the user
    # answers (feature 020·17). resolved flips True once the UI answers (or a cancel resolves it).
    resolved: bool = False
    # Set on a resolved-Yes delete card: the undo affordance (feature 020·17).
    recover: RecoverInfo | None = None


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
    usage: SessionUsage = field(default_factory=SessionUsage)
