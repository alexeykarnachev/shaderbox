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


@dataclass
class Message:
    role: MessageRole
    text: str = ""
    # For role == "pending_action": the agent loop is blocked on the request queue
    # until the user answers (the action-required gate, skeleton 10 §7). resolved flips
    # True once the UI enqueues the answer.
    resolved: bool = False


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
