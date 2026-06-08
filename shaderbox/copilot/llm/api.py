from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

# Seam B: the provider-neutral LLM boundary. No `openai` import here (that lives only
# in openrouter.py) — the SDK types never leak into this seam, so the impl is swappable
# and a fake client is injectable for headless tests. OpenAI/OpenRouter-shaped (NOT
# Anthropic): `parameters` not `input_schema`, tool results as role="tool" messages,
# `arguments` a raw JSON string the caller parses. See report 09.


@dataclass(frozen=True)
class LLMToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema (OpenRouter key is `parameters`)


@dataclass(frozen=True)
class LLMToolCall:
    id: str  # echo back on the matching role="tool" result message
    name: str
    arguments: (
        str  # raw JSON string — caller parses (may be malformed / double-escaped)
    )


@dataclass(frozen=True)
class LLMMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None  # None only on an assistant tool-only turn
    tool_call_id: str | None = None  # required when role == "tool"
    tool_calls: list[LLMToolCall] | None = None  # on assistant turns that called tools


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0  # OpenRouter returns the charged cost on usage.cost

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


# ---- stream events (the union the agent loop consumes) ----


@dataclass(frozen=True)
class LLMTextDelta:
    text: str


@dataclass(frozen=True)
class LLMToolCallStarted:
    index: int
    id: str
    name: str


@dataclass(frozen=True)
class LLMToolCallCompleted:
    index: int
    id: str
    name: str
    arguments: str  # raw JSON


@dataclass(frozen=True)
class LLMDone:
    finish_reason: str  # "stop" | "tool_calls" | "length" | "content_filter"
    usage: LLMUsage = field(default_factory=LLMUsage)


LLMStreamEvent = LLMTextDelta | LLMToolCallStarted | LLMToolCallCompleted | LLMDone


class LLMClient(Protocol):
    """Synchronous (runs on the copilot worker thread — no asyncio). The client does
    ONE call and yields events; it never loops, executes tools, or assembles prompts —
    that is the agent loop's job (agent.py)."""

    def stream(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[LLMToolSpec] | None = None,
        max_tokens: int,
    ) -> Iterator[LLMStreamEvent]: ...
