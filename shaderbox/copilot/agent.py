import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass

from shaderbox.copilot.config import CopilotConfig
from shaderbox.copilot.context import CopilotContext
from shaderbox.copilot.gate import GateChannel
from shaderbox.copilot.llm.api import (
    LLMDone,
    LLMMessage,
    LLMTextDelta,
    LLMToolCall,
    LLMToolCallCompleted,
    LLMToolCallStarted,
    LLMUsage,
)
from shaderbox.copilot.prompt import build_messages
from shaderbox.copilot.tools.registry import ToolRegistry

# The agent loop: own a growing conversation, stream one assistant turn, execute any
# tool calls, append the results, re-stream until the model stops calling tools or a
# limit trips (spec §2; faithful to cc-server AgentLoop.run). The GateChannel param is
# present but never triggered in slice 1 (all three tools are non-destructive).


@dataclass(frozen=True)
class AgentTextDelta:
    text: str


@dataclass(frozen=True)
class AgentStatus:
    text: str


@dataclass(frozen=True)
class AgentToolCard:
    name: str
    ok: bool
    payload: dict | None


@dataclass(frozen=True)
class AgentTurnDone:
    note: str
    usage: LLMUsage


@dataclass(frozen=True)
class AgentError:
    message: str


@dataclass(frozen=True)
class AgentCancelled:
    pass


AgentEvent = (
    AgentTextDelta
    | AgentStatus
    | AgentToolCard
    | AgentTurnDone
    | AgentError
    | AgentCancelled
)


@dataclass
class _ToolCallBuilder:
    id: str
    name: str
    arguments: str


class _UsageRollup:
    # LLMUsage is frozen (no in-place add); accumulate here, materialize on demand.
    def __init__(self) -> None:
        self._in = 0
        self._out = 0
        self._cost = 0.0

    def add(self, u: LLMUsage) -> None:
        self._in += u.input_tokens
        self._out += u.output_tokens
        self._cost += u.cost_usd

    def value(self) -> LLMUsage:
        return LLMUsage(
            input_tokens=self._in, output_tokens=self._out, cost_usd=self._cost
        )

    @property
    def input_tokens(self) -> int:
        return self._in


class _RunLog:
    # The loop-local action ledger (§2.3). Loop-private — never on state (§T2).
    def __init__(self) -> None:
        self._entries: list[tuple[str, bool, str]] = []

    def record(self, name: str, ok: bool, msg: str) -> None:
        self._entries.append((name, ok, msg))

    def executed_actions_note(self, registry: ToolRegistry) -> str:
        # The cutoff "what mutating work already committed" note (§I4): filter to
        # mutating + ok.
        mutating = [
            name for name, ok, _ in self._entries if ok and registry.is_mutating(name)
        ]
        if not mutating:
            return ""
        uniq = list(dict.fromkeys(mutating))
        return "Already done this turn: " + ", ".join(uniq)


def _parse_args(raw: str) -> dict | None:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return _unescape_double_escaped(parsed)


def _unescape_double_escaped(args: dict) -> dict:
    # grok footgun (§J6): a string value can be double-escaped JSON ({"x": "\"y\""}).
    # Unwrap one level when a value is a quoted JSON string.
    out: dict = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) >= 2 and v[0] == '"' and v[-1] == '"':
            try:
                out[k] = json.loads(v)
            except json.JSONDecodeError:
                out[k] = v
        else:
            out[k] = v
    return out


def _assistant_message(text: str, calls: list[_ToolCallBuilder]) -> LLMMessage:
    # content="" -> None when there are tool calls (grok, §J5).
    return LLMMessage(
        role="assistant",
        content=text or None,
        tool_calls=[
            LLMToolCall(id=c.id, name=c.name, arguments=c.arguments) for c in calls
        ],
    )


def _tool_message(tool_call_id: str, content: str) -> LLMMessage:
    return LLMMessage(role="tool", tool_call_id=tool_call_id, content=content)


def run_turn(
    client: object,
    registry: ToolRegistry,
    config: CopilotConfig,
    context: CopilotContext,
    history: list[LLMMessage],
    user_text: str,
    gate: GateChannel,
    cancel: threading.Event,
) -> Iterator[AgentEvent]:
    # `client` is an llm.api.LLMClient — kept as `object` here so this module imports no
    # provider impl. The duck-typed `.stream(...)` is the only call.
    _ = gate  # present for the seam; slice-1 tools never trigger requires_gate (§16.1)
    messages = build_messages(context, history, user_text)
    specs = registry.eager_specs()
    usage = _UsageRollup()
    ran = _RunLog()

    for _iteration in range(config.max_iterations):
        if cancel.is_set():
            yield AgentCancelled()
            return

        text_buf = ""
        builders: dict[int, _ToolCallBuilder] = {}
        done: LLMDone | None = None
        for ev in client.stream(  # type: ignore[attr-defined]
            messages, tools=specs, max_tokens=config.max_tokens_per_turn
        ):
            match ev:
                case LLMTextDelta():
                    text_buf += ev.text
                    yield AgentTextDelta(ev.text)
                case LLMToolCallStarted():
                    yield AgentStatus(registry.status_for(ev.name, None))
                case LLMToolCallCompleted():
                    builders[ev.index] = _ToolCallBuilder(
                        id=ev.id, name=ev.name, arguments=ev.arguments
                    )
                case LLMDone():
                    usage.add(ev.usage)
                    done = ev

        if done is None or done.finish_reason != "tool_calls" or not builders:
            yield AgentTurnDone(ran.executed_actions_note(registry), usage.value())
            return

        calls = [builders[i] for i in sorted(builders)]
        messages.append(_assistant_message(text_buf, calls))
        for tc in calls:
            args = _parse_args(tc.arguments)
            if args is None:
                messages.append(_tool_message(tc.id, "error: invalid arguments JSON"))
                continue
            yield AgentStatus(registry.status_for(tc.name, args))
            if cancel.is_set():
                yield AgentCancelled()
                return
            ok, msg, payload = registry.execute(tc.name, args)
            ran.record(tc.name, ok, msg)
            yield AgentToolCard(tc.name, ok, payload)
            messages.append(_tool_message(tc.id, msg))

    yield AgentTurnDone(ran.executed_actions_note(registry), usage.value())
