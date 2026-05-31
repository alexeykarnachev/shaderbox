import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass

from loguru import logger

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
from shaderbox.copilot.trace import NULL_TRACE, TraceLog

_NULL_TRACE = NULL_TRACE

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
    trace: TraceLog | None = None,
) -> Iterator[AgentEvent]:
    # `client` is an llm.api.LLMClient — kept as `object` here so this module imports no
    # provider impl. The duck-typed `.stream(...)` is the only call. `trace` is the
    # full-transcript sink (None in tests) — it records everything, in full (trace.py).
    _ = gate  # present for the seam; slice-1 tools never trigger requires_gate (§16.1)
    tr = trace if trace is not None else _NULL_TRACE
    messages = build_messages(context, history, user_text)
    specs = registry.eager_specs()
    usage = _UsageRollup()
    ran = _RunLog()
    total_tool_calls = 0
    logger.info(
        f"copilot turn start | user={user_text[:80]!r} "
        f"history_msgs={len(history)} eager_tools={[s.name for s in specs]}"
    )
    tr.event("turn_start", user_text=user_text, history=history, eager_tools=specs)

    for iteration in range(config.max_iterations):
        if cancel.is_set():
            logger.info(f"copilot turn cancelled at iteration {iteration}")
            yield AgentCancelled()
            return

        text_buf = ""
        builders: dict[int, _ToolCallBuilder] = {}
        done: LLMDone | None = None
        tr.event(
            "llm_request",
            iteration=iteration,
            messages=messages,
            tools=specs,
            max_tokens=config.max_tokens_per_turn,
        )
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

        fr = done.finish_reason if done else "no-done-event"
        u = done.usage if done else None
        tokens = (
            f"in={u.input_tokens} out={u.output_tokens} cost=${u.cost_usd:.6f}"
            if u
            else "in=? out=? cost=?"
        )
        logger.info(
            f"copilot iter {iteration} | finish={fr} {tokens} "
            f"text={len(text_buf)}ch tool_calls={[b.name for b in builders.values()]}"
        )
        tr.event(
            "llm_response",
            iteration=iteration,
            finish_reason=fr,
            text=text_buf,
            tool_calls=[
                {"id": b.id, "name": b.name, "arguments": b.arguments}
                for b in builders.values()
            ],
            usage=u,
        )

        if done is None or done.finish_reason != "tool_calls" or not builders:
            if done is None:
                logger.warning("copilot stream ended with no LLMDone event")
            logger.info(
                f"copilot turn done | iterations={iteration + 1} "
                f"tool_calls={total_tool_calls} reply={len(text_buf)}ch "
                f"total_in={usage.input_tokens} cost=${usage.value().cost_usd:.6f}"
            )
            tr.event(
                "turn_done",
                iterations=iteration + 1,
                tool_calls=total_tool_calls,
                reply=text_buf,
                usage=usage.value(),
            )
            yield AgentTurnDone(ran.executed_actions_note(registry), usage.value())
            return

        calls = [builders[i] for i in sorted(builders)]
        messages.append(_assistant_message(text_buf, calls))
        for tc in calls:
            args = _parse_args(tc.arguments)
            if args is None:
                logger.warning(
                    f"copilot tool {tc.name} | bad args JSON: {tc.arguments[:120]!r}"
                )
                messages.append(_tool_message(tc.id, "error: invalid arguments JSON"))
                continue
            yield AgentStatus(registry.status_for(tc.name, args))
            if cancel.is_set():
                logger.info(f"copilot turn cancelled before tool {tc.name}")
                yield AgentCancelled()
                return
            ok, msg, payload = registry.execute(tc.name, args)
            total_tool_calls += 1
            logger.info(
                f"copilot tool #{total_tool_calls} {tc.name}({args}) "
                f"-> ok={ok} | {msg[:120]!r}"
            )
            tr.event(
                "tool_call",
                n=total_tool_calls,
                name=tc.name,
                args=args,
                ok=ok,
                result=msg,
                payload=payload,
            )
            ran.record(tc.name, ok, msg)
            yield AgentToolCard(tc.name, ok, payload)
            messages.append(_tool_message(tc.id, msg))

    logger.warning(
        f"copilot turn hit max_iterations={config.max_iterations} | "
        f"tool_calls={total_tool_calls} total_in={usage.input_tokens} "
        f"cost=${usage.value().cost_usd:.6f}"
    )
    tr.event(
        "turn_done",
        cutoff="max_iterations",
        iterations=config.max_iterations,
        tool_calls=total_tool_calls,
        usage=usage.value(),
    )
    yield AgentTurnDone(ran.executed_actions_note(registry), usage.value())
