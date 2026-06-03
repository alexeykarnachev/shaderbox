import json
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from loguru import logger

from shaderbox.copilot.config import CopilotConfig
from shaderbox.copilot.context import CopilotContext
from shaderbox.copilot.gate import GateChannel, GateKind, GateRequest
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

# The agent loop: own a growing conversation, stream one assistant turn, execute any
# tool calls, append the results, re-stream until the model stops calling tools or a
# limit trips (spec §2; faithful to cc-server AgentLoop.run). A tool whose gate_policy
# trips requires_gate blocks on the GateChannel for a user Yes/No before it runs (§7 /
# feature 020·17).

_MODEL_INCOMPATIBLE_MSG = (
    "The selected model isn't compatible with tool calling — after using a tool it "
    "produced neither a native tool call nor a text reply. Pick a different model in "
    "Settings -> Integrations -> Copilot."
)


def _trunc(text: str, limit: int) -> str:
    # Log-line truncation with an explicit ASCII marker so a cut value never reads as
    # the whole thing. The full text always lives in the trace (trace.py); these caps
    # are for the terse console/file log only.
    return (
        text if len(text) <= limit else f"{text[:limit]}[...{len(text) - limit} more]"
    )


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


@dataclass(frozen=True)
class AgentGateOpened:
    # A gated tool is about to run; the worker is blocking on the user's Yes/No. pump_events
    # materializes a pending_action Message from this so the UI can draw the confirm (§7.1).
    request: GateRequest


AgentEvent = (
    AgentTextDelta
    | AgentStatus
    | AgentToolCard
    | AgentTurnDone
    | AgentError
    | AgentCancelled
    | AgentGateOpened
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


_GATE_PROMPTS: dict[str, Callable[[dict], str]] = {
    "delete_node": lambda a: f"Delete node '{a.get('node', '')}'? It moves to the project trash (recoverable).",
}


def build_gate(name: str, args: dict) -> GateRequest:
    # Engine-built confirm prompt (§7.2 / feature 020·17): the engine owns the destructive-
    # action phrasing so it's accurate, not the model. Falls back to a generic line for any
    # future ALWAYS-gated tool without a template.
    template = _GATE_PROMPTS.get(name)
    prompt = (
        template(args)
        if template is not None
        else f"Run {name}? This action will change your project."
    )
    return GateRequest(kind=GateKind.CONFIRM, prompt=prompt, options=["Yes", "No"])


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
    tr = trace if trace is not None else NULL_TRACE
    messages = build_messages(context, history, user_text)
    specs = registry.eager_specs()
    usage = _UsageRollup()
    ran = _RunLog()
    total_tool_calls = 0
    consecutive_failed_edits = 0  # §I2 self-correction cap (reset on any other outcome)
    logger.info(f"copilot turn start | user={_trunc(user_text, 80)!r}")
    logger.debug(
        f"copilot turn detail | history_msgs={len(history)} "
        f"eager_tools={[s.name for s in specs]}"
    )
    tr.event("turn_start", user_text=user_text, history=history, eager_tools=specs)

    for iteration in range(config.max_iterations):
        if cancel.is_set():
            logger.debug(f"copilot turn cancelled at iteration {iteration}")
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
        logger.debug(
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
            # A turn that ends after a tool ran with NEITHER a native tool call NOR text
            # is a model that can't continue a tool-call conversation. A compatible model
            # always does one or the other; the absence of both is the proof — no need to
            # sniff what garbage it emitted. Reject, don't work around (maintainer rule).
            if not text_buf and total_tool_calls > 0:
                logger.warning(
                    f"copilot: empty reply after {total_tool_calls} tool call(s) — "
                    "model is not tool-call compatible"
                )
                tr.event(
                    "model_incompatible", iteration=iteration, reason="empty_after_tool"
                )
                yield AgentError(_MODEL_INCOMPATIBLE_MSG)
                return
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
                    f"copilot tool {tc.name} | bad args JSON: {_trunc(tc.arguments, 120)!r}"
                )
                tr.event(
                    "tool_args_parse_error",
                    name=tc.name,
                    arguments_raw=tc.arguments,
                )
                messages.append(_tool_message(tc.id, "error: invalid arguments JSON"))
                continue
            yield AgentStatus(registry.status_for(tc.name, args))
            if cancel.is_set():
                logger.debug(f"copilot turn cancelled before tool {tc.name}")
                yield AgentCancelled()
                return
            # Gate a destructive/publish tool on a user Yes/No before it runs (§7 / 020·17).
            # On decline: record + append the tool result (a declined call STILL needs a
            # matching tool message — an orphaned tool_call_id 400s the next stream) + continue
            # to the next call in the batch. The continue lands BEFORE the execute + the
            # consecutive_failed_edits logic, so a decline never reaches either: a user choice
            # is not a convergence failure and must not count toward the edit-retry cap.
            if registry.requires_gate(tc.name, args, config):
                req = build_gate(tc.name, args)
                tr.event("gate_open", name=tc.name, prompt=req.prompt)
                yield AgentGateOpened(req)
                resp = gate.ask(req)
                if resp.cancelled:
                    logger.debug(f"copilot turn cancelled at gate for {tc.name}")
                    tr.event("gate_cancelled", name=tc.name)
                    yield AgentCancelled()
                    return
                if not resp.approved:
                    logger.info(f"copilot tool {tc.name} | user declined")
                    tr.event("gate_declined", name=tc.name)
                    ran.record(tc.name, False, "error: user declined")
                    messages.append(_tool_message(tc.id, "error: user declined"))
                    continue
            ok, msg, payload = registry.execute(tc.name, args)
            total_tool_calls += 1
            logger.info(f"copilot tool #{total_tool_calls} {tc.name} -> ok={ok}")
            logger.debug(f"copilot tool #{total_tool_calls} args={args} result={msg!r}")
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

            # §I2 self-correction cap: a model stuck on an edit (an old_str that keeps not
            # matching, a line range that keeps not resolving) would otherwise retry to the
            # max_iterations ceiling. Count CONSECUTIVE failed MUTATING tools; any success or
            # non-mutating tool resets it. A freshness reject (payload["stale"], §15) is a
            # benign "re-read and continue", NOT a convergence failure, so it does not count.
            # payload is None on a malformed-args/unknown-tool result — guard the .get.
            stale = bool((payload or {}).get("stale"))
            if registry.is_mutating(tc.name) and not ok and not stale:
                consecutive_failed_edits += 1
            else:
                consecutive_failed_edits = 0
            if consecutive_failed_edits >= config.max_edit_retries:
                logger.warning(
                    f"copilot edit giveup after {consecutive_failed_edits} failed "
                    f"edits | total_in={usage.input_tokens} "
                    f"cost=${usage.value().cost_usd:.6f}"
                )
                tr.event(
                    "edit_giveup",
                    consecutive_failed_edits=consecutive_failed_edits,
                    usage=usage.value(),
                )
                note = (
                    f"I couldn't apply that edit after {consecutive_failed_edits} tries "
                    "— the edit kept not applying to the shader source. I've stopped to "
                    "avoid looping. Tell me to try again, or describe the change "
                    "differently."
                )
                yield AgentError(note)
                return

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
    cutoff_note = (
        f"I stopped after {config.max_iterations} steps without finishing this turn. "
        "Ask me to continue, or rephrase what you need."
    )
    yield AgentError(cutoff_note)
