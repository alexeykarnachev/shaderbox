import json
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

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
from shaderbox.copilot.state import ResultWidget
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
    # The tool's result string (render path / publish URL / error text) so the transcript can show
    # a result line under the card (feature 020·20 D1), not just "name: ok/failed". Engine-built,
    # already concise; goes to the LLM history + trace too.
    result: str = ""
    # A first-class result widget the engine renders from payload["widget"] (feature 020·21): a button
    # the user clicks; the raw target never reaches the model. None = no widget (the default).
    widget: ResultWidget | None = None
    # A terse chat-display line from payload["display"] (feature 020·23): when a tool's full `result`
    # is heavy (read_shader's full source listing), the USER sees this summary instead — the full
    # result still goes to the AGENT's context. "" = show `result` as before.
    display: str = ""


# The terminal events carry the per-turn TAIL (feature 020·23 D4): the assistant/tool messages this
# turn produced, orphan-cleaned, for _commit_turn to persist into the replay history. Empty default so
# the session's bare-except AgentError fallbacks (which never see run_turn's messages) commit no tail.


@dataclass(frozen=True)
class AgentTurnDone:
    note: str
    usage: LLMUsage
    messages: list[LLMMessage] = field(default_factory=list)


@dataclass(frozen=True)
class AgentError:
    message: str
    messages: list[LLMMessage] = field(default_factory=list)


@dataclass(frozen=True)
class AgentCancelled:
    messages: list[LLMMessage] = field(default_factory=list)


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


_ESCAPE_MARKERS: tuple[str, ...] = ("\\n", "\\t", "\\r", '\\"')


def _looks_double_escaped(v: str) -> bool:
    # The double-escape signature (cc-server _maybe_unescape, §J6): the value is a quoted JSON
    # string whose BODY carries escape markers (\n \t \r \") but no real whitespace — meaning the
    # provider serialized a newline as the two chars `\` `n`. A plainly-quoted payload that the
    # model legitimately wrapped in literal double-quotes (e.g. an `#include "x"` token) has no
    # such marker, so it is left untouched: unwrapping it would silently strip a real quote level.
    if len(v) < 2 or v[0] != '"' or v[-1] != '"':
        return False
    body = v[1:-1]
    return any(m in body for m in _ESCAPE_MARKERS) and not any(
        c in body for c in " \t\n\r"
    )


def _unescape_double_escaped(args: dict) -> dict:
    # grok footgun (§J6): a string value can be double-escaped JSON ({"x": "\"y\""}). Unwrap one
    # level ONLY when the value carries the double-escape signature (_looks_double_escaped).
    out: dict = {}
    for k, v in args.items():
        if isinstance(v, str) and _looks_double_escaped(v):
            try:
                out[k] = json.loads(v)
            except json.JSONDecodeError:
                out[k] = v
        else:
            out[k] = v
    return out


_RESULT_WIDGET_KINDS: frozenset[str] = frozenset({"open_url", "open_path"})


def _widget_from_payload(payload: dict | None) -> ResultWidget | None:
    # A tool surfaces a first-class result widget by putting a {"kind","label","target"} dict under
    # payload["widget"] (feature 020·21). Built engine-side, so it's well-formed — but guard defensively
    # (a known kind + a non-empty target) so a malformed entry yields no widget rather than a bad button.
    spec = (payload or {}).get("widget")
    if not isinstance(spec, dict):
        return None
    kind = spec.get("kind")
    target = spec.get("target", "")
    if kind not in _RESULT_WIDGET_KINDS or not target:
        return None
    return ResultWidget(
        kind=kind, label=str(spec.get("label", "Open")), target=str(target)
    )


def _turn_tail(messages: list[LLMMessage], head_len: int) -> list[LLMMessage]:
    # The per-turn tail to persist (feature 020·23 D4): messages[head_len:], with a trailing assistant
    # whose tool_calls lack matching tool results DROPPED — a cancel can return mid-batch (the assistant-
    # with-tool_calls is appended before the per-call results), and an orphaned tool_call_id 400s the
    # next stream. Drop that trailing assistant + any partial tool messages that follow it.
    tail = messages[head_len:]
    last_assistant = next(
        (i for i in range(len(tail) - 1, -1, -1) if tail[i].tool_calls), None
    )
    if last_assistant is None:
        return tail
    wanted = {c.id for c in (tail[last_assistant].tool_calls or [])}
    have = {m.tool_call_id for m in tail[last_assistant + 1 :] if m.tool_call_id}
    if wanted <= have:
        return tail
    return tail[
        :last_assistant
    ]  # drop the orphaned assistant + its partial tool messages


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
    "render_image": lambda a: "Render an image of this shader? The app pauses while it encodes.",
    "render_video": lambda a: f"Render a {a.get('seconds', '?')}s video of this shader? The app pauses while it encodes.",
    "publish_telegram": lambda a: "Publish this shader to your Telegram sticker pack? This uploads the sticker (external + live).",
    "publish_youtube": lambda a: f"Publish this shader to YouTube as '{a.get('title', '')}'? The video goes live on your channel (private; external).",
    "set_telegram_token": lambda a: "Paste your Telegram bot token below (from @BotFather). It's stored locally; I never see it.",
    "set_youtube_credentials": lambda a: "Set up YouTube below: paste your client_secret JSON, then press Connect (a browser sign-in opens). Or Cancel.",
    "create_telegram_pack": lambda a: f"Create a new Telegram sticker pack '{a.get('title', '')}'?",
    "select_telegram_pack": lambda a: f"Switch your active Telegram pack to '{a.get('set_name', '')}'?",
    "delete_telegram_pack": lambda a: f"Delete the Telegram sticker pack '{a.get('set_name', '')}'? This removes it from Telegram (external + irreversible).",
}


def build_gate(registry: ToolRegistry, name: str, args: dict) -> GateRequest:
    # Engine-built gate request (§7.2 / feature 020·17, 020·19): the engine owns the prompt
    # phrasing so it's accurate, not the model. A CREDENTIAL tool (gate_kind) gets a secret-input
    # gate; everything else the CONFIRM Yes/No. Falls back to a generic line for any ALWAYS-gated
    # tool without a template.
    template = _GATE_PROMPTS.get(name)
    prompt = (
        template(args)
        if template is not None
        else f"Run {name}? This action will change your project."
    )
    tool = registry.definition_for(name)
    if tool is not None and tool.gate_kind is GateKind.CREDENTIAL:
        return GateRequest(
            kind=GateKind.CREDENTIAL, prompt=prompt, secret_field=tool.secret_field
        )
    if tool is not None and tool.gate_kind is GateKind.CONFIG:
        # secret_field names the INTEGRATION whose draw_config_ui the card renders (feature 020·21).
        return GateRequest(
            kind=GateKind.CONFIG, prompt=prompt, secret_field=tool.secret_field
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
    # The head = [system, system, *history, user] (build_messages); the per-turn TAIL the model produced
    # is messages[head_len:] (feature 020·23 D4). _commit_turn persists that tail (the user message is
    # re-added separately), so cross-turn replay carries what the agent actually DID, not just its reply.
    head_len = len(messages)
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
            yield AgentCancelled(_turn_tail(messages, head_len))
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
            # Tool-incompatible model: empty text after a tool ran AND the stream did not end
            # with a recognized terminal reason. A native tool call already executed this turn
            # is itself PROOF the model is tool-compatible, so the only thing that still reads
            # as "couldn't continue" is a torn stream (done is None) or an unknown finish_reason
            # — `stop` / `length` / `content_filter` are all legitimate provider terminations,
            # not an incompatibility (length is a token-budget cutoff, handled separately below).
            fr = done.finish_reason if done is not None else ""
            incompatible = (
                not text_buf
                and total_tool_calls > 0
                and fr not in ("stop", "length", "content_filter")
            )
            if incompatible:
                logger.warning(
                    f"copilot: empty reply after {total_tool_calls} tool call(s) — "
                    "model is not tool-call compatible"
                )
                tr.event(
                    "model_incompatible", iteration=iteration, reason="empty_after_tool"
                )
                yield AgentError(
                    _MODEL_INCOMPATIBLE_MSG, messages=_turn_tail(messages, head_len)
                )
                return
            if not text_buf and fr == "length":
                # The model was cut off mid-reply by the per-turn token budget. Tell the user
                # honestly (not "incompatible") so they can ask it to continue.
                logger.warning(
                    f"copilot turn truncated (length) after {total_tool_calls} tool call(s)"
                )
                tr.event("turn_truncated", iteration=iteration, reason="length")
                yield AgentError(
                    "I ran out of my per-reply token budget before I could summarize. "
                    "The actions above did complete — ask me to continue or recap.",
                    messages=_turn_tail(messages, head_len),
                )
                return
            # The final assistant reply (text-only, no tool calls) is in text_buf but not yet in
            # `messages` (only tool-call iterations append an assistant message) — append it so the
            # persisted turn tail carries the agent's last word (feature 020·23 D4).
            if text_buf:
                messages.append(LLMMessage(role="assistant", content=text_buf))
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
            yield AgentTurnDone(
                ran.executed_actions_note(registry),
                usage.value(),
                messages=_turn_tail(messages, head_len),
            )
            return

        calls = [builders[i] for i in sorted(builders)]
        messages.append(_assistant_message(text_buf, calls))
        giveup = False
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
                # Malformed args for an EDIT tool is a non-converging retry too (§020·20 D4) — a
                # model that keeps emitting unparseable edit JSON must hit the same giveup cap, not
                # loop to max_iterations. Non-edit malformed calls don't count (same as a failed
                # non-edit tool below).
                if registry.is_edit_tool(tc.name):
                    consecutive_failed_edits += 1
                    if consecutive_failed_edits >= config.max_edit_retries:
                        giveup = True
                        break
                continue
            yield AgentStatus(registry.status_for(tc.name, args))
            if cancel.is_set():
                logger.debug(f"copilot turn cancelled before tool {tc.name}")
                yield AgentCancelled(_turn_tail(messages, head_len))
                return
            # Pre-gate guard (feature 020·18): a publish that can't run (no creds / no pack)
            # returns a guided-handoff message BEFORE the gate, so the user never gets a
            # confirm dialog for an action that would fail. Routes around execute + the gate +
            # the retry cap (a cred miss is not a convergence failure), exactly like a decline.
            handoff = registry.precheck(tc.name, args)
            if handoff is not None:
                logger.info(f"copilot tool {tc.name} | precheck handoff")
                tr.event("tool_precheck_handoff", name=tc.name, message=handoff)
                ran.record(tc.name, False, handoff)
                messages.append(_tool_message(tc.id, handoff))
                continue
            # Gate a destructive/publish tool on a user Yes/No before it runs (§7 / 020·17).
            # On decline: record + append the tool result (a declined call STILL needs a
            # matching tool message — an orphaned tool_call_id 400s the next stream) + continue
            # to the next call in the batch. The continue lands BEFORE the execute + the
            # consecutive_failed_edits logic, so a decline never reaches either: a user choice
            # is not a convergence failure and must not count toward the edit-retry cap.
            secret = ""  # a CREDENTIAL gate's typed key, forwarded to execute OUT of args (020·19)
            if registry.requires_gate(tc.name, args, config):
                req = build_gate(registry, tc.name, args)
                tr.event("gate_open", name=tc.name, prompt=req.prompt)
                yield AgentGateOpened(req)
                resp = gate.ask(req)
                if resp.cancelled:
                    logger.debug(f"copilot turn cancelled at gate for {tc.name}")
                    tr.event("gate_cancelled", name=tc.name)
                    yield AgentCancelled(_turn_tail(messages, head_len))
                    return
                if not resp.approved:
                    logger.info(f"copilot tool {tc.name} | user declined")
                    tr.event("gate_declined", name=tc.name)
                    ran.record(tc.name, False, "error: user declined")
                    messages.append(_tool_message(tc.id, "error: user declined"))
                    continue
                secret = resp.secret
            ok, msg, payload = registry.execute(tc.name, args, secret)
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
            display = str((payload or {}).get("display", ""))
            yield AgentToolCard(
                tc.name,
                ok,
                payload,
                result=msg,
                widget=_widget_from_payload(payload),
                display=display,
            )
            messages.append(_tool_message(tc.id, msg))

            # §I2 self-correction cap: a model stuck on an edit (an old_str that keeps not
            # matching, a line range that keeps not resolving) would otherwise retry to the
            # max_iterations ceiling. Count CONSECUTIVE failed shader-EDIT tools (not all mutating
            # tools — a failed render/publish is non-convergence, not a stuck edit, §020·20 D4);
            # any success or non-edit tool resets it. A freshness reject (payload["stale"], §15) is
            # a benign "re-read and continue", NOT a convergence failure, so it does not count.
            # payload is None on a malformed-args/unknown-tool result — guard the .get.
            stale = bool((payload or {}).get("stale"))
            if registry.is_edit_tool(tc.name) and not ok and not stale:
                consecutive_failed_edits += 1
            else:
                consecutive_failed_edits = 0
            if consecutive_failed_edits >= config.max_edit_retries:
                giveup = True
                break

        if giveup:
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
                "avoid looping. Tell me to try again, or describe the change differently."
            )
            yield AgentError(note, messages=_turn_tail(messages, head_len))
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
    yield AgentError(cutoff_note, messages=_turn_tail(messages, head_len))
