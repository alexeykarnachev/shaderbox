import json
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from loguru import logger

from shaderbox.copilot.config import CopilotConfig
from shaderbox.copilot.errors import CopilotConfigError
from shaderbox.copilot.gate import GateChannel, GateKind, GateRequest
from shaderbox.copilot.llm.api import (
    LLMClient,
    LLMDone,
    LLMMessage,
    LLMTextDelta,
    LLMToolCall,
    LLMToolCallCompleted,
    LLMToolCallStarted,
    LLMUsage,
)
from shaderbox.copilot.prompt import build_messages
from shaderbox.copilot.prompt_context import CopilotContext
from shaderbox.copilot.state import RESULT_WIDGET_KINDS, ResultWidget, TurnStats
from shaderbox.copilot.tools.registry import ToolRegistry
from shaderbox.copilot.trace import NULL_TRACE, TraceLog

# The agent loop: own a growing conversation, stream one assistant turn, execute any tool
# calls, append the results, re-stream until the model stops calling tools or a limit trips.
# A tool whose gate_policy trips requires_gate blocks on the GateChannel for a user Yes/No
# before it runs.

_MODEL_INCOMPATIBLE_MSG = (
    "The selected model isn't compatible with tool calling — after using a tool it "
    "produced neither a native tool call nor a text reply. Pick a different model in "
    "Settings -> Integrations -> Copilot."
)

_FINAL_REPLY_NUDGE = (
    "[engine] Tool budget exhausted for this turn. Reply to the USER now, plain text: "
    "address their request directly — state the file's NET state vs the start of the turn "
    "(the working set below is the live truth), what is still missing. Do not state "
    "intentions as done and do not claim visual results. Short. No tool calls."
)

_COMPILE_THRASH_NUDGE = (
    "\n\n[hint] That's several edits in a row that applied but still left compile errors. "
    "Stop patching line by line: re-read the FULL function/block from the working set, work out "
    "the whole correct version, and rewrite it in ONE edit (replace_lines over the whole range)."
)

_CLEAN_STREAK_NUDGE = (
    "\n\n[hint] That's many clean edits in one turn that the user has not seen — and you "
    "cannot see the visual result either. Unless something is still broken, STOP here: reply "
    "with a short summary of what you changed and let the user look before iterating further."
)


def _trunc(text: str, limit: int) -> str:
    # Log-line truncation with an ASCII marker so a cut value never reads as the whole thing.
    # The full text lives in the trace; these caps are for the terse log only.
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
    # The tool's result string (render path / publish URL / error text) for a result line under
    # the card. Goes to the LLM history + trace too.
    result: str = ""
    # Engine-rendered result widget from payload["widget"]: a button; the raw target never reaches
    # the model. None = no widget.
    widget: ResultWidget | None = None
    # Terse chat-display line from payload["display"]: when `result` is heavy (read_shader's full
    # source), the USER sees this summary while the full result still goes to the AGENT. "" = show `result`.
    display: str = ""


@dataclass(frozen=True)
class TurnSummary:
    # The engine-derived NL summary of a committed turn; replaces the verbatim tool tail in history.
    # `reply` is the agent's prose (final reply at clean-done; the note/error at a cutoff); `ledger`
    # is the mutating-action lines (new values + irreversible identities); `nodes` is every node
    # referenced this turn. _commit_turn renders these into one assistant history message.
    reply: str = ""
    ledger: list[str] = field(default_factory=list)
    nodes: list[str] = field(default_factory=list)


# The terminal events carry the engine-derived NL TurnSummary for _commit_turn to persist as one
# assistant history message. Empty default so the session's bare-except AgentError fallbacks (which
# never see run_turn's run-log) commit an empty summary.


@dataclass(frozen=True)
class AgentTurnDone:
    summary: TurnSummary = field(default_factory=TurnSummary)
    # The turn's stats for the header gauge: context = the FIRST iteration's input (standing
    # context, NOT the summed input which re-counts the growing context each iteration); reply/cost
    # are the turn totals. None only if no LLMDone ever fired (torn stream).
    stats: TurnStats | None = None


@dataclass(frozen=True)
class AgentError:
    message: str
    summary: TurnSummary = field(default_factory=TurnSummary)
    # Usage stats for the errored turn — errored spend must still reach the session
    # cost accounting (033; None on session-level fallback errors with no run).
    stats: TurnStats | None = None


@dataclass(frozen=True)
class AgentCancelled:
    summary: TurnSummary = field(default_factory=TurnSummary)
    # Cancelled turns still billed their iterations (033) — None on pre-stream cancels.
    stats: TurnStats | None = None


@dataclass(frozen=True)
class AgentGateOpened:
    # A gated tool is about to run; the worker is blocking on the user's Yes/No. pump_events
    # materializes a pending_action Message from this so the UI can draw the confirm.
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


@dataclass
class _RunEntry:
    name: str
    ok: bool
    msg: str  # the tool's terse model-facing result (carries set_uniform's new value)
    args: dict  # the call args — for node names referenced/targeted this turn
    payload: (
        dict | None
    )  # the structured side-channel — carries id / pack / url (NOT in msg)


# Max non-irreversible mutating lines kept in a turn-summary ledger; irreversible (publish/delete)
# lines are always kept verbatim (the don't-re-do safety invariant).
_LEDGER_SOFT_CAP: int = 8
# Tool-arg keys that name a node (every node touched or referenced this turn).
_NODE_ARG_KEYS: tuple[str, ...] = ("node", "target", "nodes")


class _RunLog:
    # The loop-local action ledger. Loop-private — never on state. Feeds the engine-derived NL
    # turn-summary persisted to history.
    def __init__(self) -> None:
        self._entries: list[_RunEntry] = []

    def record(
        self, name: str, ok: bool, msg: str, args: dict, payload: dict | None
    ) -> None:
        self._entries.append(_RunEntry(name, ok, msg, args, payload))

    def referenced_nodes(self) -> list[str]:
        # Every node name/handle the turn touched or referenced: args of every call, deduped,
        # order-preserved. A later turn's "do the same to C" needs the prior referent named.
        seen: dict[str, None] = {}
        for e in self._entries:
            for key in _NODE_ARG_KEYS:
                val = e.args.get(key)
                for handle in val if isinstance(val, list) else [val]:
                    if isinstance(handle, str) and handle:
                        seen[handle] = None
        return list(seen)

    def applied_mutations(self, registry: ToolRegistry) -> list[_RunEntry]:
        return [e for e in self._entries if registry.is_mutating(e.name) and e.ok]

    def summary_lines(self, registry: ToolRegistry) -> list[str]:
        # The ledger lines for the NL turn-summary. Irreversible actions (publish/delete — gated
        # always) carry their identity (id / pack / url, which live in `payload`, not `msg`) verbatim
        # and uncapped, so a "continue" after a cutoff never re-does them. Other mutating actions
        # carry verb + result and are soft-capped so a many-call turn can't bloat history.
        irreversible: list[str] = []
        other: list[str] = []
        for e in self._entries:
            if not registry.is_mutating(e.name):
                continue
            if registry.requires_gate_always(e.name):
                ident = _identity_from_payload(e.payload)
                status = "" if e.ok else " (FAILED)"
                tail = f" [{ident}]" if ident else ""
                irreversible.append(f"{e.name}{status}: {e.msg}{tail}")
            elif e.ok:
                other.append(f"{e.name}: {e.msg}")
            else:
                other.append(f"{e.name} FAILED: {e.msg}")
        if len(other) > _LEDGER_SOFT_CAP:
            kept = other[:_LEDGER_SOFT_CAP]
            kept.append(f"... and {len(other) - _LEDGER_SOFT_CAP} more edits")
            other = kept
        return irreversible + other


def _identity_from_payload(payload: dict | None) -> str:
    # Pull the action's durable identity out of a tool payload: a published URL or a deleted node
    # id/trash-name — whichever the tool surfaced. "" if none. (Pack ops carry their set_name only
    # in the verbatim `msg`, which the irreversible bucket keeps uncapped, so no payload key here.)
    if not payload:
        return ""
    for key in ("url", "node_id", "trash_name"):
        val = payload.get(key)
        if isinstance(val, str) and val:
            return f"{key}={val}"
    return ""


def _build_turn_summary(
    reply: str, run_log: _RunLog, registry: ToolRegistry
) -> TurnSummary:
    return TurnSummary(
        reply=reply,
        ledger=run_log.summary_lines(registry),
        nodes=run_log.referenced_nodes(),
    )


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
    # The double-escape signature: a quoted JSON string whose body carries escape markers
    # (\n \t \r \") but no real whitespace — the provider serialized a newline as the two chars
    # `\` `n`. A payload the model legitimately wrapped in literal double-quotes (e.g. `#include "x"`)
    # has no such marker, so it's left untouched: unwrapping it would strip a real quote level.
    if len(v) < 2 or v[0] != '"' or v[-1] != '"':
        return False
    body = v[1:-1]
    return any(m in body for m in _ESCAPE_MARKERS) and not any(
        c in body for c in " \t\n\r"
    )


def _unescape_double_escaped(args: dict) -> dict:
    # grok footgun: a string value can be double-escaped JSON ({"x": "\"y\""}). Unwrap one level
    # ONLY when the value carries the double-escape signature (_looks_double_escaped).
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


def _widget_from_payload(payload: dict | None) -> ResultWidget | None:
    # A tool surfaces a result widget via a {"kind","label","target"} dict under payload["widget"].
    # Guard defensively (known kind + non-empty target) so a malformed entry yields no widget rather
    # than a bad button.
    spec = (payload or {}).get("widget")
    if not isinstance(spec, dict):
        return None
    kind = spec.get("kind")
    target = spec.get("target", "")
    if kind not in RESULT_WIDGET_KINDS or not target:
        return None
    return ResultWidget(
        kind=kind, label=str(spec.get("label", "Open")), target=str(target)
    )


def _assistant_message(text: str, calls: list[_ToolCallBuilder]) -> LLMMessage:
    # content="" -> None when there are tool calls (grok quirk).
    return LLMMessage(
        role="assistant",
        content=text or None,
        tool_calls=[
            LLMToolCall(id=c.id, name=c.name, arguments=c.arguments) for c in calls
        ],
    )


def _tool_message(tool_call_id: str, content: str) -> LLMMessage:
    return LLMMessage(role="tool", tool_call_id=tool_call_id, content=content)


def build_gate(registry: ToolRegistry, name: str, args: dict) -> GateRequest:
    # Engine-built gate request: the engine owns the prompt phrasing so it's accurate, not the model.
    # A CREDENTIAL tool (gate_kind) gets a secret-input gate; everything else the CONFIRM Yes/No.
    # Falls back to a generic line for a gated tool without a gate_prompt (unknown names +
    # future BULK-gated tools, whose right prompt is plausibly the count-aware generic line).
    tool = registry.definition_for(name)
    prompt = (
        tool.gate_prompt(args)
        if tool is not None and tool.gate_prompt is not None
        else f"Run {name}? This action will change your project."
    )
    if tool is not None and tool.gate_kind is GateKind.CREDENTIAL:
        return GateRequest(
            kind=GateKind.CREDENTIAL, prompt=prompt, secret_field=tool.secret_field
        )
    if tool is not None and tool.gate_kind is GateKind.CONFIG:
        # secret_field names the integration whose draw_config_ui the card renders.
        return GateRequest(
            kind=GateKind.CONFIG, prompt=prompt, secret_field=tool.secret_field
        )
    return GateRequest(kind=GateKind.CONFIRM, prompt=prompt, options=["Yes", "No"])


def run_turn(
    client: LLMClient,
    registry: ToolRegistry,
    config: CopilotConfig,
    context: CopilotContext,
    history: list[LLMMessage],
    user_text: str,
    gate: GateChannel,
    cancel: threading.Event,
    trace: TraceLog | None = None,
    scratchpad_render: Callable[[], list[LLMMessage]] | None = None,
    batch_begin: Callable[[], None] | None = None,
) -> Iterator[AgentEvent]:
    # `trace` is the full-transcript sink (None in
    # tests). `scratchpad_render` rebuilds the live per-turn working-set block each iteration, spliced
    # onto the bottom of `messages` for the stream + trace, never into the durable list. `batch_begin`
    # clears the App-side per-batch line-edit guard once per tool-call batch (the batch boundary is the
    # only signal App can't see itself). Both default to no-ops (tests).
    tr = trace if trace is not None else NULL_TRACE
    render_scratchpad = (
        scratchpad_render if scratchpad_render is not None else (lambda: [])
    )
    begin_batch = batch_begin if batch_begin is not None else (lambda: None)
    # `messages` is the within-turn context: full assistant/tool pairs accumulate here as the loop
    # runs (the provider 400s on an orphaned tool_call_id). Never persisted — at commit the turn
    # collapses to one engine-derived NL TurnSummary, so history stays natural-language only.
    messages = build_messages(context, history, user_text)
    specs = registry.eager_specs()
    usage = LLMUsage()  # running per-turn total (LLMUsage.__add__)
    ran = _RunLog()
    total_tool_calls = 0
    consecutive_failed_edits = 0  # self-correction cap (reset on any other outcome)
    consecutive_compile_failures = 0  # applies-but-broken thrash counter
    compile_nudge_sent = (
        False  # latched once the nudge fires; re-armed by a non-thrash step
    )
    clean_edits_this_turn = (
        0  # cumulative clean source edits (the render-blind spree brake)
    )
    clean_streak_nudge_sent = False  # once per turn
    first_input_tokens: int | None = None  # iter-0 context size for the usage bar
    logger.info(f"copilot turn start | user={_trunc(user_text, 80)!r}")
    logger.debug(
        f"copilot turn detail | history_msgs={len(history)} "
        f"eager_tools={[s.name for s in specs]}"
    )
    tr.event("turn_start", user_text=user_text, history=history, eager_tools=specs)

    final_reply_text: list[str] = []

    def stream_final_reply() -> "Iterator[AgentEvent]":
        # One extra NO-TOOLS stream so a budget-exhausted turn still ends with the model
        # addressing the USER (3/10 round-3 turns ended silent — feature 033). Appends
        # nothing durable; usage folds into the turn total.
        nonlocal usage
        request_messages = (
            messages
            + render_scratchpad()
            + [LLMMessage(role="user", content=_FINAL_REPLY_NUDGE)]
        )
        tr.event(
            "llm_request",
            iteration=-1,
            messages=request_messages,
            tools=[],
            max_tokens=config.max_tokens_per_turn,
        )
        buf = ""
        done: LLMDone | None = None
        # A torn stream must not escape run_turn: the caller's empty-reply path
        # carries the REAL summary + stats (ledger, accumulated cost) downstream,
        # while a propagated exception would drop both at the session boundary.
        try:
            for ev in client.stream(
                request_messages, tools=None, max_tokens=config.max_tokens_per_turn
            ):
                match ev:
                    case LLMTextDelta():
                        buf += ev.text
                        yield AgentTextDelta(ev.text)
                    case LLMDone():
                        usage += ev.usage
                        done = ev
                    case _:
                        pass
        except Exception as exc:
            logger.warning(f"copilot final reply stream failed: {exc}")
            tr.event("final_reply_stream_error", error=str(exc))
        tr.event(
            "llm_response",
            iteration=-1,
            finish_reason=done.finish_reason if done else "no-done-event",
            text=buf,
            tool_calls=[],
            usage=done.usage if done else None,
        )
        final_reply_text.append(buf.strip())

    for iteration in range(config.max_iterations):
        if cancel.is_set():
            logger.debug(f"copilot turn cancelled at iteration {iteration}")
            yield AgentCancelled(
                _build_turn_summary("", ran, registry),
                stats=TurnStats(
                    context_tokens=first_input_tokens or 0,
                    reply_tokens=usage.output_tokens,
                    cost_usd=usage.cost_usd,
                ),
            )
            return

        text_buf = ""
        builders: dict[int, _ToolCallBuilder] = {}
        done: LLMDone | None = None
        # Rebuild the working-set scratchpad ONCE per iteration and splice it onto the bottom for
        # both the trace AND the stream (two render_scratchpad calls could diverge if a mutation
        # interleaved). The durable `messages` is never mutated.
        request_messages = messages + render_scratchpad()
        tr.event(
            "llm_request",
            iteration=iteration,
            messages=request_messages,
            tools=specs,
            max_tokens=config.max_tokens_per_turn,
        )
        # Same containment as stream_final_reply: a stream torn mid-turn (tools may already
        # have run) must terminate as an error terminal CARRYING the accumulated summary +
        # stats — a propagated exception would commit an empty ledger (re-publish risk) and
        # drop the turn's spend. A pre-stream config reject still propagates: the session
        # surfaces its message verbatim.
        try:
            for ev in client.stream(
                request_messages, tools=specs, max_tokens=config.max_tokens_per_turn
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
                        usage += ev.usage
                        if first_input_tokens is None:
                            first_input_tokens = ev.usage.input_tokens
                        done = ev
        except CopilotConfigError:
            raise
        except Exception as exc:
            logger.warning(f"copilot stream failed mid-turn: {exc}")
            tr.event("stream_error", iteration=iteration, error=str(exc))
            note = (
                f"The model stream failed mid-turn ({type(exc).__name__}). "
                "Any actions shown above did complete - ask me to continue or recap."
            )
            yield AgentError(
                note,
                summary=_build_turn_summary(note, ran, registry),
                stats=TurnStats(
                    context_tokens=first_input_tokens or 0,
                    reply_tokens=usage.output_tokens,
                    cost_usd=usage.cost_usd,
                ),
            )
            return

        fr = done.finish_reason if done else "no-done-event"
        u = done.usage if done else None
        tokens = (
            f"in={u.input_tokens} out={u.output_tokens} rsn={u.reasoning_tokens} "
            f"cost=${u.cost_usd:.6f}"
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
            # Tool-incompatible model: empty text after a tool ran AND the stream ended with no
            # recognized terminal reason. A native tool call having executed is itself proof the
            # model is tool-compatible, so the only "couldn't continue" left is a torn stream
            # (done is None) or an unknown finish_reason — `stop` / `length` / `content_filter`
            # are legitimate terminations (length is a token-budget cutoff, handled below).
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
                    _MODEL_INCOMPATIBLE_MSG,
                    summary=_build_turn_summary("", ran, registry),
                    stats=TurnStats(
                        context_tokens=first_input_tokens or 0,
                        reply_tokens=usage.output_tokens,
                        cost_usd=usage.cost_usd,
                    ),
                )
                return
            if not text_buf and fr in ("length", "stop", "content_filter"):
                # Cut off by the per-iteration token budget with nothing produced (a hidden
                # reasoning burn). Force one NO-TOOLS reply so the user never gets silence.
                logger.warning(
                    f"copilot turn truncated (length) after {total_tool_calls} tool call(s)"
                )
                tr.event("turn_truncated", iteration=iteration, reason=fr)
                if cancel.is_set():
                    yield AgentCancelled(
                        _build_turn_summary("", ran, registry),
                        stats=TurnStats(
                            context_tokens=first_input_tokens or 0,
                            reply_tokens=usage.output_tokens,
                            cost_usd=usage.cost_usd,
                        ),
                    )
                    return
                yield from stream_final_reply()
                reply = final_reply_text[-1] if final_reply_text else ""
                if not reply:
                    reply = (
                        "I could not produce a reply this turn. The actions above did "
                        "complete — ask me to continue or recap."
                    )
                    yield AgentError(
                        reply,
                        summary=_build_turn_summary(reply, ran, registry),
                        stats=TurnStats(
                            context_tokens=first_input_tokens or 0,
                            reply_tokens=usage.output_tokens,
                            cost_usd=usage.cost_usd,
                        ),
                    )
                    return
                stats = TurnStats(
                    context_tokens=first_input_tokens or 0,
                    reply_tokens=usage.output_tokens,
                    cost_usd=usage.cost_usd,
                )
                yield AgentTurnDone(
                    summary=_build_turn_summary(reply, ran, registry), stats=stats
                )
                return
            logger.info(
                f"copilot turn done | iterations={iteration + 1} "
                f"tool_calls={total_tool_calls} reply={len(text_buf)}ch "
                f"total_in={usage.input_tokens} cost=${usage.cost_usd:.6f}"
            )
            tr.event(
                "turn_done",
                iterations=iteration + 1,
                tool_calls=total_tool_calls,
                reply=text_buf,
                usage=usage,
            )
            turn_total = usage
            stats = TurnStats(
                context_tokens=first_input_tokens or 0,
                reply_tokens=turn_total.output_tokens,
                cost_usd=turn_total.cost_usd,
            )
            # text_buf is the agent's final reply, carrying its stated assumption.
            yield AgentTurnDone(
                summary=_build_turn_summary(text_buf, ran, registry),
                stats=stats,
            )
            return

        calls = [builders[i] for i in sorted(builders)]
        messages.append(_assistant_message(text_buf, calls))
        begin_batch()  # reset the per-batch line-edit guard
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
                # Malformed args for an EDIT tool is a non-converging retry too — a model that keeps
                # emitting unparseable edit JSON must hit the same giveup cap, not loop to
                # max_iterations. Non-edit malformed calls don't count.
                if registry.is_edit_tool(tc.name):
                    consecutive_failed_edits += 1
                    if consecutive_failed_edits >= config.max_edit_retries:
                        giveup = True
                        break
                continue
            yield AgentStatus(registry.status_for(tc.name, args))
            if cancel.is_set():
                logger.debug(f"copilot turn cancelled before tool {tc.name}")
                yield AgentCancelled(
                    _build_turn_summary(text_buf, ran, registry),
                    stats=TurnStats(
                        context_tokens=first_input_tokens or 0,
                        reply_tokens=usage.output_tokens,
                        cost_usd=usage.cost_usd,
                    ),
                )
                return
            # Pre-gate guard: a publish that can't run (no creds / no pack) returns a guided-handoff
            # message BEFORE the gate, so the user never gets a confirm dialog for an action that
            # would fail. Routes around execute + gate + retry cap (a cred miss is not a convergence
            # failure), like a decline.
            handoff = registry.precheck(tc.name, args)
            if handoff is not None:
                logger.info(f"copilot tool {tc.name} | precheck handoff")
                tr.event("tool_precheck_handoff", name=tc.name, message=handoff)
                ran.record(tc.name, False, handoff, args, None)
                messages.append(_tool_message(tc.id, handoff))
                continue
            # Gate a destructive/publish tool on a user Yes/No before it runs. On decline: record +
            # append the tool result (a declined call STILL needs a matching tool message — an
            # orphaned tool_call_id 400s the next stream) + continue. The continue lands BEFORE
            # execute and the consecutive_failed_edits logic, so a user decline never counts toward
            # the edit-retry cap.
            secret = ""  # a CREDENTIAL gate's typed key, forwarded to execute
            if registry.requires_gate(tc.name, args, config):
                req = build_gate(registry, tc.name, args)
                tr.event("gate_open", name=tc.name, prompt=req.prompt)
                yield AgentGateOpened(req)
                resp = gate.ask(req)
                if resp.cancelled:
                    logger.debug(f"copilot turn cancelled at gate for {tc.name}")
                    tr.event("gate_cancelled", name=tc.name)
                    yield AgentCancelled(
                        _build_turn_summary(text_buf, ran, registry),
                        stats=TurnStats(
                            context_tokens=first_input_tokens or 0,
                            reply_tokens=usage.output_tokens,
                            cost_usd=usage.cost_usd,
                        ),
                    )
                    return
                if not resp.approved:
                    logger.info(f"copilot tool {tc.name} | user declined")
                    tr.event("gate_declined", name=tc.name)
                    ran.record(tc.name, False, "error: user declined", args, None)
                    messages.append(
                        _tool_message(
                            tc.id,
                            f"error: user declined — the {tc.name} did NOT happen. "
                            "Tell the user it was not done; do not retry it this turn.",
                        )
                    )
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
            ran.record(tc.name, ok, msg, args, payload)
            display = str((payload or {}).get("display", ""))
            yield AgentToolCard(
                tc.name,
                ok,
                payload,
                result=msg,
                widget=_widget_from_payload(payload),
                display=display,
            )

            # Self-correction cap: a model stuck on an edit (an old_str that keeps not matching, a
            # line range that keeps not resolving) would otherwise retry to the max_iterations
            # ceiling. Count CONSECUTIVE failed shader-EDIT tools (not all mutating tools — a failed
            # render/publish is non-convergence, not a stuck edit); any success or non-edit tool
            # resets it.
            if registry.is_edit_tool(tc.name) and not ok:
                consecutive_failed_edits += 1
            else:
                consecutive_failed_edits = 0

            # Applies-but-broken thrash: an edit that APPLIES (ok=True) but leaves compile errors
            # resets the failed-edit cap above, so a model that keeps producing broken-but-applying
            # edits would loop to max_iterations. Count those separately and, at the cap, splice a
            # rewrite nudge onto THIS edit's tool message — not a giveup; the model usually recovers.
            # The latch fires the nudge ONCE per thrash run; a non-thrash step re-arms it (so a fresh
            # thrash run after a clean edit nudges again, but a model ignoring it isn't re-nudged
            # every max_compile_failures steps).
            applied_with_errors = (
                registry.is_edit_tool(tc.name)
                and ok
                and bool((payload or {}).get("errors"))
            )
            if applied_with_errors:
                consecutive_compile_failures += 1
            else:
                consecutive_compile_failures = 0
                compile_nudge_sent = False
            if (
                config.max_compile_failures > 0
                and consecutive_compile_failures >= config.max_compile_failures
                and not compile_nudge_sent
            ):
                msg += _COMPILE_THRASH_NUDGE
                compile_nudge_sent = True
                tr.event("compile_thrash_nudge", iteration=iteration)

            # Render-blind spree brake: clean edits never trip either counter above, so a
            # model iterating on AESTHETICS can stack them unbounded with the user seeing
            # nothing. Cumulative per turn (a read/grep between edits is not user contact);
            # one nudge per turn, while a broken compile is in flight the turn stays exempt
            # (fixing comes first — those edits count toward the thrash nudge instead).
            if registry.is_edit_tool(tc.name) and ok and not applied_with_errors:
                clean_edits_this_turn += 1
            if (
                config.max_clean_edit_streak > 0
                and clean_edits_this_turn >= config.max_clean_edit_streak
                and not clean_streak_nudge_sent
            ):
                msg += _CLEAN_STREAK_NUDGE
                clean_streak_nudge_sent = True
                tr.event("clean_streak_nudge", iteration=iteration)

            messages.append(_tool_message(tc.id, msg))

            if consecutive_failed_edits >= config.max_edit_retries:
                giveup = True
                break

        if giveup:
            logger.warning(
                f"copilot edit giveup after {consecutive_failed_edits} failed "
                f"edits | total_in={usage.input_tokens} "
                f"cost=${usage.cost_usd:.6f}"
            )
            tr.event(
                "edit_giveup",
                consecutive_failed_edits=consecutive_failed_edits,
                usage=usage,
            )
            note = (
                f"I couldn't apply that edit after {consecutive_failed_edits} tries "
                "— the edit kept not applying to the shader source. I've stopped to "
                "avoid looping. Tell me to try again, or describe the change differently."
            )
            applied = ran.applied_mutations(registry)
            if applied:
                note += "\nWhat DID apply this turn:\n" + "\n".join(
                    f"{e.name}: {e.msg}" for e in applied
                )
                last = applied[-1]
                if (last.payload or {}).get("errors"):
                    target = last.args.get("target")
                    node = (
                        target
                        if isinstance(target, str) and target
                        else "the current node"
                    )
                    note += f"\nnote: {node} is currently left with compile errors."
            yield AgentError(
                note,
                summary=_build_turn_summary(note, ran, registry),
                stats=TurnStats(
                    context_tokens=first_input_tokens or 0,
                    reply_tokens=usage.output_tokens,
                    cost_usd=usage.cost_usd,
                ),
            )
            return

    logger.warning(
        f"copilot turn hit max_iterations={config.max_iterations} | "
        f"tool_calls={total_tool_calls} total_in={usage.input_tokens} "
        f"cost=${usage.cost_usd:.6f}"
    )
    if cancel.is_set():
        yield AgentCancelled(
            _build_turn_summary("", ran, registry),
            stats=TurnStats(
                context_tokens=first_input_tokens or 0,
                reply_tokens=usage.output_tokens,
                cost_usd=usage.cost_usd,
            ),
        )
        return
    yield from stream_final_reply()
    tr.event(
        "turn_done",
        cutoff="max_iterations",
        iterations=config.max_iterations,
        tool_calls=total_tool_calls,
        usage=usage,
    )
    reply = final_reply_text[-1] if final_reply_text else ""
    if not reply:
        reply = (
            f"I stopped after {config.max_iterations} steps without finishing this "
            "turn. Ask me to continue, or rephrase what you need."
        )
        yield AgentError(
            reply,
            summary=_build_turn_summary(reply, ran, registry),
            stats=TurnStats(
                context_tokens=first_input_tokens or 0,
                reply_tokens=usage.output_tokens,
                cost_usd=usage.cost_usd,
            ),
        )
        return
    stats = TurnStats(
        context_tokens=first_input_tokens or 0,
        reply_tokens=usage.output_tokens,
        cost_usd=usage.cost_usd,
    )
    yield AgentTurnDone(summary=_build_turn_summary(reply, ran, registry), stats=stats)
