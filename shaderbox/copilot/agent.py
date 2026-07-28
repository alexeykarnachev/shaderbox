import json
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from loguru import logger

from shaderbox.copilot.address import is_example_address, is_lib_address
from shaderbox.copilot.config import COPILOT_CONFIG, CopilotConfig
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
from shaderbox.copilot.tools.registry import LOAD_TOOLS_NAME, ToolRegistry
from shaderbox.copilot.trace import NULL_TRACE, TraceLog
from shaderbox.copilot.vision_contract import ASK_NOT_MET, VisionUsage

# The agent loop: own a growing conversation, stream one assistant turn, execute any tool
# calls, append the results, re-stream until the model stops calling tools or a limit trips.
# A tool whose gate_policy trips requires_gate blocks on the GateChannel for a user Yes/No
# before it runs.

_MODEL_INCOMPATIBLE_MSG = (
    "The selected model isn't compatible with tool calling — after using a tool it "
    "produced neither a native tool call nor a text reply. Pick a different model in "
    "Settings -> Integrations -> Copilot."
)

# A stream that ended with no terminal event at all: a transport tear, not a model trait —
# so no Settings advice (the model is fine; the connection wasn't).
_TORN_STREAM_MSG = "The connection dropped mid-reply — try again. Any actions shown above did complete."


def _final_reply_nudge(cause: str) -> str:
    # The closing no-tools nudge at a FORCED turn-end. `cause` names the ENGINE limit that ended
    # the turn (NOT a user pause) so the reply owns the stop ("I hit my own limit and stopped")
    # instead of crediting the user with a pause they didn't make (feature 050: the "you're right
    # to pause now" misattribution). Stays plain ASCII - it is engine text injected as the user.
    return (
        f"[engine] {cause} (this is an engine limit, NOT a pause the user asked for). The turn "
        "is ending now. Reply to the USER, plain text: tell them you hit your own limit and "
        "stopped, state the file's NET state vs the start of the turn (the working set below is "
        "the live truth) and what is still missing, and ask if they want you to continue. Do not "
        "state intentions as done and do not claim visual results. Short. No tool calls."
    )


_COMPILE_THRASH_NUDGE = (
    "\n\n[hint] That's several edits in a row that applied but still left compile errors. "
    "Stop patching line by line: re-read the FULL function/block from the working set, work out "
    "the whole correct version, and rewrite it in ONE edit (write_shader, or edit_shader for "
    "a single block in a large file)."
)

# Tools that change what the CURRENT node renders — a turn that ran one of these ok produced a
# visual result (053 slice B). Structural ops (rename/delete/switch/canvas-size) are excluded.
_RENDER_AUTHORING_TOOLS = frozenset(
    {"edit_shader", "write_shader", "set_uniform", "edit_script", "write_script"}
)


def _turn_intent_look_for(user_text: str) -> str:
    # Aim the turn-end auto-look at the ACTUAL ask (054): an empty look_for gave the eye only a bland
    # baseline read that couldn't tell whether the result matches what was requested. Carrying the ask
    # makes the eye critique against the goal ("stripes muddy, no stars visible"). Empty/whitespace ask
    # -> "" (fall back to the baseline; don't fabricate an intent). Bounded so a long ask can't bloat.
    ask = " ".join(user_text.split())[: COPILOT_CONFIG.auto_look_intent_max_chars]
    if not ask:
        return ""
    return f'does the current frame actually achieve what the user asked: "{ask}"'


def _auto_look_fact(
    look_line: str, round_index: int, node_label: str, not_met: bool
) -> str:
    # Injected as the USER role after an ENGINE look. The provenance is stated THREE ways ("you did
    # NOT call this" / "the engine took a look FOR you" / "engine-injected data, not something you
    # did") so the model is never confused about who looked. Every clause must be TRUE in context:
    # the round is named (the model may have looked itself, and later rounds are not "the first"),
    # and a probed node that is NOT the current one is named instead of claiming "the current
    # frame". Plain ASCII (engine text on the user channel).
    which = "ONE vision look" if round_index <= 1 else f"vision look #{round_index}"
    where = f"at node {node_label}" if node_label else "on the current frame"
    framing = (
        "The engine's eye reports the ask is NOT met. Fix it, or reply honestly stating what is "
        "missing -- do NOT claim it is done. (An honest staged reply -- what landed, what is "
        "next -- is fine; a done-claim the read does not support is not.)\n"
        if not_met
        else ""
    )
    return (
        "[automatic visual check -- you did NOT call this] You changed the render this turn, so the "
        f"engine took {which} FOR you {where}, aimed at what the USER asked (this is engine-injected "
        "data, not a look you performed):\n\n"
        f"{look_line}\n\n"
        f"{framing}"
        "Treat it as the eye's OBSERVATION, not a verdict and not a beauty judgment. If it contradicts "
        "what you intended, fix it now before the user sees it; if it matches, briefly tell the user "
        "what the render shows. Do NOT call probe_render again just to repeat this same look.\n"
        "CRITICAL: do NOT claim in your reply anything this read does not support. If the thing you just "
        "changed is NOT visible here (a pole, a colour, stars, a whole element), then it did NOT land -- "
        "say so plainly and either fix it or tell the user it is still missing. NEVER assert a result the "
        "eye does not show."
    )


def _eye_summary_line(looks: int, read: str) -> str:
    # The not-met verdict's durable trace in the turn record: without it the history keeps only the
    # model's claim and the eye has no counter-voice next turn.
    return f"eye: ask not-met after {looks} look{'s' if looks != 1 else ''} -- {_trunc(' '.join(read.split()), COPILOT_CONFIG.eye_summary_max_chars)}"


def _clean_streak_fact(n: int) -> str:
    # Escalating per-file nudge: rides EVERY clean edit_shader result past the soft threshold,
    # getting more imperative — the one-shot version (a single soft nudge) was blown past in a
    # 16-edit spiral. The model is render-blind between looks, so nothing else brakes a clean
    # micro-edit spree.
    return (
        f"\n\n[hint] {n} clean edits to this file in a row, none of which the user has seen "
        "(the engine's eye checks the frame at turn-end, but that is a bounded check, not the "
        "user looking). Unless something is still broken, STOP: "
        "if more changes remain, make them in ONE write_shader, then reply with a short summary "
        "and let the user look before iterating further."
    )


# Edit tools whose artifact is the node's script.py rather than its GLSL — the streak keys
# namespace on this so a node's two files never share one brake.
_SCRIPT_EDIT_TOOLS = frozenset({"write_script", "edit_script"})
_WRITE_TOOLS = frozenset({"write_shader", "write_script"})


def _node_arg(args: dict[str, object]) -> str:
    # The raw node-address string an edit/authoring call targeted ("" = the current node): the
    # tools disagree on the arg NAME (`target` for shaders, `node` for scripts), so read through
    # the shared vocabulary instead of one hardcoded key.
    for key in _NODE_ARG_KEYS:
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _edit_target_key(name: str, args: dict[str, object]) -> tuple[str, str]:
    # The per-FILE key for the clean-edit streak: (artifact kind, raw target) — a node's GLSL and
    # its script.py are two files and must not share a streak. Empty target = the current node,
    # keyed to a stable sentinel so consecutive no-target edits count as ONE file. The key holds
    # the RAW target string, not a resolved node id (run_turn has no resolver in scope), so editing
    # one node both by-empty and by-its-id splits the streak across two keys and the brake counts
    # slower for that file. Harmless: it only DELAYS the brake (never disables it), and a real
    # spree uses one addressing style throughout.
    kind = "script" if name in _SCRIPT_EDIT_TOOLS else "shader"
    return kind, _node_arg(args) or "<current>"


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
    # The engine eye's closing verdict line, set only when the turn's FINAL engine look reported
    # the ask not met — so the record carries the eye's voice, not just the model's claim.
    eye: str = ""


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
# Tool-arg keys that name a node (every node touched or referenced this turn).
_NODE_ARG_KEYS: tuple[str, ...] = ("node", "target", "nodes")


class _RunLog:
    # The loop-local action ledger. Loop-private — never on state. Feeds the engine-derived NL
    # turn-summary persisted to history.
    def __init__(self) -> None:
        self._entries: list[_RunEntry] = []
        # The eye's not-met line for the turn record, rewritten by each engine look (only the
        # FINAL verdict is reported).
        self.eye_note: str = ""

    def record(
        self, name: str, ok: bool, msg: str, args: dict, payload: dict | None
    ) -> None:
        self._entries.append(_RunEntry(name, ok, msg, args, payload))

    def __bool__(self) -> bool:
        # True once any tool has run this turn — the "did work worth preserving on an error" signal.
        return bool(self._entries)

    def last_index(self) -> int:
        return len(self._entries) - 1

    def mutated_since(self, index: int) -> bool:
        # A render-authoring tool ran ok AFTER `index` (-1 = the whole turn). Explicit index rather
        # than a tool-name scan: the engine's own look records as `probe_render` exactly like the
        # model's, so a name scan would let a model probe reopen the window.
        return any(
            e.ok and e.name in _RENDER_AUTHORING_TOOLS
            for e in self._entries[index + 1 :]
        )

    def last_render_target(self) -> str:
        # The raw node-address the LAST successful render-authoring call targeted, for aiming the
        # engine look at the node the turn actually changed. "" = the current node — including when
        # that call targeted a lib/example address, which the probe resolver can't resolve (a look
        # at the current frame beats erroring the look away).
        for e in reversed(self._entries):
            if not (e.ok and e.name in _RENDER_AUTHORING_TOOLS):
                continue
            raw = _node_arg(e.args)
            if is_lib_address(raw) or is_example_address(raw):
                return ""
            return raw
        return ""

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
        cap = COPILOT_CONFIG.turn_ledger_soft_cap
        if len(other) > cap:
            kept = other[:cap]
            kept.append(f"... and {len(other) - cap} more edits")
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
        eye=run_log.eye_note,
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
    model: str = "",
) -> Iterator[AgentEvent]:
    # `trace` is the full-transcript sink (None in
    # tests). `scratchpad_render` rebuilds the live per-turn working-set block each iteration, spliced
    # onto the bottom of `messages` for the stream + trace, never into the durable list. `batch_begin`
    # clears the App-side per-batch rewrite guard once per tool-call batch (the batch boundary is the
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
    loaded_tools: set[str] = set()  # lazily-loaded tools, turn-scoped (feature 052 §0)
    specs = registry.assemble_specs(
        loaded_tools
    )  # eager core (sorted); grows on load_tools
    usage = LLMUsage()  # running per-turn total (LLMUsage.__add__)
    ran = _RunLog()
    total_tool_calls = 0
    consecutive_failed_edits = 0  # self-correction cap (reset on any other outcome)
    # Convergence loop (056): engine looks taken this turn, the _RunLog index at the last one (the
    # window the next look's mutation gate reads), and the visible reply of each round the engine
    # re-opened (the screen accumulates them, so the turn record must too).
    looks_used = 0
    last_look_index = -1
    round_replies: list[str] = []
    consecutive_compile_failures = 0  # applies-but-broken thrash counter
    compile_nudge_sent = (
        False  # latched once the nudge fires; re-armed by a non-thrash step
    )
    clean_edits_by_file: dict[
        tuple[str, str], int
    ] = {}  # per-(kind, file) clean-edit streak (spree brake)
    first_input_tokens: int | None = None  # iter-0 context size for the usage bar
    logger.info(f"copilot turn start | user={_trunc(user_text, 80)!r}")
    logger.debug(
        f"copilot turn detail | history_msgs={len(history)} "
        f"eager_tools={[s.name for s in specs]}"
    )
    tr.event(
        "turn_start",
        model=model or "(unset)",
        user_text=user_text,
        history=history,
        eager_tools=specs,
    )

    final_reply_text: list[str] = []

    def turn_reply(last_text: str) -> str:
        # The turn's whole visible prose: every earlier CONV round's reply plus this one, in the
        # order the user saw them stream.
        return "\n\n".join([p for p in (*round_replies, last_text.strip()) if p])

    def stream_final_reply(cause: str) -> "Iterator[AgentEvent]":
        # One extra NO-TOOLS stream so a forced turn-end still ends with the model addressing the
        # USER (3/10 round-3 turns ended silent — feature 033). `cause` names the engine limit so
        # the closing reply owns the stop, not the user (feature 050). Appends nothing durable;
        # usage folds into the turn total.
        nonlocal usage
        request_messages = (
            messages
            + render_scratchpad()
            + [LLMMessage(role="user", content=_final_reply_nudge(cause))]
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

    turn_started = time.monotonic()
    time_budget_hit = False

    for iteration in range(config.max_iterations):
        if cancel.is_set():
            logger.debug(f"copilot turn cancelled at iteration {iteration}")
            yield AgentCancelled(
                _build_turn_summary(turn_reply(""), ran, registry),
                stats=TurnStats(
                    context_tokens=first_input_tokens or 0,
                    reply_tokens=usage.output_tokens,
                    cost_usd=usage.cost_usd,
                ),
            )
            return
        if (
            config.turn_time_budget_s > 0
            and iteration > 0
            and time.monotonic() - turn_started > config.turn_time_budget_s
        ):
            time_budget_hit = True
            break

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
                if cancel.is_set():
                    # Per-delta cancel: a Stop / release during a SLOW stream is responsive here,
                    # not only at the iteration top — else the worker rides the request timeout
                    # (043 hang). The partial text/tool-calls are discarded; the loop's top-of-
                    # iteration cancel check below returns the cancelled terminal.
                    break
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
            # A pre-work config reject (key/model wiped before any tool ran) still propagates — the
            # session surfaces its message verbatim with an empty ledger (correct: nothing happened).
            # But once tools HAVE run (credentials cleared in Settings mid-turn), the same containment
            # as the stream-error path below applies: terminate carrying the accumulated summary +
            # stats, or the turn's ledger + spend are silently dropped.
            if not ran:
                raise
            note = (
                "The turn was aborted (copilot credentials cleared mid-turn). "
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

        if cancel.is_set():
            # The stream broke out on a mid-delta cancel (043) — terminate the turn now with the
            # accumulated summary + spend, before executing any partial tool calls.
            tr.event("turn_cancelled", iteration=iteration)
            yield AgentCancelled(
                _build_turn_summary(turn_reply(text_buf), ran, registry),
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
            # Empty text after a tool ran has two very different causes, diagnosed apart below: a
            # torn transport (no terminal event at all) vs an unknown finish_reason (a model that
            # can't continue after a tool call). `stop` / `length` / `content_filter` are
            # legitimate terminations (length is a token-budget cutoff, handled below).
            fr = done.finish_reason if done is not None else ""
            silent_after_tool = not text_buf and total_tool_calls > 0
            if silent_after_tool and done is None:
                # A stream that produced no terminal event at all is a torn connection, not an
                # incompatible model — the Settings advice would send the user chasing the wrong fix.
                logger.warning("copilot: stream torn before any terminal event")
                tr.event("stream_torn", iteration=iteration)
                yield AgentError(
                    _TORN_STREAM_MSG,
                    summary=_build_turn_summary("", ran, registry),
                    stats=TurnStats(
                        context_tokens=first_input_tokens or 0,
                        reply_tokens=usage.output_tokens,
                        cost_usd=usage.cost_usd,
                    ),
                )
                return
            incompatible = silent_after_tool and fr not in (
                "stop",
                "length",
                "content_filter",
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
                        _build_turn_summary(turn_reply(""), ran, registry),
                        stats=TurnStats(
                            context_tokens=first_input_tokens or 0,
                            reply_tokens=usage.output_tokens,
                            cost_usd=usage.cost_usd,
                        ),
                    )
                    return
                yield from stream_final_reply("You reached the per-turn token budget")
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
                    summary=_build_turn_summary(turn_reply(reply), ran, registry),
                    stats=stats,
                )
                return
            # Convergence loop (056): the turn changed the render since the last ENGINE look, so the
            # engine takes an aimed look FOR the model and injects the observation (as data, with
            # round- and target-aware provenance), giving it one more iteration. UNCONDITIONAL on the
            # model's own probes — its look answers ITS question, not the user's ask. Bounded by the
            # cap and gated on tool facts alone. With vision OFF the block is skipped whole: no
            # render, no card, no line — a vision-less user pays nothing new.
            if (
                config.copilot_vision_enabled
                and looks_used < config.copilot_convergence_max_looks
                and fr == "stop"
                and text_buf.strip()
                and iteration + 1 < config.max_iterations
                and not cancel.is_set()
                and ran.mutated_since(last_look_index)
            ):
                look_node = ran.last_render_target()
                look_args: dict[str, object] = {
                    "node": look_node,
                    "t": 0.0,
                    "look_for": _turn_intent_look_for(user_text),
                }
                ok_l, look_msg, look_payload = registry.execute(
                    "probe_render", look_args, ""
                )
                looks_used += 1
                ran.record("probe_render", ok_l, look_msg, look_args, look_payload)
                last_look_index = ran.last_index()
                payload = look_payload or {}
                vision_ok = ok_l and bool(payload.get("vision_ok"))
                verdict = payload.get("verdict")
                not_met = vision_ok and verdict == ASK_NOT_MET
                look_usage = payload.get("usage")
                if isinstance(look_usage, VisionUsage):
                    # COST only: the vision tokens are a different model's and must not move the
                    # main model's reply gauge.
                    usage += LLMUsage(cost_usd=look_usage.cost_usd)
                    tr.event(
                        "engine_look_usage",
                        iteration=iteration,
                        input_tokens=look_usage.input_tokens,
                        output_tokens=look_usage.output_tokens,
                        cost_usd=look_usage.cost_usd,
                    )
                # Every engine look leaves a verdict event, blind ones included ("none") — a look
                # that came back sightless is exactly what a trace review needs to see.
                tr.event(
                    "ask_verdict",
                    iteration=iteration,
                    verdict=verdict if vision_ok and verdict is not None else "none",
                    ask_line=payload.get("ask_line", ""),
                    node=look_node or "(current)",
                    look=looks_used,
                    vision_ok=vision_ok,
                )
                # The eye's note tracks THIS look: a not-met from an earlier round must not survive
                # a later look that saw the ask met (or saw nothing) into the turn record.
                ran.eye_note = (
                    _eye_summary_line(looks_used, str(payload.get("read", "")))
                    if not_met
                    else ""
                )
                yield AgentToolCard(
                    "probe_render",
                    ok_l,
                    {**payload, "engine_look": True},
                    result=look_msg,
                    widget=None,
                    display="",
                )
                if vision_ok:
                    logger.info(
                        f"copilot engine look #{looks_used} | verdict={verdict}"
                    )
                    round_replies.append(text_buf.strip())
                    if text_buf.strip():
                        yield AgentTextDelta("\n\n")
                    messages.append(_assistant_message(text_buf, []))
                    messages.append(
                        LLMMessage(
                            role="user",
                            content=_auto_look_fact(
                                look_msg, looks_used, look_node, not_met
                            ),
                        )
                    )
                    continue
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
                summary=_build_turn_summary(turn_reply(text_buf), ran, registry),
                stats=stats,
            )
            return

        calls = [builders[i] for i in sorted(builders)]
        messages.append(_assistant_message(text_buf, calls))
        begin_batch()  # reset the per-batch rewrite guard
        giveup = False
        clean_streak_giveup = (
            False  # hard clean-edit cap: force-end the turn (vs the failed-edit giveup)
        )
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
            if tc.name == LOAD_TOOLS_NAME:
                # Meta-tool (feature 052 §0): intercept BEFORE execute — it mutates the turn's tools=
                # set (engine state), not project state. Add the valid lazy names to the loaded set +
                # rebuild specs so the NEXT iteration's stream carries them.
                raw = args.get("names", [])
                requested = raw if isinstance(raw, list) else []
                newly = [
                    n
                    for n in requested
                    if registry.is_lazy(n) and n not in loaded_tools
                ]
                loaded_tools.update(newly)
                specs = registry.assemble_specs(loaded_tools)
                load_msg = (
                    f"loaded {', '.join(newly)} — callable for the rest of this turn."
                    if newly
                    else "no new tools loaded (already loaded, or not a lazy tool name)."
                )
                total_tool_calls += 1
                tr.event(
                    "tool_call",
                    n=total_tool_calls,
                    name=tc.name,
                    args=args,
                    ok=True,
                    result=load_msg,
                    payload=None,
                )
                ran.record(tc.name, True, load_msg, args, None)
                yield AgentToolCard(
                    tc.name, True, None, result=load_msg, widget=None, display=""
                )
                messages.append(_tool_message(tc.id, load_msg))
                continue
            yield AgentStatus(registry.status_for(tc.name, args))
            if cancel.is_set():
                logger.debug(f"copilot turn cancelled before tool {tc.name}")
                yield AgentCancelled(
                    _build_turn_summary(turn_reply(text_buf), ran, registry),
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
                # ok=True on both the record and the card: a deflection is not a failure (the
                # snippet square colours on ok, and the ledger would otherwise persist "(FAILED)").
                # total_tool_calls stays put — it feeds the incompatible heuristic + giveup notes.
                ran.record(tc.name, True, handoff, args, {"handoff": True})
                yield AgentToolCard(
                    tc.name, True, {"handoff": True}, result=handoff, display=""
                )
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
                        _build_turn_summary(turn_reply(text_buf), ran, registry),
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
                tr.event("gate_approved", name=tc.name)
                secret = resp.secret
            ok, msg, payload = registry.execute(tc.name, args, secret)
            tool_usage = (payload or {}).get("usage")
            if isinstance(tool_usage, VisionUsage):
                # A tool that made its OWN billed vision call (the model's probe_render) — same
                # cost-only fold as the engine look, or half the turn's vision spend is invisible.
                usage += LLMUsage(cost_usd=tool_usage.cost_usd)
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
            # render/publish is non-convergence, not a stuck edit). A non-mutating tool (a read)
            # carries no new edit information, so it does NOT reset the streak — else a read between
            # two failed edits would let a 3-strikes loop stretch indefinitely. Only a success or a
            # genuine state change (any mutating tool) resets it.
            if registry.is_edit_tool(tc.name) and not ok:
                consecutive_failed_edits += 1
            elif registry.is_edit_tool(tc.name) or registry.is_mutating(tc.name):
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

            # Render-blind spree brake (per FILE): clean edit_shader edits never trip either
            # counter above, so a model iterating on AESTHETICS can stack them unbounded with the
            # user seeing nothing. A clean edit_shader on a file increments its streak; a CLEAN
            # write_shader (the sanctioned whole-file convergence) RESETS it — finishing in one
            # write must never be the straw that trips the hard stop. (A write that applies WITH
            # errors isn't clean, so it neither counts nor resets — the turn is thrash-exempt while
            # a compile is broken anyway.) While a broken compile is in flight the turn stays exempt
            # (fixing comes first — those count toward the thrash nudge). At the soft threshold an
            # escalating fact rides every result; at the hard threshold the turn force-ends (the lone
            # soft nudge was blown past in a 16-edit spree).
            clean_edit = (
                registry.is_edit_tool(tc.name) and ok and not applied_with_errors
            )
            if clean_edit:
                key = _edit_target_key(tc.name, args)
                if tc.name in _WRITE_TOOLS:
                    clean_edits_by_file[key] = 0
                else:
                    streak = clean_edits_by_file.get(key, 0) + 1
                    clean_edits_by_file[key] = streak
                    if (
                        config.clean_edit_soft_streak > 0
                        and streak >= config.clean_edit_soft_streak
                    ):
                        msg += _clean_streak_fact(streak)
                        tr.event(
                            "clean_streak_nudge", iteration=iteration, streak=streak
                        )
                    if (
                        config.clean_edit_hard_streak > 0
                        and streak >= config.clean_edit_hard_streak
                    ):
                        clean_streak_giveup = True

            messages.append(_tool_message(tc.id, msg))

            if consecutive_failed_edits >= config.max_edit_retries:
                giveup = True
                break
            if clean_streak_giveup:
                giveup = True
                break

        if giveup:
            if clean_streak_giveup:
                logger.warning(
                    f"copilot clean-edit hard stop at {config.clean_edit_hard_streak} "
                    f"edits on one file | total_in={usage.input_tokens} "
                    f"cost=${usage.cost_usd:.6f}"
                )
                tr.event(
                    "clean_streak_giveup",
                    streak=config.clean_edit_hard_streak,
                    usage=usage,
                )
                note = (
                    f"[engine] I hit my own limit of {config.clean_edit_hard_streak} edits to "
                    "one file in a turn (NOT a pause you asked for), so I stopped to keep from "
                    "churning while you can't see the result. If more is needed, tell me to "
                    "continue and I'll finish in one rewrite."
                )
            else:
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
                    "- the edit kept not applying to the shader source. I've stopped to "
                    "avoid looping. Tell me to try again, or describe the change differently."
                )
            applied = ran.applied_mutations(registry)
            if applied:
                note += "\nWhat DID apply this turn:\n" + "\n".join(
                    f"{e.name}: {e.msg}" for e in applied
                )
                last = applied[-1]
                if (last.payload or {}).get("errors"):
                    target = _node_arg(last.args)
                    node = target or "the current node"
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

    if time_budget_hit:
        cutoff = "time_budget"
        cutoff_cause = (
            f"You reached the per-turn time budget of {config.turn_time_budget_s}s"
        )
        cutoff_note = (
            f"I hit the per-turn time budget ({config.turn_time_budget_s}s) without "
            "finishing this turn. Ask me to continue, or rephrase what you need."
        )
    else:
        cutoff = "max_iterations"
        cutoff_cause = (
            f"You reached the per-turn limit of {config.max_iterations} tool-call steps"
        )
        cutoff_note = (
            f"I stopped after {config.max_iterations} steps without finishing this "
            "turn. Ask me to continue, or rephrase what you need."
        )
    logger.warning(
        f"copilot turn hit {cutoff} | "
        f"tool_calls={total_tool_calls} total_in={usage.input_tokens} "
        f"cost=${usage.cost_usd:.6f}"
    )
    if cancel.is_set():
        yield AgentCancelled(
            _build_turn_summary(turn_reply(""), ran, registry),
            stats=TurnStats(
                context_tokens=first_input_tokens or 0,
                reply_tokens=usage.output_tokens,
                cost_usd=usage.cost_usd,
            ),
        )
        return
    yield from stream_final_reply(cutoff_cause)
    tr.event(
        "turn_done",
        cutoff=cutoff,
        iterations=config.max_iterations,
        tool_calls=total_tool_calls,
        usage=usage,
    )
    reply = final_reply_text[-1] if final_reply_text else ""
    if not reply:
        reply = cutoff_note
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
        summary=_build_turn_summary(turn_reply(reply), ran, registry), stats=stats
    )
