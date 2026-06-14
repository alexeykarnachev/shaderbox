"""Auto-analyze a dogfood run from its trace transcripts + dump JSONs (feature 027).

Drive a scenario by hand (the `/dogfood` skill), then point this at the run's data dir to
extract — WITHOUT reading logs by hand — tool coverage, per-turn iteration/token/cost stats,
recovery sequences, and the token-growth shape. Emits a machine-readable JSON + a paste-ready
markdown block, and (with --template/--report-out) fills the {{AUTO:...}} slots in
REPORT_TEMPLATE.md, leaving the {{HUMAN:...}} slots for the human's judgment.

A run spans several transcripts when clear_context/reload rotates the trace; this globs a data
dir's copilot_traces/ in timestamp order and reconstructs the full session.

Token note (load-bearing — do NOT conflate): the `turn_done` summary `in=` is the CUMULATIVE
billed input (sum of every iteration's input); the real per-turn context SIZE is the max
per-iteration `in=` (`peak_iter_in_tokens`). The cumulative figure is the cost driver, the peak
is the context-size driver.

Usage:
    uv run python scripts/dogfood/analyze.py scripts/dogfood/runs/data-XXXX
    uv run python scripts/dogfood/analyze.py <data_dir> --json-out a.json --md-out a.md
    uv run python scripts/dogfood/analyze.py <data_dir> \
        --template scripts/dogfood/REPORT_TEMPLATE.md --report-out ai_docs/features/NNN_*.md

Default (no flags): the markdown block to stdout, the JSON to stderr.
"""

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path

CANONICAL_TOOLS: frozenset[str] = frozenset(
    {
        "read_shader",
        "edit_shader",
        "write_shader",
        "set_uniform",
        "read_script",
        "write_script",
        "edit_script",
        "create_node",
        "delete_node",
        "switch_node",
        "grep",
        "read_lib",
        "render_image",
        "render_video",
        "publish_telegram",
        "publish_youtube",
        "set_telegram_token",
        "telegram_connect",
        "list_telegram_packs",
        "select_telegram_pack",
        "create_telegram_pack",
        "delete_telegram_pack",
        "set_youtube_credentials",
    }
)
# Tools that existed in PAST runs but not the live registry — recognized when parsing a
# historical transcript (no unknown-tool warning), never part of the coverage denominator.
HISTORICAL_TOOLS: frozenset[str] = frozenset({"replace_lines", "insert_after"})
# Coverage is measured against these — the telegram/youtube/publish set precheck-fails on the
# empty ExporterRegistry the harness builds, so it's excluded from the gap metric.
REACHABLE_TOOLS: tuple[str, ...] = (
    "read_shader",
    "edit_shader",
    "write_shader",
    "set_uniform",
    "read_script",
    "write_script",
    "edit_script",
    "create_node",
    "delete_node",
    "switch_node",
    "grep",
    "read_lib",
    "render_image",
    "render_video",
)
# Node-mutating edit tools — a broken edit recovered by ANY later clean one of these (same turn)
# counts as a compile-error recovery, even across tool names.
_EDIT_TOOLS: frozenset[str] = frozenset(
    {"edit_shader", "write_shader", "write_script", "edit_script", "create_node"}
    | HISTORICAL_TOOLS
)
DEFAULT_MODEL_NOTE = "unknown (in-tree default)"
# The trace event kinds that END a turn. A clean turn closes on turn_done; the rest are failure
# terminals that emit no turn_done. The result glyph keys on which of these closed the turn — NOT
# on whether some mid-turn attempt errored (an attempt the agent recovered from is not a failure).
_TERMINAL_KINDS: frozenset[str] = frozenset(
    {
        "turn_done",
        "edit_giveup",
        "stream_error",
        "turn_truncated",
        "model_incompatible",
    }
)
_HARD_FAIL_TERMINALS: frozenset[str] = frozenset(
    {"edit_giveup", "stream_error", "model_incompatible"}
)

_TS_RE = re.compile(r"copilot_.*_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d+)\.transcript")
_SECTION_RE = re.compile(r"^### (?P<kind>\w+)\s")
_USAGE_RE = re.compile(
    r"^usage: in=(\d+) out=(\d+)(?: rsn=(\d+))?(?: cache=(\d+))? cost=\$([\d.]+)"
)
_INVOKE_RE = re.compile(r"^\s*-> tool_call (?P<name>\w+)\(id=(?P<id>call_\w+)\)")


@dataclass
class ToolAttempt:
    n: int
    name: str
    ok: bool
    result_head: str
    applied_with_errors: bool = (
        False  # ok=True but compiled WITH errors (the _applied_result quirk)
    )


@dataclass
class Iteration:
    index: int
    finish_reason: str
    in_tokens: int
    out_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    cost_usd: float


@dataclass
class Turn:
    user_text: str
    model: str
    iterations: list[Iteration]
    tool_attempts: list[ToolAttempt]
    iterations_count: int
    tool_calls_count: int
    reply: str
    billed_in_tokens: int
    peak_iter_in_tokens: int
    dump_context_tokens: int | None
    reply_tokens: int
    cost_usd: float
    segment: str
    # The trace event kind that ENDED the turn (turn_done / edit_giveup / stream_error /
    # turn_truncated / model_incompatible / "" if none seen). A real failure glyph keys on
    # THIS, not on whether some attempt errored — an attempt the agent handled is not a
    # failed turn.
    terminal_kind: str
    gate_approvals: int
    gate_declines: int
    errored: bool
    recovered: bool


@dataclass
class Recovery:
    turn_index: int  # 1-based turn the recovery happened in
    tool: str  # the tool whose attempt broke
    fixer: str  # the (possibly different) edit tool whose later attempt compiled clean
    error_head: str
    error_attempt: int  # the broke attempt's block ordinal (n:), not the LLM iteration
    recovered_attempt: int


@dataclass
class RunAnalysis:
    data_dir: str
    model: str
    segments: list[str]
    turns: list[Turn]
    invocations: dict[str, str]
    tool_counts: dict[str, int]
    reachable_used: list[str]
    reachable_gap: list[str]
    coverage: str
    recoveries: list[Recovery]
    errored_attempts: list[ToolAttempt]
    recovery_summary: str
    total_cost_usd: float
    total_billed_in_tokens: int
    total_out_tokens: int
    peak_iter_in_tokens: int
    peak_iter_turn_index: int
    dump_session_cost_usd: float | None
    last_render_path: str | None
    growth_shape: str
    warnings: list[str] = field(default_factory=list)


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


def _as_int(value: object) -> int:
    if isinstance(value, int):
        return value
    s = str(value).strip() if isinstance(value, str) else ""
    return int(float(s)) if s else 0


def _as_float(value: object) -> float:
    return (
        float(value)
        if isinstance(value, int | float | str) and str(value).strip()
        else 0.0
    )


def _as_str(value: object) -> str:
    return str(value) if value is not None else ""


def _collect_transcripts(target: Path) -> list[Path]:
    if target.is_dir():
        files = list((target / "copilot_traces").glob("*.transcript"))
    elif target.suffix == ".transcript":
        files = [target]
    else:
        # a glob string handed in as the positional
        files = [Path(p) for p in sorted(Path().glob(str(target)))]

    def _key(p: Path) -> str:
        m = _TS_RE.search(p.name)
        return m.group(1) if m else f"~{p.stat().st_mtime:020.6f}"

    return sorted(files, key=_key)


def _parse_transcript(path: Path, turns: list[Turn], warnings: list[str]) -> None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    kind = ""
    cur: dict[str, object] = {}
    cur_turn: Turn | None = None

    def _finalize_turn(t: Turn | None) -> None:
        # An error terminal (giveup / stream_error / truncated / model_incompatible) never emits a
        # turn_done, so the turn_done-keyed cost/token fields stay at their 0 defaults and the rollup
        # understates spend. Fall back to summing the per-iteration usage the turn already parsed.
        if t is None or t.cost_usd != 0.0 or not t.iterations:
            return
        t.cost_usd = sum(it.cost_usd for it in t.iterations)
        t.reply_tokens = sum(it.out_tokens for it in t.iterations)
        t.billed_in_tokens = sum(it.in_tokens for it in t.iterations)
        t.peak_iter_in_tokens = max(it.in_tokens for it in t.iterations)

    def _close_section() -> None:
        nonlocal cur_turn
        if kind == "turn_start":
            _finalize_turn(cur_turn)
            cur_turn = Turn(
                user_text=_as_str(cur.get("user_text", "")),
                model=_as_str(cur.get("model", "")),
                iterations=[],
                tool_attempts=[],
                iterations_count=0,
                tool_calls_count=0,
                reply="",
                billed_in_tokens=0,
                peak_iter_in_tokens=0,
                dump_context_tokens=None,
                reply_tokens=0,
                cost_usd=0.0,
                segment=path.name,
                terminal_kind="",
                gate_approvals=0,
                gate_declines=0,
                errored=False,
                recovered=False,
            )
            turns.append(cur_turn)
        elif kind == "llm_response" and cur_turn is not None:
            cur_turn.iterations.append(
                Iteration(
                    index=_as_int(cur.get("iteration")),
                    finish_reason=_as_str(cur.get("finish_reason", "")),
                    in_tokens=_as_int(cur.get("in")),
                    out_tokens=_as_int(cur.get("out")),
                    cached_tokens=_as_int(cur.get("cache")),
                    reasoning_tokens=_as_int(cur.get("rsn")),
                    cost_usd=_as_float(cur.get("cost")),
                )
            )
        elif kind == "tool_call" and cur_turn is not None:
            head = _as_str(cur.get("result_head", ""))
            cur_turn.tool_attempts.append(
                ToolAttempt(
                    n=_as_int(cur.get("n")),
                    name=_as_str(cur.get("name", "")),
                    ok=_as_str(cur.get("ok", "")).strip().lower() == "true",
                    result_head=head,
                    applied_with_errors="compiled with errors" in head.lower(),
                )
            )
        elif kind == "gate_approved" and cur_turn is not None:
            cur_turn.gate_approvals += 1
        elif kind == "gate_declined" and cur_turn is not None:
            cur_turn.gate_declines += 1
        elif kind in _TERMINAL_KINDS and cur_turn is not None:
            cur_turn.terminal_kind = kind
        if kind == "turn_done" and cur_turn is not None:
            cur_turn.iterations_count = _as_int(cur.get("iterations"))
            cur_turn.tool_calls_count = _as_int(cur.get("tool_calls"))
            cur_turn.reply = _as_str(cur.get("reply", ""))
            cur_turn.billed_in_tokens = _as_int(cur.get("in"))
            cur_turn.reply_tokens = _as_int(cur.get("out"))
            cur_turn.cost_usd = _as_float(cur.get("cost"))
            cur_turn.peak_iter_in_tokens = max(
                (it.in_tokens for it in cur_turn.iterations), default=0
            )

    pending_key = ""  # a `key:` whose value continues on indented lines (result:)
    for line in lines:
        m = _SECTION_RE.match(line)
        if m is not None:
            _close_section()
            kind = m.group("kind")
            cur = {}
            pending_key = ""
            continue
        usage = _USAGE_RE.match(line)
        if usage is not None:
            cur["in"], cur["out"], cur["rsn"], cur["cache"], cur["cost"] = (
                int(usage.group(1)),
                int(usage.group(2)),
                int(usage.group(3) or 0),
                int(usage.group(4) or 0),
                float(usage.group(5)),
            )
            continue
        if line.startswith("result:"):
            cur["result_head"] = _first_line(line[len("result:") :])
            pending_key = "result"
            continue
        if (
            pending_key == "result"
            and line.startswith("    ")
            and not cur.get("result_head")
        ):
            cur["result_head"] = line.strip()
            continue
        # simple `key: value` field lines (user_text / iteration / finish_reason / n / name / ok /
        # iterations / tool_calls / reply)
        if ":" in line and not line.startswith(" "):
            key, _, val = line.partition(":")
            key = key.strip()
            if key in {
                "user_text",
                "model",
                "iteration",
                "finish_reason",
                "n",
                "name",
                "ok",
                "iterations",
                "tool_calls",
                "reply",
            }:
                cur[key] = val.strip()
                pending_key = ""
    _close_section()
    _finalize_turn(cur_turn)


def _scan_invocations(transcripts: list[Path]) -> dict[str, str]:
    invocations: dict[str, str] = {}
    for path in transcripts:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _INVOKE_RE.match(line)
            if m is not None and m.group("id") not in invocations:
                invocations[m.group("id")] = m.group("name")
    return invocations


def _count_tool_blocks(turns: list[Turn]) -> int:
    return sum(len(t.tool_attempts) for t in turns)


def _count_precheck_handoffs(transcripts: list[Path]) -> int:
    # A precheck handoff (no creds / no pack) emits its own ### section and never executes, so it
    # echoes in history without an execution block — counted here to explain that echo/block gap.
    n = 0
    for path in transcripts:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _SECTION_RE.match(line)
            if m is not None and m.group("kind") == "tool_precheck_handoff":
                n += 1
    return n


def _find_recoveries(turns: list[Turn]) -> tuple[list[Recovery], list[ToolAttempt]]:
    recoveries: list[Recovery] = []
    errored: list[ToolAttempt] = []

    # "broke" = a hard failure (ok=False) OR an edit that applied but compiled WITH errors (the
    # _applied_result quirk: ok=True yet broken — the todo.md broken-compile-thrash class). "clean" =
    # ok and not applied-with-errors. A broke edit followed (same turn) by ANY later clean
    # node-mutating edit is a recovery — the agent commonly self-corrects with a DIFFERENT edit tool
    # (create_node-broke -> replace_lines-clean), so pairing on the same tool name undercounts.
    def _broke(a: ToolAttempt) -> bool:
        return (not a.ok) or a.applied_with_errors

    def _clean_edit(a: ToolAttempt) -> bool:
        return a.ok and not a.applied_with_errors and a.name in _EDIT_TOOLS

    for turn_index, turn in enumerate(turns, 1):
        for i, att in enumerate(turn.tool_attempts):
            if not _broke(att):
                continue
            errored.append(att)
            fixer = next(
                (a for a in turn.tool_attempts[i + 1 :] if _clean_edit(a)),
                None,
            )
            if fixer is not None:
                recoveries.append(
                    Recovery(
                        turn_index=turn_index,
                        tool=att.name,
                        fixer=fixer.name,
                        error_head=att.result_head,
                        error_attempt=att.n,
                        recovered_attempt=fixer.n,
                    )
                )
                turn.recovered = True
            turn.errored = True
    return recoveries, errored


def _resolve_model(
    target: Path, cli_model: str, turns: list[Turn], warnings: list[str]
) -> str:
    if cli_model:
        return cli_model
    # The model the API actually resolved is recorded per turn at turn_start (the authoritative source
    # on a wiped/multi-process run, where integrations.json may not reflect what was billed). Take the
    # last non-empty/non-placeholder one; a turn from before this trace field shows "".
    traced = next(
        (t.model for t in reversed(turns) if t.model and t.model != "(unset)"), ""
    )
    if traced:
        return traced
    integ = (target / "integrations.json") if target.is_dir() else None
    if integ is not None and integ.exists():
        try:
            data = json.loads(integ.read_text(encoding="utf-8"))
            model = data.get("copilot", {}).get("model", "")
            if model:
                return str(model)
        except (OSError, json.JSONDecodeError):
            pass
    warnings.append(
        "model not in trace/integrations.json/CLI; using in-tree-default note"
    )
    return DEFAULT_MODEL_NOTE


def _load_dumps(runs_dir: Path, data_dir: Path) -> list[dict[str, object]]:
    # The runs/ dir is flat — every run's dumps share it. Keep only THIS run's dumps (matched by their
    # own `data_dir` field), or analyzing run A reports run B's render/cost (they cross-contaminate).
    want = data_dir.resolve()
    dumps: list[dict[str, object]] = []
    for p in sorted(runs_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not (
            isinstance(data, dict)
            and "last_turn" in data
            and "session_cost_usd" in data
        ):
            continue
        dd = str(data.get("data_dir", ""))
        if dd and Path(dd).resolve() != want:
            continue
        dumps.append(data)
    return dumps


def _growth_shape(turns: list[Turn]) -> str:
    # Peak context (max per-iteration in=) per turn. A context wipe does NOT shrink the peak (the system
    # prompt + working-set dominate), so don't infer a wipe from a peak drop — the wipe shows in the dump
    # session_cost reset (the cross-check series), not here.
    peaks = [t.peak_iter_in_tokens for t in turns]
    if not peaks:
        return "no turns"
    mx = max(peaks)
    k = peaks.index(mx) + 1
    return f"peak context {mx} tok at turn {k}; series {' -> '.join(str(p) for p in peaks)}"


def analyze(target: Path, cli_model: str) -> RunAnalysis:
    warnings: list[str] = []
    transcripts = _collect_transcripts(target)
    if not transcripts:
        raise SystemExit(f"no transcripts found under {target}")

    turns: list[Turn] = []
    for path in transcripts:
        _parse_transcript(path, turns, warnings)

    # Coverage keys on EXECUTED tool_call blocks (emitted only after execute() ran), not on the
    # history-echo id scan: an echo fires for a call the model EMITTED, which includes a gate-declined
    # or precheck-handoff call that never executed (mega run: a declined call inflated coverage). The
    # echo scan is kept as a cross-check below.
    tool_counts: dict[str, int] = dict(
        Counter(a.name for t in turns for a in t.tool_attempts)
    )
    invocations = _scan_invocations(transcripts)
    block_count = _count_tool_blocks(turns)
    deduped = len(invocations)
    # echo vs execution reconcile: an echoed id with NO execution block is a call the model emitted
    # that never reached execute — a gate decline or a precheck handoff (both recorded on the turn) —
    # OR the un-echoed-final-call case (a max_iterations turn ends ON a tool call, so its echo lands in
    # no later iteration: that goes the OTHER way, blocks > echoes). Account for the known causes; warn
    # only on an unexplained residual.
    declined = sum(t.gate_declines for t in turns)
    handoffs = _count_precheck_handoffs(transcripts)
    explained_gap = declined + handoffs
    if deduped > block_count + explained_gap:
        warnings.append(
            f"invocation/block mismatch: {deduped} echoed ids vs {block_count} execution "
            f"blocks + {explained_gap} declined/handoff (unexplained echo without execution — "
            "format drift?)"
        )
    for name in tool_counts:
        if name not in CANONICAL_TOOLS | HISTORICAL_TOOLS:
            warnings.append(f"unknown tool executed: {name}")

    reachable_used = [t for t in REACHABLE_TOOLS if t in tool_counts]
    reachable_gap = [t for t in REACHABLE_TOOLS if t not in tool_counts]
    recoveries, errored = _find_recoveries(turns)

    runs_dir = target.parent
    dumps = _load_dumps(runs_dir, target) if runs_dir.is_dir() else []
    dump_costs = [_as_float(d["session_cost_usd"]) for d in dumps]
    dump_session_cost = max(dump_costs, default=None)
    # session_cost_usd accumulates WITHIN one ProjectSession process and restarts each new process (and
    # again on a clear_context wipe), so on a one-process-per-turn / wiped run the per-dump series resets
    # and no single dump equals the whole-run total. Only when the series is monotonic AND covers every
    # turn is it a single cumulative figure the trace total should match — otherwise it's a known subset.
    dumps_are_full_series = len(dumps) == len(turns) and all(
        b >= a for a, b in pairwise(dump_costs)
    )
    last_render = ""
    for d in dumps:
        rp = str(d.get("last_render_path", "") or "")
        if rp:
            last_render = rp

    total_cost = sum(t.cost_usd for t in turns)
    peaks = [t.peak_iter_in_tokens for t in turns]
    peak = max(peaks, default=0)
    peak_idx = (peaks.index(peak) + 1) if peaks else 0

    # Two distinct failure notions, kept separate so the report doesn't conflate them:
    #  - unrecovered_attempts: an attempt broke and no later clean edit fixed it IN THE SAME TURN.
    #    This is NOT a failed turn — it includes a deliberate bad-id/bad-range probe the agent
    #    correctly answered without re-editing (which ends on a clean turn_done, so glyph ✅).
    #  - failed_turns: the turn ended on a hard-fail terminal (giveup / stream error / incompatible).
    unrecovered_attempts = sum(1 for t in turns if t.errored and not t.recovered)
    failed_turns = sum(1 for t in turns if t.terminal_kind in _HARD_FAIL_TERMINALS)
    glerr = sum(1 for a in errored if "GLError 1282" in a.result_head)
    recovery_summary = (
        f"{len(recoveries)} compile-error recoveries; {failed_turns} failed turns; "
        f"{unrecovered_attempts} unrecovered-in-turn attempts; {glerr} GLError 1282; "
        f"{len(errored)} total failed attempts"
    )

    if (
        dumps_are_full_series
        and dump_session_cost is not None
        and total_cost > 0
        and abs(dump_session_cost - total_cost) / total_cost > 0.01
    ):
        warnings.append(
            f"dump session_cost ${dump_session_cost:.4f} != trace total ${total_cost:.4f} "
            f"(trace authoritative; dumps may be a subset)"
        )

    return RunAnalysis(
        data_dir=str(target),
        model=_resolve_model(target, cli_model, turns, warnings),
        segments=[p.name for p in transcripts],
        turns=turns,
        invocations=invocations,
        tool_counts=tool_counts,
        reachable_used=reachable_used,
        reachable_gap=reachable_gap,
        coverage=f"{len(reachable_used)}/{len(REACHABLE_TOOLS)} reachable tools",
        recoveries=recoveries,
        errored_attempts=errored,
        recovery_summary=recovery_summary,
        total_cost_usd=round(total_cost, 6),
        total_billed_in_tokens=sum(t.billed_in_tokens for t in turns),
        total_out_tokens=sum(t.reply_tokens for t in turns),
        peak_iter_in_tokens=peak,
        peak_iter_turn_index=peak_idx,
        dump_session_cost_usd=dump_session_cost,
        last_render_path=last_render or None,
        growth_shape=_growth_shape(turns),
        warnings=warnings,
    )


def _result_glyph(turn: Turn) -> str:
    # 🔴 = the TURN failed (a hard-fail terminal: giveup / stream error / model-incompatible). NOT a
    # mid-turn attempt error the agent handled — a turn that deliberately probes a bad-id/bad-range
    # path and ends on a clean turn_done is a PASS, not a failure. ⚠️ = recovered or degraded
    # (recovered-from-error, or truncated-but-replied).
    if turn.terminal_kind in _HARD_FAIL_TERMINALS:
        return "🔴"
    if turn.recovered or turn.terminal_kind == "turn_truncated":
        return "⚠️"
    return "✅"


def _per_turn_table(turns: list[Turn]) -> str:
    rows = [
        "| # | Ask (head) | tools fired | result | peak ctx | billed in | cost |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, t in enumerate(turns, 1):
        fired = ", ".join(a.name for a in t.tool_attempts) or "-"
        ask = t.user_text.strip().replace("\n", " ")[:48]
        rows.append(
            f"| {i} | {ask} | {fired} | {_result_glyph(t)} | "
            f"{t.peak_iter_in_tokens} | {t.billed_in_tokens} | ${t.cost_usd:.4f} |"
        )
    return "\n".join(rows)


def _coverage_table(an: RunAnalysis) -> str:
    rows = ["| Tool | Used | Count |", "|---|---|---|"]
    for tool in REACHABLE_TOOLS:
        used = tool in an.tool_counts
        rows.append(
            f"| {tool} | {'✅' if used else '❌'} | {an.tool_counts.get(tool, 0)} |"
        )
    rows.append(f"\n**Coverage: {an.coverage}**")
    return "\n".join(rows)


def _render_list(dumps_render: str | None) -> str:
    if not dumps_render:
        return "(no copilot-side renders captured)"
    return f"- `{dumps_render}` (open with Read)"


def _cache_share_line(an: RunAnalysis) -> str:
    # Provider-cached share of all billed input — the prompt-cache hit-rate. Old traces
    # (no cache= field) show 0; the line says so instead of implying a cold cache.
    total_in = sum(it.in_tokens for t in an.turns for it in t.iterations)
    cached = sum(it.cached_tokens for t in an.turns for it in t.iterations)
    if total_in == 0:
        return "Cache: no usage data."
    if cached == 0:
        return "Cache: 0 cached tokens recorded (pre-cache-telemetry trace, or cold cache)."
    return f"Cache: {cached}/{total_in} input tokens cached ({cached / total_in:.0%})."


def _reasoning_share_line(an: RunAnalysis) -> str:
    # Reasoning (hidden CoT) share of billed OUTPUT — a reasoning model bills thinking into the output
    # budget, so a turn's `out` can be ~all rsn with little visible reply. Old traces (no rsn=) read 0.
    total_out = sum(it.out_tokens for t in an.turns for it in t.iterations)
    rsn = sum(it.reasoning_tokens for t in an.turns for it in t.iterations)
    if total_out == 0:
        return "Reasoning: no usage data."
    if rsn == 0:
        return "Reasoning: 0 reasoning tokens recorded (non-reasoning model, or pre-rsn-telemetry trace)."
    return f"Reasoning: {rsn}/{total_out} output tokens were hidden reasoning ({rsn / total_out:.0%})."


def _markdown_block(an: RunAnalysis) -> str:
    cost_vals = [t.cost_usd for t in an.turns]
    cmin, cmax = (min(cost_vals), max(cost_vals)) if cost_vals else (0.0, 0.0)
    dearest = (cost_vals.index(cmax) + 1) if cost_vals else 0
    peaks = [t.peak_iter_in_tokens for t in an.turns]
    pmin, pmax = (min(peaks), max(peaks)) if peaks else (0, 0)
    rec_lines = "\n".join(
        f"- turn {r.turn_index}: {r.tool} broke at step {r.error_attempt} "
        f"(`{r.error_head[:60]}`) -> {r.fixer} clean at step {r.recovered_attempt}."
        for r in an.recoveries
    )
    warn = "\n**Warnings:** " + "; ".join(an.warnings) if an.warnings else ""
    return f"""### Tool coverage  ({an.coverage})
{_coverage_table(an)}

**Gap (reachable, never used):** {", ".join(an.reachable_gap) or "none"}

### Turn / iteration stats
{_per_turn_table(an.turns)}

**Recoveries:**
{rec_lines or "(none)"}

### Token growth
Per-turn peak context: {pmin}-{pmax}. {an.growth_shape}.
(Per-turn billed input is the CUMULATIVE sum of iterations; total billed in = {an.total_billed_in_tokens} tok.)

### Cost
Total **${an.total_cost_usd:.4f}** ; per-turn ${cmin:.4f}-${cmax:.4f} ; dearest turn {dearest}.
{_cache_share_line(an)}
{_reasoning_share_line(an)}
Dump session_cost cross-check: {f"${an.dump_session_cost_usd:.4f}" if an.dump_session_cost_usd is not None else "n/a"} (trace authoritative).
Recovery summary: {an.recovery_summary}.{warn}"""


def _auto_fields(an: RunAnalysis) -> dict[str, str]:
    cost_vals = [t.cost_usd for t in an.turns]
    cmin, cmax = (min(cost_vals), max(cost_vals)) if cost_vals else (0.0, 0.0)
    dearest = (cost_vals.index(cmax) + 1) if cost_vals else 0
    peaks = [t.peak_iter_in_tokens for t in an.turns]
    pmin, pmax = (min(peaks), max(peaks)) if peaks else (0, 0)
    model_short = an.model.split("/")[-1]
    run_id = Path(an.data_dir).name
    date = ""
    if an.segments:
        m = _TS_RE.search(an.segments[0])
        date = m.group(1)[:10] if m else ""
    return {
        "run_label": f"{run_id} — {model_short}",
        "run_id": run_id,
        "date": date,
        "scenario_list": "(unspecified)",
        "model": an.model,
        "turn_count": str(len(an.turns)),
        "total_cost_usd": f"${an.total_cost_usd:.4f}",
        "per_turn_table": _per_turn_table(an.turns),
        "render_list": _render_list(an.last_render_path),
        "tool_coverage_table": _coverage_table(an),
        "cold_tools": ", ".join(an.reachable_gap) or "none",
        "ctx_token_range": f"{pmin}-{pmax}",
        "peak_ctx_turn": str(an.peak_iter_turn_index),
        "cost_range": f"${cmin:.4f}-${cmax:.4f}",
        "dearest_turn": str(dearest),
        "token_growth_note": an.growth_shape,
        "recovery_summary": an.recovery_summary,
    }


def _fill_template(template: str, auto: dict[str, str]) -> str:
    out = template
    for key, val in auto.items():
        out = out.replace(f"{{{{AUTO:{key}}}}}", val)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", help="a run data dir, a .transcript, or a quoted glob")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--md-out", default="")
    ap.add_argument("--template", default="")
    ap.add_argument("--report-out", default="")
    ap.add_argument("--model", default="")
    args = ap.parse_args()

    an = analyze(Path(args.target), args.model)
    auto = _auto_fields(an)
    payload = asdict(an)
    payload["auto_fields"] = auto
    payload["per_turn"] = [
        {
            "index": i + 1,
            "user_text_head": t.user_text.strip().replace("\n", " ")[:60],
            "iterations": t.iterations_count,
            "tool_calls": t.tool_calls_count,
            "billed_in_tokens": t.billed_in_tokens,
            "peak_iter_in_tokens": t.peak_iter_in_tokens,
            "cost_usd": t.cost_usd,
            "errored": t.errored,
            "recovered": t.recovered,
            "result_glyph": _result_glyph(t),
        }
        for i, t in enumerate(an.turns)
    ]
    json_str = json.dumps(payload, indent=2, ensure_ascii=False)
    md_str = _markdown_block(an)

    if args.template and args.report_out:
        filled = _fill_template(Path(args.template).read_text(encoding="utf-8"), auto)
        Path(args.report_out).write_text(filled, encoding="utf-8")
        print(f"filled report -> {args.report_out}", file=sys.stderr)
    elif args.template:
        print(_fill_template(Path(args.template).read_text(encoding="utf-8"), auto))

    if args.json_out:
        Path(args.json_out).write_text(json_str, encoding="utf-8")
    else:
        print(json_str, file=sys.stderr)
    if args.md_out:
        Path(args.md_out).write_text(md_str, encoding="utf-8")
    elif not args.template:
        print(md_str)


if __name__ == "__main__":
    main()
