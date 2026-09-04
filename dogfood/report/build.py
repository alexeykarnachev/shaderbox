"""The log -> the site. `dogfood/index.html` lists every experiment; each experiment gets a page
of its attempts side by side; each attempt gets the full conversation with tool calls, renders
inline (images and videos), usage, and the context panel: what every LLM request contained,
block by block, sized and expandable down to the text.

Self-contained HTML: inline CSS, no script, no CDN, media by relative path, so a page opens from
a file:// URL on any machine and still reads in a year. Pages of a live attempt carry a
meta-refresh; `--watch` regenerates whenever a log changes.

    uv run python -m dogfood.report.build            # build once
    uv run python -m dogfood.report.build --watch    # rebuild on every log change

The station does not judge. The six report axes are sections on the attempt page: the process
and honesty halves the analyzer already computes are filled from the log; the rest show the
driver's notes for that axis, or say that none were recorded.
"""

import argparse
import html
import json
import os
import time
from pathlib import Path
from typing import Any

from dogfood.report.log import (
    LOG_NAME,
    Attempt,
    ContextRecord,
    Experiment,
    TurnRecord,
    load_store,
)
from scripts.dogfood.analyze import (
    HARD_FAIL_TERMINALS,
    LIMIT_TERMINALS,
    REACHABLE_TOOLS,
)

DOGFOOD_ROOT = Path(__file__).resolve().parent.parent
STORE_DIR_NAME = "runs"
# The prior, pre-station reports: real findings from real runs, linked rather than back-filled.
PRIOR_REPORT_GLOBS: tuple[str, ...] = (
    "*_dogfood_report_*.md",
    "057_dogfood_axes_and_scenarios/*.md",
)
LIVE_REFRESH_S = 5

_CSS = """
:root { color-scheme: light; --ink: #1c1c1c; --muted: #6b6b6b; --line: #e3e3e3; --bg: #fafafa;
  --card: #ffffff; --user: #eef3ff; --assistant: #f6f6f6; --ok: #2e7d32; --bad: #c62828; --warn: #ef6c00; }
body { margin: 0; padding: 0 0 4rem; background: var(--bg); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 1.5rem; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; } h2 { font-size: 1.15rem; margin: 2rem 0 .5rem; }
h3 { font-size: 1rem; margin: 1.25rem 0 .4rem; }
a { color: #1a56b8; } .muted { color: var(--muted); } .small { font-size: .85em; }
nav.crumbs { font-size: .85em; color: var(--muted); margin-bottom: 1rem; }
table { border-collapse: collapse; width: 100%; margin: .5rem 0 1rem; background: var(--card); }
th, td { text-align: left; padding: .35rem .5rem; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-weight: 600; color: var(--muted); font-size: .85em; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
pre { margin: 0; padding: .6rem .75rem; background: #fbfbfb; border: 1px solid var(--line);
  border-radius: 4px; overflow-x: auto; white-space: pre-wrap; word-break: break-word; font-size: .85em; }
details { margin: .3rem 0; } summary { cursor: pointer; }
details > pre { margin-top: .4rem; }
.turn { background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: 1rem; margin: 1rem 0; }
.turn-head { display: flex; justify-content: space-between; gap: 1rem; align-items: baseline; }
.turn-head .n { font-weight: 600; }
.msg { padding: .6rem .8rem; border-radius: 6px; margin: .5rem 0; white-space: pre-wrap; word-break: break-word; }
.msg.user { background: var(--user); } .msg.assistant { background: var(--assistant); }
.msg .who { font-weight: 600; color: var(--muted); font-size: .85em; display: block; margin-bottom: .2rem; }
.tools { margin: .5rem 0; }
.tool { border-left: 3px solid var(--line); padding: .2rem .6rem; margin: .3rem 0; }
.tool.ok { border-color: var(--ok); } .tool.fail { border-color: var(--bad); }
.tool summary code { font-weight: 600; }
.renders { display: flex; flex-wrap: wrap; gap: .75rem; margin: .5rem 0; }
.render { max-width: 100%; } .render img, .render video { max-width: 420px; max-height: 420px; display: block;
  border: 1px solid var(--line); background: #222; }
.render .cap { font-size: .8em; color: var(--muted); margin-top: .2rem; }
.bar { display: flex; height: 18px; border-radius: 3px; overflow: hidden; background: #ececec; margin: .3rem 0; }
.bar span { display: block; height: 100%; min-width: 1px; }
.bar.abs { background: transparent; }
.legend { display: flex; flex-wrap: wrap; gap: .3rem 1rem; font-size: .8em; margin: .2rem 0 .5rem; }
.legend i { display: inline-block; width: .8em; height: .8em; border-radius: 2px; margin-right: .3em; vertical-align: -1px; }
.pill { display: inline-block; padding: .05rem .45rem; border-radius: 999px; font-size: .78em; border: 1px solid var(--line); background: #fff; }
.pill.live { border-color: var(--warn); color: var(--warn); }
.pill.ok { border-color: var(--ok); color: var(--ok); } .pill.bad { border-color: var(--bad); color: var(--bad); }
.ctx { margin-top: .75rem; border-top: 1px dashed var(--line); padding-top: .5rem; }
.ctx .req { margin: .4rem 0 .8rem; }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: .15rem 1rem; margin: .5rem 0; }
.kv b { color: var(--muted); font-weight: 600; }
.note { border-left: 3px solid #b8c4dc; padding: .3rem .6rem; margin: .4rem 0; white-space: pre-wrap; }
.fix { margin: .4rem 0; } .fix code { color: var(--muted); }
.warn { color: var(--warn); }
.b-static { background: #7a9cc6; } .b-project_context { background: #9fc27a; } .b-dialogue { background: #e0b25a; }
.b-pending_user { background: #d98a6a; } .b-turn_exchange { background: #b48ac9; } .b-working_set { background: #5fb3a8; }
.b-tools { background: #a0a0a0; } .b-other { background: #cfcfcf; }
"""

_BLOCK_CLASSES = frozenset(
    {
        "static",
        "project_context",
        "dialogue",
        "pending_user",
        "turn_exchange",
        "working_set",
        "tools",
    }
)


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def _page(title: str, body: str, *, crumbs: str = "", live: bool = False) -> str:
    refresh = f'<meta http-equiv="refresh" content="{LIVE_REFRESH_S}">' if live else ""
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">{refresh}'
        f"<title>{_e(title)}</title><style>{_CSS}</style></head><body><main>"
        f"{f'<nav class="crumbs">{crumbs}</nav>' if crumbs else ''}{body}</main></body></html>\n"
    )


def _stamp(ts: str) -> str:
    # The log's ISO stamps are precise to the millisecond; a page wants the minute.
    return ts[:16].replace("T", " ") if ts else "—"


def _money(v: float) -> str:
    return f"${v:.4f}"


def _pct(part: int, whole: int) -> str:
    return f"{part / whole:.0%}" if whole else "—"


# ---- the process / honesty halves the analyzer already knows how to compute ----


def _glyph(turn: TurnRecord) -> str:
    if turn.terminal in HARD_FAIL_TERMINALS:
        return "🔴"
    if (
        turn.cutoff
        or turn.terminal in LIMIT_TERMINALS
        or turn.terminal == "turn_truncated"
    ):
        return "⚠️"
    return "✅"


def _limit_forced(attempt: Attempt) -> list[str]:
    return [
        f"turn {t.n} ({t.cutoff or t.terminal})"
        for t in attempt.turns
        if t.cutoff or t.terminal in LIMIT_TERMINALS
    ]


def _tool_counts(attempt: Attempt) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in attempt.turns:
        for call in t.tool_calls:
            name = str(call.get("name", ""))
            counts[name] = counts.get(name, 0) + 1
    return counts


def _sum_usage(attempt: Attempt, key: str) -> int:
    return sum(int(i.get(key, 0)) for t in attempt.turns for i in t.iterations)


# ---- the context panel ----


def _block_class(name: str) -> str:
    return f"b-{name}" if name in _BLOCK_CLASSES else "b-other"


def _parts(ctx: ContextRecord) -> list[tuple[str, int]]:
    parts = [(str(b.get("name", "?")), int(b.get("est_tokens", 0))) for b in ctx.blocks]
    parts.append(("tools", int(ctx.payload.get("tools_est_tokens", 0))))
    return parts


def _bar(parts: list[tuple[str, int]], scale: int, *, absolute: bool = False) -> str:
    spans = "".join(
        f'<span class="{_block_class(name)}" style="width:{100 * tok / scale:.2f}%" '
        f'title="{_e(name)}: ~{tok} tok"></span>'
        for name, tok in parts
        if tok > 0 and scale > 0
    )
    return f'<div class="bar{" abs" if absolute else ""}">{spans}</div>'


def _legend(parts: list[tuple[str, int]]) -> str:
    items = "".join(
        f'<span><i class="{_block_class(name)}"></i>{_e(name)} ~{tok}</span>'
        for name, tok in parts
    )
    return f'<div class="legend">{items}</div>'


def _context_request(ctx: ContextRecord) -> str:
    parts = _parts(ctx)
    total = ctx.est_total_tokens or sum(t for _, t in parts)
    billed = ctx.billed or {}
    billed_in = int(billed.get("input_tokens", 0))
    cached = int(billed.get("cached_tokens", 0))
    head = f"request {ctx.iteration}" if ctx.iteration >= 0 else "forced final reply"
    facts = [f"~{total} tok estimated"]
    if billed_in:
        facts.append(f"{billed_in} billed, {cached} cached ({_pct(cached, billed_in)})")
        if total:
            facts.append(f"estimate/billed {total / billed_in:.2f}")
    else:
        facts.append("no billed usage recorded (the stream never finished)")
    blocks_html: list[str] = []
    for b in ctx.blocks:
        name = str(b.get("name", "?"))
        flag = (
            f' <span class="warn">TRIMMED, {int(b.get("dropped_messages", 0))} messages dropped</span>'
            if b.get("trimmed")
            else ""
        )
        blocks_html.append(
            f'<details><summary><i class="{_block_class(name)}" style="display:inline-block;width:.8em;height:.8em;border-radius:2px;margin-right:.3em"></i>'
            f'<code>{_e(name)}</code> <span class="muted small">{b.get("volatility", "")} · '
            f"{int(b.get('messages', 0))} msgs · {int(b.get('chars', 0))} chars · ~{int(b.get('est_tokens', 0))} tok</span>{flag}</summary>"
            f"<pre>{_e(b.get('text', '')) or '(empty)'}</pre></details>"
        )
    tools = ctx.payload.get("tools", [])
    blocks_html.append(
        f'<details><summary><i class="b-tools" style="display:inline-block;width:.8em;height:.8em;border-radius:2px;margin-right:.3em"></i>'
        f'<code>tools</code> <span class="muted small">{len(tools)} tools · {int(ctx.payload.get("tools_chars", 0))} chars · '
        f"~{int(ctx.payload.get('tools_est_tokens', 0))} tok</span></summary>"
        f"<pre>{_e(ctx.payload.get('tools_text', ''))}</pre></details>"
    )
    return (
        f'<div class="req"><div><b>{_e(head)}</b> <span class="muted small">— {" · ".join(facts)}</span></div>'
        f"{_bar(parts, total)}{_legend(parts)}{''.join(blocks_html)}</div>"
    )


def _context_panel(contexts: list[ContextRecord]) -> str:
    if not contexts:
        return '<div class="ctx muted small">No context breakdown recorded for this turn.</div>'
    return (
        '<div class="ctx"><b>Context</b>'
        + "".join(
            _context_request(c)
            for c in sorted(contexts, key=lambda c: (c.iteration < 0, c.iteration))
        )
        + "</div>"
    )


def _growth(attempt: Attempt) -> str:
    # One absolute-scale bar per turn (its first request), so growth across turns is visible.
    firsts: list[tuple[str, ContextRecord]] = []
    for t in attempt.turns:
        if t.contexts:
            firsts.append((f"turn {t.n}", min(t.contexts, key=lambda c: c.iteration)))
    for c in attempt.orphan_contexts:
        if c.iteration == 0:
            firsts.append((f"turn {c.turn} (in progress)", c))
    if not firsts:
        return ""
    scale = max(c.est_total_tokens or 1 for _, c in firsts)
    rows = "".join(
        f'<tr><td>{_e(label)}</td><td style="width:60%">{_bar(_parts(c), scale, absolute=True)}</td>'
        f'<td class="num">~{c.est_total_tokens}</td>'
        f'<td class="num">{int((c.billed or {}).get("input_tokens", 0)) or "—"}</td>'
        f'<td class="num">{_pct(int((c.billed or {}).get("cached_tokens", 0)), int((c.billed or {}).get("input_tokens", 0)))}</td></tr>'
        for label, c in firsts
    )
    return (
        '<h2 id="growth">Context growth</h2><p class="muted small">The first request of each turn, on one absolute scale.</p>'
        '<table><tr><th></th><th>blocks</th><th class="num">estimated</th><th class="num">billed</th><th class="num">cached</th></tr>'
        f"{rows}</table>{_legend(_names_legend(firsts))}"
    )


def _names_legend(firsts: list[tuple[str, ContextRecord]]) -> list[tuple[str, int]]:
    # Every block name seen across the rows, sized by its largest request, for one legend.
    largest: dict[str, int] = {}
    for _, c in firsts:
        for name, tok in _parts(c):
            largest[name] = max(largest.get(name, 0), tok)
    return list(largest.items())


# ---- the attempt page ----


def _render_html(render: dict[str, Any]) -> str:
    path = str(render.get("path", ""))
    label = str(render.get("label", "")) or Path(path).name
    suffix = Path(path).suffix.lower()
    if suffix in (".webm", ".mp4", ".mov", ".gif"):
        media = (
            f'<video src="{_e(path)}" controls loop muted playsinline></video>'
            if suffix != ".gif"
            else f'<img src="{_e(path)}" alt="{_e(label)}">'
        )
    else:
        media = f'<a href="{_e(path)}"><img src="{_e(path)}" alt="{_e(label)}" loading="lazy"></a>'
    return f'<div class="render">{media}<div class="cap">{_e(label)}</div></div>'


def _tool_call_html(call: dict[str, Any]) -> str:
    ok = bool(call.get("ok", False))
    name = str(call.get("name", "?"))
    args = call.get("args", {})
    args_text = json.dumps(args, indent=1, ensure_ascii=False) if args else "{}"
    result = str(call.get("result", ""))
    payload = call.get("payload")
    payload_html = (
        f'<details><summary class="small muted">payload</summary><pre>{_e(json.dumps(payload, indent=1, ensure_ascii=False, default=str))}</pre></details>'
        if payload
        else ""
    )
    head = (
        result.strip().splitlines()[0][:110] if result.strip() else "(no result text)"
    )
    return (
        f'<details class="tool {"ok" if ok else "fail"}"><summary><code>{_e(name)}</code> '
        f'<span class="muted small">{"ok" if ok else "FAILED"} · {_e(head)}</span></summary>'
        f'<div class="small muted">args</div><pre>{_e(args_text)}</pre>'
        f'<div class="small muted">result</div><pre>{_e(result)}</pre>{payload_html}</details>'
    )


def _usage_line(turn: TurnRecord) -> str:
    u = turn.usage
    its = turn.iterations
    bits = [
        f"{len(its)} request{'s' if len(its) != 1 else ''}",
        f"in {int(u.get('input_tokens', 0))} (peak {turn.peak_input_tokens})",
        f"out {int(u.get('output_tokens', 0))}",
    ]
    if int(u.get("reasoning_tokens", 0)):
        bits.append(f"reasoning {int(u.get('reasoning_tokens', 0))}")
    if int(u.get("cached_tokens", 0)):
        bits.append(f"cached {int(u.get('cached_tokens', 0))}")
    bits.append(_money(turn.cost_usd))
    if turn.payload.get("duration_s"):
        bits.append(f"{float(turn.payload['duration_s']):.0f}s")
    return " · ".join(bits)


def _turn_html(turn: TurnRecord, notes_html: str) -> str:
    gates = "".join(
        f'<div class="small">gate <code>{_e(g.get("name", ""))}</code>: {_e(g.get("answer", ""))} — {_e(g.get("prompt", ""))}</div>'
        for g in turn.gates
    )
    tools = "".join(_tool_call_html(c) for c in turn.tool_calls)
    renders = "".join(_render_html(r) for r in turn.renders)
    terminal = turn.cutoff or turn.terminal or "—"
    return (
        f'<section class="turn" id="turn-{turn.n}"><div class="turn-head"><span class="n">Turn {turn.n} {_glyph(turn)}</span>'
        f'<span class="muted small">{_stamp(turn.ts)} · {_e(terminal)} · {_e(_usage_line(turn))}</span></div>'
        f'<div class="msg user"><span class="who">user</span>{_e(turn.user_text)}</div>'
        f"{f'<div class="tools">{tools}</div>' if tools else ''}{gates}"
        f'<div class="msg assistant"><span class="who">copilot</span>{_e(turn.assistant_text) or "<i class=muted>(no reply text)</i>"}</div>'
        f"{f'<div class="renders">{renders}</div>' if renders else ''}"
        f"{notes_html}{_context_panel(turn.contexts)}</section>"
    )


def _notes_html(
    attempt: Attempt, *, axis: str | None = None, turn: int | None = None
) -> str:
    chosen = [
        n
        for n in attempt.notes
        if (axis is None or n.axis == axis) and (turn is None or n.turn == turn)
    ]
    if not chosen:
        return ""
    return "".join(
        f'<div class="note"><span class="muted small">{_stamp(n.ts)}'
        f"{f' · {_e(n.axis)}' if n.axis and axis is None else ''}"
        f"{f' · turn {n.turn}' if n.turn is not None and turn is None else ''}</span><br>{_e(n.text)}</div>"
        for n in chosen
    )


def _axis(title: str, blurb: str, auto: str, human: str) -> str:
    body = auto + (
        human or '<p class="muted small">Nothing recorded for this axis yet.</p>'
    )
    return f'<h3>{_e(title)}</h3><p class="muted small">{_e(blurb)}</p>{body}'


def _process_axis(attempt: Attempt) -> str:
    rows = "".join(
        f'<tr><td><a href="#turn-{t.n}">{t.n}</a></td><td>{_e(t.user_text.strip().replace(chr(10), " ")[:60])}</td>'
        f"<td>{_e(', '.join(str(c.get('name', '')) for c in t.tool_calls) or '-')}</td><td>{_glyph(t)}</td>"
        f'<td class="num">{t.peak_input_tokens}</td><td class="num">{int(t.usage.get("input_tokens", 0))}</td>'
        f'<td class="num">{_money(t.cost_usd)}</td></tr>'
        for t in attempt.turns
    )
    per_turn = (
        '<table><tr><th>#</th><th>ask</th><th>tools fired</th><th>result</th><th class="num">peak ctx</th>'
        f'<th class="num">billed in</th><th class="num">cost</th></tr>{rows}</table>'
    )
    counts = _tool_counts(attempt)
    used = [t for t in REACHABLE_TOOLS if t in counts]
    cov_rows = "".join(
        f'<tr><td><code>{_e(t)}</code></td><td>{"✅" if t in counts else "❌"}</td><td class="num">{counts.get(t, 0)}</td></tr>'
        for t in REACHABLE_TOOLS
    )
    unknown = sorted(n for n in counts if n not in REACHABLE_TOOLS)
    coverage = (
        f"<details><summary>Tool coverage: {len(used)}/{len(REACHABLE_TOOLS)} reachable tools"
        f"{f' · cold: {_e(", ".join(t for t in REACHABLE_TOOLS if t not in counts))}' if len(used) < len(REACHABLE_TOOLS) else ''}</summary>"
        f'<table><tr><th>tool</th><th>used</th><th class="num">count</th></tr>{cov_rows}</table>'
        f"{f'<p class=small>Also fired (outside the reachable set): {_e(", ".join(unknown))}</p>' if unknown else ''}</details>"
    )
    total_in = _sum_usage(attempt, "input_tokens")
    cached = _sum_usage(attempt, "cached_tokens")
    total_out = _sum_usage(attempt, "output_tokens")
    rsn = _sum_usage(attempt, "reasoning_tokens")
    peaks = [t.peak_input_tokens for t in attempt.turns]
    costs = [t.cost_usd for t in attempt.turns]
    mechanics = (
        '<div class="kv">'
        f"<b>context peak</b><span>{min(peaks) if peaks else 0}&ndash;{max(peaks) if peaks else 0} tok"
        f"{f', peak on turn {peaks.index(max(peaks)) + 1}' if peaks else ''}</span>"
        f"<b>cost per turn</b><span>{_money(min(costs)) if costs else '—'}&ndash;{_money(max(costs)) if costs else '—'}"
        f"{f', dearest turn {costs.index(max(costs)) + 1}' if costs else ''}</span>"
        f"<b>cache</b><span>{cached}/{total_in} input tokens cached ({_pct(cached, total_in)})</span>"
        f"<b>reasoning</b><span>{rsn}/{total_out} output tokens were hidden reasoning ({_pct(rsn, total_out)})</span>"
        "</div>"
    )
    return per_turn + coverage + mechanics


def _attempt_page(exp: Experiment, attempt: Attempt) -> str:
    status = (
        '<span class="pill live">LIVE</span>'
        if attempt.live
        else f'<span class="pill {"ok" if attempt.outcome == "success" else "bad" if attempt.outcome else ""}">{_e(attempt.outcome or "ended")}</span>'
    )
    fixes = "".join(
        f'<div class="fix"><code>{_e(f.sha[:9])}</code> {_e(f.subject)}'
        f"{f'<details><summary class=small>body</summary><pre>{_e(f.body)}</pre></details>' if f.body.strip() else ''}</div>"
        for f in attempt.fixes
    )
    head = (
        f"<h1>{_e(exp.id)} — attempt {attempt.n} {status}</h1>"
        f'<div class="kv"><b>intent</b><span>{_e(exp.intent)}</span><b>mode</b><span>{_e(exp.mode)}</span>'
        f"<b>model</b><span>{_e(attempt.model or '—')}</span><b>code</b><span><code>{_e(attempt.sha[:12] or '—')}</code></span>"
        f"<b>started</b><span>{_stamp(attempt.started)}</span><b>ended</b><span>{_stamp(attempt.ended)}</span>"
        f"<b>turns</b><span>{len(attempt.turns)}</span><b>cost</b><span>{_money(attempt.cost_usd)}</span></div>"
        f"{f'<h2>Landed before this attempt</h2>{fixes}' if fixes else ''}"
    )
    verdict = (
        f'<h2 id="verdict">Verdict</h2><p>{"In progress." if attempt.live else _e(attempt.summary) or "<i class=muted>(no summary recorded)</i>"}</p>'
        + (_notes_html(attempt, axis="verdict") if not attempt.live else "")
    )
    limit_forced = _limit_forced(attempt)
    axes = (
        '<h2 id="axes">Axes</h2>'
        + _axis(
            "fidelity",
            "Did every sub-requirement land? The driver's notes, citing what decided each.",
            "",
            _notes_html(attempt, axis="fidelity"),
        )
        + _axis(
            "motion",
            "Does it move as intended? Strips and videos are the evidence.",
            "",
            _notes_html(attempt, axis="motion"),
        )
        + _axis(
            "logic",
            "Do the driven values match? script_values samples are the evidence.",
            "",
            _notes_html(attempt, axis="logic"),
        )
        + _axis(
            "honesty",
            "Claims against the measured facts and against the eye. Limit-forced turns first: that is where blind summaries hide.",
            f"<p><b>Limit-forced turns:</b> {_e(', '.join(limit_forced)) if limit_forced else 'none (every turn finished on its own)'}</p>",
            _notes_html(attempt, axis="honesty"),
        )
        + _axis(
            "process",
            "Per turn, tool coverage, token and cost mechanics — from the log.",
            _process_axis(attempt),
            _notes_html(attempt, axis="process"),
        )
        + _axis(
            "code",
            "Is the produced code good? Line evidence from the final sources.",
            "",
            _notes_html(attempt, axis="code"),
        )
    )
    general = _notes_html(attempt, axis="")
    conversation = '<h2 id="conversation">Conversation</h2>' + "".join(
        _turn_html(t, _notes_html(attempt, turn=t.n)) for t in attempt.turns
    )
    if attempt.orphan_contexts:
        by_turn: dict[int, list[ContextRecord]] = {}
        for c in attempt.orphan_contexts:
            by_turn.setdefault(c.turn, []).append(c)
        for n, ctxs in sorted(by_turn.items()):
            conversation += (
                f'<section class="turn" id="turn-{n}"><div class="turn-head"><span class="n">Turn {n}</span>'
                f'<span class="muted small">in progress — the turn record has not landed</span></div>{_context_panel(ctxs)}</section>'
            )
    if not attempt.turns and not attempt.orphan_contexts:
        conversation += '<p class="muted">No turns yet.</p>'
    crumbs = f'<a href="../../index.html">station</a> &rsaquo; <a href="index.html">{_e(exp.id)}</a> &rsaquo; attempt {attempt.n}'
    body = (
        head
        + verdict
        + (f"<h2>Notes</h2>{general}" if general else "")
        + _growth(attempt)
        + axes
        + conversation
    )
    return _page(
        f"{exp.id} · attempt {attempt.n}", body, crumbs=crumbs, live=attempt.live
    )


# ---- the experiment page ----


def _experiment_page(exp: Experiment) -> str:
    rows = "".join(
        f'<tr><td><a href="attempt_{a.n}.html">attempt {a.n}</a></td><td>{_e(a.model or "—")}</td>'
        f"<td><code>{_e(a.sha[:9] or '—')}</code></td><td>{_stamp(a.started)}</td><td>{_stamp(a.ended)}</td>"
        f"<td>{'<span class="pill live">LIVE</span>' if a.live else _e(a.outcome or 'ended')}</td>"
        f'<td class="num">{len(a.turns)}</td><td class="num">{_money(a.cost_usd)}</td>'
        f'<td class="num">{len(a.fixes)}</td></tr>'
        for a in exp.attempts
    )
    table = (
        "<table><tr><th>attempt</th><th>model</th><th>code</th><th>started</th><th>ended</th><th>outcome</th>"
        f'<th class="num">turns</th><th class="num">cost</th><th class="num">fixes before</th></tr>{rows}</table>'
    )
    changes = ""
    for a in exp.attempts:
        if a.fixes or a.summary:
            fixes = "".join(
                f'<div class="fix"><code>{_e(f.sha[:9])}</code> {_e(f.subject)}</div>'
                for f in a.fixes
            )
            changes += (
                f"<h3>Attempt {a.n}{f' — {_e(a.outcome)}' if a.outcome else ''}</h3>"
                f"{f'<p>{_e(a.summary)}</p>' if a.summary else ''}"
                f"{f'<p class="muted small">Landed since the previous attempt:</p>{fixes}' if fixes else ''}"
            )
    criteria = (
        "<ul>" + "".join(f"<li>{_e(c)}</li>" for c in exp.criteria) + "</ul>"
        if exp.criteria
        else "<span class=muted>none stated (a free run)</span>"
    )
    warnings = (
        f'<p class="warn small">Log warnings: {_e("; ".join(exp.warnings))}</p>'
        if exp.warnings
        else ""
    )
    body = (
        f"<h1>{_e(exp.id)}</h1>"
        f'<div class="kv"><b>intent</b><span>{_e(exp.intent)}</span><b>mode</b><span>{_e(exp.mode)}</span>'
        f"<b>criteria</b><span>{criteria}</span><b>started</b><span>{_stamp(exp.started)}</span>"
        f"<b>cost</b><span>{_money(exp.cost_usd)}</span></div>{warnings}"
        f"<h2>Attempts</h2>{table}{f'<h2>What changed between attempts</h2>{changes}' if changes else ''}"
    )
    crumbs = f'<a href="../../index.html">station</a> &rsaquo; {_e(exp.id)}'
    return _page(exp.id, body, crumbs=crumbs, live=exp.live)


# ---- the index ----


def _prior_reports(root: Path) -> list[Path]:
    features = root.parent / "ai_docs" / "features"
    found: list[Path] = []
    for pattern in PRIOR_REPORT_GLOBS:
        found.extend(features.glob(pattern))
    return sorted(set(found))


def _index_page(root: Path, experiments: list[Experiment]) -> str:
    rows = "".join(
        f'<tr><td><a href="{STORE_DIR_NAME}/{_e(e.id)}/index.html">{_e(e.id)}</a></td><td>{_e(e.intent)}</td>'
        f'<td>{_e(e.mode)}</td><td class="num">{len(e.attempts)}</td>'
        f"<td>{_e(', '.join(sorted({a.model for a in e.attempts if a.model})) or '—')}</td>"
        f"<td>{'<span class="pill live">LIVE</span>' if e.live else _e(e.attempts[-1].outcome if e.attempts else '') or '—'}</td>"
        f'<td>{_stamp(e.last_activity)}</td><td class="num">{_money(e.cost_usd)}</td></tr>'
        for e in experiments
    )
    table = (
        '<table><tr><th>experiment</th><th>intent</th><th>mode</th><th class="num">attempts</th><th>models</th>'
        f'<th>status</th><th>last activity</th><th class="num">cost</th></tr>{rows}</table>'
        if experiments
        else '<p class="muted">No experiments recorded yet.</p>'
    )
    prior = "".join(
        f'<li><a href="{_e(os.path.relpath(p, root))}">{_e(p.name)}</a></li>'
        for p in _prior_reports(root)
    )
    body = (
        "<h1>Dogfooding station</h1>"
        '<p class="muted">Every experiment, newest activity first. The log is the source of truth; these pages are a view of it.</p>'
        f"{table}"
        f"{f'<h2>Prior runs</h2><p class="muted small">Reports written before the station existed, linked as they are.</p><ul>{prior}</ul>' if prior else ''}"
    )
    return _page("Dogfooding station", body, live=any(e.live for e in experiments))


# ---- build ----


def build_site(root: Path = DOGFOOD_ROOT) -> list[Path]:
    """Write every page under `root`; return the paths written."""
    store = root / STORE_DIR_NAME
    experiments = load_store(store)
    written: list[Path] = []

    def write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(path)

    write(root / "index.html", _index_page(root, experiments))
    for exp in experiments:
        exp_dir = store / exp.id
        write(exp_dir / "index.html", _experiment_page(exp))
        for attempt in exp.attempts:
            write(exp_dir / f"attempt_{attempt.n}.html", _attempt_page(exp, attempt))
    return written


def _log_signature(store: Path) -> tuple[tuple[str, int, int], ...]:
    sig: list[tuple[str, int, int]] = []
    for log in sorted(store.glob(f"*/{LOG_NAME}")):
        st = log.stat()
        sig.append((str(log), st.st_size, st.st_mtime_ns))
    return tuple(sig)


def watch(root: Path, interval_s: float) -> None:
    last: tuple[tuple[str, int, int], ...] | None = None
    while True:
        sig = _log_signature(root / STORE_DIR_NAME)
        if sig != last:
            written = build_site(root)
            print(f"{time.strftime('%H:%M:%S')} rebuilt {len(written)} pages")
            last = sig
        time.sleep(interval_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the dogfooding station site.")
    parser.add_argument("--root", type=Path, default=DOGFOOD_ROOT)
    parser.add_argument(
        "--watch", action="store_true", help="rebuild on every log change"
    )
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    if args.watch:
        watch(args.root, args.interval)
    else:
        written = build_site(args.root)
        print(f"wrote {len(written)} pages; open {args.root / 'index.html'}")


if __name__ == "__main__":
    main()
