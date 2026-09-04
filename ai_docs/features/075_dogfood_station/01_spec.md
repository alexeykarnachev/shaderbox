# 075 — the dogfooding station

A durable, browsable home for every dogfooding experiment: a static HTML site over an
append-only event log, showing the conversation, the renders, and — the new half — what the
copilot's context actually CONTAINED at each turn, block by block, down to the text.

This is infrastructure, not a test run. The first real experiment (a radiance-cascades build)
runs AFTER this lands, on finished tooling.

## Status

| Wave | State | Commit |
|---|---|---|
| W-0 event log + writer | not started | — |
| W-1 `context_breakdown` trace event | not started | — |
| W-2 the static site | not started | — |
| W-3 harness + skill integration | not started | — |
| W-4 sanitize | not started | — |

## Why this exists

Today a dogfood run produces: a trace file, per-turn JSON dumps, PNGs scattered in a run dir, and
a hand-written markdown report at the end. Three things are missing, and the maintainer named all
three:

1. **It is not live.** The report is written after the fact. During a long run there is no way to
   see where the run is without reading raw JSON.
2. **The context is opaque.** `analyze.py` reports token COUNTS, but nothing shows what was IN
   the context — which blocks, how big each was, what text they held, what the cache hit. The
   per-block breakdown has been deferred since feature 026.
3. **Experiments do not accumulate.** Each run's report is a separate markdown file; there is no
   index, no way to compare attempt 2 against attempt 5, no way to see that one model failed
   where another did not.

## Decisions

- **D1. Static files, not a server.** Everything asked for — a live-updating index, drilling into
  context blocks, navigating every experiment — works as static HTML reading a JSONL log. Static
  survives a crash, needs no process or port, opens from any machine, and still reads in a year.
  A server would buy live PUSH and cross-run queries; that is not worth a daemon. The live half is
  covered by a meta-refresh while a run is active. **This is reversible**: a server can be added
  later over the SAME log without redoing the site.
- **D2. The log is the source of truth; HTML is a view.** Every turn appends one JSON object to
  `events.jsonl` as it happens. The HTML is REGENERATED from the log, never edited. So a crash, a
  `/clear`, or a killed process loses nothing, and the site can be rebuilt from scratch at any
  time. This is the property that makes the whole thing robust.
- **D3. A dedicated root: `dogfood/`.** Not in `shaderbox/` (that ships to users; this does not)
  and not under `scripts/dogfood/` (that is the harness — the engine driver; this is the station —
  the record). `dogfood/report/` is the compiler; `dogfood/runs/` is the store; `dogfood/index.html`
  is the one bookmark the maintainer opens.
- **D4. An experiment has a stated INTENT, not a scenario file.** The maintainer is right that the
  discipline should emerge rather than be presumed. But the honesty axis needs something to judge a
  claim against, so each experiment records: a one-line intent, a driving MODE (`end_to_end` |
  `babysat` | `free_run`), and optional success criteria. A free run states that it has none — which
  is a legitimate value, not a gap. No scenario-file format is invented here.
- **D5. Attempts are first-class.** One experiment holds N attempts; each attempt records its model,
  its start/end, its outcome, and what changed since the previous attempt (the fixes made between
  them). Comparing attempt 2 to attempt 5 is the point.
- **D6. Fixes between attempts land as their own commits, and the log records the sha.** So an
  attempt's page can say "these three commits happened before this attempt" and link the reasoning.

## Waves

### W-0 — the event log and its writer

`dogfood/report/log.py`: an append-only JSONL writer plus the typed records. One record per event,
each carrying `experiment_id`, `attempt`, `ts` (UTC, explicit offset), `kind`, and a payload.

Record kinds, minimum: `experiment_start`, `attempt_start`, `turn` (user text, assistant text, tool
calls, usage, cost, render paths, gate), `context` (the W-1 breakdown), `fix` (a commit landed
between attempts), `attempt_end`, `note` (a driver observation mid-run).

Done-condition: the writer appends valid JSONL under concurrent-ish use (a turn process per turn,
each opening and appending), a reader reconstructs a full run from the log alone, and a test pins
that an interrupted write does not corrupt the file.

### W-1 — `context_breakdown`, the trace event deferred since 026

The heart of the debugging station. `build_prompt` composes named `PromptBlock`s; nothing records
what each contributed. Emit a trace event per LLM request carrying, per block: its name, its
volatility, its char count, its approximate token count, and its TEXT.

Notes for the implementer:
- `PromptBlock.name` already exists and is the key. (It was nearly deleted in 074 as write-only —
  see `074_nightly_sweep/00_inventory.md`; it survived precisely because the convention names a
  prompt tier "a named block". This wave is why that mattered.)
- The `tools=` block is not a `PromptBlock` but IS context — account for it separately.
- Cached-token data already flows end to end (`LLMUsage.cached_tokens`, `openrouter.py`), so the
  cache half is free; join it to the breakdown per request.
- Storing full block TEXT is the point (the maintainer asked to "fall into these blocks"), but it
  is large — the log stores it, the HTML lazy-renders it behind a disclosure.

Done-condition: for a turn, the sum of block token estimates is within a stated tolerance of the
request's billed input tokens, and a test asserts every block `build_prompt` composed appears in the
event — enumerated from the block list, not a hardcoded set of names.

### W-2 — the static site

`dogfood/report/build.py` compiles the log into:
- `dogfood/index.html` — every experiment, newest first: intent, mode, attempts, models, status,
  cost. The one bookmark.
- one page per experiment — its attempts side by side, what changed between them.
- one page per attempt — the full conversation in order, each turn showing the user message, the
  assistant reply, the tool calls with args and results, the renders INLINE, the usage, and the
  context panel.
- the context panel: a proportional bar of block sizes, each block expandable to its full text,
  the cache hit rate, and growth across turns.

Self-contained HTML: images referenced by relative path, no CDN, no build step. A `--watch` mode
regenerates on log change so an open tab is current, and pages carry a meta-refresh while their
attempt is live.

Done-condition: the site builds from a fixture log with no network; every turn in the log appears on
the attempt page; opening `index.html` from a file:// URL works with images and disclosures.

### W-3 — harness and skill integration

Make logging the harness's default rather than the driver's chore: `dump()` also appends its turn
record, the render helpers record their outputs, and an experiment/attempt is opened by one call.
Then update the `dogfood` skill so the documented flow IS this flow.

Done-condition: driving a turn writes its record with no explicit logging call in the command, and
the skill's §1 commands produce a browsable page.

### W-4 — sanitize

Roadmap banner and row, conventions if a decision here constrains future code, and the cold-context
check.

## Constraints

- **No behaviour change to the copilot engine.** W-1 adds a trace event; it must not alter what is
  SENT to the model. A test pins that the composed prompt is byte-identical with the event on and off.
- **No secrets in the log.** Run data dirs hold a live OpenRouter key in `integrations.json`; the
  station records model IDs and costs, never credentials. The log is committed; the runs store is not.
- **`make gates` green before and after each wave**, exit code read unpiped.
- **The harness's one-process-per-turn shape stays.** The station observes; it does not restructure
  how turns are driven.

## What looks wrong and is correct

- **`PromptBlock.name` has no code reader.** It is the identity of a prompt tier by convention, and
  W-1 is the consumer that makes it load-bearing. Do not "simplify" it away.
- **Storing block text duplicates the trace.** Deliberate: the trace is a debugging artifact that gets
  purged with the run dir; the station's log is durable and must stand alone.
- **The station does not judge.** No pass/fail assertions about shader quality live here. The report's
  six axes stay the human's, per the dogfood skill.

## Cold start

Read this file, then `00_progress.md` beside it, then `git log --oneline -15`. Order: W-0, W-1
(independent of W-0's writer but feeds it), W-2, W-3, W-4.

Settled as CONSTRAINTS rather than open questions:
- Static files, not a server (D1) — revisit only if cross-run queries become the bottleneck.
- The JSONL log is the source of truth; HTML is a regenerable view (D2).
- No scenario-file format is invented (D4). Intent + mode + optional criteria, nothing more.
- The first experiment runs only after this lands.
- **Model:** experiments start on a cheap model. `openai/gpt-5.1-codex-mini` is the in-tree default
  (400k ctx). `openai/gpt-5.6-luna` was verified available on 2026-09-04 — tool-capable, 1.05M ctx,
  and at $0.20/$1.20 per Mtok it is CHEAPER than codex-mini's $0.25/$2.00. Comparing models is an
  objective of the experiments, not a prerequisite; the station records the model per attempt.
