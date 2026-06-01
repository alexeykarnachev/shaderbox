# 021 — logging_refactor

A clean, observable logging system across the **whole** app, plus a refined full-fidelity **agentic
trace** for debugging the copilot. Two parallel streams with distinct jobs:

1. **App logging** — three-level, three-destination. A human-readable CONSOLE stream (INFO+, terse,
   never spammy) + a rotated FILE stream (DEBUG+, the same lines plus all the lifecycle noise). One
   central setup, all 24 logger-using modules re-leveled so the console/file split actually holds.
2. **Agentic trace** — the copilot's full-fidelity transcript (every prompt incl. system, every wire
   message, every tool call WITH ALL ARGS, every result, per-iteration + per-turn tokens/cost),
   written to its own per-session file. Format optimized for an **AI agent reading it back cold** to
   debug the copilot flow token-efficiently. Configurable file-count retention. Application-internal
   config (NOT in Settings).

Today: one loguru sink added ad-hoc in `ui.py::main` (file only, default level, no console tuning),
console gets loguru's default stderr sink (everything incl. DEBUG), 118 logger calls across 24 files
with no DEBUG discipline (almost all INFO/WARNING), and a sketch JSONL trace (`copilot/trace.py`) that
reads terribly cold. See `ai_docs/roadmap.md` Active context for the wider state.

---

## Goal

### App logging — three streams
- **CONSOLE** (stderr): INFO and above. Terse, human-readable, NOT spammy. Only genuine high-level
  events: copilot turn start/done (with cost+tokens), message sent, tool called by name, project
  loaded, node saved/deleted, lib-file CRUD, image/video exported, exporter connect/upload — plus all
  WARNING/ERROR/EXCEPTION.
- **FILE** (`app_data_dir()/logs/`): DEBUG and above (i.e. everything the console gets, PLUS the
  lifecycle/diagnostic detail). Rotated by size, bounded retention, human-readable, same format as
  console.
- **Level discipline:** worker-thread lifecycle, file-watcher reloads, queue mechanics, per-iteration
  LLM internals, per-uniform render detail, bootstrap dir creation → **DEBUG, file-only**. High-level
  user actions → **INFO, console**. WARNING/ERROR → file-only by default; only app-level crash goes to
  console (it must — the app is dying). All re-leveling per the audit table in **Design decision 7**.
- **One central setup.** `configure_logging()` called once at startup (replaces the inline
  `logger.add` in `ui.py::main` and `scripts/smoke.py`'s implicit default). No module calls
  `logger.add`/`logger.remove`/sets handlers. Modules keep doing `from loguru import logger;
  logger.info(...)` — loguru is a global singleton, so only the *sinks* are centralized, not the call
  sites (no `get_logger(__name__)` wrapper — that's ceremony loguru doesn't need; decision 2).

### Agentic trace — full fidelity, agent-readable
- **Capture everything** the copilot turn touches: the system prompt, the full message list per LLM
  request, the wire-serialized request (what actually goes over HTTP), every LLM response (text +
  tool-calls + finish_reason + usage), every tool call with **complete arguments** + **full result** +
  payload, args-parse failures, tool exceptions, per-iteration cost, the turn summary. Close the
  fidelity gaps the survey found (decision 5).
- **Format: human/agent-readable transcript, NOT JSONL.** A structured plain-text transcript with real
  newlines (a shader renders as a shader, not a `\n`-escaped blob), turn/iteration section headers, and
  indented blocks. Optimized for `Read`-ing cold: dense per token, greppable, turn boundaries visible.
  Append-per-event, flushed per write (a crash loses nothing). Decision 4.
- **Configurable file retention.** Keep the last N trace files (default N=20); prune older on session
  start. Application-internal config (decision 1) — NOT a Settings field (matches `CopilotConfig`).

## Out of scope (each with a trigger)
- **Per-module log-level overrides / a logger registry.** Not needed — loguru's global singleton +
  one central `configure_logging()` covers it. **Trigger:** a module genuinely needs a different format
  or its own sink (none does today).
- **Surfacing log level / trace retention in the Settings UI.** Deliberately internal (decision 1).
  **Trigger:** a user asks to change verbosity without editing code, OR a support workflow needs
  end-users to bump verbosity.
- **Structured/JSON app logs for machine ingestion.** The app log is for humans; the trace is the
  machine-fidelity stream. **Trigger:** the app ever ships telemetry/log-shipping.
- **`SHADERBOX_LOG_LEVEL` env override for console verbosity.** Tempting and ~3 lines, but no current
  need. **Trigger:** first time debugging needs console DEBUG without a code edit. (Noted so it's a
  known easy add, not a redesign.)
- **Re-leveling `scripts/smoke.py`'s own pass/fail prints.** Those stay INFO+console (dev-facing test
  output) — in scope only insofar as smoke must call `configure_logging()` so its run matches the app.
- **Chat persistence / restore-on-reopen.** The restorable per-project conversation is **feature 022**,
  not 021. 021 locks the on-disk cut (decision 9) that 022 builds on, but the conversation schema,
  save-on-turn, load-on-open, and archive UX live in 022's spec. **Trigger:** 022.

## Design decisions
*(numbered, lock-in only; open questions are separate below)*

1. **Logging config is application-internal, mirroring `copilot/config.py`.** A frozen `LoggingConfig`
   dataclass + a `LOGGING_CONFIG` singleton, holding: `console_level` (INFO), `file_level` (DEBUG),
   `rotation` (size string, e.g. `"5 MB"`), `retention` (app-log file count), `trace_retention` (agentic
   trace file count, default 20). Not on `UIAppState` (no migration discipline needed — constants).

2. **No `get_logger(__name__)` wrapper.** loguru's `logger` is a process-global singleton; centralizing
   means configuring *sinks* once, not gatekeeping *access*. Call sites keep `from loguru import logger`.
   The only rule: nobody but `configure_logging()` touches `logger.add`/`.remove`/handlers. (The survey
   suggested a `get_logger` gatekeeper; rejected as ceremony that fights loguru's design — recorded as a
   `conventions.md ## Design decisions` rule with a revisit trigger instead.)

3. **Module layout.**
   - New `shaderbox/logging_setup.py`: `LoggingConfig` + `LOGGING_CONFIG` + `configure_logging()`
     (removes the default stderr sink, adds the leveled console sink + the rotated file sink).
   - `paths.py` gains `log_dir() -> Path` (parallel to `shader_lib_root()`), the single home for the log
     directory path. `configure_logging()` and any crash-path message read it from here.
   - The agentic trace stays in `shaderbox/copilot/trace.py` but is rewritten (format + fidelity +
     retention). Its directory path (`app_data_dir()/copilot_traces/`, decision 9) routes through a
     `paths.py` helper for consistency (`copilot_trace_dir()`), so all on-disk roots live in one module.
     The trace filename is project-stamped (`copilot_<project-slug>_<timestamp>.transcript`) — the slug
     derives from the active project dir's name; resolve the exact slug source in impl.

4. **Trace format = a plain-text transcript, append-per-event.** Each event is a delimited section with
   a header line (`▼ TURN 1 · llm_request · iter 0 · 11:03:06.779`) followed by indented, real-newline
   bodies for multi-line payloads (system prompt, message list, shader source, tool args/results). Turn
   boundaries are visible banners. Rationale: the trace's ONLY consumer is an agent reading it cold to
   debug; JSONL re-pays repeated keys per line and escapes every newline (a 50-line shader → one
   unreadable line). A transcript is denser per token and directly readable. (JSONL's machine-parse
   benefit is irrelevant — nothing parses the trace programmatically; it's read, not ingested.)

5. **Trace fidelity — close the gaps the survey found.** Resolved as built (the transcript format
   changed which gaps were real gaps vs rendering artifacts):
   - (a) **System prompt** — NOT a separate event. It is already `messages[0]` of every `llm_request`,
     and the transcript now renders the full message list (real newlines), so the system prompt is
     visible in full. The old JSONL merely buried it; the format fix surfaces it. No redundant event.
   - (b) **Wire-serialized request** — NOT traced separately. The `llm_request` event already records
     the full semantic request (every `LLMMessage`, every `LLMToolSpec`, in full). The wire form is a
     deterministic 1:1 transform (`_to_wire_message`: null-content coercion + tool-call nesting) visible
     in `openrouter.py` source. Threading a trace handle through the provider-neutral `LLMClient`
     Protocol would pollute that seam (it deliberately knows nothing of tracing) — not worth it for a
     mechanical reshape. SKIPPED to keep the seam clean.
   - (c) **Args-parse failures** — ADDED. New `tool_args_parse_error` event with the raw malformed
     arguments string (`agent.py`).
   - (d) **Tool exceptions** — already captured via `registry.execute`'s `logger.exception` (lands in
     the app log file at DEBUG+, traceback included). The `tool_call` trace event records `ok=False` +
     the safe result. So a debugger cross-references trace (turn flow) + log file (the traceback) — the
     superset invariant (decision 6) working as designed. No new seam into the registry.
   - (e) **Per-iteration usage** — already on each `llm_response` event (`usage=u`). The transcript
     renders each iteration's cost on its own line; `turn_done` keeps the cumulative. No redundant
     per-iter list added.
   The existing full-result + payload capture already works (verified) — kept. Per-chunk raw-stream
   tracing is NOT added (too noisy; revisit only if a truncation bug needs it — O3).

6. **Console vs file is a level cut, not a content cut.** The file gets a strict superset of the
   console (everything at DEBUG+). We never write something to console that's absent from the file. This
   keeps "reproduce what the user saw" trivial: the file is the source of truth, the console is a
   filtered view.

7. **Re-leveling audit (the bulk of the diff).** The 118-call survey audited every logger call across
   24 modules; ~37 calls actually shifted level (the rest were already correct — WARNING/ERROR that stay
   file-only, or already-INFO user events). Net shifts: ~25 INFO→DEBUG (watcher reloads, worker/bridge
   lifecycle, bootstrap dirs, trace open, lib-index counts, per-frame render detail, copilot per-iter/
   enqueue), ~6 WARNING→DEBUG (cleanup-failure / per-uniform / recursive-delete-skip detail; the
   queue-full "dropping job" lines stay WARNING — a dropped export job is user-relevant), 4 ERROR→WARNING
   (the two fallback-config loads — `integrations.json` + `app_state.json` — each via two paths; the app
   keeps running on defaults, not a crash). High-level user actions stay INFO+console; WARNING/ERROR stay
   file-only except the app-crash path (`ui.py::main`) which is console too. **Override of the survey:**
   `notifications.py` toast-echo lines (both the error-toast and success-toast `logger` calls) → **DEBUG
   file-only**, NOT INFO+console — the toast IS the user-facing surface; echoing every toast to console
   is exactly the spam we're killing.

8. **`copilot/agent.py` keeps its concise `logger` lines AND the trace `tr.event` calls** — they serve
   different streams. The `logger.info` lines become the console/file high-level view (re-leveled: turn
   start/done stay INFO; per-iter `finish=`, per-tool truncated-result preview → DEBUG). The `tr.event`
   calls feed the full-fidelity trace. They are not redundant; they're the summary-vs-transcript split.

9. **The on-disk cut: durable-portable state in the project dir; disposable machine-local output in
   `app_data_dir()`.** A project dir (`app_state.json` + `nodes/` + `media/` + `trash/`) is a
   self-contained, relocatable unit — it can live anywhere and travels with the user. So the rule:
   - **App log** → `app_data_dir()/logs/` (central). It is app-GLOBAL, not project-scoped — the file
     watcher, exporters, and startup all log before/across any project. Never per-project.
   - **Agentic trace** → `app_data_dir()/copilot_traces/` (central, NOT in the project dir). It is debug
     ephemera: large, disposable, read by humans/agents only, nothing reads it back. Putting it in the
     project dir would bloat the user's portable folder with debug transcripts they never asked for.
     Filename is project-stamped so sessions are distinguishable across projects:
     `copilot_<project-slug>_<timestamp>.transcript`.
   - **Conversation state** (the restorable chat) → goes in the PROJECT dir, but that is **feature
     022's** scope, not 021. 022 inherits this same cut; 021 only locks where the trace + app-log live.
   This principle (durable-portable→project dir; disposable-local→`app_data_dir()`) is recorded in
   `conventions.md ## Design decisions` so 022 and future artifacts honor it.

## Files touched
- **New:** `shaderbox/logging_setup.py` (`LoggingConfig`, `LOGGING_CONFIG`, `configure_logging()`).
- **`shaderbox/paths.py`:** add `log_dir()` + `copilot_trace_dir()`.
- **`shaderbox/copilot/trace.py`:** rewrite — transcript format, new event kinds/fields (decision 5),
  retention pruning on session start (decision 1).
- **`shaderbox/copilot/agent.py`:** add the new `tr.event` calls (system prompt, args-parse fail, tool
  exception, per-iter usage in turn_done); re-level the `logger` lines (decision 8).
- **`shaderbox/copilot/llm/openrouter.py`:** re-level the LLM-request `logger` line INFO→DEBUG. (The
  `llm_wire_request` trace event was NOT added — decision 5b: it would need a trace handle threaded
  through the provider-neutral `LLMClient` Protocol, and the agent-level `llm_request` event already
  captures the full semantic request.)
- **`shaderbox/copilot/session.py`:** wire trace-retention pruning; re-level the 7 lifecycle lines to
  DEBUG.
- **`shaderbox/ui.py`:** `main()` calls `configure_logging()` instead of the inline `logger.add`;
  re-level the watcher/frame-loop lines per the audit.
- **`scripts/smoke.py`:** call `configure_logging()` so a smoke run matches the app; keep its own
  pass/fail at INFO+console.
- **Re-level only (level changes, no structural change):** `app.py`, `core.py`,
  `shader_lib/file_ops.py`, `exporters/telegram.py`, `exporters/youtube.py`,
  `exporters/integrations.py`, `ui_models.py`, `notifications.py` — these are the files whose calls
  actually shifted level, per the audit table.
- **Audited, already correct (NO diff):** `ui_primitives.py`, `tabs/share.py`, `tabs/share_state.py`,
  `tabs/render.py`, `popups/lib_picker/filtering.py`, `shader_lib/tags.py`, `shader_lib/favorites.py`,
  `widgets/details.py`, `copilot/bridge.py`, `copilot/tools/registry.py` — the survey checked these;
  their calls are WARNING/ERROR (stay file-only) or already DEBUG, so no change was needed.
- **Docs:** `conventions.md ## Design decisions` (the central-setup rule + the no-`get_logger` rule,
  with revisit triggers), `conventions.md ## Known quirks` (any loguru footgun found), `roadmap.md`
  (one row + banner refresh).

## Manual verification
*(headless where possible; the console/file split + trace readability need a real run)*
- `make check` clean (ruff + pyright, zero suppressions).
- `make smoke` passes (200 headless frames); confirm `configure_logging()` ran (a log file appears
  under `logs/`, DEBUG lines present in file).
- **Console terseness:** `uv run python ./shaderbox/ui.py`, drive a copilot read→edit turn; confirm the
  console shows ONLY high-level lines (turn start/done with cost, tool called by name, node saved) and
  NO worker-lifecycle / per-iteration / watcher spam. (Maintainer eyeballs the console — UN-headless.)
- **File completeness:** the same run's `logs/shaderbox_*.log` contains the DEBUG lifecycle lines absent
  from console (superset invariant, decision 6).
- **Trace readability:** open the session's trace file with `Read`; confirm it reads cleanly cold —
  visible turn banners, real-newline shader source + system prompt, full tool args + results, per-iter
  cost. (This is the acceptance test for decision 4 — an agent must find it more readable than the old
  JSONL.)
- **Retention:** generate >20 trace files (or temporarily lower `trace_retention`), restart a session,
  confirm old files pruned to the cap.

## Resolved questions
*(all locked by the maintainer; defaults taken)*
- **O1 — trace_retention = 20 files.** Pure debug ephemera (chat restore split to 022); keep the last 20
  past debug sessions.
- **O2 — app-log `rotation="5 MB", retention=5`** unchanged.
- **O3 — no per-chunk raw-stream tracing.** Excluded as too noisy (decision 5). Revisit only if an
  argument-truncation/ordering bug needs the raw chunk sequence.
- **O4 — no `SHADERBOX_LOG_LEVEL` env override** in this feature. Deferred to its trigger (first time
  console DEBUG is needed without a code edit) — it's a known ~3-line add, recorded in Out-of-scope.
