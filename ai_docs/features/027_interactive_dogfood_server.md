# 027 — Interactive dogfood (resume/dump, one blocking call per turn)

> **Filename kept (`027_interactive_dogfood_server.md`) for the roadmap-row pointer; there is NO
> server.** The original draft proposed a long-running server process. A devil's-advocate storm pass
> (2026-06-09), grounded in the actual code, killed that: the only expensive non-rebuildable state is
> the LLM conversation, and that is ALREADY serialized for free (`ConversationStore` / NL-only history
> / zero LLM calls on reload). The rebuild it would protect (EGL ctx + worker + node compiles) is ~1s.
> So we ship the cheap shape: a `resume`/`dump` seam on the harness + one documented blocking
> `uv run` per turn. See `## Review history` for the full reversal.

## Goal

Make dogfooding genuinely INTERACTIVE — the driver (Claude) sends ONE message, the call BLOCKS until
the copilot's reply comes back, Claude READS it and composes the next message from what actually
happened — WITHOUT a pre-scripted reply sequence and WITHOUT losing harness state between the
read-and-think gaps.

The insight that shapes the whole feature: **each turn is already one blocking call** (a `uv run
python -c '...'` blocks until it exits and prints), and **the only state worth persisting across turns
is on disk already**:
- the conversation (NL-only history + chat messages + cost) — `ConversationStore` at
  `project_dir/copilot/conversation.json`, restored by `load_conversation` with ZERO LLM calls;
- the shaders — the copilot's edits were written to `project_dir/nodes/*/shader.frag.glsl` by the
  tools, so a fresh load re-reads the real current source;
- the lib + integrations — under `SHADERBOX_DATA_DIR`.

So a per-turn process that REUSES the same `project_dir` + `SHADERBOX_DATA_DIR` reconstructs the full
state in ~1s (node compile ≈3.5ms, render ≈2.6ms on V3D — the rebuild is cheaper than one grok
round-trip) and is **stateless + crash-proof by construction**: if a turn dies, the next call just
resumes. No long-lived process, no PID to juggle, no transport protocol, no liveness/timeout/crash
recovery — every one of those risks existed ONLY because the rejected design kept a process alive.

**Everything dogfood-related is consolidated under ONE directory — `scripts/dogfood/`** (decision 7):
the harness, the scenarios, and ALL run artifacts (the per-run project dirs, the data dir with
lib/integrations, the JSON dumps, the trace copies, the rendered PNGs). Nothing scatters to `/tmp` or
`ai_docs/` any more.

## Out of scope (each a trigger for later)

- **A long-running server / client / FIFO / request-reply transport — NO.** Killed by the storm
  (`## Review history`). Trigger to revisit: a scenario provably needs to answer a gate based on
  reading the gate's OWN prompt text first (so the answer can't be decided up front), OR multi-gate
  fidelity WITHIN one in-flight turn (a publish that opens a token gate THEN a pack gate, answered
  gate-by-gate). Both need the worker turn to SURVIVE a read-and-think gap — which only a long-lived
  process gives (the paused turn dies on process exit; decision 4). Today the multi-gate path
  precheck-FAILS in the harness anyway (`ExporterRegistry` is empty — `harness.py::create`), so it
  isn't dogfoodable; a single gate whose answer is decided up front is fully handled by the
  single-process one-liner (decision 4). If gate-prompt-dependent answering ever becomes a real
  scenario, file a new feature for a blocking server then — not before.
- **Auto-judging / assertions — NO.** The judge stays the human (Claude) reading the JSON reply + the
  PNGs. `dump` only transports the turn's outcome; it never decides pass/fail (026's contract).
- **Verbatim mid-turn history fidelity — NO.** The persisted history is the NL turn-summary (reply
  prose + action ledger + nodes touched), not the verbatim tool-call trace — by design (it's exactly
  what the live App feeds the model on restart). A scenario needing the agent to recall its EXACT prior
  tool calls is the one case resume is lossy; the live trace file (`trace_path`) still has the full
  detail for the HUMAN's analysis, just not re-fed to the model. Acceptable — matches production.
- **GUI / TUI / sockets — NO.** Filesystem + stdout JSON only.

## Design decisions (numbered, lock-in)

1. **Resume reuses one stable `project_dir` + `SHADERBOX_DATA_DIR`; the conversation lives inside the
   project dir.** `DogfoodHarness.create()` today mkdtemp's BOTH a project dir and (module-level) a
   data dir. Resume threads BOTH forward. The conversation file is already derived from the project
   dir (`paths.py`: `copilot_conversation_path = project_dir/copilot/conversation.json`), so there is
   NO separate conversation-path argument — one stable `project_dir` carries shaders + conversation,
   one stable `SHADERBOX_DATA_DIR` carries lib + integrations. The caller (the `/dogfood` one-liner)
   passes `SHADERBOX_DATA_DIR` in the COMMAND env every turn (it's read at module import — `paths.py`
   / `harness.py` top), and `project_dir` as a `create()` arg. Both default UNDER `scripts/dogfood/runs/`
   (decision 7), not `/tmp`.

2. **`DogfoodHarness.create(project_dir: Path | None = None, *, seed_templates: bool = True)` — resume
   when `project_dir` is given.** When `project_dir` is None (today's default) → mkdtemp a fresh one +
   seed templates (unchanged 026 behavior). When given an EXISTING project dir → SKIP seeding (the
   nodes are already there from prior turns), `session.load(project_dir)` (re-reads shaders +
   app_state), THEN restore the conversation: `ConversationStore.load_and_migrate(session.paths.
   copilot_conversation_path)` → `copilot.load_conversation(store)`. Both message cursors
   (`_seen_msg_count` AND `_dumped_msg_count` — decision 3) are set to `len(state.messages)` so the
   restored chat counts as "already seen" for both printing and dumping. The CURRENT-NODE pointer
   is restored by `load()` from the persisted `app_state.json` (which `dump`/`reload` now write —
   decision 3); `create()` only picks a default current node when it's UNSET (a fresh project), so a
   resumed turn keeps the node a prior `switch_node` left it on. All composed from methods that
   ALREADY exist on `CopilotSession` / `ConversationStore` — no new production (`shaderbox/`) code.
   NOTE (so a future reader doesn't "fix" it): `load_conversation`'s
   docstring says "after reset_conversation (fresh session)", but a freshly-constructed `ProjectSession`
   already has empty `state`/`history` (`load_conversation` ASSIGNS, not appends) — so the resume branch
   needs NO `reset_conversation`. This matches App's FIRST-init path (`app.py` skips reset on first
   `__init__`, only resets on a project SWITCH); don't add a reset here (harmless but unnecessary).

3. **`dump(path: Path) -> None` persists the conversation + app_state AND writes a structured JSON
   turn-result.** Effects: (a) `copilot.save_conversation(paths.copilot_conversation_path)` so the NEXT
   turn's `create(project_dir)` can resume it; (a2) `session.app_state.save(paths.app_state_file)` so a
   `switch_node`'d current node survives the resume (load() restores it; without this the resume falls
   back to the oldest node); (b) write a JSON blob to `path` (a stdout-noise-free channel —
   loguru/GLFW spew goes to stderr/stdout, the structured result goes to a FILE the caller `cat`s, OR
   to a clearly-delimited stdout marker). The JSON is built from the harness's STRUCTURED state
   (`state.messages` slice since `_seen_msg_count`, `state.last_turn`, `_open_gate()`,
   `session_cost_usd`, `str(trace_path)`), NOT scraped from the existing `print()` output. Shape:
   ```json
   {
     "new_messages": [{"role": "assistant", "text": "..."}, {"role": "tool_status", "text": "..."}],
     "assistant_text": "the final assistant bubble, for convenience",
     "open_gate": {"text": "...", "kind": "confirm|credential|config"} | null,
     "last_turn": {"context_tokens": 8123, "reply_tokens": 412, "cost_usd": 0.0123} | null,
     "session_cost_usd": 0.0481,
     "trace_path": "/.../copilot_traces/copilot_<slug>_<stamp>.transcript",
     "project_dir": "/tmp/shaderbox-dogfood-proj-XXXX",
     "data_dir": "/tmp/shaderbox-dogfood-data-XXXX"
   }
   ```
   `project_dir`/`data_dir` echo the two stable paths so the caller copies them into the NEXT turn's
   command without bookkeeping.

   **`dump` uses its OWN cursor (`_dumped_msg_count`), NOT the printer's `_seen_msg_count`.** Critical:
   `drive_until_idle` → `_print_new_messages` ALREADY advances `_seen_msg_count` to
   `len(state.messages)` before `dump` runs, so sharing that counter makes `dump`'s `new_messages`
   always empty. `dump` slices `state.messages[_dumped_msg_count:]`, then advances
   `_dumped_msg_count = len(state.messages)` (slice-then-advance) — independent of the print cursor.
   The resume branch (decision 2) sets BOTH cursors to `len(state.messages)` (the restored chat is
   already-seen for both printing and dumping).

4. **The existing `send` / `drive_until_idle` / `approve` / `decline` / `render` / `nodes` printing
   methods stay BYTE-FOR-BYTE (026 REPL contract is sacred).** `dump` is purely additive — it reads
   state, it does not replace the prints. The one-liner calls the SAME methods a human REPL would, then
   `dump`s at the end.

   **A gate MUST be answered WITHIN the one process that opened it — it CANNOT span two turns.** A
   gate pauses the worker on `gate.ask()`/`pending.done.wait()`; the worker is `daemon=False` and dies
   with the process, so the paused turn is LOST on process exit. (The unresolved `pending_action` card
   CAN end up persisted — `save_conversation` doesn't hard-gate on `in_flight` — so `_restore_conversation`
   force-resolves any restored open gate on resume, since no worker is parked on it any more.) There is
   no "dump the gate, exit, resume and answer in turn N+1" — that paused turn is gone. The ONLY correct
   pattern for a turn that may gate: answer it inside the SAME one-liner —
   ```python
   h.send("delete the Media node")
   h.drive_until_idle()                 # stops on the gate
   if h._open_gate() is not None:
       h.decline()                      # or approve() — the driver DECIDES UP FRONT, per the scenario
       h.drive_until_idle()             # let the copilot react to the decision
   h.dump(out_path)
   ```
   The driver (Claude) decides the gate answer when it COMPOSES that turn's one-liner (it knows the
   request will gate — e.g. scenario C declines on purpose). This is still interactive: Claude reads the
   PRIOR turn's reply, then writes the next one-liner with the gate decision baked in. What's NOT
   possible is answering a gate based on reading the gate's OWN prompt text first across a process
   boundary — that's the multi-gate-fidelity case reserved for the (out-of-scope) server. For a turn
   whose gate answer is unconditional, `drive_until_idle(auto_approve_gates=True)` is the shortcut.

5. **`reload()` IS the resume path, exposed as a no-op-friendly harness method for the persistence
   scenario.** Spec'd as: persist the live conversation (`save_conversation`) → `reset_conversation`
   → `load_and_migrate` → `load_conversation` → reset `_seen_msg_count`, guarded on `not in_flight`
   (mirrors `reset_conversation`'s invariant — a mid-turn reload strands the worker). In the
   one-liner-per-turn model `reload()` is rarely needed (each process is already a fresh load), but it
   lets a SINGLE-process REPL session simulate an app restart without exiting — and it's the literal
   composition `create(project_dir=...)` resume uses, so it's free. This **folds the `todo.md`
   "conversation-restart" coverage gap**. NOTE re: `trace_path` — `reset_conversation` rotates the
   trace file, so a caller must re-read `trace_path` after a `reload()`, never cache it.

6. **`render()` already returns the PNG path; keep it.** No structured `RenderOutcome` needed — the
   path (or `""` on failure) + the existing print is enough, and `dump`'s JSON can carry the last
   render path if a turn rendered. The threading caveat is UNCHANGED and load-bearing: `render()` runs
   the GL work on a throwaway thread while the context-owning thread drains the bridge; the one-liner
   MUST call `render()` on the same (main) thread that called `create()` — which it does (the script
   is single-threaded top-to-bottom). Renders default into `<project_dir>/renders/` (already the
   copilot's render dir), which lives under `scripts/dogfood/runs/` (decision 7) — so PNGs are
   consolidated too, no manual `cp` to `/tmp`.

7. **Everything dogfood lives under `scripts/dogfood/` — one consolidated home.** The layout:
   ```
   scripts/dogfood/
     __init__.py        # re-exports DogfoodHarness so `from scripts.dogfood import DogfoodHarness` STILL works
     harness.py         # the former scripts/dogfood.py (DogfoodHarness + the module-top env block)
     analyze.py         # auto run-rollup: coverage/cost/tokens/recoveries + report-template fill
     REPORT_TEMPLATE.md # {{AUTO}}/{{HUMAN}} report skeleton analyze.py fills
     scenarios/         # the goal-driven mission(s) + a README (moved out of ai_docs/scenarios/)
     runs/              # GITIGNORED — per-run project dirs, the data dir, JSON dumps, trace copies, PNGs
   ```
   - `harness.py`'s module-top env block changes the two `mkdtemp` defaults to mkdtemp UNDER
     `scripts/dogfood/runs/` (a `_RUNS_DIR` constant resolved relative to the file), not the system
     temp dir — so `SHADERBOX_DATA_DIR` and the default `project_dir` both land in `runs/`. A
     caller-provided `SHADERBOX_DATA_DIR` / `project_dir` still wins (resume).
   - `scripts/dogfood/runs/` is `.gitignore`d (run artifacts are regenerable junk, never fixtures —
     same rule as `projects/*/renders/`). `harness.py` + `scenarios/` + `__init__.py` ARE tracked.
   - The `scenarios/` move is a `git mv ai_docs/scenarios/* scripts/dogfood/scenarios/` (the old files
     are tracked); the 6 shallow probes are then replaced by ONE goal-driven mission (`01_shape_gallery.md`).
     Scenarios are dogfood-rig material, not product docs — they belong with the harness, not in `ai_docs/`.
   - No build.sh change: `scripts/` ships only an explicit allowlist (`scripts/README.md` + launchers),
     so `scripts/dogfood/` is already excluded from the bundle; confirm the FORBIDDEN gate still passes.

8. **Context-wipe is an engine seam, not harness-only plumbing.** A scenario's cold-start half needs a
   FRESH agent (zero conversation memory) on the SAME project — the inverse of resume. The clear-chat logic
   already lived in `App.copilot_clear_chat` (archive -> clear_checkpoints -> reset_conversation -> save
   empty). It's pure project/copilot-core work with no UI dependency, so it belongs on the engine that was
   factored out of `App`: **`ProjectSession.clear_conversation()`** is the canonical home; `App.copilot_clear_chat`
   becomes a one-line forwarder (`self.session.clear_conversation()`), and the dead `_conversation_stamp`
   helper + the `archive_conversation` import drop from `app.py`. The harness exposes
   `DogfoodHarness.clear_context()` -> `session.clear_conversation()` + resets both message cursors. This is
   the one production (`shaderbox/`) change in 027 — a lift, not new behavior (App's clear path is unchanged
   in effect; it now routes through the engine, matching the 025 "core lives in ProjectSession, App forwards"
   decision).

9. **Default model -> `openai/gpt-5.1-codex-mini`** (`CopilotIntegration.model`). grok-4.3 writes BAD GLSL —
   you can't dogfood the authoring pipeline on a model that can't write a shader. codex-mini is the
   cheap-but-code-competent pick (~$0.25/$2.00 per Mtok, tool-call compatible, 400k ctx). This is the live
   in-app default too (not a dogfood-only override), so the shipped copilot uses it as well.

## Files touched

- `scripts/dogfood.py` → **`scripts/dogfood/harness.py`** (`git mv`). `create()` gains the optional
  `project_dir` resume branch (decision 2); new `dump(path)` (decision 3) + `reload()` (decision 5) +
  `clear_context()` (decision 8); the module-top env defaults point under `scripts/dogfood/runs/`
  (decision 7). NO change to `send`/`drive_until_idle`/`approve`/`decline`/`render`/`nodes` signatures
  (decision 4).
- **`shaderbox/project_session.py`** — new `clear_conversation()` (decision 8: archive → clear_checkpoints
  → reset_conversation → save empty; the engine seam for context-wipe), + the `archive_conversation` import.
- **`shaderbox/app.py`** — `copilot_clear_chat` collapses to a one-line forwarder to
  `session.clear_conversation()`; the dead `_conversation_stamp` helper + the `archive_conversation` import
  are removed (decision 8).
- **`shaderbox/exporters/integrations.py`** — `CopilotIntegration.model` default →
  `openai/gpt-5.1-codex-mini` (decision 9).
- **`scripts/dogfood/__init__.py`** (new) — `from scripts.dogfood.harness import DogfoodHarness` so the
  public import path is unchanged for the skill/scenarios/REPL. VERIFY at impl that
  `from scripts.dogfood import DogfoodHarness` still resolves under `uv run` + pyright (PEP 420 implicit
  namespace package — `scripts/` has no `__init__.py`; should cover it, but a live
  `uv run python -c 'from scripts.dogfood import DogfoodHarness'` is the gate). HAZARD: importing the
  package eagerly runs `harness.py`'s module-top env block (the `mkdtemp` + `setdefault` +
  integrations write), so the "`SHADERBOX_DATA_DIR` MUST be set in the process env on the command line,
  never assigned in-script after import" rule is load-bearing — bold it in the SKILL.md rewrite.
- **`scripts/dogfood/scenarios/`** (moved from `ai_docs/scenarios/` via `git mv`, then rewritten) —
  the 6 shallow probes deleted; ONE goal-driven mission to OBKATAT the mechanism first
  (`01_shape_gallery.md` — a 2×2 grid of simple 2D shapes built as separate nodes then combined by a
  context-wiped fresh agent; visually unambiguous on purpose), + README. Harder/code-quality/token-overflow
  scenarios come LATER once 01 is proven. See `## The scenario`.
- `.claude/skills/dogfood/SKILL.md` — rewrite §1: mandate the resume/dump one-liner, FORBID
  pre-scripted multi-turn drivers, document the exact per-turn command + the gate two-step. Fix EVERY
  path (`scripts/dogfood/harness.py`, `scripts/dogfood/scenarios/`, `scripts/dogfood/runs/`). Update §5
  cleanup (no throwaway driver to delete; artifacts already live in `runs/`).
- `.gitignore` — add `scripts/dogfood/runs/`.
- `ai_docs/todo.md` — delete BOTH the "dogfood must be driven INTERACTIVELY" deferral and the
  "conversation-restart / gate-decline coverage" deferral in the SAME commit this lands (027 folds both
  in via resume + the gate two-step + the deep scenarios exercising decline). Fix the
  `ai_docs/scenarios/` path reference in the lazy-tool-catalogue entry if any survives.
- `ai_docs/conventions.md` — fix the `scripts/dogfood.py` path reference (`## Known quirks` type-ignore
  note) → `scripts/dogfood/harness.py`.
- `ai_docs/features/026_copilot_dogfood_harness.md` — fix the `ai_docs/scenarios/` path reference.
- `ai_docs/roadmap.md` — flip the 027 row to `done` (or `partial` if scenarios trail) + rewrite the
  banner; fix the `scripts/dogfood.py` reference in the 026 row.
- `build.sh` — NOT edited (the FORBIDDEN gate ships only an explicit allowlist; `scripts/dogfood/` is
  excluded). VERIFY at impl: grep `build.sh` for any literal `scripts/dogfood.py` reference the rename
  would orphan, and confirm `bash build.sh` (or the FORBIDDEN gate) still passes. Confirm the sibling
  probe scripts (`scripts/token_probe.py`, `scripts/copilot_*_check.py`) don't import `scripts.dogfood`
  (grep showed none do).

## The scenario

Start with ONE goal-driven mission to OBKATAT the mechanism — `scripts/dogfood/scenarios/01_shape_gallery.md`.
Deliberately SIMPLE + visually unambiguous (flat 2D shapes you can judge correct/wrong at a glance — NOT
SDF/3D/lighting, which you can't eyeball for quality). A scenario = a final GOAL + an iterative build-up +
the pressure axes it attacks, in free text — never a numbered step-script (that just gets replayed).

- **01 Shape gallery.** Final goal: one `Gallery` node drawing a 2×2 grid of four distinct white shapes
  (circle/square/triangle/ring) on a dark background. Build-up: each shape as a SEPARATE node first, then
  **`clear_context()` (the context wipe — a fresh agent, zero memory)** and make the wiped agent READ the
  four nodes from disk + combine them into Gallery. Pressure axes: tool-use under a wipe (does the fresh
  agent `read_shader`/`grep` the four, or hallucinate from its weights? — pin it with load-bearing
  constants the Gallery must reuse), visual honesty (clean compile vs a broken grid it can't see), and a
  token-growth observation (a 4-node read vs a 1-node turn — a baseline for the later overflow scenario).

Harder scenarios (code-quality grading, real token-overflow provocation, trickier composites) come LATER,
once 01 proves the mechanism end-to-end.

## Manual verification

- **Consolidation:** after a run, confirm the project dir + data dir + dumps + traces + PNGs ALL live
  under `scripts/dogfood/runs/` and nothing landed in `/tmp` or `ai_docs/`. `git status` shows
  `runs/` is ignored (no run artifacts staged).
- **Resume round-trip:** turn 1 one-liner (`create()` fresh → `send` → `drive_until_idle` → `dump`),
  note the printed `project_dir` + `data_dir` (both under `runs/`). Turn 2 one-liner REUSING both →
  confirm the conversation restored (the dump's `new_messages` shows only turn-2's output, not
  turn-1's) AND a shader edited in turn 1 is present (read it, or `render()` it). State persisted
  across two separate processes.
- **Gate two-step:** a turn that gates (e.g. `delete_node` → always-gate). Turn N dumps with
  `open_gate` set; turn N+1 resumes + `decline()`s → confirm the node survives (`nodes()` still lists
  it) and the copilot reacts to the decline (the gate-decline path, previously unexercised).
- **`reload()` in one process:** a few turns → `reload()` → confirm the conversation is restored and
  `trace_path` changed (new trace file).
- **`make check`** green (ruff + pyright). Harness-only change — `make smoke` is N/A (no App/UI touch),
  but run a real dogfood turn live (billed) to confirm the one-liner works end-to-end on the Pi.

## Open questions for the user

None blocking — the resume approach was chosen by the maintainer after the storm (devil's-advocate
brief). One judgment call deferred to impl: whether `dump` writes JSON to a FILE the caller `cat`s, or
to stdout behind a `===DOGFOOD-JSON===` delimiter the caller greps past the loguru noise. Lean: a FILE
(`--out` path) — clean separation, no delimiter fragility; the one-liner ends `h.dump(Path("/tmp/df.json"))`
and the caller's command tails it. Decide at impl; it's a 1-line difference.

## Review history

- **2026-06-09 — the server was REVERSED to resume/dump before plan-lock.** The draft (and three of
  four storm agents) designed a long-running blocking server (transport B req/reply files + a
  `dogfood_client.py` CLI + a multi-gate state machine + liveness/timeout/crash recovery). A
  devil's-advocate agent, grounded in the code (not training-data recall), showed the premise was
  false: the conversation is already a free, NL-only, zero-LLM-call serialized replay log
  (`ConversationStore`/`load_conversation`), and the EGL+worker+node-compile rebuild it protects is
  ~1s (cheaper than one LLM round-trip). The server's entire complexity (5 enumerated risk areas)
  existed only to avoid that rebuild. The maintainer chose the resume/dump one-liner. The three storm
  briefs' GOOD parts survive: the structured turn-result JSON shape (from the harness-API agent) is
  decision 3; the gate-kind surfacing (`GateKind` is CONFIRM/CREDENTIAL/CONFIG, already on the gate
  `Message`) informs the `open_gate` field; the scenario designs (from the scenario agent) seeded
  `## The scenario` (later simplified by the maintainer to ONE goal-driven mission to obkatat the
  mechanism first). The server design is preserved only as the Out-of-scope trigger
  (multi-gate-in-one-turn fidelity).
