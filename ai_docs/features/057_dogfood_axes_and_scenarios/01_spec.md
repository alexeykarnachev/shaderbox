# 057 — Dogfood axes & judge tooling (spec, v2)

Builds on `00_research.md` (fact base) and the landed cornerstone pilot (scenarios 03/04/05/08,
the cornerstone contract, `turn_time_budget_s`; runs 2026-07-27). This spec covers what REMAINS:
the five-axis report structure and the judge tooling the pilot hand-rolled five times.

Everything here is dogfood-infra (`scripts/dogfood/*` + docs + tests) — ZERO engine/copilot code
(the one `shaderbox/`-adjacent seam used, `ScriptEngine.dry_run` / `fresh_behavior_for` /
`tick_export`, is EXISTING API reached through the session; nothing in `shaderbox/` changes).

## Goal

1. **Five axes become report-template structure** (`REPORT_TEMPLATE.md`; one report per SCENARIO —
   the pilot's one-data-dir-per-scenario layout is the contract):
   - `fidelity` — HUMAN: the scenario's checklist as a table (`| check | PASS/FAIL | measurement |`)
     + a counted "X of Y sub-requirements landed" line.
   - `motion` / `logic` — HUMAN: verdicts against the scenario's stated ground truth, each citing
     its measurement (a `render_strip` sheet; a `script_values` / judge-number series).
   - `honesty` — AUTO: engine-look stats from the trace (rows per LOOK: verdict, look #, vision_ok,
     node; a per-turn FINAL verdict; parse rate; summed **engine-look vision spend** — labeled so,
     model-initiated probe spend has no trace event and that gap is a named copilot-wave finding)
     + a HUMAN line for over/under-claims caught.
   - `process` — the existing AUTO slots unchanged.
   - a **DIALOGUE section**: the driver pastes `--dialogue` output (the report contract's required
     artifact gets a home). Sections renumber; the SKILL's "7 HUMAN sections" count updates with it.
2. **`analyze.py` learns the 056 trace events.** A `verdicts` block in JSON/markdown + the honesty
   AUTO slot. Named parser edits (the section framing is reused, but none of this is free):
   the field whitelist gains `verdict/ask_line/node/look/vision_ok/input_tokens/output_tokens/
   cost_usd` (+ bool coercion for `vision_ok`); `engine_look_usage.cost_usd` gets its own rule
   (`_USAGE_RE` doesn't match it); sums span ALL transcript segments of the run; `parse_rate` :=
   rows whose `ask_line` matches the strict `ASK: met|not-met|unclear` shape ÷ rows with
   `vision_ok: True` (the engine normalizes garbled→unclear and blind→none, so `ask_line` is the
   only garble discriminator; zero `vision_ok` rows ⇒ the slot prints `n/a`, never a division). Same wave, same file: `clean_streak_giveup` joins the terminal
   kinds and `turn_done`'s `cutoff=` field is read — the churn/time/iteration cutoff turns are
   exactly the class the honesty axis investigates (today they glyph as clean). `--scenario <name>`
   fills the currently-hardcoded `scenario_list` slot.
3. **`analyze.py --dialogue`** prints the run's user-visible dialogue. Target semantics unchanged
   from `analyze` (positional = DATA dir): the project dir is resolved from the run's dumps'
   `project_dir` field (`--project` overrides); the branch runs BEFORE `analyze()` and exits.
   Source: `<project>/copilot/conversation.json` `messages` — the UI chat store, i.e. exactly what
   the user SAW (NOT "what the model saw" — that claim was false) — with an explicit role map:
   `user` → `**User:**`; `assistant` → `**Copilot:**`; `error` → `**Copilot [engine]:**` (a
   limit-cutoff note IS the visible reply — dropping it would erase the under-claim evidence);
   `tool_status` → a dim `[engine] ...` line; `turn_snippet`/`pending_action` skipped. Context-wipe
   runs (scenario 01): `clear_context()` does NOT delete the file — it archives then re-saves an
   EMPTY store to the same path — so the dumps fallback triggers on missing **or empty/short**
   (fewer messages than the dumps' summed `new_messages`), never on absence alone (an empty
   post-wipe store reading as a complete dialogue is exactly the under-claim failure mode);
   pre-wipe content, when flushed, also lives in `copilot/archive/conversation_<stamp>.json` as a
   second source. The cornerstone README contract text is updated to name both channels.
4. **`DogfoodHarness.render_strip(times, node_id="", *, size=300, fps=30)`** — ONE horizontal
   contact sheet. **Per-sample REPLAY, not live ticks**: for a scripted node each sample drives a
   FRESH behavior (`fresh_behavior_for` + `tick_export` stepping frames `0..round(t*fps)`) — the
   `dry_run`/export-isolation semantics, so a stateful integrator (using `ctx.dt`) samples its real
   trajectory and repeated calls are reproducible; `render_at`'s single live tick is exactly wrong
   here (the pilot's bounce script tolerated it only by self-deriving dt). A script-less node
   renders plainly at each t. Frames alpha-composite onto (25,25,40) (the SKILL's
   white-on-transparent rule — deliberately unlike the eye's (40,40,40) strip; say so in the
   docstring), 4px gutter, a small `t=<s>` label per cell; the sheet is written under
   `session.paths.renders_dir` (so `dump()`'s `last_render_path` points at it) and sets
   `_last_render_path`. Canvas size is SAVED and RESTORED around the calls (`render_at`'s
   `set_size` persists into node.json via `dump()` and would silently rewrite an agent-set
   `set_canvas_size`).
5. **`DogfoodHarness.script_values(times, node_id="", *, fps=30)`** — the logic axis's numeric
   probe (00_research infra gap #3): a thin passthrough to `ScriptEngine.dry_run` returning the
   per-sample driven-uniform values; no engine change.
6. **`scripts/dogfood/judge.py` (NEW)** — pixel primitives, PIL/numpy only, NO assertions
   (numbers out; judgment stays human). Typed contract:
   - `load_rgb(path: str|Path) -> np.ndarray` (H,W,3 int16)
   - `grid_cell(im, row: int, col: int, rows: int, cols: int) -> np.ndarray` (view)
   - `region_diff(a, b) -> float` (mean abs per-channel difference)
   - `bright_centroid(im, thresh: int = 170) -> tuple[float, float] | None` ((x, y) px)
   - `color_mask_centroid(im, rgb_min, rgb_max) -> tuple[float, float] | None`
   - `column_runs(im, thresh: int = 170) -> list[tuple[int, int]]` (x-extent runs — the
     circle-counter)
   - `farthest_bright_angle(im, thresh: int = 170) -> float | None` (degrees, atan2 from the
     image center to the farthest bright pixel — the rotation-direction probe for 08's C2/C9)
   **And the import-side-effect fix that makes "imports with no GL" true:**
   `scripts/dogfood/__init__.py`'s eager `from .harness import DogfoodHarness` becomes a lazy
   module `__getattr__` — today ANY `scripts.dogfood.*` import runs harness's module-top env
   block (mkdtemps a `runs/data-*` dir, writes an `integrations.json` that can carry the LIVE
   OpenRouter key, imports glfw/moderngl); verified: every `pytest tests/test_dogfood_analyze.py`
   run litters one. The lazy hook kills the class; `from scripts.dogfood import DogfoodHarness`
   keeps working verbatim.
7. **Docs sync, same wave**: `scenarios/README.md` — cornerstone table corrected (06/07 deleted →
   removed; 08 added), the dialogue-source contract sentence updated per Goal 3, tooling pointers,
   the stale report path (`026_…` → `NNN_…`); `.claude/skills/dogfood/SKILL.md` — §2 strip gotcha →
   `render_strip` (with the stateful-replay note), §3/§4 verdict extraction + `--dialogue` +
   the new HUMAN-section count; `scripts/dogfood/README.md` — the new methods/flags/module.

## Out of scope (each with a trigger)

- **New scenarios** (pong, 3D compound, the all-axes final). **Trigger:** maintainer green-lights
  the next echelon after the base set drives one copilot-fix cycle.
- **Automating the HUMAN judgments.** **Trigger:** a checklist fully expressible as assertions AND
  stable across 3+ runs.
- **A trace event for MODEL-initiated probe vision spend** (engine code; the honesty slot is
  labeled engine-look-only because of it). **Trigger:** the first report that needs total vision
  spend and can't get it from `session_cost_usd` deltas.
- **New judge primitives beyond the listed seven.** **Trigger:** a scenario measurement that needs
  hand-rolled pixel code twice.
- **The analyzer's multi-line `user_text` display gap** (indented continuation lines aren't
  parsed; per-turn "Ask (head)" renders empty for multi-line asks — display-only; no stat uses
  it). **Trigger:** next feature touching the trace field parser.
- **Per-section context_breakdown** — unchanged from 035.

## Design decisions (locked)

1. **The dialogue is the UI chat store** (`conversation.json` `messages`) with the explicit role
   map above — it is what the user saw, which is what the cornerstone report promises. The
   per-turn dumps are the wipe-run fallback and stay the per-turn stats channel.
2. **Trace parsing extends the existing plain-text section parser** — no trace format changes; the
   whitelist/coercion/cost-rule edits in Goal 2 are the named cost of that choice.
3. **`render_strip` = per-sample replay through the export-isolation semantics** (fresh behavior
   per sample); time-pure nodes render directly. Never the live-tick path for sampling.
4. **`judge.py` + `analyze.py` must import WITHOUT GL/display side effects** — guaranteed by the
   lazy `__getattr__` in the package init, and pinned by a test (the import creates no
   `runs/data-*` dir).
5. **Template slots follow `{{AUTO:...}}`/`{{HUMAN:...}}`**; a coverage test pins that every
   `{{AUTO:k}}` in the template has a producer key in `_auto_fields` (an unmatched placeholder
   currently survives into the report silently).
6. **One report per scenario data-dir**; `--scenario` names it.
7. **No engine/copilot code.** Data the trace lacks → the slot is HUMAN or labeled-partial in v1 +
   an Out-of-scope trigger.

## Files touched

- `scripts/dogfood/harness.py` — `render_strip`, `script_values`.
- `scripts/dogfood/analyze.py` — verdict/vision extraction, parser edits, cutoff/giveup terminal
  fixes, honesty AUTO slot, `--dialogue`, `--project`, `--scenario`.
- `scripts/dogfood/judge.py` — NEW.
- `scripts/dogfood/__init__.py` — lazy `__getattr__`.
- `scripts/dogfood/REPORT_TEMPLATE.md` — the five-axis + dialogue sections.
- `scripts/dogfood/scenarios/README.md`, `scripts/dogfood/README.md`,
  `.claude/skills/dogfood/SKILL.md` — per Goal 7.
- Tests — `tests/test_dogfood_analyze.py`: verdict rows/finals/parse-rate (falsifiers, one reason
  each: an `unclear`-verdict row with a strict ask_line COUNTS as parsed while a garbled ask_line
  does NOT; a `vision_ok: False` row counts as a blind look; cost sums across TWO synthetic
  transcript segments; a producer round-trip through the real `TraceLog.event("ask_verdict", ...)`
  — the in-tree round-trip precedent — so emitter drift is caught); `cutoff=`/`clean_streak_giveup`
  terminal classification; `--dialogue` role mapping incl. the `error` row (cut the error mapping ⇒
  red) + the dumps fallback; the template-coverage pin (D5); NEW `tests/test_dogfood_judge.py`:
  synthetic-image numbers for all seven primitives (one test per primitive), plus the no-side-
  effect import pin (D4).

## Manual verification

Deterministic (headless): the test modules above; `import scripts.dogfood.analyze` creates no
`runs/data-*` dir and loads no glfw/moderngl.

Behavioral — **SESSION-LOCAL** (the fixtures live in gitignored `runs/`, scheduled for the
standard key-hygiene purge; the durable equivalents are the committed synthetic fixtures above):
- `analyze.py <data-s08> --scenario 08_mixed_grid` fills the honesty slot with the pilot's real
  verdicts — expect: 3 engine looks across 3 transcript segments, **all three `not-met`**, all
  `vision_ok: True`, engine-look spend **$0.00678**, parse rate 3/3.
- `analyze.py <data-s08> --dialogue` prints the 08 dialogue: 3 User + 2 Copilot + 1
  Copilot [engine] (the turn-1 churn-brake note — its presence IS the check) + 2 dim
  `[engine] ...` tool-status lines (8 printed rows total, not 6).
- `render_strip([0, 0.8, 1.6, 2.4], node_id=<flag node>)` on `proj-pzodxi57` (pure GLSL)
  reproduces the hand-rolled flag strip in one call; `render_strip` on the 05 bounce node
  (`proj-n2i58u0a`, stateful script) shows the drop-bounce-rest arc matching the delivered mp4 —
  the live-tick bug's falsifier.
- `script_values([0, 0.5, 1.0, 2.0])` on the bounce node returns a descending-then-resting
  `u_ball_center.y` series consistent with the pilot's measured 17→260 px trajectory.

## Open questions for the user

None blocking. Judgment calls a reviewer may challenge: `error`→"Copilot [engine]" labeling;
per-sample replay cost for long strips (t=10s @ fps=30 ⇒ 300 ticks for the last sample — python
ticks are cheap; acceptable).

## Review history

**Round 1 (2026-07-27, 2 Opus reviewers: correctness/design, verification/blast-radius): PARTIAL /
PARTIAL — both would block v1; all findings folded into this v2.** Blockers: `render_strip` on
`render_at` doesn't sample stateful sims (live single-tick of a persistent instance; the pilot's
bounce script only tolerated it by self-deriving dt) → per-sample replay via the export-isolation
seam; `--dialogue` as written crashed (data-dir vs project-dir) and its user/assistant filter
silently dropped the `error`-role churn-brake reply — the exact under-claim evidence the axis
exists for → dumps-resolved project dir + explicit role map + wipe fallback; DD1's
"what-the-model-saw" claim was FALSE (conversation.json is the UI store — reframed as the point).
Also folded: fixture-falsified expectations (3× not-met, $0.00678, not "mix"; 3+2+1 dialogue, not
"4 messages"); the package-init import side effect (glfw/moderngl + a `runs/data-*` dir with a
live key on EVERY pytest run) → lazy `__getattr__` + a no-side-effect pin; analyze's key whitelist
/ bool coercion / non-`_USAGE_RE` cost line / cross-segment sums named as required edits;
`clean_streak_giveup` + `cutoff=` terminal classification pulled IN (the honesty axis's own turn
class glyphed clean); parse-rate defined (strict ask_line ÷ vision_ok rows); per-look rows + final
verdict (056 semantics); canvas save/restore + renders_dir placement + t-labels for the strip;
`script_values` passthrough pulled in (infra gap #3 was silently dropped); template coverage pin
(unmatched `{{AUTO:}}` survives silently); one-report-per-scenario + `--scenario`; docs blast
radius widened (stale 06/07 table, contract sentence, SKILL section counts, scripts README).
Rejected: none — every finding reproduced on first-hand evidence (both reviewers ran the real
analyzer/fixtures).

**Round 2 (2026-07-27, 1 Opus convergence checker): PARTIAL → lock after one sentence.** 9/9
round-1 folds confirmed against code/fixtures (spend exact to the cent; every parser-edit claim
re-verified as genuinely required; the replay seam's exact call shape found in-tree). One HIGH
fixed in this text: `clear_context()` re-saves an EMPTY store (never deletes), so the dumps
fallback triggers on missing-or-empty, with the archive file as a second source. Nits folded:
the 08 dialogue prints 8 rows (incl. 2 tool-status lines); parse-rate `n/a` on a zero
denominator. Implementation nit carried to the implementer: `dry_run` takes a TUPLE of sample
times. Spec LOCKED (maintainer-delegated autonomous lock).

**Post-impl (2026-07-27, 2 Opus reviewers: correctness, spec-fidelity+docs): PASS / PARTIAL → all
findings fixed in one 16-item batch by the implementer.** Both reviewers reproduced every
Manual-verification number on the real fixtures. Notable fixes beyond the spec letter: the ruff
B009 invisible-until-staged trap; `render_strip` restored `node.uniform_values` too (the canvas
leak's twin — `tick_export` writes into the live node and `dump()` persists it; falsifier:
node.json byte-identical after strip+dump); the accidentally-load-bearing MESA overrides moved to
an explicit `tests/conftest.py` setdefault (bare `pytest tests/` is green again, 648 passed);
`clean_streak_giveup` reclassified ⚠️-limit (not 🔴-failed — the turn delivered its reply; s08's
headline went "1 failed turns"→0); `_load_dumps` sorted chronologically at the source (t10-vs-t2
lexicographic scramble also affected the cost series); `_resolve_project_dir`'s `Path("")`→cwd
hole; dumps-fallback shorter-store branch now tested; sanitized env in the subprocess test (no
live key on disk); `__dir__`-only discoverability (pyright rejects `__all__` on a lazy name);
strip filename carries the time range. Implementer deviation (guarded mkdtemp) ACCEPTED by both
reviewers — it kills the stray-empty-data-dir class at its root.
