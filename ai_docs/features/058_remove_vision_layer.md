# 058 — Remove the vision layer entirely; facts-based forced-reply honesty

Maintainer decision (2026-07-28, after the validation experiments): the vision eye is out —
ShaderBox is a HUMAN+copilot tool, the human is the visual judge, and necessity of a vision model
for the interactive loop could not be demonstrated (its real catches are either covered by the
free NUMERIC facts line — blank/FLAT frames — or by the user's own eye in one message). No
default-off half-measure: the layer is deleted whole, no dead code, no compat shims.

**The product thesis this locks in (maintainer, verbatim intent): the copilot's job is to write
EXCELLENT CODE for visual applications; visual tuning is the HUMAN's job.** Consequently the
dogfood's first-class question becomes "is the produced code good?", not "is the frame pretty?".

What this feature does: (1) DELETE every vision-model tendril (053's eye + 056's convergence
loop); (2) ADD the one validated vision-free fix — a limit-forced final reply carries the CURRENT
measured render facts, closing the three observed over/under-claim incidents without any billed
call; (3) ADD the **CODE axis** to the dogfood report — the driver reads the run's final sources
(shader + script) end-to-end and grades them as a code reviewer: dead code, duplication, edit
sediment, structure/naming, needless complexity, idiomatic GLSL/Python, right tool for the job
(per-pixel→GLSL, stateful→script). The end-of-mission SWEEP turn (skill §1a) is this axis's
standing probe: "remove dead code and leftovers; change no behavior" — what the sweep removes IS
the sediment measurement.

## What DIES (the vision layer)

- `llm/openrouter.py`: `describe_image`, `_VISION_SYSTEM`, `fetch_model_image_support`, vision
  usage plumbing.
- `copilot/vision_contract.py` — the whole module (ASK verdict parse/strip).
- `copilot/vision_probe.py` — the Settings capability badge prober, whole module.
- `backend.py`: `_probe_png`, the vision cache, the 3-frame strip, the `describe_image` closure
  dep, vision branches of `probe_render` (incl. the `(vision look unavailable)` suffix);
  `ProbeResult` reverts to a plain `str` capability return (the struct existed to carry
  verdict/vision_ok/usage — all dead).
- `agent.py`: the ENTIRE turn-end auto-look / convergence block (`looks_used`, the since-index
  look gating, `_auto_look_fact`, `_turn_intent_look_for`, `_eye_summary_line`, `eye_note`,
  `TurnSummary.eye`, the engine-look `AgentToolCard`, `ask_verdict`/`engine_look_usage` trace
  events, the vision-usage cost fold at both call sites).
- `session.py`: the engine-look visible line branch; `_render_summary`'s eye field rendering.
- `project_session.py`: the `describe_image` closure wiring.
- `config.py`: `copilot_vision_enabled`, `copilot_vision_model`, `copilot_vision_probe_size`,
  `copilot_vision_max_tokens`, `copilot_convergence_max_looks`, `vision_models_fetch_timeout_s`,
  `auto_look_intent_max_chars`, `eye_summary_max_chars`.
- `exporters/integrations.py`: `vision_enabled`, `vision_model`, `copilot_convergence_max_looks`
  fields + their `apply_user_limits` legs.
- `popups/settings.py`: the whole Vision block (model field, badge, status slot) + the
  convergence limits row.
- `prompt.py`: every eye/vision sentence — the `probe_render` vision paragraph ("VISION look…
  WITNESS not a judge…"), the aimed-look mentions, `_clean_streak_fact`'s "the engine's eye
  checks the frame at turn-end" clause (reverts to the pre-056 wording: the user sees nothing
  until they look). `probe_render` stays documented as the NUMERIC probe at a chosen t.
- `tools/inspect.py`: vision fields of the payload; the description loses vision text.
- Tests: `test_vision_probe.py`, `test_vision_auto_look.py`, `test_vision_contract.py` DELETED;
  vision/convergence cases stripped from `test_copilot_loop.py`, `test_copilot_error_paths.py`,
  `test_copilot_user_limits.py`, `tests/_caps.py` (probe signature reverts to `str`), any
  `_model_cards` engine-look filtering simplifies away.
- Dogfood tooling (057, same wave): `analyze.py` verdict extraction (`ask_verdict` /
  `engine_look_usage` parsing, look table, final verdicts, parse rate, engine-look spend) — the
  emitters die, the parser follows; the honesty template section becomes HUMAN
  (claims-vs-facts/claims-vs-user-eye) + the AUTO limit-forced-turns list (cutoff extraction
  STAYS — it is not vision). `REPORT_TEMPLATE.md`, `scenarios/README.md`, `scripts/dogfood/
  README.md`, `.claude/skills/dogfood/SKILL.md` swept of eye/verdict/look_for references.
- `todo.md` [VERIFY] entry: the vision-badge and engine-look-line checks dissolve; the entry
  shrinks to the surviving live-UI items (handoff line, turn-time Settings row, liveness
  counter, dev-box `make test`).
- Roadmap: 053 row → **superseded** (points here); 056 row stays **done** with its convergence
  half noted superseded by this row (the robustness half — script brakes, working-set hygiene,
  truthful error paths, probe-clock — remains live); banner rewritten.

## What STAYS (explicitly — a remover must not overreach)

- **The NUMERIC facts pipeline** (no LLM): `_render_facts_for`, `edit_hints.render_facts`, the
  facts line on every clean mutation, no-op detection, the STATIC/ANIMATES motion verdict, the
  script dry-run motion facts. This is the copilot's sight and it is free.
- **`probe_render(node, t)`** as the ungated numeric probe (feature 050) — returns the facts
  line for a chosen t.
- All 056 Half-2 robustness fixes (brakes, batch guard, working-set reset+LRU, E1-E4 error
  paths, handoff cards), `turn_time_budget_s`, the liveness counter, all 057 judge tooling
  (`render_strip`, `script_values`, `judge.py`, `--dialogue`, cutoff extraction).
- `_RunLog.last_render_target()` — re-consumed by the addition below.

## The ONE addition — facts-based forced-reply honesty

All three observed dishonest limit-endings (bounce "still isn't rendering" while fixed; blink
"remains incorrect" while fixed; blank flag "matches your request") happened on the forced-reply
path, which summarizes blind. Fix, engine-side, zero billed calls: when a turn is force-ended
(token budget, `max_iterations`, `turn_time_budget_s`, clean-streak hard stop) AND the turn ran a
successful render-authoring tool, the engine renders ONE fresh facts probe of the last-authored
(or current) node and splices it into `_final_reply_nudge`:
`"the frame currently measures: <facts line>"` + the standing instruction to state the net result
from THESE measurements, not from intentions. Data on the channel, no conscience rule, no vision.
`_RunLog.last_render_target()` picks the node (same fallback-to-current rule as before).

## Out of scope (each with a trigger)

- **Re-introducing any vision capability.** **Trigger:** the maintainer reverses the product
  call (autonomous/headless operation without a human judge becomes a real use case).
- **Facts-line enrichment** (shape counts, symmetry metrics). **Trigger:** a dogfood where the
  facts line's vocabulary is the proven bottleneck for a wrong claim.
- **uv-y prompt line / craft deformation line** — both failed necessity validation 2026-07-28
  (flip didn't reproduce; craft-line untestable under cheap-model variance). **Triggers:** the
  057 fidelity axis catches a y-flip recurrence; a run shows the model failing to fix a
  user-reported incoherence even with the report as data.

## Design decisions (locked)

1. **Removal is total** — grep-clean: no `vision`, `describe_image`, `ask_verdict`,
   `look_for`, `ASK_`, `auto_look`, `convergence` identifiers survive in `shaderbox/`
   (docs/specs are history and keep their text; roadmap rows get superseded pointers).
2. **`probe_render` reverts to the 050 numeric contract** (`str` facts). No struct, no payload
   vision fields.
3. **The forced-reply facts probe reuses `_render_facts_for`** (64px, t=0 export clock) — the
   existing free probe; no new render path.
4. **NO backward-compat**: config/integration fields are deleted outright (additive-default
   pattern covers old integrations.json files — unknown fields are... `extra: forbid` on
   `CopilotIntegration`! So deleted fields in a PERSISTED integrations.json would REJECT the
   load. Per the hard rule, fix the dev sandbox files BY HAND (delete the stale keys from the
   maintainer's `integrations.json` via the normal load+save path or hand-edit) and note it in
   the report — never a migration shim. A fresh user file never had them (pre-release).
5. **The dogfood honesty axis stays an axis** — its AUTO half shrinks to limit-forced turns; the
   claims-vs-reality half is HUMAN (the driver's pixel tools are the instrument).
6. **The CODE axis is HUMAN and source-anchored**: the report template gains a `code` section
   (`| aspect | verdict | evidence |` over: dead code / duplication / structure & naming /
   complexity vs task / tool choice), the driver quotes line evidence from the FINAL sources, and
   the sweep-turn's diff (what the sweep deleted) is recorded as the sediment number. The
   template's axis list becomes fidelity/motion/logic/honesty/process/code.

## Files touched

`shaderbox/copilot/{agent,backend,session,config,prompt,project_session... }` per the death list;
`llm/openrouter.py`; DELETE `vision_contract.py`, `vision_probe.py`; `tools/inspect.py`;
`exporters/integrations.py`; `popups/settings.py`; `shaderbox/project_session.py`;
`scripts/dogfood/{analyze.py,REPORT_TEMPLATE.md,README.md}`; `scripts/dogfood/scenarios/README.md`;
`.claude/skills/dogfood/SKILL.md`; `ai_docs/{roadmap,todo}.md`; tests per the death list + a NEW
test: the forced-reply nudge carries the facts line (spy on the probe; falsifier: cut the splice ⇒
red) for each force-end path.

## Manual verification

- `make check` 0 errors; `make test` full suite green (expect ~600 after test deletions).
- Grep gate (D1): zero survivors of the identifier list in `shaderbox/` + `scripts/`.
- Behavioral (one cheap run): a cornerstone one-shot (03 static) drives clean end-to-end without
  a vision call anywhere in the trace; a forced-end turn's final reply quotes the facts line
  (drive with a low `turn_time_budget_s` override).
- Maintainer live-UI (folded into the standing [VERIFY] entry): Settings no longer shows the
  Vision block; no layout gap.

Amendment (post-impl review): the facts clause rides only the three MODEL-facing forced-reply
paths (token budget, max_iterations, time_budget). The clean-streak hard stop is USER-facing
engine prose and the user can see the live render — raw telemetry there is noise; that path
carries no clause (pinned by test).

## Review history

**Post-impl (2026-07-28, 1 Opus reviewer): PARTIAL → all three findings closed same-wave.** The
removal verified genuinely total (D1 grep gate: 4 survivors, all the edit-convergence word sense
in 056 brake comments — the STAYS list; every STAYS item verified by reading current code; a
synthetic eye-era conversation.json loads clean; falsifiers of the new test file proven red by
two independent neutering runs). Closed: the missing 058 roadmap row; the D4 stale-keys note —
filed LOUDLY in the todo [VERIFY] entry because `IntegrationsStore.load()` fail-softs an
unparseable file to DEFAULTS and the next save would wipe creds (this box verified clean; Pi/dev
box must hand-delete the three removed keys before first launch); the clean-streak user-facing
telemetry — resolved by the amendment above. Also swept: two stale "eye" comments in
scripts/dogfood, the craft-test marker comment, the 057 row's pre-058 axis description.
