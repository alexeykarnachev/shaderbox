# 057 — Dogfood axes & ground-truth scenarios: research + cold-start brief

Pre-spec research note (2026-07-27, written for a FRESH session to pick up after `/clear`). Feature
NOT yet spec'd or locked — this file is the fact base + the proposed shape + the open questions.
Nothing here is implemented. Read this, then draft `01_spec.md` in this dir and plan-lock with the
maintainer.

## Where we are (context)

Feature 056 (copilot convergence & robustness) is DONE, committed+pushed on dev (`3f0d380`), spec:
`ai_docs/features/056_copilot_convergence_and_robustness.md` — its Review history holds the
micro-dogfood numbers AND the two open copilot-side findings that motivate 057:

- the eye is blind to MOTION COHERENCE (the flag's canton waved independently of the stripes; two
  aimed not-met verdicts missed it — they were about stars/dullness);
- the craft block lacks "one object = ONE deformation field".

The maintainer's direction: BEFORE fixing those, build the measuring stick — a system of dogfood
AXES and ground-truth-checkable scenarios (the all-axes-at-once flag can't attribute a failure).
Copilot-side fixes are the feature AFTER 057, validated against 057's fixtures.

Also landed this session, relevant to any run on this box: **WSL renders on the real GPU** with
`GALLIUM_DRIVER=d3d12` (RTX 3090; gotcha filed in `/dogfood` skill §0). `OPENROUTER_API_KEY` is in
`~/.bashrc` here (interactive-shell footgun applies, skill §0).

## The scenario system as it IS (verified first-hand 2026-07-27)

- **Contract** (`scripts/dogfood/scenarios/README.md`): scenarios are free-text goal-driven
  MISSIONS a Claude drives LIVE (one blocking `uv run` per turn, resume/dump on disk), never
  numbered step-scripts, never auto-run, **no code assertions** — the judge is the driving Claude
  reading the trace + opening PNGs. Two scenarios exist: `01_shape_gallery` (tool sweep + context
  wipe, glance-judgeable 2D shapes), `02_logo_design` (creative iteration, human eye via
  SendUserFile). The README's own "Next (harder, later)" hook anticipates exactly this feature.
- **Harness** (`scripts/dogfood/harness.py`): `create/send/drive_until_idle/approve/decline/
  render/render_at/export_at/render_video/render_video_mp4/clear_context/reload/dump/release`,
  plus `nodes`, `trace_path`, `session_cost_usd`. NO frame-strip helper (hand-rolled 3× in the
  043 run, TODO'd there, still open). No judge-side numeric probe of script-driven uniform values
  (needed for logic ground truth — likely a thin wrapper over the engine's `dry_run`; verify).
- **Analyzer** (`scripts/dogfood/analyze.py` CLI: `target/--json-out/--md-out/--template/
  --report-out/--model`): fills the report template's AUTO slots (per-turn table, tool coverage,
  token/cost, recoveries). NO extraction of: 056's `ask_verdict` trace events, vision look
  counts/spend, per-file edit streaks, fidelity checklists. (056 deferred verdict extraction with
  the trigger "a second dogfood metric wants the same extraction" — 057's axes ARE that trigger.)
- **Report template** (`scripts/dogfood/REPORT_TEMPLATE.md`): AUTO+HUMAN slots as above; HUMAN =
  verdict, per-render eyeball, honesty, 3 TODO buckets. No fidelity/motion/logic slots.
- Doc drift (fix in 057's wave): scenarios README says reports go to
  `026_dogfood_report_<run>.md`; the skill (canonical) says `NNN_dogfood_report_<run>.md`.

## Failure-axis taxonomy (mined from ALL dogfood reports; full table relayed from a review agent —
## re-verify any line before building on it)

CLOSED by 050-056 (don't re-solve): facts-vs-narrative over-claim, blind done-claims, unaimed/
sycophantic/wall-clock eye, model opting out of the engine look, comment-splice duplication, churn
brakes (shader AND script), auto-look mistargeting, working-set hygiene, misleading error paths,
invisible vision spend.

OPEN and clustering into 057's motivation:

1. **Motion/animation: ZERO coverage.** No scenario judges animation; `render_video` cold in every
   recorded run; no strip helper; the eye's 3-frame strip answers "does it change", not "do the
   regions move together" (axis C7 — the flag case).
2. **CPU logic / scripting: no repeatable scenario.** Both 043 runs were ad-hoc; script trio /
   dry_run / motion verdict / orphan-key steer have no scenario home.
3. **Instruction fidelity is not counted.** "50 stars" is now honestly REPORTED unmet (056) but no
   template slot counts sub-requirements landed (X of Y).
4. **Quality axes live as prose** (palette drift, over-reach, precise-diagnosis dependency — all in
   scenario 02, unmeasured).
5. Residual watches: `old_str` mismatch loop on long files (035, no dedicated fix); one-giant-write
   budget blowout (054, prompt-only); model over-generalizing a nudge into a false rule (050 TODO,
   parked); `set_uniform` resume trap (scenario 02, documented not fixed); 052 workspace tools +
   publish tools have no scenario reason to ever fire.

## Proposed 057 shape (NOT locked — discuss at plan-lock)

**Axes** (every run reports all five): `fidelity` (counted sub-requirement checklist) · `motion`
(strip/video vs known ground truth: period, trajectory, cross-region coherence) · `logic` (numeric
sript-value checks vs analytic truth) · `honesty` (ASK-verdict parse rate, over-claim count — data
already in the 056 trace) · `process` (tool coverage/cost — already AUTO).

**Scenarios** (ground-truth ladder, each isolating ONE axis cluster; sketches discussed with the
maintainer 2026-07-27 in-chat):
- `03_bounce2d` — ball(s) + gravity in a box, physics in `script.py`; analytic trajectory = logic
  ground truth; parabola visible in a strip = motion ground truth; eye checks are binary/pointable.
- `04_pong` — self-playing pong (state machine, discrete events, score); first scenario that
  legitimately heats `render_video`.
- `05_spin3d` — rotating SDF primitive, one light + shadow; known period/silhouette; minimal 3D
  craft without cloth/layers. (Maintainer question open: cube vs die-with-pips — pips give counted
  ground truth.)

**Infra prerequisites** (do BEFORE or WITH the scenarios — this is the "is the infrastructure
ready" answer: NO, three gaps):
1. `h.render_strip(times, size)` in the harness (unblocks motion judging; 043 TODO).
2. `analyze.py`: extract `ask_verdict` events + vision look count/spend; new template slots
   (fidelity checklist, motion/logic verdicts, ASK stats). Its deferral trigger has fired.
3. A judge-side numeric script probe (assert script-driven uniform values against analytic truth
   without an LLM turn) — likely thin; confirm `dry_run` reachability from the harness.

**Out of scope for 057** (the feature AFTER, validated against 057): the eye's motion-coherence
dimension; the craft-block "one deformation field" lesson; any copilot prompt/engine change.

**Open questions for plan-lock:** (1) all three scenarios or bounce2d first? (2) spin3d: cube or
die? (3) quality axes D4/D5 (over-reach, palette drift) — keep as prose in 02 or invent a counted
signal? (maintainer leaned unanswered; agent's default: prose for now).

## Cold-start checklist for the next session

1. `CLAUDE.md` chain (roadmap banner points here) → this file → `/dogfood` skill →
   `scripts/dogfood/scenarios/README.md` + both scenarios end-to-end.
2. Re-verify the relayed taxonomy lines you build on (provenance: review-agent, 2026-07-27).
3. Draft `01_spec.md` (Goal/Out-of-scope/Design decisions/Files touched/Manual verification per
   dev_flow), plan-lock with the maintainer, then the usual review loop (the maintainer's standing
   preference this arc: reviewers/implementer on Opus agents, moderator on Fable).
