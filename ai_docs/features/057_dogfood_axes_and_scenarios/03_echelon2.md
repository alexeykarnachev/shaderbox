# Echelon-2 runs (2026-07-28): 10 pong, 11 die3d — results + mined classes

First runs of the second-difficulty echelon (scenarios `10_pong.md`, `11_die3d.md`; dialogues +
videos delivered in-chat). Both FAIL-at-budget on ONE sub-requirement each, with everything else
landing — the echelon is within reach, not out of it.

## 10 pong — FAIL-at-budget on scoring (driver-asterisked)

Landed: field/ball/two AI paddles/score-dot rows, continuous motion, wall bounces, paddle chase
with lag, bounds fixed by correction 1. The score wiring is CORRECT end-to-end in code (script
increments on miss -> u_*_score uniforms -> shader lights dots) — but NO miss ever occurs: after
the model's own tuning the paddles (speed 1.6) outrun the ball (~1.1-1.3), so the "limited speed
so they sometimes miss" sub-ask is unmet in any watchable window. Asterisk: correction 2 was
burned by a DRIVER judge artifact (see the judge lesson) and its edits made paddles FASTER.

**Code findings:** (1) DEAD-STORE EDIT, twice — the model bumped `ball_vel` in `__init__`, which
`_reset_ball()` (called one line later) overwrites from its own speed formula; both "speed up"
edits had zero effect and were claimed as landed. The facts channel cannot catch pace changes
(frame-pair sampling is pace-blind), so the claim went unchallenged. (2) Otherwise clean state
machine: states, per-paddle chase helper (no copy-paste), score clamp, serve-angle variation.

## 11 die3d — FAIL-at-budget on pips

Landed: a real raymarched rounded cube, RIGID rotation (face-on <-> edge-on silhouette cycle),
directional light, soft floor shadow (landed on the final turn), fixed camera, checkered floor.
NOT landed in 3 attempts (build + full rewrite + targeted turn): the PIPS — per-face detail on a
rotating body (face-local UVs) is beyond the cheap model's reliable reach today. Honesty was
flawless throughout: two limit-forced turns self-reported "pips still aren't appearing and
there's no visible shadow" against their own facts.

## Cross-echelon observations

- `turn_time_budget_s=180` splits big builds into 2-3 honest turns ("Continue" tax) — works as
  designed; every limit-forced reply was truthful, zero over-claims across both scenarios.
- Plan-first fired on both (go-ahead turns cost ~$0.002).
- JUDGE LESSON #2 (filed in the skill): `render_at` series on a STATEFUL script measures live
  single-ticks, not time — a pong ball read 70x too slow, and the resulting "speed up" correction
  gaslit the agent. Replay pixels (`render_strip`) are the only time-faithful source for scripted
  nodes; raw `script_values` are for rates/counters only (their coordinate frame is unknowable).

## Next-wave candidates (evidence-ranked)

1. **Dead-store edits claimed as landed** (2x one run): the model edits a value its own code
   overwrites downstream. Candidate DATA-channel fix: script-write feedback could echo the
   EFFECTIVE sampled values of the edited names (dry_run already samples them) so a dead store is
   visible immediately. Needs a necessity+actionability experiment before building.
2. **Per-face detail on rotating 3D** (pips, 3 failed attempts, self-acknowledged): craft-block
   candidate (face-local UV recipe) — but the see-saw precedent warns craft lines may not lift
   the cheap model; experiment first.
3. **Emergent-behavior blindness** (paddles never miss): the model cannot observe event-level
   outcomes of its sims. Candidate: a model-facing script probe tool (the driver-side
   `script_values` exists; the MODEL has no equivalent) — a real feature decision, maintainer's
   call, cost/complexity nontrivial.
