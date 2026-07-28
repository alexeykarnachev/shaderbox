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

## Fix wave (2026-07-28, same session)

- **Dead-store class: FIXED engine-side, the data-channel way.** Script edits now get the shader
  no-op's structural twin: `_apply_script_text` compares the clean dry-run samples against the
  node's previous clean edit and appends "this edit changed NOTHING in the driven values..." when
  identical (`backend._last_script_samples`; unit-tested both directions in
  `test_content_editing.py`). Deterministic truth on the existing channel — no prompt tax.
- **Live coda:** re-asked the pong "Double the ball speed" — the model went STRAIGHT to the
  effective formula in `_reset_ball` this time (the no-op line never needed to fire), replay
  confirmed 2x speed, and with the faster ball the paddles finally MISS: the first score dot
  lights at t~40s (brightness 32 -> 186). The full game loop — pace, misses, scoring, display —
  works end-to-end. (Original run stays FAIL-at-budget in the books; this was a post-budget
  experiment turn.)
- **Pips (per-face 3D detail): DEFERRED with trigger** — 3 failed attempts with focused
  corrections say a single craft line won't lift the cheap model (see-saw precedent); revisit on
  a model change or if the next echelon needs it.
- **Model-facing script probe tool: maintainer's decision pending** — tool-count tax vs
  emergent-behavior blindness; not built.

## Prompt-education wave (2026-07-28, maintainer-directed: teach via prompt, no new tools)

Directive: maximize the weak model's LOWER BOUND through a concise-but-strong prompt; no new
tools/features (surface area, bugs, support); tests stay diversified and mutually perpendicular.

- **3D & LOCAL FRAMES lesson added to VISUAL CRAFT** (~9 lines + 1 formula; generalized: local
  frame per object, inverse-transform the sample point, surface detail via the dominant-axis face
  pick — pips/digits/panels are examples, not the rule). Fixture validation (fresh die run):
  the model now CARVES PIPS IN THE CORRECT LOCAL FRAME (never did in any pre-lesson run) — but
  they stayed invisible: it centered pip spheres against the un-inflated box half-size while
  `sdRoundBox(b, r)`'s surface sits at b+r, burying the pips 0.09 under the surface, and two
  symptom corrections couldn't surface them. Class progressed from "no per-face concept" to
  "SDF-primitive semantics slip + can't debug buried geometry" — a cheap-model limit; trigger:
  the stronger-model pass.
- **Scenario 12_radar added** (the maintainer's perpendicular-axis directive: polar/radial 2D) and
  run: PASS with 1 correction — sweep period exactly 4.0s, DIRECTIONAL afterglow (trailing sector
  +47% brighter, measured), blips, clean dark scope after one look-correction. Polar craft needs
  no lesson at this tier.
- **Prompt size audit:** 20,507 chars (~5.1k tokens) TOTAL system prompt — net +435 chars vs the
  pre-058 baseline despite three conventions lines + the 3D lesson (the vision removal paid for
  the education). The "concise but strong" bar holds.
- Die bookkeeping: 2/2 runs FAIL-at-budget on pips (pre- and post-lesson), everything else lands;
  the lesson measurably moved the failure class inward.
