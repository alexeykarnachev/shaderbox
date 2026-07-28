# Post-058 cornerstone re-run (2026-07-28) — six-axis results + mined code findings

The first run of the cornerstone set on the vision-less, code-first copilot (058), same cheap
model (`gpt-5.1-codex-mini`), GPU box. Dialogues + renders/videos delivered in-chat; this file is
the durable synthesis. ALSO the 058 validation run: ZERO vision calls in every trace.

## Results vs the pilot (2026-07-27, pre-058)

| Scenario | Pilot | Re-run | Cost (turn 1) |
|---|---|---|---|
| 03 five circles | PASS, 3 msgs (layout see-saw) | PASS, 3 msgs (SAME see-saw class) | $0.004 vs pilot $0.057 |
| 04 orbit 2s | PASS, 1 msg | PASS, 1 msg (period 2.000s) | $0.0035 |
| 05 bounce script | PASS at budget, ball invisible for 2 turns | PASS, one go-ahead, textbook physics first try | $0.010 |
| 08 mixed grid 3x3 | 3 fails (y-flip, dead pulse, wrong blink rate) | **11/11, ZERO corrections** — the model self-handled the y-flip (`2 - int(cell.y)`) | $0.02 |

Costs dropped ~3-10x on simple asks (no auto-look rounds, no vision spend, leaner endings).
Honesty: no over/under-claims anywhere. Sweep turns: behavior-neutral everywhere (pixel-verified),
removed real sediment in 03/04/05 (unused engine uniforms, stray indentation, leftovers); 08's
sweep restructured slightly (+3 net lines).

## CODE axis — what the copilot's sources actually look like

GOOD (consistent across all four): parameterized layout math (derived gaps/steps, not magic
positions), smoothstep AA on every edge, factored helpers when needed (08: `rotate()`, `PI`
const), correct tool choice every time (per-pixel GLSL vs stateful script), clean stateful script
(05: dt-guarded Euler, explicit `at_rest`, restitution).

WEAKNESSES (the next wave's candidates, each seen at least once):
1. **Layout see-saw (repeated, 2 runs)**: asked to fix margins, breaks gaps; asked to fix gaps,
   breaks margins — never solves the one linear equation (N circles, N+1 spaces). A craft/prompt
   candidate: "layout = solve all constraints at once, don't iterate one at a time".
2. **Aspect-correction inconsistency**: 04 used `u_aspect` properly; 03's sweep DELETED the unused
   `u_aspect` instead of USING it (circles round only on a square canvas). The engine-provided
   uniforms' PURPOSE isn't internalized.
3. **Naming style drift** within one file (`circleCount` + `circle_mask`).
4. **Minor logic duplication** (05: two overlapping rest thresholds).
5. Pre-sweep sediment always exists (unused uniforms, stray indent) — the sweep turn reliably
   finds material, so keep it standing.

## Process notes

- Plan-first fired on 05/08 (complex asks) even with "just build it" absent; "Go ahead" turns are
  cheap ($0.003). Fine as-is.
- The judge tooling paid off end-to-end: `script_values` proved the bounce parabola numerically,
  `judge.py` angles measured the 3.000s rotation period AND direction, per-cell `region_diff`
  verified every motion; `farthest_bright_angle` needs off-symmetry sample times for square/cross
  shapes (corner ties at poses like -45deg) — judge-tooling note, not a defect.

## Fix wave (2026-07-28, same session) — experiment-gated

Of the four mined weaknesses, two were dropped without code (naming drift, minor duplication —
better-model territory, zero functional harm) and two went through necessity+actionability
experiments:

- **u_aspect (weakness 2): FIXED, A/B-validated.** Necessity: "a centered circle" on a 640x360
  canvas rendered as an ellipse tracking the canvas (bbox aspect 1.782) — the prompt only said
  u_aspect EXISTS, never when it's required. Fix: ONE generalized line in the `_CONVENTIONS`
  block ("the canvas is NOT square in general … aspect-corrected coordinates before the shape
  math"). Re-run: aspect 1.000. Marker test guards the line (`test_craft_prompt.py`).
- **Layout see-saw (weakness 1): fix candidate KILLED by its own experiment.** With a
  LAYOUT-IS-ONE-EQUATION craft line present, turn-1 was WORSE (margins 0/1) and the
  single-symptom correction still didn't re-solve (margins 1/2, gaps 33→13 — the edit missed the
  complaint entirely). Verdict: cheap-model capability limit, not a knowledge gap; the line was
  REMOVED (no unproven prompt tax). **Trigger to revisit:** the default model changes, or the
  see-saw appears on a stronger model.

**New scenario `09_implicit_affordances`** (maintainer-designed): a 640x360 fixture + a stopwatch
ask where `u_aspect`/`u_time`/`u_resolution` are each physically REQUIRED but never named. First
run: ONE-SHOT PASS — dial aspect 1.000, tick strokes median 1.05px (asked "exactly one pixel"),
hand period exactly 60.0s, all three uniforms used idiomatically in source (incl. a
`max(min(res.x,res.y),1.0)` guard). One cosmetic miss logged: an unrelated fixture run had
inverted fg/bg colors once (single instance, no class).

## Base-echelon stability sweep (2026-07-28, 3 one-shot runs per scenario, no corrections)

The maintainer's bar: the base set must WORK before harder scenarios are added. One-shot pass
rates (cheap model; corrections would fix most fails — this measures raw reliability):

| Scenario | One-shot result | Classes seen |
|---|---|---|
| 03 circles | count/visibility 3/3 AFTER the domain amendment (was 1/3: hard-coded +-0.6/0.8 layouts fell off the SQUARE canvas after aspect correction); balance ~1/3 | layout balance = known model limit |
| 04 orbit | 3/3 exact (period 2.000s each) | rock stable |
| 05 bounce | 3/3 (a claimed floor-penetration was RETRACTED — a judge error, see below) | — |
| 08 grid | builds 5/6 (1x plan-loop: replied with a SECOND plan to "go ahead"); rows correct 2/2 built runs AFTER the uv-y line (1/3 before); per-cell motion bugs in most runs (static red square 2x, wrong blink rate 2x, one static magenta, one empty cell) | compound per-cell reliability is the base gap |
| 09 stopwatch | 1/3 one-shot, +1 after go-ahead, 1 stuck (token-limit turn, failed continue) | of the BUILT runs: dial aspect 1.0 in 3/3, hand period correct 2/2 measured |

**Prompt-line validations (all experiment-gated):**
- Aspect line + DOMAIN amendment ("x spans +-u_aspect/2, derive layout from the live range"): the
  initial line alone CAUSED off-frame layouts on square canvases (religious correction + wide-canvas
  constants) — caught only by the sweep, fixed by the amendment; zero ellipse/off-frame cases since.
- uv-y direction line ("y=0 is the BOTTOM; 'top row' = high y"): revived by sweep evidence (the
  earlier minimal-ask non-repro was misleading — the GRID context reproduces 60%); 2/2 correct after.
  Both lines carry marker tests in `test_craft_prompt.py`.
- 058 honesty observed live: a token-limited turn correctly reported "the shader source is
  unchanged from the start of this turn" — no fabrication.

**Open classes for the next wave (evidence-ranked):** (1) compound per-cell motion reliability
(blink rates, cells going static) — the dominant base-echelon gap; (2) plan-loop on go-ahead (1x,
watch); (3) layout balance (model limit, revisit on model change).

**JUDGE LESSON (the sweep's most expensive mistake, 2026-07-28):** the "floor penetration" class
was a JUDGE artifact — `script_values` returns RAW script values whose coordinate frame is
unknowable without reading the shader (that run's script+shader shared a centered frame; the ball
was on-screen the whole time). Three symptom corrections gaslit the model about a non-existent
bug, and a candidate SCRIPTING prompt line was added and then removed when the retraction landed.
THE RULE: judge POSITIONS from PIXELS (renders/strips — frame-independent truth); use raw script
values only for rates/periods/monotony, never for on-screen-ness. Filed in the dogfood skill.
