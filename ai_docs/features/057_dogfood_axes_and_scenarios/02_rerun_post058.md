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
