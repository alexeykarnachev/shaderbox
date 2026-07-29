# Wave-A gates: new prompt @ effort=none vs the 02_controls baselines (2026-07-29)

| gate | verdict | vs control |
|---|---|---|
| 04 orbit (must NOT script) | **PASS** one-shot, no script, period exactly 2.0s, $0.009 | same |
| de-hinted 05 (routing falsifier) | **visual PASS / routing CHANGED** — 9 msgs, $0.25, 2 corrections, pure-GLSL closed form | control: script, 3 msgs, $0.02 |
| 08 grid (all time-pure) | **PASS**, 2 corrections (gray bg instead of near-black; colored separator lines), no script, timings exact | control-era: 1 correction |
| 10 pong (routing probe) | **PASS routing** — game logic in script on the first build; paddles-never-miss persists (model class, present in control too) | same class |
| 13 console (mixed routing probe) | **PASS routing** — boat logic scripted, sonar/mine GLSL | same |

## The one substantive finding: the de-hinted-05 route flip

On the corrected watershed the model classified a damped bounce as a PURE FUNCTION OF TIME and
built it as closed-form GLSL (piecewise segments summed from u_time). That is formally legal —
the closed form exists — but the cheap model could not execute it: the ball fell through the
floor for 3 grinding turns (honestly self-reported each time) before landing a working version.
The watershed text DOES list "a collision response" as state; the model missed it. Control run
(old prompt, "physics -> script, never faked with GLSL time math"): script, PASS in 3 messages at
1/10th the cost.

Candidate amendment (NOT applied — maintainer call, anti-overfit rule): strengthen the watershed's
state list with the practicality clause, e.g. "bounces/collisions/friction are STATE even though a
closed form may exist — simulate in a script rather than reconstructing segment math in GLSL."
Risk of applying: pushes borderline time-pure asks back toward scripts (the old bias). Risk of
not: this exact class (cheap model grinding on closed-form physics) recurs.

## Non-watershed observations

- First-shot lesson application is weaker at effort=none across the board (03 control: clipped
  circles; 08 gate: gray bg + colored lines) — corrections converge to baseline quality.
- Plan-loop persists on compound asks (13 gate needed one "no more plans" push).
- All engine-side pathologies (12k starvation, plan-loop) are pre-existing effort=none behavior,
  not wave-A effects: 04/05/08 routing and quality track their controls.
