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

Maintainer decision (2026-07-30): NO amendment. Closed form is a legitimate route for simple
tasks; one-shot quality is the USER's responsibility via a tighter spec (if they want a script,
they say so — as the original hinted 05 does). The watershed text stays as landed. The de-hinted
route flip is recorded as expected-behavior, not a defect.

## Non-watershed observations

- First-shot lesson application is weaker at effort=none across the board (03 control: clipped
  circles; 08 gate: gray bg + colored lines) — corrections converge to baseline quality.
- Plan-loop persists on compound asks (13 gate needed one "no more plans" push).
- All engine-side pathologies (12k starvation, plan-loop) are pre-existing effort=none behavior,
  not wave-A effects: 04/05/08 routing and quality track their controls.

# Wave-B gates (prompt 15517: SCRIPT API block, dedup cuts, conventions moves)

| gate | verdict |
|---|---|
| 04 orbit | PASS one-shot, no script, period exactly 2.0s (expensive turn — reasoning variance, content clean) |
| de-hinted 05 | PASS, 3 msgs, $0.037 — routed to a SCRIPT this time (decaying bounces, rest); the SCRIPT API block in RARE plausibly raised script salience vs wave-A's 9-msg GLSL grind |
| 08 grid | PASS with ZERO corrections (near-black bg, gray lines, 9 figures, blink verified anti-aliased at 0.25s sampling) — best 08 result across all prompt versions |

No cut-surface regression observed; the D5 coverage is additionally pinned by the marker tests.

# Wave-C gates (prompt 14435 + 500-char legend spliced per turn)

| gate | verdict |
|---|---|
| 04 orbit | PASS one-shot, $0.005, period exactly 2.0s; legend rides the turn exactly once (trace-verified) |
| 03 circles | PASS with 1 correction (edge-tangent circles -> margins 35/35, widths 5x51, gaps 18-19) — best 03 across all versions |

059 complete: prompt 20507 -> 14435 (-29.6%), all three waves landed with green echelon gates.
