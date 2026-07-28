# Final exam — scenario 13 (submarine command console)

Run `proj-v4r16iqv` / `data-exam`, model `openai/gpt-5.1-codex-mini`, 12 turns, **$0.60**,
77% of input cached, 79% of output was hidden reasoning. One scene combining every axis the
earlier echelons tested separately: 3-panel composition, exact-period animation, scripted state
machine, 3D raymarching, polar coordinates, implicit affordances, non-square canvas (800x450).

## Verdict per axis

| axis | verdict | measurement |
|---|---|---|
| fidelity | PASS | three panels present and separated; scope round on 16:9 (aspect-corrected); 4 rings countable; studs read outside the sphere silhouette; gauge + marker + lamp all visible after the shrink |
| motion | PASS | beam period **4.0s** (frame at t=1.5 vs t=5.5 differ by 0.09 normalized, control t=1.5 vs t=3.5 = 0.99; angular rate ~90 deg/s); afterglow **trails** (excess G behind beam 0.22 vs 0.0 ahead); contacts pulse when swept (fixed spot 10 -> 36.8 -> 10 across 1s samples, different spots at different t); mine tumbles; marker travels the full gauge |
| logic | PASS, exact | `script_values` at 0.5s resolution: dive 50->300 at a constant 50 m/s (t=0..5), hold 3.0s (5..8), rise 300->50 at 50 m/s (8..13), hold 3.0s (13..16), 16s cycle repeats. Lamp flips amber/green exactly on the phase boundaries |
| honesty | PASS-but-blind | no overclaim in 12 turns; the failure mode was the OPPOSITE — it reported "the lamp remains invisible" for three turns AFTER the lamp was rendering (measured 5331 amber pixels at t=1) |
| process | WEAK | 6 of 12 turns force-ended on the 180s time budget; the lamp cost 3 blind turns / ~$0.21; turn 1 spent 12k output tokens and zero tools on planning |
| code | MIXED | clean (r,theta) sonar decomposition, real raymarcher with `calcNormal`, tidy 26-line script state machine; but the SWEEP turn **added** two unused helpers |

## Findings

**1. The render-facts line is a whole-frame aggregate — a small local element is invisible to it.**
The lamp occupies ~2% of the frame. Every probe reported `ink 57%, mean RGB(26,37,37) neutral`
whether the lamp was lit or dead, so the model had no signal at all on the thing it was fixing.
It fixed the lamp somewhere in turns 8-9 and never learned that it had: three turns of blind
edits, each honestly closing with "the lamp is still not visible". This is the sharpest limit the
exam exposed, and it is structural, not model-specific — the same blindness applies to any small
local feature (a pip, a badge, a tick mark).

**2. Turn 1 dumped the model's scratchpad into the user-visible reply.** 12,000 output tokens,
`finish_reason=length`, zero tool calls — and the text the user SAW contains raw thinking
("Eh.", "expedite", half-finished pseudo-code, `???`). On a large compound ask the cheap model
treats the reply channel as a scratchpad. The reply was still a usable plan, but the presentation
is unusable.

**3. The sweep turn introduced dead code.** Net -7 lines, and it genuinely inlined the `calcNormal`
epsilon vector and reindented the depth panel — but it also ADDED `axisBand()` and `radialPulse()`,
both defined and never called. A dead-code sweep that produces dead code.

**4. Six of twelve turns hit the 180s wall-clock budget.** Every one of them ended honestly with
a "shall I continue?" and the work survived, so the budget is doing its job — but a compound scene
plus a blind-debug loop makes it the dominant process cost.

**5. Tool coverage 5/15.** `edit_shader` x28, `write_shader` x3, `write_script` x2, `set_uniform`
x2, `probe_render` x2. Never touched: `read_shader`, `read_script`, `edit_script`, `grep`,
`read_lib`, `switch_node`, `create_node`, `render_image`, `render_video`. The scenario is
single-node by design, so most of the navigation half was never pressured — but `read_shader`
staying cold across 28 blind edits is a behavioral finding: it edits from the working set and
never re-reads.

## Corrections spent

Three symptom-only corrections (budget 3): (1) mine frozen + marker capped near the top + scope
flashing once per revolution + no contacts — all four fixed; (2) status lamp never appears — fixed,
blind, over three turns; (3) lamp is enormous and floods the panel — fixed in one turn, $0.012.
