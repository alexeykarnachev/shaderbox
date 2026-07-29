# Wave-A controls: current prompt @ effort=none (2026-07-29)

Baselines the 059 gates compare against. All on HEAD prompt (pre-059), `llm_reasoning_effort="none"`.

## De-hinted 05 (bounce, no mention of a script) — proj-nxwzgpjx

PASS. The model chose a script from the rule alone (stateful physics), integrated gravity with
damped bounces, rest at 0.22 by t=3. 3 messages ($0.019). Quirk: replied with a PLAN twice in a
row (the second "Go ahead" produced a re-stated plan, zero tools) before an imperative unstuck it.

## 10 pong — proj-ryi4pqbz

Functionally PASS at the end: field/paddles/ball, scoring works (5:7 @30s → 20:28 @150s), score
dot rows fill per side. 10 turns, $0.39, 1 real correction (paddles never missed + no dots) plus
one frozen-game report after a mid-edit breakage.

Process findings (the real payload of this control):
1. **effort=none is NOT honored on compound asks.** Turn 1 and the imperative turn both burned the
   ENTIRE 12k turn budget as hidden reasoning (`out=11968 rsn=11968`), emitting zero text and zero
   tool calls — the turn loops with nothing landing. With `max_tokens_per_turn=30000` the same ask
   went through (reasoning burst fits, edits land). The wave-1 "none is honored" result holds for
   simple asks only; on hard asks reasoning fires regardless of the flag, and 12k is starvation.
   Candidate fix (maintainer call): raise the default `max_tokens_per_turn` 12k → ~24-30k.
2. **Plan-loop**: three consecutive plan-only replies (zero tools) until "Stop planning" +
   imperative verb. Same quirk as de-hinted 05 but deeper.
3. **Engine bug found & fixed on the spot**: the forced final reply streamed with the full
   `max_tokens_per_turn` budget — up to ~400s of legal streaming PAST the 180s turn budget
   (checked only at iteration boundaries); a control turn died to the external 300s kill mid-
   stream. Fix: `COPILOT_ENGINE.final_reply_max_tokens = 1500` (config.py + agent.py).
4. **Time-budget mid-edit breakage**: a 180s cut mid-refactor left the script with a runtime
   AttributeError (`left_error` used, never initialized) — the game froze; the model repaired it
   on the next continue. The write-probe did not surface the error in the reply.
5. **Underclaim habit persists**: final reply said "score values remain zero" while script_values
   showed 20:28 — its probe samples t=0/1.5 where no points exist yet. Same whole-frame/short-t
   blindness class as the exam's lamp.

## 13 final exam control — NOT YET RUN

Required before wave-A rows citing 13. Run next.

## 03 static comp — proj-ge19pnm3

PASS with 2 corrections, 3 messages, $0.02. Turn 1 regressed to the CLIPPED-circles class (row
wider than the frame — the aspect-domain lesson is IN the prompt but was not applied at
effort=none); correction 1 fixed fit but circles touched; correction 2 landed the exact baseline
quality (widths 5x63px, gaps 15px, margins 12/12 — minimal-baseline was gaps 15, dia 62-63).
Signal: at effort=none the model applies learned prompt lessons LESS reliably on the first shot,
but converges to the same final under the same correction budget.

## 13 final exam — proj-5fph6ws9 (effort=none, max_tokens_per_turn=30000)

10 messages, $0.31, 2 corrections (vs minimal-exam: 12 msgs, $0.60, 3 corrections).
BETTER than minimal: lamp phase-correct on the FIRST build (minimal burned 4 blind turns on it);
no once-per-rev flood; depth cycle exact from the start (24 m/s steady, holds 3.0s, state flips on
boundaries); sweep removed real dead code (rot2, merged hold branches) and ADDED nothing.
WORSE: plan-loop needed 2 unstick messages; afterglow initially LED the beam (excess G 21.0 ahead /
0.0 behind, CCW) — fixed by correction 2 (8.3 behind / 0.8 ahead after). Beam/contacts/mine
visibility took correction 1.
Cross-control conclusion: effort=none + 30k budget = comparable-or-better quality at half cost;
the pathologies are plan-loop (3/3 compound controls) and 12k starvation (2/3) — both engine-side,
not prompt-side.
