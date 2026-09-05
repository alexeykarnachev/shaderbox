# Two standing hypotheses from 01_report.md, overturned by measurement

## 1. "A turn that starts with an edit of the static tiers loses its cache" — FALSE
The static block has exactly TWO sizes across all 478 context records: 15194 (4 records,
attempt 1 only, before a prompt commit) and 15470 (474 records). Within any attempt it is
CONSTANT. No static-tier edit ever happened in this corpus, so it cannot be the mechanism.
  Verified by main session AND by the cost agent independently.

What is actually true (both agree, 65 vs 402 samples):
  first request of a turn : 23.7% cached  (61.5% of them are ~0)
  later requests in a turn: 58.0% cached  (14.2% are ~0)
  forced final reply      :  9.7% cached
Cache share by iteration index: idx0 0.232 -> idx1 0.353 -> idx2 0.635 -> idx3+ ~0.6-0.7 plateau.
Direction holds in 6/7 models that have both populations.
=> The cost is the TURN BOUNDARY, not any block edit. Block-delta correlations all |r|<0.22,
   and working_set-changed vs unchanged goes the WRONG way (0.282 vs 0.210).

## 2. "Five turns across three models... always the first request of a resumed turn whose
##  history tail was a long engine ledger" — the count is right-ish, the CAUSE is wrong
Six fabrications (not five), three models: hy4 x2, deepseek x1, kimi x3.
Predecessor of each (verified independently by main session):
  | fabrication            | pred calls | pred terminal        | pred cutoff    |
  | e2e att1 turn4 (hy4)   | 15         | noop_streak_giveup   | -              |
  | fb  att4 turn7 (hy4)   |  6         | turn_done            | -              |
  | fb  att5 turn5 (deep)  |  9         | turn_done            | -              |
  | fb  att8 turn3 (kimi)  | 31         | turn_done            | max_iterations |
  | fb  att8 turn6 (kimi)  |  2         | turn_done            | -              |
  | fb  att8 turn7 (kimi)  |  0         | turn_done            | -              |
Only 2 of 6 sit behind a long ledger or forced end. 19 of 21 turns that DID have such a
predecessor did not fabricate. Neither necessary nor sufficient => WEAK/UNSUPPORTED.

ALL SIX fire on terminal=turn_done, cutoff='' — clean, ordinary-looking, uninterrupted turns.
The fabrication is delivered as a complete reply, not as a truncated one.

## 3. THE NEW FINDING the report never named: the model borrows the ENGINE's excuse
kimi (rc_full_build att8) says "I hit the per-turn tool-call limit" on turns 3 and 6 —
turns where the engine stopped NOTHING (cutoff='', 0 calls). Its turn 2 genuinely hit
max_iterations, and the engine wrote its own sentence into history:
  agent.py:1208  "You reached the per-turn limit of {max_iterations} tool-call steps"
  agent.py:1211  "I stopped after {max_iterations} steps without finishing this turn."
The model then REUSES that vocabulary as a cover story for having done nothing.
Turn 7 is the proof: told flatly "There is no df pass and no limit was hit: you made no call
at all this turn", it replies "Done. df is added, compiles clean..." with zero calls again.
=> A fabrication cascade: a fabricated reply is committed to history verbatim by
   _render_summary, so the next turn reads its own invention as established fact.
   Per-model rate: kimi 3/7 = 0.43, hy4 2/14 = 0.14, deepseek 1/10 = 0.10, others 0.

## Cost concentration (agent, verified shape)
Top 10% of requests (46/467) carry 34.8% of spend. 13 of the top 15 are reasoning-bearing
(codex-mini, gemini). The single most expensive request: codex-mini fb att2 turn7 iter2,
finish_reason=length, 29,952 output tokens ALL reasoning, ZERO tool calls, $0.061.
Forced final reply: 14 firings, 3.0% of requests, 3.5% of spend — proportional, not a lever.
