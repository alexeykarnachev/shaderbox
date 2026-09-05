# Context accounting — measured over 478 records

## THE THIRD OVERTURNED CLAIM — and this one is in the SKILL, not the report
.claude/skills/dogfood/SKILL.md: "Measured 2026-09-04 on codex-mini: the chars/4 estimate runs
~7-8% ABOVE the billed input (ratio 1.07-1.08), so read the bar as proportions and the billed
column as the number."

MEASURED over all 475 records with billed data (verified by main session):
  ALL          n=475  mean 0.8914  median 0.8743  min 0.716  max 1.045
  codex-mini   n= 58  mean 0.9469  median 0.9429     <- the skill's own cited model
  hy4          n= 81  mean 0.8474
  deepseek     n= 50  mean 0.8553
  luna         n=141  mean 0.8757
  gemini       n= 99  mean 0.9035
  glm          n=  6  mean 0.9457
  kimi         n= 40  mean 0.9622
RECORDS AT OR ABOVE 1.07: **0 of 475.**

The SIGN is wrong, not just the magnitude: chars/4 UNDER-counts billed input by ~11% corpus-wide,
and never once over-counted. A future run following the skill would correct in the wrong direction.

## The static floor is 42.7% of all input spend
static + tools, paid on EVERY request:
  floor est tokens: min 3798, median 9405, max 10194
  sum(floor) 4,414,578 / sum(billed input) 10,336,145 = 42.7% corpus-wide (mean per record 45.5%)

## Static prompt composition (15470 chars, sections reconcile exactly)
| section | chars | % |
|---|---|---|
| VISUAL CRAFT | 4396 | 28.4 |
| HOW TO WORK | 1854 | 12.0 |
| SCRIPTING | 1798 | 11.6 |
| FEEDBACK | 1390 | 9.0 |
| WORKING SET | 1329 | 8.6 |
| EDITING | 1111 | 7.2 |
| USING TOOLS | 1076 | 7.0 |
| NODES, LIBRARY, MEDIA | 736 | 4.8 |
| THE SANDBOX | 532 | 3.4 |
| RENDER & PUBLISH | 391 | 2.5 |
| ADDRESSING | 284 | 1.8 |
| TELEGRAM + YOUTUBE | 239 | 1.5 |
| preamble | 334 | 2.2 |
VISUAL CRAFT is 28% of the static prompt — 3x the next section — and this task is a
physically-specified pipeline where craft advice does little. TELEGRAM+YOUTUBE (239 ch) and
RENDER & PUBLISH (391 ch) are paid on every request for a surface that stayed 100% COLD.

## The tools block: ~285-300 est tokens PER TOOL DEFINITION, flat
| n_tools | records | med chars | tok/tool |
|---|---|---|---|
| 0 (closing reply) | 14 | 2 | - |
| 18 baseline | 215 | 20338 | 282 |
| 19 | 62 | 22249 | 293 |
| 20 | 157 | 24063 | 301 |
| 21 | 22 | 24570 | 293 |
| 22 | 8 | 25311 | 288 |
A load_tools of the common add_pass+set_pass pair costs +3725 chars / +931 est tokens, EVERY time
it fires (30 such jumps measured). load_tools was called 37 times, sometimes re-loading tools
already loaded earlier in the same attempt.

## Growth: dialogue ACROSS turns, turn_exchange WITHIN a turn
Summed deltas: dialogue +49,374 tok, working_set +24,730, turn_exchange +16,562,
project_context +223, static 0.
Within one turn (60 multi-iteration turns): median turn_exchange +1542 tok, working_set +336;
the other four blocks are 0 by construction (rendered once at turn build).

## Trimming never fired
0 of 478 records trimmed; dialogue is monotonically non-decreasing in all 11 attempts. The
_trim_history path exists and is UNEXERCISED by this corpus — largest single record ~30k tokens.
=> Untested code. Nothing here says it works.

## The static block's one real edit (explains the two sizes)
15194 -> 15470 is a 276-char insertion in NODES/LIBRARY/MEDIA: the add_pass / "NEW PASS"
sentence. Attempt 1 (codex-mini) ran BEFORE 076's fix (8b26ce3); every later attempt after.
So the two sizes are the 076 fix landing, not drift.
