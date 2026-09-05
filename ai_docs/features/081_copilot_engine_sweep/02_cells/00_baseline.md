# Sweep baseline — measured by the main session (corpus.py)

CORPUS: $3.51, 68 turns, 467 requests, 11 attempts, 8 models, 2 experiments.
  input 10,169,352 (53% cached) · output 548,888 · reasoning 392,767 (72% of output)

NOTE: $3.51 vs the report's $2.53 + $0.98 = $3.51. Consistent.

## Per model
| model | $ | turns | reqs | input | cache% | output | rsn% |
|---|---|---|---|---|---|---|---|
| google/gemini-3.8-flash | 1.224 | 13 | 99 | 1,917,318 | 57 | 138,991 | 86 |
| tencent/hy4-preview | 0.790 | 14 | 81 | 2,008,907 | 61 | 36,732 | 0 |
| openai/gpt-5.1-codex-mini | 0.600 | 10 | 50 | 821,075 | 60 | 250,940 | 95 |
| openai/gpt-5.6-luna | 0.597 | 11 | 141 | 3,579,087 | 43 | 44,849 | 0 |
| moonshotai/kimi-k2.7-code | 0.228 | 7 | 40 | 660,525 | 80 | 13,645 | 44 |
| deepseek/deepseek-v4-flash-0731 | 0.055 | 10 | 50 | 1,094,872 | 44 | 33,288 | 0 |
| z-ai/glm-5.3-flash | 0.016 | 3 | 6 | 87,568 | 0 | 30,443 | 97 |

## The re-send tax (context is re-sent whole per iteration)
| model | peak in/turn | summed in | ratio | reqs/turn |
|---|---|---|---|---|
| gpt-5.6-luna | 308,322 | 3,579,087 | 11.6 | 12.8 |
| gemini-3.8-flash | 261,682 | 1,917,318 | 7.3 | 7.6 |
| kimi-k2.7-code | 100,492 | 660,525 | 6.6 | 5.7 |
| hy4-preview | 375,178 | 2,008,907 | 5.4 | 5.8 |
| codex-mini | 173,812 | 821,075 | 4.7 | 5.0 |
| deepseek-v4-flash | 243,177 | 1,094,872 | 4.5 | 5.0 |
| glm-5.3-flash | 30,156 | 87,568 | 2.9 | 2.0 |
Ratio tracks reqs/turn ~1:1 — the input cost driver is ITERATION COUNT, not context size.

## Cache by position in turn
- first request of a turn: 23.7% cached (293,072/1,235,281)
- later requests:          58.0% cached (5,026,391/8,659,946)
- forced final reply:       9.7% cached (26,656/274,125)
A turn boundary is where the cache is lost, and the forced reply loses it hardest.

## Forced final reply (iteration == -1)
14 firings, $0.124 = 3.5% of corpus spend.

## Terminals / cutoffs
terminal: turn_done 58, noop_streak_giveup 4, stream_error 3, turn_truncated 1, model_incompatible 1, stream_torn 1
cutoff:   (none) 55, max_iterations 10, time_budget 3

## Report cross-check (main session)
Every figure in 01_report.md's two tables reproduces EXACTLY from the log (turns, requests,
cost, reasoning share, per attempt, all 11). The report is accurate; the sweep's value is in
what was never recorded, not in corrections.

## Wall clock
7,350s = 2.0h across 68 turns for $3.51.
| model | total s | s/turn |
|---|---|---|
| codex-mini | 3065 | 306 |
| gemini-3.8-flash | 1078 | 83 |
| hy4-preview | 858 | 61 |
| deepseek-v4-flash | 777 | 78 |
| glm-5.3-flash | 747 | 249 |
| gpt-5.6-luna | 538 | 49 |
| kimi-k2.7-code | 286 | 41 |
codex-mini burned 42% of the total wall clock on a build it abandoned.

## Engine-stop rate per model (cutoff or non-clean terminal)
| model | turns | clean | stopped how |
|---|---|---|---|
| deepseek-v4-flash | 10 | 10 | — |
| hy4-preview | 14 | 11 | noop_streak_giveup 2, max_iterations 1 |
| gemini-3.8-flash | 13 | 9 | max_iterations 3, stream_error 1 |
| codex-mini | 10 | 6 | time_budget 3, turn_truncated 1 |
| kimi-k2.7-code | 7 | 4 | max_iterations 2, stream_error 1 |
| gpt-5.6-luna | 11 | 5 | max_iterations 4, noop_streak_giveup 2 |
| glm-5.3-flash | 3 | 0 | stream_error, model_incompatible, stream_torn |

## Driver steering effort (chars of user_text per attempt)
The end-to-end round cost the driver the SAME total steering as babysat (~5.1-5.4k chars),
just front-loaded into one message. The autonomy gain is in TURNS, not in words.
| exp | # | model | outcome | chars | turns | per turn |
|---|---|---|---|---|---|---|
| rc_full_build | 7 | gemini-3.8-flash | built | 6072 | 10 | 607 |
| rc_full_build | 4 | hy4-preview | built | 5884 | 10 | 588 |
| rc_full_build | 5 | deepseek-v4-flash | abandoned | 5572 | 10 | 557 |
| rc_end_to_end | 3 | gpt-5.6-luna | abandoned | 5416 | 4 | 1354 |
| rc_end_to_end | 1 | hy4-preview | built | 5302 | 4 | 1325 |
| rc_end_to_end | 2 | gemini-3.8-flash | built | 5069 | 3 | 1689 |
| rc_full_build | 2 | codex-mini | abandoned | 4777 | 9 | 530 |
| rc_full_build | 3 | gpt-5.6-luna | built | 4404 | 7 | 629 |
| rc_full_build | 8 | kimi-k2.7-code | abandoned | 3038 | 7 | 434 |

## Ledger sizes (what the NEXT turn sees in history)
Mutating calls per turn reach 30 (kimi t2), 23 (luna e2e t2), 21 (hy4 e2e t2) — far past the
soft cap of 8, so the "... and N more edits" overflow line fires often.
