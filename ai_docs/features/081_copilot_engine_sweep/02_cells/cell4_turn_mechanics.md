# CELL 4 — F4 turn/iteration mechanics. Moderator: opus. KEY CLAIMS VERIFIED by main session.

## THE DECISIVE NUMBER: the engine CANNOT brake its way out of the re-send tax
VERIFIED (main session): beyond-first iterations that made ZERO tool calls — pure round-trip waste:
  glm 50.0% (n=4) | deepseek 22.5% | codex-mini 19.4% | hy4 13.6% | gemini 10.7% |
  luna 4.0% (5/126) | kimi 3.1%
luna, the model with the WORST re-send ratio (12.45 reqs/turn, 11.6x), wastes the LEAST. Its
iterations are 96% productive — it is UN-BATCHED, not looping (reqs/call 0.867, one call at a
time). A brake tuned to cut luna's turns would cut PRODUCTIVE steps.

## Mechanism
WITHIN a turn the message list is APPEND-ONLY (agent.py:475-478, 615, 641-643), so the provider's
implicit prefix cache matches more each pass: 0.232 -> 0.353 -> 0.635 -> 0.639.
AT THE BOUNDARY the prefix is DESTROYED, not extended. session.py:449-453 commits exactly two
messages; _render_summary synthesises prose that NEVER appeared in any turn-N request. The whole
tool exchange iterations 1..N warmed is discarded. Turn N+1's dialogue differs in every byte after
turn N began — a REWRITE, not an append. session.py:437-440 states the intent (NL-only history, no
stale persisted source).
Compounding: project_context (Volatility.RARE, sorted ABOVE dialogue) is rebuilt each turn from
live project state; any create/edit/compile-flip shifts a block near the TOP of the prompt.
PROVIDER vs SHADERBOX: no cache_control is ever sent (grep finds only the READ of cached_tokens,
llm/api.py:45, openrouter.py:209). Caching is implicit per-provider. Mid-turn requests already hit
0.58-0.64 with no breakpoints => implicit caching WORKS once bytes stop changing. The miss is a
ShaderBox design choice, and breakpoints could NOT rescue it (a breakpoint pins identical bytes;
it cannot save content that changed).
NOTHING IS INCREMENTAL: no delta path exists; _trim_history runs ONCE per turn (prompt.py:416-436);
the working set is rebuilt every pass and says so ("live shader source, rebuilt EVERY step",
prompt.py:319).
  corr(iteration count, turn input) r = 0.934 ; corr(peak context size) r = 0.434.
  87.5% of ALL corpus input tokens are paid past the first iteration of a turn.
  1 iter: 19,907 in / $0.0097  ->  16+ iters: 361,785 in / $0.1068.
  Worst single turn: luna fb att3 t5, 16 requests, 495,773 input tokens.

## THE GAUGE UNDER-REPORTS BY ~20x — VERIFIED AT SOURCE
agent.py:502  `first_input_tokens: int | None = None  # iter-0 context size for the usage bar`
feeds TurnStats.context_tokens at all yield sites.
state.py:106-122 context_gauge_readout prints it AS "Context: N / budget" in the SAME TOOLTIP as
`Last turn cost: $X` — and cost_usd IS the turn total. Two numbers on different bases, side by
side. On the luna turn: gauge ~24,381 vs 495,773 actually billed.
grep: `context_gauge_readout` has **0 tests**. `summed_input` appears **0 times** in shaderbox/.

## The brake family: 3 of 8 have NEVER fired
| brake | default | scope | corpus firings |
|---|---|---|---|
| max_iterations | 16 | per-turn | 10 turns |
| turn_time_budget_s | 180 | per-turn | 3 (codex-mini) |
| max_tokens_per_turn | 30k | PER-REQUEST (agent.py:643) | — |
| max_edit_retries | 3 | per-turn global | **0** |
| max_compile_failures | 5 | per-turn global | **0** |
| clean_edit soft/hard | 6/12 | per-(kind,file) | soft 6; hard **0** |
| noop soft/hard | 3/6 | mixed per-file + global | soft 7; hard 4 |
| auto_revert | 6 | per-doc, CROSS-TURN | unmeasured |
NO brake counts cost or tokens across a turn. max_tokens_per_turn is per-REQUEST, so a
16-iteration turn may emit 16 x 30k. (The Settings label is honest: "per LLM step"; only the
internal field name over-claims.)

## Blast radius (arithmetic)
CACHE LEVER: first-of-turn 65 reqs, 1,235,281 in, 0.237 cached. At the later rate (0.580),
423,907 tokens move to cache price = 34.3% of first-of-turn input but only 4.3% of ALL corpus
input; the dollar figure is less again (cached tokens still bill). CEILING: low single digits.
BATCHING LEVER: counterfactual at kimi's 0.655 reqs/call, per model's own per-request cost =
$0.9959 of $3.3865 = **29.4% of spend**. SEVEN TIMES the cache lever.
OUTPUT-SIDE is model disposition, not engine: reasoning share codex-mini 94.9%, glm 96.6%,
gemini 86.4%, kimi 44.5%, luna/hy4/deepseek 0.0%.

## Proposals
P1 (DO FIRST) — re-justify batching in the prompt BY COST, with a worked multi-call example.
  prompt.py:214 currently sells it as conserving STEPS ("steps are the scarce budget"). A model
  with 11 of 16 steps left has no reason to obey. Saying every step re-sends the ENTIRE
  conversation gives a reason that holds at step 1.
  CLASS: the 29.4% lever, the only one touching all seven models. Zero new state, zero counters.
  CANNOT make things worse (no brake, no rejection path).
  VERIFICATION: a re-run measuring reqs/call per model. This is a PROMPT change, NOT a gate —
  and explicitly DO NOT add a test asserting the prompt contains the word "BATCH" (the
  reader-agreeing-with-itself shape test_brake_falsifiers.py's own docstring warns about).
P2 — make the gauge report the turn's REAL billed input, in the TOOLTIP beside the cost line;
  leave the fill fraction as the per-request fullness signal it correctly is (summed input is not
  comparable to a per-request budget — conflating them makes the bar meaningless).
  THE BREAK: a fake 3-iteration turn with known sizes; assert the readout reports their SUM, then
  change one iteration's size and assert the number moves. Today's code reports only iteration 0
  and FAILS the second assertion — that is the break. Worth doing: 0 tests today.
P3 — DO NOTHING about the turn-boundary cache miss. Ceiling 4.3% of input; the alternative
  (persisting the real tool exchange) costs a permanently larger dialogue block re-paid forever
  AND reintroduces the stale-source hazard session.py:437-440 deliberately closed. Recorded as a
  DELIBERATE DECISION so the next session does not re-derive it.
P4 — DO NOT add a cost/token brake (explicit non-proposal). Expensive turns are PRODUCTIVE turns.
  A cost brake would cut a model off mid-build, and the forced-final-reply path already produces
  the F1 fabrication behaviour — a new stop reason is new excuse vocabulary to borrow.
  THE PATTERN THE THREE PATCHES TEACH: 8fa8ece built one global counter; c8960e1 widened WHAT it
  counted into the same bucket; b54baad had to SPLIT THE SCOPE because the global count
  false-fired and killed a legitimate seven-pass build on its second pass. Every patch's hole was
  created by the previous patch's SCOPE choice. Three brakes have never fired at all. Adding a
  fourth counter to that family is the highest-risk, lowest-evidence move available.
P5 — hygiene only if the area is open: clean_edits_by_file is write_shader-reset (agent.py:1053)
  so 11 edits + 1 write never trips the cap at 12, while its sibling noops_by_file deliberately is
  not — asymmetry flagged, not proposed. _edit_target_key uses the RAW target string
  (agent.py:143-144), so F2's <id>#<pass> inconsistency makes streak-splitting LIVE, not
  theoretical. auto_revert counts CROSS-TURN while every other brake resets per turn
  (backend.py:2481) — undocumented as a decision.

## False trails
- "Add explicit cache_control breakpoints" — wrong provider path; mid-turn already 0.58-0.64.
- "luna is looping, brake it" — 96% of its beyond-first iterations made a real call.
- "max_tokens_per_turn caps a turn" — per-REQUEST.
- "Trimming will help" — 0 of 478 trimmed; budget 150k vs ~37k worst case. _trim_history is
  untested-by-use and its chars/4 UNDER-counts by ~11% (F8) — unsafe in the direction that
  matters if any bound is ever tightened.
- "kimi batches best at 61.8%" — inflated by one 2-call request repeated ~11x; use reqs/call.
