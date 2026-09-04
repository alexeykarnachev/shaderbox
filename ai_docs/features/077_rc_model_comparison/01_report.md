# 077 — the radiance-cascades build, eight attempts on seven models

What the first station experiment (`rc_full_build`) measured across models, and what it found in
the engine. The station record (`dogfood/runs/rc_full_build/`, local) holds every turn, tool call,
render and context breakdown; this file is the part that must outlive the box.

The task in every attempt: the same babysat build, one ask per pipeline stage — paint + seed, jfa
(12 runs), df, cascade in two steps (packing + march, then the merge), composite, the emitter script,
the drawing canvas — with the driver adapting the next ask to what actually happened, and naming a
symptom rather than a fix when something was wrong. Attempts 1–2 predate 076 and the encoding lessons
(their asks were rougher); attempt 3 continued attempt 2's half-built project; attempts 4–8 each
started from an empty project with the refined asks (solidity in alpha and HDR emitters from turn 1).

## The table

| # | model | outcome | turns | requests | cost | s/turn | hidden reasoning | limit-forced turns | replies claiming work with zero calls |
|---|---|---|---|---|---|---|---|---|---|
| 1 | openai/gpt-5.1-codex-mini | blocked (no pass tools → 076) | 1 | 4 | $0.04 | 194 | 94% | 1 | 0 |
| 2 | openai/gpt-5.1-codex-mini | abandoned at the merge | 9 | 46 | $0.56 | 118–686 | 95% | 3 | 0 |
| 3 | openai/gpt-5.6-luna | **built** (from attempt 2's project) | 7 | 85 | $0.42 | 10–91 | 0% | 3 | 0 |
| 4 | tencent/hy4-preview | **built** | 10 | 49 | $0.52 | 24–108 | 0% | 0 | 1 |
| 5 | deepseek/deepseek-v4-flash-0731 | abandoned at the merge | 10 | 50 | $0.05 | 11–336 | 0% | 0 | 1 |
| 6 | z-ai/glm-5.3-flash | abandoned at turn 3 | 3 | 6 | $0.02 | 1–450 | 97% | 0 | 0 |
| 7 | google/gemini-3.8-flash | **built** | 10 | 62 | $0.70 | 1–180 | 85% | 1 | 0 |
| 8 | moonshotai/kimi-k2.7-code | abandoned at df | 7 | 40 | $0.23 | 1–126 | 44% | 2 | 3 |

Whole experiment: $2.53. "Limit-forced" = the engine ended the turn (time budget, `max_iterations`,
a brake). "Zero calls" = a reply describing tools it never called, counted from the log.

## Per model

- **hy4-preview** — the most decisive builder: every pass on the first or second ask, the merge
  fixed after one symptom report, script and canvas one turn each, no hidden reasoning, no
  limit-forced turn. Two honesty faults: it claimed a merge edit "never landed" (it had; the
  working set it was shown listed it) and rewrote the file; and one report of a composite pass it
  never created. Both corrected the turn after they were named.
- **gemini-3.8-flash** — the smoothest picture and the only finisher without a false report. The
  merge and the composite landed first try. It wanders before settling: eleven greps, reads of
  the shipped example, a throwaway document created and deleted, one turn to `max_iterations`; it
  declared its own `int u_run` for the run index, and scaled df by the aspect. Reasoning cannot be
  turned off on this endpoint and is 85% of its output tokens; $0.70 is the dearest finisher.
- **gpt-5.6-luna** (the default since `3da796a`) — finished attempt 2's project in seven turns with
  zero hidden reasoning; its habit is a no-op style-pass churn after every substantive change
  (three limit-forced turns), which the no-op brake now stops. Not a full build from empty, so its
  numbers are not directly comparable with 4 and 7.
- **deepseek-v4-flash** — $0.05 for ten turns and 11 s turns once it stops wandering (turn 3: 16
  requests reading the shipped example despite the ask). Built paint, seed, jfa, df and a lit
  cascade, then could not converge on the merge: streaks survived two symptom rounds and the
  reference addressing pasted verbatim (measured: luna's cascade on its scene is smooth). One
  fabricated action report.
- **kimi-k2.7-code** — a working jump flood in four real turns, then two turns of an A,B,A,B
  cycle of identical `set_pass` calls to `max_iterations`, and three replies in four turns
  describing work with no tool call, twice narrating an engine stop that did not happen. Unusable
  as an actor here.
- **glm-5.3-flash** — reasoning is mandatory on its endpoint and it spends 5–7 minutes per request
  on it; one request ended by the provider, one torn. Nothing usable.
- **gpt-5.1-codex-mini** (the former default) — 95% hidden reasoning under `effort=none`, two 30k
  reasoning burns landing nothing on the cascade; retired by `3da796a`.

## What the run found in the engine (all landed)

- `076` — the copilot could not add a pass (`8b26ce3`).
- A wrong-typed engine builtin was a silent zero → a compile error (`d329d04`).
- The harness resized only the output pass (`3e8e9c4`); `set_pass` refused an empty dtype (`7fe63e9`).
- No-op brake: unchanged-frame edits, then any call already made with the same arguments this
  turn (`8fa8ece`, `c8960e1`).
- The probe renders at the document's size (`8fa8ece`) and takes a pass address (`c8960e1`).
- The reasoning-effort setting is dropped and remembered for a provider that refuses it; a
  provider `finish_reason: error` is a torn stream, not an incompatible model; engine-driven
  uniforms show a marker in the working set instead of a stale number (`c8960e1`).

## Left for the sweep

- **Replies describing tools never called.** Five turns across three models, always the first
  request of a resumed turn whose history tail was a long engine ledger (a forced-end note, a
  ledger of twenty edits, a rejected-tool list). The model reads the ledger as its own recent work
  and narrates a continuation. Where the replay history and the forced-end notes meet is the
  place to look; the prompt rule ("claiming an action REQUIRES a tool result THIS turn") did not
  hold for any of the three.
- **Reading the shipped example despite the ask.** deepseek and gemini both read the Radiance
  Cascades example repeatedly for reference (never instantiated it). Whether that is a fault at
  all is the maintainer's call; the catalogue makes it one grep away.
- **Wandering before the first edit** (gemini: eleven greps and a throwaway document): the cost
  is requests, and the tools are cheap; a "plan first" rule does not stop it.
- **Hidden reasoning is the cost driver for the models that have it** — 85–97% of output tokens
  on gemini, glm and codex-mini. Luna and hy4 spend none and were not worse.
- **Cache share** per request is on every attempt page (0–95% across turns; a turn that starts
  with an edit of the static tiers loses it) — input for the caching half of the sweep.

## Next round, if wanted

An `end_to_end` ask (the whole build in one message) on hy4, gemini and luna would measure the
autonomy this babysat round could not; and hy4 against luna on an equal footing (both from an
empty project) decides the default on data rather than on one continued attempt.

## The end-to-end round (`rc_end_to_end`)

The three finishers got the whole build as ONE message — the design document in
`02_design_document.md` — on empty projects; after it the driver only reacted to the result as a
whole ("continue", "it is dark, the wall is missing", the sweep), never per stage.

| # | model | outcome | turns | requests | cost | s/turn | hidden reasoning | engine-ended turns | zero-call claims |
|---|---|---|---|---|---|---|---|---|---|
| 1 | tencent/hy4-preview | **built** | 4 | 32 | $0.27 | 11–168 | 0% | 3 | 1 |
| 2 | google/gemini-3.8-flash | **built** | 3 | 37 | $0.52 | 36–246 | 87% | 2 | 0 |
| 3 | openai/gpt-5.6-luna | abandoned — never lit | 4 | 56 | $0.18 | 52–60 | 0% | 3 | 0 |

$0.98 for the round. **hy4 and gemini both built the whole pipeline from the document in two turns**
(the first turn ends at `max_iterations` for everyone — seven passes and a script do not fit in
one turn's step budget; "continue" is all it takes), with a lit scene, the wall's shadow and moving
emitters. Two of hy4's three engine-ended turns were false stops by the no-op brake (below), and
its sweep report after the second one listed a cascade edit that never happened. **luna could not
carry the design on its own**: every shader compiled, nothing lit, and two diagnostic turns with
pointers (paint edited 24 times) found nothing — the model that finished the babysat build, where
each stage was checked, is not the one for a whole spec.

Engine findings this round, both landed in `b54baad`: the edit probe measured the document's
OUTPUT even when the edit targeted another pass, so every write to `paint` or `canvas` while the
output was still the stub read "changed NOTHING on screen" and the no-op brake ended a seven-pass
build on its second pass; it now probes the edited pass. And a dead-code sweep is a run of unchanged
frames by nature, so unchanged-frame edits count per file (the repeated-call count stays global).

**On the default.** hy4-preview finished both rounds, fastest and cheapest of the finishers per
turn, with no hidden reasoning; its faults (a zero-call narration after an engine stop) are the
shared class, not its own. gemini is the smoothest picture and never fabricated, at twice the cost
and with reasoning it cannot switch off. luna, the current default, failed the end-to-end. The maintainer
chose hy4 on this data; it is the default since the commit that lands this paragraph.
