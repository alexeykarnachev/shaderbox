# 082b — dogfood round: three models on the engine after the edit-tool fixes

**What this was.** The same radiance-cascades build — one spec, asked as a whole — given to three
models in parallel, three turns each. The previous round found two engine bugs in the script editor
and the compile brake; this round ran after those were fixed, to establish whether the models
behave differently when the tooling is no longer working against them, and to settle two cost
predictions the previous round left open.

**Models:** tencent/hy4-preview, google/gemini-3.8-flash, openai/gpt-5.6-luna — the three finishers
of the previous round, so the two rounds sit side by side on the same ask.

**Commit:** `3b2438d` · **Station:** `dogfood/runs/rc_post081_fixed/` · **Run:** 2026-09-06

---

## The headline

**Two of three met the brief. The previous round met it zero times.**

Every attempt of the earlier round ended with a black or near-black canvas after four turns of
debugging. Here all three had real light in their **first** turn, and two finished with the scene
the spec asked for: both halves lit with a soft falloff, the wall casting a visible shadow, the
emitters moving. The cheapest of them cost seven cents.

---

## Results

| Model | Outcome | Cost | Turns | Requests | Requests per tool call | Hidden reasoning |
|---|---|---|---|---|---|---|
| gpt-5.6-luna | `built` | $0.069 | 3 | 26 | 0.684 | 0% |
| hy4-preview | `partial` | $0.165 | 3 | 22 | 0.579 | 0% |
| gemini-3.8-flash | `built` | $1.045 | 3 | 35 | 0.875 | 97% |

**Outcome vocabulary** (`dogfood/report/log.py::OUTCOMES`, enforced at `end_attempt`) — an outcome
says what the MODEL reached, never that the driver stopped driving:
`built` goal met · `partial` real progress, goal not reached · `regressed` ended worse than it
started · `blocked` an engine defect stopped it · `abandoned` the model gave up · `smoke` infra check.

---

## The animation

`dogfood/runs/rc_post081_fixed/media/3/t003_99bca598-6ce1-43f6-8f16-8ff792041dbb.mp4` — luna's
final render: the wall standing as a solid black segment, warm light filling the left half and cool
the right, each falling off softly, the shadow moving as the emitters bounce.

`dogfood/runs/rc_post081_fixed/media/2/t003_77ae8301-6273-4961-bbe7-52ad9022d956.mp4` — gemini's,
after it recovered: the same scene with a harder shadow edge and a visible seam down the middle.

Both are on their attempt pages.

---

## Per model

### gpt-5.6-luna — `built`, and the whole thing for seven cents

A fifteenth of gemini's bill for a better ending.

Turn 1 built the pipeline with real, smooth light. Turn 2 fixed the emitter bounce. Turn 3 is the
one worth reading: told to *read* the paint pass before editing again — its four edits the turn
before had produced nothing, two of them reported as changing nothing — it read, and found the
cause. The wall endpoint had been given an already aspect-scaled x coordinate, which the segment
function then scaled a second time, putting the wall off-canvas entirely. One edit, $0.008, and the
scene is correct.

Four guessing edits achieved nothing; one instruction to read first solved it in a single call.

### hy4-preview — `partial`, never regressed, never converged

Built the whole pipeline in turn 1 for $0.118. It **opened by reading the shipped Radiance Cascades
example and the shader library** — navigation tools that went completely unused through every
attempt of the previous round — and that read returned all six passes rather than only the final
one, which is the earlier reference-reading fix working on exactly the reference these failures
need.

Its behaviour was the most disciplined of the three. Turn 2 ran one probe, could not confirm its
diagnosis, and stopped and asked rather than guessing ($0.027). Turn 3 read its own source and
found a genuine bug — `length()` applied to a scalar in the wall SDF, which happens to work as
`abs()` — plus the observation that a *black* wall on a black background is invisible in paint's
own probe, since only alpha marks it.

None of it reached the picture. Across three turns the render stayed blocky and the wall never
appeared. Good judgement, no landing.

### gemini-3.8-flash — `built`, at fifteen times the price

**Turn 1 met the entire spec in one go** — all seven passes and the script from the bare
description, both halves lit, a real wall shadow, emitters drifting. No model has done that here
before. $0.266.

**Turn 2 broke it, expensively.** Asked to fix a brightness fade, it spent **$0.752 in a single
turn** — 617 seconds, 153,818 output tokens of which all but ~800 were hidden reasoning — hit the
wall-clock budget, and left the canvas flat red. Its own probe reported `FLAT — one uniform color
rgba(255,0,0)` mid-turn and it kept editing past that.

**Turn 3 undid it in 13.7 seconds for $0.027.** It had left a debug red test in the cascade pass.
Told plainly what its own probe had already said, it found it, reverted it, and the render is
correct again.

The damage was self-inflicted, reversible, and cost **28× more to cause than to undo**. The attempt
ends where turn 1 had it, having spent $1.045 to get back.

---

## What this found in the engine

**Nothing stops a turn that is expensive and destructive rather than repetitive. Open.**

Gemini's second turn spent three quarters of a dollar making the picture worse while every existing
brake watched. Those brakes count *repetition* — the same edit twice, edits that change nothing,
edits that fail to compile. That turn did none of it: varied, clean, compiling edits, each doing
something, collectively ruining the render.

Two separate gaps. **No cost or time signal reaches the model** — it had no idea it was running
fifteen times more expensive than its own first turn. And **nothing notices a frame going from lit
to flat** — the engine measured `FLAT — one uniform color` and passed it through weighted like any
other line of feedback.

The second is the tractable one: the engine already computes the number, and "this frame was rich
and is now uniform" is a comparison it could make. Whether that stops a turn or is merely said
louder is a design call this round does not settle.

The case is replayable from `dogfood/runs/rc_post081_fixed/`, attempt 2, turn 2. **That store is
gitignored** — it exists on this machine only, so the figures quoted here are the durable record of
it.

**The encouraging half:** every model corrected course immediately when told plainly what the
render showed, and did so cheaply ($0.008–$0.027). The recovery path works. What is missing is
noticing without a human in the loop.

**Fixed this round, found while writing the report:** the station's `outcome` was a free-form
string where its sibling `mode` was a validated vocabulary, so seven ad-hoc words accumulated and
the site styled exactly one of them green — `"success"`, which was never a legal value, meaning
every attempt page ever built showed a red pill including the nine successful ones. Now a closed
enumeration rejected at `end_attempt`, with three colour states.

---

## Predictions from the previous round

**Held.** Keeping the pass-editing tools permanently loaded protects the prompt cache. Whenever the
copilot loads a tool mid-turn, the very next request drops to about 4% cache reuse against 78–85%
on the requests either side. The two tools made permanent were used heavily here and triggered no
such reload.

**Did not hold.** Rewriting the batching instruction was meant to group work into fewer round
trips. It did not: hy4 at 0.579 requests per tool call against 0.552 before, luna 0.684, gemini
0.875 — all flat or worse. The instruction is not doing what it was rewritten to do, and that is
now measured on three models rather than the one that produced the original claim.

*A note on two number sets.* Commit `fcbe374` quotes 0.528 / 0.649 / 0.868 and luna at $0.062;
this report quotes 0.579 / 0.684 / 0.875 and $0.069. Both are correct measurements — the commit
landed when the round was two turns per model, and `d0d9b82` then ran the third turn each so the
round matched the length of the one it compares against. The three-turn figures above are the
round's. The conclusion is the same under either set.

---

## Tool coverage

Fired: `edit_shader` 29, `write_shader` 23, `probe_render` 21, `add_pass` 19, `grep` 7, `set_pass`
4, `create_document` 3, `write_script` 3, `read_shader` 3, `edit_script` 2, `read_lib` 1,
`read_script` 1.

**Cold:** `switch_document`, `delete_document`, `delete_pass`, `set_uniform`, `render_image`,
`render_video`, `set_canvas_size`, `load_tools` — the scenario never pressured any of them. This
was one document built from a spec, so there was nothing to navigate between, nothing to throw
away, and no adjustable look to dial. That is the mission's shape, not the models dodging: a
single-document build has nowhere to navigate to. A round aimed at the navigation surface needs a
mission with more than one document in it.

---

## What a next round would test

Whether gemini's one-turn build is repeatable or was luck. It is the only evidence this spec is
buildable in a single turn and it rests on one sample.

The destructive-turn brake: gemini's turn 2 is a clean, reproducible case sitting in the log, ready
to replay against any guard built for it.

Whether hy4's pattern holds. Three turns of sound reasoning that never reached the frame is a
different failure from luna's, and worth understanding separately.

A mission spanning several documents, to put the navigation half of the tool surface under real
pressure for the first time.
