# Dogfood round: three models on the fixed engine

**What this was.** The same radiance-cascades build, asked once as a whole, given to three models
in parallel on one commit. The previous round found two engine bugs; this round ran on the engine
after those were fixed, to see what the models do when the tooling is no longer working against
them. Three turns each — build, then react to what I could see in the render and they could not —
matching the previous round's length so the two sit side by side.

**Models:** `tencent/hy4-preview`, `google/gemini-3.8-flash`, `openai/gpt-5.6-luna` — the three
finishers of the previous round.

---

## The headline

**Two of three met the brief. The previous round met it zero times.**

Every attempt of the earlier round ended with a black or near-black canvas after four turns of
debugging. This round, all three models had real light in their **first turn**, and two finished
with the scene the spec asked for: both halves lit with a soft falloff, the wall casting a visible
shadow, the emitters moving.

---

## Per model

| Model | Outcome | Cost | Turns | Requests per tool call | Hidden reasoning |
|---|---|---|---|---|---|
| gpt-5.6-luna | **built** | **$0.069** | 3 | 0.684 | 0% |
| gemini-3.8-flash | built | **$1.045** | 3 | 0.875 | **97%** |
| hy4-preview | partial | $0.165 | 3 | 0.579 | 0% |

### gpt-5.6-luna — the clear winner

**The whole spec, essentially met, for seven cents.** A fifteenth of gemini's bill.

Turn 1 built the pipeline with real, smooth light. Turn 2 fixed the emitter bounce I pointed out.
Turn 3 is the one worth reading: I told it to *read* the paint pass before editing again, because
its four edits the turn before had produced nothing. It read, and found the actual cause — the wall
endpoint had been given an already aspect-scaled x coordinate, which the segment function then
scaled a second time, putting the wall off-canvas entirely. One edit, $0.008, and the render has
the wall casting a shadow with both halves lit.

That contrast is the lesson: four guessing edits achieved nothing; one instruction to read first
solved it in a single call.

### gemini-3.8-flash — the best capability, the worst cost control

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

So the damage was self-inflicted, reversible, and cost **28× more to cause than to undo**. The
attempt ends exactly where turn 1 had it, having spent $1.045 to get back.

### hy4-preview — never regressed, never converged

Built the whole pipeline in turn 1 for $0.118. It **opened by reading the shipped Radiance Cascades
example and the shader library** — navigation tools that went completely unused through every
attempt of the previous round — and that read returned all six passes rather than only the final
one, which is the earlier reference-reading fix working on precisely the reference these failures
need.

Its behaviour was the most disciplined of the three: turn 2 ran one probe, could not confirm its
diagnosis, and stopped and asked rather than guessing ($0.027); turn 3 read its own source and
found a genuine bug — `length()` applied to a scalar in the wall SDF, which happens to work as
`abs()` — plus the sharp observation that a *black* wall on a black background is invisible in
paint's own probe, since only alpha marks it.

But none of it reached the picture. Across three turns the render stayed blocky and the wall never
appeared. Good judgement, no landing.

---

## The engine finding

**Nothing stops a turn that is expensive and destructive rather than repetitive.**

Gemini's second turn spent three quarters of a dollar to make the picture worse while every
existing brake watched. Those brakes count *repetition* — the same edit twice, edits that change
nothing, edits that fail to compile. That turn did none of it: varied, clean, compiling edits, each
doing something, collectively ruining the render.

Two separate gaps:

1. **No cost or time signal reaches the model.** It had no idea it was running fifteen times more
   expensive than its own first turn.
2. **Nothing notices a frame going from lit to flat.** The engine measured `FLAT — one uniform
   color` and passed it through as one more line of feedback, weighted like any other.

The second is the tractable one — the engine already computes the number, and "this frame was rich
and is now uniform" is a comparison it could make. Whether that stops a turn or is merely said
louder is a design call this round does not settle.

**Turn 3 is the encouraging half.** Every model corrected course immediately when told plainly what
the render showed, and did so cheaply ($0.008–$0.027). The recovery path works; what is missing is
noticing without a human in the loop.

---

## The previous round's two predictions

**Held.** Keeping the pass-editing tools permanently loaded protects the prompt cache. Whenever the
copilot loads a tool mid-turn, the very next request drops to about 4% cache reuse against 78–85%
on the requests either side. The two tools made permanent were used heavily here and triggered no
such reload.

**Did not hold.** Rewriting the batching instruction was meant to group work into fewer round
trips. It did not: hy4 at 0.579 requests per tool call against 0.552 before, luna 0.684, gemini
0.875 — all flat or worse. The instruction is not doing what it was rewritten to do, and that is
now measured on three models rather than one.

---

## What a next round would test

Whether gemini's one-turn build is repeatable or was luck — it is the only evidence this spec is
buildable in a single turn, and it rests on one sample.

The destructive-turn brake: gemini's turn 2 is a clean, reproducible case sitting in the log,
ready to replay against any guard built for it.

And whether hy4's pattern holds — three turns of good reasoning that never reached the frame is a
different failure from luna's, and worth understanding separately.
