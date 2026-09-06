# Dogfood round: three models on the fixed engine

**What this was.** The same radiance-cascades build, asked once as a whole, given to three models
in parallel on one commit. The previous round found two engine bugs; this round ran on the engine
after those were fixed, to see what the models do when the tooling is no longer working against
them. Two turns each: build, then react to what I could see in the render and they could not.

**Models:** `tencent/hy4-preview`, `google/gemini-3.8-flash`, `openai/gpt-5.6-luna` — the three
finishers of the previous round, so the two rounds sit side by side.

---

## The headline

**All three produced real light this time. The previous round produced none.**

Every attempt of the earlier round ended with a black or near-black canvas after four turns of
debugging. This round, all three models had light spreading from both emitters in their **first
turn**, and one of them met the full spec — both halves lit with a soft falloff, the wall casting a
real shadow, the emitters moving.

That is the difference the engine fixes made, and it is bigger than expected.

---

## Per model

| Model | Total cost | Turns | Requests | Requests per tool call | Hidden reasoning |
|---|---|---|---|---|---|
| gpt-5.6-luna | **$0.062** | 2 | 24 | 0.649 | 0% |
| hy4-preview | $0.145 | 2 | 19 | **0.528** | 0% |
| gemini-3.8-flash | **$1.018** | 2 | 33 | 0.868 | **97%** |

### gemini-3.8-flash — the best turn and the worst turn

**Turn 1 was the best single turn any model has produced in this project.** From the spec alone it
built all seven passes and the script, and the render was correct radiance cascades: the warm
emitter filling its half, the cool one filling the other, a genuine shadow cast by the wall between
them, both drifting. That is the full spec, met in one go, for $0.266.

**Turn 2 destroyed it.** Asked to fix a brightness fade, it spent **$0.752 in a single turn** —
more than the entire previous round cost — burning 617 seconds and 153,818 output tokens, of which
**all but 800 were hidden reasoning**. The turn ended by hitting the wall-clock budget. The frame it
left behind is flat red with no light anywhere.

The engine told it what was happening. Mid-turn its own probe reported `FLAT — one uniform color
rgba(255,0,0)`, and it kept editing past that.

### hy4-preview — the best value, and the only one that did no harm

Built the whole pipeline in turn 1 for $0.118. It **opened by reading the shipped Radiance Cascades
example and the shader library** — navigation tools that stayed completely unused through every
attempt of the previous round — and that read returned all six passes of the example rather than
only the final one, which is the earlier fix for reference-reading doing its job on the very
reference these failures need.

Its render has real coloured light from both emitters and they move, but the light is built out of
large rectangular blocks and there is no wall.

Turn 2 is the interesting one: asked to measure before editing, it ran one probe, found it could not
confirm the wall was in the scene, and **stopped and asked rather than guessing** — 2 requests,
$0.027. No progress, but the only second turn of the three that left the frame no worse than it
found it.

### gpt-5.6-luna — cheapest by a wide margin

$0.062 for the whole attempt — a sixteenth of gemini. Turn 1 produced smooth, correct-looking light
with both emitters glowing and moving; turn 2 fixed the emitter drift I pointed out, so they now
bounce independently.

It never produced a wall or a shadow across two turns, so what it renders is a pretty glow rather
than a lit scene. But nothing it did made anything worse, and per dollar it is the standout.

---

## The one new engine finding

**Nothing stops a turn that is expensive and destructive rather than repetitive.**

Gemini's second turn spent three quarters of a dollar to make the picture worse, and every existing
brake watched it happen. Those brakes count *repetition* — the same edit twice, edits that change
nothing, edits that fail to compile. This turn did none of that. It made varied, clean, compiling
edits that each did something, and collectively ruined the render.

Two things are missing, and they are separate:

1. **No cost or time signal reaches the model.** It had no idea it was fifteen times more expensive
   than its own first turn.
2. **Nothing notices a frame going from lit to flat.** The engine measured `FLAT — one uniform
   color` and passed it through as one more line of feedback, the same weight as any other.

The second is the more tractable of the two: the engine already computes the number, and "the frame
was rich and is now uniform" is a comparison it could make. Whether that should stop a turn or just
be said louder is a design call, not something this round settles.

---

## Where things stand against the previous round's predictions

**Prediction that held.** Keeping the pass-editing tools permanently loaded protects the prompt
cache. Whenever the copilot loads a tool mid-turn, the very next request drops to about 4% cache
reuse against 78–85% on the requests either side. The two tools made permanent were used heavily
this round and triggered no such reload.

**Prediction that did not.** Rewriting the batching instruction was meant to make the copilot group
its work into fewer round trips. It did not: hy4 came in at 0.528 requests per tool call against
0.552 before — flat, within noise — while gemini at 0.868 and luna at 0.649 are both worse. The
instruction is not doing what it was rewritten to do.

---

## What a next round would test

The obvious one is whether gemini's turn 1 is repeatable or was luck — it is the only evidence that
this spec is buildable in a single turn, and it rests on one sample.

The other is the destructive-turn brake: gemini's turn 2 is a clean, reproducible case to build
against, and it is sitting in the log ready to be replayed.
