# 068 — Iterated passes, and radiance cascades on top

**Status: LANDED** (`6871d07`, `659961d`, `3d98c1d`, `ac747d6`), reviewed by a six-agent round
whose findings are folded in below. Two decisions were superseded during that round — D3 and D7,
each marked in place.

Two deliverables, in order: the engine gains a way to run one pass N times within a frame, and
that capability is used to build radiance cascades — shipped as an example document and taught
as a step-by-step tutorial the maintainer follows by hand.

Prior wave: `../063_radiance_cascades_gaps/` researched whether ShaderBox could host RC and
concluded the engine had no representation of a pass chain. **Feature 065 built that chain**, so
most of 063's "irreducible floor" is now met. Read `063/01_reference.md` for the algorithm; read
its `README.md` supersession map before trusting any recommendation in it.

---

## Goal

1. **Iterated passes.** A pass may declare `iterations: N`; the engine draws it N times in
   sequence within one frame, advancing its ping-pong between each, and tells the shader which
   iteration it is. This is a general primitive: nothing about radiance cascades enters the
   engine.
2. **Radiance cascades**, built on it, shipped in `shaderbox/resources/document_examples/`. (The
   original "draw with the mouse" half was retracted — see D7.)
3. **A tutorial** — a local HTML file, handed over by path — that walks the build one pass at a
   time, so the maintainer can implement it themselves and learn RC in the process.

## Out of scope

- **Per-iteration target resizing.** `TargetConfig.scale` stays one number per pass, so an
  iterated pass cannot halve its resolution per step. This blocks a true mip-pyramid bloom; it
  does NOT block RC, whose cascade levels all share one fixed-size target
  (`063/01_reference.md`: "two shared targets, not N"). *Trigger to revisit: the first effect
  that needs a resolution pyramid — say a real bloom replacing the Bloom Chain's single blur.*
- **Derived iteration counts** (`"log2_max_dim"` and friends). Rejected in D3 below.
- **Cross-pass iteration** — a group of passes cycled together. RC does not need it (its two
  iterated stages are each a single shader). *Trigger: an effect needing two shaders alternating
  within one loop.*
- **Re-vendoring the editor, shipping, or itch.** Unrelated.

## Design decisions

**D1 — The iteration loop lives INSIDE a pass's turn, not by repeating it in the plan order.**
`assert_plan_invariants` fails on "a pass appears twice in the order", and that guard exists
because the bug it catches "reads as slow rather than wrong" (065). Expanding `["jfa"]` into nine
entries would mean weakening it, so instead `plan_passes` is untouched: the order still holds one
entry per pass, and `Document.render` loops N times when it reaches an iterated pass. The
draw-once invariant keeps its exact current meaning — one *pass turn* per frame.

**D2 — Two engine-driven uniforms, carrying the INDEX and nothing derived.**
`u_pass_iteration` (0-based, float) and `u_pass_iterations` (the total, float). Both join
`ENGINE_DRIVEN_UNIFORMS`, so they are excluded from the generated UI and from `document.json`
exactly like `u_time`. The shader derives its own parameter — JFA's offset is
`pow(2.0, u_pass_iterations - u_pass_iteration - 1.0)`, one line. **No `u_jfa_offset`**: a
uniform named for an algorithm is that algorithm leaking into the engine.

**D3 — `iterations` is a plain int the author sets. RETRACTED: the warning half was unsound.**
*Superseded during the review round (commit `ac747d6`).* The shipped check assumed a base-2
chain and fired on this repo's own base-4 cascade pass at its shipped canvas size, telling the
user a correctly-configured stack was "subtly wrong". The engine cannot distinguish a base-2
jump flood from a base-4 cascade stack, so a check assuming either is wrong for the other by
construction — it is not tunable. What stands: the count is the author's, the help text explains
what it means, and nothing warns. Original reasoning, kept because the trade-off it names is
real:
A canvas resize changes how many JFA steps a correct distance field needs (9 at 512, 10 at 1024).
The alternative — a small enum of derived formulas — bakes two RC-specific formulas into
`graph.json`'s model and is a mini expression language pretending not to be one. So the count
stays dumb, and the dangerous half is fixed instead: silently-wrong output becomes a visible
warning. The failure 063 warned about is precisely this shape — "a plausible render is not a
numerical check."

**D4 — Bounded, like every other `graph.json` number.** `iterations: int = Field(default=1, ge=1,
le=64)`. `TargetConfig` already documents why: these knobs live in a file where nothing
type-checks them, and an unbounded count is a frame-time bomb. 64 covers JFA at 2^64 and every
plausible cascade depth.

**D5 — Ping-pong swaps PER ITERATION for a self-reading iterated pass.** This is the real engine
work. `begin_frame` swaps feedback once per frame, which is correct for a frame-to-frame trail
and wrong for an iteration chain: all N runs would read the same stale texture and the chain
would not advance. An iterated pass that reads itself swaps between iterations instead. A pass
with `iterations: 1` keeps today's behaviour byte for byte.

**D6 — RC ships as ONE example document, not a scene of fragments.** The tutorial builds it in
steps, but the artifact is a single document whose passes are the finished stages.

**D7 — Drawing by script. RETRACTED: the engine cannot deliver it.**

**Retraction lifted by 069 (W-G).** The script engine now addresses a named pass and `ctx.mouse`
carries `down` plus the previous position, which is exactly the trigger this retraction records.
The tutorial's paint step is rewritten against them in 069 W-H.

*Superseded during the review round (`ac747d6`).* The scene is now built analytically in
`paint.frag.glsl` from SDFs and `u_time`, with no script — the same shape the Bloom Chain example
uses.

Two independent reasons the original could not work, both found by execution rather than reading:
`ProjectSession.tick` binds the script engine to a document's **OUTPUT** pass, so a script can
only drive uniforms declared there — a brush uniform on `paint` was dropped as an orphan key every
frame and the example rendered BLACK in the app. And `ctx.mouse` carries position only, no
buttons, so "drag to paint, right-drag for walls" was never expressible.

This also cost the feature its "stress the scripting path" goal, which is worth stating plainly
rather than quietly dropping. *Trigger to revisit: a script engine that can address a named pass
rather than only the output.*

## Files touched

| File | Change |
|---|---|
| `shaderbox/pass_graph.py` | `PassEntry.iterations` (D4). The D3 warning landed and was then retracted — see D3. |
| `shaderbox/core.py` | `ENGINE_DRIVEN_UNIFORMS` += the two names; bind them in `Pass.render` (D2) |
| `shaderbox/document.py` | the iteration loop in `render`; per-iteration feedback swap (D1, D5) |
| `shaderbox/popups/pass_settings.py` | an `iterations` control on the pass-settings modal |
| `tests/test_pass_graph.py` | count bounds (the warning's tests went with D3) |
| `tests/test_document_graph.py` | N draws happen; ping-pong advances per iteration; `iterations: 1` unchanged |
| `shaderbox/resources/document_examples/<uuid>/` | the RC document (D6) |
| `ai_docs/features/068_radiance_cascades/tutorial.html` | the walkthrough |

## Manual verification

Each step fails for exactly one reason.

1. **The loop runs.** A pass with `iterations: 4` whose shader writes `u_pass_iteration / 4.0`
   renders 0.75, not 0.0. *Falsifier: with the loop broken it renders 0.0.*
2. **Ping-pong advances per iteration.** A self-reading pass with `iterations: 8` that adds 1 per
   run reads 8, not 1. *Falsifier: a per-frame swap gives 1.*
3. **`iterations: 1` is unchanged.** The Bloom Chain example renders bit-identically to its
   pre-change output. *Falsifier: any pixel differs.*
4. ~~**The resize warning fires.**~~ Withdrawn with D3 — there is no warning to fire.
5. **RC renders light.** The example shows a lit region around a painted emitter with a shadow
   behind a painted occluder. *Falsifier: uniform brightness, or black.*
6. **RC is numerically right, not just plausible, AND the check itself is sensitive.**
   `oracle.py` measures the merge against a converged brute-force reference over bounced light:
   1.0134 energy ratio, 3.65% relMAE, matching 063's corrected 4.5%. **The gate is
   mutation-tested** — clean 3.6%, merge disabled 98.3%, transposed slot 29.9% (063 measured
   30.3% for that same bug). The second half is not optional: the first version of this oracle
   scored 1.2087 and called it an inherent artifact, when it was measuring a broken reference,
   and an earlier version of the metric passed a stack computing zero global illumination.
7. ~~**Drawing works.**~~ Withdrawn with D7 — there is nothing to drag.

Checks 1-3 and 6 are headless; 5 needs `make run`. 4 and 7 are withdrawn.

## Locked by the maintainer

- **B+ from the start** — no hand-authored-15-passes stage first.
- **Tutorial granularity is the agent's call**, chosen to make every mechanism land. It covers
  everything BOTH articles cover, rewritten against this engine, with explanation per step. One
  example document holds the finished product.
- **Tutorial lives in `ai_docs/` and is never shipped.** It is a learning + dogfooding artifact.
- **Primary sources: fetched** (`jason.today/gi`, `jason.today/rc`), as authorized.

## What reading the primary sources changed

Both articles were read in full after 063's summary. Three things the spec now rests on:

1. **The merge's slot address is the easiest thing here to get wrong, and 063's corrected proof
   is the reference.** Addressing it with a slot dimension in one component and a probe spacing
   in the other `063/rc_proof.py` (post-fix) uses `usp = sp * 2.0` for BOTH:
   `vec2(mod(uS, usp), floor(uS / usp))`. This is exactly the class 063 warned about — its own
   first attempt had 1364/1364 directions reading the wrong slot at 30.3% error while rendering
   convincing shadows. **Build the merge from `rc_proof.py`.** *Corrected during the review
   round: the article's SHIPPED demo source agrees with this implementation — only its base-16
   prose differs. An earlier draft of this spec accused it of a bug it does not contain.*
2. **The GI article supplies the stages RC assumes** and is not optional reading: the drawing
   surface (`sdfLineSquared`), the seed pass (`vec4(vUv * alpha, 0, 1)`), the JFA 3x3 kernel, the
   distance-field pass, and naive raymarching as the pedagogical control. The tutorial covers all
   of it — a reader who skips to cascades has no distance field to march.
3. **Naive GI needs temporal accumulation; RC does not.** The GI article blends
   `mix(finalRadiance, prevRadiance, 0.9)` with a time-seeded noise offset to hide ray noise. RC
   drops both — deterministic angles, no noise, no history. Worth teaching as the contrast that
   motivates cascades, and it means the naive stage needs a `persist` target the RC stage does not.

## Owed before check 6 can run

`063/rc_proof.py` **no longer imports** — it uses `from shaderbox.core import Node`, and 065
renamed that to `Document`/`Pass`. Its GLSL (the corrected merge) is intact and is the algorithmic
reference regardless. Repairing the harness is a prerequisite for the numerical check, and is
small: it drives raw moderngl and only touches the engine at that one import. *If the repair turns
out not to be small, the fallback is to port its GLSL into a standalone script under this
feature's folder — the value is the shader, not the harness.*
