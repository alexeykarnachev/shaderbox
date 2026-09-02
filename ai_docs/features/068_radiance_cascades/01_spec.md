# 068 — Iterated passes, and radiance cascades on top

**Status: SPEC, awaiting plan-lock.**

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
2. **Radiance cascades**, built on it, as a playable document: draw light and occluders with the
   mouse, see bounced light. Ships in `shaderbox/resources/document_examples/`.
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

**D3 — `iterations` is a plain int the author sets; the engine WARNS when the canvas outgrows it.**
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

**D7 — Drawing is a script, not an engine feature.** RC is only interesting if you can paint
light and occluders. `ctx.mouse` and a `persist` target already carry this: a `script.py` tracks
the mouse and the shader accumulates into a persistent canvas. No engine change, and it exercises
the scripting path — which the maintainer explicitly wanted stressed.

## Files touched

| File | Change |
|---|---|
| `shaderbox/pass_graph.py` | `PassEntry.iterations` (D4); a `GraphError` when the canvas outgrows a count (D3) |
| `shaderbox/core.py` | `ENGINE_DRIVEN_UNIFORMS` += the two names; bind them in `Pass.render` (D2) |
| `shaderbox/document.py` | the iteration loop in `render`; per-iteration feedback swap (D1, D5) |
| `shaderbox/popups/pass_settings.py` | an `iterations` control on the pass-settings modal |
| `tests/test_pass_graph.py` | count bounds, the resize warning |
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
4. **The resize warning fires.** A JFA pass with `iterations: 9` on a 512 canvas is clean; resize
   to 1024 and the warning appears naming the pass. *Falsifier: silence.*
5. **RC renders light.** The example shows a lit region around a painted emitter with a shadow
   behind a painted occluder. *Falsifier: uniform brightness, or black.*
6. **RC is numerically right, not just plausible.** Compare against
   `063/rc_proof.py`'s corrected implementation on the same input. **This is the check 063's whole
   wave exists to demand** — its proof rendered convincing shadows while 30.3% wrong. A visual
   match is not a pass.
7. **Drawing works** (needs a display): drag paints an emitter, light updates; drag an occluder,
   a shadow appears.

Checks 1-4 and 6 are headless. 5 and 7 need `make run`.

## Locked by the maintainer

- **B+ from the start** — no hand-authored-15-passes stage first.
- **Tutorial granularity is the agent's call**, chosen to make every mechanism land. It covers
  everything BOTH articles cover, rewritten against this engine, with explanation per step. One
  example document holds the finished product.
- **Tutorial lives in `ai_docs/` and is never shipped.** It is a learning + dogfooding artifact.
- **Primary sources: fetched** (`jason.today/gi`, `jason.today/rc`), as authorized.

## What reading the primary sources changed

Both articles were read in full after 063's summary. Three things the spec now rests on:

1. **The article's merge snippet is WRONG as printed, and 063's corrected proof is the
   reference.** The article computes the upper-cascade slot as
   `vec2(mod(index, sqrtBase), floor(index / upperSpacing))` — mixing a slot dimension with a
   probe spacing in one expression. `063/rc_proof.py` (post-fix) uses `usp = sp * 2.0` for BOTH:
   `vec2(mod(uS, usp), floor(uS / usp))`. This is exactly the class 063 warned about — its own
   first attempt had 1364/1364 directions reading the wrong slot at 30.3% error while rendering
   convincing shadows. **Build the merge from `rc_proof.py`, not from the article.**
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
