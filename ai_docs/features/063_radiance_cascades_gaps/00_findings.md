# Radiance Cascades in ShaderBox — what's missing, what isn't

**Status: research CLOSED. No code was written. The direction was decided
(`17_direction.md`) and the work handed to feature 064.**
This is the summary; the numbered docs beside it carry the evidence.

**Read `README.md` first** — it is the folder index and carries the supersession map. Then this
file's VERDICT block below. `09_measurements.md`, `15_fidelity.md` and `16_stress_test.md` were
executed rather than reasoned; `13`-`17` are the review that changed the conclusion.

---

## VERDICT (after a 5-agent reliability review)

Read `13_reliability.md`, `15_fidelity.md`, `16_stress_test.md`, `14_ergonomics.md` for the
evidence. The short version:

**The GPU capability is real and survived attack.** Performance, correctness, canvas resize,
multi-node isolation and mid-pass error recovery all PASS, several bit-exactly. Radiance
cascades genuinely runs in the unmodified engine.

**The script route is not a way to work.** Not because of taste — because of hard failures on
ordinary user actions, each verified independently:

| | |
|---|---|
| `UINode.save` **crashes** | every Ctrl+S and every copilot checkpoint. Fix the mkdir and it writes 2 MB of garbage per save, then hard-crashes on load (`512 != 256`, no `dtype`) |
| revert / external node.json edit | **destroys live GL, never self-heals** — 100 frames later still wrong, error points nowhere near the cause |
| `dry_run` isolation | **25% of pixels change** with no tick between; invisible (no error, no log); **not fixable small** |
| copilot feedback | **inverted, not degraded** — `STATIC` on a demonstrably animating node, plus a confident "dead code" diagnosis. ~12/31 tools degraded, 7 lying |
| the editor and the sliders | GLSL errors mislocated + node freezes black; one slider needs a **decoy uniform with a no-op multiply** in a second file |

**And the proof itself was miswired** — 1364/1364 merge directions read the wrong slot, 30.3%
error vs a 65536-ray ground truth, now fixed to 4.5%. It had convinced me because it rendered
shadows and beat brute force 16:1. **Neither was evidence the merge was correct.**

### Three fixes owed NOW, independent of any decision

Each is a latent defect in today's codebase:

1. `ctx.gc_mode = "auto"` at context creation — every GL-touching script leaks on every save
   (206 MiB / 50 edits). Caveat: leaves a bounded GC residual, not a flat line.
2. The `textures/` mkdir in `ui_models.py::save` — that branch crashes on first contact.
3. The missing `dtype` in `core.py::load_from_dir` — the round-trip is broken for any non-`f1`
   texture.

### The recommendation

**Build the minimal engine feature; do not "play with the script first."** The deferral's stated
purpose was to learn what the UI needs — but a script **has no UI seam, generates no controls,
and bypasses `node.json` entirely**, which is exactly where the open question lives. It cannot
teach the thing it was meant to teach.

What the experiment DID teach, and it is worth the whole wave: **friction concentrates in
authoring and tuning, not in the pass chain.** A first-class design must deliver **parameters
and error locality** — not a DAG.

The shape, from the survey's two convergences and the maintainer's own prior art: source-inferred
passes (glslViewer) with config riding the uniform declaration (offline-shadertoy), implicit
ping-pong, ratio-of-output sizing through the existing `RenderShape`, bound **by name**, and
evaluation by **topological sort with memoization** (the deleted DAG had order without
memoization; freska had memoization without order).

Honest cost: `dev_flow.md` triages this as a feature AND high-blast-radius — upper-end review, a
spec-fidelity auditor, a mandatory sanitization sweep, and the copilot needs a pass-aware probe
or it inherits the same lying verdict. Not a weekend.

**If working the script way anyway before those land:** one node only, never save or export it,
set `gc_mode` first, and **turn the copilot off for it** — a lying preview plus a broken undo is
the worst possible pairing.

## The one-paragraph answer (SUPERSEDED — kept for history)

> **This paragraph is the PRE-REVIEW framing and is retracted.** It called the missing pieces
> "small" and recommended playing with the script route. The five-agent review (`13`-`16`)
> established that route crashes on save, destroys live GL on revert, and corrupts `dry_run`.
> **The current answer is the VERDICT block above.** Kept only so the reasoning arc is legible.

## What RC actually requires

From the article's **shipped source**, which contradicts its own prose on three load-bearing
points (full detail in `01_reference.md`):

- **~19 draw calls**, but passes 2-4 run only on `frame == 0` — the distance field is cached
  and invalidated on draw, so steady state is far cheaper.
- **All cascade levels share TWO fixed-size targets**, not one per level. A coarser probe grid
  is encoded *within* the fixed texture. Buffer count is bounded and small; **per-pass
  resolution is not needed.**
- **HalfFloat/Float RGBA** targets, mandatory.
- **Per-target filtering** — Nearest for JFA/seed, Linear+mipmaps for the cascades.
- A **persistent accumulating draw canvas** (mouse strokes interpolated from the previous
  point) and a cached distance field.

## What was measured, not assumed

Run on this box, NVIDIA 580.173.02, standalone EGL (`09_measurements.md`):

| Assumed blocker | Measured reality |
|---|---|
| No float targets | `dtype="f2"` — wrote `4.5` and `-2.0`, read back intact |
| No filter control | `filter`, `repeat_x/y`, `build_mipmaps()` all settable, every dtype |
| 19 passes too costly | **0.52 ms/frame at 256x256 — ~1930 fps**, with per-frame readback |

The ping-pong probe is also the falsifier for "maybe 8-bit is fine": seeded to 1.0 then six
accumulate passes, `f2` reached exactly 7.0 while `f1` **saturated at 255 on the first pass**.

So format and filtering are three unset attributes on a 15-line `Canvas` class, and cost is a
non-issue by three orders of magnitude. **Any remaining objection is about complexity or
product fit, not feasibility.**

## The gaps that are real

**1. Sequencing — nothing renders a node into another node's sampler.**

Everything needed is already present and separately exercised:
- `Node.render(canvas=...)` takes a caller-supplied target, and **the current node already
  renders twice per frame** (once into `preview_canvas`, once into its own).
- `Node.render`'s sampler branch **already accepts a raw `moderngl.Texture`** — a path plumbed
  end to end (`load_from_dir` <-> `UINode.save`) whose only producer is an unreachable
  `textures/*.bin` round-trip.
- Every node is **already live in GL simultaneously** with its own program, texture and FBO,
  so a cascade chain adds no new *kind* of resource.

What is absent is the wiring and an evaluation order. `\.canvas\.texture` filtered of
`.size`/`.glo`/`.read()` returns **zero hits** — no consumer ever binds one as a value.

**2. Input — the preview is display-only.**

`MouseState` is `{x, y}`: no buttons, no drag delta, no previous position, no hover flag.
`imgui.image_with_bg` submits no interactive item, so a click on the render reaches nothing.
There is no writable texture anywhere to paint into.

Note 042 **deliberately cut** `u_mouse` as "a SECOND write-path for the same cursor value" and
cut `MouseState.down`/`.inside` explicitly. **A spec must engage with that decision, not
silently reverse it** — its stated trigger ("a concrete stateless cursor-reactive shader is
wanted with no script") is arguably now fired.

## The history that reframes this

**ShaderBox already shipped a multi-pass DAG** (April 2025, `shaderbox/renderer.py`): per-node
FBO+texture, positional `u_input0..N` binding, per-node resolution, and a working three-pass
bloom demo. `238acb1 "re-writing this shit"` deleted it with an **empty commit body**. No
rationale exists anywhere; pickaxe over all history for multipass/render-target/ping-pong
returns nothing.

**It was abandoned in a rewrite, never argued down.** So this is revisiting an abandonment,
not overturning a decision. It also explains the vocabulary: "node" is the vestigial name of a
real graph vertex — the class outlived the graph.

Supporting: 052 already files multi-pass buffers as a legitimate future feature with the
trigger *"a user asks for a feedback/trail/reaction-diffusion effect"*, which this request
fires. The 8-bit canvas was never a decision — it is moderngl's `dtype="f1"` default, and a
float target was **never once tried**. And no doc anywhere contains the words "single-file" or
"single-pass"; the filed thesis is uniform-introspection-to-controls, and the product has moved
steadily *away* from minimalism (shader library, node scripts, multi-tab editing).

## The escape hatch, tested

A `script.py` **can** orchestrate extra passes today with zero engine changes — verified
through the real `ScriptEngine`, zero errors, correct pixels (`10_script_route.md`). Scripts
are unsandboxed by locked posture, so `import moderngl` works; a script can build its own `f2`
target and become the missing producer for the raw-`Texture` sampler branch.

**The wall is narrow:** `EngineContext` is exactly `(t, dt, frame, mouse)` — no node handle. My
probe only ran because I injected it from outside. And returning a sampler key is refused on
purpose (`"is a sampler/block — not a scriptable value"`), so the texture must arrive as a
**side effect** on `uniform_values`.

That makes both extremes wrong. Not "needs a big new architecture" — every capability is
present. Not "just write a script" either — granting node access via side-effecting writes
would falsify three guarantees that currently hold: `dry_run` promises the live node is
byte-identical after probing (a drawing script breaks it), `export_isolation` would allocate a
second GL resource set per export, and script-owned textures appear in neither `UINode.save`
nor the copilot checkpoint, escaping both persistence and revert.

## Prior art worth reusing

**freska** (his own repo) is the closest structural precedent: a node graph where TEXTURE is a
first-class pin type, one node = one shader + one lazily-sized RenderTexture whose size derives
from its input, uniforms bound **by pin name**. Its `PinKind::MANUAL` — a UI-edited parameter
that is *not* a graph edge — maps 1:1 onto ShaderBox's generated uniform controls, suggesting
**texture inputs and uniform controls should stay separate mechanisms.**

freska's evaluation order is broken and he knew it (`// TODO: this is incorrect! Nodes must be
sorted topologically` — an `unordered_map` iteration with links propagated afterwards, so an
N-chain lags up to N-1 frames). The deleted ShaderBox DAG had the opposite pair: correct
pull-recursion order, **no memoization** (a diamond re-renders a shared ancestor per consumer).
**Between them the right shape is topological sort WITH memoization.**

Clean negatives across ~115 repos: he has **never** written a ping-pong buffer, a downsample
chain, a JFA, or an SDF baked to a texture. `py2glsl`, his only other moderngl project, is
single-pass. ShaderBox would be the first moderngl multipass in the corpus.

## What the ecosystem converged on

Two signals stronger than any single tool's choice (`11_playground_survey.md`):

1. **Every tool that sizes render targets defaults to ratio-of-output**, absolute as opt-in.
2. **Nobody makes the user manage a ping-pong pair** — self-reference is implicit in Shadertoy,
   ISF, glslViewer and shadertoy-local alike. **If ShaderBox copies one idea, this is it.**

The model closest to ShaderBox's grain is **glslViewer's inference**: write
`uniform sampler2D u_doubleBuffer0;` and the tool infers the pass, recompiles with a `#define`,
and ping-pongs it — the same introspect-and-generate mechanism ShaderBox already runs on,
extended from uniforms to buffers. Its cost is exposing no per-pass format/filter/resolution,
which is exactly what RC needs, so it cannot be adopted whole.

## The open decision — handed to feature 064

Not *whether* it is feasible — that is settled. The question is **which seam expresses a pass
chain**. Feature 064 revised its approach: the seam is decided AFTER a superset scenario
pins the requirements and the UI is designed against it (`../064_multistep/00_scenario.md`). The
options differ in what they cost the product's grain:

- **Inference from uniform names** (glslViewer) — zero config, `node.json` untouched, but no
  per-pass control without inventing a syntax for it.
- **A pragma/comment on the declaration** (offline-shadertoy) — wiring rides the declaration it
  configures so the two **cannot desync**, which is the exact failure KodeLife documents in
  itself.
- **A `PASSES` block** (ISF) — most expressive, but ShaderBox would want it in `node.json`,
  which is today **app-written derived state**. Hand-authored declarations make it a file the
  user edits and the app must not clobber. That is a real change in what `node.json` means.
- **Node-to-node wiring** (the deleted DAG, freska) — most powerful, but reintroduces an
  inter-node document model the app currently does not have.

Explicitly NOT recommended: SHADERed's positional slot binding (a demonstrated footgun — the
same slot is `posTex` in one shipped example and `clr` in another; a tool that introspects
uniforms should bind by name), and a spatial node-graph editor (a different product).

## Traps for whoever implements this

- `texture.repeat_x/y` **default to `True` (GL_REPEAT)** in moderngl — wrong for a feedback
  target, and RC's edge-clamp requirement makes it actively wrong.
- `ctx.sampler()` state **leaks across passes** and does not clear on `texture.use()`;
  `ctx.clear_samplers()` is needed per pass. Invisible in single-pass code — i.e. exactly the
  regime being left.
- **V3D goes blank** on a >=256px canvas rendered hundreds of times without a per-frame
  `texture.read()` (mean alpha ~7). Read or flush each frame.
- The glyph tables already hold **~600 of ~1024 constant-register slots**; large new uniform
  arrays risk `C6020: Constant register limit exceeded`.
- **Every render encode shares ONE post-swap firing point.** "A NEW render entry point MUST
  route its encode here, never call it inline."
- Any new per-node state must be reachable from `UINode.save`, or it escapes both persistence
  and copilot revert.
- `RenderShape` is the decided size vocabulary — per-pass sizing must respect it rather than
  reintroducing raw dims.

## Two stale in-tree comments found en route

Both are real defects, unrelated to RC, worth fixing in whatever wave touches these files:

- `paths.py::shader_lib_root` still describes an `#include "name"` mechanism. **No such
  mechanism exists** — resolution is by `SB_*` identifier scan.
- `core.py::Node.render` claims "the caller passes an explicit u_time on every real render path
  (the live loop, export, the probe)". **The live loop does not** — both live sites call
  `render()` bare, so live `u_time` is `time.monotonic()`, a different clock from the
  `glfw.get_time()` used for script `dt` in the same frame.
