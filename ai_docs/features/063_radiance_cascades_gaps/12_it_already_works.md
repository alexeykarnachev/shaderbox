# Proven: RC runs in the unmodified engine, today

**This overturns the headline of `00_findings.md`.** A devil's-advocate pass was tasked with
attacking the "ShaderBox needs a multipass architecture" framing. It produced a **working
radiance cascades implementation running inside the real, unmodified ShaderBox engine, driven
from a `script.py`.** Independently reproduced here before being written down.

```
node compile errors: []
script compile errors: {}
tick errors: {}
FULL engine loop (script multipass + Node.render): 0.45 ms/frame -> 2212 fps @512x512
  of which the cascade chain alone: 0.38 ms
node canvas mean: 23.1
```

Artifacts committed beside this doc: `rc_proof.py` (repro: `uv run python` it from the repo
root), `rc_proof.png`, `rc_bruteforce_control.png`.

`rc_proof.png` shows two emitters (a warm circle, a blue capsule) with correct radial falloff,
**a vertical wall casting a real shadow**, and a dark circle occluding the warm source. That is
propagated radiance with visibility, not a gradient.

## Why it works — four facts that were each known, never connected

1. **A script gets the live GL context.** `moderngl.get_context()` inside `update()` returns
   the app's context. `__builtins__` is real — a **locked design posture**
   (`behavior.py::_build_globals`: "No sandbox (a personal IDE)").
2. **A script reaches its own `Node` with no injection.** `_tick_script` holds `node` in frame
   locals at stack depth 2; a `sys._getframe()` walk finds it. Ugly, and it works today.
3. **Float targets, per-texture filtering, mipmaps and clamp-to-edge exist on the raw
   context.** `03_uniforms.md`'s "no float texture path exists anywhere" is true **of
   `Canvas`** — but `Canvas` is not the only way to make a texture.
4. **The result binds back through the raw-`Texture` sampler branch** that `04_render_pipeline.md`
   correctly flagged as "plumbed end to end and has no producer". **The script is the missing
   producer.**

`04` found the loose thread and `05` found the other ("a boundary, not a sandbox"). The error
was treating the engine's *sanctioned surface* as the boundary of the possible, when the
codebase's own locked decision puts the whole GL context one `import` away.

## The pass count collapses too

The working implementation has **no seed pass, no JFA, no distance-field pass** — the scene is
sphere-traced analytically in the cascade shader, from the same circle/box/union primitives the
`SB_sd_*` library already ships. **19 passes -> 7.**

That also retires the strictest format requirement: full `Float32` was needed only for the
UV-encoded JFA seeds (`01_reference.md` is right that half-float would quantize them). With no
JFA, `f2` suffices throughout.

**What is lost: painting arbitrary occluders with the mouse.** That is essential to the
*article's demo*, not to RC. RC propagates radiance through a scene; how the scene is defined
is orthogonal.

## The single-pass alternative fails on quality, not speed

Measured at 512x512: 16 rays 0.08 ms, 256 rays 1.10 ms, 1024 rays 4.42 ms, 4096 rays 17.5 ms
(57 fps — over budget).

But the decisive result is `rc_bruteforce_control.png`: **brute force at 256 rays/pixel — 16x
more rays than the cascade version uses — renders essentially black** (mean luminance 1.3 vs
the cascade version's 23.1). A small light subtends a tiny solid angle, so uniform sampling
almost always misses it. This is exactly the sampling problem RC exists to solve, demonstrated
rather than asserted.

**Where the data dependency truly bites: redundancy, not parallelism.** Nothing *forces*
sharing — a single pass could recompute the hierarchy per pixel. The multipass version computes
each probe's rays once and shares them across the tile the probe serves; recomputing per-pixel
is ~5460 rays/px instead of ~4, a ~1000x waste. Multipass is a **sharing optimization, not a
correctness requirement** — though at that ratio the distinction is academic.

## Two real bugs found, both independent of RC

**1. GL leak on every script hot-reload — VERIFIED INDEPENDENTLY.**

`grep -rn "gc_mode" shaderbox/` returns **nothing**, so moderngl's default applies:

```
default gc_mode: None
glo reuse without gc_mode:   1 -> 2   (leaked)
glo reuse with gc_mode="auto": 3 -> 3  (reused)
```

The engine rebuilds the Behavior on each script edit and nothing releases the old one's GL
objects. Measured at **80 MB of textures leaked over 40 script edits**. One-line fix:
`ctx.gc_mode = "auto"` at context creation. **This is a latent defect today, with or without
this feature** — any script that allocates GL leaks on every save.

**2. Blend-state leakage across the script/render boundary.** A script leaving `BLEND` enabled
corrupts the subsequent `Node.render` (verified: the node rendered fully transparent instead of
red). `fbo.use()` restores the viewport but **not** blend state. Fixed by a
`ctx.disable(moderngl.BLEND)` epilogue.

## What is genuinely, unavoidably missing

Almost nothing:

1. **A supported way for a script to reach its `Node`.** Today it is a `sys._getframe()` walk.
   A sanctioned route is **one field on `EngineContext`**, not an architecture.
2. **`ctx.gc_mode = "auto"`** — a bug fix owed regardless.
3. **Mouse buttons/drag**, *only if* the painting interaction is wanted. Available today via a
   direct `from imgui_bundle import imgui` in the script — which notably does **not** reverse
   the recorded 042 `u_mouse` decision, because it is not a second engine write-path; it is a
   script reading imgui itself.

**Not missing:** float render targets, per-target filtering, mipmaps, clamp-to-edge, multiple
sequenced draws, ping-pong buffers, texture-to-sampler binding, inter-frame persistence, export
compatibility (a fixed-size cascade texture stretches onto a 640x480 export target correctly,
because uv-space sampling is resolution-independent).

## What this means for the decision

The seam question in `00_findings.md` is **not urgent**. The cheapest probe of the design space
is to play with the working script version first — then decide whether pass chains deserve
first-class engine support, knowing from use what the UI actually needs.

Building a node graph or a `PASSES` block *before* that is designing the abstraction before
meeting the use case. The costs of waiting are two one-line fixes; the cost of guessing wrong
is a persistence format and a UI to live with.

**Caveat, stated plainly:** the script route works but reaches around the engine's contract —
`10_script_route.md` records why that matters (`dry_run` promises the live node is unchanged
after probing; `export_isolation` would allocate a second GL set per export; script-owned
textures escape `UINode.save` and copilot revert). Those are real and unresolved. They are
arguments for eventually giving pass chains a sanctioned home — **not** arguments against
playing with it now.
