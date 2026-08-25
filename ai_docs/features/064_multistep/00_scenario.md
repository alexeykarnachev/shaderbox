# 064 — Multi-step nodes: the superset scenario

**Status: DRAFT, awaiting maintainer sign-off.** Nothing downstream may start until this is signed.

This document exists to be the **anchor**. It defines ONE scenario chosen to be a superset of the
effect family ShaderBox is for, and derives a numbered requirement list from it. Every downstream
decision — UI, backend, implementation — must trace to a numbered requirement here. **A proposal
that cannot name the requirement it serves is invention, and gets cut.**

## Why the approach was reset

An earlier spec (`064_multipass/01_spec.md`, deleted unimplemented) opened with "where do pass
declarations live?" That is a seam question, and it was asked before establishing what the feature
must be able to EXPRESS. Answering it first would have locked a syntax against an unexamined
requirement set — and one of its own options was already ruled out by a requirement it had not yet
written down.

The corrected order: **scenario -> requirements -> UI/UX -> backend -> implementation.** The UI is
designed against the scenario; the backend is designed against the locked UI; the seam falls out of
both instead of preceding them.

## What 063 established that this inherits

Read `../063_radiance_cascades_gaps/README.md` first (it carries the supersession map — some early
recommendations are retracted). Load-bearing conclusions:

- **Cost is a non-issue.** 19 passes at 256x256 `f2` with per-frame readback = 0.52 ms/frame,
  ~3% of a 60 fps budget. Any argument against this feature must be about complexity or product
  fit, never milliseconds.
- **8-bit is demonstrated fatal** for accumulation: seeded to 1.0 then six accumulate passes, `f2`
  reached exactly 7.0 while `f1` saturated at 255 on the FIRST pass. Not an argument from
  authority — a measurement on this box.
- **Float targets, per-target filter/wrap/mipmaps are three unset attributes**, not a subsystem.
  `Canvas._init` simply never passes them.
- **Friction lives in authoring and tuning, not in the pass chain.** (`14_ergonomics.md`.) This is
  the single most important design input: the feature must deliver **parameters and error
  locality**, not a DAG for its own sake.
- **The script-GL route is abandoned** (`17_direction.md`). Its failure modes are the negative
  spec: crashes on Ctrl+S, revert destroying live GL with no self-heal, silent pixel corruption,
  and inverted copilot feedback. Every one is something this feature must NOT reproduce.

## The scenario: "Living Scene"

One node. It does all of the following at once.

A scene of a few glowing shapes that move. Light spills from them and bounces around corners —
**radiance cascades**, built from ~8 steps at shrinking resolutions, where the smaller levels are
merged back up into the larger ones.

The lit picture feeds a **bloom**: bright regions are extracted, blurred wide across several
shrinking steps, then added back over the original. This chain hangs off to the side of the main
result rather than sitting in its line.

The scene leaves **trails** — what was bright a moment ago fades out instead of vanishing. That
step reads its own picture from the previous frame.

**Smoke** drifts through the scene: a simulation that each frame reads its own previous state,
advects it, and writes a new state. It never restarts; it accumulates for as long as the node
is open.

A final step **combines** the lit scene and the smoke, and tonemaps the result to something
displayable.

While working, the author can **watch any stage**: just the smoke, just the bloom's blurred layer,
just cascade level 4, then back to the finished picture.

Three **sliders** exist: smoke drift speed, bloom threshold, trail fade rate — each belonging to a
specific step, driven by the ordinary uniform-introspection controls that already exist.

## The requirements

| # | Requirement | Demanded by |
|---|---|---|
| R1 | Several steps inside one node | the whole scenario |
| R2 | Steps at differing resolutions | cascades, bloom blur |
| R3 | One step reading several earlier steps | cascade merge, final combine |
| R4 | A branching order, not a straight line | bloom hangs off the main line |
| R5 | A step reading its OWN output from the previous frame | trails, smoke |
| R6 | State that persists across frames indefinitely | smoke |
| R7 | Buffers holding values above 1.0 and below 0.0 | cascades, bloom (measured fatal in 8-bit) |
| R8 | Per-step filter and edge/wrap behaviour | blur needs linear; feedback needs clamped edges |
| R9 | Viewing any intermediate step's output | debugging every part of it |
| R10 | Uniform controls belonging to a specific step | the three sliders |

**Every simpler effect is a subset.** A blur chain is R1+R2+R8. Bloom is R1+R2+R3+R4+R7+R8.
Reaction-diffusion is R1+R5+R6+R7. Plain radiance cascades is R1+R2+R3+R4+R7+R8. Nothing in the
full-screen fragment-shader effect family requires an eleventh row.

## Deliberately out of scope

Chosen boundaries, not overlooked ones. Each would change what ShaderBox is; wanting one later is a
new decision, taken deliberately.

- **Per-object simulation** (particles with individual positions). Everything above operates on a
  whole image at once; a million independent particles is a different machine (compute shaders,
  storage buffers).
- **3D geometry** — meshes, camera, depth buffer. Full-screen only.
- **Volumes / 3D textures.** Trigger: 2D is done and the maintainer wants volumetrics.
- **Reading pixels back to the CPU to branch on them mid-chain.** Everything flows forward.
- **Effects spanning several nodes.** A node stays one self-contained document. Trigger: a user
  genuinely needs one node's live output in another and a file bind is insufficient.
- **MRT** (several outputs from one step). 8 attachments are available (measured) but nothing in
  the scenario needs them. Trigger: an effect needs a G-buffer-shaped step.

## The open question this scenario raises

**R6 is the odd one out and must be designed deliberately.** Every other requirement describes work
that starts and finishes within a frame. R6 describes a buffer that IS the effect — the smoke's
accumulated state. That raises questions nothing else here does:

- On save and reload, does the smoke resume as it was, or start cold?
- What does copilot revert do to it? (`05_node_model.md`: a node's mutable state is
  `UINode.save()` output + `script.py`. Anything unreachable from there escapes persistence AND
  revert.)
- What does export do — start cold and warm up N frames, or capture the live state as-is?

These are answerable, and they are a genuinely separate design question from "how do steps
connect." They are called out here so the UI/UX phase engages them rather than discovering them at
implementation time.

## The fixes owed regardless

Each is a latent defect in today's codebase, independent of this feature, landing as their own
commit BEFORE any feature code:

1. **`ctx.gc_mode = "auto"`** — `grep -rn "gc_mode" shaderbox/` returns nothing, so moderngl's
   `None` default applies and dropped GL objects never free (measured: 103 textures / ~206 MiB
   after 50 script edits). Caveat (`16_stress_test.md`): `auto` leaves a bounded residual because
   the VAO<->program<->buffer graph is cyclic — a lag, not a leak. **Siting is an open detail:**
   the codebase never CREATES a context, it only calls `moderngl.get_context()`; the setter needs a
   real home — `App.__init__` right after `glfw.make_context_current(window)` is the likely one,
   but confirm rather than assume.
2. **The `textures/` mkdir** in `ui_models.py::UINode.save` — the raw-`Texture` branch writes
   `textures/<name>.bin` but only `dir` is mkdir'd. The branch has demonstrably never executed; it
   raises `FileNotFoundError` on first contact.
3. **The missing `dtype`** — verified `data size mismatch 512 != 256` for an `f2` texture. BOTH
   sides need it: `core.py::Node.load_from_dir` passes no `dtype` to `gl.texture(...)`, AND
   `ui_models.py::UINode.save` never records one (it writes only `file_path`/`size`/`components`).
   Fixing the loader alone is impossible — there is nothing to read. R7 makes non-`f1` targets a
   first-class case, so this blocks the feature.

A fourth, in the same area and cheap: **the false comment in `core.py::Node.render`** claims "the
caller passes an explicit `u_time` on every real render path (the live loop, export, the probe)".
The live loop does not — both live sites call `render()` bare, so live `u_time` is
`time.monotonic()`. This feature rewrites that exact funnel.
