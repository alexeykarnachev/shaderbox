# 064 — Engine-native multipass

**Status: DRAFT, not plan-locked.** Section `## Open questions for the user` must be answered
before implementation starts (`dev_flow.md` step 3).

**Size: HIGH-BLAST-RADIUS.** It touches `Node.render` (the single funnel every render path goes
through), `Canvas`, the persistence format, the uniform panel, and the copilot's probe. Per
`dev_flow.md`: upper-end review (2 pre-impl, 3+ post-impl including a **spec-fidelity auditor**),
plus a **mandatory sanitization sweep** even though it would not normally warrant one.

## Goal

A node can declare additional render passes, so effects that need a chain of draws — radiance
cascades, blur/bloom chains, reaction-diffusion, feedback trails — are authored the way ShaderBox
already authors everything else: **write a shader, get controls, see errors in the strip, hot-reload
on save.**

The research (feature 063, 21 documents) established that the GPU capability is entirely present —
float targets, per-target filtering, ping-pong and a 19-pass chain at 0.52 ms are all **measured
working** on this box (`063_radiance_cascades_gaps/09_measurements.md`). Nothing here is a GPU
question. What is missing is that the ENGINE has no representation of a pass chain, so a shader
author cannot express one and the engine cannot persist, checkpoint, probe or export it.

**The success test:** a radiance-cascades node authored entirely through ShaderBox's normal
surfaces — real `.glsl` files, real uniform sliders, real error strip — with no Python and no GL
calls outside the engine.

## What 063 established that this spec inherits

Read `063_radiance_cascades_gaps/00_findings.md` first; its verdict block is the summary. The
load-bearing conclusions:

- **Cost is a non-issue.** 19 passes at 256x256 f2 with per-frame readback = 0.52 ms/frame
  (~1930 fps). A cascade chain is ~3% of a 60 fps budget.
- **Float + filtering are three unset attributes**, not a subsystem. `ctx.texture(size, 4,
  dtype="f2")` works; `tex.filter`, `tex.repeat_x/y`, `tex.build_mipmaps()` all work. `Canvas`
  simply never passes them.
- **`Node.render(canvas=...)` already takes a caller-supplied target**, and the current node
  already renders twice per frame. Multi-draw is the status quo; **sequencing** is what is absent.
- **The raw-`moderngl.Texture` sampler branch is already plumbed** end to end and has no producer.
- **8-bit is genuinely fatal** for cascade merging — demonstrated: seeded to 1.0 then six
  accumulate passes, `f2` reached exactly 7.0 while `f1` saturated at 255 on the FIRST pass.
- **Friction lives in authoring and tuning, not the pass chain** (`14_ergonomics.md`). This is the
  single most important design input: the feature must deliver **parameters and error locality**,
  not a DAG.
- **The script-GL route is abandoned** (`17_direction.md`), and its failure modes are the negative
  spec — every one of them is something this feature must NOT reproduce.

## Design decisions (numbered; lock-in only)

**D1. Passes belong to a node, not between nodes.** A pass chain is internal to one node. No
inter-node graph, no edges, no evaluation order between documents. Rationale: `04`/`05` established
nodes are independent documents with no graph, and `11_playground_survey.md` found the node-graph
model "fights hardest" — it makes the graph the document and takes the single-file property with
it. Revisit if a user genuinely needs one node's output in another (today that is a file bind).

**D2. Every pass is a real file on disk.** A pass's shader is a `.glsl` file in the node directory,
never a string embedded elsewhere. Rationale: `14_ergonomics.md` — shaders-as-strings cost the
error strip, `SourceMap`/`#line` remapping, click-to-jump, `SB_*` splicing, and syntax
highlighting, and the mislocated-error problem is **structurally unfixable** while a shader is a
string. This decision is what buys error locality.

**D3. Pass uniforms generate controls exactly like node uniforms.** No second mechanism, no decoy
declarations. Rationale: `14_ergonomics.md` — the script route needed a decoy uniform plus a no-op
multiply per tunable, which is the failure this feature exists to avoid.

**D4. Ping-pong is implicit.** A pass that reads its own output gets double-buffering from the
engine; the user never manages a pair. Rationale: `11_playground_survey.md` — unanimous across
Shadertoy, ISF, glslViewer and shadertoy-local. Nobody makes the user do this.

**D5. Per-pass sizing defaults to ratio-of-output.** Absolute pixels are opt-in, and must go
through the existing `RenderShape` vocabulary rather than raw dims. Rationale: four independent
designs converged on ratio-default (`11`), and `conventions.md` already decided "Render output size
is ONE named vocabulary, not raw dims per caller."

**D6. Inputs bind BY NAME, never by slot index.** Rationale: `11` — SHADERed's positional
`register(t0)` is a demonstrated footgun (the same slot is `posTex` in one shipped example and
`clr` in another), and ShaderBox already introspects uniforms by name.

**D7. Evaluation is topological sort WITH memoization.** Rationale: `08_prior_art.md` — the
maintainer's own deleted DAG had correct pull-recursion order but **no memoization** (a diamond
re-renders a shared ancestor per consumer); freska had per-node targets but **broken order** (an
`unordered_map` iteration under his own `// TODO: this is incorrect!`). Neither prior design got
both; this one must.

**D8. A pass target's format is declarable, defaulting to the canvas format.** At minimum `f2`
must be reachable, since 8-bit is demonstrated fatal for accumulation. Filter and wrap likewise
declarable, defaulting to linear/clamp — **not** moderngl's `GL_REPEAT` default, which is wrong
for a feedback border.

**D9. Pass state must be reachable from `UINode.save`.** Rationale: `05_node_model.md` — the
codebase's own definition of a node's mutable state is `UINode.save()` output + `script.py`.
Anything not reachable from there escapes both persistence and copilot revert. This is what the
script route got wrong.

**D10. The copilot's probe must see passes, or say it cannot.** `_render_facts_for` must either
evaluate the chain or stamp the facts line honestly. Rationale: `13_reliability.md` — a confident
stale frame is strictly worse than a blank one, which the agent is trained to read as failure.

## Out of scope

- **Inter-node wiring / a node graph.** Trigger: a user wants one node's live output in another
  node, and a file bind is genuinely insufficient.
- **Restricting scripts to CPU-only.** Recorded direction (`17_direction.md`) but a separate
  decision. Trigger: this feature has landed, so scripts no longer need GL for any legitimate
  purpose.
- **MRT (multiple outputs per pass).** 8 attachments are available (measured) but nothing needs
  them yet. Trigger: an effect needs a G-buffer-shaped pass.
- **Mouse painting / buttons.** RC's demo paints occluders; the analytic-SDF route does not need
  it (`15_fidelity.md`: JFA-from-painted matched analytic at 5.0% vs 4.5%). Trigger: the maintainer
  wants painted occluders — and note 042 deliberately CUT `u_mouse`, so that decision must be
  engaged, not reversed silently.
- **3D cascades / texture arrays.** Trigger: 2D RC is done and the maintainer wants depth.

## The three fixes owed regardless

Each is a **latent defect in today's codebase**, independent of this feature. They may land as a
separate prior commit or inside this wave, but they must not be deferred:

1. **`ctx.gc_mode = "auto"`** at context creation — `grep -rn "gc_mode" shaderbox/` returns
   nothing, so moderngl's `None` default applies and dropped GL objects never free. Measured 103
   textures / ~206 MiB after 50 script edits. Caveat (`16_stress_test.md`): `auto` leaves a bounded
   GC residual because the VAO<->program<->buffer graph is cyclic — a lag, not a leak.
2. **The `textures/` mkdir** in `ui_models.py::UINode.save` — the raw-`Texture` branch writes
   `textures/<name>.bin` but only `dir` is mkdir'd. The branch has demonstrably never executed;
   it raises `FileNotFoundError` on first contact.
3. **The missing `dtype`** in `core.py::Node.load_from_dir`'s texture reconstruction — verified
   `data size mismatch 512 != 256` for an `f2` texture. The round-trip is broken for anything but
   `f1`, and D8 makes non-`f1` targets a first-class case.

## Files touched (anticipated)

- `shaderbox/core.py` — `Canvas` gains format/filter/wrap params; `Node` gains the pass chain and
  its evaluation; `load_from_dir` texture `dtype` fix.
- `shaderbox/ui_models.py` — `UINode.save` persists pass state (D9) + the `textures/` mkdir fix.
- `shaderbox/app.py` — `gc_mode` at context creation.
- The uniform panel (`tabs/node.py`, `widgets/uniform.py`) — pass uniforms as controls (D3).
- `shaderbox/copilot/backend.py` — `_render_facts_for` pass-awareness (D10).
- Wherever pass declarations are parsed — depends on Q1 below.
- `shaderbox/help_content.py` — the contract is user-facing and generated docs must not rot.
- Tests: pass ordering + memoization (D7), format round-trip, persistence completeness.

**Cycle-from-types watch** (`dev_flow.md`): if a new pass module needs `app: App`, the
no-`TYPE_CHECKING` rule forces a structural split — anticipate it in the spec ("module X holds the
type, module Y the orchestration"), don't discover it at impl time.

## Manual verification

Each check must fail for exactly one reason, and each names its falsifier.

1. **A 2-pass node renders correctly.** Falsifier: pass B shows pass A's input, or black.
2. **A pass uniform gets a slider that works.** Drag it, the render changes. Falsifier: no control
   appears, or it appears and does nothing.
3. **A GLSL error in pass 2 lands in the error strip with the right file and line, click-to-jump
   works.** Falsifier: the error names the wrong file, the wrong line, or freezes the node.
4. **Hot-reload:** edit a pass shader, Ctrl+S, the render updates without restart. Falsifier: stale
   render or a required restart.
5. **Save/reload round-trip:** quit and reopen the project; the node renders identically.
   Falsifier: a crash, or a changed image. **This is the check the script route fails.**
6. **Copilot revert** on a turn that edited a pass restores it. Falsifier: "could not restore".
7. **Export** produces the same frames as live. Falsifier: divergence, or a leaked resource set.
8. **An `f2` accumulation pass exceeds 1.0.** Falsifier: it clamps (proving the target is 8-bit).
9. **Radiance cascades, authored natively**, matches the 063 reference render.

## Open questions for the user (MUST be answered at plan-lock)

**Q1 — Where do pass declarations live?** The biggest seam decision; `11_playground_survey.md`
prices four options.

- **(a) Inferred from the shader source** (glslViewer): declare `uniform sampler2D u_buffer0;` and
  the engine infers a pass. Zero config, `node.json` untouched, deleted-with-the-uniform so it
  cannot drift. Cost: cannot express per-pass format/size — **which D8 needs**.
- **(b) Inference + a line comment** (offline-shadertoy):
  `uniform sampler2D u_bloom;  // pass, scale: 0.25, float`. Config rides the declaration it
  configures, so the two cannot desync. Buys back exactly what (a) lacks. Cost: a bespoke
  micro-syntax. **This is the survey's recommended hybrid.**
- **(c) A declaration block in `node.json`** (ISF-shaped). Most expressive. Cost: `node.json` is
  today **app-written derived state**; hand-authored declarations change what the file IS, and the
  manifest can drift from the code.
- **(d) One file per pass, wired explicitly.** Most powerful, most ceremony, closest to the deleted
  DAG.

**Q2 — How are intermediate passes visualised?** The maintainer named this explicitly, and
`01_reference.md` shows the reference demo ships a "Stage To Render" control and a single-cascade
view because **looking at an intermediate is the primary debugging tool** for this class of effect.
Options: thumbnails in the node panel (the node grid already renders live canvas thumbnails — the
closest existing precedent); a dedicated tab (note `Ctrl+1/2/3` are taken, so a fourth needs a
chord decision); a toggle/combo on the main preview. Also: is a pass *selectable* — and if so, does
clicking one do anything, given the preview submits no interactive imgui item?

**Q3 — Does a pass get its own editor tab?** `EditorTabKind` is a Literal (`shader`/`script`/`lib`)
and tabs are node-derived. If each pass is a file (D2), is it a `shader` tab, a new kind, or does
the node panel's Entry-points zone list them?

**Q4 — Do the three owed fixes land first, as their own commit?** Recommended yes — they are
independent defects, and landing them separately keeps this feature's diff honest.
