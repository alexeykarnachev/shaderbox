# Proposal A — "Steps are uniforms" (bias: MINIMAL ADDITION)

> Produced by a design agent anchored to `00_scenario.md` R1-R10, the primary-source playground
> survey, and the real UI code. Verbatim agent deliverable, saved for the judging round.

**Design bias: MINIMAL ADDITION.** No new tab, no new panel, no new editor-tab kind, no new modal,
no new persisted document type. One new uniform *input type*, one thumbnail strip inside a panel
that already exists, and one line-comment micro-syntax on a declaration the user already writes.

## 1. The core idea

A step is declared by writing an ordinary `uniform sampler2D` in the node's shader with a trailing
`// step` comment — the same declaration the author already writes to *read* a texture now also
*creates* the thing being read, and the uniform-introspection machinery that already generates a row
for every active uniform generates a **step row** instead of a "Load image" row.

The step's own code is a function in the same file, `void step_bloom(out vec4 o)`, so a node stays
one directory and one `shader.frag.glsl` (`paths.py:8-10`) — the existing error strip, `#line` /
`SourceMap` remapping (`shader_errors.py:15-32`), click-to-jump and lib splicing all keep working,
because there is still exactly ONE compile unit per node.

In the Node tab each step row shows a live thumbnail via the same `preview_cell` the node grid
already draws (`node_grid.py:23-33`); clicking it swaps the big preview to that step.

## 2. The editor content

```glsl
uniform sampler2D u_scene;    // step
uniform sampler2D u_bright;   // step, scale: 0.5, f2
uniform sampler2D u_blur;     // step, scale: 0.25, f2, linear
uniform sampler2D u_smoke;    // step, feedback, persist

uniform float u_bloom_threshold;   // ordinary uniform

void step_scene(out vec4 o) { }
void step_bright(out vec4 o) {
    vec4 c = texture(u_scene, v_uv);
    o = max(c - u_bloom_threshold, 0.0);
}
void step_blur(out vec4 o)  { /* texture(u_bright, ...) */ }
void step_smoke(out vec4 o) { /* texture(u_smoke, ...) -> last frame */ }

void main() {              // main() IS the final step
    f_color = texture(u_scene, v_uv) + texture(u_blur, v_uv);
}
```

Why one file: the compile path is `Node.compile()` -> `resolve_usage(...)` -> one `gl.program(...)`
(`core.py:253-301`), with errors parsed from that one driver string through `SourceMap`
(`core.py:286`). N files means N compile units, which fragments the single `SourceMap` that makes
click-to-jump work — re-creating the mislocated-error blocker with better syntax highlighting.

## 3. The Node tab

```
|  Steps  (5)                                     [ show: u_blur   reset ] |
|  +------++------++------++------++------++------+                        |
|  |      ||      || #### || .... || ~~~~ || IMG  |                        |
|  |scene ||bright|| blur || blur ||smoke || out  |   <- preview_cell tiles |
|  +------++------++------++------++------++------+                        |
|   1:1     1:2 f2  1:4 f2  1:4 f2  1:1 fb  1:1                            |
|  ----------------------------------------------------------------------- |
|  [drag ] u_bloom_threshold      [ 0.850 -------------- ]   -> u_bright   |
|  [step ] u_blur      320x240 f2 [ show ]  [ pin ]  linear . clamp        |
|  [step ] u_smoke    1280x960 f2 [ show ]  [ pin ]  feedback . persist    |
```

Two elements, both reusing existing primitives:

**(a) The step strip** — a horizontal row of `preview_cell` tiles (`ui_primitives.py:923`) in a
`begin_child` with horizontal scroll; the identical call the node grid makes (`node_grid.py:23-33`).
`selected=True` gives the same selection border the grid uses. A click sets
`node_ui_state.shown_step`, which the big preview reads.

**(b) The step uniform row** — a NEW `UIUniformInputType` value `"step"` (`ui_models.py:26-28`).
This is the largest reuse in the proposal: a step is not a new object in the panel, it is a sampler
uniform whose input-type chip reads `step` instead of `texture`. `draw_ui_uniform` gains one
`elif` beside the existing `"texture"` branch (`widgets/uniform.py:233-273`). The name cell stays
`uniform_name_label`, so click-to-jump to the declaration works for free via
`find_uniform_declaration_line` (`shader_errors.py:74-84`), which already splits on `//`.

**(c) The big preview** — `ui.py:523-528` takes its texture glo from `shown_step_texture_glo`,
falling back to `canvas.texture.glo`. The `item_normalized_mouse` hit-test feeding `app.script_mouse`
is untouched.

Unchanged: the Render tab (always renders the OUTPUT, never a pinned step), the node grid
(always the node's finished output), `EditorTabKind`, `NodeTab`, `_NODE_TABS`. No `Ctrl+4`.

## 4. The micro-syntax

| Token | Meaning | Default | Req |
|---|---|---|---|
| `step` | required marker: this sampler is a step's target | — | R1 |
| `scale: N` | size = round(canvas_size * N), min 1 | `1.0` | R2 |
| `size: WxH` | absolute size | — | R2 |
| `f1`/`f2`/`f4` | component dtype | `f1` | R7 |
| `linear`/`nearest` | filter | `nearest` | R8 |
| `repeat`/`clamp` | wrap | `clamp` | R8 |
| `feedback` | may read itself (last frame) | inferred | R5 |
| `persist` | state survives the whole open session | off | R6 |

Precedent: offline-shadertoy's parser (`11_playground_survey.md:136-142`) plus glslViewer's
inference — the survey's own recommended hybrid (`11:159-176`). Defaults follow the survey's two
convergences: ratio-of-output sizing and implicit ping-pong (`11:25-32`). `clamp` is the deliberate
reversal of moderngl's `repeat_x/y = True` default, which the survey flags twice (`11:110-112`,
`11:188-191`).

## 5. Requirement coverage

| # | Requirement | How |
|---|---|---|
| R1 | several steps per node | `// step` marker + `step_*` body; one row + one tile each |
| R2 | differing resolutions | `scale:` / `size:` tokens; row caption shows resolved pixels |
| R3 | one step reading several | **nothing** — a step body calls `texture(u_bright,..)` and `texture(u_c1,..)`; reading a step IS reading a sampler |
| R4 | branching order | **nothing** — two steps reading `u_scene` IS a branch; order inferred from which step-samplers each body reads |
| R5 | self-read previous frame | a body calling `texture(u_smoke,..)` inside `step_smoke`; ping-pong is the runtime's problem |
| R6 | forever-state | `persist` token + `reset` overlay on the tile (`preview_cell`'s existing `overlay` slot) |
| R7 | values outside [0,1] | `f2`/`f4` token; shown in row caption |
| R8 | filter + wrap | `linear`/`nearest`/`repeat`/`clamp` |
| R9 | view any intermediate | step strip tiles + per-row `show` + `pin`; big preview switches |
| R10 | uniforms per step | **nothing new** — every step's uniforms are active on the ONE program, so existing rows serve them. Directly fixes the ergonomics blocker where one slider cost a decoy uniform in a second file |

## 6. Naming

A node stays a **node**. The second level is a **step** (`step_` prefix, `// step` marker, section
header `Steps (5)`). The finished picture is the **output**.

Rejected: **pass** (industry word, but collides with export passes and reads as machinery);
**buffer** (names the storage, not the work); **layer** (implies compositing order, wrong for a
branch); **node** (taken — reusing it for a sub-unit is the worst possible collision). "step" is
already this repo's own noun: `/shader-lab` builds effects as "versioned steps", and
`00_scenario.md` R1 itself reads "several **steps** inside one node".

## 7. R6 — forever-state, answered

`persist` opt-in token, three explicit rules:

- **Save/reload -> starts cold, nothing written to disk.** Step buffers are skipped by `UINode.save`
  exactly as an unbound sampler is today. Three reasons from code: the raw-`Texture` branch had never
  executed; the ergonomics doc measured ~2 MB of transient cascade data per save for state
  regenerated in seconds; and persisting would stop `node.json` being small app-written derived
  state. Tooltip states the contract: *"survives shader edits; does not survive closing the project."*
- **Copilot revert -> buffer cleared, visibly.** A revert restoring old source while keeping smoke
  state accumulated under the NEW source would be a state/code mismatch — the silent-corruption class
  in the negative spec. The revert summary gains *"Simulation state was reset."*
- **Export -> starts cold and warms up.** `render_media` already brackets exports in
  `export_isolation()` (`core.py:545-552`); step buffers join that bracket. The Render tab gains ONE
  control, `Warm-up frames [60]`, shown only when the node has a `persist` step. Capturing live state
  was rejected: the live buffer is at canvas size while an export may be at a different preset size,
  so "as-is" is undefined the moment they differ, and it makes exports non-reproducible.

Full tooltip: *"Survives shader edits. Cleared by project reload, copilot revert, and canvas resize.
Export starts cold — set Warm-up frames."* Canvas resize is in that list because `Canvas.set_size`
releases and reallocates and the resolution combo is one click away.

## 8. What this makes HARD (agent's own honest list)

1. **No spatial view of the branch.** R4 is satisfied by *existence*, not *visualisation*. The strip
   is linear; "what does step_c4 read?" means opening the shader. For an 8-level cascade with merges
   this is a real cost. Deliberate price of "no new surface".
2. **Reordering means editing text.** No drag-to-reorder.
3. **A typo in the comment is silent.** `// stp, scale: 0.5` yields an ordinary texture uniform bound
   to the default image — a picture appears, just the wrong one. The bare misspelled-marker case
   cannot be distinguished from an ordinary comment without guessing.
4. **HDR steps preview clipped.** An `f2` buffer holding 7.0 renders white; `image_with_bg` does a
   straight sRGB blit. Labelled (`f2` in caption) but not solved.
5. **The strip competes for vertical space** — ~115px of a 600px control-panel budget, squeezing the
   uniform list, which is the product. A collapsing header probably has to ship with it.
6. **Two steps cannot share one body.** ISF's `PASSINDEX` lets one body serve N passes; here eight
   cascade levels are eight `step_c*` functions (helper + eight two-line wrappers).
7. **One file gets long** — a 9-step cascade in one several-hundred-line file.

## 9. Self-verdict

R1 COVERED, R2 COVERED, R3 COVERED, R4 **COVERED (expressible, not visualised)**, R5 COVERED,
R6 COVERED, R7 COVERED (storage; preview *display* of >1.0 clipped), R8 COVERED, R9 COVERED,
R10 COVERED.

Least confident, in the agent's own order:

1. **The owner caption (`-> u_bright`)** on a uniform row is a *textual* claim sitting next to a
   control whose existence is a *driver* fact. Shadowing, a name inside a spliced `SB_*` lib body, or
   a macro makes it wrong. Decoration, not mechanism — R10 holds without it — but a wrong label is
   worse than none. Gate it on exact-token match with no lib splicing in scope, or drop it.
2. **Whether the strip earns 115px of a 600px panel.** Not measured; the squeeze lands on the
   sliders, which the ergonomics doc names as the product.
3. **Whether `// step` reads as ceremony-free or as a hidden mode.** The survey's evidence is about
   *drift*, not *discoverability*. There is NO affordance anywhere telling a user steps exist — no
   "Add step" button, no template. Genuine cost of the bias; the agent is not certain it is right.
4. **Execution-order inference** was asserted, not designed. A cycle between two non-feedback steps
   has no UI answer here beyond "surface it as an error".

## 10. False trails (considered, rejected, with reasons)

- **A fourth node-settings tab `Steps` (Ctrl+4).** Cheapest mechanical extension, but puts the step
  thumbnails on a DIFFERENT screen from the uniform sliders — separating the picture you debug from
  the slider you drag is the one split this feature most needs to avoid.
- **A new `EditorTabKind`, one editor tab per step.** N files means N compile units, fragmenting the
  single `SourceMap` that makes click-to-jump work.
- **Declarations in `node.json`** (ISF-shaped, most expressive). `node.json` is app-written derived
  state and `UINode.save` rewrites it wholesale every save; hand-authored declarations would make it
  a file the user edits and the app must not clobber. Biggest seam in the feature; minimal-addition
  answer is to not spend it.
- **A JSON / `#pragma` header block at the top of the shader.** Keeps one file but reintroduces
  manifest drift within it (add a `step_*`, remember the block entry) — the survey's documented
  KodeLife failure. The comment-rider cannot drift because it is deleted with the uniform.
- **A node-graph patch view.** Would answer R4 beautifully. Means a graph editor + an inter-node
  document model; both surveyed graph tools REFUSE to put multipass in the shader, the opposite of
  what a file-based tool needs. A DAG was already built here and deleted.
- **Pure glslViewer inference (`u_buffer0`, zero config).** Exposes no per-pass resolution, format,
  filter or wrap whatsoever — fails R2, R7, R8 outright. The survey says so itself.
- **Positional channel slots (`iChannel0..3` / `register(t0)`).** ShaderBox binds by NAME everywhere;
  a slot table would be a positional layer over a name-addressed engine. SHADERed's documented
  failure: the same slot is `posTex` in one example and `clr` in another.
- **A float/HDR checkbox on the step row.** Splits configuration between the declaration and panel
  state persisted to `node.json` — the desync the comment-rider exists to avoid.
- **A node per step, wired by file bind.** Cannot do per-frame feedback at all (R5, R6), and
  cross-node effects are explicitly out of scope in `00_scenario.md`.
- **Clicking the big preview to select a step.** `image_with_bg` submits no interactive item and
  `item_normalized_mouse` already hit-tests that rect for `script_mouse`; a click would contend with
  the 042 mouse feed. Strip tiles are already free click targets.
