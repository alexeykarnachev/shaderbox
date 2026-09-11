# Resolution flow: where a document's size lives, everywhere it goes

Research for the maintainer's proposal (090's aftermath): a document should carry only an ASPECT
RATIO, the live render target sizes to what is DISPLAYED (viewer panel / grid tile / pass-strip
thumbnail), and RESOLUTION becomes an export-only (render-preset) concept. This file maps every
place a size value is stored, derived, or consumed today, before that plan is designed.

## 1. The data model

**The one stored field: `Document.canvas_size: tuple[int, int]`** (`shaderbox/document.py:275-278`).
Set in `__init__` (normalized through `_as_canvas_size`, `document.py:70-84`) and by the single
writer `Document.set_canvas_size` (`document.py:355-364`), which resizes the OUTPUT pass's canvas
immediately and leaves every other pass to catch its new size on its own next `render()` call
(`document.py:719-723`, the `if name != output: wanted = entry.target.target_size(...)` branch).
No aspect-only field exists anywhere in the model; width and height ARE the resolution and the
aspect is `w/h`, computed fresh wherever needed, never cached.

- **On disk**: `document.json`'s top-level `"canvas_size": [w, h]` (`ui_models.py:381`, written by
  `UIDocument.save`; read by `Document.load_from_dir` via `_load_document_metadata` +
  `_as_canvas_size`, `document.py:275-278, 70-84`). No separate aspect key.
- **Default**: `DEFAULT_CANVAS_SIZE` (`shaderbox/constants.py`, used at `document.py:71` and
  `core.py:28,147` — not read in this pass but referenced by both `Document.__init__` and
  `Canvas._init`).
- **Bounds**: `MIN_CANVAS_PX = 16`, `MAX_CANVAS_PX = 4096`, enforced by `clamp_canvas_size`
  (`shaderbox/pass_graph.py:58-67`). Both entry points funnel through it: the Document tab's W/H
  fields (`tabs/document.py:63-70,190-192`) and the copilot's `set_canvas_size`
  (`copilot/capabilities.py:430-…`, clamps at `copilot/backend.py:1233`).
- **The UI that edits it**: `shaderbox/tabs/document.py::draw` (lines 118-196) — two `imgui.input_int`
  fields (`##canvas_w`/`##canvas_h`, `document.py:155-179`) plus a presets combo
  (`_draw_canvas_presets`, `document.py:29-40`, sourced from `_canvas_presets`,
  `document.py:73-115`: square presets, the `RenderShape` table's sizes at the CURRENT size, and
  any bound texture's native size). Committing either field calls `_apply_canvas_size` →
  `Document.set_canvas_size` (`document.py:63-70`). There is no separate "aspect" control anywhere
  in the UI — W and H are typed as a pair, and the aspect is whatever ratio results.

### Every derived size

| Derived value | file:function | formula |
|---|---|---|
| Per-pass (non-output) canvas size | `document.py:719-723` (`Document.render`) | `entry.target.target_size(canvas_size)` = `TargetConfig.target_size` (`pass_graph.py:90-94`): `(max(1, round(w*scale)), max(1, round(h*scale)))`, `scale` in `(0,1]` |
| `u_resolution` (per pass, per frame) | `core.py:508-509` (`Pass.render`) | `value = canvas.texture.size` — literally the CANVAS the pass is drawing into this call (its own, or an export's override) |
| `u_aspect` (per pass, per frame) | `core.py:504-505` | `np.divide(*canvas.texture.size)` = `width/height` of THAT SAME canvas |
| Feedback history canvas size | `document.py:518-534` (`_feedback_canvas`), `document.py:406-455` (`_seed_feedback`) | born at the live pass canvas's current `texture.size` (i.e. already-scaled, post-`target_size`) |
| Export target size (`RENDER_AT_TARGET`) | `document.py:882-895` (`render_media`) | `resolve_dims(preset, render_pass.canvas.texture.size)` (`render_preset.py:37-71`); a FRESH `Canvas(size=(target_w,target_h))` is allocated and the WHOLE document renders into it |
| Export target size (`SCALE_DISTORT`, the default `RenderPreset()`) | `document.py:878-881` | none — renders straight into `render_pass.canvas` at the LIVE size, no resize |
| Video export alignment | `document.py:592-596` (`_render_video`) | rounds up to `VIDEO_RESOLUTION_ALIGNMENT` (also `render_preset.py:33-34`, `_align`) |
| Image export post-resize (rare path) | `document.py:571-576` (`_render_image`) | PIL `.resize()` ONLY when `canvas.texture.size != (details.width, details.height)` — a belt-and-suspenders case; `render_media`'s canvas is already minted at the target size when `RENDER_AT_TARGET`, so this fires mainly for `MediaDetails` set by a caller that disagrees with the preset |
| `RenderShape` → concrete dims | `render_shape.py:60-89` (`shape_to_preset`) → `render_preset.py:37-71` (`resolve_dims`) | `FIXED_ASPECT`: `w = longest_edge`, `h = round(w*ah/aw)` (or the reverse for portrait); `NATIVE` → `ResolutionPolicy.FREE`, i.e. the document's own size unchanged |
| Copilot's "look" frame (`_probe_frame`) | `copilot/backend.py:148-163` | renders the FULL document at native `canvas_size` into `render_pass.canvas` (unchanged), THEN `PILImage.resize((width,height), Resampling.BOX)` down to `render_facts_size` (default **64px**, `copilot/config.py:89`) — box-filter downscale of an already-rendered full frame, never a native small render |
| Viewer display size (imgui only) | `ui.py:670-676` (`_draw_document_image`) | `image_aspect = w/h` of the render_pass texture; `image_width/height = min(avail, avail*aspect)` — a LAYOUT computation, never touches the texture |
| Pass-strip / document-grid tile display size | `widgets/pass_list.py:117-132`, `widgets/document_grid.py:22-32` | fixed `cell_w` (`SIZE.PASS_TILE` / `SIZE.THUMB_LG`); `preview_cell` / `centered_image` scale the SAME live texture down via imgui's `image_size` param — no GPU work |

## 2. Consumer table

| Site | Uses the size for | What changes if live size = displayed size, runtime-variable | Risk |
|---|---|---|---|
| `core.py:508-509` `Pass.render` (`u_resolution`) | Sets the shader-visible resolution uniform | Every pixel-space shader (JFA, cascade, any px-radius kernel) recomputes correctly IF the pass canvas is reallocated to the new size before this line runs; the shader itself needs no change | none, if allocation precedes the uniform write in the same frame |
| `core.py:504-505` `Pass.render` (`u_aspect`) | Aspect-corrected shader coordinate systems (`SB_center_uv`) | Unaffected in principle — aspect tracks whatever the canvas is, even if the canvas now tracks a resized viewer panel | none |
| `core.py:145-164` `Canvas._init`/`set_size` | Allocates the GL texture+FBO at a given size | Becomes the reallocation point for EVERY resize event (panel drag, window resize, tab switch to a differently-sized surface) instead of only an explicit user edit | **reallocation per resize event** — today `set_size` fires only on an explicit `set_canvas_size` call (a few times per session); tying it to imgui panel geometry makes it fire every frame the panel is actively being dragged, unless debounced |
| `document.py:719-723` `Document.render` (non-output pass sizing) | Scales each non-output pass from `canvas_size` | Unaffected structurally — `scale` still multiplies whatever the base is; the base itself becomes viewport-derived | none, mechanically — but see feedback risk below |
| `document.py:355-364` `Document.set_canvas_size` | The single funnel: resizes output canvas immediately | Becomes the target of a per-frame (or per-resize-event) call from the UI layer instead of a user-typed commit | **state lost** — every `set_size` call that changes shape releases+reallocates the texture (`core.py:158-164`), which orphans feedback history (see §3) and the persisted `.bin`. A panel that resizes continuously (a drag) would thrash this. |
| `document.py:406-455` `_seed_feedback` (load-time) | Expects an EXACT size match between the persisted `.bin` and `entry.target.target_size(canvas_size)` computed from the GRAPH, not the live canvas | If `canvas_size` no longer has one persisted value but is instead "whatever the viewer was last sized to," a reload has no stable `canvas_size` to recompute `expected_size` from — the field this function reads must still exist and be stable across saves | **state lost**: a strict mismatch on ANY field (size, dtype, components) silently drops the history (`document.py:437-447`, logged warning, "it starts black") |
| `document.py:518-534` `_feedback_canvas` (runtime) | Reallocates a feedback canvas whenever the live canvas's size disagrees (`canvas.texture.size != live.texture.size`, line 532-533) | Already has a live-resize path — `canvas.set_size(live.texture.size)` (line 533). This is the ONE place today that already tolerates a canvas resize without dropping the whole history; but it does not preserve CONTENT, it discards it (`Canvas.set_size` releases and reallocates, `core.py:158-164`) | **content lost on every resize** — the accumulated feedback trail (used for real effects: `bloom_chain` fixture, cascade probes, jump-flood) goes black on any live resize under the proposal, unless a resample step is added |
| `core.py:255` `Pass.set_target` (`target_generation` bump) | Marks a feedback history stale when the TARGET FORMAT changes (dtype/filter/wrap), not size | A pure size change today does NOT bump `target_generation` — only `_feedback_canvas`'s direct size compare (line 532) catches it. A resize-driven live size adds a second, more frequent trigger for the same "history predates the current shape" problem the generation counter exists for | risk of the two mechanisms (generation bump vs. direct size compare) diverging once resize is frequent — worth unifying |
| `ui.py:670-676` `_draw_document_image` | Computes viewer LAYOUT (imgui image size) from the texture's current aspect | Under the proposal this becomes the size the texture must actually BE (the render target sizes to the panel), inverting today's "texture is fixed, layout adapts" into "layout drives texture size" | **mismatch live vs export** — the export path (`RENDER_AT_TARGET`) already renders at an INDEPENDENT target size (a scratch `Canvas`) unrelated to whatever the viewer happened to show, so this consumer's risk is really about the LIVE canvas's identity, not export |
| `widgets/pass_list.py:98-132` `_draw_pass_tile` / `preview_cell` | Draws the pass's OWN live full-resolution texture, scaled down by imgui | Comment at `pass_list.py:98-100` is explicit: "not a second render at thumbnail size... rendering a separate small frame would double the document's per-frame draw count." If per-pass canvases start sizing to their DISPLAYED tile (a small fixed `SIZE.PASS_TILE`), every non-output pass's `u_resolution` becomes the TILE size, not a fraction of `canvas_size` — a structural break of "the strip mirrors the graph's real sizes," not just a display change | **GPU cost changes shader OUTPUT, not just cost** — any pixel-space shader in a non-output pass (a JFA/cascade helper pass) would compute at tile resolution and feed the output pass wrong data, unless the strip keeps reading a SEPARATE full-res texture from what it displays |
| `widgets/document_grid.py:14-32` `draw_document_preview_button` | Same pattern for the document-switcher grid: reads `render_pass.canvas.texture` directly, scaled by imgui | Same risk as above, but for the OUTPUT pass specifically — if the output canvas itself becomes sized to whatever's smallest visible use (a grid thumbnail) whenever the viewer isn't open, the document's "native" render quality becomes ambiguous across UI states (viewer open vs. grid-only) | **quality varies by UI state** — a document shown only in the grid would render at grid-thumbnail resolution, silently degrading anything that reads `render_pass.canvas` elsewhere (export's `SCALE_DISTORT` default reads this SAME texture, `document.py:878-881`) |
| `tabs/render.py:34-35` Render tab preview | `centered_image(tex.glo, tex.size, ...)` — same live texture, scaled by imgui | No allocation change; purely a display box | none |
| `document.py:878-881` `render_media` (`SCALE_DISTORT`, the DEFAULT `RenderPreset()`) | Renders directly into `render_pass.canvas` — i.e. EXPORTS AT WHATEVER THE LIVE CANVAS CURRENTLY IS | If live canvas size becomes viewport-derived, the default export path (used whenever no preset/shape is picked, e.g. the bare Render tab) exports at "whatever the viewer happened to be sized to at render time" — an unstable, UI-driven export resolution unless callers are forced onto `RENDER_AT_TARGET` | **export resolution becomes accidental** — this is the sharpest conflict the proposal has to resolve: today `SCALE_DISTORT`'s use of the live canvas is deliberate ("export the document as you see it"); under the proposal "as you see it" stops meaning a stable number |
| `document.py:882-895` `render_media` (`RENDER_AT_TARGET`, every `RenderShape` other than `NATIVE`) | Renders into a FRESH scratch `Canvas` at `resolve_dims(preset, ...)` — already fully decoupled from the live canvas | Unaffected — this path is already "size is an export-preset property," which is exactly the target shape the maintainer wants generalized | none — this is the existing precedent to build on |
| `copilot/backend.py:148-163` `_probe_frame` | Renders the FULL document at native size, THEN box-filters down to a small fixed probe size (default 64px) | Already fully decoupled: the model's "look" is always a small fixed size regardless of live canvas size, and — per `test_probe_native_size.py` — the render happens at NATIVE resolution first so pixel-space shaders (`u_resolution`) see the true canvas, not the probe size | none — second existing precedent for "render full, downscale for display" |
| `copilot/backend.py:1647-1704` `render_image`/`render_video` tools | `render_job.render_to` → `Document.render_media` with a `shape_to_preset`-derived preset | Same path as the Share tab's exports — already decoupled via `RenderShape`/`RenderPreset` | none |
| `ui_models.py:507-528` `UIDocument.save` (feedback persist) | Writes `canvas.texture.read()` raw bytes + `canvas.texture.size` to `feedback/<pass>.bin` | The size written is whatever the live canvas is AT SAVE TIME — if that varies with viewport, two saves of the same document (different viewer sizes) persist DIFFERENT feedback shapes, and a subsequent load's `_seed_feedback` strict-match (`document.py:437-447`) becomes far more likely to reject the seed | **state lost more often** — today this rarely misfires (canvas_size changes only on explicit edit); under the proposal it would misfire on ordinary use (open app with viewer panel a different width than last session) |
| `channel_blit.py:64-72` `ChannelBlit.render` | `self.canvas.set_size(source.size)` — mirrors whatever texture it's fed, every call | Already resize-tolerant by construction (a NEW canvas at the SOURCE's current size every call, no persisted state to lose) | none |
| `media.py` (`Image`/`Video` uniform-bound textures) | Intrinsic size of a bound asset, independent of canvas | Untouched — media textures are never resized to the canvas; they are sampled at their own resolution regardless of `canvas_size` | none |
| `tabs/document.py:73-115` `_canvas_presets` | Reads `canvas_size` to compute preset target sizes (relative sizing, e.g. `RenderShape` at "current" longest edge) | If there is no longer a stable `canvas_size` to compute FROM, this preset list's baseline becomes whatever the viewport happened to be — presets stop being reproducible across sessions | **UI becomes non-deterministic** unless an aspect-only field replaces `canvas_size` as this function's input |
| `pass_graph.py:58-67` `clamp_canvas_size`, `MIN/MAX_CANVAS_PX` | Bounds any caller's requested canvas size | Would need to become "bounds on the render TARGET a viewport can request," not a document field the user types — same clamp, different call site (every viewport resize instead of a field commit) | none if the funnel is preserved, but the funnel moves |

## 3. Feedback in detail

**Sizing.** A feedback canvas is born at whatever the live pass canvas's CURRENT size is
(`_feedback_canvas`, `document.py:518-534`, `live = self.passes[name].canvas`), i.e. already
post-`scale` for a non-output pass. It is NOT derived from `canvas_size` directly at allocation
time; only the LOAD-time seed (`_seed_feedback`) recomputes the expected size from the graph
(`document.py:427-431`: `self.canvas_size if name == output else entry.target.target_size(self.canvas_size)`),
specifically because at load every pass canvas sits at the document's FULL size and `scale` is
only applied inside the first `render()` call (`document.py:426-427` comment explains this exact
trap).

**Today's resize path.** `_feedback_canvas` (`document.py:531-533`) DOES tolerate a live size
change: `elif canvas.texture.size != live.texture.size: canvas.set_size(live.texture.size)`. But
`Canvas.set_size` (`core.py:158-164`) is release-then-reallocate — it does NOT resample or
preserve the old contents. So today's "resize path" is really a silent content-drop path: any
live resize (which currently only happens via `Document.set_canvas_size`, an explicit user edit)
throws away the accumulated feedback trail and the pass starts black again that frame. There is
no resample step anywhere in the codebase for feedback textures.

**Does the persisted `.bin` carry its size?** Yes — `{"file_path", "size", "components", "dtype"}`
(`ui_models.py:520-524`). The size is `list(canvas.texture.size)`, i.e. the exact pixel dims at
save time.

**What `_seed_feedback` does on a size mismatch.** Strict rejection, no resample, no partial
recovery (`document.py:437-450`): if the recomputed `expected_size` (from the GRAPH, not the live
canvas) disagrees with the persisted `size`, OR `dtype`/`components` disagree, the history is
dropped entirely with a warning ("it starts black") and the loop moves to the next pass. This is
covered exactly by `tests/test_feedback_persistence.py::test_a_feedback_entry_that_does_not_match_is_ignored`
(`tests/test_feedback_persistence.py:270-291`) — the test literally corrupts the persisted `size`
field and asserts the history is silently dropped, the document still loads, and the pass still
renders (starting black).

**What an export at a DIFFERENT size than live state does today — three regimes already exist,
which is directly relevant to the proposal:**

1. **Cold start (implicit today).** `render_media` calls `self.reset_feedback()` UNCONDITIONALLY
   before every export (`document.py:875-877`, D10 in the inline comment: "a feedback target is
   the same class of state as a stateful script, so it is reset HERE"). So an export NEVER
   inherits the live feedback trail regardless of size — every export already starts every
   feedback pass at black and re-accumulates for the export's own duration. This sidesteps the
   whole size-mismatch question for exports specifically: there is nothing to resample because
   nothing carries over.
2. **Render at target size directly** (`RENDER_AT_TARGET`, `document.py:882-895`): a scratch
   `Canvas` at `resolve_dims(...)` is minted, and the WHOLE document (`_render_media_into` →
   `_render_image`/`_render_video` → `self.render(u_time=t, canvas=target)`) draws with that
   canvas as the OUTPUT pass's target for that frame only. Non-output (including feedback) passes
   still size from `entry.target.target_size(self.canvas_size)` — i.e. relative to the LIVE
   `canvas_size`, not the export target — so a feedback pass in an exported document renders at a
   size tied to the live canvas even while the OUTPUT frame is captured at a different size. This
   is confirmed by `test_an_off_size_export_renders_the_shader_not_a_blank`
   (`tests/test_render_for.py:104-121`): the scratch canvas is handed down correctly for the
   OUTPUT pass, but the test fixture has no feedback pass, so the interaction between a scaled
   feedback pass and an off-size export target is untested territory.
3. **Render at live size, no resize** (`SCALE_DISTORT`, the default): identical machinery, target
   `canvas = self.render_pass.canvas` (the live one) — no allocation happens at all.

None of the three regimes ever resamples a feedback texture to a new size; the only two operations
that exist on a feedback canvas are "start black" (reset) and "reallocate and start black"
(`Canvas.set_size`, the resize path). **Upscale-after-render, a fourth option the maintainer's
prompt names, does not exist anywhere in the codebase today** — no code path renders at one size
and then resizes the OUTPUT image after the fact except the rare PIL `.resize()` belt-and-suspenders
branch in `_render_image` (`document.py:571-576`), which only fires when the caller's requested
`MediaDetails` size disagrees with the canvas actually rendered — a defensive fallback, not a
designed resample step.

## 4. Shipped shaders: pixel-dependent vs. resolution-independent

Grepped `u_resolution`, `gl_FragCoord`, `textureSize`, `dFdx`/`dFdy` across `shaderbox/resources/`
and `docs/`.

- `gl_FragCoord`: **zero occurrences** anywhere in `resources/` or `docs/`.
- `textureSize`: **zero occurrences**.
- `dFdx`/`dFdy`: **zero occurrences**.
- `u_resolution`: **16 occurrences** across 4 files (excluding the Odin ABI probe test harness,
  which only checks the uniform NAME exists, not shader logic):

| File | Use | Classification |
|---|---|---|
| `document_examples/77a84d27.../passes/jfa.frag.glsl:24,27,42` | Jump-flood-algorithm offset in PIXELS: `exp2(ceil(log2(max(u_resolution.x,u_resolution.y)))-1.0-u_pass_iteration)`, then `vs_uv + vec2(x,y)*offset/u_resolution` | **pixel-dependent** — the JFA step size is defined in texel units; a resolution change changes the NUMBER OF ITERATIONS needed to converge (the graph's `iterations` field, author-set, `pass_graph.py:106-117`), not just the visual scale |
| `document_examples/77a84d27.../passes/cascade.frag.glsl:31,63,67,73,93,102-105` | Radiance-cascade probe grid: `floor(vs_uv*u_resolution)` (pixel grid index), `u_prev` sampled at `c00/u_resolution` etc. (explicit texel-to-uv conversion for bilinear taps) | **pixel-dependent** — the probe spacing (`u_pass_iteration`-driven) and the cascade's base (power-of-2 or power-of-4 chain length) are tied to the canvas's actual pixel count; a resize changes how many iterations converge the cascade, same class of dependency as JFA |
| `document_examples/0b0d16bb.../passes/main.frag.glsl:7` | A COMMENTED-OUT declaration (`// uniform vec2 u_resolution;`) in what reads as a starter/example shader | **unused** — not a live dependency |
| `resources/shaders/editor.frag.glsl:7,21` | The code editor's own glyph-grid shader: `cell_size_uv = u_glyph_size_px / u_resolution` | **pixel-dependent**, but OUT OF SCOPE for this proposal — this is the embedded code editor's own rendering (`editor/render.py`), sized to its own `EditorPanel` FBO (`editor/render.py:179-200`, `_ensure_target`), never a user document's canvas. Not a document shader. |

**The shipped SDF text/glyph library is resolution-independent**, by explicit design and
docstring: `shader_lib/text/layout.glsl:1-7` states "All distances are in the same units as uv,"
and every function (`SB_text_size`, `SB_text_fit`, `SB_text_char_center`, `SB_sd_text`) computes
purely in `char_height`/`spacing` (uv-space) parameters — no `u_resolution` reference anywhere in
`layout.glsl` or `glyphs.glsl`. `shader_lib/text/glyphs.glsl:107-109` documents glyph strokes in
"LOCAL units," explicitly uv-relative. `shader_lib/space/center_uv.glsl` (`SB_center_uv`) is
aspect-only (`u_aspect`, not `u_resolution`).

**Count: 2 pixel-dependent document-shader files (JFA + cascade, both in ONE example document,
`77a84d27...`, the bloom-chain-style multi-pass example), 0 pixel-dependent in the shipped
`shader_lib/` helper library, 1 unused/commented reference.** Everything else that reads
`u_aspect` (11 files, grepped separately) is resolution-INDEPENDENT by construction — aspect
tracks the RATIO, not the pixel count, and survives a resize with no visual change beyond the
shape the ratio itself implies.

## 5. The viewer and tiles today

**Nothing is ever re-rendered at a smaller size for display.** Every display site scales the SAME
full-resolution GPU texture via imgui's own `image_size` parameter (a GL blit/sample done by
imgui's renderer at draw time, not a second application draw call):

- **Viewer** (`ui.py:648-766`, `_draw_document_image`): `image_width`/`image_height` computed from
  `ui_document.document.render_pass.canvas.texture.size`'s aspect, fit into
  `imgui.get_content_region_avail()`; `imgui.image_with_bg(imgui.ImTextureRef(shown_texture.glo),
  image_size=(image_width, image_height), ...)` (`ui.py:695-701`) — one texture, drawn once, at
  whatever box size the panel currently has.
- **Grid tile** (`widgets/document_grid.py:14-32`): `preview_cell(..., texture_glo=render_pass.canvas.texture.glo,
  texture_size=render_pass.canvas.texture.size, cell_w=size, ...)` — `preview_cell`
  (`ui_primitives.py:1182-…`) draws the same live texture into a fixed-size cell.
- **Pass-strip thumbnail** (`widgets/pass_list.py:90-132`, `_draw_pass_tile`): explicit comment,
  lines 98-100: "The pass's OWN live target, scaled down by imgui — not a second render at
  thumbnail size. Every pass already draws once per frame into that texture, so the tile costs
  nothing but the blit; rendering a separate small frame would double the document's per-frame
  draw count." This is a DELIBERATE, documented design choice today, not an oversight.
- **Copilot's frame** (`copilot/backend.py:148-163`, `_probe_frame`): the one exception — this
  DOES render at native size then produce a genuinely separate, smaller buffer (PIL resize), but
  the render itself is still the SAME full-resolution GPU draw; only the CPU-side readback is
  downscaled before being sent to the LLM. No GPU-side reduced-resolution render exists anywhere
  in the codebase today.

## 6. Existing settled decisions touching resolution/aspect/export size

Grepped `ai_docs/conventions.md ## Design decisions` for `resolution|aspect|canvas|export`.

- **"A document is N passes; a Pass owns everything about ONE shader" (feature 065)**
  (`conventions.md:226-236`): *"The canvas travels DOWN with the pass (each needs its own
  target); the document owns the canvas SIZE and applies each pass's `scale`, so a pass never
  sizes itself from a number it does not hold."* — **The proposal EXTENDS this**: it keeps the
  document as the size authority but changes what the document's size number MEANS (a
  viewport-derived live size vs. a stored field) and where the AUTHORITATIVE size for a pixel-space
  shader's semantics comes from (needs restating: a pass's `scale` today is relative to a stored,
  stable `canvas_size` — if that becomes ephemeral, `scale`'s meaning drifts with the viewer's
  panel geometry).
- **"Render output size is ONE named vocabulary (`RenderShape`), not raw dims per caller"**
  (`conventions.md:735-748`): *"The shape owns ONLY size+aspect; fps/container/duration_max stay
  per-OUTLET caller args... DELIBERATELY out of the vocabulary: the Render-tab `ResolutionDetails`
  (a free-form WxH artist control that is ALSO the actual-rendered-dims record, persisted in
  document.json — a 7-value enum can't carry concrete dims, and it's a different concept with its
  own home)."* — **The proposal HONORS and generalizes this**: `RenderShape`/`RenderPreset` are
  already the "resolution is an export-preset property" mechanism the proposal wants to make the
  ONLY mechanism. The friction is `ResolutionDetails` (`media.py:41-44`, free-form WxH), which this
  same decision explicitly calls "a different concept with its own home" — under the proposal that
  home changes meaning (it stops mirroring a document field and becomes purely export-authored).
- **`ENGINE_DRIVEN_UNIFORMS` / uniform-defaulting home** (`conventions.md:884-891`): *"`Pass.render()`
  owns per-type uniform defaulting... `ENGINE_DRIVEN_UNIFORMS` (in `core.py`) is the one home for
  the `u_time/u_aspect/u_resolution` skip set — never re-list the three names."* — **Directly
  relevant, unaffected by the proposal's mechanics**: `u_resolution` stays a real, distinct,
  engine-written uniform (`core.py:508-509`) regardless of what determines the CANVAS size that
  feeds it; the proposal does not remove `u_resolution`, it changes what number ends up in it.
- **`evaluation is memoized` / feedback swaps at the FRAME boundary** (`conventions.md:230-236`):
  no direct resolution content, but the invariant ("a shared ancestor draws once… `begin_frame`
  takes the frame NUMBER and is idempotent") is the mechanism any resize-triggered re-render must
  not violate — a resize is not a new frame, so triggering `render()` off a panel-size change needs
  its own hook, not a `begin_frame` call.
- **No decision anywhere in `conventions.md` currently states "the document's canvas texture is
  sized to the viewer panel"** — today's rule is the opposite (`document.py:355` docstring: "The
  single funnel, because `canvas_size` is what every other pass scales FROM," driven by explicit
  user input, never by imgui layout). The proposal would be a NEW decision, not an extension of an
  existing one, for the LIVE-size half; the EXPORT-size half (decoupling render preset from live
  canvas) is already fully precedented by `RenderShape`/`RENDER_AT_TARGET`/the copilot probe.

## 7. False trails and coverage

**False trails ruled out during this research:**

- Expected `gl_FragCoord`/`textureSize`/`dFdx`/`dFdy` usage in the shipped shader library for
  anti-aliasing or screen-space effects — none exists; the SDF glyph system is fully uv-space and
  the only pixel-space shipped example shaders are the two multi-pass algorithms (JFA, cascade) in
  ONE example document, not in `shader_lib/` itself.
  - **Confirmed absent** in `shaderbox/resources/` and `docs/`.
- Expected the Render tab's `ResolutionDetails` (`media.py:41-44`) to be a second, independent
  size-storage path competing with `canvas_size` — it is not: it is a transient field on
  `MediaDetails`, populated per-render from either the live canvas size (`SCALE_DISTORT`) or
  `resolve_dims` (`RENDER_AT_TARGET`), never itself an input to rendering.
- Expected the copilot's `render_image`/`render_video` tools to bypass `RenderShape` with raw
  width/height — they do not; `copilot/tools/document_ops.py` only exposes raw width/height for
  `set_canvas_size` (the DOCUMENT's canvas, not a render), and `render_image`/`render_video`
  (`copilot/backend.py:1647-1704`) take `shape: RenderShape` exclusively.
- Considered whether `channel_blit.py`'s `ALPHA`/`RGB` views might hold their own persisted size —
  they do not; `ChannelBlit.render` resizes to match its source every call (`channel_blit.py:66`),
  no independent state.
- Considered whether the profiler (`profiling.py`, feature 088) keys any state by resolution — not
  reached in this research; not grepped. Flagged as unexamined below.

**Coverage — read end to end:** `document.py`, `core.py`, `render_shape.py`, `render_preset.py`,
`pass_graph.py`, `media.py`, `channel_blit.py`, `engine_uniforms.py`, `widgets/document_grid.py`,
`widgets/pass_list.py`, `tabs/document.py`, `tabs/render.py`, `tabs/share.py`, `render_job.py`,
`render_defer.py`, `shader_lib/text/layout.glsl`, `shader_lib/text/glyphs.glsl` (grepped for
resolution terms, not read in full), `shader_lib/space/center_uv.glsl`,
`resources/shaders/default.frag.glsl`, `resources/shaders/editor.frag.glsl`,
`tests/test_feedback_persistence.py`, `tests/test_probe_native_size.py`.

**Read in relevant part (grepped first, then read the matching region):** `ui_models.py` (the
`canvas_size`/feedback-save section, lines ~275-540), `ui.py` (`_draw_document_image` and callers,
lines 600-830), `ui_primitives.py` (`preview_cell`/`centered_image`/`preview_box`, lines 200-260 +
1180-1260), `copilot/backend.py` (grepped broadly for `resolution|canvas_size|render_pass|width|
height`, then read `_probe_frame`/`_probe_target_for` lines 95-163, `render_image`/`render_video`
lines 1647-1704, `_render_facts_for` lines 2140-2160), `copilot/capabilities.py` and
`copilot/tools/document_ops.py` (grepped, `set_canvas_size` signature read), `editor/render.py`
(grepped + the `EditorPanel._ensure_target` region, lines 150-200), `ai_docs/conventions.md`
(grepped `## Design decisions` for resolution/aspect/canvas/export, four matching sections read in
full), `090_render_decoupling/research/decisions_audit.md` and `shared_state.md` (grepped for
resolution/canvas/aspect, matching rows read).

**Grepped only, not read in full (breadth check, no matches or matches judged out of scope):**
`docs/` (only `branding/logo.frag.glsl` matched "aspect," judged irrelevant — a static branding
asset, not a document shader), all `tests/` files for `resolution|width|height|u_resolution|aspect`
(full file list enumerated; `test_canvas_fields.py`, `test_canvas_presets.py`, `test_render_for.py`,
`test_render_preset.py`, `test_render_shape.py` spot-read for the sharpest assertions;
`test_document_graph.py`, `test_uniform_seed_save.py`, `test_graph_persistence.py` and the
remaining ~25 matched files were NOT individually opened).

**Not examined at all (out of scope or not reached):** `profiling.py`'s GPU-query keying (088) —
flagged by 090's own research as needing its own per-thread instance regardless of this proposal,
not re-derived here; the `youtube.py`/`telegram.py` exporter UI internals beyond the grep (no
`resolution`/`width`/`height` matches surfaced beyond the `ResolutionPolicy.LONGEST_EDGE` reference
already covered via `render_shape.py`); `intel/` (editor intelligence) — no resolution-relevant
surface expected or found in the module map.
