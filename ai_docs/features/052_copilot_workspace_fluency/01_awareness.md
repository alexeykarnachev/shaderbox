# Slice 1 — Awareness (read-only enrichment; no new tools)

The cheapest, highest-ROI slice and zero guard risk: pure corollary-2 work. It puts three facts the
model is currently blind to onto the channel it already reads (the working set), so no new tool, no
gate, no checkpoint impact.

## Goal

Make textures, canvas size, and control ranges VISIBLE in the working set (and, where cheap, the
project map), so the model can reason about them and — once slice 2/3 land — act on them.

## Design decisions (lock-in)

1. **Sampler bindings render in the working-set uniform rows.** `_format_uniforms` gains a sampler
   branch: for a `sampler2D` uniform show what is bound, from the live value + its `MediaDetails`:
   ```
   u_tex   sampler2D  <- (512x512, image)
   u_noise sampler2D  <- (no media bound)
   ```
   Today a sampler is a "= value" row whose value is a `MediaWithTexture` with **no `__repr__`**, so it
   renders as `u_tex sampler2D = <shaderbox.media.Image object at 0x7f…>` — a per-run memory address
   (also a silent cache-buster). This replaces that with `<- (<w>x<h>, image|video)` or
   `(no media bound)`.
   - **"(no media bound)" detection — must survive persist+reload (round-2 correction).**
     `core.py:333` returns `Image(_DEFAULT_IMAGE_FILE_PATH)` for every unbound sampler (no null state).
     A NAIVE path-comparison at read time BREAKS for disk-loaded nodes: `ui_models.save`
     (`ui_models.py:289`) persists EVERY active sampler — including a still-default one — to
     `media/<uniform>.<ext>`, and `core.py:181` reloads it from that path, so a reloaded default
     sampler's `details.file_details.path` is `<node>/media/u_tex.png`, NEVER `_DEFAULT_IMAGE_FILE_PATH`
     → it would read as bound. The robust fix is at the SAVE boundary: **`ui_models.save` SKIPS a
     sampler still holding the default `Image` (path == `_DEFAULT_IMAGE_FILE_PATH`, reliably true
     in-memory at save time), writing no `media/` file for it.** Then load leaves it out of
     `metadata["uniforms"]` and `seed_uniform_values` (`core.py:316-326`) re-seeds it to the default —
     path == `_DEFAULT_IMAGE_FILE_PATH` again. So the awareness check is: value is a `MediaWithTexture`
     whose source path == `_DEFAULT_IMAGE_FILE_PATH` ⇒ `(no media bound)`, else dims+kind. This
     round-trips for both default and user-bound samplers (verified against `core.py` load+seed).
     - **Blast radius (flag for the slicer):** the save-skip changes `ui_models.save` for ALL nodes,
       not just the copilot — a default sampler stops writing a per-node copy of the shipped default
       (a strict improvement: load re-seeds it). It needs its OWN falsifier (a default-sampler node
       saved→reloaded renders identically; a user-bound one still round-trips) since it touches the
       general persistence path. This is the one app-wide change in slice 1.
   - **NO absolute path in the row (corollary-1):** the bound `MediaWithTexture` carries the FULL abs
     source path in `details.file_details.path` (`media.py`). The row shows ONLY dimensions + kind
     (and, if useful, the persisted `media/<uniform>.<ext>` basename — which is just the uniform name,
     `ui_models.py:290`, so it adds little). NEVER render `details.file_details.path`. The working set
     is a model-visible channel; the path-leak falsifier (below) covers it.
   Source: `media.py` `MediaDetails`, `ui_models.py` sampler branch. This is the fact that makes slice 2
   usable — the model must SEE a sampler (bound or default) to offer binding.

2. **Canvas size rides the working-set node header.**
   ```
   === Fire (id: a1b2) [current] canvas 512x512 ===
   ```
   From `node.canvas.texture.size`. Add a `canvas` field to `WorkingSetView`; render it in the header.
   (The project map `node_tree` stays lean — canvas is per-working-node detail, not map-wide; adding
   it to every map row taxes the cacheable prefix for a fact only relevant to nodes in play.)

3. **Uniform control ranges append to the numeric rows (LEAN form).**
   ```
   u_glow  float = 0.4  [0..1]
   ```
   Only when the UI declares a non-default min/max (`UIUniform`); omitted otherwise (don't pad every
   row). This grounds relative asks. **If it measurably bloats the working set, drop it** — it is the
   softest of the three and can be its own micro-task or cut.

4. **A short AWARENESS note in the prompt FEEDBACK section** teaches the three new facts (what `<-`
   means, that `(no media bound)` is the cue to offer `bind_media`, that canvas size is the render
   resolution). Kept tight — the facts are self-describing data, the note is one or two lines.

## Out of scope

- A standalone `list_media` tool — **dropped, not deferred.** Media is per-node/per-uniform (no shared
  asset pool on disk), so "what textures does this node have" IS the sampler rows of decision 1. A
  separate tool would duplicate an existing channel (skill §4: enrich the channel, don't add a tool).
- Showing the sampler's texture CONTENT (thumbnail / pixel facts). The copilot is render-blind; a
  `probe_render` already gives the composed-frame facts. Trigger: a real need to inspect a bound
  texture in isolation.

## Files touched

- `copilot/capabilities.py` — `WorkingSetView` gains `canvas: str = ""` (DEFAULTED — appended after the
  existing defaulted `script_listing`/`script_errors`, or every `WorkingSetView(...)` constructor in
  `tests/test_working_set.py` breaks); the sampler-binding string folds into the existing
  `uniforms: list[str]` rows (no new field — the row IS the carrier).
- `copilot/backend.py` — `_format_uniforms` sampler branch (default-vs-bound detection + dims/kind, NO
  path); `_copilot_node_working_view` fills `canvas` from `node.canvas.texture.size`.
- `ui_models.py` `UINode.save` — SKIP a still-default sampler (don't write its `media/` file);
  app-wide persistence change (see decision 1 blast-radius). Its falsifier is below.
- `copilot/prompt.py` — `_render_working_set_member` renders `canvas` in the header; the AWARENESS
  note in `_SYSTEM_PROMPT`.
- `tests/_caps.py` — no new method here (awareness adds no capability method), but `tests/test_working_set.py`
  constructors must pass with the new defaulted field.

## Manual verification

- **Falsifier for decision 1:** a node with a USER-bound texture shows `<- (WxH, image)`; reset it to
  default (unbind / a fresh sampler) and the same node's row flips to `(no media bound)`. A row that
  reads identically bound-vs-default verifies nothing.
- **Falsifier for the save-skip round-trip (the round-2 fix):** a node whose sampler is DEFAULT →
  `save()` writes NO `media/<uniform>` file (assert the file is absent on disk) → reload → the working
  row reads `(no media bound)` (NOT `(WxH, image)`). Without the save-skip this goes red (the reloaded
  path is `media/…`, not the default). Then a USER-bound sampler: save writes `media/u_tex.<ext>`,
  reload still reads `<- (WxH, image)`. Both round-trips in one test.
- **Falsifier (no path leak in the row):** bind a texture whose source dir is a known sentinel path;
  build the working view; assert the sentinel abs path is NOT in any row string (only dims/kind). This
  extends the corollary-1 guard to the working-set channel (slice-2's grep covers msg/history, not the
  working set).
- **Falsifier for decision 2:** `set_canvas_size` (slice 3) or a UI resize changes the header string;
  assert the working-set header reflects the NEW size, not a cached one (rebuilt-every-step invariant).
- Headless: construct a `Node` with a `sampler2D`, bind an `Image`, build the working view, assert the
  row string. No `App` needed (the working-view build is bridge-marshalled but the render is on a
  standalone GL context — see `dev_flow.md ## Authoring nodes directly`).
