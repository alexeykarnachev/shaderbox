# 00 — Grounding: current surface, on-disk reality, hard constraints

> Grounded by reading the code on disk (2026-07-02), not from memory. Cites are by file:symbol so a
> re-verify is a grep, not a recall.

## Current copilot capability surface (what exists)

| Layer | Tools |
|---|---|
| Awareness (in prompt) | project map (`node_tree`), `SB_*` catalogue, template catalogue, conventions |
| Reads | `read_shader`, `grep`, `read_lib`, `read_script`, `read_working_set`, `probe_render(t)` |
| Source mutations | `edit_shader`/`write_shader` (content-addressed), `edit_script`/`write_script`, `set_uniform` |
| Node ops | `create_node`, `delete_node`, `switch_node` |
| Deliverables | `render_image`/`render_video`, `publish_telegram`/`publish_youtube`, Telegram pack CRUD, YT creds |

Well covered: GLSL logic, scripts, scalar/vector uniforms, publishing.

## The blind spots (grounded)

1. **Textures / media are invisible AND unmutatable.**
   - Nodes really do carry `sampler2D` uniforms bound to `Image`/`Video` (`media.py`
     `MediaWithTexture`), persisted to `nodes/<id>/media/<uniform>.<ext>` on save
     (`ui_models.py` `save` — `GL_SAMPLER_2D` branch writes `media/{name}`, records
     `local_file_path`). Loaded back by `core.py` from `MEDIA_DIR_NAME` / `TEXTURES_DIR_NAME`.
   - `set_uniform` **explicitly rejects** samplers (`backend.py:918` — `label.startswith("sampler")`
     → *"samplers and uniform blocks are not settable"*).
   - The working-set uniform rows (`_format_uniforms`) do not surface **what** is bound to a sampler.
   - `bind_media` was **parked** in feature 020 (`11_capability_wave_spec.md`; re-named as a trigger
     in `30_turn_rollback.md` decision 9).
   - Net: for a real shader playground, the single most important input type (a texture) is entirely
     outside the copilot's reach — it can't see it, change it, or add one.

2. **Node canvas size is invisible.** A node has a `canvas_size` (persisted in `node.json`,
   `ui_models.py` `save`; loaded `core.py` `canvas_size=metadata.get("canvas_size")`), but it is in
   NO copilot read channel. "Why is my render blurry / pixelated" → the copilot can't see the node is
   100×100.

3. **No file-manipulation within the project.** No `rename_node`, no `duplicate_node` (fork a shader
   to experiment — a playground staple), no `set_canvas_size`, no lib-file delete/rename
   (`delete_lib_file` also parked).

4. **No path into the project from the user's disk.** The user can't say "import this shader file" or
   "use this image". The app already opens the native picker for media in the UI
   (`widgets/uniform.py:246` — `pfd.open_file`), but the copilot has no equivalent.

5. **Uniform control ranges are invisible.** The UI slider min/max/step (`UIUniform`) never reach the
   model, so a relative ask ("brighter", "slower") has no grounded range.

## Two hard constraints every slice inherits

### C-1. All tools are `eager=True` today; the lazy path is dead code.
`registry.eager_specs()` returns every tool (all are eager); `registry.specs_for(names)` — the lazy
loader — is defined but **never called** (`agent.py:405` uses `eager_specs()` only; the lazy
"D5" catalogue was parked across features 020 + 027). So **every tool's schema is re-billed on every
iteration** (skill §4/§6: input is the SUM across iterations). Adding 7 tools eagerly is a permanent
per-turn tax. → this is why slice 0 (wire the lazy catalogue) is a prerequisite, not a nicety.

### C-2. Checkpoint scope decision 9 names `bind_media` as its re-verify trigger.
Turn-rollback (feature 030) snapshots **only** `shader.frag.glsl` + `node.json` per touched node —
NEVER `media/` / `textures/` — and the spec says verbatim: *"Re-verify this scope if `bind_media` ever
lands (then a media-binding turn would need its media/textures captured too — a trigger for this
feature)."* (`30_turn_rollback.md` decision 9). → slice 2 MUST resolve the revert semantics of a
media bind (see `02_media_literacy.md` + the open question in `99_synthesis.md`).

## Actor-model fit (why this is the right shape, not raw fs access)

The prompt's sandbox line — *"no shell, no Python, no filesystem beyond the tools"* — is not being
broken. Raw shell / arbitrary-path fs access would violate corollary 1 (the model synthesizing paths
= silent mislocation). Instead every "touch the user's system" op is a **typed, user-in-the-loop
picker**: the model triggers a native dialog, the USER supplies the path, the model gets back only a
safe basename + metadata. "More skilled at system interaction" = more of these typed, consent-gated
primitives — not a raw escape hatch.
