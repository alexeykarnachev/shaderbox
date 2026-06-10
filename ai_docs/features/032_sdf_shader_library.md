# 032 — SDF shader library (seed) + copilot quality wave

**Status:** in progress (library seeded + copilot fixes landed; the seed-loading mechanism and
further dogfood-driven iteration continue next session, on the desktop).

## Goal

Make the copilot write GOOD shaders by giving it a lean, layered GLSL library it can actually
reuse — instead of re-deriving (badly) per shader. Practice-first: the library shape was driven by
live ad-hoc dogfood experiments (drive the agent, eyeball renders, fix what visibly fails), not by
up-front design.

## The library contract (the load-bearing decision)

Everything is built around **signed distance** as the universal interface, in three layers marked
by name prefix:

1. **Sources `SB_sd_*`** — return a signed distance (negative inside, uv or documented local
   units): `SB_sd_text` / `SB_sd_char` (with `weight` = boldness), `SB_sd_circle`, `SB_sd_box`,
   `SB_sd_segment` (documented unsigned-skeleton exception).
2. **Operators `SB_op_*`** — SDF in, SDF out: `union`, `smooth_union`, `intersect`, `subtract`,
   `round`, `onion` (onion = outline/ring as an SDF).
3. **Renderers** — SDF in, 0..1 mask out: `SB_fill(d, smoothness)`, `SB_fill_aa(d)`
   (fwidth-clamped — see quirks), `SB_glow(d, radius)`.

The agent composes: source -> ops -> render (one CONVENTIONS line in `prompt_context.py` states
this). Utilities outside the contract: `SB_center_uv`, `SB_rotate`, `SB_hash21`, `SB_value_noise`,
`SB_text_size`, `SB_text_fit`, `SB_text_char_center`. ~20 public functions, catalogue cost ~0.9k
tokens. Non-`SB_` names are library-private (filtered from the copilot catalogue — `backend.py`).

## What landed

- **The seed** at `shaderbox/resources/shader_lib/` (canonical, in-repo, ships via `build.sh`'s
  whole-package copy): `sdf2d/` (segment, shapes, ops), `draw/render.glsl`, `space/` (center_uv,
  rotate), `noise/value_noise.glsl`, `text/` (glyphs, layout).
- **The text stack** (extracted + extended from the segment-glyph template, kept rounded-arc
  style): full A-Z, Cyrillic А-Я/Ё (new glyphs on the same lattice; rounded per maintainer
  feedback rounds), digits 0-9, punctuation `! ? : ; , . - ' &`; per-line-centered block layout
  (`SB_sd_text`), measuring (`SB_text_size`), auto-fit (`SB_text_fit` — kills the off-screen
  class), per-char access (`SB_text_char_center` — jitter/wave/per-char effects).
- **Copilot quality fixes** (driven by dogfood experiment 1 findings):
  - Prompt: new-scalar/vec uniforms get INLINE defaults in source (the old blanket "NEVER
    default-initialize" was an overgeneralization of the array-only GLSL limit); arrays via
    `set_uniform`. Killed the 10-iteration set_uniform spam.
  - Prompt: batch independent tool calls in one step (the loop always executed multiple calls;
    the model just never emitted them together).
  - `max_iterations` 12 -> 16 (`config.py`).
  - `lib_catalog` filters to `SB_`-prefixed (the prompt always promised "every SB_* signature";
    private helpers would have been ~76 noise lines).
  - CONVENTIONS: the SDF-layer contract line.
- **Review swarm** (3 adversarial reviewers, render-verified): 20+ findings, all HIGH/MEDIUM
  applied — exact-ish tall-arc distance (Ж rendered 2x bolder: first-order |phi|/|grad phi| fix),
  `SB_rotate` direction trap documented (+angle spins the shape CW), `SB_fill_aa` fwidth clamp
  (full-frame seam line on runtime-mutated arrays — the typewriter idiom), doc typical values
  everywhere, `u`-suffixed switch labels, Ё dot separation, Й breve deepened.

## Dogfood evidence (2 ad-hoc experiments, codex-mini, ~$0.10 total)

- Exp 1 (neon sign + CRT): full library reuse turn 1, zero compile errors across all turns; BUT
  visual blindness fired 3x (spacing left (0,0) -> letter blob reported as success; 4 blind
  spacing re-edits in a row; ring asked -> filled disc). Iteration budget burned by per-uniform
  set_uniform calls (fixed, see above).
- Exp 2 (spinning neon poster, fresh agent on the polished lib): first turn, 10 edits, clean
  compile, full reuse incl. `SB_rotate`/`SB_op_onion` echo, inline defaults everywhere, creative
  extras. Renders: `proj-917r1qmq` + time-sampled stills.

## Known quirks (this feature's own)

- **V3D first-draw codegen (SOLVED by data-driven glyphs):** Mesa v3d compiles the final GPU
  code at first draw on the CPU, retrying up to 13 strategies; the old code-based glyph switch
  cost ~20s per shader (10+ min for multi-text-layer). `scripts/gen_glyphs.py` now generates
  glyphs as constant stroke TABLES + a tiny evaluator loop: compile ~1s, pixel-identical output
  (diff max=0), warm render ~60-180ms/300px on the Pi (slightly slower warm than the old inlined
  code — stroke data now reads through memory; a cell-cull optimization that wins it back is
  parked in todo with a quad-seam artifact note). Glyph edits go in the python tables, then
  regenerate.
- `SB_text_char_center` is two text scans per call — per-char loops must bound at the real text
  length (doc warns; a full-64 loop nesting it is pathologically slow).
- Engine `u_text` arrays cap at 64 codepoints (`str_to_unicode` truncates silently).
- `shader_lib` index keeps the FIRST of duplicate function names silently (`index.py`) — fine at
  seed size, a diagnostic when the lib grows.
- Agent chat replies render Cyrillic as `??????` (the D2 ASCII sanitizer) — shader text itself is
  fine (set_uniform path). Filed in todo.

## Resume on the desktop (cold-start)

1. The canonical library = `shaderbox/resources/shader_lib/`. No load mechanism yet — copy it:
   `cp -r shaderbox/resources/shader_lib/* <app_data_dir>/shader_lib/` (Linux:
   `~/.local/share/shaderbox/shader_lib/`). NEXT step is wiring the seed properly
   (copy-on-first-run vs second search root — open question below).
2. Dogfood missions need the lib INSIDE the run's sandbox:
   `mkdir -p scripts/dogfood/runs/data-<run> && cp -r shaderbox/resources/shader_lib
   scripts/dogfood/runs/data-<run>/shader_lib`, then pass
   `SHADERBOX_DATA_DIR=$PWD/scripts/dogfood/runs/data-<run>` on every turn command (`/dogfood` §1).
3. The ad-hoc experiment flow (no scenario files, maintainer steers live): drive turns via the
   harness, render, send the maintainer stills every 3-5 turns; wipe memory between experiments by
   creating a fresh project.

## Out of scope / deferred (each with its trigger)

- **Seed-loading mechanism** (shipped seed -> user lib dir; versioning per the shipped-default +
  user-sidecar convention) — NEXT session's first decision.
- **`inspect_render`** — the todo deferral's trigger has now FIRED repeatedly (blind spacing
  roulette, disc-vs-ring, imperceptible-effect class across sessions); spec-first, build when the
  maintainer green-lights.
- **Cyrillic chat replies** (ASCII sanitizer) — todo.
- **Harder text missions** (typewriter probe, per-char jitter mission, THRASH recovery) — next
  dogfood rounds; the library-side primitives are ready.
- Д glyph shape the maintainer dislikes (parked: "пока так и оставим").
- Lygia/hg_sdf/stegu MIT vendoring for `noise/`/`sdf3d`/`color` modules — when the maintainer
  wants those domains (licensing map in the 2026-06-10 research chat: stegu/hg_sdf/iq MIT-safe,
  LYGIA Prosperity = reference-only).

## Open questions for the user

1. Seed loading: copy-on-first-run into the user lib (user edits win, updates need a version
   stamp) vs a second read-only search root (shipped half stays pristine)?
2. Should the in-app Text template eventually call the library instead of carrying its own 616
   lines (couples templates to the lib — needs the seed mechanism first)?
