---
name: shader-lab
description: "Run an iterative, step-by-step shader-development session in ShaderBox — build a visual effect (fire, lightning, smoke, a portal, …) as a sequence of small atomic steps the user watches evolve, versioning every step so we can go back and compare. Use when the user says 'let's build/iterate on an effect', 'start a shader experiment', 'shader lab', 'let's develop a <X> shader step by step', 'давай поэкспериментируем с шейдером', 'итеративная разработка шейдера'. The deeper PURPOSE: synthesise reusable techniques/findings for the copilot + ShaderBox, then promote the core into copilot instructions / SB_* library helpers / app features. Living skill — improve it each run; mark not-yet-ready ideas as follow-ups."
user_invocable: true
---

<command-name>shader-lab</command-name>

# shader-lab — iterative step-by-step shader development

Build a visual effect as a sequence of small, reviewable steps the user watches unfold in real
time (or via rendered MP4 when offscreen), **versioning every step** so the evolution is replayable
and comparable. The point isn't just the effect — it's to **mine reusable knowledge** (techniques,
formulas, the user's aesthetic verdicts) that later gets promoted into the copilot / the `SB_*`
library / ShaderBox itself.

This is a **living skill**: every run, fix the steps you tripped on and add to `## Follow-ups`.

**How to read this** — four stable blocks:

- **READ NOW, every session (PROCESS):** The ONE rule · Setup · project location · live-preview
  contract · Versioning · NOTES.md · Rendering · Researching/Past-labs. These run the loop; read top
  to bottom at session start.
- **CONSULT WHEN RELEVANT (GRAPHICS CRAFT):** Shader craft · Raymarching/SDF *(marched scene)* ·
  Lighting *(a render reads flat)* · Color & tone · Motion & timing · Step-reveal+crossfade *(a reveal
  document)*. These name LEVERS; the working code lives in the reference labs (`## Past labs`).
- **ONLY IF the effect runs a Python sim (CPU-SIM):** CPU scripting · Specific-algorithm sims ·
  Selective-variant tuning · Verify the premise. Pure-GLSL effects skip this whole block.
- **END:** extract value · Past-labs reference map · Follow-ups.

**Extend it ADDITIVELY — never re-cut the structure to add a lesson:** drop a bullet into the matching
section, or append a whole new `##` section at the END of its block. A new technique → a bullet + a
pointer to the lab that implements it. A new lab → a `## Past labs` bullet. A not-ready idea →
`## Follow-ups`.

---

## The ONE rule that matters most

**Confirm the FORM before you build. NEVER overwrite a previous version. NEVER write your own visual
conclusions into the notes.**

The costly mistakes from past sessions:
- **Restate the structure the user described, in your own words, BEFORE generating anything — a wrong
  mental model multiplies into N wrong artifacts.** (Asked for "one shader where steps reveal over
  time", an entire turn was burnt building NINE separate documents — the decomposition was right, the FORM
  was wrong.) When the user describes a shape ("a single X that does Y over time", "steps that stack"),
  say it back and get a nod before writing files.
- Overwriting a shader in place destroyed earlier versions — we couldn't go back, compare, or watch
  the evolution. **Each step is a NEW document** (project-native: a new `documents/<id>/` grid cell). You
  author a fresh document per version and never touch an existing one. (Versioning section below.)
- "Looks like a good flame" from *your own* read of an image is unreliable — your visual judgment is
  poor. The notes log records the USER's feedback, your WEB findings, and concrete formulas — **never
  your own "this looks good/bad" conclusions.** (Notes section below.)
- **For a visual BUG, find the root cause — don't guess-and-patch, and don't override the user's
  report with your own theory.** Reproduce/instrument before fixing: paint a debug field by a
  variable's sign/value to SEE which screen region it covers; measure per-pixel frame deltas to locate
  a one-frame pop; crop the suspect area and look. Twice this session a guess sent the fix the wrong
  way (chasing brightness for a coordinate bug; concluding "occluded" when the user kept saying "I
  clearly see it" — the user was right, my instrument reading was wrong). When the user repeats a
  correction, your MODEL is broken — re-derive, don't defend.

---

## Setup — at the START of a session, settle these with the user

**Precondition — start from a CLEAN working tree.** A lab session assumes `git status` is clean at
start; if it isn't, ASK the user before proceeding — trivial ambient churn (a scratch file, an
unrelated formatting nit) is ignorable, but flag anything substantive (an uncommitted source / skill /
doc change) and let them decide. Don't silently build on another session's in-flight work.

1. **Effect + concept.** What are we building (fire / lightning / smoke / …)? Restate it back in your
   own words (the ONE rule) before writing files.

2. **Iteration model — WHICH kind of versioning? ASK up front; do NOT presume.** The two modes produce
   completely different document sequences, and picking the wrong one multiplies into N wrong artifacts (I
   have repeatedly defaulted to the wrong one — so ASK, never assume):
   - **Whole-image iteration (default when the goal is realism / a hero shot):** each version is a
     COMPLETE, best-effort render of the ENTIRE effect — the max you can do that round. Between versions
     the user critiques and you improve the whole thing (swap a fake for the real model, better light,
     better material). The vNN are complete standalone pictures to compare side by side — NOT partial
     build stages. This is the mode the realism meta-lever assumes; the `us_flag` arc (v01→v18, each a
     full flag) is its shape.
   - **Component progression (a "how it's built" teaching decomposition):** each version ADDS one
     mechanic on top of the previous (e.g. fire: base gradient → warped-noise → flame shape → color
     ramp → glow → embers → smoke); early versions are intentionally incomplete. Good when the
     deliverable is a learning reel (see the reveal-reel pattern near the end).
   - Ask verbatim: "Each version a COMPLETE standalone attempt we critique & improve, or INCREMENTAL
     build-steps that each add one mechanic?" Honour the answer for the whole session.

3. **Environment — live vs offscreen. Infer from context, then CONFIRM.**
   - **Live** (the user has ShaderBox open and is watching the preview): they open the project and
     watch the preview update live as you rewrite the shader. You do NOT render anything to show them —
     the app does it. (You still render stills for YOUR OWN eyeballing.)
   - **Offscreen** (no live app/display — the user is away from the app, on another device, or the
     session is headless): you render a **~10s MP4** of each result you want to show and deliver it on
     whatever channel you're talking to the user through. Same iteration loop, no live app. Prefer
     **.mp4** (H.264) — it's the most universally-playable format across devices; avoid WebM.
   - When unsure which, ASK: "are you watching live in ShaderBox, or should I render MP4s to show you?"

4. **Stop cadence.** Ask up front: "how often should I stop for your review?" — options: after every
   step / once at the very end / at my discretion (stop only at a real fork or when I need a verdict).
   Honour it for the whole session unless the user changes it.

---

## The project lives in `projects/_lab/<name>/` (gitignored)

- Create one fresh ShaderBox project **per experiment** under `projects/_lab/<slug>/` — it's a
  collection of documents (each = a `document.json` + `passes/<name>.frag.glsl` + optional `graph.json`
  and `scripts/script.py`).
- `projects/_lab/` is **gitignored** (`.gitignore`), so nothing here pollutes the repo. A worthwhile
  result is promoted **by hand at the very end** — only if the user decides to keep it. Two targets:
  a real project (move the document dir + `git add`), or — for a polished showcase — a **shipped
  example**: copy the document dir into `shaderbox/resources/document_examples/<uuid>/`, append the uuid to
  `constants.py::EXAMPLE_ORDER`, give it a stranger-facing `ui_name`/`description` in `document.json`,
  promote any live-root-only `SB_*` helpers it uses into `resources/shader_lib/` (the fresh-env
  resolve is pinned by `tests/test_examples_resolve.py` — regenerate its api-lock snapshot). The
  canonical reference: the fire showcase (feature 051).
- Tell the user the path and to **open that project in ShaderBox** (File → Open project → the
  `projects/_lab/<slug>/` folder), THEN you begin iterating, so the app has the document loaded.

### Project / document layout to write

A document dir is `document.json` + `passes/<name>.frag.glsl` (+ optional `graph.json` when it has more
than one pass, and `scripts/script.py` for CPU-driven
uniforms). `SB_*` helpers resolve from the live shader lib automatically. The smallest valid
`document.json`:

> **VERIFY the `SB_*` inventory before using a helper — the shipped set is SMALLER than old labs
> assume.** The canonical library ships in `shaderbox/resources/shader_lib/`; the current set is only:
> `SB_value_noise`, `SB_hash21`, `SB_sd_box/circle/segment`, `SB_op_*` (union/subtract/intersect/
> smooth_union/round/onion), `SB_fill`, `SB_fill_aa`, `SB_glow`, `SB_center_uv`, `SB_rotate`,
> `SB_sd_char/text` + `SB_text_*`. **`SB_fbm` was REMOVED** — the `fire`/`us_flag` labs reference it and
> WON'T compile as-is; write a local fbm from `SB_value_noise` (`fbm(p)=Σ aᵢ·SB_value_noise(2ⁱp)`), and
> a local star SDF (iq `sdStar5`) — neither ships. Confirm with
> `grep -rhoE '\bSB_[a-z0-9_]+\s*\(' shaderbox/resources/shader_lib/ | sort -u`. If a `SB_*` won't
> resolve at all (empty live lib), seed it once:
> `python -c "from pathlib import Path; import shaderbox; from shaderbox.shader_lib.seed import sync_shipped_lib; from shaderbox.paths import shader_lib_root; sync_shipped_lib(Path(shaderbox.__file__).parent/'resources'/'shader_lib', shader_lib_root())"`.

```json
{ "canvas_size": [720, 1280],
  "uniforms": {},
  "ui_state": { "ui_name": "<effect> v01 (port)",
                "render_media_details": { "duration": 1.0 } } }
```

- `uniforms` is **REQUIRED** — leave it `{}` for a fresh document (it holds SAVED tuned values; the engine
  subscripts it, so a document missing the key is skipped). Defaults come from the shader's inline-default
  uniforms, not here.
- **Do NOT copy `uniforms` / `ui_uniforms` from an unrelated document** — you'll drag its saved state in.
  Leave `ui_uniforms` out entirely; introspection rebuilds it on load (you never hand-author the keys).
- `ui_name` = grid label; `canvas_size` = preview size + aspect; `ui_state.render_media_details.duration`
  = render loop length. (JSON has no comments — keep the legend in your head, not in the file.)
- For a fuller real example to mirror, open any committed document, e.g.
  `projects/_lab/night_city/documents/<uuid>/document.json`.

---

## The live-preview contract (live mode) — why it Just Works, and the one requirement

ShaderBox's per-frame mtime watcher (`watch.py::reload_document_if_changed`, called for EVERY loaded document
in `ui.py::update_and_draw`) **recompiles a pass the instant its `.frag.glsl` mtime changes on
disk**, and `reload_scripts()` does the same for `script.py`. AND ShaderBox reconciles `documents/` to
disk every frame (disk is the source of truth), so a document dir you CREATE / DELETE on disk syncs into
the grid AUTOMATICALLY — no reload, no bookkeeping. Together that means:

- **A new version = a new document dir** → it just appears in the grid as a fresh cell, live.
- **Editing a document's shader** → that cell recompiles in place.

So the flow is: user opens the project once; thereafter you write a new `documents/<id>/` per step and it
shows up on its own. The only requirement is that the project was opened (so the app is watching that
dir); everything after is automatic.

**The user will NOT edit the shader during iterations** (they only watch / may view the code / may
have the editor focused). So there is no edit-collision to fear — you own the files, they observe.

---

## Versioning — one DOCUMENT per step, never overwrite

Use the project's NATIVE layout — no parallel snapshot dir, no scaffolding. **Each step is its own
document** under `documents/`; the grid IS the version history. To preserve a version you simply leave its
document alone; to make a new one you author a fresh document dir. You never overwrite an existing document, so
nothing is ever lost.

```
projects/_lab/<slug>/
  documents/<id-1>/passes/main.frag.glsl   # v01 — its own grid cell, named "<effect> v01 (<short-name>)"
  documents/<id-2>/passes/main.frag.glsl   # v02
  documents/<id-3>/passes/main.frag.glsl   # v03
  documents/<id-3>/scripts/script.py  # if that version had a script
  ...
  NOTES.md                        # the experiment log (see below)
```

**Each step:** (1) create a NEW `documents/<id>/` (the dir name is the document id — any readable string like
`v03-warp`, not a uuid; the grid sorts by creation time, not name; copy the document.json shape from an
existing document and set `ui_state.ui_name` to `"<effect> vNN (<short-name>)"` so the grid reads as a
progression); (2) write its `passes/main.frag.glsl` (+ `scripts/script.py` if any) — it appears in the grid
on its own (the per-frame disk-sync from `## The live-preview contract`); (3) append a NOTES.md entry.
All versions sit side-by-side as their own grid cells, so "I like v3's tip but v5's color, take both"
is just opening both — every version stays a live document to diff and cherry-pick from.

**Caveat:** don't hand-edit a document the user has OPEN in the app — on exit/save it rewrites that document's
files from its in-memory state. Author NEW documents for new steps (never contended); leave the user's
open document alone.

---

## NOTES.md — the experiment encyclopedia

A per-project log so we can see *exactly what changed and why* across the evolution. One entry per
version. **Record only verifiable inputs, NEVER your own visual judgments:**

- ✅ **The user's verdict** (verbatim or close: "still looks like a triangle", "drop the wind").
- ✅ **Web findings / techniques / formulas** you researched, WITH the source URL.
- ✅ **What changed** mechanically in this version (the diff in plain words) — the MECHANIC, not just
  the verdict: the term/function/formula that changed, so the entry is a real reference later.
- ❌ **NOT** "this looks great now" / "the interior is nicely structured" — your read of a rendered
  image is unreliable; leave the aesthetic verdict to the user.

Entry shape (effect-agnostic skeleton; `vNN` so a draft doesn't pollute the outline):

```markdown
## vNN — <short name> (document <id or ui_name>)
- Change: <the mechanical diff in plain words — the term/function/formula that changed>.
- Source: <technique name> <URL>        # only when researched
- Shader: documents/<id>/passes/main.frag.glsl   # link the code so the entry IS a reference
- User verdict: <verbatim, or "pending review">
```

Start NOTES.md with a header: effect name, date, environment (live/offscreen), stop cadence.

---

## Rendering (for YOUR eyeballing, and for offscreen delivery)

`.claude/skills/shader-lab/render_document.py` — headless EGL render, no app needed:

```bash
# still PNG at time t — for YOU to eyeball a result before/while iterating:
uv run python .claude/skills/shader-lab/render_document.py image <document_dir> <out.png> --t 0.5 --size 512

# MP4 clip — the OFFSCREEN deliverable (use .mp4/H.264 — most universally playable; avoid WebM):
uv run python .claude/skills/shader-lab/render_document.py video <document_dir> <out.mp4> --seconds 10 --fps 30 --size 512
```

- Renders against the live `SB_*` lib; video goes through `Document.render_media` (so a `script.py` ticks
  a fresh per-export instance — export-isolation, same as a real export).
- **`render_document.py` forces a SQUARE canvas** (`--size` → size×size). For a non-square aspect (e.g.
  9:16 portrait) it distorts — write a small inline render with the real `canvas.set_size((w, h))`
  instead (the snippet in `dev_flow.md ## Recipes > Authoring / debugging documents directly`). Cropping a
  region of the render (PIL `.crop`) to eyeball one area (a corner seam, a rooftop, a street) is the
  fastest way to inspect a localized issue.
- **Eyeball every step yourself IF you can read a render** — render a still and look at it to inform
  your NEXT iteration. But per the ONE rule, your own read informs the next move only; it does NOT go
  into NOTES.md as a verdict (the user's verdict does). If you CAN'T reliably read a render, don't
  invent a visual read — get the artifact in front of the user and act on their verdict.
- **Offscreen mode:** render the MP4 and deliver it however you're communicating with the user (attach
  it / surface the file on your channel; if you have no way to send files, give them the output path).
  Intermediate clips at review stops, a final clip at the end.

---

## Researching techniques

When an effect needs a technique you don't have cold (how real fire avoids a flat core, a teardrop
silhouette, lightning branching, how a night cityscape gets depth, etc.), **WebSearch / WebFetch real
sources** — Shadertoy breakdowns, iquilezles.org, Cyanilux, game-engine VFX writeups, and for
art-direction questions (value/contrast/color) digital-painting + concept-art sources (Mitch Albala,
ctrl+paint, Marco Bucci, art blogs). Working example code is the canonical pattern; read it, find the
divergence from your code. Put the technique + URL in NOTES.md.

**Check past labs before researching from scratch** — a prior session may already carry the
technique, the maintainer's aesthetic verdicts, and the dead-ends. See `## Past labs` below; read the
relevant `projects/_lab/<slug>/NOTES.md` FIRST.

- **Read the lab's OWN NOTES (and its shader), not the digest in THIS skill.** The `## Lighting`,
  `## Raymarching/SDF` etc. sections here are a one-line-per-idea INDEX — by construction they drop the
  formulas and the why. (The boids lab reimplemented "night look" from this skill's bullet summary and
  it read FLAT — it had silently dropped the directional sky/moon key, which the night_city v06 NOTES
  states in full as `key = 0.55 + 0.45*skyAmt + 0.40*moonAmt`, surface-keyed with emissive split. The
  NOTES were complete; the skill digest was not, and I lazily read the digest.) The skill points you AT
  the source; the NOTES entry + shader ARE the source. Open them.
- **So the duty cuts both ways: NOTES must be COMPLETE (record the mechanic — formula, split, the
  numbers — not just the maintainer's verdict), and they must LINK their shader** (`documents/<v>/passes/main.frag.glsl`)
  for the exact code. A reusable lesson that only lives as a verdict ("added a night key") forces a
  re-derivation next time.

---

## The realism meta-lever: REAL models beat hand-tuned fakes

The single biggest lesson, above any specific technique. When something must READ AS REAL, implement the
actual physical/mathematical model from a PRIMARY source — not a plausible-looking hand-tuned
approximation. **A fake plateaus and reads cheap; polishing it burns more iterations than building the
real thing would.** The examples below are just instances — the rule is effect-agnostic.

- **Real model > "looks about right" fake.** Every hand-rolled approximation hit a ceiling and got
  rejected as "cheap/fake"; the real model read right immediately. (Instances from one lab: a sine-wave
  displacement "cloth" → real Verlet mass-spring sim; a hand-picked color-stop sunset → analytic
  Rayleigh/Mie single-scatter; an ad-hoc `pow(1-NdotV)` rim → the Charlie/Ashikhmin cloth sheen BRDF; a
  diffuse yellow disc → a metal sphere shaded by environment reflection.)
- **Organic motion EMERGES from feedback; it is not imposed.** An applied oscillating field (a travelling
  sine) is periodic by construction → mechanical. Real motion emerges from a system with the right
  coupling — e.g. flag/membrane flutter is an aeroelastic instability where the force depends on the
  RELATIVE velocity between medium and surface, so the object extracts energy from the flow and
  self-oscillates into a NON-periodic limit cycle. "Looks looped/fake" → you're imposing motion instead
  of letting it emerge; add the feedback term (relative velocity, added mass) and seed the instability.
  Verify non-periodicity numerically (autocorrelation with no repeated peaks).
- **Ground each "make it real" model in a PRIMARY source, not memory.** "Make it realistic" → WebSearch/
  read the actual paper / reference implementation / domain writeup and lift the exact formula + parameter
  regime. Training-data recall gives a plausible-but-wrong approximation that wastes the iteration. Record
  the source URL in NOTES.
- **A metal / glass / polished surface is shaded by its REFLECTION, not by diffuse.** It is a mirror of
  the environment + a sharp light hotspot + Fresnel; a diffuse+spec Lambert reads as plastic. Reflect the
  ACTUAL scene so the object integrates into it.
- **Depth-cue hierarchy: cast SHADOWS > occlusion/AO > normal shading.** A surface lit only by its normals
  reads FLAT however good the normals. Real self-shadowing (parts occluding the light from each other) is
  what gives volume — compute it (ray vs the geometry) where you have the geometry, and light GRAZING (low
  angle to the surface); a frontal light casts almost no shadow.
- **Decouple constraints that fight.** When one stiffness/parameter secretly controls two things you need
  independently (inextensibility vs freedom-to-fold; coverage vs contrast), give each its own knob.
  Cranking the shared one to fix a symptom kills the other.
- **Backdrop detail comes from a PERSPECTIVE projection, not more noise.** A flat noise field reads blurry;
  projecting it onto a receding plane (detail compresses toward the horizon/vanishing point) gives depth +
  crisp structure. (Cloudscape, terrain haze, any far backdrop.)
- **Drama = value contrast + saturation, NOT more detail; structured texture reads as material, random
  color-mottle reads as cheap.** "Looks garish/cheap" is usually washed mid-tones + random color noise;
  deepen the base colors, raise contrast, and replace mottle with a STRUCTURED texture (a real weave/
  grain). A soft occluder over a blown highlight (a cloud across the sun) beats just dimming it.
- **Process discipline this lab paid for:** verify the premise in ISOLATION before rendering (a sim's
  stability/ranges/amplitude/autocorrelation in numpy catches a blow-up the render only shows as a black
  frame); tune the expensive part on the cheapest proxy (sky on a `t≈0` still — GL-dominated, no sim
  stepping; the sim numerically before any render); a REPEATED complaint ("still fake") means the MODEL is
  wrong — replace it, don't re-tune the fake; cull expensive per-pixel work (SDF raymarch) to the screen
  region where it can matter; keep ONE source of truth for a shared quantity (a light dir driving both the
  CPU shadow rays and the shader lighting; a camera basis driving both the mesh projection and a raymarch).

---

## Shader craft — making an effect READ right (general, not effect-specific)

Hard-won levers that apply to fire, smoke, water, lightning, energy — anything organic. Generalized
from real maintainer corrections; the parenthetical is the evidence, not a prescription to copy.

> **These sections name the LEVER, not the code.** The working implementation lives in a committed
> lab — that's WHY labs are kept. When you need a technique, REFERENCE the lab that has it (read its
> NOTES + open the cited shader) instead of re-deriving or copy-pasting a snippet here. The lab is the
> source of truth; this skill is the index. (Lab → technique map is in `## Past labs`.)

- **Shape from a thresholded/eroded NOISE field, not an SDF-primitive mask.** A circle/ellipse mask
  reads as a static blob; an organic silhouette should emerge from the noise itself — gate a soft
  envelope by the field and erode the edge. (A circle-union flame looked like a balloon; the
  noise-gated one read as flame.)
- **Domain-warp the noise for organic motion:** `fbm(p + k·fbm(p + k·fbm(p)))` curls where plain
  `fbm` only clouds — the single biggest "reads as alive" lever for fluid/fire/smoke.
- **For interior detail, feed the RAW continuous field to the color ramp — don't threshold-then-fill.**
  A binary burn mask × a flat color = a flat slab; mapping the raw field through the ramp keeps the
  internal veins/structure. (A whole iteration came out flat-yellow because the field was thresholded
  into a mask, then filled.)
- **Animate the BOUNDARY, not just the interior.** If the envelope is a static function of position
  and only the texture inside scrolls, it reads as "static shape with stuff moving inside" — displace
  the silhouette's sample point by a flow field so the whole outline licks/sways.
- **Light a scene by what it CASTS, not by recoloring the emitter.** A flame's flicker belongs on the
  glow it throws into the scene (a wide, scene-covering, sharp-falloff radial bloom), not on the flame
  body's color. (Maintainer: "the light spreads, it doesn't work like this" — the flicker was wrongly
  on the body.)
- **Self-illuminate anything meant to read on BLACK.** Smoke/haze over black has nothing to occlude →
  invisible unless it emits a little. (And if it still won't read after a fair attempt, cutting it is a
  legitimate call — don't ship an invisible feature.)
- **Anti-alias every hard threshold by ~1px in screen space**, not a hand-picked constant: soften with
  `smoothstep(-w, w, field)` where `w = fwidth(field)` (or `length(vec2(dFdx(field), dFdy(field)))`).
  Renders are small (512²) — the size where aliasing/crawl is worst and most visibly degrades the clip.
  Thin features (bolt cores, road lines, distant windows) need a min screen-width of ~1px or they
  sparkle and drop out under motion. Where `fwidth` is unreliable (raymarched SDF surfaces) render at
  2× and downsample, or jitter 2–4 samples.
- **Expose tunables as uniforms EARLY and hand tuning to the user.** When the right value is aesthetic,
  STOP guess-rendering — declare it as an **inline-default uniform** and let the user dial it live. The
  `= default` both seeds the value AND tells the engine the control type:
  `uniform float u_glow = 1.2;` (a drag), `uniform vec3 u_tint = vec3(1.0,0.4,0.1);`,
  `uniform uint u_octaves = 4u;` (an integer slider). After the user dials a value live and SAVES, read
  it back off the document's `uniforms{}` in document.json and bake it as the new inline default. (Several
  blind tune-render cycles evaporated the moment the knobs were exposed and the user dialed it in one
  pass.) For the engine mechanics of document files / headless render / compile-check / aspect, see
  `dev_flow.md ## Recipes > Authoring / debugging documents directly`.
- **Cheap cloth fake (ONE height field, used twice) — a QUICK-AND-DIRTY with a ceiling; know when it's
  not enough.** For a throwaway/background cloth: one scalar field `H(x,y,t)` bends the texture
  (`f.y -= H`) AND gives the shading normal (`n = normalize(vec3(-dH/dx, -dH/dy, k))`), pinned by ramping
  amplitude to 0 at an edge. BUT this session proved the fake reads "cheap/cartoonish" for a hero shot and
  no amount of polish fixes it — a real mass-spring Verlet sim (script-driven, pushed to the shader as a
  mesh) was what read as fabric. Reach for the fake only when the cloth is incidental; for anything that
  must read real, do the real sim (see the meta-lever above + the `us_flag` lab). Reference:
  `us_flag/documents/v05-scene/shader.frag.glsl` (fake) vs the `v13`/`v14` scripts (real Verlet aeroelastic).
- **Placing a fixed-RATIO rect on a non-square canvas: correct for `u_aspect`, or it stretches.** Equal
  `vs_uv.x` and `vs_uv.y` spans are UNEQUAL pixels when the canvas isn't square, so a shape sized in raw
  uv comes out the wrong ratio. For a rect that must read as physical ratio `R:1` in PIXELS:
  `h_uv = w_uv * u_aspect / R` (NOT `w_uv / R`). Cost a whole "why is my 1.9:1 flag drawing 3.6:1" fix.
  Same trap hits any aspect-round primitive — multiply the x-delta by `u_aspect` before `length()`
  (finial sphere, sun glow). Fake a lit sphere cheaply from a circle: `nz = sqrt(1 - (r/R)²)`,
  `n = vec3(p/R, nz)`, lambert + `pow(spec,32)`.

## Raymarching / SDF craft (generic)

Levers for any SDF-raymarched scene (cities, terrain, abstract solids — anything marched).

- **Accumulate TRUE ray distance, not a per-step sum.** `dist = length(p - ro)` (recompute each step),
  NOT `dist += length(p - ro)` — the `+=` re-adds the whole origin-distance every step, so it balloons
  super-linearly. A wrong distance silently breaks every distance-driven effect (fog, LOD, glow). It
  can look fine on a tiny/close scene and only blow up when the scene gets deep. (Cost a whole "why is
  the fog saturating everything" debug.)
- **Bounded domain repetition needs a NEIGHBOR-cell min, not a clamped round().** Tiling an SDF over a
  finite footprint via `id = clamp(round(p/spacing), -R, R)` as a one-shot pick is NOT a valid SDF at
  the boundary — near the edge the clamped cell can be the *wrong* (farther) one, the field
  over-reports distance, the march OVERSHOOTS and grazes a primitive's infinite-plane extension →
  phantom diagonal/triangular slivers in empty space. FIX (iq `opLimitedRepetition`): sample the
  rounded cell AND its 3×3 (or 3×3×3) neighbours, each clamped, and take the min. https://iquilezles.org/articles/distfunctions/
- **Detail by PAINTING the shaded surface beats adding geometry.** Recessed-window AO, sills, panel
  lines, grime, pilasters — fake them in the surface-shading function (zero march cost) rather than in
  the SDF. Reserve real geometry (extra `min()`-unioned primitives) for things that must change the
  SILHOUETTE (rooftop antennae, balconies). Each SDF primitive multiplies the march cost (a 3×3 repeat
  of one extra box = 9 more evals/step).
- **Discrete lights/props as emissive screen-space/ground glows, not 3D objects.** Cars, streetlamps,
  signs at a distance read perfectly as `attenuate(dist_to_point)` glows added to the ground/wall —
  far cheaper than SDF geometry, and you control them by simple 2D math. Use `1/(1 + b·d + c·d²)`
  falloff; for a *pair* of lights (headlights), draw two offset points and tighten each so they don't
  merge into one blob at normal distance.
- **A "cell-local" coordinate must round to the cell's CENTER, and the cell boundary must not land on a
  feature you care about.** Computing `local = p - round(p/S)*S` puts the seam at the half-cell; if
  your feature (a road centerline, a tile join) sits exactly on a `floor()`/`round()` boundary, half of
  it falls into the neighbour cell with a garbage local coord and renders nothing. Round to the
  feature's center explicitly (e.g. roads centered on half-integer cells → `round(p/S - 0.5) + 0.5`).
  (This was the "left lane of every street is empty" bug — a cell-assignment error, NOT occlusion.)
- **A sky feature (sun/moon/plane/cloud) lives in a fixed WORLD direction → whether it's on-screen is
  camera-dependent and unforgiving.** Don't guess its placement. Project the desired direction through
  the camera basis to screen NDC and SOLVE for the direction that lands it where you want:
  `a = dir·fwd; b = dir·right; c = dir·up; ndc = (b/a/aspect, c/a) * screen_dist` (a one-off numpy
  snippet; bake the resulting `vec3` as the default). Also: a feature drawn only where rays MISS
  geometry is OCCLUDED by tall foreground — keep it above the skyline or in an open gap.
- **Mind the render cost — it can be slow, and the deepest/most-nested lines dominate.** Bound the
  march step count and cap max distance (fogging out the far field is cheaper than marching it). Each
  added SDF primitive multiplies per-step cost; each fbm octave and each nested domain-warp level
  multiplies it again (so `fbm(p + k·fbm(p + k·fbm(p)))` is the single most expensive line in a noise
  effect — reach for it deliberately). A per-pixel `for` needs a compile-time-constant bound. Iterate
  at a small `--size` and short clips; render full-res / long only for the final deliverable. If a
  render is slow or times out, cut `--size` or the step count first.

## Lighting & art-direction (generic; the night-scene set is concrete)

When a render reads FLAT, the cause is almost always a value/contrast problem, not a missing feature.
General rule: **work the VALUE structure first** (large dark masses + a few bright accents + readable
fore/mid/background value groups); color and detail come after. Hue is the *weakest* depth cue —
lean on value + saturation + contrast.

**Night-scene rules (web-researched, broadly reusable; sources in `night_city/NOTES.md` v06):**
- **At night the SKY is the brightest large value; solids read as dark silhouettes against it.** This
  INVERTS daytime aerial perspective. Sky = vertical gradient: warm skyglow hugging the HORIZON
  (`pow(1 - rd.y, ~2.5)`) fading to a cool dark ZENITH. Horizon ≈ 8–12× a near unlit surface.
- **Even with no sun, light faces differently for FORM.** The sky-dome + moon act as a soft key: the
  face toward the sky/moon catches more (cooler) light, the away face is darker/warmer. Keep it
  SUBTLE (~1.3–1.6× value spread between faces) — night contrast is gentle. Apply to the SURFACE term
  only; split out emissive (lit windows/lights) so the key doesn't dim them.
- **Aerial perspective AT NIGHT fades distant solids TOWARD the horizon-glow color** (not pale grey):
  `mix(surface, horizonColor, f)` up to ~0.6–0.8 at max distance, drop saturation, collapse internal
  contrast — so receding rows separate into fore/mid/background value groups.
- **Reserve the only near-whites for the actual light sources** (windows, lamps). If walls, sky and
  lights all sit mid-grey it reads flat — let shadow masses go near-black, keep the bright accents rare.

## Color & tone (generic)

Levers for how light VALUES read, independent of the effect. (Reference: `lightning` lab for the
HDR-core + posterize electric look; `night_city` for the value-structure + emissive-split discipline.)

- **Work in HDR — let emitters exceed 1.0; don't pre-clamp the core.** A bright core pushed above 1.0
  is what a bloom/threshold pass has to bloom; clamping flat-white first kills the glow. (lightning:
  HDR core + posterize = the sharp electric read.)
- **Tonemap before output so highlights ROLL OFF instead of clipping** (`x/(1+x)` Reinhard, or ACES) —
  clipped highlights read as dead flat white. Check first whether the engine already sRGB-encodes the
  framebuffer before adding a final gamma, or you double-correct.
- **Bloom = bright-pass threshold → blur → add.** Flicker/flutter rides the bloom, not the body (echoes
  the "light by what it CASTS" lever in Shader craft).
- **Posterize a continuous field into a few bands** to read as sharp/graphic (forks, cel edges) rather
  than a soft gradient.
- **Kill 8-bit banding on smooth dark gradients** (a `pow(1-rd.y, k)` sky bands on a 512 render): add a
  tiny ordered/hash dither `+ (hash(uv) - 0.5)/255.0` just before output.
- **Saturation is a depth/heat cue** — distant/cooler things desaturate; the hottest/nearest stay
  saturated. Hue is the WEAKEST cue; lean on value + saturation + contrast.

## Motion & timing (generic)

What makes motion read ALIVE vs mechanical. (Reference: `lightning` for noise-domain scrolling +
strobe-vs-fps; `fire` for domain-warp turbulence; `boids` for the easing/maneuver lessons.)

- **Shape motion with easing and INCOMMENSURATE rates** (sum sines/noise at non-integer-ratio speeds)
  → organic; linear or single-period motion → mechanical/clockwork.
- **Temporal coherence: animate by scrolling the noise DOMAIN over time** (`noise_uv.y -= t*speed`), NOT
  by toggling alpha/visibility frame to frame — scrolling the field keeps it coherent; toggling pops.
  (Spatial counterpart: "animate the BOUNDARY" in Shader craft.)
- **Frame-rate budget — an event shorter than ~2 frames at render fps is INVISIBLE.** When strobing /
  flashing, check the lit-frame cadence against the render fps (a sub-frame strobe never shows). The
  durable lightning lesson.

The lab's headline deliverable is often a **single document** that animates the effect's construction with
per-step caption text — a "how it's built" reel. This is a reusable PATTERN (seed: the fire & city
labs). Build it as ONE shader, NOT N documents:

- A looping clock `reveal_time() = mod(u_time, PERIOD)`; per step a weight `w_i = smoothstep(start_i,
  start_i + FADE, reveal_time())` that ramps 0→1 then HOLDS (steps STACK). Gate each feature by `w_i`.
- A `u_step` uint uniform: 0 = autoplay, 1..N = freeze/inspect a step. **Tell the user 0 is autoplay**
  — leaving it frozen (e.g. at N) renders the static final scene, a classic "why doesn't it start from
  the beginning" surprise.
- Captions: one codepoint array per step. **ANY `uniform uint name[N]` (N>1, scalar) is auto-encoded
  from a string** — the engine fills it from the string's codepoints (`uniform_coerce.is_text_array`
  keys off the GL SHAPE `uint[N]`, NOT the name; the `…text` suffix is only our naming convention).
  Render via `SB_sd_char`, fade per a per-slot alpha. Glyph set is UPPERCASE latin + digits +
  `()+=*/<>%&':;,.-!?` — transliterate formulas. Caption convention that works: **line 1 = the IDEA,
  line 2 = the core MATH trick** (e.g. `GEOMETRY` / `ID = ROUND(P/S)`). Reference implementation:
  `night_city` learning-reel document (`projects/_lab/night_city/documents/<uuid>/passes/main.frag.glsl`).
- **Entrance ANIMATION, not just a fade**, sells it: scale a building's SDF height by the step weight
  to GROW it from the ground; sweep a per-floor threshold to light windows bottom-up; lerp a rise
  offset; stream props in. Geometry-affecting weights (a grow factor, a rooftop rise) must be set
  BEFORE the march (as a global the SDF reads); shading weights branch in `main()`.
- **Compute & set the total duration for the render.** Sum the step durations + hold = the loop period;
  set the document's `ui_state.render_media_details.duration` (in document.json) to exactly one period so the artifact loops seamlessly
  (verify t=0 ≈ t=period). The final step can be longer than the rest — make step start/duration helper
  functions rather than a single `STEP_DUR` if so.

### The crossfade-switch trap (it WILL bite — read before animating a reveal)

A step that "develops in" (texture, detail) is usually gated by a binary `USE_DETAIL` switch plus a
0→1 crossfade weight. **Two hard rules or it pops in one frame:**
1. **Flip the binary switch at the SLOT START, when the crossfade weight is still 0** — NOT at
   `reveal ≥ 0.5`. If the switch turns on partway through the slot, the weight is already nonzero and
   every detail term jumps to a partial value in a single frame. Gate the switch on `step(0.001,
   weight)`.
2. **EVERY branch inside that switch must scale by the crossfade weight** — including AO, sills,
   curtains, a plinth/early-return, a cornice. An unscaled additive term, a hard bool passed into a
   helper, or an early `return` that swaps the whole pixel will pop even if the main color crossfades.
   Convert bools to eased amounts (`has_curtains ? MIX : 0.0`); lerp early-return paths
   (`mix(normal, special, MIX)`).

**How to FIND a one-frame pop you can't see clearly:** render consecutive frames across the boundary
and measure the **per-pixel max delta / count of changed pixels** frame-to-frame — a real pop shows a
huge spike (thousands of pixels) at one timestamp vs the steady-state flicker baseline. A whole-frame
MEAN hides a localized pop; use max/count.

# ─── CPU-SIM block — skip unless the effect drives uniforms from a Python sim ───
*Most effects (fire / smoke / lightning / abstract) are pure GLSL and never touch the next four
sections. (Exception: `## Selective-variant tuning` is general aesthetic-parameter search — useful for
ANY effect where the agent can't judge the look by eye, not just sims.)*

## CPU scripting (optional)

A document can carry `scripts/script.py` = `class Behavior(ScriptBehavior)` with `update(self, ctx) -> dict`
returning `{uniform_name: value}` to drive uniforms from Python state each frame (the engine hot-reloads
it live, same as the shader). Good for time-varying SCALAR/VECTOR uniforms with persistent state
(flicker amplitude random-walk, a pulsing intensity), AND for a whole per-instance SIMULATION pushed to
the shader as an array uniform (`Array([flat floats])` → `uniform vecN arr[M]`) — e.g. N agents whose
positions a Python sim integrates each frame. NOT for per-pixel work (that stays in GLSL). Reach for it
only when an effect genuinely wants stateful CPU-driven parameters; many effects don't.

- **The `render_document.py` still path and video path BOTH must coerce + tick like the live engine.** A
  script that returns `Array`/`Vec3`/`Text` (not a bare scalar) only works in the headless helper if
  the still path runs `normalize_output` + `coerce_uniform_value` against the live `moderngl.Uniform`,
  and the VIDEO path wires `document.on_pre_render` to tick a fresh `Behavior` per frame (a bare Document has
  no `ProjectSession` to wire it, so without this the sim is frozen at `__init__` state). Both were
  fixed in the boids lab — if a future lab's first array/sim script renders blank or static, check
  these two in `render_document.py` first.
- **A CONSTANT-REGISTER budget caps array uniforms.** The glyph tables already eat ~600/1024 slots, so
  a large `uniform vecN arr[M]` can overflow with `C6020`. Keep M modest (boids ran fine at N=40);
  if you need text captions AND a big array in the same shader, that's the collision to watch.

## Specific-algorithm sims — research the real thing, don't hand-tune from intuition

When the CPU sim implements a NAMED algorithm (flocking/boids, a physics integrator, a cellular
automaton, an L-system, SPH, …), your invented parameters will be wrong and you'll burn rounds chasing
them. (The boids lab: many rounds of guessed weights gave a frozen ball, then a comet line, before a
deep-research pass on Reynolds/Shiffman/the starling papers gave the canonical structure that fixed it
in one shot.) The protocol:

- **Get the canonical STRUCTURE + parameter REGIME from real sources** (the algorithm's primary paper,
  a reference implementation, a respected tutorial), not from memory. Transfer the RATIOS and the
  structure, not raw numbers tuned for someone else's world scale.
- **If the maintainer has their OWN implementation (a repo, a past project), port THAT** — it's already
  debugged to their taste. (The boids lab ported the maintainer's `boids_demo` C/raylib repo; its
  alignment-as-capped-rotation + damping-drag + min-speed-floor was the robust recipe intuition missed.)
- **A CPU mirror of a shader function MUST be verified numerically EQUAL to the GLSL, including on
  negatives/edges — never assumed.** A "looks equivalent" builtin can silently diverge: `np.modf(x)[0]`
  keeps the sign while GLSL `fract` doesn't, so on negative inputs the Python city heights diverged
  from the drawn ones by ~38 units and the agents navigated a phantom world ("flying through
  buildings"). Cross-check by sampling BOTH implementations over the real input domain and asserting
  max-diff ≈ 0. A self-consistency check against your own mirror proves nothing — the oracle must be
  the OTHER implementation (the shader).
- **Per-agent obstacle avoidance is anti-cohesive for a group that must read as ONE cluster** — each
  agent dodges its own way and the group shatters/stretches. Either avoid as a GROUP (centroid-based)
  or route the group where avoidance rarely fires (the boids ended up cruising ABOVE the rooftops).

## Selective-variant tuning — when the agent CAN'T judge the look, let the maintainer pick

The agent's read of a rendered image (and especially of MOTION) is unreliable, and numeric proxies
(elongation, alignment, spread) only weakly predict "looks good". When a parameter is AESTHETIC and you
catch yourself guessing-and-re-rendering, STOP and run a **side-by-side selection loop** instead — this
was the one part of the boids lab that worked cleanly:

1. Render **2×2 (or N) variants in ONE frame/clip**, each a different parameter set, labelled. (Cheap
   way for a sim: a small pure-Python/numpy splat renderer drawing all N grids into one MP4 — you don't
   even need the shader for a motion-shape comparison.)
2. The maintainer picks the best **by eye** (and says which to drop).
3. **Mutate the winner** into a new N variants (vary one or two axes each; keep one near-copy as a
   control), render the grid again, repeat. A few rounds converge fast.
4. Crossbreed survivors when several are "good" (average / mix their genes), and only THEN graduate the
   chosen params into the full scene.

This beats both blind agent tuning and a single-number objective for anything where the target is "looks
alive / looks right". Treat it as the default move for aesthetic-parameter search.

## Verify the premise before patching (and verify WHICH part is broken)

A recurring time-sink this lab: a symptom gets attributed to the wrong cause and you optimize/patch the
wrong thing. Discipline:

- **A "still broken / it's worse" report is a signal your MODEL is wrong — re-derive from real data, do
  not defend or re-guess.** When the maintainer names a likely cause ("maybe it's the heights?"), test
  THAT first — they were right and it was a numeric mirror bug.
- **Confirm WHICH function/stage is the problem before fixing it.** A "20 update() calls didn't finish
  in 90s" was read as "update() is slow" and the wrong code was optimized for a long time — the process
  was actually hung in `__init__` (an infinite rejection-sampling loop). The cheap decisive measurement
  (time `__init__` in isolation) would have localized it in seconds.
- **One constant doing two unrelated jobs is a trap.** A clearance value reused as both an in-flight
  margin (wants to be large) and a spawn-acceptance gate (must be small/satisfiable) hung the
  constructor forever when widened. Give each job its own named constant; bound any rejection-sampling
  loop + assert its post-condition (never a bare `while` on a possibly-unsatisfiable predicate).

---

## End of session — extract the value

The experiment's *output* is knowledge, not just a pretty document. At the end:
- Make sure NOTES.md is complete (every version, every user verdict, every source).
- Propose what to PROMOTE: a reusable technique → an `SB_*` lib helper; a recurring need → a ShaderBox
  feature or a copilot instruction; a finding → `ai_docs/conventions.md` or a feature spec.
  - **When promoting into the copilot, strip everything specific to a lab session.** The copilot runs
    INSIDE the ShaderBox app on the user's own machine: the rendered preview is always on the user's
    screen, so the user does the visual judging through the app — there is no "render a file and send
    it" / "compare clips" step, no external delivery channel. Promote only the GRAPHICS CRAFT
    (shader/SDF/lighting/color/motion levers) and generic engineering discipline; drop the lab's
    orchestration (offscreen rendering, file delivery, session-versioning, NOTES bookkeeping).
- Ask the user whether to **preserve** the lab project (promote a document dir out of `projects/_lab/`
  into a real project + `git add`) or let it stay gitignored/disposable.
- **Add/update this lab's bullet in `## Past labs`** so the next run can find it (the reference map
  must stay current — a missing or stale entry is why a technique gets re-derived). A worthwhile lab
  becomes a real reference only if it's COMMITTED (`git add -f projects/_lab/<slug>` — the dir is
  gitignored; mirror what `night_city`/`fire` did) so it travels to every machine.

---

## Past labs — the technique REFERENCE MAP (the code lives here, not in this skill)

A lab is a kept, working REFERENCE: when you need a technique, open the lab that already solved it —
read its `NOTES.md` for the evolution + verdicts, then open the cited `documents/<id>/passes/*.frag.glsl`
for the real code. This is WHY labs are saved; do not re-derive or paste snippets when a reference
exists. When the user references "the X lab", it's `projects/_lab/X/`. (COMMITTED labs travel via git
and are reliable references on any machine; LOCAL-ONLY labs exist only where they were made.)

**Need it? → go here:**
- **Turbulent flame / fire / organic rising heat / smoke** → **`fire`** (committed). Domain-warp
  turbulence, fuel-envelope + eroded silhouette, blackbody/temperature color ramp, flicker-on-the-
  cast-glow (not on the body), volumetric-ish smoke, and the **timed-reveal document** pattern (its seed).
  NOTES: `projects/_lab/fire/NOTES.md`. Matured flame: `documents/<fire v09…>/shader.frag.glsl`; domain-warp
  + CPU-script wind: `documents/<fire v03…>/`; reveal reel: `documents/<fire timed reveal>/` (NOTES lists ids).
- **SDF box city / bounded tower repetition / painted facade detail / NIGHT CITY LIGHTING** →
  **`night_city`** (committed). Bounded domain-repetition w/ neighbor-min, the directional night-key
  (sky-dome+moon, surface keyed, emissive split — the FORM-giver), facade unwrap (floors×bays, AO,
  sills, pilasters, plinth, cornice), night aerial fog, ground-glow cars, the learning reel. NOTES:
  `projects/_lab/night_city/NOTES.md` (v06 entry = the full lighting recipe); canonical shader:
  `documents/v06-depth/shader.frag.glsl`.
- **3D agent sim / flocking / CPU-driven particle positions** → **`boids`** (LOCAL-ONLY, NOT promoted —
  the maintainer rejected the motion). Use it for the PROCESS lessons (now in the generic sections
  above): selective-variant tuning, research/port a named algorithm, the CPU↔GPU numeric-mirror check,
  per-agent avoidance is anti-cohesive, the array-uniform / `render_document.py` ticking gotchas. NOTES:
  `projects/_lab/boids/NOTES.md`.
- **2D lightning / electric bolts / branching energy** → **`lightning`** (LOCAL-ONLY). `1/dist`
  ridge-glow, HDR-core+posterize for an electric read, segment-polyline + branch geometry, the strobe-
  cadence-vs-fps lesson. NOTES: `projects/_lab/lightning/NOTES.md`.
- **Real cloth sim + PBR sunset scene (the "realism meta-lever" case study)** → **`us_flag`**
  (LOCAL-ONLY unless committed). The arc IS the lesson: v01–v05 hand-rolled fakes (sine cloth, ad-hoc
  sheen, gradient sky, plastic finial) each rejected as cheap; v06–v18 replaced each with the real model.
  What it demonstrates, by document:
  - **Script-driven Verlet mass-spring cloth** pushed to the shader as a mesh (per-vertex screen pos +
    normal as `vec3[NV]` array uniforms; shader rasterizes the mesh per-pixel via barycentric + z-test →
    real self-occlusion). Valence-normalized Jacobi constraint solve; SOFT bend / stiff structural to keep
    folds without rubber-stretch. `v14` script.
  - **AEROELASTIC wind**: `F = K_AERO*(n·vrel)*n + K_DRAG*vrel`, `vrel = wind - cloth_velocity` → emergent
    non-periodic flutter (the imposed-sine versions v04/v10 read fake). Gaussian gusts. `v14` script.
  - **CPU self-shadow**: ray from each vertex to the sun vs the mesh triangles (Möller-Trumbore, jittered
    for penumbra) → `u_shadow[NV]`; grazing light. The depth cue that killed the flat look.
  - **Cloth BRDF**: Charlie sheen (Sony Imageworks) + Ashikhmin visibility + subsurface wrap diffuse;
    deep saturated albedo + structured plain-weave (not color mottle). `v13`.
  - **Analytic scattering sky**: Rayleigh+Mie+ozone single-scatter + transmittance → the sunset gradient
    emerges from physics; PERSPECTIVE-projected multi-octave lit cloudscape; sun occluded by cloud cover.
    `v11`/`v16`/`v18`.
  - **Raymarched 3D SDF pole** (capsule+sphere+collar) in the SAME camera as the mesh, metal shaded by
    ENV REFLECTION, SDF soft shadow, depth-composited with the flag; screen-cull the march to the pole
    strip. `v17`.
  - Generic fixes also here: non-square proportion (`h_uv = w_uv*aspect/R`), ACES tonemap, hash dither,
    lens flare, aerial perspective. NOTES: `projects/_lab/us_flag/NOTES.md` (full v01→v18 evolution +
    every source URL); canonical shader/script: `documents/v18-suncloud/` + `documents/v14-shadow/scripts/`.

## Follow-ups (NOT ready — don't build these mid-session; capture the idea)

- **Generalise the render helper into the app.** `render_document.py` is the lab's tool; a built-in
  "render this document to MP4 at size/duration" command in ShaderBox proper would remove the script.
- **A built-in N-up variant grid render.** The selective-variant loop above was hand-rolled as a
  throwaway numpy splat script each round. A reusable "render these K param-sets into one labelled grid
  MP4" helper would make the selection loop a first-class lab tool. (Promoted from a prior follow-up
  after the boids lab proved the loop's value.)
- **Group-coherent obstacle avoidance.** Per-agent SDF avoidance shatters a flock; a centroid-based or
  shared-steering avoidance that keeps the group cohesive while dodging is unsolved — deferred (the
  boids lab sidestepped it by flying above the obstacles).
