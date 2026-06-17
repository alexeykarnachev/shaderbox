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

---

## The ONE rule that matters most

**Confirm the FORM before you build. NEVER overwrite a previous version. NEVER write your own visual
conclusions into the notes.**

The costly mistakes from past sessions:
- **Restate the structure the user described, in your own words, BEFORE generating anything — a wrong
  mental model multiplies into N wrong artifacts.** (Asked for "one shader where steps reveal over
  time", an entire turn was burnt building NINE separate nodes — the decomposition was right, the FORM
  was wrong.) When the user describes a shape ("a single X that does Y over time", "steps that stack"),
  say it back and get a nod before writing files.
- Overwriting a shader in place destroyed earlier versions — we couldn't go back, compare, or watch
  the evolution. **Each step is a NEW node** (project-native: a new `nodes/<id>/` grid cell). You
  author a fresh node per version and never touch an existing one. (Versioning section below.)
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

1. **Effect + concept.** What are we building (fire / lightning / smoke / …)? Restate the teachable
   progression you intend (e.g. fire: base gradient → warped-noise field → flame shape → color ramp
   → glow → embers → smoke).

2. **Environment — desktop vs offscreen. Infer from context, then CONFIRM.**
   - **Desktop** (the user is at the machine running ShaderBox): they open the project and watch the
     preview update live as you rewrite the shader. You do NOT render anything to show them — the app
     does it. (You still render stills for YOUR OWN eyeballing.)
   - **Offscreen / headless** (the user is on an iPad / a phone / a Pi over SSH — no app, no display;
     they said so, or the session is clearly remote): you render a **~10s MP4** (NOT WebM — iPad won't
     play it) of each result you want to show and send it to the chat. Same iteration loop, no live app.
   - When unsure which, ASK: "are you watching live in ShaderBox, or should I render MP4s to chat?"

3. **Stop cadence.** Ask up front: "how often should I stop for your review?" — options: after every
   step / once at the very end / at my discretion (stop only at a real fork or when I need a verdict).
   Honour it for the whole session unless the user changes it.

---

## The project lives in `projects/_lab/<name>/` (gitignored)

- Create one fresh ShaderBox project **per experiment** under `projects/_lab/<slug>/` — it's a
  collection of nodes (each node = a `node.json` + `shader.frag.glsl` + optional `scripts/script.py`).
- `projects/_lab/` is **gitignored** (`.gitignore`), so nothing here pollutes the repo. A worthwhile
  result is promoted **by hand at the very end** (move the node dir out of `_lab/` into a real project
  + `git add`) — only if the user decides to keep it.
- Tell the user the path and to **open that project in ShaderBox** (File → Open project → the
  `projects/_lab/<slug>/` folder), THEN you begin iterating, so the app has the node loaded.

### Project / node layout to write

A minimal node dir (copy the shape from any `projects/dev/nodes/<id>/node.json` — `ui_state.ui_name`
is the grid label, `canvas_size` the preview size). A node needs `node.json` + `shader.frag.glsl`;
`scripts/script.py` is optional (CPU-driven uniforms). `SB_*` helpers resolve from the live shader
lib automatically.

---

## The live-preview contract (desktop) — why it Just Works, and the one requirement

ShaderBox's per-frame mtime watcher (`watch.py::reload_node_if_changed`, called for EVERY loaded node
in `ui.py::update_and_draw`) **recompiles a node the instant its `shader.frag.glsl` mtime changes on
disk**, and `reload_scripts()` does the same for `script.py`. AND ShaderBox reconciles `nodes/` to
disk every frame (disk is the source of truth), so a node dir you CREATE / DELETE on disk syncs into
the grid AUTOMATICALLY — no reload, no bookkeeping. Together that means:

- **A new version = a new node dir** → it just appears in the grid as a fresh cell, live.
- **Editing a node's shader** → that cell recompiles in place.

So the flow is: user opens the project once; thereafter you write a new `nodes/<id>/` per step and it
shows up on its own. The only requirement is that the project was opened (so the app is watching that
dir); everything after is automatic.

**The user will NOT edit the shader during iterations** (they only watch / may view the code / may
have the editor focused). So there is no edit-collision to fear — you own the files, they observe.

---

## Versioning — one NODE per step, never overwrite

Use the project's NATIVE layout — no parallel snapshot dir, no scaffolding. **Each step is its own
node** under `nodes/`; the grid IS the version history. To preserve a version you simply leave its
node alone; to make a new one you author a fresh node dir. You never overwrite an existing node, so
nothing is ever lost.

```
projects/_lab/<slug>/
  nodes/<id-1>/shader.frag.glsl   # v01 — its own grid cell, named "<effect> v01 (<short-name>)"
  nodes/<id-2>/shader.frag.glsl   # v02
  nodes/<id-3>/shader.frag.glsl   # v03
  nodes/<id-3>/scripts/script.py  # if that version had a script
  ...
  NOTES.md                        # the experiment log (see below)
```

**Each step:** (1) create a NEW `nodes/<uuid>/` (copy the node.json shape from an existing node;
set `ui_state.ui_name` to `"<effect> vNN (<short-name>)"` so the grid reads as a progression);
(2) write its `shader.frag.glsl` (+ `scripts/script.py` if any) — the new cell appears in the grid
automatically (per-frame `nodes/` sync); (3) append a NOTES.md entry. To let the user **compare**,
all versions are already side-by-side as their own grid cells — no swapping needed.

This is what makes "I like v3's tip but v5's color, take both" possible — every version is still a
live node to diff and cherry-pick from.

**Caveat — the app can re-save a node's files on exit/edit.** A node that's currently selected/edited
in the app may get its `node.json`/shader rewritten from the app's in-memory state on shutdown. You
own the files between sessions, so just be aware: if you hand-edit a node dir while the app holds it,
reconcile (the app's per-frame disk-sync usually wins for an unselected node, but don't fight a node
the user has open). In practice: author NEW nodes for new steps (never contended) and don't hand-edit
a node the user is actively viewing.

---

## NOTES.md — the experiment encyclopedia

A per-project log so we can see *exactly what changed and why* across the evolution. One entry per
version. **Record only verifiable inputs, NEVER your own visual judgments:**

- ✅ **The user's verdict** (verbatim or close: "still looks like a triangle", "wind is cringe, drop it").
- ✅ **Web findings / techniques / formulas** you researched, WITH the source URL (domain warping, the
  teardrop width profile, the temperature ramp, etc.).
- ✅ **What changed** mechanically in this version (the diff in plain words).
- ❌ **NOT** "this looks like a great flame now" / "the interior is nicely structured" — your read of a
  rendered image is unreliable; leave the aesthetic verdict to the user.

Entry shape:

```markdown
## v03 — teardrop body
- Change: replaced the width-narrows cone with a curved teardrop profile (sin-belly + convex taper).
- Source: Cyanilux fire breakdown (Y-remap teardrop) https://www.cyanilux.com/tutorials/fire-shader-breakdown/
- User verdict: <what the user said, or "pending review">
```

Start NOTES.md with a header: effect name, date, environment (desktop/offscreen), stop cadence.

---

## Rendering (for YOUR eyeballing, and for offscreen delivery)

`.claude/skills/shader-lab/render_node.py` — headless EGL render, no app needed:

```bash
# still PNG at time t — for YOU to eyeball a result before/while iterating:
uv run python .claude/skills/shader-lab/render_node.py image <node_dir> <out.png> --t 0.5 --size 512

# MP4 clip — the OFFSCREEN deliverable (mandatory .mp4; WebM won't play on iPad):
uv run python .claude/skills/shader-lab/render_node.py video <node_dir> <out.mp4> --seconds 10 --fps 30 --size 512
```

- Renders against the live `SB_*` lib; video goes through `Node.render_media` (so a `script.py` ticks
  a fresh per-export instance — export-isolation, same as a real export).
- **`render_node.py` forces a SQUARE canvas** (`--size` → size×size). For a non-square aspect (e.g.
  9:16 portrait) it distorts — write a small inline render with the real `canvas.set_size((w, h))`
  instead (the snippet in `dev_flow.md ## Recipes > Authoring / debugging nodes directly`). Cropping a
  region of the render (PIL `.crop`) to eyeball one area (a corner seam, a rooftop, a street) is the
  fastest way to inspect a localized issue.
- **Eyeball every step yourself** (the copilot's core handicap is render-blindness — you, here, are
  NOT blind, so use it). But per the ONE rule, your eyeball informs your NEXT iteration; it does NOT
  go into NOTES.md as a verdict.
- **Offscreen mode:** render the MP4 and send it to the user's chat (use the `mytools-tg` skill to
  send, or hand them the path if that's the channel). Intermediate clips at review stops, a final clip
  at the end.

---

## Researching techniques

When an effect needs a technique you don't have cold (how real fire avoids a flat core, a teardrop
silhouette, lightning branching, how a night cityscape gets depth, etc.), **WebSearch / WebFetch real
sources** — Shadertoy breakdowns, iquilezles.org, Cyanilux, game-engine VFX writeups, and for
art-direction questions (value/contrast/colour) digital-painting + concept-art sources (Mitch Albala,
ctrl+paint, Marco Bucci, art blogs). Working example code is the canonical pattern; read it, find the
divergence from your code. Put the technique + URL in NOTES.md.

**Check past labs before researching from scratch** — a prior session may already carry the
technique, the maintainer's aesthetic verdicts, and the dead-ends. See `## Past labs` below; read the
relevant `projects/_lab/<slug>/NOTES.md` first.

---

## Shader craft — making an effect READ right (general, not effect-specific)

Hard-won levers that apply to fire, smoke, water, lightning, energy — anything organic. Generalized
from real maintainer corrections; the parenthetical is the evidence, not a prescription to copy.

- **Shape from a thresholded/eroded NOISE field, not an SDF-primitive mask.** A circle/ellipse mask
  reads as a static blob; an organic silhouette should emerge from the noise itself — gate a soft
  envelope by the field and erode the edge. (A circle-union flame looked like a balloon; the
  noise-gated one read as flame.)
- **Domain-warp the noise for organic motion:** `fbm(p + k·fbm(p + k·fbm(p)))` curls where plain
  `fbm` only clouds — the single biggest "reads as alive" lever for fluid/fire/smoke.
- **For interior detail, feed the RAW continuous field to the colour ramp — don't threshold-then-fill.**
  A binary burn mask × a flat colour = a flat slab; mapping the raw field through the ramp keeps the
  internal veins/structure. (A whole iteration came out flat-yellow because the field was thresholded
  into a mask, then filled.)
- **Animate the BOUNDARY, not just the interior.** If the envelope is a static function of position
  and only the texture inside scrolls, it reads as "static shape with stuff moving inside" — displace
  the silhouette's sample point by a flow field so the whole outline licks/sways.
- **Light a scene by what it CASTS, not by recolouring the emitter.** A flame's flicker belongs on the
  glow it throws into the scene (a wide, scene-covering, sharp-falloff radial bloom), not on the flame
  body's colour. (Maintainer: "the light spreads, it doesn't work like this" — the flicker was wrongly
  on the body.)
- **Self-illuminate anything meant to read on BLACK.** Smoke/haze over black has nothing to occlude →
  invisible unless it emits a little. (And if it still won't read after a fair attempt, cutting it is a
  legitimate call — don't ship an invisible feature.)
- **Expose tunables as uniforms EARLY and hand tuning to the user.** When the right value is aesthetic,
  STOP guess-rendering — add inline-default uniforms (auto-generates ImGui controls; `uint`/`int` → an
  integer slider, `float` → a drag), let the user dial them live, then bake their chosen values back as
  the new defaults. (Several blind tune-render cycles evaporated the moment the knobs were exposed and
  the maintainer dialed it in one pass.) For the engine mechanics of node files / headless render /
  compile-check / aspect, see `dev_flow.md ## Recipes > Authoring / debugging nodes directly`.

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

## Lighting & art-direction (generic; the night-scene set is concrete)

When a render reads FLAT, the cause is almost always a value/contrast problem, not a missing feature.
General rule: **work the VALUE structure first** (large dark masses + a few bright accents + readable
fore/mid/background value groups); colour and detail come after. Hue is the *weakest* depth cue —
lean on value + saturation + contrast.

**Night-scene rules (web-researched, broadly reusable; sources in `night_city/NOTES.md` v06):**
- **At night the SKY is the brightest large value; solids read as dark silhouettes against it.** This
  INVERTS daytime aerial perspective. Sky = vertical gradient: warm skyglow hugging the HORIZON
  (`pow(1 - rd.y, ~2.5)`) fading to a cool dark ZENITH. Horizon ≈ 8–12× a near unlit surface.
- **Even with no sun, light faces differently for FORM.** The sky-dome + moon act as a soft key: the
  face toward the sky/moon catches more (cooler) light, the away face is darker/warmer. Keep it
  SUBTLE (~1.3–1.6× value spread between faces) — night contrast is gentle. Apply to the SURFACE term
  only; split out emissive (lit windows/lights) so the key doesn't dim them.
- **Aerial perspective AT NIGHT fades distant solids TOWARD the horizon-glow colour** (not pale grey):
  `mix(surface, horizonColor, f)` up to ~0.6–0.8 at max distance, drop saturation, collapse internal
  contrast — so receding rows separate into fore/mid/background value groups.
- **Reserve the only near-whites for the actual light sources** (windows, lamps). If walls, sky and
  lights all sit mid-grey it reads flat — let shadow masses go near-black, keep the bright accents rare.

## Step-reveal animation (the learning-deliverable pattern)

The lab's headline deliverable is often a **single node** that animates the effect's construction with
per-step caption text — a "how it's built" reel. This is a reusable PATTERN (seed: the fire & city
labs). Build it as ONE shader, NOT N nodes:

- A looping clock `reveal_time() = mod(u_time, PERIOD)`; per step a weight `w_i = smoothstep(start_i,
  start_i + FADE, reveal_time())` that ramps 0→1 then HOLDS (steps STACK). Gate each feature by `w_i`.
- A `u_step` uint uniform: 0 = autoplay, 1..N = freeze/inspect a step. **Tell the user 0 is autoplay**
  — leaving it frozen (e.g. at N) renders the static final scene, a classic "why doesn't it start from
  the beginning" surprise.
- Captions: one `uint u_stepNtext[LEN]` codepoint array per step (engine encodes a typed string when
  the uniform name ENDS in `text`), rendered via `SB_sd_char`, faded per a per-slot alpha. Glyph set is
  UPPERCASE latin + digits + `()+=*/<>%&':;,.-!?` — transliterate formulas. Caption convention that
  works: **line 1 = the IDEA, line 2 = the core MATH trick** (e.g. `GEOMETRY` / `ID = ROUND(P/S)`).
- **Entrance ANIMATION, not just a fade**, sells it: scale a building's SDF height by the step weight
  to GROW it from the ground; sweep a per-floor threshold to light windows bottom-up; lerp a rise
  offset; stream props in. Geometry-affecting weights (a grow factor, a rooftop rise) must be set
  BEFORE the march (as a global the SDF reads); shading weights branch in `main()`.
- **Compute & set the total duration for the render.** Sum the step durations + hold = the loop period;
  set the node's `render_media_details.duration` to exactly one period so the artifact loops seamlessly
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
   helper, or an early `return` that swaps the whole pixel will pop even if the main colour crossfades.
   Convert bools to eased amounts (`has_curtains ? MIX : 0.0`); lerp early-return paths
   (`mix(normal, special, MIX)`).

**How to FIND a one-frame pop you can't see clearly:** render consecutive frames across the boundary
and measure the **per-pixel max delta / count of changed pixels** frame-to-frame — a real pop shows a
huge spike (thousands of pixels) at one timestamp vs the steady-state flicker baseline. A whole-frame
MEAN hides a localized pop; use max/count.

## CPU scripting (optional)

A node can carry `scripts/script.py` = `class Behavior(ScriptBehavior)` with `update(self, ctx) -> dict`
returning `{uniform_name: value}` to drive uniforms from Python state each frame (the engine hot-reloads
it live, same as the shader). Good for time-varying SCALAR/VECTOR uniforms with persistent state
(flicker amplitude random-walk, a pulsing intensity) — NOT for per-pixel work (that stays in GLSL).
Reach for it only when an effect genuinely wants stateful CPU-driven parameters; many effects don't.

---

## End of session — extract the value

The experiment's *output* is knowledge, not just a pretty node. At the end:
- Make sure NOTES.md is complete (every version, every user verdict, every source).
- Propose what to PROMOTE: a reusable technique → an `SB_*` lib helper; a recurring need → a ShaderBox
  feature or a copilot instruction; a finding → `ai_docs/conventions.md` or a feature spec.
- Ask the user whether to **preserve** the lab project (promote a node dir out of `projects/_lab/`
  into a real project + `git add`) or let it stay gitignored/disposable.

---

## Past labs (read the NOTES before re-deriving anything)

Each lab leaves a `projects/_lab/<slug>/NOTES.md` — the full evolution, the maintainer's verdicts, the
dead-ends, and the reusable techniques. When a new effect overlaps a past one, READ that NOTES first
(it's the cheapest source — better than web). Gitignored labs are disposable; a promoted one keeps its
NOTES. When the user references "the X project / lab", it's `projects/_lab/X/`.

- **`fire`** — raymarched-free 2D flame: domain-warp turbulence `fbm(p + k·fbm(p + k·fbm(p)))`,
  teardrop width profile, blackbody temp ramp, flicker-on-the-cast-glow, the **timed-reveal node**
  pattern (seed). NOTES: `projects/_lab/fire/NOTES.md`.
- **`night_city`** — SDF-raymarched night cityscape (committed): bounded domain-repetition w/
  neighbor-min, ground-glow cars w/ right-hand lanes, painted facade detail, rooftop-clutter geometry,
  the night-lighting art-rule set, per-step entrance animations, and the matured single-node learning
  reel. Everything in the Raymarching/SDF, Lighting, and Step-reveal sections above was mined here.
  NOTES: `projects/_lab/night_city/NOTES.md`.

## Follow-ups (NOT ready — don't build these mid-session; capture the idea)

- **Variant nodes → visual pick → aggregate.** At a fork, generate several variants of one idea as
  parallel sibling nodes (not just a linear chain), let the user pick visually ("take the tip from
  this one, the color from that one"), then aggregate the chosen parts into a new node and delete the
  rejected variants. A branching compare/merge flow on top of the linear node progression.
- **Generalise the render helper into the app.** `render_node.py` is the lab's tool; a built-in
  "render this node to MP4 at size/duration" command in ShaderBox proper would remove the script.
