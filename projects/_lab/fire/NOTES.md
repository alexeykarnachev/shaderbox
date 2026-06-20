# fire — shader lab notes

- **Effect:** procedural fire (raw GLSL fragment shader, no textures).
- **Started:** 2026-06-15 (ad-hoc, in `projects/dev/`); migrated to a proper lab session 2026-06-16.
- **Environment:** desktop — user watches the previews in ShaderBox; new versions appear as new grid
  nodes (per-frame `nodes/` disk-sync). Stills rendered only for the agent's own eyeballing.
- **Stop cadence:** once at the end (iterate freely, show the final result).

**Origin:** not a clean lab — it grew from a bad real copilot trace. A `gpt-5.3-codex` copilot was
asked to build a step-by-step "fire tutorial" from the UV Mango template; the result was poor (and
spawned the 16-edit duplicate-comment spiral that became feature 050). The user asked the agent to
**manually play the copilot's role** — edit the shader directly, WITH eyes — to find the achievable
ceiling and isolate render-blindness from model skill. The findings seeded the `shader-lab` skill.

**Teachable progression intended:** base gradient → warped-noise field → flame shape → color ramp →
glow → embers → smoke → (optional) CPU-script behaviour.

**Layout:** project-native — each version is its OWN node (its own grid cell), so the grid IS the
version history; they sit side-by-side to A/B directly, nothing to swap. A new step = a new node;
existing nodes are never overwritten. Node ids per version are listed in each entry below. There is
also a `fire-lab (live)` scratch node (= v04 right now) used as the active iteration target.

---

## v01 — codex (the blind copilot baseline)
- Node: `fire v01 (codex)` — `nodes/0d886c5e-7761-47fb-82a0-9b6d1bdaf3ea/` (preserved verbatim — NOT authored in this lab).
- What it is: the copilot's own attempt. Shape from an **`SB_sd_circle` union mask** (body + mid +
  tip, `SB_op_smooth_union`), a flat 2-colour cool→hot `mix`, ember field as a static cellular grid
  of dots, `SB_tri_wave` flicker, step-reveal uniforms (`u_show_shape`/`u_show_glow`/`u_show_embers`/
  `u_step_mix`). Compiles fine — the helpers (`SB_fbm`, `SB_tri_wave`, `SB_hash3`) are real, in the
  user lib `noise.glsl`.
- User verdict: **"this is not a fire, this is a pile of crap."** (rounded light-bulb/pylon blob with
  floating dots; no upward tongues, static embers.)

## v02 — hand, pre-research (soft gradient)
- Node: `fire v02 (gradient)` — `nodes/6850b994-6f41-42ed-a494-6264002290eb/`.
- Change: from scratch — `SB_center_uv` flame space (y: 0 bottom → 2 top); a **single `SB_fbm`**
  upward-scrolling field; a Gaussian column envelope (`exp(-(x/width)^2)`, width narrows with height);
  noisy height threshold for a licking top; blackbody-ish 4-stop temperature ramp (red→orange→yellow→
  white-hot core).
- Source: none yet (pre-research).
- User verdict: **"a little bit better, but the shape still looks very very bad… very boring, no
  sparks, no smoke, no flickering."**

## v03 — hand, post-research (domain-warp + fuel envelope + glow + embers + smoke + CPU script)
- Node: `fire v03 (warp+wind)` — `nodes/61995135-faf1-4176-a323-750a0bfb31cf/` (has `scripts/script.py`).
- Change (the research-driven rebuild, built as steps 1–6):
  1. **iq domain warping** — `fbm(p + 3·fbm(p + 3·fbm(p)))` instead of one fbm lookup → curling
     tongues, not soft clouds. The single biggest improvement.
  2. Shape carved from a **noise-thresholded fuel envelope** (column × vertical fuel, burn =
     `smoothstep` of `field·fuel·column`), tongues tied to the column so they stay attached (no
     free-floating islands). NOT an SDF mask.
  3. Over-scaled temperature ramp (hot core blows to white, edges stay red).
  4. Additive emissive **glow/halo** + tight core bloom.
  5. **Embers** done "properly": sparse cellular sparks, vertically streaked, funnelled into a
     narrowing cone above the body, advecting upward faster than the flame (vs codex's static grid).
  6. Self-lit **smoke** wisps above the body (dark-on-black is invisible, so it emits a little).
  - **CPU script** (`script.py`): per-frame state drives `u_intensity` (breathe), `u_flicker` (fast
    jitter, random-walk amplitude), `u_wind` (gusting horizontal lean via smoothed random walk).
- Sources:
  - iq — domain warping https://iquilezles.org/articles/warp/
  - willdoenlen — realistic fire shader https://www.willdoenlen.com/blog/realistic-fire-shader
  - greentec — Shadertoy fire https://greentec.github.io/shadertoy-fire-shader-en/
  - WebGL fire shader (fbm) https://blog.fixermark.com/posts/2025/webgl-fire-shader-based-on-fbm/
  - vfxapprentice — fire VFX properties https://www.vfxapprentice.com/blog/everything-know-about-fire-fx
- User verdict: **"this one is better, but I still strongly dislike it."** Specifics:
  - "the shape still looks like a triangle, especially on the preview."
  - "this 'wind' stuff looks cringy: it makes the flame diagonal, this can not be. Flame should be
    vertical. Also this wind changes direction in strange discrete periods." → **drop wind + scripting.**
  - "smoke is barely visible, it is absolutely flat and not interesting."
  - "flame is completely flat and boring inside… interesting gradients near the edges, but the body
    is completely flat."

## v04 — hand, teardrop rebuild (current live)
- Node: `fire v04 (teardrop)` — `nodes/c2f7c748-086d-4d18-bd1a-1a1d0ef5a3ec/` (= the `fire-lab (live)` scratch node).
- Change (addressing each v03 complaint; no wind, no script):
  - **Teardrop body** via a curved width profile — wide flat base, `sin`-curve belly bulge in the
    lower third, convex taper to the tip — NOT a width-narrows cone (a straight ramp = triangle).
  - **Interior structure** — heat = teardrop body with the multi-scale warped noise (coarse body
    motion + fine crackle) showing THROUGH; the colour ramp maps the *raw* field so internal hot/cool
    veins survive instead of clipping to a flat white slab.
  - **Stronger top edge perturbation** (height-weighted) for licking tips.
  - **Tip frays** — the upper flame dissolves into the noise so it doesn't end in a clean geometric
    needle. (Only partially solved — see below.)
  - Kept domain warp + glow + sparse embers. Dropped smoke (didn't read on black) and wind/script.
- Sources:
  - Cyanilux — fire shader breakdown (Y-remap teardrop, posterize banding, distort vertical-only)
    https://www.cyanilux.com/tutorials/fire-shader-breakdown/
  - libretro xt9s-flame.glsl (squashed-sphere body + height-weighted noise)
    https://github.com/libretro/glsl-shaders/blob/master/procedural/xt9s-flame.glsl
  - clockworkchilli — GLSL fire shader (multi-layer noise, top-weighted perturb)
    https://clockworkchilli.com/blog/8_a_fire_shader_in_glsl_for_your_webgl_games
- User verdict: experiment paused after v03's feedback; **v04 not yet reviewed by the user this
  session.** Open complaints carried from v03 likely still apply (interior, tip). User has "further
  ideas" to drive the next iterations.

---

## Open threads / agent's reserve techniques (NOT user verdicts — candidate next moves)
- Needle tip only partially broken up by the noise-dissolve.
- Posterize/colour-temperature banding (Cyanilux) — not yet applied; candidate for the "flat boring
  body" complaint.
- Volumetric / raymarched smoke density (vs the flat self-lit wisp that didn't read).
- Secondary detached flame licks; a base "ground" glow.

## Promotion candidates (decide at session end)
- Reusable `SB_*` helpers: a domain-warp turbulence (`SB_fbm`-based), a teardrop/flame width profile,
  a blackbody temperature ramp. (Fire findings already referenced from `ai_docs/features/050_*`.)
- The render-blindness finding (copilot can't tell a flame from a blob) — already drove feature 050
  and this skill.
