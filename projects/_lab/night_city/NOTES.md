# night_city — shader lab log

- **Effect:** raymarched night building / city (amber-on-black, CRT-liminal mood).
- **Date:** 2026-06-17
- **Environment:** desktop (live preview in ShaderBox).
- **Stop cadence:** at the end (agent iterates freely; polish into teaching progression later).
- **Source of truth:** the maintainer's shadertoy night-building shader (inspiration + code samples,
  NOT a straight port). Pasted in the session transcript.
- **Aspect:** YouTube Shorts 9:16 (`canvas_size [720,1280]`).

Each version is a NEW node under `nodes/<slug>/` — the grid IS the version history. NOTES record only
the mechanical change, web sources w/ URL, and the maintainer's verdict — never the agent's own
visual judgment.

---

## v01 — faithful port (`city v01 (port)`)
- Change: ported the shadertoy raymarcher to ShaderBox conventions — `mainImage/fragColor/iTime/
  iResolution` → `main()/fs_color/u_time/u_aspect`, screen pos from `vs_uv`, `sp.x *= u_aspect`
  (faithful to the shadertoy's `iResolution.x/iResolution.y`). Camera/box/window constants unchanged
  from the original (`u_cam_pos (2.75,-0.22,3.12)`, fov 53, `u_home_length 2.0`, distortion 20.48).
  Kept the y-split horizontal-tear band, the per-window hash light/curtains/frame, the door, distance
  fog, barrel distortion, scanlines.
- Source: maintainer's shadertoy (the building SDF is `sdBox` + ground plane; iq distfunctions
  https://iquilezles.org/articles/distfunctions/).
- Agent note (NOT a verdict): compiles + animates live; portrait framing is cramped — the original
  camera was tuned for landscape, so in 9:16 the building fills the frame and the floor/horizon-fog
  band is mostly off-screen.
- Maintainer verdict: pending review.

## v02 — city grid (`city v02 (grid)`)
- Direction (maintainer): more vertical, wider FOV, more distant camera, more/bigger buildings,
  greater scale, respect the portrait layout; REMOVE scanlines + fisheye (camera FX tuned later).
- Change:
  - Geometry: one `sdBox` → an SDF **XZ domain-repetition** grid. `id = clamp(round(p.xz/SPACING),
    -R, R)` gives a finite city footprint (cells [-5..5]²); per-cell half-height from `hash(id)`
    (`2 + 5*h²`, so a mix of low blocks and tall towers). Ground plane kept. Cell id + height carried
    out of the march so each tower seeds its own window pattern.
  - Window unwrap rewritten for local cell space: floors from the building base (`(p.y+1)/FLOOR_H`),
    along-face axis chosen by the dominant normal component, mapped to ~3 windows/face; per-face seed
    so the 4 sides differ. `draw_window`/curtains/frame machinery reused from the original.
  - Camera: pulled OUT of the grid to `(6,6,40)` looking in, FOV 70 — needed because an infinite grid
    surrounded/clipped the camera (caused floating-chunk artifacts in the first attempt).
  - Removed: scanlines, barrel/fisheye distortion, the y-split horizontal-tear band.
  - Sky: amber glow hugging the horizon → black overhead. Fog: dim amber, starts at dist 20.
- BUG FIXED (real, worth promoting as a lesson): the original march accumulates
  `rm.dist += length(rm.p - rm.ro)` EVERY step — that's the full origin-distance re-added per step, so
  `dist` balloons super-linearly. It only "worked" on the original tiny/close scene; over a deep city
  it saturated the distance-fog everywhere (whole frame flat olive). Fixed to `rm.dist = length(rm.p
  - rm.ro)` (true ray distance from camera). Distance fog is honest now.
- Source: SDF domain repetition — iq https://iquilezles.org/articles/distfunctions/ (the `round()`
  lattice; finite footprint via the limited-repetition pattern).
- BUG FIXED #2 (maintainer-reported "plane artifacts here and there" — faint diagonal/triangular
  slivers ghosting across the sky + between towers): the first finite-city attempt used
  `clamp(round(p.xz/SPACING), -R, R)` as a one-shot cell pick. That is NOT a valid SDF at the
  footprint boundary — for a point near the edge the clamped cell can be the WRONG (farther) cell, so
  `map()` over-reports distance, the march OVERSHOOTS, grazes a building's infinite-plane extension
  from far away, and registers a phantom near-zero hit -> thin planes. FIX = iq's **limited/bounded
  repetition done correctly**: sample the rounded cell AND its 3x3 neighbors (each clamped into the
  footprint) and take the min, so the genuinely-nearest building always wins and the field stays a
  valid SDF. Source: iq distfunctions, `opLimitedRepetition`
  https://iquilezles.org/articles/distfunctions/. PROMOTE candidate: "bounded SDF repetition needs a
  neighbor-cell min, not a clamped round()".
- Agent note (NOT a verdict): after the neighbor-min fix the sky is clean (no slivers) at t=0/2.5/5;
  window tiling consistent; camera bob subtle. Cost: `map()` is now 9 box-evals; fine on desktop.
- Maintainer verdict: pending review.

## v03 — life (`city v03 (life)`)
- Direction (maintainer): variate window light colors; add VERY slight distance fog (relax distant
  aliasing/tearing); enhance the roads (cars / trees / static detail); improve sky + add night-sky
  detail.
- Change:
  - **Window colors**: `window_light_color(hash)` palette — ~62% warm tungsten, ~22% cool white,
    ~10% teal screen-glow, ~6% warm-red; hue picked by a different hash than brightness so color and
    on/off vary independently. (Replaced the single `vec3(1,0.8,0.2)`.)
  - **Streets**: ground plane now `draw_street(xz)` — avenues run on the cell-boundary gaps (offset by
    half-spacing), dark asphalt + dashed center lines on the road strips that aren't under a building
    footprint. **Cars** = emissive ground glows (no geometry): per z-avenue, 5 cars at
    `mod(speed*t + span*seed, span)` world-z, keep-right lane offset, white headlights one way / red
    taillights the other; `attenuate()` falloff, glow ×3. (Chose ground-glow cars over 3D SDF cars —
    far cheaper, reads great at this distance; 3D would balloon the 9-box `map`.)
  - **Sky**: `draw_sky(rd)` — starfield (hash lattice ×26, ~7% density, twinkle + rare bright ones),
    a **moon** disc + halo, dim amber horizon glow.
  - **Fog**: slight (`*0.7`, starts at dist 25, gentler exponent) — softens far-tower aliasing.
- Moon-positioning lesson (PROMOTE candidate): the moon first rendered OFF-screen (only its halo
  bled in). A sky feature lives in a fixed WORLD direction, so whether it's in frame is camera-
  dependent. Fix: project the desired moon `dir` through the camera basis to screen NDC
  (`a=dir·fwd, b=dir·right, c=dir·up; ndc=(b/a/aspect, c/a)*sd`) and SOLVE the direction for a target
  on-screen position instead of guessing — placed it at NDC ~(0.35,0.78). (Computed in a one-off numpy
  snippet, baked the resulting `vec3(-0.015,0.410,-0.912)` as a const.)
- Source: car/road-as-emissive-ground-glow + starfield-from-hash-lattice are standard procedural
  city idioms (no single URL); SDF grid from v02 (iq).
- Agent note (NOT a verdict): all four asks read in one frame (moon top-right, starfield, varied
  window hues, head/tail-light cars sliding down the avenues, gentle far fog). Cars animate t=2→4.5.
- Maintainer verdict: see v04 — voice note 1880 (tg foo, 2026-06-17 16:36) was a polish punch-list.

## v04 — polish (`city v04 (polish)`)
- Direction: maintainer voice note 1880 (tg foo channel, 16:36 local). Verbatim asks (translated):
  1. Window colors: too many reds/greens + some odd whites — pull everything closer to a warm
     yellow middle, keep SOME variation but not radical.
  2. Every building shows a black VERTICAL stripe on one of its sections ("like a tank") — buildings
     look oddly painted.
  3. The grey building color needs more variation, but again not strongly standing out.
  4. Cars: red cars barely visible vs the white headlights coming at us; lanes are wrong. RIGHT-HAND
     traffic — white (toward us) should be on the LEFT side of the road from our view, red (away) on
     the RIGHT. Currently the away-cars drive in the middle and are barely visible. Normalize.
  5. Add a TEXTURE to the buildings — the flat grey is uninteresting.
- Change:
  1. `window_light_color` collapsed to 3 warm-anchored tiers (80% warm tungsten / 15% softer
     white-yellow / 5% neutral-cool) — dropped the teal + saturated-red + stark-blue tiers.
  2+5. **Black-stripe ROOT CAUSE** (confirmed by a corner crop, not guessed): the facade unwrapped
     `along∈[0,1]` across each face into exactly 3 windows with NO edge margin, and `draw_window`
     returned a near-black fall-through outside the glass — so the building CORNER (where the two
     visible faces meet, and `face_x`/seed flip) became a fat dark column. FIX: (a) split wall from
     light — `draw_window` now returns PURE emissive light (0 outside glass), and a new
     `wall_color(uv, bldg_seed)` draws textured concrete UNDER it everywhere (no fall-through); (b)
     inset the window band (`inset=0.12`) so the corner margins are solid wall, not a window column.
  3+5. `wall_color`: per-building base grey `mix(warm-concrete, cool-concrete, hash(cell))` in a tight
     band + fine grain + faint horizontal poured-floor banding. Rooftops use the same per-building
     varied palette.
  4. Cars: lane = `-dir * 0.42 * road_half_x` — `dir=+1` (toward us, +z, white) → our LEFT (-x),
     `dir=-1` (away, red) → our RIGHT (+x), matching right-hand traffic. Red taillights boosted
     (×4.2 vs headlights ×2.6, color `vec3(1,0.16,0.10)`) so they read against the brighter heads.
- Agent note (NOT a verdict): corner crop confirms the black stripe is gone (continuous textured
  wall across the corner); windows warm + cohesive; per-building grey varies; a red taillight car is
  clearly visible now. Cars sparse per still (realistic) — full traffic reads in live animation.
- Maintainer verdict: pending review.

## v05 — detail (`city v05 (detail)`)
- Direction (maintainer): facades too boring — more detail/geometry; add plane(s) in the sky.
- Change:
  - **Facade detail (all painted in `draw_facade`, zero march cost):** recessed-window AO (darken a
    ring just OUTSIDE the glass so windows look inset), window sills (bright horizontal ledge under
    each), subtle pilasters between bays, a darker window-less PLINTH at ground floor, a brighter
    CORNICE band at the top floor. `draw_window` already returns pure light (v04) so these layer on
    the wall cleanly. Needed `n_floors = floor(2*height/FLOOR_H)` passed in for top/bottom detection.
  - **Rooftop geometry (one cheap bit of real SDF):** `sd_building` now mins in a per-building
    water-tank box + a thin antenna mast (offset by `hash(id)`), so the skyline silhouette gets
    clutter. Cost: `map` boxes 9→27 evals; still fine on desktop.
  - **Planes:** emissive sky sprites in `draw_sky` (NOT geometry) — fuselage body + fast white
    anti-collision strobe + steady red(port)/green(starboard) wingtip nav lights; 2 planes drifting
    across, wrapping.
- PLANE-PLACEMENT lesson (PROMOTE candidate, same family as the v03 moon trick): a sky sprite lives
  in the `suv = rd.xz/rd.y` projection plane, and whether it's IN FRAME + ABOVE THE SKYLINE is
  camera-dependent and very sensitive. Spent several render cycles before projecting candidate `suv`
  positions to screen NDC in numpy: found the planes were at NDC y > 1 (above frame) and later that
  they kept landing BEHIND the central tower (sky sprites only draw where rays MISS geometry). Fix:
  solve `suv.y` for NDC y ~0.82–0.88 (above the skyline, clear sky) and size the x-sweep to the
  in-frame `suv.x` band. Reusable rule: place sky features by projecting to NDC, and remember they're
  occluded by tall geometry — keep them above the skyline.
- Source: facade-as-painted-detail + rooftop-clutter-SDF + plane-nav-lights are standard procedural
  idioms; SDF min for rooftop from iq.
- Agent note (NOT a verdict): facade crop shows recessed/silled windows + pilasters + texture (much
  less flat); rooftops have tank/mast silhouettes; a plane with white body + green nav light reads
  clearly in the left open sky at t=4. Planes are small + pass behind towers (only seen in sky gaps).
- Maintainer verdict: planes "look like dicks", white cars invisible, image too flat — see v06.

## v06 — depth (`city v06 (depth)`)
- Direction (maintainer, 3 asks): (1) planes awful (phallic) — reduce to a SINGLE blinking light,
  smaller, less frequent; (2) cars bug — red on the right is correct but the white car is barely
  visible (only part of its light on the right, the LEFT side of the road shows no lights at all);
  (3) image is very flat, no depth — RESEARCH basic night-cityscape art rules (sky vs city
  brightness? directional facade coloring by facing? how to add depth) and apply them.
- Change:
  1. **Plane:** dropped the fuselage body + red/green wing lights (those read as a phallus); now ONE
     small white dot with a slow blink, one pass every ~22s (off-frame most of the period).
  2. **Empty-left-lane BUG — REAL ROOT CAUSE (took several wrong turns first):** the road-cell math
     `acell = floor((xz + 0.5*SPACING)/SPACING)` put the `floor()` cell boundary EXACTLY on the road
     centerline, so each road was split across TWO cells — the -x (left) half got a garbage
     `local.x` (~+4.9) and the `on_z_road`/car logic rendered NOTHING there; only the +x half worked.
     That's why the left side was empty and only car GLOW leaked across the centerline.
     WRONG TURNS I burned (recorded as a process lesson): first chased brightness ("make white
     brighter"), then lane-sign, then — fatally — instrumented the ground by `local.x` sign, saw it
     "all one side", and CONCLUDED OCCLUSION. That was misreading the instrument: the road WASN'T
     occluded (maintainer: "I clearly see the left side"), the left half was being MIS-ASSIGNED to the
     neighbor cell. The maintainer's repeated "it's not occluded" was the correct signal my model was
     broken. FIX: round to the nearest road CENTERLINE — `rcell = round(xz/SPACING - 0.5) + 0.5;
     local = xz - rcell*SPACING` — so each road maps to one cell with a symmetric `[-half..half]`
     offset (verified by the sign-instrument: both red+green halves now render side by side). Lanes
     then per maintainer's call: red taillights on the RIGHT (local.x>0), white headlights on the
     LEFT (`lane = -dir*0.45*road_half_x`).
     **TWO lights per car** (maintainer follow-up): the single glow became a left/right PAIR across a
     track width (`track = 0.28*road_half_x`, each light tightened to `dx*3.2` + steeper attenuation
     so the pair reads as two distinct points, not one blob, at normal distance).
  3. **Lighting/depth overhaul** from a websearch synthesis (sources in the note below):
     - **Sky gradient:** the skyglow horizon is the BRIGHTEST large value at night; buildings
       silhouette dark against it. Sky = `mix(SKY_ZENITH cool-dark, SKY_HORIZON warm-amber,
       pow(1-rd.y, 2.6))` — glow hugs the horizon, dark cool zenith. (Was a near-flat dark sky.)
     - **Directional facade shading:** sky-dome + moon act as a soft key even at night — face turned
       toward sky/moon catches more (cooler) skylight, away face darker/warmer. `key = 0.55 +
       0.45*skyAmt + 0.40*moonAmt` (~0.55..1.4), cool/warm tint by facing. SUBTLE per the rule;
       applied to the SURFACE term only (emissive windows split out so they aren't dimmed). Verified:
       the hero tower's two faces now read ~32 vs ~23 mean value -> the box has form.
     - **Night aerial perspective:** distance fades a building TOWARD the horizon-glow color
       (`mix(color, haze, f*0.7)`), desaturates (`mix(color, luma, f*0.45)`), lifting far rows into
       the glow so FG/MG/BG separate into depth groups. (Replaced the flat grey fog.)
- Source (night-cityscape art rules, web-researched): Mitch Albala "Colors of Night"
  https://mitchalbala.com/colors-of-night-color-strategies-for-painting-landscape-nocturnes/ +
  "warm advances/cool recedes is secondary to value+saturation"
  https://mitchalbala.com/lies-my-art-teacher-told-me-warm-colors-advance-cool-colors-recede/ ;
  21-Draw atmospheric perspective at night
  https://www.21-draw.com/mastering-atmospheric-perspective-in-night-time-scenes/ ; skyglow
  horizon>zenith https://www.astropix.com/html/observing/skybrite.html . Key rules: sky is the
  brightest large value (buildings = dark silhouettes); horizon ~8-12x a near unlit wall; per-face
  ~1.3-1.6x value spread, subtle; distant buildings lerp up to ~0.7 toward horizon-glow color,
  desaturate, collapse contrast; reserve the only near-whites for windows/lights; three value groups.
- PROMOTE candidates: (a) the instrument-the-coordinate-by-color debug method (paint a field by a
  variable's sign to SEE which screen region a value covers — found the occluded-lane bug instantly
  after brightness-guessing failed); (b) the night-cityscape art-rule set itself (sky-gradient +
  directional ambient key + night aerial perspective) as a reusable lighting recipe / copilot note.
- Agent note (NOT a verdict): depth reads now — warm horizon glow, far buildings fade into it, FG
  tower a dark mass with two distinctly-valued faces; both car lanes populated with paired lights;
  plane is a single subtle dot.
- Maintainer verdict: pending review.
- FOLLOW-ON (same node, maintainer ask): **exposed ~57 tunables as uniforms** with inline defaults
  (the hardcoded falloffs, light/atmosphere params, sky/moon/plane, building colors + lights, car
  params). Grouped by semantic PREFIX so a name-sort clusters them: `u_cam_ u_city_ u_win_ u_wall_
  u_light_ u_fog_ u_sky_ u_plane_ u_car_`. ENGINE NOTES that shaped this (verified in code, PROMOTE
  candidates for the node-authoring recipe):
  - ShaderBox auto-discovers active uniforms each frame (`tabs/node.py` `UIUniform.from_uniform` +
    `snap_input_type`) — NO need to pre-list them in `node.json`'s `ui_uniforms`; inline
    `uniform T u_x = default;` is enough and the control appears live.
  - A vec3/vec4 only auto-gets the COLOR picker when its name ENDS in `color` (`ui_models.py
    reset_input_type`); else it's a 3/4-float drag. So colors are named `u_<group>_<role>_color`
    (e.g. `u_win_warm_color`, `u_sky_horizon_color`) — keeps the prefix grouping AND the suffix rule.
  - A GLSL `for` bound must be a CONSTANT, so cars-per-avenue / star density / the 3x3 march can't be
    uniform loop limits — kept `#define CARS_PER_AVENUE`; the per-iteration PARAMS are the uniforms.
  - Defaults reproduce the prior v06 image pixel-for-pixel (pure structural change).

## Learning sequence — ONE node (`city (learning)`)
- Direction (maintainer): a SINGLE node containing the whole learning sequence (timed step reveal +
  per-step caption text) — the `fire (timed reveal)` model — NOT individual per-step nodes.
  (WRONG FIRST ATTEMPT: I built 8 separate `stepN` nodes; deleted them. The ask is one node where
  the build animates over time with caption overlays, exactly like the fire timed-reveal node.)
- Built by porting the fire timed-reveal machinery onto the v06 scene:
  - `reveal_time()` / `reveal(i)` looping clock (STEP_DUR=2s, HOLD=4s, 8 steps, FADE=0.8) +
    `u_debug_step` (1..8 freezes the build at a step for inspection).
  - Each feature gated by its step weight `wN`. Geometry-affecting gates (rooftops, facade detail,
    sky, streets, plane) are set as GLOBALS before the march; shading-only gates (windows, light, fog)
    branch in main(). Captions: 8 `u_stepNtext[40]` codepoint arrays (engine encodes typed strings,
    name ends in `text`), drawn via `SB_sd_char`, faded per `caption_alpha(i)` — copied from fire.
  - Seeded the 8 captions ("LABEL\nDETAIL", UPPERCASE — only uppercase latin glyphs exist): GEOMETRY/
    WINDOWS/DETAIL/ROOFTOPS/NIGHT SKY/STREETS/LIGHT/DEPTH. Scene uniforms = v06's tuned values.
- Steps: 1 geometry (bare boxes) -> 2 windows -> 3 facade detail+concrete -> 4 rooftops -> 5 sky ->
  6 streets+cars -> 7 directional light -> 8 aerial perspective + plane (== tuned v06).
- Verified: compiles; the reveal builds up over the timeline and the final frame matches v06; captions
  render (two-line, centred, fade in/out per slot).
- PROMOTE candidate (/shader-lab): the fire timed-reveal node is now a reusable PATTERN for any lab —
  gate features by a looping per-step weight + per-step `u_stepNtext` captions, in ONE node. The
  step-by-step development uses throwaway vNN nodes; the learning DELIVERABLE is this single node.
- Maintainer verdict: kept; then asked for entrance ANIMATIONS per step + a roof-clutter fix.
- FOLLOW-ONS (same node):
  - Text style copied from the fire node verbatim (spacing 0.73 / size 0.18 / weight 0.10 /
    sharp 0.06 / line_gap 0.19 / color / fades); `u_text_pos` stays in this node's SCREEN space
    (fire's was flame-space).
  - `u_debug_step` -> renamed `u_step` (0 = autoplay, 1..8 = jump/freeze a step; uint -> int slider).
  - Road lane dashes softened: dimmer color (0.30->0.13), narrower, hard `step()` dash replaced with
    `smoothstep` ramps so dash ends taper.
  - **Per-step entrance animations** via a new `anim(i, dur)` eased 0->1 ramp across each step's slot
    (separate from the `reveal()` fade): step1 buildings GROW from the ground (animated half-height
    `grown_height()` scales the SDF box, per-building stagger — set as a global BEFORE the march since
    it's geometry); step2 windows light up bottom-up (`WIN_LIGHT` sweeps a per-floor threshold);
    step4 rooftop clutter rises onto the roof (`ROOF_RISE` lifts the tank/mast); step6 traffic
    streams in (`CAR_FADE`); step8 plane rolls in (`PLANE_FADE`). Crossfade-only steps (detail, sky,
    light, fog) keep the plain fade — forcing motion on those looked gimmicky.
  - **Roof-clutter bug fixed** (maintainer: "roof elements still colored as main body, I see windows
    on them"): a building-face hit ABOVE the roof (`rm.p.y > ground_y + 2*height`) is the tank/mast,
    so it now shades as a plain dark metal/concrete structure instead of running `draw_facade`
    (windows). roof_y uses FULL height (grow finishes long before clutter appears at step 4).
- Maintainer verdict (voice 1881): animations still not enough — (a) detail (step 3) pops abruptly,
  should "проступать" (develop in gradually); (b) rooftop elements too small/invisible while rising —
  HIGHLIGHT them with a bright over-exposed bounding-box wireframe (we know the SDF distance to the
  box), expose the color/intensity; (c) fog should fade in smoothly; (d) sky+stars+moon should fade
  in, and the moon could arc up from below interestingly.
- FOLLOW-ON (same node) — richer entrance animations:
  - Smooth "проступать" gates: step 3 detail (texture/AO/sills/window-variety) crossfades via
    `DETAIL_MIX` (no hard flip); step 5 sky+stars fade via `SKY_FADE`; step 8 fog fades via `FOG_FADE`.
    (Geometry/sprite-spawning gates — ROOF/STREET/PLANE — stay binary; you can't half-spawn geometry.)
  - **Rooftop bounding-box highlight**: a new `sdBoxFrame` (iq box-frame SDF); the march accumulates
    the nearest distance to the clutter's bounding-box wireframe into a global `CLUTTER_EDGE` (only
    while rising). main() draws `u_roofbox_color * exp(-w²·edge²) * u_roofbox_gain` modulated by
    `sin(ROOF_RISE·PI)` so the wireframe glows over-exposed mid-rise and fades to nothing once
    settled. Exposed: `u_roofbox_color` / `u_roofbox_gain` / `u_roofbox_width`. (PROMOTE candidate:
    SDF box-frame wireframe glow as a reusable "highlight a bounding box" technique — cheap in a
    raymarcher since the distance is already known.)
  - **Moon arc-in**: `moon_dir_anim()` rotates the final `u_sky_moon_dir` down/back at `MOON_RISE`=0
    (below horizon) and eases up along a vertical arc (Rodrigues rotation about the dir's azimuth
    axis) to its final spot by =1; runs slower than its step (dur 1.4×STEP_DUR) so it lingers.
- Verified each: detail develops gradually, rooftop wireframe glows during rise then vanishes, sky/
  moon fade+arc in, fog fades; final frame settles to the tuned look with no leftover highlight.
- Maintainer verdict (voice 1882): 3 problems — (a) step-7 light snaps on in one frame; (b) WHITE
  horizontal stripe artifact crossing windows when the texture appears (step 3); (c) the rooftop
  bounding-box wireframe looks awful/chunky/buggy — just color the elements bright instead.
- FOLLOW-ON fixes (same node):
  - Step 7 light now EASED in (`anim(7)` lerps key->1 and tint->white at fade start) — was applying
    full key the instant `w7>=0.5`.
  - **White-stripe artifact ROOT CAUSE**: `DETAIL_ON` flipped at `reveal(3)>=0.5`, which lands
    PARTWAY into the slot where `DETAIL_MIX` is already ~0.13 — so the detail terms (sill/band/AO)
    jumped from 0 to a 13%-blend in ONE frame (the white horizontal sill+band lines popping across the
    windows). FIX: flip `DETAIL_ON` (and `SKY_ON`) at the SLOT START — gate on `step(0.001, anim_ramp)`
    so the detailed path turns on while its blend is still 0, then eases up. (Lesson: a binary
    use-detailed-path switch must flip at the same instant its crossfade weight is 0, not at
    reveal>=0.5 — otherwise the eased term snaps to a partial value.)
  - Rooftop wireframe DELETED (the `sdBoxFrame` accumulation read as chunky boxes). Replaced per the
    maintainer's suggestion: TINT the clutter geometry itself bright (`u_roofbox_color`, boosted by
    `u_roofbox_gain`) while it rises (`sin(ROOF_RISE*PI)` envelope), settling to plain. `u_roofbox_width`
    dropped.
- Maintainer verdict: kept; tuned uniforms; then 3 asks: drop fog-desat (broken — large values just
  turn everything an artifact-y blue, small values do nothing — set to 0), rework caption text, +0.5s
  per step.
- FOLLOW-ON:
  - **`u_fog_desat` REMOVED** (shader + node.json). The aerial-perspective desat-toward-luma step was
    the offender; kept only the haze-tint (`u_fog_haze`). (Likely it desaturated toward LUMA then the
    haze tint pushed blue — over-strong values washed to blue, weak did ~nothing.)
  - **Captions reworked**: line 1 = the IDEA, line 2 = the core MATH trick. GEOMETRY/`ID=ROUND(P/S)`,
    WINDOWS/`UV=FRACT(FACE)`, DETAIL/`AO=SMOOTHSTEP(D)`, ROOFTOPS/`MIN(BODY,TANK,MAST)`, NIGHT SKY/
    `MIX(ZEN,HOR,POW(1-Y))`, TRAFFIC/`I=1/(1+B*D+C*D*D)`, MOONLIGHT/`KEY=DOT(N,L)`, DEPTH/
    `MIX(COL,HAZE,DIST)`. (Glyph set is UPPERCASE latin + digits + `()+=*/<>%&':;,.-!?` — formulas
    transliterated to fit; the auto-fit shrinks the longer math line to frame width.)
  - **STEP_DUR 2.0 -> 2.5** (+0.5s/step).
  - NOTE: maintainer left `u_step=9` (debug-freeze past the last step) -> the live node shows the
    frozen final scene with NO captions; set `u_step=0` to watch the autoplay sequence.
- Maintainer verdict: traffic-intensity control missing; 2->3 transition still clicks in one frame.
- FOLLOW-ON:
  - **`u_car_intensity`** (0..1, default 0.6) added — a density slider. GLSL loop bounds must be const
    so the COUNT can't be a uniform; instead `CARS_PER_AVENUE` raised to 8 (max slots) and a per-car
    `hash > u_car_intensity` `continue` skips slots -> effective density control.
  - **2->3 one-frame click FIXED** (found by measuring per-pixel frame-to-frame delta across the
    boundary: a 4973-pixel spike at the slot start, vs ~600 baseline). ROOT CAUSE: two binary flips at
    the `DETAIL_ON` switch that `DETAIL_MIX` did NOT cover — (1) the PLINTH (`idx.y<0.5`) hard-switched
    the whole ground floor to the dark window-less base AND early-returned (killing its windows) in one
    frame; (2) `has_curtains` was a hard bool applied fully inside `draw_window`. FIX: ease both by
    `DETAIL_MIX` — plinth lerps `mix(wall, plinth, DETAIL_MIX)` + its window fades via `win_mul =
    1-DETAIL_MIX`; `draw_window` now takes a `curtain_amt` float (`has_curtains ? DETAIL_MIX : 0`).
    Spike dropped 4973 -> 603 px. LESSON (extends the earlier DETAIL_ON-flip note): when a step's
    crossfade is gated by a binary "use-detail-path" switch, EVERY branch inside that switch must also
    scale by the crossfade weight — an early-return or an unscaled bool inside it still pops.
- FINAL timing: steps 1-7 = 3.0s each, step 8 = 5.0s, HOLD = 4.0s -> build 26.0s + hold = **30.0s
  loop** (set `render_media_details.duration = 30.0`; t=0 == t=30 verified seamless). `u_step=0` for
  autoplay render (NOT 9/frozen).
- SESSION CLOSED 2026-06-17. Generic knowledge (raymarch/SDF craft, night-lighting art rules,
  step-reveal animation pattern + the crossfade-switch trap, debug discipline) promoted into the
  `/shader-lab` skill. The YouTube Shorts resolution-preset feature (720/1080/1440) shipped to the app
  proper; two Share/copilot UX gaps filed in `todo.md`. This lab committed to git as a referenceable
  past lab.
