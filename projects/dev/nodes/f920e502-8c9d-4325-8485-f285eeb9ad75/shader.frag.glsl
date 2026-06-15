#version 460 core

in vec2 vs_uv; // Coordinate of the current pixel to be shaded

uniform float u_time;   // Time (s) since the application started
uniform float u_aspect; // Aspect ratio of the canvas (width / height)
// uniform vec2 u_resolution;  // Resolution of the canvas (width, height)

// Step 6: driven by the CPU script (scripts/script.py) each frame. The fire's
// BEHAVIOUR lives in Python state; the shader just reads the current values.
uniform float u_intensity = 1.0; // overall brightness — the fire breathes
uniform float u_flicker = 1.0;   // fast per-frame brightness jitter
uniform float u_wind = 0.0;      // horizontal wind: leans the flame, gusts

out vec4 fs_color;

// Flame space: x in [-aspect, aspect], y from 0 (bottom) -> ~2 (top). Wind
// shears the column sideways, more the higher you go (the tip swings most).
vec2 flame_space(vec2 uv, float aspect) {
    vec2 p = SB_center_uv(uv, aspect);
    p.y += 1.0;
    p.x -= u_wind * smoothstep(0.0, 1.6, p.y) * 0.6;
    return p;
}

// Step 1: a domain-warped, upward-scrolling turbulence field (iq's recursive
// warp). Plain fbm reads as soft clouds; feeding fbm into fbm's coordinates
// curls it into licking flame tongues. Scroll y downward in time so the field
// flows UP.
float flame_field(vec2 p, float t) {
    vec2 sp = vec2(p.x * 1.6, p.y * 1.1 - t * 1.6); // scrolled sample space

    vec2 q = vec2(
        SB_fbm(sp + vec2(0.0, 0.0), 5),
        SB_fbm(sp + vec2(5.2, 1.3), 5)
    );
    vec2 r = vec2(
        SB_fbm(sp + 3.0 * q + vec2(1.7, 9.2), 5),
        SB_fbm(sp + 3.0 * q + vec2(8.3, 2.8), 5)
    );
    return SB_fbm(sp + 3.0 * r, 5); // [0, 1]
}

// Step 1: temperature ramp. Over-scale the intensity so the hot core blows out
// to white while the edges stay deep red — the blackbody gradient that reads as
// fire rather than a flat glow.
vec3 fire_ramp(float h) {
    vec3 c = vec3(0.0);
    c = mix(c, vec3(0.6, 0.05, 0.0), smoothstep(0.0, 0.3, h));  // ember red
    c = mix(c, vec3(1.0, 0.3, 0.02), smoothstep(0.25, 0.55, h)); // orange
    c = mix(c, vec3(1.0, 0.78, 0.18), smoothstep(0.5, 0.8, h));  // yellow
    c = mix(c, vec3(1.0, 0.98, 0.88), smoothstep(0.8, 1.05, h)); // white-hot
    return c;
}

// Step 2: carve a flame shape out of the field. The flame is "fuel" that burns
// hotter at the base and is consumed with height; the turbulent field decides
// WHERE within that envelope it still burns, so the silhouette licks and splits
// instead of being a hard mask.
float flame_heat(vec2 p, float t) {
    float f = flame_field(p, u_time);

    // Horizontal envelope: a soft column, clearly narrower toward the top so the
    // flame is a rising body, not a dome. Teardrop: pinch in slightly at the very
    // base too.
    float h = clamp(p.y * 0.55, 0.0, 1.0);
    float width = mix(0.55, 0.16, h) * (0.7 + 0.3 * smoothstep(0.0, 0.25, p.y));
    float column = exp(-pow(abs(p.x) / width, 2.0));

    // Vertical fuel: lots at the base, running out with height. Tie the extra
    // fuel to the column so tongues stay ATTACHED to the body (no free-floating
    // islands) — high-noise spots inside the column flare up taller.
    float fuel = (1.0 - smoothstep(0.1, 1.8, p.y)) + f * 0.25 * column;

    // Burn: the field must exceed a rising threshold to stay lit, so the flame
    // frays into separate tongues near the top where fuel is scarce.
    float burn = smoothstep(0.42, 0.9, f * fuel * column * 2.6);
    return burn;
}

// Step 3: emissive glow. Fire throws light, so add a soft warm halo around the
// body — a low, wide envelope of the flame field that isn't clipped by the burn
// threshold — plus a tight bloom on the white-hot core. Additive, so it spills
// colour into the dark instead of just brightening the flame.
vec3 flame_glow(vec2 p, float t) {
    float f = flame_field(p, u_time);

    // Wide, soft halo: a fat column around the body, modulated by the field so
    // the glow flickers with the flame rather than being a static gradient.
    float width = mix(0.8, 0.35, clamp(p.y * 0.5, 0.0, 1.0));
    float halo = exp(-pow(abs(p.x) / width, 2.0));
    halo *= (1.0 - smoothstep(0.0, 1.9, p.y)); // fade upward
    halo *= 0.35 + 0.65 * f;                   // flicker with the field

    vec3 glow = vec3(1.0, 0.35, 0.06) * halo * 0.5;
    return glow;
}

// Step 4: rising embers. Sparks born near the flame and carried UP with the
// flow, scrolling faster than the flame body. A cellular grid: each cell holds
// at most one spark at a random sub-position, twinkling on its own phase, fading
// as it climbs. They drift sideways with a little turbulence so they don't rise
// in straight lines.
vec3 embers(vec2 p, float t) {
    // Cell space: scroll upward faster than the flame, sway in x with time.
    float rise = t * 0.9;
    vec2 cs = vec2(p.x * 6.0 + sin(p.y * 3.0 + t) * 0.4, (p.y + rise) * 6.0);
    vec2 cell = floor(cs);
    vec2 fr = fract(cs) - 0.5;

    float spark = 0.0;
    // Per-cell random: does it emit, at what offset, what twinkle phase. Few
    // cells emit (sparks are sparse).
    float emit = SB_hash21(cell);
    if (emit > 0.82) {
        vec2 off = vec2(SB_hash21(cell + 3.1) - 0.5, SB_hash21(cell + 7.7) - 0.5) * 0.5;
        vec2 dp = fr - off;
        dp.y *= 0.45; // vertical streak — motion-blurred rising spark, not a dot
        float d = length(dp);
        float twinkle = max(0.0, sin(t * 7.0 + emit * 40.0)); // blink in and out
        spark = exp(-260.0 * d * d) * twinkle;
    }

    // A cone above the flame that NARROWS with height — sparks funnel up, not a
    // starfield. Born near the flame top, gone before they reach the ceiling.
    float band = smoothstep(0.35, 0.7, p.y) * (1.0 - smoothstep(1.2, 1.8, p.y));
    float cone_w = mix(0.45, 0.18, clamp(p.y * 0.55, 0.0, 1.0));
    float inside = exp(-pow(abs(p.x) / cone_w, 2.0));
    return vec3(1.0, 0.6, 0.25) * spark * band * inside * 1.6;
}

// Step 5: smoke. Rises ABOVE the flame, slower and at a larger scale than the
// fire, drifting and eroding as it climbs. Returns (density) so main() can blend
// a dark wisp over the background — it lifts off the flame tips and fades out
// near the top.
float smoke(vec2 p, float t) {
    // Large, slow, upward-scrolling noise (slower rise than the flame).
    vec2 sp = vec2(p.x * 1.1 + sin(p.y * 1.3 + t * 0.5) * 0.3, p.y * 0.9 - t * 0.7);
    float n = SB_fbm(sp, 4);

    // Appears just above the flame body, thickening then dissipating with height.
    float rise = smoothstep(0.8, 1.4, p.y) * (1.0 - smoothstep(1.6, 2.1, p.y));
    float column = exp(-pow(abs(p.x) / 0.6, 2.0));

    // Erode: only the denser noise survives, so it breaks into ragged puffs.
    float density = smoothstep(0.4, 0.75, n) * rise * column;
    return density;
}

void main() {
    vec2 p = flame_space(vs_uv, u_aspect);

    // Script breathes + flickers the heat BEFORE colour mapping, so a swell
    // brightens the body without clipping the already-white core to a flat blob.
    float heat = flame_heat(p, u_time) * u_intensity * u_flicker;

    vec3 color = fire_ramp(heat);
    color += flame_glow(p, u_time);

    // Smoke sits over the background — SELF-LIT (a black background gives it
    // nothing to occlude, so it must emit a little to read), warm near the flame
    // base and cooling to gray as it rises.
    float sm = smoke(p, u_time);
    vec3 smoke_warm = vec3(0.22, 0.13, 0.09); // lit by the fire below
    vec3 smoke_cool = vec3(0.10, 0.10, 0.12); // cooled, higher up
    vec3 smoke_col = mix(smoke_warm, smoke_cool, smoothstep(1.0, 1.7, p.y));
    color = mix(color, smoke_col, sm * (1.0 - heat));

    color += embers(p, u_time);

    // Tight bloom on the hottest core.
    color += vec3(1.0, 0.85, 0.6) * smoothstep(0.7, 1.0, heat) * 0.4;

    color = min(color, vec3(1.0));
    fs_color = vec4(color, 1.0);
}
