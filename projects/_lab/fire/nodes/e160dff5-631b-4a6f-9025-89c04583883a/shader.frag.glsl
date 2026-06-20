#version 460 core

in vec2 vs_uv; // Coordinate of the current pixel to be shaded

uniform float u_time;   // Time (s) since the application started
uniform float u_aspect; // Aspect ratio of the canvas (width / height)
// uniform vec2 u_resolution;  // Resolution of the canvas (width, height)

// --- Live-tunable knobs (auto-generated ImGui controls). Flicker/glow defaults
// seeded from the user's v08 tuning. ----------------------------------------
uniform float u_flicker_amp = 0.65;   // brightness swing amplitude (0 = steady)
uniform float u_flicker_speed = 0.88; // flicker rate multiplier
uniform float u_glow_strength = 0.68; // round-glow brightness
uniform float u_glow_radius = 0.44;   // round-glow falloff (lower = wider aura)
uniform float u_depth = 0.45;         // depth-layer strength behind the flame (0 = flat)

out vec4 fs_color;

// Flame space: x in [-aspect, aspect], y from 0 (bottom) -> ~2 (top). No wind.
vec2 flame_space(vec2 uv, float aspect) {
    vec2 p = SB_center_uv(uv, aspect);
    p.y += 1.0;
    return p;
}

// Domain-warped, upward-scrolling turbulence (iq recursive warp). scale/speed
// sample the SAME field coarse (body motion) or fine (interior detail).
float warp_noise(vec2 p, float t, float scale, float speed) {
    vec2 sp = vec2(p.x * scale, p.y * scale * 0.7 - t * speed);
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

// Two-scale warped field (v04 interior texture): coarse body motion + fine crackle.
float interior_field(vec2 p, float t) {
    float coarse = warp_noise(p, t, 1.4, 2.4);
    float fine = warp_noise(p, t, 3.6, 3.4);
    return coarse * 0.7 + fine * 0.3;
}

// Low-frequency flow used to DISPLACE the silhouette sample point so the whole
// boundary swims (v07) instead of sitting as a static cone.
vec2 flow_offset(vec2 p, float t) {
    vec2 sp = vec2(p.x * 1.2, p.y * 0.9 - t * 1.8);
    float ox = SB_fbm(sp + vec2(11.5, 2.1), 4) - 0.5;
    float oy = SB_fbm(sp + vec2(3.7, 19.3), 4) - 0.5;
    return vec2(ox, oy);
}

// Heat field (v07): warped moving silhouette + field-eaten edges + raw continuous
// interior (v04 veins).
float flame_heat(vec2 p, float t, float n) {
    float topness = smoothstep(0.0, 1.6, p.y);
    vec2 w = p + flow_offset(p, t) * (0.12 + topness * 0.55);

    float hh = clamp(w.y * 0.55, 0.0, 1.0);
    float width = mix(0.5, 0.14, hh) * (0.7 + 0.3 * smoothstep(0.0, 0.25, w.y));
    float column = exp(-pow(abs(w.x) / width, 2.0));

    float fuel = (1.0 - smoothstep(0.1, 1.8, w.y)) + n * 0.35;
    float gate = column * smoothstep(0.05, 0.5, fuel);
    float heat = n * gate;

    heat -= (1.0 - n) * (0.15 + topness * 0.7) * gate;
    heat *= smoothstep(0.06, 0.4, n * gate + n * 0.2);

    heat = pow(clamp(heat, 0.0, 1.0), 0.8) * 1.2;
    heat *= 1.0 + (1.0 - smoothstep(0.0, 0.6, p.y)) * 0.35;
    return clamp(heat, 0.0, 1.4);
}

// LIGHT FLICKER. Two parts, both scaled by u_flicker_amp:
//  - a GLOBAL swing (no spatial term) -> the whole flame brightens/dims together,
//    the part you can actually SEE as the fire pulsing (and that differs frame to
//    frame in a still). A fast jitter + a slow swell.
//  - a SPATIAL jitter -> different heights flutter out of phase, for liveliness.
// u_flicker_speed scales the rate; u_flicker_amp scales how hard it swings.
float flicker(vec2 p, float t) {
    // GLOBAL light swing (no spatial term — the cast light pulses as a whole). A
    // fast jitter + a mid wobble + a slow swell so it isn't a clean sine.
    float s = u_flicker_speed;
    float f = 0.5 * sin(t * 13.0 * s)
            + 0.3 * sin(t * 6.7 * s + 1.1)
            + 0.2 * sin(t * 3.1 * s + 0.5); // [-1, 1]
    return 1.0 + f * u_flicker_amp;
}

// Blackbody-ish temperature ramp (v04, 5 stops).
vec3 fire_ramp(float h) {
    vec3 c = vec3(0.0);
    c = mix(c, vec3(0.5, 0.04, 0.0), smoothstep(0.05, 0.35, h)); // deep ember
    c = mix(c, vec3(0.95, 0.25, 0.02), smoothstep(0.3, 0.55, h)); // orange
    c = mix(c, vec3(1.0, 0.65, 0.12), smoothstep(0.5, 0.78, h));  // yellow
    c = mix(c, vec3(1.0, 0.92, 0.55), smoothstep(0.75, 1.0, h));  // near-white
    c = mix(c, vec3(1.0, 1.0, 0.95), smoothstep(1.0, 1.3, h));    // white core
    return c;
}

// ROUND radiating glow (v05): radial bloom near the base, decoupled from the
// silhouette. The FLICKER lives HERE: a flame's flicker is the ambient LIGHT it
// casts pulsing, not the flame body changing colour — so `light` (the flicker
// multiplier) scales the spreading glow + its radius, while the flame stays steady.
vec3 round_glow(vec2 p, float n, float light) {
    vec2 src = vec2(0.0, 0.35);
    float d = length((p - src) * vec2(1.0, 0.85));
    // The aura BREATHES: brighter flicker spreads the light a little wider too.
    float radius = u_glow_radius / max(0.4, light);
    float halo = exp(-radius * d * d);
    halo *= 0.45 + 0.55 * n;
    return vec3(1.0, 0.38, 0.1) * halo * u_glow_strength * light;
}

// Sparse rising embers (v04).
vec3 embers(vec2 p, float t) {
    vec2 cs = vec2(p.x * 7.0, (p.y + t * 1.0) * 7.0);
    vec2 cell = floor(cs);
    vec2 fr = fract(cs) - 0.5;
    float emit = SB_hash21(cell);
    float spark = 0.0;
    if (emit > 0.84) {
        vec2 off = vec2(SB_hash21(cell + 3.1) - 0.5, SB_hash21(cell + 7.7) - 0.5) * 0.5;
        vec2 dp = fr - off;
        dp.y *= 0.4;
        spark = exp(-300.0 * dot(dp, dp)) * max(0.0, sin(t * 6.0 + emit * 40.0));
    }
    float band = smoothstep(0.5, 0.9, p.y) * (1.0 - smoothstep(1.4, 2.0, p.y));
    float cone = exp(-pow(abs(p.x) / mix(0.4, 0.15, clamp(p.y * 0.5, 0.0, 1.0)), 2.0));
    return vec3(1.0, 0.6, 0.25) * spark * band * cone * 1.5;
}

void main() {
    vec2 p = flame_space(vs_uv, u_aspect);
    float light = flicker(p, u_time);

    vec3 color = vec3(0.0);

    // DEPTH (behind): a second flame burning further back — same machinery on a
    // shifted, slightly-scaled, time-offset point, then DIMMED + cooled. Reads as a
    // flame behind the main one, giving the image depth instead of one flat sheet.
    // Subtle by default (u_depth); the front flame is unchanged from v09.
    vec2 bp = vec2(p.x * 1.12 + 0.16, p.y * 1.05 + 0.05);
    float bn = interior_field(bp, u_time + 7.3);
    float bheat = flame_heat(bp, u_time + 7.3, bn) * 0.75; // cooler -> oranger, dimmer
    vec3 bcol = fire_ramp(bheat) * vec3(0.85, 0.7, 0.7);   // tint back toward red/dim
    float bcover = smoothstep(0.02, 0.14, bheat) * u_depth;
    color = mix(color, bcol, bcover);

    // FRONT flame — identical to v09.
    float n = interior_field(p, u_time);
    float heat = flame_heat(p, u_time, n);
    vec3 fcol = fire_ramp(heat);
    float fcover = smoothstep(0.02, 0.12, heat);
    color = mix(color, fcol, fcover);

    // Glow + flicker (v09): ambient light the fire casts, breathing.
    color += round_glow(p, n, light);

    color += embers(p, u_time);
    color += vec3(1.0, 0.85, 0.6) * smoothstep(0.8, 1.05, heat) * 0.35;

    color = min(color, vec3(1.0));
    fs_color = vec4(color, 1.0);
}
