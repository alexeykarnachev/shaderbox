#version 460 core

in vec2 vs_uv; // Coordinate of the current pixel to be shaded

uniform float u_time;   // Time (s) since the application started
uniform float u_aspect; // Aspect ratio of the canvas (width / height)
// uniform vec2 u_resolution;  // Resolution of the canvas (width, height)

out vec4 fs_color;

// Flame space: x in [-aspect, aspect], y from 0 (bottom) -> ~2 (top). No wind.
vec2 flame_space(vec2 uv, float aspect) {
    vec2 p = SB_center_uv(uv, aspect);
    p.y += 1.0;
    return p;
}

// Domain-warped, upward-scrolling turbulence (iq recursive warp): plain fbm is
// cloudy, warped fbm curls like flame. scale/speed let the SAME field be sampled
// coarse (body motion) or fine (interior detail) — from v04.
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

// Two-scale warped field (v04 interior texture): coarse body motion + fine
// interior crackle, so the body has structure at every scale.
float interior_field(vec2 p, float t) {
    float coarse = warp_noise(p, t, 1.4, 1.7);
    float fine = warp_noise(p, t, 3.6, 2.6);
    return coarse * 0.7 + fine * 0.3;
}

// Flame SHAPE from v03: a soft column envelope + vertical fuel, with the noise
// field deciding WHERE within it stays lit. Rising body with licking tongues,
// NOT a teardrop. Returns the heat field with v04's interior veins riding in it.
float flame_heat(vec2 p, float t, float n) {
    // Horizontal envelope (v03): narrower toward the top -> rising body, not dome.
    float h = clamp(p.y * 0.55, 0.0, 1.0);
    float width = mix(0.55, 0.16, h) * (0.7 + 0.3 * smoothstep(0.0, 0.25, p.y));
    float column = exp(-pow(abs(p.x) / width, 2.0));

    // Vertical fuel (v03): lots at the base, running out with height; tongues tied
    // to the column so they stay attached.
    float fuel = (1.0 - smoothstep(0.1, 1.8, p.y)) + n * 0.25 * column;

    // Burn (v03): field must exceed a rising threshold to stay lit -> frays into
    // tongues near the top where fuel is scarce.
    float burn = smoothstep(0.42, 0.9, n * fuel * column * 2.6);

    // v04 interior: the lit body keeps the noise veins (hot/cool depth) and runs
    // hotter at the rooted base, instead of a flat fill.
    float heat = burn * mix(0.65, 1.3, n);
    heat *= 1.0 + (1.0 - smoothstep(0.0, 0.6, p.y)) * 0.6;
    return clamp(heat, 0.0, 1.5);
}

// Blackbody-ish temperature ramp (v04, 5 stops): over-scaled so the hottest field
// peaks blow to white while mid values stay orange/yellow -> visible interior veins.
vec3 fire_ramp(float h) {
    vec3 c = vec3(0.0);
    c = mix(c, vec3(0.5, 0.04, 0.0), smoothstep(0.05, 0.35, h)); // deep ember
    c = mix(c, vec3(0.95, 0.25, 0.02), smoothstep(0.3, 0.55, h)); // orange
    c = mix(c, vec3(1.0, 0.65, 0.12), smoothstep(0.5, 0.78, h));  // yellow
    c = mix(c, vec3(1.0, 0.92, 0.55), smoothstep(0.75, 1.0, h));  // near-white
    c = mix(c, vec3(1.0, 1.0, 0.95), smoothstep(1.0, 1.3, h));    // white core
    return c;
}

// ROUND radiating glow: a radial bloom centred near the flame base that bleeds
// into the dark as an aura — decoupled from the (triangular) silhouette, so the
// light spreads circularly into space instead of hugging the flame outline. Its
// brightness flickers with the field so it lives with the flame.
vec3 round_glow(vec2 p, float n) {
    vec2 src = vec2(0.0, 0.35);          // glow centre, low in the body
    float d = length((p - src) * vec2(1.0, 0.85)); // near-circular, slightly tall
    float halo = exp(-2.2 * d * d);      // radial falloff -> round aura
    halo *= 0.45 + 0.55 * n;             // flicker with the field
    return vec3(1.0, 0.38, 0.1) * halo * 0.5;
}

// Sparse rising embers (v04): cellular grid scrolling up faster than the flame.
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

    float n = interior_field(p, u_time);
    float heat = flame_heat(p, u_time, n);

    vec3 color = fire_ramp(heat);
    color += round_glow(p, n);
    color += embers(p, u_time);

    // Tight bloom on the hottest core.
    color += vec3(1.0, 0.85, 0.6) * smoothstep(0.7, 1.0, heat) * 0.4;

    color = min(color, vec3(1.0));
    fs_color = vec4(color, 1.0);
}
