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
// cloudy, warped fbm curls like flame. scale/speed sample the SAME field coarse
// (body motion) or fine (interior detail).
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
// interior crackle. Scrolls a bit FASTER than v06 so the motion reads.
float interior_field(vec2 p, float t) {
    float coarse = warp_noise(p, t, 1.4, 2.4);
    float fine = warp_noise(p, t, 3.6, 3.4);
    return coarse * 0.7 + fine * 0.3;
}

// A separate, lower-frequency flow field used to DISPLACE the silhouette sample
// point — this is what makes the whole boundary swim instead of sitting as a
// static cone. Returns a signed 2D offset that scrolls upward over time.
vec2 flow_offset(vec2 p, float t) {
    vec2 sp = vec2(p.x * 1.2, p.y * 0.9 - t * 1.8);
    float ox = SB_fbm(sp + vec2(11.5, 2.1), 4) - 0.5;
    float oy = SB_fbm(sp + vec2(3.7, 19.3), 4) - 0.5;
    return vec2(ox, oy);
}

// Heat field. vs v06: the envelope is built on a DOMAIN-WARPED point (the whole
// silhouette licks/sways with the flow), and the edge is EATEN by the field
// height-weighted (frays into tongues, not straight triangle sides). Interior is
// still the raw continuous field (v04), so the body keeps its veins.
float flame_heat(vec2 p, float t, float n) {
    // Warp the sample point by the flow, MORE toward the top (calm base, wild
    // tip). The envelope below is computed on `w`, so its boundary moves.
    float topness = smoothstep(0.0, 1.6, p.y);
    vec2 w = p + flow_offset(p, t) * (0.12 + topness * 0.55);

    // Soft column envelope on the WARPED point: narrower toward the top.
    float hh = clamp(w.y * 0.55, 0.0, 1.0);
    float width = mix(0.5, 0.14, hh) * (0.7 + 0.3 * smoothstep(0.0, 0.25, w.y));
    float column = exp(-pow(abs(w.x) / width, 2.0));

    // Vertical fuel: lots at the base, running out with height (on the warped y so
    // the top tongues rise and fall with the flow).
    float fuel = (1.0 - smoothstep(0.1, 1.8, w.y)) + n * 0.35;

    // Continuous gate (NOT a threshold), so the interior is the raw field.
    float gate = column * smoothstep(0.05, 0.5, fuel);
    float heat = n * gate;

    // EAT the edge with the field, height-weighted: subtract a rising amount of
    // (1-field) so low-field pixels near the top drop out -> the outline frays
    // into separate tongues instead of two clean diagonal sides (kills triangle).
    heat -= (1.0 - n) * (0.15 + topness * 0.7) * gate;

    // Soft rim fade kills the background without flattening the body.
    heat *= smoothstep(0.06, 0.4, n * gate + n * 0.2);

    // v04 vein contrast: mid-tones orange/yellow, only true peaks go white.
    heat = pow(clamp(heat, 0.0, 1.0), 0.8) * 1.2;
    heat *= 1.0 + (1.0 - smoothstep(0.0, 0.6, p.y)) * 0.35; // hotter at the root
    return clamp(heat, 0.0, 1.4);
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

// ROUND radiating glow: radial bloom near the base, decoupled from the silhouette.
vec3 round_glow(vec2 p, float n) {
    vec2 src = vec2(0.0, 0.35);
    float d = length((p - src) * vec2(1.0, 0.85));
    float halo = exp(-2.2 * d * d);
    halo *= 0.45 + 0.55 * n;
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

    // Tight bloom on the hottest core only.
    color += vec3(1.0, 0.85, 0.6) * smoothstep(0.8, 1.05, heat) * 0.35;

    color = min(color, vec3(1.0));
    fs_color = vec4(color, 1.0);
}
