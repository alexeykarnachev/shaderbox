#version 460 core

in vec2 vs_uv; // Coordinate of the current pixel to be shaded

uniform float u_time;   // Time (s) since the application started
uniform float u_aspect; // Aspect ratio of the canvas (width / height)
// uniform vec2 u_resolution;  // Resolution of the canvas (width, height)

// --- Live-tunable knobs (auto-generated ImGui controls) ---------------------
uniform float u_flicker_amp = 0.28;   // brightness swing amplitude (0 = steady, 0.5 = wild)
uniform float u_flicker_speed = 1.0;  // flicker rate multiplier
uniform float u_smoke_strength = 0.75; // smoke opacity/brightness (0 = off)
uniform float u_smoke_top = 2.1;      // height where smoke fully dissipates
uniform float u_glow_strength = 0.5;  // round-glow brightness
uniform float u_glow_radius = 2.2;    // round-glow falloff (lower = wider aura)

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
    float s = u_flicker_speed;
    float global = 0.6 * sin(t * 13.0 * s) + 0.4 * sin(t * 4.3 * s + 1.1); // [-1,1]
    float spatial = sin(t * 17.0 * s + p.y * 5.0);                          // [-1,1]
    float f = global + 0.4 * spatial; // global dominates so the swing reads
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

// --- Volumetric smoke -------------------------------------------------------
// 3D value noise from the lib's vec3 hash (no SB_ vec3 noise exists): trilinear
// blend of the 8 lattice-corner hashes. Gives the smoke a real third dimension to
// march through.
float noise3(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f); // smoothstep weights
    float n000 = SB_hash31(i + vec3(0.0, 0.0, 0.0));
    float n100 = SB_hash31(i + vec3(1.0, 0.0, 0.0));
    float n010 = SB_hash31(i + vec3(0.0, 1.0, 0.0));
    float n110 = SB_hash31(i + vec3(1.0, 1.0, 0.0));
    float n001 = SB_hash31(i + vec3(0.0, 0.0, 1.0));
    float n101 = SB_hash31(i + vec3(1.0, 0.0, 1.0));
    float n011 = SB_hash31(i + vec3(0.0, 1.0, 1.0));
    float n111 = SB_hash31(i + vec3(1.0, 1.0, 1.0));
    float nx00 = mix(n000, n100, u.x);
    float nx10 = mix(n010, n110, u.x);
    float nx01 = mix(n001, n101, u.x);
    float nx11 = mix(n011, n111, u.x);
    float nxy0 = mix(nx00, nx10, u.y);
    float nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z); // [0, 1]
}

// 3D fbm density: a synthetic z gives the 2D canvas layers to march through, so
// the smoke has DEPTH instead of being one flat noise lookup. Scrolls upward (and
// slightly inward in z) over time -> rising, evolving puffs.
float smoke_density(vec3 q, float t) {
    q += vec3(0.0, -t * 0.5, t * 0.35); // rise + drift through depth
    float f = 0.0, amp = 0.5, freq = 1.0;
    for (int i = 0; i < 5; i++) {
        f += amp * noise3(q * freq);
        freq *= 2.05;
        amp *= 0.5;
    }
    // Lift off the flame body (~y 0.9) and DISSIPATE upward: density thins with
    // height so it trails off into nothing instead of pooling into a disc near the
    // ceiling. The column WIDENS as it rises (smoke spreads) but gets sparser.
    float startY = 0.85;
    float col = exp(-pow(abs(q.x) / (0.4 + 0.35 * max(q.y - startY, 0.0)), 2.0));
    float rise = smoothstep(startY, startY + 0.35, q.y);   // fade IN just above the flame
    float thin = 1.0 - smoothstep(startY + 0.2, u_smoke_top, q.y); // thin OUT to u_smoke_top
    float band = rise * thin * thin;                       // squared -> faster dissipate
    return clamp(f * col * band - 0.28, 0.0, 1.0); // -0.28 erodes hard into ragged wisps
}

// Beer-Lambert self-shadow: a few steps toward an upper light; dense regions above
// darken the point below, giving the volume soft internal shadow (not flat gray).
float smoke_light(vec3 pos, float t) {
    vec3 ldir = normalize(vec3(0.3, 1.0, 0.2));
    float dsum = 0.0;
    for (int i = 0; i < 4; i++) {
        pos += ldir * 0.12;
        dsum += smoke_density(pos, t);
    }
    return exp(-dsum * 1.4); // transmittance toward the light
}

// Front-to-back raymarch through the density (the GM-shaders / Heckel recipe):
// accumulate colour*alpha with Beer-Lambert absorption -> volumetric depth.
vec4 smoke_volume(vec2 p, float t) {
    vec3 ro = vec3(p, -0.6);          // start behind the canvas plane
    vec3 rd = vec3(0.0, 0.0, 1.0);    // march into the page
    vec4 acc = vec4(0.0);
    float depth = 0.0;
    for (int i = 0; i < 28; i++) {
        vec3 pos = ro + rd * depth;
        float dens = smoke_density(pos, t);
        if (dens > 0.001) {
            float lit = smoke_light(pos, t);
            // SELF-LIT (black bg = nothing to occlude, so it must emit to read):
            // warm haze lit by the fire just above the tips, cooling to dim gray and
            // DARKENING as it rises so it fades into the background, not a glow.
            vec3 warm = vec3(0.55, 0.3, 0.16);
            vec3 cool = vec3(0.14, 0.14, 0.16);
            float fade = 1.0 - smoothstep(0.95, 2.0, pos.y);   // dim with height
            vec3 col = mix(cool, warm, fade) * (0.4 + 0.6 * lit) * (0.35 + 0.65 * fade);
            float a = dens * u_smoke_strength;
            acc.rgb += (1.0 - acc.a) * col * a; // front-to-back over
            acc.a += (1.0 - acc.a) * a;
            if (acc.a > 0.97) break;
        }
        depth += 0.12;
    }
    return acc;
}

// ROUND radiating glow (v05): radial bloom near the base, decoupled from silhouette.
vec3 round_glow(vec2 p, float n) {
    vec2 src = vec2(0.0, 0.35);
    float d = length((p - src) * vec2(1.0, 0.85));
    float halo = exp(-u_glow_radius * d * d);
    halo *= 0.45 + 0.55 * n;
    return vec3(1.0, 0.38, 0.1) * halo * u_glow_strength;
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

    float n = interior_field(p, u_time);
    float heat = flame_heat(p, u_time, n) * flicker(p, u_time);

    vec3 color = fire_ramp(heat);
    color += round_glow(p, n);

    // Volumetric smoke composited OVER the fire+glow, BELOW the embers/core bloom.
    vec4 sm = smoke_volume(p, u_time);
    color = color * (1.0 - sm.a) + sm.rgb;

    color += embers(p, u_time);
    color += vec3(1.0, 0.85, 0.6) * smoothstep(0.8, 1.05, heat) * 0.35;

    color = min(color, vec3(1.0));
    fs_color = vec4(color, 1.0);
}
