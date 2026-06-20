#version 460 core

in vec2 vs_uv; // Coordinate of the current pixel to be shaded

uniform float u_time;   // Time (s) since the application started
uniform float u_aspect; // Aspect ratio of the canvas (width / height)
// uniform vec2 u_resolution;  // Resolution of the canvas (width, height)

// Step reveal controls (0..1): keep at 1.0 for full final effect.
uniform float u_show_shape = 1.0;
uniform float u_show_glow = 1.0;
uniform float u_show_embers = 1.0;
uniform float u_step_mix = 1.0;


out vec4 fs_color;

vec2 fire_space(vec2 uv, float aspect) {
    vec2 p = SB_center_uv(uv, aspect);
    p.y += 0.9; // place origin near the bottom
    return p;
}

float fire_heat_base(vec2 p) {
    float heat = exp(-1.7 * p.y);
    heat *= exp(-2.2 * abs(p.x));
    return heat;
}

float fire_heat_distorted(vec2 p, float t) {
    // Stronger distortion toward the top, calmer near the base.
    float top_mask = smoothstep(0.0, 1.1, p.y);

    // Scroll noise upward over time.
    vec2 n_uv = vec2(p.x * 2.2, p.y * 1.8 - t * 1.6);
    float n = SB_fbm(n_uv, 5); // [0..1]

    // Center to [-1..1] and use it to bend the flame body sideways.
    float warp = (n * 2.0 - 1.0) * 0.30 * top_mask;
    vec2 q = p;
    q.x += warp;

    // Evaluate base heat in the warped space.
    float heat = fire_heat_base(q);

    // Add slight breakup in intensity to form tongues.
    heat *= mix(0.82, 1.30, n) * (0.88 + 0.12 * top_mask);
    return heat;
}

float fire_shape_mask(vec2 p) {
    // Round body + pointed cap for a more natural flame silhouette.
    float d_body = SB_sd_circle(p - vec2(0.0, 0.25), 0.34);
    float d_mid  = SB_sd_circle(p - vec2(0.0, 0.62), 0.24);
    float d_tip  = SB_sd_circle(p - vec2(0.0, 1.02), 0.12);

    float d = SB_op_smooth_union(d_body, d_mid, 0.20);
    d = SB_op_smooth_union(d, d_tip, 0.16);

    float mask = SB_fill(d, 0.08);

    // Keep the flame rooted near the base and fade the very top a bit.
    float base_gate = smoothstep(-0.08, 0.08, p.y);
    float top_fade = 1.0 - smoothstep(1.18, 1.55, p.y);

    return mask * base_gate * top_fade;
}

vec3 fire_palette(float heat) {
    vec3 cool = vec3(0.02, 0.01, 0.03);
    vec3 hot = vec3(1.0, 0.45, 0.05);
    return mix(cool, hot, clamp(heat, 0.0, 1.0));
}

float ember_field(vec2 p, float t) {
    // Rising ember particles on a coarse moving grid.
    vec2 g = vec2(p.x * 7.0, p.y * 9.0 - t * 2.4);
    vec2 id = floor(g);
    vec2 f = fract(g) - 0.5;

    float emb = 0.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            vec2 cid = id + vec2(i, j);
            float rnd = SB_hash21(cid);
            vec2 off = vec2(SB_hash21(cid + 13.1) - 0.5, SB_hash21(cid + 27.7) - 0.5);
            vec2 dp = f - vec2(i, j) - off * vec2(0.45, 0.35);

            float d = length(dp);
            float spark = exp(-58.0 * d * d);
            spark *= smoothstep(0.70, 1.0, rnd); // only some cells emit embers
            emb += spark;
        }
    }

    // Mostly visible around/above upper flame body.
    float zone = smoothstep(0.20, 1.00, p.y) * (1.0 - smoothstep(1.35, 1.85, p.y));
    emb *= zone;
    return emb;
}

void main() {
    vec2 p = fire_space(vs_uv, u_aspect);

    // Step 2: animate with fbm-based lateral distortion + intensity breakup.
    float heat = fire_heat_distorted(p, u_time);

    // Global blend for tutorial reveal controls.
    float step_mix = clamp(u_step_mix, 0.0, 1.0);
    float show_shape = mix(1.0, clamp(u_show_shape, 0.0, 1.0), step_mix);
    float show_glow = mix(1.0, clamp(u_show_glow, 0.0, 1.0), step_mix);
    float show_embers = mix(1.0, clamp(u_show_embers, 0.0, 1.0), step_mix);

    // Step 3b: shape the flame without hard-cutting all outer energy.
    float shape = fire_shape_mask(p);
    float shaped_heat = heat * mix(0.45, 1.0, shape);
    heat = mix(heat, shaped_heat, show_shape);

    // Step 5: subtle temporal flicker + rising ember specks.
    float flicker = 0.92 + 0.08 * SB_tri_wave(u_time * 3.4 + p.y * 6.0);
    heat *= flicker;

    // Step 4: brighter core and soft outer glow.
    float core = smoothstep(0.62, 1.08, heat) * show_glow;
    float d_heat = 0.38 - heat;       // pseudo-SDF from heat field
    float halo = SB_glow(d_heat, 0.22) * show_glow;
    float embers = ember_field(p, u_time) * show_embers;

    vec3 base_col = fire_palette(heat);
    vec3 core_col = vec3(1.0, 0.92, 0.70) * core;
    vec3 glow_col = vec3(1.0, 0.42, 0.06) * halo * 0.35;
    vec3 ember_col = vec3(1.0, 0.75, 0.30) * embers * 0.85;

    vec3 color = base_col + core_col + glow_col + ember_col;
    color = min(color, vec3(1.0));

    fs_color = vec4(color, 1.0);
}
