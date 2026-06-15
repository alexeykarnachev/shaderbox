#version 460 core

in vec2 vs_uv;

uniform float u_time;
uniform float u_aspect;

uniform vec3 u_bg_color = vec3(0.03, 0.04, 0.07);

uniform vec2 u_ball_pos;
uniform vec2 u_ball_vel;
uniform float u_ball_radius;
uniform vec2 u_trail[24];
uniform float u_trail_count;
uniform float u_paddle_left_y;
uniform float u_paddle_right_y;
uniform float u_paddle_half_height;
uniform float u_paddle_half_width;
uniform float u_score_left;
uniform float u_score_right;
uniform float u_hit_flash;

out vec4 fs_color;

float sd_box(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float fill(float d, float s) {
    return smoothstep(s, 0.0, d);
}

void main() {
    vec2 p = (vs_uv - 0.5) * 2.0;

    // Animated background energy field.
    float radial = length(p);
    float ang = atan(p.y, p.x);
    float wave = 0.5 + 0.5 * sin(8.0 * radial - 2.0 * u_time + 3.0 * ang);
    vec3 bg_fx = vec3(0.02, 0.08, 0.14) * wave;
    vec3 col = u_bg_color + bg_fx;

    // Center dashed line with slight pulse.
    float stripe = step(0.5, fract((p.y + 1.0) * 8.0 + 0.5 * sin(u_time * 1.7)));
    float center_line = fill(abs(p.x) - 0.008, 0.01) * stripe;
    col += vec3(0.15, 0.2, 0.3) * center_line;

    // Arena vignette.
    float vignette = smoothstep(1.45, 0.2, radial);
    col *= 0.72 + 0.28 * vignette;

    // Paddles.
    vec2 lp = p - vec2(-0.92, u_paddle_left_y);
    vec2 rp = p - vec2(0.92, u_paddle_right_y);
    float d_left = sd_box(lp, vec2(u_paddle_half_width, u_paddle_half_height));
    float d_right = sd_box(rp, vec2(u_paddle_half_width, u_paddle_half_height));
    float m_left = fill(d_left, 0.01);
    float m_right = fill(d_right, 0.01);
    float paddle_glow = exp(-24.0 * max(min(d_left, d_right), 0.0));

    vec3 paddle_col = vec3(0.88, 0.92, 1.0);
    col = mix(col, paddle_col, max(m_left, m_right));
    col += vec3(0.2, 0.35, 0.8) * paddle_glow * 0.2;

    // Ball.
    float d_ball = length(p - u_ball_pos) - u_ball_radius;
    float m_ball = fill(d_ball, 0.012);
    vec3 vel_tint = vec3(0.8, 0.4, 0.2) + 0.5 * vec3(abs(u_ball_vel.x), abs(u_ball_vel.y), 0.2);
    col = mix(col, vel_tint, m_ball);

    // Trail from script-provided history.
    vec3 trail_col = vec3(0.25, 0.75, 1.25);
    for (int i = 0; i < 24; i++) {
        float fi = float(i);
                float trail_on = step(fi + 0.5, u_trail_count);
        float age = fi / 23.0;                    // 0 = newest, 1 = oldest
        float r = mix(u_ball_radius * 0.9, u_ball_radius * 0.2, age);
        float d = length(p - u_trail[i]) - r;
        float dot_m = fill(d, 0.01) * (1.0 - age) * trail_on;
        float halo = exp(-26.0 * max(d, 0.0)) * (1.0 - age) * trail_on;
        col += trail_col * (0.12 * dot_m + 0.14 * halo);
    }

    // Hit/impact glow.
    float glow = exp(-18.0 * max(d_ball, 0.0)) * u_hit_flash;
    col += vec3(1.0, 0.5, 0.2) * glow;

    // Shock ring on hit.
    float ring_r = 0.1 + (1.0 - u_hit_flash) * 0.35;
    float ring = fill(abs(length(p - u_ball_pos) - ring_r) - 0.01, 0.01) * u_hit_flash;
    col += vec3(1.0, 0.7, 0.3) * ring * 0.45;

    // Top score bars (0..9).
    float score_l = clamp(u_score_left, 0.0, 9.0);
    float score_r = clamp(u_score_right, 0.0, 9.0);

    for (int i = 0; i < 9; i++) {
        float fi = float(i);
        float on_l = step(fi + 0.5, score_l);
        float on_r = step(fi + 0.5, score_r);

        vec2 cell_l = p - vec2(-0.35 + fi * 0.05, 0.9);
        vec2 cell_r = p - vec2(0.35 - fi * 0.05, 0.9);

        float bar_l = fill(sd_box(cell_l, vec2(0.018, 0.03)), 0.008) * on_l;
        float bar_r = fill(sd_box(cell_r, vec2(0.018, 0.03)), 0.008) * on_r;

        col += vec3(0.95, 0.85, 0.35) * (bar_l + bar_r);
    }

    // Subtle scanline/post effect.
    float scan = 0.96 + 0.04 * sin(vs_uv.y * 900.0 + u_time * 8.0);
    col *= scan;

    // Mild tonemap-ish compression.
    col = col / (1.0 + 0.4 * col);

    fs_color = vec4(col, 1.0);
}
