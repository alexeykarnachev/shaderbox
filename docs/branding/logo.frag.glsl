#version 460 core

in vec2 vs_uv;
out vec4 fs_color;

uniform float u_time;
uniform float u_aspect;

uniform uint u_text[64];
uniform float u_char_height = 0.08;
uniform vec2 u_spacing = vec2(1.8, 0.4);
uniform float u_weight = 0.008;

uniform float u_wave_thickness = 0.045;
uniform float u_wave_smooth = 0.008;
uniform float u_glow_radius = 0.12;
uniform float u_box_border = 0.02;

uniform float u_underline_width = 0.68;
uniform float u_underline_thickness = 0.0065;
uniform float u_accent_length = 0.08;
uniform float u_accent_thickness = 0.018;

uniform vec3 u_text_color = vec3(0.2, 0.95, 1.0);
uniform vec3 u_glow_color = vec3(1.0, 0.55, 0.2);
uniform vec3 u_bg_top = vec3(0.01, 0.02, 0.06);
uniform vec3 u_bg_bottom = vec3(0.0, 0.0, 0.03);

uniform float u_barrel = 0.15;
uniform float u_scanline_freq = 500.0;
uniform float u_scanline_strength = 0.12;
  uniform vec2 u_barrel_center = vec2(0.0, 0.18);


    void main() {
        vec2 centered_uv = SB_center_uv(vs_uv, u_aspect);
        vec2 p = centered_uv - u_barrel_center;
        float radius_sq = dot(p, p);
        vec2 crt_uv = u_barrel_center + p + p * radius_sq * u_barrel;

        float gradient = smoothstep(-0.9, 1.0, crt_uv.y + 0.15 * sin(u_time * 0.5));

    vec3 background = mix(u_bg_bottom, u_bg_top, gradient);

    vec2 emblem_center = vec2(0.0, 0.25);
    vec2 emblem_uv = crt_uv - emblem_center;
    float box = SB_sd_box(emblem_uv, vec2(0.45, 0.4));
    float rounded_box = SB_op_round(box, 0.08);
    float border_sdf = abs(rounded_box) - u_box_border;
    float border_mask = SB_fill(border_sdf, 0.002);
    float border_glow = SB_glow(border_sdf, u_glow_radius * 0.6);

    vec2 wave_uv = emblem_uv;
    float wave = sin(wave_uv.x * 5.5 + u_time * 2.5) * 0.24;
    float wave_line = abs(wave_uv.y - wave) - u_wave_thickness;
    float wave_mask = SB_fill(wave_line, u_wave_smooth);
    wave_mask *= step(rounded_box, 0.0);
    float wave_glow = SB_glow(wave_line, u_glow_radius);
    wave_glow *= step(rounded_box, 0.0);

    vec2 accent_horiz = vec2(u_accent_length * 0.5, u_accent_thickness * 0.5);
    vec2 accent_vert = vec2(u_accent_thickness * 0.5, u_accent_length * 0.5);
    vec2 top_left = vec2(-0.45 - u_accent_thickness * 0.5, 0.4 - u_accent_length * 0.5);
    vec2 top_right = vec2(0.45 + u_accent_thickness * 0.5, 0.4 - u_accent_length * 0.5);
    vec2 bottom_left = vec2(-0.45 - u_accent_thickness * 0.5, -0.4 + u_accent_length * 0.5);
    vec2 bottom_right = vec2(0.45 + u_accent_thickness * 0.5, -0.4 + u_accent_length * 0.5);

    float accent_sdf = 1e5;
    accent_sdf = min(accent_sdf, SB_sd_box(crt_uv - (emblem_center + top_left), accent_vert));
    accent_sdf = min(accent_sdf, SB_sd_box(crt_uv - (emblem_center + top_left + vec2(u_accent_length * 0.5, 0.0)), accent_horiz));
    accent_sdf = min(accent_sdf, SB_sd_box(crt_uv - (emblem_center + top_right + vec2(-u_accent_length * 0.5, 0.0)), accent_horiz));
    accent_sdf = min(accent_sdf, SB_sd_box(crt_uv - (emblem_center + top_right), accent_vert));
    accent_sdf = min(accent_sdf, SB_sd_box(crt_uv - (emblem_center + bottom_left), accent_vert));
    accent_sdf = min(accent_sdf, SB_sd_box(crt_uv - (emblem_center + bottom_left + vec2(u_accent_length * 0.5, 0.0)), accent_horiz));
    accent_sdf = min(accent_sdf, SB_sd_box(crt_uv - (emblem_center + bottom_right + vec2(-u_accent_length * 0.5, 0.0)), accent_horiz));
    accent_sdf = min(accent_sdf, SB_sd_box(crt_uv - (emblem_center + bottom_right), accent_vert));

    float accent_mask = SB_fill(accent_sdf, 0.002);
    float accent_glow = SB_glow(accent_sdf, u_glow_radius * 0.35);

    vec2 text_center = vec2(0.0, -0.45);
    vec2 text_uv = crt_uv - text_center;
    float text_sdf = SB_sd_text(text_uv, u_text, u_char_height, u_spacing, u_weight);
    float text_mask = SB_fill(text_sdf, 0.003);
    float text_glow = SB_glow(text_sdf, u_glow_radius * 0.35);

    vec2 underline_center = text_center + vec2(0.0, -u_char_height * 0.9);
    float underline_box = SB_sd_box(crt_uv - underline_center, vec2(u_underline_width * 0.5, u_underline_thickness * 0.5));
    float underline_mask = SB_fill(underline_box, 0.001);
    float underline_glow = SB_glow(underline_box, u_glow_radius * 0.45);

    vec3 neon = u_text_color;
    vec3 border_color = neon;
    vec3 wave_color = neon;
    vec3 text_color = neon;
    vec3 accent_color = neon;
    vec3 underline_color = neon;

    vec3 glow = vec3(0.0);
    glow += u_glow_color * border_glow * 0.6;
    glow += u_glow_color * wave_glow * 1.4;
    glow += u_glow_color * accent_glow * 0.25;
    glow += u_glow_color * text_glow * 0.3;
    glow += u_glow_color * underline_glow * 0.35;

    vec3 base = background;
    base = mix(base, border_color, border_mask);
    base = mix(base, wave_color, wave_mask);
    base = mix(base, accent_color, accent_mask);
    base = mix(base, text_color, text_mask);
    base = mix(base, underline_color, underline_mask);
    base += glow * 0.35;
    base = clamp(base, 0.0, 1.0);

    float vignette = smoothstep(0.55, 1.0, length(crt_uv));
    base *= mix(1.0, 0.7, vignette);

    float scanline = mix(1.0, sin(vs_uv.y * u_scanline_freq) * 0.5 + 0.5, u_scanline_strength);
    base *= scanline;

        fs_color = vec4(base, 1.0);
}

