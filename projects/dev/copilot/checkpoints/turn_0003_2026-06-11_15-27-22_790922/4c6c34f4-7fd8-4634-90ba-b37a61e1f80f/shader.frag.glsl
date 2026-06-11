#version 460 core

in vec2 vs_uv;
out vec4 fs_color;

uniform float u_time;
uniform float u_aspect;

uniform uint u_text[64];
uniform vec3 u_color1 = vec3(0.95, 0.55, 0.20);
uniform vec3 u_color2 = vec3(0.35, 0.85, 1.00);
uniform float u_char_height = 0.28;
uniform vec2 u_spacing = vec2(0.30, 0.40);
uniform float u_weight = 0.012;

void main() {
    vec2 p = SB_center_uv(vs_uv, u_aspect);

    float ch = SB_text_fit(u_text, u_char_height, u_spacing, vec2(2.0 * u_aspect - 0.18, 1.80));
    ch = max(ch, 0.06);

    vec2 wp = p;
    wp.x += 0.035 * sin(7.0 * p.y + 2.2 * u_time);
    wp.y += 0.020 * sin(9.0 * p.x - 1.6 * u_time);

    float d = SB_sd_text(wp, u_text, ch, u_spacing, u_weight);
    float fill = SB_fill_aa(d);
    float glow = SB_glow(d, 0.05);

    float t = 0.5 + 0.5 * sin(1.8 * u_time + 2.0 * p.y);
    vec3 ink = mix(u_color1, u_color2, t);
    vec3 bg = vec3(0.03, 0.02, 0.06) + 0.05 * vec3(p.y + 0.7);

    vec3 color = bg + ink * fill + ink * 0.30 * glow;
    fs_color = vec4(color, 1.0);
}