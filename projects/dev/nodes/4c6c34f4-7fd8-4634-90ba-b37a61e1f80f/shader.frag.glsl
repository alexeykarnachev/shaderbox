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

    // Measure line lengths from u_text (up to 8 lines).
    int line_len[8];
    for (int l = 0; l < 8; ++l) line_len[l] = 0;

    int line_count = 1;
    int line_idx = 0;
    for (int i = 0; i < 64; ++i) {
        uint cp = u_text[i];
        if (cp == 0u) break;
        if (cp == 10u) {
            line_idx = min(line_idx + 1, 7);
            line_count = max(line_count, line_idx + 1);
        } else {
            line_len[line_idx] += 1;
        }
    }

    // Reading tempo: line duration depends on symbol count.
    float chars_per_sec = 9.0;
    float line_gap = 0.22;
    float line_start[8];
    float line_dur[8];
    float total_cycle = 0.0;
    for (int l = 0; l < 8; ++l) {
        line_start[l] = total_cycle;
        float d = max(0.65, float(line_len[l]) / chars_per_sec);
        line_dur[l] = d;
        if (l < line_count) total_cycle += d + line_gap;
    }
    total_cycle = max(total_cycle, 1.0);

        // Compress/stretch full cycle to exactly 3 seconds.
    float target_cycle = 3.0;
    float t_cycle = mod(u_time * (total_cycle / target_cycle), total_cycle);


    float d_text = 1e5;
    float half_ch = 0.5 * ch;
    float w_local = u_weight / max(half_ch, 1e-5);

    line_idx = 0;
    for (int i = 0; i < 64; ++i) {
        uint cp = u_text[i];
        if (cp == 0u) break;
        if (cp == 10u) {
            line_idx = min(line_idx + 1, 7);
            continue;
        }

        float p_line = clamp((t_cycle - line_start[line_idx]) / line_dur[line_idx], 0.0, 1.0);
        p_line = SB_ease_in_out(p_line);

        float side = (mod(float(line_idx), 2.0) < 0.5) ? -1.0 : 1.0;
        float travel = side * (2.2 * u_aspect) * (1.0 - p_line);

        vec2 c = SB_text_char_center(u_text, i, ch, u_spacing);
        c.x += travel;

        vec2 g = (p - c) / half_ch;
        float dc = SB_sd_char(g, cp, w_local) * half_ch;
        d_text = min(d_text, dc);
    }

    float fill = SB_fill_aa(d_text);
    float glow = SB_glow(d_text, 0.05);

    float t = 0.5 + 0.5 * sin(1.8 * u_time + 1.7 * p.y);
    vec3 ink = mix(u_color1, u_color2, t);
    vec3 bg = vec3(0.03, 0.02, 0.06) + 0.05 * vec3(p.y + 0.7);

    vec3 color = bg + ink * fill + ink * 0.28 * glow;
    fs_color = vec4(color, 1.0);
}