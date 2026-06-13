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

float ease_out_back(float x) {
    float c1 = 1.70158;
    float c3 = c1 + 1.0;
    float t = x - 1.0;
    return 1.0 + c3 * t * t * t + c1 * t * t;
}

void main() {
    vec2 p = SB_center_uv(vs_uv, u_aspect);

    float ch = SB_text_fit(u_text, u_char_height, u_spacing, vec2(2.0 * u_aspect - 0.18, 1.80));
    ch = max(ch, 0.06);

    // Measure line lengths and word counts from u_text (up to 8 lines).
    int line_len[8];
    int word_count[8];
    for (int l = 0; l < 8; ++l) {
        line_len[l] = 0;
        word_count[l] = 0;
    }

    int line_count = 1;
    int scan_line = 0;
    int in_word = 0;
    for (int i = 0; i < 64; ++i) {
        uint cp = u_text[i];
        if (cp == 0u) break;

        if (cp == 10u) {
            in_word = 0;
            scan_line = min(scan_line + 1, 7);
            line_count = max(line_count, scan_line + 1);
            continue;
        }

        line_len[scan_line] += 1;

        bool is_space = (cp == 32u);
        if (!is_space && in_word == 0) {
            word_count[scan_line] += 1;
            in_word = 1;
        }
        if (is_space) {
            in_word = 0;
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

    // Active line state for color/timing accents.
    int active_line = 0;
    for (int l = 0; l < 8; ++l) {
        if (l < line_count && t_cycle >= line_start[l]) {
            active_line = l;
        }
    }
    float active_phase = clamp((t_cycle - line_start[active_line]) / max(line_dur[active_line], 1e-4), 0.0, 1.0);

    float d_text = 1e5;
    float d_echo = 1e5;

    float half_ch = 0.5 * ch;
    float w_local = u_weight / max(half_ch, 1e-5);

    int line_idx = 0;
    int word_idx = 0;
    int draw_in_word = 0;

    for (int i = 0; i < 64; ++i) {
        uint cp = u_text[i];
        if (cp == 0u) break;

        if (cp == 10u) {
            line_idx = min(line_idx + 1, 7);
            word_idx = 0;
            draw_in_word = 0;
            continue;
        }

        bool is_space = (cp == 32u);
        if (!is_space && draw_in_word == 0) {
            draw_in_word = 1;
        }

        vec2 c0 = SB_text_char_center(u_text, i, ch, u_spacing);

        float line_phase = clamp((t_cycle - line_start[line_idx]) / line_dur[line_idx], 0.0, 1.0);
        float char_delay = 0.03 * c0.x / max(u_aspect, 1e-4);
        float word_delay = 0.07 * float(word_idx);
        float p_line = clamp((line_phase - char_delay - word_delay) / 0.90, 0.0, 1.0);

        float enter = ease_out_back(p_line);
        float settle = SB_ease_in_out(p_line);

        float side = (mod(float(line_idx), 2.0) < 0.5) ? -1.0 : 1.0;
        float travel = side * (2.3 * u_aspect) * (1.0 - enter);

        vec2 c = c0;
        c.x += travel;

        float wobble_amp = (1.0 - settle) * (0.07 + 0.02 * float(line_idx));
        c.y += wobble_amp * sin(9.0 * c0.x + 7.5 * p_line + 2.0 * float(line_idx));

        float ang = side * (1.0 - settle) * (0.45 + 0.20 * sin(6.0 * c0.x + 4.0 * p_line));
        vec2 g = SB_rotate((p - c) / half_ch, -ang);

        float pulse = 0.75 + 0.25 * sin(12.0 * p_line + 6.0 * c0.x + 0.8 * float(line_idx));

        int last_word = max(word_count[line_idx] - 1, 0);
        float is_last_word = (!is_space && word_idx == last_word) ? 1.0 : 0.0;
        float word_emph = 1.0 + is_last_word * (0.30 + 0.20 * sin(10.0 * p_line));

        float dc = SB_sd_char(g, cp, w_local * pulse * word_emph) * half_ch;
        d_text = min(d_text, dc);

        // Step 3: depth/parallax echo (delayed, offset duplicate layer).
        float lag = clamp(p_line - 0.12, 0.0, 1.0);
        float enter_echo = ease_out_back(lag);
        float travel_echo = side * (2.0 * u_aspect) * (1.0 - enter_echo);

        vec2 c_echo = c0;
        c_echo.x += travel_echo;
        c_echo += vec2(-0.030 * side, -0.018 - 0.006 * sin(5.0 * p_line + float(line_idx)));

        float ang_echo = 0.6 * ang;
        vec2 g_echo = SB_rotate((p - c_echo) / half_ch, -ang_echo);
        float de = SB_sd_char(g_echo, cp, w_local * 0.95) * half_ch;
        d_echo = min(d_echo, de);

        if (is_space && draw_in_word == 1) {
            draw_in_word = 0;
            word_idx += 1;
        }
    }

    float fill = SB_fill_aa(d_text);
    float glow = SB_glow(d_text, 0.05);

    float echo_fill = SB_fill_aa(d_echo);
    float echo_glow = SB_glow(d_echo, 0.08);

    // Step 2: emotional color beats (line tint + arrival flash).
    float t = 0.5 + 0.5 * sin(1.8 * u_time + 1.7 * p.y);
    vec3 ink_base = mix(u_color1, u_color2, t);

    float line_band = float(active_line) / max(float(line_count - 1), 1.0);
    vec3 line_tint = SB_palette_sunset(0.15 + 0.70 * line_band);

    float arrive_flash = exp(-18.0 * abs(active_phase - 0.12));
    float beat = 0.5 + 0.5 * sin(6.28318 * (active_phase - 0.08));
    float flash = 0.30 * arrive_flash * beat;

    vec3 ink = mix(ink_base, line_tint, 0.28) + flash * line_tint;

    vec3 bg = vec3(0.03, 0.02, 0.06) + 0.05 * vec3(p.y + 0.7);
    bg += 0.03 * arrive_flash * vec3(0.3, 0.2, 0.45);

    vec3 echo_col = mix(vec3(0.06, 0.10, 0.18), ink, 0.28);
    vec3 color = bg;
    color += echo_col * (0.30 * echo_fill + 0.18 * echo_glow);
    color += ink * fill + ink * (0.28 + 0.20 * arrive_flash) * glow;

    fs_color = vec4(color, 1.0);
}
