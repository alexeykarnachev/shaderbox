// Multi-line text layout over the glyph set. SB_sd_text returns a SIGNED distance
// (negative inside the ink) like every SB_sd_* source — render it with SB_fill /
// SB_glow or transform it with SB_op_* first. The text block is CENTERED at the
// origin (each line centered too), so text never starts off-screen: place it by
// subtracting a position from uv, scale it by char_height. All distances are in
// the same units as uv. Text arrays are the engine's `uniform uint u_text[64];`
// convention (0-terminated, 10 = newline, 32 = space).

/// Size (width, height) of the text block SB_sd_text draws, in uv units.
/// Use it to fit text: e.g. scale char_height down if size.x exceeds the view.
vec2 SB_text_size(uint text[64], float char_height, vec2 spacing) {
    float s = 0.5 * char_height;
    float adv = (1.0 + spacing.x) * s;
    float max_chars = 0.0;
    float chars = 0.0;
    int n_rows = 1;
    for (int i = 0; i < 64; ++i) {
        uint cp = text[i];
        if (cp == 0u) break;
        if (cp == 10u) {
            max_chars = max(max_chars, chars);
            chars = 0.0;
            if (n_rows < 64) n_rows += 1;
            continue;
        }
        chars += 1.0;
    }
    max_chars = max(max_chars, chars);
    float width = max_chars > 0.0 ? max_chars * adv - spacing.x * s : 0.0;
    float height = float(n_rows) * char_height + float(n_rows - 1) * spacing.y * char_height;
    return vec2(width, height);
}

/// Largest char_height (capped at the given one) that keeps the text block's CELLS
/// within max_size (width, height in uv units). Pass the visible extent minus a
/// margin (e.g. vec2(2.0 * u_aspect - 0.2, 1.8) for SB_center_uv coords) — stroke
/// weight and diacritics overshoot the cells slightly, the margin absorbs them.
float SB_text_fit(uint text[64], float char_height, vec2 spacing, vec2 max_size) {
    vec2 size = SB_text_size(text, char_height, spacing);
    float k = min(1.0, min(max_size.x / max(size.x, 0.000001),
                           max_size.y / max(size.y, 0.000001)));
    return char_height * k;
}

/// Center (uv units) of glyph cell `index` (i.e. text[index]) in the SAME centered
/// block frame as SB_sd_text with identical char_height/spacing. For per-char
/// effects (jitter, wave, per-char rotation/color), loop chars yourself:
///   vec2 c = SB_text_char_center(text, i, ch, sp) + your_offset;
///   d = min(d, SB_sd_char((uv - c) / (0.5*ch), text[i], w_uv / (0.5*ch)) * (0.5*ch));
/// (SB_sd_char takes a LOCAL weight — divide your uv weight by 0.5*ch, as above, or
/// strokes come out a different thickness than SB_sd_text's.) Draw EITHER the whole
/// block via SB_sd_text OR per-char copies — both at once doubles every letter.
/// Newline/terminator cells return 1e5. PERF: two text scans per call — bound your
/// per-char loop at the REAL text length (e.g. i < 16), never a full 64.
vec2 SB_text_char_center(uint text[64], int index, float char_height, vec2 spacing) {
    float s = 0.5 * char_height;
    float adv = (1.0 + spacing.x) * s;
    float line_step = (1.0 + spacing.y) * char_height;

    float row_chars[64];
    int n_rows = 1;
    {
        float chars = 0.0;
        for (int i = 0; i < 64; ++i) {
            uint cp = text[i];
            if (cp == 0u) break;
            if (cp == 10u) {
                row_chars[n_rows - 1] = chars;
                chars = 0.0;
                if (n_rows < 64) n_rows += 1;
                continue;
            }
            chars += 1.0;
        }
        row_chars[n_rows - 1] = chars;
    }

    float block_h = float(n_rows) * char_height + float(n_rows - 1) * spacing.y * char_height;
    float top_y = 0.5 * block_h - 0.5 * char_height;

    int row = 0;
    float col = 0.0;
    for (int i = 0; i < 64; ++i) {
        uint cp = text[i];
        if (cp == 0u) break;
        if (cp == 10u) {
            if (row < 63) row += 1;
            col = 0.0;
            continue;
        }
        if (i == index) {
            float row_w = row_chars[row] * adv - spacing.x * s;
            return vec2(-0.5 * row_w + 0.5 * s + col * adv,
                        top_y - float(row) * line_step);
        }
        col += 1.0;
    }
    return vec2(100000.0);
}

/// SIGNED distance (uv units, negative inside the ink) to a text block centered
/// at the origin. char_height = glyph height in uv units (glyph width is half of
/// that) — compute it with SB_text_fit so arbitrary text never clips. spacing =
/// extra gap between chars (x, fraction of glyph width) and lines (y, fraction of
/// char_height) — (0,0) packs cells edge-to-edge, use x 0.25..0.45. weight = stroke
/// half-width in uv units (~0.05 * char_height regular, more = bold; keep weight +
/// spacing in balance or neighbor strokes merge). For a typewriter reveal replace
/// hidden chars with SPACES (32u advances but draws nothing) — truncating the
/// array re-centers the block every frame. Keep the tail BEYOND your text 0u
/// (terminator) — space-padding to 64 makes the layout treat all 64 cells as one
/// huge line (tiny, off-center text).
/// Typical: SB_fill(SB_sd_text(p, u_text, ch, vec2(0.35, 0.4), 0.05 * ch), 0.005).
float SB_sd_text(vec2 uv, uint text[64], float char_height, vec2 spacing, float weight) {
    float s = 0.5 * char_height;
    float adv = (1.0 + spacing.x) * s;
    float line_step = (1.0 + spacing.y) * char_height;

    float row_chars[64];
    int n_rows = 1;
    {
        float chars = 0.0;
        for (int i = 0; i < 64; ++i) {
            uint cp = text[i];
            if (cp == 0u) break;
            if (cp == 10u) {
                row_chars[n_rows - 1] = chars;
                chars = 0.0;
                if (n_rows < 64) n_rows += 1;
                continue;
            }
            chars += 1.0;
        }
        row_chars[n_rows - 1] = chars;
    }

    float block_h = float(n_rows) * char_height + float(n_rows - 1) * spacing.y * char_height;
    float top_y = 0.5 * block_h - 0.5 * char_height;

    float d = 100000.0;
    int row = 0;
    float col = 0.0;
    for (int i = 0; i < 64; ++i) {
        uint cp = text[i];
        if (cp == 0u) break;
        if (cp == 10u) {
            if (row < 63) row += 1;
            col = 0.0;
            continue;
        }
        float row_w = row_chars[row] * adv - spacing.x * s;
        vec2 center = vec2(-0.5 * row_w + 0.5 * s + col * adv,
                           top_y - float(row) * line_step);
        d = min(d, sbt_char_skel((uv - center) / s, cp));
        col += 1.0;
    }
    return d * s - weight;
}
