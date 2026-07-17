#version 460 core

in vec2 vs_uv; // Coordinate of the current pixel to be shaded

uniform float u_time;   // Time (s) since the application started
uniform float u_aspect; // Aspect ratio of the canvas (width / height)
// uniform vec2 u_resolution;  // Resolution of the canvas (width, height)

// --- Live-tunable knobs (seeded from v11 tuning) ----------------------------
uniform float u_flicker_amp = 0.32;
uniform float u_flicker_speed = 1.17;
uniform float u_glow_strength = 0.79;
uniform float u_glow_radius = 1.73;
uniform float u_depth = 4.49;

// Global scale of the WHOLE composition (flame + glow + embers + depth), applied
// in flame space so no tuned flame param changes. 1 = as tuned; >1 = shrink
// (bigger divisor => content zooms out). Stays rooted at the bottom (y=0 base).
uniform float u_scale = 1.0;

// DEBUG: 0 = autoplay the timed reveal; 1..9 freezes the build at that step
// (e.g. 3 = steps 1-3 fully shown, 4+ hidden) for inspection. Integer slider.
uniform uint u_debug_step = 0u;

// --- Step captions. ONE array per step, holding "LABEL\nMATH" (newline = code 10,
// split into two rendered lines). One combined array (not two) keeps the uniform
// register footprint under the driver's constant limit (the glyph tables already
// consume most of it). The engine encodes each typed string -> codepoints (name
// ends in "text"). Glyph set: A-Z 0-9 ! ? : ; , . - ' & (math transliterated). ----
#define TLEN 36
uniform uint u_step1text[TLEN];
uniform uint u_step2text[TLEN];
uniform uint u_step3text[TLEN];
uniform uint u_step4text[TLEN];
uniform uint u_step5text[TLEN];
uniform uint u_step6text[TLEN];
uniform uint u_step7text[TLEN];
uniform uint u_step8text[TLEN];
uniform uint u_step9text[TLEN];

// --- Text layout / style (all tunable) ---
uniform vec2 u_text_pos = vec2(0.0, 1.62); // caption anchor in flame space (y near top; range 0..2)
uniform float u_text_size = 0.16;          // glyph height in flame-space units (auto-shrinks to fit width)
uniform float u_text_weight = 0.12;        // stroke half-width (bold = higher)
uniform float u_text_sharp = 0.18;         // edge sharpness (lower = crisper, higher = softer)
uniform float u_text_spacing = 0.62;       // per-char advance (lower = tighter, higher = looser)
uniform float u_text_line_gap = 0.22;      // vertical gap between label and math line
uniform vec3 u_text_color = vec3(1.0, 0.95, 0.85);
uniform float u_text_fade_in = 0.4;        // seconds to fade a caption IN at its step start
uniform float u_text_fade_out = 0.4;       // seconds to fade OUT as the next step begins
uniform float u_text_hold_extra = 0.0;     // extra seconds a caption lingers past its slot

// --- Timed step reveal ------------------------------------------------------
// Each of the 9 build steps fades in 1.5s apart and STACKS on the previous, then
// the full image holds 3.5s, then the whole reveal loops.
const float STEP_DUR = 1.5;   // seconds per step reveal
const float HOLD = 3.5;       // seconds to hold the finished image
const float N_STEPS = 9.0;
const float FADE = 0.6;       // crossfade length within each step's slot

out vec4 fs_color;

// Looping reveal clock: 0 at the start of each cycle, runs to BUILD+HOLD then wraps.
// DEBUG override: u_debug_step in 1..9 pins the clock to JUST PAST step N's fade, so
// steps 1..N read fully on and N+1.. fully off (frozen build state). 0 = autoplay.
float reveal_time() {
    if (u_debug_step > 0u) {
        float n = min(float(u_debug_step), N_STEPS);
        return (n - 1.0) * STEP_DUR + FADE + 0.001; // just past step n's reveal
    }
    float period = N_STEPS * STEP_DUR + HOLD;
    return mod(u_time, period);
}

// Weight [0,1] for step `i` (1-based): 0 before its slot, ramps to 1 over FADE,
// stays 1 after (it has been revealed and holds through the rest of the cycle).
float reveal(float i) {
    float start = (i - 1.0) * STEP_DUR;
    return smoothstep(start, start + FADE, reveal_time());
}

// Flame space: x in [-aspect, aspect], y from 0 (bottom) -> ~2 (top).
vec2 flame_space(vec2 uv, float aspect) {
    vec2 p = SB_center_uv(uv, aspect);
    p.y += 1.0;
    return p;
}

// Domain-warped, upward-scrolling turbulence (iq recursive warp).
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

// Two-scale warped field: coarse body motion + fine interior crackle.
float interior_field(vec2 p, float t) {
    float coarse = warp_noise(p, t, 1.4, 2.4);
    float fine = warp_noise(p, t, 3.6, 3.4);
    return coarse * 0.7 + fine * 0.3;
}

// Low-frequency flow that displaces the silhouette so the boundary swims.
vec2 flow_offset(vec2 p, float t) {
    vec2 sp = vec2(p.x * 1.2, p.y * 0.9 - t * 1.8);
    float ox = SB_fbm(sp + vec2(11.5, 2.1), 4) - 0.5;
    float oy = SB_fbm(sp + vec2(3.7, 19.3), 4) - 0.5;
    return vec2(ox, oy);
}

// Flame heat. `warpAmt` (step 5) eases in the flow-warp + edge-eat: at 0 it's the
// STATIC silhouette (step 3); at 1 it's the moving/frayed v11 shape.
float flame_heat(vec2 p, float t, float n, float warpAmt) {
    float topness = smoothstep(0.0, 1.6, p.y);
    vec2 w = p + flow_offset(p, t) * (0.12 + topness * 0.55) * warpAmt;

    float hh = clamp(w.y * 0.55, 0.0, 1.0);
    float width = mix(0.5, 0.14, hh) * (0.7 + 0.3 * smoothstep(0.0, 0.25, w.y));
    float column = exp(-pow(abs(w.x) / width, 2.0));

    float fuel = (1.0 - smoothstep(0.1, 1.8, w.y)) + n * 0.35;
    float gate = column * smoothstep(0.05, 0.5, fuel);
    float heat = n * gate;

    heat -= (1.0 - n) * (0.15 + topness * 0.7) * gate * warpAmt; // edge-eat eases in
    heat *= smoothstep(0.06, 0.4, n * gate + n * 0.2);

    heat = pow(clamp(heat, 0.0, 1.0), 0.8) * 1.2;
    heat *= 1.0 + (1.0 - smoothstep(0.0, 0.6, p.y)) * 0.35;
    return clamp(heat, 0.0, 1.4);
}

// Blackbody-ish temperature ramp (5 stops).
vec3 fire_ramp(float h) {
    vec3 c = vec3(0.0);
    c = mix(c, vec3(0.5, 0.04, 0.0), smoothstep(0.05, 0.35, h));
    c = mix(c, vec3(0.95, 0.25, 0.02), smoothstep(0.3, 0.55, h));
    c = mix(c, vec3(1.0, 0.65, 0.12), smoothstep(0.5, 0.78, h));
    c = mix(c, vec3(1.0, 0.92, 0.55), smoothstep(0.75, 1.0, h));
    c = mix(c, vec3(1.0, 1.0, 0.95), smoothstep(1.0, 1.3, h));
    return c;
}

// Global light flicker swing.
float flicker(float t) {
    float s = u_flicker_speed;
    float f = 0.5 * sin(t * 13.0 * s)
            + 0.3 * sin(t * 6.7 * s + 1.1)
            + 0.2 * sin(t * 3.1 * s + 0.5);
    return 1.0 + f * u_flicker_amp;
}

// Round radiating glow (ambient light the fire casts).
vec3 round_glow(vec2 p, float n, float light) {
    vec2 src = vec2(0.0, 0.35);
    float d = length((p - src) * vec2(1.0, 0.85));
    float radius = u_glow_radius / max(0.4, light);
    float halo = exp(-radius * d * d);
    halo *= 0.45 + 0.55 * n;
    return vec3(1.0, 0.38, 0.1) * halo * u_glow_strength * light;
}

// Sparse rising embers.
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

// --- Text -------------------------------------------------------------------
// Render "LABEL\nMATH" (one codepoint array, 10 = newline) as TWO centred lines:
// label at u_text_pos, math one u_text_line_gap below (smaller). Each line is
// centred on its own char count. Returns combined ink coverage [0,1].
float draw_caption(vec2 p, uint s[TLEN], float ch, float weight) {
    // Pass 1: per-line char counts (line 0 = label, line 1 = math).
    int cnt0 = 0, cnt1 = 0, line = 0;
    for (int i = 0; i < TLEN; ++i) {
        uint c = s[i];
        if (c == 0u) break;
        if (c == 10u) { line = 1; continue; }
        if (line == 0) cnt0++; else cnt1++;
    }
    if (cnt0 == 0 && cnt1 == 0) return 0.0;

    // Auto-fit: a line of N glyphs spans ~N*ch*0.62 wide; keep it inside the frame
    // (half-width = u_aspect in flame space, minus a small margin) by shrinking ch.
    float maxW = (u_aspect - 0.08) * 2.0;
    float ch0 = ch;
    float ch1 = ch * 0.8;
    float w0 = float(cnt0) * ch0 * u_text_spacing;
    float w1 = float(cnt1) * ch1 * u_text_spacing;
    if (w0 > maxW) ch0 *= maxW / w0;
    if (w1 > maxW) ch1 *= maxW / w1;

    float adv0 = ch0 * u_text_spacing;
    float adv1 = ch1 * u_text_spacing;
    float x0_0 = u_text_pos.x - adv0 * float(cnt0 - 1) * 0.5;
    float x0_1 = u_text_pos.x - adv1 * float(max(cnt1 - 1, 0)) * 0.5;
    float ymath = u_text_pos.y - u_text_line_gap;

    float d = 1e9;
    int k0 = 0, k1 = 0; line = 0;
    for (int i = 0; i < TLEN; ++i) {
        uint cp = s[i];
        if (cp == 0u) break;
        if (cp == 10u) { line = 1; continue; }
        if (line == 0) {
            vec2 c = vec2(x0_0 + adv0 * float(k0++), u_text_pos.y);
            d = min(d, SB_sd_char((p - c) / (0.5 * ch0), cp, weight) * (0.5 * ch0));
        } else {
            vec2 c = vec2(x0_1 + adv1 * float(k1++), ymath);
            d = min(d, SB_sd_char((p - c) / (0.5 * ch1), cp, weight) * (0.5 * ch1));
        }
    }
    return 1.0 - smoothstep(0.0, 0.5 * ch0 * max(u_text_sharp, 0.001), d);
}

// Per-step caption alpha: fades IN over u_text_fade_in at the step's start, holds
// through its slot, fades OUT over u_text_fade_out as the next step begins. In
// debug-freeze the selected step's caption stays fully on.
float caption_alpha(float i) {
    float start = (i - 1.0) * STEP_DUR;
    float slotEnd = start + STEP_DUR + u_text_hold_extra;
    if (u_debug_step > 0u) return (float(u_debug_step) == i) ? 1.0 : 0.0;
    float rt = reveal_time();
    float fin = smoothstep(start, start + max(u_text_fade_in, 0.001), rt);
    float fout = 1.0 - smoothstep(slotEnd - max(u_text_fade_out, 0.001), slotEnd, rt);
    return fin * fout;
}

// Coverage of step `i`'s two-line caption at pixel p. GLSL can't index an
// array-of-arrays, so dispatch the per-step uniform array.
float step_caption(int i, vec2 p) {
    float w = u_text_weight;
    float ch = u_text_size;
    if (i == 1) return draw_caption(p, u_step1text, ch, w);
    if (i == 2) return draw_caption(p, u_step2text, ch, w);
    if (i == 3) return draw_caption(p, u_step3text, ch, w);
    if (i == 4) return draw_caption(p, u_step4text, ch, w);
    if (i == 5) return draw_caption(p, u_step5text, ch, w);
    if (i == 6) return draw_caption(p, u_step6text, ch, w);
    if (i == 7) return draw_caption(p, u_step7text, ch, w);
    if (i == 8) return draw_caption(p, u_step8text, ch, w);
    return draw_caption(p, u_step9text, ch, w);
}

void main() {
    vec2 tp = flame_space(vs_uv, u_aspect); // text space (UNscaled — captions stay put)
    vec2 p = tp * max(u_scale, 0.01);       // flame space (global shrink applies here)
    float t = u_time;

    // Per-step reveal weights (each stacks on the previous, holds once revealed).
    float w1 = reveal(1.0); // base gradient
    float w2 = reveal(2.0); // warped noise field
    float w3 = reveal(3.0); // flame silhouette (static)
    float w4 = reveal(4.0); // temperature color ramp
    float w5 = reveal(5.0); // moving + frayed shape (warp eases in)
    float w6 = reveal(6.0); // round glow
    float w7 = reveal(7.0); // flicker on the glow
    float w8 = reveal(8.0); // embers
    float w9 = reveal(9.0); // depth layer behind

    float light = mix(1.0, flicker(t), w7);              // flicker fades in (step 7)
    float n = interior_field(p, t);
    float heat = flame_heat(p, t, n, w5);                // warp/edge-eat eases in (step 5)

    // --- build the colour by stacking steps ---
    // Step 1: base vertical heat gradient (greyscale), the very first thing shown.
    float base = exp(-1.4 * p.y);
    vec3 color = vec3(base) * w1;

    // Step 2: cross-fade the gradient into the raw warped field (greyscale).
    color = mix(color, vec3(n), w2);

    // Step 3: cross-fade into the carved flame silhouette (still greyscale heat).
    color = mix(color, vec3(heat), w3);

    // Step 4: cross-fade greyscale heat into the temperature ramp (colour).
    color = mix(color, fire_ramp(heat), w4);

    // (Step 5 already folded into `heat` via w5 — the shape starts moving/fraying.)

    // Step 6+7: add the glow (step 6), which begins to breathe once flicker is in (7).
    color += round_glow(p, n, light) * w6;

    // Step 8: embers + core bloom fade in.
    color += embers(p, t) * w8;
    color += vec3(1.0, 0.85, 0.6) * smoothstep(0.8, 1.05, heat) * 0.35 * w8;

    // Step 9: the depth layer behind the flame fades in. It reads as a second flame
    // further back, so it must appear ONLY where the front flame body isn't — added
    // into the surrounding dark, never subtracted from the already-built front.
    vec2 bp = vec2(p.x * 1.12 + 0.16, p.y * 1.05 + 0.05);
    float bn = interior_field(bp, t + 7.3);
    float bheat = flame_heat(bp, t + 7.3, bn, w5) * 0.75;
    vec3 bcol = fire_ramp(bheat) * vec3(0.85, 0.7, 0.7);
    float bcover = smoothstep(0.02, 0.14, bheat) * u_depth * w9;
    float frontCover = smoothstep(0.02, 0.12, heat); // where the front flame sits
    color += bcol * bcover * (1.0 - frontCover);      // back layer, behind the front

    // --- Step captions on top (each step's text, faded per its window) ---
    // Only the 1-2 steps whose windows overlap `t` contribute; accumulate their
    // ink * per-step alpha, then composite as opaque text over the flame.
    float ink = 0.0;
    for (int i = 1; i <= 9; ++i) {
        float a = caption_alpha(float(i));
        if (a > 0.0) ink = max(ink, step_caption(i, tp) * a);
    }
    color = mix(color, u_text_color, clamp(ink, 0.0, 1.0));

    color = min(color, vec3(1.0));
    fs_color = vec4(color, 1.0);
}
