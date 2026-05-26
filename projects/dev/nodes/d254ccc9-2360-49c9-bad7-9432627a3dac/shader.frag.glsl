#version 460 core

#define MAX_DIST 9999.9
#define MIN_SMOOTHNESS 0.0001
#define MAX_TEXT_LEN 64

in vec2 vs_uv;
out vec4 fs_color;

uniform float u_time;
uniform float u_aspect;

uniform float u_zoomout = 10.0;
uniform float u_char_scale = 0.5;
uniform vec2 u_offset = vec2(0.0, 0.0);
uniform vec2 u_text_spacing = vec2(0.0, 0.0);

uniform float u_text_thickness = 0.05;
uniform float u_text_smoothness = 0.01;

uniform uint u_text[MAX_TEXT_LEN];
uniform vec3 u_color = vec3(1.0, 0.0, 0.0);

// Fake volume: shade the stroke by its SDF normal lit from u_light_angle.
uniform float u_bevel = 0.6;          // 0 = flat fill, 1 = full bevel shading
uniform float u_bevel_hardness = 0.3; // 0 = soft rounded tube, 1 = hard chisel
uniform float u_light_angle = 2.2;    // light direction (radians)

float get_dist_to_line(vec2 p, vec2 a, vec2 b) {
    vec2 ab = b - a;
    vec2 ap = p - a;
    float t = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
    vec2 closest = a + t * ab;
    return length(p - closest);
}

const float CW = 0.5;
const float CH = 1.0;

// Glyph anchor points on a 7-segment-style cell (CW x CH half-extents).
const vec2 A = vec2(-CW, -CH);
const vec2 B = vec2(-CW, -0.5 * CH);
const vec2 C = vec2(-CW, 0.0);
const vec2 D = vec2(-CW, 0.5 * CH);
const vec2 E = vec2(-CW, CH);
const vec2 F = vec2(0.0, CH);
const vec2 G = vec2(CW, CH);
const vec2 H = vec2(CW, 0.5 * CH);
const vec2 J = vec2(CW, 0.0);
const vec2 K = vec2(CW, -0.5 * CH);
const vec2 L = vec2(CW, -CH);
const vec2 M = vec2(0.0, -CH);
const vec2 N = vec2(0.0, -0.5 * CH);
const vec2 O = vec2(0.0, 0.0);
const vec2 P = vec2(0.0, 0.5 * CH);

float seg0(vec2 p) { return get_dist_to_line(p, A, B); }
float seg1(vec2 p) { return get_dist_to_line(p, B, C); }
float seg2(vec2 p) { return get_dist_to_line(p, C, D); }
float seg3(vec2 p) { return get_dist_to_line(p, D, E); }
float seg4(vec2 p) { return get_dist_to_line(p, E, F); }
float seg5(vec2 p) { return get_dist_to_line(p, F, G); }
float seg6(vec2 p) { return get_dist_to_line(p, G, H); }
float seg7(vec2 p) { return get_dist_to_line(p, H, J); }
float seg8(vec2 p) { return get_dist_to_line(p, J, K); }
float seg9(vec2 p) { return get_dist_to_line(p, K, L); }
float seg10(vec2 p) { return get_dist_to_line(p, M, L); }
float seg11(vec2 p) { return get_dist_to_line(p, A, M); }
float seg12(vec2 p) { return get_dist_to_line(p, M, N); }
float seg14(vec2 p) { return get_dist_to_line(p, N, O); }
float seg16(vec2 p) { return get_dist_to_line(p, O, P); }
float seg18(vec2 p) { return get_dist_to_line(p, P, F); }
float seg20(vec2 p) { return get_dist_to_line(p, A, O); }
float seg21(vec2 p) { return get_dist_to_line(p, E, O); }
float seg22(vec2 p) { return get_dist_to_line(p, O, G); }
float seg23(vec2 p) { return get_dist_to_line(p, O, L); }
float seg24(vec2 p) { return get_dist_to_line(p, C, O); }
float seg25(vec2 p) { return get_dist_to_line(p, O, J); }

float quarter_ellipse_dist(vec2 p, vec2 c, float x_cond, float y_cond,
                           vec2 endpoint1, vec2 endpoint2) {
    float rx = CW;
    float ry = 0.5 * CH;

    vec2 v = (p - c) / vec2(rx, ry);
    float dist = abs(length(v) - 1.0) * min(rx, ry);
    float in_quad_x = mix(1.0 - step(v.x, 0.0), step(v.x, 0.0), x_cond);
    float in_quad_y = mix(1.0 - step(v.y, 0.0), step(v.y, 0.0), y_cond);
    float in_quad = in_quad_x * in_quad_y;
    return mix(min(distance(p, endpoint1), distance(p, endpoint2)), dist,
               in_quad);
}

float seg26(vec2 p) {
    return quarter_ellipse_dist(p, vec2(0.0, -0.5 * CH), 1.0, 1.0,
                                vec2(-CW, -0.5 * CH), vec2(0.0, -CH));
}

float seg28(vec2 p) {
    return quarter_ellipse_dist(p, vec2(0.0, -0.5 * CH), 0.0, 0.0,
                                vec2(CW, -0.5 * CH), vec2(0.0, 0.0));
}

float seg29(vec2 p) {
    return quarter_ellipse_dist(p, vec2(0.0, -0.5 * CH), 0.0, 1.0,
                                vec2(0.0, -CH), vec2(CW, -0.5 * CH));
}

float seg30(vec2 p) {
    return quarter_ellipse_dist(p, vec2(0.0, 0.5 * CH), 1.0, 1.0,
                                vec2(-CW, 0.5 * CH), vec2(0.0, 0.0));
}

float seg31(vec2 p) {
    return quarter_ellipse_dist(p, vec2(0.0, 0.5 * CH), 1.0, 0.0, vec2(0.0, CH),
                                vec2(-CW, 0.5 * CH));
}

float seg32(vec2 p) {
    return quarter_ellipse_dist(p, vec2(0.0, 0.5 * CH), 0.0, 0.0,
                                vec2(CW, 0.5 * CH), vec2(0.0, CH));
}

float seg33(vec2 p) {
    return quarter_ellipse_dist(p, vec2(0.0, 0.5 * CH), 0.0, 1.0,
                                vec2(0.0, 0.0), vec2(CW, 0.5 * CH));
}

float get_dist_to_latin_A(vec2 p) {
    float d = seg0(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg31(p));
    d = min(d, seg32(p));
    d = min(d, seg7(p));
    d = min(d, seg8(p));
    d = min(d, seg9(p));
    d = min(d, seg24(p));
    d = min(d, seg25(p));
    return d;
}

float get_dist_to_latin_B(vec2 p) {
    float d = seg0(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg3(p));
    d = min(d, seg4(p));
    d = min(d, seg32(p));
    d = min(d, seg33(p));
    d = min(d, seg24(p));
    d = min(d, seg28(p));
    d = min(d, seg29(p));
    d = min(d, seg11(p));
    return d;
}

float get_dist_to_latin_C(vec2 p) {
    float d = seg29(p);
    d = min(d, seg26(p));
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg31(p));
    d = min(d, seg32(p));
    return d;
}

float get_dist_to_latin_D(vec2 p) {
    float d = seg0(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg3(p));
    d = min(d, seg4(p));
    d = min(d, seg32(p));
    d = min(d, seg7(p));
    d = min(d, seg8(p));
    d = min(d, seg29(p));
    d = min(d, seg11(p));
    return d;
}

float get_dist_to_latin_E(vec2 p) {
    float d = seg0(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg3(p));
    d = min(d, seg4(p));
    d = min(d, seg5(p));
    d = min(d, seg24(p));
    d = min(d, seg25(p));
    d = min(d, seg11(p));
    d = min(d, seg10(p));
    return d;
}

float get_dist_to_latin_F(vec2 p) {
    float d = seg0(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg3(p));
    d = min(d, seg4(p));
    d = min(d, seg5(p));
    d = min(d, seg24(p));
    return d;
}

float get_dist_to_latin_G(vec2 p) {
    float d = seg26(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg31(p));
    d = min(d, seg5(p));
    d = min(d, seg29(p));
    d = min(d, seg8(p));
    d = min(d, seg25(p));
    return d;
}

float get_dist_to_latin_H(vec2 p) {
    float d = seg0(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg3(p));
    d = min(d, seg24(p));
    d = min(d, seg25(p));
    d = min(d, seg9(p));
    d = min(d, seg8(p));
    d = min(d, seg7(p));
    d = min(d, seg6(p));
    return d;
}

float get_dist_to_latin_I(vec2 p) {
    float d = seg12(p);
    d = min(d, seg14(p));
    d = min(d, seg16(p));
    d = min(d, seg18(p));
    d = min(d, seg4(p));
    d = min(d, seg5(p));
    d = min(d, seg11(p));
    d = min(d, seg10(p));
    return d;
}

float get_dist_to_latin_J(vec2 p) {
    float d = seg8(p);
    d = min(d, seg7(p));
    d = min(d, seg6(p));
    d = min(d, seg5(p));
    d = min(d, seg29(p));
    d = min(d, seg26(p));
    return d;
}

float get_dist_to_latin_K(vec2 p) {
    float d = seg0(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg3(p));
    d = min(d, seg24(p));
    d = min(d, seg22(p));
    d = min(d, seg23(p));
    return d;
}

float get_dist_to_latin_L(vec2 p) {
    float d = seg0(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg3(p));
    d = min(d, seg11(p));
    d = min(d, seg10(p));
    return d;
}

float get_dist_to_latin_M(vec2 p) {
    float d = seg0(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg3(p));
    d = min(d, seg3(p));
    d = min(d, seg21(p));
    d = min(d, seg22(p));
    d = min(d, seg6(p));
    d = min(d, seg7(p));
    d = min(d, seg8(p));
    d = min(d, seg9(p));
    return d;
}

float get_dist_to_latin_N(vec2 p) {
    float d = seg0(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg3(p));
    d = min(d, seg3(p));
    d = min(d, seg21(p));
    d = min(d, seg23(p));
    d = min(d, seg6(p));
    d = min(d, seg7(p));
    d = min(d, seg8(p));
    d = min(d, seg9(p));
    return d;
}

float get_dist_to_latin_O(vec2 p) {
    float d = seg26(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg31(p));
    d = min(d, seg32(p));
    d = min(d, seg7(p));
    d = min(d, seg8(p));
    d = min(d, seg29(p));
    return d;
}

float get_dist_to_latin_P(vec2 p) {
    float d = seg0(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg3(p));
    d = min(d, seg4(p));
    d = min(d, seg32(p));
    d = min(d, seg33(p));
    d = min(d, seg24(p));
    return d;
}

float get_dist_to_latin_Q(vec2 p) {
    float d = seg26(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg31(p));
    d = min(d, seg32(p));
    d = min(d, seg7(p));
    d = min(d, seg8(p));
    d = min(d, seg29(p));
    d = min(d, seg23(p));
    return d;
}

float get_dist_to_latin_R(vec2 p) {
    float d = seg0(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg3(p));
    d = min(d, seg4(p));
    d = min(d, seg32(p));
    d = min(d, seg33(p));
    d = min(d, seg23(p));
    d = min(d, seg24(p));
    return d;
}

float get_dist_to_latin_S(vec2 p) {
    float d = seg32(p);
    d = min(d, seg31(p));
    d = min(d, seg30(p));
    d = min(d, seg28(p));
    d = min(d, seg29(p));
    d = min(d, seg26(p));
    return d;
}

float get_dist_to_latin_T(vec2 p) {
    float d = seg12(p);
    d = min(d, seg14(p));
    d = min(d, seg16(p));
    d = min(d, seg18(p));
    d = min(d, seg4(p));
    d = min(d, seg5(p));
    return d;
}

float get_dist_to_latin_U(vec2 p) {
    float d = seg1(p);
    d = min(d, seg2(p));
    d = min(d, seg3(p));
    d = min(d, seg26(p));
    d = min(d, seg29(p));
    d = min(d, seg8(p));
    d = min(d, seg7(p));
    d = min(d, seg6(p));
    return d;
}

float get_dist_to_latin_V(vec2 p) {
    float d = seg0(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg3(p));
    d = min(d, seg20(p));
    d = min(d, seg22(p));
    return d;
}

float get_dist_to_latin_W(vec2 p) {
    float d = seg0(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg3(p));
    d = min(d, seg20(p));
    d = min(d, seg23(p));
    d = min(d, seg9(p));
    d = min(d, seg8(p));
    d = min(d, seg7(p));
    d = min(d, seg6(p));
    return d;
}

float get_dist_to_latin_X(vec2 p) {
    float d = seg20(p);
    d = min(d, seg22(p));
    d = min(d, seg21(p));
    d = min(d, seg23(p));
    return d;
}

float get_dist_to_latin_Y(vec2 p) {
    float d = seg12(p);
    d = min(d, seg14(p));
    d = min(d, seg21(p));
    d = min(d, seg22(p));
    return d;
}

float get_dist_to_latin_Z(vec2 p) {
    float d = seg4(p);
    d = min(d, seg5(p));
    d = min(d, seg22(p));
    d = min(d, seg20(p));
    d = min(d, seg11(p));
    d = min(d, seg10(p));
    return d;
}

float get_dist_to_dash(vec2 p) {
    float d = seg24(p);
    d = min(d, seg25(p));
    return d;
}

// Punctuation. A "dot" is a filled disc of radius DOT_R; strokes reuse
// get_dist_to_line. Marks sit near the baseline (y = -CH).
const float DOT_R = 0.12;

float dot_dist(vec2 p, vec2 c) { return distance(p, c) - DOT_R; }

float get_dist_to_period(vec2 p) {
    return dot_dist(p, vec2(0.0, -CH + DOT_R));
}

float get_dist_to_comma(vec2 p) {
    vec2 head = vec2(0.0, -0.5 * CH);
    float d = dot_dist(p, head);
    // A short tail curling down-left below the baseline.
    d = min(d, get_dist_to_line(p, head, vec2(-0.25 * CW, -CH)));
    return d;
}

float get_dist_to_semicolon(vec2 p) {
    float d = dot_dist(p, vec2(0.0, 0.0)); // upper dot
    d = min(d, get_dist_to_comma(p));      // lower comma
    return d;
}

float get_dist_to_ampersand(vec2 p) {
    // Approximation on the segment lattice: a loop in the upper cell plus a
    // diagonal kicking out to the lower right.
    float d = seg30(p);
    d = min(d, seg31(p));
    d = min(d, seg32(p));
    d = min(d, seg33(p));
    d = min(d, seg2(p));
    d = min(d, seg20(p)); // A -> O diagonal
    d = min(d, seg25(p)); // O -> J
    return d;
}

float get_dist_to_latin_char(vec2 p, uint char_unicode_idx) {
    // This is an all-caps display face: fold lowercase a-z (97-122) onto the
    // uppercase glyphs A-Z (65-90).
    if (char_unicode_idx >= 97u && char_unicode_idx <= 122u) {
        char_unicode_idx -= 32u;
    }

    switch (char_unicode_idx) {
    case 65: // A
        return get_dist_to_latin_A(p);
    case 66: // B
        return get_dist_to_latin_B(p);
    case 67: // C
        return get_dist_to_latin_C(p);
    case 68: // D
        return get_dist_to_latin_D(p);
    case 69: // E
        return get_dist_to_latin_E(p);
    case 70: // F
        return get_dist_to_latin_F(p);
    case 71: // G
        return get_dist_to_latin_G(p);
    case 72: // H
        return get_dist_to_latin_H(p);
    case 73: // I
        return get_dist_to_latin_I(p);
    case 74: // J
        return get_dist_to_latin_J(p);
    case 75: // K
        return get_dist_to_latin_K(p);
    case 76: // L
        return get_dist_to_latin_L(p);
    case 77: // M
        return get_dist_to_latin_M(p);
    case 78: // N
        return get_dist_to_latin_N(p);
    case 79: // O
        return get_dist_to_latin_O(p);
    case 80: // P
        return get_dist_to_latin_P(p);
    case 81: // Q
        return get_dist_to_latin_Q(p);
    case 82: // R
        return get_dist_to_latin_R(p);
    case 83: // S
        return get_dist_to_latin_S(p);
    case 84: // T
        return get_dist_to_latin_T(p);
    case 85: // U
        return get_dist_to_latin_U(p);
    case 86: // V
        return get_dist_to_latin_V(p);
    case 87: // W
        return get_dist_to_latin_W(p);
    case 88: // X
        return get_dist_to_latin_X(p);
    case 89: // Y
        return get_dist_to_latin_Y(p);
    case 90: // Z
        return get_dist_to_latin_Z(p);
    // -------------------------------------------------------------------
    // Punctuation
    case 38: // &
        return get_dist_to_ampersand(p);
    case 44: // ,
        return get_dist_to_comma(p);
    case 45: // -
        return get_dist_to_dash(p);
    case 46: // .
        return get_dist_to_period(p);
    case 59: // ;
        return get_dist_to_semicolon(p);
    default:
        return MAX_DIST;
    }
}

float get_line(float dist, float width, float smoothness) {
    smoothness = max(MIN_SMOOTHNESS, smoothness);
    return 1.0 - smoothstep(0.0, smoothness, dist - width);
}

void main() {
    vec2 uv = (vs_uv - u_offset) * vec2(u_aspect, 1.0) * u_zoomout;

    float dist = MAX_DIST;

    float col = 0.0; // advancing x position, reset on newline
    float row = 0.0; // line index, grows downward

    for (uint i = 0; i < MAX_TEXT_LEN; ++i) {
        uint char_unicode_idx = u_text[i];

        if (char_unicode_idx == 0u) { // end of string
            break;
        }

        if (char_unicode_idx == 10u) { // '\n' -> carriage return + line feed
            col = 0.0;
            row += 1.0;
            continue;
        }

        // Lines stack downward: y starts at u_zoomout and each row drops by one
        // glyph height (in uv units, = u_char_scale * CH) plus the vertical spacing.
        float line_step = u_char_scale * CH + u_text_spacing.y;
        vec2 char_pos = vec2(col * (1.0 + u_text_spacing.x),
                             u_zoomout - row * line_step);

        vec2 p = 2.0 * (uv - char_pos) / u_char_scale;
        dist = min(dist, get_dist_to_latin_char(p, char_unicode_idx));

        col += 1.0;
    }

    float line = get_line(dist, u_text_thickness, u_text_smoothness);

    // Round-tube volume: treat the stroke cross-section as a half-cylinder of
    // radius u_text_thickness. `t` is the normalised distance from the spine
    // (0 at center, 1 at the rim) -- continuous across the stroke, so there is no
    // medial-axis ridge. The surface tilts outward by `t`: flat-on at the spine,
    // edge-on at the rim. Hardness biases the tilt toward the rim (sharper bevel).
    float t = clamp(dist / max(u_text_thickness, MIN_SMOOTHNESS), 0.0, 1.0);
    vec2 across = normalize(vec2(dFdx(dist), dFdy(dist)) + 1e-6);
    float tilt = mix(t, smoothstep(0.0, 1.0, t), u_bevel_hardness);
    vec3 normal = normalize(vec3(across * tilt, sqrt(1.0 - tilt * tilt)));

    vec3 light = normalize(vec3(cos(u_light_angle), sin(u_light_angle), 0.8));
    float lambert = dot(normal, light) * 0.5 + 0.5; // [0, 1], smooth everywhere

    float shade = mix(1.0, 0.35 + 1.0 * lambert, u_bevel);
    vec3 color = u_color * shade * line;

    fs_color = vec4(color, 1.0);
}
