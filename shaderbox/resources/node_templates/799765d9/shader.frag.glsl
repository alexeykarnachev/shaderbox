#version 460 core

#define MAX_DIST 9999.9
#define MIN_SMOOTHNESS 0.0001
#define MAX_TEXT_LEN 64
#define N_TEXT_LINES 3

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
uniform vec3 u_text_color = vec3(1.0, 0.0, 0.0);

uniform uint u_text[N_TEXT_LINES][MAX_TEXT_LEN]; // Changed to 2D array

float get_dist_to_line(vec2 p, vec2 a, vec2 b) {
    vec2 ab = b - a;
    vec2 ap = p - a;
    float t = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
    vec2 closest = a + t * ab;
    return length(p - closest);
}

const float CW = 0.5;
const float CH = 1.0;
const float DOTS_THICKNESS = 0.05;

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
const vec2 Q = vec2(-0.5 * CW, 1.3 * CH);
const vec2 R = vec2(0.0, 1.3 * CH);
const vec2 S = vec2(0.5 * CW, 1.3 * CH);
const vec2 T = vec2(-1.3 * CW, -CH);
const vec2 U = vec2(1.3 * CW, -CH);
const vec2 V = vec2(-1.3 * CW, CH);
const vec2 W = vec2(1.3 * CW, CH);

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
float seg13(vec2 p) { return get_dist_to_line(p, B, N); }
float seg14(vec2 p) { return get_dist_to_line(p, N, O); }
float seg15(vec2 p) { return get_dist_to_line(p, N, K); }
float seg16(vec2 p) { return get_dist_to_line(p, O, P); }
float seg17(vec2 p) { return get_dist_to_line(p, D, P); }
float seg18(vec2 p) { return get_dist_to_line(p, P, F); }
float seg19(vec2 p) { return get_dist_to_line(p, P, H); }
float seg20(vec2 p) { return get_dist_to_line(p, A, O); }
float seg21(vec2 p) { return get_dist_to_line(p, E, O); }
float seg22(vec2 p) { return get_dist_to_line(p, O, G); }
float seg23(vec2 p) { return get_dist_to_line(p, O, L); }
float seg24(vec2 p) { return get_dist_to_line(p, C, O); }
float seg25(vec2 p) { return get_dist_to_line(p, O, J); }

float quarter_ellipse_dist(vec2 p,
                           vec2 c,
                           float x_cond,
                           float y_cond,
                           vec2 endpoint1,
                           vec2 endpoint2) {
    float rx = CW;
    float ry = 0.5 * CH;

    // Scale coordinates to transform ellipse into a unit circle
    vec2 v = (p - c) / vec2(rx, ry);
    // Approximate distance to the ellipse (unit radius in scaled space)
    float dist =
        abs(length(v) - 1.0) *
        min(rx, ry); // Scale back by min dimension for better approximation
    // Quadrant selection
    float in_quad_x = mix(1.0 - step(v.x, 0.0), step(v.x, 0.0), x_cond);
    float in_quad_y = mix(1.0 - step(v.y, 0.0), step(v.y, 0.0), y_cond);
    float in_quad = in_quad_x * in_quad_y;
    // Return distance, blending with endpoints outside the quadrant
    return mix(
        min(distance(p, endpoint1), distance(p, endpoint2)), dist, in_quad);
}

float seg26(vec2 p) {
    return quarter_ellipse_dist(p,
                                vec2(0.0, -0.5 * CH),
                                1.0,
                                1.0,
                                vec2(-CW, -0.5 * CH),
                                vec2(0.0, -CH));
}

float seg27(vec2 p) {
    return quarter_ellipse_dist(p,
                                vec2(0.0, -0.5 * CH),
                                1.0,
                                0.0,
                                vec2(0.0, 0.0),
                                vec2(-CW, -0.5 * CH));
}

float seg28(vec2 p) {
    return quarter_ellipse_dist(
        p, vec2(0.0, -0.5 * CH), 0.0, 0.0, vec2(CW, -0.5 * CH), vec2(0.0, 0.0));
}

float seg29(vec2 p) {
    return quarter_ellipse_dist(
        p, vec2(0.0, -0.5 * CH), 0.0, 1.0, vec2(0.0, -CH), vec2(CW, -0.5 * CH));
}

float seg30(vec2 p) {
    return quarter_ellipse_dist(
        p, vec2(0.0, 0.5 * CH), 1.0, 1.0, vec2(-CW, 0.5 * CH), vec2(0.0, 0.0));
}

float seg31(vec2 p) {
    return quarter_ellipse_dist(
        p, vec2(0.0, 0.5 * CH), 1.0, 0.0, vec2(0.0, CH), vec2(-CW, 0.5 * CH));
}

float seg32(vec2 p) {
    return quarter_ellipse_dist(
        p, vec2(0.0, 0.5 * CH), 0.0, 0.0, vec2(CW, 0.5 * CH), vec2(0.0, CH));
}

float seg33(vec2 p) {
    return quarter_ellipse_dist(
        p, vec2(0.0, 0.5 * CH), 0.0, 1.0, vec2(0.0, 0.0), vec2(CW, 0.5 * CH));
}

float seg34(vec2 p) { return max(distance(p, Q) - DOTS_THICKNESS, 0.0); }
float seg35(vec2 p) { return max(distance(p, R) - DOTS_THICKNESS, 0.0); }
float seg36(vec2 p) { return max(distance(p, S) - DOTS_THICKNESS, 0.0); }

float seg37(vec2 p) { return get_dist_to_line(p, T, A); }
float seg38(vec2 p) { return get_dist_to_line(p, L, U); }
float seg39(vec2 p) { return get_dist_to_line(p, V, E); }
float seg40(vec2 p) { return get_dist_to_line(p, G, W); }

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

float get_dist_to_cyrillic_YO(vec2 p) {
    float d = get_dist_to_latin_E(p);
    d = min(d, seg34(p));
    d = min(d, seg36(p));
    return d;
}

float get_dist_to_cyrillic_A(vec2 p) { return get_dist_to_latin_A(p); }

float get_dist_to_cyrillic_B(vec2 p) {
    float d = seg0(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg3(p));
    d = min(d, seg4(p));
    d = min(d, seg5(p));
    d = min(d, seg24(p));
    d = min(d, seg28(p));
    d = min(d, seg29(p));
    d = min(d, seg11(p));
    return d;
}

float get_dist_to_cyrillic_V(vec2 p) { return get_dist_to_latin_B(p); }

float get_dist_to_cyrillic_G(vec2 p) {
    float d = seg0(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg3(p));
    d = min(d, seg4(p));
    d = min(d, seg5(p));
    return d;
}

float get_dist_to_cyrillic_L(vec2 p) {
    float d = seg6(p);
    d = min(d, seg7(p));
    d = min(d, seg8(p));
    d = min(d, seg9(p));

    d = min(d, seg22(p));
    d = min(d, seg20(p));

    return d;
}

float get_dist_to_cyrillic_D(vec2 p) {
    float d = get_dist_to_cyrillic_L(p);

    d = min(d, seg37(p));
    d = min(d, seg11(p));
    d = min(d, seg10(p));
    d = min(d, seg38(p));
    return d;
}

float get_dist_to_cyrillic_E(vec2 p) { return get_dist_to_latin_E(p); }

float get_dist_to_cyrillic_ZH(vec2 p) {
    float d = seg0(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg3(p));

    d = min(d, seg12(p));
    d = min(d, seg14(p));
    d = min(d, seg16(p));
    d = min(d, seg18(p));

    d = min(d, seg9(p));
    d = min(d, seg8(p));
    d = min(d, seg7(p));
    d = min(d, seg6(p));

    d = min(d, seg24(p));
    d = min(d, seg25(p));

    return d;
}

float get_dist_to_cyrillic_Z(vec2 p) {
    float d = seg31(p);
    d = min(d, seg32(p));
    d = min(d, seg32(p));
    d = min(d, seg33(p));

    d = min(d, seg28(p));
    d = min(d, seg29(p));
    d = min(d, seg26(p));

    return d;
}

float get_dist_to_cyrillic_I(vec2 p) {
    float d = seg0(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg3(p));

    d = min(d, seg20(p));
    d = min(d, seg22(p));

    d = min(d, seg6(p));
    d = min(d, seg7(p));
    d = min(d, seg8(p));
    d = min(d, seg9(p));

    return d;
}

float get_dist_to_cyrillic_I_KRATKOYE(vec2 p) {
    float d = get_dist_to_cyrillic_I(p);
    d = min(d, seg35(p));
    return d;
}

float get_dist_to_cyrillic_K(vec2 p) {
    float d = get_dist_to_latin_K(p);
    return d;
}

float get_dist_to_cyrillic_M(vec2 p) { return get_dist_to_latin_M(p); }

float get_dist_to_cyrillic_N(vec2 p) { return get_dist_to_latin_H(p); }

float get_dist_to_cyrillic_O(vec2 p) { return get_dist_to_latin_O(p); }

float get_dist_to_cyrillic_P(vec2 p) {
    float d = seg0(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg3(p));

    d = min(d, seg6(p));
    d = min(d, seg7(p));
    d = min(d, seg8(p));
    d = min(d, seg9(p));

    d = min(d, seg4(p));
    d = min(d, seg5(p));
    return d;
}

float get_dist_to_cyrillic_R(vec2 p) { return get_dist_to_latin_P(p); }
float get_dist_to_cyrillic_S(vec2 p) { return get_dist_to_latin_C(p); }
float get_dist_to_cyrillic_T(vec2 p) { return get_dist_to_latin_T(p); }

float get_dist_to_cyrillic_U(vec2 p) {
    float d = seg3(p);
    d = min(d, seg30(p));
    d = min(d, seg25(p));

    d = min(d, seg6(p));
    d = min(d, seg7(p));
    d = min(d, seg8(p));

    d = min(d, seg29(p));
    d = min(d, seg11(p));
    return d;
}

float get_dist_to_cyrillic_F(vec2 p) {
    float d = seg31(p);
    d = min(d, seg32(p));
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg7(p));
    d = min(d, seg8(p));
    d = min(d, seg13(p));
    d = min(d, seg15(p));
    d = min(d, seg12(p));
    d = min(d, seg14(p));
    d = min(d, seg16(p));
    d = min(d, seg18(p));

    return d;
}

float get_dist_to_cyrillic_H(vec2 p) { return get_dist_to_latin_X(p); }

float get_dist_to_cyrillic_TS(vec2 p) {
    float d = seg1(p);
    d = min(d, seg2(p));
    d = min(d, seg3(p));
    d = min(d, seg0(p));
    d = min(d, seg11(p));
    d = min(d, seg8(p));
    d = min(d, seg7(p));
    d = min(d, seg6(p));

    d = min(d, seg9(p));
    d = min(d, seg10(p));

    d = min(d, seg38(p));
    return d;
}

float get_dist_to_cyrillic_CH(vec2 p) {
    float d = seg3(p);
    d = min(d, seg30(p));
    d = min(d, seg25(p));

    d = min(d, seg6(p));
    d = min(d, seg7(p));
    d = min(d, seg8(p));

    d = min(d, seg9(p));
    return d;
}

float get_dist_to_cyrillic_SH(vec2 p) {
    float d = seg0(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg3(p));

    d = min(d, seg6(p));
    d = min(d, seg7(p));
    d = min(d, seg8(p));
    d = min(d, seg9(p));

    d = min(d, seg12(p));
    d = min(d, seg14(p));
    d = min(d, seg16(p));
    d = min(d, seg18(p));

    d = min(d, seg11(p));
    d = min(d, seg10(p));
    return d;
}

float get_dist_to_cyrillic_SHCH(vec2 p) {
    float d = get_dist_to_cyrillic_SH(p);
    d = min(d, seg38(p));
    return d;
}

float get_dist_to_cyrillic_SOFT_SIGN(vec2 p) {
    float d = seg0(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg3(p));

    d = min(d, seg24(p));
    d = min(d, seg28(p));
    d = min(d, seg29(p));
    d = min(d, seg11(p));
    return d;
}

float get_dist_to_cyrillic_HARD_SIGN(vec2 p) {
    float d = get_dist_to_cyrillic_SOFT_SIGN(p);
    d = min(d, seg39(p));
    return d;
}

float get_dist_to_cyrillic_Y(vec2 p) {
    float d = get_dist_to_cyrillic_SOFT_SIGN(p);

    d = min(d, seg6(p));
    d = min(d, seg7(p));
    d = min(d, seg9(p));
    return d;
}

float get_dist_to_cyrillic_EH(vec2 p) {
    float d = seg4(p);
    d = min(d, seg32(p));
    d = min(d, seg7(p));
    d = min(d, seg8(p));
    d = min(d, seg29(p));
    d = min(d, seg11(p));

    d = min(d, seg25(p));

    return d;
}

float get_dist_to_cyrillic_YU(vec2 p) {
    float d = seg0(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg3(p));

    d = min(d, seg26(p));
    d = min(d, seg29(p));
    d = min(d, seg8(p));
    d = min(d, seg7(p));
    d = min(d, seg32(p));
    d = min(d, seg31(p));

    d = min(d, seg31(p));

    return d;
}

float get_dist_to_cyrillic_YA(vec2 p) {
    float d = seg6(p);
    d = min(d, seg7(p));
    d = min(d, seg8(p));
    d = min(d, seg9(p));

    d = min(d, seg5(p));
    d = min(d, seg31(p));
    d = min(d, seg30(p));
    d = min(d, seg25(p));

    d = min(d, seg20(p));

    return d;
}

float get_dist_to_dash(vec2 p) {
    float d = seg24(p);
    d = min(d, seg25(p));
    return d;
}

float get_dist_to_all_segments(vec2 p) {
    float d = seg0(p);
    d = min(d, seg1(p));
    d = min(d, seg2(p));
    d = min(d, seg3(p));
    d = min(d, seg4(p));
    d = min(d, seg5(p));
    d = min(d, seg6(p));
    d = min(d, seg7(p));
    d = min(d, seg8(p));
    d = min(d, seg9(p));
    d = min(d, seg10(p));
    d = min(d, seg11(p));
    d = min(d, seg12(p));
    d = min(d, seg13(p));
    d = min(d, seg14(p));
    d = min(d, seg15(p));
    d = min(d, seg16(p));
    d = min(d, seg17(p));
    d = min(d, seg18(p));
    d = min(d, seg19(p));
    d = min(d, seg20(p));
    d = min(d, seg21(p));
    d = min(d, seg22(p));
    d = min(d, seg23(p));
    d = min(d, seg24(p));
    d = min(d, seg25(p));
    d = min(d, seg26(p));
    d = min(d, seg27(p));
    d = min(d, seg28(p));
    d = min(d, seg29(p));
    d = min(d, seg30(p));
    d = min(d, seg31(p));
    d = min(d, seg32(p));
    d = min(d, seg33(p));

    return d;
}

float get_dist_to_latin_char(vec2 p, uint char_unicode_idx) {
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
    // Cyrillic
    case 1025: // Ё
        return get_dist_to_cyrillic_YO(p);
    case 1040: // А
        return get_dist_to_cyrillic_A(p);
    case 1041: // Б
        return get_dist_to_cyrillic_B(p);
    case 1042: // В
        return get_dist_to_cyrillic_V(p);
    case 1043: // Г
        return get_dist_to_cyrillic_G(p);
    case 1044: // Д
        return get_dist_to_cyrillic_D(p);
    case 1045: // Е
        return get_dist_to_cyrillic_E(p);
    case 1046: // Ж
        return get_dist_to_cyrillic_ZH(p);
    case 1047: // З
        return get_dist_to_cyrillic_Z(p);
    case 1048: // И
        return get_dist_to_cyrillic_I(p);
    case 1049: // Й
        return get_dist_to_cyrillic_I_KRATKOYE(p);
    case 1050: // К
        return get_dist_to_cyrillic_K(p);
    case 1051: // Л
        return get_dist_to_cyrillic_L(p);
    case 1052: // М
        return get_dist_to_cyrillic_M(p);
    case 1053: // Н
        return get_dist_to_cyrillic_N(p);
    case 1054: // О
        return get_dist_to_cyrillic_O(p);
    case 1055: // П
        return get_dist_to_cyrillic_P(p);
    case 1056: // Р
        return get_dist_to_cyrillic_R(p);
    case 1057: // С
        return get_dist_to_cyrillic_S(p);
    case 1058: // Т
        return get_dist_to_cyrillic_T(p);
    case 1059: // У
        return get_dist_to_cyrillic_U(p);
    case 1060: // Ф
        return get_dist_to_cyrillic_F(p);
    case 1061: // Х
        return get_dist_to_cyrillic_H(p);
    case 1062: // Ц
        return get_dist_to_cyrillic_TS(p);
    case 1063: // Ч
        return get_dist_to_cyrillic_CH(p);
    case 1064: // Ш
        return get_dist_to_cyrillic_SH(p);
    case 1065: // Щ
        return get_dist_to_cyrillic_SHCH(p);
    case 1066: // Ъ
        return get_dist_to_cyrillic_HARD_SIGN(p);
    case 1067: // Ы
        return get_dist_to_cyrillic_Y(p);
    case 1068: // Ь
        return get_dist_to_cyrillic_SOFT_SIGN(p);
    case 1069: // Э
        return get_dist_to_cyrillic_EH(p);
    case 1070: // Ю
        return get_dist_to_cyrillic_YU(p);
    case 1071: // Я
        return get_dist_to_cyrillic_YA(p);
    // -------------------------------------------------------------------
    // Punctuation
    case 45: // -
        return get_dist_to_dash(p);
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
    float y = u_zoomout;
    for (uint i = 0; i < N_TEXT_LINES; ++i, y -= u_text_spacing.y) {
        for (uint j = 0; j < MAX_TEXT_LEN; ++j) {
            uint char_unicode_idx = u_text[i][j];

            if (char_unicode_idx == 0) {
                break;
            }

            vec2 char_pos = vec2(float(j) * (1.0 + u_text_spacing.x), y);
            vec2 p = 2.0 * (uv - char_pos) / u_char_scale;
            float char_dist = get_dist_to_latin_char(p, char_unicode_idx);
            dist = min(dist, char_dist);
        }
    }

    float line = get_line(dist, u_text_thickness, u_text_smoothness);
    vec3 color = u_text_color * line;
    fs_color = vec4(color, 1.0);
}
