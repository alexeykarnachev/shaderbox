#version 460 core
#define PI 3.141592
#define MAX_DIST 9999.9
#define MIN_SMOOTHNESS 0.0001
#define MAX_TEXT_LEN 64
#define N_TEXT_LINES 3

in vec2 vs_uv;
out vec4 fs_color;

uniform float u_time;
uniform float u_aspect;
uniform vec2 u_resolution;

uniform sampler2D u_photo;
uniform sampler2D u_depth;
uniform sampler2D u_background;

uniform float plane_speed = 0.1;

uniform float u_zoomout = 10.0;
uniform float u_char_scale = 0.5;
uniform vec2 u_offset = vec2(0.0, 0.0);
uniform vec2 u_text_spacing = vec2(0.0, 0.0);

uniform float u_text_thickness = 0.05;
uniform float u_text_smoothness = 0.01;
uniform vec3 u_text_color = vec3(1.0, 0.0, 0.0);

uniform uint u_text[N_TEXT_LINES][MAX_TEXT_LEN]; // Changed to 2D array

const vec2 poisson_disk87[87] = vec2[87](vec2(-0.488690, 0.046349),
                                         vec2(0.496064, 0.018367),
                                         vec2(-0.027347, -0.461505),
                                         vec2(-0.090074, 0.490283),
                                         vec2(0.294474, 0.366950),
                                         vec2(0.305608, -0.360041),
                                         vec2(-0.346198, -0.357278),
                                         vec2(-0.308924, 0.353038),
                                         vec2(-0.437547, -0.177748),
                                         vec2(0.446996, -0.129850),
                                         vec2(0.117621, -0.444649),
                                         vec2(0.171424, 0.418258),
                                         vec2(-0.227789, -0.410446),
                                         vec2(0.210264, -0.422608),
                                         vec2(-0.414136, -0.268376),
                                         vec2(0.368202, 0.316549),
                                         vec2(-0.480689, 0.127069),
                                         vec2(0.481128, -0.056358),
                                         vec2(-0.458004, -0.063002),
                                         vec2(0.409361, 0.201972),
                                         vec2(-0.176597, 0.424044),
                                         vec2(-0.095380, -0.441734),
                                         vec2(0.326086, -0.280594),
                                         vec2(-0.411327, 0.184757),
                                         vec2(-0.291534, -0.300406),
                                         vec2(0.400901, -0.002308),
                                         vec2(0.020255, 0.445511),
                                         vec2(0.302251, 0.275637),
                                         vec2(0.387805, -0.223370),
                                         vec2(-0.378395, 0.062614),
                                         vec2(0.405052, 0.101681),
                                         vec2(-0.010340, -0.355322),
                                         vec2(-0.034931, 0.383699),
                                         vec2(-0.318953, -0.225899),
                                         vec2(0.349283, -0.140001),
                                         vec2(-0.253974, 0.299183),
                                         vec2(0.188226, 0.342914),
                                         vec2(0.212083, -0.294545),
                                         vec2(-0.188320, -0.308466),
                                         vec2(-0.373708, -0.070538),
                                         vec2(0.114322, -0.356677),
                                         vec2(-0.154401, 0.348207),
                                         vec2(-0.321713, 0.260043),
                                         vec2(-0.086797, -0.349277),
                                         vec2(-0.360294, -0.144808),
                                         vec2(-0.323996, 0.188199),
                                         vec2(0.277830, -0.204128),
                                         vec2(0.087828, 0.351992),
                                         vec2(-0.215777, -0.234955),
                                         vec2(0.291437, 0.171860),
                                         vec2(0.027249, -0.255925),
                                         vec2(-0.316361, -0.013941),
                                         vec2(0.346679, -0.066942),
                                         vec2(-0.103280, -0.273636),
                                         vec2(-0.017802, 0.310973),
                                         vec2(-0.280809, -0.120043),
                                         vec2(-0.282912, 0.117500),
                                         vec2(0.267574, -0.036973),
                                         vec2(-0.034965, -0.223502),
                                         vec2(0.109677, 0.256372),
                                         vec2(-0.204519, -0.116846),
                                         vec2(0.144105, -0.181736),
                                         vec2(-0.140560, 0.215101),
                                         vec2(0.271573, 0.102406),
                                         vec2(0.220437, 0.203459),
                                         vec2(-0.242979, -0.027494),
                                         vec2(-0.050135, 0.239871),
                                         vec2(-0.152652, -0.193125),
                                         vec2(-0.220532, 0.179600),
                                         vec2(0.216867, -0.096770),
                                         vec2(-0.164884, 0.122109),
                                         vec2(0.251078, 0.034090),
                                         vec2(0.016515, -0.175206),
                                         vec2(0.042304, 0.216117),
                                         vec2(-0.133933, -0.060601),
                                         vec2(0.184659, 0.135680),
                                         vec2(-0.161273, 0.024207),
                                         vec2(-0.056532, -0.154410),
                                         vec2(-0.082706, 0.083129),
                                         vec2(0.081409, -0.088060),
                                         vec2(0.115078, 0.156566),
                                         vec2(0.133209, 0.061211),
                                         vec2(0.002618, -0.101328),
                                         vec2(0.132926, -0.013988),
                                         vec2(-0.027172, -0.017586),
                                         vec2(0.022969, 0.116469),
                                         vec2(0.036262, 0.015085));

// Simple pseudo-random hash function
float hash(vec2 p) {
    p = fract(p * vec2(123.45, 678.90));
    p += dot(p, p + vec2(45.67, 89.01));
    return fract(p.x * p.y * 43758.5453);
}

// Description : Array and textureless GLSL 2D/3D/4D simplex
//               noise functions.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : stegu
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//               https://github.com/stegu/webgl-noise

vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }

float mod289(float x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }

vec4 permute(vec4 x) { return mod289(((x * 34.0) + 10.0) * x); }

float permute(float x) { return mod289(((x * 34.0) + 10.0) * x); }

vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float taylorInvSqrt(float r) { return 1.79284291400159 - 0.85373472095314 * r; }

vec4 grad4(float j, vec4 ip) {
    const vec4 ones = vec4(1.0, 1.0, 1.0, -1.0);
    vec4 p, s;

    p.xyz = floor(fract(vec3(j) * ip.xyz) * 7.0) * ip.z - 1.0;
    p.w = 1.5 - dot(abs(p.xyz), ones.xyz);
    s = vec4(lessThan(p, vec4(0.0)));
    p.xyz = p.xyz + (s.xyz * 2.0 - 1.0) * s.www;

    return p;
}

// (sqrt(5) - 1)/4 = F4, used once below
#define F4 0.309016994374947451

float snoise(vec4 v) {
    const vec4 C = vec4(0.138196601125011,   // (5 - sqrt(5))/20  G4
                        0.276393202250021,   // 2 * G4
                        0.414589803375032,   // 3 * G4
                        -0.447213595499958); // -1 + 4 * G4

    // First corner
    vec4 i = floor(v + dot(v, vec4(F4)));
    vec4 x0 = v - i + dot(i, C.xxxx);

    // Other corners

    // Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly
    // ATI)
    vec4 i0;
    vec3 isX = step(x0.yzw, x0.xxx);
    vec3 isYZ = step(x0.zww, x0.yyz);
    //  i0.x = dot( isX, vec3( 1.0 ) );
    i0.x = isX.x + isX.y + isX.z;
    i0.yzw = 1.0 - isX;
    //  i0.y += dot( isYZ.xy, vec2( 1.0 ) );
    i0.y += isYZ.x + isYZ.y;
    i0.zw += 1.0 - isYZ.xy;
    i0.z += isYZ.z;
    i0.w += 1.0 - isYZ.z;

    // i0 now contains the unique values 0,1,2,3 in each channel
    vec4 i3 = clamp(i0, 0.0, 1.0);
    vec4 i2 = clamp(i0 - 1.0, 0.0, 1.0);
    vec4 i1 = clamp(i0 - 2.0, 0.0, 1.0);

    //  x0 = x0 - 0.0 + 0.0 * C.xxxx
    //  x1 = x0 - i1  + 1.0 * C.xxxx
    //  x2 = x0 - i2  + 2.0 * C.xxxx
    //  x3 = x0 - i3  + 3.0 * C.xxxx
    //  x4 = x0 - 1.0 + 4.0 * C.xxxx
    vec4 x1 = x0 - i1 + C.xxxx;
    vec4 x2 = x0 - i2 + C.yyyy;
    vec4 x3 = x0 - i3 + C.zzzz;
    vec4 x4 = x0 + C.wwww;

    // Permutations
    i = mod289(i);
    float j0 = permute(permute(permute(permute(i.w) + i.z) + i.y) + i.x);
    vec4 j1 =
        permute(permute(permute(permute(i.w + vec4(i1.w, i2.w, i3.w, 1.0)) +
                                i.z + vec4(i1.z, i2.z, i3.z, 1.0)) +
                        i.y + vec4(i1.y, i2.y, i3.y, 1.0)) +
                i.x + vec4(i1.x, i2.x, i3.x, 1.0));

    // Gradients: 7x7x6 points over a cube, mapped onto a 4-cross polytope
    // 7*7*6 = 294, which is close to the ring size 17*17 = 289.
    vec4 ip = vec4(1.0 / 294.0, 1.0 / 49.0, 1.0 / 7.0, 0.0);

    vec4 p0 = grad4(j0, ip);
    vec4 p1 = grad4(j1.x, ip);
    vec4 p2 = grad4(j1.y, ip);
    vec4 p3 = grad4(j1.z, ip);
    vec4 p4 = grad4(j1.w, ip);

    // Normalise gradients
    vec4 norm =
        taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
    p4 *= taylorInvSqrt(dot(p4, p4));

    // Mix contributions from the five corners
    vec3 m0 = max(0.57 - vec3(dot(x0, x0), dot(x1, x1), dot(x2, x2)), 0.0);
    vec2 m1 = max(0.57 - vec2(dot(x3, x3), dot(x4, x4)), 0.0);
    m0 = m0 * m0;
    m1 = m1 * m1;
    return 60.1 * (dot(m0 * m0, vec3(dot(p0, x0), dot(p1, x1), dot(p2, x2))) +
                   dot(m1 * m1, vec2(dot(p3, x3), dot(p4, x4))));
}

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

float get_line(float dist, float thickness, float smoothness) {
    smoothness = max(MIN_SMOOTHNESS, smoothness);
    return 1.0 - smoothstep(0.0, smoothness, dist - thickness);
}

float get_text_line(float line_width, float line_smoothness) {
    vec2 uv = (vs_uv - u_offset) * vec2(u_aspect, 1.0) * u_zoomout;

    float dist = MAX_DIST;
    float y = u_zoomout;
    for (uint i = 0; i < N_TEXT_LINES; ++i, y -= u_text_spacing.y) {
        for (uint j = 0; j < MAX_TEXT_LEN; ++j) {
            // uint char_unicode_idx = u_text[i][j];
            uint char_unicode_idx = 0;

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

    return line;
}

void main() {
    // Plane and ray setup
    float noise = snoise(vec4(4.0 * vs_uv.x, 128.0 * vs_uv.y, 0.0, 0.0));
    float plane_z = 2.0 * abs(fract(plane_speed * u_time) - 0.5);
    plane_z = pow(plane_z, 2.0);
    plane_z += 0.035 * noise;
    plane_z = clamp(plane_z, 0.1, 0.4);

    vec3 plane_p = vec3(0.0, 0.0, plane_z);
    vec3 plane_n = normalize(vec3(0.0, 0.0, 1.0));
    vec3 sp = vec3((vs_uv - 0.5) * vec2(u_aspect, 1.0), 0.0);

    vec3 cam_pos = vec3(0.15 * sin(0.33 * u_time * 2.0 * PI),
                        0.15 * cos(0.33 * u_time * 2.0 * PI),
                        -0.75);
    vec3 p = sp;
    vec3 ray_dir = normalize(sp - cam_pos);
    float plane_depth =
        (dot(plane_n, plane_p) - dot(plane_n, p)) / dot(plane_n, ray_dir);

    float max_n_steps = 10000.0;
    float step_size = 1.0 / max_n_steps;

    // Ray marching loop
    float nearest_depth_to_plane = 1.0;
    float step_idx = 0.0;
    vec2 plane_uv = vec2(0.0);
    vec3 color = vec3(1.0, 0.0, 0.0);
    float depth;
    vec2 uv;
    float diff = 0.0;
    while (true) {
        if (step_idx >= max_n_steps) {
            color = vec3(0.0, 0.0, 0.0);
            break;
        }

        uv = p.xy / vec2(u_aspect, 1.0) + 0.5;
        if (uv.x >= 1.0 || uv.x <= 0.0 || uv.y >= 1.0 || uv.y <= 0.0) {
            color = vec3(0.0, 0.0, 0.0);
            break;
        }

        depth = texture2D(u_depth, uv).r;
        depth = clamp(depth, 0.01, 0.99);

        plane_uv = p.xy / vec2(u_aspect) + 0.5;
        nearest_depth_to_plane =
            min(nearest_depth_to_plane, abs(p.z - plane_depth));

        diff = p.z - depth;
        if (diff >= 0.0) {
            color = texture2D(u_photo, uv).rgb;
            color *= texture2D(u_background, uv).r;
            break;
        }

        p += step_size * ray_dir;
        step_idx += 1.0;
    }

    // Apply plane effect
    float d = nearest_depth_to_plane;
    float plane_brightness = 2.0;
    float plane_intensity =
        plane_brightness / dot(vec3(1.0, 500.0, 1000.0), vec3(1.0, d, d * d));
    vec3 plane_color = vec3(0.12, 1.0, 0.1);

    color = plane_color * plane_intensity +
            clamp(1.0 - plane_intensity, 0.0, 1.0) * color;

    // Outline detection with Poisson disk sampling

    if (nearest_depth_to_plane <= step_size) {
        depth = texture2D(u_depth, uv).r;
        d = 1.0 - clamp(depth, 0.5, 1.0);
        color = d * vec3(0.0, 1.0, 0.0);

        float sample_radius = 8.0;
        int max_num_samples = 256;
        int num_samples = 0;

        float rand = hash(uv * u_resolution);
        vec2 rand_offset =
            vec2(cos(rand * 6.28318530718), sin(rand * 6.28318530718)) * 0.1;

        float center_depth = texture2D(u_depth, uv).r;
        float outline_strength = 0.0;
        for (int i = 0; i < max_num_samples; i++) {
            vec2 poisson_point = poisson_disk87[i] + rand_offset / u_resolution;
            vec2 sample_uv = uv + poisson_point * sample_radius / u_resolution;

            if (sample_uv.x >= 0.0 && sample_uv.x <= 1.0 &&
                sample_uv.y >= 0.0 && sample_uv.y <= 1.0) {
                float sample_depth = texture2D(u_depth, sample_uv).r;
                float depth_diff = abs(center_depth - sample_depth);
                outline_strength += depth_diff;
                num_samples += 1;
            }
        }

        outline_strength /= float(num_samples);

        outline_strength =
            (1.0 - depth) * smoothstep(0.0, 0.001, outline_strength);
        color = vec3(0.1, outline_strength, 0.05);
    }

    float alpha = 1.0;
    if (color.r < 0.01 && color.g < 0.01 && color.b < 0.01) {
        alpha = 0.0;
    }

    vec2 text_uv = (plane_uv - u_offset) * vec2(u_aspect, 1.0) * u_zoomout;

    float dist = MAX_DIST;
    float y = u_zoomout;
    for (uint i = 0; i < N_TEXT_LINES; ++i, y -= u_text_spacing.y) {
        for (uint j = 0; j < MAX_TEXT_LEN; ++j) {
            // uint char_unicode_idx = u_text[i][j];
            uint char_unicode_idx = 0;

            if (char_unicode_idx == 0) {
                break;
            }

            vec2 char_pos = vec2(float(j) * (1.0 + u_text_spacing.x), y);
            vec2 p = 2.0 * (text_uv - char_pos) / u_char_scale;
            float char_dist = get_dist_to_latin_char(p, char_unicode_idx);
            dist = min(dist, char_dist);
        }
    }

    float line = get_line(dist, u_text_thickness, u_text_smoothness);
    color = (1.0 - line) * color + line * u_text_color;

    fs_color = vec4(color, alpha);
}
