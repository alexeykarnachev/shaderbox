// Segment-glyph face: every glyph is a min-union of strokes on a fixed lattice.
// Glyph-local cell: x in [-0.5, 0.5], y in [-1, 1] (width 1, height 2 units).
// Lattice anchors (inlined as literals — lib files splice FUNCTIONS only, no consts):
//   A(-.5,-1) B(-.5,-.5) C(-.5,0) D(-.5,.5) E(-.5,1) F(0,1) G(.5,1) H(.5,.5)
//   J(.5,0)  K(.5,-.5)  L(.5,-1) M(0,-1)   N(0,-.5) O(0,0) P(0,.5)
// Public surface: SB_sd_char. sbt_* names are library-private (not catalogued;
// reachable transitively — layout.glsl composes sbt_char_skel).

float sbt_qarc(vec2 p, vec2 c, float x_cond, float y_cond, vec2 e1, vec2 e2) {
    vec2 v = (p - c) / 0.5;
    float dist = abs(length(v) - 1.0) * 0.5;
    float in_quad_x = mix(1.0 - step(v.x, 0.0), step(v.x, 0.0), x_cond);
    float in_quad_y = mix(1.0 - step(v.y, 0.0), step(v.y, 0.0), y_cond);
    float in_quad = in_quad_x * in_quad_y;
    return mix(min(distance(p, e1), distance(p, e2)), dist, in_quad);
}

float sbt_dot(vec2 p, vec2 c) { return distance(p, c) - 0.12; }

// Tall quarter-ellipse (rx=0.5, ry=1.0): a full-height sweep from a horizontal
// start to a vertical end — the arm shape the small qarc can't make.
float sbt_qarc_tall(vec2 p, vec2 c, float x_cond, float y_cond, vec2 e1, vec2 e2) {
    vec2 v = (p - c) / vec2(0.5, 1.0);
    float len = max(length(v), 0.0001);
    // First-order true distance |phi|/|grad phi| — the naive abs(len-1)*0.5 underestimates
    // by up to 2x near the vertical endpoints (rendered arc-armed glyphs visibly bolder).
    float dist = abs(len - 1.0) * len / max(length(vec2(v.x / 0.5, v.y)), 0.0001);
    float in_quad_x = mix(1.0 - step(v.x, 0.0), step(v.x, 0.0), x_cond);
    float in_quad_y = mix(1.0 - step(v.y, 0.0), step(v.y, 0.0), y_cond);
    float in_quad = in_quad_x * in_quad_y;
    return mix(min(distance(p, e1), distance(p, e2)), dist, in_quad);
}

float sbt_seg0(vec2 p) { return SB_sd_segment(p, vec2(-0.5, -1.0), vec2(-0.5, -0.5)); }
float sbt_seg1(vec2 p) { return SB_sd_segment(p, vec2(-0.5, -0.5), vec2(-0.5, 0.0)); }
float sbt_seg2(vec2 p) { return SB_sd_segment(p, vec2(-0.5, 0.0), vec2(-0.5, 0.5)); }
float sbt_seg3(vec2 p) { return SB_sd_segment(p, vec2(-0.5, 0.5), vec2(-0.5, 1.0)); }
float sbt_seg4(vec2 p) { return SB_sd_segment(p, vec2(-0.5, 1.0), vec2(0.0, 1.0)); }
float sbt_seg5(vec2 p) { return SB_sd_segment(p, vec2(0.0, 1.0), vec2(0.5, 1.0)); }
float sbt_seg6(vec2 p) { return SB_sd_segment(p, vec2(0.5, 1.0), vec2(0.5, 0.5)); }
float sbt_seg7(vec2 p) { return SB_sd_segment(p, vec2(0.5, 0.5), vec2(0.5, 0.0)); }
float sbt_seg8(vec2 p) { return SB_sd_segment(p, vec2(0.5, 0.0), vec2(0.5, -0.5)); }
float sbt_seg9(vec2 p) { return SB_sd_segment(p, vec2(0.5, -0.5), vec2(0.5, -1.0)); }
float sbt_seg10(vec2 p) { return SB_sd_segment(p, vec2(0.0, -1.0), vec2(0.5, -1.0)); }
float sbt_seg11(vec2 p) { return SB_sd_segment(p, vec2(-0.5, -1.0), vec2(0.0, -1.0)); }
float sbt_seg12(vec2 p) { return SB_sd_segment(p, vec2(0.0, -1.0), vec2(0.0, -0.5)); }
float sbt_seg14(vec2 p) { return SB_sd_segment(p, vec2(0.0, -0.5), vec2(0.0, 0.0)); }
float sbt_seg16(vec2 p) { return SB_sd_segment(p, vec2(0.0, 0.0), vec2(0.0, 0.5)); }
float sbt_seg18(vec2 p) { return SB_sd_segment(p, vec2(0.0, 0.5), vec2(0.0, 1.0)); }
float sbt_seg20(vec2 p) { return SB_sd_segment(p, vec2(-0.5, -1.0), vec2(0.0, 0.0)); }
float sbt_seg21(vec2 p) { return SB_sd_segment(p, vec2(-0.5, 1.0), vec2(0.0, 0.0)); }
float sbt_seg22(vec2 p) { return SB_sd_segment(p, vec2(0.0, 0.0), vec2(0.5, 1.0)); }
float sbt_seg23(vec2 p) { return SB_sd_segment(p, vec2(0.0, 0.0), vec2(0.5, -1.0)); }
float sbt_seg24(vec2 p) { return SB_sd_segment(p, vec2(-0.5, 0.0), vec2(0.0, 0.0)); }
float sbt_seg25(vec2 p) { return SB_sd_segment(p, vec2(0.0, 0.0), vec2(0.5, 0.0)); }

float sbt_seg26(vec2 p) {
    return sbt_qarc(p, vec2(0.0, -0.5), 1.0, 1.0, vec2(-0.5, -0.5), vec2(0.0, -1.0));
}
float sbt_seg27(vec2 p) {
    return sbt_qarc(p, vec2(0.0, -0.5), 1.0, 0.0, vec2(0.0, 0.0), vec2(-0.5, -0.5));
}
float sbt_seg28(vec2 p) {
    return sbt_qarc(p, vec2(0.0, -0.5), 0.0, 0.0, vec2(0.5, -0.5), vec2(0.0, 0.0));
}
float sbt_seg29(vec2 p) {
    return sbt_qarc(p, vec2(0.0, -0.5), 0.0, 1.0, vec2(0.0, -1.0), vec2(0.5, -0.5));
}
float sbt_seg30(vec2 p) {
    return sbt_qarc(p, vec2(0.0, 0.5), 1.0, 1.0, vec2(-0.5, 0.5), vec2(0.0, 0.0));
}
float sbt_seg31(vec2 p) {
    return sbt_qarc(p, vec2(0.0, 0.5), 1.0, 0.0, vec2(0.0, 1.0), vec2(-0.5, 0.5));
}
float sbt_seg32(vec2 p) {
    return sbt_qarc(p, vec2(0.0, 0.5), 0.0, 0.0, vec2(0.5, 0.5), vec2(0.0, 1.0));
}
float sbt_seg33(vec2 p) {
    return sbt_qarc(p, vec2(0.0, 0.5), 0.0, 1.0, vec2(0.0, 0.0), vec2(0.5, 0.5));
}

float sbt_latin_A(vec2 p) {
    float d = sbt_seg0(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg31(p));
    d = min(d, sbt_seg32(p));
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg9(p));
    d = min(d, sbt_seg24(p));
    d = min(d, sbt_seg25(p));
    return d;
}

float sbt_latin_B(vec2 p) {
    float d = sbt_seg0(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg3(p));
    d = min(d, sbt_seg4(p));
    d = min(d, sbt_seg32(p));
    d = min(d, sbt_seg33(p));
    d = min(d, sbt_seg24(p));
    d = min(d, sbt_seg28(p));
    d = min(d, sbt_seg29(p));
    d = min(d, sbt_seg11(p));
    return d;
}

float sbt_latin_C(vec2 p) {
    float d = sbt_seg29(p);
    d = min(d, sbt_seg26(p));
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg31(p));
    d = min(d, sbt_seg32(p));
    return d;
}

float sbt_latin_D(vec2 p) {
    float d = sbt_seg0(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg3(p));
    d = min(d, sbt_seg4(p));
    d = min(d, sbt_seg32(p));
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg29(p));
    d = min(d, sbt_seg11(p));
    return d;
}

float sbt_latin_E(vec2 p) {
    float d = sbt_seg0(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg3(p));
    d = min(d, sbt_seg4(p));
    d = min(d, sbt_seg5(p));
    d = min(d, sbt_seg24(p));
    d = min(d, sbt_seg25(p));
    d = min(d, sbt_seg11(p));
    d = min(d, sbt_seg10(p));
    return d;
}

float sbt_latin_F(vec2 p) {
    float d = sbt_seg0(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg3(p));
    d = min(d, sbt_seg4(p));
    d = min(d, sbt_seg5(p));
    d = min(d, sbt_seg24(p));
    return d;
}

float sbt_latin_G(vec2 p) {
    float d = sbt_seg26(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg31(p));
    d = min(d, sbt_seg5(p));
    d = min(d, sbt_seg29(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg25(p));
    return d;
}

float sbt_latin_H(vec2 p) {
    float d = sbt_seg0(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg3(p));
    d = min(d, sbt_seg24(p));
    d = min(d, sbt_seg25(p));
    d = min(d, sbt_seg9(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg6(p));
    return d;
}

float sbt_latin_I(vec2 p) {
    float d = sbt_seg12(p);
    d = min(d, sbt_seg14(p));
    d = min(d, sbt_seg16(p));
    d = min(d, sbt_seg18(p));
    d = min(d, sbt_seg4(p));
    d = min(d, sbt_seg5(p));
    d = min(d, sbt_seg11(p));
    d = min(d, sbt_seg10(p));
    return d;
}

float sbt_latin_J(vec2 p) {
    float d = sbt_seg8(p);
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg6(p));
    d = min(d, sbt_seg5(p));
    d = min(d, sbt_seg29(p));
    d = min(d, sbt_seg26(p));
    return d;
}

float sbt_latin_K(vec2 p) {
    float d = sbt_seg0(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg3(p));
    d = min(d, sbt_seg24(p));
    d = min(d, sbt_seg22(p));
    d = min(d, sbt_seg23(p));
    return d;
}

float sbt_latin_L(vec2 p) {
    float d = sbt_seg0(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg3(p));
    d = min(d, sbt_seg11(p));
    d = min(d, sbt_seg10(p));
    return d;
}

float sbt_latin_M(vec2 p) {
    float d = sbt_seg0(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg3(p));
    d = min(d, sbt_seg21(p));
    d = min(d, sbt_seg22(p));
    d = min(d, sbt_seg6(p));
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg9(p));
    return d;
}

float sbt_latin_N(vec2 p) {
    float d = sbt_seg0(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg3(p));
    d = min(d, sbt_seg21(p));
    d = min(d, sbt_seg23(p));
    d = min(d, sbt_seg6(p));
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg9(p));
    return d;
}

float sbt_latin_O(vec2 p) {
    float d = sbt_seg26(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg31(p));
    d = min(d, sbt_seg32(p));
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg29(p));
    return d;
}

float sbt_latin_P(vec2 p) {
    float d = sbt_seg0(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg3(p));
    d = min(d, sbt_seg4(p));
    d = min(d, sbt_seg32(p));
    d = min(d, sbt_seg33(p));
    d = min(d, sbt_seg24(p));
    return d;
}

float sbt_latin_Q(vec2 p) {
    float d = sbt_seg26(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg31(p));
    d = min(d, sbt_seg32(p));
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg29(p));
    d = min(d, sbt_seg23(p));
    return d;
}

float sbt_latin_R(vec2 p) {
    float d = sbt_seg0(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg3(p));
    d = min(d, sbt_seg4(p));
    d = min(d, sbt_seg32(p));
    d = min(d, sbt_seg33(p));
    d = min(d, sbt_seg23(p));
    d = min(d, sbt_seg24(p));
    return d;
}

float sbt_latin_S(vec2 p) {
    float d = sbt_seg32(p);
    d = min(d, sbt_seg31(p));
    d = min(d, sbt_seg30(p));
    d = min(d, sbt_seg28(p));
    d = min(d, sbt_seg29(p));
    d = min(d, sbt_seg26(p));
    return d;
}

float sbt_latin_T(vec2 p) {
    float d = sbt_seg12(p);
    d = min(d, sbt_seg14(p));
    d = min(d, sbt_seg16(p));
    d = min(d, sbt_seg18(p));
    d = min(d, sbt_seg4(p));
    d = min(d, sbt_seg5(p));
    return d;
}

float sbt_latin_U(vec2 p) {
    float d = sbt_seg1(p);
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg3(p));
    d = min(d, sbt_seg26(p));
    d = min(d, sbt_seg29(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg6(p));
    return d;
}

float sbt_latin_V(vec2 p) {
    float d = sbt_seg0(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg3(p));
    d = min(d, sbt_seg20(p));
    d = min(d, sbt_seg22(p));
    return d;
}

float sbt_latin_W(vec2 p) {
    float d = sbt_seg0(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg3(p));
    d = min(d, sbt_seg20(p));
    d = min(d, sbt_seg23(p));
    d = min(d, sbt_seg9(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg6(p));
    return d;
}

float sbt_latin_X(vec2 p) {
    float d = sbt_seg20(p);
    d = min(d, sbt_seg22(p));
    d = min(d, sbt_seg21(p));
    d = min(d, sbt_seg23(p));
    return d;
}

float sbt_latin_Y(vec2 p) {
    float d = sbt_seg12(p);
    d = min(d, sbt_seg14(p));
    d = min(d, sbt_seg21(p));
    d = min(d, sbt_seg22(p));
    return d;
}

float sbt_latin_Z(vec2 p) {
    float d = sbt_seg4(p);
    d = min(d, sbt_seg5(p));
    d = min(d, sbt_seg22(p));
    d = min(d, sbt_seg20(p));
    d = min(d, sbt_seg11(p));
    d = min(d, sbt_seg10(p));
    return d;
}

// Digits on the same lattice (7-segment style; 0 is slashed to differ from O).
float sbt_digit_0(vec2 p) {
    float d = sbt_latin_O(p);
    d = min(d, sbt_seg20(p));
    d = min(d, sbt_seg22(p));
    return d;
}

float sbt_digit_1(vec2 p) {
    float d = sbt_seg12(p);
    d = min(d, sbt_seg14(p));
    d = min(d, sbt_seg16(p));
    d = min(d, sbt_seg18(p));
    d = min(d, sbt_seg4(p));
    d = min(d, sbt_seg11(p));
    d = min(d, sbt_seg10(p));
    return d;
}

float sbt_digit_2(vec2 p) {
    float d = sbt_seg31(p);
    d = min(d, sbt_seg32(p));
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg25(p));
    d = min(d, sbt_seg27(p));
    d = min(d, sbt_seg0(p));
    d = min(d, sbt_seg11(p));
    d = min(d, sbt_seg10(p));
    return d;
}

float sbt_digit_3(vec2 p) {
    float d = sbt_seg31(p);
    d = min(d, sbt_seg32(p));
    d = min(d, sbt_seg33(p));
    d = min(d, sbt_seg25(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg9(p));
    d = min(d, sbt_seg10(p));
    d = min(d, sbt_seg11(p));
    return d;
}

float sbt_digit_4(vec2 p) {
    float d = sbt_seg2(p);
    d = min(d, sbt_seg3(p));
    d = min(d, sbt_seg24(p));
    d = min(d, sbt_seg25(p));
    d = min(d, sbt_seg6(p));
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg9(p));
    return d;
}

float sbt_digit_5(vec2 p) {
    float d = sbt_seg4(p);
    d = min(d, sbt_seg5(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg3(p));
    d = min(d, sbt_seg24(p));
    d = min(d, sbt_seg28(p));
    d = min(d, sbt_seg29(p));
    d = min(d, sbt_seg11(p));
    return d;
}

float sbt_digit_6(vec2 p) {
    float d = sbt_seg31(p);
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg26(p));
    d = min(d, sbt_seg27(p));
    d = min(d, sbt_seg28(p));
    d = min(d, sbt_seg29(p));
    return d;
}

float sbt_digit_7(vec2 p) {
    float d = sbt_seg4(p);
    d = min(d, sbt_seg5(p));
    d = min(d, sbt_seg22(p));
    d = min(d, sbt_seg14(p));
    d = min(d, sbt_seg12(p));
    return d;
}

float sbt_digit_8(vec2 p) {
    float d = sbt_seg30(p);
    d = min(d, sbt_seg31(p));
    d = min(d, sbt_seg32(p));
    d = min(d, sbt_seg33(p));
    d = min(d, sbt_seg26(p));
    d = min(d, sbt_seg27(p));
    d = min(d, sbt_seg28(p));
    d = min(d, sbt_seg29(p));
    return d;
}

float sbt_digit_9(vec2 p) {
    float d = sbt_seg30(p);
    d = min(d, sbt_seg31(p));
    d = min(d, sbt_seg32(p));
    d = min(d, sbt_seg33(p));
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg29(p));
    d = min(d, sbt_seg26(p));
    return d;
}

float sbt_dash(vec2 p) {
    float d = sbt_seg24(p);
    d = min(d, sbt_seg25(p));
    return d;
}

float sbt_period(vec2 p) {
    return sbt_dot(p, vec2(0.0, -0.88));
}

float sbt_comma(vec2 p) {
    vec2 head = vec2(0.0, -0.5);
    float d = sbt_dot(p, head);
    d = min(d, SB_sd_segment(p, head, vec2(-0.125, -1.0)));
    return d;
}

float sbt_semicolon(vec2 p) {
    float d = sbt_dot(p, vec2(0.0, 0.0));
    d = min(d, sbt_comma(p));
    return d;
}

float sbt_colon(vec2 p) {
    float d = sbt_dot(p, vec2(0.0, -0.88));
    d = min(d, sbt_dot(p, vec2(0.0, 0.0)));
    return d;
}

float sbt_exclaim(vec2 p) {
    float d = sbt_seg16(p);
    d = min(d, sbt_seg18(p));
    d = min(d, sbt_dot(p, vec2(0.0, -0.88)));
    return d;
}

float sbt_question(vec2 p) {
    float d = sbt_seg31(p);
    d = min(d, sbt_seg32(p));
    d = min(d, sbt_seg33(p));
    d = min(d, sbt_seg14(p));
    d = min(d, sbt_dot(p, vec2(0.0, -0.88)));
    return d;
}

float sbt_apostrophe(vec2 p) {
    return SB_sd_segment(p, vec2(0.0, 1.0), vec2(-0.15, 0.6));
}

float sbt_ampersand(vec2 p) {
    float d = sbt_seg30(p);
    d = min(d, sbt_seg31(p));
    d = min(d, sbt_seg32(p));
    d = min(d, sbt_seg33(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg20(p));
    d = min(d, sbt_seg25(p));
    return d;
}

// Cyrillic glyphs. Letters identical to latin shapes dispatch straight to those
// (А=A, В=B, Е=E, К=K, М=M, Н=H, О=O, Р=P, С=C, Т=T, Х=X); У/З have their own
// rounded forms (sbt_cyr_u / sbt_cyr_ze). Strokes the lattice lacks (curled legs,
// breve, loop, tails) are inline segments/arcs; marks may overshoot the cell
// slightly (into line spacing) like the comma tail does.

float sbt_cyr_be(vec2 p) {
    float d = sbt_seg0(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg3(p));
    d = min(d, sbt_seg4(p));
    d = min(d, sbt_seg5(p));
    d = min(d, sbt_seg24(p));
    d = min(d, sbt_seg28(p));
    d = min(d, sbt_seg29(p));
    d = min(d, sbt_seg11(p));
    return d;
}

float sbt_cyr_ghe(vec2 p) {
    float d = sbt_seg0(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg3(p));
    d = min(d, sbt_seg4(p));
    d = min(d, sbt_seg5(p));
    return d;
}

float sbt_cyr_el(vec2 p) {
    float d = sbt_seg0(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg31(p));
    d = min(d, sbt_seg5(p));
    d = min(d, sbt_seg6(p));
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg9(p));
    return d;
}

float sbt_cyr_de(vec2 p) {
    float d = sbt_cyr_el(p);
    d = min(d, sbt_seg11(p));
    d = min(d, sbt_seg10(p));
    d = min(d, SB_sd_segment(p, vec2(-0.5, -1.0), vec2(-0.5, -1.22)));
    d = min(d, SB_sd_segment(p, vec2(0.5, -1.0), vec2(0.5, -1.22)));
    return d;
}

float sbt_cyr_yo(vec2 p) {
    float d = sbt_latin_E(p);
    d = min(d, distance(p, vec2(-0.22, 1.32)) - 0.07);
    d = min(d, distance(p, vec2(0.22, 1.32)) - 0.07);
    return d;
}

float sbt_cyr_zhe(vec2 p) {
    float d = sbt_seg12(p);
    d = min(d, sbt_seg14(p));
    d = min(d, sbt_seg16(p));
    d = min(d, sbt_seg18(p));
    d = min(d, sbt_qarc_tall(p, vec2(0.0, 1.0), 0.0, 1.0, vec2(0.0, 0.0), vec2(0.5, 1.0)));
    d = min(d, sbt_qarc_tall(p, vec2(0.0, 1.0), 1.0, 1.0, vec2(0.0, 0.0), vec2(-0.5, 1.0)));
    d = min(d, sbt_qarc_tall(p, vec2(0.0, -1.0), 0.0, 0.0, vec2(0.0, 0.0), vec2(0.5, -1.0)));
    d = min(d, sbt_qarc_tall(p, vec2(0.0, -1.0), 1.0, 0.0, vec2(0.0, 0.0), vec2(-0.5, -1.0)));
    return d;
}

float sbt_cyr_ze(vec2 p) {
    float d = sbt_seg31(p);
    d = min(d, sbt_seg32(p));
    d = min(d, sbt_seg33(p));
    d = min(d, sbt_seg28(p));
    d = min(d, sbt_seg29(p));
    d = min(d, sbt_seg26(p));
    return d;
}

float sbt_cyr_u(vec2 p) {
    float d = sbt_seg3(p);
    d = min(d, sbt_seg30(p));
    d = min(d, sbt_seg25(p));
    d = min(d, sbt_seg6(p));
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg29(p));
    d = min(d, sbt_seg26(p));
    return d;
}

float sbt_cyr_i(vec2 p) {
    float d = sbt_seg0(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg3(p));
    d = min(d, sbt_seg6(p));
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg9(p));
    d = min(d, sbt_seg20(p));
    d = min(d, sbt_seg22(p));
    return d;
}

float sbt_cyr_short_i(vec2 p) {
    float d = sbt_cyr_i(p);
    d = min(d, SB_sd_segment(p, vec2(-0.26, 1.38), vec2(0.0, 1.1)));
    d = min(d, SB_sd_segment(p, vec2(0.0, 1.1), vec2(0.26, 1.38)));
    return d;
}

float sbt_cyr_pe(vec2 p) {
    float d = sbt_seg0(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg31(p));
    d = min(d, sbt_seg32(p));
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg9(p));
    return d;
}

float sbt_cyr_ef(vec2 p) {
    float d = sbt_seg12(p);
    d = min(d, sbt_seg14(p));
    d = min(d, sbt_seg16(p));
    d = min(d, sbt_seg18(p));
    d = min(d, sbt_seg30(p));
    d = min(d, sbt_seg31(p));
    d = min(d, sbt_seg32(p));
    d = min(d, sbt_seg33(p));
    return d;
}

float sbt_cyr_tse(vec2 p) {
    float d = sbt_seg1(p);
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg3(p));
    d = min(d, sbt_seg26(p));
    d = min(d, sbt_seg10(p));
    d = min(d, sbt_seg6(p));
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg9(p));
    d = min(d, SB_sd_segment(p, vec2(0.5, -1.0), vec2(0.62, -1.22)));
    return d;
}

float sbt_cyr_che(vec2 p) {
    float d = sbt_seg3(p);
    d = min(d, sbt_seg30(p));
    d = min(d, sbt_seg25(p));
    d = min(d, sbt_seg6(p));
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg9(p));
    return d;
}

float sbt_cyr_sha(vec2 p) {
    float d = sbt_seg0(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg3(p));
    d = min(d, sbt_seg12(p));
    d = min(d, sbt_seg14(p));
    d = min(d, sbt_seg16(p));
    d = min(d, sbt_seg18(p));
    d = min(d, sbt_seg6(p));
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg9(p));
    d = min(d, sbt_seg11(p));
    d = min(d, sbt_seg10(p));
    return d;
}

float sbt_cyr_shcha(vec2 p) {
    float d = sbt_cyr_sha(p);
    d = min(d, SB_sd_segment(p, vec2(0.5, -1.0), vec2(0.62, -1.22)));
    return d;
}

float sbt_cyr_hard_sign(vec2 p) {
    float d = sbt_cyr_soft_sign(p);
    d = min(d, SB_sd_segment(p, vec2(-0.5, 1.0), vec2(-0.85, 1.0)));
    return d;
}

float sbt_cyr_soft_sign(vec2 p) {
    float d = sbt_seg0(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg3(p));
    d = min(d, sbt_seg24(p));
    d = min(d, sbt_seg28(p));
    d = min(d, sbt_seg29(p));
    d = min(d, sbt_seg11(p));
    return d;
}

float sbt_cyr_yeru(vec2 p) {
    float d = sbt_cyr_soft_sign(p);
    d = min(d, sbt_seg6(p));
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg9(p));
    return d;
}

float sbt_cyr_e(vec2 p) {
    float d = sbt_seg4(p);
    d = min(d, sbt_seg32(p));
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg29(p));
    d = min(d, sbt_seg11(p));
    d = min(d, sbt_seg25(p));
    return d;
}

float sbt_cyr_yu(vec2 p) {
    float d = sbt_seg0(p);
    d = min(d, sbt_seg1(p));
    d = min(d, sbt_seg2(p));
    d = min(d, sbt_seg3(p));
    d = min(d, sbt_seg24(p));
    d = min(d, sbt_seg12(p));
    d = min(d, sbt_seg14(p));
    d = min(d, sbt_seg16(p));
    d = min(d, sbt_seg18(p));
    d = min(d, sbt_seg32(p));
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg29(p));
    return d;
}

float sbt_cyr_ya(vec2 p) {
    float d = sbt_seg6(p);
    d = min(d, sbt_seg7(p));
    d = min(d, sbt_seg8(p));
    d = min(d, sbt_seg9(p));
    d = min(d, sbt_seg5(p));
    d = min(d, sbt_seg31(p));
    d = min(d, sbt_seg30(p));
    d = min(d, sbt_seg25(p));
    d = min(d, sbt_seg20(p));
    return d;
}

float sbt_char_skel(vec2 p, uint codepoint) {
    if (codepoint >= 97u && codepoint <= 122u) {
        codepoint -= 32u;
    }
    if (codepoint >= 1072u && codepoint <= 1103u) {
        codepoint -= 32u;
    }
    if (codepoint == 1105u) {
        codepoint = 1025u;
    }

    switch (codepoint) {
    case 65u: return sbt_latin_A(p);
    case 66u: return sbt_latin_B(p);
    case 67u: return sbt_latin_C(p);
    case 68u: return sbt_latin_D(p);
    case 69u: return sbt_latin_E(p);
    case 70u: return sbt_latin_F(p);
    case 71u: return sbt_latin_G(p);
    case 72u: return sbt_latin_H(p);
    case 73u: return sbt_latin_I(p);
    case 74u: return sbt_latin_J(p);
    case 75u: return sbt_latin_K(p);
    case 76u: return sbt_latin_L(p);
    case 77u: return sbt_latin_M(p);
    case 78u: return sbt_latin_N(p);
    case 79u: return sbt_latin_O(p);
    case 80u: return sbt_latin_P(p);
    case 81u: return sbt_latin_Q(p);
    case 82u: return sbt_latin_R(p);
    case 83u: return sbt_latin_S(p);
    case 84u: return sbt_latin_T(p);
    case 85u: return sbt_latin_U(p);
    case 86u: return sbt_latin_V(p);
    case 87u: return sbt_latin_W(p);
    case 88u: return sbt_latin_X(p);
    case 89u: return sbt_latin_Y(p);
    case 90u: return sbt_latin_Z(p);
    case 48u: return sbt_digit_0(p);
    case 49u: return sbt_digit_1(p);
    case 50u: return sbt_digit_2(p);
    case 51u: return sbt_digit_3(p);
    case 52u: return sbt_digit_4(p);
    case 53u: return sbt_digit_5(p);
    case 54u: return sbt_digit_6(p);
    case 55u: return sbt_digit_7(p);
    case 56u: return sbt_digit_8(p);
    case 57u: return sbt_digit_9(p);
    case 33u: return sbt_exclaim(p);
    case 38u: return sbt_ampersand(p);
    case 39u: return sbt_apostrophe(p);
    case 44u: return sbt_comma(p);
    case 45u: return sbt_dash(p);
    case 46u: return sbt_period(p);
    case 58u: return sbt_colon(p);
    case 59u: return sbt_semicolon(p);
    case 63u: return sbt_question(p);
    case 1040u: return sbt_latin_A(p);
    case 1041u: return sbt_cyr_be(p);
    case 1042u: return sbt_latin_B(p);
    case 1043u: return sbt_cyr_ghe(p);
    case 1044u: return sbt_cyr_de(p);
    case 1045u: return sbt_latin_E(p);
    case 1025u: return sbt_cyr_yo(p);
    case 1046u: return sbt_cyr_zhe(p);
    case 1047u: return sbt_cyr_ze(p);
    case 1048u: return sbt_cyr_i(p);
    case 1049u: return sbt_cyr_short_i(p);
    case 1050u: return sbt_latin_K(p);
    case 1051u: return sbt_cyr_el(p);
    case 1052u: return sbt_latin_M(p);
    case 1053u: return sbt_latin_H(p);
    case 1054u: return sbt_latin_O(p);
    case 1055u: return sbt_cyr_pe(p);
    case 1056u: return sbt_latin_P(p);
    case 1057u: return sbt_latin_C(p);
    case 1058u: return sbt_latin_T(p);
    case 1059u: return sbt_cyr_u(p);
    case 1060u: return sbt_cyr_ef(p);
    case 1061u: return sbt_latin_X(p);
    case 1062u: return sbt_cyr_tse(p);
    case 1063u: return sbt_cyr_che(p);
    case 1064u: return sbt_cyr_sha(p);
    case 1065u: return sbt_cyr_shcha(p);
    case 1066u: return sbt_cyr_hard_sign(p);
    case 1067u: return sbt_cyr_yeru(p);
    case 1068u: return sbt_cyr_soft_sign(p);
    case 1069u: return sbt_cyr_e(p);
    case 1070u: return sbt_cyr_yu(p);
    case 1071u: return sbt_cyr_ya(p);
    default: return 100000.0;
    }
}

/// SIGNED distance (negative inside the ink) to one glyph in glyph-local coords
/// (cell x in [-0.5,0.5], y in [-1,1]). `weight` = stroke half-width in the same
/// LOCAL units (0.1 regular .. 0.25 bold). To place a glyph of height ch at uv
/// position c: SB_sd_char((uv - c) / (0.5*ch), cp, 0.1) * (0.5*ch) — the *0.5*ch
/// converts the result back to uv units. Lowercase folds to uppercase; supports
/// A-Z, Cyrillic А-Я/Ё, 0-9 and ! ? : ; , . - ' &. Unknown codepoints draw nothing.
float SB_sd_char(vec2 p, uint codepoint, float weight) {
    return sbt_char_skel(p, codepoint) - weight;
}
