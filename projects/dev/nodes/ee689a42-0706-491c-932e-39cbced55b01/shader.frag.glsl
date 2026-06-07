#version 460 core

#define MAX_DIST 9999.9
#define MAX_TEXT_LEN 64

in vec2 vs_uv;
out vec4 fs_color;

uniform float u_time;
uniform float u_aspect;

uniform float u_char_scale     = 0.24;
uniform vec3  u_color          = vec3(1.0, 0.9, 0.6);
uniform float u_text_thickness = 0.035;
uniform float u_text_smoothness = 0.006;
uniform uint  u_text[MAX_TEXT_LEN];

// ---------- SDF 7-segment helpers (full table from template) ----------
float get_dist_to_line(vec2 p, vec2 a, vec2 b) {
    vec2 ab = b - a; vec2 ap = p - a;
    float t = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
    return length(p - (a + t * ab));
}

const float CW = 0.5, CH = 1.0;
const vec2 A = vec2(-CW,-CH), B = vec2(-CW,-0.5*CH), C = vec2(-CW,0.0),
           D = vec2(-CW,0.5*CH), E = vec2(-CW,CH),  F = vec2(0.0,CH),
           G = vec2(CW,CH),  H = vec2(CW,0.5*CH), J = vec2(CW,0.0),
           K = vec2(CW,-0.5*CH), L = vec2(CW,-CH), M = vec2(0.0,-CH),
           N = vec2(0.0,-0.5*CH), O = vec2(0.0,0.0), P = vec2(0.0,0.5*CH);

float seg0(vec2 p){return get_dist_to_line(p,A,B);} float seg1(vec2 p){return get_dist_to_line(p,B,C);}
float seg2(vec2 p){return get_dist_to_line(p,C,D);} float seg3(vec2 p){return get_dist_to_line(p,D,E);}
float seg4(vec2 p){return get_dist_to_line(p,E,F);} float seg5(vec2 p){return get_dist_to_line(p,F,G);}
float seg6(vec2 p){return get_dist_to_line(p,G,H);} float seg7(vec2 p){return get_dist_to_line(p,H,J);}
float seg8(vec2 p){return get_dist_to_line(p,J,K);} float seg9(vec2 p){return get_dist_to_line(p,K,L);}
float seg10(vec2 p){return get_dist_to_line(p,L,M);} float seg11(vec2 p){return get_dist_to_line(p,M,A);}
float seg12(vec2 p){return get_dist_to_line(p,M,N);} float seg13(vec2 p){return get_dist_to_line(p,N,O);}
float seg14(vec2 p){return get_dist_to_line(p,O,P);} float seg15(vec2 p){return get_dist_to_line(p,P,G);}
float seg16(vec2 p){return get_dist_to_line(p,A,L);} float seg17(vec2 p){return get_dist_to_line(p,E,G);}
float seg18(vec2 p){return get_dist_to_line(p,C,K);} float seg19(vec2 p){return get_dist_to_line(p,B,J);}
float seg20(vec2 p){return get_dist_to_line(p,D,H);}

float get_dist_to_digit(vec2 p, int d) {
    if (d==0) return seg0(p)+seg1(p)+seg2(p)+seg3(p)+seg4(p)+seg5(p)+seg6(p)+seg7(p)+seg8(p)+seg9(p)+seg10(p)+seg11(p);
    if (d==1) return min(min(seg2(p),seg3(p)),min(seg8(p),seg9(p)));
    if (d==2) return seg0(p)+seg1(p)+seg2(p)+seg3(p)+seg7(p)+seg8(p)+seg9(p)+seg10(p)+seg11(p)+seg12(p)+seg13(p)+seg15(p);
    if (d==3) return seg0(p)+seg1(p)+seg2(p)+seg3(p)+seg7(p)+seg8(p)+seg9(p)+seg10(p)+seg11(p)+seg15(p)+seg16(p);
    if (d==4) return seg4(p)+seg5(p)+seg6(p)+seg7(p)+seg8(p)+seg9(p)+seg15(p)+seg19(p);
    if (d==5) return seg0(p)+seg4(p)+seg5(p)+seg6(p)+seg7(p)+seg8(p)+seg9(p)+seg10(p)+seg11(p)+seg15(p)+seg20(p);
    if (d==6) return seg0(p)+seg4(p)+seg5(p)+seg6(p)+seg7(p)+seg8(p)+seg9(p)+seg10(p)+seg11(p)+seg12(p)+seg13(p)+seg14(p)+seg15(p);
    if (d==7) return seg0(p)+seg1(p)+seg2(p)+seg3(p)+seg5(p)+seg6(p)+seg7(p)+seg8(p)+seg9(p);
    if (d==8) return seg0(p)+seg1(p)+seg2(p)+seg3(p)+seg4(p)+seg5(p)+seg6(p)+seg7(p)+seg8(p)+seg9(p)+seg10(p)+seg11(p)+seg15(p)+seg16(p);
    if (d==9) return seg0(p)+seg1(p)+seg2(p)+seg3(p)+seg4(p)+seg5(p)+seg6(p)+seg7(p)+seg8(p)+seg9(p)+seg15(p)+seg19(p);
    return MAX_DIST;
}

float get_dist_to_latin_char(vec2 p, uint ch) {
    // digits first
    int d = int(ch) - int('0');
    if (d>=0 && d<=9) return get_dist_to_digit(p, d);

    // basic Latin letters (S H A D E R B O X) approximated with segments
    if (ch == uint('S')) return seg0(p)+seg1(p)+seg4(p)+seg5(p)+seg6(p)+seg7(p)+seg8(p)+seg9(p)+seg10(p)+seg11(p)+seg15(p)+seg20(p);
    if (ch == uint('H')) return seg4(p)+seg5(p)+seg6(p)+seg7(p)+seg8(p)+seg9(p)+seg12(p)+seg13(p)+seg14(p)+seg15(p);
    if (ch == uint('A')) return seg0(p)+seg1(p)+seg2(p)+seg3(p)+seg4(p)+seg5(p)+seg6(p)+seg7(p)+seg8(p)+seg9(p)+seg12(p)+seg13(p)+seg14(p)+seg15(p);
    if (ch == uint('D')) return seg0(p)+seg1(p)+seg2(p)+seg3(p)+seg8(p)+seg9(p)+seg10(p)+seg11(p)+seg15(p)+seg16(p);
    if (ch == uint('E')) return seg0(p)+seg1(p)+seg4(p)+seg5(p)+seg6(p)+seg7(p)+seg8(p)+seg9(p)+seg10(p)+seg11(p)+seg12(p)+seg13(p)+seg15(p)+seg20(p);
    if (ch == uint('R')) return seg0(p)+seg1(p)+seg2(p)+seg3(p)+seg4(p)+seg5(p)+seg6(p)+seg7(p)+seg8(p)+seg9(p)+seg12(p)+seg13(p)+seg14(p)+seg15(p)+seg19(p);
    if (ch == uint('B')) return seg0(p)+seg1(p)+seg2(p)+seg3(p)+seg4(p)+seg5(p)+seg6(p)+seg7(p)+seg8(p)+seg9(p)+seg10(p)+seg11(p)+seg12(p)+seg13(p)+seg14(p)+seg15(p)+seg16(p);
    if (ch == uint('O')) return seg0(p)+seg1(p)+seg2(p)+seg3(p)+seg4(p)+seg5(p)+seg6(p)+seg7(p)+seg8(p)+seg9(p)+seg10(p)+seg11(p)+seg15(p)+seg16(p);
    if (ch == uint('X')) return seg4(p)+seg5(p)+seg6(p)+seg7(p)+seg8(p)+seg9(p)+seg19(p)+seg20(p);

    if (ch == uint(' ')) return MAX_DIST;
    return MAX_DIST;
}

void main() {
    // 1) flat/default 2D text (full screen)  -----------------------------------
    float dist = MAX_DIST;
    float pos  = 0.0;
    for (uint i = 0; i < MAX_TEXT_LEN; ++i) {
        uint ch = u_text[i];
        if (ch == 0u) break;
        vec2 p_text = vec2((pos * 0.055 - 0.5) * u_char_scale,
                           (vs_uv.y - 0.5) * u_char_scale * -1.0);
        dist = min(dist, get_dist_to_latin_char(p_text, ch));
        pos += 1.0;
    }
    float tline = 1.0 - smoothstep(0.0, u_text_smoothness,
                                   dist - u_text_thickness);

    // 2) green sphere + spiral under the text  ---------------------------------
    vec3  col = vec3(0.0);
    vec2  uv  = vs_uv * 2.0 - 1.0;
    uv.x *= u_aspect;
    float r2 = dot(uv, uv);
    if (r2 <= 1.0) {
        float z   = sqrt(1.0 - r2);
        vec3  p   = vec3(uv, z);

        float rot = u_time * 0.7;
        float c = cos(rot), s = sin(rot);
        p = vec3(c*p.x + s*p.z, p.y, -s*p.x + c*p.z);

        float phi   = atan(p.y, p.x);
        float theta = acos(clamp(p.z, -1.0, 1.0));

        float t   = u_time * 0.8;
        float r   = theta;
        float a   = phi;

        float sp = sin(6.0*a + 10.0*r - 5.0*t)*0.45 +
                   sin(9.0*a -14.0*r + 3.5*t)*0.25 +
                   sin(22.0*a+31.0*r -18.0*t)*0.3 +
                   sin(47.0*a-52.0*r +29.0*t)*0.18 +
                   SB_fbm(vec2(phi, theta)*13.0 + t*1.3, 5) * 0.09;

                  vec3 sph = mix(vec3(0.05,0.02,0.2), vec3(0.1,0.95,0.15),
                         smoothstep(-0.005,0.005,sp));

        vec3 n = p;
        float ambient = 0.25 + 0.35 * (1.0 - theta / 3.14159);
        float diff = max(dot(n, normalize(vec3(0.6,0.8,1.0))), 0.0);
        sph *= ambient + 0.6*diff;

        col = sph;
    }

    // 3) text on top  ----------------------------------------------------------
    col = mix(col, u_color, tline);
    fs_color = vec4(col, 1.0);
}
