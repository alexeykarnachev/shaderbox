#version 460 core

in vec2 vs_uv;

uniform float u_time;
uniform float u_aspect;
uniform vec2 u_resolution;

uniform sampler2D u_image;
uniform sampler2D u_depth;
uniform sampler2D u_background;

uniform float u_zoomout = 1.0;
uniform vec2 u_offset = vec2(0.0, 0.0);

uniform float u_depth_n_samples = 32.0;
uniform float u_depth_sample_radius = 8.0;

uniform float u_background_n_samples = 32.0;
uniform float u_background_sample_radius = 8.0;

uniform float u_outline_center = 0.5;
uniform float u_outline_thickness = 0.1;
uniform float u_outline_softness = 0.1;

out vec4 fs_color;

const vec2 poisson_disk87[87] =
    vec2[87](vec2(-0.488690, 0.046349), vec2(0.496064, 0.018367),
             vec2(-0.027347, -0.461505), vec2(-0.090074, 0.490283),
             vec2(0.294474, 0.366950), vec2(0.305608, -0.360041),
             vec2(-0.346198, -0.357278), vec2(-0.308924, 0.353038),
             vec2(-0.437547, -0.177748), vec2(0.446996, -0.129850),
             vec2(0.117621, -0.444649), vec2(0.171424, 0.418258),
             vec2(-0.227789, -0.410446), vec2(0.210264, -0.422608),
             vec2(-0.414136, -0.268376), vec2(0.368202, 0.316549),
             vec2(-0.480689, 0.127069), vec2(0.481128, -0.056358),
             vec2(-0.458004, -0.063002), vec2(0.409361, 0.201972),
             vec2(-0.176597, 0.424044), vec2(-0.095380, -0.441734),
             vec2(0.326086, -0.280594), vec2(-0.411327, 0.184757),
             vec2(-0.291534, -0.300406), vec2(0.400901, -0.002308),
             vec2(0.020255, 0.445511), vec2(0.302251, 0.275637),
             vec2(0.387805, -0.223370), vec2(-0.378395, 0.062614),
             vec2(0.405052, 0.101681), vec2(-0.010340, -0.355322),
             vec2(-0.034931, 0.383699), vec2(-0.318953, -0.225899),
             vec2(0.349283, -0.140001), vec2(-0.253974, 0.299183),
             vec2(0.188226, 0.342914), vec2(0.212083, -0.294545),
             vec2(-0.188320, -0.308466), vec2(-0.373708, -0.070538),
             vec2(0.114322, -0.356677), vec2(-0.154401, 0.348207),
             vec2(-0.321713, 0.260043), vec2(-0.086797, -0.349277),
             vec2(-0.360294, -0.144808), vec2(-0.323996, 0.188199),
             vec2(0.277830, -0.204128), vec2(0.087828, 0.351992),
             vec2(-0.215777, -0.234955), vec2(0.291437, 0.171860),
             vec2(0.027249, -0.255925), vec2(-0.316361, -0.013941),
             vec2(0.346679, -0.066942), vec2(-0.103280, -0.273636),
             vec2(-0.017802, 0.310973), vec2(-0.280809, -0.120043),
             vec2(-0.282912, 0.117500), vec2(0.267574, -0.036973),
             vec2(-0.034965, -0.223502), vec2(0.109677, 0.256372),
             vec2(-0.204519, -0.116846), vec2(0.144105, -0.181736),
             vec2(-0.140560, 0.215101), vec2(0.271573, 0.102406),
             vec2(0.220437, 0.203459), vec2(-0.242979, -0.027494),
             vec2(-0.050135, 0.239871), vec2(-0.152652, -0.193125),
             vec2(-0.220532, 0.179600), vec2(0.216867, -0.096770),
             vec2(-0.164884, 0.122109), vec2(0.251078, 0.034090),
             vec2(0.016515, -0.175206), vec2(0.042304, 0.216117),
             vec2(-0.133933, -0.060601), vec2(0.184659, 0.135680),
             vec2(-0.161273, 0.024207), vec2(-0.056532, -0.154410),
             vec2(-0.082706, 0.083129), vec2(0.081409, -0.088060),
             vec2(0.115078, 0.156566), vec2(0.133209, 0.061211),
             vec2(0.002618, -0.101328), vec2(0.132926, -0.013988),
             vec2(-0.027172, -0.017586), vec2(0.022969, 0.116469),
             vec2(0.036262, 0.015085));

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
    float n =
        60.1 * (dot(m0 * m0, vec3(dot(p0, x0), dot(p1, x1), dot(p2, x2))) +
                dot(m1 * m1, vec2(dot(p3, x3), dot(p4, x4))));
    return 0.5 * (n + 1.0);
}

struct ParallaxResult {
    vec2 uv;

    float step_idx;
    float depth;
    float diff;

    float is_success;
};

ParallaxResult do_parallax(vec2 input_uv, vec3 cam_pos, float max_n_steps) {
    vec3 p = vec3((input_uv - 0.5) * vec2(u_aspect, 1.0), 0.0);

    vec3 ray_dir = normalize(p - cam_pos);
    float step_size = 1.0 / max_n_steps;

    float is_success = 0.0;
    float step_idx = 0.0;
    float depth;
    vec2 uv;

    float diff = 0.0;
    while (true) {
        if (step_idx >= max_n_steps) {
            break;
        }

        uv = p.xy / vec2(u_aspect, 1.0) + 0.5;
        if (uv.x >= 1.0 || uv.x <= 0.0 || uv.y >= 1.0 || uv.y <= 0.0) {
            break;
        }

        depth = texture2D(u_depth, uv).r;
        depth = clamp(depth, 0.001, 0.999);

        diff = p.z - depth;
        if (diff >= 0.0) {
            is_success = 1.0;
            break;
        }

        p += step_size * ray_dir;
        step_idx += 1.0;
    }

    return ParallaxResult(uv, step_idx, depth, diff, is_success);
}

float get_outline(vec2 uv) {
    vec2 texel_size = 1.0 / u_resolution;

    float depth = 0.0;
    {
        for (float i = 0.0; i < u_depth_n_samples; ++i) {
            vec2 offset =
                u_depth_sample_radius * texel_size * poisson_disk87[uint(i)];
            depth += texture(u_depth, uv + offset).r;
        }
        depth /= u_depth_n_samples;
        depth = 1.0 - depth;
    }

    float background = 0.0;
    {
        for (float i = 0.0; i < u_background_n_samples; ++i) {
            vec2 offset = u_background_sample_radius * texel_size *
                          poisson_disk87[uint(i)];
            background += texture(u_background, uv + offset).r;
        }
        background /= u_background_n_samples;
        background *= depth;
    }

    float outline = 0.0;
    {
        float right =
            1.0 - smoothstep(u_outline_center + 0.5 * u_outline_thickness,
                             u_outline_center + 0.5 * u_outline_thickness +
                                 0.5 * u_outline_softness,
                             background);
        float left =
            1.0 - smoothstep(u_outline_center - 0.5 * u_outline_thickness,
                             u_outline_center - 0.5 * u_outline_thickness -
                                 0.5 * u_outline_softness,
                             background);
        return right * left;
    }
}

float get_eyes(vec2 uv) {
    vec3 color = texture(u_image, uv).rgb;

    float eyes = dot(vec3(0.8, 0.8, 0.0), color);

    return eyes;
}

void main() {
    vec2 uv = (vs_uv - 0.5 - u_offset) * u_zoomout + 0.5;

    float outline = get_outline(uv);
    float eyes = get_eyes(uv);

    vec3 color = vec3(outline);
    fs_color = vec4(color, 1.0);
}
