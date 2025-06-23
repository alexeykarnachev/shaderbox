#version 460 core
#define PI 3.141592

in vec2 vs_uv;

uniform float u_time;
uniform float u_aspect;
uniform vec2 u_resolution;

uniform sampler2D u_image;
uniform sampler2D u_depth;
uniform sampler2D u_background;

uniform float u_zoomout = 1.0;
uniform vec2 u_offset = vec2(0.0, 0.0);

out vec4 fs_color;

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

vec3 compute_normal(vec2 uv) {
    vec2 texel_size = 1.0 / u_resolution;
    float k = 0.2;

    float depth_left = texture(u_depth, uv - k * vec2(texel_size.x, 0.0)).r;
    float depth_right = texture(u_depth, uv + k * vec2(texel_size.x, 0.0)).r;
    float depth_bottom = texture(u_depth, uv - k * vec2(0.0, texel_size.y)).r;
    float depth_top = texture(u_depth, uv + k * vec2(0.0, texel_size.y)).r;

    float dx = (depth_left - depth_right) * 0.5 / texel_size.x;
    float dy = (depth_bottom - depth_top) * 0.5 / texel_size.y;

    vec3 normal = normalize(vec3(-dx, -dy, 1.0));
    return normal;
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

void main() {
    float T = abs(mod(u_time, 3.0) / 3.0 * 2.0 - 1.0);

    vec2 uv = (vs_uv - 0.5 - u_offset) * u_zoomout + 0.5;

    vec3 cam_pos =
        vec3(0.15 * sin(mod(u_time, 3.0) * PI * 2.0),
             0.05 * cos(mod(u_time, 3.0) * PI * 2.0), mix(-6, -4, T));
    ParallaxResult pr = do_parallax(uv, cam_pos, 1024);
    uv = pr.uv;

    vec3 albedo = texture(u_image, uv).rgb;
    float depth = texture(u_depth, uv).r;
    float background = 1.0 - texture(u_background, uv).r;

    float foreground = (1.0 - background) * (1.0 - depth);

    // -------------------------------------------------------------------
    // Foreground
    vec3 foreground_color = albedo;
    {
        vec3 normal = compute_normal(uv);

        vec3 noise = normalize(vec3(snoise(vec4(64.0 * uv, 10.0 * depth, T)),
                                    snoise(vec4(64.0 * uv, T, 10.0 * depth)),
                                    snoise(vec4(10.0 * depth, T, 64.0 * uv))));

        normal = normalize(normal + 0.5 * noise);

        foreground_color =
            albedo * mix(vec3(0.6, 0.7, 1.2), vec3(0.5), 1.0 - depth);
        float d = pow(depth, 4.0);
        foreground_color = (1.0 - d) * foreground_color + d * vec3(1.0);

        vec3 light_dir = normalize(vec3(-1.0, -1.0, 1.0));
        float light = max(0.0, dot(-normal, light_dir));
        foreground_color =
            foreground_color * (1.0 + 6.0 * vec3(1.0, 1.0, 0.5) * light);

        foreground_color =
            pow(foreground_color, vec3(1.0)) + pow(foreground_color, vec3(2.0));

        foreground_color = foreground_color * foreground;
    }

    // -------------------------------------------------------------------
    // Background
    vec3 background_color = vec3(0.0, 0.0, 1.0);
    {
        vec2 uv = uv;
        uv.x = uv.x * 2.0 - 0.3;
        vec4 t = vec4(uv.x, uv.y, 0.0, 0.0);

        vec4 f0 = vec4(4.0 - T, 32.0, 0.0, 0.0);
        vec4 f1 = vec4(4.0, 16.0, 0.0, 0.0);

        float n0 = smoothstep(0.0, 0.8, snoise(f0 * t));
        float n1 = smoothstep(0.2, 0.7, snoise(f1 * t));

        float n = (n0 + n1) / 2.0;
        n = n * uv.y;

        background_color =
            n * vec3(0.3, 0.5, 1.0) + (1.0 - n) * vec3(0.8, 0.9, 1.0);

        float d = distance(uv, vec2(1.0, 1.0));
        background_color = 10.0 * background_color / (1.0 + 16.0 * d + d * d);

        background_color = background_color * background;
    }

    vec3 color = foreground_color + background_color;

    fs_color = vec4(color, 1.0);
}
