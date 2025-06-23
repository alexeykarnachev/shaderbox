#version 460 core

in vec2 vs_uv;

uniform float u_time;
uniform float u_aspect;

uniform vec2 u_resolution;

uniform sampler2D u_image;
uniform sampler2D u_depth;

uniform vec3 u_light_dir = vec3(1.0, -1.0, 0.0);
uniform vec3 u_light_color = vec3(1.0, 0.2, 0.1);
uniform float u_light_brightness = 2.0;

uniform vec3 u_background_color = vec3(0.0, 0.0, 1.0);

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
    return 60.1 * (dot(m0 * m0, vec3(dot(p0, x0), dot(p1, x1), dot(p2, x2))) +
                   dot(m1 * m1, vec2(dot(p3, x3), dot(p4, x4))));
}

vec3 compute_normal(vec2 uv, float depth_scale, vec2 texel_size) {
    float k = 12.0;
    float depth_left = texture(u_depth, uv - k * vec2(texel_size.x, 0.0)).r;
    float depth_right = texture(u_depth, uv + k * vec2(texel_size.x, 0.0)).r;
    float depth_bottom = texture(u_depth, uv - k * vec2(0.0, texel_size.y)).r;
    float depth_top = texture(u_depth, uv + k * vec2(0.0, texel_size.y)).r;

    float dx = (depth_left - depth_right) * 0.5 / texel_size.x * depth_scale;
    float dy = (depth_bottom - depth_top) * 0.5 / texel_size.y * depth_scale;

    vec3 normal = normalize(vec3(-dx, -dy, 1.0));
    return normal;
}

void main() {
    float t = abs(mod(u_time, 3.0) - 1.5);

    vec2 uv = vs_uv;
    vec3 image_color = texture(u_image, uv).rgb;
    float depth = texture(u_depth, uv).r;

    vec2 texel_size = 1.0 / u_resolution;
    float depth_scale = 1.0;

    vec3 normal = compute_normal(uv, depth_scale, texel_size);

    float d = (1.0 - depth);
    vec3 light_dir = u_light_dir;
    light_dir.y += 0.2 * t - 0.2;
    light_dir.x += 0.2 * t - 0.2;
    light_dir.z += 0.1 * t - 0.05;
    light_dir = normalize(light_dir);
    float k = max(0.0, dot(-normal, light_dir));
    k = pow(k, 4.0) * pow(1.0 + light_dir.z, 2.0);

    vec3 foreground_color =
        d * k * image_color * u_light_color * u_light_brightness;
    foreground_color += 0.1;
    foreground_color = pow(foreground_color, vec3(1.2));

    vec2 uv_offset = vec2(snoise(vec4(16.0 * uv.x, 32.0 * uv.y - t, t, 0.0)),
                          snoise(vec4(16.0 * uv.x, 32.0 * uv.y - t, 10.0, t)));
    uv_offset *= 0.1;
    vec3 background_color = texture(u_image, uv + uv_offset).rgb;
    background_color = smoothstep(0.8, 1.0, depth) * background_color;
    background_color = background_color * u_background_color;

    vec3 color = foreground_color + background_color;

    uv = abs(vs_uv * 2.0 - 1.0);
    k = smoothstep(0.025, 0.03, 1.0 - max(uv.x, uv.y));
    color = k * color + (1.0 - k) * vec3(0.5, 0.5, 0.25);

    fs_color = vec4(color, 1.0);
}
