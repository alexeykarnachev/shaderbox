#version 460 core
#define TEXT_LEN 16
#define ARR_LEN 4

// minimal self-contained hash + value noise (polar noise version no seam)
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 234.56));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y);
}
float value_noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    float a = hash21(i),              b = hash21(i + vec2(1,0));
    float c = hash21(i + vec2(0,1)),  d = hash21(i + vec2(1,1));
    vec2 u = f*f*(3.0-2.0*f);
    return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}

in vec2 vs_uv;
out vec4 fs_color;

uniform float u_time;
uniform float u_aspect;

uniform float u_drag_float;
uniform vec2  u_drag_vec2;
uniform vec3  u_drag_vec3;
uniform vec4  u_drag_vec4;
uniform float u_enable;
uniform float u_brightness;

uniform vec3 u_color;
uniform vec4 u_tint_color;

uniform sampler2D u_texture;

uniform float u_floats[ARR_LEN];
uniform uint  u_uints[ARR_LEN];
uniform uint  u_label_text[TEXT_LEN];

layout(std140) uniform u_params {
    vec4 a;
    vec4 b;
} params;

float sdSphere(vec3 p, float r) { return length(p) - r; }

float map(vec3 p) {
    float t = u_time * 0.8 + u_drag_float * 2.0;
    vec3 center = u_drag_vec3 * 1.5;
    // simple automatic orbit animation around Y axis
    center.x += sin(u_time * 1.2) * 0.6;
    center.z += cos(u_time * 1.2) * 0.6;
    float r = 0.8 + u_drag_vec4.x * 0.6 + params.a.x * 0.2;
    r += sin(u_floats[0] * 6.28318) * 0.05;
    return sdSphere(p - center, r);
}

vec3 calcNormal(vec3 p) {
    const float h = 0.001;
    return normalize(vec3(
        map(p + vec3(h,0,0)) - map(p - vec3(h,0,0)),
        map(p + vec3(0,h,0)) - map(p - vec3(0,h,0)),
        map(p + vec3(0,0,h)) - map(p - vec3(0,0,h))
    ));
}

void main() {
// center-based, aspect-corrected coordinates (rings remain circular)
        vec2 p = vs_uv * 2.0 - 1.0;
        p.x *= u_aspect;
      float radius = length(p);
      float angle  = atan(p.y, p.x);

// polar wiggly noise on each ring (seamless via circular direction: constant radius around unit circle)
        vec2 dir = vec2(cos(angle), sin(angle));
        float rmod = radius * 4.5 + sin(u_time * 0.6) * 0.4;
        float polar_noise = value_noise(dir * rmod) - 0.5;
        radius += 0.22 * polar_noise * (0.9 + 0.5 * sin(u_time * 0.9));

      radius += 0.02 * sin(u_time * 2.0 + u_drag_vec2.y * 6.28);

    // concentric rings
    float ring = abs(fract(radius * 6.0 + sin(u_time) * 0.5) - 0.5) * 2.0;
    float mask = smoothstep(0.08, 0.0, ring);

    vec3 ring_color = mix(u_color, u_tint_color.rgb, 0.5 + 0.5 * sin(angle * 4.0 + u_time));
    vec3 col = ring_color * mask * 1.8;

    col *= max(u_enable, 1.0);
    col = clamp(col, 0.0, 1.0);

    fs_color = vec4(col * u_brightness, 1.0);
}

