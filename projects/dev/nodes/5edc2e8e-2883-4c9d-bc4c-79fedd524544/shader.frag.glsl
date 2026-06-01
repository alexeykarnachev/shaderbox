#version 460 core
#define TEXT_LEN 16
#define ARR_LEN 4

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
    vec2 uv = vs_uv;
    uv.x *= u_aspect;

    // center-based polar coordinates
    vec2 p = uv * 2.0 - 1.0;
    float radius = length(p) + 0.02 * sin(u_time * 2.0 + u_drag_vec2.y * 6.28);
    float angle  = atan(p.y, p.x);

    // concentric rings
    float ring = abs(fract(radius * 6.0 + sin(u_time) * 0.5) - 0.5) * 2.0;
    float mask = smoothstep(0.08, 0.0, ring);

    vec3 ring_color = mix(u_color, u_tint_color.rgb, 0.5 + 0.5 * sin(angle * 4.0 + u_time));
    vec3 col = ring_color * mask * 1.8;

    col *= max(u_enable, 1.0);
    col = clamp(col, 0.0, 1.0);

    fs_color = vec4(col * u_brightness, 1.0);
}

