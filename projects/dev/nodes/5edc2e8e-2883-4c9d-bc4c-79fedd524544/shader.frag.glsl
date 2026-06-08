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

void main() {
    vec2 uv = vs_uv;

    // simple test pattern driven by the various input uniforms
    vec3 base = u_color * u_brightness;
    base += u_drag_vec3 * 0.5;
    base += vec3(u_drag_vec2, 0.0) * 0.3;

    float t = u_time * 0.5 + u_drag_float * 3.0;
    float wave = 0.5 + 0.5 * sin(t + uv.x * 8.0);

    vec3 col = mix(base, u_tint_color.rgb, 0.4) * wave;

    // show texture if present
    vec4 tex = texture(u_texture, uv);
    col = mix(col, tex.rgb, tex.a * 0.5);

    col *= max(u_enable, 0.0);
    col = clamp(col, 0.0, 1.0);
    fs_color = vec4(col, 1.0);
}

