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

    vec3 ro = vec3(0.0, 0.0, -3.5 + u_drag_vec2.y * 2.0);
    vec3 rd = normalize(vec3(uv, 1.8));

    float t = 0.0;
    for (int i = 0; i < 64; i++) {
        vec3 p = ro + rd * t;
        float d = map(p);
        if (d < 0.001 || t > 20.0) break;
        t += d;
    }

    vec3 col = vec3(0.02);
    if (t < 20.0) {
        vec3 p = ro + rd * t;
        vec3 n = calcNormal(p);
        vec3 light = normalize(vec3(0.6, 1.2, -0.8) + u_drag_vec3);
        float diff = max(dot(n, light), 0.0);
        float spec = pow(max(dot(reflect(-light, n), -rd), 0.0), 32.0);

        vec3 tex = texture(u_texture, uv * 0.5 + 0.5).rgb;
        uint seed = u_uints[0] + u_label_text[0];
        float extra = float(seed % 7u) * 0.01;

                  col = u_color * diff + u_tint_color.rgb * spec + tex * 0.15;
          col += params.b.xyz * 0.1 + extra;
          col *= max(u_enable, 1.0);  // default visible even when drag-float is zero
      }

          col = clamp(col, 0.0, 1.0);   // tone down over-bright red blob
      fs_color = vec4(col, 1.0);
}

