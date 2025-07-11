#version 460 core
#define PI 3.141592

in vec2 vs_uv;

uniform float u_time;
uniform float u_aspect;

uniform float u_k_top = 0.9;
uniform float u_k_bot = 0.9;
uniform float u_s_top = 2.0;
uniform float u_s_bot = 2.0;

out vec4 fs_color;

void main() {
    vec2 sp = 0.5 * (vs_uv * 2.0 - 1.0);
    sp.x *= u_aspect;

    float head = 0.0;
    {
        vec2 center = vec2(0.0, 0.0);
        float d = distance(center, sp);

        float s = 0.5 * (sign(sp.y) + 1.0);

        float k = (1.0 - s) * u_k_bot + s * u_k_top;
        float p = (1.0 - s) * u_s_bot + s * u_s_top;

        float size = 0.2 + k * pow(abs(sp.y), p);
        float smoothness = clamp(0.01 - 0.075 * pow(abs(sp.y), 2.0), 0.0, 1.0);
        head = 1.0 - smoothstep(size, size + smoothness, d);
    }

    float hair = 0.0;
    {
        
    }

    vec3 color = vec3(head);
    fs_color = vec4(color, 1.0);
}
