#version 460 core

in vec2 vs_uv;

uniform float u_time;
uniform float u_aspect;

uniform sampler2D u_video;
uniform sampler2D u_bg;

out vec4 fs_color;

void main() {
    vec3 color = texture(u_video, vs_uv).rgb;

    vec3 w = vec3(1.0, 1.0, 1.0);
    float b =
        (w.r * color.r + w.g * color.g + w.b * color.b) / (w.r + w.g + w.b);
    b = 1.0 - b;
    b = pow(1.3 * b, 8.0);

    color = vec3(b);

    vec2 sp = vs_uv * 2.0 - 1.0;
    float circle = 1.0 - smoothstep(0.9, 0.92, length(sp));
    color = color * circle;
    fs_color = vec4(color, 1.0);
}
