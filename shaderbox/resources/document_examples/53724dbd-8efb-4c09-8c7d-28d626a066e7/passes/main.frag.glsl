#version 460 core

in vec2 vs_uv;

uniform float u_time;
uniform float u_aspect;

out vec4 fs_color;

void main() {
    float blue = abs(fract(0.5 * u_time) - 0.5);
    vec3 color = vec3(vs_uv, blue);
    fs_color = vec4(color, 1.0);
}
