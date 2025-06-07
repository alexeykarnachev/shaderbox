#version 460 core

in vec2 vs_uv;

uniform float u_time;
uniform float u_aspect;

out vec4 fs_color;

void main() {
    vec3 color = vec3(vs_uv, 0.0);
    fs_color = vec4(color, 1.0);
}
