#version 460

in vec2 vs_uv;
out vec4 fs_color;

uniform float u_time;
uniform vec3 u_color;

void main() {
    vec3 color = vec3(vs_uv, 0.5 * (sin(u_time) + 1.0));
    color += 0.1 * u_color;
    fs_color = vec4(color, 1.0);
}

