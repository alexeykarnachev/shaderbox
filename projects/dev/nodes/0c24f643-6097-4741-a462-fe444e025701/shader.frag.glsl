#version 460 core

in vec2 vs_uv; // Coordinate of the current pixel to be shaded

uniform float u_time;   // Time (s) since the application started
uniform float u_aspect; // Aspect ratio of the canvas (width / height)
// uniform vec2 u_resolution;  // Resolution of the canvas (width, height)

out vec4 fs_color;

void main() {
    float blue = abs(fract(0.5 * u_time) - 0.5);
    vec3 color = vec3(vs_uv, blue);
    fs_color = vec4(color, 1.0);
}
