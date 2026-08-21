#version 460 core

in vec2 vs_uv; // Coordinate of the current pixel to be shaded

uniform float u_time;   // Time (s) since the application started
uniform float u_aspect; // Aspect ratio of the canvas (width / height)
// uniform vec2 u_resolution;  // Resolution of the canvas (width, height)

out vec4 fs_color;

void main() {
    
    float freq = 0.01 * ((vs_uv.x * 2.0) - 1.0) + 0.001 * u_time;
    vec3 color = vec3(0.5 * (sin(freq * u_time) + 1.0));
    
    fs_color = vec4(color, 1.0);
}
