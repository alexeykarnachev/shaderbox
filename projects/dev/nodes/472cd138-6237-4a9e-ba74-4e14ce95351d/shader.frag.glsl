#version 460 core

in vec2 vs_uv; // Coordinate of the current pixel to be shaded

uniform float u_time;   // Time (s) since the application started
uniform float u_aspect; // Aspect ratio of the canvas (width / height)
// uniform vec2 u_resolution;  // Resolution of the canvas (width, height)

out vec4 fs_color;

void main() {
    vec2 uv = vs_uv * 2.0 - 1.0;
    uv.x *= u_aspect;

    float r = length(uv);
    float a = atan(uv.y, uv.x);

    float spiral = sin(8.0 * a + 12.0 * r - 4.0 * u_time);

    vec3 col = mix(vec3(0.05, 0.02, 0.2), vec3(0.4, 0.8, 1.0), smoothstep(-0.2, 0.6, spiral));
    fs_color = vec4(col, 1.0);
}
