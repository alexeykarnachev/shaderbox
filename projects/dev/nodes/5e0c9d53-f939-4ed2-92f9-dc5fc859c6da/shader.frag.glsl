#version 300 es
precision highp float;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_aspect;
out vec4 fs_color;

void main() {
    vec2 uv = (gl_FragCoord.xy / u_resolution) * 2.0 - 1.0;
    uv.x *= u_aspect;
    vec2 ab = abs(uv);
    float square = step(max(ab.x, ab.y), 0.5);
    fs_color = vec4(square * vec3(0.0, 1.0, 0.0), 1.0);
}