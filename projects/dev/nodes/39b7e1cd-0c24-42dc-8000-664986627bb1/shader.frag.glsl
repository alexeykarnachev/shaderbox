#version 300 es
precision highp float;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_aspect;
out vec4 fs_color;

void main() {
    vec2 uv = (gl_FragCoord.xy / u_resolution) * 2.0 - 1.0;
    uv.x *= u_aspect;
    float d = length(uv);
    float circle = smoothstep(0.6, 0.58, d);
    fs_color = vec4(circle * vec3(1.0, 0.0, 0.0), 1.0);
}