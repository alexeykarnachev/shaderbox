#version 300 es
precision highp float;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_aspect;
out vec4 fs_color;

void main() {
    vec2 uv = (gl_FragCoord.xy / u_resolution) * 2.0 - 1.0;
    uv.x *= u_aspect;
    vec2 uvabs = abs(uv);
    float d = max(uvabs.x, uvabs.y * 0.866 + uvabs.x * 0.5);
    float hex = smoothstep(0.7, 0.68, d);
    vec3 col = 0.5 + 0.5 * cos(u_time + uv.xyx + vec3(0,2,4));
    fs_color = vec4(hex * col, 1.0);
}