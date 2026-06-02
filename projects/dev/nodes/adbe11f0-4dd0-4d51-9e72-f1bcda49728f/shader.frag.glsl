#version 460 core

in vec2 vs_uv;
out vec4 fs_color;

uniform vec3 u_color = vec3(1.0, 0.0, 0.0); // red
uniform float u_aspect; // red

void main() {
    vec2 uv = vs_uv * 2.0 - 1.0;
    uv.x *= u_aspect;
    
    vec2 size = vec2(0.8, 0.8);
    vec2 d = abs(uv) - size;
    float dist = length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
    
    float alpha = 1.0 - smoothstep(-0.01, 0.01, dist);
    
    vec3 color = u_color * alpha;
    fs_color = vec4(color, alpha);
}