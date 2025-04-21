#version 460

in vec2 vs_uv;
out vec4 fs_color;

uniform float u_time;
uniform float u_aspect;

#define PI 3.141592

void main() {
    vec2 sp = (vs_uv * 2.0) - 1.0;
    sp.x *= u_aspect;
    
    vec3 color = vec3(sp, 0.0);
    fs_color = vec4(color, 1.0);
}
