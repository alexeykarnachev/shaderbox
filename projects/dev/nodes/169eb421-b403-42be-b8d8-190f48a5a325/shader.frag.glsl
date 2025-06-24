#version 460 core

in vec2 vs_uv;

uniform float u_time;
uniform float u_aspect;

out vec4 fs_color;

layout(std140) uniform ubo_0 { 
    float r; 
    float g; 
    float b; 
    float a; 
};

void main() {
    vec3 color = vec3(r, g, b);
    fs_color = vec4(color, 1.0);
}
