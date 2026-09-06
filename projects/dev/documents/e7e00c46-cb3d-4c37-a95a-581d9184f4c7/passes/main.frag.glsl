#version 460 core

in vec2 vs_uv;

uniform float u_time;
uniform float u_aspect;
uniform sampler2D u_video;

out vec4 fs_color;

void main() {
    vec3 color = texture(u_video, vs_uv).rgb;
    fs_color = vec4(color, 1.0);
}

