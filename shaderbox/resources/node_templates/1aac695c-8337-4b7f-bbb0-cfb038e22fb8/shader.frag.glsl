#version 460 core

in vec2 vs_uv;

uniform float u_time;
uniform float u_aspect;

uniform sampler2D u_image;
uniform sampler2D u_depth;
uniform sampler2D u_background;

uniform float u_zoomout = 1.0;
uniform vec2 u_offset = vec2(0.0, 0.0);

out vec4 fs_color;

void main() {
    vec2 uv = (vs_uv - 0.5 - u_offset) * u_zoomout + 0.5;

    vec3 albedo = texture(u_image, uv).rgb;
    float depth = texture(u_depth, uv).r;
    float background = texture(u_background, uv).r;

    vec3 color = albedo * clamp(depth, 0.25, 1.0) * clamp(background, 0.25, 1.0);

    fs_color = vec4(color, 1.0);
}
