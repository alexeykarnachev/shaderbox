#version 460 core

in vec2 vs_uv;

uniform float u_time;
uniform float u_aspect;

uniform sampler2D u_image;

out vec4 fs_color;

void main() {
    vec3 image_color = texture(u_image, vs_uv).rgb;
    vec3 color = 0.5 * image_color + 0.5 * vec3(vs_uv, 0.0);
    fs_color = vec4(color, 1.0);
}
