#version 460 core

in vec2 vs_uv; // Coordinate of the current pixel to be shaded

uniform sampler2D u_video;
uniform vec3 u_color;

out vec4 fs_color;

void main() {
    vec3 video_color = texture(u_video, vs_uv).rgb;
    vec3 color = video_color * u_color;
    color = color * pow(1.0 - distance(vs_uv * 2.0 - 1.0, vec2(0.0)), 0.2);

    fs_color = vec4(color, 1.0);
}
