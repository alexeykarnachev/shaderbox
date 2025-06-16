#version 460 core

in vec2 vs_uv;

uniform float u_time;
uniform float u_aspect;

uniform vec3 u_rgb_weights = vec3(1.0, 1.0, 1.0);

uniform sampler2D u_video;

out vec4 fs_color;

void main() {
    vec3 video_color = texture(u_video, vs_uv).rgb;

    float c = dot(video_color, u_rgb_weights);
    c = pow(c * 4.0, 2.0);

    vec3 color = vec3(c);

    fs_color = vec4(color, 1.0);
}
