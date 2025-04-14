#version 460

in vec2 vs_uv;
out vec4 fs_color;

uniform float u_time;
uniform sampler2D u_photo_texture;
uniform sampler2D u_depth_texture;

void main() {
    vec2 uv = vs_uv;

	float depth = texture(u_depth_texture, uv).r;
    vec3 color = texture(u_photo_texture, uv).rgb;
    vec2 texture_size = vec2(textureSize(u_photo_texture, 0));

    vec2 sp = vs_uv - 0.5;
    float focal = 1.0;

    fs_color = vec4(vec3(color), 1.0);
}
