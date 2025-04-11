#version 460

in vec2 vs_uv;
out vec4 fs_color;
uniform sampler2D u_base_texture;
uniform sampler2D u_depth_map;
uniform float u_time;
uniform float u_focal_length = 1480.0;
uniform float u_parallax_amount = 0.05;

void main() {
	vec2 uv = vs_uv;
	float depth = texture(u_depth_map, uv).r;
	vec2 camera_move = vec2(sin(u_time), cos(u_time)) * u_focal_length * u_parallax_amount;

	vec2 texture_size = vec2(textureSize(u_base_texture, 0));
	vec2 offset = -camera_move * depth / texture_size;
	uv += offset;
	uv = clamp(uv, 0.0, 1.0);
	vec4 color = texture(u_base_texture, uv);
	fs_color = vec4(color.rgb, 1.0);
}
