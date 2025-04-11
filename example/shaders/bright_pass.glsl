#version 460

in vec2 vs_uv;
out vec4 fs_color;
uniform sampler2D u_source_texture;
uniform float u_threshold = 0.5;

void main() {
	vec4 color = texture(u_source_texture, vs_uv);
	float brightness = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
	fs_color = brightness > u_threshold ? color : vec4(0.0, 0.0, 0.0, 0.0);
}

