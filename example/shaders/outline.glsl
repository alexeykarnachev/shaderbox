#version 460

in vec2 vs_uv;
out vec4 fs_color;
uniform sampler2D u_outline_source;
uniform vec2 u_pixel_size;
uniform float u_outline_thickness = 1.0;

void main() {
	vec4 center = texture(u_outline_source, vs_uv);
	float edge = 0.0;
	for (int i = -1; i <= 1; i++)
	    for (int j = -1; j <= 1; j++) {
		if (i == 0 && j == 0) continue;
		vec4 neighbor = texture(u_outline_source, vs_uv + vec2(float(i), float(j)) * u_pixel_size);
		edge += length(center.rgb - neighbor.rgb);
	    }
	edge = smoothstep(0.0, 1.0, edge) * u_outline_thickness;
	fs_color = vec4(edge, edge, 0.0, 1.0);
}
