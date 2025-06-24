#version 460 core

in vec2 vs_uv;
out vec4 fs_color;

uniform float u_aspect;
uniform vec2 u_resolution;

void main() {
    vec2 uv = vs_uv;
    uv.x /= u_aspect;

    vec2 pixel_size = 1.0 / u_resolution;
    vec2 glyph_size = pixel_size * vec2(32.0, 32.0);

    uvec2 cell_idx = uvec2(uv / glyph_size);
    vec2 cell_uv = fract(uv / glyph_size);

    vec3 color = vec3(0.15);
    if (cell_idx.x == 2) {
        color = vec3(cell_uv, 0.0);
    }

    fs_color = vec4(color, 1.0);
}
