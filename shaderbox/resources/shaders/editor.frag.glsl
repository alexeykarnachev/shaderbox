#version 460 core

in vec2 vs_uv;
out vec4 fs_color;

uniform float u_aspect;
uniform vec2 u_resolution;
uniform vec2 u_grid_size;
uniform vec2 glyph_size_px;
uniform sampler2D u_glyph_uvs;
uniform sampler2D u_glyph_atlas;

void main() {
    // Input UV from vertex shader
    vec2 uv = vs_uv;
    uv.x /= u_aspect; // Adjust x-coordinate for aspect ratio

    // Size of each glyph in UV space
    vec2 glyph_size_uv = glyph_size_px / u_resolution;

    // Determine which grid cell this pixel belongs to
    vec2 cell_idx = floor(uv / glyph_size_uv);

    // Compute UV coordinate to sample u_glyph_uvs (center of texel)
    vec2 sample_uv = (cell_idx + 0.5) / u_grid_size;

    // Sample glyph UV rectangle (u0, v0, u1, v1)
    vec4 glyph_uv_rect = texture(u_glyph_uvs, sample_uv);

    // Local UV within the glyph cell (0 to 1)
    vec2 glyph_local_uv = fract(uv / glyph_size_uv);

    // Map local UV to the glyph's UV range in the atlas
    vec2 glyph_uv = mix(glyph_uv_rect.xy, glyph_uv_rect.zw, glyph_local_uv);

    // Sample the glyph atlas
    float glyph = texture(u_glyph_atlas, glyph_uv).r;

    // Output color
    vec3 color = vec3(glyph);

    // -------------------------
    color = vec3(texture(u_glyph_atlas, vs_uv).r);
    // color = vec3(texture(u_glyph_uvs, vs_uv).rgb);
    // -------------------------

    fs_color = vec4(color, 1.0);
}
