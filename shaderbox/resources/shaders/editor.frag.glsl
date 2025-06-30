#version 460 core

in vec2 vs_uv;
out vec4 fs_color;
uniform float u_aspect;

uniform vec2 u_resolution;        // Screen resolution in pixels
uniform vec2 u_grid_size;         // Grid dimensions (number of cells)
uniform vec2 u_glyph_size_px;     // Cell size in pixels
uniform sampler2D u_grid_uvs;     // Texture with glyph UV rectangles
uniform sampler2D u_grid_metrics; // Texture with glyph metrics
uniform sampler2D u_glyph_atlas;  // Glyph atlas texture
uniform vec2 u_cursor_grid_pos;   // Cursor position (line_idx, char_idx)

vec2 flip_y(vec2 v) { return vec2(v.x, 1.0 - v.y); }

void main() {
    vec2 uv = flip_y(vs_uv); // Flip input UVs to match OpenGL convention

    // Calculate cell size in UV space (normalized to [0, 1])
    vec2 cell_size_uv = u_glyph_size_px / u_resolution;

    // Determine which grid cell we're in
    vec2 cell_idx = floor(uv / cell_size_uv);

    // Discard fragments outside the grid bounds
    if (cell_idx.x < 0.0 || cell_idx.x >= u_grid_size.x || cell_idx.y < 0.0 ||
        cell_idx.y >= u_grid_size.y) {
        discard;
    }

    // Sample glyph data from textures at the cell's center
    vec2 sample_uv = (cell_idx + 0.5) / u_grid_size;
    vec4 glyph_uv_rect = texture(u_grid_uvs, sample_uv);
    vec4 grid_metrics = texture(u_grid_metrics, sample_uv);

    // Unpack metrics
    float glyph_width = grid_metrics.x;  // Normalized glyph bitmap width
    float glyph_height = grid_metrics.y; // Normalized glyph bitmap height
    float bearing_x = grid_metrics.z;    // Normalized horizontal offset
    float bearing_y =
        grid_metrics.w; // Normalized vertical offset (baseline-adjusted)

    // Compute local UV within the current cell (0 to 1)
    vec2 local_uv = flip_y(fract(uv / cell_size_uv));

    // Map local UV to glyph's UV coordinates in the atlas
    vec2 glyph_uv = vec2((local_uv.x - bearing_x) /
                             (glyph_width + 0.0001), // Avoid division by zero
                         (local_uv.y - bearing_y) / (glyph_height + 0.0001));

    // Check if the fragment is within the glyph's bounds
    bool is_glyph_present =
        (glyph_width > 0.0 && glyph_height > 0.0 && glyph_uv.x >= 0.0 &&
         glyph_uv.x <= 1.0 && glyph_uv.y >= 0.0 && glyph_uv.y <= 1.0);

    // Define cursor width (2 pixels in UV space)
    float cursor_width_uv = 2.0 / u_glyph_size_px.x;

    // Render cursor if in the cursor cell
    if (cell_idx.x == u_cursor_grid_pos.y &&
        cell_idx.y == u_cursor_grid_pos.x && local_uv.x < cursor_width_uv) {
        fs_color = vec4(1.0); // White cursor
    }
    // Render glyph if present
    else if (is_glyph_present) {
        vec2 atlas_uv = mix(glyph_uv_rect.xy, glyph_uv_rect.zw, glyph_uv);
        float alpha = texture(u_glyph_atlas, atlas_uv).r;
        fs_color = vec4(vec3(1.0), alpha); // White glyph with alpha
    }
    // Discard fragments outside glyph or cursor
    else {
        discard;
    }
}
