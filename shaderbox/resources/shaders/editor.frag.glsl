#version 460 core
in vec2 vs_uv;
out vec4 fs_color;
uniform float u_aspect;
uniform vec2 u_resolution;
uniform vec2 u_grid_size;
uniform vec2 u_glyph_size_px;
uniform sampler2D u_glyph_uvs;
uniform sampler2D u_glyph_metrics;
uniform sampler2D u_glyph_atlas;

vec2 flip_y(vec2 v) { return vec2(v.x, 1.0 - v.y); }

void main() {
    vec2 uv = flip_y(vs_uv);

    vec2 cell_size_uv = u_glyph_size_px / u_resolution;
    vec2 cell_idx = floor(uv / cell_size_uv);

    if (cell_idx.x < 0.0 || cell_idx.x >= u_grid_size.x || cell_idx.y < 0.0 ||
        cell_idx.y >= u_grid_size.y) {
        discard;
    }

    vec2 sample_uv = (cell_idx + 0.5) / u_grid_size;
    vec4 glyph_uv_rect = texture(u_glyph_uvs, sample_uv);
    vec4 glyph_metrics = texture(u_glyph_metrics, sample_uv);

    float glyph_width = glyph_metrics.x;
    float glyph_height = glyph_metrics.y;

    float bearing_x = glyph_metrics.z;
    float adjusted_bearing_y = glyph_metrics.w;

    vec2 local_uv = flip_y(fract(uv / cell_size_uv));

    vec2 glyph_uv = vec2((local_uv.x - bearing_x) / glyph_width,
                         (local_uv.y - adjusted_bearing_y) / glyph_height);

    if (glyph_uv.x < 0.0 || glyph_uv.x > 1.0 || glyph_uv.y < 0.0 ||
        glyph_uv.y > 1.0) {
        discard;
    }

    vec2 atlas_uv = mix(glyph_uv_rect.xy, glyph_uv_rect.zw, glyph_uv);
    float alpha = texture(u_glyph_atlas, atlas_uv).r;

    fs_color = vec4(vec3(alpha), 1.0);
}
