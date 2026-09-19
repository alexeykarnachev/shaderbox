in vec2 v_uv;
in vec4 v_color;
flat in vec4 v_uv_bounds;

uniform sampler2D u_atlas;
uniform float u_distance_range;

out vec4 finalColor;

// The derivative form is mandatory. A uniform screenPxRange is wrong on a
// continuously zooming canvas, where local scale differs per fragment; omitting
// it renders text that looks almost right and is aliased at every edge.
float msdf_range(vec2 uv) {
    vec2 unit = vec2(u_distance_range) / vec2(textureSize(u_atlas, 0));
    return max(0.5 * dot(unit / fwidth(uv), vec2(1.0)), 1.0);
}

void main() {
    // Glyphs are packed edge to edge with no padding, so the box's outer half
    // texel bilinearly blends into whatever glyph was packed next to it -- and
    // a median of two unrelated fields crosses 0.5 where neither glyph has ink,
    // which is the neighbour bleeding a pixel into this one. Clamping to the
    // texel centres this glyph owns makes that unreachable.
    vec2 texel_size = 1.0 / vec2(textureSize(u_atlas, 0));
    vec2 lo = v_uv_bounds.xy + 0.5 * texel_size;
    vec2 hi = v_uv_bounds.zw - 0.5 * texel_size;
    // Computed unconditionally: derivatives are undefined in non-uniform
    // control flow, so branching before this works on one vendor and not
    // another.
    vec4 texel = texture(u_atlas, clamp(v_uv, min(lo, hi), max(lo, hi)));
    float a = clamp(msdf_range(v_uv) * (median3(texel.rgb) - 0.5) + 0.5, 0.0, 1.0);
    if (a <= 0.0) discard;
    finalColor = vec4(v_color.rgb, v_color.a * a);
}
