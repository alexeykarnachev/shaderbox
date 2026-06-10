/// Map vs_uv [0,1] to centered aspect-corrected coords: y in [-1,1], x in [-u_aspect,u_aspect].
vec2 SB_center_uv(vec2 uv, float aspect) {
    return (uv - 0.5) * 2.0 * vec2(aspect, 1.0);
}
