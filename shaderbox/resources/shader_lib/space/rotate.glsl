/// Rotate p around the origin by angle radians. NOTE: rotating the COORDINATE
/// spins the rendered SHAPE clockwise for +angle — negate the angle for a
/// counter-clockwise shape: SB_sd_text(SB_rotate(p, -a), ...) spins text CCW.
vec2 SB_rotate(vec2 p, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return vec2(c * p.x - s * p.y, s * p.x + c * p.y);
}
