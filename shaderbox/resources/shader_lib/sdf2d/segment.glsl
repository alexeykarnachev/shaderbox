/// Distance from p to the line segment a-b (a zero-width skeleton: never negative;
/// subtract a radius for a capsule shape: SB_sd_segment(p,a,b) - r).
float SB_sd_segment(vec2 p, vec2 a, vec2 b) {
    vec2 ab = b - a;
    vec2 ap = p - a;
    float t = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
    return length(p - (a + t * ab));
}
