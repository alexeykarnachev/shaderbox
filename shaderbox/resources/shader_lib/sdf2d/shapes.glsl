// Canonical 2D SDF sources.

/// SIGNED distance (negative inside) to a circle of radius r centered at the origin.
float SB_sd_circle(vec2 p, float r) { return length(p) - r; }

/// SIGNED distance (negative inside) to an axis-aligned box with half-extents b,
/// centered at the origin.
float SB_sd_box(vec2 p, vec2 b) {
    vec2 q = abs(p) - b;
    return length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0);
}
