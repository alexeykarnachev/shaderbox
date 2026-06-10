// SDF operators: signed distance in -> signed distance out. Compose freely,
// then hand the result to a renderer (SB_fill / SB_glow).

/// Union of two SDFs.
float SB_op_union(float a, float b) { return min(a, b); }

/// Intersection of two SDFs.
float SB_op_intersect(float a, float b) { return max(a, b); }

/// Subtract shape b from shape a.
float SB_op_subtract(float a, float b) { return max(a, -b); }

/// Smooth union: like SB_op_union but the shapes BLEND where they meet;
/// k = blend radius in the same units as the distances (try 0.05..0.3).
float SB_op_smooth_union(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / max(k, 0.0001), 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

/// Round/dilate: grow the shape outward by r (r < 0 shrinks).
float SB_op_round(float d, float r) { return d - r; }

/// Onion: the concentric shell of half-thickness t, as an SDF — the way to draw
/// an outline / contour / ring / annulus: SB_fill_aa(SB_op_onion(d, t)).
float SB_op_onion(float d, float t) { return abs(d) - t; }
