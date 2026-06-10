// Renderers: signed distance in -> 0..1 mask out. The last step of the chain
// (SDF source -> SB_op_* transforms -> render).

/// Fill mask of an SDF's interior; the soft edge extends OUTWARD over `smoothness`
/// (uv units, try 0.005..0.02 — or use SB_fill_aa for automatic pixel AA).
float SB_fill(float d, float smoothness) {
    return 1.0 - smoothstep(0.0, max(smoothness, 0.0001), d);
}

/// Fill mask with pixel-perfect anti-aliasing (fwidth-based) — no knob to tune.
float SB_fill_aa(float d) {
    // Upper clamp kills the full-frame quad-seam line fwidth emits when d comes
    // from a runtime-mutated array (the typewriter/shuffle pattern).
    float w = clamp(fwidth(d), 0.0001, 0.05);
    return 1.0 - smoothstep(-w, w, d);
}

/// Neon-style glow: 1 inside the shape, exponential falloff outside. The value is
/// 1/e at `radius` — the visible halo reaches ~3-5x radius (try 0.05..0.15).
float SB_glow(float d, float radius) {
    return exp(-max(d, 0.0) / max(radius, 0.0001));
}
