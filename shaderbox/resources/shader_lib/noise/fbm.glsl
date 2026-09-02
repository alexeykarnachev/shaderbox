/// Cheap 3D->1D hash in [0,1] (3D noise, per-cell spark/star seeds).
float SB_hash31(vec3 p) {
    return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453123);
}

/// Cheap 2D->2D hash, each component in [0,1] (warp offsets, jitter, per-cell 2D seeds).
vec2 SB_hash22(vec2 p) {
    return fract(sin(vec2(dot(p, vec2(127.1, 311.7)),
                          dot(p, vec2(269.5, 183.3)))) * 43758.5453123);
}

/// Triangle wave in [0,1], period 1 -- a folded sawtooth for symmetric ramps
/// (flicker, seams) without the cost of sin.
float SB_tri_wave(float x) {
    return abs(fract(x) * 2.0 - 1.0);
}

/// Fractal Brownian motion in [0,1]: `octaves` of SB_value_noise summed at
/// doubling frequency + halving amplitude -- the base cloud/terrain/marble field.
/// SCALE the input (p*4..16 over centered uv) like SB_value_noise. octaves 4..6
/// typical (capped at 8). Animate by scrolling p over time along the flow axis.
float SB_fbm(vec2 p, int octaves) {
    float sum = 0.0;
    float amp = 0.5;
    float norm = 0.0;
    for (int i = 0; i < 8; i++) {
        if (i >= octaves) break;
        sum += amp * SB_value_noise(p);
        norm += amp;
        p *= 2.0;
        amp *= 0.5;
    }
    return sum / max(norm, 1e-4);
}

/// Recursive domain-warp turbulence (iq): folds fbm into its own sample coords
/// twice so the field CURLS into flame/smoke/marble/liquid structure instead of
/// soft isotropic clouds. THE single biggest "reads as alive" lever for organic
/// and fluid effects. Returns [0,1]; scale the input like SB_fbm. Heavy (fbm x5)
/// -- reach for it deliberately.
float SB_domain_warp(vec2 p, int octaves) {
    vec2 q = vec2(SB_fbm(p, octaves),
                  SB_fbm(p + vec2(5.2, 1.3), octaves));
    vec2 r = vec2(SB_fbm(p + 4.0 * q + vec2(1.7, 9.2), octaves),
                  SB_fbm(p + 4.0 * q + vec2(8.3, 2.8), octaves));
    return SB_fbm(p + 4.0 * r, octaves);
}
