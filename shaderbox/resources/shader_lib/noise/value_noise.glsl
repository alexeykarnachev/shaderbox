/// Cheap 2D hash in [0,1].
float SB_hash21(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

/// Bilinear value noise in [0,1]; feature size = 1 unit, so SCALE the input
/// (SB_value_noise(p * 8.0) — over centered uv try x4..x16, else it looks flat).
float SB_value_noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = SB_hash21(i);
    float b = SB_hash21(i + vec2(1.0, 0.0));
    float c = SB_hash21(i + vec2(0.0, 1.0));
    float d = SB_hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
