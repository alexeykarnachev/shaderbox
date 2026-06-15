#version 460 core

in vec2 vs_uv; // Coordinate of the current pixel to be shaded

uniform float u_time;   // Time (s) since the application started
uniform float u_aspect; // Aspect ratio of the canvas (width / height)
// uniform vec2 u_resolution;  // Resolution of the canvas (width, height)

out vec4 fs_color;

// Step 1: lay down a vertical heat gradient in flame space.
// Flame space: x in [-aspect, aspect], y running 0 (bottom) -> ~2 (top).
vec2 flame_space(vec2 uv, float aspect) {
    vec2 p = SB_center_uv(uv, aspect); // y in [-1, 1]
    p.y = p.y + 1.0;                   // y in [0, 2], origin at the bottom
    return p;
}

// Step 2: a turbulent upward-flowing heat field (single fbm, NO domain warp —
// this is the pre-research version, so it reads cloudy/soft rather than licking).
float flame_heat(vec2 p, float t) {
    // Horizontal column FIRST: narrow near the top, wider at the base. This is
    // what keeps the flame centered; turbulence only perturbs inside it.
    float width = mix(0.55, 0.14, clamp(p.y * 0.55, 0.0, 1.0));
    float column = exp(-pow(abs(p.x) / width, 2.0));

    // Upward-scrolling turbulence. Stretch y so tongues are tall, not blobby.
    vec2 nuv = vec2(p.x * 2.6, p.y * 1.6 - t * 1.9);
    float n = SB_fbm(nuv, 5); // [0, 1]

    // The flame rises higher where noise is high: a noisy height threshold.
    // edge in [~0.6 .. ~1.6] — the licking top of the flame.
    float top = 0.6 + n * 1.0;
    float body = 1.0 - smoothstep(top - 0.45, top, p.y);

    // Bias intensity toward the base so the core is hottest where it's rooted.
    float falloff = mix(0.5, 1.0, exp(-1.1 * p.y));

    return column * body * falloff;
}

// Step 3: blackbody-ish fire ramp. Cold edges are deep red, the core climbs
// through orange and yellow to a near-white tip.
vec3 fire_ramp(float h) {
    vec3 c = vec3(0.0);
    c = mix(c, vec3(0.55, 0.05, 0.0), smoothstep(0.0, 0.25, h)); // emberous red
    c = mix(c, vec3(1.0, 0.32, 0.02), smoothstep(0.2, 0.5, h));  // orange
    c = mix(c, vec3(1.0, 0.75, 0.15), smoothstep(0.45, 0.8, h)); // yellow
    c = mix(c, vec3(1.0, 0.97, 0.85), smoothstep(0.8, 1.0, h));  // white-hot core
    return c;
}

void main() {
    vec2 p = flame_space(vs_uv, u_aspect);

    float heat = flame_heat(p, u_time);

    vec3 color = fire_ramp(heat);
    fs_color = vec4(color, 1.0);
}
