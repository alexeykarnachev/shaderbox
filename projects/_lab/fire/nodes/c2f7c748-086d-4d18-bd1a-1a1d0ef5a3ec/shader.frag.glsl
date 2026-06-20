#version 460 core

in vec2 vs_uv; // Coordinate of the current pixel to be shaded

uniform float u_time;   // Time (s) since the application started
uniform float u_aspect; // Aspect ratio of the canvas (width / height)
// uniform vec2 u_resolution;  // Resolution of the canvas (width, height)

out vec4 fs_color;

// Flame space: x in [-aspect, aspect], y from 0 (bottom) -> ~2 (top).
vec2 flame_space(vec2 uv, float aspect) {
    vec2 p = SB_center_uv(uv, aspect);
    p.y += 1.0;
    return p;
}

// Domain-warped, upward-scrolling turbulence (iq recursive warp): plain fbm is
// cloudy, warped fbm curls like flame. `scale`/`speed` let callers sample the
// SAME field coarse (body motion) or fine (interior detail).
float warp_noise(vec2 p, float t, float scale, float speed) {
    vec2 sp = vec2(p.x * scale, p.y * scale * 0.7 - t * speed);
    vec2 q = vec2(
        SB_fbm(sp + vec2(0.0, 0.0), 5),
        SB_fbm(sp + vec2(5.2, 1.3), 5)
    );
    vec2 r = vec2(
        SB_fbm(sp + 3.0 * q + vec2(1.7, 9.2), 5),
        SB_fbm(sp + 3.0 * q + vec2(8.3, 2.8), 5)
    );
    return SB_fbm(sp + 3.0 * r, 5); // [0, 1]
}

// Teardrop body as a signed field (NOT a width-narrows cone — that reads as a
// triangle). An ellipse whose WIDTH is squeezed by height via the Cyanilux
// Y-remap: width follows (1-y)^p, so it's round and wide at the base and pinches
// to a sharp point at the top — candle/teardrop, not an egg or a cone. Returns
// ~1 deep inside, <=0 at/outside the edge.
float teardrop(vec2 p) {
    float y = clamp(p.y / 1.75, 0.0, 1.0);    // 0 base -> 1 tip
    // Width profile: WIDE and flat at the base (fire sits on the ground), a fat
    // rounded bulge in the lower third, then a smooth CURVED taper to the tip
    // (sin gives the convex teardrop curve; a straight ramp here = triangle).
    float bulge = sin(pow(y, 0.55) * 3.14159);          // 0 at base+tip, 1 mid
    float w = mix(0.46, 0.0, smoothstep(0.0, 1.0, y))    // overall taper, curved
            + bulge * 0.16;                              // belly in the middle
    float edge = abs(p.x) / max(w, 1e-3);     // 0 at axis, 1 at the silhouette
    return 1.0 - edge;
}

// Heat field: the teardrop body, its EDGE eaten away by height-weighted noise so
// the silhouette licks into tongues AND the interior is filled with the noise
// structure (hot veins) instead of a flat slab.
float flame_heat(vec2 p, float t) {
    float body = teardrop(p);

    // Two noise scales: coarse (big tongues / body motion) + fine (interior
    // crackle). Combined, the body has structure at every scale.
    float coarse = warp_noise(p, t, 1.4, 1.7);
    float fine = warp_noise(p, t, 3.6, 2.6);
    float n = coarse * 0.7 + fine * 0.3;

    // Perturb the body edge MORE toward the top (calm base, licking tip). The
    // strong top perturbation is what tears the rounded top into separate tongues
    // instead of a smooth dome.
    float perturb = 0.2 + smoothstep(0.0, 1.5, p.y) * 1.5;
    float h = body + (n - 0.5) * perturb;

    // Dissolve the upper flame into the noise so the tip FRAYS into tongues
    // rather than ending in a clean geometric needle.
    h -= smoothstep(0.9, 1.9, p.y) * (1.0 - n) * 0.9;

    // Keep the raw field (not a hard mask): the interior keeps the noise veins,
    // and intensity still climbs toward the rooted base.
    h *= mix(0.65, 1.3, n);                              // veins across the body
    h *= 1.0 + (1.0 - smoothstep(0.0, 0.6, p.y)) * 0.6;  // hotter at the root
    return clamp(h, 0.0, 1.5);
}

// Blackbody-ish temperature ramp. Over-scaled so the HOTTEST field peaks blow to
// white, but mid values stay orange/yellow — so the body shows internal veins.
vec3 fire_ramp(float h) {
    vec3 c = vec3(0.0);
    c = mix(c, vec3(0.5, 0.04, 0.0), smoothstep(0.05, 0.35, h)); // deep ember
    c = mix(c, vec3(0.95, 0.25, 0.02), smoothstep(0.3, 0.55, h)); // orange
    c = mix(c, vec3(1.0, 0.65, 0.12), smoothstep(0.5, 0.78, h));  // yellow
    c = mix(c, vec3(1.0, 0.92, 0.55), smoothstep(0.75, 1.0, h));  // near-white
    c = mix(c, vec3(1.0, 1.0, 0.95), smoothstep(1.0, 1.3, h));    // white core
    return c;
}

// Sparse rising embers: a cellular grid scrolling up faster than the flame; few
// cells emit, each a tiny vertically-streaked twinkle, funnelled into a narrow
// cone above the body. Sparks, not a starfield.
vec3 embers(vec2 p, float t) {
    vec2 cs = vec2(p.x * 7.0, (p.y + t * 1.0) * 7.0);
    vec2 cell = floor(cs);
    vec2 fr = fract(cs) - 0.5;
    float emit = SB_hash21(cell);
    float spark = 0.0;
    if (emit > 0.84) {
        vec2 off = vec2(SB_hash21(cell + 3.1) - 0.5, SB_hash21(cell + 7.7) - 0.5) * 0.5;
        vec2 dp = fr - off;
        dp.y *= 0.4;
        spark = exp(-300.0 * dot(dp, dp)) * max(0.0, sin(t * 6.0 + emit * 40.0));
    }
    float band = smoothstep(0.5, 0.9, p.y) * (1.0 - smoothstep(1.4, 2.0, p.y));
    float cone = exp(-pow(abs(p.x) / mix(0.4, 0.15, clamp(p.y * 0.5, 0.0, 1.0)), 2.0));
    return vec3(1.0, 0.6, 0.25) * spark * band * cone * 1.5;
}

void main() {
    vec2 p = flame_space(vs_uv, u_aspect);

    float heat = flame_heat(p, u_time);

    vec3 color = fire_ramp(heat);

    // Soft emissive halo so the fire throws warm light into the dark.
    float halo = exp(-pow(abs(p.x) / mix(0.5, 0.2, clamp(p.y * 0.5, 0.0, 1.0)), 2.0));
    halo *= (1.0 - smoothstep(0.0, 1.9, p.y)) * smoothstep(-0.1, 0.2, p.y);
    color += vec3(1.0, 0.35, 0.08) * halo * 0.22;

    color += embers(p, u_time);

    color = min(color, vec3(1.0));
    fs_color = vec4(color, 1.0);
}
