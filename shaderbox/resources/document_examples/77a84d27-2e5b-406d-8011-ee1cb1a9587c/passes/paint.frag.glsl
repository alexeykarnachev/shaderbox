#version 460 core

// PAINT -- the scene the light bounces around in.
//
// RGB is emitted colour, ALPHA is "this texel is solid". A wall is opaque and black; a light is
// opaque and bright. Everything downstream keys off that alpha: the seed pass marks the solid
// texels, and a ray stops at the first one it reaches.
//
// Built analytically from SDFs and u_time, so it carries no state and the shipped example needs
// no script: a scene that redraws itself every frame is what lets the two lights drift and the
// shadows follow.

in vec2 vs_uv;
out vec4 fs_color;

uniform float u_time;
uniform float u_drift = 0.06;        // how far the warm light wanders
uniform float u_wall_height = 0.15;
uniform float u_light_radius = 0.035;

float sd_circle(vec2 p, float r) { return length(p) - r; }

float sd_box(vec2 p, vec2 b) {
    vec2 q = abs(p) - b;
    return length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0);
}

void main() {
    vec2 p = vs_uv;

    // Two emitters. The warm one drifts, so the shadows move and you can see the lighting is
    // recomputed every frame rather than baked once.
    vec2 warm_at = vec2(0.28, 0.70 + u_drift * sin(u_time * 0.7));
    float warm = sd_circle(p - warm_at, u_light_radius);
    float cool = sd_circle(p - vec2(0.80, 0.80), 0.030);

    // Two occluders: a wall between the lights, and a round one low-left.
    float wall = sd_box(p - vec2(0.50, 0.50), vec2(0.008, u_wall_height));
    float blob = sd_circle(p - vec2(0.22, 0.26), 0.055);

    if (warm < 0.0) { fs_color = vec4(4.0, 3.3, 2.0, 1.0); return; }
    if (cool < 0.0) { fs_color = vec4(0.5, 1.2, 4.0, 1.0); return; }
    // Solid and black: stops a ray and emits nothing, which is what casts a shadow.
    if (min(wall, blob) < 0.0) { fs_color = vec4(0.0, 0.0, 0.0, 1.0); return; }
    fs_color = vec4(0.0);
}
