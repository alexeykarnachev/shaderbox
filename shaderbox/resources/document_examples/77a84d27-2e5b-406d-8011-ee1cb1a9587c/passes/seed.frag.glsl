#version 460 core

// SEED -- step 1 of the jump flood.
//
// The distance field is built by asking every texel "where is the nearest solid thing?". This
// pass writes the STARTING answer: a solid texel knows its own position, everything else knows
// nothing. Storing a POSITION rather than a distance is what makes the flood work -- the answer
// propagates outward by copying coordinates, and the distance is derived at the end.
//
// vUv * alpha is the article's trick: a solid texel (alpha 1) stores its own UV, an empty one
// stores (0,0), which the flood reads as "no seed here". It costs a real seed exactly at the
// origin texel, which nobody notices.

in vec2 vs_uv;
out vec4 fs_color;

uniform sampler2D u_paint;

void main() {
    float alpha = texture(u_paint, vs_uv).a;
    fs_color = vec4(vs_uv * alpha, 0.0, 1.0);
}
