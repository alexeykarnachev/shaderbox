#version 460 core

// JFA -- the jump flood, and the pass this whole feature exists for.
//
// ONE shader, run 12 times. The offset follows the canvas, so 12 runs cover every canvas size
// the app allows. The engine draws it twelve times in a row, handing each run its index in
// u_pass_iteration and feeding each run the previous run's output through u_prev (the pass
// reads ITSELF).
//
// Each run samples 8 neighbours at a HALVING offset derived from the canvas -- at 512 that is
// 256 texels, then 128, 64 ... 1. Big jumps first spread coordinates across the canvas; small
// jumps refine. After ceil(log2(max side)) runs every texel holds the UV of its nearest solid
// texel; the runs past that copy their input forward.
//
// The offset is derived from the index and the canvas, not handed over: the engine's job is to
// say WHICH run this is, and the algorithm's job is to know what that means.

in vec2 vs_uv;
out vec4 fs_color;

uniform sampler2D u_prev;
uniform sampler2D u_seed;
uniform float u_pass_iteration;
uniform vec2 u_resolution;

void main() {
    float offset = exp2(ceil(log2(max(u_resolution.x, u_resolution.y))) - 1.0 - u_pass_iteration);
    // Runs past the end of the chain have nothing left to spread; pass the buffer through.
    if (offset < 1.0) {
        fs_color = u_pass_iteration < 0.5 ? texture(u_seed, vs_uv) : texture(u_prev, vs_uv);
        return;
    }
    // Run 0 reads the SEED; every later run reads what the run before it wrote. An iterated
    // pass that read only itself would never receive the seed at all -- the chain has to be
    // started from outside, and this is the seam where that happens.
    bool first = u_pass_iteration < 0.5;
    vec4 nearest = vec4(0.0);
    float nearest_dist = 1e9;

    for (float y = -1.0; y <= 1.0; y += 1.0) {
        for (float x = -1.0; x <= 1.0; x += 1.0) {
            vec2 uv = vs_uv + vec2(x, y) * offset / u_resolution;
            if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) continue;
            vec4 sampled = first ? texture(u_seed, uv) : texture(u_prev, uv);
            // (0,0) means "this texel has not been reached yet" -- skip it, or every empty
            // neighbour would claim to be a seed at the canvas corner.
            if (sampled.x == 0.0 && sampled.y == 0.0) continue;
            vec2 diff = sampled.xy - vs_uv;
            float d = dot(diff, diff);
            if (d < nearest_dist) {
                nearest_dist = d;
                nearest = sampled;
            }
        }
    }
    fs_color = nearest;
}
