#version 460 core

// CASCADE -- radiance cascades, one shader run 6 times, coarse level first.
//
// The idea in one sentence: light far away needs many DIRECTIONS but few POSITIONS, and light
// close by needs the opposite -- so compute the far field on a coarse grid with many angles,
// the near field on a fine grid with few, and add them.
//
// Set "Runs per frame" to 6. Run 0 is the COARSEST level (this shader reverses the index, so
// u_pass_iteration 0 means level 5), and each run merges the one above it through u_prev.
//
// HOW A LEVEL IS PACKED. Every level writes the same full-size texture. At level c the probes
// sit sp = 2^c texels apart, and the sp x sp block of texels belonging to one probe stores that
// probe's directions -- one direction per texel. So a coarse level has few probes each holding
// many directions, and level 0 is one probe per pixel holding 4. The texture never changes
// size; only the meaning of its texels does.
//
// INTERVALS. Level c marches the shell [BASE*(4^c-1)/3, BASE*(4^(c+1)-1)/3). Each level reaches
// four times further than the last, they tile without gap or overlap, and level 5 spans the
// canvas. A ray that finds nothing in its own shell asks the level above what lies beyond.

in vec2 vs_uv;
out vec4 fs_color;

uniform sampler2D u_paint;
uniform sampler2D u_df;
uniform sampler2D u_prev;
uniform float u_pass_iteration;
uniform float u_pass_iterations;
uniform vec2 u_resolution;

const float TAU = 6.28318530718;
const float BASE = 0.012;

// Sphere-march from o along d, between t0 and t1. Returns the light hit and whether anything
// stopped the ray. The distance field is what makes each step a jump rather than a crawl.
vec4 march(vec2 o, vec2 d, float t0, float t1) {
    float s = t0;
    for (int i = 0; i < 96; i++) {
        vec2 p = o + d * s;
        if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0) return vec4(0.0, 0.0, 0.0, 1.0);
        float h = texture(u_df, p).r;
        if (h < 0.0012) {
            vec4 hit = texture(u_paint, p);
            // Solid: emit its colour and STOP. A wall is solid and black, which is how a
            // shadow happens -- the ray is consumed and contributes nothing.
            return vec4(hit.rgb, 1.0);
        }
        s += max(h, 0.0012);
        if (s > t1) return vec4(0.0);   // ran out of shell: not blocked, ask the level above
    }
    return vec4(0.0);
}

void main() {
    // Run 0 is the COARSEST level. Merging has to go coarse -> fine, because a level reads the
    // one above it, and the engine only counts upward.
    float level = u_pass_iterations - 1.0 - u_pass_iteration;
    float sp = exp2(level);
    float rays = 4.0 * sp * sp;

    vec2 co = floor(vs_uv * u_resolution);
    vec2 probe = floor(co / sp);          // which probe this texel belongs to
    vec2 slot = mod(co, sp);              // which direction inside that probe
    float si = slot.x + slot.y * sp;
    vec2 probe_uv = (probe + 0.5) * sp / u_resolution;

    float t0 = (level == 0.0) ? 0.0 : BASE * (pow(4.0, level) - 1.0) / 3.0;
    float t1 = BASE * (pow(4.0, level + 1.0) - 1.0) / 3.0;

    float usp = sp * 2.0;                 // the level above is spaced twice as wide
    vec2 upf = probe_uv * u_resolution / usp - 0.5;

    vec3 acc = vec3(0.0);
    for (float k = 0.0; k < 4.0; k++) {
        float idx = si * 4.0 + k;
        float a = TAU * (idx + 0.5) / rays;
        vec4 hit = march(probe_uv, vec2(cos(a), sin(a)), t0, t1);
        vec3 r = hit.rgb;

        if (hit.a < 0.5 && level < u_pass_iterations - 1.0) {
            // THE MERGE. Get this wrong and the picture still looks like global illumination
            // while being numerically far off -- it happened during 063 (every direction read
            // the wrong slot, 30% error, convincing shadows). The published article's own
            // snippet mixes two different quantities here; this is the corrected form.
            //
            // The level above wrote the MEAN of its 4 sub-directions into each slot, so its
            // slot `idx` already holds exactly this ray's angular children. One tap, and BOTH
            // components of the slot address use usp.
            vec2 uslot = vec2(mod(idx, usp), floor(idx / usp));
            vec2 base = floor(upf), fr = fract(upf);
            vec2 lim = vec2(u_resolution / usp - 1.0);
            vec2 c00 = clamp(base, vec2(0.0), lim) * usp + uslot + 0.5;
            vec2 c10 = clamp(base + vec2(1, 0), vec2(0.0), lim) * usp + uslot + 0.5;
            vec2 c01 = clamp(base + vec2(0, 1), vec2(0.0), lim) * usp + uslot + 0.5;
            vec2 c11 = clamp(base + vec2(1, 1), vec2(0.0), lim) * usp + uslot + 0.5;
            // Bilinear BY HAND across the four neighbouring probes, sampling the SAME slot in
            // each. Letting the sampler do it would blend neighbouring slots -- different
            // directions -- which is a different and wrong quantity. The clamp keeps a probe at
            // the edge from reading across to the far side and leaking light.
            r += mix(mix(texture(u_prev, c00 / u_resolution).rgb,
                         texture(u_prev, c10 / u_resolution).rgb, fr.x),
                     mix(texture(u_prev, c01 / u_resolution).rgb,
                         texture(u_prev, c11 / u_resolution).rgb, fr.x), fr.y);
        }
        acc += r;
    }
    fs_color = vec4(acc * 0.25, 1.0);
}
