#version 460 core

// One quad per entity. `vs_quad` is this entity's own coordinate, -1 at the quad's
// edges and 0 at its centre, so a shape written here is drawn once per entity rather
// than once per canvas.
in vec2 vs_quad;

// A `flat in` is one value PER ENTITY, filled by script.py under "@instances".
// `pos` and `radius` are the two the engine needs to place the quad; the rest are
// yours. `flat` is what says per-entity: there is nothing to interpolate across a
// value that is constant over its own quad.
flat in vec2  pos;
flat in float radius;
flat in float heat;

uniform float u_glow = 0.55;      // an ordinary uniform: one slider, one value per pass
uniform float u_core = 0.22;      // how much of the disc is solid before it falls off

out vec4 frag_color;

void main() {
    float d = length(vs_quad);
    if (d > 1.0) discard;                       // outside this entity's disc

    // A soft core with an exponential falloff. Additive blending sums the overlaps, so
    // a dense part of the flock reads as brighter without anything counting neighbours.
    float core = 1.0 - smoothstep(u_core, 1.0, d);
    float glow = exp(-3.0 * d) * u_glow;

    // `heat` is a per-entity value the shader reads exactly like a uniform, except it
    // differs for every one of the twenty thousand. Here it says which flock an entity
    // belongs to and how far out it is orbiting.
    vec3 cold = vec3(0.20, 0.55, 1.00);
    vec3 hot  = vec3(1.00, 0.45, 0.12);
    vec3 tint = mix(cold, hot, clamp(heat, 0.0, 1.0));

    frag_color = vec4(tint * (core + glow), 1.0);
}
