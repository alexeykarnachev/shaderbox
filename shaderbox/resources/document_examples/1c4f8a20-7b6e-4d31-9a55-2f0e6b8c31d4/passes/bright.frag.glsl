#version 460 core

// PASS 2 of 4 -- the bright pass. Keeps only what is brighter than a threshold, so the blur
// downstream smears the highlights rather than the whole image. Its target is HALF size
// (scale 0.5 in the pass panel), which is both cheaper and a wider blur for the same radius.

in vec2 vs_uv;

// Filled by `scene` -- see the pass list's inputs, or graph.json. An unfilled input reads black.
uniform sampler2D u_src;

uniform float u_threshold = 0.75;

out vec4 fs_color;

void main() {
    vec3 col = texture(u_src, vs_uv).rgb;
    float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    // Soft knee rather than a hard cut: a hard one makes the bloom pop on and off as things move.
    fs_color = vec4(col * smoothstep(u_threshold, u_threshold + 0.25, luma), 1.0);
}
