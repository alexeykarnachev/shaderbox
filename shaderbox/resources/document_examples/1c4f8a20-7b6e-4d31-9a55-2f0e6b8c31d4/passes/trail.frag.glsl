#version 460 core

// PASS 4a -- FEEDBACK. `u_prev` is wired to THIS pass, which means its own previous frame: the
// engine hands it last frame's target and swaps the pair once per frame, so nothing here manages
// a ping-pong buffer. Fading the history slightly each frame is what makes a trail rather than
// an ever-brightening smear.

in vec2 vs_uv;

uniform sampler2D u_src;   // filled by `scene`
uniform sampler2D u_prev;  // filled by `trail` -- itself, i.e. the previous frame

uniform float u_decay = 0.9;
uniform float u_gain = 0.35;

out vec4 fs_color;

void main() {
    vec3 history = texture(u_prev, vs_uv).rgb * u_decay;
    vec3 fresh = texture(u_src, vs_uv).rgb * u_gain;
    fs_color = vec4(max(history, fresh), 1.0);
}
