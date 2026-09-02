#version 460 core

// PASS 4 of 4 -- the output. Adds the three upstream targets and tonemaps once, at the end. This
// is the pass the preview and the export show; the others exist only to feed it.

in vec2 vs_uv;

uniform sampler2D u_lit;    // filled by `scene`
uniform sampler2D u_glow;   // filled by `blur`
uniform sampler2D u_trail;  // filled by `trail`

uniform float u_bloom = 1.15;
uniform float u_trail_mix = 0.55;

out vec4 fs_color;

void main() {
    vec3 col = texture(u_lit, vs_uv).rgb;
    col += texture(u_glow, vs_uv).rgb * u_bloom;
    col += texture(u_trail, vs_uv).rgb * u_trail_mix;

    // Reinhard, then sRGB: the upstream targets are f2 and hold values well past 1.0, so
    // displaying them raw would clip to white.
    col = col / (1.0 + col);
    fs_color = vec4(pow(col, vec3(1.0 / 2.2)), 1.0);
}
