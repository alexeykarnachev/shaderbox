#version 460 core

// DISTANCE FIELD -- turn the flood's coordinates into a number.
//
// The flood left every texel holding the UV of its nearest solid texel. The distance is just
// how far that is. One line of real work.
//
// This is what makes the light passes affordable: a ray can now ask "how far can I move before
// I could possibly hit anything?" and jump that whole distance in one step, instead of creeping
// forward a texel at a time. That is sphere marching, and it is why the naive version in the
// tutorial is slow and this one is not.

in vec2 vs_uv;
out vec4 fs_color;

uniform sampler2D u_jfa;

void main() {
    vec2 nearest = texture(u_jfa, vs_uv).xy;
    // A texel the flood never reached (no solid texel anywhere) reads (0,0); clamping to 1
    // makes it "as far as it gets", which lets a ray cross an empty canvas in one jump.
    float d = (nearest.x == 0.0 && nearest.y == 0.0) ? 1.0 : distance(vs_uv, nearest);
    fs_color = vec4(vec3(clamp(d, 0.0, 1.0)), 1.0);
}
