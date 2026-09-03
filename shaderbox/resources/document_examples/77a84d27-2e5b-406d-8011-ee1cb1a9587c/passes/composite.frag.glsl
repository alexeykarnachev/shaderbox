#version 460 core

// COMPOSITE -- what you actually look at.
//
// Cascade level 0 holds four directions per pixel, one per texel of each 1x1 probe... which at
// level 0 means the pixel itself already holds its own averaged radiance. So this pass is
// mostly presentation: tonemap the unbounded light into 0-1, then draw the painted scene back
// on top so lights read as bright and walls as solid.

in vec2 vs_uv;
out vec4 fs_color;

uniform sampler2D u_cascade;
uniform sampler2D u_paint;
uniform float u_exposure = 0.35;  // slider: brighter or dimmer overall

void main() {
    vec3 light = texture(u_cascade, vs_uv).rgb * u_exposure;
    // Reinhard: light/(1+light) maps any brightness into 0-1 without clipping the highlights
    // to flat white, which matters because emitters here are far above 1.
    vec3 mapped = light / (1.0 + light);
    vec4 scene = texture(u_paint, vs_uv);
    // A solid texel draws itself: an emitter its color, a wall its black. Without this the
    // walls would be lit by the light in front of them and stop reading as occluders.
    vec3 rgb = mix(mapped, scene.rgb / (1.0 + scene.rgb), scene.a);
    // Linear -> sRGB. Skipping it makes everything look muddy and crushes the falloff.
    fs_color = vec4(pow(max(rgb, 0.0), vec3(1.0 / 2.2)), 1.0);
}
