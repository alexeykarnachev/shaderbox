#version 460 core

// PASS 3 of 4 -- the blur. A 13-tap Gaussian in ONE pass for clarity; a separable
// horizontal+vertical pair would be two passes and cheaper, which is a good exercise: add a
// pass, wire it in between, and watch the cost drop with the look unchanged.

in vec2 vs_uv;

uniform sampler2D u_bright;

uniform float u_radius = 2.6;

out vec4 fs_color;

void main() {
    vec2 texel = u_radius / vec2(textureSize(u_bright, 0));
    vec3 sum = vec3(0.0);
    float weight_sum = 0.0;
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            vec2 offset = vec2(float(x), float(y));
            float w = exp(-dot(offset, offset) * 0.35);
            sum += texture(u_bright, vs_uv + offset * texel).rgb * w;
            weight_sum += w;
        }
    }
    fs_color = vec4(sum / weight_sum, 1.0);
}
