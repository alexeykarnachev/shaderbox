#version 460 core

in vec2 vs_uv;
out vec4 fs_color;

// A three-step chain: emitters -> a half-res blur -> a trail that reads itself.
// Order is never listed anywhere; the engine takes it from what each step reads.

uniform sampler2D u_scene;   // step, f2
uniform sampler2D u_blur;    // step, scale: 0.25, f2, linear
uniform sampler2D u_trail;   // step, f2

uniform float u_time;
uniform float u_emit;
uniform float u_fade;

void step_scene(out vec4 o) {
    vec2 p = vs_uv - vec2(0.5 + 0.3 * sin(u_time), 0.5 + 0.2 * cos(u_time * 0.7));
    float d = length(p);
    float e = smoothstep(0.06, 0.0, d) * u_emit;
    o = vec4(e, e * 0.5, e * 0.15, 1.0);
}

void step_blur(out vec4 o) {
    // The quarter-res target does most of the widening; four taps finish it.
    vec2 t = 1.5 / vec2(textureSize(u_scene, 0));
    vec4 s = texture(u_scene, vs_uv + vec2( t.x,  t.y))
           + texture(u_scene, vs_uv + vec2(-t.x,  t.y))
           + texture(u_scene, vs_uv + vec2( t.x, -t.y))
           + texture(u_scene, vs_uv + vec2(-t.x, -t.y));
    o = s * 0.25;
}

void step_trail(out vec4 o) {
    // Reads ITSELF: last frame, handed over by the engine. No second buffer.
    vec4 prev = texture(u_trail, vs_uv) * u_fade;
    o = max(texture(u_blur, vs_uv), prev);
}

void main() {
    vec3 c = texture(u_scene, vs_uv).rgb + texture(u_trail, vs_uv).rgb;
    fs_color = vec4(c / (c + 1.0), 1.0);   // tonemap: the targets hold values > 1
}
