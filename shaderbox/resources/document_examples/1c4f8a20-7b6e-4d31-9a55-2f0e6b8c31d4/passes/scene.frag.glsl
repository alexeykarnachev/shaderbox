#version 460 core

// PASS 1 of 4 — the scene. Three orbiting blobs on a dark ground, written to this pass's own
// render target. Nothing here knows about the passes downstream; they read this target by name.

in vec2 vs_uv;

uniform float u_time;
uniform float u_aspect;

uniform float u_orbit_speed = 0.45;
uniform float u_blob_size = 0.13;

out vec4 fs_color;

vec3 blob(vec2 p, vec2 at, float radius, vec3 tint) {
    float d = length(p - at);
    // Bright core, soft falloff: the core is what the bright-pass will find.
    return tint * (smoothstep(radius, 0.0, d) + 2.4 * smoothstep(radius * 0.35, 0.0, d));
}

void main() {
    vec2 p = (vs_uv - 0.5) * vec2(u_aspect, 1.0);
    float t = u_time * u_orbit_speed;

    vec3 col = vec3(0.02, 0.025, 0.05);
    col += blob(p, 0.28 * vec2(cos(t), sin(t)), u_blob_size, vec3(1.0, 0.35, 0.15));
    col += blob(p, 0.28 * vec2(cos(t + 2.09), sin(t + 2.09)), u_blob_size, vec3(0.2, 0.7, 1.0));
    col += blob(p, 0.28 * vec2(cos(t + 4.19), sin(t + 4.19)), u_blob_size, vec3(0.7, 0.25, 1.0));

    fs_color = vec4(col, 1.0);
}
