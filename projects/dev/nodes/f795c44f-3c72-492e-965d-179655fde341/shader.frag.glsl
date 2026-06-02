#version 460 core
in vec2 vs_uv;
out vec4 fs_color;

uniform float u_time;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }

void main() {
    vec2 uv = vs_uv * 6.0;
    vec2 gv = fract(uv) - 0.5;
    vec2 id = floor(uv);

    float n = hash(id);
    if (n > 0.5) gv.x = -gv.x;

    float d = abs(length(gv) - 0.3);
    d = min(d, abs(length(gv - vec2(0.5)) - 0.3));

    vec3 col = vec3(smoothstep(0.02, 0.03, d));
    col *= 0.7 + 0.3 * sin(u_time + id.x + id.y);
    fs_color = vec4(col, 1.0);
}