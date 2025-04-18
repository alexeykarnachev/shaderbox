#version 460

in vec2 vs_uv;
out vec4 fs_color;

uniform float u_time;
uniform float u_aspect;
uniform float u_blink_frequency = 2.0;
uniform float u_radius = 0.74;
uniform float u_thickness = 0.14;
uniform float u_smoothness = 0.03;

uniform vec2 u_sp_center = vec2(0.0, 0.0);
uniform vec3 u_color = vec3(0.894, 0.494, 0.047);

uniform sampler2D u_texture;

void main() {
    vec2 sp = (vs_uv * 2.0) - 1.0;
    sp.x *= u_aspect;

    float dist = distance(sp, u_sp_center);
    float diff = abs(dist - u_radius);
    float half_thickness = u_thickness / 2.0;
    float line = 1.0 - smoothstep(half_thickness - u_smoothness, half_thickness, diff);

    vec3 texture_color = texture2D(u_texture, vs_uv).rgb;

    float brightness = 0.5 * (sin(u_time * u_blink_frequency) + 1.0);

    vec3 color = (1.0 - brightness) * texture_color + brightness * line * u_color;
    fs_color = vec4(color, 1.0);
}
