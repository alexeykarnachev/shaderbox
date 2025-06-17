#version 460 core

in vec2 vs_uv;

uniform float u_time;
uniform float u_aspect;

uniform vec3 u_hand_w0 = vec3(1.31, -1.46, 0.16);
uniform vec3 u_hand_w1 = vec3(0.1, 1.69, -0.52);
uniform vec2 u_hand_e0 = vec2(0.0, 0.3);
uniform vec2 u_hand_e1 = vec2(0.1, 0.6);
uniform float u_hand_p0 = 0.7;

uniform vec3 u_ring_w0 = vec3(0.93, -0.02, -0.7);
uniform vec3 u_ring_w1 = vec3(2.37, -0.84, -1.38);
uniform vec2 u_ring_e0 = vec2(0.0, 0.2);
uniform vec2 u_ring_e1 = vec2(0.05, 0.25);
uniform float u_ring_p0 = 1.0;

uniform vec3 u_ring_attenuation = vec3(0.1, 10.0, 20.0);

uniform vec3 u_base_hand_color = vec3(0.0, 0.0, 1.0);
uniform vec3 u_base_ring_color = vec3(1.0, 0.0, 1.0);

uniform sampler2D u_video;

out vec4 fs_color;

float get_mask(vec3 base_color_0, vec3 base_color_1, vec3 w0, vec3 w1, vec2 e0,
               vec2 e1, float p0) {
    float c = dot(base_color_0, w0);
    c = smoothstep(e0.x, e0.y, c);

    vec3 color = c * base_color_1;

    c = dot(color, w1);
    c = smoothstep(e1.x, e1.y, c);

    c = pow(c, p0);

    return c;
}

void main() {
    vec3 video_color = texture(u_video, vs_uv).rgb;
    float hand = get_mask(video_color, video_color, u_hand_w0, u_hand_w1,
                          u_hand_e0, u_hand_e1, u_hand_p0);
    float ring = get_mask(1.0 - video_color, video_color, u_ring_w0, u_ring_w1,
                          u_ring_e0, u_ring_e1, u_ring_p0);

    float d = distance(vs_uv, vec2(0.5, 0.5));
    ring *= 1.0 / (dot(vec3(1.0, d, d * d), u_ring_attenuation));
    ring = smoothstep(0.1, 0.2, ring);

    vec3 hand_color = hand * u_base_hand_color;
    vec3 ring_color = ring * u_base_ring_color;

    float c = ring + hand;
    vec3 color = hand_color + ring_color;

    fs_color = vec4(color, 1.0);
}
