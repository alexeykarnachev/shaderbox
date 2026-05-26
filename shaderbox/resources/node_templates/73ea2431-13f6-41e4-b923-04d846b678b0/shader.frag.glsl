#version 460 core

in vec2 vs_uv; // Coordinate of the current pixel to be shaded

uniform sampler2D u_image; // Static image, sampled on the left half
uniform sampler2D u_video; // Video frame, sampled on the right half
uniform vec3 u_color;      // Tint multiplied over both halves

out vec4 fs_color;

void main() {
    // Split the canvas down the middle: image on the left, video on the right.
    bool is_left = vs_uv.x < 0.5;
    vec3 media = is_left ? texture(u_image, vs_uv).rgb
                         : texture(u_video, vs_uv).rgb;

    vec3 color = media * u_color;

    // A thin dark divider line at the seam.
    float seam = smoothstep(0.002, 0.0, abs(vs_uv.x - 0.5));
    color *= 1.0 - seam;

    fs_color = vec4(color, 1.0);
}
