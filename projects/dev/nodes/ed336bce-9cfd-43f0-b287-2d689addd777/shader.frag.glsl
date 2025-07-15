#version 460 core

in vec2 vs_uv; // Coordinate of the current pixel to be shaded

uniform sampler2D u_image;
uniform vec3 u_color;

out vec4 fs_color;

void main() {
    vec3 image_color = texture(u_image, vs_uv).rgb;
    vec3 color = image_color * u_color;
    color = pow(color, vec3(1.2));

    fs_color = vec4(color, 1.0);
}
