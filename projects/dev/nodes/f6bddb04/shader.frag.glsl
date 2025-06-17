#version 460 core
#define PI 3.141592

in vec2 vs_uv;
out vec4 fs_color;

uniform float u_time;
uniform float u_aspect;

uniform float u_rm_max_n_steps = 128.0;

uniform sampler2D u_image_0;
uniform sampler2D u_depth_0;

uniform sampler2D u_image_1;
uniform vec2 u_image_1_offset = vec2(0.0, 0.0);
uniform float u_image_1_zoomout = 1.0;

struct ParallaxResult {
    vec2 uv;

    float is_converged;
    float step_idx;
    float depth;
    float diff;
};

ParallaxResult do_parallax(sampler2D depth_texture, vec3 cam_pos, vec2 input_uv,
                           float aspect, float max_n_steps) {
    vec3 p = vec3((input_uv - 0.5) * vec2(u_aspect, 1.0), 0.0);
    vec3 ray_dir = normalize(p - cam_pos);

    float step_size = 1.0 / max_n_steps;
    float step_idx = 0.0;
    float depth = 9999.9;
    float diff = 0.0;
    vec2 output_uv = vec2(0.0);
    float is_converged = 0.0;

    while (true) {
        if (step_idx >= max_n_steps) {
            break;
        }

        output_uv = p.xy / vec2(u_aspect, 1.0) + 0.5;
        if (output_uv.x >= 1.0 || output_uv.x <= 0.0 || output_uv.y >= 1.0 ||
            output_uv.y <= 0.0) {
            break;
        }

        depth = texture2D(depth_texture, output_uv).r;
        depth = clamp(depth, 0.001, 0.999);

        diff = p.z - depth;
        if (diff >= 0.0) {
            is_converged = 1.0;
            break;
        }

        p += step_size * ray_dir;
        step_idx += 1.0;
    }

    return ParallaxResult(output_uv, is_converged, step_idx, depth, diff);
}

void main() {
    vec3 cam_pos = vec3(0.15 * sin(0.33 * u_time * 2.0 * PI),
                        0.15 * cos(0.33 * u_time * 2.0 * PI), -0.75);
    ParallaxResult pr =
        do_parallax(u_depth_0, cam_pos, vs_uv, u_aspect, u_rm_max_n_steps);
    vec3 color_0 = texture(u_image_0, pr.uv).rgb;

    vec2 uv_1 = ((pr.uv - 0.5) + u_image_1_offset) * u_image_1_zoomout;
    uv_1 += 0.5;
    vec3 color_1 = vec3(0.0);
    if (uv_1.x <= 1.0 && uv_1.x >= 0.0 && uv_1.y <= 1.0 && uv_1.x >= 0.0) {
        color_1 = texture(u_image_1, uv_1).rgb;
    }

    float d = pow((1.0 - pr.depth), 2.0);
    vec3 color = color_1 * pr.is_converged * d + 0.35 * color_0 * d;
    float alpha = pr.is_converged;
    fs_color = vec4(color, alpha);
}
