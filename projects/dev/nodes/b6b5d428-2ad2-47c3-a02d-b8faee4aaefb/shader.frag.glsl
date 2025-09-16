#version 460 core

#define PI 3.141592
#define BOX_EPS 0.025

in vec2 vs_uv;

uniform float u_time;
uniform float u_aspect;

uniform vec3 u_look_at = vec3(0.0, 0.0, 0.0);
uniform vec3 u_cam_pos = vec3(5.0, 5.0, 5.0);
uniform float u_focal_len = 0.5;
uniform vec3 u_box_size = vec3(4.0, 1.0, 4.0);

out vec4 fs_color;

float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    float n = fract((p3.x + p3.y) * p3.z);

    return n;
}

float noise(vec2 p) {
    float x = floor(p.x);
    float y = floor(p.y);
    float u = smoothstep(0, 1, fract(p.x));
    float v = smoothstep(0, 1, fract(p.y));

    vec2 xy = vec2(x, y);
    float a = hash(xy + vec2(-0.5, -0.5));
    float b = hash(xy + vec2(-0.5, +0.5));
    float c = hash(xy + vec2(+0.5, +0.5));
    float d = hash(xy + vec2(+0.5, -0.5));

    float n = mix(mix(a, b, v), mix(d, c, v), u);
    return n;
}

float brownian(vec2 p) {
    int n_octaves = 4;
    float a = 1.0;
    float f = 1.0;
    float k = 0.0;
    float n = 0.0;

    float angle = radians(27.39);
    float angle_sin = sin(angle);
    float angle_cos = cos(angle);

    for (int i = 0; i < n_octaves; ++i) {
        p = vec2(p.x * angle_cos + p.y * angle_sin,
                 -p.x * angle_sin + p.y * angle_cos);

        n += a * noise(f * p);
        k += a;

        a *= 0.5;
        f *= 2.0;
    }

    n /= k;
    return n;
}

float sin01(float x) { return 0.5 * (sin(x) + 1.0); }

float get_terrain_height(vec2 p) {
    float h = brownian(p);
    float k = max(0.0, sin01(u_time) * (1.0 - 0.1 * length(p)));

    h = h * k;

    return h;
}

float get_sd_box(vec3 p, vec3 size) {
    vec3 q = abs(p) - size;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float attenuate(float dist, vec3 coeffs) {
    return 1.0 / dot(vec3(1.0, dist, dist * dist), coeffs);
}

void main() {
    // -------------------------------------------------------------------
    // Screen point
    vec2 sp = vs_uv * 2.0 - 1.0;
    sp.x *= u_aspect;

    // -------------------------------------------------------------------
    // Camera
    vec3 ro;
    vec3 rd;
    {
        vec3 look_at = u_look_at;
        vec3 cam_pos = u_cam_pos;
        float focal_len = u_focal_len;

        vec3 forward = normalize(look_at - cam_pos);
        vec3 up = normalize(vec3(0.0, 1.0, 0.0));
        vec3 right = normalize(cross(forward, up));
        up = normalize(cross(right, forward));

        ro = cam_pos + focal_len * forward + sp.x * right + sp.y * up;
        rd = normalize(ro - cam_pos);
    }

    // -------------------------------------------------------------------
    // Find box hit point
    bool is_hit_box = false;
    vec3 p = ro;
    {
        int max_n_steps = 32;
        float sd = 9999.9;

        for (int i = 0; i < max_n_steps; ++i) {
            float current_sd = get_sd_box(p, u_box_size);

            if (current_sd > sd) {
                is_hit_box = false;
                break;
            }

            if (current_sd <= BOX_EPS) {
                is_hit_box = true;
                break;
            }

            sd = current_sd;
            p += sd * rd;
        }
    }

    // -------------------------------------------------------------------
    // Get terrain height inside the box
    bool is_hit_terrain = false;
    float terrain_height = 0.0;
    {
        float step_size = 0.025;
        int max_n_steps = 128;

        for (int i = 0; i < max_n_steps; ++i) {
            if (get_sd_box(p, u_box_size) > BOX_EPS) {
                break;
            }

            terrain_height = get_terrain_height(p.xz);

            if (p.y <= terrain_height) {
                is_hit_terrain = true;
                break;
            }

            p += step_size * rd;
        }
    }

    // -------------------------------------------------------------------
    // Draw grid on the terrain
    float grid;
    {
        vec2 q = abs(2.0 * (fract(p.xz) - 0.5));
        float d = min(q.x, q.y);
        grid = attenuate(d, vec3(1.0, 5.0, 50.0));
    }

    // -------------------------------------------------------------------
    // Color
    vec3 grid_color = vec3(0.1, 1.0, 0.1);
    vec3 color = grid_color * vec3(grid) * float(is_hit_terrain);
    fs_color = vec4(color, 1.0);
}
