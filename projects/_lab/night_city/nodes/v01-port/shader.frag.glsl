#version 460 core

in vec2 vs_uv;
out vec4 fs_color;

uniform float u_time;
uniform float u_aspect;

#define PI 3.141592
#define RM_MAX_DIST 10000.0
#define RM_MAX_N_STEPS 128
#define RM_EPS 0.0001
#define NORMAL_DERIVATIVE_STEP 0.015

const float u_home_length = 2.0;
const float u_distortion = 20.48;
const vec3 u_cam_pos = vec3(2.75, -0.22, 3.12);
const float u_cam_fov = 53.0;
const vec2 u_window_size = vec2(0.63, 0.47);
const vec2 u_door_size = vec2(0.71, 0.53);

struct RayMarchResult {
    int i;
    vec3 p;
    vec3 n;
    vec3 ro;
    vec3 rd;
    float dist;
    float sd_last;
    float sd_min;
    float sd_min_shape;
    float is_hit;
};

// https://iquilezles.org/articles/distfunctions/
float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float get_sd_shape(vec3 p) {
    float d = sdBox(p, vec3(1.0, 1.0, u_home_length));
    d = min(d, p.y + 1.0);
    return d;
}

RayMarchResult march(vec3 ro, vec3 rd) {
    RayMarchResult rm = RayMarchResult(
        0,
        ro,
        vec3(0.0),
        ro,
        rd,
        0.0,
        0.0,
        RM_MAX_DIST,
        RM_MAX_DIST,
        0.0
    );

    for (; rm.i < RM_MAX_N_STEPS; ++rm.i) {
        rm.p = rm.p + rm.rd * rm.sd_last;
        float sd_step_shape = get_sd_shape(rm.p);
        rm.sd_last = sd_step_shape;
        rm.sd_min_shape = min(rm.sd_min_shape, sd_step_shape);
        rm.sd_min = min(rm.sd_min, sd_step_shape);
        rm.dist += length(rm.p - rm.ro);
        if (rm.sd_last < RM_EPS || rm.dist > RM_MAX_DIST) {
            if (rm.sd_last < RM_EPS) {
                rm.n = vec3(1.0);
            }
            break;
        }
    }

    if (rm.sd_last < RM_EPS) {
        rm.n = vec3(0.0);
        rm.is_hit = 1.0;
        if (rm.sd_last == rm.sd_min_shape) {
            vec2 e = vec2(NORMAL_DERIVATIVE_STEP, 0.0);
            rm.n = normalize(vec3(
                get_sd_shape(rm.p + e.xyy) - get_sd_shape(rm.p - e.xyy),
                get_sd_shape(rm.p + e.yxy) - get_sd_shape(rm.p - e.yxy),
                get_sd_shape(rm.p + e.yyx) - get_sd_shape(rm.p - e.yyx)
            ));
        }
    }

    return rm;
}

float attenuate(float d, vec3 coeffs) {
    return 1.0 / (coeffs.x + coeffs.y * d + coeffs.z * d * d);
}

float hash(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.x + p.y) * p.z);
}

vec3 draw_window(vec2 uv, vec2 size, vec3 light_color, bool has_frame, bool has_curtains) {
    vec2 p = 0.5 * (uv * 2.0 - 1.0);
    bool is_inside = (abs(p.x) <= 0.5 * size.x) && (abs(p.y) <= 0.5 * size.y);
    vec2 light_pos = vec2(0.0, 0.0);
    float d = distance(p, light_pos);
    float light = attenuate(d, vec3(1.0, 5.0, 10.0));
    vec3 color = vec3(light) * light_color + vec3(0.01);
    color = float(is_inside) * color;

    if (has_curtains) {
        color = color * (0.1 + smoothstep(0.25, 0.0, abs(p.x)));
    }

    if (has_frame) {
        color *= step(0.05 * size.x, abs(p.x));
        color *= step(0.05 * size.x, abs(p.y - 0.2 * size.y));
    }

    return color;
}

vec3 draw_door(vec2 uv, vec2 size) {
    vec2 p = 0.5 * (uv * 2.0 - 1.0);
    p.y += 0.5;
    bool is_inside = (abs(p.x) <= 0.5 * size.x) && (p.y <= size.y);
    vec3 color = float(is_inside) * vec3(0.08, 0.05, 0.0);
    color *= step(0.05, abs(p.x));
    return color;
}

vec3 draw_floor(vec2 uv, vec2 idx, bool is_forward_face) {
    vec3 color;
    float _x = uv.x * 3.0;
    float window_x = fract(_x);
    float window_idx = floor(_x);
    vec2 window_uv = vec2(window_x, uv.y);
    vec2 window_size = u_window_size;
    vec3 door_color = vec3(0.0);
    vec3 window_light_color = vec3(0.0);
    bool window_has_frame = false;
    bool window_has_curtains = false;

    if (is_forward_face && idx.y == 0.0 && window_idx == 1.0) {
        door_color = draw_door(window_uv, u_door_size);
        window_light_color = vec3(0.3, 0.3, 0.0);
        window_size.y *= 0.5;
        window_uv.y -= 0.75 * window_size.y;
    } else {
        float intensity = hash(vec3(idx, window_idx)) * 2.0;
        window_light_color = vec3(1.0, 0.8, 0.2) * intensity;
        window_has_curtains = hash(vec3(idx, window_idx) + 99.9) > 0.3;
        window_has_frame = true;
    }

    vec3 window_color = draw_window(window_uv, window_size, window_light_color, window_has_frame, window_has_curtains);
    color = window_color + door_color;

    if (length(color) == 0.0) {
        vec2 p = abs(uv * 2.0 - 1.0);
        float seam = max(p.x, p.y);
        seam = smoothstep(0.9, 1.0, seam);
        float k = 0.05 * hash(vec3(idx, 0.0));
        color = (1.0 - seam) * vec3(k) + seam * vec3(0.1);
    }

    return color;
}

void main() {
    vec2 sp = vs_uv * 2.0 - 1.0;
    sp.x *= u_aspect;
    float u_time_local = u_time;

    {
        float r = length(sp);
        float power = PI * u_distortion / 100.0;
        sp = normalize(sp) * tan(r * power) / tan(power);
    }

    {
        float p = mod(u_time_local, 6.0) / 3.0;
        float y_split = mix(1.0, -1.0, p);
        if (sp.y <= y_split) {
            sp.x -= 0.1;
        }
    }

    float fov = radians(u_cam_fov);
    float screen_dist = 1.0 / tan(0.5 * fov);
    vec3 cam_pos = u_cam_pos;

    vec3 look_at = vec3(0.1 * sin(u_time_local), 0.05 * sin(u_time_local * 2.0), 0.0);
    vec3 forward = normalize(look_at - cam_pos);
    vec3 world_up = vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(forward, world_up));
    vec3 up = normalize(cross(right, forward));

    RayMarchResult rm;
    {
        vec3 screen_center = cam_pos + forward * screen_dist;
        vec3 sp_world = screen_center + right * sp.x + up * sp.y;
        vec3 ro = cam_pos;
        vec3 rd = normalize(sp_world - cam_pos);
        rm = march(ro, rd);
    }

    float y = rm.p.y * 2.0 + 2.0;
    float floor_y = fract(y);
    float floor_idx_y = floor(y);
    bool is_forward_face = rm.n.x > 0.5;

    float _x = is_forward_face ? -rm.p.z + u_home_length : rm.p.x;
    float floor_x = fract(_x);
    float floor_idx_x = floor(_x);
    vec2 floor_idx = vec2(floor_idx_x, floor_idx_y);

    vec2 floor_uv = vec2(floor_x, floor_y);

    vec3 fog_color = vec3(0.3, 0.3, 0.0);
    vec3 color = draw_floor(floor_uv, floor_idx, is_forward_face);
    float d = pow(clamp(rm.dist / 150.0, 0.0, 1.0), 2.0);
    color = (1.0 - d) * color + d * fog_color;

    vec3 background_color = vec3(0.0);
    {
        float k = smoothstep(0.5, 0.0, rm.rd.y);
        background_color = vec3(0.2 * k, 0.2 * k, 0.0);
    }

    color = color * rm.is_hit + background_color * (1.0 - rm.is_hit);

    float scanline = 1.0;
    {
        scanline = 0.5 * (sin(200.0 * sp.y) + 1.0);
        scanline = clamp(scanline, 0.5, 1.0);
    }

    color = color * scanline;

    fs_color = vec4(color, 1.0);
}
