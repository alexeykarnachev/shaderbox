#version 460 core

in vec2 vs_uv;
out vec4 fs_color;

uniform float u_time;
uniform float u_aspect;

#define PI 3.141592
#define RM_MAX_DIST 10000.0
#define RM_MAX_N_STEPS 160
#define RM_EPS 0.0001
#define NORMAL_DERIVATIVE_STEP 0.015

const float u_cam_fov = 70.0;
const vec3 u_cam_pos = vec3(6.0, 6.0, 40.0);
const vec2 u_window_size = vec2(0.63, 0.47);
const vec2 u_door_size = vec2(0.71, 0.53);

// city grid
const vec2 CITY_SPACING = vec2(5.0, 6.0);    // x,z distance between building centers
const vec2 BUILDING_HALF = vec2(1.4, 1.6);   // half width (x) / half depth (z)
const float FLOOR_H = 0.5;                    // world height of one floor (y unit)
const float CITY_RADIUS = 5.0;               // city is cells [-R..R] in both axes

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
    vec2 cell;        // building cell id (x,z)
    float height;     // hit building's half-height
};

// https://iquilezles.org/articles/distfunctions/
float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float hash(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.x + p.y) * p.z);
}

float cell_height(vec2 cell) {
    // half-height in world units; varied per building, taller toward the back (-z)
    float h = hash(vec3(cell, 7.0));
    float base = 2.0 + 5.0 * h * h;
    return base;
}

float sd_building(vec3 p, vec2 id, out float height) {
    vec2 local_xz = p.xz - id * CITY_SPACING;
    float h = cell_height(id);
    // building sits on the ground: center its box so the base is at y=-1
    vec3 q = vec3(local_xz.x, p.y - (-1.0 + h), local_xz.y);
    height = h;
    return sdBox(q, vec3(BUILDING_HALF.x, h, BUILDING_HALF.y));
}

// scene distance + which building cell we're nearest. cell/height written via out-params.
// Bounded XZ domain-repetition done CORRECTLY: the nearest building may be in a neighbor of the
// rounded cell, so sample the 3x3 candidates and take the min (iq opLimitedRepetition). Clamping
// round() alone yields an invalid SDF at the footprint edge -> overshoot -> phantom planes.
float map(vec3 p, out vec2 cell, out float height) {
    float ground = p.y + 1.0;

    vec2 base = round(p.xz / CITY_SPACING);
    float building = RM_MAX_DIST;
    cell = vec2(0.0);
    height = 0.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            vec2 id = clamp(base + vec2(i, j), vec2(-CITY_RADIUS), vec2(CITY_RADIUS));
            float h;
            float d = sd_building(p, id, h);
            if (d < building) {
                building = d;
                cell = id;
                height = h;
            }
        }
    }

    if (ground < building) {
        cell = vec2(0.0);
        height = 0.0;
    }
    return min(ground, building);
}

float map(vec3 p) {
    vec2 c;
    float h;
    return map(p, c, h);
}

RayMarchResult march(vec3 ro, vec3 rd) {
    RayMarchResult rm = RayMarchResult(
        0, ro, vec3(0.0), ro, rd, 0.0, 0.0,
        RM_MAX_DIST, RM_MAX_DIST, 0.0, vec2(0.0), 0.0
    );

    for (; rm.i < RM_MAX_N_STEPS; ++rm.i) {
        rm.p = rm.p + rm.rd * rm.sd_last;
        vec2 cell;
        float height;
        float sd_step = map(rm.p, cell, height);
        rm.sd_last = sd_step;
        rm.sd_min_shape = min(rm.sd_min_shape, sd_step);
        rm.sd_min = min(rm.sd_min, sd_step);
        rm.cell = cell;
        rm.height = height;
        rm.dist = length(rm.p - rm.ro);
        if (rm.sd_last < RM_EPS || rm.dist > RM_MAX_DIST) {
            break;
        }
    }

    if (rm.sd_last < RM_EPS) {
        rm.is_hit = 1.0;
        vec2 e = vec2(NORMAL_DERIVATIVE_STEP, 0.0);
        rm.n = normalize(vec3(
            map(rm.p + e.xyy) - map(rm.p - e.xyy),
            map(rm.p + e.yxy) - map(rm.p - e.yxy),
            map(rm.p + e.yyx) - map(rm.p - e.yyx)
        ));
    }

    return rm;
}

float attenuate(float d, vec3 coeffs) {
    return 1.0 / (coeffs.x + coeffs.y * d + coeffs.z * d * d);
}

vec3 draw_window(vec2 uv, vec2 size, vec3 light_color, bool has_frame, bool has_curtains) {
    vec2 p = 0.5 * (uv * 2.0 - 1.0);
    bool is_inside = (abs(p.x) <= 0.5 * size.x) && (abs(p.y) <= 0.5 * size.y);
    float d = distance(p, vec2(0.0));
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

// unwrap a hit on a building face into a tiled grid of windows.
// seed is the building cell id so each tower has its own window pattern.
vec3 draw_facade(vec2 uv, vec2 idx, vec2 seed) {
    vec3 color;
    vec2 window_uv = vec2(fract(uv.x), uv.y);
    float window_idx = floor(uv.x);
    vec2 window_size = u_window_size;
    vec3 light_color;
    float intensity = hash(vec3(seed + idx, window_idx)) * 2.0;
    light_color = vec3(1.0, 0.8, 0.2) * intensity;
    bool has_curtains = hash(vec3(seed + idx, window_idx) + 99.9) > 0.3;

    color = draw_window(window_uv, window_size, light_color, true, has_curtains);

    if (length(color) == 0.0) {
        vec2 p = abs(uv * 2.0 - 1.0);
        float seam = max(p.x, p.y);
        seam = smoothstep(0.9, 1.0, seam);
        float k = 0.05 * hash(vec3(seed + idx, 0.0));
        color = (1.0 - seam) * vec3(k) + seam * vec3(0.1);
    }
    return color;
}

void main() {
    vec2 sp = vs_uv * 2.0 - 1.0;
    sp.x *= u_aspect;

    float fov = radians(u_cam_fov);
    float screen_dist = 1.0 / tan(0.5 * fov);
    vec3 cam_pos = u_cam_pos;

    vec3 look_at = vec3(0.0, 3.0, 0.0) + vec3(0.2 * sin(u_time * 0.3), 0.1 * sin(u_time * 0.5), 0.0);
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

    bool is_building = rm.height > 0.0 && rm.is_hit > 0.5;
    bool is_vertical_face = abs(rm.n.y) < 0.5;

    vec3 color;
    if (is_building && is_vertical_face) {
        // local position inside the hit cell
        vec2 local_xz = rm.p.xz - rm.cell * CITY_SPACING;
        // floors: vertical unwrap from the building base (y=-1) upward
        float fy = (rm.p.y + 1.0) / FLOOR_H;
        float floor_y = fract(fy);
        float floor_idx_y = floor(fy);

        // horizontal unwrap: pick the axis of the face we hit
        bool face_x = abs(rm.n.x) > abs(rm.n.z);
        float u = face_x ? local_xz.y : local_xz.x;   // along-face coordinate
        // map [-half, half] of the face into ~3 windows
        float along = (u / (face_x ? BUILDING_HALF.y : BUILDING_HALF.x)) * 0.5 + 0.5; // 0..1
        float facade_x = along * 3.0;
        vec2 facade_uv = vec2(facade_x, floor_y);
        vec2 facade_idx = vec2(floor(facade_x), floor_idx_y);
        // seed differs per face so the four sides aren't identical
        vec2 seed = rm.cell * 13.0 + (face_x ? vec2(1.0, 0.0) : vec2(0.0, 1.0)) * 7.0;
        color = draw_facade(facade_uv, facade_idx, seed);
    } else if (rm.is_hit > 0.5) {
        // ground / rooftops: dark asphalt with a faint hash grain
        float g = 0.03 + 0.03 * hash(vec3(floor(rm.p.xz * 2.0), 0.0));
        color = vec3(g, g, g * 0.6);
    } else {
        color = vec3(0.0);
    }

    // distance fog toward a dim amber horizon
    vec3 fog_color = vec3(0.10, 0.08, 0.02);
    float d = pow(clamp((rm.dist - 20.0) / 60.0, 0.0, 1.0), 1.5);
    color = mix(color, fog_color, d);

    // sky: amber glow hugging the horizon, fading to black overhead (only where we missed geometry)
    vec3 sky;
    {
        float k = smoothstep(0.25, -0.05, rm.rd.y);
        sky = vec3(0.14, 0.10, 0.02) * k;
    }
    color = color * rm.is_hit + sky * (1.0 - rm.is_hit);

    fs_color = vec4(color, 1.0);
}
