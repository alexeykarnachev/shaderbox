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

// city grid
const vec2 CITY_SPACING = vec2(5.0, 6.0);    // x,z distance between building centers
const vec2 BUILDING_HALF = vec2(1.4, 1.6);   // half width (x) / half depth (z)
const float FLOOR_H = 0.5;                    // world height of one floor (y unit)
const float CITY_RADIUS = 5.0;               // city is cells [-R..R] in both axes
const float GROUND_Y = -1.0;

struct RayMarchResult {
    int i;
    vec3 p;
    vec3 n;
    vec3 ro;
    vec3 rd;
    float dist;
    float sd_last;
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

float hash2(vec2 p) { return hash(vec3(p, 0.0)); }

float cell_height(vec2 cell) {
    float h = hash(vec3(cell, 7.0));
    return 2.0 + 5.0 * h * h;
}

float sd_building(vec3 p, vec2 id, out float height) {
    vec2 local_xz = p.xz - id * CITY_SPACING;
    float h = cell_height(id);
    vec3 q = vec3(local_xz.x, p.y - (GROUND_Y + h), local_xz.y);
    height = h;
    return sdBox(q, vec3(BUILDING_HALF.x, h, BUILDING_HALF.y));
}

// bounded XZ repetition: nearest building may be a neighbor of the rounded cell -> 3x3 min (iq).
float map(vec3 p, out vec2 cell, out float height) {
    float ground = p.y - GROUND_Y;

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
        0, ro, vec3(0.0), ro, rd, 0.0, 0.0, 0.0, vec2(0.0), 0.0
    );

    for (; rm.i < RM_MAX_N_STEPS; ++rm.i) {
        rm.p = rm.p + rm.rd * rm.sd_last;
        vec2 cell;
        float height;
        float sd_step = map(rm.p, cell, height);
        rm.sd_last = sd_step;
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

// --- window light color: mostly warm tungsten, a minority cool, rare teal / warm-red ---
vec3 window_light_color(float r) {
    if (r < 0.62)      return vec3(1.00, 0.78, 0.34);  // warm tungsten (majority)
    else if (r < 0.84) return vec3(0.85, 0.92, 1.00);  // cool white / fluorescent
    else if (r < 0.94) return vec3(0.45, 1.00, 0.85);  // teal screen-glow
    else               return vec3(1.00, 0.45, 0.30);  // warm red / lamp
}

vec3 draw_window(vec2 uv, vec2 size, vec3 light_color, bool has_curtains) {
    vec2 p = 0.5 * (uv * 2.0 - 1.0);
    bool is_inside = (abs(p.x) <= 0.5 * size.x) && (abs(p.y) <= 0.5 * size.y);
    float d = distance(p, vec2(0.0));
    float light = attenuate(d, vec3(1.0, 5.0, 10.0));
    vec3 color = vec3(light) * light_color + vec3(0.01);
    color = float(is_inside) * color;

    if (has_curtains) {
        color = color * (0.1 + smoothstep(0.25, 0.0, abs(p.x)));
    }
    // frame mullions
    color *= step(0.05 * size.x, abs(p.x));
    color *= step(0.05 * size.x, abs(p.y - 0.2 * size.y));
    return color;
}

vec3 draw_facade(vec2 uv, vec2 idx, vec2 seed) {
    vec2 window_uv = vec2(fract(uv.x), uv.y);
    float window_idx = floor(uv.x);
    vec3 wkey = vec3(seed + idx, window_idx);
    float lit = hash(wkey) * 2.0;                          // 0..2 brightness
    vec3 col = window_light_color(hash(wkey + 4.7)) * lit; // hue from a different hash
    bool has_curtains = hash(wkey + 99.9) > 0.3;

    vec3 color = draw_window(window_uv, u_window_size, col, has_curtains);

    if (length(color) == 0.0) {
        vec2 p = abs(uv * 2.0 - 1.0);
        float seam = smoothstep(0.9, 1.0, max(p.x, p.y));
        float k = 0.05 * hash(vec3(seed + idx, 0.0));
        color = (1.0 - seam) * vec3(k) + seam * vec3(0.10, 0.10, 0.11);
    }
    return color;
}

// --- streets: roads run along the gaps between building rows (the half-spacing offset) ---
// returns emissive street color for a ground point.
vec3 draw_street(vec2 xz) {
    // road coordinate within an avenue cell (avenues centered on cell boundaries -> +half spacing)
    vec2 acell = floor((xz + 0.5 * CITY_SPACING) / CITY_SPACING);
    vec2 local = xz - (acell * CITY_SPACING - 0.5 * CITY_SPACING); // [-half..half] around road center

    float asphalt = 0.018;
    vec3 color = vec3(asphalt, asphalt, asphalt * 0.9);

    // is this point on a road strip (not under a building footprint)?
    float road_half_x = 0.5 * CITY_SPACING.x - BUILDING_HALF.x - 0.1; // x-running roads
    float road_half_z = 0.5 * CITY_SPACING.y - BUILDING_HALF.y - 0.1; // z-running roads
    bool on_x_road = abs(local.y) < road_half_z; // road running along x
    bool on_z_road = abs(local.x) < road_half_x; // road running along z

    // lane center dashes
    if (on_z_road) {
        float dash = step(0.5, fract(xz.y * 0.6));
        float line = smoothstep(0.06, 0.0, abs(local.x)) * dash;
        color += vec3(0.30, 0.26, 0.10) * line;
    }
    if (on_x_road) {
        float dash = step(0.5, fract(xz.x * 0.6));
        float line = smoothstep(0.06, 0.0, abs(local.y)) * dash;
        color += vec3(0.30, 0.26, 0.10) * line;
    }

    // --- cars: emissive dots gliding along the z-running avenues ---
    // each avenue (column acell.x) carries several cars; right-hand lane offset by travel direction
    // (white headlights moving one way, red taillights the other).
    if (on_z_road) {
        float span = CITY_SPACING.y * (2.0 * CITY_RADIUS + 2.0); // full avenue length
        for (int k = 0; k < 5; ++k) {
            float seed = hash2(vec2(acell.x * 31.0 + 3.0, float(k)));
            float dir = seed > 0.5 ? 1.0 : -1.0;
            float lane = dir * 0.45 * road_half_x;                // keep-right offset from center
            float speed = (4.0 + 5.0 * fract(seed * 7.0)) * dir;
            // car world-z position, wrapped over the avenue span
            float carz = mod(speed * u_time + span * seed, span) - 0.5 * span;
            float dx = local.x - lane;        // lateral offset within the road (world units)
            float dz = xz.y - carz;           // along-avenue offset (world z)
            float cd = length(vec2(dx * 2.2, dz));
            float glow = attenuate(cd, vec3(1.0, 3.5, 14.0));
            vec3 car_col = dir > 0.0 ? vec3(1.0, 0.95, 0.85) : vec3(1.0, 0.20, 0.13);
            color += car_col * glow * 3.0;
        }
    }
    return color;
}

// --- night sky: stars + moon + dim horizon city-glow ---
vec3 draw_sky(vec3 rd) {
    // horizon glow (amber, hugging the horizon)
    float hk = smoothstep(0.28, -0.05, rd.y);
    vec3 col = vec3(0.13, 0.10, 0.03) * hk;

    // stars: above the horizon, denser, twinkling
    if (rd.y > 0.02) {
        vec2 suv = rd.xz / max(rd.y, 0.05);     // project onto a sky plane
        vec2 g = floor(suv * 26.0);
        float s = hash2(g);
        if (s > 0.93) {
            vec2 f = fract(suv * 26.0) - 0.5;
            float star = smoothstep(0.18, 0.0, length(f));
            float tw = 0.5 + 0.5 * sin(u_time * 3.0 + s * 100.0);
            float bright = smoothstep(0.93, 1.0, s);   // a few rare bright ones
            col += vec3(0.85, 0.9, 1.0) * star * tw * (0.4 + bright) * smoothstep(0.0, 0.2, rd.y);
        }
    }

    // moon — placed in open sky upper area, in-frame (see NOTES: projected to NDC to position it)
    vec3 moon_dir = normalize(vec3(-0.015, 0.410, -0.912));
    float md = distance(normalize(rd), moon_dir);
    float disc = smoothstep(0.085, 0.070, md);
    float crater = 0.85 + 0.15 * hash2(normalize(rd).xy * 40.0);
    float halo = attenuate(md, vec3(1.0, 8.0, 70.0));
    col += vec3(0.97, 0.95, 0.88) * disc * crater;
    col += vec3(0.35, 0.38, 0.48) * halo * 0.8;
    return col;
}

void main() {
    vec2 sp = vs_uv * 2.0 - 1.0;
    sp.x *= u_aspect;

    float fov = radians(u_cam_fov);
    float screen_dist = 1.0 / tan(0.5 * fov);
    vec3 cam_pos = u_cam_pos;

    vec3 look_at = vec3(0.0, 3.0, 0.0) + vec3(0.2 * sin(u_time * 0.3), 0.1 * sin(u_time * 0.5), 0.0);
    vec3 forward = normalize(look_at - cam_pos);
    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up = normalize(cross(right, forward));

    vec3 ro = cam_pos;
    vec3 rd = normalize(cam_pos + forward * screen_dist + right * sp.x + up * sp.y - cam_pos);
    RayMarchResult rm = march(ro, rd);

    bool is_building = rm.height > 0.0 && rm.is_hit > 0.5;
    bool is_vertical_face = abs(rm.n.y) < 0.5;

    vec3 color;
    if (is_building && is_vertical_face) {
        vec2 local_xz = rm.p.xz - rm.cell * CITY_SPACING;
        float fy = (rm.p.y - GROUND_Y) / FLOOR_H;
        float floor_y = fract(fy);
        float floor_idx_y = floor(fy);

        bool face_x = abs(rm.n.x) > abs(rm.n.z);
        float u = face_x ? local_xz.y : local_xz.x;
        float along = (u / (face_x ? BUILDING_HALF.y : BUILDING_HALF.x)) * 0.5 + 0.5;
        float facade_x = along * 3.0;
        vec2 facade_uv = vec2(facade_x, floor_y);
        vec2 facade_idx = vec2(floor(facade_x), floor_idx_y);
        vec2 seed = rm.cell * 13.0 + (face_x ? vec2(1.0, 0.0) : vec2(0.0, 1.0)) * 7.0;
        color = draw_facade(facade_uv, facade_idx, seed);
    } else if (rm.is_hit > 0.5 && !is_building) {
        color = draw_street(rm.p.xz);              // ground plane
    } else if (rm.is_hit > 0.5) {
        // building rooftop
        float g = 0.05 + 0.03 * hash(vec3(floor(rm.p.xz * 2.0), 1.0));
        color = vec3(g, g, g * 0.7);
    } else {
        color = draw_sky(rd);
    }

    // slight distance fog (mostly to relax far aliasing/tearing); only on geometry hits
    if (rm.is_hit > 0.5) {
        vec3 fog_color = vec3(0.09, 0.075, 0.03);
        float d = pow(clamp((rm.dist - 25.0) / 90.0, 0.0, 1.0), 1.4) * 0.7;
        color = mix(color, fog_color, d);
    }

    fs_color = vec4(color, 1.0);
}
