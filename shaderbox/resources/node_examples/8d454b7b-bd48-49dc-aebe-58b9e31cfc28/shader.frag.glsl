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
#define CARS_PER_AVENUE 5         // GLSL loop bound must be const -> not a uniform (see u_car_*)

// ============================ TUNABLE UNIFORMS (sorted by prefix) ============================
// Prefixes group by subsystem when the uniform list is name-sorted:
//   u_cam_   camera        u_city_  grid/buildings    u_win_  window lights
//   u_wall_  facade/concrete   u_light_ directional night key   u_fog_  aerial perspective
//   u_sky_   sky+stars+moon    u_plane_ aircraft       u_car_  street traffic

// --- camera ---
uniform vec3  u_cam_pos = vec3(6.0, 6.0, 40.0);
uniform float u_cam_fov = 70.0;
uniform vec3  u_cam_look = vec3(0.0, 3.0, 0.0);
uniform float u_cam_sway = 0.2;            // look-at horizontal sway amplitude

// --- city grid / buildings ---
uniform vec2  u_city_spacing = vec2(5.0, 6.0);   // x,z distance between building centers
uniform vec2  u_city_bldg_half = vec2(1.4, 1.6); // half width(x)/depth(z) of a building
uniform float u_city_floor_h = 0.5;              // world height of one floor
uniform float u_city_radius = 5.0;               // city is cells [-R..R]
uniform float u_city_ground_y = -1.0;
uniform float u_city_height_base = 2.0;          // shortest building half-height
uniform float u_city_height_var = 5.0;           // extra half-height range (x hash^2)

// --- window lights ---
uniform vec2  u_win_size = vec2(0.63, 0.47);     // window glass size within a cell
uniform float u_win_brightness = 2.0;            // max per-window brightness multiplier
uniform float u_win_curtain_frac = 0.3;          // fraction of windows with curtains (hash thresh)
uniform vec3  u_win_warm_color = vec3(1.00, 0.80, 0.42);  // majority tungsten
uniform vec3  u_win_soft_color = vec3(1.00, 0.88, 0.62);  // softer white-yellow
uniform vec3  u_win_cool_color = vec3(0.92, 0.93, 0.88);  // neutral-cool minority
uniform float u_win_falloff_lin = 5.0;           // window glow attenuation linear coeff
uniform float u_win_falloff_quad = 10.0;         // window glow attenuation quadratic coeff

// --- facade / concrete wall ---
uniform vec3  u_wall_grey_lo = vec3(0.085, 0.082, 0.078); // per-building grey range
uniform vec3  u_wall_grey_hi = vec3(0.135, 0.130, 0.140);
uniform float u_wall_grain = 0.03;               // fine grain amplitude
uniform float u_wall_band = 0.012;               // poured-floor horizontal banding amplitude
uniform float u_wall_ao = 0.55;                  // recessed-window AO strength
uniform float u_wall_plinth = 0.55;              // ground-floor base darkening factor

// --- directional night key (sky-dome + moon soft lighting on surfaces) ---
uniform float u_light_ambient = 0.55;            // base (away-face) surface level
uniform float u_light_sky_key = 0.45;            // extra from sky/up-facing
uniform float u_light_moon_key = 0.40;           // extra from facing the moon
uniform vec3  u_light_cool_color = vec3(0.86, 0.92, 1.06); // tint on sky/moon-lit faces
uniform vec3  u_light_warm_color = vec3(1.06, 0.98, 0.88); // tint on away faces

// --- night aerial perspective (depth) ---
uniform float u_fog_start = 18.0;                // distance where haze begins
uniform float u_fog_range = 80.0;                // distance over which it saturates
uniform float u_fog_desat = 0.45;               // how much distance desaturates
uniform float u_fog_haze = 0.7;                  // how much distance tints toward sky glow

// --- sky / stars / moon ---
uniform vec3  u_sky_horizon_color = vec3(0.32, 0.20, 0.09); // warm skyglow at horizon
uniform vec3  u_sky_zenith_color = vec3(0.020, 0.028, 0.055); // cool dark overhead
uniform float u_sky_glow_pow = 2.6;              // how tightly the glow hugs the horizon
uniform float u_sky_star_density = 0.93;         // hash threshold (higher = fewer stars)
uniform float u_sky_star_bright = 1.0;           // star brightness scale
uniform vec3  u_sky_moon_dir = vec3(-0.015, 0.410, -0.912); // moon direction (normalized in code)
uniform float u_sky_moon_size = 0.085;           // moon angular radius
uniform float u_sky_moon_halo = 0.8;             // moon halo strength

// --- aircraft ---
uniform float u_plane_period = 22.0;             // seconds between passes (higher = rarer)
uniform float u_plane_speed = 0.16;              // drift speed across the sky
uniform float u_plane_size = 0.035;              // light radius
uniform float u_plane_blink = 4.0;               // blink rate
uniform vec3  u_plane_color = vec3(1.0, 0.95, 0.9);

// --- street traffic ---
uniform float u_car_speed = 4.0;                 // base car speed
uniform float u_car_speed_var = 5.0;             // per-car speed variation
uniform float u_car_lane = 0.45;                 // lane offset from road center (frac of road half)
uniform float u_car_track = 0.28;                // headlight pair separation (frac of road half)
uniform float u_car_falloff_lin = 5.0;           // car-light attenuation linear
uniform float u_car_falloff_quad = 26.0;         // car-light attenuation quadratic
uniform float u_car_head_boost = 4.5;            // white headlight brightness
uniform float u_car_tail_boost = 4.0;            // red taillight brightness
uniform vec3  u_car_head_color = vec3(1.00, 0.97, 0.90);
uniform vec3  u_car_tail_color = vec3(1.00, 0.16, 0.10);
// =============================================================================================

// horizon-glow color sampled at a given view-y (used by both sky and aerial-perspective fade)
vec3 sky_gradient(float ry) {
    float t = pow(clamp(1.0 - ry, 0.0, 1.0), u_sky_glow_pow);
    return mix(u_sky_zenith_color, u_sky_horizon_color, t);
}

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
    return u_city_height_base + u_city_height_var * h * h;
}

float sd_building(vec3 p, vec2 id, out float height) {
    vec2 local_xz = p.xz - id * u_city_spacing;
    float h = cell_height(id);
    height = h;
    float roof_y = u_city_ground_y + 2.0 * h;
    vec3 q = vec3(local_xz.x, p.y - (u_city_ground_y + h), local_xz.y);
    float body = sdBox(q, vec3(u_city_bldg_half.x, h, u_city_bldg_half.y));

    // cheap rooftop clutter: a water-tank/penthouse box + a thin antenna mast, offset per building
    vec2 ro = (vec2(hash(vec3(id, 11.0)), hash(vec3(id, 12.0))) - 0.5) * u_city_bldg_half;
    vec3 tq = vec3(local_xz.x - ro.x, p.y - (roof_y + 0.25), local_xz.y - ro.y);
    float tank = sdBox(tq, vec3(0.35, 0.25, 0.35));
    vec3 mq = vec3(local_xz.x - ro.x * 0.4, p.y - (roof_y + 0.9), local_xz.y - ro.y * 0.4);
    float mast = sdBox(mq, vec3(0.04, 0.65, 0.04));

    return min(body, min(tank, mast));
}

// bounded XZ repetition: nearest building may be a neighbor of the rounded cell -> 3x3 min (iq).
float map(vec3 p, out vec2 cell, out float height) {
    float ground = p.y - u_city_ground_y;

    vec2 base = round(p.xz / u_city_spacing);
    float building = RM_MAX_DIST;
    cell = vec2(0.0);
    height = 0.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            vec2 id = clamp(base + vec2(i, j), vec2(-u_city_radius), vec2(u_city_radius));
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

// window light color: anchored on warm tungsten, only gentle drift around it.
vec3 window_light_color(float r) {
    if (r < 0.80)      return u_win_warm_color;
    else if (r < 0.95) return u_win_soft_color;
    else               return u_win_cool_color;
}

// pure emissive light from a lit window (0 outside the glass) — wall is drawn separately.
vec3 draw_window(vec2 uv, vec2 size, vec3 light_color, bool has_curtains) {
    vec2 p = 0.5 * (uv * 2.0 - 1.0);
    bool is_inside = (abs(p.x) <= 0.5 * size.x) && (abs(p.y) <= 0.5 * size.y);
    float d = distance(p, vec2(0.0));
    float light = attenuate(d, vec3(1.0, u_win_falloff_lin, u_win_falloff_quad));
    vec3 color = vec3(light) * light_color;
    color = float(is_inside) * color;

    if (has_curtains) {
        color = color * (0.1 + smoothstep(0.25, 0.0, abs(p.x)));
    }
    // frame mullions
    color *= step(0.05 * size.x, abs(p.x));
    color *= step(0.05 * size.x, abs(p.y - 0.2 * size.y));
    return color;
}

// textured concrete wall — varied per building (bldg_seed), gentle so it doesn't stand out.
vec3 wall_color(vec2 facade_uv, vec2 bldg_seed) {
    float bh = hash(vec3(bldg_seed, 5.0));
    vec3 base = mix(u_wall_grey_lo, u_wall_grey_hi, bh);
    float grain = (hash(vec3(floor(facade_uv * vec2(40.0, 60.0)), bldg_seed.x)) - 0.5) * u_wall_grain;
    float band = u_wall_band * sin(facade_uv.y * 6.2831 * 1.0);
    return max(base + grain + band, vec3(0.0));
}

// idx = (window column, floor); n_floors = total floors of this building; bottom/top flags drive
// the plinth + cornice. uv.x carries the column coord (fract = within-window), uv.y the within-floor.
vec3 draw_facade(vec2 uv, vec2 idx, vec2 seed, vec2 bldg_seed, float n_floors) {
    vec2 window_uv = vec2(fract(uv.x), uv.y);
    float window_idx = floor(uv.x);
    vec3 wkey = vec3(seed + idx, window_idx);
    float lit = hash(wkey) * u_win_brightness;
    vec3 col = window_light_color(hash(wkey + 4.7)) * lit;
    bool has_curtains = hash(wkey + 99.9) > (1.0 - u_win_curtain_frac);

    vec3 wall = wall_color(uv, bldg_seed);

    vec2 q = vec2(window_uv.x - 0.5, window_uv.y - 0.5);
    vec2 hw = 0.5 * u_win_size;

    // recessed-window AO: darken a ring just OUTSIDE the glass so windows look inset
    float outside = max(abs(q.x) - hw.x, abs(q.y) - hw.y);
    float ao = smoothstep(0.10, 0.0, outside) * smoothstep(-0.02, 0.04, outside);
    wall *= (1.0 - u_wall_ao * ao);

    // pilasters between window bays
    float pil = smoothstep(0.5, 0.42, abs(q.x));
    wall += vec3(0.010, 0.010, 0.011) * pil;

    // window sill just under the glass
    float sill = smoothstep(0.012, 0.0, abs(q.y - (hw.y + 0.03))) * step(abs(q.x), hw.x + 0.04);
    wall += vec3(0.05, 0.048, 0.045) * sill;

    bool is_bottom = idx.y < 0.5;
    bool is_top = idx.y > n_floors - 1.5;

    // plinth: a darker, window-less base at street level
    if (is_bottom) {
        wall = wall_color(uv, bldg_seed) * u_wall_plinth;
        wall += vec3(0.012) * smoothstep(0.5, 0.46, abs(q.x));
        return wall;
    }

    // cornice: a brighter capping band at the very top
    if (is_top) {
        float cap = smoothstep(0.45, 0.5, window_uv.y);
        wall += vec3(0.03, 0.029, 0.027) * cap;
    }

    vec3 light = draw_window(window_uv, u_win_size, col, has_curtains);
    return wall + light;
}

// streets: roads run between building rows; round to the nearest road CENTERLINE so local is the
// symmetric offset within ONE road (a floor() boundary on the centerline split each road in two).
vec3 draw_street(vec2 xz) {
    vec2 rcell = round(xz / u_city_spacing - 0.5) + 0.5;
    vec2 acell = rcell;
    vec2 local = xz - rcell * u_city_spacing;          // [-half..half] across the road

    float asphalt = 0.018;
    vec3 color = vec3(asphalt, asphalt, asphalt * 0.9);

    float road_half_x = 0.5 * u_city_spacing.x - u_city_bldg_half.x - 0.1;
    float road_half_z = 0.5 * u_city_spacing.y - u_city_bldg_half.y - 0.1;
    bool on_x_road = abs(local.y) < road_half_z;
    bool on_z_road = abs(local.x) < road_half_x;

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

    // cars: red taillights (away) on the RIGHT (local.x>0), white headlights (toward us) on the LEFT,
    // two lights per car across the track width.
    if (on_z_road) {
        float span = u_city_spacing.y * (2.0 * u_city_radius + 2.0);
        float track = u_car_track * road_half_x;
        for (int k = 0; k < CARS_PER_AVENUE; ++k) {
            float seed = hash2(vec2(acell.x * 31.0 + 3.0, float(k)));
            float dir = seed > 0.5 ? 1.0 : -1.0;
            float lane = -dir * u_car_lane * road_half_x;
            float speed = (u_car_speed + u_car_speed_var * fract(seed * 7.0)) * dir;
            float carz = mod(speed * u_time + span * seed, span) - 0.5 * span;
            float dz = xz.y - carz;
            vec3 car_col = dir > 0.0 ? u_car_head_color : u_car_tail_color;
            float boost = dir > 0.0 ? u_car_head_boost : u_car_tail_boost;
            for (int s = -1; s <= 1; s += 2) {
                float dx = local.x - (lane + float(s) * track);
                float cd = length(vec2(dx * 3.2, dz * 1.4));
                float glow = attenuate(cd, vec3(1.0, u_car_falloff_lin, u_car_falloff_quad));
                color += car_col * glow * boost;
            }
        }
    }
    return color;
}

// night sky: gradient + stars + moon + aircraft
vec3 draw_sky(vec3 rd) {
    vec3 col = sky_gradient(rd.y);

    // stars
    if (rd.y > 0.02) {
        vec2 suv = rd.xz / max(rd.y, 0.05);
        vec2 g = floor(suv * 26.0);
        float s = hash2(g);
        if (s > u_sky_star_density) {
            vec2 f = fract(suv * 26.0) - 0.5;
            float star = smoothstep(0.18, 0.0, length(f));
            float tw = 0.5 + 0.5 * sin(u_time * 3.0 + s * 100.0);
            float bright = smoothstep(0.93, 1.0, s);
            col += vec3(0.85, 0.9, 1.0) * star * tw * (0.4 + bright) * u_sky_star_bright
                 * smoothstep(0.0, 0.2, rd.y);
        }
    }

    // moon
    vec3 moon_dir = normalize(u_sky_moon_dir);
    float md = distance(normalize(rd), moon_dir);
    float disc = smoothstep(u_sky_moon_size, u_sky_moon_size - 0.015, md);
    float crater = 0.85 + 0.15 * hash2(normalize(rd).xy * 40.0);
    float halo = attenuate(md, vec3(1.0, 8.0, 70.0));
    col += vec3(0.97, 0.95, 0.88) * disc * crater;
    col += vec3(0.35, 0.38, 0.48) * halo * u_sky_moon_halo;

    // a distant aircraft: ONE small blinking light crossing the sky, infrequently
    if (rd.y > 0.03) {
        vec2 suv = rd.xz / max(rd.y, 0.05);
        float cycle = floor(u_time / u_plane_period);
        float local_t = u_time - cycle * u_plane_period;
        float ylane = mix(-1.85, -2.25, fract(cycle * 0.37));
        float xpos = -1.9 + local_t * u_plane_speed;
        vec2 dlt = suv - vec2(xpos, ylane);
        float blink = 0.5 + 0.5 * sin(u_time * u_plane_blink);
        float light = smoothstep(u_plane_size, 0.0, length(dlt));
        col += u_plane_color * light * (0.25 + 0.95 * blink);
    }
    return col;
}

void main() {
    vec2 sp = vs_uv * 2.0 - 1.0;
    sp.x *= u_aspect;

    float fov = radians(u_cam_fov);
    float screen_dist = 1.0 / tan(0.5 * fov);
    vec3 cam_pos = u_cam_pos;

    vec3 look_at = u_cam_look + vec3(u_cam_sway * sin(u_time * 0.3), 0.5 * u_cam_sway * sin(u_time * 0.5), 0.0);
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
        vec2 local_xz = rm.p.xz - rm.cell * u_city_spacing;
        float fy = (rm.p.y - u_city_ground_y) / u_city_floor_h;
        float floor_y = fract(fy);
        float floor_idx_y = floor(fy);

        bool face_x = abs(rm.n.x) > abs(rm.n.z);
        float u = face_x ? local_xz.y : local_xz.x;
        float along = (u / (face_x ? u_city_bldg_half.y : u_city_bldg_half.x)) * 0.5 + 0.5;
        float inset = 0.12;
        float band = (along - inset) / (1.0 - 2.0 * inset);
        vec2 facade_uv = vec2(along, floor_y);
        float n_floors = floor((2.0 * rm.height) / u_city_floor_h);
        if (band < 0.0 || band > 1.0) {
            color = wall_color(facade_uv, rm.cell);
        } else {
            float facade_x = band * 3.0;
            vec2 facade_idx = vec2(floor(facade_x), floor_idx_y);
            vec2 seed = rm.cell * 13.0 + (face_x ? vec2(1.0, 0.0) : vec2(0.0, 1.0)) * 7.0;
            color = draw_facade(vec2(facade_x, floor_y), facade_idx, seed, rm.cell, n_floors);
        }
    } else if (rm.is_hit > 0.5 && !is_building) {
        color = draw_street(rm.p.xz);
    } else if (rm.is_hit > 0.5) {
        // building rooftop: dark gravel, varied per building, with faint grain
        float bh = hash(vec3(rm.cell, 5.0));
        vec3 base = mix(vec3(0.045, 0.044, 0.042), vec3(0.075, 0.072, 0.070), bh);
        float grain = (hash(vec3(floor(rm.p.xz * 8.0), 2.0)) - 0.5) * 0.02;
        color = max(base + grain, vec3(0.0));
    } else {
        color = draw_sky(rd);
    }

    // directional night key: sky-dome + moon give surfaces form (cooler/lighter toward sky+moon,
    // darker/warmer away). Applied to the SURFACE term only; window/light EMISSION is left untouched.
    if (rm.is_hit > 0.5) {
        vec3 moon_dir = normalize(u_sky_moon_dir);
        float sky_amt = 0.5 + 0.5 * rm.n.y;
        float moon_amt = max(dot(rm.n, moon_dir), 0.0);
        float key = u_light_ambient + u_light_sky_key * sky_amt + u_light_moon_key * moon_amt;
        vec3 tint = mix(u_light_warm_color, u_light_cool_color, clamp(0.5 * sky_amt + 0.5 * moon_amt, 0.0, 1.0));
        vec3 surface = min(color, vec3(0.22));
        vec3 emissive = color - surface;
        color = surface * key * tint + emissive;
    }

    // night aerial perspective: distance fades the surface toward the horizon-glow color +
    // desaturates, so receding rows separate into depth groups.
    if (rm.is_hit > 0.5) {
        float f = pow(clamp((rm.dist - u_fog_start) / u_fog_range, 0.0, 1.0), 1.3);
        vec3 haze = sky_gradient(max(rd.y, 0.0)) * 1.1;
        float lum = dot(color, vec3(0.299, 0.587, 0.114));
        color = mix(color, vec3(lum), f * u_fog_desat);
        color = mix(color, haze, f * u_fog_haze);
    }

    fs_color = vec4(color, 1.0);
}
