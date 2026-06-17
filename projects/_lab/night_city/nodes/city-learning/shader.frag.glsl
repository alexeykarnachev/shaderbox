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
#define CARS_PER_AVENUE 8   // max slots; u_car_intensity gates how many actually render

// ============================ scene tunables (seeded from v06 tuning) =========================
uniform vec3  u_cam_pos = vec3(6.0, 6.68, 40.0);
uniform float u_cam_fov = 76.62;
uniform vec3  u_cam_look = vec3(-2.22, 8.19, 0.0);
uniform float u_cam_sway = 1.92;

uniform vec2  u_city_spacing = vec2(5.0, 6.0);
uniform vec2  u_city_bldg_half = vec2(1.38, 1.36);
uniform float u_city_floor_h = 0.74;
uniform float u_city_radius = 5.0;
uniform float u_city_ground_y = -1.3;
uniform float u_city_height_base = 2.1;
uniform float u_city_height_var = 4.89;

uniform vec2  u_win_size = vec2(0.60, 0.49);
uniform float u_win_brightness = 2.0;
uniform float u_win_curtain_frac = 0.51;
uniform vec3  u_win_warm_color = vec3(0.744, 0.588, 0.291);
uniform vec3  u_win_soft_color = vec3(0.572, 0.442, 0.160);
uniform vec3  u_win_cool_color = vec3(0.623, 0.435, 0.0);
uniform float u_win_falloff_lin = 2.97;
uniform float u_win_falloff_quad = 20.0;

uniform vec3  u_wall_grey_lo = vec3(0.093, 0.093, 0.093);
uniform vec3  u_wall_grey_hi = vec3(0.126, 0.123, 0.113);
uniform float u_wall_grain = 0.04;
uniform float u_wall_band = 0.01;
uniform float u_wall_ao = 0.5;
uniform float u_wall_plinth = 0.55;

uniform float u_light_ambient = 0.4;
uniform float u_light_sky_key = 0.5;
uniform float u_light_moon_key = 4.38;
uniform vec3  u_light_cool_color = vec3(0.549, 0.717, 1.0);
uniform vec3  u_light_warm_color = vec3(1.0, 1.0, 1.0);

uniform float u_fog_start = 20.0;
uniform float u_fog_range = 70.0;
uniform float u_fog_haze = 0.89;

uniform vec3  u_sky_horizon_color = vec3(0.437, 0.344, 0.195);
uniform vec3  u_sky_zenith_color = vec3(0.071, 0.074, 0.084);
uniform float u_sky_glow_pow = 11.67;
uniform float u_sky_star_density = 0.95;
uniform float u_sky_star_bright = 1.04;
uniform vec3  u_sky_moon_dir = vec3(-0.455, 0.59, -0.882);
uniform float u_sky_moon_size = 0.055;
uniform float u_sky_moon_halo = 0.35;

uniform float u_plane_period = 22.0;
uniform float u_plane_speed = 0.16;
uniform float u_plane_size = 0.035;
uniform float u_plane_blink = 5.54;
uniform vec3  u_plane_color = vec3(1.0, 0.95, 0.9);

uniform float u_car_intensity = 0.6;  // 0..1 traffic density (fraction of the per-avenue slots used)
uniform float u_car_speed = 3.0;
uniform float u_car_speed_var = 1.0;
uniform float u_car_lane = 0.55;
uniform float u_car_track = 0.16;
uniform float u_car_falloff_lin = 10.0;
uniform float u_car_falloff_quad = 100.0;
uniform float u_car_head_boost = 6.38;
uniform float u_car_tail_boost = 4.0;
uniform vec3  u_car_head_color = vec3(1.0, 0.97, 0.9);
uniform vec3  u_car_tail_color = vec3(1.0, 0.16, 0.10);

// ============================ timed step reveal + captions ====================================
// 8 build steps STACK; steps 1-7 run STEP_DUR each, the final step runs FINAL_DUR, then HOLD, loop.
const float STEP_DUR = 3.0;     // duration of steps 1..7
const float FINAL_DUR = 5.0;    // duration of the last step (step 8)
const float HOLD = 4.0;
const float N_STEPS = 8.0;
const float FADE = 0.8;

// start time of step i (1-based): steps 1..7 are STEP_DUR apart; step 8 starts after 7*STEP_DUR.
float step_start(float i) { return (i - 1.0) * STEP_DUR; }
// duration of step i: the final step is longer.
float step_dur(float i) { return (i >= N_STEPS) ? FINAL_DUR : STEP_DUR; }
// total build time (last step start + its duration) and the full loop period.
const float BUILD_END = 7.0 * 3.0 + 5.0;   // = 26.0  (STEP_DUR*7 + FINAL_DUR)
const float LOOP_PERIOD = BUILD_END + HOLD; // = 30.0

// Step index control: 0 = autoplay the timed reveal; 1..8 = jump to / freeze that step.
uniform uint u_step = 0u;

// rooftop highlight (step 4 entrance): the clutter itself is tinted a bright color while it rises,
// then settles to plain. (Replaced an earlier wireframe-box that read as chunky/buggy.)
uniform vec3  u_roofbox_color = vec3(1.0, 0.85, 0.2);  // highlight color (try white/red too)
uniform float u_roofbox_gain = 6.0;                    // extra brightness boost while rising

// Per-step caption "LABEL\nDETAIL" (codepoint arrays; name ends in "text" so the engine encodes
// typed strings). 8 steps.
#define TLEN 40
uniform uint u_step1text[TLEN];
uniform uint u_step2text[TLEN];
uniform uint u_step3text[TLEN];
uniform uint u_step4text[TLEN];
uniform uint u_step5text[TLEN];
uniform uint u_step6text[TLEN];
uniform uint u_step7text[TLEN];
uniform uint u_step8text[TLEN];

// text style copied from the fire timed-reveal node (u_text_pos in THIS node's screen space).
uniform vec2  u_text_pos = vec2(0.0, 0.80);  // caption anchor in screen space (x centred, y near top)
uniform float u_text_size = 0.18;            // glyph height (screen units)
uniform float u_text_weight = 0.10;
uniform float u_text_sharp = 0.06;
uniform float u_text_spacing = 0.73;
uniform float u_text_line_gap = 0.19;
uniform vec3  u_text_color = vec3(1.0, 0.95, 0.85);
uniform float u_text_fade_in = 0.4;
uniform float u_text_fade_out = 0.4;

float reveal_time() {
    if (u_step > 0u) {
        float n = min(float(u_step), N_STEPS);
        return step_start(n) + FADE + 0.001;
    }
    return mod(u_time, LOOP_PERIOD);
}

// weight [0,1] for step i: 0 before its slot, ramps to 1 over FADE, holds at 1 after.
float reveal(float i) {
    float start = step_start(i);
    return smoothstep(start, start + FADE, reveal_time());
}

// animated entrance ramp for step i: 0 -> 1 over the step's whole slot (longer than the fade),
// eased, so motion (grow/rise/stream) plays across the step then holds. In a frozen step it's 1.
float anim(float i, float dur) {
    float start = step_start(i);
    if (u_step > 0u) return (float(u_step) >= i) ? 1.0 : 0.0;
    float t = (reveal_time() - start) / max(dur, 0.001);
    t = clamp(t, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);   // smoothstep ease
}

// =============================================================================================

vec3 sky_gradient(float ry) {
    float t = pow(clamp(1.0 - ry, 0.0, 1.0), u_sky_glow_pow);
    return mix(u_sky_zenith_color, u_sky_horizon_color, t);
}

struct RayMarchResult {
    int i; vec3 p; vec3 n; vec3 ro; vec3 rd;
    float dist; float sd_last; float is_hit; vec2 cell; float height;
};

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

// rooftop clutter gated by ROOF_ON (step 4); GROW (step 1) scales building height from the ground.
// ROOF_RISE (step 4) raises the clutter up onto the roof. All set as globals in main() pre-march.
float ROOF_ON = 0.0;
float GROW = 1.0;
float ROOF_RISE = 1.0;
float CAR_FADE = 1.0;     // step 6: traffic streams in
float PLANE_FADE = 1.0;   // step 8: plane roll-in
float WIN_LIGHT = 1.0;    // step 2: windows light up by height
float DETAIL_MIX = 1.0;   // step 3: detail/texture/window-variety eases in (0..1, not a hard flip)
float SKY_FADE = 1.0;     // step 5: sky+stars+moon fade in
float FOG_FADE = 1.0;     // step 8: aerial-perspective haze fades in
float MOON_RISE = 1.0;    // step 5: moon arcs up into place (0 = below, 1 = final)

// per-building animated half-height (grows from the ground over step 1; staggered per building).
float grown_height(vec2 id, out float full_h) {
    full_h = cell_height(id);
    float stagger = 0.35 * hash(vec3(id, 21.0));      // each tower starts a touch later
    float g = clamp((GROW - stagger) / max(1.0 - stagger, 0.001), 0.0, 1.0);
    g = g * g * (3.0 - 2.0 * g);
    return max(full_h * g, 0.001);
}

float sd_building(vec3 p, vec2 id, out float height) {
    vec2 local_xz = p.xz - id * u_city_spacing;
    float full_h;
    float h = grown_height(id, full_h);               // animated (growing) half-height
    height = full_h;                                  // report FULL height (for floor unwrap / facade)
    float roof_y = u_city_ground_y + 2.0 * h;         // current (animated) top
    vec3 q = vec3(local_xz.x, p.y - (u_city_ground_y + h), local_xz.y);
    float body = sdBox(q, vec3(u_city_bldg_half.x, h, u_city_bldg_half.y));
    if (ROOF_ON < 0.5) return body;

    // clutter rises up from the current roof as it appears (step 4)
    float lift = mix(-0.6, 0.0, ROOF_RISE);
    vec2 ro = (vec2(hash(vec3(id, 11.0)), hash(vec3(id, 12.0))) - 0.5) * u_city_bldg_half;
    vec3 tq = vec3(local_xz.x - ro.x, p.y - (roof_y + 0.25 + lift), local_xz.y - ro.y);
    float tank = sdBox(tq, vec3(0.35, 0.25, 0.35));
    vec3 mq = vec3(local_xz.x - ro.x * 0.4, p.y - (roof_y + 0.9 + lift), local_xz.y - ro.y * 0.4);
    float mast = sdBox(mq, vec3(0.04, 0.65, 0.04));
    return min(body, min(tank, mast));
}

float map(vec3 p, out vec2 cell, out float height) {
    float ground = p.y - u_city_ground_y;
    vec2 base = round(p.xz / u_city_spacing);
    float building = RM_MAX_DIST;
    cell = vec2(0.0); height = 0.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            vec2 id = clamp(base + vec2(i, j), vec2(-u_city_radius), vec2(u_city_radius));
            float h; float d = sd_building(p, id, h);
            if (d < building) { building = d; cell = id; height = h; }
        }
    }
    if (ground < building) { cell = vec2(0.0); height = 0.0; }
    return min(ground, building);
}
float map(vec3 p) { vec2 c; float h; return map(p, c, h); }

RayMarchResult march(vec3 ro, vec3 rd) {
    RayMarchResult rm = RayMarchResult(0, ro, vec3(0.0), ro, rd, 0.0, 0.0, 0.0, vec2(0.0), 0.0);
    for (; rm.i < RM_MAX_N_STEPS; ++rm.i) {
        rm.p = rm.p + rm.rd * rm.sd_last;
        vec2 cell; float height;
        rm.sd_last = map(rm.p, cell, height);
        rm.cell = cell; rm.height = height;
        rm.dist = length(rm.p - rm.ro);
        if (rm.sd_last < RM_EPS || rm.dist > RM_MAX_DIST) break;
    }
    if (rm.sd_last < RM_EPS) {
        rm.is_hit = 1.0;
        vec2 e = vec2(NORMAL_DERIVATIVE_STEP, 0.0);
        rm.n = normalize(vec3(
            map(rm.p + e.xyy) - map(rm.p - e.xyy),
            map(rm.p + e.yxy) - map(rm.p - e.yxy),
            map(rm.p + e.yyx) - map(rm.p - e.yyx)));
    }
    return rm;
}

float attenuate(float d, vec3 coeffs) { return 1.0 / (coeffs.x + coeffs.y * d + coeffs.z * d * d); }

vec3 window_light_color(float r) {
    if (r < 0.80)      return u_win_warm_color;
    else if (r < 0.95) return u_win_soft_color;
    else               return u_win_cool_color;
}

// curtain_amt 0..1: 0 = no curtain, 1 = full curtain darkening (eased in by DETAIL_MIX, not a flip).
vec3 draw_window(vec2 uv, vec2 size, vec3 light_color, float curtain_amt) {
    vec2 p = 0.5 * (uv * 2.0 - 1.0);
    bool is_inside = (abs(p.x) <= 0.5 * size.x) && (abs(p.y) <= 0.5 * size.y);
    float d = distance(p, vec2(0.0));
    float light = attenuate(d, vec3(1.0, u_win_falloff_lin, u_win_falloff_quad));
    vec3 color = float(is_inside) * vec3(light) * light_color;
    float curtain = 0.1 + smoothstep(0.25, 0.0, abs(p.x));
    color *= mix(1.0, curtain, curtain_amt);
    color *= step(0.05 * size.x, abs(p.x));
    color *= step(0.05 * size.x, abs(p.y - 0.2 * size.y));
    return color;
}

// DETAIL_ON (step 3) gates wall texture + AO/sill/pilaster/plinth/cornice + window variety.
float DETAIL_ON = 0.0;

vec3 wall_color(vec2 facade_uv, vec2 bldg_seed) {
    vec3 plain = vec3(0.10);
    if (DETAIL_ON < 0.5) return plain;
    float bh = hash(vec3(bldg_seed, 5.0));
    vec3 base = mix(u_wall_grey_lo, u_wall_grey_hi, bh);
    float grain = (hash(vec3(floor(facade_uv * vec2(40.0, 60.0)), bldg_seed.x)) - 0.5) * u_wall_grain;
    float band = u_wall_band * sin(facade_uv.y * 6.2831);
    vec3 detailed = max(base + grain + band, vec3(0.0));
    return mix(plain, detailed, DETAIL_MIX);   // step 3: texture prostupает (eases in), no hard pop
}

vec3 draw_facade(vec2 uv, vec2 idx, vec2 seed, vec2 bldg_seed, float n_floors) {
    vec2 window_uv = vec2(fract(uv.x), uv.y);
    float window_idx = floor(uv.x);
    vec3 wkey = vec3(seed + idx, window_idx);

    // step 2: windows light up from the ground up (WIN_LIGHT 0->1 sweeps the lit threshold upward),
    // with a little per-window jitter so they pop on rather than march in a clean line.
    float floor_frac = clamp(idx.y / max(n_floors - 1.0, 1.0), 0.0, 1.0);
    float jitter = 0.12 * hash(wkey + 13.0);
    float win_on = smoothstep(floor_frac + jitter - 0.06, floor_frac + jitter, WIN_LIGHT);

    // window color/brightness: crossfade the plain (warm, on/off) into the varied version by
    // DETAIL_MIX so step 3 develops in smoothly rather than snapping.
    float lit_plain = step(0.5, hash(wkey)) * u_win_brightness;
    vec3  col_plain = u_win_warm_color * lit_plain;
    float curtain_amt = 0.0;
    vec3 col = col_plain;
    if (DETAIL_ON >= 0.5) {
        float lit_v = hash(wkey) * u_win_brightness;
        vec3  col_v = window_light_color(hash(wkey + 4.7)) * lit_v;
        bool has_curtains = hash(wkey + 99.9) > (1.0 - u_win_curtain_frac);
        curtain_amt = has_curtains ? DETAIL_MIX : 0.0;   // ease the curtain in, no one-frame flip
        col = mix(col_plain, col_v, DETAIL_MIX);
    }
    col *= win_on;

    vec3 wall = wall_color(uv, bldg_seed);

    float win_mul = 1.0;   // how much of the lit window survives (plinth eases it to 0)
    if (DETAIL_ON >= 0.5) {
        vec2 q = vec2(window_uv.x - 0.5, window_uv.y - 0.5);
        vec2 hw = 0.5 * u_win_size;
        float outside = max(abs(q.x) - hw.x, abs(q.y) - hw.y);
        float ao = smoothstep(0.10, 0.0, outside) * smoothstep(-0.02, 0.04, outside);
        wall *= (1.0 - u_wall_ao * ao * DETAIL_MIX);
        wall += vec3(0.010, 0.010, 0.011) * smoothstep(0.5, 0.42, abs(q.x)) * DETAIL_MIX;
        float sill = smoothstep(0.012, 0.0, abs(q.y - (hw.y + 0.03))) * step(abs(q.x), hw.x + 0.04) * DETAIL_MIX;
        wall += vec3(0.05, 0.048, 0.045) * sill;
        // plinth: EASE the ground floor toward the dark window-less base (no hard one-frame flip).
        if (idx.y < 0.5) {
            vec3 plinth = wall_color(uv, bldg_seed) * u_wall_plinth
                        + vec3(0.012) * smoothstep(0.5, 0.46, abs(q.x));
            wall = mix(wall, plinth, DETAIL_MIX);
            win_mul = 1.0 - DETAIL_MIX;       // its window fades out as the plinth fades in
        }
        if (idx.y > n_floors - 1.5)
            wall += vec3(0.03, 0.029, 0.027) * smoothstep(0.45, 0.5, window_uv.y) * DETAIL_MIX;
    }

    return wall + draw_window(window_uv, u_win_size, col, curtain_amt) * win_mul;
}

// STREET_ON (step 6) gates roads + cars; before that the ground is plain asphalt.
float STREET_ON = 0.0;

vec3 draw_street(vec2 xz) {
    if (STREET_ON < 0.5) return vec3(0.02, 0.02, 0.018);
    vec2 rcell = round(xz / u_city_spacing - 0.5) + 0.5;
    vec2 local = xz - rcell * u_city_spacing;
    vec3 color = vec3(0.018, 0.018, 0.016);
    float road_half_x = 0.5 * u_city_spacing.x - u_city_bldg_half.x - 0.1;
    float road_half_z = 0.5 * u_city_spacing.y - u_city_bldg_half.y - 0.1;
    bool on_x_road = abs(local.y) < road_half_z;
    bool on_z_road = abs(local.x) < road_half_x;
    if (on_z_road) {
        float d = fract(xz.y * 0.6);
        float dash = smoothstep(0.45, 0.55, d) * smoothstep(1.0, 0.9, d);  // soft dash ends
        color += vec3(0.13, 0.11, 0.05) * smoothstep(0.045, 0.0, abs(local.x)) * dash;
    }
    if (on_x_road) {
        float d = fract(xz.x * 0.6);
        float dash = smoothstep(0.45, 0.55, d) * smoothstep(1.0, 0.9, d);
        color += vec3(0.13, 0.11, 0.05) * smoothstep(0.045, 0.0, abs(local.y)) * dash;
    }
    if (on_z_road) {
        float span = u_city_spacing.y * (2.0 * u_city_radius + 2.0);
        float track = u_car_track * road_half_x;
        for (int k = 0; k < CARS_PER_AVENUE; ++k) {
            float seed = hash2(vec2(rcell.x * 31.0 + 3.0, float(k)));
            // intensity gates how many slots are occupied (per-car hash vs threshold) -> density slider
            if (hash2(vec2(rcell.x * 17.0 + 1.0, float(k))) > u_car_intensity) continue;
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
                color += car_col * attenuate(cd, vec3(1.0, u_car_falloff_lin, u_car_falloff_quad)) * boost * CAR_FADE;
            }
        }
    }
    return color;
}

// SKY_ON (step 5) + PLANE_ON (step 8) gate the sky detail / aircraft.
float SKY_ON = 0.0;
float PLANE_ON = 0.0;

// moon direction animated to arc UP into place over step 5: at rise=0 it sits well below the
// horizon (and swung aside), at rise=1 it's the final u_sky_moon_dir. A simple eased rotation
// about the camera's right-ish axis gives a rising-arc sweep.
vec3 moon_dir_anim() {
    vec3 final_dir = normalize(u_sky_moon_dir);
    float a = (1.0 - MOON_RISE) * 1.6;             // radians to swing down/back at the start
    float ca = cos(a), sa = sin(a);
    // rotate in the plane spanned by (y-up) and the final dir's azimuth -> a vertical arc
    vec3 axis = normalize(cross(final_dir, vec3(0.0, 1.0, 0.0)));
    // Rodrigues rotation of final_dir about `axis` by -a (downward at start)
    vec3 r = final_dir * ca + cross(axis, final_dir) * (-sa) + axis * dot(axis, final_dir) * (1.0 - ca);
    return normalize(r);
}

vec3 draw_sky(vec3 rd) {
    if (SKY_ON < 0.5) return vec3(0.02, 0.02, 0.03);
    // sky gradient + stars fade up from the flat pre-sky color
    vec3 flat_sky = vec3(0.02, 0.02, 0.03);
    vec3 col = mix(flat_sky, sky_gradient(rd.y), SKY_FADE);
    if (rd.y > 0.02) {
        vec2 suv = rd.xz / max(rd.y, 0.05);
        vec2 g = floor(suv * 26.0);
        float s = hash2(g);
        if (s > u_sky_star_density) {
            vec2 f = fract(suv * 26.0) - 0.5;
            float star = smoothstep(0.18, 0.0, length(f));
            float tw = 0.5 + 0.5 * sin(u_time * 3.0 + s * 100.0);
            float bright = smoothstep(0.93, 1.0, s);
            col += vec3(0.85, 0.9, 1.0) * star * tw * (0.4 + bright) * u_sky_star_bright * smoothstep(0.0, 0.2, rd.y) * SKY_FADE;
        }
    }
    vec3 moon_dir = moon_dir_anim();
    float md = distance(normalize(rd), moon_dir);
    float disc = smoothstep(u_sky_moon_size, u_sky_moon_size - 0.015, md);
    float crater = 0.85 + 0.15 * hash2(normalize(rd).xy * 40.0);
    col += vec3(0.97, 0.95, 0.88) * disc * crater * SKY_FADE;
    col += vec3(0.35, 0.38, 0.48) * attenuate(md, vec3(1.0, 8.0, 70.0)) * u_sky_moon_halo * SKY_FADE;
    if (PLANE_ON >= 0.5 && rd.y > 0.03) {
        vec2 suv = rd.xz / max(rd.y, 0.05);
        float cycle = floor(u_time / u_plane_period);
        float local_t = u_time - cycle * u_plane_period;
        float ylane = mix(-1.85, -2.25, fract(cycle * 0.37));
        float xpos = -1.9 + local_t * u_plane_speed;
        vec2 dlt = suv - vec2(xpos, ylane);
        float blink = 0.5 + 0.5 * sin(u_time * u_plane_blink);
        col += u_plane_color * smoothstep(u_plane_size, 0.0, length(dlt)) * (0.25 + 0.95 * blink) * PLANE_FADE;
    }
    return col;
}

// --- captions (ported from the fire timed-reveal node) ---
float draw_caption(vec2 p, uint s[TLEN], float ch, float weight) {
    int cnt0 = 0, cnt1 = 0, line = 0;
    for (int i = 0; i < TLEN; ++i) {
        uint c = s[i];
        if (c == 0u) break;
        if (c == 10u) { line = 1; continue; }
        if (line == 0) cnt0++; else cnt1++;
    }
    if (cnt0 == 0 && cnt1 == 0) return 0.0;
    float maxW = (u_aspect - 0.08) * 2.0;
    float ch0 = ch, ch1 = ch * 0.8;
    float w0 = float(cnt0) * ch0 * u_text_spacing;
    float w1 = float(cnt1) * ch1 * u_text_spacing;
    if (w0 > maxW) ch0 *= maxW / w0;
    if (w1 > maxW) ch1 *= maxW / w1;
    float adv0 = ch0 * u_text_spacing, adv1 = ch1 * u_text_spacing;
    float x0_0 = u_text_pos.x - adv0 * float(cnt0 - 1) * 0.5;
    float x0_1 = u_text_pos.x - adv1 * float(max(cnt1 - 1, 0)) * 0.5;
    float ymath = u_text_pos.y - u_text_line_gap;
    float d = 1e9;
    int k0 = 0, k1 = 0; line = 0;
    for (int i = 0; i < TLEN; ++i) {
        uint cp = s[i];
        if (cp == 0u) break;
        if (cp == 10u) { line = 1; continue; }
        if (line == 0) {
            vec2 c = vec2(x0_0 + adv0 * float(k0++), u_text_pos.y);
            d = min(d, SB_sd_char((p - c) / (0.5 * ch0), cp, weight) * (0.5 * ch0));
        } else {
            vec2 c = vec2(x0_1 + adv1 * float(k1++), ymath);
            d = min(d, SB_sd_char((p - c) / (0.5 * ch1), cp, weight) * (0.5 * ch1));
        }
    }
    return 1.0 - smoothstep(0.0, 0.5 * ch0 * max(u_text_sharp, 0.001), d);
}

float caption_alpha(float i) {
    float start = step_start(i);
    float slotEnd = start + step_dur(i);
    if (u_step > 0u) return (float(u_step) == i) ? 1.0 : 0.0;
    float rt = reveal_time();
    float fin = smoothstep(start, start + max(u_text_fade_in, 0.001), rt);
    float fout = 1.0 - smoothstep(slotEnd - max(u_text_fade_out, 0.001), slotEnd, rt);
    return fin * fout;
}

float step_caption(int i, vec2 p) {
    float w = u_text_weight, ch = u_text_size;
    if (i == 1) return draw_caption(p, u_step1text, ch, w);
    if (i == 2) return draw_caption(p, u_step2text, ch, w);
    if (i == 3) return draw_caption(p, u_step3text, ch, w);
    if (i == 4) return draw_caption(p, u_step4text, ch, w);
    if (i == 5) return draw_caption(p, u_step5text, ch, w);
    if (i == 6) return draw_caption(p, u_step6text, ch, w);
    if (i == 7) return draw_caption(p, u_step7text, ch, w);
    return draw_caption(p, u_step8text, ch, w);
}

void main() {
    // per-step reveal weights (each stacks once revealed)
    float w1 = reveal(1.0); // geometry (bare boxes)
    float w2 = reveal(2.0); // windows
    float w3 = reveal(3.0); // facade detail + concrete
    float w4 = reveal(4.0); // rooftops
    float w5 = reveal(5.0); // sky
    float w6 = reveal(6.0); // streets + cars
    float w7 = reveal(7.0); // directional light
    float w8 = reveal(8.0); // aerial perspective + plane

    // feature presence gates (geometry-affecting ones must be set BEFORE marching).
    // ROOF/STREET/PLANE stay binary (they spawn geometry/sprites); detail/sky/fog ease in smoothly.
    ROOF_ON   = step(0.5, w4);
    STREET_ON = step(0.5, w6);
    PLANE_ON  = step(0.5, w8);
    // SKY_ON / DETAIL_ON flip at their SLOT START (from the anim ramp) so their fade-in starts at 0
    // and nothing pops in one frame. (Set below where the anim ramps are computed.)

    // entrance animations (eased 0->1 across each step's slot, then hold)
    GROW       = anim(1.0, STEP_DUR);  // step 1: buildings grow from the ground
    WIN_LIGHT  = anim(2.0, STEP_DUR);  // step 2: windows light up progressively (by height)
    DETAIL_MIX = anim(3.0, STEP_DUR);  // step 3: texture/AO/window-variety eases in (no hard pop)
    // DETAIL_ON must turn on at the SLOT START (when DETAIL_MIX is still 0), NOT at reveal>=0.5 — else
    // the detail terms jump to a nonzero blend in one frame (the white sill/band stripe artifact).
    DETAIL_ON = step(0.001, DETAIL_MIX);
    ROOF_RISE  = anim(4.0, STEP_DUR);  // step 4: rooftop clutter rises onto the roof
    SKY_FADE   = anim(5.0, STEP_DUR);  // step 5: sky+stars fade in
    SKY_ON    = step(0.001, SKY_FADE);
    MOON_RISE  = anim(5.0, STEP_DUR * 1.4); // step 5: moon arcs up (slower, lingers into the hold)
    CAR_FADE   = anim(6.0, STEP_DUR);  // step 6: traffic streams in
    PLANE_FADE = anim(8.0, FINAL_DUR); // step 8: plane roll-in (final step is longer)
    FOG_FADE   = anim(8.0, FINAL_DUR); // step 8: haze fades in

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

    // rooftop clutter (tanks/masts) sits ABOVE the building roof — shade it as a plain dark
    // structure, never as windowed facade. roof_y uses the FULL height (grow is done by step 4).
    float roof_y = u_city_ground_y + 2.0 * rm.height;
    bool is_clutter = is_building && (rm.p.y > roof_y + 0.02);

    vec3 color;
    if (is_clutter) {
        float bh = hash(vec3(rm.cell, 5.0));
        color = mix(vec3(0.05, 0.05, 0.052), vec3(0.085, 0.083, 0.080), bh);  // plain metal/concrete
        // while it rises (step 4), TINT the clutter itself a bright color so it's clearly visible.
        float rise_activity = sin(clamp(ROOF_RISE, 0.0, 1.0) * PI);  // 0 at start/end, 1 mid-rise
        color = mix(color, u_roofbox_color, rise_activity * 0.85);
        color += u_roofbox_color * rise_activity * (u_roofbox_gain - 1.0) * 0.15;
    } else if (is_building && is_vertical_face) {
        if (w2 < 0.5) {
            color = vec3(0.12);                       // step1: flat grey box face
        } else {
            vec2 local_xz = rm.p.xz - rm.cell * u_city_spacing;
            float fy = (rm.p.y - u_city_ground_y) / u_city_floor_h;
            float floor_y = fract(fy), floor_idx_y = floor(fy);
            bool face_x = abs(rm.n.x) > abs(rm.n.z);
            float u = face_x ? local_xz.y : local_xz.x;
            float along = (u / (face_x ? u_city_bldg_half.y : u_city_bldg_half.x)) * 0.5 + 0.5;
            float inset = 0.12;
            float band = (along - inset) / (1.0 - 2.0 * inset);
            float n_floors = floor((2.0 * rm.height) / u_city_floor_h);
            if (band < 0.0 || band > 1.0) {
                color = wall_color(vec2(along, floor_y), rm.cell);
            } else {
                float facade_x = band * 3.0;
                vec2 facade_idx = vec2(floor(facade_x), floor_idx_y);
                vec2 seed = rm.cell * 13.0 + (face_x ? vec2(1.0, 0.0) : vec2(0.0, 1.0)) * 7.0;
                color = draw_facade(vec2(facade_x, floor_y), facade_idx, seed, rm.cell, n_floors);
            }
        }
    } else if (rm.is_hit > 0.5 && !is_building) {
        color = draw_street(rm.p.xz);
    } else if (rm.is_hit > 0.5) {
        float bh = hash(vec3(rm.cell, 5.0));
        vec3 base = mix(vec3(0.045, 0.044, 0.042), vec3(0.075, 0.072, 0.070), bh);
        color = max(base + (hash(vec3(floor(rm.p.xz * 8.0), 2.0)) - 0.5) * 0.02, vec3(0.0));
    } else {
        color = draw_sky(rd);
    }

    // step 7: directional night key (surface term only), eased in so it doesn't snap on in one frame.
    if (rm.is_hit > 0.5 && w7 >= 0.5) {
        float lf = anim(7.0, STEP_DUR);
        vec3 moon_dir = normalize(u_sky_moon_dir);
        float sky_amt = 0.5 + 0.5 * rm.n.y;
        float moon_amt = max(dot(rm.n, moon_dir), 0.0);
        float key = u_light_ambient + u_light_sky_key * sky_amt + u_light_moon_key * moon_amt;
        vec3 tint = mix(u_light_warm_color, u_light_cool_color, clamp(0.5 * sky_amt + 0.5 * moon_amt, 0.0, 1.0));
        key = mix(1.0, key, lf);          // 1.0 = no change -> ease toward the full key
        tint = mix(vec3(1.0), tint, lf);
        vec3 surface = min(color, vec3(0.22));
        color = surface * key * tint + (color - surface);
    }

    // step 8: night aerial perspective (haze fades in smoothly)
    if (rm.is_hit > 0.5 && w8 >= 0.5) {
        float f = pow(clamp((rm.dist - u_fog_start) / u_fog_range, 0.0, 1.0), 1.3) * FOG_FADE;
        vec3 haze = sky_gradient(max(rd.y, 0.0)) * 1.1;
        color = mix(color, haze, f * u_fog_haze);
    }

    color *= w1;  // the whole scene fades up at step 1

    // step captions on top (screen space; centred), faded per their slot
    vec2 tp = vec2(sp.x, (vs_uv.y * 2.0 - 1.0));   // x already *aspect; y in [-1,1]
    float ink = 0.0;
    for (int i = 1; i <= 8; ++i) {
        float a = caption_alpha(float(i));
        if (a > 0.0) ink = max(ink, step_caption(i, tp) * a);
    }
    color = mix(color, u_text_color, clamp(ink, 0.0, 1.0));

    fs_color = vec4(min(color, vec3(1.0)), 1.0);
}
