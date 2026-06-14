#version 460 core

in vec2 vs_uv; // Coordinate of the current pixel to be shaded

// --- engine-driven (the renderer sets these; off-limits to the script) ---
uniform float u_time;   // Time (s) since the application started
uniform float u_aspect; // Aspect ratio of the canvas (width / height)

// --- MANUAL: the script never returns these, so they stay hand-editable in the panel ---
uniform float u_zoom;        // overall zoom of the field
uniform vec3  u_base_color;  // background base colour
uniform float u_grid_density;// background grid line frequency

// --- SCRIPT-DRIVEN: the node brain (script.py) returns these from ONE stateful object ---
uniform float u_pulse;       // 0..1 breathing pulse
uniform float u_swirl;       // ever-accumulating swirl angle (stateful)
uniform vec2  u_wave_offset; // lissajous offset of the whole field
uniform vec3  u_tint;        // cycling hue tint over the scene
uniform vec2  u_orbit_pos;   // orbiting blob centre
uniform float u_orbit_radius;// blob radius (pulses with the orbit)
uniform float u_flash;       // 0..1 flash that fires once per orbit lap
uniform float u_spin;        // blob inner-spin angle (stateful)

out vec4 fs_color;

mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

float sd_box(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

void main() {
    // Field space: centered, aspect-corrected, zoomed, wave-shifted, swirled.
    vec2 p = (vs_uv - 0.5) * 2.0;
    p.x *= u_aspect;
    p /= max(u_zoom, 0.05);
    p += u_wave_offset;
    p *= rot(u_swirl);

    // Background grid that breathes with u_pulse.
    vec2 g = abs(fract(p * u_grid_density) - 0.5);
    float line = smoothstep(0.45, 0.5, max(g.x, g.y));
    vec3 col = mix(u_base_color, u_base_color * (0.4 + 0.6 * u_pulse), line);

    // The orbital blob (a spinning square at u_orbit_pos).
    vec2 bp = (p - u_orbit_pos) * rot(u_spin);
    float box = sd_box(bp, vec2(u_orbit_radius));
    float blob = smoothstep(0.02, 0.0, box);
    vec3 blob_col = vec3(1.0, 0.6, 0.2) + u_flash;
    col = mix(col, blob_col, blob);

    // Hue tint over the whole frame.
    col *= u_tint;

    fs_color = vec4(col, 1.0);
}
