#version 460 core

// RENDER STEPS — several draws in one node.
//
// A `// step` comment on a sampler declares a draw. Its body is `step_<name>`, and the
// engine works out the running order from who reads whom: nothing is listed anywhere.
// Read a step's own sampler inside its body and you get LAST FRAME — that is feedback,
// with no second buffer to manage.
//
// This chain is: sparks -> bright -> blur (quarter-res) -> trail (reads itself) -> out.

in vec2 vs_uv;
out vec4 fs_color;

uniform float u_time;
uniform float u_aspect;

// f2 keeps values above 1.0, which bloom needs: 8-bit would clip the highlights away
// before the blur ever sees them.
uniform sampler2D u_sparks;  // step, f2
uniform sampler2D u_bright;  // step, scale: 0.5, f2
uniform sampler2D u_blur;    // step, scale: 0.25, f2, linear
uniform sampler2D u_trail;   // step, f2

uniform int u_spark_count = 4;
uniform float u_spark_size = 0.055;
uniform float u_glow = 4.5;
uniform float u_threshold = 0.7;
uniform float u_bloom = 1.1;
uniform float u_fade = 0.93;

// Where spark i sits at time t. Lissajous, so they weave without ever repeating.
vec2 spark_pos(int i, float t) {
    float k = float(i) * 2.4;
    // Two incommensurable rates per axis, so the paths weave and never quite repeat.
    return vec2(
        0.5 + 0.33 * sin(t * 0.61 + k) + 0.08 * sin(t * 1.7 + k * 2.0),
        0.5 + 0.30 * cos(t * 0.47 + k * 1.3) + 0.08 * cos(t * 1.3 + k)
    );
}

// --- step 1: the emitters. Nothing reads a step here, so it runs first. -------------
void step_sparks(out vec4 o) {
    vec2 p = vec2(vs_uv.x * u_aspect, vs_uv.y);
    vec3 col = vec3(0.0);

    for (int i = 0; i < u_spark_count; i++) {
        vec2 c = spark_pos(i, u_time);
        float d = length(p - vec2(c.x * u_aspect, c.y));
        // A tight centre over a soft falloff: the falloff is what the bloom picks up,
        // the centre is what keeps the spark reading as a point rather than a blob.
        float core = smoothstep(u_spark_size, 0.0, d)
                   + 2.0 * smoothstep(u_spark_size * 0.28, 0.0, d);

        // Each spark takes its own hue from the palette.
        float h = float(i) / max(1.0, float(u_spark_count));
        vec3 tint = 0.6 + 0.4 * cos(6.2831 * (h + vec3(0.0, 0.33, 0.67)));
        col += tint * core * u_glow;
    }
    o = vec4(col, 1.0);
}

// --- step 2: keep only what is brighter than the threshold. Half resolution. --------
void step_bright(out vec4 o) {
    vec3 c = texture(u_sparks, vs_uv).rgb;
    o = vec4(max(c - u_threshold, 0.0), 1.0);
}

// --- step 3: widen it. A quarter-res linear target does most of the work for free. --
void step_blur(out vec4 o) {
    vec2 t = 1.5 / vec2(textureSize(u_bright, 0));
    vec3 s = vec3(0.0);
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            s += texture(u_bright, vs_uv + vec2(x, y) * t).rgb;
        }
    }
    o = vec4(s / 25.0, 1.0);
}

// --- step 4: FEEDBACK. Reading u_trail inside step_trail gives last frame. ----------
void step_trail(out vec4 o) {
    // Decay last frame, then re-seed from the sparks. Turn u_fade down and the tails
    // shorten; turn it up towards 1.0 and they persist for seconds.
    vec3 prev = texture(u_trail, vs_uv).rgb * u_fade;
    o = vec4(max(texture(u_sparks, vs_uv).rgb, prev), 1.0);
}

// --- the final draw: main() composes the chain. -------------------------------------
void main() {
    // The trail carries the history, the sparks layer their crisp heads back on top,
    // and the blur adds the glow around both.
    vec3 c = texture(u_trail, vs_uv).rgb
           + texture(u_sparks, vs_uv).rgb
           + texture(u_blur, vs_uv).rgb * u_bloom;
    c = c / (c + 1.0);                       // tonemap: the targets hold values > 1
    c = pow(c, vec3(1.0 / 2.2));             // to sRGB
    fs_color = vec4(c, 1.0);
}
