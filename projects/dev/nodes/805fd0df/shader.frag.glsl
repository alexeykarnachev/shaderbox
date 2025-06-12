#version 460 core
#define PI 3.141592

in vec2 vs_uv;

uniform float u_time;
uniform float u_aspect;

uniform float u_cam_dist = 0.75;
uniform float u_cam_pow = 0.5;
uniform float u_zoomout = 0.7;
uniform float u_radius_0;

uniform float u_edge_0 = 0.7;
uniform float u_smoothness_0 = 0.05;

uniform vec2 u_center_0 = vec2(0.0, 0.5);
uniform vec2 u_offset = vec2(0.0, 0.0);

uniform sampler2D u_image;
uniform sampler2D u_depth;
uniform sampler2D u_background;

uniform vec3 u_global_tint_color = vec3(1.0, 0.0, 0.0);

out vec4 fs_color;

const vec2 poisson_disk87[87] = vec2[87](vec2(-0.488690, 0.046349),
                                         vec2(0.496064, 0.018367),
                                         vec2(-0.027347, -0.461505),
                                         vec2(-0.090074, 0.490283),
                                         vec2(0.294474, 0.366950),
                                         vec2(0.305608, -0.360041),
                                         vec2(-0.346198, -0.357278),
                                         vec2(-0.308924, 0.353038),
                                         vec2(-0.437547, -0.177748),
                                         vec2(0.446996, -0.129850),
                                         vec2(0.117621, -0.444649),
                                         vec2(0.171424, 0.418258),
                                         vec2(-0.227789, -0.410446),
                                         vec2(0.210264, -0.422608),
                                         vec2(-0.414136, -0.268376),
                                         vec2(0.368202, 0.316549),
                                         vec2(-0.480689, 0.127069),
                                         vec2(0.481128, -0.056358),
                                         vec2(-0.458004, -0.063002),
                                         vec2(0.409361, 0.201972),
                                         vec2(-0.176597, 0.424044),
                                         vec2(-0.095380, -0.441734),
                                         vec2(0.326086, -0.280594),
                                         vec2(-0.411327, 0.184757),
                                         vec2(-0.291534, -0.300406),
                                         vec2(0.400901, -0.002308),
                                         vec2(0.020255, 0.445511),
                                         vec2(0.302251, 0.275637),
                                         vec2(0.387805, -0.223370),
                                         vec2(-0.378395, 0.062614),
                                         vec2(0.405052, 0.101681),
                                         vec2(-0.010340, -0.355322),
                                         vec2(-0.034931, 0.383699),
                                         vec2(-0.318953, -0.225899),
                                         vec2(0.349283, -0.140001),
                                         vec2(-0.253974, 0.299183),
                                         vec2(0.188226, 0.342914),
                                         vec2(0.212083, -0.294545),
                                         vec2(-0.188320, -0.308466),
                                         vec2(-0.373708, -0.070538),
                                         vec2(0.114322, -0.356677),
                                         vec2(-0.154401, 0.348207),
                                         vec2(-0.321713, 0.260043),
                                         vec2(-0.086797, -0.349277),
                                         vec2(-0.360294, -0.144808),
                                         vec2(-0.323996, 0.188199),
                                         vec2(0.277830, -0.204128),
                                         vec2(0.087828, 0.351992),
                                         vec2(-0.215777, -0.234955),
                                         vec2(0.291437, 0.171860),
                                         vec2(0.027249, -0.255925),
                                         vec2(-0.316361, -0.013941),
                                         vec2(0.346679, -0.066942),
                                         vec2(-0.103280, -0.273636),
                                         vec2(-0.017802, 0.310973),
                                         vec2(-0.280809, -0.120043),
                                         vec2(-0.282912, 0.117500),
                                         vec2(0.267574, -0.036973),
                                         vec2(-0.034965, -0.223502),
                                         vec2(0.109677, 0.256372),
                                         vec2(-0.204519, -0.116846),
                                         vec2(0.144105, -0.181736),
                                         vec2(-0.140560, 0.215101),
                                         vec2(0.271573, 0.102406),
                                         vec2(0.220437, 0.203459),
                                         vec2(-0.242979, -0.027494),
                                         vec2(-0.050135, 0.239871),
                                         vec2(-0.152652, -0.193125),
                                         vec2(-0.220532, 0.179600),
                                         vec2(0.216867, -0.096770),
                                         vec2(-0.164884, 0.122109),
                                         vec2(0.251078, 0.034090),
                                         vec2(0.016515, -0.175206),
                                         vec2(0.042304, 0.216117),
                                         vec2(-0.133933, -0.060601),
                                         vec2(0.184659, 0.135680),
                                         vec2(-0.161273, 0.024207),
                                         vec2(-0.056532, -0.154410),
                                         vec2(-0.082706, 0.083129),
                                         vec2(0.081409, -0.088060),
                                         vec2(0.115078, 0.156566),
                                         vec2(0.133209, 0.061211),
                                         vec2(0.002618, -0.101328),
                                         vec2(0.132926, -0.013988),
                                         vec2(-0.027172, -0.017586),
                                         vec2(0.022969, 0.116469),
                                         vec2(0.036262, 0.015085));

// **Helper Functions**

float hash(vec2 p) {
    p = fract(p * vec2(123.45, 678.90));
    p += dot(p, p + vec2(45.67, 89.01));
    return fract(p.x * p.y * 43758.5453);
}

vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }

float mod289(float x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }

vec4 permute(vec4 x) { return mod289(((x * 34.0) + 10.0) * x); }

float permute(float x) { return mod289(((x * 34.0) + 10.0) * x); }

vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float taylorInvSqrt(float r) { return 1.79284291400159 - 0.85373472095314 * r; }

vec4 grad4(float j, vec4 ip) {
    const vec4 ones = vec4(1.0, 1.0, 1.0, -1.0);
    vec4 p, s;
    p.xyz = floor(fract(vec3(j) * ip.xyz) * 7.0) * ip.z - 1.0;
    p.w = 1.5 - dot(abs(p.xyz), ones.xyz);
    s = vec4(lessThan(p, vec4(0.0)));
    p.xyz = p.xyz + (s.xyz * 2.0 - 1.0) * s.www;
    return p;
}

#define F4 0.309016994374947451

float snoise(vec4 v) {
    const vec4 C = vec4(0.138196601125011,
                        0.276393202250021,
                        0.414589803375032,
                        -0.447213595499958);
    vec4 i = floor(v + dot(v, vec4(F4)));
    vec4 x0 = v - i + dot(i, C.xxxx);
    vec4 i0;
    vec3 isX = step(x0.yzw, x0.xxx);
    vec3 isYZ = step(x0.zww, x0.yyz);
    i0.x = isX.x + isX.y + isX.z;
    i0.yzw = 1.0 - isX;
    i0.y += isYZ.x + isYZ.y;
    i0.zw += 1.0 - isYZ.xy;
    i0.z += isYZ.z;
    i0.w += 1.0 - isYZ.z;
    vec4 i3 = clamp(i0, 0.0, 1.0);
    vec4 i2 = clamp(i0 - 1.0, 0.0, 1.0);
    vec4 i1 = clamp(i0 - 2.0, 0.0, 1.0);
    vec4 x1 = x0 - i1 + C.xxxx;
    vec4 x2 = x0 - i2 + C.yyyy;
    vec4 x3 = x0 - i3 + C.zzzz;
    vec4 x4 = x0 + C.wwww;
    i = mod289(i);
    float j0 = permute(permute(permute(permute(i.w) + i.z) + i.y) + i.x);
    vec4 j1 =
        permute(permute(permute(permute(i.w + vec4(i1.w, i2.w, i3.w, 1.0)) +
                                i.z + vec4(i1.z, i2.z, i3.z, 1.0)) +
                        i.y + vec4(i1.y, i2.y, i3.y, 1.0)) +
                i.x + vec4(i1.x, i2.x, i3.x, 1.0));
    vec4 ip = vec4(1.0 / 294.0, 1.0 / 49.0, 1.0 / 7.0, 0.0);
    vec4 p0 = grad4(j0, ip);
    vec4 p1 = grad4(j1.x, ip);
    vec4 p2 = grad4(j1.y, ip);
    vec4 p3 = grad4(j1.z, ip);
    vec4 p4 = grad4(j1.w, ip);
    vec4 norm =
        taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
    p4 *= taylorInvSqrt(dot(p4, p4));
    vec3 m0 = max(0.57 - vec3(dot(x0, x0), dot(x1, x1), dot(x2, x2)), 0.0);
    vec2 m1 = max(0.57 - vec2(dot(x3, x3), dot(x4, x4)), 0.0);
    m0 = m0 * m0;
    m1 = m1 * m1;
    return 60.1 * (dot(m0 * m0, vec3(dot(p0, x0), dot(p1, x1), dot(p2, x2))) +
                   dot(m1 * m1, vec2(dot(p3, x3), dot(p4, x4))));
}

float color_to_brightness(vec3 color) {
    vec3 w = vec3(1.0, 1.0, 1.0);
    return (w.r * color.r + w.g * color.g + w.b * color.b) / (w.r + w.g + w.b);
}

// **Main Function**

void main() {
    // Initial UV transformation and distance to center
    vec2 initial_uv = (vs_uv * u_zoomout) + u_offset;
    float dist_to_center = distance(initial_uv, u_center_0);

    // Screen position setup
    vec3 screen_pos = vec3((initial_uv - 0.5) * vec2(u_aspect, 1.0), 0.0);

    // Camera distance and position
    float camera_dist = u_cam_dist * (1.0 - pow(dist_to_center, u_cam_pow));
    float time_factor = 1.0 / 1.0; // Kept as original, possibly a placeholder
    vec3 camera_pos = vec3(0.025 * sin(time_factor * u_time * 2.0 * PI),
                           0.025 * cos(time_factor * u_time * 2.0 * PI),
                           -camera_dist);

    // Ray marching setup
    vec3 ray_pos = screen_pos;
    vec3 ray_dir = normalize(screen_pos - camera_pos);
    const int max_steps = 1024;
    float step_size = 1.0 / float(max_steps);
    int step_count = 0;
    vec3 hit_color;
    vec2 current_uv;

    // Ray marching loop
    while (true) {
        if (step_count >= max_steps) {
            hit_color = vec3(0.0, 0.0, 0.0);
            break;
        }

        current_uv = ray_pos.xy / vec2(u_aspect, 1.0) + 0.5;
        if (current_uv.x >= 1.0 || current_uv.x <= 0.0 || current_uv.y >= 1.0 ||
            current_uv.y <= 0.0) {
            hit_color = vec3(0.0, 0.0, 0.0);
            break;
        }

        float sampled_depth = texture(u_depth, current_uv).r;
        sampled_depth = clamp(sampled_depth, 0.01, 0.99);
        float depth_diff = ray_pos.z - sampled_depth;

        if (depth_diff >= 0.0) {
            hit_color = texture(u_image, current_uv).rgb;
            break;
        }

        ray_pos += step_size * ray_dir;
        step_count += 1;
    }

    // Set final UV coordinate after ray marching
    vec2 final_uv = current_uv;

    // Discard fragments outside UV bounds
    if (final_uv.x < 0.0 || final_uv.x > 1.0 || final_uv.y < 0.0 ||
        final_uv.y > 1.0) {
        discard;
    }

    // Sample textures at final UV
    vec3 image_color = texture(u_image, final_uv).rgb;
    float depth = texture(u_depth, final_uv).r;
    float background = texture(u_background, final_uv).r;

    // Compute brightness and scene color
    float image_brightness = color_to_brightness(image_color);
    vec3 scene_color = (1.0 - depth) * u_global_tint_color;
    float background_factor = smoothstep(-2.0, 1.0, background);
    background_factor = pow(background_factor, 1.5);
    scene_color *= background_factor;

    // Outline calculation
    const int num_samples = 32;
    float outline_strength = 0.0;
    for (int i = 0; i < num_samples; i++) {
        vec2 sample_uv = final_uv + poisson_disk87[i] * u_radius_0;
        vec3 sample_color = texture(u_image, sample_uv).rgb;
        float sample_brightness = color_to_brightness(sample_color);
        outline_strength +=
            sample_brightness / (image_brightness + sample_brightness);
    }
    outline_strength /= float(num_samples);
    outline_strength = smoothstep(u_edge_0 - 0.5 * u_smoothness_0,
                                  u_edge_0 + 0.5 * u_smoothness_0,
                                  outline_strength);

    // Modulate outline based on distance to center
    float outline_modulator = 1.0 - smoothstep(0.25, 0.5, dist_to_center);
    outline_strength *= outline_modulator;

    // Combine scene and outline colors
    vec3 outline_contribution = vec3(outline_strength);
    vec3 final_color =
        (scene_color + outline_contribution) * 4.0 * pow(1.0 - depth, 8.0);
    final_color = pow(4.0 * final_color, vec3(4.0));

    // Set alpha based on color brightness
    float alpha = 1.0;
    if (final_color.r + final_color.g + final_color.b < 0.05) {
        alpha = 0.0;
    }

    // Output final fragment color
    fs_color = vec4(final_color, alpha);
}
