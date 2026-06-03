#version 460 core
in vec2 vs_uv;
out vec4 fs_color;

uniform float u_time;
uniform float u_aspect;

float sdSphere(vec3 p, float r) { return length(p) - r; }
float sdPlane(vec3 p) { return p.y; }

float map(vec3 p) {
    float s = sdSphere(p - vec3(0.0, 1.0, 0.0), 1.0);
    float plane = sdPlane(p);
    return min(s, plane);
}

vec3 calcNormal(vec3 p) {
    const float eps = 0.001;
    return normalize(vec3(
        map(p + vec3(eps,0,0)) - map(p - vec3(eps,0,0)),
        map(p + vec3(0,eps,0)) - map(p - vec3(0,eps,0)),
        map(p + vec3(0,0,eps)) - map(p - vec3(0,0,eps))
    ));
}

void main() {
    vec2 uv = (vs_uv - 0.5) * vec2(u_aspect, 1.0);
    vec3 ro = vec3(0.0, 2.0, -4.0);
    vec3 rd = normalize(vec3(uv, 1.5));

    float t = 0.0;
    for (int i = 0; i < 64; i++) {
        vec3 p = ro + rd * t;
        float d = map(p);
        t += d;
        if (d < 0.001 || t > 20.0) break;
    }

    vec3 col = vec3(0.1);
    if (t < 20.0) {
        vec3 p = ro + rd * t;
        vec3 n = calcNormal(p);
        vec3 light = normalize(vec3(1.0, 2.0, -1.0));
        float diff = max(dot(n, light), 0.0);
        float shadow = step(map(p + light * 0.1), 0.05) * 0.6 + 0.4; // simple shadow
        col = vec3(0.8) * diff * shadow;
        if (p.y < 0.01) col = vec3(0.2); // ground tint
    }
    fs_color = vec4(col, 1.0);
}