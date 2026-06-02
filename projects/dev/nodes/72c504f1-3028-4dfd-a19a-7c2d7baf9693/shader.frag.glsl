#version 460 core

in vec2 vs_uv;
out vec4 fs_color;

uniform float u_time;
uniform float u_aspect;

float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

vec3 raymarch(vec3 ro, vec3 rd) {
    float t = 0.0;
    for (int i = 0; i < 64; i++) {
        vec3 p = ro + rd * t;
        float d = sdSphere(p, 1.0);
        if (d < 0.001) break;
        t += d;
        if (t > 20.0) return vec3(0.0);
    }
    return ro + rd * t;
}

vec3 getNormal(vec3 p) {
    vec2 e = vec2(0.001, 0.0);
    return normalize(vec3(
        sdSphere(p + e.xyy, 1.0) - sdSphere(p - e.xyy, 1.0),
        sdSphere(p + e.yxy, 1.0) - sdSphere(p - e.yxy, 1.0),
        sdSphere(p + e.yyx, 1.0) - sdSphere(p - e.yyx, 1.0)
    ));
}

void main() {
    vec2 uv = (vs_uv * 2.0 - 1.0);
    uv.x *= u_aspect;
    
    vec3 ro = vec3(0.0, 0.0, -3.0);
    vec3 rd = normalize(vec3(uv, 2.0));
    
    vec3 hit = raymarch(ro, rd);
    float dist = length(hit - ro);
    
    if (dist > 19.0) {
        fs_color = vec4(0.1, 0.1, 0.2, 1.0);
        return;
    }
    
    vec3 normal = getNormal(hit);
    vec3 lightDir = normalize(vec3(1.0, 1.0, -1.0));
    float diff = max(dot(normal, lightDir), 0.0);
    
    vec3 color = vec3(0.2, 0.6, 1.0) * (0.2 + 0.8 * diff);
    fs_color = vec4(color, 1.0);
}