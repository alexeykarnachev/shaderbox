#version 460 core

in vec2 vs_uv;
out vec4 fs_color;

uniform vec3 u_color = vec3(1.0, 1.0, 0.0); // yellow
uniform float u_aspect; // yellow

float sdEquilateralTriangle(vec2 p, float r) {
    const float k = sqrt(3.0);
    p.x = abs(p.x) - r;
    p.y = p.y + r / k;
    if (p.x + k * p.y > 0.0) p = vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0;
    p.x -= clamp(p.x, -r, 0.0);
    return -length(p) * sign(p.y);
}

void main() {
    vec2 uv = vs_uv * 2.0 - 1.0;
    uv.x *= u_aspect;
    
    float d = sdEquilateralTriangle(uv, 0.6);
    float alpha = 1.0 - smoothstep(-0.01, 0.01, d);
    
    vec3 color = u_color * alpha;
    fs_color = vec4(color, alpha);
}