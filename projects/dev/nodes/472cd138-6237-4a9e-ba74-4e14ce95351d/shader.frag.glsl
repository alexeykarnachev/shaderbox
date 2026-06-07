#version 460 core

in vec2 vs_uv; // Coordinate of the current pixel to be shaded

uniform float u_time;   // Time (s) since the application started
uniform float u_aspect; // Aspect ratio of the canvas (width / height)
// uniform vec2 u_resolution;  // Resolution of the canvas (width, height)

out vec4 fs_color;

void main() {
    vec2 uv = vs_uv * 2.0 - 1.0;
    uv.x *= u_aspect;

    float r2 = dot(uv, uv);
    if (r2 > 1.0) {
        fs_color = vec4(0.0);
        return;
    }

          float z = sqrt(1.0 - r2);
      vec3 p = vec3(uv, z);

      // rotate the sphere around Y axis
      float rot = u_time * 0.7;
      float c = cos(rot), s = sin(rot);
      p = vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);

      // spherical angles
      float phi = atan(p.y, p.x);
      float theta = acos(clamp(p.z, -1.0, 1.0));

      float r = theta;           // use polar angle as radial input
      float a = phi;

      float t = u_time * 0.8;
      float spiral1 = sin(6.0 * a + 10.0 * r - 5.0 * t);
      float spiral2 = sin(9.0 * a - 14.0 * r + 3.5 * t);
      float spiral3 = sin(22.0 * a + 31.0 * r - 18.0 * t) * 0.3;
      float spiral4 = sin(47.0 * a - 52.0 * r + 29.0 * t) * 0.18;

      // FBM now on the sphere surface
      float noise = SB_fbm(vec2(phi, theta) * 13.0 + t * 1.3, 5) * 0.09;
      float spiral = spiral1 * 0.45 + spiral2 * 0.25 + spiral3 + spiral4 + noise;
    vec3 col = mix(vec3(0.05, 0.02, 0.2), vec3(0.4, 0.8, 1.0), smoothstep(-0.005, 0.005, spiral));

    // simple directional shading for volume
    vec3 n = p;
    float diff = max(dot(n, normalize(vec3(0.6, 0.8, 1.0))), 0.0);
    col *= 0.6 + 0.6 * diff;

    fs_color = vec4(col, 1.0);
}
