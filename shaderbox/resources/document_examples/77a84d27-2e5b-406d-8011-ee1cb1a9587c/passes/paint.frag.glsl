#version 460 core

// PAINT -- the drawable canvas, and the only pass that remembers anything.
//
// It reads ITSELF (u_prev), so whatever was here last frame is still here this frame: the graph
// calls that feedback, and it is what makes a canvas accumulate instead of flashing. The script
// beside this document (scripts/script.py) feeds the cursor in through u_brush, so dragging the
// mouse over the preview lays down light or walls.
//
// RGB is emitted colour, ALPHA is "this texel is solid". A wall is opaque and black; a light is
// opaque and bright. Everything the rest of the document does keys off that alpha.

in vec2 vs_uv;
out vec4 fs_color;

uniform sampler2D u_prev;
uniform vec4 u_brush;      // xy = cursor in UV, z = radius, w = 0 off / 1 painting
uniform vec3 u_brush_color;
uniform float u_brush_emissive;  // 1 = a light, 0 = a wall
uniform float u_clear;

// Squared distance from p to the segment a-b. Squared, so no sqrt in the inner test -- the
// article's sdfLineSquared. A segment rather than a dot because a fast drag skips pixels
// between frames, and a dot brush would lay down beads instead of a stroke.
float sd_seg_sq(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float len_sq = max(dot(ba, ba), 1e-9);
    float t = clamp(dot(pa, ba) / len_sq, 0.0, 1.0);
    vec2 d = pa - ba * t;
    return dot(d, d);
}

void main() {
    vec4 current = texture(u_prev, vs_uv);
    if (u_clear > 0.5) {
        fs_color = vec4(0.0);
        return;
    }
    if (u_brush.w > 0.5) {
        // The script hands over one point per frame; the segment is that point to itself until
        // the script starts sending the previous one too. Radius is in UV.
        float r = u_brush.z;
        if (sd_seg_sq(vs_uv, u_brush.xy, u_brush.xy) <= r * r) {
            current = vec4(u_brush_color * u_brush_emissive, 1.0);
        }
    }
    fs_color = current;
}
