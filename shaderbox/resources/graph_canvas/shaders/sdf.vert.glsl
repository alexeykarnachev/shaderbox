// One unit quad, instanced. Every shape's parameters arrive ONCE per instance
// rather than repeated across six vertices, which is what takes the upload from
// 432 bytes per shape to 72.

layout(location = ATTR_CORNER) in vec2 a_corner;
layout(location = ATTR_RECT) in vec4 a_rect;
layout(location = ATTR_FILL_TOP) in vec4 a_fill_top;
layout(location = ATTR_SHAPE) in vec4 a_shape;
layout(location = ATTR_FILL_BOT) in vec4 a_fill_bot;
layout(location = ATTR_EDGE) in vec4 a_edge;
// cos/sin of the shape's rotation. (1,0) is unrotated, which is what every
// shape that never rotates uploads.
layout(location = ATTR_ROTATION) in vec2 a_rotation;
// The field's box relative to the quad: scale in xy, centre offset in zw, both
// as multiples of the quad's own size. (1, 1, 0, 0) evaluates the field over
// exactly the quad, which is what every shape but a selectively rounded box
// uploads.
layout(location = ATTR_FIELD) in vec4 a_field;
layout(location = ATTR_UV) in vec4 a_uv;
// The border band's own colour. A zero alpha defers to the luminance in
// a_edge.z, which is what every shape that never asked for one sends.
layout(location = ATTR_BORDER) in vec4 a_border;

uniform mat4 u_mvp;

out vec2 v_local;
out vec4 v_fill_top;
out vec4 v_shape;
out vec4 v_fill_bot;
out vec4 v_edge;
out vec2 v_uv;
out float v_textured;
out vec4 v_border;

void main() {
    // Pixels from the QUAD's centre. This is the geometry -- what gets
    // rasterised -- and it is the rect the caller asked for, always.
    vec2 local = (a_corner - 0.5) * a_rect.zw;

    // Pixels from the FIELD's centre, which is what the fragment shader
    // measures its distance against. The two differ only where a shape wants
    // its field evaluated over a larger box than it paints: a box rounded at
    // the top alone is the upper half of a fully rounded box twice as tall, so
    // it asks for a field of double height centred half a box lower and lets
    // the quad show only the half that wanted rounding.
    //
    // KEEPING THEM SEPARATE IS THE POINT. Fused -- v_local computed from the
    // quad -- the only way to get a square bottom corner was to make the QUAD
    // twice as tall too, which painted a whole extra box of fill past the rect
    // the caller named.
    //
    // Still computed BEFORE the rotation, so the fragment shader evaluates an
    // axis-aligned box: rotating the quad and leaving the field alone is what
    // lets one primitive serve both a panel and a wire segment.
    v_local = local - a_field.zw * a_rect.zw;

    vec2 center = a_rect.xy + a_rect.zw * 0.5;
    vec2 rotated = vec2(
        local.x * a_rotation.x - local.y * a_rotation.y,
        local.x * a_rotation.y + local.y * a_rotation.x
    );
    vec2 pos = center + rotated;
    v_fill_top = a_fill_top;
    v_shape = a_shape;
    v_fill_bot = a_fill_bot;
    v_edge = a_edge;
    v_border = a_border;
    // Addressed by the QUAD's own unit coordinate, not by the field's: a
    // selectively-rounded instance evaluates its field over a larger box, and
    // a UV taken from that would slide the image off the rect it paints.
    v_uv = mix(a_uv.xy, a_uv.zw, a_corner);
    v_textured = ((a_uv.z - a_uv.x) * (a_uv.w - a_uv.y) != 0.0) ? 1.0 : 0.0;
    gl_Position = u_mvp * vec4(pos, 0.0, 1.0);
}
