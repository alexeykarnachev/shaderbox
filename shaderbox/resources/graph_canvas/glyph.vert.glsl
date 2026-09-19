layout(location = ATTR_POSITION) in vec2 a_position;
layout(location = ATTR_TEXCOORD) in vec2 a_texcoord;
layout(location = ATTR_COLOR)    in vec4 a_color;
layout(location = ATTR_UV_BOUNDS) in vec4 a_uv_bounds;

uniform mat4 u_mvp;

out vec2 v_uv;
out vec4 v_color;
flat out vec4 v_uv_bounds;

void main() {
    v_uv = a_texcoord;
    v_color = a_color;
    v_uv_bounds = a_uv_bounds;
    gl_Position = u_mvp * vec4(a_position, 0.0, 1.0);
}
