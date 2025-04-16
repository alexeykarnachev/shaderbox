#version 460

in vec2 a_pos;
out vec2 vs_uv;

void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
    vs_uv = a_pos * 0.5 + 0.5;
}
