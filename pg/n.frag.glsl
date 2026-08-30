#version 330
out vec4 f_color;
in vec2 vs_uv;
uniform sampler2D u_acc;  // step, f4, persist
void step_acc(out vec4 o) { o = texture(u_acc, vs_uv) + vec4(1.0,0,0,1); }
void main() { f_color = texture(u_acc, vs_uv); }
