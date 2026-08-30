#version 330
out vec4 f;
in vec2 vs_uv;
uniform sampler2D u_a;  // stp
void step_a(out vec4 o){ o = vec4(1.0); }
void main(){ f = texture(u_a, vs_uv); }
