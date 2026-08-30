#version 330
out vec4 f;
in vec2 vs_uv;
uniform sampler2D u_tex;  // stop
void main(){ f = texture(u_tex, vs_uv); }
