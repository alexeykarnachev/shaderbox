#version 330
out vec4 f;
void step_noise(out vec4 o){ o = vec4(1.0); }
void main(){ vec4 o; step_noise(o); f = o; }
