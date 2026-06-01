#version 460 core

#define TEXT_LEN 16
#define ARR_LEN 4

in vec2 vs_uv;
out vec4 fs_color;

// auto
uniform float u_time;
uniform float u_aspect;

// drag (scalar / vec, non-color names)
uniform float u_drag_float = 0.5;
uniform vec2 u_drag_vec2 = vec2(0.5, 0.5);
uniform vec3 u_drag_vec3 = vec3(0.5, 0.5, 0.5);
uniform vec4 u_drag_vec4 = vec4(0.5, 0.5, 0.5, 1.0);

// color (vec3 / vec4, name ends "color")
uniform vec3 u_color = vec3(1.0, 0.2, 0.1);
uniform vec4 u_tint_color = vec4(0.1, 0.4, 1.0, 1.0);

// texture (sampler2D)
uniform sampler2D u_texture;

// array (float array -> read-only) and uint array (-> array<->text chip)
uniform float u_floats[ARR_LEN];
uniform uint u_uints[ARR_LEN];

// text (uint array, name ends "text")
uniform uint u_label_text[TEXT_LEN];

// buffer (UBO)
layout(std140) uniform u_params {
    vec4 a;
    vec4 b;
} params;

float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float noise(vec2 p){
    vec2 i=floor(p),f=fract(p);
    float a=hash(i),b=hash(i+vec2(1,0)),c=hash(i+vec2(0,1)),d=hash(i+vec2(1));
    vec2 u=f*f*(3.-2.*f);
    return mix(mix(a,b,u.x),mix(c,d,u.x),u.y);
}
float fbm(vec2 p){
    float v=0.,a=.5;
    for(int i=0;i<5;++i){v+=a*noise(p);p*=2.;a*=.5;}
    return v;
}
void main(){
    vec2 uv=vs_uv; uv.x*=u_aspect;
    vec2 p=uv*3.-u_drag_vec2;
    float t=u_time*(.3+u_drag_float*.3);
    p+=vec2(sin(t*.7),cos(t*.5))*u_drag_vec3.xy*.4;
    float w=fbm(p+fbm(p*1.4+t*.2)*.8);
    vec3 col=mix(u_color,u_tint_color.rgb,w);
    col=mix(col,vec3(.9),smoothstep(.55,.75,w));
    col+= .07*hash(uv*300.);

    // extra effects
    // 1. vignette
    float vig = smoothstep(1.1, .3, length(uv));
    col *= vig;

    // 2. subtle RGB shift (chromatic aberration)
    float shift = .008 * fbm(p * 2. + t * .4);
    col.r = mix(col.r, texture(u_texture, vs_uv + vec2(shift)).r, .3);
    col.b = mix(col.b, texture(u_texture, vs_uv - vec2(shift)).b, .3);

    // 3. pulsing ring highlight
    float ring = abs(fract(length(p) * 1.8 - t * 1.5) - .5) * 2.;
    col += .15 * smoothstep(.3, .0, ring) * u_drag_float;

    fs_color = vec4(col, 1.);
}
