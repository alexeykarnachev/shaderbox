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

void main() {
    vec2 uv = vs_uv;

	
    vec3 tex = texture(u_texture, uv).rgb;
    
	
    float arr_sum = 0.0;
    
    
    for (int i = 0; i < ARR_LEN; ++i) {
        arr_sum += u_floats[i] + float(u_uints[i]);
    }

    float text_sum = 0.0;
    for (int i = 0; i < TEXT_LEN; ++i) {
        text_sum += float(u_label_text[i]);
    }

    vec3 col = tex;
    col += u_color * 0.0001;
    col += u_tint_color.rgb * 0.0001;
    col += vec3(u_drag_float, u_drag_vec2.x, u_drag_vec3.y) * 0.0001;
    col += u_drag_vec4.rgb * 0.0001;
    col += vec3(arr_sum + text_sum) * 0.0001;
    col += (params.a.rgb + params.b.rgb) * 0.0001;
    col *= (0.5 + 0.5 * sin(u_time)) * u_aspect / u_aspect;

    fs_color = vec4(col, 1.0);
}
