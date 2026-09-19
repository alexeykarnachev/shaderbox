in vec2 v_local;
in vec4 v_fill_top;
in vec4 v_shape;
in vec4 v_fill_bot;
in vec4 v_edge;
in vec2 v_uv;
in float v_textured;
uniform sampler2D u_image;

out vec4 finalColor;

// Signed distance to a rounded box: negative inside, zero on the edge.
float sd_round_box(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + r;
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}

void main() {
    vec2 half_size = v_shape.xy;
    float radius = v_shape.z;
    float chamfer = v_shape.w;

    float d = sd_round_box(v_local, half_size, radius);

    // Antialiasing from the field's own screen-space rate of change, so an edge
    // is one pixel wide at every zoom rather than faceted at some and blurred
    // at others.
    float aa = fwidth(d);
    // `softness` widens the edge from a one-pixel antialias into a real
    // falloff, which is how a shadow is drawn: one shape whose field fades out
    // over its spread, rather than a stack of rings approximating a blur.
    float softness = max(v_edge.w, aa);
    float coverage = 1.0 - smoothstep(-softness, softness, d);
    if (coverage <= 0.0) discard;

    // The vertical gradient across the shape.
    float t = clamp((v_local.y + half_size.y) / max(half_size.y * 2.0, 0.001), 0.0, 1.0);
    vec3 fill = mix(v_fill_top.rgb, v_fill_bot.rgb, t);
    float alpha = v_fill_top.a;

    // The chamfer: within `chamfer` of the edge, the surface turns to face
    // outward. Its normal is the field's gradient, and the light comes from
    // above -- so the top of the shape catches it and the bottom loses it.
    //
    // This is what the ring-and-band loop was approximating with geometry.
    if (chamfer > 0.0) {
        float edge_t = clamp(-d / chamfer, 0.0, 1.0);
        if (edge_t < 1.0) {
            // The field's OWN gradient is the outward normal -- an SDF has unit
            // gradient by construction, so it is sampled directly rather than
            // by differencing `d` across the pixel, which measures the screen's
            // scale instead of the surface's orientation and is why the chamfer
            // barely responded to its own strength.
            vec2 e = vec2(0.5, 0.0);
            vec2 n = normalize(
                vec2(
                    sd_round_box(v_local + e.xy, half_size, radius)
                        - sd_round_box(v_local - e.xy, half_size, radius),
                    sd_round_box(v_local + e.yx, half_size, radius)
                        - sd_round_box(v_local - e.yx, half_size, radius)
                ) + vec2(1e-6)
            );
            // -n.y is up in screen space; +1 on the top face, -1 on the bottom.
            float facing = -n.y;
            float k = 1.0 - edge_t;
            float light = v_fill_bot.w;
            float dark = v_edge.x;
            float amount = facing > 0.0 ? light * facing * k : dark * facing * k;
            fill = amount >= 0.0
                ? fill + (vec3(1.0) - fill) * amount
                : fill * (1.0 + amount);
        }
    }


    // An explicit border, drawn as a band just inside the edge.
    float border_w = v_edge.y;
    if (border_w > 0.0) {
        float band = smoothstep(-border_w - aa, -border_w + aa, d);
        fill = mix(fill, vec3(v_edge.z), band);
    }

    // The image replaces the FILL, after the chamfer and the border have run,
    // so a preview keeps its rounded corners, its antialiased edge and its
    // border band and only the flat colour inside them is sampled.
    if (v_textured > 0.5) {
        fill = texture(u_image, v_uv).rgb;
    }

    finalColor = vec4(fill, alpha * coverage);
}
