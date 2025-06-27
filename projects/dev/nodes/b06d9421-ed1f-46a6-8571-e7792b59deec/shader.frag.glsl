#version 460 core

in vec2 vs_uv;

uniform float u_time;
uniform float u_aspect;

uniform vec3 u_grid_size = vec3(16.0, 16.0, 0.0);
layout(std140) uniform CellColors { vec4 u_cell_colors[256]; };

out vec4 fs_color;

void main() {
    uvec2 cell = uvec2(vs_uv * u_grid_size.xy);
    uint cell_idx = uint(cell.y * u_grid_size.x + cell.x);
    vec4 cell_color = u_cell_colors[cell_idx];

    fs_color = cell_color;
}
