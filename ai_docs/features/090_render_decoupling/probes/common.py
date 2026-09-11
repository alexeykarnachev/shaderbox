# Shared helpers for the 090 GPU-preemption probes. Not run directly; imported by probe_*.py.
import statistics
import time

import glfw
import moderngl

HEAVY_FRAG = """
#version 330
uniform int u_iters;
uniform float u_time;
out vec4 f_color;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

void main() {
    vec2 uv = gl_FragCoord.xy * 0.001;
    float acc = 0.0;
    for (int i = 0; i < u_iters; i++) {
        float fi = float(i);
        vec2 q = uv * (1.0 + fi * 0.017) + vec2(fi * 0.31, u_time * 0.07);
        acc += sin(q.x * 13.0 + acc) * cos(q.y * 7.0 - acc) * hash(floor(q * 64.0));
        acc = fract(acc * 1.0001 + 0.0003);
    }
    f_color = vec4(acc, fract(acc * 3.0), fract(acc * 7.0), 1.0);
}
"""

TRIVIAL_FRAG = """
#version 330
uniform float u_time;
out vec4 f_color;
void main() {
    f_color = vec4(fract(u_time), 0.2, 0.3, 1.0);
}
"""

SAMPLER_FRAG = """
#version 330
uniform sampler2D u_tex;
uniform float u_time;
out vec4 f_color;
void main() {
    vec4 t = texelFetch(u_tex, ivec2(gl_FragCoord.xy) % textureSize(u_tex, 0), 0);
    f_color = vec4(t.rgb * 0.5 + vec3(fract(u_time) * 0.1), 1.0);
}
"""

FULLSCREEN_VERT = """
#version 330
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""


def make_window(width: int, height: int, title: str, visible: bool, share=None):
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)
    glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
    win = glfw.create_window(width, height, title, None, share)
    if not win:
        raise RuntimeError(f"glfw.create_window failed for {title!r}")
    return win


def fullscreen_quad(ctx: moderngl.Context, prog: moderngl.Program) -> moderngl.VertexArray:
    import numpy as np

    verts = np.array([-1, -1, 3, -1, -1, 3], dtype="f4")
    vbo = ctx.buffer(verts.tobytes())
    return ctx.vertex_array(prog, [(vbo, "2f", "in_pos")])


def stats(samples: list[float]) -> dict[str, float]:
    s = sorted(samples)
    if not s:
        return {"n": 0, "median": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "n": len(s),
        "median": statistics.median(s),
        "p95": s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))],
        "max": s[-1],
        "mean": statistics.fmean(s),
    }


def fmt(label: str, samples: list[float]) -> str:
    st = stats(samples)
    return (
        f"{label}: n={st['n']:>5} median={st['median']:7.2f} p95={st['p95']:8.2f} "
        f"max={st['max']:8.2f} mean={st['mean']:7.2f} (ms)"
    )


def now() -> float:
    return time.perf_counter()


# Calibrated on this machine (RTX 3090, driver 580.173.02) by calibrate_heavy.py:
# 35000 iterations of HEAVY_FRAG at 1280x720 == 100.93 ms median per fullscreen pass.
HEAVY_ITERS = 35000
HEAVY_W = 1280
HEAVY_H = 720
