"""Calibration probe for 090's BEFORE baseline.

Finds a noise-loop iteration count whose GPU cost lands near ~8 ms (medium) and ~100 ms
(heavy) on THIS box (RTX 3090, GL 3.3 core via moderngl). Renders a narrow standalone
Document (per dev_flow.md's "Authoring / debugging documents directly" recipe) on a
headless EGL-less context (a real glfw hidden window, since the baseline itself needs a
real window later; this probe reuses the same context style for consistency) and times
`ctx.finish()`-bounded draws directly rather than going through the frame profiler, so the
calibration has no dependency on the thing being calibrated.

Usage: `uv run python ai_docs/features/090_render_decoupling/probes/00_calibrate_shaders.py`
"""

import time

import glfw
import moderngl

_VERTEX = """#version 460 core
in vec2 in_pos;
out vec2 vs_uv;
void main() {
    vs_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_FRAGMENT_TEMPLATE = """#version 460 core
in vec2 vs_uv;
out vec4 fs_color;

float hash(vec2 p) {{
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}}

float noise(vec2 p) {{
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}}

void main() {{
    vec2 uv = vs_uv * 8.0;
    float acc = 0.0;
    float amp = 0.5;
    for (int i = 0; i < {n}; i++) {{
        acc += noise(uv) * amp;
        uv = uv * 2.03 + vec2(37.1, 91.7);
        amp *= 0.998;
    }}
    fs_color = vec4(vec3(acc), 1.0);
}}
"""

_CANVAS = 1024


def build_context() -> tuple[moderngl.Context, object]:
    assert glfw.init(), "glfw.init failed"
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(64, 64, "calibrate", None, None)
    assert window is not None, "glfw.create_window failed"
    glfw.make_context_current(window)
    ctx = moderngl.create_context()
    return ctx, window


def time_n(ctx: moderngl.Context, n: int, samples: int = 20) -> float:
    prog = ctx.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT_TEMPLATE.format(n=n))
    vbo = ctx.buffer(
        data=bytes(
            bytearray(
                __import__("struct").pack(
                    "12f", -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1
                )
            )
        )
    )
    vao = ctx.vertex_array(prog, [(vbo, "2f", "in_pos")])
    target = ctx.texture((_CANVAS, _CANVAS), 4, dtype="f2")
    fbo = ctx.framebuffer(color_attachments=[target])
    fbo.use()

    # Warm-up: compile-time cost and first-draw driver overhead must not pollute the timing.
    vao.render(moderngl.TRIANGLES)
    ctx.finish()

    times: list[float] = []
    for _ in range(samples):
        start = time.perf_counter()
        vao.render(moderngl.TRIANGLES)
        ctx.finish()
        times.append((time.perf_counter() - start) * 1000.0)
    times.sort()
    median = times[len(times) // 2]
    vao.release()
    vbo.release()
    prog.release()
    fbo.release()
    target.release()
    return median


def main() -> None:
    ctx, window = build_context()
    print(f"{'n':>8} {'median ms':>12}")
    # Coarse sweep first to bracket both targets, then refine.
    for n in (50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600):
        ms = time_n(ctx, n, samples=10)
        print(f"{n:>8} {ms:>12.3f}")
    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()
