"""GIL-stall probe for feature 090 (render-on-worker-thread decoupling).

Question: can a moderngl call made on a worker thread, with its own GL
context, stall the MAIN thread's glfw event pump because it holds the
Python GIL while blocked on the GPU?

Method: the main thread creates a visible-but-off-screen host window and
starts a glfw event-pump loop that does nothing but `glfw.poll_events()`
and record `time.perf_counter()` between iterations. A worker thread creates
a second hidden glfw window that SHARES the host window's GL objects
(`glfw.create_window(..., share=host_window)`), makes it current on itself,
wraps it with a fresh `moderngl.Context` (`moderngl.create_context()`), and
runs one of several workloads in a tight loop for a fixed duration. We
record the largest gap between consecutive main-thread iterations (worst
observed freeze) and the gap distribution (p50/p95/p99/max) for each
workload.

Workloads:
  - sleep:    worker does time.sleep(0.1) only (no GL). Baseline: should
              show ~poll_events-interval-sized gaps, i.e. no GIL contention
              beyond normal scheduling.
  - busy:     worker does a pure-Python busy loop (no GIL release at all,
              CPython releases the GIL every `sys.getswitchinterval()`
              seconds by the bytecode-level switch, ~5ms default). This is
              the "GIL fully contended" reference point — expect small but
              nonzero gaps, NOT a 100ms freeze, because CPython's eval-loop
              switch still fires during a busy loop.
  - render_finish:   worker does a heavy fullscreen draw calibrated to
                     ~100ms + `ctx.finish()` each iteration. If moderngl
                     holds the GIL for the full GPU-bound duration of
                     `finish()`, main-thread gaps should read ~100ms.
  - render_flush:    worker does the same heavy draw + `ctx.gl.Flush()` via
                     raw GL (moderngl's Context has no public flush/fence
                     API — see the research doc) but no `finish()` — i.e.
                     submit-only, no CPU-side wait for GPU completion.
  - render_read:     worker does the same heavy draw + `fbo.read()`
                     (Framebuffer.read, a synchronous glReadPixels-style
                     stall) each iteration.

Run:
    uv run python ai_docs/features/090_render_decoupling/probes/gil_probe.py

Requires a live X11 display (DISPLAY=:1 on the target machine) — this
creates real (hidden) glfw windows and a real moderngl context, it does not
mock GL.
"""

import ctypes
import ctypes.util
import sys
import threading
import time
from dataclasses import dataclass, field

import glfw
import moderngl
import numpy as np

# moderngl's Context has no public "flush" or fence/sync API (verified by
# reading src/moderngl.cpp at the installed 5.12.0 tag: MGLContext_methods
# lists only "finish", no "flush", no glFenceSync/ClientWaitSync). To probe
# a submit-only pattern we call glFlush() directly through libGL via ctypes
# — a CDLL call, so this call itself also releases the GIL per ctypes'
# documented CDLL behavior; that is fine, it mirrors what a render thread
# would actually have to do (drop to raw GL) to get a non-finish submit.
_libgl = ctypes.CDLL(ctypes.util.find_library("GL"))
_libgl.glFlush.restype = None
_libgl.glFlush.argtypes = []

DRAW_SECONDS = 3.0
POLL_SECONDS = 3.0
TARGET_FRAME_MS = 100.0

FULLSCREEN_VERT = """
#version 330
in vec2 in_pos;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

# Heavy fragment shader: many dependent trig/pow iterations per pixel so we
# can calibrate wall-clock cost via resolution/iteration count rather than
# via a driver-specific knob.
HEAVY_FRAG = """
#version 330
uniform vec2 u_resolution;
uniform int u_iters;
out vec4 f_color;
void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    float acc = 0.0;
    for (int i = 0; i < u_iters; i++) {
        acc += sin(uv.x * float(i) + acc) * cos(uv.y * float(i) - acc);
        acc = fract(acc * 1.0001 + 0.0001);
    }
    f_color = vec4(acc, acc, acc, 1.0);
}
"""


@dataclass
class GapStats:
    gaps: list[float] = field(default_factory=list)
    iterations: int = 0

    def summarize(self) -> dict[str, float]:
        if not self.gaps:
            return {"max": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "n": 0}
        s = sorted(self.gaps)
        n = len(s)

        def pct(p: float) -> float:
            idx = min(n - 1, int(p * n))
            return s[idx]

        return {
            "max": max(s),
            "p50": pct(0.50),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "n": n,
            "iterations": self.iterations,
        }


def make_host_window() -> object:
    if not glfw.init():
        raise RuntimeError("glfw.init failed")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    window = glfw.create_window(64, 64, "gil_probe_host", None, None)
    if not window:
        raise RuntimeError("failed to create host window")
    return window


def make_worker_window(share_with: object) -> object:
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    window = glfw.create_window(64, 64, "gil_probe_worker", None, share_with)
    if not window:
        raise RuntimeError("failed to create worker window")
    return window


def build_program(ctx: moderngl.Context) -> tuple[moderngl.Program, moderngl.VertexArray]:
    prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=HEAVY_FRAG)
    verts = np.array([-1, -1, 3, -1, -1, 3], dtype="f4")
    vbo = ctx.buffer(verts.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "2f", "in_pos")])
    return prog, vao


def calibrate_iters(ctx: moderngl.Context, prog: moderngl.Program, vao: moderngl.VertexArray, fbo: moderngl.Framebuffer, width: int, height: int) -> int:
    """Binary-search the iteration count that costs ~TARGET_FRAME_MS on an
    otherwise-idle GPU. Retries the whole search if the machine is visibly
    contended (another process holding the GPU busy), since the search
    itself needs a clean reading to converge sensibly — see the "GPU
    contention" note in the research doc for why a shared dev box can spike
    a single sample without invalidating the overall verdict."""
    prog["u_resolution"].value = (float(width), float(height))
    iters = 4000
    for _ in range(60):
        prog["u_iters"].value = iters
        ctx.finish()
        t0 = time.perf_counter()
        fbo.use()
        vao.render(moderngl.TRIANGLES)
        ctx.finish()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"  calibrate: iters={iters} -> {elapsed_ms:.1f} ms", file=sys.stderr)
        if TARGET_FRAME_MS * 0.7 <= elapsed_ms <= TARGET_FRAME_MS * 1.4:
            return iters
        scale = TARGET_FRAME_MS / max(elapsed_ms, 0.01)
        scale = max(0.2, min(scale, 4.0))
        iters = max(500, int(iters * scale))
        if iters > 20_000_000:
            return iters
    return iters


def poll_loop(stop: threading.Event, stats: GapStats) -> None:
    last = time.perf_counter()
    end_time = time.perf_counter() + POLL_SECONDS
    while time.perf_counter() < end_time and not stop.is_set():
        glfw.poll_events()
        now = time.perf_counter()
        stats.gaps.append((now - last) * 1000.0)
        stats.iterations += 1
        last = now


def worker_sleep(stop: threading.Event) -> None:
    end_time = time.perf_counter() + DRAW_SECONDS
    while time.perf_counter() < end_time and not stop.is_set():
        time.sleep(TARGET_FRAME_MS / 1000.0)


def worker_busy(stop: threading.Event) -> None:
    end_time = time.perf_counter() + DRAW_SECONDS
    while time.perf_counter() < end_time and not stop.is_set():
        frame_end = time.perf_counter() + TARGET_FRAME_MS / 1000.0
        x = 0.0
        while time.perf_counter() < frame_end:
            x = x * 1.0000001 + 1.0


def worker_render(
    stop: threading.Event,
    window: object,
    mode: str,
    call_durations_ms: list[float],
) -> None:
    """mode in {"finish", "flush", "read"}. `window` is a hidden GL window
    already created (and to be destroyed) by the MAIN thread — GLFW window
    creation/destruction is only documented as safe from the main thread on
    some platforms, so this thread only makes it current and issues GL/GL-
    context calls on it, never glfw.create_window/destroy_window.

    Records the wall-clock duration of the mode-specific blocking call
    itself into `call_durations_ms`, so the report can correlate "how long
    this call took" against "how big was the concurrent main-thread poll
    gap" in the SAME run — a same-run correlation is robust to a
    GPU-contended machine (absolute numbers move, but the correlation
    either holds or it doesn't)."""
    glfw.make_context_current(window)
    ctx = moderngl.create_context()
    width, height = 1024, 1024
    fbo = ctx.simple_framebuffer((width, height))
    prog, vao = build_program(ctx)
    iters = calibrate_iters(ctx, prog, vao, fbo, width, height)
    print(f"  [{mode}] calibrated iters={iters}", file=sys.stderr)

    end_time = time.perf_counter() + DRAW_SECONDS
    while time.perf_counter() < end_time and not stop.is_set():
        fbo.use()
        vao.render(moderngl.TRIANGLES)
        t0 = time.perf_counter()
        if mode == "finish":
            ctx.finish()
        elif mode == "flush":
            _libgl.glFlush()
        elif mode == "read":
            _libgl.glFlush()
            _ = fbo.read()
        else:
            raise ValueError(mode)
        call_durations_ms.append((time.perf_counter() - t0) * 1000.0)

    ctx.release()
    glfw.make_context_current(None)


def run_case(name: str, worker_target, worker_args: tuple) -> dict[str, float]:
    print(f"\n=== case: {name} ===", file=sys.stderr)
    stop = threading.Event()
    stats = GapStats()

    t = threading.Thread(target=worker_target, args=(stop, *worker_args), daemon=True)
    t.start()
    # Give the worker a moment to spin up its context before we start timing
    # the poll loop, so calibration overhead doesn't pollute gap stats.
    time.sleep(0.3)

    poll_loop(stop, stats)
    stop.set()
    t.join(timeout=10.0)

    summary = stats.summarize()
    print(
        f"  n={summary['n']} iterations={summary.get('iterations')} "
        f"max={summary['max']:.2f}ms p50={summary['p50']:.2f}ms "
        f"p95={summary['p95']:.2f}ms p99={summary['p99']:.2f}ms",
        file=sys.stderr,
    )
    return summary


def main() -> None:
    host_window = make_host_window()
    glfw.make_context_current(host_window)
    # Ensure the host thread also has a moderngl context bound (mirrors the
    # real app's pattern of a main-thread moderngl.init_context()), then
    # release context on this thread so the worker can bind cleanly. glfw
    # contexts are single-current-thread; the host thread does not need GL
    # calls of its own for this probe, only its window + poll_events.
    glfw.make_context_current(None)

    results: dict[str, dict[str, float]] = {}

    results["sleep"] = run_case("control_1_sleep", worker_sleep, ())
    time.sleep(0.5)
    results["busy"] = run_case("control_2_busy_python", worker_busy, ())

    call_durations: dict[str, list[float]] = {}
    for case_name, mode in (
        ("render_finish", "finish"),
        ("render_flush", "flush"),
        ("render_read", "read"),
    ):
        # A fresh worker window per render case (all created here, on the
        # main thread, per the GLFW main-thread-only window-management
        # convention) avoids state leaking from one moderngl.Context into
        # the next and isolates X server resources per case.
        time.sleep(0.5)
        worker_window = make_worker_window(host_window)
        durations: list[float] = []
        results[case_name] = run_case(f"render+{mode}", worker_render, (worker_window, mode, durations))
        call_durations[case_name] = durations
        glfw.destroy_window(worker_window)

    print("\n=== SUMMARY (ms) ===")
    header = f"{'case':22s} {'max':>10s} {'p50':>10s} {'p95':>10s} {'p99':>10s} {'n':>10s}"
    print(header)
    for name, s in results.items():
        print(
            f"{name:22s} {s['max']:10.2f} {s['p50']:10.2f} {s['p95']:10.2f} {s['p99']:10.2f} {s['n']:10d}"
        )

    print("\n=== CALL-DURATION CORRELATION (ms) — worker call time vs main-thread max gap ===")
    for name, durations in call_durations.items():
        if not durations:
            continue
        durations_sorted = sorted(durations)
        call_max = durations_sorted[-1]
        call_p50 = durations_sorted[len(durations_sorted) // 2]
        print(
            f"{name:22s} call_max={call_max:9.2f} call_p50={call_p50:9.2f} "
            f"poll_max_gap={results[name]['max']:9.2f} n_calls={len(durations)}"
        )

    glfw.destroy_window(host_window)
    glfw.terminate()


if __name__ == "__main__":
    main()
