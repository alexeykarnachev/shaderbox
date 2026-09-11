# Run: cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run python ai_docs/features/090_render_decoupling/probes/throttle_single.py
"""Throttle hitch pattern: today's loop shape (one thread, one context), paced at 60 fps.

Like probe_d_single.py (heavy + light in one loop, one context) crossed with probe_e_paced.py's
pacing discipline (each UI frame sleeps out the rest of the 16.7 ms budget so what is measured is
a DEADLINE, not throughput). The document is rendered only on every k-th UI frame; on the other
frames the UI reuses the last document texture. This is lever 2 from the maintainer's proposal:
throttle a heavy document to a lower fps than the UI, with no thread/context change.

Configurations: k in (1, 2, 3, 6, 12) for the calibrated 100 ms document (HEAVY_ITERS), and
k in (1, 2, 3) for a 30 ms document (a separate iteration count, calibrated below at startup
against the live GPU baseline rather than assumed).

Uses prio_guard's baseline-relative guard: the maintainer's own ShaderBox is running on this box.
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import glfw
import moderngl
from common import (
    FULLSCREEN_VERT,
    HEAVY_FRAG,
    HEAVY_H,
    HEAVY_ITERS,
    HEAVY_W,
    SAMPLER_FRAG,
    fmt,
    fullscreen_quad,
    make_window,
    now,
)
from prio_guard import wait_for_stable

DUR = 5.0
BUDGET_MS = 1000.0 / 60.0
N_FRAMES_TARGET = 300


def find_iters_for_cost(target_ms: float) -> int:
    """Binary-search HEAVY_FRAG's iteration count for a target per-frame cost at 1280x720."""
    win = make_window(64, 64, "calib", visible=False)
    glfw.make_context_current(win)
    ctx = moderngl.create_context()
    prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=HEAVY_FRAG)
    vao = fullscreen_quad(ctx, prog)
    tex = ctx.texture((HEAVY_W, HEAVY_H), 4, dtype="f1")
    fbo = ctx.framebuffer(color_attachments=[tex])
    fbo.use()

    def cost(iters: int) -> float:
        prog["u_iters"].value = iters
        for k in range(2):
            prog["u_time"].value = float(k)
            vao.render()
        ctx.finish()
        samples = []
        for k in range(5):
            prog["u_time"].value = float(k)
            t0 = now()
            vao.render()
            ctx.finish()
            samples.append((now() - t0) * 1000.0)
        samples.sort()
        return samples[len(samples) // 2]

    lo, hi = 100, HEAVY_ITERS
    for _ in range(14):
        mid = (lo + hi) // 2
        c = cost(mid)
        if c < target_ms:
            lo = mid
        else:
            hi = mid
    result_iters = hi
    result_cost = cost(result_iters)
    glfw.destroy_window(win)
    return result_iters, result_cost


def run(win, ctx, doc_prog, doc_vao, doc_fbo, doc_tex, ui_prog, ui_vao, k: int, label: str) -> None:
    periods: list[float] = []
    doc_frame_costs: list[float] = []
    gpu_doc_ms_total = 0.0
    frame = 0
    doc_renders = 0
    t_wall_start = now()
    while frame < N_FRAMES_TARGET + 5:
        t0 = now()
        glfw.poll_events()
        if frame % k == 0:
            doc_fbo.use()
            doc_prog["u_time"].value = frame * 0.01
            td0 = now()
            doc_vao.render()
            ctx.finish()
            doc_frame_costs.append((now() - td0) * 1000.0)
            gpu_doc_ms_total += (now() - td0) * 1000.0
            doc_renders += 1
        ctx.screen.use()
        doc_tex.use(0)
        ui_prog["u_time"].value = frame * 0.01
        ui_vao.render()
        glfw.swap_buffers(win)
        spent = now() - t0
        periods.append(spent * 1000.0)
        if spent < BUDGET_MS / 1000.0:
            time.sleep(BUDGET_MS / 1000.0 - spent)
        frame += 1
    t_wall_total = now() - t_wall_start

    body = periods[5:]
    misses = sum(1 for p in body if p > 16.7)
    doc_fps = doc_renders / t_wall_total
    gpu_share = gpu_doc_ms_total / (t_wall_total * 1000.0)
    st = {
        "median": statistics.median(body),
        "p95": sorted(body)[min(len(body) - 1, int(round(0.95 * (len(body) - 1))))],
        "max": max(body),
    }
    doc_cost_med = statistics.median(doc_frame_costs) if doc_frame_costs else 0.0
    print(
        f"{label:<28} k={k:<3} UI period ms: median={st['median']:6.2f} p95={st['p95']:6.2f} "
        f"max={st['max']:7.2f}  missed>16.7ms={misses}/{len(body)}  "
        f"doc_fps={doc_fps:6.2f}  doc_cost_med={doc_cost_med:7.2f}ms  "
        f"gpu_share_of_wall={gpu_share:5.1%}",
        flush=True,
    )


def run_config(doc_iters: int, doc_cost_label: str, ks: list[int]) -> None:
    win = make_window(HEAVY_W, HEAVY_H, "throttle_single", visible=True)
    glfw.set_window_pos(win, 100, 60)
    glfw.make_context_current(win)
    glfw.swap_interval(0)
    ctx = moderngl.create_context()

    doc_prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=HEAVY_FRAG)
    doc_prog["u_iters"].value = doc_iters
    doc_vao = fullscreen_quad(ctx, doc_prog)
    doc_tex = ctx.texture((HEAVY_W, HEAVY_H), 4, dtype="f1")
    doc_fbo = ctx.framebuffer(color_attachments=[doc_tex])

    ui_prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=SAMPLER_FRAG)
    ui_prog["u_tex"].value = 0
    ui_vao = fullscreen_quad(ctx, ui_prog)

    print(f"--- document {doc_cost_label} (iters={doc_iters}) ---", flush=True)
    for trial in (1, 2):
        for k in ks:
            run(win, ctx, doc_prog, doc_vao, doc_fbo, doc_tex, ui_prog, ui_vao, k, f"  t{trial}")
            time.sleep(0.5)

    glfw.destroy_window(win)


def main() -> None:
    base = wait_for_stable()
    if not glfw.init():
        raise RuntimeError("glfw.init failed")

    iters_30ms, cost_30ms = find_iters_for_cost(30.0)
    print(
        f"[run header] throttle_single.py  GPU baseline {base}%  "
        f"HEAVY_ITERS={HEAVY_ITERS} (~100ms)  30ms_iters={iters_30ms} (measured {cost_30ms:.2f}ms)  "
        f"pace_budget=16.7ms  frames_per_config={N_FRAMES_TARGET}",
        flush=True,
    )

    run_config(HEAVY_ITERS, "~100ms", [1, 2, 3, 6, 12])
    run_config(iters_30ms, "~30ms", [1, 2, 3])

    glfw.terminate()


main()
