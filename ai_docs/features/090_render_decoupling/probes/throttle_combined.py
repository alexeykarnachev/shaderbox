# Run: cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run python ai_docs/features/090_render_decoupling/probes/throttle_combined.py
"""Both levers together: the heavy document rendered at the viewer's derived size, throttled by
the share-of-wall-time policy's k, in today's single-thread/single-context loop paced at 60 fps.

Lever 1 (resolution): the document renders into a VIEWER_W x VIEWER_H target instead of the
calibrated 1280x720, then the UI samples it at the same size (no upscale cost hidden in a
second blit -- the UI pass itself runs at the smaller size, matching how _draw_document_image
would actually present it).

Lever 2 (throttle): k chosen by the share-of-wall-time policy also used in throttle_single.py --
the smallest integer with cost / (k * 16.7ms) <= 0.5 -- computed here from the MEASURED cost of
the heavy shader at the viewer size (not the 1280x720 cost), since that is the cost this
configuration actually pays per document frame.

Uses prio_guard's baseline-relative guard: the maintainer's own ShaderBox is running on this box.
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import glfw
import moderngl
from common import FULLSCREEN_VERT, HEAVY_FRAG, HEAVY_ITERS, SAMPLER_FRAG, fullscreen_quad, make_window, now
from prio_guard import wait_for_stable

BUDGET_MS = 1000.0 / 60.0
N_FRAMES_TARGET = 300

# Derived in the report from _draw_document_image's aspect-fit math at a 1920x1080 window with
# the default 50/50 editor/app split (ui_models.editor_split_fraction) and PANEL_CTRL_MINH=600.
VIEWER_W, VIEWER_H = 764, 430


def measure_doc_cost(ctx: moderngl.Context, prog, vao, w: int, h: int) -> float:
    tex = ctx.texture((w, h), 4, dtype="f1")
    fbo = ctx.framebuffer(color_attachments=[tex])
    fbo.use()
    ctx.viewport = (0, 0, w, h)
    for k in range(3):
        prog["u_time"].value = float(k)
        vao.render()
    ctx.finish()
    samples = []
    for k in range(7):
        prog["u_time"].value = float(k)
        t0 = now()
        vao.render()
        ctx.finish()
        samples.append((now() - t0) * 1000.0)
    fbo.release()
    tex.release()
    return statistics.median(samples)


def share_policy_k(cost_ms: float) -> int:
    k = 1
    while cost_ms / (k * 16.7) > 0.5:
        k += 1
    return k


def run(win, ctx, doc_prog, doc_vao, ui_prog, ui_vao, w: int, h: int, k: int, label: str) -> None:
    doc_tex = ctx.texture((w, h), 4, dtype="f1")
    doc_fbo = ctx.framebuffer(color_attachments=[doc_tex])

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
            ctx.viewport = (0, 0, w, h)
            doc_prog["u_time"].value = frame * 0.01
            td0 = now()
            doc_vao.render()
            ctx.finish()
            dt = (now() - td0) * 1000.0
            doc_frame_costs.append(dt)
            gpu_doc_ms_total += dt
            doc_renders += 1
        ctx.screen.use()
        ctx.viewport = (0, 0, 1280, 720)
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
    doc_fbo.release()
    doc_tex.release()

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
        f"{label:<10} size={w}x{h} k={k:<3} UI period ms: median={st['median']:6.2f} "
        f"p95={st['p95']:6.2f} max={st['max']:7.2f}  missed>16.7ms={misses}/{len(body)}  "
        f"doc_fps={doc_fps:6.2f}  doc_cost_med={doc_cost_med:7.2f}ms  "
        f"gpu_share_of_wall={gpu_share:5.1%}",
        flush=True,
    )


def main() -> None:
    base = wait_for_stable()
    if not glfw.init():
        raise RuntimeError("glfw.init failed")

    win = make_window(1280, 720, "throttle_combined", visible=True)
    glfw.set_window_pos(win, 100, 60)
    glfw.make_context_current(win)
    glfw.swap_interval(0)
    ctx = moderngl.create_context()

    doc_prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=HEAVY_FRAG)
    doc_prog["u_iters"].value = HEAVY_ITERS
    doc_vao = fullscreen_quad(ctx, doc_prog)

    ui_prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=SAMPLER_FRAG)
    ui_prog["u_tex"].value = 0
    ui_vao = fullscreen_quad(ctx, ui_prog)

    cost_at_viewer = measure_doc_cost(ctx, doc_prog, doc_vao, VIEWER_W, VIEWER_H)
    k = share_policy_k(cost_at_viewer)

    print(
        f"[run header] throttle_combined.py  GPU baseline {base}%  HEAVY_ITERS={HEAVY_ITERS}  "
        f"viewer_size={VIEWER_W}x{VIEWER_H}  measured_cost_at_viewer={cost_at_viewer:.2f}ms  "
        f"share_policy_k={k}  frames_per_config={N_FRAMES_TARGET}",
        flush=True,
    )

    for trial in (1, 2):
        run(win, ctx, doc_prog, doc_vao, ui_prog, ui_vao, VIEWER_W, VIEWER_H, k, f"combined-t{trial}")
        time.sleep(0.5)

    # Reference row: 1280x720, k=1 (today, no levers) for direct comparison in the same run.
    for trial in (1, 2):
        run(win, ctx, doc_prog, doc_vao, ui_prog, ui_vao, 1280, 720, 1, f"baseline-t{trial}")
        time.sleep(0.5)

    glfw.destroy_window(win)
    glfw.terminate()


main()
