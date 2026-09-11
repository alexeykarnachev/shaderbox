# Run: cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run python ai_docs/features/090_render_decoupling/probes/cost_resolution.py
"""Per-frame GPU cost of the calibrated heavy shader and a light shader, across resolutions.

Feeds decision item 1 of the cost_and_throttle research: does rendering the live document at the
DISPLAYED size instead of a fixed 1280x720 buy anything, and is the cost linear in pixel count?

Uses prio_guard's baseline-relative guard, not gpu_clear's fixed-threshold one: the maintainer's
own ShaderBox is running on this box during this measurement and holds the GPU at a nonzero
baseline, so a fixed <=20% guard would never pass. Two trials per configuration, both printed.
"""

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import glfw
import moderngl
from common import FULLSCREEN_VERT, HEAVY_FRAG, HEAVY_ITERS, fullscreen_quad, make_window, now
from prio_guard import wait_for_stable

# The "light" shader: same structural shape as HEAVY_FRAG (hash + sin/cos accumulation loop) so
# resolution scaling is measured on comparable ALU-bound work, just far fewer iterations.
# calibrate_heavy.py's sweep put iters=1600 at 4.90 ms and iters=3200 in between; iters=3200 is
# used here and its actual per-configuration cost is measured (not assumed) at 1280x720 below.
LIGHT_ITERS = 3200

SIZES: list[tuple[int, int, str]] = [
    (3840, 2160, "3840x2160 (4K)"),
    (1920, 1080, "1920x1080"),
    (1280, 720, "1280x720 (calibration size)"),
    (960, 540, "960x540"),
    (640, 360, "640x360"),
    (320, 180, "320x180"),
    # ShaderBox's actual viewer size at a 1920x1080 window, default 50/50 editor split, 16:9
    # document, derived in the report from _draw_document_image + the layout constants.
    (764, 430, "764x430 (derived typical viewer @ 1920x1080 window)"),
]

N_SAMPLES = 9
N_WARMUP = 3


def measure(ctx: moderngl.Context, prog: moderngl.Program, w: int, h: int) -> dict[str, float]:
    tex = ctx.texture((w, h), 4, dtype="f1")
    fbo = ctx.framebuffer(color_attachments=[tex])
    fbo.use()
    ctx.viewport = (0, 0, w, h)
    vao_map = measure._vao_cache  # type: ignore[attr-defined]
    vao = vao_map.setdefault(id(prog), fullscreen_quad(ctx, prog))
    for k in range(N_WARMUP):
        prog["u_time"].value = float(k)
        vao.render()
    ctx.finish()
    samples = []
    for k in range(N_SAMPLES):
        prog["u_time"].value = float(k)
        t0 = now()
        vao.render()
        ctx.finish()
        samples.append((now() - t0) * 1000.0)
    fbo.release()
    tex.release()
    samples.sort()
    return {
        "median": statistics.median(samples),
        "min": samples[0],
        "max": samples[-1],
    }


measure._vao_cache = {}  # type: ignore[attr-defined]


def run_trial(trial: int) -> None:
    win = make_window(64, 64, f"cost_resolution t{trial}", visible=False)
    glfw.make_context_current(win)
    ctx = moderngl.create_context()

    heavy_prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=HEAVY_FRAG)
    heavy_prog["u_iters"].value = HEAVY_ITERS
    light_prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=HEAVY_FRAG)
    light_prog["u_iters"].value = LIGHT_ITERS

    print(f"--- trial {trial} ---", flush=True)
    for shader_label, prog in (("HEAVY iters=35000", heavy_prog), ("LIGHT iters=3200", light_prog)):
        print(f"  {shader_label}", flush=True)
        for w, h, label in SIZES:
            st = measure(ctx, prog, w, h)
            mpix = (w * h) / 1_000_000.0
            per_mpix = st["median"] / mpix if mpix > 0 else float("nan")
            print(
                f"    {label:<48} median={st['median']:8.3f} ms  "
                f"min={st['min']:8.3f}  max={st['max']:8.3f}  "
                f"mpix={mpix:6.3f}  ms/mpix={per_mpix:7.3f}",
                flush=True,
            )

    glfw.destroy_window(win)


def main() -> None:
    base = wait_for_stable()
    print(f"[run header] cost_resolution.py  GPU baseline {base}%  HEAVY_ITERS={HEAVY_ITERS} "
          f"LIGHT_ITERS={LIGHT_ITERS}  samples={N_SAMPLES} warmup={N_WARMUP}", flush=True)
    if not glfw.init():
        raise RuntimeError("glfw.init failed")
    for trial in (1, 2):
        run_trial(trial)
    glfw.terminate()


main()
