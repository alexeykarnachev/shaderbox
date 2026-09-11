# Run: cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run python ai_docs/features/090_render_decoupling/probes/calibrate_heavy.py
"""Find the HEAVY_FRAG iteration count that costs ~100 ms per fullscreen frame at 1280x720."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import glfw
import moderngl
from gpu_clear import wait_for_idle
from common import FULLSCREEN_VERT, HEAVY_FRAG, fullscreen_quad, make_window, now

W, H = 1280, 720


def main() -> None:
    wait_for_idle()
    if not glfw.init():
        raise RuntimeError("glfw.init failed")
    win = make_window(W, H, "calibrate", visible=False)
    glfw.make_context_current(win)
    ctx = moderngl.create_context()
    prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=HEAVY_FRAG)
    vao = fullscreen_quad(ctx, prog)
    tex = ctx.texture((W, H), 4, dtype="f1")
    fbo = ctx.framebuffer(color_attachments=[tex])
    fbo.use()

    print(f"renderer: {ctx.info['GL_RENDERER']}")
    print(f"version:  {ctx.info['GL_VERSION']}")
    print(f"target:   {W}x{H} fullscreen fragment pass")
    print()

    for iters in (100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600):
        prog["u_iters"].value = iters
        # warmup
        for _ in range(2):
            prog["u_time"].value = 0.0
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
        med = samples[len(samples) // 2]
        print(f"iters={iters:>6}  median={med:8.2f} ms  all={[round(s, 2) for s in samples]}")
        if med > 400.0:
            break

    glfw.destroy_window(win)
    glfw.terminate()


main()
