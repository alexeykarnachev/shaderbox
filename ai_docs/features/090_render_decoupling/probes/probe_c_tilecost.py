# Run: cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run python ai_docs/features/090_render_decoupling/probes/probe_c_tilecost.py
"""Does splitting the heavy pass into N scissor tiles cost more than the single pass?

Config C only means something if tiling is close to free. Measured with NOTHING else on the
GPU, so any excess is the tiling itself, not contention.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import glfw
import moderngl
from gpu_clear import wait_for_idle
from common import (
    FULLSCREEN_VERT,
    HEAVY_FRAG,
    HEAVY_H,
    HEAVY_ITERS,
    HEAVY_W,
    fullscreen_quad,
    make_window,
    now,
)


def main() -> None:
    wait_for_idle()
    if not glfw.init():
        raise RuntimeError("glfw.init failed")
    win = make_window(64, 64, "tilecost", visible=False)
    glfw.make_context_current(win)
    ctx = moderngl.create_context()
    prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=HEAVY_FRAG)
    prog["u_iters"].value = HEAVY_ITERS
    vao = fullscreen_quad(ctx, prog)
    tex = ctx.texture((HEAVY_W, HEAVY_H), 4, dtype="f1")
    fbo = ctx.framebuffer(color_attachments=[tex])
    fbo.use()

    for tiles in (1, 4, 16):
        side = int(tiles**0.5)
        tw, th = HEAVY_W // side, HEAVY_H // side
        samples = []
        for k in range(6):
            prog["u_time"].value = float(k)
            t0 = now()
            if side == 1:
                ctx.scissor = None
                vao.render()
            else:
                for ty in range(side):
                    for tx in range(side):
                        ctx.scissor = (tx * tw, ty * th, tw, th)
                        vao.render()
                ctx.scissor = None
            ctx.finish()
            samples.append((now() - t0) * 1000.0)
        samples.sort()
        med = samples[len(samples) // 2]
        print(
            f"tiles={tiles:>3} (scissor)  median={med:8.2f} ms  "
            f"all={[round(s, 1) for s in samples]}",
            flush=True,
        )

    # Same split, but each tile also shrinks the drawn quad so vertex-stage culling, not just
    # scissor, limits the fragments. Distinguishes "scissor is free" from "scissor is not".
    import numpy as np

    for tiles in (4, 16):
        side = int(tiles**0.5)
        samples = []
        vaos = []
        for ty in range(side):
            for tx in range(side):
                x0 = -1.0 + 2.0 * tx / side
                x1 = -1.0 + 2.0 * (tx + 1) / side
                y0 = -1.0 + 2.0 * ty / side
                y1 = -1.0 + 2.0 * (ty + 1) / side
                v = np.array([x0, y0, x1, y0, x1, y1, x0, y0, x1, y1, x0, y1], dtype="f4")
                vaos.append(ctx.vertex_array(prog, [(ctx.buffer(v.tobytes()), "2f", "in_pos")]))
        for k in range(6):
            prog["u_time"].value = float(k)
            t0 = now()
            for v in vaos:
                v.render()
            ctx.finish()
            samples.append((now() - t0) * 1000.0)
        samples.sort()
        med = samples[len(samples) // 2]
        print(
            f"tiles={tiles:>3} (geometry) median={med:8.2f} ms  "
            f"all={[round(s, 1) for s in samples]}",
            flush=True,
        )

    glfw.destroy_window(win)
    glfw.terminate()


main()
