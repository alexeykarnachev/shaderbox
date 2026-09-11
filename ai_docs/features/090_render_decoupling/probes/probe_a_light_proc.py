# Run (usually launched by probe_a_run.py, not by hand):
#   cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run python ai_docs/features/090_render_decoupling/probes/probe_a_light_proc.py <seconds> <label>
"""Config A light side: a separate process drawing a trivial frame, recording swap-to-swap period."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import glfw
import moderngl
from common import FULLSCREEN_VERT, TRIVIAL_FRAG, fmt, fullscreen_quad, make_window, now


def main() -> None:
    duration = float(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else "light"
    if not glfw.init():
        raise RuntimeError("glfw.init failed")
    win = make_window(640, 360, "LIGHT", visible=True)
    glfw.set_window_pos(win, 1400, 60)
    glfw.make_context_current(win)
    glfw.swap_interval(0)  # vsync off: measure what the driver gives us, not the 60 Hz cap
    ctx = moderngl.create_context()
    prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=TRIVIAL_FRAG)
    vao = fullscreen_quad(ctx, prog)

    print(f"[light] pid={__import__('os').getpid()} ready", flush=True)
    periods: list[float] = []
    t_end = now() + duration
    prev = now()
    frame = 0
    while now() < t_end:
        glfw.poll_events()
        ctx.screen.use()
        prog["u_time"].value = frame * 0.01
        vao.render()
        glfw.swap_buffers(win)
        t = now()
        periods.append((t - prev) * 1000.0)
        prev = t
        frame += 1
    body = periods[2:]
    print("[light] " + fmt(label, body), flush=True)
    Path(f"/tmp/claude-1000/probe_a_{label}.json").write_text(json.dumps(body))
    glfw.destroy_window(win)
    glfw.terminate()


main()
