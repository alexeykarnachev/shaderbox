# Run: cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run --with python-xlib python ai_docs/features/090_render_decoupling/probes/prio_desktop_x11.py
"""Why does the desktop stay smooth while ShaderBox renders a heavy document?

mutter on an X11 session composites through GLX, not EGL (meta-renderer-x11.c picks
_cogl_winsys_glx_get_vtable for COGL_DRIVER_GL3 when not a Wayland compositor), so cogl's
EGL_CONTEXT_PRIORITY_HIGH_IMG request never runs here -- the compositor's smoothness cannot be
the priority extension. This probe measures the other candidate: a CPU-drawn X client, which
never enters the GPU's draw queue at all. It pushes a full-window image with XPutImage + XSync
and records the period, first with the GPU idle and then against the 100 ms document.

A terminal redrawing text is exactly this shape, which is why vim stays responsive.
"""

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import fmt, now
from prio_guard import wait_for_stable
from Xlib import X, display

W, H = 640, 360
DUR = 5.0
HERE = Path(__file__).parent


def measure(label: str) -> None:
    d = display.Display()
    scr = d.screen()
    win = scr.root.create_window(
        100, 60, W, H, 0, scr.root_depth, X.InputOutput, X.CopyFromParent,
        background_pixel=scr.black_pixel, event_mask=X.ExposureMask,
    )
    win.map()
    d.sync()
    gc = win.create_gc(foreground=scr.white_pixel, background=scr.black_pixel)

    # One PutImage carries the whole window in a real client, but python-xlib packs the request
    # length into a 16-bit field, so a 640x360x4 image (921 KB) overflows it with
    # "struct.error: 'H' format requires 0 <= number <= 65535". Sending horizontal bands is what
    # a toolkit does anyway, and the whole window still lands before each XSync.
    band = 16
    frames: list[bytes] = [bytes([(i * 7 + x) % 256 for x in range(W * band * 4)]) for i in range(3)]
    periods: list[float] = []
    roundtrips: list[float] = []
    t_end = now() + DUR
    prev = now()
    k = 0
    while now() < t_end:
        buf = frames[k % 3]
        for y in range(0, H, band):
            win.put_image(gc, 0, y, W, band, X.ZPixmap, scr.root_depth, 0, buf)
        d.sync()
        t = now()
        rt0 = now()
        d.get_input_focus()  # a pure X round-trip: no drawing, no compositor involvement
        roundtrips.append((now() - rt0) * 1000.0)
        periods.append((t - prev) * 1000.0)
        prev = t
        k += 1
    body = periods[2:]
    misses = sum(1 for p in body if p > 1000.0 / 60.0)
    print(f"  {fmt(label, body)}  over-16.7ms {misses}/{len(body)}", flush=True)
    print(f"    {fmt('X round-trip', roundtrips[2:])}", flush=True)
    win.destroy()
    d.close()


def main() -> None:
    wait_for_stable()
    for trial in (1, 2):
        measure(f"X11 CPU window, GPU idle      t{trial}")
        heavy = subprocess.Popen(
            [sys.executable, str(HERE / "prio_heavy_proc.py"), "MEDIUM", "12", "1"],
            stdout=subprocess.PIPE, text=True,
        )
        assert heavy.stdout is not None
        print("  " + heavy.stdout.readline().strip(), flush=True)
        time.sleep(1.5)
        measure(f"X11 CPU window, 100ms document t{trial}")
        heavy.wait(timeout=40)


main()
