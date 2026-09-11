"""090 BEFORE baseline, deliverable 3: editor input latency under the heavy document vs the
trivial document, isolating the frame-period effect from the X11/ibus input-method effect
(established separately: with ibus, keys drain one per poll when the frame is slower than
30 ms — this probe never touches glfw's char callback or X11 at all, it injects a KeyEvent
directly into app.editor_key_events, which is where hotkeys.py's _drain_editor_input reads
from every frame).

Focuses the editor the way tests/test_code_panel.py does: a real mouse click into
app.editor_rect through imgui's io, driven through actual update_and_draw frames (not a
raw attribute poke) so the editor_focused flip is the real one imgui computes. Then, per
sample: set the cursor to a known line, note wall-clock t0, append one KeyEvent('j') to
app.editor_key_events (translate_char('j'), matching shaderbox/editor/input.py's char-
callback path), and drive update_and_draw frames until the cursor's line advances by one,
recording the wall delay. 50 samples per document.

Usage: `uv run python ai_docs/features/090_render_decoupling/probes/02_input_latency.py <project_dir>`
"""

import json
import sys
import time
from pathlib import Path

from imgui_bundle import imgui

from shaderbox.app import App
from shaderbox.editor.input import translate_char
from shaderbox.ui import update_and_draw

N_SAMPLES = 50
MAX_FRAMES_WAIT = 120  # generous upper bound so a stalled sample fails loud, not silent


def _frames(app: App, n: int) -> None:
    for _ in range(n):
        update_and_draw(app)


def _focus_editor_by_click(app: App) -> None:
    io = imgui.get_io()
    _frames(app, 3)
    x, y, w, h = app.editor_rect
    io.add_mouse_pos_event(x + w * 0.5, y + h * 0.5)
    _frames(app, 2)
    io.add_mouse_button_event(0, True)
    _frames(app, 2)
    io.add_mouse_button_event(0, False)
    _frames(app, 3)
    assert app.editor_focused, f"editor did not focus after a click (rect={app.editor_rect})"


def _ensure_wide_buffer(app: App, min_lines: int = 200) -> None:
    # A single sample moves the cursor one line via 'j'; a buffer with headroom means the
    # 50-sample loop never runs off the bottom (vim's 'j' is a no-op on the last line).
    session = app.get_current_session_if_exists()
    assert session is not None
    editor = session.editor
    text = editor.get_text()
    lines = text.split("\n")
    if len(lines) < min_lines:
        pad = "\n".join(f"// pad {i}" for i in range(min_lines - len(lines)))
        last_line = len(lines) - 1
        editor.set_selection((last_line, len(lines[-1])), (last_line, len(lines[-1])))
        editor.replace_selection("\n" + pad)


def measure_document(app: App, document_id: str, label: str) -> dict:
    app.app_state.is_render_all_documents = False
    app.set_current_document_id(document_id)
    app.ensure_shader_tab(document_id)
    _frames(app, 3)
    _focus_editor_by_click(app)
    session = app.get_current_session_if_exists()
    assert session is not None
    editor = session.editor
    _ensure_wide_buffer(app, min_lines=N_SAMPLES + 20)

    delays_ms: list[float] = []
    frame_counts: list[int] = []
    for i in range(N_SAMPLES):
        editor.set_cursor(i, 0)
        _frames(app, 1)  # let the cursor-set land before injecting the key
        before_line = editor.get_current_cursor_position().line
        event = translate_char(ord("j"))
        t0 = time.perf_counter()
        app.editor_key_events.append(event)
        moved = False
        n_frames = 0
        for f in range(MAX_FRAMES_WAIT):
            update_and_draw(app)
            n_frames = f + 1
            if editor.get_current_cursor_position().line != before_line:
                moved = True
                break
        delay_ms = (time.perf_counter() - t0) * 1000.0
        assert moved, (
            f"{label} sample {i}: cursor line did not change within "
            f"{MAX_FRAMES_WAIT} frames after injecting 'j'"
        )
        delays_ms.append(delay_ms)
        frame_counts.append(n_frames)

    delays_ms.sort()

    def pct(p: float) -> float:
        idx = min(len(delays_ms) - 1, int(round(p * (len(delays_ms) - 1))))
        return delays_ms[idx]

    return {
        "label": label,
        "n_samples": len(delays_ms),
        "delay_median_ms": pct(0.5),
        "delay_p95_ms": pct(0.95),
        "delay_max_ms": max(delays_ms),
        "delay_min_ms": min(delays_ms),
        "frames_median": sorted(frame_counts)[len(frame_counts) // 2],
        "frames_max": max(frame_counts),
        "all_delays_ms": delays_ms,
    }


def main() -> None:
    project_dir = Path(sys.argv[1])
    app = App(project_dir=project_dir, headless=True)
    results = []
    try:
        results.append(measure_document(app, "heavy", "heavy"))
        results.append(measure_document(app, "trivial", "trivial"))
    finally:
        app.release()
    out_path = Path(sys.argv[0]).parent / "results_input_latency.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    for r in results:
        print(
            f"{r['label']:>10}  median={r['delay_median_ms']:.2f}ms  "
            f"p95={r['delay_p95_ms']:.2f}ms  max={r['delay_max_ms']:.2f}ms  "
            f"frames_median={r['frames_median']}  frames_max={r['frames_max']}"
        )


if __name__ == "__main__":
    main()
