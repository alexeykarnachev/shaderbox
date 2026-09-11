"""090 BEFORE baseline, deliverable 5: is the editor's own draw
(`layout_following_cursor` + `EditorPanel` redraw) under 3ms per frame, under real editing
activity (not an idle loop, where `should_redraw`'s gate means `editor:draw` never fires at
all -- see the False trails section of the baseline report).

Focuses the editor exactly as 02_input_latency.py does, then drives 40 real keystrokes ('j',
each moving the cursor and so tripping should_redraw) with the profiler recording, and reports
the max of the `editor` (CPU layout) and `editor:draw` (GPU panel redraw) spans across both the
heavy and trivial documents -- the max, not the median, since the claim is a per-frame BOUND.

Usage: `uv run python ai_docs/features/090_render_decoupling/probes/03_editor_draw_cost.py <project_dir>`
"""

import json
import sys
from pathlib import Path

from imgui_bundle import imgui

from shaderbox.app import App
from shaderbox.editor.input import translate_char
from shaderbox.profiling import Span
from shaderbox.ui import update_and_draw

N_KEYS = 40


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
    assert app.editor_focused


def _walk(span: Span, path: str) -> list[tuple[str, Span]]:
    out: list[tuple[str, Span]] = []
    for child in span.children:
        child_path = f"{path}/{child.name}"
        out.append((child_path, child))
        out.extend(_walk(child, child_path))
    return out


def measure(app: App, document_id: str) -> dict:
    app.app_state.is_render_all_documents = False
    app.set_current_document_id(document_id)
    app.ensure_shader_tab(document_id)
    app.fps_details_open = True
    _frames(app, 3)
    _focus_editor_by_click(app)
    session = app.get_current_session_if_exists()
    assert session is not None
    editor = session.editor
    lines = editor.get_text().split("\n")
    last = len(lines) - 1
    editor.set_selection((last, len(lines[-1])), (last, len(lines[-1])))
    editor.replace_selection("\n" + "\n".join(f"// pad {i}" for i in range(N_KEYS + 10)))

    editor_cpu: list[float] = []
    draw_cpu: list[float] = []
    draw_gpu: list[float] = []
    for i in range(N_KEYS):
        editor.set_cursor(i, 0)
        app.editor_key_events.append(translate_char(ord("j")))
        update_and_draw(app)
        profile = app.last_profile
        if profile is None:
            continue
        for path, span in _walk(profile.root, ""):
            if path.endswith("/editor:draw"):
                draw_cpu.append(span.cpu_ms)
                if span.gpu_ms is not None:
                    draw_gpu.append(span.gpu_ms)
            elif path.endswith("/editor"):
                editor_cpu.append(span.cpu_ms)

    return {
        "document": document_id,
        "editor_layout_cpu_max_ms": max(editor_cpu) if editor_cpu else None,
        "editor_draw_gpu_samples": len(draw_gpu),
        "editor_draw_gpu_max_ms": max(draw_gpu) if draw_gpu else None,
        "editor_draw_cpu_wrapper_max_ms": max(draw_cpu) if draw_cpu else None,
    }


def main() -> None:
    project_dir = Path(sys.argv[1])
    app = App(project_dir=project_dir, headless=True)
    results = []
    try:
        results.append(measure(app, "heavy"))
        results.append(measure(app, "trivial"))
    finally:
        app.release()
    out_path = Path(sys.argv[0]).parent / "results_editor_draw_cost.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    for r in results:
        print(
            f"{r['document']:>10}  layout_cpu_max={r['editor_layout_cpu_max_ms']:.3f}ms  "
            f"draw_gpu_max={r['editor_draw_gpu_max_ms']:.3f}ms "
            f"(n={r['editor_draw_gpu_samples']})"
        )


if __name__ == "__main__":
    main()
