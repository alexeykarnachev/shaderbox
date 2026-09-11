"""090 BEFORE baseline, deliverables 2 and 4: drive the real App (smoke-style) for N frames
with each of the three calibrated documents current, profiler recording enabled, and report
frame period (median/p95/max) + the profiler's per-span breakdown. Also runs a "Render all"
sweep across the three documents in one App instance.

Builds the throwaway project via build_project.py (never touches projects/dev). Follows the
smoke-test shape (scripts/smoke.py): App(project_dir=..., headless=True) drives update_and_draw
directly, no `ui.run` sleep-to-target-fps in the loop -- this measures update_and_draw's own
wall time, which is what 088's profiler roots on and what a render-decoupling gate would assert
against. A separate run with headless=False is diffed against it to see whether a visible
window's real vsync/present changes the swap span or the wall period.

Usage: `uv run python ai_docs/features/090_render_decoupling/probes/01_frame_timing.py <project_dir> [--visible]`
"""

import json
import sys
import time
from pathlib import Path

from shaderbox.app import App
from shaderbox.profiling import FrameProfile, Span
from shaderbox.ui import update_and_draw

N_FRAMES = 300
# Warmup must clear TWO one-off costs before timing starts: every document in the project gets
# ONE first-render slot per frame (066 D2 -- a first render pays the pass's shader compile), so
# with three documents in the project the last one's compile can land several frames in; and the
# GPU query ring needs a few frames to fill (088 D2, 3-deep). 30 clears both with margin.
N_WARMUP = 30


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(round(p * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def _flatten_spans(span: Span, path: str, out: dict[str, dict]) -> None:
    key = f"{path}/{span.name}" if path else span.name
    entry = out.setdefault(key, {"cpu_ms": [], "gpu_ms": [], "count": []})
    entry["cpu_ms"].append(span.cpu_ms)
    entry["gpu_ms"].append(span.gpu_ms)
    entry["count"].append(span.count)
    for child in span.children:
        _flatten_spans(child, key, out)


def drive(
    app: App, document_id: str, label: str, render_all: bool, n_frames: int = N_FRAMES
) -> dict:
    app.app_state.is_render_all_documents = render_all
    app.set_current_document_id(document_id)
    # NOT app.profiler.enabled directly: ui.py's own frame body overwrites that every frame
    # from app.fps_details_open (088 D4 -- recording follows the FPS panel's open state, applied
    # at the next frame boundary). Driving the panel's own flag is what "enable recording
    # programmatically" means outside the UI.
    app.fps_details_open = True
    periods_ms: list[float] = []
    span_series: dict[str, dict] = {}
    last_complete_index_seen = -1
    for i in range(n_frames + N_WARMUP):
        start = time.perf_counter()
        update_and_draw(app)
        elapsed = (time.perf_counter() - start) * 1000.0
        if i >= N_WARMUP:
            periods_ms.append(elapsed)
            profile: FrameProfile | None = app.last_profile
            if profile is not None and profile.complete and profile.index != last_complete_index_seen:
                last_complete_index_seen = profile.index
                _flatten_spans(profile.root, "", span_series)
    periods_ms.sort()
    span_summary = {}
    for key, vals in span_series.items():
        cpu = sorted(vals["cpu_ms"])
        gpu = sorted(v for v in vals["gpu_ms"] if v is not None)
        span_summary[key] = {
            "n_samples": len(cpu),
            "cpu_median": _percentile(cpu, 0.5),
            "cpu_p95": _percentile(cpu, 0.95),
            "cpu_max": max(cpu) if cpu else 0.0,
            "gpu_median": _percentile(gpu, 0.5) if gpu else None,
            "gpu_p95": _percentile(gpu, 0.95) if gpu else None,
            "gpu_max": max(gpu) if gpu else None,
            "count_max": max(vals["count"]) if vals["count"] else 1,
        }
    return {
        "label": label,
        "render_all": render_all,
        "n_frames": len(periods_ms),
        "period_median_ms": _percentile(periods_ms, 0.5),
        "period_p95_ms": _percentile(periods_ms, 0.95),
        "period_max_ms": max(periods_ms) if periods_ms else 0.0,
        "spans": span_summary,
    }


def main() -> None:
    project_dir = Path(sys.argv[1])
    visible = "--visible" in sys.argv[2:]
    headless = not visible
    app = App(project_dir=project_dir, headless=headless)
    results = []
    try:
        for doc_id in ("trivial", "medium", "heavy"):
            results.append(
                drive(app, doc_id, f"{doc_id}_render_all_off", render_all=False)
            )
        for doc_id in ("trivial", "medium", "heavy"):
            results.append(
                drive(app, doc_id, f"{doc_id}_render_all_on", render_all=True)
            )
    finally:
        app.release()
    out_path = Path(sys.argv[0]).parent / (
        "results_visible.json" if visible else "results_headless.json"
    )
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    for r in results:
        print(
            f"{r['label']:>26}  median={r['period_median_ms']:.2f}ms  "
            f"p95={r['period_p95_ms']:.2f}ms  max={r['period_max_ms']:.2f}ms  "
            f"n={r['n_frames']}"
        )


if __name__ == "__main__":
    main()
