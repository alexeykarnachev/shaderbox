"""090 D9a's premise: what the GPU timer queries cost when they record on EVERY frame.

088 measured the ring's READ stall but never the `begin()`/`end()` pair as a permanent
background cost, and D9a turns recording always-on to feed the throttle. So the question is the
DELTA between a frame that opens its GPU spans and one that does not, under a load that opens
many of them: six shipped example documents with "Render all" on.

Shape, and why each part is load-bearing:

- **Interleaved blocks, not one arm then the other.** The GPU's clocks drift over a run, so two
  contiguous arms measure the drift as much as the change; alternating blocks hit both arms with
  it equally.
- **The p95 is the number, not the median.** Across two runs the median delta CHANGED SIGN
  (-0.169, then +0.337 ms) while the p95 stayed inside a hundredth of a millisecond -- so a
  single run's median would read as a real 2 % cost it is not.
- **A warm-up before either arm.** A first render pays a document's shader compiles (066 D1,
  admitted one document per frame by 066 D2) and the query ring is three deep (088 D2), so the
  first frames measure compilation and an unfilled ring.
- **`app.profiler.enabled` directly.** Before 090 the panel's `fps_details_open` drove it; since
  090 nothing does, and the flag is applied at the next frame boundary either way.

Pass threshold: **p95 delta <= 0.1 ms**, which is 0.6 % of a 16.7 ms frame.

**The delta scales with the number of GPU SPANS a frame opens, so read it per span.** This run
opens 12 (six documents' passes plus `ui:draw`) and measures a p95 total around +0.12 to +0.24 ms
depending on which half is being compared -- about 0.01 ms per span, which is the number that
transfers to another load. A third arm isolating the two halves (`Profiler.gpu` stubbed to a
no-op context manager, so the CPU tree still builds) measured +0.098 ms p95 for the tree and
+0.121 ms p95 for the queries themselves. A run with fewer documents open sees proportionally
less: the threshold is about a frame, and a frame's cost here is `spans x ~0.01 ms`.

Usage: `uv run python ai_docs/features/090_render_decoupling/probes/always_on_queries.py [frames_per_block]`
Seeds its own throwaway project under a temp dir from the shipped examples; never reads or
writes `projects/dev/`.
"""

import shutil
import sys
import tempfile
import time
from pathlib import Path

from shaderbox.app import App
from shaderbox.constants import DOCUMENT_EXAMPLES_DIR
from shaderbox.profiling import FrameProfile, Span
from shaderbox.ui import update_and_draw

FRAMES_PER_BLOCK = 50
N_BLOCKS = 6  # 6 x 50 = 300 frames per arm, the spec's run 1
N_WARMUP = 120


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(p * (len(ordered) - 1)))
    return ordered[index]


def _seed_project(root: Path) -> Path:
    project = root / "project"
    documents = project / "documents"
    documents.mkdir(parents=True)
    for example in sorted(DOCUMENT_EXAMPLES_DIR.iterdir()):
        if example.is_dir():
            shutil.copytree(example, documents / example.name)
    return project


def _gpu_spans(profile: FrameProfile | None) -> int:
    """How many GPU spans a frame opened -- the multiplier the delta scales with."""
    if profile is None:
        return 0

    def walk(span: Span) -> int:
        return (1 if span.gpu_ms is not None else 0) + sum(
            walk(child) for child in span.children
        )

    return walk(profile.root)


def _block(app: App, enabled: bool, frames: int) -> list[float]:
    app.profiler.enabled = enabled
    periods: list[float] = []
    for _ in range(frames):
        start = time.perf_counter()
        update_and_draw(app)
        periods.append((time.perf_counter() - start) * 1000.0)
    return periods


def main() -> None:
    frames = int(sys.argv[1]) if len(sys.argv) > 1 else FRAMES_PER_BLOCK
    root = Path(tempfile.mkdtemp(prefix="shaderbox-probe-"))
    try:
        app = App(project_dir=_seed_project(root), headless=True)
        app.app_state.is_render_all_documents = True
        try:
            _block(app, enabled=False, frames=N_WARMUP)
            off: list[float] = []
            on: list[float] = []
            for _ in range(N_BLOCKS):
                off += _block(app, enabled=False, frames=frames)
                on += _block(app, enabled=True, frames=frames)
            app_profile = app.last_profile
        finally:
            app.shutdown()
    finally:
        shutil.rmtree(root, ignore_errors=True)

    documents = len([d for d in DOCUMENT_EXAMPLES_DIR.iterdir() if d.is_dir()])
    spans = _gpu_spans(app_profile)
    print(
        f"{len(off)} frames/arm, {documents} documents, render-all on, "
        f"{spans} GPU spans per frame"
    )
    for label, series in (("OFF", off), ("ON", on)):
        print(
            f"  profiler {label:3s}  median {_percentile(series, 0.5):7.3f} ms"
            f"   p95 {_percentile(series, 0.95):7.3f} ms"
        )
    median_delta = _percentile(on, 0.5) - _percentile(off, 0.5)
    p95_delta = _percentile(on, 0.95) - _percentile(off, 0.95)
    print(f"  DELTA         median {median_delta:+7.3f} ms   p95 {p95_delta:+7.3f} ms")
    per_span = p95_delta / spans if spans else 0.0
    print(f"  per GPU span: {per_span:+7.4f} ms p95")
    print(
        f"  threshold: p95 delta <= 0.1 ms -> {'PASS' if p95_delta <= 0.1 else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
