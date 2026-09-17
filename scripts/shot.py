"""Render N frames of the real app into a hidden window and write a PNG of the last one.

Why this exists: 094 shipped the node's uniform rows with every gate green and the canvas
visibly broken -- rows stacked in a column on the left, a `texture` row printing
`AutoSource()`, port labels doubled. None of that has a headless assertion, and the box the
app is developed on cannot screenshot a running window from the agent's side. So the frame is
rendered offscreen and read back, which is the one check that looks at what the user looks at.

    uv run python scripts/shot.py [--frames N] [--out PATH] [--project DIR]

With no `--project`, a throwaway copy of the shipped examples is used.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import glfw
import moderngl
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shaderbox.app import App
from shaderbox.constants import DOCUMENT_EXAMPLES_DIR
from shaderbox.logging_setup import configure_logging
from shaderbox.ui import update_and_draw


def _seed(tmp: Path) -> Path:
    project = tmp / "project"
    documents = project / "documents"
    documents.mkdir(parents=True)
    source = DOCUMENT_EXAMPLES_DIR
    for example in sorted(source.iterdir())[:3]:
        if example.is_dir():
            shutil.copytree(example, documents / example.name)
    return project


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--out", type=Path, default=Path("/tmp/shaderbox-shot.png"))
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    args = parser.parse_args()
    configure_logging()

    with tempfile.TemporaryDirectory(prefix="shaderbox-shot-") as tmp:
        project = args.project or _seed(Path(tmp))
        app = App(project_dir=project, headless=True)
        glfw.set_window_size(app.window, args.width, args.height)
        for _ in range(args.frames):
            update_and_draw(app)

        # Read the SCREEN framebuffer's own viewport, not the window size: on a HiDPI or
        # scaled display the two differ, and a mismatched stride shears every row -- which is
        # what the first cut of this script produced and is easy to mistake for a render bug.
        gl = moderngl.get_context()
        width, height = gl.screen.viewport[2], gl.screen.viewport[3]
        data = gl.screen.read(components=3, alignment=1)
        expected = width * height * 3
        assert len(data) == expected, (
            f"framebuffer read {len(data)} bytes for {width}x{height} (expected {expected})"
        )
        image = Image.frombytes("RGB", (width, height), data).transpose(
            Image.Transpose.FLIP_TOP_BOTTOM
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        image.save(args.out)
        print(f"shot: {args.out} ({width}x{height}, {args.frames} frames)")
        app.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
