"""A live tuner for the graph canvas's theme.

Serves one page of colour pickers and re-renders the REAL canvas on every
change, so what you are tuning is what the app draws rather than a mock: the
same `pack_nodes`, the same library, the same renderer. Run it, open the
URL, drag; press Export to get a `theme.py` block to paste.

    uv run python scripts/graph_theme/server.py

Headless: the render goes through a standalone GL context, so the app does
not need to be running and the two cannot disagree about the palette.
"""

import base64
import ctypes
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import moderngl
import numpy as np
from PIL import Image

from shaderbox.graph_canvas import ffi
from shaderbox.graph_canvas.adapter import (
    BodyRow,
    NodePalette,
    flat_view,
    pack_nodes,
    pass_key,
)
from shaderbox.graph_canvas.panel import GraphCanvasState, render_to_texture
from shaderbox.graph_canvas.render import CanvasRenderer
from shaderbox.pass_graph import Port
from shaderbox.theme import COLOR
from shaderbox.widgets.pass_graph import canvas_theme

HERE = Path(__file__).parent
PORT = 8765

# The GL context and the renderer are built once and reused: a context per
# request exhausts the driver's handles within a few dozen drags.
_LOCK = threading.Lock()
_CTX: moderngl.Context | None = None
_RENDERER: CanvasRenderer | None = None


def _colour_fields() -> list[str]:
    # A colour is the four-float array; everything else is a shading scalar.
    return [
        str(field[0])
        for field in ffi.Theme._fields_
        if getattr(field[1], "_length_", 0) == 4
    ]


def _scalar_fields() -> list[str]:
    return [
        str(field[0])
        for field in ffi.Theme._fields_
        if not getattr(field[1], "_length_", 0)
    ]


def _theme_as_dict(theme: ffi.Theme) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in _colour_fields():
        out[name] = [round(float(v), 4) for v in list(getattr(theme, name))]
    for name in _scalar_fields():
        out[name] = round(float(getattr(theme, name)), 4)
    return out


def _theme_from_dict(values: dict[str, Any]) -> ffi.Theme:
    theme = canvas_theme()
    for name in _colour_fields():
        if name in values:
            rgba = values[name]
            setattr(theme, name, (ctypes.c_float * 4)(*[float(v) for v in rgba]))
    for name in _scalar_fields():
        if name in values:
            setattr(theme, name, ctypes.c_float(float(values[name])).value)
    return theme


def _scene() -> Any:
    """One node of every kind the canvas draws, so a change is visible.

    A sampler that is wired and one that is not, the output row, an engine
    value, a script-driven one, a plain constant, a selected node and the
    output node -- every colour the theme carries has something on screen.
    """
    return pack_nodes(
        flat_view(
            ["src", "blur"],
            {
                "src": [],
                "blur": [
                    Port("u_src", "wired", "src"),
                    Port("u_mask", "unfilled"),
                ],
            },
            {"src": (0.0, 0.0), "blur": (260.0, 30.0)},
        ),
        {},
        output=pass_key("blur"),
        selected=frozenset({pass_key("src")}),
        body={
            "blur": [
                BodyRow("u_time", (0.69,), None),
                BodyRow("u_mouse", (0.5, 0.25), None, editable=True),
                BodyRow("u_gain", (1.0,), None, editable=True),
            ]
        },
        palette=NodePalette(
            hover=COLOR.GRAPH_HOVER,
            select=COLOR.SELECT,
            output=COLOR.ACCENT_PRIMARY,
        ),
    )


def render(theme: ffi.Theme, size: tuple[int, int]) -> bytes:
    global _CTX, _RENDERER
    with _LOCK:
        if _CTX is None:
            _CTX = moderngl.create_standalone_context()
            _RENDERER = CanvasRenderer(gl=_CTX)
        assert _RENDERER is not None
        state = GraphCanvasState()
        packed = _scene()
        background = [float(v) for v in list(theme.canvas)]
        canvas_rgba = (background[0], background[1], background[2], background[3])
        # Several frames: the eased highlights settle, so what is shown is
        # the resting picture rather than frame one of an animation.
        for _ in range(12):
            render_to_texture(
                state,
                _RENDERER,
                packed,
                size,
                canvas_rgba,
                ffi.PointerState(x=-1e6, y=-1e6),
                theme=theme,
                dt=1.0 / 60.0,
            )
        assert state.panel is not None and state.panel.fbo is not None
        raw = state.panel.fbo.read(components=3)
        pixels = np.frombuffer(raw, dtype="u1").reshape(size[1], size[0], 3)[::-1]
        state.release()
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        # Quiet: one line per keystroke while dragging a slider is noise.
        return

    def _send(self, code: int, kind: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html", (HERE / "index.html").read_bytes())
        elif self.path == "/theme":
            payload = {
                "current": _theme_as_dict(canvas_theme()),
                "library": _theme_as_dict(ffi.default_theme()),
                "colours": _colour_fields(),
                "scalars": _scalar_fields(),
            }
            self._send(200, "application/json", json.dumps(payload).encode())
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self) -> None:
        if self.path != "/render":
            self._send(404, "text/plain", b"not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        size = (int(request.get("w", 820)), int(request.get("h", 560)))
        png = render(_theme_from_dict(request.get("theme", {})), size)
        body = json.dumps({"png": base64.b64encode(png).decode()}).encode()
        self._send(200, "application/json", body)


def main() -> None:
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"graph theme tuner: http://127.0.0.1:{PORT}")
    print("ctrl-c to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
