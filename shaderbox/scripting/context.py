"""The one read-only object every behavior's `update` receives (features 041 + 042)."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MouseState:
    # Cursor over the node's canvas, normalized 0..1, y-UP (origin bottom-left — the GLSL
    # convention; the preview draws uv-flipped, so the hit-test flips y back). Outside the
    # canvas the live value clamps to the last in-bounds position; export uses EXPORT_MOUSE.
    x: float = 0.5
    y: float = 0.5


# The fixed cursor an EXPORTED render sees, so a video is deterministic regardless of where the
# live cursor was (feature 042). Injected by the export-isolation seam, never the live App.
EXPORT_MOUSE = MouseState(0.5, 0.5)


@dataclass(frozen=True)
class EngineContext:
    # The clock + the cursor. state lives in the behavior instance (self.*), not here. `Ctx` is
    # the name in scope inside a script. `mouse` defaults so the bare-clock construct sites compile.
    t: float
    dt: float
    frame: int
    mouse: MouseState = field(default_factory=lambda: EXPORT_MOUSE)


Ctx = EngineContext
