"""The one read-only object every behavior's `update` receives (features 041 + 042)."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MouseState:
    # Cursor over the document's canvas, normalized 0..1, y-UP (origin bottom-left — the GLSL
    # convention; the preview draws uv-flipped, so the hit-test flips y back). Outside the canvas
    # the position clamps to the last in-bounds sample and `down` clears; the next in-bounds sample
    # restarts prev at the current position. Export uses EXPORT_MOUSE.
    x: float = 0.5
    y: float = 0.5
    # Left button held with the cursor over the canvas. False on export and in the headless probe.
    down: bool = False
    # Last frame's position (equal to x/y on the first frame and after re-entering the canvas),
    # so a shader can stamp the CAPSULE from prev to current instead of one disc per frame.
    prev_x: float = 0.5
    prev_y: float = 0.5


# The fixed cursor an EXPORTED render sees, so a video is deterministic regardless of where the
# live cursor was (feature 042). Injected by the export-isolation seam, never the live App.
# `down` is False and prev equals the position: a script gated on the button paints nothing in an
# export, and one reading prev-to-current sees a zero-length capsule rather than a jump.
EXPORT_MOUSE = MouseState(0.5, 0.5, False, 0.5, 0.5)


@dataclass(frozen=True)
class EngineContext:
    # The clock + the cursor. state lives in the behavior instance (self.*), not here. `Ctx` is
    # the name in scope inside a script. `mouse` defaults so the bare-clock construct sites compile.
    t: float
    dt: float
    frame: int
    mouse: MouseState = field(default_factory=lambda: EXPORT_MOUSE)


Ctx = EngineContext
