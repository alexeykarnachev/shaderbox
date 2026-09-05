"""The one read-only object every behavior's `update` receives (features 041 + 042)."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MouseState:
    """The cursor over the document's canvas, as a script reads it.

    Coordinates are normalized to 0..1 with y pointing up, the GLSL convention: the origin is
    the bottom-left corner. Outside the canvas the position holds at the last in-bounds sample
    and `down` clears; the next in-bounds sample restarts prev at the current position, so a
    stroke that left the canvas does not resume as one long line. An exported render sees
    `EXPORT_MOUSE` instead, whatever the live cursor was doing.

    Attributes:
        x: Horizontal position, 0 at the left edge and 1 at the right.
        y: Vertical position, 0 at the bottom edge and 1 at the top.
        down: Whether the left button is held with the cursor over the canvas.
        prev_x: Last frame's x, equal to x on the first frame and on re-entry.
        prev_y: Last frame's y, under the same rule.
    """

    x: float = 0.5
    y: float = 0.5
    down: bool = False
    prev_x: float = 0.5
    prev_y: float = 0.5


# The fixed cursor an EXPORTED render sees, so a video is deterministic regardless of where the
# live cursor was (feature 042). Injected by the export-isolation seam, never the live App.
# `down` is False and prev equals the position: a script gated on the button paints nothing in an
# export, and one reading prev-to-current sees a zero-length capsule rather than a jump.
EXPORT_MOUSE = MouseState(0.5, 0.5, False, 0.5, 0.5)


@dataclass(frozen=True)
class ScriptContext:
    """The engine state for one frame, handed to `update` once per drawn frame.

    A script's own state lives on the behavior instance (`self.*`), never here: this object is
    rebuilt every frame and is frozen.

    Attributes:
        t: Seconds since the document started playing.
        dt: Seconds since the previous frame.
        frame: The frame index, counting from 0 at the start of playback.
        mouse: The cursor over the canvas, as a `MouseState`.
    """

    t: float
    dt: float
    frame: int
    mouse: MouseState = field(default_factory=lambda: EXPORT_MOUSE)
