import math

class Behavior(ScriptBehavior):
    """Drive u_spacing each frame. Keep state on self (persists across frames)."""
    def __init__(self) -> None:
        """Set up state (runs ONCE — at app start, before the first render, and on reload)."""
        pass

    def update(self, ctx: Ctx) -> Vec2:
        """Compute this frame's value.
        ctx.t  elapsed seconds | ctx.dt  delta seconds | ctx.frame  frame index
        ctx.mouse.x / ctx.mouse.y  cursor over the canvas, 0..1, y-up (0.5,0.5 on export)
        """
        return Vec2(0.0, 0.0)
