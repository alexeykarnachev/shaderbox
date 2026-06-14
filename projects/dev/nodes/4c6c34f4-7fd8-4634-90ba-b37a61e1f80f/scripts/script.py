import math

class Behavior(ScriptBehavior):
    """Drive many uniforms from one object each frame (node-brain). Keep state on self."""
    def __init__(self) -> None:
        """Set up state (runs ONCE — at app start, before the first render, and on reload)."""
        pass

    def update(self, ctx: Ctx) -> dict:
        """Compute this frame's value.
        ctx.t  elapsed seconds | ctx.dt  delta seconds | ctx.frame  frame index
        ctx.mouse.x / ctx.mouse.y  cursor over the canvas, 0..1, y-up (0.5,0.5 on export)
        """
        return {
            'SBT_SPANS': Array([0.0] * 312),  # Array
            'SBT_STROKES': Array([0.0] * 2080),  # Array
            'u_aspect': 0.0,  # float
            'u_char_height': 0.0,  # float
            'u_color1': Vec3(0.0, 0.0, 0.0),  # Vec3
            'u_color2': Vec3(0.0, 0.0, 0.0),  # Vec3
            'u_spacing': Vec2(0.0, 0.0),  # Vec2
            'u_text': Text(""),  # Text
            'u_time': 0.0,  # float
            'u_weight': 0.0,  # float
        }
