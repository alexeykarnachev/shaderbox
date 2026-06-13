class Behavior(ScriptBehavior):
    """Drive u_color2 each frame.
    ctx.t  elapsed seconds | ctx.dt  delta seconds | ctx.frame  frame index
    ctx.mouse.x / ctx.mouse.y  cursor over the canvas, 0..1, y-up (0.5,0.5 on export)
    Return Vec3. Keep state on self (persists across frames).
    Math is pre-loaded — call sin/cos/sqrt/clamp/lerp(=mix)/... or math.* directly (no import).
    """
    def __init__(self) -> None:
        pass

    def update(self, ctx: Ctx) -> Vec3:
        return Vec3(0.0, 0.0, 0.0)
