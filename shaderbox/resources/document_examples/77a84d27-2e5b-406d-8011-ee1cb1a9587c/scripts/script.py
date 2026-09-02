import math

from shaderbox.scripting import Ctx, ScriptBehavior, Vec3, Vec4


class Behavior(ScriptBehavior):
    """Turns the cursor into a brush, and seeds the canvas so it is not empty on first open.

    The shader cannot read the mouse; the engine gives it to a script, and the script hands it
    over as an ordinary uniform. `paint.frag.glsl` never knows a cursor exists -- it just gets
    told where to put paint.

    ctx.mouse carries POSITION only (x, y), not buttons, so there is no click to gate on: the
    brush follows the cursor whenever it MOVES, and holding still stops painting. That reads
    naturally in practice and costs the engine nothing.
    """

    def __init__(self) -> None:
        # Laid down one dab per frame so the document shows a lit scene the moment it opens:
        # a warm light, a cool light, a wall between them, and a round occluder.
        self.intro: list[tuple[float, float, float, float, float, float]] = [
            (0.28, 0.70, 1.0, 4.0, 3.3, 2.0),
            (0.80, 0.80, 1.0, 0.5, 1.2, 4.0),
            *[(0.50, 0.36 + 0.02 * i, 0.0, 0.0, 0.0, 0.0) for i in range(15)],
            (0.22, 0.26, 0.0, 0.0, 0.0, 0.0),
        ]
        self.last: tuple[float, float] | None = None

    def update(self, ctx: Ctx) -> dict[str, object]:
        if self.intro:
            x, y, emissive, r, g, b = self.intro.pop(0)
            return {
                "u_brush": Vec4(x, y, 0.035 if emissive else 0.02, 1.0),
                "u_brush_color": Vec3(r, g, b),
                "u_brush_emissive": emissive,
                "u_clear": 0.0,
            }

        # Paint while the cursor is moving. The threshold keeps a resting cursor from burning a
        # permanent dot into the canvas, which a persistent target would never forget.
        x, y = ctx.mouse.x, ctx.mouse.y
        moved = self.last is not None and math.dist((x, y), self.last) > 0.002
        self.last = (x, y)
        return {
            "u_brush": Vec4(x, y, 0.02, 1.0 if moved else 0.0),
            "u_brush_color": Vec3(4.0, 3.0, 1.6),
            "u_brush_emissive": 1.0,
            "u_clear": 0.0,
        }
