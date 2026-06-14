import colorsys
import math

from shaderbox.scripting import Ctx, ScriptBehavior, Vec2, Vec3


class Behavior(ScriptBehavior):
    """Node script showcase (feature 048): ONE stateful object drives many uniforms via private
    helper methods. State lives on self and persists across frames — that is the whole point of
    scripting (a stateless sin(t) belongs in the shader). Uniforms NOT returned here (u_zoom,
    u_base_color, u_grid_density) stay manual in the panel. Stop a driven uniform (its row's stop
    button, or just drag it) to take it over by hand without editing this file."""

    def __init__(self) -> None:
        """Set up state (runs ONCE — at app start, before the first render, and on reload)."""
        self.swirl: float = 0.0      # accumulating field rotation
        self.spin: float = 0.0       # accumulating blob inner-spin
        self.hue: float = 0.0        # cycling tint hue
        self.theta: float = 0.0      # orbit phase
        self.flash: float = 0.0      # decays each frame, kicks once per lap
        self.laps: int = 0           # completed orbits, to detect a new lap
        self.orbit_dist: float = 0.6

    def _pulse(self, ctx: Ctx) -> float:
        return 0.5 + 0.5 * math.sin(ctx.t * 1.7)

    def _swirl(self, ctx: Ctx) -> float:
        # Faster swirl as the mouse moves right — integrate, never snap.
        self.swirl += ctx.dt * 0.35 * (0.5 + 2.0 * ctx.mouse.x)
        return self.swirl

    def _wave_offset(self, ctx: Ctx) -> Vec2:
        return Vec2(0.25 * math.sin(ctx.t * 0.9), 0.25 * math.sin(ctx.t * 1.3 + 1.57))

    def _tint(self, ctx: Ctx) -> Vec3:
        self.hue = (self.hue + ctx.dt * 0.08) % 1.0
        r, g, b = colorsys.hsv_to_rgb(self.hue, 0.45, 1.0)
        return Vec3(r, g, b)

    def _orbit(self, ctx: Ctx) -> tuple[Vec2, float, float, float]:
        self.theta += ctx.dt * 1.1
        self.spin += ctx.dt * 2.0
        lap = int(self.theta / (2.0 * math.pi))
        if lap > self.laps:
            self.laps = lap
            self.flash = 1.0
        self.flash = max(0.0, self.flash - ctx.dt * 1.5)
        pos = Vec2(self.orbit_dist * math.cos(self.theta), self.orbit_dist * math.sin(self.theta))
        radius = 0.12 + 0.04 * math.sin(self.theta * 3.0)
        return pos, radius, self.flash, self.spin

    def update(self, ctx: Ctx) -> dict:
        """Compute this frame's uniform values.

        Return a dict mapping uniform NAME -> value. A returned uniform PLAYS (the script drives it);
        an omitted one stays MANUAL.

        Args:
            ctx.t: Elapsed seconds since start.
            ctx.dt: Delta seconds since the previous frame.
            ctx.frame: Frame index.
            ctx.mouse: Cursor over the canvas (x, y in 0..1, y-up; 0.5,0.5 on export).
        """
        orbit_pos, orbit_radius, flash, spin = self._orbit(ctx)
        return {
            "u_pulse": self._pulse(ctx),
            "u_swirl": self._swirl(ctx),
            "u_wave_offset": self._wave_offset(ctx),
            "u_grid_density": 2,
            "u_tint": self._tint(ctx),
            "u_orbit_pos": orbit_pos,
            "u_orbit_radius": orbit_radius,
            "u_flash": flash,
            "u_spin": spin,
        }
