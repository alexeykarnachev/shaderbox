import math
import random

from shaderbox.scripting import Ctx, ScriptBehavior


class Behavior(ScriptBehavior):
    """Drives the fire's behaviour: it breathes, flickers, and the wind gusts.

    The look lives in the shader; the LIFE lives here. Per-frame state (random
    walks for flicker and wind) persists on self.* across frames — the reason
    CPU scripting exists, vs. baking everything into u_time math in GLSL.
    """

    def __init__(self) -> None:
        # Wind: a smoothed random walk in [-1, 1]. The flame leans + gusts.
        self.wind = 0.0
        self.wind_target = 0.0
        self.gust_timer = 0.0

        # Flicker: a fast jitter amplitude that itself wanders, so the flicker
        # isn't a clean sine — sometimes it's calm, sometimes agitated.
        self.flicker_amp = 0.06
        self.flicker_phase = random.uniform(0.0, 10.0)

    def update(self, ctx: Ctx) -> dict:
        dt = min(ctx.dt, 1.0 / 20.0)
        t = ctx.t

        # --- wind: pick a new gust target now and then, ease toward it ---
        self.gust_timer -= dt
        if self.gust_timer <= 0.0:
            self.wind_target = random.uniform(-1.0, 1.0)
            self.gust_timer = random.uniform(0.7, 2.2)
        # Critically-damped-ish ease so the lean is smooth, never snappy.
        self.wind += (self.wind_target - self.wind) * min(1.0, dt * 2.2)

        # --- flicker: wandering amplitude * fast oscillation ---
        # Amplitude drifts via a slow random walk, clamped to a sane band.
        self.flicker_amp += random.uniform(-0.04, 0.04) * dt * 8.0
        self.flicker_amp = max(0.02, min(0.14, self.flicker_amp))
        self.flicker_phase += dt * random.uniform(11.0, 15.0)
        flicker = 1.0 - self.flicker_amp * (0.5 + 0.5 * math.sin(self.flicker_phase))

        # --- breathe: slow brightness swell so the fire isn't metronomic ---
        breathe = 0.92 + 0.08 * math.sin(t * 0.9) + 0.04 * math.sin(t * 2.3 + 1.7)

        return {
            "u_wind": self.wind,
            "u_flicker": flicker,
            "u_intensity": breathe,
        }
