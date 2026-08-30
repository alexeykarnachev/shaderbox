import math

from shaderbox.scripting import ScriptBehavior, Ctx

class Behavior(ScriptBehavior):
    """Drive many uniforms from one object each frame (node script). Keep state on self."""

    def __init__(self) -> None:
        """Set up state (runs ONCE — at app start, before the first render, and on reload)."""
        pass

    def update(self, ctx: Ctx) -> dict:
        """Compute this frame's uniform values.

        Return a dict mapping uniform NAME -> value. A uniform you return is DRIVEN by the
        script (it PLAYS); a uniform you omit (or map to None) stays MANUAL — you edit it by
        hand in the panel. Stop a playing uniform (its row's stop button, or just drag it) to
        edit it by hand without deleting it from the dict.

        A value that is a pure function of `ctx.t` usually belongs in the shader instead;
        this class is for state you keep on `self`.

        Args:
            ctx.t: Elapsed seconds since start.
            ctx.dt: Delta seconds since the previous frame.
            ctx.frame: Frame index.
            ctx.mouse: Cursor over the canvas (x, y in 0..1, y-up; 0.5,0.5 on export).
        """
        return {
            # Uncomment a line + replace the value to drive that uniform:
            # 'u_bloom': 0.0,  # float
            # 'u_glow': 0.0,  # float
            # 'u_spark_count': 0,  # int
            # 'u_spark_size': 0.0,  # float
            # 'u_threshold': 0.0,  # float
            # 'u_fade': 0.0,  # float
        }
