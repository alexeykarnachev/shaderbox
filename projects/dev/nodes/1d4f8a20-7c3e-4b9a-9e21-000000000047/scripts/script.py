import math
import random

from shaderbox.scripting import Array, Ctx, ScriptBehavior, Vec2


class Behavior(ScriptBehavior):
    """Autonomous stateful ping-pong simulation with trail history."""

    def __init__(self) -> None:
        self.arena_half_h = 1.0
        self.arena_half_w = 1.0

        self.paddle_x = 0.92
        self.paddle_half_h = 0.22
        self.paddle_half_w = 0.028
        self.paddle_speed = 1.45

        self.ball_radius = 0.045
        self.ball_speed = 1.05
        self.max_ball_speed = 1.85

        self.left_y = 0.0
        self.right_y = 0.0
        self.ball_x = 0.0
        self.ball_y = 0.0
        self.vx = 0.0
        self.vy = 0.0

        self.left_score = 0
        self.right_score = 0

        self.flash = 0.0
        self.reset_timer = 0.8

        self.trail_size = 24
        self.trail = [(0.0, 0.0)] * self.trail_size
        self.trail_count = 0
        self.trail_tick = 0.0
        self.trail_step = 0.022

        self._serve(direction=1.0)

    def _serve(self, direction: float) -> None:
        angle = random.uniform(-0.6, 0.6)
        self.ball_x = 0.0
        self.ball_y = random.uniform(-0.25, 0.25)
        self.vx = math.cos(angle) * self.ball_speed * direction
        self.vy = math.sin(angle) * self.ball_speed
        self.reset_timer = 0.55
        self._reset_trail()

    def _reset_trail(self) -> None:
        self.trail = [(self.ball_x, self.ball_y)] * self.trail_size
        self.trail_count = 1
        self.trail_tick = 0.0

    def _push_trail(self) -> None:
        self.trail.insert(0, (self.ball_x, self.ball_y))
        if len(self.trail) > self.trail_size:
            self.trail.pop()
        self.trail_count = min(self.trail_count + 1, self.trail_size)

    def _move_ai_paddle(self, y: float, target: float, dt: float) -> float:
        dead_zone = 0.03
        err = target - y
        if abs(err) < dead_zone:
            return y
        step = self.paddle_speed * dt
        y += max(-step, min(step, err))
        limit = self.arena_half_h - self.paddle_half_h - 0.02
        return max(-limit, min(limit, y))

    def _predict_target_left(self) -> float:
        look_ahead = 0.18
        return self.ball_y + self.vy * look_ahead

    def _predict_target_right(self) -> float:
        look_ahead = 0.18
        return self.ball_y + self.vy * look_ahead

    def _bounce_off_top_bottom(self) -> None:
        top = self.arena_half_h - self.ball_radius
        if self.ball_y > top:
            self.ball_y = top
            self.vy = -abs(self.vy)
            self.flash = 0.35
        if self.ball_y < -top:
            self.ball_y = -top
            self.vy = abs(self.vy)
            self.flash = 0.35

    def _check_paddle_hits(self) -> None:
        left_face = -self.paddle_x + self.paddle_half_w
        right_face = self.paddle_x - self.paddle_half_w

        if self.vx < 0.0 and self.ball_x - self.ball_radius <= left_face:
            if abs(self.ball_y - self.left_y) <= self.paddle_half_h + self.ball_radius:
                offset = (self.ball_y - self.left_y) / max(self.paddle_half_h, 1e-6)
                self.ball_x = left_face + self.ball_radius
                self.vx = abs(self.vx) * 1.04
                self.vy += offset * 0.55
                self.flash = 1.0

        if self.vx > 0.0 and self.ball_x + self.ball_radius >= right_face:
            if abs(self.ball_y - self.right_y) <= self.paddle_half_h + self.ball_radius:
                offset = (self.ball_y - self.right_y) / max(self.paddle_half_h, 1e-6)
                self.ball_x = right_face - self.ball_radius
                self.vx = -abs(self.vx) * 1.04
                self.vy += offset * 0.55
                self.flash = 1.0

        speed = math.hypot(self.vx, self.vy)
        if speed > self.max_ball_speed:
            k = self.max_ball_speed / speed
            self.vx *= k
            self.vy *= k

    def _check_score(self) -> None:
        if self.ball_x < -1.08:
            self.right_score = (self.right_score + 1) % 10
            self.flash = 1.0
            self._serve(direction=-1.0)
        elif self.ball_x > 1.08:
            self.left_score = (self.left_score + 1) % 10
            self.flash = 1.0
            self._serve(direction=1.0)

    def _trail_uniform(self) -> Array:
        flat = []
        for x, y in self.trail:
            flat.extend([x, y])
        return Array(flat)

    def update(self, ctx: Ctx) -> dict:
        dt = min(ctx.dt, 1.0 / 20.0)

        self.left_y = self._move_ai_paddle(self.left_y, self._predict_target_left(), dt)
        self.right_y = self._move_ai_paddle(self.right_y, self._predict_target_right(), dt)

        if self.reset_timer > 0.0:
            self.reset_timer = max(0.0, self.reset_timer - dt)
        else:
            self.ball_x += self.vx * dt
            self.ball_y += self.vy * dt
            self._bounce_off_top_bottom()
            self._check_paddle_hits()
            self._check_score()

        self.trail_tick += dt
        while self.trail_tick >= self.trail_step:
            self.trail_tick -= self.trail_step
            self._push_trail()

        self.flash = max(0.0, self.flash - dt * 1.8)

        return {
            "u_ball_pos": Vec2(self.ball_x, self.ball_y),
            "u_ball_vel": Vec2(self.vx, self.vy),
            "u_ball_radius": self.ball_radius,
            "u_trail": self._trail_uniform(),
            "u_trail_count": float(self.trail_count),
            "u_paddle_left_y": self.left_y,
            "u_paddle_right_y": self.right_y,
            "u_paddle_half_height": self.paddle_half_h,
            "u_paddle_half_width": self.paddle_half_w,
            "u_score_left": float(self.left_score),
            "u_score_right": float(self.right_score),
            "u_hit_flash": self.flash,
        }
