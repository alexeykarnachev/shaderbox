"""A flock of 20000 entities, simulated on the CPU and drawn in one call.

The simulation is plain numpy over whole arrays -- no loop over entities, which is what
keeps 20000 of them inside a frame. Returning them under "@instances" is what makes the
pass draw one quad each instead of one quad over the canvas.
"""

import numpy as np

from shaderbox.scripting import ScriptBehavior, ScriptContext

COUNT = 20000
# Each entity ORBITS its attractor rather than falling into it: the pull is turned 90
# degrees into a tangential push, with only a weak radial term holding the orbit at a
# radius. Pure attraction collapses the whole flock onto two points within a second --
# measured, and it renders as two white blobs.
PULL = 1.6
SWIRL = 2.4
ORBIT = 0.20
DRAG = 0.97
JITTER = 0.05
MAX_SPEED = 0.014


class Behavior(ScriptBehavior):
    def __init__(self) -> None:
        rng = np.random.default_rng(11)
        angle = rng.random(COUNT) * 2.0 * np.pi
        spread = np.sqrt(rng.random(COUNT))
        # Columns, not rows of structs: every rule below is one array operation over the
        # whole flock, and that is the difference between 20000 entities and 200.
        self.x = (np.cos(angle) * spread * 0.8).astype("f4")
        self.y = (np.sin(angle) * spread * 0.8).astype("f4")
        self.vx = np.zeros(COUNT, "f4")
        self.vy = np.zeros(COUNT, "f4")
        self.radius = (rng.random(COUNT) * 0.010 + 0.004).astype("f4")
        self._rng = rng

    def update(self, context: ScriptContext) -> dict:
        # Two attractors circling at different rates. Each entity follows the nearer one,
        # so the flock splits and merges on its own.
        t = context.t
        ax = np.float32(-0.42 + np.cos(t * 0.7) * 0.16)
        ay = np.float32(-0.08 + np.sin(t * 0.9) * 0.22)
        bx = np.float32(0.42 + np.cos(t * 1.3 + 2.2) * 0.16)
        by = np.float32(-0.08 + np.sin(t * 0.5 + 1.0) * 0.22)

        to_a = (self.x - ax) ** 2 + (self.y - ay) ** 2
        to_b = (self.x - bx) ** 2 + (self.y - by) ** 2
        near_a = to_a < to_b
        tx = np.where(near_a, ax, bx)
        ty = np.where(near_a, ay, by)

        dt = np.float32(min(context.dt, 1.0 / 30.0))
        # Toward the attractor, and the distance to it.
        rx = tx - self.x
        ry = ty - self.y
        dist = np.sqrt(rx**2 + ry**2) + np.float32(1e-5)
        nx = rx / dist
        ny = ry / dist

        # Radial: pull in when outside the orbit radius, push out when inside. This is
        # what stops the collapse -- an entity at the centre is pushed back out.
        radial = (dist - np.float32(ORBIT)) * PULL
        self.vx += nx * radial * dt
        self.vy += ny * radial * dt

        # Tangential: the same direction turned 90 degrees, which is what makes it orbit
        # rather than oscillate through the centre.
        self.vx += -ny * SWIRL * dt
        self.vy += nx * SWIRL * dt
        self.vx += self._rng.standard_normal(COUNT).astype("f4") * JITTER * dt
        self.vy += self._rng.standard_normal(COUNT).astype("f4") * JITTER * dt
        self.vx *= DRAG
        self.vy *= DRAG

        speed = np.sqrt(self.vx**2 + self.vy**2)
        # Which flock, and how far out. Speed itself is nearly constant here -- the clamp
        # below sees to that -- so colouring by it would make every entity the same shade.
        heat = np.where(near_a, 0.0, 1.0).astype("f4") * np.float32(0.72)
        heat += np.clip(dist / np.float32(ORBIT * 2.2), 0.0, 1.0).astype("f4") * np.float32(0.28)
        too_fast = speed > MAX_SPEED
        scale = np.where(too_fast, MAX_SPEED / np.maximum(speed, 1e-6), 1.0).astype("f4")
        self.vx *= scale
        self.vy *= scale

        self.x += self.vx
        self.y += self.vy

        # A soft wall at the canvas edge. Without it the two orbits drift as a pair and
        # the flock leaves the frame within a few seconds -- visible, and the kind of
        # thing only a rendered frame shows.
        for axis, velocity, limit in ((self.x, self.vx, 0.95), (self.y, self.vy, 0.92)):
            outside = np.abs(axis) > limit
            velocity[outside] *= np.float32(-0.5)
            np.clip(axis, -limit, limit, out=axis)

        # The shader's `flat in` names, matched by name. `pos` is (N, 2) because it is a
        # vec2 there; the scalars are (N,). Both must be f4 -- the engine checks rather
        # than converting, because numpy's own assignment would cast an f8 silently.
        return {
            "swarm": {
                "@instances": {
                    "pos": np.ascontiguousarray(np.stack([self.x, self.y], axis=1)),
                    "radius": self.radius,
                    "heat": np.ascontiguousarray(heat),
                },
            }
        }
