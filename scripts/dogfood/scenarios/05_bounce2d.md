# Cornerstone 05 — Bouncing ball (physics in `script.py`)

**Base capability:** CPU-side physics — integrated state (gravity, velocity, damped restitution)
living in the node's python script and driving the shader through uniforms. GLSL-side `u_time`
trigonometry is a FAIL even if it looks right.

## Opening message (verbatim)

> Make a ball drop from the top of the frame and bounce off the floor, losing some energy each
> bounce until it comes to rest on the floor. The physics must live in the python script —
> integrate gravity and velocity there; do not fake it with GLSL time math.

## Ground truth / checklist

- [ ] free-fall arcs are parabolic (visibly accelerating downward, not constant-speed or sine)
- [ ] each bounce peak is LOWER than the previous (damping), and the ball eventually RESTS on the
      floor (no perpetual jitter, no sinking through)
- [ ] the ball never leaves the frame or penetrates the floor
- [ ] physics is in `script.py`: the script integrates `self.*` state (position/velocity) and
      pushes a position uniform; the shader only draws at the given position
- [ ] restart coherence: the mp4 (which replays the script from t=0) shows the same drop

## Drive

Fresh seeded project. Correction budget ≤2. Verify the script claim by reading the trace (script
tools fired, shader has a position uniform) — not by trusting the reply. Report: dialogue + 8-10s
mp4 (drop + bounces + rest must fit; if the sim rests later, ask for a faster sim, that's a
legitimate correction).
