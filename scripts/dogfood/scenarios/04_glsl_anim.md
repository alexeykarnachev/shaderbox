# Cornerstone 04 — u_time animation (GLSL only, no script)

**Base capability:** simple periodic motion with an EXACT stated period, done in the shader (the
model should NOT reach for a python script for a pure per-pixel orbit).

## Opening message (verbatim)

> Make a small white ball orbit the center of the frame in a perfect circle, one full revolution
> every 2 seconds, on a black background.

## Ground truth / checklist

- [ ] one ball, white, on black
- [ ] path is a circle centered in the frame (not an ellipse drifting off-center)
- [ ] period EXACTLY 2s: in a 10s video the ball completes exactly 5 revolutions; frames at t and
      t+2 match, frames at t and t+1 are opposite
- [ ] motion is smooth (no stutter/jumps between frames)
- [ ] no python script created (a `probe`/trace check — GLSL time math is the right tool here)

## Drive

Fresh seeded project. Correction budget ≤2. Judge period via `render_at` pairs (t=0 vs t=2 equal;
t=0 vs t=1 opposite) before trusting the eye. Report: dialogue + 10s mp4 (5 revolutions visible).
