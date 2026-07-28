# Echelon-2 scenario 12 — Radar sweep (2D radial/polar coordinates)

**Capability cluster:** polar-coordinate craft — angle/radius decomposition, angular periodicity,
angular falloff (afterglow), radial layout. Perpendicular to 10 (state machine) and 11 (3D): pure
2D GLSL where the natural frame is (r, theta), not (x, y).

## Opening message (verbatim)

> Make a radar screen: a dark circular scope with a few concentric range rings and a bright beam
> sweeping around steadily, one revolution every 4 seconds, leaving a fading green afterglow
> behind it. Add a few small blips on the scope that light up when the beam passes over them and
> then fade.

## Ground truth / checklist (pixel-judged)

- [ ] circular scope, concentric rings (countable), radial layout centered
- [ ] the beam sweeps: period EXACTLY 4s (frame at t matches t+4; beam angle advances 90deg/s)
- [ ] afterglow trails BEHIND the beam (brightness falls off with angular distance behind it, not
      symmetric around it)
- [ ] blips light when swept and FADE afterwards (a blip is brighter just after the beam passes
      than half a revolution later)
- [ ] scope stays round (aspect-correct on any canvas)

## Drive

Fresh seeded project. Correction budget ≤2 (symptoms only). Judge angles/period with
`judge.py::farthest_bright_angle` over `render_strip` replay frames. Report: dialogue + 8-12s mp4
+ six-axis verdicts (CODE: is there a clean (r, theta) decomposition, or cartesian spaghetti?).
