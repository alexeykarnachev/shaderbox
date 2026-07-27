# Cornerstone 08 — Mixed motion grid (compound-of-simples)

**Base capability under test:** holding MANY simultaneous simple constraints in one build — the
difficulty is VOLUME, not concept (maintainer direction 2026-07-27: single trivial asks prove
nothing; merge them). Every cell is independently checkable, so a failure attributes to a CELL,
not to the scene.

## Opening message (verbatim)

> Make a 3x3 grid of cells separated by thin gray lines on a near-black background. Each cell
> contains its own small figure with its own motion:
> top row: (1) a static white circle; (2) a red square spinning clockwise, one turn every 3
> seconds; (3) a static green triangle pointing up.
> middle row: (4) a white ball orbiting the cell center, one revolution every 2 seconds; (5) a
> yellow circle pulsing in size, one pulse per second; (6) a blue square sliding left and right,
> one full back-and-forth every 2 seconds.
> bottom row: (7) a static white ring; (8) a magenta circle blinking on and off twice per second;
> (9) a white cross spinning counterclockwise, one turn every 4 seconds.
> Keep every figure inside its cell.

## Ground truth / checklist (judge per CELL — 9 verdicts + 2 global)

- [ ] G1 grid: 3×3, thin gray lines, near-black bg
- [ ] G2 containment: no figure leaks outside its cell at any sampled t
- [ ] C1 white circle, static (t=0 ≡ t=3)
- [ ] C2 red square, CW, period 3s (t vs t+3 equal; t vs t+1.5 rotated 180°)
- [ ] C3 green triangle, up, static
- [ ] C4 white ball orbit, period 2s (t vs t+2 equal; t vs t+1 opposite side)
- [ ] C5 yellow circle pulse, period 1s (size at t vs t+0.5 differs, t vs t+1 equal)
- [ ] C6 blue square slide, period 2s (x at t vs t+1 mirrored, t vs t+2 equal)
- [ ] C7 white ring (annulus — hole visible), static
- [ ] C8 magenta blink 2 Hz (on/off alternates every 0.25s; present at t, absent at t+0.25)
- [ ] C9 white cross, CCW, period 4s (t vs t+2 rotated 180°, direction opposite C2)

Periods are verified by `render_at` pairs BEFORE trusting the video; direction (CW vs CCW) from
frame-to-frame deltas. The mp4 (10s) shows: 3+ turns of C2, 5 orbits of C4, 10 pulses of C5, 2.5
turns of C9.

## Drive

Fresh seeded project. Correction budget ≤2, each correction names the failing CELLS ("cell 5 does
not pulse; cell 8 never blinks"), never the fix. Tool choice is the agent's (pure GLSL is fine —
nothing here needs a script; a script is not a fail either). Report: dialogue + per-cell verdict
table + 10s mp4.
