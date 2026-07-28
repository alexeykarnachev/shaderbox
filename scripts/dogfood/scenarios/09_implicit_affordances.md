# Cornerstone 09 — Implicit engine affordances (maintainer-designed, 2026-07-28)

**Base capability:** using the engine's affordances (`u_aspect`, `u_time`, `u_resolution`) when the
TASK requires them — with NONE of them named in the ask. The fixture makes each one physically
necessary: a correct result cannot exist without it. This probes whether the copilot understands
its environment, not whether it can follow a named instruction.

## Fixture setup (driver-side, before turn 1 — part of the fixture, not the dialogue)

Set the current node's canvas to **640x360** (16:9) and persist it. Every render/measurement in
this scenario is at 640x360 — never a square render (`render_at` squares the canvas; measure via
a direct render like the aspect experiment did).

## Opening message (verbatim)

> Make a stopwatch face: a round dial filling most of the frame height, sixty evenly spaced tick
> marks around its rim, each tick exactly one pixel wide, and a second hand that completes one
> full revolution per minute. Dark background, light dial, dark hand.

## What each requirement implicitly demands

- "round dial" on a 16:9 canvas → aspect-corrected coordinates (`u_aspect`) — else an ellipse.
- "exactly one pixel wide" → resolution-aware line width (`u_resolution`) — uv-space constants
  give ~2.6px-wide ticks at 640px.
- "one full revolution per minute" → time-driven rotation (`u_time`) with an exact rate.

## Ground truth / checklist (binary, measured at 640x360)

- [ ] dial is ROUND: bright-region bbox aspect 1.00 ± 0.03 (the killer check)
- [ ] 60 tick marks, evenly spaced (count dark/bright transitions around the rim circle)
- [ ] tick stroke ≈ 1px: cross-section FWHM ≤ 2px at the rim (AA allowed)
- [ ] hand rotates: angle(t+15s) − angle(t) = 90° ± 3°; period 60s (angle at t and t+60 equal)
- [ ] hand is a straight radial segment from center (not an off-center blob)
- [ ] palette as asked (dark bg, light dial, dark hand)

## Drive

Correction budget ≤2, corrections name the SYMPTOM only ("the dial is stretched wide", "the ticks
are thick"), never a uniform name — naming the mechanism would defeat the scenario. Report:
dialogue + a 640x360 PNG + (if the hand moves) a short mp4 + the six-axis verdicts. CODE axis:
does the source read the engine uniforms idiomatically (declared, used where needed) or hack
around them?
