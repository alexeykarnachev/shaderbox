# Cornerstone 03 — Static composition (pure GLSL, no time)

**Base capability:** layout + counting + aspect correctness in a single static frame. No animation,
no script, no 3D — if this fails, nothing above it can be attributed.

## Opening message (verbatim)

> Draw exactly 5 white circles of equal size in a horizontal row, evenly spaced, on a dark navy
> background. Static image, no animation.

## Ground truth / checklist (all binary, judged on the PNG)

- [ ] exactly 5 circles (count them)
- [ ] all white (not gray/tinted), background dark navy (not black/blue-bright)
- [ ] equal radii (no visible size drift)
- [ ] one horizontal row, vertically centered
- [ ] even spacing incl. margins (no bunching at an edge)
- [ ] circles are ROUND (aspect-corrected — an ellipse fails)
- [ ] no animation: two renders at t=0 and t=3 are pixel-identical

## Drive

Fresh seeded project, current node. Correction budget ≤2 (state what's wrong, not how to fix).
Report: dialogue + final PNG (400px). This one's artifact is a PNG, not a video.
