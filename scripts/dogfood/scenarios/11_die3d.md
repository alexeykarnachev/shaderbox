# Echelon-2 scenario 11 — Rotating die (minimal real 3D)

**Capability cluster:** raymarched 3D in isolation — an SDF primitive with countable surface
features, one light, a shadow, a steady rigid rotation, fixed camera. No cloth, no layers: the
minimal test of the 3D craft (promoted from the deleted base-set 07 per the echelon plan).

## Opening message (verbatim)

> Raymarch a 3D die — a rounded cube with dark pips (1 to 6 dots on its faces) — slowly rotating
> on a neutral floor, one directional light casting a soft shadow. Keep the camera fixed.

## Ground truth / checklist (pixel-judged)

- [ ] reads as a CUBE: flat faces, visible edges, silhouette cycles face-on <-> edge-on as it
      turns (a real 3D rotation, not a 2D wobble)
- [ ] pips countable on visible faces, die-like counts (1..6), dark-on-light
- [ ] RIGID rotation: faces and their pips move together (pips glued to faces across a strip)
- [ ] one light direction evident; the shadow sits on the floor OPPOSITE the light and moves
      consistently with the rotation
- [ ] camera fixed (floor horizon static across the strip)
- [ ] rotation period steady (frame at t and t+period match)

## Drive

Fresh seeded project. Correction budget ≤2 (symptoms only). Expect this to be the hardest
scenario yet — attribution per checklist line matters more than a pass. Judge from a 6-8 frame
`render_strip` + a 10s mp4. Report: dialogue + mp4 + six-axis verdicts (CODE: SDF composition,
rotation as one transform, no duplicated per-face math).
