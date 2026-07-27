# Cornerstone 07 — Spinning die (raymarched 3D + light)

**Base capability:** minimal real 3D — an SDF primitive with countable surface features, one light,
a shadow, and a steady rotation. No cloth, no layers, no noise: 3D craft in isolation.

## Opening message (verbatim)

> Raymarch a 3D die — a rounded cube with dark pips (1 to 6 dots on its faces) — slowly rotating on
> a neutral floor, one directional light casting a soft shadow. Keep the camera fixed.

## Ground truth / checklist

- [ ] it reads as a CUBE (flat faces, edges visible) — not a blob
- [ ] pips are countable on visible faces and the counts are die-like (1-6, dark on light)
- [ ] rotation is steady and COHERENT (the whole die turns as one rigid body — faces/pips move
      together; pips glued to their faces)
- [ ] one light direction is evident; the shadow sits on the floor OPPOSITE the light
- [ ] camera fixed (the floor horizon doesn't drift)
- [ ] silhouette cycles face-on ↔ edge-on as it turns (a real 3D rotation, not a 2D wobble)

## Drive

Fresh seeded project. Correction budget ≤2. This is the HARDEST cornerstone — expect corrections;
what matters is attribution (which checklist line failed). Rigid-body coherence is the axis the
flag run exposed — judge it from the mp4, not a still. Report: dialogue + 10s mp4.
