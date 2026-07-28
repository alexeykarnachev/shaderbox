# Final exam — scenario 13: Submarine command console (all axes in one run)

**Capability cluster — everything at once:** multi-panel layout/composition (03/08), exact-period
animation (04), scripted physics + state machine (05/10), 3D raymarching with lighting (11), polar
coordinates with angular falloff (12), implicit affordances — no uniform is ever named, non-square
canvas (09). One scene, one run.

## Opening message (verbatim)

> Build a submarine command console as one scene on an 800 by 450 canvas. Left half: a circular
> sonar scope on a dark background - concentric range rings, a bright beam sweeping one revolution
> every 4 seconds with a fading green afterglow trailing behind it, and a few small contacts that
> light up when the beam passes over them and fade after. Right top: a small 3D viewport showing a
> naval mine - a dark metal sphere with a few round studs sticking out - slowly tumbling, lit from
> the upper left so it reads as properly 3D. Right bottom: a depth panel - a vertical gauge showing
> the boat's depth between 0 and 300 meters with a moving marker, and a status lamp. Run the boat's
> logic in the node's script: it dives at a steady rate to 300, holds for 3 seconds, rises back to
> 50, holds 3 seconds, and repeats; the lamp is green while holding and amber while moving. Keep
> the scope perfectly round on this canvas.

## Ground truth / checklist

Layout/fidelity (pixels): three panels present and separated; scope round (aspect-correct on 16:9);
rings countable; studs read OUTSIDE the sphere silhouette; gauge + lamp visible.
Motion (replay strips + judge.py): sweep period EXACTLY 4.0s (`farthest_bright_angle` on left crop);
afterglow asymmetric (brighter behind the beam); contacts brighter just after sweep than half a rev
later; mine tumbles (`region_diff` on right-top crop between frames); depth marker moves.
Logic (`script_values` — rates/periods/counters ONLY): dive rate constant; top/bottom holds ≈3s;
cycle shape 300→hold→50→hold; lamp state flips exactly at phase boundaries (color via pixels).
Code: clean (r,theta) decomposition; local-frame 3D; readable script state machine (no magic
soup); tool choice (script for boat logic, GLSL for the rest).

## Drive

Fresh seeded project. Correction budget ≤3 (final exam compound; symptoms only, never solutions or
GLSL terms). Go-ahead turns free. End with the standard sweep turn (code-axis probe). Report:
dialogue + 8-12s mp4 + strips + six-axis verdicts; add to the agent-hub page.
