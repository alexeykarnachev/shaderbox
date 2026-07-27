# Cornerstone 06 — Traffic light (discrete state logic in `script.py`)

**Base capability:** discrete-state sequencing with EXACT timings — a state machine in the script,
not a continuous animation. The video IS the ground truth readout.

## Opening message (verbatim)

> Make a traffic light: three stacked round lamps (red on top, yellow middle, green bottom) on a
> dark housing. It cycles: red for 2 seconds, then green for 2 seconds, then yellow for 1 second,
> then back to red, repeating forever. The python script must decide which lamp is lit.

## Ground truth / checklist

- [ ] three lamps, correct vertical order (red top / yellow mid / green bottom)
- [ ] exactly ONE lamp lit at any moment; unlit lamps visibly dim (not invisible, not all-on)
- [ ] phase timings from the video: red 2.0s → green 2.0s → yellow 1.0s → red … (5s cycle: a 10s
      mp4 shows exactly 2 full cycles)
- [ ] the switching is INSTANT (a state change, not a crossfade — unless asked)
- [ ] the state machine lives in `script.py` (script tools fired; the lit-lamp selector is a
      uniform the script drives)

## Drive

Fresh seeded project. Correction budget ≤2. Judge timings by `render_at` samples at t=0.5/2.5/4.5
(red/green/yellow) before the video. Report: dialogue + 10s mp4 (2 full cycles).
