# 077 — the design document (the single end-to-end ask)

The one message each model received in `rc_end_to_end`; assembled from what the babysat round taught (solidity in alpha, HDR emitters, float builtins, nearest filtering on the cascade, the 4-ray packing, the merge addressing). Sent verbatim through `scripts/dogfood/drive.py`.

```
Build me a radiance cascades demo from scratch, as a new document named RC. Do NOT open or copy the shipped Radiance Cascades example -- I want your own build from this spec. Read the whole spec, then build it stage by stage, checking each stage with a probe, and tell me when it is done. Here is the design.

WHAT IT IS. Real-time 2D global illumination: two coloured light emitters and a dark wall on a black canvas; the light bounces around the wall and casts a soft shadow; the emitters drift; I can draw extra emitters with the mouse and they stay.

PASSES (one shader per pass; a pass reads another pass through a sampler named u_<pass>; a pass reads its own previous run or frame through u_prev; the engine sets float u_pass_iteration / float u_pass_iterations -- declare them as float, never int; u_resolution is a vec2, u_aspect a float):

1. paint -- the scene to light, f2 target, 1 run. Two discs, warm vec3(4.0, 3.3, 2.0) and cool vec3(0.5, 1.2, 4.0) (HDR, tonemapped later), radius about 0.1 in centred aspect-corrected coordinates, and a thin dark wall SEGMENT between them (black, about 0.02 wide, from y = -0.7 to y = 0.35 in those coordinates) so light can pass around its top. Solidity goes in ALPHA: 1 where a disc, the wall or a drawn stroke is, 0 elsewhere, hard edges, no anti-aliasing. The disc centres come from two vec2 uniforms u_warm and u_cool that the script drives. paint also reads u_canvas (below) and treats its strokes as solid emitters of a bright colour.

2. canvas -- the drawing layer, f2 target, 1 run. Declares u_prev and a vec4 u_line (a segment a.xy-b.xy in uv, driven by the script). Draws a solid stroke of width ~0.006 along the segment (aspect-corrected so it is round) with alpha 1 and keeps whatever was there: max with u_prev.

3. seed -- f4 target, 1 run. Reads paint; where alpha is 1 stores that texel's uv in rg with a = 1, else a = 0.

4. jfa -- the jump flood, f4 target, 12 runs. Run 0 reads seed, every later run reads its own previous run via u_prev. Each run looks at the 9 neighbours at offset 2^(ceil(log2(max resolution)) - 1 - run) pixels and keeps the candidate whose stored uv is closest to this texel; candidates with a = 0 are empty. The result is a nearest-solid-texel map covering the whole canvas.

5. df -- f2 target, 1 run. Reads jfa; writes the distance in plain uv units (length(jfa.xy - vs_uv), no aspect scaling; 0 inside solid texels) into red, alpha 1.

6. cascade -- f2 target, 6 runs, NEAREST filtering (its texels are different directions of one probe; linear sampling would blend directions). Reads paint, df and its own previous run via u_prev. Every run writes the same full-size texture. Run 0 computes the COARSEST level: level = u_pass_iterations - 1 - u_pass_iteration. At level c the probes sit sp = 2^c texels apart and the sp x sp block of texels belonging to a probe stores its directions; a probe at level c casts rays = 4*sp*sp directions and each texel (slot si within the block) holds the MEAN of four rays idx = si*4 + k, k = 0..3, at angle 2pi*(idx + 0.5)/rays. Each ray sphere-marches from the probe centre through only its level's shell, from BASE*(4^c - 1)/3 to BASE*(4^(c+1) - 1)/3 with BASE = 0.012 in uv units: step by the df value, a hit when df < 0.0012 takes paint's colour, leaving the canvas is a black hit, running out of the shell is a MISS. On a miss and when the level is not the coarsest, MERGE from the level above: it wrote the mean of ITS four sub-directions into each of its slots, so ray idx reads exactly the upper slot number idx, at texel (idx mod usp, idx div usp) from the upper probe block's origin with usp = 2*sp, sampled at the texel centre; blend by hand bilinearly over the four upper probes around this probe's position (probe uv * resolution / usp - 0.5; floor and fract; indices clamped to the upper grid so nothing wraps). Write the mean of the four rays with alpha 1.

7. composite -- f1 target, linear, 1 run, THE OUTPUT. Reads cascade and paint: ACES-tonemap the cascade radiance, then where paint's alpha is 1 show paint's colour (tonemapped too), elsewhere the light. Alpha 1.

SCRIPT (a document script, CPU state): keep a position and a velocity per emitter on self, integrate with ctx.dt, bounce off the canvas edges, push them to paint as u_warm / u_cool; and set canvas.u_line from ctx.mouse: prev_x, prev_y, x, y while the button is down, an off-canvas segment like -1,-1,-1,-1 otherwise. Keep the wall static.

DONE MEANS: composite shows both halves lit with a soft falloff, the wall casting a visible shadow, the emitters moving; the probe on composite should report ink well above 80% and ANIMATES. If a stage looks wrong in its probe, fix it before moving on. Report what you built and what the probes say; do not claim a stage you did not check.
```
