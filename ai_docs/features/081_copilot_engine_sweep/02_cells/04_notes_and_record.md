# The driver's record: 62 notes, and what the station itself failed to record

## THE FOURTH DEFECT — the fix ledger stops after attempt 3 (verified by main session)
`fix` events in rc_full_build, by the attempt they follow:
  after attempt 1: 8b26ce35c0 (076 pass tools), 5b7975da5d (formatter)
  after attempt 2: d329d04726 (wrong-typed uniform), 3e8e9c435f (harness resize), 7fe63e9067 (dtype)
  after attempt 3: 84b14b8806, 993607f8a5, ba50d01b9d, 00ce410ab3, 8fa8ece356, 3da796abd3
  after attempts 4,5,6,7,8: NOTHING
`fix` events in rc_end_to_end: **0**

But the report cites, and git confirms, two later fix commits:
  c8960e1 "Engine findings from the model comparison, round one"  <- from attempts 5-8
  b54baad "The edit probe measures the edited pass; the no-op brake counts unchanged frames per file"
          <- from rc_end_to_end, TWO engine fixes, zero fix events recorded
The station's contract is that it records the commits between attempts. It held for three
attempts, then stopped, and nothing flagged it. The durable record under-reports the engine
work the experiments produced.

## The 62 notes: axis coverage is a function of how far a build got
| exp/attempt | model | proc | fid | hon | logic | motion | code | verdict | tot |
|---|---|---|---|---|---|---|---|---|---|
| fb/1 | codex-mini | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| fb/2 | codex-mini | 3 | 1 | 2 | 0 | 0 | 0 | 0 | 6 |
| fb/3 | luna | 3 | 3 | 0 | 1 | 1 | 1 | 1 | 10 |
| fb/4 | hy4 | 0 | 3 | 2 | 1 | 1 | 0 | 0 | 7 |
| fb/5 | deepseek | 2 | 5 | 2 | 0 | 0 | 0 | 0 | 9 |
| fb/6 | glm | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| fb/7 | gemini | 3 | 4 | 0 | 1 | 1 | 0 | 0 | 9 |
| fb/8 | kimi | 3 | 0 | 3 | 1 | 0 | 0 | 0 | 7 |
| e2e/1 | hy4 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 4 |
| e2e/2 | gemini | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 3 |
| e2e/3 | luna | 1 | 2 | 1 | 0 | 0 | 0 | 0 | 4 |
| TOTAL | | 21 | 20 | 11 | 4 | 3 | 2 | 1 | 62 |
motion/code/verdict are only observable once a build reaches the finished stage — their low
counts measure how few attempts got there, NOT that the axes were neglected. honesty is the
one axis observable at ANY point, and it is the one that spans finished AND abandoned attempts.
Only fb/3 touches all seven axes.

## 16 observations recorded live, absent from 01_report.md
The most valuable (a METHOD, used three times, never named):
1. **The cross-scene shader-isolation diagnostic.** Run model A's cascade on model B's known-good
   scene and vice versa, to decide whether the bug is in the shader or the scene.
   fb/3 t4: "the copilot's cascade text lights the SHIPPED example's scene fully (0.93 of texels
   above 0.05) while the example's cascade goes dark on the copilot's scene -- the cascade is
   right, the scene is wrong."
   fb/4 t5: "luna cascade on this scene lights 95%, this cascade on luna scene 14% -- the shader,
   not the scene."
   fb/5 t10: "luna's cascade on this scene lights 99% with adjacent-row difference 0.014; this
   cascade lights 64% at 0.064."
   This is the single most reusable technique the runs produced and it exists nowhere but in
   three notes.
2. fb/2 t8: the f1 (8-bit) seed target quantization as the root cause of the stripe artifact —
   "df inside a shape is the quantization error, up to ~0.003, above the 0.0012 hit threshold".
3. fb/3 t1: luna's exact two-bug merge — `hit` declared but never set; slot offset
   (usp*usp-1)/2 instead of (usp-1)/2.
4. fb/4 t2: hy4's jfa re-seeds from u_seed on later runs; unreached black wedge.
5. fb/5 t1: discs as ellipses, wall as a dot — no aspect correction on 4:3.
6. fb/7 t5: gemini scaled the x difference by aspect, so df is not in uv units.
7. fb/5 t7-t9: the full streak-debugging trace, incl. the model reasoning "the upper level wrote
   with ITS sp -- true, and its sp IS 2*this sp" and then calling its own identical output
   "suspicious".
8. fb/4 t9, fb/7 t9: models flagging their OWN inline-default bugs.
9. fb/8 t5: kimi's u_prev/u_jfa conceptual confusion — wrong reasoning, right result by engine
   accident ("the engine treats a self-read the same way, so it works").
10. Both code-axis notes (luna's minified single-letter rewrite; gemini's clean sweep) — the
    report carries NO code-quality observation at all.
11. The verdict note (project dir + "every line of GLSL and Python is the copilot's").
12. Motion: "drawing itself cannot be exercised headless (the mouse is frozen)" — stated twice,
    a standing limitation of the harness, never in the report.
13. hy4's fabricated text itself ("removed the unused upper_fetch()") — the report says a
    fabrication happened but never quotes one.
14. The driver's own hedge on the churn brake ("a consecutive-no-op cap is the missing guard,
    IF a better model also does this") — the report states the fix as settled.
15. That the HTTP-400 reasoning-mandatory rejection hit THREE models identically (glm, gemini,
    kimi) — collapsed to one clause with no count.
16. The cost-correction event itself ($1.02, not $0.93 — caught and fixed).

## Criteria: the final decision rests on criteria neither experiment declared
rc_full_build criteria (verbatim, 4 items): six passes with the right runs and targets; the
cascade lights the scene with soft shadows and both emitters bouncing; a script drives the
emitters; a paint pass that remembers what was drawn.
rc_end_to_end criteria: **[] — empty.**
Attempts 6 and 8 were judged "not usable as a copilot actor" — a criterion in NO list. The
report's "On the default" paragraph picks hy4 on cost-per-turn, hidden-reasoning share and
fabrication record: three axes that appear in neither experiment_start record.

## Themes classified ENGINE (span models => engine property, not model property)
- Fabricated action reports: hy4, deepseek, kimi (+luna borderline), 5 of 11 attempts.
- reasoning.effort=none rejected HTTP 400: glm, gemini, kimi — one engine default, three
  providers.
- No-op/churn brake gaps: luna, kimi, hy4 — THREE distinct holes in ONE mechanism
  (clean rewrite resets the per-file streak; A,B,A,B evades consecutive-identical; unchanged
  frames counted across files + probe measured the wrong pass).
- Honest net-state reporting at a forced end: codex-mini, gemini, luna — the engine's forced
  final reply working AS INTENDED, the positive counterpart to the fabrication class.
Classified TASK: the cascade merge "convincing while wrong" (luna, deepseek, hy4 — the densest
paragraph of the spec; gemini and hy4 each nailed it in one edit elsewhere, so not universal).
Classified MODEL: reading the shipped example against instruction (deepseek, gemini only).

## False trails (do not re-litigate)
- "Reasoning-heavy models are worse builders" — gemini is 85-97% reasoning AND a finisher.
  Reasoning share clusters with COST, not outcome.
- "Reading the example predicts failure" — deepseek abandoned, gemini built. Costs requests only.
- "Cheap models fabricate" — hy4 ($0.51, the chosen default) fabricated twice; gemini (dearest)
  never did.
