# 104 — The instancing on-ramp

Feature 100's instanced passes are usable only by someone already told they exist. Ten
surfaces, all in, ordered by harm.

Research and evidence: `ai_docs/features/101_instancing_research/01_spec.md` (Part 3, D1–D10).
Gates on 102 and 103, so that what this teaches is true.

## Goal

An author who has never been told about instancing can find it, and nothing the app says
about shaders is false for an instanced pass.

## Design decisions

- **D1. All ten items ship.** A subset leaves an author on a partial on-ramp, which is the
  state this feature exists to end.
- **D2. Ordered: wrong text, then wrong picture, then silence.** The research's original
  "ordered by harm" claim was false — it called the prompt the only item that did damage,
  when the help modal states invariants instancing breaks. Text that is WRONG outranks text
  that is MISSING.
- **D3. A capability badge is per-PASS; a live count is per-FRAME.** Instancing is a property
  of a pass, read from the FLATTENED source — so a `flat in` spliced in from a `lib:` include
  counts and the open tab may not reveal it, which is itself a reason the badge earns its
  place. Capability is fixed at compile and belongs on the pass surfaces; the live count is
  per-frame state and belongs where per-frame state already shows, reading 102's outcome.
- **D4. `sb_instanced` is NOT documented.** It is engine-internal, exists only in the
  generated vertex stage, and is in `RESERVED_NAMES` — an author who declared it is refused.
  Documenting it would invite exactly that. Only `vs_quad` is user-facing.

## The ten

| # | Surface | What it needs |
|---|---|---|
| 1 | help modal, first section | It calls `vs_uv` and the full-screen quad "fixed". Both are false for an instanced pass. **Wrong text, in the section a new user reads first.** |
| 2 | `vs_quad` documented nowhere user-facing | The one name an author cannot guess, and guessing wrong is silent — reaching for `vs_uv` draws a canvas-wide vignette with no error. |
| 3 | help content's engine vocabulary | `vs_quad` beside `vs_uv`. Not `sb_instanced` (D4). |
| 4 | shader autocomplete | No notion of a `flat in` field; `vs_quad` offered nowhere. |
| 5 | the script stub | Generated from introspected scriptable uniforms, so an instanced-only document gets "(no scriptable uniforms)" and `return {}`. It can know: `entity_fields` is already on the pass. |
| 6 | blend mode undocumented | Now a per-pass choice (102 D5); document the choice, not the old constant. |
| 7 | uniform panel | No mark for an instanced pass. Per D3. |
| 8 | graph canvas | Same. Per D3. |
| 9 | README | Never mentions it; the scripting bullet names "a physics step, an integrator" and stops at uniform-driving. This is the itch.io pitch. |
| 10 | examples browser ordering | Entity Flock sorts LAST — and its description is the single best in-app explanation of the feature, gated behind the last gallery item. |

Script autocomplete for `@instances` is folded into 4 rather than listed separately:
completion inside a dict literal is the same surface and the fiddliest part of it.

## Gates

- Every user-facing name the engine provides is documented, with the domain enumerated from
  the engine rather than from a list in the test — and `sb_instanced` asserted ABSENT, so D4
  cannot erode.
- No help text asserts an invariant that an instanced pass breaks. Break it by restoring the
  "three things are fixed" sentence and watching the gate fail.
- An instanced pass is marked in the panel and the graph; a fullscreen pass is not.

## Out of scope

Nothing. This is the whole on-ramp.

## Open questions

None.
