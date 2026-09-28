# 103 — The copilot learns instancing

**The copilot is misinformed, not uninformed, and that is worse.** Its own probe reports the
shipped, working Entity Flock example as driving nothing and animating nothing, so an agent
that writes a correct instanced script is told it failed and "fixes" what was right.

Research and evidence: `ai_docs/features/101_instancing_research/01_spec.md` (Part 2, C1–C10).
Gates on 102, whose outcome seam this consumes.

## Goal

Full integration (D-A): the copilot understands the mechanism, writes instanced passes, and
its tools carry populations — in the only shape the boundary permits.

## Design decisions

- **D1. The copilot writes the GENERATOR and reads back STATISTICS.** Settled by arithmetic,
  not preference: tool arguments are JSON Schema, and 20k entities × 4 f4 columns is ~320 KB.
  A population can never cross the tool boundary as data. The copilot already has
  `write_script` / `edit_script`; what it lacks is a result channel carrying count,
  per-column dtype/shape/range, and the validation verdict. **The work is in the RESULT
  types, not in a new argument type.** Revisit only if the boundary stops being JSON.
- **D2. The dry-run REPORTS a population without WRITING one — 063 is satisfied, not bent.**
  063 protects one property: a `dry_run` leaves the live document byte-identical. Reporting
  is not writing.

  READ, and verified in review by spying on a real `dry_run` over the shipped Entity Flock:
  `validate_population` is called 61 times (once per probed frame), each returning
  `count=20000, problem=None`. `engine.py:843` runs it before the write guard at
  `engine.py:885`, so the statistics genuinely exist at that point. (The research marked this
  reasoning-unverified; it is now measured, and the register is restored below.)

  **Two things the premise does not cover, both of which the builder needs:**
  - **Validation is CONDITIONAL.** `engine.py:836-841` returns early when `entity_fields` is
    empty, which is the case for a pass whose compile FAILED — the population passes through
    unvalidated and no statistics exist. The report must say "not validated", never a false
    zero.
  - **The sink is the actual work, and it does not exist.** `populations` is computed and then
    falls out of scope on the dry-run path; `values_sink` cannot carry it, because
    `tests/test_instances_routing.py:250-264` gates that no sampled value is an `ndarray` or
    a `dict` — correctly. A second sink carrying STATISTICS ONLY is the one new piece of
    engine machinery this feature needs, and an earlier draft of this spec never named it.

  **Revisit if** a probe needs the columns themselves rather than statistics — at which point
  the 063 ruling is genuinely in the way and this decision no longer applies.
- **D3. The prompt gets an instancing BLOCK, not a deleted sentence.** Sized on the order of
  the existing SCRIPTING section. MEASURED in review: both candidate homes (`_SYSTEM_PROMPT`,
  STATIC; `_context_block`, RARE) sit in the cacheable prefix above the DIALOGUE trim, so the
  block is a one-time prefix cost and not a per-turn one — the token-cost worry is unfounded.
  One assembly constraint: `tests/test_script_api_doc.py:82-89` pins the order
  EXAMPLE LIBRARY < SCRIPT API < CONVENTIONS, so a block inserted between those breaks it.
  **Revisit if** a dogfood run shows the block does not change what the model builds when
  asked for a particle system — the measurement that would show it earning its tokens. `prompt.py` currently
  recommends the technique feature 100 replaced, naming a boids flock. Removing that is half
  the job; the model then needs the mechanism — `flat in` as the declaration, `pos`/`radius`
  reserved and clip-space, `vs_quad`'s range against `vs_uv`'s, the f4/i4/u4 dtype rule,
  C-contiguity, the `(N, components)`/`(N,)` shapes, the every-column-needs-a-field
  bijection, and 102's blend mode. Plus the watershed it must now state: a POPULATION of
  similar things → instancing; a small fixed vector of parameters → an array uniform. The
  block must NOT name `sb_instanced` (104 D4's rule binds here too — it is engine-internal
  and declaring it is refused).
- **D4. The copilot must be able to SEE that a pass is instanced — on BOTH surfaces.**
  `ShaderView` carries no entity fields, so an instanced pass reads as a fullscreen pass with
  odd inputs and `edit_shader` on it is blind. **But `ShaderView` is the surface the agent
  reads LESS**: the working set, rebuilt every step and rendered at `prompt.py:351-382`, is
  `WorkingSetView`/`PassView`, and an agent that never calls `read_shader` still sees nothing.
  Both carry it. `WorkingSetView` already has the precedent of appended defaulted fields.
  Note both are frozen dataclasses with no defaulted fields today, so a new field must be
  appended WITH a default or six construction sites break.
  **Revisit if** a third view of a pass appears — at which point the entity-field fact wants a
  single accessor rather than a field on each view.
- **D4a. `sb_instanced` is added to `ENGINE_DRIVEN_UNIFORMS`.** Found in review by running it:
  the engine's own instancing mode flag is NOT in that set, so a script can "drive" it, the
  probe reports it as cleanly driven, `Pass.render` overwrites it unconditionally every frame
  (`core.py:629-631`), and it also surfaces to the USER as an editable uniform row. A
  one-line fix at the root corrects the probe, the stub, `_format_uniforms` and the UI at
  once. This is 103's because D1/D5's domain widening is where it belongs, and because the
  copilot is the consumer that acts on the false report.
- **D4b. The script stub teaches `@instances`.** MEASURED: generating the stub for the flock
  document with its script removed offers `sb_instanced` among the scriptable uniforms and
  mentions `@instances` nowhere. That stub is the copilot's FIRST sight of an instanced
  document (`tools/script.py:116-120`), which makes it more likely first contact than the
  tools C10 names. D4a removes the false offer; this adds the true one. Shares a code site
  with 104 item 5 (the human-facing half) — whichever wave lands first does it, and the other
  cites it.
- **D6. C9 is 103's work, and 102's outcome does not reach it by itself.** Found in review:
  each spec pointed at the other, and neither owned it. `_probe_frame` (`backend.py:154`)
  calls `document.render(...)` and **discards the return**, computing the facts line from
  pixels. So even with 102's outcome type in place, threading it out of `document.render`
  into `_render_facts_for` and the facts string is copilot-side plumbing. 102 correctly lists
  copilot work as out of scope; this decision claims it here so it is owned.
- **D5. The API doc is fixed AND its gate is widened to a domain that can catch the next
  drift.** The generated doc asserts every value is plain Python while the engine requires
  numpy under `@instances`. Its gate enumerates from `_stub_kind`, which dispatches on
  `moderngl.Uniform`, so a reserved key is outside its domain BY CONSTRUCTION — the
  checker-narrows-its-own-domain shape. Fixing the text without fixing the domain leaves the
  next drift unguarded. The new domain comes from the engine's accepted value space.

## What this wave fixes

| id | Defect | Closed by |
|---|---|---|
| C1 | probe calls the working example broken | D2 + 102's outcome |
| C2 | API doc asserts a value space the engine refuses | D5 |
| C3 | that doc's gate cannot see the gap | D5 |
| C4 | prompt recommends the replaced technique by name | D3 |
| C5 | `ShaderView` cannot tell instanced from fullscreen | D4 |
| C6 | dry-run discards populations | D2 |
| C7 | `ScriptWriteResult` has no population channel | D1 |
| — | `orphan_keys` goes empty under 102 D2; its docstrings and `tools/script.py`'s hint rendering state 079 D5's silence | 102 owns this (its blast radius), listed here so 103 does not re-do it |
| C8 | populations cannot cross a tool call | D1 (settled) |
| C9 | `probe_render` has no vocabulary for the instanced case | **D6 — 103's, not 102's** |
| C10 | `add_pass`/`set_pass` silent on the `f1` trap | dissolved by 102 D5 |

## Gates

Broken and watched to fail before believed; the commit names the break.

- **The end-to-end one that would have caught C1**, and it must cover BOTH false verdicts. A
  `dry_run` over the shipped Entity Flock **asserts the population's COUNT appears** — not
  merely that a string is absent, which is a silence assertion that cannot tell you it missed.
  Two cases, because review found the second passes the narrow version:
  1. the flock as shipped → must not report "drives 0 uniforms" (`backend.py:406`);
  2. **the flock plus one constant scalar** → must not report "values UNCHANGED across t
     (STATIC)" (`backend.py:432`). MEASURED: 20000 entities orbiting, and the probe says
     nothing varies.
  Break by reverting the reporting path and watch each false verdict return.
- **The API-doc gate's domain** covers every value shape the engine accepts, enumerated from
  the engine rather than from `_stub_kind` — whose AST is parsed today and which dispatches on
  `moderngl.Uniform`, so a reserved key is outside it by construction. Break by adding an
  accepted shape the doc omits.
- **`sb_instanced` is not script-drivable** (D4a) — break by removing it from
  `ENGINE_DRIVEN_UNIFORMS` and watching a script drive it.
- **Both views report entity fields** for an instanced pass and none for a fullscreen one
  (D4). The fullscreen half is a real contrast pair, differing only in the property under
  test.
- NOT a gate this wave builds: `dry_run` leaves `pending_instances` untouched. It exists at
  `tests/test_instances_routing.py:357-391`, and review broke it and watched it fail with the
  right message. Inherited, not owed.

## Files touched

- `shaderbox/scripting/engine.py` — the statistics sink (D2), `ScriptProbe`'s population
  field (appended with a default; four positional construction sites).
- `shaderbox/engine_uniforms.py` — `sb_instanced` into `ENGINE_DRIVEN_UNIFORMS` (D4a).
- `shaderbox/copilot/capabilities.py` — `ScriptWriteResult`, `ShaderView`, `WorkingSetView`.
- `shaderbox/copilot/backend.py` — `_motion_verdict` (three branches, not one),
  `_apply_script_text`, `_probe_frame`/`_render_facts_for` (D6), `_format_uniforms`.
- `shaderbox/copilot/prompt.py` — the instancing block (D3).
- `shaderbox/copilot/prompt_context.py` — `vs_quad` beside `vs_uv`.
- `shaderbox/copilot/tools/script.py` — the `if not result.driven` branch (`:89`), the stub
  path (`:116-120`).
- `shaderbox/scripting/api_doc.py` — the false "all PLAIN PYTHON" text (D5).
- Tests: `test_script_api_doc.py`, `test_motion_verdict.py`, `test_script_dry_run.py`,
  `test_content_editing.py` (positional `ScriptProbe` sites).

## Out of scope

- A tool taking population DATA as an argument. Closed by D1, not deferred.
- Discoverability for humans (104).

## Open questions for the user

None blocking. One decision the builder takes with evidence rather than a maintainer answer:
**what "animating" means for a population.** `_uniform_changes` (`backend.py:385-391`) diffs
sampled scalars and cannot express a population's motion. Either the verdict stays silent on
populations and says so, or it names an observable (entity count over time, a positional
extent). Decide it in the wave rather than letting the new branch invent an answer.
