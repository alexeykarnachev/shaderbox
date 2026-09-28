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
  063 protects one property: a `dry_run` leaves the live document byte-identical.
  `validate_population` already runs before the write is skipped, so the statistics exist at
  that point. Reporting is not writing. **The isolation itself gets a gate**, because that
  is the invariant a reporting path could quietly break.
- **D3. The prompt gets an instancing BLOCK, not a deleted sentence.** `prompt.py` currently
  recommends the technique feature 100 replaced, naming a boids flock. Removing that is half
  the job; the model then needs the mechanism — `flat in` as the declaration, `pos`/`radius`
  reserved and clip-space, `vs_quad`'s range against `vs_uv`'s, the f4/i4/u4 dtype rule,
  C-contiguity, the `(N, components)`/`(N,)` shapes, the every-column-needs-a-field
  bijection, and 102's blend mode. Plus the watershed it must now state: a POPULATION of
  similar things → instancing; a small fixed vector of parameters → an array uniform.
- **D4. The copilot must be able to SEE that a pass is instanced.** `ShaderView` carries no
  entity fields, so an instanced pass reads as a fullscreen pass with odd inputs and
  `edit_shader` on it is blind.
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
| C8 | populations cannot cross a tool call | D1 (settled) |
| C9 | `probe_render` has no vocabulary for the instanced case | 102's outcome |
| C10 | `add_pass`/`set_pass` silent on the `f1` trap | dissolved by 102 D5 |

## Gates

Broken and watched to fail before believed; the commit names the break.

- **The end-to-end one that would have caught C1:** a `dry_run` over the shipped Entity
  Flock example reports a population and does NOT report "drives 0 uniforms / nothing
  animates". Break it by reverting the reporting path and watch the false verdict return.
  This is the gate the whole feature exists for.
- A `dry_run` over an instanced document leaves `pending_instances` untouched (D2's
  isolation, shared with 102).
- The API-doc gate's domain covers every value shape the engine accepts, enumerated from the
  engine rather than from `_stub_kind`. Break it by adding an accepted shape the doc omits.
- `ShaderView` reports entity fields for an instanced pass and none for a fullscreen one.

## Out of scope

- A tool taking population DATA as an argument. Closed by D1, not deferred.
- Discoverability for humans (104).

## Open questions

None.
