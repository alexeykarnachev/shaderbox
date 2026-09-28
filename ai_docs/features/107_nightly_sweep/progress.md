# 107 nightly sweep — progress log

A log of what happened, appended after each wave. Not a plan; the plan is
`01_spec.md`. On resume, read this file and `git log` first — they are the truth
about where the work stopped.

## Phase 1 — presence scan (in progress)

Six scans, one question each, presence + one example only (no inventories).

- dead code: **PRESENT**. `theme.py::_muted` has no call sites — verified by grep,
  only its own `def` remains. It became dead when 106 deleted the four
  `GRAPH_PORT_*` tokens that were its only consumers; the deletion left the
  helper. Also `tests/test_graph_tab.py::_fitted_window`, same shape.
  Tooling note: `vulture` is NOT installed and pyright's config does not enable
  unused-symbol reporting, so the scan was grep-based. `ruff F401,F841` is clean.
- repeated decision: **PRESENT**. "is this pass instanced" = truthy
  `entity_fields`, factored into `widgets/pass_graph.py::instanced_pass_keys`
  and then re-spelled inline at `popups/pass_settings.py:140` and
  `tabs/uniforms.py:61`. All three AGREE today, so it is drift risk rather than
  a live bug — but that function's own docstring says it exists because an
  inline version was deleted once and the suite stayed green.
- vacuous checks: **PRESENT**, and verified independently by the main session.
  `intel/document.py:41` — replacing `return entry.index, False` with
  `return build(), False` makes the cache rebuild on every hit while still
  reporting `changed=False`, and `test_the_cache_entry_belongs_to_its_handle`
  stays green. The test reads only the boolean, never the returned index.
- live facts: **PRESENT**. `dev_flow.md`'s Module map (~800 lines) is a closed-list
  inventory citing dozens of symbols and paths as current-tense claims. Spot-checked
  entries still resolve, but nothing gates the section against the code it describes.
- comment history-narration: **ABSENT**. Searched repo-wide for narration markers;
  every hit is a single-sentence reason naming a current constraint, often with a
  measurement. That is the convention working, not a finding.
- repeating defect class (from 40 commit BODIES): **PRESENT, and it is the night's
  sharpest finding.** Six instances in five days of ONE shape: *a gate that asserts on
  SOURCE TEXT or on an import-time copy rather than driving the runtime object, so it
  reports green while the mechanism underneath is missing.* Instances: b6db29f5
  (three `inspect.getsource` gates), 3d9727b2 (wiring ungated), e7e7accd (seam complete
  on one side only), 7645da03 (in-process palette swap is a no-op), 861473ea (same shape,
  second palette), 3cd8a2a2 (gate read the repair, not the artifact).

### Verified independently by the main session
- `intel/document.py:41` cache gate is vacuous — break confirmed, suite green.
- `theme.py::_muted` has no call sites — grep confirmed.
- `tests/test_keymap_disjoint.py:303` is a SECOND live instance of the class:
  `assert "editor.bind(key, index, leader=True)" in source`. Replacing the call with
  `... if False else None` keeps the searched string, kills the binding, and all 14
  tests pass.

### A blanket ban on `inspect.getsource` would be the WRONG instrument
Four files use it. Three are AST-based structural checks (`test_button_tiers.py`,
`test_modal_chrome.py`) or pair the source check with a behavioural assertion
(`test_probe_clock_and_turn_end.py` asserts `_facts_for(None)` really renders at 0.0).
Those are correct. The dangerous shape is narrower: a POSITIVE presence assertion on
source text standing in for behaviour. Absence assertions ("this pattern is gone") are
also fine — a deleted thing has no runtime to drive.

## Spec written and checked

`01_spec.md` passes its structural check. Three real violations it caught, all mine:
live facts (a line count and a file size) inside a spec whose own wave deletes live
facts; a missing re-measure instruction; and coverage lines carrying counts.

Two of its complaints were FALSE POSITIVES worth knowing about, because both are
multi-line regex matches that a per-line grep cannot show: "Phase **1** ... its
**finding**s" and "W-**2** — **Dead** code" both match a defect-count pattern that is
looking for "delete the 6 dead symbols". Reworded rather than argued with.
