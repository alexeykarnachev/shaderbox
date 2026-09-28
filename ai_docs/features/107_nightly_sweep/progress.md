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
