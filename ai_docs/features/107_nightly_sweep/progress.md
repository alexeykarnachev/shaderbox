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

## W-0 — copilot slice DONE

25 breaks, 22 caught, 1 vacuous, 2 false trails. **The copilot suite is strong** —
the density of real catches here is the finding, and it argues against a broad
"strengthen the tests" wave.

VACUOUS: `copilot/agent.py::run_turn`'s `time_budget_hit` flag. Setting it to `False`
while leaving the `break` in place still passes
`test_turn_time_budget_forces_a_final_reply`. The turn ends either way because the
fixture scripts only one tool round, so "the budget forced an early stop" and "the
script ran out" are indistinguishable. A reader would believe the wall-clock budget is
pinned; only the one-round-ends-after-one-round shape is.

FALSE TRAILS (do not re-litigate):
- `document.py::canvas_size_for`'s output special-case — equivalent under the fixture,
  because `TargetConfig().scale` defaults to 1.0 and the test never sets another. A
  scaled non-output pass would likely expose a real gap; not probed.
- `shader_lib/parser.py::top_level_names` depth guard — caught by the suite, but the
  specific nested-`else if` assertion passes without it, since the regex does not match
  that syntax at any depth.

## PROCESS FAILURE, mine, and it recurred after I had already logged it

I ran four file-mutating W-0 agents against ONE shared working tree. The copilot agent
reported the hazard unprompted: files outside its slice showed foreign mutations
appearing and disappearing mid-run, and it could not assert a clean tree at exit
because siblings were still writing.

I had already written this exact lesson into this file earlier tonight, after a
reviewer's first measurements were taken against a probe agent's debris. I logged it
and then did it again at four times the scale.

Consequences seen: a stop-hook caught me reporting status while a sibling's mutation
was mid-flight; one file showed as modified with byte-identical content (an mtime
touch); `shaderbox/watch.py` is currently carrying a live unrestored break from an
agent that is still running, and I am deliberately NOT restoring it, because clearing
a tree under a running mutation corrupts its measurement in the other direction.

**The rule for every future mutating wave: one git worktree per agent, created before
launch.** A restore-and-verify discipline protects an agent's own sequence and does
nothing for a concurrent reader.

## W-0 — UI/app slice DONE

**13 breaks, 13 caught, ZERO vacuous.** Several the agent expected to be hollow were
caught, including the button-tier AST detector, the command-chord prose gate, and an
absence check that fires even on a planted call inside a comment.

Taken with the copilot slice's 22-of-25, **the suite is in far better shape than the
premise of this sweep assumed.** W-1 should be a short, named list rather than a broad
hardening pass.

W-2 item confirmed by the main session: `tests/test_graph_view.py` carries EIGHT dead
rig helpers as one cluster — `_open_graph`, `_click_at`, `_let_the_double_click_lapse`,
`_press_key`, `_close_graph`, `_hover_fields`, `_drag_node`, `_park`. The agent listed
seven; `_park` looks live at four occurrences but is called only from `_hover_fields`
and `_drag_node`, which are themselves dead, so the whole cluster goes. Commit db68cfb1
("Retire the imgui canvas the library replaced") moved gesture coverage to
`test_graph_canvas_gestures.py` and left the rig behind.

FALSE TRAILS (do not re-litigate):
- `test_button_tiers.py`'s AST detector matches the literal alias `imgui`, so
  `import imgui_bundle.imgui as ig` would evade it. Every file uses
  `from imgui_bundle import imgui`, so it is equivalent under the actual convention.
- `test_uniform_panel.py`'s docstring claims to pin "the row's three states" while the
  tests drive only the resolver. A documentation overclaim, not a vacuous gate — the
  resolver tests are real.
- A group-tint assertion that looked like a stale-scope fixture: traced and disproved,
  `view.scope` is reset to root after `dissolve_group`. Retracted by instrumentation.

## W-0 COMPLETE — all four slices

| slice | breaks | caught | vacuous |
|---|---|---|---|
| copilot | 25 | 22 | 1 |
| UI / app | 13 | 13 | 0 |
| editor / intel | 63 | 59 | 4 |
| engine / core | ~90 | ~76 | 14 |

**The suite is strong where it was written against a spec and weak where a fixture
happens to satisfy the assertion by accident.** The engine/core slice holds most of the
yield; the UI slice holds none.

### The recurring sub-shape, which is NOT what the spec predicted
The spec named "asserts on source text". That is real (3 instances) but the dominant
shape in the inventory is different and worse: **a fixture that cannot distinguish the
correct implementation from the broken one, because the case it builds happens to give
both the same answer.** Examples:
- `watch.py` root-path match: `path == root_path` -> `i == 0` survives, because no
  fixture ever builds a multi-source compile unit, so index 0 always IS the root. The
  file's own docstring narrates this exact historical bug and the test named for it does
  not construct the case.
- `pass_graph.py::rank_layout` group anchor: the test's own comment names the falsifier
  ("sort a column by strip order alone"), and that falsifier survives because the
  fixture's group members are already strip-adjacent.
- `instanced.py::_extent_expression`: dropping `/ u_aspect` survives.
- `canvas_choice_groups` dedup: the fixture never produces a duplicate.

### W-1 ITEMS LANDED
1. `theme.py::resolve_palette_refs` alpha override — my own, one commit old, no test at
   all. Two fields (`wire_outline`, `pin_ring`) would have drawn opaque. Fixed + gated;
   the exact mutation now fails.
2. `test_keymap_disjoint.py` — BOTH the zero-iteration loop and its source-string
   sibling. Now drives `_apply_editor_settings_to` with a recording spy. Both breaks
   caught, including the one that defeated the previous fix.

### W-1 REMAINING, ranked by what a defect would cost
- `watch.py` root-path (a wrong file reloads a document — user-visible, and a KNOWN
  historical bug with a test named for it)
- `copilot/backend.py::rename_document` never re-reads from disk, so a rename that does
  not persist passes
- `scripting/engine.py::_write_one` last-good fallback (freezes at the wrong value)
- `pass_graph.py::rank_layout` group anchor
- `ui_models.py::save` orphan sweep, masked end-to-end by an earlier unbind
- the turn time-budget flag (copilot slice)

### FALSE TRAILS — do not re-litigate
- `_resolve_behavior_class` fallback scan and `_upload_instances`'s `program is None`
  guard are UNREACHED by any fixture, not vacuous checks. Coverage gaps.
- `document.py::begin_frame` idempotency: a second independent guard masks it, and the
  test's own docstring says so.
- Two `ui_models.py` carry-forward guards are individually equivalent and correctly fail
  when broken together — genuine defence in depth.
- `SILENT_KEY_FAIL_REASONS` drop is caught by `test_script_warn_path.py`, out of slice.
- `test_button_tiers.py`'s AST detector matches the literal alias `imgui`; every file
  uses that spelling, so it is equivalent under the actual convention.

## W-1 — FIVE LANDED, each verified by re-running the break that defeated it

| # | what | the break that used to pass |
|---|---|---|
| 1 | `theme.py` alpha override | drop the override branch; two fields draw opaque |
| 2 | `test_keymap_disjoint` (two tests) | make the bind unreachable, keep the string |
| 3 | `watch.py` root-by-path | revert to `if i == 0` -- the historical bug |
| 4 | `rename_document` persistence | delete `_save_ui_document` |
| 5 | `_write_one` last-good freeze | freeze at the live value |
| 6 | `rank_layout` group anchor | sort a column by strip order alone |

### What the fixtures were actually wrong about
Not one of these was a missing assertion. Every one asserted the right property against
a case where BOTH implementations give the same answer:
- `side` sorted after both group members, so strip order and group order agreed.
- `u_b` started at 0.0 with no last-good, so "freeze at last-good" and "freeze at live"
  were the same number.
- No compile unit had a source ahead of its root, so index 0 always WAS the root.
- The rename was read back off the object the method had just mutated.

### Three things learned by running, not reading -- each cost an attempt
- `rank_layout` derives strip order from the WIRING via `plan_passes`, ignoring its own
  `names` argument. Two attempts to interleave a pass by reordering the dict or the
  argument changed nothing.
- The script engine recompiles on an explicit `reload`, never by polling mtime. Writing a
  new script file and ticking runs the OLD behaviour.
- A clean tick restores last-good over a live edit, so the "move the live value" step has
  to come after the last good tick, not before.

### REMAINING
- the turn time-budget flag (copilot slice)
- `ui_models.py::save` orphan sweep, masked end-to-end by an earlier unbind
- W-2: `theme.py::_muted` (dead), `tests/test_graph_view.py`'s eight-helper rig cluster
- W-3: point `pass_settings.py:140` and `tabs/uniforms.py:61` at `instanced_pass_keys`
- W-R: `dev_flow.md`'s Module map

## W-R — live facts: NOTHING TO DELETE. Wave closed empty.

The presence scan flagged `dev_flow.md`'s Module map as the harness's highest-risk zone
for silent staleness. Checked rather than trusted, and it is accurate:

- all 79 cited `.py` files resolve (two live outside `shaderbox/` -- `scripts/dogfood/`
  and a skill directory -- which a naive glob reports as missing);
- all 8 cited `file::symbol` pairs resolve.

Every countable figure in `conventions.md` and `roadmap.md` turns out to be FROZEN
HISTORY, which the rule says to keep:
- `2028 tests green` narrates a specific mutation test that happened, not the suite's
  size now (it is 2226);
- `2734 tests to 1968` records what feature 097 DID;
- `~1024 slots` is a hardware constant;
- `368 lines when ... 423` is the worked example inside the passage that TEACHES this
  rule. Deleting it would remove the lesson about deleting live facts.

The one `currently` outside the dated banner is "as the code currently is", a rule about
comments rather than a status claim.

**The scan flagged the Module map on SHAPE -- it is a closed list of symbols, which is the
risky form -- and the wave's job was to find out whether the risk had materialised. It has
not.** Recording that is the wave's output; deleting an accurate doc because its shape is
risky would have been the error.
