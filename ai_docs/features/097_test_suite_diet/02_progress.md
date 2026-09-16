# 097 — The test-suite diet: the log

What actually happened, wave by wave. The plan is `01_spec.md`. **On resume, read this file and
`git log --oneline` first.**

## Baseline, re-measured 2026-09-16 at the start of the work

Box: 24 cores, `-n 8` (which stayed at 8).

| stage | before | after |
|---|---|---|
| check | 8.5s | 8.5s |
| test | 41.5s | 16.8s |
| smoke | 3.1s | 3.1s |
| **`make gates`** | **45-70s** | **28.2s** |

The spec's baseline said the test stage was 26s; it measured 41.5s on the day the work started,
which is why the spec says to re-measure rather than trust its own numbers.

`make gates` runs 28.2s across three consecutive runs, `rc=0` captured unpiped.

## W-0 / W-1 — the fixture was the whole story

Of 261 CPU-seconds in the test stage, **202.7s was `setup`** against 56.3s of `call`. `App.__init__`
measures 0.151s: glfw's `create_window` is 0.073s of it and reloading the six shipped example
documents most of the rest. 38 files took the `app` fixture, which built one per test.

What forced one App per PROCESS was never one App per TEST — the imgui font atlas dies with the App
that built it, and nothing else does. `App._init` is the project-switch path the running app already
takes. Reusing one App through it costs **0.050s against 0.151s**, and the stage went 41.5s -> 21.5s.

The reuse turned four latent project-switch defects into failures. Each was reproduced on a REAL
switch with a probe (not only in the suite), and each was re-checked by removing the fix and
watching the probe report the old value:

| field | what carried across a switch |
|---|---|
| `App.copilot_turn_active` | the incoming project locked against a turn that could not finish |
| `ProjectSession._copilot_working_set` | the agent handed document ids the new project lacks |
| `App.graph_views` | a document's pan/zoom/selection followed its id into the next project |
| `CopilotSession._released` | latched on the first switch; the next sentinel would exit a live worker |

A fifth failure was a crash: two copilot workers generating pydantic JSON schemas at once took the
interpreter down. An args model's schema is fixed at import, so it is computed once per model now.

The fixture restores a SNAPSHOT of the App's fields rather than resetting the ones a failure happens
to name. That list was tried first and grew to five before the shape became obvious.

## W-2 — four scans that ran once per test

After W-1 the split was 35s setup / 37s call: assertion-bound, not fixture-bound. The cost had moved
to individual tests repeating a whole-repo scan per case.

| test | was | now | what changed |
|---|---|---|---|
| `test_editor_ffi` builtin slot | 5.22s | 0.06s | one Editor for all 171 names, not one each |
| `test_region_system_is_gone` | 5.55s | 0.64s | module functions cached; 1560 AST walks -> 3 |
| `test_document_dir_layout` | 1.76s | 0.58s | one parse for every literal, not one per literal |
| `test_model_kwargs` | 2.02s | 1.35s | the construction sweep cached across both tests |

Each was re-proven by reintroducing the defect it names and watching it fail.

**Two mutation attempts passed and the gate was right, not dead.** One unflagged a container whose
only widgets are `button` and `checkbox`, which `_NAV_ONLY_FOCUSABLE` excludes by name and by
measurement. One landed a `sed` on the line above the flag. A mutation that misses its target reads
exactly like a dead gate, which is the trap the spec's own method is meant to avoid — so a survivor
is checked against the gate's stated domain before it is called a survivor.

## What was measured and deliberately NOT cut

`test_ui_prose_budget.py` (644 tests) and `test_modal_chrome.py` (208) are 31% of the suite's COUNT
and 0.65s of its wall clock — measured by deselecting both. Each parametrized case is one call site,
and the id is what names the offending site on failure. Collapsing them would move the count without
moving the clock, and would trade a named site for a list in one assertion message.

The remaining shape of the stage is flat: `test_graph_view`'s 27 tests each drive real imgui frames
at ~0.3s and no single test dominates. Importing `shaderbox.app` costs 0.65s and every worker pays
it; that is the codebase's import graph, not the suite's.
