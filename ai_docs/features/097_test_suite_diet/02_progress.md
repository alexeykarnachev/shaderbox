# 097 — The test-suite diet: the log

What actually happened, wave by wave. The plan is `01_spec.md`. **On resume, read this file and
`git log --oneline` first.**

## Baseline, re-measured 2026-09-16 at the start of the work

Box: 24 cores, `-n 8` (which stayed at 8).

| stage | before | after |
|---|---|---|
| check | 8.5s | 8.5s |
| test | 41.5s | ~15.5s |
| smoke | 3.1s | 3.1s |
| **`make gates`** | **45-70s** | **27.2s** |

Suite: 2734 tests / 41151 lines / 162 files, down to 1968 / 37374 / 146.

The spec's baseline said the test stage was 26s; it measured 41.5s on the day the work started,
which is why the spec says to re-measure rather than trust its own numbers.

`make gates` is green at 27.2s, `rc=0` captured unpiped. Re-measure rather than trusting these:
the numbers above moved three times inside this feature alone.

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

## W-3 — the cut, by reading rather than by measuring

The waves above bought the budget; the maintainer then asked for the suite to get SMALLER, and
said plainly that a mutation sweep was the wrong instrument for deciding it. So the rest was
decided by reading each file and asking one question: **what does a USER lose if this check
disappears and the code breaks?** Silent data corruption, a leaked GPU object, a document opening
at the wrong size, the wrong file saved -- keep. A misspelled label, a doc heading, a report
nobody ships -- the person running it sees it immediately.

Four categories came out, 766 tests and ~3800 lines in total.

**Tests that pin a STRING rather than behaviour.** `test_craft_prompt` was the proof: replacing
the whole system prompt with a bare list of the 27 phrases it pinned -- 15658 characters down to
532, every instruction to the copilot gone -- left all twelve green, as did inverting a rule's
meaning while keeping its marker words. Also cut: `test_ui_prose_budget` (918 lines, 644 tests,
one word-count per call site), `test_prose_spelling`, `test_roadmap_shape`, and later the two
docstring-shape checks in `test_script_api_doc`.

**Tests of tooling that does not ship** -- the category the maintainer called the worst of it.
The dependency runs one way: dogfood imports shaderbox, shaderbox never imports dogfood, and no
production module mentions the tutorial. 2019 lines across the five `test_dogfood_*` files and
`test_tutorial_build`, then a second pass for the remainder: the Khronos refpage parser's tests,
the importability check over `.claude/skills`/`scripts`/`dogfood`, the control-byte sweep over
every committed text file, and the one test pinning the copilot registry to the dogfood harness's
coverage denominator.

**Checks that cannot fail.** `test_credential_redaction` asserted `card.gate_input == ""` under
the message "the typed-secret buffer was not cleared on answer" -- the fixture never set the field
and its default is already `""`. `test_integration_defaults_mirror_config_defaults` compared twelve
fields against the config defaults they are DECLARED as. One test asserted that a function emitting
`spec.label` per spec emits `spec.label` per spec. One pinned `COMMAND_SPECS` to an ASCII diagram
in a closed feature's spec.

**Checks that read a source file and search it for a substring** -- four in `test_menus`, one in
`test_project_management`.

## The GL race the reuse introduced, found late

About one run in eight went red, in a different test each time, always as a WORKER CRASH rather
than an assertion -- once in a test that touches no GL at all. It was W-1's doing.

moderngl's `gc_mode="auto"` frees a dropped GL object from its `__del__`, so the release runs on
whatever thread Python collects on. The app only ever collects on its GL thread, which is why auto
is right there. Under one App per worker its objects instead survive until a collection that
xdist's receiver thread can trigger, and freeing a GL object on a thread with no current context
segfaults the worker. The fixture now uses `context_gc`, which queues the drops, and frees the
queue itself at the head of each test where the window is current. 20 consecutive runs green.

The first attempt put that collection at TEARDOWN, where a module using `gl_ctx` has left its own
context current -- one flake became 367 teardown errors. And the flake was called "pre-existing"
after a single green re-run before it was investigated: a re-run is not evidence, and nothing else
was changing the tree.

## What was measured and deliberately NOT cut

`test_modal_chrome.py` (208) are 31% of the suite's COUNT
and 0.65s of its wall clock — measured by deselecting both. Each parametrized case is one call site,
and the id is what names the offending site on failure. Collapsing them would move the count without
moving the clock, and would trade a named site for a list in one assertion message.

The remaining shape of the stage is flat: `test_graph_view`'s 27 tests each drive real imgui frames
at ~0.3s and no single test dominates. Importing `shaderbox.app` costs 0.65s and every worker pays
it; that is the codebase's import graph, not the suite's.
