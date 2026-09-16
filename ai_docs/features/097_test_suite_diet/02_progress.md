# 097 — The test-suite diet: the log

What actually happened, wave by wave. The plan is `01_spec.md`. **On resume, read this file and
`git log --oneline` first.**

## Baseline, measured 2026-09-16 (before any wave)

Box: 24 cores, `-n 8` (which stays at 8 — the budget is met by removing work, not spreading it).

| stage | wall clock |
|---|---|
| check | 8.2s |
| test | 26.0s (2734 tests, 4 skipped, 159 files) |
| smoke | 2.6s |
| **`make gates`** | **45-70s across runs** |

**Target: 30s for all three together.**

Shape of the test stage, from `--durations=25`: `setup` accounts for **24.6s** against **15.5s**
of `call` time in those 25 entries. The two slowest calls are
`test_region_system_is_gone.py::test_every_child_hosting_a_focusable_widget_blocks_tab` (6.4s) and
`test_editor_ffi.py::test_every_documented_builtin_draws_in_the_builtin_slot` (5.6s); everything
below them is dominated by fixture setup, not by assertions. 38 of 159 test files take the `app`
fixture, which builds a real App, a glfw window and a GL context per test.

**That is the headline: the suite is fixture-bound, and the fixture now costs more than the
assertions do.** W-1 exists because of this number, and it precedes any deletion.

## Why the weakness half is in scope

From 096's post-implementation review, same day: three reviewers mutation-tested that feature's
own gates. Four of six passed a VERBATIM reintroduction of the defect they were named for — both
user-facing defects among them. Also found: a test that was a tautology about `Canvas.__init__`, a
test whose target branch was never entered (proved with a canary), and an invariant checker
covering seven of the eight invariants its own ledger named.

One feature's tests, three rounds of review. The other 158 files have never been asked.

## W-0 measurement harness — NOT STARTED
