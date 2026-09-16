# 097 — The test-suite diet

The gate is too slow to run and too weak to trust. Both halves are the same problem: a suite
grown by accretion, where a test was added whenever a bug was fixed and none was ever removed or
checked for whether it can fail.

**The maintainer's constraint, and it is hard: `make gates` completes in 30 SECONDS or less**,
check + test + smoke together, on this box at `-n 8`. Not 45, not 35. Today it is 45-70s.

**Parallelism is NOT the lever and must not be raised.** The box has 24 cores and the suite runs
at `-n 8`; leave it at 8. The budget is met by removing work, not by spreading it wider. A
suite that only fits because it is sprayed across 24 workers is the same suite.

## Why this is worth a feature

Two measurements, both taken 2026-09-16.

**Cost.** 2734 tests, 159 files, ~26s in the test stage alone. Of the 25 slowest entries, `setup`
accounts for 24.6s against 15.5s of actual `call` time: the `app` fixture builds a real App, a
glfw window and a GL context per test, and 38 files use it. The suite is fixture-bound, which is
already written in the Makefile — what is new is that the fixture cost now exceeds the assertion
cost by 60%.

**Weakness.** 096's post-implementation review mutation-tested that feature's own gates and found
four of six passing a VERBATIM reintroduction of the defect they were written for, both
user-facing defects among them. It also found a test that was a tautology about `Canvas.__init__`,
a test whose target branch was never entered (proved with a canary), and a checker silently
covering seven of the eight invariants its ledger named. That is one feature's worth of tests.
Nobody has ever asked the same question of the other 158 files.

These are one problem. A test that cannot fail costs its full setup time and buys nothing, so
deleting it is free speed AND a more honest suite.

## The method, and it is not judgement

**A test earns its place by failing.** The tool is mutation testing, and the rule 096 paid for is
already in `conventions.md`: restore the ORIGINAL defect shape, not a nearby one, and apply it at
the layer that DECIDES the behavior rather than one below it.

Do NOT decide a test is useless by reading it. Read-and-judge is how the four dead gates got
written in the first place. A test is a deletion candidate when a mutation to the code it names
leaves it green — and the mutation has to be a fair one.

Explicitly NOT deletion criteria: a test being short, a test looking obvious, a test covering
something "surely already covered", a test with no assert on production state (some are
regression pins for a crash — the test IS "this does not raise").

## The waves

Wave order matters: measure before cutting, and the structural fix may make much of the cutting
unnecessary.

- **W-0 — the measurement harness. No test deleted.**
  A repeatable way to ask "which tests fail when I break X?" over the whole suite, and a per-test
  cost table (setup vs call, separated) so a cut can be aimed at cost rather than at count.
  Establish the baseline: total wall clock, the three stages, and the current distribution.
  Done when a single command answers both questions and its numbers are written into
  `02_progress.md`.

- **W-1 — the `app` fixture, the structural cost.**
  38 files build a real App per test. Ask, with numbers: what would a module-scoped or
  session-scoped App cost in isolation risk, and how much time does it return? `conftest.py`
  already documents why the fixture is per-test and why xdist groups exist (a process-global imgui
  font atlas torn down with the App that built it) — that constraint is REAL and must be honored,
  not deleted. If a shared App is unsafe, say so with the failure it produces and move on; the
  answer may be that fewer tests should need an App at all.
  **This wave may well deliver the whole 30s budget on its own, which is why it precedes any
  deletion.**

- **W-2 — the mutation census.** Run W-0's harness across the suite. Produce a table: test ->
  mutation attempted -> survived?  Every survivor is a CANDIDATE, not a verdict. Expect false
  candidates where the fair mutation was hard to construct; record those separately rather than
  padding the kill list.

- **W-3 — the cut.** Delete the candidates that survived a fair mutation, in reviewable batches,
  gate green between each. A deletion commit names the mutation the test survived. Where a test is
  weak rather than dead (it asserts something true but trivial), STRENGTHEN it instead of deleting
  it — 096 did exactly that three times and each rewrite caught a defect the original missed.

- **W-4 — the budget check, and the gate that keeps it.**
  Confirm 30s. Then make the budget enforceable, because a number in a doc is a wish: a check that
  fails when the suite exceeds it. Decide deliberately where it lives (the gate itself, or a
  separate target) and what it does on a slow box — a wall-clock assertion that fails on someone
  else's hardware is worse than none.
  **Break it before believing it**: add a deliberately slow test, watch the budget check fail,
  remove it.

## The other half: how the gate is USED

The 30s ceiling is worth little if the gate runs twenty times per session. The rule already
exists in `CLAUDE.md` and `dev_flow.md` — the full gate runs ONCE per unit of work, before the
commit; between edits it is the fast check or the single test covering the change — and this
session violated it repeatedly, running the full ~50s gate after one-line doc edits and after
formatting autofixes.

So W-4 also states the iteration loop in `dev_flow.md` in operational terms: which command to run
while iterating (the covering test file, typically under 2s), and that the full gate is a
pre-commit step. If a fast per-area target does not exist, add one.

## Constraints

- **`-n 8` stays.** Not negotiable, see above.
- **Never weaken a test to make it faster.** A test that stops asserting is a deleted test that
  still costs setup time — the worst of both. Delete it honestly or leave it.
- **No new coverage debt.** A defect currently caught must still be caught after the diet. W-2's
  census is the before-picture; re-run the relevant mutations after W-3 and show they still fail.
- 096's battery (`tests/test_canvas_ownership.py`, 18 tests) has every member broken and seen to
  fail as of 2026-09-16. It is a worked example of the target state, not a cut candidate.

## Cold start

1. Read this file, then `02_progress.md` and `git log --oneline`.
2. Re-measure before acting. Every number above was taken on 2026-09-16 and the suite changes.
3. Wave order W-0 → W-4. W-1 before any deletion, since it may make deletion unnecessary.
4. `conventions.md`'s mutation-fidelity bullet is the method. Read it before constructing a single
   mutation.
