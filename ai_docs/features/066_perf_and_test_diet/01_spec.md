# 066 — perf and test diet

Three maintainer concerns, raised together after shipping v0.26.0, that turn out to share one
root plus one editorial problem:

1. App startup takes 3-5 seconds.
2. The test suite takes ~40-50 seconds.
3. Parts of the suite are theater — tests that assert the air and would pass with the feature
   deleted.

## Measured evidence (dev box, 2026-09-01, v0.26.0)

All numbers reproduced this session; re-measure before optimizing further.

- **Startup = 0.78s imports + 2.88s `App()`.** Of the 2.88s, **2.49s is
  `mgl.Context.program`** — 14 GLSL compiles across 8 documents, driven by the
  `document.render()  # warm-up` line at the end of `Document.load_from_dir`. Everything else
  is noise (glfw window 0.17s, PIL image decode 0.12s).
- **Imports: 0.78s**, of which openai (pulled by `copilot/llm/openrouter.py` at module top)
  is 0.23s and google-auth (pulled by `exporters/youtube.py`) ~0.11s. Neither is needed until
  a copilot turn / a YouTube upload.
- **Suite: 937 tests in ~39s, of which ~28s is `app` fixture SETUP** (78 tests x ~0.36s).
  Each setup copies ALL example documents into a tmp project and pays the same eager
  compile-every-pass cost — the startup bug, 78 times. No individual test is slow; the two
  worst calls are 0.9s (a repo-wide file scan and a subprocess import check).

## Decisions

- **D1 — compilation is lazy: a pass compiles when something actually needs its program, never
  at load.** Delete the load-time warm-up render. The consumers that need a program pull it
  on demand: `render()` already compiles per pass; `get_active_uniforms()` on a never-compiled
  pass triggers `compile()` (the save path already does exactly this for off-plan passes —
  that self-heal becomes the norm, not the exception). Uniform VALUES already load without a
  program (they sit in `uniform_values` until seed/bind) — pin that with a test.
- **D2 — first-frame compile cost is bounded, not moved.** Frame 0 currently renders every
  document for grid thumbnails; without care D1 just relocates the 2.5s stall into the first
  frame. Budget it: the current document renders on frame 0; every other document compiles
  lazily (its grid tile shows the stale wash until its first render — the vocabulary for
  "not rendered yet" already exists). No threads, no async — a per-frame budget (e.g. one
  document per frame) is enough and stays deterministic.
- **D3 — heavy SDKs import lazily.** `openai` and the google-auth stack move inside the
  functions that use them (first copilot turn / first YouTube auth). Budget: `import
  shaderbox.ui` under 0.5s. Guard with a test asserting `openai`/`googleapiclient` are absent
  from `sys.modules` after importing the app.
- **D4 — the `app` fixture seeds ONE document by default.** The full six-example seed becomes
  an opt-in fixture for the few tests that genuinely need the example library
  (`test_example_library`, examples-resolve). Everything else gets the starter document only.
  With D1 this turns ~0.36s setups into a few tens of ms.
- **D5 — the theater cull has a criterion, not a mood.** A test earns its place by having a
  FALSIFIER: a plausible regression that flips it red. Delete on any of: (a) it restates the
  implementation (asserting a constant against the same constant's source); (b) it tests the
  framework (pydantic defaults, stdlib, imgui bindings); (c) it duplicates a sibling's
  falsifier at lower fidelity; (d) it would still pass with the feature under test deleted
  (the deletion probe — spot-check by actually stubbing the unit out). The sweep walks every
  test module, files a keep/delete verdict per test, and the deletions land in ONE commit so
  the cull is reviewable and revertable as a unit.
- **D6 — budgets are pinned so they cannot regress silently.** After the wave: a startup
  budget check (import + `App()` under a threshold on the dev box, tracked in the spec — a
  hard CI gate only if it proves stable) and the suite wall-clock recorded in the spec's
  verification table. A rule with no gate is a wish.

## Stages

1. **Lazy compile (D1 + D2).** Delete the warm-up render; make `get_active_uniforms` compile
   on demand; budget frame-0 rendering; verify grid tiles wash-in correctly. Measure startup.
2. **Import diet (D3).** Lazy openai/google imports + the sys.modules guard test. Measure.
3. **Fixture diet (D4).** Starter-only default seed; opt-in full-library fixture. Measure the
   suite.
4. **The cull (D5).** Module-by-module keep/delete sweep with per-test verdicts; one commit.
5. **Pin the budgets (D6)** and close out (banner, spec verification table filled with the
   after numbers).

## Verification

Each check names its falsifier; numbers filled in as stages land.

1. **Startup on the dev box under 1.5s** (was 3.66s measured: 0.78 + 2.88). Falsifier: a
   stopwatch. Target is total to first interactive frame, not just `App()`.
2. **`import shaderbox.ui` under 0.5s and no openai/google in `sys.modules`.** Falsifier: the
   guard test.
3. **A loaded-but-never-rendered document still: saves without losing tuned uniform values,
   answers the copilot working set, and draws its panel.** Falsifier: any consumer that
   crashes or returns empty where it used to introspect the warm program.
4. **Suite under ~15s** (was 39.2s / 937 tests) with the same real coverage. Falsifier: the
   wall clock, and no deleted test had a live falsifier.
5. **First frame does not stall** — the per-frame budget keeps frame 0 under a normal frame
   budget with the 8-document dev project. Falsifier: a frame-time print on the dev box.

## Non-goals

- No threaded/async shader compilation, no pytest-xdist parallelism — reach for those only if
  the lazy/diet work leaves the budgets unmet (they add nondeterminism for cost that may
  vanish anyway).
- No behavior changes to rendering, persistence, or the copilot beyond compile timing.
