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

## Verification (closed 2026-09-01, dev box; commits 70b07f6 + d4786f3 + 87fa501)

1. **Startup under 1.5s — MET: 0.67s** (0.50 imports + 0.18 `App()`; was 3.66 = 0.78 + 2.88).
   Zero compiles at load.
2. **`import shaderbox.ui` under 0.5s, no heavy SDKs — MET: ~0.46-0.50s.** Guard:
   `tests/test_import_diet.py` (subprocess, pins openai + the google stack out of
   `sys.modules`; mutation-tested by the post-impl review).
3. **A never-rendered document still saves / answers the working set / draws its panel —
   MET.** Pinned by `tests/test_lazy_compile.py` + the reworked
   `test_document_save_preserves_values.py` (incl. the broken-source carry-forward). The
   post-impl review found and closed the two real gaps here: seeding now rides the lazy
   compile (the panel indexes `uniform_values` directly), and a foreign-canvas probe/export
   no longer consumes the first-render budget.
4. **Suite under ~15s — NEAR: ~16s / 925 tests** (was 39.2s / 937). The remaining floor is
   the 78 `app`-fixture setups at ~0.12s each (~9s): a real glfw window + imgui context per
   test, not compiles. The next levers — a shared-window fixture or xdist — are this spec's
   non-goals; take them up only if the suite grows past tolerance again. The cull removed 18
   theater tests (2% of the suite — far denser than the framing above guessed; per-test
   verdict letters in commit d4786f3) and replaced one with a REAL guard test
   (`test_render_failure_feedback.py`, mutation-verified: the stale-artifact guard in
   `tabs/share.py` had only pseudo-coverage).
5. **First frame does not stall — MET.** With the six examples loaded (10 passes), every
   program compiles within the first ~6 frames; worst warm-in frame 76ms (the five-pass
   bloom document's one budget slot), steady frames 6-9ms. Old behavior: one 2.5s stall. A
   tile waiting for its first render draws the stale wash (grid + Examples popup).

## Landed reality vs the outline

- **D4 landed simpler than written:** no opt-in full-library fixture exists — the example
  library loads from resources regardless of the project seed, so `test_example_library` and
  examples-resolve never needed the full seed. The four tests needing a second project
  document use `tests/conftest.py::seed_extra_document`.
- **D3 landed stronger than written:** the google stack moved whole into
  `exporters/youtube_api.py` (one module seam, SDK exceptions mapped to typed errors) rather
  than per-function imports; the two seams are recorded in
  `conventions.md ## Design decisions` and gated by `test_import_diet.py`.
- **D6:** the budgets live in this table plus the two structural gates (the import-diet test;
  the lazy-load contract tests). A wall-clock CI gate was not added — the suite number moved
  ±3s between runs on this box depending on load, exactly the instability the decision
  anticipated.

## Non-goals

- No threaded/async shader compilation, no pytest-xdist parallelism — reach for those only if
  the lazy/diet work leaves the budgets unmet (they add nondeterminism for cost that may
  vanish anyway).
- No behavior changes to rendering, persistence, or the copilot beyond compile timing.
