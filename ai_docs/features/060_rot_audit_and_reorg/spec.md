# 060 — rot audit & architectural reorg

**Status:** spec (audit in flight)

## Goal

The project sat untouched for a while. Establish, with evidence, what has rotted and what the
module/package layout should become — then fix what the audit proves.

Two halves:
1. **Rot** — anything that decayed while nobody was looking: broken gates, dead code, doc drift,
   duplication, stale deps, tests that no longer verify what their names promise.
2. **Architecture** — is `shaderbox/`'s module/sub-package shape still right at 26.5k LOC, or has
   it accreted past its boundaries? Concrete reorg proposal, not vibes.

## Ground truth established before the swarm (2026-08-20)

- `make check` — **GREEN** (ruff + pyright, 0 errors, 10 known upstream stub warnings).
- `make test` — **RED, and it was red silently.** 6 failed + 1 error.
  - Every test file passes **in isolation** (all 60 files, one-by-one: green).
  - The failure is **cross-file state pollution**, not a logic bug.
  - Minimal repro: `pytest tests/test_node_ops.py tests/test_revert_executor.py -p no:randomly`
    -> `6 failed, 8 passed, 1 error`.
  - Mechanism: a module-scoped `gl_ctx` fixture calls `ctx.release()` on a standalone context,
    which breaks the shared X connection on `:1`; `test_revert_executor.py`'s `pytest.mark.forked`
    tests then die with `RuntimeError: Cannot detect window with OpenGL support` /
    `X connection to :1 broken`. `test_node_ops.py` is the first module that mixes BOTH the
    glfw `app` fixture and a standalone `gl_ctx` fixture.
  - Same standalone-release pattern also in `test_cross_project_tools.py`,
    `test_uniform_seed_save.py`, `test_script_engine_gl.py`.
  - Aggravating: the crash kills the pytest process before it can print its own summary
    (`-rf`, `--junitxml` both produce nothing) — which is WHY this stayed invisible.
  - `todo.md` already carries a `[VERIFY]` entry predicting exactly this class on this box.

## Out of scope

(filled at plan-lock, after the swarm reports)

## Design decisions

(filled at plan-lock)

## Files touched

(filled at plan-lock)

## Manual verification

(filled at plan-lock)

## Open questions for the user

(none — maintainer delegated the full flow end-to-end, no approval gates)
