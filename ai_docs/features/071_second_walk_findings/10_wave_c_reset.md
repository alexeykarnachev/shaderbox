# 071 W-C — Document Reset (#7 + D4 D5)

Parent: `01_spec.md § W-C`. Landed as commit `0e007a3` (plus the post-impl fix-ups recorded
below). The maintainer's words this wave answers: "we should introduce a global reset button for
the whole document ... user is not forced to think about individual feedbacks", "reset everything:
clocks, accumulated textures if any, scripts", "make sure that all this machinery is correctly
encapsulated. Don't scatter this logic across the whole engine."

## What landed

- **The command.** `CommandId.RESET_FEEDBACK` is `RESET_DOCUMENT`, label "Reset document", F6,
  scope DOCUMENT. The generated Help shortcuts table follows the spec. `projects/dev` held no
  override under the old name, so nothing to hand-fix.
- **One funnel.** `ProjectSession.reset_document(document_id)`: `Document.reset()` then
  `ScriptEngine.reset(document_id)`. `App.reset_current_document` forwards; the command table and
  the viewer button call the App method. `ScriptEngine.reset` (re-run `__init__` on the compiled
  class, no recompile, errors re-synced) existed with no production caller; this is its first.
- **A document clock.** `Document.time_origin`, on the process clock (`core.process_time()`, the
  expression `Pass.render`'s fallback always used), set at open and by `reset`. So a document
  opened five minutes into a session and a document just reset both start at `u_time` 0.
  `Document.render(u_time=None)` resolves to `live_time()`; the live tick hands each script
  `document.live_time(now)`. Export, the copilot probe and the dogfood harness pass an explicit
  `u_time` and never touch the origin. `ui.py`'s `now` is the process clock too, so `ctx.t` and
  `u_time` share one zero (they used to come from two clocks, glfw's and monotonic-since-import).
- **Videos follow the clock.** `Pass.render` already calls `MediaWithTexture.update(render_time)`;
  `Video.update` seeks backwards on a negative delta. No video-specific reset code exists.
- **The button.** Always drawn over the preview's top-left, "Reset", tooltip "Reset document
  F6". The `has_feedback` gate is gone and so is the property: its only production reader was
  that gate. The two tests that pinned it now assert `plan_passes(effective_graph()).feedback`
  directly, which is what the property computed.

## Found on the way

`tests/test_export_script_wiring.py` failed deterministically after `test_document_graph` +
`test_default_wiring` in that order, at HEAD before this wave (`red=113`, `red=254` — the
exported pixel was garbage). moderngl binds its process-wide default context once, to whatever
GL context is current at the first `get_context()`; a module-scoped standalone fixture context
had taken that role and been released, and every later `App` inherited the dead wrapper.
`App.__init__` now calls `moderngl.init_context()` right after `make_context_current`, so the
default is the window's whatever the process had current before. Pinned by
`test_gl_lifetime_guards.py::test_the_app_rebinds_the_default_context_to_its_window`, which
builds the poisoning order inside one module (a `stale_default_context` fixture requested before
`app`). Mutation: drop the `init_context()` call and the test reads the stale object back.

## Review history

**Pre-impl (one reviewer, opus; it judged the working-tree diff, which was already in place).**
Verdict LAND on three conditions, all taken: the tick's `t - time_origin` moved into
`Document.live_time(now)` so the arithmetic has one home; `time_origin` is set at open, not left
at 0, so "as if just opened" is literally true; the `has_feedback` deletion is recorded here
rather than contradicted by the brief. Two live blockers it found were the implementer's own and
fixed during the review: `process_time` used in `ui.py` before its import existed (caught by
`make check`), and two new tests that asserted how long pytest had been running (made hermetic:
the control sets the origin five seconds back, the script test hands the tick a literal clock).
Noted for W-F: `tutorial_body.html:253,687` still say "F6 (Clear canvas)". Noted, not W-C's: no
test exercises `ui.py`'s live loop, which is why the missing import got past `make test`.

**Post-impl, spec fidelity (opus, anchored to the maintainer's words).** Every W-C promise
landed; three landed differently and deliberately (the funnel takes no `now`, `time_origin` is
set at open rather than left at 0, `ScriptEngine.reset` re-runs `__init__` on the compiled class
instead of recompiling -- a recompile would have thrown away `last_good` and the per-key errors
for nothing). Encapsulation: exactly one funnel, nothing reaches past it; the two remaining
`reset_feedback` callers (export's cold start, the smoke harness) are the narrower verb on
purpose. Every wire mutation-tested by the auditor: cutting the live-time resolve, the tick
re-base, the engine reset, or the context re-init each turns exactly one test red. One FIX
finding: the committed `tests/test_document_reset.py` carried a 100-column line the format hook
rewrote, so `make gates` on the bare commit was red at `check` -- the working tree had the
formatted file (the gate had rewritten it after the add), and the fix-up commit carries it. Also
fixed there: `conventions.md` named the deleted `has_feedback` among `effective_graph`'s
consumers. Recorded, not changed: `ctx.frame` is the app's frame index and does not restart on
Reset (a document opened mid-session never saw it at 0 either); script module-level state
survives a reset where a recompile would restart it; the always-drawn button has no test beyond
smoke's draw path and manual step 1.

**Post-impl, code correctness (opus).** Same format blocker (fixed above). Three test-hygiene
findings taken in the second fix-up: the histories test seeded `_feedback` with the pass's own
live canvas, which the reset then released out from under the pass (a throwaway `Canvas` now);
`time_origin > 0` was vacuous since the origin is set at open (now `> before`); the stale-context
fixture called `init_context()` after `create_standalone_context`, which already installs its
context as the default, so it yielded a third wrapper (now yields the standalone one). One
comment in `core.py` still called the fallback clock "the live loop's" (reworded). Recorded, not
changed: `moderngl.create_standalone_context` installs itself as the process default
unconditionally, so a standalone context created AFTER an App steals the default back for the
App's lifetime; nothing in the app or the suite does that (fixtures build standalone contexts
before an App, never during one), and the durable fix is an App-owned `moderngl.Context` in
place of the per-frame `get_context()` at `ui.py:473`, which is its own change. Everything else
cleared: gc_mode and the imgui renderer come after the re-init, `Document` has one construction
site so the origin is always set, the origin never persists, every non-live render passes an
explicit `u_time`, `ScriptEngine.reset` handles no-script / compile-error / raising `__init__`,
Reset is correctly ungated mid-copilot-turn (a runtime restart like play/stop), the button and
tooltip pass the prose budget, six mutations each turn the expected test red.

## Manual verification (the maintainer, in the app)

1. Open the Radiance Cascades example; the Reset button sits at the preview's top-left on every
   pass, `paint` included. Falsifier: a pass with no feedback shows no button.
2. Let the warm light drift for a few seconds, press F6: it snaps to its t=0 position and drifts
   again from there. Falsifier: the light keeps its phase.
3. Paste the `Behavior` script from `tests/test_document_reset.py` into a document script
   (`u_n`, `u_t` declared in the shader), watch `u_n` count in the uniforms panel, press F6:
   `u_n` restarts at 1 and `u_t` at ~0. Falsifier: `u_n` keeps counting.
4. Bind a video to a sampler, let it play, press F6: it restarts from its first frame.
5. Wire `u_prev` on a pass that accumulates (069 W-G's paint recipe), draw, press F6: the canvas
   empties.
6. Export an image at `t=0` before and after a live Reset: byte-identical files.
