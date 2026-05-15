# TODO — blockers & deferrals

Known issues parked for later, each with a **Trigger** — the condition under which someone working in
the repo should pick it up. NOT the ordered resumption backlog (that's the worklog top entry's
`open thread:` line); this is a "heads up, here's a landmine / a deferred fix" registry. **Grep this
file by `Trigger` before starting work in an area.**

`[BLOCKER]` vs `[DEFERRAL]`: a BLOCKER is a silent foot-gun that fires, stops work, and must be
resolved before proceeding. A DEFERRAL is deferred cleanup that the triggering work absorbs as
in-scope when the trigger fires.

What makes a good Trigger: it fires at a moment that *demands attention* —
- ❌ "before the next release" / "when we have time" / "eventually" — passes silently, never fires.
- ✅ "next time you edit `telegram_provider.py`" / "when you grep for `_loop.run_until_complete`" /
  "before plan-locking any feature spec that lists `core.py` in `## Files touched`" / "first user
  report of `<observable symptom>`".

If you can't name a moment that demands attention, the deferral is wrong-shaped — resolve it now or
don't file it. When a deferral resolves, delete the entry in the SAME commit as the fix (git history
is authoritative — no "Resolved YYYY-MM-DD" headers).

Format:
```
## [BLOCKER|DEFERRAL] <short title>
- **Trigger:** <when to pick this up>
- <context: what / why / where>
```

---

## [DEFERRAL] blocking HTTP in the render loop (ModelBox)
- **Trigger:** first user report of the ModelBox call freezing the UI, or next time you grep for
  `requests.post` / edit `modelbox.py`.
- `modelbox.infer_media_model` (`modelbox.py:52`) is synchronous `requests.post(timeout=600.0)`
  called from the render thread → blocks the GL frame on a long inference. The Telegram half of
  this deferral was resolved by feature 001 (worker-thread + mailbox in `exporters/telegram.py`);
  ModelBox needs the same shape (worker thread + result queue) or to be made async.

## [DEFERRAL] re-tighten pyright
- **Trigger:** when the cleanup backlog reaches a clean state (i.e. after `[DEFERRAL] split ui.py`
  or whatever audit pass clears the remaining type debt).
- The pyright pre-commit hook is `|| true`'d (`.pre-commit-config.yaml`) because of pre-existing
  type debt across `ui.py`, `media.py`, `core.py`, `modelbox.py`. Feature 001 cleaned up the
  share-tab `hasattr`-dispatch errors but other modules weren't audited. Drop the `|| true` once
  a sweep brings the total to zero. Don't add new errors in the meantime.

## [DEFERRAL] split `ui.py` (still ~1500-line god-class after feature 001)
- **Trigger:** next time you add a tab or a reusable widget to `ui.py`.
- `App` class spans ~1500 lines, 40+ methods (was 1778 before feature 001 extracted `tabs/share.py`).
  Remaining extractions: `widgets.py` (the `draw_*_details` family), `tabs/render.py`, `tabs/node.py`,
  `hotkeys.py`, `project.py`. New UI code should already go in the smallest plausible new module
  (`conventions.md ## Design decisions`). The `tabs/*.py` pattern is set by `tabs/share.py` (free
  `draw()` + optional `update()` + module-level `TabState`). Likely run as a high-blast-radius
  feature, not a drive-by.
