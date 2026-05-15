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

## [DEFERRAL] split `ui.py` / `app.py` further
- **Trigger:** next time you add a tab or want to extract a remaining chunk.
- Progress: 1778 (single ui.py) → 1508 (feature 001, `tabs/share.py`) → after feature 002:
  `app.py` 373 + `ui.py` 294 + `tabs/{node,render,share,share_state}.py` 398 +
  `widgets/*.py` 547 + `popups/*.py` 166. Remaining extractions inside `app.py`: `hotkeys.py`
  (the ~40-line hotkey block currently in `ui.py:update_and_draw`), `project.py` (the
  `@property` paths + `save` / `open_project` / `delete_current_node` lifecycle). `App` is the
  state-holder; widgets/popups/tabs take `app: App` directly (no `AppContext` wrapper).

## [DEFERRAL] headless smoke test
- **Trigger:** next time a refactor lands in `ui.py` / `widgets/*.py` / `popups/*.py` and you
  want a faster verification than the 11-step manual UX sweep — or first time a regression slips
  through manual testing.
- Build a `scripts/smoke.py` that: creates `App` against `projects/dev/` with an invisible glfw
  window (`glfw.window_hint(glfw.VISIBLE, glfw.FALSE)`), runs ~200 frames of `update_and_draw(app)`,
  asserts no exception + invariants (`not (app.is_node_creator_open and app.is_settings_open)`,
  `app.current_node_id == "" or app.current_node_id in app.ui_nodes`, no released textures in
  `uniform_values`). Exit 0 on success. Wire into `make check` or a separate `make smoke`. ~60
  lines. Catches import errors, callback dispatch failures, popup state machine crashes — the
  bulk of refactor regressions. Doesn't catch visual bugs.

## [DEFERRAL] in-app replay mechanism (debug)
- **Trigger:** next time you hit a multi-step bug that's painful to reproduce manually, or when
  you want to share a repro with future-you.
- Add a "Debug → Replays" UI surface that lists JSON files from `replays/`, plays them back by
  injecting synthetic actions into the existing hotkey/button code paths. DSL (rough): list of
  `{frame, action: hotkey|click|assert, ...}` dicts. Intercept points: imgui io-state setting
  for hotkeys (reuse the hotkey block in `ui.py:update_and_draw` as-is — currently ~`ui.py:86-124`),
  thin `replay_aware_button(label)` wrapper for clicks (or globally wrap `imgui.button` in
  replay mode). Not a test framework — for manual debugging / shareable repros. The smoke test
  above is the right tool for actual regression testing. Probably worth a small feature spec
  before building (touches imgui boundary, adds UI surface).
