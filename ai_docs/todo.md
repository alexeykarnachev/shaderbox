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

## [DEFERRAL] two near-identical sticker models — also gates re-tightening pyright
- **Trigger:** next time you touch the share tab (`ui.py draw_share_tab`) or `telegram_provider.py`.
- `TelegramShareableMedia` (`telegram_provider.py:16`) and `ShareableMedia` (`sharing.py:22`) both
  model "video|image + preview_canvas + log_message". `draw_share_tab` does `hasattr`-driven dispatch
  over them (`ui.py:1180,1192,1215,1232,1247,…`) and the provider `cast`s `TelegramShareableMedia` →
  `ShareableMedia` (`telegram_provider.py:193,272`). Collapse to one model behind a real interface.
  (Backlog item 4 — see worklog `open thread:`.) **When this lands: the ~16 pyright
  `reportAttributeAccessIssue` errors in `ui.py` go away → drop the `|| true` from the pyright hook
  in `.pre-commit-config.yaml` so it's blocking again** (see `conventions.md ## Known quirks`).

## [DEFERRAL] blocking asyncio / blocking HTTP in the render loop
- **Trigger:** first user report of the share tab freezing the UI, or next time you grep for
  `_loop.run_until_complete`.
- `_loop.run_until_complete(...)` runs inside imgui-frame draw paths (`ui.py:276` refresh, `:1226`
  delete, `:1294` upload) → blocks the GL frame on Telegram round-trips. ModelBox calls block too
  (synchronous `requests` with `timeout=600.0`, `modelbox.py:52`). Move to a worker thread + result
  queue, or drop async. (Backlog item 5.)

## [DEFERRAL] split `ui.py` (1778-line god-class)
- **Trigger:** next time you add a tab or a reusable widget to `ui.py`.
- One `App` class spanning `ui.py:62-1771`, 40+ methods. Extract `widgets.py` (the `draw_*_details`
  family — `:1348,1385,1429`), `tabs/*.py` (Node / Render / Share), `hotkeys.py`, `project.py`. New
  UI code should already go in the smallest plausible new module (`conventions.md ## Design decisions`).
  (Backlog item 6 — likely run as a high-blast-radius feature, not a drive-by.)

## [DEFERRAL] dead/orphaned `ui_utils` helpers
- **Trigger:** next time you touch `ui_utils.py`, or when wiring ModelBox's depth-map / bg-removal
  features (README advertises them).
- `mod()` (`ui_utils.py:73`) is dead. `depth_mask_to_normals` (`:90`) and `zero_low_alpha_pixels`
  (`:109`) are unused and reach into `Image._image` (private) — they look like *intended* future
  ModelBox wiring, not pure dead code. Decide: wire up or delete. (Backlog item 3.)
