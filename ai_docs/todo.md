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

## [DEFERRAL] split `ui.py` / `app.py` further
- **Trigger:** when editing `app.py` feels painful (search-and-replace across the file misses
  something, or a method's blast radius is unclear because too many siblings share state), OR
  when a 4th tab module needs cross-cutting `App` operations not currently on its public API.
  NOT a default next-step — a 2026-05-15 parallel-agent assessment of `project.py` extraction
  concluded the current 373-line `app.py` is coherent state on a single entity and the
  extraction would be premature abstraction (same shape as feature 002's reversed AppContext).
- Progress: 1778 (single ui.py) → 1508 (feature 001, `tabs/share.py`) → after feature 002:
  `app.py` 373 + `ui.py` 294 + `tabs/{node,render,share,share_state}.py` 398 +
  `widgets/*.py` 547 + `popups/*.py` 166 → after `hotkeys.py` extraction: `ui.py` 255 +
  `hotkeys.py` 45 → after feature 003 (ModelBox removal): `app.py` 354 +
  `widgets/*.py` 497 + `popups/*.py` 134.
- Candidate shapes (if the trigger ever fires): `ProjectPaths` frozen dataclass (extract the 9
  `@property` paths into a value type, `app.paths.nodes_dir` etc.) OR `shaderbox/project.py`
  free functions taking `app: App` (`save` / `open_project` / `delete_current_node` /
  `create_node_from_selected_template` / `select_next_*`). The two are orthogonal — paths are a
  value domain, lifecycle is an action domain. `App` is the state-holder; widgets/popups/tabs/
  hotkeys take `app: App` directly (no `AppContext` wrapper).

## [DEFERRAL] in-app replay mechanism (debug)
- **Trigger:** next time you hit a multi-step bug that's painful to reproduce manually, or when
  you want to share a repro with future-you.
- Add a "Debug → Replays" UI surface that lists JSON files from `replays/`, plays them back by
  injecting synthetic actions into the existing hotkey/button code paths. DSL (rough): list of
  `{frame, action: hotkey|click|assert, ...}` dicts. Intercept points: imgui io-state setting
  for hotkeys (reuse `shaderbox/hotkeys.py::process_hotkeys` as-is),
  thin `replay_aware_button(label)` wrapper for clicks (or globally wrap `imgui.button` in
  replay mode). Not a test framework — for manual debugging / shareable repros (the headless
  smoke test in `scripts/smoke.py` is the right tool for actual regression testing). Probably
  worth a small feature spec before building (touches imgui boundary, adds UI surface).
