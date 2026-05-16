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
  `widgets/*.py` 497 + `popups/*.py` 134 → after feature 004 (imgui-bundle migration):
  `app.py` 351 + `ui.py` 251 + `widgets/*.py` 503 + `popups/*.py` 135.
- Candidate shapes (if the trigger ever fires): `ProjectPaths` frozen dataclass (extract the 9
  `@property` paths into a value type, `app.paths.nodes_dir` etc.) OR `shaderbox/project.py`
  free functions taking `app: App` (`save` / `open_project` / `delete_current_node` /
  `create_node_from_selected_template` / `select_next_*`). The two are orthogonal — paths are a
  value domain, lifecycle is an action domain. `App` is the state-holder; widgets/popups/tabs/
  hotkeys take `app: App` directly (no `AppContext` wrapper).

## [DEFERRAL] feature 006 — embedded GLSL editor (PR 5 from designer's adoption sequence)
- **Trigger:** when the "GLSL editor / coming in feature 006" placeholder on the left
  half of the main window starts feeling like real friction in daily use — i.e. the user
  notices "I switch to vim more than I'd like" or "I want to see compile errors next to
  the code that caused them".
- Integrate `imgui_color_text_edit` (bundled in `imgui-bundle`). Wire syntax palette to
  the `COLOR.SYN_*` tokens already exported by `shaderbox/theme.py`. Replace the
  placeholder in `ui.py::_draw_editor_placeholder` with a real `imgui_color_text_edit`
  `TextEditor` instance. Each `UINode` gets its own `TextEditor` lazily allocated and
  switched when `current_node_id` changes. Save on `Ctrl+S` (file write + node
  `release_program(source)` reload — already the existing reload path). Keep the
  external-editor "Pop out" button as a fallback. Details: designer's `SPEC.md §3.1`.
- Re-evaluate the 4 open questions in designer's `SPEC.md §16` during this feature's
  plan-lock — particularly Q1 (editor library pick: `imgui_color_text_edit` is the
  default; verify it actually works in imgui-bundle 1.92.801 before plan-locking) and
  Q2 (multi-file: deferred unless a vertex-shader or `node.json` editor surface lands
  in scope).

## [DEFERRAL] feature 007 — Node tab restructure (PR 6 from designer's adoption sequence)
- **Trigger:** blocked on feature 006 completing (the Node tab and the embedded editor
  share screen real estate; restructuring without the editor leaves the right-half
  layout in an awkward intermediate shape).
- Collapse the current two-child Node tab (uniforms left, selected-uniform editor
  right) into a single uniform list. Each row becomes a 3-column layout: name |
  type pill | controls. Click the type pill to switch input_type (color↔drag,
  array↔text, etc) via a popup menu — replacing the current `draw_selected_ui_uniform_
  settings` side panel. Details: designer's `SPEC.md §8`.

## [DEFERRAL] feature 008 — Inline shader-error banner (PR 7 from designer's adoption sequence)
- **Trigger:** blocked on feature 006 completing (the banner's click-to-jump targets
  the embedded editor's cursor).
- Replace today's red `add_text` overlay on the dimmed render image with a real banner
  widget rendered between the render card and the node panel. Parse GLSL error output
  (`ERROR: 0:55:` or `0(55) : error:`) for `(line, col, message)`. Clicking the
  `line:col` link jumps the editor cursor via `TextEditor.SetCursorPosition`. Details:
  designer's `SPEC.md §5`.

## [DEFERRAL] feature 009 — Tweaks panel (PR 8 from designer's adoption sequence)
- **Trigger:** after the user has lived with the default gruvbox/yellow/tight/subtle
  combination for ~1 week of daily use and has a signal (e.g. "I'd like to try the
  aqua accent" or "comfortable spacing is annoying me, I want tighter"). Premature
  to ship before a preference exists.
- In-app Tweaks panel (toggled by the `Tweaks` placeholder button on the topbar) for
  runtime accent / density / rounding / editor-side swaps. Persist in
  `app_state.json::tweaks` (`AccentName / DensityName / RoundingName + side: left/right`).
  On change, re-call `apply_theme(...)` with the new args; editor-side change is
  layout-level and re-evaluates the split next frame. Details: designer's `SPEC.md §14`.

## [DEFERRAL] adopt `hello_imgui.apply_theme()` + `imgui-knobs` during UI/UX refactor
- **Trigger:** when starting the planned UI/UX refactor with custom themes — i.e. the moment a
  concrete theme design starts taking shape (not before; adopting now is premature scaffolding
  without a target).
- `hello_imgui` ships ~15 named themes (`hello_imgui.apply_theme(ImGuiTheme_.darcula_darker)`)
  + a live tweak GUI (`hello_imgui.show_theme_tweak_gui`). Adoption cost is ~5 LOC at theme
  swap time. `imgui-knobs` (rotary knobs for shader parameter UX) is in the same trigger —
  evaluate replacing `drag_float` widgets in `widgets/uniform.py` when drag UX feels
  insufficient. Both rejected from feature 004 scope to keep blast radius bounded; both
  available with no new dependency since `imgui-bundle` already bundles them.

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
