# TODO — blockers & deferrals

Known issues parked for later, each with a **Trigger** — the condition under which someone working in
the repo should pick it up. NOT "what's next" (that's `roadmap.md`'s Active-context banner); this is a
"heads up, here's a landmine / a deferred fix" registry. **Grep this file by `Trigger` before
starting work in an area.**

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

<!-- Shape (per entry):
     ## [BLOCKER|DEFERRAL] <short title>
     - **Trigger:** <a concrete observable moment — file/code touch, count threshold, user complaint with a measurable surface, a specific upstream change>
     - <context: what / why / where>
     Entries are an index of triggers, not a home for designs. Resolved → delete in the resolving commit.
-->

---

## [DEFERRAL] imgui_color_text_edit render() FPE — worked around with two guards
- **Trigger:** if imgui-bundle fixes the upstream glyph-metric div-by-zero (or you upgrade and
  the crash no longer reproduces), DROP both guards below and restore the lost features
  (live syntax-highlighted editor behind modals; live-preview while changing editor options).
  Also: if you touch `tabs/code.py`'s render gate or `popups/editor_settings.py`'s apply
  timing, re-verify the crash stays suppressed.
- The bug (NOT fixable from Python — it's a div-by-zero inside imgui-bundle's C++
  `TextEditor::Render`, by `glyphSize.x/.y` from `ImGui::GetFontSize()` + the 1.92 dynamic font
  atlas, when the editor window isn't focused). Two triggers, two workarounds in place:
  1. Rendering the editor on a frame a popup is active → `tabs/code.py` shows a read-only
     plain-text snapshot (`_draw_code_snapshot`) instead of `editor.render()` while
     `any_popup_open()`. **Lost feature:** no syntax colors / cursor behind a modal.
  2. Calling the editor's `set_*()` setters while a modal is open corrupts its glyph metrics →
     next render FPEs. `popups/editor_settings.py` applies settings ONLY on close (Close button
     + `hotkeys.py` Esc path), never live. **Lost feature:** no live-preview while dragging the
     options sliders.
- Reproduces only in the full app (bisected via headless `update_and_draw` cycles), on the
  latest imgui-bundle (1.92.801). Surfaced building the editor-options popup (feature 006).

## [DEFERRAL] gruvbox palette match for the inline editor
- **Trigger:** when imgui-bundle exposes a writable `imgui_color_text_edit.TextEditor.Palette`
  (per-slot setter or list-based constructor), OR if the built-in dark palette's mismatch with
  the rest of the gruvbox UI visibly annoys in daily use.
- Feature 006 ships the inline editor with the built-in `TextEditor.get_dark_palette()` because
  the Palette has no Python write path in imgui-bundle 1.92.801 (spike-confirmed: `.get()` only,
  `set_palette` rejects `list[int]`). The `COLOR.SYN_*` tokens in `theme.py` + the
  `Color`→`SYN_*` mapping table in `ai_docs/features/006_inline_editor.md` (Decision 5) document
  the intended mapping for when this becomes possible.

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

## [DEFERRAL] inline shader-error display (replace raw add_text overlay)
- **Trigger:** when the next person works in the render-area drawing code in `ui.py`
  (the `has_error` branch), OR first time a shader error is hard to read against the
  dimmed image.
- Today shader-compile errors render as raw red `add_text` overlaid top-left on the
  dimmed render image (`ui.py`, `_draw_render_image`-equivalent / the `has_error`
  block). Now that the inline editor has landed (feature 006), promote this to a
  proper error surface that parses GLSL error output (`ERROR: 0:55:` or `0(55) :
  error:`) for `(line, col, message)` and can jump the editor cursor. NOTE: the
  designer's prototype `.error-panel` is one reference, but the prototype layout was
  reverted — design the error surface to fit the CURRENT layout (image-top, control-
  panel-below), not the prototype's.

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
