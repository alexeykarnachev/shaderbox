# 045 — script UX redesign

Builds on 042 (the headless script engine surface + management verbs behind a placeholder UI the
maintainer judged poor). 045 rebuilds that UI: a discoverable in-app scripting UX — attach/edit a
script from a visible per-row control, edit ALL scripts inside ShaderBox's own tabbed code editor,
a mirrored node-header control for the node-brain, script errors shown exactly like shader errors,
and the Python script behaves like a normal file (no curated sandbox globals). All engine verbs
already exist (`create_script` / `detach_script` / `reset_script` / `get_brain_status` /
`get_script_file_for` / `is_uniform_scriptable` / `stub_for` / `brain_stub_for` on `ProjectSession`).

The design was captured from the maintainer's voice messages (foo chat, 2026-06-13) and locked in a
plan discussion the same day; the open questions the draft carried are resolved in **Design
decisions** below.

## Goal

One visible per-row affordance (a text pill) to create/open/edit a uniform's script; the same pill
mirrored in the node header for the node-brain; ALL scripts edited in our own code editor behind a
multi-tab bar; a single **active/inactive** toggle (in the editor) that decides whether a present
script is applied; the value widget disabled while a script is active (the type pill stays live);
script errors rendered through the SAME mechanism as shader errors; and a plain-Python script body
(strip the curated math globals — the user `import math`s themselves).

## Out of scope

- **Docstring-generation opt-out setting** (a future Settings toggle to skip the generated
  docstrings). Not this wave. **Trigger:** a maintainer wants to create a script with no docstring
  boilerplate.
- **The three-dots (`...##node_actions`) mechanism removal + "Save as template" relocation.** The
  three-dots is acknowledged-temporary; this wave only removes its "New node-brain script" item.
  **Trigger:** the maintainer decides where "Save as template" should live instead.
- **Detach / trash a script from the UI.** 045 collapses the script lifecycle to one toggle
  (active/inactive); there is no UI "detach" anymore (the file stays on disk, inactive). Removing the
  file is a manual on-disk delete, which the engine already auto-handles (the binding drops, the
  uniform returns to manual). **Trigger:** a real need to delete a script's file from inside the app
  surfaces (then add a delete affordance to the editor's script tab, mirroring the toggle).
- **Per-uniform "Restart script" (re-run `__init__`).** 042 carried it in the right-click menu; the
  redesign drops the menu and does not re-home it (a save already re-instantiates; the node-brain
  keeps no separate restart either). **Trigger:** a workflow needs to re-run `__init__` without a
  source edit.

## Design decisions

Locked. (1)–(8) resolve the draft's eight open questions; (9)–(12) lock the mechanics.

1. **One state machine: a script is ACTIVE or INACTIVE — nothing else.** A uniform/node either has no
   script (the pill reads "+ script" / "+ brain") or has one that is being applied (active) or not
   (inactive). There is no separate "attached vs disabled" axis and no detach: an inactive script
   still exists on disk and reopens instantly. The *file's existence* is an implementation detail the
   user never reasons about — it is created lazily on first open and never deleted by the redesign.
   The active/inactive bit is the only state the user sees.

2. **The per-row affordance is a TEXT PILL, not an icon** (no glyph to draw/maintain). Three captions:
   `+ script` (no script) · `script` accent-filled (active) · `script` muted/faded (inactive). The
   accent fill on the active state is the at-a-glance "which uniforms are script-driven" cue in the
   uniform list. Clicking the pill in ANY state opens (creating first if absent) the script in the
   editor — it never toggles. Built on the existing `pill_button`; the new wrapper is ONE shared
   primitive used by both the per-uniform row and the node header (decision 9).

3. **Enable/disable lives in the EDITOR, not the row.** Toggling active/inactive is a rare operation,
   so it is a single well-visible button at the **top-right of the editor pane, above the tab bar**,
   shown only when the active tab is a node/uniform script. It uses the existing `toggle_button`
   primitive (filled-accent when active, bordered-ghost when inactive). The row pill only opens; the
   editor button only toggles — the two gestures never conflict (the draft's central failure).

4. **Tab labels are SEMANTIC, not file paths.** The user never sees a path (everything is
   project-relative and path-irrelevant). Labels: `shader` (the node's `shader.frag.glsl`), `node
   script` (the node-brain `script.py`), `script · <uniform>` (a `u_<name>.py`). Lib files keep their
   filename (`SB_foo.glsl`) — they are genuinely path-named resources.

5. **The editor holds an ORDERED LIST of open tabs; everything opened in the editor is a tab.** Node
   shader, node-brain, per-uniform scripts, and lib files all become closable tabs (an `x` per tab,
   reopen via the originating affordance). No pinned/permanent tab, no special first tab — uniform
   treatment. This supersedes the single-slot `_explicit_editor_path` + the `< back to node` /
   `back to lib >` chrome. Closing the last tab leaves the editor empty (a "no file open" caption).

6. **Disabled persistence is a by-filename sibling marker** (`<script>.disabled` next to the `.py`).
   Presence of the marker = inactive. It is by-filename (preserves the 041 stateless-by-filename
   invariant), survives reload + restart, and is EXTERNAL to the script body so export-isolation
   (`fresh_behaviors_for` / `tick_behaviors`) honors it for free (a disabled binding is never compiled
   into the fresh export set). The engine skips a marked binding at reload (it is not discovered, the
   uniform returns to manual). This is a real engine change (B4).

7. **Script errors render through the SHADER-ERROR mechanism, in one place — the editor's bottom error
   strip.** 042's per-row `notice_strip` + the node-brain strip are REMOVED. When a script tab is
   open, the active script's `ScriptError` (compile/coercion/runtime, with its line) is shown in the
   editor's bottom error strip — the same `_draw_error_strip` visual the shader uses, click-to-jump to
   the offending line. Script errors and shader errors look identical and live in the same spot. The
   node-brain's homeless soft-key errors (typo/orphan keys that name no uniform row) surface the same
   way when its tab is open.

8. **The text/array char-count moves into the uniform NAME.** The `{len}/{cap}` caption (text uniform)
   and the array branch's `{len}/{cap}` / `[...]  ({cap})` captions vacate the trailing column (now
   the pill's home). The count moves to the name cell as `u_label (46/64)`.

9. **ONE shared script-pill primitive, ONE shared toggle.** Per the reuse mandate: a single
   `script_pill(...)` in `ui_primitives.py` draws the row/header pill (three states), and the editor's
   active/inactive control reuses the existing `toggle_button`. No per-surface duplicate widgets. The
   `script_chip` primitive (the old "py" pill) is deleted (no longer used).

10. **Active = the value widget is `begin_disabled`'d but the type pill stays live.** 042 skipped the
    whole control branch and drew a read-only `format_auto_value` readout; 045 instead draws the REAL
    widget inside `begin_disabled` (the live script value shows through it), and `draw_input_type_selector`
    (the type pill) is drawn OUTSIDE the disabled scope so the user can still switch the view type. An
    INACTIVE script's row is fully normal (manual control, widget live).

11. **The exec globals keep only the class machinery; the math vocab + the `_no_import` shim go; real
    `import` is restored.** `behavior._build_globals` drops `sin/cos/.../pi/tau/clamp/lerp/mix/math`
    and `_no_import`; `__builtins__` becomes the real builtins module (so `import math` works and the
    user has the full standard library). The injected top-level globals stay (`ScriptBehavior` / `Ctx`
    / `MouseState` / `Vec2/3/4` / `Array` / `Text` — eager annotations resolve against them). Real
    builtins subsume the formerly-curated `float/int/bool/super/list/...`, so those explicit entries go
    too. **CONSEQUENCE:** every bare-math usage (stubs, `scripts/smoke.py` seed,
    `scripts/dogfood/verify_script_engine.py`, on-disk dev-sandbox scripts) must migrate to `import
    math` + `math.sin(...)` or it raises `NameError` at tick (not compile). The stub generators emit
    the `import math` pattern.

12. **Stub docstrings are scoped correctly (E1).** The `ctx` field reference (`ctx.t/dt/frame/
    mouse.x/y`) moves into the `update` method docstring; the class docstring carries the high-level
    "what this drives"; the `__init__` docstring states it may do work and runs ONCE (app start /
    before render / on reload). The `_MATH_DOC` line is replaced by an `import math` hint.

## Part A — remove the wrong 042 affordances

- **A1.** Delete `widgets/uniform.py::_script_actions_menu` (the right-click context menu) + its
  `confirm_delete_popup` + the `tabs/node.py` "Right-click a uniform for script actions" hint line.
- **A2.** Delete the `script_chip` driven-row branch + the `script_chip` primitive (decision 9).
- **A3.** Remove the "New node-brain script" item from `node_actions_popup` (keep only "Save as
  template"). Brain creation moves to the header pill (Part D).
- **A4.** Remove the OS-editor launch for scripts (`App.open_script_file` / `App.create_script_for`
  stop calling `open_file_in_default_app`; they route into the editor). `open_file_in_default_app`
  (util.py) becomes unused → delete it (its only two callers are these).
- **A5.** Strip the curated math vocab + `_no_import` from `behavior._build_globals`; restore real
  builtins/import (decision 11).

## Part B — the per-uniform script pill

- **B1.** Row spine unchanged `[type pill][name][control]`; ADD the `script_pill` trailing (end of
  row), drawn before the existing early-return path collapses.
- **B2/B3.** The pill is the open affordance (decision 2); the toggle lives in the editor (decision 3).
- **B4.** The inactive (disabled-but-present) state (decisions 1, 6) — a real engine change.
- **B5/decision 10.** Active → value widget `begin_disabled`, type pill stays live.
- **B6/decision 8.** Char-count → the name cell.

## Part C — the tabbed multi-file editor

- **C1.** `App.get_session` becomes suffix-aware: `.py → Language.python()`, else `Language.glsl()`.
- **C2–C5/decision 5.** An ordered open-tabs list + active index replaces `_explicit_editor_path`; a
  tab bar atop the editor (`tabs/code.py`); open/close/switch ops; semantic labels (decision 4); the
  FPE-behind-modal guard still gates the WHOLE pane (tab bar + editor) — `tabs/code.py::draw` returns
  early when `any_popup_open()`, before drawing either. Opening a script switches/creates its tab +
  `EditorSession`; Ctrl+S + the mtime hot-reload are unchanged.

## Part D — the node-brain header pill

- **D1–D4.** The same `script_pill` (states `+ brain` / `brain` active / `brain` inactive) in the node
  header row (`tabs/node.py`, beside node-name / resolution / `...`). Click opens/creates `script.py`
  in a tab; the editor toggle enables/disables it. The brain's errors render via decision 7 (the
  editor error strip when its tab is open) — the separate `_draw_brain_strip` is REMOVED.

## Part E — stub / docstring rework

- **E1/decision 12** + **E3** (bodies compile under the stripped globals — current literal defaults
  already do; the docstring shows the `import math` pattern).

## Files touched

- `scripting/behavior.py` — `_build_globals` (strip math + `_no_import`, restore real builtins);
  delete `_no_import`.
- `scripting/engine.py` — the DISABLED marker: `reload`/`_reload_brain` skip a `<file>.disabled`-marked
  binding; new verbs to set/clear the marker + query active/inactive; `stub_for`/`brain_stub_for`
  docstrings + `import math` hint (replace `_MATH_DOC`).
- `project_session.py` — wrappers for the new enable/disable + script-state query; `create_script`
  now also the lazy-create-on-open seam.
- `app.py` — the open-tabs list replacing `_explicit_editor_path`; `get_session` suffix-aware;
  `open_script_file` / `create_script_for` route into the editor (no OS launch); tab open/close/switch
  + the active/inactive toggle wiring; delete the `open_file_in_default_app` import.
- `widgets/uniform.py` — remove `_script_actions_menu` + `_draw_row_error` + the `script_chip` branch;
  add the trailing `script_pill`; `begin_disabled` the value widget when active (type pill outside);
  move the char-count into the name cell.
- `tabs/node.py` — remove the hint line + the "New node-brain script" item + `_draw_brain_strip`; add
  the header `script_pill`.
- `tabs/code.py` — the tab bar + the per-tab editor render + the active/inactive toggle button; the
  script-error feed into `_draw_error_strip`; replace `draw_chrome`'s single-file logic.
- `ui_primitives.py` — add `script_pill`; delete `script_chip` (+ `ScriptChipState`). `notice_strip` /
  `confirm_delete_popup` lose their script callers (keep them — still used elsewhere; confirm at impl).
- `util.py` — delete `open_file_in_default_app`.
- `scripts/smoke.py`, `scripts/dogfood/verify_script_engine.py`, `projects/dev/nodes/*/scripts/*.py` —
  migrate bare-math → `import math`.

## Manual verification

(Headless impl; the maintainer runs `make run` and checks.) — Per-uniform: a non-scripted row shows
`+ script`; click → a `script · u_x` tab opens with Python highlighting + a default that renders;
the row pill goes accent-filled, the value widget greys out, the type pill still switches. The
editor toggle (top-right) flips the pill to muted + restores the live widget; flip back. A syntax
error shows in the bottom error strip (same look as a shader error), click jumps to the line. The
node-brain header pill mirrors all of it. Open several tabs, switch/close them, reopen. A freshly
created script compiles + runs without any `import`-less math NameError. Restart the app — an inactive
script stays inactive.

## Open questions for the user

None — all eight draft questions resolved in Design decisions, signed off in the 2026-06-13 plan
discussion.

## Review history

**Pre-impl swarm (2 adversarial reviewers, 2026-06-13).** Reviewer 2 returned FAIL but on a false
premise — it reviewed the un-written engine code as if it should already exist; its real value is
the precise locations + the complete bare-math list, adopted as impl targets (not pre-existing bugs).
Reviewer 1 surfaced genuine spec-precision gaps, all resolved here as locked impl decisions (robust
defaults, no escalation — none was a "should not land"):

- **Tab model.** An `EditorTab` (path + kind) ordered list + active index on `App` replaces
  `_explicit_editor_path`. `current_editor_path` returns the active tab's path (None when empty).
  `open_shader_lib_file` / `show_node_editor` / `_on_shader_lib_paths_removed` /
  `_on_shader_lib_path_renamed` / `detach_script` cleanup all re-expressed over the tab list;
  `draw_chrome`'s `is_lib` check becomes the active tab's kind.
- **FPE guard.** The guard is for `TextEditor.render` only (not generic imgui). The tab bar draws
  ABOVE the render guard (tabs stay visible while a modal is open); only `editor.render()` + the
  error strip are gated by `any_popup_open()`.
- **Script errors in the strip.** `_draw_error_strip` stays `list[ShaderError]`; when a script tab
  is active, `code.py` adapts the active script's `ScriptError` → a `ShaderError(path=<script
  file>, line, message)` so the existing click-to-jump works unchanged (no union, no new strip).
- **Disabled-widget write-back guard.** `begin_disabled` blocks interaction, so the value widget
  returns its unchanged value — but the write-back `if new_value is not None` would still fire.
  Guard: skip the write-back when the script is ACTIVE (the engine owns the value).
- **`script_pill` signature.** `script_pill(id_: str, label: str, state: Literal["absent",
  "active", "inactive"]) -> bool` in `ui_primitives.py` (built on `pill_button`; absent/inactive
  faded, active accent-filled). One primitive, both surfaces.
- **`get_session` suffix path.** `open_script_file` builds a `ShaderSource` from the script path
  and routes through `get_session`, which picks `Language.python()` for a `.py` suffix.
- **Engine disabled state.** The skip goes in `reload` (per-uniform: a `<file>.disabled` sibling →
  don't `found.add`, so the drop loop clears behaviors/mtimes/sources/errors — same as a vanish)
  and `_reload_brain` (`script.py.disabled` → return without `found.add`). Export isolation honors
  it BY CONSTRUCTION once the skip is in (a dropped binding leaves `scripts.sources`, so
  `fresh_behaviors_for` can't recompile it). New `ProjectSession` verbs: `set_script_active` /
  `is_script_active`. The conventions.md script-engine bullet + the Known-quirks exec-globals note
  are updated in the same wave (the curated-math-vocab claim goes stale).
- **`reset_script` orphan.** Revised at post-impl (see below): the dead `ProjectSession.reset_script`
  wrapper is REMOVED; the engine primitive `ScriptEngine.reset` stays (3 test callers exercise it as a
  documented engine capability a future restart affordance would reuse).

**Post-impl swarm (3 adversarial reviewers, 2026-06-13).** Round 1: spec-fidelity PASS (all 9 verbatim
requirements + 12 decisions covered), code-correctness PASS (tab-index -1 sentinel safe, disabled-marker
lifecycle correct, write-back guard + begin_disabled balance verified, FPE gate placement correct,
strip-height capped), architecture PARTIAL. Triage of the PARTIAL:

- **REAL — dead code (fixed).** `notice_strip` / `confirm_delete_popup` / `reset_state_button` (+
  `NoticeTone` / `_NOTICE_COLOR`) lost their only callers when the right-click menu + brain strip went;
  the spec's "keep, confirm at impl" resolved to delete. `ProjectSession.reset_script` likewise had no
  caller — removed (the engine `reset` + its tests stay).
- **FALSE POSITIVE — `Language.python()` unverified.** The reviewer couldn't run the dep; verified
  present in the installed `imgui_color_text_edit` (`Language.python()` returns a real Language object).
- **MINOR (fixed) — row/header pill clickable-but-no-op mid-copilot-turn.** The editor toggle was
  `begin_disabled`'d but the pills weren't; wrapped both in `begin_disabled(copilot_turn_active)` so the
  freeze reads visually, matching the toggle.

`make check` + 467 tests green after the fixes. Shipped as commit e2dffd0.

**Ultracode UX-gap audit (31 agents, 2026-06-13, post-commit e2dffd0).** A second, deeper swarm hunted the
specific class the maintainer flagged — UI/UX holes 045 might be deferring the way 042 did. It found the
redesign HAD reintroduced that class, and a fix wave closed it (re-review swarm PASS):

- **Shared-root regression (the main one): the per-row pill was blind to brain-driven + error.** The pill
  read the disk-only own-file state, so a uniform driven by the NODE-BRAIN (no own file — the DEFAULT the
  instant a brain is created, since `brain_stub_for` seeds every scriptable uniform) showed a misleading
  `+ script` pill AND kept its value widget editable (the brain rewrote it each tick → user drags snapped
  back). The engine already knew the truth (`script_driven_uniforms` includes brain keys; `errors`); the UI
  never asked. Fix: a brain+error-aware `ProjectSession.uniform_pill_state` + `is_uniform_script_owned` (the
  pill reflects the own-file intent incl. inactive; the widget-lock independently respects the brain) + an
  `"error"` state on `script_pill` / `ScriptState` (a broken script/brain shows a red `! script` / `! brain`
  pill on the row + header, not only inside the editor — replacing 042's deleted per-row error line + brain
  strip, the discoverability 045 had dropped).
- **`flush_current_editor` KeyError (crash):** `ui_nodes[current_node_id]` was unguarded — Ctrl+S/quit with
  a dirty lib/script tab while all nodes are deleted (`current_node_id == ""`) crashed. Guarded with the
  sibling `ui_nodes.get` pattern.
- **Editor tab-bar overflow:** the flat `same_line` strip clipped tabs (+ their close `x`) off-screen past
  ~3-4 tabs. Wrapped in a horizontal-scroll child with the enable/disable toggle pinned right.
- **Audited clean:** the rest of the tab lifecycle, `ensure_shader_tab` coverage, the `__builtins__` exec
  semantics, the disable→enable mtime race, and the bare-math migration (100%) — all verified solid.
- 8 new `test_uniform_pill_state.py` tests pin the brain-driven / error / inactive-shadow row states;
  `make check` + 475 tests green.
