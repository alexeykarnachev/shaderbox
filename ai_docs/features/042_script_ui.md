# 042 — script UI (surface the headless script engine in the app)

The in-app affordance for the CPU-script engine. The engine (041 per-uniform + 044 node-brain) is DONE
and headless; **042 is a MIRROR + LAUNCHER, not a new editor** — it surfaces the already-running engine
through the Node tab the user already lives in. Nothing here changes the engine's compute; it reads the
engine's existing state (`script_engine.errors`, `script_file_for`, `script_driven_uniforms`,
`reset_script`, `stub_for`, `is_scriptable`) and adds the affordances 041/044 explicitly punted to 042.

Source-of-ideas: a 5-lens brainstorm swarm + devil's-advocate + synthesis (run 2026-06-13). The
swarm's framing — "smallest complete 042 = a row state, a node strip, creation, lifecycle verbs; the
in-app Python editor is a strictly-larger fast-follow" — is adopted.

## Goal

A user who never reads the spec can, from the Node tab:
1. **See** which uniforms are script-driven and by what (a per-uniform `u_<name>.py` vs the node-brain
   `script.py`), and watch the live value the script writes each frame.
2. **See** a script's errors in-app (per-uniform compile/runtime/coercion errors on the row; the
   node-brain's compile error + its homeless soft-key errors in a node-level strip) — today they are
   loguru-only.
3. **Create** a script: "Make scriptable" on an undriven scriptable uniform (writes `stub_for`), and
   "New node-brain script" in the node-actions popup (writes a new `brain_stub_for` skeleton).
4. **Manage** a script: restart it (re-run `__init__` via `reset_script`) and detach it (trash the file,
   the engine drops the binding next reload).
5. **(should)** Read the cursor from a script via `ctx.mouse`, wired live over the node preview, frozen
   to a fixed value on export.

Editing the script file itself is, in the first 042 wave, **launched to the OS editor** via
`open_in_file_manager` (the engine already hot-reloads on mtime). The in-app Python editor is a `could`
captured in Out-of-scope.

## Out of scope (each deferral carries a trigger)

- **In-app Python editing in the goossens `TextEditor` pane.** The path-keyed editor (`app.editor_sessions[Path]`)
  could be switched to `Language.python()` for a `.py` path with a "back to node" chrome twin + a Python
  error strip — a one-window flow. CUT from the first wave: the OS-editor launch (`open_in_file_manager`)
  already delivers edit→save→hot-reload with ZERO editor coupling and ZERO new FPE surface; the in-app
  editor is a strictly-larger second wave (generalize `get_session`'s `ShaderSource` coupling, a 3-way
  editor-mode chrome, short-circuit the GLSL marker path for a `.py` file, and the copilot read-only lock
  now reaching script files). **Trigger:** the OS-editor round-trip (alt-tab to an external editor) proves
  too much friction in daily use, OR feature 043 (copilot writes scripts) wants the edits visible in-pane.
- **Auto-fed `u_mouse` vec2 UNIFORM (engine-driven like `u_time`).** A stateless cursor effect
  (ripple/spotlight) could read the cursor in pure GLSL with no `.py`. CUT: it's a SECOND write-path for
  the same cursor value (drift risk — the y-convention + the export-freeze must hold identically on both)
  and it touches the engine-free `Node.render` path. `ctx.mouse` alone satisfies 041's named deferral.
  **Trigger:** a concrete stateless cursor-reactive shader is wanted with no script.
- **Brain-vs-`u_<name>.py` conflict (shadowed brain write) disclosure.** When both files drive one slot
  the `u_<name>.py` wins (044 decision 7) and the brain's write to that slot is silently shadowed. CUT
  from 042: surfacing it needs a NEW engine accessor (`last_driven` INTERSECT `behaviors`) for a corner
  case the winning per-uniform chip already partially discloses (the row shows a `per_uniform` chip, not
  a `brain` chip). **Trigger:** a user hits "I edited the brain and that uniform didn't change."
- **`MouseState.down` / `.inside` / buttons.** v1 `ctx.mouse` carries position only (`x`, `y`). **Trigger:**
  a concrete click/leave behavior in a script (then add the field deliberately).
- **A "mouse live" overlay chip on the preview.** A non-problem — cursor-reactivity is its own signal once
  it works — and "a script reads `ctx.mouse`" isn't statically detectable without source introspection, so
  the chip would over-claim. CUT (do not build).
- **A scripts-on-node OVERVIEW list (in the node-actions popup) + a generalized shared `ErrorRow` protocol
  unifying `ShaderError` + `ScriptError`.** Both CUT: the overview duplicates the per-row chips + the brain
  strip (two action homes for one verb = slop, `/imgui-ui` §7.4); the shared error-protocol abstraction only
  pays off IF the in-app Python editor lands (it's that wave's same-fix-at-the-root move). **Trigger:** the
  in-app editor lands → revisit the protocol then.

## Design decisions (numbered; lock-in only)

1. **042 is read-of-engine-state + file-write + launch — NOT a parallel script model.** Every indicator
   reads an EXISTING engine query; every creation writes a `scripts/` file and lets the existing
   `reload_scripts` poll bind it; every lifecycle verb calls an EXISTING wired method (`reset_script`) or
   trashes a file. NO new per-script state on `App`/`UINode`/`node.json` (binding stays by-filename, 041
   decision). The ONLY engine ADDITIONS are `brain_stub_for` (the one gap 044 left) + the `ctx.mouse`
   subsystem (decision 9); everything else is UI over a done engine.

2. **The script chip rides the EXISTING `CHIP_W` slot on the uniform row — it does NOT add a column.**
   `draw_ui_uniform` draws `chip → name → control`. When a uniform is script-driven, the input-type chip
   slot shows a `script_chip` (decision 5) INSTEAD of the input-type selector (a script-driven uniform's
   input shape is irrelevant — the engine writes it). When a uniform is undriven+scriptable, the
   input-type chip stays AND a "Make scriptable" action is reachable via the row's right-click context menu
   (decision 6). No `_NAME_X`/`_CTRL_X` shift → no row jitter (`/imgui-ui` §3). The auto-row (engine-owned
   `u_time`…) gets NO chip — those are `_binding_reject`ed, never scriptable.

3. **A script-driven control is READ-ONLY and shows the live engine value; the write-back is SKIPPED.**
   This is the one CORRECTNESS item, not a nicety: `session.tick` overwrites `uniform_values` every frame
   before render, so a live slider on a scripted uniform visibly fights the script (flicker), and the
   `new_value` write-back at the bottom of `draw_ui_uniform` would clobber the script for one frame. On a
   driven row, after the chip + name cell, `draw_ui_uniform` shows the read-only readout and RETURNS before
   the control/write-back/`try_to_release`. The readout reuses the `util.format_auto_value` function (the
   `_draw_auto_row` treatment — `[a, b, c]` for vectors, int-aware, NaN-safe), NOT `draw_ui_uniform`'s own
   inline `input_type=="auto"` hand-formatting (subtly different) — so the scripted readout matches the
   engine-driven (`u_time`) rows exactly. This covers EVERY scriptable KIND uniformly: scalar, vec/color,
   array, text (all `is_scriptable`; samplers/blocks are never driven). A scripted uniform IS conceptually
   an engine-driven uniform. (Pre-impl NIT folded: name the `util` function, not the inline branch.)

4. **Per-row errors and node-level errors have DIFFERENT homes, keyed by the engine's own split.**
   `script_engine.errors` is keyed `(node_id, binding_key)`. A per-uniform key `(node_id, "u_x")` or a
   brain PER-KEY coercion error `(node_id, "u_x")` (a real uniform the brain drove that failed to coerce)
   → the **uniform row's error line** (decision 7). The brain SENTINEL `(node_id, "script.py")` (the brain
   failed to compile/run — it drives ZERO rows) AND the brain's HOMELESS soft-key errors (`last_skipped`:
   a typo'd / engine-owned / orphan key that names no real uniform) → the **node-brain strip** (decision 8).
   No double-reporting: a brain key that bound a real uniform shows on its row; only the sentinel +
   `last_skipped` keys (which have no row) go to the strip.

5. **`script_chip(state, id_) -> bool` is a new tri-state ui_primitive built over `pill_button`.** States:
   `none` (not drawn as a chip — per decision 2 the input-type chip occupies that slot; "Make scriptable"
   lives in the context menu; `none` is reserved for a future explicit affordance), `per_uniform`
   (`ACCENT_PRIMARY` role color, a `u_<name>.py` drives it), `brain` (`STATE_INFO` role color, the node-brain
   drives it). It builds over `pill_button(label, *, color=..., active=...)` — NOT `chip_button`: pre-impl
   review caught that `chip_button` exposes only `active` (a HARDCODED `ACCENT_PRIMARY` fill) + `faded`/`disabled`
   and takes NO arbitrary color, so it physically cannot render the `brain` tint; `pill_button` is the
   primitive documented for exactly this ("when `chip_button`'s fixed palette doesn't fit — e.g. blue tags").
   `script_chip` maps `state` → `pill_button`'s `color`. State is read PURELY from
   `get_script_file_for(node_id, name)` → `"u_x.py"` / `"script.py"` / `None`. The chip's tooltip names the
   driving file; a LEFT click is the primary "open the script" (OS editor); the row's right-click context
   menu carries the verbs (decision 6). The `brain` state appears only AFTER the brain's first tick
   (`last_driven` is post-tick) — it self-corrects next frame; never assert on it.

6. **Per-row script ACTIONS are a right-click context menu, anchored to the NAME cell, never inline icon
   buttons** (`/imgui-ui` §7.4). The anchor matters (pre-impl BLOCKER): `begin_popup_context_item` attaches
   to the LAST SUBMITTED item. On a driven row decision 3 returns before the control, leaving only
   `caption_text` (a plain text — NOT an item a context menu can attach to). So the menu attaches to the
   `uniform_name_label` cell (a `clickable_label` = a `selectable`, always interactive on every row, driven
   or not): call `begin_popup_context_item(f"##uctx_{name}")` immediately AFTER `uniform_name_label`.
   Right-click (the menu) and left-click (the existing jump-to-declaration + hover bridge) do NOT conflict
   on a selectable, so the name cell keeps both behaviors. A one-line "Right-click a uniform for script
   actions" hint sits above the uniform list (the affordance, set once). Menu items, gated by state:
   undriven+scriptable → "Make scriptable"; per_uniform-driven → "Open script", "Restart script", "Detach
   script"; brain-driven → "Open node-brain" (restart/detach are node-level → the brain strip). Every
   destructive/creating item is GATED IN PYTHON on the real condition (`is_scriptable`, `copilot_turn_active`)
   — `menu_item_simple(enabled=False)` can still fire (`/imgui-ui` §7.4). Wrap the popup in
   `context_menu_style()`. NEVER a modal (per-row action menus are `begin_popup`).

7. **`notice_strip` is the ONE error/status surface, fixed-height, shared by the per-row error line, the
   brain strip's sentinel error, and the homeless soft-key list.** `notice_strip(id_, text, *, tone, on_click=None, line=None)`
   builds on `status_slot`'s fixed-height jitter-free child: `tone ∈ {info,ok,warn,error}` → `STATE_*`; an
   optional `line` renders the shader-strip idiom (`"Line N · message"` when `line >= 0`, bare `message`
   when `line < 0` for coercion/non-dict errors — mirrors `code.py::_draw_error_strip`); `on_click` opens
   the script (synthesize the path from `scripts_dir_for + filename`, since `ScriptError` carries no `.path`).
   The text must `push_text_wrap_pos` (a `text_colored` never wraps — `/imgui-ui` §5). **Height trade
   (pre-impl SHOULD):** the slot is reserved ONLY on a row that currently HAS an error, not on every row —
   reserving a `status_slot` `begin_child` on all N uniforms every frame is a large permanent vertical cost
   (and N extra child windows). "Fixed-height" therefore means fixed WHEN PRESENT (the error text wrapping
   never changes the slot's height), NOT present-on-every-row. The error/no-error transition jitter is
   accepted: an error appears on hot-reload (a save), not on hover/interaction, so the one-frame shift is
   rare and non-interactive — the opposite of the grid-click jitter §3 targets. The error slot is
   non-interactive, so it does not disturb the `nav_flattened` uniform-child traversal.

8. **The node-brain strip is a fixed-height node-level banner, sibling to the auto-row, drawn only when the
   node has a brain.** Position: in `tabs/node.py::draw`, after the sort-combo / auto-row block, before
   `begin_child("ui_uniforms")`. Healthy: a `brain` pill + dim "drives N uniforms" (N from `last_driven`).
   Sentinel error: the `(node_id, "script.py")` `ScriptError` as an `error`-tone `notice_strip` (click-to-open
   `script.py`) — the ONLY home for a brain that compiles to nothing. Homeless soft-keys: a capped
   "{N} brain key issues" line (mirror `_MAX_ERROR_ROWS`) so a typo-spewing brain can't blow panel height,
   each listing the bad key + its reason. Strip height is FIXED across healthy/error/absent (or the uniform
   list jitters, `/imgui-ui` §3). A node with no brain shows nothing (no strip). Reads need a thin
   ENGINE-side accessor for `last_skipped` + its soft-error keys + the driven count — added on `ScriptEngine`
   and forwarded by `ProjectSession` (the tab MUST NOT poke `_nodes`; `conventions.md` headless-encapsulation).

9. **`ctx.mouse` is a SHOULD subsystem: a defaulted `EngineContext` field, threaded via a defaulted `tick`
   kwarg, frozen on export.** `EngineContext` gains a `mouse: MouseState` field — a frozen `MouseState(x, y)`
   normalized `0..1`, **y-up over the canvas** (documented ONCE — the preview draws uv-flipped, so pick y-up
   and state it), **with a default** (`= EXPORT_MOUSE`) so the existing `EngineContext(...)` call sites keep
   compiling (the 041 decision-5 compile-break hazard — verified there are exactly 2 PRODUCTION construct
   sites: `project_session.py::tick` + `_make_export_isolation`, plus ~15 in `tests/test_script_engine*.py`
   and 2 in `scripts/dogfood/{verify_script_engine,harness}.py` — a default keeps every one compiling).
   **The `tick` signature also gains a defaulted kwarg** (pre-impl BLOCKER/SHOULD): `ProjectSession.tick`
   becomes `tick(node_ids, t, dt, frame, *, mouse: MouseState = EXPORT_MOUSE)` — the 2 dogfood callers call
   it POSITIONALLY with 4 args (`session.tick([...], t, dt, i)`) and MUST keep compiling (they get the
   deterministic default); only `ui.py` passes `App`'s stashed live mouse. `ProjectSession` reads NO `App`
   (headless invariant) — the value is PASSED IN.
   **Live hit-test (pre-impl BLOCKER):** the preview is drawn with `imgui.image_with_bg(...)`, which submits
   NO interactive item — so `is_item_hovered()` after it reads the WRONG (previous) item and never fires.
   The hit-test therefore captures the image's screen rect EXPLICITLY: `get_cursor_screen_pos()` before the
   `image_with_bg` + the `image_size`, then `is_mouse_hovering_rect(rect)` AND
   `is_window_hovered(HoveredFlags_.child_windows)` for popup-blocking (the exact `/imgui-ui` §3 pattern for a
   non-item rect — NOT bare `is_mouse_hovering_rect`, which ignores popups). That is what the
   `item_normalized_mouse(rect_min, rect_max, *, flip_y=True) -> tuple[float,float,bool] | None` primitive
   takes (an explicit rect, NOT "the last item") — `None` when the mouse pos is invalid/outside. `ui.py`
   stashes the result on `App`. **One-frame lag is by construction (pre-impl NIT):** `session.tick` runs at
   the TOP of `update_and_draw`, the preview is drawn later in `_draw_app_panel`, so the mouse fed to tick is
   the PRIOR frame's hit-test — harmless and self-correcting, identical to the existing `dt`/`last_tick_time`
   pattern; do not reorder tick to "fix" it.
   **Export:** the `_make_export_isolation` closure injects the FIXED export value (not the live cursor) so a
   rendered video is deterministic — it MUST go in the export-isolation closure (the bypass-proof seam, 041
   decision 11), never on `App`, or GUI/Share/copilot/dogfood exports leak the live cursor. The fixed value
   is a module constant in `scripting/context.py`: `EXPORT_MOUSE = MouseState(0.5, 0.5)`.
   `behavior._build_globals` seeds `MouseState` into the exec-globals (scripts READ `ctx.mouse`; seed the
   type so a referenced annotation resolves). `stub_for`'s docstring gains a `ctx.mouse` line.

10. **`brain_stub_for(uniforms) -> str` is the one engine GAP 044 left; it MUST annotate bare `-> dict`.**
    A `class Behavior(ScriptBehavior)` whose `update` returns a dict literal pre-seeded with the node's
    `is_scriptable` uniforms at coercion-valid defaults. **NEVER `-> dict[str, Any]`** — `Any` is absent from
    the exec-globals → a permanent compile-freeze (044 OOS / 041 decision 12); use bare `-> dict`. Factor
    `stub_for`'s per-kind `(ann, default)` decision into a shared `_stub_kind(uniform)` helper so the
    per-uniform stub and the brain stub agree on defaults. Filter to `is_scriptable` uniforms; do NOT pre-list
    a uniform that already has a `u_<name>.py` (its value would be silently shadowed — confusing). A node with
    zero scriptable uniforms emits a valid empty-`{}` stub, not a crash. Its own pytest (compile the emitted
    text in the engine, assert no compile error — the bare-dict annotation is the load-bearing trap).

11. **"Make scriptable" + "New node-brain" write a file and reveal it; the next reload binds it.** "Make
    scriptable" (context menu, decision 6) calls a new `ProjectSession.create_script(node_id, name)` →
    `scripts_dir_for(node_id)/u_<name>.py` (lazy mkdir — `scripts_dir_for` is NOT eagerly created) writing
    `stub_for(uniform)` (pass the REAL `moderngl.Uniform` from `get_active_uniforms`, not the `UIUniform`
    wrapper), then `open_in_file_manager`. "New node-brain" (the "..." node-actions popup) calls
    `create_script(node_id, None)` → `scripts_dir_for/script.py` writing `brain_stub_for`, disabled when
    `script.py` already exists (one brain per node). Both gated on `not copilot_turn_active` (writing
    mid-turn races the reload). `create_script` lives on `ProjectSession` (headless; one home shared by both
    entry points).

12. **Detach is confirm-gated + trashes the file to a node-scoped subdir (recoverable). NO editor-session
    cleanup in wave 1.** "Detach script" / "Detach node-brain" uses `confirm_delete_popup(id_, label, on_confirm)`
    (a `begin_popup` — NOT modal, FPE-safe — with a `danger_button` + ghost cancel; the inline sibling of the
    grid-only `cell_delete_confirm`). The detach itself is a `ProjectSession` method (sibling to
    `create_script`, headless — the file mutation is project DATA). **Trash target (pre-impl SHOULD):** NOT
    bare `paths.trash_dir` — that holds node DIRECTORIES named by `node_id` and is scanned by
    `restore_node_from_trash`/`restore_checkpoint` (a bare `u_wave.py` there collides across nodes and
    pollutes the node-recovery namespace). Trash to a dedicated node-scoped subdir
    `trash_dir/scripts/<node_id>/<filename>` (preserves node scoping, no collision, outside the node-recovery
    scan). The next `reload_scripts` drops the binding and clears the stuck error. A brain detach frees ALL
    its driven uniforms — the confirm label says so.
    **No editor-session cleanup (pre-impl SHOULD):** in wave 1 a script opens via `open_in_file_manager` (OS
    editor) and NEVER through `get_session`/`open_shader_lib_file`, so a `.py` path can never be in
    `editor_sessions` / `_explicit_editor_path` / an `editor_jump_request`. The cleanup mirroring
    `_on_shader_lib_paths_removed` is therefore DEAD code for wave 1 — omit it. (When the in-app Python editor
    lands — Out-of-scope — detach then fires an `on_script_detached(path)` callback, default no-op, whose App
    handler pops the editor session, exactly the `on_node_deleted` idiom — `conventions.md` headless rule. Not
    needed until then.)

13. **Script management stays LIVE during a copilot turn (the read-only lock is GLSL-editor-only).** The
    `editor.set_read_only_enabled(copilot_turn_active)` lock is on the GLSL `TextEditor` only; the script
    chips/menus/strip are not that editor. Script management is NOT wrapped in `begin_disabled(copilot_turn_active)`
    EXCEPT the file-WRITE actions (Make scriptable / New node-brain / Detach), which ARE gated on
    `not copilot_turn_active` (decision 11/12) to avoid racing the turn's reload. Reading state (chips, errors,
    live value) and Restart are fine mid-turn. This is a convention note, not a component — recorded here +
    `conventions.md ## Known quirks` if it surprises later.

## Engine / session additions (the non-UI surface)

| Where | Add | Why |
|---|---|---|
| `scripting/engine.py` | `brain_stub_for(uniforms) -> str` + `_stub_kind(uniform)` (factored from `stub_for`) | decision 10 |
| `scripting/engine.py` | `ScriptEngine` accessor(s) exposing the brain's `last_skipped` soft-errors + driven count for a node (no `_nodes` poke from outside) | decision 8 |
| `scripting/context.py` | `MouseState(x, y)` frozen value type + `EngineContext.mouse` defaulted field + `EXPORT_MOUSE` constant | decision 9 (should) |
| `scripting/__init__.py` | export `MouseState` (+ `brain_stub_for`) | public surface |
| `scripting/behavior.py` | seed `MouseState` into `_build_globals` exec-globals | decision 9 (should) |
| `project_session.py` | `create_script(node_id, name \| None) -> Path`; the detach method (trash to `trash_dir/scripts/<node_id>/`); forward the brain-soft-error accessor; `tick(..., *, mouse: MouseState = EXPORT_MOUSE)` builds the live `EngineContext` from it; inject `EXPORT_MOUSE` in `_make_export_isolation` | decisions 8/9/11/12 |

## Reusable primitives (the ui_primitives.py additions — each general, not a call-site one-off)

| Primitive | Signature | Role |
|---|---|---|
| `script_chip` | `script_chip(state: Literal["none","per_uniform","brain"], id_: str) -> bool` | the tri-state row indicator + open launcher (over `pill_button`, which takes an arbitrary role color) — decision 5 |
| `notice_strip` | `notice_strip(id_, text, *, tone: Literal["info","ok","warn","error"], on_click=None, line: int \| None = None) -> None` | the ONE fixed-height inline banner — per-row error, brain sentinel, soft-key list — decision 7 |
| `confirm_delete_popup` | `confirm_delete_popup(id_, label, on_confirm) -> None` | the FPE-safe `begin_popup` destructive-confirm gate (detach + orphan) — decision 12 |
| `reset_state_button` | `reset_state_button(id_) -> bool` | the restart affordance reusing `revert_icon_button`'s drawn CCW-undo arc (no new glyph) — used in the brain strip; the per-row restart is a context-menu item, not this button |
| `item_normalized_mouse` | `item_normalized_mouse(rect_min, rect_max, *, flip_y=True) -> tuple[float,float,bool] \| None` | the reusable canvas hit-test — takes an EXPLICIT rect (not "the last item": `image_with_bg` submits none), ANDs `is_mouse_hovering_rect` with `is_window_hovered(child_windows)` for popup-blocking — decision 9 (should) |

`readonly_value_text` (decision 3) is NOT a new primitive — it's a direct reuse of `caption_text` +
`format_auto_value` (the auto-row treatment).

## Files touched

- `shaderbox/scripting/engine.py` — `brain_stub_for`, `_stub_kind`, the brain-soft-error accessor.
- `shaderbox/scripting/context.py` — `MouseState`, `EngineContext.mouse` (defaulted), `EXPORT_MOUSE`. (should)
- `shaderbox/scripting/behavior.py` — seed `MouseState` into exec-globals. (should)
- `shaderbox/scripting/__init__.py` — exports.
- `shaderbox/project_session.py` — `create_script`, the soft-error forward, mouse threading into `tick` +
  `_make_export_isolation`.
- `shaderbox/ui_primitives.py` — `script_chip`, `notice_strip`, `confirm_delete_popup`, `reset_state_button`,
  `item_normalized_mouse`.
- `shaderbox/widgets/uniform.py` — the driven-row branch (chip + read-only value + skip write-back), the
  per-row error line, the right-click context menu (open/make-scriptable/restart/detach).
- `shaderbox/tabs/node.py` — the node-brain strip; the "Right-click for script actions" hint; the
  "New node-brain script" item in the node-actions popup.
- `shaderbox/ui.py` — capture the preview-image rect + the mouse hit-test (`item_normalized_mouse`) → stash
  on `App`; pass `App`'s stashed mouse into `session.tick(..., mouse=...)`. (should)
- `shaderbox/app.py` — the stashed normalized-mouse field (default `EXPORT_MOUSE`). NO detach editor-session
  cleanup in wave 1 (decision 12 — scripts never enter `editor_sessions` via the OS-editor path).
- `tests/` — `brain_stub_for` compile test (the bare-`dict` annotation trap); the `ctx.mouse`
  live-vs-export-freeze invariant (extend the script-engine suite). `scripts/smoke.py` — extend its seed: it
  currently seeds ONLY a per-uniform `u_wave.py` node, so add a SECOND node carrying a `script.py` brain that
  drives ≥1 real uniform + 1 typo'd homeless key, so `make smoke` actually draws the brain strip + brain chip
  + soft-key list (today's seed covers only the per_uniform chip + per-row error path).
- Docs: `roadmap.md` (042 row + banner → next 043), `todo.md` (the type-change/orphan 042 deferral: 042
  RESOLVES the orphan half via Detach + SURFACES the type-change half via the per-row error line — an
  automated type-aware re-stub is NOT in scope, so rewrite the entry to the residual "type-change surfaced,
  not auto-resolved" rather than delete it; the node-brain `stub_for` deferral is RESOLVED by `brain_stub_for`
  → delete in the same commit; the 043 rollback-capture deferral is verified UNCHANGED — 042's file writes are
  user-initiated, not copilot mutating tools, and are gated `not copilot_turn_active`),
  `conventions.md` (the decision-13 read-only-lock note if it earns a Known-quirk).

## Manual verification (run the app — `make run`)

The whole feature is visual; `make smoke` only proves it draws without crashing. Hand to the maintainer:
1. A node with a `u_<name>.py` → the row shows a `per_uniform` chip, the control is read-only and the value
   ticks live; right-click the NAME cell → Open/Restart/Detach work; the OS editor opens the right file;
   left-click the name still jumps to the declaration (right- and left-click coexist on the selectable).
2. Break the `u_<name>.py` (a syntax error / a vec2→vec3 type mismatch) → the per-row error line shows the
   message + line; fix it → the line clears on hot-reload. (The type-mismatch case is the surfaced half of
   the type-change deferral — the fix is hand-editing the return shape, no auto-restub.)
3. A driven NON-scalar: a vec3/color and a text uniform each driven by a `u_<name>.py` → the read-only
   readout shows the live value, no flicker, no write-back clobber (covers the color/array/text branches,
   not just scalar).
4. A node with `script.py` → the brain strip shows "drives N uniforms"; the brain-driven REAL-uniform rows
   show the `brain` chip (distinct from per_uniform); a typo'd brain key → the strip's soft-key list shows
   it (NOT a row); a broken brain → the strip shows the sentinel error (no per-row errors, since it drives
   nothing).
5. "Make scriptable" on an undriven scalar → writes the stub, the OS editor opens, the next frame the row
   flips to `per_uniform`. "New node-brain" → writes `brain_stub_for`, disabled when one exists.
6. Detach → confirm → the file is trashed (to `trash_dir/scripts/<node_id>/`), the binding drops next
   reload, the chip clears. (No editor session to clean in wave 1 — scripts open in the OS editor.)
7. File-write actions (Make scriptable / New node-brain / Detach) are disabled during a copilot turn;
   Restart + reading state stay live.
8. (should) A script reading `ctx.mouse` reacts to the cursor over the preview; an exported video is
   deterministic (the cursor is frozen at the fixed value, not the live position).

## Open questions for the user (defaults already applied — proceeding autonomously)

The swarm surfaced 5; the dev-flow "robust default" was applied to each. Listed for retro-adjust:
1. **Edit in-app vs OS-editor?** → DEFAULT: **OS-editor for wave 1** (minimum-complete, zero FPE/coupling
   risk); in-app editor is a captured `could` (Out-of-scope).
2. **Driven control: live read-only readout vs static caption?** → DEFAULT: **live read-only readout**
   (the existing auto-row treatment, costs nothing — the value is already in `uniform_values`).
3. **Detach: confirm-gated + trash vs hard-delete?** → DEFAULT: **confirm-gated + trash** (matches the
   node-delete precedent).
4. **`ctx.mouse` only vs also an auto-fed `u_mouse` uniform?** → DEFAULT: **`ctx.mouse` only** (one seam;
   the auto-fed uniform is CUT to Out-of-scope).
5. **Surface the brain-vs-`u_<name>.py` shadow conflict?** → DEFAULT: **no** (CUT — corner case the row
   chip already partially discloses).

Plus one scoping question the swarm implied: **is the `ctx.mouse` subsystem (a `should`) IN this 042, or
split to a 042b?** → DEFAULT: **in 042 but implemented LAST**, after the must row/strip/lifecycle items land
and pass review — so if it slips it doesn't block the core. If the maintainer wants 042 leaner, it cleanly
splits out (its own engine+UI seam, no entanglement with the row/strip work).

## Review history

**Pre-implementation review (2026-06-13, 2 adversarial agents anchored to the engine source).** Verdict:
needs-changes (no blocker that should-not-land; all findings folded into the decisions above). Folded:
- **BLOCKER — `image_with_bg` submits no interactive item** → `is_item_hovered` after it is non-functional.
  Decision 9 + `item_normalized_mouse` rewritten to take an EXPLICIT rect (`get_cursor_screen_pos` + image
  size) and AND `is_mouse_hovering_rect` with `is_window_hovered(child_windows)` (the `/imgui-ui` §3 pattern).
- **BLOCKER — the per-row context menu had no anchor on a driven row** (decision 3 returns before the control,
  leaving only plain `caption_text`). Decision 6 now anchors the menu to the `uniform_name_label` selectable
  (interactive on every row; right- and left-click coexist).
- **SHOULD — `script_chip` "over `chip_button`" was impossible** (it can't take an arbitrary `brain` color).
  Decision 5 + the primitives table re-based it on `pill_button` (which takes `color=`).
- **SHOULD — detach-trash collision** in the shared node `trash_dir`. Decision 12 now trashes to a node-scoped
  `trash_dir/scripts/<node_id>/` subdir (no cross-node collision, outside the node-recovery scan).
- **SHOULD — detach editor-session cleanup is DEAD code in wave 1** (scripts never enter `editor_sessions` via
  the OS-editor path). Dropped from decision 12 + Files-touched; the `on_script_detached` callback is deferred
  to the in-app-editor wave.
- **SHOULD — `tick()` signature** would break the 2 positional dogfood callers. Decision 9 pins
  `tick(..., *, mouse: MouseState = EXPORT_MOUSE)` (defaulted kwarg).
- **SHOULD — type-change deferral only PARTIALLY folded.** Made honest: 042 RESOLVES the orphan half (Detach)
  and SURFACES the type-change half (per-row error line); no auto-restub. The todo.md entry is rewritten to the
  residual, not deleted.
- **SHOULD — smoke + manual-verification gaps** (no brain seed, no non-scalar driven row, no brain-real-uniform
  chip). Extended both: smoke seeds a second brain node; manual-verification adds the vec/color/text driven row
  + the brain-chip-on-a-real-uniform checks.
- **NITs folded:** the one-frame mouse lag is documented as by-construction; the readout reuses
  `util.format_auto_value` (not the inline auto-branch); the 043 rollback-capture deferral is confirmed
  unfired.

(Post-impl review populated after implementation.)
