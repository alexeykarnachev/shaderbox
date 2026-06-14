# 047 — Scripting UX refinement wave

A maintainer-walkthrough-driven UI/UX + behavior refinement of the scripting feature (041/042/044/045),
same shape as 028/034. Driven by two voice walkthroughs (foo msgs 1868/1869, 2026-06-14). Fourteen
findings (F1–F14) in scope.

Two architectural corrections anchor the wave, both maintainer-called:
- **Active/inactive is UI intent and belongs on the model (`UIUniform` / `UINodeState`), NOT smeared on
  the filesystem as a `<file>.disabled` marker.** The 045 marker mechanism is deleted; a boolean flag
  replaces it. (Decisions 1–4.)
- **A per-uniform script binds by `(name, type)`, not by `name` alone.** The script filename encodes the
  uniform's type tag, so a retype (`vec2`→`vec3`) can no longer corrupt a script (the old one is kept,
  the new shape gets a fresh script, and the change is reversible), and a delete-then-readd rebinds
  automatically. Rename recovery + cross-node reuse come via a "copy content from another script of the
  same type" selector, not by re-pointing a binding. (Decisions 13–15 — this is F14.)

Several smaller findings fall out of these cleanly.

## Goal

Refine the scripting UI and its active/inactive semantics so the surfaces are calm, the state model is
honest, engine-owned uniforms stay off every script surface, and script code survives shader-side
uniform edits. Specifically: replace the bulky pills with a small theme-colored glyph; make
active/inactive a model flag (born inactive); give scripts a script-local control bar (activate +
copy-from-another-script); fix the brain engine-owned error; tidy the error strip; fix the docstrings;
adopt the native tab bar with bare-filename labels; let users hand-edit script-driven uniforms (the tick
re-asserts); and key per-uniform scripts by `(name, type)` so retype/delete-readd are lossless.

## Out of scope

- **The copilot write-behavior tool (043)** — unrelated; stays next on the roadmap after this wave.
- **A scripts LIBRARY (fuzzy tree picker, cross-project reuse, à la the shader lib).** The F14 selector
  is a flat whole-project dropdown for now. **Trigger:** first time the maintainer wants to maintain a
  reusable library of scripts (reuse across projects, fuzzy search, tags) — lift the flat selector to a
  shader-lib-style picker (`popups/lib_picker/` is the pattern). Parked deliberately; build the flat
  selector first and let real use prove the need.
- **Rename auto-rebind (re-pointing a script's binding to a renamed uniform).** F14 does NOT re-point
  bindings; a renamed uniform's old script strands on disk and the user recovers its code via the
  copy-content selector into the new uniform's fresh script. An automatic orphan→rebind affordance is
  out of scope (the copy path covers the real need without orphan-binding machinery).
- Per-file "modified vs shipped" lib badges, and the other open `todo.md` scripting deferrals not named
  here. The existing `todo.md` "script-engine type-change error" deferral is RESOLVED by F14 (the
  `(name, type)` key removes the corruption it described) — delete that entry in the impl commit.

## Design decisions (locked)

1. **Active/inactive is a model flag, not a marker file.** Add `is_script_active: bool = False` to
   `UIUniform` and `is_brain_active: bool = False` to `UINodeState` (`ui_models.py`). Both default
   `False`. Delete the entire `.disabled` mechanism: `_DISABLED_SUFFIX`, `_is_disabled`, the two
   `_is_disabled(path)` gates (`engine.py` per-uniform discovery + `_reload_brain`), the marker
   create/unlink in `set_script_active`, and the stale-marker cleanup in `detach_script`. The flag is
   the single source of truth, persisted in `node.json`.

2. **The engine reads active-intent via an explicit param, not the node model.** `ScriptEngine.reload`
   gains an `active_scripts: set[str]` arg (active filenames: `"u_<name>.py"`, `"script.py"`).
   `ProjectSession.reload_scripts` builds it from the flags on `ui_node` (it owns the UI model) and
   passes it in. The engine never imports/learns `UIUniform` — the headless boundary (041) holds;
   intent flows through a param exactly as `engine_driven` already does. Discovery skips any script
   whose filename isn't in `active_scripts` (replacing the `_is_disabled` skip); the drop loop then
   tears down a binding that just went inactive (same as today).

3. **Born inactive (F2).** Because the flags default `False`, a freshly-created script is inactive with
   zero extra code — `create_script` writes only the file. Opening a script (pill/glyph click) creates
   (if absent) + opens it in the editor; it does NOT flip any flag. Activation is manual, via the
   script-local bar's Activate button (decision 6). On upgrade, every existing script reads inactive
   (absent flag → default `False`); the maintainer re-arms via the bar. Additive optional pydantic fields
   — no migration code (same precedent as 018 `key_bindings`).

4. **Brain-active and each uniform-script-active are fully independent (F12).** Nothing cascades:
   activating the brain does not activate any `u_*.py`; they are separate flags. This is automatic once
   the flags are independent model fields — the old coupling was an artifact of the marker scan.

5. **Engine-owned uniforms are hidden from the SCRIPT, never script-targetable (F4).** `u_time`,
   `u_aspect`, `u_resolution`, and the table uniforms (`ENGINE_DRIVEN_UNIFORMS`) are excluded from
   `stub_for`/`brain_stub_for` (they already aren't `is_scriptable`, so confirm the stub generators
   filter them out), and the brain SILENTLY drops any engine-owned key it returns — no error, no
   "engine-owned" soft-error class. The whole "engine-owned" branch of the bad-key error path is
   removed: an engine-owned key can no longer reach the brain (not in the stub) and is dropped if
   hand-added. **Node UI is UNCHANGED** — engine-owned uniforms still render as the blue read-only rows
   at the top of the node tab (they are driven by the renderer; that display is correct). "Hidden from
   the user" here means only that they are not script-targetable; a user who wants a controllable time
   declares their own uniform (e.g. `u_my_time`) and scripts that.

6. **A script-local control bar under the tab bar (F3 + F14 home).** Remove the active/inactive toggle
   from the editor tab row entirely. Add a NEW thin bar directly under the file-selector/tab bar, drawn
   ONLY when the active tab is a script (`node_script` / `uniform_script`), holding (for now) two
   controls left-to-right:
   1. **Activate / Deactivate button** — larger, clear label: `Activate script` when inactive,
      `Deactivate script` when active. The one place the active flag is toggled (decision 1). Frozen
      mid-copilot-turn.
   2. **"Copy content from…" selector** (F14, decision 15) — a dropdown of other scripts of the SAME
      uniform type across the whole project; picking one copies its body into the current editor buffer.
   The floating-editor-overlay idea is dropped — the bar is fixed, discoverable, scroll-independent, and
   has room to grow (future script-level controls land here). A node-brain tab shows the bar with the
   Activate button only (no copy selector — the brain isn't keyed to a uniform type; see decision 15).

7. **The pill becomes a small `</>` glyph (F1).** Replace `script_pill` (the per-uniform row trailing
   control AND the node-header brain control) with a small `</>` text glyph (literal `</>`), theme-colored
   by state: `FG_DIM`/`FG_SECONDARY` grey when `absent`, `ACCENT_PRIMARY` when `active`, faded accent
   (`fade(ACCENT_PRIMARY, …)`) when `inactive`, `STATE_ERROR` red when `error`. Click opens (creating if
   absent); never toggles. The node-brain header uses the same glyph. Tooltips carry the semantics the
   pill text used to.

8. **Native tab bar, bare-filename labels (F9 + F10).** Swap `_draw_tab_row`'s hand-rolled
   toggle-button-strip-in-a-scroll-child for `imgui.begin_tab_bar` (`reorderable | fitting_policy_scroll
   | tab_list_popup_button | draw_selected_overline`), per-tab `p_open` close, `unsaved_document` flag for
   the dirty dot, and `STATE_ERROR`-tinted tab for an erroring script tab (push `Col_.tab*` around its
   `begin_tab_item`). Tab labels are the **bare file name** (`u_position.py`, `script.py`,
   `shader.frag.glsl`, a lib's filename) — delete `_tab_label`'s semantic-alias branching entirely.
   NOTE: a per-uniform script's on-disk filename now encodes the type tag (decision 13), so its bare
   filename label reads e.g. `u_position__vec2.py`; acceptable (it IS the file). The tab bar no longer
   hosts the active toggle (it moved to the script-local bar, decision 6). Spike validated in
   `scripts/_tabbar_spike.py` (throwaway, deleted at sweep). Tab selection must still drive
   `active_tab_index` and survive reorder — read back imgui's selected tab rather than assuming index.

9. **Script-driven uniform widgets stay editable; the tick re-asserts (F11).** Stop disabling a uniform's
   value widget when a script owns it. The user CAN drag/edit it, but the script's next `tick` overwrites
   the value (it visibly snaps back) for any uniform the script actually drives — that snap-back is the
   "the script owns this" cue. A uniform the script does NOT drive (brain active but doesn't return that
   key) stays fully live and the manual value sticks. Concretely: remove the `imgui.begin_disabled(owned)`
   wrap in `draw_ui_uniform`; KEEP the "don't write the unchanged return back" skip for owned slots
   (`if new_value is not None and not owned`) so a manual edit during the same frame still applies but the
   tick wins next frame. (`is_uniform_script_owned` stays — it gates the write path, not the widget.)

10. **Error strip tidy (F5 + F6).** Remove the dead empty trailing row under "+N more" (a height /
    row-count miscount in `_draw_error_strip` / its `strip_height` math). Make "+N more" a clickable
    control that expands the strip to show ALL errors (lift the `_MAX_ERROR_ROWS` cap on click — a
    per-strip expanded flag on App, reset per tab). Clicking an error still jumps to its line.

11. **Docstrings (F7 + F8).** Rewrite `_UPDATE_DOC` from the pipe-crammed one-liner into a proper
    Google-style docstring (an `Args:`-style block documenting `ctx.t` / `ctx.dt` / `ctx.frame` /
    `ctx.mouse`). No invented prose. Add the missing blank line between the class docstring and
    `__init__` in both `stub_for` and `brain_stub_for`.

12. **Text-uniform count caption overlap (F13).** The `(n/cap)` caption for a `text` uniform overlaps the
    text input field. Re-anchor it (glue it to the uniform name column, per `_count_suffix`'s intent) so
    it no longer collides with the control. Fix in `widgets/uniform.py` (`_count_suffix` placement /
    `_begin_ctrl`).

13. **A per-uniform script binds by `(name, type)` — the filename encodes the type tag (F14 core).** Today
    a uniform script is `nodes/<id>/scripts/u_<name>.py`, bound by name alone — so a shader-side retype
    (`vec2 u_x` → `vec3 u_x`) keeps the binding but breaks the return shape (permanent coercion error), and
    the by-name match can't tell a retype from a rename. Change the per-uniform filename to encode the
    type tag: `u_<name>__<tag>.py`, where `<tag>` is the coercion signature `_stub_kind` already computes
    (`vec2`/`vec3`/`vec4`/`array`/`text`/`float`/`int` — a lowercased, filename-safe form of the existing
    annotation). The engine binds a script to a uniform only when BOTH name and type tag match the live
    uniform. Consequences, all desirable:
    - **Retype is lossless + reversible.** `vec2 u_x` → `vec3 u_x`: the live uniform is now `(u_x, vec3)`
      → looks for `u_x__vec3.py` → absent → a fresh script (no error). `u_x__vec2.py` stays on disk
      untouched; retype back to `vec2` and it rebinds. No coercion error ever fires from a retype.
    - **Delete-then-readd rebinds automatically.** Re-adding `vec2 u_x` finds `u_x__vec2.py` still on disk
      and rebinds it.
    - **Rename strands the old file** (engine can't distinguish rename from delete+add) — recovered via the
      copy-content selector (decision 15), NOT an auto-rebind.
    The node-brain (`script.py`) is UNCHANGED — it's name-agnostic (returns a dict, validated per key
    per tick), so it keeps its single filename and by-key behavior. This change ripples through every
    per-uniform `f"{name}.py"` / `script_path_for` / `script_state_for` / glob (`_SCRIPT_GLOB` →
    `u_*__*.py`) / error-key path: a per-uniform binding is now keyed by `(name, tag)`, the error key by
    `(node_id, "u_<name>__<tag>.py")` or an equivalent stable key. The type-tag derivation has ONE home (a
    helper beside `_stub_kind`) shared by the filename builder and the discovery matcher, so the two never
    disagree. The existing `todo.md` "type-change error" deferral is resolved by this — delete it.

14. **`active_scripts` keying follows the new filename (F14 ↔ decision 2).** Since the per-uniform file is
    now `u_<name>__<tag>.py`, the `active_scripts` set decision 2 passes to `reload` holds those full
    filenames (plus `script.py` for the brain). The `is_script_active` flag on `UIUniform` is keyed to the
    uniform (name+type live on the uniform itself), so the session builds the active filename for a uniform
    from its live `(name, tag)` — a uniform that retypes gets a fresh flag-default-False script, consistent
    with decision 3 (born inactive).

15. **"Copy content from another script of the same type" selector (F14 UI).** In the script-local bar
    (decision 6), for a per-uniform script tab, a dropdown lists every OTHER per-uniform script across the
    WHOLE PROJECT whose type tag matches the current uniform's tag (scan each node's `scripts/` dir, parse
    the tag from each filename, filter to matching tag, exclude the current file). Each option is labeled by
    its node + uniform (e.g. `nodeA / u_velocity`) so duplicates are distinguishable. Picking one COPIES
    that file's body into the current editor buffer (a plain text copy into the live `TextEditor`, marking
    it dirty — the user saves to persist) — it is NOT a live re-bind or an alias ("copy content from", not
    "use that script"). This is how rename-recovery + cross-node reuse work without any orphan-binding
    machinery: a renamed uniform's stranded `u_old__vec2.py` shows up in the new `u_new__vec2.py`'s selector
    (same tag), and one click pulls the code over. The selector is empty (disabled/hidden) when no other
    same-type script exists. NOT shown for a node-brain tab (the brain has no uniform type). A future
    scripts-library (fuzzy picker, cross-project, tags) replaces this flat dropdown — parked (Out of scope).

## Files touched

- `shaderbox/ui_models.py` — `is_script_active` on `UIUniform`; `is_brain_active` on `UINodeState`.
- `shaderbox/scripting/engine.py` — delete `_DISABLED_SUFFIX`/`_is_disabled` + the two skip gates;
  `reload(active_scripts=…)`; drop the engine-owned bad-key error branch; rewrite `_UPDATE_DOC` +
  the stub blank-line (F7/F8); confirm stubs exclude engine-owned uniforms (F4); the `(name, type)`
  filename key (F14 decision 13): `_SCRIPT_GLOB` → `u_*__*.py`, a shared type-tag helper beside
  `_stub_kind`, discovery matches `(name, tag)`, per-uniform error keys re-keyed.
- `shaderbox/project_session.py` — `reload_scripts` builds `active_scripts` from the flags;
  `set_script_active` flips the model flag (not a marker); `script_state_for` reads the flag;
  `create_script`/`script_path_for` build the `u_<name>__<tag>.py` name (F14); `detach_script` drop the
  marker cleanup; brain returns drop engine-owned; a new `same_type_scripts(node_id, name)` enumerating
  whole-project matching-type scripts (F14 selector source) + a `copy_script_body(src_path)` helper.
- `shaderbox/ui_primitives.py` — `script_pill` → the small `</>` glyph (`script_glyph`?), theme-colored.
- `shaderbox/widgets/uniform.py` — glyph call; remove `begin_disabled(owned)` (F11); F13 caption fix.
- `shaderbox/tabs/node.py` — brain header uses the glyph.
- `shaderbox/tabs/code.py` — native tab bar + bare-filename labels (F9/F10); the NEW script-local bar
  (F3+F14: Activate/Deactivate button + copy-content selector); error-strip tidy + expandable "+N more"
  (F5/F6); remove `_draw_script_toggle` from the tab row.
- `shaderbox/app.py` (or `editor_types.py`) — tab selection read-back for the native bar; the
  expanded-errors flag; the script-bar wiring (activate + copy-from); `EditorTab` label simplification.
- `scripts/_tabbar_spike.py` — DELETED at the sweep.

## Manual verification (maintainer `make run` — headless can't judge these)

Per dev_flow step 7, each check falsifiable + names the consumer:
- **F1 glyph** — every scriptable uniform row + the node header show a small `</>`; grey when no script,
  accent when active, faded when inactive, red on a known-broken script. Falsifier: a broken script shows
  red, a fresh one grey.
- **F2/F12 born-inactive + independence** — create a uniform script → it is inactive (value still
  hand-editable, no snap-back). Activate the brain → the uniform script stays inactive. Falsifier:
  activating the brain must NOT turn the uniform glyph accent.
- **F3 script-local bar** — a thin bar appears under the tab bar ONLY for a script tab, holding the
  Activate/Deactivate button + (per-uniform only) the copy-content selector; it does NOT appear for a
  shader/lib tab. Activate label flips Activate↔Deactivate and toggles the glyph/snap-back. Falsifier:
  switch to a shader tab — the bar vanishes.
- **F4** — activate a brain whose stub drives only real uniforms → NO "engine-owned" error in the strip.
  Hand-add `"u_time": ...` to the brain dict → it is silently dropped, no error. Falsifier: the strip
  stays empty.
- **F5/F6** — an erroring script with >3 errors shows no blank trailing row; "+N more" is clickable and
  expands to all errors; clicking an error jumps. Falsifier: count rows — no empty last row.
- **F7/F8** — open a fresh script → the `update` docstring is a clean Google-style block (no `|` pipes);
  a blank line separates the class docstring from `__init__`.
- **F9/F10** — the tab row is the native bar (real tabs, ×-close, ▾ overflow, drag-reorder, unsaved-dot,
  red error tab); labels are bare filenames. Falsifier: open 10+ scripts → overflow scrolls + dropdown.
- **F11** — with a uniform script active, drag its value → it visibly snaps back next tick; with the
  brain active but not driving a given uniform, that uniform edits and STICKS. Falsifier: the driven one
  snaps back, the undriven one does not.
- **F13** — a text uniform's `(n/cap)` caption does not overlap its input field.
- **F14 retype** — write a script for `vec2 u_x`, activate it; in the shader change to `vec3 u_x` → the
  uniform gets a fresh (inactive) script, NO coercion error; change back to `vec2` → the original script
  rebinds with its code intact. Falsifier: retyping must not error AND must not lose the vec2 body.
- **F14 delete-readd** — delete `u_x` from the shader (script strands), re-add `vec2 u_x` → its script
  rebinds. Falsifier: the re-added uniform's script is the original, not a blank stub.
- **F14 copy-content** — rename `u_x`→`u_y` (same type); open `u_y`'s fresh script → the copy selector
  lists the stranded `u_x` script (same tag); pick it → its body lands in the editor. A script from
  ANOTHER node of the same type also appears. Falsifier: the selector lists only same-type scripts and
  excludes the current file.
- **`make check` + `make smoke` green** (smoke seeds a script-driven node — the flag-model + `(name,type)`
  filename reload must not crash it; update the smoke seed's script filename to the new scheme).

## Open questions for the user

- F3 button label wording: `Activate script`/`Deactivate script` vs `Script inactive`/`Script active`
  (settled at impl unless you have a preference).
- F14 type-tag spelling in the filename: `u_x__vec2.py` (double-underscore separator) — flagging the
  exact on-disk name since it becomes the tab label too. Change if you dislike the look.
- None blocking.

## Review history

(empty — to be filled at pre/post-impl review.)
