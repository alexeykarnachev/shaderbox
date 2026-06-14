# 047 — Scripting UX refinement wave

> **SUPERSEDED IN PART by 048 (`ai_docs/features/048_single_script_play_stop.md`).** The per-uniform-script
> half of this wave is removed: the `(name,type)` type-tag binding (F14, decisions 13–15), the copy-content
> selector, the small `</>` glyph as a per-row affordance, the script-local Activate/Deactivate bar, and the
> `is_script_active`/`is_brain_active` MODEL FLAGS (decisions 1–6) are all DELETED — a node now has ONE
> script (the brain), bound by existence, with a play/stop model. What SURVIVES from 047: the native imgui
> tab bar (decision 8, now with node-derived labels), the editable-widget-with-snap-back idea (decision 9,
> recast as auto-stop), the error-strip tidy (decision 10), the docstring fixes (decision 11), and the
> engine-owned-uniform hiding (decision 5). Read 048 for the current model; this spec is the history.

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
- **No copilot-reachable durable mutation (the turn-rollback deferral does not fire).** 047 adds
  `set_script_active` (a UI-bar flag flip, frozen mid-turn) and `copy_script_body` (a write into the live
  `TextEditor` buffer, not durable disk — the user saves to persist). The copilot has NO script-writing
  tool today (043 is out of scope). So the `todo.md` "copilot turn-rollback: a NEW mutating tool"
  deferral's trigger does NOT fire here — it fires for 043. Stated explicitly so the negative is provable.

## One-time migration (impl-commit, not deferred)

The `_SCRIPT_GLOB` change `u_*.py` → `u_*__*.py` (decision 13) means EXISTING on-disk per-uniform scripts
no longer match — they go undiscovered (not even orphan-warned; they never enter the glob loop), so their
uniforms silently revert to manual on first launch. This affects the live dev sandbox: the four scripts in
`projects/dev/nodes/4c6c34f4-…/scripts/` (`u_char_height.py` / `u_color1.py` / `u_color2.py` /
`u_spacing.py`). The impl commit RENAMES them to the tagged scheme (e.g. `u_color1__vec3.py`) and re-saves
the node's `node.json` flags in the same `git add projects/dev` sandbox-sync wave the hard rules already
mandate. The `script.py` brain is unaffected (separate discovery branch). Verified by the F14-migration
manual check below.

## Design decisions (locked)

1. **Active/inactive is a model flag, not a marker file.** Add `is_script_active: bool = False` to
   `UIUniform` and `is_brain_active: bool = False` to `UINodeState` (`ui_models.py`). Both default
   `False`. Delete the entire `.disabled` mechanism: `_DISABLED_SUFFIX`, `_is_disabled`, the two
   `_is_disabled(path)` gates (`engine.py` per-uniform discovery + `_reload_brain`), the marker
   create/unlink in `set_script_active`, and the stale-marker cleanup in `detach_script`. The flag is
   the single source of truth, persisted in `node.json`.

2. **The engine reads active-intent via an explicit param, not the node model.** `ScriptEngine.reload`
   gains an `active_scripts: set[str]` arg (active filenames: `"u_<name>__<tag>.py"`, `"script.py"`).
   `ProjectSession.reload_scripts` builds it from the flags on `ui_node` (it owns the UI model) and
   passes it in. The engine never imports/learns `UIUniform` — the headless boundary (041) holds;
   intent flows through a param exactly as `engine_driven` already does. Discovery skips any script
   whose filename isn't in `active_scripts` (replacing the `_is_disabled` skip); the drop loop then
   tears down a binding that just went inactive (same as today). **The skip is the active-flag's only
   READER — verified by the F2-wire check** (manual-verification): set the flag false out-of-app, the
   driven value must stop snapping back; if discovery ignores `active_scripts` the value keeps snapping
   back (the dead-mechanism failure dev_flow step 7 warns about). **`active_scripts` is the keyed gate,
   not a default-on:** `reload` takes it as a required arg from the two live callers
   (`project_session.py:308` `_resolve_scripts`, `:362` `reload_scripts`). The ~34 test call sites
   (`tests/test_script_engine.py` 28×, `tests/test_script_engine_gl.py` 6×) need it too — give it a
   default `frozenset()` ONLY if a per-test helper passes the active set explicitly; the 6 `.disabled`
   tests are deleted and replaced with active-set tests (see Files touched).

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
   hand-added. **This touches the BRAIN TICK path, not just the stub (pre-impl review A5/B):** the
   engine-owned reason in `_binding_reject` (`engine.py:384`) is consulted at TWO sites — per-uniform
   reload (now moot: an engine-owned key can't be a per-uniform filename) AND the brain's per-key lazy
   validation at tick (`engine.py:608`), which today records a SOFT error + skips the key. Decision 5
   changes the brain branch to `continue` SILENTLY (no `errors`/`soft_errors`/`skipped` record) for an
   engine-owned key — a behavior change to `_apply_behavior`'s brain branch + `BrainStatus.soft_errors`,
   not merely deleting line 384. **Node UI is UNCHANGED** — engine-owned uniforms still render as the blue read-only rows
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
   `shader.frag.glsl`, a lib's filename) — delete `_tab_label`'s semantic-alias branching entirely
   (it has a SECOND caller, `draw_chrome` at `code.py:198` — switch it to `tab.path.name` too).
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
    per tick), so it keeps its single filename and by-key behavior.
    **FILENAME-tag vs internal-KEY — the load-bearing distinction (pre-impl review A1/B):** the `__<tag>`
    lives ONLY in the on-disk filename and the tab label. Every INTERNAL key stays the **bare uniform
    name** (`u_x`, not `u_x__vec2`). Today `reload` does `name = path.stem` (`engine.py:291`) and keys
    `behaviors[name]` / `mtimes[name]` / `sources[name]` / `last_good[name]` / `errors[(node_id, name)]` /
    the `found` set / the drop loop / `script_driven_uniforms` by that stem, AND `_binding_reject(name,
    active)` looks the name up in the live-uniform map. If the stem becomes `u_x__vec2` every one of those
    lookups false-orphans. So discovery PARSES the filename into `(uniform_name, tag)`, validates `tag`
    against the live uniform's tag, and then keys all internal state by `uniform_name` — **the tag is a
    match gate, never a dict key.** Two shared helpers beside `_stub_kind` (the ONE home, so builder and
    matcher never disagree): `_uniform_tag(uniform) -> str` (the lowercased coercion-signature tag) and
    `_parse_script_filename(filename) -> tuple[str, str] | None` (`u_x__vec2.py` → `("u_x", "vec2")`,
    `None` for a non-conforming name). The error key stays `(node_id, "u_x")` (bare), so
    `script_state_for` / `is_uniform_script_owned` / `uniform_pill_state` — which key by bare uniform name
    — keep working unchanged. `script_file_for` (`engine.py:261`) + its copilot consumer
    (`backend.py:690`, the `f"{name}.py"` fallback) must reconstruct the tagged filename (the engine stores
    the discovered filename alongside the binding, or `script_file_for` re-derives the tag from the live
    uniform). This change ripples through `_SCRIPT_GLOB` (→ `u_*__*.py`), `script_path_for` /
    `create_script` (build the tagged name), and `script_file_for`. The existing `todo.md` "type-change
    error" deferral is resolved by this — delete it in the impl commit.

14. **`active_scripts` keying follows the new filename (F14 ↔ decision 2).** Since the per-uniform file is
    now `u_<name>__<tag>.py`, the `active_scripts` set decision 2 passes to `reload` holds those full
    filenames (plus `script.py` for the brain). The `is_script_active` flag on `UIUniform` is keyed to the
    uniform (name+type live on the uniform itself), so the session builds the active filename for a uniform
    from its live `(name, tag)`. **Retype must not leak the old active flag onto the fresh file
    (pre-impl review B):** one `UIUniform` named `u_x` carries ONE `is_script_active` flag; after
    `vec2`→`vec3` that same flag (possibly still `True` from the vec2 era) would otherwise mark the brand-new
    `u_x__vec3.py` active, contradicting decision 3 (born inactive). The gate: `reload_scripts` adds
    `u_x__<currenttag>.py` to `active_scripts` ONLY when (`is_script_active` is True) AND (that exact tagged
    file exists on disk). A retyped uniform's vec3 file is absent → not added → born inactive even with a
    stale `True` flag; retyping back to vec2 finds `u_x__vec2.py` again and (flag still True) re-arms it.

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
  `reload(active_scripts=…)`; drop the engine-owned per-uniform reject branch AND make the brain tick
  (`_apply_behavior` at `engine.py:608`) drop an engine-owned key SILENTLY (no `soft_errors` record — F4,
  decision 5); rewrite `_UPDATE_DOC` +
  the stub blank-line (F7/F8); confirm stubs exclude engine-owned uniforms (F4); the `(name, type)`
  filename key (F14 decision 13): `_SCRIPT_GLOB` → `u_*__*.py`, two shared helpers beside `_stub_kind`
  (`_uniform_tag` + `_parse_script_filename`), discovery parses `(name, tag)` from the filename and keys
  internal state by the BARE name (the tag is a match gate, not a key), `script_file_for` (`engine.py:261`)
  reconstructs the tagged filename.
- `shaderbox/copilot/backend.py` — the `f"{name}.py"` fallback at `backend.py:690` becomes the tagged
  filename (rides `script_file_for`; the fallback string updates to the new scheme).
- `ai_docs/conventions.md` — the 041 Design-decision bullet says "Binding is by FILENAME (stateless — no
  `node.json` entry, no `UIUniform` field)" (`conventions.md ## Design decisions`, the scripting bullet).
  047 makes the second clause false: the BINDING stays filename-based, but ACTIVE-INTENT now lives on a
  persisted `UIUniform`/`UINodeState` flag. Update the clause to that nuance (add active-intent-on-model;
  keep binding-by-filename) — same-wave per "docs are living".
- `tests/test_script_engine.py` / `tests/test_script_engine_gl.py` — ~34 `reload(...)` call sites gain the
  `active_scripts` arg; the per-uniform test fixtures adopt the `u_<name>__<tag>.py` filename; the 6
  `.disabled` tests (the deleted marker mechanism) are removed and replaced with active-set tests.
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
- **F2/F12 born-inactive + independence** — create a uniform script → its glyph reads `inactive`
  (faded accent), NOT `absent` (grey) — proving the file was discovered-but-flag-off, not silently
  undiscovered — and the value stays hand-editable with no snap-back. Activate the brain → the uniform
  script stays inactive. Falsifier: activating the brain must NOT turn the uniform glyph accent; a GREY
  (absent) glyph on a just-created script means discovery never found it (the glob/migration bug), not
  born-inactive.
- **F2-wire (the flag's only consumer)** — with a uniform script ACTIVE and visibly driving its value,
  edit `node.json` to set `is_script_active: false` out-of-app, reload → the value STOPS snapping back
  (the engine dropped the binding via the `active_scripts` skip). Falsifier: the value keeps snapping
  back = `active_scripts` is built but the discovery skip never reads it (the flag defined-but-unwired —
  the exact dead-mechanism failure dev_flow step 7 names).
- **F3-freeze** — start a copilot turn; the Activate/Deactivate button + the copy-content selector in the
  script bar are disabled (greyed) until the turn ends. Falsifier: clicking Activate mid-turn flips the
  flag (the `copilot_turn_active` freeze was dropped in the tab-row→bar rewrite).
- **F3 script-local bar** — a thin bar appears under the tab bar ONLY for a script tab, holding the
  Activate/Deactivate button + (per-uniform only) the copy-content selector; it does NOT appear for a
  shader/lib tab. Activate label flips Activate↔Deactivate and toggles the glyph/snap-back. Falsifier:
  switch to a shader tab — the bar vanishes.
- **F4** — activate a brain whose stub drives only real uniforms → NO "engine-owned" error in the strip.
  Hand-add `"u_time": ...` to the brain dict → it is silently dropped, no error AND `u_time`'s blue
  read-only row in the Node tab still ADVANCES (renderer-driven), proving a silent drop, not a freeze or a
  takeover. The Node tab still shows `u_time`/`u_resolution`/`u_aspect` as blue read-only rows (decision 5
  "Node UI is UNCHANGED"). Falsifier: an "engine-owned" error appears in the strip, OR `u_time`'s blue row
  stops moving (the brain captured it), OR the blue rows vanish from the Node tab.
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
- **F14 migration** — `make run` against `projects/dev/` → the brain node's `u_char_height` / `u_color1` /
  `u_color2` / `u_spacing` rows still show the ACTIVE-or-INACTIVE glyph (their renamed `u_*__<tag>.py`
  scripts rebound), NOT grey/absent. Falsifier: a grey (absent) glyph on any of the four = the impl-commit
  rename was skipped and the scripts went undiscovered under the new glob.
- **`make check` + `make smoke` green, AND smoke proves the renamed seed actually TICKED** — the smoke seed
  is a script-driven node; rename its seed file to the new scheme and add an assertion that
  `app.session.script_engine.script_driven_uniforms(<scripted_node_id>)` contains the seeded uniform after
  the run. Falsifier: an empty driven-set passes smoke green = the new `u_*__<tag>.py` filename didn't match
  the discovery matcher (a builder-vs-matcher tag disagreement) and smoke green-washed it. Without this
  canary, smoke only proves "didn't crash", not "the new filename scheme binds".

## Open questions for the user

Both settled at the pre-impl review (see Review history → Open-question resolutions); change either if you
disagree:
- F3 button label → `Activate script` / `Deactivate script`.
- F14 filename separator → `u_x__vec2.py` (double underscore). This is the on-disk name AND the tab label.
- None blocking.

## Review history

### Pre-impl review (2026-06-14, 2 adversarial agents — correctness/design + verification/blast-radius)

High-blast-radius wave (deletes a mechanism, changes an on-disk filename scheme, touches conventions), so
2 reviewers per dev_flow step 4. Both returned non-PASS (PARTIAL / FAIL) with code-anchored findings; the
two highest-stakes were independently re-verified against the live source before folding. All findings were
REAL (none manufactured — this was round 1). Resolutions, folded into the decisions above:

- **Binding-key collision (CRITICAL, verified at `engine.py:291`).** `reload` does `name = path.stem`, so a
  tagged filename `u_x__vec2.py` would key `behaviors`/`errors`/`_binding_reject` by `u_x__vec2` and
  false-orphan every lookup. → Decision 13 rewritten: the tag lives only in the filename + tab label;
  internal keys stay the bare uniform name; two shared helpers (`_uniform_tag`, `_parse_script_filename`);
  the tag is a match gate, not a key. `script_file_for` + `backend.py:690` named as ripple sites.
- **Retype leaks the active flag (verified — one `UIUniform`, one flag).** → Decision 14 gains the gate:
  add to `active_scripts` only when flag True AND the exact tagged file exists, so a retyped uniform's fresh
  file is born inactive even with a stale True flag.
- **`reload(active_scripts)` ripples to 2 live callers + ~34 test sites (verified by grep).** → Decision 2 +
  Files touched name `project_session.py:308/362`, both test files, and the 6 deleted `.disabled` tests.
- **Engine-owned drop touches the BRAIN tick path, not just the stub (verified at `engine.py:608`).** →
  Decision 5 + Files touched: `_apply_behavior` brain branch drops an engine-owned key silently.
- **`conventions.md` "no `UIUniform` field" clause goes stale (verified at the scripting Design-decision
  bullet).** → added to Files touched (update the active-intent nuance, keep binding-by-filename).
- **Dev-sandbox one-time orphan (verified: the 4 `u_*.py` files don't match `u_*__*.py`).** → new
  "One-time migration" section + the F14-migration manual check.
- **Verification gaps (all folded into Manual verification):** the active-flag had no consumer-side check
  (the dead-mechanism risk) → F2-wire; smoke couldn't detect a dead script → smoke tick-canary; the
  "Frozen mid-copilot-turn" freeze was unverified → F3-freeze; F4 passed for >1 reason and didn't verify
  "Node UI unchanged" → strengthened; born-inactive couldn't be told from never-discovered → F2 split
  (inactive-faded vs absent-grey).
- **No-spec-change (acknowledged, no edit):** deleting the "type-change error" todo entry is correct and
  loses no live info (both halves dead after 047) — already specced to happen in the impl commit; the
  copilot-rollback deferral does not fire (047 adds no copilot-reachable durable mutation) — now stated
  explicitly in Out of scope.

Convergence: round 1 surfaced real systemic gaps in the binding-key + verification layers; all patched in
the spec. The spec is now implementable as written. No "should not land" finding — no escalation to the
user beyond the two settled open questions below.

### Open-question resolutions (settled at this review)

- F3 button label: `Activate script` / `Deactivate script` (the spec's first option; clearer than the
  state-as-label phrasing).
- F14 type-tag separator: `u_x__vec2.py` (double underscore) — kept; it reads cleanly as a tab label and
  the double-underscore can't collide with a single-underscore uniform name segment.

### Post-impl review (2026-06-14, 3 adversarial agents + 1 convergence pass)

High-blast-radius diff → 3 reviewers (code-correctness, architecture/conventions, spec-fidelity) in one
batch. Verdicts: spec-fidelity PASS (all 15 decisions LANDED, with independent live F14-retype + F4
silent-drop proofs); architecture PASS (one minor duplication smell); code-correctness FAIL — one real,
empirically-verified bug.

- **FAIL → fixed: copy-content was silently unsaveable.** `App.copy_into_current_editor` did
  `editor.set_text(body)`, but `TextEditor.set_text` resets the undo index to 0, and a fresh script
  session has `saved_undo == 0`, so `is_current_editor_dirty()` read False → the copied body never
  flushed. Fix: `session.saved_undo = -1` after `set_text` (a sentinel `get_undo_index()` ≥ 0 can never
  match, re-baselined on the next save). Re-verified independently: `set_text` does keep the index at 0.
- **Smell → consolidated: filename assembly was duplicated.** `f"{name}__{tag}.py"` lived in two
  `project_session.py` sites. Lifted to `engine.py::per_uniform_filename(uniform)` (the ONE home beside
  `uniform_tag`), exported, both sites routed through it. (The `script.py` brain literal + the one-off
  `u_*__*.py` glob in `same_type_scripts` left as-is — stable, never-computed strings, no drift risk.)

A focused convergence pass re-verified both fixes against running behavior (the `saved_undo` read/write
audit + the `per_uniform_filename`↔`parse_script_filename` round-trip + 104 green script tests) → PASS,
no regression. `make check` 0 errors; `make smoke` skips on the display-less dev box (its 047 tick-canary
is wired and the equivalent binding proven headlessly). The maintainer `make run` checks below still pend.

### Ultra-review (2026-06-14, mega adversarial swarm — 6 dimensions × 3-skeptic verify, 52 agents)

A maintainer-requested ultracode swarm (6 parallel finders → each finding voted by 3 skeptics with
distinct lenses incl. one that RUNS the code, ≥2/3 to confirm) found **14 real defects the 3 text-only
review rounds missed** — because the swarm executed code (`uv run`) and traced imgui draw-order, which a
text trace can't. Two ROOT causes + clusters; all fixed:

- **ROOT 1 (CRITICAL) — the native tab bar was read-back-only.** `_draw_tab_row` only READ imgui's
  selected tab into `active_tab_index`; it never DROVE imgui's selection, so a programmatic switch
  (glyph open / node-select / lib-jump / re-focus) silently reverted — the editor stayed on the prior
  file. Empirically reproduced on imgui-bundle 1.92.801. Fix: a one-shot `App.tab_select_pending` set at
  every programmatic `active_tab_index` assignment (`_focus_or_add_tab`/`close_tab`/`_remove_tabs_for_node`),
  consumed in `_draw_tab_row` as `TabItemFlags_.set_selected` on the target tab, read-back gated to genuine
  clicks — the exact pattern the working `ui.py` node-settings bar already used. (Decision 8 was a no-op as
  shipped.)
- **ROOT 2 (HIGH) — `set_script_active` silently no-op'd before a row was drawn.** `is_script_active` lives
  on a `UIUniform` created lazily ONLY in the Node-tab draw loop, so `_ui_uniform_for` returned None and the
  flag write was skipped for any programmatic/headless activation. This made the 047 smoke tick-canary a
  guaranteed FALSE-RED on its first GPU run AND broke the canonical `scripts/dogfood/verify_script_engine.py`
  (empirically red). Fix: `_ui_uniform_for` now `setdefault`s `UIUniform.from_uniform(u)` for the matched
  live uniform — activation works without a prior draw; the born-inactive-on-retype guarantee holds (a
  retyped uniform has a different hash → fresh False row). Verified by the now-green dogfood harness end to end.
- **ROOT 3 (medium+low) — a script tab went stale on a shader-side retype.** The script-local bar resolved
  state by the LIVE uniform's current tag, not the tab's open file, so after a retype it mislabeled
  Activate/Deactivate over an absent file. Fix: the bar detects `script_path_for(node) != tab.path` and
  shows "this uniform was retyped or removed — its script is detached" instead of acting on the wrong file.
- **Coverage holes (high+medium+low) — fixed:** `test_non_utf8_file_does_not_crash_reload` passed vacuously
  (untagged name skipped the read) → retagged; added falsifiable tests for retype-drop + retype-back +
  delete-readd, `parse_script_filename` (incl. the underscore-in-name `rfind` edge), `same_type_scripts` /
  `copy_script_body`, and a real-GL `get_uniform_hash` retype-distinguishes test — each confirmed RED under
  the regression it guards.
- **Stragglers (low) — fixed:** the dogfood harness migrated to the tagged scheme + activation; the copilot
  `set_uniform` reject's dead `f"{name}.py"` fallback removed (spec decision 13 straggler); two stale
  marker-era comments rewritten.
- **Refuted (1/15):** a claim that mutating `per_uniform_filename` to a single underscore leaves the suite
  green — all 3 skeptics ran it and `test_uniform_pill_state` failed, falsifying the trigger.

After all fixes: `make check` 0 errors, full suite 483 passed / 27 skipped (+8 new), the dogfood harness
green end to end. ROOT 1's frame behavior + the F1/F3/F9/F11/F14 falsifiers below still need the maintainer
`make run` (imgui selection can't be judged headless). **Durable lessons filed:** the read-back-tab-bar trap
+ the lazy-model-row activation trap → `/imgui-ui` skill (§ native tab bar) + `conventions.md`.
