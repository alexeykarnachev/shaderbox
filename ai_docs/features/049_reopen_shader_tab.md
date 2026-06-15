# 049 — Node entry-point header (shader / script controls, unified)

> **Status: DONE (2026-06-15, maintainer-verified live).** Implemented to Option C below (design locked
> after a brainstorm + voice-note review). Supersedes the earlier DRAFT, which framed this narrowly as
> "reopen the closed shader tab" — that is one *symptom* of the real gap (see ## Goal). A post-impl
> visual pass fixed two layout bugs (entry-row label/button misalignment from a font-size mix; a
> vec-array uniform value overflowing its row and hiding the trailing `stop` — now `clipped_caption`
> ellipsizes to the control column). Wording defaults ("Entry points" / "open") are live.

Filed from a maintainer observation (2026-06-15) while reasoning about the post-048 tabbed editor,
then re-scoped after a design brainstorm.

## Goal

A node has two editable entry-points — its fragment **shader** (`shader.frag.glsl`, GPU) and its
optional **script** (`scripts/script.py`, CPU brain). Today the UI represents that duality
**inconsistently across two panels**, and three problems all trace to that one root:

1. **No way to re-open a closed shader tab** (the original report). `App.ensure_shader_tab` is called
   from exactly one place — `_on_current_node_changed` — so once the shader tab is closed, reselecting
   the *same* node is a no-op (`current_node_id` doesn't change) and the shader is unreachable except
   via a side trip through another node.
2. **The script has a re-open affordance the shader lacks** (the `</>` header glyph →
   `open_script_for`), so the two entry-points are asymmetric by construction.
3. **The node-level `stop` button has no visible owner.** It sits in the node top-bar beside
   name/resolution (a node-*identity* zone), but it is an *execution* control for the script's loop —
   nothing on screen says "this stops the script."

The fix is **not** "add a reopen button." It is: give the node panel a small **Entry points** zone
where each entry-point has a symmetric `open` action that summons its tab into the editor, and move
the node-level play/stop onto the **Script** row where it finally has an owner. This collapses all
three problems into one coherent surface and files the fix at the right altitude (the shared root, not
the one reported instance).

The editor tab bar itself is **sound and out of scope** — it stays exactly as-is (it carries the
fragile §8 read-back-only quirk and the FPE-modal guards; we don't touch it). The node-panel controls
are *summoners* ("bring this entry-point into the editor"); the tab bar is the editor's own document
state. That distinction is honest, not redundant.

## Design decisions (locked)

1. **Option C — node-panel Entry-points zone; tab bar untouched.** Rejected: Option A (tabs own all
   nav, drop the top-bar glyph) and Option B (a GPU/CPU toggle inside the editor that swaps editor
   content). B was rejected because it introduces a *second* mental model of "what's in the editor"
   that fights the existing tab bar; A was rejected because "+ shader for the current node" is an odd
   citizen on a tab bar shared with lib + other-nodes' tabs. C is lowest-risk (editor tab machinery
   untouched) and matches the maintainer's stated mental model.

2. **Lightweight rows, not bordered boxes.** The zone is a single `small_caption("Entry points")`
   followed by two compact rows (Shader, Script) — NOT two bordered cards. Rationale: `/imgui-ui` §2
   (whitespace separates zones, rules/boxes read as noise; don't stack chrome). The name/resolution
   row sits directly above; two boxes there would be heavyweight competing chrome.

3. **`open` is a `ghost_button`, not a glyph.** Per the hard rule (no hand-rolled glyph at a call
   site) and `/imgui-ui` §2 (real words, not cryptic glyphs), each row's open action is
   `ghost_button("open")`. The old `</>` `script_glyph` is **folded into** the Script row's `open` —
   one home, not two. `script_glyph` becomes dead and is **deleted** from `ui_primitives.py` (no other
   caller — verified at impl time; if a caller appears, that's a design surprise to surface, not work
   around).

4. **Shader row.** `Shader   [ open ]`. `open` → `app.ensure_shader_tab(app.current_node_id)`
   (focus-or-open, idempotent — already does the right thing; this is the original 049 fix, now just a
   wired trigger). Always present (every node has a shader). No error state on this row (a shader's
   compile error already surfaces in the editor strip; nothing new here).

5. **Script row.** `Script   [ open ]   [ play | stop ]`.
   - `open` → `app.open_script_for(app.current_node_id)` (lazy-creates the file if absent — the old
     `</>` behavior, unchanged).
   - When the script has an error, the row's `open` tints `COLOR.STATE_ERROR` and its tooltip says
     "Node script error — click to open and fix" (preserves the old error-glyph behavior; the tint
     moves from the deleted glyph onto the `open` button's text via a local color push).
   - The node-level `play_stop_toggle("node", ...)` renders on this row **only when a script exists**
     (`app.session.has_script(node_id)` — the existing `if present:` gate, relocated). Click →
     `app.set_node_all_stopped(node_id, playing)` (unchanged). This is the ONLY relocation — the
     per-uniform play/stop toggles stay on their uniform rows (correct where they are).

6. **Active-entry accent tick.** The row whose tab is currently active in the editor gets a thin
   accent vertical tick at its left edge, so the node panel and editor stay legibly in sync ("which
   one am I looking at?"). Read from `app.active_tab`: tick Shader when `active_tab.kind == "shader"
   and active_tab.node_id == current_node_id`; tick Script likewise for `"script"`. Drawn **inset**
   (inside the row rect, draw-list `add_line`), never straddling an edge — `/imgui-ui` §3 (a selection
   indicator changes color/presence, never size; inset to avoid jitter). No tick when the active tab
   is a lib / another node / nothing.

7. **Disabled-during-turn unchanged.** The whole zone stays wrapped in
   `imgui.begin_disabled(app.copilot_turn_active)` exactly as `_draw_script_controls` is today (a write
   races the mid-turn reload).

8. **"brain" naming kept as-is for now.** The maintainer dislikes "brain" but is deferring the rename
   until the concrete naming is decided, so it can be done in ONE sweep. So this feature keeps every
   existing "brain"/"script" identifier untouched (`open_script_for`, `has_script`, the tooltips' word
   choice) — it only MOVES and RE-LAYS-OUT controls, it does not rename. (See ## Out of scope.)

## Out of scope

- **The "brain" → final-name rename.** Deferred deliberately (decision 8) to a single later sweep once
  the concrete name is chosen. **Trigger:** the maintainer picks the replacement word for "brain" —
  then rename every identifier/tooltip/glyph-id in one wave (`grep -rn brain shaderbox/`). Until then,
  every "brain" string stays.
- **An editor-side GPU/CPU toggle (Option B)** and **tab-bar `+` re-add affordances (Option A).** Both
  considered and rejected (decision 1). **Trigger to revisit:** the node-panel summoner model proves
  confusing in daily use (the maintainer reports "I keep looking in the editor for the switch, not the
  node panel"); only then reconsider moving nav into the editor column.
- **Per-uniform play/stop relocation.** They stay on the uniform rows (decision 5) — not in scope.

## Files touched

- `shaderbox/tabs/node.py` — `_draw_script_controls` rewritten into the Entry-points zone (rename the
  fn to `_draw_entry_points`; both rows + the relocated toggle + the accent tick live here). The
  `imgui.same_line(); _draw_script_controls(app)` call site in `draw()` becomes a block placed AFTER
  the name/resolution row and `[ … ]` actions, BEFORE `_section_break()`.
- `shaderbox/ui_primitives.py` — DELETE `script_glyph` (decision 3, folded into `ghost_button`).
  Possibly add a tiny `entry_point_row(...)` helper IF the two rows share enough draw code to warrant
  it (judge at impl time per the no-duplication rule; a 2-call site may not need extraction).
- No change to `shaderbox/app.py` — `ensure_shader_tab` / `open_script_for` / `set_node_all_stopped`
  all already exist and do the right thing; this feature only adds UI triggers for them. (Confirm at
  impl: no new App method is needed. If one is, that's a scope surprise to flag.)
- No change to `shaderbox/tabs/code.py` (the tab bar) — decision 1.

## Manual verification (maintainer `make run` — UI/nav is un-headless-able per `/imgui-ui` §0)

The agent verifies headlessly (`make check` 0 errors; `make smoke`; a standalone-context driver of
`tabs/node.draw` for no-crash / stack-balance). These are the by-hand checks the maintainer runs,
each tied to a decision:

1. **Reopen shader (the original bug):** open a node, close its shader tab, click Shader `open` →
   the shader tab reappears and focuses, no side trip. (decision 4)
2. **Reopen does not duplicate:** with the shader tab already open, click Shader `open` → it just
   focuses the existing tab (idempotent), no second tab. (decision 4)
3. **Script open + lazy create:** on a node with no script, click Script `open` → the file is created
   and its tab opens. (decision 5)
4. **Stop owner:** the `play | stop` toggle is on the Script row and only appears when a script
   exists; clicking it freezes/resumes all driven uniforms. The name/resolution row no longer carries
   it. (decisions 5, 3)
5. **Script error tint:** a node with a broken script shows the Script row's `open` in error red with
   the fix-it tooltip. (decision 5)
6. **Active tick sync:** switching the editor's active tab between a node's shader and script moves the
   accent tick between the two rows; selecting a lib / another node's tab shows no tick on this node's
   rows. No layout jitter as the tick appears/moves. (decision 6)
7. **Turn-disabled:** the whole zone greys out mid-copilot-turn. (decision 7)

## Relevant code (entry points)

- `App.ensure_shader_tab` / `App.open_script_for` / `App.set_node_all_stopped` / `App.active_tab`
  (`shaderbox/app.py`) — the methods the rows wire to (all pre-existing).
- `_draw_script_controls` → becomes `_draw_entry_points` (`shaderbox/tabs/node.py`, today ~L192).
- `script_glyph` / `play_stop_toggle` / `ghost_button` / `small_caption` (`shaderbox/ui_primitives.py`).
- The tab bar render (`shaderbox/tabs/code.py`) — read-only reference for the §8 quirk; not edited.

## Open questions (wording only — does not block impl; default and move on)

- Zone caption: "Entry points" (chosen default) vs "Editor" vs "Sources". Defaulting to "Entry points".
- Row action label: "open" (chosen default) vs "show" vs "edit". Defaulting to "open".
- These are pure copy; the maintainer can adjust on the `make run` pass without a re-lock.
