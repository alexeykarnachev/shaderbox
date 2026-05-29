# 019 — keyboard_navigation

The **focus/navigation layer** — the deferred second half of the keyboard story (feature 018 shipped
the *command* layer: named-verb dispatch). This adds mouse-less **widget interaction**: a two-level
focus model where an explicit chord cycles between three regions (code editor / node grid / settings
panel — NOT sibling windows; they nest, see decision 3), and imgui's built-in `nav_enable_keyboard`
drives Tab/arrow traversal + Space/Enter activation + arrow-driven sliders *within* the focused region.

Splits cleanly from 018: 018 is "fire a named action by chord, anywhere"; 019 is "move focus through
widgets and operate them with the keyboard." 018's registry/dispatch/cheatsheet is the spine this
reuses (the four region/tab commands are ordinary `CommandSpec`s).

---

## Goal

- **`nav_enable_keyboard` ON** — every standard imgui widget becomes keyboard-operable: Tab/Shift+Tab
  to traverse, arrows for directional move + slider stepping, Space/Enter to activate, Esc to cancel
  an active widget. This is real mouse-less operation (drag sliders, toggle checkboxes, pick combos,
  type in fields — all without the mouse).
- **Two-level focus model.** Level 1: three **regions** — (1) code editor, (2) node grid,
  (3) the Node/Render/Share settings panel (they NEST, not siblings — decision 3). Level 2: nav
  operates *within* the focused region.
- **The region-cycle chord cycles the three regions** (wrap-around), driving which region owns nav
  focus. Default `Ctrl+`` ` `` (NOT `Ctrl+Tab` — imgui reserves that for window-cycle; decision 9);
  rebindable.
- **`Ctrl+1` / `Ctrl+2` / `Ctrl+3` jump the inner tab** (Node / Render / Share) when the settings
  panel region is focused.
- **Region-scoped nav, if feasible** — Tab/arrows stay *inside* the focused region rather than
  escaping across region borders (the C1 question — see Design decision 4 for the verified mechanism
  + the fallback).
- **The editor is a focus stop, not a traversal surface** — the region-cycle chord enters/leaves it;
  while it's focused all bare input is the caret; only Ctrl-modified registry commands
  (Save/lib-picker/palette/region-cycle) escape to the app.

## Out of scope (each with a trigger)

- **Removing the old bare-arrow node-prev/next entirely is IN scope** (not deferred) — see Design
  decision 5. (Listed here only to mark that the 018 `NODE_PREV`/`NODE_NEXT` commands change.)
- **2D spatial arrow-nav tuning across the node grid.** imgui's directional (Left/Right/Up/Down)
  nav between hand-wrapped `invisible_button` cells is its weakest case (the cells are per-tile child
  windows laid out with `same_line` + manual wrap). We make the grid *reachable + traversable* (via
  `nav_flattened`, decision 4) but do NOT hand-tune row/column adjacency. **Trigger:** the maintainer
  reports grid arrow-nav skips/misorders cells badly enough to be unusable, AND a layout change
  (uniform cell width, a real `ImGuiListClipper`/columns grid) would fix it.
- **Gamepad nav** (`nav_enable_gamepad`). **Trigger:** a user asks for controller support.
- **Persisting the focused region across restart.** The region resets to a sensible default each
  launch (decision 6). **Trigger:** the maintainer wants the app to reopen on the last-focused region.
- **A visible "which region is active" chrome treatment beyond imgui's own nav highlight.** v1 relies
  on imgui's nav cursor rectangle + the existing per-region borders/dim. **Trigger:** the maintainer
  finds the active region ambiguous in daily use (then add an accent border on the focused region).

## Design decisions (numbered, lock-in)

1. **`nav_enable_keyboard` is set once at `App.__init__`**, right after `imgui.create_context()` +
   `set_ini_filename` (alongside `config_input_text_cursor_blink = False`, `app.py:118`):
   `imgui.get_io().config_flags |= imgui.ConfigFlags_.nav_enable_keyboard`. Spike-confirmed: the flag
   takes, and `io.nav_active` goes True from the first frame a window exists. No per-frame re-set.

2. **Region identity is a new `ActiveRegion` StrEnum + a transient `active_region` field on `App`**
   (a plain instance field, like `editor_focused` at `app.py:170` — NOT on the persisted `UIAppState`,
   no migration; decision 6). Three members: `EDITOR`, `GRID`, `PANEL`.
   `App.active_region: ActiveRegion = ActiveRegion.EDITOR` (reset each launch). The enum lives in the
   **`commands.py`** leaf (it already holds the keyboard-control vocabulary and imports `imgui` only —
   adding two StrEnums keeps it leaf-clean, no cycle). Same home for `NodeTab` (decision 10).

3. **The three "regions" are NOT sibling `begin_child`s — they nest, so the focus latch lives INSIDE
   the nested draw fns, not in `update_and_draw`.** ← **critical correction from review.** The real
   `ui.py` tree (verified):
   ```
   code_editor          (ui.py:227, top-level child)            ← EDITOR
   app_panel            (ui.py:236, top-level child)
     control_panel      (ui.py:405)
       node_preview_grid (node_grid.py:29)                       ← GRID  (3 levels deep)
       node_settings     (ui.py:423, _draw_node_settings)        ← PANEL (3 levels deep)
   ```
   Only EDITOR is a top-level child; GRID and PANEL are grandchildren under a shared
   `app_panel`/`control_panel`. So a single `set_next_window_focus()` in `update_and_draw` would latch
   onto `app_panel` (the next child begun), not the grid. **Mechanism (spike-verified, this session):**
   - `set_window_focus("name")` (by-name string overload) **segfaults** on imgui-bundle 1.92.801 —
     never use it. `set_next_window_focus()` (no-arg) works AND correctly targets a **grandchild**
     when called immediately before that grandchild's own `begin_child` (spike-confirmed: focus landed
     on the `grid` grandchild, and moved to `panel` when the latch flipped). `set_window_focus(None)`
     (defocus) works (`tabs/code.py` already uses it).
   - A one-shot `App.region_focus_pending: bool`, set True by the region-cycle / tab-jump callbacks.
     The call `imgui.set_next_window_focus()` is issued **inside the nested draw fn that owns the
     target region's `begin_child`**: `draw_node_preview_grid` (`node_grid.py`) for GRID,
     `_draw_node_settings` (`ui.py`) for PANEL — each gates on `region_focus_pending and active_region
     == <its region>`, calls the latch right before its `begin_child`, and the flag is cleared after
     the panel draws (one consumer clears it). So `node_grid.py` IS a focus-latch touch site (not just
     the `nav_flatten` passthrough).
   - **The editor region is the exception** (decision 7): it focuses via the editor's existing
     machinery (`editor_focus_requested` → `editor.set_focus()` post-render), not a window focus on the
     `code_editor` child, because the `TextEditor` owns its own inner focusable window.

4. **Region-scoped nav via `no_nav_inputs` on the INACTIVE region grandchildren + `nav_flattened` on
   the grid cells** — the C1 "confinement" answer, with a documented fallback.
   - **The clean model (attempt first):** each frame the regions that are NOT `active_region` get
     `WindowFlags_.no_nav_inputs` on their OWN `begin_child` ("No keyboard/gamepad navigation within
     the window" — spike-confirmed applies through nesting, no crash). Because GRID and PANEL are
     siblings under one `app_panel`, the flag must go on each inner child **individually**
     (`node_preview_grid` in `node_grid.py`, `node_settings` in `ui.py`) — NOT on the shared
     `app_panel` (which would kill both at once). The active region draws WITHOUT the flag, so nav
     lives only there.
   - **The editor pane ALWAYS carries `no_nav_inputs`** (the `code_editor` child, `ui.py:227`) — the
     editor is a focus *stop*, never a nav-traversal surface (decision 7). This also prevents nav's
     cursor from "parking" on the editor child and spuriously reading `editor_focused` True
     (review finding) and removes the error-strip-reachability contradiction (the strip stays
     mouse-only; decision 7). So: EDITOR child = always `no_nav_inputs`; GRID/PANEL children =
     `no_nav_inputs` iff inactive.
   - **Grid cell click target MUST change from `invisible_button` to a nav-focusable item** ←
     **critical correction from review + spike.** Spike result: an `invisible_button` is **NOT a nav
     stop** (nav never lands on it; Space/Enter can't activate it) — with the current
     `preview_cell` (`ui_primitives.py:523`) you literally cannot select a node by keyboard. A
     `selectable` (and a real `button`) ARE nav stops (spike-confirmed, with and without the per-tile
     child wrap). Fix: `preview_cell`'s whole-cell click target switches from `invisible_button` to
     `imgui.selectable` (the semantically-correct widget for "pick this tile" anyway). The cells get
     `ChildFlags_.nav_flattened` ("[BETA] share focus scope, allow nav to cross sibling child
     borders" — spike-confirmed applies) so the grid traverses as ONE ring. `preview_cell` gains
     `nav_flatten: bool = False`; only the node grid passes True. **Blast radius:** `preview_cell` is
     shared by the node grid, the node-creator template grid, AND the sticker carousel — switching the
     click target to `selectable` must preserve their existing mouse-click + overlay-✕ behavior
     (`set_next_item_allow_overlap` + the delete overlay). Verify all three render + click correctly
     (impl-time + manual). The template-grid `nav_flatten` value is tied to decision 8's node-creator
     resolution.
   - **The fallback (if `no_nav_inputs` confinement proves leaky in the maintainer's hands):** drop the
     per-region `no_nav_inputs` on GRID/PANEL; nav becomes one flat app-wide Tab chain, and the
     region-cycle chord / `set_next_window_focus` merely *seeds* focus into a region as a fast jump (Tab from there walks
     into the next region). **This SHIPS A WEAKER MODEL than the user asked for** — "actions local to
     the selected region" collapses to "fast-jump + flat chain," Tab crosses region borders. The
     editor's always-on `no_nav_inputs` and the `selectable` cell-target change stay regardless (they
     aren't part of the confinement gamble). Per `dev_flow.md` mid-flight rule 4: if the clean model
     fails the manual wave, this fallback is **surfaced to the user before shipping** (not auto-adopted)
     and recorded in Review history. The user accepted "fallback acceptable" in chat (C1 = "idk, let's
     see"), but sees which behavior actually shipped.

5. **The 018 bare-arrow `NODE_PREV` / `NODE_NEXT` commands are REMOVED** — subsumed by nav-within-grid.
   - With nav ON, bare arrows belong to imgui (slider stepping, widget traversal). A global
     bare-arrow command would contend with nav on the same key and lose unpredictably (nav consumes
     arrows at a lower level than `imgui.shortcut()`). Node-switching is no longer global: it's "focus
     the grid region (region-cycle chord), arrow to a node, Space to select."
   - **Removed:** the two `CommandSpec`s (`NODE_PREV`, `NODE_NEXT`), their `CommandId` enum members
     (`commands.py:27-28`), their callbacks in `app.py` (`_build_command_callbacks`, app.py:247-248),
     AND the `editor_blocks` flag on `CommandSpec` (it existed ONLY for these two; with them gone it's
     dead — drop the field + the `_dispatch_registry` branch at `hotkeys.py:39` that reads it).
   - **`select_next_current_node` becomes dead code and is DELETED** (app.py:946-959) ← corrected from
     the draft, which wrongly claimed `delete_node` uses it. It does NOT: `delete_node` calls
     `select_next_value` directly (app.py:923). After removing the two callbacks, `select_next_current_node`
     has zero callers. Delete it; update the stale referencing comment at `code.py:220`. (Node-switching
     is now grid-focus + nav + Space; `select_next_value` — the generic helper — stays.)
   - `_handle_node_creator_nav` is resolved in decision 8 (not deferred to "verify").

6. **`active_region` is transient, reset to `ActiveRegion.EDITOR` each launch.** Not on `UIAppState`,
   no migration, no persistence (Out of scope). Mirrors `editor_focused` / `editor_defocus_requested`
   (transient focus state). The editor-default matches today's launch behavior (the editor auto-grabs
   focus on first render; `editor_defocus_requested=True` at init drops it — that interplay is
   preserved, see decision 7).

7. **Editor region = focus stop, not a nav-traversal surface; reuses the existing editor-focus
   machine.** (C2.)
   - The `imgui_color_text_edit.TextEditor` owns ALL keys when focused (Tab=indent, arrows=caret,
     Enter=newline). So "editor region focused" means "caret is live"; there is no widget traversal
     there. **The whole `code_editor` pane carries `no_nav_inputs`** (decision 4) so imgui's nav
     cursor never parks on it — this is the design mitigation (not just a hope) for the
     "nav-cursor-sits-on-the-editor-child spuriously reads `editor_focused` True" failure the review
     flagged. The error-strip `selectable` (`tabs/code.py:70`) is inside its own `##shader_errors`
     `begin_child` (`code.py:59`) within the no-nav pane → it is reachable by MOUSE only, NOT nav.
     That's the deliberate resolution of the "strip nav-reachable vs editor caret-only" contradiction:
     editor pane = no nav, period; jump-to-next-error stays on the `F8` command + mouse-click.
   - **Region-cycle chord INTO the editor:** set `active_region = EDITOR` + reuse `editor_focus_requested =
     True` (consumed by `tabs/code.py` post-render via `editor.set_focus()`), NOT a `begin_child`
     window focus. **Region-cycle chord OUT of the editor:** set `active_region` to the next region + reuse
     `editor_defocus_requested = True` (the existing `set_window_focus(None)` post-render path) so the
     caret goes inactive, then the region-focus-pending latch focuses the target region.
   - **Only Ctrl-modified registry commands fire while the editor is focused** — this already holds in
     018 (the Ctrl-chords aren't `editor_blocks`-gated; only the now-removed arrows were). The
     region-cycle chord, `Ctrl+1/2/3`, `Ctrl+S`, `Ctrl+P`, `Ctrl+O`, `Ctrl+N`, `Ctrl+Shift+P` all dispatch through
     `_dispatch_registry` regardless of editor focus. Bare keys go to the caret. The `no_nav_inputs`
     on the pane (above) is what guarantees nav doesn't compete for those bare keys — so the caret
     wins cleanly (still a manual-verify item, but now backed by the flag, not hope).

8. **Modal-internal nav is left to imgui's defaults; the node-creator's bespoke arrow block is
   REMOVED in favor of nav.** ← corrected from review (the draft kept it + deferred a double-fire to
   "verify"). The four modal popups (node creator, settings, emoji picker, lib picker) are NOT regions —
   when one is open it traps focus (`begin_popup_modal`). Inside a modal, `nav_enable_keyboard` gives
   Tab/arrow traversal for free across that modal's widgets (the audit lists them). We add NO region
   logic inside modals.
   - **The node-creator double-fire is real, not hypothetical:** `_handle_node_creator_nav`
     (`hotkeys.py:67-78`) reads bare Left/Right/Enter with `repeat=True`; with nav ON the SAME bare
     arrows drive imgui's nav cursor across the template grid — two consumers, one key, and they
     desync (the bespoke block steps by insertion order via `select_next_value`; nav moves spatially).
     **Resolution: delete `_handle_node_creator_nav`** and let nav own the template grid (the template
     cells already route through `preview_cell` → now a nav-focusable `selectable`, decision 4, so
     arrows traverse + Space/Enter selects + the Create button is a nav stop). The template grid gets
     `nav_flatten=True` (decision 4) for a smooth ring. This is consistent with "modals inherit nav for
     free" and avoids the desync. (`app.select_next_template` stays on `App` — still callable, just no
     longer wired to bare arrows.)
   - The lib-picker tree, emoji grid, settings sliders all just inherit nav.

9. **Four new `CommandSpec`s in `COMMAND_SPECS`** (feature 018's table), GLOBAL scope:
   - `CYCLE_REGION` — "Cycle region", default `Ctrl+grave_accent` (the `` ` `` key, `_chord(K.grave_accent,
     K.mod_ctrl)`), callback `app.cycle_region()`, `repeat=False`. **Why not `Ctrl+Tab`:** the
     imgui-bundle stub documents that "basic Tabbing and Ctrl+Tab are enabled regardless of [the nav]
     flag" — imgui's built-in window-cycle owns `Ctrl+Tab` whenever nav is on, so binding our command
     there is the likely-losing case. `Ctrl+`` ` `` (the tmux/terminal pane-switch idiom) avoids it and
     respects the no-PageUp/Home/End constraint. Rebindable (the user tunes at test time); if the
     maintainer prefers `Ctrl+Tab` and it wins the contest in practice, rebind to it.
   - `FOCUS_TAB_NODE` / `FOCUS_TAB_RENDER` / `FOCUS_TAB_SHARE` — direct-jump to the inner tab, defaults
     `Ctrl+1` / `Ctrl+2` / `Ctrl+3` (`_chord(K._1, K.mod_ctrl)` etc. — `K._1` exists; `commands.py:162`
     already uses `getattr(K, f"_{d}")`). Direct-jump sidesteps cycle-direction ambiguity. Each sets
     `active_region = PANEL` + `active_node_tab = <that tab>` + `region_focus_pending = True` +
     `node_tab_select_pending = True`. `in_palette=True`, `rebindable=True`.
   - All four are ordinary registry commands: cheatsheet-listed, palette-listed, rebindable,
     menu-hint-able — zero new dispatch machinery.

10. **`active_node_tab` field + `TabItemFlags_.set_selected` driving** (the deferral's named piece).
    - New `App.active_node_tab: NodeTab` (a StrEnum `NODE` / `RENDER` / `SHARE`, in `commands.py` with
      `ActiveRegion`). Transient, reset to `NODE` each launch (matches today's default-open Node tab).
    - `ui.py::_draw_node_settings` currently uses imgui-implicit tab selection. Change: each
      `begin_tab_item(name, flags=...)` passes `TabItemFlags_.set_selected` for ONE frame when a tab
      jump was requested (a one-shot `App.node_tab_select_pending: bool`, set by the `FOCUS_TAB_*`
      callbacks, cleared after the tab-bar draws). **Spike-confirmed:** `set_selected` drives the
      active tab headlessly + persists after the one-shot frame (imgui remembers). The ctx wrapper
      `imgui_ctx.begin_tab_item(label, flags)` forwards flags (verified).
    - **Read-back is trivial (not the uncertain thing the draft implied):** exactly one tab's
      `begin_tab_item` returns True per frame (the selected one) — `_draw_node_settings` already uses
      `with imgui_ctx.begin_tab_item("Node") as tab: if tab:` (ui.py:427-437). Record which of the
      three `if tab:` branches fires into `active_node_tab`. No second return value, no
      visible-vs-selected ambiguity (in a tab bar only the selected tab's body draws).

11. **No changes to the cheatsheet / palette / rebinder rendering** — they iterate `COMMAND_SPECS` +
    read `effective_bindings` (018). The four new commands appear automatically. The cheatsheet's
    scope filter (GLOBAL shown when no popup) shows them in the default context. `CommandScope.EDITOR`
    remains an unused-but-documented extension point (unchanged by this feature). No new `theme.py`
    tokens expected (relies on imgui's built-in nav cursor color, `Col_.nav_cursor`, already in the
    palette — leave default or map to `ACCENT_PRIMARY` only if the maintainer finds it invisible).

## Files touched

- **`shaderbox/commands.py`** — add `ActiveRegion` + `NodeTab` StrEnums (decision 2); add 4
  `CommandSpec`s (`CYCLE_REGION` on `Ctrl+`` ` ``, `FOCUS_TAB_NODE`/`_RENDER`/`_SHARE` on `Ctrl+1/2/3`);
  **remove** `NODE_PREV` / `NODE_NEXT` specs + their `CommandId` members + the `editor_blocks` field on
  `CommandSpec`.
- **`shaderbox/app.py`** — `nav_enable_keyboard` in `__init__` (decision 1, right after
  `set_ini_filename`/`apply_theme`, app.py:115-119); transient fields `active_region`, `active_node_tab`,
  `region_focus_pending`, `node_tab_select_pending`; `cycle_region()` + `focus_node_tab(tab)` verbs (or
  3 thin tab callbacks); wire the 4 new callbacks in `_build_command_callbacks`; **remove** the
  `NODE_PREV`/`NODE_NEXT` callbacks (app.py:247-248) AND **delete `select_next_current_node`**
  (app.py:946-959, now dead — decision 5). Reuse `editor_focus_requested`/`editor_defocus_requested`
  for the editor region (decision 7).
- **`shaderbox/hotkeys.py`** — `_dispatch_registry`: drop the `editor_blocks` branch (hotkeys.py:39,
  field removed). **Delete `_handle_node_creator_nav`** (hotkeys.py:67-78) — nav owns the template grid
  now (decision 8); remove its call from `dispatch_commands`.
- **`shaderbox/ui.py`** —
  - `update_and_draw`: editor region — the `code_editor` `begin_child` (ui.py:227) gets a permanent
    `window_flags=no_nav_inputs` (decision 4/7).
  - `_draw_node_settings` (ui.py:421): if `region_focus_pending and active_region == PANEL`, call
    `set_next_window_focus()` before the `node_settings` `begin_child`; add `no_nav_inputs` to that
    `begin_child` when PANEL is inactive; pass `TabItemFlags_.set_selected` on the matching
    `begin_tab_item` for the one-shot `node_tab_select_pending` frame; record which `if tab:` fires
    into `active_node_tab` (decision 10). Clear `region_focus_pending` after this draw (last consumer).
- **`shaderbox/widgets/node_grid.py`** — `draw_node_preview_grid`: if `region_focus_pending and
  active_region == GRID`, call `set_next_window_focus()` before the `node_preview_grid` `begin_child`
  (decision 3 — the latch lives HERE, the grid's child is in this file, not `ui.py`); add
  `no_nav_inputs` to that `begin_child` when GRID is inactive (decision 4); pass `nav_flatten=True`
  through `draw_node_preview_button` → `preview_cell` (node grid only).
- **`shaderbox/ui_primitives.py`** — `preview_cell` (ui_primitives.py:463): **switch the whole-cell
  click target from `invisible_button` (line 523) to `imgui.selectable`** (spike: `invisible_button`
  is not a nav stop, `selectable` is — decision 4); add `nav_flatten: bool = False` param → OR
  `ChildFlags_.nav_flattened` onto the per-tile `begin_child` (line 495) when True. Preserve the
  existing `set_next_item_allow_overlap` + delete-✕ overlay + selection-border behavior (the
  `selectable` must still allow the overlay button to win the click — verify the overlap call still
  applies to it).
- **`scripts/smoke.py`** — assert `imgui.get_io().config_flags & ConfigFlags_.nav_enable_keyboard`
  **immediately after `App(project_dir=...)` construction** (the flag is set in `__init__` before any
  frame — checking it there avoids the get_io()-outside-frame footgun; NOT inside `_check_invariants`,
  which runs mid-loop). Exercise `app.cycle_region()` + `app.focus_node_tab(NodeTab.RENDER)` once mid-loop
  (a callback throw surfaces via `main`'s except). Assert `active_region` / `active_node_tab` stay valid
  enum members. (Nav *behavior* is un-headless-able — smoke guards wiring + no-assert only, per
  `dev_flow.md ### Run the app`.)
- **`pyproject.toml`** — none (no new dep).
- **`ai_docs/todo.md`** — DELETE the `[DEFERRAL] keyboard focus/navigation layer` entry in the impl
  commit (this feature resolves it). The grid-arrow-nav-tuning + region-chrome + persist-region items
  become this spec's Out-of-scope (already triggered there). Also update the stale `code.py:220` comment
  that references the deleted `select_next_current_node`.

## Manual verification (only the maintainer's hands — `make run`; nav is un-headless-able)

Nav behavior CANNOT be screenshotted or asserted headlessly on the dev box (no WM; `imgui_color_text_edit`
+ nav are interaction-only). Every item below is a hands-on check the maintainer runs. This is the
load-bearing wave — the C1 clean-vs-fallback (decision 4) is DECIDED here.

1. **Region cycle.** `Ctrl+`` ` `` moves focus editor → grid → panel → editor (wrap). The focused
   region shows imgui's nav cursor / is the one responding to Tab+arrows. Each region is reachable.
   (If `Ctrl+`` ` `` is awkward on the maintainer's layout, rebind in Settings — chord is tuned here.)
2. **Confinement (the C1 decision).** With the grid focused, Tab/arrows stay among the grid's nodes +
   its "New node"/"Render all" controls — they do NOT escape into the panel's sliders. Same for panel
   (Tab stays among Node/Render/Share widgets) + editor (caret only — its pane is always `no_nav_inputs`).
   **If Tab escapes the region despite `no_nav_inputs`** → the clean model failed; switch to the
   fallback (decision 4) AND surface it to the user before shipping (dev_flow mid-flight rule 4) + record
   in Review history. (Verify in all three regions.)
3. **MAKE-OR-BREAK: select a node by keyboard.** Grid focused → arrows move the nav cursor onto a node
   thumbnail (the cell is now a `selectable`, not an `invisible_button` — spike proved the latter is
   unreachable) → Space/Enter sets `current_node_id` to that node (preview updates). **If the nav
   cursor never lands on a thumbnail, the user cannot select nodes by keyboard at all** — the core
   promise fails; the cell target / flatten needs rework. Exact row/column adjacency across the wrapped
   grid is NOT guaranteed (Out of scope); reachability + activation IS the bar.
4. **Inner tab jump.** With the panel focused (or from anywhere — they're GLOBAL), `Ctrl+1`/`Ctrl+2`/
   `Ctrl+3` switch the Node/Render/Share tab on the SAME press (not one frame late); clicking a tab
   also updates which one the `Ctrl+`-jump tracks (`active_node_tab` reads back from the live tab — no
   desync).
5. **Sliders by keyboard.** Focus a uniform drag (Node tab) or a Render-tab slider via Tab; arrows
   step its value; the shader/preview updates. A `drag_float` vec2/3/4 steps the focused component.
   (Spot-check 3 of the 13 drag controls — a uniform drag, render fps, Settings font-size.)
6. **Combos by keyboard** (distinct nav surface — popup-on-activate, a known imgui rough edge). Tab to
   the resolution combo (Node tab) and the quality combo (Render tab); Space/Enter opens it, arrows
   move the highlight, Enter commits, the value changes. (If combos can't be operated, "real mouseless"
   has a hole — flag it.)
7. **Text fields + Tab-to-exit + the `set_keyboard_focus_here` interaction.** Tab INTO an
   always-available text field (node-name inline, lib-picker search — NOT the share-tab fields, which
   need creds to render), type, Tab OUT to the next widget (Tab exits rather than inserting). **The
   one-shot auto-focus grabs must stay one-shot under nav:** open the lib picker (search auto-focuses
   via `set_keyboard_focus_here`, search.py:50) → press Tab → focus moves to the tree, search does NOT
   re-grab it; start an inline rename in the lib tree (`needs_focus` grab, tree.py:304) → the grab
   fires once, Tab then leaves cleanly. (These are the surfaces the old deferral named; nav must not
   fight the grabs.)
8. **Editor focus boundary (C2).** `Ctrl+`` ` `` into the editor → caret is live, bare keys type,
   arrows move the caret, Tab indents (NOT nav-traverse — the pane is `no_nav_inputs`). Ctrl-commands
   still fire from inside (Ctrl+S saves, Ctrl+P lib picker, Ctrl+Shift+P palette, Ctrl+`` ` `` leaves).
   `Ctrl+`` ` `` OUT → caret goes inactive (pane dims per `EDITOR_UNFOCUSED_ALPHA`) and the next region
   takes focus. The error strip below the editor is NOT nav-reachable (mouse-click only) — confirm F8
   still jumps errors.
9. **Modals inherit nav.** Open each modal (node creator, settings, emoji picker, lib picker): Tab/
   arrows traverse its widgets, Space/Enter activates, Esc closes (018's bespoke Esc still works).
   - **Node creator:** the bespoke arrow block is GONE — nav now drives the template grid (cells are
     `selectable` + `nav_flatten`). Arrows move the nav cursor between templates, Space/Enter on a
     template + the Create button both work, NO double-step (the old desync is gone with the block).
   - **Settings rebinder vs Tab capture:** while a "Rebind" capture is armed, pressing Tab is CAPTURED
     as a chord key (Tab is in `_BINDABLE_KEYS`), NOT consumed by nav to move focus out of the row.
     (If nav steals it, the rebinder body needs `no_nav_inputs` during capture — mirror the existing
     Esc special-case at hotkeys.py:51.)
   - **Lib-picker tree:** arrows move rows, Left/Right collapse/expand dirs, Enter on a function leaf
     inserts/selects.
10. **All three `preview_cell` users still mouse-work** (the `invisible_button`→`selectable` swap is
    shared): node grid, node-creator template grid, AND the sticker carousel (share tab, if connected)
    — each still selects on mouse-click, the delete-✕ overlay still wins its own click, the selection
    border still shows. (Regression check for the shared-primitive change.)
11. **Cheatsheet + palette + rebinder.** The 4 new commands appear in the cheatsheet (default context),
    the palette (searchable + fire), and the Settings rebinder (rebindable); rebinding `CYCLE_REGION`
    to another chord takes effect live + persists across restart. Removing the old node-arrow rows from
    the cheatsheet is correct (they're gone).
12. **No regression in the 018 command layer** — every 018 chord still fires (save/new/delete/settings/
    quit/lib-picker/jump-error/palette/toggle-cheatsheet); popup suppression still holds.
13. `make smoke` green; `make check` 0 errors.

## Decisions taken as defaults (user steer: "do reasonable defaults, we'll tune later")

The user explicitly deferred the nuanced/tune-at-test-time choices. Locked defaults (all rebindable /
reversible at test time):

- **N1 — enum home:** `ActiveRegion` + `NodeTab` in `commands.py` (keyboard-control vocabulary, leaf-clean).
- **N2 — region-cycle chord:** `Ctrl+`` ` `` (NOT `Ctrl+Tab` — imgui owns Ctrl+Tab for window-cycle
  whenever nav is on, per the stub note). Rebindable; tune at test time.
- **N3 — inner-tab chords:** `Ctrl+1/2/3` direct-jump (not a cycle chord). Tune at test time.
- **N4 — region scoping:** attempt the clean `no_nav_inputs` confinement; if the manual wave shows
  leakage, the flat-chain fallback ships ONLY after being surfaced to the user (decision 4 / dev_flow
  mid-flight rule 4) — the fallback is a weaker model than asked, so it's not auto-adopted silently.
- **N5 — node-prev/next removal:** removed (user confirmed in chat — node-switching is now grid-focus
  + nav + Space). The dead `select_next_current_node` is deleted with it.

## Review history

**Pre-implementation review (round 1) — 2 adversarial reviewers (correctness/design + blast-radius/
intent-fidelity), both PARTIAL. Zero false positives — every finding verified real against the code.**
Findings applied:
- **CRITICAL — region nesting model was wrong.** The draft treated editor/grid/panel as sibling
  `begin_child`s; they nest (`app_panel`→`control_panel`→{grid,panel}), so a single
  `set_next_window_focus()` in `update_and_draw` would latch onto `app_panel`. Fixed: decision 3
  rewritten — the focus latch lives INSIDE `draw_node_preview_grid` / `_draw_node_settings` (each owns
  its grandchild `begin_child`); `node_grid.py` added to Files-touched for the latch + `no_nav_inputs`.
  **Spike-verified this session:** `set_next_window_focus()` DOES correctly target a grandchild.
- **CRITICAL — node-by-keyboard selection was impossible as designed.** Spike: an `invisible_button`
  (the current `preview_cell` click target) is NOT a nav stop; `selectable`/`button` ARE. Fixed:
  decision 4 switches `preview_cell`'s click target to `selectable`; added the make-or-break manual
  item (verification 3) + the shared-primitive regression item (10).
- **CRITICAL — node-creator double-fire** (bare arrows drive both `_handle_node_creator_nav` AND nav,
  desyncing). Fixed: decision 8 now DELETES `_handle_node_creator_nav` and lets nav own the template
  grid — no deferred "verify and hope."
- **CRITICAL — `set_keyboard_focus_here` vs nav** (the old deferral named these surfaces; the draft
  dropped them). Fixed: verification item 7 now checks the one-shot auto-focus grabs stay one-shot
  under nav (lib search, inline rename).
- **HIGH — `Ctrl+Tab` collision is the expected case** (stub: "Ctrl+Tab enabled regardless of the nav
  flag"). Fixed: default chord changed to `Ctrl+`` ` `` (decision 9 / N2).
- **HIGH — fallback ships a weaker model; was buried as a unilateral mid-flight call.** Fixed:
  decision 4 + verification 2 now require surfacing the fallback to the user before shipping.
- **HIGH — rebinder Tab-capture vs nav.** Added verification item 9 (Settings rebinder sub-bullet).
- **MEDIUM — `select_next_current_node` claimed a caller it doesn't have** (`delete_node` uses
  `select_next_value`). Fixed: decision 5 now DELETES it as dead code + the stale `code.py:220` comment.
- **MEDIUM — editor focus read-back could misfire under nav + error-strip reachability contradiction.**
  Fixed: decisions 4/7 give the `code_editor` pane a permanent `no_nav_inputs` (editor = focus stop,
  never a nav surface) — design mitigation, not a hope; resolves the strip contradiction (mouse-only).
- **MEDIUM — combo keyboard-open unverified.** Added verification item 6.
- **MEDIUM — share-tab verification un-runnable without creds.** Fixed: verification 7 leans the
  text-field check on the always-available node-name / lib-search inputs instead of share fields.
- **MEDIUM — smoke `nav_enable_keyboard` assert placement.** Fixed: assert right after `App()`
  construction (not in `_check_invariants`, which runs mid-loop / get_io-outside-frame footgun).
- **LOW — `commands.py` removal list omitted the `CommandId` enum members + decision 2 phrasing
  garbled "`UIAppState`-free".** Both fixed.
- **LOW — decision 10 read-back framed as uncertain when it's trivial** (one tab returns True/frame).
  Clarified.

Two CRITICALs (grandchild focus, invisible_button nav-stop) were resolved by NEW spikes this round, not
just prose — the spec now starts from confirmed mechanics on all four load-bearing APIs
(set_next_window_focus-on-grandchild, no_nav_inputs-through-nesting, selectable-is-a-nav-stop,
set_selected-drives-tabs).

**Pre-implementation review (round 2) — same 2 reviewer roles, re-spawned against the patched spec:
one PASS, one PARTIAL.** Both confirmed 12 of 13 round-1 findings landed with accurate code anchors
(every file:line re-checked matched). The lone PARTIAL: the region-cycle chord rename to `Ctrl+`` ` ``
hadn't propagated from the decisions into the Goal headline + four body-prose mentions (a real
internal contradiction, flagged independently by both reviewers). Fixed in the same wave — the
body now refers to "the region-cycle chord" generically (it's rebindable) and reserves `Ctrl+Tab`
mentions for the why-not-this-chord rationale only. Two clearing-order / focus-mechanism worries the
round-2 reviewers raised were traced and found NON-issues (the `region_focus_pending` clear is
unconditional in the last-drawn panel fn, and `no_nav_inputs` doesn't block the editor's direct
`set_focus()`) — recorded as verified-sound, no change. **Converged — ready for plan-lock.**

**Post-implementation maintainer verification + polish wave (manual, on `make run`).** The 13-item
manual wave was walked live; nav behaves. Outcomes + the fixes the wave surfaced (all landed):
- **C1 (region confinement) — the CLEAN model shipped.** Per-region `no_nav_inputs` holds; Tab stays
  inside the focused region. The flat-chain fallback was NOT needed.
- **Active-region cue redesigned mid-wave** (the locked outline-on-`begin_child` proved un-workable):
  a `Col_.border` push bled into child cells; `get_item_rect` after `end_child` returned a collapsed
  rect (drew a stub box). Final shape: `ui_primitives.active_region_outline()` reads the child's own
  `get_window_pos/size` from INSIDE the child, strokes on the FOREGROUND draw list (unclipped), and
  the outline + `active_region` follow LIVE `is_window_focused` (gated by `App.focus_move_in_flight()`
  so a chord cycle's optimistic target isn't reverted by lagging focus). The `code_editor` child gained
  `ChildFlags_.borders` (it had none) — which also gave it the content-padding band the outline needs.
- **Color scheme:** active-region/tab = swappable `ACCENT_PRIMARY`; selection (node/template/sticker
  border) + context-menu chrome = fixed `COLOR.SELECT` (`purple_b`, the only accent-free hue). A
  theme-portability invariant now asserts this at import (`theme.py`).
- **Ctrl+Tab collision (predicted in pre-review):** confirmed live — imgui's built-in window-cycle
  fired. Fixed with `WindowFlags_.no_nav_focus` on the main window (skips it from Ctrl+Tab).
- **Esc:** imgui nav's Esc climbs the window hierarchy (no clean off-switch — upstream #8059). Swallowed
  at the glfw key-callback layer (`App._install_escape_filter`) unless Esc has a job (popup/editor).
- **Uniforms sub-window:** `ChildFlags_.nav_flattened` on `tabs/node.py`'s `ui_uniforms` child so nav
  reaches the sliders directly (no Enter-to-descend / Esc-to-escape boundary).
- **`select_node` keeps grid focus** after Enter (defocus the auto-grabbing editor + re-latch grid).
- **Launch lands on the node grid** (`active_region` defaults GRID + focus latch on frame 1).
- Two cosmetic tails deferred (`todo.md`): nav-cursor resets to cell 0 after Enter; 2D grid arrow
  adjacency. **019 verified + done.**
