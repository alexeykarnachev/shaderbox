# Review — `04_menus_inventory.md` from the verb/state side

Anchor: the VERB and STATE side, enumerated from the code independently of the inventory.
Sources read directly: `shaderbox/commands.py` (all 32 `CommandSpec`s), `shaderbox/app.py`
(`_build_command_callbacks`, `__init__`'s field run, every `open_*`/`close_*`/`toggle_*`/
`cycle_*`/`pick_*`/`choose_*`/`arrange_*`/`reset_*`/`delete_*`/`dissolve_*`/`leave_*`/
`group_*`/`create_*`/`import_*`/`revert_*`/`recover_*`/`insert_*`/`request_*` method),
every `menu_item_simple(`/`imgui.menu_item(` in `shaderbox/`, every `app.session.*(` call
from a widget/popup/tab/exporter, `widgets/graph_state.py`, `shader_lib/file_ops.py`,
`hotkeys.py:_handle_escape`.

## Verdict: **PARTIAL**

Counts:

| | |
|---|---|
| `CommandSpec`s in `COMMAND_SPECS` | 32 |
| …named anywhere in the inventory | 10 |
| …never named (§7 defers to `commands.py` without enumerating) | **22** |
| `menu_item_simple` / `menu_item` call sites | 29 (22 `menu_item_simple` + 7 menu-bar `menu_item`) |
| …all present in §3 / the label table / the menu-bar note | 29 — **complete, zero drift** |
| Combos | 12 — **complete, zero drift** |
| `pfd.*` dialog call sites | 5 — **complete, zero drift** |
| App/GraphViewState/ShaderLibFileManager state fields in the table | 22 rows |
| …state fields the table omits | **8** |
| Findings | 9 (2 factual errors, 7 omissions) |

---

## Findings

### F1 — §8 and the cross-surface table assert a dead control that no longer exists (factual error)

`04_menus_inventory.md:228` calls the pass-tile corner delete-✕ a **"Dead control"**, claiming
`preview_cell` draws the glyph "whenever `selected=True`" while `_draw_pass_tile` only reads
`result.clicked`. That is no longer true:

- `shaderbox/widgets/pass_list.py:151` passes `deletable=False`.
- `shaderbox/ui_primitives.py:1338` gates the whole glyph block on `if deletable:`, so nothing
  is drawn for a pass tile at all — there is no glyph to click, dead or otherwise.

The fix landed in `97841ec` ("093: the output tile drew a dead delete glyph after wave 3"),
which is the current `HEAD` and postdates the (still-untracked) inventory. Three places now
carry the stale claim:

- `:228` the §8 row (the whole "Dead control" note).
- `:229` "Fully wired (contrast with the pass-tile dead control above)".
- `:280` the cross-surface row: "dead on pass tiles" / "pass-tile instance is non-functional".
- `:364` the Coverage paragraph: "the dead pass-tile delete-✕ (`pass_list.py:135-158` …)".

Also the §5 tooltip row at `:171` lists `widgets/pass_list.py` among the surfaces sharing
`preview_cell`'s "Delete" tooltip — `pass_list.py` no longer reaches that tooltip either
(`ui_primitives.py:1343-1344` sits inside the same `if deletable:` block).

### F2 — "Insert at caret" is reachable from two surfaces, not one (factual error)

`:288` ("Verbs reachable from exactly one surface") explicitly names "all three lib-tree
context menus' New/Rename/Copy-name/Favorite items" but the sentence's construction also
sweeps in the function-leaf menu's **"Insert at caret"**, which is a second surface for the
same verb:

- `shaderbox/popups/lib_picker/tree.py:359-360` — `menu_item_simple("Insert at caret", enabled=has_editor)` → `insert_name(app, fn)`
- `shaderbox/popups/lib_picker/__init__.py:135,145` — `primary_button("Insert at caret")` → `filtering.insert_name(app, selected)`

Both land on `filtering.insert_name` (`popups/lib_picker/filtering.py:91`). Identical label,
identical verb, two surfaces — this belongs in the "reachable from more than one surface"
table (`:272-284`) alongside "Open a pass's settings". The Help modal's own
`primary_button("Insert at caret")` (`popups/help.py:87` → `app.insert_text_at_caret`) is a
third instance of the label, on a different verb — worth naming since the table is about
label collisions as much as verb collisions.

### F3 — "New document" reaches three surfaces; the inventory credits none of them as a set

`app.create_document_from_example(STARTER_EXAMPLE_ID)` is reached from:

- `shaderbox/ui.py:720-723` — File menu `menu_item("New document", …)`
- `shaderbox/widgets/document_grid.py:45-46` — `standard_button("New document")`
- `shaderbox/app.py:641-643` — `CommandId.NEW_DOCUMENT` (Ctrl+Shift+N), palette + cheatsheet

The inventory names the menu-bar item only, in the parenthetical at `:316`. The document-grid
button is absent from §8 entirely, and the verb is missing from the cross-surface table at
`:272-284` — which is exactly the "should this become a menu" question the document exists to
answer, and this is a 3-surface verb with a **label match** (both surfaces say "New document"),
the counterexample to the two label-case drifts the table does record.

A fourth surface for the same modal family: `popups/examples.py:84`
`app.create_document_from_example(selected)` — the Examples modal's "Open a copy", which §1
does record.

### F4 — The document-grid surface is largely unrecorded in §8

`shaderbox/widgets/document_grid.py` contributes three inline action controls; §8 lists one.

- `document_grid.py:45-46` `standard_button("New document")` — **absent** (see F3).
- `document_grid.py:51-59` `imgui.checkbox("Render all", …)` writing
  `app.app_state.is_render_all_documents`, with a two-line explanatory tooltip at `:56-58`
  — **absent from §8 and from §5**, though the tooltip is exactly the §5 kind (carries more
  than the control's name). `:364` says "`document_grid.py:56` tooltip already counted" — it
  is not in the §5 table.
- `document_grid.py:92-99,108` the tile delete-✕/confirm — **present** (`:229`).

The grid's tile-click → `app.select_document(id)` (`document_grid.py:93`) is also an
unrecorded verb on the "choose a thing" family (compare the pass-tile click → `choose_output`,
which the cross-surface table does record at `:283`).

### F5 — §7 never enumerates the 32 commands; 22 are named nowhere in the document

§7 (`:211-218`) describes the palette / cheatsheet / rebinder / dispatcher as three surfaces
walking one table, then defers: "`shaderbox/commands.py` is the source of truth". The task's
premise is that the inventory records verb→surface reachability, and for two thirds of the
command table it does not. Never named anywhere in the file:

`SAVE`, `NEW_DOCUMENT`, `DELETE_DOCUMENT`, `TOGGLE_DOCUMENT_PLAY`, `RESET_DOCUMENT`,
`QUIT`, `JUMP_NEXT_ERROR`, `FORMAT_BUFFER`, `FOCUS_TAB_DOCUMENT`, `FOCUS_TAB_UNIFORMS`,
`FOCUS_TAB_RENDER`, `FOCUS_TAB_SHARE`, `TOGGLE_COPILOT`, `CYCLE_COPILOT_LAYOUT`,
`OPEN_SHADER`, `OPEN_SCRIPT`, `OPEN_GRAPH`, `CYCLE_CODE_TAB`, `CLOSE_CODE_TAB`,
`CYCLE_CHANNEL_VIEW`, `NEXT_PASS`, `PREV_PASS`
(`commands.py:17-48`, callbacks at `app.py:639-688`).

Several of these are genuine multi-surface verbs the cross-surface table should carry, each
verified by grep:

| Verb | Hotkey | Other surface(s) |
|---|---|---|
| Reset document | F6 (`commands.py:126-131`) | `tabs/document.py:243-246` `danger_button` + "Reset document" tooltip |
| Cycle channel view | Alt+V (`commands.py:181-186`) | `ui.py:990-999` channel-view chip |
| Toggle copilot | Alt+J (`commands.py:187-189`) | `ui.py:783` `toggle_button("Copilot")` — note two different `App` verbs: the hotkey calls `toggle_copilot` (`app.py:888`), the chip calls `toggle_copilot_open` (`app.py:900`) |
| Cycle copilot layout | Ctrl+H (`commands.py:190-196`) | `copilot_chat.py:711-712` layout-cycle icon |
| Open script | Alt+R (`commands.py:136`) | `tabs/document.py:437` entry-point "open" button; `tabs/code.py:362` |
| Open graph | Alt+G (`commands.py:137`) | `tabs/document.py:457` entry-point "open" button |
| Open shader | Alt+C (`commands.py:135`) | pass context menu "Open shader" (`pass_list.py:68-69`) — §3 records the menu item but the cross-surface row at `:279` says "pass-tile context menu, graph-node context menu" and omits the hotkey |
| Close code tab | Ctrl+W (`commands.py:152-158`) | `tabs/code.py:132` tab-bar ✕; `hotkeys.py:184` |
| Delete document | Alt+D (`commands.py:113-118`) | `document_grid.py:108` tile delete-confirm |

The `:279` row for "Open a pass's shader" is the sharpest case: it lists two surfaces for a
verb that has three.

### F6 — Eight state fields missing from the "App fields" table (`:318-342`)

Each holds an input-buffer / focus-pending / selected / armed state for a surface §8 already
lists, so the table under-reports its own §8:

| Field | `app.py:line` | Surface (already in §8) |
|---|---|---|
| `canvas_size_buf` / `canvas_w_editing` / `canvas_h_editing` | 401-403 | "Canvas W/H fields" (`:252`, `tabs/document.py:158-209`) — the per-half mirroring these implement is the field's whole mechanism |
| `aspect_buf` / `aspect_w_editing` / `aspect_h_editing` | 406-408 | "Aspect preset chips + custom W/H" (`:251`, `tabs/document.py:100-155`) |
| `copilot_input: str` | 442 | the chat's input box (`copilot_chat.py:300-302,327-328`) — the one text buffer of the chat surface |
| `errors_expanded: bool` | 560 | the code tab's error strip "+N more" / "show less" toggle (`tabs/code.py:329,367,372`) — an inline action control §8 does not list either |
| `active_document_tab` / `document_tab_select_pending` | 480, 483 | "Document/Uniforms/Render/Share tab bar" (`:261`, `ui.py:1028-1067`) — the one-shot that DRIVES the tab bar |
| `tab_select_pending: bool` | 557 | the code-editor tab bar (same one-shot pattern; `app.py:554-557` comment calls it the mirror of `document_tab_select_pending`) |
| `splitter_dragging` / `_splitter_press_on_splitter` | 528-529 | "Editor/app-panel splitter" (`:260`, `ui.py:786-794`) |
| `editor_focus_requested: bool` | 502 | the lib picker's insert → editor re-focus one-shot (`app.py:1289`) — a focus-pending field, the category the table's own header names |

Borderline, called out for completeness rather than as findings: `pending_project_switch` /
`pending_project_seed` (`app.py:535-536`) are the Projects modal's deferred switch — arguably
in-flight state for a surface the table does cover; `editor_error_note_pending` (`app.py:351`)
and `copilot_hovered` (`app.py:438`) are frame-transient, not surface state.

### F7 — `GraphViewState` row understates the fields it covers

`:342` lists `selection` / `scope` / `selected_wire` / `node_drag` / `wire_drag` for "graph
canvas inline controls (§8)". Two more fields hold state §8 describes and the row skips:

- `band_anchor: Position | None` (`graph_state.py:88`) — the rubber-band's press point, the
  state behind §8's "Rubber-band multi-select" row (`:234`).
- `fitted: bool` (`graph_state.py:79`) — §3 does record it under the canvas menu (`:133`),
  so the App-fields table is the inconsistency, not the document.

`press_blocked` (`graph_state.py:105`) is the gesture latch for the copilot-turn freeze §3
mentions at `:126`; worth a row for the same reason.

### F8 — §7's Esc funnel omits the rebind branch

`:218` describes `_handle_escape` as "closing exactly one thing per press, most-modal-first:
copilot revert target → `any_popup_open()` … → palette → copilot chat defocus". It omits the
**first** branch: `hotkeys.py:357-358` returns early when `app.rebinding_command is not None`
— the chord capture owns Esc, and the modal stays open. That is the same class of Esc-ownership
carve-out the sentence does record for the lib picker's inline inputs and the Projects name
input, and the inventory names `rebinding_command` in the Settings row (`:40`) and the fields
table (`:325`) without connecting it to the funnel.

### F9 — `app.session` verbs called from UI: one uncredited surface

The 18 `app.session.*(` call sites from widgets/popups/tabs/ui/exporters are otherwise
correctly folded into their surfaces. One gap: `:155` credits the sampler-source combo with
`app.session.set_sampler_source` (`widgets/uniform.py:325`) — correct — but
`app.session.delete_pass` (`widgets/pass_list.py:101`, inside `_delete_pass`) is the pass
context menu's "Delete", and §3's item list at `:124` says "→ `_delete_pass`" without naming
the session verb, so the "Delete a pass" cross-surface row at `:277` cannot be checked against
the core verb. Minor, and the reachability itself is right.

---

## False trail — checked and found correctly covered

These were enumerated from the code and matched the inventory with no drift:

1. **Every `menu_item_simple` call site (22) and every menu-bar `imgui.menu_item` (7).**
   Grepped all of `shaderbox/`; every label in the `:290-316` table resolves, the surface
   attribution is right in all 21 label rows, and the menu-bar parenthetical at `:316`
   correctly distinguishes the two call shapes. The "Favorite / Unfavorite label flips on
   state" note matches `tree.py:368`.

2. **All 12 combos.** `imgui.combo(` / `begin_combo(` / `labeled_combo(` / `grouped_combo(`
   across `shaderbox/` yields exactly the 8 rows of §4 plus the 4 deferred-to-their-modal
   combos at `:162`. Every `file:line` resolves to a combo call. No combo anywhere is
   missing.

3. **All 5 `pfd.*` dialog call sites.** §6's table plus `app.py:445` (the field declaration,
   not a call). The blocking/async split at `:207` is right: `ui.py:115` is the only
   non-`pfd_block` picker.

4. **Every `app.py` and `graph_state.py` line number in the fields table.** Spot-read all
   21 cited lines — `453` `popup_state`, `434` `copilot_revert_target`, `462`
   `is_palette_open`, `472` `rebinding_command`, `474` `lib_reset_armed`, `477`/`479`
   settings focus/mark, `412` `pass_draft`, `419` `import_draft`, `445` `file_pick_dialog`,
   `484`/`486` emoji, `487` `document_delete_armed`, `538` `projects_selected`, `567`
   `shader_lib_files`, `426` `is_copilot_open`, `518` `fps_details_open`, `347`/`352`
   editor lookup, `330` completion — every one lands on the field it claims.

5. **Every popup-opener line number in §1.** `app.py:1021` `any_popup_open`, `1029`
   `_open_popup`, `1036` `open_copilot_revert`, `1042` `open_examples`, `1045`
   `open_settings`, `1053` `open_pass_settings`, `1118` `open_add_pass`, `1163`
   `open_import_passes`, `1262` `open_emoji_picker`, `1267` `open_shader_lib_picker`,
   `1275` `open_help`, `2399` `pick_project_dir`'s `pfd.select_folder`, `2403`
   `open_projects`, `132` `PopupState` — all correct.

6. **The Settings modal's four programmatic openers.** `:38` claims copilot's gate plus "any
   exporter's config-not-connected gate"; confirmed at `copilot_chat.py:166`,
   `youtube.py:233`, `telegram.py:335`, all routing `open_settings(focus=...)` — and the
   cross-surface label-drift row at `:281` ("Settings..." vs "Open Settings") is right.

7. **§3's whole structure.** Read `pass_list.py:62-87` and `pass_graph.py:872-886,1478-1496`
   line by line: `pass_menu_items` is genuinely shared by exactly two callers, the
   `begin_popup_context_item` id asymmetry (explicit id vs `None`) is as described, the
   Python-side `and deletable` double-gate at `pass_list.py:75` matches the note at `:124`,
   the box-node branch really does bypass `pass_menu_items`, and "Group..." really is gated
   on `node.kind == "pass"` (so a ghost node gets the shared items without it — a nuance
   `:125` gets right by saying "only `node.kind == "pass"`").

8. **`ShaderLibFileManager`'s field set.** Read `shader_lib/file_ops.py:39-55`: the 7
   `picker_*` fields, 3 `InlineInput`s, 2 `*_delete_armed` — the summary row at `:335`
   enumerates all twelve correctly, including the mutex those `arm_*`/`begin_*` openers
   enforce via `reset_inline_state`.

9. **Both label-case drifts.** Verified against source: `tabs/document.py:477,480` say
   "add pass" / "import..." (lowercase) while `pass_graph.py:875,877` say "Add pass" /
   "Import..." (capitalized). Both `:275` and `:276` are right.

10. **The §5 tooltip anchors.** Every tooltip string cited resolves — `ui.py:1001` "Channel
    view", `copilot_chat.py:538` "Copy", `:548` "Revert this turn's changes", `:714`
    "Layout: {…}", `telegram.py:311` "Click to change emoji", `emoji_picker.py:65`
    `entry.name`, `document.py:246` "Reset document", `code.py:1128` the cursor-following
    value. Five `ui_primitives.py` citations (`1735` `clickable_label`, `1644`
    `draw_copyable_text`, `1672` `draw_link`, `1560`/`1567` the FPS panel) land a few lines
    off the primitive's own `def` (`1708`, `1630`, `1658`, `1540`, `1570`) — inside the
    right function body in each case, so these are anchor imprecision, not errors, and are
    not filed as findings.

---

## Round 2

Re-anchored to the code, not to round 1's notes: `COMMAND_SPECS` re-dumped by importing
`shaderbox.commands` and printing every spec's label / chord int / category / scope /
`in_palette` / `rebindable` / `repeat`, with the chord ints decoded back to key names through
`imgui.Key`; `app.py:639-688` `_build_command_callbacks` read in full; every `file:line` cited
in §7.1, the cross-surface tables and the App-fields table opened and read.

### Verdict: **PARTIAL** — all 9 round-1 findings CLOSED, 3 new findings (2 consistency, 1 fresh verb)

### 1. Closure of F1-F9

| Finding | Status | Inventory text that covers it |
|---|---|---|
| F1 — dead pass-tile delete-✕ | **CLOSED** | `:274` (the §8 row, rewritten to "**No glyph drawn — not a control.**", naming `deletable=False` at `pass_list.py:151`, `armed=False` at `:141`, the `if deletable:` guard at `ui_primitives.py:1338`, the `if selected and armed:` guard at `:1324`, and commit `97841ec`); `:276` ("contrast with the pass tile above, which draws no delete glyph at all"); `:355` (cross-surface row, "**not** the pass tile, which passes `deletable=False` and draws no glyph at all"); `:527-529` (Coverage). §5's shared-"Delete"-tooltip row at `:177` now carries the same carve-out. Re-verified in source: `pass_list.py:141` `armed=False`, `:151` `deletable=False`, `ui_primitives.py:1324`/`:1338`/`:1344` — all four land exactly. |
| F2 — "Insert at caret" is 2 surfaces | **CLOSED** | `:354` — a full multi-surface row naming `lib_picker/__init__.py:135,145` and `tree.py:359-360` as the same `filtering.insert_name` verb, and `popups/help.py:76-87` as a third surface with the same label on a different verb. Verified: `__init__.py:135` `primary_button("Insert at caret")`, `:145` the `insert_name` call. |
| F3 — "New document" reaches 3 surfaces | **CLOSED** | `:236` (§7.1 row listing File menu `ui.py:720-723` + `document_grid.py:45-46`), `:306` (§8 row for the grid button), `:347` (cross-surface row, all three plus the Examples-modal 4th path). Verified: `ui.py:720-723` and `document_grid.py:45-46` both call `create_document_from_example(STARTER_EXAMPLE_ID)`. |
| F4 — document-grid surface unrecorded | **CLOSED** | `:275` (tile click → `select_document`), `:306` ("New document"), `:307` ("Render all" checkbox), `:200` (the §5 tooltip row for `document_grid.py:56-58`). Verified against `document_grid.py:44-47,51-58,93-99`. |
| F5 — §7 never enumerates the 32 commands | **CLOSED** | §7.1 at `:227-266` — 32 rows, one per `CommandId`, table-declaration order. **All 32 rows re-verified field by field against `COMMAND_SPECS`: every label, every decoded chord, every category, every scope, and `in_palette=True` on all 32 are correct, zero drift.** The two scope subtleties are right: `CLOSE_CODE_TAB`/`FORMAT_BUFFER` = EDITOR, `CYCLE_COPILOT_LAYOUT` = COPILOT, and `CYCLE_CODE_TAB`'s row correctly separates its GLOBAL scope from its EDITOR cheatsheet category. `IMPORT_PASSES` chord `0` = "unbound" ✓. The `:266` footnote ("`repeat` and `rebindable` are both `True` for every spec") is correct — no spec sets either off. All nine multi-surface rows round 1 listed are now carried. |
| F6 — 8 state fields missing | **CLOSED** | `:419-425` — `errors_expanded` (560), `canvas_size_buf`/`canvas_w_editing`/`canvas_h_editing` (401-403), `aspect_buf`/`aspect_w_editing`/`aspect_h_editing` (406-408), `active_document_tab`/`document_tab_select_pending` (480, 483), `tab_select_pending` (557), `splitter_dragging`/`_splitter_press_on_splitter` (528-529), `editor_focus_requested` (502); plus `copilot_input` (442) at `:414`. **All 35 cited `app.py` line numbers re-read and every one lands on the field it claims.** |
| F7 — `GraphViewState` row understates | **CLOSED** | `:427` — the row now enumerates `selection`/`scope`/`selected_wire`/`node_drag`/`wire_drag`/`band_anchor`/`fitted`/`press_blocked` and says what each backs. Verified in `graph_state.py`: `fitted:79`, `band_anchor:88`, `press_blocked:105`. |
| F8 — Esc funnel omits the rebind branch | **CLOSED** | `:225` — the funnel now opens with "**chord-rebind capture** (`app.rebinding_command is not None`, `:357-358`, returns early…)"; `:41` repeats it in the Settings Close row. Verified: `hotkeys.py:357-358` is literally the first branch after the key-press test. |
| F9 — `app.session.delete_pass` uncredited | **CLOSED** | `:340` — the "Delete a pass" cross-surface row now reads "→ `_delete_pass` → `app.session.delete_pass`, `pass_list.py:101`". Verified at that exact line. Re-ran the whole `app.session.*`-from-UI sweep: the five mutating calls (`set_sampler_source` `uniform.py:325`, `delete_pass` `pass_list.py:101`, `set_pass_target` `pass_settings.py:146`, `set_pass_iterations` `:150`, `rename_pass` `:212`) are each credited to their surface; the rest are reads. |

### 2. Consistency — §7.1 vs. the two cross-surface tables

Checked in both directions: every §7.1 row whose "Other surfaces" cell is non-empty must appear
in the multi-surface table (`:337-360`) and must not appear in the single-surface list (`:364`);
every command named in the single-surface list must have a §7.1 cell reading "none found".

**R1 — Six commands carry a real second surface in §7.1 and appear in NEITHER cross-surface
table.** They are silently dropped: not in the multi-surface table's 24 rows, not in `:364`'s
"none found" enumeration.

| `CommandId` | §7.1 line | Second surface the row itself names | Verified |
|---|---|---|---|
| `OPEN_PROJECTS` | `:233` | File menu "Projects..." | `ui.py:724-728` → `app.open_projects()` |
| `QUIT` | `:235` | File menu "Quit" | `ui.py:729-730` → `glfw.set_window_should_close` (the hotkey routes `request_quit`, `app.py:869` — a second-method-same-effect case like `TOGGLE_COPILOT`'s, which the table does record) |
| `EXAMPLES` | `:262` | menu-bar "Examples" | `ui.py:748-749` → `app.open_examples()` |
| `HELP` | `:263` | menu-bar "Help" | `ui.py:750-751` → `app.open_help()` |
| `OPEN_LIB_PICKER` | `:256` | Library menu "Browse..." | `ui.py:742-745` → `app.open_shader_lib_picker()` |
| `JUMP_NEXT_ERROR` | `:247` | error-strip row click | `code.py:355,361-363` — the row click sets `app.editor_jump_request`, the hotkey calls `app.jump_to_next_error` (`app.py:872`) |

The §7.1 header at `:229` states the multi-surface table "can be regenerated from this table" —
it currently cannot, and these six are the proof. `HELP` and `EXAMPLES` are the sharpest: `:364`
names Help and Examples explicitly in its *single*-surface enumeration ("every item in §1/§2/§3
whose 'Opened by' lists a single entry path (Examples, Help …)"), while §1's own "Opened by"
lines at `:20` and `:29` each list three paths — bar item, hotkey, palette.

**R2 — Two commands are listed as "none found" in §7.1 AND in `:364`, but have a second
surface in the code.**

- `TOGGLE_CHEATSHEET` (`:264` "— (this is the cheatsheet's own toggle)", and named in `:364`'s
  none-found list). The Settings modal's General section has `imgui.checkbox("Show keyboard
  cheatsheet", app.app_state.show_cheatsheet)` (`popups/settings.py:87-89`) writing the very
  field `App.toggle_cheatsheet` flips (`app.py:885-886`). Two surfaces, one state. The
  inventory does record the checkbox — at `:39`, inside §1's Settings Items paragraph — so this
  is the tables disagreeing with §1, not a missing surface.
- `TOGGLE_DOCUMENT_PLAY` (`:238`, and in `:364`'s none-found list). Its own cell then spends
  four lines explaining that `document.py:443-448`'s play/stop toggle IS the same state and is
  "treated as one verb, two entries, in the cross-surface table" — and `:360` does carry a
  "Play/stop" multi-surface row. So the cell's opening words "none found beyond the hotkey"
  contradict the rest of the same cell, and `:364` then files it as single-surface anyway.
  `app.py:2090`'s own comment settles it: "The hotkey mirror of the document-tab play/stop
  toggle."

Everything else agrees. The remaining five names in `:364`'s none-found list — `SAVE`,
`NEXT_PASS`, `PREV_PASS`, `CYCLE_CODE_TAB`, `OPEN_PALETTE` — each match their §7.1 cell, and
four of the five are genuinely single-surface (greps for `app.save`, `cycle_code_tab`,
`open_palette` return only the callback wiring and the definition). `NEXT_PASS`/`PREV_PASS` are
the exception — see R3.

### 3. Fresh sweep — one verb the inventory still does not credit

**R3 — `NEXT_PASS` / `PREV_PASS` are a second surface for "Choose a pass as the document
output", and §7.1 files both as "none found".**

`app.py:686-687` wires both to `lambda: self.step_output_pass(±1)`; `step_output_pass`
(`app.py:2073-2087`) walks the strip order and calls `self.pick_pass(document_id, name,
focus_editor=…)`; `pick_pass` (`app.py:1948-1952`) is `ensure_shader_tab` + **`choose_output`**.
So Alt+Right / Alt+Left reach the same `App.choose_output` as the pass-tile click, the
graph-node click and the uniform-row texture-preview click — the three surfaces the
cross-surface row at `:359` lists. The row should read four surfaces, and `:240-241`'s "none
found" plus `:364`'s single-surface filing are both wrong.

This is a hotkey-that-shadows-a-click case, the same shape as `OPEN_SHADER`, which `:342` gets
right and calls out ("**a 3-surface verb**, the hotkey is a real third path the menu-only
framing misses").

The rest of the sweep found nothing new. Re-enumerated and matched:

- **All 32 `CommandSpec`s** — covered above; §7.1 is complete and field-accurate.
- **Every `App` `open_*`/`close_*`/`toggle_*`/`cycle_*`/`pick_*`/`choose_*`/`arrange_*`/
  `reset_*`/`delete_*`/`dissolve_*`/`leave_*`/`group_*`/`create_*`/`import_*`/`revert_*`/
  `recover_*`/`insert_*`/`request_*`/`select_*`/`set_*`/`ensure_*`/`commit_*`/`drop_*`/`unwire`
  method** (grepped off `^    def ` in `app.py`) — every one with a UI caller resolves to a
  surface the inventory names. `open_shader_lib_file` (`filtering.py:98`, `tree.py:185`,
  `code.py:925`), `open_declaration_file` (`uniform.py:75`) and
  `reveal_shader_lib_file_in_manager` (`tree.py:156,275`) are all inside surfaces §3/§5/§8
  already cover, and `:357` correctly records the last as a two-menu / one-implementation verb.
- **Every `menu_item_simple(` (22) and `imgui.menu_item(` (7) call site** — 29, unchanged from
  round 1, and every label still resolves to the `:370-391` table or the `:393` menu-bar
  parenthetical. The two variable-label sites (`tree.py:167`, `:280`) resolve to "Delete
  directory (recursive)"/"Confirm delete (recursive)" and "Delete"/"Confirm delete", exactly as
  the table has them.
- **Every `app.session.*(` call from a widget/popup/tab/exporter/`ui.py`** — 22 sites, 5
  mutating, all credited (see F9 above).

### 4. False trail — checked this round and found correct

1. **All 32 chords decoded.** `Alt+O / Ctrl+S / Ctrl+Q / Ctrl+Shift+N / Alt+D / F5 / F6 /
   Alt+Right / Alt+Left / Alt+C / Alt+R / Alt+G / Ctrl+Tab / Ctrl+W / F8 / Ctrl+Shift+I /
   Ctrl+1..4 / Alt+V / Alt+J / Ctrl+H / Alt+L / Ctrl+Shift+P / Alt+S / Alt+P / Alt+A /
   (unbound) / Alt+E / F1 / Alt+/` — every one matches §7.1's cell. `CYCLE_CODE_TAB` decodes to
   `named_key_begin`, which is `imgui.Key.tab`'s numeric alias, so §7.1's "Ctrl+Tab" is right.
2. **`OPEN_PASS_SETTINGS`'s panel-pass note.** `:259` claims the hotkey opens for the panel pass
   via `open_pass_settings_for_panel_pass`; confirmed at `app.py:1112-1116` and wired at
   `app.py:681`.
3. **`TOGGLE_COPILOT`'s two-method note.** `:254`/`:349` claim the hotkey calls `toggle_copilot`
   and the chip `toggle_copilot_open`; confirmed at `app.py:673`, `app.py:888`, `app.py:900`,
   `ui.py:783`.
4. **The nine multi-surface rows round 1's F5 table demanded.** Reset document
   (`document.py:243,246`), cycle channel view (`ui.py:990-1001`), toggle copilot, cycle copilot
   layout (`copilot_chat.py:711-714`), open script (`document.py:436-439`), open graph
   (`document.py:457-459`), open shader, close code tab (`code.py:118-119,132-133`), delete
   document (`document_grid.py:92-99,108`) — all nine now carried in §7.1 and all but the six in
   R1 also in the multi-surface table, each at a verified anchor.
5. **The `FORMAT_BUFFER` leader binding.** `:248`/`:353` claim `<leader>f`; confirmed at
   `commands.py:255-258`.
6. **§9.2's Esc-bypass table.** Re-read `hotkeys.py:354-399`: the `PASS_SETTINGS`,
   `IMPORT_PASSES`, `PROJECTS` and `SHADER_LIB_PICKER` carve-outs are exactly as described and
   in that order, and the `was_settings_open` apply-on-close at `:366,398-399` is right.
7. **Anchor imprecision worth naming but not filed as findings** (each lands inside the right
   construct): `:274` cites `pass_list.py:154` for `result.clicked`, actual `:155`; `:462` cites
   `hotkeys.py:386-390` for the lib-picker carve-out, actual `:386-389`; `:242`/`:259` cite
   `pass_list.py:68-69`/`:69-70` for the menu's "Open shader"/"Settings" items, actual `:68-69`
   and `:70-71` (`:47` gets the latter right). None changes a claim.
