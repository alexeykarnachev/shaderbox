# Post-implementation review — CODE CORRECTNESS — `f045eef` (093/17 menus)

Scope: `f045eef` ("093: menus -- one shape for every menu, popup and modal") against
`05_menus_spec.md` M1-M14. Role: code correctness — frame order and imgui stack, state
writes out of draw functions, the close funnel's cleanup, the submenu confirm, the
`InlineInput` move, resource/session lifetime on the new delete paths, `menu_enabled`'s
predicate, and whether each new test fails for the break it names.

Every finding below is demonstrated by a run against the working tree, a probe on the pinned
`imgui-bundle 1.92.801`, or a mutation with the whole suite behind it. A baseline worktree at
`f497bf5` was used for the two before/after comparisons.

**Verdict: PARTIAL.** 7 findings — 1 HIGH, 1 MEDIUM-HIGH, 1 MEDIUM, 4 LOW. No imgui
stack imbalance, no state write out of a draw function that the conventions reserve for an
`App` verb, no resource or session leak on either new delete path, and no close path that
skips its cleanup. The funnel (M9) is correct on every path I walked. The defects are: one
unbounded layout feedback loop the primitive move introduced, one destructive verb the bar now
exposes with no confirm against M5's own rule, and a class of new tests that assert menu LABELS
without asserting that any item reaches its verb — four separate verbs can be gutted with all
2499 tests green.

`make gates` on the tree as committed: **exit 0**, smoke passed (captured unpiped to
`gates.log`, `$?` read first).

---

## Findings

### 1. HIGH — the graph's Group prompt grows 72px every frame until it fills the screen

`name_input_row`'s default `width=0.0` branch reads `imgui.get_content_region_avail().x`
(`ui_primitives.py:569-573`). Inside an auto-sized `begin_popup` — which is exactly what
`pass_graph._group_prompt` is (`pass_graph.py:1511-1514`) — the avail the row asks for is the
width the popup was sized to LAST frame, and the width it then requests becomes next frame's
size. The loop has no fixed point.

The pre-093/17 code did not have it: the old prompt set a constant
`imgui.set_next_item_width(float(SIZE.NAME_INPUT_W))` (`f497bf5:pass_graph.py`).

Measured, driving the real `update_and_draw` loop on the `app` fixture and reading
`imgui.get_window_size().x` inside the prompt each frame:

```
f045eef  POPUP WIDTHS PER FRAME:
  [16, 292, 364, 436, 508, 580, 652, 724, 796, 868, 940, 1012, 1084, 1156, 1228,
   1300, 1372, 1444, 1516, 1588, 1660, 1732, 1804, 1876, 1948, 2020, 2092, 2164,
   2236, 2308, 2380, 2452, 2524, 2554, 2554, 2554, ...]        (clamps at the viewport)

f497bf5  BASELINE POPUP WIDTHS:
  [16, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332]  (stable)
```

+72px per frame; at 60fps the prompt covers the whole viewport in about half a second, and it
stays there for as long as the user is typing a group name. Reproduced identically with the
primitive alone inside a bare `begin_popup` (avail 269 → 335 → 401 on successive frames), so
the cause is the primitive's avail read, not anything in the graph.

The other three `name_input_row` callers are unaffected because their container is bounded: the
lib tree's rename and new-file/dir rows sit in a fixed-width child, and Projects passes
`width=float(SIZE.NAME_INPUT_W)` explicitly.

Fix shape: the group prompt passes an explicit `width`, or `name_input_row` falls back to
`SIZE.NAME_INPUT_W + cancel_w` rather than to `avail` when the container is auto-sized.

**Related, same function, LOW:** the `width` parameter does not do what its name says. With
`width=180.0` (`SIZE.NAME_INPUT_W`, what Projects passes), `field_w = 180 - cancel_w ≈ 156`,
then `imgui.set_next_item_width(max(float(SIZE.NAME_INPUT_W), field_w))` → `max(180, 156)` →
**180**. The `cancel_w` reservation the line above computes is discarded, so the row is
`width + cancel_w` wide, not `width`. For every caller whose `width` is at or below
`NAME_INPUT_W + cancel_w` the parameter is inert.

---

### 2. MEDIUM-HIGH — the menu bar now offers `Delete document` as a one-click, unconfirmed verb

M1 renders the whole table, and `DELETE_DOCUMENT` carries the default `in_menu=True`:

```
$ uv run python -c "from shaderbox.commands import SPEC_BY_ID, CommandId, chord_to_str; \
  s=SPEC_BY_ID[CommandId.DELETE_DOCUMENT]; print(repr(s.label), chord_to_str(s.default_chord), s.scope, s.in_menu)"
'Delete document' Alt+D CommandScope.GLOBAL True
```

`command_menu_item` fires `app.command_callbacks[DELETE_DOCUMENT]()` →
`App.delete_current_document` → `delete_document` → `_delete_document_unguarded`
(`app.py:2465-2478`). There is no confirm anywhere on that path.

This lands in the same commit that moved the document grid's delete BEHIND a two-step submenu
confirm for exactly this reason. M5 is written as a rule, not a case: *"a destructive verb on a
menu confirms through a submenu; one inside a modal through an armed danger row; a tile never
carries one."* A `Document ▸ Delete document` item is a destructive verb on a menu, and it does
not confirm.

It is also new surface. The old bar had seven hand-written items and none of them was
destructive (`git show f497bf5:shaderbox/ui.py`, seven `imgui.menu_item(` calls: New document,
Projects, Quit, Settings, Browse, Examples, Help). `Alt+D` existed, but a chord is a deliberate
two-key act; a menu item one pointer-slip below `New document` is not. The commit body counts
this among the "eight commands that had no mouse home now have one" without noting that one of
the eight trashes the open document.

Fix shape: either `in_menu=False` on `DELETE_DOCUMENT` (the tile's menu is its mouse home, as
M6 decided), or `command_menu_item` gains a confirm-submenu form and this spec uses it.
`RESET_DOCUMENT` was checked and is NOT in this class — `session.reset_document` restarts the
document, it does not destroy anything on disk.

---

### 3. MEDIUM — the four new object-menu verbs can each be gutted with all 2499 tests green

The spec's Verification table asks for the wiring, per row: *"`Open` calls
`select_document(id)`; `Open folder` calls `open_document_dir(id)` (spied)"* and *"a click on
the inner item calls `delete_pass` once"*. What landed asserts the LABELS and the structure.
Four mutations, each run against the full suite the way `make test` runs it
(`-n 8 --dist loadgroup`):

| Mutation | Suite result |
|---|---|
| `pass_list.pass_menu_items`: `Delete`'s body → `pass` | **2499 passed, 4 skipped** |
| `document_grid.document_menu_items`: `Delete`'s body → `pass` | **2499 passed, 4 skipped** |
| `document_grid.document_menu_items`: `Open folder`'s body → `pass` | **green** (`-k documents_menu`) |
| `pass_graph._box_menu_items`: `Dissolve`'s body → `pass` | **2499 passed, 4 skipped** |

Why each test survives its own break:

- `test_a_documents_menu_carries_open_open_folder_and_delete` (`test_menus.py:363-379`) asserts
  `spy.labels == ["Open", "Open folder", "Delete"]`, then its "the verbs each menu item calls"
  block runs `app.select_document(document_id)` and `app.open_document_dir(document_id)`
  **directly**. Both names are monkeypatched to appenders, so `opened == [document_id] and
  revealed == [document_id]` asserts only that the two lambdas the test itself installed were
  called by the test itself. The menu item's body is never on that path.
- `test_the_group_box_menu_is_open_and_dissolve` (`:398-401`) asserts labels only.
- `test_the_pass_set_is_the_same_on_the_strip_and_the_node` (`:329-...`) asserts label order
  only, and its "node" arm rebuilds the node menu inline (`pass_menu_items(...)` +
  `imgui.menu_item_simple("Group")`) rather than driving `pass_graph._node_menu`, so a drift in
  the real node menu is also invisible to it.
- `test_the_confirm_submenu_needs_the_inner_click` (`:435-...`) drives the REAL
  `confirm_menu_item` correctly with synthetic mouse events — the commit's note about the first
  draft is right and the fix was the right one — but its confirm sink is a local
  `confirmed: list[int]`. It proves the primitive; nothing proves the primitive is wired to
  `_delete_pass`.

The wiring itself is CORRECT — probed directly rather than assumed. Forcing
`confirm_menu_item` to return True inside a real `pass_menu_items` frame removes the pass
(`PASSES AFTER: ['main']`), and a right-click on a real grid tile opens the document menu
(`OPENED SEQUENCE [False, False, False, False, False, True, True, True]`, labels
`['Open', 'Open folder', ...]`). So this is a gate hole, not a live defect — but it is the gate
hole for the four verbs this feature is about, and the spec asked for it by name.

---

### 4. LOW — an ungrouped pass's menu draws two adjacent separators

`pass_list.pass_menu_items` (`:66-87`) emits `imgui.separator()` unconditionally before the
Leave-group line and again before Delete, but the Leave-group ITEM is short-circuited away for
an ungrouped pass (`document.graph.passes.get(name, PassEntry()).group and
imgui.menu_item_simple("Leave group")`). Most passes are ungrouped, so this is the common case.

Measured by spying `imgui.separator` / `menu_item_simple` / `begin_menu` through one real frame
of `pass_menu_items` on the starter document's only pass:

```
MENU ORDER (ungrouped pass): ['Open shader', 'Settings', '---', '---', 'menu:Delete']
ADJACENT SEPARATORS: True
```

M4's item table reads `Open shader · Settings · ─ · Leave group (when grouped) · ─ · Delete`,
which the doubled rule does not match. Fix: hoist the group test into a local and gate the
first separator on it.

---

### 5. LOW — two structural tests read `shaderbox/app.py` by a cwd-relative path

`tests/test_modal_chrome.py:247` and `tests/test_import_dialog.py` (the rewritten
`test_escape_reaches_the_close_funnel`) both do
`ast.parse(Path("shaderbox/app.py").read_text(encoding="utf-8"))`. `test_modal_chrome.py`
already defines an absolute `_PKG` at line 208 and uses it for every other file read in the
module; only the funnel clause reaches around it.

```
$ cd /tmp && .venv/bin/python -m pytest .../tests/test_modal_chrome.py -q
E  FileNotFoundError: [Errno 2] No such file or directory: 'shaderbox/app.py'
FAILED test_modal_chrome.py::test_the_close_funnel_covers_every_popup_state
1 failed, 44 passed
```

`make test` always runs from the repo root, so this is latent, not live. The fix is one word:
`_PKG / "app.py"`.

---

### 6. LOW — `_group_prompt` leaves `group_input` open when the popup is dismissed by a click away

`pass_graph._group_prompt` closes `view.group_input` on the commit and on the cancel
(`:1519-1526`), but a `begin_popup` also closes when the user clicks outside it — and then
`begin_popup` returns False and the function early-returns at `:1513` with `target` and `buf`
still set. `GraphViewState.group_input.is_open` is then True with no popup on screen.

Nothing reads that state today (the popup is re-opened through
`InlineInput.open()`, which resets all three fields), so this is dead state rather than a live
bug. It is filed because `is_open` is the field's documented meaning — *"`target is None` = the
input is closed"* — and a future reader who gates on it gets a wrong answer.

---

### 7. LOW — the lib picker's Esc branch is now unreachable on the path it was written for

`lib_picker/__init__.py:145-146` still carries
`if not input_owns_keys and imgui.is_key_pressed(escape): keep_open = False`. Since M9,
`hotkeys._handle_escape` runs before the draw and calls `App.close_popup()`, whose
SHADER_LIB_PICKER branch refuses only while `inline_input_owns_esc()` — the same condition,
plus the tag input. So on every frame this branch could fire, the funnel has already set
`popup_state = CLOSED` and `draw_lib_picker`'s guard (`:33-34`) has returned before the body
runs. The branch is harmless but dead.

Noted separately, and NOT filed as a finding because it is unchanged from `f497bf5`: the
moved `ShaderLibFileManager.inline_input_owns_esc()` includes `picker_tag_input_focused`, while
the picker's own comment two lines above the dead branch says the tag input deliberately does
NOT own Esc *"because gating it on tag_input_was_focused only forced a useless first Esc"*. The
funnel produces that useless first Esc. The move preserved the pre-existing behaviour exactly,
so it is out of this commit's scope.

---

## What was checked and found CORRECT

Each of these was the target of a specific probe or walk, not a reading.

**The imgui stack.** `menus.draw_menu_bar`'s `return` inside `with imgui_ctx.begin_menu_bar()`
and `continue` inside `with imgui_ctx.begin_menu(...)` are both safe: `_BeginEndMenuBar.__exit__`
and `_BeginEndMenu.__exit__` call their `end_*` only when `self.visible`
(`imgui_ctx.py:253-255, 288-...`). `confirm_menu_item`'s
`push_style_color` / `pop_style_color(1)` pair sits entirely inside `if menu:` with no return
between them. No new `begin_disabled` without its `end_disabled`, and the grid's
`begin_disabled` / `end_disabled` bracket is unchanged around the new popup site.

**Item-scoped queries land on the right item.** `name_input_row` reads
`is_item_deactivated_after_edit()` and `is_item_focused()` on the two lines AFTER `input_text`
and BEFORE `imgui.same_line()` + the `x` button — so both answer for the field. This is the
defect the commit body claims it fixes in Projects, and the code matches the claim:
`projects.py:165` now assigns `app.projects_input_focused = result.focused` (the field) where
`f497bf5` read `imgui.is_item_focused()` at a point where the `x`/`Cancel` button was the last
submitted item. Confirmed by `test_a_focus_move_with_no_edit_commits_nothing` and by driving the
row: a bare focus move commits nothing; a click-away after an edit commits exactly once; the `x`
reports `cancelled` and never `committed` on the same frame.

**The close funnel (M9) reaches every cleanup.** Every write to `popup_state` in `shaderbox/`
was enumerated (`grep -n "popup_state" shaderbox/` → 13 hits): the constructor, the dead-pointer
recovery pre-set at `app.py:580`, `close_popup`'s own branches, `_open_popup`,
`close_pass_settings`, `close_import_passes`, `close_emoji_picker`. No draw function writes it
by hand any more — `test_no_modal_writes_the_closed_state_by_hand` covers the eight enum
modals, and I verified the revert modal (outside the enum) nulls `copilot_revert_target` in its
CALLER, the only place it could. `emoji_pick_target` is nulled on both paths that can close the
picker: `close_emoji_picker` (Esc, via `_handle_escape` → `close_popup`) and the picker's own
Close (`emoji_picker.py:22` → `app.close_popup()` → same branch). The dangling-target defect the
commit names is really fixed.

**`close_popup` does not refuse a Close the user clicked.** Both refusing branches were walked.
The lib picker's own Close calls `reset_inline_state()` and clears `picker_tag_input_focused`
BEFORE `close_popup()` (`lib_picker/__init__.py:39-42`), so `inline_input_owns_esc()` is False by
the time the funnel asks. Projects' `_draw_body` returns `_draw_name_input(...)` while the input
is open, and `_draw_name_input` returns False only after `state.close()` has run
(`projects.py:178-181`), so `projects_input_owns_esc()` is False on every path that reaches the
funnel with a False. The Close button is not reachable at all while the name row is up.

**PROJECTS' Esc now runs `reset_projects_state()`, which the old path did not.** `f497bf5`'s
`_handle_escape` wrote `popup_state = CLOSED` bare for Projects, leaving `projects_delete_armed`
and `projects_error` set for the next open. This is a fix, not a regression.

**No resource or session leak on either new delete path.** The pass menu's Delete still routes
through `pass_list._delete_pass`, which captures `source.path` before the core delete and calls
`app.close_editor_for_path(doomed)` — unchanged, and it is the path the GRAPH node menu takes
too (`_node_menu` → `pass_menu_items`). The document menu's Delete calls `app.delete_document`,
which is the same guarded verb the old tile confirm called; `App._on_document_deleted`
(`app.py:776-791`) still pops the editor session, closes it, drops the marker state, re-anchors
the tabs and forgets the render state — the only line removed is the
`document_delete_armed` clear, whose field no longer exists.

**The mid-loop delete the old code deferred is safe.** `f497bf5` deferred `id_to_delete` until
after the grid loop with a comment about mutating `app.ui_documents` mid-iteration; the new code
calls `app.delete_document` from inside `document_menu_items`, inside the loop. The loop iterates
`list(app.ui_documents.items())` — a snapshot — so the mutation is absorbed. Driven: a document
deleted from inside the loop leaves the grid drawing for three further frames with no crash and
the dict correctly reduced.

**`menu_enabled` gates the right items.** The predicate was read against the whole table
(27 in-menu specs). Only `CLOSE_CODE_TAB` and `FORMAT_BUFFER` are `CommandScope.EDITOR`; only
`CYCLE_COPILOT_LAYOUT` is COPILOT. The spec's EDITOR test is `app.active_tab is not None` rather
than the dispatcher's `app.editor_focused` (`hotkeys.py:333`) — deliberate per M2, and safe for
the graph-tab case the brief asked about: `format_current_editor` returns early when
`formatter_for(tab.kind)` is None, and `close_active_tab` closing a graph tab is what the tab
bar's own x does. GLOBAL staying enabled behind a modal is also safe, probed rather than
reasoned: with Settings open, a click at the File menu's position leaves `popup_state` at
`SETTINGS` — imgui's modal owns the input and the bar is not reachable.

**The two library facts the commit rests on hold on this build.** `begin_menu(label, enabled=False)`
never opens: aiming at the disabled `Delete` submenu's own rect for four frames fires nothing
(`inside == []`). `begin_popup_context_item` on an item inside `begin_disabled` never opens: the
same right-click that opens the enabled grid tile's menu at frames 6-8 produces
`DISABLED OPENED SEQUENCE [False × 9]`. `imgui.menu_item(label, shortcut, p_selected, enabled)`
returns `Tuple[bool, bool]` per the installed `.pyi:2296`, so `command_menu_item`'s `[0]` is the
activation bool.

**The menu bar renders.** `CATEGORY_ORDER` covers every category in `COMMAND_SPECS` with no
leftovers and no empty category (File 3, Document 6, Editor 6, View 3, Tools 9 = 27 items).
`QUIT` is the only `separator_before` spec and it is last in File, so the separator sits above
it rather than leading a menu. The right-aligned project label lands at x ∈ [779, 884] in a
900px rig — flush against the right edge, as before.

**The gates the commit claims it broke do fire.** Two were re-broken here rather than taken on
the commit's word: flipping `deletable=False` → `True` in `document_grid` fails
`test_the_document_tile_draws_no_delete_cross`; a five-word `small_caption` literal in
`_draw_passes` fails `test_every_measured_site_is_within_budget` despite that function's new
`_UNMEASURABLE` row (the row is keyed per call-site argument, not per function, so it does not
widen the exemption — see the false trails).

---

## False trails

Each of these looked like a finding and is not. Recorded so the next reader does not re-walk them.

1. **A segfault in the group prompt.** A first probe crashed reproducibly (3/3 runs) the moment
   the prompt drew, which read as a real defect. It was my probe calling
   `imgui.is_popup_open("##graph_group")` OUTSIDE a `new_frame`/`end_frame` pair. Removing that
   one line makes the same probe pass. Nothing in `f045eef` segfaults.

2. **`_UNMEASURABLE`'s two new rows widening the prose gate.** The rows
   `("shaderbox/tabs/document.py", "_draw_passes")` and
   `("shaderbox/ui_primitives.py", "confirm_menu_item")` look like whole-function exemptions.
   They are not: the list is consulted only for sites the walk already classed UNREADABLE
   (a non-literal argument), and literal strings in the same function are still measured.
   Demonstrated by giving `_draw_passes` a five-word `small_caption` literal — the gate fails on
   `shaderbox/tabs/document.py::_draw_passes:472 small_caption.text`.

3. **`ui.py` keeping `CommandId` / `chord_to_str` after `_draw_menu_bar` was deleted.** Both are
   still used at `ui.py:865` for the empty-grid hint. Not dead imports.

4. **The document tile's `begin_popup_context_item` anchoring on a child window rather than the
   selectable.** `preview_cell` wraps the whole tile in `begin_child`, so the last submitted item
   at the call site is the child, not the tile's selectable — which looked like it might make the
   right-click miss. It does not: the tile's menu opens on a real right-click at the tile's rect
   (probe above), because the explicit `##document_menu_{id}` str_id makes imgui test the last
   item's rect, and the child registers as one.

5. **The `test_the_lib_trees_armed_delete_is_gone` assertion `"COLOR.STATE_ERROR" not in
   tree_source` being weaker than the spec's `"tree.py contains no push_style_color"`.** The
   commit body says the swap was to spare the favourite star's own push, and that is exactly what
   the file shows: the only remaining `push_style_color` is `tree.py:266`, `COLOR.FAVS if is_fav
   else COLOR.FG_DIM`. The substitution is correct and §10.3 keeps the star.

6. **A GLOBAL menu item firing while a modal is open.** `spec_eligible` gates GLOBAL chords on
   `popup_suppresses(scope) and popup_open`, and `command_menu_item` has no such gate — but the
   modals are `begin_popup_modal`, which owns the input, so the bar is not clickable behind one
   (probed; `popup_state` is unchanged by the click).

---

## Coverage

Read end to end (not only the hunks): `shaderbox/menus.py`, `shaderbox/commands.py`,
`shaderbox/hotkeys.py`, `shaderbox/widgets/document_grid.py`, `shaderbox/widgets/pass_list.py`,
`shaderbox/widgets/graph_state.py`, `shaderbox/popups/emoji_picker.py`,
`shaderbox/popups/examples.py`, `shaderbox/popups/help.py`, `shaderbox/popups/projects.py`,
`shaderbox/popups/lib_picker/__init__.py`, `shaderbox/popups/lib_picker/tree.py`,
`shaderbox/popups/pass_settings.py`, `shaderbox/shader_lib/file_ops.py`,
`shaderbox/tabs/document.py`, `shaderbox/help_content.py`, `shaderbox/editor_types.py`,
`shaderbox/copilot/config.py`, `tests/test_menus.py`, `tests/test_modal_chrome.py`, and the
diffs of `tests/test_import_dialog.py`, `tests/test_project_management.py`,
`tests/test_ui_prose_budget.py`, `tests/test_lib_files.py`, `tests/test_graph_view.py`.

Read in the regions the commit touches plus everything the touched code reaches:
`shaderbox/app.py` (the close funnel, `_open_popup`, `any_popup_open`, `escape_has_job`,
`_on_document_deleted`, `delete_document` / `_delete_document_unguarded`,
`open_document_dir` / `open_current_document_dir`, `close_emoji_picker`,
`reset_projects_state`, `projects_input_owns_esc`, `close_active_tab`,
`format_current_editor`, `delete_current_document`, `reset_current_document`),
`shaderbox/ui.py` (the deleted `_draw_menu_bar` / `_hint` region, the `menus.draw_menu_bar`
call site, the import block), `shaderbox/ui_primitives.py` (`confirm_menu_item`, `InlineInput`,
`InputRowResult`, `name_input_row`, `preview_cell`, `PreviewCellResult`, `modal_window`,
`context_menu_style`), `shaderbox/widgets/pass_graph.py` (`_canvas_menu`, `_node_menu`,
`_box_menu_items`, `_group_prompt`, the draw ordering around `:1368`),
`shaderbox/widgets/copilot_chat.py` (`_draw_revert_modal`, `_draw_revert_body`, the gate label).

**Skipped, and why:** `shaderbox/exporters/telegram.py` and `shaderbox/exporters/youtube.py`
beyond their diffs — the change is one `action_label` string each, the exporter panels are
out of scope by the spec's own "Out of scope" list, and `preview_cell`'s arming path (their one
consumer) was read in `ui_primitives.py` and confirmed unchanged.
`.claude/skills/imgui-ui/SKILL.md`, `ai_docs/conventions.md`, `ai_docs/dev_flow.md`,
`ai_docs/features/092_graph_view/03_spec.md`, `00_findings.md`, `01_spec.md` — documentation,
which is the spec-fidelity reviewer's surface, not correctness.

Commands run: `make gates` (exit 0, unpiped), the full suite under `-n 8 --dist loadgroup`
four times (once clean, three times under a mutation), `tests/test_menus.py` and
`tests/test_modal_chrome.py` alone (72 passed), and eight scratchpad frame-driven probes on the
`app` fixture. A baseline worktree at `f497bf5` supplied the before/after for finding 1.
