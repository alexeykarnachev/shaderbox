# 093 / finding 17 — Menus: every menu, popup and modal on one shape

The spec for finding 17 of `00_findings.md`: the app's context menus, menu bar, popups and
settings modals reviewed together and refined, code included. It is the third stage of the
flow the maintainer laid out — an inventory (`04_menus_inventory.md` §1-9, reviewed to
convergence by three opus readers over two rounds), a design pass on it (§10, two opus
reviews and a closure round; the reasoning, the rejected shapes and the code facts each
decision rests on live THERE, cited by section) — and the input to the implementation. The
decisions below are the lock-in; §10 is why.

Size: **high-blast-radius** by the `dev_flow.md` preamble (a refactor across `ui.py`,
`commands.py`, `ui_primitives.py`, five popups, three widgets, `app.py`, `hotkeys.py`, and a
conventions bullet), so the post-implementation round is three reviewers with a
spec-fidelity audit among them, run to convergence, then `/sanitize`.

## Goal

Every verb of an object is on that object's context menu, drawn from one item set wherever the
object appears; every command is in the menu bar under its category with its chord, rendered
from the command table so a label has one spelling; a tile carries no button; a destructive
menu verb confirms one way; a name is entered one way; a modal closes one way, through its own
funnel, from Esc as from its button; the chrome the rulebook prescribes is pinned by a gate; the
copy inside the modals is inside the budget the repo already enforces.

## Out of scope

- The control panel's composition (finding 4): the Document tab's rows, the strip's caption
  and its `Add pass` / `Import passes` buttons keep their places; only their labels change
  (M1). Trigger: finding 4's own feature.
- The exporters' panels: the Telegram and YouTube config bodies, their `unconnected_gate`s'
  bodies, the Telegram pack forms and the hand-rolled red confirm child
  (`telegram.py:484-485`), the sticker grid's arm-and-wash. Their gate BUTTONS take the
  command's label (M1). Trigger: an exporter walk.
- The copilot chat's inline controls and the revert confirm's body (M6 touches its return
  shape only).
- The editor tab bar: no per-tab menu (§10 P2). Trigger: a second tab verb (`Close others`)
  is asked for.
- A general undo / recover path for a deleted document outside the copilot's card. Trigger:
  the maintainer deletes a document by mistake and asks.
- The uniform row: no menu (its one verb is the label's click).
- The group prompt's placement: it stays a popup (093 S5); only its commit rule changes (M5).

## Design decisions

**M1. The menu bar is a render of `COMMAND_SPECS`.** One top-level menu per `CommandCategory`
in `CATEGORY_ORDER`; one item per spec with `in_menu`, in table order; label = `spec.label`,
hint = `chord_to_str(app.effective_bindings[id])` (`""` for chord `0`); a `separator_before`
spec draws `imgui.separator()` first. `CommandSpec` gains `in_menu: bool = True` and
`separator_before: bool = False`; `in_menu` is `False` on `FOCUS_TAB_DOCUMENT`, `_UNIFORMS`,
`_RENDER`, `_SHARE` and `CYCLE_CODE_TAB`; `separator_before` is `True` on `QUIT`. The
right-aligned `project <name>` stays. Every `imgui.menu_item` in `ui.py::_draw_menu_bar` goes.

**M2. `shaderbox/menus.py` holds the `App`-facing menu primitives; `commands.py` holds the
pure label.** `menus.py` (imports `App`, `commands`, `ui_primitives`): `draw_menu_bar(app)`,
`command_menu_item(app, command_id) -> bool` (label, hint, `enabled=menu_enabled(app, spec)`,
fires `app.command_callbacks[id]()` on click and returns whether it fired),
`menu_enabled(app, spec) -> bool` (EDITOR scope -> `app.active_tab is not None`; COPILOT ->
`app.is_copilot_open`; GLOBAL -> `True`; per ITEM only — a `begin_menu` under
`begin_disabled` does not open, §10 P1). `commands.py`: `command_label(command_id) -> str`
(`SPEC_BY_ID[id].label`). `ui_primitives.py` stays `App`-free. `widgets/cheatsheet.py`'s
`_is_active` is not reused (it reads `editor_focused`, which a menu click has just cleared).

**M3. Every button that opens a command's surface takes `command_label`.** `tabs/document.py`'s
`add pass` -> `Add pass`, `import...` -> `Import passes`; `copilot_chat.py`'s and both
exporters' gate action label -> `Settings`; the Projects and Settings menu items lose their
`...`. Spec labels are the one spelling; no trailing ellipsis anywhere.
`tests/test_command_registry_coverage.py` pins the labels to the help snippet, so
`help_content.py`'s shortcuts section follows in the same diff.

**M4. Object menus: one item set per object kind, one function per set, the caller owns the
popup.**

| Set | Function | Items | Callers |
|---|---|---|---|
| pass | `pass_list.pass_menu_items(app, document_id, name)` | `Open shader` · `Settings` · ─ · `Leave group` (when grouped) · ─ · `Delete` ▸ `Delete pass <name>` | strip tile (`##pass_menu_{name}`), graph node (`None` id) |
| pass, node only | `pass_graph._node_menu` after the set | `Group...` between `Settings` and `Leave group` (092 D14: it seeds `view.selection`) | graph node |
| group box | `pass_graph._box_menu_items` | `Open` · `Dissolve` | graph box |
| document | `document_grid.document_menu_items(app, document_id)` | `Open` · `Open folder` · ─ · `Delete` ▸ `Move to trash` | grid tile (`##document_menu_{id}`; each tile is its own child, so an explicit id is safe) |
| canvas | `pass_graph._canvas_menu` | `Add pass` · `Import passes` (both `command_menu_item`) · ─ · `Fit` · `Arrange` | canvas background |

`Open` on a document is `app.select_document`; `Open folder` is `app.open_document_dir(id)`,
the generalization of `open_current_document_dir` (which calls it with the current id).
The document menu is inside the grid's `begin_disabled(copilot_turn_active)`, which a probe
shows keeps the popup from opening at all (§10 P2 coupling 3 — verify in the impl, and if a
disabled tile's right-click still opens it, gate the items in Python as the strip does).

**M5. A destructive menu verb confirms through a submenu.** `ui_primitives.confirm_menu_item(
label, confirm_label) -> bool`: `imgui.begin_menu(label)` holding one `menu_item_simple(
confirm_label)` drawn in `COLOR.STATE_ERROR` text; returns the inner click. Used by the pass
Delete, the document Delete, the lib tree's file Delete (`Delete` ▸ `Move to .trash`) and
directory Delete (`Delete directory` ▸ `Move every file to .trash`). The lib tree's armed
flip goes: `ShaderLibFileManager.file_delete_armed` / `dir_delete_armed`, `arm_file_delete` /
`arm_dir_delete`, and the two `push_style_color` red pushes (`tree.py:165-167`, `:279-281`).
`delete_file` / `delete_dir` keep their trash move and their toast. A modal's own destructive
button keeps its armed `danger_button` row (Projects' delete, Settings' library reset): inside
a modal the inline confirm row is the rulebook's shape. Reverses 092 D16's "the strip's
two-click arm" (already gone with 093 W3-2) and the lib tree's second-open confirm. Filed in
`conventions.md`: "a destructive verb on a menu confirms through a submenu; one inside a
modal through an armed danger row; a tile never carries one. Revisit if a submenu proves
unreachable on a touchpad."

**M6. The document grid's tile carries no button.** `draw_document_preview_button` passes
`deletable=False`; `App.document_delete_armed`, `set_document_delete_armed` and the cleanup at
`app.py:791-792` go with the grid's three result branches. `preview_cell` keeps `armed` /
`deletable` / `cell_delete_confirm` / `close_cross_button` for the sticker grid, its one
arming caller (out of scope). A dim `Right-click for actions` caption sits beside
`New document` (the one hint added; §10 P11).

**M7. One name-input row.** `InlineInput` moves from `editor_types.py` to `ui_primitives.py`
(the conventions trigger, met a third time; `file_ops.py` already imports imgui through
`theme`, so nothing new is pulled). `ui_primitives.name_input_row(id_, input, width) ->
InputRowResult(committed: bool, cancelled: bool)` draws the field with `enter_returns_true`,
consumes `input.needs_focus` once, reads `is_item_deactivated_after_edit()` on the line after
the input (before the `x`), draws the `x` cancel, and reports `committed` when Enter fired or
the deactivate did AND the cancel did not. Callers: the lib tree's rename and new-file/dir
rows (their own `wants_commit` lines go), the Projects new-name row (gains the deactivate
commit), and the group prompt, whose state becomes `GraphViewState.group_input: InlineInput`
(replacing `group_prompt: bool` + `group_name: str`; `InlineInput.target` unused there). The
group prompt stays a `begin_popup` (093 S5); a blank name still refuses to commit.

**M8. Modal chrome, one shape, gated.** `settings.py`: `is_keep_opened` -> `keep_open`.
`copilot_chat.py::_draw_revert_modal` returns `keep_open`; its caller nulls
`copilot_revert_target` and calls `close_current_popup`. Every action row is preceded by
`imgui.dummy((0, SPACE.MD))` (Examples, Help and the lib picker gain it; the revert modal's
`SPACE.XS` becomes `MD`; one spelling). Projects' Close joins the left-packed row.
Gate `tests/test_modal_chrome.py`: the domain is `PopupState` minus `CLOSED` plus the revert
modal, each resolved to its draw function through a table in the test (a member with no row
fails); for each, the AST of the function that returns the bool: a local named `keep_open`
is bound and returned; the last `standard_button(...)` call in it has label `Close` or
`Cancel`; an `imgui.dummy` whose first argument's second element is `SPACE.MD` (tuple or
`ImVec2` spelling) precedes the action row. Breaks tried and restored, named in the commit:
`keep_open` -> `ok` in one modal; one spacer deleted; a member added whose draw has no
Close row (on the lib picker).

**M9. `App.close_popup() -> bool` is the one close funnel.** Dispatch on `popup_state`:
`PASS_SETTINGS` -> `close_pass_settings()`; `IMPORT_PASSES` -> `close_import_passes()`;
`EMOJI_PICKER` -> new `close_emoji_picker()` (nulls `emoji_pick_target`, clears the query,
`CLOSED`); `SETTINGS` -> `apply_editor_settings()` + `CLOSED`; `PROJECTS` -> returns `False`
while `projects_input_owns_esc()`, else `CLOSED`; `SHADER_LIB_PICKER` -> returns `False`
while `inline_input_owns_esc(app)`, else `CLOSED`; `EXAMPLES`, `HELP` -> `CLOSED`; the
revert modal is not in the enum and keeps its own branch in `_handle_escape` first. Returns
`True` when it closed. `hotkeys._handle_escape` calls it in place of its four carve-outs and
the `was_settings_open` latch (`:366`, `:398-399`), which goes; the `rebinding_command` early
return stays. Each modal's own Close / Cancel branch calls the same verb. `tests/
test_import_dialog.py:63-67` is repointed at `close_popup` and made structural (the AST of
`close_popup` has a branch naming `IMPORT_PASSES` that calls `close_import_passes`).
`test_every_popup_state_has_a_draw_call` gains a clause: every member but `CLOSED` names a
branch in `close_popup`. Break: add a member, wire its draw, omit its close branch.

**M10. The disabled-item double gate goes; the two docs that prescribed it are corrected.**
`pass_list.py`: `if imgui.menu_item_simple("Delete", enabled=deletable):` with no `and
deletable` and no comment. `.claude/skills/imgui-ui/SKILL.md` §7.4's bullet becomes: "On
imgui-bundle 1.92.801 a `menu_item_simple(..., enabled=False)` refuses the click (measured
with a positive control, 093/17); gate in Python only where the item must ALSO refuse a
programmatic path." The inventory's §9.4 already records the measurement.

**M11. Copy inside budget.** `help.py:85` and `lib_picker/__init__.py:141`'s disabled-Insert
tooltips become `needs a shader caret`; the ten `_COPILOT_LIMITS` help markers become one
clause each (<= 8 words), their long form appended to the Help panel's copilot section; the
twelve `_OVER_BUDGET` rows in `tests/test_ui_prose_budget.py` for them are deleted, so the
gate holds them from now on.

**M12. The hint rule and the destructive rule are conventions.** `conventions.md ## Design
decisions` gains two bullets in "we decided X; revisit if Y" form: the confirm rule (M5) and
"a right-click hint sits over a modal's or panel's list; a canvas, a strip, or a card row
with a visible primary click gets none; revisit if a walk finds a menu undiscovered". The
repo's `.claude/skills/imgui-ui/SKILL.md` §7.4 is amended to match (it is the repo's file).

**M13. Prose-budget domain preserved.** `tests/test_ui_prose_budget.py` gains a row scoring
every `CommandSpec.label` in `commands.py` against the `menu_item_simple(label=)` budget (4
words), since `command_menu_item` takes an id and is invisible to the AST walk. Break: a
five-word label in `COMMAND_SPECS`.

**M14. Stays as it is** (§10.3): the nine modals and the mutex; the chips, combos and tab
bars; the editor tab bar; the lib tree's favorite star; the error strip, the lookup note,
the completion popup; the Examples selection persisting across opens; `toggle_copilot` /
`toggle_copilot_open`; the exporters' panels; the chat's inline controls.

## Files touched

- `shaderbox/commands.py` — `in_menu`, `separator_before`, `command_label`; the five `in_menu=False`
  and the one `separator_before=True`.
- `shaderbox/menus.py` (new) — `draw_menu_bar`, `command_menu_item`, `menu_enabled`.
- `shaderbox/ui.py` — `_draw_menu_bar` and `_hint` go; `menus.draw_menu_bar(app)` called.
- `shaderbox/ui_primitives.py` — `confirm_menu_item`, `name_input_row`, `InputRowResult`,
  `InlineInput` (moved in).
- `shaderbox/editor_types.py` — `InlineInput` moved out; its importers repointed
  (`app.py`, `popups/projects.py`, `shader_lib/file_ops.py`, `popups/lib_picker/tree.py`).
- `shaderbox/widgets/pass_list.py` — the item set per M4, `confirm_menu_item`, M10.
- `shaderbox/widgets/pass_graph.py` — `_node_menu` (Group... after the set), `_box_menu_items`,
  `_canvas_menu` through `command_menu_item`, `_group_prompt` through `name_input_row`.
- `shaderbox/widgets/graph_state.py` — `group_input: InlineInput`.
- `shaderbox/widgets/document_grid.py` — `document_menu_items`, the tile's menu, the hint,
  `deletable=False`, the result branches gone.
- `shaderbox/app.py` — `open_document_dir`, `close_popup`, `close_emoji_picker`;
  `document_delete_armed` + `set_document_delete_armed` + the `:791-792` cleanup gone.
- `shaderbox/hotkeys.py` — `_handle_escape` through `close_popup`.
- `shaderbox/popups/settings.py` (`keep_open`), `examples.py` / `help.py` /
  `lib_picker/__init__.py` (spacer), `projects.py` (Close position, `name_input_row`),
  `emoji_picker.py` (close through the verb), `lib_picker/tree.py` (M5, M7).
- `shaderbox/shader_lib/file_ops.py` — the two armed fields and their arm verbs gone.
- `shaderbox/widgets/copilot_chat.py` — the revert modal's return shape; the gate button label.
- `shaderbox/exporters/telegram.py`, `youtube.py` — the gate button label only.
- `shaderbox/tabs/document.py` — the two button labels.
- `shaderbox/help_content.py` — the shortcuts snippet's labels; the copilot limits' long form.
- `shaderbox/popups/help.py` — the tooltip (M11).
- Tests: `tests/test_modal_chrome.py` (new), `tests/test_menus.py` (new), `test_import_dialog.py`,
  `test_project_management.py`, `test_ui_prose_budget.py`, `test_button_tiers.py` (the
  `_NOT_A_VERB` allowlist if a hit rect moves), `test_pass_verbs.py` (the spy on `preview_cell`).
- Docs: `conventions.md` (M12, the `InlineInput` bullet's home, the graph bullet's menu
  sentence), `dev_flow.md` module map (`menus.py`, `pass_list.py`, `document_grid.py`,
  `ui_primitives.py`), `.claude/skills/imgui-ui/SKILL.md` §7.4 (M10, M12), 092's D14 / D16
  pointers, `00_findings.md` row 17's "Landed in", `01_spec.md`'s wave list, the roadmap.
- `projects/dev/` — nothing persisted changes shape (no migration; `git add` any sandbox drift).

## Verification

Each row fails for one reason. Frame-driven tests use the `app` fixture and the
`tests/test_graph_view.py::_frames` shape with `io.add_mouse_*` events; the menu-bar and
context-menu tests aim at item rects captured through `get_item_rect_*` inside the frame,
the way the probe in `reviews/menus_design_feasibility.md` §0 did (a pinned rig window, not
an auto-sized one).

| Guarantee | Test | Kind |
|---|---|---|
| M1: the bar is the table | `draw_menu_bar` inside a rig frame: the set of item labels drawn equals `{spec.label for spec in COMMAND_SPECS if spec.in_menu}` and each sits under its category's menu; break: hand-add one `imgui.menu_item` in `menus.py` — the label set gains one | frame-driven |
| M1: `in_menu` | the five view-focus specs have `in_menu=False` and no item is drawn for them; every other spec has one | pure + frame |
| M2: `menu_enabled` | EDITOR scope with `active_tab=None` -> `False`, with a tab -> `True`; COPILOT with the chat closed -> `False`; GLOBAL -> `True` while a popup is open | pure |
| M2: layering | `ui_primitives.py`'s source imports no `App` and no `commands`; `menus.py` imports both | pure |
| M3: one spelling | an AST walk over `shaderbox/`: no string literal equal to a spec label with a trailing `...`, and no `standard_button` / `primary_button` literal that case-insensitively matches a spec label but is not equal to it (`add pass` vs `Add pass`); break: restore `add pass` | pure |
| M4: the pass set is shared | the strip tile's and the node's menus draw the same labels in the same order except the node's `Group...` (spy on `menu_item_simple` / `confirm_menu_item` while each popup is open) | frame-driven |
| M4: a document's menu | right-click a grid tile: the popup opens with `Open` / `Open folder` / `Delete`; `Open` calls `select_document(id)`; `Open folder` calls `open_document_dir(id)` (spied) | frame-driven |
| M4: the canvas menu's commands | its `Add pass` item fires `command_callbacks[ADD_PASS]` (spied) and the modal opens in draft mode | frame-driven |
| M5: the submenu | `confirm_menu_item` in a context popup: hovering `Delete` opens the submenu; a click on the label alone deletes nothing (the pass count is unchanged after the frame); a click on the inner item calls `delete_pass` once. Break: return the outer click — the label click deletes | frame-driven |
| M5: the lib tree's arm is gone | `ShaderLibFileManager` has no `file_delete_armed` / `dir_delete_armed`; `tree.py` contains no `push_style_color`; a click on the inner item moves the file into `.trash/` and toasts | pure + frame |
| M6: no button on the document tile | `document_grid.py` passes `deletable=False`; `App` has no `document_delete_armed`; with a document selected, no item named `del_document_*` is submitted in the frame (spy on `close_cross_button`: zero calls) | pure + frame |
| M7: `name_input_row` commits on deactivate | focus the row, type, click elsewhere: `committed=True` once; type, press Esc / click the `x`: `cancelled=True`, `committed=False`; Enter: `committed=True`. On the group prompt: a click-away with `blur` typed calls `group_selection` once; blank does not | frame-driven |
| M7: the promotion | `editor_types.py` has no `InlineInput`; `ui_primitives.py` has it; `file_ops.py` imports it from there | pure |
| M8: the chrome gate | `tests/test_modal_chrome.py` as specified; the three breaks named in the commit | pure |
| M9: Esc through the funnel | open the emoji picker with a target set; press Esc: `emoji_pick_target is None` and `popup_state == CLOSED`. Open Projects, arm the new-name input, press Esc: the modal stays open and the input closes. Open Settings, change a field, Esc: `apply_editor_settings` called once (spied). Break: restore the direct `CLOSED` write in `_handle_escape` — the target dangles | frame-driven |
| M9: the dispatch gate | `test_every_popup_state_has_a_draw_call`'s new clause: every member but `CLOSED` names a branch in `close_popup`'s AST; the break named in M9 | pure |
| M10: no double gate | `pass_list.py`'s `Delete` site has `enabled=deletable` and no `and deletable`; a click with one pass in the document deletes nothing (the item is disabled; the probe's fact) | pure + frame |
| M11: budget | `tests/test_ui_prose_budget.py` passes with the twelve allowlist rows removed; break: restore one long marker | pure |
| M13: labels in the budget | a five-word label added to `COMMAND_SPECS` in a temporary copy fails the new row | pure |
| M14: nothing else moved | the exporter files' diffs touch label strings only; `preview_cell`'s signature is unchanged; `toggle_copilot` / `toggle_copilot_open` both exist | review |

## Open questions for the user

None blocking. One call he may overturn at the review: the pass `Delete` confirms through
the submenu (M5, §10.4); an immediate delete is a one-line change to the pass row.

## Review history

Design review of §10 revision 1: `reviews/menus_design_brief.md` (PARTIAL, 12 findings,
three blocking) and `reviews/menus_design_feasibility.md` (FEASIBLE WITH CHANGES; P4, P8,
P9 not as written). Revision 2 folded every finding (§10.5). Closure round:
`reviews/menus_design_closure.md`.
