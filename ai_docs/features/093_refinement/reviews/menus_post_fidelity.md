# Post-implementation review — 093/17 menus: SPEC-FIDELITY AUDIT

Commit under review: `f045eef` ("093: menus -- one shape for every menu, popup and modal").
Spec: `ai_docs/features/093_refinement/05_menus_spec.md` (M1-M14 + the 19-row Verification
table). Role: spec fidelity — every sentence that states a behavior, a name, a location, a
label, a field, a default or a deletion, walked against the diff and the resulting tree.

Method: the tree at `f045eef` read directly; every verification row traced to the test that
claims it; four mutation probes run in a throwaway worktree (`git worktree add … f045eef`,
removed after) to decide whether a test is decisive or merely green. The working tree was not
modified.

**Verdict: PARTIAL — 4 findings.** M1-M14 land in the code almost without exception: one
DIVERGENT item position, one spec sentence that contradicts itself, and three verification
rows whose test cannot fail for the reason the row names — one of them the live defect M9 was
written to fix.

---

## 1. Decisions, sentence by sentence

### M1 — the bar is a render of `COMMAND_SPECS` — **LANDED**

| Sentence | Code | Verdict |
|---|---|---|
| one top-level menu per `CommandCategory` in `CATEGORY_ORDER` | `shaderbox/menus.py:57-72` | PRESENT |
| one item per spec with `in_menu`, in table order | `menus.py:58-63` (list comp preserves `COMMAND_SPECS` order) | PRESENT |
| label = `spec.label`; hint = `chord_to_str(...)`, `""` for chord `0` | `menus.py:44-46` | PRESENT |
| a `separator_before` spec draws `imgui.separator()` first | `menus.py:69-70` | PRESENT |
| `CommandSpec` gains `in_menu: bool = True`, `separator_before: bool = False` | `commands.py:95`, `:97` | PRESENT |
| `in_menu=False` on `FOCUS_TAB_DOCUMENT`, `_UNIFORMS`, `_RENDER`, `_SHARE`, `CYCLE_CODE_TAB` | `commands.py:173` (CYCLE_CODE_TAB), `:197`, `:204`, `:211`, `:218` — five, exactly those | PRESENT |
| `separator_before=True` on `QUIT` | `commands.py:121` | PRESENT |
| the right-aligned `project <name>` stays | `menus.py:74-85` | PRESENT |
| every `imgui.menu_item` in `ui.py::_draw_menu_bar` goes | `ui.py` now has zero `menu_item` hits; `ui.py:555` calls `menus.draw_menu_bar(app)`; `_hint` gone | PRESENT |

### M2 — `menus.py` holds the App-facing primitives — **LANDED**

`menu_enabled`'s three branches, `menus.py:27-38`: EDITOR → `app.active_tab is not None`
(`:33-34`); COPILOT → `app.is_copilot_open` (`:35-36`); GLOBAL → `True` (`:37`, the fallthrough
`return True`). `command_menu_item` returns whether it fired (`menus.py:41-49`).
`ui_primitives.py` imports neither `App` nor `commands` (verified by grep); `menus.py` imports
both. `widgets/cheatsheet.py::_is_active` is not reused.

### M3 — every button that opens a command's surface takes `command_label` — **LANDED**

All five named sites:

- `tabs/document.py:477` `standard_button(command_label(CommandId.ADD_PASS))` (was `add pass`)
- `tabs/document.py:480` `standard_button(command_label(CommandId.IMPORT_PASSES))` (was `import...`)
- `widgets/copilot_chat.py:166` `action_label=command_label(CommandId.OPEN_SETTINGS)` (was `Open Settings`)
- `exporters/telegram.py:411` `command_label(CommandId.OPEN_SETTINGS)` (was `Set up token`)
- `exporters/youtube.py:393` `command_label(CommandId.OPEN_SETTINGS)` (was `Set up credentials`)

`OPEN_PROJECTS`'s label is `"Projects"` (`commands.py:114`) and `OPEN_SETTINGS`'s is
`"Settings"` (`:245`) — no trailing `...`. The remaining `...` literals in `shaderbox/` are
progress cues (`"Rendering..."`, `"Uploading..."`) and the `Open other...` file-picker button,
none of which is a spec label. `help_content.py`'s shortcuts snippet already generated from
the table, so it followed.

### M4 — object menus, one item set per kind — **LANDED except one item's position**

| Set | Function | Landed order | Spec order | Verdict |
|---|---|---|---|---|
| pass | `pass_list.pass_menu_items` (`:66-86`) | Open shader · Settings · ─ · Leave group · ─ · Delete ▸ `Delete pass <name>` | same | PRESENT |
| group box | `pass_graph._box_menu_items` (`:1479-1488`) | Open · Dissolve | same | PRESENT |
| document | `document_grid.document_menu_items` (`:43-52`) | Open · Open folder · ─ · Delete ▸ `Move to trash` | same | PRESENT |
| canvas | `pass_graph._canvas_menu` (`:875-885`) | Add pass · Import passes (both `command_menu_item`) · ─ · Fit · Arrange | same | PRESENT |
| pass, node only | `pass_graph._node_menu` (`:1499`) | `Group` drawn **after the whole set** | "`Group...` **between** `Settings` and `Leave group`" | **DIVERGENT — finding 1** |

`Open` is `app.select_document` (`document_grid.py:47`); `Open folder` is
`app.open_document_dir(document_id)` (`:49`); `open_current_document_dir` is now its
one-argument caller (`app.py:1928-1929`). The disabled-tile probe the spec asked for was run
and reported in the commit body — a `begin_popup_context_item` inside `begin_disabled` never
opens, so no Python-side gate was added. That is the spec's own instruction followed.

### M5 — a destructive menu verb confirms through a submenu — **LANDED**

`ui_primitives.confirm_menu_item(label, confirm_label, enabled=True)` at `:501-515`: a
`begin_menu(label, enabled)` holding one `menu_item_simple(confirm_label)` pushed in
`COLOR.STATE_ERROR`, returning the inner click (`:513`). Four callers, all four the spec names:
`pass_list.py:84`, `document_grid.py:51`, `lib_picker/tree.py:242` (`Delete` ▸ `Move to .trash`),
`lib_picker/tree.py:160` (`Delete directory` ▸ `Move every file to .trash`).

Both armed fields and both arm verbs are gone: a repo-wide grep for `file_delete_armed`,
`dir_delete_armed`, `arm_file_delete`, `arm_dir_delete` returns zero hits in `shaderbox/`. The
only `push_style_color` left in `tree.py` is the favorite star's at `:266` — which §10.3
explicitly keeps (see §3, deviation 3). `delete_file` / `delete_dir` keep their trash move and
toast. Projects' delete and Settings' library reset keep their armed `danger_button` rows
(`projects.py:130`, `settings.py`).

The `enabled` third parameter is an addition to the spec's two-argument signature, and a
correct one: it is what carries the last-pass gate M10 removes from the call site.

### M6 — the document tile carries no button — **LANDED**

`document_grid.py:39` passes `deletable=False`; `:35` `armed=False`.
`App.document_delete_armed` and `set_document_delete_armed` are gone (zero grep hits), and so
are the grid's three result branches (`delete_armed` / `delete_confirmed` / `delete_cancelled`,
all removed in the diff along with the deferred `id_to_delete` pop). `document_grid.py:70`
draws `imgui.text_colored(COLOR.FG_DIM, "Right-click for actions")` beside New document.
`preview_cell` keeps `armed` / `deletable` / `cell_delete_confirm` / `close_cross_button` for
the sticker grid — its signature is untouched in the diff.

### M7 — one name-input row — **LANDED**

`InlineInput` is gone from `editor_types.py` (zero grep hits) and lives at
`ui_primitives.py:518-545`. `InputRowResult` at `:548-563`, `name_input_row` at `:566-604`.
The row reads `is_item_deactivated_after_edit()` on the line after the input (`:594`), before
the `x` (`:598`); `committed` is Enter-or-deactivate and cleared when cancelled (`:602-603`);
`needs_focus` is consumed once (`:587-590`).

All four callers the spec names: `lib_picker/tree.py:205` (new-file/dir), `tree.py:251`
(rename), `popups/projects.py:166` (new name), `pass_graph.py:1514` (the group prompt).
`GraphViewState.group_input: InlineInput` at `graph_state.py:86`; `group_prompt` and
`group_name` are gone (zero grep hits). The group prompt stays a `begin_popup`
(`pass_graph.py:1512`) and a blank name still refuses to commit (`:1518-1520`).

### M8 — modal chrome, one shape, gated — **LANDED**

`settings.py`'s `is_keep_opened` → `keep_open` (`settings.py:188`). The revert modal returns
`keep_open` through the new `_draw_revert_body` (`copilot_chat.py:192-209`) whose caller nulls
`copilot_revert_target` (`:188`) and calls `close_current_popup` (`:189`). Every action row is
preceded by a `SPACE.MD` dummy: `examples.py:80`, `help.py:74`, `lib_picker/__init__.py:114`,
`emoji_picker.py:73`, `projects.py:97`, `settings.py:186`, `pass_settings.py:95`/`:136`,
`copilot_chat.py:202`. Projects' `Close` joins the left-packed verb row (`projects.py:133`,
after Delete, all on `same_line`).

One small spec-text drift, not a defect: the spec says "the revert modal's `SPACE.XS` becomes
`MD`". The action-row spacer that changed was the `SM` at the old `:186`; an `XS` remains at
`copilot_chat.py:197` as a mid-body spacer, which the rule (spacer *above the action row*)
does not cover.

`tests/test_modal_chrome.py` implements the gate as specified: the domain is enumerated from
`PopupState` minus `CLOSED` plus `copilot_revert`, each resolved through a `_BODIES` table in
the test (`:43-57`), a member with no row failing at `:123`. The three AST clauses — a bound-and-
returned `keep_open` (`:143`), a last `standard_button` labelled `Close`/`Cancel` (`:166`), a
`SPACE.MD` dummy above the row (`:186`) — are all present, plus a fourth clause the spec did
not ask for (see §3, deviation 3). Every break it names is recorded in the commit body.

### M9 — `close_popup()` is the one close funnel — **LANDED in code**

`app.py:1026-1063`, branch by branch against M9's list:

| M9 says | `app.py` | Verdict |
|---|---|---|
| `PASS_SETTINGS` → `close_pass_settings()` | `:1036-1038` | PRESENT |
| `IMPORT_PASSES` → `close_import_passes()` | `:1039-1041` | PRESENT |
| `EMOJI_PICKER` → new `close_emoji_picker()` (nulls target, clears query, `CLOSED`) | `:1042-1044`; verb at `:1303-1306` does all three | PRESENT |
| `SETTINGS` → `apply_editor_settings()` + `CLOSED` | `:1045-1048` | PRESENT |
| `PROJECTS` → returns `False` while `projects_input_owns_esc()`, else `CLOSED` | `:1049-1054` | PRESENT (plus `reset_projects_state()`, a correct addition — the modal's own state would otherwise survive the close) |
| `SHADER_LIB_PICKER` → returns `False` while `inline_input_owns_esc`, else `CLOSED` | `:1055-1059` | PRESENT |
| `EXAMPLES`, `HELP` → `CLOSED` | `:1060-1062` | PRESENT |
| returns `True` when it closed | every closing branch returns `True`; `:1035` and `:1063` return `False` | PRESENT |

`hotkeys._handle_escape` (`:353-378`): the four carve-outs are gone, replaced by
`app.close_popup()` at `:371`; the `was_settings_open` latch is gone (zero grep hits in the
repo); the `rebinding_command` early return is kept (`:357-358`); the revert modal keeps its own
branch first (`:368-369`). `inline_input_owns_esc` moved onto `ShaderLibFileManager`
(`shader_lib/file_ops.py:59`), which is how `App` reaches it without importing a popup.

Every modal's own close path funnels: `emoji_picker.py:22`, `help.py:31`, `examples.py:64`,
`projects.py:46`, `lib_picker/__init__.py:43`, `settings.py:68` all call `app.close_popup()`.
`import_passes.py:51` and `pass_settings.py:63` call their own dedicated verbs
(`close_import_passes` / `close_pass_settings`) — which is what `close_popup` itself dispatches
to, so the per-state cleanup is identical. The chrome gate's fourth clause
(`test_no_modal_writes_the_closed_state_by_hand`, `:212-229`) pins that no draw function writes
`PopupState.CLOSED` by hand.

The code is right. Three of the M9 verification row's scenarios are not — see findings 2-4.

### M10 — the disabled-item double gate goes — **LANDED**

`pass_list.py:84-86`: `if confirm_menu_item("Delete", f"Delete pass {name}", enabled=len(document.passes) > 1):`.
The `and deletable` is gone and so is the comment that justified it (both visible as deletions
in the diff). The new two-line comment above it states why the last pass cannot go — a fact
about the code as it is, which the repo's comment rule allows; it does not narrate the fix.

`.claude/skills/imgui-ui/SKILL.md` §7.4 (`:393-397`) carries the measured fact. Its opening and
closing clauses are the spec's verbatim; the parenthetical is expanded ("the same item, the
same click path, `enabled` the only difference") and one sentence is added about
`begin_menu`/`begin_disabled` not opening. Substance preserved, wording richer — an improvement,
not a divergence worth a finding.

### M11 — copy inside budget — **LANDED (with the deviation §3.1 records)**

Both tooltips read exactly `needs a shader caret` (`popups/help.py:85`,
`popups/lib_picker/__init__.py:131`). `COPILOT_LIMIT_ROWS` (`copilot/config.py:204`) holds ten
rows, every `hint` at 4-6 words — all inside the ≤ 8-word cap, verified row by row. Each row's
long `explanation` is printed by the new Help copilot section
(`help_content.py:99-100`, registered at `:240`).

### M12 — the hint rule and the destructive rule are conventions — **LANDED**

`conventions.md:567-571` (the confirm rule) and `:577-579` (the right-click-hint rule), both in
"we decided X; revisit if Y" form, both ending in an explicit `Revisit if …` clause. Each is a
fuller rewrite of the spec's suggested sentence rather than a transcription — correct, since
the spec's text was a sketch of the decision, not a string to copy. `SKILL.md` §7.4 is amended
to match.

### M13 — prose-budget domain preserved — **LANDED**

`tests/test_ui_prose_budget.py::test_every_command_label_is_within_the_menu_budget` (`:848-864`)
scores every `CommandSpec.label` at ≤ 4 words plus a clause-joiner check. The break the spec
names (a five-word label in `COMMAND_SPECS`) is recorded as tried and restored in the commit
body.

### M14 — stays as it is — **LANDED**

- Both exporters: the diff touches nothing but the gate label and its import
  (`telegram.py` `"Set up token"` → `command_label(...)`, `youtube.py` `"Set up credentials"` →
  the same). Their panel bodies, `unconnected_gate` bodies, pack forms and the hand-rolled red
  child at `telegram.py:484-485` are untouched.
- `toggle_copilot` (`app.py:885`) and `toggle_copilot_open` (`app.py:897`) both exist.
- The lib tree's favorite star keeps its own `push_style_color` (`tree.py:266`).
- The Examples selection still persists across opens (`app_state.selected_example_id`, not
  popup-local state).
- `preview_cell`'s signature is unchanged.

---

## 2. The Verification table, row by row

19 rows. 13 have a test that fails for the reason the row names. 3 do not (findings 2-4), 2 are
weaker in kind than the row claims but still decisive on their substance, 1 is a review row with
no test by design.

| # | Guarantee | Test | Decisive? |
|---|---|---|---|
| 1 | M1: the bar is the table | `test_menus.py::test_the_bar_draws_exactly_the_tables_in_menu_specs` — `assert drawn == expected` (`:160`) | Yes. Real frames, spy on submitted labels. The commit records the break (a hand-added `imgui.menu_item("Projects...")` fired it). |
| 2 | M1: `in_menu` | `::test_the_view_focus_verbs_are_the_ones_out_of_the_menu` (`:204`) + `::test_no_item_is_drawn_for_an_out_of_menu_spec` (`:210`) | Yes |
| 3 | M2: `menu_enabled` | `::test_menu_enabled_reads_the_scope_not_the_editor_focus` — all four clauses incl. GLOBAL-behind-a-modal (`:237-251`) | Yes |
| 4 | M2: layering | `::test_ui_primitives_stays_app_free_and_menus_does_not` — `assert "from shaderbox.app import" not in primitives` (`:258`) | Yes |
| 5 | M3: one spelling | `::test_no_command_label_is_respelled_with_an_ellipsis` (`:289`) + `::test_no_button_respells_a_command_label_in_another_case` (`:314`) | Yes. Both breaks recorded in the commit. |
| 6 | M4: the pass set is shared | `::test_the_pass_set_is_the_same_on_the_strip_and_the_node` — `assert node == [*shared, "Group"]` (`:359`) | Yes. Two real frames, full ordered label lists compared. |
| 7 | M4: a document's menu | `::test_a_documents_menu_carries_open_open_folder_and_delete` (`:364-380`) | **No — finding 2.** Labels are frame-verified; the verb wiring is not. |
| 8 | M4: the canvas menu's commands | `::test_the_canvas_menu_fires_the_add_pass_command` (`:404-427`) | Partly. The row's Kind says "frame-driven"; the test is AST over `_canvas_menu` plus a direct `command_callbacks[ADD_PASS]()` call. Both halves are real checks (the AST clause does catch a call-site that bypasses `command_menu_item`) but no frame draws or clicks the menu. |
| 9 | M5: the submenu | `::test_the_confirm_submenu_needs_the_inner_click` — `assert confirmed == []` (`:501`), `assert confirmed == [1]` (`:513`) | Yes — **confirmed by mutation**: rewriting `confirm_menu_item` to a plain `menu_item_simple` turns this test red. Clicks the real outer-label rect, then the real inner rect. |
| 10 | M5: the lib tree's arm is gone | `::test_the_lib_trees_armed_delete_is_gone` (`:539-557`) + `::test_the_lib_delete_still_trashes_and_toasts` (`:559`) | Yes for the fields/verbs and the trash+toast. The `push_style_color` clause is asserted on `COLOR.STATE_ERROR` instead (the deviation §3.3 records, correctly). |
| 11 | M6: no button on the document tile | `::test_the_document_tile_draws_no_delete_cross` — `assert crosses == []` (`:592`) + `::test_the_armed_document_delete_state_is_gone` (`:595`) | Yes. Real frame, spy on `close_cross_button`, zero calls. |
| 12 | M7: `name_input_row` commits on deactivate | `::test_the_name_row_commits_on_a_click_away_and_cancels_on_the_x` (`:616-672`), `::test_a_focus_move_with_no_edit_commits_nothing` (`:674`), `::test_the_group_prompt_holds_one_inline_input` (`:697`) | Mostly. Deactivate-commit and `x`-cancel are frame-driven and decisive. The row's Enter clause has no dedicated assertion (it shares the `committed` expression, so it is implicitly covered) and the group prompt's blank-refuses clause is checked by AST rather than driven with an empty buffer. |
| 13 | M7: the promotion | `::test_inline_input_lives_in_the_primitives_now` (`:610-611`) | Yes |
| 14 | M8: the chrome gate | `test_modal_chrome.py`, whole file, parametrized over the enumerated domain | Yes. Four breaks tried and restored, each named in the commit with the red message it produced. |
| 15 | **M9: Esc through the funnel** | — | **No — findings 3 and 4.** |
| 16 | M9: the dispatch gate | `test_modal_chrome.py::test_the_close_funnel_covers_every_popup_state` — `assert not missing` (`:259`) | Yes. The break (a GHOST member) is recorded as tried. |
| 17 | M10: no double gate | `::test_the_pass_delete_has_no_python_side_double_gate` — `assert "and deletable" not in source` (`:741`) + `::test_the_last_pass_cannot_be_deleted_through_the_menu` (`:746-752`) | The pure half, yes. The frame half is not frame-driven: the second test calls `app.session.delete_pass` directly, proving the session guard, not that a click on the disabled item is refused. A separate test, `::test_a_disabled_confirm_submenu_never_opens` (`:516`), does drive a frame and covers the substance the row was after. |
| 18 | M11: budget | `test_ui_prose_budget.py::test_every_copilot_limit_hint_is_within_the_help_budget` (`:866-883`), plus the file's own site walk | Yes, per the deviation §3.1 — a direct table assertion, not an allowlist deletion. The break is recorded. |
| 19 | M13: labels in the budget | `test_ui_prose_budget.py::test_every_command_label_is_within_the_menu_budget` (`:848`) | Yes. The break is recorded. |
| — | M14: nothing else moved | none (Kind: "review") | By design. Verified manually in §1 above. |

---

## 3. Findings

### Finding 1 — `Group` sits after the pass set, not between `Settings` and `Leave group` (DIVERGENT, minor)

The spec, M4's table (`05_menus_spec.md:76`):

> | pass, node only | `pass_graph._node_menu` after the set | `Group...` **between `Settings` and `Leave group`** (092 D14: it seeds `view.selection`) | graph node |

The design pass agrees (`04_menus_inventory.md:625`):

> | pass, node only | + Group... (**before Leave group**) | graph node |

The code (`shaderbox/widgets/pass_graph.py:1498-1499`):

```python
                pass_menu_items(app, document_id, node.name)
                if node.kind == "pass" and imgui.menu_item_simple("Group"):
```

`pass_menu_items` draws the whole set — including `Leave group` and `Delete` — so `Group` lands
*last*, below the destructive verb, not between `Settings` and `Leave group`.

The spec's own row is self-contradictory: its Function column says "after the set" and its Items
column says "between `Settings` and `Leave group`". Only one of the two can be built without
splitting `pass_menu_items` into two halves, which no decision authorizes. The implementer read
the Function column; the design pass and the Items column say the other thing. The gate
(`test_menus.py:359`) pins the landed order, so this cannot now drift — but it pins the reading
the spec's *other* half contradicts.

The label also lost its ellipsis (`Group...` → `Group`), which is M3's rule applied
consistently — correct, and worth naming only because the spec still spells it with one in three
places.

**Fix (one of two, the maintainer's call):** either move the `Group` item inside
`pass_menu_items` behind a `node_only: bool = False` parameter and update the gate, or amend
M4's Items column to "`Group` after the set" and strike the "between `Settings` and `Leave
group`" clause plus `04_menus_inventory.md:625`'s "(before Leave group)". The second is the
smaller change and matches what shipped; a destructive verb is arguably better not sandwiched
by a creation verb, which is an argument for the landed order.

### Finding 2 — the document menu's verb wiring is asserted by a vacuous line (test defect)

Verification row 7 says:

> right-click a grid tile: the popup opens with `Open` / `Open folder` / `Delete`; `Open` calls
> `select_document(id)`; **`Open folder` calls `open_document_dir(id)` (spied)**

The test (`tests/test_menus.py:376-380`):

```python
    # The verbs each menu item calls, exercised at the seam the item fires.
    app.select_document(document_id)
    app.open_document_dir(document_id)
    assert opened == [document_id] and revealed == [document_id]
```

Both methods are monkeypatched to append to `opened` / `revealed`, and the test then calls the
*monkeypatched attributes directly*. The assertion therefore proves that `monkeypatch.setattr`
works — nothing about `document_menu_items`. The comment "exercised at the seam the item fires"
describes something the code does not do.

**Mutation confirming it** (worktree at `f045eef`, `document_grid.py:49`
`app.open_document_dir(document_id)` replaced with `pass`):

```
1 passed, 26 deselected in 0.87s
```

`Open folder` can be wired to nothing at all and the test that claims to check it stays green.
The label half of the test is real (`assert spy.labels == ["Open", "Open folder", "Delete"]` at
`:375`, frame-driven) — only the wiring half is vacuous.

**Fix:** drive the two items the way M5's test drives its submenu — `test_the_confirm_submenu_needs_the_inner_click`
(`:435-514`) already has the rig: open the popup in a frame, capture the item rects via
`get_item_rect_*`, click `Open folder`'s rect, assert `revealed == [document_id]`. Then break it
(the `pass` mutation above) and confirm it goes red before believing it.

### Finding 3 — M9's emoji-picker scenario has no behavioral test, and the live defect can be reintroduced silently (test defect)

Verification row 15's first scenario:

> open the emoji picker with a target set; press Esc: **`emoji_pick_target is None`** and
> `popup_state == CLOSED`. … Break: restore the direct `CLOSED` write in `_handle_escape` — the
> target dangles

This is the defect M9 exists to fix; the commit body calls the dangling `emoji_pick_target` "the
live defect". No test asserts it. A repo-wide grep for `emoji_pick_target` in `tests/` returns
three hits, all of them prose inside `test_modal_chrome.py` docstrings (`:215`, `:243`) or a
name in a set literal (`:205`) — not one assertion on the field's value.

**Mutation confirming it** (`app.py:1303-1306`, `close_emoji_picker` reduced to just
`self.popup_state = PopupState.CLOSED`, so the target and the query both dangle):

```
tests/test_menus.py tests/test_modal_chrome.py tests/test_import_dialog.py tests/test_project_management.py
120 passed in 6.48s
```

The exact bug the decision was written to eliminate can be put straight back with the whole
relevant suite green. What the chrome gate pins is *structural* — that a branch exists and that
no draw function writes `CLOSED` by hand — which is a different claim from "the branch does the
cleanup".

**Fix:** a headless test in the shape `test_project_management.py::test_escape_is_owned_by_an_open_name_input_at_the_close_funnel`
already uses — no frames needed, so the font-atlas constraint that test documents does not
apply: set `app.emoji_pick_target` to a value, `app.popup_state = PopupState.EMOJI_PICKER`, call
`app.close_popup()`, assert `app.emoji_pick_target is None` and `app.emoji_picker_query == ""`
and `popup_state == CLOSED`. Then run the mutation above and confirm it fails.

### Finding 4 — M9's Settings-apply scenario has no test either (test defect)

Same row, third scenario:

> Open Settings, change a field, Esc: **`apply_editor_settings` called once (spied)**

`apply_editor_settings` appears nowhere in `tests/` (the one grep hit,
`test_keymap_disjoint.py:286`, is prose about `_apply_editor_settings_to`, a different function).

**Mutation confirming it** (`app.py:1045-1048`, the `SETTINGS` branch's
`self.apply_editor_settings()` line deleted):

```
tests/test_menus.py tests/test_modal_chrome.py tests/test_import_dialog.py
tests/test_project_management.py tests/test_keymap_disjoint.py
134 passed in 6.54s
```

Esc on the Settings modal can silently stop applying the user's edits and the suite says nothing.
This is the same class as finding 3: `close_popup`'s *dispatch* is gated, its *per-branch
cleanup* is not, and those two branches are the only ones whose cleanup is more than a state
write.

**Fix:** the same shape — spy `apply_editor_settings`, set `popup_state = SETTINGS`, call
`close_popup()`, assert one call. Since findings 3 and 4 are one class, the honest fix is one
parametrized test over the branches that carry cleanup (`EMOJI_PICKER`, `SETTINGS`,
`PASS_SETTINGS`, `IMPORT_PASSES`, `PROJECTS`), each asserting the cleanup it owns actually ran —
which also closes the gap the chrome gate's structural clause leaves open for every future
branch.

---

## 4. The implementer's declared deviations

Three where the spec did not hold as written, four beyond it. Each judged against §10's
reasoning, the conventions, whether the commit records it, and whether the spec text is now
stale.

### 4.1 M11's allowlist rows were partly in `_UNMEASURABLE` — **justified, recorded, spec stale**

The spec (`:160-161`): "the **twelve** `_OVER_BUDGET` rows in `tests/test_ui_prose_budget.py`
for them are deleted, so the gate holds them from now on."

`_OVER_BUDGET` held nine entries in total, of which two were the tooltips; the copilot hints were
in `_UNMEASURABLE` because they are reached through a loop variable, which no call-site AST walk
can score however short they become. Deleting an `_UNMEASURABLE` row would not have created a
gate — it would have created nothing. The implementer instead wrote a direct assertion over the
table (`test_every_copilot_limit_hint_is_within_the_help_budget`), which is the treatment
`_FORMATS` already gets in the same file for the identical reason.

Justified by the spec's own goal sentence ("the copy inside the modals is inside the budget the
repo already enforces") — the goal was enforcement, and only the direct assertion delivers it.
Recorded in the commit body at length, with the red message from the break. **Spec text now
stale:** `05_menus_spec.md:160-161`'s clause "the twelve `_OVER_BUDGET` rows in
`tests/test_ui_prose_budget.py` for them are deleted" should read "the two `_OVER_BUDGET`
tooltip rows are deleted; the copilot hints, being `_UNMEASURABLE` through a loop variable, get
a direct assertion over `COPILOT_LIMIT_ROWS` as `_FORMATS` does".

### 4.2 No Help copilot section existed — **justified, recorded, spec stale**

The spec (`:160`) sends the long form to "the Help panel's copilot section". There was none. The
implementer built one (`help_content.py:99-100`, registered `:240`), reading each row's
`explanation`. Creating the destination a spec assumes is the only way to satisfy the sentence;
the alternative (drop the long form) loses content the spec wanted kept. Recorded in the commit
body. **Spec text now stale:** `:160`'s "appended to the Help panel's copilot section" should say
the section is new.

### 4.3 M5's "no `push_style_color`" would have deleted the favorite star's — **justified, recorded, spec stale**

Verification row 10 asks for "`tree.py` contains no `push_style_color`". §10.3 explicitly keeps
the lib tree's favorite star, whose colouring *is* a `push_style_color`
(`tree.py:266`) — the rulebook's own carve-out. The two clauses of the same spec contradict; the
implementer kept the star (the decision) and narrowed the assertion to `COLOR.STATE_ERROR`
(`test_menus.py:554`), which is what actually names the armed-delete red the decision retires.

Justified: §10.3 is a decision, the verification row is a mechanism for checking it, and when a
mechanism would undo the decision the mechanism is what is wrong. Recorded in the commit body.
**Spec text now stale:** `05_menus_spec.md:235`'s "`tree.py` contains no `push_style_color`"
should read "`tree.py` contains no `COLOR.STATE_ERROR` push (the favorite star's own push stays,
§10.3)".

### 4.4 The disabled-tile probe — **the spec asked for it; correct**

M4 (`:83-85`) instructs: "verify in the impl, and if a disabled tile's right-click still opens
it, gate the items in Python as the strip does". The implementer probed
(`begin_popup_context_item` inside `begin_disabled` never opens; a positive control opened at
frames 6-7) and added no gate. That is the instruction followed, with the measurement reported —
not a deviation at all. Recorded in the commit body with the frame numbers.

### 4.5 `InputRowResult.focused` — **justified, recorded, spec stale**

The spec (`:113-115`) declares `InputRowResult(committed: bool, cancelled: bool)` — two fields.
The shipped dataclass has three (`ui_primitives.py:561-563`). The reason is a real defect the
move exposed: the `x` button is submitted *after* the input, so an `is_item_focused()` at the
call site answers for the button, not the field. `popups/projects.py:167`
(`app.projects_input_focused = result.focused`) is what keeps the Projects modal's outer Enter
from firing while the user types a name — without the field carrying its own focus, the caller
cannot ask the right question.

Justified by M7's own purpose (one row that owns the whole interaction, callers stop
hand-rolling the queries) and by the conventions' "a convention collision means the design is
wrong — fix the design" — the two-field shape would have forced a wrong query at the call site.
Recorded in the commit body, with the failure it prevents spelled out. **Spec text now stale:**
`:113`'s `InputRowResult(committed: bool, cancelled: bool)` should carry the third field and its
reason.

### 4.6 Projects and the lib picker routed through `close_popup`; a fourth chrome clause — **justified, recorded, no staleness**

M9's list has `PROJECTS` and `SHADER_LIB_PICKER` branches, so routing those two modals through
the funnel is the decision, not an extension. What is beyond the spec is the chrome gate's fourth
clause (`test_no_modal_writes_the_closed_state_by_hand`, `test_modal_chrome.py:212-229`): M8
names three AST clauses, and this is a fourth.

It earns its place — it is the only clause that pins M9's "one funnel" claim at the draw sites,
and it is the gate that would have caught the emoji picker's original defect. The commit records
the break tried (`app.popup_state = PopupState.CLOSED` put back in `emoji_picker`, red, restored).
The conventions' "ship the check that prevents the recurrence in the same commit as the sweep"
is exactly this. **Spec text:** M8's clause list (`:129-131`) could gain the fourth clause, but
"a gate grew a clause" is not staleness in the sense that misleads a reader — optional.

Note that the two branches this clause forced (`import_passes.py:51`, `pass_settings.py:63`) call
their own close verbs rather than `close_popup()`. That is correct — those verbs are what
`close_popup` dispatches to — and the clause passes because neither writes `CLOSED` by hand.

### 4.7 `inline_input_owns_esc` moved to `ShaderLibFileManager` — **justified, recorded, spec stale**

M9 writes the predicate as `inline_input_owns_esc(app)` — a free function taking `App`. Shipped
as `ShaderLibFileManager.inline_input_owns_esc()` (`shader_lib/file_ops.py:59`), called at
`app.py:1056` as `self.shader_lib_files.inline_input_owns_esc()`.

Justified and in fact required by the layering M2 pins: `App` calling a free function that lives
in `popups/lib_picker/` would make `app.py` import a popup, inverting the dependency the whole
decision rests on. The state the predicate reads is the file manager's own, so the method belongs
on it. Recorded in the commit body ("so App can reach it without importing a popup"). **Spec text
now stale:** `05_menus_spec.md:140`'s `inline_input_owns_esc(app)` should read
`app.shader_lib_files.inline_input_owns_esc()`.

---

## 5. Working tree

`git status --porcelain` is empty at `f045eef` on `dev`; `projects/dev/` is untouched by the
commit (`git show --stat f045eef -- projects/dev` is empty) and carries no working-tree diff.

One transient: at the start of this review an untracked `tests/test_probe_group_tmp.py` was
present (mtime 16:38, four minutes after the commit — a leftover probe from the implementation
session, asking whether a click-away dismissal leaves `group_input` open). It is no longer on
disk and the tree is clean. Worth a glance from the maintainer only to confirm nothing of value
was in it; the question it asked is covered by
`test_menus.py::test_the_group_prompt_holds_one_inline_input`.

The probe worktree this review created was removed; `git worktree list` shows only the main
checkout.

---

## 6. False trails — checked and clean

Named so a later reader does not re-walk them.

- **`Group`'s missing ellipsis** is not a second finding. M3's "no trailing ellipsis anywhere"
  is a rule about spec labels; `Group` is a menu item, and dropping the `...` applies the same
  spelling discipline. Only the position (finding 1) is a divergence.
- **`import_passes.py` and `pass_settings.py` not calling `app.close_popup()`** reads like an M9
  hole. It is not: they call `close_import_passes` / `close_pass_settings`, which are exactly
  what `close_popup` dispatches to for their states, so the cleanup is identical by construction
  and the chrome gate's hand-written-close clause covers the failure mode.
- **`reset_projects_state()` in the PROJECTS close branch** is an addition to M9's text, not a
  divergence — the spec says "else `CLOSED`" about the *decision* to close; leaving the modal's
  own state behind would be the defect.
- **The revert modal's surviving `SPACE.XS`** (`copilot_chat.py:197`) is a mid-body spacer, not
  the action-row spacer M8 governs. The action row's is `MD` at `:202`.
- **The new comment above `pass_list.py`'s Delete** is not the comment M10 deletes. M10 removes
  the one that justified the Python double gate; the new two lines state why the last pass cannot
  be deleted — a fact about the code as it stands, which the repo's comment rule permits.
- **`"Open other..."` in `projects.py:124`** matches no spec label, so the ellipsis rule does not
  reach it. The gate agrees (`test_no_command_label_is_respelled_with_an_ellipsis` compares
  against `COMMAND_SPECS` labels only).
- **`test_ui_prose_budget.py`'s four skips** are the file's long-standing exemptions
  (`help_marker`, `markdown_text`, `modal_window`, `parse_markdown_lines`), not new ones this
  commit introduced.
- **M5's submenu test** was the row most likely to be theatre (the commit body says its first
  draft built its own submenu inline and stayed green under the break). The shipped version is
  genuinely decisive — mutating `confirm_menu_item` into a plain `menu_item_simple` turns it red,
  confirmed by running it.
- **M4's "the pass set is shared" row** likewise: it really opens both menus in two frames and
  compares full ordered label lists.

---

## 7. Verdict

**PARTIAL**, four findings:

1. `Group` sits after the pass set, where the spec's Items column and the design pass both say
   between `Settings` and `Leave group` — and the spec's own Function column says "after the
   set". One of the two spec sentences must be struck, or the code moved. (M4)
2. The document menu's `Open` / `Open folder` wiring is asserted by calling the monkeypatched
   methods directly; severing `open_document_dir` from the menu item leaves the test green.
   (Verification row 7)
3. M9's emoji-picker cleanup has no behavioral test; the dangling `emoji_pick_target` — the live
   defect the decision names — can be reintroduced with the suite green. (Verification row 15)
4. M9's Settings-apply cleanup has no test; deleting `apply_editor_settings()` from the branch
   leaves the suite green. (Verification row 15)

Findings 3 and 4 are one class — `close_popup`'s dispatch is gated structurally, its per-branch
cleanup is not — and one parametrized test closes both plus every future branch.

Everything else lands. M1, M2, M3, M5, M6, M7, M8, M10, M11, M12, M13 and M14 are present
sentence for sentence, including all five `in_menu=False` specs, the one `separator_before`,
`menu_enabled`'s three branches, `command_label` at every named button, four of the five M4 item
sets in exact order, both deleted armed fields and both arm verbs, `InlineInput`'s move,
`name_input_row`'s four callers, `GraphViewState.group_input`, every spacer, Projects' Close
position, `close_popup`'s eight branches with both `False` returns and the Settings apply, the
`was_settings_open` latch's deletion, the `rebinding_command` return's survival, `pass_list.py`'s
un-double-gated Delete, §7.4's wording, both tooltips, ten copilot hints inside eight words, the
new Help section, the two conventions bullets, and M14's stays list untouched.

All seven declared deviations are justified by the spec's own reasoning and the conventions, and
all seven are recorded in the commit body. Five leave a spec sentence stale; the sentences are
named in §4 and, the spec being a living doc, want the update in the next wave.
