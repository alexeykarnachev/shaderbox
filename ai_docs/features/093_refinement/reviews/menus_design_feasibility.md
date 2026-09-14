# Design-pass feasibility — §10 of `04_menus_inventory.md`, read against the code

Scope: proposals P1-P11 in `## 10. Design pass`. The anchor is the code, not the inventory:
every claim below was established by reading the named file or by a probe against the installed
`imgui_bundle` (1.92.801). Where a probe settled a question the inventory answered from memory,
the probe wins and the transcript is given.

**Overall verdict: FEASIBLE WITH CHANGES.** Eight of eleven proposals are implementable roughly
as written. Three are NOT AS WRITTEN — P9 rests on a library fact the build contradicts, P4
inverts the recoverability of the two deletes it re-sorts, and P8 would drop a focus call it does
not mention. Two more (P1, P3) are feasible but land a larger diff than §10.2 implies, and both
touch surfaces §10.3 declares out of scope.

---

## 0. Library facts, settled by probe

Four questions decide several proposals. Each was driven through a real imgui frame on the
`app` fixture's live context (the `_imgui_frame` shape from `tests/test_pass_verbs.py:392`),
with synthetic mouse events and a positive control in the same run.

### 0.1 `menu_item_simple(enabled=False)` does NOT register a click on this build

The same item, the same click path, the only difference the `enabled` argument:

```
SAME_ITEM enabled_clicked=True disabled_clicked=False
```

A separate run drove a click onto an item wrapped in `begin_disabled(True)`:

```
BEGIN_DISABLED_MENU_ITEM_CLICKED False
```

Both are negative. The positive control (`enabled=True`, same geometry, same event sequence)
returned `True`, so the harness demonstrably delivers the click.

This contradicts:

- `shaderbox/widgets/pass_list.py:72-74` — the comment "`menu_item_simple` can still register a
  click while disabled on this imgui-bundle build (/imgui-ui §7.4)", and the `and deletable`
  double-gate it justifies;
- `§9.4`'s whole table, which sorts two sites by whether they carry the Python-side guard;
- `§10.1` item 9 ("The `enabled=` footgun is guarded in one of its two sites");
- **P9's entire stated purpose** — "the wrapper returns `False` when `enabled` is `False`
  whatever imgui reported, so the build's disabled-click footgun is dead at every site and the
  lib tree's Insert at caret is fixed by the move".

`popups/lib_picker/tree.py:359`'s missing guard is therefore not a live defect. It may have been
real on an older bundle; on the pinned one it is not. **Re-run the probe before acting on P9** —
if it reproduces, P9 loses its justification and `pass_list.py:72-74`'s comment plus `§9.4` need
correcting rather than generalizing.

### 0.2 A `begin_menu` under `begin_disabled` does not open at all

```
BEGIN_MENU_UNDER_BEGIN_DISABLED_opened [False, False, False, False]
```

Relevant to P1: per-ITEM disabling works (an item draws greyed and refuses the click), but a
whole category submenu wrapped in `begin_disabled` becomes unopenable — the user cannot even see
what is in it. P1 says "disabled when the spec's scope is not active", which is per-item and
fine; do not extend it to the category menu.

### 0.3 `begin_popup_context_item(None)` after `begin_tab_item` DOES anchor to that tab

P2's editor-tab menu is feasible. With the tab bar drawn under the app's real flags
(`reorderable | fitting_policy_scroll | tab_list_popup_button | draw_selected_overline`) and a
right-click aimed at the third, non-selected tab:

```
SELECTED_STATE        {'Alpha_sel': True,  'Beta_sel': False, 'Gamma_sel': False}
NONSELECTED_GAMMA_CTX {'Alpha_ctx': False, 'Beta_ctx': False, 'Gamma_ctx': True}
```

The menu opened for Gamma alone — the `None` id anchors to the tab item just submitted, and a
non-selected tab gets its own menu. The call must sit **outside** the `if opened:` branch, since
only the selected tab enters it.

**But the right-click also selects the tab:**

```
SELECTION_AFTER_RCLICK {'Alpha_sel': False, 'Beta_sel': False, 'Gamma_sel': True}
```

In `tabs/code.py:122-125` that read-back fires `app.set_active_tab(i)`, so right-clicking a tab
to reach its menu would also switch the editor to it — a behaviour change P2 does not mention.
Either accept it (defensible: right-click-to-focus is common) or suppress the read-back on a
frame where the context menu opened. Say which in the spec.

*Method note.* An earlier round of this probe reported the opposite — `is_item_hovered()` after
`begin_tab_item` always `False`, no menu ever opening. That was a harness artifact: the rig
window was auto-sized, so its extent did not cover the aimed point and `is_window_hovered()` was
`False` too. Pinning the window with `set_next_window_pos`/`set_next_window_size` before the tab
bar made both the hover and the menu work. The negative result was not a library fact.

### 0.4 `imgui.menu_item` vs `menu_item_simple` return shapes

From `.venv/lib/python3.12/site-packages/imgui_bundle/imgui/__init__.pyi:2289-2298`:

```python
def menu_item_simple(label: str, shortcut: Optional[str] = None,
                     selected: bool = False, enabled: bool = True) -> bool
def menu_item(label: str, shortcut: str, p_selected: bool,
              enabled: bool = True) -> Tuple[bool, bool]
```

`menu_item` returns a 2-tuple (activated, toggled) and takes `shortcut` as a REQUIRED positional;
`menu_item_simple` returns a bare bool and defaults `shortcut` to `None`. The menu bar
(`ui.py:720-751`) uses the tuple form for its chord hints and indexes `[0]`; every context menu
uses the simple form. P1 replaces the seven tuple-form calls; P9 wraps the simple form. A single
primitive covering both needs the chord hint to be optional, i.e. it forwards to `menu_item` when
a hint exists and `menu_item_simple` otherwise, or always to `menu_item` with `""`.

---

## P1 — one verb registry drives every menu

**Verdict: FEASIBLE WITH CHANGES.**

### Files and functions

- `shaderbox/ui.py:714-763` `_draw_menu_bar` — rewritten entirely. All seven `imgui.menu_item`
  calls (`:720, :724, :729, :734, :742, :748, :750`) go. The right-aligned project name
  (`:752-763`) stays as P1 says.
- `shaderbox/ui.py:711-712` `_hint` — already `chord_to_str(app.effective_bindings[command_id])`;
  becomes the chord source inside the new primitive.
- `shaderbox/commands.py:83-92` `CommandSpec` — gains `separator_before: bool` (P1) and, per
  §10.4, probably `in_menu: bool = True`.
- `shaderbox/commands.py:107-238` `COMMAND_SPECS` — its declaration order becomes a UI fact.
- `shaderbox/app.py:1016` `effective_bindings` and `app.command_callbacks` (`app.py:464-467`) —
  the primitive reads both.

### Where `command_menu_item` and `command_label` must live

**Not in `ui_primitives.py`.** That module is `App`-free by design — grep for `App` in
`shaderbox/ui_primitives.py` returns only two docstring mentions, no import. `command_menu_item(app,
command_id)` takes an `App`, so importing it there would create the cycle `conventions.md`'s
three-layer bullet (`## Design decisions`, "Three-layer UI architecture") says forced `App` into
its own module in the first place.

It belongs in `ui.py` (which already imports `App`, `CommandId` and `chord_to_str`) or a new
`shaderbox/menus.py` that imports `App`. This matters for P9: its proposed gate ("a raw
`menu_item_simple` outside `ui_primitives.py` fails") would then have to allow the new module too.

`command_label(command_id)` needs only `SPEC_BY_ID` (`commands.py:241`), so it could live in
`commands.py` as a pure function — that is the better home, since `commands.py` is already
declared a leaf ("imports `imgui` only, never `App`", `commands.py:3`).

### What `command_menu_item` needs

- `app.effective_bindings[command_id]` → `chord_to_str` for the hint (`ui.py:711-712` is the
  existing shape).
- The scope-activity test. **The function to name is `widgets/cheatsheet.py:17-22` `_is_active`**
  — it is the only existing predicate that answers "is this scope active right now":

  ```python
  def _is_active(scope: CommandScope, app: App) -> bool:
      if scope == CommandScope.EDITOR:   return app.editor_focused
      if scope == CommandScope.COPILOT:  return app.copilot_focused
      return not app.any_popup_open()
  ```

  `hotkeys.py:331-338` `spec_eligible` is the dispatcher's version and is NOT interchangeable: it
  also rejects `chord == 0` and `chord in app.editor_consumed_chords`, both of which are about a
  key press, not about whether a verb is available to a mouse. A menu item for `IMPORT_PASSES`
  (default chord `0`, `commands.py:222`) must still be clickable. Use `_is_active`, promoted out
  of `cheatsheet.py` to wherever the primitive lands, with `cheatsheet.py` importing it back.
- `app.command_callbacks[spec.id]()` to fire.

### Hidden couplings

1. **`_is_active` returns `False` for every GLOBAL command while a popup is open** (`not
   app.any_popup_open()`). A menu bar drawn while a modal is open would grey out every File and
   Tools item. Today that is moot — the menu bar is unreachable behind a modal — but the same
   primitive is proposed for context menus, and the pass/graph menus are reachable in states the
   cheatsheet is not. Decide what "active" means for a menu item explicitly rather than
   inheriting the cheatsheet's answer.
2. **`tests/test_command_registry_coverage.py:20-24`** asserts every bound spec's `label` appears
   in `help_content._shortcuts_section().snippet`. P1 makes spec labels the UI's labels; any label
   edit (e.g. "Settings" → "Settings...") must land in the help snippet in the same commit.
3. **The prose-budget gate does not see `command_menu_item`.** `tests/test_ui_prose_budget.py`
   scores by call name and parameter name (`_IMGUI_ROWS:117` gives `menu_item_simple(label=)` a
   4-word budget; `_derived_rows()` reflects over `ui_primitives` only). A primitive taking
   `command_id` and no label string is invisible to it, so every menu label would leave the gate's
   domain at once. If P1 lands, the gate needs a row that scores `CommandSpec.label` from
   `commands.py` directly, or the pass silently removes menu labels from the budget it claims to
   hold (P10's own concern, inverted).
4. `imgui.menu_item`'s `shortcut` is a required positional (0.4), so a spec with chord `0` needs
   `""` rather than `None` through that path.

### Blast radius

`ui.py`, `commands.py`, `widgets/cheatsheet.py` (if `_is_active` moves), plus the new module and
`tests/test_ui_prose_budget.py`. Four to five modules. The §10.4 "every command in the menu bar"
call is the real sizing question: with `in_menu` defaulting `True`, the bar gains 25 items across
five menus, including the four Focus-tab chords and Cycle code tab.

---

## P2 — object menus, one item set per object kind

**Verdict: FEASIBLE WITH CHANGES.**

### Files and functions

- `shaderbox/widgets/pass_list.py:62-79` `pass_menu_items` — gains `Group...` and two separators.
- `shaderbox/widgets/pass_graph.py:1478-1496` `_node_menu` — its `Group...` branch
  (`:1491-1495`) moves into `pass_menu_items`; the `box` branch (`:1484-1488`) becomes the
  `group box` item set.
- `shaderbox/widgets/document_grid.py:73-104` — the grid tile gains
  `begin_popup_context_item`; new item set (Open / Open folder / ─ / Delete).
- `shaderbox/tabs/code.py:86-133` `_draw_tab_row` — new tab menu (see 0.3).
- `shaderbox/widgets/pass_graph.py:872-884` `_canvas_menu` — stays, two items re-render through
  `command_menu_item`.

### Hidden couplings

1. **`Group...` moving into the shared set needs `GraphViewState`.** The node branch writes
   `view.selection`, `view.group_prompt` and `view.group_name` (`pass_graph.py:1492-1495`), and
   `pass_menu_items` currently takes `(app, document_id, name)` with no view. The strip has no
   `GraphViewState` at all. Either the signature gains an optional view, or the group-prompt state
   moves onto `App` — the second is cleaner and is what P5's "row at the top of the canvas child"
   would want anyway. Say which.
2. **`tests/test_graph_view.py:172-183`** `test_the_widget_makes_no_session_write_of_its_own`
   greps both `pass_graph.py` and `pass_list.py` for `set_pass_group(` and friends, asserting
   absence. A `Group...` implementation that calls `app.group_selection` (as today,
   `pass_graph.py:1522`) stays clean; one that reaches `app.session.set_pass_group` fails. The
   gate already covers the moved item — good.
3. **The document grid's menu must respect `copilot_turn_active`.** The grid wraps its tiles in
   `begin_disabled(app.copilot_turn_active)` (`document_grid.py:70`). Per 0.2, a context menu
   opened on a disabled item does not open at all, so the freeze is inherited for free — but
   verify, since `begin_popup_context_item` is not a `menu_item`.
4. **`Open folder` on a document** has no existing verb. `app.reveal_shader_lib_file_in_manager`
   (`tree.py:152`) is lib-specific; `util.open_in_file_manager` (imported at `app.py:125`) is the
   generic one. A new `App.reveal_document_in_manager` is needed.
5. **The editor-tab menu's right-click-selects side effect** (0.3).
6. `Close others` is a new verb with no `App` method; `app.close_tab(i)` (`app.py`, called at
   `code.py:132`) closes one by index, and closing N tabs by index requires iterating backwards or
   a new bulk verb.

### Blast radius

`pass_list.py`, `pass_graph.py`, `document_grid.py`, `tabs/code.py`, `app.py`, plus tests. Six
modules. Proportionate — this is the pass's core UX claim.

---

## P3 — a tile carries no button

**Verdict: FEASIBLE WITH CHANGES.** The proposal undercounts `preview_cell`'s callers.

### Files and functions

- `shaderbox/ui_primitives.py:1195-1345` `preview_cell` — loses `armed`, `deletable`, and the
  `if selected and armed:` / `if deletable:` branches (`:1324-1344`).
- `shaderbox/ui_primitives.py:1149-1154` `PreviewCellResult` — loses three of its four fields.
- `shaderbox/ui_primitives.py:1117-1146` `cell_delete_confirm`, `:932-949` `close_cross_button` —
  P3 says both go "when their last caller goes". Verified: each has exactly one caller, both
  inside `preview_cell` (`:1325` and `:1341`). Both can go cleanly.
- `shaderbox/app.py:487` `document_delete_armed`, `:1378-1379` `set_document_delete_armed`,
  and `:791-792` (the cleanup inside the document-deleted handler) — all three go.
- `shaderbox/widgets/document_grid.py:80-99` — the `armed=` argument and the three result branches.

### The caller P3 does not name

§10.2 names two `preview_cell` callers for the delete machinery (document grid, sticker grid) and
one for `overlay` (sticker). There are **five call sites in `shaderbox/`**:

| site | `deletable` | `armed` | `overlay` |
|---|---|---|---|
| `widgets/pass_list.py:135-152` | `False` explicitly | `False` | — |
| `widgets/uniform.py:194-201` `_draw_texture_preview` | default `True` | `False` | — |
| `widgets/document_grid.py:22-32` | default `True` | caller-driven | — |
| `exporters/telegram.py:672-679` (empty slot) | default `True` | `False` | — |
| `exporters/telegram.py:691-706` (sticker) | default `True` | caller-driven | yes |

`widgets/uniform.py:194` — the sampler row's texture thumbnail — takes the default
`deletable=True`. It passes `selected=False`, so the `elif selected:` branch never runs and no ✕
is drawn today. Removing the parameter is therefore safe for it, but the inventory's claim that
the machinery has two delete callers is wrong: it has two *arming* callers and four sites
inheriting `deletable=True`.

### Does `overlay` survive removing the delete machinery?

Yes, with one structural caveat. `overlay` is drawn at `ui_primitives.py:1334-1337`, inside the
`elif selected:` branch — i.e. it renders only when the cell is selected AND NOT armed. The
sticker caller (`telegram.py:691-706`) passes `selected=(idx == selected_index and not
in_flight)` and `armed=(sticker_delete_armed == slot.file_id)`.

If P3 option A keeps `armed` for the sticker caller, nothing changes. If option B removes it, the
`if selected and armed:` branch goes and `elif selected:` becomes `if selected:` — `overlay`
survives and in fact becomes *more* available (it no longer disappears while armed). Either way
the emoji glyph keeps working. The `x_side` local (`:1332`) is computed in the same branch and is
passed to `overlay(x_side)`; it must survive the delete removal.

### Gates

- **`tests/test_import_dialog.py:123-124`** calls `preview_cell("bordered", 168.0, None, (0, 0),
  False, False, footer="a")` — six positional arguments, the last two being `selected` and
  `armed`. Dropping the `armed` parameter shifts every later positional and **fails this test as
  written**. It needs updating in the same commit.
- **`tests/test_pass_verbs.py:535-549`** spies on `pass_list.preview_cell` and asserts over the
  captured kwargs (`chips`, `chip_font`, absence of `sublines`). It does not read `deletable`, so
  it survives — but `pass_list.py:151`'s `deletable=False` argument must go with the parameter.
- No test asserts a `PreviewCellResult` delete field.

### Blast radius

`ui_primitives.py`, `document_grid.py`, `pass_list.py`, `app.py`, `tests/test_import_dialog.py`,
plus `exporters/telegram.py` under option B. Five to six modules. Option B touches an exporter
§10.3 declares out of scope; the recommendation (A) does not — that call is correctly placed in
§10.4.

---

## P4 — one destructive-verb rule

**Verdict: NOT AS WRITTEN.** The two factual claims that sort the verbs are both wrong, and they
point in opposite directions from the proposal.

### What the code actually does

**Pass delete — P4 is right.** `project_session.py:980-999` `delete_pass` pops the entry, calls
`release()`, `drop_feedback`, `forget_pass_sources`, rewrites the graph, saves. **No `unlink`, no
`shutil.move`** — the `.frag.glsl` file stays on disk. `widgets/pass_list.py:160-...` `_delete_pass`
additionally captures the source path first to tear down the editor session and tab. So "a
deleted pass leaves its shader file on disk and drops the entry (`delete_pass` unlinks nothing)"
is accurate.

**Document delete — P4's recoverability claim is wrong.** `project_session.py:462-495`
`_delete_document_unguarded` does move the directory to trash (`:487-492`), so the *data* is
recoverable. But the Recover affordance P4 cites is not general:
`app.py:810-825` `recover_deleted_document` takes a **`Message`** and reads `msg.recover`
(`copilot/state.py:43-50` `RecoverInfo`), which is constructed only in the copilot's own delete
path (`copilot/backend.py:1177-1188`). A user deleting a document from the grid produces no
`Message` and therefore **no Recover card**. §10.2's "a deleted document moves to the project
trash (recoverable, and the copilot's Recover card already reads it)" conflates data
recoverability with a user-facing undo that does not exist on this path. Removing the confirm
leaves the user with a one-click, irreversible-from-the-UI document deletion.

**Lib file / directory delete — P4's claim is wrong in the other direction.** §10.2 says "a lib
file / directory delete is a real filesystem delete", and upgrades it to a modal confirm on that
basis. `shader_lib/file_ops.py:223-250` `delete_file` moves the file into `shader_lib_trash_dir()`
with a numeric-suffix collision rule and **already toasts** (`msg = f"Deleted {path.name}"`).
`:252-...` `delete_dir` moves every file individually into the same trash, refusing symlinked or
escaping dirs, explicitly "so nothing is silently rmtree'd". Both are trash moves, both toast.

So P4's own rule — "recoverable verbs fire at once and toast; irreversible ones confirm in a
modal" — applied to the real facts inverts its conclusions: the lib deletes are the ones that
already satisfy the rule, and the document delete is the one with no recovery path.

### What is needed

Re-derive the taxonomy from the code. A defensible reading of the same rule:

- pass delete → fire + toast (P4 is right; no file is lost);
- lib file/dir delete → fire + toast, dropping the armed-label flip (the trash + existing toast
  already satisfy the rule — no `confirm_modal` needed);
- document delete → keep a confirm, OR add a real Recover affordance (a notification with an
  undo, or a general recover path not keyed to a copilot `Message`) before removing it.

If `confirm_modal(title, line, verb)` is still wanted, it would generalize
`widgets/copilot_chat.py:177-203` `_draw_revert_modal`, which today closes inline
(`:194-200`) rather than returning `keep_open` — P6 changes exactly that, so the two proposals are
ordered: P6 before P4.

The hand-rolled red pushes P4 retires are `tree.py:165-167` and `:279-281`, both real and both
flagged in §9.5. Retiring them is independently worthwhile.

### Blast radius

`popups/lib_picker/tree.py`, `shader_lib/file_ops.py` (if the arming state goes),
`widgets/pass_list.py`, `widgets/copilot_chat.py`, `ui_primitives.py`. Five modules.

---

## P5 — one name-prompt shape

**Verdict: FEASIBLE.** One claim is already true today, and the layering question resolves clean.

### Files and functions

- `shaderbox/widgets/pass_graph.py:1499-1531` `_group_prompt` — the hand-rolled `begin_popup`
  goes; becomes an inline row.
- `shaderbox/popups/projects.py:155-190` `_draw_name_input` — gains commit-on-deactivate.
- `shaderbox/popups/lib_picker/tree.py:198-...` `_draw_inline_new_input`, `:290-...`
  `_draw_file_rename_input` — the reference shape (`wants_commit = changed or
  imgui.is_item_deactivated_after_edit()`, `tree.py:224` and `:304`).
- `shaderbox/editor_types.py:109-131` `InlineInput` → `ui_primitives.py`.

### Already true

§10.2 says "the Projects new-name row **becomes** `InlineInput`-backed". It already is:
`popups/projects.py:19` imports `InlineInput` from `editor_types`, `app.py:540` holds
`projects_new_input: InlineInput`, and `_draw_name_input` takes `state: InlineInput`. What is
actually missing is commit-on-deactivate — `projects.py:168` commits on Enter and `:180` on the
button, and clicking away discards (§9.1 gets this right). State the proposal as the behaviour
change it is, not as an adoption.

### Import cycle: none

`ui_primitives.py` imports `profiling`, `render_plan`, `theme` and third-party only — no `App`,
no `editor_types`. `InlineInput` needs `pathlib.Path` alone. The promotion is clean in both
directions.

One second-order note: `editor_types.py` is currently imgui-free (verified — importing it leaves
`imgui_bundle` out of `sys.modules`), and `shader_lib/file_ops.py:17` imports `InlineInput` from
it while documenting itself as drivable by "a non-UI caller". Moving `InlineInput` to
`ui_primitives` would make `file_ops` import imgui. **This is already the case** — `file_ops`
imports `shaderbox.theme` (`:21`), which pulls `imgui_bundle` (verified). So the promotion costs
nothing that is not already paid. Worth one line in the spec so a future reader does not
re-litigate it.

The conventions bullet authorizing this is `ai_docs/conventions.md ## Design decisions`,
`InlineInput` entry: "Revisit if a second multi-inline-input surface lands (promote `InlineInput`
to `ui_primitives.py`)." P5's trigger reading is correct.

### The group prompt's state

`_group_prompt` reads and writes `view.group_prompt`, `view.group_name`, `view.selection` on
`GraphViewState`. An inline row at the top of the canvas child can keep them there. But see P2's
coupling 1: if `Group...` moves to the shared `pass_menu_items`, the strip needs the same state
and has no view — decide the two together.

### Blast radius

`ui_primitives.py`, `editor_types.py`, `popups/projects.py`, `widgets/pass_graph.py`,
`popups/lib_picker/tree.py`, `shader_lib/file_ops.py` (import line), `app.py`. Seven modules, but
most are one-line import edits.

---

## P6 — modal chrome to one shape

**Verdict: FEASIBLE WITH CHANGES.** The proposed gate as worded would not hold what it names.

### Files and functions

- `shaderbox/popups/settings.py:190-193` — `is_keep_opened` → `keep_open`. Purely local; the name
  appears only in `_draw_body` (`:190, :192, :193`).
- `shaderbox/widgets/copilot_chat.py:177-203` `_draw_revert_modal` — restructured to return
  `keep_open`, with the caller reading it. Note this modal's state is `app.copilot_revert_target`,
  outside `PopupState` (`app.py:1036-1040`), so the caller's close branch nulls that field rather
  than setting `popup_state`.
- Spacer additions: `popups/examples.py` (before its action row, §9.1 says none at `:79`),
  `popups/help.py:76` (none), `popups/lib_picker/__init__.py:123` (plain `imgui.spacing()`).
- `shaderbox/popups/projects.py:130-132` — the right-anchored Close
  (`imgui.same_line(imgui.get_content_region_avail().x - float(SIZE.BTN_SM_W))`) joins the
  left-packed row.

### The proposed gate

`tests/test_modal_chrome.py` is to walk `popups/*.py` and the revert modal's AST and fail a body
that "does not end in a `standard_button("Close")` / `("Cancel")` row preceded by
`imgui.dummy((0, SPACE.MD))`, or that names its return anything but `keep_open`."

Three problems:

1. **`popups/lib_picker/` is a package**, not a `popups/*.py` file. `draw_lib_picker` lives at
   `popups/lib_picker/__init__.py:45` with the body split across `search.py`, `tree.py`,
   `preview.py`, `filtering.py`. A glob over `popups/*.py` misses it entirely — the
   narrowed-domain failure the repo's own conventions call the most expensive bug family.
   Enumerate the domain from `PopupState` (plus the revert modal) rather than from a glob.
2. **"names its return anything but `keep_open`"** is not checkable from a `return` statement
   alone — `pass_settings._draw_body` and others return a local; the AST check must follow the
   returned `Name` to its assignment. Simpler and stronger: assert the function has a local
   binding named `keep_open` that is returned.
3. **The spacer check must tolerate the two spellings already in use.** `settings.py:188` uses
   `imgui.dummy((0.0, SPACE.MD))` (a tuple), `copilot_chat.py:193` uses
   `imgui.dummy(imgui.ImVec2(0, float(SPACE.XS)))` — different constructor, different token.
   Normalize or the gate passes on syntax rather than on layout.

### The break that proves it

Three, one per clause, each to be tried and restored, and named in the commit:

- rename `keep_open` back to `is_keep_opened` in one modal → the naming clause must fail;
- delete the `imgui.dummy((0, SPACE.MD))` line above one modal's action row → the spacer clause
  must fail;
- add a new `PopupState` member with a modal whose body returns `ok` and has no Close row → the
  domain must include it and fail (this is the one that catches problem 1 — try it with the lib
  picker specifically, whose file layout the glob misses).

### Blast radius

Six `popups/` modules, `widgets/copilot_chat.py`, one new test. Small diff, and the one proposal
whose entire value is mechanical consistency — the gate is the deliverable.

---

## P7 — Esc closes through the modal's own funnel

**Verdict: FEASIBLE WITH CHANGES.** One existing test fails as written; the proposed gate needs a
different assertion than the one named.

### Files and functions

- `shaderbox/app.py` — new `close_popup()` dispatching on `popup_state`. The per-modal verbs it
  would call: `close_pass_settings` (`app.py:1081-1110`), `close_import_passes`, a new
  `close_emoji_picker` nulling `emoji_pick_target` (today nulled only at
  `popups/emoji_picker.py:23` and overwritten at `app.py:1264`).
- `shaderbox/hotkeys.py:354-399` `_handle_escape` — loses the four carve-outs at `:375-390`.
- The modal draw functions' own close branches — `examples.py:63-65`, `help.py:29-31`,
  `settings.py:66-72`, `emoji_picker.py:21-24`, etc.

### The coupling P7 must preserve

`_handle_escape` is not only a dispatcher. Two behaviours live in it that a plain `close_popup()`
does not carry:

1. **`was_settings_open` / `apply_editor_settings()`** (`hotkeys.py:366` and `:398-399`). Settings'
   editor settings apply at the close funnel; `draw_settings` (`settings.py:66-72`) does its own
   `apply_editor_settings()` on the button path. If `close_popup()` becomes the one funnel, the
   apply moves into it — and then the latch at `:366` is dead code that must go, or the apply runs
   twice.
2. **The two "leave it open" carve-outs are not closes at all.** `PROJECTS` +
   `projects_input_owns_esc()` (`:381-385`) and `SHADER_LIB_PICKER` + `inline_input_owns_esc(app)`
   (`:386-389`) do NOT close the modal — they deliberately do nothing so the inline input's own
   cancel runs later in the frame. A `close_popup()` that dispatches on `popup_state` alone would
   close them. Either `close_popup()` returns a bool ("I declined") and `_handle_escape` keeps the
   ownership predicates, or the predicates move inside it. Both are fine; the proposal reads as
   if all four carve-outs are the same kind of thing, and two of them are not.
3. `app.rebinding_command is not None` (`:357-358`) returns before anything — unrelated to the
   dispatch, keep it where it is.

### Gates

- **`tests/test_import_dialog.py:63-67` fails as written.** It asserts
  `"PopupState.IMPORT_PASSES" in source and "close_import_passes()" in source` against
  **`shaderbox/hotkeys.py`**. P7 moves both strings to `app.py`. The test must be repointed at
  `close_popup`'s module in the same commit — and, since it is a source-substring test, this is
  the moment to make it structural instead.
- **P7's proposed gate** — "`tests/test_project_management.py`'s enum-count test gains the
  dispatch table (an enum member with no branch fails)". That test is
  `test_every_popup_state_has_a_draw_call` (`:636-667`); it parses `ui.py`'s AST for calls to
  names imported from `shaderbox.popups` and asserts `len(called) == len(PopupState) - 1`. It
  already carries a documented falsifier ("add a `PopupState` member and no draw call") and a
  note about an earlier substring version that a comment satisfied. Extending it means parsing
  `close_popup`'s body for a branch per member. **To be breakable, the new assertion must fail
  when a `PopupState` member has a draw call but no close branch** — the existing count assertion
  would still pass in that state, so a naive extension adds nothing. The break to demonstrate:
  add a member, wire its draw, omit its close branch; the new clause must go red while the old
  one stays green.

### Blast radius

`app.py`, `hotkeys.py`, six `popups/` modules, two tests. Eight modules — large for what it buys
(one dangling callback fixed), but it removes a class of forget-to-add-a-carve-out bug, which is
the §10.1-7 argument and a fair one.

---

## P8 — one copilot toggle

**Verdict: NOT AS WRITTEN.** The two methods differ by more than focus-awareness.

`app.py:888-898` `toggle_copilot`:

```python
if not self.is_copilot_open:
    self.is_copilot_open = True
    self.focus_copilot()
elif self.copilot_focused:
    self.is_copilot_open = False
else:
    self.focus_copilot()
```

`app.py:900-905` `toggle_copilot_open`:

```python
self.is_copilot_open = not self.is_copilot_open
if self.is_copilot_open:
    self.focus_copilot()
```

P8 says "`toggle_copilot_open` folds into `toggle_copilot(focus_aware: bool)`; the chip passes
`False`". But `toggle_copilot_open` **does** call `focus_copilot()` — on the open edge. A
`focus_aware=False` branch that skips focusing would change the chip's behaviour: clicking
"Copilot" to open the chat would leave it unfocused, and the user would have to click again into
the input.

The real difference is the CLOSE edge: the chord closes only when the chat is focused (so an
unfocused chat gets focused instead of closed), the chip closes unconditionally. A correct fold
is a parameter naming that:

```python
def toggle_copilot(self, focus_closes: bool = True) -> None:
    if not self.is_copilot_open:
        self.is_copilot_open = True
        self.focus_copilot()
    elif self.copilot_focused or not focus_closes:
        self.is_copilot_open = False
    else:
        self.focus_copilot()
```

with the chip passing `focus_closes=False`. Verify against the two comments at `app.py:889-891`
and `:901-903`, which document the intent ("a click already moved focus off the chat, so the
focus-aware `toggle_copilot` would blink it back open") — that comment is about the close edge,
confirming the reading.

### Files and functions

`shaderbox/app.py:888-905`; `shaderbox/ui.py:783` (the chip's call site); the callback wiring for
`CommandId.TOGGLE_COPILOT`. Two modules. Small and worth doing, but the spec's parameter is
mis-named and mis-described.

---

## P9 — `menu_item` through one primitive

**Verdict: NOT AS WRITTEN.** Its stated justification is contradicted by the build (0.1), and the
proposed gate would fail on existing labels.

### The justification is gone

P9 exists to kill "the build's disabled-click footgun". The probe (0.1) shows
`menu_item_simple(enabled=False)` returns `False` under a real click on imgui-bundle 1.92.801, and
so does an item under `begin_disabled`. The lib tree's unguarded `Insert at caret`
(`tree.py:359`) is not firing with no editor target on this build.

A wrapper may still be wanted for the `danger=True` colour push — that consolidates
`tree.py:165-167` and `:279-281`, which is real (§9.5). But it is a styling proposal, not a
correctness one, and it should be argued as such. If P4 retires the armed-delete labels those two
pushes go anyway, and the `danger` parameter has no caller left.

### The gate would fail on existing labels

P9 proposes `tests/test_button_tiers.py` fail on a raw `menu_item_simple` outside
`ui_primitives.py`. Two problems.

**First, the prose budget.** `tests/test_ui_prose_budget.py` scores `menu_item_simple(label=)` at
**4 words** via an explicit `_IMGUI_ROWS` entry (`:117`), with a written reason ("A menu item is
an action phrase like a button, and may carry a qualifier the button column has no room for
(`Reveal in file manager`): four words (092)"). A new `ui_primitives.menu_item(label, ...)` is
picked up automatically by `_derived_rows()` (`:79-103`), which maps the parameter name `label`
to **2 words** from `_PARAMETER_BUDGETS` (`:41`) — `_BUTTON_LABEL_BUDGET`'s 3-word override
applies only to names in `_BUTTON_TIERS` (`:99-100`), which the new primitive would not be in.

Five current menu labels exceed 2 words:

```
3  'New file here'
4  'Reveal in file manager'
3  'Insert at caret'
4  'Open file at declaration'
3  'Confirm delete (recursive)' / 'Delete directory (recursive)'
```

So P9 as written turns the prose gate red on five sites. The fix is one line — add `menu_item` to
a 4-word set alongside `_BUTTON_TIERS`' 3-word one — but it must be in the same commit, and the
`_IMGUI_ROWS` entry for `menu_item_simple` must stay (the primitive still calls it internally,
and `ui_primitives.py` is excluded from the button-tier walk but not from the prose walk — check
which).

**Second, `command_menu_item` cannot live in `ui_primitives.py`** (see P1). A gate phrased as
"outside `ui_primitives.py`" would fail the very primitive P1 introduces. Phrase the allowlist by
module set, not by one filename.

### The break that proves the gate

Add a raw `imgui.menu_item_simple("X")` to `widgets/pass_list.py` and confirm the new clause goes
red; then restore. Name it in the commit. Note the existing `_NOT_A_VERB` machinery
(`test_button_tiers.py:25-54`) is keyed on `"button" in call.attr` (`:67`) — a `menu_item_simple`
call does not match that filter at all, so this is a new walk, not a widened one. Give it its own
allowlist with its own `test_every_listed_exception_still_exists` sibling, matching the existing
pattern.

### Blast radius

`ui_primitives.py`, every module drawing a menu (5), `tests/test_button_tiers.py`,
`tests/test_ui_prose_budget.py`. Seven modules, for a correctness benefit the build says is not
there.

---

## P10 — prose inside budget

**Verdict: FEASIBLE.** One mechanical caution.

### Files and functions

- `shaderbox/popups/help.py:84-87` — the 15-word tooltip. Its `_OVER_BUDGET` row is
  `("shaderbox/popups/help.py", "_draw_body", 15)` (`test_ui_prose_budget.py:300-304`).
- `shaderbox/popups/lib_picker/__init__.py:140-142` — the 11-word tooltip; row at `:305-309`.
- `shaderbox/popups/settings.py:225-320` `_COPILOT_LIMITS` — the ten help markers.
- `shaderbox/help_content.py` — where the long forms move.

### The caution

`_OVER_BUDGET` keys on `(module, function, word_count)` and `_EXEMPT = set(_OVER_BUDGET)`
(`:582`), so an exemption matches only at its recorded count. Shortening a string to a
*different* over-budget count silently loses its exemption and goes red — which is the intended
design ("a rewrite that changes the string changes this line too", `:286-287`). P10's proposed
replacements are within budget ("needs a shader caret" = 3 ≤ 5), so the rows can simply be
deleted. Verify each rewrite actually lands under budget before deleting its row, or the suite
goes red with a confusing message.

The `_COPILOT_LIMITS` markers are scored through `help_marker`'s explicit 8-word row
(`_IMGUI_ROWS:116`), and `help_marker` is in `_NOT_UI_COPY` for the domain test (`:75`) — the
markers are measured at their call sites, so cutting the strings is sufficient; no test structure
changes.

Moving the long form into the Help panel is sound: `markdown_text` is exempted as documentation
(`_NOT_UI_COPY:73`), and `tests/test_help_content.py` exists to hold that surface.

### Blast radius

Three `popups/` modules, `help_content.py`, one test. Small, and the only proposal with no
structural risk.

---

## P11 — discoverability, one line where it earns it

**Verdict: FEASIBLE.**

`shaderbox/widgets/document_grid.py:45-58` — the caption goes beside `New document`. The existing
hint to copy is `popups/lib_picker/__init__.py:116`,
`imgui.text_colored(COLOR.FG_DIM, "Right-click for actions")` — 3 words, and `text_colored` is
scored at 4 when the colour is `COLOR.FG_DIM` (`_IMGUI_ROWS:115`), so it fits.

The `/imgui-ui` §7.4 amendment is a skill edit, outside the repo. Per the repo's own rule that a
project must not reference the harness, the amended rule needs a home in
`ai_docs/conventions.md` if it is to bind ShaderBox — or the amendment is purely to the skill and
the spec should say so.

One module. No gate proposed and none needed.

---

## False trails

Claims in §10.1 / §10.2 the code contradicts, or that would send an implementer the wrong way.

1. **The `enabled=` footgun does not exist on this build.** `widgets/pass_list.py:72-74`'s comment,
   §9.4's table, §10.1-9, and P9's justification all rest on it. Probe transcript in 0.1. The
   `and deletable` double-gate at `pass_list.py:75` is harmless but unnecessary; the missing guard
   at `popups/lib_picker/tree.py:359` is not a defect.

2. **The document delete has no user-facing Recover.** §10.2 P4: "a deleted document moves to the
   project trash (recoverable, and the copilot's Recover card already reads it)".
   `app.py:810-825` `recover_deleted_document` requires a `Message` carrying `RecoverInfo`
   (`copilot/state.py:43-50`), built only at `copilot/backend.py:1177-1188`. A grid delete
   produces none.

3. **The lib delete is not "a real filesystem delete".** §10.2 P4. `shader_lib/file_ops.py:223-250`
   and `:252-...` both move into `shader_lib_trash_dir()`, and both already push a toast.

4. **`preview_cell` has five call sites, not three.** §10.2 P3 and §5's tooltip row name the
   document grid, sticker grid and pass tile. `widgets/uniform.py:194-201` is a fourth (sampler
   thumbnail, inherits `deletable=True`), `exporters/telegram.py:672-679` a fifth (the empty
   sticker slot).

5. **The Projects new-name row is already `InlineInput`-backed.** §10.2 P5 says it "becomes" one.
   `popups/projects.py:19` imports it; `app.py:540` holds it. The missing piece is
   commit-on-deactivate alone.

6. **`toggle_copilot_open` is not merely non-focus-aware.** §10.1-8 and P8. It calls
   `focus_copilot()` on the open edge (`app.py:903-905`); the difference from the chord is on the
   CLOSE edge.

7. **`command_menu_item` cannot live in `ui_primitives.py`.** Implied by P9's gate wording ("a raw
   `menu_item_simple` outside `ui_primitives.py`"). The module is `App`-free; importing `App`
   there recreates the cycle `conventions.md`'s three-layer bullet exists to prevent.

8. **A `popups/*.py` glob does not reach the lib picker.** P6's gate. `draw_lib_picker` is
   `popups/lib_picker/__init__.py:45`.

9. **`tests/test_import_dialog.py:63-67` asserts against `hotkeys.py` by substring.** P7 moves the
   strings it looks for. Not a claim in §10, but an unnamed casualty of it.

10. **§10.1-2's "seven commands have no mouse-reachable home".** The listed seven are Save,
    Next/Previous pass, Cycle code tab and the four Focus-tab chords — that is eight items for a
    count of seven. `NEXT_PASS`/`PREV_PASS` are also credited in §7.1 with reaching
    `app.choose_output`, which the pass tile and graph node both do by click. Recount before the
    number lands in a spec.

---

## Per-proposal verdict table

| | Verdict | Modules | Notes |
|---|---|---|---|
| P1 | FEASIBLE WITH CHANGES | 4-5 | primitive cannot live in `ui_primitives`; name `cheatsheet._is_active` as the scope test; prose gate loses menu labels |
| P2 | FEASIBLE WITH CHANGES | 6 | right-click selects the tab (0.3); `Group...` needs view state on the strip; `Open folder`/`Close others` are new verbs |
| P3 | FEASIBLE WITH CHANGES | 5-6 | five callers not three; `test_import_dialog.py:123-124` passes `armed` positionally |
| P4 | NOT AS WRITTEN | 5 | both recoverability claims wrong, in opposite directions |
| P5 | FEASIBLE | 7 (mostly imports) | no cycle; Projects row already `InlineInput` |
| P6 | FEASIBLE WITH CHANGES | 8 | gate must not glob `popups/*.py`; three named breaks |
| P7 | FEASIBLE WITH CHANGES | 8 | two carve-outs are "decline", not "close"; one existing test fails |
| P8 | NOT AS WRITTEN | 2 | the difference is the close edge, not focus-awareness |
| P9 | NOT AS WRITTEN | 7 | footgun absent on this build; gate reds five labels |
| P10 | FEASIBLE | 5 | verify each rewrite lands under budget before deleting its row |
| P11 | FEASIBLE | 1 | amended §7.4 rule needs a home inside the repo |

## Out-of-scope touches

§10.3 declares the exporters and the copilot out of scope. Two proposals cross that line:

- **P3 option B** touches `exporters/telegram.py` — correctly surfaced as a maintainer call in
  §10.4, and the recommendation (A) stays inside.
- **P6 touches `widgets/copilot_chat.py`** (`_draw_revert_modal`) — unavoidable, since that modal
  is the chrome outlier, and §10.3's carve-out is for the *transcript's* inline controls, not the
  modal. Worth stating explicitly so it does not read as scope creep.
- **P4 depends on the revert modal becoming `confirm_modal`'s basis**, so it inherits the same
  touch and must land after P6.

No proposal touches `shaderbox/copilot/` itself.

## Ordering constraint

P6 before P4 (`confirm_modal` generalizes `_draw_revert_modal`, which P6 restructures).
P1 before P9 (the gate's allowlist must know where `command_menu_item` lives).
P2 and P5 decided together (the `Group...` state question is shared).
