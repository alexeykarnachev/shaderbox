# Post-implementation CLOSURE round — 093/17 menus — `d6bc425`

Scope: the fix round `d6bc425` against the three post-implementation reports on `f045eef`
(`menus_post_correctness.md` 7 findings, `menus_post_architecture.md` 3,
`menus_post_fidelity.md` 4 + §4's five stale spec sentences). Role: closure — every finding
either closed with the line that closes it and a re-run red mutation, or named STILL OPEN.

Anchor: a clean worktree at `d6bc425`, the main checkout's `.venv` with the worktree first on
`PYTHONPATH`. Every mutation below was applied in that worktree, run, and restored; the
worktree's `git status --porcelain` is empty at the end of the round. The shared working tree
was not modified.

`d6bc425` as committed: **pyright 0 errors**, `ruff check` clean, `ruff format --check` 306
files already formatted, **2510 passed, 4 skipped** under `-n 8 --dist loadgroup`.

**Verdict: PASS.** All 14 findings closed. The eight fixes the reports proved by mutation were
re-proved here — each mutation reapplied, each named test red, each restored. Two more
mutations were run against the fix round's own new rig (`_MenuDriver`) and two against the
fixes' logic, all red. One partial-by-design close is recorded below (architecture 3) with the
reason it is the fix the report asked for. No new defect.

---

## 1. Closure, finding by finding

### `menus_post_correctness.md` — 7 findings

#### 1. HIGH — the Group prompt grows 72px a frame — **CLOSED**

Two lines close it. `ui_primitives.py:585-586` — the clamp that discarded every caller's
number is gone and `width` is the field's, verbatim:

```python
    cancel_w = imgui.calc_text_size("x").x + float(SPACE.MD) * 2.0
    field_w = width if width > 0.0 else imgui.get_content_region_avail().x - cancel_w
    imgui.set_next_item_width(field_w)
```

`pass_graph.py:1523-1525` — the prompt now passes one:

```python
    result = name_input_row(
        "graph_group_name", view.group_input, width=float(SIZE.NAME_INPUT_W)
    )
```

The docstring states the mechanism as a present fact (`ui_primitives.py:575-577`): "The
`width <= 0.0` fallback reads the content region, which only has a fixed point inside a
container whose own width does not follow its content — an auto-sized popup must pass a width
or it grows every frame."

**Mutation re-run (M7):** drop the explicit `width=` at the prompt →
`test_the_group_prompt_keeps_one_width_across_frames` **red** (`1 failed`). Restored.

The related LOW (the `width` parameter inert at every caller) is closed by the same edit and
is architecture finding 2 below.

#### 2. MEDIUM-HIGH — `Delete document` on the bar with no confirm — **CLOSED**

`CommandSpec` gains the field (`commands.py:98-100`), `DELETE_DOCUMENT` carries it
(`commands.py:137`, `confirm_label="Move to trash"`), and `command_menu_item` branches on it
(`menus.py:56-62`):

```python
    enabled = menu_enabled(app, spec)
    if spec.confirm_label:
        fired = confirm_menu_item(spec.label, spec.confirm_label, enabled=enabled)
    else:
        chord = app.effective_bindings.get(command_id, spec.default_chord)
        hint = chord_to_str(chord) if chord else ""
        fired = imgui.menu_item(spec.label, hint, False, enabled=enabled)[0]
```

`document_grid.py:52` reads the same string off the spec rather than retyping it, so the bar
and the tile cannot drift:

```python
    if confirm_menu_item("Delete", SPEC_BY_ID[CommandId.DELETE_DOCUMENT].confirm_label):
```

**Mutation re-run (M8):** drop `confirm_label="Move to trash"` from the spec →
`test_the_bars_delete_document_is_a_confirm_submenu` **red**. Restored.

Two facts probed rather than taken on the commit's word. Exactly one spec carries a
`confirm_label` (`[('delete_document', 'Delete document', 'Move to trash')]`), so no other
verb silently gained a submenu. And `Alt+D` still fires unconfirmed — driving
`app.command_callbacks[DELETE_DOCUMENT]()` took the document count 1 → 0. That is the intended
split the commit body names: a chord is a deliberate two-key act, a menu item one pointer slip.
The chord HINT is genuinely lost from the item (`begin_menu` takes no shortcut argument); the
commit says so, and the palette and cheatsheet still carry the chord.

#### 3. MEDIUM — four object-menu verbs no test reached — **CLOSED**

`tests/test_menus.py:809-847` adds `_MenuDriver`, which right-clicks the target and clicks each
item **by the rect it was submitted at**, and the four verbs are spied with `wraps` so the real
call still runs. Four tests at `:850-925`.

**All four mutations re-run, each against its own named test:**

| Mutation | Test | Result |
|---|---|---|
| `pass_list.pass_menu_items` Delete body → `pass` | `test_the_pass_delete_item_reaches_the_session_verb` | **red** (1 failed) |
| `document_grid.document_menu_items` Delete body → `pass` | `test_the_document_delete_item_reaches_the_app_verb` | **red** (1 failed) |
| `document_grid` `Open folder` body → `pass` | `test_the_open_folder_item_reaches_the_app_verb` | **red** (1 failed) |
| `pass_graph._box_menu_items` Dissolve body → `pass` | `test_the_box_dissolve_item_reaches_the_app_verb` | **red** (1 failed) |

Each restored; the file returns to 35 passed.

#### 4. LOW — an ungrouped pass draws two adjacent separators — **CLOSED**

`pass_list.py:86-93` gates the first separator on the group test instead of emitting it
unconditionally:

```python
    grouped = bool(document.graph.passes.get(name, PassEntry()).group)
    if grouped:
        imgui.separator()
        if imgui.menu_item_simple("Leave group"):
            app.leave_group(document_id, name)
    imgui.separator()
```

Driven, recording `imgui.separator` / `menu_item_simple` / `confirm_menu_item` through one real
frame each way:

```
PASS MENU (strip, ungrouped): ['Open shader', 'Settings', '---', 'Delete > [Delete pass main]']
PASS MENU (node, GROUPED):    ['Open shader', 'Settings', 'Group', '---', 'Leave group', '---', 'Delete > [...]']
```

One rule for the common case, two when grouped — M4's table exactly.

**Mutation re-run (S1):** hoist the separator back out of the `if grouped:` →
`test_an_ungrouped_pass_draws_one_separator_before_delete` **red**. Restored.

#### 5. LOW — two structural tests read `app.py` by a cwd-relative path — **CLOSED**

`test_import_dialog.py:21` gains `_PKG = Path(__file__).resolve().parent.parent / "shaderbox"`
and `:73` uses it; `test_modal_chrome.py:247` reaches through the `_PKG` it already defined.

Verified by running both files **from `/tmp`**: `50 passed`. Before the fix both raised
`FileNotFoundError`.

Recorded, NOT filed as a finding: five other test files still read `Path("shaderbox/...")`
cwd-relatively (`test_roadmap_shape.py:91`, `test_region_system_is_gone.py:78`,
`test_keymap_disjoint.py:31,302`, `test_project_management.py:138,633,666`). All predate this
feature — the same lines are present at `f497bf5` — and the report scoped its finding to the
two files the commit touched. Out of this round's scope.

#### 6. LOW — `_group_prompt` strands `group_input` on a click-away — **CLOSED**

`pass_graph.py:1515-1521`:

```python
    if not imgui.begin_popup("##graph_group"):
        # A click outside dismisses the popup without reaching either commit branch, so the
        # input is closed here or `is_open` reports a prompt that is no longer on screen.
        # `begin_popup` returns True on the frame `open_popup` ran, so this cannot fire on
        # the opening frame.
        if view.group_input.is_open:
            view.group_input.close()
        return
```

The hazard this branch could introduce — cancelling a prompt the user just asked for — was
probed rather than reasoned. Driving 8 real frames after `group_input.open()` and reading
`is_open` before and after each `_group_prompt` call:

```
PROMPT is_open (before, after) per frame: [(True, True) × 8]
```

The opening frame is never eaten. And the branch does fire on a genuine dismissal: dismissing
the popup the way imgui itself does leaves `is_open = False` on the following frames
(`AFTER ESC-DISMISS is_open = False`).

#### 7. LOW — the lib picker's Esc branch is dead — **CLOSED**

The branch is deleted (`lib_picker/__init__.py`, 7 lines removed). `input_owns_keys` survives
with three other consumers (`:68, :70, :87`), so no dead local was left behind.

Probed that the funnel still owns the close, both ways: with the picker open,
`app.close_popup()` returns True and lands `PopupState.CLOSED`; with `file_rename` open,
`inline_input_owns_esc()` is True and `close_popup()` returns **False** with the picker still
open. The deletion removed a branch that could not fire, not a behavior.

### `menus_post_architecture.md` — 3 findings

#### 1. The stale comment on `close_active_tab` — **CLOSED**

`app.py:966-969` now states what holds:

```python
        # Close the active editor tab (the Ctrl+W / tab-bar-x path share close_tab). No-op
        # with no tabs open. The chord needs editor focus; the menu item asks only for an
        # active tab, since a menu click has already taken the focus away.
```

The false clause ("only fires while the editor is focused") is gone, and the replacement names
the divergence the finding identified rather than narrating the fix.

#### 2. `name_input_row`'s `width` is inert — **CLOSED**

Same edit as correctness 1: the `max(NAME_INPUT_W, field_w)` clamp is deleted, so the caller's
number is the field's width. Projects' 180 is now honored because it is passed, not because it
coincides with the clamp.

#### 3. A third scope predicate over `CommandScope`, un-funneled — **CLOSED (partially, by the
report's own cheaper option)**

The report offered two fixes explicitly: a funnel table, "or — the cheaper one — add one
sentence to the `conventions.md` menu-bar bullet stating that the menu's enabled test is
deliberately *not* the hotkey's, and why". The fixer took the cheaper option and added a gate
on top of it.

The recorded decision landed in `dev_flow.md:333-336` rather than `conventions.md`:

> `menu_enabled(app, spec)`. That predicate asks "does the verb have a target" while
> `hotkeys.spec_eligible` asks "may this key press fire" — deliberately different questions,
> since the click that opened the menu has already taken the focus a chord's gate reads.

And `menus.py:36-45` became exhaustive:

```python
    match spec.scope:
        case CommandScope.EDITOR:
            return app.active_tab is not None
        case CommandScope.COPILOT:
            return app.is_copilot_open
        case CommandScope.GLOBAL:
            return True
        case _:
            assert_never(spec.scope)
```

**Mutation re-run, and it exposes the shape of the close.** Adding a fourth `CommandScope`
member (`GHOST_SCOPE = auto()`) and running pyright **with the repo's own config** (a path
argument overrides `[tool.pyright] include` — the trap `pyproject.toml:106-107` warns about, so
a bare `pyright shaderbox/menus.py` reports 0 errors and proves nothing):

```
shaderbox/menus.py:44:26 - error: Argument of type "Literal[CommandScope.GHOST_SCOPE]"
    cannot be assigned to parameter "arg" of type "Never" in function "assert_never"
1 error, 7 warnings, 0 informations
```

Baseline is 0 errors. Restored.

**What is closed and what is not.** One of the three sites is exhaustive; the other two still
answer a new member by fallthrough — `cheatsheet._is_active` would call it active whenever no
popup is open, `hotkeys.spec_eligible` would let its chord dispatch. Those two silent answers
survive. But the gate does fire on the change, and a type error on `menus.py` is the thing that
sends the author to the enum and therefore to its readers. That is what the report asked for,
by the option it named cheaper, so this is closed rather than open — recorded here because
"exhaustive over `CommandScope`" reads broader than the one site it covers.

### `menus_post_fidelity.md` — 4 findings

#### 1. `Group` sits after the pass set — **CLOSED**

The fixer took the first of the two options (move the code), which is the one that matches the
design pass and the Items column. `pass_list.pass_menu_items` gains a `slot`
(`pass_list.py:66-71`) drawn after `Settings` (`:84-85`), and the node passes its `Group` drawer
into it (`pass_graph.py:1498-1507`).

Driven, real frames:

```
PASS MENU (node, ungrouped): ['Open shader', 'Settings', 'Group', '---', 'Delete > [...]']
PASS MENU (node, GROUPED):   ['Open shader', 'Settings', 'Group', '---', 'Leave group', '---', 'Delete > [...]']
```

`Group` is between `Settings` and `Leave group`, and no longer below the destructive verb.

**Mutation re-run (S2):** remove the `slot()` call so the node's `Group` draws last again →
`test_the_pass_set_is_the_same_on_the_strip_and_the_node` **red**. Restored.

The self-contradictory spec row is repaired on the other side too: M4's Function column now
reads "`pass_graph._node_menu` through `pass_menu_items`'s `slot`" and the inventory's P2 row
(`04_menus_inventory.md:625`) reads "(after Settings, before Leave group)".

#### 2. The document menu's verb wiring asserted by a vacuous line — **CLOSED**

Covered by correctness 3's rig; the `Open folder` mutation is one of the four re-run red above.
The vacuous "exercised at the seam the item fires" block is gone from
`test_a_documents_menu_carries_open_open_folder_and_delete`, which now asserts labels only —
its honest scope — and the wiring lives in the driven tests.

#### 3. M9's emoji-picker cleanup has no behavioral test — **CLOSED**
#### 4. M9's Settings-apply cleanup has no test — **CLOSED**

Both close on the one parametrized test the report asked for
(`test_menus.py:987-1019`, `test_each_close_branch_runs_its_own_cleanup`, ids
`emoji_picker` / `settings`).

**Both mutations re-run:**

| Mutation | Result |
|---|---|
| `close_emoji_picker` reduced to the `CLOSED` write (target + query dangle) | **red** (1 failed, 1 passed — the emoji case fails, settings passes) |
| the `SETTINGS` branch's `self.apply_editor_settings()` deleted | **red** (1 failed, 1 passed — the settings case fails) |

Each restored. The per-case split is the right shape: each mutation fails exactly its own
parametrization and leaves the sibling green, which is what proves the two cases are
independent rather than one assertion covering both.

### The five stale spec sentences (fidelity §4) — **all CLOSED**

| § | Was | Now |
|---|---|---|
| 4.1 | "the twelve `_OVER_BUDGET` rows … are deleted" | `05_menus_spec.md:164-169` — `COPILOT_LIMIT_ROWS` in `copilot/config.py`, "The two `_OVER_BUDGET` tooltip rows … are deleted; the copilot hints were never in that list -- being reached through a loop variable they are `_UNMEASURABLE`", with the direct assertion "as `_FORMATS` already does". Verification row `:251` follows. |
| 4.2 | "appended to the Help panel's copilot section" | `:165` — "printed by a NEW Help copilot section (there was none to append to)" |
| 4.3 | "`tree.py` contains no `push_style_color`" | `:243` — "`tree.py` contains no `COLOR.STATE_ERROR` push (the favorite star keeps its own, §10.3)" |
| 4.5 | `InputRowResult(committed, cancelled)` | `:112-118` — the third field plus its reason ("the `x` is submitted after the input, so an `is_item_focused()` at the call site answers for the button") |
| 4.7 | `inline_input_owns_esc(app)` | `:145` — `app.shader_lib_files.inline_input_owns_esc()` |

The `## Review history` paragraph (`:266-278`) records all three reports and this round.

---

## 2. New-diff audit

`git show d6bc425` read in full; every file it touches read end to end in the worktree. The
specific shapes a fix round tends to introduce were each probed, not reasoned about.

**`_MenuDriver` — does it click what imgui submitted, or a rect it guessed?** It records
`get_item_rect_min/max` inside wrappers that call the real `menu_item_simple` /
`confirm_menu_item`, so the rect is the one the item was submitted at. Two mutations against
the rig itself, over the five tests that use it:

| Rig mutation | Result |
|---|---|
| `click()` aims 500px away from the recorded rect | **5 failed** |
| `open_menu()` never sends the right-click | **5 failed** |

So neither the click nor the menu-opening is decorative. A third: dropping `menus` from the
driver's per-module patch list turns the bar's confirm test red (`1 failed, 2 passed`) — the
list is load-bearing and correctly scoped, and the `raising=False` is not hiding a typo.

The verbs are spied with `wraps`, so the mock does not replace the call the menu makes — which
is what makes the four verb mutations above meaningful rather than assertions about the mock.

**`pass_menu_items`'s new `slot` — do the two callers diverge?** Yes, by design, and the
divergence is exactly one item. `pass_graph.py:1507` passes `slot=group_item`;
`pass_list.py:103` (the strip) does not. M4's table states this as two rows — the shared set
and "pass, node only … `Group` between `Settings` and `Leave group`" — and
`pass_list.py:77-78` states it at the code: "`slot` draws a caller's own items after
`Settings` — the graph node's `Group`, which the strip has no selection to seed." The order
test drives the real slot rather than rebuilding the node menu inline, which is what the
correctness report faulted the old version for.

**The `match` + `assert_never` in `menu_enabled`.** Verified against the enum: `CommandScope`
has exactly the three members the match arms cover, the fourth arm is `assert_never`, and the
break (a fourth member) produces a pyright error. Covered above under architecture 3, including
what it does not reach.

**The `_UNMEASURABLE` row for `menus.py::confirm_menu_item` — is it call-site-scoped, and does
the gate still catch a five-word literal elsewhere in `menus.py`?** Both verified by mutation
rather than by reading the list's shape:

| Mutation | `tests/test_ui_prose_budget.py` |
|---|---|
| a five-word `menu_item_simple` literal added **inside `command_menu_item` itself** | **red** (1 failed, 662 passed) |
| a five-word `menu_item_simple` literal added in `draw_menu_bar` | **red** (1 failed, 662 passed) |

The row does not widen to the function. The mechanism is `test_every_unmeasurable_site_is_listed`
consulting `_UNMEASURABLE` only for sites the walk already classed UNREADABLE (a non-literal
argument), with `test_no_site_is_both_measured_and_unmeasurable_listed` (`:650-658`) refusing
an entry that suppresses a measurable function: "An entry that suppresses a measurable function
is a hole, not an exemption." The entry itself names its reason and points at the test that
measures the table.

**`document_grid` importing `commands` — right direction?** Yes, and precedented. The
three-layer rule as `conventions.md:586` states it bars `ui_primitives.py` from `App` and
`commands`; it says nothing barring a widget. `document_grid.py:9` already imported `App`, so
it sits in the `App`-aware layer, and eleven modules already import `commands` — including four
widgets (`cheatsheet`, `copilot_chat`, `pass_graph`, and now `document_grid`), `tabs/document`,
both exporters and `popups/settings`. `commands.py` remains a leaf (the import is
widget → leaf, never the reverse), which is the property the rule protects. The import is also
the point of the fix: two surfaces reading one string instead of retyping it.

**The group prompt's `input.close()` on dismissal.** Probed above (correctness 6): the opening
frame is safe across 8 driven frames, and the branch does fire on a real dismissal.

**The spec's updated sentences.** All five rewritten sentences describe the code as it now is;
each was checked against the line it describes. One thing the round did NOT update, correctly:
M5's signature at `:86-87` still reads `confirm_menu_item(label, confirm_label)` without the
`enabled` third parameter. The fidelity report judged that addition correct and did not file it
as stale ("an addition to the spec's two-argument signature, and a correct one"), so leaving it
is consistent with the round's own scope rather than a miss.

**Tests that pass for two reasons.** Each of the eight new tests was mutated against the thing
it names and went red; none stayed green under its own break. The two `close_popup` cases fail
independently, so neither is riding on the other.

---

## 3. The user-facing result

### The menu bar — five menus, as they render

Read off `COMMAND_SPECS` filtered on `in_menu`, in `CATEGORY_ORDER` (`File · Document · Editor
· View · Tools`), 27 items.

**File** (3) — `Projects` `Alt+O` · `Save` `Ctrl+S` · ─ · `Quit` `Ctrl+Q`

**Document** (6) — `New document` `Ctrl+Shift+N` · `Delete document` ▸ `Move to trash` ·
`Play/stop document script` `F5` · `Reset document` `F6` · `Next pass` `Alt+Right` ·
`Previous pass` `Alt+Left`

**Editor** (6) — `Open shader` `Alt+C` · `Open script` `Alt+R` · `Open graph` `Alt+G` ·
`Close code tab` `Ctrl+W` (greyed with no active tab) · `Jump to next error` `F8` ·
`Format` `Ctrl+Shift+I` (greyed with no active tab)
— out of menu: `Cycle code tab`

**View** (3) — `Cycle channel view` `Alt+V` · `Toggle copilot` `Alt+J` ·
`Cycle copilot layout` `Ctrl+H` (greyed while the copilot is closed)
— out of menu: `Document tab`, `Uniforms tab`, `Render tab`, `Share tab`

**Tools** (9) — `Shader library` `Alt+L` · `Command palette` `Ctrl+Shift+P` · `Settings`
`Alt+S` · `Pass settings` `Alt+P` · `Add pass` `Alt+A` · `Import passes` (unbound) ·
`Examples` `Alt+E` · `Help` `F1` · `Toggle keyboard cheatsheet` `Alt+/`

**Does anything read wrong?** Read as a user of a node-graph shader tool: no.

- Every label is a verb or the noun of the surface it opens, and no label carries an ellipsis.
- The one destructive verb on the bar (`Delete document`) is behind its submenu, which is the
  finding this round closed. It is the only spec with a `confirm_label`, verified.
- The separator sits above `Quit` and nowhere else, so no menu opens with a rule.
- Categories hold what their name promises. Two worth naming because they could read wrong and
  do not: `Add pass` / `Import passes` sit under **Tools** rather than **Document**, which
  matches where the palette and the canvas menu put them (they act on the graph, not the
  document record); and `Open shader` / `Open script` / `Open graph` sit under **Editor**
  rather than **Document** because they open editor tabs. Both are pre-existing table
  decisions, unchanged by either commit.
- `Delete document`'s chord hint is absent from the item (a `begin_menu` carries no shortcut),
  which is the one visible cost of the confirm submenu. `Alt+D` still works and is shown in the
  palette and cheatsheet. Worth the trade the round made, but a user who learns chords off the
  menu will not learn this one there.
- `Import passes` is the one item with no chord, so its hint column is blank. Correct — the
  spec carries chord `0` deliberately.

### The four context menus, from their item-set functions

Driven through real frames, recording `separator` / `menu_item_simple` / `confirm_menu_item`:

**Pass** (`pass_list.pass_menu_items`) — strip tile and graph node

```
strip, ungrouped:  Open shader · Settings · ─ · Delete ▸ Delete pass <name>
strip, one pass:   Open shader · Settings · ─ · Delete ▸ (greyed, does not open)
node,  ungrouped:  Open shader · Settings · Group · ─ · Delete ▸ Delete pass <name>
node,  grouped:    Open shader · Settings · Group · ─ · Leave group · ─ · Delete ▸ Delete pass <name>
```

**Document** (`document_grid.document_menu_items`) — grid tile

```
Open · Open folder · ─ · Delete ▸ Move to trash
```

**Group box** (`pass_graph._box_menu_items`) — graph box

```
Open · Dissolve
```

**Canvas** (`pass_graph._canvas_menu`) — canvas background

```
Add pass · Import passes · ─ · Fit · Arrange
```

**Does anything read wrong?** No.

- Both destructive verbs (`Delete` on a pass, `Delete` on a document) carry their submenu, and
  the confirm text names the consequence rather than repeating the verb — `Move to trash`,
  `Delete pass <name>`. `Dissolve` is not destructive in the same sense (it unwraps a group,
  the passes survive), so its bare item is right.
- Every label is a verb. `Settings` on the pass menu is the one noun, and it names the surface
  it opens, consistent with the bar.
- The last pass's `Delete` draws greyed and does not open, so the item that cannot act cannot
  be clicked into.
- `Group` now sits between `Settings` and `Leave group`, so the create verb and the leave verb
  are adjacent and the destructive verb is last behind its own rule — which reads better than
  the shipped order it replaced (`Group` below `Delete`).
- One rule for an ungrouped pass, two when grouped; no menu draws two rules in a row.

---

## 4. False trails

Each looked like a finding and is not. Recorded so the next reader does not re-walk them.

1. **`make gates` is RED on the working tree.** Running it from the main checkout gives
   `EXIT=2`, `1 failed, 2508 passed` on
   `test_formatting.py::test_the_chord_is_registered_on_the_editor_scope`
   (`assert 'Format code' == 'Format'`). This is NOT a defect in either commit under review.
   `git status --porcelain` shows ten files modified beyond `d6bc425` (including
   `commands.py`, +234/-…, which renames the label) — in-progress work by the maintainer. The
   worktree at `d6bc425` itself is green: pyright 0 errors, ruff clean, **2510 passed, 4
   skipped**. The review was anchored there throughout.

2. **`pyright` reporting 0 errors under the `assert_never` break.** A first run of
   `pyright shaderbox/menus.py` with the ghost member in place reported `0 errors`, which read
   as a gate that does not fire. It was my command line: a path argument overrides
   `[tool.pyright] include`, which `pyproject.toml:106-107` and `.pre-commit-config.yaml:26-28`
   both warn about in writing ("NO path argument: a path on the command line OVERRIDES
   `[tool.pyright] include`"). Run from the worktree with no path, the error fires. The gate is
   real; the first probe was not.

3. **A click-away failing to close the group prompt.** Clicking at (3,3) in the test rig left
   `group_input.is_open` True across five further frames, which read as the LOW fix not working.
   Instrumenting showed the row still drawing — the popup was never dismissed by that click, so
   the branch under test was never reached. (The architecture report hit the same thing and
   recorded it: "the popup survives the outside click".) Dismissing the popup the way imgui
   itself does closes the input. The fix works; the probe did not reproduce the scenario.

4. **The `_UNMEASURABLE` row looking like a function-wide exemption.** The key is
   `(file, function)`, which reads as suppressing everything in `command_menu_item`. It does
   not — the list is consulted only for sites already classed UNREADABLE, and
   `test_no_site_is_both_measured_and_unmeasurable_listed` refuses an entry that covers a
   measurable function. Demonstrated by two mutations, both red.

5. **Five other test files still reading `shaderbox/…` cwd-relatively.** Found while verifying
   correctness 5 and it looks like an incomplete fix. Every one predates this feature (the same
   lines are present at `f497bf5`), and the report scoped its finding to the two files the
   commit touched. Not this round's surface.

6. **`_MenuDriver`'s `raising=False` hiding a bad module name.** `monkeypatch.setattr(module,
   "confirm_menu_item", confirm, raising=False)` would silently do nothing if a module in the
   list did not import the symbol. Mutated the list to drop `menus`: the bar's confirm test
   goes red, so the list is checked by the tests that depend on it rather than by the flag.

7. **`document_grid` importing `commands` inverting a layer.** A widget importing the command
   registry reads like the wrong direction. It is the right one — `commands.py` is the leaf and
   stays one; the widget already imported `App`; and eleven modules including four other
   widgets already do the same.

---

## 5. Working tree

The review created one worktree at `d6bc425` under the session scratchpad, ran every mutation
there, and restored each one; `git status --porcelain` in that worktree is empty. Four
throwaway probe files were written into the worktree's `tests/` and deleted. The shared working
tree at `/home/akarnachev/src/shaderbox` was not modified by this review — its ten modified
files are the maintainer's own in-progress work, present before the round began.

---

## 6. Verdict

**PASS.** All 14 findings across the three reports are closed, plus fidelity §4's five stale
spec sentences. Ten mutations were re-run — the eight the reports and the commit body named,
each red against its named test and restored, plus two against the fix round's own rig — and
four more against the round's new logic and the prose gate. The new tests do not pass for a
second reason: each goes red under its own break, and the two parametrized close-funnel cases
fail independently.

One close is narrower than its commit body reads, and is recorded rather than filed:
`menu_enabled` is exhaustive over `CommandScope`, but `cheatsheet._is_active` and
`hotkeys.spec_eligible` still answer a new member by fallthrough. The `assert_never` does fire
on the change, which is the gate the report asked for by the option it called cheaper, so this
is a closure with a named edge rather than an open finding.

No new defect. The menu bar and the four context menus render in an order that reads correctly
for the tool: every destructive verb behind its submenu, every label a verb, `Group` between
`Settings` and `Leave group`, and no menu drawing two rules in a row.
