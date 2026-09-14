# Post-implementation review of `f045eef` — architecture and conventions

Role: ARCHITECTURE AND CONVENTIONS. Anchors read in full before the diff: `CLAUDE.md`
(`## Hard rules`, `## Code rules`), `ai_docs/conventions.md` (`## Code rules`, the
`## Design decisions` three-layer bullet, the `popups/*.py` + `PopupState` bullet, the
rewritten `InlineInput` bullet, the destructive-verb bullet, the right-click-hint bullet,
the menu-bar bullet, the graph-view bullet, the copilot-config-by-tunability bullet, the
speculative-machinery bullet), `.claude/skills/imgui-ui/SKILL.md` §1/§6/§7,
`ai_docs/dev_flow.md ### Module map` and `## Documentation discipline`.

Every changed file was read end to end. **Skipped: none of the 32 source/doc files.** The
five changed test files (`test_graph_view.py`, `test_import_dialog.py`, `test_lib_files.py`,
`test_menus.py`, `test_modal_chrome.py`, `test_project_management.py`,
`test_ui_prose_budget.py`) were read only where they bear on an architecture claim — the
gate-correctness review is another reviewer's role.

`make gates` re-run from this tree: **exit 0**, captured unpiped
(`make gates > /tmp/.../gates.log 2>&1; echo $?` → `EXIT=0`; "GREEN -- check passed, test
passed, smoke passed"). `uv run pyright` on the changed modules: 0 errors. `ruff check
shaderbox/`: all checks passed.

---

## Verdict: **PARTIAL** — three findings, none blocking, all demonstrated.

The layering, the module placement, the deletion hygiene and the doc updates are all
correct; I could not break any of them. The three findings are one stale comment the diff
invalidated, one inert parameter, and one un-funneled predicate — each a convention the repo
states in writing.

---

## What holds (each checked, not assumed)

**Layering.** `ui_primitives.py` imports `profiling`, `render_plan`, `theme` and stdlib
only — no `App`, no feature module:

```
$ grep -nE "^from shaderbox|^import shaderbox" shaderbox/ui_primitives.py shaderbox/theme.py
shaderbox/theme.py:33:from shaderbox.editor import ffi as editor_ffi
shaderbox/theme.py:34:from shaderbox.intel.symbols import SymbolKind
shaderbox/ui_primitives.py:14:from shaderbox.profiling import FrameProfile, Span, by_cost, headline_ms, other_ms
shaderbox/ui_primitives.py:15:from shaderbox.render_plan import RenderPlan, document_id_of_span
shaderbox/ui_primitives.py:16:from shaderbox.theme import (
```

`menus.py` imports exactly `App`, `commands` and `theme` — the split the conventions bullet
prescribes, and the reason `draw_menu_bar` could not live in `ui_primitives.py`.
`commands.py` is still a leaf (`dataclasses`, `enum`, `imgui_bundle` only), so
`command_label` did not cost it its leaf status — which matters, because `exporters/
telegram.py`, `exporters/youtube.py`, `tabs/document.py` and `widgets/copilot_chat.py` all
now import it.

**No import inside a function body** among the changed files except the two sanctioned
lazy-SDK seams:

```
$ for f in $(git show f045eef --name-only --format="" | grep '\.py$' | grep -v '^tests/'); do
    awk -v F="$f" '/^[ \t]+(import |from .* import )/ {print F": "NR": "$0}' "$f"; done
shaderbox/exporters/youtube.py: 566:         from shaderbox.exporters import youtube_api
shaderbox/exporters/youtube.py: 605:         from shaderbox.exporters import youtube_api
```

Those are the `youtube_api` seam the 066 bullet names; both predate this commit.

**No suppression, no `TYPE_CHECKING`, no `@staticmethod`, no `TODO`, no `Any` on a
real-typed parameter** anywhere in the added lines:

```
$ git show f045eef --format="" -- 'shaderbox/*.py' | grep -E '^\+' \
    | grep -E 'noqa|type: ?ignore|pyright: ?ignore|TODO|FIXME|TYPE_CHECKING|@staticmethod|@classmethod|\bAny\b|from __future__'
(no output)
```

**Full annotations** on every function the commit touched. An AST walk over all changed
`.py` files found exactly one gap, `widgets/copilot_chat.py::_tooltip_stat_row(value_color)`,
and `git log -S` places it in `4bbf28e` (feature 034) — pre-existing, untouched here.

**Deleted symbols leave no live reference.** Grepping every name the diff retired against
the LIVE docs (`conventions.md`, `dev_flow.md`, `todo.md`, `roadmap.md`, `.claude/skills/`)
and all of `shaderbox/`:

| symbol | live-doc hit | source hit |
|---|---|---|
| `document_delete_armed`, `set_document_delete_armed` | none | none (only `test_menus.py`'s `not hasattr` pins) |
| `file_delete_armed`, `dir_delete_armed` | none | none (only the `not in annotations` pins) |
| `arm_file_delete`, `arm_dir_delete` | none | none |
| `group_prompt`, `group_name` (the `GraphViewState` fields) | none | none — `pass_graph._group_prompt` is the *function*, kept; `group_name_error` / `group_names_in_order` are unrelated symbols |
| `is_keep_opened` | `SKILL.md:356` only, as the counter-example the rule names | none |
| `_hint`, `_draw_menu_bar`, `_COPILOT_LIMITS` | none | none |
| `inline_input_owns_esc` (module-level, `lib_picker/__init__.py`) | none | only the method on `ShaderLibFileManager` |

The hits under `ai_docs/features/093_refinement/{04_menus_inventory,05_menus_spec,reviews/*}.md`
are the feature RECORD describing what was removed — the conventions' "this file is not a
changelog / the story belongs in the feature spec" split puts them there correctly.

`preview_cell`'s `armed` / `deletable` / `cell_delete_confirm` / `close_cross_button`
survive in `ui_primitives.py` with one live arming caller
(`exporters/telegram.py`'s sticker grid, `:699` / `:708-720`), exactly as M6 scoped.

**Where the new symbols live.** Each is in the module the module map would put it:

- `command_label` in `commands.py` — pure, keeps the leaf, reachable from the exporters.
- `close_popup` / `close_emoji_picker` / `open_document_dir` on `App` — the `popups/*.py`
  bullet's "open/closed state lives on `App`"; `close_popup` dispatching on `popup_state` is
  that bullet's mutex read back as a funnel, and it is the "cross-cutting guarantee at the
  single FUNNEL" law applied (`hotkeys._handle_escape` lost its four per-caller carve-outs).
- `inline_input_owns_esc` as a method on `ShaderLibFileManager` — this is the load-bearing
  move: it is what let `hotkeys.py` drop `from shaderbox.popups.lib_picker import
  inline_input_owns_esc`, so no `App`-adjacent module imports a popup any more.
- `confirm_menu_item`, `InlineInput`, `InputRowResult`, `name_input_row` in
  `ui_primitives.py` — `App`-free, and the conventions bullet's own trigger ("promote
  `InlineInput` to `ui_primitives.py` when a second multi-inline-input surface lands") is
  what fired.
- `COPILOT_LIMIT_ROWS` in `copilot/config.py` — **the right home, verified rather than
  assumed.** The module's whole import list is `from dataclasses import dataclass`, and a
  probe confirms it pulls no imgui or GL:
  ```
  $ uv run python -c "import sys; b=set(sys.modules); import shaderbox.copilot.config; \
      print([m for m in set(sys.modules)-b if m.split('.')[0] in ('imgui_bundle','glfw','moderngl','OpenGL')])"
  []
  ```
  So the headless-drivable claim survives. The placement also follows the precedent
  `help_content.py` states in its own docstring — "the two generated ones read their facts
  from the code that owns them (`ENGINE_DRIVEN_UNIFORMS`, `COMMAND_SPECS`)". `COPILOT_LIMIT_ROWS`
  is that shape: the module that owns the knobs owns the copy describing them, and two UI
  surfaces read it rather than each typing its own.

**No duplicated draw block.** The four item-set functions have exactly one caller each, and
the caller owns the popup in every case:

```
document_grid.py:111   begin_popup_context_item(f"##document_menu_{id}") -> document_menu_items(app, id)
pass_list.py:92        begin_popup_context_item(f"##pass_menu_{name}")   -> pass_menu_items(app, document_id, name)
pass_graph.py:1494     begin_popup_context_item(None)                    -> _box_menu_items | pass_menu_items
```

`name_input_row` absorbed every hand-rolled name-entry row the spec enumerated; the nine
surviving `imgui.input_text` sites outside `ui_primitives.py` are search fields
(`lib_picker/search.py`, `emoji_picker.py`), `label_row` form fields inside a settings modal
(`pass_settings.py` ×3, `import_passes.py`), a uniform value field (`uniform.py`) and a tag
buffer (`lib_picker/preview.py`) — none of them is the `x`-cancel name-entry row the bullet
scopes, and M7 enumerated its four callers exactly.

`confirm_menu_item` left no armed label behind: the only `push_style_color(..., STATE_ERROR)`
outside `ui_primitives.py` are the code tab's error tint and telegram's confirm wash, neither
of them a menu.

The menu bar and the palette both render from `COMMAND_SPECS` filtered on their own flag
(`in_menu` / `in_palette`) and resolve the chord the same way — the same table, two widgets,
not the same block twice.

**No development-history comment.** Every added comment states a present fact: the two
`CommandSpec` field comments, `menus.py`'s right-align note (carried over verbatim from
`ui.py`), `confirm_menu_item`'s and `name_input_row`'s docstrings, `pass_list`'s
"the last pass of a document cannot go", `document_grid`'s module docstring,
`lib_picker/__init__.py`'s "a Close reached with an inline input still armed cancels it
first", `pass_graph`'s "Node-only (092 D14)". The one that reads closest to a backstory —
`projects.py`'s "`App.close_popup` leaves the modal open for exactly this" — names a live
coupling the next reader needs, and its predecessor said the same thing about
`hotkeys._handle_escape`.

**Docs describe the code as it now is.** `conventions.md`'s four new bullets each carry a
"Revisit if" (submenu unreachable on a touchpad / a menu walk finds an undiscovered hint /
a menu needs an item that is not a command / a row needs a third commit rule);
`dev_flow.md`'s module map gained `menus.py` and rewrote the `ui.py`, `commands.py`,
`hotkeys.py`, `ui_primitives.py` and `widgets/` entries; the skill's §7.1 gained the chrome
gate and §7.4 replaced the retired `enabled=` footgun with the measured fact plus the submenu
rule and the hint's placement; 092's D14 and D16 both now point forward to what 093/17 did to
them rather than stating what is no longer true; `00_findings.md` row 17's Landed-in and
`01_spec.md`'s wave list are both filled. **No TODO / deferred / "later" marker was
introduced anywhere in the doc diff** — checked by the same grep as the source.

---

## Findings

### 1. A comment the diff invalidated but did not update: `App.close_active_tab`

`shaderbox/app.py:966-970`:

```python
def close_active_tab(self) -> None:
    # Close the focused editor tab (the Ctrl+W / tab-bar-x path share close_tab). No-op
    # with no tabs open; only fires while the editor is focused (CommandScope.EDITOR gate).
    if self.editor_tabs:
        self.close_tab(self.active_tab_index)
```

The final clause is now false. `CLOSE_CODE_TAB` carries `in_menu=True` (it is not in the
five specs the commit set to `False`), so the Editor menu renders it, and `menu_enabled`
gates on `active_tab is not None` — not on focus. A probe, run against this tree and then
deleted:

```
CLOSE_CODE_TAB scope = editor   in_menu = True
active_tab is None?             False
editor_focused =                False
menu_enabled(app, spec) =       True        <- the menu WILL fire it
spec_eligible (hotkey path) =   False       <- the chord will not
```

The divergence is correct behavior — a menu click necessarily removes editor focus, so a
menu item could never satisfy `editor_focused` — but the comment now tells the next reader
the opposite, and it is exactly the class `conventions.md ## Code rules` targets ("a comment
IS warranted [when] it states what's non-obvious about the code as it is NOW") and
`dev_flow.md ## Documentation discipline` targets ("a fact that makes a doc stale → update
the right file in the same wave"). `git log -S "only fires while the editor is focused"`
places the line in `da69ae6`, so the commit inherited it rather than writing it — but the
commit is what made it wrong.

**Fix:** drop the clause, or replace it with what is now true ("the menu fires it without
focus; the chord needs it").

### 2. `name_input_row`'s `width` parameter is inert at its only passing caller

`ui_primitives.py:579-585`:

```python
cancel_w = imgui.calc_text_size("x").x + float(SPACE.MD) * 2.0
field_w = (
    width - cancel_w
    if width > 0.0
    else imgui.get_content_region_avail().x - cancel_w
)
imgui.set_next_item_width(max(float(SIZE.NAME_INPUT_W), field_w))
```

The only caller that passes `width` is `popups/projects.py:166`:
`name_input_row("project_name", state, width=float(SIZE.NAME_INPUT_W))`, i.e. `width=180.0`.
With `SIZE.NAME_INPUT_W = 180` and `cancel_w ≈ 24`, `field_w ≈ 156`, and the clamp returns
`180`. A probe spying `imgui.set_next_item_width` across three real frames of the Projects
modal with its new-name input open:

```
NAME_INPUT_W = 180
widths passed to set_next_item_width: [36.0, 180]
```

`180` — the clamp, not the caller's computed `156`. The parameter only changes anything when
`width > NAME_INPUT_W + cancel_w` (~204), which no caller does. That is the
`## Design decisions` speculative-machinery bullet's exact test ("is REMOVING it churn?"):
removing it is a three-line diff at one call site, and the knob has zero effective consumers.
The three other callers omit it and get the correct content-region behavior.

**Fix:** delete the parameter and the branch, or drop the `max(NAME_INPUT_W, …)` clamp so the
argument the caller passes is the width it gets. Either is a change to `ui_primitives.py`
plus the one `projects.py` call.

### 3. A third scope predicate over `CommandScope`, un-funneled

`menus.menu_enabled` is now the third independent dispatch over the same two enum members,
each with a different answer:

```
shaderbox/hotkeys.py:333   if spec.scope == CommandScope.EDITOR and not app.editor_focused: return False
shaderbox/hotkeys.py:335   if spec.scope == CommandScope.COPILOT and not app.copilot_focused: return False

shaderbox/widgets/cheatsheet.py:18   if scope == CommandScope.EDITOR:  return app.editor_focused
shaderbox/widgets/cheatsheet.py:20   if scope == CommandScope.COPILOT: return app.copilot_focused
                                     return not app.any_popup_open()

shaderbox/menus.py:33      if spec.scope is CommandScope.EDITOR:  return app.active_tab is not None
shaderbox/menus.py:35      if spec.scope is CommandScope.COPILOT: return app.is_copilot_open
                                     return True
```

All three answer "may this command run now?" from `spec.scope`, and all three differ. The
menu's answers are the correct ones for a menu (finding 1 explains why focus cannot be the
test there), so this is not a defect today. It is the shape
`conventions.md`'s funnel law names: "a SECOND fix of the same bug at a sibling site is the
trigger to move to the funnel" — this is the third site, and a new `CommandScope` member now
has to be remembered in three places, with nothing failing if one is missed. The commit also
did not widen the module map or the menu-bar bullet to say that `menu_enabled` answers a
different question from `spec_eligible`.

**Fix (either):** put the answer on the enum or in `commands.py` as a table keyed by
(scope, surface), with the three sites reading it; or — the cheaper one — add one sentence to
the `conventions.md` menu-bar bullet stating that the menu's enabled test is deliberately
*not* the hotkey's, and why, so the divergence is a recorded decision rather than a drift.

---

## False trails (checked, nothing there)

- **`GraphViewState.group_input` leaking open after a click-away.** The rewrite replaced a
  `bool` latch with `needs_focus`, which `name_input_row` consumes on the first draw, so I
  expected a dismissed popup to strand `group_input.is_open == True` and block a reopen. A
  probe driving real frames (open → click at (3,3) → reopen, spying
  `pass_graph.name_input_row`) shows the row drawing 3 / 10 / 3 times across the three
  phases: the popup survives the outside click, and the reopen works because
  `InlineInput.open()` re-arms `needs_focus` unconditionally. `is_open` does go stale, but
  `grep -rn group_input shaderbox/ tests/` shows nothing reads it — only `needs_focus`,
  `buf`, `open` and `close` have consumers. Harmless.
- **`pass_settings.py`'s three surviving `imgui.input_text` rows as un-migrated
  `name_input_row` callers.** They are `label_row` form fields inside a settings modal, not
  the `x`-cancel name-entry row; `05_menus_spec.md:109-118` enumerates M7's four callers and
  `pass_settings` is not among them. Out of scope by the spec, and the conventions bullet
  scopes itself to "rename / new-file / new-dir / a group name / a project name".
- **`_box_menu_items` as an N=1 extraction.** It has one caller (`_node_menu`), which reads
  like abstracting from N=1. But the conventions' graph bullet now states the rule as "each
  object kind has exactly one item-set function in the widget module that owns it" — the
  symmetry IS the decision, and the group box is an object kind. Not a violation.
- **The menu bar duplicating `_register_palette_commands`.** Both walk `COMMAND_SPECS` and
  call `chord_to_str(effective_bindings.get(...))`, but they render into different widgets
  (`imcmd.Command` vs `imgui.menu_item`) with different filters. The shared thing — the table
  and the label — is already shared. No draw block exists twice.
- **`copilot/config.py` becoming imgui-coupled by hosting UI copy.** Probed above: it still
  imports only `dataclasses` and pulls no imgui/GL module. The Settings panel and
  `help_content.py` import *it*, never the reverse.
- **A leftover double gate on a disabled menu item.** `grep -rn "menu_item_simple(.*enabled="`
  returns one site, `lib_picker/tree.py:297` (`Insert at caret`), and it carries no redundant
  Python re-check — consistent with the skill's newly-measured §7.4 fact. `pass_list`'s
  `and deletable` is gone as M10 claimed.
