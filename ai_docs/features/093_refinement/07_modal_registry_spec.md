# 093 / wave 4 — One modal mechanism, and the confirm modal as its first client

The maintainer's question after the menus wave: the confirm-through-a-submenu shape
(`05_menus_spec.md` M5) read as a choice of delete kinds, not a confirmation; he asked why not
a modal, and whether the app has one reusable modal mechanism, because more modals are coming.
The assessment (chat, 2026-09-14, recorded here): the submenu is hover-reachable, carries no
consequence text and no chord hint, and is a shape no app the user knows confirms with; the
modal is right. And the mechanism is half there — `ui_primitives.modal_window` owns the imgui
chrome, but the ROSTER is hand-maintained in five places (a `PopupState` enum on `App`, one
`open_*` each, a `close_popup` dispatching by hand, eight `draw_*` calls listed in `ui.py`, a
hand-written table in the chrome gate), the revert dialog sits outside all of it, and the
conventions carry a warning about forgetting one of the five. This wave replaces the roster
with a registry the whole roster derives from, and lands the confirm modal on it.

Size: **high-blast-radius** (`app.py`, `ui.py`, `hotkeys.py`, every `popups/` module, the
copilot chat, three widgets, three tests): one pre-implementation reviewer, an opus
implementer, three post-implementation reviewers to convergence, sanitize.

## Goal

A modal is one `Modal(...)` value in its own module and one enum member; everything else —
the draw call, the Esc and Close funnel, the "at most one open" mutex, the chrome gate, the
"every member is drawn" gate — derives from the registry. A destructive verb confirms in a
modal that names its target and its consequence, wherever the verb is fired from.

## Out of scope

- The two confirms INSIDE a modal (Projects' delete row, Settings' library reset): a modal
  over a modal is not the mechanism's shape; their armed `danger_button` rows stay
  (`05_menus_spec.md` M5's second half). Trigger: a third in-modal confirm.
- The command palette (`is_palette_open`), non-modal by decision.
- The sticker grid's arm-and-wash (an exporter's; `05_menus_spec.md` out of scope).
- Modal sizes, labels and bodies: unchanged except where a decision below names one.

## Design decisions

**R1. `ModalId` replaces `PopupState`; `App.modal: ModalId | None` is the mutex.** A
`StrEnum` in `app.py` (the layer that owns the state): `EXAMPLES`, `HELP`, `SETTINGS`,
`PASS_SETTINGS`, `IMPORT_PASSES`, `EMOJI_PICKER`, `SHADER_LIB_PICKER`, `PROJECTS`,
`CONFIRM`. No `CLOSED` member: `None` is closed. `App.any_popup_open()` is `self.modal is
not None` and keeps its name (its callers are the render gates, not the roster).
`App._open_modal(id)` replaces `_open_popup`; the per-modal `open_*` verbs stay on `App`
(they set the modal's own state first, then open). `ui.py`'s three `PopupState` reads become
`ModalId` reads.

**R2. `popups/registry.py` holds the registry and the two verbs every surface uses.**

```python
@dataclass(frozen=True)
class Modal:
    id: ModalId
    label: str                                   # the imgui popup id, `"Help##help"`
    size: Callable[[App], tuple[float, float]]   # Examples computes its own
    body: Callable[[App], bool]                  # returns keep_open
    flags: int = 0
    fixed_size: bool = False
    before: Callable[[App], None] | None = None  # Pass settings' size constraints
    on_close: Callable[[App], None] | None = None  # the per-modal cleanup, ONE place
    owns_esc: Callable[[App], bool] | None = None  # an inline input has Esc: decline

MODALS: tuple[Modal, ...] = (examples.MODAL, help.MODAL, settings.MODAL, pass_settings.MODAL,
    import_passes.MODAL, emoji_picker.MODAL, lib_picker.MODAL, projects.MODAL, confirm.MODAL)
BY_ID: dict[ModalId, Modal]

def draw_modal(app: App) -> None      # the ONE call in ui.py's popup block
def close_modal(app: App) -> bool     # the ONE close funnel: Esc, and a body returning False
```

`draw_modal`: `modal = BY_ID.get(app.modal)`; if none, return; `modal.before(app)` if set;
`with modal_window(modal.label, modal.size(app), modal.flags, modal.fixed_size) as visible`;
if visible and `not modal.body(app)`: `close_modal(app)` + `imgui.close_current_popup()`.
`close_modal`: returns `False` without closing when `modal.owns_esc(app)`; else runs
`modal.on_close(app)` if set, sets `app.modal = None`, returns `True`. `registry.py` imports
`App` and every popup module; no popup module imports the registry (the popups import `App`
only, as today), and `app.py` never imports `popups` — so the registry is a leaf of the
popups layer, the way `menus.py` is of the commands layer. `hotkeys.py` imports `close_modal`;
`App.close_popup` is deleted. An `App` method that closes a modal programmatically after its
own cleanup (`close_pass_settings` after a rename commit, `close_import_passes`, the project
switch) sets `self.modal = None` itself — it IS the cleanup owner; a `popups/` module never
writes `app.modal`.

**R3. Each popup module is a body and a spec.** `draw_X(app)` goes; `_draw_body(app) -> bool`
stays (binding and returning `keep_open`, per the chrome rule) and `MODAL = Modal(...)` is a
module constant beside it. `on_close` values are the cleanups `close_popup` dispatched by
hand: `pass_settings` -> `App.close_pass_settings` (commits a pending rename / group, then
clears); `import_passes` -> `App.close_import_passes`; `emoji_picker` -> `App.close_emoji_picker`
(nulls the target, clears the query); `settings` -> `App.apply_editor_settings`; `projects` ->
`App.reset_projects_state`; `examples`, `help`, `lib_picker` none. `owns_esc`: `projects` ->
`App.projects_input_owns_esc`; `lib_picker` -> `app.shader_lib_files.inline_input_owns_esc`.
(The three `close_*` App methods stop writing the state field; they are cleanups.)

**R4. `popups/confirm.py`: the confirm modal, one client for every destructive verb.**

```python
@dataclass(frozen=True)
class ConfirmRequest:
    title: str            # "Delete pass blur?"  — names the target
    line: str             # "Its wiring and position are lost; the shader file stays."
    verb: str             # the button: "Delete", "Reset", "Clear", "Revert"
    on_confirm: Callable[[], None]
```

`App.confirm: ConfirmRequest | None`; `App.request_confirm(request)` sets it and opens
`ModalId.CONFIRM`. The body: the title (wrapped, `FG_TITLE`), the line as `caption_text`,
the `SPACE.MD` spacer, `danger_button(verb)` left and `standard_button("Cancel")` right;
Enter fires the verb (the modal is the deliberate step; the click that opened it was a mouse
gesture and no Enter is in flight), Esc cancels through `close_modal`. `on_close` nulls
`app.confirm`; confirming calls `on_confirm()` then returns `False`. Size `(380, 0)`, the
revert dialog's. The prose budget: `title` and `line` are authored at every call site, so the
gate's `_UNMEASURABLE` gains `confirm.py::_draw_body` with the reason, and each call site's
strings are literals the AST walk scores at their own sites (the per-site budget: the title
under the heading budget, the line under one clause).

**R5. The verbs own their confirm; the surfaces do not.** Every destructive verb becomes (or
already is) an `App` method that builds the request with its target's name and calls
`request_confirm`; a menu item, a bar item, a button, a chord and the palette all call that
one method, so the confirm is the same from every surface:

| Verb | `App` method (new or changed) | Title / line / verb |
|---|---|---|
| delete a pass | `delete_pass_confirmed(document_id, name)` — wraps today's `pass_list._delete_pass` (session delete + editor teardown), which moves to `App` | `Delete pass {name}?` / `Its wiring and position are lost; the shader file stays.` / `Delete` |
| delete a document | `delete_document_confirmed(document_id)` — the bar's `DELETE_DOCUMENT` callback and the tile's item | `Move {ui_name} to the trash?` / `Nothing in the app brings it back.` / `Delete` |
| reset a document | `reset_document_confirmed()` — the `RESET_DOCUMENT` callback and the Document tab's `danger_button` | `Reset {ui_name}?` / `Feedback histories, the clock and the script restart.` / `Reset` |
| delete a lib file / dir | `ShaderLibFileManager.delete_file` / `delete_dir` stay the verbs; the tree's items call `app.request_confirm` with the trash line | `Delete {name}?` / `It moves to .trash.` / `Delete` |
| clear the chat | `copilot_clear_chat_confirmed()` — the `CLEAR_COPILOT_CHAT` callback and the chat's button | `Clear the conversation?` / `The transcript and its checkpoints are dropped.` / `Clear` |
| revert a turn | `open_copilot_revert(msg)` builds the request (`copilot_revert_target` and `_draw_revert_modal` are deleted) | `Revert "{head}"?` / today's caption / `Revert` |

`ui_primitives.confirm_menu_item` is deleted; `CommandSpec.confirm_label` is deleted;
`command_menu_item` draws a plain item for every spec (the bar's `Delete document` gets its
`Alt+D` hint back); the palette offers every `in_palette` spec again
(`_register_palette_commands` loses the filter; `tests/test_menus.py`'s palette test is
replaced by one asserting every `in_palette` spec is offered). `06_command_system.md`'s rule
5 is rewritten: "a destructive verb confirms in the confirm modal, from every surface".
**Reverses `05_menus_spec.md` M5's submenu and its conventions bullet**, on the
maintainer's call; the bullet is rewritten to the modal rule with the reasons above.

**R6. Gates derive from the registry.** `tests/test_modal_chrome.py`'s domain is `MODALS`
(every `ModalId` has exactly one `Modal`, every `Modal.id` is a member — the structural half
that replaces `test_every_popup_state_has_a_draw_call`, which is deleted with the `ui.py`
draw list it parsed); for each `Modal`, `body` binds and returns `keep_open`, ends in the
action row with the spacer (the existing clauses); the "no hand-written close" clause becomes
"no `popups/` module assigns `app.modal`". `tests/test_import_dialog.py`'s structural check
on `close_popup` is repointed at `import_passes.MODAL.on_close is App.close_import_passes`.
Breaks to try and name in the commit: a `ModalId` member with no `Modal`; a `Modal` whose body
returns `ok`; a popup module writing `app.modal = None`; a confirm client whose `on_confirm`
is dropped (the confirm test's verb spy goes red); Esc on the confirm modal calling the verb
(the cancel test goes red).

**R7. Tests for the confirm.** Frame-driven, through the real surfaces: right-click a pass
tile, click `Delete`: `app.modal is ModalId.CONFIRM` and `app.confirm.title == "Delete pass
b?"`, the pass still exists; press Enter: `delete_pass` ran once (spied with `wraps`), the
modal is closed; the same through Esc: nothing ran, `app.confirm is None`. The bar's `Delete
document` and `Reset document`, the chat's `Clear`, the revert glyph: each opens the confirm
with its title and its verb spied. The lib tree's delete: one test through its menu.

## Files touched

- `shaderbox/app.py` — `ModalId`, `modal`, `confirm`, `_open_modal`, `request_confirm`, the
  three cleanups no longer writing the state, `delete_pass_confirmed`,
  `delete_document_confirmed`, `reset_document_confirmed`, `copilot_clear_chat_confirmed`,
  `open_copilot_revert` rebuilt; `close_popup`, `PopupState`, `copilot_revert_target` gone.
- `shaderbox/popups/registry.py` (new), `shaderbox/popups/confirm.py` (new).
- `shaderbox/popups/{examples,help,settings,pass_settings,import_passes,emoji_picker,projects}.py`,
  `popups/lib_picker/__init__.py` — `draw_*` gone, `MODAL` added.
- `shaderbox/ui.py` — one `draw_modal(app)`; the `ModalId` reads.
- `shaderbox/hotkeys.py` — `_handle_escape` through `close_modal`; the revert branch gone.
- `shaderbox/widgets/copilot_chat.py` — `_draw_revert_modal` / `_draw_revert_body` gone; the
  Clear button and the revert glyph through the App verbs.
- `shaderbox/widgets/pass_list.py` (`_delete_pass` -> `App`; the item), `widgets/document_grid.py`,
  `widgets/pass_graph.py` (the item sets), `popups/lib_picker/tree.py`, `tabs/document.py`
  (the reset button), `menus.py` and `commands.py` (`confirm_label` gone), `ui_primitives.py`
  (`confirm_menu_item` gone).
- Tests: `test_modal_chrome.py`, `test_project_management.py`, `test_import_dialog.py`,
  `test_menus.py`, `test_ui_prose_budget.py` (the `_UNMEASURABLE` row), a new
  `tests/test_confirm.py`; every test that sets `popup_state` / `PopupState` (eight files)
  repointed.
- Docs: `conventions.md` (the popups bullet rewritten around the registry; the M5 bullet
  reversed to the modal rule; the `InlineInput` bullet unchanged), `dev_flow.md`'s module map
  (`popups/registry.py`, `popups/confirm.py`, the popups entry), `.claude/skills/imgui-ui/SKILL.md`
  §7.2 and §7.3 (the wrapper shape is now body + `Modal`), `05_menus_spec.md` (M5, M8, M9
  pointers), `06_command_system.md` (rule 5), `01_spec.md`'s wave list, `00_findings.md` row
  18 (his question as the finding), the roadmap banner.
- `projects/dev/` — nothing persisted changes.

## Verification

| Guarantee | Test | Kind |
|---|---|---|
| R1/R6: the registry is the roster | `{m.id for m in MODALS} == set(ModalId)` and the ids are unique; break: add a member | pure |
| R2: one draw call | `ui.py`'s AST calls `draw_modal` once and imports no `draw_*` from `popups`; break: re-add `draw_help(app)` | pure |
| R2: no popup writes the mutex | an AST walk over `popups/`: no assignment whose target is `app.modal`; break: write one in `help.py` | pure |
| R2: the close funnel | for each `Modal` with `on_close`: open it (its `open_*`), call `close_modal(app)`, the cleanup ran (spied) and `app.modal is None`; for each with `owns_esc`: arm the input, `close_modal` returns `False` and the modal stays | app fixture |
| R3: chrome | the existing `test_modal_chrome` clauses over `MODALS` | pure |
| R4/R5/R7: the confirm from every surface | as R7 lists, one test per client, each with its verb spied and the Esc case | frame-driven |
| R5: the palette offers every spec | `app._palette_command_names` covers every `in_palette` spec | app fixture |
| R5: the bar's Delete document is a plain item with its hint | the M1 label-set test; the item is not a submenu (spy on `begin_menu` inside the Document menu: zero calls) | frame-driven |
| Deletions | no `PopupState`, `close_popup`, `copilot_revert_target`, `confirm_menu_item`, `confirm_label` anywhere in `shaderbox/` or `tests/` | pure |

## Open questions for the user

None. Enter confirming (R4) is the one call he may want reversed; it is one line.

## Review history

(filled by the flow)
