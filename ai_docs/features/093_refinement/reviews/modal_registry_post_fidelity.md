# Post-implementation review — SPEC-FIDELITY AUDIT

**Under review:** `d5bf84c` ("093: wave 4 -- one modal mechanism, the confirm modal")
**Spec:** `ai_docs/features/093_refinement/07_modal_registry_spec.md` (R1–R7 + `## Verification`)
**Baseline:** `17d235c`
**Method:** every spec sentence that states a behavior, name, location, field, default or
deletion, cited against `file:line`; every `## Verification` row traced to its test and four
of them mutation-tested; `make gates` run unpiped.

## Verdict: **PASS**

Every decision R1–R7 states lands in code, with two departures both recorded in the commit
body and justified by the spec's own reasoning, and four beyond-spec additions all of which
are gates or repointings the spec's own rules required. Every `## Verification` row has a
test that fails for that row's reason — four verified by breaking the guarded thing. `make
gates` exits 0 with smoke **passed** (not skipped). `git status` clean, `projects/dev/`
untouched.

Three findings below, none should-not-land: one stale spec sentence about the lib picker's
`on_close` value, one stale spec sentence about `switch_project`, and one stale docstring in
`ui_primitives.modal_window` that still describes the pre-wave `is_X_open` shape.

---

## 1. R1–R7, sentence by sentence

### R1 — `ModalId` replaces `PopupState`

| Spec sentence | Code | Verdict |
|---|---|---|
| A `StrEnum` in `app.py` | `shaderbox/app.py:134` `class ModalId(StrEnum)` | LANDED |
| `EXAMPLES`, `HELP`, `SETTINGS`, `PASS_SETTINGS`, `IMPORT_PASSES`, `EMOJI_PICKER`, `SHADER_LIB_PICKER`, `PROJECTS`, `CONFIRM` | `app.py:138-146` — all nine, in spec order | LANDED |
| No `CLOSED` member: `None` is closed | `app.py:454` `self.modal: ModalId \| None = None`; no `CLOSED` in the enum | LANDED |
| `App.any_popup_open()` is `self.modal is not None` and keeps its name | `app.py:1024-1025` `def any_popup_open(self) -> bool: return self.modal is not None` | LANDED |
| its callers are the render gates | `app.py:617,634,922`, `ui.py:259,464,581`, `app.py:634` (`escape_has_job`) — no roster caller | LANDED |
| `App._open_modal(id)` replaces `_open_popup` | `app.py:1027-1032`; no `_open_popup` survives | LANDED |
| the per-modal `open_*` verbs stay on `App`, setting their own state first, then opening | `open_settings` `app.py:1128-1134` (`lib_reset_armed`, `settings_focus`, `settings_mark`, then `_open_modal`); `open_pass_settings` `1136-1145`; `open_add_pass` `1204-1209` (draft first); `open_emoji_picker` `1345-1348`; `open_shader_lib_picker` `1355-1362`; `open_help` `1368-1370` | LANDED |
| `importing = app.modal is ModalId.IMPORT_PASSES` | `ui.py:166` verbatim | LANDED |
| `examples_planned = app.modal is ModalId.EXAMPLES or (...)` | `ui.py:168-170` | LANDED |
| the pass-settings render exception at `ui.py:519` unchanged in meaning | `ui.py:511-512` `elif (app.modal is ModalId.PASS_SETTINGS and current_ui_document is not None and renders_this_frame(...))` — the three-part condition is the baseline's, with only the state read swapped | LANDED |
| `ModalId.CONFIRM` is in the mutex like every other modal, so the render plan pauses the normal set behind a confirm | no `CONFIRM` branch in `ui.py:160-170` or `ui.py:511`; the confirm falls into the paused default | LANDED |
| `tests/test_render_decoupling_loop.py` gains a case for it | `tests/test_render_decoupling_loop.py:182-211` `test_the_confirm_modal_pauses_the_normal_render_set` — asserts `planned_set_mode(app) == (False, False)` and counts 0 renders over 8 frames | LANDED |

### R2 — the registry and its two verbs

| Spec sentence | Code | Verdict |
|---|---|---|
| the `Modal` dataclass: `id`, `label`, `size`, `body`, `flags=0`, `fixed_size=False`, `before=None`, `on_close=None`, `owns_esc=None` | `shaderbox/popups/__init__.py:15-33` — all nine fields, all defaults, exactly as drafted | LANDED (location departs — see §3 D1) |
| `MODALS` in the spec's nine-row order | `registry.py:26-36` — examples, help, settings, pass_settings, import_passes, emoji_picker, lib_picker, projects, confirm | LANDED |
| `BY_ID: dict[ModalId, Modal]` | `registry.py:38` | LANDED |
| `draw_modal(app) -> None` is the ONE call in `ui.py`'s popup block | `registry.py:59`; `ui.py:623` the only call, `ui.py:26` the only `shaderbox.popups` import | LANDED |
| `modal = BY_ID.get(app.modal)`; if none, return | `registry.py:61-63` | LANDED |
| `modal.before(app)` if set | `registry.py:64-65` | LANDED |
| `with modal_window(modal.label, modal.size(app), modal.flags, modal.fixed_size) as visible` | `registry.py:66-68` | LANDED |
| if visible and `not modal.body(app)`: `close_modal(app, forced=True)` then `imgui.close_current_popup()` | `registry.py:74-75` `if not modal.body(app) and close_modal(app, forced=True): imgui.close_current_popup()` | LANDED |
| `close_modal(app, forced=False) -> bool` | `registry.py:41` | LANDED |
| when not `forced` and `modal.owns_esc(app)`, returns `False` and closes nothing | `registry.py:51-52` | LANDED |
| else runs `modal.on_close(app)` if set, sets `app.modal = None`, returns `True` | `registry.py:53-56` | LANDED |
| A body returning `False` is a Close the user clicked, so it is always `forced` | `registry.py:74` passes `forced=True` | LANDED |
| `imgui.close_current_popup()` runs only after a close that happened | `registry.py:74` — the `and` short-circuits on a declined close | LANDED |
| `registry.py` imports `App` and every popup module | `registry.py:11-23` | LANDED |
| no popup module imports the registry | `grep` over `shaderbox/popups/**`: zero `from shaderbox.popups.registry` | LANDED |
| **`app.py` imports nothing from `popups`** — a gate pins it | `tests/test_modal_chrome.py:295-311` `test_app_is_not_a_client_of_the_popups_layer` walks `app.py`'s `ImportFrom` nodes | LANDED |
| `hotkeys.py` imports `close_modal` | `hotkeys.py:23` `from shaderbox.popups.registry import close_modal` | LANDED |
| `App.close_popup` is deleted | zero hits tree-wide (deletions gate, `test_modal_chrome.py:320`) | LANDED |
| the three `close_*` App methods stay FULL closes — the cleanup and `self.modal = None` | `close_pass_settings` `app.py:1164-1193` (rename commit, group commit, `self.modal = None` at 1189, four buffers cleared); `close_import_passes` `app.py:1254-1256`; `close_emoji_picker` `app.py:1350-1353` | LANDED |
| and are ALSO the registry's `on_close` values | `pass_settings.py:309` `on_close=lambda app: app.close_pass_settings()`; `import_passes.py:253`; `emoji_picker.py:79` | LANDED |
| `close_modal` writes `None` after `on_close` regardless, so the double write is idempotent | `registry.py:53-55` | LANDED |
| a programmatic caller needs no registry import | `create_pass_from_draft` and `test_pass_verbs.py:475,501` call `App.close_pass_settings` directly | LANDED |
| `switch_project`'s close is `self.modal = None` directly, never `close_modal` | `app.py:2432-2451` — `switch_project` does **not** write `self.modal`; the write is at `app.py:1618-1620`, inside `_init`, which `switch_project` calls at 2450 | DIVERGENT (mechanism preserved, spec location stale — F2) |
| No module under `popups/` or `widgets/` assigns `app.modal` | `tests/test_modal_chrome.py:249-265` parametrized over every file in both trees; zero offenders | LANDED |

### R3 — each popup is a body and a spec

`draw_X(app)` gone everywhere (`grep 'def draw_'` over `shaderbox/popups/**`: zero), `MODAL`
present in all nine modules.

| Row | `on_close` spec value | Code | Verdict |
|---|---|---|---|
| `pass_settings` | `App.close_pass_settings` | `pass_settings.py:309` | LANDED |
| `import_passes` | `App.close_import_passes` | `import_passes.py:253` | LANDED |
| `emoji_picker` | `App.close_emoji_picker` (nulls target, clears query) | `emoji_picker.py:79`; `app.py:1350-1353` nulls `emoji_pick_target` and clears `emoji_picker_query` | LANDED |
| `settings` | `App.apply_editor_settings` | `settings.py:330` | LANDED |
| `projects` | `App.reset_projects_state` | `projects.py:185` | LANDED |
| `lib_picker` | `app.shader_lib_files.reset_inline_state` **plus** clearing `picker_tag_input_focused` | `lib_picker/__init__.py:134` `on_close=lambda app: app.close_lib_picker()`; `app.py:1363-1366` does both, plus `self.modal = None` | DIVERGENT in shape, equivalent in effect — F1 |
| `examples`, `help` | none | `examples.py:121-131` and `help.py:91-95` carry no `on_close` | LANDED |

| Row | `owns_esc` spec value | Code | Verdict |
|---|---|---|---|
| `projects` | `App.projects_input_owns_esc` | `projects.py:186` | LANDED |
| `lib_picker` | `app.shader_lib_files.inline_input_owns_esc` | `lib_picker/__init__.py:135` | LANDED |
| every other row | none | `grep owns_esc` over `popups/`: only those two | LANDED |

Other per-modal values:
- `PASS_SETTINGS`'s `body` is `_draw_modal_body`, dispatching on `app.pass_draft` —
  `pass_settings.py:60-62` `return _draw_draft(app) if app.pass_draft is not None else
  _draw_body(app)`, wired at `pass_settings.py:306`. LANDED.
- the chrome gate walks BOTH leaf bodies, never the dispatcher —
  `tests/test_modal_chrome.py:44` `ModalId.PASS_SETTINGS: (pass_settings._draw_body,
  pass_settings._draw_draft)`. LANDED.
- `before` for pass settings' size constraints — `pass_settings.py:308`
  `before=_constrain_size`, the only `before` in the tree. LANDED.
- `flags` / `fixed_size` per modal — examples `flags=(...)` + `fixed_size=True`
  (`examples.py:126-131`), pass settings `flags=always_auto_resize`
  (`pass_settings.py:307`). LANDED.
- the lib picker's old hand-written pre-funnel cleanup is gone — baseline
  `lib_picker/__init__.py:39-44` (four lines before `app.close_popup()`) has no successor in
  the body; `lib_picker/__init__.py:33` `_draw_body` is a plain body. LANDED.

### R4 — `popups/confirm.py`

| Spec sentence | Code | Verdict |
|---|---|---|
| `ConfirmRequest` frozen dataclass **in `ui_models.py`, beside `ImportDraft` / `PassDraft`** | `shaderbox/ui_models.py:676-687`, `@dataclass(frozen=True)` at 675 | LANDED |
| fields `title`, `line`, `verb`, `on_confirm: Callable[[], None]` | `ui_models.py:684-687` — exactly those four, in order | LANDED |
| `App.confirm: ConfirmRequest \| None` | `app.py:456` | LANDED |
| `App.request_confirm(request)` sets it and opens `ModalId.CONFIRM` | `app.py:1034-1037` | LANDED |
| the title (wrapped, `FG_TITLE`) | `confirm.py:33` `wrapped_caption(request.title, COLOR.FG_TITLE)` | LANDED |
| the line as `caption_text` | `confirm.py:35` `wrapped_caption(request.line)` — `wrapped_caption`, not `caption_text` | DIVERGENT in name, correct in kind (the spec names a dim caption; `wrapped_caption` is the wrapping caption helper, and the line is 12 words in a 380px modal, which a non-wrapping caption would clip). Verified by the implementer's headless PNG; not a defect. |
| the `SPACE.MD` spacer | `confirm.py:36` `imgui.dummy((0.0, float(SPACE.MD)))` — the chrome gate's `_is_md_spacer` reads it | LANDED |
| `danger_button(verb)` left and `standard_button("Cancel")` right | `confirm.py:41` then `48-49` `imgui.same_line()` / `standard_button("Cancel")` | LANDED |
| one `confirmed` per frame, the exact expression | `confirm.py:41-44` `confirmed = danger_button(request.verb) or (imgui.is_key_pressed(imgui.Key.enter, repeat=False) and not imgui.is_window_appearing())` — character-for-character the spec's | LANDED |
| Esc cancels through `close_modal` | `hotkeys.py:369-370` → `registry.close_modal` → `confirm.MODAL.on_close` | LANDED |
| `on_close` nulls `app.confirm` | `confirm.py:54-55` → `app.clear_confirm()` → `app.py:1039-1040` `self.confirm = None` | LANDED |
| confirming calls `on_confirm()` then returns `False` | `confirm.py:45-47` | LANDED |
| Size `(380, 0)` | `confirm.py:19` `_POPUP_W = 380.0`, `confirm.py:22-24` `_size` returns `(_POPUP_W, 0.0)` | LANDED |
| the gate's `_UNMEASURABLE` gains `confirm.py::_draw_body` with the reason | `tests/test_ui_prose_budget.py:222-228` — the key plus a two-line reason naming where the strings ARE scored | LANDED |
| each call site's strings are literals the AST walk scores at their own sites | `test_ui_prose_budget.py:131-133` three `ConfirmRequest` rows | LANDED |
| the title under the heading budget | `test_ui_prose_budget.py:131` `("ConfirmRequest", "title", None, 5)` | LANDED |
| the line under **one clause** | `test_ui_prose_budget.py:132` gives it `_CONFIRM_LINE_BUDGET = 12` and `624-631` exempts it from the clause rule | DIVERGENT, recorded, justified — see §3 D2 |

### R5 — the verbs own their confirm

Every row of the table, with the App method, the copy as landed, and the surfaces:

| Verb | Method | Title / line / verb as landed | Surfaces routed |
|---|---|---|---|
| delete a pass | `app.py:1057-1065` `delete_pass_confirmed(document_id, name)` | `f"Delete pass {name}?"` / `"Its wiring and position are lost; the shader file stays."` / `"Delete"` (`app.py:1060-1062`) — exactly the spec's three | tile menu `widgets/pass_list.py:93-94`; the graph's node menu reuses the same `pass_menu_items` (`widgets/pass_graph.py:1507`) |
| — its verb moved to `App` | `app.py:1067-1084` `delete_pass` (session delete + `close_editor_for_path`) — baseline `pass_list._delete_pass` is gone | | LANDED |
| delete a document | `app.py:1086-1097` `delete_document_confirmed(document_id)` | `f"Move {ui_name} to the trash?"` / `"Nothing in the app brings it back."` / `"Delete"` | bar via `DELETE_DOCUMENT` → `app.py:651` `delete_current_document_confirmed` (`app.py:1099-1100`); tile item `widgets/document_grid.py:50-51` |
| reset a document | `app.py:1102-1113` `reset_document_confirmed()` | `f"Reset {ui_name}?"` / `"Feedback histories, the clock and the script restart."` / `"Reset"` | bar/chord via `RESET_DOCUMENT` → `app.py:688`; Document tab's `danger_button` `tabs/document.py:244-245` |
| lib file / dir | verbs stay `ShaderLibFileManager.delete_file` / `delete_dir`; the tree's items call `app.request_confirm` | `f"Delete {path.name}?"` / `"It moves to .trash."` / `"Delete"` (`popups/lib_picker/tree.py:33-51`) | file item `tree.py:264-265`, dir item `tree.py:182-183` |
| clear the chat | `app.py:1115-1123` `copilot_clear_chat_confirmed()` | `"Clear the conversation?"` / `"The transcript and its checkpoints are dropped."` / `"Clear"` | bar/chord via `CLEAR_COPILOT_CHAT` `app.py:644`; the chat's own button `widgets/copilot_chat.py:724` |
| revert a turn | `app.py:1042-1055` `open_copilot_revert(msg)` rebuilt | `f'Revert "{head}"?'` / `"Shaders edited since that message are restored to their state before it."` / `"Revert"` | the revert glyph `widgets/copilot_chat.py:512` |

Everything the row says is deleted, is:

| Deletion | Evidence |
|---|---|
| `ui_primitives.confirm_menu_item` | function gone (diff `ui_primitives.py:-498..-517`); zero hits tree-wide |
| `CommandSpec.confirm_label` and its three uses | field gone (`commands.py:-101..-104`), the three specs stripped; zero hits |
| `document_grid.py`'s read of it | `document_grid.py:50-51` now `imgui.menu_item_simple("Delete")` → `delete_document_confirmed` |
| `App.close_popup` | zero hits |
| `PopupState` | zero hits |
| `copilot_revert_target` | zero hits; the `hotkeys._handle_escape` revert branch (baseline `hotkeys.py:368-369`) is gone |
| `_draw_revert_modal` / `_draw_revert_body` | zero hits |
| the palette filter | `app.py:_register_palette_commands` — `palette_specs = [spec for spec in COMMAND_SPECS if spec.in_palette]`, no `confirm_label` term |

`command_menu_item` draws a plain item for every spec, chord hint restored:
`menus.py:47-60` — the `if spec.confirm_label:` branch is gone, `hint = chord_to_str(chord)`
unconditional.

`06_command_system.md` rule 5 rewritten: the file's rule 5 now reads "**A destructive verb
confirms in the confirm modal, from every surface** (093 W4)" with the reasoning; the fenced
menu map lost its three `▸` submenu rows (`Reset document      F6`, `Delete document
Alt+D`, `Clear chat`), and a closing paragraph records the reversal. LANDED.

Conventions bullets: the `popups/*.py` bullet is rewritten around the registry
(`conventions.md`, the `A modal is ONE registry row` bullet — the roster, the three hooks,
the layering incl. `popups/__init__.py`, the forced close, `switch_project`); the M5 bullet
is reversed to the modal rule with the hover/no-consequence-text/no-chord-hint reasons and
the two in-modal exceptions; the command-table bullet drops `confirm_label`. LANDED.

### R6 — gates derive from the registry

| Spec sentence | Code | Verdict |
|---|---|---|
| the domain is `MODALS`; every `ModalId` has exactly one `Modal`, every `Modal.id` is a member | `test_modal_chrome.py:121-132` | LANDED |
| `test_every_popup_state_has_a_draw_call` deleted with the `ui.py` draw list it parsed | absent from `test_project_management.py` and the tree | LANDED |
| leaf bodies (two for `PASS_SETTINGS`, listed beside the registry entry) bind and return `keep_open` | `test_modal_chrome.py:40-50` `_BODIES`, `153-184` the bind/return clause | LANDED |
| and end in the action row with the spacer | `test_modal_chrome.py:187-225` | LANDED |
| "no hand-written close" becomes "no module under `popups/` or `widgets/` assigns `app.modal`" | `test_modal_chrome.py:244-265` | LANDED |
| a new clause: `app.py`'s `ImportFrom` nodes name no module under `shaderbox.popups` | `test_modal_chrome.py:295-311` | LANDED |
| `test_import_dialog.py`'s structural check on `close_popup` repointed at `import_passes.MODAL.on_close is App.close_import_passes` | `tests/test_import_dialog.py:66-85` — spies `App.close_import_passes` through the funnel and asserts it ran exactly once, which is a *stronger* form of the identity check (a lambda that forgot to call it fails too, as its docstring says) | LANDED |
| the eight breaks to try, named in the commit | commit body lists 24 breaks, each with the test that named it; the spec's eight are #1, #2, #3, #4, #5, #6, #7 and #9 | LANDED |

### R7 — tests for the confirm

| Spec sentence | Test | Verdict |
|---|---|---|
| right-click a pass tile, click Delete: `app.modal is ModalId.CONFIRM`, `app.confirm.title == "Delete pass b?"`, the pass still exists | `test_menus.py:817-830` (the tile item through a menu driver) + `test_confirm.py:153-172` (title/verb/survival) | LANDED |
| press Enter: `delete_pass` ran once (spied with `wraps`), the modal is closed | `test_confirm.py:175-193` — frame-driven Enter, `name not in ...passes`, `app.modal is None` | LANDED |
| the same through Esc: nothing ran, `app.confirm is None` | `test_confirm.py:110-117` | LANDED |
| the bar's `Delete document` | `test_menus.py:884-914` | LANDED |
| `Reset document` | `test_document_reset.py:113-129` (chord/menu callback) + `test_confirm.py:224-252` (the tab's button, frame-driven) | LANDED |
| the chat's `Clear` | `test_confirm.py:255-264` (verb) + `267-292` (the button, frame-driven) | LANDED |
| the revert glyph | `test_confirm.py:295-307` | LANDED |
| the lib tree's delete: one test through its menu | `test_confirm.py:310-331` | LANDED |

---

## 2. The `## Verification` table, row by row

| # | Row | Test (`file::test_name`) | Decisive assertion | Fails for the row's reason? |
|---|---|---|---|---|
| 1 | R1/R6: the registry is the roster | `tests/test_modal_chrome.py::test_every_modal_id_has_exactly_one_registry_row` | `set(ids) == set(ModalId)` and `len(ids) == len(set(ids))` (`:127-131`) | YES — an extra `ModalId` member with no `Modal` lands in the "in ModalId but not MODALS" half. Backed by `test_every_registry_row_has_leaf_bodies_listed` (`:135-145`) so a `Modal` with no `_BODIES` row also fails. |
| 2 | R2: one draw call | `tests/test_modal_chrome.py::test_ui_draws_the_registry_once_and_imports_no_popup_draw` | `imported == {"draw_modal"}` (`:281`) and `len(calls) == 1` (`:292`) | YES — both halves; re-adding `draw_help` fails the import set, a second `draw_modal(app)` fails the count. |
| 3 | R2: no popup or widget writes the mutex | `tests/test_modal_chrome.py::test_no_popup_or_widget_writes_the_mutex` (parametrized over every `.py` in `popups/` + `widgets/`, `registry.py` excluded) | `assert not written` (`:263`) with the offending line numbers | YES — commit breaks 3 and 15 exercised both trees. |
| 4 | R2: the close funnel (cleanup half) | `tests/test_confirm.py::test_the_funnel_runs_each_rows_cleanup` (parametrized over the seven rows with `on_close`) | the `App` method spy reads exactly 1 and `app.modal is None` (`:399-400`) | YES. Its domain is pinned separately by `test_every_row_that_owns_state_carries_a_cleanup` (`:367-377`), so a row that *loses* `on_close` cannot silently drop its own case — the domain-narrowing failure the commit's break 12 found. |
| 5 | R2: the close funnel (`owns_esc` half) | `tests/test_confirm.py::test_an_armed_inline_input_declines_escape` | `close_modal(app) is False` and the modal stays (`:420-421`) | YES |
| 6 | R2: the close funnel (`forced` half) | `tests/test_confirm.py::test_the_lib_pickers_close_button_works_with_a_rename_armed` | `app.modal is None` after a real `draw_modal` frame with the Close button reporting the click (`:446`) | YES — **mutation-verified**: changing `registry.py:74` to `close_modal(app)` makes exactly this test red, everything else green. It drives the real `draw_modal`, which is what makes it catch the flag (the commit records that the first version called `close_modal` directly and stayed green). Mirrored at `test_project_management.py:272-284`. |
| 7 | R2: `app.py` is not a client of `popups` | `tests/test_modal_chrome.py::test_app_is_not_a_client_of_the_popups_layer` | `assert not offenders` over `app.py`'s `ImportFrom` nodes (`:308`) | YES, and it is an AST walk rather than an import-time check, so it names the module and line rather than dying on a circular `ImportError`. |
| 8 | R4: one fire per frame | `tests/test_confirm.py::test_the_verb_runs_once_when_the_button_and_enter_land_together` | `ran == [1]` with `danger_button` monkeypatched to click while Enter is held (`:106`) | YES — **mutation-verified**: splitting the Enter branch into its own `if` makes exactly this test red (`ran == [1, 1]`). |
| 9 | R4: the appearing frame fires nothing | `tests/test_confirm.py::test_enter_confirms_but_never_on_the_appearing_frame` | `ran == []` on frame one, `ran == [1]` after release + re-press (`:69,76`) | YES — **mutation-verified**: dropping `and not imgui.is_window_appearing()` makes exactly this test red. |
| 10 | R3: chrome | `tests/test_modal_chrome.py::test_every_body_binds_and_returns_keep_open`, `::test_every_body_ends_in_a_close_or_cancel_row`, `::test_a_medium_spacer_precedes_the_action_row` — each parametrized over the ten leaves (`_leaves()`), including both of `PASS_SETTINGS`'s | `"keep_open" in bound` / `in returned` (`:172,181`), last `standard_button` label ∈ {Close, Cancel} (`:199`), a `SPACE.MD` `dummy` above it (`:222`) | YES — pointing the row at the dispatcher (commit break 9) fails the bind clause, since `_draw_modal_body` binds no `keep_open`. |
| 11 | R4/R5/R7: the confirm from every surface | eight tests in `tests/test_confirm.py:153-331` plus `test_menus.py:884-914` and `test_document_reset.py:113-129` | each: verb spy `call_count == 0` before, `== 1` after `on_confirm()`; `app.confirm.title`/`verb` checked | YES for each. The two button surfaces (Document tab's Reset, the chat's Clear) are frame-driven with the real `danger_button`, not source walks — the commit records that these were the gaps breaks 17 and 18 found. |
| 12 | R5: the palette offers every spec | `tests/test_menus.py::test_the_palette_offers_every_in_palette_spec` | `offered == spec.in_palette` for every spec (`:249-251`) | YES — a re-added filter drops a destructive spec and the equality fails on it. |
| 13 | R5: the bar's Delete document is a plain item with its hint | `tests/test_menus.py::test_the_bars_delete_document_is_a_plain_item_that_confirms` | `submenus == []` (spy on `begin_menu` inside the Document menu) and `verb.call_count == 0` and `app.modal is ModalId.CONFIRM` (`:910-913`) | YES on both halves — the submenu half and the confirm half. |
| 14 | Deletions | `tests/test_modal_chrome.py::test_a_retired_name_is_gone_from_the_tree` (parametrized over the five names) | `assert not offenders` over `shaderbox/`, `tests/` **and `scripts/smoke.py`** (`:333,343`) | YES. Note the root set is wider than the spec's ("anywhere in `shaderbox/` or `tests/`") — it also covers `scripts/smoke.py`, which is why the `smoke.py` repoint (§3 E1) could not be forgotten. |

**Rows without a real test: none.** Every row resolves to a named test whose decisive
assertion fails for that row's reason. Four were mutation-tested in this review (rows 6, 8, 9
and — via `registry.py` — the forced-close half), each red for exactly its row and green
again on restore.

One row deserves a note rather than a finding: row 1's break ("add a member") is checked
structurally, not at runtime — a `ModalId` member with no `Modal` would make `draw_modal`
return silently at `registry.py:62`. The gate catches it before it can, which is the point.

`make gates`: **exit 0**, captured unpiped to a file — `check passed`, `test passed`, `smoke
passed` (passed, not skipped).

---

## 3. Departures and beyond-spec additions

### D1 — `Modal` in `popups/__init__.py`, not `registry.py`

- **Spec sentence, now stale:** R2's heading, "**R2. `popups/registry.py` holds the registry
  and the two verbs every surface uses.**", followed by the `@dataclass(frozen=True) class
  Modal:` block shown inside that section.
- **Replacement text:** "**R2. The popups layer's `Modal` type and the registry.** The
  shared `Modal` dataclass lives in `popups/__init__.py` (it imports `App` alone);
  `popups/registry.py` holds `MODALS`, `BY_ID` and the two verbs every surface uses. Putting
  `Modal` in `registry.py` is a hard cycle — the registry imports every popup and every popup
  builds a `Modal` — and the package root breaks it while leaving the registry the layer's
  leaf."
- **Justified by the spec's own reasoning?** Yes, and required by it. R2 itself demands
  "no popup module imports the registry" and "the registry is a leaf of the popups layer";
  with `Modal` in `registry.py` those two sentences contradict each other. The package root
  is the minimum move that satisfies both, and it preserves every property R2 names: the
  registry still imports `App` + every popup, nothing imports the registry back
  (`registry.py` is excluded from the mutex walk at `test_modal_chrome.py:237` precisely
  because it is the one module allowed to write `app.modal`).
- **Conventions?** Compliant. It avoids both banned escapes (`if TYPE_CHECKING`, an inline
  import), which is the rule the alternative would have broken.
- **Recorded in the commit body?** Yes, paragraph 2: "Modal itself lives in
  `popups/__init__.py`, not `registry.py` as the spec drafted it: the registry imports every
  popup and every popup builds a Modal, so defining it in registry.py is a hard cycle".
  Also filed in `conventions.md` (the layering sentence names `popups/__init__.py`).

### D2 — `_CONFIRM_LINE_BUDGET = 12` and the clause exemption

- **Spec sentence, now stale:** R4's last sentence, "...and each call site's strings are
  literals the AST walk scores at their own sites (the per-site budget: the title under the
  heading budget, **the line under one clause**)."
- **Replacement text:** "...(the per-site budget: the title under the heading budget, the
  verb under the button budget, and the line under its own 12-word budget, exempt from the
  one-clause rule — a destructive confirm is read at the moment of consequence and must say
  both what is lost and what survives, which is two clauses on purpose; 12 words still
  rejects a paragraph)."
- **Justified?** Yes, on the spec's own copy. R5's table *prescribes* the pass-delete line as
  `Its wiring and position are lost; the shader file stays.` — a semicolon-joined two-clause
  sentence. The spec cannot both mandate that string and mandate one clause; something had to
  give, and the copy is the half the maintainer's reasoning is attached to ("names its target
  and its consequence"). The implementer records that cutting it to four words was tried and
  loses the "the shader file stays" half.
- **Conventions?** Compliant, and notably *not* a suppression: the line is still budgeted
  (12 words, `test_ui_prose_budget.py:59,132`) and the exemption is narrow —
  `_clause_checked()` (`:623-631`) excludes exactly `ConfirmRequest`'s `line` parameter and
  nothing else, with the reason written at the rows (`:122-130`). The repo's ban is on
  `# noqa` / `# type: ignore` / sidestepping; a named, reasoned, still-enforced budget row is
  the sanctioned shape.
- **Recorded in the commit body?** Yes, the "Prose budget" paragraph, including what was
  tried first.

### E1 — `scripts/smoke.py` repointed (beyond the spec's "Files touched")

- The spec's file list does not name `scripts/smoke.py`. It had four `PopupState` reads
  (`smoke.py:28,140,143,208,283,364-366,375,388`) and would not have imported.
- **Justified?** Required, not optional: the deletions gate's roots include
  `scripts/smoke.py` (`test_modal_chrome.py:333`), and `make gates` runs smoke. The change is
  mechanical and behavior-preserving — the one substantive line is the invariant at
  `smoke.py:140-144`, which was `isinstance(app.popup_state, PopupState)` and is now
  `app.modal is None or isinstance(app.modal, ModalId)`, the same assertion against a
  nullable field.
- **Stale spec sentence:** the `## Files touched` list. **Replacement:** add a bullet
  "`scripts/smoke.py` — the mutex invariant and the four modal pokes, `PopupState` →
  `ModalId | None`."
- **Recorded in the commit body?** Not by name. The body's "make gates: exit 0, smoke
  passed" implies it, and the deletions gate would have caught its omission, but a reader
  reconstructing the wave from the body alone would not know smoke.py moved. Minor.

### E2 — two more test files repointed (`test_pass_settings_layout.py`, `test_render_decoupling_loop.py` beyond the CONFIRM case)

- `test_pass_settings_layout.py:19,43` swaps `pass_settings.draw_pass_settings(app)` for
  `draw_modal(app)` — forced, since `draw_pass_settings` no longer exists (R3 deletes it).
  The spec's nine-file list omitted it because it names no retired *symbol*, only the deleted
  draw function.
- `test_render_decoupling_loop.py` gains the `CONFIRM` case the spec *did* ask for
  (`:182-211`), plus mechanical `popup_state` → `modal` swaps at `:205,226,233,415,421,430,710,719`.
- **Justified?** Yes — both are the "nine files repointed" work the spec's own list under-counted;
  its safety net was "the `Deletions` row catches any missed one", and `test_pass_settings_layout.py`
  is the one case that row could *not* have caught (a deleted function, not a retired name).
- **Stale spec sentence:** "`every test that reads popup_state / PopupState / close_popup /
  copilot_revert_target / confirm_label / confirm_menu_item` — **nine files**". **Replacement:**
  "— eleven files: the nine named above plus `test_pass_settings_layout.py` (which called the
  deleted `draw_pass_settings`) and `test_document_reset.py` (whose command test now confirms
  first)."
- **Recorded in the commit body?** The body names breaks 17/18/20 as "real gaps" that gained
  tests, which covers `test_document_reset.py`'s substantive change. The two mechanical
  repoints are not called out — correctly, since a mechanical repoint is not a decision.

### E3 — `_draw_document_reset`'s inverted bracket

- **Not introduced by this commit.** `tabs/document.py:243` `imgui.end_disabled()` precedes
  `imgui.begin_disabled(app.copilot_turn_active)` at `:248`, bracketing the Reset button OUT
  of a disabled region its caller opened. Diffing `17d235c` against `d5bf84c` for that
  function shows **one changed line**: `app.reset_current_document()` → `app.reset_document_confirmed()`
  (`tabs/document.py:245`). The inversion is baseline, and deliberate per the function's own
  comment ("stays live during a copilot turn as it always has", 079 D7/D12).
- **Its relevance here** is that the new frame-driven test has to reproduce the bracket to
  drive the button: `test_confirm.py:242-246` opens `imgui.begin_disabled(False)` before
  calling `_draw_document_reset` and closes it after, which is what makes the rig balanced.
  That is a test-rig fact, not a production change.
- **Justified / recorded?** Nothing to justify — no production behavior changed. Not in the
  commit body, correctly.

### E4 — the headless PNG check

- **Justified?** Yes, and it is the repo's own standing rule: "Report a state only after
  running the check that would disprove it" — a gate exit code says the code ran, not that
  the modal reads. It also discharges R4's unstated layout claims (`(380, 0)`, the wrap, the
  danger tier left of Cancel), none of which any test asserts.
- **Recorded in the commit body?** Yes: "Verified by eye as well as by exit code: the modal
  rendered headless and the PNG read -- the heading names the pass, the consequence line
  wraps inside 380px, Delete sits in the danger tier left of Cancel."
- **Stale spec sentence:** none. The spec's `## Verification` table has no eyeball row and
  does not need one; the PNG is evidence, not a gate.

---

## 4. Findings

### F1 — the spec's lib-picker `on_close` value is stale (DIVERGENT, correct as landed)

**Spec (R3):** "`lib_picker` -> `app.shader_lib_files.reset_inline_state` plus clearing
`picker_tag_input_focused` (today's pre-funnel cleanup at `lib_picker/__init__.py:39-44`,
moved into the registry's hook)".

**Code:** `shaderbox/popups/lib_picker/__init__.py:134`

```python
    on_close=lambda app: app.close_lib_picker(),
```

with `shaderbox/app.py:1363-1366`:

```python
    def close_lib_picker(self) -> None:
        self.modal = None
        self.shader_lib_files.reset_inline_state()
        self.shader_lib_files.picker_tag_input_focused = False
```

The spec asks for the two calls; the implementation introduces a new `App` method holding
them. **Effect is identical** (`close_modal` writes `app.modal = None` after `on_close`
regardless — R2 states that double write is idempotent), and the shape matches the other
five rows, every one of which delegates to an `App` method. `test_confirm.py:361` names
`close_lib_picker` in `_CLEANUP_METHOD` and spies it, so the cleanup is gated. Not a defect
— **the spec sentence is what needs updating**, since a reader following R3 literally would
look for a two-call lambda.

**Replacement text for R3:** "`lib_picker` -> `App.close_lib_picker` (a new method: resets
the inline state and clears `picker_tag_input_focused`, the pre-funnel cleanup at
`lib_picker/__init__.py:39-44` today), matching the other rows' shape".

### F2 — the spec's `switch_project` sentence names the wrong function (DIVERGENT, mechanism intact)

**Spec (R2):** "`switch_project`'s close is `self.modal = None` directly, never `close_modal`:
it runs in `_tick_frame_state`, outside the draw phase, where `imgui.close_current_popup()`
asserts (084 D5)."

**Code:** `switch_project` (`shaderbox/app.py:2432-2451`) writes no `modal` at all. The
direct write is in `_init`, which `switch_project` calls at `app.py:2450`:

```python
        # A pending confirm closes over the outgoing project's state.
        self.confirm = None
        if self.modal is ModalId.CONFIRM:
            self.modal = None
```

The *property* R2 cares about holds — the switch path clears the mutex by direct assignment,
never through `close_modal`, and `close_current_popup` is never reached outside the draw
phase. Two things differ from the sentence:

1. the write is in `_init`, one frame deeper than the sentence says. This is where the
   baseline had it too (`17d235c:app.py` clears state in `_init`, not `switch_project`), so
   the spec sentence was already describing the wrong function when it was drafted.
2. the write is **conditional on `CONFIRM`**, not unconditional. Every other modal's state is
   torn down by `_init` reloading the app state wholesale; the confirm is the one modal whose
   payload (`self.confirm`) closes over the *outgoing* project, which is why it gets an
   explicit null. The code's comment says exactly that.

No test pins this directly, and none needs to: the mutex walk
(`test_modal_chrome.py:249-265`) covers `popups/` and `widgets/` only, and `app.py` is
allowed to write its own field. Nothing regresses. **The spec sentence needs replacing.**

**Replacement text for R2:** "`_init` (which `switch_project` calls) clears a pending
confirm directly — `self.confirm = None` and `self.modal = None` when it is `CONFIRM` — never
through `close_modal`: it runs in `_tick_frame_state`, outside the draw phase, where
`imgui.close_current_popup()` asserts (084 D5). The confirm is the one modal needing an
explicit null, since its request closes over the outgoing project's state."

### F3 — `ui_primitives.modal_window`'s docstring still describes the deleted `is_X_open` shape

**Code:** `shaderbox/ui_primitives.py:325-341`

```python
    """Boilerplate-free modal-popup wrapper. Caller owns the `is_X_open` flag on `App`
    (allows per-modal cleanup on close); this owns the imgui dance: ...

        if not app.is_X_open:
            return
        with modal_window(LABEL, (W, H)) as visible:
            if not visible:
                return
            if not _draw_body(app):
                app.is_X_open = False
                imgui.close_current_popup()
    """
```

Nothing in the tree has an `is_X_open` flag — the shape it prescribes is exactly the
hand-rolled per-modal close R2 replaced, and the only caller is now
`registry.draw_modal` (`registry.py:66`). The imgui-ui skill's §7.2 was updated in this same
commit to say "**The open/closed state stays on `App`** (not on the wrapper): one field
holding which modal is open... each modal's per-close cleanup is a hook on its registry row
rather than a branch someone has to remember to write" — so the skill and the docstring now
contradict each other, and the docstring is the one a reader hits first when writing a modal.

This is the repo's own "docs are living" rule, and the only place in the wave where a stale
doc survived. It is prose, not behavior: `make gates` is green and no test reads the
docstring.

**Suggested replacement (the usage block):**

```python
    """Boilerplate-free modal-popup wrapper, called once from `popups/registry.py`: it owns
    the imgui dance (open by label, seed size, centre, enter the popup scope, yield
    visibility). Which modal is open lives on `App.modal`; a modal's per-close cleanup is its
    registry row's `on_close` hook, never a write at a call site. `flags` passes window flags
    through...; `fixed_size` forces `size` every frame...
    """
```

---

## 5. False trails — things that look like findings and are not

- **`close_modal` never calls `imgui.close_current_popup()` on the Esc path.** Looks like a
  desync: imgui's popup stack would keep the modal open while `app.modal` says closed.
  **Probed and false.** A frame-driven probe (Help opened, three frames, `close_modal`, two
  more frames) reads `imgui.is_popup_open("Help##help") == True` while open and `False`
  after, with `is_popup_open("", any_popup_id|any_popup_level) == False` too, and a
  subsequently opened Settings modal behaves normally. `draw_modal` returns before
  `modal_window` once `app.modal is None`, so nothing calls `begin_popup_modal` and imgui
  retires the popup itself — the same mechanism the baseline's early-returning `draw_X`
  relied on. R2's "runs only after a close that happened" is about the *declined* path
  inside the popup scope, which `registry.py:74`'s short-circuit handles.

- **`confirm.py` uses `wrapped_caption`, but R4 says `caption_text`.** Not a divergence in
  kind. The spec names the dim-caption tier; `wrapped_caption` is that tier's wrapping
  variant, and a 12-word line in a 380px fixed-width modal requires wrapping — a
  non-wrapping caption would clip the consequence text, which is the one string the modal
  exists to show. Verified visually by the implementer's headless PNG ("the consequence line
  wraps inside 380px"). The title uses the same helper with `COLOR.FG_TITLE`, as R4 asks.

- **`test_modal_chrome.py` excludes `registry.py` from the mutex walk
  (`:237`).** Looks like the checker narrowing its own domain. It is the opposite: the
  registry is the one module R2 *authorises* to write `app.modal` (`registry.py:55`), so
  including it would make the gate unsatisfiable. Every other file in both trees is walked,
  parametrized one case per file, and the exclusion is a single named filename rather than a
  pattern that could silently swallow a second module.

- **`_draw_document_reset`'s `end_disabled()` before `begin_disabled()`.** Flagged by the
  implementer as beyond-spec. Diffing the function against `17d235c` shows one changed line
  (the verb call). The inverted bracket is baseline, deliberate, and explained by the
  function's own comment; this wave neither introduced nor fixed it.

- **`delete_current_document_confirmed` is not in R5's table.** R5 names
  `delete_document_confirmed(document_id)` as "the bar's `DELETE_DOCUMENT` callback and the
  tile's item". The bar's callback takes no argument, so `app.py:1099-1100` adds a
  two-line adapter passing `self.current_document_id`. This is the spec's own requirement
  expressed in the only way the callback signature permits, and the confirm copy it produces
  is identical. Not a divergence worth a spec edit.

- **The Enter guard could fire on a modal *behind* the confirm.** No such case: `CONFIRM` is
  in the mutex (`app.modal` is one field), so no other modal is drawing when the confirm's
  body reads the key. `test_render_decoupling_loop.py:182-211` pins that the rest of the app
  is paused behind it.

- **`ConfirmRequest.on_confirm` captures `self` in a lambda, so a request could outlive its
  target.** Handled: `_init` nulls `self.confirm` on a project switch (`app.py:1618`) with a
  comment saying why. That is F2's second half, and it is the code being *more* careful than
  the spec, not less.

## 6. Tree state

- `git status`: **clean** (verified after the review's own mutation probes were restored; an
  untracked `tests/test_zz_probe.py` appeared mid-review from a concurrent reviewer session
  and was gone by the final check — not this commit's).
- `projects/dev/`: **unchanged** — `git diff HEAD -- projects/dev` is empty, matching the
  spec's "`projects/dev/` — nothing persisted changes."
- `make gates`: **exit 0**, captured to a file unpiped — `check passed`, `test passed`,
  `smoke passed`.
