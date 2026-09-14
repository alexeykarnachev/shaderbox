# Interactive-overlay inventory (093)

Every modal, popup, context menu, action-carrying combo, load-bearing tooltip, native file
dialog, palette/hotkey verb, and non-menu inline action control in `shaderbox/`, anchored to
`file:line`. Built for the follow-up review of "should this become a menu."

Popup mutex: `App.popup_state: PopupState` (`app.py:132-141`, CLOSED / EXAMPLES / HELP /
SETTINGS / PASS_SETTINGS / IMPORT_PASSES / EMOJI_PICKER / SHADER_LIB_PICKER / PROJECTS) — at
most one open. `App.any_popup_open()` (`app.py:1021-1027`) ORs in `copilot_revert_target is
not None` (the one modal outside the enum). Every opener funnels through `App._open_popup`
(`app.py:1029-1034`). The command palette (`app.is_palette_open`) is explicitly non-modal and
outside `PopupState`.

---

## 1. Modals (`PopupState` + the one outside it)

### Examples — `"Examples##popup"`
`shaderbox/popups/examples.py:44` `draw_examples`.
- **Opened by**: `App.open_examples` (`app.py:1042-1043`) ← File menu "Examples" bar item (`ui.py:748-749`), hotkey `CommandId.EXAMPLES` (`commands.py:229`, Alt+E), palette.
- **Items**: 3-col scrollable grid of example tiles (`_draw_grid`, `examples.py:92-111`, via `draw_document_preview_button`/`preview_cell`) → click selects; description slot (`_draw_description_slot`, 114-125); bottom row `primary_button("Open a copy")` (disabled until selected; Enter also accepts, 77-89) → `app.create_document_from_example`; `standard_button("Close")`.
- **State**: `app.app_state.selected_example_id`.
- **Close**: Open (commits), Close, imgui default click-away/Esc (plain `modal_window`, no custom Esc).
- **Gating**: Open button `begin_disabled(not is_selected)`.
- **Styling**: `modal_window` (fixed-size, `no_resize|no_scrollbar`).

### Help — `"Help##help"`
`shaderbox/popups/help.py:24` `draw_help`.
- **Opened by**: `App.open_help` (`app.py:1275-1277`) ← Bar item "Help" (`ui.py:750-751`), hotkey `CommandId.HELP` (F1), palette.
- **Items**: left section list (`selectable` per `HelpSection`, `help.py:50-53`); right pane: title, `markdown_text` body, optional GLSL snippet fenced as a code block (62-68); bottom row `primary_button("Insert at caret")` (only when `section.snippet and section.insertable` — a display-only snippet with no `insertable` flag draws no button at all, `help.py:76`; disabled + tooltip "Open a document's shader and click into the editor first..." when no valid target, 76-89) → `app.insert_text_at_caret`; `standard_button("Close")`. This is a distinct third "Insert at caret" surface from the Shader Library Picker's and the lib-tree function-leaf menu's (same label, different verb — see the cross-surface table).
- **State**: `app.help_section`; content sourced from `help_content.help_sections()` (data-only module, no imgui).
- **Close**: explicit `imgui.is_key_pressed(Key.escape)` check inside the body (71-72, redundant with the modal's own Esc — both close), Close button, Insert (only if it lands).
- **Gating**: Insert button disabled unless `editor_was_ever_focused and active tab.kind == "shader"`.
- **Styling**: `modal_window`.

### Settings — `"Settings##popup"`
`shaderbox/popups/settings.py:58` `draw_settings`.
- **Opened by**: `App.open_settings(focus="")` (`app.py:1045-1051`) ← Edit menu "Settings..." (`ui.py:731-737`), hotkey `CommandId.OPEN_SETTINGS` (Alt+S), palette, and programmatically with a `focus` field key from copilot's "Open Settings" unconnected-gate action (`copilot_chat.py:166`) and any exporter's config-not-connected gate.
- **Items** (single scrolling body, `_draw_body`, `settings.py:74-194`): General (Target FPS drag, Show cheatsheet checkbox, Throttle documents checkbox, GPU budget drag); Editor (whitespace/line-numbers/matching-brackets checkboxes, Keymap combo `settings.py:121` — grep hit, Font size / Tab size / Line spacing drags); **Integrations** (`settings.py:150-174`, one collapsible `imgui.tree_node(exporter.display_name)` per **available** exporter → `exporter.draw_config_ui(focus=focus)` (`:161`); an unavailable exporter draws a dim reason string instead of a node, `f"{exporter.display_name} — {exporter.unavailable_reason}"` (`:165-167`), no node at all; a pending `app.settings_focus` on that exporter's `config_field` force-opens its tree node via `imgui.set_next_item_open(True, Cond_.always)` (`:158-159`) before `focus_field` scrolls to the control; a Copilot tree node with the same force-open rule (`:169-173`) — OpenRouter key + Model text inputs, then 10 numeric "Agent limits" rows each with a `help_marker` tooltip, `_COPILOT_LIMITS`); **Telegram's `draw_config_ui`** (`exporters/telegram.py:345-388`): not-connected branch — 4-line `setup_steps` (`:352-361`); password-masked `labeled_text_input("Bot token", ..., password=True, focus=focus)` (`:367-369`); `primary_button("Connect")` → `self.begin_auth()` (`:370-371`); `danger_button("Clear token")` → `self.disconnect()`, shown only when a token is already typed (`:374-375`); always-drawn `connection_status(connected=..., is_error=..., message=..., who=..., on_disconnect=self.disconnect if connected else None)` (`:382-388`) whose Disconnect is a real `danger_button("Disconnect")` inside the shared primitive (`ui_primitives.py:768`). **YouTube's `draw_config_ui`** (`exporters/youtube.py:269-353`): not-connected branch — 7-entry `setup_steps` whose first four are `(text, url)` tuples rendering clickable `draw_link` rows to `console.cloud.google.com/...` (`:277-296`); `primary_button("Load client_secret.json...")` → `self._pick_client_secret()` (`:310-312`) with a focus-field wrapper (`:308-309`); `standard_button(paste_label)` toggling `"Paste instead"`/`"Hide"` → `self._render_state.show_paste` (`:315-320`); revealed `imgui.input_text_multiline("##yt_secret", ...)` paste box committing via `self._ingest_client_secret(pasted)` on non-empty change (`:322-327`); once a key is loaded but not connected — `primary_button("Connect")` wrapped in `begin_disabled(busy)` (`:334-339`) → `self.begin_auth()`, `danger_button("Clear credentials")` → `self.clear_credentials()` (`:342-343`); a `"Waiting for authorization..."` warning line while `busy` (`:345-349`); same shared `connection_status(...)` Disconnect as Telegram (`:351-357`). Library (`_draw_library_reset` — armed danger-button "Reset library..." → warning text → "Confirm reset"/"Cancel"); Keyboard (`_draw_keybindings` — one row per `CommandSpec` grouped by `CommandCategory`, each a `chord_row` + "Rebind" button that arms one-shot chord capture, `capture_chord()`); bottom `standard_button("Close")`.
- **State**: `app.app_state.*` (persisted global settings), `app.app_state.editor_settings.*`, `app.integrations_store.copilot.*` (persisted), `app.lib_reset_armed`, `app.settings_focus` / `app.settings_mark` (one-shot jump-to-field), `app.rebinding_command`, `app.effective_bindings` / `app.app_state.key_bindings` (persisted rebindings).
- **Close**: Close button → `app.apply_editor_settings()` then closes (editor settings apply at the one close funnel, not per-edit); Esc handled by `hotkeys._handle_escape` with the same apply-on-close carve-out (`was_settings_open` check, `hotkeys.py:366,398-399`) — but Esc is first intercepted by the chord-rebind capture (`hotkeys.py:357-358`, returns early when `app.rebinding_command is not None`) so the modal stays open while a chord is being captured. Local flag is named **`is_keep_opened`** (`settings.py:190`), the one modal in the codebase that inverts the `keep_open` naming rulebook §7.3 calls out by name — every other modal uses `keep_open`.
- **Gating**: each `SettingsField` jump (`settings.py:39-47`) force-opens + keyboard-focuses + outlines its section via `FieldFocus`/`focus_field`; rebind row `begin_disabled(not spec.rebindable)`.
- **Styling**: `modal_window`.

### Pass Settings — `"Pass settings##popup"`
`shaderbox/popups/pass_settings.py:41` `draw_pass_settings`. **Two modes, one modal**: edit (`_draw_body`) vs. create/draft (`_draw_draft`, branches on `app.pass_draft is not None`).
- **Opened by**: `App.open_pass_settings(name)` (`app.py:1053-1062`, edit mode) ← pass tile gear/context-menu "Settings" (`pass_list.py:70-71`, shared `pass_menu_items`), graph node context menu (`pass_graph.py:1490`, same shared items), hotkey `CommandId.OPEN_PASS_SETTINGS` (Alt+P, opens for the panel pass via `open_pass_settings_for_panel_pass`), palette. `App.open_add_pass()` (`app.py:1118-1126`, create/draft mode) ← Document tab "add pass" button (`document.py:466-...`), graph canvas context menu "Add pass" (`pass_graph.py:875-876`), hotkey `CommandId.ADD_PASS` (Alt+A).
- **Items — edit mode** (`_draw_body`, `pass_settings.py:108-137`): name input (rename-on-commit, `_draw_name`); group field + picker combo (`_draw_group`, combo at line 193 — grep hit — `##group_pick_{id}`, no-preview, lists existing groups + "none"); "Draws into" section (`_draw_target`): format combo (line 237 — grep hit — 8-bit/16-bit float/32-bit float, each with a `help_marker` explaining it), size slider (% of canvas, disabled when `is_output`), sampling/edges checkboxes; "Runs" section (`_draw_repeat`): iterations slider; bottom `standard_button("Close")`.
- **Items — draft/create mode** (`_draw_draft`, `pass_settings.py:67-105`): name input (auto-focused, Enter submits), group field+picker, target section, repeat section (same widgets as edit, operating on the in-memory `PassDraft`); bottom `primary_button("Create")` (or Enter) → `app.create_pass_from_draft()`, `standard_button("Cancel")`.
- **State**: `app.pass_settings_name` / `_name_buf` / `_group_buf` (edit mode), `app.pass_draft: PassDraft | None` (create mode — non-None routes the whole modal to draft mode), writes land via `app.session.set_pass_target/set_pass_iterations/rename_pass/set_pass_group` (persisted, `graph.json`).
- **Close**: `app.close_pass_settings()` (`app.py:1081-1110`) — the one funnel for both Close and Esc, commits a pending rename/group edit before clearing state; Cancel/Create in draft mode drop or commit the draft.
- **Gating**: size slider disabled when this pass is the document's output (`is_output=True` forces full-canvas scale).
- **Styling**: `modal_window` with `always_auto_resize` + width/height constraints (scrollbar left on).

### Import Passes — `"Import passes##popup"`
`shaderbox/popups/import_passes.py:44` `draw_import_passes`.
- **Opened by**: `App.open_import_passes()` (`app.py:1163-1169`) ← Document tab "import..." button, graph canvas context menu "Import..." (`pass_graph.py:877-878`), hotkey `CommandId.IMPORT_PASSES` (unbound default).
- **Items**: tab bar "This project" / "Examples" (`_draw_tabs`); 4×2 source-document grid (`_draw_grid`, `draw_document_preview_button`); description slot; group name input; per-entry-point row (`_draw_entry_points`): a combo (line 198 — grep hit — `##entry_{root}`, options "theirs" / each host pass "(mine)") plus, when fed by a host pass, a row of handover checkboxes (`_draw_handovers`); bottom `primary_button(f"Import {N} passes")` (disabled while `draft.rejection`) → `app.import_passes_from_draft()`, `standard_button("Cancel")`.
- **State**: `app.import_draft: ImportDraft | None`.
- **Close**: `app.close_import_passes()` (Cancel or successful Import); Esc via the generic `PopupState.IMPORT_PASSES` carve-out in `hotkeys._handle_escape` (`hotkeys.py:379-380`, routes to the same funnel).
- **Gating**: Import button disabled while the plan is rejected (name collision, cycle, etc. — `draft.rejection`).
- **Styling**: `modal_window`.

### Emoji Picker — `"Emoji##picker"`
`shaderbox/popups/emoji_picker.py:15` `draw_emoji_picker`.
- **Opened by**: `App.open_emoji_picker(target=callback)` (`app.py:1262-1266`) ← Telegram exporter's `EmojiControl` (built at `exporters/telegram.py:301-336`, the closure holding the `set_tooltip("Click to change emoji")` string at `:311`), from **two** real call sites, not one: the **pending-sticker preview overlay** (`telegram.py:545-547`, `if emoji.emoji_button(emoji.emoji, ROW_HEIGHT): emoji.open_emoji_picker(emoji.set_emoji)` — sets the pending sticker's emoji before it is added to a pack) and the **per-existing-sticker-cell overlay** `_draw_sticker_emoji` (`telegram.py:721-739`, passed as `preview_cell`'s `overlay=` param at `:703` — the only `preview_cell` call in the codebase that uses `overlay`; enqueues a `set_emoji` job against an already-added sticker via `file_id`). Two different targets, two different write paths (`set_emoji` direct vs. `_Job(kind="set_emoji", ...)` enqueued).
- **Items**: search input; scrollable grid grouped by Unicode emoji group (`load_emoji_groups`, cached from `emoji-test.txt`), each cell a button with hover tooltip = the entry's name (line 65 — grep hit); bottom `standard_button("Close")`.
- **State**: `app.emoji_picker_query`, `app.emoji_pick_target: Callable[[str], None] | None` (delivery callback, cleared on close).
- **Close**: pick (invokes target then closes), Close, imgui default.
- **Gating**: none.
- **Styling**: `modal_window`.

### Shader Library Picker — `"Shader Library##picker"`
`shaderbox/popups/lib_picker/__init__.py:45` `draw_lib_picker` (composes submodules `search.py`, `tree.py`, `preview.py`, `filtering.py`).
- **Opened by**: `App.open_shader_lib_picker()` (`app.py:1267-1274`) ← Library menu "Browse..." (`ui.py:739-745`), hotkey `CommandId.OPEN_LIB_PICKER` (Alt+L), palette.
- **Items**: top bar — search input + live-match count (`search.draw_search_row`), Favs/Reset pills (`draw_favs_and_reset_row`), tag pill bar (`draw_tag_bar`, ctrl-click isolates a tag); body — left tree column (dir/file/function, `tree.draw_tree`, right-click context menus, see §3) + right preview pane (`preview.draw_preview` — path click-to-copy with tooltip "Click to copy file path" via `draw_copyable_text`, signature, doc, tag editor with add/remove pills + autocomplete suggestions, body text); bottom `primary_button("Insert at caret")` (disabled + tooltip when no editor target, `lib_picker/__init__.py:137-142`) → `filtering.insert_name`, `standard_button("Close")`.
- **State**: `app.shader_lib_files: ShaderLibFileManager` (a sub-object holding `picker_query`, `picker_selected_function`, `picker_favs_only`, `picker_disabled_tags`, `picker_tag_input_focused`, `picker_new_tag_buf`, `picker_just_opened`, `file_rename`/`file_new`/`dir_new: InlineInput`, `file_delete_armed`/`dir_delete_armed`), `app.shader_lib_favorites`, `app.shader_lib_tags` — persisted (favorites/tags on disk; the lib itself is files on disk).
- **Close**: Insert (on success), Close, Esc (`lib_picker/__init__.py:151-156`) — **two different Esc-ownership predicates, deliberately not one**: (1) `inline_input_owns_esc(app)` (`__init__.py:32-42`), consulted by `hotkeys._handle_escape` (the FRAME-BEFORE gate, `hotkeys.py:386-389`) — true while a rename/new-file/new-dir inline input is armed OR `picker_tag_input_focused` is set; (2) the picker's own in-body Esc gate uses `input_owns_keys` (`__init__.py:77`, consumed at `:155`), which covers only rename/new-file/new-dir and **excludes** the tag-input-focused case — the comment at `:151-154` explains why: Esc on a focused tag input has nothing to commit, so it should close the picker, not be swallowed as a no-op defocus.
- **Gating**: Insert disabled without an editor target; arrow-nav and Enter suppressed while an inline input owns keys.
- **Styling**: `modal_window`.

### Projects — `"Projects##projects"`
`shaderbox/popups/projects.py:36` `draw_projects`.
- **Opened by**: `App.open_projects` (`app.py:2403-2410`, calls `reset_projects_state()` then `_open_popup(PopupState.PROJECTS)` at `:2410`) ← File menu "Projects..." (`ui.py:724-728`), hotkey `CommandId.OPEN_PROJECTS` (Alt+O), palette. `app.py:581` is unrelated: a **bare** `self.popup_state = PopupState.PROJECTS` on the dead-pointer-recovery path (`app.py:575-589`), set before `_init` so the first-run gallery stands down, immediately followed by a real `open_projects()` call at `:589` that fills the rows.
- **Items**: scrollable project row list (click selects via `_select_row`, disarming any pending delete, `projects.py:63-68`; double-click switches, gated on `not app.projects_input_focused`, `:78-83`; `_draw_row`); then one of three bottom states: verb row (`_draw_verb_row`, `projects.py:96-132` — Enter-to-switch when a row is selected and no name input is focused, `:101-107`, returns before drawing the row; `primary_button("Open")` disabled when nothing selected or the selection is already the open project, `:115-118`; `standard_button("New")` → `app.reset_projects_state()` + arms `projects_new_input`, `:120-122`; `standard_button("Open other...")` → `app.pick_project_dir()` which calls `pfd.select_folder` at `app.py:2399` (grep hit), `:123-125`; `danger_button("Delete")`, same disable rule as Open, → arms `projects_delete_armed`, `:126-129`; `standard_button("Close")` right-anchored at `imgui.get_content_region_avail().x - SIZE.BTN_SM_W`, `:130-132`); delete-confirm row (`_draw_delete_confirm` — "Delete to trash?" + Yes/No); new-name inline input row (`_draw_name_input` — text input + primary "New" + Cancel).
- **State**: `app.projects_rows`, `app.projects_selected`, `app.projects_delete_armed`, `app.projects_new_input: InlineInput`, `app.projects_error`, `app.projects_input_focused`.
- **Close**: Open (switch, via button, Enter, or double-click), Close, Esc — suppressed while the new-name input is open (`projects_input_owns_esc`, consulted by `hotkeys.py:381-385`) so its own inline Esc-cancel runs instead.
- **Gating**: Open/Delete disabled when the selected row is already the open project or nothing is selected.
- **Styling**: `modal_window`.

### Copilot revert confirm — `"Revert turn?"` (outside `PopupState`)
`shaderbox/widgets/copilot_chat.py:177` `_draw_revert_modal`. The only modal in the codebase that closes inline inside its own body rather than returning a `keep_open` bool the caller reads (`copilot_chat.py:194-200`).
- **Opened by**: `App.open_copilot_revert(msg)` (`app.py:1036-1040`) ← the per-user-bubble Revert glyph (`copilot_chat.py:544-545`, only shown when a live checkpoint exists for that turn).
- **Items**: excerpt of the reverted message + explanatory caption; `primary_button("Revert")` → `app.revert_turn(target)`; `standard_button("Cancel")`.
- **State**: `app.copilot_revert_target: Message | None` — the ONE modal state that lives outside the `PopupState` enum but still joins `any_popup_open()`'s mutex.
- **Close**: Revert (commits), Cancel, imgui default. `hotkeys._handle_escape` checks this first, before `any_popup_open()`'s other branches (`hotkeys.py:371-372`) — most-modal-first.
- **Gating**: none inside; the glyph that opens it is disabled while `app.copilot.state.in_flight`.
- **Styling**: `modal_window`.

---

## 2. Non-modal popups / inline prompts / pickers

| Name | `file:function` | Mechanism |
|---|---|---|
| Group-name prompt `##graph_group` | `pass_graph.py:1499` `_group_prompt` | Plain `imgui.begin_popup` (NOT `modal_window`, NOT `context_menu_style` — hand-rolled, despite behaving like a small modal). Opened by the node menu's "Group..." item. Text input + Create/Cancel. |
| Lib-picker inline rename/new-file/new-dir | `popups/lib_picker/tree.py:198` `_draw_inline_new_input`, `:290` `_draw_file_rename_input` | Not a popup window — an indented inline row inside the tree, armed via `ShaderLibFileManager.file_rename/file_new/dir_new: InlineInput`. Owns Esc while open (see Shader Library Picker above). |
| Projects new-name inline input | `popups/projects.py:157` `_draw_name_input` | Same shape — an inline row inside the modal body, not a nested popup. |
| Lib-picker tag editor add-tag row | `popups/lib_picker/preview.py:69-104` `_draw_function_tag_editor` | Inline text input + "+ Add" button + autocomplete suggestion pills; not a popup. |
| Telegram new-pack / delete-pack inline forms | `exporters/telegram.py:451-475`, `:483-500` | Armed inline reveal (new-pack: text input + Create; delete: hand-rolled red-tinted confirm child, NOT the shared `cell_delete_confirm`). |
| FPS details panel | `ui_primitives.py:1567` `fps_overlay`, called `ui.py:976-989` | Click-toggled inline child anchored under the FPS chip; not a `PopupState` member, not a popup window — a plain child with `auto_resize_y`. |
| `K`-lookup note | `tabs/code.py:695` `_draw_lookup_popup`, primitive `ui_primitives.py:361` `anchored_note` | A non-interactive, non-popup floating window (`no_inputs`) pinned near the caret; opened via `app.editor_lookup_requested` → `app.editor_lookup: LookupPopup`. Closes on any click or key. |
| Completion popup | `tabs/code.py:597-641` `_drive_completion` | Owned by the vendored editor library (`editor.complete_open()`), not an imgui popup; ShaderBox only drives the request/filter and overlays a companion doc note (`_draw_candidate_doc`, `code.py:713-741`, sourced from `app.editor_completion_offered`). |
| Telegram exporter unconnected gate | `exporters/telegram.py:405-413` `draw_target_panel` | Full-panel replacement when `not self._is_connected()`: `unconnected_gate("Not connected to Telegram.", "Connect a bot in Settings to share stickers.", "Set up token", extras.get(_OPEN_SETTINGS_KEY))` — the CTA button routes through `render_control.extras`' `_OPEN_SETTINGS_KEY` closure to `app.open_settings(focus=self.config_field)`. Replaces the whole sticker-grid target panel below (§8). |
| YouTube exporter unconnected gate | `exporters/youtube.py:387-395` `draw_target_panel` | Same shape: `unconnected_gate("Not connected to YouTube.", "Connect your channel in Settings to upload.", "Set up credentials", extras.get(_OPEN_SETTINGS_KEY))` — action label **differs** from Telegram's ("Set up credentials" vs "Set up token"), a genuine label drift on the same verb (open Settings, focused on this exporter's field). |
| Copilot chat unconnected gate | `widgets/copilot_chat.py:162-168` (called from `_draw_body`, above) | Full-transcript replacement when `not app.integrations_store.copilot.openrouter_key`: `unconnected_gate("Copilot is not set up.", "Add your OpenRouter API key in Settings", "Open Settings", on_action=lambda: app.open_settings(focus=SettingsField.COPILOT_KEY))`. Also drains `app.copilot_focus_pending` before drawing (`:163-166`) — the gate has no input box to focus, and a stale pending-focus latch would re-grab focus every frame, which a modal reads as a dismiss. |
| Copilot chat input + Send/Stop | `widgets/copilot_chat.py:299-328` (inside `_draw_transcript`'s input row) | `imgui.input_text_multiline("##copilot_input", app.copilot_input, ..., flags=enter_returns_true\|ctrl_enter_for_new_line\|word_wrap)` (`:300-307`), frozen via `imgui.begin_disabled(in_flight)` (`:299`) — one layout for both states, only the trailing slot changes. `in_flight`: `standard_button("Stop")` → `app.copilot.cancel_turn()` (`:322-323`). Idle: `primary_button("Send")` or Enter-submit, gated on non-empty stripped text (`:324-328`) → `app.copilot_send(...)`, clears the buffer. |
| Editor tab bar (incl. ▾ tab-list popup) | `tabs/code.py:86-135` `_draw_tab_row` | `imgui.begin_tab_bar("##editor_tabs", flags)` (`:104`) with `TabBarFlags_.reorderable\|fitting_policy_scroll\|tab_list_popup_button\|draw_selected_overline` (`:97-101`) — `tab_list_popup_button` renders an actual imgui popup listing every open tab (an overflow-scroll aid, distinct from the Document/Uniforms/Render/Share tab bar in §8). Per-tab: unsaved-dot flag (`item_flags \|= unsaved_document`, `:107-108`), error-tinted push/pop of `Col_.tab`/`tab_hovered`/`tab_selected` when the tab is a script tab with an active error (`:113-121,131`), drag-reorder read back via `_display_order`/`_apply_display_order` (`:120,134`, driven by imgui's own drag, applied post-loop), close-✕ via `begin_tab_item`'s `keep` out-param → `close_index` → `app.close_tab(close_index)` (`:118-119,132-133`). Programmatic tab switches (glyph-open / document-select / lib-jump / close) drive imgui via `item_flags \|= set_selected` from `app.tab_select_pending` (`:94-96,110-111`); a genuine user click is read back into `app.active_tab_index` only when no programmatic drive is in flight (`:122-125`). |
| Editor error strip | `tabs/code.py:329-374` `_draw_error_strip` | Clickable `imgui.selectable` rows, one per compile error, the caret's own error drawn selected (`:346-347,355`); click jumps the caret via `app.editor_jump_request = JumpRequest(...)`, opening the script file first when the error is cross-file (`:361-363`). Row count capped at `_MAX_ERROR_ROWS`; a `"+N more"` / `"show less"` `imgui.selectable` toggle (`:369-374`) flips `app.errors_expanded`, shown only when `n > _MAX_ERROR_ROWS`. |

---

## 3. Context menus (right-click)

### Pass-tile / graph-node shared menu — `pass_menu_items`
`shaderbox/widgets/pass_list.py:62` `pass_menu_items` (the shared item list — NOT itself a popup, the caller owns `begin_popup_context_item`).
- **Callers** (two, confirmed both read the identical function so the two surfaces cannot drift per the docstring at `pass_list.py:63-66`):
  1. `pass_list.py:83-87` `_draw_context_menu`, anchored `begin_popup_context_item(f"##pass_menu_{name}")` (explicit id — safe since each tile is its own child window). Right-click on a pass strip tile.
  2. `pass_graph.py:1478-1496` `_node_menu`, anchored `begin_popup_context_item(None)` (explicit `None` id — fires on right-click anywhere in the shared canvas child, per the module docstring at `pass_graph.py:23-24`). Right-click on a pass/ghost graph node.
- **Items** (`pass_menu_items`, draw order): "Open shader" → `app.ensure_shader_tab`; "Settings" → `app.open_pass_settings` (opens the Pass Settings modal above); "Delete" (`enabled=deletable`, `deletable = len(passes) > 1`, gated in Python too since `menu_item_simple` can register a click while `enabled=False` on this imgui-bundle build) → `_delete_pass`; "Leave group" (shown only if the pass has a group) → `app.leave_group`.
- **Graph node caller adds its own items** (`pass_graph.py:1489-1495`, `else` branch after calling `pass_menu_items`): "Group..." (only `node.kind == "pass"`) → seeds selection + arms `view.group_prompt` (opens `##graph_group` above). The graph node menu's `if node.kind == "box"` branch is entirely separate and does NOT call `pass_menu_items`: "Open" → drills into the group's tab scope; "Dissolve" → `app.dissolve_group`. **"Open" has a second surface**: double-clicking a `box` node (`pass_graph.py:1120-1122`, hit-tested on `is_mouse_double_clicked` → `_double_click`, `:1466-1475`) writes the identical `view.scope = node.group` / `view.fitted = False` / `view.selection = set(node.members)` — same verb, no menu involved.
- **Gating**: whole pass strip / graph canvas wrapped in `begin_disabled(app.copilot_turn_active)` — the menu is unreachable during a copilot turn.
- **Styling**: both callers wrap `context_menu_style()`.

### Graph canvas background menu — `##graph_canvas_menu`
`shaderbox/widgets/pass_graph.py:872` `_canvas_menu`.
- **Opened by**: right-click on empty canvas, hand hit-tested (`bg_hovered and node_hovered is None and mouse released right`, `pass_graph.py:1360-1365`) → `imgui.open_popup`.
- **Items**: "Add pass" → `app.open_add_pass()` (opens Pass Settings in draft mode); "Import..." → `app.open_import_passes()`; separator; "Fit" → `view.fitted = False`; "Arrange" → `app.arrange_graph`.
- **State**: `GraphViewState.fitted` (transient, per-document).
- **Gating**: canvas-wide `begin_disabled(copilot_turn_active)`.
- **Styling**: `context_menu_style()`.

### Shader-lib tree context menus
`shaderbox/popups/lib_picker/tree.py` — three independently-coded menus (no shared item list, unlike `pass_menu_items`):
1. **Directory header menu** — `_draw_dir_context_menu` (`tree.py:142-174`), `begin_popup_context_item(f"##dirctx_{dir}")`. Items: "New file here" → `begin_file_new_in`; "New subdirectory" → `begin_dir_new_in`; "Reveal in file manager" → `app.reveal_shader_lib_file_in_manager`; (non-root only) separator + armed "Delete directory (recursive)" / "Confirm delete (recursive)" (red text) → `arm_dir_delete` / `delete_dir`.
2. **File node menu** — `_draw_file_context_menu` (`tree.py:269-287`), `begin_popup_context_item(f"##filectx_{path}")`. Items: "Rename" → `begin_file_rename`; "Reveal in file manager"; separator + armed "Delete"/"Confirm delete" (red) → `arm_file_delete` / `delete_file`.
3. **Function-leaf menu** — `_draw_function_context_menu` (`tree.py:354-371`), `begin_popup_context_item(f"##fnctx_{fn.name}")`. Items: "Insert at caret" (`menu_item_simple("Insert at caret", enabled=has_editor)`, `tree.py:358-360` — **no Python-side `and has_editor` guard on the callback**, unlike the pass menu's "Delete"; on this imgui-bundle build `enabled=False` alone does not block a registered click, so this item can fire with no editor target) → `filtering.insert_name`; "Open file at declaration" → `filtering.open_at_decl`; "Copy name" → `filtering.copy_to_clipboard`; "Unfavorite"/"Favorite" (label flips on state) → `app.shader_lib_favorites.toggle`. **Sibling inline control, not part of this menu**: every function-leaf row also draws its own one-click favorite star directly on the row (`tree.py:324-328`, `imgui.small_button(("*" if is_fav else "o") + f"##fav_{fn.name}")` → `app.shader_lib_favorites.toggle(fn.name)`, colored `COLOR.FAVS`/`COLOR.FG_DIM`) — the same verb as this menu's Favorite/Unfavorite item, reachable without opening the menu.
- **Gating**: none of the three checks `copilot_turn_active` (the lib picker has no such freeze — it edits library files, not document state).
- **Styling**: all three wrap `context_menu_style()`.

---

## 4. Dropdowns / combos that carry actions or settings

(A plain value combo living inside a modal is listed under that modal in §1, not repeated here; this table is combos that live directly on a panel/tab/exporter row.)

| Combo | `file:line` | Options | Writes |
|---|---|---|---|
| Canvas-size presets | `tabs/document.py:43` `##canvas_presets` | square presets, named video shapes, any bound-texture size in the document | `_apply_canvas_size` → `document.resolution` — deliberately **a chip, not a combo** per its own comment (`document.py:38-39`); listed here for the same "picks a value" shape, but out of the button-tier count (rulebook §1). |
| Uniform sort-key | `tabs/uniforms.py:87` `##uniform_sort_key` | code / name / type | `document_ui_state.uniform_sort_key` |
| Sampler source | `widgets/uniform.py:312` `grouped_combo` `##source_{name}` | none, every pass name, "file..." | `app.session.set_sampler_source`, or opens the native media-file dialog (below) |
| YouTube resolution | `exporters/youtube.py:421` `labeled_combo("Resolution", ...)` | `MENU_SHAPES` (native/short/wide) | `self._render_state.shape` |
| YouTube category | `exporters/youtube.py:437` `labeled_combo("Category", ...)` | `CATEGORY_CHOICES` | `self._render_state.category_id` |
| Telegram pack | `exporters/telegram.py:437` `##pack` | one row per `PackEntry.title` | `self._select_pack(...)` |
| Details-panel media output type | `widgets/details.py:112` `##render_output_type` | video / image | `details.is_video` |
| Details-panel video quality | `widgets/details.py:121` `##video_quality` | low/medium-low/medium-high/high | `details.quality` |

Combos already covered inside their owning modal (not repeated): Pass Settings' group-picker (`pass_settings.py:193`) and format (`pass_settings.py:237`); Import Passes' entry-point substitution (`import_passes.py:198`, `begin_disabled(is_output)` at `:196`); Settings' keymap (`settings.py:121`).

---

## 5. Tooltips carrying more than a control's name

| Tooltip | `file:line` | Decorates |
|---|---|---|
| "Jump to declaration" | `ui_primitives.py:1735` (`clickable_label`) | `widgets/uniform.py:58` uniform-name label |
| "Delete" | `ui_primitives.py:1344` (inside `preview_cell`, guarded `if deletable:`) | every selected/unarmed tile's corner delete-✕ — shared by `widgets/uniform.py`, `widgets/document_grid.py`, `exporters/telegram.py` sticker grid. **Not** `widgets/pass_list.py`: the pass tile passes `deletable=False` (`pass_list.py:151`), so it never reaches this tooltip (see §8's pass-tile row). |
| Delete-armed cell value tooltip (ellipsized readouts) | `ui_primitives.py:580` (`clipped_caption`) | `widgets/uniform.py:244,270` truncated array/text value readouts |
| Settings help `(?)` | `ui_primitives.py:592-596` (`help_marker`) | every field in Pass Settings, Import Passes, Settings' Copilot limits |
| Play/stop reason | `ui_primitives.py:225-226` (`play_stop_toggle`) | Two call sites, five distinct strings between them. `tabs/document.py:446` whole-script toggle: "Stop the whole script" (playing) / "Resume the whole script" (stopped). `widgets/uniform.py:170-175` per-uniform toggle: "Whole script is stopped" (document-stopped override) / "Stop this uniform" (playing) / "Resume this uniform" (stopped) |
| Context-usage gauge | `ui_primitives.py:1057` (`gauge_bar`) | `widgets/copilot_chat.py:750` context gauge — shows the `context_gauge_readout` string |
| FPS detail row ms readout | `ui_primitives.py:1560` (`_profile_rows`) | throttled-document rows in the FPS panel |
| "Click to copy" / "Click to copy file path" | `ui_primitives.py:1644` (`draw_copyable_text`) | `popups/lib_picker/preview.py:26-31` file path, `widgets/details.py:43` export path |
| "Click to open + copy" | `ui_primitives.py:1672` (`draw_link`) | `exporters/youtube.py:503` "Open in YouTube Studio", `exporters/telegram.py:442-445` `t.me/addstickers/...` |
| Cursor-following uniform value | `tabs/code.py:1128` | live word-under-cursor in the code editor, when it names a uniform — also lights the panel row via `app.code_hovered_uniform` |
| "Channel view" | `ui.py:1001` | preview top-left channel-view chip |
| "Copy" | `copilot_chat.py:538` | per-bubble copy-icon button |
| "Revert this turn's changes" | `copilot_chat.py:548` | per-bubble revert-icon button (only when not `in_flight`) |
| "Recover from trash" | `copilot_chat.py:620` | pending-action-card recover-icon button |
| "Denies further changes this turn" | `copilot_chat.py:632` | source-lock gate's Deny button |
| "Layout: {value}" | `copilot_chat.py:714` | layout-cycle icon button |
| Source-lock mode explanation | `copilot_chat.py:736` | source-lock cycle chip (3 texts by mode) |
| "Click to change emoji" | `exporters/telegram.py:311` | sticker/new-sticker emoji glyph button |
| Per-step tool breakdown + token/cost stats | `copilot_chat.py:402` `_draw_snippet_tooltip` (`begin_tooltip` at 407), shown from `_draw_turn_snippet` at `copilot_chat.py:494-495` | a turn's square-bar snippet in the chat feed |
| Help-panel insert-disabled reason | `popups/help.py:84` | Insert-at-caret button when no editor target |
| Lib-picker insert-disabled reason | `popups/lib_picker/__init__.py:140` | Insert-at-caret button when no editor target |
| Emoji-entry name | `popups/emoji_picker.py:65` | each emoji grid cell |
| "Reset document" | `tabs/document.py:246` | Document tab's Reset button |
| Script/graph open reason | `tabs/document.py:439,459` | entry-point "open" buttons |
| "If checked, renders all documents, otherwise, renders only the selected one." | `widgets/document_grid.py:56-58` | Document-grid "Render all" checkbox |

---

## 6. Native file dialogs (`portable_file_dialogs`)

| Dialog | `file:line` | Trigger | Consumer |
|---|---|---|---|
| Media file open (blocking) | `widgets/uniform.py:211` (`_pick_media_file`, via `pfd_block`) | sampler-source combo's "file..." entry | `app.session` sampler binding |
| Export output path (blocking) | `widgets/details.py:31` (`pfd_block(pfd.save_file(...))`) | Details panel "Choose file..." button | `details.file_details.path` (Render/Share tabs) |
| YouTube client_secret.json (blocking) | `exporters/youtube.py:249` (`_pick_client_secret`) | "Load client_secret.json..." button | `self._yt.client_id/client_secret` |
| Open-project folder (blocking) | `app.py:2399` (`pick_project_dir`) | Projects modal "Open other..." button | `app.request_project_switch` |
| Copilot FILE-gate (async, main-thread poll) | `ui.py:115` (opened), `app.py:445-446` (state fields), polled every frame by `_pump_file_gate` (`ui.py:97-143`) | the copilot worker thread blocking on a `bind_media`/`import_document` tool call | `gate.answer_file(...)`, unblocking the worker |

The FILE-gate is the one async dialog: the native picker is opened once (`ui.py:114-117`) and polled non-blockingly across live frames (`dialog.ready()`) so the render loop never freezes, versus the four `pfd_block(...)`-wrapped calls above which block the calling frame until the OS dialog closes.

---

## 7. Command palette / hotkey-reachable verbs

`shaderbox/commands.py` is the source of truth: `CommandId` enum + `COMMAND_SPECS: list[CommandSpec]` (id, label, default_chord, category, scope, `in_palette`, `rebindable`). Consumed by three surfaces that must never drift apart because all three walk the same table:
- **Command palette** — `imcmd.command_palette_window("CommandPalette", ...)` (`ui.py:644-646`), non-modal, `app.is_palette_open`, opened by `CommandId.OPEN_PALETTE` (Ctrl+Shift+P) or the Tools-category default; registered via `App._register_palette_commands` (not itself an overlay surface — feeds imgui_command_palette's own UI).
- **Keyboard cheatsheet** — `widgets/cheatsheet.py:25` `draw`, a read-only HUD drawn on the foreground draw list (immune to window z-order), toggled by `app.app_state.show_cheatsheet` / `CommandId.TOGGLE_CHEATSHEET` (Alt+/). Lists every chord whose scope is currently active (`_is_active`, `cheatsheet.py:17-22`), grouped by `CommandCategory` in `CATEGORY_ORDER`. Zero interactive elements — pure `add_text`/`add_rect` on the draw list.
- **Settings rebinder** — `popups/settings.py:377` `_draw_keybindings`, inside the Settings modal (§1) — the one place chords are actually rebound.
- **Hotkey dispatch** — `hotkeys.py:_dispatch_registry` fires `app.command_callbacks[spec.id]()` per matched chord every frame; `CommandScope` (GLOBAL/EDITOR/COPILOT) plus `popup_suppresses` gate which chords fire while a modal is open (all scopes suppressed) or the palette is open.
- **Esc funnel** — `hotkeys.py:354-399` `_handle_escape`, closing exactly one thing per press, most-modal-first: **chord-rebind capture** (`app.rebinding_command is not None`, `:357-358`, returns early — Esc cancels the capture instead of closing Settings) → copilot revert target → `any_popup_open()` (with per-modal carve-outs for Pass Settings/Import Passes/Projects/Shader Lib Picker's own inline-input Esc ownership) → palette → copilot chat defocus.

### 7.1 `COMMAND_SPECS` — all 32 commands (`shaderbox/commands.py:108-238`)

(The table as it stood when inventoried. The command system was then redesigned from
scratch -- categories, order, labels, two commands added -- in `06_command_system.md`.)

One row per `CommandId`, in table-declaration order. "Other surfaces" lists every additional path to the same verb, so the cross-surface tables below (§ "Verbs reachable from more than one surface") can be regenerated from this table.

| `CommandId` | Label | Default chord | Category (cheatsheet) | Scope | `in_palette` | Other surfaces reaching the same verb |
|---|---|---|---|---|---|---|
| `OPEN_PROJECTS` | Projects | Alt+O | File | GLOBAL | yes | File menu "Projects..." (`ui.py:724-728`) |
| `SAVE` | Save | Ctrl+S | File | GLOBAL | yes | none found (no menu item, no button) |
| `QUIT` | Quit | Ctrl+Q | File | GLOBAL | yes | File menu "Quit" (`ui.py:729-730`) |
| `NEW_DOCUMENT` | New document | Ctrl+Shift+N | Document | GLOBAL | yes | File menu "New document" (`ui.py:720-723`); document-grid `standard_button("New document")` (`document_grid.py:45-46`) — 3-surface verb, see F3 fold-in below |
| `DELETE_DOCUMENT` | Delete document | Alt+D | Document | GLOBAL | yes | document-grid tile delete-✕/confirm (`document_grid.py:92-99,108`) |
| `TOGGLE_DOCUMENT_PLAY` | Play/stop document script | F5 | Document | GLOBAL | yes | Document tab's own play/stop toggle (`document.py:443-448`) — both call `self.toggle_current_document_play`/`set_document_all_stopped` on the same state; one verb, two entries, in the cross-surface table |
| `RESET_DOCUMENT` | Reset document | F6 | Document | GLOBAL | yes | Document tab `danger_button` "Reset document" (`tabs/document.py:243-246`) |
| `NEXT_PASS` | Next pass | Alt+Right | Document | GLOBAL | yes | reaches `app.choose_output` via `step_output_pass`(1) → `pick_pass` — a 4th surface for "Choose a pass as the document output", alongside pass-tile click, graph-node click, texture-preview click |
| `PREV_PASS` | Previous pass | Alt+Left | Document | GLOBAL | yes | reaches `app.choose_output` via `step_output_pass`(-1) → `pick_pass` — same 4th-surface verb as `NEXT_PASS` |
| `OPEN_SHADER` | Open shader | Alt+C | Editor | GLOBAL | yes | pass-tile / graph-node context menu "Open shader" (`pass_list.py:68-69`, shared `pass_menu_items`) |
| `OPEN_SCRIPT` | Open script | Alt+R | Editor | GLOBAL | yes | Document tab script entry-point "open" button (`document.py:437-438`) |
| `OPEN_GRAPH` | Open graph | Alt+G | Editor | GLOBAL | yes | Document tab graph entry-point "open" button (`document.py:457-458`) |
| `CYCLE_CODE_TAB` | Cycle code tab | Ctrl+Tab | Editor | GLOBAL (cheatsheet category EDITOR) | yes | none found (imgui's built-in cycle is off app-wide by design) |
| `CLOSE_CODE_TAB` | Close code tab | Ctrl+W | Editor | EDITOR | yes | editor tab bar's per-tab close-✕ (`tabs/code.py:118-119,132-133`) |
| `JUMP_NEXT_ERROR` | Jump to next error | F8 | Editor | GLOBAL | yes | error-strip row click (`tabs/code.py:355-363`, jumps to one error; F8 cycles to the "next") |
| `FORMAT_BUFFER` | Format | Ctrl+Shift+I | Editor | EDITOR | yes | vim-keymap leader `<leader>f` (`commands.py:255-258`, `LEADER_BINDINGS`) |
| `FOCUS_TAB_DOCUMENT` | Document tab | Ctrl+1 | View | GLOBAL | yes | Document/Uniforms/Render/Share tab bar click (`ui.py:1028-1067`) |
| `FOCUS_TAB_UNIFORMS` | Uniforms tab | Ctrl+2 | View | GLOBAL | yes | same tab bar |
| `FOCUS_TAB_RENDER` | Render tab | Ctrl+3 | View | GLOBAL | yes | same tab bar |
| `FOCUS_TAB_SHARE` | Share tab | Ctrl+4 | View | GLOBAL | yes | same tab bar |
| `CYCLE_CHANNEL_VIEW` | Cycle channel view | Alt+V | View | GLOBAL | yes | preview's channel-view chip (`ui.py:990-999`) |
| `TOGGLE_COPILOT` | Toggle copilot | Alt+J | View | GLOBAL | yes | copilot toggle chip (`ui.py:767-783`, `toggle_button("Copilot")`) — **note**: the hotkey calls `app.toggle_copilot` (`app.py:888`), the chip calls `app.toggle_copilot_open` (`app.py:900`) — two different `App` methods for the same visible effect |
| `CYCLE_COPILOT_LAYOUT` | Cycle copilot layout | Ctrl+H | View | COPILOT | yes | chat window's layout-cycle icon (`copilot_chat.py:711-714`) |
| `OPEN_LIB_PICKER` | Shader library | Alt+L | Tools | GLOBAL | yes | Library menu "Browse..." (`ui.py:739-745`) |
| `OPEN_PALETTE` | Command palette | Ctrl+Shift+P | Tools | GLOBAL | yes | — (this is the palette's own opener) |
| `OPEN_SETTINGS` | Settings | Alt+S | Tools | GLOBAL | yes | Edit menu "Settings..." (`ui.py:731-737`); programmatic `open_settings(focus=...)` from the copilot unconnected gate (`copilot_chat.py:162-168`) and both exporters' unconnected gates (`telegram.py:405-413`, `youtube.py:387-395`) — label drift "Settings..." (menu) vs. "Open Settings" (gate CTA) |
| `OPEN_PASS_SETTINGS` | Pass settings | Alt+P | Tools | GLOBAL | yes | pass-tile / graph-node context menu "Settings" (`pass_list.py:69-70`, shared `pass_menu_items`); hotkey opens for the panel pass via `App.open_pass_settings_for_panel_pass` (`app.py:1112-...`, callback wired at `app.py:681`) |
| `ADD_PASS` | Add pass | Alt+A | Tools | GLOBAL | yes | Document tab "add pass" button (`document.py:477`); graph canvas context menu "Add pass" (`pass_graph.py:875-876`) — label-case drift |
| `IMPORT_PASSES` | Import passes | unbound (chord `0`) | Tools | GLOBAL | yes | Document tab "import..." button (`document.py:480`); graph canvas context menu "Import..." (`pass_graph.py:877-878`) — label-case drift |
| `EXAMPLES` | Examples | Alt+E | Tools | GLOBAL | yes | menu-bar "Examples" bar item (`ui.py:748-749`) |
| `HELP` | Help | F1 | Tools | GLOBAL | yes | menu-bar "Help" bar item (`ui.py:750-751`) |
| `TOGGLE_CHEATSHEET` | Toggle keyboard cheatsheet | Alt+/ | Tools | GLOBAL | yes | Settings modal General section checkbox "Show keyboard cheatsheet" (`popups/settings.py:87-89`) |

`repeat` defaults to `False` (`commands.py:88`) and no spec overrides it, so `repeat` is `False` for all 32. `rebindable` defaults `True` and is never overridden either. None of the 32 sets `in_palette=False`, so all 32 are palette-reachable.

## 8. Inline action controls (not in a menu, act on a row/tile/card)

These are the "move into a menu?" candidates.

| Control | `file:line` | Acts on | Notes |
|---|---|---|---|
| Pass-tile corner delete-✕ | `ui_primitives.py:1195` `preview_cell`, called via `pass_list.py:135-158` | the output pass's tile | **No glyph drawn — not a control.** `_draw_pass_tile` passes `deletable=False` (`pass_list.py:151`) and `armed=False` (`:141`); `preview_cell` guards the ✕ on `if deletable:` (`ui_primitives.py:1338`) and the confirm wash on `if selected and armed:` (`:1324`), so neither is submitted. Fixed in commit `97841ec` ("093: the output tile drew a dead delete glyph after wave 3") — an earlier wave of this doc described a state the tree has since moved past; only `result.clicked` is read (`pass_list.py:154`), which is correct because there is nothing else to read. |
| Document-tile click | `widgets/document_grid.py:93-94` (`result.clicked`) | the document set | `app.select_document(id)` — the "choose a thing" family the cross-surface table already tracks for pass-tile/graph-node/texture-preview clicks (`choose_output`); this is the document-level sibling. |
| Document-tile delete-✕ + confirm wash | `widgets/document_grid.py:92-99` | one document tile | Fully wired (contrast with the pass tile above, which draws no delete glyph at all): armed → confirmed (deferred delete after the loop to avoid mid-iteration mutation) → cancelled. |
| Sticker-grid delete-✕ + confirm wash | `exporters/telegram.py:668-719` via `preview_cell` | one sticker cell | Fully wired; `selected` also gated on `not in_flight`. |
| Unwire "✕" badge | `pass_graph.py:620` `_draw_wire_x`, hit-tested `pass_graph.py:957-969` | the selected graph wire | Not an imgui item — hand hit-tested because a bezier curve has no rect. |
| Delete/Backspace on selected wire | `pass_graph.py:1347-1358` | the selected graph wire | Same `app.unwire` as the badge; gated by `delete_allowed`. |
| Node drag / port drag / output-dot wire-drag | `pass_graph.py:1138-1231` | graph nodes/ports | Persists via `app.commit_node_drag` / `app.drop_wire`. |
| Rubber-band multi-select | `pass_graph.py:1094-1105,1298-1325` | graph selection | View-only. |
| Graph wheel-zoom | `pass_graph.py:987-997` | graph canvas view | `if hovered and io.mouse_wheel != 0.0:` — zoom about the cursor position, clamped to `SIZE.GRAPH_ZOOM_MIN/MAX`; writes `view.zoom` + `view.pan` (per-document `GraphViewState`). |
| Graph pan | `pass_graph.py:1086-1093` | graph canvas view | Middle-drag or Alt+left-drag (`bg_active and (mouse_down(middle) or io.key_alt)`); writes `view.pan`. |
| Copy icon | `copilot_chat.py:534-538` `copy_icon_button` | any chat bubble | Clipboard copy. |
| Revert icon | `copilot_chat.py:544-548` `revert_icon_button` | a user bubble with a live checkpoint | Opens the Revert-turn modal (§1). |
| Recover-from-trash icon | `copilot_chat.py:616-620` | a pending-action card | `app.recover_deleted_document`. |
| Gate Yes/No, credential input, config panel, source-lock Allow/Deny | `copilot_chat.py:567-666` `_draw_pending_action` and helpers | the in-flight copilot gate | Decision UI embedded in the transcript, not a popup — exists because the turn is paused awaiting it. |
| Result-widget open buttons | `copilot_chat.py:552-564` | a tool-result message | `open_url_button` / `open_path_button`. |
| Layout-cycle icon | `copilot_chat.py:711-714` | the whole chat window | `app.cycle_copilot_layout`. |
| Source-lock cycle chip | `copilot_chat.py:724-730` | the whole chat window | Cycles `SourceLock`. |
| Clear / Close | `copilot_chat.py:755-760` | the whole chat window | `app.copilot_clear_chat` / `app.is_copilot_open = False`. |
| Feed/input splitter drag | `copilot_chat.py:218-246` | the chat window layout | `app.app_state.copilot_input_h` (persisted). |
| Play/stop toggle | `widgets/uniform.py:157-180` `_draw_play_stop` | one uniform row | Shown only when script-driven; disabled while the whole document is stopped or a copilot turn runs. |
| Input-type cycle chip | `widgets/uniform.py:131-139` `draw_input_type_selector` | one uniform row | Disabled when only one valid type exists. |
| "Randomize" | `widgets/uniform.py:249-251` | a buffer-type uniform row | Overwrites with random floats. |
| Texture-preview click | `widgets/uniform.py:183-204,336-341` `_draw_texture_preview` | a sampler row reading another pass | `app.choose_output` — same semantics as clicking that pass's strip tile. |
| Uniform-name clickable label | `widgets/uniform.py:47-79` `uniform_name_label` | one uniform row | Jump-to-declaration on click; hover lights the code editor's matching line. |
| Value editor | `widgets/uniform.py:239-390` `draw_uniform_control` | one uniform row's value | The control that actually changes a uniform — a seven-branch switch on `input_type`: `auto` read-only caption, `buffer` "Randomize" + byte count, `array` comma-separated `input_text`, `text` `input_text_multiline`, `texture` sampler combo + preview, `color` `color_edit{N}` (imgui's own picker popup), `drag` `drag_int`/`drag_float{,2,3,4}`. Grabbing the widget while a uniform is playing auto-stops it (`:385-390`, `is_item_activated()` → `app.set_uniform_stopped(..., True)`) — the reason the row's play/stop toggle can flip with no click on it. |
| FPS chip / details toggle | `ui_primitives.py:1567` `fps_overlay` | the document preview | Click-toggles `app.fps_details_open` inline (not a popup). |
| Channel-view chip | `ui.py:990-1001` | the document preview | `app.cycle_channel_view`. |
| Aspect preset chips + custom W/H | `tabs/document.py:100-155` | canvas aspect (Auto mode) | |
| Canvas W/H fields | `tabs/document.py:158-209` | canvas size (Fixed mode) | |
| Resolution-mode segmented control | `tabs/document.py:54-61` | canvas resolution mode | Auto/Fixed. |
| Passes strip "add pass" / "import..." | `tabs/document.py:471-481` (`standard_button("add pass")` `:477`, `standard_button("import...")` `:480`) | the document's pass list | Opens Pass Settings (draft) / Import Passes modal. |
| Details-panel "Choose file..." | `widgets/details.py:30-42` `draw_file_details` | Render/Share tabs' output path field | `if standard_button("Choose file..."):` → `pfd_block(pfd.save_file(...))` (native dialog, also in §6); rejects a path whose extension is outside `extensions` with a toast. |
| Details-panel resolution presets + W/H drags | `widgets/details.py:88-120` `draw_resolution_details` | Render/Share output size | Two one-click size verbs — `standard_button(f"{full_w}x{full_h}")` (`:94-95`, also fires when width/height is unset) and `standard_button(f"{half_w}x{half_h}")` (`:97-98`) — plus `imgui.drag_int("##width"/"##height")` (`:60-66`) with aspect-locked cross-derivation when an aspect is passed. |
| Document-grid "New document" | `widgets/document_grid.py:45-46` | the document set | `standard_button("New document")` → `app.create_document_from_example(STARTER_EXAMPLE_ID)`, wrapped in `begin_disabled(app.copilot_turn_active)` (`:44,47`). Third surface for this verb (see cross-surface table). |
| Document-grid "Render all" checkbox | `widgets/document_grid.py:51-58` | the document set's render gate | `imgui.checkbox("Render all", ...)` → `app.app_state.is_render_all_documents`; carries the §5 tooltip explaining the non-current-document render freeze. |
| Render-tab "Render" button | `tabs/render.py:40-70` `_draw_render_button` | the current document's render output | `primary_button("Render")`, `begin_disabled(not has_path)` (`:47-48`) → defers `target.document.render_media(pending)` one frame via `app.render_defer.submit` so a "Rendering..." cue paints first; failure pushes a toast, not just a log. |
| Editor chrome "Open dir" | `tabs/code.py:797-798` `draw_chrome` | the active tab's document dir | `standard_button("Open dir", width=BTN_SM_W)` → `app.open_current_document_dir()`. |
| Editor chrome copyable path | `tabs/code.py:790-792` `draw_chrome` | the active shader tab's file | Third `draw_copyable_text` call site (the other two are in §5) — `draw_copyable_text(str(local_file_path), copy_value=str(full_file_path))`, pushes a "Copied to clipboard!" toast on click. |
| Code-editor text surface | `tabs/code.py:1019` `##editor_surface` `invisible_button`, `_handle_mouse` (`:805-836`), `_handle_wheel` | the active shader/script tab's text | The app's largest interactive region. Press → `editor.set_cursor(...)` + `app.editor_drag_anchor`; double-click → word-select (`editor.feed("viw")` in vim NORMAL mode); left-drag → `editor.set_selection(anchor, head)`; wheel → scroll. `is_item_activated()` also latches `app.editor_was_ever_focused = True`, the gate the Help modal's and lib picker's Insert buttons both check. Listed in §9.5's `invisible_button` allowlist as a button-tier exception, which records only that the call exists. |
| Uniform sort-direction arrow | `tabs/uniforms.py:96-98` | uniform panel ordering | Companion to the sort-key combo. |
| "Apply" temporal-smoothing | `widgets/media_ops.py:37-54` | a bound video uniform | Writes a new file under `trash_dir`, swaps the bound `Video`. |
| Telegram carousel arrows | `exporters/telegram.py:658-666` | the sticker grid scroll | Disabled at scroll bounds. |
| Telegram "Render" | `exporters/telegram.py:600-601` | the pending sticker artifact | `standard_button("Render", width=render_w)` → `rc.render()`, sits beside "Add to pack" on the same row; a constant-height status slot below (`_draw_status_slot`, `:741-763`) holds four states including a progress bar, so the row never resizes mid-render. |
| Telegram Add-to-pack | `exporters/telegram.py:609-631` | the pending sticker | Disabled unless artifact present + pack active + not in-flight. |
| Telegram sticker-cell overlay | `exporters/telegram.py:691-719` `_draw_grid_cell` | one sticker cell | The only `preview_cell` call in the codebase passing `overlay=lambda side: self._draw_sticker_emoji(...)` (`:703`); `result.clicked` writes `self._render_state.selected_index = idx` (`:706-707`), separate from the delete-✕/confirm wired the same as §8's existing row. |
| YouTube "Render" | `exporters/youtube.py:456-458` | the current document's render output | `standard_button("Render")` → `rc.render()`, beside Upload. |
| YouTube "Upload" | `exporters/youtube.py:461-482` | the rendered artifact | `primary_button("Upload")`, `begin_disabled` behind a four-clause gate: `artifact is not None and rc.artifact_is_fresh and not rs.in_flight and size_ok` (`:463-467`) → enqueues an upload job with title/description/tags/category/is_short. |
| YouTube target-panel text fields | `exporters/youtube.py:427-431` | the pending upload's metadata | Title (`labeled_text_input`), Description (`labeled_multiline_input`), Tags comma-separated (`labeled_text_input`) — typed ahead of Upload, alongside the Resolution/Category combos already in §4. |
| Copilot toggle | `ui.py:768-784` | the whole app | `app.toggle_copilot_open`. |
| Editor/app-panel splitter | `ui.py:786-794` | panel layout | Drag-resize. |
| Document/Uniforms/Render/Share tab bar | `ui.py:1028-1067` | the settings panel | Native `imgui.begin_tab_bar`. |
| Uniforms-tab pass selector | `tabs/uniforms.py:37-54` via `text_tab_row` | which pass's uniforms show | `app.set_panel_pass`. |
| Graph scope tab row | `pass_graph.py:848-869` via `text_tab_row` | which group scope the canvas shows | |
| Share-tab outlet accordion headers | `tabs/share.py:85-102`, `collapsing_header` at `:91` | which exporter panel is open | Skipped entirely when only one exporter is available (`:70-81`, `len(available) == 1`). One-at-a-time is enforced by force-collapsing every NON-active header each frame — `imgui.set_next_item_open(False, Cond_.always)` (`:89-90`) — while the active header is left free to toggle (else it could never close). |

---

## Cross-surface facts

### Verbs reachable from more than one surface

Regenerated from §7.1's "Other surfaces" column plus the non-command verbs (context-menu shares, clicks, inline stars) that §7.1 doesn't cover.

| Verb | Surfaces | Label(s) |
|---|---|---|
| Open a pass's settings | pass-tile context menu, graph-node context menu (shared `pass_menu_items`), hotkey Alt+P (panel pass) | "Settings" (both menus, identical — no drift) |
| Add a new pass (draft mode) | Document tab button, graph canvas context menu, hotkey Alt+A | "add pass" (tab button, lowercase) vs. "Add pass" (canvas menu, capitalized) — **label-case drift** |
| Import passes | Document tab button, graph canvas context menu, hotkey (unbound) | "import..." (tab, lowercase+ellipsis) vs. "Import..." (canvas menu, capitalized) — **label-case drift** |
| Delete a pass | pass-tile context menu, graph-node context menu (both via shared `pass_menu_items` → `_delete_pass` → `app.session.delete_pass`, `pass_list.py:101`), hotkey Alt+D deletes a *document*, not a pass — separate verb, not folded in here | "Delete" (identical, same function) |
| Leave a pass's group | pass-tile context menu, graph-node context menu (shared `pass_menu_items`) | "Leave group" (identical) |
| Open a pass's shader | pass-tile / graph-node context menu (shared `pass_menu_items`), hotkey Alt+C (`OPEN_SHADER`) | "Open shader" (menus) — **a 3-surface verb**, the hotkey is a real third path the menu-only framing misses |
| Open a document's script | Document tab script entry-point "open" button, hotkey Alt+R (`OPEN_SCRIPT`) | "open" (button) — hotkey has no visible label |
| Open a document's graph | Document tab graph entry-point "open" button, hotkey Alt+G (`OPEN_GRAPH`) | "open" (button) |
| Reset document | Document tab `danger_button`, hotkey F6 (`RESET_DOCUMENT`) | "Reset document" (button) |
| Delete document | document-grid tile delete-✕/confirm, hotkey Alt+D (`DELETE_DOCUMENT`) | glyph + confirm wash vs. hotkey, no visible label collision |
| New document | File menu "New document", document-grid `standard_button("New document")`, hotkey Ctrl+Shift+N (`NEW_DOCUMENT`) — a 4th surface for the same underlying call, `app.create_document_from_example`, is the Examples modal's "Open a copy" (`popups/examples.py:84`, different example id, same method) | "New document" (menu + grid button, **identical** — the one multi-surface verb in this table with a label match, not a drift) |
| Close code tab | hotkey Ctrl+W (`CLOSE_CODE_TAB`), editor tab bar's per-tab close-✕ | no label (glyph) vs. hotkey |
| Toggle copilot | hotkey Alt+J (`TOGGLE_COPILOT` → `app.toggle_copilot`), copilot toggle chip (`ui.py:767-783`, `toggle_button("Copilot")` → `app.toggle_copilot_open`) | "Copilot" (chip) — **two different `App` methods for the same visible effect**, not a shared callback |
| Cycle copilot layout | hotkey Ctrl+H (`CYCLE_COPILOT_LAYOUT`), chat window's layout-cycle icon | "Layout: {value}" (icon tooltip) |
| Cycle channel view | hotkey Alt+V (`CYCLE_CHANNEL_VIEW`), preview's channel-view chip | "Channel view" (chip tooltip) |
| Focus a settings-panel tab | hotkeys Ctrl+1..4 (`FOCUS_TAB_DOCUMENT/UNIFORMS/RENDER/SHARE`), the Document/Uniforms/Render/Share tab bar click | tab bar labels vs. hotkey |
| Format the buffer | hotkey Ctrl+Shift+I (`FORMAT_BUFFER`), vim-keymap leader `<leader>f` (`commands.py:255-258`) | no visible label on either |
| Insert at caret | Shader Library Picker's bottom button (`lib_picker/__init__.py:135,145` → `filtering.insert_name`), lib-tree function-leaf context menu (`tree.py:359-360` → same `filtering.insert_name`) | "Insert at caret" (identical label, identical verb — two surfaces). Help modal's own `primary_button("Insert at caret")` (`popups/help.py:76-87` → `app.insert_text_at_caret`) is a **third, distinct surface with the same label but a different verb** (inserts a help snippet, not a lib function) — a label collision, not a verb collision. |
| Delete/confirm a row (tile, sticker) | `preview_cell`'s shared corner-✕ + wash (document grid, sticker grid) — **not** the pass tile, which passes `deletable=False` and draws no glyph at all (fixed in `97841ec`) | glyph only, tooltip "Delete" — identical mechanism across the two live surfaces |
| Open Settings | Edit menu, hotkey Alt+S, palette, programmatic jump from any unconnected-gate ("Open Settings" action in copilot chat, "Set up token"/"Set up credentials" in the exporter gates) | "Settings..." (menu) vs. "Open Settings" (copilot gate CTA) vs. "Set up token" / "Set up credentials" (exporter gate CTAs, label drift **between the two exporters too**) — all four route to the same modal |
| Reveal in file manager | Shader-lib directory context menu, file context menu — two separate `menu_item_simple` call sites, **both calling the one shared `App.reveal_shader_lib_file_in_manager`** (`app.py:2138`) | "Reveal in file manager" (identical text, single shared implementation — not duplicated) |
| Toggle a lib function's favorite | lib-tree function-leaf context menu's "Favorite"/"Unfavorite" item, the same row's inline star button (`tree.py:324-328`) | menu label flips on state; star glyph `*`/`o` flips color — both call `app.shader_lib_favorites.toggle` |
| Open a group's scope | graph box-node context menu's "Open" item (`pass_graph.py:1484-1486`), the same box node's double-click (`pass_graph.py:1120-1122` → `_double_click`, `:1466-1472`) | "Open" (menu) — no label on the double-click; both write `view.scope`/`view.fitted`/`view.selection` identically |
| Choose a pass as the document output | pass-tile click, graph-node click, uniform-row texture-preview click, hotkeys Alt+Right/Alt+Left (`NEXT_PASS`/`PREV_PASS`) | no label on the three clicks — all four reach `app.choose_output`, the hotkeys via `step_output_pass` (`app.py:2073-2087`) → `pick_pass` (`app.py:1948-1952`) → `choose_output`, the same hotkey-shadows-a-click shape as `OPEN_SHADER` |
| Open Projects | hotkey Alt+O (`OPEN_PROJECTS`), palette, File menu "Projects..." | "Projects..." (menu) |
| Quit | hotkey Ctrl+Q (`QUIT`), palette, File menu "Quit" | "Quit" (menu) — hotkey routes through `App.request_quit` (`app.py:869-870`), the menu item calls `glfw.set_window_should_close` directly: two paths, same effect, the same shape as `TOGGLE_COPILOT`'s two-method case |
| Open Examples | hotkey Alt+E (`EXAMPLES`), palette, menu-bar "Examples" bar item | "Examples" (bar item) |
| Open Help | hotkey F1 (`HELP`), palette, menu-bar "Help" bar item | "Help" (bar item) |
| Open Shader Library Picker | hotkey Alt+L (`OPEN_LIB_PICKER`), palette, Library menu "Browse..." | "Browse..." (menu) |
| Jump to next compile error | hotkey F8 (`JUMP_NEXT_ERROR`), error-strip row click | error-strip click jumps to the clicked error (`code.py:355,361-363`); F8 calls `App.jump_to_next_error` (`app.py:872`) to advance to the next one — same jump mechanism, different error selection |
| Play/stop | Document tab (whole-script), uniform row (per-uniform) | shared `play_stop_toggle` primitive, tooltip text differs by scope — document tab: "Stop the whole script"/"Resume the whole script" (`document.py:446`); uniform row: "Whole script is stopped"/"Stop this uniform"/"Resume this uniform" (`uniform.py:170-175`) |
| Play/stop the current document's whole script | hotkey F5 (`TOGGLE_DOCUMENT_PLAY`), Document tab's own play/stop toggle (`document.py:443-448`) | both call `self.toggle_current_document_play`/`set_document_all_stopped` on the same state (`app.py:2090`'s own comment: "The hotkey mirror of the document-tab play/stop toggle") — one verb, two entries, not a "none found" single-surface command |
| Toggle keyboard cheatsheet | hotkey Alt+/ (`TOGGLE_CHEATSHEET`), Settings modal General section checkbox | "Show keyboard cheatsheet" (checkbox, `popups/settings.py:87-89`) — both write `app.app_state.show_cheatsheet` (checkbox directly, hotkey via `App.toggle_cheatsheet`, `app.py:885-886`) |

### Verbs reachable from exactly one surface

Every item in §8 not listed above (document-tile delete-alone-without-hotkey-context, wire unwire, node drag/port drag/wire-drag, pan/zoom, rubber-band, copy/revert/recover chat icons, gate answers, source-lock cycle, chat Clear/Close, splitter drags, input-type cycle, randomize buffer, canvas presets chip, aspect chips, resolution-mode toggle, FPS chip, carousel arrows, add-to-pack, Telegram/YouTube Render and Upload buttons, details Choose-file/resolution-presets/W-H drags, document-grid Render-all checkbox, editor "Open dir"/copyable path, error-strip expand ("+N more"/"show less" — the jump-to-error click is now multi-surface, see above), tab bars, scope tabs, outlet accordion, copilot input box) plus every item in §1/§2/§3 whose "Opened by" lists a single entry path (the two Telegram emoji-picker call sites — two paths but one opener family, not a cross-surface verb, the `K`-lookup note, the completion popup, all three lib-tree context menus' New/Rename/Copy-name items, the group-name prompt, both Telegram inline pack forms) plus every `CommandId` in §7.1 whose "Other surfaces" column says "none found": `SAVE`, `CYCLE_CODE_TAB`, `OPEN_PALETTE`.

### `menu_item_simple` labels across the app (sorted)

| Label | Surface |
|---|---|
| Add pass | graph canvas menu |
| Arrange | graph canvas menu |
| Confirm delete | shader-lib file context menu (armed) |
| Confirm delete (recursive) | shader-lib directory context menu (armed) |
| Copy name | shader-lib function-leaf context menu |
| Delete | pass context menu (shared); shader-lib file context menu (unarmed) |
| Delete directory (recursive) | shader-lib directory context menu (unarmed) |
| Dissolve | graph box-node context menu |
| Favorite / Unfavorite | shader-lib function-leaf context menu (label flips on state) |
| Fit | graph canvas menu |
| Group... | graph pass-node context menu |
| Import... | graph canvas menu |
| Insert at caret | shader-lib function-leaf context menu |
| Leave group | pass context menu (shared) |
| New file here | shader-lib directory context menu |
| New subdirectory | shader-lib directory context menu |
| Open | graph box-node context menu |
| Open file at declaration | shader-lib function-leaf context menu |
| Open shader | pass context menu (shared) |
| Rename | shader-lib file context menu |
| Reveal in file manager | shader-lib directory context menu; shader-lib file context menu |
| Settings | pass context menu (shared) |

(Menu-bar `imgui.menu_item` labels — New document, Projects..., Quit, Settings..., Browse..., Examples, Help — are a distinct call (`imgui.menu_item`, carries a hint string) from `menu_item_simple` and are listed separately in §7/§1's openers.)

### `App` fields holding is-open / armed / in-flight state

| Field | `app.py:line` | Surface |
|---|---|---|
| `popup_state: PopupState` | 453 | the modal mutex (all of §1 except the revert modal) |
| `copilot_revert_target: Message \| None` | 434 | Revert-turn modal (outside the enum, still in the mutex) |
| `is_palette_open: bool` | 462 | Command palette (explicitly outside the mutex) |
| `rebinding_command: CommandId \| None` | 472 | Settings modal's chord-rebind capture |
| `lib_reset_armed: bool` | 474 | Settings modal's library factory-reset confirm |
| `settings_focus: str` / `settings_mark: tuple[str, float]` | 477, 479 | Settings modal's jump-to-field |
| `pass_draft: PassDraft \| None` | 412 | Pass Settings modal's create-mode branch |
| `pass_settings_name` / `_name_buf` / `_group_buf` | 413-417 | Pass Settings modal's edit-mode target + buffers |
| `import_draft: ImportDraft \| None` | 419 | Import Passes modal |
| `file_pick_dialog` / `file_pick_request` | 445-446 | copilot FILE-gate native picker poll |
| `emoji_picker_query: str` / `emoji_pick_target` | 484, 486 | Emoji Picker |
| `document_delete_armed: str` | 487 | Document grid tile delete-arm |
| `projects_selected` / `_delete_armed` / `_new_input` / `_error` / `_input_focused` | 538-544 | Projects modal |
| `shader_lib_files.picker_*`, `.file_rename/.file_new/.dir_new`, `.*_delete_armed` | (own object, `app.py:567` `ShaderLibFileManager`) | Shader Library Picker |
| `is_copilot_open` / `copilot_focus_pending` / `copilot_defocus_requested` / `copilot_turn_active` | 426-441 | Copilot chat window |
| `copilot_input: str` | 442 | copilot chat's `##copilot_input` text buffer (§8) |
| `copilot.state.in_flight` (not an `App` field — lives on the copilot engine state) | — | gates most copilot_chat.py inline controls. **Mirrored into, not disjoint from, `app.copilot_turn_active`**: `ui.py:461` assigns `app.copilot_turn_active = app.copilot.state.in_flight` every frame (with a falling-edge carve-out at `:457` that runs the turn-completion side-effects — `seal_checkpoint`, `save_conversation`, re-focus — before the copy lands). After reconciliation the two hold the same value; `copilot_turn_active` is a one-frame-lagged App-side copy used to gate pass/graph/document-grid, not an independent flag. A third, unrelated `in_flight` lives on the Telegram exporter's own render state (`exporters/telegram.py:166`) and is not comparable to either. |
| `fps_details_open: bool` | 518 | FPS chip's inline detail panel |
| `editor_lookup_requested: bool` / `editor_lookup: LookupPopup \| None` | 347, 352 | `K`-lookup note |
| `editor_completion_requested` / `_prefix` / `_seen` / `_auto` / `_offered` | 330-361 | code-editor completion popup (owned by the vendored editor library) |
| `errors_expanded: bool` | 560 | code tab's error strip "+N more" / "show less" toggle (§8) |
| `canvas_size_buf` / `canvas_w_editing` / `canvas_h_editing` | 401-403 | Canvas W/H fields (§8, `tabs/document.py:158-209`) — the per-half mirroring these implement is the field's whole mechanism |
| `aspect_buf` / `aspect_w_editing` / `aspect_h_editing` | 406-408 | Aspect preset chips + custom W/H (§8, `tabs/document.py:100-155`) |
| `active_document_tab` / `document_tab_select_pending` | 480, 483 | Document/Uniforms/Render/Share tab bar (§8, `ui.py:1028-1067`) — the one-shot that drives the tab bar's programmatic selection |
| `tab_select_pending: bool` | 557 | code-editor tab bar's own one-shot programmatic-select mirror (`tabs/code.py:94-96`) |
| `splitter_dragging` / `_splitter_press_on_splitter` | 528-529 | Editor/app-panel splitter (§8, `ui.py:786-794`) |
| `editor_focus_requested: bool` | 502 | lib picker's insert → editor re-focus one-shot (`app.py:1289`) |
| `GraphViewState.group_prompt` / `.group_name` (per-document, not on `App`) | `widgets/graph_state.py:85-86` | graph's `##graph_group` inline prompt |
| `GraphViewState.selection` / `.scope` / `.selected_wire` / `.node_drag` / `.wire_drag` / `.band_anchor` / `.fitted` / `.press_blocked` (per-document) | `widgets/graph_state.py` | graph canvas inline controls (§8): `band_anchor` is the rubber-band's press point, `fitted` backs the canvas menu's "Fit" item (§3), `press_blocked` is the copilot-turn gesture latch (§3's freeze note). `.scope` has two writers: the box-node menu's "Open" item and the box node's double-click (`pass_graph.py:1466-1472`) |

---

## 9. Facts for the design pass

One row per §1-§3 surface, carrying the rulebook facts (`/imgui-ui` skill §1/§2/§7/§8) the design
pass needs. `unknown` where the code does not settle the question rather than guessing.

### 9.1 Modals (§1) — action row, close labelling, `keep_open`, reset-on-open, inline-input commit

| Modal | Action row / spacer | Close vs Cancel | `keep_open` shape | Reset on open | Inline-input commit-on-deactivate |
|---|---|---|---|---|---|
| Examples | bottom (`examples.py:80-88`); **no** `dummy(SPACE.MD)` spacer above it (`:79` goes straight from the description slot) | "Close" (view-only) | `keep_open` | **No** — `app_state.selected_example_id` persists across opens (deliberate, `examples.py:69`) | n/a (no inline input) |
| Help | bottom (`help.py:76-91`); **no** spacer | "Close" | `keep_open` | reset (`app.py:1277`, section) | n/a |
| Settings | bottom (`settings.py:190-192`), Close alone; spacer present (`:188`) | "Close" | **`is_keep_opened`** — the one modal in the codebase that inverts the rulebook's `keep_open` naming (`settings.py:190`) | `lib_reset_armed = False` (`app.py:1048`); Integrations/Copilot tree nodes force-open only when a focus jump targets them | n/a at the modal's own bottom; the keybindings rebinder is a one-shot capture, not a committing text field |
| Pass Settings (edit) | bottom (`pass_settings.py:137`), Close alone; spacer (`:136`) | "Close" | `keep_open` | n/a (opens onto an existing pass, no reset) | name: Enter (`:163`) and **deactivate** (`:166`) both commit; group: Enter (`:187`) and **deactivate** (`:189`) both commit |
| Pass Settings (draft) | bottom (`pass_settings.py:102-104`); spacer (`:101`) | "Cancel" (form whose commit mutates) | `keep_open` | fresh `PassDraft` per open | same commit-on-deactivate shape as edit mode, operating on the draft |
| Import Passes | bottom (`import_passes.py:83-94`); spacer (`:81`) | "Cancel" — worth a design call: nothing is mutated until Import fires, so "Cancel" describes discarding a draft rather than undoing a mutation, unlike every other Cancel in the app | `keep_open` | fresh `ImportDraft` (`app.py:1168`) | n/a (no free-text inline input in the body; the group-name field has no distinct commit-on-deactivate note in the code) |
| Emoji Picker | bottom (`emoji_picker.py:75`), Close alone; spacer (`:74`) | "Close" | `keep_open` | query reset (`app.py:1265`) | n/a |
| Shader Lib Picker | bottom (`lib_picker/__init__.py:134-149`); **no** spacer (plain `imgui.spacing()` at `:123`) | "Close" | `keep_open` | `reset_inline_state()` + query + tag focus (`app.py:1270-1273`) | rename / new-file / new-dir inputs reserve an `x` cancel per rulebook §7.5 (`tree.py:292-293`); both **do** commit on deactivate — `wants_commit = changed or imgui.is_item_deactivated_after_edit()`, identically for new-file/new-dir (`tree.py:224`) and rename (`tree.py:304`) |
| Projects | bottom (`projects.py:113-132`); Close **right-anchored**, the only modal that does (`:130-132`); spacer (`:54`) | "Close" | `keep_open` | `reset_projects_state()` (`app.py:2405`, also called from `open_projects`) — the modal DOES survive the generic Esc bypass (contrast §9.2) | new-name input: Enter commits (`:168`), but **deactivate does NOT** — only Enter or the "New" button fires it (`:180`); clicking away silently discards a typed name. "Cancel" button + Esc both explicitly cancel (`:182`, `:172`). This is a genuine §7.5 divergence. |
| Copilot revert | bottom (`copilot_chat.py:195-200`); spacer `SPACE.SM` (`:193`) | "Cancel" | **No `keep_open` at all** — the only modal that closes inline inside its own body (`copilot_chat.py:194-200`) rather than returning a bool the caller reads | n/a (target set fresh by the opening click) | n/a |

Graph `##graph_group` prompt (§2, not a `PopupState` modal): Enter commits (`pass_graph.py:1509`), deactivate does **not**, "Cancel" button explicit (`:1536`) — same divergence shape as the Projects new-name input.

### 9.2 Close-funnel cleanup vs. the generic Esc bypass

`hotkeys._handle_escape`'s fallthrough branch sets `app.popup_state = PopupState.CLOSED` **directly** (`hotkeys.py:389`), skipping each modal's own wrapper-branch cleanup for any modal without an explicit carve-out in that function.

| Modal | Survives the generic Esc bypass? | Evidence |
|---|---|---|
| Pass Settings | yes — explicit carve-out calls `app.close_pass_settings()` (`hotkeys.py:375-377`) | commits a pending rename/group edit before clearing |
| Import Passes | yes — explicit carve-out calls `app.close_import_passes()` (`hotkeys.py:379-380`) | — |
| Projects | yes — `reset_projects_state()` also runs from `open_projects()`, so state is clean on next open regardless of how the previous close happened (`app.py:2405`) | no dangling state observed |
| Shader Lib Picker | yes for the inline-input case — `inline_input_owns_esc` suppresses the bypass entirely while an input is armed (`hotkeys.py:386-390`) | — |
| Emoji Picker | **no** — `emoji_pick_target` is nulled only at `popups/emoji_picker.py:23` (the picker's own close path) and at `app.py:1264` (`open_emoji_picker`'s set); a bypass-driven Esc-close leaves the callback dangling until the next open overwrites it | rulebook §7.6 names exactly this failure mode |
| Examples, Help, Settings | **yes, all three** — each modal's own `keep_open`-false branch does nothing beyond `popup_state = PopupState.CLOSED` (`examples.py:63-65`, `help.py:29-31`), so the bypass's direct set is equivalent: there is no cleanup for it to skip. Settings' one extra step, `apply_editor_settings()` (`settings.py:66-69`), is separately replicated by the bypass's own `was_settings_open` latch (`hotkeys.py:366,398-399`), so Settings survives too |

### 9.3 Context-menu discoverability hint (rulebook §7.4)

Exactly **one** hint exists anywhere in the app: `imgui.text_colored(COLOR.FG_DIM, "Right-click for actions")` above the shader-lib tree (`popups/lib_picker/__init__.py:116`). The pass strip (`pass_list.py`), the graph canvas background, and graph nodes — all three carrying right-click menus per §3 — have **no such hint**.

### 9.4 `menu_item_simple(enabled=...)` — no Python-side gate is needed on this build

A probe on the pinned imgui-bundle (1.92.801) drove a real click onto the same menu item with
`enabled` the only difference: `enabled=True` clicked, `enabled=False` did not; an item under
`begin_disabled(True)` did not either (`reviews/menus_design_feasibility.md` §0.1, reproduced
by the main session with a positive control in the same run). So:

| Site | `enabled=` condition | Python-side guard on the callback? |
|---|---|---|
| Pass menu "Delete" | `enabled=deletable` (`pass_list.py:72-73`) | yes — `... and deletable`, justified by a comment naming a footgun this build does not have |
| Lib-tree function-leaf "Insert at caret" | `enabled=has_editor` (`tree.py:358`) | no — and none is needed |

The comment at `pass_list.py:72-74` and `/imgui-ui` §7.4's bullet ("`menu_item_simple(label,
enabled=False)` can still register a click depending on the imgui-bundle version") are wrong
for the pinned build; §10 P9 corrects both.

### 9.5 Raw `imgui.button` / hand-rolled styling outside the tier system (rulebook §1)

Sanctioned exceptions are enumerated in `tests/test_button_tiers.py:25-54`'s `_NOT_A_VERB` allowlist (each with a written reason): `popups/emoji_picker.py` (one emoji cell in the glyph grid), `exporters/telegram.py` (the emoji glyph button + carousel arrows), `popups/lib_picker/tree.py` (the favorite star, §3/§8), plus four `invisible_button` hit-rects (`tabs/code.py`, `ui.py`, `widgets/copilot_chat.py`, `widgets/pass_graph.py`). `test_the_tier_set_stays_at_four` pins the tier count; `test_every_listed_exception_still_exists` fails if an allowlisted call site is removed without updating the list — both gates exist and run.

Hand-rolled `push_style_color` **not** covered by that gate: `popups/lib_picker/tree.py:165-167` and `:279-281` (red text on an armed delete item); `exporters/telegram.py:484-485` (the hand-rolled red delete-confirm child for pack deletion — already flagged in §2 as hand-rolled, additionally a tier/theming question here).

### 9.6 Word-budget gate (rulebook §2) — strings over budget

`tests/test_ui_prose_budget.py` measures every UI string against a per-widget-kind budget, with a checked-in over-budget allowlist (`_OVER_BUDGET`, `:288`; `_EXEMPT = set(_OVER_BUDGET)`, `:582`). Strings a design pass will want to see named:

- `popups/help.py:85` disabled-Insert tooltip, 15 words, against a 5-word tooltip budget.
- `popups/lib_picker/__init__.py:141` disabled-Insert tooltip, 11 words, same budget class.
- The ten `_COPILOT_LIMITS` help-marker hints (`popups/settings.py:225-320`), 20-35 words each with em-dash-joined second clauses, against an 8-word `help_marker` budget.
- `popups/settings.py:206-209` (library-reset warning) and `copilot_chat.py:189-192` (revert caption) — multi-sentence prose inside a modal body, not tooltips, so a different budget class applies.
- By contrast, Pass Settings' help markers (`pass_settings.py:88,202,243,263,292`) are all inside budget.

---

## 10. Design pass — gaps, defects, and the proposed shape

Revision 2, after `reviews/menus_design_brief.md` (12 findings) and
`reviews/menus_design_feasibility.md` (four library facts settled by probe; three proposals not
as written). What changed and why is in 10.5.

The brief (finding 17, verbatim in `00_findings.md`): every menu, popup, context menu and
settings modal reviewed together; good coverage, a convenient UX, no overblown state, "the
exact balance"; the code side included — one context menu for the node and the tile, the
refactors and unifications that needs. Findings 14 and 15 (the shared pass menu gained
`Open shader`; the tile lost its gear and ✕) are already landed and are the pattern this pass
generalizes: **a verb lives on its object's context menu; a tile or card carries no button.**

### 10.1 What the inventory shows, read as a whole

1. **Coverage is uneven by object.** A pass has a full menu (open shader, settings, delete,
   group, leave group). A group box has two items. The canvas has four. A DOCUMENT has no menu
   at all: its verbs are an armed corner ✕ on the tile (the shape finding 15 just removed from
   the pass tile) and a `New document` button above the grid. A uniform row has none (its one
   verb, jump to declaration, is the label's click). The lib picker's tree has three menus with
   the richest verb set in the app.
2. **Every verb has two or three homes and the labels drift between them.** `add pass` /
   `Add pass`, `import...` / `Import...`, `Settings...` / `Open Settings` / `Set up token` /
   `Set up credentials`, `Projects...` vs the palette's `Projects`. The palette and the
   cheatsheet read one table (`COMMAND_SPECS`) and cannot drift; the menu bar and every
   button and menu item are hand-written and do. Eight commands (Save, Next/Previous pass,
   Cycle code tab, the four Focus-tab chords) have no mouse-reachable home at all.
3. **The menu bar is five hand-written entries** (File: New document / Projects... / Quit;
   Edit: Settings...; Library: Browse...; Examples; Help) beside a 32-row command table that
   already carries a category per command. Add pass, Import passes, Open graph, Open script,
   Reset document, Save, Toggle copilot, the cheatsheet — none is in the bar.
4. **Four codings of "confirm a destructive verb", and the recoverability they guard does not
   match.** The corner ✕ + in-cell wash (`preview_cell`: document grid, sticker grid); a menu
   label that flips to `Confirm delete` on a second open (lib tree, twice, with a hand-rolled
   red text push); a `danger_button` that arms a confirm row (Projects delete, Settings
   library reset); a hand-rolled red child (Telegram pack delete). Against the code: a lib
   file or directory delete MOVES to `.trash/` and already toasts "recoverable in .trash/"
   (`shader_lib/file_ops.py:223-250`) — and confirms twice; a document delete moves to the
   project trash but its only Recover affordance is the copilot's card, built for a copilot
   delete alone (`app.py:810-825`, `copilot/backend.py:1177-1188`) — a grid delete has no
   undo, and confirms once; a pass delete drops the entry, its wiring, its position and
   every downstream sampler's source (`project_session.py:980-999`, the file stays) — and
   the shared menu's `Delete` confirms nothing since W3-2 removed the tile's arm.
5. **Two "ask for a name" rows commit only on Enter.** The group prompt (a hand-rolled
   `begin_popup`) and the Projects new-name row (already an `InlineInput`) both discard a
   click-away; the lib tree's rename / new-file rows commit on deactivate with an `x` cancel
   (`tree.py:224`, `:304`), which is the rulebook's shape (§7.5).
6. **Modal chrome drifts in the small.** Settings' body returns `is_keep_opened` (the one
   inverted name); the revert confirm closes itself inside its body instead of returning
   `keep_open`; three modals have no `SPACE.MD` spacer above the action row (two spellings of
   the spacer exist); Projects right-anchors its Close while every other modal left-packs the row.
7. **The generic Esc branch bypasses the close funnels.** `hotkeys._handle_escape` sets
   `popup_state = CLOSED` directly for any modal without a carve-out; four carve-outs exist
   (two of them "leave it open, an inline input owns Esc"), the emoji picker has none and
   leaves `emoji_pick_target` dangling (§9.2). Each new modal has to remember its own — the
   same forgettable step the `PopupState` decision already warns about for the draw block.
8. **Two `App` methods for one visible toggle** (`toggle_copilot`, `toggle_copilot_open`).
   Three branches against two; the difference is deliberate and the comment at `app.py:901`
   says why (a click has already moved focus). Stays (10.3).
9. **The `enabled=` "footgun" is not on this build.** The skill's §7.4 and
   `pass_list.py:72-74` say a disabled `menu_item_simple` can still register a click, and
   the pass menu double-gates its Delete for it. A probe on the pinned bundle (a real click
   on the same item, `enabled` the only difference) reads `True` / `False`
   (`reviews/menus_design_feasibility.md` §0.1, reproduced by the main session). §9.4's
   table sorts two sites by a guard that guards nothing.
10. **Three tooltips and ten help markers are over the budget the repo enforces** (§9.6),
    all inside the modals this pass owns (Help, the lib picker, Settings' copilot limits).
11. **Discoverability.** One "Right-click for actions" hint exists (the lib tree). The strip,
    the canvas and the nodes have none, and the maintainer found the node menu himself; the
    documents grid, which gains a menu here, is the one surface a first-time user meets first.

### 10.2 The shape proposed

**P1. One verb registry drives every menu.** `COMMAND_SPECS` already carries label, chord,
category and scope for 32 verbs, and the palette and cheatsheet already render from it. The
menu bar becomes a RENDER of that table: one top-level menu per `CommandCategory` in
`CATEGORY_ORDER` (File, Document, Editor, View, Tools), one item per spec in table order, the
label from the spec, the chord hint from `effective_bindings`. Every hand-written
`imgui.menu_item` in `ui.py` goes; the right-aligned project name stays.
- `CommandSpec` gains `in_menu: bool = True` and `separator_before: bool = False`. `in_menu`
  is `False` for the four Focus-tab chords and Cycle code tab: view-focus verbs whose menu
  item would duplicate the tab bar under it (§7.4's two-affordances rule); Save, Next /
  Previous pass and every other command get their first mouse home.
- `command_menu_item(app, command_id)` and the bar live in a new `shaderbox/menus.py`
  (imports `App`; `ui_primitives.py` is `App`-free by the three-layer rule and cannot host it).
  `command_label(command_id)` is a pure function in `commands.py` (a leaf), the one spelling a
  button uses when it opens a command's surface (`add pass` -> `Add pass`, `import...` ->
  `Import passes`, the three `Open Settings` / `Set up ...` gate buttons -> `Settings`).
- A per-item enabled test, `menu_enabled(app, spec)`: EDITOR scope -> `app.active_tab is
  not None`; COPILOT scope -> `app.is_copilot_open`; GLOBAL -> always. Per item, never on
  the category (`begin_menu` under `begin_disabled` does not open at all — probe §0.2). Not
  the cheatsheet's `_is_active` (it reads `editor_focused`, which the menu click has just
  cleared) and not `spec_eligible` (it rejects chord `0`, and Import passes is unbound).
- The chord hint: `imgui.menu_item` takes `shortcut` as a required positional and returns a
  tuple (§0.4); a spec with chord `0` passes `""`.
- Labels: the spec's, verbatim, no trailing `...` anywhere (the palette and the cheatsheet
  already show them so). `tests/test_command_registry_coverage.py` pins spec labels to the
  help snippet, so the help text follows in the same commit.
- Gate: `tests/test_ui_prose_budget.py` gains a row scoring `CommandSpec.label` from
  `commands.py`, since a primitive taking a `command_id` is invisible to its AST walk — without
  the row P1 removes every menu label from the budget's domain. Break: a 5-word label in
  `COMMAND_SPECS` must fail.

**P2. Object menus, one item set per object kind, drawn by every surface that shows the
object.** Finding 14/15's `pass_menu_items` is the template: a free function in the widget
module that owns the object, each caller owning only the popup.

| Object | Items (draw order) | Surfaces that draw it |
|---|---|---|
| pass | Open shader · Settings · ─ · Leave group (when grouped) · ─ · Delete ▸ | strip tile, graph node (both today) |
| pass, node only | + Group (after Settings, before Leave group) | graph node (today; 092 D14 — it seeds `view.selection`, which the strip does not have and could not show) |
| group box | Open · Dissolve | graph box (today; plus its double-click) |
| document | Open · Open folder · ─ · Delete ▸ | documents grid tile (new) |

`Delete ▸` is the confirm submenu of P4. `Open folder` on a document generalizes
`App.open_current_document_dir` to `open_document_dir(document_id)` (three lines; the
current-document verb calls it). The canvas background menu (Add pass, Import passes, ─, Fit,
Arrange) stays canvas-only; its first two render through `command_menu_item`. No menu for: the
uniform row (its one verb is the label's click), the strip's empty background (the buttons
under it are the entry; finding 4 redesigns that row), the editor tab (its one verb, Close,
already has the tab's ✕ and `Ctrl+W`; a `Close others` would be a new verb and a right-click
would also select the tab — probe §0.3).

**P3. A tile carries no button; the document grid follows the pass strip.** The grid tile
passes `deletable=False` as the pass tile does; `App.document_delete_armed`,
`set_document_delete_armed` and the handler's cleanup at `app.py:791-792` go; the grid's
three result branches go. `preview_cell` keeps `armed` / `deletable` / the wash for its one
remaining arming caller, the Telegram sticker grid: a sticker delete is a server-side verb
inside an exporter panel this pass does not own (10.3), and a wash on the cell is the right
confirm for an irreversible verb with no menu. The `overlay` slot stays (the sticker's emoji
glyph is a per-cell control, not a verb).

**P4. Every destructive verb on a menu confirms through a submenu; a modal's own destructive
button keeps its armed row.** One primitive, `confirm_menu_item(label, confirm_label) -> bool`
in `ui_primitives.py`: `begin_menu(label)` with the one item `confirm_label` drawn in
`STATE_ERROR` text, so the second click is a hover-and-click inside the same open menu — no
armed state, no reopen, no modal, no `PopupState` change. Used by: the pass `Delete`
(`Delete pass blur` — the entry, its wiring, its position and its readers' sources are lost
and there is no undo; the file stays and the toast says so), the document `Delete` (`Move
to trash` — the directory moves to the project trash; no undo affordance exists outside the
copilot), the lib file and directory deletes (whose armed label flip, `file_delete_armed` /
`dir_delete_armed` on `ShaderLibFileManager` and the two hand-rolled red pushes at
`tree.py:165-167` / `:279-281` all go; the trash move and its toast stay). Projects' delete
and Settings' library reset keep their armed `danger_button` rows: inside a modal an inline
confirm row is the rulebook's own shape. The revert confirm stays a modal (it is the copilot's,
and it shows the message being reverted). Reverses 092 D16's "the strip's two-click arm"
(already gone with W3-2) and the lib tree's second-open confirm; records the rule in
`conventions.md`.

**P5. One name-input shape.** `InlineInput` is promoted from `editor_types.py` to
`ui_primitives.py` (the conventions trigger: a second multi-inline-input surface; this is the
third — and `file_ops.py` already imports imgui through `theme`, so the move costs no new
import). A `name_input_row(id, input) -> InputRowResult(committed, cancelled)` primitive draws
the field, commits on Enter OR `is_item_deactivated_after_edit`, and reserves the `x` cancel;
the lib tree's two rows, the Projects new-name row and the group prompt draw through it. The
group prompt STAYS a popup (093 S5's Delete-key gate relies on it being one: an inline row in
the canvas child would be hovered, and only `is_any_item_active` would stand between a Delete
typed into the field and an unwire); its state moves from `group_prompt: bool` + `group_name:
str` to one `InlineInput` on `GraphViewState`. Only the commit rule changes: a click away
commits a non-blank name (a blank still refuses, as today).

**P6. Modal chrome to one shape, mechanically.** `settings.py`'s `is_keep_opened` renamed;
the revert confirm returns `keep_open` through `modal_window` like the other eight (its
caller nulls `copilot_revert_target`); every action row gets its `imgui.dummy((0, SPACE.MD))`
spacer, one spelling; Projects' Close joins the left-packed row. Gate:
`tests/test_modal_chrome.py` enumerates its domain from `PopupState` plus the revert modal
(never a `popups/*.py` glob — the lib picker is a package), resolves each member to its draw
function, and asserts the body binds a local `keep_open` that it returns, ends in a
`standard_button("Close")` / `("Cancel")` row, and has the spacer call before it (normalized
across `imgui.dummy((0, SPACE.MD))` and `ImVec2` spellings). Breaks to try, one per clause:
rename `keep_open` back in one modal; delete one spacer; add an enum member whose body returns
`ok` with no Close row (tried on the lib picker, whose layout a glob misses).

**P7. Esc closes through the modal's own funnel.** `App.close_popup() -> bool` dispatches on
`popup_state`: the per-modal close verb where one exists (`close_pass_settings`,
`close_import_passes`, a new `close_emoji_picker` that nulls the target), `apply_editor_settings`
+ `CLOSED` for Settings (the `was_settings_open` latch at `hotkeys.py:366` / `:398-399` goes,
or the apply runs twice), plain `CLOSED` for the rest; it returns `False` without closing when
an inline input owns Esc (`projects_input_owns_esc`, `inline_input_owns_esc` move inside it).
`hotkeys._handle_escape` calls it and loses its four carve-outs; the `rebinding_command` early
return stays where it is. The modal draw functions' own close branches call the same verb.
Gates: `tests/test_import_dialog.py:63-67` (a source-substring test on `hotkeys.py`) is
repointed at `close_popup` and made structural; `test_every_popup_state_has_a_draw_call` gains
a clause that parses `close_popup`'s body for a branch per member. Break: add a member, wire
its draw, omit its close branch — the new clause goes red while the old count stays green.

**P8. The two copilot toggles stay.** (10.1/8.)

**P9. The disabled-item double gate goes, and the two docs that prescribe it are corrected.**
`pass_list.py`'s `and deletable` and its comment go; §9.4 is rewritten as "no Python-side
guard is needed on this build"; `.claude/skills/imgui-ui/SKILL.md` §7.4's bullet is
rewritten to state the probe's result and the build it was measured on. A menu-item wrapper
primitive is NOT added: 21 of the 23 `menu_item_simple` sites pass no `enabled=` and gain
nothing from one, and the red-text pushes P9 would have absorbed go with P4's primitive.

**P10. Prose inside budget.** The two disabled-Insert tooltips shorten to one clause under the
5-word tooltip budget ("needs a shader caret"); the ten copilot-limit help markers cut to one
clause each and their long form moves to the Help panel's copilot section, which the prose
gate exempts as documentation; the over-budget allowlist rows for all twelve are deleted so
the gate holds them.

**P11. One hint where it earns it.** The documents grid gets the dim "Right-click for
actions" caption beside `New document`; the strip, the canvas and the nodes get none. The
rule is filed as a `conventions.md` design decision ("a hint on a modal's or panel's list; a
canvas or a strip with a visible primary click gets none; revisit if a walk finds a menu
undiscovered") and mirrored in the repo's own `.claude/skills/imgui-ui/SKILL.md` §7.4.

### 10.3 What stays as it is, and why

- The nine modals and their `PopupState` mutex: no modal is added or merged (P4's confirm is
  a submenu, not a modal). The pass settings modal's two modes stay one modal.
- The Uniforms tab's pass selector, the graph's scope tabs, the Document / Uniforms / Render /
  Share tab bar, every chip and every combo in §4: they name a STATE and are not verbs.
- The editor tab bar: its one verb has the tab's ✕ and `Ctrl+W` (P2).
- The lib tree's inline favorite star beside the menu's Favorite / Unfavorite: the rulebook's
  own carve-out (§7.4, "toggling a favorite — the inline star is fine"); P3's "no button on a
  tile" is about verbs on a tile, and the star is a one-click state toggle on a row.
- The editor error strip (a row whose primary click IS the action), the `K`-lookup note
  (non-interactive) and the completion popup (the editor library's).
- The Examples modal's selection persisting across opens: a browser's selection, not a query
  (§7.6 is about transient search state); deliberate at `examples.py:69`.
- `toggle_copilot` / `toggle_copilot_open` (P8).
- The exporters' config panels, their `unconnected_gate`s and the Telegram pack forms —
  including the hand-rolled red child at `telegram.py:484-485` §9.5 flags: the panels' bodies
  are the exporters' own. Their gate buttons' labels do change under P1 (`command_label`).
- The copilot chat's inline controls (copy, revert, recover, the gate answers): the
  transcript is a conversation, its per-message icons are the convention for chats, and the
  gate answers exist because a turn is paused on them.

### 10.4 The one call that is the maintainer's

**A pass `Delete` with a confirm submenu (P4).** Today the shared menu's Delete fires at once
(the tile's two-click arm went with W3-2). P4 puts the submenu on it because the loss (the
wiring, the position, every reader's source) has no undo and a wired pass in a six-node graph
is minutes of work; the cost is one hover inside the menu. If he prefers the immediate
delete, the pass row uses a plain `menu_item` and the toast names what was lost.

### 10.5 Revision 2 — what the reviews changed

- P4 rewritten from the code: the lib deletes were trash moves with a toast (not "a real
  filesystem delete"), the document delete has no user-facing undo (the copilot's Recover card
  is the copilot's), so the modal-vs-toast split inverted on three of four cases; replaced by
  one submenu primitive that adds no state and no modal. 10.3's "no modal is added" and P4 now
  agree. 092 D16 is named.
- P2: `Group...` stays node-only (092 D14, silently reversed before); `Open folder` named as
  the generalization it is; the editor-tab menu dropped (two new verbs it did not name, and a
  right-click that selects).
- P1: the primitive's home (`menus.py`, not `ui_primitives.py`), the enabled test, the
  positional `shortcut`, the prose-budget row, the help-snippet coupling, `in_menu` decided.
- P3: `preview_cell` keeps its machinery for the sticker grid (option A decided); the fifth
  caller (`uniform.py:194`) noted as unaffected.
- P5: the group prompt stays a popup (093 S5); the Projects row already IS an `InlineInput`,
  so P5 is the commit rule, not an adoption.
- P6, P7: gates made breakable as the feasibility review specified; P7 preserves the two
  "leave it open" carve-outs and the Settings apply.
- P8 dropped (a deliberate, documented difference). P9 replaced by the probe's result.
- P11: the rule's home is `conventions.md`; the skill file is the repo's own
  (`.claude/skills/imgui-ui/`), so it is amended too — F8's "outside the repo" was wrong.
- 10.3 now rules on the favorite star, the error strip, the lookup note, the completion
  popup, the Examples selection, the Telegram red child. 10.4 keeps the one genuine fork.

---

## Coverage

Reviewed against `reviews/menus_inventory_surfaces.md`, `reviews/menus_inventory_verbs.md`, and
`reviews/menus_inventory_accuracy.md` — round 1 of every missing surface, wrong claim, and anchor
error the three reports named has been folded into the sections above (§1's Settings/Help/
Emoji-Picker/Projects/Copilot-revert entries, §2's three new unconnected-gate rows plus copilot
input/editor tab bar/error strip, §3's lib-tree favorite star, §4's chip/disabled notes, §7.1's
full `COMMAND_SPECS` table, §8's new rows for exporter Render/Upload/text-fields/Choose-file/
resolution-presets/document-grid chrome/editor chrome/graph pan-zoom, the regenerated
cross-surface tables, the corrected `App`-fields table, and the new §9 design-pass facts).

**Round 2 of all three reports is also folded in**: the accuracy pass's anchor corrections
(revert-modal def `copilot_chat.py:177`, Import Passes Esc `hotkeys.py:379-380`, §9.6 word
counts, `repeat`'s `False` default, the five-string play/stop tooltip breakdown, §9.1's lib-tree
commit-on-deactivate cell, §9.2's resolved Examples/Help/Settings row); the verbs pass's six
newly-credited multi-surface commands (`OPEN_PROJECTS`, `QUIT`, `EXAMPLES`, `HELP`,
`OPEN_LIB_PICKER`, `JUMP_NEXT_ERROR`), `TOGGLE_CHEATSHEET`'s Settings-checkbox surface,
`TOGGLE_DOCUMENT_PLAY`'s de-contradicted cell, and `NEXT_PASS`/`PREV_PASS` as a fourth surface
for "choose a pass as the output"; and the surfaces pass's three fresh surfaces (the uniform-row
value editor, the code-editor text surface, the graph box-node double-click) in §8, §3, and the
cross-surface table.

Grep commands run (both over `shaderbox/`):

```
grep -rn "begin_popup_modal\|modal_window(\|begin_popup(\|begin_popup_context_item\|begin_popup_context_window\|open_popup(\|begin_menu(\|begin_main_menu_bar\|begin_menu_bar\|menu_item\|begin_combo(\|combo(\|pfd\.\|begin_tooltip\|set_tooltip\|command_palette\|CommandPalette" shaderbox/
```
→ **133 hits** (saved and counted; every line individually verified against a direct read of its file).

Supplementary verification greps (each returned zero or the stated hits, confirming no gap beyond the 133):
```
grep -rln "begin_popup|open_popup|begin_menu|menu_item|begin_combo|combo(|pfd\.|set_tooltip|begin_tooltip" shaderbox/copilot/     → no hits (package imports no imgui at all)
grep -n "pfd\." shaderbox/exporters/*.py                                                                                          → only youtube.py:249 (already in the 133)
grep -n "begin_popup|open_popup|begin_menu|menu_item|begin_combo|combo(|pfd\.|set_tooltip|begin_tooltip" shaderbox/tabs/render.py shaderbox/tabs/share.py shaderbox/tabs/share_state.py shaderbox/tabs/__init__.py   → no hits
```

Every one of the 133 hits appears above, either as its own surface/row or folded into the surface it belongs to (e.g. every `context_menu_style()`/`modal_window()` call site is named under the surface that wraps it, every generic-primitive tooltip line in `ui_primitives.py` is named under §5 with its call site(s)). No hit required a "not a surface" table — every grep hit was either a real interactive surface, a shared-primitive definition whose call sites are surfaces, or infrastructure (imports, the popup-dispatch block, docstring mentions of `begin_popup_context_item` explaining the anchoring pattern) folded into its surface's narrative.

Surfaces found beyond the 133 literal grep lines (extra `pfd.`/tooltip/combo call sites, plus
plain-button and `imgui.checkbox`/`imgui.drag_int`/`imgui.selectable` controls the grep pattern
does not match at all — these came from the three reports' full-file reads, not from the grep,
and are folded into their surfaces above rather than re-grepped): `app.py:2399` (Projects "Open
other..." `pfd.select_folder`), `document_grid.py:56-58` "Render all" tooltip, the non-functional
pass-tile delete-✕ that draws no glyph at all (`pass_list.py:135-158`, `deletable=False` — a
consequence of `preview_cell`'s shared logic rather than a new grep line, and corrected from an
earlier "dead control" description that predated commit `97841ec`), the fully-wired document-grid/
sticker-grid delete affordances (consequences of the same `preview_cell`/`close_cross_button`/
`cell_delete_confirm` primitives already in the 133 at `ui_primitives.py:1344`), both exporters'
`draw_config_ui` bodies and `draw_target_panel` unconnected gates, the copilot chat input/Send-
Stop row, the editor tab bar's `tab_list_popup_button` popup and error strip, the Render/Details/
document-grid chrome buttons, and the lib-tree per-row favorite star.
