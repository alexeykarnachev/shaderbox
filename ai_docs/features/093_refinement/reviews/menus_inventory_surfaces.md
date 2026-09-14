# Review — `04_menus_inventory.md` against the code

Anchor: the code, enumerated independently (grep over the eleven constructs, then an end-to-end
read of `shaderbox/ui.py`, `ui_primitives.py`, `tabs/*`, `widgets/*`, `popups/**`,
`exporters/telegram.py`, `exporters/youtube.py`, plus `app.py`, `commands.py`, `hotkeys.py`).

## Verdict

**FAIL.** A whole surface kind is absent — every exporter's **Settings config panel**
(`draw_config_ui`: token/credential inputs, Connect, Clear, Disconnect, setup-step link lists,
paste box) appears nowhere, and both exporters' **target-panel unconnected gates** are named only
as an "Opened by" clause for Settings, never as surfaces. Beyond that, 14 surfaces are wholly
missing and one §8 row makes a claim the code contradicts.

Counts: **14 missing**, **9 partial**, **1 wrong**, plus 8 line-citation drifts under ten lines
(listed at the end, not counted as findings).

---

## 1. Missing surfaces

| Surface | `file:line` | In inventory? | What is wrong or absent |
|---|---|---|---|
| **Exporter Settings config panel — Telegram** | `exporters/telegram.py:341-388` | **missing** | The whole `draw_config_ui` body. Contains: a 4-item `setup_steps` list (`:352-361`); a password-masked Bot-token input `labeled_text_input("Bot token", …, password=True, focus=focus)` (`:367-369`); `if primary_button("Connect"): self.begin_auth()` (`:370-371`); `if danger_button("Clear token"): self.disconnect()` (`:374-375`); and `connection_status(…, on_disconnect=self.disconnect if connected else None)` (`:382-388`) whose `Disconnect` is a real button (`ui_primitives.py:768` `if danger_button("Disconnect")`). The inventory reduces all of it to five words in the Settings entry: "one collapsible `tree_node` per exporter — `draw_config_ui`". |
| **Exporter Settings config panel — YouTube** | `exporters/youtube.py:272-357` | **missing** | Same class, more controls: a 7-entry `setup_steps` whose first four items are `(text, url)` tuples rendering **clickable console.cloud.google.com links** via `draw_link` (`ui_primitives.py:765` `draw_link(url)`); `primary_button("Load client_secret.json...")` (`:310`); a `standard_button(paste_label)` toggling between `"Paste instead"` and `"Hide"` (`:319-320`); a revealed `imgui.input_text_multiline("##yt_secret", …)` paste box (`:324-326`); `primary_button("Connect")` (`:337`) wrapped in `begin_disabled` while `busy`; `danger_button("Clear credentials")` (`:342`); and the same `connection_status` Disconnect (`:351-357`). None of it appears. |
| **Telegram target-panel unconnected gate** | `exporters/telegram.py:405-413` | **missing** | `unconnected_gate("Not connected to Telegram.", "Connect a bot in Settings to share stickers.", "Set up token", extras.get(_OPEN_SETTINGS_KEY))` — a full-panel replacement with its own CTA button. The inventory mentions "any exporter's config-not-connected gate" only as an *opener of Settings* (line 38) and in the cross-surface table (line 281); it is never a row of its own, and its button label ("Set up token") is never recorded. |
| **YouTube target-panel unconnected gate** | `exporters/youtube.py:387-395` | **missing** | `unconnected_gate(…, "Set up credentials", …)`. Same omission; the two gates carry *different* action labels ("Set up token" vs "Set up credentials") — exactly the label-drift the inventory's §"Verbs reachable from more than one surface" exists to catch. |
| **Copilot-chat unconnected gate** | `widgets/copilot_chat.py:162-168` | **missing** | `unconnected_gate("Copilot is not set up.", "Add your OpenRouter API key in Settings", "Open Settings", on_action=lambda: app.open_settings(focus=SettingsField.COPILOT_KEY))`. Cited at line 38 only as a *source of* the Settings focus key; it replaces the entire transcript when no key is set, so it is a surface. |
| **Copilot chat input + Send/Stop button** | `widgets/copilot_chat.py:299-328` | **missing** | The primary way a user acts through the copilot. `imgui.input_text_multiline("##copilot_input", …, flags=enter_returns_true \| ctrl_enter_for_new_line \| word_wrap)` (`:300-307`), frozen by `imgui.begin_disabled(in_flight)` (`:299`); then the state-swapping trailing slot — `if standard_button("Stop", width=btn_w): app.copilot.cancel_turn()` (`:322-323`) when in flight, else `primary_button("Send", …) or submitted` (`:325`). §8 lists Clear/Close, the layout icon, the source-lock chip and the splitter, but not the input or Send/Stop. |
| **Editor tab bar (incl. its ▾ tab-list popup)** | `tabs/code.py:86-135` | **missing** | `imgui.begin_tab_bar("##editor_tabs", flags)` (`:104`) with `flags` including `imgui.TabBarFlags_.tab_list_popup_button.value` (`:99`) — that flag renders an actual **imgui popup** listing every open tab, which the brief's grep cannot match and which no other surface duplicates. The row also carries per-tab close ✕ (`opened, keep = imgui.begin_tab_item(…, True, item_flags)` at `:116`, `close_index = i` at `:120`) and drag-reorder (`reorderable`, applied via `_apply_display_order` at `:134`). §8's "Document/Uniforms/Render/Share tab bar" row covers a *different* tab bar (`ui.py:1028-1067`); this one is absent. |
| **Editor error strip** | `tabs/code.py:332-374` | **missing** | Clickable rows: `clicked = imgui.selectable(f"{label}##err{i}", err is at_caret)[0]` (`:355`) jumps the caret to the error, opening the script file first when the error is cross-file (`:361-363`); plus a `+N more` / `show less` expander `toggled = imgui.selectable(f"{more}##errmore", False)[0]` (`:369`) writing `app.errors_expanded`. Neither the strip nor `app.errors_expanded` appears anywhere, including the state-field table. |
| **Editor chrome — "Open dir" button + copyable path** | `tabs/code.py:763-804` | **missing** | `if standard_button("Open dir", width=float(SIZE.BTN_SM_W)): app.open_current_document_dir()` (`:797-798`), and `if draw_copyable_text(str(local_file_path), copy_value=str(full_file_path))` (`:790`) — a *third* `draw_copyable_text` call site. §5's "Click to copy" row names only `lib_picker/preview.py:26-31` and `widgets/details.py:43`. |
| **Render-tab "Render" button** | `tabs/render.py:40-70` | **missing** | `if primary_button("Render"): … app.render_defer.submit(_run_render)` (`:48`), gated by `imgui.begin_disabled(not has_path)` (`:47`). The Render tab appears in §8 only as a name inside the tab-bar row; its one verb is unlisted. |
| **Details-panel "Choose file..." button** | `widgets/details.py:30` | **partial→missing as a control** | §6 records the *native dialog* it opens, but the button itself (`if standard_button("Choose file..."):`) is not in §8's inline-action table, unlike every other button-that-does-a-thing. |
| **Details-panel resolution presets + W/H drags** | `widgets/details.py:88-120` | **missing** | `if standard_button(f"{full_w}x{full_h}") or not details.width or not details.height:` (`:94`) and `if standard_button(f"{half_w}x{half_h}"):` (`:97`) — two one-click size verbs; plus `imgui.drag_int("##width", …)` / `"##height"` (`:89-95`). None listed. |
| **Document-grid "New document" button + "Render all" checkbox** | `widgets/document_grid.py:45-59` | **missing** | `if standard_button("New document"): app.create_document_from_example(STARTER_EXAMPLE_ID)` (`:45-46`) inside `begin_disabled(app.copilot_turn_active)`, and `app.app_state.is_render_all_documents = imgui.checkbox("Render all", …)[1]` (`:51-53`) with an information-carrying `begin_tooltip` at `:56` ("If checked, renders all documents, otherwise, renders only the selected one."). The grid's *tiles* are in §8 but its two chrome controls are not; the `:56` tooltip is claimed "already counted" in the Coverage section (line 364) yet never appears in §5. |
| **Lib-tree per-leaf favorite star** | `popups/lib_picker/tree.py:324-328` | **missing** | `star_label = ("*" if is_fav else "o") + f"##fav_{fn.name}"` … `if imgui.small_button(star_label): app.shader_lib_favorites.toggle(fn.name)` — a one-click toggle on every function row, separate from the context menu's "Favorite"/"Unfavorite" item that §3 does record. |

---

## 2. Wrong claim

| Surface | `file:line` | In inventory? | What the code says |
|---|---|---|---|
| **Pass-tile corner delete-✕** | `widgets/pass_list.py:135-152` | **wrong** | §8's first row calls it a "**Dead control**: `preview_cell` draws the delete-✕ … whenever `selected=True`, and `_draw_pass_tile` passes `selected=is_output`, but only reads `result.clicked`". The code passes `deletable=False`: `pass_list.py:151` reads `        deletable=False,`. In `preview_cell` the glyph is inside `if deletable:` (`ui_primitives.py:1338`), so **nothing is drawn** — there is no dead glyph to click. The doc describes a state the tree has moved past: commit `97841ec` is titled "093: the output tile drew a dead delete glyph after wave 3". The §5 tooltip row's claim that the shared "Delete" tooltip is "shared by … `widgets/pass_list.py`" is wrong for the same reason. |

---

## 3. Partial rows

| Surface | `file:line` | In inventory? | What is absent or recorded wrongly |
|---|---|---|---|
| Emoji Picker's entry paths | `exporters/telegram.py:546`, `:721-739` | **partial** | The inventory cites the opener as "`exporters/telegram.py:305-311`". Lines 301-340 are `build_render_extras`, which *defines* the `emoji_button` closure (the `set_tooltip("Click to change emoji")` at `:311` lives there). The two real call sites are never named: the **preview-box overlay** (`:545-547`, `if emoji.emoji_button(emoji.emoji, float(SIZE.ROW_HEIGHT)): emoji.open_emoji_picker(emoji.set_emoji)`) which sets the *pending* sticker's emoji, and the **per-sticker-cell overlay** `_draw_sticker_emoji` (`:721-739`), which enqueues a `set_emoji` job against an *existing* sticker. Two different targets, two different write paths; the inventory records one entry path. |
| Telegram sticker grid cell | `exporters/telegram.py:691-703` | **partial** | §8 records the delete-✕ + wash, but not that the cell passes `overlay=lambda side: self._draw_sticker_emoji(rc, slot, side)` (`:703`) — the only `preview_cell` in the codebase that uses the `overlay` parameter — nor that `result.clicked` writes `selected_index` (`:706`). |
| Telegram "Render" / status slot | `exporters/telegram.py:600-601`, `:741-763` | **partial** | §8 lists "Telegram Add-to-pack" but not the sibling `if standard_button("Render", width=render_w): rc.render()` (`:600-601`) beside it, nor the status slot whose four states include a progress bar. |
| YouTube "Render" / "Upload" | `exporters/youtube.py:456-458`, `:469-482` | **partial** | §4 records the two YouTube combos; §8 records neither verb button. `Upload` carries a four-clause gate — `imgui.begin_disabled(artifact is None or not rc.artifact_is_fresh or rs.in_flight or not size_ok)` (`:467`) — which is the kind of gating §8's "Notes" column exists to record. |
| YouTube target-panel text fields | `exporters/youtube.py:427-431` | **partial** | Title / Description / Tags inputs (`labeled_text_input`, `labeled_multiline_input`) — the metadata a user actually types before publishing. §4's combo table covers Resolution and Category from the same block and stops. |
| Share-tab outlet accordion | `tabs/share.py:91-93` | **partial** | Cited as `tabs/share.py:85-102`; the `collapsing_header` is at `:91`. The recorded note ("Skipped entirely when only one exporter is available") is right (`:70-81`), but the forced-collapse rule — `imgui.set_next_item_open(False, imgui.Cond_.always)` for every non-active header (`:89-90`) — is what makes it one-at-a-time, and is unrecorded. |
| Settings modal — Integrations section | `popups/settings.py:156-174` | **partial** | Recorded as "one collapsible `tree_node` per exporter — `draw_config_ui`; a Copilot tree node". Absent: the `imgui.set_next_item_open(True, …)` force-open that the `SettingsField` deep link drives, and the unavailable-exporter branch (`:165-167`) that draws a reason string instead of a node. |
| Graph canvas — pan / zoom | `widgets/pass_graph.py:987-997`, `:1086-1093` | **partial** | §8 lists node drag, wire drag, rubber band and the unwire badge, but not wheel-zoom (`if hovered and io.mouse_wheel != 0.0:` at `:987`, clamped to `SIZE.GRAPH_ZOOM_MIN/MAX`) or pan (middle-drag / Alt+left-drag, `:1086-1093`). Both write `view.pan` / `view.zoom`, per-document state the table's last row claims to enumerate. |
| Projects modal — Enter-to-switch | `popups/projects.py:101-107` | **partial** | The Projects entry's "Close" line records Open / Close / Esc. A fourth path exists: `if selected is not None and not app.projects_input_focused and imgui.is_key_pressed(imgui.Key.enter, repeat=False): app.request_project_switch(selected)` (`:101-107`), plus double-click on a row (`:71-92`). Neither is recorded. |

---

## 4. Inventory rows confirmed present at the cited line

Every `file:line` the inventory gives resolves to the construct it names. Eight drift by fewer than
ten lines — noted and moved past, not findings:

| Cited | Actual | Construct |
|---|---|---|
| `ui_primitives.py:1341` | `:1344` | the `"Delete"` `set_tooltip` |
| `ui_primitives.py:1567` | `:1570` | `def fps_overlay` |
| `ui_primitives.py:1560` | `:1563` | `_profile_rows` row tooltip |
| `ui_primitives.py:1735` / `1644` / `1672` | `:1738` / `:1647` / `:1675` | `clickable_label` / `draw_copyable_text` / `draw_link` tooltips |
| `ui_primitives.py:592-596` | `:589-596` | `def help_marker` |
| `pass_graph.py:848` / `872` / `1478` / `1499` | `:847` / `:871` / `:1477` / `:1498` | `_tab_row` / `_canvas_menu` / `_node_menu` / `_group_prompt` |
| `tabs/share.py:85-102` | `:85-102` spans it; header at `:91` | outlet `collapsing_header` |
| `copilot_chat.py:166` | `:162-168` | the `unconnected_gate` call |

No surface the inventory lists is absent from the code.

---

## 5. False trails — checked and correctly covered, do not re-check

- **`app.py` state-field table (§"`App` fields holding is-open / armed / in-flight state")** — every
  line number verified against `app.py`: `popup_state` 453, `copilot_revert_target` 434,
  `is_palette_open` 462, `rebinding_command` 472, `lib_reset_armed` 474, `settings_focus` 477,
  `settings_mark` 479, `pass_draft` 412, `pass_settings_name`/`_name_buf`/`_group_buf` 415-417,
  `import_draft` 419, `file_pick_dialog`/`_request` 445-446, `emoji_picker_query`/`_pick_target`
  484/486, `document_delete_armed` 487, `projects_*` 538-544, `fps_details_open` 518,
  `editor_lookup_requested`/`editor_lookup` 347/352, `is_copilot_open` 426 … `copilot_turn_active`
  441. All correct, including the "413-417" range for the three pass-settings buffers (a third
  buffer, `pass_settings_group_buf`, does exist at `:417` — the range is right).
- **Popup mutex claims** — `PopupState` at `app.py:132-141` with exactly the nine members listed;
  `any_popup_open` at `:1021-1027` ORing `copilot_revert_target is not None`; `_open_popup` at
  `:1029-1034`. Verified line by line.
- **`pass_menu_items` sharing** — `pass_list.py:62-80` is genuinely the single item list, called
  from `pass_list.py:86` and `pass_graph.py:1490`; the `deletable = len(document.passes) > 1`
  double-gate and its comment are exactly as described.
- **Esc funnel** — `hotkeys.py:354-399` matches the inventory's most-modal-first ordering and all
  four carve-outs (Pass Settings, Import Passes, Projects' `projects_input_owns_esc`, the lib
  picker's `inline_input_owns_esc`), including the `was_settings_open` apply-on-close.
- **Settings modal body** — every control the inventory names is at the cited line: keymap combo
  `settings.py:121`, the four General widgets `:80-102`, the six Editor widgets `:108-148`, the
  library armed reset `:197-220`, `_draw_keybindings` `:377-424` with `begin_disabled(not
  spec.rebindable)`.
- **Native file dialogs (§6)** — all five confirmed: `widgets/uniform.py:211`, `widgets/details.py:31`,
  `exporters/youtube.py:249`, `app.py:2399` (`pfd.select_folder` inside `pick_project_dir`), and the
  async FILE gate at `ui.py:115` polled by `_pump_file_gate` (`ui.py:97-143`). The blocking /
  non-blocking distinction the section draws is real (`pfd_block` vs `dialog.ready()`).
- **Menu bar** — `ui.py:715-751`, three dropdowns plus two direct-click bar items, hints via
  `_hint(app, CommandId.…)`. The right-aligned dim `project {name}` label (`ui.py:753-764`) is
  correctly *not* listed: `imgui.text_colored`, not a control.
- **Command palette / cheatsheet / rebinder triad (§7)** — `ui.py:640-641`, `cheatsheet.py:25`
  (foreground draw list, zero interactive elements — correct), `settings.py:377`. The claim that
  `commands.py` is the single table all three walk holds (`COMMAND_SPECS`, `in_palette` filter at
  `app.py:850`).
- **Lib-picker top bar** — search input, Favs/Reset pills and the ctrl-click tag isolate are all
  present as described (`search.py:39-50`, `:113-135`, with the ctrl-branch at `:127-132`).
- **Graph node/canvas menus and the `##graph_group` prompt** — items, order and the
  `begin_popup_context_item(None)` anchoring rationale all match `pass_graph.py:871-884`,
  `:1477-1496`, `:1498-1528`.

---

# Round 2

Anchor: the code, re-opened for every row below. Inventory read end-to-end at its current 534
lines; the sweep re-read `shaderbox/popups/**`, `shaderbox/widgets/**`, `shaderbox/tabs/**`,
`shaderbox/ui.py`, `shaderbox/ui_primitives.py`, `shaderbox/exporters/telegram.py`,
`shaderbox/exporters/youtube.py`, plus `shaderbox/notifications.py`.

## Verdict

**PARTIAL.** All 24 round-1 rows (14 missing + 1 wrong + 9 partial) are CLOSED — each is carried
by inventory text I quote below, and each quoted claim matches the code at the line it cites.
The fresh sweep found **3 new surfaces** the inventory does not record, all of them state-writing
controls with no taste question attached: the uniform-row **value editor** (the per-input-type
widget switch that is the whole point of the Uniforms panel), the code editor's own **text
surface** (caret placement, word-select, drag-select, wheel scroll), and the graph node
**double-click** (enters a group's scope — a second path to the node menu's "Open" item).

---

## 1. Closure of round-1 §1 "Missing surfaces" (14 rows)

| Round-1 row | Status | Closing inventory line | Code check |
|---|---|---|---|
| Exporter Settings config panel — Telegram | **CLOSED** | Line 39: "**Telegram's `draw_config_ui`** (`exporters/telegram.py:345-388`): not-connected branch — 4-line `setup_steps` (`:352-361`); password-masked `labeled_text_input("Bot token", ..., password=True, focus=focus)` (`:367-369`); `primary_button("Connect")` → `self.begin_auth()` (`:370-371`); `danger_button("Clear token")` → `self.disconnect()`, shown only when a token is already typed (`:374-375`); always-drawn `connection_status(...)` (`:382-388`)" | `telegram.py:345` is `def draw_config_ui`, `:354` `setup_steps(`, `:367` `labeled_text_input(`, `:370` `if primary_button("Connect")`, `:374` `if danger_button("Clear token")`, `:382` `connection_status(`. Every control present; the `have_token` guard on Clear is real (`:373`). Two drifts under ten lines, not findings: `setup_steps` cited `:352-361` vs. actual `:354-363`, and the shared Disconnect cited `ui_primitives.py:768` vs. actual **`:793`** (`if danger_button("Disconnect")` inside `connection_status`, which begins at `:770`). |
| Exporter Settings config panel — YouTube | **CLOSED** | Line 39: "**YouTube's `draw_config_ui`** (`exporters/youtube.py:269-353`) … 7-entry `setup_steps` whose first four are `(text, url)` tuples rendering clickable `draw_link` rows … `primary_button("Load client_secret.json...")` … `standard_button(paste_label)` toggling `"Paste instead"`/`"Hide"` … revealed `imgui.input_text_multiline("##yt_secret", ...)` … `primary_button("Connect")` wrapped in `begin_disabled(busy)` … `danger_button("Clear credentials")` … a `"Waiting for authorization..."` warning line while `busy`" | `youtube.py:270` `def draw_config_ui`, `:276` `setup_steps(` with 7 entries of which the first four are tuples, `:312` `primary_button("Load client_secret.json...")`, `:316-319` the `paste_label` toggle, `:324` `input_text_multiline("##yt_secret"`, `:337` `primary_button("Connect")` inside the `busy` disable at `:335`, `:342` `danger_button("Clear credentials")`, `:345-349` the busy warning, `:351` `connection_status(`. The `draw_link` rendering is in the shared `setup_steps` at `ui_primitives.py:765`. All drifts ≤ 4 lines. |
| Telegram target-panel unconnected gate | **CLOSED** | Line 114: "Full-panel replacement when `not self._is_connected()`: `unconnected_gate("Not connected to Telegram.", "Connect a bot in Settings to share stickers.", "Set up token", extras.get(_OPEN_SETTINGS_KEY))` — the CTA button routes through `render_control.extras`' `_OPEN_SETTINGS_KEY` closure to `app.open_settings(focus=self.config_field)`" | `telegram.py:399` `def draw_target_panel`, `:406-413` the exact call with those four arguments, `:414` `return` (full-panel replacement confirmed). The closure is `telegram.py:334` `_OPEN_SETTINGS_KEY: lambda: deps.open_settings(self.config_field)`. |
| YouTube target-panel unconnected gate | **CLOSED** | Line 115: "`unconnected_gate("Not connected to YouTube.", "Connect your channel in Settings to upload.", "Set up credentials", extras.get(_OPEN_SETTINGS_KEY))` — action label **differs** from Telegram's ("Set up credentials" vs "Set up token"), a genuine label drift on the same verb" | `youtube.py:382` `def draw_target_panel`, `:389-394` the call, `:395` `return`. The label drift the row names is real and is also carried in the cross-surface "Open Settings" row (line 357). |
| Copilot-chat unconnected gate | **CLOSED** | Line 116: "Full-transcript replacement when `not app.integrations_store.copilot.openrouter_key`: `unconnected_gate("Copilot is not set up.", "Add your OpenRouter API key in Settings", "Open Settings", on_action=lambda: app.open_settings(focus=SettingsField.COPILOT_KEY))`. Also drains `app.copilot_focus_pending` before drawing (`:163-166`)" | `copilot_chat.py:157` is the `if not ...openrouter_key:` branch, `:161` `app.copilot_focus_pending = False`, `:162-168` the gate call with those exact strings, `:169-170` `else: _draw_transcript(app)` — the replacement is whole-transcript, as claimed. |
| Copilot chat input + Send/Stop button | **CLOSED** | Line 117: "`imgui.input_text_multiline("##copilot_input", app.copilot_input, ..., flags=enter_returns_true\|ctrl_enter_for_new_line\|word_wrap)` (`:300-307`), frozen via `imgui.begin_disabled(in_flight)` (`:299`) … `in_flight`: `standard_button("Stop")` → `app.copilot.cancel_turn()` (`:322-323`). Idle: `primary_button("Send")` or Enter-submit, gated on non-empty stripped text (`:324-328`)" | `copilot_chat.py:299` `imgui.begin_disabled(in_flight)`, `:300-307` the multiline with those three flags, `:321-323` the Stop branch, `:324-328` `elif (primary_button("Send", width=btn_w) or submitted) and app.copilot_input.strip():`. Also newly recorded in the state table (line 409, `copilot_input: str`, app.py:442). |
| Editor tab bar (incl. ▾ tab-list popup) | **CLOSED** | Line 118: "`imgui.begin_tab_bar("##editor_tabs", flags)` (`:104`) with `TabBarFlags_.reorderable\|fitting_policy_scroll\|tab_list_popup_button\|draw_selected_overline` (`:97-101`) — `tab_list_popup_button` renders an actual imgui popup listing every open tab" + per-tab close-✕, error tint, drag-reorder and the `set_selected` programmatic drive | `code.py:104` `if imgui.begin_tab_bar("##editor_tabs", flags):`, flags built at `:96-101` with all four members including `tab_list_popup_button`, `:116-118` `begin_tab_item(..., True, item_flags)` + `if keep is not None and not keep: close_index = i`, `:128` `display_order = _display_order(app)`, `:131-134` the close/apply-order tail. Every clause holds. |
| Editor error strip | **CLOSED** | Line 119: "Clickable `imgui.selectable` rows, one per compile error … click jumps the caret via `app.editor_jump_request = JumpRequest(...)`, opening the script file first when the error is cross-file (`:361-363`) … a `"+N more"` / `"show less"` `imgui.selectable` toggle (`:369-374`) flips `app.errors_expanded`" | `code.py:332` `def _draw_error_strip`, `:355` `clicked = imgui.selectable(f"{label}##err{i}", err is at_caret)[0]`, `:361-363` the `err.path != app.current_editor_path` → `open_script_for` branch, `:364` `app.editor_jump_request = JumpRequest(...)`, `:369` the `##errmore` selectable, `:372` `app.errors_expanded = not app.errors_expanded`. `errors_expanded` is now in the state table too (line 419, app.py:560). |
| Editor chrome — "Open dir" + copyable path | **CLOSED** | Lines 309-310: "`standard_button("Open dir", width=BTN_SM_W)` → `app.open_current_document_dir()`" and "Third `draw_copyable_text` call site … `draw_copyable_text(str(local_file_path), copy_value=str(full_file_path))`, pushes a "Copied to clipboard!" toast on click" | `code.py:797-798` is exactly the Open-dir button; `:790-791` is `if draw_copyable_text(str(local_file_path), copy_value=str(full_file_path)): app.notifications.push("Copied to clipboard!")`. Both rows correct. |
| Render-tab "Render" button | **CLOSED** | Line 308: "`primary_button("Render")`, `begin_disabled(not has_path)` (`:47-48`) → defers `target.document.render_media(pending)` one frame via `app.render_defer.submit` so a "Rendering..." cue paints first; failure pushes a toast, not just a log" | `render.py:47` `imgui.begin_disabled(not has_path)`, `:48` `if primary_button("Render"):`, `:69` `app.render_defer.submit(_run_render)`, `:64-66` the `notifications.push(f"Render failed: …")`. |
| Details-panel "Choose file..." button | **CLOSED** | Line 304: "`if standard_button("Choose file..."):` → `pfd_block(pfd.save_file(...))` (native dialog, also in §6); rejects a path whose extension is outside `extensions` with a toast" | `details.py:30` `if standard_button("Choose file..."):`, `:31` `pfd_block(pfd.save_file(...))`, `:35-40` the extension rejection + `notifications.push`. |
| Details-panel resolution presets + W/H drags | **CLOSED** | Line 305: "Two one-click size verbs — `standard_button(f"{full_w}x{full_h}")` (`:94-95`, also fires when width/height is unset) and `standard_button(f"{half_w}x{half_h}")` (`:97-98`) — plus `imgui.drag_int("##width"/"##height")` (`:60-66`)" | `details.py:90` `if standard_button(f"{full_w}x{full_h}") or not details.width or not details.height:`, `:93` the half preset, `:65` / `:69` the two `drag_int`s. Small drifts (`:90` vs. cited `:94-95`, `:93` vs. `:97-98`, `:65-69` vs. `:60-66`) — under ten lines, not findings. |
| Document-grid "New document" + "Render all" | **CLOSED** | Lines 306-307: "`standard_button("New document")` → `app.create_document_from_example(STARTER_EXAMPLE_ID)`, wrapped in `begin_disabled(app.copilot_turn_active)` (`:44,47`)" and "`imgui.checkbox("Render all", ...)` → `app.app_state.is_render_all_documents`; carries the §5 tooltip" — with the tooltip itself now at line 200 | `document_grid.py:44` `imgui.begin_disabled(app.copilot_turn_active)`, `:45-46` the button + call, `:47` `end_disabled()`, `:51-53` the checkbox, `:55-59` the `begin_tooltip` with the exact sentence. The round-1 complaint that the tooltip was claimed-but-absent is fixed: it is now §5's last row. |
| Lib-tree per-leaf favorite star | **CLOSED** | Line 147 (inside §3's function-leaf menu entry, as a "**Sibling inline control, not part of this menu**"): "every function-leaf row also draws its own one-click favorite star directly on the row (`tree.py:324-328`, `imgui.small_button(("*" if is_fav else "o") + f"##fav_{fn.name}")` → `app.shader_lib_favorites.toggle(fn.name)`, colored `COLOR.FAVS`/`COLOR.FG_DIM`)" — plus a cross-surface row at line 358 | `tree.py:323-328` is exactly that: `star_label = ("*" if is_fav else "o") + f"##fav_{fn.name}"`, the `push_style_color(… COLOR.FAVS if is_fav else COLOR.FG_DIM)`, and `if imgui.small_button(star_label): app.shader_lib_favorites.toggle(fn.name)`. |

## 2. Closure of round-1 §2 "Wrong claim" (1 row)

| Round-1 row | Status | Closing inventory line | Code check |
|---|---|---|---|
| Pass-tile corner delete-✕ described as a "dead control" | **CLOSED** | Line 274: "**No glyph drawn — not a control.** `_draw_pass_tile` passes `deletable=False` (`pass_list.py:151`) and `armed=False` (`:141`); `preview_cell` guards the ✕ on `if deletable:` (`ui_primitives.py:1338`) and the confirm wash on `if selected and armed:` (`:1324`), so neither is submitted. Fixed in commit `97841ec`" | `pass_list.py:151` reads `deletable=False,` and `:141` reads `armed=False,`; `ui_primitives.py:1324` is `if selected and armed:` and `:1337` is `if deletable:` (cited `:1338`, a one-line drift). The §5 "Delete" tooltip row was corrected too — line 177 now says "**Not** `widgets/pass_list.py`: the pass tile passes `deletable=False` (`pass_list.py:151`), so it never reaches this tooltip", and the cross-surface delete row (line 361) carries the same exclusion. |

## 3. Closure of round-1 §3 "Partial rows" (9 rows)

| Round-1 row | Status | Closing inventory line | Code check |
|---|---|---|---|
| Emoji Picker's two entry paths | **CLOSED** | Line 66: "from **two** real call sites, not one: the **pending-sticker preview overlay** (`telegram.py:545-547`, `if emoji.emoji_button(emoji.emoji, ROW_HEIGHT): emoji.open_emoji_picker(emoji.set_emoji)`) … and the **per-existing-sticker-cell overlay** `_draw_sticker_emoji` (`telegram.py:721-739`, passed as `preview_cell`'s `overlay=` param at `:703`) … Two different targets, two different write paths (`set_emoji` direct vs. `_Job(kind="set_emoji", ...)` enqueued)" | `telegram.py:544-546` is the `_overlay` closure with that exact body (3-line drift); `:718-738` is `_draw_sticker_emoji`, ending in `emoji.open_emoji_picker(lambda e: self._enqueue(_Job(kind="set_emoji", …)))`. The `overlay=` hand-off is `:703`. Both paths and both write shapes match. |
| Telegram sticker grid cell overlay + `selected_index` | **CLOSED** | Line 316: "The only `preview_cell` call in the codebase passing `overlay=lambda side: self._draw_sticker_emoji(...)` (`:703`); `result.clicked` writes `self._render_state.selected_index = idx` (`:706-707`)" | `telegram.py:703` `overlay=lambda side: self._draw_sticker_emoji(rc, slot, side),` and `:705-706` `if result.clicked: self._render_state.selected_index = idx`. A codebase-wide grep for `overlay=` into `preview_cell` returns this one call site, so the uniqueness claim holds. |
| Telegram "Render" + status slot | **CLOSED** | Line 314: "`standard_button("Render", width=render_w)` → `rc.render()`, sits beside "Add to pack" on the same row; a constant-height status slot below (`_draw_status_slot`, `:741-763`) holds four states including a progress bar, so the row never resizes mid-render" | `telegram.py:600-601` is the Render button, `:602-603` the `same_line()` + `_draw_add_button`, `:740` `def _draw_status_slot`. Confirmed. |
| YouTube "Render" / "Upload" | **CLOSED** | Lines 317-318: "`standard_button("Render")` → `rc.render()`, beside Upload" and "`primary_button("Upload")`, `begin_disabled` behind a four-clause gate: `artifact is not None and rc.artifact_is_fresh and not rs.in_flight and size_ok` (`:463-467`) → enqueues an upload job" | `youtube.py:456` `if standard_button("Render"):`, `:458` `imgui.same_line()`, `:461-466` the four-clause `upload_enabled`, `:467-468` `if not upload_enabled: imgui.begin_disabled()`, `:469` `if primary_button("Upload") and artifact is not None:`. The gate is expressed as a computed bool rather than an inline `begin_disabled(...)` expression, but all four clauses and the disable are as described. |
| YouTube target-panel text fields | **CLOSED** | Line 319: "Title (`labeled_text_input`), Description (`labeled_multiline_input`), Tags comma-separated (`labeled_text_input`) — typed ahead of Upload, alongside the Resolution/Category combos already in §4" | `youtube.py:427` `rs.title = labeled_text_input("Title", …)`, `:428-430` `rs.description = labeled_multiline_input("Description", …)`, `:431` `rs.tags_raw = labeled_text_input("Tags (comma-separated)", …)`. |
| Share-tab outlet accordion forced collapse | **CLOSED** | Line 325: "One-at-a-time is enforced by force-collapsing every NON-active header each frame — `imgui.set_next_item_open(False, Cond_.always)` (`:89-90`) — while the active header is left free to toggle (else it could never close)" | `share.py:89-90` `if not is_active: imgui.set_next_item_open(False, imgui.Cond_.always)`, `:91-93` `header_open = imgui.collapsing_header(...)`, `:69-80` the single-outlet skip. The code's own comment states the same rule. |
| Settings modal — Integrations section | **CLOSED** | Line 39: "one collapsible `imgui.tree_node(exporter.display_name)` per **available** exporter … an unavailable exporter draws a dim reason string instead of a node, `f"{exporter.display_name} — {exporter.unavailable_reason}"` (`:165-167`), no node at all; a pending `app.settings_focus` … force-opens its tree node via `imgui.set_next_item_open(True, Cond_.always)` (`:158-159`) … a Copilot tree node with the same force-open rule (`:169-173`)" | `settings.py:156-167` is the exporter loop with `if exporter.is_available:` / `else: imgui.text_colored(COLOR.FG_DIM, f"{…} — {…unavailable_reason}")`; the force-open is `:159-160`; the Copilot node with its own force-open is `:169-174`. Both absences round 1 named are now recorded. |
| Graph canvas — pan / zoom | **CLOSED** | Lines 282-283: "`if hovered and io.mouse_wheel != 0.0:` — zoom about the cursor position, clamped to `SIZE.GRAPH_ZOOM_MIN/MAX`; writes `view.zoom` + `view.pan` (per-document `GraphViewState`)" and "Middle-drag or Alt+left-drag (`bg_active and (mouse_down(middle) or io.key_alt)`); writes `view.pan`" | `pass_graph.py:987` `if hovered and io.mouse_wheel != 0.0:`, `:991-993` the `GRAPH_ZOOM_MIN/MAX` clamp + `view.zoom`, `:994-997` the pan rewrite; `:1086-1088` `panning = bg_active and (imgui.is_mouse_down(MouseButton_.middle) or io.key_alt)`, `:1089-1093` the `view.pan` write. |
| Projects modal — Enter-to-switch + double-click | **CLOSED** | Line 85: "double-click switches, gated on `not app.projects_input_focused`, `:78-83`" and "Enter-to-switch when a row is selected and no name input is focused, `:101-107`, returns before drawing the row"; line 87's Close list now reads "Open (switch, via button, Enter, or double-click)" | `projects.py:76-83` is the double-click branch (`selected and not app.projects_input_focused and is_mouse_double_clicked(0) and is_item_hovered()` → `request_project_switch`), `:100-107` the Enter branch ending in `return False`. Both paths and both gates match. |

**Round-1 closure score: 24 / 24 CLOSED.** Nothing from round 1 remains open. The only drifts
found while re-checking are sub-ten-line citation offsets plus one larger one worth naming for
the accuracy pass rather than as a surfaces finding: the shared Disconnect button is at
`ui_primitives.py:793`, not the cited `:768` (line 39) — `:768` falls inside `unconnected_gate`,
a different primitive.

---

## 4. Fresh sweep — surfaces the inventory does not carry

Three. Each has an opener or a state write the inventory records nowhere; none is a "does a text
field count" judgement call.

### N1 — Uniform-row value editor (the per-input-type widget switch)

`shaderbox/widgets/uniform.py:239-390`, inside `draw_uniform_control`.

The control a user actually turns to change a uniform. It is a seven-branch switch on
`ui_uniform.input_type`, each branch a different widget:

- `auto` → `clipped_caption` readout, no control (`:239-240`);
- `buffer` → `standard_button("Randomize")` + a byte-count caption (`:242-254`);
- `array` → `imgui.input_text(hidden, value_str)` parsing a comma-separated list, truncated to
  `ui_uniform.array_length` (`:258-268`);
- `text` → `imgui.input_text_multiline(hidden, text, size=(UNIFORM_CTRL_W, UNIFORM_TEXT_H))`
  round-tripping through `str_to_unicode` (`:272-283`);
- `texture` → the sampler combo + preview (`:285-357`);
- `color` → `getattr(imgui, f"color_edit{ui_uniform.dimension}")(hidden, list(current_value))`
  (`:359-364`) — a click-to-open **color picker popup**, imgui's own;
- `drag` → `imgui.drag_int` / `drag_float` / `drag_float{2,3,4}` (`:366-383`).

Plus the behaviour that ties them together, at `:385-390`:

```
if playing and imgui.is_item_activated():
    app.set_uniform_stopped(document_id, panel_pass_name, name, True)
    playing = False
```

Grabbing the value widget **auto-stops a playing uniform** — a state write (`set_uniform_stopped`,
persisted per uniform) triggered by touching the control, and the exact reason the row's
play/stop toggle can flip without the user clicking it.

The inventory records this row's *satellites* — Play/stop (line 293), Input-type cycle chip
(line 294), "Randomize" (line 295), Texture-preview click (line 296), Uniform-name clickable
label (line 297) — and the two clipped-caption tooltips (line 178), but never the value editor
itself. `color_edit{N}` is also the only imgui-owned popup in the app that §2 does not name
(the completion popup, the other library-owned one, is recorded at line 113).

### N2 — Code-editor text surface (caret, selection, wheel)

`shaderbox/tabs/code.py:1019-1025` (the hit-rect) and `:805-836` (`_handle_mouse`), `_handle_wheel`.

```
imgui.invisible_button(
    "##editor_surface",
    imgui.ImVec2(editor_size.x, max(1.0, editor_size.y - cell_h)),
)
```

The app's largest interactive region. It carries: press → `editor.set_cursor(pos.line,
pos.column)` + `app.editor_drag_anchor` (`:824-827`); **double-click → word-select**, fed to the
vim layer as `editor.feed("viw")` when the mode is NORMAL (`:821-823`); left-drag →
`editor.set_selection(anchor, head)` (`:828-834`); wheel → `_handle_wheel` scroll; and
`imgui.is_item_activated()` → `app.editor_was_ever_focused = True` (`code.py:1027-1028`), the
latch two *recorded* surfaces gate on (the Help modal's Insert button, inventory line 33; the
lib picker's, line 79). The rect is deliberately one row short of the image bottom so a click on
the mode badge cannot place a caret (`:1010-1017`).

The inventory names the editor's chrome (tab bar, error strip, Open dir, copyable path), its
overlays (`K`-lookup note, completion popup, cursor-following uniform tooltip at line 185), and
its hotkeys — but never the editing surface. It is in §9.5's `invisible_button` allowlist
("four `invisible_button` hit-rects (`tabs/code.py`, …)", line 479) as a *button-tier exception*,
which records that the call exists, not what it does or what it writes.

### N3 — Graph node double-click

`shaderbox/widgets/pass_graph.py:1120-1122`, dispatching to `_double_click` at `:1466-1475`.

```
if imgui.is_item_hovered():
    node_hovered = node.key
    if imgui.is_mouse_double_clicked(imgui.MouseButton_.left):
        _double_click(app, document_id, view, node)
```

On a `box` node it writes `view.scope = node.group`, `view.fitted = False` and
`view.selection = set(node.members)` (`:1469-1472`) — **it enters the group's scope**, the same
verb as the node context menu's "Open" item (`:1484-1486`, recorded at inventory line 131). On a
`ghost` node it re-dispatches to `_click` (`:1474-1475`).

The inventory records the node menu's "Open", the node *single*-click (line 359, as one of the
three `choose_output` paths), node drag (line 280) and the scope tab row (line 324) — but not
the double-click. It is therefore a two-surface verb ("Open" a group: menu item + double-click)
that the cross-surface table at line 331 does not list, and `view.scope`'s second writer that the
`GraphViewState` row (line 425) does not name.

---

## 5. False trails — checked in round 2 and correctly covered, do not re-check

- **Telegram pack management** — the §2 row at line 110 cites `telegram.py:451-475` / `:483-500`,
  and the *arming buttons* round 2 went looking for are inside those ranges: `standard_button("New
  pack")` at `:451` toggling `pack_create_armed`, `danger_button("Delete pack")` at `:458` toggling
  `pack_delete_armed`, the `input_text("##new_pack")` at `:466`, `primary_button("Create")` at
  `:470`, and `_draw_delete_confirm`'s Delete/Cancel at `:494`/`:498`. Covered, not a finding.
- **Telegram / YouTube Duration drag** (`telegram.py:581`, `youtube.py:442` `labeled_drag_float`)
  — a plain value drag on a panel already inventoried control-by-control; a taste call, not a
  surface with an opener or unrecorded state. Left alone deliberately.
- **Document rename field** (`tabs/document.py:348` `input_text_with_hint("##document_name", …)`)
  — same class: a plain text field writing `ui_state.ui_name`, no opener, no armed state.
- **`media_ops.draw_video_filters` smoothing drags** (`media_ops.py:21,29`) — the block's verb,
  `standard_button("Apply##video_to_video_smoothing")`, IS recorded (inventory line 312, with the
  `trash_dir` write); the two `drag_int`/`drag_float` sliders feeding it are plain value inputs on
  a recorded control. Not a finding.
- **FPS detail panel rows** (`ui_primitives.py:1540` `_profile_rows`, `:1570` `fps_overlay`) — read
  end-to-end: the panel is a bordered child of `add_text`/`add_rect` rows plus the row tooltip
  already at inventory line 182. The chip itself is the only control, as line 298 says.
- **Notifications / toasts** (`shaderbox/notifications.py:32-62`) — pure `imgui.text_colored`
  bottom-right, no item submitted, no dismiss control. Correctly absent from the inventory as a
  surface; the three places that *push* a toast are recorded on their own rows (lines 304, 308,
  310).
- **The popups package** — `emoji_picker.py`, `examples.py`, `help.py`, `import_passes.py`,
  `pass_settings.py`, `projects.py`, `settings.py` and all five `lib_picker/` modules were swept
  control by control against §1/§3/§4/§5. Every control found has an inventory home, including
  the ones round 1 never reached: Import Passes' per-reader handover checkboxes
  (`import_passes.py:242`, line 58), Pass Settings' `smooth`/`repeat` checkboxes
  (`pass_settings.py:266,271`, line 48 "sampling/edges checkboxes"), Settings' ten
  `input_int` Copilot-limit rows (`settings.py:344`, line 39), the lib-tree `tree_node_ex`
  expanders (`tree.py:75,112,256`, line 76's "left tree column (dir/file/function)"), and the tag
  editor's remove-pills and suggestion-pills (`preview.py:60-102`, line 109).
- **`copilot_chat.py` and `pass_graph.py`** — swept end-to-end. Every control resolves to an
  inventory row: the chat's gate Yes/No, credential input, Allow/Deny and the exporter config
  panel embedded in a gate (line 287), the result-widget open buttons (line 288), `step_squares`'
  hover breakdown (line 194), the context gauge (line 181), the splitter (line 292); the graph's
  port drags, output-dot wire drags, rubber band, wire-select, Delete/Backspace unwire, the
  unwire badge, both context menus and the group prompt (lines 106, 278-283). N3 is the single
  exception.
