# Accuracy review — `04_menus_inventory.md`

Anchor: the two rulebooks (`/imgui-ui` SKILL §1, §2, §7; `conventions.md ## Design decisions`)
plus a direct read of every file the inventory cites. Every line number below was read, not
recalled.

**Verdict: PARTIAL.** The inventory's structure, its roster of surfaces, and the great majority
of its `file:line` anchors are right — the 133-hit coverage claim reproduces exactly, all eight
`PopupState` modals really are in `ui.py`'s popup block, and every §3 context menu matches the
code. But it carries **12 wrong or unverifiable claims** (2 of them substantive enough to
mislead a design pass outright) and **9 classes of missing fact** that the design pass needs and
cannot get from the document as written.

---

## 1. Wrong or unverifiable claims

Format: `surface | claim | what the code actually does (file:line)`.

### Substantive (a design reviewer would draw a wrong conclusion)

1. **Pass-tile corner delete-✕ (§8 row 1, and the cross-surface "Delete/confirm a row" row)** |
   "**Dead control**: `preview_cell` draws the delete-✕ (and would-be confirm wash) whenever
   `selected=True`, and `_draw_pass_tile` passes `selected=is_output`, but only reads
   `result.clicked` — `delete_armed`/`delete_confirmed`/`delete_cancelled` are silently dropped.
   Clicking the glyph does nothing." |
   **No glyph is drawn at all.** `_draw_pass_tile` passes `deletable=False`
   (`shaderbox/widgets/pass_list.py:151`) and `armed=False` (`:141`). `preview_cell` guards the
   ✕ on `if deletable:` (`shaderbox/ui_primitives.py:1338`) and the confirm wash on
   `if selected and armed:` (`:1324`), so neither is submitted. There is no dead control and
   nothing to click. The cross-surface table's "pass-tile instance is non-functional" repeats
   the same error. (The working tree is clean — `git diff` is empty — so this is not a
   post-inventory edit.)

2. **Copilot flags (cross-surface App-fields table, last row)** | "`copilot.state.in_flight` …
   distinct from `app.copilot_turn_active`, which gates pass/graph/document-grid instead — **the
   two flags never overlap in this codebase**" |
   They are a **mirror, not a disjoint pair**: `shaderbox/ui.py:461` assigns
   `app.copilot_turn_active = app.copilot.state.in_flight` every frame (with a falling-edge
   carve-out at `:457`). After reconciliation they hold the same value; `copilot_turn_active` is
   the one-frame-lagged App-side copy, not an independent gate. A third, unrelated `in_flight`
   lives on the Telegram exporter's own render state
   (`shaderbox/exporters/telegram.py:166`), which the row does not distinguish.

### Anchor and detail errors

3. **Projects modal — opener** | "`App.open_projects` (`app.py:581` / `_open_popup(...)` near
   `app.py:2410`)" | `open_projects` is defined at `shaderbox/app.py:2403` and its `_open_popup`
   is at `:2409`. `app.py:581` is a different thing entirely — a **bare**
   `self.popup_state = PopupState.PROJECTS` on the dead-pointer recovery path
   (`shaderbox/app.py:575-589`), deliberately set before `_init` and then followed by a real
   `open_projects()` call at `:589`.

4. **Copilot revert confirm — draw fn** | "`copilot_chat.py:177` `_draw_revert_modal`" |
   `def _draw_revert_modal` is at `shaderbox/widgets/copilot_chat.py:176`.

5. **Command palette (§7)** | "`imcmd.command_palette_window("CommandPalette", ...)`
   (`ui.py:640-641`)" | The call is at `shaderbox/ui.py:644-646`. Lines 640-641 are
   `draw_lib_picker(app)` / `draw_projects(app)`.

6. **Import Passes — Esc carve-out** | "`hotkeys.py:379-380`" | The
   `PopupState.IMPORT_PASSES` branch is at `shaderbox/hotkeys.py:378-379`.

7. **Reveal in file manager (cross-surface "verbs from more than one surface")** | "identical
   text, **duplicated implementation**" | The two `menu_item_simple` call sites are separate
   (`popups/lib_picker/tree.py:152` and `:274`), but both call the **one** shared
   `App.reveal_shader_lib_file_in_manager` (`shaderbox/app.py:2138`). The implementation is not
   duplicated.

8. **Play/stop (cross-surface table)** | tooltip text "Stop the whole script" | The actual
   string is **"Whole script is stopped"** (`shaderbox/widgets/uniform.py:171`). §5 quotes it
   correctly; the cross-surface row does not.

9. **`menu_item_simple` label table** | presented as "labels across the app (sorted)" |
   **"Arrange" is missing.** It is a graph-canvas menu item
   (`shaderbox/widgets/pass_graph.py:882`), and §3 names it — the table does not. Enumerating
   the literal-label call sites over `shaderbox/` yields 18 literals plus three dynamic
   (armed-delete pairs, "Leave group"); the table lists 21 rows and omits this one.

10. **Help — Insert button gating** | "bottom row `primary_button("Insert at caret")` (only when
    `section.insertable` …)" | The condition is `section.snippet and section.insertable`
    (`shaderbox/popups/help.py:76`) — a section with `insertable=True` and no snippet draws no
    button.

11. **Document tab "add pass" button** | "(`document.py:466-...`)" in §1 and "`466-481`" in §8 |
    `standard_button("add pass")` is at `shaderbox/tabs/document.py:477` and
    `standard_button("import...")` at `:480`. Line 466 is an `imgui.end_disabled()` belonging to
    an earlier block; `_draw_passes` runs 471-481.

12. **Shader Library Picker — Esc suppression** | "Esc … suppressed while a rename/new-file/
    new-dir inline input owns it (`inline_input_owns_esc`, `__init__.py:32-42`)" — presented as
    one predicate | There are **two different predicates, deliberately**. `inline_input_owns_esc`
    (`popups/lib_picker/__init__.py:32-42`, read by `hotkeys.py:386-390`) includes
    `picker_tag_input_focused`; the picker's own in-body Esc gate uses `input_owns_keys`
    (`:77`, consumed at `:155`), which **excludes** the tag input — the comment at `:151-154`
    says the tag input must not suppress Esc. Collapsing them hides a real behavioral split.

Minor off-by-ones, listed for completeness and not counted above: `uniform.py:211` (the
`_pick_media_file` def is at `:212`), `youtube.py:249` (the def is at `:247`, `:249` is the
`pfd_block`), `ui_primitives.py:1735` (the tooltip check is at `:1739`), `ui.py:768-784` for the
copilot toggle (actual `:767-783`), `app.py:413-417` for the pass-settings buffers (actual
`:412`, `:415-417`), `telegram.py:668-719` (`_draw_grid_cell` starts at `:667`),
`pass_graph.py:1490` for the node menu's shared items (`pass_menu_items` is called at `:1489`).

---

## 2. Missing fact classes a design reviewer needs

These are the rulebook questions the inventory does not answer. Each is read from the code below
so the design pass can use it directly.

### A. Action-row shape (rulebook §7.1) — recorded for no surface

The inventory never says, for any modal, whether the action row is at the bottom, whether the
primary is left of Close, or whether the `dummy(0, SPACE.MD)` spacer is there. Measured:

| Modal | Action row | Primary left / close right | `dummy(SPACE.MD)` above |
|---|---|---|---|
| Examples | bottom, `popups/examples.py:80-88` | yes | **no** (`:79` goes straight from the desc slot) |
| Help | bottom, `popups/help.py:76-91` | yes | **no** |
| Settings | bottom, `popups/settings.py:190-192` | Close alone | yes (`:188`) |
| Pass Settings (edit) | bottom, `pass_settings.py:137` | Close alone | yes (`:136`) |
| Pass Settings (draft) | bottom, `pass_settings.py:102-104` | yes | yes (`:101`) |
| Import Passes | bottom, `import_passes.py:83-94` | yes | yes (`:81`) |
| Emoji Picker | bottom, `emoji_picker.py:75` | Close alone | yes (`:74`) |
| Shader Lib Picker | bottom, `lib_picker/__init__.py:134-149` | yes | **no** (`imgui.spacing()` at `:123`) |
| Projects | bottom, `projects.py:113-132` | yes; Close **right-anchored** (`:130`) | yes (`:54`) |
| Copilot revert | bottom, `copilot_chat.py:195-200` | yes | yes, `SPACE.SM` (`:193`) |

So three modals (Examples, Help, Shader Lib Picker) skip the §7.1 spacer, and Projects is the
only one that right-anchors Close.

### B. Close vs Cancel labelling (rulebook §7.1) — never checked against the rule

- "Close" (browser/view-only): Examples, Help, Settings, Pass Settings **edit**, Emoji Picker,
  Shader Lib Picker, Projects. All correct per the rule.
- "Cancel" (form whose commit mutates): Pass Settings **draft** (`pass_settings.py:104`),
  Import Passes (`import_passes.py:93`), Copilot revert (`copilot_chat.py:198`), Projects'
  new-name inline row (`projects.py:182`), Settings' library-reset confirm (`settings.py:220`),
  Telegram's delete-pack confirm (`telegram.py:498`). All correct.
- The one label to question: **Import Passes uses "Cancel"** even though nothing is mutated
  until Import — defensible (a draft exists), worth a design call.

### C. `keep_open` shape (rulebook §7.3) — two divergences

- `popups/settings.py:190` names the flag **`is_keep_opened`**, which §7.3 calls out by name as
  the wrong spelling ("Don't invert the boolean — `keep_open` reads better than
  `is_keep_opened`"). Every other modal uses `keep_open`.
- **The copilot revert modal has no body-returns-bool at all** — it closes inline inside the
  body (`copilot_chat.py:194-200`), the only modal in the codebase not following the
  wrapper/`keep_open` shape. The inventory records its close paths but not this structural
  difference.

### D. Close-funnel cleanup is bypassed by the generic Esc branch

`hotkeys._handle_escape`'s fallthrough sets `app.popup_state = PopupState.CLOSED` **directly**
(`shaderbox/hotkeys.py:389`), skipping each modal's wrapper-branch cleanup. Consequence the
inventory does not record: **`emoji_pick_target` is nulled only at
`popups/emoji_picker.py:23`**, so an Esc-close leaves the callback dangling (it is the only
writer besides `open_emoji_picker` at `app.py:1264`). Rulebook §7.6 names exactly this. By
contrast Projects *does* survive it — `reset_projects_state()` runs at `popups/projects.py:44`
and is also called from `open_projects` (`app.py:2405`) — and Pass Settings / Import Passes have
explicit carve-outs.

### E. State reset on open (rulebook §7.6) — recorded for none

- Reset on open: Projects (`app.py:2405` `reset_projects_state()`), Shader Lib Picker
  (`app.py:1270-1273` — `reset_inline_state()`, query, tag focus), Settings
  (`app.py:1048` `lib_reset_armed = False`), Emoji Picker (`app.py:1265` query), Help
  (`app.py:1277` section), Import Passes (`app.py:1168` fresh `ImportDraft`).
- **No reset on open**: Examples keeps `app_state.selected_example_id` across opens
  (deliberate — it is persisted app state, `popups/examples.py:69`).

### F. Inline inputs: commit-on-deactivate (rulebook §7.5)

| Inline input | Enter | Deactivate | `x` cancel |
|---|---|---|---|
| Pass Settings name | yes (`pass_settings.py:163`) | **yes** (`:166`, read on the next line) | Cancel button (draft only) |
| Pass Settings group | yes (`:187`) | **yes** (`:189`) | — |
| Projects new-name | yes (`projects.py:168`) | **no** — only Enter or the `New` button (`:180`) | "Cancel" button (`:182`), plus Esc (`:172`) |
| Lib tree rename / new file / new dir | `popups/lib_picker/tree.py:290`, `:198` | (see file) | reserved `x` per §7.5 (`tree.py:292-293`) |
| Graph `##graph_group` name | yes (`pass_graph.py:1509`) | no | "Cancel" (`:1536`) |

The Projects row is a genuine §7.5 divergence: a transaction that only Enter fires is silently
discarded when the user clicks away. The inventory records the surface but not the commit rule.

### G. Context-menu discoverability hint (rulebook §7.4) — recorded for none

Exactly **one** hint exists in the app: `imgui.text_colored(COLOR.FG_DIM, "Right-click for
actions")` above the lib tree (`popups/lib_picker/__init__.py:116`). The pass strip, the graph
canvas and the graph nodes — all three of which carry right-click menus (§3) — have **no hint**.
This is the single most design-relevant omission in the document.

### H. `menu_item_simple(enabled=False)` Python-side gate (rulebook §7.4)

The inventory praises the pass menu for gating in Python (`pass_list.py:75`,
`... and deletable`) but does not check the other `enabled=` site:
`popups/lib_picker/tree.py:358` draws `menu_item_simple("Insert at caret", enabled=has_editor)`
and calls `insert_name(app, fn)` with **no Python guard**. On this imgui-bundle build that item
can still register a click while disabled — the exact failure `pass_list.py:72-73`'s comment
documents.

### I. Raw `imgui.button` / hand-rolled `push_style_color` at call sites (rulebook §1)

The inventory never states which surfaces style their own buttons. The repo already answers
this mechanically and the design pass should start from the answer, not re-derive it:

- `tests/test_button_tiers.py:25-54` is the gate; its `_NOT_A_VERB` allowlist enumerates every
  sanctioned raw button with a written reason — `popups/emoji_picker.py` ("one emoji cell in the
  glyph grid"), `exporters/telegram.py` (the glyph button + carousel arrows),
  `popups/lib_picker/tree.py` (the favorite star), plus four `invisible_button` hit rects
  (`tabs/code.py`, `ui.py`, `widgets/copilot_chat.py`, `widgets/pass_graph.py`).
  `test_the_tier_set_stays_at_four` pins the tier count; `test_every_listed_exception_still_exists`
  kills stale entries.
- Call-site `push_style_color` that is *not* covered by that gate and that the inventory glosses
  over: `popups/lib_picker/tree.py:165-167` and `:279-281` (red text on the armed delete item),
  and `exporters/telegram.py:484-485` (the hand-rolled red delete-confirm child, which the
  inventory does flag as hand-rolled but not as a tier/theming question).

### J. Word budgets (rulebook §2) — no surface is measured

The repo has a gate here too (`tests/test_ui_prose_budget.py`; `conventions.md` line 61), with
an allowlist (`_OVER_BUDGET` / `_EXEMPT`, `tests/test_ui_prose_budget.py:582`). The strings a
design pass will want to see named:

- `popups/help.py:85` disabled-Insert tooltip, 13 words ("Open a document's shader and click
  into the editor first (so the caret is positioned)") — against a 5-word tooltip budget.
- `popups/lib_picker/__init__.py:141`, 9 words, same class.
- The ten `_COPILOT_LIMITS` hints (`popups/settings.py:225-320`) are 20-35 words each with
  em-dash-joined second clauses — e.g. `:231-233` — against an 8-word `help_marker` budget.
- `popups/settings.py:206-209` (library-reset warning) and `copilot_chat.py:189-192` (revert
  caption) are multi-sentence prose inside a modal body.

The Pass Settings help markers, by contrast, are all inside budget
(`pass_settings.py:88`, `:202`, `:243`, `:263`, `:292`).

---

## 3. Sections 4-8 and the cross-surface tables — spot-check results

Every third row checked against the code; all rows below verified **correct** unless named in
§1 above.

- **§4 combos**: canvas presets (`tabs/document.py:43`), sampler source
  (`widgets/uniform.py:312`), YouTube resolution (`exporters/youtube.py:421`) and category
  (`:437`), Telegram pack (`exporters/telegram.py:437`), details output type
  (`widgets/details.py:112`) and quality (`:121`) — all land on the right call with the right
  options and the right write target. Two facts the table omits: the canvas-presets control is
  deliberately **a chip, not a combo** per its own comment (`tabs/document.py:38-39` — relevant
  because rulebook §1 puts chips outside the tier count), and the Import Passes entry-point
  combo is `begin_disabled(is_output)` (`import_passes.py:196`).
- **§5 tooltips**: `clickable_label` (`ui_primitives.py:1739`), `clipped_caption` (`:580`),
  `gauge_bar` (`:1057`), `draw_copyable_text` (`:1644`), code-editor uniform value
  (`tabs/code.py:1128`), Copy (`copilot_chat.py:538`), Deny (`:632`), Layout (`:714`), emoji
  glyph (`telegram.py:311`), snippet tooltip (`copilot_chat.py:402`), help Insert
  (`help.py:84`), emoji entry (`emoji_picker.py:65`), Reset document
  (`tabs/document.py:246`) — all correct.
- **§6 native dialogs**: all five verified, including the async FILE-gate
  (`ui.py:97-143`, picker opened at `:115-117`, state fields at `app.py:445-446`). The
  blocking/non-blocking contrast is stated correctly.
- **§7 commands**: every chord verified against `shaderbox/commands.py` — Alt+E examples
  (`:229`), F1 help (`:230`), Alt+S settings (`:220`), Alt+P pass settings (`:222-226`), Alt+A
  add pass (`:227`), IMPORT_PASSES unbound (`:228`, chord `0`), Alt+L lib picker (`:212`),
  Alt+O projects (`:109`), Ctrl+Shift+P palette (`:215-219`), Alt+/ cheatsheet (`:231-237`).
  The Esc funnel's most-modal-first ordering is exactly as described
  (`hotkeys.py:354-399`).
- **§8 inline controls**: document-grid delete (`widgets/document_grid.py:92-99`), sticker-grid
  delete (`exporters/telegram.py:667-719`, `selected` gated on `not in_flight` at `:695-696`),
  wire-✕ (`pass_graph.py:620`), play/stop (`widgets/uniform.py:157-180`), media-ops Apply
  (`widgets/media_ops.py:37-54`), share accordion (`tabs/share.py:85-102`, single-outlet
  skip at `:68-80`), uniforms pass selector (`tabs/uniforms.py:37-54`) — all correct.
- **App-fields table**: every line number verified against `shaderbox/app.py`; only the
  pass-settings buffer range is off (see §1 minor list). `shader_lib_files` at `:567`,
  `fps_details_open` at `:518`, `editor_lookup*` at `:347`/`:352`, `editor_completion_*` at
  `:330-361`, the projects cluster at `:537-544` — all correct.

---

## 4. False trail — what was checked and is right

So the next reader does not re-verify these:

- **The coverage claim reproduces exactly.** Running the inventory's own grep over `shaderbox/`
  returns **133** hits, the stated number.
- **The popup mutex is described correctly.** `PopupState` has the nine stated members
  (`app.py:132-141`); `any_popup_open()` (`:1021-1027`) really does OR in
  `copilot_revert_target is not None`; `_open_popup` (`:1029-1034`) is the one opener funnel;
  all eight `draw_*` calls are present in `ui.py`'s popup block (`ui.py:630-637`), which
  `conventions.md`'s `popups/*.py` bullet names as the step that gets forgotten.
- **`pass_menu_items` really is shared and cannot drift.** One definition
  (`pass_list.py:62-80`), two callers (`pass_list.py:83-87` with an explicit id;
  `pass_graph.py:1478-1496` with `begin_popup_context_item(None)`), and the id-vs-`None`
  reasoning matches the module docstring (`pass_graph.py:23-24`) and rulebook §8.
- **All three lib-tree context menus** match item-for-item at `tree.py:142-174`, `:269-287`,
  `:354-371`, and none of them checks `copilot_turn_active` (grep over `shaderbox/popups/`
  returns nothing) — exactly as the inventory says.
- **Both copilot-turn brackets are real**: `pass_list.py:170`/`:227` and
  `pass_graph.py:907`/`:922` (plus the canvas's own `frozen` gesture cancel at `:942-950`).
- **`close_pass_settings` is the one funnel** and does commit a pending rename and group edit
  before clearing (`app.py:1081-1110`), as claimed.
- **The `##graph_group` prompt is genuinely hand-rolled** — plain `imgui.begin_popup`
  (`pass_graph.py:1504`), no `modal_window`, no `context_menu_style` — and it does one-shot
  focus on `is_window_appearing` (`:1506-1507`).
- **The label-case drift findings are real**: `"add pass"` / `"import..."`
  (`tabs/document.py:477`, `:480`) against `"Add pass"` / `"Import..."`
  (`pass_graph.py:875`, `:877`).
- **`modal_window` is the mandated wrapper and every modal uses it** —
  `ui_primitives.py:318-352`, with `Cond_.first_use_ever` by default and `Cond_.always` only
  under `fixed_size=True` (Examples). No surface hand-rolls `begin_popup_modal`.

---

## Round 2

Anchor: a direct read of every file the fold-in touched, plus the two rulebooks again. Every
line number below was opened, not recalled — including the round-1 corrections themselves, one
of which turned out to be wrong (see §R2.3).

**Verdict: PARTIAL.** All nine missing fact classes (A-J) are now carried, most of them in the
new §9, and the two substantive round-1 errors are genuinely fixed. But **4 of the 12 round-1
claims are still open** (three anchors the fold-in did not apply, one wrong tooltip string that
survived verbatim), and the ~170 lines of new text introduce **12 wrong claims of their own** —
two of them substantive. One further defect sits in the §7.1 closing sentence (R2.4).

---

### R2.1 Closure of round-1 wrong claims

| # | Round-1 claim | Status | Inventory line carrying it |
|---|---|---|---|
| 1 | Pass-tile delete-✕ "dead control" | **CLOSED** | L274 now reads "**No glyph drawn — not a control.**", cites `deletable=False` (`pass_list.py:151`), `armed=False` (`:141`), and `preview_cell`'s two guards (`ui_primitives.py:1338`, `:1324`). All four verified. The cross-surface row (L358) repeats the corrected version. |
| 2 | Copilot flags "never overlap" | **CLOSED** | L415 now says "**Mirrored into, not disjoint from**", cites `ui.py:461` and the `:457` falling-edge carve-out, and names the third Telegram `in_flight` (`telegram.py:166`). Verified at `ui.py:457-461`. |
| 3 | Projects opener `app.py:581` | **CLOSED** | L84 gives `app.py:2403-2410` with `_open_popup` at `:2410`, and explains `:581` as the bare dead-pointer-recovery assignment followed by a real `open_projects()` at `:589`. Verified. (Round 1 said `_open_popup` is at `:2409`; it is at `:2410` — the inventory is right and round 1 was off by one.) |
| 4 | Revert modal draw fn `:177` | **STILL OPEN — and round 1 was wrong** | L92 now says `copilot_chat.py:176`. The actual `def _draw_revert_modal` is at **`:177`**; `:176` is blank and `:174` is `_REVERT_MODAL_LABEL`. The **original inventory was correct**; round 1 "corrected" it to a wrong value and the fold-in adopted it. |
| 5 | Palette `ui.py:640-641` | **CLOSED** | L221 gives `ui.py:644-646`. Verified. |
| 6 | Import-Passes Esc `hotkeys.py:379-380` | **STILL OPEN** | L60 still reads `hotkeys.py:379-380`. The `IMPORT_PASSES` branch is at **`:379-380`** — round 1's "`:378-379`" was itself off by one; `:379` is the `elif`, `:380` the call. So L60 is right and round 1 was wrong. Marked closed-by-accident. §9.2 (L470) independently says `:378-379`, which is **wrong**. |
| 7 | Reveal "duplicated implementation" | **CLOSED** | L367 now says "identical text, single shared implementation — not duplicated", citing `app.py:2138`. Verified: two call sites (`tree.py:156`, `:275`), one method. |
| 8 | Play/stop tooltip "Stop the whole script" | **STILL OPEN** | The cross-surface row (L360) still reads `("Stop the whole script" vs. "Stop this uniform")`. See R2.3 #1 — this is now a *different* error than round 1 diagnosed. |
| 9 | "Arrange" missing from label table | **CLOSED** | The table now carries `| Arrange | graph canvas menu |`. Verified at `pass_graph.py:882`. Enumerated all 22 `menu_item_simple` call sites; the table's 22 rows match one-for-one. |
| 10 | Help Insert gating | **CLOSED** | L30 now reads "only when `section.snippet and section.insertable`". Verified at `help.py:76`. |
| 11 | "add pass" `document.py:466-...` | **STILL OPEN** | L47 still reads `(document.py:466-...)`. §8 (L303) *is* fixed (`:471-481`, `:477`, `:480`, all verified) — so the same fact is right in one place and wrong in another. |
| 12 | Lib-picker Esc, one predicate vs two | **CLOSED** | L78 now spells out both predicates with the reason. Verified: `inline_input_owns_esc` `:32-42` (read by `hotkeys.py:386-389`), `input_owns_keys` `:77` consumed at `:155`, comment `:151-154`. |

Round-1 minors: `uniform.py:211` (actual `pfd_block` at `:210`), `youtube.py:249` (actual `:248`),
`ui_primitives.py:1735` (actual tooltip `:1738`), `pass_graph.py:1490` (actual `pass_menu_items`
call `:1489`), `app.py:413-417` (actual `:412`, `:415-417`) — **all five still uncorrected** at
L208, L210, L176, L47 and L406 respectively. `ui.py:768-784` → `:767-783` **was** fixed (L254).

### R2.2 Closure of missing fact classes A-J

| Class | Where it now lives | Status |
|---|---|---|
| A. Action-row shape | §9.1 col 2 | **CLOSED**, all ten rows verified. Spacers, Close-alone, and Projects' right-anchor (`projects.py:130-131`) all correct. |
| B. Close vs Cancel | §9.1 col 3 | **CLOSED**, verified against every button. The Import-Passes design call is carried. |
| C. `keep_open` shape | §9.1 col 4 + L41 | **CLOSED**. `is_keep_opened` at `settings.py:190` verified; revert modal's inline close verified at `copilot_chat.py:195-202`. |
| D. Esc bypass vs close-funnel | §9.2 | **CLOSED** as a section, but two of its rows are wrong — see R2.3 #4, #5. |
| E. Reset on open | §9.1 col 5 | **CLOSED**, all six resets verified at their `app.py` openers. |
| F. Inline-input commit | §9.1 col 6 | **CLOSED** but one cell is wrong — see R2.3 #2. |
| G. Context-menu hint | §9.3 | **CLOSED**. Sole hint at `lib_picker/__init__.py:116` verified; no hint in `pass_list.py` / `pass_graph.py` (grepped). |
| H. `enabled=` Python gate | §9.4 | **CLOSED**, both rows verified (`pass_list.py:75` guards, `tree.py:359` does not). |
| I. Raw button / hand-rolled styling | §9.5 | **CLOSED**. `_NOT_A_VERB` (7 entries) and both test names verified in `tests/test_button_tiers.py:25-53,93,100`. |
| J. Word budgets | §9.6 | **CLOSED** as a section, but the two word counts are wrong — see R2.3 #3. |

**`unknown` resolved.** §9.2's last row marks Examples / Help / Settings `unknown`. The code
settles all three: each modal's `keep_open`-false branch does nothing beyond
`popup_state = CLOSED` (`examples.py:63-65`, `help.py:29-31`) — so there is no cleanup for the
bypass to skip. Settings' one extra step is `apply_editor_settings()` (`settings.py:66-69`), and
the Esc path replicates it explicitly via the `was_settings_open` latch
(`hotkeys.py:366,398-399`). **All three survive the bypass**; the row can be filled in.

### R2.3 Wrong claims in the new text

Substantive first.

1. **§5 play/stop tooltip row (L180) and the cross-surface row (L360)** | §5 lists the strings as
   `("Whole script is stopped" / "Stop this uniform" / "Resume this uniform")` and attributes all
   three to `tabs/document.py:443-448` **and** `widgets/uniform.py:178`. | The document-tab
   toggle's own strings are **"Stop the whole script" / "Resume the whole script"**
   (`tabs/document.py:446`) — two strings §5 does not list. The three §5 *does* list are the
   uniform row's alone (`widgets/uniform.py:170-175`). So the surface carries **five** strings
   across two call sites, and the table merges them into one row of three. The cross-surface row's
   "Stop the whole script" is the real document-tab string — right text, wrong attribution (it
   is paired against "Stop this uniform" as if both came from one scope-varying tooltip).

2. **§9.1 Shader Lib Picker row, inline-input cell** | "commit-on-deactivate behavior for each
   lives in `tree.py` per-input (**not centrally recorded**)". | It **is** recorded, and
   identically in both: `wants_commit = changed or imgui.is_item_deactivated_after_edit()` at
   `tree.py:224` (new file/dir) and `tree.py:304` (rename), each under the same comment. Both
   inline inputs **do** commit on deactivate. The cell should say so — it is the rulebook §7.5
   answer the column exists to give, and the picker is compliant.

Anchor and detail errors in new text:

3. **§9.6 word counts** | "`help.py:85` … 13 words" and "`lib_picker/__init__.py:141` … 9 words".
   | The strings are `"Open a document's shader and click into the editor first (so the caret is
   positioned)"` = **15 words**, and `"Click into the code editor first (so the caret is
   positioned)"` = **11 words**. Both still blow the 5-word tooltip budget, so the finding holds;
   the numbers do not.

4. **§9.2 Import Passes row** | "`hotkeys.py:378-379`". | The branch is at `:379-380`
   (`elif app.popup_state == PopupState.IMPORT_PASSES:` / `app.close_import_passes()`). §1's L60
   has it right.

5. **§9.2 Shader Lib Picker row** | "`hotkeys.py:386-390`". | The `elif` runs `:386-389` and its
   body (`popup_state = CLOSED`) is `:390`. The fallthrough the section's own lead sentence cites
   as "`hotkeys.py:389`" is actually at **`:390`**.

6. **§1 Telegram block (L39)** | the shared `connection_status` Disconnect "is a real
   `danger_button("Disconnect")` inside the shared primitive (`ui_primitives.py:768`)". |
   `danger_button("Disconnect")` is at **`ui_primitives.py:793`**; `:768` is blank and
   `connection_status` starts at `:770`.

7. **§2 lib-picker tag-editor row (L109)** | "`popups/lib_picker/preview.py:69-104`
   `_draw_function_tag_editor`". | The def is at **`preview.py:55`**.

8. **§9.1 Import Passes row, inline-input cell** | "n/a (**no free-text inline input in the
   body**; the group-name field has no distinct commit-on-deactivate note in the code)". | There
   **is** one: `imgui.input_text("##import_group", draft.group_buf)` at `import_passes.py:74`.
   It writes the draft buffer every frame, so there is no commit edge to record — but the cell's
   stated reason is false, and §1's own Items line (L58) lists "group name input".

9. **§9.1 graph-group prompt footnote** | "Cancel button explicit (`:1536`)". | The
   `standard_button("Cancel")` is at **`pass_graph.py:1526`**. (The Enter claim, `:1509`, and the
   no-deactivate-commit claim are both correct — `_group_prompt` contains no
   `is_item_deactivated_after_edit()`.)

10. **§2 editor-chrome / copilot-gate / completion anchors (L113, L116, L310)** | three
    call-site anchors in new text point at neighbouring code. | `draw_copyable_text` in
    `draw_chrome` is at **`code.py:788-789`**, not `:790-792` (`:790-792` is the "(unsaved)"
    branch); `_drive_completion` is at **`code.py:599`**, not `:597`; and the copilot gate's
    `app.copilot_focus_pending = False` drain is at **`copilot_chat.py:161`**, not `:163-166`
    (those four lines are the `unconnected_gate` call's own kwargs).

11. **§2 editor tab-bar row (L118)** | "error-tinted push/pop … (`:113-121,131`)" and
    "`_display_order`/`_apply_display_order` (`:120,134`)". | The tint push block is
    `code.py:112-115` and its `pop_style_color(3)` is `:127-128`; `:131` is
    `if close_index is not None:`. `_display_order(app)` is called at **`:129`**, not `:120`
    (`:120` is the close-✕ `keep` test). `_apply_display_order` at `:134` is correct.

12. **§2 error-strip row (L119)** | "the caret's own error drawn selected (`:346-347,355`)". |
    The `err is at_caret` selectable is at **`code.py:354`**; `:346-347` is the unrelated
    `f"{n} errors"` header and `:355` is a `pop_style_color`. The jump anchor (`:361-363`) is
    exact.

Off-by-ones in new text, listed and not counted: `youtube.py:269-353` (def at `:270`),
`youtube.py:463-467` for the Upload gate (actual `:460-464`), `code.py:329-374` for
`_draw_error_strip` (def at `:332`; its inner claims at `:355`, `:363`, `:367-372` are exact),
`details.py:60-66` for the W/H drags (actual `:65-69`), `ui.py:115-117` for the FILE-gate picker
(actual `:114-116`), `projects.py:115-118` / `:120-122` / `:123-125` (actual `:112-116` /
`:118-120` / `:122-124`), `projects.py:96-132` (def at `:95`), `ui_primitives.py:1567` for
`fps_overlay` (def `:1570`), `:1560` for `_profile_rows`' tooltip (actual `:1563`), `:1644` for
`draw_copyable_text`'s (actual `:1647`), `:1672` for `draw_link`'s (actual `:1675`),
`copilot_chat.py:189-192` for the revert caption (actual `:190-193`) and `:193` for its
`SPACE.SM` spacer (actual `:194`), `settings.py:161` for `draw_config_ui` (actual `:162`),
`settings.py:158-159` for the force-open (actual `:160`), `settings.py:220` for the
library-reset Cancel (actual `:218`), `telegram.py:691-719` for `_draw_grid_cell` (def at
`:668`; §8's own `:668-719` is right) and `:706-707` for its `selected_index` write (actual
`:705-706`), `youtube.py:308-312` for the client-secret loader (`focus_field` `:311`,
button `:312`, call `:314`), and `youtube.py:322-327` for the paste box (input `:324-326`,
commit `:328`).

### R2.4 False trail — new text checked and correct

So the next reader does not re-verify these:

- **§7.1's 32-row `COMMAND_SPECS` table is correct end to end.** Every id, label, chord, category
  and scope matches `commands.py:108-238`, including the two that look like errors and are not:
  `CYCLE_CODE_TAB` really is category EDITOR with `scope=GLOBAL` (`:152-158`, comment explains),
  and `IMPORT_PASSES` really is `_chord`-less `0` (`:228`). `LEADER_BINDINGS` `<leader>f` verified
  at `:255-258`.
- **One real defect in that table's closing sentence (L266)**: "`repeat` and `rebindable` are both
  `True` for every spec above except where noted". `repeat` defaults to **`False`**
  (`commands.py:88`) and **no spec overrides it** — so `repeat` is False for all 32. `rebindable`
  and `in_palette` default True and are never overridden, so those two halves are right.
- **The 133-hit coverage claim reproduces exactly** on a re-run of the inventory's own grep.
- **The popup dispatch block** really does carry all eight `draw_*` calls (`ui.py:630-637`), and
  `modal_window` is `ui_primitives.py:318-352` with the `Cond_` split as described.
- **Every new §1 exporter claim about Telegram's config UI** checks out: `setup_steps` `:354`,
  bot-token input `:367-369`, Connect `:370-371`, Clear token `:374-375`, `connection_status`
  `:382-388`, the unconnected gate's three literal strings `:407-413`, `EmojiControl` `:327`,
  the `"Click to change emoji"` tooltip `:311`, and both emoji-picker call sites (`:545-547`
  pending preview, `_draw_sticker_emoji` `:721-739` passed as `overlay=` at `:703`).
- **YouTube's gate label really does differ** from Telegram's ("Set up credentials" vs "Set up
  token"), both verified at their `unconnected_gate` calls — the label-drift finding is real.
- **The copilot chat input row** matches `:299-328` exactly: `begin_disabled(in_flight)` `:299`,
  the multiline input `:300-307`, Stop `:322-323`, Send `:324-328`.
- **§9.4's two rows are the only `enabled=` sites in the codebase** — grepped all 22
  `menu_item_simple` calls; exactly two pass `enabled=`.
- **§9.5's allowlist is complete and current**: `_NOT_A_VERB` has seven entries
  (`tests/test_button_tiers.py:25-53`), both named tests exist (`:93`, `:100`), and the three
  call-site `push_style_color` sites outside the gate (`tree.py:165-167`, `:279-281`,
  `telegram.py:484-485`) are all real.
- **§9.1's Projects commit divergence is genuine**: `_draw_name_input` (`projects.py:157-194`)
  has Enter (`:167-168`), the verb button (`:180`) and Cancel (`:182`) but **no**
  `is_item_deactivated_after_edit()` — clicking away silently discards. Contrast Pass Settings,
  which has it on both fields (`:166`, `:189`).
- **The App-fields table** verifies line-for-line against `app.py` — all 30-odd rows — with the
  single exception of the uncorrected `413-417` noted in R2.1.
- **Graph §8 rows** verified: wheel-zoom `:987-997`, pan `:1086-1093`, wire delete `:1347-1358`,
  `graph_state.py:85-86`/`:88`/`:79`/`:105` for the named `GraphViewState` fields.
