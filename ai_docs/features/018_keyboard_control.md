# 018 — keyboard_control

The **command layer**: fire named actions by chord or by name, see them, rebind them. Two axes on one
spine (a central command registry):
**(A) rebindable chord shortcuts + an always-on cheatsheet**, **(B) a command palette**
(name-search any action). Pure verb dispatch — no notion of "where focus is."

The **focus/navigation layer** — `nav_enable_keyboard` (Tab/arrows through widgets) + region-cycling
(editor↔panel, Node/Render/Share tabs) — is a SEPARATE deferred feature (it shares the awkward bits:
`active_node_tab` state, fighting the editor caret, arbitrating arrow keys). Tracked in `todo.md`.

Staged so each axis is reviewable on its own: A is the foundation (registry + dispatch + cheatsheet +
rebinding); B is an additive layer reading the same registry.

---

## Goal

- Let the user fire ShaderBox **commands** without a mouse: open-project / save / new-node /
  delete-node / settings / quit / open-lib-picker / jump-to-next-error / node-prev / node-next.
- Make the keybindings **discoverable** without opening this spec: an always-visible (opt-out)
  cheatsheet showing the chords valid in the current context.
- Make the keybindings **rebindable** and persisted via the existing `UIAppState` machinery.
- Add a **command palette**: one chord opens a fuzzy-search box over every registered command's human
  label; Enter runs it. Covers actions whose chord the user forgot or never bound.
- One registry is the single source of truth — it feeds dispatch, the cheatsheet, the palette, AND the
  menu-bar hint strings (kill the hand-typed `"Ctrl+N"` literals in `ui.py`).

## Out of scope (each with a trigger)

- **The whole focus/navigation layer** — `nav_enable_keyboard` (Tab/Shift+Tab + arrow widget
  traversal + keyboard slider editing) AND region/tab-cycling (editor↔panel, Node/Render/Share). These
  belong together: region-cycling needs an `active_node_tab` field + `TabItemFlags_.set_selected`
  driving, and both fight the editor caret + the node-arrow commands this feature ships, so they need
  a unified focus model + a dedicated manual-verification wave (nav is app-wide: it silently changes
  Tab/arrow/Space/Enter on every standard widget — the node-preview grid, the lib-picker tree, the
  emoji grid, every `input_text`, the share-tab fields). Filed as `todo.md [DEFERRAL] keyboard
  focus/navigation layer`. **Trigger:** when the command layer (018) is landed and the user wants
  mouse-less *widget* interaction (cycling uniform controls, region/tab switching). *(Both pieces
  were prototyped headlessly this session: `nav_enable_keyboard` is settable and `io.nav_active` goes
  True; `TabItemFlags_.set_selected` exists — so the deferred feature starts from confirmed APIs.)*
- **Chord capture for mouse buttons / gamepad.** The rebinding capture widget records keyboard chords
  only (mods + one keyboard `imgui.Key`). **Trigger:** a user asks to bind a mouse button.
- **Multi-key sequences (Emacs `C-x C-s`).** Single chord per command. **Trigger:** the command count
  outgrows the single-chord space and collisions force sequences.
- **Per-command palette sub-prompts beyond the one we ship.** `imgui_command_palette` supports two-step
  commands (`initial_callback` → `imcmd.prompt([...])` → `subsequent_callback(idx)`). We use it for
  exactly **"New node → pick template"**. **Trigger:** a third action wants a sub-prompt (then
  generalize the two-step wiring).
- **Conflict *resolution* UI** (auto-suggest a free chord, swap-on-collide), AND cross-scope
  duplicate detection. The rebinder only **detects + warns** on a duplicate chord *within the same
  scope* and refuses to bind. A chord deliberately shared across DIFFERENT scopes (e.g. an EDITOR and
  a GLOBAL command) is allowed and unwarned — by design, since the dispatcher fires each chord in
  exactly one active scope (decision 4), so a cross-scope share can't double-fire. **Trigger:**
  warn-only rebinding friction is reported.

## Design decisions (numbered, lock-in)

1. **The command registry is the spine.** A new leaf module `commands.py` holds:
   - `CommandId` — a `StrEnum` of stable string ids (`"open_project"`, `"save"`, `"new_node"`,
     `"delete_node"`, `"open_settings"`, `"open_lib_picker"`, `"open_palette"`, `"quit"`,
     `"jump_next_error"`, `"node_prev"`, `"node_next"`, `"toggle_cheatsheet"`). String-valued so it
     round-trips through JSON as the persistence key (decision 6).
   - `CommandScope` — a `StrEnum`: `GLOBAL` (fires anywhere, but explicitly suppressed while a modal
     popup is open — decision 4), `EDITOR` (only when the code editor child is focused),
     `NODE_CREATOR` (only when that popup is open). Scope drives the suppression gate AND the
     cheatsheet filter.
   - `CommandSpec` — a frozen dataclass: `id: CommandId`, `label: str` (human; palette + cheatsheet +
     menu hint), `default_chord: int` (an imgui KeyChord int — built from `imgui.Key` members, e.g.
     `int(imgui.Key.s) | int(imgui.Key.mod_ctrl)`; **note the mod symbol is `imgui.Key.mod_ctrl`, NOT
     `imgui.Mod_ctrl` — the latter does not exist in this bundle**; `0` = unbound), `scope:
     CommandScope`, `repeat: bool` (arrow-style auto-repeat). **No callback in `CommandSpec`** — see
     decision 2.
   - `COMMAND_SPECS: list[CommandSpec]` — the static default table (no `App` reference, leaf-clean).
   - `chord_to_str(chord: int) -> str` — builds `"Ctrl+Shift+P"` from the mod bits + `get_key_name`
     (**display only**, never persistence — decision 8). Lives here (not `util.py`, which is imgui-free;
     not `ui_primitives.py`, which is for *draw* helpers — `chord_to_str` returns a `str`).
   - `popup_suppresses(scope: CommandScope) -> bool` — pure helper the dispatcher uses (decision 4).

2. **`CommandSpec` carries NO callback; the id→callback binding is built at `App` init.** The
   cycle-from-types split the dev-flow names: `commands.py` is a leaf (imports `imgui` only), so it
   cannot reference `App`. The wiring `dict[CommandId, Callable[[], None]]` lives on `App` (built in
   `App.__init__`, closing over `self`), e.g. `CommandId.SAVE: self.save`. A `commands.py` table
   referencing `app.save` would force `commands.py` to import `App` while `App` imports `commands` →
   cycle. **Lock:** the *what* (specs) is a leaf; the *how* (callbacks) lives with the state that owns
   the verbs.
   - *Agent note (corrected from review):* `app.py` ALREADY imports imgui (`from imgui_bundle import
     imgui` + `imgui_color_text_edit`), so the App-verb layer is **not** imgui-free and this feature
     does not change that. What genuinely serves the future coding-agent is the *leaf* split:
     `commands.py` is imgui-only-importing (no `App`), and `ui_models.py` stores chords as raw `int`
     (no imgui). The earlier draft over-claimed "keeps the verb layer imgui-free / helps the agent" —
     struck. The stale `todo.md` clause naming `app.py` as a must-stay-imgui-free module is reconciled
     in this wave (it never was; the requirement is scoped to a *future node-ops module* +
     `shader_lib/file_ops.py`, which ARE clean).

3. **Dispatch lives in `hotkeys.py`, rewritten to drive the registry via `imgui.shortcut()`, and is
   CALLED FROM INSIDE THE FRAME (after `imgui.new_frame()`).** ← **critical correction from review.**
   `imgui.shortcut()` asserts (`imgui.cpp:10465`, reproduced) if called *outside* an active frame.
   Today `process_hotkeys(app)` is called at `ui.py:190`, BEFORE `imgui.new_frame()` (`ui.py:194`) —
   the legacy `is_key_pressed`-based dispatch works pre-frame, but `shortcut()` does NOT. So:
   - **Split the current `process_hotkeys` into two calls.** The pre-frame part stays where it is:
     `glfw.poll_events()` + `app.imgui_renderer.process_inputs()` (these MUST run before `new_frame`).
     The registry `shortcut()` dispatch + the ESC handler (decision 5) move to a new
     `dispatch_commands(app)` called from *inside* the main window, after `new_frame`. (`is_key_pressed`
     / `io.key_*` used by ESC work in both positions — confirmed — so co-locating ESC with the
     registry dispatch in-frame is safe.)
   - **Placement is load-bearing: call `dispatch_commands` at the TOP of the main-window `begin`
     block, BEFORE `code_tab.draw` (the editor child at `ui.py:218-221`).** ESC sets
     `editor_defocus_requested`, whose only consumer is `tabs/code.py` (after `editor.render()`, same
     frame). Today ESC runs pre-frame so the editor consumes it that frame; a top-of-block in-frame
     call preserves that — anything after the editor child would lag defocus one frame.
     `apply_editor_settings()` on the ESC-from-Settings path is a direct call (timing-independent),
     and the `is_*_open=False` writes are consumed by the popup draws later in the same block — all
     safe at top-of-block.
   - Body of the in-frame dispatch: for each `CommandSpec` whose effective chord ≠ 0, if its scope is
     active (decision 4) call `imgui.shortcut(chord, flags=route_flag(scope) | (Repeat if repeat))`;
     on True, invoke `app.command_callbacks[spec.id]()`.
   - The `_jump_to_next_error(app)` helper stays (it's the verb `jump_next_error`'s callback calls).

4. **Scope suppression is EXPLICIT, not "routing for free."** ← corrected from review (the earlier
   draft wrongly claimed `route_global` reproduces the `any_popup_open()` guards; it does not —
   `route_global` is documented to fire *through* a modal unless another scope claims the exact chord,
   and the four ShaderBox popups are true `begin_popup_modal` modals that claim nothing). The honest
   model:
   - `GLOBAL` commands dispatch with `route_global` **AND** are skipped entirely while
     `app.any_popup_open()` (the explicit gate the current code already uses — preserved verbatim, not
     replaced). This reproduces today's `not any_popup_open()` guards on F8 / Ctrl+P / arrows.
   - `EDITOR` commands dispatch with `route_focused` **AND** gated on `app.editor_focused` (O1
     default). Routing's job here is narrow: arbitrate a chord that exists in BOTH editor and global
     scope so the editor wins when focused — it does NOT replace the manual focus flag.
   - `NODE_CREATOR` commands dispatch only when `app.is_node_creator_open` (replaces the current
     arrow-nav + Enter-to-create block, which becomes three NODE_CREATOR-scoped commands).
   - **No command may fire in two scopes the same frame.** The guarantee is NOT that the scope
     active-conditions are mutually exclusive — GLOBAL (no-popup) and EDITOR (editor-focused) *both*
     hold during normal editing. The guarantee is that **each chord is registered in exactly one
     scope** (each `CommandSpec` has one `scope`), so the same chord is never dispatched twice. In the
     shipped table EDITOR scope is currently UNPOPULATED (the arrow commands `node_prev`/`node_next`
     are GLOBAL, folding the current `not editor_focused` guard into the GLOBAL active-condition), so
     a same-chord collision is structurally impossible today — but the "one chord, one scope" rule is
     what must hold if an EDITOR command is ever added. Verified by manual check (no double-fire).

5. **ESC stays a bespoke, non-rebindable handler** (moved in-frame with the dispatcher per decision 3).
   ESC's job — close any popup, defocus editor, apply-editor-settings-on-close (the modal-FPE quirk) —
   is structural app-state plumbing, not a user command. NOT in the registry, NOT rebindable, NOT
   listed by the rebinder UI.

6. **Persistence: `key_bindings: dict[str, int]` + `show_cheatsheet: bool` on `UIAppState`, one new
   migration gen.** `key_bindings` maps `CommandId` value → chord int. On load, `App` merges
   `app_state.key_bindings` over the spec defaults into `effective_bindings` it dispatches from
   (absent key → spec default; explicit `0` → user-unbound). Adding two new fields with defaults works
   with `extra="forbid"` (once in `model_fields`, the unknown-key drop in `load_and_migrate` leaves
   them; old states get the defaults) — additive, **non-breaking → minor version bump**.
   - **Persist only diffs-from-default** (review note): `key_bindings` must hold ONLY the bindings
     that DIFFER from the spec default, so "absent = default" stays meaningful across saves (otherwise
     `model_dump` freezes the full map on first save and a future default change can't reach an old
     state). This needs **custom logic in `UIAppState.save`** (compute the diff before/at dump time —
     `model_dump()` alone serializes whatever's in the field) OR the rebinder maintains `key_bindings`
     as a diff-only dict directly (write a differing entry; reset-to-default removes the key). Edge
     (harmless, noted): unbinding (`0`) a command whose spec default is ALSO `0` produces no diff, so
     it's indistinguishable from untouched — fine, the command was unbound either way.

7. **Cheatsheet = a new `widgets/cheatsheet.py` floating corner overlay, opt-out via
   `UIAppState.show_cheatsheet: bool = True`.** Floating overlay (NOT a reserved footer row) → zero
   layout cost, can't shift panels (jitter rule). Reads `effective_bindings` + `COMMAND_SPECS`, filters
   to commands whose scope is active *now* (GLOBAL when no popup; EDITOR when `editor_focused`;
   NODE_CREATOR when that popup is open), renders `label : chord-str` rows via `theme` tokens + a
   `ui_primitives` row helper. **Absolute positioning is confined to its own `begin_child` / draw-list
   path** so it can't trip the SetCursorPos assert (imgui-ui §3/§4). Default corner: **bottom-right**
   (O3). A "Show keyboard cheatsheet" toggle lives in Settings AND is a registry command
   (`toggle_cheatsheet`) so it's keyboard-reachable.

8. **Command palette: `imgui_command_palette`, one `ContextWrapper` owned by `App`, commands
   registered once at init.** A GLOBAL registry command `open_palette` (default **Ctrl+Shift+P**, O4 —
   avoids the `Ctrl+P` lib-picker collision) toggles `app.is_palette_open`. When open, `ui.py`'s popup
   section calls `app.is_palette_open = imcmd.command_palette_window("CommandPalette",
   app.is_palette_open)` — **the widget returns the new open-state by value (no out-param); read the
   return to clear the flag.** Each palette `imcmd.Command.initial_callback` calls the SAME
   `app.command_callbacks[id]()` the chord dispatch uses (one verb, two entry points), registered from
   the SAME `COMMAND_SPECS` table (no second list). "New node" is the one two-step command
   (`initial_callback` → `imcmd.prompt([template labels])` → `subsequent_callback(idx)` → create that
   template).
   - **NOT in the popup-mutex `any_popup_open()` set** — the palette is a transient floating search
     box, not one of the four modal popups. It may coexist with a modal (it isn't in any opener's
     mutex-clear list); the spec accepts this (opening it doesn't close the others). The
     palette-over-modal keyboard-input interaction is **manual-only** to verify (imgui modal
     focus-trapping may swallow palette input).
   - *Headless prototype (this session):* construction, `add_command`, `command_palette_window`, and
     `set_next_command_palette_search` render without crashing in our glfw backend. The
     row-select→`initial_callback` path is **manual-only** (no synthetic Enter into the C++ input box
     headlessly — confirmed the callback does not fire headless).

9. **No new `App` verbs beyond registry plumbing + two flags + one toggle.** Most commands map to
   existing verbs (`open_project`, `save`, `delete_current_node`, `open_node_creator`, `open_settings`,
   `open_shader_lib_picker`, `select_next_current_node`, `create_node_from_selected_template`). New on
   `App`: `command_callbacks: dict[CommandId, Callable[[], None]]`, `effective_bindings: dict[CommandId,
   int]`, `is_palette_open: bool`, `palette_ctx: imcmd.ContextWrapper`, and `toggle_cheatsheet()`. The
   rebinder mutates `app_state.key_bindings` AND re-merges `effective_bindings` immediately (so a
   rebind takes effect the same frame, not after restart — review gap).

## Files touched

- **`shaderbox/commands.py`** *(new, leaf)* — `CommandId` / `CommandScope` (StrEnum), `CommandSpec`
  (frozen dataclass), `COMMAND_SPECS`, `chord_to_str`, `route_flag(scope) -> InputFlags`,
  `popup_suppresses(scope)`. Imports `imgui` only.
- **`shaderbox/hotkeys.py`** — keep `process_hotkeys` doing ONLY the pre-frame `poll_events` +
  `process_inputs`; add `dispatch_commands(app)` (registry `imgui.shortcut()` dispatch + the ESC block)
  called in-frame; keep `_jump_to_next_error`.
- **`shaderbox/ui.py`** — call `process_hotkeys` before `new_frame` (unchanged slot) and
  `dispatch_commands` inside the main window after `new_frame`; call `imcmd.command_palette_window`
  in the popup section when `is_palette_open` (read its return into the flag); draw the cheatsheet
  overlay; **menu-bar hints read `chord_to_str(effective_bindings[id])`** instead of literal strings.
- **`shaderbox/app.py`** — build `command_callbacks` + `effective_bindings` in `__init__`; add
  `is_palette_open`, `palette_ctx`, `toggle_cheatsheet`; merge `key_bindings` over defaults; rebinder
  re-merge.
- **`shaderbox/ui_models.py`** — `UIAppState.key_bindings: dict[str, int] = {}` +
  `show_cheatsheet: bool = True`; new migration-gen comment; diff-from-default persistence in `save`.
- **`shaderbox/widgets/cheatsheet.py`** *(new)* — `draw(app)` floating overlay.
- **`shaderbox/popups/settings.py`** — "Show keyboard cheatsheet" toggle + the **rebinding section**
  (a row per rebindable `CommandSpec`: label, current chord display, a "Rebind" capture button that
  enters one-frame capture mode reading `io.key_mods` + the first pressed non-mod `imgui.Key`;
  duplicate-in-scope detection warns + refuses to bind).
- **`shaderbox/theme.py`** — cheatsheet overlay tokens (corner padding, bg fade, row spacing) + any
  rebinder-row tokens.
- **`shaderbox/ui_primitives.py`** — a cheatsheet-row / chord-pill draw helper (an imgui+theme draw
  reused by the overlay AND the rebinder rows → belongs here per the "don't repeat a widget" rule).
- **`pyproject.toml`** — none expected (`imgui_bundle` already ships `imgui_command_palette`); confirm.
- **`scripts/smoke.py`** — **mandatory** (not "optional") new invariants: `effective_bindings` is
  populated; force the cheatsheet overlay to draw on at least one frame (exercise its no-assert path
  headlessly); call the in-frame `dispatch_commands` every frame (a callback throw surfaces via
  `smoke.py::main`'s `except` — there is NO whole-frame `except` in `ui.py::update_and_draw`, so the
  value here is that `shortcut()` runs inside an active frame, which is what catches the round-1
  pre-frame assert). Also **fix the existing `_check_invariants` popup
  sum to include `is_shader_lib_picker_open`** (it currently sums only 3 of the 4 modals — pre-existing
  gap surfaced in review; fold the one-line fix into this wave).

## Manual verification (only the maintainer's hands — `make run`)

1. Every default chord fires its command (open/save/new/delete/settings/quit/lib-picker/jump-error/
   node-prev/node-next).
2. **Popup suppression:** with a modal open (node creator / settings / emoji / lib picker), GLOBAL
   chords (jump-error, lib-picker, node-arrows) do NOT fire through it; close it and they fire again.
3. **No cross-scope double-fire:** a chord that conceptually exists in both editor and global context
   fires exactly once per press in each context (editor-focused → editor action only; no-popup,
   editor-unfocused → global action only).
4. `Ctrl+Shift+P` opens the palette; typing filters; Enter runs the highlighted command; "New node"
   shows the template sub-prompt and creates the picked one. *(Palette select→callback is the
   load-bearing manual check — headless can't press Enter into the C++ widget.)*
5. Cheatsheet overlay shows the right rows per context (editor-focused vs node-creator-open vs
   default); toggling it off via the `toggle_cheatsheet` command hides it AND the Settings checkbox
   reflects that change (both drive the one `show_cheatsheet` field); toggling via the Settings
   checkbox likewise persists across restart; the overlay never trips an assert (no crash on any frame
   it's visible — verify in BOTH a no-popup and a popup-open context, since the scope-filter changes
   the rows drawn).
6. **Rebind:** capture a new chord in Settings → the new chord fires **immediately (no restart
   needed)**, the old chord doesn't, the cheatsheet + menu hint update live, and it persists across
   restart. Binding a duplicate chord in the same scope warns and refuses.
7. `make smoke` green; `make check` 0 errors.

## Open questions for the user (defaults applied; confirm at plan-lock)

- **O1 — editor-scoped dispatch location.** **Default applied:** all `imgui.shortcut()` calls in the
  in-frame `dispatch_commands`, editor-scope gated on `app.editor_focused` (one dispatch site, small
  blast radius; routing arbitrates the global-vs-editor chord contest). Alternative: move editor
  shortcuts physically into `tabs/code.py`'s draw for native focus routing (spreads dispatch). Confirm
  the default.
- **O3 — cheatsheet corner + default-on.** **Default applied:** bottom-right, default-ON (opt-out).
  Confirm.
- **O4 — palette chord.** **Default applied:** `Ctrl+Shift+P` (matches VSCode/Sublime; avoids the
  `Ctrl+P` lib-picker collision). Confirm.
- *(O2 — tab-cycling — RESOLVED: deferred into the focus/navigation layer feature, with
  region-cycling. Not in 018.)*

## Review history

**Pre-implementation review (round 1) — 2 adversarial reviewers, both FAIL.** Findings applied:
- **CRITICAL — `imgui.shortcut()` crashes pre-`new_frame`** (reproduced `imgui.cpp:10465`). Fixed:
  decision 3 splits dispatch — pre-frame `poll/process_inputs` stays, registry dispatch + ESC move
  in-frame.
- **HIGH — `route_global` fires through modals** (doesn't reproduce `any_popup_open()` guards). Fixed:
  decision 4 keeps the explicit popup-suppression gate; routing's role narrowed to global-vs-focused
  arbitration only. The earlier "routing replaces guards for free" claim struck.
- **HIGH — `nav_enable_keyboard` blast radius / axis C framed as "free."** Resolved by SCOPE CUT: the
  whole focus/navigation layer (nav + tab/region-cycling) is deferred to its own feature (`todo.md`),
  per the user's split decision.
- **HIGH — decision 2's "keeps verb layer imgui-free / helps agent" claim false** (`app.py` already
  imports imgui). Struck; corrected to the honest leaf-split claim; stale `todo.md` clause reconciled
  in this wave.
- **`imgui.Mod_ctrl` doesn't exist** → `imgui.Key.mod_ctrl`. Fixed in decision 1.
- **MEDIUM — verification gaps** (cross-scope double-fire, rebind-without-restart, cheatsheet
  no-assert). Added as verification items 3/6/5 + decision 9 (immediate re-merge).
- **LOW — smoke invariant too weak / "optional"** + smoke popup-sum tracks 3 of 4 modals. Made
  mandatory; folded the `is_shader_lib_picker_open` fix into this wave.
- **LOW — persistence "absent vs 0" only holds until first full save.** Fixed: decision 6 persists
  only diffs-from-default.
- **COSMETIC — `command_palette_window` return-value handling.** Specified in decision 8 (returns bool
  → drives `is_palette_open`).

Zero false positives in round 1.

**Pre-implementation review (round 2) — same 2 reviewers, both PASS.** All eight round-1 fixes
verified-landed against the actual code/stubs; no new architectural/impl problem. Four prose
precision-nits applied: (1) decision 4's no-double-fire reasoning corrected ("one chord, one scope",
not gate mutual-exclusivity — EDITOR scope is currently unpopulated); (2) decision 3 pins
`dispatch_commands` at the TOP of the main-window block, before the editor child, so ESC defocus
stays same-frame; (3) decision 6 notes `save` needs custom diff-from-default logic + the harmless
unbind-to-0-of-default-0 edge; (4) the smoke "whole-frame except" wording corrected (the catch lives
in `smoke.py::main` / `ui.py::main`, not `update_and_draw`). Two cosmetic clarifications: cross-scope
duplicate chords are allowed by design (Out-of-scope); the `toggle_cheatsheet`/Settings-checkbox sync
is an explicit verification line. **Converged — ready for plan-lock.**

**Post-implementation review (round 1) — 3 reviewers: correctness PASS, architecture PARTIAL,
spec-fidelity PARTIAL. No bugs found.** Quality/bookkeeping findings applied:
- **`chord_row` now shared** by both the cheatsheet overlay and the rebinder rows (the rebinder
  hand-rolled its own row; the primitive existed precisely to avoid that). `chord_row` gained a
  `highlight` flag for the rebinder's capture state.
- **Last literal chord killed:** the empty-state "press Ctrl+N" message in `ui.py` now reads from
  `chord_to_str(effective_bindings[NEW_NODE])`.
- **Locked-decision deviation, recorded + reconciled:** decision 4 said the node-creator arrow/Enter
  nav "becomes three NODE_CREATOR-scoped commands." Implemented instead as a **bespoke fixed-key
  block** (`hotkeys.py::_handle_node_creator_nav`) — the right call (popup-internal, fixed keys, not
  rebindable / not palette-eligible). Consequently `CommandScope.NODE_CREATOR` was **removed** (it
  would have been dead); `CommandScope` is now `GLOBAL` + `EDITOR` (EDITOR kept as the documented
  editor-only extension point, currently unpopulated).
- **Rebinder rejects modifier-less chords** (`chord_needs_modifier`): a bare ordinary key (e.g. `n`)
  would fire its command while typing in the editor. Function keys (F1-F12) are exempt (safe
  standalone). Warns + refuses.
- Cosmetic: palette call passes `app.is_palette_open` (not literal `True`); decision-number
  cross-refs trimmed from code comments. Zero false positives across the 3 reviewers.

**Maintainer review (rounds 1-2):**
- **Cheatsheet overlay wasn't visible** — first moved out of the main-window `begin` block (still
  obscured by the full-screen main window), then rewritten to draw on the **foreground draw list**
  (absolute screen coords, immune to window z-order). Now **top-right**, faint (`CHEATSHEET_ALPHA`),
  with a live `press <toggle-chord> to hide` footer.
- **`Ctrl+/` toggles the cheatsheet** — `TOGGLE_CHEATSHEET` was unbound-by-default (so it was listed
  but unusable); given a real default chord. Display labels cleaned (`Slash`→`/`, `LeftArrow`→`Left`).
- **Unified every keybindings surface on the one registry.** Cheatsheet, command palette, Settings
  rebinder, and menu hints all iterate `COMMAND_SPECS` + read `effective_bindings`, all show chords.
  The rebinder lists ALL commands (fixed-key arrows show a disabled Rebind button). The palette entry
  names carry the chord (label-padded so chords align) and are re-registered on rebind to stay live.
  **Accepted divergence:** the palette reorders by relevance/recency (inherent widget behavior, no
  order-control API) while the cheatsheet keeps fixed registry order — same commands+chords, different
  sort, by design.
- **Removed the node-grid `"?"` hover tooltip** (`widgets/node_grid.py`) — stale hardcoded list; the
  live cheatsheet replaces it.
- **Renamed the top-bar menu "Shader Library" → "Library"** (read as two options next to "Edit").
