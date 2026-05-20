# Feature 006 — Inline GLSL code editor

> **FINAL STATE (supersedes the plan below).** What actually shipped, after the user reworked
> it mid-implementation: the editor is the **always-visible LEFT half** of the main window (app
> on the right, draggable splitter between) — NOT a tab. The external editor / "Pop out" /
> `Ctrl+E` / `text_editor_cmd` were **removed entirely**. An "Options" toolbar button opens a
> popup for editor visuals (whitespace/line-numbers/brackets/font/tab/spacing), applied on
> close; Ctrl+scroll changes font size. Palette is the built-in dark (gruvbox impossible —
> read-only Palette). The sections below are the original plan; where they say "Code tab" or
> "Pop out", read Decision 1 (REVISED) and Decision 6 (REVISED) for the real design.

## Goal

Replace the external-command-only "Edit code" workflow with an **in-app GLSL editor**:
syntax-highlighted, editing the current node's `shader.frag.glsl` directly inside the shaderbox
window. (Original plan kept an external "Pop out" fallback for vim — REMOVED in the final state;
see the banner above + Decision 6.)

After this lands: select a node → the editor (LEFT half of the window, see Decision 1 REVISED)
shows its shader source with GLSL syntax highlighting; edit + `Ctrl+S` saves to disk and
hot-reloads the shader.

## Research note (why this shape)

Spiked `imgui_color_text_edit` (bundled in imgui-bundle 1.92.801) — works, has a built-in
`TextEditor.Language.glsl()` syntax definition, custom palette support, cursor positioning.
**No vim/modal keybindings** (fixed standard keymap; not configurable from Python). The only
real vim-capable embeddable editor (Zep) has no Python binding and would require a C++
pybind11 subproject — disproportionate. Decision (locked with user): inline editor with the
standard keymap for fast edits + keep external nvim as "Pop out" for full vim. See the
2026-05-20 worklog research entry.

## Out of scope

- **Vim/modal keybindings inside the editor** — not available in `imgui_color_text_edit`;
  Zep binding is a C++ project. (Original plan covered vim via the "Pop out" external editor;
  that was REMOVED in the final state — vim is simply unsupported now, re-introducible as a
  clean opt-in later if wanted.)
- **Multi-file editing** (vertex shader, `node.json`) — a node has one `.frag.glsl`; single-
  file only. Trigger to revisit: if a vertex-shader-per-node feature ever lands.
- **Inline error annotations / click-to-jump** — the editor API supports `set_cursor(DocPos)`
  so this is feasible later, but it's a separate surface. Stays as the existing red
  `add_text` overlay on the render image for now. Tracked: `todo.md [DEFERRAL] inline
  shader-error display`.
- **Live-reload-on-keystroke** — reload happens on explicit save (`Ctrl+S`), not on every
  keystroke. (Keystroke reload would recompile the shader constantly mid-edit → flicker +
  spurious compile errors while typing.) Trigger to revisit: if the save-to-see-changes loop
  feels slow in practice.
- ~~**The reverted prototype layout** (left-half editor pane) — the editor goes in a **Code
  tab**, not a left pane.~~ **SUPERSEDED by Decision 1 (REVISED):** the user reverted this — the
  editor IS the left half now. (Kept here struck-through because it's the exact thing that flipped.)

## Design decisions (locked)

### 1. Editor occupies the LEFT half of the window; the existing app moves RIGHT (REVISED 2026-05-20)
**Superseded the original "Code tab in the node-settings tab bar" decision** after the user
saw the cramped tab-bar version running and rejected it: "the code should occupy the left
half of the screen. this must be a separate large tab." Layout is now a vertical split of the
main window:
- **Left column** = the GLSL editor, full remaining height, always visible when a node is
  selected. Empty-state message when no node.
- **Right column** = the existing layout unchanged (render image on top, control panel below
  with node-grid-left + Node/Render/Share tab bar right).
- A **draggable splitter** between them (imgui `invisible_button` + `is_item_active` + mouse
  delta, `MouseCursor_.resize_ew` on hover). Starts ~50/50. The split fraction is state →
  lives on `App` (`app.editor_split_fraction: float`, clamped to a sane range, persisted in
  `UIAppState` so it survives restarts).

This DELIBERATELY revives a left-editor split — a layout shape the user rejected twice before
(feature 005 + `feature/ui-redesign-realize`, per the worklog). The difference the user is
asserting now: a *code editor* on the left is wanted, even though the earlier *prototype*
left-pane (with its different right-pane redesign) was not. Recorded as a conscious reversal,
not a regression. Revisit only if the user reverts again.

The Code tab in the node-settings tab bar is REMOVED (`Node / Render / Share` restored there).

### 2. One `TextEditor` instance per node, lazily allocated, owned by `App`
`imgui_color_text_edit.TextEditor` holds editing state across frames (cursor, undo stack,
selection) — it cannot be recreated each frame. Store a `dict[str, TextEditor]` on `App`
(`app.editors`), keyed by node id. Lazily create on first access for a node; configure once
on creation: `set_language(TextEditor.Language.glsl())` + `set_palette(get_dark_palette())`
(Decision 5 — v1 ships the built-in dark palette, no custom gruvbox palette). Set initial text
from the node's current `fs_source`, then capture the dirty baseline
(`app.editor_saved_undo[node_id] = editor.get_undo_index()`).

`App` owns it (per the three-layer rule: state lives on `App`, not in the tab-draw function).
Add `app.editors: dict[str, TextEditor]` AND `app.editor_saved_undo: dict[str, int]`
(the dirty baseline, Decision 7) in `App.__init__`. Add `get_editor(node_id) -> TextEditor`
(lazy create + configure). Clean up BOTH dicts when a node is deleted: `delete_current_node`
pops `app.editors.pop(node_id, None)` and `app.editor_saved_undo.pop(node_id, None)`
(pre-impl review finding 4).

### 3. Save model: editor buffer → disk on Ctrl+S → existing reload path
- `Ctrl+S` (the existing global save hotkey) gains: if the Code tab's editor for the current
  node is dirty, flush `editor.get_text()` BEFORE the existing `app.save()` runs.
- **CRITICAL ORDERING (locked, pre-impl review finding 1)**: `app.save()` →
  `save_ui_node` → `UINode.save` writes `self.node.fs_source` to
  `nodes/<id>/shader.frag.glsl` (`ui_models.py:214`). So the editor flush MUST update
  `node.fs_source` *before* `app.save()` runs — otherwise `node.save()` re-writes the file
  from the STALE `fs_source` and silently clobbers the inline edit. The flush calls
  `node.release_program(editor_text)` (which sets `fs_source = text`, `core.py:162`, and
  recompiles), so the subsequent `app.save()` writes the SAME text. This is NOT "decide at
  impl" — the order is mandated.
- **Decision**: add `App.flush_current_editor()` — if the current node has a dirty editor,
  call `node.release_program(editor.get_text())`, then re-capture the dirty baseline (Decision
  7). `app.save()` calls `flush_current_editor()` as its FIRST line, so both the Ctrl+S hotkey
  path and any other `app.save()` caller flush deterministically. After `release_program`,
  `UINode.save` writes the file from the now-synced `fs_source` and sets `ui_node.mtime` to the
  new mtime — so the watcher (ui.py:54-65) sees no change and does NOT double-reload. No
  separate watcher-driven reload for the inline path; the watcher remains the path for EXTERNAL
  edits only.

### 4. mtime-watcher ↔ editor buffer sync (the conflict case)
The existing watcher reloads a node when its file mtime changes on disk. With an inline
editor holding an in-memory buffer, an external edit (pop-out nvim, or any external change)
must ALSO refresh the editor buffer, else the inline buffer goes stale and the next inline
save clobbers the external edit.

**Rule**: when the watcher detects an external mtime change for a node that has an
`app.editors[id]` instance (guard: `if name in app.editors`), re-`set_text()` that editor from
disk (disk is the source of truth). The watcher already reads `fs_file_path.read_text()` for
the reload — reuse it to sync the editor. **After `set_text`, re-capture the dirty baseline**
(`app.editor_saved_undo[name] = editor.get_undo_index()`) — `set_text` does NOT advance the
undo index (spike-confirmed: stays at its current value), so without re-baselining the dirty
dot would read stale (Decision 7, pre-impl review finding 2). Caveat: this discards unsaved
inline edits if the file changes externally mid-edit — acceptable (disk-wins is the simplest
correct rule; the user explicitly chose to edit externally).

### 5. Gruvbox palette wired to `COLOR.SYN_*` tokens
`theme.py` already exports `COLOR.SYN_KEYWORD / SYN_TYPE / SYN_BUILTIN / SYN_NUMBER /
SYN_STRING / SYN_COMMENT / SYN_PREPROC / SYN_UNIFORM / SYN_IDENT / SYN_OP`. The editor's
`TextEditor.Color` enum has 22 slots. Build a `TextEditor.Palette` mapping the gruvbox tokens
onto the relevant `Color` slots (text/keyword/number/string/comment/preprocessor/identifier/
known_identifier/punctuation + background/cursor/selection from the neutral tokens). Apply via
`editor.set_palette(palette)` once at editor creation.

`TextEditor.Color` → `COLOR.SYN_*` mapping (the load-bearing ones):
- `keyword` → `SYN_KEYWORD`, `declaration` → `SYN_TYPE`, `number` → `SYN_NUMBER`,
  `string` → `SYN_STRING`, `punctuation` → `SYN_OP`, `preprocessor` → `SYN_PREPROC`,
  `identifier` → `SYN_IDENT`, `known_identifier` → `SYN_BUILTIN`, `comment` → `SYN_COMMENT`,
  `text` → `FG_PRIMARY`, `background` → `BG_APP`, `cursor` → `ACCENT_PRIMARY`,
  `selection` → `ACCENT_ALPHA` (or a faded accent), `line_number` → `FG_DIM`,
  `current_line_number` → `FG_SECONDARY`.

**RESOLVED (pre-impl spike, 2026-05-20): the custom gruvbox palette CANNOT ship in v1.**
`TextEditor.Palette` exposes only `.get(color) -> ImU32` from Python — no per-slot setter, no
indexed write, and `set_palette()` rejects a plain `list[int]` (only accepts a `Palette`
object, which is unbuildable with custom colors from Python). So v1 uses the built-in
`TextEditor.get_dark_palette()` as-is (a good dark palette, just not gruvbox-token-matched).
The `COLOR.SYN_*` tokens stay exported for when a palette-write path exists. **Deferral filed**
in `todo.md`: gruvbox editor palette match — trigger: when imgui-bundle exposes a writable
Palette (per-slot setter or list constructor), or if the dark-palette mismatch visibly annoys.
The `Color`→`SYN_*` mapping table above documents the intended mapping for that future work.

### 6. External editor — REMOVED (REVISED 2026-05-20 post-ship)
Originally: relabel the `Edit code` button to "Pop out ↗", keep `Ctrl+E` + the configured
external editor as a vim fallback. After shipping, the user rejected keeping it ("cleanup this
legacy shit") — the inline editor fully replaces the external one. **REMOVED entirely**:
`text_editor_cmd` (model field + Settings popup field + tooltip), `App.edit_current_node_fs_file`,
the `Ctrl+E` hotkey, and the "Pop out" button. `load_and_migrate` gen-4 drops the stale
`text_editor_cmd` key. The Node-tab "Edit code" button was already removed. "Open dir" remains
for manual file access. (If a vim user ever needs an external editor again, re-introduce it as a
clean opt-in, not the always-present default.)

### 7. Dirty indicator
The editor tracks its own undo state; show a dirty dot on the Code tab label (`Code ●`).
**Decision**: track `app.editor_saved_undo: dict[str, int]` (the baseline undo index per node,
NOT a per-`TextEditor` attribute — `TextEditor` has no user slot, pre-impl review finding 2);
dirty = `editor.get_undo_index() != app.editor_saved_undo[node_id]`. Re-capture the baseline
(`app.editor_saved_undo[node_id] = editor.get_undo_index()`) at THREE moments: (a) on lazy
editor creation (Decision 2), (b) on inline save / flush (`flush_current_editor`, Decision 3),
(c) on external-sync `set_text` (Decision 4). `set_text` does not advance the undo index, so
(a) and (c) are required, not optional. Simpler than text comparison each frame.

## Files touched

**New deps**: none (`imgui_color_text_edit` is already in imgui-bundle).

**Modified**:
- `shaderbox/app.py` — add `self.editors: dict[str, TextEditor] = {}` and
  `self.editor_saved_undo: dict[str, int] = {}` in `__init__`; add `get_editor(node_id) ->
  TextEditor` (lazy create + configure language + `set_palette(get_dark_palette())` + initial
  text + baseline capture); add `flush_current_editor()` (Decision 3 — `release_program` from
  editor text + re-baseline, no-op when no node / no editor / not dirty); `save()` calls
  `flush_current_editor()` as its first line; `delete_current_node` pops BOTH `editors` and
  `editor_saved_undo`.
- `shaderbox/ui.py` — the mtime watcher re-syncs `app.editors[name]` on external change
  (guard `if name in app.editors`, re-baseline after `set_text`). Main window split into a
  LEFT editor child + draggable splitter + RIGHT app child (Decision 1 REVISED): new helpers
  `_draw_splitter(app, total_width, height)` (invisible_button + `is_item_active` mouse delta,
  clamps `app_state.editor_split_fraction` to 0.15..0.85, `resize_ew` cursor) and
  `_draw_app_panel(app)` (the former main-window body: image + control panel). `code_tab.draw`
  is called in the left child. `_draw_node_settings` reverted to `Node / Render / Share`.
- `shaderbox/tabs/code.py` — **NEW** tab module (per the `tabs/*.py` free-`draw(app)`
  convention): shows "No node selected" when `current_node_id` is empty (the left column is
  always visible now); else a toolbar (file-path label, "(unsaved)" marker, "Open dir" /
  "Options" buttons) + the editor filling the rest via `app.get_editor(id).render(...)`.
  (No "Pop out" — external editor removed, Decision 6 REVISED.)
- `shaderbox/ui_models.py` — `UIAppState.editor_split_fraction: float = 0.5` (persists the
  splitter position). Plus a latent-bug fix surfaced by the inline-save path: `UINode.save`
  captured `self.mtime` INSIDE the `with open()` block (before close-flush bumped the real
  mtime), so the watcher saw a stale mtime and double-reloaded one frame after every save.
  Moved the `lstat` after the `with` block.
- `shaderbox/hotkeys.py` — UNCHANGED. `Ctrl+S` already calls `app.save()`, which now flushes
  the editor first (Decision 3) — the flush logic lives on `App`, not in hotkeys.py (keeps
  hotkeys.py thin).
- `shaderbox/theme.py` — UNCHANGED in v1. Decision 5 is RESOLVED: no custom `Palette` is
  buildable from Python in imgui-bundle 1.92.801, so v1 uses `get_dark_palette()` and writes NO
  `make_editor_palette` helper (it would be dead code). The `COLOR.SYN_*` tokens stay exported;
  the `Color`→`SYN_*` mapping lives in the `todo.md` deferral for when a palette-write path
  exists (pre-impl review finding 3).
- `shaderbox/tabs/node.py` — remove the now-duplicate "Edit code" button (Decision 6).

**Docs**:
- `ai_docs/conventions.md ## Design decisions` — new bullet: editor state (`app.editors`)
  lives on `App`, one `TextEditor` per node, disk-wins on external change. Revisit if multi-
  file editing lands.
- `ai_docs/conventions.md ## Known quirks` — `imgui_color_text_edit` notes if any footgun
  surfaces (e.g. the Palette-mutation question).
- `ai_docs/todo.md` — if the gruvbox-palette match is deferred (Decision 5 fallback), file it.
- `ai_docs/worklog.md` — entry.

## Manual verification

### Pre-impl
0a. `make smoke 2>&1 | tee /tmp/smoke_baseline_pre_006.log` on master. Exit 0.
0b. Palette-mutation question — **already resolved** during plan-lock spike: no Python write
    path; v1 uses `get_dark_palette()`. (No further pre-impl spike needed.)

### Post-impl
1. `make check` clean.
2. `make smoke` clean; diff vs baseline = timestamps only. (Note: smoke doesn't open the Code
   tab or render the editor — it exercises import + boot. The editor's `render()` needs manual
   verification.)
3. **Boot + split layout**: launch app, select a node → editor occupies the LEFT half showing
   the shader source with GLSL syntax highlighting (keywords/types/comments colored); the app
   (render image + control panel) is on the RIGHT half; a draggable splitter sits between them.
   [VERIFIED 2026-05-20 via screenshot /tmp/sb_split.png]
3b. **Splitter drag**: drag the divider → left/right widths rebalance; clamped so neither side
   collapses; position persists across restart (`editor_split_fraction` in app_state.json).
4. **Edit + save (disk-readback, the clobber check)**: type a change, `Ctrl+S` → dirty dot
   clears + render updates (hot reload). **THEN read `nodes/<id>/shader.frag.glsl` from disk
   and confirm it contains the edit** — the render + dot can both pass off the in-memory
   `release_program` even if `app.save` clobbered the file (pre-impl review finding 1, the
   flagged likeliest bug). Bonus: relaunch the app, confirm the edit persisted.
4b. **Dirty marker appears**: confirm the `(unsaved)` marker in the editor toolbar APPEARS on
   the first edit (not just clears on save), and is ABSENT on a freshly-switched, never-edited
   node (no false positive).
5. **Compile error path (incl. the GL-crash fix)**: introduce a deliberate GLSL typo, `Ctrl+S`
   → red error overlay appears on the render image (existing behavior). CRUCIAL: this must NOT
   crash with `GLError 1281` — `release_program` now binds `glUseProgram(0)` so the freed
   program isn't left GL-current when the recompile fails. (This is the crash the user actually
   hit; the original GLError-1281 repro is interactive-only.)
6. **External-edit re-sync (incl. the data-loss caveat)**: change `shader.frag.glsl` on disk
   externally (e.g. `Open dir` then edit in another editor) → inline editor buffer re-syncs from
   disk (Decision 4), render reloads. Then the disk-wins caveat: make an UNSAVED inline edit,
   change the file externally → the inline buffer is replaced from disk, unsaved edit gone
   (intended). (No "Pop out" button — external editor removed, Decision 6 REVISED.)
7. **Node switch — edit-state survival (the reason instances exist)**: edit + move the cursor
   + do a partial undo on node A, switch to B, switch back to A → A's source, cursor position,
   and undo stack are all intact (a single shared editor would fail this; per-node instances
   pass it).
8. **Delete node mid-edit**: with the editor visible, delete the current node → no crash on the
   next frame's editor draw; editor shows the next node (or the empty state if it was the last).
   Its editor + baseline entries are popped (no leak).
8b. **Empty-node editor**: with NO node selected (delete all, or fresh project), the editor pane
   shows "No node selected" and `Ctrl+S` doesn't crash (draw + flush no-op on
   `current_node_id == ""`).
8c. **Options popup (FPE guards)**: open "Options", change settings, Close → no crash, settings
   applied; while open, the code shows as a dimmed plain-text snapshot (not the live editor);
   repeat closing via Esc → same. (Both are the FPE workarounds — see `todo.md` DEFERRAL.)
8d. **Ctrl+scroll font size**: hover the editor, Ctrl+scroll → font grows/shrinks (clamped
   8–48); plain scroll still scrolls the editor.
9. **Palette**: confirm the highlighting uses the built-in dark palette (Decision-5 fallback —
   gruvbox match is deferred).
10. Hotkeys still fire (Ctrl+S/N/D/O, Alt+S, arrows, Esc — NOTE: Ctrl+E removed with the
    external editor); other tabs (Node/Render/Share) unchanged. Confirm the Node tab no longer
    shows an "Edit code" button (Decision 6 removal).

## Open questions (LOCKED at plan time)

1. **Editor in a tab vs a pane** — was LOCKED to a Code tab at plan time; **SUPERSEDED by
   Decision 1 (REVISED)** — shipped as the left-half split pane after the user rejected the tab.
2. **Reload on save vs on keystroke** — LOCKED: on save (out-of-scope note).
3. **Palette match** — LOCKED: built-in `get_dark_palette()` for v1 (gruvbox match is
   impossible — no Python palette-write path; spike-confirmed). Gruvbox match deferred to
   `todo.md` pending an imgui-bundle API that allows per-slot palette writes.
4. **Node-tab "Edit code" button** — LOCKED: removed (Code tab owns code editing).

## Review history

Plan-locked 2026-05-20 after spiking `imgui_color_text_edit` (API confirmed: GLSL language,
text round-trip, cursor, palette accessors all work) + web research on vim options (Zep has
no Python binding; standard keymap is the only Python-reachable path). One un-derisked API
detail (Palette mutation from Python) gated behind pre-impl step 0b with a defined fallback.

**Pre-impl review (2026-05-20, 2 parallel agents — correctness/design + verification/blast-
radius)**: both returned PARTIAL, converging on the same findings. All folded into the spec
above (none escalated to "shouldn't land"):
1. **Ctrl+S clobber order** — `app.save()`→`node.save()` writes `node.fs_source` to the same
   file; the editor flush MUST sync `fs_source` (via `release_program`) first. Resolved:
   mandated ordering via `App.flush_current_editor()` called as `app.save()`'s first line
   (Decision 3 rewritten; hotkeys.py now UNCHANGED).
2. **`saved_undo_index` had no home + no re-baseline rule** — `TextEditor` has no user slot;
   `set_text` doesn't advance the undo index. Resolved: `app.editor_saved_undo: dict[str, int]`
   re-captured on create / flush / external-sync (Decisions 2, 3, 7).
3. **Dead-code palette helper** — Decision 5 resolved to `get_dark_palette()`, so
   `make_editor_palette` would be unwritable dead code. Resolved: theme.py UNCHANGED in v1.
4. **`delete_current_node` cleanup** — must pop both `editors` and `editor_saved_undo`.
5. **Empty-node guard** — Code-tab draw + flush must no-op on `current_node_id == ""`.
Verification gaps folded into Manual verification: disk-readback after save (the clobber
check), cursor/undo survival across node switch, delete-while-Code-tab-open, empty-node Code
tab, disk-wins data-loss caveat, dirty-dot appears-on-edit, Node-tab button removal.

**Post-impl review (2026-05-20, 2 parallel agents — code-correctness + architecture/
conventions)**: correctness PASS (all 7 decisions + 5 pre-impl findings honored; clobber
order verified); architecture PARTIAL with one HIGH finding:
- **Tab-identity footgun**: a varying tab label (`"Code"` ↔ `"Code ●"`) changes the imgui
  tab ID (imgui hashes the label up to `###`), dropping the Code tab's selection on the
  frame the dirty state flips. Fixed with the stable-id idiom: `f"Code{dirty}###code_tab"`
  (visible part varies, hashed id stays `code_tab`). ui.py.
Duplication of the file-path label + "Open dir" button between `tabs/node.py` and
`tabs/code.py` flagged LOW / acceptable (~6 lines, already diverging — node.py uses a
clipboard-copy selectable, code.py a dim label) — no extraction.

**Layout revision + post-revision verification (2026-05-20)**: after seeing the tab-bar
version running, the user rejected it and asked for the editor on the LEFT half as a separate
large pane with a draggable splitter (Decision 1 REVISED). The `App`-level machinery (editors
dict, flush/dirty/sync, save ordering) was layout-independent and kept verbatim; only `ui.py`'s
draw structure changed. Headless clobber+lifecycle test (temp project copy, full edit→save→
disk-readback cycle) confirmed all 9 invariants: clobber-safe save, dirty cleared, fs_source
synced, mtime matches disk (no double-reload), external re-sync + re-baseline, delete pops both
dicts. That test surfaced the `UINode.save` stale-mtime double-reload bug (fixed). Split layout
verified via screenshot.
