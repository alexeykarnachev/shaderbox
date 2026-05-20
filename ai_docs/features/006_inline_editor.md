# Feature 006 — Inline GLSL code editor

## Goal

Replace the external-command-only "Edit code" workflow with an **in-app GLSL editor**:
syntax-highlighted, gruvbox-themed, editing the current node's `shader.frag.glsl` directly
inside the shaderbox window. The external editor stays available as a "Pop out" fallback
(for when the user wants full vim — see research note below).

After this lands: select a node → a **Code tab** shows its shader source with GLSL syntax
highlighting; edit + `Ctrl+S` saves to disk and hot-reloads the shader; "Pop out" still
launches the configured external editor.

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
  Zep binding is a C++ project. The "Pop out" external editor covers the vim use case.
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
- **The reverted prototype layout** (left-half editor pane) — the editor goes in a **Code
  tab**, not a left pane. The prototype's side-by-side layout is shelved.

## Design decisions (locked)

### 1. Editor lives in a new "Code" tab in the node-settings tab bar
The current layout (post feature-005-revert) is image-top / control-panel-below, with the
node-settings tab bar (Node / Render / Share) on the right of the control panel. Add a
**Code** tab as the FIRST tab (leftmost, default-selected when a node is active) →
`Code / Node / Render / Share`. This fits the editor into the existing layout with zero
structural change — no new panes, no window-geometry change.

Rationale for a tab vs a pane: the user just reverted the wide-screen split layout; adding a
left pane would re-introduce exactly what they rejected. A tab is the minimal-blast-radius
home for the editor. Revisit if the editor-in-a-tab feels too cramped (it shares the control-
panel height ~600px; the tab body gets the full tab area).

### 2. One `TextEditor` instance per node, lazily allocated, owned by `App`
`imgui_color_text_edit.TextEditor` holds editing state across frames (cursor, undo stack,
selection) — it cannot be recreated each frame. Store a `dict[str, TextEditor]` on `App`
(`app.editors`), keyed by node id. Lazily create on first access for a node; configure once
on creation: `set_language(TextEditor.Language.glsl())` + apply the gruvbox palette (Decision
4). Set initial text from the node's current `fs_source`.

`App` owns it (per the three-layer rule: state lives on `App`, not in the tab-draw function).
Add `app.editors: dict[str, TextEditor]` initialized in `App.__init__`. Clean up an entry
when its node is deleted (`delete_current_node` pops `app.editors.pop(node_id, None)`).

### 3. Save model: editor buffer → disk on Ctrl+S → existing reload path
- `Ctrl+S` (the existing global save hotkey) gains: if the Code tab's editor for the current
  node is dirty, write `editor.get_text()` to `nodes/<id>/shader.frag.glsl` BEFORE the
  existing `app.save()` runs. Then the existing mtime-watcher in `ui.py::update_and_draw`
  sees the changed file and calls `release_program(new_source)` → hot reload. (We update
  `ui_node.mtime` after our write so the watcher doesn't double-reload, OR let it reload once
  — either is correct; simpler to let the watcher handle the reload uniformly.)
- **Decision**: on save, write the file + update `ui_node.mtime` to the new mtime + call
  `release_program(text)` directly (don't rely on the watcher for the inline-save path — the
  watcher remains the path for EXTERNAL edits). This makes inline-save reload deterministic
  and avoids a one-frame stale render.

### 4. mtime-watcher ↔ editor buffer sync (the conflict case)
The existing watcher reloads a node when its file mtime changes on disk. With an inline
editor holding an in-memory buffer, an external edit (pop-out nvim, or any external change)
must ALSO refresh the editor buffer, else the inline buffer goes stale and the next inline
save clobbers the external edit.

**Rule**: when the watcher detects an external mtime change for a node that has an
`app.editors[id]` instance, re-`set_text()` that editor from disk (disk is the source of
truth). The watcher already reads `fs_file_path.read_text()` for the reload — reuse it to
sync the editor. Caveat: this discards unsaved inline edits if the file changes externally
mid-edit — acceptable (disk-wins is the simplest correct rule; the user explicitly chose to
edit externally).

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

### 6. "Pop out" = existing external editor, relabeled
The current `Edit code` button (calls `app.edit_current_node_fs_file()` → spawns the
configured external editor) moves into the Code tab as a **"Pop out ↗"** button. Behavior
unchanged. The `Ctrl+E` hotkey keeps working (still calls `edit_current_node_fs_file`).
The Node tab's existing "Edit code" button: keep it (harmless, same action) OR remove it to
avoid duplication. **Decision**: remove the Node-tab "Edit code" button — the Code tab is now
the home for code editing; leaving a duplicate in the Node tab is clutter. `Ctrl+E` stays.

### 7. Dirty indicator
The editor tracks its own undo state; compare `editor.get_text()` against the last-saved
source to show a dirty dot on the Code tab label (`Code ●`). Cheap: store last-saved text per
node, or use `editor.get_undo_index()` vs a saved-index marker. **Decision**: track a
`saved_undo_index` per editor; dirty = `editor.get_undo_index() != saved_undo_index`. Update
the marker on every save. Simpler than text comparison each frame.

## Files touched

**New deps**: none (`imgui_color_text_edit` is already in imgui-bundle).

**Modified**:
- `shaderbox/app.py` — add `self.editors: dict[str, TextEditor] = {}` in `__init__`; add a
  helper `get_editor(node_id) -> TextEditor` (lazy create + configure language/palette/text);
  `delete_current_node` pops the editor; the save path gains the inline-save write+reload.
- `shaderbox/ui.py` — the mtime watcher re-syncs `app.editors[name]` on external change; the
  `_draw_node_settings` tab bar gains a "Code" tab (first position).
- `shaderbox/tabs/code.py` — **NEW** tab module (per the `tabs/*.py` free-`draw(app)`
  convention): renders the editor for the current node + a "Pop out ↗" button + a small
  toolbar (file path label, dirty dot). Mirrors `tabs/node.py` shape.
- `shaderbox/hotkeys.py` — `Ctrl+S` save path: if current node's editor is dirty, flush it to
  disk first. (Or route through `app.save()` — decide at impl; keep hotkeys.py thin, put the
  logic in an `App` method like `app.save_current_editor()`.)
- `shaderbox/theme.py` — add a `make_editor_palette() -> TextEditor.Palette` helper (or keep
  it in `tabs/code.py` / a new `editor_palette.py` if it pulls in the text-edit import — avoid
  importing `imgui_color_text_edit` into `theme.py` if it muddies the theme module; decide at
  impl). Maps `COLOR.SYN_*` → `TextEditor.Color`.
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
3. **Boot + Code tab**: launch app, select a node → Code tab present + default-selected →
   shows the shader source with GLSL syntax highlighting (keywords/types/comments colored).
4. **Edit + save**: type a change (e.g. tweak a uniform default), `Ctrl+S` → file saved on
   disk + render updates (hot reload). Dirty dot clears on save.
5. **Compile error path**: introduce a deliberate GLSL typo, save → red error overlay appears
   on the render image (existing behavior, unchanged).
6. **Pop out**: click "Pop out ↗" → external editor launches on the same file. Edit + save
   externally → inline editor buffer re-syncs from disk (Decision 4), render reloads.
7. **Node switch**: switch to a different node → editor shows that node's source. Switch back
   → original node's source + edit state intact (per-node instances).
8. **Delete node**: delete a node → its editor instance is cleaned up (no leak; no crash on
   re-create if a new node reuses... node ids are uuids, so no reuse).
9. **Palette**: confirm the highlighting uses gruvbox colors (or, if Decision-5 fallback, the
   built-in dark palette — note which in the worklog).
10. Hotkeys still fire (Ctrl+S/E/N/D/O, Alt+S, arrows, Esc); other tabs (Node/Render/Share)
    unchanged.

## Open questions (LOCKED at plan time)

1. **Editor in a tab vs a pane** — LOCKED: Code tab (Decision 1). Minimal blast radius; no
   layout change.
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

**Post-impl review**: to be filled.
