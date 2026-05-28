# 016 — Lib file management UI

In-app **delete / rename / move / reveal** for shader-library files (`<app_data_dir>/lib/**.glsl`).
Today the picker (`popups/lib_picker.py`) can only CREATE files (`+ New library file`); managing
existing files means dropping into the file system. This feature closes that gap.

Builds on 015's `LibIndex` (name-keyed lookup) + the mtime watcher in `ui.py::_maybe_rebuild_lib_index`
(already polls `lib_root` per frame and invalidates dependent nodes on change). The deletion / rename
paths produce on-disk mutations — the watcher handles the index rebuild; we add filtering for the
new `.trash/` subdir and explicit session/state plumbing.

## Goal

1. **Delete** a lib file from the picker with a 2-click armed-confirm (mirrors `node_delete_armed`).
   File goes to `<lib_root>/.trash/<basename>` (recoverable on misclick) — never `unlink`.
   Mtime watcher rebuilds the index; dependents pick up the missing function as a normal
   "undeclared identifier" driver error on next compile.
2. **Rename** a lib file with collision check (new name can't exist as a file). If an
   `EditorSession` is open on the file: update `source.path` and re-key `editor_sessions`; if
   `_explicit_editor_path` / `last_lib_path` / `editor_jump_request.path` point at the old path,
   update them too.
3. **Move into / out of subdir** is the same code path as rename (`Path.rename` across dirs). The
   resolver doesn't care about file structure — lookups are by function name.
4. **Reveal in file manager** as an always-available escape hatch. Uses the same `xdg-open` /
   `explorer` / `open` recipe as `App.open_current_node_dir` — both call a new shared helper
   `util.open_in_file_manager(path)` (the helper opens the **parent dir** of `path`, since
   `xdg-open <file>` opens the file in its default app, not the manager).
5. **No new popup.** All four affordances live INSIDE the existing `Ctrl+P` lib picker — keeps the
   feature self-contained and re-uses the picker's already-open keyboard context.

## Non-goals — this feature does NOT

- **Multi-select / bulk delete.** One file at a time.
- **Restore-from-trash UI.** The `.trash/` dir is a "save you from a misclick" escape hatch;
  recovery is manual (open the dir in the file manager, drag the file back). A restore UI is its
  own feature; build when triggered.
- **Trash garbage collection.** Files stay in `.trash/` until the user deletes them by hand. The
  trash dir grows unbounded — acceptable for a personal tool with small files (<10KB typical).
  Revisit if it becomes a real footgun.
- **Drag-to-rename / inline-rename in the function-list row.** Rename is a dedicated input via the
  per-file Manage-files row; no in-place editing of the function rows.
- **Reveal a specific function** (jumping the file manager to a function definition inside the
  file). File-level reveal only; the picker already supports "Open file" to jump to the function
  in the editor.
- **Sibling deferrals checked, none triggered:** `cross-file uniform declaration jump (lib files)`
  (touches `find_uniform_declaration_line` / `JumpRequest` — not in this work),
  `export-from-selection`, `multi-file editor — tab bar`. All left alone.

## Design decisions (locked)

> **Note (post-impl rewrite):** D1, D2, D12 describe the SHIPPED shape. The original spec locked
> a collapsing "Manage files" section with a tree+detail split + a Rename/Delete/Reveal button row
> on the detail pane. That was rejected mid-implementation as visually noisy ("looks like slop");
> the picker was unified into a single tree+preview view with right-click context menus for all
> file/dir/function actions. The story lives in git history; below reflects what landed.

1. **UI shape: unified tree+preview is the picker itself.** Left column = a `/` root
   `tree_node_ex` containing nested subdirs (collapsible), files (collapsible, with their function
   leaves as children), and a "Right-click for actions" hint at the top. Right column = the
   preview pane for the currently-selected function (clickable copyable path, signature, doc,
   tags editor, body). Above both panes: search input (row 1), Favs+Reset pills (row 2), tag bar
   (row 3). Below: Insert at caret + Close. No separate "Manage files" section, no per-row
   action buttons.

2. **Right-click context menus per node kind** (`context_menu_style()` in `ui_primitives.py`
   gives them a distinct fill + accent border + accent hover so they pop visually off the
   underlying modal):
   - **Directory** menu: *New file here* / *New subdirectory* / *Reveal in file manager* /
     (separator + red) *Delete directory (recursive)*. Root `/` has no Delete.
   - **File** menu: *Rename* / *Reveal in file manager* / (separator + red) *Delete*.
   - **Function leaf** menu: *Insert at caret* (closes picker; disabled when no editor target
     exists) / *Open file at declaration* (closes picker) / *Copy name* / *Favorite / Unfavorite*.
   - Delete actions are **armed-confirm**: the menu item flips to "Confirm delete" on the next
     open; the file or dir row tints red in the tree while armed. Clicking a function leaf
     (a clear "I'm not deleting" signal) disarms.
   - Inline inputs (Rename / New file / New subdirectory) replace the row/draw under it with a
     borderless `input_text(enter_returns_true)` + a small `x` cancel button. Enter commits,
     Esc cancels (scoped to the focused input). Mutual exclusion: every `begin_*` on `App` calls
     `reset_lib_inline_state()` so only one inline input is ever live.

3. **Trash dir: `<lib_root>/.trash/`, leading-dot hidden from BOTH the `LibIndex.build` glob AND
   the mtime watcher's glob.** Python's `Path.glob("**/*.glsl")` walks into dot-directories by
   default. The picker's index already correctly de-glob's `.trash/` once we add the filter — but
   the mtime watcher in `ui.py::_maybe_rebuild_lib_index` does its OWN independent glob walk for
   change detection. **Both globs MUST apply the same filter** or `current != cached` will fire
   every frame on every trashed file → infinite rebuild loop. Centralize the filter as
   `lib_index.is_lib_path(path: Path, lib_root: Path) -> bool` and call it from both sites.

4. **Trash filename: `<basename>` (NO timestamp prefix). On collision: bump a numeric suffix.**
   First delete of `noise.glsl` lands at `.trash/noise.glsl`. A second delete of any file whose
   basename is `noise.glsl` (across subdirs) lands at `.trash/noise_1.glsl`, then
   `noise_2.glsl`, etc. — `_delete_lib_file` loops `i` until the target path doesn't exist before
   `shutil.move`. This is collision-proof and human-readable (the trash dir reads like a list of
   filenames, not a list of epoch ints). Trade-off: file ordering by deletion time is lost — but
   restore is manual ("which file is `noise_3.glsl`?" is a non-question; the user grep's the body
   or opens it). Subdirectory structure is FLATTENED into the trash dir.

5. **Rename input: validate before commit.** Empty name → no-op. Auto-append `.glsl` if missing
   (mirrors `App.commit_lib_file_new`, validated via `App._validate_lib_target`). Subdirectory rename via `subdir/name.glsl` is
   supported — the input is a path **relative to `lib_root`**. Validation:
   - **Path traversal guard:** the resolved target must be `is_relative_to(lib_root.resolve())`.
     Rejects `../foo.glsl`, absolute paths, symlink escapes. Log warning, leave input open.
   - **Existence collision:** `target.exists()` → reject. Log warning, leave input open. (No
     case-insensitive variant check — let the filesystem adjudicate; on case-sensitive Linux,
     `Foo.glsl` next to `foo.glsl` is allowed.)
   - **Self-rename (same path):** silent no-op (don't bother the user with a "rename to itself"
     warning).
   - Parent dirs are created on demand (`target.parent.mkdir(parents=True, exist_ok=True)`).

6. **Session re-key on rename: in-place mutation of `EditorSession.source`.** `EditorSession` is
   `@dataclass` (NOT frozen); `ShaderSource` IS `@dataclass(frozen=True)`, so re-bind via
   `dataclasses.replace`. Steps:
   - If `editor_sessions[old_path]` exists: pop it; `session.source = replace(session.source, path=new_path)`;
     re-insert under `new_path`.
   - If `_explicit_editor_path == old_path`: set to `new_path`.
   - If `last_lib_path == old_path`: set to `new_path`.
   - If `editor_jump_request is not None and editor_jump_request.path == old_path`: re-point to
     `new_path` (preserve the queued jump).
   - Editor's text content is untouched (only the file's identity changes).

7. **Watcher-driven invalidation handles both delete and rename naturally.** After the on-disk
   mutation, the next frame's `_maybe_rebuild_lib_index` walk produces a `current` dict differing
   from `cached`, triggers `rebuild_lib_index()` + invalidates nodes with `len(sources) > 1`. The
   filter from Decision 3 keeps `.trash/` out of this. **`lib_favorites` / `lib_tags` stores are
   keyed by FUNCTION NAME (not path)** — so favs/tags survive rename automatically (function name
   is preserved), and become harmless orphan keys on delete (no GC needed; trivial JSON).

8. **Delete-while-displayed handling.** If the file being deleted is the one currently shown in
   the editor (`_explicit_editor_path == deleted_path`): `_delete_lib_file` calls
   `show_node_editor()` to drop the explicit override (editor falls back to the current node),
   then `editor_sessions.pop(deleted_path, None)` to evict the stale session, and clears
   `last_lib_path` if equal. This prevents a phantom editor pane displaying a file that no longer
   exists.

9. **`paths.py` gets `lib_trash_dir()` next to `lib_root()`.** Same posture — `mkdir(parents=True,
   exist_ok=True)` on first call. Module-leaf, no `App` import.

10. **Reveal helper extraction.** `App.open_current_node_dir` and the new
    `App.reveal_lib_file_in_manager` share identical platform-dispatch bodies. Extract a
    `util.open_in_file_manager(path: Path)` free function (opens `path` if it's a dir, else
    `path.parent`). Both methods become one-liners that delegate.

11. **Error handling.** Wrap all three new `App` methods (`delete_lib_file`, `rename_lib_file`,
    `reveal_lib_file_in_manager`) in `try/except OSError` → `logger.error(...)` + a
    `Notifications.push(...)` toast. Matches the existing `open_current_node_dir` /
    `flush_current_editor` patterns. Read-only mounts, missing-file races, and permission errors
    surface as a user-visible toast rather than a crash.

12. **Keyboard surface inside the picker:** arrow keys nav the visible function leaves in tree
    order; Enter inserts the selected function at the caret (gated on `current_editor_path is
    not None`); Esc closes the picker. All three are suppressed when an inline input
    (rename / new-file / new-dir / add-tag) owns the keyboard this frame. Ctrl+P opens the
    picker globally — no editor-focus prerequisite (the Insert-at-caret action self-disables
    when no editor target exists).

13. **Directory management** (added during impl, not in the original spec):
    - **New subdirectory**: `commit_lib_dir_new` mkdir's the dir and seeds it with a real
      `SB_<sanitized-dirname>_placeholder` stub in `placeholder.glsl` (an empty-but-visible
      file would render as an un-expandable tree leaf — looks like a bug).
    - **Recursive delete**: `delete_lib_dir` refuses symlinked dirs or any path that resolves
      outside `lib_root`, then trashes **every** file under the subtree (not just `.glsl`) so
      sibling `notes.md` / `.bak` files aren't silently destroyed by the `rmtree`. The dir
      shell is then `rmtree`'d. Numeric-suffix collision rule on each trashed file matches
      single-file delete.

## Files touched

- **`shaderbox/paths.py`** — `lib_trash_dir()`.
- **`shaderbox/lib_index.py`** — shared `is_lib_path(path, lib_root)` predicate applied by
  `LibIndex.build`.
- **`shaderbox/ui.py`** — `_maybe_rebuild_lib_index` applies the same `is_lib_path` filter
  (REQUIRED — prevents the infinite-rebuild loop on trashed files); `_draw_menu_bar` gains a
  `Library → Browse... (Ctrl+P)` entry.
- **`shaderbox/util.py`** — `open_in_file_manager(path: Path)` (shared by `open_current_node_dir`
  and the picker's reveal action).
- **`shaderbox/theme.py`** — three new role tokens: `TAG` (blue_b), `FAVS` (yellow_b),
  `RESET_PILL` (purple_n).
- **`shaderbox/ui_primitives.py`** — `context_menu_style()` ctx-manager (distinct popup fill +
  accent border + accent hover for right-click context menus); `pill_button(label, *, color,
  active, ...)` for colored toggle pills (Favs, tags, Reset).
- **`shaderbox/app.py`** —
  - Picker UI state: `lib_picker_selected_function`, `lib_picker_just_opened`,
    `lib_picker_favs_only`, `lib_picker_disabled_tags`, `lib_picker_new_tag_buf`,
    `lib_picker_tag_input_focused`.
  - Inline-input state: three `InlineInput` instances on `App` — `lib_file_rename`,
    `lib_file_new`, `lib_dir_new` (each carries `.target` / `.buf` / `.needs_focus`); plus
    `lib_file_delete_armed`, `lib_dir_delete_armed`.
  - `reset_lib_inline_state()` — single source of mutex; called by every `begin_*` opener and
    by `open_lib_picker`.
  - File ops: `delete_lib_file`, `rename_lib_file`, `reveal_lib_file_in_manager`,
    `arm_lib_file_delete`, `begin_lib_file_rename`, `cancel_lib_file_rename`,
    `begin_lib_file_new_in`, `cancel_lib_file_new`, `commit_lib_file_new`.
  - Dir ops: `delete_lib_dir` (symlink-refusing, trashes all sibling files),
    `arm_lib_dir_delete`, `begin_lib_dir_new_in`, `cancel_lib_dir_new`,
    `commit_lib_dir_new` (seeds a real `SB_<dirname>_placeholder` stub).
  - `open_current_node_dir` slimmed to delegate to `util.open_in_file_manager`.
- **`shaderbox/popups/lib_picker.py`** — full rewrite: unified tree+preview shape, right-click
  context menus on dir/file/function nodes (via `context_menu_style()`), inline rename / new-file
  / new-dir inputs with × cancel, function tag editor + autocomplete, search + tag-prefix +
  favs filters, arrow-nav by function name.
- **`shaderbox/hotkeys.py`** — Ctrl+P no longer gated on `app.editor_focused`.

## Manual verification

(Maintainer runs in the live app after impl — no headless coverage possible for the UI surface.)

1. **Open via Ctrl+P (editor focused).** Picker opens with the first function leaf selected
   and the search input focused. Insert at caret is enabled.
2. **Open via menu (`Library → Browse...`).** Picker opens; Insert is enabled as long as a
   node is selected or a lib file is currently displayed.
3. **Click into the node grid (no node selected), Ctrl+P.** Insert is disabled with a tooltip:
   "Select a node or open a lib file first — nowhere to insert into".
4. **Right-click the `/` root → New file here.** Inline "New file:" input opens directly under
   the root row. Type `foo`, press Enter → `foo.glsl` lands at lib_root, the picker stays open,
   and the file is opened in the editor.
5. **Right-click a COLLAPSED subdir → New file here.** The subdir auto-expands and the input
   draws inside it. Without the auto-expand the input would be invisible.
6. **Right-click a directory → New subdirectory.** Inline "New dir:" input. Type `complex`,
   press Enter → `complex/` is created with `placeholder.glsl` containing a real
   `SB_complex_placeholder` stub; the new dir + function are visible in the tree on the next
   frame.
7. **Right-click a file → Rename.** Inline input prefilled with the relative path. Type a new
   name, press Enter → file renames on disk; if a session was open on it, the editor still
   shows the correct content under the new identity. Click the small `x` cancel button or
   press Esc to abort.
8. **Rename (path traversal blocked).** Type `../foo.glsl` → toast rejects; input stays open.
9. **Rename (subdir).** Type `subdir/foo.glsl` → file moves into the subdir (created on
   demand); function still resolvable from dependents.
10. **Right-click a file → Delete.** Menu item flips red; file row tints red in the tree.
    Right-click again → "Confirm delete". Click → file goes to `.trash/`; dependents show
    "undeclared identifier" on next compile. Clicking a function leaf (or another file's menu)
    while armed disarms.
11. **Right-click a non-root directory → Delete directory (recursive).** Armed-confirm; on
    confirm, every file under the subtree (`.glsl` + siblings) lands in `.trash/` with the
    numeric-suffix collision rule. The dir shell is removed. A symlinked dir is refused
    (toast: "Delete refused: symlinked or outside lib_root").
12. **Right-click the `/` root.** No Delete item (root is not deletable).
13. **Right-click a function leaf → Insert at caret.** Picker closes; function name lands at
    the editor caret. Repeat with no editor target → the menu item is greyed out.
14. **Right-click function → Open file at declaration.** Picker closes; editor pane shows the
    lib file with the caret at the function's line.
15. **Right-click function → Copy name.** Name in clipboard.
16. **Right-click function → Favorite.** Star turns yellow on the leaf and in the function
    list under "Favs" filter.
17. **Favs / tag pills** above the tree: Favs (yellow) toggles favs-only; Reset (pink) appears
    when any tag is disabled, clears the disabled set; tag pills (blue) toggle one at a time,
    Ctrl+click isolates.
18. **Click the clickable path** in the preview header (gray text under the function name)
    → absolute OS path copied to clipboard; tooltip reads "Click to copy file path".
19. **Per-frame log regression check.** After any delete, the loguru line "Lib index: N
    functions" must appear ONCE (not every frame) — confirms the `.trash/` filter is in both
    `LibIndex.build` and `_maybe_rebuild_lib_index`.

## Open questions for the user

None blocking — robust defaults picked for everything. The Resolution block below records the
calls.

## Review history

- **Pre-impl round 1 (2026-05-28).** Two reviewers (correctness & design + verification & blast
  radius) returned PARTIAL. Findings folded into this revision:
  - Decision 3 now mandates the watcher filter (was hedged).
  - Decision 4 dropped the timestamp scheme (had a real same-second collision footgun); switched
    to basename + numeric-suffix-on-collision.
  - Decision 2 pins the armed-state visual to **color-only** (no label width change → no jitter).
  - Decision 5 added path-traversal sanitization.
  - Decision 6 added the `editor_jump_request.path` re-point.
  - Decision 8 (new) handles delete-while-displayed (drop the explicit override + evict the
    stale session).
  - Decision 11 (new) wraps the methods in `try/except OSError` + toast.
  - Decision 10 (new) extracts the reveal helper.
  - Section-name citations replace line numbers per `dev_flow.md ## Documentation discipline`.
  - Files-touched names the new App state fields and the new `util.open_in_file_manager`
    helper. (Field shape later consolidated into `InlineInput` dataclasses — see current
    Files-touched section.)
  - Manual verification grew steps 4/5 (rename session/displayed split), 7 (path traversal), 9
    (self-rename), 11 (trash collision), 12 (delete-while-displayed), 13b (per-frame log
    regression), 14 (read-only fs).

## Resolution (calls made without user input)

- **Trash filename: basename + numeric suffix on collision.** Alternative was a UUID or epoch
  timestamp; rejected as opaque / collision-prone respectively.
- **Reveal opens the parent dir.** `xdg-open <file>` opens the file in its default app, not the
  file manager. Trade-off: the user doesn't see the file pre-selected; the manager opens at the
  dir listing.
- **UI shape: unified tree+preview with right-click context menus** (Decision 1). The original
  spec locked a separate "Manage files" collapsing section with a per-file action-row; rejected
  mid-impl as visually noisy and replaced by the picker-IS-the-tree shape.
- **Keyboard surface** (Decision 12): arrow-nav over function leaves, Enter inserts, Esc closes —
  all suppressed while an inline input owns the keyboard. Not "mouse only" (the original spec's
  call); the tree needs nav keys.
- **Trash collision strategy: numeric suffix bump in a loop** (`_1`, `_2`, ...). Robust against
  any number of same-basename collisions; trivially readable.
- **Path-traversal sanitization: `is_relative_to(lib_root.resolve())` after `target.resolve()`.**
  Rejects `..`, absolute paths, symlink escapes in one check.
