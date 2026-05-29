# 017 — structure_reorg

High-blast-radius structural refactoring wave. Domain-separation, **not** packaging: group the
genuinely-cohesive clusters, split the files that glue two concerns together, kill the
backward/UI-into-data dependencies — and rename the confusing `lib_*` cluster to `shader_lib`.
No behavior change anywhere.

The shape was chosen via an audit → adversarial-verification convergence loop (4 structural
auditors + a 3-agent devil's-advocate round on the macro layout). The verdict: this is a
domain-separation phase, not a packaging phase. The `ui/` and `render/` mega-packages were
**rejected** — they regroup already-isolated modules for aesthetics, incur ~59 import-line churn,
and force a `theme.py`-into-`ui/` layer inversion (`app.py`, which sits below UI, would import
`from shaderbox.ui.theme`). The moves below each isolate a cohesive concern or remove a real
inversion.

## Goal

Reorganize `shaderbox/`'s flat 30-module-plus-subpackages layout so each module has a clear,
cohesive home, the two domain/UI inversions are removed, and the `lib_*` naming stops reading as
"vendored Python library". Pure structural refactor — every behavior, render path, and UI surface
is byte-for-byte equivalent after.

Concretely, in dependency-safe order:

1. **Rename `lib_*` → `shader_lib` everywhere** (LOCKED by maintainer). New `shader_lib/` package;
   symbols, `paths.py` helpers, the on-disk `<app_data_dir>/lib/` directory, the `Library` menu
   label, and all docs follow. **No migration shim** — no users have downloaded the app yet, so
   backward compat is explicitly out.
2. **Split `lib_index.py` (370 L)** → `shader_lib/{index,resolver,parser}.py` (low-risk, three
   distinct concerns: index data structure + singleton / per-compile resolution / regex parsing).
3. **Split `popups/lib_picker.py` (821 L)** → `popups/lib_picker/{__init__,tree,preview,search,filtering}.py`.
4. **Split `util.py` (220 L) at line 138** — extract the shader-compilation-error domain
   (`ShaderError`, `SourceMap`, `parse_shader_errors`, `next_error_line`,
   `find_uniform_declaration_line`) into `shader_errors.py`; `util.py` keeps the true grab-bag
   helpers (size math, hashing, `pfd_block`, `open_in_file_manager`, formatting). `core.py` then
   imports its own error types from a domain module, not backwards from a junk drawer.
5. **De-tangle `ui_models.py`** — extract `UINode.draw_preview_button` (the sole reason a *data*
   model imports `ui_primitives`) into a free UI helper. `ui_models.py` drops its `ui_primitives`
   import and becomes pure data + (de)serialization.
6. **Tidy `exporters/`** — move `integrations.py` → `exporters/integrations.py`;
   `telegram_util.py` → `exporters/telegram_util.py`; `youtube_util.py` → `exporters/youtube_util.py`.
7. **Move `emoji_data.py` → `popups/emoji_data.py`** (next to its only consumer, `popups/emoji_picker.py`).
8. **Extract App's shader-lib-file CRUD** (~350 L of `app.py`) into a `ShaderLibFileManager`
   collaborator (`shader_lib/file_ops.py`). Requires first moving the shared editor/inline-input
   types to a leaf so the collaborator and `app.py` import them without cycling (see Decision 8).

## Out of scope

Each deferral keeps a concrete trigger; folded into `todo.md` on landing.

- **`ui/` and `render/` mega-packages.** Rejected outright (see header). NOT a deferral — a decision.
  Revisit trigger if ever: the flat top level exceeds ~100 modules, or onboarding friction is
  measured (grep-counts of "where is X"), not assumed.
- **`exporters/telegram.py` (1253 L) and `exporters/youtube.py` (752 L) internal decomposition**
  into `exporter`/`worker`/`panel`/`util` subpackages. Genuinely cohesive but the riskiest diff
  (shared mutable state — `_render_state`, the job/progress queues, `_worker` lifecycle — crosses
  every seam). Deferred. **Trigger:** next time a third concrete exporter lands (the shared
  worker/panel patterns become worth hoisting), OR the first time a telegram/youtube change has to
  touch >300 lines to land a localized fix.
- **`exporters/telegram/` + `exporters/youtube/` subpackages.** This wave moves the `*_util.py`
  helpers *flat into* `exporters/`, not into per-exporter subpackages — the subpackages only pay off
  once the big-file decomposition above happens, so they ride that trigger.
- **Splitting `ui_primitives.py` (714 L).** Cohesive enough (button tiers + draw helpers +
  `preview_cell` + labeled fields all serve one role: the shared imgui+theme primitive set). Audit
  rated it KEEP. **Trigger:** if it crosses ~900 L or a clearly separable cluster (e.g. the exporter
  panel chrome) gets reused outside the exporter context.
- **The built-in coding-copilot agent + its tool-layer (FUTURE FEATURE).** A chat widget driving an
  agent that can manipulate the whole app (create shaders/nodes, set uniforms, manage lib files), its
  tools wrapping the app's mutation verbs. NOT built here. This wave only ensures the structure is
  *expandable* to it (Decision 16: rewritten verbs are agent-callable). **Trigger:** when that
  feature is specced. Known gaps the tool-layer will have to close (discovered during this audit, so
  the future spec doesn't re-derive them): (a) **no `set_uniform_value(node_id, name, value)` verb
  exists** — uniform mutation happens inline in `widgets/uniform.py`'s draw loop, mutating
  `UIUniform` through the imgui drag widget; a tool needs a headless verb. (b) **`create_node_from_selected_template`
  reads `app_state.selected_node_template_id`** (grid selection), not a `template_id` arg — a tool
  wants `create_node(template_id)`. (c) the agent layer must reach the mutation verbs **without
  importing imgui** — confirm the verb-holding modules (`app.py`, `ShaderLibFileManager`, a future
  node-ops module) stay imgui-free so a headless/tool context can import them. The seam to attach to
  is the existing `App.<verb>()` surface, not a new code path.

## Design decisions (LOCKED)

1. **Flat base, no `ui/`/`render/` packages.** The UI modules (`ui.py`, `theme.py`,
   `ui_primitives.py`, `notifications.py`, `hotkeys.py`) and render-domain modules (`core.py`,
   `media.py`, `render_preset.py`, `fonts.py`) stay flat at top level. `tabs/`, `widgets/`,
   `popups/` already group the UI sub-concerns; a parent `ui/` adds depth and a layer inversion for
   zero navigational gain. Render modules have near-zero internal coupling (only `core` imports
   `media`+`render_preset`) — a package implies cohesion that isn't there.

2. **`shader_lib/` package** holds `index.py`, `resolver.py`, `parser.py`, `favorites.py`,
   `tags.py`, `file_ops.py`. It imports only leaf modules (`paths`, `shader_source`, `util`,
   `shader_errors`) — it MUST NOT import `app.py` (would cycle: `app.py` imports `shader_lib`).
   `popups/lib_picker/` (the picker UI) imports `App` and `shader_lib`, and is imported by `ui.py`
   — that direction is fine (UI → app + domain).

3. **`shader_lib` naming is total.** Package dir, module names, the `LibIndex`/`LibFunction`
   classes → `ShaderLibIndex`/`ShaderLibFunction`(*) , `LibFavoritesStore` → `ShaderLibFavoritesStore`,
   `LibTagsStore` → `ShaderLibTagsStore`, `lib_root()`/`lib_trash_dir()` → `shader_lib_root()`/
   `shader_lib_trash_dir()`, `is_lib_path()` → `is_shader_lib_path()`, the on-disk
   `app_data_dir()/lib/` → `app_data_dir()/shader_lib/`, the `Library` main-menu label →
   `Shader Library`, and **every** `lib`-named App state field. The field set is exhaustive (per
   review — the `lib_picker_*` shorthand below is NOT the full list): **`is_lib_picker_open`**
   (→ `is_shader_lib_picker_open`; the popup-mutex flag — set/cleared in `any_popup_open` + every
   `open_*` helper in `app.py`, read at `popups/lib_picker.py:44,50`; the opener method
   `open_lib_picker` → `open_shader_lib_picker` renames with it), `lib_picker_query`,
   `lib_picker_selected_function`, `lib_picker_just_opened`, `lib_picker_favs_only`,
   `lib_picker_disabled_tags`, `lib_picker_new_tag_buf`, `lib_picker_tag_input_focused`,
   `lib_file_rename`, `lib_file_new`, `lib_dir_new`, `lib_file_delete_armed`,
   `lib_dir_delete_armed`, `lib_index`, `lib_favorites`, `lib_tags`, **`last_lib_path`** (→
   `last_shader_lib_path`; read at `tabs/code.py:122-127` for the "back to lib" button — that
   call site updates too), and `_explicit_editor_path` keeps its name (not lib-specific). No
   `lib`-prefixed identifier survives where it refers to this feature. (*) Class renames are
   in-scope but stay greppable: rename the class, update every reference. The `App` method names
   (`open_lib_file`, `rebuild_lib_index`, `begin_lib_file_new_in`, …) rename to `shader_lib_*` /
   `*_shader_lib_*` consistently. **The single source of truth for completeness is a clean
   `grep -rin 'lib' shaderbox/ tests/` after the rename: every remaining hit must be either an
   unrelated word (e.g. `.venv/lib`, `available`, `calibrate`) or intentional — zero
   feature-`lib` identifiers left.**

4. **No migration / back-compat.** The on-disk dir rename is a hard cut. No reading the old `lib/`
   path, no symlink, no one-time move. (Maintainer: nobody has downloaded it.)

5. **`lib_index.py` → three modules.** `shader_lib/index.py` = `ShaderLibIndex`,
   `ShaderLibFunction`, the module-level `_ACTIVE_INDEX` singleton + `active()`/`set_active()`,
   `is_shader_lib_path()` (the public surface `app`/`core`/picker import). `shader_lib/resolver.py`
   = `resolve_usage()` + `ResolveError` + the per-compile flatten internals (`_ResolveState`,
   topo-sort, `#line` emission). `shader_lib/parser.py` = `_extract_functions` + the
   comment-strip/brace-match/doc-extract regex machinery (called only by `ShaderLibIndex.build`).
   `core.py` imports `active`/`resolve_usage` from `shader_lib` (re-exported via
   `shader_lib/__init__.py` so the call sites read `from shaderbox.shader_lib import …`).

6. **`lib_picker.py` → package `popups/lib_picker/`.** `__init__.py` exposes `draw_lib_picker` (the
   entry `ui.py` imports — its import line stays `from shaderbox.popups.lib_picker import
   draw_lib_picker`) + the `_draw_body` orchestrator. `tree.py` = tree build/walk + the file/dir/
   function context menus + inline new/rename inputs. `preview.py` = preview pane + tag editor.
   `search.py` = search row, tag bar, query-tag parsing. `filtering.py` = filter logic, arrow-nav,
   selection, insert-at-caret, open-at-decl, clipboard. All take `app: App` (the existing stateless
   style) — no new shared state introduced.

7. **`util.py` split at the shader-error seam.** New `shader_errors.py` (leaf: imports only stdlib +
   `pathlib`) holds `ShaderError`, `SourceMap`, `parse_shader_errors`, `next_error_line`,
   `find_uniform_declaration_line`, the two driver-error regexes. `util.py` keeps `adjust_size`,
   `select_next_value`, `get_resolution_str`, `get_uniform_hash`, `unicode_to_str`, `str_to_unicode`,
   `try_to_release`, `pfd_block`, `open_in_file_manager`, `format_auto_value`. Update importers:
   `core.py`, `hotkeys.py`, `shader_lib/index.py`+`resolver.py` (was `lib_index`), `tabs/code.py`,
   `widgets/uniform.py`, `tests/test_util.py`.

8. **`UINode.draw_preview_button` → free helper.** It must leave `ui_models.py` so the data model
   stops importing the UI layer. It can NOT go in `ui_primitives.py` (generic — must not import
   `ui_models`/`UINode`, would invert). So it goes where the callers are, as a free
   `draw_node_preview_button(ui_node, border_color, size, selected=False, armed=False)`. **Home:
   `widgets/node_grid.py`** (the primary caller). **Two call sites must both switch from
   `ui_node.draw_preview_button(...)` to the free fn:** `widgets/node_grid.py` (the node grid) AND
   `popups/node_creator.py:39` (the template grid — `node_creator.py` therefore joins the
   touched-files list and imports the helper from `widgets/node_grid`). The helper imports
   `preview_cell` from `ui_primitives` (a new `node_grid → ui_primitives` edge — verified
   non-cyclic: `ui_primitives` imports only `theme`). `ui_models.py` loses its
   `from shaderbox.ui_primitives import PreviewCellResult, preview_cell` line entirely and becomes
   pure data.

9. **Shared editor types move to a leaf for the CRUD extraction.** `EditorSession`, `InlineInput`,
   `JumpRequest`, `HoverMark` currently live on `app.py` and are imported FROM `app.py` by
   `hotkeys.py`, `tabs/code.py`, `widgets/uniform.py`, `popups/lib_picker`. Move all four to a new
   leaf `editor_types.py` (imports only stdlib + `imgui_color_text_edit` for the `TextEditor`
   field). Every current `from shaderbox.app import App, JumpRequest` splits into
   `from shaderbox.app import App` + `from shaderbox.editor_types import JumpRequest`. This is the
   prerequisite that makes Decision 10 cycle-free.

10. **`ShaderLibFileManager` collaborator.** Extract the ~350 L of shader-lib-file CRUD from
    `app.py` (the `reset_lib_inline_state`, `arm_*`, `begin_*`, `cancel_*`, `commit_*`,
    `delete_lib_file`, `delete_lib_dir`, `rename_lib_file`, `reveal_lib_file_in_manager`,
    `_validate_lib_target`, `open_lib_file`, `get_session`, `show_node_editor`, `rebuild_lib_index`
    methods + the `shader_lib_picker_*`/`shader_lib_file_*`/inline-input fields) into
    `shader_lib/file_ops.py::ShaderLibFileManager`. The manager holds the editor-session +
    inline-input + selection state and takes injected collaborators it needs:
    `editor_sessions: dict[Path, EditorSession]`, `notifications: Notifications`, and a small
    callback/handle set for the things that reach back into node state
    (`current_editor_path`, the active node's source path). `App` holds one
    `self.shader_lib: ShaderLibFileManager` and delegates. `file_ops.py` imports `editor_types`,
    `notifications`, `paths`, `shader_lib.index`, `shader_source` — NOT `app.py`. **If at impl a
    method needs an `App` symbol that can't be cleanly injected, that's the cycle-from-types signal
    (dev_flow.md): stop and reconcile the seam, don't `TYPE_CHECKING` past it.** Target: `app.py`
    drops from 1103 to ~750 L. **Per Decision 16, the manager's `create_file` / `create_dir` /
    `rename` / `delete_*` take explicit args (path, body) — they do NOT read the inline-input
    buffers. The picker keeps a thin shim that reads its buffer then calls the explicit method, so
    the same verb is reachable by a non-UI caller.**

11. **`integrations.py` → `exporters/integrations.py`.** 4 of its 5 importers are under
    `exporters/`; `app.py`'s one import (load/save the store) is the only top-level user. Move it
    in; update `app.py`, `exporters/{base,registry,telegram,youtube}.py`, `tests/test_youtube_exporter.py`.

12. **`telegram_util.py` / `youtube_util.py` → flat into `exporters/`.** Single-caller exporter
    helpers. `youtube_util.py` imports `exporters.base.ExporterValueError` (stays valid in-package).
    Update `exporters/{telegram,youtube}.py` + `tests/test_youtube_util.py`.

13. **`emoji_data.py` → `popups/emoji_data.py`.** Sole importer is `popups/emoji_picker.py`.
    `emoji_picker.py` stays in `popups/` (NOT moved under `exporters/telegram/` — it's reached via
    the generic `OutletUiDeps.open_glyph_picker` callback + an `App` method; pulling it into
    `exporters/` would make the exporter layer import imgui draw code + the App callback, violating
    `conventions.md`'s "generic exporter seam carries no exporter-domain vocabulary" decision).

14. **`tabs/share_state.py` stays in `tabs/`.** `conventions.md ## Design decisions` explicitly
    sanctions a state-only sibling module in `tabs/` as the cycle-break (`app.py` imports
    `share_state`, not `share`). Moving it is churn that re-opens a settled decision. KEEP.

15. **`core.py` importing `shader_lib` is NOT a layer violation.** Compile-time `#include`
    resolution is intrinsic to what `Node.compile()` does; the render domain legitimately depends on
    the shader-library resolver. Preserved as-is (just the import path + symbol names change).

16. **Imperative mutation verbs take explicit args, never read UI state — the agent-seam lens.**
    A future built-in coding-copilot agent (chat widget that manipulates the app: create shaders /
    nodes, set uniforms, manage lib files) will attach as *another caller of the mutation verbs* —
    a sibling to `hotkeys.py` / the tabs, not a new code path. The structure already supports this:
    nearly every state change flows through an `App.<verb>()` method that the UI layer thinly calls
    (`hotkeys.py` → `app.create_node_from_selected_template`, `node_grid.py` → `app.delete_node`).
    This wave is rewriting `ShaderLibFileManager`'s methods anyway, so — at no extra cost — shape
    every verb it touches to be **agent-callable**: take explicit arguments (path, body, value,
    node_id), return the result, push notifications, and do NOT read picker/inline-input buffers or
    `app_state.selected_*` to discover their inputs. Where a current method reads UI state to find
    its input (`commit_lib_file_new` reads `self.lib_file_new.buf`), split it: an explicit
    `create_file(path, body)` core + a thin UI shim that reads the buffer then calls the core. This
    is a CONSTRAINT on the rewrite, **not** a new abstraction — no `api/`/`commands/` layer is built
    now (premature before the agent consumer exists to validate its shape). It only ensures the
    verbs the refactor rewrites don't have to be rewritten again when the agent lands. Scope
    boundary: this applies ONLY to verbs already being rewritten this wave (the `ShaderLibFileManager`
    methods). Node/uniform verbs that aren't otherwise touched stay as-is — the gaps there are
    captured in the `todo.md` deferral, not retrofitted now.

## Files touched

New files:
- `shaderbox/shader_lib/__init__.py` (re-exports the public surface)
- `shaderbox/shader_lib/index.py`, `resolver.py`, `parser.py` (from `lib_index.py`)
- `shaderbox/shader_lib/favorites.py` (from `lib_favorites.py`), `tags.py` (from `lib_tags.py`)
- `shaderbox/shader_lib/file_ops.py` (`ShaderLibFileManager`, extracted from `app.py`)
- `shaderbox/popups/lib_picker/{__init__,tree,preview,search,filtering}.py` (from `popups/lib_picker.py`)
- `shaderbox/shader_errors.py` (from `util.py` lines 138-220)
- `shaderbox/editor_types.py` (`EditorSession`, `InlineInput`, `JumpRequest`, `HoverMark` from `app.py`)
- `shaderbox/exporters/integrations.py`, `exporters/telegram_util.py`, `exporters/youtube_util.py` (moved)
- `shaderbox/popups/emoji_data.py` (moved)

Deleted (moved-from): `lib_index.py`, `lib_favorites.py`, `lib_tags.py`, `integrations.py`,
`telegram_util.py`, `youtube_util.py`, `emoji_data.py`, `popups/lib_picker.py`.

Heavily edited: `app.py` (lose CRUD + types; gain `self.shader_lib`; rename methods/fields),
`core.py`, `ui.py`, `ui_models.py`, `hotkeys.py`, `paths.py`, `widgets/uniform.py`,
`widgets/node_grid.py` (gains `draw_node_preview_button`), `popups/node_creator.py` (its
`draw_preview_button` call site — review-added), `tabs/code.py` (the `last_shader_lib_path`
back-to-lib button + `shader_errors` import), `tabs/share.py`,
`exporters/{base,registry,telegram,youtube}.py`. (`tabs/node.py` was previously listed — REMOVED
per review: zero `lib` refs, uses only kept-half `util` symbols + `ui_models`; untouched by both
the rename and the `ui_models` de-tangle.)

Tests: `test_lib_index.py` → `test_shader_lib.py` (or keep name, fix imports), `test_lib_tags.py`,
`test_util.py`, `test_youtube_util.py`, `test_youtube_exporter.py`. (`test_render_for.py` imports
`tabs.share_state` — unchanged.)

Docs (specific lines pinned per review): `dev_flow.md ## Recipes > Module map` (stale + missing 6
modules — rewrite to reality; the `paths.py` bullet's `lib_root()`/`lib_trash_dir()` →
`shader_lib_root()`/`shader_lib_trash_dir()`), `roadmap.md` (017 row + banner), `conventions.md`
(the `## Known quirks` GLSL-`#line` bullet citing `util.SourceMap.file_id_to_path` +
`lib_index.py::resolve_usage` → `shader_errors.SourceMap` + `shader_lib/resolver.py::resolve_usage`;
the `InlineInput` Design-decision bullet — clarify its new home is `editor_types.py`, keep its
"promote to `ui_primitives.py`" revisit-note as the future trigger; the three-layer-UI +
`tabs/share_state.py` bullets stay valid), `CLAUDE.md` (any nav references), `todo.md` (**the
cross-file-uniform-jump entry** cites `util.find_uniform_declaration_line` →
`shader_errors.find_uniform_declaration_line`; **the macro-indirection entry** cites
`lib_index._extract_functions` → `shader_lib.parser._extract_functions`; the multi-file-editor
deferrals' lib paths; ADD the new cut/deferred entries: `ui/`+`render/` rejection,
telegram/youtube decomposition, `ui_primitives` split, the agent-API feature). On-disk `lib/` →
`shader_lib/` references in the `/imgui-ui` skill / any skill that names the path. **UNCHANGED (do
NOT touch): the sticker-loop-offset `todo.md` entry citing `tabs/share_state.py`/`core.py` — those
modules don't move.**

## Implementation order (separate commits — move before split, so git history stays followable)

The convergence loop flagged the git-blame landmine: moving AND splitting a file in one commit
double-fragments its history. **And renaming symbols/imports *inside* a file in the same commit that
moves the file defeats git's rename detection** (git sees delete+add, not a rename, so `git log
--follow` breaks). So step 1 splits into 1a (pure `git mv`, zero content edits) then 1b (the
content rename), each its own commit:

1a. **Pure `git mv` only** — relocate `lib_*.py → shader_lib/`, `integrations.py` +
    `telegram_util.py` + `youtube_util.py` → `exporters/`, `emoji_data.py → popups/`. NO content
    edits in this commit (not even the `import` lines — the tree is intentionally red between 1a and
    1b). This is the commit git records as renames, preserving blame. (If a fully-green intermediate
    is preferred over a red one, 1a may instead keep shim re-export stubs at the old paths, deleted
    in 1b — but the maintainer's "no back-compat" steer favors the simple red-then-green pair.)
1b. **The `shader_lib` content rename** — fix all imports to the new paths, rename the
    classes/functions/methods/fields/menu-label/on-disk-dir per Decisions 3-4. `make check` +
    `make smoke`. Tree green again.
2. **`util.py` split** (extract `shader_errors.py`) + `editor_types.py` extraction (move the four
   types out of `app.py`). `make check`.
3. **`lib_index.py` → index/resolver/parser** split (now inside `shader_lib/`). `make check`.
4. **`lib_picker.py` → package** split. `make check` + `make smoke`.
5. **`ui_models` de-tangle** (`draw_preview_button` → `widgets/node_grid.py`; both call sites). `make check` + `make smoke`.
6. **`ShaderLibFileManager` extraction** from `app.py`. `make check` + `make smoke`.
7. **Docs reconciliation** + `/sanitize`. Sandbox sync (`git add projects/dev`).

(Each step from 1b onward lands independently green; if a later step proves wrong, earlier steps
already shipped. 1a is the one deliberately-red commit — keep it adjacent to 1b in the same wave.)

## Manual verification

`make smoke` (200 headless frames + popup-mutex/`current_node_id` invariants) after every UI-touching
step gates regressions an import-path break would cause. But several surfaces smoke can't exercise —
hand to the maintainer after the wave:
- **Shader-library round-trip:** `Ctrl+P` opens the picker; fuzzy search finds an `SB_*` function;
  insert-at-caret splices it; the node compiles with the spliced preamble. (Exercises `shader_lib`
  rename + `lib_picker/` split + `ShaderLibFileManager`.)
- **Lib-file CRUD:** right-click context menus → new file / new dir / rename / delete (armed
  confirm) / reveal all work; deletion lands in `.trash/`; the index rebuilds. (Exercises the
  `ShaderLibFileManager` extraction end-to-end.)
- **On-disk path:** the app reads/writes `<app_data_dir>/shader_lib/`, not `lib/`. (A fresh
  `make run-bundle` with a throwaway data dir confirms the new-user path.)
- **Shader-error click-to-jump:** introduce a GLSL error; the error strip lists it; clicking jumps
  the caret. (Exercises `shader_errors.py` extraction + the `SourceMap`/`#line` path.)
- **Node preview grid + template grid** render with borders/selection/armed tint. (Exercises the
  `draw_node_preview_button` extraction — BOTH the node grid `widgets/node_grid.py` AND the
  node-creator template grid `popups/node_creator.py:39`.)
- **Main-menu label** reads **`Shader Library`**, not `Library` (review-added — `ui.py:283
  begin_menu`), and the menu's items still open the picker / reveal the dir. (Exercises the
  menu-label rename.)
- **mtime hot-reload of a lib file** (review-added — uncovered by the other checks): edit an
  `SB_*` lib file on disk while the app runs → the index rebuilds, dependent nodes recompile, and
  an open editor session for that file re-syncs from disk. (Exercises `sync_editor_from_disk` +
  `rebuild_shader_lib_index` after the rename + `ShaderLibFileManager` extraction.)
- **Telegram + YouTube panels** still draw their credential/upload UI (no live secrets needed to
  confirm the panels render). (Exercises the `integrations`/`*_util` moves.)

## Open questions for the user

None outstanding — scope, layout, split-depth, the CRUD-extraction inclusion, the `shader_lib` name,
and the docs-in-same-wave call are all locked (see conversation). Decision 8's exact home for the
preview helper is an impl-time call within the locked constraint (must not invert `ui_primitives` →
`ui_models`).

## Review history

**Pre-impl swarm, round 1** (3 adversarial reviewers, `review-agent-loop` discipline; one anchored
to source-on-disk, not to this spec).
- **Fact-checker (anchored to source): PASS** — verified 40/40 factual claims (import edges, line
  ranges, sole-importer assertions, the `util.py` line-138 seam, `shader_errors` being moderngl-free,
  the `lib_index` 3-way separability) against the real files. Zero claims wrong. Only nit: `_strip_comments`
  is shared by parser + resolver paths — an impl-time placement call, non-blocking (put it in `parser.py`).
- **Cycle/correctness auditor: PASS (98/100)** — post-refactor import graph stays acyclic; no
  `TYPE_CHECKING` needed. Confirmed the `shader_lib/__init__.py` re-export does NOT drag a cyclic
  chain into `core.py`. The 2-point reservation is the `ShaderLibFileManager` callback-injection seam
  (Decision 10) — already gated in-spec ("if a method needs an un-injectable `App` symbol, stop and
  reconcile, don't `TYPE_CHECKING` past it").
- **Blast-radius/docs auditor: PARTIAL → all 5 findings triaged REAL and FIXED this round:**
  1. `last_lib_path` (app.py:262 +5 sites + `tabs/code.py:122-127`) was outside Decision 3's field
     list → Decision 3 now enumerates the full field set + a `grep -rin 'lib'` completeness gate.
  2. `tabs/node.py` was wrongly listed heavily-edited (zero `lib` refs) → REMOVED from Files-touched.
  3. `popups/node_creator.py:39` also calls `draw_preview_button` → added to Files-touched + Decision 8.
  4. Library menu label (`ui.py:283`) + mtime hot-reload had no manual check → both added.
  5. todo.md (`find_uniform_declaration_line`, `_extract_functions`) + conventions.md (`SourceMap`,
     `resolve_usage`) stale paths → pinned by exact entry in the Docs scope.
  Plus the git-blame landmine (symbol-rename-in-move-commit defeats rename detection) → step 1 split
  into 1a (pure `git mv`) + 1b (content rename).
  Every finding re-verified by the main agent against source before patching (not taken on trust).

**Pre-impl swarm, round 2** (focused re-spawn of the blast-radius reviewer only — fact-checker +
cycle-auditor PASSed round 1 and the patches touched neither the factual premises nor the import
graph, so re-running them would be wasted tokens).
- **Blast-radius/docs auditor: PARTIAL → 1 real miss, FIXED:** 5 of 6 round-1 fixes confirmed
  landed + both new-damage checks (popups→widgets direction, the `grep` gate soundness) PASS. The
  one genuine miss: Decision 3's "exhaustive" field list omitted **`is_lib_picker_open`**
  (app.py:190 — the popup-mutex flag, used in `any_popup_open`/`open_*` + `popups/lib_picker.py:44,50`).
  Re-verified against source (the full `self.*lib*` field set), then added to Decision 3 along with
  its opener method `open_lib_picker → open_shader_lib_picker`. The reviewer self-reported the other
  checks as genuine PASSes (not manufactured) — the convergence signal.

**Converged.** Round 2's sole finding was a one-field enumeration gap (now closed); everything else
PASS. The factual premises (round 1 fact-checker, 40/40) and acyclicity (round 1 cycle-auditor) are
unaffected by the round-2 patch. No further round warranted — a round 3 would be manufacturing
findings against a spec the swarm now agrees is sound. Spec is implementation-ready pending
maintainer plan-lock sign-off.
