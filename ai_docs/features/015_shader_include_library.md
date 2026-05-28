# 015 — Shader library (auto-resolve)

A cross-project GLSL utility library. The user accumulates reusable helpers (noise, SDF, palette,
hash) in one place (`<app_data_dir>/lib/**.glsl`) and **calls them by name** from any shader —
`float r = SB_perlin_noise_3(x, y, z);` Just Works, no `#include`, no ceremony. Library functions
behave like GLSL built-ins.

Builds on feature 014's seams (`ShaderSource`, `CompileUnit`, `SourceMap`, `EditorSession` keyed by
path). Without 014, this feature would be cross-cutting; with it, the resolver + picker + multi-file
editor drop in cleanly.

## Goal

1. **Cross-project library on disk.** `<app_data_dir>/lib/**.glsl` — files survive projects (same
   posture as `integrations.json`). The `.glsl` extension is the convention; arbitrary names +
   subdirectories allowed.
2. **Auto-resolve by identifier.** The user writes `SB_perlin_noise_3(...)` in their shader; the
   host scans the user's text for `SB_\w+` identifiers, intersects with the lib index, prepends
   ONLY the needed functions (plus transitive callees), and hands the result to the driver. No
   `#include` syntax. No directive. No opt-in.
3. **Lib functions feel like built-ins.** No order-of-declaration issues (host topologically sorts
   the prepended subset). No "did I forget the include?" (there's no include to forget). User can
   shadow a lib function by defining `SB_foo` in their own shader — the lib version is suppressed
   for that compile.
4. **Driver errors land in the right file.** The `SourceMap` machinery from 014 (file_id → path)
   stays — each prepended lib chunk gets its own file-id; `#line N <id>` markers route driver
   errors back to the lib file's line. The error strip + click-to-jump + F8 + gutter markers all
   thread through `ShaderError.path` unchanged.
5. **Save fan-out via the existing mtime watcher.** Each node's `compile_unit.sources` lists
   exactly the lib files its used functions came from (no more, no less). Editing a lib file fans
   out to every node whose include set contains it; active recompiles immediately, dormant
   recompile lazily on activation. Adding a new function to a lib file invalidates the lib index
   cache; nodes that newly reference it pick it up on their next compile.
6. **Picker for discovery.** `Ctrl+P` opens a fuzzy-search modal listing every `SB_*` function in
   the lib (built from the same index). Each row: name · signature · description (`///` doc-comment
   above the function, optional). Enter inserts the function name at the caret; Ctrl+Enter opens
   the lib file at the function's declaration line.
7. **First-class multi-file editor UI.** Lib files open in their own path-keyed `EditorSession`s
   (014). Tab bar above the editor: pinned current-node-shader tab + N closable lib tabs. Switching
   nodes pivots the pinned tab; lib tabs persist across node switches.

## Non-goals — this feature does NOT

- **Use `#include` syntax.** Anywhere. Throw out the resolver from steps 1-3 of the earlier
  design; the picker doesn't insert `#include` lines, the user can't write them (they'd pass
  through as text and produce a driver error). The lib is implicit, like built-ins.
- **Validate the library at boot.** No "Library Status" panel, no per-file stub compile. The user
  IS the lib author; broken lib functions surface as normal compile errors in the user's shader,
  with the source map landing them in the lib file. Total cost: zero new UI surface, zero
  fragile stub-compile machinery.
- **Enforce the `SB_` prefix at the engine layer.** The resolver doesn't care. The prefix is a
  convention enforced at the **picker** layer only: the picker shows `SB_*` functions; anything
  not `SB_`-prefixed is treated as a private helper of its lib file (still usable internally by
  other lib functions, just not surfaced to the user).
- **Support per-project libs.** One root: `<app_data_dir>/lib/`. If a real need surfaces later
  (sharing a shader between users via the project folder), add a second search root; **don't
  pre-build it.**
- **Adopt a full preprocessor** (`pcpp`, `mcpp`). No `#define` substitution, no conditional
  compilation, no overload resolution. The user's `#define MYNOISE SB_perlin_noise_3` works
  because we scan the user text for `SB_\w+` tokens AS TEXT — but `#define HASH SB_hash3` *inside
  a lib file* used via macro indirection in *another lib file* is undefined behaviour (lib-author
  responsibility — see decision 5).
- **Build a snippet-palette UI** (the "insert this helper at cursor" affordance from research).
  The picker IS the discovery UI. Inserting the function NAME is enough.
- **Export-from-selection** ("select function in editor → push to lib"). Coming in a later
  feature; out of scope here.
- **Auto-inject lib top-level globals (uniforms / `in` / `out` / interface blocks).** Lib files
  are pure-function helpers. If a lib function references `u_time`, the user's shader had better
  declare `u_time` itself; the lib doesn't get to inject globals. This is the cognitive-clarity
  guarantee — uniforms are the user's vocabulary, not the lib's.
- **Track / detect cross-file uniform declarations from the lib.** `find_uniform_declaration_line`
  still scans only the user's editor text. Filed as `[DEFERRAL]` if this becomes a friction.

## Design decisions (locked)

1. **One lib root: `<app_data_dir>/lib/`.** Created lazily by `paths.lib_root()`. Recursion via
   `Path.glob("**/*.glsl")` for discovery.

2. **`LibIndex` is the central data type.** A new module `shaderbox/lib_index.py`. Shape:
   ```python
   @dataclass(frozen=True)
   class LibFunction:
       name: str
       signature: str         # the full declaration line, e.g. "float SB_hash(vec2 p)"
       body: str              # the full function definition (signature + braced body)
       file: Path
       line_in_file: int      # 0-based line of the signature
       calls: frozenset[str]  # other identifiers referenced inside the body (call graph edges)
       doc: str               # /// comment block immediately above (may be empty)

   @dataclass(frozen=True)
   class LibIndex:
       functions: dict[str, LibFunction]  # name -> function
       sources: dict[Path, ShaderSource]  # file -> snapshot (for mtime fan-out)

       @classmethod
       def build(cls, lib_root: Path) -> "LibIndex": ...
   ```
   Built by walking `lib_root.glob("**/*.glsl")`, stripping comments, regex-extracting top-level
   function definitions. Cached on the `App`; invalidated when any `lib_root/**.glsl` mtime
   changes (mtime check runs in the existing watcher loop).

3. **Usage resolution at compile time.** New module `shaderbox/lib_resolver.py` (or fold into
   `lib_index.py` — keep small). Pure function:
   ```python
   def resolve_usage(
       root: ShaderSource, index: LibIndex
   ) -> tuple[str, list[ShaderSource], SourceMap, list[ResolveError]]
   ```
   Algorithm:
   1. Strip comments from `root.text`.
   2. Extract `\bSB_\w+\b` identifiers referenced in stripped text → `used`.
   3. Extract `\bSB_\w+\b` identifiers DEFINED in `root.text` (regex: `\b\w+\s+(SB_\w+)\s*\(`) →
      `defined_by_user`. The user shadows a lib function by defining it locally.
   4. `to_prepend = {n for n in used if n not in defined_by_user and n in index.functions}`.
   5. Transitive-close `to_prepend` over `index.functions[n].calls`.
   6. Topologically sort `to_prepend` (a function must be declared before its callers).
   7. Build the preamble: for each lib file containing a function in `to_prepend`, emit a
      `#line 1 <id>` marker + that file's content slice (just the used functions, in order).
   8. Insert the preamble into `root.text` AFTER any leading `#version` / `#extension` lines and
      BEFORE the rest. Emit `#line <root_resume_line> 0` to restore user line numbering.
   9. Return flattened text + sources list (one per touched lib file) + populated `SourceMap` +
      any resolve errors (cycle in lib call graph, missing transitive dep, etc. — rare).
   `ResolveError` mirrors `ShaderError` shape; surfaced via `CompileUnit.errors`.

4. **No `SB_` prefix enforcement at the engine layer.** A function not prefixed `SB_` in a lib
   file is still callable from a lib file that lists it in `calls` (e.g. an internal helper).
   But: only `SB_*` identifiers are auto-resolved from user text (the regex in step 3.2 hardcodes
   `SB_\w+`). Non-prefixed lib functions = private to the lib.

5. **Lib authors are responsible for their own macro indirection.** If `lib/a.glsl` has
   `#define HASH SB_hash3` and `lib/b.glsl` uses `HASH(x)` to call a lib function, the regex
   won't trace the dependency. **Convention**: lib files don't use `#define` for function
   dispatch. Document in `conventions.md ## Design decisions` if it becomes load-bearing.

6. **The preamble is inserted after `#version` and any `#extension` lines.** GLSL requires
   `#version` as the first non-comment, non-whitespace token. Inserting before would fail to
   compile. The resolver detects the header boundary by scanning for the first line that's not
   blank, not a comment, and not a `#version`/`#extension`/`#pragma` directive.

7. **Picker is built from the `LibIndex`, not from filesystem walks.** Open with `Ctrl+P`; fuzzy
   match against function names + descriptions; show ALL functions whose name starts with `SB_`
   (anything not prefixed is hidden from the picker — the convention layer). Per-row UI: name +
   signature + doc; preview pane on the right shows the body. Two actions:
   - **Enter** (or click): insert just the function name at the caret. The user will type the
     parens themselves.
   - **Ctrl+Enter** (or a button): open the lib file at the function's declaration line via the
     existing `JumpRequest(path, line, column)` mechanism (014).
   - **"+ New library file"** action at the bottom of the picker: prompts for a filename, creates
     it empty in `lib_root`, opens a new `EditorSession` on it.

8. **Multi-file editor: tab bar above the editor.** Pinned current-node-shader tab on the left
   (cannot be closed; switches content when the user changes nodes via the node-grid or arrow
   keys); N closable lib tabs to the right (persist across node switches). Click a tab → switch
   active editor; the existing `tabs/code.py` becomes `tabs/code_pane.py` with the tab-bar at
   the top and the editor body unchanged. `app.current_editor_path` becomes the source of truth
   for which session is visible.

9. **Lib mtime fan-out (carries over from earlier step 3).** The watcher in `ui.py` iterates
   `compile_unit.sources` per node — already shipped. The new design narrows the set: instead of
   "every lib file ever included by anything," each node's `sources` lists exactly the lib files
   whose functions it transitively used at last compile. Less wasted work; same shape.

10. **No on-disk schema change for nodes.** `node.json` unchanged. The lib index is a runtime
    derivation from filesystem state; not persisted.

11. **Resolver-domain failures surface as synthetic `ShaderError`s in `CompileUnit.errors`** —
    e.g. a cycle in the lib call graph (`SB_a` calls `SB_b` calls `SB_a`). Reported at the
    USER's `#include`-equivalent — wait, there is none. Reported at line 0 of the root with a
    clear "library cycle: SB_a → SB_b → SB_a" message. Rare in practice (the lib author has to
    create the cycle deliberately); a fallback row, not a primary error path.

## The layering contract being preserved

- **Engine side.** `lib_index.py` (LibIndex), `lib_resolver.py` or fold-in (resolve_usage),
  `core.py` (Node.compile calls resolver), `util.py` (SourceMap.resolve from 014). None of these
  import imgui or know about TextEditor.
- **UI side.** `app.py` (lib editor session creation), `popups/lib_picker.py` (the modal),
  `tabs/code_pane.py` (multi-file draw), `hotkeys.py` (Ctrl+P). All can import engine types but
  not vice-versa.

## Files touched

- **NEW: `shaderbox/lib_index.py`** — `LibFunction`, `LibIndex`, `LibIndex.build(lib_root)`,
  comment-strip helper, top-level function-def regex.
- **NEW: `shaderbox/lib_resolver.py`** — `resolve_usage(root, index)`. (Or merge into
  `lib_index.py` to keep file count down; ~150 lines combined is fine.)
- **REWRITE: `shaderbox/shader_include.py`** — delete the file; its responsibilities move to
  `lib_index.py` + `lib_resolver.py`. Or keep the name and rewrite — TBD during implementation;
  the name `shader_include.py` is a lie now (we don't have includes), so renaming is better.
- `shaderbox/core.py` — `Node.compile()` calls `resolve_usage(self.source, app_lib_index)`.
  But `Node` doesn't know about `App` — see ## Build order step 2 for the wiring detail.
- `shaderbox/ui.py` — mtime watcher continues to iterate `compile_unit.sources` (no change from
  feature 014's step 3); add a `lib_root` mtime check to invalidate the cached `LibIndex` when
  any lib file changes.
- `shaderbox/app.py` — owns the cached `LibIndex`; rebuilds it on lib-mtime change. New helper:
  `App.open_lib_file(path)` to create / focus a lib editor session.
- `shaderbox/paths.py` — `lib_root()` already exists from earlier step 2. Unchanged.
- **NEW: `shaderbox/popups/lib_picker.py`** — fuzzy-search modal.
- `shaderbox/hotkeys.py` — Ctrl+P binding opens the picker.
- **NEW or REFACTORED: `shaderbox/tabs/code_pane.py`** — tab bar + active editor. Today's
  `tabs/code.py` body becomes the "draw editor for active session" core.
- `shaderbox/tabs/code.py` — becomes a thin shim that calls into `code_pane.py`, OR is replaced.
- **REWRITE: `tests/test_shader_include.py`** — rename to `tests/test_lib_index.py` +
  `tests/test_lib_resolver.py`. Drop all `#include`-syntax tests. Add: function extraction,
  comment stripping, doc extraction, usage detection, transitive close, user-shadowing, topo
  sort, preamble insertion after `#version`, error remap via SourceMap.
- `ai_docs/conventions.md ## Known quirks` — add the `#line` integer-only footgun.
- `ai_docs/todo.md` — entries for any deferrals pulled out of scope.

## Build order

Sequential, each step leaves `make check + make smoke + make test` green:

1. **Spec amendment** (THIS commit) — replace the `#include` design with the auto-resolve
   design. No code changes yet.
2. **`LibIndex` + `resolve_usage` as pure functions, fully unit-tested.** Throw out the old
   `shader_include.py` + its tests. `Node.compile()` continues using the OLD path until step 3.
3. **Wire `LibIndex` into `Node.compile()`.** The clean way: `App` owns the `LibIndex` (rebuilt on
   lib-mtime change); `App` passes it into `Node.compile()` somehow. But `Node` is constructed
   inside `core.py` and called from many places... cleanest solution: `Node.compile()` calls a
   module-level `_active_lib_index()` accessor that the `App` populates. Or: `Node` carries a
   `lib_index` reference set by `App` after load. Decide during implementation; the goal is "no
   import cycle, no global state, and `Node.compile()` doesn't construct the index itself."
4. **Lib-mtime invalidation of the `LibIndex`.** Extend the watcher in `ui.py` to detect
   `lib_root` changes and rebuild the index. Touching a lib file rebuilds the index AND fans out
   to every node whose `compile_unit.sources` contains the path.
5. **Picker UI (`Ctrl+P`).** `popups/lib_picker.py`. Fuzzy match against `LibIndex.functions`.
   Filter `^SB_`. Insert-at-caret + open-file actions.
6. **Multi-file editor UI (tab bar).** `tabs/code_pane.py`. Pinned node-shader tab + N closable
   lib tabs.
7. **Polish + docs + commit.** `conventions.md ## Known quirks` + `todo.md` deferrals +
   `roadmap.md` banner. Single commit on `dev`.

## Verification

- `make check` (ruff + pyright) green, 0 errors.
- `make smoke` green (200 headless frames, popup-mutex + `current_node_id` invariants).
- `make test` green.
- **Manual sanity in the running app:**
  1. With an empty lib root, app behaves exactly as before.
  2. Author `<app_data_dir>/lib/hash.glsl` with `/// returns a pseudorandom float\nfloat
     SB_hash21(vec2 p) { ... }` (no `#include` in any shader); call `SB_hash21(uv)` from a node
     shader — compiles cleanly, render shows the helper's effect.
  3. Introduce a typo in `lib/hash.glsl`'s body, save — node's error strip lights up, line
     number points at the right line in `hash.glsl`. Click → editor switches to the lib file
     with the caret on the bad line.
  4. Add a second function `SB_fbm` in `lib/noise.glsl` that calls `SB_hash21`; call `SB_fbm`
     from a shader — both `SB_fbm` AND its dep `SB_hash21` are prepended automatically.
  5. Define `SB_hash21` locally in the user's shader — the lib version is suppressed; the
     user's version is the one that compiles.
  6. `Ctrl+P` opens the picker; type "hash" → `SB_hash21` matches; Enter inserts the name at
     the caret; Ctrl+Enter opens `lib/hash.glsl` at the function's line.
  7. Edit a lib file externally (in `vim` or VS Code) → mtime watcher fans out, every node that
     transitively uses any function from that file recompiles.
  8. Save with a clean shader — "compiled" cue fires.

## Open questions — RESOLVED (locked 2026-05-28, all leans accepted)

1. **`SB_` prefix policy**: strict at picker (anything not `SB_`-prefixed is invisible). Engine
   doesn't enforce — non-prefixed functions are private helpers, callable inside the lib.
2. **`///` docstring above functions**: optional convention. Missing docstring → picker shows
   the signature as the description.
3. **Top-level lib uniforms / globals**: not auto-injected. Lib = pure functions. If a lib
   function references `u_time`, the user's shader must declare it.
4. **Preamble insertion point**: after `#version` and `#extension` lines, before the rest.
5. **Picker insert behavior**: paste just the function name. The user types parens themselves.

## Review history

(To be filled in when the implementation lands.)
