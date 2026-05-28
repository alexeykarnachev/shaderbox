# 014 — CompileUnit refactor (prep for shader include library)

Pure shape refactor — **zero behavior change for the end user**. Introduces four engine/UI seams
that the upcoming shader-include feature (`015_shader_include_library.md`, not yet written) will
plug into without further cross-cutting churn. Today's hardcoded "one file per node" assumptions
are pulled into named types so multi-file (root + N includes) is later additive, not invasive.

This spec is a **scaffolding feature**. Done well, the user notices nothing on ship; the next
feature lands twice as cheap.

## Goal

Replace today's scattered shader-source state (a `str` on `Node`, a `float` on `UINode`, a raw
error string, frame-recomputed parsed errors, two parallel `dict[str, _]` editor maps on `App`,
five transient `editor_*` fields) with four named types that own the same data:

1. `ShaderSource` — `(path, text, mtime)` value type. Replaces `Node.fs_source: str` +
   `UINode.mtime: float` + the hardcoded `"shader.frag.glsl"` basename.
2. `CompileUnit` — `(sources, flattened, source_map, program, error_raw, errors)`. Replaces
   `Node.shader_error: str` + the per-frame `parse_shader_errors(...)` calls in `tabs/code.py` and
   `hotkeys.py`. Errors are parsed **once** per compile, not every frame.
3. `SourceMap` — `(driver_line) -> (path, source_line)`. Identity-mapped today (single source);
   the include feature populates it during flatten. `ShaderError` gains a `path: Path` field so
   downstream consumers thread file identity through unchanged.
4. `EditorSession` — `(editor, source, saved_undo)` keyed by **path**, not by `node_id`, on
   `App.editor_sessions: dict[Path, EditorSession]`. Eliminates the parallel-dict bug surface and
   makes "edit a non-node file" (lib file) a non-decision later.

The seams compose: `Node.compile()` reads its `ShaderSource`, builds a `CompileUnit`, caches it.
The editor session reads the same `ShaderSource`. The mtime watcher walks `compile_unit.sources`
(a list — today of length 1, tomorrow N). Error UI, F8 hotkey, error strip, gutter markers, and
the uniform-jump path all read `compile_unit.errors` (already parsed, already path-tagged) and
gate on `error.path == current_editor_path` before placing the caret/marker.

## Non-goals — this feature does NOT

- **Add `#include` support.** No resolver, no `#line` injection, no lib directory, no search path,
  no multi-file flatten. `CompileUnit.sources` is always `[root]`. `SourceMap` is always identity.
  That all lands in feature 015.
- **Change the UI.** No tabs, no file picker, no split pane. The code tab draws exactly one editor
  (the current node's), exactly as it does today. `EditorSession` keyed by path is the
  preparation; using more than one key is feature 015.
- **Change on-disk state.** `Node`'s serialized form (`node.json` + sibling `shader.frag.glsl`) is
  unchanged. No `app_state.json` migration. No version bump on ship (this lands as a `patch` —
  no breaking change to users' projects).
- **Add new behavior tests.** The 4 `parse_shader_errors` tests + 7 `find_uniform_declaration_line`
  tests + 2 `next_error_line` tests grow only as needed to thread the new arg shapes (e.g.
  `SourceMap.identity()` as an arg) — no new assertions. **A new behavior assertion in this
  feature is a smell**; defer it to feature 015 along with the behavior change that motivates it.
- **Re-architect the editor.** `TextEditor` stays the same `imgui_color_text_edit` binding, lazy
  per-key, write-locked palette, FPE-while-modal caveat (`conventions.md ## Known quirks`).
  `EditorSession` is a structural wrapper, not a behavioral one.
- **Remove `editor_jump_request` / `editor_hover_line` / etc.** They grow a `path` field
  (`JumpRequest(path, line, index)` etc.); they don't get replaced.

## Design decisions (locked)

1. **`ShaderSource` is a frozen dataclass with three fields: `path: Path`, `text: str`,
   `mtime: float`.** The `text` is mutable-by-replacement (new value type instance on edit), not
   mutated in place. Constructed by one factory: `ShaderSource.load(path) -> ShaderSource`. The
   `"shader.frag.glsl"` basename constant moves from `core.py:111` into a single private
   `_NODE_SHADER_BASENAME = "shader.frag.glsl"` at the top of `core.py`; no other module knows it.
   `DEFAULT_FS_FILE_PATH` in `constants.py` keeps its role (default content for new nodes), but
   stops being a `Path`-vs-`str` mixed type — load it via the same factory.

2. **`CompileUnit` is owned by `Node`, cached.** Replaces `Node.shader_error: str` (deleted) plus
   the frame-local `parse_shader_errors(...)` calls in `tabs/code.py:101` and `hotkeys.py:12`
   (those become reads of `node.compile_unit.errors`). Shape:
   ```python
   @dataclass
   class CompileUnit:
       sources: list[ShaderSource]   # [root] today; list to make 015 additive
       flattened: str                # what's handed to ctx.program; == sources[0].text today
       source_map: SourceMap         # identity today
       error_raw: str                # raw driver string (debug + the source for parse)
       errors: list[ShaderError]     # parsed once, path-tagged
   ```
   Built by `Node.compile()` (extracted from the inline block in `Node.render()`). On successful
   compile `error_raw == ""` and `errors == []`. On compile failure, `Node.program` stays at the
   previous valid program (feature-013 invariant: last-good render stays bright) and `error_raw` +
   `errors` populate. The `Node.shader_error` attribute is deleted; consumers move to
   `node.compile_unit.error_raw` (rare — only the error strip's "show raw fallback when no regex
   matches" path) or `node.compile_unit.errors` (the normal path).

   **Landed deviation:** `program: moderngl.Program | None` is NOT on `CompileUnit`; it stays on
   `Node` (where ~5 sites already read it — `get_active_uniforms`, the render path, the release
   path, etc.). The compile unit describes *what was compiled* + *its diagnostics*; the GL
   resource itself is a `Node` concern. Moving it added churn for zero behavior benefit.

3. **`SourceMap` is a pure-data type, identity by default.** Shape:
   ```python
   @dataclass(frozen=True)
   class SourceMap:
       entries: tuple[tuple[int, Path, int], ...]  # (driver_line, path, source_line), sorted
       def resolve(self, driver_line: int) -> tuple[Path, int]: ...
       @classmethod
       def identity(cls, source: ShaderSource) -> "SourceMap": ...
   ```
   `identity()` returns a map where `resolve(N) == (source.path, N)` for all N. The include
   feature later replaces `identity()` with a real map produced during flatten; **no consumer
   changes**. `parse_shader_errors` grows a `source_map: SourceMap` arg and returns
   `ShaderError(path=..., line=...)`. The "1→0 line shift" stays inside `parse_shader_errors`,
   applied AFTER `source_map.resolve` (driver speaks 1-based; resolve maps 1-based driver line to
   1-based source line; the −1 happens on the source-line side at the return boundary).

4. **`ShaderError` grows a `path: Path` field; line stays 0-based.** Shape:
   ```python
   @dataclass(frozen=True)
   class ShaderError:
       path: Path
       line: int            # 0-based, already remapped via SourceMap
       message: str
   ```
   The fallback case (regex doesn't match) becomes `ShaderError(path=root.path, line=-1,
   message=raw)` — `path` defaults to the root source, since the driver gave us no file context.

5. **`EditorSession` is keyed by path on `App`.** Shape:
   ```python
   @dataclass
   class EditorSession:
       editor: TextEditor
       source: ShaderSource
       saved_undo: int          # dirty baseline (undo-index at last save)
   ```
   `App.editors: dict[str, TextEditor]` (at `app.py:151`) and `App.editor_saved_undo: dict[str,
   int]` (at `app.py:152`) collapse into `App.editor_sessions: dict[Path, EditorSession]`. The
   key is `source.path` — for a node shader today, that's `<project>/nodes/<uuid>/shader.frag.glsl`.
   `App.get_editor(node_id)` becomes `App.get_session(path) -> EditorSession`; the node-id
   convenience caller (`App.get_session_for_current_node()`) reads
   `node.compile_unit.sources[0].path`. Lazy creation is unchanged.

6. **Transient editor request types gain `path`.** Five fields:
   - `editor_jump_request: tuple[int, int] | None` → `editor_jump_request: JumpRequest | None`
     where `JumpRequest = (path: Path, line: int, column: int)`.
   - `editor_hover_line: int | None` → `editor_hover_line: HoverMark | None` where
     `HoverMark = (path: Path, line: int)`.
   - `editor_focused: bool` — unchanged (it's about the editor widget, not a file).
   - `editor_defocus_requested: bool` — unchanged.
   - `code_hovered_uniform: str` — unchanged (host-side regex, no driver line involvement).
   The consumer gate is uniform: a request is honored only if its `path == current_editor_path`
   (today, always true; tomorrow, the gate is what makes "error in lib file → don't move the
   node-shader caret" work). The set/clear pattern (`_consume_jump`) is unchanged.

7. **The mtime watcher walks `compile_unit.sources`.** The loop in `ui.py:63-79` changes from
   "stat `<project>/nodes/<id>/shader.frag.glsl`" to "for each `s` in
   `ui_node.node.compile_unit.sources`, stat `s.path`, compare to `s.mtime`." Today the list has
   length 1 — identical to current behavior; tomorrow it has length N with includes. The mtime
   field on `UINode` is **deleted** (lived twice — also on the `ShaderSource`).

8. **`Node.fs_source` is deleted; `Node.source: ShaderSource` replaces it.** Every read of
   `node.fs_source` becomes `node.source.text`; every write becomes
   `node.source = replace(node.source, text=new_text, mtime=time.time())` (then on disk by
   `UINode.save()` which writes `source.text` to `source.path` and refreshes `mtime`).
   `Node.release_program(new_fs_source: str)` keeps its `str` parameter for now (it's the editor's
   text, no path involved at the call site), but inside it builds the new `ShaderSource` via
   `replace(...)`. The path stability is the invariant: a node's `source.path` never changes after
   `load_from_dir`.

9. **No `app_state.json` schema change.** `node.json` continues to point at the sibling
   `shader.frag.glsl` by convention (the basename constant); nothing about the on-disk shape
   moves. Loading: `Node.load_from_dir` builds `ShaderSource.load(node_dir / _NODE_SHADER_BASENAME)`
   and assigns it. Saving (`UINode.save`): writes `node.source.text` to `node.source.path`. The
   generation counter in `ui_models.py` is untouched.

10. **No new `# type: ignore` markers; no new pyright errors.** The repo is at 0 pyright errors;
    keep it that way. `make check` blocks on failure (already enforced).

## The layering contract being preserved

The audit found a clean engine/UI split that this refactor must not muddy:

- **Engine side** (knows about files, source text, line numbers in source):
  `core.py` (Node, CompileUnit), `util.py` (parse, regexes, SourceMap), the four new types.
- **UI side** (knows about TextEditor instances, mouse clicks, gutter markers):
  `app.py` (EditorSession dict, transient requests), `tabs/code.py` (draw + apply markers),
  `widgets/uniform.py` (hover/click handlers), `hotkeys.py` (F8 glue).

Concretely:
- `parse_shader_errors` stays in `util.py`, pure, no imgui import.
- `EditorSession` stays in `app.py` (or a sibling `editor_session.py` if `app.py` grows), imgui
  side; never imported by `core.py` or `util.py`.
- `ShaderError.path` is the **engine's** vocabulary; the UI gate (`error.path ==
  current_editor_path`) is the **UI's** translation layer.

A draw function importing `ShaderSource` is fine. A `core.py` function importing `EditorSession`
is a layering bug — `core.py` doesn't know what an editor is.

## Files touched

(Indicative — confirmed against the touchpoint inventory.)

- `shaderbox/core.py` — introduce `ShaderSource`, `CompileUnit`, `_NODE_SHADER_BASENAME`; replace
  `Node.fs_source` with `Node.source`; extract `Node.compile()`; delete `Node.shader_error`.
- `shaderbox/util.py` — extend `ShaderError` (add `path`); extend `parse_shader_errors`
  (add `source_map` arg); add `SourceMap` (frozen dataclass + `.identity()` + `.resolve()`).
- `shaderbox/app.py` — `editor_sessions: dict[Path, EditorSession]` replaces the two parallel
  dicts; `JumpRequest` + `HoverMark` types; `get_session_for_current_node` helper; update the
  ~5 callers (`flush_current_editor`, `sync_editor_from_disk`, `get_editor`, etc.).
- `shaderbox/ui.py` — mtime watcher loops over `compile_unit.sources`.
- `shaderbox/ui_models.py` — `UINode.save` writes via `source.text`/`source.path`; delete
  `UINode.mtime` (lived on `ShaderSource` now).
- `shaderbox/tabs/code.py` — read `node.compile_unit.errors` instead of calling
  `parse_shader_errors` per frame; gate caret/marker apply on `err.path == current_editor_path`.
- `shaderbox/widgets/uniform.py` — `editor_jump_request` and `editor_hover_line` carry path now
  (current editor path, always); no behavior change.
- `shaderbox/hotkeys.py` — F8 reads `node.compile_unit.errors`; jump request carries path.
- `tests/test_util.py` — the existing 13 tests adapt to the new `SourceMap` arg + `ShaderError`
  shape; **no new tests** in this feature.

(File list is the working estimate; the actual diff is whatever it takes to land the seams with
`make check` + `make smoke` + tests green and zero behavior change.)

## Build order

Single wave, one commit (or two — see Open questions). The seams cross-cut: a half-applied
refactor leaves the editor talking to old types and the engine to new ones, which is the worst
state to land in. Order within the wave:

1. **`ShaderSource` + `_NODE_SHADER_BASENAME`** (`core.py`, `ui_models.py`, `ui.py`,
   `constants.py`). Smallest churn — every `node.fs_source` becomes `node.source.text`; every
   `ui_node.mtime` becomes `ui_node.node.source.mtime`.
2. **`SourceMap` + `ShaderError.path`** (`util.py`, tests). `SourceMap.identity()` everywhere a
   parse happens. Tests adapt.
3. **`CompileUnit`** (`core.py`, `tabs/code.py`, `hotkeys.py`). Extract `Node.compile()`; cache
   the unit; downstream switches from per-frame parse to `compile_unit.errors`.
4. **`EditorSession` + path-keyed transient requests** (`app.py`, `widgets/uniform.py`, the 5
   transient fields). Last because it touches the most UI sites.

Each step leaves `make check` + `make smoke` + tests green. If a step doesn't, stop and figure out
why — don't push through.

## Verification

- `make check` (ruff + pyright) green, zero new errors.
- `make smoke` green (200 headless frames, popup-mutex + `current_node_id` invariants).
- `make test` green (the existing 13 line-number tests, adapted to new arg shapes).
- **Manual sanity in the running app** (one short pass, handed to the maintainer):
  1. Open app, switch between 2+ nodes — editor text + uniforms + render all behave as today.
  2. Edit a node's shader, Ctrl+S — compile fires, error strip + gutter markers + F8 + click-to-jump
     + uniform hover + uniform-click-to-declaration all behave as today.
  3. Touch a node's `shader.frag.glsl` externally (`touch` or edit in a side editor) — mtime
     watcher fires, disk wins, editor text refreshes (existing behavior).
  4. Introduce a syntax error, save — last-good render stays bright (feature-013 invariant),
     errors populate the strip with the right lines.
  5. Save with a clean shader — "compiled" cue fires (feature-013).

If any of (1)-(5) changes user-visibly, the refactor has leaked behavior — fix before merging.

## Out of scope (for absolute clarity)

- Anything from the upcoming feature 015 (include resolver, `#line` injection, lib directory,
  multi-file UI, search path, file tabs, file picker, click-to-jump into a lib file, save-fanout
  to dependents). All of it lands on top of seams 1–4 once they're in.
- Refactoring `tabs/code.py`'s draw flow, the error strip layout, the uniform widget shape, or
  any other UI surface. The 4 seams reshape state, not rendering.
- Cleaning up the 5 transient `editor_*` fields into a single `EditorRequests` bag. Tempting, not
  load-bearing for feature 015. Defer if the field count grows.
- Anything in `todo.md` not explicitly named here.

## Open questions — RESOLVED (locked 2026-05-28, all leans accepted)

1. **Single commit on `dev`.** Cross-cutting nature means intermediate states are uglier than the
   final one; one atomic commit.
2. **`JumpRequest` / `HoverMark` as `@dataclass(frozen=True)`.** Consistent with `ShaderSource`,
   `SourceMap`, `ShaderError`.
3. **`Node.release_program(new_fs_source: str)` keeps `str` arg.** Builds the new `ShaderSource`
   inside the method via `replace(self.source, text=new_text, mtime=time.time())`.
4. **`SourceMap` lives in `util.py`.** Promote to its own module in feature 015 only if the
   include resolver brings enough volume to justify.

## Review history

(To be filled in when the implementation lands.)
