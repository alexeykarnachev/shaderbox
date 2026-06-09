# Feature 025 — ProjectSession extraction (headless project core)

Extract a headless **project core** — `ProjectSession` — out of `App`, so the whole copilot stack
(`CopilotBackend` + `CopilotSession` + `RevertExecutor`) can be constructed and driven WITHOUT
`App`/glfw/imgui. `App` becomes a thin UI wrapper that OWNS one `ProjectSession` and delegates every
project operation to it. This is the long-deferred `app.py` split (`todo.md ## split ui.py / app.py
further` — the headless-copilot need is the fresh pain signal that unblocks it) AND the foundation the
dogfood harness (feature 026) builds on. Pure refactor: `App` behavior is unchanged except ONE intended
delta (the mid-turn "Node saved" toast on the copilot path is dropped — decision 7).

This is a high-blast-radius refactor (same class as feature 023, which lifted the copilot backend out of
`app.py`). It lands as a sequence of green-gated commits; no behavior change is allowed to ride along
except the documented one.

## Goal

- A new `shaderbox/project_session.py` holding `ProjectSession`: the project-lifecycle + copilot core,
  with NO imgui/glfw imports (the headless invariant — assert it in review, the way
  `copilot/capabilities.py` is App-free by contract).
- `App` constructs one `self.session = ProjectSession(...)` and forwards project state/ops to it via
  explicit `@property` accessors (NOT `__getattr__` — pyright must see the surface).
- The copilot capability wiring (`_build_copilot_capabilities` + `CopilotBackend` + `RevertExecutor` +
  `CopilotSession`) moves INTO `ProjectSession`; its injected getters/callbacks re-root from `App` to
  `self`.
- UI side effects of project mutations (notification toast, editor-session rehydration, delete-arm
  clear, sticky-focus reset) flow back to `App` through **injected callbacks the core invokes** — the
  `ShaderLibFileManager` idiom (`shader_lib/file_ops.py` `on_paths_removed`/`on_path_renamed`), applied a
  4th time. The harness passes no-op defaults.

## Why callbacks (the load-bearing decision)

The seam MUST be callbacks-the-core-invokes, not return-values and not "UI reads core each frame",
because **all four UI-tail methods fire from inside a copilot turn, on the worker-marshalled main-thread
drain path, with `App` NOT on the call stack** (verified, not recalled):

- `sync_editor_from_disk` ← `backend.py::_copilot_persist_shader` on every `apply_shader_edit` /
  `apply_line_edit` (backend.py:1146).
- `save_ui_node` ← `backend.py::create_node` (backend.py:684).
- `set_current_node_id` ← `create_node` (690), `switch_node` (738), and `delete_node`'s teardown via
  `_delete_node_unguarded` (app.py:1191).
- `delete_node_unguarded` ← `backend.py::delete_node` (713).

`CopilotBridge.drain()` runs the op closure on the thread that calls `drain()` (bridge.py:67-96) — in-app
that's `ui.py:88 drain_bridge()` on the render thread, frames after `App` returned from
`enqueue_turn()`. A return value can't reach an `App` that isn't on the stack. `RevertExecutor.restore_checkpoint`
adds a second path that reaches three of the tails from deep inside one call (revert.py:55,74,111,130).
So the only mechanism correct for every tail is one where the **core method itself invokes the reaction**:
an injected callback. A return value is correct only for the purely-App-synchronous callers (App.save,
App.delete_node) — and there it's redundant, because those already route through an App wrapper that also
serves the worker path. One mechanism (callbacks) handles both timings; result-objects / per-frame
reconcile only re-cover cases the callback already covers. (`sync_editor_from_disk` also CAN'T be a
per-frame reconcile: the mtime watcher must PUSH new disk text into a specific `EditorSession.editor`, and
a blind reconcile would clobber unsaved in-editor typing — `is_current_editor_dirty`.)

## Design decisions

1. **`ProjectSession` is the headless project core; `App` is a thin UI wrapper owning one.** Cut line:
   everything imgui-bound or windowing stays in `App`; everything project-state / file-I/O / copilot moves
   to the core. The core imports no imgui/glfw (review-asserted invariant).

2. **Constructor** (keyword-only):
   ```
   ProjectSession(*, project_dir: Path, node_templates_dir: Path, starter_template_id: str,
       notifier: Notifications,
       on_node_source_synced: Callable[[str, str], None] = _noop,
       on_node_saved: Callable[[UINode, Path], None] = _noop,
       on_node_deleted: Callable[[str, Path], None] = _noop,
       on_current_node_changed: Callable[[str, str], None] = _noop) -> None
   ```
   No GL-context parameter — the documented precondition is "a moderngl context is ALREADY current on the
   constructing thread" (Node/Canvas do `self._gl = gl or moderngl.get_context()`, core.py:65,111). The
   CALLER makes it current first: `App` via `glfw.make_context_current` (app.py:161) before it builds the
   session; the harness via `create_standalone_context(backend='egl')` before it constructs
   `ProjectSession`. `notifier` is INJECTED, never constructed by the core (`Notifications` imports imgui,
   notifications.py:4) — `App` passes its real `self.notifications`, the harness a no-op stub.
   **Only `project_dir` + `node_templates_dir` + `starter_template_id` + `notifier` + the callbacks are
   constructor params; everything else is LOADED from disk by `session._load(project_dir)`** — `paths`
   (`ProjectPaths.for_root`), `ui_nodes`/`ui_node_templates` (`load_nodes_from_dir`), `app_state`
   (`UIAppState.load_and_migrate`), `shader_lib_index` (`rebuild`), the `shader_lib_*` cross-project stores
   (their own `.load()`), `template_descriptions`, `exporter_registry` (constructed + registered), and the
   `CopilotSession`/`CopilotBackend`/`RevertExecutor` cluster. `integrations_store` is loaded too (its key/
   model feed the OpenRouter client getters). A project switch re-runs `session._load(new_dir)` (mirrors
   `App._init` today).

3. **The four UI-tail splits (all via the callback seam):**
   - `set_current_node_id`: core sets `app_state.current_node_id`, fires `on_current_node_changed(old, new)`
     on the `old != new` edge. App's handler clears `editor_was_ever_focused` (app.py:779). The callback
     is bound on the CORE method itself (not an App wrapper bound into the backend) so a migration binding
     swap can't bypass it.
   - `save_ui_node`: the backend injection signature is UNCHANGED — `save_ui_node: Callable[[UINode], …]`,
     called `self._save_ui_node(new_node)` (backend.py:684, no path arg). The CORE method `save_ui_node`
     does `ui_node.save(...)` + `logger.info`, returns the dir, AND fires `on_node_saved(ui_node, dir)`
     (the callback carries the dir the core just computed; the backend caller ignores the return as today).
     The toast is dropped on the copilot path by App NOT BINDING `on_node_saved` to a toast there — App
     toasts only from its user-initiated save path, not the copilot path's handler (decision 7). The core
     never calls `notifier.push` itself.
   - `sync_editor_from_disk`: NOTHING stays in the core (the body is all `editor_sessions`/`TextEditor`).
     The core's `_on_node_source_synced(node_id, source)` is JUST the `on_node_source_synced` callback
     invocation, fired from `_copilot_persist_shader`'s tail and `RevertExecutor`'s in-place reload. App's
     handler IS the current app.py:995-1005 body. (This is the tail that proves the seam must be a callback
     — a per-frame reconcile here would clobber unsaved in-editor typing, `is_current_editor_dirty`.)
   - `_delete_node_unguarded`: the CORE method does the teardown (reselect via `set_current_node_id`,
     `ui_nodes.pop().node.release()`, working-set remove, `shutil.move` to trash) and fires
     `on_node_deleted(node_id, source_path)` at its tail. The backend injection
     `delete_node_unguarded: Callable[[str], str]` (returns trash-name) is unchanged — `on_node_deleted` is
     a NEW callback the CORE method fires (not the backend); the backend still just gets its trash-name
     back. **Capture `source_path` BEFORE the pop** (app.py:1183 already does — it's read at 1183, popped
     at 1184; lost after, and App's `editor_sessions` is path-keyed). App's handler pops
     `editor_sessions[path]` + clears `node_delete_armed` if it matches.

4. **The copilot cluster moves into the core.** `_build_copilot_capabilities` body + `CopilotBackend` +
   `RevertExecutor` + `CopilotSession` construction all live on `ProjectSession`; every getter/callback
   re-roots `self.X`. The OpenRouter client getters move with `integrations_store` (core). `App.copilot`
   becomes `self.session.copilot`. `App` keeps only copilot UI state + the per-frame `drain_bridge` /
   `pump_events` calls + the chat draw. `revert_turn`/`recover_deleted_node` stay App-side thin wrappers
   (they add `notifications.push` + `save_conversation`) delegating to `session.revert_executor` — the
   split `revert.py`'s docstring already documents.

5. **`App` forwards via explicit `@property`, not `__getattr__`.** ~30 forwarders (`App.ui_nodes` →
   `self.session.ui_nodes`, `App.paths`, `App.current_node_id`, …) so the hundreds of `widgets/`/`tabs/`/
   `popups/`/watcher call sites keep working unchanged AND pyright verifies the surface. `__getattr__`
   would hide the surface from the type checker (a convention-collision, not allowed).

6. **`share_tab_state` STAYS in `App`** (it's the share-tab UI preview; `tabs/share.py` owns its mutation).
   The copilot render path calls the FREE function `share_state.render_to(node, preset, duration, out)`
   (backend.py:768,794,859) which needs NO `TabState` — so the core renders to files without it. (Verified
   this session: `render_to` signature is `(Node, RenderPreset, float, Path)`, TabState-free.) This keeps
   the core narrower than one design proposed.

7. **One intended behavior delta: drop the mid-turn "Node saved" toast on the copilot path.** Today
   worker-driven `create_node` fires `App.save_ui_node`'s `notifications.push` inside the `_on_main`
   closure mid-turn — a spurious toast (the chat already reports node creation). After the split,
   `on_node_saved` is bound to App's toast only where App wants it; the user-initiated save still toasts
   unchanged. Maintainer-approved.

8. **`preview_canvas` stays in `App`, built right after `session._load`.** It's a live-GL UI preview the
   frame loop drives (the core renders to files via `Node` directly, never a preview canvas). App._init
   constructs it after `session._load` has warmed the nodes, on the still-current glfw context (app.py:161)
   — preserving today's ordering.

9. **`flush_current_editor` stays App-side, called BEFORE a core save.** `App.save` splits into
   `flush_current_editor()` (App, reads the live `TextEditor`) THEN `session.save_project()` — so a dirty
   in-editor change isn't lost on a copilot-triggered save. The core's `save_project` does NOT own the
   flush.

10. **Re-entrancy is acceptable as-is.** A core delete fires BOTH `on_node_deleted` AND (via the internal
    `set_current_node_id`) `on_current_node_changed` in one call. App's two handlers touch disjoint attrs
    (`editor_sessions`/`node_delete_armed` vs `editor_was_ever_focused`) so they're order-independent —
    matching what the single method does today, just split. No handler re-enters the session.

## Migration order (each commit green by `make check` + `make smoke`)

- **C1** — introduce `project_session.py` + `ProjectSession` holding ONLY zero-tail pure-core state
  (paths, ui_nodes, templates, app_state, integrations_store, exporter_registry, shader_lib_* stores/index,
  template_descriptions, _copilot_working_set) + pure methods (`template_description`, `_copilot_ws_add`,
  `rebuild_shader_lib_index`, `_seed_starter_node`). App constructs `self.session` + forwards via explicit
  `@property`.
- **C2** — move the `_init` project-load body + `release` teardown (MINUS `preview_canvas`) into
  `session._load`/`release`. App keeps `preview_canvas` + glfw/imgui bring-up, builds the canvas after
  `session._load`.
- **C3** — move the copilot cluster (`_build_copilot_capabilities` body + `CopilotSession` +
  `CopilotBackend` + `RevertExecutor`) into the core, re-rooting getters to `self`. The 4 UI-tail methods
  move INTO the core **whole, UI-tail included** (the notification/editor-session/delete-arm lines copied
  verbatim from App) — so `CopilotBackend`'s injections (`save_ui_node`/`sync_editor_from_disk`/
  `set_current_node_id`/`delete_node_unguarded`, all unchanged single-callable signatures) bind to real
  working core methods and C3 stays green with NO behavior change. To keep the core import-clean at C3 the
  UI-tail lines that touch imgui-bound state (the `notifications`/`editor_sessions` references) route
  through the injected `notifier` + a temporary `app.editor_sessions` ref the core holds; C4 removes that
  coupling. (C3 does NOT yet split — it relocates. The split is C4. This ordering is deliberate: relocate-
  then-split keeps every commit a pure move or a pure behavior-preserving split.)
  `App.revert_turn`/`recover_deleted_node` delegate to `session.revert_executor`.
- **C4** — split the 4 UI-tail methods: cut the UI-tail OUT of each core method and replace it with an
  `on_*` callback the core fires; App binds its handlers (the cut lines move to App's handler bodies) +
  the `ShaderLibFileManager` removed/renamed callbacks. The copilot path simply does NOT bind
  `on_node_saved` to a toast → the mid-turn "Node saved" toast is dropped (decision 7). After C4 the core
  holds no imgui-bound ref. Green: behavior identical except the dropped toast.
- **C6** — `/sanitize`: conventions.md `## Design decisions` gains the ProjectSession bullet (+ the
  "callbacks over return-values because of the worker-mid-turn timing" rationale + revisit trigger); flip
  `todo.md` `split ui.py/app.py` + `headless GL`; roadmap banner + row.

(Feature 026 — the dogfood harness + `ProjectSession.pump_until_idle()` — is its OWN feature, NOT a commit
in this sequence. 025 covers C1–C4 + C6; 026 owns everything headless-harness.)

## Files touched

- **`shaderbox/project_session.py`** (new) — `ProjectSession`.
- **`shaderbox/app.py`** — sheds the project/copilot cluster; gains `self.session` + ~30 `@property`
  forwarders + the 4 callback handlers; keeps all UI/glfw/editor/preview state.
- **`shaderbox/copilot/`** — NO engine change; the getters/callbacks just bind to `self` (core) instead of
  `App`. (`backend.py`/`revert.py`/`session.py` already App-free.)
- **`tests/`** — `conftest.py`'s `app` fixture + the copilot test helpers may simplify (construct a
  `ProjectSession` directly where they don't need the UI), but existing tests must stay green through every
  commit (smoke still drives a full App through the forwarders).
- **`ai_docs/conventions.md` / `todo.md` / `roadmap.md`** — at C6.

## Manual verification

- `make check` + `make smoke` green after EVERY commit (C1–C4).
- A live `make run` pass on a display box: create/switch/delete nodes, save, run a copilot turn
  (edit + create + delete), hit Revert — confirm notifications, editor rehydration, delete-arm, and the
  Revert glyph all behave as before, EXCEPT no mid-turn "Node saved" toast during a copilot create.
- Headless construction smoke (the C5/026 payoff, but provable at C3): on the Pi, set the MESA + DATA_DIR
  env, make an EGL context current, construct `ProjectSession(project_dir=tmp, notifier=<noop>)`, confirm
  it builds with no glfw and a copilot turn can run a compile-only tool.

## Out of scope

- **The dogfood harness + `pump_until_idle()`** — feature 026, its own feature (not a commit here). 025
  ends at C4 + C6 when the core is extracted + green. Trigger: 025 lands → start 026.
- **Further `app.py` splits** (node-CRUD, path-properties, picker forwards) — still net-negative per
  `todo.md`; this refactor lifts the project/copilot tenant only, the genuine remaining one. Trigger: a
  fresh pain signal (lost search-and-replace / unclear blast radius), per the `todo.md` entry.
- **Any copilot ENGINE change** — the engine is consumed as-is. Trigger: a dogfood scenario (026) surfaces
  an engine bug worth its own fix.
- **`share_tab_state` into the core** — stays in App (decision 6). Trigger: any change that gives
  `share_state.render_to` a `TabState` dependency (re-check decision 6's TabState-free premise).

## Review history

(Pre-impl design: a 4-agent ultracode workflow — 3 independent boundary designs [callbacks-idiom /
ui-reads-core / return-value-driven] + a deciding architect. The architect's load-bearing finding: ALL
four UI-tail methods fire mid-copilot-turn on the worker-drain thread with App off the stack, which
structurally rules out return-value seams and per-frame reconcile, leaving callbacks-the-core-invokes as
the only correct single mechanism — the `ShaderLibFileManager` idiom a 4th time. Grafted corrections: bind
the callback on the core method (not an App wrapper) so a migration binding-swap can't bypass it; capture
the deleted node's source.path before the pop; drop the mid-turn toast rather than keep it via the injected
notifier. The `share_tab_state` open question was resolved AGAINST the core this session: `render_to` is
TabState-free, so the share preview stays in App.

A surface-level pre-impl review ran 2 adversarial agents (one code-anchored, one consistency). Code-anchored:
PASS — every cited line number + call path verified accurate against the source (the 4 UI-tail call sites in
backend.py, the bridge thread semantics, revert.py's reach, render_to's signature, the 14 init lines).
Consistency: PARTIAL → resolved. Real findings fixed in this spec: the C3/C4 ordering was clarified
(relocate-whole-then-split, NOT a swap — backend injections keep their unchanged single-callable signatures
and bind to whole core methods at C3); the `save_ui_node` callback arity (backend injection stays `(UINode)`,
core fires `on_node_saved(ui_node, dir)`); how the toast is dropped (App doesn't bind the copilot-path
handler, core never calls notifier); who fires `on_node_deleted` (the core delete method, not the backend);
what's constructor-injected vs `_load`-from-disk; the GL-context-current-before-construction precondition is
the caller's job; and the 025/026 boundary (026 owns the harness + `pump_until_idle`, not a commit here).
Rejected as over-specification: listing the architect's 5 questions verbatim, and pinning the internal
callback-firing ORDER in a delete (an impl detail — the handlers touch disjoint attrs, decision 10). Pre-impl spec-fidelity review TBD before plan-lock if
requested.)
