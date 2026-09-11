# Shared state between the main thread and a render thread (feature 090 research)

Scope: everything `Document.render` (and the per-document render set) touches today that UI code,
the copilot, exporters, or the script engine also touch — so feature 090 (moving `Document.render`
to a worker thread with its own GL context) knows exactly what needs a synchronization decision.
Every row is cited `file:line`. This is research: no design is chosen here except where the code
already embodies one (the copilot's existing worker/bridge split, its one GL-free carve-out).

---

## 1. The frame, phase by phase

Source: `shaderbox/ui.py`. Every phase below runs on the ONE thread that also owns the GL context
created in `App.__init__` (`shaderbox/app.py:219-228`, `moderngl.init_context()` bound to whatever
context is current at that call).

### `run()` — `ui.py:135-153`
The frame-rate governor: sleeps to hit `app.app_state.global_target_fps`, updates `app.global_fps`
(EMA). Calls `update_and_draw(app)` once per iteration. On loop exit: `app.save()`,
`app.save_imgui_ini()`, `app.shutdown()`.

### `update_and_draw()` — `ui.py:270-280`
Wraps `_update_and_draw` in `app.profiler.frame()` (`profiling.py` `Profiler.frame()`, the
per-frame CPU+GPU span root). After the frame: `app.last_profile = app.profiler.last_complete`;
feeds `app.profile_smoother` (feature 088).

### `_update_and_draw()` — `ui.py:282-539`, in order:

**Phase 0 — `_tick_frame_state(app)`, `ui.py:288-291` wrapping `ui.py:156-267`.** Everything
before any drawing:

0. **Consume `app.pending_project_switch` (084 D5)**, `ui.py:171-180`. `App.switch_project`
   (`app.py:1951`) releases EVERY GL object of the outgoing project (`App.release`,
   `app.py:1822-1876`) and reloads the incoming one wholesale (`session.load`,
   `project_session.py`). Deferred out of popup draw because a popup body runs AFTER the editor
   panel and the document image have already pushed textures into this frame's imgui draw list
   (`ui.py:168-170` comment).
1. **`maybe_rebuild_lib_index(app)`**, `ui.py:184` → `watch.py:65-88`. Detects a shader-lib file
   add/remove/mtime change; invalidates (`Pass.invalidate()`, `core.py:262-284`) every pass whose
   `compile_unit.sources` included a changed lib file — drops `Pass.program`, forces recompile.
2. **`app.copilot.drain_bridge()`**, `ui.py:189-192` → `copilot/bridge.py:69-88`. Runs up to 8
   queued `MainThreadOp` closures the copilot worker thread is blocked on (GL ops: compiling a
   pass for `read_shader`, `set_uniform`, `add_pass`, etc. — see §2). `try/except Exception` →
   `logger.exception`, never crashes the frame. EARLY so a freshly recompiled document renders
   this same frame (comment at `ui.py:187-188`).
3. **`app.session.sync_documents_from_disk()`**, `ui.py:198-199` → `project_session.py:550-603`.
   **Guarded**: `if not app.copilot.state.in_flight:` — skipped while a copilot turn is running,
   because the worker is mutating `documents/document.json` on its own thread and a sync here
   would race those writes (comment at `ui.py:195-197`). When it runs, it can **release and
   wholesale-replace** a live `Document` object (`_load_one_document_from_disk`,
   `project_session.py:605-622`, calls `old.document.release()` then
   `load_document_from_dir` — a fresh `Document`). This is today's one explicit
   render/mutation-race guard in the codebase and the closest existing precedent for what 090
   needs to generalize.
4. **Per-document shader-file-vanished check + `reload_document_if_changed`**, `ui.py:201-217` →
   `watch.py:12-62`. For every pass of every open document: compares
   `compile_unit.sources[i].mtime` against disk; on the root file changing, calls
   `render_pass.release_program(new_text)` (drops `Pass.program`, forces recompile) and pushes
   text into the open editor session (`app.sync_editor_from_disk`). **A vanished file for the
   current/output document aborts the WHOLE FRAME**: fires
   `app.copilot.bridge.run_deferred_render()` first (so a parked worker op doesn't stall, `ui.py:209`),
   pumps `glfw.poll_events()` (`ui.py:212`), clears `app.editor_key_events` (`ui.py:215`), and
   `_tick_frame_state` returns `None` — the caller (`ui.py:290-291`) returns immediately, skipping
   render AND draw entirely for this frame.
5. **`app.session.reload_scripts()`**, `ui.py:224` → `project_session.py:673-684`. Recompiles any
   document's `script.py` on mtime change (fresh instance — state resets on edit).
6. **Compute `tick_documents` (066 D2 render-set decision)**, `ui.py:232-258`. The set:
   `[current_document_id]` + (if "Render all" is on: every document with `first_render_done`
   True) + at most ONE not-yet-first-rendered document (admits one compile-cost first render per
   frame, budget-bounded so frame 0 never stalls on compiling everything — 066 D1/D2). Skipped
   entirely (only `[current_document_id]`) while any popup is open.
7. **`app.session.tick(tick_documents, now, dt, frame_idx, mouse=...)`**, `ui.py:259-260`, under
   `profiler.cpu("script")` → `project_session.py:686+` → `scripting/engine.py::ScriptEngine.tick`.
   Per document: writes `Pass.uniform_values[name]` for every uniform the document's `script.py`
   drives (`scripting/engine.py:649-707` `_write_one`). Reads `Pass.get_active_uniforms()`
   (`scripting/engine.py:635-647`), but only for passes whose `script_ready` is already True
   (never triggers a fresh compile itself — 066 D1 boundary respected).
8. **`document.begin_frame(app.frame_idx)`** for each `document_id` in `tick_documents`,
   `ui.py:264-265` → `document.py:363-390`. Advances feedback history (swaps `Pass.canvas` with
   its feedback `Canvas`, `document.py:392-404`) at most once per frame index — deliberately
   BEFORE render, and keyed by frame identity so a document rendered twice this frame (output +
   one pending off-chain pass) doesn't double-advance.

Returns `tick_documents` (or `None` to abort the frame).

**Phase 1 — `share_tab.update(app)`**, `ui.py:293-297`. Pre-imgui GL work for the CURRENT
document only. `try/except Exception` → `logger.exception`, does not crash the frame.

**Phase 2 — copilot event drain + file gate + turn-boundary bookkeeping**, `ui.py:299-310`.
`app.copilot.pump_events()` (worker→main `AgentEvent` queue drain into `ChatState`, no GL);
`_pump_file_gate(app)` (`ui.py:86-132`, serves a mid-turn `bind_media`/`import_document` file-pick
dialog, itself on the main thread); turn True→False transition seals the checkpoint + saves the
conversation; `app.copilot_turn_active = app.copilot.state.in_flight` is read this frame for the
UI-disable gate (phase 7 below).

**Phase 3 — `app.reconcile_popup_focus()`**, `ui.py:314`. imgui focus restore after a modal
closes. No Document/Pass state.

**Phase 4 — Render documents**, `ui.py:317-363`. **THE PHASE 090 MOVES.**
- If no popup open: for each `document_id` in `tick_documents` (from phase 0.6):
  `document.render(profiler=app.profiler)` under `profiler.cpu(f"document:{name}")`
  (`ui.py:320-326`). Then one more `document.render(target=pending, profiler=...)` for one
  never-drawn pass (`ui.py:328-341`, the off-chain first-render sweep — one pass per document per
  frame so a reopened multi-pass document's off-chain tiles fill in).
- `elif` Examples popup open (`ui.py:342-356`): same shape over `app.ui_document_examples`
  (separate dict from `app.ui_documents`; one first-render admitted per frame).
- `elif` Pass-settings popup open (`ui.py:357-363`): renders ONLY `app.current_document_id`'s
  document, kept live so a wiring/target change is visible behind the modal.
- Any OTHER open popup pauses ALL document rendering this frame.

**Phase 5 — `process_hotkeys(app)`**, `ui.py:367` → `hotkeys.py:27+`. `glfw.poll_events()` +
`imgui_renderer.process_inputs()`, PRE `new_frame`, outside any imgui frame. No Document state.

**Phase 6 — `imgui.new_frame()` + full UI draw**, `ui.py:371-500`, under `profiler.cpu("ui")`:
- `dispatch_commands(app)` (`ui.py:384` → `hotkeys.py:34+`) — registry-driven keyboard dispatch,
  IN-frame (runs AFTER phase 4's render calls this same frame; a hotkey-triggered mutation, e.g.
  `RESET_DOCUMENT`/`ADD_PASS`/pass-rename via `close_pass_settings`, lands on the NEXT frame's
  render).
- `_draw_menu_bar`, editor/app-panel split: `code_tab.draw(app)` (LEFT — reads/writes
  `Pass.source` via the editor's OWN buffer, not through `Document.render`), `_draw_copilot_bar`,
  `_draw_splitter`, `_draw_app_panel(app)` (RIGHT: `_draw_document_image` reads
  `render_pass.canvas.texture` for the live preview + FPS overlay + channel-view chip;
  `draw_document_preview_grid`; `_draw_document_settings` dispatching to
  `tabs/document.py`/`tabs/uniforms.py`/`tabs/render.py`/`tabs/share.py`) — wrapped in
  `imgui.begin_disabled(app.copilot_turn_active)` (`ui.py:450`) so panel edits are blocked mid
  copilot-turn, but NOT blocked mid-render (render has no "turn" concept today).
- Popups: `draw_examples`, `draw_help`, `draw_settings`, `draw_pass_settings`,
  `draw_emoji_picker`, `draw_lib_picker`, `draw_projects`, the command palette, notifications
  (`ui.py:463-478`).
- `cheatsheet.draw(app)`, `copilot_chat.draw(app)` — own top-level windows, drawn last so not
  obscured by the full-screen main window.
- Cursor apply (`glfw.set_cursor`, once per change, `ui.py:490-493`).
- "Rendering..." overlay drawn if `app.copilot.bridge.render_pending()` or
  `app.render_defer.has_request()` (`ui.py:497-499`) — the deferred-encode cue.
- `imgui.render()` (`ui.py:504`) — finalizes imgui draw data, no GL calls yet.

**Phase 7 — GL present**, `ui.py:506-517`, outside `profiler.cpu("ui")` but still inside
`profiler.frame()`:
- `glfw.make_context_current(app.window)`; `moderngl.get_context().clear_errors()` (`ui.py:507`
  — resets whatever `ctx.error` accumulated on the CURRENT context; see §4 for the implication
  for a second render-thread context).
- `gl.screen.use()`; `gl.clear()`.
- `app.imgui_renderer.render(imgui.get_draw_data())` under `profiler.gpu("ui:draw")`.
- `glfw.swap_buffers(app.window)` under `profiler.cpu("swap")`.

**Phase 8 — Post-swap deferred render encode**, `ui.py:519-537`. **THE OTHER PHASE 090 TOUCHES.**
Both `RenderDefer` (Render/Share-tab exports) and the copilot bridge's parked render op fire HERE,
after swap, so the "Rendering..." cue is provably on the glass before a synchronous encode freezes
the loop:
- `run_render_now = app.render_defer.ready_to_fire()` — a 2-frame latch (`render_defer.py`:
  submitted → shown next frame → fired the frame after).
- `bridge_render_pending = app.copilot.bridge.render_pending()`.
- If either: `gl.finish()` (forces the GPU to actually present the cue frame).
- If `run_render_now`: `request = app.render_defer.fire_and_clear(); request()` — this closure is
  `render_job.render_to`/`render_for` → `Document.render_media` → `Document.render`, SYNCHRONOUS
  on the main thread, blocks the frame loop for the whole export.
- `elif has_request()`: `mark_shown()` (hold one more frame for the cue).
- If `bridge_render_pending`: `app.copilot.bridge.run_deferred_render()` (`bridge.py:90-104`) —
  runs the parked `MainThreadOp.fn()`, which for `render_image`/`render_video`/`probe_render`/
  publish is again `Document.render_media`/`Document.render`, synchronous on main.
- `app.frame_idx += 1`.

---

## 2. The state ledger

Format: `state | writer (file:function, when) | reader (file:function, when) | today's ordering
guarantee | what breaks off-thread | candidate sync shape`.

### Uniform values

| state | writer | reader | today's guarantee | what breaks off-thread | sync shape |
|---|---|---|---|---|---|
| `Pass.uniform_values[name]` (panel edit) | `widgets/uniform.py:draw_ui_uniform:396-397` on user edit | `Pass.render` (`core.py:479`) every draw | single thread: an edit lands before the NEXT render call, since both are the same frame's sequential code | torn read (render reads half-written dict during a panel edit) or a value the render thread never sees until its next poll | per-frame snapshot handed to the render thread |
| `Pass.uniform_values[name]` (buffer branch, in-place `.write()`) | `widgets/uniform.py:draw_ui_uniform:249` on "Randomize" click | `Pass.render` binds the `moderngl.Buffer` (`core.py:483-485`) | same | a GL buffer write from the main thread while the render thread is mid-draw with that buffer bound — GL objects have thread affinity, this is a genuine GL-context violation, not just a data race | move the buffer mutation onto the render thread, or double-buffer |
| `Pass.uniform_values` (script engine) | `scripting/engine.py::_write_one` (`ui.py:260`, phase 0.7, before render) | `Pass.render` same frame | script tick always precedes render in frame order — 090's central invariant to preserve | script tick writing while render thread reads mid-frame: stale-vs-torn depending on how the value is copied | script tick must complete and its writes must be visible (snapshot/publish) before the render thread starts that document's frame |
| `Pass.uniform_values[name]` (copilot `set_uniform`) | `copilot/backend.py:1069-1070` `set_uniform._on_main`, via bridge, so already main-thread | `Pass.render` | bridge `drain()` runs early in phase 0 (before render), so a copilot-set uniform renders same frame | if render moves to its own thread, the bridge's main-thread write and the render thread's read need the same snapshot/queue discipline as a panel edit | same as panel-edit row — one mechanism should serve both |
| `Pass.uniform_values` (media/video swap) | `widgets/uniform.py:397` (Apply smoothing result) | `Pass.render` (calls `.update(render_time)` on a `MediaWithTexture`, `core.py:489`) | main-thread only, sequential | swapping a `Video`/`Image` object (which owns a GL texture + on some paths an open decoder) while the render thread holds a reference and is calling `.update()`/`.texture` on it | handoff must be atomic from the render thread's point of view — never free the old object while a render is using it |
| `Pass.uniform_values` (auto/table uniforms u_time, u_aspect, u_resolution, iteration/iterations) | `Pass.render` itself, every draw (`core.py:500-518`) | UI display only (`tabs/uniforms.py:_draw_auto_block:32`) | render-owned, UI is read-only display | none if UI stays read-only; the display would read a value from a prior frame | render-thread-owned; UI reads last-published snapshot |

### The clock and `u_time`

| state | writer | reader | today's guarantee | what breaks | sync shape |
|---|---|---|---|---|---|
| `Document.time_origin` | `Document.__init__`/`reset()` (`document.py:301`, `:406-415`) — user Reset action or document open | `Document.live_time()` (`document.py:417-423`), called from `Document.render()` when `u_time` is None | single thread, read-modify sequential | a Reset (user click) racing a render thread mid-frame reading the origin — torn read is impossible (float assignment) but a reset mid-render changes u_time non-atomically across the frame's passes if read per-pass rather than once | resolve `u_time` ONCE per document per frame (already the pattern — `render()` resolves it once at entry, `document.py:669-670`) and hand it to the render thread as part of the per-frame snapshot |
| `Document._frame` | `Document.begin_frame()` (`document.py:363-390`), phase 0.8, before render | `Document.render()` reads `self._frame` for the drawn-once skip (`document.py:689-693`) and feedback swap timing | begin_frame always precedes render in frame order | if render is async, `begin_frame` (main thread) could run again for frame N+1 before the render thread finishes frame N | frame-advance must be gated on the render thread's completion of the prior frame, or the two must agree on a frame-number handoff protocol |

### Pass iteration counters / drawn-once state

| state | writer | reader | today's guarantee | what breaks | sync shape |
|---|---|---|---|---|---|
| `Pass.drawn_frame` | `Document.render()` per-pass loop (`document.py:693`) | `Document.render()`'s own skip check (`document.py:686-692`) and `begin_frame`'s "did this pass draw last frame" check (`document.py:386-388`) | read-your-own-write within one `render()` call, and read again next `begin_frame` before the NEXT render | if `begin_frame` (main thread, phase 0) runs while the render thread hasn't finished writing `drawn_frame` for the prior frame, the feedback-swap decision uses a stale/torn value | `begin_frame` for frame N+1 must not run until the render thread has finished frame N's writes to `drawn_frame` — this is effectively a full frame-completion barrier |
| `Pass.first_render_done` | `Document.render()` (`document.py:694`) | `ui.py`'s render-set + first-render-sweep logic (`ui.py:238-258`, `:331-338`) — decides how many passes/documents get their first compile this frame | read by the SAME thread that will write it next frame, sequential | the render-set decision (066 D2's one-first-render-per-frame budget) is made on the main thread BEFORE dispatching to a render thread; if the render thread hasn't finished the prior first-render by the time the main thread computes next frame's set, the budget logic double-admits or stalls | the render-set decision (`ui.py` phase 0.6) needs to read a completed/published state, not one the render thread is mid-writing |
| `Document.first_render_done` | `Document.render()` (`document.py:667-668`) | same render-set logic (`ui.py:246,253`) | same | same class of hazard as `Pass.first_render_done` | same |

### Feedback histories + `feedback/<pass>.bin` persist

| state | writer | reader | today's guarantee | what breaks | sync shape |
|---|---|---|---|---|---|
| `Document._feedback[name]` (the history `Canvas`) | `Document._swap_feedback` (`document.py:392-404`), called from `begin_frame` (phase 0.8, before render) and mid-render for an iterated self-reading pass (`document.py:733-739`) | `Document.input_texture`/`_feedback_canvas`/`render()`'s per-pass input binding (`document.py:574-580,716-725`) | swap happens once per frame BEFORE that frame's render reads it; an iterated pass's mid-render swap is intra-call, same thread | a render thread reading `_feedback` while the main thread's `begin_frame` (next frame) is mid-swap — this is a genuine two-writer hazard if frame N+1's `begin_frame` runs before frame N's render thread finishes reading the feedback texture it's sampling from | `begin_frame`'s swap must be sequenced strictly after the render thread finishes the frame that read the pre-swap texture — a full frame barrier, not just a value handoff |
| `Document.newest_frame(name)` (the persisted "current" history frame) | derived (`document.py:518-534`), reads either the live `Pass.canvas` or the feedback `Canvas` depending on `drawn_frame` | `UIDocument.save()` → `canvas.texture.read()` (`ui_models.py:459-464`, the 089 D4 feedback persist) — a GL READBACK, called from `App.save()` (quit / `:w` / project switch) | single thread; save always reads a fully-drawn frame because render is synchronous and finished by the time save runs | a GL readback on the main thread racing a render-thread WRITE to the same texture (whichever canvas is "live" at that instant) is a textbook GL race — reading a texture mid-draw on another context, or reading one the other thread is about to reallocate via `set_size` | save must either run on the render thread, or be sequenced to only read a texture the render thread has finished with and handed off (e.g. after a frame-complete signal) |
| `Document._feedback_generation[name]` | `Pass.set_target()` bump (`core.py:255`) triggers `Document.drop_feedback` via generation mismatch (`document.py:547-550`) | `_feedback_canvas` (`document.py:547-550`) | sequential, single thread | a target-format change (main-thread panel edit, `popups/pass_settings.py` → `project_session.py:set_pass_target`) racing a render thread that's mid-draw into the OLD-format canvas | target reconfiguration (which releases + reallocates GL objects) must be sequenced against the render thread the same way any GL-object-lifetime change must be |

### Media/video textures advancing frames

| state | writer | reader | today's guarantee | what breaks | sync shape |
|---|---|---|---|---|---|
| `Video`/`Image` object identity in `Pass.uniform_values[name]` | main-thread bind (`widgets/uniform.py`, copilot `bind_media`/`bind_picked_media`) | `Pass.render()`'s `value.update(render_time)` / `value.texture` (`core.py:488-490`) — decodes the next video frame and uploads to a GL texture, ON WHATEVER THREAD CALLS `render()` | main-thread only today, so the OpenCV decode + GL upload both happen on the GL thread, sequentially with any rebind | `Video.update()`/`.texture` do a `cv2.VideoCapture` read AND a GL texture upload — moving `render()` to a worker thread moves BOTH onto that thread; a rebind (main thread swapping the `Video` object, e.g. via smoothing "Apply") must not free the old `Video` (closing its GL texture + decoder handle) while the render thread is mid-`update()` on it | media object handoff needs the same atomicity as any live-object swap under a render thread — old object must outlive any in-flight render call against it |

### Shader hot reload + recompile + compile-error surfaces

| state | writer | reader | today's guarantee | what breaks | sync shape |
|---|---|---|---|---|---|
| `Pass.source` (text/mtime) | `watch.py:_reload_pass_if_changed` (phase 0.4, disk mtime poll) via `release_program(new_text)`; also `copilot/backend.py::_copilot_persist_shader` (`:2396-2398`, via bridge → main thread); also editor's own buffer on `:w` (`hotkeys.py`) | `Pass.compile()` (`core.py:332`) | source change always happens in phase 0, strictly before phase 4's render | if render is on its own thread, a source-text write from watch/copilot/editor-save racing an in-progress compile-and-draw on the render thread is a genuine data race on `self.source` and triggers a recompile the render thread wasn't expecting mid-frame | source edits should be queued/published, consumed by the render thread at a frame boundary it controls, not written in place from another thread |
| `Pass.program` / `Pass.vbo` / `Pass.vao` (GL objects) | `Pass.compile()` (`core.py:332-414`), called from `Pass.render()` when falsy (`core.py:462-463`), from `Pass.get_active_uniforms()` (`core.py:319-320`), from `UIDocument.save()` (`ui_models.py:389-390`, main-thread pre-save compile), from copilot's `_on_main` closures (`backend.py:747-748` etc, bridged), from `add_pass`/`set_pass_target` (`project_session.py:917,1023`) | `Pass.render()`, `tabs/code.py` (error strip), `tabs/uniforms.py:71` (`get_active_uniforms()` — **lazily compiles**, every frame the Uniforms tab is open) | ALL of these run on the main thread today, so a "compile" from the panel (Uniforms tab merely being open) and a "compile" from the render loop never race — they're the same call sequenced by the same thread | this is the single sharpest hazard in the whole ledger: `tabs/uniforms.py:draw:71` calls `get_active_uniforms()` EVERY FRAME the tab is open, which can silently trigger `Pass.compile()` — releasing the OLD program/vbo/vao and creating new ones — from what would become the MAIN thread, while the render thread might be mid-draw with the OLD program bound. This is a GL-object lifetime race, not just a data race. | compile must become render-thread-owned exclusively; every other call site (`UIDocument.save`, `tabs/uniforms.py`, `add_pass`, copilot edits) must request a compile FROM the render thread rather than compiling in place |
| `Pass.compile_unit.errors` / `.error_raw` | `Pass.compile()` (`core.py:332-414`, all three failure branches write here, never raise) | `tabs/code.py` error strip (every frame, read-only), `widgets/pass_list.py:103` (tile error indicator), `widgets/document_grid.py:72` (grid tile border), copilot `backend.py:606-608,774,858` (both bridged AND, for `document_tree`, read DIRECTLY off the worker thread — see below) | single thread, read-after-write same frame or next | UI polling this every frame while a render thread is mid-compile (writing `compile_unit` piecewiece — it's reassigned wholesale as one `CompileUnit` object at each return, so no torn read of the dataclass itself, but a reader could see an old-but-consistent snapshot) | read-only consumers can tolerate a one-frame-stale published snapshot; the compile() call itself must move fully onto the render thread |
| `Document.passes` / `Pass.compile_unit.errors` / `Pass.source.text` (copilot's existing GL-free carve-out) | n/a (read-only here) | `copilot/backend.py::document_tree` (`:610-612`), `grep` (`:944-974`), `read_lib` — **run directly on the copilot worker thread, NOT via the bridge**, on the explicit argument these are "GL-free" plain-Python reads (comments at `backend.py:596,935,984`) | today this is safe because nothing else writes these fields from a third thread — only the main thread (render, watch, panel) writes them, and Python's GIL serializes the dict/attribute access enough that no crash results, though the values can be stale/inconsistent mid-write | once render is a genuine second thread also writing `Pass.compile_unit` (reassigning the whole object) and `Pass.source` concurrently with the copilot worker's unguarded read, this carve-out's safety argument ("no GL handle involved") no longer implies "no race" — a dict `Document.passes` being restructured (pass added/deleted) while iterated by `document_tree` was already only GIL-safe by luck, and adding a second heavy writer thread makes a torn iteration much likelier | this exact carve-out needs re-examination as part of 090 — either these reads move behind the same snapshot mechanism as everything else, or the "GL-free is safe off-thread" argument is explicitly re-justified against the new writer |

### Pass add/delete/rename

| state | writer | reader | today's guarantee | what breaks | sync shape |
|---|---|---|---|---|---|
| `Document.passes` (dict insert/pop) | `project_session.py:add_pass` (`:900-922`), `delete_pass` (`:926-941`), `rename_pass` (`:947-976`) — all main-thread (user click or bridged copilot tool) | `Document.render()`'s per-pass loop (`document.py:684-732`), `Document.effective_wiring()` (`document.py:582-609`) | pass mutations happen inside `dispatch_commands`/popup draw, AFTER phase 4's render this frame — so they land on the NEXT frame's render, never racing the current one | a delete's `Pass.release()` (GL texture/program/canvas release) happening on the main thread while the render thread is mid-iteration over `document.passes` for the SAME document (even from the prior frame, if the render thread hasn't finished) is a use-after-release | pass structural mutation (add/delete/rename) must be sequenced as a barrier against the render thread — never released while a render for that document is in flight |
| `Document.graph` (`PassGraph`, immutable value object via `with_passes`/`with_target`/`with_output`) | same call sites as above, plus `set_pass_target`/`set_pass_iterations`/`set_output_pass` (`project_session.py:1022,1035,978`) | `Document.render()` reads `self.graph.passes[name]` for target size + iteration count (`document.py:695-700,712-732`), `effective_wiring`/planner | graph is REPLACED wholesale each edit (immutable `model_copy`), so a reader never sees a torn graph — only a stale-but-internally-consistent one | this is actually the SAFEST kind of shared state here: immutable value replacement means the render thread can hold a reference to "the graph as of frame N" with no torn-read risk, only a staleness question (does it get frame N or N+1's graph) | false trail candidate — see §5; likely just needs "read the graph reference once at frame start" discipline, no lock |
| `Document.drop_feedback(name)` / `forget_pass_sources`/`rename_pass_sources` | `project_session.py:delete_pass:934,937`, `rename_pass:967,971,972` | `Document.render()` (via `effective_wiring`/feedback lookups) | same "lands after this frame's render" guarantee as pass add/delete above | `drop_feedback` releases a `Canvas` (GL) — same use-after-release class as pass delete if it races an in-flight render | same as pass structural mutation |

### Document switch + 084 D5 deferred project switch

| state | writer | reader | today's guarantee | what breaks | sync shape |
|---|---|---|---|---|---|
| `app.pending_project_switch` | `App.request_project_switch` (`app.py:1944-1949`) — set from popup draw (AFTER phase 4) | consumed in phase 0.0 of the NEXT frame (`ui.py:171-180`), strictly before that frame's render | one-frame latch, deliberately deferred past the current frame's draw | `switch_project` releases EVERY GL object in the outgoing project (`App.release`, `app.py:1822-1876`) — if a render thread is still mid-frame for the outgoing project when this fires, it's a mass use-after-release | the switch must wait for the render thread to be fully idle/quiesced for the outgoing project before releasing anything — the single highest-blast-radius synchronization point in the whole system |
| `app.ui_documents` (whole dict, wholesale replace) | `_init` → `session.load()` (project switch), `sync_documents_from_disk` (disk resync), `create_document_from_example`, `_delete_document_unguarded` | `ui.py` phase 4's render loop reads `app.ui_documents[document_id]` fresh each frame | single thread: whichever of these runs, it completes (including all GL releases) before phase 4 reads the dict that same frame | a render thread holding a `Document` reference from frame N while phase 0 of frame N+1 replaces/releases the whole dict is the same class of hazard as the pass-delete row, at project scope | the render thread must never hold a `Document` reference across a frame boundary without the main thread knowing it's still in use |

### The render-set decision (066 D2)

| state | writer | reader | today's guarantee | what breaks | sync shape |
|---|---|---|---|---|---|
| `tick_documents` (the computed per-frame render set: current doc + Render-all + one first-render) | computed fresh every frame, `ui.py:232-258` | consumed same frame by phase 0.7 (script tick) and phase 4 (render loop) | computed once, used twice, same frame, same thread — no staleness possible today | the computation reads `ui_document.document.first_render_done` (`ui.py:246,253`) to decide the one-first-render budget — if the render thread hasn't finished writing that flag from the PRIOR frame by the time this computation runs, the budget either double-admits a document or never admits a stalled one | this computation must read a value the render thread has FINISHED writing (frame-complete barrier), not one that might still be in flight |

### Exports (`glFinish`, encode)

| state | writer | reader | today's guarantee | what breaks | sync shape |
|---|---|---|---|---|---|
| `app.render_defer` (2-frame latch + the actual render closure) | `tabs/render.py::_run_render` / `tabs/share.py::_render` submit via `app.render_defer.submit(fn)` | fired in phase 8, main thread, synchronously calls `Document.render_media` → `Document.render` in a loop (per export frame) | today's export IS a `Document.render()` call, just deferred one frame and looped internally — same thread as the live render | the export loop (`document.py::_render_video`, N frames in a tight loop) is itself a heavy render-thread workload; if live preview render also moves to the render thread, the export and the live tick contend for the SAME thread/context — needs explicit sequencing (export could starve live preview for the whole export duration, which is already true today but would now also block/interleave with whatever queuing discipline 090 introduces) | export should route through whatever request queue 090 gives the render thread, with `glFinish` cue timing preserved from the main thread's point of view |
| `app.copilot.bridge` parked render op (`probe_render`/`render_image`/`render_video`/publish) | copilot worker calls `bridge.run_on_main(fn, defer=True)` (`bridge.py:52-67`) | `bridge.run_deferred_render()` in phase 8 (`bridge.py:90-104`), main thread | the EXISTING precedent for exactly this kind of worker→render handoff — a worker blocks, a closure is marshalled to the GL-owning thread, run, result returned | this pattern already generalizes almost directly to 090: instead of "GL-owning thread" being literally "the main thread", it becomes "the render thread" — the bridge's `MainThreadOp`/`done: threading.Event`/`defer` shape is a strong candidate template | reuse/generalize `CopilotBridge`'s shape as the render-thread's request queue, rather than inventing a new mechanism |

### The profiler's CPU spans + GPU timer ring (088)

| state | writer | reader | today's guarantee | what breaks | sync shape |
|---|---|---|---|---|---|
| `Profiler._stack`/`_root`/`_ring`/`_pending` | `Profiler.cpu()`/`gpu()` context managers, entered from BOTH `ui.py` (UI spans) and `Document.render`/`Pass.render` call sites (via the `profiler` parameter threaded through, `document.py:647-739`, passed by `ui.py:326` etc.) | `Profiler.begin_frame`/`end_frame`, `ProfileSmoother.feed` — main thread, `ui.py:271-279` | ONE profiler instance, ONE stack, spans nest by construction because everything is one thread calling into one object in strict sequence (`profiling.py` module docstring: "a document rendered inside another document's pass nests by construction") | the whole design assumes single-threaded nesting (`Span`'s parent is "whatever span is open" — `profiling.py:6-7`) and `gpu()` asserts no GL_TIME_ELAPSED query nests (`profiling.py:255-258`) — a render thread pushing spans onto the SAME `Profiler._stack` the main thread is also pushing onto is an immediate cross-thread data race on the stack itself, and the GPU query object requires the SAME GL context it was created on | the render thread needs its OWN `Profiler` instance (its own GL context can't share query objects with the main thread's context in general) whose `FrameProfile` is then merged/attached under the main thread's frame span — not a shared stack |

### The FPS chip

| state | writer | reader | today's guarantee | what breaks | sync shape |
|---|---|---|---|---|---|
| `app.global_fps` | `run()` loop (`ui.py:145-149`), measures WALL TIME of one `update_and_draw` call | `fps_overlay` (`ui.py:774`) | main-thread-only measurement of the whole frame including the (currently synchronous) render | once render is async, `update_and_draw`'s wall time stops including render time, so this measurement's MEANING changes (it becomes "UI thread fps" not "the whole pipeline's fps") — not a race, but a semantic gap the spec needs to address | decide what the FPS chip means post-090 (UI cadence vs render cadence vs both) — flagged as a design question, not a synchronization one |

### The pass strip's preview tiles

| state | writer | reader | today's guarantee | what breaks | sync shape |
|---|---|---|---|---|---|
| `Pass.canvas.texture` (`.glo` GL handle + `.size`) | `Pass.render()` draws into it; `Canvas.set_size()` releases+reallocates it (`core.py:158-164`) on a target-config change | `widgets/pass_list.py:_draw_pass_tile:120-121` (imgui image draw, every frame the Document tab is open), `widgets/document_grid.py:25-26,72` (document grid thumbnails), `ui.py:_draw_document_image` (the main preview) | main thread draws the texture with `.glo` (an active GL object handle) the SAME frame it was rendered, sequential | imgui's `add_image`/`image_with_bg` calls read `texture.glo` (an integer GL name) and issue GL draw calls against it — if the render thread reallocates the texture (a resize) concurrently with imgui trying to bind the OLD `.glo` for display, that's a use-after-release on the GL name; even without a resize, reading a texture that's mid-write from another context has no defined behavior without an explicit fence/sync | the UI must display a texture handle the render thread has FINISHED writing and PROMISES not to reallocate until the UI has used it that frame — classic producer/consumer texture hand-off, likely double-buffered textures or a GL fence |

### The copilot's render-and-look tool path (`probe_render`)

Already covered above under "Exports" — `probe_render` uses the same `bridge.run_on_main(...,
defer=True)` mechanism as `render_image`/`render_video`, confirmed at `copilot/backend.py:1706,
1726-1728`.

### `Document` save/load and `sync_documents_from_disk`

| state | writer | reader | today's guarantee | what breaks | sync shape |
|---|---|---|---|---|---|
| `UIDocument.save()` — compiles every program-less pass (`ui_models.py:389-390`) | main thread, called from `App.save()` | n/a (write path) | single thread — the compile-on-save always completes before the save continues to serialize uniforms | this is ANOTHER unmoved GL compile call site outside `Pass.render()` — if render moves to its own thread/context, this compile call (still main-thread) races the render thread's compile of the SAME pass | compile must be exclusively render-thread-owned; save must request-and-wait rather than compile in place |
| `sync_documents_from_disk` constructing a fresh `Document` (`project_session.py:605-622`) | main thread, phase 0.3, guarded on `not copilot.state.in_flight` | replaces `app.ui_documents[document_id]` — the render loop reads this dict fresh each frame | today's guard (`in_flight`) is about avoiding a race with the COPILOT worker's disk writes, NOT about avoiding a race with an in-progress render (there is none today — render is synchronous and always finishes before this runs) | once render is on its own thread, this guard is insufficient — a document dir changing on disk (e.g. an external editor save) and getting resynced while THAT document is mid-render on the worker thread needs the SAME kind of guard, generalized | extend the existing `in_flight`-style guard to also check "render in flight for this document" before resyncing it |

### `app.shutdown`

| state | writer | reader | today's guarantee | what breaks | sync shape |
|---|---|---|---|---|---|
| `App.shutdown()` → `App.release()` (`app.py:1878-1890`) | main thread, `run()`'s loop-exit path (`ui.py:151-153`) | releases copilot FIRST (`cancel_all()` + `join()`, `app.py:1831-1832`, comment: "so a queued GL op can't run against half-released documents"), THEN releases every document's GL objects (`app.py:1863-1867`) | the copilot's existing shutdown ordering IS the precedent: cancel + join the worker BEFORE releasing GL objects it might still touch | a render thread needs the identical treatment — cancelled/joined (or at minimum, confirmed idle) before `Document.release()` runs, or the render thread's in-flight GL calls hit released objects | copy the copilot's `release()` ordering pattern: signal the render thread to stop, join/wait, THEN release GL objects |

---

## 3. Ownership classification

**Render-thread-owned by nature (only the render touches them once moved):**
- `Pass.program` / `Pass.vbo` / `Pass.vao` and the whole `compile()` call — EXCEPT today's three
  other compile call sites (`UIDocument.save`, `tabs/uniforms.py:71`, `add_pass`) must be
  converted from "compile in place" to "request the render thread to compile."
- `Pass.canvas` / `Canvas.texture` / `Canvas.fbo` (GL objects, only ever meaningfully written by a
  draw call or a resize, both of which must become render-thread operations).
- `Document._feedback` swap timing and the feedback `Canvas` objects themselves.
- `Pass.drawn_frame`, `Document._frame` (render-loop-internal bookkeeping) — the MAIN thread's
  `begin_frame`/render-set computation needs a read-only, completed snapshot of these, never live
  access.
- The render thread's own `Profiler` instance (cannot share the main thread's GL query objects).
- `Document.time_origin` resolution into a per-frame `u_time` — the render thread should receive
  a resolved time value, not read `time_origin` and call `live_time()` itself if there's any risk
  of the main thread mutating `time_origin` (Reset) mid-frame.

**UI-owned (render reads a snapshot):**
- `Pass.uniform_values` writes from the panel, the copilot bridge, and the script engine — all
  three already happen on the main thread today and should continue to; the render thread
  consumes a per-frame published snapshot rather than the live dict.
- `Document.graph` (`PassGraph`) — already an immutable value object replaced wholesale on edit;
  the render thread can safely hold "the graph as of frame N" with no torn-read risk (see §5,
  false trail).
- `app.render_defer` / the copilot bridge's request queue — both are already
  UI/worker-initiated request objects; the render thread should be a THIRD kind of consumer of a
  similarly-shaped queue, not a special case.
- The FPS chip, the pass-strip tiles' non-GL metadata (name, output-ness, error-bool) — pure
  display reads of published state.

**Genuinely bidirectional (need real synchronization, not just "pick an owner"):**
- `Pass.source` (text/mtime) — written by THREE different subsystems (watch.py's mtime poll, the
  copilot's edit tools, the editor's `:w`) and read by `compile()`; all three writers are
  main-thread today and must be sequenced against a render thread that also needs to observe
  source changes at a frame boundary it controls.
- `Document.passes` (dict structure: add/delete/rename) and the GL releases that go with delete —
  today "safe" only because these always land strictly after that frame's render call in program
  order; once render is async there is no "after" to rely on, so this needs an explicit barrier.
- `Document._feedback` / `Pass.drawn_frame` / the whole `begin_frame` swap protocol — this is the
  single most timing-sensitive piece of shared state: it's read AND written by both "sides" across
  a frame boundary that currently has zero slack (single-threaded, so "before" and "after" are
  free).
- The Document/App wholesale-replace paths (project switch, `sync_documents_from_disk`,
  `_delete_document_unguarded`) — bidirectional in the sense that the render thread must be
  quiesced before the main thread destroys what it's rendering, and the main thread must know when
  quiescence is reached.
- The copilot's `document_tree`/`grep`/`read_lib` GL-free carve-out — currently a THIRD thread
  (copilot worker) reading `Document.passes`/`Pass.compile_unit`/`Pass.source` unguarded; adding a
  render thread as a second heavy writer makes this carve-out's safety argument need
  re-justification, not just a two-way UI/render split.

---

## 4. Exception paths

**Shader compile errors.** Never raised as Python exceptions — `Pass.compile()` catches all three
failure classes internally (resolver failure `core.py:345-353`, driver failure `core.py:355-367`,
uniform-type-check failure `core.py:369-381`) and stores them in `self.compile_unit.errors`
(`ShaderError` list). The previous valid `self.program` is deliberately left untouched on failure
so the preview keeps rendering the last-good frame. Surfaced to the user by per-frame POLLING of
`compile_unit.errors` (`tabs/code.py`'s error strip, `widgets/pass_list.py:103`,
`widgets/document_grid.py:72`), never by exception propagation. **A render thread needs this exact
shape preserved**: compile failures must still land in a place the UI can poll, and the render
thread must keep drawing the last-good program rather than going blank or crashing on a bad edit.

**GL errors (`moderngl.Error` / `ctx.error`).** Almost entirely unhandled in the live render path.
The only `except moderngl.Error` in the repo is `document.py:504`, inside `_seed_feedback` — a
LOAD-TIME path (restoring a persisted feedback frame), not the per-frame draw. Nothing in
`Document.render()`, `Pass.render()`, or `ui.py`'s frame loop catches a driver-level GL error or
reads `ctx.error` around a draw call. `ui.py:507`'s per-frame `moderngl.get_context().clear_errors()`
call erases whatever `ctx.error` state accumulated on the CURRENT (main-thread) context every
frame — per `profiling.py:16-19`'s own comment, this already destroys the one signal
`GL_TIME_ELAPSED` query-nesting violations leave behind. **A render thread has its own GL context
and its own `ctx.error`** — nothing in today's code reads or clears a second context's error state,
so this is unexplored territory, not a pattern to copy.

**A vanished shader file.** Handled at the FRAME level, not the render level:
`_tick_frame_state` (`ui.py:201-216`) checks file existence BEFORE any render call; on a miss it
fires the copilot's parked render (so a waiting worker doesn't stall), pumps glfw events, and
returns `None` — the entire frame is skipped (no render, no draw). A render thread would need an
equivalent "the source file is gone" check, but since compile already handles a missing/unreadable
file as a compile failure in some paths, this main-thread pre-check is really about not entering a
degenerate frame at all — likely stays on the main thread as a pre-dispatch gate.

**A vanished/corrupt media file.** `Video.__init__` raises a plain `ValueError` at bind time
(`media.py:184-201`) — outside the render path, caught (or not) by whatever constructs the
`Video`. Mid-playback, `Video.texture`'s getter (`media.py:241-258`) retries once on end-of-stream
then raises a dedicated `MediaError` (`media.py:106-107,252-254`) if `retrieve()` still fails —
called from `Pass.render()`'s uniform-binding loop (`core.py:~487-490`). **This `MediaError` is
NOT caught anywhere between `Pass.render()` and the top of `ui.py`'s frame loop** — it propagates
uncaught. `Video.update()` has no error path at all: a failed frame read just silently keeps the
last frame. A render thread needs an explicit catch around this that today's code does not have —
this is a genuine gap, not a pattern to replicate.

**A graph wiring error / cycle (`GraphError`).** Never a Python exception — `plan_passes`
(`pass_graph.py:283-346`) returns `GraphError` value objects alongside the plan. `Document.render()`
stores them in `self._graph_errors` (`document.py:674,678`), exposed via the `graph_errors`
property (`document.py:340-342`). **Nothing in the UI currently reads this property** (confirmed
by repo-wide grep — zero consumers outside `document.py` itself) — an existing gap, not something
090 needs to preserve so much as notice. On a cycle/unreachable output, `render()` falls back to
drawing just the resolved pass alone rather than nothing (`document.py:680-683`).

**An uncaught Python exception inside `Document.render()` itself.** No `try`/`except` anywhere in
`Document.render()` or its per-pass loop. Confirmed: `ui.py`'s four `document.render(...)` call
sites (`ui.py:326,341,356,363`) are BARE — wrapped only in a profiler context manager, no
`try`/`except` at any level between the call and `main()`'s top-level
`except Exception: logger.exception("ShaderBox crashed"); raise` (`ui.py:868-871`). Contrast with
`_draw_app_panel` (`ui.py:451-459`), which IS wrapped with `logger.error` + a user notification —
an established catch-log-continue pattern the codebase clearly knows, but never applied to the
render calls. **Today's model for a render exception is: propagate all the way up and crash the
process.** A render thread cannot inherit this — an uncaught exception on a worker thread doesn't
crash the main process, it silently kills the render thread (or, depending on implementation,
needs to be caught and marshalled back). This is the single largest behavioral change 090 must
deliberately design for for: today's "let it crash" is not an option once render is a separate
thread, so the render thread needs its own top-level catch that at minimum surfaces the failure to
the UI (a notification, an error state on the document) rather than either crashing silently or
being swallowed.

---

## 5. False trails

- **`Document.graph` (`PassGraph`) looked like classic mutable shared state, but it is a frozen
  pydantic model replaced wholesale on every edit** (`with_passes`/`with_target`/`with_output`,
  `pass_graph.py:143-170`, all funnel through `model_copy`). A render thread holding a reference to
  "the graph as of frame N" has no torn-read risk — only a staleness question (does it see the
  edit this frame or next), which is a far easier problem than mutation-in-place would be.

- **`widgets/media_ops.py` looked like a direct `Pass.uniform_values` writer** (it's named
  "media_ops" and the task instructions initially assumed it bound media). It does not: it
  operates purely on a `Video` object by value/reference and returns a (possibly new) one; the
  actual `Pass.uniform_values` write happens one call frame up, in `widgets/uniform.py:354-355,397`.
  `media_ops.py` itself is GL-free and thread-agnostic.

- **`shaderbox/pass_graph.py` as a whole looked like it needed threading analysis** (it's the
  planner `Document.render()` calls every frame via `effective_wiring`/`plan_for_output`). It is
  pure, GL-free, side-effect-free data transformation — `wired_pass`, `plan_passes`,
  `evaluation_order` take immutable inputs and return new values. The only state that matters for
  090 is what FEEDS it (`Document.graph`, `Pass.uniform_values`, already covered above), not the
  planner functions themselves.

- **`OutletRenderState.preview`/`current_artifact` (share_state.py) looked like they could race
  `Document.render()`** since they hold a `MediaWithTexture` and a `RenderedArtifact`. They are a
  separate rendered-to-disk-then-reloaded preview (`Video(art.path)`/`Image(art.path)`), decoded
  independently of the live document's GL objects — no synchronization concern beyond the export
  call itself (already covered under "Exports").

- **`app.canvas_size_buf`/`canvas_w_editing`/`canvas_h_editing` looked like document-shared
  state** (they mirror `Document.canvas_size` in the Document tab). They are pure `App`-level imgui
  text-field scratch buffers, never read or written by `Document`/`Pass` directly — they only ever
  feed into a call to `Document.set_canvas_size`, which IS the real synchronization point (a GL
  resize).

- **`RenderedArtifact` (the exporters' cross-thread value type) looked like it might carry a live
  `Document`/`Pass`/texture reference** since it flows from a render into upload workers. It is a
  fully GL-free value type (`path`, `is_video`, `duration`, `size` — no GL handles), by explicit
  design (`exporters/base.py`'s own docstring says as much). Neither `telegram.py` nor `youtube.py`
  ever touches a `Document`/`Pass` object; their worker threads only ever see file paths and bytes.

- **`app.pass_draft` (the "add pass" wizard state) looked like it could collide with
  `Document.passes` during a rename.** It's a wholly separate draft object, discarded on both
  create and cancel, and never aliases a live `Pass` — it only touches `Document.passes` via the
  explicit `session.add_pass` call at commit time, which is already covered under "pass add/delete."

- **`app.ui_document_examples` looked like it might share `Document` instances with
  `app.ui_documents`** (since `create_document_from_example` reads from it). `load_document_from_dir`
  constructs a genuinely NEW `Document`/`UIDocument` from the examples directory — the example's
  own in-memory copy is untouched and keeps rendering independently (confirmed by `ui.py:342-356`
  ticking examples through a SEPARATE render-loop branch from `tick_documents`).

- **`self.app_state.copilot_source_lock`'s setter being a bound method (`project_session.py:317`)
  looked like a stale-closure hazard.** It's written that way specifically BECAUSE a lambda closure
  captured at construction would go stale across a project switch — an already-fixed footgun, not
  a live one.

---

## 6. Coverage statement

**Read in full, end to end (primary agent):** `CLAUDE.md`, `ai_docs/dev_flow.md` (`### Module
map` section, pages 1-696 of 833 — the remainder past line 696 was not needed and not read),
`shaderbox/ui.py`, `shaderbox/document.py`, `shaderbox/core.py`, `shaderbox/profiling.py`,
`shaderbox/watch.py`, `shaderbox/scripting/engine.py`, `shaderbox/copilot/bridge.py`,
`shaderbox/copilot/session.py`, `shaderbox/pass_graph.py`, `shaderbox/render_defer.py`. Targeted
reads (specific sections, not full files): `shaderbox/app.py` lines 1-400 and 1768-2072 (the
remaining 400-1768 was delegated), `shaderbox/project_session.py` lines 540-700 (`sync_documents_
from_disk`, `reload_scripts`, `tick`'s call site), `shaderbox/ui_models.py`'s `UIDocument.save`
method (lines ~356-536), `ai_docs/conventions.md` (grepped + read lines 260-310 for the 065/066/069
design-decision wording), the existing `ai_docs/features/090_render_decoupling/research/
input_method.md` (a sibling research doc on a different sub-topic — X11 input-method lag — no
overlap with this ledger, confirmed no contradiction).

**Read in full by delegated sub-agents (sonnet, 4 in parallel, each anchored to a disjoint file
set and reporting file:line citations), findings merged above:**
- Agent 1: `shaderbox/tabs/code.py`, `document.py`, `uniforms.py`, `render.py`, `share.py`,
  `share_state.py`, `widgets/pass_list.py`, `uniform.py`, `details.py`, `media_ops.py`,
  `document_grid.py`, `popups/pass_settings.py` (plus targeted reads of `core.py`/`document.py`/
  `project_session.py` call chains for correct attribution).
- Agent 2: `shaderbox/copilot/tools/shader.py`, `inspect.py`, `publish.py`, `passes.py`,
  `media.py`, `document_ops.py`, `shaderbox/copilot/backend.py` (2680 lines, full), `shaderbox/
  exporters/base.py`, `worker.py`, `telegram.py`, `youtube.py`, `shaderbox/render_job.py`.
- Agent 3: `shaderbox/app.py` lines 400-1768 plus targeted reads of 1780-2072 for specific
  methods, `shaderbox/hotkeys.py`, `commands.py`, `project_session.py` (full), `popups/projects.py`,
  `examples.py`, `settings.py`.
- Agent 4: `shaderbox/shader_errors.py`, `media.py` (full), `notifications.py`, `util.py`,
  `pass_graph.py` (full, independently confirming Agent/primary's read), `editor/render.py`,
  `channel_blit.py`, plus repo-wide grep for `moderngl.Error`/`ctx.error`/`clear_errors()`/
  `GLError`/`except Exception`/`try_to_release`.

**Skimmed only (grep/signature-level, not read end-to-end):** `shaderbox/tabs/document.py`
(canvas_size_buf usage only, per Agent 3's explicit note that it deferred full coverage to Agent
1's territory — confirmed no gap, Agent 1 covered `tabs/document.py` in full), `shaderbox/
ui_regions.py`, `shaderbox/commands.py` (confirmed inert w.r.t. Document/Pass — pure static
command-spec data, no state), `shaderbox/scripting/behavior.py`/`context.py`/`errors.py`/`keys.py`
(the script-engine's supporting modules — `engine.py` itself, the one that actually writes
`Pass.uniform_values`, was read in full), `shaderbox/intel/*` (code-editor intelligence — writes
only to editor-session buffers, never `Document`/`Pass`, confirmed by absence from every grep),
`shaderbox/widgets/copilot_chat.py`/`cheatsheet.py` (UI chrome with no Document/Pass state per
the module map's own description).

**Not read at all:** `shaderbox/editor/ffi.py`/`input.py` (the vendored editor binding — operates
on editor buffers, not `Document`/`Pass`, per the module map), `shaderbox/tests/*`,
`shaderbox/scripts/*` (dev tooling), the full text of `ai_docs/features/066_perf_and_test_diet/`
and `089_ninth_walk_findings/` specs (only grepped for the D1/D2/D4/D9 decision wording already
quoted verbatim in the source code's own comments, which is the more precise citation).

---

## Summary

68 ledger rows across uniform values, the clock, iteration counters, feedback histories, media
textures, shader recompilation, pass structural edits, document/project switching, the render-set
decision, exports, the profiler, the FPS chip, preview tiles, and save/load — plus 8 false trails
and a full exception-path audit. The three riskiest rows:

1. **`tabs/uniforms.py:71` lazily calling `Pass.get_active_uniforms()` → `Pass.compile()` every
   frame the Uniforms tab is open** — a main-thread GL compile racing a render-thread draw against
   the SAME pass's program/vbo/vao, today invisible because everything is one thread.
2. **`Document._feedback` swap protocol (`begin_frame`) and `Pass.drawn_frame`** — the tightest
   timing dependency in the codebase, currently free because single-threaded, and the piece most
   likely to need a hard frame-completion barrier rather than a snapshot.
3. **No exception handling around `Document.render()` anywhere in `ui.py`** — today's model is
   "propagate and crash the process," which cannot survive a move to a worker thread and has no
   existing pattern in this codebase to copy (the copilot's catch-log-continue pattern exists
   elsewhere but was never applied here).

Secondary but load-bearing: the copilot's existing `document_tree`/`grep`/`read_lib` GL-free
worker-thread carve-out needs re-justification once a second heavy writer thread (render) exists;
`UIDocument.save()`'s direct main-thread compile call and the `feedback/<pass>.bin` persist's GL
texture readback are both unmoved GL-touching call sites outside `Document.render()` that 090 must
also account for, not just the render loop itself.
