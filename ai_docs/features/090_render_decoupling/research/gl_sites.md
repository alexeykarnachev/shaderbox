# 090 research: every GL touch point

Exhaustive inventory of moderngl / glfw / imgui-image-presentation / GL-sync call sites across
`shaderbox/` and `scripts/`, with today's thread + frame phase and a reasoned split proposal
for feature 090 (document rendering on a worker thread, its own GL context shared with the
window's — glfw's event pump, imgui and the code editor stay on main).

Every site today runs on the **main thread** — there is currently exactly one thread that ever
touches `moderngl`. The "thread today" column below names the FRAME PHASE instead, since that is
the only axis of variation that exists pre-090. The copilot worker thread exists today but is
proven **GL-free by construction** (`copilot/bridge.py` marshals every GL-affine call back to
main) — see part 3.

## 1. Per-frame GL call sites

Legend for the last column: **RENDER** = move to the render thread; **MAIN** = stays on main
(imgui/editor/window-adjacent); **HANDOFF** = the boundary itself (a texture handle or pixel
buffer crossing threads).

| file:function | GL action | object | frame phase (today) | consumer | 090 proposal |
|---|---|---|---|---|---|
| `core.py:124 Canvas.__init__/_init` | create texture+fbo (`gl.texture`, `gl.framebuffer`) | document pass canvas | lifecycle (pass construction) — see part 2 | `Pass.render` writes it, imgui reads its `.glo` | RENDER (owned by document render state) |
| `core.py:158 Canvas.set_size` | release+recreate texture+fbo | document pass canvas | document-render block (`Document.render`, on a target-size change) | same | RENDER |
| `core.py:154 Canvas.release` | `.texture.release()` / `.fbo.release()` | document pass canvas | lifecycle | — | RENDER |
| `core.py:299 Pass._black_texture` | create 1x1 texture | per-pass "no source" fill | document-render block (lazy, first sampler bind) | `Pass.render`'s texture unit bind | RENDER |
| `core.py:332 Pass.compile` | `gl.program(...)`, `gl.buffer(...)`, `gl.vertex_array(...)`, releases old program/vbo/vao | pass program+geometry | document-render block (lazy compile, first `render()`/`get_active_uniforms()` call) — also hot-reload, see part 2 | `Pass.render` | RENDER |
| `core.py:412 Pass.compile` (glyph table write) | `member.write(table_data)` | program-resident uniform (glyph strokes) | document-render block, inside compile | shader draw | RENDER |
| `core.py:432 Pass._default_uniform_value` | `gl.buffer(zeros)` | uniform-block buffer default | document-render block (lazy, first uniform seed) | `Pass.render`'s `bind_to_uniform_block` | RENDER |
| `core.py:437 Pass.render` | `texture.use(location=...)`, `program[name]=value`, `canvas.fbo.use()`, `gl.clear()`, `vao.render()` | pass program + canvas + input textures | document-render block (`ui.py:326/341/356/363`, export loops in `document.py`, copilot probe) | imgui preview (`.glo` handle), disk (export), copilot probe pixel readback | RENDER |
| `core.py:257/262/286 Pass.release_program/invalidate/release` | releases program/vbo/vao (+ `glUseProgram(0)` suppressed) | pass program+geometry | hot-reload (`watch.py`), copilot edit, editor flush-to-disk — see part 2 | — | RENDER |
| `document.py:230 _load_uniform_value` | `gl.texture(...)` (loading a persisted texture uniform) | uniform-bound texture asset | lifecycle: document load (`load_from_dir`) | `Pass.render`'s sampler bind | RENDER |
| `document.py:240 _load_uniform_value` | `gl.buffer(base64 data)` | uniform-block buffer asset | lifecycle: document load | `Pass.render`'s `bind_to_uniform_block` | RENDER |
| `document.py:495-511 Document._seed_feedback` | `Canvas(...)` create + `canvas.texture.write(data)` | feedback-pass seed canvas | lifecycle: document load (089 D5 seed restore) | feedback read next frame | RENDER |
| `document.py:536 Document._feedback_canvas` | `Canvas(...)` create (or `.set_size`) on demand | feedback-pass history canvas | document-render block (first frame a feedback pass reads itself) | `Document.render`'s self-read input, and `_swap_feedback` | RENDER |
| `document.py:392 Document._swap_feedback` | swaps `Canvas` object refs (no new GL call, but reassigns which Python object owns which live GL texture) | feedback pass canvas ↔ history | document-render block, `begin_frame` (once/frame) + between iterated-pass iterations | next read of that pass | RENDER (must stay atomic with the render step it swaps around) |
| `document.py:647 Document.render` | orchestrates the pass-by-pass draw order, dispatches to `Pass.render` per part above | whole document | document-render block (`ui.py`), export loops, copilot probe (`inspect.py::_probe_frame`) | see individual `Pass.render` rows | RENDER |
| `document.py:809 Document._render_image` / `837 _render_video` | `texture_to_pil` / `texture_to_rgba8` (texture readback), drives `Document.render` per frame | output canvas | Render-tab/Share-tab deferred fire (`render_defer`, post-swap) or copilot `render_image`/`render_video` (via bridge, deferred) | disk (image/video file) | RENDER, with the readback crossing to whichever thread does the file I/O (can stay on render thread — no imgui dependency) |
| `document.py:951 Document.render_media` | `Canvas(gl=self._gl, size=...)` create + release (fit-policy scratch canvas) | export scratch canvas | same as above | export encode | RENDER |
| `media.py:153/232 Image/Video.texture` | `moderngl.get_context().texture(...)` (lazy first access) | media (Image/Video) texture | document-render block (`Pass.render`'s `MediaWithTexture.update`), or UI draw (`media_ops.py`, share preview) | sampler bind, or imgui preview | RENDER for document-bound media; the Share/Telegram preview instances (part below) are a HANDOFF case — see note under `media_ops.py` |
| `media.py:70 texture_to_rgba8` | `texture.read()` | any texture (used by export + smoke canary) | document-render block / export | disk / test assertion | RENDER (pure readback, no imgui dependency) |
| `media.py:239/300 Video._upload_frame / release` | `texture.write(data)` / `.release()` | video media texture | document-render block (every `update(t)` that advances a frame) | sampler bind | RENDER (document-bound) |
| `channel_blit.py:45 ChannelBlit.__init__` | `gl.program`, `gl.buffer`, `gl.vertex_array`, `Canvas(...)` | Alpha/RGB channel-view blit | lifecycle: `App._init` (`app.py:1234-1235`) | viewer draw | RENDER (reads the document's output texture, writes a view texture consumed by imgui — same shape as the document canvas itself) |
| `channel_blit.py:64 ChannelBlit.render` | `source.use()`, `canvas.fbo.use()`, `gl.clear()`, `vao.render()` | channel-view canvas | imgui draw phase, called from `ui.py:690/693` inside `_draw_document_image` (**not** the document-render block — runs during imgui drawing, gated on `ChannelView` state) | imgui preview (`.glo`) | RENDER, but this is the one call site that today executes AFTER imgui drawing has started (`ui.py`'s main-window body) — it must move earlier, into the document-render block, or the handoff must resolve the channel view server-side before imgui asks for the texture. Flagged explicitly below. |
| `channel_blit.py:74 ChannelBlit.release` | releases canvas/vao/vbo/program | channel-view blit | lifecycle (`App.release`, `app.py:1873/1876`) | — | RENDER |
| `ui.py:507 clear_errors()` | `moderngl.get_context().clear_errors()` | GL error state | imgui draw phase, top of `_update_and_draw`'s post-frame section | swallows the profiler's `GL_INVALID_OPERATION` before it's readable (see `profiling.py` docstring) | Ambiguous — this clears errors for the CURRENT context. Under 090 there will be two contexts (main + render); each needs its own error-clear discipline. MAIN (for the window/imgui context) with an equivalent added on the render thread. |
| `ui.py:509-511 gl.screen.use()/gl.clear()` | binds the default framebuffer, clears it | window backbuffer | imgui draw phase | swap | MAIN (this is the window's own backbuffer, imgui's output target — never a document texture) |
| `ui.py:514 app.imgui_renderer.render(...)` | draws imgui's draw-data (internally issues GL draw calls against every bound texture id, including every `.glo` the frame referenced) | imgui draw list | imgui draw phase | screen | MAIN |
| `ui.py:517 glfw.swap_buffers` | swap | window | imgui draw phase | display | MAIN |
| `ui.py:526 gl.finish()` | GPU sync barrier | — | post-swap, gates the deferred-render firing point (both `RenderDefer` and the copilot bridge's parked op) | — | MAIN today; under 090 this barrier's PURPOSE (make sure the cue frame is on the glass before a synchronous encode freezes the loop) goes away once export runs on the render thread without blocking main — see part 3 discussion |
| `ui.py:528/530 render_defer.fire_and_clear()` | fires the queued closure (a `Document.render_media` call) | whichever document was queued | post-swap deferred | disk / share preview | RENDER (the queued closure becomes cross-thread work; `RenderDefer`'s one-frame latch either moves to the render thread or becomes a request the render thread services asynchronously) |
| `ui.py:537 copilot.bridge.run_deferred_render()` | fires the copilot's parked GL op | copilot render/publish op | post-swap deferred | copilot worker (unblocks it) | RENDER (bridge target becomes the render thread instead of main; see part 3) |
| `editor/render.py:158 EditorRenderer.__init__` | `gl.program`, `gl.texture` (MTSDF atlas) | editor glyph atlas | lifecycle: first `code_tab.draw` call (`tabs/code.py:1042`) | editor draw | MAIN — see part 4, no document dependency |
| `editor/render.py:178 EditorPanel._ensure_target` | releases+creates FBO+texture | one editor pane's render target | imgui draw phase (`tabs/code.py`, inside `code_tab.draw`, gated by `should_redraw`) | imgui `add_image` (`tabs/code.py:1056`) | MAIN |
| `editor/render.py:201 EditorPanel.render` | `fbo.use()`, `fbo.clear()`, `vbo.write()`, `vao.render()`, `atlas.use()` | editor pane FBO | imgui draw phase | imgui `add_image` | MAIN |
| `editor/render.py:256 EditorPanel.release` | releases vao/vbo/fbo/texture | editor pane FBO | lifecycle (`App.release`, tab close) | — | MAIN |
| `profiling.py:295-306 Profiler._query_for` | `gl.query(time=True)` create | GPU timer-query ring | inside `gpu()` spans — both document-render-block spans (`document:{name}`, `pass:{name}`) AND imgui-draw-phase spans (`ui:draw`, `editor:draw`, `viewer`) | frame profile panel | SPLIT — a document-render GPU span belongs on the render thread's own ring; an imgui/editor GPU span stays on main's ring. `Profiler` as written is a single object with ONE `_gl`; 090 needs two instances (or a thread-keyed ring) — flagged as a design question, not just a move. |
| `widgets/uniform.py:249 _draw_texture_preview` (`current_value.write(data)`) | `.write()` on a uniform-block Buffer, from a slider edit | pass uniform buffer | imgui draw phase (uniform panel edit) | next `Pass.render`'s `bind_to_uniform_block` | HANDOFF — a UI edit on main must reach a GL buffer owned by the render thread; this becomes a cross-thread write request, not a same-thread mutation |
| `widgets/document_grid.py:22-26` (`preview_cell` call) | none itself — reads `.glo`/`.size` off `document.render_pass.canvas.texture` | document preview tile | imgui draw phase | imgui image | HANDOFF (int texture-id read across the boundary; presentation itself is MAIN) |
| `widgets/pass_list.py:117-121` (`preview_cell` call) | same — reads `.glo`/`.size` | pass-strip tile | imgui draw phase | imgui image | HANDOFF |
| `widgets/uniform.py:192/334/346` (`_draw_texture_preview`) | reads `.glo`/`.size` off a sampler's source texture | sampler preview | imgui draw phase | imgui image | HANDOFF |
| `ui_primitives.py:213/410/1255` (`imgui.image`/`add_image`) | presents a `texture_glo` int | generic preview primitive (`centered_image`, `preview_box`, `preview_cell`) | imgui draw phase | screen | MAIN (presentation only — takes an `int`, never a live moderngl object) |
| `ui.py:639-645 _draw_canvas_backdrop` | `add_image` on `checker.glo` | alpha-checker backdrop | imgui draw phase | screen | MAIN (the checker texture itself lives in `App`, is GL-free of any document — created once in `_make_checker_texture`, `app.py:130`) |
| `ui.py:703-709` (`imgui.image_with_bg`) | presents `shown_texture.glo` (the document output OR a channel-blit result) | document preview | imgui draw phase | screen | MAIN presents; the texture itself is a HANDOFF (see `ChannelBlit` flag above) |
| `tabs/code.py:1056-1063` (`add_image`) | presents `panel.texture.glo` | editor pane | imgui draw phase | screen | MAIN |
| `tabs/render.py:35 centered_image` | presents `tex.glo` (`document.render_pass.canvas.texture`) | Render-tab preview | imgui draw phase | screen | MAIN presents; texture is a HANDOFF |
| `tabs/share.py:124-125` (`preview.texture.glo`) | reads `.glo`/`.size` off the outlet's rendered-artifact preview (`Image`/`Video` built from the exported FILE, not a document canvas) | Share-tab artifact preview | imgui draw phase (`_draw_outlet`) | screen | This texture is created fresh from a FILE the render already finished writing (`tabs/share_state.py:36`) — it is GL-free of any document object, so it's really a lifecycle site (see part 2) that happens to run inline in a draw call today. MAIN is fine to keep it here since it never touches a live document/pass texture. |
| `exporters/telegram.py:688-695 _draw_grid_cell` | reads `.glo`/`.size` off a sticker-slot thumbnail (`Image`/`Video`, lazily built from a downloaded/cached file) | Telegram sticker preview | imgui draw phase | screen | Same as `tabs/share.py` above — file-backed, not document-backed. MAIN. |
| `exporters/youtube.py:401 draw_target_panel` | `render_control.preview_texture_glo` | YouTube outlet preview (delegates to Share tab's `preview_texture_glo`) | imgui draw phase | screen | MAIN (same handle as `tabs/share.py`) |

### The one call site out of place today: `ChannelBlit.render`

`ui.py:690/693` calls `app.alpha_view.render(output_texture)` / `app.rgb_view.render(...)`
**during imgui drawing** (`_draw_document_image`, inside the main window's body), not during the
document-render block earlier in the frame. This is the only GL-render call that currently
executes after `imgui.new_frame()`. Under 090 either (a) the channel view is resolved as part of
the document-render block on the render thread, with imgui only ever reading a resolved handle,
or (b) `ChannelView` state has to be read by the render thread before it starts rendering that
document — option (a) is the smaller change since it needs no new signal path.

## 2. Lifecycle sites (create/release outside the per-frame render path)

| file:function | event | GL action |
|---|---|---|
| `app.py:152 App.__init__` | process start | `glfw.init/create_window/make_context_current`, `moderngl.init_context()`, `moderngl.get_context().gc_mode = "auto"` — the window's GL context is born here |
| `app.py:1230 App._init` (via `_rewire_exporters`/direct) | project load / switch (`switch_project`, `app.py:1971`) | `_make_checker_texture` (`app.py:142`), `ChannelBlit(ALPHA_FS)`/`ChannelBlit(RGB_FS)` (`app.py:1234-1235`) — recreated on EVERY project switch |
| `project_session.py:491 ProjectSession.load` | project load (called from `App._init`) | `load_documents_from_dir` → `Document.load_from_dir` per document dir → `Canvas`/`Pass` GL object creation, per document in the project |
| `app.py:1822 App.release` | project switch (top of `_init`) AND process shutdown (`App.shutdown`) | `copilot.release()` (cancels bridge — see part 3), `exporter_registry.release()`, `editor_panel.release()`, `share_tab_state.release()`, every `ui_documents[*].document.release()`, every `ui_document_examples[*].document.release()`, `checker_texture.release()`, `alpha_view.release()`, `rgb_view.release()` |
| `app.py:1878 App.shutdown` | process exit (`ui.py:153`, end of `run()`) | calls `release()` above, then `imgui_renderer.shutdown()`, `imgui.destroy_context()` |
| `app.py:1951 App.switch_project` | user picks a different project (Projects modal → `request_project_switch` → consumed in `ui.py:171-180 _tick_frame_state`, **before any drawing** — the 084 D5 deferral) | calls `App.release()` then `App._init(path)`, i.e. the ENTIRE project's GL state (every document's Canvas/Pass, the checker texture, both channel blits, the editor panel) is torn down and rebuilt on ONE frame boundary |
| `ui.py:171-180 _tick_frame_state` | consumes `app.pending_project_switch` | this is the 084 D5 deferral point itself: the modal (`popups/projects.py`) never switches inline, because a popup body draws AFTER the editor panel and the document image have already pushed their GL texture handles into the current frame's imgui draw list — releasing those objects mid-frame would leave imgui rendering freed GL names. The switch is deferred to the TOP of the next `_tick_frame_state`, strictly before any imgui call that frame. **This deferral pattern is exactly the shape 090 needs to generalize**: any render-thread teardown/rebuild must land at a point neither thread has already queued GL work referencing the old objects. |
| `project_session.py:410 _delete_document_unguarded` | user deletes a document (`App.delete_document`) or copilot deletes one (`backend.py::delete_document`, via `run_on_main`) | `ui_documents.pop(document_id).document.release()` |
| `project_session.py:550 sync_documents_from_disk` | every frame (`_tick_frame_state`, `ui.py:199`), gated `not app.copilot.state.in_flight` | diffs `documents/*/document.json` mtimes; for a removed dir: `.document.release()`; for an added/changed dir: `_load_one_document_from_disk` → releases the old live copy (if any) then `load_document_from_dir` (fresh Canvas/Pass GL objects) |
| `project_session.py:605 _load_one_document_from_disk` | called from `sync_documents_from_disk` (external file change) AND indirectly wherever a fresh disk copy must replace a live one | `old.document.release()` then a fresh `load_document_from_dir` |
| `watch.py:19 reload_document_if_changed` / `_reload_pass_if_changed` | every frame (`_tick_frame_state`, `ui.py:217`), per pass whose source mtime changed on disk | `render_pass.release_program(new_text)` (root shader changed) — releases program/vbo/vao and recompiles; `render_pass.invalidate()` (an included lib file changed) — same release, recompile deferred to next need |
| `watch.py:66 maybe_rebuild_lib_index` | every frame, when the shader-lib root's file set/mtimes changed | invalidates every pass that pulled in a lib file (`render_pass.invalidate()`) across every open document |
| `core.py:240 Pass.set_target` | pass-settings change (target format/scale edit, `popups/pass_settings.py` → `App`/`ProjectSession` plumbing) | `canvas.release()` + fresh `Canvas(...)` at the new format |
| `document.py:437 Document.drop_feedback` | pass deleted or renamed | releases that pass's feedback-history `Canvas` |
| `document.py:425 Document.reset_feedback` | export entry (`render_media`, D10) or a live Reset command | releases every feedback `Canvas` in the document |
| copilot pass tools (`backend.py`, `tools/passes.py::add_pass/set_pass/delete_pass`) | copilot adds/edits/deletes a pass, mid-turn | marshalled via `self._bridge.run_on_main` — the actual `Pass(...)` construction / `.release()` happens on MAIN inside the closure the worker handed over |
| `backend.py:1146 create_document` (`_create_document_on_main`), `:1587 import_document`, `:1405 duplicate_document` | copilot/UI creates a document | `load_document_from_dir` (fresh GL objects) — copilot path via `run_on_main`; UI path (`App.create_document_from_example`, `app.py:2038`) directly on main since it already IS main |
| `tools/media.py::bind_media/unbind_media` (→ `backend.py`) | copilot binds/unbinds a media uniform | media texture creation is deferred (lazy `.texture` property on `Image`/`Video`); the bind itself is a `run_on_main` closure that constructs the `Image`/`Video` object and assigns it into `uniform_values` |
| `app.py:1613/1925 edited_pass.release_program(text)` | editor tab flushed to disk (`App.flush_all_dirty_editors`, `App.flush_current_editor`) — hotkey save, project switch, document delete guard | releases + recompiles the edited pass's program |
| `app.py:1042-1045 (tabs/code.py)` `EditorRenderer(...)`/`EditorPanel(...)` | first `code_tab.draw` call after `App.release()` reset them to `None` (project switch, or process start) | creates the shared MTSDF program+atlas and the per-panel FBO |
| `app.py:1234-1235` `ChannelBlit(ALPHA_FS)`/`ChannelBlit(RGB_FS)` | every `App._init` (project switch AND process start) | full GL object set for both channel-view blits |
| `app.py:130 _make_checker_texture` | every `App._init` | 2x2 checker texture |
| `exporters/registry.py:40 ExporterRegistry.release` | `App.release` (project switch / shutdown) | delegates to each `Exporter.release()` — `TelegramExporter.release` (`telegram.py:601`) additionally releases sticker-slot `Image`/`Video` thumbnails (`_release_sticker_slots`, `telegram.py:864-869`) |
| `tabs/share_state.py:39 OutletRenderState._release_preview` | new artifact rendered (`set_artifact`) or `TabState.release()` (project switch) | releases the rendered-artifact preview `Image`/`Video` |

## 3. The existing worker→main GL bridge (`copilot/bridge.py` + `render_defer.py`)

Read end to end. This is the ONE proven pattern in the codebase for a non-GL thread getting GL
work done, and it is what 090 will either replace outright (the render thread becomes the new
GL-capable side) or fold the copilot worker's needs into.

**Shape.** `CopilotBridge` (`copilot/bridge.py:32`) is a synchronous, blocking round-trip queue.
The copilot's worker thread (spawned in `copilot/session.py:416`, never touches GL — see the
`exporters/base.py:116` contract quote below, which the copilot mirrors) wraps any GL-affine work
as a closure and calls `bridge.run_on_main(fn, timeout=..., defer=...)`
(`copilot/bridge.py:52-67`): it enqueues a `MainThreadOp(fn=fn, defer=defer)` onto a
`queue.Queue(maxsize=64)` and blocks on `op.done: threading.Event` until the main thread runs it.

**Draining.** `App`/`ui.py` calls `app.copilot.drain_bridge()` → `bridge.drain(max_ops=8)`
(`copilot/bridge.py:69`) once per frame, at the TOP of `_tick_frame_state` (`ui.py:190`, "EARLY so
a freshly recompiled document renders this same frame"). `drain` runs up to 8 queued ops inline,
each wrapped so a raising op sets `op.error` rather than crashing the frame loop, then
`op.done.set()` unblocks the worker with the result (or re-raises the error on the worker side).

**The `defer` twist.** An op marked `defer=True` is NOT run inline by `drain()` — it's parked in
`self._deferred_render` (a single slot; the worker is single-threaded and blocks on this one op,
so nothing else can queue behind it) and `drain()` stops draining that frame the instant it hits
one. `run_deferred_render()` (`copilot/bridge.py:90`) fires it LATER, from `ui.py:537`, strictly
AFTER that frame's `glfw.swap_buffers` + `gl.finish()` (`ui.py:516-526`). The reason (comment,
`copilot/bridge.py:24-28` and `ui.py:519-523`): a render/video-encode op freezes the main loop for
potentially seconds, and the "Rendering..." cue (an imgui overlay) must be provably ON THE GLASS
before that freeze starts — `gl.finish()` forces the GPU to actually display the swapped buffer
before the thread blocks in a synchronous encode. `render_pending()` tells the UI to keep drawing
the cue while a deferred op is parked. `RenderDefer` (`render_defer.py`) is the SAME pattern for
the Render-tab and Share-tab's own render buttons (`tabs/render.py:69`, `tabs/share.py:140`) —
independent of the copilot, same one-frame latch + same post-swap firing point in `ui.py`.

**Timeout / cancellation.** `run_on_main` takes an optional `timeout` (default
`COPILOT_ENGINE.bridge_op_timeout_s`) since some ops (a video encode) legitimately run long;
`cancel_all(reusable=...)` (`copilot/bridge.py:111`) unblocks every parked/queued op with a
`CopilotCancelled` error so a `join()` on the worker thread can't deadlock — called from
`CopilotSession.release()` before the document teardown in `App.release()` (`app.py:1829-1832`,
comment: "Copilot first... so a queued GL op can't run against half-released documents").
`reopen()` clears a `_shutdown` latch so a reused bridge (same process, new project) serves again.

**What 090 changes.** Once document rendering has its own thread with its own (shared) GL
context, the copilot worker's GL-affine calls no longer NEED to marshal to the MAIN thread — they
need to marshal to the RENDER thread instead. The bridge's blocking-round-trip shape (queue +
`threading.Event` + timeout + cancel-all) is directly reusable; only the thread on the other end
of `drain()`/`run_deferred_render()` changes. The `defer` + post-swap-cue mechanism is more
interesting: it exists ONLY because a synchronous encode currently freezes the SAME thread that
draws the cue. If document render + encode move to a separate thread, main never freezes, so the
cue no longer needs to be "provably on the glass before the freeze" — the cue can simply be drawn
every frame the render thread reports work in flight, and `gl.finish()` at `ui.py:526` (today
forcing the swap to composite before the encode blocks) loses its reason to exist on main. Net:
the bridge's transport survives, but `render_defer.py`'s one-frame latch and the `defer`
parking half of `CopilotBridge` are solving a problem 090 removes by construction.

## 4. The editor's own renderer (`shaderbox/editor/render.py`)

Read end to end (262 lines). `EditorRenderer` (`editor/render.py:158`) owns ONE GL program (the
MTSDF glyph-decode shader) + ONE glyph-atlas texture, shared by every open editor tab/session —
built once, lazily, the first time `tabs/code.py::draw` runs after `app.editor_renderer` is
`None` (fresh process, or just after a project-switch `App.release()` cleared it). `EditorPanel`
(`editor/render.py:178`) is one per DRAWN editor region: an FBO+texture pair, resized on demand
(`_ensure_target`), plus its own VBO/VAO for the interleaved glyph-quad+solid-geometry vertex
buffer the layout emits (`build_vertices`). `render()` draws the last `Editor.layout()`'s
primitive array into the panel's own FBO and returns the panel's texture; `tabs/code.py:1056`
presents that texture's `.glo` via `imgui.get_window_draw_list().add_image(...)`, gated by
`should_redraw` comparing `render_state()` tuples (a pure, GL-free redraw gate — `render_state`/
`should_redraw` are free functions specifically so this gate is unit-testable without GL,
per the module docstring).

**Dependency on document textures: none.** `EditorRenderer`/`EditorPanel` read only from the
editor FFI (`Editor.prims_array()`, cursor/scroll/mode getters) and a static atlas PNG/JSON
shipped in `shaderbox/resources/editor/`. Nothing in this module imports `core.py`, `document.py`,
or touches a `Pass`/`Canvas`/`Document` object. It is fully independent of whatever the render
thread will own — the strongest case in the whole codebase for "stays on main, unchanged."

## 5. Counts

- **Total per-frame GL call sites (part 1 table):** 47 rows.
- **By category:**
  - create (texture/fbo/program/buffer/vao/query): 15
  - render/draw (`.render()`, `.use()` + clear, program dispatch): 7
  - upload (`.write()`): 5
  - read back (`.read()`, `texture_to_rgba8`/`texture_to_pil`): 3
  - release: 6
  - present (imgui `.image`/`add_image` reading a `.glo`): 12 (some sites overlap with "read back
    of a handle" — counted once under present since that's the operative action)
  - GL error/sync state (`clear_errors`, `gl.finish`, `screen.use`/`clear`): 4 (window/imgui-only)
- **Lifecycle sites (part 2 table):** 23 distinct file:function entries.
- **Distinct GL object kinds touched:** 8 — `moderngl.Texture`, `moderngl.Framebuffer`,
  `moderngl.Program`, `moderngl.Buffer` (both vertex/index buffers and uniform-block buffers),
  `moderngl.VertexArray`, `moderngl.Query`, `moderngl.Context` itself (error state / `gc_mode` /
  `screen`), and the window's default framebuffer (`gl.screen`, not a `Framebuffer` object but a
  distinct target).
- **Threads with GL access today:** 1 (main). The copilot worker thread and both exporter worker
  threads (`ExporterWorker`, `telegram.py`, `youtube.py`) are GL-free by construction and
  documented as such (`exporters/base.py:101-119`).

## 6. False trails

- `grep "ctx\."` across the repo returns nothing relevant — the GL context variable is named
  `gl` / `self._gl` / `self.gl` throughout, never `ctx`. A search anchored on `ctx.` would have
  silently missed the entire codebase; every real search had to key on `gl.`/`_gl.` and
  `moderngl.get_context()` instead.
- `grep "finish\|flush"` surfaces `scripts/dogfood/drive.py:100 sys.stdout.flush()` and
  `scripts/dogfood/harness.py:262 self.station.flush()` — both are unrelated I/O flushes, not GL
  sync. The only real GL-sync site is `ui.py:526 gl.finish()`.
- `grep "\.error\b"` surfaces the `scripting/engine.py` script-error dict (`self.errors[key] =
  behavior.error`) and copilot's `formatting.py`/`ui_models.py` error fields — all unrelated to GL
  error state. The one real GL-error-state site is `ui.py:507 moderngl.get_context().clear_errors()`.
- `grep "query("` initially returned nothing under a `ctx.query(` pattern; the real call is
  `profiling.py:304 self._gl.query(time=True)` — same `gl`-not-`ctx` naming trap.
- `shaderbox/resources/editor/abi_probe.py` looked GL-adjacent by path (`resources/editor/`) but
  is a pure ctypes ABI-shape probe for the vendored editor binary — no GL at all.
- `scripts/dogfood/harness.py`'s `moderngl.create_standalone_context(backend="egl")`
  (`harness.py:192`) looked like a second-context precedent worth citing verbatim for 090 — it IS
  useful precedent for "a thread can own its own live GL context" — but it is a **standalone**
  (unshared) context for a headless, glfw-less, imgui-less harness process, not a shared context
  alongside a live window. It proves the mechanics of `create_standalone_context` work on this
  driver stack; it does NOT exercise context SHARING, which is 090's actual requirement. Flagged
  so it isn't mistaken for a solved instance of the same problem.
- `popups/lib_picker/preview.py` sounds like a shader preview (renders something) by name — it is
  actually a text/metadata preview panel for the shader-lib browser (function signature + doc
  comment), entirely GL-free.
- `tabs/share.py`'s and `exporters/telegram.py`'s texture-presenting sites
  (`preview.texture.glo`, sticker thumbnails) look like document-render sites by pattern match
  (`.texture.glo` is the same shape as every document preview) but are actually `Image`/`Video`
  objects built from FILES the render already finished writing — no live document/pass dependency,
  so they don't need to move to the render thread at all.
- `grep "\.use(""` also matches `moderngl`-unrelated `.use()`-shaped calls in imgui/pydantic
  contexts in principle, but none appeared in this codebase — the six real hits
  (`channel_blit.py:67/69`, `core.py:496/536`, `ui.py:510`, `editor/render.py:214/249`) were all
  genuine GL binds.

## 7. Coverage statement

**Read in full** (every line, via the `Read` tool): `ai_docs/dev_flow.md` (Module map section),
`CLAUDE.md`, `shaderbox/core.py`, `shaderbox/document.py`, `shaderbox/ui.py`, `shaderbox/media.py`,
`shaderbox/channel_blit.py`, `shaderbox/profiling.py`, `shaderbox/editor/render.py`,
`shaderbox/copilot/bridge.py`, `shaderbox/render_defer.py`, `shaderbox/render_job.py`,
`shaderbox/render_shape.py`, `shaderbox/exporters/worker.py`, `shaderbox/exporters/base.py`,
`shaderbox/exporters/registry.py`, `shaderbox/tabs/share_state.py`, `shaderbox/tabs/render.py`,
`shaderbox/widgets/media_ops.py`, `shaderbox/tabs/share.py`, `shaderbox/util.py` (the
`try_to_release` helper), `shaderbox/watch.py`.

**Read the relevant sections/functions of** (targeted `Read`/`sed -n` after a grep hit, not the
whole file): `shaderbox/app.py` (init through line ~260; `_init`/`release`/`shutdown`/
`switch_project`/`request_project_switch`/`new_project`/`delete_project` around lines 1150-2072),
`shaderbox/project_session.py` (lines 380-620: `save_ui_document`, `_delete_document_unguarded`,
`load`, `_resolve_scripts`, `sync_documents_from_disk`, `_load_one_document_from_disk`),
`shaderbox/copilot/backend.py` (header docstring, `_probe_frame`/`_probe_target_for` context,
`create_document`/`delete_document`/`duplicate_document` bodies), `shaderbox/copilot/tools/*.py`
(structural grep of every `def`/`run_on_main` line — confirmed thin-dispatcher shape, no direct
GL), `shaderbox/exporters/telegram.py` (render-button + sticker-grid + release sections, lines
~580-700 and ~840-875), `shaderbox/exporters/youtube.py` (render-panel + shape-matching sections,
lines ~380-525), `shaderbox/tabs/code.py` (lines 990-1070: the editor-panel draw call site),
`shaderbox/ui_primitives.py` (`centered_image`, `preview_cell` signatures + docstrings),
`shaderbox/widgets/document_grid.py`, `shaderbox/widgets/pass_list.py`,
`shaderbox/widgets/uniform.py` (grep-located texture/write sites, read in context),
`shaderbox/popups/pass_settings.py`, `shaderbox/popups/projects.py` (grep confirmed no direct
GL — pure dispatch to `App.request_project_switch`), `shaderbox/copilot/session.py` (thread-spawn
+ bridge-ownership lines), `scripts/dogfood/harness.py` (EGL context creation + thread-ownership
comment, lines 180-245), `scripts/smoke.py` (grep-located GL/glfw lines).

**Grepped only** (pattern match across `shaderbox/` + `scripts/` for `ctx.`/`gl.`/
`moderngl.get_context()`/`.release(`/`.render(`/`.write(`/`.read(`/`.use(`/`glfw.`/`imgui.image`/
`add_image`/`finish`/`fence`/`flush`/`query(` and structural `def `/`class ` markers) but not
read in full: `shaderbox/intel/worker.py` (confirmed GL-free by grep — no `moderngl`/`GL` hits),
`shaderbox/popups/lib_picker/preview.py` (confirmed GL-free by grep), `shaderbox/scripting/*.py`
(grepped for `.error` false trail only — not otherwise inspected, no GL import in this
subpackage per the module map's own description), every other file in `copilot/`, `tabs/`,
`widgets/`, `popups/`, `shader_lib/`, `exporters/` not named above (grepped for the full pattern
set, zero GL hits returned, not individually opened).

**Not examined:** test files (`tests/`) — out of scope per the task (call sites in the shipping
app + scripts only); `shaderbox/resources/` binary/data assets.
