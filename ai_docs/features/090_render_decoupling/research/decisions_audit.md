# 090 — locked-decisions audit

Research only. Read per the task: `CLAUDE.md`, `ai_docs/conventions.md` (full), `ai_docs/todo.md`
(empty of live entries — see §3), `ai_docs/dev_flow.md` (Feature flow / Module map / `make smoke` /
`make test`), `088_frame_profiler/01_spec.md`, `084_project_management/01_spec.md` D5,
`068_radiance_cascades/01_spec.md`, `089_ninth_walk_findings/01_spec.md` D4-D6, `066_perf_and_test_diet/01_spec.md`,
`081_copilot_engine_sweep/01_spec.md`, `075_dogfood_station/01_spec.md`, `.claude/skills/imgui-ui/SKILL.md`,
plus the source of every mechanism a render thread touches (`shaderbox/copilot/bridge.py`, `gate.py`,
`shaderbox/render_defer.py`, `shaderbox/watch.py`, `shaderbox/ui.py::_tick_frame_state`,
`shaderbox/app.py`'s GL-context creation, `shaderbox/exporters/worker.py`, `shaderbox/intel/worker.py`).
No `040_*` spec exists in the tree (the module map's scripting section is the record for that engine);
no `066_*` markdown other than the one spec file exists.

---

## 1. Decisions/quirks/triggers a render thread interacts with

| Source (file + section) | How 090 interacts | Consequence for the design |
|---|---|---|
| `conventions.md` — *"Thread/GL affinity is enforced by METHOD ownership, not import boundaries; cross-thread reactions are injected callbacks."* | **Is the reason for a constraint, and 090 must explicitly reverse its premise.** Today's law is "GL objects live with the render thread [= main thread]; a worker thread never touches moderngl." 090 makes GL objects live with a NEW dedicated render thread instead, so every existing worker (copilot, exporters, intel) that currently treats "the render thread" as synonymous with "the main thread" needs that assumption re-stated, not just the code re-pointed. | The bullet's literal sentence survives (GL still lives with "the render thread"); its unstated premise (render thread == main thread == the thread glfw/imgui also run on) does not. This is the single biggest doc-debt item: the phrase needs updating in the same wave or every future reader re-derives the wrong assumption. |
| `conventions.md` — the `CopilotBridge`/`GateChannel` bullet: *"A worker↔main blocking primitive that latches `_shutdown` on `release()` MUST expose `reopen()`… Its teardown contract: `cancel_all()` … BEFORE `join(timeout)`… so the worker thread is **daemon**."* | **Honors as-is, and is the template to extend or mirror.** The copilot worker already blocks on a main-thread GL round-trip through exactly this primitive. If 090 keeps the copilot backend calling into GL through a queue, that queue's target moves from "the main thread" to "the render thread" — a third leg on an already worker↔main shape, or a re-aim of the existing one. | Whatever shape 090 picks for worker(copilot)→render-thread GL calls, it must carry the whole bundle this bullet demands: `reopen()`, daemon thread, cancel-before-join, abandon-on-timeout. Skipping any piece reproduces a bug this repo already paid for once (the dropped-`reopen()` bug named in the bullet). |
| `shaderbox/copilot/bridge.py` — `CopilotBridge`, and its own comment: *"GL has thread-affinity (only the main thread owns the context); a worker-thread tool sometimes needs a GL result mid-turn."* | **Is the reason for a constraint, and 090 must explicitly reverse it.** The bridge's whole justification is "only the main thread owns the context." 090 puts a second context (on the render thread) into existence, sharing GL objects with the main thread's imgui context. The bridge's rationale comment becomes false the moment a second GL-capable thread exists — even if the copilot still routes through some "the-thread-that-owns-the-document" indirection. | Bridge either gets re-aimed at the render thread (its `run_on_main` becomes `run_on_render`, its `defer`/`run_deferred_render` post-swap contract has to be re-derived against a swap that no longer happens on the same thread as the copilot's caller) or copilot's document mutations get re-routed to go through the render thread's own request queue instead of a bespoke bridge. Either way this is a "must extend" item, not a "leave alone" one — silently forking a second, subtly different bridge is the trap conventions.md's parallel-dict bullet warns against generically. |
| `conventions.md` — *"A pre-freeze repaint needs `gl.finish()`… EVERY render encode shares ONE post-swap firing point (`ui.py::update_and_draw`, after `swap_buffers`)… A NEW render entry point MUST route its encode here, never call it inline."* | **Must explicitly reverse or re-derive.** This law assumes ONE thread does both the swap and the encode, so a "present-before-freeze" guarantee can be enforced by ordering within one call stack. Once `Document.render` runs on a separate thread from `glfw.swap_buffers`, "after swap_buffers" is no longer a program-order fact on the same thread — it becomes a cross-thread ordering/synchronization question. | This is one of the sharpest open contradictions (see §5): the funnel this bullet mandates cannot exist in its current form once render and swap are on different threads. 090's spec must state the NEW funnel explicitly (a documented cross-thread happens-before, e.g. a fence or a "wait for render thread's frame N to be visible" primitive) rather than silently dropping the guarantee for exports/renders that now originate off the render thread. |
| `shaderbox/render_defer.py` + its use in `ui.py` (the Render tab / Share tab / `CopilotBridge`'s `defer=True`) | **Must extend.** `RenderDefer` is the one-frame "hold a request until the cue has painted" latch that exists BECAUSE encode blocks the frame loop on the swap-owning thread. If encode moves to its own thread, the reason `RenderDefer` exists (a synchronous main-thread block) goes away for that caller, but the mechanism itself — "don't fire until the user has SEEN the cue" — still needs an answer once the thing painting the cue (main thread/imgui) and the thing doing the rendering (render thread) are different actors. | 090 needs an explicit decision: does export/render still park behind a one-frame visible-cue latch, now implemented as a cross-thread signal, or does the whole "freeze the loop" problem disappear because the render thread never blocks the main thread's frame pump at all (the actual point of 090)? If the latter, `RenderDefer` and the bridge's `defer`/`_deferred_render`/`run_deferred_render` machinery may become **dead code to explicitly retire**, not something to port forward unexamined. |
| `088_frame_profiler/01_spec.md` D2/D3 — *"GPU spans are `GL_TIME_ELAPSED` queries in a three-deep ring keyed by PATH, read two frames late"*; *"the profiler reaches `Document.render` as an explicit parameter; the loop passes `app.profiler`."* | **Must extend.** The profiler's entire mechanism (one `Profiler` object, threaded down as a parameter, GPU queries opened/read against ONE `moderngl.Context`) assumes a single GL context and a single-threaded frame cadence (`begin_frame`/`end_frame` called once per iteration of one loop, spans opened and closed on that same thread). | If `Document.render` moves to a render thread with its own shared GL context, the profiler's queries (`ctx.query(time=True)`) must be issued on THAT context/thread, but `Profiler`'s ring-read side (`begin_frame` reading 2-frames-late slots) currently lives on the same call stack as the spans it reads. A render-thread `Document.render` now needs either its own `Profiler`/ring per render thread (breaking D3's "one profiler object threaded down" simplicity and D6's "dynamic nesting is free" property) or a cross-thread handoff of completed `FrameProfile`s to the main thread's overlay — which is a NEW seam 088 never anticipated. The "read two frames late" cadence is ALSO built on "one document render = one point in one frame's sequence"; a worker thread rendering N documents at its own pace, decoupled from imgui's frame cadence, breaks the notion of "frame N" the ring's `N % 3` indexing depends on. |
| `088_frame_profiler/01_spec.md` D2 — *"`gpu(...)` asserts no GPU span is open… nested query… a silent wrong number"* | **Constraint 090 must honor as-is if any GPU profiling survives on the render thread.** Two GL contexts sharing objects does not change the fact that ONE context's queries still cannot nest. | If 090 lets the render thread AND the main thread each hold profiler-relevant GPU queries against their own contexts, that's fine per-context, but if any design accidentally issues overlapping queries against the SAME shared context from two threads, this assert (and the underlying GL rule) fires or silently corrupts — GL command streams from two threads against one context are themselves the deeper hazard (see the shared-context row below). |
| `084_project_management/01_spec.md` D5 — *"The switch is DEFERRED to the top of the next frame, never run inside the modal's draw… Releasing textures from inside the popup would leave the draw list holding freed GL names… `release()` freeing every GL object is safe [only at the top of the frame, before any texture push]."* | **Is the reason for a constraint 090 must honor or explicitly reverse.** This is the imgui-texture-handle-lifetime rule the task calls out by name. It says GL objects (specifically texture handles pushed into imgui's draw list via `add_image`) can only be released/recreated at a point in the frame BEFORE any of that frame's draw calls have referenced them — i.e., synchronously, on the thread that also builds the draw list. | If document rendering (which owns the canvas/output textures the imgui draw list references every frame) moves to a worker thread, the "newest completed frame" texture handle the main thread displays must still obey this rule: the main thread can only swap which texture handle it's displaying at a point BEFORE it has pushed that frame's `add_image` calls, and the render thread must never delete/recreate a texture the main thread is mid-draw-list-referencing. This generalizes D5's single instance (project switch) into a standing invariant 090 must design for explicitly: a texture handle handed from render-thread to main-thread needs a lifetime contract (double-buffering / generation-tagging) so the main thread never reads a `glo` the render thread has since freed or reallocated (e.g. on a canvas resize). |
| `.claude/skills/imgui-ui/SKILL.md` §9 — *(implicitly, via the general imgui-bundle GL/thread rules; no explicit multi-thread-texture section exists)* | **False trail flag, see §6.** The skill has extensive texture/image handling guidance (image-in-child-sizing, `image_with_bg`, tint loss) but nothing about cross-thread texture handle lifetime — that guidance lives only in `conventions.md`'s 084 D5 bullet, not the imgui skill. | Don't expect the imgui-ui skill to carry the thread-safety answer; it's a UI-authoring skill, GL-thread-safe from imgui's own single-threaded assumption. 090's design doc needs its OWN section on texture handle handoff; it cannot point at the imgui skill for this. |
| `conventions.md` — *"An unfilled pass input reads BLACK… A pass compiles when something first NEEDS its program — never at load (feature 066)."* + `ui.py::_tick_frame_state`'s render-set comment (*"The frame's render set (066 D2): the current document; with 'Render all' on, every document that already had its first render; plus AT MOST ONE not-yet-rendered document."*) | **Must extend — this IS the render-set rule the task asks about**, and it does not live in `066_perf_and_test_diet/01_spec.md`'s D2 text verbatim (that spec's D2 is about bounding frame-0 compile cost); the CURRENT, fuller rule is implemented directly in `ui.py::_tick_frame_state` and only cites "066 D2" in a code comment. | This is the single mechanism 090 relocates. Its two invariants — "at most one first-render admitted per frame" (a compile-cost budget spread across imgui's frame cadence) and "the tick/render set exactly matches what `session.tick`'s scripted uniforms were computed for" — were both designed assuming ONE thread ticks scripts, decides the set, and renders it inside one frame. A worker thread rendering at its own cadence, decoupled from imgui frame boundaries, needs an explicit re-derivation of both: what throttles compile cost once it's not bounded by "one imgui frame tick", and how script-tick/render-set coherence (`Document.begin_frame(frame)` — see below) is preserved across two independently-paced loops. |
| `conventions.md` — *"A document is N passes… feedback swaps at the FRAME boundary (`Document.begin_frame(frame)`)… `begin_frame` takes the frame NUMBER and is idempotent within one [call]… Revisit if a pass ever needs to draw more than once per frame."* | **Must extend / is a direct hazard.** `begin_frame(frame)`'s idempotency is keyed on a caller-supplied integer "frame number" that today is `app.frame_idx`, incremented once per imgui loop iteration on the main thread. `_tick_frame_state` calls `begin_frame` for exactly the tick set, once, on the same thread that then calls `document.render()` for the same set later in the same function. | If rendering moves off the main thread, "frame number" needs a definition that the RENDER thread owns and increments at ITS cadence (which may differ from imgui's), and `_tick_frame_state`'s script tick (which computes uniform values FOR a specific tick set, on the main thread today via `session.tick`) must stay coherent with whichever frame the render thread is about to draw — script tick and document render are currently coupled by both running inside the same function call on the same thread. Splitting them across threads reopens the exact class of bug 065's design law calls out ("correctness by call COUNT was unfalsifiable") unless the new frame-number/tick-set handoff is made an explicit, testable contract. |
| `conventions.md` — *"The live path ticks once (`session.tick` in `ui.py`)… export ticks a FRESH per-export instance… isolation is STRUCTURAL: `Document.render_media` enters an injected `Document.export_isolation` factory."* | **Honors as-is, but is a load-bearing precedent for isolation design.** The script engine already has a working "isolate this render's script state from the live loop's" mechanism structurally enforced at the ONE funnel (`render_media`), not per-caller. | If 090 needs the render thread to render a document without stepping on the live (main-thread-ticked) script state — e.g. for the current document's live preview vs. an export the render thread is also asked to do — `export_isolation` is the existing precedent for how that isolation is achieved (a factory, not a flag) and should be the template rather than a bespoke render-thread/live split. |
| `conventions.md` — *"A live moderngl context must exist before constructing `Image`/`Video`/`Font`/`Canvas`/`Document`… In the app, `glfw.make_context_current(window)` handles it."* | **Must extend.** This is a per-THREAD fact in OpenGL (a context is current on at most one thread at a time), stated here as if "the app" has one context. A render thread needs its OWN context made current on ITS thread, sharing object namespaces with the main thread's context (the standard GL shared-context idiom) — moderngl/glfw support this (`glfw.create_window(..., share=main_window)`), confirmed already exercised in the 090 probe scripts (`probes/common.py::make_window(..., share=None)` takes a `share` param). | Every constructor site that currently assumes "the" context is current (Document/Pass/Canvas/Image/Video/Font construction) must be audited for WHICH thread calls it — construction on the wrong thread against the wrong current context is a silent-wrong-object bug class, not a crash, matching the repo's general "silent wrong number, not an exception" GL-hazard pattern (088's nested-query lesson, the `texture.read()` f2 lesson). |
| `conventions.md ## Known quirks` — *"Read a render target through `media.texture_to_rgba8`, never `texture.read()[0]`… A raw read returns the texture's OWN bytes… on an `f2` target the first 'pixel' is half of one float16 channel."* | **Honors as-is.** Orthogonal to threading — a data-format footgun, not a thread-affinity one. | Any NEW code 090 adds that reads pixels back (a render-thread readback for hand-off to the main thread's display, or for export) must still go through `texture_to_rgba8`, not `texture.read()[0]`. Worth restating because a render/export path is exactly where this quirk has bitten three times before. |
| `conventions.md` — *"Only ONE test per process may drive `ui.update_and_draw`"* (the imgui-context-is-per-process quirk) | **Must extend to the test-writing plan, not the runtime design.** A per-PROCESS singleton (imgui context, and by the same logic the process's set of live GL contexts) constrains how 090's tests can be structured. | Any new test suite for a render thread that itself spins up glfw windows/contexts inherits this same "only one App-level thing per process" hazard — worth flagging for the verification-design step (dev_flow.md step 7), not the architecture itself. |
| `dev_flow.md ### make smoke` — *"headless smoke test… runs ~200 frames of `update_and_draw`… catches import errors, callback dispatch failures, popup state-machine crashes, released-texture binding errors."* | **Must extend.** Smoke is explicitly described as catching "released-texture binding errors" — exactly the hazard class a render thread introduces (main thread reading a texture handle the render thread has released/reallocated). | Smoke's headless, single-threaded, ~200-frame drive is the natural home for a regression test of the new cross-thread texture-handle contract, but it currently drives everything on ONE thread/process; 090 needs to decide whether smoke gains a second thread (matching production) or stays a structural/inline simulation — an inline simulation would NOT exercise the actual race, so per the profiler's own "gate must be broken to prove it catches something" law, a smoke-only check is likely insufficient here (see §5). |
| `dev_flow.md ### make test` — *"the three env vars… are load-bearing… `MESA_GL_VERSION_OVERRIDE`… Each xdist worker is its own PROCESS with its own glfw window and GL context, which is what makes this safe where `pytest-forked` is NOT: a forked child inherits the parent's open X11 socket."* | **Honors as-is / is a hazard to note.** This documents that even PROCESS-level GL context sharing across a fork is unsafe on this stack (X11 socket sharing breaks it) — the reasoning is adjacent to, but distinct from, in-process THREAD-level context sharing 090 introduces. | Not a direct constraint on 090's design, but establishes that this codebase's GL/X11 stack is known-fragile under naive concurrency; any new thread's context creation needs the same care already documented for `-n 8` xdist workers (each worker's own window+context, never shared blindly). |
| `conventions.md ## Code rules` — *"Imports at module top only — never inside function bodies"* + `conventions.md`'s *"Heavy SDKs import lazily behind exactly two seams"* (the only two sanctioned exceptions) | **Must honor as-is.** A new `shaderbox/render_thread.py` (or wherever the worker lives) imports `moderngl`/`glfw`/`document` at module top like every other module; it does NOT get a third lazy-import seam. | Purely a code-rule constraint, not a design one — flagged because the task explicitly asked about it. No tension found: nothing about a render-thread module needs a lazy import (unlike `openai`/google-auth, `moderngl`/`glfw` are already always-imported, cheap, and load-bearing from frame 0). |
| `conventions.md ## Code rules` — *"No `if TYPE_CHECKING:`… Circular imports is a sign of a bad design"* + `conventions.md`'s "Three-layer UI architecture" bullet (*"The split is forced by the no-`TYPE_CHECKING` rule: a draw fn annotating `app: App` while `App` imports it would cycle."*) | **Is the reason for a constraint 090's module boundary must honor.** `dev_flow.md`'s size-preamble explicitly calls out *"the cycle-from-types signal: if a new module needs `app: App`… the no-`TYPE_CHECKING` rule will force a structural split — anticipate it in the spec."* | A render-thread module cannot import `App` if `App` needs to import IT (to hold a handle/queue to the thread) — which it will, symmetrically to how `ProjectSession` and `CopilotBackend` avoid importing `App` today. The precedent (`project_session.py`, `copilot/backend.py`, `intel/worker.py` — none import `App`; they take explicit deps/getters/callbacks) is the template: the render-thread module must be a LEAF or near-leaf taking `Document`/`Canvas`/queues as explicit params, never `App`, with `App` holding the thread handle and forwarding via `@property`/callbacks the way it does for `self.session`. |
| `conventions.md` — *"`ProjectSession` is the headless project + copilot core; `App` owns one and forwards to it… creates no glfw window and no imgui context at import — so a headless harness (feature 026) constructs it on a standalone EGL context without `App`."* | **Constrains where the render thread's owning object should live.** `ProjectSession` is already the "headless-safe" home for engine-adjacent state (documents, the script engine). A render thread that owns `Document.render` calls is engine-adjacent, not UI-adjacent. | Candidate: the render thread's lifecycle (start/stop/queue) is owned by `ProjectSession`, not `App` directly — mirroring how the script engine and copilot cluster are already there — UNLESS the render thread's GL context needs to be created relative to the glfw window (which only `App` constructs), in which case `App` must own context CREATION (it already owns the one window) while `ProjectSession` or a new leaf owns the render loop's document-side logic. This is a genuine open design question (see §5), not a settled one. |
| `conventions.md` — *"`popups/pass_settings.py`… what a pass reads is not here since 072: that is each sampler's row"* + `pass_graph.py`'s planner (`plan_passes`/`evaluation_order`/`assert_plan_invariants`) — *"evaluation is memoized — a shared ancestor draws once per frame, never once per consuming path."* | **Must honor as-is.** The per-frame memoization invariant (`assert_plan_invariants` runs inside `evaluation_order`, the function that actually draws) is a correctness guarantee independent of which thread calls `render()` — it is scoped to ONE call of `Document.render` for ONE frame. | No conflict: as long as the render thread still calls `Document.render` once per (document, frame) the way the main thread does today, this invariant holds unchanged. Only a hazard if 090's design lets TWO threads race to render the same document's frame (main thread's preview render vs. render thread's own) — which would violate the "shared ancestor draws once per frame" guarantee across threads, not just within one call stack. Flag as a design constraint: only ONE thread may ever call `render()` on a given `Document` for a given frame number. |
| `shaderbox/intel/worker.py` docstring — *"The one thread that talks to jedi… jedi is not safe to call from two threads at once… A request carries the editor revision and cursor it was made for; the reader drops a result whose stamp is no longer current."* | **Existing pattern to match or consciously differ from** (see §2). This is the repo's template for "single-purpose worker, request replaces pending-of-same-kind, results carry a staleness stamp the consumer checks." | Directly transferable pattern for a render thread: if the render thread accepts "render this document" requests faster than it can service them, `intel/worker.py`'s "the newest request of each kind replaces an older one still waiting" + revision-stamped staleness check is the exact shape for "only the newest completed frame per document matters," which is literally the feature's stated goal ("the main thread... displays each document's newest completed frame"). |

---

## 2. Existing thread inventory

Grepped `threading.Thread(`, `Thread(target`, `import threading`, `queue.Queue` across
`shaderbox/` and `scripts/` (excluding tests). Four threads exist today, all daemon, none touching
GL from off the main thread except via a blocking round-trip:

1. **`shaderbox/intel/worker.py::PythonWorker`** (thread name `shaderbox-intel-python`) — the one
   thread that calls jedi. Started in `__init__`, `daemon=True`. Hand-off: `submit()` pushes into a
   `dict[kind, Request]` "latest wins" slot (not a FIFO — a `threading.Condition` wakes the worker;
   a new request of the same KIND overwrites the pending one), the worker computes and pushes a
   `PythonResult` onto a `queue.Queue`; the main thread's `poll()` drains everything since the last
   poll. Staleness handled by the CONSUMER (the reader checks `PythonRequest.matches(path, revision,
   line, column)` before trusting a result — no server-side cancellation). Teardown: `close()` sets a
   flag + notifies; no explicit `join`/timeout visible in this file (the shutdown call site was not
   re-derived here — out of scope for this audit's read list).

2. **`shaderbox/copilot/session.py::CopilotSession`** (thread name `copilot-worker`) — runs the
   agent loop (`run_turn`) that drives the LLM + tool calls. Started lazily via `_ensure_worker()`
   (spawn-if-not-alive, matching `ExporterWorker.ensure`'s idiom). `daemon=True`, with the SAME
   documented reason as `exporters/worker.py`: *"a worker blocked in a stalled stream past
   release()'s join timeout is ABANDONED."* Hand-off is NOT a simple queue: the worker calls back
   into `CopilotBackend` methods, and any GL-affine one of those marshals synchronously through
   `CopilotBridge.run_on_main` (queue + `threading.Event`, blocking the worker until the main thread's
   per-frame `drain()` services it) or blocks on `GateChannel.ask()` for a user-facing confirm/credential
   round-trip. Teardown: `self._worker.join(timeout=COPILOT_ENGINE.worker_join_timeout_s)`, abandon on
   timeout (matches the documented contract).

3. **`shaderbox/exporters/worker.py::ExporterWorker`** — the shared worker-thread machinery behind
   `TelegramExporter`/`YouTubeExporter`. Generic over `Job`/`Event`; `ensure(run)` lazily spawns
   (`daemon=True`, name `f"{label}-worker"`), `submit(job, run)` enqueues (bounded `queue.Queue`,
   drop-on-full with a warning), `poll_event()` drains a SEPARATE bounded, LOSSY-NEWEST progress
   queue (oldest evicted on full, so an `in_flight`-clearing event is never the one dropped). Explicitly
   MUST NOT touch moderngl (`Exporter` ABC's thread-affinity split — worker-thread methods
   `prepare`/`export`/job handlers see only `RenderedArtifact`, a GL-free value type; only
   render-thread [= main thread] methods may touch GL). Teardown: `stop()` pushes a generation-stamped
   `_Stop` sentinel, `join(timeout=DRAIN_TIMEOUT_SEC)`, returns the thread (for the caller to abandon)
   on timeout rather than blocking forever — the STOP-generation stamp exists specifically so an
   abandoned worker's leftover STOP doesn't kill the NEXT worker.

4. **`shaderbox/copilot/bridge.py::CopilotBridge`** and **`shaderbox/copilot/gate.py::GateChannel`**
   — not threads themselves, but the two worker→main blocking-round-trip PRIMITIVES every GL-affine
   or user-facing copilot call goes through. `CopilotBridge`: worker calls `run_on_main(fn, timeout,
   defer)`, which enqueues a `MainThreadOp` (a closure + `threading.Event` + result/error slot) and
   blocks on the event; the main thread's `drain()` (called at the TOP of `_tick_frame_state`, bounded
   to 8 ops/frame) runs the closure and sets the event, OR — if `defer=True` — parks it for
   `run_deferred_render()` to fire strictly after that frame's `swap_buffers` + `gl.finish()` (the
   render-defer/cue-visibility guarantee). `GateChannel`: same shape for user-facing confirm/credential/
   file-picker prompts, with a `_generation` counter serializing `cancel_all()`'s release sweep against
   a request published concurrently (closing a narrow race where a slot published just after a sweep
   would wait forever). Both: `cancel_all(reusable=...)` releases every blocked waiter BEFORE any
   `join`, `reopen()` clears a non-reusable shutdown latch, and `App._init` calls `release()` then
   re-arms via `reopen()` in `enqueue_turn` — the exact contract `conventions.md`'s bullet documents.

**Non-thread pattern worth matching**: `shaderbox/watch.py` (`reload_document_if_changed`,
`maybe_rebuild_lib_index`) is a main-thread-only, per-frame poll+react pattern (mtime checks, no
thread at all) — `conventions.md`'s worker-thread bullet calls it out explicitly: *"a worker that
must touch GL after a file write rides the watcher rather than inventing a queue."* This is the
"free lunch" precedent for NOT building a new cross-thread mechanism where a per-frame poll already
suffices; worth checking whether any piece of 090's hand-off (e.g. "is a new frame ready?") can be a
poll rather than a queue.

**What the pattern establishes, as a checklist for a render thread**: (a) daemon thread, named; (b)
lazy or eager spawn decided by whether the work exists before first use; (c) hand-off queues are
bounded, with an explicit policy for full (drop-newest-job vs. drop-oldest-progress vs. block); (d)
staleness is either prevented by "latest-wins" submission (`intel/worker.py`) or checked by the
consumer against a stamp (revision, generation); (e) teardown always pushes a release/stop signal,
joins with a bound, and ABANDONS the thread past that bound rather than block shutdown, which is
only safe because the thread is daemon; (f) GL-affine work is either done exclusively by the
thread that owns the context (exporters: never; copilot: marshaled via bridge) — no existing thread
does GL work directly today, so 090's render thread is the FIRST to break that "only main thread
touches GL" invariant outright rather than route around it, which is exactly why the task frames
this as touching "the App's ownership of GL objects."

---

## 3. `todo.md` triggers

`ai_docs/todo.md` is FROZEN drain-only (declared 2026-07-27) and currently contains **zero live
entries** — the file is header/instructions plus a closing `---` with no `## [BUG]`/`## [DEBT]`
blocks. Grepped `^## \[` and found nothing. There is therefore nothing to fire a Trigger against; no
todo.md entry constrains or informs 090. (This itself is worth noting to the maintainer only as a
negative result, not as a gap — the file's own header says the goal is zero entries, and it is at
zero.)

---

## 4. Sizing per `dev_flow.md`'s preamble

`dev_flow.md`'s **Size preamble** names three bands: **Mid** (default: multi-file, new module, real
behavior change → 1-2 pre-impl + 2-3 post-impl reviewers), **Tiny/small** (≤3 files, 1 module, pure
code, no new public API, no async/lifecycle → 0-1 reviewers), and **High-blast-radius** (a
refactor across many modules, anything touching conventions → upper-mid or beyond, extra reviewers,
a sanitization sweep even if not otherwise warranted), with an explicit named symptom: *"if a new
module needs `app: App`… the no-`TYPE_CHECKING` rule will force a structural split — anticipate it
in the spec."*

**090 is High-blast-radius**, on the preamble's own criteria, not a judgment call at the margin:

- It is explicitly **async/lifecycle** — the preamble's Tiny/small band names "no async/lifecycle"
  as a DISQUALIFIER, and 090 is nothing but async/lifecycle (a new thread with its own GL context,
  its own frame cadence, and a teardown contract).
- It touches **more modules than any single feature audited above**: `document.py` (the render
  entry point + `begin_frame`), `app.py` (GL/window ownership), `ui.py` (`_tick_frame_state`, the
  render-set rule, the profiler's frame root), `render_defer.py`, `copilot/bridge.py` (and by
  extension `copilot/backend.py`'s ~30 `run_on_main` call sites), `profiling.py` (088's whole
  design), `watch.py`, every exporter's render path, and the smoke/test harness's GL-context
  assumptions (`make test`'s per-worker-process contexts, `make smoke`'s single-thread drive).
- It **touches conventions.md directly** — at minimum the "Thread/GL affinity… GL objects live with
  the render thread" bullet, the render-defer post-swap-funnel bullet, and 088's D3 (profiler as an
  explicit parameter reached from one thread) all need to be revised or explicitly superseded, which
  is the preamble's named trigger for scaling up ("anything touching conventions").
- It changes a **structural invariant three other locked features were built assuming** (084 D5's
  texture-handle-lifetime timing, 088's whole profiler design, 089 D4-D6's save-funnel cost
  accounting) — a "refactor with blast radius" in the preamble's own Feature-vs-small-change table
  language ("unblocking the render loop" is literally the preamble's own example of a
  feature-flow-triggering refactor).

**Review shape this demands**: the upper end of mid or beyond — at least 2 pre-implementation
reviewers (one on GL/thread-safety correctness specifically, one on convention fidelity across the
touched specs) plus a **spec-fidelity audit** at post-implementation (the preamble's named addition
for high-blast-radius: "walks the spec end-to-end against the diff — every locked decision actually
landed"), escalating to a **swarm convergence loop** per the post-impl-review section's own
escalation rule (*"for high-blast-radius diffs… escalate to a larger parallel swarm run as a
convergence loop"*), and a full **sanitization sweep** regardless of whether it would otherwise be
warranted (the preamble says so explicitly for this band). Given the GL-timing hazards involved
(silent-wrong-number classes, not crashes — the profiler's nested-query lesson and the
`texture.read()` f2 lesson both establish this repo's GL bugs fail silently), a reviewer role
dedicated to "does this actually break under load, not just on paper" is warranted — the existing
090 probe scripts (`probes/gil_probe.py`, `probes/probe_a_*`, `probes/calibrate_heavy.py`) are
already this repo's own precedent for measuring rather than reasoning about GPU/thread behavior, and
the spec should demand the same empirical standard for the shipped design (mirroring 088's own
pre-implementation reviewer, who re-measured the ring depth under real load and overturned the
spec's first-draft number).

---

## 5. Open contradictions — for the maintainer

1. **The post-swap render-encode funnel vs. a render thread that doesn't share the main thread's
   swap call.** `conventions.md`'s law is *"EVERY render encode shares ONE post-swap firing point…
   after `swap_buffers`… A NEW render entry point MUST route its encode here."* This assumes one
   thread does both the swap and the encode in program order. Once `Document.render` runs on a
   worker thread, "after swap_buffers" isn't a same-thread fact anymore. Does 090 (a) keep exports/
   copilot-renders as main-thread-only work that still routes through the existing funnel (the
   render thread handles only the LIVE per-frame document draws, nothing else), or (b) move export
   encoding onto the render thread too, which then needs an explicit cross-thread "the cue has been
   presented" signal the main thread sends the render thread before it's allowed to start? These are
   different architectures with different blast radii, and the spec needs to say which.

2. **Where does the render thread's GL context get created, and who owns teardown?** `App.__init__`
   is the one place `glfw.create_window`/`make_context_current` happens today, and `App.shutdown`
   (not `release()`) is the one sanctioned place the imgui context/window dies, per the
   "imgui CONTEXT is per PROCESS" quirk. A shared GL context for the render thread needs its own
   glfw window (invisible, sharing the main window's context per `glfw.create_window(..., share=...)`)
   — does that living object belong on `App` (which already owns the window) or on `ProjectSession`
   (which is the headless-safe, engine-adjacent home per the module-map's own logic)? The two
   existing precedents point opposite ways: `App` owns the ONLY current context-creation code; but
   `ProjectSession`'s whole reason for existing is "no glfw window/imgui context, so a headless
   harness can construct it" — and the render thread's job (render documents) is exactly
   `ProjectSession`'s domain, not `App`'s UI domain.

3. **Does the profiler run on the render thread, the main thread, or both — and what does "two
   frames late" mean when two threads have independent cadences?** 088's ring-read design assumes
   ONE cadence (`N % 3`, read at `N+2`) tied to imgui's frame loop. If the render thread renders
   documents at its own pace (potentially faster or slower than imgui's frame rate, which is the
   whole point of decoupling), "frame N" for GPU-query purposes needs a NEW definition scoped to the
   render thread's own loop, and the main-thread overlay (which reads `app.last_profile`) needs a
   cross-thread hand-off of completed profiles that 088 never designed for. This is not a small gap:
   088 is the newest, most carefully measured spec in the repo, and its central mechanism (a
   thread-local GL query ring keyed to one loop's frame count) is the piece most directly upended by
   090's premise.

4. **Is `_tick_frame_state`'s render-set rule (current + render-all + at-most-one-pending-first) a
   main-thread SCHEDULING decision or a render-thread one?** The rule exists to bound COMPILE cost
   spread across imgui's frame cadence (066's stated motivation: "admitting first renders one per
   frame bounds the frame cost instead of stalling frame 0"). Once rendering doesn't block the main
   thread's frame cadence at all (090's whole point), does the "one pending-first per frame" throttle
   still serve any purpose, or does it become dead weight the render thread should discard in favor
   of "compile everything as fast as the render thread can, main thread never notices the cost
   either way"? If the throttle is kept, WHOSE frame counter throttles it — the render thread's own,
   uncoupled from imgui?

5. **Does `Document.begin_frame(frame)`'s idempotency contract (one call per frame NUMBER) still
   make sense when the number is supplied by a thread other than the one computing script-driven
   uniform values for that same frame?** `session.tick()` (script engine) and `document.render()`
   currently run back-to-back on the SAME thread inside `_tick_frame_state`/`_update_and_draw`, so
   "the uniforms computed for tick set X are the ones document X renders this frame" is true by
   program order. Splitting tick (main thread, driving imgui/UI-visible state like Play/Stop) from
   render (worker thread) reopens exactly the coherence question 065's "feedback swaps at the frame
   boundary" design law was written to close — the spec needs an explicit answer, not an implicit
   assumption that it still holds.

---

## 6. False trails

- **`066_perf_and_test_diet/01_spec.md`'s own decisions (D1-D6)** looked like the home for "the
  render-set rule" (the task explicitly asks about "the 066 render-set rule") but are NOT — 066 is
  entirely about STARTUP and TEST-SUITE latency (lazy compile, lazy imports, fixture diet, a test
  cull). Its D2 is about bounding frame-0 compile cost, and `ui.py`'s code comment on the actual
  render-set logic cites "066 D2" loosely, but the fuller rule (current + render-all + one pending)
  is NOT written out anywhere in 066's spec text — it lives only in the `_tick_frame_state`
  docstring/comment in `ui.py`. Don't go looking for a "render set" design decision numbered
  elsewhere in 066; the code IS the spec for this particular mechanism.

- **No `066_*` other than `01_spec.md`, and no standalone `066_render_set` doc** — confirmed only
  one file exists under that feature directory.

- **`040_uniform_script_engine.md`** exists but is the ORIGINAL, since-superseded script-engine
  design (041 redesigned it into the stateful engine, 048 collapsed it further per
  `conventions.md`'s CPU-script-engine bullet, which cites "feature 041→048" as the current design's
  lineage). Reading 040 itself would describe a per-uniform `u_*.py` binding scheme that no longer
  exists in the codebase — the task's mention of "the script engine tick" is answered by
  `conventions.md`'s CPU-script-engine bullet and the `scripting/` module-map entry (both current),
  not by 040's text.

- **`.claude/skills/imgui-ui/SKILL.md`'s §8 imgui-bundle version-pinned quirks** (monochrome emoji,
  `push_font` sizing, `image()` losing `tint_col`, glfw cursor sync) looked relevant given the task's
  "texture handle lifetimes in the draw list" framing, but none of §8 is about multi-threaded GL —
  it's all single-threaded imgui-bundle API-surface footguns. The actual texture-lifetime rule for
  090 is `conventions.md`'s 084-D5 bullet (§1 above), not anything in the imgui skill.

- **`081_copilot_engine_sweep/01_spec.md`** mentions `probe_render` and the bridge only in the
  context of a TOOL-classification gate (which tools get a "document" argument), not the GL-marshal
  mechanism itself — it's about the copilot's tool taxonomy, not about threading. Its hits on
  "bridge"/"GL" are incidental; the actual bridge design lives entirely in `copilot/bridge.py` and
  `conventions.md`'s dedicated bullet, not in 081's spec text.

- **`075_dogfood_station/01_spec.md`** has zero hits for "bridge"/"render_on_main"/"worker…GL" — the
  dogfood STATION (075) is a recording/reporting layer over the copilot's `TraceLog` listener seam,
  unrelated to the render/GL bridge. The task's inclusion of 075 in the read list is reasonable
  (it's a copilot-adjacent spec) but it contributes nothing to this audit beyond confirming it's a
  false trail.

- **`conventions.md`'s "No `async` except where python-telegram-bot forces it"** bullet looked like
  it might constrain 090 (a new concurrency mechanism), but it specifically scopes to Python
  `async`/`asyncio` (coroutines), not OS threads — 090's render thread is a plain `threading.Thread`
  like every existing worker, so this bullet is inert here. Flagged so a future reader doesn't
  mistake "no async" for "no new threads."
