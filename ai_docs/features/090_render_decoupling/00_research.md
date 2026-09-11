# 090 — Render decoupling: research

The editor lags while a heavy document renders. The maintainer's constraint: ShaderBox is a
shader-centered IDE, a document may cost any amount of GPU time, and no budget or quality cap on
a document is acceptable. The editor is a small part of the app and must not wait on the document.
This file synthesizes seven research reports and one measurement of the maintainer's own machine
into what the design has to be. Nothing is implemented. The spec (`01_spec.md`) follows the
discussion this file opens.

Reports live in `research/`, every number in them comes from a script under `probes/` (kept
verbatim as evidence; `ruff` excludes `ai_docs/`). Machine: X11 `:1`, GNOME with ibus, NVIDIA
RTX 3090, driver 580.173.02, OpenGL 4.6, glfw 3.4 (pyGLFW 2.10), moderngl 5.12.0, PyOpenGL
3.1.10 (already a dependency), Python 3.12.

## The headline

Three mechanisms stack, and each one was measured separately:

1. **The frame IS the input latency.** One loop renders the document, drains input, draws the
   UI, swaps. A 100 ms document puts the UI frame at 104 ms median, all of it blocked inside
   `swap_buffers`, and a keystroke lands exactly one frame later: 104 ms against 7 ms with a
   trivial document. With "Render all" on (the shipped default) the period is the SUM of every
   open document's cost, whichever is in front. (`research/baseline.md`)
2. **The driver does not preempt a draw call.** A 100 ms `glDrawArrays` runs to completion and
   every other GL client waits behind it: another context in the same process measured 104 ms,
   another PROCESS measured 99 ms, against 104 ms for today's single loop. A render thread with
   its own context, by itself, changes nothing for a GPU-bound document. Splitting the same
   pass into scissor tiles gives the scheduler points to interleave: 16 tiles of ~6 ms put the UI
   thread at 0.06 ms median work, 0 missed frames of 291 at a 60 fps pace, and cost the document
   25–35 % throughput. Tiling itself is ~1 ms of overhead in isolation. (`research/gpu_preemption.md`)
3. **The X11 input method turns a slow frame into motion after release.** With ibus, each key
   press is forwarded and comes back at the next event pump, so throughput is one key per frame;
   once the frame is slower than the 30 ms repeat interval the backlog drains after the key is
   up (903 ms of trailing `j` at a 100 ms frame). Without ibus the backlog lands in one batch at
   the next poll and nothing trails. (`research/input_method.md`)

Two constraints bound any implementation:

- **moderngl holds the GIL for every call.** `finish`, `Framebuffer.read`, `Texture.read` and a
  blocking `Query.elapsed` stall the main thread for the full GPU wait (measured 1:1 to 0.01 ms).
  `render`, `clear`, `write` return immediately on the common path but can stall under driver
  back-pressure, and moderngl offers no polling escape. PyOpenGL's `glClientWaitSync` releases
  the GIL. (`research/gil_and_blocking.md`)
- **moderngl 5.12 exposes no fence or sync API at all**, so the cross-context handoff goes
  through PyOpenGL (`glFenceSync`, `glClientWaitSync`, and the producer-side `glFlush` the
  `ARB_sync` spec requires). (`research/prior_art.md`, verified against the installed package)

A correction to what was said in chat before the research: the browser analogy ("a heavy WebGL
tab does not freeze the browser") does not transfer. On this driver, isolation by context or by
process gives no preemption inside one draw. Draw duration is the only lever.

## The shape the evidence supports

Not "move rendering to a thread" but **"move rendering to a thread AND bound every draw's
duration"**. The thread is necessary (the UI thread must never issue or wait on document work);
tiling is what makes the GPU interleave. Each part below has a measured reason.

**A render thread owning a second GL context.** A hidden glfw window created on the main thread
with `share=` the main window (glfw's documented offscreen pattern; window creation and the event
pump are main-thread-only, `make_context_current` and `swap_buffers` are any-thread). The context
is made current on the worker once and stays there for its life (moderngl's maintainer's own
recommendation, issue #623). Shared: textures, buffers, programs, sync objects. Not shared: VAOs,
FBOs, queries — the worker creates its own.

**Engine-level adaptive tiling per pass.** The shader is untouched; `Pass.render` issues N
scissored draws instead of one. N is chosen per pass from the measured tile duration (the 088
timer-query machinery, on the render thread) toward a target of ~5–8 ms per tile. A fixed N is
wrong twice over: pure overhead for a 5 ms document and still 60 ms tiles for a 1 s one. The
maintainer's "no budget" constraint is honored: nothing about the document is reduced, it runs
at 65–75 % throughput while the UI is interactive. Whether that tax is acceptable, and whether
tiling should relax when the UI is idle, is an open question below.

**Front/back output textures swapped on the completing fence.** A tiled document is torn across
tiles while it renders; the UI thread samples a document texture continuously (viewer, pass
strip previews, channel views), so it must read only a frame whose fence has signaled. The
worker renders into the back texture, fences, flushes, and publishes; the main thread polls the
fence with a zero timeout before presenting.

**One frame in flight, self-throttled.** Because `render` can block with the GIL held under
back-pressure, the worker never queues a second frame before the first's fence signals; it waits
with `glClientWaitSync` (GIL released) or sleeps and polls. It never calls `finish`, never reads
back on the hot path.

**Message passing, never live state.** The render↔UI ledger (`research/shared_state.md`) has ~30
rows. Their ownership classifies cleanly: GL objects and the feedback ring belong to the render
thread; uniform values, the pass graph, the clock origin and the render set are UI-owned and
reach the worker as a per-frame snapshot; source text, pass add/delete/rename, document and
project switches, and the feedback frame protocol are genuinely bidirectional and need a barrier
(the worker quiesces, the main thread proceeds). Errors flow back as data: compile errors already
do (polled `compile_unit.errors`, last-good program kept); GL errors and `MediaError` have no
handler today and propagate to the process-level catch, which a worker thread cannot do.

**The existing bridge is the precedent, not a special case.** The copilot's worker→main bridge
(`copilot/bridge.py`, `render_defer.py`) already marshals GL work to whoever owns GL with a
blocking round trip and a timeout. The render thread becomes that owner, and the copilot's
render-and-look, exports, and the pass strip's readbacks route to it rather than to the main
thread. The one-frame cue latch in `render_defer.py` exists only because a synchronous encode
froze the thread that draws the cue; that reason disappears.

## What the code says today

`research/gl_sites.md`: 47 per-frame GL call sites (15 create, 7 render, 5 upload, 3 read back,
6 release, 12 present-to-imgui, 4 window/error state), 23 lifecycle sites, 8 GL object kinds,
exactly one thread touching GL. Notable:

- `ChannelBlit.render` (alpha and RGB views) runs INSIDE the imgui draw phase reading the
  document's output texture directly (`ui.py::_draw_document_image`). Under a render thread that
  is a cross-context read of a texture the worker may be writing; it moves to the worker or reads
  the published frame.
- `get_active_uniforms` compiles lazily (066 D1) and the Uniforms tab calls it from the draw
  phase every frame it is open, so a compile can start on the UI thread while the worker holds
  the program. Three other compile sites outside the render path (`UIDocument.save`, `add_pass`,
  the Uniforms tab) become requests to the render thread.
- The four `document.render` calls in `ui.py` have no local handler; an exception ends the app.
- `editor/render.py` has no dependency on document objects: the strongest "stays on main" case.
- The profiler's GPU query ring (088) is keyed to one loop's frame count; queries are per-context
  and cannot be shared. The worker needs its own instance and a definition of "frame N" scoped to
  its own loop, with completed profiles handed across.
- The dogfood harness's standalone EGL context does not exercise context sharing; it is not a
  precedent for this.

`research/decisions_audit.md`: `conventions.md`'s "GL objects live with the render thread" bullet
means the MAIN thread today and must be redefined; `exporters/base.py` already uses the phrase in
the same sense. Four threads exist (intel worker, copilot loop, two exporter workers), all GL-free
by construction, all handing results to main via queues or injected callbacks — the pattern the
render thread's outbound side should match. `todo.md` is empty; no trigger fires. Sizing per the
`dev_flow.md` preamble: high-blast-radius (async/lifecycle, touches `conventions.md`, reshapes 084
D5, 088 and the export funnel) — 2+ pre-implementation reviewers, a post-implementation
spec-fidelity audit, swarm convergence, sanitization sweep.

## Decisions the spec must take (open, for discussion)

1. **The document tax.** Tiling costs a heavy document 25–35 % throughput while the UI is
   interactive. Options: always tile to the target (simplest, the document runs slower whenever
   it is heavy); tile only while the editor is focused or input is live and run untiled when the
   UI is idle (the document gets full throughput when nobody is typing; a mode switch to design);
   expose the target tile duration as a setting. The measured knee is 16–25 tiles for a 100 ms
   pass; 36+ degrade the document without helping the UI.
2. **What the render thread owns beyond the live frame.** (a) Only per-frame document draws move;
   exports and copilot renders stay main-thread through the existing post-swap funnel. (b)
   Everything GL moves, and the funnel's "after swap" ordering becomes an explicit signal. The
   audit calls these different architectures with different blast radii. The evidence favors (b):
   a main-thread export still issues 100 ms draws on the UI thread's context, which is the exact
   stall being removed.
3. **Where the second context lives.** `App` owns the only context creation today;
   `ProjectSession` is the headless, engine-adjacent home whose job is rendering documents. The
   worker's context and lifecycle need one owner, and the smoke harness and the `app` fixture must
   still construct it headless.
4. **The frame-number contract.** `session.tick()` (script uniforms, main thread) and
   `document.render()` run back-to-back today, so "the uniforms computed for tick X are what
   frame X renders" is program order. Across threads the snapshot must carry the frame number
   and the resolved `u_time`, and `begin_frame`'s once-per-frame idempotency needs restating on
   the worker's loop.
5. **The render-set rule.** 066 D2's "one first render per frame" bounded compile cost on the UI
   frame; once compiles happen on the worker the throttle may be dead weight or may need the
   worker's own counter. The "current document plus Render all" half of the rule has no spec home
   and lives in `ui.py::_tick_frame_state`.
6. **The profiler across two cadences.** 088's three-deep ring read two frames late assumes one
   loop. The worker profiles its own loop; the FPS panel shows the UI frame beside the document's
   own period. The "GPU share of budget" coloring changes meaning when the document no longer
   shares the UI's budget.
7. **Opting out of XIM.** `os.environ["XMODIFIERS"] = "@im=none"` before `glfw.init()` removes
   the one-key-per-pump protocol (measured). Cost: no compose/dead keys, no CJK engines inside the
   app; XKB layouts still work. With the UI at 60 fps the backlog never builds, so this is a
   separate, smaller decision; the evidence says take it.
8. **Error surfaces on the worker.** GL errors and `MediaError` need a catch that keeps the
   last-good frame and reports, matching the compile-error shape the UI already polls.
9. **The copilot's GL-free carve-out** (`document_tree`, `grep`, `read_lib` read `Document`
   fields from the copilot thread unguarded) must be re-justified once the render thread is a
   second writer, or routed through the same snapshot.

## Verification the spec will carry

From `research/baseline.md`, measured before any change and re-runnable from `probes/`:

| Measure | Today (heavy document current) | Gate after 090 |
|---|---|---|
| UI frame period, p95 | 104.5 ms | under 15 ms |
| Editor key latency, p95 | 111.8 ms | under 20 ms |
| Editor draw (`layout_following_cursor` + panel redraw), max | 1.55 ms | unchanged |

Falsifiers the spec must name: cut the tiling and the frame gate goes red (measured: 1 tile
misses 48/48 frames); publish the back texture without the fence and a torn-frame canary reads
mixed tiles; call `finish` on the worker and the main-thread gap gate goes red (measured: 103 ms
gaps).

## Scope and caveats of the measurements

One driver, one GPU, one document shape (a fragment-bound loop, no texture reads, no dependent
branching). Draw-call-granularity scheduling is a property of this NVIDIA driver; the adaptive
tiling conclusion follows from the mechanism and should carry, the constants will not. Two
research agents ran GPU probes concurrently on the same box; the preemption agent added a GPU-idle
guard (`probes/gpu_clear.py`) and re-ran every affected measurement, and each reported number
reproduced across two trials. Hidden windows were used for the baseline; the report states where
that could change swap behavior.

## Reports

| File | What it establishes | Verdict |
|---|---|---|
| `research/gpu_preemption.md` | No draw preemption across contexts or processes; tiling works; the knee | PARTIAL for "a thread alone", SUPPORTS for "thread + tiling" |
| `research/gil_and_blocking.md` | moderngl holds the GIL on every call; which calls block; PyOpenGL waits release it | CONDITIONAL, with the rule |
| `research/baseline.md` | Frame period and key latency today; the gate numbers | 104 ms → gate p95 < 15 ms |
| `research/shared_state.md` | ~30 shared-state rows, ownership classification, exception paths | — |
| `research/gl_sites.md` | 47 per-frame + 23 lifecycle GL sites, the bridge, the out-of-place blit | — |
| `research/decisions_audit.md` | Which locked decisions 090 honors, extends or reverses; sizing | high-blast-radius |
| `research/prior_art.md` | glfw / moderngl / GL spec / driver facts with fetched quotes; what is folklore | — |
| `research/input_method.md` | The ibus one-per-pump protocol and the in-process opt-out | — |
