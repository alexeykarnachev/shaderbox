# GIL and GPU-blocking survey for a moderngl render worker thread (feature 090)

Machine: X11 `:1`, NVIDIA RTX 3090, driver 580.173.02. `moderngl==5.12.0`
(`uv run python -c "import moderngl; print(moderngl.__version__)"`), `glfw==2.10.0` (pyGLFW,
ctypes-based). moderngl's C++ extension source is not shipped in the wheel — the wheel only
ships the compiled `mgl.cpython-312-x86_64-linux-gnu.so` plus a pure-Python wrapper
(`moderngl/__init__.py`, present in `.venv` and read directly). The C++ source was fetched from
`https://raw.githubusercontent.com/moderngl/moderngl/5.12.0/src/moderngl.cpp` (the exact
installed-version tag) to inspect the calls actually asked about.

## Headline finding

**`src/moderngl.cpp` contains zero occurrences of `Py_BEGIN_ALLOW_THREADS`, `Py_UNBLOCK_THREADS`,
`PyEval_SaveThread`, or any other GIL-release primitive, anywhere in the 9509-line file.** Every
moderngl method — render, finish, read, write, clear, texture creation, query-result read — is a
plain CPython C-API function (`PyCFunction`, `METH_VARARGS`/`METH_NOARGS`) that calls into the GL
driver directly and returns, all under the GIL the interpreter already held when it entered the
function. There is no per-call opt-out; this is a blanket property of the extension, confirmed by
grep across the whole file, not by function-by-function inspection.

pyGLFW is the opposite: it loads `libglfw.so` via **`ctypes.CDLL`** (`glfw/library.py`,
`_load_library` → `ctypes.CDLL(...)`), never `ctypes.PyDLL`. ctypes' own docstring for `CDLL`
states plainly: *"Calling the functions releases the Python GIL during the call and reacquires it
afterwards."* `PyDLL`'s docstring says the opposite: *"The GIL is not released."* Every pyGLFW
function (`swap_buffers`, `poll_events`, `make_context_current`, all ~150 others) is a thin
Python `def` that calls straight through this `CDLL` handle (`glfw/__init__.py`), so every glfw
call releases the GIL for its C-side duration — including `poll_events`, which is why the main
thread's event pump keeps ticking even while the interpreter as a whole is "contended."

This asymmetry is the crux of the design question: **glfw calls are GIL-safe by construction;
moderngl calls are not — the only thing that saves the main thread is whether the blocked
moderngl call is *fast* (returns to Python quickly) or *slow* (waits on the GPU while still
holding the GIL).**

## Table

| Call | Releases GIL? | Source citation | Can block on GPU? | Measured max main-thread gap |
|---|---|---|---|---|
| `VertexArray.render` | No | `moderngl.cpp:6823` `MGLVertexArray_render` — `gl.DrawArraysInstanced`/`DrawElementsInstanced`, no `ALLOW_THREADS` | No (submit-only; async command enqueue) — unless the driver's internal queue is full, see False trails | not isolated separately; folded into `render_flush`/`render_finish` below (render is followed immediately by the mode call each loop iteration) |
| `Context.finish` | No | `moderngl.cpp:7546` `MGLContext_finish` — `self->gl.Finish()` directly | Yes — `glFinish` blocks the CPU until the GPU is fully idle, by definition | 220.01 ms (run 2), 117.06 ms (run 3) — matches the call's own measured duration almost exactly (`call_max`=220.00/117.06) |
| `Context.flush` | N/A — **does not exist**. `MGLContext_methods` (`moderngl.cpp:9091-9137`) lists no `"flush"` entry; only `"finish"`. A worker must drop to raw GL (`libGL.so.1` via ctypes) to submit-without-waiting. | `moderngl.cpp:9091-9137` (full method table, grepped) | `glFlush()` itself is non-blocking by GL spec, but the driver can still stall the CPU if its own command buffer/ring is full (back-pressure) | measured via raw `glFlush()`: p50 = 0.00 ms (most calls return instantly, n≈11000-12000 calls in 3s), but occasional stalls up to 108-126 ms tracked the poll gap almost 1:1 |
| `Framebuffer.read` / `Texture.read` | No | `Framebuffer.read` (Python) → `mglo.read_into` → `moderngl.cpp:1842` `MGLFramebuffer_read_into` (`glReadPixels`-equivalent path via `gl.ReadBuffer`+pack); `Texture.read` → `moderngl.cpp:3965` `MGLTexture_read` — `gl.GetTexImage` | Yes — both are synchronous CPU-side readbacks; the driver must wait for prior GPU work touching that surface to finish before the pixels are valid | 119.75 ms (run 2), 198.56 ms (run 3) — call_max 119.74/140.91 ms, poll gap tracked it (run 3's gap exceeding call_max slightly is explained in Method, not evidence of GIL release) |
| `Texture.write` | No | `moderngl.cpp:4138` `MGLTexture_write` — `gl.TexSubImage2D` | Rarely; async on most drivers unless a PBO path forces a stall (not exercised here) | not measured directly (out of scope of the render-thread hot loop; write is a setup-time call in this design) |
| `Buffer.write` | No | `moderngl.cpp:918` `MGLBuffer_write` — `gl.BufferSubData` | Rarely; can stall if the buffer is still in flight on the GPU and the driver can't orphan it | not measured directly |
| `Context.fence` / any sync API | **Does not exist at 5.12.0.** Zero hits for `Fence`, `Sync`, `glFenceSync`, `ClientWaitSync` anywhere in `moderngl.cpp` | full-file grep, `moderngl.cpp` | N/A | N/A |
| `Query.elapsed` (property getter) | No | `moderngl.cpp:2860` `MGLQuery_get_elapsed` — `gl.GetQueryObjectuiv(..., GL_QUERY_RESULT, ...)` | Yes — `GL_QUERY_RESULT` (not `_AVAILABLE`) is the blocking variant of the query-result call; it stalls the CPU until that specific query's result lands | not measured directly (established from source: the non-`_AVAILABLE` query-result call is a known synchronous stall point in the GL spec, and moderngl only exposes this blocking form for `.elapsed` — no polling `_AVAILABLE` variant is exposed) |
| `Context.clear` | No | `moderngl.cpp:1732` `MGLFramebuffer_clear` (what `ctx.screen.clear`/`fbo.clear` dispatch to) — `gl.Clear(...)` | No — `glClear` is an async command-buffer entry like a draw call | not measured separately |
| `Texture` creation (`Context.texture(...)`) | No | `moderngl.cpp:3552` `MGLContext_texture` — `gl.GenTextures` + `gl.TexImage2D` | Occasionally — allocation is usually async, but a data upload with no PBO, or VRAM pressure forcing eviction, can stall | not measured directly |
| `glfw.swap_buffers` | **Yes** | `glfw/__init__.py:2373-2381` → `_glfw.glfwSwapBuffers(window)`, `_glfw` is a `ctypes.CDLL` (`glfw/library.py`) | Can still block the CALLING thread on vsync/driver present-queue, but that block happens with the GIL released, so it does not stall OTHER threads | n/a (main-thread-only call in the target design; not exercised as a worker call) |
| `glfw.poll_events` | **Yes** | `glfw/__init__.py:1854-1861` → `_glfw.glfwPollEvents()`, same `CDLL` | No | this IS the measurement instrument (see Method) |
| `glfw.make_context_current` | **Yes** | `glfw/__init__.py:2350-2358` → `_glfw.glfwMakeContextCurrent(window)`, same `CDLL` | No (cheap, no GPU work) | n/a |

## Method

`ai_docs/features/090_render_decoupling/probes/gil_probe.py` (run: `uv run python
ai_docs/features/090_render_decoupling/probes/gil_probe.py`, needs `DISPLAY=:1`).

Design: the main thread creates a hidden host glfw window (`glfw.window_hint(VISIBLE, FALSE)`)
and runs a tight loop of `glfw.poll_events()` for 3 s, recording `time.perf_counter()` gaps
between consecutive iterations into a `GapStats` (max, p50/p95/p99). A worker thread is given a
second hidden glfw window **created on the main thread and shared with the host**
(`glfw.create_window(..., share=host_window)` — GLFW's window-management calls are only
documented as main-thread-safe on some platforms, so window create/destroy always happens on the
main thread; only `make_context_current` + GL/moderngl calls happen on the worker thread), makes
it current on itself, and builds a fresh `moderngl.Context` there. The worker renders a fullscreen
triangle with a fragment shader that does a data-dependent trig loop (`u_iters` iterations),
calibrated per-run by binary search to land the render itself near ~100 ms on an idle GPU
(`calibrate_iters`), then loops: render, then one of `{ctx.finish()`, `glFlush()` via `libGL.so.1`
through ctypes since moderngl has no flush method, `glFlush()+fbo.read()}`.

Two controls isolate scheduler/GIL-tick effects that have nothing to do with GL:
- **Control 1 (`sleep`)**: worker does `time.sleep(0.1)` only, no GL at all.
- **Control 2 (`busy`)**: worker does a pure-Python busy loop (`x = x*1.0000001+1.0` for
  ~100 ms/iteration) — this holds the GIL "conceptually" but CPython's bytecode-level eval-loop
  switch (`sys.getswitchinterval()` = 0.005 s on this install) still yields it periodically, so
  this is the reference point for "GIL fully Python-contended but never released for a long
  stretch."

**Same-run call-duration correlation** (added after the first pass, see False trails): the worker
also timestamps each blocking call (`finish`/`flush`/`read`) itself and records its wall-clock
duration. The report cross-references `call_max`/`call_p50` for that call against the poll loop's
`max gap` in the *same run*. This makes the result robust to a busy shared GPU (see below) — the
question isn't "is the absolute gap 100ms," it's "does the poll-loop gap track the call's own
duration 1:1," which is the direct signature of "GIL held for the call's full length."

## Raw numbers

Two independent runs (`run2`, `run3` in scratchpad logs), both under a **contended GPU** (see
False trails — another concurrent process was saturating the RTX 3090 during both runs, so
absolute durations are 1-2x inflated versus an idle GPU, but the within-run correlation is what
matters):

```
=== case: control_1_sleep ===        max=1.41-1.66ms   p50=0.00ms  n≈1.1M iterations/3s
=== case: control_2_busy_python ===  max=5.10-5.48ms   p50=0.00ms  n≈115-119k iterations/3s

run2:
render_finish   call_max=220.00  call_p50=108.08  poll_max_gap=220.01  n_calls=23
render_flush    call_max=108.89  call_p50=  0.00  poll_max_gap=112.40  n_calls=11008
render_read     call_max=119.74  call_p50= 36.48  poll_max_gap=119.75  n_calls=46

run3:
render_finish   call_max=117.06  call_p50= 13.87  poll_max_gap=117.06  n_calls=54
render_flush    call_max=125.54  call_p50=  0.00  poll_max_gap=124.03  n_calls=10973
render_read     call_max=140.91  call_p50= 39.60  poll_max_gap=198.56  n_calls=28
```

Interpretation:
- **Controls are clean**: sleep and pure-Python-busy both cap out under 6 ms — the poll loop is
  never meaningfully starved by ordinary GIL contention or scheduling, in either run. This is the
  noise floor.
- **`render_finish`**: `poll_max_gap` equals `call_max` to within 0.01-0.1 ms in both runs. The
  main thread's event pump was frozen for exactly as long as the worker's `ctx.finish()` call took
  — direct behavioral proof the GIL was held for the call's entire GPU-bound duration.
- **`render_flush`**: `call_p50 = 0.00 ms` and `n_calls` an order of magnitude higher than the
  other two modes (≈11000 calls in 3 s vs. tens) — confirms `glFlush()` is non-blocking *on the
  common path*, matching the GL spec. But its outlier `call_max` (108-126 ms, i.e. the driver's
  command queue occasionally pushing back under this machine's GPU contention) tracks
  `poll_max_gap` just as tightly as `finish` does. The mechanism is the same either way: whatever
  makes the *call itself* take long, the GIL is held for the whole thing, so the main thread stalls
  for exactly that long.
- **`render_read`**: tracks closely in run2 (119.74 vs 119.75), less exactly in run3 (140.91 vs
  198.56 — see False trails for why this is still consistent with "GIL held," not evidence of a
  release).

## False trails

- **GPU contention from a concurrent process.** Both probe runs used to produce the numbers above
  executed while another process on this same machine (`ai_docs/features/090_render_decoupling/probes/01_frame_timing.py`,
  visible in `ps aux` throughout, `nvidia-smi` showing 93-100% GPU utilization for the whole
  session) was independently saturating the RTX 3090 — apparently a sibling research pass on this
  same feature, running concurrently. This explains why calibration sometimes needed only the
  starting `iters=4000` to already hit ~100-135 ms (normally that would take ~70000 iterations on
  an idle GPU, per an early pre-contention run that got `iters=69731 -> 100.7ms`) and why absolute
  numbers vary 2x between otherwise-identical runs. It does **not** undermine the verdict: the
  question is whether the GIL is released during a blocking call, and the same-run
  call-duration-vs-poll-gap correlation answers that directly regardless of how long the call
  happens to take on a given day. A rerun on a quiet GPU would show smaller absolute numbers with
  the same 1:1 correlation, not a different mechanism.
- **First script draft crashed the `flush`/`read` worker immediately.** The first version called
  `ctx.gl.Flush()`, guessing moderngl's Python `Context` exposes a `.gl` GL-function-table
  attribute the way the internal C++ struct does. It doesn't (`AttributeError: 'Context' object
  has no attribute 'gl'`); the worker thread died right after calibration, and the main thread's
  poll loop then ran alone for the rest of the window, producing a spuriously tiny "max gap"
  (0.09-0.13 ms) that would have read as "flush and read are GIL-safe" if not caught. Fixed by
  dropping to raw `libGL.so.1` via `ctypes.CDLL` for `glFlush()` (moderngl has no flush method at
  all — see the table) and by making the worker function raise on `stop`/errors loudly rather than
  swallow them. **Lesson for the report: a crashed worker thread produces an artificially clean
  result, not a null result — always check `n_calls`/exception output before trusting a "no stall"
  reading.**
- **`glfw.destroy_window` called from the worker thread caused an intermittent X `BadWindow`
  crash** (`Major opcode 18, X_ChangeProperty`) in the first cross-thread version, where the
  worker created *and* destroyed its own hidden window. GLFW's own docs mark window creation and
  destruction as safe only on the main thread on some platforms; X11 window-property calls issued
  from a non-owning thread raced the main thread's own X11 traffic. Fixed by moving all
  `glfw.create_window`/`glfw.destroy_window` calls to the main thread; the worker thread only calls
  `glfw.make_context_current` (documented as callable from any thread) plus GL/moderngl calls on
  the context it was handed.
- **`run3`'s `render_read` gap (198.56 ms) exceeds its own `call_max` (140.91 ms).** This looks at
  first like evidence the GIL was released mid-call. It isn't: the recorded `call_max` only wraps
  `_libgl.glFlush(); fbo.read()` — it does not include the immediately preceding
  `vao.render(...)` call in the same loop body, which is also a GIL-held (if normally cheap) call.
  Under this run's GPU contention, a render call queued right before a slow read can itself take
  tens of ms if the command queue is backed up, and the poll loop's gap spans from the *start* of
  that render to the *end* of the read — both GIL-held, contiguous, just not both inside the timed
  window. This is consistent with "GIL held throughout," not a counter-example; it's an artifact of
  timing only one of the two calls in the loop body.

## Verdict: CONDITIONAL

**A moderngl render thread CAN stall the main thread — every moderngl call is a plain GIL-holding
C function, with no exception, at the installed version (5.12.0), confirmed by source (zero
`Py_BEGIN_ALLOW_THREADS` in the whole extension) and by direct measurement (poll-loop gaps track
blocking-call duration 1:1).** glfw itself is not the risk — every pyGLFW call releases the GIL by
construction (`ctypes.CDLL`) — the risk is entirely on the moderngl side of the worker thread.

The exact rule a render thread must follow to keep the main thread safe:

1. **Never call `Context.finish()` on the worker's hot path.** It is a direct, unconditional
   `glFinish()` under the GIL — the single worst call in the API for this design, and the one this
   probe demonstrates most cleanly (poll gap = call duration, to the millisecond).
2. **Never call `Framebuffer.read()` / `Texture.read()` / `Texture.read_into()` on the worker's hot
   path**, for the same reason — they are synchronous CPU-side readbacks under the GIL. If a
   worker thread needs pixel data back on the CPU (e.g. for export), that read must itself move to
   a place where a multi-hundred-ms stall is acceptable, or be paced so it happens rarely (once at
   the end of an export, not once per frame).
3. **`Query.elapsed` reads via the exposed `.elapsed` property are also a blocking `GL_QUERY_RESULT`
   call** (source-confirmed; moderngl exposes no `_AVAILABLE`/non-blocking polling variant at this
   version) — do not read it every frame from the worker unless the query is known-complete
   already (e.g. read last frame's query at the start of the next one, after enough GPU work has
   elapsed that the result is virtually guaranteed ready — this reduces but does not eliminate the
   theoretical stall risk, since there's no polling API to check readiness first).
4. **There is no fence/sync primitive in moderngl 5.12.0 to build a non-blocking "is the GPU done
   yet" poll.** `Context.finish()` (block until totally idle) and a blocking `Query.elapsed` read
   are the only two ways moderngl lets you find out GPU completion state, and both hold the GIL
   for their CPU-side wait. A worker thread that needs frame-pacing without stalling the main
   thread must throttle some other way — e.g. cap frames-in-flight by tracking how many
   `render()` calls have been issued since the last confirmed-complete point, or accept the
   occasional real stall and keep it off the render thread's *submission* path (submit, don't
   wait).
5. **`render()`, `clear()`, `write()`, and texture creation are safe to call every frame** — they
   are async command-buffer submissions under normal conditions (confirmed: `glFlush()`'s p50 was
   0.00 ms across ~11000 calls/3s in both runs, meaning the preceding render+flush pair returned
   near-instantly the overwhelming majority of the time). The measured outlier stalls on `flush`
   (up to ~126 ms) came from this specific machine's GPU being saturated by a concurrent process
   during the test — a legitimate finding in its own right: **even a "safe," non-blocking call can
   occasionally stall if the GPU's command queue backs up under contention, and when it does, that
   stall is just as GIL-holding as `finish()`.** A production worker thread should not assume any
   GL call is unconditionally cheap; it should assume every GL call CAN occasionally take as long
   as the GPU needs, and design the main thread's tolerance (or a watchdog / frame-skip) around
   that possibility rather than around the common-case latency alone.

In short: this is not "moderngl is safe" nor "moderngl is unsafe" — it is safe under a specific,
enforceable discipline (never call `finish`/`read`/blocking-`elapsed` from the hot path; submit
and let the swap chain/driver pace you), and unsafe the moment that discipline is violated even
once, because there is no per-call escape hatch in the library to fall back on.
