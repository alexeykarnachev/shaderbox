# GPU preemption between GL contexts — what this machine's driver actually does

Research for feature 090 (render decoupling). The question: if document rendering moves to a
worker thread with its own GL context, does the UI thread keep its 16.7 ms frame while a 100 ms
document renders? Everything below is measured; scripts are in `../probes/`.

## Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 |
| Driver | 580.173.02 (CUDA 13.0), Compute Mode `Default`, Persistence Mode disabled |
| GL | 4.6.0 NVIDIA 580.173.02 (probes request a 3.3 core context) |
| Display | X11 `:1`, Xorg 21.1.11, 2560x1440, `Composite` extension present, `mutter-x11-frames` running |
| CPU / kernel | 24 cores, Linux 6.14.0-37-generic |
| Python | 3.12.8, uv 0.11.6 |
| moderngl | 5.12.0 |
| PyOpenGL | 3.1.10 |
| glfw (pyGLFW) | 2.10.0 |

**moderngl 5.12.0 has no fence/sync API.** `dir(moderngl.Context)` offers only `finish` and
`query`; there is no `Context.fence`, no sync object wrapper, and `moderngl-stubs` contains no
match for `fence`/`sync`. The GL loader inside `mgl.cpython-312-x86_64-linux-gnu.so` does resolve
`glFenceSync` / `glClientWaitSync`, but nothing exposes them to Python. The probes therefore use
PyOpenGL 3.1.10's `GL.glFenceSync` / `GL.glClientWaitSync`, which act on whatever context is
current on the calling thread — so they compose with moderngl without conflict.

### The heavy document, calibrated

`probes/calibrate_heavy.py` sweeps the iteration count of a fullscreen noise/sin loop at
1280x720 and times it with `ctx.finish()`:

```
cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run python ai_docs/features/090_render_decoupling/probes/calibrate_heavy.py
```
```
iters=   100  median=    0.32 ms      iters=  6400  median=   17.93 ms
iters=   400  median=    1.24 ms      iters= 12800  median=   37.32 ms
iters=  1600  median=    4.90 ms      iters= 25600  median=   74.36 ms
```
A refinement pass near the target gave `iters=34500 → 99.43 ms`, `iters=35000 → 100.93 ms`,
`iters=36000 → 103.53 ms`. **`HEAVY_ITERS = 35000` (100.93 ms per fullscreen pass)** is pinned in
`probes/common.py` and used by every probe below.

### A contended-GPU incident, and the guard that came out of it

An early tile-cost run produced 206–221 ms outliers on draws that should have been 103 ms. The
cause was not the probe: a second Claude session was running its own GPU probes on this box at
the same time (`fuser -v /dev/nvidia0` showed two foreign `python3` clients; `nvidia-smi` read
100% utilization with my probes finished). Every affected measurement was rerun after the GPU
went quiet, and `probes/gpu_clear.py` now blocks each probe until utilization has been ≤20% for
3 consecutive seconds. All numbers in the results table come from post-guard runs.

## Method per configuration

Every configuration draws the same 100.93 ms document and the same trivial light frame; only the
isolation mechanism differs. Each was run twice; both trials are reported.

**A — two processes, two windows, two independent contexts** (`probe_a_run.py`, which drives
`probe_a_heavy_proc.py` and `probe_a_light_proc.py`). The strongest isolation the system can
offer: separate address spaces, separate GIL, separate GL contexts. The light process draws one
trivial fullscreen triangle and records swap-to-swap period over 5 s, first with the heavy
process absent, then with it running. vsync off in both (`swap_interval(0)`), so the period
measures what the driver grants rather than a 60 Hz cap.

**B — one process, two threads, two shared contexts, 1 tile** (`probe_b_threads.py 1`). The
worker creates a hidden window with `share=<main window>` so its document texture is visible to
the main context. It renders the document into an FBO, then signals with `glFenceSync` and waits
on its *own* thread via `glClientWaitSync` — the main thread never waits for the worker; it
samples whatever texture is currently there and swaps. The worker's GL-issue time is recorded
separately to confirm submission really is asynchronous.

**C — same as B, heavy draw split into N scissor tiles** (`probe_b_threads.py 4|16`, and
`probe_e_paced.py` for the budget view). Each tile is a separate `vao.render()` under a scissor
rect with a `glFlush()` after it, giving the scheduler a submission boundary per tile.
`probe_c_tilecost.py` first measures tiling's own cost with nothing else on the GPU, so any
excess seen under load is attributable to contention rather than to tiling.

**D — one process, one context** (`probe_d_single.py`): today's shape. Heavy pass into an FBO,
then the UI pass sampling it, then swap — all in one loop on one thread. A `time.sleep(1.5)`
separates phases; without it an "idle" phase inherits the previous phase's queue and reports a
100 ms p95 that is not real.

**E — the paced 60 fps view** (`probe_e_paced.py`). A, B, C and D all free-run, which answers
"how fast can the UI thread go". The feature asks something different: "can the UI thread hit
its 16.7 ms deadline". Here each UI frame does its trivial draw and then sleeps out the rest of
the 16.7 ms budget, and what is reported is *time spent working* plus a count of frames that
overran. This is the configuration whose numbers decide the verdict.

**GIL isolation** (`probe_gil.py`). The main thread runs a pure-Python tick loop with **no GL at
all** and records inter-tick gaps, against three workers of equal ~100 ms occupancy: the
fence-based GL worker, a `ctx.finish()` GL worker, and a pure-Python busy worker as control. A
gap near 100 ms means the worker held the GIL across its wait.

## Results

Light/UI-thread swap-to-swap period, milliseconds. Both trials shown; all reproduced.

| Config | Heavy idle (median / p95 / max) | Heavy running (median / p95 / max) |
|---|---|---|
| **A** two processes, t1 | 0.19 / 0.21 / 1.38 | 50.22 / 100.97 / 101.39 |
| **A** two processes, t2 | 0.19 / 0.20 / 1.39 | 98.77 / 103.34 / 199.48 |
| **B** two contexts, 1 tile, t1 | 0.22 / 0.25 / 1.39 | 104.45 / 209.00 / 221.35 |
| **B** two contexts, 1 tile, t2 | 0.22 / 0.24 / 129.29 | 104.66 / 208.75 / 212.40 |
| **C** 4 tiles, t1 | 0.22 / 0.25 / 1.39 | 28.05 / 28.84 / 29.60 |
| **C** 4 tiles, t2 | 0.22 / 0.24 / 27.27 | 27.96 / 28.73 / 29.26 |
| **C** 16 tiles, t1 | 0.23 / 0.34 / 104.96 | 8.64 / 9.96 / 135.10 |
| **C** 16 tiles, t2 | 0.22 / 0.24 / 6.92 | 8.60 / 9.89 / 11.00 |
| **D** single context, t1 | 0.23 / 0.25 / 1.43 | 103.81 / 105.75 / 106.64 |
| **D** single context, t2 | 0.22 / 0.24 / 98.82 | 110.58 / 114.26 / 115.15 |
| *control* pure-Python worker, t1 | 0.24 / 0.98 / 206.59 | 10.19 / 10.21 / 14.12 |
| *control* pure-Python worker, t2 | 0.22 / 0.25 / 29.23 | 10.18 / 10.23 / 13.75 |

Two readings stand out. **Config A is no better than config D** — 98.77 ms across two OS
processes versus 103.81 ms in one loop on one context. And **config B is no better than config D
either**: 104.45 ms versus 103.81 ms. Neither process isolation nor context isolation buys
anything.

### Where the stall lives

`probe_a_where.py` splits the light frame into poll / draw-issue / swap:

```
cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run python ai_docs/features/090_render_decoupling/probes/probe_a_where.py
```
```
  idle    poll: median 0.00   draw: median 0.00   swap: median   0.18
  running poll: median 0.01   draw: median 0.01   swap: median  99.13   max 101.32
```
All of it is inside `swap_buffers`; `poll_events` and the draw call stay at 0.01 ms. The light
client's CPU is never starved — it is queued behind the heavy client's in-flight draw. In config
B the same split shows worker GL-issue at **0.02 ms median** while the main thread's swap costs
104.43 ms: submission is genuinely asynchronous, and the main thread still waits. The wait is not
in the API, it is in the GPU's work queue.

**This driver does not preempt a draw call.** A 100 ms `glDrawArrays` runs to completion, and
every other GL client — same context, another context, another process — waits behind it. The
scheduling quantum is one draw call.

### Tiling is free in isolation

`probe_c_tilecost.py`, nothing else on the GPU:
```
tiles=  1 (scissor)  median=  101.49 ms      tiles=  4 (geometry) median=  103.97 ms
tiles=  4 (scissor)  median=  102.68 ms      tiles= 16 (geometry) median=  103.92 ms
tiles= 16 (scissor)  median=  102.48 ms
```
Splitting into 16 scissor tiles costs ~1 ms over one fullscreen pass. Scissor and shrunken
geometry behave the same, so scissor culls fragments before shading as expected. Any slowdown
seen under load in config C is contention, not tiling overhead.

### The paced 60 fps view — the numbers that decide the feature

`probe_e_paced.py`. `ui-work` excludes the pacing sleep; `MISSED` counts frames over 16.7 ms.

```
cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run python ai_docs/features/090_render_decoupling/probes/probe_e_paced.py
```

| Worker | doc cost (median) | UI work median / p95 | Missed frames, t1 / t2 |
|---|---|---|---|
| none | — | 0.06 / 0.07 | 0/297 and 0/297 |
| 1 tile | 102.26 | 102.24 / 103.29 | **48/48 and 48/48** |
| 4 tiles | 110.62 | 22.03 / 27.83 | **134/223 and 130/218** |
| 16 tiles | 134.92 | 0.06 / 0.12 | **0/291 and 2/286** |

A tile sweep (`probe_e_paced.py 9 16 25 36 64`) locates the knee:

| Tiles | doc cost (median) | UI work median / p95 | Missed, t1 / t2 |
|---|---|---|---|
| 9 | 122.43 | 0.78 / 8.14 | 0/297 and 3/283 |
| 16 | 135.69 | 0.06 / 0.12 | 0/291 and 0/297 |
| 25 | 123.69 | 0.07 / 0.12 | 0/297 and 2/287 |
| 36 | 179.29 | 0.06 / 0.13 | 0/297 and 0/297 |
| 64 | 252.96 | 0.07 / 0.17 | 0/297 and 2/275 |

The document tax is not monotonic in tile count, because what matters is per-tile *duration*, not
tile count: 16 tiles of ~8.5 ms each costs 136 ms, while 25 tiles of ~5 ms each costs 124 ms. Past
25 the per-draw overhead takes over and the document degrades badly (36 tiles → 179 ms, 64 tiles
→ 253 ms). **16–25 tiles is the working range**, costing the document roughly 25–35%.

At 9 tiles the p95 is 8.14 ms — under budget but within a factor of two of it, with an occasional
104 ms outlier. That is the boundary: a tile must be short enough that one tile plus one UI frame
fits the budget.

## GIL observations

`probe_gil.py`, main thread doing pure Python with no GL, gap between ticks:

```
sys.getswitchinterval() = 0.005 s
  worker[gl]     frames=13  cost median 105.71   main tick gap: max   6.65  (n=1291014)
  worker[glfin]  frames=36  cost median 106.55   main tick gap: max 103.25  (n= 143565)
  worker[pybusy] frames=40  cost median 101.20   main tick gap: max   5.11  (n= 653356)
```
Trial 2: `gl` max 1.49, `glfin` max 103.02, `pybusy` max 5.33. Reproduced exactly.

**`moderngl`'s `Context.finish()` holds the GIL for the entire GPU wait.** A worker calling it
freezes the main thread for 103 ms — as bad as no threading at all — and the main thread's tick
count collapses ~9× (1.29M → 143K). This is confirmed in moderngl's C source: `MGLContext_finish`
is `self->gl.Finish(); Py_RETURN_NONE;` with no `Py_BEGIN_ALLOW_THREADS`, and **`ALLOW_THREADS`
does not appear anywhere in moderngl.cpp**. Every moderngl entry point holds the GIL for its whole
duration. For non-blocking calls that is microseconds and harmless; for blocking ones it is fatal
to a threaded design.

**PyOpenGL's `glClientWaitSync` releases the GIL**: max main-thread gap 6.65 / 1.49 ms, against
the pure-Python control's 5.11 / 5.33 ms — i.e. at the interpreter's own 5 ms switch interval, the
floor for any two-thread Python program. The fence path is not merely better, it is the only one
that works.

The pure-Python control also sets the realistic ceiling for a free-running UI thread: **10.18 ms
period with a busy Python worker**, from GIL handoff alone. Config C at 16 tiles measured 8.6 ms,
*below* that, precisely because the GL worker spends its 100 ms blocked in `glClientWaitSync` with
the GIL released rather than competing for it.

## Interpretation

**A render thread can rely on:** asynchronous submission (worker GL-issue 0.02 ms while the GPU
works for 100 ms); `share=` contexts making the worker's FBO texture readable from the main
context with no copy; `glFenceSync`/`glClientWaitSync` via PyOpenGL keeping the wait off the main
thread and off the GIL; and scissor tiling being nearly free in isolation (~1 ms for 16 tiles).

**A render thread cannot rely on the driver preempting a long draw.** This is the finding that
shapes the design. A second GL context does not get the UI thread a frame — config B measured
104.45 ms against the single-context baseline's 103.81 ms, and even two separate *processes*
measured 98.77 ms. The GPU serializes at draw-call granularity, so the only lever available is
making the document's individual draw calls short. Threading alone changes nothing; threading
plus tiling changes everything.

That reframes the feature. The design cannot be "move rendering to a thread and the UI is fixed";
it has to be "move rendering to a thread **and split each document pass into tiles sized so one
tile fits inside the UI's frame budget**". The tile size, not the thread, is the mechanism. On
this machine a tile of ~5–8 ms (16–25 tiles of a 100 ms fullscreen pass) puts UI p95 at 0.12 ms
while costing the document 25–35%.

Two consequences worth carrying into the design. Tiling must be adaptive rather than a fixed N:
the right split depends on the document's cost, which is exactly the quantity that is unbounded
by decision, so the worker has to measure its own frame and adjust — a fixed 16 stays correct for
a 100 ms document and is wrong for a 5 ms one (where it is pure overhead) and for a 1 s one
(where each tile is still 60 ms). And a tiled document is torn across tiles unless it renders into
a back texture that is swapped to the front only on the completing fence, since the UI thread
samples it continuously.

The document tax is real and should be stated plainly: the same document takes 25–35% longer when
tiled to keep the UI responsive. That is the trade the feature buys, and it is a good one — an
unresponsive editor is worse than a document at 75% throughput — but it is not free.

Scope note: measured on one driver, one GPU, one document shape (a fragment-bound loop with no
texture reads or dependent branching). The draw-call-granularity finding is a property of this
NVIDIA driver on this machine, and the tile counts are calibrated to this GPU's throughput; the
adaptive-tiling conclusion follows from the mechanism and should carry, but the constants will not.

## False trails

**NVIDIA environment variables do nothing here.** Per NVIDIA's documented OpenGL environment
variables, `__GL_YIELD` accepts unset (`sched_yield()`), `"NOTHING"` (never yield), and
`"USLEEP"` (`usleep(0)`); `__GL_THREADED_OPTIMIZATIONS=1` offloads CPU work to a driver worker
thread. All three were run against config A (`probe_a_run.py __GL_YIELD=USLEEP` etc.), two trials
each:

| Env | Light median, heavy running (t1 / t2) |
|---|---|
| *(none)* | 50.22 / 98.77 |
| `__GL_YIELD=USLEEP` | 101.36 / 101.55 |
| `__GL_YIELD=NOTHING` | 51.55 / 102.27 |
| `__GL_THREADED_OPTIMIZATIONS=1` | 101.66 / 101.96 |

No effect. These are CPU-side yield hints and cannot influence how the GPU schedules submitted
draw calls, which is where the entire stall lives.
Source: <https://download.nvidia.com/XFree86/Linux-x86_64/580.82.09/README/openglenvvariables.html>

**Process isolation.** The intuition that a separate process would be scheduled independently is
wrong on this driver; config A performs the same as config D.

**`nvidia-smi` preemption reporting.** `nvidia-smi -q | grep -i preempt` returns nothing on this
driver — there is no field describing preemption granularity, so the behaviour had to be measured
rather than read off. `Compute Mode` is `Default`, which permits multiple contexts (and so is not
the cause of the serialization) but says nothing about scheduling granularity.

**An early "config C is barely better" reading** came from the contended-GPU incident above, with
the worker showing 314 ms and 580 ms document frames. Those numbers were an artifact of another
session's GPU load; after the guard, the same configuration measured 113 ms and 139 ms.

## Verdict

**PARTIAL.**

For the claim as literally stated — *"a second GL context on a worker thread keeps the UI thread at
~16 ms while a 100 ms document renders"* — the deciding number is **48 of 48 UI frames missing the
16.7 ms budget, with UI work at 102.24 ms median**, in exactly that configuration (two shared
contexts, worker thread, fence-based handoff, one fullscreen draw). The second context on its own
buys nothing: 104.45 ms against the 103.81 ms single-context baseline.

The claim becomes true when the document's draw is tiled. At 16 tiles the same design measures
**0 of 291 and 2 of 286 frames missed, UI work 0.06 ms median and 0.12 ms p95** — well inside
budget — at a document cost of 135 ms instead of 102 ms.

So the worker thread is necessary but not sufficient. What actually keeps the UI at 60 fps is
bounding the duration of each individual draw call; the thread is what lets the UI proceed between
them.
