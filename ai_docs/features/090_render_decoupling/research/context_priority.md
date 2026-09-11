# EGL context priority — does a high-priority context jump a 100 ms draw?

Research for feature 090 (render decoupling), following `gpu_preemption.md`. That experiment
established that this driver does not preempt a draw call: a 100 ms fullscreen `glDrawArrays`
runs to completion and every other GL client waits behind it, whether it is another context in
the same process or another process entirely. The maintainer then observed that his desktop and
a terminal vim stay smooth while ShaderBox renders a heavy document, and that mutter is known to
ask for a high-priority EGL context — which raises the question this experiment answers:

> Does a context created with `EGL_CONTEXT_PRIORITY_HIGH_IMG` get its draws scheduled ahead of,
> or preempt mid-draw, a single 100 ms draw on a low/medium-priority context on this driver?

Everything below is measured; scripts are `../probes/prio_*.py`.

## Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 |
| Driver | 580.173.02 |
| EGL | `EGL_VERSION` 1.5, `EGL_VENDOR` NVIDIA, `EGL_IMG_context_priority` present |
| Display | X11 `:1`, GNOME 46 session (`XDG_SESSION_TYPE=x11`), gnome-shell 46.0, mutter 46.2 |
| Python | 3.12.8 |
| PyOpenGL | 3.1.10 (its `OpenGL.EGL` bindings drive libEGL directly) |
| moderngl / glcontext | 5.12.0 / 2.3.7 |
| glfw (pyGLFW) | 2.10.0, `3.4.0 X11 GLX Null EGL OSMesa monotonic shared` |

The document under test is the one `gpu_preemption.md` calibrated: `HEAVY_FRAG` at 35000
iterations, 1280x720, pinned in `../probes/common.py` as 100.93 ms per fullscreen pass. Re-timed
through this experiment's raw-GL path it measured **100.44 ms median**, so the two experiments
are measuring the same quantity.

## The extension, from the primary source

`EGL_IMG_context_priority`, Khronos EGL extension #10, version 1.1 (8 September 2009), fetched
from <https://registry.khronos.org/EGL/extensions/IMG/EGL_IMG_context_priority.txt>.

New tokens, quoted verbatim:

```
    New attributes accepted by the <attrib_list> argument of
    eglCreateContext

        EGL_CONTEXT_PRIORITY_LEVEL_IMG          0x3100

    New attribute values accepted in the <attrib_list> argument
    of eglCreateContext:

        EGL_CONTEXT_PRIORITY_HIGH_IMG           0x3101
        EGL_CONTEXT_PRIORITY_MEDIUM_IMG         0x3102
        EGL_CONTEXT_PRIORITY_LOW_IMG            0x3103
```

Three levels, not four. (`EGL_CONTEXT_PRIORITY_REALTIME_NV` 0x3357 exists in `/usr/include/EGL/
eglext.h` but belongs to a different extension, `EGL_NV_context_priority_realtime`, which this
driver does not advertise.)

The priority is explicitly a hint, and the spec says so three times. From the Overview:

> "It is possible that an implementation will not honour the hint, especially if there are
> constraints on the number of high priority contexts available in the system, or system policy
> limits access to high priority contexts to appropriate system privilege level. A query is
> provided to find the real priority level assigned to the context after creation."

From the `eglCreateContext` wording:

> "EGL_CONTEXT_PRIORITY_LEVEL_IMG determines the priority level of the context to be created.
> This attribute is a hint, as an implementation may not support multiple contexts at some
> priority levels and system policy may limit access to high priority contexts to appropriate
> system privilege level. The default value for EGL_CONTEXT_PRIORITY_LEVEL_IMG is
> EGL_CONTEXT_PRIORITY_MEDIUM_IMG."

So **the default is MEDIUM**, and an application that never heard of the extension sits in the
middle rather than at the top. And on the query, from the `eglQueryContext` wording:

> "Querying EGL_CONTEXT_PRIORITY_LEVEL_IMG returns the priority this context was actually
> created with. Note: this may not be the same as specified at context creation time, due to
> implementation limits on the number of contexts that can be created at a specific priority
> level in the system."

Because the driver may clamp silently, every context these probes create is queried with
`eglQueryContext(EGL_CONTEXT_PRIORITY_LEVEL_IMG)` and the granted level is printed in every run
header. Issue 3 in the spec warns specifically that "a request for LOW will actually return
MEDIUM on an implementation that doesn't differentiate between the lower two levels" — which is
the failure mode that would have made this whole experiment vacuous.

**It does not happen here.** `prio_feasibility.py`:

```
cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run python ai_docs/features/090_render_decoupling/probes/prio_feasibility.py
```
```
EGL_VERSION: b'1.5'
EGL_VENDOR:  b'NVIDIA'
EGL_IMG_context_priority present: True

  requested DEFAULT -> eglQueryContext granted MEDIUM
  requested LOW     -> eglQueryContext granted LOW
  requested MEDIUM  -> eglQueryContext granted MEDIUM
  requested HIGH    -> eglQueryContext granted HIGH
```

All three levels are distinct and all three are granted to an ordinary unprivileged user — no
clamping, no privilege requirement, and the unspecified default lands on MEDIUM exactly as the
spec says. Every configuration below therefore tests a real priority difference, and each run
header repeats the granted level so this can be checked per measurement rather than assumed.

## Method

Every configuration runs the same shape as `probe_e_paced.py`, the probe whose numbers decided
the earlier verdict: a light loop paced to a 16.7 ms budget beside a worker rendering the 100 ms
document, reporting the light frame's *work* (pacing sleep excluded) and a count of frames that
overran the budget. What changed is that both contexts are now created through EGL with an
explicit `EGL_CONTEXT_PRIORITY_LEVEL_IMG`, and each run header prints the level `eglQueryContext`
actually granted.

Two trials each, 5 s per phase, each preceded by the GPU-idle guard, and each configuration is
measured twice — once with the worker idle (to show the light loop's own floor) and once with it
running.

**Surfaces.** Both contexts render to pbuffers. The light context was meant to present to an X
window, which would have matched the earlier probes' `swap_buffers` timing exactly, but the only
window path available here is glfw, and glfw cannot be given a priority attribute (the
feasibility section below establishes this) — so a windowed light context could not have carried
the priority under test. The light frame is therefore closed with
`glFenceSync`/`glClientWaitSync` instead of a swap. This measures the same thing the earlier
swap measured: the earlier experiment's `probe_a_where.py` split the light frame into
poll/draw/swap and found the entire stall inside `swap_buffers` waiting on the GPU queue, at 0.01
ms of CPU. A fence waits on that same queue. The control below confirms the substitution is
sound by reproducing the earlier experiment's number.

**GL through PyOpenGL, not moderngl.** moderngl could not be attached to an externally-created
EGL context on this stack, so the probes issue raw GL with the byte-identical shaders from
`common.py`. `prio_gl.py` records the reason; the False trails section gives the detail.

The configurations:

| | Heavy context | Light context | Tiles | Asks |
|---|---|---|---|---|
| **P0** | MEDIUM | default | 1 | control — does the earlier ~100 ms reproduce through EGL contexts and a fence? |
| **P1** | LOW | HIGH | 1 | the question: widest priority gap, one 100 ms draw |
| **P2** | MEDIUM | HIGH | 1 | the realistic gap — an unmodified heavy client sits at the MEDIUM default |
| **P3** | LOW | HIGH | 16 | does priority *plus* tiles beat tiles alone? |
| **P3b** | LOW | default | 16 | tiles alone, the comparison P3 needs |
| **P4** | LOW | HIGH | 1 | P1 across two OS processes rather than two threads |
| **P5** | each | — | 1, 16 | what a priority costs a context running alone |

## Results

Light-loop work per frame in milliseconds, with the priority each context was granted. `MISSED`
counts frames over the 16.7 ms budget. Both trials shown; everything reproduced.

| Config | Granted (heavy / light) | Heavy idle: median / p95 / max | Heavy running: median / p95 / max | Missed | Heavy cost |
|---|---|---|---|---|---|
| **P0** control, t1 | MEDIUM / MEDIUM | 0.16 / 0.28 / 1.16 | 101.47 / 103.26 / 103.95 | **48/48** | 101.41 |
| **P0** control, t2 | MEDIUM / MEDIUM | 0.22 / 0.38 / 0.98 | 101.44 / 102.99 / 104.14 | **48/48** | 101.48 |
| **P1** LOW vs HIGH, t1 | LOW / **HIGH** | 0.16 / 0.31 / 0.79 | **101.05** / 113.75 / 123.68 | **47/50** | 101.14 |
| **P1** LOW vs HIGH, t2 | LOW / **HIGH** | 0.22 / 0.36 / 1.10 | **101.25** / 103.07 / 103.24 | **48/48** | 101.38 |
| **P2** MEDIUM vs HIGH, t1 | MEDIUM / **HIGH** | 0.15 / 0.24 / 1.33 | 101.54 / 103.44 / 104.32 | **48/48** | 101.52 |
| **P2** MEDIUM vs HIGH, t2 | MEDIUM / **HIGH** | 0.16 / 0.41 / 4.41 | 101.37 / 103.73 / 104.02 | **48/48** | 101.56 |
| **P3** 16 tiles + HIGH, t1 | LOW / **HIGH** | 0.17 / 0.33 / 1.37 | 4.55 / 9.49 / 18.56 | 1/297 | 142.86 |
| **P3** 16 tiles + HIGH, t2 | LOW / **HIGH** | 0.50 / 1.61 / 2.22 | 5.81 / 10.31 / 16.71 | 1/297 | 142.91 |
| **P3b** 16 tiles, no priority, t1 | LOW / MEDIUM | 0.17 / 0.39 / 1.70 | 4.68 / 9.27 / 13.91 | 0/297 | 142.83 |
| **P3b** 16 tiles, no priority, t2 | LOW / MEDIUM | 0.24 / 0.62 / 1.61 | 4.94 / 9.44 / 13.43 | 0/297 | 142.67 |
| **P4** two processes, t1 | LOW / **HIGH** | — | 102.34 / 104.14 / 104.25 | **48/48** | 102.49 |
| **P4** two processes, t2 | LOW / **HIGH** | — | 102.60 / 104.18 / 104.38 | **47/48** | 102.69 |

Read the first six rows together. **The light context was granted HIGH in P1, P2 and P4, and it
made no difference whatsoever**: 101.05, 101.25, 101.54 and 101.37 ms in the two-thread cases,
against the equal-priority control's 101.47 and 101.44. Those four sit within 0.4 ms of the
control, and two of them are *slower* than it. P4's 102.34 and 102.60 are about 1 ms higher, but
that is the two-process harness rather than the priority — the same offset appears in its heavy
cost (102.49 / 102.69 against 101.4), and the earlier experiment saw the same thing when it
compared processes to threads at equal priority. A HIGH-priority context waits out a
LOW-priority 100 ms draw for as long as an equal-priority one does, whether the two contexts live
in one process or two.

### What a priority costs when it runs alone (P5)

`prio_solo_cost.py`, one context, nothing else on the GPU, median of 24 frames:

| Requested | Granted | 1 tile, t1 / t2 | 16 tiles, t1 / t2 |
|---|---|---|---|
| default | MEDIUM | 102.55 / 102.93 | — |
| LOW | LOW | 102.63 / 103.12 | 143.95 / 145.19 |
| MEDIUM | MEDIUM | 102.11 / 102.72 | — |
| HIGH | HIGH | 102.30 / 103.01 | 143.78 / 145.27 |

A priority is free and it is also worthless. Every level renders the same document in the same
~102-103 ms, and a tiled document costs ~144 ms whether the context asking for it is LOW or
HIGH. Asking for HIGH neither buys throughput nor costs any.

### Does priority reduce the tiling tax? (P3 vs P3b)

This is the question worth the most to the feature, because tiling's 25-35% document tax is the
price the earlier experiment's design pays. If priority shrank it, fewer tiles would suffice.

It does not. Tiles plus HIGH measured **4.55 and 5.81 ms light median with the heavy document at
142.86 and 142.91 ms**; tiles alone measured **4.68 and 4.94 ms with the document at 142.83 and
142.67 ms**. The document cost is identical to within 0.2 ms, which is a quarter of the gap
between the two trials of either configuration. The tiles do all the work, and the priority does
none of it — in fact the only missed frames in the whole tiled group (1/297, twice) are in the
HIGH rows, which is noise rather than a penalty but is certainly not a benefit.

## How mutter does it — and why it is not what is happening here

mutter does ask for a high-priority context, and the code is exactly where the premise expected.
From `cogl/cogl/winsys/cogl-winsys-egl.c` at tag 46.2 (the version installed here), inside
`try_create_context`:

```c
  if (egl_renderer->private_features &
      COGL_EGL_WINSYS_FEATURE_CONTEXT_PRIORITY)
    {
      attribs[i++] = EGL_CONTEXT_PRIORITY_LEVEL_IMG;
      attribs[i++] = EGL_CONTEXT_PRIORITY_HIGH_IMG;
    }
```

and, after creation, it does precisely what the spec section above says an application must do —
it does not trust the request:

```c
  if (egl_renderer->private_features &
      COGL_EGL_WINSYS_FEATURE_CONTEXT_PRIORITY)
    {
      EGLint value = EGL_CONTEXT_PRIORITY_MEDIUM_IMG;

      eglQueryContext (egl_renderer->edpy,
                       egl_display->egl_context,
                       EGL_CONTEXT_PRIORITY_LEVEL_IMG,
                       &value);

      if (value != EGL_CONTEXT_PRIORITY_HIGH_IMG)
        g_message ("Failed to obtain high priority context");
      else
        g_message ("Obtained a high priority EGL context");
    }
```

**But that file does not run on this session.** It is the EGL winsys, and mutter picks a winsys
per backend. From `src/backends/x11/meta-renderer-x11.c` at 46.2:

```c
static const CoglWinsysVtable *
get_x11_cogl_winsys_vtable (CoglRenderer *renderer)
{
#ifdef HAVE_EGL_PLATFORM_XLIB
  if (meta_is_wayland_compositor ())
    return _cogl_winsys_egl_xlib_get_vtable ();
#endif

  switch (renderer->driver)
    {
    case COGL_DRIVER_GLES2:
#ifdef HAVE_EGL_PLATFORM_XLIB
      return _cogl_winsys_egl_xlib_get_vtable ();
#else
      break;
#endif
    case COGL_DRIVER_GL3:
#ifdef HAVE_GLX
      return _cogl_winsys_glx_get_vtable ();
```

This session is `XDG_SESSION_TYPE=x11` (so `meta_is_wayland_compositor()` is false) and cogl
tries `COGL_DRIVER_GL3` first — it heads the driver list in `cogl/cogl/cogl-renderer.c`, and
gnome-shell's environment sets no `COGL_DRIVER` override, so a GL 4.6 NVIDIA card takes it. That
lands on `_cogl_winsys_glx_get_vtable()`: **mutter composites this desktop through GLX, and its
EGL priority request never executes.** Consistent with that, `journalctl` for this boot contains
neither "Obtained a high priority EGL context" nor "Failed to obtain high priority context".

So the compositor's priority request cannot be the explanation for a smooth desktop here. The
prior art is real, but it applies to a Wayland session (or a GLES2 X11 one), not this one.

### What keeps the desktop smooth, then?

Not context priority — and, it turns out, not "CPU clients avoid the GPU queue" either, which was
the obvious next hypothesis. `prio_desktop_x11.py` draws a 640x360 window purely on the CPU with
`XPutImage` + `XSync`, never touching GL, and times its period against the 100 ms document:

```
cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run --with python-xlib python ai_docs/features/090_render_decoupling/probes/prio_desktop_x11.py
```
```
  X11 CPU window, GPU idle      t1: median=   1.87  p95=   2.89  max=  14.35  over-16.7ms 0/2411
    X round-trip:                   median=   0.04  p95=   0.05  max=   6.28
  X11 CPU window, 100ms document t1: median= 102.21  p95= 411.46  max= 504.89  over-16.7ms 25/41
    X round-trip:                   median=   0.06  p95=   0.08  max= 102.06
  X11 CPU window, GPU idle      t2: median=   1.83  p95=   2.64  max=   7.53  over-16.7ms 0/2564
    X round-trip:                   median=   0.04  p95=   0.05  max=   2.05
  X11 CPU window, 100ms document t2: median=   1.88  p95= 513.49  max= 619.81  over-16.7ms 16/34
    X round-trip:                   median=   0.06  p95= 103.26  max= 210.69
```

**A CPU-drawn X client is stalled by the GPU document just as badly as a GL client is** — 102.21
ms against its own 1.87 ms idle floor. The round-trip line separates the two things that could
mean: a bare `GetInputFocus` round-trip to Xorg, which does no drawing at all, stays at **0.06 ms
median** throughout. So neither the client nor the X server's request handling is starved; what
costs 102 ms is the *drawing*. On this driver Xorg renders `XPutImage` on the GPU, so a "CPU-drawn"
X client is not outside the GPU queue at all, it is merely in it at one remove.

Trial 2 landed differently and the difference is worth keeping rather than smoothing away: median
1.88 ms but p95 513.49 ms and max 619.81 ms, with 16 of 34 frames over budget and the round-trip
p95 blown out to 103.26 ms. The client is bimodal — several `XPutImage` batches slip through
cheaply and then one pays for all of them. Both trials agree on the thing that matters (the CPU
window is severely disrupted, 25/41 and 16/34 frames over a 16.7 ms budget against 0/2411 and
0/2564 when the GPU is idle); they disagree on where in the distribution the cost shows up, so the
median alone would misrepresent trial 2 and the p95/max alone would misrepresent trial 1.

That leaves the maintainer's observation genuinely unexplained by any of the mechanisms tested
here, and worth stating as such rather than papering over: the desktop's smoothness is not
context priority (mutter's request does not run on this session), and it is not CPU-versus-GPU
drawing (a pure CPU X client stalls identically). The most likely remaining explanation is that
ShaderBox's real document is not a single 100 ms draw the way this synthetic one is; a desktop
that stays smooth beside the real app is evidence about the real app's draw granularity, not
evidence that the driver schedules anyone ahead of a long draw. The 100 ms single-draw case,
which is what the feature has to survive, stalls everything on this machine without exception.

## Can the app adopt priorities without replacing glfw?

No. Three separate gates, each checked against source or a run.

**glfw has no priority hint.** glfw 3.4 defines no `GLFW_CONTEXT_PRIORITY`-anything; pyGLFW 2.10.0
exposes no symbol matching `PRIORITY` (`prio_glfw_check.py` asserts this at runtime and prints
`False`). There is no hint to set and no documented way to inject attributes into glfw's own
`eglCreateContext` call.

**`GLFW_CONTEXT_CREATION_API=GLFW_EGL_CONTEXT_API` gets an EGL context, but only a MEDIUM one.**
This is the one opening worth testing, because glfw's native-access header exposes
`glfwGetEGLContext`/`glfwGetEGLDisplay`, so the resulting context is at least addressable:

```
cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run python ai_docs/features/090_render_decoupling/probes/prio_glfw_check.py
```
```
glfw b'3.4.0 X11 GLX Null EGL OSMesa monotonic shared'
pyGLFW has a priority hint: False

  NATIVE_CONTEXT_API (default; GLX on X11): glfw.get_egl_context -> None, get_egl_display -> None
    no EGL handles: this context is not an EGL context (GLX), nothing to query
  EGL_CONTEXT_API: glfw.get_egl_context -> 564877057, get_egl_display -> 564794560
    eglQueryContext priority: MEDIUM
```

The handles are real and queryable, which confirms the EGL path works — and confirms the context
lands on the spec's MEDIUM default. Native access is read-only here: priority is fixed at
`eglCreateContext` time and there is no EGL call to change it afterwards, so getting the handle
back after the fact is too late. Note also that the default `NATIVE_CONTEXT_API` gives a GLX
context, which has no EGL priority concept at all — so today's ShaderBox window could not be
queried, let alone prioritised.

**moderngl's `glcontext` EGL backend accepts no extra attributes.** `glcontext/egl.cpp`,
`meth_create_context`, line 94 in the installed 2.3.7:

```c
    static char * keywords[] = {"mode", "libgl", "libegl", "glversion", "device_index", NULL};
```

Five keywords, none of them an attribute list. The context attributes are a fixed literal, lines
261-267 for `mode="standalone"`:

```c
        int ctxattribs[] = {
            EGL_CONTEXT_MAJOR_VERSION, glversion / 100 % 10,
            EGL_CONTEXT_MINOR_VERSION, glversion / 10 % 10,
            EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
            // EGL_CONTEXT_OPENGL_FORWARD_COMPATIBLE, 1,
            EGL_NONE,
        };
```

and again at lines 321-327 for `mode="share"`. Current upstream glcontext 3.0.0 has the identical
keyword list and the identical literals, so this is not a matter of upgrading.

The conclusion is unambiguous, and it is moot: a priority-carrying UI context would have to be
created outside glfw with raw EGL, which is a real cost — and since priority changes nothing on
this driver, there is nothing to buy with it.

## Interpretation

The priority hint is honoured as an *attribute* and ignored as a *scheduling policy*. This driver
grants all three levels to an unprivileged process, reports them back distinctly through
`eglQueryContext`, and then schedules identically regardless: a context holding HIGH waits the
full 100 ms behind a context holding LOW, and the spread across every priority pairing is smaller
than the run-to-run spread of the equal-priority control.

That is consistent with the earlier experiment rather than a surprise on top of it.
`gpu_preemption.md` found the scheduling quantum is one draw call — a 100 ms `glDrawArrays` runs
to completion and everything else queues behind it. A priority can only decide *which work runs
next*; it cannot decide *when the current work stops*. With no preemption, "next" does not arrive
until the 100 ms draw is finished, and by then there is nothing left to prioritise. Priority and
preemption are different mechanisms, and this driver exposes the first without the second.

The practical consequence for feature 090 is that the design does not change. Bounding each
draw's duration remains the only lever, tiling remains the mechanism, and the 25-35% document tax
remains the price — priority does not reduce it by a measurable amount (142.86 ms with, 142.83 ms
without). The one thing this experiment adds is that the option is now closed with a number
rather than left open as a hope, and closed cheaply: no glfw replacement, no EGL port, nothing to
maintain.

Scope note: measured on one driver (NVIDIA 580.173.02), one GPU, one X11 session, one document
shape. The `EGL_IMG_context_priority` extension is advertised by many drivers and its behaviour is
explicitly implementation-defined, so "granted but not scheduled" is a fact about this driver, not
about the extension. A Wayland session, an AMD or Intel driver, or a GPU with hardware preemption
could each answer differently.

## False trails

**moderngl cannot be attached to an external EGL context on this stack, and it fails in a way
that looks like success.** The plan was to wrap the EGL contexts with `moderngl.create_context()`
so the draw code would match the earlier probes exactly. `glcontext` 2.3.7's EGL backend has no
`detect` mode at all — `egl.cpp`'s `meth_create_context` accepts only `"standalone"` and
`"share"`, and anything else returns `"unknown mode"`. The default x11/GLX backend does have a
`detect` mode, but it requires `glXGetCurrentContext()` to be non-NULL, and that returns NULL when
the thread's current context came from `eglMakeCurrent`.

The trap is that `moderngl.create_context()` with no arguments *appeared* to work on the main
thread. It does not consult glcontext at all in that case — it returns the cached
`_store.default_context`. The same call on a worker thread raises `(detect) glXGetCurrentContext:
cannot detect OpenGL context`, which is what exposed it. Had the probes only ever run on the main
thread, they would have measured a stale context while reporting the right renderer string. The
probes use raw PyOpenGL instead, with the byte-identical shaders from `common.py`; the P0 control
reproducing the earlier experiment's 101.4 ms is what confirms the substitution measures the same
thing.

**PyOpenGL needs `PYOPENGL_PLATFORM=egl`, set before `OpenGL` is imported.** PyOpenGL tracks
per-context client state through its platform module, and the default platform on Linux is GLX,
whose `glXGetCurrentContext()` is NULL under an EGL context — so `glVertexAttribPointer` dies with
`OpenGL.error.Error: Attempt to retrieve context when no valid context`. Setting it inside
`prio_common.py` was not enough, because the probes import `common.py` first and that pulls in
`moderngl`/`glfw` and therefore `OpenGL`. Each probe now sets it as its first statement.

**`EGL_CONTEXT_PRIORITY_REALTIME_NV` is not available.** `/usr/include/EGL/eglext.h` defines it at
0x3357, which makes it look like a fourth level worth trying. It belongs to
`EGL_NV_context_priority_realtime`, which this driver's `EGL_EXTENSIONS` string does not
advertise; only `EGL_IMG_context_priority` with its three levels is present. A header definition
is not a driver capability.

**The 20% GPU-idle guard could not pass.** `gpu_clear.wait_for_idle` refuses to measure above 20%
utilization. During the first P1 attempt the maintainer's own ShaderBox instance was running and
held the GPU at a steady 35-40%, so the guard timed out with "GPU never went idle" and measured
nothing. `prio_guard.py` replaces it with a baseline-relative version: it samples utilization,
takes the median as the session's floor, and refuses only when utilization climbs 25 points above
that floor. Every run prints the baseline it measured. The guard change is not a silent lowering
of standards — the heavy document was re-timed under a 43% baseline and cost 100.4 ms against the
100.93 ms calibrated on an idle GPU, and P1 trial 1 (baseline 43%, app running) and trial 2
(baseline ~14%, app closed) returned 101.05 and 101.25 ms. The background load does not move the
quantity under test.

**python-xlib cannot `XPutImage` a whole 640x360 window in one request.** The image is 921 KB and
python-xlib packs the request length into a 16-bit field, so it raises `struct.error: 'H' format
requires 0 <= number <= 65535`. The probe sends 16-row bands, which is what a real toolkit does
anyway; the whole window still lands before each `XSync`.

**"CPU-drawn clients bypass the GPU queue" is false here.** It was the natural explanation for a
smooth desktop once mutter's GLX path ruled out priority, and it does not survive measurement —
see the desktop section above.

## Verdict

**REFUTES** — for the claim *"a high-priority UI context stays at ~16 ms beside a single 100 ms
low-priority draw on this driver"*.

The deciding number: with the light context **granted HIGH** and the heavy context **granted
LOW**, both confirmed by `eglQueryContext`, the light loop measured **101.05 and 101.25 ms median
with 47/50 and 48/48 frames missing the 16.7 ms budget** — against an equal-priority control in
the same harness measuring 101.47 and 101.44 ms with 48/48 missed. **The widest available priority
gap bought 0.3 ms of a 100 ms stall** — 0.3% — where keeping the budget would have required
saving 85 ms. The same result holds at the realistic MEDIUM-versus-HIGH gap (101.54, 101.37) and
across two OS processes (102.34, 102.60).

**REFUTES** — for the second claim, *"priority reduces the tiling tax"*.

16 tiles with the light context granted HIGH cost the document **142.86 and 142.91 ms**; the same
16 tiles with no priority at all cost **142.83 and 142.67 ms**. The difference is under 0.2 ms,
roughly a quarter of the gap between either configuration's own two trials. Solo, the document
costs ~102-103 ms at every priority level and ~144 ms tiled at both LOW and HIGH. Priority is
free, and it buys nothing: the tax is paid to tiling and priority does not discount it.
