# 090 render_decoupling — prior art

Research for moving document rendering off the UI thread onto a worker thread with a
second, shared GL context. Stack: moderngl 5.12.0 (compiled extension; context creation
delegated to the `glcontext` package since moderngl 5.6) + glfw (Python binding 2.10.0,
wrapping GLFW 3.4) + imgui-bundle, X11, NVIDIA proprietary driver 580.

Every claim below carries a URL that was actually fetched and a short verbatim quote.
Where a primary source could not be reached, that is stated explicitly and the claim is
marked unverified rather than filled in from memory.

Local version check performed directly in the repo venv:

```
$ uv run python -c "import moderngl; print(moderngl.__version__)"
5.12.0
$ uv run python -c "import glfw; print(glfw.__version__)"
2.10.0
```

---

## 1. GLFW 3.4 — contexts and threads

**Main-thread-only calls.**
https://www.glfw.org/docs/3.4/intro_guide.html
> "Most GLFW functions must only be called from the main thread, but some may be called
> from any thread once the library has been initialized. [...] Initialization, termination,
> event processing and the creation and destruction of windows, cursors and OpenGL and
> OpenGL ES contexts are all restricted to the main thread due to limitations of one or
> several platforms."

Confirmed per-function on https://www.glfw.org/docs/3.4/group__window.html: `glfwCreateWindow`,
`glfwPollEvents`, `glfwWaitEvents`, `glfwWaitEventsTimeout` each carry "**Thread safety:** This
function must only be called from the main thread."

**`glfwCreateWindow(..., share)` semantics.**
https://www.glfw.org/docs/3.4/group__window.html — `share` parameter: "The window whose
context to share resources with, or `NULL` to not share resources."
https://www.glfw.org/docs/3.4/context_guide.html
> "When creating a window and its OpenGL or OpenGL ES context with glfwCreateWindow, you can
> specify another window whose context the new one should share its objects (textures, vertex
> and element buffers, etc.) with." ... "Object sharing is implemented by the operating system
> and graphics driver. On platforms where it is possible to choose which types of objects are
> shared, GLFW requests that all types are shared."

`share` gives the new context the existing window's *object namespace* (textures/buffers/etc.),
not its GL state (bindings, current program) — state is always per-context. This is the
documented mechanism for `090`: create a hidden render window sharing with the visible UI window.

**Making a context current on another thread.**
https://www.glfw.org/docs/3.4/group__context.html — `glfwMakeContextCurrent`: "**Thread
safety:** This function may be called from any thread."
https://www.glfw.org/docs/3.4/context_guide.html
> "A context can only be current for a single thread at a time, and a thread can only have a
> single context current at a time." ... "When moving a context between threads, you must make
> it non-current on the old thread before making it current on the new one."

Making the render context current on the worker thread (a thread other than the one that
called `glfwCreateWindow`) is explicitly allowed and is GLFW's documented intended use,
subject to the 1-context-per-thread / 1-thread-per-context-at-a-time invariant.

**Hidden windows as offscreen contexts.**
https://www.glfw.org/docs/3.4/context_guide.html (Offscreen contexts section)
> "GLFW doesn't support creating contexts without an associated window. However, contexts with
> hidden windows can be created with the GLFW_VISIBLE window hint." ... "The window never needs
> to be shown and its context can be used as a plain offscreen context." Caveat: "Depending on
> the window manager, the size of a hidden window's framebuffer may not be usable or
> modifiable, so framebuffer objects are recommended for rendering with such contexts."

This is GLFW's own sanctioned pattern for a worker-thread render context: create a
`GLFW_VISIBLE=FALSE` window on the main thread (rule above), then hand its context to the
worker thread. The FBO caveat matters if the worker ever needs a usable default framebuffer —
render to an FBO, not the hidden window's own backbuffer.

**`glfwSwapBuffers`.**
https://www.glfw.org/docs/3.4/group__window.html
> "This function swaps the front and back buffers of the specified window... This is typically
> called after rendering, but before returning control to the operating system." — "**Thread
> safety:** This function may be called from any thread."

Callable from any thread (operates on whichever context is current on the calling thread); can
block on vsync depending on swap interval, but that is driver/vsync blocking, not an
event-queue interaction. **Unverified sub-point:** no single primary-source sentence directly
states "`glfwSwapBuffers` does not pump the event queue" — this is inferred from the two
functions' separately documented, disjoint scopes (swap = any-thread buffer op; event pump =
main-thread-only), not a direct quote.

**`glfwPostEmptyEvent`.**
https://www.glfw.org/docs/3.4/group__window.html — "This function posts an empty event to the
event queue." — "**Thread safety:** This function may be called from any thread."
https://www.glfw.org/docs/3.4/input_guide.html
> "If your main thread is blocked in glfwWaitEvents... you can wake it from another thread by
> posting an empty event to the event queue with glfwPostEmptyEvent()."

Exact documented purpose: wake a main thread blocked in `glfwWaitEvents`/`glfwWaitEventsTimeout`
from any other thread. Relevant if the render worker needs to nudge the UI thread out of a wait
when a frame is ready.

**General reentrancy note.**
https://www.glfw.org/docs/3.4/intro_guide.html
> "GLFW event processing and object destruction are not reentrant." Listed non-reentrant:
> `glfwDestroyWindow`, `glfwDestroyCursor`, `glfwPollEvents`, `glfwWaitEvents`,
> `glfwWaitEventsTimeout`, `glfwTerminate`.

**Sanctioned pattern per GLFW's own docs:** create the visible UI window and a
`GLFW_VISIBLE=FALSE` render window on the main thread via `glfwCreateWindow(..., share=ui_window)`;
hand the hidden window's context to the worker thread via `glfwMakeContextCurrent` (any-thread-safe);
worker calls `glfwSwapBuffers` on its own context without touching the event queue; main thread
keeps exclusive ownership of `glfwPollEvents`/`glfwWaitEvents`; `glfwPostEmptyEvent` to wake the
main thread if it's waiting when a worker frame completes.

---

## 2. moderngl 5.12.0

**`create_context(share=True)`.**
https://raw.githubusercontent.com/moderngl/moderngl/5.12.0/docs/topics/context.rst
("Context Sharing" section)
> ".. Warning:: Object sharing is an experimental feature" ... "Some context support the
> `share` parameters enabling object sharing between contexts. This is not needed if you are
> attaching to existing context with share mode enabled. For example if you create two windows
> with glfw enabling object sharing." ... "ModernGL objects (such as `moderngl.Buffer`,
> `moderngl.Texture`, ..) has a `ctx` property containing the context they were created in.
> Still **ModernGL do not check what context is currently active when accessing these
> objects.**" ... "there are some limitations to object sharing. Especially objects that
> reference other objects (framebuffer, vertex array object, etc.)"

Since moderngl 5.6, context creation is delegated to the separate `glcontext` package. Source
at `glcontext/x11.cpp` (GLX backend), the `share` code path:
> `GLXContext ctx_share = res->m_glXGetCurrentContext();` ... `res->ctx =
> res->m_glXCreateContextAttribsARB(res->dpy, *res->fbc, ctx_share, true, attribs);`

`share=True` in moderngl's own `create_context()` means "share with whatever GLX context is
current **on the calling thread right now**" — not "share with a specific context object passed
in" — and requires the source context to already be current on the calling thread. For 090's
own GLFW-managed windows, `glfwCreateWindow(..., share=ui_window)` is the mechanism actually used
(sharing is implemented at the GLFW/GLX level below moderngl); moderngl's `share=True` matters
only if moderngl is asked to create the context itself rather than wrapping a GLFW-created one.

**Fence/sync objects — absent from the public API.**
https://raw.githubusercontent.com/moderngl/moderngl/5.12.0/src/gl_methods.hpp — the GL function
loader resolves `FenceSync`/`IsSync`/`DeleteSync`/`ClientWaitSync`/`WaitSync`/`GetSynciv`
pointers, but a full grep of `moderngl.cpp` (~7800 lines) at the 5.12.0 tag found zero call
sites for any of them outside that loader table, and no Python method wraps them. Locally
confirmed independently: `grep -in "fence\|sync"` against the installed `moderngl/__init__.py`
returns nothing. **Fence/sync objects are genuinely unreachable from moderngl's public API** —
not a docs gap. Any cross-context synchronization 090 needs (see §3) must go through raw
`ctypes`/`ModernGL`-adjacent GL calls or a small custom binding, not moderngl itself.

**Documented/maintainer statements on threads.**
No moderngl docs page mentions threads. The issue tracker is the primary source. Maintainer
`einarf`, GitHub issue #623 ("Can't create shader program from a python thread",
https://github.com/moderngl/moderngl/issues/623):
> "if you are using threads you must handle opengl context switching, the opengl context can be
> current only in one thread. [...] also please do keep in mind that due to the Python GIL you
> have no real benefits of using threads unless those are idling on IO without the GIL being
> held. the OpenGL driver is not multithreaded so whatever you do with it on multiple threads it
> will land in sync calls one after the other. [...] If you want to put rendering on a dedicated
> thread, that may be totally reasonable and you should make the opengl context current there
> and leave it like that."

Same issue, reporter's resolution: "I've ended up using a dedicated render process which works
quite nicely" (abandoned threads for a process in that case).

Issue #398 ("Buffer access from a separate Thread",
https://github.com/moderngl/moderngl/issues/398): a `.read()` call from a non-owning thread
throws `cannot map the buffer`; never resolved by a maintainer.

Issue #414 ("Multiple contexts hangs when context is created in main thread",
https://github.com/moderngl/moderngl/issues/414), `einarf`: "Yes, it does hang for me when
context is also created in main thread. I'm not entirely sure why." — open, acknowledged,
unexplained hang in a specific cross-process/thread context-creation ordering.

The maintainers' repeated, explicit guidance is exactly 090's target shape: one GL context
current on exactly one thread, for the lifetime of that thread — never migrated back and forth,
never shared by two threads taking turns.

**`Context.__enter__`/`__exit__`.**
https://raw.githubusercontent.com/moderngl/moderngl/5.12.0/moderngl/__init__.py (confirmed
locally too, lines ~2183-2189):
```
def __enter__(self):
    self.mglo.__enter__()
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    self.mglo.__exit__(exc_type, exc_val, exc_tb)
```
Forwards to `MGLContext_enter`/`_exit` in `moderngl.cpp`, which call `self->ctx->__enter__()` /
`__exit__()` on the `glcontext` backend object. On Linux (`glcontext/x11.cpp`), enter calls
`glXMakeCurrent(dpy, wnd, ctx)` (saving the previously-current display/window/context); exit
restores the previous context (or clears current if there was none). `with ctx:` is a real
make-current/restore-previous pair, scoped to whichever thread executes the `with` block — the
correct mechanism for a worker thread to activate its context.

**Thread-affinity diagnostics.** No dedicated issue combines "thread" + "context" + "current",
but #623 and #398 show the actual failure mode: calling `ctx.program()` / `.vao()` /
`buffer.read()` from a thread where the context is not current raises generic driver-level
errors (`cannot create program`, `cannot map the buffer` — confirmed as literal strings in
`moderngl.cpp`), not a clear "wrong thread" diagnostic. moderngl exposes no current-context
registry of its own; tracking is entirely the OS GL driver's per-thread state via
`glXMakeCurrent`. **090's design needs its own guard** (e.g. an assertion that the worker thread
owns/holds its context before any GL call) — moderngl will not catch misuse for you.

**Free-threaded wheels.** PyPI JSON API (`pypi.org/pypi/moderngl/5.12.0/json`) and the PyPI
files page confirm moderngl 5.12.0 ships wheels for `cp38` through `cp313` (standard ABI) plus
one sdist — no `cp3XXt` (free-threaded) tag exists for any platform. See §6.

---

## 3. OpenGL object sharing between contexts (spec)

Sources fetched: OpenGL 4.6 Core spec PDF
(`https://registry.khronos.org/OpenGL/specs/gl/glspec46.core.pdf`, via curl + `pdftotext`),
Chapter 5 "Shared Objects and Multiple Contexts", pp. 53-58; the `GL_ARB_sync` extension spec
(`https://registry.khronos.org/OpenGL/extensions/ARB/ARB_sync.txt`). **Attempted but
inaccessible:** the Khronos wiki (`www.khronos.org/opengl/wiki/*`) — every fetch (WebFetch and
curl) failed with TLS/403/redirect errors, an apparent network-level block on that host from
this environment. `registry.khronos.org` was reachable and is authoritative regardless (the
wiki paraphrases these same documents).

**What is shared.**
Spec §5 opening, p. 53:
> "Objects that may be shared between contexts include buffer objects, program and shader
> objects, renderbuffer objects, sampler objects, sync objects, and texture objects (except for
> the texture objects named zero)."

Shareable set: buffer objects (VBO/UBO/SSBO are all "buffer objects"), programs/shaders,
renderbuffers, samplers, **sync objects**, textures (except the default texture object 0).

**What is not shared — container objects.**
Same page, immediately following:
> "Objects which contain references to other objects include framebuffer, program pipeline,
> transform feedback, and vertex array objects. Such objects are called container objects and
> are not shared."

VAOs, FBOs, program pipeline objects, and transform feedback objects are explicitly named
non-shared. Query objects are not named in either enumeration sentence; the spec treats an
active query as bound to the context that began it, so queries are per-context in practice, but
this is not from the same explicit enumeration.

**Sync objects specifically ARE shareable** — confirmed in `ARB_sync.txt`, Issues §1:
> "1) Are sync objects shareable between multiple contexts? RESOLVED: YES. The sync object
> namespace is shared, and sync objects themselves may be shared or not. Shared sync objects
> can be blocked upon or deleted from any context they're shared with."

A fence created in the producer context can be waited on directly from the consumer
context/thread — this is the mechanism 090 needs for handoff, workable in raw GL even though
moderngl doesn't wrap it (§2).

**Synchronization requirement — Finish/fence, not bare Flush.**
Spec §5.3.1 "Determining Completion of Changes to an object", p. 56:
> "Completion of a command may be determined either by calling Finish, or by calling FenceSync
> and executing a WaitSync command on the associated sync object. The second method does not
> require a round trip to the GL server and may be more efficient, particularly when changes to
> T in one context must be known to have completed before executing commands dependent on those
> changes in another context."

Spec §5.3.3, Rule 3, p. 58:
> "Changes to the contents of shared objects are not automatically propagated between contexts.
> If the contents of a shared object T are changed in a context other than the current context,
> and T is already directly or indirectly attached to the current context, any operations on the
> current context involving T via those attachments are not guaranteed to use its new contents."

The spec presents `Finish` (blocking) or `FenceSync`+`WaitSync` (non-blocking, preferred) as the
two documented mechanisms for cross-context completion — **not** `glFlush` alone.

**The `glFlush`-on-producer requirement.** `ARB_sync.txt` §5.2.2 "Signaling" + footnote 4:
> "If the sync object being blocked upon will not be signaled in finite time (for example, by an
> associated fence command issued previously, but not yet flushed to the graphics pipeline),
> then ClientWaitSync may hang forever. [...] if the SYNC_FLUSH_COMMANDS_BIT bit is set [...]
> then the equivalent of Flush will be performed before blocking on <sync>." Footnote 4: "The
> simple flushing behavior defined by SYNC_FLUSH_COMMANDS_BIT will not help when waiting for a
> fence command issued in another context's command stream to complete. Applications which
> block on a fence sync object must take additional steps to assure that the context from which
> the corresponding fence command was issued has flushed that command to the graphics pipeline."

`glClientWaitSync(..., GL_SYNC_FLUSH_COMMANDS_BIT, ...)`'s automatic flush only flushes the
*waiting* context's own stream, never the *producer's*. **The producer thread/context must call
`glFlush()` itself, after `glFenceSync`, before a consumer on another context/thread can safely
`glWaitSync`/`glClientWaitSync`** — otherwise the wait can hang indefinitely.

**Documented cross-context pattern for 090:** producer does
`sync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0); glFlush();`; consumer does
`glWaitSync(sync, 0, GL_TIMEOUT_IGNORED)` (GPU-side, no CPU stall) or
`glClientWaitSync(sync, GL_SYNC_FLUSH_COMMANDS_BIT, timeout)` (CPU-side blocking) before touching
the shared texture. Rule 4 (p. 58) additionally requires re-binding/re-attaching the shared
object at a binding point in the consumer context after the wait — completion alone is not
sufficient if the object was already attached before the change.

---

## 4. GPU scheduling — NVIDIA 580 and other vendors

Sources: NVIDIA Linux driver README 580.95.05
(`https://download.nvidia.com/XFree86/Linux-x86_64/580.95.05/README/openglenvvariables.html`);
NVIDIA developer blog on CUDA time-slicing; NVIDIA MPS docs; NVIDIA DriveOS/Tegra scheduling
docs; Linux kernel DRM scheduler source (`sched_main.c`).

**[DOCUMENTED] `__GL_YIELD` is a CPU-side scheduling knob, not a GPU scheduling control.**
> "There are several cases where the NVIDIA OpenGL driver needs to wait for external state to
> change before continuing. To avoid consuming too much CPU time in these cases, the driver will
> sometimes yield so the kernel can schedule other processes to run while the driver waits. For
> example, when waiting for free space in a command buffer... the driver will yield before it
> continues to loop... You can use the __GL_YIELD environment variable to work around these
> scheduling problems."

Values: unset (default, `sched_yield()`), `"NOTHING"` (never yields), `"USLEEP"` (`usleep(0)`).
This governs a CPU thread spinning on a CPU-side wait (e.g. command-buffer space) via the host
OS scheduler — it says nothing about how the GPU itself arbitrates between contexts.
`__GL_SYNC_TO_VBLANK` (same doc) is a swap/vsync toggle, likewise unrelated to multi-context
GPU scheduling.

**[DOCUMENTED — absent] No `__GL_MaxFramesAllowed`, preemption, or multi-context scheduling
language appears in that README chapter.** Full-text search for "MaxFramesAllowed",
"preemption", "time-slic", "context switch", "multiple contexts" returned nothing. This absence
is itself the finding: NVIDIA's official Linux README does not document desktop-driver GPU-side
arbitration between contexts, in the one place that would naturally cover it.

**[FOLKLORE/UNVERIFIED] "GeForce time-slices contexts round-robin; Quadro gets true
concurrency"** is a widely repeated community claim (e.g.
`forums.developer.nvidia.com/t/problems-with-multiple-opengl-applications-running-simultaneously-with-375.20-on-a-gtx970/46255`)
with no NVIDIA README/blog/docs backing found. Treat as plausible, not verified.

**[DOCUMENTED, scoped to CUDA only]** NVIDIA developer blog on Kubernetes GPU sharing:
> "you can use a simple oversubscription strategy to leverage the GPU's time-slicing
> scheduler... This technique, sometimes called temporal GPU sharing, does carry a cost for
> context-switching between the different CUDA applications", tied to "compute preemption
> starting with the Pascal architecture."

Real and documented, but written entirely in CUDA-application terms — never mentions OpenGL/GLX
contexts. Extrapolating this to two GL contexts on one process is inference, not documented
fact. NVIDIA's MPS docs (`docs.nvidia.com/deploy/mps/latest/`) describe MPS as CUDA-only
multi-process sharing with no mention of OpenGL anywhere — not applicable here.

**[DOCUMENTED, wrong product line]** NVIDIA's only official description of runlist-based,
round-robin time-sliced GPU scheduling with hardware preemption is written for the embedded
DriveOS/Tegra platform
(`developer.nvidia.com/docs/drive/drive-os/7.0.3/.../GPU_Scheduling_Improvements/...`), a
different codebase from desktop `nvidia.ko`. Citing it as proof of desktop behavior would be an
unjustified cross-product-line leap; it documents that the *concept* exists somewhere in
NVIDIA's stack, not that driver 580 on a GeForce/RTX card behaves the same way.

**Bottom line for 090:** nothing NVIDIA publishes describes how the desktop proprietary driver
schedules GPU-side work from two GL contexts (same process, two threads). `__GL_YIELD` and
`__GL_SYNC_TO_VBLANK` are the only two documented knobs in this area, and both are CPU-side.
Any claim about GPU-side time-slicing/preemption granularity for this exact setup is folklore.

**Other vendors (one paragraph, cited).** The DRM/kernel scheduler used by AMDGPU and (via a
shim) Intel i915/Xe is open and documented in-source. `drivers/gpu/drm/scheduler/sched_main.c`
(kernel source, AMD-authored):
> "The GPU scheduler provides entities which allow userspace to push jobs into software queues
> which are then scheduled on a hardware run queue... The scheduler selects the entities from
> the run queue using a FIFO... Each hw run queue has one scheduler[;] each scheduler has
> multiple run queues with different priorities... entities themselves maintain a queue of jobs
> that will be scheduled on the hardware."

FIFO-by-default (configurable to round-robin), dependency tracking via DMA fences
(`dma-resv`/`drm_syncobj`), and a credit-based flow-control limit per scheduler to prevent
hardware-queue starvation. This is the open-source counterpart to what NVIDIA does not
document — genuinely inspectable, unlike the closed desktop NVIDIA driver.

---

## 5. Comparable tools

**Blender.** The GPU module is built around one primary GHOST/GPU context owned by the main
thread, plus an explicit secondary-context API for worker threads. Current source
(`source/blender/gpu/GPU_context.hh`,
`projects.blender.org/blender/blender/raw/branch/main/source/blender/gpu/GPU_context.hh`):
> `void GPU_context_active_set(GPUContext *);` ... "Creates a secondary off-screen GHOST and GPU
> contexts. Must be called on the main thread." (`GPU_create_secondary_context()`) ... "Must be
> created from the main thread and destructed from the thread they where activated in." —
> `class GPUSecondaryContext`, "/** Must be called from a secondary thread. */ void activate();"

Legacy-hardware note in the same header:
> "Legacy GPU (Intel HD4000 series) do not support sharing GPU objects between GPU contexts. [...]
> When a legacy GPU is detected... any worker threads should use the draw manager opengl context
> and make sure that they are the only one using it by locking the main context using these two
> functions" (`GPU_context_main_lock()`/`_unlock()`).

This is the same pattern 090 targets: context created on the main thread, handed to a worker,
`activate()`/`_active_set()` called there before any GL call, never two threads driving one
context concurrently. **Unverified:** no narrative developer-docs page (`developer.blender.org`)
was found explaining the threading rationale in prose — only the header comments, which are
real, sourced, and load-bearing on their own.

**KodeLife / Bonzomatic.** Bonzomatic (https://github.com/Gargaj/Bonzomatic, source fetched
directly) is confirmed single-threaded: `src/main.cpp`'s main loop does input polling,
`Renderer::StartFrame()`/render/`EndFrame()`, and ImGui editor painting all in one
`while (!Renderer::WantsToQuit())` loop with no `std::thread`/`CreateThread`/`pthread` calls
anywhere in it. The smallest, most directly comparable livecoding shader tool does exactly what
ShaderBox does today — a valid finding, not a research gap. KodeLife (hexler.net/kodelife) is
closed-source with no architecture docs found; **unverified** for KodeLife specifically.

**Shadertoy.** Runs as WebGL in a browser tab — there is no app-level "render thread" to
inspect; the relevant boundary is the browser's own GPU process architecture (item below), not
anything Shadertoy's own JS manages.

**Chromium's GPU process model.**
https://www.chromium.org/developers/design-documents/gpu-accelerated-compositing-in-chrome/
> "Restricted by its sandbox, the Renderer process... cannot directly issue calls to the 3D APIs
> provided by the OS (GL / D3D)." ... "The client... serializes them and puts them in a ring
> buffer (the command buffer) residing in memory shared between itself and the server process."
> ... "The server (GPU process...) picks up the serialized commands from shared memory, parses
> them and executes the appropriate graphics calls."

A *process*, not just a thread — driven by sandbox security requirements that don't apply to
ShaderBox's single-trust-domain desktop app. The transferable idea is the command-buffer
shape: producer serializes work into a queue, consumer drains it asynchronously, neither side
touches the other's live GL state directly.

**Unreal Engine's Game Thread / Render Thread split.**
https://dev.epicgames.com/documentation/unreal-engine/threaded-rendering-in-unreal-engine
> "In Unreal Engine, the entire renderer operates in its own thread that is a frame or two
> behind the game thread." ... "The game thread inserts the command into the rendering command
> queue, and the rendering thread calls the Execute function when it gets around to it." ...
> "FRenderCommandFence provides a convenient way to track the progress of the rendering thread
> on the game thread" — `FlushRenderingCommands` is "the standard method of blocking the game
> thread until the rendering thread has caught up."

Same shape as Chromium's command buffer, minus the process boundary: UI/game logic never
touches render state directly, posts commands into a queue the render thread drains, and a
fence is the synchronization primitive when the producer needs to know the consumer caught up —
directly analogous to what 090 needs when, e.g., an export needs a completed frame from the
worker.

**Dear ImGui threading.** `docs/FAQ.md` in `ocornut/imgui` (raw file, "About Multi-Threading"):
> "A same Dear ImGui context may be not used from multiple threads in parallel." ... "If you
> want to submit contents from a main/update thread but render Dear ImGui output in a dedicated
> render thread, you'll need to stage ImDrawData and texture requests. See the
> ImDrawDataSnapshot and ImTextureQueue helpers in imgui_threaded_rendering." ... "If you use
> multiple Dear ImGui contexts and want to use them from multiple threads, you need to `#define
> GImGui` to become a TLS variable."

All `ImGui::*` calls and `ImGuiContext` access must stay on the main/UI thread. ImGui itself
doesn't need to move — that's the payload 090 wants to move off it — but if the worker thread
ever produces something ImGui displays directly (not just a texture handle bound normally),
that handoff needs a snapshot/queue mechanism, never a live pointer touched from both threads.

---

## 6. Python-specific: GIL, ctypes, free-threading

**CPython C-API — GIL release around native calls.**
https://docs.python.org/3/c-api/threads.html (the `init.html` thread-state/GIL material now
lives here)
> "Most extension code manipulating the thread state has the following simple structure: Save
> the thread state... Do some blocking I/O operation... Restore the thread state... This is so
> common that a pair of macros exists to simplify it: Py_BEGIN_ALLOW_THREADS ... Do some
> blocking I/O operation... Py_END_ALLOW_THREADS" ... "By detaching the thread state, the GIL is
> released, which allows other threads to attach to the interpreter and execute while the
> current thread performs blocking I/O." ... "it is also useful to call it over long-running
> native code that doesn't need access to Python objects or Python's C API. For example, the
> standard zlib and hashlib modules detach the thread state when compressing or hashing data."

The safety condition: code running inside a detached-thread-state block must not touch Python
objects or the C-API until the GIL is reacquired. Whether moderngl's own compiled extension
releases the GIL around individual GL calls is a moderngl-implementation question this research
did not find documented either way in moderngl's own docs — worth checking moderngl's Cython
source directly at implementation time if GIL contention turns out to matter.

**`ctypes` default GIL behavior.** https://docs.python.org/3/library/ctypes.html
> "The Python global interpreter lock is released before calling any function exported by these
> libraries, and reacquired afterwards." (`CDLL`) ... "`PyDLL` instances... the Python GIL is
> not released during the function call."

Standard `ctypes.CDLL` (the common FFI pattern for wrapping a C/GL library) releases the GIL by
default per native call — calling into GL through ctypes doesn't by itself block other Python
threads, subject to the same "no Python API re-entry while released" constraint from the C-API
docs above.

**PEP 703 / free-threading status — a note, not a design driver.**
https://peps.python.org/pep-0703/
> "This PEP proposes adding a build configuration (--disable-gil) to CPython to let it run
> Python code without the global interpreter lock... The GIL is a major obstacle to
> concurrency."

https://docs.python.org/3/howto/free-threading-python.html
> "Starting with the 3.13 release, CPython has support for a build of Python called free
> threading where the global interpreter lock (GIL) is disabled." ... "Some third-party
> packages, in particular ones with an extension module, may not be ready for use in a
> free-threaded build, and will re-enable the GIL." ... "The GIL may also automatically be
> enabled when importing a C-API extension module that is not explicitly marked as supporting
> free threading."

**Confirmed directly from PyPI** (`pypi.org/pypi/moderngl/5.12.0/json` and the files page):
moderngl 5.12.0 ships wheels for `cp38` through `cp313` only (standard GIL ABI), across
Windows/macOS/Linux — **no `cp313t`/`cp314t` free-threaded wheel tag exists**. Free-threading is
not usable for this project today regardless of what 090 chooses architecturally: even under a
free-threaded interpreter, moderngl would force the GIL back on (or fail to import cleanly as
an unmarked extension). This is purely a status note for future revisit, not something 090's
design should route around.

---

## 7. Alternatives to a render worker thread

**Separate render process, shared memory / DMA-BUF.**
https://docs.kernel.org/driver-api/dma-buf.html
> "The dma-buf subsystem provides the framework for sharing buffers for hardware (DMA) access
> across multiple device drivers and subsystems, and for synchronizing asynchronous hardware
> access."

Corroborated at the application level by Chromium's GPU process model (§5): GL calls flow over
IPC to a dedicated process, with EGL-image/dmabuf or shared-memory buffer handoff avoiding a
CPU-side copy where available. Buys: full fault isolation (a driver crash/hang in the render
process doesn't take imgui/glfw down with it) and true OS-level scheduling/preemption between
processes. Costs: no shared GL context across a process boundary (needs explicit dmabuf/EGL-image
plumbing or IPC-marshaled shared memory instead of moderngl's simple `share=` object sharing), a
real serialization/IPC layer, and without dmabuf an extra copy per frame. Substantially heavier
than a worker thread for a single-window desktop app.

**Tiled/chunked rendering as cooperative preemption points.**
https://developer.blender.org/docs/features/cycles/tiling/
> "A GPU scheduling tile can be small and is used for incrementally scheduling more paths to be
> rendered to keep the GPU occupied. Small tiles in GPU scheduling can also help improve
> coherence, by rendering nearby pixels on the same multiprocessor."

Cycles deliberately keeps GPU dispatch granularity small so the CPU-side loop retains a
checkpoint between dispatches — submit a chunk, check a cancel/yield flag, submit the next
chunk. Buys: bounded worst-case latency on the main thread, no second context needed at all.
Costs: extra bookkeeping (chunk iteration state, partial-result compositing), and it does
nothing if a *single* dispatch (one large compute pass, one slow full-screen fragment shader)
is itself the bottleneck — there's no granularity left to preempt within one draw call.

**Timeslicing via `glFlush` between passes.**
https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glFlush.xml (canonical Khronos
reference page for the call)
> "glFlush empties all of these buffers, causing all issued commands to be executed as quickly
> as they are accepted by the actual rendering engine. Though this execution may not be
> completed in any particular time period, it does complete in finite time... glFlush can return
> at any time. It does not wait until the execution of all previously issued GL commands is
> complete."

That is the entire documented contract — nothing official about yielding the CPU, cooperating
with an OS/GPU scheduler, or enabling preemption of in-flight GPU work. The stronger claim
("flush between passes lets other GPU clients interleave") appears only in community forum
threads — **folklore, not spec**. `glFlush` is a submission-timing hint, not a concurrency
primitive; it would not bound UI-thread latency on its own.

**Compute-based progressive/accumulated rendering.**
https://dev.epicgames.com/documentation/unreal-engine/path-tracer-in-unreal-engine
> "When enabled, the renderer progressively accumulates samples from the current view by
> continuously adding samples while the camera is not moving."

Unreal's path tracer runs a bounded sample count per frame, additively blending into a
persistent accumulation buffer, converging over several frames instead of computing full
quality in one dispatch — the same pattern documented informally in WebGL/shader-toy-style
progressive renderers. Buys: bounded, predictable per-frame GPU cost with zero threading/process
changes — each frame does a small fixed amount of work. Costs: visible latency to final quality
(the image is visibly incomplete immediately after any parameter/camera change until enough
frames accumulate), and it only applies to effects that are mathematically accumulable — not a
general answer for a single expensive deterministic shader pass that must complete atomically
(most of ShaderBox's fragment-shader documents).

---

## What the sources settle

- GLFW: window/context creation and all event processing (`glfwPollEvents`/`glfwWaitEvents`/
  `glfwWaitEventsTimeout`) are main-thread-only by GLFW's own explicit statement and per-function
  reference docs; `glfwMakeContextCurrent`, `glfwSwapBuffers`, and `glfwPostEmptyEvent` are
  documented as callable from any thread.
- GLFW's `glfwCreateWindow(..., share=other_window)` shares the GL object namespace
  (textures/buffers/programs/etc.) between the two windows' contexts; GL state (bindings,
  current program) stays per-context.
- GLFW's documented pattern for an offscreen/worker context is a `GLFW_VISIBLE=FALSE` window
  created on the main thread, its context later made current on a worker thread.
- A GL context may be current on only one thread at a time, and a thread may have only one
  context current at a time (GLFW context guide, matches the general GL contract).
- moderngl exposes no fence/sync-object API at all — confirmed absent from both the Python
  layer and the compiled extension's call sites, despite the GL function pointers being loaded.
- moderngl's maintainer (`einarf`, issue #623) states the driver is not multithreaded and
  recommends exactly the render-thread-owns-its-context-for-life pattern 090 is aiming at.
- moderngl 5.12.0 ships no free-threaded (`cp3XXt`) wheels on PyPI — confirmed directly from the
  PyPI file listing, independently, twice.
- OpenGL 4.6 core spec: textures, buffers, programs/shaders, renderbuffers, samplers, and sync
  objects are shared across a context group; VAOs, FBOs, program pipelines, and transform
  feedback objects are per-context "container objects" and are not shared.
- OpenGL 4.6 core spec: changes to a shared object made in one context are not automatically
  visible in another; the documented mechanisms to guarantee visibility are `glFinish` (blocking)
  or `glFenceSync` + `glWaitSync`/`glClientWaitSync` (non-blocking, preferred) — not `glFlush`
  alone.
- `ARB_sync` extension spec: a fence's completion is only visible to a waiter on another
  context/thread once the *producer* context has called `glFlush()` after `glFenceSync` — the
  waiter's own automatic flush (`GL_SYNC_FLUSH_COMMANDS_BIT`) does not flush the producer's
  stream.
- NVIDIA's Linux driver README (580.95.05) documents `__GL_YIELD` and `__GL_SYNC_TO_VBLANK` as
  CPU-side scheduling/vsync knobs; it documents nothing about desktop GPU-side scheduling
  between contexts.
- Dear ImGui's own FAQ states its context is not safe for concurrent multi-thread use and names
  `ImDrawDataSnapshot`/`ImTextureQueue` as the sanctioned staging mechanism when render output
  must cross a thread boundary.
- Blender's GPU module source requires secondary-context creation on the main thread and
  activation on the worker thread — matching GLFW's and moderngl-maintainer's guidance.
- Chromium and Unreal both bridge a UI/logic thread (or process) to a render thread/process via
  an async command queue plus an explicit fence/completion primitive, never shared live state.
- CPython's C-API docs state that code running with the GIL released must not touch Python
  objects or the C-API; `ctypes.CDLL` releases the GIL by default per native call.

## What remains folklore

- Whether the NVIDIA desktop proprietary driver time-slices or truly concurrently executes GPU
  work submitted from two GL contexts on the same GPU (same process, two threads) — no official
  NVIDIA source found; only CUDA-specific (MPS, Kubernetes time-slicing blog) and
  wrong-product-line (DriveOS/Tegra) documentation exists, plus unverified forum claims about a
  GeForce-vs-Quadro concurrency split.
- The precise CPU-yield vs GPU-scheduling boundary implied by community discussion of
  `glFlush` "helping" interleave GPU clients — the Khronos reference page documents no such
  effect; this is forum folklore.
- Whether moderngl's own compiled extension releases the GIL around individual GL calls (not
  found documented either way; would need checking moderngl's Cython/C++ source directly if it
  becomes load-bearing).
- The exact latency/behavior of the GLFW claim "`glfwSwapBuffers` does not pump the event
  queue" — inferred from disjoint documented scopes, not a single explicit sentence.
- Whether Blender's GPU-context-threading design is explained anywhere in prose developer docs —
  only source-level header comments were found and verified; no narrative rationale page exists
  publicly (or wasn't found).

## False trails

- The Khronos wiki (`www.khronos.org/opengl/wiki/*`) — repeatedly targeted as a source for §3
  (OpenGL Context / Sync Object pages) but unreachable from this environment on every attempt
  (TLS handshake failures / 403 / redirect-then-403). Not evidence the wiki is wrong or
  unhelpful — just inaccessible here. The core spec PDF and the `ARB_sync` extension spec (both
  from `registry.khronos.org`, which *was* reachable) fully covered the same ground with more
  precision, since the wiki itself paraphrases these documents.
- KodeLife's own site (`hexler.net/kodelife`) — marketing copy only, no architecture
  documentation; contributes nothing beyond "code is checked/evaluated/updated in the
  background," which does not establish threading model. Bonzomatic's actual source code was a
  far better source for "does a livecoding shader tool run on the UI thread" and is used instead.
- NVIDIA's MPS documentation — initially plausible as a lead for "multi-context GPU sharing,"
  but MPS is explicitly CUDA-process-only and never mentions OpenGL; not applicable to a
  same-process, two-GL-context design.
- NVIDIA's DriveOS/Tegra GPU scheduling docs — the only NVIDIA-authored description of
  runlist/time-slice GPU scheduling found anywhere, but scoped to embedded automotive hardware
  and a different driver codebase from desktop `nvidia.ko`; cannot be cited as desktop-driver
  behavior without a cross-product-line inference the sources themselves don't support.
- Blender's narrative developer docs (`developer.blender.org/docs/features/gpu/` and
  `.../features/core/context/`) — fetched directly on the assumption they'd explain GPU
  context/thread handling in prose; the GPU-module page only states responsibilities with no
  threading discussion, and the "Context" page is about the unrelated logical `bContext`
  (UI/window-manager state), not GL context threading at all.
