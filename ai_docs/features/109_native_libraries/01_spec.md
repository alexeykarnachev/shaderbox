# 109 — Native libraries from a script

A document script can load a shared library the author built themselves, in any language
that can export a C function, and call it. ShaderBox provides the contract and the
machinery around it, and does not know what the library computes: particles, a fluid
solver, a slow algorithm that merely happens to be written in Odin are the same case.

The requirement, in the maintainer's words: "if I as a user want something fast, can I
build myself a library and use it at runtime"; "we should provide a generic mechanism";
"a wrapper so the whole app does not go down, errors propagate somehow"; "thin enough to
work cross-platform, and easy to implement in my own library, written in C or Odin or
anything".

Research measured on the dev box (RTX 3090 / 24 threads, Linux), scripts kept out of the
repo:

- A neighbour pass over 100k points costs 49 ms in pure numpy and 13.5 ms as an Odin
  `.so` called from Python on one thread, same result checked against brute force. A
  ctypes call costs 0.12 us; numpy arrays cross as pointers, uncopied.
- The same `.so` run in a CHILD process, with its arrays in `multiprocessing.shared_memory`
  and one request/reply over a pipe per call: 29 us round trip (p99 44 us), 13.8 ms for the
  same 100k step, 77 ms to start the child (spawn, import numpy, dlopen). A deliberate
  SIGSEGV in the child left the parent running, reported exit code -11, and the shared data
  readable.
- Raw `ctypes` already works in `script.py` today: the narrowed `__import__` gates only
  `shaderbox.*`. What it lacks is everything this spec adds: a crash kills the app, a rebuilt
  library cannot be reloaded under the same path, and a script has no way to name a file
  next to its own document.

## Goal

An author writes `lib = native("fluid")` in `script.py`, allocates buffers through it, and
calls any export of the shape `int32_t fn(sb_call*)` with named arguments. A failure the
library reports, a crash, a hang, a missing export and an ABI mismatch each arrive as an
ordinary script error with the library's own words; the app keeps running. Rebuilding the
library picks up the new code on the next call, and the buffers survive the swap.

## Out of scope

- **An in-process mode** (`native("fluid", isolated=False)`). See open question 2.
  **Trigger:** a library whose per-call round trip (~30 us) is measured as the frame's
  bottleneck, or a debugging session that `gdb -p <host pid>` cannot serve.
- **A script returning a texture.** A script cannot drive a `sampler2D` today
  (`engine.py::_binding_reject`), from Python or from a library, so a CPU-side 2D field
  reaches a shader only as a uniform array or as instances. That is an engine-output gap
  independent of this feature. **Trigger:** the first document that needs a CPU-computed
  field drawn as an image.
- **A native `Behavior` replacing `script.py`.** `behavior.py` anticipates "a future C
  backend implements the same protocol over a `.so`". Not needed: the Python glue around a
  native call is thin, and keeping it means every engine output a script can produce stays
  reachable without a second protocol. **Trigger:** the Python side of a native-driven
  script is measured as the bottleneck.
- **Asynchronous calls** (start the step, collect next frame). **Trigger:** a native step
  and the frame's render cannot fit one frame serially.
- **Building from source inside ShaderBox** (a toolchain, a build button). The author builds
  with their own toolchain. **Trigger:** a maintainer request.
- **Shipped kernels.** ShaderBox ships the contract (a C header and an Odin binding), no
  computation.
- **The copilot authoring native code.** It cannot run a compiler. It must READ a script
  that uses `native` correctly, which D14 covers.

## Design decisions

- **D1. A library always runs in a child host process.** Isolation is the only way a
  segfault, an abort or an infinite loop in someone else's code does not take the app down,
  and Python has no in-process recovery from a native fault: catching SIGSEGV leaves
  corrupted state and has no Windows equivalent worth trusting. One host process per loaded
  library handle. The host is a small Python program (`shaderbox/scripting/native_host.py`)
  started with the `spawn` start method on every platform: `fork` from a process holding a
  GL context and a glfw window is undefined, and spawn is the only method Windows has. A
  Python host needs no per-platform build of its own, which is what keeps the mechanism
  cross-platform with nothing shipped but source.
  **Revisit if** host start (77 ms measured) becomes visible on reload, or the round trip
  is a measured bottleneck (see Out of scope).

- **D2. One calling convention for every export: `int32_t fn(sb_call *c)`.** A pointer
  means nothing in another process, so arbitrary C signatures cannot cross the boundary.
  Every argument is NAMED and is one of: a buffer, a number (`double`), an integer
  (`int64_t`), a string (UTF-8). Return 0 for success; anything else is a failure whose
  message the library set with `sb_fail`. That one shape fits a particle step, a
  Navier–Stokes grid and a JSON-in/JSON-out algorithm alike, and is what makes the contract
  the same for every library. Raw `ctypes` with arbitrary signatures stays available in a
  script for an author who accepts losing isolation; nothing forbids it and nothing here
  wraps it.

- **D3. Buffers are allocated through the handle and live in shared memory the HANDLE
  owns.** `lib.buffer(shape, dtype)` returns a plain `numpy.ndarray` (C-contiguous, ndim
  1..4, dtype one of `f4 f8 i4 u4 i8 u1`) backed by a `SharedMemory` segment. Only such
  arrays may be passed as buffer arguments; an ordinary ndarray is refused with a message
  naming `lib.buffer`, because silently copying it would make an in-place write vanish.
  Because the handle owns the memory and the host only maps it, buffers survive a host
  crash and a reload (D7). Because they are plain ndarrays, they go into `@instances` or a
  uniform array unchanged.

- **D4. The ABI is proven at load, field by field, by exports the header generates.** The
  header is compiled into the author's library and the host reads the same structs through
  ctypes, so the two layouts are a duplicate. Following `graph_canvas/ffi.py::_verify_layout`,
  one macro the author writes once, `SB_LIBRARY()`, emits `sb_abi()` (the contract version)
  and a field table (`sb_field_name` / `sb_offsetof` over `sb_buffer` and `sb_arg`). The host
  refuses a library whose version differs or whose any field sits at a different offset,
  naming the field. The Odin binding emits the same exports, so an Odin author writes nothing
  extra. A library without `sb_abi` is refused with "not a ShaderBox library: no `sb_abi`
  export — add `SB_LIBRARY()`".

- **D5. Errors are typed and arrive in the script as ONE exception class,
  `NativeError`, with a `kind`.** The kinds: `failed` (the library returned nonzero; the
  message is its `sb_fail` text), `crashed` (the host died; the message names the export,
  the frame and the exit code or signal), `timeout` (the host exceeded the per-call deadline
  and was killed), `missing` (no such export), `load` (file not found, not loadable, ABI
  refusal). Raised at `lib.call(...)` so an author can catch it; uncaught, it is an ordinary
  runtime `ScriptError` on the line of the call, and the existing freeze-at-last-good path
  handles the rest. Nothing new reaches the frame loop.

- **D6. A dead host stays dead until something changes.** After `crashed` or `timeout` every
  further call raises the same error without starting a process, until the library file's
  mtime changes (D7), the script is reset, or the script reloads. Restarting on the next call
  would turn a deterministic crash into a 77 ms spawn every frame.

- **D7. A rebuilt library is picked up on the next call, and the buffers carry over.** Each
  call stats the library file (one `stat`, about a microsecond). A changed mtime stops the
  host and starts a new one; buffers are untouched because the handle owns them (D3). The
  library's own globals do not survive — that is the documented rule: state that must
  outlive a rebuild lives in buffers. An optional export `int32_t sb_load(sb_call *c)` runs
  after every load, with no arguments, for an author whose library needs setup.

- **D8. The host loads a COPY of the library.** It copies the file to a unique name in a
  temp directory and loads that. On Windows a loaded DLL is locked, so loading in place would
  make the author's own rebuild fail to write the file and would make a document delete or
  move fail while the host lives. On Linux `dlopen` of an already-loaded path returns the old
  image. One rule covers both.

- **D9. A name resolves next to the document.** `native("fluid")` looks in the document's
  `scripts/` directory for, in order, `fluid.so` and `libfluid.so` on Linux, `fluid.dll` on
  Windows, `libfluid.dylib` and `fluid.dylib` on macOS. A name with an extension or a
  separator is a path, relative to `scripts/` or absolute. A miss is a `load` error listing
  every path tried. The library lives with the document as project data, beside
  `script.py`; ShaderBox never writes it.

- **D10. A handle belongs to the script INSTANCE that created it, and is released with it.**
  `PythonBehavior` sets a context variable around every entry into user code (the module
  exec, `__init__`, `update`), and `native()` registers its handle with the behavior found
  there. `PythonBehavior.release()` stops every host and unlinks every segment, and the engine
  calls it wherever an instance stops being used: reload, reset, drop, document close, the
  end of an export, the end of a dry run. A `weakref.finalize` on each handle is the backstop
  for an instance dropped by a path that missed the call, and is not the mechanism. `native()`
  called outside any script instance raises.

- **D11. A dry run and an export get their own hosts and their own buffers.** Both already
  construct a fresh instance (`fresh_behavior_for`, `dry_run`), and D10 ties handles to the
  instance, so a probe's native calls cannot touch the live instance's buffers: the 063
  isolation holds by construction. The cost is a host start per probe.

- **D11a. The completion probe runs once per SCRIPT source, not once per shader keystroke.**
  `engine.py::returned_value_types` builds a fresh instance and runs one frame on every
  shader keystroke, so the editor can type the values a script returns. With `native` in
  `__init__` that is a host start (77 ms) plus a real call per keystroke. The answer depends
  only on the script's source -- a fresh instance's first frame -- so it is cached on
  `DocumentScripts.source` and recomputed when the script changes, and its instance is
  released right after the frame. **Revisit if** a script's first-frame value TYPES turn out
  to depend on something other than its source.

- **D12. The per-call deadline is the handle's, default 2 s.** `native("fluid", timeout=0.5)`
  sets it. A deadline exists so a hung library cannot freeze the UI; it is long by default
  because a slow algorithm is a legitimate use and a frame-rate deadline would kill it.

- **D13. The host never touches GL.** It is a separate process, so the thread/GL affinity
  rule is not engaged; the host imports numpy, ctypes and `native_host.py` only.

- **D14. The script surface grows by two names: `native` and `NativeError`.** Both join
  `_INJECTED_NAMES`, and the generated script API doc documents them, so the copilot reads a
  script using them correctly. This does not reopen 079's "no wrapper types": that ruling is
  about VALUES a script hands the engine, and those stay plain (a buffer is an ndarray). The
  handle is a tool a script uses, like a numpy RNG.

- **D15. The contract ships as source, in the bundle.** `shaderbox/resources/native/`
  holds `shaderbox_native.h` (C, header-only helpers: `sb_buf`, `sb_num`, `sb_int`,
  `sb_str`, `sb_fail`, plus `SB_LIBRARY()` and `SB_EXPORT`) and `shaderbox_native.odin` (the
  same in Odin), plus one minimal example per language that builds with one command. The help
  modal names the directory. `build.sh`'s allowlist gains it.

## The contract, as the author sees it

```c
#include "shaderbox_native.h"
SB_LIBRARY()

SB_EXPORT int32_t step(sb_call *c) {
    sb_buffer *vel = sb_buf(c, "vel");
    if (!vel || vel->ndim != 3 || vel->dtype != SB_F32)
        return sb_fail(c, "vel must be an H x W x 2 f4 buffer");
    double dt = sb_num(c, "dt", 1.0 / 60.0);
    /* ... */
    return 0;
}
```

```python
from shaderbox.scripting import ScriptBehavior, ScriptContext, native

class Behavior(ScriptBehavior):
    def __init__(self) -> None:
        self.lib = native("fluid")
        self.vel = self.lib.buffer((256, 256, 2), "f4")

    def update(self, context: ScriptContext) -> dict:
        self.lib.call("step", vel=self.vel, dt=context.dt, viscosity=0.001)
        return {...}
```

## Files touched

- `shaderbox/scripting/native.py` — new: `native()`, the handle, buffer allocation, the
  pipe protocol's client side, `NativeError`, ABI verification.
- `shaderbox/scripting/native_host.py` — new: the child's entry point; maps segments,
  loads the copy, dispatches calls, never imports GL.
- `shaderbox/scripting/behavior.py` — `_INJECTED_NAMES` gains `native`, `NativeError`; the
  context variable around user-code entry; `PythonBehavior.release()`.
- `shaderbox/scripting/engine.py` — `release()` at reload, reset, drop, and after the
  completion probe; the probe's cache (D11a); the behavior carries its `scripts_dir` so D9
  can resolve.
- `shaderbox/scripting/api_doc.py` — the two names documented.
- `shaderbox/project_session.py` — release at document close, export end, dry-run end.
- `shaderbox/resources/native/` — new: header, Odin binding, two examples.
- `shaderbox/help_content.py` — a short section pointing at the directory.
- `build.sh` — allowlist.
- `tests/` — the gates below. The fixture library is C, compiled at test time with the
  system `cc` into a temp dir; with no `cc` the tests report SKIPPED, which is not a pass.

## Verification

Each gate names the break it must catch; each break gets tried once and recorded in the
commit.

1. **Crash isolation.** A fixture export dereferences NULL. Assert the test process is alive,
   the call raised `NativeError(kind="crashed")` naming the export, and a buffer written
   before the crash still reads back. Break: load the library in-process; the test process
   dies, which is red.
2. **Hang.** An export loops forever under `timeout=0.2`. Assert `kind="timeout"` within
   0.2 s plus a stated slack, and the host pid no longer exists. Break: remove the kill.
3. **Dead stays dead.** After a crash, ten calls raise without a new host pid appearing;
   touching the file's mtime brings one back. Break: restart on every call.
4. **Reload carries buffers and runs new code.** Write a buffer, rebuild the fixture with a
   different constant, call: the export's output reflects the NEW constant (contact) and the
   untouched half of the buffer holds the old values. Break: reallocate buffers on reload.
5. **ABI proof.** Compile the fixture against a header copy with two same-width `sb_buffer`
   fields swapped. Load refuses, naming the field. Break: verify by struct size only — the
   swap passes, which is red.
6. **Copy-load.** The host's loaded path differs from the source path, and the source file
   can be overwritten while the host lives. Break: load in place.
7. **Release.** After reload, reset and document close, every host pid the old instance
   owned is gone and every segment name fails to attach. Break: drop the `release()` call at
   each site in turn.
8. **Dry-run isolation.** A dry run calls an export that writes its buffer; the live
   instance's buffer is unchanged. Contact: the dry run's own buffer did change.
9. **Errors land as script errors.** An uncaught `failed` produces a runtime `ScriptError` on
   the line of the `lib.call`, carrying the library's message.
10. **A plain ndarray is refused**, with the message naming `lib.buffer`.
11. **The completion probe starts one host per script source.** Twenty shader keystrokes
    against an unchanged script start at most one host; editing the script starts one more.
    Contact: the probe's answer includes a key whose value came from the native call. Break:
    remove the cache.

## Open questions for the user

1. **Scalar results.** Should an export be able to return named scalars
   (`sb_ret_num(c, "residual", 1e-4)` → `lib.call(...)` returns `{"residual": 1e-4}`), or is
   in-place writing into buffers enough? Recommendation: yes, numbers, integers and strings
   — cheap to add, and the slow-algorithm case (a status, a count, a string result) wants it.
2. **In-process mode for debugging** (`isolated=False`). Recommendation: no, for now. A
   debugger attaches to the host with `gdb -p`, and the host can print its pid; an in-process
   mode reintroduces exactly the crash this feature exists to remove, through a flag that will
   be left on.
3. **One host per handle, or one per document?** Recommendation: per handle. Two libraries in
   one document are independent, and a crash in one should not take the other down.
