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

Research measured on the dev box (RTX 3090 / 24 threads, Linux, Python 3.12.8), scripts
kept out of the repo:

- A neighbour pass over 100k points costs 49 ms in pure numpy and 13.5 ms as an Odin
  `.so` called from Python on one thread, same result checked against brute force. A
  ctypes call costs 0.12 us; numpy arrays cross as pointers, uncopied.
- The same `.so` in a CHILD process with its arrays in shared memory and one
  request/reply per call: 29 us round trip (p99 44 us), 13.8 ms for the same step. A
  deliberate SIGSEGV in the child left the parent running with the shared data intact.
- Raw `ctypes` already works in `script.py` today: the narrowed `__import__` gates only
  `shaderbox.*`. What it lacks is everything this spec adds: a crash kills the app, a rebuilt
  library cannot be reloaded under the same path, and a script has no way to name a file
  next to its own document.

The pre-implementation review changed the transport, the buffer mechanism and the release
rules; see Review history.

## Goal

An author writes `lib = native("fluid")` in `script.py`, allocates buffers through it, and
calls any export of the shape `int32_t fn(sb_call*)` with named arguments. A failure the
library reports, a crash, a hang, a missing export and an ABI mismatch each arrive as an
ordinary script error with the library's own words; the app keeps running. Rebuilding the
library picks up the new code once the file has settled (D7), and the buffers survive the
swap.

## Out of scope

- **An in-process mode.** Open question 2 proposes a debug mode on the host instead.
  **Trigger:** a library whose per-call round trip (about 50 us with a handful of
  arguments, measured) is the frame's bottleneck.
- **A script returning a texture.** Closed by feature 110
  (`ai_docs/features/110_script_textures/01_spec.md`): a script returns a numpy array -- a
  native buffer is one -- for a `sampler2D`, and the pass samples it that frame.
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
- **Completion types refreshing when only the library changes.** The editor's
  return-type probe is cached on the script's stamp (`tabs/code.py::_typed_script_returns`),
  so a probe that ran before the library was built keeps an empty answer until the script is
  edited. Completion is the only consumer. **Trigger:** a report of a missing completion
  that a script save fixes.

## Design decisions

- **D1. A library always runs in a child host process, started as a standalone script.**
  Isolation is the only way a segfault, an abort or an infinite loop in someone else's code
  does not take the app down; Python has no in-process recovery from a native fault. One
  host per handle, so two libraries in one document fail independently. The host is `shaderbox/scripting/native_host.py`, launched with
  `subprocess.Popen([sys.executable, "-I", "-S", <its path>])`. Run by PATH it imports no
  package, so `shaderbox.scripting/__init__` never runs; `-S` skips `site`, so no venv `.pth`
  hook runs in it either (round 2 saw coverage's `.pth` start under `-I` alone and write a
  `.coverage` into the repo). It imports the stdlib only -- no numpy, no moderngl, nothing
  from site-packages -- and gate 14 holds that set. The review measured why
  `multiprocessing` spawn is rejected: spawn re-runs the parent's `__main__`, which in the
  app is `shaderbox/ui.py` (`scripts/run.sh`, `run.bat`, `make run`), so each host imported
  glfw, moderngl and imgui_bundle and took 617 ms against 79 ms from a bare parent. A
  stdlib-only host by path started in 18 ms, the same from a parent with the GL stack
  loaded. Tests never see the spawn cost because
  pytest is `__main__` there.
  **Revisit if** the round trip is a measured bottleneck (see Out of scope).

- **D2. One calling convention for every export: `int32_t fn(sb_call *c)`.** A pointer
  means nothing in another process, so arbitrary C signatures cannot cross the boundary.
  Every argument is NAMED and is a number, an integer, a string or a buffer (see The ABI).
  Return 0 for success. That one shape fits a particle step, a Navier–Stokes grid and a
  JSON-in/JSON-out algorithm alike. Raw `ctypes` with arbitrary signatures stays available in
  a script for an author who accepts losing isolation; nothing here wraps it.

- **D3. Buffers are file-backed memory maps the HANDLE owns.** `lib.buffer(shape, dtype)`
  creates a file (in `/dev/shm` on Linux, the temp directory elsewhere; the name carries the
  app's pid), RESERVES its pages (`os.posix_fallocate` on Linux), maps it with `mmap`, and
  returns `np.frombuffer(the_mmap, ...)`: a plain C-contiguous ndarray, ndim 1..4, every
  extent at least 1, dtype one of `f4 f8 i4 u4 i8 u1`. The host maps the same file by path.
  The reservation matters: a sparse file on a full tmpfs maps fine and then SIGBUS-kills the
  app on the first write to an unbacked page (measured), while `posix_fallocate` refuses up
  front with `ENOSPC`, which becomes a `load`-kind error at `lib.buffer`.
  Two measured hazards decide this shape over `multiprocessing.shared_memory`: an ndarray
  over `SharedMemory.buf` SEGFAULTS the app when written after the segment is closed --
  which `SharedMemory.__del__` does on its own when the owner is dropped -- while an ndarray
  over an `mmap` pins it (`mmap.close` raises `BufferError` while exports exist); and a host
  not started by `multiprocessing` that attaches with `SharedMemory(name)` UNLINKS the segment
  when it exits, through its own resource tracker (3.12 has no `track=False`). A plain file
  has neither: no tracker, the same code on every platform, and the mapping lives as long as
  any array does. The handle keeps every file until it is released, because a reload maps
  them again by path (a deleted file made the next host fail to map, measured). Release (D10)
  stops the host and deletes the files; a mapped file that
  cannot be deleted (Windows) is removed at the next app start by a sweep of files whose
  name carries a pid no longer alive. Only arrays from `lib.buffer` may be passed as buffer
  arguments, and only as the EXACT object `lib.buffer` returned, checked by identity: a slice
  of a buffer is C-contiguous and resolves to the same mmap at a byte offset, so a check on
  the base chain would hand the library the file's start and it would write the wrong rows.
  An ordinary ndarray, a view, and another handle's buffer are refused, each with a message
  naming `lib.buffer`; copying would make an in-place write vanish.
  A buffer has an owner phase like a handle (D10) and is released with it, so
  `lib.buffer(...)` in `__init__` on a module-level handle does not leave a file behind per
  reset. What a buffer can feed downstream is the
  engine's existing rule, unchanged: `@instances` takes `f4`/`i4`/`u4` columns as they are, and
  a uniform array takes a list (`.tolist()`).

- **D4. The ABI is proven at load, entry by entry, against a table the library exports.**
  The header is compiled into the author's library and the host builds the same structs
  through ctypes, so the two layouts are a duplicate. Following
  `graph_canvas/ffi.py::_verify_layout`, the library exports a flat table of named integers
  (see The ABI): the contract version, the size of each struct, every field's offset by
  name, and every enum member by name. The host compares the whole table against its own and
  refuses the library on the first difference, naming the entry. A library without `sb_abi`
  is refused with "not a ShaderBox library: no `sb_abi` export -- add `SB_LIBRARY()`".

- **D5. Errors arrive in the script as ONE exception class, `NativeError`, with a `kind`.**
  `failed` (nonzero return; the message is the library's `sb_fail` text, or "`step` returned
  3 without sb_fail"), `crashed` (the host died; names the export, the exit code or signal,
  and the last lines the library wrote to stderr), `timeout` (deadline passed; host killed),
  `missing` (no such export), `load` (file not found -- listing every path tried -- not
  loadable, ABI refusal, or `sb_load` failed). Raised at `native(...)` for `load` (the host
  starts there, so a missing library is reported where it is named) and at `lib.call(...)`
  for the rest. Uncaught, it is an ordinary `ScriptError` on the user's line through the
  existing path (`engine.py::_tick_script` maps to the deepest `<u:script.py>` frame; one
  raised in `__init__` reads "`__init__` raised: NativeError..." through
  `behavior.py::_instantiate`). Nothing new reaches the frame loop.

- **D5a. A crash is seen in milliseconds, and is never reported as a timeout.** On Linux the
  host calls `prctl(PR_SET_DUMPABLE, 0)` at start. Without it, a SIGSEGV on a box whose
  `core_pattern` pipes to apport took 1080 ms to reach the parent, so a 0.5 s deadline
  reported a crash as `timeout`; with it, 1 ms. On Windows the host calls
  `SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX)` so a crash does not wait on
  an error dialog (unmeasured; from the Win32 documentation).

- **D6. A dead host stays dead until something changes.** After `crashed` or `timeout` every
  further call raises the same error without starting a process, until the library file
  changes (D7), the script is reset (which also revives a module-level handle D10 keeps), or
  the script reloads. Restarting on the next call would
  turn a deterministic crash into a host start every frame.

- **D7. A rebuilt library is picked up on a later call, and the buffers carry over.** Each
  call stats the library file. A file that is missing keeps the current host (a linker that
  unlinks and recreates leaves a moment with no file: GNU ld changed the inode during a
  rebuild in the review's probe). A changed `(mtime, size)` that has stayed unchanged for at
  least 0.5 s of wall time, seen on at least two calls, stops the host and starts a new one.
  Round 2 refuted a pure two-call rule: a real `gcc` link of a 64 MB library grew over about
  80 ms and the rule reloaded mid-write, and a writer pausing between chunks produced a
  SIGBUS in the loading host. The 0.5 s rule narrows the window and does not close it -- a
  writer that pauses longer can still be caught -- so a failed load is `load`/`crashed` like
  any other and is D6-dead until the file changes again, which the finished write does.
  Buffers are untouched because the handle owns them. The library's own globals do not
  survive -- that is the documented rule: state that must outlive a rebuild lives in buffers.
  An optional export `int32_t sb_load(sb_call *c)` runs after every load, with no arguments,
  for an author whose library needs setup.

- **D8. The host loads a COPY of the library.** It copies the file to a unique name, loads
  that, and on Linux unlinks the copy right after loading (the parent deletes it after the
  host stops on Windows). The PARENT chooses the copy's name, carrying its pid, and deletes
  it after the host exits on every platform as well, because a host that crashes inside
  `dlopen` never reaches its own unlink (round 2 found such copies left in `/tmp`); the D3
  sweep removes what a killed app left. Measured on Linux: a library loaded in place and overwritten in
  place (`cp`, same inode) silently switched the LIVE host to the new code, and a larger
  replacement SIGSEGVed it; with copy-load both left the host on the old code. On Windows a
  loaded DLL is locked, so loading in place would make the author's own rebuild fail to
  write the file and a document delete fail while the host lives (from the Win32
  documentation; unmeasured).

- **D9. A name resolves next to the document.** `native("fluid")` looks in the document's
  `scripts/` directory (`paths.py::scripts_dir_for`) for, in order, `fluid.so` and
  `libfluid.so` on Linux, `fluid.dll` on Windows, `libfluid.dylib` and `fluid.dylib` on
  macOS. A name with an extension or a separator is a path, relative to `scripts/` or
  absolute. A miss is a `load` error listing every path tried. The library lives with the
  document as project data, beside `script.py`; ShaderBox never writes it.

- **D10. A handle belongs to the PHASE of the script that created it, and is released with
  it.** `PythonBehavior` sets a context variable around each entry into user code, naming
  the owner: the compiled behavior itself during the module exec, and an instance TOKEN during
  `__init__` and `update`. The token is created before `self._cls()` runs, because the
  instance does not exist while its `__init__` executes (`behavior.py::_instantiate` assigns
  `self._instance = self._cls()`); it is released on reset, on replacement, and when
  `__init__` raises. `native()` registers its handle with that owner and memoizes by
  `(name, options)`, so calling it inside `update` returns the same handle every frame.
  - A module-level handle is released when the compiled behavior is: reload, drop, or any
    other point it stops being used. `reset` keeps it, since `PythonBehavior.reset` re-runs
    `__init__` without re-executing the module and the new instance still holds the module's
    objects.
  - An instance-level handle is released at `reset` as well.
  - `native()` with no owner -- called from a thread the script started, where the context
    variable is not set -- raises a `NativeError` saying to create handles in `__init__` or
    `update` and pass them to the thread.
  - A released handle raises `NativeError(kind="load", "released")` on any call.
  - A handle serializes its calls with a lock, so passing it to a script's own thread is
    safe; one host serves one request at a time.
  - Release is idempotent: `App.release` runs both at shutdown and on project switch.

  Release sites, enumerated from the code rather than from intent: `engine.reload` (behavior
  replaced, `scripts_dir` changed, script vanished via `_drop_script`), `engine.reset`
  (instance phase only), `engine.drop_document`, `project_session._resolve_scripts` (the old
  engine on project load: a new `ScriptEngine.release_all()`), App shutdown, and every user of
  `fresh_behavior_for` in a `try/finally` -- `engine.returned_value_types` (including its three
  early returns), `engine.dry_run` (including its compile-error return), and
  `project_session._make_export_isolation`. The dogfood harness
  (`scripts/dogfood/harness.py`) builds a `ProjectSession` it never shuts down and relies on
  the next item. A `weakref.finalize` per handle is the backstop
  for a path that missed, and is not the mechanism.

- **D11. A dry run, an export and the completion probe get their own hosts and their own
  buffers.** All three construct a fresh behavior, and D10 ties handles to it, so a probe's
  native calls cannot touch the live instance's buffers: the 063 isolation holds by
  construction. The cost is a host start (about 18 ms) per probe. The completion probe runs
  on the UI thread, so its hosts get a short deadline (0.25 s) in place of the handle's
  default; a library whose `__init__`-time call outlasts it just yields no completion types. The completion probe is
  already cached on the script's stamp in `tabs/code.py::_typed_script_returns`, so it runs
  once per script edit; the engine docstring of `returned_value_types` claiming "once per
  shader keystroke" is stale and gets corrected.

- **D12. Deadlines: per call 2 s by default, and the same deadline on host start and
  `sb_load`.** `native("fluid", timeout=0.5)` sets it. The parent reads the host's replies on
  a reader thread per host, so a deadline is a bounded `queue.get` on every platform (`select`
  on a pipe does not exist on Windows). The host's end of the pipe is held only by the host,
  so its death is EOF on the reader at once (measured: 1 ms from kill to detection; a parent
  that kept a copy of the child's end blocked forever). A reply the reader cannot parse kills
  the host and is `crashed` with "protocol corrupted", rather than a timeout. Every wait on
  the host after a kill is bounded. The host exits on EOF of its request pipe, so a parent
  that is SIGKILLed or dies in a driver leaves no host behind. Long by default because a
  slow algorithm is a legitimate use.

- **D12a. The library's stdout and stderr reach the app log.** The host moves its protocol
  pipe off fd 1 before loading the library (`os.dup` the original stdout for the protocol,
  then point fd 1 at stderr), so a library's `printf` cannot corrupt the protocol. On Windows
  it also calls `SetStdHandle(STD_OUTPUT_HANDLE, <stderr>)`, since a library writing through
  `GetStdHandle` (as Odin's `core:os` does) bypasses the C runtime's fd 1 (unverified). The parent
  reads the host's stderr on a thread and logs each line with the library's name; the last
  lines ride along in a `crashed` message.

- **D13. The host never touches GL and imports nothing from ShaderBox.** It is a separate
  process with a stdlib-only import set (D1); gate 14 asserts that set. On Windows a venv's
  `python.exe` may be a launcher that starts the real interpreter as a child, which would
  make `Popen.pid` and `kill` target the launcher; the implementation resolves the real
  executable (`sys._base_executable`), or kills by the pid the host reports. Unverified on
  Windows.

- **D14. The script surface grows by two names: `native` and `NativeError`.** They join
  `behavior.py::_INJECTED_NAMES`, `_build_globals`' injection, `scripting/__init__.py`'s
  exports (the import gate ends in a real import) and `api_doc.API_NAMES` with a
  `api_symbol_doc` entry each (`tests/test_script_api_doc.py` already holds `API_NAMES` equal
  to `_INJECTED_NAMES`). The copilot reads a script using them through that doc. This does not
  reopen 079's "no wrapper types": that ruling is about VALUES a script hands the engine, and
  those stay plain. The handle is a tool a script uses, like a numpy RNG. The
  `conventions.md` 041 entry ("`ScriptBehavior`, `ScriptContext` and `MouseState` alone") is
  updated in the same wave.

- **D15. The contract ships as source.** `shaderbox/resources/native/` holds
  `shaderbox_native.h` (C, header-only), `shaderbox_native/shaderbox_native.odin` (the same,
  as a package an Odin library imports), and one minimal example per language with its one-line build
  command. The Odin example shows `context = runtime.default_context()` in each export,
  without which a `proc "c"` export calling ordinary Odin code does not compile. `build.sh`
  already copies the whole `shaderbox/` tree, so nothing changes there. The help modal names
  the directory.

## The ABI

Normative; the header and the Odin binding are its two transcriptions, and D4's table is how
a third language checks itself.

```c
#define SB_ABI 1

enum { SB_NUMBER = 1, SB_INTEGER = 2, SB_STRING = 3, SB_BUFFER = 4 };            /* sb_arg.kind  */
enum { SB_F32 = 1, SB_F64 = 2, SB_I32 = 3, SB_U32 = 4, SB_I64 = 5, SB_U8 = 6 };  /* sb_arg.dtype */

typedef struct {
    const char *name;     /* UTF-8, NUL-terminated                                     */
    int32_t     kind;
    int32_t     dtype;    /* SB_BUFFER only, else 0                                    */
    int32_t     ndim;     /* SB_BUFFER only: 1..4, else 0                              */
    int32_t     reserved; /* 0                                                         */
    int64_t     shape[4]; /* SB_BUFFER only: element counts, unused trailing dims 0    */
    void       *data;     /* SB_BUFFER: C-contiguous elements; SB_STRING: UTF-8, NUL-terminated */
    double      number;   /* SB_NUMBER; for SB_INTEGER the value widened to double     */
    int64_t     integer;  /* SB_INTEGER, else 0                                        */
} sb_arg;

typedef struct {
    int32_t  abi;         /* SB_ABI                                                    */
    int32_t  nargs;
    sb_arg  *args;
    char    *error;       /* sb_fail writes here, NUL-terminated                       */
    int32_t  error_cap;   /* bytes, including the NUL; a longer message is truncated   */
    int32_t  nresults;    /* written by sb_ret_*                                       */
    sb_arg  *results;     /* filled by sb_ret_*: name, kind and number/integer/data    */
    int32_t  results_cap;
    int32_t  reserved;
} sb_call;
```

- **Results.** `sb_ret_num(c, name, v)`, `sb_ret_int(c, name, v)` and `sb_ret_str(c, name, s)`
  append to `results` (32 slots); `lib.call` returns them as a dict. The host copies names and
  strings after the export returns, so a result string must stay valid until then.

- **Before each call** the host sets `error[0] = 0`. A nonzero return with an empty `error`
  is reported as "`step` returned N without sb_fail"; text written by `sb_fail` before a
  return of 0 is ignored. The host decodes `error` as UTF-8 with replacement, since
  truncation at `error_cap` can split a character.
- **Lifetime.** Every pointer in an `sb_call` -- names, strings, buffer `data` -- is valid
  only for the duration of the call. A buffer's contents persist across calls; its ADDRESS
  does not across a reload, so a library keeps no pointer between calls.
- **Python to kind.** `int` and `bool` and numpy integer scalars → `SB_INTEGER`; `float` and
  numpy floating scalars → `SB_NUMBER` (`np.bool_` counts as an integer); `str` →
  `SB_STRING`, refused if it contains a NUL since the string carries no length; an integer
  outside int64 is refused rather than wrapped; an array from `lib.buffer` →
  `SB_BUFFER`; `None` → the argument is absent. Anything else is refused at `call` naming the
  argument and its type. `call(export, /, **args)`: the export name is positional-only, so
  an argument may itself be called `name`.
- **Helpers in the header** (static inline, so nothing to link): `sb_arg *sb_find(c, name)`,
  `sb_arg *sb_buf(c, name)` (NULL unless the kind is `SB_BUFFER`), `double sb_num(c, name,
  fallback)` (accepts both `SB_NUMBER` and `SB_INTEGER`), `int64_t sb_int(c, name, fallback)`
  (`SB_INTEGER` only), `const char *sb_str(c, name, fallback)`, `int32_t sb_fail(c, fmt, ...)`
  (formats into `error`, returns 1).
- **Required exports**, written by `SB_LIBRARY()` in C and by importing the binding in Odin;
  a third language writes them by hand:
  - `int32_t sb_abi(void)` returns `SB_ABI`.
  - `const char *sb_layout_name(int32_t i)` and `int64_t sb_layout_value(int32_t i)`: entry
    `i` of the table, NULL name past the end. The library's set of names must EQUAL the
    host's -- a missing or an extra name is refused by name, then each value is compared.
    The 30 entries, in any order (the version is `sb_abi()`'s, not an entry):
    `sizeof.sb_arg`, `sizeof.sb_call`;
    `sb_arg.name`, `sb_arg.kind`, `sb_arg.dtype`, `sb_arg.ndim`, `sb_arg.reserved`,
    `sb_arg.shape`, `sb_arg.data`, `sb_arg.number`, `sb_arg.integer` (offsets);
    `sb_call.abi`, `sb_call.nargs`, `sb_call.args`, `sb_call.error`, `sb_call.error_cap`,
    `sb_call.nresults`, `sb_call.results`, `sb_call.results_cap`, `sb_call.reserved`
    (offsets); `SB_NUMBER`, `SB_INTEGER`, `SB_STRING`, `SB_BUFFER`,
    `SB_F32`, `SB_F64`, `SB_I32`, `SB_U32`, `SB_I64`, `SB_U8` (values). The table carries
    offsets, not types, so a field that changes type at the same size and offset passes it;
    `SB_ABI` is the guard for that, and is bumped on any change to the structs.
- **Optional export:** `int32_t sb_load(sb_call *c)`, called after each load with
  `nargs = 0` and `args = NULL`; nonzero is a `load` error carrying its `sb_fail` text.
- **Every other export** with the signature `int32_t name(sb_call *c)` is callable by name.
  The host cannot check a signature, so calling a symbol that is not such a function is the
  author's error and ends as `crashed`.

The script side:

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

- `shaderbox/scripting/native_library.py` — new: `native()`, the handle, buffer files and maps, the
  pipe protocol's client side with its reader threads, `NativeError`, ABI verification, the
  stale-file sweep.
- `shaderbox/scripting/native_host.py` — new, run by path: stdlib only; maps buffer files,
  loads the copy, builds `sb_call`s, dispatches.
- `shaderbox/scripting/behavior.py` — the two names; the owner context variable around the
  three user-code entries; module-phase and instance-phase release.
- `shaderbox/scripting/engine.py` — release at every site D10 lists; `release_all()`; the
  `returned_value_types` docstring; the behavior learns its `scripts_dir`.
- `shaderbox/scripting/__init__.py` — exports the two names.
- `shaderbox/scripting/api_doc.py` — `API_NAMES` and `api_symbol_doc`.
- `shaderbox/project_session.py` — `_resolve_scripts` releases the old engine; export
  isolation in `try/finally`.
- `shaderbox/app.py` — release at shutdown; the stale-file sweep at start.
- `shaderbox/resources/native/` — new: header, Odin binding, two examples.
- `shaderbox/help_content.py` — a short section pointing at the directory.
- `ai_docs/conventions.md` — the 041 entry's list of script names; a new design-decision
  entry for this feature.
- `tests/` — the gates below. The fixture library is C, compiled at test time with the
  system `cc` into a temp dir; with no `cc` the tests report SKIPPED, which is not a pass.

## Verification

Each gate names the break it must catch; each break gets tried once and recorded in the
commit.

1. **Crash isolation.** An export writes a known nonzero pattern into a buffer, then a second
   export dereferences NULL. The test process is alive, the call raised `kind="crashed"`
   naming the export, and the pattern reads back both through the handle's array and from a
   fresh map of the buffer's file (the write came from the library, so this shows the data
   crossed and survived). Break: load the library in-process.
2. **Crash is a crash.** The same NULL dereference under `timeout=0.5` is `crashed`, seen in
   well under the deadline. Break: drop `PR_SET_DUMPABLE` (on a box whose `core_pattern`
   pipes to a handler; the test states which).
3. **Hang.** An export loops forever under `timeout=0.2`; `kind="timeout"` within 0.2 s plus a
   stated slack, and after a bounded `wait` the host pid is gone. Break: remove the kill --
   the bounded wait makes this fail the assertion rather than the suite timeout.
4. **Dead stays dead.** After a crash, ten calls raise without a new host pid; changing the
   file brings one back once it has settled (D7), and so does `reset` for a module-level
   handle. Break: restart on every call; separately, leave reset out of the revive rule.
5. **Reload carries buffers and runs new code.** Fill a buffer with a nonzero pattern, rebuild
   the fixture with a different constant, call twice: the export's output reflects the NEW
   constant and the untouched part of the buffer still holds the pattern. Break: reallocate
   buffers on reload (a zero fill would pass this, hence nonzero).
6. **Copy-load.** With a host live, overwrite the source file IN PLACE with a build returning a
   different constant (same inode, `shutil.copyfile`, not the linker) and call before the
   two-call reload settles: the live host still returns the old constant. Break: load in place
   -- the host returns the new constant or dies.
7. **ABI proof.** Compile the fixture against header copies with, in turn: two same-width
   `sb_arg` fields swapped; `SB_ABI 2`; `SB_F32` renumbered; one table entry removed. Each is
   refused, naming the entry. A library with no `sb_abi`: the "not a ShaderBox library"
   message. Break: verify by struct size only (the swap passes), and separately compare only
   the names the library reports (the removal passes).
8. **Release.** Holding a reference to the OLD behavior (so the `weakref.finalize` backstop
   cannot fire in its place), trigger each site in turn -- reload, reset (instance handle
   gone, module handle kept), drop_document, engine replacement, dry run, export, completion
   probe -- and assert the released hosts' pids are gone and their files deleted. Break: drop
   the release call at each site; each makes exactly its own case red.
9. **Write after release is safe, and pages are reserved.** Release a handle while its array
   is still referenced, write the array: the process is alive. Break: back the arrays by
   `SharedMemory.buf`. Separately, a new buffer's file has its pages allocated
   (`st_blocks * 512 >= size`), which is what makes a full tmpfs fail at `lib.buffer` rather
   than at a later write. Mounting a small tmpfs to show the SIGBUS needs root, so the gate
   checks the allocation itself. Break: drop the reservation -- the file is sparse.
10. **Dry-run isolation.** A dry run calls an export that writes its buffer; the live
    instance's buffer is unchanged while the dry run's own buffer did change. Break: return
    the live instance's handle from the memo.
11. **Errors land as script errors.** An uncaught `failed` produces a `ScriptError` on the line
    of the `lib.call` carrying the library's message; a nonzero return without `sb_fail`
    carries the "returned N without sb_fail" text. Break: ignore the return value.
12. **Missing and load.** Calling an absent export is `missing`, naming it; `native("nope")`
    is `load` listing every path tried in D9's order. Break: reorder the lookup.
13. **Only `lib.buffer`'s own objects pass.** A plain ndarray, a slice of a buffer, and
    another handle's buffer are each refused, naming `lib.buffer`. Break: accept by base
    chain -- the slice passes.
14. **The host's import set.** The host reports `sys.modules`; it contains no `shaderbox`,
    `numpy`, `moderngl`, `glfw` or `imgui_bundle` module. Break: start the host through
    `multiprocessing` from a parent whose `__main__` imports moderngl.
15. **stdout cannot corrupt the protocol.** An export that `printf`s a line that looks like a
    protocol reply still returns its real result, and the line appears in the log. Break:
    leave the protocol on fd 1 -- the call is `crashed` with "protocol corrupted".
16. **Lifecycle extras.** `sb_load` returning nonzero is `load` with its text; an `sb_load`
    that hangs is bounded by the deadline. A missing library file during a rebuild keeps the
    live host. `native()` from a script thread raises the thread message. A released handle
    raises "released". The memo: `native("x")` called in `update` for ten frames keeps one
    host pid (break: no memo). A crashing export's last stderr line appears in the
    `crashed` message. SIGKILL of a parent process leaves its host's pid gone. Each break is
    the removal of the named rule.
17. **The shipped examples build and load.** The C example builds with `cc` and the Odin
    example with `odin` from `shaderbox/resources/native/`, and both load and answer one call
    (SKIPPED without the compiler). Break: rename one table entry in the Odin binding.

No gate inspects the host through `/proc`: with `PR_SET_DUMPABLE=0` its entries there are
unreadable. Checking whether a pid exists still works.

## Open questions for the user

None. Both were settled at plan-lock on the recommended answers: exports return named scalars
(The ABI, Results), and `native("fluid", debug=True)` is the debugging mode -- the host skips
`PR_SET_DUMPABLE`, calls `PR_SET_PTRACER_ANY`, has no deadline, and the handle exposes its pid.
A debugger then attaches while isolation holds.

## Review history

Round 1, two reviewers (design against code and conventions; feasibility by probes), both
PARTIAL. Probes are kept outside the repo. Accepted and applied:

- Transport: `multiprocessing` spawn re-imports the app's `__main__` (617 ms, GL modules
  loaded); replaced by a stdlib-only host run by path (D1, D13, gate 14).
- Buffers: `SharedMemory.buf` arrays segfault after close, and a non-`multiprocessing` host
  unlinks segments on exit; replaced by file-backed `mmap` (D3, gate 9).
- Crash latency under apport misreported crashes as timeouts (D5a, gate 2).
- No normative ABI: added The ABI, with a flat layout table a third language can write by
  hand, and broadened D4 to sizes, every field, and the enums.
- Release: three missing sites (`scripts_dir` change, engine replacement, shutdown), early
  returns in the fresh-behavior users, and `reset` releasing module-level handles the new
  instance still uses (D10). Memoized `native()`; threads named in an error.
- D11a's premise was stale: `tabs/code.py` already caches the completion probe on the script
  stamp. Folded into D11 with the docstring fix; library-only staleness moved to Out of scope.
- D7's reload raced a linker that unlinks and recreates; D8's Linux reason was replaced by the
  measured one (in-place overwrite of a loaded library swaps code or crashes the host).
- D3's claim that a buffer feeds a uniform array "unchanged" was false (needs `.tolist()`),
  and three of its dtypes are not `@instances` dtypes; restated as the engine's existing rule.
- `build.sh` needs no change; `scripting/__init__.py`, `api_doc`, `app.py` and `conventions.md`
  were missing from Files touched.
- Gates: release would have stayed green under its own break (the finalize backstop fires
  synchronously on rebind), copy-load's "path differs" check could not fail on Linux, reload
  passed a zero fixture, and three gates named no break. Each rewritten.

Rejected: a `subprocess` host attaching with `SharedMemory(name)` plus
`resource_tracker.unregister` -- superseded by plain files, which need no tracker at all.

Round 2, the same two roles against the rewrite, both PARTIAL; every round-1 item was found
closed in the text. Accepted and applied:

- D7's two-call settle rule loaded half-written libraries from a real `gcc` link (refuted by
  probe); replaced by a wall-time rule, with the residual window stated rather than claimed
  closed.
- A sparse tmpfs file SIGBUS-kills the app on a full `/dev/shm`; pages are reserved (D3,
  gate 9). Zero extents refused.
- A slice passed a base-chain check and would have pointed the library at the wrong rows;
  identity check (D3, gate 13). Buffers given owner phases so a kept handle does not
  accumulate files.
- The `__init__` owner cannot be the instance, which does not exist yet; a token (D10).
  Per-handle lock; idempotent release; the dogfood harness named.
- Reset now revives a kept handle (D6), closing a conflict between D6 and D10.
- Host copies leaked when a host crashed inside `dlopen`; the parent names and deletes them
  (D8). The handle keeps buffer files until release, since a reload maps them again (D3).
- `-I` still ran site `.pth` hooks; `-S` added (D1). The host exits on EOF and a malformed
  reply is `crashed` (D12); the Windows stdout handle and launcher pid are named as
  unverified (D12a, D13).
- ABI: `error[0]` zeroed per call, UTF-8 decoded with replacement, NUL in strings and
  out-of-range integers refused, `np.bool_` mapped, `sb_load` gets `args = NULL`, the 27
  table entries enumerated with set equality.
- The completion probe runs on the UI thread; its hosts get a 0.25 s deadline (D11).
- Gates: hang bounded, the ABI gate covers enums and missing entries, gates 16-17 added for
  the decisions that had none. Open question 3 removed: D1 decides it.
- The round-trip figure is about 50 us with a handful of arguments, not ~30 us.

Found at a site this feature edits and fixed separately before it: `_resolve_scripts`
rebuilt the engine without `on_key_warning`, so after any project load script key warnings
were dropped.

Post-implementation review, two reviewers on the commit (correctness by probes; spec fidelity,
conventions, and a library written from the header alone), both PARTIAL. Fixed in the
follow-up commit:

- `make check` passed over ruff errors in the new modules: `pre-commit run --all-files` checks
  TRACKED files only, and they were not yet added. The Makefile now passes tracked plus
  untracked-not-ignored files.
- Stopping a host closed the readers' pipes, and a close waits for EOF, which a process the
  library started can withhold indefinitely; `_stop` now closes stdin only.
- A host killed mid-call outlived its parent: the deadline lives in the parent, and EOF is read
  only between requests. The host now watches for its parent (D12).
- The host never unmapped a released buffer, so a module-level handle kept every old buffer's
  pages through resets; release now sends an `unmap` (D3).
- A load failure at `native()` started a host on every call; it is memoized against the file's
  stamp (D6).
- A library's `printf` never reached the log: stdio buffers fully on a pipe and the host dies
  by SIGKILL; the host flushes C stdio after every call (D12a).
- The header's `sb_ret_str` rule said "until your export returns", while the host reads the
  string after; a stack string came back as garbage. The header and the Odin binding now say
  the string must outlive the call.
- `debug=True` escaped the completion probe's cap; a reload built the new script before
  releasing the old, so a library holding something exclusive in `sb_load` failed every other
  edit; a lone surrogate in a string killed the host; stderr was logged per line, which the
  app's sinks throttled into timeouts; `scripts/dogfood/harness.py` never released its fresh
  behaviors.
- The module `shaderbox.scripting.native` was shadowed by the re-exported `native()` function;
  it is `native_library.py`.
- The header now says where a library goes, how dtypes map, the default deadline and where
  output lands -- each a gap a first-time author hit.
- Gates added for export release, `_drop_script`, a moved scripts dir, reset's revive, the log
  line of gate 15, unmapping, the failure memo, a library's child holding stderr, a parent killed
  mid-call, and debug under the probe cap. Each was broken and seen red.

A verification round on the fixes (one reviewer) found every fix closed its own case and found
narrower defects in the new code, fixed in the next commit: C stdout is now unbuffered in the
host, so output written just before a crash reaches the crash message; an unterminated final
stderr line is logged; output with no newline is logged in capped pieces as it streams; names
are checked for NUL and UTF-8 like values; a load failure is retried after RETRY_SECONDS rather
than held until the file changes; a phase's buffers release without waiting for a busy host;
`make check` passes file names NUL-separated. Accepted as named limits rather than fixed: the
parent watchdog cannot fire while `dlopen` holds the GIL (a constructor that hangs outlives a
parent killed during that load; `native_host._watch_parent` says so), and a library linked
against its own C runtime (static musl, or a C++ stream with `sync_with_stdio(false)`) keeps
buffers the host cannot flush. The loop stopped here: this round's findings were edge cases in
the fix code, against the previous round's broken mechanisms.
