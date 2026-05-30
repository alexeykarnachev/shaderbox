# 020 Copilot — Threading, concurrency & GL-marshalling architecture

> Research report (angle: threading). Sibling of `00_grounding.md`. Not a spec — seeds it.
> Every claim is grounded in source read 2026-05-29; `file:line` cited where load-bearing.

---

## Recommendation (read this first)

**Run the LLM agent loop on a single dedicated worker thread (plain `threading.Thread` +
blocking `queue.Queue`, the YouTube-exporter shape, NOT the telegram asyncio shape). GL work is
marshalled to the main thread via a `queue.Queue` of "main-thread ops" drained once per frame in
`update_and_draw`, each op carrying a `threading.Event` the worker blocks on for the result. The
worker owns the Anthropic streaming client (the SDK's sync streaming API, no asyncio needed).**

Concretely, mirror the exporter contract that already exists and works:

- **Two queues**, exactly like `TelegramExporter` (`telegram.py:188-191`): a *request* queue the UI
  pushes user turns into (worker drains, blocking), and an *event* queue the worker pushes streaming
  tokens / status / tool-cards into (UI drains per frame). **Plus a third, new queue** that doesn't
  exist in the exporters: a *main-thread-op* queue the worker pushes GL ops into and **blocks** on.
- **Drain point** = a new `app.copilot.update()` call added next to `share_tab.update(app)` at
  `ui.py:179`, run every frame inside `update_and_draw` on the main thread. It drains BOTH the
  event queue (cheap, for the chat widget) and the main-thread-op queue (runs the GL op, sets the
  op's `Event`).
- **Most GL-touching tools should NOT use the op queue at all** — they should *write the `.glsl`
  file and let the existing hot-reload machinery (`ui.py:_reload_if_changed` + `_maybe_rebuild_lib_index`)
  recompile on the main thread next frame*. The op queue is the fallback for the handful of mutations
  with no file-write path (set a uniform's *value*, change input-shape, create/delete a node, force a
  render-for-readback). See §3 — the file-write path is the free lunch and should be the default.
- **Lifecycle**: a `Copilot` object owned by `App` (constructed in `App.__init__` like
  `exporter_registry`, `app.py:169-172`), torn down in `App.release()` (`app.py:998`) which `run()`
  already calls at shutdown (`ui.py:69`). Cancellation mirrors `Exporter.release()`'s
  stop-sentinel + `join(timeout=…)` (`telegram.py:827-846`).

**Why this shape and not the alternatives:** the codebase *already* solved "worker thread that can't
touch GL, talking to a single-threaded GL frame loop" twice (telegram, youtube). Reusing that exact
pattern (queues drained in `update_and_draw`, GL-free worker, stop-sentinel teardown) means the
copilot inherits a contract the maintainer already trusts, the smoke-test discipline already covers,
and `dev_flow.md` already documents ("Worker thread MUST NOT touch moderngl — thread-affinity
contract"). The ONE thing the exporters never needed — a *synchronous round-trip* from the worker
INTO the main thread (a tool call must block until its GL op produces a result the LLM reads back) —
is the only genuinely new mechanism, and it's a small, well-contained addition (§2).

---

## 1. Where does the agent loop run? Options + trade-offs

The frame loop is a blocking `while not glfw.window_should_close` in `run()` (`ui.py:52-66`); each
iteration calls `update_and_draw` then `time.sleep` to hit target FPS. An LLM streaming turn is
multi-second. It MUST be off the main thread. Four candidate homes:

### Option A — Plain worker thread, sync Anthropic streaming (RECOMMENDED)
A single `threading.Thread` running a blocking loop: `job = self._turn_queue.get()` → run the agent
loop (LLM call → tool calls → repeat) → push events. The Anthropic Python SDK ships a **synchronous**
streaming context manager (`client.messages.stream(...)` yielding text deltas); no event loop needed.

- **Mirrors** `YouTubeExporter` (`youtube.py:565` `_worker_main`: `while True: job = queue.get(); if
  job == _STOP_SENTINEL: return`). The *simpler* of the two existing workers.
- **Pro:** zero asyncio. Python threads + blocking queues are exactly what the rest of the app uses.
  A tool handler that needs a GL op can *synchronously block* on a `threading.Event` (trivial on a
  plain thread; awkward inside an asyncio task). Streaming is a plain `for delta in stream:` loop
  that pushes each delta to the event queue.
- **Pro:** the agent loop is naturally re-entrant per turn — one job = one user message = one full
  loop; the queue serializes turns for free.
- **Con:** one blocking network call per LLM round; can't interleave two LLM calls. We never want to
  (one chat, one turn at a time). Non-issue.

### Option B — Worker thread + asyncio loop (the telegram shape)
`TelegramExporter` runs `asyncio.new_event_loop()` on its worker (`telegram.py:884-886`) because
`python-telegram-bot` is async-only. The Anthropic SDK has both sync and async clients.

- **Pro:** if we ever want token-budget timeouts via `asyncio.wait_for`, or to cancel mid-stream via
  `task.cancel()` (telegram does this at `telegram.py:833-834`), asyncio gives clean primitives.
- **Con:** the synchronous-round-trip-into-the-main-thread (§2) becomes painful: an `async` tool
  handler that must block on a main-thread GL result has to `await loop.run_in_executor(...)` or
  bridge a `threading.Event` into asyncio — extra machinery for no benefit. **The desktop app's
  concurrency need is "one background turn at a time", which a plain thread serves more simply.**
- **Verdict:** only adopt if the chosen LLM SDK is async-only. Anthropic's isn't. Skip.

### Option C — Separate process (multiprocessing / subprocess)
- **Pro:** crash isolation — an LLM-loop segfault/OOM can't take the GL app down.
- **Con:** every tool result + every GL-op request must cross a process boundary (pickle / pipe). The
  whole point of an *in-process* copilot (grounding §4: "ShaderBox is the first in-process one") is
  that tools reach the live app cheaply. A process throws that away. Massive overkill for a solo
  desktop tool. **Reject.**

### Option D — No thread; cooperative slices on the main thread
Pump the LLM stream a little each frame (read N bytes, yield). Discussed fully in the adversarial
section (§8) — it's the "do we even need a thread" attack. Short answer: the *network read itself*
blocks (the SDK's stream iterator does a blocking socket read); slicing it without a thread means a
non-blocking HTTP client + manual SSE parse, which is *more* code than a thread, not less. Reject for
the streaming case; a degraded non-streaming variant is the honest fallback (§8).

**Decision: Option A.** One plain worker thread, sync Anthropic streaming, blocking queues.

---

## 2. GL marshalling — the only genuinely new mechanism

### The problem, precisely
A tool handler runs on the worker thread. Some tools must do GL work: `create_node` warm-up renders
(`core.py:186` `node.render()` inside `load_from_dir`), forcing a render to read back a result,
introspecting active uniforms (`core.py:215` `get_active_uniforms()` needs a live `program`),
releasing/recompiling a program. **The worker MUST NOT call these** (thread-affinity contract,
`base.py:104-110`, `dev_flow.md:213`). And — the new part the exporters never faced — **the tool call
is synchronous from the LLM loop's point of view**: the handler returns a string the model reads back
THIS turn, so it must wait for the GL op to actually run (a frame later) and produce its result.

### The data structures

```python
# copilot/marshal.py  (GL-free module; imports no imgui, no App)

@dataclass
class MainThreadOp:
    """A unit of GL/main-thread work the worker needs done. The worker fills
    `fn`, pushes it, and blocks on `done`. The main thread runs `fn`, stores
    `result`/`error`, sets `done`."""
    fn: Callable[[], Any]          # closure capturing the AppHandle + args
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


class MainThreadBridge:
    """Worker -> main-thread synchronous round-trip. One per Copilot."""
    def __init__(self) -> None:
        self._ops: queue.Queue[MainThreadOp] = queue.Queue(maxsize=64)
        self._shutdown = threading.Event()

    # ---- called ON THE WORKER THREAD ----
    def run_on_main(self, fn: Callable[[], T], timeout: float = 5.0) -> T:
        if self._shutdown.is_set():
            raise CopilotCancelled("app shutting down")
        op = MainThreadOp(fn=fn)
        self._ops.put(op)                       # hand off to main thread
        if not op.done.wait(timeout):           # BLOCK this worker thread
            raise CopilotToolError("main-thread op timed out (UI busy?)")
        if op.error is not None:
            raise op.error
        return op.result

    # ---- called ON THE MAIN THREAD, once per frame ----
    def drain(self, max_ops: int = 8) -> None:
        for _ in range(max_ops):                # bounded: never starve the frame
            try:
                op = self._ops.get_nowait()
            except queue.Empty:
                return
            try:
                op.result = op.fn()             # the GL op runs HERE, main thread
            except BaseException as e:          # noqa: BLE001 — must not crash frame
                op.error = e
            finally:
                op.done.set()                   # unblock the worker

    # ---- called from Copilot.release(), main thread ----
    def cancel_all(self) -> None:
        self._shutdown.set()
        while True:
            try:
                op = self._ops.get_nowait()
            except queue.Empty:
                break
            op.error = CopilotCancelled("app shutting down")
            op.done.set()                       # release any blocked worker
```

### The drain point (exact location)

`ui.py:update_and_draw`, right beside the existing exporter drain at `ui.py:179`
(`share_tab.update(app)`), and importantly **before** the per-frame node renders so a freshly
created/recompiled node renders the same frame:

```python
def update_and_draw(app: App) -> None:
    _maybe_rebuild_lib_index(app)                 # ui.py:157 — hot-reload, unchanged

    # NEW: run any GL ops the copilot worker is blocked on. Bounded; wrapped so a
    # tool's GL op that raises cannot crash the frame loop (mirrors the panel
    # try/except at ui.py:277-283).
    try:
        app.copilot.bridge.drain()
    except Exception as e:                        # belt-and-suspenders; drain() already isolates per-op
        logger.exception(f"Copilot main-thread drain failed: {e}")

    for name in list(app.ui_nodes.keys()):        # ui.py:161 — mtime reload, unchanged
        ...
    # ... preview render, node renders, share_tab.update(app), copilot.update() ...
```

`app.copilot.update()` (the event-drain for the chat widget, §4) goes *near the imgui draw* — it
only mutates the chat-render-state struct, no GL. Put it adjacent to `share_tab.update(app)` for
symmetry, OR just inside the chat-widget draw. Both are fine; it touches no GL.

### Why `MainThreadOp.fn` is a closure, not a tagged union
The telegram/youtube workers use a tagged `_Job` dataclass (`telegram.py:123-131`) dispatched by a
`kind` string (`_handle_job`, `telegram.py:946`). That's right for *them* — a fixed, small set of
job kinds. The copilot's main-thread ops are open-ended (every GL-touching tool contributes one),
so a closure (`fn: Callable[[], Any]`) is cleaner than growing a giant `kind` enum + dispatch. The
closure captures the `AppHandle` (§ below) + parsed args; the main thread just calls it. This keeps
**the GL call itself textually inside the tool handler's module**, not in a central dispatcher.

### The AppHandle seam (resolves grounding gap (c), the "reach App without importing imgui")
Tool handlers must not import `App` (it pulls in imgui, text_edit, pfd — `app.py:9-13`). Define a
narrow capability interface (marginalia's `ToolRegistrationHelpers` pattern, grounding §4):

```python
# copilot/handle.py  (GL-free; the ONLY app surface the tool layer imports)
@dataclass
class AppHandle:
    bridge: MainThreadBridge
    # GL-FREE, safe to call directly on the worker:
    write_shader_file: Callable[[Path, str], None]       # disk write -> hot-reload picks up
    list_lib_functions: Callable[[], list[str]]          # reads shader_lib_index
    read_node_source: Callable[[str], str]               # reads ShaderSource.text (in-mem)
    snapshot_context: Callable[[], CopilotContext]       # current node, uniforms, errors
    # GL-TOUCHING wrappers — these internally do bridge.run_on_main(...):
    create_node: Callable[[str], str]                    # template_id -> new node_id
    delete_node: Callable[[str], None]
    set_uniform_value: Callable[[str, str, Any], None]   # node_id, name, value
    set_uniform_input_shape: Callable[[str, str, str], None]
    force_render_and_read_errors: Callable[[str], list[ShaderError]]
```

`App` builds this handle once and passes it to the copilot (constructor injection). The GL-touching
callbacks are thin: `lambda nid: self.bridge.run_on_main(lambda: app._set_uniform_value_gl(nid, ...))`.
`App` (which legitimately touches GL) supplies the actual GL closures; the tool layer never sees
`App` or `moderngl`. This is the type/orchestration split grounding §6 warns about — and it falls out
naturally because the handle is GL-free and lives in its own module.

---

## 3. The file-write free-lunch path vs. explicit marshalling

Grounding §3 calls this "the cleanest possible tool seam." It is. The hot-reload machinery
(`ui.py:_reload_if_changed:72-119` + `_maybe_rebuild_lib_index:122-148`) already runs every frame on
the main thread and: detects a changed `.glsl` by mtime, reads it, `release_program(new_text)`,
re-syncs the open editor session (`app.sync_editor_from_disk`), recompiles on next render. **A
worker-thread file write is GL-free and triggers all of that with zero marshalling.**

### When file-write is BETTER than the op queue
- **Editing shader source** (the headline use case: "make the background pulse red"). The worker
  writes the node's `shader.frag.glsl` or a lib file; the main thread reloads + recompiles. No
  `MainThreadOp`, no blocking, no GL on the worker. This should be the **default for all source
  edits**.
- It reuses a *battle-tested* path (the mtime watcher already handles root reload, lib reload, index
  rebuild, editor-session resync, undo-history preservation — `ui.py:108-119`). Re-implementing
  compile-on-edit through the op queue would duplicate all of that.

### When file-write BREAKS / is insufficient
1. **Unsaved editor edits get clobbered.** The root-reload branch (`ui.py:85-98`) unconditionally
   `set_text(new_text)` on the session via `sync_editor_from_disk` (`app.py:839-849`) — it does NOT
   diff first (only the *lib* branch diffs, `ui.py:112`). So if the user has unsaved changes in the
   node's own shader and the agent writes the file, **the user's edits are lost**. The agent must
   either (a) refuse to edit a dirty node (check `app.is_current_editor_dirty()`, `app.py:801` via the
   handle) and tell the LLM "the file has unsaved edits", or (b) flush first. *Open question for the
   maintainer (§9).* Note: the lib-file branch is safe (it diffs), only the node-root branch clobbers.
2. **Multi-frame latency before the agent can read compile errors.** The write lands, but the
   recompile only happens on the *next render* of that node (`core.py:281-282`: compile is lazy,
   inside `render()`), which is the next frame at earliest — and only if that node is the current
   node or `is_render_all_nodes` (`ui.py:186-192`). A non-current node may not re-render for many
   frames. So "write file, then read `compile_unit.errors`" is a **race**: the errors aren't there yet.
3. **Compile-error readback needs a forced render.** The agent's "did it work?" signal is
   `node.compile_unit.errors` (grounding §3, `core.py:51`). After writing, the agent must force a
   compile *on the main thread* and read the errors back — which IS an op-queue round-trip.

### The design: edit-then-readback as ONE op-queue round-trip after the file write
The clean composition is: **write the file (worker, GL-free) → then a single `run_on_main` op that
forces the reload+compile+error-read on the main thread and returns the errors as a value.** This
sidesteps the multi-frame race by doing the recompile synchronously inside the op:

```python
def _edit_shader_handler(handle: AppHandle, args: EditShaderArgs) -> str:
    handle.write_shader_file(args.path, args.new_source)        # GL-free disk write
    # Force the reload+compile+read on the main thread, get errors back as a value:
    errors = handle.force_render_and_read_errors(args.node_id)  # -> bridge.run_on_main(...)
    if errors:
        return "Compile errors:\n" + "\n".join(f"  {e.path}:{e.line}: {e.message}" for e in errors)
    return f"Edited {args.path.name}; compiles clean."
```

where `force_render_and_read_errors` on the main thread does: re-read the file if mtime changed (or
just `release_program(text)` directly), `node.render()` (compiles, `core.py:281-282`), return
`node.compile_unit.errors`. This is deterministic — no frame-count guessing — because the op runs
*inside* `drain()` on the main thread with a live GL context, and `render()` compiles synchronously.

**Net rule for §3:** *write the file for the edit (free lunch, reuses hot-reload + editor resync), but
read the compile result back through ONE op-queue round-trip that forces the compile* — don't poll
`compile_unit.errors` from the worker hoping the watcher caught up.

---

## 4. Streaming tokens worker → chat widget without tearing

Mirror the telegram progress queue exactly (`telegram.py:189-191` event queue, drained in `update()`
at `telegram.py:378-385` with `get_nowait()` until `queue.Empty`). The event types:

```python
@dataclass
class TokenDelta:    text: str                    # one streamed text chunk
@dataclass
class ToolStarted:   tool_name: str; summary: str  # "Editing shader…"
@dataclass
class ToolFinished:  tool_name: str; ok: bool; card: ToolCard | None  # ovelia's payload triple
@dataclass
class TurnFinished:  pass
@dataclass
class TurnError:     message: str

CopilotEvent = TokenDelta | ToolStarted | ToolFinished | TurnFinished | TurnError
```

Worker pushes via `put_nowait` (drop-on-full is wrong for tokens — see below). Main thread drains in
`Copilot.update()`:

```python
def update(self) -> None:                          # main thread, every frame
    while True:
        try:
            ev = self._event_queue.get_nowait()
        except queue.Empty:
            break
        self._apply_event(ev)                      # mutate render-state, no GL

def _apply_event(self, ev: CopilotEvent) -> None:
    if isinstance(ev, TokenDelta):
        self._render_state.streaming_text += ev.text     # accumulate partial msg
    elif isinstance(ev, TurnFinished):
        self._render_state.messages.append(Message(role="assistant",
                                                    text=self._render_state.streaming_text))
        self._render_state.streaming_text = ""
        self._render_state.in_flight = False
    # ToolStarted/ToolFinished -> append/patch a tool-card row; TurnError -> notification + flush
```

**No tearing because:** the chat widget reads `self._render_state` (a plain struct on the main
thread) when it draws — it never touches the queue or the worker. Tokens accumulate into
`streaming_text`; the widget renders `streaming_text` as the in-progress assistant bubble each frame.
A token that arrives mid-frame just shows next frame. This is the *exact* render-state-mutated-from-
drained-events pattern `TelegramExporter._render_state` uses (`telegram.py:166-181`, applied at
`_apply_event:1222-1245`).

**Queue-full policy differs from telegram's.** Telegram's progress queue drops/overwrites on full
(`_push_progress:1206-1214`) because only the *latest* progress fraction matters. **Tokens are
cumulative — dropping one corrupts the message.** So the token queue should be generously sized
(`maxsize` large, e.g. 4096) and `put` should *block* the worker briefly rather than drop (back-
pressure is fine — if the UI can't keep up the worker waits a frame). Use `queue.Queue.put(ev,
timeout=…)` and on `queue.Full` log + coalesce only as a last resort.

---

## 5. Cancellation — user hits Stop / closes app mid-run

Three things can be in flight: (1) a blocking LLM network read on the worker, (2) the worker blocked
in `bridge.run_on_main(...).done.wait()`, (3) ops already queued for the main thread.

### Stop button (turn cancel, app keeps running)
- Set a `threading.Event` `self._cancel` the worker checks between agent-loop iterations and between
  stream deltas (`for delta in stream: if self._cancel.is_set(): break`). The Anthropic sync stream
  context manager supports early `break` (closes the HTTP stream on `__exit__`).
- Any *committed* mutations already applied stay applied (ovelia's `_executed_actions_note`,
  grounding §4 — tell the user/LLM what already happened). Push a `TurnError("Stopped")` event.
- A `run_on_main` op already blocked: `_cancel` doesn't directly unblock `done.wait()`, but the op
  will complete on the next `drain()` (the main thread is still running). The wait has a `timeout`
  (5s) as a backstop. For a Stop *during* an op, simplest correct behavior: let the op finish (it's
  one bounded GL call), then the next loop-iteration cancel check fires.

### App close mid-run (the hard one — main thread is going away)
This is where a deadlock lurks (§8): the worker is blocked in `op.done.wait()`, but the main thread
has *left* `update_and_draw` (the `while` loop in `run()` exited, `ui.py:53`) and will never `drain()`
again → the worker blocks forever → `worker.join()` in `release()` hangs.

**Solution (mirrors `Exporter.release`, `telegram.py:827-846`):** `Copilot.release()` runs on the
main thread at shutdown (`App.release` → `run()` at `ui.py:69`). It must, IN ORDER:

```python
def release(self) -> None:
    self._cancel.set()                       # tell the loop to stop between steps
    self.bridge.cancel_all()                 # FIRST: release any worker blocked in run_on_main
                                             #   (sets each pending op.error + op.done.set())
                                             #   and marks bridge shutdown so future run_on_main raises
    self._turn_queue.put(_STOP_SENTINEL)     # unblock the worker if it's idle on turn_queue.get()
    if self._worker is not None:
        self._worker.join(timeout=_DRAIN_TIMEOUT_SEC)   # 5.0s, like telegram.py:836
        if self._worker.is_alive():
            logger.warning("Copilot worker did not exit in time; abandoning (network read may leak)")
```

`bridge.cancel_all()` BEFORE `join()` is the deadlock-breaker: it sets every blocked op's `done`
event so the worker's `run_on_main` returns (raising `CopilotCancelled`), the worker unwinds the
agent loop, hits the `_cancel`/stop-sentinel, and exits — then `join()` succeeds. The blocking LLM
network read is the residual risk (a `join` may time out mid-read); the `daemon=False` +
warn-and-abandon is exactly telegram's accepted compromise (`telegram.py:837-842`).

---

## 6. Lifecycle & ownership

| Concern | Exporter precedent | Copilot |
|---|---|---|
| Construction | `exporter_registry` built in `App.__init__` (`app.py:169-172`) | `self.copilot = Copilot(handle=self._build_copilot_handle())` in `App.__init__` |
| Worker spawn | lazy `_ensure_worker()` on first job (`telegram.py:859-868`) | lazy on first user turn — don't spawn a thread for a user who never opens the chat |
| Per-frame pump | `share_tab.update(app)` → `exporter.update()` at `ui.py:179` | `app.copilot.bridge.drain()` (early, pre-render) + `app.copilot.update()` (near draw) |
| Teardown | `exporter_registry.release()` in `App.release` (`app.py:999`) | `self.copilot.release()` in `App.release`, called by `run()` at shutdown (`ui.py:69`) |
| Save | `App.save()` persists exporter settings (`app.py:985-996`) | persist chat history + API key here if desired (see §9) |
| Project switch | `_init` calls `self.release()` first (`app.py:640`) then rebuilds | `_init`→`release()` already tears the copilot down; reconstruct or `reset()` per project |

**Note on `_init`/`release` coupling:** `App._init` calls `self.release()` at its top (`app.py:640`)
on every project open. `App.release` currently releases exporters, share state, nodes, preview canvas.
Adding `self.copilot.release()` there means **opening a project cancels an in-flight copilot turn** —
correct (the nodes it was editing are about to be swapped out). But the copilot must then be
*recreated* (or `reset()`-ed) at the end of `_init`, since the handle's closures capture project-
scoped state. Cleanest: `Copilot` holds no project state directly — the `AppHandle` closures always
read live `app.*`, so a project switch only needs `copilot.reset_conversation()` (clear history +
spawn a fresh worker on next turn), not full reconstruction. *Decide at spec time.*

---

## 7. Failure isolation — a raising tool must not crash the frame

The frame loop already wraps the two big draw regions in try/except + pushes a notification
(`ui.py:277-283` app panel, `ui.py:452-458` node settings; share-tab update too at `ui.py:178-181`).
Three isolation layers, all mirroring existing code:

1. **Per-op isolation in `drain()`** — `op.fn()` is wrapped in `try/except BaseException` (the
   `MainThreadBridge.drain` code in §2). A tool's GL op that raises sets `op.error` and `op.done` —
   the *worker* receives the exception (re-raised in `run_on_main`), the *frame loop continues*. This
   is the critical one: it guarantees a buggy tool can't take down the GL loop.
2. **Handler-level isolation on the worker** — every handler funnels through a `_run_op` wrapper
   (ovelia's pattern, grounding §4): `try: ... except Exception as e: logger.error(...); return
   "error: <generic>"`. The LLM gets a generic string (never leaks internals — grounding §4
   cc-server rule), the agent loop survives, the turn continues or ends gracefully.
3. **Turn-level isolation** — the worker's top-level loop wraps a whole turn in try/except (like
   `_handle_job` at `telegram.py:946-994`): on an unexpected exception, push `TurnError(...)` →
   the main thread surfaces it as a notification (`app.notifications.push(..., COLOR.STATE_ERROR[:3])`,
   the exact call shape at `ui.py:281`) and ends the turn. The worker thread *never dies* on a turn
   error; it loops back to `turn_queue.get()`.

---

## 8. Adversarial section (attack the design)

### 8a. "We don't need a worker thread at all" — the strongest case
**The argument:** It's a solo, single-user, local desktop app. The user typed a request and *expects*
to wait. A 3-second freeze with a spinner is not great but not fatal — the app already does blocking
work on the main thread routinely: `pfd_block` (`app.py:1019`) blocks on a native file dialog;
project load does synchronous node warm-up renders. A *non-streaming, single-shot* agent — call the
LLM with `stream=False`, block the frame loop, get the full response + tool calls, run tools inline
(GL on the main thread, *no marshalling at all*), draw the result — is dramatically simpler: no
queues, no bridge, no worker, no cancellation-deadlock, no thread-affinity contract to honor. The
whole §2 marshalling problem **evaporates** because there's only one thread.

**Honest concession:** for the *first* prototype this is genuinely attractive and I'd build it first.
The marshalling architecture is real complexity and "premature" is a fair charge if streaming/UX
isn't proven necessary. The boring version (8b) is close to this.

**Rebuttal (why we still land on the worker):**
- A multi-tool agentic turn isn't 3 seconds — it's *several* LLM round-trips (edit → read errors →
  fix → set uniform → …), each multi-second. A frozen, unresponsive window for 15-30s with no
  rendered preview updating (the shader can't animate — the frame loop is blocked) is a bad enough
  experience that the maintainer will want streaming + a live preview *immediately*, i.e. the worker
  comes back within the first feedback round. Building the blocking version first risks throwing it
  away.
- The app's whole identity is a *live animated preview*. Freezing it for the duration of an agent
  turn is uniquely bad here vs. a text-only tool — you can't watch the shader you asked to change.
- Streaming token output is table-stakes UX for a chat and is the maintainer's house style (the three
  reference agents all stream). A blocking call can't stream into the widget.
- **Verdict:** the worker is *not* premature for the real feature, but a blocking single-shot is the
  right *first commit* to de-risk the tool layer (§8b). Build tools + the AppHandle against the
  blocking path, then lift the loop onto the worker once tools work. The AppHandle seam (§2) makes
  this lift cheap — the GL-touching callbacks just gain a `bridge.run_on_main` wrapper.

### 8b. The most BORING design that still works
- One worker thread, `threading.Thread`, `daemon=False`. Sync Anthropic client.
- **No `MainThreadBridge` at all.** Every GL-touching tool works by *writing a file* (§3) and the
  existing hot-reload does the GL. The only readback the agent gets is: after writing, push a
  "please recompile node X and tell me the errors" *event*; the main thread, on its next `update()`,
  does the recompile and pushes the errors back as another event the worker… can't synchronously
  read. So the boring version is **non-synchronous tools**: a tool that edits a shader returns
  immediately ("edit submitted") and the *next user turn* (or an injected follow-up) carries the
  compile result in the context snapshot. Crude, but: zero bridge, zero blocking-into-main-thread,
  zero deadlock surface. Uniform-value changes: also file-based (write them into the node's saved
  uniform values + trigger a reload) OR deferred to a tiny op list with no return value.
- This is genuinely viable and is the fallback if `MainThreadBridge` proves fiddly. Cost: the agent
  can't *see* a compile error within the same turn to self-correct — a real capability loss.

### 8c. Deadlock enumeration
1. **App-close while worker blocked in `run_on_main`** (the main one) — main thread left the frame
   loop, never `drain()`s again, worker's `op.done.wait()` blocks forever, `join()` hangs.
   *Mitigated:* `bridge.cancel_all()` BEFORE `join()` (§5) sets all pending `done` events; plus the
   `timeout` on `op.done.wait()` is a backstop.
2. **Op queue full + worker blocked pushing it, main thread blocked elsewhere** — if `self._ops` is
   bounded (maxsize=64) and the main thread is itself stuck (e.g. a `pfd_block` native dialog,
   `app.py:1019`), the worker's `self._ops.put(op)` blocks. *Mitigated:* main-thread blocking ops
   (file dialogs) are rare and short; size the op queue so a single turn never fills it; `put` with a
   timeout that raises a tool error rather than blocking forever.
3. **Worker waits on main, main waits on worker** — could only happen if a *main-thread* code path
   ever blocks waiting on the copilot worker. It must NOT: the main thread only ever `drain()`s
   (non-blocking `get_nowait`) and reads render-state. *Rule:* the main thread NEVER calls
   `turn_queue.get()` or blocks on a copilot Event. One-directional blocking only (worker→main).
4. **Two ops, ordering** — worker does `run_on_main(A)` then `run_on_main(B)`; both serialize through
   the single op queue + single worker, so no cross-lock. Safe (one worker = no concurrent ops).
5. **`drain()` re-entrancy** — a GL op `fn` must not itself call `bridge.run_on_main` (it's already on
   the main thread → would enqueue an op that only the *same* drain loop services, but `fn` is
   blocked inside `drain`). *Rule:* main-thread op closures call `App`/GL directly, never go back
   through the bridge. Easy to honor (they're authored as direct GL calls).

---

## 9. Open questions for the maintainer

1. **Dirty-editor clobber on shader edit (§3, item 1).** When the agent edits a node's
   `shader.frag.glsl` while the user has *unsaved* edits in that editor session,
   `sync_editor_from_disk` (`app.py:839-849`) overwrites the user's buffer with no diff (unlike the
   lib branch which diffs, `ui.py:112`). Preferred policy? (a) agent refuses to edit a dirty node and
   says so to the LLM; (b) agent auto-flushes the user's edits first; (c) agent edits the *session
   buffer* (in-memory) instead of disk, so it composes with the user's edits and the user still
   Ctrl+S's. (c) is nicest but means the agent's edits don't recompile until save — reintroduces the
   readback race. Leaning (a) for v1.

2. **Where does the Anthropic API key live, and is it persisted cleartext?** `IntegrationsStore`
   (`integrations.py`, pydantic → `integrations.json` under `app_data_dir()`, cleartext like the TG
   bot token) is the natural home — add an `anthropic_api_key: str = ""` field. There's already a
   `todo.md` deferral about cleartext secrets there (grounding §4). Or env var `ANTHROPIC_API_KEY`
   (zero persistence, but the user re-exports every shell). Or both (env wins, store as fallback).
   Decision affects the settings UI + the worker's client construction.

3. **Blocking-first vs worker-first build order (§8a).** Do we build the blocking single-shot agent
   first (de-risk the tool layer + AppHandle, throw-away the loop), or go straight to the worker +
   bridge? The AppHandle seam makes the blocking→worker lift cheap, so blocking-first is low-regret —
   but it's an extra commit. Maintainer's call on appetite.

4. **Does `set_uniform_value` actually need the bridge, or is it file/dict-only?** The mutation is a
   plain dict write (`widgets/uniform.py:230` `node.uniform_values[name] = new_value`) — GL-free *in
   itself*; the bind happens on the next main-thread `render()` (`core.py:340-342`). So a uniform-
   value tool could write the dict from the worker **without the bridge** and let the next frame's
   render bind it — IF dict writes from another thread are acceptable (they're racy with the render
   loop reading the same dict, `core.py:289-338`). Safer to route through the bridge (one tiny op).
   Confirm: is a cross-thread `uniform_values` dict write tolerable, or always marshal? (I recommend
   always marshal — the render loop iterates that dict; a concurrent write is a data race.)

5. **Per-frame `drain()` budget.** I propose `max_ops=8` per frame to bound frame-time impact. Is a
   single agent turn ever going to need >8 GL ops *in one frame*? Unlikely (ops are serialized by the
   single worker, one in flight at a time), so even `max_ops=1` would work and is strictly safest for
   frame pacing. Confirm the cap.
