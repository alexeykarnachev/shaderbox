# 020 Copilot Agent — module skeleton & seams plan

> **A DESIGN doc, not an implementation plan.** Per the maintainer's reframe (`99_synthesis.md §0`,
> answers 3/8/9): outline the clean encapsulated modules + robust seams FIRST; brainstorm the actual
> capabilities (the tool catalog, the prompt content, the UX flourishes) LATER. This file fixes the
> **module boundaries, the seams between them, and the import directions** — the skeleton the eventual
> spec + implementation hang off. It deliberately does NOT enumerate tools, write prompt text, or pin
> UX details.
>
> **Sources:** `01` (threading/bridge), `02` (tool registry + capability seam), `08` (auto-save hook),
> `09` (LLM layer), `05` (chat UI placement), `99 §0` (maintainer decisions). All structural claims
> re-grounded against source 2026-05-29 (`exporters/` layout, `IntegrationsStore`, `App.__init__`,
> `ui.py` frame loop).
>
> **Scope guard:** this is the skeleton. When it reads "a tool that…", that's an example to show the
> seam's shape, NOT a committed capability. The tool catalog is a later brainstorm.

---

## 0. The shape in one picture

```
                     ┌──────────────────────── MAIN THREAD (GL + imgui) ────────────────────────┐
                     │  app.py (App)                                                             │
                     │   • owns the Copilot handle (like exporter_registry)                      │
                     │   • builds CopilotCapabilities (like _build_command_callbacks)            │
                     │   • builds CopilotBridge                                                   │
                     │                                                                            │
                     │  ui.py::update_and_draw                                                    │
                     │   • app.copilot.bridge.drain()      ← runs queued GL ops (NEW)             │
                     │   • app.copilot.pump_events()       ← drains worker→UI event queue (NEW)   │
                     │   • tabs/copilot.py::draw(app)       ← the 4th tab, reads render-state      │
                     │                                                                            │
                     │  editor auto-flush hook (08) — focus-falling-edge → flush_session(path)    │
                     └───────────────────────────────────┬────────────────────────────────────────┘
                                                          │  (two queues + one bridge)
                     ┌───────────────────────────────────┴──────── WORKER THREAD (no GL, no imgui) ┐
                     │  copilot/  package                                                           │
                     │   session.py   — the worker thread + lifecycle + the two queues + bridge     │
                     │   agent.py     — the loop: llm.stream → tools → repeat; usage rollup         │
                     │   prompt.py    — assemble system + context + history → list[LLMMessage]      │
                     │   tools/       — registry + specs + executors (over CopilotCapabilities)     │
                     │   llm/         — api.py (seam) + openrouter.py (impl)                         │
                     │   bridge.py    — the worker→main GL marshalling primitive                    │
                     │   capabilities.py — the leaf seam App implements (no imgui, no App import)   │
                     │   context.py   — the per-turn app-state snapshot (GL-free reads)             │
                     │   state.py     — chat transcript model + render-state (read by the UI)       │
                     │   config.py    — frozen constants (max iters, budget, …)                     │
                     │   errors.py    — typed copilot errors                                        │
                     └──────────────────────────────────────────────────────────────────────────────┘
```

The copilot is **its own package mirroring `exporters/`** (verified layout: `base/registry/telegram/
youtube/integrations/…`, `__init__.py` empty, callers import submodules directly). `App` owns a handle
and drains per frame — exactly the exporter relationship. **`app.py` gains ~handle + 2 drain calls +
the capabilities builder, NOT the copilot's guts** (the "app.py does NOT need splitting" verdict, `03 F6`).

---

## 1. The package layout

```
shaderbox/copilot/
├── __init__.py          # empty (like exporters/__init__.py); import submodules directly
├── capabilities.py      # LEAF SEAM: the dataclass-of-callables App implements. No imgui, no App,
│                        #   no GL, no moderngl import. The ONLY app surface the tool layer sees.
├── bridge.py            # LEAF: the worker→main GL-marshalling primitive (MainThreadOp + bridge).
│                        #   stdlib (queue, threading) only.
├── errors.py            # LEAF: typed copilot errors (CopilotCancelled, CopilotToolError, …).
├── config.py            # LEAF: frozen CopilotConfig (max_iterations, budget, timeouts). Constants.
├── state.py             # the chat transcript model (messages, streaming buffer, usage rollup,
│                        #   pending-action) — the render-state the UI reads. Plain dataclasses.
├── context.py           # per-turn snapshot builder: CopilotCapabilities (reads) → a context object.
│                        #   GL-free reads only. Imports capabilities + state types.
├── prompt.py            # prompt assembly: (system text, context, history) → list[LLMMessage].
│                        #   Imports llm/api (for LLMMessage) + context types. Knows nothing of tools
│                        #   or the client.
├── llm/
│   ├── __init__.py
│   ├── api.py           # LEAF SEAM: frozen LLM dataclasses + the LLMClient Protocol + typed LLM
│   │                    #   errors. stdlib + pydantic only. Imported by everyone in the LLM layer.
│   └── openrouter.py    # the impl: openai-SDK→OpenRouter. Imports llm/api + the secret store.
├── tools/
│   ├── __init__.py      # build_registry(caps) -> ToolRegistry  (the composition point)
│   ├── registry.py      # ToolDefinition + ToolRegistry + execute(). Imports llm/api (LLMToolSpec)
│   │                    #   + capabilities + errors. Does NOT import the client or the loop.
│   └── <domain>.py …    # tool groups (LATER — capability brainstorm). Each: register(reg, caps).
├── agent.py             # THE LOOP. Orchestrates llm.stream + tools + prompt; owns the usage rollup,
│                        #   the tool-use cycle, arg parse/repair, max-iter + budget guards, the
│                        #   pending-action gate. Imports llm/api, prompt, tools, state, config.
└── session.py           # THE ROOT / lifecycle. Owns the worker Thread, the two queues, the bridge,
                         #   the LLMClient instance, the chat state. start/stop/release; enqueue a
                         #   user turn; pump_events. Imports ~everything in copilot. The composition
                         #   root App talks to.
```

Plus, OUTSIDE the package:
- `shaderbox/tabs/copilot.py` — the 4th-tab draw fn (`draw(app)`), like `tabs/node.py`. Imports `App`
  + `ui_primitives` + reads `app.copilot.state`. (UI layer; the copilot package never imports it.)
- `shaderbox/exporters/integrations.py` — gains a `CopilotIntegration` (the key + model). (§5.)
- `shaderbox/commands.py` — gains `NodeTab.COPILOT` + a focus command. (§6.)
- `shaderbox/app.py` — the handle + capabilities builder + bridge ownership + the 2 drain calls. (§4.)
- `shaderbox/ui.py` — the 2 drain calls + the 4th tab row. (§4, §6.)

---

## 2. The import DAG (strict, acyclic, leaf-first)

ShaderBox bans `if TYPE_CHECKING` ("a circular import is a design bug"). The skeleton is cycle-free
**by construction** — the leaf seams (`capabilities`, `bridge`, `errors`, `config`, `llm/api`) are the
shared vocabulary every higher module depends on, and nothing depends sideways.

```
   LEAVES (no copilot imports):  capabilities   bridge   errors   config   llm/api
                                       │           │        │        │        │
   context  ── reads ──────────────────┘           │        │        │        │
   state    ───────────────────────────────────────┘────────┘        │        │   (state uses errors/config types)
   llm/openrouter ──────────────────────────────────────────────────┘────────┘   (impl ← api + secret store)
   prompt   ── llm/api + context ────────────────────────────────────────────────
   tools/   ── llm/api + capabilities + errors ──────────────────────────────────
   agent.py ── llm/api + prompt + tools + state + config + errors ────────────────
   session.py ── agent + llm/openrouter + bridge + state + capabilities + config ─┘   (ROOT)
                                       │
   app.py  ── copilot.session + copilot.capabilities (+ builds CopilotCapabilities) ──── (App is the
                                                                                          outer root)
```

Rules that keep it clean (each is load-bearing):
1. **`capabilities.py` imports NOTHING from `copilot` and nothing from `app`/imgui/moderngl.** It's a
   leaf dataclass of `Callable` fields + GL-free value types. `App` *implements* it (builds an instance
   in `__init__`), so the dependency is `app → copilot.capabilities`, never the reverse. This is the
   cycle-break — the same trick `commands.py` (leaf, "imports imgui only, never App") and
   `ShaderLibFileManager` (injected callbacks) already use. (`02 §2`, `03 F4/F10`.)
2. **`llm/api.py` imports NOTHING from `copilot`.** Provider-neutral dataclasses + the `LLMClient`
   Protocol. `openrouter.py` is the only thing that imports the `openai` SDK; the SDK types NEVER leak
   into `api.py` (cc-server's one seam mistake — `09 §1`). Everyone else depends on the Protocol, so a
   fake client is injectable for headless tests.
3. **No sideways edges among `prompt` / `tools` / `openrouter` / `context` / `state`.** They share only
   the leaf seams. `prompt` doesn't know about `tools`; `tools` doesn't know about the client; the
   client doesn't know about the loop. `agent.py` is the only module that knows several of them — it's
   the LLM-layer composition root. `session.py` is the package composition root.
4. **The tool layer imports `CopilotCapabilities`, never `App`.** `app.py` imports `copilot` (to own
   the session + build the caps), so the one-directional edge is `app → copilot`. If the tool layer
   imported `App`, `app → copilot → app` would cycle — forbidden.
5. **CHECKABLE RULE — no `CopilotCapabilities` field may be annotated with a type that transitively
   imports `App` / `imgui` / `moderngl`.** Every callable field's signature uses only leaf types
   (primitives, `Path`, and GL-free value objects defined in `capabilities.py` itself — `NodeSummary`,
   `CompileErrorInfo`, …). This is what keeps rule 1 honest: a future field typed
   `Callable[[], moderngl.Texture]` or `Callable[[App], …]` would silently reintroduce the banned
   import (and there's no `if TYPE_CHECKING` escape hatch). State it as an enforced constraint in the
   spec, not just a hope. (Reviewer finding.)

---

## 3. The seams (the contracts between modules)

Five seams carry the whole design. Each is a narrow, typed interface; the modules on either side know
only the seam, not each other's internals.

### Seam A — `CopilotCapabilities` (the app ↔ tool-layer boundary)
A frozen dataclass of **bound callables + GL-free value types**, built by `App` in `__init__` (parallel
to `_build_command_callbacks`, `app.py:302`). The tool layer's executors are closures over this. It is
the *only* thing the copilot package knows about the app's verbs. Two halves (the partition from
`03 §2`, the authoritative GL-free/GL-touching table):
- **GL-free callables** (run inline on the worker): reads (node list, shader source, compile errors,
  lib functions), shader-lib file CRUD (already explicit-arg in `file_ops.py`), writing a `.glsl` file
  to disk (the hot-reload free-lunch, `01 §3`).
- **GL-touching callables** (internally do `bridge.run_on_main(...)`): create/delete node, set uniform
  value/shape, force-recompile-and-read-errors, render/export. The *caller* (a tool executor) doesn't
  know or care which half it's in — it calls a method that returns a result; the marshalling is hidden
  inside the capability's implementation on the `App` side.

> **What goes IN the dataclass is a later decision** (the verb list is the capability brainstorm). The
> skeleton fixes only: it's a frozen dataclass of callables, built by App, imported by `tools/` —
> never `App` itself. (`02 §2`.)

### Seam B — `LLMClient` Protocol (the agent ↔ provider boundary)
`copilot/llm/api.py`. A **synchronous** provider-neutral Protocol (`09 §0.1`): `stream(messages, *,
tools, max_tokens) -> Iterator[LLMStreamEvent]` (the workhorse), `complete_text(...)`,
`complete_structured(prompt, *, schema) -> T` (a thin seam for a future planning/triage or the
action-required message — wired only when a capability needs it). Typed frozen dataclasses cross it
(`LLMMessage`, `LLMToolSpec` with `parameters` — OpenAI-shaped, NOT Anthropic `input_schema`,
`LLMToolCall` with raw-JSON `arguments`, `LLMUsage` with `cost_usd`, the `LLMStreamEvent` union). The
`agent.py` loop depends on the Protocol, so `openrouter.py` is swappable and a fake client is
injectable for tests. **The client does ONE call and returns events; it never loops, never executes
tools, never assembles prompts.** (`09 §0.1`, §3.5.)

### Seam C — `CopilotBridge` (the worker → main-thread GL boundary)
`copilot/bridge.py`. The one genuinely new mechanism (`01 §2`). A worker-thread call
`run_on_main(fn) -> result` enqueues a `MainThreadOp(closure + threading.Event)` and **blocks** on
`done.wait(timeout)`; the main thread drains the queue once per frame (`drain()` in `update_and_draw`),
runs the closure with a live GL context, sets the event. `cancel_all()` (called from `release()` BEFORE
`worker.join()`) releases any blocked worker — the deadlock-breaker. Per-op try/except so a raising GL
op can't crash the frame loop (mirrors the `ui.py` panel try/except). **The bridge knows nothing about
LLMs or tools** — it's a pure worker→main RPC primitive; the GL-touching `CopilotCapabilities` callables
(Seam A, App side) are what *use* it.

### Seam D — the two queues (worker ↔ UI, async, non-blocking)
`copilot/session.py` owns them (mirrors `TelegramExporter`'s two-queue pattern):
- **request queue** (UI → worker): the UI enqueues a user turn; the worker drains it blocking.
- **event queue** (worker → UI): the worker pushes `CopilotEvent`s (token deltas, tool-status,
  tool-result, turn-done, error, **pending-action**); `session.pump_events()` drains it per frame on the
  main thread and mutates `copilot.state` (the render-state struct). The UI draws `state` — it never
  touches the queue or the worker. No tearing (immediate-mode redraw of a growing string). (`01 §4`.)

### Seam E — the chat state (`copilot/state.py`, UI ↔ everything-else boundary)
A plain-dataclass transcript model the UI reads and `pump_events` mutates: ordered messages
(user/assistant/tool-status/error/**pending-action**), the streaming-text accumulator, the
`TurnUsage`/`SessionUsage` cost rollup, the in-flight flag, the cancel flag. The UI
(`tabs/copilot.py`) is **dumb** — it renders whatever `state` describes. The agent/worker never touch
imgui; they only mutate state (via the event queue). This is the seam that keeps the GL/imgui thread
and the worker thread cleanly separated.

> **THREAD-SAFETY PIN (single-writer rule).** `copilot.state` is written ONLY by `pump_events` on the
> main thread (draining `CopilotEvent`s) and read ONLY by the UI on the main thread → single-writer,
> no lock needed. The worker thread **never** writes `state` directly. In particular the
> `TurnUsage`/`SessionUsage` cost rollup: the worker/loop (`agent.py`) accumulates its OWN copy as it
> runs, and emits it as a `CopilotEvent` (on turn/usage events); `pump_events` folds that into the
> `state`-side `SessionUsage` the UI reads. There are conceptually two copies (the loop's working
> rollup, the UI's display rollup) bridged by the event queue — NOT one shared object the worker and
> UI both touch. (`09 §2.3` puts the rollup on the loop side; this pin resolves the apparent ambiguity
> with Seam E by routing it through the queue, not a shared field. Reviewer finding.)

---

## 4. How `App` / `ui.py` integrate (the minimal main-thread surface)

The exporter relationship, mirrored. `App` gains (NOT the copilot's logic — just ownership + wiring).
**Three lifecycle subtleties the real `app.py` sequence forces — pin them in the spec, they are
guaranteed impl-time bugs otherwise (reviewer findings, verified against `app.py`):**

- **`App.__init__` construction ORDER (CRITICAL).** `__init__` calls `self._init(...)` at `app.py:280`,
  and `_init` calls `self.release()` at its very top (`app.py:640`). So `self.copilot` MUST be
  constructed **before** line 280 (alongside `exporter_registry` at ~169) AND `App.release()` must
  guard it — exactly as `release()` already guards `preview_canvas` with `hasattr(self,
  "preview_canvas")` (`app.py:1010`) for this same "_init→release runs before attrs exist" reason. Or
  make `CopilotSession.release()` a no-op when no worker/state exists. A naive `self.copilot.release()`
  in `release()` without this guard `AttributeError`s on the first construction. (7a)
  - `self._copilot_caps = self._build_copilot_capabilities()` — parallel to `_build_command_callbacks`.
  - `self.copilot = CopilotSession(caps=self._copilot_caps, ...)` — owns the worker/queues/bridge/client.
  - The worker is **lazily spawned on the first user turn** (don't start a thread for a user who never
    opens the chat — `TelegramExporter._ensure_worker` precedent, `01 §6`). (This mitigates the *thread*,
    not the *handle/bridge attribute existence* — hence the guard above is still required.)
- **The secret must be read LIVE, not captured (CRITICAL).** `self.integrations_store` is the empty
  default at `__init__` (`app.py:167`) but is **reassigned** by `IntegrationsStore.load()` inside `_init`
  (`app.py:683`), and the exporters are re-wired via `exporter_registry.set_integrations(...)` every
  `_init` (`app.py:692`). If the `OpenRouterLLMClient` captures the line-167 object at construction it
  holds a **stale reference** and the user's saved key (loaded at 683) is invisible. Fix: the client
  reads `self.integrations_store` *live* through a capability/closure (NOT capture the object), OR add a
  `copilot.set_integrations(store)` re-wire beside the `app.py:692` exporter re-wire. Pin one. (7b)
- **`App._build_copilot_capabilities()`** — builds Seam A: the GL-free callables point at small adapter
  methods on `App`; the GL-touching ones are `lambda …: self.copilot.bridge.run_on_main(lambda: <the
  GL call>)`. `App` (which legitimately touches GL) supplies the GL closures; the tool layer never sees
  `App` or `moderngl`. (`01 §2 AppHandle`, `02 §2`.)
- **`App.release()`** (`app.py:998`): `self.copilot.release()` — `cancel_all()` then `join(timeout)`
  (`01 §5`), guarded per 7a. **Sequence within `release()`:** the copilot teardown
  (`cancel_all`→`join`) must complete **before** the node release (`app.py:1004-1008` releases all
  `ui_nodes`) — otherwise a queued GL op could run against half-released nodes. So
  `self.copilot.release()` goes at the TOP of `release()`, before the node/exporter releases. (7c)
  `run()` already calls `App.release()` at shutdown (`ui.py:69`).
- **`App._init` (project switch, `app.py:639`).** `_init`→`release()` already tears the copilot down
  (cancels the in-flight turn). The caps closures read live `app.*`, so the session needs only
  `reset_conversation()`, not full reconstruction (`01 §6`). Re-wire the (reloaded) `integrations_store`
  per 7b in the same `_init` pass.

`ui.py::update_and_draw` gains **two calls** (`01 §2`, §4):
- `app.copilot.bridge.drain()` — EARLY, before the per-node renders, beside `_maybe_rebuild_lib_index`
  (`ui.py:157`), so a freshly recompiled node renders the same frame. Wrapped in try/except (the panel
  try/except shape).
- `app.copilot.pump_events()` — near the draw, beside `share_tab.update(app)` (`ui.py:179`). No GL;
  just drains the event queue into `state`.

And the auto-flush hook (`08`, a **separate near-term change**, not strictly copilot): the
focus-falling-edge detector around `tabs/code.py:226` calling a generalized `flush_session(path)`. It
ships independently and means the copilot inherits a mostly-clean editor world (the dirty-editor
clobber dissolves for the common case; the residual focused-editor race is a small tool-side guard).

---

## 5. The secret store (one field-set on the existing seam)

`IntegrationsStore` (`exporters/integrations.py`) is a flat pydantic model persisted to
`integrations.json` at `app_data_dir()`, with a `_SAVE_LOCK` (`integrations.py:15`) **already
serializing render-thread + worker-thread writes** (exactly the concurrency the copilot worker needs).
Add, parallel to `TelegramIntegration`/`YouTubeIntegration`:

```
class CopilotIntegration(BaseModel):
    openrouter_key: str = ""
    model: str = ""              # OpenRouter "provider/model-id" string; default chosen LATER
    model_config = {"extra": "forbid"}
```

- Cleartext, same posture as the existing two; the existing `[DEFERRAL] integration credentials stored
  cleartext` (`todo.md`) covers the keyring migration of all three at the one seam. (Extend that
  deferral to name the OpenRouter key when this lands.)
- Default dual-stack egress; the IPv4 pin is **opt-in** (a flag), NOT default — do not overfit to the
  maintainer's dead-v6 box (`09 §6.1`). The flag could live here too (`ipv4_only: bool = False`).
- `openrouter.py`'s constructor reads this; nothing else in `copilot/` sees the key.

---

## 6. Commands / nav / UI placement (the seams to the existing app)

> **⚠️ REVERSED (maintainer decision, post-review): NO Copilot tab.** The chat is a **floating
> top-level window** (drawn like the cheatsheet, after the main window closes), NOT a 4th tab in the
> settings panel. Rationale: the chat should occupy only a corner/strip (not stretch a full panel
> column), be movable/resizable, and float on top — which a tab can't do. This supersedes report `05`'s
> "4th tab" recommendation (and the diagram/package-list mentions of `tabs/copilot.py` above). The
> floating-window facts (it's a real interactive `imgui.begin`, NOT the cheatsheet's non-interactive
> draw-list; needs `no_nav_focus` to stay out of imgui's Ctrl+Tab window-switcher; the input uses the
> one-shot `set_keyboard_focus_here`; `set_window_focus(name)` segfaults so focus via
> `set_next_window_focus()`) are the `/imgui-ui` skill §8 rules.

The landed shape (commit on `dev`):

- **`commands.py`**: a `TOGGLE_COPILOT` `CommandId` (chord `Ctrl+J`) — NOT a `FOCUS_TAB_*`/`NodeTab`
  entry. The cheatsheet picks it up automatically (iterates `COMMAND_SPECS`).
- **`widgets/copilot_chat.py`**: a floating `imgui.begin("Copilot", flags=no_nav_focus)` window, drawn
  from `update_and_draw` after the main window closes (beside `cheatsheet.draw`). Reads `app`'s
  open/layout/focus state + `app.copilot.state`. Three layout presets cycled by a top-bar button:
  CORNER (bottom-right box) / BOTTOM_STRIP (full-width) / FREE (user-moved, imgui.ini-persisted). The
  transcript / input / streaming / pending-action UI is the capability wave; this is the window shell.
- **Launcher**: a small `ghost_button("Copilot")` pinned to the editor child's top-right (`ui.py`
  `_draw_copilot_launcher`, `set_next_item_allow_overlap`), shown only while the chat is closed; click
  → `toggle_copilot`.
- **Focus model**: `Ctrl+J` = toggle open + focus (closed→open+focus; open&focused→close;
  open&unfocused→focus); opening focuses the chat input + defocuses the editor caret
  (`editor_defocus_requested`). Esc defocuses the chat back to the editor (stays open) — wired via
  `escape_has_job()` + `_handle_escape` + a `copilot_defocus_requested` one-shot. The chat is NOT a
  region — `Ctrl+Tab` still cycles EDITOR→GRID→PANEL unchanged.
- **Egress**: automatic (default dual-stack; transparent v4 fallback is an impl detail) — NO user
  setting. (The earlier `ipv4_only` field was removed — egress must not bother the user.)

---

## 7. The action-required message type (maintainer answer #9 — seam, not detail)

The maintainer wants a special message the agent can push that **blocks further streaming until the
user responds** (the generalized confirm/safety primitive). Where it lives in the skeleton:
- It's a **`CopilotEvent` variant** (Seam D) the worker pushes and a **`state.py` message entry**
  (Seam E) the UI renders as an interactive gate.
- The **agent loop** (`agent.py`) pauses after pushing it: the loop is waiting on the **request queue**
  for the user's answer (a new request-queue item kind = "action response") before it continues the
  turn. So the gate is: worker pushes pending-action event → loop blocks on request queue → UI renders
  the gate → user clicks → UI enqueues the answer → loop resumes. The two-queue seam already carries
  this; the action-required type is a new variant on each side, no new mechanism.
- **The exact shape (what an action requires, how it renders, the confirm vs choice variants) is a
  later detail.** The skeleton fixes only that it's a queue event + a state entry + a loop pause, not a
  separate channel.

---

## 8. What this skeleton deliberately does NOT decide (the later brainstorms)

Listed so the boundary is explicit (and so the spec doesn't accidentally pull these forward):
- **The tool catalog** — which verbs the copilot has. (`tools/<domain>.py` modules; the
  `CopilotCapabilities` field list.) The capability brainstorm.
- **The prompt content** — the system-prompt text, the GLSL authoring rules, the context-snapshot
  fields. (`prompt.py` / `context.py` internals.)
- **The model + provider tuning** — which cheap model, retry policy, structured-output usage.
- **The UX detail** — MVP-vs-gold rows, chips, code-block rendering, streaming polish. (`05`.)
- **Phasing** — internal build order is in `99 §4` (read-only tools are GL-free + easy, so they come
  first as a *build* convenience), but per maintainer answer #3 there is NO user-facing phase gate /
  "read-only product" — it's one chat from day one.
- **Persistence format** — the transcript survives restart (answer #6); the on-disk shape is a later
  detail (a JSON log under `app_data_dir()`; NOT a `UIAppState` field — no migration tax).

---

## 9. Why this skeleton is right (the one-paragraph defense)

It mirrors the precedent the repo already trusts twice (the exporters: own package, own worker +
queues, `App` holds a handle + drains per frame, secret in `IntegrationsStore`). It breaks the
copilot↔App cycle with the exact leaf-seam trick `commands.py` + `ShaderLibFileManager` already use
(`CopilotCapabilities`). It keeps the LLM provider behind a Protocol so the impl is swappable and the
loop is headlessly testable (a hard requirement — the GLFW app can't be screenshotted). It isolates the
one genuinely new piece (the worker→main GL bridge) in its own leaf module with a single drain point
and a deadlock-safe teardown. And it puts every "what can it actually do / say / look like" decision
behind a clean seam so the capability brainstorm can happen *after* the skeleton compiles — exactly the
architecture-before-details order the maintainer asked for. **The one genuinely new mechanism is the
worker→main GL bridge** (Seam C) — the exporters never needed a synchronous round-trip *back into* the
main thread, so it carries a real deadlock surface that `01 §8c` enumerates and the §4 lifecycle pins
guard; everything *else* is assembly of patterns the repo + the three reference agents already prove,
arranged as a strict acyclic leaf-first DAG. (Not "nothing novel" — one novel piece, isolated and
analyzed; the rest precedented.)
