# 020 Copilot Agent — grounding research (shared anchor)

> **Status: research phase.** This is the pre-spec dump. The maintainer asked for a deep audit
> BEFORE building the copilot: find gaps + leaking seams, figure out how to wire an agent's tools
> smoothly into the whole app, and produce a refactoring-prep plan. NOT a spec yet — this seeds it.
>
> This file is the factual anchor every brainstorming agent and the eventual spec builds on.
> The numbered `NN_*.md` siblings are the swarm's idea reports; `99_synthesis.md` is the merged plan.

---

## 1. What we're building (the eventual feature, stated once)

A **built-in coding-copilot agent**: a chat widget inside ShaderBox that manipulates the app on the
user's behalf — create shaders/nodes, edit shader source, set uniform values + input shapes, manage
shader-lib files, drive render/export. Its tools wrap the app's mutation verbs. The user types
"make the background pulse red" and the agent edits the GLSL + sets uniforms + the preview updates.

The teed-up `todo.md [DEFERRAL] built-in coding-copilot agent + its tool-layer` is the seed. Feature
017 (structure reorg) deliberately made the structure *expandable* to it but built none of it.

This research is the audit + refactor-prep that should happen FIRST.

---

## 2. The stack + the single hardest constraint

- **moderngl + glfw + imgui-bundle, Python 3.12.8.** Solo project, ships on itch.io as a source
  distribution (user runs `uv sync` + `uv run` on first launch).
- **No LLM/agent/anthropic dependency exists yet.** `pyproject.toml` deps: loguru, pydantic,
  pydantic-settings (+ moderngl/glfw/imgui-bundle/imageio/numpy/Pillow + telegram/google libs).
  The `grep` hits for "llm/agent/chat" in shaderbox are false positives (a var named `chat`, a
  telegram method). **Greenfield: no agent code exists.**

### THE constraint: single-threaded synchronous frame loop

`ui.py::run(app)` is a blocking `while not glfw.window_should_close` loop. Each iteration:
`update_and_draw(app)` → `time.sleep` to hit target FPS. Inside `update_and_draw`: lib-index
rebuild, per-node mtime reload, **GL renders**, `process_hotkeys`, then `imgui.new_frame()` … all
the imgui draw calls … `imgui.render()` + `imgui_renderer.render()` + `swap_buffers`. **All on the
main thread. All synchronous.**

An LLM call is a multi-second streaming network operation. **It cannot run inline in the frame
loop** — that would freeze the UI for seconds per turn. So the agent loop MUST run off-thread (a
worker, like the exporters already do) and communicate back via a queue the frame loop drains.

**The hard rule the worker thread inherits (from `dev_flow.md` Module map / exporters):**
> Worker thread MUST NOT touch moderngl (thread-affinity contract).

This is the crux of the whole feature: **tool handlers run on the agent's worker thread, but most
ShaderBox mutations ultimately touch GL** (compile a shader, set a uniform that gets bound, render a
preview). So tool handlers can't just call `App.<verb>()` directly if that verb does GL work on the
calling thread. The design must thread mutations back onto the main thread (a command queue the
frame loop applies), OR carefully partition which verbs are GL-free.

The exporters solved the same problem (worker thread for network/ffmpeg, GL renders marshalled).
Study `exporters/telegram.py` + `share_state.py` `_render_state` for the existing pattern.

---

## 3. The seams the copilot must attach to (verified from source, 2026-05-29)

### `App` (app.py, 1083 L) — the central state holder + verb surface
The mutation verbs already on `App` (the agent's natural attach points):
- `create_node_from_selected_template()` — **reads `app_state.selected_node_template_id`, NOT an
  arg.** A tool wants `create_node(template_id)`. (todo gap (b))
- `delete_node(node_id)` / `delete_current_node()` — clean, takes an id. ✅
- `select_node(node_id)` / `set_current_node_id(id)` — clean. ✅
- `save()` / `save_ui_node(ui_node, ...)` / `flush_current_editor()` — save path.
- `open_shader_lib_file(path)` / `show_node_editor()` — editor target switching.
- `rebuild_shader_lib_index()` — lib index refresh.
- Shader-lib file CRUD: ~20 `*_shader_lib_*` methods delegating to `ShaderLibFileManager`
  (`shader_lib/file_ops.py`, GL-free, explicit-args — the cleanest agent-ready surface in the repo).
- `get_session(source)` / `get_current_session()` — editor sessions (hold a `TextEditor`, imgui).

### The GAPS (from the 017 audit, re-verified):
- **(a) no `set_uniform_value(node_id, name, value)` verb.** Uniform mutation happens INLINE in
  `widgets/uniform.py::draw_ui_uniform` (lines ~228-230): `ui_node.node.uniform_values[name] =
  new_value` directly inside the imgui draw loop, after a `try_to_release(current_value)`. There is
  no headless setter. A tool needs one. Also note the input-shape (`UIUniform.input_type`) is mutated
  the same inline way (`draw_input_type_selector`).
- **(b) `create_node_from_selected_template` reads grid selection, not a `template_id` arg** (above).
- **(c) the imgui-free reach problem.** A tool must reach mutation verbs WITHOUT the agent layer
  importing imgui. But `app.py` is NOT imgui-free — it imports `imgui`, `text_edit` (TextEditor),
  `imgui_command_palette`, `portable_file_dialogs`, the GlfwRenderer. So "reach the verbs" means
  *calling* `App.<verb>()` (App stays the orchestrator) or routing through the genuinely imgui-free
  modules (`shader_lib/file_ops.py`, a future headless `node_ops` module, `core.py`, `ui_models.py`),
  NOT importing `App` into the tool layer wholesale.

### GL-free vs GL-touching verbs (the partition the worker must respect)
- **GL-free (safe on worker thread):** shader-lib file CRUD (`file_ops.py`), reading
  `ui_models` state, `app_state` reads, building text. Writing a `.glsl` file to disk is GL-free
  (the mtime watcher picks it up on the main thread next frame — this is a KEY insight: the agent can
  edit shader source by writing the file, and the existing hot-reload machinery applies it on the
  main thread with no extra marshalling).
- **GL-touching (MUST be marshalled to main thread):** `node.render()`, `node.compile()` /
  `release_program()`, `set_uniform_value` if it triggers a bind, `create_node` (warm-up render in
  `load_from_dir`), texture/buffer uniform loads, preview canvas resizes, anything reading
  `program[...]` or `get_active_uniforms()` (needs a live program).

### `core.py` (495 L) — `Node` / `Canvas`, all GL.
- `Node.uniform_values: dict[str, Any]` — the live uniform store the agent will write.
- `Node.get_active_uniforms()` — needs a compiled program; the introspection surface.
- `Node.compile_unit.errors: list[ShaderError]` — the agent's feedback signal after editing a shader
  (did it compile? what broke? — `shader_errors.ShaderError` has path/line/message).
- `UniformValue` type alias documents what a uniform can hold (int|float|seq|MediaWithTexture|
  Texture|Buffer).

### `ui_models.py` (396 L) — pydantic state (mostly GL-free).
- `UIUniform` (name/gl_type/dimension/array_length/input_type) + `valid_input_types()` /
  `snap_input_type()` — the agent must respect these when setting input shapes.
- `UINodeState` (ui_name, render_media_details, ui_uniforms dict, sort keys, smoothing).
- `UIAppState` (`extra="forbid"`, `load_and_migrate` with 4 migration gens) — if the agent adds a
  chat-state field here it inherits the migration discipline.

### Hot-reload machinery (ui.py) — the agent's free lunch for shader edits
`_reload_if_changed` + `_maybe_rebuild_lib_index` run every frame on the main thread. If the agent
writes a node's `shader.frag.glsl` (or a lib file) to disk, the watcher reloads it, recompiles, and
re-syncs the open editor session — **all on the main thread, no marshalling needed.** This is the
cleanest possible tool seam for "edit the shader". BUT: the editor session's in-memory text would be
clobbered by a disk write while the user has unsaved edits (the reload only re-syncs if texts
diverge — see `_reload_if_changed` lib branch). Edge cases to design for.

---

## 4. Reference agents studied (inspiration, NOT to copy)

Three of the maintainer's own agents. Each contributes distinct patterns. **All three are
client→LLM-over-network designs; ShaderBox is the first DESKTOP/in-process one — the biggest
divergence (threading, no per-user/role/account, local secrets).**

### cc-server — `src/core/ai/chat/{agent,tools,api}.py` (Python, OpenAI-style, the closest stack)
- **`ToolDefinition`** frozen dataclass: `name`, `description`, `parameters` (JSON Schema dict),
  `required_role`, `handler`. **`ToolRegistry`**: `register`, `get_openai_tools(role)` (role-filtered
  wire format), `execute(name, …)` with a **role re-check (defense in depth)** + a try/except that
  returns a **generic `"Error: …"` string** (never leaks exception text — could carry internal
  URIs/ids into LLM context).
- **Closure-built handlers:** `_make_<tool>_handler(service…) -> ToolHandler`. Services resolved
  once in `build_tool_registry()` and captured. Handler signature `(account_id, ctx, args) -> str`.
- **`ToolContext`** dataclass carries per-turn state: focus `scope`, `created_docs` dedup map,
  `deleted_doc_ids` set → prevents duplicate mutations within one agent run.
- **Tool result is always a string** the LLM reads back. Mutating tools return `"Done: …"` with the
  affected entity NAMED (so a later turn can answer "did you delete X?"). Links embedded as markdown.
- **Agent loop** (`AgentLoop.run`, a generator yielding `StatusEvent`/`TextEvent`): max-iterations +
  max-input-tokens budget; **`_compress_old_tool_results`** rewrites all-but-last tool round to a
  short marker (context doesn't grow linearly); `_executed_actions_note` tells the user what mutating
  work already committed if the run is cut off; status templates per tool; a final **polish pass**
  (fast model rewrites the answer to obey output rules, fail-open).
- **Prompt-injection hardening:** `sanitize_name` collapses control chars + caps length on any
  user-supplied name spliced into the prompt/tool-output (folder/recording names are DATA not layout).
- **`describe_tools(role)`** renders the live registry → prose for the suggestion prompt → ONE source
  of truth for "what the agent can do", no hand-maintained capability list.
- `_MUTATING_TOOLS` frozenset gates the "what I did" note + (implicitly) which tools are destructive.
- Double-escaped-JSON repair (`_unescape_double_escaped_strings`) — provider quirk workaround.

### marginalia — `svelte-app/src/lib/core/{tools,agent,tools-shared}.ts` (TS, browser, in-process)
- **THE key pattern for ShaderBox: the injected-capability interface.** `ToolRegistrationHelpers`
  (tools-shared.ts) is a fat interface of ~30 narrow capability fns (`getBookPageText`, `saveBook`,
  `getCurrentBookId`, …). Tools NEVER touch the viewer/DB/app directly — they reach everything
  through this injected bundle. `tools.ts` builds the bundle once (wiring real impls) and passes it
  to each domain registrar. **This is how an in-process agent decouples tools from app internals —
  and is directly the answer to ShaderBox's "reach App without importing imgui" problem: define a
  narrow `CopilotCapabilities`/`AppHandle` interface of GL-free + marshalled verbs; the tool layer
  imports only THAT, never `App`.**
- **Testability via swappable provider:** `setBookPageProvider(inMemoryImpl)` replaces the real
  viewer-backed provider in tests. ShaderBox should mirror this — tools testable headlessly by
  swapping the capability impl, no glfw/GL needed.
- **Domain-split tool modules:** `registerReadingTools` / `registerNavigationTools` /
  `registerLibraryTools`, each `(registrar, helpers) => void`. Scales without one giant file.
- **Per-tool enable/disable** persisted (localStorage); `getToolDefinitions()` filters disabled.
- **`buildLibraryContext()`** snapshots app state (current book, page, selection, library stats)
  into a context object injected into the system prompt + shown in UI. ShaderBox analog: a
  "current node / its uniforms / compile errors / lib functions available" snapshot.
- Agent loop: SSE streaming parse, `_compressOldToolResults` (same idea as cc-server, pattern-aware:
  keeps headers, drops page bodies), token-budget cutoff, AbortSignal/timeout.

### ovelia — `ovelia_server/copilot/{tools,prompt}.py` + `llm/api.py` (Python, the closest in spirit)
- **An actual `copilot/` package** — the closest analog to what we're building.
- **Pydantic args models → `model_json_schema()` as the schema source.** `_CreateEventArgs(BaseModel,
  extra="forbid")` with `Field(description=…)`; `_spec()` does `args_model.model_json_schema()`. **No
  hand-written JSON Schema** (cc-server hand-writes it; ovelia derives it — cleaner, and validation +
  schema share one definition). Args validated with `model_validate` → `ValidationError` →
  friendly `"error: invalid arguments - <msg>"` back to the LLM.
- **Handler return triple `(ok, message_for_llm, payload_for_client)`** — richer than cc-server's
  bare string: the bool drives flow, the message is the LLM-facing tool result, the payload is a
  structured side-channel for the UI (e.g. render a card). ShaderBox analog: the UI could render a
  "created node X" affordance from the payload.
- **`_run_op` wrapper:** uniform try/except around the service call, logs `copilot_tool_failed` with
  tool_name + code/exc_class, returns `(result|None, "error: …"|None)`. Every handler funnels through
  it → consistent error handling, no leak.
- **id-prefix resolution** (`_resolve_id`): accepts a full UUID or a ≥4-hex-char prefix, disambiguates
  against the candidate set, returns a friendly error on miss/ambiguity. ShaderBox node ids are UUIDs
  — the agent will want short-id addressing too.
- **Per-turn dedup cache** (`create_event_dedup`, 30s TTL keyed by (user,title,start)) — defends
  against an LLM emitting the same create twice in a turn.
- **`extra="forbid"` on every args model** → the LLM gets told off for hallucinated params.
- **System prompt assembled least-volatile → most-volatile** (static prefix → profile → memory →
  events → tasks → current time) for **prompt-prefix cache friendliness** — only the volatile tail
  invalidates per turn. ShaderBox analog: static rules → lib-functions-available → current node +
  uniforms + errors (volatile).
- **"Action requires a tool call" prompt discipline:** never claim a past-tense action ("I created…")
  unless a tool with that effect returned THIS turn. Mutation tools that render a card return EMPTY
  text. Strong, transferable prompt rules.
- **`ILLMService` Protocol** (`stream` / `oneshot_text` / `oneshot_structured`) with typed
  `LLMMessage` / `LLMToolSpec` / `LLMToolCall` / `LLMStreamEvent` union — a clean LLM seam that hides
  the provider. ShaderBox should define its own small version (provider = Anthropic for us — the
  maintainer's house default per global CLAUDE).

### Cross-cutting patterns ALL THREE share (the consensus design)
1. A **registry of tools**, each = (name, description, schema, handler).
2. **Handler = closure** capturing the app/service dependencies; signature takes parsed args, returns
   a string (or triple) for the LLM.
3. **An agent loop**: LLM-call → tool-calls → execute → feed results back → repeat until text;
   max-iterations + token budget; old-tool-result compression.
4. **Tool result is a string the model reads** (mutating tools name the affected entity).
5. **Generic error strings** out of handlers — never leak exception internals to the LLM.
6. **A context snapshot** of current app state injected into the prompt.
7. **Prompt-injection awareness** for user-supplied text spliced into prompts.

### What ShaderBox does DIFFERENTLY (don't blindly port)
- **No accounts/roles/multi-tenant.** cc-server's `required_role` / role hierarchy / account_id
  scoping is irrelevant — single local user. Drop it (maybe keep a tool enable/disable like
  marginalia for the cheatsheet/settings).
- **In-process + GL-threaded**, not client→server. The whole threading/marshalling problem (§2) is
  unique to ShaderBox — none of the three references face it (they're stateless request handlers or a
  browser with async-everywhere). This is where we innovate, not imitate.
- **Local secret** (an Anthropic API key) — lives where? `integrations.json` already holds the
  Telegram token + YouTube creds cleartext under `app_data_dir()`. The key likely joins it (there's
  already a `todo.md` deferral about cleartext secrets there). Or env var. Open question.
- **Tools mutate a live visual app the user is watching**, not a database. Feedback is the rendered
  preview + compile errors, not a row count. The agent's "did it work?" signal is `compile_unit.errors`.
- **The maintainer ships on itch with name-your-price** — cost/billing of the user's own API key is
  the user's concern; no server-side billing like cc-server. But token budget per turn still matters
  for UX (don't hang).

---

## 5. Repo size signals (refactor-prep candidates)

`exporters/telegram.py` 1257 L, `app.py` 1083 L, `ui_primitives.py` 781 L, `exporters/youtube.py`
752 L, `ui.py` 536 L, `theme.py` 507, `core.py` 495, `ui_models.py` 396. Existing `todo.md` splits:
decompose telegram/youtube; split ui_primitives at ~900 L; split ui.py/app.py "when painful". The
copilot will ADD a package (`copilot/` or similar) + likely grow `app.py` (chat state, worker
lifecycle) — so "is app.py at the painful threshold NOW, pre-feature?" is a live refactor-prep
question.

## 6. Conventions that constrain the design (from CLAUDE.md / conventions.md)
- Full type annotations; no `from __future__ import annotations`; imports at module top only.
- No `@staticmethod`/`@classmethod` (except real alt constructors); no `if TYPE_CHECKING` (a cycle
  is a design bug — **this will bite the copilot↔App relationship hard; anticipate the type/orchestration
  split like feature 002**).
- All UI flows through `ui_primitives.py` + `theme.py` — the chat widget is no exception.
- `uv` not pip; `make check` (ruff+pyright, blocking) before done; `make smoke` after UI/lifecycle.
- Library docs/source are source of truth — verify against installed package, never guess.
- The `claude-api` skill exists for Anthropic-SDK work (caching, tool use, streaming) — the eventual
  impl should invoke it.
