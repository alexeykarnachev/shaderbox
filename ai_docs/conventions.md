# Conventions

NOT auto-loaded — re-read end-to-end from disk before: spec drafting, spec validation,
implementation, code review, the sanitization sweep. (`CLAUDE.md` inlines only the handful of
code rules that fire on every edit; this file is the full home.)

Everything here is a **generic, future-constraining rule** — if a bullet doesn't constrain code you
haven't written yet, it's noise. Three sections, three buckets; put each rule in exactly one:

- **`## Code rules`** — how source + editable files (Python, md/yaml/json/toml) must be *written*:
  formatting, typing, imports, comments, tooling. Mechanical, language-level.
- **`## Design decisions`** — how the *architecture* is shaped ("we decided X; revisit if Y"): module
  boundaries, ownership, the patterns that span files. Each carries a revisit trigger.
- **`## Known quirks`** — library / SDK footguns and their workaround: where an upstream API bites and
  what to do instead.

NOT here: per-feature mechanics, instance lists, and the story of how a decision was reached — those
belong in the feature spec (`ai_docs/features/NNN_*.md`). This file is not a changelog.

## Code rules

- Full type annotations on all params and variables.
- Imports at module top only — never inside function bodies.
- **Default to NO comment.** A comment restating what the code plainly says (`# re-focus the input`
  over `focus_pending = True`, `# send the message` over `send(...)`) is noise — delete it. The bar:
  would a competent reader be confused WITHOUT it? Only then does it earn a line. The bias to watch:
  freshly-discussed code attracts a comment narrating the conversation we just had — that's exactly
  the comment to NOT write. The rationale lives in the commit message and the spec, not the source.
- **When a comment IS warranted, it states what's non-obvious about the code as it is NOW — never
  narrates development history.** Banned: the bug-we-hit story, the why-we-changed-it backstory, the
  "see <doc> for the full saga", paragraph-length rationale. One terse line naming the non-obvious
  fact (a GL invariant, an upstream-bug workaround, an ordering constraint); if the rationale already
  lives in its canonical home (`## Known quirks` / `todo.md` / a skill), the comment shrinks to a
  ≤1-line pointer (`/imgui-ui §8`) or disappears. The section-banner `# ----` separators are fine;
  the multi-line "here's what happened during development" blocks are the target. (Enforced in
  review + `/sanitize`.)
- **Stale counts rot faster than stale prose** — they look authoritative. When you change a number
  the code reflects (uniform-type count, line counts, "N tabs"), update every doc that quotes it in
  the same commit, or drop the count from the doc.
- **No raw line numbers OR file-length counts in docs** (`todo.md`, specs, conventions, comments).
  Both rot on every edit and force a pointless re-sync: a `file.py:NN` citation is wrong the moment
  code shifts, and a "`foo.py` (800 L)" length is wrong the moment anyone touches the file. Cite the
  **symbol** instead of a line (`App.save`, `widgets/uniform.py::draw_ui_uniform` — greppable, survives
  edits); for size triggers, describe the condition qualitatively ("when it grows a clearly separable
  cluster" / "when editing it feels painful"), not a line count. If a number truly must appear, it's a
  smell — prefer the symbol/condition. (Generic restatement: `dev_flow.md ## Cite by section name`.)
- Don't sidestep a convention with `# noqa` / `# pyright: ignore` / `# type: ignore` / inline import /
  circular-import hack — a collision means the design is wrong. The sanctioned type-suppression
  allowlist (upstream library-stub gaps only) is in `## Known quirks`.
- Never use `if TYPE_CHECKING:` to work-around circular imports. Circular imports is a sign of a bad design.
- **No `@staticmethod`.** A method that doesn't use `self` isn't a method — make it a module-level
  free function. A stateless helper used by one class lives as a private `_name()` function in that
  module; an imgui+theme draw helper reused across modules goes in `ui_primitives.py`, a non-UI
  helper in `util.py` (or the relevant leaf). Same bar for `@classmethod` unless it's a genuine
  alternate constructor (`cls(...)`).
- **UI authoring rules live in the `/imgui-ui` skill, not here.** Button tiers, jitter-free
  overlays, "don't repeat a widget", the SetCursorPos assert, modal chrome, context menus,
  imgui-bundle quirks — all in `.claude/skills/imgui-ui/SKILL.md`. Read it at the start of any
  UI work. This file holds only project-architecture rules (module shapes, ownership, threading).
- Type checker: **pyright** (not mypy), basic mode, via `make check` — blocks on failure.
  Repo is at 0 errors; keep it that way.
- `uv` for everything (`uv run`, `uv add`, `uv add --group dev`) — never bare `python` / `pip`.
- Never use "from __future__ import annotations" - this is a noise.

## Design decisions (we decided X; revisit if Y)

- **Three-layer UI architecture.** `app.py` = UI/glfw/imgui owner + lifecycle wrapper (windowing, GL
  context, editor sessions, popups, nav, the exporter panels) — no imgui drawing in the orchestrator
  sense, but it owns all imgui-bound state. `ui.py` = thin orchestrator owning the frame loop (`run` +
  `update_and_draw`). `widgets`/`popups`/`tabs` = pure draw functions taking `app: App`. (The split is
  forced by the no-`TYPE_CHECKING` rule: a draw fn annotating `app: App` while `App` imports it would
  cycle — so `App` lives in its own module.) Revisit if a 4th UI sub-package is needed. Tracked:
  `todo.md [DEFERRAL] split ui.py`.
- **`ProjectSession` is the headless project + copilot core; `App` owns one and forwards to it.** The
  project lifecycle (paths, nodes, app_state, lib index + cross-project stores, integrations) and the
  whole copilot cluster (`CopilotSession`/`CopilotBackend`/`RevertExecutor` + the capability wiring)
  live in `project_session.py`, which imports no imgui/glfw and creates no glfw window / imgui context
  — so a headless harness (feature 026) constructs it on a standalone EGL context without `App`. `App`
  holds one `self.session` and forwards project state/ops via explicit `@property` accessors (NOT
  `__getattr__` — pyright must see the surface). **UI side effects of a core mutation ride injected
  `on_*` callbacks the core invokes** (`on_current_node_changed` / `on_node_source_synced` /
  `on_node_deleted` — the `ShaderLibFileManager` idiom); a return-value seam is WRONG here because the
  reactions fire mid-copilot-turn on the worker-drain thread with `App` off the call stack. The toast on
  a node save lives in `App.save_ui_node` (the user-path forwarder), NOT the core — so the copilot's
  mid-turn saves don't toast. Imgui-coupled services the core needs (`exporter_registry` because the
  exporters carry panels, `shader_lib_files`) stay App-side, reached via injected getters. A NEW core
  mutation whose UI reaction touches imgui state adds an `on_*` callback (default no-op), never a direct
  imgui import. Spec: `ai_docs/features/025_project_session_extraction.md`. Revisit if a core op needs a
  UI reaction that a fire-and-forget callback can't express (e.g. it needs a return value App reads).
- **`tabs/*.py`: free `draw(app: App)` + optional `update(app: App)`.** `draw()` does imgui calls
  only; `update()` runs *before* imgui draws, for GL/canvas/mtime work outside the frame body. Tab
  state goes on `App` directly; a state-only sibling module (e.g. `tabs/share_state.py`) may hold
  its dataclass to keep `app.py` import-cycle-free. Revisit when a 4th tab module exists.
- **App-wide keyboard nav is region-confined (`nav_enable_keyboard` ON).** imgui nav is scoped to the
  one focused region via `WindowFlags_.no_nav_inputs` on the *inactive* region `begin_child`s; a
  caret-owning pane (the code editor) carries it *permanently* (it's a focus stop, not a nav surface).
  So a new top-level focusable region MUST flag itself `no_nav_inputs` when not the active region, and
  a new caret/text-edit pane MUST carry it always — else nav leaks across borders or fights the caret.
  Region focus moves via `set_next_window_focus()` before the target `begin_child` (the
  `set_window_focus(name)` overload segfaults — `/imgui-ui` skill §8). Mechanics + the
  clean-vs-flat-chain fallback: `ai_docs/features/019_keyboard_navigation.md`. Revisit if a region is
  added/removed or the confinement model changes.
- **Color roles are SWAPPABLE accent vs FIXED semantic; fixed hues must not collide with any accent
  preset.** `theme.py`: `_P` (palette, the only home for literal colors) → `_ACCENTS` (presets; the one
  user-chosen "active/interactive" hue, rewritten by `apply_theme`) → `_ColorBag` role tokens (each
  maps to a `_P` entry, never a literal). A role is swappable (`ACCENT_*`) or fixed (`SELECT` /
  `STATE_*` / `TAG` / ...). The fixed role whose cue shares spatial context with the accent — `SELECT`
  (its outline nests inside the accent's region outline) — MUST use a hue no accent preset and no state
  color uses, or the two cues merge under some accent. Enforced by an import-time assertion in
  `theme.py`. A new theme supplies its own `_P` + `_ACCENTS` + role mapping; the assertion validates
  the `SELECT` choice. Revisit if a new fixed role gains accent-adjacent *outline* context (add it to
  the assertion).
- **`widgets/*.py`: free functions taking `app: App`, no wrapper, no protocol.** Widgets are an
  organizational convention, not a polymorphic contract — no `Widget` ABC, no shared return shape;
  each gets the shape that fits its job. Revisit if a polymorphic `list[Widget]` dispatcher materializes.
- **`popups/*.py`: free `draw(app: App)` functions; open/closed state lives on `App` as a single
  `PopupState` enum field.** The four modal popups (node creator, settings, emoji picker, shader-lib
  picker) share one `app.popup_state` field — `CLOSED` / `NODE_CREATOR` / `SETTINGS` / `EMOJI_PICKER`
  / `SHADER_LIB_PICKER`. Each `app.open_*()` helper sets `popup_state`; the single field IS the mutex
  ("at most one modal open" holds by construction — one field can't be two states). A new modal popup
  adds an enum member + its `open_*()` + a self-close to `CLOSED`. `app.any_popup_open()`
  (`popup_state != CLOSED`) is the render-gate question. The command palette (`is_palette_open`) stays
  a separate bool — non-modal, coexists with any modal. No popup classes. Revisit if a popup grows
  internal state that doesn't belong on `App`.
- **Inline editor state lives on `App`; disk is the source of truth.** One `TextEditor` per node
  (+ a parallel dirty-baseline dict), created lazily; `app.save()` flushes the dirty editor before
  writing the file; the mtime watcher re-syncs from disk on external change (disk wins). Editor
  per-instance footguns (palette, FPE-while-modal, cursor, font sizing) live in `## Known quirks`.
  Revisit if multi-file-per-node editing lands.
- **`InlineInput` dataclass for mutually-exclusive inline editors.** A picker / panel hosting
  multiple inline text-input affordances (rename / new-file / new-dir) uses one `InlineInput`
  instance per kind — `target: Path | None`, `buf: str`, `needs_focus: bool` with
  `open()` / `close()` / `is_open` (defined in `editor_types.py`). A single `reset_inline_state()`
  method enforces the mutex: every `begin_*` opener calls it first, then sets only its own fields.
  `needs_focus` is the one-shot the input's first draw consumes via `set_keyboard_focus_here(0)`.
  The shader-lib instances live on `ShaderLibFileManager` (`shader_lib/file_ops.py`); the picker
  reads them directly via `app.shader_lib_files.*` (no delegating facade on `App`). Revisit if a
  second multi-inline-input surface lands (promote `InlineInput` to `ui_primitives.py`).
- **One shared row primitive per row-KIND, not per-kind special-case rows.** A list/grid whose
  items come in kinds (regular uniforms vs engine/`auto` uniforms) draws every kind through ONE
  row helper (`uniform_name_label`) with style overrides, never a separate hand-rolled row per
  kind. A per-kind special-case row silently excludes that kind from cross-cutting features —
  feature 008 special-cased engine uniforms into a dim caption row, which left them out of the
  code↔panel hover/jump bridge until it was generalized. Revisit if a kind genuinely needs to
  opt OUT of a shared behavior (then gate the behavior, don't fork the row).
- **No `async` except where python-telegram-bot forces it** — and that runs off the render thread
  (worker thread + own asyncio loop), never `run_until_complete` inside the imgui frame. A *synchronous*
  network client (YouTube's Google libs) uses the same worker-thread pattern but WITHOUT the asyncio
  loop — `_worker_main` calls the blocking client directly. Revisit if a new async-required dep doesn't
  fit the worker-thread + own-loop pattern.
- **A worker↔main blocking primitive that latches `_shutdown` on `release()` MUST expose `reopen()` and
  be re-armed at turn start.** The copilot has two (`CopilotBridge` worker→main-GL, `GateChannel`
  worker→UI-confirm). Both latch `_shutdown` on a non-reusable `cancel_all()` (release), and `App._init`
  calls `release()` on the freshly constructed session before first use — so without a `reopen()` cleared
  in `enqueue_turn`, every `run_on_main`/`ask` short-circuits ("shutting down" / instant-cancel) forever.
  `cancel_turn`/`reset_conversation` use `reusable=True` (no latch), so only the release path needs the
  re-arm. A NEW such primitive must mirror this: `reopen()` + a call beside the existing two in
  `enqueue_turn`. Revisit if the latch model changes (e.g. a per-primitive ready flag replaces `_shutdown`).
- **The "current node" is a first-class subject; how a copilot tool addresses a node scales with the
  side effect's reversibility.** The app has exactly one selected node (`App.current_node_id`); the UI
  shows it, the editor binds to it; `switch_node` is the one tool whose job is to change it. A NEW
  copilot tool picks its addressing by RISK, not by reflex symmetry:
  - **Reversible / project-internal (read, edit, delete-to-trash, render-to-file)** → take an explicit
    node id, act WITHOUT switching. `read_shader` / the edit tools / `delete_node` / `render_image` /
    `render_video` all do this: they work across the project (or produce a local file) without
    disturbing the user's view, and the worst case is undoable.
  - **External + irreversible (publish_telegram / publish_youtube)** → operate on the CURRENT node
    ONLY, no node arg. A live post of the wrong shader can't be undone, so it must be the node the user
    is looking at: the copilot `switch_node`s first (the prompt enforces verify-current-before-publish).
  Spraying a node-id arg onto the *publish* tools is the anti-pattern — it lets the agent silently post
  a node the user isn't watching. Revisit the split if a real workflow needs background publish of a
  non-current node often enough that switching first is friction — then add the arg consciously, with a
  matching "you're publishing X, not the current Y" confirmation, not by default.
- **A copilot tool's interactive output is a STRUCTURED entity the engine renders, never a raw value in
  the model-facing message.** A tool returns `(ok, msg, payload)`: `msg` reaches the LLM, `payload` does
  NOT. A URL / file path / button / panel a tool surfaces goes in `payload` as a structured spec the UI
  renders as a first-class chat entity; `msg` stays a TERSE fact that also TELLS the agent a widget was
  shown (so it points the user at the button instead of pasting a raw value it shouldn't have). Two
  orthogonal vehicles, do NOT conflate: a **result widget** (`state.ResultWidget`, kind-dispatched in
  `copilot_chat`) is NON-BLOCKING (a link/path button — the agent doesn't wait); a **gate**
  (`GateKind`, the blocking worker↔UI round-trip) is for input the worker must wait on (CONFIRM /
  CREDENTIAL secret / CONFIG setup-panel). A new such affordance picks its vehicle by blocking-ness and
  reuses the existing channel — never a raw URL in `msg`, never a new event type, never overloading one
  `GateKind` for both. An inline setup panel REUSES the exporter's `draw_config_ui()` verbatim (entropy:
  the Settings widget set is the source of truth) + a Cancel the chat adds. There is a THIRD channel for
  the agent-vs-user split: when a tool's `msg` is heavy (read_shader's full source listing) the AGENT
  still needs it (it edits by line number) but the USER doesn't (the editor shows the code), so the tool
  puts a terse `payload["display"]` summary that the chat shows INSTEAD of `msg` (`AgentToolCard.display`,
  feature 020·23) — the full `msg` still rides the model's context + history. Revisit if a widget needs to
  carry typed input back (then it's a gate, not a result widget) or persist live state.
- **The copilot replay `history` is NATURAL-LANGUAGE ONLY: user messages + one engine-derived turn-summary
  each, never tool messages.** `_commit_turn` (`session.py`) appends `user` + ONE `assistant` rendered from
  a `TurnSummary` (`agent.py` — built deterministically from the loop's `_RunLog`, no extra LLM call); the
  full tool tail (incl. `read_shader`'s source listing) is consumed only to derive the summary, then
  discarded (feature 020·28, supersedes 020·23 D4). The full source is re-fetched live each turn, never
  persisted (a stale copy is worse than no copy). Invariants any change MUST keep: (1) WITHIN a turn the
  live loop's `messages[]` still carries full assistant+tool pairs (the provider 400s on an orphaned
  `tool_call_id`) — NL-only applies ONLY at the commit boundary, never mid-turn. (2) the summary must
  preserve the four cross-turn facts (every node referenced; a mutation's new value; the agent's stated
  assumption — verbatim reply at clean-done, the branch note at a cutoff; the irreversible-action ledger
  with identity, verbatim + uncapped) or a real intent regresses. (3) the block-prompt constructor
  (`prompt.py` `PromptBlock`/`Volatility`/`build_prompt`) composes `[static < rare < dialogue < pending]`;
  a new prompt tier is a named block at its volatility rank, not a string-concat. Prompt composition lives
  ONLY in `build_prompt`; the summary is produced in `agent.py` and rendered to a message in `session.py`
  (clean producer/render/compose split — `prompt.py` imports no agent/registry). The within-turn read
  de-dup + line-drift follow-up was CLOSED by 020·29 (the PER_TURN working-set block); reasoning-notes
  scratchpad + cross-shader derived-edit memory remain deferred in `todo.md`.
- **A new addressable copilot SOURCE kind gets a `<kind>:` prefix + rides the EXISTING read/grep, never
  a parallel tool.** Nodes are bare ids, library files are `lib:<path>`, templates are `template:<id>`
  (feature 020·22). A new readable source (a future preset, an example, etc.) mirrors this: a
  self-describing prefix the catalogue emits, a branch in `_copilot_resolve_source` + the read/grep
  builders (the SAME `ShaderView`/`GrepHit`, one implementation), and — if it's read-only — an EXPLICIT
  reject in the edit-target resolver (a `<kind>:` target returns an unresolved EditResult with an
  actionable message, BEFORE the node resolver, so a lenient-resolver refactor can't make it a silent
  edit target). Don't fork a `read_<kind>` method; don't merge the id namespaces (separate dicts, the
  prefix carries the read-only-vs-editable semantics). Revisit if a source needs WRITE access (then it's
  not just an address — it's an edit target with its own freshness/guard).
- **A copilot tooling/prompt GUARD earns its place only if a strictly BETTER model would still need it.**
  When a trace shows the agent doing something wrong, the fix must target a CLASS of failure derivable from
  OUR pipeline's design — a missing affordance, a false/misleading tool message, a real coherence hole — NOT
  an INSTANCE of the current model being careless or its provider's arg-transport being buggy. The test: "would
  a more careful LLM still trip here?" If no, it's model competence the vendor amortizes for free; building a
  guard for it is permanent prompt tax (every line is paid on every request + dilutes attention from the
  load-bearing rules) against a transient flaw. Two hard corollaries: (1) NEVER change a destructive edit's
  behavior on a heuristic GUESS the model can't see (an EOF-overshoot clamp that silently deletes lines under a
  green checkmark masks the agent's error — a loud reject that lets it self-correct is the floor); (2) a
  provider/transport quirk gets a TOOL-SIDE invisible normalizer at the parse boundary (the
  `_unescape_double_escaped` §J6 pattern), never a standing prompt rule. STOPPING RULE for trace-review-and-patch
  rounds: stop once a fresh session has no terminal failure, no NEW failure class, and a residual that fails the
  "better model" test — a new INSTANCE of a known class is NOT a trigger for another round; a new CLASS in a
  different session is. (Established by the 020·29 overfit audit, which cut 3.5 of 6 proposed guards.)
- **Editable metadata on a SHIPPED (read-only-in-bundle) resource is two-tier: a shipped default + a
  user sidecar at `app_data_dir()`.** A template description (feature 020·22) lives in the shipped
  `node.json` (the dev default, ships immutable) AND in a `TemplateDescriptionsStore` sidecar keyed by
  the stable full id; lookup is override-else-shipped AT THE CONSUMPTION SITE (never mutate the
  in-memory shipped object — that keeps "reset" = delete the sidecar key). The sidecar mirrors the
  shader-lib favorites/tags store posture (cross-project, fail-soft load/save, loaded in `__init__`).
  User-edit-wins-forever (a later shipped-default change is shadowed) — accepted, matches favorites.
  Revisit if a shipped-default update must win until the user touches it (needs a version/hash stamp).
- **Exporters: own thread, own panel, GL-free artifacts.** The `Exporter` ABC enforces thread
  affinity — render-thread methods may touch moderngl; worker-thread methods (`prepare`, `export`,
  the `_do_*`/`_handle_*` job handlers) MUST NOT, they see only `RenderedArtifact` (a pure value
  type). Each exporter owns its own panel UI. A disabled/stub exporter overrides `is_available ->
  False` + `unavailable_reason` (registry won't auto-activate it; UI gates on it). Revisit when a
  third *concrete* exporter lands.
- **The generic exporter seam carries NO exporter-domain vocabulary.** `RenderControl` is pure render
  plumbing; `exporters/base.py`, `registry.py`, `tabs/share.py`, `popups/emoji_picker.py` name no
  Telegram/sticker/pack/emoji concept. A per-exporter UI need (e.g. Telegram's emoji affordances)
  flows through the opaque `RenderControl.extras` bag, built by the exporter's `build_render_extras`
  from generic `OutletUiDeps` — never as a named field on the contract, never via `isinstance` in the
  share-draw loop. Exporter-only methods (e.g. Telegram's `set_default_pack`) stay concrete on the
  exporter and are reached via `isinstance` at the one `app.py` call site, not hoisted to the ABC.
  Revisit if a capability is genuinely shared by ≥2 concrete exporters (then promote it to the ABC).
- **Exporter share-panels are built from shared `ui_primitives`, not ad-hoc imgui.** The panel chrome
  every exporter repeats — the fixed preview box (`preview_box`, one shared `SIZE.SHARE_PREVIEW_*`
  size so all outlets match + can't jitter), the labelled fields (`labeled_text_input` /
  `labeled_multiline_input` / `labeled_drag_float` / `labeled_combo`, caption-on-top), the
  reserved fixed-height progress/result row (`status_slot`), the not-connected CTA
  (`unconnected_gate`), the connected status line (`connection_status`), the first-run steps
  (`setup_steps`) — lives in `ui_primitives.py`, generic. A new exporter composes these; it does NOT
  re-roll `caption_text + set_next_item_width + imgui.xxx` inline. Layout rule for the panel:
  preview-left fixed + taller than the controls column, controls stack top-down beside it — no
  vertical-alignment math. Revisit if a panel needs a layout these primitives can't express (then
  add a primitive, don't inline).
- **Integration credentials are GLOBAL, not per-project; everything else per-project stays.**
  Telegram bot token / linked user / pack list live in `integrations.json` (one `IntegrationsStore`
  rooted at `app_data_dir()`), injected into exporters via `registry.set_integrations(store)`. An
  exporter whose creds are global returns `current_settings() -> {}` so the per-project
  `exporter_settings` stays non-authoritative. The per-project pointer that remains (e.g.
  `telegram_default_pack`) lives on `UIAppState`. Revisit if an integration brings genuinely
  per-project credentials.
- **All on-disk state roots at `paths.py::app_data_dir()`** — projects, the active-project pointer,
  logs, `integrations.json`. Never call `platformdirs.user_data_dir(...)` directly (that path
  silently ignores the override); go through `app_data_dir()`, which honors `SHADERBOX_DATA_DIR`
  (cross-platform; used by `make run-bundle` for a throwaway fresh-install run). It lives in its own
  leaf `paths.py` (not `app.py`) so credential/store modules can root files without importing `App`
  (the no-`TYPE_CHECKING` rule would otherwise force a cycle). Revisit if a state root needs to
  diverge from this single base.
- **Release versioning is manual semver, bumped only via `make release VERSION=x.y.z`.** Bump
  rule: **major** = breaks users' existing projects / saved state (e.g. a non-round-trip app_state
  migration); **minor** = a backward-compatible feature; **patch** = fixes / other non-breaking
  changes. The version lives in `pyproject.toml`; not auto-derived from git. Bumped once at ship
  time, not per-commit (ship flow: `dev_flow.md`). Revisit if manual bumps are repeatedly forgotten
  before a release (then consider `git describe`-derived versions).

- **On-disk artifacts split by lifetime: durable-portable → the project dir; disposable-local →
  `app_data_dir()`.** A project dir (`app_state.json` + `nodes/` + `media/` + `trash/`) is a
  self-contained relocatable unit — it can live anywhere and travels with the user. So state that is
  durable, read back by the app, and conceptually part of the project goes INSIDE the project dir
  (e.g. the copilot conversation — feature 022 — lands at `project_dir/copilot/`). Machine-local
  disposable output goes in `app_data_dir()`: the app log (`logs/`, app-global — the watcher/exporters/
  startup log before any project) and the copilot trace (`copilot_traces/`, large debug ephemera, never
  read back, retention-capped). The test: would the user expect it to travel when they copy the project
  folder? Yes → project dir. No → `app_data_dir()`. Revisit if an artifact is genuinely both (then it
  needs an explicit copy/export path, not a default location).
- **Logging is configured ONCE at startup, never per-module.** `logging_setup.configure_logging()` (one
  call in `ui.py::main` + `scripts/smoke.py`) owns all loguru sinks: a terse INFO+ console + a rotated
  DEBUG+ file that is a strict SUPERSET of the console. Call sites only do `from loguru import logger;
  logger.X(...)` — no module calls `logger.add`/`.remove`/sets handlers (loguru is a process-global
  singleton, so centralizing the sinks is enough; a `get_logger()` gatekeeper would be ceremony that
  fights its design). Level discipline: high-level user events (node saved, export done, project loaded,
  copilot turn start/done, tool called) = INFO (console); lifecycle/diagnostic detail (worker/watcher/
  queue/bootstrap/per-frame) = DEBUG (file-only); WARNING/ERROR file-only except an app-level crash.
  New code adds `logger.X(...)` calls at these levels — it never reaches for `print`, a per-module
  format, or its own sink. The copilot is the one exception that ADDS a stream, not reconfigures one: its
  full-fidelity transcript is a SEPARATE sink (`copilot/trace.py` `tr.event(...)`), not a log level — a
  terse `logger.info("copilot tool #N <name> -> ok")` line is for the console, the full args/result go to
  the trace, never spelled out in a log line. Revisit if a module genuinely needs its own format/sink
  (none does today). Spec: `021_logging_refactor.md`.

*(Each bullet is a generic constraint on future code + a revisit trigger — NOT a feature changelog.
The `/sanitize` noise audit deletes bullets that narrate a one-off implementation choice; per-feature
mechanics live in the feature spec, SDK footguns in `## Known quirks`.)*

## Known quirks (library / SDK footguns + the workaround)

- **GLSL `#line N M` accepts INTEGERS ONLY for the file-id (`M`).** Outside the
  `GL_ARB_shading_language_include` extension (unreliable across drivers), no GL driver accepts
  `#line N "filename"`. The host emits `#line N <integer_id>` and keeps an `id -> Path` table
  itself (`shader_errors.SourceMap.file_id_to_path`); the error-parse regexes (`_NVIDIA_ERROR_RE`,
  `_MESA_ERROR_RE`) capture the file-id too. If you ever see a driver emitting `0:LINE` for a
  spliced library function's body, you forgot to emit `#line N <lib_file_id>` before that splice
  (`shaderbox/shader_lib/resolver.py::resolve_usage`).
- **Both log sinks set `diagnose=False` — so an exception log NEVER dumps local variable values.**
  loguru's default `diagnose=True` prints the full frame locals on `logger.exception(...)`, which would
  echo the OpenRouter key / Telegram bot token / OAuth refresh token straight into the console + the
  shipped log file. `logging_setup.configure_logging()` pins both sinks to `diagnose=False` (the
  traceback frames stay; the variable values don't). So a log line only ever contains what you put in its
  message string — when you log around a secret-holding call, log the OUTCOME (`ok=False`, an error class),
  never the value. Re-verify the pin survives if you touch `logging_setup.py`.
- **A pre-freeze repaint needs `gl.finish()` to actually reach the screen.** `glfwSwapBuffers` only
  QUEUES the back buffer; the compositor presents it on its own cycle. If the main thread swaps and then
  immediately blocks for seconds (a synchronous render/encode that freezes the frame loop), the queued
  "Rendering…" cue frame may NEVER composite — the user sees the pre-render frame the whole time. The fix:
  draw the cue, `swap_buffers`, then `gl.finish()` (blocks until the GPU has executed the present) BEFORE
  the blocking encode (`ui.py::update_and_draw`, the Render-tab `render_request` path). Verified live on the
  maintainer's X11 box: the cue was correctly scheduled three times and stayed invisible until the swap was
  followed by `gl.finish()` before the encode. The copilot bridge render path runs its encode at the TOP of
  the frame (`drain_bridge`, before the swap) and so has the SAME latent invisibility — tracked in `todo.md`.
- **imgui / imgui-bundle quirks live in the `/imgui-ui` skill §8.** That includes: TextEditor
  palette read-only, monochrome emoji, dynamic glyph loading, `push_font` rasterized-size,
  `image()` lost `tint_col`, glfw cursor sync gap, pfd non-blocking handles, TextEditor
  first-render focus grab, the `.pyi`-only stub pyright warning, the SetCursorPos assert.
  Non-UI library quirks (telegram, moderngl, GLSL `#line`) stay below.
- **A live moderngl context must exist before constructing `Image` / `Video` / `Font` / `Canvas` /
  `Node`** — they call `moderngl.get_context()` lazily. In the app,
  `glfw.make_context_current(window)` handles it.
- **python-telegram-bot's `Bot` has TWO request pools; both need the IPv4 bind.** On an
  IPv6-incapable network (AAAA resolves but the route is dead — the dev box, see vpn-stack Gotcha #4),
  ptb dials the v6 address and the TLS handshake fails (`ConnectError(EndOfStream())`, surfaced as a
  bare `httpx.ConnectError: `). `_ipv4_request()` (`local_address="0.0.0.0"`) forces v4 — but `Bot`
  takes it via BOTH `request=` (regular calls) AND `get_updates_request=` (the separate long-poll pool
  `get_updates()` uses). Pass it to both, or `get_updates()` silently dials v6 and the link flow fails
  at the user-id capture step while `get_me()` succeeds (`exporters/telegram.py::_with_bot`). Also:
  ptb's default `HTTPXRequest` timeouts are **5s** (read/connect/write), which a VPN/tunnel routinely
  exceeds → `ReadTimeout`/`TimedOut`. `_ipv4_request()` passes generous explicit timeouts (30s + 120s
  `media_write_timeout` for the upload).
- **The sanctioned `# type: ignore` allowlist (upstream stub gaps only).** The no-suppression rule
  has exactly these exceptions — all are missing/wrong annotations in third-party stubs, never our
  own type errors. New markers outside this list are a design smell; fix the design, don't add to
  the list. Re-audit when bumping `moderngl` / `pydantic`.
  - `moderngl.Uniform.gl_type` — not in moderngl's stub.
  - `@model_validator(mode="after")` on a method returning `Self` — pydantic's decorator stub
    mistypes the wrapped method.
