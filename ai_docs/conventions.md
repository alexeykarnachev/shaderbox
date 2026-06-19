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

- Full type annotations on all params and variables. Never use `from __future__ import
  annotations` — it's noise.
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

## Design decisions (we decided X; revisit if Y)

*(The first cluster below are cross-cutting design LAWS — how to shape a decision — re-derived under a
dozen names across features; read them at spec-review time. The rest are concrete architecture
decisions. Source for the laws: the 2026-06-13 audit, `046_knowledge_base_refactor.md`.)*

- **Structural impossibility over guard-piles — the first question of any validation-heavy review.**
  If you find yourself adding a SECOND wave of guards to second-guess what an actor (a model, a caller,
  a migration) MEANT, the CONTRACT is unsound — redesign so the unsafe outcome can't be EXPRESSED, then
  delete the guards. The repo keeps converging on this: `extra='forbid'` makes a bad migration LOUD
  instead of silently dropping keys; gate on actual-rendered-size, not stamped intent; pick the
  dirty-signal the success path CLEARS (`node.program is None`), not a derived value a reset keeps
  coherent; content-addressed edits make silent mislocation impossible by construction. The canonical
  counter-example is the five 038 brace-structure guards, ALL deleted by 039 (the 020·14→036→038→039
  arc — two wasted guard waves). When a spec's decision section is mostly validation logic, ask this
  first. Revisit: never — a guard pile that keeps growing IS the trigger.
- **Speculative machinery: the test is "is REMOVING it churn?"** Two rules coexist — "cut speculative
  API surface at design review" (AppContext deleted 002, `ctx.state` deleted 041) AND "keep latent
  machinery dormant if cheap and reversible" (010 FitPolicy enums kept). The reconciling axis: surface
  you must TEACH or MAINTAIN — an ABC method, a model-visible enum, a curated namespace whose every
  omission is a NameError — gets CUT; inert latent code with zero teach/maintain cost AND a named
  near-future consumer stays dormant. Don't abstract from N=1 (write the second real consumer's full
  field list — <30% lands inside ⇒ wrong axis); DO generalize when a simplicity constraint forces
  copy-paste of the feature's own essence (044's Verlet step across 4 files → the node script). Re-litigated
  from scratch in 002/010/041 for want of this bullet. Revisit if a third reconciling axis appears.
- **A cross-cutting guarantee is enforced at the single FUNNEL, not per-caller.** When an invariant
  must hold on EVERY path of a fan-out, put the bracket at the one shared funnel (or lift the fact ONTO
  the entity) + ONE invariant test asserting coverage across all paths — never a per-call-site bracket
  that a sibling silently misses. The per-caller bracket is a KNOWN dead-end (041 did it 3× before the
  `Node.render_media` funnel; 028's pointer clobber fixed in smoke then re-clobbered by the fixture →
  real fix one gate in `App.__init__`; the `.trash/` filter on one glob but not the watcher's). A
  SECOND fix of the same bug at a sibling site is the trigger to move to the funnel. This is the
  in-repo instance of the global blast-radius rule (`~/.claude/CLAUDE.md` — fix at the shared root).
- **Stateless-rebuild over stateful-daemon; consensus is not evidence.** Before building a daemon to
  hold expensive state, check whether that state is ALREADY cheaply serialized — if a code-read shows
  the rebuild is ~1 s (027: the conversation is a free NL-only replay; the EGL+worker rebuild it'd
  protect is trivial), the daemon is solving a non-problem. Corollary, hard-won: a MAJORITY of design
  agents agreeing is NOT evidence (one code-grounded contrarian overturned three on 027). When a
  majority converges on a design resting on a "state is expensive" / "this is slow" premise, require a
  code-grounded devil's-advocate that checks the premise against the runtime. Revisit: never (a law).
- **A model flag whose ROW is created lazily-on-draw cannot be set programmatically — the writer must
  eager-create.** When per-entity state lives on an object created lazily inside a DRAW loop (the
  `UIUniform` row born only in `tabs/node.py`'s uniform loop; feature 047's `is_script_active`), any
  programmatic or HEADLESS mutation that runs before that draw silently no-ops — the lookup returns
  None and the write is skipped. The bug is latent because the interactive path always draws the row
  first (the click that mutates is itself in the drawn UI), so it only bites smoke/dogfood/copilot/any
  off-draw caller. Fix at the WRITE seam: `setdefault` the row from the live source before reading or
  writing the flag (`_ui_uniform_for` now `setdefault`s `UIUniform.from_uniform(u)`), so activation is
  independent of a prior draw. The general rule: if model state is keyed by a lazily-drawn row, the
  setter owns the row's creation — never assume "drawn at least once". The verification that catches it
  is a HEADLESS one that exercises the consumer, not the producer (047's smoke tick-canary + the
  dogfood harness were the falsifiers — both were silently red until the eager-create). Revisit if row
  creation ever moves to an eager load-time pass (then the trap can't fire).

- **The shader library is layered around SIGNED distance (feature 032).** Sources `SB_sd_*` return
  an SDF (negative inside; documented exceptions like the zero-width `SB_sd_segment`); operators
  `SB_op_*` map SDF->SDF; renderers (`SB_fill`/`SB_fill_aa`/`SB_glow`) map SDF->mask. A new public
  helper must name its layer by prefix and take/return SDFs at the layer boundary — never a
  one-off mask utility (that's how a library rots into an effects zoo). Non-`SB_` names are
  library-private: catalogued nowhere, reachable transitively. Canonical seed lives in-repo at
  `shaderbox/resources/shader_lib/`. Revisit if a needed helper genuinely can't be expressed as
  source/op/renderer (the first candidate defines layer 4, not an exception). Spec:
  `ai_docs/features/032_sdf_shader_library.md`.
  - **To know what `SB_*` actually exist, grep the ACTIVE lib root, not `resources/`.** The compiler
    resolves includes from `paths.shader_lib_root()` = `app_data_dir()/shader_lib` (the user-side seed),
    NOT from the in-repo `resources/shader_lib/`; the active root is authoritative — `grep -rn SB_
    "$(uv run python -c 'from shaderbox.paths import shader_lib_root as r; print(r())')"`.
    (`shaderbox/shader_lib/` is the Python index/parser/resolver package, NOT GLSL — don't grep there
    for helpers.) NOTE: the active root can DIVERGE from `resources/` — helpers the maintainer authored
    only on the desktop live root (`SB_fbm`, `SB_tri_wave`) aren't resolvable on a fresh env; tracked in
    `todo.md [DEFERRAL] shader-lib edits on the desktop never flow back into the repo seed (one-way sync)`.

- **Three-layer UI architecture.** `app.py` = UI/glfw/imgui owner + lifecycle wrapper (windowing, GL
  context, editor sessions, popups, nav, the exporter panels) — no imgui drawing in the orchestrator
  sense, but it owns all imgui-bound state. `ui.py` = thin orchestrator owning the frame loop (`run` +
  `update_and_draw`). `widgets`/`popups`/`tabs` = pure draw functions taking `app: App`. (The split is
  forced by the no-`TYPE_CHECKING` rule: a draw fn annotating `app: App` while `App` imports it would
  cycle — so `App` lives in its own module.) Revisit if a 4th UI sub-package is needed. Tracked:
  `todo.md [DEFERRAL] split ui.py / app.py further`.
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
- **The CPU-script engine is headless `ProjectSession` code; scripts are project DATA; a node has ONE
  STATEFUL-class script; a new script LANGUAGE is a `Behavior` backend, not an engine-loop change (feature
  041→048).** A node carries at most ONE script, `nodes/<id>/scripts/script.py` (the node script), a
  user-finalized CLASS subclassing `ScriptBehavior` with `update(self, ctx) -> dict[str, value]` driving
  MANY uniforms from one instance (feature 048 collapsed the 044/047 second per-uniform path — `u_*.py`,
  the `(name,type)` tag binding, the copy-content selector, the two-pass script-vs-per-uniform override — to
  this single path; per-uniform private HELPER METHODS inside `update` cover the per-value case). **Per-
  instance state (`self.*`) persists across frames — the reason CPU scripting exists** (a stateless
  `sin(t)` belongs in the shader). Each dict value is validated against the live uniform via the shared
  `uniform_coerce` (a bare number for a scalar; `Vec2/3/4`/`Array`/`Text` for the shaped kinds). The engine
  (`shaderbox/scripting/`) is repo code owned by `ProjectSession`, imports no imgui/glfw and no concrete
  `Node` type (it works against the `EngineNode` protocol — `uniform_values` + `get_active_uniforms()`), so
  it stays in the 025 headless core. **Binding is by EXISTENCE**: `script.py` on disk IS the binding — no
  active flag, no activate step (`is_script_active`/`is_brain_active` retired in 048). **The body is `exec`'d
  VERBATIM** (no AST surgery; the file IS a class def). A script is plain Python — `import math`, the stdlib,
  AND a real `from shaderbox.scripting import Vec2, Vec3, …` all work (the exec `__builtins__` carries
  `__import__`); the 048 stub EMITS that explicit import so the available types are visible, and
  `behavior.py::_build_globals` ALSO injects the same names (base + `Ctx` + output types) as a FALLBACK so a
  deleted import line degrades gracefully instead of an opaque eager-annotation-eval freeze. An error's
  lineno points at the user source (the 039 ghost removed by construction). A broken script is
  **error-as-data**: the uniform freezes at last-good + records a `ScriptError`, never raising into the
  frame loop (mirrors `shader_errors.ShaderError`). Freeze granularity: a per-KEY coercion mismatch freezes
  only that key (`(node_id, name)`); a raw throw / non-dict return is behavior-level — freezes every name it
  drove last frame, records under the sentinel `(node_id, "script.py")`. A key naming an engine-owned
  (`u_time`…) uniform is dropped SILENTLY; an orphan/typo/sampler key records a soft `(node_id, name)` error
  + skip. NaN/Inf is frozen-as-data like a shape error.
  **PLAY/STOP is node-scoped + name-keyed model state, NOT a per-`UIUniform` flag (feature 048).** A uniform
  the dict returns PLAYS (the engine writes it each tick); the user STOPS it to edit by hand. STOP state is
  `UINodeState.stopped_uniforms: list[str]` + a node-wide `all_stopped: bool` (stored as a LIST, not a set —
  `UINode.save`→`model_dump()`→`json.dump` raises on a Python set; coerced to a set per-frame). Node-scoped
  name-keyed state survives a retype (the name is stable) and is reachable before any row draws — this is
  the deliberate avoidance of the lazy-row law (a per-`UIUniform` flag would re-trip the 047 ROOT-2 trap).
  The engine learns STOP via a fresh per-frame `tick(stopped=…)` param (`ProjectSession._stopped_for` builds
  it; the headless boundary holds — the engine never learns `UINodeState`, intent flows through a param like
  `engine_driven`): a stopped name STILL ticks the script (state advances) + STILL counts as driven (so its
  play button shows), but its WRITE is skipped — `driven.add(name)` precedes the coerce/write, and BOTH the
  success-write and the coercion-failure-freeze are guarded `if name not in stopped` so a stopped uniform's
  manual value is never clobbered. Node-STOP freezes WRITES, NOT ticking (else a later node-PLAY would resume
  from stale `self.*`); the UI auto-STOPs a playing uniform when the user grabs its widget
  (`is_item_activated()`, gated on PLAYING). **Determinism is SCOPED**: a `ctx.t`-pure `update` is identical
  live vs export; a stateful integrator is frame-rate-dependent live by design BUT its EXPORT is reproducible
  — **export ticks a FRESH per-export instance** (`fresh_behavior_for` + `tick_export`, recompiled from
  cached source, NO stopped set — an export always plays), so live state never poisons an exported render.
  The isolation is STRUCTURAL: `Node.render_media` enters an injected `Node.export_isolation` factory around
  its whole body, so EVERY export (Render tab / Share scratch / copilot tools all funnel through
  `render_media`) is isolated with no per-caller opt-in to forget; `Node` stays engine-free. The **live path
  ticks once** (`session.tick` in `ui.py`). A future C backend implements the same `Behavior.run` protocol
  over a `.so` with no engine-loop change. Spec: `ai_docs/features/048_single_script_play_stop.md` (041 the
  origin, 044/047's per-uniform half superseded). Revisit `ctx` when a script needs cross-NODE state (the
  mini-game milestone — `todo.md`).
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
- **Inline editor state lives on `App`; disk is the source of truth; one `TextEditor` per opened
  FILE.** The editor is a TAB BAR (feature 045): an ordered `editor_tabs: list[EditorTab]` +
  `active_tab_index`, over a path-keyed `editor_sessions: dict[Path, EditorSession]` (the lazily-built
  `TextEditor` + its dirty baseline, one per on-disk file). A tab's `kind` (`shader` / `script` / `lib`,
  feature 048) drives its node-derived display label (`tab_label`: `<node> (shader)` / `(script)` /
  `library - <file>`) + the error tint; the imgui `##id` keys on the stable path/index, never the mutable
  label. The same file is never opened twice (`_focus_or_add_tab` focuses the existing tab). Editing
  acts on the ACTIVE tab: `flush_current_editor()` flushes its dirty editor before any save; the mtime
  watcher re-syncs every open session from disk on external change (disk wins). A node's editors close
  with the node (lib tabs survive); a renamed file re-keys its session in place. Editor per-instance
  footguns (palette, FPE-while-modal, cursor, font sizing) live in `## Known quirks`. Revisit if a tab
  needs durable per-tab state beyond its open files (e.g. persisting the open-tab set across restart) or
  a 4th editable `kind` lands.
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
  More generally: a blocking worker↔main round-trip is ONE primitive shape — `CopilotBridge`
  (worker→main-GL) and `GateChannel` (worker→UI-confirm) are mirror DIRECTIONS of it, not two designs
  (the mirror-the-sibling default produced both AND the dropped-`reopen()` bug). Its teardown contract:
  `cancel_all()` to release every blocked waiter BEFORE `join(timeout)`; on a join timeout you ABANDON the
  survivor rather than block shutdown forever — so the worker thread is **daemon** (a non-daemon worker
  blocked in a stalled stream past the join timeout would be re-joined by interpreter `_shutdown` and hang
  the process forever, the 043 headless hang; the cancel-before-join already tells it to stop, so abandoning
  it is safe). A new such primitive carries the whole bundle, not just the happy-path round-trip.
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
- **Copilot source edits are content-addressed ONLY: `edit_shader` (old_str/new_str, token-matched)
  + `write_shader` (whole file).** No tool addresses source by line number or by location-anchor
  quote. The line/anchor scheme was built (020·14), re-anchored to text (036), grew five
  deterministic guards (038), and was REMOVED by 039 after an adversarial review showed the
  addressing class itself is unsound: any location token the model must reproduce inherits model
  imprecision, and the engine either trusts it (silent mislocation) or second-guesses it (a guard
  pile + false rejects). Content addressing is safe by construction — you can only replace what you
  quoted verbatim; every failure is loud (no-match / multi-match). The whole-file silent risks are
  covered by deterministic facts: a truncated rewrite by the removed-names note
  (`EditResult.rewrite_note` via `parser.top_level_names`, filtered against the new text so a
  restyled signature is never falsely claimed removed), an unbalanced LIB write by the
  persist-seam brace warning (libs have no standalone compile to scream). Revisit ONLY per 039's Out-of-scope trigger (quoted there verbatim),
  and then via a verified number+text checksum design, never bare anchors.
- **The copilot SCRIPT-authoring tools MIRROR the shader tools exactly, fed SYNCHRONOUSLY by an isolated
  dry-tick (feature 043).** `read_script`/`write_script`/`edit_script` ↔
  `read_shader`/`write_shader`/`edit_shader` — a SEPARATE trio, NOT a `target:"script:"` overload. The
  split is structural, not symmetry: a script is a different LANGUAGE (Python, not GLSL → `edit_script`
  matches PLAIN TEXT via `_plain_text_spans`, NOT the GLSL `token_match` — Python indentation is
  semantic, so whitespace-tolerance is unsafe; `comment_loss`/the near-match hint are correctly omitted
  for the same reason — under exact-match the risks they defend can't occur). The GENERIC resilience
  ports: `edit_script`/`write_script` share the 0/1/N-match contract, the `_splice`, the same write tail
  (`_apply_script_text` — an edit and a write give identical feedback), and the 033 force-restore (N
  broken edits → revert to last clean; a script has TWO failure modes, compile + runtime, so it is at
  least as loop-prone as a shader). A different result type (`ScriptError{compile|runtime}` not
  `CompileErrorInfo`), and carries a "drives uniforms" concept with no GLSL analogue — overloading
  would force a union return + a matcher-language branch (the guard-pile the structural-impossibility
  law forbids). The `target:"lib:"` precedent does NOT generalize (it overloads the SAME artifact kind,
  GLSL). Feedback is the make-or-break: a script's facts are TICK-GATED (the compile verdict comes from
  `reload`, but the driven set / coercion errors / values need `update` to RUN, and headless has no
  frame loop), so `ScriptEngine.dry_run` reads the live compile verdict then steps ONE fresh script instance
  CONTINUOUSLY through the export-clock frames (so `self.*` accumulates — an integrator animates) into a
  `values_sink` that leaves the live node byte-identical. The MOTION verdict is the value-diff across t
  (GL-free, exact — catches a pulse/color-cycle a pixel-bbox misses) + ONE corroborating render for the
  "visible / FLAT" honesty case a value-diff can't see. A script write captures into the turn checkpoint
  (`_capture_script`) like any mutating tool.
  **THE ROOT LAW the bug-density taught (read this before adding any HEADLESS reader of the script
  engine): the engine's per-frame bookkeeping SELF-HEALS — `errors` clears on a good tick, `last_driven`
  re-warms each live tick, `u_time` is real wall-clock matching the live frame. A multi-frame HEADLESS
  probe (the copilot's `dry_run`, a dogfood driver) is the first consumer that reads BETWEEN ticks, where
  none of that self-healing runs — so the probe MUST own its own ACCUMULATED, CLOCK-COHERENT copy and never
  fall through to the live bookkeeping. Concretely: `dry_run` accumulates "did it EVER fail across the
  window" (not the final-frame snapshot of the self-healing `errors`); it STASHES its driven set into
  `last_driven` so the working-set marker + the `set_uniform` reject (which read `script_driven_uniforms`)
  agree with the write verdict in a tick-less path; the corroborating render takes its `u_time` from the
  SAME sample the values came from (`_render_facts_for(node, t=mid[0])`), so the rendered frame is the one
  the values describe. Every 043 bug (the swallowed transient raise, the phantom-numeric driven row, the
  wall-clock-at-t=0 render, even C1's un-captured `scripts/`) was one instance of this class — a live
  self-healing fact leaking into a headless probe context.** Revisit if a second script LANGUAGE lands
  (the dry-tick is language-agnostic via the `Behavior` protocol) or if the agent needs play/stop control
  (a `stop_uniform` pair, deferred). Spec: `ai_docs/features/043_copilot_scripting.md`.
  **Corollary (feature 050): a copilot probe render's clock is a DEFINED, STABLE time, never live
  `glfw.get_time()` wall-clock.** `_render_facts_for` defaults `t=0.0` (the export clock the user
  renders), so an animated shader's facts don't drift with app uptime and correlate with what the user
  sees; a caller wanting another moment passes `t` explicitly (the script probe's `t=mid[0]`, the
  `probe_render` tool's chosen `t`). A wall-clock probe was the reference-trace bug (facts stamped
  `t=951s` after 16 min open). Revisit if a probe ever legitimately needs the live preview clock
  (unknowable headless — t=0 is the robust default).
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
- **A static per-tool fact is a `ToolDefinition` field; a per-RESULT rendering trigger is a payload-shape
  key.** Everything true of a tool regardless of any one call (labels, gate prompt + policy, schema,
  precheck) lives ON the entity at its single definition site (`tools/{shader,publish,telegram,youtube}.py`)
  and resolves through `ToolRegistry` — never a parallel name-keyed dict (feature 029 deleted the two that
  existed, `_TOOL_VERBS` + `_GATE_PROMPTS`). What a tool's RESULT looks like in chat keys on the payload's
  shape (`"errors"` / `"hits"` in payload — `session._tool_card_outcome`), not on the tool's name. The one
  sanctioned name-key left: `delete_node`'s Recover affordance (n=1; a `recoverable` trait field is
  speculation). Revisit at a second recoverable tool, or a third payload-shape trigger (then consider an
  explicit payload contract type).
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
- **Render output size is ONE named vocabulary (`RenderShape`), not raw dims per caller.**
  `render_shape.py` owns the single home: a `RenderShape` StrEnum (`NATIVE` + `SHORT_720/1080/1440`
  9:16 + `WIDE_720/1080/1440` 16:9) with the aspect BAKED into each tier, a `SHAPE_TABLE` mapping each
  to its spec, and one `shape_to_preset(shape, *, is_video, fps, container, duration_max)` resolver
  that lowers a tier into a transient `RenderPreset`. The Share UI (`_RenderState.shape`), the copilot
  render tools (`render_image`/`render_video` take `shape: RenderShape`), and copilot publish
  (`publish_youtube(shape)`) all speak it — so a render matches what publish emits BY CONSTRUCTION
  (the old raw-w/h render vs exporter-preset publish disagreement is gone), and the closed Pydantic
  enum makes an off-aspect Short structurally unrepresentable (the structural-impossibility law, not a
  guard). The shape owns ONLY size+aspect; fps/container/duration_max stay per-OUTLET caller args (a
  Short is 60s .mp4, a sticker 3s .webm — not a shape fact). DELIBERATELY out of the vocabulary: the
  Render-tab `ResolutionDetails` (a free-form WxH artist control that is ALSO the actual-rendered-dims
  record, persisted in node.json — a 7-value enum can't carry concrete dims, and it's a different
  concept with its own home) and Telegram's hard 512px cap (a platform limit, not a user choice).
  Revisit if a third exporter needs a size tier the table lacks, or a real need for free copilot dims
  surfaces (then a `CUSTOM` member, NOT a return to raw w/h — that re-admits the foot-gun).
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
- **The active-project pointer (`app_data_dir()/project_dir`) is persisted ONLY for a real launch,
  never for an explicit-dir process.** `App._init` writes the pointer (so the next launch reopens the
  same project), but `App.__init__` passes `persist_pointer=False` whenever `project_dir` was given
  explicitly — that means a test/smoke harness (`scripts/smoke.py`, the pytest `app` fixture) driving
  `App(project_dir=<tmp>)`. WITHOUT this, an explicit-dir process overwrites the user's pointer with a
  throwaway tmp path that's deleted on exit, so the next real launch reads a dead pointer and silently
  falls back to a different/empty project — the user's just-created nodes appear "gone" (they were
  saved into the tmp project they were unknowingly working in). `open_project` (a real user action)
  uses the default `persist_pointer=True`. Revisit only if a headless harness ever legitimately needs
  to set the user's active project.
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
- **`Node.render()` owns per-type uniform defaulting; a persist path SEEDS, it never relies on a prior
  render.** A node's `uniform_values` dict is filled by `Node.seed_uniform_values()` (block → zero
  buffer, sampler → default `Image`, scalar/vector → `uniform.value`) — the single home for "default of
  a uniform of type X". `render()` calls it; so must any code that reads `uniform_values` for an active
  uniform without a guaranteed prior render — `UINode.save` does. A new persist/serialize path that
  swaps source then reads `uniform_values` MUST call `seed_uniform_values()` first (else it KeyErrors on
  an unseeded uniform; a naive `.get(name, uniform.value)` is WRONG for samplers — their GL default is
  an int texture-unit). `ENGINE_DRIVEN_UNIFORMS` (in `core.py`) is the one home for the
  `u_time/u_aspect/u_resolution` skip set — never re-list the three names. Revisit if uniform defaulting
  needs a value the GL default can't express.
- **Thread/GL affinity is enforced by METHOD ownership, not import boundaries; cross-thread reactions
  are injected callbacks.** GL objects live with the render thread; a worker thread never touches
  moderngl — affinity is a property of WHICH method runs where (the `Exporter` ABC's render-thread vs
  worker-thread split), checked by review, not by what a module can import. Free lunches the repo
  reuses: the mtime watcher already marshals work to the main thread, so a worker that must touch GL
  after a file write rides the watcher rather than inventing a queue. A worker→main reaction returns
  NOTHING — it fires an injected `on_*` callback (the `ProjectSession` idiom), because the consumer
  (`App`) is off the call stack mid-turn. Persist a project split at the QUIESCENT between-turns
  boundary (the worker is idle — no lock needed; 022's save gate). Classify a constructor's deps by
  VOLATILITY: project-dependent state comes through getters (it changes when the project switches),
  only build-time-shipped values are frozen into the instance. Revisit if a worker genuinely needs a
  synchronous return from main (then it's a blocking primitive — see the latch bullet, not a callback).
- **NO backward-compatibility / migration code, EVER (unless the maintainer explicitly asks).** The
  project is unreleased-evolving + solo: the ONLY data that exists is the tracked `projects/dev/` sandbox,
  which the maintainer edits by hand. So when a change reshapes a model / on-disk format / filename scheme /
  protocol, just CHANGE it and fix the sandbox by hand in the same `git add projects/dev` wave — NEVER write
  a migration function, a compat shim, an old-format reader, a "fold the old artifact into the new" step, a
  deprecation alias, or a one-time-cleanup pass. These are pure dead weight (no user has the old data) and
  they rot. The recurring failure: a spec/review/swarm "helpfully" proposes a migration for the deleted old
  shape (048 invented a `u_*.py`→`script.py` migration that shipped + had to be ripped out) — when you see a
  migration/back-compat proposal, DELETE it, don't implement it. This OVERRIDES any instinct (and any
  swarm finding) toward graceful evolution. The one sanctioned exception is the persistence-LOADABILITY
  posture below — keeping a model *openable*, NOT migrating its data. Revisit only if the maintainer ships a
  real release with real users AND explicitly asks for a migration.
- **Persistence-evolution posture: a reshaped model stays LOADABLE, it is NOT migrated.** `extra='forbid'`
  on a persisted model is the VERIFIABILITY primitive — it makes a stale/foreign key LOUD instead of
  silently dropped, so it's the default for any state read back by the app (`UIAppState`). Adding a field:
  make it defaulted + loose-typed-optional + fail-soft on load (no migration). A field whose meaning
  intentionally CHANGED, or that is removed: just drop it — `extra='forbid'` rejects a stale file into the
  fail-soft path (or the unknown-key filter ignores it), rather than silently reinterpreting it. An unknown
  enum member / message role degrades to a plain-text fallback, never a hard load failure (a future-written
  file must still open). This is about LOADABILITY (a dev-era file still opens clean), NOT data migration —
  per the no-backward-compat rule above, there is no `load_and_migrate` ladder; the App-state version stamp
  exists only to fail-soft-reset a foreign file, not to walk it forward.
- **Two parallel name-keyed dicts that must stay in lockstep are a drift smell — lift the fact ONTO
  the entity + pin it with ONE invariant test.** When two `dict[name, X]` and `dict[name, Y]` are
  keyed by the same identifier and a new entity must be added to BOTH, a caller WILL forget one (029
  found `_TOOL_VERBS` + `_GATE_PROMPTS`; the 031 sweep found 10 instances repo-wide). The remedy is
  structural, not vigilance: make the facts FIELDS on the one entity (`ToolDefinition`), resolved
  through one registry, and add a single test asserting every entity carries every fact. Revisit only
  if a fact genuinely can't live on the entity (then it's a different concern, not a parallel dict).
- **Snapshot/restore: serialize the LIVE object, restore by reload-and-replace across EVERY live
  surface; a serialize routine must not MUTATE what it serializes; capture is best-effort.** Disk is
  NOT live state — a blind dir-copy captures a stale `node.json`, so a snapshot serializes the live
  in-memory object. Restore replaces the live object on every surface that holds a reference (not just
  the one you're looking at), via reload-and-replace. A serialize path that mutates as a side effect
  (rebinding `source.path` while writing) is LATENT CORRUPTION — give it a no-rebind/pure mode.
  Capture is BEST-EFFORT: it must never fail the operation it's guarding (a checkpoint that can't
  snapshot logs and proceeds, it doesn't abort the user's edit). Revisit if a restore must be
  transactional (all-or-nothing across surfaces) rather than best-effort.
- **The copilot tool boundary (`registry.execute`) splits domain-reject from bug.** A `CopilotToolError`
  is a DELIBERATE reject whose message is authored for the model — surfaced verbatim, logged at warning.
  Any other exception is an unexpected bug — only its class name reaches the model (message/traceback
  could carry paths/secrets), full traceback to the debug log. A tool handler signals an expected
  failure by `raise CopilotToolError(<model-facing message>)`, never by returning a bare string through
  the generic path. Revisit if a third class (e.g. a retryable-vs-terminal distinction) earns a branch.

*(Each bullet is a generic constraint on future code + a revisit trigger — NOT a feature changelog.
The `/sanitize` noise audit deletes bullets that narrate a one-off implementation choice; per-feature
mechanics live in the feature spec, SDK footguns in `## Known quirks`.)*

## Known quirks (library / SDK footguns + the workaround)

- **A dynamically-indexed `const` array is NOT constant storage on NVIDIA — big lookup tables
  must be UNIFORM arrays (engine-bound).** Measured on the glyph stroke tables (RTX 3090,
  575.xx, text stack @800px): function-local const ~432 ms/frame (re-materialized per call);
  global const ~10 ms (still demoted to per-thread local memory); `uniform vec4 […]` ~0.13 ms
  (true constant bank — on par with fully inlined code). Mesa/V3D is indifferent to all three
  (~30 ms @300px) but chokes on the INLINED alternative (the pre-032 glyph switch cost ~20 s of
  codegen), which is why tables exist at all. The plumbing: `scripts/gen_glyphs.py` emits
  `text/glyphs.glsl` (uniform declarations + readers) together with `shaderbox/glyph_tables.py`
  (the packed values); `Node.compile()` writes them into any program that uses them;
  `ENGINE_DRIVEN_UNIFORMS` keeps them off save/seed/UI/set_uniform. The shader-lib index
  extracts top-level `const`/`uniform` declarations as spliceable entries
  (`shader_lib/parser.py::DECL_SIG_RE`, one declarator per line). Small consts (an `SB_PI`)
  are fine as global const; it's dynamic indexing into a big table that hits the demotion. If a
  lib/table shader is mysteriously slow on one vendor only, check WHERE its table data lives
  first. Mesa's linker also constant-folds a compile-time-constant glyph index and TRIMS the
  uniform array's ACTIVE size to a prefix of the declaration (verified on Mesa 24.2.8/V3D:
  `array_length` reports 1, a full-size `write` raises and the table stays zero) — `Node.compile()`
  clamps the table write to `array_length * element_size`. The deeper reason tables beat branches here:
  V3D defers final GPU codegen to the FIRST DRAW (it benchmarks up to ~13 register-allocation
  strategies then), so a big branchy `switch`/`if`-ladder pays its whole codegen as a one-time
  first-draw stall (the pre-032 inlined glyph switch: ~20 s) — a flat data-table lookup doesn't. Prefer
  data TABLES over branchy GLSL on any V3D path.
  - **Two consequences when AUTHORING a shader that adds uniform arrays (e.g. text captions):**
    (1) the glyph tables already consume most of the driver's constant-register budget (~600 of
    ~1024 slots), so adding several/large `uniform <T> arr[N]` on top can overflow with
    `C6020: Constant register limit exceeded` — keep added uniform arrays small + few, pack N strings
    into one array rather than N arrays. (2) `gen_glyphs.py` regen requires an app **RESTART** to take
    effect: the engine binds the table VALUES from `shaderbox/glyph_tables.py` (a Python module loaded
    at startup, NOT hot-reloaded like a shader), so a running app keeps the old tables and renders
    new-glyph text blank/garbage until relaunched.
- **A `MESA_*` / `SHADERBOX_DATA_DIR` env override must be a MODULE-TOP side-effect set BEFORE the first
  `shaderbox` import, and fail LOUD if it can't be.** `MESA_GL_VERSION_OVERRIDE` /
  `MESA_GLSL_VERSION_OVERRIDE` (lifting the reported GL version to 4.6/460 so `#version 460` compiles on
  a v3d/llvmpipe context that reports ≤4.5) are read by the driver AT context creation;
  `SHADERBOX_DATA_DIR` is read by `paths.app_data_dir()` AT import time. So a harness/script that needs
  them sets them at the very top of its entry module, before importing anything under `shaderbox` (the
  worked pattern: `scripts/dogfood/harness.py`'s module-top `os.environ.setdefault(...)` block, then
  `import glfw  # noqa: E402`). Set too late, the context is already created / the path already resolved
  and the override silently no-ops — which reads as "the engine is broken" (a version-error compile, or
  state landing in the wrong dir), not as the config mistake it is. Never bury such an override inside a
  function.
- **A glfw key-filter callback must NEVER swallow RELEASE events.** The Esc filter
  (`app.py::_install_escape_filter`) gates on `escape_has_job()`, but the job routinely
  disappears between press and release (Esc's own handler defocused the chat) — a swallowed
  release left imgui's Escape logically held FOREVER, so every InputText activated afterwards
  self-cancelled on the key-repeat ticks (a caret that dies 2-3 frames after every click; F13).
  Gate presses/repeats only; forward releases unconditionally (releasing an already-up key is a
  no-op). Debugging this class: imgui's own debug log names every ActiveId change with its cause
  (`ctx.debug_log_flags = imgui.internal.DebugLogFlags_.event_active_id | event_io |
  output_to_tty`) — far better than hand-rolled focus logging; headless repros MISS this class
  because injected io events bypass glfw callbacks.
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
  the blocking encode. Verified live on the maintainer's X11 box: the cue was correctly scheduled three
  times and stayed invisible until the swap was followed by `gl.finish()` before the encode. **EVERY render
  encode shares ONE post-swap firing point** (`ui.py::update_and_draw`, after `swap_buffers`): the Render
  tab + the Share-tab outlets feed it through `RenderDefer`; the copilot feeds it through the bridge's
  parked render op (`bridge.drain` parks a `defer=True` op, `run_deferred_render` fires it post-swap). The
  divergence that recreated this bug was a SECOND firing point — the Render tab fired post-swap with
  `gl.finish` while the copilot ran its encode at the TOP of the frame (and the Share tab ran it inline mid-
  draw): a sibling encode silently missed the guarantee. The lesson is the funnel rule applied to a timing
  guarantee — a "present before you freeze" invariant belongs at the single post-swap funnel, not re-derived
  per render entry point. A NEW render entry point MUST route its encode here, never call it inline.
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
  - `moderngl.Uniform.gl_type` — not in moderngl's stub (`uniform_coerce.py`, `ui_models.py`, `util.py`,
    `copilot/backend.py`). NOTE: the script-engine `exec()` seam (feature 041, redesigning 040) needs
    NO suppression — ruff's `S102` (flake8-bandit) isn't in this repo's `select`, and pyright's basic
    mode flags no `Any`-flow on the namespaced `exec`. The exec globals are the REAL builtins
    (feature 045 — a script is plain Python, `import math` works) plus the injected top-level names
    the eager method annotations resolve against (`Ctx`/`MouseState`/`Vec2..4`/`Array`/`Text`); the
    `ScriptBehavior` base is injected too. `__build_class__` is in the real builtins (the class
    statement is satisfied), so no curated `__builtins__` dict is needed.
  - `@model_validator(mode="after")` on a method returning `Self` — pydantic's decorator stub
    mistypes the wrapped method.
  - `moderngl.create_standalone_context(backend="egl")` — the stub types `**kwargs` as a single
    `dict`, so a keyword arg trips `arg-type` (`scripts/dogfood/harness.py`, the headless EGL context).
  - `openai`'s `chat.completions.create(messages=, tools=)` rejects plain dict literals (its params
    are TypedDicts; production goes through `openrouter.py::_to_wire_message`). `scripts/token_probe.py` (a
    throwaway token-measurement probe) builds wire dicts by hand -> `arg-type`/`list-item`.
