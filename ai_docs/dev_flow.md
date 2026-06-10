# Dev flow

How work happens in this repo. **Reference, not a mandate** — solo project, no enforced workflow;
pick the loosest shape that proves the change. Re-open this file between steps: the known failure
mode is pattern-matching a step name to a shorter remembered checklist and silently skipping the
passes that weren't cached.

The cold-start chain: `CLAUDE.md` → `ai_docs/roadmap.md` (what's built + the Active-context banner =
what's next) + `ai_docs/todo.md` (landmines/triggers) → this file.

---

## Branch model

Develop on `dev`, ship from `master`. All freestyle work accumulates as commits on `dev` (no
per-feature branches); `master` is the line actually released to itch and only advances at ship time.
So `dev` can sit many commits ahead with no version bump — the bump happens once, when `dev` is
ship-ready (`## Build / ship to itch.io`).

---

## Three kinds of work — pick the shape

| Trigger | Shape |
|---|---|
| **Small / mechanical change** — bug fix, log line, kwarg, trivial type fix, doc tweak, dep bump, dead-code deletion that touches no public surface, a `.gitignore` edit | Just do it. `make check`. Optionally `/simplify` on the diff. No spec, no review ceremony; a roadmap-banner touch only if it changes "what's next". |
| **Feature** — new module, a real behavior change, a refactor with blast radius (splitting `ui.py`, collapsing the sticker models, unblocking the render loop), acceptance criteria a quick manual check can't pin | The feature flow below. |
| **Research / brainstorm** — "let's think about X", "should we do Y", "investigate Z"; output is *knowledge*, not code | Research freely (read code, throwaway scripts). Deliver a chat report; if worth keeping, write `ai_docs/<topic>.md`. **Don't half-start an implementation.** If it seeds a feature, re-enter at the feature flow. |

When unsure, default to the loosest shape that proves the change; escalate if it grows. The agent
proposes the shape inline ("this looks like a small fix — direct, no plan-lock") and the user corrects.

---

## Feature flow

**Size preamble (read this first):** not every feature needs the full pipeline. The agent proposes
the size inline and the user corrects.

- **Mid (the default)** — a normal feature (multi-file, new module, real behavior change): **1-2
  pre-impl review agents + 2-3 post-impl review agents in parallel**. Manual check if user-visible.
- **Tiny / small** (≤3 files, 1 module, pure code, no new public API, no async/lifecycle): scale
  down — a short plan-message instead of a spec; **0-1 review agents** (skip review if the diff is
  trivially-obviously correct); skip the manual check if not user-visible.
- **High-blast-radius** (the `ui.py` split, a refactor across many modules, anything touching
  conventions): scale up — the upper end of the mid range or beyond (extra reviewers — e.g. a
  spec-fidelity auditor), plus a sanitization sweep even if it wouldn't normally warrant one.
  **Watch for the cycle-from-types signal:** if a new module needs `app: App` (or any other
  symbol from an upstream module that already imports the new module), the convention's
  no-`TYPE_CHECKING` rule will force a structural split — anticipate it in the spec
  ("module X holds the type; module Y holds the orchestration"), don't discover it at
  impl time. (Worked example: feature 002's spec.)

The non-mid sizes are judged per situation: the agent proposes "this looks tiny — 1 reviewer, no
manual check" or "this is high-blast-radius — 2 pre + 3 post + a spec-fidelity pass", and the user
corrects.

**Steps** (re-open this file between them):

1. **Research first.** Read the code the feature touches. Don't ask code-level questions you can
   answer by reading. Throwaway scripts if useful.

2. **Plan-draft** → `ai_docs/features/NNN_<name>.md` (create `features/` on the first one).
   **Pre-flight:** re-read `conventions.md` from disk and the `todo.md` entries this feature's
   trigger fires — if one contradicts the request, halt and reconcile (or get the user to confirm
   the change) *before* drafting. Minimum spec sections (keep the headers even when "N/A"): *Goal* /
   *Out of scope* (each deferral with a trigger) / *Design decisions* (numbered, lock-in only — open
   questions separate) / *Files touched* / *Manual verification* (what's verified by hand in the app) /
   *Open questions for the user*. Either self-write or spawn the built-in `Plan` agent (read-only —
   feed it the research findings + paths + `conventions.md` + every file the spec will touch, or it
   re-derives blind).

3. **Plan-lock with the user.** Show the spec, answer open questions, get explicit sign-off. Don't
   proceed to review until locked.

4. **Pre-implementation review** (per the preamble — 1-2 in parallel for a mid feature; 0-1 for
   tiny/small; up to 2 for high-blast-radius). Roles for 2: *correctness & design* (internal
   inconsistencies, convention violations, missing touches, anything contradicting a locked Design
   decision) and *verification & blast-radius* (does the manual-verification list catch realistic
   bugs, any invariant nothing verifies, does it correctly fold the `todo.md` deferrals this
   feature's trigger fires). A reviewer may return "this shouldn't land in current form" → bubbles to the user.
   **Disagreeing reviewers:** a design disagreement → the main agent decides and documents the call
   in a "Review history" section of the spec; only a "should not land" finding escalates to the user.
   **Triage:** fix findings inline in the spec, or add the "Review history" section recording what was
   rejected and why.

5. **Implement** in one coherent diff. Re-read `conventions.md` from disk first (don't code from a
   remembered version). Don't sidestep a convention (see the hard rule — no sanctioned type-
   suppression exceptions; imgui-bundle ships proper stubs as of feature 004). **Robust defaults:**
   when an open question blocks implementation, default to the robust answer, not a shim — only
   shortcut if the robust path is hugely out of scope, and then surface the trade-off so the user
   can choose. After impl: glance the diff for new `# pyright: ignore` / `# type: ignore` / `# noqa`
   / `TODO` / inline imports / `Any` on real-typed params — any of those means a shortcut was taken;
   fix it.

6. **Post-implementation review** (per the preamble — 2-3 in parallel for a mid feature; 0-1 for
   tiny/small; 3+ for high-blast-radius). Roles: *code correctness* (bugs, races, GL-context
   lifecycle, resource leaks, error handling), *architecture & conventions* (module boundaries,
   where things live, duplication, the imgui-interop patterns), and
   for a high-blast-radius diff a *spec-fidelity audit* (walks the spec end-to-end against the diff —
   every locked decision actually landed). Each reviewer runs `git show --stat <impl-commit>` first to
   know the full surface, reads every changed file end-to-end, and states which it skipped and why (a
   finding list without a coverage statement = under-read). Findings → triage → fix inline / file as
   `todo.md` deferral with a trigger / promote new design decisions to `conventions.md ## Design decisions`.
   The "2-3" is a FLOOR for a mid feature, not a ceiling — for high-blast-radius diffs, finalize
   sweeps, or a docs-harness audit, escalate to a larger parallel **swarm run as a convergence
   loop** (spawn all reviewers in one message; triage real-vs-false-positive; re-spawn the same
   reviewers against the patched state; repeat until every reviewer returns PASS). The mechanics
   + the late-round-fabricated-gaps warning are the canonical home of the `/review-agent-loop`
   skill (global) — don't restate them here. One hard-won rule worth repeating: at least one
   reviewer must anchor to an artifact you did NOT author (a sibling file, the running app, the
   user's verbatim message) — an all-self-authored swarm ratifies its own contradictions.

7. **Manual check** (when user-visible — most ShaderBox features). Run the app
   (`uv run python ./shaderbox/ui.py`), exercise the change, screenshot if useful. A real UX gap
   found here is a FAIL, not pass-with-caveat. (See `## Recipes` for what the app needs to run and
   what *can't* be exercised without secrets — e.g. the Telegram share tab.)

8. **Sanitization sweep** — the `/sanitize` skill. Separate commit (or its own line in the commit
   message).

9. **Done** — add or flip the feature's `roadmap.md` row (status + `Spec:` pointer) and **rewrite**
   the Active-context banner if "what's next" changed; move open items to `todo.md` with concrete
   triggers. The story stays in the feature spec / commit message — the row is one line.

---

## Mid-flight escalation

Four rules (the one whose absence produces the worst outcome — an agent quietly turning a small fix
into a refactor):

1. **Scope grew** → halt, propose to the user, don't quietly grow. Say exactly what's bigger and ask
   whether to extend the spec or file a separate feature.
2. **A reviewer surfaces "shouldn't land in current form"** → escalate to the user. Don't contort the
   spec to satisfy the letter of conventions while violating the spirit.
3. **A blocking open question** → robust default (see step 5), or escalate the trade-off if the
   robust path is hugely out of scope.
4. **A locked UI/visual decision proves wrong once rendered** → a rebuild is allowed without a full
   re-lock (you genuinely can't judge a layout on paper), BUT record the reversal in the spec's
   Review-history AND surface it to the user in the same wave — don't bury it as a fait accompli in
   the Design-decisions, and don't leave the old decision's Resolution/notes contradicting the new
   shape (that's how a spec ends up self-contradicting).

---

## Closing out work

Whatever the shape — before "done", run `/sanitize`. For a truly trivial mechanical change, the
roadmap-banner touch (if even that) + the cold-context glance is enough; for a feature, the full pass.

---

## Recipes

### Module map
(`shaderbox/` package + subpackages `exporters/`, `shader_lib/`, `tabs/`, `widgets/`, `popups/` —
this is the orientation `arch.md` would have been. Reshaped by feature 017.)

- **`core.py`** — `Canvas`, `Node`: GL program lifecycle, uniform introspection + binding,
  render-to-texture, image/video export. Needs a live GL context. Imports `shader_lib` (compile-time
  `#include` resolution) + `shader_errors`.
- **`project_session.py`** — `ProjectSession`: the headless project + copilot CORE (paths, nodes,
  app_state, lib index + cross-project stores, integrations, the `CopilotSession`/`CopilotBackend`/
  `RevertExecutor` cluster). Imports no imgui/glfw, creates no window/context — a headless harness
  (feature 026) constructs it on a standalone EGL context without `App`. UI reactions ride injected
  `on_*` callbacks (the seam + why a return-value seam is wrong: `conventions.md ## Design decisions`,
  the `ProjectSession` bullet). Feature 025.
- **`app.py`** — `App` class: the UI/glfw/imgui owner + lifecycle wrapper (windowing, GL context,
  editor sessions, popup-state, nav, exporter panels). Owns one `self.session: ProjectSession` +
  forwards project state/ops via `@property` accessors. No UI drawing. Imported by `ui.py`, `widgets/`,
  `popups/`, `tabs/`. Holds `self.shader_lib_files: ShaderLibFileManager`; the picker reads its CRUD +
  inline state directly via `app.shader_lib_files.*` (no `App` forwarding facade).
- **`ui.py`** — thin entrypoint + orchestrator. `run(app)`, `update_and_draw(app)` (the imgui frame
  loop: render gates + the main-window left/right split — LEFT = code editor via `code_tab.draw`,
  RIGHT = `_draw_app_panel`), `_draw_splitter`, `_draw_app_panel`, `_draw_node_settings`
  (Node/Render/Share tab-bar dispatcher), the **Shader Library** menu, `main()`. No tab bodies /
  widget logic / hotkey dispatch — those live in `tabs/`, `widgets/`, `popups/`, `hotkeys.py`.
- **`commands.py`** — leaf (imports `imgui` only, never `App`): the command registry that drives all
  keyboard control (feature 018). `CommandId`/`CommandScope` StrEnums, the frozen `CommandSpec` + the
  `COMMAND_SPECS` default table, `chord_to_str` (display), `route_flag`/`popup_suppresses`/
  `capture_chord`/`chord_needs_modifier`. The id->callback wiring lives on `App` (closes over self),
  not here, so this stays cycle-free.
- **`hotkeys.py`** — two halves of keyboard handling: `process_hotkeys(app)` (PRE-`new_frame`: glfw
  poll + imgui `process_inputs` only) and `dispatch_commands(app)` (IN-frame, top of the main-window
  block: registry-driven `imgui.shortcut()` dispatch reading `app.effective_bindings`, the bespoke ESC
  handler, and the node-creator arrow/Enter nav). The split exists because `imgui.shortcut()` asserts
  outside an active frame.
- **`editor_types.py`** — leaf dataclasses shared between `app.py` and its collaborators:
  `EditorSession` (one `TextEditor` bound to a path), `InlineInput` (mutually-exclusive inline
  text input), `JumpRequest`, `HoverMark`. (Extracted so `ShaderLibFileManager` + the UI modules
  import them without cycling through `app.py`.)
- **`ui_models.py`** — pydantic `UINode` / `UINodeState` / `UIUniform` / `UIAppState` + node
  (de)serialization. Pure data — does NOT import the UI layer (the preview-button draw helper lives
  in `widgets/node_grid.py`).
- **`shader_errors.py`** — leaf: `ShaderError`, `SourceMap`, `parse_shader_errors`,
  `next_error_line`, `find_uniform_declaration_line` (shader-compilation error domain; `core.py` and
  the shader-lib resolver import their error types from here, not from `util.py`).
- **`media.py`** — `Image` / `Video` (`MediaWithTexture` ABC), ffmpeg temporal smoothing.
- **`render_preset.py`** — GL-free `RenderPreset` (an outlet's render constraints: size/aspect/fps/
  duration/container/byte-cap/fit) + `resolve_dims`. Drives `core.py::render_media`. Feature 010.
- **`shader_lib/`** — the auto-resolved GLSL helper library (features 015/016/017). Import the public
  surface from the package (`from shaderbox.shader_lib import …`). `index.py` = `ShaderLibIndex` /
  `ShaderLibFunction` types + `build` + the module-level active-index singleton (`active`/`set_active`)
  + `is_shader_lib_path` + `_extract_functions`. `resolver.py` = `resolve_usage` + `ResolveError` (the
  per-compile usage pruner — splices used `SB_*` functions into a `#line`-marked preamble).
  `parser.py` = pure regex/brace text machinery (leaf, no index types). `favorites.py`
  (`ShaderLibFavoritesStore`) / `tags.py` (`ShaderLibTagsStore`) — cross-project JSON stores at
  `app_data_dir()`. `file_ops.py` (`ShaderLibFileManager`) — picker inline-input/filter state +
  file/dir CRUD (create/rename/delete-to-trash); explicit-args verbs (no `App` import — editor-session
  cleanup flows back via injected callbacks). NONE of `shader_lib/` imports `app.py`.
- **`exporters/`** — `Exporter` ABC + `RenderedArtifact` value type (`base.py`), `ExporterRegistry`
  (`registry.py`), `TelegramExporter` (`telegram.py` — own worker thread + asyncio loop + sticker
  panel UI) + `YouTubeExporter` (`youtube.py` — worker thread, no asyncio). `integrations.py` =
  global `IntegrationsStore` (Telegram bot token / linked user / pack list + YouTube creds) at
  `app_data_dir()/integrations.json`. `telegram_util.py` (`derive_set_name`, pure) + `youtube_util.py`
  (metadata builders + constants) — exporter-specific helpers. Adding an exporter: subclass `Exporter`,
  register in `App.__init__`. Worker thread MUST NOT touch moderngl (thread-affinity contract).
- **`paths.py`** — `app_data_dir()`, `shader_lib_root()`, `shader_lib_trash_dir()` + `ProjectPaths`
  (frozen value object for one project dir's layout; `for_root` eagerly mkdirs the 5 subdirs;
  `App.paths` holds the live one) (leaf, no `App`).
- **`tabs/`** — `draw(app: App)` UI modules + optional `update(app: App)` (pre-imgui GL work).
  `code.py` (inline GLSL editor — main-window LEFT split), `node.py`, `render.py`, `share.py`.
  `share_state.py` holds the share-tab dataclass (`TabState`) separately to keep `app.py`
  cycle-free (app.py imports `share_state`, NOT `share`).
- **`widgets/`** — stateless imgui-drawing functions taking `app: App`. No shared contract.
  `details.py`, `media_ops.py`, `node_grid.py` (incl. `draw_node_preview_button`, the free preview
  helper both the node grid and the node-creator template grid call), `uniform.py`,
  `cheatsheet.py` (the floating bottom-right keyboard-cheatsheet overlay — own top-level window,
  scope-filtered rows, opt-out via `UIAppState.show_cheatsheet`).
- **`popups/`** — popup `draw(app: App)` free functions. Open/closed state on `App` as
  `is_node_creator_open` / `is_settings_open` / `is_emoji_picker_open` / `is_shader_lib_picker_open`
  (helpers `app.open_*()` enforce mutual exclusion; `scripts/smoke.py` asserts ≤1 open).
  `node_creator.py`, `settings.py` (global target FPS + inline-editor visual options + the
  **Integrations** credential blocks), `emoji_data.py` + `emoji_picker.py` (monochrome glyph grid),
  `lib_picker/` (package: `__init__` entry+orchestrator, `tree`, `preview`, `search`, `filtering` —
  the tree+preview shader-library browser with right-click file/dir/function context menus).
- **`ui_primitives.py`** (imgui+theme draw helpers: button
  tiers + shared draw primitives — `context_menu_style()`, `pill_button`, `preview_cell`, …) /
  **`util.py`** (non-UI helpers: `adjust_size`, `select_next_value`, `get_uniform_hash`, `pfd_block`,
  `open_in_file_manager`, `format_auto_value`, …) / **`constants.py`** / **`notifications.py`** /
  **`watch.py`** (the per-frame mtime watcher: `reload_node_if_changed` / `maybe_rebuild_lib_index`,
  called from `ui.py::update_and_draw`).
- **`templates_descriptions.py`** — `TemplateDescriptionsStore`: a user-edit sidecar
  (`template_descriptions.json` at `app_data_dir()`) overriding a shipped template's description;
  lookup is override-else-shipped, so "reset" = delete the key. Same posture as the shader-lib
  favorites/tags stores. The two-tier pattern lives in `conventions.md ## Design decisions`.
- **`copilot/`** — the in-app coding-copilot agent (feature 020, cross-project + gate-UI + render/publish +
  Telegram-connect/pack + UI/UX-polish + turn-rollback waves DONE). Mirrors `exporters/`: its own
  package + worker thread + queues + a worker→main GL `bridge`. `App` owns a `CopilotSession` handle + drains
  it per frame; the chat is a floating window (`widgets/copilot_chat.py`) launched from the editor bottom bar.
  The seams (`capabilities` Protocol / `llm.api` / `bridge` / queues / `state`), the `agent` loop, the
  `prompt`, the `trace`, and the eager tools (`tools/shader.py` — read_shader / the 3 edit tools /
  set_uniform / create_node / grep / read_lib / delete_node / switch_node; `tools/publish.py` — render_image / render_video
  / publish_telegram / publish_youtube; `tools/telegram.py` — set_telegram_token / telegram_connect + pack
  CRUD list/select/create/delete) are built. The node-id/edit/uniform/delete + render/publish + telegram
  machinery lives in `backend.py` (`CopilotBackend` — feature 023, extracted from `App`; explicit deps +
  injected getters/callbacks, never imports `App`); `App._build_copilot_capabilities` constructs it and
  binds its methods into `CopilotCapabilities`. `gate.py`
  (`GateChannel`) is the worker→UI blocking round-trip (the bridge's mirror) — BOTH gate kinds wired: CONFIRM
  Yes/No (`delete_node` + publish + pack mutations) and CREDENTIAL, a masked secret input
  (`set_telegram_token`, the token redacted to a prefix everywhere but the live store —
  `19_credential_pack_tools.md`). `checkpoint.py` (`TurnCheckpoint`/`CheckpointStore`) backs per-turn
  rollback (feature 030): a best-effort capture at each mutation seam in `backend.py` (serialize the
  LIVE node, never the on-disk dir) + `App.restore_checkpoint` reload-and-replace; the Revert glyph +
  modal live in `copilot_chat.py`. Full design: `ai_docs/features/020_copilot_agent/`
  (`30_turn_rollback.md` is the latest; `10_skeleton_plan.md` for the original structure).
- **`scripts/smoke.py`** — headless smoke test (see `## Recipes > make smoke`). Not part of
  `shaderbox/` proper; one-off script that imports `App` + `update_and_draw` and runs frames in
  an invisible glfw window via `App(headless=True)` (which sets the `VISIBLE=FALSE` window hint —
  WITHOUT it `App` creates a visible MAXIMIZED window that pops up + hangs the loop on a real
  display). Any offscreen driver should use `App(headless=True)`, not a hand-rolled hidden window.
- **Node-dir data format:** a project lives in `<project>/nodes/<uuid>/{node.json, shader.frag.glsl,
  media/, textures/}` + `<project>/app_state.json`. The active-project pointer is
  `~/.local/share/shaderbox/project_dir`; templates ship under `shaderbox/resources/node_templates/`.
  Exporter render-output scratch files live in `<project>/exporter_scratch/` (cleaned per export).

### Run the app
- **Dev / personal:** `make run` (= `uv run python ./shaderbox/ui.py`). For an agent smoke-launch,
  use `timeout 12 uv run python ./shaderbox/ui.py` (exits 124 on the timeout = ran clean). Run it as
  its **own** command — don't prefix `pkill ... ;`: a no-match `pkill` exits non-zero and aborts the
  chain (the harness reports it as exit 144). If you must kill a prior run, do it in a separate call
  and ignore its exit code.
- **Verify the built bundle as a NEW user:** `make run-bundle` — rebuilds (`--allow-dirty`, so it
  tests current source incl. uncommitted work), unzips, runs the launcher with a throwaway
  `SHADERBOX_DATA_DIR`, so it's a true fresh first-run that never touches your real projects.

State lives in `app_data_dir()` (default `~/.local/share/shaderbox/`, overridable via
`SHADERBOX_DATA_DIR` — see `conventions.md ## Design decisions`) + the active project's files. The
repo's `projects/dev/` is the maintainer's dev sandbox (**tracked** — `scripts/smoke.py` needs it as
a fixture). It's where features get tested, so its `app_state.json` / `nodes/*/node.json` drift
between runs every time the app runs. **The rule is binary: the working tree must never sit with
unstaged `projects/dev/` changes.** Either it's gitignored or it's committed — no "leave it
uncommitted, it's fine" middle state (that just re-surfaces as staging noise next time). So:
**always `git add projects/dev && commit` the config/node drift in the same wave as your work**
(it's an authorized sandbox-sync, no review needed — a `chore: sync dev sandbox` commit is fine).
The sticker-thumbnail cache `projects/*/media/` (Telegram `file_id`-named `.webm`/`.webp`) is
**gitignored** — it's regenerable download cache, not fixture data; never commit it.
`imgui.ini` in the repo root is layout cruft (gitignored). Telegram credentials live in the global
`integrations.json` under `app_data_dir()` (outside the repo, never committed) — populated by a real
Connect, not committable fixture state.

**No screenshot-driven loop.** This is a glfw window, not a headless browser — there is no reliable
way to screenshot it from the agent: with no window manager on the dev display the freshly-mapped
window stays buried behind the terminal, and capturing it would mean stealing the full screen, which
disrupts the maintainer. So **don't** spin on `import -window root` / window-raise tricks. Verify
behavior the way that actually works: `make smoke` (headless invariant check — it already drives the
full `update_and_draw` loop) + a focused headless introspection script that constructs a NARROW
object (e.g. a `Node`, call `render()` once to compile lazily, then assert on
`get_active_uniforms()`), and **hand visual confirmation to the maintainer** with a one-line "run
`make run` and check X". State the limitation once and move on.

**Do NOT write throwaway scripts that construct a second `App(...)` to "verify" a UI change.** A full
`App` spins up a glfw window + GL context + fonts + project load + copilot session — slow, and it
hangs / pops a window (it did, repeatedly, across a session). `make smoke` already covers the
`update_and_draw` path; anything genuinely visual (cursor, scroll position, layout geometry) headless
can't judge anyway — hand it to a `make run` pass. If a headless assertion is truly needed, drive a
narrow object or a bare GL+imgui context, never a whole `App`.

### `make check`
The single canonical lint/typecheck command — delegates to `uv run pre-commit run --all-files`:
ruff fix, ruff format, then **pyright** (chosen over mypy on purpose — fewer false positives, less
friction; `[tool.pyright]` in `pyproject.toml`, basic mode). `.pre-commit-config.yaml` is the source
of truth for the config. Run before declaring anything done. Both ruff and pyright **block on
failure** — the repo is currently at 0 pyright errors; keep it that way.

### `make smoke`
Headless smoke test (`scripts/smoke.py`) — runs ~200 frames of `update_and_draw` against
`projects/dev/` in an invisible glfw window, asserts popup-mutex + `current_node_id` invariants.
~1.5s; catches import errors, callback dispatch failures, popup state-machine crashes, released-
texture binding errors. Doesn't catch visual bugs. Run after any refactor in `ui.py` / `app.py` /
`widgets/` / `popups/` / `tabs/` / `hotkeys.py` before declaring done; **not** wired into
`make check` (needs a real GL context). Save/restore the user's
`~/.local/share/shaderbox/project_dir` pointer is handled inside the script.

### Build / ship to itch.io
**The full ship procedure is the `/ship` skill** (`.claude/skills/ship/SKILL.md`) — the canonical home
for the flow (sanitize → commit/push `dev` → promote to `master` → auto-pick semver → `make release` →
gated `build.sh` → butler upload → page sync → land clean on `dev`). Maintainer-triggered; the agent
never runs it unprompted. The Windows runtime-verify checklist lives in `BUILDING.md`. This section
holds only the two *facts* that constrain code (not procedure): the bundle shape and the clean-bundle
invariant. The bundle is a **source distribution** (ships `shaderbox/` + `uv.lock`; the user's machine
runs `uv sync` + `uv run` via `run.sh`/`run.bat` on first launch) — not a frozen binary.

**Clean-bundle invariant.** The bundle is an explicit allowlist (`shaderbox/` package +
`pyproject.toml` / `uv.lock` / `.python-version` / `LICENSE` + the launcher + `scripts/README.md`).
NO coding-agent or dev-flow files ship — `CLAUDE.md`, `ai_docs/`, `.claude/`, `Makefile`,
`.pre-commit-config.yaml`, bytecode (`__pycache__` / `*.pyc`) are all excluded. `build.sh` strips
bytecode and runs a verification gate that **aborts the build** if any forbidden pattern is found in
the staged tree — so this is asserted, not assumed. When you add a file the app needs at runtime,
add it to the `build.sh` allowlist (`ROOT_FILES` or under `shaderbox/`); when you add a new dev
artifact, confirm it matches a `FORBIDDEN_*` pattern or the gate won't catch a future leak.

### Sync the itch.io page
**Source of truth:** `ai_docs/itch/page.yaml` (title / tagline / description / tags + the declared
AI-disclosure mirror). The repo `README.md` is GitHub-only; the store page is NOT a copy of it.
itch has **no write API** (server-side API is read-only; butler pushes builds only) — so the page is
edited by an **agent-driven Playwright session**, not a script. Done last in the ship flow (after
`upload-itch.sh`), so the page describes what's already downloadable.

Procedure (the agent runs it; the maintainer can't headless it):
1. Open `https://itch.io/game/edit/3722606` (login persists in the Playwright profile).
2. Diff `page.yaml` against the live fields; edit **only** what changed.
3. **Stop before Save** — show the staged form, the maintainer reviews, then Save (assisted, never
   auto-submit: publishing is outward-facing + hard to reverse).
4. Verify on the public page (`…/shaderbox`), not the editor.

Footguns (each cost a real failure this session):
- **Description must be written into the backing `<textarea name="game[description]">`, NOT the
  visible `.redactor-in` contenteditable** — the redactor does not sync editable→textarea on a
  scripted edit, so Save silently submits the OLD body. Use the HTML view or set the textarea value
  directly, then Save.
- **Tags are a Selectize widget** with a `create` (slugify) function — add via the instance API
  (`addOption`+`addItem` / `createItem`), not simulated typing (keystroke filtering scrambles input,
  e.g. `glsl`→`sllg`). Custom tags are allowed but get less discovery than canonical itch tags.
- **Snapshot refs go stale** after any edit that re-renders — re-snapshot or use stable selectors.
- Screenshots / cover are binary — uploaded by hand, never from `page.yaml`.
- **Never hand-edit the live page outside `page.yaml`** — the next sync diffs from `page.yaml` and
  silently reverts a manual tweak. `page.yaml` is the only authoring surface; that's what makes the
  single-source-of-truth claim hold.

---

## Documentation discipline

This section is the canonical home for documentation discipline (read it at the moment you author
or edit a doc; `CLAUDE.md`'s "Code rules" only points here). These rules keep the docs
**robust** (failures loud), **smooth** (no re-deriving rules), **cold-reloadable** (a fresh agent
finds "what's next?" in a few reads), **unbiased** (resists sympathetic reading). **The leftovers
train the next entry** — agents pattern-match on what's already in a file far harder than on a rule
in another doc, so the shape-pin comments at the top of each artifact are load-bearing; match them,
don't strip them.

### One canonical home per concept
A rule lives in exactly one file; everywhere else points at it. Restating the same rule in 3-5
places guarantees drift. The **false-inheritance hybrid** — "inherits from X" then paraphrases and
silently drops a clause — is banned: either point-only, or (rarely) copy literally + greppably.

### Roadmap rows index; feature specs narrate
Each row is one markdown table line (shape pinned at the top of `roadmap.md`). The landed-reality —
file paths touched, line counts, the bug-fix story, the review-round trail — belongs in
`ai_docs/features/NNN_*.md` or the commit message, not the row. If a row would spawn a second
descriptive sentence, expand the spec instead. (This is what replaced the old narrative `worklog.md`:
the narrative was useless noise; the durable facts are the spec + the row + the banner.)

### The Active-context banner gets rewritten, NOT appended
≤200 words. Date stamp = last-edit of the block, not the date of the work it summarises. "Banner
history kept for traceability" is the append-rot anti-pattern — git log is traceability. Banned
phrases inside the banner: "carry-over from earlier banner", "previous-action (archived…)".

### `todo.md` is a grep-by-trigger index
Each entry names a concrete observable trigger (a file/code touch, a count threshold, a user
complaint with a measurable surface, a specific upstream change). Designs don't go here — they go in
the feature spec or `conventions.md`. If N entries fire on the same trigger, they're ONE rolling
entry, not N.

### Code comments state the now, not the history
A comment names what's non-obvious about the code as it currently is — a GL invariant, an
upstream-bug workaround, an ordering constraint — in as few lines as possible. It does NOT narrate
what bug we hit, why we changed it, or recount the development saga; that rationale lives in its
canonical home (`conventions.md ## Known quirks` / `todo.md`) and the comment shrinks to a ≤1-line
pointer or vanishes. The full rule is `conventions.md ## Code rules`; review + `/sanitize` enforce it.

### Resolved entries get deleted in the resolving commit
Git history is authoritative. Banned: `[RESOLVED]` headers, `~~strikethrough~~` retentions,
`<details>`-collapsed "kept for posterity" blocks. They rot the live doc to memorialise work the
diff already records.

### Cite by section name, not line number
Line numbers drift on every edit; cross-doc step ordinals diverge. Reference a section header or a
quoted phrase — `conventions.md ## Known quirks` is greppable, "line 47" is not.

### When you feel the pressure to violate one of these
That's the signal the rule applies. The urge to cram one more useful fact into a row, append "for
traceability" to a banner, or keep the comment-saga "because it's helpful" — is exactly the pressure
these rules exist to resist.

---

## Maintainer habits

Why the docs are shaped this way. Short list, kept honest:

- **Re-read, don't recall.** Open the file before citing it — compaction flattens the qualifier that
  mattered. Seconds to re-read, hours to undo a wrong decision built on a misremembered rule.
- **Sessions are disposable; knowledge is durable.** If a decision only lives in this conversation,
  it's lost on `/clear` — push it into a file (roadmap row/banner, todo deferral, `conventions.md ##
  Design decisions`, a commit message). The cold-context check (`/sanitize`'s cold-context step) is the gate.
- **Docs are living.** User drops a fact that makes a doc stale → update the right file in the same
  wave, don't keep it in chat. Small → do it now and mention it; substantial → confirm first.
- **Don't trust, verify.** Before "done": run `make check`, check `git status`, read the diff,
  confirm the behavior in the actual app (run it).
- **Only change what's asked; show options for trade-offs.** A bug fix doesn't need surrounding
  cleanup. When the user has a decision, show 2-4 options with short snippets — prose is too abstract
  for trade-offs. When the user says "fix everything", show the list first.
- **Robust defaults on blocking open questions.** No shim, no "we'll fix it later" — default to the
  robust answer; only shortcut if the robust path is hugely out of scope, and then surface the
  trade-off so the user can choose.
- **Audit before "done" on big sweeps.** A substantial refactor, a docs-harness sweep, a
  multi-file feature → don't self-certify with one read. Spawn an adversarial review swarm anchored
  to a checklist (`/review-agent-loop`), converge, THEN declare done. A sympathetic single pass is
  how a "done" stamp lands on incomplete work.
