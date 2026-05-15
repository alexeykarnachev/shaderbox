# Dev flow

How work happens in this repo. **Reference, not a mandate** — solo project, no enforced workflow;
pick the loosest shape that proves the change. Re-open this file between steps: the known failure
mode is pattern-matching a step name to a shorter remembered checklist and silently skipping the
passes that weren't cached.

The cold-start chain: `CLAUDE.md` → `ai_docs/worklog.md` (where work left off + the resumption
backlog) + `ai_docs/todo.md` (landmines/triggers) → this file.

---

## Three kinds of work — pick the shape

| Trigger | Shape |
|---|---|
| **Small / mechanical change** — bug fix, log line, kwarg, trivial type fix, doc tweak, dep bump, dead-code deletion that touches no public surface, a `.gitignore` edit | Just do it. `make check`. Optionally `/simplify` on the diff. No spec, no review ceremony; worklog entry only if it's worth a fresh agent knowing. |
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
  impl time. Feature 002 surfaced this — see its post-impl reversal trail.

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
   re-derives blind). Once 2-3 specs exist, add a small "specs by shape" index to `CLAUDE.md`.

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
   remembered version). Don't sidestep a convention (see the hard rule — imgui `# type: ignore` is
   the one sanctioned exception). **Robust defaults:** when an open question blocks implementation,
   default to the robust answer, not a shim — only shortcut if the robust path is hugely out of
   scope, and then surface the trade-off so the user can choose. After impl: glance the diff for new
   `# pyright: ignore` / `# type: ignore` (outside imgui calls) / `# noqa` / `TODO` / inline imports /
   `Any` on real-typed params — any of those means a shortcut was taken; fix it.

6. **Post-implementation review** (per the preamble — 2-3 in parallel for a mid feature; 0-1 for
   tiny/small; 3+ for high-blast-radius). Roles: *code correctness* (bugs, races, GL-context
   lifecycle, resource leaks, error handling), *architecture & conventions* (module boundaries,
   where things live, duplication, the imgui-interop patterns), and
   for a high-blast-radius diff a *spec-fidelity audit* (walks the spec end-to-end against the diff —
   every locked decision actually landed). Each reviewer runs `git show --stat <impl-commit>` first to
   know the full surface, reads every changed file end-to-end, and states which it skipped and why (a
   finding list without a coverage statement = under-read). Findings → triage → fix inline / file as
   `todo.md` deferral with a trigger / promote new design decisions to `conventions.md ## Design decisions`.

7. **Manual check** (when user-visible — most ShaderBox features). Run the app
   (`uv run python ./shaderbox/ui.py`), exercise the change, screenshot if useful. A real UX gap
   found here is a FAIL, not pass-with-caveat. (See `## Recipes` for what the app needs to run and
   what *can't* be exercised without secrets — e.g. the Telegram share tab.)

8. **Sanitization sweep** — the `/sanitize` skill. Separate commit (or its own line in the commit
   message).

9. **Done** — append a `worklog.md` entry; move open items to `todo.md` with concrete triggers; if
   the feature cleared something off the resumption backlog or reshuffled the order, update the
   worklog top entry's `open thread:` line.

---

## Mid-flight escalation

Three rules (the one whose absence produces the worst outcome — an agent quietly turning a small fix
into a refactor):

1. **Scope grew** → halt, propose to the user, don't quietly grow. Say exactly what's bigger and ask
   whether to extend the spec or file a separate feature.
2. **A reviewer surfaces "shouldn't land in current form"** → escalate to the user. Don't contort the
   spec to satisfy the letter of conventions while violating the spirit.
3. **A blocking open question** → robust default (see step 5), or escalate the trade-off if the
   robust path is hugely out of scope.

---

## Closing out work

Whatever the shape — before "done", run `/sanitize`. For a truly trivial mechanical change, the
worklog entry (if even that) + the cold-context glance is enough; for a feature, the full pass.

---

## Recipes

### Module map
(Flat `shaderbox/` package + two subpackages — this is the orientation `arch.md` would have been:)

- **`core.py`** — `Canvas`, `Node`: GL program lifecycle, uniform introspection + binding,
  render-to-texture, image/video export. Needs a live GL context.
- **`app.py`** — `App` class: state holder + lifecycle (project, GL context, node management,
  popup-state booleans). ~370 lines. No UI drawing. Imported by `ui.py`, `widgets/`, `popups/`,
  `tabs/`.
- **`ui.py`** — thin entrypoint + orchestrator. ~255 lines. Contains `run(app)`,
  `update_and_draw(app)` (the imgui frame loop: render gates, main window + image,
  tab-bar dispatch), `_draw_node_settings(app)` (tab-bar dispatcher), and `main()`. No tab
  bodies, no widget logic, no hotkey dispatch — those live in `tabs/`, `widgets/`, `popups/`,
  `hotkeys.py`.
- **`hotkeys.py`** — `process_hotkeys(app: App)`: glfw poll + imgui input processing + all
  keyboard shortcut dispatch (Ctrl+O/S/E/D/N, Alt+S, Esc, arrows, Enter). Called from
  `update_and_draw` once per frame.
- **`ui_models.py`** — pydantic-ish `UINode` / `UINodeState` / `UIUniform` / `UIAppState` + node
  (de)serialization.
- **`media.py`** — `Image` / `Video` (`MediaWithTexture` ABC), ffmpeg temporal smoothing.
- **`exporters/`** — `Exporter` ABC + `RenderedArtifact` value type (`base.py`), `ExporterRegistry`
  (`registry.py`), `TelegramExporter` (`telegram.py` — own worker thread + asyncio loop + sticker
  panel UI). Adding a new exporter: subclass `Exporter`, register in `App.__init__`. The
  thread-affinity contract (worker thread MUST NOT touch moderngl) is enforced by design.
- **`tabs/`** — tab modules. Each tab is one file with `draw(app: App)` (imgui calls only) and
  optional `update(app: App)` (pre-imgui GL work). Modules: `node.py`, `render.py`, `share.py`.
  `share_state.py` holds the share-tab dataclass (`TabState`) separately to keep `app.py` free
  of cyclic imports (app.py imports `share_state`, NOT `share`).
- **`widgets/`** — stateless imgui-drawing functions taking `app: App`. Shape per widget fits its
  job (no shared contract). Modules: `details.py`, `media_ops.py`, `node_grid.py`, `uniform.py`.
- **`popups/`** — popup `draw(app: App)` free functions. Open/closed state lives on `App` as
  `is_node_creator_open` / `is_settings_open` (helpers `app.open_node_creator()` /
  `app.open_settings()` enforce mutual exclusion). Modules: `node_creator.py`, `settings.py`.
- **`modelbox.py`** — thin HTTP client for the optional external ModelBox service.
- **`fonts.py`** — freetype → GL atlas. **`ui_utils.py`** / **`constants.py`** / **`notifications.py`** — helpers.
- **Node-dir data format:** a project lives in `<project>/nodes/<uuid>/{node.json, shader.frag.glsl,
  media/, textures/}` + `<project>/app_state.json`. The active-project pointer is
  `~/.local/share/shaderbox/project_dir`; templates ship under `shaderbox/resources/node_templates/`.
  Exporter render-output scratch files live in `<project>/exporter_scratch/` (cleaned per export).

### Run the app
`uv run python ./shaderbox/ui.py`. State lives in `~/.local/share/shaderbox/` + the active project's
files. The repo's `projects/dev/` is the maintainer's dev project (tracked, intentional). `imgui.ini`
in the repo root is layout cruft (gitignored). **Can't be exercised unconfigured:** the Telegram
share tab needs a bot token / user id / sticker-set name in app settings; ModelBox is an optional
external HTTP service (the app degrades gracefully without it).

### `make check`
The single canonical lint/typecheck command — delegates to `uv run pre-commit run --all-files`:
ruff fix, ruff format, then **pyright** (chosen over mypy on purpose — fewer false positives, less
friction; `[tool.pyright]` in `pyproject.toml`, basic mode). `.pre-commit-config.yaml` is the source
of truth for the config. Run before declaring anything done. **Pyright is non-blocking for now** —
the repo has pre-existing type debt across `ui.py`, `media.py`, `core.py`, `modelbox.py` (see
`conventions.md ## Known quirks` and `todo.md [DEFERRAL] re-tighten pyright`), so the hook prints
pyright's findings but `|| true`'s past a non-zero exit. Don't add *new* pyright errors in the
meantime.

### Build / ship to itch.io
`./build.sh` → `dist/shaderbox-{windows.zip,linux.tar.gz}` → `./upload-itch.sh` (needs `butler` + an
`itch-config` file). Maintainer-triggered, not the agent's.

---

## Maintainer habits

Why the docs are shaped this way. Short list, kept honest:

- **Re-read, don't recall.** Open the file before citing it — compaction flattens the qualifier that
  mattered. Seconds to re-read, hours to undo a wrong decision built on a misremembered rule.
- **Sessions are disposable; knowledge is durable.** If a decision only lives in this conversation,
  it's lost on `/clear` — push it into a file (worklog entry, todo deferral, `conventions.md ##
  Design decisions`, a commit message). The cold-context check (`/sanitize` step 6) is the gate.
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
