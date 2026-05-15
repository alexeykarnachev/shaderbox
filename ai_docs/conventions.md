# Conventions

Code rules + design decisions + known quirks. Auto-loaded via `@ai_docs/conventions.md` at the bottom
of `CLAUDE.md`. Re-read end-to-end from disk before: spec drafting, spec validation, implementation,
code review, the sanitization sweep.

## Code rules

- Full type annotations on all params and variables.
- Imports at module top only — never inside function bodies.
- Minimal comments — only for non-obvious logic.
- Don't sidestep a convention with `# noqa` / `# pyright: ignore` / `# type: ignore` / inline import /
  circular-import hack — a collision means the design is wrong. The one sanctioned type-suppression
  exception is in `## Known quirks`.
- Type checker: **pyright** (not mypy), basic mode, via `make check`. **Non-blocking for now**
  (pre-existing type debt across the repo — see `## Known quirks`). Findings print but the
  pre-commit hook `|| true`'s past a non-zero exit. Don't add *new* pyright errors.
- `uv` for everything (`uv run`, `uv add`, `uv add --group dev`) — never bare `python` / `pip`.
- Commit messages: short, concise, single-line, ASCII; no footers / co-authored-by.
- Never use "from __future__ import annotations" - this is a noise.

## Design decisions (we decided X; revisit if Y)

- **`ui.py` is a god-class being incrementally extracted.** New UI code goes in the smallest plausible
  *new* module (`widgets.py` / `tabs/*.py` / `hotkeys.py` / `project.py`), not back into `ui.py`.
  Revisit the target structure once 3+ extractions land. First real extraction: `tabs/share.py`
  (feature 001). (Tracked: `todo.md [DEFERRAL] split ui.py`; backlog item 4 in the worklog `open
  thread:`.)
- **`tabs/*.py` pattern: free `draw()` + optional `update()`, module-level `TabState`.** First
  instance: `tabs/share.py`. `draw(state, ...) -> None` does imgui calls only;
  `update(state, ...) -> None` (when present) runs from `App.update_and_draw` *before* imgui
  draws, for canvas ticks / mtime polls / anything that touches GL state. The `TabState`
  dataclass is owned by `App`, instantiated once in `App.__init__`, threaded into both calls.
  Revisit when 3+ tab modules exist and a different shape emerges.
- **No `async` in the codebase except where python-telegram-bot forces it** — and that runs off the
  render thread (per-exporter worker thread + own asyncio loop, see `exporters/telegram.py`),
  never via `run_until_complete` inside the imgui frame. Revisit if a future exporter brings a
  new async-required dep that doesn't fit the worker-thread + own-loop pattern. (Tracked:
  `todo.md [DEFERRAL] blocking HTTP in render loop` — narrowed to ModelBox after feature 001.)
- **Exporters: own thread, own panel, GL-free artifacts.** The `Exporter` ABC in
  `shaderbox/exporters/base.py` enforces a thread-affinity contract: render-thread methods may
  touch moderngl; worker-thread methods (`prepare`, `export`) MUST NOT — they only see
  `RenderedArtifact` (a pure value type, no GL handles). Each exporter owns its own per-target
  panel UI (no shared "list of remote items" widget). Revisit when a third concrete exporter lands.

*(Grows as features land — each new cross-cutting choice gets a bullet with a revisit trigger. The
sanitization sweep's noise audit deletes bullets that narrate a one-off implementation choice instead
of constraining future code; SDK footguns go to `## Known quirks`, not here.)*

## Known quirks (library / SDK footguns + the workaround)

- **pyimgui's stubs are incomplete** → a scoped `# pyright: ignore[reportAttributeAccessIssue]` (or
  `# type: ignore`) on an `imgui.*` call that pyright can't resolve is sanctioned. NOT a general
  license — type suppression anywhere else means a real error to fix.
- **Pre-existing pyright debt across the repo** — `ui.py`, `media.py`, `core.py`, `modelbox.py`
  haven't been audited for strict-typing compliance. The pre-commit pyright hook is `|| true`'d
  (non-blocking) because of this. Don't add *new* errors. Re-tightening (drop the `|| true`) is
  tracked separately and gated on the cleanup backlog reaching a clean state — see
  `todo.md [DEFERRAL] re-tighten pyright`.
- **A live moderngl context must exist before constructing `Image` / `Video` / `Font` / `Canvas` /
  `Node`** — they call `moderngl.get_context()` lazily. In the app,
  `glfw.make_context_current(window)` handles it.
- **`modelbox.py` imports `Image` / `Video` from `shaderbox.core`** (which re-exports them from
  `media`) — reaching through `core` for `media` types. Minor module-boundary smell; left as-is.
