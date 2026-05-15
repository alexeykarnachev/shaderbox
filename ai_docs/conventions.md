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
- Type checker: **pyright** (not mypy), basic mode, via `make check`. It's **non-blocking for now**
  (pre-existing type debt in `ui.py` — see `## Known quirks`) but its findings print; don't add new
  pyright errors.
- `uv` for everything (`uv run`, `uv add`, `uv add --group dev`) — never bare `python` / `pip`.
- Commit messages: short, concise, single-line, ASCII; no footers / co-authored-by.

## Design decisions (we decided X; revisit if Y)

- **`ui.py` is a god-class being incrementally extracted.** New UI code goes in the smallest plausible
  *new* module (`widgets.py` / `tabs/*.py` / `hotkeys.py` / `project.py`), not back into `ui.py`.
  Revisit the target structure once 3+ extractions land. (Tracked: `todo.md [DEFERRAL] split ui.py`;
  backlog item 6 in the worklog `open thread:`.)
- **No `async` in the codebase except where python-telegram-bot forces it** — and that should run off
  the render thread (a worker thread + result queue), not via `run_until_complete` inside the imgui
  frame. Revisit if the share path is rewritten. (Tracked: `todo.md [DEFERRAL] blocking asyncio…`.)

*(Grows as features land — each new cross-cutting choice gets a bullet with a revisit trigger. The
sanitization sweep's noise audit deletes bullets that narrate a one-off implementation choice instead
of constraining future code; SDK footguns go to `## Known quirks`, not here.)*

## Known quirks (library / SDK footguns + the workaround)

- **pyimgui's stubs are incomplete** → a scoped `# pyright: ignore[reportAttributeAccessIssue]` (or
  `# type: ignore`) on an `imgui.*` call that pyright can't resolve is sanctioned. NOT a general
  license — type suppression anywhere else means a real error to fix.
- **`ui.py` has pre-existing type debt** — the share-tab `hasattr`-dispatch (`draw_share_tab`) calls
  attributes like `_sticker` / `video` / `image` / `preview_canvas` on `ShareableMedia`, which doesn't
  declare them (they live on the concrete sticker classes the code `cast`s to). That's ~16 pyright
  errors. The pre-commit pyright hook is `|| true`'d (non-blocking) because of this; it'll be
  re-tightened (drop the `|| true`) once the sticker-models refactor lands — see
  `todo.md [DEFERRAL] three near-identical sticker models`.
- **A live moderngl context must exist before constructing `Image` / `Video` / `Font` / `Canvas` /
  `Node`** — they call `moderngl.get_context()` lazily. In the app, `glfw.make_context_current(window)`
  handles it; in tests, `moderngl.create_standalone_context()` in a fixture.
- **`modelbox.py` imports `Image` / `Video` from `shaderbox.core`** (which re-exports them from
  `media`) — reaching through `core` for `media` types. Minor module-boundary smell; left as-is.
