# Conventions

Code rules + design decisions + known quirks. Auto-loaded via `@ai_docs/conventions.md` at the bottom
of `CLAUDE.md`. Re-read end-to-end from disk before: spec drafting, spec validation, implementation,
code review, the sanitization sweep.

This file holds **generic, future-constraining rules** — how code, architecture, and design must be
shaped. It is NOT a feature changelog: per-feature mechanics, instance lists, and the story of how a
decision was reached belong in the feature spec (`ai_docs/features/NNN_*.md`); library/SDK footguns
belong in `## Known quirks`. If a bullet doesn't constrain code you haven't written yet, it's noise.

## Code rules

- Full type annotations on all params and variables.
- Imports at module top only — never inside function bodies.
- **Comments state what's non-obvious about the code as it is NOW — never narrate development
  history.** Banned: the bug-we-hit story, the why-we-changed-it backstory, the "see <doc> for the
  full saga", paragraph-length rationale. If a line needs a comment, it's one terse line naming the
  non-obvious fact (a GL invariant, an upstream-bug workaround, an ordering constraint) — and if the
  rationale is already captured in its canonical home (`## Known quirks` / `todo.md`), the comment
  shrinks to a ≤1-line pointer or disappears. The section-banner `# ----` separators are fine; the
  multi-line "here's what happened during development" blocks are the target. (Enforced in review +
  `/sanitize`.)
- **Stale counts rot faster than stale prose** — they look authoritative. When you change a number
  the code reflects (uniform-type count, line counts, "N tabs"), update every doc that quotes it in
  the same commit, or drop the count from the doc.
- Don't sidestep a convention with `# noqa` / `# pyright: ignore` / `# type: ignore` / inline import /
  circular-import hack — a collision means the design is wrong. The sanctioned type-suppression
  allowlist (upstream library-stub gaps only) is in `## Known quirks`.
- Never use `if TYPE_CHECKING:` to work-around circular imports. Circular imports is a sign of a bad design.
- Type checker: **pyright** (not mypy), basic mode, via `make check` — blocks on failure.
  Repo is at 0 errors; keep it that way.
- `uv` for everything (`uv run`, `uv add`, `uv add --group dev`) — never bare `python` / `pip`.
- Commit messages: short, concise, single-line, ASCII; no footers / co-authored-by.
- Never use "from __future__ import annotations" - this is a noise.

## Design decisions (we decided X; revisit if Y)

- **Three-layer UI architecture.** `app.py` = state holder + lifecycle (project, GL context, node
  management, popup booleans) — no imgui drawing. `ui.py` = thin orchestrator owning the frame loop
  (`run` + `update_and_draw`). `widgets`/`popups`/`tabs` = pure draw functions taking `app: App`.
  (The split is forced by the no-`TYPE_CHECKING` rule: a draw fn annotating `app: App` while `App`
  imports it would cycle — so `App` lives in its own module.) Revisit if a 4th UI sub-package is
  needed. Tracked: `todo.md [DEFERRAL] split ui.py`.
- **`tabs/*.py`: free `draw(app: App)` + optional `update(app: App)`.** `draw()` does imgui calls
  only; `update()` runs *before* imgui draws, for GL/canvas/mtime work outside the frame body. Tab
  state goes on `App` directly; a state-only sibling module (e.g. `tabs/share_state.py`) may hold
  its dataclass to keep `app.py` import-cycle-free. Revisit when a 4th tab module exists.
- **`widgets/*.py`: free functions taking `app: App`, no wrapper, no protocol.** Widgets are an
  organizational convention, not a polymorphic contract — no `Widget` ABC, no shared return shape;
  each gets the shape that fits its job. (An `AppContext` wrapper was tried and reverted — passing
  `app` gave every claimed benefit.) Revisit if a polymorphic `list[Widget]` dispatcher materializes.
- **`popups/*.py`: free `draw(app: App)` functions; open/closed state lives on `App` as booleans.**
  Each `app.open_*()` helper sets its own flag True and clears ALL siblings — the "at most one popup
  open" invariant; keep it when adding a popup. `app.any_popup_open()` is the render-gate question.
  No popup classes. Revisit if a popup grows internal state that doesn't belong on `App`.
- **All UI colors / sizes / spacing flow through `theme.py`'s `COLOR` / `SIZE` / `SPACE` bags** — no
  hardcoded hex or magic pixel values in code. `apply_theme(style, …)` writes the theme into the
  imgui style at boot and is re-callable at runtime. Revisit if a 4th token bag emerges (e.g.
  animation timing) — extend `theme.py`, don't carve a parallel module.
- **Inline editor state lives on `App`; disk is the source of truth.** One `TextEditor` per node
  (+ a parallel dirty-baseline dict), created lazily; `app.save()` flushes the dirty editor before
  writing the file; the mtime watcher re-syncs from disk on external change (disk wins). Editor
  per-instance footguns (palette, FPE-while-modal, cursor, font sizing) live in `## Known quirks`.
  Revisit if multi-file-per-node editing lands.
- **No `async` except where python-telegram-bot forces it** — and that runs off the render thread
  (worker thread + own asyncio loop), never `run_until_complete` inside the imgui frame. Revisit if
  a new async-required dep doesn't fit the worker-thread + own-loop pattern.
- **Exporters: own thread, own panel, GL-free artifacts.** The `Exporter` ABC enforces thread
  affinity — render-thread methods may touch moderngl; worker-thread methods (`prepare`, `export`)
  MUST NOT, they see only `RenderedArtifact` (a pure value type). Each exporter owns its own panel
  UI. Revisit when a third concrete exporter lands.
- **All on-disk state roots at `app.py::app_data_dir()`** — projects, the active-project pointer,
  logs. Never call `platformdirs.user_data_dir(...)` directly (that path silently ignores the
  override); go through `app_data_dir()`, which honors `SHADERBOX_DATA_DIR` (cross-platform; used by
  `make run-bundle` for a throwaway fresh-install run). Revisit if a state root needs to diverge from
  this single base.
- **Release versioning is manual semver, bumped only via `make release VERSION=x.y.z`.** Bump
  rule: **major** = breaks users' existing projects / saved state (e.g. a non-round-trip app_state
  migration); **minor** = a backward-compatible feature; **patch** = fixes / other non-breaking
  changes. The version lives in `pyproject.toml`; not auto-derived from git. Bumped once at ship
  time, not per-commit (ship flow: `dev_flow.md`). Revisit if manual bumps are repeatedly forgotten
  before a release (then consider `git describe`-derived versions).

*(Each bullet is a generic constraint on future code + a revisit trigger — NOT a feature changelog.
The `/sanitize` noise audit deletes bullets that narrate a one-off implementation choice; per-feature
mechanics live in the feature spec, SDK footguns in `## Known quirks`.)*

## Known quirks (library / SDK footguns + the workaround)

- **imgui-bundle's C++-backed submodules ship only `.pyi` stubs** (`portable_file_dialogs`,
  `imgui_color_text_edit`) — pyright emits a `reportMissingModuleSource` warning at the import
  line. The warning is harmless (no `.py` source to find). Warnings don't fail `make check`;
  ignore them. Don't suppress with `# pyright: ignore` — that hides genuine resolve failures
  elsewhere.
- **`imgui_color_text_edit.TextEditor.Palette` is read-only from Python** — only `.get(color)
  -> ImU32`; no per-slot setter, no list-based constructor, and `set_palette()` accepts only a
  `Palette` object (unbuildable with custom colors). Use a built-in (`get_dark_palette()` /
  `get_light_palette()`); a custom gruvbox palette is impossible until imgui-bundle exposes a
  write path (feature 006; tracked in `todo.md`).
- **Dear ImGui 1.92 dropped pre-baked glyph ranges + `refresh_font_texture()`** in favor of
  dynamic on-demand glyph loading (`BackendFlags_.renderer_has_textures`, set automatically by
  imgui-bundle's `BaseOpenGLRenderer.__init__`). `add_font_from_file_ttf(path, size_pixels=N)`
  is enough — no `glyph_ranges=` kwarg, no manual texture refresh. Cyrillic glyphs load when
  text is first drawn.
- **`imgui.push_font` now requires `(font, size_base_unscaled)`** — pass the rasterized size
  (the one used in `add_font_from_file_ttf(size_pixels=...)`). Never pass `imgui.get_font_size()`
  — that's the *post-scaling* value and would scale twice.
- **`imgui.image(...)` lost `tint_col` / `border_col` since 1.91.9**. For tint, switch to
  `imgui.image_with_bg(...)`. For border, push the `ImageBorderSize` style var (or live without).
- **`pfd.open_file` / `save_file` / `select_folder` are non-blocking class handles, not blocking
  functions.** Use the `pfd_block(dialog)` helper in `ui_utils.py` to spin until `.ready(20)` —
  that mirrors crossfiledialog's synchronous shape at the call site.
- **imgui-bundle's Python glfw backend (`python_backends/glfw_backend.py`) does NOT sync imgui's
  mouse cursor to the OS** — it never sets `BackendFlags_.has_mouse_cursors` and never calls
  `glfwSetCursor`. So `imgui.set_mouse_cursor(...)` is a silent no-op at the OS level. To change
  the actual cursor, create glfw cursors (`glfw.create_standard_cursor(...)`, stored on `App`:
  `app.ibeam_cursor`, `app.resize_ew_cursor`) and call `glfw.set_cursor(window, cursor_or_None)`
  yourself. Used by the Code editor (I-beam on hover, `tabs/code.py`) and the split divider
  (resize-EW on hover/drag, `ui.py::_draw_splitter`); restore with `glfw.set_cursor(window,
  None)`. NEVER use `imgui.set_mouse_cursor` for these — it does nothing in this backend.
- **A live moderngl context must exist before constructing `Image` / `Video` / `Font` / `Canvas` /
  `Node`** — they call `moderngl.get_context()` lazily. In the app,
  `glfw.make_context_current(window)` handles it.
- **`imgui_color_text_edit.TextEditor.render()` auto-grabs imgui keyboard focus on a child window's
  first frame** — so a never-yet-rendered editor (app open, or just-switched node) steals focus and
  the caret goes live without a click. The editor exposes no `is_focused()` getter. Track focus by
  reading `imgui.is_window_focused(FocusedFlags_.child_windows)` *after* `render()` (`tabs/code.py`),
  not a hand-maintained click flag. To defocus (Esc, arrow-nav, startup), set
  `app.editor_defocus_requested` and consume it with `set_window_focus(None)` AFTER `render()` —
  clearing before render is undone by the editor's own first-render grab. `hotkeys.py` gates arrow
  node-nav on `app.editor_focused`.
- **The sanctioned `# type: ignore` allowlist (upstream stub gaps only).** The no-suppression rule
  has exactly these exceptions — all are missing/wrong annotations in third-party stubs, never our
  own type errors. New markers outside this list are a design smell; fix the design, don't add to
  the list. Re-audit when bumping `moderngl` / `freetype-py` / `pydantic`.
  - `moderngl.Uniform.gl_type` — not in moderngl's stub (3 sites: `ui_models.py`, `ui_utils.py`,
    `tabs/node.py`).
  - `freetype.load_char(...)` — `freetype-py` ships no stubs (2 sites in `fonts.py`).
  - `@model_validator(mode="after")` on a method returning `Self` — pydantic's decorator stub
    mistypes the wrapped method (`ui_models.py`).
