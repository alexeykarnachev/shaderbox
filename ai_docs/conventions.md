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
- Never use `if TYPE_CHECKING:` to work-around circular imports. Circular imports is a sign of a bad design.
- Type checker: **pyright** (not mypy), basic mode, via `make check` — blocks on failure.
  Repo is at 0 errors; keep it that way.
- `uv` for everything (`uv run`, `uv add`, `uv add --group dev`) — never bare `python` / `pip`.
- Commit messages: short, concise, single-line, ASCII; no footers / co-authored-by.
- Never use "from __future__ import annotations" - this is a noise.

## Design decisions (we decided X; revisit if Y)

- **Three-layer architecture: `app.py` = state, `ui.py` = orchestrator, `widgets`/`popups`/`tabs` = pure logic.**
  `App` (in `shaderbox/app.py`) holds project state, GL context, and lifecycle methods — no UI
  drawing. `ui.py` is a thin entrypoint owning the imgui frame loop (`run` + `update_and_draw`).
  Widget/popup/tab modules contain stateless drawing functions that take `app: App` and read or
  call through it. The split was forced by a circular-import surfaced when widgets needed to
  type-annotate `app: App` while `App` imported widgets — addressed by extracting `App` into its
  own module so widgets can import it without cycle. Revisit if a 4th UI sub-package feels
  needed. (Tracked: `todo.md [DEFERRAL] split ui.py`.)
- **`tabs/*.py` pattern: free `draw(app: App)` + optional `update(app: App)`.** Each tab is one
  file with a public `draw()` that does imgui calls only and an optional `update()` that runs
  from `update_and_draw` *before* imgui draws — for canvas ticks / mtime polls / anything that
  touches GL state outside the frame body. Instances: `tabs/share.py` (feature 001),
  `tabs/node.py` + `tabs/render.py` (feature 002 + harmonized `share.py`). If a tab needs persistent state, expose it on `App`
  directly (e.g. `app.share_tab_state`); a state-only sibling module (`tabs/share_state.py`)
  may hold its dataclass to keep `app.py` free of cyclic UI imports. Revisit when a 4th tab
  module exists.
- **Widgets in `widgets/*.py` take `app: App`; no wrapper, no protocol.** Each widget is a free
  function. Widgets are an organizational convention, not a polymorphic contract — there is no
  `Widget` ABC, no shared return shape. Each widget gets the shape that fits its job
  (value-object editor returns updated model, value transformer returns value, action-firer
  returns `None`). Decision rationale: an `AppContext` dataclass was tried briefly but every
  claimed benefit (bounded coupling, test surface, callback decoupling) was either illusory or
  equally provided by passing `app`. Revisit if a polymorphic widget consumer ever materializes
  (a `list[Widget]` dispatcher) — that's when an interface earns its keep.
- **Popups in `popups/*.py` are free functions; their open/closed state lives on `App`.** Each
  popup module has one public `draw(app: App)` function that early-returns when the popup is
  closed and renders the modal body when open. Open/closed state is two plain booleans on
  `App` (`is_node_creator_open`, `is_settings_open`); helpers `app.open_node_creator()` /
  `app.open_settings()` set one True and clear the other to preserve the "at most one popup
  open" invariant. `app.any_popup_open()` answers the render-gate question. No popup classes —
  state belongs to the runtime (`App`), not the popup module. Revisit if a popup grows
  widget-internal state that doesn't belong on `App`.
- **No `async` in the codebase except where python-telegram-bot forces it** — and that runs off the
  render thread (per-exporter worker thread + own asyncio loop, see `exporters/telegram.py`),
  never via `run_until_complete` inside the imgui frame. Revisit if a future exporter brings a
  new async-required dep that doesn't fit the worker-thread + own-loop pattern.
- **Exporters: own thread, own panel, GL-free artifacts.** The `Exporter` ABC in
  `shaderbox/exporters/base.py` enforces a thread-affinity contract: render-thread methods may
  touch moderngl; worker-thread methods (`prepare`, `export`) MUST NOT — they only see
  `RenderedArtifact` (a pure value type, no GL handles). Each exporter owns its own per-target
  panel UI (no shared "list of remote items" widget). Revisit when a third concrete exporter lands.

*(Grows as features land — each new cross-cutting choice gets a bullet with a revisit trigger. The
sanitization sweep's noise audit deletes bullets that narrate a one-off implementation choice instead
of constraining future code; SDK footguns go to `## Known quirks`, not here.)*

## Known quirks (library / SDK footguns + the workaround)

- **imgui-bundle's `portable_file_dialogs` ships only `.pyi` stubs** — pyright emits a
  `reportMissingModuleSource` warning at the import line. The warning is harmless (the package is
  C++-backed; there's no `.py` source to find). Warnings don't fail `make check`; ignore them.
  Don't suppress with `# pyright: ignore` — that hides genuine resolve failures elsewhere.
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
- **A live moderngl context must exist before constructing `Image` / `Video` / `Font` / `Canvas` /
  `Node`** — they call `moderngl.get_context()` lazily. In the app,
  `glfw.make_context_current(window)` handles it.
