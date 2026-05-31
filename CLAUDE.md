# ShaderBox

A real-time GLSL fragment-shader playground. Stack: **moderngl + glfw + imgui-bundle** (Python 3.12).
You write a `.frag.glsl`; ShaderBox introspects the program's active uniforms and auto-generates
imgui controls for them; hot-reloads on file mtime; exports to image/video; has a custom freetype
glyph-atlas text-rendering shader; uploads to Telegram sticker sets. Ships on itch.io. Solo project.

## Cold start chain

For ANY work, follow this chain in order. Don't read sideways, don't pre-read auxiliary files.

1. **This file.** Stack, hard rules, navigation, skills.
2. **`ai_docs/roadmap.md`** — what's built and what's next (the Active-context banner is "what's next?").
3. **`ai_docs/todo.md`** — known blockers/deferrals, each with a **Trigger**. Grep by `Trigger` before working in an area.
4. **`ai_docs/dev_flow.md`** — the single source of truth for HOW work happens (feature flow, recipes, doc discipline, maintainer habits).

Quick routing (full version → `dev_flow.md`):
- **small / mechanical change** → just do it + `make check` (+ `make smoke` if it touched UI/lifecycle code — roster in `dev_flow.md ### make smoke`).
- **feature** → the feature flow in `dev_flow.md`.
- **research / brainstorm** → chat report; don't half-start an implementation.

## Hard rules

- **Commits.** Per global `## Commits`, plus: ASCII-only subject; commit/push only when asked (standing instructions stand until revoked). No review-status tags in the message (no `UNREVIEWED`, `WIP`, etc.) — review state is tracked out-of-band, not in git history.
- **Commit on `dev`, never `master`** — no per-feature branches. The dev→master ship promotion lives in `dev_flow.md ## Branch model`.
- **Never leave `projects/dev/` unstaged** — sandbox drift gets `git add projects/dev && commit`ed in the same wave; never `checkout`-ed away. Stripped from shipped bundle by `build.sh`. (Full rule: `dev_flow.md`.)
- **`uv`, not `python`/`pip`** — `uv sync`, `uv add <pkg>`, `uv add --group dev <pkg>`, `uv run …`.
- **Run `make check` before declaring anything done** — ruff + pyright, both block on failure.
- **Don't sidestep a convention.** A convention collision means the design is wrong — fix the design, don't `# noqa` / `# type: ignore` / inline-import past it. Sanctioned suppression allowlist (upstream stub gaps only) is in `conventions.md ## Known quirks`.

## Code rules (the ones that fire on every edit)

- Full type annotations on all params and variables. No `from __future__ import annotations`.
- Imports at module top only — never inside function bodies.
- Comments state what's non-obvious about the code AS IT IS NOW — never narrate development history. The "bug we hit / why we changed it" story belongs in `conventions.md ## Known quirks` / commit message / spec.
- No `@staticmethod` / `@classmethod` (except genuine alternate constructors). A method that doesn't use `self` is a free function.
- No `if TYPE_CHECKING:` — a circular import is a design bug.
- All UI work flows through `ui_primitives.py` (button tiers + shared draw helpers) and `theme.py` (colour/size/spacing tokens). Never hand-roll `push_style_color(Col_.button, …)` at a call site.
- Library docs + source are the source of truth — verify against the installed package (`.venv/lib/python*/site-packages/<lib>`) or upstream repo, never guess from training data.

Full design decisions, library quirks, and the sanctioned type-ignore allowlist → `ai_docs/conventions.md`. Read it at spec-drafting, impl, and review time per `dev_flow.md`.

## Skills

- `/sanitize` — closing-out sweep (run before "done").
- `/imgui-ui` — read at the START of ANY UI work. Single source of truth for imgui patterns (button tiers, jitter, modals, context menus, imgui-bundle quirks).
- `/ship` — release-to-itch.io procedure. Maintainer-triggered only.

## Two meta-rules

(Full text in `dev_flow.md ## Maintainer habits`.)

- **Sessions are disposable; knowledge is durable.** If a decision lives only in chat it's lost on `/clear` — file it. The cold-context check (`/sanitize`'s cold-context step) is the gate.
- **Docs are living.** A fact that makes a doc stale → update the right file in the same wave; small fix do-it-now, substantial fix confirm first.

## Reply language

Reply in the language of the user's latest message (per global `~/.claude/CLAUDE.md`).
