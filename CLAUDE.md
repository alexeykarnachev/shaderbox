# ShaderBox

A real-time GLSL fragment-shader playground. Stack: **moderngl + glfw + pyimgui** (Python 3.12).
You write a `.frag.glsl`; ShaderBox introspects the program's active uniforms and auto-generates
imgui controls for them; hot-reloads on file mtime; exports to image/video; has a custom freetype
glyph-atlas text-rendering shader; uploads to Telegram sticker sets. Ships on itch.io. Solo project.

## Cold start chain

For ANY work, follow this chain in order. Don't read sideways, don't pre-read auxiliary files.

1. **This file.** Stack, hard rules, navigation, output style.
2. **`ai_docs/worklog.md`** — where work left off, key decisions, what's next (the top entry's
   `open thread:` line is the resumption backlog). Read first on "what's here / what's next". If the
   top entry looks much older than `git log -1`, the worklog wasn't kept current — reconstruct recent
   state from `git log` first.
3. **`ai_docs/todo.md`** — known blockers/deferrals, each with a **Trigger**. Grep by `Trigger`
   before working in an area. (Header explains `[BLOCKER]` vs `[DEFERRAL]` + what makes a good trigger.)
4. **`ai_docs/dev_flow.md`** — the single source of truth for HOW work happens.

Quick routing (full version + the feature flow → `dev_flow.md`):
- **small / mechanical change** → just do it + `make check`
- **feature** (new module, real behavior change, refactor with blast radius) → the feature flow
- **research / brainstorm** → deliver a chat report (+ `ai_docs/<topic>.md` if worth keeping); don't
  half-start an implementation

All other files (`features/*`, …) are referenced from `dev_flow.md` and `worklog.md` at the step
where they're needed. `conventions.md` is auto-loaded below.

## Hard rules

- **Commits.** Commit/push when the user asks for it — including standing instructions ("commit and
  push automatically this session"); a once-given instruction stands until revoked. **Commit messages
  must be short, concise, single-line, ASCII** — no footers, no co-authored-by. **Work on the current
  branch** — don't create per-feature branches (a feature is a sequence of commits, not a branch);
  branch only if the user explicitly asks. (This overrides the generic "branch before committing on
  the default branch" agent default.)
- **`uv`, not `python`/`pip`:** `uv sync`, `uv add <pkg>`, `uv add --group dev <pkg>` (this repo uses
  `[dependency-groups] dev`), `uv run …`.
- **Run `make check` before declaring anything done** — ruff fix+format → pyright (delegates to
  `pre-commit run --all-files`). Both block the commit on failure. (`dev_flow.md ## Recipes` has
  the details.)
- **Don't sidestep a convention.** A convention collision means the design is wrong — fix the design,
  don't `# noqa` / `# type: ignore` / inline-import past it. *The one sanctioned exception:* `# type:
  ignore` scoped to `imgui.*` calls (pyimgui's stubs genuinely lack symbols) — see
  `conventions.md ## Known quirks`. Never use it to paper over a real type error in your own code.
- **Code conventions** (full list → `conventions.md`): full type annotations on params/vars; minimal
  comments (only non-obvious logic); imports at module top only.

## Two meta-rules

(One-sentence summaries; full text → `dev_flow.md ## Maintainer habits`.)

- **Sessions are disposable; knowledge is durable.** If a decision lives only in this chat it's lost
  on `/clear` — file it (worklog entry / todo deferral / `conventions.md ## Design decisions` /
  commit message). The cold-context resumability check (`/sanitize` step 6) gates suggesting `/clear`:
  simulate a fresh agent asking "what's next?", walk the chain, confirm it lands on the same
  next-action with the same blockers in a few reads — if not, the missing context IS the bug; write
  it to a file, not the chat.
- **Docs are living.** User drops a fact that makes a doc stale → update the right file in the same
  wave, don't keep it in chat. Small → do it now and mention it; substantial → confirm first.

## Output style

- **Plan messages:** decisions + open questions only. Skip rationale unless asked. 5-15 lines per
  option when comparing trade-offs.
- **Status during work:** one sentence per turn. State the action, run tools, state the result. No
  running commentary.
- **End-of-turn:** one or two sentences — what changed and what's next.
- **Audits / triage / findings:** tables. Each row 5-15 words. No prose between rows.
- **Code-review feedback in a session:** file:line + severity + what to change. Skip the explanation
  unless severity is critical.
- **Don't restate.** Don't echo the user's question, the file path you just read, or the plan after
  every step.
- **Cut hedging.** "I think we should…" → "Let's…". "It might be worth…" → either do it or don't
  propose it.
- **Editor's eye:** before sending, "would the user skip past any paragraph here?" If yes, delete it.

## Skills

- `/sanitize` — the closing-out sweep (run before "done": walk `todo.md`, fix stale refs/docs,
  append a `worklog.md` entry, cold-context check).

## Reply language

Reply in the language of the user's latest message (per global `~/.claude/CLAUDE.md`).

@ai_docs/conventions.md
