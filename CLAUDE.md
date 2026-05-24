# ShaderBox

A real-time GLSL fragment-shader playground. Stack: **moderngl + glfw + imgui-bundle** (Python 3.12).
You write a `.frag.glsl`; ShaderBox introspects the program's active uniforms and auto-generates
imgui controls for them; hot-reloads on file mtime; exports to image/video; has a custom freetype
glyph-atlas text-rendering shader; uploads to Telegram sticker sets. Ships on itch.io. Solo project.

## Cold start chain

For ANY work, follow this chain in order. Don't read sideways, don't pre-read auxiliary files.

1. **This file.** Stack, hard rules, navigation, output style.
2. **`ai_docs/roadmap.md`** — what's built and what's next. The **Active context** banner answers
   "what's here / what's next"; the feature table indexes each feature to its spec. Read first on
   "what's the state here?". (Rows index, specs narrate — see `## Documentation discipline`.)
3. **`ai_docs/todo.md`** — known blockers/deferrals, each with a **Trigger**. Grep by `Trigger`
   before working in an area. (Header explains `[BLOCKER]` vs `[DEFERRAL]` + what makes a good trigger.)
4. **`ai_docs/dev_flow.md`** — the single source of truth for HOW work happens.

Quick routing (full version + the feature flow → `dev_flow.md`):
- **small / mechanical change** → just do it + `make check`
- **feature** (new module, real behavior change, refactor with blast radius) → the feature flow
- **research / brainstorm** → deliver a chat report (+ `ai_docs/<topic>.md` if worth keeping); don't
  half-start an implementation

All other files (`features/*`, …) are referenced from `dev_flow.md` and `roadmap.md` rows at the
step where they're needed. `conventions.md` is auto-loaded below.

## Hard rules

- **Commits.** Commit/push when the user asks for it — including standing instructions ("commit and
  push automatically this session"); a once-given instruction stands until revoked. **Commit messages
  must be short, concise, single-line, ASCII** — no footers, no co-authored-by.
- **Commit on `dev`, never directly on `master`** — no per-feature branches; branch off `dev` only
  if the user explicitly asks. (Overrides the generic "branch before committing on the default
  branch" agent default.) The why + the dev→master ship promotion live in `dev_flow.md ## Branch model`.
- **`uv`, not `python`/`pip`:** `uv sync`, `uv add <pkg>`, `uv add --group dev <pkg>` (this repo uses
  `[dependency-groups] dev`), `uv run …`.
- **Run `make check` before declaring anything done** — ruff fix+format → pyright (delegates to
  `pre-commit run --all-files`). Both block the commit on failure. (`dev_flow.md ## Recipes` has
  the details.)
- **Don't sidestep a convention.** A convention collision means the design is wrong — fix the design,
  don't `# noqa` / `# type: ignore` / inline-import past it. Never use type-suppression markers to
  paper over a real type error in your own code. (Historical note: pyimgui's stub gaps used to need
  scoped suppressions; imgui-bundle's stubs are complete, so that exception is retired — see
  `conventions.md ## Known quirks` for the residual library-stub footguns that remain.)
- **Code conventions** (full list → `conventions.md`): full type annotations on params/vars;
  minimal comments (only non-obvious logic); imports at module top only.
- **Library docs + source are the source of truth.** For any non-trivial library API use, verify
  against the official docs and grep the library's source (in `.venv/lib/python*/site-packages/<lib>`
  or upstream repo via WebFetch) — never guess from training data, never invent signatures. Adopt
  the library's documented best-practice idiom even if it differs from the existing in-repo shape.

## Two meta-rules

(One-sentence summaries; full text → `dev_flow.md ## Maintainer habits`.)

- **Sessions are disposable; knowledge is durable.** If a decision lives only in this chat it's lost
  on `/clear` — file it (roadmap row/banner / todo deferral / `conventions.md ## Design decisions` /
  commit message). The cold-context resumability check (`/sanitize` step 6) gates suggesting `/clear`:
  simulate a fresh agent asking "what's next?", walk the chain, confirm it lands on the same
  next-action with the same blockers in a few reads — if not, the missing context IS the bug; write
  it to a file, not the chat.
- **Docs are living.** User drops a fact that makes a doc stale → update the right file in the same
  wave, don't keep it in chat. Small → do it now and mention it; substantial → confirm first.

## Documentation discipline

The bar: docs stay **robust** (failures loud, not silent), **smooth** (a fresh agent executes without
re-deriving rules), **cold-reloadable** (picks up "what's next?" in a few reads), **unbiased**
(resists confirmation bias + sympathetic reading). Full rationale + banned-pattern list →
`dev_flow.md ## Documentation discipline`.

- **One canonical home per concept.** Pointers, not copies — a rule restated in 3 places drifts.
- **Roadmap rows index; feature specs narrate.** A row is one table line + a `Spec:` pointer; the
  story (what landed, the bug, the review rounds) lives in the spec or the commit message. If a row
  wants a second sentence, that sentence belongs in the spec.
- **The Active-context banner gets rewritten, not appended.** ≤200 words; date stamp = last edit of
  the block. "Kept for traceability" history is the append-rot anti-pattern — git log is traceability.
- **`todo.md` indexes triggers, not designs.** Each entry names a concrete observable trigger moment.
- **Resolved entries get deleted in the resolving commit.** No `[RESOLVED]` headers, no strikethrough
  retentions — git history is authoritative.
- **Code comments state what's non-obvious about the code as it is now — never narrate development
  history.** The "why we did it / what bug we hit / where the backstory lives" belongs in its
  canonical home (`conventions.md ## Known quirks` / `todo.md`); the code points there in ≤1 line, or
  says nothing. (`conventions.md ## Code rules` is the enforced form.)
- **Cite by section name, not line number** — line numbers drift on every edit.
- **Shape comments are load-bearing.** The `<!-- Shape: ... -->` / `<!-- Row shape: ... -->` pins at
  the top of `roadmap.md`, `todo.md`, etc. train the next entry — don't strip them, don't violate them.

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

- `/sanitize` — the closing-out sweep (run before "done": walk `todo.md` + `conventions.md`, audit
  the code against them, fix stale refs/docs, update the `roadmap.md` banner + rows, cold-context
  check). Ends with a mandatory sweep-report table.
- `/imgui-ui` — read at the START of ANY UI work (building/editing/moving a button/panel/widget/tab/
  popup/layout, theming, refactoring draw code) and when reacting to a UI problem. Carries the
  hard-won imgui rules (button tiers, jitter-free overlays, the SetCursorPos assert, font/emoji
  caveats, the no-screenshot loop). Keeps the deep guidance out of the always-loaded context.
- `/ship` — the release-to-itch.io procedure (sanitize → commit/push `dev` → promote to `master` →
  auto-pick semver → `make release` → gated build → butler upload → page sync → land clean on `dev`).
  The single canonical home for the ship flow; `BUILDING.md` + `dev_flow.md` point here. **Run ONLY
  when the developer explicitly asks to ship/release/publish** — it tags, pushes, and publishes.

## Reply language

Reply in the language of the user's latest message (per global `~/.claude/CLAUDE.md`).

@ai_docs/conventions.md
