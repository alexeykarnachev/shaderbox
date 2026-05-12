# Worklog

Repo-scoped memory: where work left off, key decisions, references. **Newest entry on top.** Keep
entries terse — a few lines each. Prune old entries when they stop being load-bearing. This is what a
fresh agent (`/clear`-ed context) reads after `CLAUDE.md` to answer "what's the state here?"

The top entry's `open thread:` line is the resumable "what's next" carrier — it holds the rough,
ordered resumption backlog (there's no separate `roadmap.md`).

Sanity check: if the top entry's date is much older than `git log -1 --format=%cd`, the worklog
wasn't kept current — reconstruct recent state from `git log` first, then add the missing entry
(that's `/sanitize` step 5).

Format per entry:
```
## YYYY-MM-DD — <short title>
- <what was done>
- decisions: <load-bearing choices, if any>
- refs: <commit SHAs / key files; if work landed on a branch or opened/closed an MR, say so>
- open thread: <what's next, or "none">
```

---

## 2026-05-12 — AI dev-flow scaffold landed
- Created `CLAUDE.md` (anchor + cold-start chain), `ai_docs/{dev_flow,worklog,todo,conventions}.md`,
  `.claude/skills/sanitize/SKILL.md`, `Makefile` (`make check`); gitignored `imgui.ini` + `tmp.mp4`
  (and `git rm --cached`'d them). Modeled on the owner's cc-server / cc-android / ovelia setups,
  sized for a solo desktop tool (loosest = cc-server is the spirit). Plan history: a
  `_dev_flow_plan_DRAFT.md` predecessor went through a 3-agent review round + maintainer decisions,
  then was deleted.
- decisions: no `arch.md` (module map → `dev_flow.md ## Recipes`; refactor end-state →
  `conventions.md ## Design decisions` + `todo.md` deferrals). No `roadmap.md` (this `open thread:`
  line carries the backlog). Feature-flow default = 1-2 pre + 2-3 post review agents in parallel;
  other sizes per situation. The lint/typecheck command lives in one place — the `Makefile`
  (`make check` → `pre-commit run --all-files`). **Type checker: pyright, not mypy** (fewer false
  positives; `[tool.pyright]` in `pyproject.toml`, basic mode) — and **non-blocking for now**: the
  pre-commit pyright hook is `|| true`'d because of ~16 pre-existing `reportAttributeAccessIssue`
  errors in `ui.py`'s share-tab dispatch (the sticker-models debt); re-tighten when that refactor
  lands (`todo.md`). Type suppression is a scoped imgui-only exception otherwise. `projects/dev/`
  stays tracked (it's the maintainer's dev project). Commits: short single-line ASCII; allowed when
  the user asks (including standing instructions); current branch only, no per-feature branches.
- refs: this commit; files as above.
- open thread: resumption backlog, rough order — re-confirm with the maintainer before starting each:
  1. **Tier-1 cleanup batch** — strip unused/wrong deps from `pyproject.toml` (`litestar`, `uvicorn`,
     `uuid` — the last is the dead PyPI backport shadowing stdlib `uuid`); migrate
     `[tool.uv] dev-dependencies` → `[dependency-groups] dev` (deprecation warning on every `uv`
     command); remove the 3 inline-import violations (`telegram_provider.py:50`, `ui_models.py:224` +
     `:231`); delete the dead `"image"` branch in `UIUniform.get_ui_height` (`ui_models.py:184` — not
     in the `UIUniformInputType` literal). **Run this *through the new flow* as its first shakedown.**
  2. Bootstrap a minimal pytest suite (headless GL fixture, `moderngl.create_standalone_context()`)
     around `core.py` (uniform-type dispatch, `VIDEO_RESOLUTION_ALIGNMENT` rounding, node round-trip)
     + `ui_utils.py` pure helpers — the safety net for items 3-6.
  3. Delete dead code: `UITgSticker` (`ui_models.py:51`, superseded); the unused `ui_utils` helpers
     `mod` / `depth_mask_to_normals` / `zero_low_alpha_pixels` — or wire the latter two to ModelBox
     (decide).
  4. Collapse the 3 near-identical sticker models (`UITgSticker` / `TelegramShareableMedia` /
     `ShareableMedia`) into one behind a real interface; kill the `hasattr`-driven dispatch in the
     share tab.
  5. Move the blocking Telegram/ModelBox calls off the render thread (`_loop.run_until_complete` in
     imgui-frame draw paths; ModelBox's synchronous `requests`) — worker thread + result queue.
  6. Split `ui.py` (1778-line `App` god-class) — extract `widgets.py` (the `draw_*_details` family),
     `tabs/*.py`, `hotkeys.py`, `project.py`. The big one — high-blast-radius feature.

  The landmines for items 3-6 are seeded in `todo.md` with triggers; items 1-2 don't have `todo.md`
  entries (1 is mechanical, 2 is "do it before the next feature touching `core.py`").
