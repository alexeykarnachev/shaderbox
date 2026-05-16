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

## 2026-05-16 — feature 003 (ModelBox removal) IMPLEMENTED
- Spec: `ai_docs/features/003_modelbox_removal.md`. Goal: wipe all ModelBox integration
  (HTTP client, settings UI, app-state fields, dependency). Clean slate, no compat shims.
  -238 lines net across 14 files (1 delete, 6 code modifications, 1 dep removal, 6 doc
  updates).
- Diff shape: deleted `shaderbox/modelbox.py` (68 LOC) + `draw_media_models` widget (~40
  LOC) + Settings popup ModelBox block (28 LOC) + `App.modelbox_info` field +
  `App.fetch_modelbox_info()` method + 2 `UIAppState` fields + the `requests` dep
  (transitive `urllib3` / `charset-normalizer` auto-cleaned). Added `UIAppState`
  migration generation 3 (`data.pop("modelbox_url"); data.pop("media_model_idx")`)
  to keep existing `app_state.json` files loadable under pydantic's
  `extra="forbid"`. Migration is silent + idempotent; first save after load writes a
  clean file.
- decisions (locked in spec): clean delete no shims; `requests` dropped; silent pop
  migration; `widgets/media_ops.py` kept as single-function module; Settings section
  deleted not flag-hidden; README + CLAUDE.md prose updated alongside code; no dead
  `app.modelbox_info` preserved.
- review: 2 pre-impl reviewers (correctness/design + verification/blast-radius) →
  SHIP-WITH-EDITS, edits applied inline (docstring "Two"→"Three", explicit import
  surgery list, settings line range 73-94, transitive-dep note, exit-0 in manual step).
  3 post-impl reviewers (code correctness + architecture/conventions + spec fidelity) →
  all 3 SHIP, no blockers. Code-correctness reviewer ran `make check` + `make smoke` +
  4 integration tests independently. Architecture reviewer flagged pre-existing pyright
  drift in `conventions.md:16-18` as OBSERVATION (cleaned up in sanitize).
- `make check` clean (0 pyright errors); `make smoke` clean (200 frames exit 0, no more
  ModelBox connection-refused log noise). User skipped manual UX check; relied on
  reviewer convergence + smoke + zero pyright.
- refs: working tree (uncommitted).
- open thread: **no urgent next-step.** Backlog: (a) `[DEFERRAL] in-app replay` — wait
  for a multi-step bug that's painful to repro; (b) `[DEFERRAL] split ui.py / app.py
  further` — wait for app.py editing to feel painful (parallel-agent assessment
  parked this 2026-05-15).

## 2026-05-15 — headless smoke test (tiny mechanical)
- Added `scripts/smoke.py` (~65 lines): creates `App(project_dir=projects/dev/)` with
  `glfw.window_hint(VISIBLE, FALSE)` set before App's `glfw.create_window` call, runs 200
  frames of `update_and_draw`, asserts popup-mutex + `current_node_id ∈ ui_nodes ∪ {""}`.
  Save/restore the user's `~/.local/share/shaderbox/project_dir` pointer around the test
  (otherwise App's `_init` would clobber it). Exit 0 on success. ~1.5s runtime against the
  2-node dev project.
- Wired `make smoke` target (separate from `make check` — needs a real GL context, prints
  noisy ModelBox connection errors when the optional service isn't running). Fixed stale
  "pyright is non-blocking" comment in the existing `make check` target.
- Doc updates in same wave: `[DEFERRAL] headless smoke test` deleted from `todo.md`
  (resolved); `dev_flow.md ## Recipes` gets a `### make smoke` section + module-map entry
  for `scripts/smoke.py`.
- `make check` clean; `make smoke` clean.
- refs: working tree (uncommitted). `scripts/smoke.py`, `Makefile`.
- open thread: **next = re-evaluate project.py extraction with smoke test as safety net.**
  The aggregated agent decision parked it (`todo.md [DEFERRAL] split ui.py / app.py further`
  has the sharpened trigger) — default is to keep deferred unless `app.py` editing surfaces
  real pain. Alternative next-steps if not project.py: ModelBox blocking HTTP unblock
  (real UX bug, mid scope — apply feature-001 worker-thread + mailbox pattern to
  `modelbox.infer_media_model`); in-app replay mechanism (debug aid, deferred until a
  multi-step bug demands it).

## 2026-05-15 — pyright re-tightening (tiny, decision-driven)
- Dropped `|| true` from the pyright pre-commit hook in `.pre-commit-config.yaml` (line 29); the
  hook now blocks on failure. Verified with `make check`: 0 errors, 0 warnings, 0 informations.
  Stale doc bullets removed: `[DEFERRAL] re-tighten pyright` deleted from `todo.md`; the
  "Pre-existing pyright debt across the repo" bullet in `conventions.md ## Known quirks` deleted;
  `CLAUDE.md` and `dev_flow.md ## Recipes` paragraphs that described pyright as non-blocking
  rewritten as blocking.
- decisions: triggered by a parallel-agent assessment of the queued `project.py` extraction. Three
  read-only research agents (pro-extraction architect, skeptic, third-options scout) ran in
  parallel: 2/3 recommended against `project.py` now (skeptic: "same AppContext shape feature 002
  reversed"; scout: project.py ranked 3rd by ROI). Scout surfaced the free win — pyright was
  already at 0 errors, the `|| true` was legacy debt protection no longer load-bearing. Skipped
  project.py; ordering revised to pyright-now → smoke-test-next → re-evaluate project.py.
- refs: working tree (uncommitted).
- open thread: **next = headless smoke test** per `todo.md [DEFERRAL] headless smoke test` —
  `scripts/smoke.py` (~60 lines) creates `App` against `projects/dev/` with an invisible glfw
  window, runs ~200 frames of `update_and_draw`, asserts no exception + invariants (popup-mutex,
  current_node_id sanity, no released-texture leaks). Wire into `make check` or `make smoke`.
  Tiny-to-mid scope. After that: re-evaluate whether `project.py` extraction earns its keep
  (default: keep deferred with sharper trigger — see `[DEFERRAL] split ui.py / app.py further`).
  Parallel-track items still in `todo.md`: ModelBox blocking HTTP (real UX bug), in-app replay
  (debug aid).

## 2026-05-15 — hotkeys.py extraction (tiny mechanical)
- Pulled the ~40-line hotkey block out of `ui.py:update_and_draw` into new
  `shaderbox/hotkeys.py` exposing `process_hotkeys(app: App)`. Literal cut/paste — zero
  behavior change. `ui.py` 294 → 255; new `hotkeys.py` 45 lines.
- Treated as tiny per `dev_flow.md` (1 module, mechanical, no new public API): no spec, no
  review agents, no manual smoke (user skipped — diff is trivially obvious).
- `make check` clean: 0 pyright errors. Doc patches in same wave: `todo.md` line-numbered
  refs to the old block updated (`split ui.py` deferral progress line; in-app replay
  intercept-point pointer), `dev_flow.md ## Recipes` module map adds the new module.
- refs: working tree (uncommitted). `shaderbox/hotkeys.py`, `shaderbox/ui.py`.
- open thread: **next = `project.py` extraction** — App's lifecycle methods (`save`,
  `open_project`, `delete_current_node`, `create_node_from_selected_template`,
  `select_next_*`, the `@property` paths). Mid-sized feature (multi-method, behavioral
  surface) — likely warrants a short spec + 1 reviewer per dev_flow's mid shape. After
  that: pyright re-tightening once `media.py` / `core.py` / `modelbox.py` debt clears.
  Parallel-track items in `todo.md` unchanged (smoke test, in-app replay, ModelBox
  blocking HTTP).

## 2026-05-15 — feature 002 (UI widgets + popups extraction) IMPLEMENTED with 3 post-impl reversals
- Implemented per `ai_docs/features/002_ui_widgets_extraction.md`: extracted 10 widget/popup
  methods from `ui.py` into `shaderbox/widgets/{details,media_ops,node_grid,uniform}.py` (547
  lines) and `shaderbox/popups/{node_creator,settings}.py` (166 lines). Three post-impl reversals
  reshaped the design: (1) `AppContext` dataclass deleted after investigation showed every
  claimed benefit illusory — widgets/popups take `app: App` directly; (2) popup classes flattened
  to free functions, `_is_open` state moved to plain booleans on `App` (`is_node_creator_open` /
  `is_settings_open`) with mutex-preserving helpers `app.open_node_creator()` /
  `app.open_settings()`; (3) `App` extracted to `shaderbox/app.py` (373 lines) and dispatch
  methods moved to `ui.py` (294 lines) as free functions — broke the circular import surfaced
  by widgets needing `App` for type annotations while `app.py` imported them.
- Also in same wave (originally feature 003 scope): `tabs/node.py` + `tabs/render.py` extracted;
  `tabs/share.py` harmonized to `draw(app: App)` / `update(app: App)` (state moved to
  `tabs/share_state.py` to keep `app.py` cycle-free). `App` attrs `_share_tab_state`,
  `_exporter_registry`, `_font_14/18` made public (cross-module accessed from `ui.py`).
- Final layout (all `make check` clean, 0 pyright errors):
  `app.py` 373 + `ui.py` 294 + `tabs/{node,render,share,share_state}.py` 398 +
  `widgets/*` 547 + `popups/*` 166. `ui.py` net: 1508 → 294 (-1214).
- decisions (load-bearing, see `conventions.md ## Design decisions` for full text):
  three-layer architecture `app.py` (state) / `ui.py` (orchestrator) / `widgets`+`popups`+`tabs`
  (pure logic); widgets are an organizational convention with no shape contract; popups are
  free functions with state on App; `App.open_node_creator()` / `app.open_settings()` enforce
  popup mutual exclusion.
- review: 3 pre-impl reviewers (spec fidelity, architecture, devil's advocate) + 4 final
  reviewers (architecture, semantic correctness, conventions/drift, devil's advocate) flagged
  one BLOCKER (popup mutex broken when popup classes flattened — fixed via the `open_*` helpers)
  and several MAJORs (underscore-private attrs accessed cross-module — dropped; `tabs/share.py`
  inconsistency — harmonized in same wave; doc drift — patched in sanitize).
- refs: commit `7ad427f`; `ai_docs/features/002_ui_widgets_extraction.md` (Review history shows
  full reversal trail). Manual UX verification passed before commit.
- open thread: **next feature = `hotkeys.py` extraction** — the ~40-line hotkey block at
  `ui.py:86-124` inside `update_and_draw` (Ctrl+N/Alt+S/Esc/arrow/Enter handling). Small/mid
  feature: extract as `shaderbox/hotkeys.py` with one entrypoint `process_hotkeys(app)` called
  from `update_and_draw`. Likely zero blockers — pure mechanical extraction; widgets/popups
  pattern already set. Backlog after that: `project.py` extraction (App's lifecycle methods —
  `save`, `open_project`, `delete_current_node`, `create_node_from_selected_template`,
  `select_next_*`, the `@property` paths) — mid feature; then pyright re-tightening (drop
  `|| true` once `app.py` / `media.py` / `core.py` / `modelbox.py` pyright debt is cleared).
  Parallel-track items in `todo.md`: smoke test (60 lines, useful for future refactor
  verification), in-app replay mechanism (debug aid), ModelBox blocking HTTP unblock.

## 2026-05-15 — feature 001 (exporter refactor) IMPLEMENTED end-to-end
- Implemented per `ai_docs/features/001_exporter_refactor.md`: new `shaderbox/exporters/` subpkg
  (`base.py` ABC + value types, `registry.py`, `telegram.py` ~740 lines — own worker thread + own
  asyncio loop + own sticker-grid UI + ffmpeg prepare + mailbox progress); new
  `shaderbox/tabs/share.py` (first `tabs/*.py` — sets the convention: free `draw()` + optional
  `update()` + module-level `TabState`); deleted `shaderbox/{sharing,telegram_provider}.py`
  (433 lines); modified `ui.py` (1778 → 1508, -270; deleted `draw_share_tab` + asyncio loop +
  per-frame share preview + safe-wrapper) and `ui_models.py` (rename `share_provider_configs`
  → `exporter_settings`, `active_share_provider` → `active_exporter_id`, added
  `model_config = {"extra": "forbid"}`, `load_and_migrate` extended with new key-rename block
  after the legacy `tg_*` block); updated `conventions.md` / `dev_flow.md` / `todo.md` /
  `CLAUDE.md`. **Pyright 16 → 0 errors** (share-tab type debt fully cleaned up; broader debt
  in `media.py` / `core.py` / `modelbox.py` parked under new `[DEFERRAL] re-tighten pyright`).
- Two review rounds: 3 reviewers post-impl (code correctness, arch & conventions, spec-fidelity),
  then 1 round-2 convergence reviewer after fixes. Convergent blockers fixed: removed banned
  `from __future__ import annotations`, dropped banned `# type: ignore[union-attr]`, wrapped
  `imgui.begin_child` in try/finally, cancellable shutdown drainage via
  `loop.call_soon_threadsafe(task.cancel)`, cleanup on `prepare()` raise. Convergent majors
  fixed: replaced `__media_dir` settings side-channel with `Exporter.set_media_dir()` ABC
  method, `auth_state` → `@property`, relaxed Decision 15b from "no imports from media/core"
  to "method affinity" (with spec amendment), `draw_config_ui()` no-args + new
  `current_settings()` method, `rebind()` releases sticker GL handles, `_with_bot()` helper
  for the 5 bot-init-shutdown copies, full `current_node: UINode | None` typing.
- decisions (during impl, all amended into the spec's Review history): pyright re-tightening
  de-scoped to a separate effort (filed as new deferral); import discipline is method-affinity
  not import-affinity; `auth_state` is property not method; `draw_config_ui` is no-args mutating
  self-state, not (settings)->settings; `set_media_dir()` ABC method beats settings side-channel.
- refs: commit `de7059d`; `ai_docs/features/001_exporter_refactor.md` (Review history fully
  populated).
- open thread: **feature 002 (UI widgets + popups extraction) drafted but NOT plan-locked with
  user** — `ai_docs/features/002_ui_widgets_extraction.md`. User asked to organize the repo
  before any new feature work, this is the next chunk. Fresh-session resumption: walk
  `CLAUDE.md` → this entry → `ai_docs/features/002_ui_widgets_extraction.md`, answer the
  4 `## Open questions for the user`, then proceed to `dev_flow.md` step 3 (plan-lock with
  user) → step 4 (pre-impl review) → step 5 (implement). Backlog after 002: feature 003
  (extract `tabs/node.py` + `tabs/render.py`), feature 004 (extract `hotkeys.py`), feature 005
  (extract `project.py`), then revisit pyright re-tightening (`todo.md [DEFERRAL]`). ModelBox
  blocking-HTTP deferral can be a parallel-track mid-feature when the user feels like it.

## 2026-05-15 — exporter refactor spec drafted + plan-locked (backlog items 2 + 3)
- Drafted `ai_docs/features/001_exporter_refactor.md` — first feature spec; high-blast-radius
  shape per `dev_flow.md`. Three parallel reviewers (domain-fit on YouTube/X/Telegram APIs,
  architecture on GL lifecycle + threading, devil's-advocate steelmanning the boring fix) ran
  pre-spec; their findings reshaped the design substantially (GL-free `RenderedArtifact`, mailbox
  progress stream, `prepare()`/`export()` split, auth as a first-class sub-protocol). User then
  pushed back hard on the keyring decision — dropped it, credentials stay in `app_state.json`
  (revisit when YouTube/X land). All 12 design decisions locked; 0 open questions remaining.
- Scope: Telegram exporter ported onto the new abstractions; YouTube + X explicitly out of scope.
  Resolves at impl time: `todo.md [DEFERRAL] two near-identical sticker models` (incl.
  re-tightening pyright by dropping `|| true`) and `todo.md [DEFERRAL] blocking asyncio` (folded
  in via per-exporter worker-thread + mailbox). Advances `[DEFERRAL] split ui.py` by landing
  `tabs/share.py` as the first real `tabs/*.py` extraction.
- decisions: kept `python-telegram-bot` (wrap async in worker thread); stayed on `master` (no
  feature branch — solo project); credentials stay in JSON (no keyring — no real threat model
  yet); kept current share-tab live-preview UX (selected sticker shows live shader render).
- refs: `ai_docs/features/001_exporter_refactor.md`. No code changes this session.
- open thread: **start a fresh session for impl** — context is dense from spec-drafting + 3
  reviewer reports; cold context will be cleaner. Fresh session walks `CLAUDE.md` → this entry →
  `ai_docs/features/001_exporter_refactor.md` and proceeds to `dev_flow.md` step 4 (pre-impl
  review) with 1-2 reviewers, then step 5 (implement). Telegram is the only concrete exporter;
  manual verification needs your bot token + user id + sticker set name.

## 2026-05-15 — dead `ui_utils` sweep (backlog item 1)
- Deleted 4 dead helpers from `ui_utils.py`: `mod` (no callers), `depth_mask_to_normals` and
  `zero_low_alpha_pixels` (no callers; both reached into `Image._image` private — the only
  `_image` reach-throughs outside `media.py`), `get_dir_hash` (no callers — spotted while sweeping).
  Dropped now-orphan imports (`cv2`, `Image`, `from shaderbox.media import Image`, `Path`). 160 →
  114 lines. Resolved `todo.md [DEFERRAL] dead/orphaned ui_utils helpers` in the same commit — its
  premise ("future ModelBox wiring?") was wrong: ModelBox is server-side model dispatch
  (`modelbox.infer_media_model`), not client-side numpy/cv2. README claim ("depth maps, background
  removal") holds — those run through the existing `media_model_names` UI path. `make check` clean.
- decisions: mechanical, no design surface — straight implementation, no planning agents.
- refs: files: `shaderbox/ui_utils.py`, `ai_docs/{todo,conventions,worklog}.md`. (commit pending.)
- open thread: continue the cleanup backlog:
  1. ~~Dead `ui_utils` helpers — done this commit.~~
  2. Collapse the 2 sticker models (`TelegramShareableMedia` / `ShareableMedia`) into one behind a
     real interface; kill the `hasattr`-driven dispatch in the share tab. Re-tighten pyright (drop
     `|| true` from `.pre-commit-config.yaml`).
  3. Move blocking Telegram/ModelBox calls off the render thread (`_loop.run_until_complete` in
     imgui-frame draw paths; ModelBox's synchronous `requests`) — worker thread + result queue.
  4. Split `ui.py` (1778-line `App` god-class) — extract `widgets.py`, `tabs/*.py`, `hotkeys.py`,
     `project.py`. The big one.

## 2026-05-15 — Tier-1 cleanup batch (backlog item 1)
- Dropped unused deps (`litestar`, `uvicorn`, `uuid`-backport) from `pyproject.toml`; migrated
  `[tool.uv] dev-dependencies` → `[dependency-groups] dev` (kills the `uv` deprecation warning).
  Removed 3 inline-import violations (`telegram_provider.py:50`, `ui_models.py:224,231`). Deleted
  dead `UITgSticker` class (`ui_models.py:51-121`) — superseded by `ShareableMedia` +
  `TelegramShareableMedia` in `7cee0b4`; pruned its now-orphan imports (`hashlib`, `telegram as tg`,
  `Image`, `Canvas`, `Video`, `TYPE_CHECKING`). Removed dead `"image"` branch in
  `UIUniform.get_ui_height` (`ui_models.py:184`) — was a stale enum-era reference, the literal type
  uses `"texture"` (the rename in `82f974a` missed this branch). Updated `CLAUDE.md` +
  `conventions.md` to reference `uv add --group dev` instead of `--dev`. `make check` clean — 16
  pre-existing pyright errors remain (the ShareableMedia type debt, unchanged).
- decisions: ran as a mechanical change, not the feature flow — all-local, no design surface. The
  sticker-models deferral (`todo.md`) is narrowed from 3 models → 2 (UITgSticker no longer in the
  picture).
- refs: `7d3c44e`; files: `pyproject.toml`, `shaderbox/{telegram_provider,ui_models}.py`, `uv.lock`,
  `CLAUDE.md`, `ai_docs/conventions.md`.
- open thread: continue the cleanup backlog, in order:
  1. Decide on the dead `ui_utils` helpers: `mod` (delete), `depth_mask_to_normals` /
     `zero_low_alpha_pixels` (wire to ModelBox or delete).
  2. Collapse the 2 sticker models (`TelegramShareableMedia` / `ShareableMedia`) into one behind a
     real interface; kill the `hasattr`-driven dispatch in the share tab. Re-tighten pyright (drop
     `|| true` from `.pre-commit-config.yaml`).
  3. Move blocking Telegram/ModelBox calls off the render thread (`_loop.run_until_complete` in
     imgui-frame draw paths; ModelBox's synchronous `requests`) — worker thread + result queue.
  4. Split `ui.py` (1778-line `App` god-class) — extract `widgets.py`, `tabs/*.py`, `hotkeys.py`,
     `project.py`. The big one.

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
