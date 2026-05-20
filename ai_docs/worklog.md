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

## 2026-05-20 — feature 005 LAYOUT reverted; gruvbox theme kept
- User reviewed the running app side-by-side with `ai_docs/design/prototype.html`
  and rejected the layout (feature 005's wide-screen 50/50 split + editor
  placeholder + bottom status bar). A follow-on attempt to match the prototype's
  right-pane layout (feature 006, branch `feature/ui-redesign-realize`) was also
  rejected before merge.
- Resolution: keep the gruvbox **theme + token sweep** (it refines the code —
  every color/size now flows through `shaderbox/theme.py`), revert only the
  **layout** back to the feature-004 shape (image on top, control panel below
  with node-grid-left + node-settings-tabs-right; half-monitor window;
  top-right notification overlay).
- Mechanics: restored `ui.py` + `notifications.py` from feature 004 (`41f2737`),
  reverted `app.py` window geometry to half-monitor, then re-applied the token
  sweep onto those 3 files. The other 11 feature-005 files (theme.py + 10
  token-only swaps in widgets/popups/tabs/exporters/ui_models) kept as-is.
  Net diff vs feature 005: 3 files, layout-only.
- `theme.py` (`apply_theme` + `COLOR`/`SIZE`/`SPACE` bags) stays. `app.py`
  still calls `apply_theme(imgui.get_style())` at boot, so the app is
  gruvbox-skinned with the OLD layout.
- The unmerged `feature/ui-redesign-realize` branch (feature 006 — horizontal
  node strip, type-pill uniforms, right-pane-only) is abandoned. Its spec
  (`ai_docs/features/006_ui_redesign_realize.md`) and the prototype archive
  stay on disk for reference but the code is not merged.
- refs: branch `feature/keep-theme-revert-layout` (this work), to squash-merge
  to master. `make check` 0 errors, `make smoke` 200 frames exit 0.
- open thread: **gruvbox theme is live on the OLD (feature-004) layout.** The
  designer's prototype layout is shelved — if revisited, start fresh from
  `ai_docs/design/prototype.html` as the visual source-of-truth (don't
  paraphrase via SPEC.md prose; that's what caused the two failed attempts).
  features 007-009 deferrals in `todo.md` (embedded editor / error banner /
  Tweaks panel) all assumed the new layout — re-scope them if the layout is
  ever revisited.

## 2026-05-16 — feature 005 (UI/UX redesign foundation, PRs 1-4) IMPLEMENTED
- Spec: `ai_docs/features/005_ui_redesign_foundation.md`. Squash-merged from
  `feature/ui-redesign-foundation` (7 sub-commits) into master. Implements the
  foundation of the gruvbox+wide-screen UI redesign delivered by the Claude
  Design agent: theme drop-in, full token sweep, status bar, wide-screen layout.
- Designer's deliverables archived at `ai_docs/design/{SPEC.md,tokens.json,
  prototype.html,README.md}`. Token vocabulary is the source of truth for
  `shaderbox/theme.py`.
- Three `theme.py` fixes applied during drop-in:
  - Removed `from __future__ import annotations` (conventions violation).
  - Removed `_setattr_if` try/except slot-defense; replaced with direct
    `style.set_color_(col.name, value)` against the pinned imgui-bundle 1.92.801
    pyi. Pre-1.91 names (`nav_highlight`, `tab_active`, `tab_unfocused*`) dropped
    entirely — they were renamed in Dear ImGui 1.91+.
  - Removed `load_fonts(io)`; `App.get_font()` stays the single font-loading
    entry point. JetBrains Mono adoption deferred (separate trigger).
  - Replaced `style.colors[idx] = value` with `style.set_color_(idx, value)` —
    imgui-bundle's `Style.Colors[]` array isn't exposed as a Python attribute;
    only the `set_color_` method wraps it. Caught during PR 1 `make check`.
- Token sweep landed: 28 hardcoded color tuples + 11 hardcoded size constants
  consolidated into `COLOR.* / SIZE.* / SPACE.*` bags. Sanity grep returns 0
  semantic color literals; only `(0.2, 0.2, 0.2)` shader-error dim tint remains
  (explicitly deferred to feature 008 — replaced when error banner widget lands).
- Wide-screen layout: glfw window switched from half-monitor to full-monitor.
  Frame body restructured: topbar (~32px) → main split (left editor placeholder
  + right pane with render card + node panel) → status bar (~24px pinned bottom)
  → modals. Status bar absorbs FPS readout (relocated from topbar) + notifications
  (relocated from top-right overlay). Shortcut legend always-visible right-aligned
  in status bar. Editor placeholder is a centered "coming in feature 006" panel.
- Post-impl review: 2 parallel adversarial reviewers (correctness/arch +
  spec-fidelity). Iter1 surfaced: GLSL version segment missing from statusbar,
  telegram.py thumb-height literal not migrated to SIZE token (spec-fidelity);
  width math off-by-`SPACE.MD` in `_draw_main_split`, three `(1,0,0)` color
  literals in app.py/details.py (architecture). Iter2 patches: added
  `_draw_statusbar_glsl`, migrated telegram constants, replaced `SPACE.SM` with
  `imgui.get_style().item_spacing.x` for split-gap math, swept the 3 stragglers
  to `COLOR.STATE_ERROR[:3]`. Iter2 convergence reviewer: CONVERGED.
- `make check`: 0 errors, 3 portable_file_dialogs `reportMissingModuleSource`
  warnings (pre-existing, documented in conventions). `make smoke`: 200 frames,
  exit 0, diff vs `/tmp/smoke_baseline_pre_005.log` is timestamps only.
- Deferrals filed in `todo.md`: features 006 (embedded GLSL editor — PR 5), 007
  (Node tab restructure — PR 6), 008 (inline shader-error banner — PR 7), 009
  (Tweaks panel — PR 8). Each has a concrete trigger.
- refs: squash-merged to master as `194aa0f` "feature 005: UI/UX redesign
  foundation". Feature branch deleted. Pushed.
- open thread: **feature 006 (embedded GLSL editor) is the next natural feature
  to pick up, but trigger is "when the placeholder feels like friction in daily
  use" — not a default next-step.** Manual UX sweep of master after squash-merge
  (visually confirm: full-monitor window, gruvbox skin, status bar at bottom,
  editor placeholder on left, render card + nodes on right) is recommended
  before next session. Pre-impl validation for feature 006 should grep
  `imgui_color_text_edit` symbols in the installed `imgui-bundle 1.92.801` pyi
  before plan-locking.

## 2026-05-16 — feature 004 (pyimgui → imgui-bundle migration) IMPLEMENTED
- Spec: `ai_docs/features/004_imgui_bundle_migration.md`. 5 commits on
  `feature/imgui-bundle-migration`: spec patch (`3c6b6e7`) + boot/frame-loop/hotkeys
  (`1db7209`) + remaining widgets/popups/tabs/exporters (`16bdfee`) + docs
  (`7a31202`) + convergence-iter2 fix (`6cf1d95`). 22 files, +539/-380.
- Pre-impl validation 0b/0c surfaced 4 spec gaps the 3-round convergence review had
  missed (the prior reviews fetched the upstream pyi via WebFetch but didn't install
  + grep the actual `1.92.801` stubs):
  - `push_font(font)` now requires `(font, size)` — 2 args mandatory.
  - `get_glyph_ranges_cyrillic()` + `glyph_ranges=` kwarg REMOVED (1.92 dynamic
    on-demand glyph loading via `BackendFlags_.renderer_has_textures`).
  - `imgui_renderer.refresh_font_texture()` REMOVED.
  - pfd `open_file`/`save_file`/`select_folder` are non-blocking class handles, not
    blocking functions — needed a `pfd_block(dialog)` helper to preserve the
    crossfiledialog synchronous call-site shape.
  Patched as new Decisions 10 + 11 in the spec (commit `3c6b6e7`) before any
  production code moved. Lesson: pre-impl convergence reviews must include an
  "install + grep" pass, not just upstream-fetch — frozen training data on
  imgui 1.91 vs. live 1.92.801 caused the miss.
- Additional 1.92 breaks found during impl (not in spec; handled inline):
  `imgui.image()` lost `tint_col`/`border_col` (moved to `image_with_bg`);
  `is_key_pressed(ord("O"))` → `is_key_pressed(imgui.Key.o)`; `COLOR_TEXT` /
  `WINDOW_NO_SCROLLBAR` / `INPUT_TEXT_PASSWORD` etc. all moved to namespaced
  enums (`Col_.text`, `WindowFlags_.no_scrollbar`, `InputTextFlags_.password`);
  `image_button` requires explicit `str_id` first arg; `imgui.begin` returns a
  tuple — `imgui_ctx.begin` is the ctx-mgr wrapper; `imgui_ctx.begin_child`
  requires `size: ImVec2` (strict), not bare tuples — must wrap as
  `imgui.ImVec2(w, h)`. Captured in `conventions.md ## Known quirks`.
- Post-impl review: 3 parallel adversarial reviewers (correctness +
  architecture + spec-fidelity). Iter1: 1 PASS, 2 PARTIAL — correctness MINOR
  (5 `begin_child`/`end_child` pairs missed Decision 5 ctx-mgr migration),
  architecture MAJOR (notifications.py rewrite to dataclass violated spec
  out-of-scope "stays as-is"). Iter2 patch (`6cf1d95`): migrated all 5
  `begin_child` sites + reverted notifications.py to minimal diff (only import
  swap + line-42 `text_colored` arg swap). Iter2 both PASS — CONVERGED.
- `# type: ignore` audit: 20 markers on master → 8 on branch. The 12 imgui-related
  markers stripped cleanly; the 6 non-imgui survivors (`moderngl.gl_type`,
  `freetype.load_char`, `pydantic.model_validator`) untouched per Decision 8.
  The CLAUDE.md "imgui exception" line + `conventions.md ## Known quirks` bullet
  rewritten — imgui-bundle's stubs are complete, exception retired.
- `make check`: 0 errors, 3 `reportMissingModuleSource` warnings on
  `portable_file_dialogs` (`.pyi`-only stub, harmless, documented as a known
  quirk). `make smoke`: 200 frames, exit 0, diff vs
  `/tmp/smoke_baseline_pyimgui.log` is timestamps only.
- Filed deferrals: `hello_imgui.apply_theme()` themes + `imgui-knobs` rotary
  knobs — both rejected from this feature's scope to bound blast radius, both
  trigger at the start of the planned UI/UX refactor with custom themes.
- refs: squash-merged to master as `41f2737` "feature 004: migrate to
  imgui-bundle". Feature branch deleted. Not pushed yet.
- open thread: **manual UX sweep on master (deferred per user — squash-merge
  done on smoke confidence).** Procedure: `ai_docs/features/004_imgui_bundle_
  migration.md ## Manual verification`, gates 3-13 (settings popup, node
  creator, hotkeys, uniform editor, tab bar, telegram UI w/ fallback, project
  open, node grid, project switching FPS canary 60→45, save/restart). If a
  gate fails, fix-forward on master (single squashed commit is a clean revert
  target via `git revert 41f2737` if any catastrophic regression). Then
  `git push origin master` when user gives the word.

## 2026-05-16 — feature 004 (pyimgui → imgui-bundle migration) PLAN-LOCKED, ready to implement
- Spec: `ai_docs/features/004_imgui_bundle_migration.md` — large feature, high blast radius.
  9 locked design decisions, every API choice citation-verified against official imgui-bundle
  source (the pyi stub at github.com/pthom/imgui_bundle and python_backends/ source files).
- Process so far: 2 research rounds (call inventory + integration layer) → 2 spike rounds
  (5 UNCONFIRMED items resolved + bundled-addons audit) → spec draft → plan-lock with user →
  pre-impl review round 1 (2 reviewers parallel, SHIP-WITH-EDITS, ~10 findings) → convergence
  iter2 (1 NEW BLOCKER caught — `text_colored` arg-swap missed by 3 prior passes) → iter3
  (file-count typo + 2 MAJOR clarifications) → loop converged PASS. The new global
  "convergence loop" rule earned its keep on the `text_colored` catch specifically.
- Scope: full pyimgui → imgui-bundle migration + adopt context-manager idiom (`with
  imgui_ctx.begin_*`) per library's official demos + swap `crossfiledialog` →
  `imgui_bundle.portable_file_dialogs` + aggressive strip of 8 imgui-related `# type: ignore`
  (12 non-imgui markers stay untouched per Decision 8 scoping). 322 imgui calls across 15
  files. Multi-day work.
- Out of scope: `hello_imgui` boot/utilities (manual loop preserved); all other addons
  (markdown, implot, imspinner, knobs, node-editor, etc. evaluated and rejected); custom
  notifications class (44 LOC homegrown, no swap target).
- Branch strategy (explicit override of repo's "current branch" default): work on
  `feature/imgui-bundle-migration`. Squash-merge to master at completion. Rebase on
  conflict during impl.
- Pre-impl validation steps captured in spec (0a, 0b, 0c). **Step 0a baseline ALREADY
  captured** at `/tmp/smoke_baseline_pyimgui.log` (242 bytes, exit 0, "200 frames, 2 nodes").
  If the `/tmp/` file is missing when impl resumes, spec step 0a has the recovery procedure
  (re-capture from master via git stash + checkout dance).
- Future work parked in spec out-of-scope + filed as sanitize-time deferral: adopt
  `hello_imgui.apply_theme()` utilities when the planned UI/UX refactor with custom themes
  starts. `imgui-knobs` deferred for the same reason.
- refs: working tree (uncommitted spec + this worklog entry).
- open thread: **fresh agent picks up at task #25 — create `feature/imgui-bundle-migration`
  branch off master, then execute the spec.** Sub-commits should be organized by logical
  group (boot → frame loop → flags & inputs → widgets → popups → tabs → file dialogs →
  cleanup) per Decision 9. Each commit aims to keep `make smoke` green where possible.
  After implementation: 2-3 post-impl reviewers in parallel (architecture + correctness +
  spec-fidelity, since this is high-blast-radius), full 13-step manual UX sweep, sanitize,
  squash-merge.

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
