# Roadmap

What's built and what's next. Per-feature **how** (plan → review → implement → sweep) lives in
`ai_docs/dev_flow.md`. Operational landmines/triggers live in `ai_docs/todo.md`. Design decisions
with revisit triggers live in `ai_docs/conventions.md ## Design decisions`.

Features are the planning unit: each is `ai_docs/features/NNN_name.md` (or a commit, for small ones
with no standalone spec). The **Active context** banner answers "what's next?"; the first `pending`
row (if any) is the next feature.

Status vocabulary: **pending** / **in progress** / **done** / **partial** (code shipped but a
follow-up is tracked — see linked spec/todo) / **superseded** (an earlier shape replaced by a later
feature; brief points at the superseder).

<!-- Row shape:
     | NNN | name | status | <what-it-does, ONE sentence>. Spec: <ai_docs/features/NNN_*.md | commit <sha>>. |
     Hard rule: ONE markdown table row, no multi-line narrative. Landed-reality (file paths touched,
     test counts, the bug-fix story, review-round detail) belongs in the feature spec or the commit
     message, NOT the row. If a row wants a second descriptive sentence, that sentence belongs in the
     spec. Status is ONE value from the vocabulary above — no compound forms.
-->

## Active context

<!-- Rewrite this block IN FULL each time it changes. Do NOT append. <=200 words. -->
<!-- Date stamp = last edit of this block, not the date of the work it summarises. -->

**As of 2026-05-24.**

- **Feature 011 (UI library consolidation + share-stack de-leak) — DONE (refactor, no behaviour
  change).** All six decisions landed: (C) `TabState.release()` fixes the preview/artifact GL leak
  on project switch; (B) dead `SIZE` tokens + `_FONT_DIR` + the unreachable Telegram `replace` job
  path deleted; (A) Telegram tokens moved out of generic `theme.SIZE` into telegram-local constants;
  (E) `ui_utils.py` split into `ui_primitives.py` (imgui+theme draws) + `util.py` (non-UI helpers),
  `fade()` exported from `theme.py`, plus a new `tests/` suite (`resolve_dims` + GL-backed
  `render_for`/`render_media`, run via `make test`); (D) the generic exporter seam de-leaked —
  `RenderControl` is pure render plumbing, per-exporter UI extras ride in an opaque `.extras` bag
  built via `Exporter.build_render_extras(OutletUiDeps)`, and the pack methods left the ABC (concrete
  on Telegram, reached via `isinstance`); (F) `duration_slider` parametrized. Fragile carousel/grid
  draw code untouched (Decision 1). `make check` + `make test` (15) + `make smoke` green; connected
  Telegram panel headless-driven clean. Spec: `ai_docs/features/011_ui_library_consolidation.md`.
- **Invoke `/imgui-ui` before any UI work** — it carries the button tiers, jitter-free overlays,
  SetCursorPos assert, font/emoji caveats, no-screenshot loop.
- **NEXT ACTION: none queued.** No `pending` feature row. Candidate next waves all sit in `todo.md`
  as trigger-gated deferrals (the deferred-N=2 extractions in the 011 spec's DEFER table; the
  sticker loop-offset; the inline-editor upstream items) — none is a default next-step; pick up only
  when its trigger fires.
- **Shipped: `v0.4.0`** (009+010+011) — tagged on `master`, both itch channels (linux+windows)
  pushed via butler and live at 0.4.0. `dev` == `master` == `v0.4.0`; tree clean.
- **Branch model:** develop on `dev`, ship from `master` (`dev_flow.md ## Branch model`).
- **Token hygiene:** the dev bot token lives only in `~/.local/share/shaderbox/integrations.json`
  (outside the repo, never committed); maintainer rotates it post-iteration.
- **Three non-blocking follow-ups (no users yet):** (1) no build runtime-verified on Windows — verify
  per `BUILDING.md`; (2) live-page screenshots stale (predate 005/006); (3) itch **store page** not
  synced this ship — `page.yaml` unchanged, but the page-sync (Playwright, review-gated) was skipped;
  run `dev_flow.md ### Sync the itch.io page` if the page needs refreshing.
- **No open BLOCKERs.** `todo.md` deferrals fire on their own triggers — don't pick up speculatively.

## Features

| # | Name | Status | Brief |
|---|---|---|---|
| 011 | ui_library_consolidation | done | Post-010 refactor (no behaviour change): de-leaked Telegram tokens out of generic `SIZE`; deleted dead tokens + the unreachable `replace` path; fixed the preview-GL leak on project switch (`TabState.release`); de-leaked the exporter seam (`RenderControl` pure plumbing + `.extras`/`build_render_extras`, pack methods off the ABC); split `ui_utils.py` → `ui_primitives.py`+`util.py`; landed a `tests/` suite (`make test`). Fragile carousel/grid draw code out of scope; GridCell/Widget/RenderPreset-FIXED_* extraction deferred to N=2. Spec: `ai_docs/features/011_ui_library_consolidation.md`. |
| 010 | outlet_render_rework | done | Shared GL-free `RenderPreset` drives `render_media` to render natively at the outlet's target size; per-outlet concise share UI. Telegram panel redesigned (many screenshot rounds): pack row + copyable link, new-sticker block (emoji overlay, Duration, Render + Add), single-row carousel with per-cell emoji-change + delete-confirm; status via notifications. Produced the 4-tier button system + UI primitives + the `imgui-ui` skill. Spec: `ai_docs/features/010_outlet_render_rework.md`. |
| 009 | integrations_rework | done | Telegram UX collapse: token+Connect (getUpdates user_id capture) in Settings→Integrations; derived+auto-created sticker-set names; share-tab pack create/select/delete + sticker delete; monochrome emoji picker (Unicode order; color impossible in this build); bot DMs pack link on add; connection+packs persist to global `integrations.json` (connect once); no legacy migration; YouTube/X stubs. Live-verified end-to-end; 2 pre+2 post review waves PASS. UI-optimization is a separate next wave. Spec: `ai_docs/features/009_integrations_rework.md`. |
| 008 | uniform_input_shapes | done | Every uniform row leads with a changeable input-shape pill (click-to-cycle, disabled when one shape); removed the separate settings pane; per-node uniform sort (code/name/type + dir); engine uniforms collapsed to one dim readout row; node-tab header compaction; accent tab styling. Spec: `ai_docs/features/008_uniform_input_shapes.md`. |
| 007 | release_pipeline_hardening | done | `make release VERSION=` (semver bump + tag), `build.sh` gated on check+smoke + clean tree (`--allow-dirty`), `.zip` both platforms, Windows+Ubuntu CI, `BUILDING.md`. Spec: `ai_docs/features/007_release_pipeline_hardening.md`. |
| 006 | inline_glsl_editor | done | Syntax-highlighted inline GLSL editor in the main window's left split; Ctrl+S saves + hot-reloads; visual-options popup; replaced the external-editor mechanism. Spec: `ai_docs/features/006_inline_editor.md`. |
| 005 | ui_redesign_foundation | partial | Gruvbox theme + full color/size/spacing token sweep through `theme.py`. The wide-screen layout half was reverted to the feature-004 shape; only the theme survived. Spec: `ai_docs/features/005_ui_redesign_foundation.md`. |
| 004 | imgui_bundle_migration | done | Full pyimgui → imgui-bundle migration; adopted the `imgui_ctx` context-manager idiom + `portable_file_dialogs`; stripped the 8 imgui `# type: ignore` markers. Spec: `ai_docs/features/004_imgui_bundle_migration.md`. |
| 003 | modelbox_removal | done | Wiped all ModelBox integration (HTTP client, settings UI, app-state fields, `requests` dep); app_state migration gen 3. Spec: `ai_docs/features/003_modelbox_removal.md`. |
| 002 | ui_widgets_extraction | done | Three-layer split: `app.py` (state) / `ui.py` (orchestrator) / `widgets`+`popups`+`tabs` (pure draw fns); `ui.py` 1508 → 294. Spec: `ai_docs/features/002_ui_widgets_extraction.md`. |
| 001 | exporter_refactor | done | `exporters/` subpkg — `Exporter` ABC + `RenderedArtifact` value type + registry; Telegram ported to own worker thread + asyncio loop + sticker UI. Spec: `ai_docs/features/001_exporter_refactor.md`. |
| — | editor_unfocused_dim | done | Code editor dims (`style.Alpha`) when it lacks keyboard focus; `EDITOR_UNFOCUSED_ALPHA` token in `theme.py`. Spec: commit `225076a`. |
| — | hotkeys_extraction | done | Pulled the hotkey block out of `ui.py` into `hotkeys.py::process_hotkeys(app)`. Spec: commit `c7d6359`. |
| — | smoke_test | done | `scripts/smoke.py` + `make smoke`: 200 headless frames of `update_and_draw`, asserts popup-mutex + `current_node_id` invariants. Spec: commit `92f221a`. |
| — | pyright_blocking | done | Dropped `\|\| true` from the pyright pre-commit hook — typecheck now blocks on failure; repo at 0 errors. Spec: commit `d1fdf65`. |
| — | devflow_scaffold | done | Created `CLAUDE.md` cold-start chain + `ai_docs/{dev_flow,todo,conventions}.md` + `/sanitize` skill + `Makefile`. Spec: commit `cd430d1`. |
