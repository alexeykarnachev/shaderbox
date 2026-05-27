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

**As of 2026-05-27.**

- **Shipped: `v0.8.0`** (feature 012 — YouTube upload exporter) — `master`, both itch channels live,
  tag pushed. The second concrete exporter: bring-your-own-OAuth (paste your own
  `client_secret.json`, Connect once), **private-only** uploads of long-form + Shorts (in-panel
  shape toggle), copyable Studio deep-link. Sync worker thread (no asyncio, unlike Telegram). New
  deps: `google-api-python-client`, `google-auth-oauthlib`. Same wave: Telegram `HTTPXRequest`
  timeouts raised (5s→30s, 120s media — tunnels were timing out); notifications moved to bottom-right
  (were hidden under the tab bar); the share-panel UI factored into shared `ui_primitives`
  (`preview_box`, `status_slot`, `labeled_*`, `unconnected_gate`, `connection_status`, `setup_steps`)
  + one `SIZE.SHARE_PREVIEW_*` so Telegram/YouTube panels match and can't jitter.
- **v0.7.0** (prior ship): template polish + `preview_cell` consolidation + delete-path crash fixes.
- **Invoke `/imgui-ui` before any UI work** — button tiers, jitter-free overlays, SetCursorPos assert,
  font/emoji caveats, the `is_mouse_hovering_rect`-ignores-popups + text-never-wraps traps, no-screenshot loop.
  Share-panels compose `ui_primitives` (preview-left fixed/taller + controls stacked top-down; no
  vertical-alignment math) — `conventions.md ## Design decisions`.
- **NEXT ACTION: none queued.** No `pending` feature row. Candidates sit in `todo.md` as trigger-gated
  deferrals — pick up only when a trigger fires.
- **`dev` == `master`** at `v0.8.0`. Tree clean.
- **Branch model:** develop on `dev`, ship from `master` (`dev_flow.md ## Branch model`).
- **Token hygiene:** the dev bot token + YouTube creds live only in
  `~/.local/share/shaderbox/integrations.json` (outside the repo, never committed; cleartext — see
  `todo.md` deferral); maintainer rotates post-iteration.
- **No open BLOCKERs.** `todo.md` deferrals fire on their own triggers — don't pick up speculatively.

## Features

| # | Name | Status | Brief |
|---|---|---|---|
| 012 | youtube_export | done | Second concrete exporter: YouTube upload (long-form + Shorts) via user-owned OAuth (bring-your-own client_secret, Connect once); private-only uploads + copyable Studio edit-link; in-panel shape toggle + title/description/tags/category; sync worker thread. Same wave: share-panel UI factored into shared `ui_primitives` + `SIZE.SHARE_PREVIEW_*`; Telegram timeouts raised; notifications moved bottom-right. Spec: `ai_docs/features/012_youtube_export.md`. |
| — | template_polish | done | Authored template order (`_TEMPLATE_ORDER`); merged Image+Video into one split-screen Media Input; text shader gained lowercase (folded to caps) + punctuation (`. , ; & -`) + newline layout + homegrown value-noise (dropped vendored Ashima simplex); compact video-smoothing slider block beside the thumbnail. Spec: commit `6b474f5` + `f44cf82`. |
| — | preview_cell_consolidation | done | Merged node-grid + sticker-carousel cell draw into one `preview_cell` primitive (+ `chip_button`, `centered_image`, `_ellipsize`); closed 011's deferred GridCell-at-N=2. Spec: commit `44b97d0`. |
| — | node_delete_crash_fixes | done | Two delete-path crashes: node-grid mid-iteration mutation (snapshot + deferred `delete_node`); trash-dir collision on deleting the same id twice (timestamp-suffixed dest). Spec: commit `1ab5d56` + `a386713`. |
| 011 | ui_library_consolidation | done | Post-010 refactor (no behaviour change): de-leaked Telegram tokens out of generic `SIZE`; deleted dead tokens + the unreachable `replace` path; fixed the preview-GL leak on project switch (`TabState.release`); de-leaked the exporter seam (`RenderControl` pure plumbing + `.extras`/`build_render_extras`, pack methods off the ABC); split `ui_utils.py` → `ui_primitives.py`+`util.py`; landed a `tests/` suite (`make test`). The N=2 GridCell extraction it deferred landed as `preview_cell` (row above). Spec: `ai_docs/features/011_ui_library_consolidation.md`. |
| 010 | outlet_render_rework | done | Shared GL-free `RenderPreset` drives `render_media` to render natively at the outlet's target size; per-outlet concise share UI. Telegram panel redesigned (many screenshot rounds): pack row + copyable link, new-sticker block (emoji overlay, Duration, Render + Add), single-row carousel with per-cell emoji-change + delete-confirm; status via notifications. Produced the 4-tier button system + UI primitives + the `imgui-ui` skill. Spec: `ai_docs/features/010_outlet_render_rework.md`. |
| 009 | integrations_rework | done | Telegram UX collapse: token+Connect (getUpdates user_id capture) in Settings→Integrations; derived+auto-created sticker-set names; share-tab pack create/select/delete + sticker delete; monochrome emoji picker (Unicode order; color impossible in this build); bot DMs pack link on add; connection+packs persist to global `integrations.json` (connect once); no legacy migration. Live-verified end-to-end; 2 pre+2 post review waves PASS. UI-optimization is a separate next wave. Spec: `ai_docs/features/009_integrations_rework.md`. |
| 008 | uniform_input_shapes | done | Every uniform row leads with a changeable input-shape pill (click-to-cycle, disabled when one shape); removed the separate settings pane; per-node uniform sort (code/name/type + dir); engine uniforms collapsed to one dim readout row; node-tab header compaction; accent tab styling. Spec: `ai_docs/features/008_uniform_input_shapes.md`. |
| 007 | release_pipeline_hardening | done | `make release VERSION=` (semver bump + tag), `build.sh` gated on check+smoke + clean tree (`--allow-dirty`), `.zip` both platforms, Windows+Ubuntu CI, `BUILDING.md`. Spec: `ai_docs/features/007_release_pipeline_hardening.md`. |
| 006 | inline_glsl_editor | done | Syntax-highlighted inline GLSL editor in the main window's left split; Ctrl+S saves + hot-reloads; visual-options popup; replaced the external-editor mechanism. Spec: `ai_docs/features/006_inline_editor.md`. |
| 005 | ui_redesign_foundation | partial | Gruvbox theme + full color/size/spacing token sweep through `theme.py`. The wide-screen layout half was reverted to the feature-004 shape; only the theme survived. Spec: `ai_docs/features/005_ui_redesign_foundation.md`. |
| 004 | imgui_bundle_migration | done | Full pyimgui → imgui-bundle migration; adopted the `imgui_ctx` context-manager idiom + `portable_file_dialogs`; stripped the 8 imgui `# type: ignore` markers. Spec: `ai_docs/features/004_imgui_bundle_migration.md`. |
| 003 | modelbox_removal | done | Wiped all ModelBox integration (HTTP client, settings UI, app-state fields, `requests` dep); app_state migration gen 3. Spec: `ai_docs/features/003_modelbox_removal.md`. |
| 002 | ui_widgets_extraction | done | Three-layer split: `app.py` (state) / `ui.py` (orchestrator) / `widgets`+`popups`+`tabs` (pure draw fns); `ui.py` 1508 → 294. Spec: `ai_docs/features/002_ui_widgets_extraction.md`. |
| 001 | exporter_refactor | done | `exporters/` subpkg — `Exporter` ABC + `RenderedArtifact` value type + registry; Telegram ported to own worker thread + asyncio loop + sticker UI. Spec: `ai_docs/features/001_exporter_refactor.md`. |
| — | telegram_ipv4_fix | done | Telegram uploads failed with `httpx.ConnectError` on an IPv6-incapable VPN tunnel — ptb's default HTTP/2 client picks Telegram's IPv6 with no IPv4 fallback. Fixed by binding the bot to IPv4 (`_ipv4_request`, `local_address="0.0.0.0"`). Same wave: `Notifications.push` mirrors to loguru; bot-token log leak masked. Spec: commit `0ba11ff`. |
| — | render_tab_refresh | done | Render sub-tab brought to the Node/Share standard: small-font label-column rows (`ui_primitives.label_row`/`row_label`), ghost-style resolution presets, `primary_button` Render (disabled until a file is chosen), preview drawn left of the controls with its box height measured from the controls group so the Render button's bottom aligns with the preview-box bottom (letterboxed, any aspect). All numeric sliders -> drags: `duration_slider` -> `duration_drag` (`drag_float`); FPS/Duration flipped to drags. Spec: commit `a716d24`. |
| — | node_tab_polish | done | Node-tab uniform rows flipped to a left label-column layout (chip / dim name / capped control); two-line header grid (matching-uniform name under the resolution combo, jitter-free); separator cascade → calm gaps; resolution string de-pipe'd to `WxH (ratio) - name`. Fixed a text-uniform crash (full `char[N]` left no null terminator → `unicode_to_str` raised); `used/cap` shown on text/array rows. `small_caption` primitive + `theme.SIZE.UNIFORM_*` tokens. Spec: commit `62a6644`. |
| — | editor_unfocused_dim | done | Code editor dims (`style.Alpha`) when it lacks keyboard focus; `EDITOR_UNFOCUSED_ALPHA` token in `theme.py`. Spec: commit `225076a`. |
| — | hotkeys_extraction | done | Pulled the hotkey block out of `ui.py` into `hotkeys.py::process_hotkeys(app)`. Spec: commit `c7d6359`. |
| — | smoke_test | done | `scripts/smoke.py` + `make smoke`: 200 headless frames of `update_and_draw`, asserts popup-mutex + `current_node_id` invariants. Spec: commit `92f221a`. |
| — | pyright_blocking | done | Dropped `\|\| true` from the pyright pre-commit hook — typecheck now blocks on failure; repo at 0 errors. Spec: commit `d1fdf65`. |
| — | devflow_scaffold | done | Created `CLAUDE.md` cold-start chain + `ai_docs/{dev_flow,todo,conventions}.md` + `/sanitize` skill + `Makefile`. Spec: commit `cd430d1`. |
