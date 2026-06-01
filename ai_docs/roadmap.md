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

**As of 2026-06-01.**

- **Shipped: `v0.12.1`** (feature 019). Live on itch.io. Unshipped commits on `dev` (020 copilot wave +
  021 logging + 020·12 edit-robustness + 020·13 lexer) — `master` NOT advanced; do not ship without an
  explicit ask.
- **020 copilot agent — Slice 1 DONE + VERIFIED + robustness-hardened + the matcher widened.** The
  edit/compile-feedback vertical works end-to-end; default model `x-ai/grok-4-fast`. Slice-1
  self-correction (`12_edit_robustness.md`: retry cap, chat cutoffs, near-miss hint) maintainer-verified
  live. **020·13 lexer JUST LANDED** (step 1 of the maintainer's 2-step sequence): `edit_shader` now
  matches `old_str` by GLSL TOKEN-STREAM equality (`copilot/glsl_lex.py` — `glsl_lex` + `token_match`),
  so a whitespace-divergent copy *succeeds* at the match layer instead of needing the slice-12 re-copy
  hint (now the no-op fallback). Pre+post review converged; the post-impl reviewer caught a unicode-digit
  assert-crash (fixed). 119 tests green; maintainer-verified live (a multi-line indented edit applied
  first-try, no spiral/hint — though the model copied bytes exactly, so the whitespace-rescue path is
  owned by the unit tests, not the live run). Spec: `13_glsl_lexer.md`.
- **021 logging refactor — DONE + maintainer-verified live.** Three streams: terse INFO+ console, rotated
  DEBUG+ file (`logs/`, strict superset), full-fidelity copilot **transcript** (`copilot_traces/
  copilot_<slug>_<stamp>.transcript`, human/agent-readable, replaced jsonl). Spec: `021_logging_refactor.md`.
- **NEXT — Slice 2, token-level editing tools** (step 2 of the 2-step sequence, now unblocked by the
  lexer). On top of `glsl_lex`: line-anchored editing (`replace_lines`/`insert_after`, so the model never
  reproduces anchor whitespace) + concise "what changed" apply-feedback. Scope deliberately KEPT TIGHT —
  no semantic-editing suite (rename/outline/add_uniform deferred until a trace shows need). Build-then-spec
  or spec-first per the maintainer's call.
- **Feature 022 — copilot chat persistence (SPECCED, not started).** Conversation tied to its project +
  restored on reopen + clear-chat button; `project_dir/copilot/conversation.json`. Independent of Slice 2;
  sequence it whenever. Spec: `022_copilot_chat_persistence.md`.
- **No open BLOCKERs.** Cosmetic nav tails + a deferred copilot-trace cross-project bleed (folds into
  022's worker-quiesce) parked in `todo.md`, all trigger-gated.

## Features

| # | Name | Status | Brief |
|---|---|---|---|
| 022 | copilot_chat_persistence | pending | The copilot conversation tied to its project + restored on reopen (today it's memory-only, dropped on switch/exit). Persists both the UI render messages and the LLM history to `project_dir/copilot/conversation.json` (versioned, fail-soft like `app_state.json`), restores on project open, saves at turn-completion/switch/shutdown, and adds a user-facing clear-chat button that archives to `copilot/archive/`. Builds on 021's on-disk cut. Spec: `ai_docs/features/022_copilot_chat_persistence.md`. |
| 021 | logging_refactor | done | Three-stream logging: a terse INFO+ console, a rotated DEBUG+ file (`logs/`) that is a strict superset, and a full-fidelity copilot transcript (`copilot_traces/copilot_<slug>_<stamp>.transcript` — human/agent-readable plain text replacing the old jsonl). `shaderbox/logging_setup.py` configures all loguru sinks once; `LoggingConfig` holds the internal config (console/file levels, rotation, retention, trace-retention=20); the 118-call logger survey audited 24 modules with ~37 calls shifted (lifecycle→DEBUG, user events stay INFO, fallback-config ERROR→WARNING); trace gains a transcript renderer + `tool_args_parse_error` event + mtime-pruned retention. Maintainer-verified live (console terse, transcript readable cold). Spec: `ai_docs/features/021_logging_refactor.md`. |
| 020 | copilot_agent | in progress | In-app coding-copilot agent (free-form chat over OpenRouter; in-process compile-feedback is the differentiator). Scaffold + Slice 1 (edit/compile-feedback vertical) landed, runs, and verified end-to-end on a real read→edit turn: the `shaderbox/copilot/` package (capabilities / `LLMClient` / worker→main `CopilotBridge` / worker↔UI queues / chat `state` / `agent` loop / `prompt` / `trace`), the three current-node tools (`get_current_shader` / `edit_shader` / `get_compile_errors`), the OpenRouter stream + key/model in Settings, the editor lock, and a full per-session transcript trace. Default model `x-ai/grok-4-fast` (tool-call compatible, verified); the agent rejects tool-incompatible models. Slice-1 self-correction completed (`12_edit_robustness.md`): `edit_shader` whitespace near-miss hint (echoes exact bytes on a 0-match), enforced `max_edit_retries=3` (was dead config), and giveup/`max_iterations` cutoffs now surface as chat errors. The GLSL token matcher landed (`13_glsl_lexer.md`): `copilot/glsl_lex.py` (`glsl_lex` + `token_match`) replaces `edit_shader`'s byte-exact match with whitespace-invariant token-stream equality, so a whitespace-divergent `old_str` succeeds at the match layer (the slice-12 hint becomes the no-op fallback). Slice 2 (line-anchored editing tools) next, unspecced. Spec: `ai_docs/features/020_copilot_agent/11_capability_wave_spec.md §16` + `12_edit_robustness.md` + `13_glsl_lexer.md`. |
| 019 | keyboard_navigation | done | The focus/nav layer (018's deferred half): app-wide `nav_enable_keyboard` + a two-level focus model — `Ctrl+`` ` `` cycles 3 regions (editor/grid/panel, region-confined via `no_nav_inputs`, active region shown by a live-focus accent outline), `Ctrl+1/2/3` jump the inner Node/Render/Share tab; editor is a permanent focus-stop; grid cells are nav-reachable `selectable`s; 018 bare-arrow node-prev/next removed. The polish wave added a selection-vs-accent color split (fixed `COLOR.SELECT`) + a theme-portability invariant, Ctrl+Tab suppression, glfw-layer Esc swallowing, `nav_flattened` uniforms. Maintainer-verified. Spec: `ai_docs/features/019_keyboard_navigation.md`. |
| 018 | keyboard_control | done | The command layer: a central `commands.py` registry drives rebindable chord shortcuts + an opt-out cheatsheet overlay + an `imgui_command_palette` (Ctrl+Shift+P); dispatch split pre-frame/in-frame; rebindings persist diff-from-default on `UIAppState`. The focus/navigation layer (nav widget-traversal + tab-cycling) was split out to a `todo.md` deferral. Spec: `ai_docs/features/018_keyboard_control.md`. |
| 017 | structure_reorg | done | Domain-separation refactoring wave (no behavior change): `lib_*`→`shader_lib/` package + total rename, shader_lib split into index/resolver/parser, `lib_picker`→package, `util.py`→`shader_errors.py`+`editor_types.py`, `ui_models` de-tangled from UI, exporters/ tidy, App shader-lib CRUD→`ShaderLibFileManager`. `ui/`+`render/` packages rejected. Spec: `ai_docs/features/017_structure_reorg.md`. |
| 016 | lib_file_management | done | Unified tree+preview lib picker with right-click context menus for create / rename / delete (armed-confirm, file or recursive subdir) / reveal; `.trash/` filter shared by `LibIndex.build` and the mtime watcher; `Library` menu in the main menu bar. Spec: `ai_docs/features/016_lib_file_management.md`. |
| 015 | shader_library | done | Auto-resolved GLSL helper library — type `SB_perlin_noise_3(...)`, host splices declarations + topo-sorted preamble; `Ctrl+P` picker with fuzzy search + body preview. Spec: `ai_docs/features/015_shader_include_library.md`. |
| 014 | compile_unit_refactor | done | Pure-shape refactor introducing `CompileUnit` + `EditorSession` to prepare for #include support. Spec: `ai_docs/features/014_compile_unit_refactor.md`. |
| 013 | authoring_feedback_loop | done | Tighter write→compile→fix loop with click-to-jump error strip + bidirectional uniform↔code bridge. Spec: `ai_docs/features/013_authoring_feedback_loop.md`. |
| 012 | youtube_export | done | Second concrete exporter: YouTube upload (long-form + Shorts) via user-owned OAuth. Spec: `ai_docs/features/012_youtube_export.md`. |
| — | template_polish | done | Authored template order + merged Image/Video into Media Input + text-shader polish. Spec: commit `6b474f5` + `f44cf82`. |
| — | preview_cell_consolidation | done | Unified node-grid + sticker-carousel cell draw into `preview_cell` primitive. Spec: commit `44b97d0`. |
| — | node_delete_crash_fixes | done | Two delete-path crashes fixed (mid-iteration mutation + trash-dir collision). Spec: commit `1ab5d56` + `a386713`. |
| 011 | ui_library_consolidation | done | Post-010 refactor: de-leaked Telegram tokens, exporter seam cleanup, `ui_utils.py` split, `tests/` suite landed. Spec: `ai_docs/features/011_ui_library_consolidation.md`. |
| 010 | outlet_render_rework | done | Shared `RenderPreset` drives native-size rendering; Telegram panel redesigned; 4-tier button system + `imgui-ui` skill produced. Spec: `ai_docs/features/010_outlet_render_rework.md`. |
| 009 | integrations_rework | done | Telegram UX collapse: token+Connect in Settings, derived sticker-set names, share-tab pack CRUD, monochrome emoji picker. Spec: `ai_docs/features/009_integrations_rework.md`. |
| 008 | uniform_input_shapes | done | Per-uniform input-shape pill + per-node uniform sort + engine-uniform collapse + accent tab styling. Spec: `ai_docs/features/008_uniform_input_shapes.md`. |
| 007 | release_pipeline_hardening | done | `make release VERSION=` (semver bump + tag), `build.sh` gated on check+smoke + clean tree (`--allow-dirty`), `.zip` both platforms, Windows+Ubuntu CI, `BUILDING.md`. Spec: `ai_docs/features/007_release_pipeline_hardening.md`. |
| 006 | inline_glsl_editor | done | Syntax-highlighted inline GLSL editor in the main window's left split; Ctrl+S saves + hot-reloads; visual-options popup; replaced the external-editor mechanism. Spec: `ai_docs/features/006_inline_editor.md`. |
| 005 | ui_redesign_foundation | partial | Gruvbox theme + full color/size/spacing token sweep through `theme.py`. The wide-screen layout half was reverted to the feature-004 shape; only the theme survived. Spec: `ai_docs/features/005_ui_redesign_foundation.md`. |
| 004 | imgui_bundle_migration | done | Full pyimgui → imgui-bundle migration; adopted the `imgui_ctx` context-manager idiom + `portable_file_dialogs`; stripped the 8 imgui `# type: ignore` markers. Spec: `ai_docs/features/004_imgui_bundle_migration.md`. |
| 003 | modelbox_removal | done | Wiped all ModelBox integration (HTTP client, settings UI, app-state fields, `requests` dep); app_state migration gen 3. Spec: `ai_docs/features/003_modelbox_removal.md`. |
| 002 | ui_widgets_extraction | done | Three-layer split: `app.py` (state) / `ui.py` (orchestrator) / `widgets`+`popups`+`tabs` (pure draw fns); `ui.py` 1508 → 294. Spec: `ai_docs/features/002_ui_widgets_extraction.md`. |
| 001 | exporter_refactor | done | `exporters/` subpkg — `Exporter` ABC + `RenderedArtifact` value type + registry; Telegram ported to own worker thread + asyncio loop + sticker UI. Spec: `ai_docs/features/001_exporter_refactor.md`. |
| — | telegram_ipv4_fix | done | Bound the Telegram bot to IPv4 to fix uploads on IPv6-dead networks. Spec: commit `0ba11ff`. |
| — | render_tab_refresh | done | Render sub-tab brought to the Node/Share standard (label-column rows, ghost presets, drags-not-sliders). Spec: commit `a716d24`. |
| — | node_tab_polish | done | Node-tab uniform layout polished + text-uniform crash fix. Spec: commit `62a6644`. |
| — | editor_unfocused_dim | done | Code editor dims (`style.Alpha`) when it lacks keyboard focus; `EDITOR_UNFOCUSED_ALPHA` token in `theme.py`. Spec: commit `225076a`. |
| — | hotkeys_extraction | done | Pulled the hotkey block out of `ui.py` into `hotkeys.py::process_hotkeys(app)`. Spec: commit `c7d6359`. |
| — | smoke_test | done | `scripts/smoke.py` + `make smoke`: 200 headless frames of `update_and_draw`, asserts popup-mutex + `current_node_id` invariants. Spec: commit `92f221a`. |
| — | pyright_blocking | done | Dropped `\|\| true` from the pyright pre-commit hook — typecheck now blocks on failure; repo at 0 errors. Spec: commit `d1fdf65`. |
| — | devflow_scaffold | done | Created `CLAUDE.md` cold-start chain + `ai_docs/{dev_flow,todo,conventions}.md` + `/sanitize` skill + `Makefile`. Spec: commit `cd430d1`. |
