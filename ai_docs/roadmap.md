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

<!-- As of 2026-06-09. -->
**ProjectSession extraction (025) + dogfood-harness scaffold (026) landed on `dev`. The harness CONSTRUCTS + LOADS + RENDERS live on the Pi (first real runtime proof the 025 headless core works) — only a live LLM turn is unrun, blocked on the maintainer's `OPENROUTER_API_KEY` (none on this Pi). Then: drive real dogfood scenarios, the 025 `make run` pass, ship.**

- **NOW — ProjectSession extraction (025) landed, 4 green commits, post-impl reviewed.** Extracted the
  headless project + copilot CORE out of `App` into `project_session.py` (no imgui/glfw imports, no
  window/context creation): paths/nodes/app_state/lib-index/stores/integrations + the whole copilot
  cluster (`CopilotSession`/`CopilotBackend`/`RevertExecutor`). `App` now owns one `self.session` +
  forwards via `@property`; UI side effects of a core mutation ride injected `on_*` callbacks (chosen
  over return-values because the reactions fire mid-copilot-turn on the worker-drain thread with `App`
  off the stack). One intended behavior delta: the copilot's mid-turn "Node saved" toast is dropped
  (the toast now lives in the user-path `App.save_ui_node` forwarder only). This is the long-deferred
  `app.py` split + the foundation for the dogfood harness. Spec:
  `025_project_session_extraction.md`. **Type-checked + GL-free tests green (212/24 skip) + post-impl
  reviewed; the `App.__init__`/`_init`/`release` runtime path is un-runtime-tested on the display-less
  Pi → a maintainer `make run` pass is the remaining gate.**
- **NOW — copilot dogfood harness (026) SCAFFOLD landed + verified on the Pi.** `scripts/dogfood.py`:
  `DogfoodHarness.create()` builds a real headless `ProjectSession` on a standalone EGL context (no
  App/glfw), loads a tmp project (templates seeded), and `render(size=400)` produces a real 400×400 PNG
  via the actual `render_image` capability (bridge-pumped on the context-owning thread — the worker-marshal
  pattern, NOT a sync patch, which would corrupt the GL context). Construction + load + render + the
  human-eyeball loop all verified live (the UV-Mango gradient rendered + read back correctly). This is the
  FIRST runtime proof the 025 headless core works. `send()` + `drive_until_idle()` (the real-LLM turn) are
  code-complete but UNRUN — no `OPENROUTER_API_KEY` on this Pi. Spec: `026_copilot_dogfood_harness.md`.
- **BLOCKED — the live dogfood run needs the maintainer's `OPENROUTER_API_KEY`** (billed; absent here). The
  5 weak-spot scenario checklists are WRITTEN + ready to drive (`ai_docs/scenarios/`: visual-blindness,
  wrong-node targeting, compile-thrash, circle/square/morph, grep→read_lib+uniforms). To run:
  `export OPENROUTER_API_KEY=…` then drive a scenario via the REPL (see `ai_docs/scenarios/README.md`).
- **Also landed this wave:** `make smoke` now self-skips on a no-GPU-window box (its `_has_gpu_window`
  probe) instead of crashing on the Pi.
- **Still pending (separate from 025):** copilot turn-rollback (030) awaits its own maintainer live pass
  (the Revert glyph + modal are headless-unverifiable); the wrong-node TARGETING prompt fix (`todo.md`
  trigger) is filed not started.
- **THEN — ship.** `master` stays at `v0.12.1`; the full copilot stack (020-024 + 030 + 025) sits
  ship-shaped on `dev`, unshipped, pending the live passes + the dogfood harness. Awaiting explicit go.
- **Deferred (each gates a FUTURE round only on a NEW failure class in a DIFFERENT session):** the lazy
  tool-catalogue (its ~16-tool threshold FIRED), the structural shader view (020·27), reasoning-notes /
  intent-carryover guard (DE-RISKED by 029), a broken-compile circuit-breaker, machine-readable render
  feedback, `bind_media`/`undo_edit` — all in `todo.md`.
- **Trace-gated (NOT now):** semantic-editing (rename/outline/add_uniform), GLSL-aware grep, uniforms-in-
  tree, eager-recompile for lib edits — each only if a trace shows the current tools struggling (none does).
  The visual-variant-optimizer (render N variants as clickable chat boxes) is the big future feature.
  Further `app.py` splits (node-CRUD, path-properties, picker forwards) judged net-negative — `todo.md`.
- **No open BLOCKERs.** Cosmetic nav tails parked in `todo.md`, trigger-gated.

## Features

| # | Name | Status | Brief |
|---|---|---|---|
| 026 | copilot_dogfood_harness | partial | Hand-driven headless driver to dogfood the copilot ENGINE: `scripts/dogfood.py` `DogfoodHarness` builds a real `ProjectSession` on a standalone EGL context (no App/glfw), drives `session.copilot` turn by turn (worker + bridge-pump, the App-mirror — NOT a sync patch), renders 400×400 PNGs, eyeballs them (judge = human reading the trace + images, no code assertions). Scaffold + load + render + eyeball-loop VERIFIED live on the Pi (first runtime proof of 025's headless core). `send`/`drive_until_idle` (real-LLM turn) code-complete but UNRUN — needs `OPENROUTER_API_KEY` (absent on this Pi). Probes the known weak spots (visual-blindness, wrong-node targeting, compile-thrash). Spec: `ai_docs/features/026_copilot_dogfood_harness.md`. |
| 025 | project_session_extraction | partial | Extracted the headless project + copilot CORE out of `App` into `project_session.py` (no imgui/glfw imports, no window/context creation), so the dogfood harness (026) can drive the real copilot engine without `App`/glfw. 4 green commits: C1 pure-core state + `@property` forwarders; C2 GL-free project load (`load`/`seed_starter_node`); C3 copilot cluster (`CopilotSession`/`CopilotBackend`/`RevertExecutor`) + the 4 UI-tail methods moved whole; C4 UI-tail → injected `on_*` callbacks (chosen over return-values: reactions fire mid-turn on the worker-drain thread, App off-stack) + the mid-turn "Node saved" toast dropped (now user-path-only). The long-deferred `app.py` split. Type-checked + GL-free tests green + post-impl reviewed; **awaiting a maintainer `make run` pass** (the `App.__init__`/`_init`/`release` runtime path is un-runtime-testable on the display-less Pi). Spec: `ai_docs/features/025_project_session_extraction.md`. |
| 023 | app_refinement_wave | done | Pure-shape refactor of the overgrown `app.py` (~41% was the copilot backend). Three green-gated commits: C2 extracted the 49 `_copilot_*` methods into `copilot/backend.py` (`CopilotBackend`, the `ShaderLibFileManager` idiom — explicit deps + injected getters/callbacks, never imports `App`; `_build_copilot_capabilities` constructs + binds it; working-set/batch state stays App-owned via accessors; `app.py` −~1250 lines); C3 collapsed the four modal popup booleans into a `PopupState` enum (structural mutex); C4 fixed a latent Esc bug the enum surfaced (Esc now closes the emoji + lib pickers — the one behavior change). Two design-audit swarms + pre/post-impl review. Spec: `ai_docs/features/023_app_refinement_wave.md`. |
| 022 | copilot_chat_persistence | done | The copilot conversation is tied to its project + restored on reopen (was memory-only, dropped on switch/exit). `copilot/persistence.py` (`ConversationStore`, versioned + fail-soft like `app_state.json`) persists both the UI render messages and the LLM history + usage to `project_dir/copilot/conversation.json`; `CopilotSession.save_conversation`/`load_conversation`; App saves the outgoing project's chat in `release()` (top of `_init` + shutdown) and loads the incoming one after `reset_conversation`; a `begin_disabled`-during-turn clear button archives to `copilot/archive/`. Folded the trace-bleed deferral: the orphaned-history append is guarded on `_cancel`, and the worker-is-idle invariant (020·15's `open_project` gate) closes the trace-bleed window (the residual `_ensure_open` structural weakness re-scoped in `todo.md`). Spec: `ai_docs/features/022_copilot_chat_persistence.md`. |
| 021 | logging_refactor | done | Three-stream logging: a terse INFO+ console, a rotated DEBUG+ file (`logs/`) that is a strict superset, and a full-fidelity copilot transcript (`copilot_traces/copilot_<slug>_<stamp>.transcript` — human/agent-readable plain text replacing the old jsonl). `shaderbox/logging_setup.py` configures all loguru sinks once; `LoggingConfig` holds the internal config (console/file levels, rotation, retention, trace-retention=20); the 118-call logger survey audited 24 modules with ~37 calls shifted (lifecycle→DEBUG, user events stay INFO, fallback-config ERROR→WARNING); trace gains a transcript renderer + `tool_args_parse_error` event + mtime-pruned retention. Maintainer-verified live (console terse, transcript readable cold). Spec: `ai_docs/features/021_logging_refactor.md`. |
| 020 | copilot_agent | in progress | In-app coding-copilot agent (free-form chat over OpenRouter; in-process compile-feedback is the differentiator). Scaffold + Slice 1 (edit/compile-feedback vertical) landed, runs, and verified end-to-end on a real read→edit turn: the `shaderbox/copilot/` package (capabilities / `LLMClient` / worker→main `CopilotBridge` / worker↔UI queues / chat `state` / `agent` loop / `prompt` / `trace`), the three current-node tools (`get_current_shader` / `edit_shader` / `get_compile_errors`), the OpenRouter stream + key/model in Settings, the editor lock, and a full per-session transcript trace. Default model `x-ai/grok-4-fast` (tool-call compatible, verified); the agent rejects tool-incompatible models. Slice-1 self-correction completed (`12_edit_robustness.md`): `edit_shader` whitespace near-miss hint (echoes exact bytes on a 0-match), enforced `max_edit_retries=3` (was dead config), and giveup/`max_iterations` cutoffs now surface as chat errors. The GLSL token matcher landed (`13_glsl_lexer.md`): `copilot/glsl_lex.py` (`glsl_lex` + `token_match`) replaces `edit_shader`'s byte-exact match with whitespace-invariant token-stream equality, so a whitespace-divergent `old_str` succeeds at the match layer (the slice-12 hint becomes the no-op fallback). Slice 2 landed (`14_slice2_line_editing.md`): two line-anchored editing tools (`replace_lines`/`insert_after`, addressed by line numbers — the model quotes nothing) + the retry-cap widened to all mutating tools (the "what changed" apply excerpt was retired by 020·29 — the working-set scratchpad shows the whole post-edit source). Edit-safety landed (`15_edit_safety.md`): Half A — the editor is read-only + node-switch/create/delete/save/open-project are frozen while a turn runs (RETAINED); Half B — the content source-freshness guard — was RETIRED by 020·29 (the live per-iteration working-set rebuild makes line/content drift structurally impossible; a per-batch line-edit guard closes the within-batch residual). Cross-project tools (`16_cross_project_tools.md`) DONE + live-verified across 3 maintainer test sessions: the agent works the whole project (8 tools `read_shader`/`edit_shader`/`replace_lines`/`insert_after`/`set_uniform`/`create_node`/`grep`/`read_lib`, an always-in-prompt project map + lib catalogue, live-source lib creation). Post-test fixes: `create_node` returns its compile result; a GLSL-skeleton + engine-uniform-declaration conventions note; short 4-char node ids (no UUIDs in chat). Gate-UI wave (`17_gate_ui.md`) landed: the `GateChannel` body is wired (inline Yes/No confirm blocking the worker via the built bridge-mirror) + `delete_node` (always-gate, the "remove the last 3" trigger) + a Recover-from-trash button persisted across restart; decline continues the loop so the model comments. Render/publish wave (`18_render_publish_tools.md`) landed: 4 always-gated tools — `render_image`/`render_video` (to the project `renders/` dir via the bridge with `render_op_timeout_s`) + `publish_telegram`/`publish_youtube` (render-then-enqueue-then-await the exporter's terminal progress, all marshalled through the bridge); missing creds = a pre-gate guided handoff (a new `precheck` seam on `ToolDefinition`); the exporters gained public `is_connected()`/`publish()`. Telegram connect/pack wave (`19_credential_pack_tools.md`) landed: the `GateKind.CREDENTIAL` widget is built (a masked inline secret input — `set_telegram_token`, the token redacted to a prefix everywhere but the live store) + auto-link (the `AuthState.LINKING` floor + a connect-await) + full pack CRUD (list/select/create/delete, gated); the exporters gained public token/pack wrappers. Follow-ups: a `switch_node` tool (the copilot makes any shader current — publish acts on the current node ONLY, render takes an optional node id; conventions.md ## Design decisions pins the risk-scaled addressing rule), and a prompt fix so the agent treats `set_telegram_token`/pack CRUD as ITS capabilities (never deflects to Settings). UI/UX polish wave (020·20) landed: per-tool status + tool-result transcript lines (D1), ASCII glyph sanitization at three boundaries + a prompt nudge (D2, `copilot/text_render.py`), the two-phase "Rendering…" modal via a `MainThreadOp.defer` marker (D3), and the 5 low-severity correctness footguns (D4); a pre-impl review corrected the modal's threading (main-thread one-frame hold, not a worker-side two-phase). A separate audit-driven must-fix wave closed 5 silent-correctness items (history divergence, double-escape guard, comment-loss guard, array-uniform reject). STILL IN PROGRESS: a maintainer live pass on the stack, then ship; the lazy tool-catalogue (D5) + `delete_lib_file` + semantic-editing stay deferred/trace-gated, and a `bind_media`/`undo_edit` tool are parked scope decisions (`todo.md`). Spec: `ai_docs/features/020_copilot_agent/11_capability_wave_spec.md §16` + `12`–`20`. |
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
