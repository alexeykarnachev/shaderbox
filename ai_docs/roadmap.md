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

**As of 2026-06-05.**

- **DO NOT SHIP YET.** The copilot is NOT ship-ready (maintainer, explicit). `master` stays at `v0.12.1`;
  a large unshipped stack sits on `dev` (020 sub-waves 12→21 + feature 021 logging + 022 persistence) —
  intentional.
- **Audit + must-fix wave — DONE (uncommitted).** A holistic adversarial audit of the full copilot stack
  (verdict: structurally sound, ship-shaped) fixed 5 silent-correctness items: the mid-turn-exception
  history divergence + the sub-frame commit-ordering race (`session.py`), the `_unescape_double_escaped`
  marker guard (`agent.py`), the `token_match` interior-comment guard (`glsl_lex.py`), and the
  array-uniform `set_uniform` false-success (`app.py`).
- **020·20 UI/UX polish wave — DONE + headless-verified; awaits a maintainer live pass.** The ship gate.
  D1: per-tool status line + tool-result line reach the transcript (`AgentStatus`/`ChatState.status` +
  `AgentToolCard.result`). D2: ASCII glyph sanitization (`copilot/text_render.py`) at three boundaries
  (history-commit + Message-materialize + draw) + a prompt nudge — fixes the `?`-box on arrows/em-dashes
  the maintainer hit (the font lacks arrows; 1.92 dynamic atlas). D3: two-phase "Rendering…" modal (a
  `MainThreadOp.defer` marker holds the render op one frame so the cue paints before the freeze). D4: the
  5 low-severity footguns (retry-cap scoped to edit tools, unresolved-≠-stale, empty-handle guard,
  malformed-args cap, read_shader dedup + prefix missing-report). Spec: `20_ui_ux_polish.md`.
- **020·21 first-class chat widgets — DONE + headless-verified; awaits a maintainer live pass.** Tool
  results are now STRUCTURED OUTCOMES the engine renders as first-class chat buttons; the agent reports
  the FACT + is TOLD a widget was shown, raw URLs/paths NEVER reach the model (the live-session leak +
  dead-link fix). `ResultWidget` (`state.py`) rides the existing payload channel; publish/render
  producers emit terse facts + `open_url`/`open_path` widgets; open-only buttons (`ui_primitives`, no
  clipboard). Inline YouTube connect: a new `GateKind.CONFIG` renders the EXISTING `draw_config_ui`
  panel inline + a Cancel button (`tools/youtube.py`, `set_youtube_credentials`); persistence v3->v4.
  Spec: `21_chat_widgets_and_links.md`.
- **020·22 template library — DONE + headless-verified; awaits a maintainer live pass.** The copilot
  now SEES the shipped node templates (UV Mango / Media Input / Text Rendering) with descriptions in its
  prompt, so it picks the right one on intent ("render text" -> the SDF Text Rendering template); it
  READS + GREPS them via the EXISTING read_shader/grep with a `template:` address (one impl, mirrors
  `lib:`), INSTANTIATES via `create_node(template=...)`, and the default starter is just-a-template
  (no hardcoded special-case). Editable descriptions: shipped `node.json` default + a user sidecar
  (`templates_descriptions.py`); the node-creator popup is now a template-management center.
  `UINodeState.description` (forward-compat). Spec: `22_template_library.md`.
- **020·23 the untangle — DONE + headless-verified; awaits a maintainer live pass.** A live session
  exposed a tangled failure (a "Hello World" text shader thrashed to the iteration cap, then couldn't
  explain why). A brainstorm/review swarm traced it to ONE false premise: the 020·20 array-uniform
  reject (probe-disproved — moderngl raises on a bad array write, never silently corrupts), which
  blocked the agent's correct `set_uniform('u_text', codepoints)` and forced an illegal source edit.
  Fixes: `_coerce_uniform_value` now handles arrays (probe-pinned shapes — `set_uniform("u_text",
  "Hello\nWorld")` as a string just works, reusing the UI's `str_to_unicode`); the guard is deleted;
  full-turn history persistence (the agent sees its own tool trajectory next turn — `_commit_turn` now
  persists the assistant/tool tail, orphan-cleaned); prompt rules (value-vs-declaration + the
  no-default-init GLSL invariant + the "on 'I see nothing' don't re-assert, re-read" escalation);
  read_shader shows a chat SUMMARY (the user reads code in the editor); chat `no_move` + a per-message
  Copy. Spec: `23_uniform_history_chat_untangle.md`.
- **020·24 live-pass fixes — DONE + headless-verified; awaits a maintainer re-pass.** A maintainer live
  session (11 turns: build/animate/center a text shader, then publish to TG + YT Shorts) surfaced three
  harness defects, all fixed: (1) the "Rendering…" cue never visibly painted — `bridge.drain` cleared
  `_render_pending` at the top of the run frame, so the freeze frame showed no cue; now the flag survives
  the run frame so the cue covers the freeze. (2) the chat window was un-draggable in ALL layouts — a
  false in-code premise ("title bar still drags under `no_move`"; it does not) froze even FREE; `no_move`
  is now layout-conditional (forced presets only). (3) the unbounded-history BLOCKER is RESOLVED —
  `build_messages` now trims whole leading turns once the request estimate exceeds `max_input_tokens`
  (the dead constant is wired up), keeping the last 4 turns and never orphaning a tool_call_id.
- **020·25 live-pass round 2 — DONE + headless-verified; awaits a maintainer re-pass.** A second live
  session (build/animate/publish a text shader) exposed two more: (1) the render cue STILL didn't show —
  the 020·24 fix was correct but `_copilot_publish` called the bridge WITHOUT `defer=True` (only the bare
  `render_image`/`render_video` deferred), and the session only ever PUBLISHED, so the cue path was never
  taken; publish now defers too. (2) a lib resolver bug forced the agent off the library entirely — a
  `return SB_palette_sunset(...)` call read as a user DEFINITION of `SB_palette_sunset` (`USER_FN_DEF_RE`
  matched `<word> SB_foo(`, taking `return` as the type), so the lib function was treated as shadowed,
  never spliced, and failed to compile as "undefined"; the regex now requires the full `(...) {` body so a
  keyword-preceded call can't masquerade as a def. Both pinned by regression tests.
- **020·26 Render-tab render cue — DONE + maintainer-verified live (the cue now shows).** The
  "Rendering…" cue only ever covered the COPILOT path; the Render-tab "Render" button rendered inline on
  the click frame (froze the editor with no indication — the reported symptom). The button now sets a
  deferred `app.render_request` closure that `update_and_draw` runs AFTER the frame's swap, with
  `gl.finish()` between the swap and the encode. The real root cause (three prior scheduling attempts
  failed): `glfwSwapBuffers` only QUEUES the buffer — on X11 a queued cue frame never composites while the
  main thread blocks for seconds inside the encode, so `gl.finish()` is required to force the present before
  the freeze (`conventions.md ## Known quirks`). The COPILOT render path runs its encode at the top of the
  frame (before the swap) and so has the same latent invisibility — tracked in `todo.md`.
- **020·28 prompt tier architecture (NL-only history + block constructor) — DONE + headless-verified;
  awaits a maintainer live pass.** A cost/structure investigation (measured: the SAME ~4,850-tok shader
  source appeared 10x in one session's history; ~25k avg / 230k peak input tok/turn; OpenRouter caching
  IS working ~6x within-turn) plus an intent-driven design swarm reframed the fix: history is now
  NATURAL-LANGUAGE ONLY — `_commit_turn` collapses each turn to one engine-derived `TurnSummary` (reply +
  action ledger + referenced nodes), the full source the agent reads is NEVER persisted (re-fetched live).
  Prompt assembly moved from a flat string-concat to a block constructor (`PromptBlock`/`Volatility`/
  `build_prompt`). Per-read history contribution drops ~266x (~6,115 -> ~23 tok). Spec:
  `28_prompt_tier_architecture.md`. Spec `27_structural_shader_view.md` (the read_shader STRUCTURE block —
  the shader-representation pass) is FILED but DEFERRED to after this lands per the maintainer's ordering.
- **NEXT — a maintainer live pass on the 020 stack (020·19→28), then ship.** Specifically test the copilot
  over a multi-turn edit session (animate -> "not so much" -> "no, that's wrong" -> publish): confirm the
  NL summary resolves the dial-back, the correction, and doesn't re-publish on "continue". Still deferred:
  the lazy tool-catalogue (its ~16-tool threshold FIRED), the structural shader view (020·27), the
  within-turn read de-dup + reasoning-notes scratchpad (020·28 follow-ups), `bind_media`/`undo_edit` — all
  in `todo.md`.
- **Trace-gated (NOT now):** semantic-editing (rename/outline/add_uniform), GLSL-aware grep, uniforms-in-
  tree, eager-recompile for lib edits — each only if a trace shows the current tools struggling (none does).
  The visual-variant-optimizer (render N variants as clickable chat boxes) is the big future feature.
  Scope decisions parked for the maintainer: a `bind_media` tool, an `undo_edit` tool (`todo.md`).
- **No open BLOCKERs.** Cosmetic nav tails parked in `todo.md`, trigger-gated.

## Features

| # | Name | Status | Brief |
|---|---|---|---|
| 022 | copilot_chat_persistence | done | The copilot conversation is tied to its project + restored on reopen (was memory-only, dropped on switch/exit). `copilot/persistence.py` (`ConversationStore`, versioned + fail-soft like `app_state.json`) persists both the UI render messages and the LLM history + usage to `project_dir/copilot/conversation.json`; `CopilotSession.save_conversation`/`load_conversation`; App saves the outgoing project's chat in `release()` (top of `_init` + shutdown) and loads the incoming one after `reset_conversation`; a `begin_disabled`-during-turn clear button archives to `copilot/archive/`. Folded the trace-bleed deferral: the orphaned-history append is guarded on `_cancel`, and the worker-is-idle invariant (020·15's `open_project` gate) closes the trace-bleed window (the residual `_ensure_open` structural weakness re-scoped in `todo.md`). Spec: `ai_docs/features/022_copilot_chat_persistence.md`. |
| 021 | logging_refactor | done | Three-stream logging: a terse INFO+ console, a rotated DEBUG+ file (`logs/`) that is a strict superset, and a full-fidelity copilot transcript (`copilot_traces/copilot_<slug>_<stamp>.transcript` — human/agent-readable plain text replacing the old jsonl). `shaderbox/logging_setup.py` configures all loguru sinks once; `LoggingConfig` holds the internal config (console/file levels, rotation, retention, trace-retention=20); the 118-call logger survey audited 24 modules with ~37 calls shifted (lifecycle→DEBUG, user events stay INFO, fallback-config ERROR→WARNING); trace gains a transcript renderer + `tool_args_parse_error` event + mtime-pruned retention. Maintainer-verified live (console terse, transcript readable cold). Spec: `ai_docs/features/021_logging_refactor.md`. |
| 020 | copilot_agent | in progress | In-app coding-copilot agent (free-form chat over OpenRouter; in-process compile-feedback is the differentiator). Scaffold + Slice 1 (edit/compile-feedback vertical) landed, runs, and verified end-to-end on a real read→edit turn: the `shaderbox/copilot/` package (capabilities / `LLMClient` / worker→main `CopilotBridge` / worker↔UI queues / chat `state` / `agent` loop / `prompt` / `trace`), the three current-node tools (`get_current_shader` / `edit_shader` / `get_compile_errors`), the OpenRouter stream + key/model in Settings, the editor lock, and a full per-session transcript trace. Default model `x-ai/grok-4-fast` (tool-call compatible, verified); the agent rejects tool-incompatible models. Slice-1 self-correction completed (`12_edit_robustness.md`): `edit_shader` whitespace near-miss hint (echoes exact bytes on a 0-match), enforced `max_edit_retries=3` (was dead config), and giveup/`max_iterations` cutoffs now surface as chat errors. The GLSL token matcher landed (`13_glsl_lexer.md`): `copilot/glsl_lex.py` (`glsl_lex` + `token_match`) replaces `edit_shader`'s byte-exact match with whitespace-invariant token-stream equality, so a whitespace-divergent `old_str` succeeds at the match layer (the slice-12 hint becomes the no-op fallback). Slice 2 landed (`14_slice2_line_editing.md`): two line-anchored editing tools (`replace_lines`/`insert_after`, addressed by the line numbers `get_current_shader` shows — the model quotes nothing) + a "what changed" apply-feedback excerpt on every mutating edit + the retry-cap widened to all mutating tools. Edit-safety landed (`15_edit_safety.md`): the editor is read-only + node-switch/create/delete/save/open-project are frozen while a turn runs, and a source-freshness guard rejects any edit whose `(node_id, content)` moved since the agent last read it this turn (stale-rejects don't count toward the retry cap). Cross-project tools (`16_cross_project_tools.md`) DONE + live-verified across 3 maintainer test sessions: the agent works the whole project (8 tools `read_shader`/`edit_shader`/`replace_lines`/`insert_after`/`set_uniform`/`create_node`/`grep`/`read_lib`, an always-in-prompt project map + lib catalogue, per-node freshness keying, live-source lib creation). Post-test fixes: `create_node` returns its compile result; a GLSL-skeleton + engine-uniform-declaration conventions note; short 4-char node ids (no UUIDs in chat). Gate-UI wave (`17_gate_ui.md`) landed: the `GateChannel` body is wired (inline Yes/No confirm blocking the worker via the built bridge-mirror) + `delete_node` (always-gate, the "remove the last 3" trigger) + a Recover-from-trash button persisted across restart; decline continues the loop so the model comments. Render/publish wave (`18_render_publish_tools.md`) landed: 4 always-gated tools — `render_image`/`render_video` (to the project `renders/` dir via the bridge with `render_op_timeout_s`) + `publish_telegram`/`publish_youtube` (render-then-enqueue-then-await the exporter's terminal progress, all marshalled through the bridge); missing creds = a pre-gate guided handoff (a new `precheck` seam on `ToolDefinition`); the exporters gained public `is_connected()`/`publish()`. Telegram connect/pack wave (`19_credential_pack_tools.md`) landed: the `GateKind.CREDENTIAL` widget is built (a masked inline secret input — `set_telegram_token`, the token redacted to a prefix everywhere but the live store) + auto-link (the `AuthState.LINKING` floor + a connect-await) + full pack CRUD (list/select/create/delete, gated); the exporters gained public token/pack wrappers. Follow-ups: a `switch_node` tool (the copilot makes any shader current — publish acts on the current node ONLY, render takes an optional node id; conventions.md ## Design decisions pins the risk-scaled addressing rule), and a prompt fix so the agent treats `set_telegram_token`/pack CRUD as ITS capabilities (never deflects to Settings). UI/UX polish wave (020·20) landed: per-tool status + tool-result transcript lines (D1), ASCII glyph sanitization at three boundaries + a prompt nudge (D2, `copilot/text_render.py`), the two-phase "Rendering…" modal via a `MainThreadOp.defer` marker (D3), and the 5 low-severity correctness footguns (D4); a pre-impl review corrected the modal's threading (main-thread one-frame hold, not a worker-side two-phase). A separate audit-driven must-fix wave closed 5 silent-correctness items (history divergence, double-escape guard, comment-loss guard, array-uniform reject). STILL IN PROGRESS: a maintainer live pass on the stack, then ship; the lazy tool-catalogue (D5) + `delete_lib_file` + semantic-editing stay deferred/trace-gated, and a `bind_media`/`undo_edit` tool are parked scope decisions (`todo.md`). Spec: `ai_docs/features/020_copilot_agent/11_capability_wave_spec.md §16` + `12`–`20`. |
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
