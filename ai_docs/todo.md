# TODO — blockers & deferrals

Known issues parked for later, each with a **Trigger** — the condition under which someone working in
the repo should pick it up. NOT "what's next" (that's `roadmap.md`'s Active-context banner); this is a
"heads up, here's a landmine / a deferred fix" registry. **Grep this file by `Trigger` before
starting work in an area.**

`[BLOCKER]` vs `[DEFERRAL]`: a BLOCKER is a silent foot-gun that fires, stops work, and must be
resolved before proceeding. A DEFERRAL is deferred cleanup that the triggering work absorbs as
in-scope when the trigger fires.

What makes a good Trigger: it fires at a moment that *demands attention* —
- ❌ "before the next release" / "when we have time" / "eventually" — passes silently, never fires.
- ✅ "next time you edit `telegram_provider.py`" / "when you grep for `_loop.run_until_complete`" /
  "before plan-locking any feature spec that lists `core.py` in `## Files touched`" / "first user
  report of `<observable symptom>`".

If you can't name a moment that demands attention, the deferral is wrong-shaped — resolve it now or
don't file it. When a deferral resolves, delete the entry in the SAME commit as the fix (git history
is authoritative — no "Resolved YYYY-MM-DD" headers).

<!-- Shape (per entry):
     ## [BLOCKER|DEFERRAL] <short title>
     - **Trigger:** <a concrete observable moment — file/code touch, count threshold, user complaint with a measurable surface, a specific upstream change>
     - <context: what / why / where>
     Entries are an index of triggers, not a home for designs. Resolved → delete in the resolving commit.
     State the CURRENT constraint, not a feature roll-call: "the observable bug is closed; the structural
     weakness remains" — never "Feature N CLOSED X / DE-RISKED by M". The why-it's-deferred is a
     present-tense fact, not past-tense dev history.
-->

---

## [DEFERRAL] copilot turn-rollback: a NEW mutating tool must register with the checkpoint
- **Trigger:** before plan-locking any feature that adds a new MUTATING copilot tool (one that
  writes a node / lib / uniform / creates / deletes) — it MUST capture into the active checkpoint
  (`backend.py::_capture_node`/`_capture_lib` pattern, keyed on `get_active_checkpoint`), or its
  change silently escapes the rollback net. ALSO: if `bind_media` lands, a media-binding turn needs
  its `media/`/`textures/` captured too (decision 9 snapshots only the two text files today).
- The rollback feature (020·30) is the one fragility of its log-what-you-touch model: capture is
  opt-in per mutation seam. The existing seams cover edit/replace/insert/set_uniform/create/delete/
  switch + lib edits. A new tool that mutates durable state without a capture call leaves an
  un-revertable change. Spec: `ai_docs/features/020_copilot_agent/30_turn_rollback.md` (decision 2).

## [DEFERRAL] copilot non-current-node-edit confirm-gate (the targeting structural backstop)
- **Trigger:** a trace shows the agent STILL mis-targeting a node by name-association AFTER the
  `HOW TO WORK` TARGETING rule (the prompt-level fix landed) — i.e. the prompt rule proved
  insufficient and a structural guard is warranted.
- The prompt-level fix (a TARGETING rule: bare/demonstrative reference = current node; name-match
  only when the user NAMES it or it can only be satisfied there; else ASK) landed in
  `copilot/prompt.py`. The bigger structural option the maintainer deferred deciding: a confirm-gate
  on any edit/switch that targets a NON-current node, so a wrong silent switch is caught even if the
  model ignores the prompt. Only build it if the prompt rule visibly fails in a fresh trace (per the
  "guard earns its place" convention — don't add standing structure for a transient model lapse).

## [DEFERRAL] copilot machine-readable render feedback (the visual-blindness affordance)
- **Trigger: FIRED and RESOLVED INTO 033** — the maintainer picked the ambient variant (render
  FACTS appended to every clean-compile edit result: ink %, bbox, luma grid; the enriched-results
  mechanism in `033_copilot_robustness_wave.md`). The standalone `inspect_render` TOOL variant is
  REJECTED (tool count must not grow; a lazy model wouldn't call it). What remains deferred here:
  the VLM-judge variant (a vision model critiques the render in words — judges aesthetics, not just
  presence). Trigger for THAT: after 033 lands, a harness-side pilot (driver feeds VLM verdicts back
  as user text) catches things the scalar facts miss.
- Original trigger (kept for context): the first recurrence (in a DIFFERENT session) of "the agent
  can't tell its change had no visual effect" — it reports success after a clean compile while the
  user sees no change (observed 2026-06-05: the cyrillic text never rendered; the agent claimed it
  did). OR when wiring the structural shader view (020·27).
- The copilot's ONLY correctness signal is the compiler, so a clean compile with unchanged output is
  indistinguishable from success. A lightweight `inspect_render(node?)` returning a few scalars (non-background
  pixel fraction, ink bounding box, a coarse luma grid) would let the agent detect "my change didn't take" —
  the one HONEST fix for the hallucinated-success class (the prompt-side ledger/honesty guards were CUT as
  overfit; this is the pipeline-intrinsic version). Large; spec before building. Source: the overfit-tribunal
  audit (visual-blindness lens).

## [DEFERRAL] copilot reasoning-notes scratchpad (the second PER_TURN block) — also the intent-carryover guard
- **Trigger:** the CORRECTION/COORDINATE intent regresses in practice (the agent's stated assumption is lost
  because it wasn't in its reply prose `text_buf`), OR the agent runs an UNREQUESTED action carried over from a
  prior turn and the user reports it (observed 2026-06-05: an unrequested `set_uniform(u_offset)` re-applied on a
  pure-delete turn — the "why did you touch positioning, I didn't ask" report), OR when reasoning/CoT is
  implemented for the copilot.
- The agent's assumption currently rides its verbatim reply (fragile — model-dependent). The robust home
  is a reasoning-notes scratchpad — a PER_TURN block member. This is cheap to add: the PER_TURN tier +
  per-iteration rebuild already exist (the working-set block is a live member), so reasoning-notes is a
  second member on the same machinery. Design: `28_prompt_tier_architecture.md ## Out of scope` +
  `29_working_set_scratchpad.md ## Out of scope`.

## [DEFERRAL] copilot cross-shader derived-edit memory ("do the same to C")
- **Trigger:** a trace shows the agent failing a "do the same to C" because it can't reconstruct what "the
  same" was even though the referenced shader's NAME survived in the turn-summary (fact 1).
- 020·28's NL summary records every node NAMED in a turn but not the source CONTENT an edit was derived from
  (source is never persisted). Recoverable via a live re-read of the named shader; if that proves
  insufficient, persist a richer derived-edit note. Design: `28_prompt_tier_architecture.md ## Out of scope`.

## [DEFERRAL] copilot structural shader view (the read_shader STRUCTURE block)
- **Trigger:** after the 020·28 tier structure is stable + maintainer-verified — the shader-representation
  retrospective pass the maintainer sequenced LAST. OR first time the agent re-declares an already-declared
  uniform again (the `u_time` C1038 class the spec targets).
- A concise STRUCTURE block (uniforms incl. declared-but-inactive + engine flag + decl line, defines,
  functions, entry) atop the read_shader result, so the agent sees what a shader already declares. Fully
  specced: `ai_docs/features/020_copilot_agent/27_structural_shader_view.md` (pre-impl reviewed, not yet
  implemented).

## [DEFERRAL] copilot render/publish cue likely invisible (no gl.finish before the freeze)
- **Trigger:** a maintainer reports the "Rendering…" cue is missing when the COPILOT renders/publishes
  (render_image / render_video / publish_telegram / publish_youtube), the same symptom the Render-tab
  button had — OR next time you touch `bridge.drain` / `drain_bridge` / the copilot render defer.
- The Render-tab fix (`ui.py`) presents the cue with `gl.finish()` AFTER the swap and BEFORE the encode
  (see `conventions.md ## Known quirks` — a queued buffer never composites while the main thread blocks).
  The copilot path runs its encode inside `drain_bridge()` at the TOP of the frame (in `update_and_draw`,
  before the cue is drawn and the buffer swapped) — so the cue frame it scheduled has the same "queued,
  never presented" problem and is probably invisible too. The `_render_pending` FLAG timing is correct;
  the present-before-freeze requirement is what's still unaddressed in the copilot path. Honest fix: route the copilot deferred render so its encode
  also runs after a swap + `gl.finish()` (or have the bridge's held-op run at the same post-swap point as
  `render_request`). Deferred — needs a live copilot render to confirm it's actually invisible first.

## [DEFERRAL] TraceLog._ensure_open can't distinguish "closed" from "never opened" (structural)
- **Trigger:** a NON-MODAL project switch lands (a "recent projects" menu / hotkey that doesn't go
  through the blocking `pfd` folder dialog), OR `open_project`'s `_copilot_busy_blocked` gate is ever
  removed/weakened — either re-opens the window where a worker can be mid-`run_turn` when
  `reset_conversation` swaps the trace. ALSO: a maintainer notices a copilot transcript containing a
  turn that belongs to a different project.
- The observable bug is CLOSED on all current UI paths (so this is a deferral, not a blocker): the
  orphaned-history half is guarded by the `_cancel.is_set()` check in `_run_one_turn` (it skips the
  commit on a cancelled turn), and the trace-bleed half can't fire because the `open_project` gate +
  the in-flight-gated clear button keep the worker IDLE when `reset_conversation` runs (no `run_turn`
  holds the old `TraceLog`). What REMAINS is the structural weakness, not yet hardened: `TraceLog._ensure_open` keys
  only on `self._fh is None`, so it cannot tell "never opened" from "closed" — if a worker ever DOES
  run past a `close()` (the trigger above), its next `tr.event` re-opens the just-closed file (append)
  and bleeds. Honest fix when triggered: a permanent-closed flag on `close()` that `_ensure_open`
  respects (the `_init`/release reopen path then needs its own re-arm). Deliberately deferred — adding
  it now is hardening a path the invariant already closes.

## [DEFERRAL] node-grid nav-cursor resets to cell 0 after Enter-selecting a node
- **Trigger:** maintainer finds the nav highlight jumping to the first cell ("New node") after
  picking a node with Enter annoying enough to want it fixed (it's cosmetic — focus stays in the grid,
  arrows still work, the node IS selected; only the visible nav-cursor position is wrong).
- Cause: `App.select_node` sets `region_focus_pending` so the grid re-grabs window focus (needed —
  the new node's editor auto-grabs focus on its first render, `TextEditor.render()` quirk; without
  the re-grab arrows die). But `set_next_window_focus()` resets imgui's nav cursor to the window's
  default item (cell 0). **Already tried + failed:** `set_item_default_focus()` on the selected
  cell's `selectable` inside `preview_cell` (didn't move the cursor — likely because the cell is
  wrapped in its own `begin_child` + `nav_flattened`, so "default item of the window" isn't the inner
  selectable). Next things to try: `set_keyboard_focus_here()` targeting the selected cell instead of
  a window re-focus; or `io.nav_id`/`set_nav_cursor` style direct nav-id targeting if the binding
  exposes it; or restructure the cell so the selectable is a direct child of the grid window (no
  per-tile `begin_child`) so default-focus lands. UN-HEADLESS-ABLE (nav-cursor position isn't
  readable from a headless run — every attempt needs a maintainer `make run` check).

## [DEFERRAL] node-grid 2D arrow adjacency (feature 019 follow-on)
- **Trigger:** maintainer reports arrow-nav across the node grid skips or misorders cells badly
  enough to be unusable (the cells are hand-wrapped `selectable`s in per-tile child windows —
  imgui's directional nav over that layout isn't reliably row/column-spatial).
- Honest fix is a real columns/clipper grid layout. Also parked: persisting the focused region across
  restart (transient by design today). See `ai_docs/features/019_keyboard_navigation.md` Out-of-scope.

## [DEFERRAL] SB_sd_text cell-cull (warm-render speedup) blocked by a quad-seam artifact
- **Trigger:** warm text-render cost becomes a real pain (live text preview on the Pi at speed, or
  a profiler shows SB_sd_text dominating on desktop) — then pick this up and crack the artifact.
- The data-driven glyphs (032, `scripts/gen_glyphs.py`) fixed the V3D first-draw codegen explosion
  (~20s -> ~1s) but warm render went 19ms -> ~176ms at 300px on the Pi (stroke tables read through
  memory per stroke per char per pixel). A per-char cell cull (evaluate the glyph only within
  ~1.2 cells, else use the cell-box SDF as a continuous lower bound) won back 176 -> 67ms BUT
  paints a faint full-frame DIAGONAL line (the fullscreen quad's triangle seam): with the divergent
  branch around the dynamic-indexed stroke loads, seam-quad helper pixels disagree and fwidth(d)
  spikes (920 px diff at 900x900, max 119/255; a value-continuous blend across the branch did NOT
  fix it). Shipped pixel-identical no-cull instead. Attack ideas when triggered: branchless cull
  (clamp the stroke loop count per char), screen-space-uniform AA width instead of fwidth(d) in
  SB_fill_aa, or per-row instead of per-char cull.

## [DEFERRAL] shader-library seed has no load mechanism (canonical copy sits in resources)
- **Trigger:** before the next `make run` of the APP on any box, or before any ship — decide +
  wire it (the desktop-session plan was CANCELLED, the Pi stays primary; dogfood sandboxes seed by
  copy per `/dogfood`, but the real app's lib root still starts empty on a fresh box).
- The canonical seed is `shaderbox/resources/shader_lib/` (in-repo, ships via `build.sh`); the live
  lib root `app_data_dir()/shader_lib` starts EMPTY on a fresh box. Interim: copy by hand (spec 032
  `## Resume on the desktop`). Decision pending: copy-on-first-run (user edits win; updates need a
  version/hash stamp — the shipped-default + user-sidecar convention) vs a second read-only search
  root (015 Non-goals deferred it "until a real need surfaces" — a shipped seed IS that need).

## [DEFERRAL] copilot chat renders Cyrillic replies as `??????` (ASCII sanitizer)
- **Trigger:** first real session where the maintainer chats with the copilot in Russian and the
  mangled replies actually annoy (the shader TEXT path is unaffected — set_uniform carries
  codepoints fine); OR next time you touch `copilot/text_render.py` / the D2 sanitize boundaries.
- Observed 2026-06-10 (032 dogfood exp-2): the agent's reply prose contained the word ШЕЙДЕР —
  rendered as `??????` in chat (the D2 ASCII glyph sanitization, 020·20). The chat font DOES carry
  the app's text; the sanitizer is the lossy step. Fix shape: allow Cyrillic through the sanitizer
  (the font supports it) or transliterate instead of `?`-replacing.

## [DEFERRAL] lib-author macro indirection for function dispatch
- **Trigger:** first time a lib author writes `#define HASH SB_hash3` (or similar) inside a lib
  file as a way to dispatch through the SB_-prefixed function, and finds their wrapper isn't
  pulled in transitively.
- `shader_lib.parser`'s `_extract_functions` (in `shader_lib/index.py`, via the parser regexes)
  builds `calls = set of identifiers in body`. If a lib
  function calls `HASH(x)` and `HASH` is a macro elsewhere expanding to `SB_hash3`, the regex
  sees `HASH` (no match in the index) and won't pull `SB_hash3` into the preamble. Convention
  (15.5 in the spec): lib files don't use `#define` for function dispatch — call the function
  directly. Document in `conventions.md ## Design decisions` if this becomes a real footgun.

## [DEFERRAL] export-from-selection (select function in editor → push to library)
- **Trigger:** when copy-pasting a function from a node shader into a hand-edited lib file
  becomes routine.
- Today the only way to add a function is `Ctrl+P` → "New library file" → paste manually.
  Spec when triggered.

## [DEFERRAL] multi-file editor — tab bar / file tree / split
- **Trigger:** when "back to node" + Ctrl+P feels insufficient — i.e., the user keeps 3+ lib
  files open in rotation and wants to switch between them without re-opening via the picker.
- Today the code pane shows ONE file at a time (the node's shader or one lib file). Design
  shape (pinned node-shader tab + closable lib tabs) is in `015_shader_include_library.md`
  decision 8.

## [DEFERRAL] sticker render is always t=0..N (no loop-offset / "which 3s")
- **Trigger:** first time a shader's interesting motion isn't at the start (a user wants a 3s
  window other than the first), OR when you next touch the render path in `tabs/share_state.py`
  (`render_for` / `render_to`) / `core.py` (`_render_video`).
- A shader loops infinitely but `_render_video` always renders frames `i/fps` for `i in
  range(n_frames)` — start time is hardcoded 0. Telegram stickers cap at 3s, so a shader whose
  best window is at t=4s is uncapturable. Fix: carry a `start_t` on `RenderPreset` (or a
  loop-offset control on the sticker section) and pass it into the render loop. Deferred in
  feature 010 (Out of scope); the share UI has no offset control today. The copilot `render_video`
  (020·18) inherits this — it too renders only t=0..seconds.

## [DEFERRAL] copilot context bloat: tool catalogue all-eager (no lazy load) — lever 2 of 2
- **Trigger:** when optimizing copilot cost/latency again — the remaining win is lazy-loading. Re-measure
  with `scripts/token_probe.py` before AND after (it reads real specs + hits OpenRouter; the measured
  numbers, not chars/4, are what to trust).
- MEASURED 2026-06-09 (`scripts/token_probe.py`, grok-4.3): a system+user request is ~7346 input tok —
  +3271 system prompt (now COMPRESSED to +2725) + 3941 the native 21-tool `tools=` block. Per-tool marginal
  ~131 tok, of which the DESCRIPTION (~133/tool) dominates over name (~76 incl. block overhead) + schema
  (~47). **The native `tools=` block is re-sent + re-billed in FULL on EVERY iteration** (confirmed:
  step-1 and step-2 deltas both 3941) — `agent.py` re-passes `eager_specs()` each `client.stream`.
- **CACHE NUANCE (the run1 report overstated the win):** OpenRouter prompt-caching IS active on grok-4.3 —
  a repeated prefix reports ~99% `cached_tokens` (~4x cheaper, NOT free), so the per-iteration re-send is
  amortized WITHIN a turn; full price is paid on the first iteration + after the ~5min cache TTL lapses.
  So lazy-load saves real money but LESS than "halve every request". Don't touch the per-iteration re-send
  (Q3 amortizes it; removing it risks tool-calling).
- DONE this wave (lever 1): the system prompt was compressed ~20% (13530->10606 chars, -678 tok/request,
  info-preserving — reviewer-audited) — it is now POLICY-only, no longer a verbose re-walk of every tool.
- REMAINING (lever 2): lazily load the telegram/youtube/publish tools (they carry `eager`/`category`;
  `registry.specs_for(names)` + `eager_specs()` filter already exist) so a shader turn ships ~10 tools not
  21 (measured: 2495 vs 3941 tok). A compact plaintext menu of the long tail costs ~1714 tok and a 2-stage
  `load_tools(names)` flow was verified to work on grok-4.3. Scaffold: `11 §4` search_tools/list_tools +
  `grow_specs_from_payload` (`16 ## Out of scope`). Spec: `20_ui_ux_polish.md` D5; re-measure via `scripts/token_probe.py`.
- IMPLEMENTATION NOTE: a discovery tool (`list_tools`/`load_tools`) needs registry access from inside its
  handler, but `build_registry` constructs the `ToolRegistry` AFTER the builders run — the feature needs a
  two-phase registration (build defs → construct registry → bind discovery handlers). The compact menu line
  derives from `description`'s first sentence; if that proves too long, add `brief: str = ""` to
  `ToolDefinition` then (defaulted ⇒ pure addition; resolver `d.brief or first_sentence(d.description)`).

## [DEFERRAL] `ToolDefinition.needs_gl` + `.category` are dead fields (doc-only)
- **Trigger:** when lever 2 (the lazy tool catalogue) lands — `category` goes live as the catalogue
  grouping and `needs_gl`'s fate gets decided in the same pass; OR next time a field is added to
  `ToolDefinition`.
- Neither field has a consumer today: thread-marshalling actually happens inside the backend's
  methods (`backend.py`), so a wrong `needs_gl` value is uncatchable; `category` is read by
  nothing until the catalogue exists. Decide then: delete `needs_gl` (it documents, not enforces) or
  keep both as scaffold.

## [DEFERRAL] copilot agent-level error recovery — partially proven, the THRASH + edit-mismatch classes untested
- **Trigger:** before claiming the copilot is robust to its own mistakes / shipping the copilot — run the
  recovery probes through a dedicated dogfood mission (the THRASH case in particular); not yet covered by
  `01_shape_gallery`, which only surfaces incidental single-error recoveries.
- PROVEN (2026-06-09): the broken-compile read→fix loop WORKS — across dogfood runs 3-4 (codex-mini) the
  agent self-corrected applies-with-errors edits 4-5x per run, including CROSS-tool (a broken `create_node`
  fixed by a clean `replace_lines`), confirmed by `scripts/dogfood/analyze.py`'s recovery rollup. So
  single-error comprehension + correction is solidly confirmed.
- STILL UNTESTED: (a) the THRASH case — MANY consecutive applies-but-compile-with-errors edits. The
  broken-compile circuit-breaker now LANDED (`consecutive_compile_failures` + a one-time
  `compile_thrash_nudge` at `max_compile_failures`, `agent.py`; unit-tested), but a dogfood mission must
  still drive a real thrash to confirm the nudge FIRES and actually unsticks the model. (b) `old_str`
  mismatch recovery; (c) bad-node-id recovery; (d) malformed-args recovery. A future dogfood mission should
  drive these deliberately + inspect the `edit_giveup`/`max_iterations`/`consecutive_failed_edits`/
  `compile_thrash_nudge` trace events (`agent.py`).

## [DEFERRAL] true in-line drag-selection of WRAPPED copilot chat prose
- **Trigger:** the per-message Copy button (020·23 D7) proves insufficient — a user wants to select a
  PHRASE within a message, not copy the whole thing.
- 020·23 fixed the reported bug (the window dragged from the body — now `WindowFlags_.no_move`, title-bar
  drag only) + added a per-message Copy. The old blocker ("`input_text_multiline` selects but can't
  word-wrap") no longer holds: 1.92.801 ships `InputTextFlags_.word_wrap` (verified 2026-06-07, now used
  for the chat INPUT). So a `read_only` + `word_wrap` multiline could give a selectable, wrapping
  message bubble. Untried for the transcript — open questions: per-message dynamic height, nested
  scrollbars in the history child, the disabled/read-only look vs `text_wrapped`. A spike when the
  per-message Copy proves insufficient, not a blind swap. `/imgui-ui` §5.

## [DEFERRAL] copilot has no bind_media / undo_edit tools (scope decisions)
- **Trigger:** bind_media — first real session where a user asks the copilot to load an image/video into
  a sampler uniform and the set_uniform "samplers aren't settable, edit source" handoff reads as a wall.
  undo_edit — first session where a bad multi-edit can only be re-edited (not reverted) and that friction
  is reported (today only `delete_node` has a Recover affordance; edits have no undo).
- Both are conscious SCOPE decisions parked at 020·20, not gaps to fix blind. bind_media: media binding is
  a real first-class app feature (the Media Input template stores `u_image`/`u_video` file_path; `core.py`
  binds the textures at render), so the capability is genuinely missing — the boundary is already
  documented to the agent. undo_edit: `recover_deleted_node` is the only revert affordance. Spec each when
  triggered. Source: the copilot-stack audit `missing_tools`.

## [DEFERRAL] imgui_color_text_edit render() FPE — editor hidden behind modals
- **Trigger:** if imgui-bundle fixes the upstream glyph-metric div-by-zero (or you upgrade and
  the crash no longer reproduces), the two guards below become unnecessary — let the editor render
  behind modals and apply settings live. Also: if you touch `tabs/code.py`'s render gate or
  `popups/settings.py`'s editor-settings apply timing, re-verify the crash stays suppressed.
- The bug (NOT fixable from Python — it's a div-by-zero inside imgui-bundle's C++
  `TextEditor::Render`, by `glyphSize.x/.y` from `ImGui::GetFontSize()` + the 1.92 dynamic font
  atlas, when the editor window isn't focused). Two triggers, two guards:
  1. Rendering the editor on a frame a popup is active → `tabs/code.py` simply does NOT draw the
     editor while `any_popup_open()` (the left pane is empty until the popup closes). Earlier a
     plain-text snapshot stood in here; removed as needless workaround complexity (user preferred
     it just disappear).
  2. Calling the editor's `set_*()` setters while a modal is open corrupts its glyph metrics →
     next render FPEs. The editor options live in `popups/settings.py` and apply ONLY on close
     (Close button + `hotkeys.py` Esc path), never live. **Lost feature:** no live-preview while
     dragging the options sliders.
- Reproduces only in the full app (bisected via headless `update_and_draw` cycles), on the
  latest imgui-bundle (1.92.801). Surfaced building the editor-options popup (feature 006).

## [DEFERRAL] inline editor caret invisible at column 0
- **Trigger:** first user complaint that the cursor "disappears" at line start, OR when you next
  patch the imgui-bundle `imgui_color_text_edit` binding for anything else (same upstream territory
  as the palette write-path) — fix the caret clip in the same pass.
- The goossens `TextEditor` draws the caret at `x = textStart + col*glyph - 1`; at column 0 that
  lands ~1px left of the text region and the editor's internal text clip culls it. Confirmed this
  session by framebuffer capture across a full blink cycle: caret visible 50/99 frames at col 5,
  0/99 frames at col 0 (not low-contrast — genuinely not drawn). NOT fixable from Python: window
  padding doesn't move the editor's internal clip, and there's no margin/caret-width API. Needs an
  upstream C++ change to `renderCursors()` (or a bound caret-margin setting).

## [DEFERRAL] gruvbox palette match for the inline editor
- **Trigger:** when you decide custom editor colors are worth a small upstream PR — the C++
  `TextEditor::Palette` is a mutable `std::array<ImU32>` but litgen binds only `.get()`; the write
  path is a ~3-line binding addition (a `set(Color, ImU32)` mirroring `get()` on the goossens
  `Palette`, then regenerate). Land it in `pthom/ImGuiColorTextEdit` (or the bundle's litgen config)
  and it flows into imgui-bundle on the next bump. OR: if the built-in dark palette's mismatch with
  the rest of the gruvbox UI visibly annoys in daily use.
- The Palette has no Python write path in imgui-bundle 1.92.801 (spike-confirmed: `.get()` only,
  `set_palette`/`set_default_palette` accept only an opaque `Palette`, no per-slot setter/list ctor).
  Feature 006 ships the built-in `TextEditor.get_dark_palette()`. The `COLOR.SYN_*` tokens in
  `theme.py` + the `Color`→`SYN_*` mapping table in `ai_docs/features/006_inline_editor.md`
  (Decision 5) document the intended mapping for when the write path lands.

## [DEFERRAL] split `ui.py` / `app.py` further
- **Trigger:** when editing `app.py` feels painful (lost search-and-replace, unclear blast
  radius), OR when a 4th tab module needs cross-cutting `App` operations not on its public API.
- NOT a default next-step — prior extractions already lifted the copilot backend
  (`copilot/backend.py`) and the whole headless project + copilot CORE (`project_session.py`, `App`
  owns one `ProjectSession` + forwards via `@property`) out of `App`. What remains in `App` is
  genuinely UI/glfw/imgui-bound (windowing, editor sessions, popups, nav, exporter panels). The
  further candidates (node-CRUD, path-properties, the `shader_lib_*` picker forwards) are net-negative
  on current evidence — don't re-propose without a fresh pain signal. Spec when triggered.

## [DEFERRAL] adopt `hello_imgui.apply_theme()` + `imgui-knobs` during UI/UX refactor
- **Trigger:** when starting the planned UI/UX refactor with custom themes — i.e. the moment a
  concrete theme design starts taking shape (not before; adopting now is premature scaffolding
  without a target).
- `hello_imgui` ships ~15 named themes (`hello_imgui.apply_theme(ImGuiTheme_.darcula_darker)`)
  + a live tweak GUI (`hello_imgui.show_theme_tweak_gui`). Adoption cost is ~5 LOC at theme
  swap time. `imgui-knobs` (rotary knobs for shader parameter UX) is in the same trigger —
  evaluate replacing `drag_float` widgets in `widgets/uniform.py` when drag UX feels
  insufficient. Both rejected from feature 004 scope to keep blast radius bounded; both
  available with no new dependency since `imgui-bundle` already bundles them.

## [DEFERRAL] headless smoke may not run on Windows CI
- **Trigger:** next time you touch `.github/workflows/ci.yml`, OR the first time the
  `windows-smoke` job's smoke step actually fails to run headless (GL-context absence on the
  GitHub Windows runner).
- The `windows-smoke` job's `uv run python scripts/smoke.py` step is `continue-on-error: true`
  on purpose — glfw needs a real GL context the runner may lack. `uv sync` in the same job is the
  hard gate (a dep that won't install on Windows reds the job). Runtime on Windows is verified by
  the maintainer's manual test (`BUILDING.md`), not CI. If a software-GL path (Mesa/llvmpipe on
  the runner) ever makes headless smoke reliable on Windows, drop the `continue-on-error`.

## [DEFERRAL] Pi GUI on hardware GL: labwc headless + wayvnc (researched 2026-06-10, unverified live)
- **Trigger:** next session that wants the ShaderBox GUI running ON the Pi at usable speed (today it
  only runs under xvfb -> llvmpipe at ~65ms/draw-call), OR the maintainer asks for "быстрый рендеринг
  на Pi" again.
- The split (verified): the standalone EGL path (dogfood/render scripts) ALREADY runs on the V3D
  HARDWARE (`renderer: V3D 4.2.14.0`); only the glfw GUI path is software, because glfw needs a
  display server and xvfb pins it to llvmpipe. The Pi is truly headless (HDMI disconnected) but
  `/dev/dri/renderD128` (the V3D render node) works without any display.
- The researched fix — a hardware-rendering virtual display, all pieces confirmed available:
  1. `sudo apt install labwc wayvnc` (both in the raspberrypi bookworm repo: labwc 0.8.4, wayvnc 0.9.1).
  2. `WLR_BACKENDS=headless WLR_LIBINPUT_NO_DEVICES=1 labwc &` — wlroots headless backend renders
     via the GPU render node (verify with `WLR_RENDERER=gles2`; if it silently falls back to pixman
     that's software again — check the labwc log line naming the renderer).
  3. `wayvnc 0.0.0.0 &` (option `-o HEADLESS-1` if it doesn't pick the virtual output) — view/drive
     from the desktop with any VNC client.
  4. Run the app on Wayland + the version overrides:
     `WAYLAND_DISPLAY=wayland-0 PYGLFW_LIBRARY_VARIANT=wayland MESA_GL_VERSION_OVERRIDE=4.6
     MESA_GLSL_VERSION_OVERRIDE=460 make run` — pyGLFW ships the wayland lib variant
     (`site-packages/glfw/wayland/`), the app sets no hard GL-version hints, and imgui goes through
     `imgui_bundle.python_backends.glfw_backend` (pure pyGLFW) — so the env switch covers the whole
     stack.
  Alternative (heavier, reboot): force the HDMI alive (`hdmi_force_hotplug=1` /
  `video=HDMI-A-1:1920x1080@60D`) + a real labwc session + wayvnc. Xorg+dummy rejected (glamor
  software fallback risk). NOTE: even on hardware, Pi-class V3D is slow on heavy text-stack shaders
  (~20s/300x300 — see 032 quirks); the GUI-path switch fixes the imgui/UI half, not shader cost.

## [DEFERRAL] headless GL on a display-less Linux box (Raspberry Pi dev sessions)
- **Trigger:** when working on a headless Linux box with no X/Wayland (e.g. a Pi over SSH) and
  `make test` skips the GL tests, OR when building the copilot dogfood harness (feature 026 — needs a
  GL context but not GUI rendering). NOTE: `make smoke` now self-skips on a no-GPU-window box (its
  `_has_gpu_window` probe), so it no longer crashes/crawls here — it prints SKIPPED and returns 0.
- Two separable GL needs, very different costs: **shader COMPILE** (`node.compile()` → `gl.program`)
  is ~3.5 ms and is all the copilot shader tools actually do (`read/edit/replace/insert/set_uniform/
  create/delete/switch_node` — `needs_gl=True` but compile-only; only `render_image`/`render_video`
  in `category="render"` truly render; a 400×400 image render ≈ 2.6 ms on V3D). **Frame RENDER**
  (`glDrawElements` of the imgui UI) is ~65 ms PER draw-call on llvmpipe (~1.4 s/frame for the 3-node
  grid) — the whole reason the GUI `smoke` crawls under xvfb. So a compile-only/image-only harness
  (dogfood, the GL unit tests) is fast headless; only the GUI `smoke` is slow.
- To get a GL context headless: `moderngl.create_standalone_context(backend='egl')` reaches the Pi's
  real GPU (`V3D`). v3d/llvmpipe report GL ≤4.5 by default but the project needs `#version 460`;
  `MESA_GL_VERSION_OVERRIDE=4.6 MESA_GLSL_VERSION_OVERRIDE=460` lifts the reported version (set BEFORE
  context creation — the driver reads them at creation). glfw itself CANNOT reach v3d headless (NULL
  platform fails EGL init; xvfb forces llvmpipe) — so the GUI `smoke`/App path is un-runnable here
  except slowly under `xvfb` + the MESA overrides. Feature 025 extracted `ProjectSession` (the headless
  project + copilot core, App-free) precisely so the dogfood harness (026) can drive the real copilot
  engine on a standalone EGL context WITHOUT App/glfw. NOTE: `import glfw`/`import imgui` succeed on the
  Pi (only `glfw.init()`/window/context creation fail), so a transitive imgui import (e.g. via
  `CopilotBackend`→exporter classes) does NOT block headless construction.

## [DEFERRAL] pre-commit ruff hook is pinned older than the resolved ruff
- **Trigger:** next time you bump `ruff` in `pyproject.toml`, OR the first time `make check`
  passes but `uv run ruff check .` fails (a lint the pinned hook is too old to emit). Bump the
  `.pre-commit-config.yaml` ruff `rev` to match in the same pass.
- `.pre-commit-config.yaml` pins `astral-sh/ruff-pre-commit` at `rev: v0.3.4`, but the project dep
  resolves a much newer ruff (0.11.x). So `make check` (→ pre-commit) lints with stale ruff and can
  green code a current ruff would flag (feature 011 hit this: new `# noqa: BLE001` markers were inert
  + would `RUF100`/`SIM105` under 0.11.x). Bumping the rev may surface new lints repo-wide — its own
  bounded change, not to be done mid-feature.

## [DEFERRAL] code-sign the Windows launcher (SmartScreen)
- **Trigger:** first user report that SmartScreen blocks them despite the bundled README's
  "More info → Run anyway" note, OR a paid (non name-your-own-price) release.
- The Windows `run.bat` is unsigned, so first-run trips SmartScreen ("Windows protected your
  PC"). Current mitigation is the bundled `scripts/README.md` note telling users to click
  "More info → Run anyway". A code-signing cert removes the warning but costs money/effort —
  deferred until the warning is shown to actually cost a user.

## [DEFERRAL] in-app replay mechanism (debug)
- **Trigger:** next time you hit a multi-step bug painful to reproduce manually, or want to
  share a repro with future-you.
- Manual-debug / shareable-repro tool (not a test framework — `scripts/smoke.py` covers
  regression). Spec before building; touches imgui boundary + adds UI surface.

## [DEFERRAL] Telegram has no import-existing-pack path
- **Trigger:** first user report of wanting to target a sticker pack ShaderBox did NOT create (a
  pack made by hand in @Stickers or another tool).
- The Telegram exporter only operates on packs it created/selected itself
  (`_create_pack` / `_select_pack` / `_delete_pack` / `set_default_pack` in
  `exporters/telegram.py`); there is no UI or method to add a pre-existing external pack. When a real
  use surfaces, build it as a small feature: validate the `_by_<botusername>` suffix against the
  linked bot, fetch the real title via `get_sticker_set`, surface a "pack not found / not yours"
  error, then append the `PackEntry`.

## [DEFERRAL] color emoji rendering in the picker (monochrome-only ceiling)
- **Trigger:** a future `imgui-bundle` bump — re-run a color-emoji spike (load `NotoColorEmoji.ttf`
  with the FreeType `LoadColor` path, render a row of faces in the glfw backend). If color glyphs
  render instead of blank, swap the picker's font + bump the `/imgui-ui` skill §8.
- Feature 009's emoji picker is monochrome (`NotoEmoji-Regular.ttf`) because this imgui-bundle
  build (1.92.801) renders NotoColorEmoji as blank glyphs even with `LoadColor` + the bundled
  plutosvg (spike-confirmed 2026-05-23). The chosen emoji still uploads to Telegram in full color —
  only the in-app picker preview is monochrome. See `/imgui-ui` skill §8.

## [DEFERRAL] `UINodeState` drops a node on an invalid known-key VALUE
- **Trigger:** next time you narrow a `UINodeState` Literal (e.g. add/remove a `UniformSortKey` /
  `UIUniformInputType` member) OR a user reports a node vanishing from the grid after an upgrade.
- `load_node_from_dir` filters *unknown keys* but a *known key with an out-of-Literal value* (a
  stale `uniform_sort_key`, or a bad `input_type` inside `ui_uniforms`) raises `ValidationError`,
  which `load_nodes_from_dir`'s `except` swallows → the whole node is silently skipped/lost.
  `UIUniform.snap_input_type()` (called each frame in `tabs/node.py`) covers the in-app
  `input_type` case, but only AFTER load — a value bad enough to fail pydantic construction never
  reaches it. `UINodeState` has no `model_config`/value-level fallback (unlike `UIAppState`'s
  deliberate `extra="forbid"` + `load_and_migrate`). If robustness parity is wanted, add a
  validator that resets out-of-range values to defaults instead of raising.

## [DEFERRAL] Telegram egress is pinned to IPv4 (breaks a genuinely IPv6-only client)
- **Trigger:** first user report of "can't connect to Telegram" on a network confirmed IPv6-only
  (no IPv4 at all — rare on desktop), OR next time you touch `exporters/telegram.py::_ipv4_request`.
- `_ipv4_request()` binds the httpx transport to `local_address="0.0.0.0"`, forcing v4 egress (both
  `request` and `get_updates_request` pools). This is a deliberate trade: it fixes the common
  dead-/absent-v6 case (any non-v6 VPN/tunnel — the maintainer's box, see vpn-stack Gotcha #4 +
  `conventions.md ## Known quirks`), but a host with no IPv4 can't connect at all. Acceptable for an
  itch.io desktop tool (dual-stack/v4-only dominate). Robust fix when the trade actually bites:
  replace the v4 pin with a Happy-Eyeballs resolver (try both families, use whichever connects) — not
  native in httpx 0.28, needs a custom transport/resolver, so not worth it pre-report.

## [DEFERRAL] YouTube egress is NOT IPv4-pinned (unlike Telegram)
- **Trigger:** a user reports a hanging YouTube Connect/Upload on a network where AAAA resolves but
  the v6 route is dead (any non-v6 VPN/tunnel — vpn-stack Gotcha #4), OR next time you touch
  `exporters/youtube.py`'s `build()`/`run_local_server` egress.
- The Google client (`google-api-python-client` → `httplib2`/`requests`) makes its own outbound
  connections with no IPv4 bind, unlike `exporters/telegram.py::_ipv4_request`. Verified working
  end-to-end through the full app on the maintainer's box (v0.8.0 ship), so the common dead-v6 case
  apparently resolved fine here — but the Google libs don't expose httpx's clean `local_address` knob,
  so there's no pin. If a user stalls, force v4 via a custom transport (requests adapter /
  `source_address`, or a custom `httplib2.Http`).

## [DEFERRAL] integration credentials stored cleartext in integrations.json (Telegram + YouTube + Copilot)
- **Trigger:** a security-hardening pass on stored secrets, OR the first user report of a leaked
  `integrations.json` (shared machine / synced dir). Do all three integrations in the same pass.
- `integrations.json` holds the Telegram `bot_token`, the YouTube `client_secret` + `token_json`
  (the long-lived OAuth refresh token), and the Copilot `openrouter_key` in cleartext at
  `app_data_dir()`. Acceptable posture for a single-user desktop tool (matches the original feature-001
  decision), but the YouTube refresh token is channel-scoped + long-lived and the OpenRouter key bills
  real money — worth a keyring/OS-secret-store migration if secrets ever warrant it. One
  `IntegrationsStore` centralizes all of it, so the migration has a single seam.

## [DEFERRAL] decompose exporters/telegram.py + youtube.py (both large, monolithic)
- **Trigger:** a third concrete exporter lands (the shared worker/panel patterns become worth
  hoisting into per-exporter subpackages), OR the first time a localized telegram/youtube fix has to
  wade through most of the file to land. Deferred from feature 017 — the riskiest split (shared mutable state:
  `_render_state`, the job/progress queues, `_worker` lifecycle cross every seam).
- Target shape (from the 017 audit): `exporters/telegram/` + `exporters/youtube/` subpackages —
  `exporter.py` (class + auth/connect) / `worker.py` (thread + async ops / job handlers) /
  `panel.py` (the share-panel UI) / `models.py` (internal dataclasses) / `util.py` (the existing
  `telegram_util.py` / `youtube_util.py` move IN at that point). Until then the two `*_util.py`
  helpers sit flat under `exporters/`.

## [DEFERRAL] split ui_primitives.py (growing)
- **Trigger:** when it grows a clearly separable cluster (e.g. the exporter panel chrome —
  `preview_box` / `status_slot` / `unconnected_gate` / `setup_steps`) that gets reused outside the
  exporter context, OR when editing it feels unwieldy. Rated KEEP in feature 017 (button tiers + draw
  helpers + `preview_cell` + labeled fields all serve one role: the shared imgui+theme primitive set).
