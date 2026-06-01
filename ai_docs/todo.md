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
-->

---

## [DEFERRAL] copilot trace can bleed into the prior project's file on a mid-turn project switch
- **Trigger:** feature 022 (it must quiesce the worker at project switch to save conversation state) —
  fold this fix into that quiesced window. OR a maintainer notices a copilot transcript containing a
  turn that belongs to a different project.
- `session.reset_conversation` (main thread) closes the old `TraceLog` and opens a new one without
  JOINING the worker — it only sets `_cancel` + `gate.cancel_all`. A worker still inside `run_turn`
  holds the old `TraceLog` via its local `tr`; its next `tr.event` runs `_ensure_open`, which keys only
  on `self._fh is None` and so cannot tell "never opened" from "closed" — it re-opens the just-closed
  old file (append) and writes the in-flight turn's tail into the PREVIOUS project's transcript. The
  lock makes this safe (no crash, no write to a closed handle), but events bleed across projects. Same
  un-joined-worker window also commits the aborted turn to the orphaned old `history` (pre-existing).
  Honest fix: quiesce/join the worker before swapping the trace (022 needs the same quiesce point for
  conversation save), OR give `close()` a permanent-closed flag that `_ensure_open` respects (then the
  `_init`/release reopen path needs its own re-arm). Out of scope for 021 (logging format/leveling).

## [DEFERRAL] docs describe a "freetype glyph-atlas text-rendering shader" that no shipped shader uses
- **Trigger:** before the copilot's system prompt describes the text-rendering capability (it must
  describe what's ACTUALLY on disk), OR a maintainer pass on dead code, OR the next edit to any of the
  doc lines below.
- `fonts.py::Font` genuinely builds a freetype glyph-atlas (a GL texture of rasterized glyphs + per-char
  UV/metrics) — but it has **zero consumers** (verified 2026-05-29: nothing imports `fonts.Font`; the
  `push_font` hits are unrelated imgui UI fonts). The shipped Text Rendering template (`f90f5ff9`) is a
  pure **SDF 7-segment glyph synth** driven by `uniform uint u_text[64]` (codepoints) — NO atlas, NO
  sampler. So `fonts.py` is an abandoned approach the SDF template superseded, yet the docs still
  advertise the dead path as the live feature — grep `freetype glyph-atlas` across `CLAUDE.md`,
  `README.md`, `ai_docs/itch/page.yaml`, and `ai_docs/features/004_imgui_bundle_migration.md`. Honest
  fix when triggered: either delete the dead `fonts.py` glyph-atlas + correct all four doc surfaces to
  "SDF segment glyphs", OR confirm there's a real plan to wire the atlas (then the docs are aspirational,
  not stale). Re-check the itch store-page LIVE copy too (page.yaml is the source of truth, but a prior
  sync may have pushed the wrong line — `dev_flow.md ## Sync the itch.io page`).

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
  (The active-region visual cue — the accent outline + editor border — LANDED in the color/nav polish
  wave; no longer deferred.)

## [DEFERRAL] cross-file uniform declaration jump (lib files)
- **Trigger:** first user complaint that clicking a uniform name in the panel doesn't jump
  anywhere when the uniform happens to be declared in a lib file (not the node's own shader),
  OR when a real workflow puts uniform declarations in lib files often enough to feel friction.
- `shader_errors.find_uniform_declaration_line` searches only the active session's editor text
  (`widgets/uniform.py::_begin_ctrl`). A uniform declared in a lib file is discovered via the
  driver's introspection (so it appears in the panel) but click-to-jump silently no-ops.
  Honest fix: walk `node.compile_unit.sources` in order, search each, and open the matching
  session via `App.open_shader_lib_file(path)` before issuing the `JumpRequest`. Out of scope in 015
  (the spec calls this out: "lib = pure functions only" is the cognitive-clarity guarantee — if
  this defaults to false in practice, revisit the guarantee too).

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

## [DEFERRAL] resolution combo parses (w,h) back out of its display label
- **Trigger:** next time you change `util.get_resolution_str`'s format string (anything before
  the `WxH` token, or the `x`/space layout), OR the first time the canvas resolution silently
  fails to change on combo select.
- `tabs/node.py` reconstructs `(w, h)` from the selected combo item via
  `resolution_str.split(" ")[0].split("x")` — it re-parses the human display label produced by
  `get_resolution_str`. Safe today (the `WxH` token is always first, the `(ratio)` / `- name`
  suffix can't corrupt it) and pinned by `tests/test_util.py::test_resolution_str_format_parses_back`,
  but format and parser must agree forever. Cleaner fix when touched: carry a parallel `(w, h)`
  list indexed by the combo index instead of round-tripping the string.

## [DEFERRAL] sticker render is always t=0..N (no loop-offset / "which 3s")
- **Trigger:** first time a shader's interesting motion isn't at the start (a user wants a 3s
  window other than the first), OR when you next touch the render path in `tabs/share_state.py`
  (`render_for`) / `core.py` (`_render_video`).
- A shader loops infinitely but `_render_video` always renders frames `i/fps` for `i in
  range(n_frames)` — start time is hardcoded 0. Telegram stickers cap at 3s, so a shader whose
  best window is at t=4s is uncapturable. Fix: carry a `start_t` on `RenderPreset` (or a
  loop-offset control on the sticker section) and pass it into the render loop. Deferred in
  feature 010 (Out of scope); the share UI has no offset control today.

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
- NOT a default next-step — a 2026-05-15 parallel-agent assessment concluded extraction would
  be premature abstraction (same shape as feature 002's reversed AppContext). Spec when
  triggered.

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
