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
- **Trigger:** when editing `app.py` feels painful (search-and-replace across the file misses
  something, or a method's blast radius is unclear because too many siblings share state), OR
  when a 4th tab module needs cross-cutting `App` operations not currently on its public API.
  NOT a default next-step — a 2026-05-15 parallel-agent assessment of `project.py` extraction
  concluded the current `app.py` is coherent state on a single entity and the extraction would be
  premature abstraction (same shape as feature 002's reversed AppContext).
- Candidate shapes (if the trigger ever fires): `ProjectPaths` frozen dataclass (extract the 9
  `@property` paths into a value type, `app.paths.nodes_dir` etc.) OR `shaderbox/project.py`
  free functions taking `app: App` (`save` / `open_project` / `delete_current_node` /
  `create_node_from_selected_template` / `select_next_*`). The two are orthogonal — paths are a
  value domain, lifecycle is an action domain. `App` is the state-holder; widgets/popups/tabs/
  hotkeys take `app: App` directly (no `AppContext` wrapper).

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
- **Trigger:** next time you hit a multi-step bug that's painful to reproduce manually, or when
  you want to share a repro with future-you.
- Add a "Debug → Replays" UI surface that lists JSON files from `replays/`, plays them back by
  injecting synthetic actions into the existing hotkey/button code paths. DSL (rough): list of
  `{frame, action: hotkey|click|assert, ...}` dicts. Intercept points: imgui io-state setting
  for hotkeys (reuse `shaderbox/hotkeys.py::process_hotkeys` as-is),
  thin `replay_aware_button(label)` wrapper for clicks (or globally wrap `imgui.button` in
  replay mode). Not a test framework — for manual debugging / shareable repros (the headless
  smoke test in `scripts/smoke.py` is the right tool for actual regression testing). Probably
  worth a small feature spec before building (touches imgui boundary, adds UI surface).

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
  render instead of blank, swap the picker's font + bump `conventions.md ## Known quirks`.
- Feature 009's emoji picker is monochrome (`NotoEmoji-Regular.ttf`) because this imgui-bundle
  build (1.92.801) renders NotoColorEmoji as blank glyphs even with `LoadColor` + the bundled
  plutosvg (spike-confirmed 2026-05-23). The chosen emoji still uploads to Telegram in full color —
  only the in-app picker preview is monochrome. See `conventions.md ## Known quirks`.

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

## [DEFERRAL] integration credentials stored cleartext in integrations.json (Telegram + YouTube)
- **Trigger:** a security-hardening pass on stored secrets, OR the first user report of a leaked
  `integrations.json` (shared machine / synced dir). Do both integrations in the same pass.
- `integrations.json` holds the Telegram `bot_token` and the YouTube `client_secret` + `token_json`
  (the long-lived OAuth refresh token) in cleartext at `app_data_dir()`. Acceptable posture for a
  single-user desktop tool (matches the original feature-001 decision), but the YouTube refresh token
  is channel-scoped and long-lived — worth a keyring/OS-secret-store migration if secrets ever warrant
  it. One `IntegrationsStore` already centralizes all of it, so the migration has a single seam.
