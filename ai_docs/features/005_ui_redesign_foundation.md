# Feature 005 — UI/UX redesign foundation (PRs 1-4)

## Goal

Land the **foundation** of the gruvbox + wide-screen UI redesign delivered by the design
collaborator (see `ai_docs/design/` for `SPEC.md`, `tokens.json`, `theme.py`,
`prototype.html`). This feature covers the designer's **PRs 1-4** of his 8-PR adoption
sequence (his `SPEC.md §15`):

- **PR 1** — drop in `theme.py` + `tokens.json`; call `apply_theme(...)` at app boot. App
  looks gruvbox; nothing else changes structurally.
- **PR 2** — sweep the codebase to replace every hardcoded color/size with `COLOR.*` /
  `SIZE.*` lookups from `theme.py`. No visual change; massive maintainability win
  (pain-points.md §1).
- **PR 3** — add the bottom status bar; relocate the FPS readout and notifications into
  it; delete the top-right notification overlay.
- **PR 4** — switch glfw window to full-monitor; introduce the 50/50 horizontal split;
  left half = placeholder "embedded editor coming in feature 006" panel; right half =
  existing render-image + node-strip + tabs stack, with render image capped at
  `SIZE.RENDER_MAX_H`.

After this lands, **the visual identity is gruvbox**, the **layout is wide-screen-first**,
and **every scattered size/color value lives in `theme.py`** ready for iteration.

## Out of scope (deferred to features 006-009)

- **PR 5 — embedded GLSL editor** (`imgui_color_text_edit` integration; syntax palette
  wiring; `Edit code → embedded` swap). Filed as feature 006 in `todo.md`. Trigger: when
  the left-half placeholder feels insufficient for actual day-to-day workflow.
- **PR 6 — Node tab restructure** (collapse two-child layout; type-pill switcher per
  uniform row; drop selected_uniform_settings child). Feature 007. Trigger: after the
  embedded editor lands (Node tab and editor share screen real estate).
- **PR 7 — error banner** (inline error widget with click-to-jump line:col). Feature 008.
  Trigger: blocked on feature 006 (click-to-jump needs the embedded editor).
- **PR 8 — Tweaks panel** (in-app accent/density/rounding/side swapper, persisted in
  app_state.json). Feature 009. Trigger: after the user has lived with the default skin
  for ~1 week and has a preference signal.
- The designer's `SPEC.md §16` open questions (editor-library pick, multi-file editor,
  Fit/Actual/Fill render toggle, `hello_imgui.apply_theme` registry) — all revisit at
  feature 006 plan-lock.

## Design decisions (locked)

### 1. Three `theme.py` fixes during integration
The delivered `theme.py` violates two of our `conventions.md ## Code rules` and
assumes a font we don't ship yet. Fix all three during the drop-in (PR 1):

- **`from __future__ import annotations` (line 26 in delivered theme.py)**: remove. The
  type-quoted forward refs `"imgui.Style | None"` and `"imgui.IO"` work via PEP 604
  string-annotation evaluation without the `__future__` import; verify pyright stays
  green after removing.
- **`_setattr_if` try/except (lines 401-407 in delivered theme.py)**: replace with direct
  attribute access. We're pinned to `imgui-bundle 1.92.801`; every `Col_.*` slot
  referenced in the function exists in that pinned API. Silently swallowing
  `AttributeError` violates the "don't sidestep" rule. **Implementation**: replace each
  `_setattr_if(C, getattr(Col, "name", None), value)` with `C[Col.name] = value` after
  one-time grep verification that `Col.name` exists in
  `.venv/lib/python3.12/site-packages/imgui_bundle/imgui/__init__.pyi`. If a slot doesn't
  exist (e.g. `Col.nav_highlight` was renamed in 1.91+), drop the line entirely — don't
  guard it.
- **Font path**: the delivered `load_fonts(io)` points at
  `resources/fonts/JetBrainsMono-Regular.ttf` which we don't ship. Keep the existing
  `App.get_font()` path (loads `Anonymous_Pro/AnonymousPro-Regular.ttf`) and have
  `load_fonts` delegate to it OR drop `load_fonts` entirely from theme.py and keep
  `App.get_font()` as the font-loading entry point unchanged. **Decision**: drop
  `load_fonts` from theme.py — `App.get_font()` is the single font-loading site, no
  reason to fork it through theme.py. JetBrains Mono adoption becomes its own decision
  later (file as deferral in `todo.md`, trigger = first time anyone touches font sizing).

**Rationale for fixing inline rather than v2'ing the designer**: the violations are
2-line cosmetic deltas; round-tripping for them is overkill. The font-path mismatch is
a substantive design call (which font we ship) — defer rather than absorb.

### 2. Tokens consumed via the `COLOR` / `SIZE` / `SPACE` bags exported by theme.py
PR 2's sweep targets — every hit gets replaced with the corresponding token lookup. The
audit produced this list (grep cited in spec; sweep all sites):

**Hardcoded color tuples (28 hits across 8 files, 9 unique colors)**:

| Tuple              | Semantic role                 | Token replacement      | Sites (file:line)                                                                                                                                                     |
|--------------------|-------------------------------|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `(1.0, 0.0, 0.0)`  | error                         | `COLOR.STATE_ERROR`    | `ui.py:155,194`, `widgets/node_grid.py:46`, `tabs/share.py:31,32`                                                                                                     |
| `(1.0, 1.0, 0.0)`  | warning                       | `COLOR.STATE_WARN`     | `ui.py:171`, `tabs/share.py:71,116`, `tabs/render.py:44,54`, `exporters/telegram.py:206,214,273`                                                                      |
| `(0.0, 1.0, 0.0)`  | success / selected            | `COLOR.STATE_OK` or `COLOR.ACCENT_PRIMARY` (selected) | `notifications.py:10` (default), `popups/node_creator.py:32` (selected), `widgets/node_grid.py:48` (selected), `widgets/uniform.py:169` (button highlight), `exporters/telegram.py:204,288,697` |
| `(0.5, 0.5, 0.5)`  | dim / muted text              | `COLOR.FG_DIM`         | `widgets/details.py:40,46`, `widgets/node_grid.py:29`, `popups/settings.py:40`, `widgets/uniform.py:106`, `tabs/node.py:23,99`, `exporters/telegram.py:227`           |
| `(0.8, 0.8, 1.0)`  | info heading                  | `COLOR.STATE_INFO`     | `tabs/share.py:68`                                                                                                                                                    |
| `(1.0, 0.3, 0.3)`  | error variant                 | `COLOR.STATE_ERROR`    | `exporters/telegram.py:205,290`                                                                                                                                       |
| `(0.15, 0.15, 0.15)` | dim separator               | `COLOR.BORDER` (or palette `bg_1`) | `tabs/node.py:134`                                                                                                                                          |
| `(0.2, 0.2, 0.2)`  | dimmed image overlay          | (keep — used in image_with_bg `tint_col` for shader-error state; this is the "dim by 80%" tint, not a semantic role. **Defer** to PR 7 (error banner replaces the dim trick).) | `ui.py:146` |
| `success/warning/error map` | UIMessage.get_color  | dict literal → `{success: COLOR.STATE_OK, warning: COLOR.STATE_WARN, error: COLOR.STATE_ERROR}` | `ui_models.py:23-25` |

**Hardcoded sizes (11 hits across 6 files)**:

| Value | Semantic role            | Token replacement         | Sites (file:line)                                                                  |
|-------|--------------------------|---------------------------|------------------------------------------------------------------------------------|
| `150` | node-preview thumbnail   | `SIZE.THUMB_LG`           | `widgets/node_grid.py:39`, `popups/node_creator.py:26`                             |
| `200` | small media preview      | `SIZE.PREVIEW_W`          | `ui.py:70` (kwarg `width=200` in `adjust_size`)                                    |
| `600` | control-panel min height | `SIZE.PANEL_CTRL_MINH`    | `ui.py:109`                                                                        |
| `90`  | thumbnail (telegram, uniform texture-button) | `SIZE.THUMB_SM` | `widgets/uniform.py:176,177` (`image_height=90, image_width=90`), `exporters/telegram.py:_PREVIEW_THUMB_HEIGHT=90` |
| `80`  | small button width       | `SIZE.BTN_SM_W`           | `popups/settings.py:75` (Close), `tabs/node.py:33,37` (Edit code / Open dir)       |
| `120` | medium button width      | `SIZE.BTN_MD_W`           | `tabs/share.py:123` (Render)                                                       |
| `100` | render-image min height  | `SIZE.RENDER_MIN_H` (220 per tokens — but our current value is 100; keep current value during sweep, **don't change visual behavior**; reconcile during PR 4 render-card refactor) | `ui.py:129` |
| `10.0`, `7.0` (pad, gap) | notification spacing | `SPACE.MD`, `SPACE.SM` (close enough; tokens are 8 + 4 — `1px` shift, visually equivalent) | `notifications.py:32,33` |

**Note**: `widgets/uniform.py:60` uses `max_image_height = 0.5 * avail.y` — not a magic
number, it's a derived ratio. Leave it.

### 3. PR 3 — status bar replaces overlay + relocates FPS

Per `SPEC.md §11`. New helper `_draw_statusbar(app)` in `ui.py`. Rendered last in the
main-window body (after `_draw_main_split`), pinned via `imgui.set_cursor_pos((SPACE.MD,
main_window.y - SIZE.STATUSBAR))`. Contents (left to right):

- `_dot(status_color)` + `text_colored(status_color, status_text)` — `state.ok / warn /
  error / info` from current node's compile state.
- separator
- `text_colored(FG_DIM, "node")` + `text_colored(FG_PRIMARY, current_node.name)`
- separator + resolution string
- separator + `f"{fps:.1f}"` + `text_colored(FG_DIM, "fps")` — FPS MOVES here.
- separator + `text_colored(FG_DIM, "GLSL")` + `text_colored(FG_PRIMARY, "4.60 core")` (static for now; we always render `#version 460 core`)
- toast slot — if `notifications.head` is non-empty, render it inline (TTL still applies).
- right-aligned: `_shortcut_legend()` — segmented `kbd` chips for `⌃N new ⌃S save ⌃E edit ⌃D del ← → nav ⌥S setup`.

**Notifications API**: keep `notifications.py::push()` unchanged. Drop the existing
overlay drawing in `update_and_draw`; instead expose `notifications.head` (the
non-expired head of the stack) for the status bar slot. The TTL update logic stays in
`notifications.py` — only the render site moves. (Matches `SPEC.md §11.1`.)

**FPS top-bar relocation**: drop `imgui.same_line() + imgui.text(f"Global FPS: ...")`
from `ui.py`'s top button row (lines ~120-121 in current code). Add the right-aligned
"⚙ Tweaks" placeholder button per `SPEC.md §2` — wire it to a no-op for this feature
(`logger.info("tweaks toggle — not yet implemented")`). Real tweaks panel is feature 009.

### 4. PR 4 — wide layout (full-monitor, 50/50 split)

Per `SPEC.md §1 + §3`. The glfw window switches from `monitor_width // 2` to full
`monitor_width`. Position changes from "right half of monitor" to "(0, 0)". Inside, the
main imgui window covers the full glfw window (no change in mechanism).

Layout body becomes:

```python
# In ui.py::update_and_draw, inside `with imgui_ctx.begin("ShaderBox - UI", flags=...)`
_draw_topbar(app)                    # Open project · Settings · spacer · Tweaks · FPS removed (now in statusbar)
avail = imgui.get_content_region_avail()
status_h = SIZE.STATUSBAR + SPACE.SM
split_h = avail.y - status_h
left_w = split_h_split.x = avail.x // 2  # 50/50 default
with imgui_ctx.begin_child("left", size=imgui.ImVec2(left_w, split_h)):
    _draw_editor_placeholder(app)
imgui.same_line()
with imgui_ctx.begin_child("right", size=imgui.ImVec2(avail.x - left_w - SPACE.SM, split_h)):
    _draw_render_card(app)           # render image, capped at SIZE.RENDER_MAX_H
    _draw_node_strip(app)            # current horizontal-ish node grid moves here
    _draw_node_tabs(app)             # current 3-tab bar
_draw_statusbar(app)                 # bottom 24 px
```

**Editor placeholder** content (PR 4 deliverable):

```python
def _draw_editor_placeholder(app: App) -> None:
    with imgui_ctx.begin_child("editor_placeholder", size=(0, 0),
                               child_flags=imgui.ChildFlags_.borders):
        avail = imgui.get_content_region_avail()
        msg_a = "GLSL editor"
        msg_b = "coming in feature 006"
        sz_a = imgui.calc_text_size(msg_a)
        sz_b = imgui.calc_text_size(msg_b)
        cx = avail.x / 2
        cy = avail.y / 2
        imgui.set_cursor_pos((cx - sz_a.x / 2, cy - sz_a.y))
        imgui.text_colored(COLOR.FG_PRIMARY, msg_a)
        imgui.set_cursor_pos((cx - sz_b.x / 2, cy + SPACE.SM))
        imgui.text_colored(COLOR.FG_DIM, msg_b)
```

**Render card** (replaces today's free-floating render image):

```python
def _draw_render_card(app: App) -> None:
    imgui.push_style_color(imgui.Col_.child_bg, (0.0, 0.0, 0.0, 1.0))
    with imgui_ctx.begin_child("render_card",
                               size=imgui.ImVec2(0, SIZE.RENDER_MAX_H + 2 * SPACE.MD),
                               child_flags=imgui.ChildFlags_.borders):
        ui_node = app.ui_nodes.get(app.current_node_id)
        if ui_node is None:
            _draw_empty_render_state()
        else:
            _draw_render_image(ui_node)
    imgui.pop_style_color()
```

The existing shader-error red-text overlay stays (it's a custom `add_text` on the dimmed
image) — replaced by a real banner widget in **feature 008 (PR 7)**, NOT here.

**Node strip** (PR 4 — keeps the existing 2D grid for now, since horizontal-scrolling
node strip is `SPEC.md §6` and structurally identical to current `widget_node_grid.py`
output). The redesigned horizontal strip with the type-pill node-selector chip is
deferred to **feature 007 (PR 6)**. For PR 4 we just move the existing
`draw_node_preview_grid` into the right pane's vertical stack.

**Three tabs** (PR 4 — preserved unchanged from feature 004). The Node tab's two-child
layout (uniforms left, selected-uniform editor right) stays. Restructure to single list
+ type pill is feature 007.

### 5. Branch + merge strategy

Per the repo's "branch only if user asks" rule, this feature spans 4 PRs (per designer's
adoption sequence) and high-blast-radius layout changes — work on
`feature/ui-redesign-foundation` off master. Squash-merge to master at end.

Sub-commits during impl organized by PR (3-4 commits + spec patch if needed):

1. Spec + decisions (this file)
2. PR 1 — theme drop-in + 3 fixes
3. PR 2 — token sweep (28 colors + 11 sizes across 9 files)
4. **PR 3+4 merged** — wide layout (50/50 split, editor placeholder, render card) +
   status bar (replaces notification overlay + relocates FPS readout). Merging these
   two PRs because they both rewrite the same `update_and_draw` frame body; doing
   them sequentially would mean two rewrites of the same code in two commits, which
   has no real review-ability benefit. Result: one larger commit (~150 LOC ui.py
   rewrite) but coherent.

Each commit aims to keep `make smoke` green.

## Files touched

**New files**:
- `shaderbox/theme.py` (drop-in from designer + 3 fixes per Decision 1)
- `ai_docs/design/SPEC.md` (designer's spec, archived for future reference)
- `ai_docs/design/tokens.json` (designer's tokens, archived)
- `ai_docs/design/prototype.html` (designer's visual source-of-truth, archived)
- `ai_docs/design/README.md` (designer's deliverable README, archived)

**Modified — PR 1**:
- `shaderbox/app.py` — `__init__` calls `apply_theme(imgui.get_style())` after
  `imgui.create_context()`.

**Modified — PR 2** (token sweep, no visual change):
- `shaderbox/ui.py`
- `shaderbox/ui_models.py` (UIMessage.get_color dict)
- `shaderbox/notifications.py` (default color tuple)
- `shaderbox/widgets/node_grid.py`
- `shaderbox/widgets/uniform.py`
- `shaderbox/widgets/details.py`
- `shaderbox/popups/settings.py`
- `shaderbox/popups/node_creator.py`
- `shaderbox/tabs/node.py`
- `shaderbox/tabs/share.py`
- `shaderbox/tabs/render.py`
- `shaderbox/exporters/telegram.py`

**Modified — PR 3**:
- `shaderbox/ui.py` (`_draw_statusbar`, drop FPS from topbar, drop notifications overlay
  call)
- `shaderbox/notifications.py` (expose `head` property; drop `update_and_draw` overlay
  rendering, replace with `update()` that just ages the stack)

**Modified — PR 4**:
- `shaderbox/app.py` (`__init__`: glfw window full-monitor; position `(0, 0)`)
- `shaderbox/ui.py` (split rewrite: topbar / left-right split / statusbar)

**Docs**:
- `CLAUDE.md` (stack line: keep `imgui-bundle`; mention theme.py if structure note grows)
- `ai_docs/conventions.md ## Design decisions` (new bullet: theme.py = single source of
  truth for colors/sizes/spacing — revisit if a fourth bag of tokens emerges)
- `ai_docs/todo.md` (file features 006-009 + 4 §16 open questions)
- `ai_docs/worklog.md` (entry)

## Manual verification

### Pre-impl baseline
0a. **Smoke baseline**: `make smoke 2>&1 | tee /tmp/smoke_baseline_pre_005.log` on master
    before branching. Confirm exit 0.

### Post-PR verification (per-PR)

**After PR 1**:
1. `make check` clean.
2. `make smoke` clean; no GL errors; visual output: app boots, all windows render, but
   colors are gruvbox (no longer Dear ImGui's default dark).
3. Eyeball: open the app, confirm gruvbox palette dominant — `text` is `#ebdbb2`,
   `button` is `#504945`, etc. No structural shifts (still half-monitor, still old layout).

**After PR 2**:
1. `make check` clean.
2. `make smoke` clean.
3. Eyeball: the app looks **identical to after PR 1** — only that hardcoded tuples no
   longer exist anywhere in the codebase. Sanity grep:
   ```bash
   grep -rnE "\([0-9]\.[0-9]+,\s*[0-9]\.[0-9]+,\s*[0-9]\.[0-9]+" \
        --include="*.py" shaderbox/ | grep -v COLOR | wc -l
   ```
   Should be 0 (or only the deferred-to-PR-7 `(0.2, 0.2, 0.2)` shader-error dim tint).
4. Sanity grep for hardcoded sizes:
   ```bash
   grep -rnE "(width|height|min_h|max_h)\s*=\s*[0-9]+\b" --include="*.py" \
        shaderbox/ | grep -v SIZE | grep -vE "size_pixels|tex_size|texture\.size"
   ```
   Should be 0 for the values listed in the sweep table.

**After PR 3**:
1. `make check` clean.
2. `make smoke` clean.
3. Eyeball: bottom of the window now shows a 24-px-tall status bar with: status dot,
   node name, resolution, FPS, GLSL version, shortcut legend chips.
4. FPS readout no longer in the top button row.
5. Notifications: press `Ctrl+S` — toast renders inline in the status bar (not the
   top-right overlay anymore). It fades after 5 s.

**After PR 4**:
1. `make check` clean.
2. `make smoke` clean.
3. Eyeball: app window now fills the full monitor width (not half).
4. Left half shows a centered "GLSL editor / coming in feature 006" placeholder text.
5. Right half: render image (capped at RENDER_MAX_H = 360 px), then node grid, then
   3-tab bar.
6. Status bar still pinned at the bottom of the full-width window.
7. Resize / move test: drag the OS window — layout reflows.

### Final UX sweep (gates 1-13 from feature 004's manual procedure)

After PR 4 is in, run the 13-step manual sweep from feature 004 (gates 3-13 in
`ai_docs/features/004_imgui_bundle_migration.md ## Manual verification`) — every
interaction still works. The redesign should not break: hotkeys, popups, file dialogs,
project switching, save/restart cycle.

## Open questions for the user (LOCKED at plan time)

1. **Editor library for feature 006**: `imgui_color_text_edit` (bundled). **LOCKED for
   feature 006.** Revisit if feature 006's pre-impl validation surfaces blockers.
2. **Multi-file editor**: out of scope. Single-file only.
3. **`hello_imgui.apply_theme` registry**: not adopting (we call `apply_theme()` directly
   from `theme.py`). The `register_with_hello_imgui` stub in `theme.py` stays as
   documentation.
4. **Fit/Actual/Fill render toggle**: defer to feature 008 if it ever surfaces as a
   real friction point.

## Review history

- Designer delivered `prototype.html`, `tokens.json`, `theme.py`, `SPEC.md`, `README.md`
  in `~/Downloads/shaderbox(1).zip` (round 1, partial v1). Three theme.py issues caught
  pre-impl: `from __future__ import annotations`, `try/except` slot-defense,
  JetBrains-Mono assumption. Resolutions locked under Decision 1.
- 8-PR adoption sequence in `SPEC.md §15` split: PRs 1-4 = this feature (foundation);
  PRs 5-8 = features 006-009 (filed as triggered deferrals in `todo.md`).
- Plan-locked autonomously (user away). No pre-impl multi-agent convergence loop because
  this is integration of a third-party-designed spec into existing infrastructure —
  the spec already converged on the design side; our job is faithful integration with
  the 3 fixes called out. Pre-impl review will run after Decision 1 sweep is verified
  against the live imgui-bundle pyi.

**Post-impl review**: to be filled.
