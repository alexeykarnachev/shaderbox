# shaderbox · UI/UX spec · v1

**Audience**: the coding agent integrating the new design into `shaderbox/`.

**Companion files**:
- [`prototype.html`](../prototype.html) — canonical visual + interaction source-of-truth. Open it, scroll it, toggle Tweaks. Every pixel value, color, and state is encoded in the DOM/CSS — grep it.
- [`tokens.json`](./tokens.json) — every color / size / spacing constant the codebase should consume.
- [`theme.py`](./theme.py) — drop-in Python that writes those tokens into `ImGuiStyle` + `style.colors[...]`.

**Wrapper assumed**: `imgui-bundle 1.92.801`. All code examples use the snake-case `imgui.Col_.window_bg` / `imgui_ctx.begin_*` idioms per `wrapper.txt`.

---

## 0. What changed vs the current app

| Concern                       | Current                                                                  | New                                                                                              |
|-------------------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| **Window layout**             | One panel inside one glfw window (half-monitor wide)                     | Full-monitor; 50 / 50 split horizontally                                                          |
| **Left half**                 | (Empty — user runs external vim there)                                   | Embedded GLSL editor (`imgui_color_text_edit`) — opens the current node's `shader.frag.glsl`     |
| **Right half — top**          | The render image (~50 % of window height)                                | Render image, capped at `panel.render.max_h = 360 px`                                            |
| **Right half — middle**       | Bottom strip with node grid (left) + 3 tabs (right)                      | Horizontal node strip (one row of thumbs) above the tabs                                        |
| **Right half — bottom**       | (Status info hidden inside button row)                                   | Real status bar at the bottom (24 px, full width)                                                |
| **Compile errors**            | Red text overlaid on the dimmed render image (custom `add_text` call)    | Dedicated error panel — banner above the render area; expands inline with line-number jump      |
| **Node tab — uniform editor** | Split child: list left, selected-uniform editor right                    | Single list. Click the **type pill** on a row to switch input type. No side panel needed.        |
| **Theming**                   | Default Dear ImGui dark                                                  | gruvbox-dark via `apply_theme()`; user-swappable accent / density / rounding                     |
| **Shortcut discoverability**  | Hidden in a `?` tooltip in the node grid                                 | Always-visible legend in the status bar (`⌃N new ⌃S save ⌃E edit ← →`) + tooltip cheat sheet    |
| **Notifications**             | Overlay top-right of the main window (covers other UI)                   | Inline in the status bar (`.toast` slot) — fades on TTL                                         |
| **Selected-thumbnail border** | Green / red hardcoded `(0.0, 1.0, 0.0)`                                  | `COLOR.ACCENT_PRIMARY` / `COLOR.STATE_ERROR` from `theme.py` (re-skins with accent swap)         |

---

## 1. Top-level frame

```python
# Outer glfw window: full monitor (NOT half-monitor any more).
# main_glfw_window_size = monitor_width, monitor_height
# Defaults to monitor 1's full extents.

# Single always-open imgui window covering the glfw window.
imgui.set_next_window_size((win_w, win_h))
imgui.set_next_window_pos((0, 0))
_MAIN_FLAGS = (
    imgui.WindowFlags_.no_collapse
    | imgui.WindowFlags_.no_title_bar
    | imgui.WindowFlags_.no_scrollbar
    | imgui.WindowFlags_.no_scroll_with_mouse
    | imgui.WindowFlags_.no_resize
    | imgui.WindowFlags_.no_move
)
with imgui_ctx.begin("ShaderBox - UI", flags=_MAIN_FLAGS):
    _draw_topbar(app)                          # 32 px row
    _draw_main_split(app, win_w, win_h)        # editor | (render + nodes + tabs)
    _draw_statusbar(app)                       # 24 px row pinned to bottom
    _draw_modals(app)                          # Settings + Node Creator
```

`_draw_main_split` divides the remaining vertical space into two equal-width child windows. Implementation: cache `imgui.get_content_region_avail()`, halve `.x`, render two `begin_child(...)` side-by-side. The split ratio (default 0.5) lives in `app_state` so the user can drag-resize later.

---

## 2. Top bar (32 px)

Single horizontal row. Buttons inline-flow with `same_line()` — there is no `BeginMenuBar`; the design is intentionally menubar-less.

```python
imgui.text_colored(COLOR.ACCENT_PRIMARY, "▲ shaderbox")   # logo wordmark
imgui.same_line(); imgui.text_disabled("·")
imgui.same_line(); imgui.text_colored(COLOR.FG_PRIMARY, project_name)

imgui.same_line(SPACE.LG)
if imgui.button("Open project"): app.open_project()
imgui.same_line()
if imgui.button("Settings"):     app.open_settings()

imgui.same_line(imgui.get_content_region_avail().x - 220)  # right-align block
imgui.text_disabled(f"global fps")
imgui.same_line()
imgui.text_colored(COLOR.ACCENT_PRIMARY, f"{round(app.global_fps)}")
imgui.same_line(SPACE.LG)
if imgui.button("⚙ Tweaks"): app.toggle_tweaks_panel()
```

`COLOR.ACCENT_PRIMARY` is the only place the accent appears in the topbar — the logo glyph and the FPS readout. Resists data-slop.

---

## 3. Main split

### 3.1 Left pane — Embedded GLSL editor

**Widget**: `imgui_color_text_edit` (bundled in imgui-bundle; expose with `from imgui_bundle import imgui_color_text_edit as text_edit`).

**Layout**:
```
┌────────────────────────────────────────────────────┐
│  toolbar:  <relative path>  •  [Save] [Pop out]    │  ← 22 px
├────────────────────────────────────────────────────┤
│  gutter │  syntax-highlighted code                 │
│  48 px  │                                          │
└────────────────────────────────────────────────────┘
```

```python
with imgui_ctx.begin_child("editor", size=(left_w, avail_h)):
    _editor_toolbar(app, ui_node)             # path · dirty dot · Save · Pop out
    editor = app.editors[ui_node.id]          # lazy-allocate TextEditor per node
    editor.render(
        "##glsl_editor",
        size=imgui.get_content_region_avail(),
    )
```

**Syntax palette** wiring (one-time, at theme application): map `TextEditor.PaletteIndex.*` to the `COLOR.SYN_*` tokens. See `theme.py::COLOR.SYN_KEYWORD` etc. The `tokens.json::colors.role.syntax.*` block is the source-of-truth.

**File loading**: on `ui_node` change, set the editor's text to `(nodes_dir / node.id / "shader.frag.glsl").read_text()`. Watch mtime; if the on-disk file changes externally, reload (matches today's `ui.py` mtime polling).

**Saving**: `⌃S` writes `editor.GetText()` back to disk and triggers the existing `node.release_program(source)` reload path.

**Pop out**: kept as a fallback — spawns the configured external editor (current `Edit code` behavior in `popup_settings.py`). Useful when the user wants vim macros.

### 3.2 Right pane

Three stacked regions:

```python
with imgui_ctx.begin_child("right", size=(right_w, avail_h)):
    _render_card(app, ui_node)                # render preview, capped 360 px
    _error_banner_if_any(app, ui_node)        # only when shader_error != ""
    _node_strip(app)                          # horizontal thumb strip, ~110 px
    _tabs(app)                                # fills remainder
```

---

## 4. Render card

```python
def _render_card(app, ui_node):
    # canvas-wrap: black background, padded
    imgui.push_style_color(imgui.Col_.child_bg, (0.0, 0.0, 0.0, 1.0))
    with imgui_ctx.begin_child("render_card",
                               size=(0, SIZE.RENDER_MAX_H + 2 * SPACE.MD),
                               child_flags=imgui.ChildFlags_.borders):
        if ui_node is None:
            _draw_empty_state(...)            # see §4.2
        else:
            _draw_render_image(ui_node)       # §4.1
    imgui.pop_style_color()
```

### 4.1 Render image

```python
def _draw_render_image(ui_node):
    tex = ui_node.node.canvas.texture
    has_error = ui_node.node.shader_error != ""
    img_w, img_h = _fit_aspect(tex.size,
                               max_w=imgui.get_content_region_avail().x,
                               max_h=SIZE.RENDER_MAX_H)
    imgui.image_with_bg(
        imgui.ImTextureRef(tex.glo),
        image_size=(img_w, img_h),
        uv0=(0, 1), uv1=(1, 0),
        tint_col=(0.25, 0.25, 0.25, 1.0) if has_error else (1, 1, 1, 1),
    )
```

`_fit_aspect` is `ui_utils.adjust_size` shape, just with both max_w and max_h. The "letterboxed canvas with overflow black" feel is what we want — black `child_bg` does the work.

**Empty state** (§4.2 — replaces the current "press Ctrl+N" custom add_text):

```python
def _draw_empty_state(...):
    cx, cy = _centered_in_avail(text_w, text_h)
    dl = imgui.get_window_draw_list()
    dl.add_text((cx, cy - 14), to_u32(COLOR.FG_TITLE), "shaderbox")
    dl.add_text((cx + sub_x_offset, cy + 8),
                to_u32(COLOR.FG_DIM),
                "Press  ⌃N  to create a new node")
```

Centered icon-headline-subtext. The headline uses `font_18`. The subtext is `font_14` dim with the literal `⌃N` (rendered via JetBrains Mono — supports the Unicode glyphs).

### 4.3 Removed: render toolbar

The old plan had a small `render · UV Mango · 1280×960 · 4:3` toolbar above the canvas. **Cut.** All that info lives in the status bar already (node name, resolution, fps). Removing the toolbar buys ~22 px vertical and removes a duplicate-information surface — direct hit on pain point #3 (tab inconsistency).

---

## 5. Error banner

Renders inline between the render card and the node strip, **only when** `ui_node.node.shader_error != ""`. Replaces today's red `add_text` overlay (pain point #9).

```python
def _error_banner_if_any(app, ui_node):
    if ui_node is None or ui_node.node.shader_error == "":
        return
    err = ui_node.node.shader_error
    imgui.push_style_color(imgui.Col_.child_bg,   COLOR.BG_SURFACE)
    imgui.push_style_color(imgui.Col_.border,     COLOR.STATE_ERROR)
    with imgui_ctx.begin_child("err_banner",
                               size=(0, 0),
                               child_flags=imgui.ChildFlags_.borders
                                          | imgui.ChildFlags_.auto_resize_y):
        # header row — colored dot + summary
        _dot(COLOR.STATE_ERROR); imgui.same_line()
        imgui.text_colored(COLOR.STATE_ERROR, "compile error")
        imgui.same_line(); imgui.text_disabled("· shader.frag.glsl")

        # body — one row per GLSL error line, with parsed line:col link
        for line in err.strip().split("\n"):
            loc, msg = _parse_glsl_error(line)
            if loc:
                if imgui.selectable(loc, False, flags=imgui.SelectableFlags_.no_pad_with_half_spacing)[0]:
                    app.editors[ui_node.id].set_cursor_position(loc.line, loc.col)
                imgui.same_line(); imgui.text_colored(COLOR.STATE_ERROR, msg)
            else:
                imgui.text_disabled(line)
    imgui.pop_style_color(2)
```

`_parse_glsl_error` is new — most drivers emit `ERROR: 0:55: '...'` or `0(55) : error: ...`. A simple regex captures `(line, col?, message)`. When the user clicks the location, the embedded editor jumps there (`TextEditor.SetCursorPosition`).

---

## 6. Node strip

Horizontal strip of node thumbnails (replaces today's left-side grid).

```python
def _node_strip(app):
    with imgui_ctx.begin_child("node_strip",
                               size=(0, SIZE.THUMB_MD + 32),
                               child_flags=imgui.ChildFlags_.borders):
        # ---- slim actions row (22 px) ----
        _, app.app_state.is_render_all_nodes = imgui.checkbox(
            "render all", app.app_state.is_render_all_nodes)
        imgui.same_line(imgui.get_content_region_avail().x - 80)
        imgui.text_disabled(f"{len(app.ui_nodes)} nodes")
        imgui.same_line(); _help_icon(text=_SHORTCUTS_CHEAT_SHEET)

        # ---- thumb row ----
        with imgui_ctx.begin_child("node_thumbs",
                                   size=(0, SIZE.THUMB_MD + 8),
                                   window_flags=imgui.WindowFlags_.horizontal_scrollbar):
            for i, (node_id, ui_node) in enumerate(app.ui_nodes.items()):
                if i: imgui.same_line()
                _draw_node_thumb(app, node_id, ui_node)
            imgui.same_line()
            _draw_new_node_button(app)        # the "+" tile
```

### 6.1 Thumb

```python
def _draw_node_thumb(app, node_id, ui_node):
    is_selected = (node_id == app.current_node_id)
    has_error   = ui_node.node.shader_error != ""

    border = COLOR.STATE_ERROR if has_error else (
             COLOR.ACCENT_PRIMARY if is_selected else COLOR.BORDER)
    imgui.push_style_color(imgui.Col_.border, border)
    with imgui_ctx.begin_child(f"thumb_{node_id}",
                               size=(SIZE.THUMB_MD, SIZE.THUMB_MD * 3 // 4 + 22),
                               child_flags=imgui.ChildFlags_.borders):
        # thumbnail image
        tex = ui_node.node.canvas.texture
        if imgui.invisible_button("##b", (SIZE.THUMB_MD - 2, SIZE.THUMB_MD * 3 // 4)):
            app.set_current_node_id(node_id)
        # custom-draw the texture INSIDE the invisible_button rect
        rect_min = imgui.get_item_rect_min(); rect_max = imgui.get_item_rect_max()
        imgui.get_window_draw_list().add_image(
            imgui.ImTextureRef(tex.glo),
            rect_min, rect_max,
            uv_min=(0, 1), uv_max=(1, 0),
        )
        # name (full-width tinted bar at bottom)
        imgui.push_style_color(imgui.Col_.button,
                               COLOR.ACCENT_PRIMARY if is_selected else COLOR.BG_FRAME)
        imgui.push_style_color(imgui.Col_.text,
                               COLOR.BG_SURFACE if is_selected else COLOR.FG_SECONDARY)
        imgui.button(ui_node.ui_state.ui_name, size=(-1, 0))
        imgui.pop_style_color(2)
    imgui.pop_style_color()
```

**Custom-draw step**: drawing the texture inside the `invisible_button` rect via `get_window_draw_list().add_image(...)` is the trick that lets the whole thumb (image + name bar) act as one click target with a single border treatment. Today the code uses `image_button` + an outer `begin_child` with borders — same shape, just polished.

### 6.2 "+" tile

Renders at the end of the row as a button styled like a thumb. Same `THUMB_MD` width, dashed-border on hover (`Col_.border` + alt style), single `＋` glyph centered. Clicking opens the Node Creator modal.

---

## 7. Tabs

```python
with imgui_ctx.begin_child("tabs",
                           size=(0, 0),                    # fill remaining
                           child_flags=imgui.ChildFlags_.borders):
    with imgui_ctx.begin_tab_bar("node_tabs") as bar:
        if bar:
            with imgui_ctx.begin_tab_item("Node")   as t: t.visible and _tab_node(app)
            with imgui_ctx.begin_tab_item("Render") as t: t.visible and _tab_render(app)
            with imgui_ctx.begin_tab_item("Share")  as t: t.visible and _tab_share(app)
```

The tab body uses standard `WindowFlags_.always_vertical_scrollbar` behavior — content scrolls when it exceeds the tab body. Tab bar height is unchanged from imgui defaults (the rounding token `tab=2` gives it a softened corner top).

---

## 8. Node tab

Reordered + cleaned-up from today's `tab_node.py`.

```
filepath   ──  nodes/<uuid>/shader.frag.glsl        (clickable to copy)
Name       ──  [ input_text ]
[ Edit code  Open dir  Save as template ]    [ Delete ⌃D ]
----
Resolution ──  [ combo ▾ ]   [ Apply ]
==== Uniforms ===========================
u_aspect      [ auto    ]  1.333
u_time        [ auto    ]  143.358
u_box_size    [ drag    ]  [ 4.20 ] [ 1.01 ] [ 4.53 ]    ← selected (bg highlight)
u_cam_pos     [ drag    ]  [ 5.22 ] [ 4.66 ] [ 4.77 ]
u_focal_len   [ drag    ]  [ 0.94 ]
u_look_at     [ drag    ]  [ 0.38 ] [-2.67 ] [ 0.00 ]
u_tint        [ color   ]  ▢ 8ec07c
u_diffuse     [ texture ]  ▦ 1024×768 .jpg   [Load…] [Clear]
```

### 8.1 Row layout

Each uniform = **one row, three columns**: name | type-pill | controls. Layout via `imgui.columns(3)` or a manual `same_line(offset)` ladder — the prototype uses fixed pixel offsets (130 px name, 64 px type, rest = controls).

```python
def _draw_uniform_row(app, ui_node, uniform):
    is_selected = ui_node.ui_state.selected_uniform_name == uniform.name
    if is_selected:
        imgui.push_style_color(imgui.Col_.child_bg, COLOR.BG_SURFACE)
    with imgui_ctx.begin_child(f"uni_{uniform.name}",
                               size=(0, SIZE.ROW_HEIGHT + 2 * SPACE.XS),
                               child_flags=imgui.ChildFlags_.auto_resize_y):
        # ---- col 1: name ----
        imgui.text(uniform.name)
        imgui.same_line(130)
        # ---- col 2: type pill (clickable mini-combo) ----
        _draw_type_pill(app, ui_node, uniform)   # see §8.2
        imgui.same_line(130 + 64 + SPACE.MD)
        # ---- col 3: controls (auto / drag / color / texture / etc) ----
        _draw_uniform_control(app, ui_node, uniform)
    if is_selected:
        imgui.pop_style_color()
```

**Selection** is set on row click (`if imgui.is_item_clicked(): set_selected_uniform_name(...)`), same as today. **No more side-by-side child** for the selected-uniform editor — that surface is gone. The "switch input type" affordance moves into the pill.

### 8.2 Type pill = type switcher

```python
def _draw_type_pill(app, ui_node, uniform):
    pill_label = ui_node.ui_state.ui_uniforms[get_uniform_hash(uniform)].input_type
    color = {
        "auto":    COLOR.FG_DIM,
        "drag":    _P["blue_b"],
        "color":   _P["orange_b"],
        "texture": _P["aqua_b"],
        "array":   _P["fg_3"],
        "text":    _P["fg_3"],
        "buffer":  _P["purple_b"],
    }[pill_label]
    imgui.push_style_color(imgui.Col_.button,         (0, 0, 0, 0))     # transparent
    imgui.push_style_color(imgui.Col_.button_hovered, COLOR.BG_FRAME)
    imgui.push_style_color(imgui.Col_.text,           color)
    if imgui.small_button(f"{pill_label} ▾##{uniform.name}_pill"):
        imgui.open_popup(f"##type_popup_{uniform.name}")
    imgui.pop_style_color(3)

    with imgui_ctx.begin_popup(f"##type_popup_{uniform.name}") as p:
        if p.visible:
            for option in _allowed_input_types(uniform):  # depends on uniform's GL type
                if imgui.menu_item(option, "", False)[0]:
                    _set_input_type(ui_node, uniform, option)
```

`_allowed_input_types` keeps today's logic from `draw_selected_ui_uniform_settings` (a `vec3` can be `drag` or `color`; a `sampler2D` is forced to `texture`; an `uint[]` can be `array` or `text`; etc).

### 8.3 Per-input-type controls (§8 column 3)

Unchanged widget choices from today, just inline rather than two-column:

| input_type  | Widget |
|-------------|--------|
| `auto`      | `imgui.text(f"{value:.3f}")` (or list-stringified for vectors) |
| `drag`      | `drag_float` / `drag_float2/3/4` with `v_speed=0.01` |
| `color`     | `color_edit3` / `color_edit4` — no label, `##` hash only |
| `texture`   | 64×64 `image` + `Load…` button + size meta line |
| `array`     | `input_text` (CSV) |
| `text`      | `input_text` (unicode-decoded) |
| `buffer`    | `Randomize` button + size readout |

---

## 9. Render tab

Mostly preserved from `tab_render.py` + `widget_details.py`. Two structural notes:

1. **Quick-size buttons** become a wrapping row sized via `same_line()` rather than a fixed two-button layout.
2. **Preview thumbnail** stays at `SIZE.PREVIEW_W = 200 px` wide. Image size: `_fit_aspect(tex.size, max_w=SIZE.PREVIEW_W)`.

No behavior changes; just style flows from `apply_theme`.

---

## 10. Share tab

Preserved layout from `tab_share.py`. Two visual upgrades:

1. **Auth status line** uses `COLOR.STATE_OK / WARN / ERROR` from theme — no more hardcoded `(0.0, 1.0, 0.0)`.
2. **Telegram sticker grid** thumb size is `SIZE.TG_THUMB_H = 90 px`; selected slot uses an inset `COLOR.ACCENT_PRIMARY` border via `push_style_color(Col_.border, …)` + `ChildFlags_.borders` — replacing today's "tint the entire button green" trick (pain point #7 — selection is now a 1-px accent border, much clearer at small sizes).

---

## 11. Status bar (NEW · 24 px, pinned bottom)

Replaces the old "FPS readout in the top button row" + the "notification overlay covering the menu bar" anti-patterns.

```python
def _draw_statusbar(app):
    # pinned to the bottom edge of the main window
    main_size = imgui.get_window_size()
    imgui.set_cursor_pos((SPACE.MD, main_size.y - SIZE.STATUSBAR))
    imgui.push_style_color(imgui.Col_.child_bg,    COLOR.BG_FRAME)
    imgui.push_style_color(imgui.Col_.border,      COLOR.BG_SURFACE)
    with imgui_ctx.begin_child("statusbar",
                               size=(-1, SIZE.STATUSBAR),
                               child_flags=imgui.ChildFlags_.border_top):
        # ---- left side: state · node · resolution · fps · GLSL ver ----
        _dot(_status_color(app)); imgui.same_line()
        imgui.text_colored(_status_color(app), _status_text(app))
        _sep()
        imgui.text_colored(COLOR.FG_DIM, "node");    imgui.same_line()
        imgui.text_colored(COLOR.FG_PRIMARY, current_node_name(app))
        _sep()
        imgui.text_colored(COLOR.FG_PRIMARY, _resolution_str(app))
        _sep()
        imgui.text_colored(COLOR.FG_PRIMARY, f"{app.global_fps:.1f}"); imgui.same_line()
        imgui.text_colored(COLOR.FG_DIM, "fps")
        _sep()
        imgui.text_colored(COLOR.FG_DIM, "GLSL");  imgui.same_line()
        imgui.text_colored(COLOR.FG_PRIMARY, "4.60 core")
        # ---- toast slot ----
        if (toast := app.notifications.head):
            imgui.same_line(SPACE.LG)
            _draw_toast(toast)
        # ---- right side: shortcut legend ----
        imgui.same_line(imgui.get_content_region_avail().x - 460)
        _shortcut_legend()
    imgui.pop_style_color(2)
```

### 11.1 Notifications · status-bar inline (replaces overlay)

`Notifications.push(text, color)` no longer draws a top-right overlay. Instead, the head item from the stack renders inline in the status bar's `_draw_toast` slot. The stack still TTL's items at 5 s. Old behavior preserved at the API level — only the render site moves. This solves pain point #6.

### 11.2 Shortcut legend (always-visible cheat sheet)

```python
def _shortcut_legend():
    for label, kbd in [
        ("new",      "⌃N"), ("save", "⌃S"),
        ("edit",     "⌃E"), ("del",  "⌃D"),
        ("nav",      "← →"), ("setup","⌥S"),
    ]:
        _kbd_chip(kbd)
        imgui.same_line(); imgui.text_colored(COLOR.FG_DIM, label)
        imgui.same_line(SPACE.MD)
```

`_kbd_chip(text)` is custom: a small bg0_h-filled rect with a bg2 border drawn via `get_window_draw_list()` + `add_rect_filled` + `add_rect` + `add_text`. ~14 px tall, snug.

This solves pain point #5 (invisible help). The longer cheat sheet (with `←` `→` `⎋` etc) remains accessible via the `?` icon next to the node strip — see §6 strip-actions.

---

## 12. Modals

`Settings` (`Alt+S`) and `Node Creator` (`Ctrl+N`) — preserved from today's `popup_*.py`, restyled via the theme:

- **Title bar** of the modal is now visible (`PopupBorderSize=1`, `WindowRounding=4`).
- **Veil** uses `Col_.modal_window_dim_bg = (0, 0, 0, 0.55)`.
- **Node Creator** keeps the 3-column template grid; thumbs at `SIZE.THUMB_LG = 150 px`. Selection ring color updated to `COLOR.ACCENT_PRIMARY`.

No structural changes — drop in `apply_theme()` and they look correct.

---

## 13. Custom-draw effects (cataloged)

These are NOT stock widgets — each requires `ImDrawList` work. Carry them over from today or add new:

| Effect                                | Where     | Note                                                                       |
|---------------------------------------|-----------|----------------------------------------------------------------------------|
| Render empty-state title + subtext    | §4.2      | Replaces today's single-line `add_text`. Two lines, two fonts.            |
| Render error overlay                  | §5        | **Removed**. Errors render as a banner widget now, not custom text.        |
| Status-bar kbd chips                  | §11.2     | New. `add_rect_filled` + `add_rect` + `add_text`.                          |
| Node thumb texture into rect          | §6.1      | Replaces `image_button` to allow texture + label + border in one child.    |
| Slider value-fill (optional polish)   | n/a       | The accent-alpha fill behind the grab on horizontal sliders. CSS in proto. |

**Optional polish — slider value-fill**: in CSS the slider draws an `--accent-alpha` rect from 0 to grab-position. ImGui doesn't do this natively. Cheap to add via a `WindowContextHook` or per-slider with a manual `get_window_draw_list().add_rect_filled(...)` overlay. **Defer**; only ship if it doesn't break perf.

---

## 14. Tweaks panel

The in-app Tweaks panel (registered via the existing toolbar toggle pattern) lets the user swap accent / density / rounding / editor-side at runtime. The panel state lives in `app_state.json`:

```python
class AppStateTweaks(BaseModel):
    accent:   AccentName   = "yellow"
    density:  DensityName  = "tight"
    rounding: RoundingName = "subtle"
    side:     Literal["left", "right"] = "left"   # which half holds the editor
```

When the user changes a value:

```python
def _on_tweak_changed(app, key, value):
    setattr(app.app_state.tweaks, key, value)
    apply_theme(imgui.get_style(),
                accent=app.app_state.tweaks.accent,
                density=app.app_state.tweaks.density,
                rounding=app.app_state.tweaks.rounding)
    # editor side is layout-level, not theme-level — re-evaluate split next frame.
    app.save_app_state()
```

The Tweaks panel in `prototype.html` is the visual reference for control shape (segmented radios, color swatches, named option chips).

---

## 15. Adoption sequence (suggested PRs)

A single PR is doable but big. Suggested split for review-ability:

1. **PR 1 — theme + tokens**: drop in `theme.py` + `tokens.json`, call `apply_theme()` once at startup. App looks gruvbox; nothing else changes. Pixel-by-pixel iterating opportunity.
2. **PR 2 — token sweep**: replace every hardcoded color/size in the codebase with `COLOR.*` / `SIZE.*` lookups. No visual change; massive maintainability gain (pain points #1, #7).
3. **PR 3 — status bar**: add the new bottom strip; move FPS readout + notifications into it. Topbar shrinks. Notification overlay deleted.
4. **PR 4 — wide layout**: switch glfw window to full-monitor; introduce the 50/50 split; left half hosts a placeholder for the embedded editor (just a "coming soon" panel for this PR).
5. **PR 5 — embedded GLSL editor**: integrate `imgui_color_text_edit`; wire syntax palette to `COLOR.SYN_*`; replace the "Edit code (⌃E)" external-editor spawn with the embedded one (external retained as "Pop out").
6. **PR 6 — node tab restructure**: collapse the two-child layout into the single-list + type-pill design. Drop `selected_uniform_settings` child window.
7. **PR 7 — error banner**: replace overlay `add_text` with the inline banner; parse GLSL errors for click-to-jump in the embedded editor.
8. **PR 8 — Tweaks panel**: register the runtime tweaks UI; persist in `app_state.json`.

Each PR is independently shippable. PR 1 alone gives the user the gruvbox look immediately; everything else is incremental polish.

---

## 16. Open questions for the developer

1. **Editor library**: `imgui_color_text_edit` is the obvious pick (ships with imgui-bundle). Anyone has a preference for `ImGuiColorTextEdit`'s newer fork (`SanderMertens/ImGuiColorTextEdit`)? The API is broadly compatible.
2. **Multi-file editor**: a node has one `.frag.glsl` today. Eventually a vertex shader / `node.json` / multiple passes? If yes, the editor needs a tab strip. Out of scope for v1.
3. **Render output area split**: when rendering at 1280×960, the render image becomes the visual focal point. At lower resolutions (e.g. 256×256) the canvas has lots of dead space. Want a `Fit / Actual / Fill` toggle in the render card? (Easy to add, just `_fit_aspect` mode selector.)
4. **`hello_imgui.apply_theme` registry**: if you'd rather register gruvbox as a named theme in `hello_imgui`'s table, we'll need a C++-side patch. Worth doing? Or is calling `apply_theme()` from our own module sufficient?

When you have answers, drop them in a follow-up issue/comment. Otherwise we proceed with the sensible default for each.

---

## 17. Reference

- gruvbox palette: https://github.com/morhetz/gruvbox
- imgui-bundle 1.92 docs: https://pthom.github.io/imgui_bundle/
- Dear ImGui 1.92 `ImGuiCol_` enum: https://github.com/ocornut/imgui/blob/v1.92.0/imgui.h (search `ImGuiCol_Tab`)
- `imgui_color_text_edit` bundled module: `imgui_bundle.imgui_color_text_edit`
- Theme study: `theme-study.html` (palette mapping rationale)
- Visual source-of-truth: `prototype.html`
