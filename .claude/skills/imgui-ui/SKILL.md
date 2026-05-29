---
name: imgui-ui
description: "ALWAYS use when doing ANY Dear ImGui (imgui-bundle) UI work — proactively, before writing draw code, not only when something is already broken. Use when: building or refining a UI feature; adding/editing/moving/removing a button, panel, widget, tab, popup, modal, context menu, control, or layout; theming or adjusting colors/sizes/spacing; refactoring UI code; touching any imgui draw function; OR reacting to a UI problem ('looks off', 'jitter', 'too bulky', 'misaligned', a crash in draw code, a screenshot of the running app). It carries the hard-won rules that prevent the usual back-and-forth — the button-tier system, jitter-free overlays, the SetCursorPos assert, font/emoji caveats, the no-screenshot iteration loop, layout principles, modal + right-click-context-menu patterns, and the version-pinned imgui-bundle workarounds. Read it at the START of UI work, not after a complaint."
user_invocable: true
---

# Building imgui UIs without the back-and-forth

This skill is the distilled memory of a long UI-polish session. It exists because imgui is an
**immediate-mode** toolkit with sharp edges that cause the same mistakes every time: full-width
button bulk, jittering grids, the SetCursorPos assert, frozen video previews, off-center glyphs. Read
the relevant section before writing draw code; you'll skip rounds of screenshot ping-pong.

Most of this is **imgui-generic** — reusable across any imgui project. The few project-specific facts
are quarantined in `## ShaderBox specifics` at the end; ignore that section in another codebase.

---

## 0. The iteration loop (read first if you can't see the window)

If the running app can't be screenshotted by the agent (no window manager on the display, a remote
box), **you are designing blind** and every visual call must go to a human. That loop is slow and
fails repeatedly. Defenses, in order of leverage:

1. **Verify headlessly — but know what headless CANNOT tell you.** Most "does it crash / does the data
   flow" questions don't need eyes. Drive the panel in a standalone GL+imgui context for a few frames
   (create a hidden glfw window, an imgui context, a backend, feed a fake state object, run
   `new_frame → draw → render` 3× inside a `begin/begin_child`). This catches the SetCursorPos assert,
   stack imbalance, missing-attr, and None-deref bugs before the human ever looks. **BUT focus / nav /
   layout-geometry state reads DIFFERENTLY headless than in the real (maximized, mouse-driven, settled)
   window** — `is_window_focused`, `io.nav_active`, the nav-cursor position, and `get_item_rect_*`
   after `end_child` can all report values headless that don't match the live app. A headless "PASS" on
   any of those is NOT verification — it lulls you into shipping a regression. For focus/nav/where-the-
   outline-lands, headless confirms only "no crash + the flag is set"; the actual behavior is a
   maintainer `make run` check. Don't iterate on focus/nav by trusting headless booleans. Reserve
   screenshots for genuine *aesthetic* calls.
2. **Anchor to a written spec, not taste.** For a non-trivial layout, get a numeric spec first
   (wireframe + token references + per-state coverage). "Center at x = label_w + 8" is implementable
   blind; "make it look nice" is not. A designer-style spec pass up front beats five reactive rounds.
3. **Change one structural thing at a time.** The failure mode is "fix the named complaint, break the
   global composition." When a fix's diff balloons, you're solving the wrong problem — re-derive the
   layout as a whole.
4. **When the human says it "jitters" / "is bulky" / "is a mess," it's usually one of the named
   failure modes below — not a mystery.** Match the symptom to the rule before experimenting.

---

## 1. The button-tier system (pick by role, never by look)

The #1 cause of "too bulky" / "it's a mess": every action rendered as a full-width filled bar, all
shouting equally. Mature button guidance (Carbon, etc.): in a layout with >3 actions, only the
primary is filled; the rest are low-emphasis. Encode this as a **named tier set in one utils module**
and forbid hand-rolling `push_style_color(Col_.button, …)` at call sites.

Four tiers, chosen by the action's *role*:

| Tier | Look | When |
|---|---|---|
| `primary_button` | filled accent, contrasting text | the ONE call-to-action of a section (e.g. "Add", "Create") |
| `button` | filled neutral grey (imgui default) | an ordinary, repeatable action (e.g. "Render") |
| `ghost_button` | transparent fill, secondary-colour text | low-emphasis / secondary ("New", "Cancel", "Re-render") |
| `danger_button` | transparent fill, error-colour text | destructive ("Delete") — prominence comes from a confirm step, NOT a filled-red fill |

Each is a 3-line wrapper: push 3-4 `Col_` slots, call `imgui.button(label, size=(width, 0.0))`, pop.
`width=0.0` ⇒ content-width (the default; use it). Reserve full-width for the single dominant CTA in a
genuinely narrow container — never a stack of full-width bars.

**Width policy:** content-width by default. To align several buttons' edges, compute widths from
`calc_text_size(label).x + 2 * pad` and a target row width; don't eyeball.

---

## 2. Layout principles (what makes a panel calm)

- **Hierarchy through difference, not uniformity.** One prominent thing per zone; everything else
  recedes. If two elements have equal visual weight, the eye has nowhere to land.
- **Content-width, not full-width.** A 700px-wide button in a side panel is the canonical bulk smell.
- **Whitespace separates zones; rules don't.** A stack of `separator` / `separator_text` lines under
  a tab bar reads as noise ("cascade of horizontal lines"). Prefer a `dummy(0, SPACE.MD)` gap. Use at
  most one titled separator per real section, and skip it when the context is already obvious (don't
  put a "Pack" heading above a "Pack [combo]" row — it says the word twice).
- **The label-control row.** `label` (dim) + `same_line(LABEL_W + gap)` + `set_next_item_width(W)` +
  the widget. Use a *fixed* label-column width across all rows so controls align into a column; add a
  small gap (`SPACE.MD`) between label and control so they don't touch. Factor this into one helper.
- **Don't strand elements in a wide void.** Centering an element in a very wide panel leaves it
  floating with dead space around it. Either left-anchor, or cap the content to a working column
  width, or fill the void with a related element (e.g. tuck a secondary control into the empty space
  beside a tall preview). A near-empty grid reads better as a row of *placeholder cells* (a visible
  backbone) than one lonely item.
- **Real-word labels, not cryptic glyphs.** There is usually no icon font loaded (see §5). `+` / `X`
  icon buttons were repeatedly rejected in favour of "New" / "Delete". Use words.
- **Empty states are composition, not a symbol.** A blank `[ ]` or `?` placeholder reads as a bug.
  Use a recessed/bordered box (a "slot") — its shape says "content goes here" without text.
- **One slot, not a stack, when space is tight.** A status line and a progress bar competing for the
  same row: show one OR the other in the same slot (`if in_flight: bar else: stats`), never stacked —
  there isn't room, and a conditionally-appearing line shifts everything below it.

---

## 3. Jitter-free overlays and grids (the hardest imgui trap)

Symptom: clicking between items in a grid makes the whole layout shift a pixel or two. Causes and the
durable fix:

- **A selection highlight must change *colour*, never *size*.** A thicker selected border must be
  drawn **inset** (inside the cell rect), never straddling the edge (which bleeds into the inter-cell
  gap and looks like motion).
- **An overlay control (a delete-✕ pinned to a tile corner, a badge) that uses
  `set_cursor_screen_pos` + a real `imgui.button` will perturb the parent window's content size — but
  only for the items that *have* the overlay** (e.g. the selected cell). That selection-dependent
  asymmetry IS the jitter. **Fix: wrap each such cell in its own `begin_child(cell_w, cell_h)`.** A
  child window has its own content region, so the overlay's absolute cursor moves stay inside it and
  can't touch the parent. This also fixes the SetCursorPos assert (§4).
- **For the overlay's click to win over the cell beneath it:** call
  `imgui.set_next_item_allow_overlap()` *before* the cell's (invisible) button, then draw the overlay
  button afterward at its absolute position.
- **A crisp ✕ / icon:** don't rely on a font glyph centered by button text-align (single chars sit
  off-center). Draw it with the draw list — two `add_line` calls for an ✕ — perfectly centered, no
  font dependency.
- **`is_mouse_hovering_rect` ignores window ordering and popup-blocking** (its own doc says so) — so a
  region's hover gate built on it stays "hovered" even when a menu dropdown / popup is open *on top* of
  it. Symptom: a custom cursor (or hover highlight) keyed to that rect fires through an open menu. Fix:
  AND it with `imgui.is_window_hovered(HoveredFlags_.child_windows)`, which *does* respect popup-blocking.

---

## 4. The SetCursorPos assert (a crash that masquerades as a sizing bug)

`set_cursor_screen_pos()` / `set_cursor_pos()` to a position *past the last submitted item* asserts:
*"Code uses SetCursorPos to extend window/parent boundaries. Please submit an item e.g. Dummy()
afterwards."* If a `try/except` around the frame swallows it, the imgui stack is left unbalanced and
you get a confusing **downstream** assert later (e.g. `IM_ASSERT(Size > 0)` at `end_child`). The real
fault is named in the FIRST imgui-error line, above the traceback — **read full stderr, don't chase
the downstream symptom.**

Rules to avoid it:
- After moving the cursor absolutely, **submit an item that covers the moved-to position** (a button,
  an `image`, a `dummy`).
- Better: confine absolute positioning to a **per-element `begin_child`** (see §3) so it can never
  extend the parent.
- To position a block absolutely and continue normal flow after, capture `get_cursor_screen_pos()`
  before, reserve the block with a `dummy`, then restore the cursor with `set_cursor_screen_pos`.
- **Prefer normal flow (`same_line`, `dummy`) over absolute positioning.** Reach for `set_cursor_*`
  only for genuine overlays, and contain them in a child.

> Related, same family: an `IM_ASSERT(Size > 0)` at a `begin`/`begin_child` `__exit__` is **almost
> never** a child-sizing bug — it's an exception thrown inside a per-frame draw fn (or a SetCursorPos
> extend) that left the stack unbalanced. Un-swallow the exception / read the imgui-error lines first.

---

## 5. Fonts, glyphs, previews

- **`imgui.text` / `text_colored` never wrap** — a long instruction sentence in a fixed-width
  container (a modal with `set_next_window_size`, a tree-node column) is **clipped at the right edge**,
  not folded, and (without the `horizontal_scrollbar` flag) shows no scrollbar — it just vanishes
  mid-word. Wrap it: `push_text_wrap_pos(0.0)` (0.0 = wrap at the content-region right edge) around the
  text, then `pop_text_wrap_pos()`. Standardize as a `wrapped_caption`-style primitive so call sites
  don't re-push. Measure long labels against fixed label-column widths too — a label wider than the
  column overlaps its control (see §2 label-control row).
- **No icon font unless you loaded one.** Don't design with ⚙/🗑/✏ as affordances. Use text.
- **Mixed fonts in one `imgui.button` label aren't possible** — a button label is one font. To show a
  glyph from a different font (an emoji) *as* a button, draw an empty/invisible button, then render
  the glyph centered over its rect via the draw list with the glyph font pushed.
- **`push_font` wants the rasterized size, not `get_font_size()`** — see §8 (it's a version-pinned
  imgui-bundle binding fact).
- **An image/preview inside a `begin_child` must fit the child's *content region*
  (`get_content_region_avail()`), not the child's outer size** — the outer size minus window padding
  is smaller, and fitting to the outer size overflows the content area → a spurious scrollbar and an
  offset look. Add `WindowFlags_.no_scrollbar` to a pure display box as belt-and-suspenders.
- **Video/animated previews need `.update(t)` every frame** before reading `.texture` — otherwise they
  show a frozen first frame. Easy to forget for thumbnails drawn in a loop.

---

## 6. Architecture / encapsulation (so the system stays reusable)

- **A UI sub-library lives in its own modules:** a `theme` (colour/size/spacing token bags — no
  hardcoded hex or magic px anywhere else) and a `ui_primitives` (the button tiers + shared draw
  primitives: copyable text, a labelled slider, an overlay close-✕, a caption text). Non-UI helpers
  live separately (a `util` module), never mixed in with the draw primitives. Everything visual flows
  through these. A token used by exactly one panel still belongs in the token bag, not inline.
- **Don't repeat a widget.** A draw block appearing twice (a copyable path, a styled button) is
  extracted to a `ui_primitives` free function and called from both. Two copies drift.
- **No `@staticmethod` for stateless helpers** — module-level free functions instead. (Generic
  code rule, not imgui-specific; full form in the project's conventions.)
- **Layering, strict:** generic UI primitives (theme/ui_primitives) know nothing about any feature; a
  feature-area module (e.g. a "sharing" abstraction) knows nothing about a specific implementation
  (e.g. "telegram"); the high-level app orchestrator knows nothing about either's internals. A
  feature panel should own its own draw code and receive what it needs via a small value/callback
  object — **never reach up into the app** (which also avoids import cycles). When you catch a
  specific implementation's concept leaking into generic code, that's the refactor signal.
- **Pass capability callbacks, not the app.** A panel that needs "open a picker" / "render this" /
  "draw an emoji glyph" gets those as callbacks in a dataclass, so it stays decoupled from the app and
  from threads it must not touch.

---

## 7. Modals and right-click context menus

The two heaviest patterns in a typical imgui app — every project ends up with a
"settings" / "picker" / "create" modal and at least one tree/list that wants
per-row actions. Most of the boilerplate is wrong by default; the rules below
make them consistent and de-bloat the call sites.

### 7.1 Modal chrome (close button, action row, primary)

**One close affordance shape across all modals — pick it once, don't drift.**
Recurring failure modes: raw `imgui.button("Close")` (violates §1 button tiers);
"Close" in one modal, "Cancel" in another for no semantic reason; close button
above the body in one modal, below it in another; primary action at fixed
width that overflows its label.

The convention:

- **Bottom action row.** The close button — and the primary action, when there
  is one — both sit in a row at the bottom of the modal. Never at the top,
  never floating mid-body.
- **Tier: always `ghost_button` for close, always `primary_button` for the
  one primary action.** Skip `imgui.button(...)` at modal call sites; that's a
  tier violation (§1).
- **Label: "Close" for browsers / view-only dialogs (settings, pickers).
  "Cancel" only for forms whose commit action mutates state** (e.g. a "Create"
  / "Save" flow whose Cancel is a meaningful "undo intent"). "Close" implies
  "dismiss the view"; "Cancel" implies "discard in-flight edits." If both
  read fine, prefer "Close" — it's the safer default.
- **Position: primary on the left of the action row, close on the right.**
  When there's no primary, the close button is alone in the row.
- **Primary width: content-width (no `width=` arg).** Fixed widths force a
  long label to overflow visually. Use a fixed width only when aligning a row
  of equal-tier siblings (rare).
- **Spacing above the action row: `imgui.dummy((0, SPACE.MD))`.** Not
  `imgui.new_line()` (magic line-height), not `imgui.spacing()` (too tight).
  One token for the visual rhythm.

### 7.2 Modal wrapper (kill the boilerplate)

**Popup modal size: always `Cond_.first_use_ever`, never `Cond_.appearing`.**
The first seeds the size once, then imgui.ini persists the user's manual
resize across re-opens. `Cond_.appearing` clobbers the saved size on every
reopen — and on return from a native file dialog — which reads as a visible
blink / reset.

Every modal otherwise repeats the same 8 lines: `is_popup_open` check →
`open_popup` → `set_next_window_size` → `begin_popup_modal` → visibility
guard → body call → close-flag write-back. **Wrap this once** and call sites
become 5 lines. Bonus: the `first_use_ever` rule is mechanically enforced
instead of being a copy-pasted comment.

Reference shape (used in `ui_primitives.modal_window`):

```python
@contextmanager
def modal_window(label: str, size: tuple[float, float]) -> Iterator[bool]:
    if not imgui.is_popup_open(label):
        imgui.open_popup(label)
    imgui.set_next_window_size(imgui.ImVec2(*size), imgui.Cond_.first_use_ever)
    with imgui_ctx.begin_popup_modal(label) as popup:
        yield popup.visible
```

The wrapper is the mandated shape — hand-rolling `begin_popup_modal` is the
violation. **The `is_X_open` flag stays on `App`** (not on the wrapper) so
each modal can do its own per-close cleanup (re-apply editor settings, null a
pick-target, reset a query buffer).

**`is_window_appearing()` is the first-frame signal** — but inside a
`begin_child` it returns the *child's* appearing state, not the modal's.
Cache the parent-modal's `is_window_appearing()` once at the top of the body
draw and pass it down (or stash it on App) — don't call it again from inside
a child window.

### 7.3 Modal body returns `bool keep_open`

Every `_draw_body(app) -> bool` returns True to keep the modal open, False to
close. The wrapper translates False → `is_X_open = False` +
`imgui.close_current_popup()` (plus any per-modal close cleanup). **Don't
invert the boolean** — `keep_open` reads better than `is_keep_opened` /
`should_close` / `keep_running`. One name, every modal.

Multiple close paths inside one body all set the same `keep_open = False` and
return. If a body needs branched cleanup (Settings's `apply_editor_settings()`
on close, Emoji picker's `emoji_pick_target = None`), do that **at the wrapper
call site after the body returns False**, not inside conditional cleanup logic
in the body — the body should describe what it draws, not what happens on
close.

### 7.4 Right-click context menus for per-row actions

The previous failure was a row of inline icon buttons per file/dir/function —
visually noisy, eats horizontal space, looks like slop. **For per-row tree /
list / grid actions, prefer a right-click context menu over inline buttons.**

- **When**: a row has more than ONE action (delete + rename + reveal, etc.);
  the actions are infrequent relative to the row's primary click; the action
  set might grow.
- **Discoverability**: show a one-line "Right-click for actions" hint above
  the list. No hover tooltip per row — the hint sets the affordance once.
- **Styling**: wrap the `begin_popup_context_item(...)` in
  `with context_menu_style():` (lighter fill + accent border + accent hover).
  Default popups use `popup_bg` which equals the picker modal's own `popup_bg`,
  so an unstyled context menu blends invisibly into the modal it floats over.
- **Mutual exclusion of in-flight actions**: arming a delete, opening a rename
  input, opening a new-file input — these are mutually-exclusive states. Add
  ONE reset-state method that every opener calls first; openers then set only
  their own field. Without this, two inputs can be live simultaneously, both
  grab keyboard focus on alternate frames, and `is_item_focused()` lies.
- **When NOT** to use a context menu: rows with a single common action
  (toggling a favorite — the inline star is fine), or rows where the primary
  click IS the action (selectable list of options). Two affordances for the
  same thing on the same row is the slop signal.
- **`imgui.menu_item_simple(label, enabled=False)` can still register a
  click** depending on the imgui-bundle version. Gate the action in Python
  (`if has_editor: do_action()`), don't rely on `enabled=` alone to suppress
  the call.

### 7.5 Inline inputs inside modals (rename / new-file / new-dir)

Inline inputs that replace a row when an action starts (Rename, New file):

- **Pattern**: `imgui.input_text(..., flags=enter_returns_true)` — Enter
  commits, Esc cancels. Always reserve an `x` cancel button on the right of
  the input row — Esc is invisible, the explicit cancel is the affordance.
- **Focus**: a one-shot `needs_focus` flag on a state object that the input's
  first draw consumes via `set_keyboard_focus_here(0)`. After that one frame,
  imgui keeps the focus where it is. Don't call `set_keyboard_focus_here`
  every frame — it fights other inputs.
- **Outer keyboard suppression**: when an inline input has focus, the modal's
  outer Enter (primary action) AND Esc (close) must be suppressed. Two ways:
  track `is_item_focused()` immediately after the input (the cleanest gate),
  or track a sticky "input owns keys this frame" flag on App that all inline
  inputs set during their draw and the outer handler reads before its Enter
  / Esc checks. Don't rely on imgui's auto-Esc-closes-popup — it triggers
  even when an input has focus.
- **Auto-expand the parent**: if the input renders inside a collapsible tree
  node, and a context menu opens the input on a node that's currently
  collapsed, the input is invisible. Force-open the ancestor chain via
  `set_next_item_open(True, Cond_.always)` when the input target is a
  descendant of the dir being drawn. Without this, "New file here" on a
  collapsed dir feels like a dead button.

### 7.6 State to reset on modal open / close

Two tedious-but-load-bearing housekeeping points that every modal eventually
gets wrong:

- **On open**: reset transient state from the previous session — search
  query, selection, in-flight inline-input state, armed-delete state. A
  picker that re-opens showing the previous search reads as broken
  ("why is my query still here?"). Centralize as one `reset_X_state()`
  method on App, call from `open_X()`.
- **On close**: null out callbacks / targets that point at caller state.
  A picker that holds onto its pick-target callback after close is a
  dangling reference to whoever opened it — usually fine, occasionally a
  bug. Clear it at the wrapper's close branch.

### 7.7 Gating a modal action on "was the source surface active?"

A modal often needs to know whether the surface it acts on (a code editor, a
canvas) was a real interaction target — e.g. "insert at the caret" only makes
sense if the user was actually editing. Three traps, in order of temptation:

- **A live `is_focused()` check reads False inside the modal** — the modal
  stole focus the instant it opened. Useless from within.
- **A snapshot-at-open of the source's focus is the tempting wrong answer** —
  it's also False, because by the time the open handler runs (e.g. from a menu
  click), focus already moved to the menu/modal. (We tried this; it's wrong.)
- **A "is there a target at all" proxy (a file is selected, a node exists) is
  too lax** — true before the user ever interacted, so the action fires into a
  caret at (0,0).

The right shape: a **sticky "was-ever-active" flag** the SOURCE maintains —
set True whenever the source gains focus, cleared only on explicit defocus
(Esc / nav-away / target switch). It survives the modal stealing focus, and
it's False when the user genuinely never touched the source. The modal gates
on that. (ShaderBox: `editor_was_ever_focused`, §9.)

---

## 8. imgui-bundle build quirks (version-pinned workarounds)

Library footguns specific to the imgui-bundle Python build (currently
1.92.801). Re-check on every bump.

- **`imgui_color_text_edit.TextEditor.Palette` is read-only from Python** —
  only `.get(color) -> ImU32`; no per-slot setter, no list-based constructor,
  and `set_palette()` accepts only a `Palette` object (unbuildable with
  custom colors). Use a built-in (`get_dark_palette()` /
  `get_light_palette()`); custom palettes are impossible until the binding
  exposes a write path.
- **Dear ImGui 1.92 dropped pre-baked glyph ranges + `refresh_font_texture()`** in
  favor of dynamic on-demand glyph loading
  (`BackendFlags_.renderer_has_textures`, set automatically by imgui-bundle's
  `BaseOpenGLRenderer.__init__`). `add_font_from_file_ttf(path, size_pixels=N)`
  is enough — no `glyph_ranges=` kwarg, no manual texture refresh. Non-ASCII
  glyphs load when text is first drawn.
- **`imgui.push_font` now requires `(font, size_base_unscaled)`** — pass the
  rasterized size (the one used in `add_font_from_file_ttf(size_pixels=...)`).
  Never pass `imgui.get_font_size()` — that's the *post-scaling* value and
  would scale twice. For an `ImFont` you hold, the rasterized size is
  `font.legacy_size`.
- **This imgui-bundle build renders MONOCHROME emoji only**.
  `NotoColorEmoji.ttf` loaded with the FreeType `LoadColor` flag rasterizes
  to *blank* glyphs in the glfw backend, even though the binary ships
  plutosvg. Use `NotoEmoji-Regular.ttf` (line-art). Don't re-attempt color
  without re-running the spike.
- **`imgui.image(...)` lost `tint_col` / `border_col` since 1.91.9**. For
  tint, switch to `imgui.image_with_bg(...)`. For border, push the
  `ImageBorderSize` style var (or live without).
- **imgui-bundle's `portable_file_dialogs` `pfd.open_file` /  `save_file` /
  `select_folder` are non-blocking class handles, not blocking functions.**
  Wrap with a `pfd_block(dialog)` helper that spins until `.ready(20)` so
  call sites read synchronous.
- **imgui-bundle's Python glfw backend (`python_backends/glfw_backend.py`)
  does NOT sync imgui's mouse cursor to the OS** — it never sets
  `BackendFlags_.has_mouse_cursors` and never calls `glfwSetCursor`. So
  `imgui.set_mouse_cursor(...)` is a silent no-op at the OS level. Create
  glfw cursors yourself (`glfw.create_standard_cursor(...)`) and call
  `glfw.set_cursor(window, cursor_or_None)`. Restore with
  `glfw.set_cursor(window, None)`.
- **`imgui_color_text_edit.TextEditor.render()` auto-grabs imgui keyboard
  focus on a child window's first frame** — so a never-yet-rendered editor
  (app open, or just-switched node) steals focus and the caret goes live
  without a click. The editor exposes no `is_focused()` getter. Track focus
  by reading `imgui.is_window_focused(FocusedFlags_.child_windows)` *after*
  `render()`. To programmatically defocus, set `editor_defocus_requested`
  and consume it with `set_window_focus(None)` AFTER `render()` — clearing
  before render is undone by the editor's own first-render grab.
- **imgui-bundle's C++-backed submodules ship only `.pyi` stubs**
  (`portable_file_dialogs`, `imgui_color_text_edit`) — pyright emits a
  `reportMissingModuleSource` warning at the import line. Harmless. Don't
  suppress with `# pyright: ignore` — that hides genuine resolve failures
  elsewhere.
- **`set_window_focus(name)` — the by-name string overload — SEGFAULTS**
  (hard crash, exit 139), regardless of where it's called. The no-arg
  `set_next_window_focus()` (called before the target `begin`/`begin_child`)
  works and is the documented-preferred form; `set_window_focus(None)` (defocus)
  also works. To programmatically focus a window/region, use
  `set_next_window_focus()` before it — and it correctly targets a *grandchild*
  `begin_child`, not just a top-level window (spike-confirmed). Never pass a name string.
- **An `invisible_button` is NOT a keyboard-nav stop** — with `nav_enable_keyboard`
  on, nav never lands on it, so Space/Enter can't activate it (you can't reach it
  without the mouse). A `selectable` (or a real `button`) IS a nav stop. For a
  whole-cell click target that must be keyboard-reachable, use a `selectable` with
  `SelectableFlags_.allow_overlap` + transparent `Col_.header*` (so the image/border
  carries the visual); overlay buttons drawn on top still win the click. (ShaderBox
  `ui_primitives.preview_cell`.)

---

## 9. ShaderBox specifics (ignore in other projects)

The only project-coupled facts, consolidated here:

- **Modules:** tokens in `shaderbox/theme.py` (`COLOR` / `SIZE` / `SPACE` bags + `apply_theme` +
  `fade`); primitives in `shaderbox/ui_primitives.py` (button tiers + shared draw helpers — read the
  file for the set); non-UI helpers in `shaderbox/util.py`. Theme is gruvbox-dark, accent-swappable at runtime.
- **Three-layer UI:** `app.py` (state/lifecycle, no drawing) / `ui.py` (frame-loop orchestrator) /
  `widgets`+`popups`+`tabs` (pure `draw(app)` free fns). Forced by the no-`TYPE_CHECKING` rule (a draw
  fn annotating `app: App` while `App` imports it would cycle). Full rules: `ai_docs/conventions.md`.
- **Exporters own their panel; talk via `RenderControl`.** A sharing exporter (telegram) draws its own
  operations panel and receives a `RenderControl` dataclass (callbacks + state) from the share tab —
  it must not import `App`. Worker-thread vs render-thread affinity is enforced in `exporters/base.py`.
- **imgui-bundle build quirks** — version-pinned workarounds live in §8 above (which carries the
  build version). Read §8 before working around a footgun yourself.
- **Two editor-focus flags on App:** `editor_focused` (live; flickers False whenever any popup /
  menu / picker steals focus) and `editor_was_ever_focused` (sticky; cleared only by explicit
  defocus — Esc / arrow nav / target switch). Gate "is the editor a real interaction target?"
  questions (Insert-at-caret in the lib picker, etc.) on the sticky one; the live flag reads
  False inside a modal, and `current_editor_path is not None` is too lax (a freshly-selected
  node has a session before any typing happened).
- **Can't screenshot the app on the dev box** (no WM on the display) — §0 applies hard here; hand
  visual checks to the maintainer. Verify everything else headlessly (`make smoke`, or a standalone
  GL+imgui driver). `make check` (ruff + pyright, 0 errors) gates every change; `x` and other
  ambiguous unicode trip ruff (RUF001/002) — use ASCII.

---

## Maintaining this skill

This is a living distillation. When a UI session surfaces a new durable imgui lesson (a trap, a
principle, a primitive worth standardizing), add it here — keep generic imgui rules in §1-§7,
imgui-bundle version-pinned workarounds in §8, ShaderBox-only facts in §9. Don't let it bloat into
per-feature minutiae. The bar: would this save a future session a round of iteration?
