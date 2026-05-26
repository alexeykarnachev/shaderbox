---
name: imgui-ui
description: "ALWAYS use when doing ANY Dear ImGui (imgui-bundle / pyimgui) UI work — proactively, before writing draw code, not only when something is already broken. Use when: building or refining a UI feature; adding/editing/moving/removing a button, panel, widget, tab, popup, control, or layout; theming or adjusting colors/sizes/spacing; refactoring UI code; touching any imgui draw function; OR reacting to a UI problem ('looks off', 'jitter', 'too bulky', 'misaligned', a crash in draw code, a screenshot of the running app). It carries the hard-won rules that prevent the usual back-and-forth — the button-tier system, jitter-free overlays, the SetCursorPos assert, font/emoji caveats, the no-screenshot iteration loop, and layout principles (content-width buttons, calm spacing, label-control rows). Read it at the START of UI work, not after a complaint."
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

1. **Verify headlessly.** Most "does it crash / does the data flow" questions don't need eyes. Drive
   the panel in a standalone GL+imgui context for a few frames (create a hidden glfw window, an imgui
   context, a backend, feed a fake state object, run `new_frame → draw → render` 3× inside a
   `begin/begin_child`). This catches the SetCursorPos assert, stack imbalance, missing-attr, and
   None-deref bugs before the human ever looks. Reserve screenshots for genuine *aesthetic* calls.
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
- **`push_font(font, size)`** wants the rasterized size (`font.legacy_size` for a held `ImFont`), not
  `get_font_size()` (post-scaling — would double-scale).
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
- **No `@staticmethod` for stateless helpers** — make them module-level free functions. A method that
  doesn't use `self` isn't a method.
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

## 7. ShaderBox specifics (ignore in other projects)

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
- **imgui-bundle build caveats** — this build bites in several places: monochrome emoji only; the
  glfw backend doesn't sync the OS mouse cursor; `image(...)` lost `tint_col`/`border_col`; the code
  editor auto-grabs focus on first render; `push_font` wants the rasterized size (`font.legacy_size`).
  Each has a workaround — the canonical home (with the version string, font filenames, and the exact
  fix) is `conventions.md ## Known quirks`; check there before working around one yourself.
- **Can't screenshot the app on the dev box** (no WM on the display) — §0 applies hard here; hand
  visual checks to the maintainer. Verify everything else headlessly (`make smoke`, or a standalone
  GL+imgui driver). `make check` (ruff + pyright, 0 errors) gates every change; `×` and other
  ambiguous unicode trip ruff (RUF001/002) — use ASCII (`x`).

---

## Maintaining this skill

This is a living distillation. When a UI session surfaces a new durable imgui lesson (a trap, a
principle, a primitive worth standardizing), add it here — keep it generic, keep ShaderBox-only facts
in §7, and don't let it bloat into per-feature minutiae. The bar: would this save a future session a
round of iteration?
