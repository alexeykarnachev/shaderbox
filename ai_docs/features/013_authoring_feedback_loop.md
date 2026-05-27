# 013 — Authoring feedback loop

Make the write→save→compile-fail-or-render→find-the-mistake→fix loop tighter. Two layers, both fire
**only on explicit save/compile** (no live/debounced reload — explicitly rejected: compile isn't
free, the debounce can't know when you're done, most fires hit broken intermediate code). Grounded in
a parallel brainstorm (three agents) + direct API verification against the bundled
`imgui_color_text_edit` binding.

## Goal

1. **Layer 1 — Error experience.** Replace the unreadable raw driver string painted over the dimmed
   render with: (a) a parsed, themed error strip at the **bottom of the editor pane** (left split),
   one row per error; (b) native gutter markers on the offending lines; (c) click an error row →
   jump the editor caret to that line; (d) keep the last-good render **bright** (it's still bound —
   `render()` early-returns on compile failure, keeping the previous program) instead of dimming it
   to 20% and overlaying red text.
2. **Layer 2 — Uniform ⇄ code bridge.** (a) Click a uniform's name label in the node panel → jump
   the editor caret to that uniform's declaration line. (b) Hover a uniform identifier in the editor
   → tooltip showing its live runtime value (the value `_format_auto_value` already formats in
   `tabs/node.py`).

The two layers share one piece of plumbing: an `App`-level "editor jump request" consumed in
`tabs/code.py::draw` (the established cross-pane request pattern, mirroring
`editor_defocus_requested`).

## Out of scope

- **Live / debounced hot-reload** (`set_change_callback`). Rejected by the user, not deferred — do
  not file a trigger. Ctrl+S stays the explicit build signal.
- **Gruvbox syntax palette.** Re-verified impossible in this binding (2026-05-27): `set_palette`
  rejects `list[int]` (`TypeError`); `Palette` is an opaque C++ object with `.get()` only, no
  `__setitem__`/`__len__`/list ctor. The existing `todo.md` + `conventions.md ## Known quirks` note
  is correct and stays. Do not attempt.
- **Editor view-options dump** (minimap, folding, whitespace toggles beyond what's wired). Judged
  box-ticking for a 60-line shader; every new setter widens the FPE-on-setter-while-modal surface.
  Not in this feature.
- **Find/replace, multi-cursor, move-lines, comment-toggle** — editing-velocity features; a stuck
  shader author is not slow, they're stuck. Not in this feature.
- **GL warnings / runtime (non-compile) error surfacing** — Layer 1 covers compile errors only.
  Trigger for later: first time a "compiles but renders black" debug aid is wanted.
- **Builtin (non-uniform) hover signatures** — Layer 2 hover covers uniforms only (we have their
  live values; a GLSL-builtin signature dict is separate scope). Trigger: if hover proves useful and
  users ask for builtin help.

## Design decisions (locked)

1. **Error parsing is a pure free function in `util.py`, fully unit-tested.** Signature roughly
   `parse_shader_errors(raw: str) -> list[ShaderError]` where `ShaderError` is a small frozen
   dataclass `(line: int, message: str)` with `line` **0-based** (the `-1` from the 1-based driver
   line is baked in here, once). Two regexes: NVIDIA `^\d+\((\d+)\)\s*:\s*(.*)$` and Mesa/Intel/AMD
   `^(?:ERROR|WARNING):\s*\d+:(\d+):\s*(.*)$`. **If neither matches any line, fall back to one
   `ShaderError(line=-1, message=<whole raw string>)`** so the error is never hidden — a `line < 0`
   entry renders in the strip but places no gutter marker and is not clickable.
2. **No source-offset correction needed.** Verified: `core.py` passes `self.fs_source` to moderngl
   **verbatim** (no prepended `#version`/header — the `#version` is inside the user's file). So the
   driver line maps 1:1 to the editor line (modulo the 1→0 base conversion in decision 1). If
   `core.py` ever starts wrapping the source, this decision's premise breaks — pin it with a comment
   at the parse site and a test asserting the 1:1 mapping.
3. **Markers + cursor jumps are applied in `tabs/code.py::draw`, gated behind `not
   any_popup_open()`,** never while a popup is open (setters corrupt glyph metrics → next-render FPE;
   `conventions.md ## Known quirks`). The editor is already not drawn under a popup (`code.py`
   early-returns), so this code naturally sits after that guard. `set_cursor`/`scroll_to_line` only
   take effect on the next `render()` — that's fine, the editor redraws every frame. Ordering per the
   stub: `set_cursor(DocPos(line, index))` **then** `scroll_to_line(line,
   TextEditor.Scroll.align_middle)` (the `Scroll` arg is required, no default; `scroll_to_line`
   cancels `set_cursor`'s implicit scroll, so both are needed in that order), both before `render()`.
4. **Markers are re-derived UNCONDITIONALLY every frame** (clear + re-add) from `(current editor
   identity, its node's `shader_error`)`, NOT gated on a `shader_error` *change*. Rationale:
   `core.py::render()` dedups (`if err != self.shader_error`), so `shader_error` only *changes* when
   the text changes — a change-gated apply would miss the node-switch case (switch to a node whose
   unchanged error should re-show its markers, or to a clean node that must show none). Clear+re-add
   is cheap (the spec's own premise). Per error line: `add_marker(line, line_number_color,
   text_color, line_number_tooltip=msg, text_tooltip=msg)`; `clear_markers()` first each frame.
   A `ShaderError` with `line < 0` (unparseable fallback) places NO marker. (`ErrorMarkers =
   Dict[int,str]` stub alias is a red herring — no method consumes it; use per-line `add_marker`.)
5. **The error strip is an inline child at the bottom of the editor pane, NOT an `App` popup.** A
   popup would hit the at-most-one-popup invariant AND the editor-disappears-while-popup-open rule —
   the editor would vanish exactly when you want to read the error against the code. It's drawn in
   `tabs/code.py` below the editor child, themed via `COLOR.STATE_ERROR`, collapsing to nothing when
   clean. **Height is derived, not magic:** strip height = `min(error_count, MAX_ERROR_ROWS) *
   imgui.get_text_line_height_with_spacing()` (+ padding), with a "+N more" line when
   `error_count > MAX_ERROR_ROWS` (`MAX_ERROR_ROWS = 3`, a named constant). **Critically, the editor's
   own `render(size=...)` must shrink first:** compute the strip height, subtract it from
   `editor_size.y` (which today consumes all of `get_content_region_avail()`), THEN render the
   editor, THEN draw the strip in the reclaimed space — else the strip overflows the pane / the
   editor covers it.
6. **Render-area change: stop dimming + stop the over-image `add_text`.** In `ui.py::_draw_app_panel`
   the render image's `tint_col` becomes unconditional `(1,1,1,1)` (bright last-good frame) and the
   `draw_list.add_text` error overlay is dropped — the error now lives in the editor-pane strip
   (decision 5). **This removes the only remaining consumer of the local `has_error`, so delete that
   variable** (ruff would flag it dead). `node.shader_error` itself stays on `Node` and is still read
   by `widgets/node_grid.py` (the node-grid red error border — unchanged) and by `tabs/code.py` (the
   new strip/markers). The render pane becomes a pure live reference.
7. **Click-to-jump from a uniform attaches to the uniform NAME label** (`widgets/uniform.py`'s name
   column — currently a plain `imgui.text_colored(COLOR.FG_DIM, name)` between `same_line(_NAME_X)`
   and `same_line(_CTRL_X)`). Replace it with a **SIZED** clickable region (sized
   `imgui.selectable(name, False, size=(SIZE.UNIFORM_NAME_W, 0))` or `invisible_button` + hover-tinted
   text) so the `same_line(_CTRL_X)` column math still holds — an UNSIZED selectable spans full width
   and breaks the row. Extract it as a shared `ui_primitives` helper (e.g. `clickable_label`, mirroring
   `draw_link`/`draw_copyable_text`) — NOT a button (keeps the calm row layout), NOT inlined (no-dup
   rule). Hover affordance is a `COLOR` token swap (e.g. `FG_DIM`→`FG`/`ACCENT` on `is_item_hovered`),
   never a size/underline change (jitter-free rule). Clicking sets the `App` jump request to the
   declaration line. **Declaration line is found by `find_uniform_declaration_line(source, name) ->
   int | None`** scanning the editor text for the `uniform <type> <name>` occurrence; if `None` (an
   engine/auto uniform with no explicit declaration) the click is a no-op.
8. **Hover tooltip is a passive cursor-following `imgui.set_tooltip`, drawn in `tabs/code.py` after
   `render()` — NOT the editor's `set_text_hover_callback`.** (The callback approach was tried and
   reverted in manual-verification — the editor opens its hover popup for EVERY token and positions
   it over the code, occluding what's beneath; the position isn't controllable from Python. See
   Review history.) Shipped shape: gated on `cursor_over_editor and
   editor.is_mouse_pos_over_glyph(get_mouse_pos())`; read the word via
   `editor.get_word_at_mouse_pos(get_mouse_pos())`; if it's a key of the current node's
   `uniform_values`, `imgui.set_tooltip(f"{name}: {format_auto_value(value)}")`. Non-uniform words →
   no tooltip at all (no popup is ever opened, so there's no empty-box problem). The same block sets
   `app.code_hovered_uniform` for the reverse bridge (decision 11).
9. **`format_auto_value` MUST move from `tabs/node.py` (private `_format_auto_value`) to `util.py`**
   (drop the `_` prefix; `tabs/node.py` imports it back). Forced, not optional: the hover callback
   lives in `app.py::get_editor`, and `app.py` importing from `tabs/node.py` would invert the
   dependency (tabs import `App`) → cycle, which the no-`TYPE_CHECKING` rule bans. `util.py` is a
   leaf `app.py` already imports. No duplication — single home, both call sites import it.
10. **Cross-pane request plumbing mirrors `editor_defocus_requested`.** Add `App.editor_jump_request:
    tuple[int, int] | None` semantically `(line, index)` (0-based, `DocPos`-compatible — the consumer
    builds `TextEditor.DocPos(line, index)`; do NOT name a `.col`). Set by the uniform-name click (in
    `widgets/uniform.py` draw) and consumed + cleared in `tabs/code.py::draw` AFTER `render()` — same
    shape and lifecycle as the existing defocus signal (latched this frame, executed by the editor
    next frame; clearing this frame is safe). Cleared-after-consume means a single click is a single
    jump (no sticky re-scroll fighting manual scrolling). No new module, no cycle.
11. **Bidirectional bridge — hover-highlight, both directions** (added during manual-verification, by
    request; see Review history). Two more transient `App` fields, each set every frame and read with
    no stale residue:
    - **panel → code:** `App.editor_hover_line: int | None` — hovering a uniform's name cell
      (`is_item_hovered` in `widgets/uniform.py`) sets the declaration line; `_apply_markers` folds it
      into the per-frame marker pass as a translucent accent gutter wash (distinct from the red error
      wash), cleared after consume in `tabs/code.py`. A *gutter marker*, NOT `select_line` — a
      per-frame `select_line` would hijack the user's caret. 1-frame lag (panel draws after the code
      tab). The error/hover marker fills are translucent (`fade(...)`) so glyphs read through.
    - **code → panel:** `App.code_hovered_uniform: str` — the hover block (decision 8) sets the
      hovered uniform name; the panel paints that row's `clickable_label` with the SAME translucent
      accent wash the gutter uses (`highlight` arg → the selectable's `Col_.header` background, NOT a
      text tint — same visual language as the panel→code gutter mark, jitter-free). Reset at the top
      of `tabs/code.py::draw` (before any early return, so a popup frame can't leave it stale); read
      SAME frame (code tab draws before the panel — no lag).
    Recoloring all uniform *tokens* in the editor body was ruled out: the palette is write-locked in
    this binding (re-confirmed against upstream `pthom/ImGuiColorTextEdit` — no palette-write commit as
    of 2026-05-27; `todo.md` gruvbox-palette deferral watches for it).
12. **Error-loop polish on the same plumbing** (post-review, by request): (a) **F8 jumps to the next
    error** — `next_error_line(errors, caret)` (pure, wraps, skips `line < 0`) feeds the existing
    `editor_jump_request`; gated on no-popup. (b) **"compiled" cue** — `STATE_OK` text in the header
    when the editor is clean (`not dirty`) and `shader_error == ""` (the three header states —
    unsaved / compiled / errored — are mutually exclusive; the errored state is carried by the strip).
    (c) **error-count header** — `"N errors  (F8: next)"` atop the strip when `n > 1`, with the
    strip-height row budget incremented; each strip row's selectable id is `##err{i}` (byte-identical
    rows would otherwise collide).

## Files touched

- `shaderbox/util.py` — `ShaderError` dataclass + `parse_shader_errors(raw) -> list[ShaderError]`;
  `find_uniform_declaration_line(source, name) -> int | None` (decision 7); `next_error_line(errors,
  after_line) -> int | None` (F8 cycle, decision 12); **`format_auto_value` moved here** from
  `tabs/node.py` (decision 9).
- `shaderbox/hotkeys.py` — F8 → `_jump_to_next_error(app)`: parse the current node's `shader_error`,
  `next_error_line` after the caret, set `editor_jump_request` (decision 12).
- `shaderbox/tabs/code.py` — error strip draw below the editor (shrink `editor_size.y` first,
  decision 5); apply/clear gutter markers (error + hover line) unconditionally each frame with
  translucent `fade(...)` fills (decisions 4, 11, after the existing popup guard); consume + clear
  `editor_jump_request` after `render()` (decision 10); the passive `set_tooltip` hover block +
  setting `code_hovered_uniform` (decisions 8, 11); reset `code_hovered_uniform` at the top of `draw`;
  the strip's error-count header + the header "compiled" cue (decision 12).
- `shaderbox/ui.py` — `_draw_app_panel`: unconditional bright `tint_col`, drop the over-image
  `add_text`, **delete the now-dead `has_error` local** (decision 6).
- `shaderbox/app.py` — three transient fields: `editor_jump_request` (decision 10), `editor_hover_line`
  + `code_hovered_uniform` (decision 11). No hover callback in `get_editor` (decision 8 — reverted).
- `shaderbox/widgets/uniform.py` — uniform name column becomes a sized `clickable_label`; click → jump
  request, hover → `editor_hover_line`, `highlight` arg lit by `code_hovered_uniform`
  (decisions 7, 11), all via `find_uniform_declaration_line`.
- `shaderbox/ui_primitives.py` — new `clickable_label` helper (decisions 7, 11; sized, hover-tinted,
  optional `highlight`, jitter-free).
- `shaderbox/tabs/node.py` — import `format_auto_value` from `util.py` instead of the local
  `_format_auto_value` (decision 9).
- `shaderbox/theme.py` — only if a hover-tint `COLOR` token or an error-strip token is genuinely
  missing (prefer existing `FG`/`ACCENT`/`STATE_ERROR`; avoid new tokens). `MAX_ERROR_ROWS = 3` lives
  as a module constant near its use, not necessarily a theme token.
- `tests/test_util.py` — `parse_shader_errors`: NVIDIA format, Mesa format, multi-line,
  unparseable-fallback (`line == -1`), the 1→0 base conversion (driver line 20 → `ShaderError.line ==
  19`); `find_uniform_declaration_line`: hit, miss (`None`), multiple-uniforms-pick-the-right-one.
- `ai_docs/` — flip roadmap row 013 + banner; **delete** the resolved `todo.md` `[DEFERRAL] inline
  shader-error display (replace raw add_text overlay)` entry in the impl commit.

## Manual verification (in the app — handed to maintainer)

1. Introduce a syntax error at a KNOWN line (e.g. drop a `;` on line 20), Ctrl+S → error strip
   appears at the bottom of the editor pane with a readable `Line 20 · <msg>` row; the gutter marker
   is on the **exact** offending line (line 20, not 19/21 — pins the 1→0 conversion); **the render
   stays bright** (last-good frame, not dimmed/overlaid).
2. Click the error row → caret jumps to the offending line and scrolls it into view.
3. Fix the error, Ctrl+S → strip + markers clear; render updates; a `STATE_OK` "compiled" cue
   appears in the header (replaces the "(unsaved)" slot — clean + no error).
4. Multi-error case (two mistakes on different lines) → one row + one marker each; the strip shows a
   "N errors  (F8: next)" header. Press F8 repeatedly → caret cycles through the error lines (wraps).
5. Garbage/unparseable driver string (force one if possible) → falls back to the raw string in the
   strip, **the fallback row is NOT clickable** (clicking is a no-op, no jump to line -1, no crash).
6. Click a uniform's name in the node panel → caret jumps to its `uniform ...` declaration. Then
   click a uniform with NO explicit declaration (an auto/engine uniform like `u_time`) → no jump, no
   crash (`find_uniform_declaration_line` returned `None`).
7. Hover a uniform identifier in the editor → a cursor-following tooltip shows `name: value` (live)
   AND its panel row name tints accent; hover a non-uniform word → no tooltip, no tint.
8. Open a popup (Settings) while an error is showing → editor + strip hide (existing FPE guard), no
   crash; hover where a uniform was → no tooltip / no FPE (the hover callback can't fire, editor not
   drawn); close → editor + strip reappear with markers intact.
9. Switch nodes: error on node A, switch to a CLEAN node B → B shows NO markers/strip (markers follow
   the current editor, not stale A); switch back to A → A's markers re-show.
10. **Sticky-jump check:** after a jump (error-row OR uniform-name click), manually scroll the editor
    away → the caret does NOT snap back next frame (jump request was cleared after consume).
11. `make smoke` passes (headless frame loop — confirms the new `editor_jump_request` field + the
    hover-callback registration don't break the headless run); `make check` green.

## Open questions for the user — RESOLVED (locked 2026-05-27, all leans accepted)

1. **Error strip height/behavior:** grow-to-fit, capped at ~3 rows then a "+N more" line.
2. **Hover tooltip:** `name: value` only (no type readout). Type can be added later if wanted.
3. **Clickable uniform name:** subtle hover-highlight on the name for discoverability; no permanent
   underline (keeps the row calm).

## Review history

- 2026-05-27: plan locked with the user, all three open questions resolved to their leans.
- 2026-05-27: two pre-implementation reviewers (correctness-&-design, verification-&-blast-radius),
  both **PARTIAL** — no locked decision wrong, but a cluster of impl-time gaps. All real findings
  folded into the decisions above:
  - **Hover callback was mis-specified vs. the binding** (both reviewers, critical). `PopupData`
    carries only `pos: DocPos`; the callback populates an editor-provided popup, not a `set_tooltip`.
    Rewrote decision 8 to the `pos`/`get_line_text`/populate-popup idiom; closes over the editor's
    own `node_id`.
  - **`format_auto_value` move is forced** (cycle), not optional → locked in decision 9.
  - **Marker re-derive must be unconditional** (core.py dedups `shader_error` → change-gated apply
    misses node-switch) → decision 4 rewritten.
  - **Clickable name must be sized** (unsized selectable breaks the column math) + extracted as a
    `ui_primitives.clickable_label` → decision 7.
  - **`has_error` local goes dead** in `ui.py` after the unconditional tint → decision 6 + files list.
  - **Editor `render(size=...)` must shrink by strip height first** → decision 5.
  - **`scroll_to_line` needs the `Scroll` enum arg; jump-request is `(line, index)`/`DocPos`, not
    `.col`** → decisions 3, 10.
  - Manual-verification list expanded: exact-line marker, unparseable-row-not-clickable, no-declaration
    uniform no-op, hover-under-popup no-FPE, clean-node-B-no-markers, sticky-jump check.
  - Cadence bumped to **upper-mid**: add a spec-fidelity post-impl reviewer (8 modules + two layers).
  - Rejected false positives: error-row draw need NOT be a `ui_primitives` helper (single call site);
    no dead theme token / dangling import from decision 6 (`STATE_ERROR` has many users, node-grid
    keeps `shader_error`).
- 2026-05-27: manual-verification round 1 surfaced two bugs, both fixed:
  - **Empty hover popup on every token** — the editor opens its hover popup (`BeginPopup`, id
    `TextHoverPopup`) for ANY token, not just uniforms; drawing nothing left an empty box. Fix:
    `imgui.close_current_popup()` for non-uniform words (`app.py::_make_uniform_hover`).
  - **Jump didn't highlight the target line** — `set_cursor` moved the caret but nothing was
    visibly selected, and a click outside the editor left it dimmed. Fix: `_consume_jump` now also
    `select_line(line)` + `set_focus()`, and the code pane un-dims for the jump frame
    (`tabs/code.py`).
- 2026-05-27: manual-verification round 2 — the editor's hover popup occluded the code beneath it
  (can't hover/click adjacent tokens; the C++ popup position isn't controllable from Python).
  **Reworked decision 8**: dropped `set_text_hover_callback` + the `_make_uniform_hover` callback +
  the `identifier_at` helper; the hover is now a passive cursor-following `imgui.set_tooltip` drawn
  in `tabs/code.py` after `render()`, gated on `cursor_over_editor` + `is_mouse_pos_over_glyph`,
  using the editor's `get_word_at_mouse_pos`, shown only for live uniforms. No popup is opened for
  non-uniforms (the empty-box problem is structurally gone), and the tooltip doesn't block the code
  underneath.
- 2026-05-27: added hover-to-highlight (user request) — hovering a uniform's name in the node panel
  marks its declaration line with a transient accent **gutter marker**, complementing the click-jump
  (which keeps its scroll+focus+select-line "fix"). New transient `App.editor_hover_line`, set by
  `widgets/uniform.py` on `is_item_hovered()`, folded into `_apply_markers` (accent, distinct from the
  red error markers), cleared each frame in `tabs/code.py` after consume (one-frame lag; code tab
  draws before the node panel). Gutter marker chosen over `select_line` for hover: a per-frame
  `select_line` would hijack the user's caret. Full-line highlight remains click-only.
- 2026-05-27: added the reverse bridge (user request) — hovering a uniform identifier in the code
  tints its panel-row name accent. New transient `App.code_hovered_uniform` (set in `tabs/code.py`'s
  hover block, reset at the top of its `draw` so it clears even on the popup early-return); the panel
  reads it SAME frame (code tab draws before the panel — no lag, unlike the panel→code direction).
  `ui_primitives.clickable_label` gained a `highlight` arg (accent text vs `FG_DIM`, color-only =
  jitter-free). Color-recoloring all uniform *tokens* in the editor was ruled out: the palette is
  write-locked in this binding and upstream `pthom/ImGuiColorTextEdit` has no palette-write commit as
  of the latest release (re-checked 2026-05-27) — `todo.md` gruvbox-palette deferral already watches
  for the write-path.
- 2026-05-27: four-agent gap-sweep review (correctness / architecture / UX / contrarian), all PASS on
  the code, no dead remnants found. Fixes applied: rewrote the stale locked decision 8 + roadmap to
  match the shipped `set_tooltip` reality (the contrarian + architecture reviewers caught the locked
  section still describing the reverted callback — append-rot); added a `format_auto_value` test.
  Then the UX reviewer's three follow-ups (originally deferred as new scope) were pulled in at the
  user's request → decision 12 (F8 next-error, "compiled" cue, error-count header); the deferral filed
  for them was deleted in the same wave.
