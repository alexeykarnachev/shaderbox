# 087 — Eighth walk findings

The maintainer's eighth walk over the running app, four findings. Two land here; two are editor-repo
work re-vendored in (the lexer and the search-highlight emitter both live inside `libeditor`). The
fifth item of the same batch, the frame-time breakdown behind the FPS chip, is a mechanism with its
own extensibility requirements and is feature 088.

Source: the maintainer's `../TODO` batch, verbatim under each workstream.

Size: **mid**. Four workstreams over two repos, one new primitive, one new token, one re-vendor.

---

## Goal

- **W-A — the pass strip's tiles grow 1.5x.** "since we now have a large area of the document tab
  (we separated it from the uniforms recently), now we can make the passes previews a little bit
  larger. I think we can try 1.5x. Make sure that they warps correctly and not overflow the right
  border of the tab"
- **W-B — the Uniforms tab's pass selector becomes a row of clickable names.** "we have this pass
  selection list on the uniforms tab. Let's visualize this selection as an explicit line of the
  clickable pass names, it will be more convenient here. So, something like little sub-tabs.
  Something like buttons, but I don't want the actual button boundary here. Maybe just clickable
  text with the highlighting of the active one?"
- **W-C — `true` / `false` draw in the keyword color.** "true/false values need to be colorized
  somehow..." **EDITOR REPO.**
- **W-D — search highlights survive a scrolled view.** "when I search for a symbol ('/pattern') or
  when I quick jump between a symbol occurances: '*', the matched pattern hightlights are in the
  wrong place... something is off.... but sometimes it's not... I don't know what causes the
  offsets... the investigation is required." **EDITOR REPO**, cause located below.

---

## Out of scope

- **The frame-time breakdown.** Feature 088, its own spec; the two touch nothing in common.
- **Word-boundary atoms (`\<`, `\>`) in a typed `/` pattern.** Measured through the binding at
  `410b7e7`: the library's pattern is LITERAL, `*` applies whole-word as a boundary CHECK, and a
  typed `\<foo\>` matches nothing. Nobody asked for it. Trigger: the maintainer types one and
  reports the miss.
- **The uniform rows' texture previews growing with the strip.** They share the strip's size
  token today (083 D4); D1 below splits them and keeps the rows at 112. Trigger: the maintainer
  says the row previews look small beside the new tiles.
- **A keyboard chord for the pass row.** The names are mouse targets; `Ctrl+2` still reaches the
  tab. Trigger: the maintainer asks for next/previous pass from the keyboard on this tab.

---

## Design decisions

Numbered, locked. Open questions are separate, below.

### D1 — the strip gets its own token, `SIZE.PASS_TILE = 168`; `PASS_THUMB` stays 112 for the uniform rows.

`SIZE.PASS_THUMB` (112) has two readers: `widgets/pass_list.py` (the strip's `preview_cell` width
and its wrap step) and `widgets/uniform.py::_draw_texture_preview` (each sampler row's picture, made
"the same symbol" as the strip by 083 D4). Growing the one constant grows both, and a 168-square
picture on every sampler row would spend the Uniforms tab's height on pictures the maintainer did
not ask to enlarge.

So the strip reads a new `SIZE.PASS_TILE = 168` (`1.5 * 112`, the maintainer's number) and the
uniform rows keep `PASS_THUMB`. **This reverses 083 D4's "the same symbol" in one direction**: the
uniform preview still matches what the strip WAS; the strip moves on. 083's spec gets a pointer at
D4 saying so, the way 083 D5 carries 085's.

The picture inside is `168 - 2 * SPACE.MD = 152` square (the child's padding cancels the footer
and chip rows, 083 D4's measured arithmetic). The footer (`font_14_bold`) and the chip row
(`font_12`) do not scale — the tile grows for the PICTURE.

### D2 — the wrap counts the last tile without its trailing gap, and the arithmetic becomes a testable free function.

Today `per_row = max(1, int(avail // (PASS_THUMB + SPACE.MD)))`. That charges every tile a gap,
including the last, so a panel exactly wide enough for three tiles and two gaps wraps the third
tile. The two forms differ only when `avail` falls in the 8-px band just above a column boundary
(520..527 at tile 168), so this is correctness at the boundary rather than a column recovered at
the default split — the fix is small and the function it lands in is what makes it testable.

`tiles_per_row(avail, tile, gap) -> int` = `max(1, (avail + gap) // (tile + gap))`, a free function
in `pass_list.py` (GL-free, imgui-free). A row of `n` tiles spans `n * tile + (n - 1) * gap <= avail`
by construction, which is the "not overflow the right border" guarantee: `preview_cell` is a child
window exactly `cell_w` wide, so a tile that fits the count fits the pixels. The app-panel minimum
(`_APP_PANEL_MIN_W = 360`) floors the tab's content region at about 175 px whatever the window
width or splitter position (measured by the reviewer at five shapes), so one 168 tile always fits
and the sub-one-tile branch is unreachable.

Where the width comes from, so the number is not re-guessed: the app panel is `split_region.x -
editor_width`, the control panel splits `1/2.6` to the document grid, and the settings child takes
the rest inside its own padding. At a 1920-wide window and the default 0.5 split that is about 565
px of tab content by the first estimate and 549 by the reviewer's measured frame: three 168 tiles
need 520 (`3 * 168 + 2 * 8`), four need 696, so three either way. At 1600 wide it is about 470:
two tiles.

### D3 — the pass selector is a `text_tab_row` primitive: clickable names, no frame, the active one bright.

`tabs/uniforms.py::_draw_pass_selector` draws a `pass` caption and a `begin_combo` over
`strip_order`. It becomes one call to a new `ui_primitives.text_tab_row(id_, names, active) ->
str | None`, which draws every name on one line as an `imgui.selectable` sized to its own text
(the `draw_link` / `draw_copyable_text` shape, transparent header colors so nothing frames it),
returns the name clicked this frame or `None`, and wraps to a second line when the next name would
cross the content width (`same_line` only while it fits, `SPACE.LG` between names).

State is carried by color alone: the active name in `COLOR.FG_TITLE`, the others in
`COLOR.FG_DIM`, `COLOR.FG_SECONDARY` on hover — the maintainer's pick at plan-lock (an accent
underline and a left tick were offered as mockups and declined). The maintainer ruled out a frame,
and the imgui skill's "a low-emphasis tier still needs a frame" is about VERBS — this is a
selector, which names a state the surface is in (imgui-ui §1: chips are not in the count).
`tests/test_button_tiers.py` is untouched: a `selectable` is not a raw button call. If the row
reads as a caption once seen live, the underline is the one-line addition to try first.

The `pass` caption goes: a row of pass names under the tab's own heading says what it is, and the
label was the combo's, not the row's. The `< 2 passes -> draw nothing` rule stays (083 D10). The
click writes through `app.set_panel_pass` exactly as the combo did (083 D11: one writer, never the
`or_default` accessor), so `panel_pass`'s tab-follows default and its retire-on-open are unchanged.

### D4 — W-C is a lexer fix upstream; the host expects slot 1 and changes nothing.

Measured by the editor session against its source at `b25d546`: `true` and `false` are absent from
`GLSL_KEYWORDS`, `GLSL_TYPES` and `GLSL_BUILTINS` (`src/lex_glsl.odin`), so the word loop leaves
them `Token_Class.None` and they render plain. Python's `True` / `False` / `None` ARE in
`PYTHON_KEYWORDS` and measure as class 1 already; nothing to do there.

The lexer owner's call: they are KEYWORDS (class 1, the slot `if` and `bool` draw in), not numbers
— the language spec lists them as keywords and every GLSL editor colors them so. The host pushes
NO word class for them: `_feed_classes` carries only the four host classes (conventions, the intel
bullet), and a literal is the lexer's domain. `SYNTAX_1` already maps to `COLOR.SYN_KEYWORD`, so the
re-vendor alone delivers the color.

### D5 — W-D's cause is located upstream: the search emitter subtracts the scroll twice.

The report said "wrong place" and "sometimes not". Every offset hypothesis was probed through the
binding at `410b7e7` and each came back correct: a leading tab, Cyrillic before the match, `*` on
`u_time` / `foo` beside `foobar`, line spacing 1.5, tab size 2 and 8, whitespace shown, font sizes
12 to 20 at column 60 (the cell width is an integer at every size and the glyphs advance by the
same integer, so nothing accumulates), inserts before the match, a line inserted or deleted above,
undo, `ed_set_text`, `replace_selection`. The library recomputes matches from the live buffer on
EVERY `ed_layout` (`view_emit.odin:236`, per the editor session), so there is no cached span to
drift.

What fails is scroll, and the law is measured: **the emitter subtracts the scroll TWICE.** A
rectangle lands on view row `line - 2 * scroll` and is culled when that leaves the viewport, so a
small scroll draws the band visibly high and a larger one drops it entirely. Ninety lines, `foo` on
lines 5 and 60, a 19-row view, one `ed_layout` per case:

| view | `SEARCH_MATCH` rect for line 5 | expected |
|---|---|---|
| `ed_set_scroll(0)` | view row 5 | 5 |
| `ed_set_scroll(1)` | view row **3** | 4 |
| `ed_set_scroll(2)` | view row **1** | 3 |
| `ed_set_scroll(3)` | **none** (row -1, culled) | 2 |
| `ed_set_scroll(49)`, line 60 on view row 11 | **none** | 11 |
| scroll 49 then `n`; scroll first then `/foo<CR>` | **none** | 11 |

Both halves of the report fall out of one rule: a wrong place when `2 * scroll` still leaves the
band inside the view, nothing at all once it does not, and exactly right at scroll 0 — a match
near the top of a short file. In the app scroll is the normal state of every shader taller than
the panel: `/pattern<CR>` moves the cursor, `layout_following_cursor` scrolls to it, and the band
moves a screenful up the same frame. That is the "sometimes".

The editor session found the same law from the source: `buffer_to_screen` already returns the row
scroll-adjusted and the glyph and whitespace emitters take it as is, while `view_emit_search`
alone subtracted `v_scroll_rows` again. **Fixed upstream at `c081110`** (one subtraction dropped)
with its own test, `test_search_bands_survive_vertical_scroll`, which scrolls 58 rows and asserts
the band at view row 2 — mutation-tested there by reintroducing the double subtraction.

### D6 — the host pins its exposure with a test that is born red.

Verifying the vim surface belongs to the editor repo (conventions, the re-vendor bullet), but this
integration's specific exposure is the cursor-follow: the host scrolls right after every search, so
a scrolled view IS the host's search view. `test_search_highlights_survive_a_scrolled_view` drives
the real editor through the binding at two scrolls: `ed_set_scroll(2)` must put line 5's
`Kind.SEARCH_MATCH` primitive on view row 3 (today it sits on row 1 — the OFFSET is pinned as an
offset), and `ed_set_scroll(3)` must put it on row 2 (today there is none — the culling is pinned
too). Both fail against the vendored `410b7e7` and go green with the re-vendor and nothing else,
which is the falsifier the re-vendor needs: a copy that forgot the emitter fix keeps them red, and
a fix that only stopped culling would still fail the placement half.

A second test pins W-C the same way: lay out `bool b = true;`, find the glyph primitives of `true`
by their x-order on the line the way the column-0 marker test does, and assert their color is the
palette's `SYNTAX_1` (today they measure at `Slot.TEXT`).

### D7 — one re-vendor at `760f8ea`, three upstream commits, expected ABI delta zero.

`VERSION` holds `410b7e7`. Upstream `master` is at `760f8ea`, three commits ahead, all landed and
pushed while this spec was drafted:

- `b25d546` — fill the register linewise for `dd` / `yy` on an empty line (unrelated, in the range);
- `c081110` — emit search-match bands at the scrolled row (W-D, D5);
- `760f8ea` — color GLSL `true` and `false` as keywords (W-C, D4).

Verified here rather than believed: `git log 410b7e7..origin/master` lists exactly those three,
and `git diff --stat 410b7e7..origin/master -- ffi/` is empty. The re-vendor follows the seven-file
procedure in `conventions.md ## Known quirks` from that committed sha; the ABI delta is re-derived
from `nm -D` (expected zero), the `Mode` enum and the chord list are re-checked (a re-vendor that
adds a mode must ask which key enters it). No host mitigation becomes dead: neither the lexer gap
nor the emitter had a workaround here. One upstream item is deliberately NOT in the range — the
operator-pending abort (`dp` / `dx` no-op) the editor session flagged as open; it is keymap-wide
and not asked for.

Order of landing: W-A and W-B are host-only and land first, gated as usual; the two host tests of
D6 land WITH the re-vendor commit (the search one cannot be green before it).

---

## Files touched

**Host, W-A:**
- `shaderbox/theme.py` — `SIZE.PASS_TILE = 168`.
- `shaderbox/widgets/pass_list.py` — `tiles_per_row`, the strip reads `PASS_TILE`.
- `tests/test_pass_strip_layout.py` (new) — `tiles_per_row` at the boundaries.
- `ai_docs/features/083_sixth_walk_findings/01_spec.md` — the pointer at D4.

**Host, W-B:**
- `shaderbox/ui_primitives.py` — `text_tab_row`.
- `shaderbox/tabs/uniforms.py` — `_draw_pass_selector` calls it.
- `tests/test_uniforms_tab.py` — the selector's write path through `set_panel_pass` (extend).

**Editor repo (W-C, W-D), then re-vendored here:**
- `shaderbox/resources/editor/` — the seven files + `VERSION`.
- `tests/test_editor_ffi.py` — `test_search_highlights_survive_a_scrolled_view`,
  `test_boolean_literals_draw_in_the_keyword_slot`.
- `ai_docs/conventions.md` — the re-vendor bullet gains this instance in one clause.

---

## Verification

Each step fails for exactly one reason; the falsifier is named.

- **V1 `tiles_per_row`:** `520 -> 3`, `519 -> 2`, `100 -> 1` at tile 168, gap 8. Falsifier: the
  old `avail // step` form returns 2 for 520.
- **V2 no overflow:** for every `avail` in 100..1200, `n = tiles_per_row(avail)` satisfies
  `n * 168 + (n - 1) * 8 <= avail` or `n == 1`. Falsifier: an over-counting form,
  `(avail + tile) // (tile + gap)`, returns an `n` whose row exceeds `avail` at 184 (n = 2, row 344) (the old
  `avail // step` form is merely conservative and would keep this green, so it is not the
  falsifier).
- **V3 the selector writes:** drive `tabs/uniforms.py::_draw_pass_selector` in headless imgui
  frames the way `tests/test_canvas_fields.py` drives the Document tab, feed a synthetic click
  (`add_mouse_pos_event` + `add_mouse_button_event`) at the second name's measured rect, and
  assert `panel_pass` reads back that name — the reviewer confirmed a `same_line`'d selectable
  row takes such a click on the third frame. Falsifier: delete the `set_panel_pass` call from the
  selector and the read-back stays empty. (`test_an_explicit_pass_pick_wins_over_the_active_tab`
  already covers `set_panel_pass` itself; this one covers the WIRE.)
- **V4 the scrolled search (D6):** born red at `410b7e7`, green at the re-vendored sha.
- **V5 the keyword color (D6):** red at `410b7e7`, green after.
- **V6 the re-vendor gates:** `test_the_binding_mirrors_the_upstream_signature_table` and
  `test_the_mode_enum_covers_every_value_upstream_can_return` stay green; `nm -D` diff is empty.
- **Maintainer's eyes (no WM on the dev box):** the tiles at 168 and how the third column wraps;
  whether the bright-only pass row reads as clickable; `true` in a shader.

---

## Plan-lock

Locked by the maintainer on 2026-09-08 from a rendered options page (`trash/plan_lock_087_088.html`,
gitignored): strip only, new `PASS_TILE = 168`, sampler rows stay 112 (D1); the active pass name
marked by color alone, no underline (D3); the `pass` caption dropped (D3).

---

## Review history

**Round 1 (pre-implementation, one reviewer on opus, read-only, with its own probes through the
binding and headless imgui frames).** Verdict PARTIAL; eight findings, all accepted:

- D5 named the wrong mechanism: the band is not suppressed, it lands at `line - 2 * scroll` and
  is culled when that leaves the view (measured at scroll 1 and 2). Rewritten as the law; matches
  the root cause the editor session found in the source and fixed at `c081110`.
- V4 pinned absence, not placement → the host test asserts the row at scroll 2 AND scroll 3.
- V2's falsifier left the test green (the old form is merely conservative) → an over-counting form.
- V3 rested on a false premise (the imgui test engine imports fine) and gated nothing in
  `tabs/uniforms.py` → a headless click test whose falsifier deletes the `set_panel_pass` call.
- D2's motivation overstated the gain → stated as boundary correctness; the measured content
  width at 1920/0.5 is 549, three tiles either way.
- D2's "the tab scrolls" was false and the branch unreachable → the ~175 px floor.
- D6 cited a glyph locator the precedent does not use → x-order on the line.
- D1 named a private function without its underscore → fixed.

Rejected: none. D7 was rewritten in the same round for a reason outside the review: the three
upstream commits landed while it was running. False trails the reviewer recorded so round 2 does
not re-open them: the tier cap and prose budget do not reach `text_tab_row`; `preview_cell`'s
96/152 arithmetic measures exact; no overflow at four host widths including the three-tile
boundary; the strip fits at the minimum panel width; `render_state` repaints on the returned
rects with no host change; `layout_following_cursor`'s layout-scroll-layout is exactly the
exposure D5 describes.

**Round 2 (same reviewer, against the patched text).** Verdict PASS: all eight closed by cited
passages and re-verified — both D6 cases measured red at `410b7e7` (row 1 at scroll 2, none at
scroll 3), V2's falsifier fires at 937 widths, and D7's range checked at the editor repo itself
(three commits in the named order, empty `ffi/` diff, `760f8ea` adding exactly `true` and
`false`). Two residual text defects fixed in the same pass: the D5 heading still stated the
refuted "emits nothing" claim, and V2's witness said 352 where the over-count first fires at 184.
