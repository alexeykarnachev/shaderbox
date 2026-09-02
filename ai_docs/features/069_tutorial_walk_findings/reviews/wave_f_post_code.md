# Wave F post-implementation code review — `d2ade88`

Role: code correctness (dev_flow step 6). Bugs, ctypes/ABI hazards, GL lifecycle,
imgui frame order, the redraw gate.

## Verdict

| Area | Verdict |
|---|---|
| ABI (`_SIG`, 93 entries vs `ffi.odin@c5c6ae2`) | **PASS** |
| Chrome / text origin | **PARTIAL** — finding 1 |
| Redraw gate | **PARTIAL** — finding 2 |
| Markers | **PASS** (finding 3 is unused surface, not a defect) |
| Deletions | **PASS** |
| GL / lifecycle | **PASS** |
| Tests | **PASS** — four falsifiers re-run red, restored, tree clean |
| Conventions | **PARTIAL** — finding 4 |

`make gates` GREEN, judged by exit code captured unpiped: `EXIT=0`, and the smoke
line reads `smoke passed`, not skipped (run under `xvfb-run -a` with the MESA
overrides).

---

## Findings

### 1. The status bar is a live hit-test surface: a click on it moves the caret, a hover on it pops the uniform tooltip

The library lays the text out into a viewport that stops above the status row, but
`ed_pixel_over_glyph` / `ed_word_at_pixel` / `ed_pixel_to_cursor` answer against an
unbounded row extrapolation, and the host's `invisible_button("##editor_surface")`
spans the whole `editor_size` including the bar. So every pixel of the 21px band is
a click and hover target for text hidden behind it.

Measured, 600x300 widget, 21px cell, chrome on, buffer `uAA0..uAA39`:

```
status FRAME rect y = 279.0 .. 300.0     (the opaque band)
last emitted text row  = row 13          (rows 14+ are never drawn)

y=279  over_glyph=True  word='uAA13'  cursor=(13,0)
y=285  over_glyph=True  word='uAA13'  cursor=(13,0)
y=290  over_glyph=True  word='uAA13'  cursor=(13,0)
y=295  over_glyph=True  word='uAA14'  cursor=(14,0)   <- line 14 is not drawn at all
y=299  over_glyph=True  word='uAA14'  cursor=(14,0)
```

Two host consequences, both live:

- `_handle_mouse`: `imgui.is_item_activated()` fires anywhere in the button, and
  `editor.pixel_to_cursor(rel)` is taken unconditionally. Clicking the mode badge
  or the ruler places the caret on a line the user cannot see. A drag started
  there anchors a selection to it.
- the uniform tooltip block at the end of `draw`: gated only on
  `hovering and not app.copilot_hovered`, then `editor.is_mouse_pos_over_glyph(rel)`.
  Hovering the status bar over a uniform name on the hidden row pops the
  `u_name: value` tooltip and lights the panel row, on top of the bar.

The over-reach past the last drawn row predates the commit (reproduced with
`set_draw_chrome(False)`: y=299 still answers `uAA14`). What the commit changes is
that the over-reached region is now covered by an opaque widget the user reads as
chrome, so the mis-hit went from "click below the last line, get the last line",
which reads as a clamp, to "click the mode badge, the caret jumps somewhere
invisible".

Fix: in `tabs/code.py`, subtract the status row from the interactive rect — take
`cell_h` before the invisible button and pass
`imgui.ImVec2(editor_size.x, max(1.0, editor_size.y - cell_h))` to
`invisible_button`, leaving the presented image at full `editor_size` so the bar
still draws. That confines both `_handle_mouse` and the tooltip probe to the text
area in one place, and it reuses the same `cell_h` the `editor_visible_rows`
line already reads.

### 2. `render_state` carries the command line but not its prompt, so a `:`-to-`/` switch repaints nothing

`chrome_emit_status` composes the bar text as `concat_prompt(prompt, input)` — the
prompt rune is part of the painted string. `render_state` records
`editor.get_command_line()` only; `get_command_line_prompt()` is not a member.

Measured, same buffer, chrome on:

```
after ':a'  get_command_line()='a'  prompt=':'
after '/a'  get_command_line()='a'  prompt='/'
should_redraw(a, b) = False

POPUP_GLYPH atlas cells on the bar:
  with ':a'  [(x=12.13, u=0.9885, v=0.6306), (x=19.6, ...)]
  with '/a'  [(x= 9.80, u=0.5080, v=0.2399), (x=19.6, ...)]
  PIXELS DIFFER: True
```

Different glyph, different x, identical gate tuple. This is the same class the
commit closed for `command_message`, one member short: the docstring on
`render_state` says "A member added to the layout's inputs MUST be added here",
and the prompt became a layout input the moment the library started drawing the
bar.

Not user-reachable on today's keymap: getting from `:a` to `/a` passes through a
frame where the command line is `None`, which does fire the gate (measured:
`escape -> cmdline None, redraw True`). It is a hole in the gate's stated domain
rather than a visible stale frame, which is why it is ranked below finding 1.

Fix: add `editor.get_command_line_prompt()` to the `render_state` tuple next to
`get_command_line()`, and extend
`test_render_state_reacts_to_every_editor_dimension` with a prompt case built the
way the `:zzz<CR>` case is.

### 3. Marker tooltips cross the ABI and nothing reads them back

`_apply_markers` passes `tooltip=message` on every error marker, and `ed_add_marker`
copies the string in. There is no `get_marker_tooltip` wrapper on `Editor` and no
call site — `ed_marker_tooltip` has a `_SIG` entry and no Python reader
(`grep -rn "marker_tooltip"` over `shaderbox/` and `tests/` hits only that entry).
Every error message is encoded and pushed across the boundary each time the
fingerprint changes, then dropped.

Not a defect: the error strip below the editor carries the messages, so nothing is
lost, and the argument is cheap. Worth naming because the commit message lists
"the tooltip still passed" as a preserved behaviour, and it is preserved only in
the sense that the bytes still make the trip.

Fix, if it is worth one: either drop the `tooltip=` argument at the call site until
a host reader exists, or add the hover-over-marker readback the ABI comment
describes ("a host that has hit-tested a row can show what it attached there").
Doing nothing is defensible.

### 4. `STATUS_BG` reaches past the role layer into `_P` directly

`theme.py`'s own architecture note (`conventions.md ## Design decisions`, colour
roles) is `_P` -> `_ACCENTS` -> `_ColorBag` role tokens, and `editor_palette()` is
otherwise 22 entries of `COLOR.*` and `fade(COLOR.*)`. The new line is the only
`_P["..."]` in the map.

`COLOR.BG_APP` is defined as `_P["bg_0"]` — the identical value, measured:

```
STATUS_BG now (bg_0): #1d2021
COLOR.BG_APP:         #1d2021   same as bg_0: True
```

Fix: `slot.STATUS_BG: COLOR.BG_APP`. Same pixels, keeps the map inside the role
layer.

---

## Verified clean

**ABI, all 93 entries.** Wrote an independent normaliser
(`scratchpad/cmp_abi.py`) parsing every `@(export)` proc out of
`git -C ~/src/editor show c5c6ae2:ffi/ffi.odin` — expanding Odin's grouped
parameters (`line, col: ^i32` -> two entries), mapping Odin types to ctypes, and
collapsing the platform aliases (`c_int` is `c_int32` here). Result:

```
odin exports: 93   _SIG entries: 93
in odin not in _SIG: []
in _SIG not in odin: []
mismatches: 0
```

`FFI_Primitive` is `{i32 kind; f32 x0,y0,x1,y1,u0,v0,u1,v1,r,g,b,a}`, matching
`Prim` field for field. `ed_add_marker`'s twelve floats are
fill(4) + gutter(4) + text(4) then `i32 gutter_text` then `cstring tooltip`, in that
order — the Python wrapper's `*fill, *gutter, *text, ord(glyph), tooltip` matches.
This was checked against `ffi.odin`, the primary source, not against `abi_probe.py`
that the suite compares to, so a shared error in both tables would have shown here.

**The vendored set.** All seven files are c5c6ae2's bytes:
`abi_probe.py` is byte-identical to `ffi/probe.py@c5c6ae2` (`diff -q` clean),
`vim_coverage.md` and `standard_keymap.md` md5-match `docs/` at that sha, `VERSION`
reads `c5c6ae230a51ece592114f349447b1d79d9563ef`, and `nm -D` on the `.so` yields
exactly the 93 names the source declares (`diff` clean) — so the binary is built
from that committed sha, not a dirty tree. The atlas pair is untouched by the
commit, and the atlas load path in `Editor.__init__` is unchanged.

**The chrome switch.** `set_draw_chrome(True)` sits in `App.get_session`, and
`grep -rn "Editor("` over `shaderbox/` finds exactly one construction — inside that
method. No session can exist before the flag is set, so the "created before the
flag" case has no instance. `_apply_editor_settings_to` runs after it and does not
touch the flag.

**The gutter click.** `(10, 10)` inside the gutter answers `cursor=(0,0)`,
`over_glyph=False`, `word=None` — clamped into the buffer, no tooltip. The first
text cell at `(text_origin.x + cell_w/2, cell_h/2)` answers `(0,0)` with
`over_glyph=True` and the right word. `text_origin` is `(40.0, 0.0)` at four gutter
cells and `(0.0, 0.0)` with `LINE_NUMBERS` off, matching the commit's claim, and the
host now passes `origin=(0,0)` so there is no double offset.

**`editor_visible_rows`.** The `-1` and the `cell_h > 0` guard are both right, and
the guard order is right (`max(0, ...)` cannot go negative at small heights). Swept
against the library's own emitted row set:

```
H=100  host= 3  lib_text_rows= 4
H=150  host= 6  lib_text_rows= 6
H=200  host= 8  lib_text_rows= 9
H=300  host=13  lib_text_rows=14
H=400  host=18  lib_text_rows=18
```

The host under-counts by at most one, which is the safe direction — cursor-follow
scrolls a row early rather than leaving the caret behind the bar. The status row is
emitted whether or not line numbers are on (`FRAME` count 1 in both), so the `-1` is
unconditional and correct.

**The redraw gate, everything else.** Enumerated what `chrome_emit_gutter` and
`chrome_emit_status` read: cursor line and column, mode, scroll, line count, the
chrome flags, the command line, the command message. Cursor, mode, scroll and the
revision are gate members; `LINE_NUMBERS` rides `settings_fingerprint` (all five
`EditorSettings` fields are in it, and `_apply_editor_settings_to` writes no sixth);
`command_message` is the member this commit added. `RELATIVE_NUMBERS`,
`STATUS_LINE`, `STATUS_SHOWS_MODE`, `STATUS_SHOWS_RULER` and `draw_chrome` have no
writer outside session creation, so they cannot change without a new session. The
prompt is the one real gap, finding 2.

**Anchored markers do not need a fingerprint.** The commit's "no stale-marker
fingerprint" decision holds by construction: a marker only moves when an edit moves
it, and every such edit bumps `ed_revision`, which is a gate member. Measured — the
marker moves 9 -> 10 across the `O`+char edit and `should_redraw` is `True` on the
same transition.

**Marker colour semantics.** Counted glyph colours on the marked row of a
GLSL buffer:

```
no marker           5 purple, 3 orange, 2 blue, 1 fg, 1 gray   (syntax)
hover mark (text=0) 5 purple, 3 orange, 2 blue, 1 fg, 1 gray   (identical)
err mark (text set) 10 at the marker's text colour + gutter E in red
```

Alpha 0 leaves the lexer alone, a set alpha overrides it — as documented. The
gutter `E` renders from the atlas (`MISSING_GLYPH` count 0) in `STATE_ERROR` red,
and `get_marker_gutter(1)` reads back `((1.0, 0.0, 0.0, 1.0), 'E')`. The index
semantics match `ffi.odin` (Nth marker on the line, `False` past the end).

**The WCAG numbers in the commit message reproduce.** Recomputed the composited
band and the three contrast ratios independently: band `#43211e` (the message says
`#44221e`, one unit of rounding), `BG_SURFACE` 1.26, `SYN_KEYWORD` 4.10,
`FG_PRIMARY` 10.27, and `COLOR.STATE_ERROR == COLOR.SYN_KEYWORD` is `True`. The
claim is honest. The new `STATUS_BG` separation is a 7/8/8 step on a dark ground
(contrast 1.09 against `BG_SURFACE`) — subtle, but strictly better than the zero it
replaced.

**Draw order and the vertex build.** `build_vertices` marks `GLYPH` and
`POPUP_GLYPH` textured and everything else solid, so `FRAME` renders as a plain
quad and the status text samples the atlas — both correct. The library stable-sorts
by kind, and `Frame` (4) precedes `Popup_Glyph` (6) in the enum, so the bar's
background paints under its own text. Measured on a live layout: `FRAME` at array
index 187, its `POPUP_GLYPH` run at 188..196, `FRAME` rect
`(0, 279)-(600, 300)` in `STATUS_BG`.

**No new per-frame allocation.** `Editor._prims` grows only when the count exceeds
the current buffer (`if len(self._prims) < n`) and is reused otherwise; chrome adds
roughly ten primitives against a 4096 initial capacity. `ensure_loaded` walks all
93 `_SIG` entries once at load, and every name resolves against the new `.so` (it
would raise on the first missing one; the suite runs).

**Deletions.** `_draw_gutter`, `_MODE_BADGES` and `gutter_px` have zero remaining
references across `shaderbox/` and `tests/`. `Mode` still has a reader
(`get_mode() == Mode.NORMAL` in the double-click path) and `SPACE` still has one
(`same_line(spacing=float(SPACE.LG))`), so neither import is dead.
`get_gutter_cells` keeps only test readers now that its host consumer is gone —
that is consistent with `_SIG` being a deliberate complete mirror.

**Tests, four falsifiers re-run.** Each mutation applied to a backed-up file, run,
then restored and confirmed with `git diff --quiet` before anything else ran:

| Mutation | Result |
|---|---|
| `ed_add_marker` back to 8 floats | `test_the_binding_mirrors_the_upstream_signature_table` FAILED; the names test stayed green |
| `ed_set_tab_width` argtype `c_int32` -> `c_float` (length unchanged) | same test FAILED, naming `ed_set_tab_width` — so it compares argtypes, not arity or names |
| `Editor.set_draw_chrome` body -> `pass` | both `test_draw_chrome_adds_a_gutter_and_a_status_frame` and `test_text_origin_moves_right_by_the_gutter_under_chrome` FAILED |
| drop `get_command_message()` from `render_state` | `test_render_state_reacts_to_every_editor_dimension` FAILED |

The second is the one that answers the brief's question directly: a single wrong
argtype at unchanged arity turns the suite red, which is exactly the silent-
corruption case the gate exists for.

**Configs and the bundle.** `abi_probe.py` is excluded from pyright
(`[tool.pyright] exclude`) and from pre-commit (the top-level `exclude:` regex).
`make check` runs `pre-commit run --all-files`, so the gate honours the exclusion.
The file has no importer outside the test's `ast.parse` (which deliberately parses
rather than imports, since the upstream probe opens a session at import time).

---

## False trails

- **`uv run ruff check .` reports 6 errors in `abi_probe.py`.** Not a gate failure —
  `[tool.ruff] extend-exclude` lists only `ai_docs/`, but `make check` goes through
  pre-commit, whose own `exclude:` regex covers the file. The gate is green.
- **`build.sh` ships `abi_probe.py` to users.** It does (75 KB under
  `shaderbox/resources/editor/`), and neither `FORBIDDEN_NAMES` nor
  `FORBIDDEN_PATHS` nor `verify_clean` objects to a `.py` under `resources/`. It is
  inert — nothing imports it — so this is bundle weight, not a defect.
- **`Style` / `set_style` / `get_style` have no host caller.** Dead surface by the
  speculative-machinery test, except the binding is explicitly a complete ABI
  mirror, so an unbound export would be the violation. Correct as written.
- **Marker movement could outrun the fingerprint.** It cannot: a move requires an
  edit, an edit bumps `ed_revision`, and the revision gates. Verified, not assumed.
- **`get_text_origin()` in `render_state` looked redundant** next to
  `settings_fingerprint`'s `show_line_numbers`. It is not: the gutter width tracks
  the line count, so the origin moves when a file crosses a digit boundary with no
  setting change.

---

## Coverage

Read end-to-end: `shaderbox/editor/ffi.py` (all 93 `_SIG` entries plus every
wrapper the commit touched), `shaderbox/editor/render.py`, `shaderbox/tabs/code.py`,
the changed hunks of `shaderbox/app.py` and `shaderbox/theme.py`,
`tests/test_editor_ffi.py`, `.pre-commit-config.yaml`, `pyproject.toml`,
`build.sh`, plus `ffi/ffi.odin`, `src/chrome.odin` and `src/chrome_emit.odin` at
c5c6ae2 for the library side. Did not read `abi_probe.py` line by line — verified it
byte-identical to upstream instead, which is the stronger check for a file whose
whole value is being upstream's bytes.

Probes: one independent ABI comparator and four live probes against the real
`libeditor.so`, all under `/tmp/claude-1000/-home-akarnachev-src-shaderbox/6d39c1c7-0520-4c0a-8808-186ecbf60c39/scratchpad/`.
Four test mutations, each restored and confirmed with `git diff --quiet` before the
next ran.

`git status --short` at close shows only this review plus the six untracked files
belonging to other agents:

```
?? ai_docs/features/069_tutorial_walk_findings/02_keybindings.md
?? ai_docs/features/069_tutorial_walk_findings/50_wave_e_keyboard.md
?? ai_docs/features/069_tutorial_walk_findings/60_wave_g_scripting.md
?? ai_docs/features/069_tutorial_walk_findings/reviews/wave_e_pre.md
?? ai_docs/features/069_tutorial_walk_findings/reviews/wave_g_pre_design.md
?? ai_docs/features/069_tutorial_walk_findings/reviews/wave_g_pre_tests.md
```

(plus `reviews/wave_f_post_spec.md`, which appeared mid-review from the parallel
spec reviewer). No tracked file is modified.
