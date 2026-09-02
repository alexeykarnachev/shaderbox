# W-F pre-implementation review (round 1)

Reviewer role per `dev_flow.md` step 4: correctness & design, verification & blast-radius.
Artifact: `40_wave_f_editor_chrome.md` (740 lines). Anchors: `01_spec.md § W-F / § Locked
decisions D1 D5 D6 / § Out of scope / § Manual verification`, `00_findings.md` rows 11 12 14 15 16,
the editor at `c5c6ae2` (`~/src/editor`), and the ShaderBox tree at `6a85564`.

Verification method: `nm -D` on both binaries, an `@(export)` extraction from `ffi.odin` at
`c5c6ae2`, and a throwaway ctypes probe against the parked
`scratchpad/editor_c5c6ae2/libeditor.so` (six probe scripts; `shaderbox/editor/ffi.py` untouched).

## Verdicts

- **Parent coverage: PASS.** Every surviving W-F bullet is either satisfied by the library at
  `c5c6ae2` or by a numbered decision. The supersession argument holds: the parent was written
  against `68def59`, and the bullets it voids are the ones the library now does.
- **ABI table accuracy: FAIL.** Six of the twenty-eight listed signatures disagree with
  `ffi.odin` at `c5c6ae2` — five of them by a MISSING trailing argument, which is the silent
  memory-error class the task statement singled out. Findings 2 and 3.
- **Host-origin coverage: PASS.** All six sites are enumerated and each verdict is correct;
  the completion-popup refutation is right for the reason the spec gives. One consequence the
  spec does not name is a real behaviour change, not a miss (finding 8).
- **Test falsifiability: PARTIAL.** Four of five tests are falsifiable as written and the
  completeness test is verified red today. Test 3's GLYPH assertion is buffer-dependent and goes
  red on a wide buffer at the stated widget size (finding 4).
- **Docs: PARTIAL.** Every doc the wave must touch is named, but the vendored-file count is wrong
  in the correction itself (finding 6), and `dev_flow.md`'s module map is not in the list at all
  (finding 9).

Two findings are correctness defects that ship a visible wrong result (1, 2). One is the class the
task named as silent (3). The rest are bookkeeping and precision.

---

## Findings

### 1. The marker text colour `BG_SURFACE` is illegible on the fill it composites over. Contrast 1.27:1, WORSE than the red-on-red it replaces.

The whole point of decision 7 is #14's "red keyword on a red line is unreadable". The chosen
foreground makes it less readable, not more.

The spec reasons that "`BG_SURFACE`-on-red is the same contrast relationship as vim's white-on-red".
That would be true if the marked line were a full-opacity red band. It is not. The library's own
README at `c5c6ae2` (§ Line markers) states: "**The fill is translucent by necessity** — it draws
behind the text as a `Background` primitive, and an opaque one hides the code it marks." The host
passes `fade(COLOR.STATE_ERROR, 0.20)`, and the ground behind it is `slot.BACKGROUND` =
`COLOR.BG_SURFACE` = `#161819` (`theme.py:531, 126`). The composited band is therefore

    0.20 * #fb4934 over #161819  ->  #44221e

a dark brown, not a red band. Measured WCAG contrast on that band:

| foreground | ratio on `#44221e` |
|---|---|
| `COLOR.BG_SURFACE` `#161819` (the spec's choice) | **1.27** |
| `COLOR.SYN_KEYWORD` `#fb4934` (today's unreadable case) | 4.09 |
| `COLOR.FG_PRIMARY` `#ebdbb2` (`_P["fg_1"]`) | 10.25 |
| `_P["fg_0"]` `#fbf1c7` | 12.39 |

So the spec's choice is three times worse than the defect it is fixing. At the old 0.35 alpha the
band is `#662922` and `BG_SURFACE` still only reaches 1.62. The palette claim behind the decision is
confirmed and sharp (`theme.py:167` `STATE_ERROR = _P["red_b"]`, `:179` `SYN_KEYWORD = _P["red_b"]`,
byte-identical), but the conclusion drawn from it inverts the fix.

Vim's `hi Error` is white on red because the band is opaque; the transferable half of that pairing
is "a LIGHT foreground against a dark red ground", and this palette's light foreground is
`FG_PRIMARY`, not the ground colour.

**Fix (paste):** In decision 7, replace `text=COLOR.BG_SURFACE` with `text=COLOR.FG_PRIMARY` and
replace the justifying paragraph's claim with the measurement: the marker fill is translucent by
ABI (README § Line markers, "the fill is translucent by necessity"), so the composited band at 0.20
alpha over `COLOR.BG_SURFACE` is `#44221e`; `COLOR.BG_SURFACE` on that band measures 1.27:1 while
`COLOR.FG_PRIMARY` measures 10.25:1, so the replacement foreground is the light one, which is the
transferable half of vim's white-on-red.

### 2. Five of the twenty-eight ctypes signatures drop a trailing argument that the Odin proc requires. Each is the silent-corruption class, not a `TypeError`.

Checked every entry in decision 3 against `git -C ~/src/editor show c5c6ae2:ffi/ffi.odin`. The
argtypes and restypes below are wrong as written:

| entry | spec says | `ffi.odin` at `c5c6ae2` | consequence |
|---|---|---|---|
| `ed_replace_at_cursor` | `(c_bool, [void_p, char_p])` | `:1279` `(h, pattern, replacement: cstring, ignore_case: bool) -> bool` | two args short; callee reads `replacement` and `ignore_case` off the stack |
| `ed_replace_all` | `(c_int32, [void_p, char_p, char_p])` | `:1300` `(h, pattern, replacement: cstring, ignore_case: bool) -> i32` | one arg short |
| `ed_find` | `(c_int32, [void_p, char_p, c_bool])` | `:1238` `(h, pattern: cstring, backward: bool, ignore_case: bool) -> **bool**` | wrong restype AND one arg short |
| `ed_find_count` | `(c_int32, [void_p])` | `:1263` `(h, pattern: cstring, ignore_case: bool) -> i32` | two args short; it is not a "how many hits are live" getter, it counts a pattern |
| `ed_paste` | `(None, [void_p, char_p, c_bool])` | `:1218` `(h, before: bool, count: i32) -> **bool**` | wrong restype and BOTH arg types; a `char_p` pushed where a `bool` is read |

`ed_find_next` (`:1254`), `ed_insert_at` (`:1505`), `ed_class_at` (`:819`), `ed_marker_gutter`
(`:637`), `ed_color` (`:564`), `ed_chrome_flag` (`:864`), `ed_view_flag` (`:1075`),
`ed_set_number_width` (`:878`), `ed_set_filler_glyph` (`:885`), `ed_filler_glyph` (`:890`),
`ed_marker_count` (`:626`), `ed_reset_theme` (`:577`), `ed_tab_width` (`:1744`), `ed_language`
(`:802`), `ed_line_spacing` (`:1104`), `ed_host_completion` (`:175`), `ed_primitive` (`:470`),
`ed_set_line_selection` (`:1680`), `ed_set_chrome_style` (`:924`), `ed_set_style` (`:940`),
`ed_style` (`:952`), `ed_set_draw_chrome` (`:912`) and `ed_draw_chrome` (`:917`) are all correct
as listed.

The spec does say "the implementer takes each signature from `ffi.odin` rather than from this list
where the two could disagree". That instruction does not survive contact: an implementer reading a
28-row table written by someone who checked the ABI will copy it, and the completeness test
(decision 2b) checks NAMES only and passes on every one of these. This is the exact hazard the spec
itself flags for `ed_add_marker` and then does not apply to the other twenty-seven rows.

**Fix (paste):** Correct the five rows to `"ed_replace_at_cursor": (ctypes.c_bool, [ctypes.c_void_p,
ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool])`, `"ed_replace_all": (ctypes.c_int32,
[ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool])`, `"ed_find": (ctypes.c_bool,
[ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool])`, `"ed_find_count":
(ctypes.c_int32, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool])` and `"ed_paste":
(ctypes.c_bool, [ctypes.c_void_p, ctypes.c_bool, ctypes.c_int32])`, and add a sentence saying the
implementer diffs every one of the 28 against `ffi.odin` at `c5c6ae2` before running the suite,
because the completeness test matches names and never argtypes.

### 3. The "live bug" at `code.py:175` is REFUTED. The tooltip has always crossed correctly.

Decision 7 and premise 33 both claim: "the third positional argument is `gutter`, so `message` has
been landing in the GUTTER COLOUR slot and the tooltip has always been empty".

Opened `shaderbox/editor/ffi.py:465` (AST-extracted params, not read by eye):

    add_marker(self, line, fill, gutter, tooltip)

and `shaderbox/tabs/code.py:175`:

    editor.add_marker(line, err_fill, err_fill, message)

`message` is the FOURTH positional after `self` is bound, which is `tooltip`. The call is correct
today. The spec miscounted by one and built a "live bug the changed signature forces into the open"
on it.

The real defect in the same neighbourhood is different and worth keeping, because it explains the
same symptom the spec was reaching for: `ed_marker_tooltip` is bound (`ffi.py:225`) and **never
called anywhere in the repo** (`grep -rn "marker_tooltip" shaderbox/ tests/` returns only the `_SIG`
entry), and `_draw_gutter` (`code.py:342-373`) draws numbers only and never a gutter mark. So the
tooltip crosses into the library and nothing on the host side reads it back out, and the gutter
glyph was hard-coded to `0` at the binding (`ffi.py:473`). The marker mark and its tooltip have both
been invisible, for those two reasons rather than for an argument-order one.

**Fix (paste):** In decision 7 and premise 33, replace the miscount with: `code.py:175` passes
`message` as the fourth positional, which is `tooltip`, and that has always been correct. What has
never been visible is the RESULT: `ed_marker_tooltip` is bound at `ffi.py:225` and called from
nowhere in the repo, and the binding hard-codes the gutter glyph to `0` at `ffi.py:473`, so no mark
was ever drawn to hover. `c5c6ae2` draws the glyph itself once chrome is on, which closes the mark
half; the tooltip stays unread by the host and the error strip remains where an error's text is
read.

### 4. Test 3's "GLYPH count strictly greater under chrome" is buffer-dependent and goes RED on a wide buffer.

The gutter takes four cells off the LEFT, so the text viewport narrows by four columns and glyphs
past the new right edge stop being emitted. Whether the gutter's own numbers plus the `~` filler
outnumber the clipped glyphs depends on the buffer. Probed on the parked `.so` at the spec's own
600x300 / 16 px-per-em geometry:

| buffer | chrome OFF | chrome ON |
|---|---|---|
| 20 lines of `abc` | `{GLYPH: 45, CARET: 1}` | `{GLYPH: 58, CARET: 1, FRAME: 1, POPUP_GLYPH: 9}` |
| 20 lines of 60 chars | `{GLYPH: 840, CARET: 1}` | `{GLYPH: 757, CARET: 1, FRAME: 1, POPUP_GLYPH: 9}` |

The wide case DROPS 83 glyphs. A test asserting `glyphs_on > glyphs_off` passes on the short buffer
the spec probed and fails on a realistic shader line width. The `FRAME == 1` and `POPUP_GLYPH > 0`
halves are sound and are what actually pin the switch.

Separately, the spec's recorded buckets `{0:1, 2:99, 3:1}` carry a `Background` (kind 0) that only
appears when a marker exists — a fresh handle with no marker emits none (probed: `{2: 199, 3: 1}`).
Not a defect in the test, but the recorded evidence describes a different probe than the test
describes.

**Fix (paste):** Drop the GLYPH-count comparison from test 3 and assert instead that chrome-on emits
exactly one `Kind.FRAME`, a non-zero `Kind.POPUP_GLYPH` count, and that `get_text_origin()[0]` moved
from 0.0 to `get_gutter_cells() * cell_w`; a GLYPH-count comparison is not a valid falsifier because
the gutter narrows the text viewport and a wide buffer emits FEWER glyphs under chrome (measured:
840 to 757 on twenty 60-character lines at 600x300, 16 px/em).

### 5. `render_state`'s missing `command_message` is CONFIRMED, and the reasoning through `should_redraw` holds.

`render.py:82-116` lists seventeen members; `editor.get_command_line()` is one and
`editor.get_command_message()` is not. `should_redraw` is `prev != cur` (`render.py:119-120`), so a
change invisible to all seventeen never repaints.

Probed on the parked build with chrome on: after `:zzz` + Enter, `ed_command_line` returns `None`
(unchanged from before the command), the mode is back to `NORMAL` (0), the revision is unchanged,
and the `Popup_Glyph` bucket goes 9 -> 18 as "not an editor command" lands in the status row. Every
existing member holds still while the drawn primitives change. Confirmed.

Two precisions for the spec. First, `Editor.get_command_message` already exists (`ffi.py:632`) and
`ed_command_message` is already bound (`ffi.py:291`) — the wave adds a call, not a binding.
Second, this is a gap the wave CREATES rather than one it discovers: today the message is drawn in
`draw_chrome` (`code.py:400-405`) as imgui text, repainted every frame, so it is visible today and
becomes invisible the moment decision 6 deletes that branch and the library draws it into the gated
texture instead. Stating it that way is what makes it non-optional.

**Fix (paste):** In decision 10, note that `Editor.get_command_message` and its `_SIG` entry already
exist (`ffi.py:632`, `:291`), so the change is one member in the `render_state` tuple; and that the
gap is one this wave CREATES, because `draw_chrome` draws the message as ungated imgui text today
(`code.py:400-405`) and decision 6 deletes that branch, moving it into the redraw-gated texture.

### 6. Six files are vendored, not five. The correction the spec asks for undercounts by one, the same way the entry it is correcting does.

`git ls-files shaderbox/resources/editor/` returns six: `VERSION`, `atlas.json`, `atlas.png`,
`libeditor.so`, `standard_keymap.md`, `vim_coverage.md`. Decision 1 says "Correct the list to name
all five files".

The rest of decision 1 is verified exactly right. `diff` is clean and `md5sum` identical for
`atlas.png` (`5d476903890dc4f478539ce99aa29603` both sides), and `atlas.json`, `vim_coverage.md` and
`standard_keymap.md` are all byte-identical between the tree and the parked directory — so two files
change, as claimed. Premise 18's refutation is correct: both keymap docs are already tracked and
already at `c5c6ae2`, so W-E's inputs do not wait on this wave's copy.

`build.sh` needs no change, verified: `:48-51` strips `libeditor.so` by name for the Windows stage
and `:72`'s `verify_clean` finds the same name; neither enumerates the directory. Premise 32 stands.

**Fix (paste):** Say six files, not five, and name them: `libeditor.so`, `atlas.png`, `atlas.json`,
`VERSION`, `vim_coverage.md`, `standard_keymap.md`. The `conventions.md` entry says "the three
files" in its rebuild step as well as under-naming the shipped set, so both sentences change.

### 7. The `E` gutter glyph disappears whenever `show_line_numbers` is off, which the spec's own design leaves as the only error signal in the gutter.

Decision 4 keeps `settings.show_line_numbers` working through `ChromeFlag.LINE_NUMBERS`, and that is
right (`app.py:1386`, verified). But the library draws the marker's gutter glyph only inside the
gutter it reserved. Probed with chrome on and a marker on line 3: `LINE_NUMBERS` on gives
`{BACKGROUND:1, GLYPH:91, ...}`, `LINE_NUMBERS` off gives `{BACKGROUND:1, GLYPH:74, ...}` — the
same 74 as with no marker at all, and `ed_gutter_cells` drops to 0 with `ed_text_origin` back to
`(0,0)`. The README says so directly: "**`ed_layout` draws the gutter mark only when it draws the
gutter**".

So with line numbers off, an error line has the 0.20 wash and the replaced text colour and nothing
else. That is a defensible picture, but it is a behaviour the spec asserts nowhere and the manual
pass ("An `E` shows in the gutter's separator cell on that row") would fail for a maintainer who
has the setting off.

**Fix (paste):** In decision 7, state that the `E` is drawn only while `ChromeFlag.LINE_NUMBERS` is
on, because `ed_layout` draws a marker's gutter mark only in a gutter it reserved (README § Line
markers); with `show_line_numbers` off the error line keeps its fill and its replaced text colour
and shows no gutter glyph, which is the intended degradation. Add "with line numbers on" to the
manual-verification bullet that looks for the `E`.

### 8. `ed_set_style` clobbers every chrome flag, including the host's `show_line_numbers`. W-F defines the method W-E will call, so the hazard belongs here.

`ed_set_style` (`ffi.odin:940`) calls `editor_set_keymap`, which replaces the whole `Chrome` from
`chrome_for(style)`. Probed: with `LINE_NUMBERS` set to False by the host, `ed_set_style(h, 0)`
leaves it True. `draw_chrome` itself survives a style switch (probed: still True after
`ed_set_style(h, 1)`), so the wave's own switch is safe.

W-E is the wave that calls `set_style`, and it will call it out of `_apply_editor_settings_to` or
beside it. If it calls `set_style` AFTER `set_chrome_flag(LINE_NUMBERS, ...)`, the user's setting is
silently discarded on every settings apply. W-F is where `set_style` gets written, so W-F is where
the ordering constraint is cheapest to record.

**Fix (paste):** In the out-of-scope note on the keymap setting, add: `ed_set_style` replaces the
whole `Chrome` from `chrome_for(style)` (`ffi.odin:940`), so it resets `LINE_NUMBERS`,
`RELATIVE_NUMBERS` and the status flags to the style's defaults — measured. `ed_draw_chrome` is NOT
part of `Chrome` and survives the switch. W-E must therefore call `set_style` BEFORE
`set_chrome_flag(LINE_NUMBERS, settings.show_line_numbers)` in `_apply_editor_settings_to`, or the
user's line-numbers setting is discarded on every apply.

### 9. `dev_flow.md`'s module map is not in the docs list, and both its entries go stale with this wave.

The spec's § Files touched names `conventions.md`, `067_custom_editor.md` and `roadmap.md`. The
task's docs checklist also asks for the module map, and it earns its place: `dev_flow.md:269-275`
describes `editor/` including "`render.py` (the moderngl MTSDF primitive pass + the redraw gate's
`render_state`/`should_redraw` free functions)" — still true — but the `editor/` entry and
`tabs/code.py`'s one-liner at `:325` ("inline GLSL editor — main-window LEFT split") say nothing
that this wave falsifies. Checked both: neither mentions the host gutter, the mode badge, or the
bottom bar.

So the correct answer is that the module map needs NO edit, and the spec should say so rather than
be silent, because a reader following the checklist will otherwise go looking.

`conventions.md`'s inline-editor entry (`:362-380`) is likewise unaffected by W-F: its only
editor-chrome sentence is the vendoring pointer, and its "vim-modal library" phrasing is already
assigned to W-E by the parent spec (`01_spec.md § W-E`). No collision.

**Fix (paste):** Add to § Files touched, under "Not touched, and each for a reason a reader might
otherwise doubt": `ai_docs/dev_flow.md` — the module map's `editor/` entry (`:269-275`) and
`tabs/code.py` line (`:325`) describe the modules' roles, not their chrome, so nothing in them goes
stale; and `conventions.md`'s inline-editor entry, whose only chrome-adjacent sentence is the
vendoring pointer and whose keymap phrasing is W-E's by the parent spec.

### 10. `ensure_loaded` fails at LOAD, not "inside the first frame that opens a file".

Decision 2b justifies the bound-but-absent direction with: a dropped export "currently fails at
`ensure_loaded`'s `getattr` with an `AttributeError` inside the first frame that opens a file, which
is a crash in the UI rather than a red gate".

`ensure_loaded` (`ffi.py:307-317`) runs the `getattr` loop on first call, and every test in
`tests/test_editor_ffi.py` reaches it through `Editor(...)` -> `ensure_loaded()`. So a dropped export
fails 48 tests loudly, at collection-adjacent speed. It is already a red gate. The bound-but-absent
direction is still worth having (it names WHICH export, in one assertion, instead of an
`AttributeError` from a loop), but the stated motive is wrong and a reviewer of the next re-vendor
would inherit a false belief about the failure mode.

**Fix (paste):** Replace the bound-but-absent justification with: `ensure_loaded` (`ffi.py:307`)
`getattr`s every `_SIG` name on the first `Editor` construction, so a dropped export already fails
the suite loudly; what the assertion adds is the NAME of the missing export in one message instead
of an `AttributeError` out of a binding loop.

---

## The coverage the task asked for, item by item

**The two live bugs.** `code.py:175`'s tooltip: **REFUTED**, finding 3. `render_state`'s
`command_message`: **CONFIRMED**, finding 5.

**The mirror rule.** 26 unbound at the vendored `65264dc` and 28 at `c5c6ae2` — both confirmed by
`comm` between `nm -D` output and the `_SIG` keys extracted by AST. 91 exports at `65264dc`, 93 at
`c5c6ae2`, none removed, the two new ones exactly `ed_set_draw_chrome` and `ed_draw_chrome`
(premise 2 confirmed). The 93 `nm` symbols are byte-identical to the 93 `@(export)` procs in
`ffi.odin` at that sha (`diff` clean; premise 15 confirmed). `_SIG` binds 65, and `set(_SIG) -
exported` is empty today, so the currently-vendored binary has no bound-but-absent entry.

The completeness test as specified **does** go red for any unbound export including a future one,
and it **does** read the list from the binary: I ran its exact body against the tree's `.so` and it
reports the 26. `nm` is present (`/usr/bin/nm`), and `subprocess` in a test is precedented
(`tests/test_import_diet.py:25`). The list cannot rot, because there is no list — the test
enumerates `nm -D --defined-only` and filters `startswith("ed_")` on symbol type `T`, and all 91 (and
all 93) `ed_*` symbols are type `T`, so the filter drops nothing. Its blind spot is argtypes, which
is finding 2.

**The marker colour decision.** The palette claim is confirmed and is the sharpest thing in the
spec: `theme.py:167` and `:179` are both `_P["red_b"]`. The conclusion drawn from it is wrong —
finding 1, with the measured composite and the four-way contrast table.

**`ed_text_origin` consequences.** Six sites enumerated, all six correct.

| site | spec verdict | checked |
|---|---|---|
| `_handle_mouse` press/drag (`code.py:451, 453, 465`) | unchanged, hit tests answer offset | correct — `ed_pixel_to_cursor` reads `prev_layout` (`ffi.odin:767`), built at the offset origin (`:379-392`) |
| uniform-hover tooltip (`code.py:671-673`) | unchanged | correct, same `prev_layout` for `ed_pixel_over_glyph` (`:731`) and `ed_word_at_pixel` (`:743`) |
| image blit (`code.py:637-644`) | unchanged, furniture inside the texture | correct, the blit already spans the whole widget |
| `app.editor_visible_rows` (`code.py:566`) | over-counts by the status row | correct, see below |
| `_draw_gutter` (`code.py:342-373`) | deleted; only reader of `get_text_origin` | correct, `grep` confirms `get_text_origin` has exactly one caller |
| completion popup | no host change | correct, `ed_layout` calls `layout_emit_popup` from `lr` (`ffi.odin:424-431`) |

Caret follow and scroll-into-view are the same `app.editor_visible_rows` value (`code.py:592-599`)
and are covered by that row. The `-1` fix is right and provable rather than assumed:
`chrome_text_area` (`src/chrome.odin`) computes `status := c.status_line ? cell.y : 0`, exactly one
cell, so the subtraction is a fact about the emitter and not an estimate. The `if cell_h > 0 else 0`
guard already in `code.py:566` must survive the rewrite.

One consequence the spec does not name, which is a behaviour change rather than a miss: with chrome
on, `is_mouse_pos_over_glyph` answers false over the gutter and the status row, so the uniform-hover
tooltip stops firing in the leftmost four cells. The spec calls that "what a reader wants" and I
agree; noting it here so the manual pass is not surprised by a hover that used to work over the old
host gutter (it did not — the old gutter was drawn OVER the image, outside the editor's hit surface,
so this is strictly an improvement).

**The deletions.** `_draw_gutter` at `342-373` called at `645-648` under `if
settings.show_line_numbers:` with a `push_font(app.font_12)` wrapper: exact. `draw_chrome`'s vim
half at `386-406`: exact — `:386` is `session = app.editor_sessions.get(tab.path)` and `:406` the
`same_line` closing the message branch, with `:407` opening `if tab.kind == "shader":`.
`_MODE_BADGES` at `24-29`: exact. The `Mode` import survives because of `code.py:455`
(`editor.get_mode() == Mode.NORMAL` in the double-click branch): confirmed, it is the only other
`Mode.` use in the file.

Nothing else reads what they produced. `grep -rn "_draw_gutter\|_MODE_BADGES"` finds no other caller
and no test. `show_line_numbers` IS honoured by the library through `ChromeFlag.LINE_NUMBERS`, and
the wiring already exists at `app.py:1386` inside `_apply_editor_settings_to` — the spec is right
that this path is untouched. Probed: `set_chrome_flag(LINE_NUMBERS, False)` under chrome takes
`ed_gutter_cells` to 0 and `ed_text_origin` back to `(0,0)`, so the text starts at the widget edge
exactly as claimed. `RELATIVE_NUMBERS` needs no host call: `chrome_for(.Vim)` sets it true and a
fresh handle probes true, which is why the parent's "set `ChromeFlag.RELATIVE_NUMBERS` under vim"
bullet is satisfied by the library rather than by code.

**The error strip decision (9).** Agree, and the maintainer's #14 wording supports it: the finding
is entirely about the LINE ("the line gets highlighted red but the text keeps its syntax colours"),
and names no complaint about the strip. #11's complaint is scoped by its own words to "the same line
as the file path, compile status, 'Open dir' button" — the strip is a separate child below the
editor (`code.py:553` reserves `strip_height` off the editor height, `_draw_error_strip` opens its
own `begin_child`), so it was never part of the mess #11 named. Folding it would cost the count, the
per-row jump (`code.py:215-218`) and the expand toggle (`:219-227`), none of which the walk asked
for. The stacking claim is right: strip below, status row inside the image, text above.

**`app.editor_visible_rows`.** Confirmed over-counted and confirmed load-bearing: `code.py:566` sets
it, `code.py:592-599` gates the cursor-follow on it, and `app.py:274` initialises it to 0. The fix
`max(0, int(size_px[1] / cell_h) - 1)` is correct against `chrome_text_area`'s one-cell status row.
`tests/test_editor_ffi.py:153` uses it only as a `SimpleNamespace` field, so no test breaks.

**Tests.** Test 1 verified red today (26 unbound, ran its body). Test 2's falsifier is real: probed
on the parked build, a marker on line 9 then `set_cursor(6,0)` + `O` + a character reads back at
line 10 through `ed_marker_gutter`, and the README states the pre-`c5c6ae2` behaviour it replaces —
so it pins the re-vendor, as claimed. Test 3: finding 4. Test 4 is falsifiable and its identity
holds (probed 40.0 at 4 gutter cells of 10.0 px, chrome-off 0.0). Test 5's falsifier is real
(finding 5's probe). All five live in `tests/test_editor_ffi.py`, which loads the real vendored `.so`
through `Editor.__init__` -> `ensure_loaded()` -> `EDITOR_RESOURCES_DIR / "libeditor.so"`
(`ffi.py:310`), plus `ed_load_atlas` on the vendored `atlas.json` (`ffi.py:337`) — so after the wave
they run against `c5c6ae2` with no test-side path change. There is no mock of this ABI in the repo,
confirmed by grep. The new completeness test is the only one needing an explicit path, and it takes
the same `EDITOR_RESOURCES_DIR` constant.

**The four closed open questions.**

1. **Status band grey.** DISAGREE with the default, per the ruling. `STATUS_BG` and `BACKGROUND` are
   both `COLOR.BG_SURFACE` (`theme.py:531, 541`) and the panel clear is the same
   (`code.py:631`), so the band is provably invisible before anyone looks — that is a known
   defect, not something to discover at the manual pass. Pick `_P["bg_0"]` (`#1d2021`), one step up
   from `bg_0h` (`#161819`), and let the manual pass correct it. The maintainer asked for nvim's
   look and nvim's statusline is a distinct band.
2. **`set_draw_chrome` as a Setting.** Agree with the default. D6 makes the furniture the editor's,
   and `show_line_numbers` already covers the half anyone turns off.
3. **`E` as a warning tier.** Agree with the default, with finding 7's caveat about line numbers off.
4. **Strip jump follows the compile line.** Agree with the default. The reasoning in decision 8 is
   sound and the cost argument (`ed_marker_gutter` walk per error per click) is real given the
   walk-until-refused shape the README describes.

**Findings 11-16 against what the maintainer asked to SEE.**

| # | asked to see | delivered by | verdict |
|---|---|---|---|
| 11 | vim symbols off the bottom bar, first-class editor elements | decision 6 deletes `code.py:386-406`; `chrome_emit_status` draws the row inside the rect (probed: one `Frame` + 9 `Popup_Glyph`) | YES |
| 12 | nvim's `number relativenumber`, `~` past the end | `chrome_for(.Vim)` sets `relative_numbers = true` and `filler_glyph = '~'`; `chrome_emit_gutter` is the same emitter `behavior_test.odin` pins | YES, and by re-vendoring rather than by filing, which the finding's own "if it doesn't" clause permits |
| 14 | a readable error line, vim's colour flip | the ABI delivers the override; the spec's chosen colour does not — finding 1 | NOT AS SPECIFIED |
| 15 | the red moves with the code | probed: marker 9 -> 10 after `O` above; decision 8 correctly refuses to build the parent's blank-while-dirty mechanism | YES |
| 16 | visual `p` replaces | closed by the binary swap; the vendored `vim_coverage.md:291-292` at `c5c6ae2` marks "`p` `P` over a visual selection — the selection is replaced, as one undo step" `[x]`, and `:313` records `p` taking the replaced text into the register | YES, no host code |

**Parent-bullet disposition.** Re-vendor: satisfied by the wave (decision 1), with the target sha
moved from `68def59` to `c5c6ae2` for cause. Issues to file: void by maintainer instruction, and
the spec is right that all four are fixed at `c5c6ae2`. Status line inside the rect: satisfied by
the library. Gutter: satisfied by the library, including the `behavior_test.odin:338` picture and
`ed_filler_glyph`. Error lines: partly by the library (the text override), partly by the wave
(decision 7), and the parent's "no whole-line fill + 2px left bar" is correctly superseded — it was
written as an explicit "host-side fallback until then" in #14. Stale markers: correctly dropped,
premise retired at `c5c6ae2`. `ed_set_style` / `ed_style` / `ed_filler_glyph`: bound, and the wave
correctly widens that to the whole mirror. The tests-for-a-pure-gutter-label-function bullet is void
with `_draw_gutter`, which the spec does not say in so many words but which follows from decision 6.

---

## False trails

- The spec's chrome-on / chrome-off buckets carry a `Background` (kind 0) that only exists when a
  marker is present; the shapes and the `Frame`/`Popup_Glyph` counts are otherwise reproducible.
- `ed_pixel_to_cursor` is bound with `restype=None` while the README's C block shows no return —
  checked `ffi.odin:767`, the proc returns nothing, so the existing binding is right.
- Decision 11's "the theme slot mapping is already complete" is exactly true: all six of slots 5-10
  are in `editor_palette` (`theme.py:538-543`) and applied at `app.py:1353`. Only the `STATUS_BG`
  value is wrong, which is open question 1.
- `theme.py` is the one file W-F shares with the in-flight W-B; reviewed against `6a85564` per the
  brief, and W-F touches it only if open question 1 is taken, which is a single `editor_palette`
  value in a function W-B does not enter.
- The wave's claim that W-C and W-A put no citation at risk holds: neither touches `tabs/code.py`,
  `editor/`, or `editor_palette`, and `git diff` against `6a85564` shows `code.py` unmodified.
- `ed_set_chrome_style` and `ed_set_style` both exist at the currently vendored `65264dc`, so
  premise 1's correction is right — they are unbound, not new.

## Coverage statement

Opened and read: `40_wave_f_editor_chrome.md` whole; `01_spec.md` §§ Out of scope, Locked decisions,
W-E, W-F, Order, Manual verification, Open questions, Review history; `00_findings.md` rows 11-16
verbatim; `conventions.md` the inline-editor entry (`:362-380`), the vendored-binary entry (`:854`)
and the re-vendoring entry (`:883`); `067_custom_editor.md` § Out of scope and decision 13;
`.claude/skills/imgui-ui/SKILL.md` §§ 2 and 5; `dev_flow.md:265-285` and `:325`;
`shaderbox/editor/ffi.py` whole (`_SIG` AST-extracted, every `Editor` method skimmed, `add_marker`
/ `layout` / `ensure_loaded` read in full); `shaderbox/editor/render.py:70-152`;
`shaderbox/tabs/code.py` whole; `shaderbox/app.py:1345-1400` and `:274`; `shaderbox/theme.py` at
`6a85564` (`_P`, the `COLOR` tokens cited, `editor_palette`); `shaderbox/ui.py:418, 561-570`;
`build.sh:44-76`; `tests/test_editor_ffi.py` (helpers, the redraw-domain walk, the load path).

Editor repo at `c5c6ae2`: `ffi/README.md` §§ Chrome, Line markers, Drawing; `ffi/ffi.odin` in full
for every signature in the spec's table plus `ed_layout`, `ed_add_marker`, `ed_text_origin`,
`ed_gutter_cells`, `ed_pixel_to_cursor`; `src/chrome.odin` (`chrome_for`, `chrome_text_area`,
`chrome_flag`); the vendored `vim_coverage.md:288-318`.

Measured, not read: `nm -D` on both binaries plus a `comm` diff of the export sets; an AST
extraction of `_SIG`'s 65 keys and of `add_marker`'s parameter list; `md5sum` and `diff` across all
four unchanged vendored files; six ctypes probes against the parked `.so` covering the chrome
switch, the text origin, the gutter-cells / line-numbers interaction, the twelve-float marker with
read-back through `ed_marker_gutter` and `ed_marker_tooltip`, the marker's move under `O`, the
`:zzz` command-message repaint gap, the standard-style furniture, and `ed_set_style`'s effect on
`Chrome` and on `draw_chrome`; a WCAG contrast computation over the composited marker band; and a
run of the spec's own completeness-test body against the tree's binary.

Not checked: the standard keymap's chord list and the reserved-set audit (W-E's, and the spec
correctly scopes them out); whether the status row's glyphs align with the text grid by eye, which
is a manual-pass question; the Windows `.dll` (out of scope by both specs).

---

# Round 2 (closure)

Narrow closure round against `40_wave_f_editor_chrome.md` at 991 lines, plus the `50_wave_e_keyboard.md`
and `02_keybindings.md` insertions. Scope: did each round-1 finding close, and is decision 2c sound.
Late-round rule honoured: nothing below is a preference, and I dropped two stylistic observations
that did not survive it.

**Overall: PASS.** All ten findings CLOSED. Decision 2c is sound and independently verified. Two
new defects (R2-1, R2-2) are cross-file collisions the folding introduced, both one-line fixes in
files other than the ones I was reviewing in round 1.

## Per-finding closure

| # | round-1 finding | verdict | citation in the new text |
|---|---|---|---|
| F1 | marker text `BG_SURFACE` illegible (1.27:1) | **CLOSED** | `:541` `text=COLOR.FG_PRIMARY`; `:544-563` replaces the analogy with the composite measurement and the four-row contrast table; premise 40 (`:897`) records it as a self-correction |
| F2 | five wrong ctypes signatures | **CLOSED** | `:301-315` all five corrected verbatim to the Odin; premise 36 (`:893`) cites `ffi.odin:1218, 1238, 1263, 1279, 1300`; and the class is now gated rather than warned (2c) |
| F3 | `code.py:175` tooltip bug refuted | **CLOSED** | `:518-524` states the call is correct today, names the real defect (`ed_marker_tooltip` bound at `ffi.py:225` and called nowhere; glyph hard-coded `0` at `ffi.py:473`) |
| F4 | test 3's GLYPH assertion buffer-dependent | **CLOSED** | test 4 at `:784-796` drops the comparison for `FRAME` / `POPUP_GLYPH` / text-origin and states why, with both measured directions (840→757 wide, 45→58 short); premise 39 (`:896`) |
| F5 | `render_state` missing `command_message` | **CLOSED** | `:676-682` adds the two precisions: the method and `_SIG` entry already exist (`ffi.py:632`, `:291`), and the gap is one the wave CREATES because decision 6 deletes the ungated imgui draw at `code.py:400-405` |
| F6 | six vendored files, not five | **CLOSED** | `:107-109` names all six from `git ls-files`; `:119-127` corrects BOTH `conventions.md` sentences and goes to seven after `abi_probe.py` |
| F7 | `E` glyph invisible with line numbers off | **CLOSED** | `:583-589` states the degradation and its ABI reason; premise 43 (`:900`); manual-verification bullet qualified |
| F8 | `ed_set_style` clobbers the chrome flags | **CLOSED** | `:71-81` in W-F's out-of-scope, with the measurement and the `ed_draw_chrome`-survives half; premise 42 (`:899`); mirrored in both W-E files |
| F9 | `dev_flow.md` module map not in the docs list | **CLOSED** | `:740-746` under "Not touched", with the reason it is named at all; premise 44 (`:901`) |
| F10 | `ensure_loaded` fails at load, not in-frame | **CLOSED** | `:175-180` rewrites the justification to "adds the NAME in one message, instead of an `AttributeError` thrown out of a binding loop"; premise 41 (`:898`) |

Two corrections to my own round-1 numbers, in the drafter's favour: the contrast figures are
1.26 / 4.10 / 10.27 rather than my 1.27 / 4.09 / 10.25 (rounding in the linearisation; same
conclusion, and the spec's are the ones I re-derived). No finding changed verdict.

## Decision 2c: the vendored `abi_probe.py` argtypes gate

**Sound, and the strongest single mechanism in the spec.** Everything below I ran rather than read.

**The 93-key equality, verified independently.** AST-parsed `git -C ~/src/editor show
c5c6ae2:ffi/probe.py`: the `_SIG` assignment is a plain `ast.Dict` with **93 entries, 93 unique, no
duplicate keys**, and `diff` against the sorted `nm -D --defined-only` `ed_*` symbol list is clean in
both directions. The spec's claim is exact.

**The six signatures, verified against the Odin.** All five I caught in round 1 are in the upstream
probe exactly as I corrected them (`probe.py:105, 110, 112, 113, 114`), and `ed_add_marker`
(`:118-121`) is `(None, [void_p, int32] + [float]*12 + [int32, char_p])`, matching `ffi.odin:601-609`
parameter for parameter.

I did not stop at the six. I wrote a normalising parser over all 93 `@(export)` procs in
`ffi.odin` at `c5c6ae2` and compared every restype and every argument against the probe's table:
**93 compared, 0 mismatches** (one entry, `ed_primitives`, uses `[^]FFI_Primitive`, which my type map
does not carry; checked by hand and it matches). So the artifact the gate rests on is not merely
authoritative by provenance, it is correct by measurement.

**The test mechanism, executed end to end.** Built the `_parse_upstream_sig` the spec describes
(`ast.parse`, find the `_SIG` assignment, `eval` each value in a namespace of `ctypes` and `Prim`)
and ran it against the tree's real `_SIG`:

- 93 upstream entries parsed, 65 ours;
- `ctypes.POINTER(c_float) is ctypes.POINTER(c_float)` is `True`, so the pointer-identity claim
  holds and `_P(ctypes.c_float)` compares equal to `ctypes.POINTER(ctypes.c_float)`;
- across the 65 entries the two tables share, the **only** mismatch is `ed_add_marker` — 8 floats
  ours against 12 upstream — which is precisely the signature this wave changes.

So the gate is red on today's tree for exactly the reason claimed and goes green when the wave lands.
Falsifier confirmed: substituting the first draft's `"ed_find": (c_int32, [void_p, char_p, c_bool])`
is detected and names `ed_find`.

**AST-parsing a vendored `.py` is the right shape.** It is parsed, never imported, which matters
concretely: the upstream file runs a full probe session at module scope (`:179` binds every
signature, then exercises the ABI), so an import would execute it. Parsing also means a `.so`/`.py`
version skew cannot silently pass — both come from the same sha by construction.

**What happens if the editor repo changes `probe.py`'s shape.** Three cases, and the spec's helper
handles the dangerous one correctly only if it is written as specified:

1. *Renamed or restructured `_SIG`* (moved into a function, split, renamed) — the helper finds no
   module-level `_SIG` assignment. This MUST raise rather than return `{}`: an empty dict compared
   against a 93-entry `_SIG` fails the assertion anyway, so the test is red either way, but the
   message would be misleading. The spec's prose says the helper "finds the `_SIG` assignment",
   which is satisfied by a raise; I verified a raising version behaves correctly.
2. *An entry gains a construct the evaluator does not model* (a helper alias, a comprehension) —
   `eval` raises `NameError` inside the test. Loud, at the right file, which is the correct failure.
3. *A signature legitimately changes upstream* — the gate goes red on the next `make gates` naming
   the entry, which is the whole point.

In every case the test fails loudly at the check step rather than passing vacuously. The one shape
that would be unsound — importing the module and reading `probe._SIG` — is explicitly rejected in
the spec's own code comment ("Parsed, never imported: it runs a probe session at import time").

**The formatter-exclusion half, verified.** `uv run ruff check` on the file reports exactly the six
errors claimed, and by the claimed codes: `SIM905` x3, `B905`, `RUF007`, `B007`.
`.pre-commit-config.yaml:1` carries the top-level `exclude` with exactly the
`shaderbox/resources/emoji/emoji-test\.txt$` precedent cited, and `:11-17` runs `ruff --fix` plus
`ruff-format`, so the rewrite risk is real and the proposed regex is the right instrument. This is
not a sidestepped convention: the file is not this repo's source, and the repo's rule is that a
convention collision means the design is wrong — here the design is "hold upstream's bytes", and a
formatter is what would break it.

### R2-1 (new, from 2c). Pyright is NOT clean on the vendored probe, so the spec's disjunct resolves to its second branch and the wave must take it.

Decision 2c says "the wave also confirms at implementation time that `pyright` is clean on it or
adds it to pyright's `exclude`". Ran it: **three errors**, not zero.

    probe.py:382:71  - error: Object of type "None" cannot be used as iterable value (reportOptionalIterable)
    probe.py:1391:13 - error: Tuple size mismatch; expected 2 but received 3 (reportAssignmentType)
    3 errors, 0 warnings, 0 informations

`[tool.pyright]` (`pyproject.toml:87-91`) is `include = ["shaderbox"]` with **no `exclude` key at
all**, and the hook is `pass_filenames: false` with `entry: uv run pyright shaderbox`, so
`shaderbox/resources/editor/abi_probe.py` is reached. There is no `.py` under `resources/` today
except `resources/__init__.py`, so this file would be the first, and `make gates` would go red at
the check step on the very commit that adds it. Leaving it as a conditional the implementer
evaluates invites discovering it at gate time.

**Fix (paste):** State it as a decided step, not a conditional: `pyright` reports three errors on the
vendored probe (`reportOptionalIterable` at `:382`, `reportAssignmentType` at `:1391`), and
`[tool.pyright]` (`pyproject.toml:87-91`) has `include = ["shaderbox"]` and no `exclude` key, so the
file is reached and `make check` goes red. The wave adds `exclude = ["shaderbox/resources/editor/abi_probe.py"]`
to `[tool.pyright]` in the same commit, for the same reason as the ruff exclusion: the file is
upstream's bytes, not this repo's source.

### R2-2 (new, from the folding). W-F and W-E both claim to write `Editor.set_style` / `get_style`, and they name the enum differently.

Not a preference: the two specs give conflicting instructions for the same two methods in the same
file, and they are implemented by different commits.

- W-F `:67-68`: "W-F binds `ed_set_style` / `ed_style` **and exposes them as `Editor.set_style` /
  `Editor.get_style`** so W-E has something to call", with the bodies at `:371-377` and
  `class Style(IntEnum): VIM = 0; STANDARD = 1` at `:379`. W-F's § Files touched (`:721`) lists
  "`set_draw_chrome`, `set_style`, `get_style`; the `Style` enum".
- W-E `:8-10`: "This spec assumes both are bound and that **`Editor` carries no wrapper method for
  either yet; adding the two methods is W-E's**, because W-E is what calls them", with its own
  bodies and `class EditorStyle(IntEnum)`.

W-F lands first (parent § Order, step 4 before step 5). Under W-F's text the methods and a `Style`
enum exist; W-E then adds them again under the name `EditorStyle`. The likely outcome is a duplicate
enum or a rename churning a file W-F just wrote.

W-F's placement is the correct one on the repo's own reasoning: the mirror rule W-F makes enforceable
is about `editor/ffi.py` being complete against the ABI, and W-F is the wave that touches that file.
W-E's stated reason ("W-E is what calls them") is the weaker claim, since W-F already writes
`set_draw_chrome` without W-E calling it.

**Fix (paste):** One line in `50_wave_e_keyboard.md`: replace "adding the two methods is W-E's,
because W-E is what calls them" with "W-F adds `Editor.set_style` / `Editor.get_style` and the
`Style` enum alongside the binding (W-F § Design decisions item 3); W-E calls them and adds no ffi
code", and change `EditorStyle` to `Style` at its two use sites, including the
`_apply_editor_settings_to` line.

## The W-E and `02_keybindings.md` insertions

**Both read correctly.** `50_wave_e_keyboard.md:338-353` puts `set_style` first in
`_apply_editor_settings_to` and then carries the reason under the heading "**'First' is
load-bearing, not stylistic**", citing `ffi.odin:940`, the measured `LINE_NUMBERS` result, and the
`ed_draw_chrome`-survives half that keeps W-F's `set_draw_chrome(True)` safe. It attributes the
constraint to this review, which is accurate.

`02_keybindings.md:311-323` carries the same constraint under "One ordering constraint the keymap
switch carries", opening by saying why a non-chord fact lives in a chord document (it lands in the
same function W-E edits) — which is the right justification for its placement. The two statements do
not drift from each other or from W-F `:71-81`; I diffed the three by hand and the measured claims are
identical.

The `set_style`-before-`set_chrome_flag` order is what the measurement requires, and W-E's code block
already shows it in the right position rather than only asserting it in prose.

## False trails (round 2)

- W-F's § 1 heading still reads "two files change, not five" while its body correctly says six
  vendored today and seven after. The body governs and the count in it is right; the heading's
  "not five" is vestigial. Not filed as a finding: no instruction depends on it.
- The spec's decision-10 bucket figures still carry the `Background: 1` that only appears with a
  marker present. I raised this in round 1 as a false trail and it stayed; still harmless, since
  the test it belongs to no longer asserts on those counts.
- I checked whether `abi_probe.py` under `resources/` would be swept into the shipped bundle
  incorrectly: `build.sh` copies `shaderbox/` wholesale and strips only bytecode and the Windows
  `.so`, so the file ships as data, which is the same disposition as `emoji-test.txt`. Correct as
  designed, nothing to change.
- `ed_primitives` is the one upstream entry my Odin type-map could not normalise (`[^]FFI_Primitive`);
  hand-checked against `ffi.odin:470` and it matches, so the 93/93 result stands.

## Coverage statement (round 2)

Read: the ten folded sites in `40_wave_f_editor_chrome.md` (`:71-81, 103-137, 174-185, 190-264,
301-315, 518-596, 651-712, 740-746, 784-796, 893-902, 949-990`); `50_wave_e_keyboard.md:1-13, 45-67,
336-375`; `02_keybindings.md:310-330`; `.pre-commit-config.yaml:1-20`; `pyproject.toml:87-93`;
`conventions.md:867, 869, 896`.

Measured: AST extraction of the vendored probe's `_SIG` (93 entries, uniqueness, key-set `diff`
against `nm -D`); a normalising parser over all 93 `@(export)` procs in `ffi.odin` at `c5c6ae2`
compared restype-and-argument-wise against the probe (0 mismatches); the spec's `_parse_upstream_sig`
mechanism built and run against the tree's `_SIG` (65 vs 93, pointer identity True, sole mismatch
`ed_add_marker`); the `ed_find` falsifier; `uv run ruff check` on the file (six errors, codes
confirmed); `uv run pyright` on the file (three errors — R2-1); and a check that no other `.py`
lives under `shaderbox/resources/`.

Not re-checked, because round 1 measured them and the folding did not touch the underlying claims:
the chrome switch probes, the marker anchoring, the text-origin arithmetic, and the six host sites.
