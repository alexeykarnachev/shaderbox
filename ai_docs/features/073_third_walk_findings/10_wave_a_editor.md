# 073 W-A — Editor library: closer snap, reverse-video caret, `K` seam, noselect (#1 #6 + D1; #2 #3 library halves)

Parent: `01_spec.md § W-A`. Two repos: the library side landed in the editor repo as feature
008 (`docs/features/008_reverse_video_and_closer_snap/feature.md` there), batch commit
`469eec4` plus three review-round fixes, re-vendored from `5e0e8a21c571e3707679100823fe2605fa8f237d`
on that repo's `master`. Done by the editor session from one batch message carrying the four
ledger rows verbatim; the optional fifth item (noselect) was taken. This wave is the shaderbox
half: the re-vendor and the host's binding.

## The library side, as the editor session reported it

1. **Closer snap (#1 + D1).** `Behavior.align_closers`, on by default: a closer typed as the
   first non-blank of its line replaces the leading whitespace with the indent of the line
   holding its matching opener (`bracket_scan`, any depth, all three pairs), one edit, one undo
   step with the insert session. An unmatched closer is a plain insert, which nvim's
   `smartindent` measured the same. Nothing host-side.
2. **Reverse-video caret (#6).** In normal / visual mode the caret quad is opaque and the glyph
   under it is emitted in the new theme slot 24 `Caret_Text` (default: the background). The
   recolor runs after markers, so a marker's text color on the caret's line leaves the caret
   glyph alone. Cursor line: nothing library-side; the host marks it per frame with
   `ed_add_marker`. Two markers on a line stack fills and the LATER text color wins.
3. **`K` (#2).** Stays unbound in normal / visual mode (insert types the letter), pinned in
   the editor's chord tests and probe, named in `vim_coverage.md`. Reason: vim's `K` runs
   `keywordprg`, an external program, and the host is that program. Popup host-side, anchored
   at the caret cell. A pending count or operator is dropped by any unbound key.
4. **Completion detail (#3).** Not taken; `Completion.kind` is the field it would fill when the
   spec's trigger fires.
5. **noselect.** `ed_complete_select(h, index)`: `-1` highlights nothing (Enter / Tab act as
   with no popup and close it; Down / Ctrl+N pick row 0; `ed_complete_selected` reports `-1`),
   any index selects that row. Per push batch, since every `ed_complete_begin` starts at row 0.
6. Fixed on the way, pre-existing: `cw<Esc>u<C-r>` with a host insert between `cw` and its
   typing left redo unable to remove the word again.

ABI delta from `d2f1955`, by `nm -D` of the two binaries: exactly one export added
(`ed_complete_select`), none removed or reshaped; theme slot 24 `Caret_Text` (25 slots);
`Caret` / `Caret_Insert` defaults now opaque; the primitive array is sorted by a per-kind RANK
(Caret before Glyph) rather than kind number, kind numbers unchanged; no new primitive kinds,
view flags or chrome flags. Atlas unchanged (checksums equal), `standard_keymap.md` unchanged,
`vim_coverage.md` gains "Indenting while typing" and "Various".

## The re-vendor

Rebuilt from the committed sha with `odin-linux-amd64-nightly+2026-07-10`
(`odin build ffi -build-mode:shared -no-entry-point -out:libeditor.so`): 94 `ed_*` exports
against the previous 93. Copied the whole set of seven (`libeditor.so`, `abi_probe.py` from
`ffi/probe.py`, `vim_coverage.md`, `standard_keymap.md`, `atlas.json`, `atlas.png`, `VERSION`).
The probe's draw-order check is now the rank check (`DRAW_ORDER.index`), as upstream wrote it.

Host binding:

- `editor/ffi.py`: `Slot.CARET_TEXT = 24`, `ed_complete_select` in the signature table,
  `Editor.complete_select(index)`; `complete_selected` documents `-1`.
- `theme.py`: `slot.CARET` goes opaque `ACCENT_PRIMARY` (a translucent block would leave the
  cutout glyph on a dim square), `slot.CARET_TEXT = BG_SURFACE`; `EDITOR_CURSOR_LINE_ALPHA`.
- `tabs/code.py::_apply_markers` takes the cursor line, puts it in the fingerprint, and adds
  its marker FIRST (`fade(BORDER, EDITOR_CURSOR_LINE_ALPHA)`, no text color: the band is
  vim's `cursorline`, syntax colors stay), then the error markers with their `FG_PRIMARY`
  text, then the hover mark.
- `_offer_completion` calls `complete_select(-1)` after an unasked batch. **Host mitigation
  this sha makes dead: `hotkeys.py::_track_completion_intent`**, the cancel-before-Enter
  workaround and its `editor_completion_navigated` flag, deleted.
- `K`: `hotkeys.py::_is_lookup_key` catches the unconsumed `K` in normal / visual mode and
  raises `app.editor_lookup_requested`; `code.py::_consume_lookup_request` resolves the word
  under the caret (`completion.word_at`, vim's rule: the word under or next after the cursor)
  through `completion.symbol_doc` into `app.editor_lookup: LookupPopup`;
  `_draw_lookup_popup` pins `ui_primitives.anchored_note` one cell below the caret (editor
  rect + text origin + cell, minus the scroll). Any key or click dismisses it, and the key
  still does its own work.

## Pinned by tests

`tests/test_editor_ffi.py`: `complete_select(-1)` leaves nothing highlighted, Enter is a
newline and closes, Down then Enter accepts, an index past the list is refused; the caret
glyph is emitted in `CARET_TEXT` with an opaque caret in normal mode and untouched in insert
mode; the marker-text-at-column-0 fact survives with the caret moved off the marked line (the
recolor would otherwise mask it). `tests/test_completion.py`: the unasked-popup Enter / Down /
explicit-batch sequence through the real driver, `word_at`, `K` recognized in normal and
visual mode and not in insert, a request resolving `SB_fbm` and `u_time` and nothing for a
keyword. The argtypes gate against the refreshed `abi_probe.py` passes.

## Manual verification (the maintainer, in the app)

1. In normal mode the glyph under the caret is cut out of an opaque block; in insert mode the
   bar leaves the glyph alone. The cursor line carries a faint band; an error line still flips
   its text.
2. Type `void f() {`, Enter, `if (a) {`, Enter, `x;`, Enter, `}`, Enter, `}`: the inner brace
   at 4, the outer at 0; `)` and `]` snap the same way; `u` restores, `Ctrl+R` re-applies.
3. Type `in` then Enter: a newline. Type `in`, Down, Enter: `int`. Ctrl+N on one letter opens
   with row 0 highlighted as before.
4. `K` over `SB_fbm` shows its signature and doc under the caret; over `u_time` its declaration
   and doc; over `float` nothing; any key or click dismisses it.
