# 078 W-C — Editor library re-vendor: `f738744` -> `88373ea`

What the editor session reported on the bugs batch (findings #7 #8 #9 #10), and what the
re-vendor changed here. The ABI is unchanged (94 exports, the same `nm -D` set), so the vendored
set is the binary plus the docs; no binding change, no host mitigation deleted.

## The editor session's report (its own words, condensed)

- **`dd` cursor column (#7) — real, fixed.** The rule is vim's `coladvance(curswant)` clamped:
  `j8|dd` lands on column 7, `j2|dd` on column 1, both inside the indent; not first-non-blank,
  not column 0. `operator_shift` had derived it for `>>` / `<<`; the linewise operators were
  placing at the range start. One shared helper now; `dd`, `2dd`, `dj`, `Vd` obey it.
- **Insert-mode `<Right>` (#9) — real, fixed.** Stepping by byte crossed the newline; now
  bounded by the line. Normal-mode `l` and `<Right>` at `$` were correct, and the count and
  repeat paths reproduce nothing, so the `l` half is closed as not reproduced on both sides.
- **Cursor after `u` (#8) — not the library's.** Measured before any change: `G$dd u` gives
  (1,2), `j$dd u` (1,4), `jdd k u` (1,0), nvim's answers; undo restores the caret to the first
  changed line at its own column. A coverage line now says so. The stale strip is a host
  display check.
- **The harness (#10) — built:** `tools/fuzz_diff.py`, random key walks replayed step by step
  into the nvim oracle comparing text, cursor and mode after every step, minimized to the
  shortest diverging sequence, printed as a corpus TSV row; runs under `make fuzz`.
  Mutation-gated: each of the two fixes reverted is found. It found seven unreported defects,
  all fixed: `O` with typed text did not fully undo; `J` inserted a space joining onto an empty
  line; `|` counted characters not screen columns and discarded its column; a blank line was
  not a word object (`diw` joined two lines); `o`/`O`/Enter after a tab-indented line copied
  the tab; a linewise yank moved the cursor to column 0; `X` was not implemented and not on
  the coverage checklist. Two divergences stay open and named in the harness's known set: a
  failed command not aborting the rest of a fed sequence (a keymap-wide contract, its own
  change), and `aw` starting on whitespace (three fixes each broke `vawawd`, backed out).
- Two harness lessons it recorded: `feed_keys` scanned to the first `>` anywhere, so
  `<<oN<Esc>` dropped every key after `<<`; and the oracle's tabstop was 8 against the editor's
  4, which reported a units mismatch as a motion defect three times.

## What changed here

`libeditor.so` rebuilt at `88373ea`, `vim_coverage.md` refreshed, `VERSION` updated; the atlas,
`standard_keymap.md` and `abi_probe.py` are byte-identical. Verified on the vendored binary
through `Editor.feed` / `ed_key`: `j$dd` on the three-line fixture lands at column 8 (was 0),
insert-mode `<Right>` three times on `ab\ncd` stops at (0,2) (was (1,0)). `make gates` green.

## Left for the display

- The cursor-line strip after `u` (#8's host half): the library's cursor is right, so if the
  strip still lags, it is in the panel's frame order.
- `l` held at the end of a line, if it recurs: name the mode.


## Fourth re-vendor: `6d526c6`

ABI unchanged (97 exports). The editor session's adversarial review of its own fuzzer raised
the walk count from 60 to 400 (the shipped build failed at 100) and enumerated the alphabet
from the keymap (fourteen keys were never pressed). Nine fixes, of which three touch ShaderBox
directly: `ed_set_text` no longer leaves a stale highlight cache (the buffer version restarted
at 1 and the cache was keyed to it, so old spans painted over new text until the next edit —
finding #18's color half, in all likelihood); `word_classes_apply` is linear (4000 lines:
+123 ms → +14.5 ms per layout); `>>` / `<<` replace a tab indent at a computed width. And a
correction: the cursor after `u` matches nvim for `dd`-shaped cases only; two
undo-after-visual / vertical cases diverge and are named in the fuzzer's known set. Verified on
the vendored binary: `dd` column, insert `<Right>`, a fed class survives a layout, `set_text`
re-lexes fresh. `make gates` green.
