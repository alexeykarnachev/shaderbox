# 071 W-A — Editor library: last line, shift, star, search highlight (#1 #2 #6 #11 + D10)

Parent: `01_spec.md § W-A`. Two repos: the library side landed in the editor repo as feature
007, commit `aa8c6719ac30808a5fae66df79590756a8683698` ("Close four vim gaps: empty last line,
shift, star, search highlight"), done by the editor session from one request carrying the four
ledger items verbatim, with the headless reproductions. This wave is the shaderbox half: the
re-vendor and the host's binding of the one new primitive.

## The library side, as the editor session reported it

1. `jdd` / `jVd` on `"a\n"` give `"a"`; `dd` on `""` stays; `jcc` keeps the empty line. The new
   corpus rows also caught `jdd` on `"a\n\n"` (a middle empty line) deleting both lines.
2. `>>` / `<<` with counts, `>j`, `>ip`, `Vj>`, `vj>`, `V2>` (count = indents in visual); one
   undo step, register untouched. The cursor lands at its desired column on the first line, not
   the first non-blank: nvim under `nostartofline` (its default) does `coladvance(curswant)`,
   and the oracle decided it. `<<` takes up to one indent of leading spaces.
3. `*` / `#` whole-word, `g*` / `g#` substring; `n` / `N` keep `*`'s rule until the next `/` or
   `ed_find`. Word choice in vim's order: keyword under the cursor, first keyword after it on the
   line, else the non-blank run. The cursor moves to the word's start before searching, so `#`
   mid-word skips its own word.
4. Search highlight: primitive kind `Search_Match = 10`, theme slot `Search_Match = 23`
   (default `(0.98, 0.82, 0.45, 0.30)`), view flag `Highlight_Search = 3`, ON by default. Every
   match of the stored pattern is lit whether it came from `/`, `*`, `n` or `ed_find`; the typed
   prefix is lit while a `/` or `?` line is open; a normal-mode Escape puts it out and keeps the
   pattern; `n` or `/` lights it again. No exported function was added, removed or changed
   shape: the ABI delta is the three enum extensions.

## The re-vendor

Rebuilt from the committed sha with `odin-linux-amd64-nightly+2026-07-10`
(`odin build ffi -build-mode:shared -no-entry-point -out:libeditor.so`). The rebuild matches the
editor repo's own binary in size and in its 93 `ed_*` exports and differs in header bytes only
(the build is not byte-reproducible); the rebuild is what ships. Copied the whole set of seven:
`libeditor.so`, `atlas.png`, `atlas.json` (both unchanged at this sha), `abi_probe.py` (from
`ffi/probe.py`, +75 lines of probe coverage for the new items), `vim_coverage.md` (three boxes
ticked), `standard_keymap.md` (unchanged), `VERSION`.

Host binding: `editor/ffi.py` gains `Kind.SEARCH_MATCH = 10`, `Slot.SEARCH_MATCH = 23`,
`ViewFlag.HIGHLIGHT_SEARCH = 3`. `theme.py` feeds the slot `fade(ACCENT_ACTIVE, 0.30)`, a
different hue from the bracket box and translucent because the band is drawn over the glyphs.
`editor/render.py` needs nothing: every non-glyph kind is an untextured quad and the library
sorts the array into draw order. No Settings toggle: D10 says on by default and nothing asked
for a switch.

**Host mitigations this sha makes dead: none.** The four items are additions; no host code
re-derived any of them.

Pinned in `tests/test_editor_ffi.py`, one test per item through `Editor.feed` against the
vendored binary: the empty-last-line delete, the shift operators (with the one-undo-step
check), the whole-word `*` skipping `foobar`, and the search bands counted from `prims_list()`
through the `/fo` prefix, the closed line, Esc, and `n`. The argtypes gate against
`abi_probe.py` and `tests/test_keymap_disjoint.py`'s parse of `vim_coverage.md` pass unchanged.

## What went wrong the first time

Commit `63e4c87` claimed this re-vendor and carried only this file. The vendored copies, the
enum extensions, the slot and the four tests had been reverted from the working tree before the
`git add`: a post-impl reviewer running concurrently tested "a clean checkout" of its commit in
the shared tree, and the gate that commit reported green ran on the old binary, since the tests
it named no longer existed (a `pytest -k` count of 0 for them was seen and dismissed). The
maintainer found it in the app: no `*`, no highlight. Redone in `ae94086`, each test seen
PASSED by name against the new binary, `git show --stat` checked for the seven files. Two rules
follow, both filed: a reviewer never stashes or checks out in the shared tree
(`dev_flow.md ## Feature flow` step 6), and a re-vendor commit is verified by its stat, not by
the gate's colour. A follow-up commit adds the last layout's primitive count to the panel's
redraw fingerprint, since Esc putting the highlight out changes nothing else the fingerprint
reads.

## Manual verification (the maintainer, in the app)

1. A file ending in an empty line: `G`, `dd` removes it. Falsifier: the line stays.
2. `>>` indents the line by four, `<<` undoes it, `Vj>` shifts two, `u` restores both at once.
3. `*` on a word jumps to its next whole-word occurrence and every occurrence lights up in the
   orange band; typing `/foo` lights matches as you type; Esc clears them; `n` still works and
   lights them again.
