# The standard keymap

The non-modal keymap: typing inserts, Shift with a movement selects, a few
Ctrl chords cover undo, redo, select-all and word deletion. A host picks it
with `ed_set_style(h, 1)`; the reference editor takes `--standard`, and F6 switches at runtime. The vim
keymap is `docs/vim_coverage.md`; the decisions behind this one, with the
editor each rule comes from, are in `docs/features/005_standard_keymap/`.

This table and `STANDARD_BINDINGS` in `src/keymap_standard.odin` are checked
against each other by a test: a key named here that the table lacks, or a
bound key this file omits, fails `make test`.

## Movement

Shift held with any of these extends the selection; without Shift the caret
moves and the selection ends. With a non-empty selection and no Shift, Left
and Right collapse to the selection's start and end without moving further;
Up and Page Up move on from the start, Down and Page Down from the end; the
rest move from the caret. With the completion popup open, Up and Down move
the candidate, with or without Shift; every other movement closes the popup
first.

| Key | Behaviour |
|---|---|
| `Left` / `Right` | One codepoint, crossing line ends. |
| `Up` / `Down` | One line, keeping the desired column; a sticky End keeps the caret at the line end. |
| `Home` | The first non-blank; if already there, column 0. |
| `End` | The end of the line, sticky for Up/Down. |
| `Ctrl+Left` / `Ctrl+Right` | The start of the word before / the end of the word after the caret, by vim's word classes; a line end is a stop. |
| `Ctrl+Home` / `Ctrl+End` | The buffer's start / end. |
| `PageUp` / `PageDown` | A viewport of rows less two, keeping the screen column; before the first layout, or on a line holding a tab once the buffer has changed under the last layout, the key is consumed and nothing moves. |

## Editing

Each is refused on a read-only editor. Each closes an open completion popup
unless it accepts from it or re-filters it. Shift is ignored on `Backspace`,
`Delete` and `Enter`.

| Key | Behaviour |
|---|---|
| `character` | Replaces a non-empty selection, then inserts; re-filters the popup. |
| `Enter` | Replaces a non-empty selection, then a newline with the auto-indent rules; with the popup open, accepts. |
| `Tab` | With the popup open, accepts. With a multi-line selection, or one whole line, indents those lines one step and keeps the selection. Otherwise replaces the selection and inserts spaces to the next tab stop. |
| `Shift+Tab` | Removes up to one indent step from every line the selection or caret touches, keeping the selection. |
| `Backspace` / `Delete` | A non-empty selection, else one codepoint before / after the caret, joining lines. |
| `Ctrl+Backspace` / `Ctrl+Delete` | A non-empty selection, else to where Ctrl+Left / Ctrl+Right would land. |

## Selection, undo and completion

| Key | Behaviour |
|---|---|
| `Ctrl+A` | Selects the whole buffer. Works read-only. |
| `Escape` | Closes the popup, else ends a non-empty selection; consumed only when it did one of those, so an idle Escape is the host's. |
| `Ctrl+Z` | Undo. Refused read-only. |
| `Ctrl+Y`, `Ctrl+Shift+Z` | Redo. Refused read-only. |
| `Ctrl+Space`, `Ctrl+N` | Opens the completion popup on the word before the caret; with it open, moves to the next candidate. Opening is refused read-only; moving is not. |

Undo groups typing the way VS Code does: a group closes before the first
space typed after text, a word after that single space joins the group, a word
after two or more spaces starts a new one; Enter closes the group before
itself only; consecutive Backspaces or Deletes in one direction are one step;
a multi-line indent or dedent is one step.

Every other key with Ctrl, Alt or Super held -- Ctrl+X, Ctrl+C and Ctrl+V
among them -- returns false from `ed_key` and is the host's: the editor owns
no clipboard, and `ed_selection_text`, `ed_replace_selection` and
`ed_insert_at_cursor` are what a host builds one from. Under this keymap
`ed_mode` is always insert, `ed_pending` means "Escape would cancel
something" -- a popup or a selection -- and `ed_feed` types every plain
character, except that `<...>` is read as a key name and an unrecognised one
aborts the rest of the string.
