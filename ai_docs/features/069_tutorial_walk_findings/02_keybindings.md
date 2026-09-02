# 069: the keybinding ownership audit (D7)

The table locked decision D7 owes: every app chord in `COMMAND_SPECS`, crossed with the three
states a key press can arrive in, with one owner named per cell and no cell blank. Finding #26 is
what asks for it, verbatim:

> "we will integrate the standard editor key schema; review ALL keybindings so nothing conflicts —
> standard editor, vim editor, global hotkeys. Clean and conflict-less."

The table is the input to W-E (`50_wave_e_keyboard.md`), which lands the moves as `CommandSpec`
edits and pins the result with a test that reads both keymaps' chord lists out of the vendored
docs rather than out of this file.

---

## The rule

One rule decides every cell. It is generic; there are no per-chord carve-outs and this document
grants none.

1. **A focused editor owns every chord its ACTIVE keymap lists.** The list is whatever
   `shaderbox/resources/editor/vim_coverage.md` (vim) or `standard_keymap.md` (standard) names at
   the vendored `VERSION`. Not what the host approximates, not what a reader remembers vim doing.
2. **The app owns every chord the active keymap does not list**, in every state.
3. **An app chord that must work while the editor is focused lives on Alt or an F-key**, because
   neither keymap claims either tier. That is the whole mechanism by which rule 2 is made to
   produce a usable app rather than a set of dead keys.
4. **When rule 1 takes a chord the app wants everywhere, the app chord MOVES** to the Alt or F-key
   tier. It does not get a carve-out, and it does not get to keep the chord for the unfocused case
   only: a verb the user reaches from two places must be one key.

The three states are:

- **app unfocused**: no editor session holds imgui focus (`App.editor_focused` is False). Both
  keymaps are irrelevant here; nothing reaches `ed_key`.
- **vim focused**: the editor child holds focus and `EditorSettings.keymap == "vim"`.
- **standard focused**: the editor child holds focus and `EditorSettings.keymap == "standard"`.

`App.editor_focused` is the only focus notion left after D4 removes the region system, which is
what makes a three-column table sufficient. The copilot chat's `copilot_focused` gates
`CommandScope.COPILOT` and is orthogonal to the editor; the chat is not an editor and neither
keymap runs inside it.

## What each keymap actually lists

Both lists are read from the vendored docs at `VERSION` = `65264dc4930838483b6ac3ebbcc5774a5f5ddfef`
(re-vendored by W-F; both files are byte-identical at `c5c6ae2`, so the shas differ and the lists do
not). They are reproduced here as a reading aid. **The test reads the files, never this section**:
a list retyped from an artifact stops tracking it, which is the same reason W-F's export test reads
`nm` output instead of a typed export list.

### Vim (`vim_coverage.md`)

Two notations appear in the file and both are chords: `CTRL-X` in the motion sections (vim's own
`:help` spelling) and `<C-x>` in the scrolling and word sections.

| Chord | Where | Status |
|---|---|---|
| Ctrl+H | `h` `<Left>` `CTRL-H` `<BS>`, left | `[x]` |
| Ctrl+J | `j` `<Down>` `CTRL-J` `<NL>` `CTRL-N`, down | `[x]` |
| Ctrl+N | same row as Ctrl+J | `[x]` |
| Ctrl+P | `k` `<Up>` `CTRL-P`, up | `[x]` |
| Ctrl+M | `+` `CTRL-M` `<CR>`, down, first non-blank | `[x]` |
| Ctrl+R | `u` `CTRL-R`, undo tree navigation | `[x]` |
| Ctrl+D | `<C-d>` `<C-u>`, half window, cursor and view | `[x]` |
| Ctrl+U | same row | `[x]` |
| Ctrl+F | `<C-f>` `<C-b>`, full window | `[x]` |
| Ctrl+B | same row | `[x]` |
| Ctrl+E | `<C-e>` `<C-y>`, view alone | `[x]` |
| Ctrl+Y | same row | `[x]` |
| Ctrl+Left | `B` `<C-Left>`, WORDs backward | `[x]` |
| Ctrl+Right | `W` `<C-Right>`, WORDs forward | `[x]` |
| Ctrl+Home | `<C-End>` `<C-Home>`, buffer start | `[ ]` |
| Ctrl+End | same row | `[ ]` |

Sixteen chords, fourteen implemented and two declared-not-yet. **The audit treats a `[ ]` row as
editor-owned anyway.** A chord the keymap has declared its own and not yet built is a chord the app
must not squat on, because the day it lands the app's binding becomes the collision the audit
exists to prevent. The disjointness test therefore parses both marks.

### Standard (`standard_keymap.md`)

The file is three markdown tables with a `| Key | Behaviour |` header. Every chord it lists:

| Chord | Section |
|---|---|
| Ctrl+Left, Ctrl+Right | Movement |
| Ctrl+Home, Ctrl+End | Movement |
| Ctrl+Backspace, Ctrl+Delete | Editing |
| Shift+Tab | Editing |
| Ctrl+A | Selection, undo and completion |
| Ctrl+Z | Selection, undo and completion |
| Ctrl+Y, Ctrl+Shift+Z | Selection, undo and completion |
| Ctrl+Space, Ctrl+N | Selection, undo and completion |

Thirteen chords across those eight rows. The file also states the fall-through in one sentence,
which is rule 2 written by the library rather than by us: "Every other key with Ctrl, Alt or Super
held -- Ctrl+X, Ctrl+C and Ctrl+V among them -- returns false from `ed_key` and is the host's."

The vim file makes no such statement. It does not need to: the enumeration is the claim, and the
host's `_handle_vim_chord` fallback already runs only after `ed_key` returned unconsumed.

---

## The table

Every entry in `COMMAND_SPECS` at HEAD `6a85564`, in table order, plus the chords that belong to a
keymap and to no app command. Winner per cell; **MOVE** marks a chord the rule forces to change.

Legend: **app** = the `CommandSpec` fires; **vim** / **standard** = the editor consumes it and no
app command fires; a chord in parentheses after **app** is the new chord.

| App command | Chord today | App unfocused | Vim focused | Standard focused | Verdict |
|---|---|---|---|---|---|
| `OPEN_PROJECT` | Ctrl+O | app | app | app | keep, neither keymap lists Ctrl+O |
| `SAVE` | Ctrl+S | app | app | app | keep |
| `QUIT` | Ctrl+Q | app | app | app | keep |
| `NEW_DOCUMENT` | Ctrl+N | app | **vim** (down) | **standard** (completion) | **MOVE 1** → Ctrl+Shift+N |
| `DELETE_DOCUMENT` | Ctrl+D | app | **vim** (half-page down) | app | **MOVE 2** → Alt+D |
| `TOGGLE_DOCUMENT_PLAY` | Ctrl+Space | app | app | **standard** (completion) | **MOVE 3** → F5 |
| `OPEN_SHADER` | Ctrl+E | app | **vim** (scroll down a line) | app | **MOVE 4** → Alt+C |
| `OPEN_SCRIPT` | Ctrl+R | app | **vim** (redo) | app | **MOVE 5** → Alt+R |
| `CYCLE_CODE_TAB` | Ctrl+Tab | app | app | app | keep, neither lists Ctrl+Tab |
| `CLOSE_CODE_TAB` | Ctrl+W | app (no-op, scope) | app | app | keep, neither lists Ctrl+W; see note 4 |
| `JUMP_NEXT_ERROR` | F8 | app | app | app | keep, F-key tier |
| `FOCUS_TAB_DOCUMENT` | Ctrl+1 | app | app | app | keep |
| `FOCUS_TAB_RENDER` | Ctrl+2 | app | app | app | keep |
| `FOCUS_TAB_SHARE` | Ctrl+3 | app | app | app | keep |
| `CYCLE_REGION` | Ctrl+` | n/a | n/a | n/a | **deleted by D4** (the command goes, not the chord) |
| `TOGGLE_COPILOT` | Ctrl+J | app | **vim** (down) | app | **MOVE 6** → Alt+J |
| `CYCLE_COPILOT_LAYOUT` | Ctrl+H | app (chat only) | app (chat only) | app (chat only) | keep, see note 1 |
| `OPEN_LIB_PICKER` | Ctrl+P | app | **vim** (up) | app | **MOVE 7** → Alt+L |
| `OPEN_PALETTE` | Ctrl+Shift+P | app | app | app | keep, see note 2 |
| `OPEN_SETTINGS` | Alt+S | app | app | app | keep, Alt tier |
| `OPEN_PASS_SETTINGS` | Alt+P | app | app | app | keep, Alt tier, W-C's provisional stands |
| `ADD_PASS` | Alt+A | app | app | app | keep, Alt tier, W-C's provisional stands |
| `EXAMPLES` | Alt+E | app | app | app | keep, Alt tier |
| `HELP` | F1 | app | app | app | keep, F-key tier |
| `TOGGLE_CHEATSHEET` | Alt+/ | app | app | app | keep, Alt tier |
| `RESET_FEEDBACK` (W-G, new) | none yet | app | app | app | **new** → F6, see note 3 |

### The editor-owned chords no app command uses

Recorded so nothing lands on them later. Each is a cell in the same table with the app column
reading "free". Free today, and not free to take.

| Chord | App unfocused | Vim focused | Standard focused |
|---|---|---|---|
| Ctrl+A | free | free | **standard** (select all) |
| Ctrl+B | free | **vim** (page up) | free |
| Ctrl+D | free after MOVE 2 | **vim** (half-page down) | free |
| Ctrl+E | free after MOVE 4 | **vim** (scroll down) | free |
| Ctrl+F | free | **vim** (page down) | free |
| Ctrl+H | free | **vim** (left) | free |
| Ctrl+J | free after MOVE 6 | **vim** (down) | free |
| Ctrl+M | free | **vim** (down, first non-blank) | free |
| Ctrl+N | free after MOVE 1 | **vim** (down) | **standard** (completion) |
| Ctrl+P | free after MOVE 7 | **vim** (up) | free |
| Ctrl+R | free after MOVE 5 | **vim** (redo) | free |
| Ctrl+U | free | **vim** (half-page up) | free |
| Ctrl+Y | free | **vim** (scroll up) | **standard** (redo) |
| Ctrl+Z | free | free | **standard** (undo) |
| Ctrl+Shift+Z | free | free | **standard** (redo) |
| Ctrl+Space | free after MOVE 3 | free | **standard** (completion) |
| Ctrl+Left, Ctrl+Right | free | **vim** (WORD motions) | **standard** (word motions) |
| Ctrl+Home, Ctrl+End | free | **vim** (declared, `[ ]`) | **standard** (buffer ends) |
| Ctrl+Backspace, Ctrl+Delete | free | free | **standard** (word delete) |
| Shift+Tab | free | free | **standard** (dedent) |

### The three host-owned chords that are neither

| Chord | Owner | Why it is not a keymap chord |
|---|---|---|
| Ctrl+C | host clipboard (`hotkeys.py::_handle_clipboard`) | Neither keymap binds it; the standard doc names it as the host's outright, and vim's single register is synced across the boundary at the drain. |
| Ctrl+X | host clipboard | Same. |
| Ctrl+V | host clipboard | Same. Vim has no visual-block mode here, so nothing wants the chord. |

Escape is the fourth of this kind and is not a chord in the table sense: `App._install_escape_filter`
swallows it before imgui when the editor is focused, so the editor owns it unconditionally under
both keymaps and `_handle_escape` never sees those frames.

---

## Notes on the cells the rule decides in a way worth stating

**Note 1, Ctrl+H stays on `CYCLE_COPILOT_LAYOUT` despite vim listing Ctrl+H.** This looks like a
carve-out and is not one. The command's scope is `CommandScope.COPILOT`, which `spec_eligible`
gates on `app.copilot_focused`. Editor focus and chat focus are mutually exclusive: only one
focus flag is true at a time, which `commands.py::scopes_overlap` already encodes. So the chord
never arrives at the app while the editor is focused, and never arrives at the editor while the
chat is focused. The rule's premise ("a focused editor owns...") is simply not met in the state
where the app fires. Every other row in the table is `CommandScope.GLOBAL` and does meet it.

The one thing this does cost: a user in the chat cannot use Ctrl+H as a vim left-motion, which is
correct, because the chat is an imgui `input_text` and never a vim buffer.

**Note 2, Ctrl+Shift+P is a different chord int from Ctrl+P.** `_chord` ORs the modifier bits, so
`Ctrl+Shift+P` and `Ctrl+P` are distinct integers and imgui matches the exact chord. Neither keymap
lists a Shift+Ctrl combination other than Ctrl+Shift+Z, so the palette keeps its chord even though
its unshifted sibling moves.

**Note 3, `RESET_FEEDBACK` takes F6, not a Ctrl chord.** W-G introduces the command ("Clear
canvas") and the parent spec's open question 4 proposes F6 provisionally. The rule confirms it: the
verb has to work while the user is looking at the shader they are painting into, so it must survive
editor focus, so rule 3 puts it on Alt or an F-key. F6 is chosen over an Alt letter because it sits
beside F5 (`TOGGLE_DOCUMENT_PLAY` after MOVE 3) and the two are the document's transport pair.

**And the F-key tier is disjoint by the host's translation layer, not only by the docs.** Rule 3
rests on "neither keymap claims either tier", which is a statement about the KEYMAPS, derived from
the doc parse. There is a stronger fact underneath it: `editor/input.py::translate_key` returns an
event only for a key in `_SPECIAL_KEYS` (escape, enter, tab, backspace, delete, the four arrows,
home, end, page up/down) or, under `_CHORD_MODS` (ctrl/alt/super, **not** shift), a `_key_char`
letter or digit. F-keys are in neither set, so `translate_key` returns `None` and an F-key never
enters `app.editor_key_events` and never reaches `ed_key` under either keymap. That is what
actually guarantees W-E's manual step 7 ("neither types a character into the buffer"): F5 and F6
cannot reach the editor even if a future keymap claimed them.

**Note 4, Ctrl+W is clean, and its host carve-out is now the only reason `_handle_vim_chord`
touches it.** `CLOSE_CODE_TAB` is `CommandScope.EDITOR`, so it fires only while the editor is
focused, which is why the "app unfocused" cell reads no-op rather than app. Measured against both
parsed lists, Ctrl+W appears in neither, so the rule leaves it with the app in all three states
and the chord needs no move. What DOES touch it is the host's own insert-mode word-delete, which
survives as a host behaviour with its NORMAL-mode fall-through intact (see the
`_VIM_RESERVED_CHORDS` section). The disjointness test asserts scoped specs the same as GLOBAL
ones wherever the chord is genuinely absent from both lists; only Ctrl+H needs the scope
exemption.

---

## The moves, and why each is forced

Seven, plus one deletion and one new binding. Each names the rule clause that forces it and the
keymap row that proves the collision.

| # | Command | Old | New | Forced by |
|---|---|---|---|---|
| 1 | `NEW_DOCUMENT` | Ctrl+N | **Ctrl+Shift+N** | Rule 1 in BOTH columns: vim's `j <Down> CTRL-J <NL> CTRL-N`, standard's `Ctrl+Space, Ctrl+N` completion row. |
| 2 | `DELETE_DOCUMENT` | Ctrl+D | **Alt+D** | Rule 1 under vim: `<C-d> <C-u>` half-window scroll. Rule 4 forbids keeping it unfocused-only. |
| 3 | `TOGGLE_DOCUMENT_PLAY` | Ctrl+Space | **F5** | Rule 1 under standard: the `Ctrl+Space, Ctrl+N` completion row. |
| 4 | `OPEN_SHADER` | Ctrl+E | **Alt+C** | Rule 1 under vim: `<C-e> <C-y>` view scroll. |
| 5 | `OPEN_SCRIPT` | Ctrl+R | **Alt+R** | Rule 1 under vim: `u CTRL-R` undo tree. |
| 6 | `TOGGLE_COPILOT` | Ctrl+J | **Alt+J** | Rule 1 under vim: `CTRL-J` down. |
| 7 | `OPEN_LIB_PICKER` | Ctrl+P | **Alt+L** | Rule 1 under vim: `CTRL-P` up. |
| n/a | `CYCLE_REGION` | Ctrl+` | deleted | D4 removes the region system; the command has no callback left. |
| n/a | `RESET_FEEDBACK` | none yet | **F6** | New in W-G; rule 3 places it. |

### Where the new chords come from, and why not something else

The Alt tier had three occupants before W-C (Alt+S, Alt+E, Alt+/) and five after (Alt+P, Alt+A).
Four more land here, for nine.

- **Alt+D** (`DELETE_DOCUMENT`): the letter is unchanged from Ctrl+D, so muscle memory moves one
  modifier and no letter.
- **Alt+R** (`OPEN_SCRIPT`): same, unchanged letter.
- **Alt+J** (`TOGGLE_COPILOT`): same.
- **Alt+C** (`OPEN_SHADER`): the letter DOES change, because Alt+E is `EXAMPLES` and has been since
  018. `C` for code is the next-best mnemonic and is free. The alternative considered and rejected:
  moving `EXAMPLES` to make room for Alt+E, which trades one letter change for two.
- **Alt+L** (`OPEN_LIB_PICKER`): `L` for library. Alt+P is `OPEN_PASS_SETTINGS` (W-C, landed) and
  the audit does not move a chord that lands correctly under the rule to make room for one that
  needs relocating anyway.
- **Ctrl+Shift+N** (`NEW_DOCUMENT`): the one move that does NOT go to the Alt tier. Rule 3 says an
  app chord that must work inside the editor lives on Alt or an F-key; creating a document is the
  one verb in this set that plausibly never fires from inside a buffer, and Ctrl+Shift+N is the
  convention in every editor the maintainer uses. It is disjoint from both keymaps (neither lists
  any Ctrl+Shift chord but Ctrl+Shift+Z), so the disjointness test passes on it. The cost, stated so
  it is a choice and not an oversight: under vim, Ctrl+Shift+N reaches the app while focused only
  because the keymap leaves it unconsumed, which the doc's enumeration guarantees.
- **F5 / F6**: `_STANDALONE_KEYS` in `commands.py` already permits F1-F12 without a modifier and
  `chord_needs_modifier` exempts them, so both are legal bindings with no registry change. F1 (Help)
  and F8 (Jump to next error) are taken; F5 and F6 are free.

## Where the rule contradicted the parent spec, and what wins

The parent spec's open question 4 offered five provisional chords. The rule confirms three and is
silent on none.

| Parent's provisional | Rule's answer |
|---|---|
| `ADD_PASS` = Alt+A | **Confirmed.** Alt tier, neither keymap lists it. |
| `OPEN_PASS_SETTINGS` = Alt+P | **Confirmed.** Same. |
| `NEW_DOCUMENT` Ctrl+N → Ctrl+Shift+N | **Confirmed.** |
| `TOGGLE_DOCUMENT_PLAY` Ctrl+Space → F5 | **Confirmed.** |
| `RESET_FEEDBACK` = F6 | **Confirmed**, and the rule supplies the reasoning the parent left to the audit. |

The parent's W-E bullet names only two moves, "`NEW_DOCUMENT` off Ctrl+N, `TOGGLE_DOCUMENT_PLAY`
off Ctrl+Space (both collide with standard completion)". **The rule forces five more**, all under
the vim column, and the parent spec does not name them because finding #26's own analysis stopped
at "today resolved by focus, which #24 makes the only remaining focus notion". That sentence is
the collision: #24 does not remove `editor_focused`, so focus survives as a resolution mechanism:
but D7 rejects focus-dependent ownership as a *design*, because a chord that means two things is
what the maintainer asked to be rid of. Five app commands were living on vim's chords and being
rescued by the focus test. The rule ends that, and the cost is five relocated chords.

The rule also **contradicts the host's `_VIM_RESERVED_CHORDS`**, which is not a parent-spec
provisional but is a live piece of code the audit supersedes. See the next section.

## `_VIM_RESERVED_CHORDS` under the rule

Today `hotkeys.py::_VIM_RESERVED_CHORDS` is the string `"dufbeyrownphj"`, thirteen letters,
hand-maintained, vim-only, and consulted only as a fallback after `ed_key` returns unconsumed.
Under the rule it stops being a reserved SET and becomes the vim keymap's chord list, which the
doc already holds. Two discrepancies fall out, both of which W-E resolves in the code:

- **`o` and `w` are in the host set and in NO row of `vim_coverage.md`.** `grep` for `CTRL-O`,
  `<C-o>`, `CTRL-W` and `<C-w>` returns nothing. The host added them: `o` for vim's jumplist (which
  this editor does not implement) and `w` for insert-mode word-delete (which the host implements
  itself, in `_delete_word_back`). Under the rule, neither is an editor chord. `w` stays a host
  behaviour because it is a host behaviour, an insert-mode convenience the host provides, not a
  chord the keymap claims, and it already carries the one deliberate carve-out that NORMAL-mode
  Ctrl+W falls through to `CLOSE_CODE_TAB`. `o` becomes nothing: it consumes-noop today so that
  `OPEN_PROJECT` could not fire mid-insert, and after the audit `OPEN_PROJECT` is not a vim chord,
  so there is nothing to suppress.
- **`m` is in the doc and not in the host set.** `CTRL-M` is vim's "down, first non-blank". No app
  command uses Ctrl+M and none may, which the editor-owned table above now records.

The set becoming per-keymap data is what makes both discrepancies impossible to reintroduce: under
standard, the set is standard's list; under vim, it is vim's; and the disjointness test reads both
from the docs.

## One ordering constraint the keymap switch carries

Not a chord question, but it lands in the same `_apply_editor_settings_to` W-E edits, and W-F is
where `Editor.set_style` gets written, so it is recorded in both places. `ed_set_style` calls
`editor_set_keymap` and then replaces the WHOLE `Chrome` from `chrome_for(style)`
(`ffi.odin:940` at the vendored sha), so it resets `LINE_NUMBERS`, `RELATIVE_NUMBERS` and the
status flags to that style's defaults. Measured through the ABI: with `LINE_NUMBERS` set False by
the host, `ed_set_style(h, 0)` leaves it True. `ed_draw_chrome` is NOT part of `Chrome` and
survives the switch.

So `_apply_editor_settings_to` must call `set_style` BEFORE
`set_chrome_flag(ChromeFlag.LINE_NUMBERS, settings.show_line_numbers)`, or the user's line-numbers
setting is silently discarded on every settings apply.

## What the test asserts, in one sentence

Every `default_chord` in `COMMAND_SPECS` whose scope is `GLOBAL` is absent from both keymaps'
parsed chord sets. The scope qualifier is what lets Ctrl+H stay (note 1); everything else in the
table is GLOBAL and is asserted unconditionally. The parse, the failure mode when the doc format
changes, and the falsifier are in `50_wave_e_keyboard.md § Tests`.
