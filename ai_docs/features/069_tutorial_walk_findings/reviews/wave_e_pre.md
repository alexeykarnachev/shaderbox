# W-E pre-implementation review

Reviewed: `50_wave_e_keyboard.md` + `02_keybindings.md`, against code at `ccd446b` (working tree
carries W-B/W-F edits, so every code citation below is `git show ccd446b:<path>`).

## Verdict

| Dimension | Verdict |
|---|---|
| Parent coverage (`01_spec.md § W-E`, findings 13/24/26) | **PASS** |
| Audit correctness (`02_keybindings.md`, the seven moves) | **PASS** — re-derived independently, exact match |
| Deletion completeness (019 out) | **FAIL** — the wave's central premise about `no_nav_inputs` is refuted by measurement, and four call sites are unlisted |
| Keymap-setting design | **PARTIAL** — a hard collision with W-F over `set_style`, plus one unhandled keymap-dependent branch |
| Test falsifiability | **PARTIAL** — two named falsifiers do not falsify |
| Docs | **PARTIAL** — two stale pointers survive the sweep |

Findings 1 and 2 are blockers; 3 is a merge conflict that has to be settled before either wave is
written. The rest are corrections the spec author can paste.

---

## Findings

### 1. `no_nav_inputs` is NOT inert with nav off. It is what stops Tab traversal. (blocker)

**The claim.** The spec's Goal, item 1, item 2, the parent bullet and Manual verification step 2 all
rest on one premise: *"with nav off, the flag is inert everywhere, so all four go"*
(`50_wave_e_keyboard.md § Design decisions item 1`, copilot-chat paragraph: "with nav off, the flag
is inert and the outline it was placed to suppress cannot be drawn"). Manual step 2 states the
consequence as a check: *"Then press Tab repeatedly: nothing traverses."*

**The evidence.** imgui's own flag doc, read at the installed package
(`.venv/lib/python3.12/site-packages/imgui_bundle/imgui/__init__.pyi:4519`):

> `nav_enable_keyboard` ... **Note: some features such as basic Tabbing and CtrL+Tab are enabled by
> regardless of this flag** (and may be disabled via other means, see #4828, #9218).

Measured on a headless glfw+imgui rig with `config_flags &= ~nav_enable_keyboard`, two `input_text`s
inside one `begin_child`, one synthetic Tab press
(`scratchpad/tabnav2.py`; `(is_item_active(a), is_item_active(b))` after the press):

```
child window_flags = none            ->  (False, True)     Tab MOVED focus a -> b
child window_flags = no_nav_inputs   ->  (True,  False)    Tab did NOT move focus
```

So the flag is doing live work in the nav-off world, and it is the *only* thing suppressing Tab
traversal. Deleting all five uses does not remove a dead flag; it turns Tab into a focus-traversal
key across the panel's sliders, the grid's `selectable` tiles and the chat. That is new behaviour
nobody asked for, in a wave whose whole premise is that removal is behaviour-neutral.

Note the second-order effect on the same premise: `preview_cell`'s `selectable` (item 1 keeps it,
correctly, for `allow_overlap`) IS a Tab stop, unlike an `invisible_button`. The grid child at
`document_grid.py:45` is exactly where that matters.

The editor child (`ui.py:394`) is the one site where deletion is probably safe: the editor surface is
`imgui.invisible_button("##editor_surface", ...)` (`tabs/code.py:572`), which is not a Tab stop, so
that child has no traversable widget. That is an argument for that one site, not for the class.

**Fix (paste into item 1, replacing the "with nav off the flag is inert" reasoning):** "The five
`no_nav_inputs` sites are NOT deleted on the inert argument, which measurement refutes: with
`nav_enable_keyboard` clear, imgui still runs basic Tab traversal (its own `ConfigFlags` doc says so,
and a headless rig confirms a Tab moves focus between two `input_text`s in a plain child but not in
one flagged `no_nav_inputs`). What the flag stops being is REGION-CONFINEMENT machinery: the
conditional forms (`ui.py:750`'s `panel_flags`, `document_grid.py:42-44`'s `grid_flags`) lose their
`panel_active` / `grid_active` condition and become unconditional `no_nav_inputs`, and the editor
child, the copilot bar and the chat window keep theirs as written. Only the region CONDITION goes;
the flag stays. Manual step 2's Tab clause is rewritten to assert Tab does not traverse *because*
every focusable container carries the flag."

Retire `no_nav_inputs` from `_BANNED` in `tests/test_region_system_is_gone.py` accordingly, and add a
positive test instead: every `begin_child` that hosts a focusable widget carries the flag. Leave
`nav_flatten` / `nav_flattened` banned (those are genuinely nav-only).

### 2. `no_nav_focus` is load-bearing after nav-off, and `commands.py:128` will read false. (blocker-adjacent)

Same doc sentence, same measurement class. `ui.py:65` and `copilot_chat.py:51` carry
`WindowFlags_.no_nav_focus`, and `commands.py:128` records why:

```
# Ctrl+Tab is free for us because WindowFlags_.no_nav_focus on the main window
# (ui.py) suppresses imgui's built-in window-cycle.
```

Per the `ConfigFlags` note, **Ctrl+Tab is enabled regardless of `nav_enable_keyboard`**, so that
comment stays TRUE after the wave and `CYCLE_CODE_TAB` on Ctrl+Tab keeps depending on it. The spec
never mentions `no_nav_focus` — not in the deletion list, not in item 2's "four things read the world
it created", not in the banned set. That is correct by accident (nothing deletes it), but the wave
must say so, because a reader who has just internalised "nav is off, nav flags are dead" is one
sweep away from deleting the flag that keeps Ctrl+Tab working.

**Fix (paste into item 2, as a fifth bullet):** "**`WindowFlags_.no_nav_focus` stays, on both the main
window (`ui.py:65`) and the chat (`copilot_chat.py:51`).** imgui runs Ctrl+Tab regardless of
`nav_enable_keyboard` (its `ConfigFlags` doc says so explicitly), so the flag is what keeps
`CYCLE_CODE_TAB` from fighting imgui's window-cycle, and `commands.py:128`'s comment stays accurate
with no edit. It is named here so nav-off is not read as licence to delete every flag with `nav` in
its name."

### 3. W-E and W-F both write `Editor.set_style` / `Editor.get_style` and a style enum.

W-E item 4 says: "`Editor.set_style` is a new method on the ffi wrapper over the `ed_set_style` W-F
bound" and specifies `class EditorStyle(IntEnum): VIM = 0; STANDARD = 1` plus both methods. Its
opening paragraph (`:8-10`) states W-F "carries no wrapper method for either yet; adding the two
methods is W-E's".

That is refuted by W-F's own spec at HEAD. `40_wave_f_editor_chrome.md § Out of scope` (`:66-69`):

> "W-F binds `ed_set_style` / `ed_style` **and exposes them as `Editor.set_style` / `Editor.get_style`
> so W-E has something to call.**"

and its item at `:371-379` gives the method bodies verbatim plus "`class Style(IntEnum): VIM = 0;
STANDARD = 1`", and its Files-touched list (`:721`) reads "`set_draw_chrome`, `set_style`,
`get_style`; the `Style` enum". Two waves, two enum names (`Style` vs `EditorStyle`), same two
methods, same file. W-F lands first, so W-E's version arrives as a duplicate definition.

**Fix (replace W-E item 4's ffi paragraph):** "`Editor.set_style` / `Editor.get_style` and the
`Style` enum are **W-F's** (`40_wave_f_editor_chrome.md`, its ffi item), landed before this wave.
W-E adds no ffi code: it calls `editor.set_style(Style.VIM if settings.keymap == 'vim' else
Style.STANDARD)`, importing `Style` from `shaderbox.editor.ffi`. Drop `editor/ffi.py` from § Files
touched, drop `tests/test_editor_ffi.py::test_the_style_round_trips` (W-F's mirror/round-trip test
covers the binding; if W-F's does not assert the round trip, that is a W-F finding, not a W-E file)."

Open question 4's phrasing ("An `EditorStyle` enum exists on the ffi side") also needs the rename to
`Style`; its conclusion (key `_RESERVED_CHORDS` by the persisted literal) is unaffected and correct.

### 4. Four deletion sites the list misses.

Grepped at `ccd446b` over `shaderbox/**.py` + `scripts/smoke.py` for the parent's symbol list plus
`region_`, `focus_field`, `set_nav_cursor_visible`.

**(a) `scripts/smoke.py:214-218`** — the "Feature 019" `nav_enable_keyboard` assertion:

```python
# Feature 019: nav_enable_keyboard is set in __init__, before any frame —
# check it here (get_io() reads are frame-context-sensitive mid-loop).
assert (
    imgui.get_io().config_flags & imgui.ConfigFlags_.nav_enable_keyboard
), "nav_enable_keyboard not set"
```

The spec's `scripts/smoke.py` row names only the `ActiveRegion` import half, the `active_region`
assertion (`:147-149`) and the `cycle_region()` drive (`:272`). This assertion goes red on the first
smoke run after `app.py:203` is deleted, which means `make gates` fails before any of the wave's own
tests are reached. Delete it with its comment, or invert it to assert the flag is CLEAR (the stronger
form: it pins the decision instead of just not contradicting it).

**(b) `ui.py:747-748`** — the panel's focus grab:

```python
    if focus_panel:
        imgui.set_next_window_focus()
```

Item 1's `ui.py` row names "the panel's whole region preamble and derive block (`:734-739`, `:750`,
`:757-764`)". `:747-748` sits between `:739` and `:750` and is not in any cited range, yet it reads
`focus_panel`, which `:737` defines from `region_focus_pending`. Leaving it is a `NameError` inside
the frame; the sibling grab in `document_grid.py:40` IS covered by that file's `:38-60` range.

**(c) `ui_models.py:232-233`** — a comment naming a deleted concept and a dead pointer:

```python
# NOT active_region / copilot_focused — those are transient-by-design (focus on
# launch is a separate UX decision; see todo.md feature-019 deferral).
```

`ui_models.py` is not in § Files touched at all beyond the `keymap` field. Two problems: the comment
names `active_region`, so `tests/test_region_system_is_gone.py` goes red on it (good, but the spec
should say so rather than have the wave discover it); and it points at a `todo.md` feature-019
deferral that does not exist — `grep -in "region\|keyboard nav\|keymap\|nav_" ai_docs/todo.md`
returns nothing. Rewrite the comment to name `copilot_focused` alone and drop the todo pointer.

**(d) `exporters/youtube.py:310-312`** — a comment describing the nav world:

```python
# The credential is loaded via a file pick, not typed — so a focus request lands the
# nav outline on the primary Load button (not the hidden paste box). focus_field
# (one-shot) is owned by the caller via the `focus` flag.
```

"lands the nav outline" is false once `focus_field` stops calling `set_nav_cursor_visible` (item 2).
The banned-name grep does not catch it (no banned token appears). Not load-bearing, but it is exactly
the "comment describing a mechanism that no longer runs" the repo's comment rule forbids, and item 2
invokes that rule to justify deleting `config_nav_escape_clear_focus_item`.

Also worth stating explicitly (the spec is silent): `ui_primitives.py:470` and `popups/settings.py:40`
mention `focus_field` but describe its scroll/one-shot behaviour, which survives. No edit needed.

### 5. The `no_nav_inputs` count is five, not four.

The spec says "all FOUR `no_nav_inputs` sites" (§ Verified premises, twice) and names them as `ui.py`
editor child + copilot bar + panel child, and `copilot_chat.py:53`. `git grep -n no_nav_inputs
ccd446b -- 'shaderbox/**.py'` returns six lines, one of which is a comment (`app.py:202`), leaving
**five** flag uses:

```
ui.py:394   editor child
ui.py:568   copilot bar child
ui.py:750   panel child (conditional)
copilot_chat.py:53   chat window
document_grid.py:45  grid child (conditional)   <-- the fifth
```

The grid one is not orphaned in practice — item 1's `document_grid.py` row deletes `:38-60`, which
contains it — but the wave states the count four times as a verified fact and uses it to argue the
parent's "three" was wrong. A census number in a spec is read as established; this one is off by one.
Correct it to five and add `document_grid.py:45` to the § Verified premises row.

### 6. Two named falsifiers do not falsify.

The § Tests section owes each test the bug that makes it red. Two do not.

**(a) `test_the_standard_doc_still_parses`.** The stated falsifier: *"Change `_standard_chords` to
scan whole lines instead of the first table cell and it picks up Ctrl+X / Ctrl+C / Ctrl+V from the
closing paragraph ... which the disjointness test then reports as three phantom clashes. Both were
run."*

Ran it. `_STD_KEY` requires backticks (`` r"`((?:Ctrl|Shift|Alt)..." ``), and the closing paragraph
writes those three chords as **plain text**:

```
Every other key with Ctrl, Alt or Super held -- Ctrl+X, Ctrl+C and Ctrl+V
```

Whole-line scan and first-cell scan return the identical 13-chord set
(`scratchpad/robust.py`: `whole-line extra: []`). The mutation does not go red, so the first-cell
restriction is untested. It is still the right parse (a future doc edit could backtick them), but the
spec must not claim a falsifier it does not have — this is the "checker that quietly narrows its own
domain" failure the section says it exists to prevent, one level up.

Fix: state the real guard instead. "The first-cell restriction has no falsifier against the doc as
vendored, because the closing paragraph's Ctrl+X/C/V are not backticked and the regex requires
backticks. It is kept as defence against a re-vendor that backticks them; what IS falsifiable is the
floor and the sentinel."

**(b) `test_the_vim_doc_still_parses`'s Ctrl+D sentinel rationale.** The spec says *"`Ctrl+D` is the
only chord that appears solely in the `<C-x>` notation inside a checklist item."* Measured, ten
chords are angle-only and six are `CTRL-` only, with zero overlap:

```
CTRL-X notation: H J M N P R
<C-x> notation : B D E F U Y Left Right Home End
only-in-angle: B D E F U Y END HOME LEFT RIGHT     only-in-CTRL: H J M N P R
```

So Ctrl+D is not "the only" one; it is one of ten. The sentinel works, and the stated
mutation (drop the `<C-x>` alternative, 16 -> 6 chords) does go red on both the floor and the
sentinel — I reproduced that. Only the uniqueness claim is wrong. Fix the sentence to "`Ctrl+D` is
one of ten chords carried solely in the `<C-x>` notation; `Ctrl+A` is likewise carried solely by a
standard table row."

### 7. `test_region_system_is_gone.py` will go red on files the deletion list does not touch.

The banned set includes `nav_flatten`, which is a **substring of** `nav_flattened`, and
`config_nav_escape_clear_focus_item`. Beyond finding 4's sites, the grep also hits:

- `ui_models.py:232` (`active_region`, finding 4c);
- `hotkeys.py:62` (`CYCLE_REGION`, inside the Esc comment: "Defocus lives on CYCLE_REGION and the
  mouse, never on Esc"). The spec's `hotkeys.py` row lists `_VIM_RESERVED_CHORDS`, `_handle_vim_chord`
  and the docstring block, and explicitly says `_handle_escape` needs no change; it does not name this
  comment. Rewrite it: defocus now lives on the mouse alone.
- `ui_regions.py:11` (`CYCLE_REGION` in the `ActiveRegion` comment) — covered, the enum goes whole.

Also: `region_` as a bare prefix is unsafe as a banned token and the spec correctly does not ban it
(`get_content_region_avail` appears ~15 times, `scripts/dogfood/judge.py::region_diff`). Worth one
sentence in the test's comment so a later wave does not "tighten" it.

Enumeration answer for the parent's question "how is *any* enumerated": it is a hand-written tuple,
not derived. That is acceptable here (the domain is a closed list of retired names, not an enum the
code still carries), but it means the test cannot catch a region symbol nobody thought to list. The
compile check (pyright) is the complement, and item 1 says so correctly.

### 8. The keymap switch has one unhandled branch: the Ctrl+N completion request.

Item 4 enumerates "what else must switch with the keymap" as four things and closes the `:`-command
one as needing no gate. I verified that closure and agree: `_serve_host_command` drains
`ed_take_host_command`, the standard doc lists no `:` binding and states the fall-through, and
`vim_coverage.md § Ex commands` is the only source of the four host commands. Correct.

But `hotkeys.py:77-83`, inside `_drain_editor_input`, is a fifth keymap-dependent branch the spec
does not name:

```python
if (
    event.text == "n"
    and event.mods == KeyMod.CTRL
    and editor.get_mode() == Mode.INSERT
    and not editor.complete_open()
):
    app.editor_completion_requested = True
```

Its comment calls it "the deliberate completion ask" for the vim keymap. Under standard,
`standard_keymap.md` lists `Ctrl+Space, Ctrl+N` as the keymap's OWN completion ("Opens the completion
popup ... with it open, moves to the next candidate"), so `ed_key` consumes it and opens the popup
itself. The guard `editor.get_mode() == Mode.INSERT` happens to hold under standard (the doc: "Under
this keymap `ed_mode` is always insert"), and `not editor.complete_open()` will be False once the
keymap opened its own popup — so the branch is probably harmless. "Probably harmless" is not the
standard the rest of this spec holds itself to, and the spec's own § Out of scope claims the wave
enumerated everything that switches.

**Fix (add as point 5 under "What else must switch with the keymap"):** "**The insert-mode Ctrl+N
completion ask** (`hotkeys.py:77-83`) is vim-only in intent. Under standard the keymap owns Ctrl+N
and opens its own popup, `ed_mode` is always insert, and `complete_open()` reads True the frame after,
so the branch self-gates on `not editor.complete_open()`. Verified rather than assumed: [state the
probe]. Its comment is rewritten to say the branch is reached only under vim and why, so a reader
does not add a keymap gate that would be dead code."

Related and worth one line: `Ctrl+Shift+N` reaching the editor arrives as `KeyEvent(CHAR,
CTRL|SHIFT, text="N")` (`editor/input.py:114-119`, `_key_char` uppercases under shift). The reserved
sets are lowercase, so `_handle_reserved_chord`'s `ch not in _RESERVED_CHORDS[keymap]` is False for
`"N"` and the chord falls through. Correct as designed; state it, because the audit's `NEW_DOCUMENT`
move depends on it and it is not obvious from the move table.

### 9. F5/F6 never reach the editor. Say so; it is what makes the F-key tier work.

`editor/input.py::translate_key` returns an event only for `_SPECIAL_KEYS` (`:25-39`: escape, enter,
tab, backspace, delete, arrows, home, end, page up/down) or, under `_CHORD_MODS` (ctrl/alt/super,
**not** shift), a `_key_char` letter/digit. F-keys are in neither, so `translate_key` returns `None`
and F5/F6 never enter `app.editor_key_events` at all.

The audit's rule 3 asserts "neither keymap claims either tier" from the doc parse, which is a
statement about the KEYMAPS. The stronger fact — the host never even hands an F-key to `ed_key` — is
what actually guarantees Manual step 7 ("Neither types a character into the buffer"). Add it to
`02_keybindings.md` note 3 or § Where the new chords come from: "F-keys are not in
`editor/input.py::_SPECIAL_KEYS` and carry no `_CHORD_MODS` letter, so `translate_key` returns None
for them and they never reach `ed_key` under either keymap. The F-key tier is disjoint by the host's
translation layer as well as by the docs."

### 10. The Settings combo breaks the Editor section's layout idiom.

`popups/settings.py::_draw_body` (`ccd446b:86-120`) draws the Editor section as: `separator_text
("Editor")`, then three bare `imgui.checkbox` calls with full-text labels, then `imgui.dummy((0.0,
SPACE.SM))`, then three `label_row(...)` + widget pairs (Font size / Tab size / Line spacing).
`label_row` (`ui_primitives.py:1088`) is `row_label` + `set_next_item_width`, i.e. a right-aligned
label column.

Item 4 places the combo as "the FIRST row after `separator_text("Editor")` (before the three
checkboxes)". That puts one `label_row` above three checkboxes that use no label column, so the
section reads label-column / no-column / label-column. The imgui skill § 2 is the authority the
parent's D1 cites and it asks for one alignment idiom per panel.

**Fix:** put the Keymap row at the head of the `label_row` block, immediately after `imgui.dummy((0.0,
SPACE.SM))` and above Font size, and drop the "it frames what follows" rationale (which argues for a
position the layout cannot carry). If the framing argument is load-bearing to the author, the
alternative is to move the whole Editor section to `label_row` form, which is W-B's territory, not
this wave's.

The rest of the control's design is sound: `Keymap` is one word and passes W-B's budget test as
analysed; no `help_marker` is the right call under D1; `_KEYMAPS` at module scope as
`tuple[Literal["vim","standard"], ...]` types the index round-trip correctly.

### 11. Docs: two stale pointers survive the sweep.

**(a) The imgui skill § 9.** The spec's `.claude/skills/imgui-ui/SKILL.md` bullet handles § 8's three
nav bullets correctly (keep them, they are cross-project; add one ShaderBox sentence to § 9) — I
agree with that reasoning and with the placement. But § 9's existing "Two editor-focus flags on App"
bullet says `editor_was_ever_focused` is "cleared only by explicit defocus — Esc / **arrow nav** /
target switch". Arrow nav is what this wave deletes. The spec does not name that line.

Fix: "§ 9's two-editor-focus-flags bullet names 'arrow nav' as a defocus cause; strike it, leaving
Esc / target switch."

**(b) `ui_models.py:233`'s `todo.md` pointer** — finding 4c. The § Docs touched entry says `todo.md`
is "untouched ... checked: no entry names regions, nav, or keymaps". That check is right; what it
missed is that a source comment points AT a todo entry that no longer exists.

`widgets/cheatsheet.py` (`ccd446b:24-40`): confirmed it enumerates `COMMAND_SPECS` and
`app.effective_bindings` with no hard-coded chord and no region scope; `_is_active` switches on
`CommandScope` only. Needs no edit; the spec's row is correct. `help_content.py::_shortcuts_section`
(`:72-88`): same, confirmed by reading. Both correct as listed.

### 12. Minor: `_focus_or_add_tab`'s guard-drop argument is right; one clause is not.

Item 3's argument for dropping `not self.any_popup_open()` from the focus request is sound and I
verified its load-bearing half: `ui.py:388` reads `if app.editor_focus_requested and not
app.any_popup_open(): imgui.set_next_window_focus()`, so the popup guard genuinely survives one layer
down at the site that calls `set_next_window_focus`. Good.

The clause "Setting the latch while a popup is open is harmless and correct: it is what
`reconcile_popup_focus` does deliberately on the popup's close edge (`app.py:752`)" is a weaker
citation than it reads: `reconcile_popup_focus` sets the latch on the CLOSE edge, i.e. when the popup
is no longer open, which is the opposite state from the one being argued about. The conclusion still
holds on the `ui.py:388` evidence alone. Drop the second clause.

---

## What I checked and agree with

**The audit table is correct.** Re-derived independently
(`scratchpad/audit.py`): a parser over both vendored docs using the parse `02_keybindings.md`
describes, crossed with `COMMAND_SPECS` at `ccd446b`.

- Vim: 16 chords, matching the spec's table row for row — `Ctrl+{H,J,M,N,P,R}` from `CTRL-X`
  notation, `Ctrl+{B,D,E,F,U,Y,Left,Right,Home,End}` from `<C-x>`.
- Standard: 13 chords, matching — `Ctrl+{A,Z,Y,N,Space,Left,Right,Home,End,Backspace,Delete}`,
  `Ctrl+Shift+Z`, `Shift+Tab`.
- Collisions over all `COMMAND_SPECS`: exactly eight. Seven GLOBAL (`new_document` Ctrl+N both
  keymaps; `delete_document` Ctrl+D vim; `toggle_document_play` Ctrl+Space standard; `open_shader`
  Ctrl+E vim; `open_script` Ctrl+R vim; `toggle_copilot` Ctrl+J vim; `open_lib_picker` Ctrl+P vim)
  plus one COPILOT-scoped (`cycle_copilot_layout` Ctrl+H vim). **Exactly the spec's seven moves plus
  the one documented exemption.** No chord is moved that does not collide; no collision is left
  unmoved.
- `close_code_tab` Ctrl+W: in neither parsed set, confirming note 4 and the spec's Ctrl+W handling.
- Every remaining Ctrl chord is clean by parse: Ctrl+O, Ctrl+S, Ctrl+Q, Ctrl+Tab, Ctrl+W,
  Ctrl+1/2/3, Ctrl+Shift+P.
- All eight replacement chords (Ctrl+Shift+N, Alt+D, F5, Alt+C, Alt+R, Alt+J, Alt+L, F6) are absent
  from both keymaps AND from `COMMAND_SPECS` today. Recomputing the whole table post-move leaves one
  keymap collision (`cycle_copilot_layout`, the exemption) and **zero duplicate chords**.
- `chord_needs_modifier` returns False and `chord_to_str` renders correctly for every new chord;
  `route_flag` gives `route_always` for the five Alt moves and `route_global` for Ctrl+Shift+N / F5,
  exactly as item 6 claims.
- The table has no blank cell, and its editor-owned section lists every parsed chord no app command
  uses. Cross-checked both directions.

**`_RESERVED_CHORDS`'s two letter edits.** `grep -c CTRL-D vim_coverage.md` = 0 (the doc uses `<C-d>`
only), `CTRL-O` / `<C-o>` / `CTRL-W` / `<C-w>` = 0 matches, `CTRL-M` present at the `+ CTRL-M <CR>`
row. Removing `o`, adding `m`, keeping `w` as a host behaviour: all three justified as written.

**`set_keyboard_focus_here` with nav off.** Re-ran it rather than trusting the drafter
(`scratchpad/navoff.py`, `config_flags &= ~nav_enable_keyboard`): an `input_text` focused on the first
frame reads `is_item_focused()` and `is_item_active()` True from the next frame on, and
`set_nav_cursor_visible(True)` runs without exception. The claim holds; only the frame indexing in the
spec's phrasing is off by one, which is immaterial. The six dependent sites are safe.

**The `SELECT`-hue decision (item 8): agree, keep the assertions.** Read `theme.py:196-215` at
`ccd446b`. The block asserts `COLOR.SELECT` differs from every accent primary and from every `STATE_*`
hue. The stated reason is region-nesting, which expires, but the rule does not: all three surviving
outline sites (`document_grid.py:94` and `popups/examples.py:96` via `preview_cell`'s `Col_.border`,
`exporters/telegram.py:703`) sit inside accent-chromed panels, and imgui's `Col_.border` on a focused
child is accent-adjacent independent of any region stroke. Deleting a live guard because its comment's
example expired would be the wrong trade. The two comment rewrites plus the matching `conventions.md`
parenthetical (`:357`, "its outline nests inside the accent's region outline") are the right scope.

**The five closed open questions.** 1 (no chat focus cue) — agree, #24 names the chat's outline
specifically and `Col_.title_bg_active` carries a signal; the revisit trigger is stated.
2 (`set_style` in the funnel only) — agree, `get_session` calls `_apply_editor_settings_to` at
`app.py:1361`, so one path suffices. 3 (no Alt-uniqueness assertion) — agree,
`test_no_two_specs_share_a_chord_in_overlapping_scopes` is strictly stronger and I confirmed the
post-move table has no duplicate. 4 (key by the `Literal`) — agree, modulo the `EditorStyle` ->
`Style` rename from finding 3. 5 (Ctrl+M consume-noop) — agree.

**The keymap field and its persistence.** `EditorSettings.keymap: Literal["vim","standard"] = "vim"`
is salvage-correct as claimed: `model_salvage.drop_invalid` (`:85-98`) validates each key alone via
`validate_assignment` on a `model_construct()` and pops the failures, so `"emacs"` drops to the field
default with every sibling intact. `EditorSettings` is nested under `UIAppState`, which `load_model`
runs `drop_invalid` over, so no new wiring. `Literal` is imported at `ui_models.py:5`. The falsifier
named for the `str`-instead-of-`Literal` mistake is real.

**The `_apply_editor_settings_to` ordering constraint (anchor 3): honoured.** Item 4 places the
`set_style` line "first, before the five existing calls", and `set_chrome_flag(ChromeFlag.LINE_NUMBERS,
...)` is the second of those five at `app.py:1386`. That satisfies W-F's constraint
(`40_wave_f_editor_chrome.md:71-79`, restated in `02_keybindings.md § One ordering constraint`) that
`ed_set_style` replaces the whole `Chrome` from `chrome_for(style)` and so must precede any
`set_chrome_flag`. Correct as specified.

**The 019 banner + `conventions.md` edits.** Read all three conventions entries. The region entry
(`:343-351`) is correctly deleted whole — its revisit trigger is exactly what this wave fires and it
has no surviving instances. The code-editor entry (`:374-375`) does carry the "vim-modal library"
sentence and its D5 trigger; the rewrite is right and the two surviving triggers are correctly kept.
The `SELECT` entry (`:352-361`) is as described. Keeping the 019 body and adding a banner, rather than
deleting the file, is the right call under `dev_flow.md`'s register-vs-spec distinction.

**Findings 13, 24, 26 coverage.** All three are closed by the wave as specified, with one gap each
noted above and none of them a coverage gap: #13 (keymap in global Settings, not the main UI) — the
Settings combo delivers it, finding 10 is about placement within the section. #24 (remove highlight,
active areas, arrow rotation, "rework properly") — the outline, the regions and `CYCLE_REGION` all
go; finding 1 is that "rework properly" cannot mean deleting the flag that still stops Tab. #26 (the
audit, with the disjointness test reading the docs) — delivered in full, and the test reads the
vendored files rather than a retyped list, which is what row 26 asks for.

## False trails

- **`preview_cell`'s `selectable` staying a `selectable`** — I expected this to be wrong once nav
  is off. It is right: `allow_overlap` plus transparent `Col_.header*` is what lets overlay buttons
  win the click, independent of nav. The spec's reasoning holds.
- **`theme.py:480`'s `col.nav_cursor` mapping** — checked whether keeping a dead style row is
  defensible. It is: a complete mapping of imgui's colour enum has value independent of what draws.
- **Ctrl+Y as an app collision** — refuted, as the spec says. No `CommandSpec` binds it.
- **`_STANDALONE_KEYS` / `_BINDABLE_KEYS` needing an edit for F5/F6** — no. Both already carry f1-f12
  and `chord_needs_modifier` exempts them; verified by running both functions.
- **`test_command_routing.py` needing an edit** — no. It loops `COMMAND_SPECS` pairwise and the
  post-move table has no duplicate chord, which I recomputed.
- **`region_` as a banned grep token** — would false-positive on ~15 `get_content_region_avail`
  calls plus `dogfood/judge.py::region_diff`. The spec correctly does not ban it.
- **`test_persistence_completeness.py`** — read it; a new field on an existing rostered model needs
  no edit, as the spec claims.
- **The `:`-command keymap gate** — checked `hotkeys.py::_serve_host_command` and both keymap docs.
  The spec's "needs no gate" is right.

## Coverage statement

Read at `ccd446b`: `commands.py` (whole), `hotkeys.py` (the drain, `_handle_vim_chord`,
`_serve_host_command`, `spec_eligible`, `_dispatch_registry`, `_VIM_RESERVED_CHORDS` block),
`editor/input.py` (whole translation layer), `app.py` (`__init__` nav lines, the escape filter,
`escape_has_job`, the region methods, `focus_document_tab`, `_focus_or_add_tab`,
`_apply_editor_settings_to`, `get_session`), `ui.py` (`:388-420`, `:560-575`, `:730-770`, the main
window flags), `ui_regions.py`, `ui_primitives.py` (`active_region_outline`, `focus_field`,
`label_row`, `preview_cell`'s nav parts), `widgets/document_grid.py:36-62`,
`widgets/copilot_chat.py:42-58,144-155`, `widgets/cheatsheet.py`, `help_content.py::_shortcuts_section`,
`theme.py:190-220`, `popups/settings.py:75-120`, `scripts/smoke.py:205-225`, `model_salvage.py`,
`ui_models.py:226-240`, `tabs/code.py` (focus mechanics). Working tree: both keymap docs, `VERSION`,
`conventions.md` (three entries), `todo.md`, `.claude/skills/imgui-ui/SKILL.md` §§ 8-9,
`01_spec.md` (D1-D12, § W-E), `40_wave_f_editor_chrome.md` (out-of-scope, ffi item, files touched).

Ran: the audit parser + cross-product + post-move recomputation (`scratchpad/audit.py`); the parse
robustness probes (`scratchpad/robust.py`, `phantom.py`); two headless glfw+imgui rigs — the
`set_keyboard_focus_here` / `set_nav_cursor_visible` nav-off check (`scratchpad/navoff.py`) and the
Tab-traversal check that produced finding 1 (`scratchpad/tabnav.py`, `tabnav2.py`). Grepped the full
retired-symbol list plus `region_`, `focus_field`, `set_nav_cursor_visible`, `no_nav_focus` across
`shaderbox/`, `scripts/` and `tests/` at `ccd446b`.

Not verified: the `ed_set_style` ABI behaviour itself (no probe run against `libeditor.so`; taken
from W-F's measurements, which its own review accepted). The Manual verification steps are visual and
stay with the maintainer, except step 2's Tab clause, which finding 1 shows is wrong on paper.

---

# Round 2 (closure)

Narrow closure pass over the drafter's fold. Spec re-read at 1301 lines (was 984),
`02_keybindings.md` at 340. Verdict per finding, with the new text cited.

## Overall: PARTIAL

Ten of twelve CLOSED. **F1 NOT CLOSED** on the positive test's construction: two of its three
non-vacuity properties are written against facts that do not hold, and the container rule as
specified fails on real code at `ccd446b`. **F6b NOT CLOSED**: the refuted sentence still stands in
§ 7, corrected only in the review-history appendix.

The ruling itself (flags stay, conditions go) is correctly and thoroughly propagated. What did not
land is the test that enforces it, which is the whole point of the ruling.

---

## F1 — `no_nav_inputs` stays: **NOT CLOSED**

**What did land, and it is right.** The ruling propagates everywhere it needed to:

- `:80-93` carries the refutation with the imgui doc citation and the measured table.
- `:101` states the fix exactly: the ternaries "lose their `panel_active` / `grid_active` ternary and
  become unconditional `no_nav_inputs`".
- `:160-163` (`ui.py`), `:173` (grid), `:179-181` (chat), and the § Files touched rows at `:863`,
  `:865`, `:866` all restate flag-stays / condition-goes consistently. No surviving sentence claims
  the flag is inert.
- `:254-256`: `no_nav_inputs` and `nav_enable_keyboard` are out of `_BANNED`, with the reason inline
  and a "do not tighten this tuple" note that also covers the `region_` prefix (finding 7's second
  half).
- § Verified premises `:1060` records the reversal as a REFUTED row against the spec's own first
  draft, which is the right place for it.

**Manual step 2 (`:1000-1008`): CLOSED.** Rewritten to "Arrows do nothing outside the editor, **and
Tab still does not traverse**", with the causal clause spelled out: "**That second half passes
because every focusable container still carries `no_nav_inputs`, not because nav is off** (imgui runs
basic Tabbing regardless)". It names the two surfaces to test (grid, settings panel) and says it
"fails exactly when a flag was swept away". That is the manual counterpart the ruling needs.

**The positive test (`:283-325`): NOT CLOSED.** Three problems, in severity order.

**(a) Property 1's derivation source does not exist.** `:305-307`:

> "`_FOCUSABLE_CALLS` is checked against `imgui.__all__` at test time: every name in the tuple must
> exist in the module, so a typo or an imgui rename fails rather than silently matching nothing."

Measured:

```
>>> from imgui_bundle import imgui
>>> hasattr(imgui, "__all__")
False
```

The property as written raises `AttributeError` at test time rather than validating anything. The
intent is sound and one line away: `hasattr(imgui, name.split(".", 1)[1])` over the tuple gives
exactly the stated guarantee, and I confirmed all eleven names resolve that way (`input_text`,
`input_int`, `input_float`, `drag_int`, `drag_float`, `slider_int`, `slider_float`, `checkbox`,
`combo`, `selectable`, `button` — all `True`).

**Fix:** "`_FOCUSABLE_CALLS` is checked with `hasattr(imgui, <name>)` at test time (imgui-bundle
exports no `__all__`); every name must resolve, so a typo or an imgui rename fails rather than
silently matching nothing."

**(b) Property 2's floor is wrong, and the AST walk it describes finds nine, not five.** `:311-315`
says the container set is "the four top-level `begin_child` calls the deletion list names ... plus
the chat's `begin`", "enumerated by walking the AST of those two modules for `begin_child` / `begin`
calls with a string literal first argument". Property 3 (`:316-320`) then pins `assert len(found) >= 5`.

Ran that walk over `ui.py`, `copilot_chat.py` and `document_grid.py` at `ccd446b` (the three modules
the containers live in — note the described walk names only "those two modules", which cannot reach
`document_grid.py`'s grid child, itself a defect). It returns **nine** containers:

```
ui.py            begin       "ShaderBox - UI"          :347
ui.py            begin_child "code_editor"             :390
ui.py            begin_child "app_panel"               :424
ui.py            begin_child "copilot_bar"             :565
ui.py            begin_child "control_panel"           :710
ui.py            begin_child "document_settings"       :752
copilot_chat.py  begin       "Copilot"                 :122
copilot_chat.py  begin_child "##copilot_history"       :258
document_grid.py begin_child "document_preview_grid"   :47
```

A floor of 5 against an actual 9 is slack of four, which is not a floor pinning this walk; it would
survive losing `document_grid.py` entirely (the module the falsifier at `:322` targets).

**(c) The container rule as stated fails on real code, and nesting is never addressed.** The check at
`:317-318` is "the block must contain at least one `_FOCUSABLE_CALLS` name AND its
`begin_child`/`begin` call must pass `no_nav_inputs`". Two containers in the walk host focusable
widgets and carry no flag, before and after this wave:

- `ui.py:424` `app_panel` — `begin_child("app_panel", size=...)`, no `window_flags`. Its body calls
  `_draw_app_panel`, which is the whole settings panel (sliders, checkboxes, tabs).
- `ui.py:710` `control_panel` — `begin_child("control_panel", size=...)`, no `window_flags`. Its body
  calls `draw_document_preview_grid`.

Neither is in the deletion list, and neither should gain a flag: they are outer wrappers whose
focusable widgets live in *inner* children (`document_settings`, `document_preview_grid`) that DO
carry it. That is the right design and the test as specified calls it a violation.

The spec never uses the word nesting outside the unrelated `SELECT` discussion (`grep -n
"nesting\|nested\|ancestor\|descend"` returns only `:831`, `:832`, `:1072`). So the rule needs the
missing clause.

**Fix (replace properties 2 and 3, and the check sentence):** "The container set is every
`begin_child` / `begin` call with a string-literal first argument in `ui.py`, `widgets/copilot_chat.py`
and `widgets/document_grid.py`, found by AST walk; measured at nine at `ccd446b`, so the floor is
`assert len(found) >= 8`. A container satisfies the rule when it passes `no_nav_inputs` **or an
enclosing container in the same `with` nest does**: `ui.py`'s `app_panel` and `control_panel` are
outer wrappers whose focusable widgets sit in the inner `document_settings` / `document_preview_grid`
children that carry the flag, and flagging the wrappers too would be redundant. The check is
therefore: for each container whose block contains a `_FOCUSABLE_CALLS` name **not already inside a
flagged descendant container**, that container or one of its ancestors passes `no_nav_inputs`."

The falsifier at `:322-325` (delete the flag from `document_grid.py:45`) still works under the
corrected rule and is the right one to keep: the grid child would then have a `selectable` under no
flagged ancestor.

**Can it pass vacuously as written?** Not vacuously — it fails loudly, which is worse for the wave
but better than silence. Property 1 raises `AttributeError`; the container rule reports two
false violations. The floor is the one part that could go vacuous later (slack of four).

## F2 — `no_nav_focus` stays: **CLOSED**

`:366-370` adds it as item 2's bullet, with the `ConfigFlags` reason, both sites (`ui.py:65`,
`copilot_chat.py:51`) and the `commands.py:128` comment quoted as staying true. `:163` and `:180`
repeat it at the file level; § Files touched `:863`/`:866` mark it untouched; § Verified premises
`:1061` records it as "missing from this spec entirely" and now confirmed. Item 2's preamble
(`:327`) is re-counted to "Six things read the world it created ... three need a line changed while
three need nothing", consistent with the two additions.

## F3 — W-F owns `set_style` / `Style`: **CLOSED**

`:8-12` reverses the opening paragraph: W-F "OWNS the whole editor-style surface in `editor/ffi.py`
... and exposes `Editor.set_style` / `Editor.get_style`. ... This spec touches `editor/ffi.py` for
nothing." `:504` is the single call site with W-F's enum name (`Style.VIM` / `Style.STANDARD`), not
`EditorStyle`. `:871` turns `editor/ffi.py` into a no-change row with the ownership stated.
`tests/test_editor_ffi.py` is returned to W-F (`:1292`). § Verified premises `:1084`-`:1085` record
the duplicate and its removal. `grep -n EditorStyle` returns only the history rows describing the
retraction. No surviving definition of `Style`, `set_style` or `get_style` in this spec.

## F4a — smoke assertion inverted, stated as code: **CLOSED**

`:199-218`. Both forms are given as code blocks: the current assertion quoted verbatim at `:203-207`
and the replacement at `:214-218`:

```python
            # 069 W-E: nav is OFF app-wide (D4). Checked here for the same reason the
            # old assertion was: get_io() reads are frame-context-sensitive mid-loop.
            assert not (
                imgui.get_io().config_flags & imgui.ConfigFlags_.nav_enable_keyboard
            ), "nav_enable_keyboard is set; D4 removed app-wide nav"
```

Prose at `:210-212` states the reasoning ("inverted, not deleted ... pins the decision instead of
merely not contradicting it"), and § Files touched `:875` and § Tests `:979`/`:984` carry it. The
requested check — stated as code, not prose — is met.

## F4b — `ui.py:747-748` focus grab: **CLOSED**

`:155` adds it as a named bullet ("**`:747-748`: the `if focus_panel: imgui.set_next_window_focus()`
grab.** It sits between the two ranges the first draft cited"). § Files touched `:863` names it, and
§ Tests `:989` uses it as a smoke falsifier ("dropping the `focus_panel` grab's definition while
`:747` still reads it").

## F4c — `ui_models.py:232-233`: **CLOSED**

`:222-231`. The comment is quoted as code, both defects are named (the banned-name hit and the
`todo.md` pointer at a deferral that does not exist), the `grep` that proves the second is cited,
and the rewrite is specified ("Rewritten to name `copilot_focused` alone, with the dead pointer
dropped").

## F4d — `exporters/youtube.py:310-312`: **CLOSED**

`:233-239` plus the § Files touched row at `:879`. It states why no other gate catches it ("no
banned token appears in it, so only this list catches it") and what survives the edit. It also
closes the loop on `ui_primitives.py:470` / `popups/settings.py:40` as no-edit, named so the sweep is
not re-run over them.

## F5 — the count is five: **CLOSED**

§ Verified premises `:1058` and `:1059` both corrected, the second explicitly ("the count is FIVE,
and one line is wrong") with the full five-line census and `document_grid.py:45` bolded as the one
both the parent and the first draft omitted. `:1059` also states the meta-lesson rather than arguing
the number. `grep -n "FOUR\|four `no_nav_inputs`"` finds no surviving assertion of four.

## F6a — the standard-doc falsifier: **CLOSED**

`:1248-1251` in the review history records that both scans return the identical 13-chord set
"because the paragraph writes them unbackticked and the regex requires backticks", and that the
restriction is kept as defence against a re-vendor that backticks them, "and the spec now says it has
no falsifier against the doc as vendored rather than claiming one". Checked § Tests: the false
falsifier sentence is gone from the standard-doc test's entry.

## F6b — the Ctrl+D sentinel uniqueness claim: **NOT CLOSED**

The review history at `:1252` acknowledges it ("measured, it is one of ten, with zero overlap between
the two notations. The sentinel works"). But the sentence the finding is about still stands unedited
in § Design decisions item 7, `:774-775`:

> "The two sentinel chords are the format canaries: `Ctrl+D` is **the only** chord that appears
> solely in the `<C-x>` notation inside a checklist item, and `Ctrl+A` is the only one that appears
> solely in a standard table row's first cell."

Measured again this round, unchanged: six chords (H J M N P R) are `CTRL-` only, ten (B D E F U Y
Left Right Home End) are `<C-x>` only, zero overlap. The implementer reads item 7, not the history
appendix, so a false uniqueness claim is what reaches the code. The fix is the one sentence the
history already wrote: "`Ctrl+D` is one of ten chords carried solely in the `<C-x>` notation, and
`Ctrl+A` is likewise carried solely by a standard table row's first cell."

(The `Ctrl+A` half is true as written — I checked: its only occurrence in `standard_keymap.md` is
line 52, a table row's first cell.)

## F7 — `hotkeys.py:62`'s `CYCLE_REGION` comment: **CLOSED**

`:241` adds it to item 1's deletion list ("the Esc comment at `:62` reads 'Defocus lives on
CYCLE_REGION...'"), § Files touched `:862` names it as one of two comment edits in the file, and
§ Tests `:896` names it as a token the banned-name grep hits. The `region_` prefix half of the
finding is closed by the `_BANNED` comment at `:256-258`.

## F8 — the Ctrl+N completion branch: **CLOSED, and verified rather than assumed**

`:549-561` adds it as point 5 under "What else must switch with the keymap". It is the one fold that
went further than the finding asked: the finding said "probably harmless" was not the spec's own
standard and left the probe as a bracket to fill; the drafter filled it. `:554-558`:

> "**It needs no keymap gate, verified by probe rather than assumed.** Driving the vendored `.so`
> directly: `ed_set_style(h, 1)` returns True and `ed_style` then reads 1; `ed_mode` reads INSERT
> under standard, as the doc says it always does; and a `ed_key(CHAR, CTRL, "n")` returns
> **consumed=True with `ed_complete_open()` already True**."

That is the right shape (the branch self-gates on its own `not complete_open()` conjunct), and the
comment edit is specified. I did not re-run the probe — no ABI probe was in my round-1 scope and the
same measurement underpins W-F, whose own review accepted it. The Ctrl+Shift+N uppercase note the
finding asked for lands at `:562-563`.

## F9 — F-keys never reach `translate_key`: **CLOSED**

Folded into `02_keybindings.md:210-213`, which is where the finding asked for it (the audit's rule-3
tier argument): "`editor/input.py::translate_key` returns an event only for a key in `_SPECIAL_KEYS`
... F-keys are in neither set, so `translate_key` returns `None` and an F-key never reaches
`ed_key`". `50_wave_e_keyboard.md:1266` carries the history row.

## F10 — the Settings combo's placement: **CLOSED**

`:468-469` moves it: "**head of the `label_row` block**: immediately after the `imgui.dummy((0.0,
SPACE.SM))` that closes the three checkboxes, and above Font size." That is the position the finding
named, and `:1272` records that the "it frames what follows" rationale was dropped with it.
§ Files touched `:870` matches.

## F11 — the two stale doc pointers: **CLOSED**

**(a)** `:1157-1160` adds the § 9 correction, citing the line (`SKILL.md:604`) and the strike ("`arrow
nav` is struck, leaving Esc / target switch"). The § 8-stays / § 9-gains-a-sentence split is
preserved, and the added sentence is now sharper than the first draft's: it says what nav-off does
and does not buy, naming Tab and Ctrl+Tab as still running, which folds F1 and F2 into the skill too.

**(b)** The `todo.md` pointer is closed under F4c at `:228-231`.

## F12 — the `reconcile_popup_focus` clause: **CLOSED**

`:415-418` keeps the `ui.py:388` argument and retires the weak citation in place: "an earlier draft
added that `reconcile_popup_focus` sets the same latch on the popup's CLOSE edge (`app.py:752`),
which is a weaker citation than it reads, since the close edge is the opposite state from the one in
question. The `ui.py:388` evidence carries it alone."

---

## What round 2 checked

Re-read: `50_wave_e_keyboard.md` §§ Goal, Design decisions 1-2 and 4-5 and 7, Files touched, Tests,
Manual verification, Verified premises, Docs touched, Review history; `02_keybindings.md` §§ notes and
the F9 addition. Re-ran: `hasattr(imgui, ...)` over `_FOCUSABLE_CALLS` and the `imgui.__all__` probe;
the AST container walk over `ui.py`, `copilot_chat.py`, `document_grid.py` at `ccd446b`; the vim/std
notation split for the F6b sentinel. Opened at `ccd446b`: `ui.py:344-352,420-430,705-715`,
`copilot_chat.py:255-263` to confirm the two unflagged wrapper containers and the chat's inner child.

Not re-checked (unchanged since round 1, and no fold touched them): the audit table's seven moves,
the parse floors, `_RESERVED_CHORDS`'s letter edits, the `SELECT` decision, the five open questions,
the persistence path. The F8 ABI probe is taken on the drafter's report, as noted.

---

# Round 3

Narrow pass over the rebuilt positive Tab test (`50_wave_e_keyboard.md :283-357`), manual step 2
(`:1055-1065`), the § Tests entry (`:948-968`) and § 7's sentinel sentence (`:820`). Spec at 1395
lines.

## Overall: PASS

Both open findings close. Every load-bearing claim in the rebuild was re-derived here rather than
relayed, and all four reproduce exactly, including the two that report a limit rather than a
success.

## F1 — the positive Tab test: **CLOSED** (`:283-357`, with manual step 2 at `:1055-1065`)

Each of the round-2 defects is fixed, and each fix was measured rather than accepted.

**(a) Property 1, `hasattr` not `__all__` — closed at `:302-306`.** The text now reads "Every name in
`_FOCUSABLE` must satisfy `hasattr(imgui, name)` at test time ... **imgui-bundle exports no
`__all__`** (measured: `hasattr(imgui, "__all__")` is `False`)". Re-ran it: all twelve names in the
tuple resolve under `hasattr`, including the added `input_text_multiline`. The property now does the
job it claimed.

**(b) The floor and the module set — closed at `:308-321`.** Three modules, not two, with the reason
stated ("which cannot reach `document_grid.py`'s grid child, the one the falsifier targets").

The count of eight is right and my round-2 nine was wrong. I re-ran the walk two ways: a plain
`ast.Call` walk returns nine, but the spec's rule is a `with`-statement walk over `with`-items, and
that returns exactly **eight**, matching the spec's table row for row including the `flagged=` column:

```
ui.py                     begin       "ShaderBox - UI"          :347
ui.py                     begin_child "code_editor"             :390
ui.py                     begin_child "app_panel"               :424
ui.py                     begin_child "copilot_bar"             :565
ui.py                     begin_child "control_panel"           :710
ui.py                     begin_child "document_settings"       :752
widgets/copilot_chat.py   begin       "Copilot"                 :122
widgets/document_grid.py  begin_child "document_preview_grid"   :47
```

The ninth in my round-2 count was `copilot_chat.py:258`'s `##copilot_history`, and the walk confirms
it is the sole `begin_child` in the three modules that is not a `with`-item. `:352-357` names it
explicitly as the one the walk cannot see, with the argument for why nothing is lost (it hosts no
widget of its own, its `input_text_multiline` being the sibling input drawn outside it, and it sits
inside the flagged `Copilot` window) and why it is stated anyway ("so the count of eight is not read
as a claim that eight is every container in the three files"). That is the correction to my own
round-2 number, made in the spec before I made it here.

`assert len(found) >= 8` against an actual 8 leaves no slack, which was the round-2 defect.

**(c) The ancestry rule — closed at `:337-349`.** "A container satisfies it when **it or an enclosing
container passes `no_nav_inputs`**", with ancestry from `with`-BODY nesting and the parenthetical
that names the trap ("the body, not the whole statement, or a container is its own ancestor and
everything passes"). The symmetric half is there too: a focusable call inside a flagged descendant is
subtracted from the parent's block before the parent is judged, so a wrapper is not charged for its
child's widgets.

I prototyped the rule as specified and ran it against a simulated post-wave tree (both ternaries
collapsed to unconditional `no_nav_inputs`, which is what item 1 prescribes):

```
containers=8  floor>=8 ok=True   violations=[]
```

`app_panel` and `control_panel` pass as ancestors-of-flagged rather than being reported, which is the
whole reason the clause exists. Property 4's `Name` resolution (`:329-336`) is load-bearing and
correct: `document_settings` via `panel_flags`, `document_preview_grid` via `grid_flags`, `Copilot`
via `flags = _WINDOW_FLAGS | _apply_layout(app)` all pass the flag as a Name, and without resolution
all three read unflagged. Confirmed by inspecting each of the three AST nodes.

**The falsifier fires.** Deleting `no_nav_inputs` from the grid child in the post-wave tree:

```
violations=['widgets/document_grid.py:47 document_preview_grid']
```

Red, naming the container, exactly as `:951-955` claims.

**The stated blind spot is real, and reported honestly.** `:352-355` and § Tests `:964-966` both say
deleting the flag from `copilot_chat.py`'s `_WINDOW_FLAGS` leaves the test **green**, because the
chat window's block names no focusable call directly. I ran that mutation: `violations=[]`. Green, as
stated. A spec that measured its own test's limit and wrote it down is doing the thing the round-2
finding was protecting.

**Manual step 2 covers it — closed at `:1055-1065`.** Extended to three surfaces ("the grid, the
settings panel, and the copilot chat with its input focused") with the chat's role called out:
"**The chat is the one that must be checked by hand**: the positive test cannot see it ... which
makes this step the only cover for that flag." The causal clause from round 2 survives intact.

**Can it pass vacuously?** No. The floor has zero slack against a measured eight; property 1 fails on
a rename; property 4 fails closed rather than open (an unresolvable Name reads unflagged, so the test
errs toward reporting). The one uncovered case is named, bounded to a single flag, and handed to a
manual step that names it.

## F6b — the Ctrl+D sentinel: **CLOSED** (`:820-821`)

The sentence in § Design decisions item 7 now reads:

> "The two sentinel chords are the format canaries: `Ctrl+D` is one of ten chords carried solely in
> the `<C-x>` notation inside a checklist item, and `Ctrl+A` is likewise carried solely by a standard
> table row's first cell."

Matches the measurement (ten `<C-x>`-only, six `CTRL-`-only, zero overlap) and no longer claims
uniqueness. The correction is now in the body the implementer reads, not only in the appendix
(`:1309`, `:1371-1372` keep the history). The `Ctrl+A` half stays true as written.

## What round 3 checked

Read: `50_wave_e_keyboard.md :283-357` (the rebuilt test), `:820-822` (the sentinel), `:948-968`
(§ Tests entry), `:1050-1065` (manual steps 1-2), `:178`, `:913`, `:924` (the `_WINDOW_FLAGS` rows).
Ran: the `with`-walk container census over `ui.py`, `copilot_chat.py`, `document_grid.py` at
`ccd446b` (8 with-items, plus the one bare `begin_child` at `copilot_chat.py:258`); a `hasattr`
check over all twelve `_FOCUSABLE` names; an AST inspection of the three Name-passed flag
expressions; and a prototype of the full rule (ancestry from `with` bodies, symmetric subtraction,
Name resolution through module assignments) against a simulated post-wave tree in three states:
unmutated (green), grid flag deleted (red, naming the grid), chat `_WINDOW_FLAGS` flag deleted
(green, the stated limit).

Not re-checked: everything closed in rounds 1 and 2 and untouched by this fold.
