# W-E post-implementation review: code correctness

Commit under review: `2b43f83` ("069 W-E: one owner per chord, 019's regions out").
Everything below was run against the pristine commit; every probe was restored and
`git diff --quiet` confirmed before the next step.

## Verdict

| Area | Verdict |
|---|---|
| Routing | PASS |
| Reserved chords | PASS |
| Keymap switch | PASS |
| Region removal | **PARTIAL** |
| Escape | **PARTIAL** |
| Persistence | PASS |
| Tests | **PARTIAL** |
| Conventions | **FAIL** |

The behaviour this wave set out to produce is correct: every one of the seven moved
chords routes to exactly one owner in every state I could construct, the keymap switch
preserves everything it must on a live handle, and the salvage is structurally sound.
What fails is the gate and three of the guards: the commit does not pass `make check`,
and three of the five `no_nav_inputs` flags the commit calls load-bearing are not
actually covered by the test written to protect them.

---

## Findings

### 1. The commit is RED on `make gates` — ruff rejects a new test file (FAIL, conventions)

`make gates` exits 2 at the `check` stage on the pristine commit. Two separate problems,
both in files this commit adds:

```
$ uv run ruff check tests/test_region_system_is_gone.py; echo $?
SIM102 Use a single `if` statement instead of nested `if` statements
   --> tests/test_region_system_is_gone.py:112:9
    |
110 |                   if isinstance(target, ast.Name):
111 |                       out[target.id] = node.value
112 | /         elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
113 | |             if node.value is not None:
    | |______________________________________^
114 |                   out[node.target.id] = node.value
1
```

```
$ uv run ruff format --check tests/ shaderbox/
Would reformat: tests/test_keymap_disjoint.py
Would reformat: tests/test_region_system_is_gone.py
2 files would be reformatted, 224 files already formatted
```

SIM102 does not auto-fix, so this is not a "hook rewrote files, re-run" case: the gate
stays red. `make test` (1561 passed) and `make smoke` (OK, 200 frames) are both green
when run directly, so the failure is entirely the lint gate, but `CLAUDE.md` makes the
exit code the judgement and it is 2.

**Fix:** collapse `_module_assignments`'s `elif ... : if node.value is not None:` into a
single `elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:`,
run `uv run ruff format tests/`, and re-run `make gates > /tmp/g.log 2>&1; echo $?` before
declaring the wave done.

### 2. Three of the five `no_nav_inputs` flags are unguarded — the positive test is vacuous on `ui.py` (PARTIAL, region removal + tests)

`test_every_child_hosting_a_focusable_widget_blocks_tab` is the guard the commit message
names against "a later reader deleting the flags on a name match". It cannot do that job
for `ui.py`, because `ui.py` contains no focusable-widget call at all — every widget it
shows is drawn inside a called free function, and the check walks the `with` body
syntactically:

```
$ # focusable calls per module, by the test's own _FOCUSABLE tuple
ui.py                   -> {}
widgets/copilot_chat.py -> {'input_text_multiline': [291]}
widgets/document_grid.py-> {'button': [44], 'checkbox': [50]}
```

Falsified directly: deleting `no_nav_inputs` from all three `ui.py` containers
(`code_editor`, `copilot_bar`, `document_settings`) leaves the suite green.

```
$ # after stripping all three flags from shaderbox/ui.py
$ uv run pytest tests/test_region_system_is_gone.py -q
...                                                                      [100%]
3 passed in 0.10s
```

Restored and `git diff --quiet shaderbox/ui.py` confirmed. The `_MIN_CONTAINERS = 8`
floor does not help: six of the eight containers it counts are `ui.py` ones the check can
never evaluate, so the floor is met by containers that contribute nothing. Only the grid
and chat flags are genuinely pinned — which is why the implementer's own falsifier (the
grid child) passed and gave false confidence about the other four.

**Fix:** make the check follow the call, not the `with` body — for each unflagged
container, resolve the free functions called inside it within the same package and treat
their focusable calls as the container's, or add `tabs/document.py`, `tabs/render.py`,
`tabs/share.py` and `widgets/uniform.py` to `_MODULES` and attribute their widgets to the
container that calls them. Either way, re-run the three-flag deletion above and require it
to turn the suite red before believing the guard.

### 3. Ctrl+N's standard-keymap self-gate does not work the way the comment says (PARTIAL, routing)

`hotkeys.py::_drain_editor_input` gained this claim:

> The closed-popup conjunct is also what self-gates standard, whose own Ctrl+N opens the
> popup on the same frame — no keymap check needed.

Standard's Ctrl+N does **not** open the popup, because `App.get_session` calls
`set_host_completion(True)` on every handle, which suppresses the built-in source in both
keymaps. Measured through the real `.so`:

```
Ctrl+N vim       host_completion=True:  consumed=[4655] complete_open=False requested=True
Ctrl+N vim       host_completion=False: consumed=[4655] complete_open=True  requested=False
Ctrl+N standard  host_completion=True:  consumed=[4655] complete_open=False requested=True
Ctrl+N standard  host_completion=False: consumed=[4655] complete_open=True  requested=False
```

So under the app's real configuration standard behaves exactly like vim: the keymap
consumes and opens nothing, and the drain sets `editor_completion_requested`. The gate that
actually prevents a double-fire is the same one vim uses, `not editor.complete_open()`,
verified with candidates pushed:

```
vim:      popup open after push = True sel=0 -> after Ctrl+N: requested=False open=True sel=1
standard: popup open after push = True sel=0 -> after Ctrl+N: requested=False open=True sel=1
```

No behavioural defect: one offer per press, selection advances, nothing double-fires. The
`mode == Mode.INSERT` conjunct also holds under standard, which reports INSERT permanently
(`mode=1` in every standard probe above). The comment's stated mechanism is simply wrong,
and a future reader who trusts it will reason incorrectly about the guard.

**Fix:** replace the last sentence with the true one — both keymaps consume Ctrl+N and open
nothing because `set_host_completion(True)` is set at session creation, so the closed-popup
conjunct is what stops a re-push while the host's own popup is up.

### 4. Nothing defocuses the editor any more, and three comments still say Esc does (PARTIAL, escape)

After this commit `editor_defocus_requested` is written `True` in exactly one place — the
`App.__init__` initializer. No command, no chord, no handler sets it again:

```
$ grep -rn "editor_defocus_requested" shaderbox/
shaderbox/app.py:382:        self.editor_defocus_requested: bool = True   # initializer
shaderbox/app.py:719:        # Do NOT also set editor_defocus_requested: ...   (comment)
shaderbox/tabs/code.py:600:    if app.editor_defocus_requested:                (consumer)
shaderbox/tabs/code.py:602:        app.editor_defocus_requested = False
```

That is defensible against #24, which asks to keep `editor_focused` and says nothing about
needing a keyboard defocus, and Esc was already the editor's unconditionally under 067. The
mouse remains the way out. But three comments now describe machinery that is gone:

- `app.py:381` — "Cleared ONLY by explicit defocus (Esc, arrow nav)". Esc never clears it;
  arrow nav does not exist. What actually clears `editor_was_ever_focused` is a tab or
  document switch (`app.py:549`, `:1161`, `:1187`).
- `hotkeys.py::_drain_editor_input` — "the defocus direction is closed by dropping the queue
  remainder once Esc decides to defocus". Esc never decides to defocus.
- `.claude/skills/imgui-ui/SKILL.md` §9 — the commit edited this line to "cleared only by
  explicit defocus — Esc / target switch", keeping the stale half and dropping the
  already-correct one.

**Fix:** in all three places say the surviving truth — the editor is defocused by clicking
elsewhere or by the chat taking focus, and `editor_was_ever_focused` is cleared by a tab or
document switch.

### 5. A Help-panel string still tells the user to press Ctrl+P for the library (PARTIAL, routing)

`OPEN_LIB_PICKER` moved Ctrl+P -> Alt+L, but the Help panel's prose was not updated:

```
shaderbox/help_content.py:155:
  "Press `Ctrl+P` to browse the library with a live preview of each function's source, "
```

Ctrl+P is now an unbound chord under standard and vim's scroll-up under vim, so a user
following the Help text gets nothing or a scroll. The generated shortcuts table is correct
(it reads `COMMAND_SPECS` through `chord_to_str`), which is exactly why this slipped:
`tests/test_help_content.py` asserts every bound chord *appears* somewhere in its section's
snippet, never that a stale chord string is absent.

**Fix:** change the string to Alt+L, and — since this is the second hand-typed chord in prose
— have `test_help_content.py` also assert that no `Ctrl+`/`Alt+`/`F<n>` literal appears in
any section body unless it is a current `COMMAND_SPECS` chord.

`shaderbox/app.py:550`'s "Ctrl+N is a GLOBAL chord imgui routes through an active text input"
is stale the same way (it is Ctrl+Shift+N now), but it is an internal comment whose mechanism
is unchanged, so it is cosmetic.

### 6. `test_focused_editor_consumes_and_records_chords` no longer tests what it names (PARTIAL, tests)

`tests/test_editor_ffi.py` still calls Ctrl+R "the ONLY guard against Ctrl+R double-dispatch
(decision 5)", but `OPEN_SCRIPT` is Alt+R now, so no command owns Ctrl+R. The assertion
survives only because the chord is hand-built inside a `next()` genexp whose `CommandId`
filter contributes nothing to the value:

```python
open_script_chord = next(
    int(imgui.Key.r) | int(imgui.Key.mod_ctrl)
    for s in COMMAND_SPECS
    if s.id == CommandId.OPEN_SCRIPT
)
```

Falsified by swapping the filter to `CommandId.QUIT`, whose chord is Ctrl+Q:

```
$ uv run pytest tests/test_editor_ffi.py::test_focused_editor_consumes_and_records_chords -q
.                                                                        [100%]
1 passed in 0.62s
```

Restored, `git diff --quiet` confirmed. The test still proves something real (Ctrl+R redoes
and is recorded in registry space) but its stated subject, the double-dispatch guard, is now
unexercised by any test — the live instance of that guard is Ctrl+W under vim NORMAL.

**Fix:** read the chord from the spec (`SPEC_BY_ID[CommandId.OPEN_SCRIPT].default_chord`) so
the filter is load-bearing, and rename the test to what it now checks; add the
double-dispatch assertion against a chord a command actually owns.

### 7. The `no_nav_focus` rationale for Ctrl+Tab is no longer the reason it works (cosmetic)

`commands.py` and the commit message both say Ctrl+Tab is free "because `WindowFlags_.no_nav_focus`
on the main window suppresses imgui's built-in window-cycle". With `nav_enable_keyboard` off
the flag no longer decides this — the shortcut fires and focus is unchanged in all four
combinations:

```
nav=False no_nav_focus=True : shortcut_fired=[10] focus unchanged
nav=False no_nav_focus=False: shortcut_fired=[10] focus unchanged
nav=True  no_nav_focus=True : shortcut_fired=[10] focus unchanged
nav=True  no_nav_focus=False: shortcut_fired=[10] focus unchanged
```

Behaviour is correct and the flag is harmless; only the stated reason is stale. The
`test_region_system_is_gone.py` docstring is stale in the same way — it says "three of the
eight containers reach their flag through a variable (`panel_flags`, `grid_flags`, the
chat's)", but this commit deleted `panel_flags` and `grid_flags`, leaving exactly one.

---

## What I verified as correct

**Routing, all seven cases.** Traced against the real `libeditor.so` through the actual
`_drain_editor_input`, plus a headless imgui rig for the dispatcher half.

- **(a) Alt+R, editor focused, vim INSERT.** `translate_key` does forward it —
  `KeyEvent(CHAR, mods=ALT, text='r', imgui_chord=16947)` — so the concern is real, but the
  editor leaves it unconsumed and `_handle_reserved_chord` returns False at its first line
  (`event.mods != KeyMod.CTRL`). Measured: `text_intact=True, consumed=[], mode=INSERT`. No
  `r` typed. `route_flag` gives Alt+R `route_always` (8192), and the rig confirms an
  `route_always` Alt chord fires with an active `input_text`. One owner: OPEN_SCRIPT.
- **(b) Ctrl+Shift+N, editor focused, standard.** Editor consumes nothing
  (`consumed=[]`, text intact) — the uppercase `N` fall-through the spec describes. Dispatcher
  eligible, `route_global`, fires in the rig on the press frame. One owner: NEW_DOCUMENT.
- **(c) Ctrl+N, standard, editor focused.** Consumed by the editor
  (`consumed=[4655]`), so `spec_eligible`'s `chord in app.editor_consumed_chords` test
  strikes any command. No command owns Ctrl+N any more anyway. The host's completion ask
  fires exactly once and does not double-fire with the popup open (see finding 3 for the
  mechanism, which is right even though the comment is wrong).
- **(d) Ctrl+W, vim.** NORMAL: `consumed=[]`, text intact, so CLOSE_CODE_TAB (EDITOR-scoped,
  `route_global`) reaches the dispatcher — the carve-out works. INSERT: `consumed=[4664]`
  and `'hello world foo\n'` -> `'hello world \n'`, so the host word-delete runs and
  CLOSE_CODE_TAB is correctly struck (`eligible_to_dispatch=False`). Under standard the
  reserved set is empty, Ctrl+W is unconsumed, and CLOSE_CODE_TAB fires — with no
  word-delete, which is the documented consequence of the empty standard set.
- **(e) Ctrl+D.** Focused under vim with a real `layout()` call: cursor 0 -> 9 on a 60-line
  buffer, `consumed=[4645]`, text intact — a half-page scroll, not a delete. (My first probe
  showed no movement purely because I had not called `layout()`; the editor takes its page
  size from `ed_layout`, not from `app.editor_visible_rows`.) Unfocused, Alt+D fires
  DELETE_DOCUMENT and Ctrl+D is bound to nothing. Under standard Ctrl+D is unconsumed and
  owns nothing, matching the audit's "free" cell.
- **(f) F5 with a text input active.** Fires. `route_global`, and the rig shows it reaching
  `imgui.shortcut` on the press frame with `is_item_active()` True on the input throughout.
- **(g) Ctrl+Tab.** Fires in every configuration tested (see finding 7 for the caveat about
  *why*). No state where two owners fire or none does.

**Reserved chords.** `_RESERVED_CHORDS["vim"] = "dufbeyrwnphjm"`, standard empty. The parser
reads 16 vim chords out of `vim_coverage.md` — `b d e f h j m n p r u y` plus Home/End/Left/Right
— which is exactly the vim set minus `w` plus the arrows, and 13 standard chords matching the
audit table row for row. `grep` confirms `vim_coverage.md` has no `CTRL-W` / `<C-w>` and no
`CTRL-O` / `<C-o>` entry, so dropping `o` is right and keeping `w` host-owned is right: it is a
host behaviour (`_delete_word_back`, 067 D15), not a keymap claim, and dropping it would have
made that function dead. `CTRL-M` is present at `vim_coverage.md:107`, so adding `m` is right.
A chord in neither set returns False at the `ch not in _RESERVED_CHORDS[keymap]` line and falls
through to the registry, which is exactly what the inverted
`test_ctrl_o_reaches_the_app_while_focused` now pins. A keymap value outside the dict would
`KeyError`, but the `Literal` makes that unreachable and the salvage test proves it.

**Keymap switch on a live session.** Probed with the real `.so` through
`_apply_editor_settings_to`'s exact call order, reading chrome back through `ed_chrome_flag`
(which the FFI declares but does not bind as a method):

```
1 vim, LN=False, cursor moved: style=VIM      text_ok=True cursor=(2,2) LINE_NUMBERS=False draw_chrome=... undo=1
2 -> standard, LN=False:       style=STANDARD text_ok=True cursor=(2,2) LINE_NUMBERS=False undo=1
3 -> vim, LN=False:            style=VIM      text_ok=True cursor=(2,2) LINE_NUMBERS=False undo=1
4 vim, LN=True:                style=VIM      text_ok=True cursor=(2,2) LINE_NUMBERS=True  undo=1
5 -> standard, LN=True:        style=STANDARD text_ok=True cursor=(2,2) LINE_NUMBERS=True  undo=1
```

Text, cursor and undo index survive vim -> standard -> vim; `show_line_numbers` is honoured in
both directions. The ordering claim is proven rather than asserted:

```
  set LN=False, read: False
  after set_style(VIM), LN = True     <- the pre-set WAS discarded
```

`set_draw_chrome` is not part of `Chrome` and survives a style switch (`False` stays `False`
across `set_style`), and since `App.get_session` calls `set_draw_chrome(True)` *before*
`_apply_editor_settings_to`, a session created after a switch gets both the current keymap and
a drawn gutter:

```
fresh: draw_chrome= True
after apply(standard): style=STANDARD draw_chrome=True LN=True mode=INSERT
```

`mode` reads INSERT under standard, which is correct (standard is modeless) and is what makes
the drain's `mode == Mode.INSERT` conjunct hold there.

**Region removal.** Five `no_nav_inputs` present (`ui.py` code_editor / copilot_bar /
document_settings, `copilot_chat`, `document_grid`) and two `no_nav_focus` (main window, chat);
`nav_enable_keyboard` is cleared and smoke's inverted assertion pins it. `focus_field` still
focuses without `set_nav_cursor_visible` with nav off and `no_nav_inputs` on the container —
re-probed independently, focus lands on frame 4 for a frame-3 request and holds. `FOCUS_TAB_*`
no longer touches editor focus, so Ctrl+1/2/3 with the editor focused switches the panel tab and
leaves the caret where it is; that is the intended reading of D4 and matches the spec's § 3
verbatim. `_focus_or_add_tab` matches the spec's post-state exactly, including dropping the
`not any_popup_open()` guard, which does survive one layer down at `ui.py:387` where the actual
`set_next_window_focus()` lives. No path yields the editor and strands focus.

**Persistence.** `EditorSettings.keymap: Literal["vim","standard"] = "vim"`. Falsified by
redeclaring it `str`:

```
$ uv run pytest tests/test_model_salvage.py -q
E         + emacs
FAILED tests/test_model_salvage.py::test_an_unknown_keymap_resets_only_the_keymap
```

Restored, `git diff --quiet` confirmed. `test_persistence_completeness.py` needs no edit for a
new field on a rostered model and stays green. The Settings combo round-trips through
`_KEYMAPS.index` / `_KEYMAPS[idx]`; a stale value cannot reach `.index` because the model
rejects it first, so the `ValueError` that shape would otherwise risk is structurally
unreachable.

**Falsifiers re-run.** Three of the implementer's, independently:

1. Reverting DELETE_DOCUMENT to Ctrl+D -> `test_no_global_app_chord_belongs_to_either_keymap`
   red (`assert not ['delete_document on Ctrl+D']`).
2. Deleting `no_nav_inputs` from `document_grid.py` ->
   `test_every_child_hosting_a_focusable_widget_blocks_tab` red
   (`assert not ["widgets/document_grid.py:35 'document_preview_grid'"]`).
3. `keymap: str` -> salvage test red, above.

All three restored with `git diff --quiet` verified before the next ran.

**Parse floor on a doc-format change,** mutated on a copy of the vendored doc, never the
vendored file: changing the checklist marker `- [x]` -> `* [x]` drops the vim parse to 0
chords (floor 14); indenting the standard table's first cell drops it to 0 (floor 12). Both
fire. Normalising vim's two notations to one still parses 16, which is correct — that mutation
loses no chord.

**Conventions.** No new `# noqa` / `# type: ignore` / `# pyright: ignore`, no inline imports, no
new `@staticmethod` / `@classmethod`, no `if TYPE_CHECKING`, no `from __future__`, no new bare
`Any` outside the pre-existing test/model sites. The comments this commit adds state present
facts rather than development history, except where finding 3, 4 and 7 note they state facts
that are no longer true. Docs were updated in the same wave: the 019 spec carries a REMOVED
banner, the `conventions.md` region rule is deleted, the `SELECT`-hue rationale is rewritten
without losing the assertion, and the skill gains a nav-off bullet.

---

## False trails

- Ctrl+D under vim appearing not to move the cursor: my probe had not called `Editor.layout`, so
  the editor had no page height. With layout, cursor 0 -> 9.
- `test_editor_ffi.py` crashing on `glfw.get_video_mode`: the dev box reports monitors
  disconnected; under `xvfb-run` the module passes.
- The two test files appearing to change on disk mid-review: that was `make check`'s ruff-format
  hook rewriting them, which is finding 1, not an external edit.
- `_RESERVED_CHORDS` looking larger than the vim doc's list: the doc also yields four arrow
  chords the host set correctly omits, so the subset relation holds once arrows are excluded,
  which is what the test does by comparing letters only.
- `no_nav_focus` looking load-bearing for Ctrl+Tab: it was under nav-on; the probe shows it is
  not under nav-off (finding 7). The flag is still harmless and should stay.

## Coverage

Read end-to-end: `commands.py`, `hotkeys.py`, `editor/input.py`, the changed regions of
`app.py`, `ui.py`, `ui_models.py`, `ui_primitives.py`, `ui_regions.py`, `theme.py`,
`popups/settings.py`, `popups/examples.py`, `tabs/document.py`, the relevant parts of
`tabs/code.py`, `widgets/copilot_chat.py`, `widgets/document_grid.py`, `exporters/youtube.py`,
`scripts/smoke.py`, and both new test modules plus the three modified ones. Context read first:
`CLAUDE.md`, `conventions.md` (## Code rules and the two editor entries as rewritten),
`imgui-ui` SKILL §8 and §9, the W-E spec's decision sections 1-8, and `02_keybindings.md`'s
table and notes 1-4.

Gates: `make test` 1561 passed / 4 skipped, `make smoke` OK (200 frames, 8 documents), both under
`xvfb-run` with the Mesa overrides. `make gates` exits 2 at `check` — finding 1.

`git status --short` at close shows only this review, the untracked W-H spec, and three sibling
reviews written by other reviewers during this pass:

```
?? ai_docs/features/069_tutorial_walk_findings/80_wave_h_tutorial.md
?? ai_docs/features/069_tutorial_walk_findings/reviews/wave_e_post_arch.md
?? ai_docs/features/069_tutorial_walk_findings/reviews/wave_e_post_code.md
?? ai_docs/features/069_tutorial_walk_findings/reviews/wave_e_post_spec.md
?? ai_docs/features/069_tutorial_walk_findings/reviews/wave_h_pre.md
```

No tracked file differs from `2b43f83`.

---

# Round 2 (closure)

Against `7bd7a1d` (the gate) and `ce337bc` (the rest). W-G is being implemented
concurrently in the working tree, so every file was read with `git show ce337bc:<path>`
and every probe run in a disposable `git archive ce337bc` copy under the scratchpad. No
tracked file in the repo was touched.

## Per-finding verdicts

| # | Finding | Verdict |
|---|---|---|
| 1 | Red `make gates` (ruff SIM102 + unformatted) | **CLOSED** |
| 2 | Three `no_nav_inputs` flags unguarded; test vacuous on `ui.py` | **CLOSED** |
| 3 | Ctrl+N standard self-gate comment states a mechanism that does not run | **CLOSED** |
| 4 | Nothing defocuses the editor; three comments still say Esc does | **CLOSED** |
| 5 | Help prose still says Ctrl+P for the library | **CLOSED** |
| 6 | `test_focused_editor_consumes_and_records_chords` tests nothing it names | **CLOSED** |
| 7 | Stale `no_nav_focus` rationale for Ctrl+Tab; stale test docstring | **CLOSED** |

**Overall: PASS.**

### 1 CLOSED

`uv run ruff check tests/ shaderbox/ scripts/` -> `All checks passed!` (exit 0);
`ruff format --check` -> `235 files already formatted`. Full gate at `ce337bc`, captured
unpiped in the disposable copy:

```
$ MESA_GL_VERSION_OVERRIDE=4.6 MESA_GLSL_VERSION_OVERRIDE=460 xvfb-run -a make gates > /tmp/g3.log 2>&1; echo $?
0
== gates: GREEN -- check passed, test passed, smoke passed ==
```

Smoke **passed**, not skipped. My first run of this reported exit 2 on a B007 in
`probe_reach.py` — my own probe file, which I had left in the copy and then `git add`-ed;
removing it gave the green above. That is a false trail of my own making, recorded so the
number is not misread. The root cause named in `7bd7a1d` (pre-commit sees tracked files
only, so two untracked test modules were invisible to the gate) matches what I measured in
round 1, and `dev_flow.md` now carries the staging step.

### 2 CLOSED — and the narrowed `_FOCUSABLE` is correct

I re-ran the Tab-stop measurement independently rather than accepting the report. Anchor an
`input_text`, focus it, press Tab, ask whether the candidate took focus:

```
widget                        nav OFF   nav OFF+flag     nav ON
input_text                  CANDIDATE         anchor  CANDIDATE
input_text_multiline        CANDIDATE         anchor  CANDIDATE
input_int                   CANDIDATE         anchor  CANDIDATE
input_float                 CANDIDATE         anchor  CANDIDATE
drag_int                    CANDIDATE         anchor  CANDIDATE
drag_float                  CANDIDATE         anchor  CANDIDATE
slider_int                  CANDIDATE         anchor  CANDIDATE
slider_float                CANDIDATE         anchor  CANDIDATE
checkbox                       anchor         anchor  CANDIDATE
combo                          anchor         anchor  CANDIDATE
selectable                     anchor         anchor  CANDIDATE
button                         anchor         anchor  CANDIDATE
```

The narrowing is right: with nav off Tab lands only on the eight text-entry widgets, and
`button` / `checkbox` / `combo` / `selectable` are nav-ON stops that this app does not run.
`no_nav_inputs` blocks Tab for every one of them. Keeping the nav-only four in
`_NAV_ONLY_FOCUSABLE` with the instruction to re-run the measurement before restoring them
is the right shape — the previous round's `_FOCUSABLE` was wider than the truth, which is
what made the grid falsifier look like proof.

This also corrects a claim in my own round-1 report: I wrote that the grid flag was
"genuinely pinned". It is not, and it never was — it was pinned by a widget that is not a
Tab stop. The implementer found this by measuring; I had not.

**Does the walk actually reach the three green subtrees?** Yes, which is the load-bearing
question, because "green" is only trustworthy if the walk gets there and finds nothing. I
ran the shipped walk twice, once with the shipped `_FOCUSABLE` and once widened with the
nav-only four:

```
===== SHIPPED (text-entry only) =====
  ui.py       copilot_bar           flagged=True  reaches: (none)
  ui.py       document_settings     flagged=True  reaches: tabs/document.py::draw -> imgui.input_int
  ui.py       code_editor           flagged=True  reaches: (none)
  copilot_chat Copilot              flagged=True  reaches: _draw_transcript -> imgui.input_text_multiline
  document_grid document_preview_grid flagged=True reaches: (none)

===== WIDENED (incl. the nav-only four) =====
  ui.py       copilot_bar           reaches: ui_primitives.toggle_button
  ui.py       document_settings     reaches: tabs/document.py::draw -> ui_primitives.button
  ui.py       code_editor           reaches: tabs/code.py::draw -> _draw_error_strip -> imgui.selectable
  copilot_chat Copilot              reaches: ui_primitives.unconnected_gate
  document_grid document_preview_grid reaches: ui_primitives.button
```

The walk penetrates all five subtrees — under the widened set it finds a widget in every
one, including the three that read "(none)" under the shipped set. So the three green sites
are green because they contain no *text-entry* widget, not because the walk stops short.
Verified a third way, by grepping the subtrees directly rather than trusting either walk:
`tabs/code.py` contains no text-entry widget at all; `toggle_button` and `preview_cell`
contain none. The docstring's "that is not a hole in the walk, it is the truth about those
containers" is demonstrated, not asserted.

**Five per-site falsifiers, run by me:**

```
baseline:                       .        [100%]
drop document_settings flag:    FAILED
drop chat _WINDOW_FLAGS flag:   FAILED
drop code_editor flag:          .        [100%]
drop copilot_bar flag:          .        [100%]
drop grid flag:                 .        [100%]
restored:                       .        [100%]
```

Exactly as reported. The round-1 defect is gone: the three `ui.py` flags can no longer all
be deleted silently — `document_settings` now turns the suite red, via the `_NODE_TABS`
table hop the walk had to learn to follow. `code_editor`, `copilot_bar` and the grid stay
green and the docstring says so plainly instead of implying coverage it lacks. Calling
those three "defensive, and the test starts guarding them the day an input lands there" is
the honest statement of what the guard covers.

### 3 CLOSED

```
# An insert-mode Ctrl+N is the deliberate completion ask. BOTH keymaps consume
# it and open nothing, because get_session sets host_completion(True) on every
# handle, so this branch needs no keymap check: code.draw makes the offer.
```

That is the mechanism I measured in round 1, stated correctly. The false claim about
standard opening its own popup is gone.

### 4 CLOSED

All three sites rewritten to the surviving truth. `app.py:380` now reads "Start unfocused:
an initial defocus request, consumed by the first draw" with no Esc/arrow-nav list; the
drain says the defocus direction is closed "by the gate below dropping the whole queue on a
frame the editor does not hold focus", which is what the code does; the skill's §9 line
reads "cleared only by explicit defocus, or by a tab or document switch". The behaviour is
unchanged, which is correct — #24 asked to keep `editor_focused` and never asked for a
keyboard defocus.

### 5 CLOSED, with a guard

`help_content.py:155` now reads ``Press `Alt+L` to browse the library``, and the
`lib_picker` docstring says `(Alt+L)`. Better than the point fix, the class is now gated:
`test_no_help_prose_quotes_a_chord_the_table_does_not_bind` parses every backticked chord
out of every section body and requires `COMMAND_SPECS` to bind it. Falsified by me —
restoring `Ctrl+P` turns it red, restoring `Alt+L` turns it green.

### 6 CLOSED

The genexp is gone. The Ctrl+R half keeps only what it proves and says so ("after 069 W-E
no command owns Ctrl+R ... so this half no longer exercises the double-dispatch guard"),
and a new test pins the live instance, insert-mode Ctrl+W, reading its chord from
`SPEC_BY_ID[CommandId.CLOSE_CODE_TAB]`. Falsified both ways by me:

```
baseline:                        .        [100%]
retarget spec lookup -> SAVE:    FAILED
drop w from the reserved set:    FAILED
restored:                        .        [100%]
```

The spec lookup is load-bearing now, so a future chord move breaks the test instead of
passing against a hand-built int.

### 7 CLOSED

```
# Ctrl+Tab is ours: imgui's built-in window-cycle needs nav_enable_keyboard, which is
# off app-wide (069 W-E D4). WindowFlags_.no_nav_focus on the main window and the chat
# keeps it that way if nav is ever turned back on.
```

That matches my measurement (the chord reaches the shortcut in all four nav/flag
combinations) and keeps the flag for the right reason. The test docstring no longer claims
three containers resolve their flag through a variable; the chat is now named as the one
that does.

## False trails, round 2

- My first `make gates` run at `ce337bc` exited 2 on a B007 in `probe_reach.py` — my own
  probe, left in the disposable copy and swept in by `git add -A`. Removing it: exit 0.
- The repo's `git status` showing four modified `shaderbox/scripting/*.py` files: that is
  W-G landing concurrently, none of it mine. Confirmed by `git diff --stat HEAD` naming
  only those four.

## Coverage, round 2

Read at `ce337bc` via `git show`: the rebuilt `tests/test_region_system_is_gone.py` end to
end, the changed hunks of `hotkeys.py`, `commands.py`, `app.py`, `help_content.py`,
`lib_picker/__init__.py`, `tests/test_editor_ffi.py`, `tests/test_help_content.py`, and the
skill §9 line. Probes: the twelve-widget Tab-stop matrix under three configurations, the
reachability walk under two `_FOCUSABLE` sets, direct greps of the three green subtrees,
five per-site flag falsifiers, two falsifiers of the new double-dispatch test, one of the
help-chord guard, and a full `make gates`. Everything ran in
`scratchpad/wt` (a `git archive ce337bc` extract); the repo's tracked files were never
modified, and `git status --short` shows no change of mine.
