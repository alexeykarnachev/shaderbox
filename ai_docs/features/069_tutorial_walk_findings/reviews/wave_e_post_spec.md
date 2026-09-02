# 069 W-E post-implementation spec-fidelity audit

Commit under review: `2b43f83` ("069 W-E: one owner per chord, 019's regions out").
Anchors: `50_wave_e_keyboard.md` (three review rounds, PASS), `02_keybindings.md`,
`01_spec.md § W-E` + D4/D5/D7 + open question 4, `00_findings.md` rows 13/24/26.

## Verdict

| Dimension | Verdict |
|---|---|
| Wave-spec fidelity | **PARTIAL** — every design decision landed; three deviations are real and none is recorded in the spec (F2, F3, F4). |
| Deletion-list completeness | **PASS** — all fifteen `app.py` rows, every named symbol, call site and comment gone; the five `no_nav_inputs` flags survive unconditionally, verified by census. |
| Parent fidelity | **PARTIAL** — every W-E bullet satisfied except the one the wave deliberately reverses (`no_nav_inputs`), whose supersession IS recorded in the wave spec's Review history round 1 finding 1. The parent's own text still reads "all four go" and points nowhere. |
| Findings closure | **PASS** — #13, #24 and #26 each closed by named lines; #24's "Rework properly" is met (no dead flag, no comment describing a retired mechanism). |
| Audit-table fidelity | **PASS** — regenerated independently: zero collisions except the documented `CYCLE_COPILOT_LAYOUT` exemption; all seven moves at the named chords; the editor-owned section matches the parsed sets exactly. |

**Gate state: `make gates` is RED at this commit (exit 2).** See F1. This is the most severe
finding and it is not a spec-fidelity issue but a shipped-state one.

## Coverage table

### Design decisions

| Item | Status | Evidence |
|---|---|---|
| 1. Deletion list for 019 | landed | See the deletion-list section below. |
| 1. `no_nav_inputs` STAYS unconditional at five sites | landed | `shaderbox/ui.py:393`, `:554`, `:731`; `shaderbox/widgets/copilot_chat.py:52`; `shaderbox/widgets/document_grid.py:39`. Both ternaries collapsed. |
| 2. `nav_enable_keyboard` off | landed | Deleted from `App.__init__`; `grep -rn nav_enable_keyboard shaderbox/` returns one comment only (`widgets/copilot_chat.py:46`). |
| 2. `config_nav_escape_clear_focus_item` deleted | landed | Absent from `shaderbox/app.py`; banned by `tests/test_region_system_is_gone.py:36`. |
| 2. Esc filter kept, comment rewritten | landed | `shaderbox/app.py:197-199` — imgui #8059 clause gone, surviving reason stated. |
| 2. `focus_field` loses `set_nav_cursor_visible` + 3 docstring sentences | landed | `shaderbox/ui_primitives.py:480-486`. |
| 2. `theme.py` `nav_cursor` row stays | landed | Still mapped; only two comment sentences changed (`shaderbox/theme.py:202,208`). |
| 2. `no_nav_focus` stays both windows | landed | `shaderbox/ui.py:64`, `shaderbox/widgets/copilot_chat.py:50`. |
| 3. `focus_document_tab` reduced to two lines | landed | `shaderbox/app.py:778-780`, byte-identical to the spec block. |
| 3. `_focus_or_add_tab` reduced | landed | `shaderbox/app.py:1150-1163`; the `not any_popup_open()` guard dropped as argued. |
| 3. Two region tests replaced by two latch tests | landed | `tests/test_pass_editor_wiring.py` — `test_a_summon_does_not_focus_the_editor`, `test_an_explicit_focus_request_focuses_the_editor`; `ActiveRegion` import gone. |
| 4. `EditorSettings.keymap: Literal[...] = "vim"` at top of class | landed | `shaderbox/ui_models.py:204`, above `show_whitespace`. |
| 4. Settings combo at head of the `label_row` block | landed | `shaderbox/popups/settings.py:101-106`, after the `dummy`, above Font size; `_KEYMAPS` at `:51`. |
| 4. `set_style` FIRST in `_apply_editor_settings_to` | landed | `shaderbox/app.py:1302`, before `set_chrome_flag(LINE_NUMBERS, …)` at `:1304`. |
| 4. `:`-command handling ungated | landed | `_serve_host_command` unchanged. |
| 4. Ctrl+N completion-ask comment gains the self-gate clause | landed | `shaderbox/hotkeys.py:70-76`. |
| 5. `_RESERVED_CHORDS` per-keymap, `o` out, `m` in | landed | `shaderbox/hotkeys.py:135-140`. |
| 5. `w` in the vim set | **deviation, unrecorded** | `frozenset("dufbeyrwnphjm")` at `:136` against the spec's literal `frozenset("dufbeyrnphjm")`. See F2. |
| 5. `_handle_vim_chord` → `_handle_reserved_chord`, gated on keymap | landed | `shaderbox/hotkeys.py:172-178`; call site `:66`. |
| 5. Docstring block rewritten | landed | `shaderbox/hotkeys.py:128-134`. |
| 5. Subset assertion | landed with a named exemption | `tests/test_keymap_disjoint.py:137,140-153`. |
| 6. Seven `default_chord` edits + `CYCLE_REGION` deleted | landed | See the moves table below; `CommandId.CYCLE_REGION` and its spec absent from `shaderbox/commands.py`. |
| 7. Disjointness test, parse, canaries, scope pin | landed | `tests/test_keymap_disjoint.py`, all five tests. |
| 8. `SELECT` asserts kept, two sentences changed | landed | `shaderbox/theme.py:202` and the assert message at `:208`; both asserts and every token unchanged. |

### Deletion list, row by row

Every `app.py` row of item 1's fifteen-row table verified absent at HEAD, plus every other named
symbol/call-site/comment. `grep -rn` over `shaderbox/**/*.py` + `scripts/smoke.py` for the fourteen
banned names returns hits only inside `tests/test_region_system_is_gone.py`'s own `_BANNED` tuple.

| Row | Status |
|---|---|
| `_REGION_CYCLE` | gone |
| `ActiveRegion` narrowed out of the `ui_regions` import | gone (`shaderbox/app.py:77`) |
| `config_flags |= nav_enable_keyboard` + comment | gone |
| `config_nav_escape_clear_focus_item` + comment | gone |
| `self.active_region` | gone |
| `self.region_focus_pending` | gone (replaced by a two-line comment on `document_tab_select_pending`) |
| `App.cycle_region` | gone |
| `App.focus_move_in_flight` | gone |
| `App.region_derive_allowed` | gone |
| `App.region_outline_visible` | gone |
| `App._yield_editor_to_region` | gone |
| `App._set_region` | gone |
| `CommandId.CYCLE_REGION: self.cycle_region` callback | gone |
| `_set_region(ActiveRegion.PANEL)` in `focus_document_tab` | gone |
| the `active_region stays transient` comment | gone |
| `ui_regions.ActiveRegion` + docstring rewrite | gone / rewritten |
| `commands.CommandId.CYCLE_REGION` + `CommandSpec` | gone |
| `ui_primitives.active_region_outline` + `_REGION_OUTLINE_THICKNESS` | gone |
| `preview_cell` `nav_flatten` param + docstring para + flag branch | gone |
| `preview_cell` `selectable` comment shortened | done (`shaderbox/ui_primitives.py:981-982`) |
| `ui.py` outline import, `ActiveRegion` import, editor derive block + outline | gone |
| `ui.py` panel preamble, `focus_panel` grab, derive block | gone |
| `document_grid.py` both imports, preamble, derive block, `nav_flatten` param + two uses | gone |
| `copilot_chat.py` outline import + focus-outline draw + comment | gone; `:45-46` block comment rewritten to name Tab |
| `popups/examples.py` `nav_flatten=True` | gone |
| `tabs/document.py` `nav_flattened` + comment | gone |
| `smoke.py` `ActiveRegion` half, `active_region` assertion, `cycle_region()` drive | gone |
| `smoke.py` `nav_enable_keyboard` assertion INVERTED | done (`scripts/smoke.py:210-214`) |
| `ui_models.py:232-233` comment | rewritten, dead `todo.md` pointer dropped |
| `exporters/youtube.py:310-312` nav-outline clause | dropped |
| `hotkeys.py:62` Esc comment drops `CYCLE_REGION` | done |
| `tests/test_pass_editor_wiring.py` | done (item 3) |

### The seven moves

Regenerated independently by importing `tests/test_keymap_disjoint.py`'s helpers and crossing the
parsed sets with `COMMAND_SPECS` at HEAD:

| Command | Table says | Code has | Match |
|---|---|---|---|
| `NEW_DOCUMENT` | Ctrl+Shift+N | Ctrl+Shift+N | yes |
| `DELETE_DOCUMENT` | Alt+D | Alt+D | yes |
| `TOGGLE_DOCUMENT_PLAY` | F5 | F5 | yes |
| `OPEN_SHADER` | Alt+C | Alt+C | yes |
| `OPEN_SCRIPT` | Alt+R | Alt+R | yes |
| `TOGGLE_COPILOT` | Alt+J | Alt+J | yes |
| `OPEN_LIB_PICKER` | Alt+L | Alt+L | yes |
| `CYCLE_REGION` | deleted with the id | absent | yes |

Collision sweep over ALL scopes returns exactly one entry: `cycle_copilot_layout` on Ctrl+H,
`CommandScope.COPILOT`, which is note 1's documented exemption. Zero GLOBAL collisions.

Parse counts: vim 16, standard 13, matching the audit table's stated counts and clearing the
floors of 14 and 12. The audit table's "editor-owned chords no app command uses" section is set-
equal to the parsed union in both directions (no chord in the table that the parse misses, none in
the parse that the table omits).

### Tests

| Test | Status | Falsifier re-run |
|---|---|---|
| `test_no_source_file_mentions_the_region_system` | passes | not re-run (grep is self-evident; census above is the equivalent) |
| `test_every_child_hosting_a_focusable_widget_blocks_tab` | passes | **fires**: deleting `no_nav_inputs` from the grid child gives `widgets/document_grid.py:35 'document_preview_grid'` |
| `test_the_focusable_widget_names_are_real` | passes | n/a |
| `test_the_vim_doc_still_parses` / `..._standard_...` | pass | n/a |
| `test_no_global_app_chord_belongs_to_either_keymap` | passes | **fires**: reverting `TOGGLE_COPILOT` to Ctrl+J turns it red |
| `test_the_only_scoped_chord_a_keymap_owns_is_the_copilot_layout` | passes | **fires**: re-scoping that colliding command to `EDITOR` turns it red |
| `test_every_host_reserved_letter_is_a_vim_chord` | passes | **fires**: re-adding `o` names the letter |
| `test_an_unknown_keymap_resets_only_the_keymap` | passes | n/a |
| `test_a_summon_does_not_focus_the_editor` / `..._explicit_...` | present, not run here (needs a display) | n/a |
| `make smoke` | **skipped** — no display in this environment; a skip is not a pass | n/a |

The four falsifiers I re-ran were run against the real tree (mutate, run, restore), not a copy: a
copied tree keeps the editable-install `.pth` pointing at the original repo, so a mutation there is
invisible to pytest. Worth knowing for anyone repeating this.

### Docs touched

| Doc | Status |
|---|---|
| `019_keyboard_navigation.md` banner | landed, six lines, names what survived |
| `roadmap.md` 019 row → superseded | landed |
| `roadmap.md` Active-context banner rewritten | landed |
| `conventions.md` region entry deleted | landed (grep for "region-confined" returns nothing) |
| `conventions.md` code-editor entry rewritten, trigger removed | landed |
| `conventions.md` SELECT parenthetical corrected | landed |
| `067_custom_editor.md` D3 note, D11 note, D15 note | landed |
| `todo.md` untouched | correct |
| `imgui-ui` skill § 9 addition + "arrow nav" struck | landed |

### Manual verification

Steps 1-10 are maintainer work and none can be executed here (no display, no synthetic input on
this box). Step 2's second half (Tab does not traverse the copilot chat) remains the sole cover for
`copilot_chat.py`'s flag, which the positive test measurably cannot see; the spec says so and the
code has not changed that.

## Findings

### F1. `make gates` is RED at this commit; the two new test files were never formatted or linted

**Claim.** The commit was landed without a green `make gates`. `tests/test_region_system_is_gone.py`
fails `ruff` with `SIM102`, and both new test files fail `ruff format`.

**Evidence.** At HEAD (`2b43f83`), with the working tree otherwise clean:

```
$ make gates > /tmp/g.log 2>&1; echo $?
2
```

```
ruff (legacy alias)......................................................Failed
SIM102 Use a single `if` statement instead of nested `if` statements
   --> tests/test_region_system_is_gone.py:114:9
help: Combine `if` statements using `and`
Found 1 error.
```

```
== gates: FAILED at check (exit 2); test and smoke not run ==
== gates: RED (exit 2) ==
```

The offending code is in the commit itself, not working-tree drift:

```
$ git show 2b43f83:tests/test_region_system_is_gone.py | sed -n '112,116p'
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                out[node.target.id] = node.value
```

Separately, running the gate rewrote both new files through `ruff format` (a reflow of
`_resolve_flags`'s signature in `test_region_system_is_gone.py` and of one assert message in
`test_keymap_disjoint.py`), proving neither file went through the formatter before landing. I
restored both to their committed state after measuring.

This is the repo's own named failure mode: `CLAUDE.md` says to judge the gate by an unpiped exit
code and warns the repo "has twice announced a green gate that was red exactly that way". The
commit message asserts every test was falsified and restored, which is true, but says nothing about
the gate.

**Fix.** Combine the nested `if` at `tests/test_region_system_is_gone.py:112-115` into a single
condition, run `make gates` unpiped, capture `$?`, and amend the commit with whatever `ruff format`
rewrites.

### F2. `w` stays in the vim reserved set against the spec's literal, and the spec was not updated

**Claim.** The implementer kept `w` and added a `_HOST_OWNED` exemption to the subset test. The
decision is **correct** on the merits and the reasoning is sound, but the wave spec still carries
the contradicting literal, so the next reader of `50_wave_e_keyboard.md` sees a set the code does
not have.

**Evidence.** Spec item 5 prescribes `"vim": frozenset("dufbeyrnphjm")` (no `w`) while its own third
bullet three lines later says "**`w` stays**, with its NORMAL-mode carve-out intact". The code has
`frozenset("dufbeyrwnphjm")` (`shaderbox/hotkeys.py:136`) plus
`_HOST_OWNED = frozenset("w")` (`tests/test_keymap_disjoint.py:137`). The spec is unamended: there
is no round-4 entry, no note on item 5, and `§ Review history` ends at round 2.

**On whether keeping `w` is the right reading of D7.** It is. Three independent checks:

1. `grep -inE "CTRL-W|<C-w>"` over both vendored docs returns nothing, exit 1. Ctrl+W is in
   neither keymap's list.
2. D7's rule 1 is "a focused editor owns every chord its ACTIVE keymap lists". Ctrl+W is not
   listed, so rule 1 does not fire and rule 2 gives the chord to the app in every state. That is
   exactly what `02_keybindings.md`'s table says: the `CLOSE_CODE_TAB` row reads app / app / app
   with the verdict "keep, neither keymap lists Ctrl+W; see note 4", and note 4 says in as many
   words that the host's insert-mode word-delete "survives as a host behaviour with its NORMAL-mode
   fall-through intact".
3. `067_custom_editor.md` D15 records the shipped behaviour: "insert-mode protections — Ctrl+W
   deletes the word back (must NOT close the tab; CLOSE_CODE_TAB keeps normal-mode Ctrl+W)". The
   code preserves both halves: `shaderbox/hotkeys.py:184-185` returns False for NORMAL-mode `w`
   (so `CLOSE_CODE_TAB` fires) and `:192-193` calls `_delete_word_back` in insert.

So who owns Ctrl+W in the vim-focused cell? The **app** owns the chord (`CLOSE_CODE_TAB`, and the
audit table says so); the **host** owns a behaviour on it in insert mode only, which the ownership
rule does not govern because the rule is about keymap chords. Dropping `w` would have made
`_delete_word_back` unreachable and silently retired a 067-shipped behaviour, which is what the
commit message argues. The literal in item 5 is the error, not the code.

The `_HOST_OWNED` exemption is also the right shape: it is one named letter with a comment, not a
widened check. Falsified: setting `_HOST_OWNED = frozenset()` turns
`test_every_host_reserved_letter_is_a_vim_chord` red, so the exemption is load-bearing and cannot
absorb a second letter unnoticed.

**Fix.** Correct item 5's literal in `50_wave_e_keyboard.md` to `frozenset("dufbeyrwnphjm")` and add
one line to `§ Review history` recording that the literal contradicted the item's own prose and the
prose won, with `_HOST_OWNED` as the named exemption.

### F3. `tests/test_editor_ffi.py` changed while the spec says "No change", unrecorded

**Claim.** Two edits landed in a file the spec's `§ Files touched` and `§ Tests` both declare
untouched, and neither is recorded.

**Evidence.** `§ Files touched` row: "`tests/test_editor_ffi.py` | No change. The style round-trip
belongs to W-F, which writes the methods (§ Tests)." `§ Tests` repeats it: "W-E adds no test there
and asserts no ctypes signature." The diff makes two changes:

```
+        app_state=SimpleNamespace(editor_settings=SimpleNamespace(keymap="vim")),
```

```
-def test_ctrl_o_is_consumed_noop_while_focused() -> None:
+def test_ctrl_o_reaches_the_app_while_focused() -> None:
```

Both are correct and forced. The stub field is required because `_handle_reserved_chord` now reads
`app.app_state.editor_settings.keymap`; the stub is a `SimpleNamespace`, so pyright cannot see the
gap and the failure only appears at run time. The inverted test is the same shape as smoke's
inverted nav assertion and follows directly from item 5's `o` removal: the old test pinned exactly
the behaviour the audit retires. The commit message explains both. The spec does not.

**Fix.** Change the `tests/test_editor_ffi.py` row in `§ Files touched` to name the stub field and
the inverted `test_ctrl_o_reaches_the_app_while_focused`, and keep the "the style round-trip is
W-F's" sentence, which is still true.

### F4. Two stale docstring counts fixed, one of which is not what the commit message says

**Claim.** The implementer reports "two stale docstring counts fixed". The diff contains one count
fix and one count-bearing sentence deleted with its block.

**Evidence.** The one fix:

```
-        # Five settings flow through here; font_size does NOT — it reaches the
+        # Every editor setting but font_size flows through here; font_size reaches the
```

Correct and forced: `_apply_editor_settings_to` now carries six calls, so "five" was about to be
wrong. The second candidate is `hotkeys.py`'s "The full reserved set: the six scrolls + redo +
jumplist + word/line kills + completion nav + left/down", which is not a fix but a deletion — the
whole docstring block was replaced per item 5, which the spec already prescribes. Grepping the diff
for every numeric word in a changed comment turns up no third count.

This is the least severe finding: the work is right, the count in the prose describing it is off by
one. It matters because the repo's own rule is that a number in a commit message reads as
established.

**Fix.** No code change. If the commit is amended for F1, say "one stale docstring count fixed".

### F5. The parent spec's `no_nav_inputs` bullet is still wrong and points nowhere

**Claim.** `01_spec.md § W-E` still instructs "delete … all FOUR `no_nav_inputs` sites … with nav
off the flag is inert everywhere, so all four go". That is refuted by measurement and reversed by
the wave. The supersession IS recorded in `50_wave_e_keyboard.md § Review history` round 1 finding 1
and in `§ Verified / corrected premises`, which is where the audit brief says it must be. But the
parent's own sentence carries no pointer, so a reader who opens `01_spec.md` alone gets an
instruction that would ship a regression.

**Evidence.** `01_spec.md:229-231` is unchanged by this commit (`git show --stat` lists no
`01_spec.md`). The wave spec's round-1 finding 1 states the reversal in full, with the headless-rig
measurement in both directions. Five flags survive in the code, not zero.

The audit brief accepts either "the parent's text is now pointed at" or "the supersession is
recorded in the wave spec's Review history". The second holds, so this is not a fidelity failure.
It is a durability one: the parent spec is the file a cold reader opens first, and the parent's
other W-E citations were corrected in the wave spec rather than in place, so the same reader has
already been trained that `01_spec.md`'s W-E numbers are unreliable.

**Fix.** Add one clause to `01_spec.md`'s `no_nav_inputs` bullet: the flags stay, superseded by
`50_wave_e_keyboard.md § Design decisions item 1`.

## Findings closure, and whether the complaint recurs

**#13** ("the switch goes in the global Settings — a one-time setting, don't pollute the main UI").
Closed by `shaderbox/ui_models.py:204` (the persisted field), `shaderbox/popups/settings.py:101-106`
(one combo, inside the Settings modal, nowhere in the main UI) and `shaderbox/app.py:1302` (applied
through the existing funnel to every open session). Would not recur: the control is a combo in a
modal and the apply path loops `editor_sessions.values()`.

**#24** ("remove the highlight, the need for active areas, and the arrow-key rotation … Rework
properly"). Closed: `active_region_outline` and its four draw calls are gone, `ActiveRegion` is
gone, `CYCLE_REGION` is gone, `nav_enable_keyboard` is off. **On "Rework properly" specifically —
is anything a patch rather than a rework?** I looked for the two shapes the brief names and found
neither:

- *A flag left conditional on a dead condition.* Both ternaries are gone. `ui.py:731` and
  `document_grid.py:39` pass `window_flags=imgui.WindowFlags_.no_nav_inputs` unconditionally; there
  is no surviving `panel_active` / `grid_active` and no `if` in front of any of the five.
- *A comment describing a mechanism that no longer runs.* Every such comment in the deletion list
  was rewritten rather than left: the Esc filter's #8059 clause (`app.py:197-199`), the chat's
  `no_nav_inputs` explanation (`copilot_chat.py:45-46`), `preview_cell`'s `selectable` line
  (`ui_primitives.py:981-982`), `youtube.py:308-310`, `ui_models.py:232-233`, `hotkeys.py:60-62`,
  and the two `theme.py` sentences. `focus_field`'s docstring lost the three sentences about the
  nav cursor rather than keeping them beside a deleted call.

The one thing that could read as a leftover is `preview_cell`'s `selectable`, kept even though the
nav reason expired. The spec argues it explicitly (swapping to `invisible_button` is a behaviour
change nobody asked for; `allow_overlap` plus the transparent header colours is the surviving
reason) and the comment was shortened to exactly that. That is a stated decision, not a patch.

**#26** ("review ALL keybindings so nothing conflicts … Clean and conflict-less"). Closed by the
audit table plus `tests/test_keymap_disjoint.py`. Would not recur, and this is the strongest part
of the wave: the test reads the vendored docs rather than a retyped list, so a re-vendor that grows
a keymap turns the gate red; the scope-exemption test closes the dodge; the floors and the two
sentinel chords close the silent-narrowing failure. All three falsifiers fire.

## False trails

- The `w` deviation looked at first like a spec violation; reading `vim_coverage.md`, D7's rule and 067 D15 shows the code is right and the spec's literal is the defect.
- Four falsifiers appeared not to fire when run against a copied tree; the copy's editable-install `.pth` still points at the original repo, so pytest was loading unmutated modules. Re-run in place, all four fire.
- `pytest tests/` segfaults in this environment at `glfw.get_video_mode`; that is the missing display, not a code defect. `make smoke` fails the same way and reports as unrunnable, not as a pass.
- The commit message carries the `Co-Authored-By` / `Claude-Session` trailer block twice. Cosmetic, no bearing on fidelity.
- `067_custom_editor.md` D15 still states `_VIM_RESERVED_CHORDS = d u f b e y r o w n p h j` in its body; the 069 note beneath it corrects the name, the `o`/`m` change and the keying. The spec's own § Docs touched prescribes exactly that shape ("neither rewrites the decision, which is the record of what 067 shipped"), so it is intended.

## Coverage statement

Read end to end: the full diff of `2b43f83` (all 25 files), `50_wave_e_keyboard.md` (1396 lines),
`02_keybindings.md` (341 lines), `01_spec.md § W-E` + D4/D5/D7 + open question 4,
`00_findings.md` rows 13/24/26 verbatim, both new test modules, and the post-commit state of every
file named in `§ Files touched`.

Executed rather than read: `make gates` (RED, exit 2, unpiped); `pytest` on
`test_keymap_disjoint.py`, `test_region_system_is_gone.py`, `test_model_salvage.py`,
`test_command_routing.py`, `test_editor_ffi.py` (87 passed); an independent regeneration of the
collision check importing the test's own helpers and crossing with `COMMAND_SPECS`; a set-equality
check of the audit table's editor-owned section against the parsed sets; four falsifier mutations
run in place and restored (`no_nav_inputs` off the grid child, `o` back in the reserved set, Alt+J
reverted to Ctrl+J, the colliding command re-scoped to `EDITOR`); `grep` censuses for the fourteen
banned names, the five `no_nav_inputs` sites, `no_nav_focus`, `nav_enable_keyboard`, and `CTRL-W` /
`<C-w>` across both vendored docs.

Not executed: `make smoke` (no display; reports as unrunnable), `tests/test_pass_editor_wiring.py`
and the full `pytest tests/` (same), and all ten manual-verification steps. The working tree was
left as I found it, and the untracked W-H spec file was ignored per the brief.
