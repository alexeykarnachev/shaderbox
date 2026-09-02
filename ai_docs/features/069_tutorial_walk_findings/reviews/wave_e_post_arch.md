# W-E post-implementation review — architecture & conventions

Commit under review: `2b43f83` ("069 W-E: one owner per chord, 019's regions out").
Reviewer role: `dev_flow.md` step 6 (module boundaries, where things live, duplication, dead
code, imgui-interop patterns, docs that should have moved).

## Verdict

| Area | Verdict |
|---|---|
| Dead code after the removal | **PARTIAL** — no orphaned code or App members, but five stale prose survivors (findings 2, 4, 5, 7) |
| Homes | **PARTIAL** — `_RESERVED_CHORDS` is right; the keymap literal has three homes (finding 3) |
| Chord moves | **PASS** |
| Keymap setting | **PASS** |
| Docs | **PARTIAL** — `dev_flow.md`'s module map and the imgui skill still describe the removed layer (findings 2, 5) |
| Conventions | **FAIL** — the commit does not pass `make gates` (finding 1) |

---

## Findings

### 1. FAIL — the commit is RED at `make gates`; both new test modules violate ruff

`make gates` stops at `check` with exit 2 on this commit's tree. Reproduced against a
pristine `git archive 2b43f83` extract, so it is not working-tree contamination:

```
$ uv run ruff check <pristine>/tests/test_region_system_is_gone.py
SIM102 Use a single `if` statement instead of nested `if` statements
   --> tests/test_region_system_is_gone.py:112:9
Found 1 error.        # exit 1

$ uv run ruff format --check <pristine>/tests/test_keymap_disjoint.py <pristine>/tests/test_region_system_is_gone.py
2 files would be reformatted        # exit 1
```

The two format sites are `_resolve_flags`'s signature (one char over the line budget) and the
`test_the_vim_doc_still_parses` assert message. The SIM102 site is `_module_assignments`'s
`elif isinstance(node, ast.AnnAssign) …:` / `if node.value is not None:` pair.

The rest of the gate is otherwise healthy: pyright reports 0 errors, and the 548 tests in
`test_keymap_disjoint` / `test_region_system_is_gone` / `test_command_routing` /
`test_ui_prose_budget` / `test_model_salvage` all pass. So this is purely the lint hook,
which is exactly the class `CLAUDE.md` calls out ("judge it by that exit code, captured
unpiped") — the commit shipped with the gate unrun or its exit code misread.

**Fix:** collapse the `elif`/`if` in `_module_assignments` into one `elif … and node.value is
not None:`, run `uv run ruff format tests/test_keymap_disjoint.py
tests/test_region_system_is_gone.py`, and commit the result.

### 2. HIGH — `dev_flow.md`'s module map still names `ActiveRegion`, App's "nav", and a vim-only editor

Three entries in the map describe the pre-W-E world. Grepping the map against the commit:

```
ai_docs/dev_flow.md:343:  … / **`ui_regions.py`** (`ActiveRegion` +
ai_docs/dev_flow.md:344:  `DocumentTab` — the nav/tab enums, kept out of `commands.py` so the persisted model layer doesn't
```

`ActiveRegion` no longer exists (`shaderbox/ui_regions.py` defines `DocumentTab` alone), so
the map names a symbol a reader cannot find. Two more in the same map:

- `app.py` (line 249): "the UI/glfw/imgui owner + lifecycle wrapper (windowing, GL context,
  editor sessions, popup-state, **nav**, exporter panels)" — App owns no nav after this
  commit; `nav_enable_keyboard`, `active_region`, `region_focus_pending`, `_set_region`,
  `cycle_region`, `region_derive_allowed`, `region_outline_visible`, `focus_move_in_flight`
  and `_yield_editor_to_region` are all gone.
- `editor/` (line 270): "the embedded **vim-modal** code editor (feature 067)" — the editor is
  now vim-modal *or* standard per `EditorSettings.keymap`, which is precisely the change
  `conventions.md` made in its own inline-editor bullet ("the maintainer's own library,
  vim-modal or standard per `EditorSettings.keymap` (features 067, 069)"). The two docs now
  disagree.

`commands.py` and `hotkeys.py` entries are still accurate: `commands.py` is described as the
registry (leaf, imgui-only) and `hotkeys.py` as the dispatch half plus the editor drain, which
is where `_RESERVED_CHORDS` sits.

**Fix:** in `dev_flow.md`, change the `ui_regions.py` entry to `**`ui_regions.py`**
(`DocumentTab` — the settings-panel tab enum, kept out of `commands.py` so the persisted model
layer doesn't pull in imgui)`, drop `nav` from the `app.py` entry's parenthetical, and change
`editor/`'s "vim-modal code editor" to "keymap-selectable code editor (vim or standard per
`EditorSettings.keymap`)".

### 3. MEDIUM — the keymap vocabulary has three homes; the repo's own `get_args` idiom exists for exactly this

`_KEYMAPS` in `popups/settings.py` restates the `Literal` that `ui_models.py` owns:

```python
# shaderbox/ui_models.py
keymap: Literal["vim", "standard"] = "vim"

# shaderbox/popups/settings.py
_KEYMAPS: tuple[Literal["vim", "standard"], ...] = ("vim", "standard")
```

The repo already solved this twice, and the closest precedent is a named alias plus a derived
collection:

```
shaderbox/copilot/state.py:49: ResultWidgetKind = Literal["open_url", "open_path"]
shaderbox/copilot/state.py:50: RESULT_WIDGET_KINDS: frozenset[str] = frozenset(get_args(ResultWidgetKind))
```

and `ui_models.py` itself already names two such aliases (`UIUniformInputType`,
`UniformSortKey`) and reads them back with `get_args` in its own salvage hooks
(`ui_models.py:176`, `:187`). So the inline `Literal` on the field is the outlier here, and
`_KEYMAPS` is a second home that a future third member (the doc's own `F6` reference implies
the editor already supports switching) would have to be added to by hand.

`_RESERVED_CHORDS`'s `dict[str, frozenset[str]]` keys are a third restatement, though a
benign one: I confirmed `_RESERVED_CHORDS["emacs"]` raises `KeyError`, and I confirmed
pydantic rejects `EditorSettings(keymap="emacs")` so the salvage layer resets it to `"vim"`
before `hotkeys.py` ever reads it. The KeyError is structurally unreachable, so the dict is
acceptable as-is; it is only the *vocabulary* that has no single home.

**Fix:** name the alias in `ui_models.py` (`EditorKeymap = Literal["vim", "standard"]`, field
`keymap: EditorKeymap = "vim"`), then in `settings.py` write `_KEYMAPS: tuple[EditorKeymap,
...] = get_args(EditorKeymap)` so the combo's option list cannot drift from the persisted
domain.

### 4. MEDIUM — two comments still narrate an outline that no longer draws, and one names deleted variables

`active_region_outline` is deleted, but three comments still hang off it:

```
shaderbox/ui.py:385:            # after the outline read. Gated on no-popup — a background focus grab
shaderbox/tabs/code.py:495:        # clear the latch here so the outline saw it this frame.
```

Both explain *why* `editor_focus_requested` is cleared where it is, in terms of an outline
read that no longer happens. A reader chasing "which outline?" finds none. The ordering
constraint itself is still real (ui.py consumes the imgui half before the child, code.py
clears the latch), so the comments need rewording, not deleting.

The third is inside the new test:

```
tests/test_region_system_is_gone.py:89:    module's own assignments — three of the eight containers reach their flag through a
                                            variable (`panel_flags`, `grid_flags`, the chat's …)
```

`panel_flags` and `grid_flags` were deleted by this very commit (`git grep panel_flags
shaderbox/` returns nothing; both collapsed to inline `imgui.WindowFlags_.no_nav_inputs`).
Instrumenting the test's own helpers against the shipped tree, only **two** of eight
containers reach their flag through a `Name` (`ui.py`'s `ShaderBox - UI` via
`_MAIN_WINDOW_FLAGS`, and `copilot_chat.py`'s `Copilot`), and only **one** of those is
flagged. So the docstring's "three of the eight" is wrong in the commit that wrote it, and
two of its three named examples are symbols the same commit removed.

**Fix:** reword `ui.py:385` to "…after ui.py consumed the imgui half" and `code.py:495` to
"…clear the latch here, after ui.py's `set_next_window_focus` consumed it this frame"; in the
test docstring, replace the count and the example list with "the chat's `flags =
_WINDOW_FLAGS | …` reaches its flag through a variable, so a check reading only inline text
scores correct code as unflagged".

### 5. MEDIUM — the imgui skill's §8 still cites `active_region_outline` as a live sibling

The commit added the correct §9 bullet ("ShaderBox runs with `nav_enable_keyboard` OFF"), and
its claim that "§8's two nav bullets are written against the ON case" is accurate — I counted
exactly two (`invisible_button` is not a nav stop; `set_keyboard_focus_here` leaves a nav
cursor). But §8's modal-dismissal bullet was not swept:

```
.claude/skills/imgui-ui/SKILL.md:536:  `active_region_outline` draw in the same window already carried this `any_popup_open()` guard;
.claude/skills/imgui-ui/SKILL.md:537:  the focus grab just hadn't.
```

The bullet's *lesson* (gate a per-frame focus grab on `not any_popup_open()`) is intact and
still load-bearing; only the corroborating sibling it points at is gone, so a reader who
greps for it to check the pattern finds nothing.

**Fix:** replace the two sentences with "The same window's own foreground draw already carried
this `any_popup_open()` guard; the focus grab had not." — the lesson survives with no dangling
symbol.

### 6. MEDIUM — the positive `no_nav_inputs` check does not see widgets drawn through `ui_primitives` wrappers

`test_every_child_hosting_a_focusable_widget_blocks_tab` matches only `imgui.<name>` attribute
calls (`_FOCUSABLE` against `func.attr`), so a focusable widget reached through a
`ui_primitives` wrapper is invisible to it. Driving the test's own helpers over the shipped
tree:

```
'ShaderBox - UI' line=346 flagged=False covered=False
'app_panel'      line=410 flagged=False covered=False
'control_panel'  line=696 flagged=False covered=False
'code_editor'    line=389 flagged=True  covered=True
'copilot_bar'    line=551 flagged=True  covered=True
'document_settings' line=728 flagged=True covered=True
```

`app_panel` is unflagged and uncovered, and `_draw_app_panel` calls `fps_overlay(...)` inside
it, whose first act is `chip_button(label, pill_w, pill_h, faded=True)` — a real
`imgui.button` behind a wrapper. So a genuinely focusable control sits in an unflagged
container and the check reports clean. This is the "checker that quietly narrows its own
domain" shape: the assertion reads as "every container hosting a focusable widget", while what
it enforces is "every container that inline-calls a bare `imgui.*` focusable".

The `_MIN_CONTAINERS = 8` floor does defend the container walk, and
`test_the_focusable_widget_names_are_real` defends the `_FOCUSABLE` spellings, so both
existing canaries are good; the uncovered axis is the wrapper indirection specifically.

**Fix:** extend `_FOCUSABLE` with the `ui_primitives` button/input tiers by reflection the way
`test_ui_prose_budget.py` derives its domain (`_BUTTON_TIERS` there is the existing list), or
narrow the docstring and assertion message to say the check covers inline `imgui.*` calls only
and name `app_panel` as the known wrapper-only container.

### 7. LOW — three stale user-facing and prose chord references survived the seven moves

`OPEN_LIB_PICKER` moved `Ctrl+P` → `Alt+L`, but the Help panel still tells the user the old
chord, in a section a tutorial reader lands on:

```
shaderbox/help_content.py:155:  "Press `Ctrl+P` to browse the library with a live preview of each function's source, "
```

This is the one user-visible defect in the set. The Help *shortcuts table* regenerates
correctly — `help_content.py:75-82` reads `COMMAND_SPECS` and calls `chord_to_str`, and
`widgets/cheatsheet.py:33-46` reads `app.effective_bindings` + `chord_to_str`, so neither
needed an edit — but this hand-written body paragraph is outside that derivation.

Two prose siblings:

```
shaderbox/popups/lib_picker/__init__.py:1: """Shader-library picker (Ctrl+P).
scripts/smoke.py:220:  # Exercise the region-cycle + tab-jump wiring (a callback throw surfaces
```

The smoke comment names a cycle the same commit deleted from the loop below it (`if frame_idx
== 50: app.cycle_region()` is gone; only the `frame_idx == 60` tab jump remains).

**Fix:** change `help_content.py:155` to "Press `Alt+L`", change the lib-picker module
docstring's parenthetical to `(Alt+L)`, and change the smoke comment to "Exercise the tab-jump
wiring".

### 8. LOW — `ui_regions.py`'s name outlives the concept it was named for

The module now holds one enum and its docstring was correctly rewritten ("The settings-panel
tab enum…"). Judged against the map's stated reason for the file — keep the persisted enum out
of `commands.py`, which builds `K = imgui.Key` at module scope and so really does load imgui —
the file still earns its existence: `ui_models.py`, `ui.py`, `app.py` and `scripts/smoke.py`
all import `DocumentTab` from it, and `ui_models.py` is the headless model layer the split
protects. So keep the module; only the filename now names a concept the repo no longer has.

**Fix (optional, one wave's worth of churn):** rename to `ui_tabs.py` and update the four
importers plus the `dev_flow.md` entry from finding 2. If the rename is not worth it, leave
it — the docstring already carries the correct meaning, and the four-site blast radius is the
only cost either way.

---

## Verified clean (no finding)

- **Chord moves as `CommandSpec` edits.** All seven are `default_chord` swaps with `scope`
  untouched; `CYCLE_REGION` is deleted from both `CommandId` and `COMMAND_SPECS` rather than
  reserved. `route_flag` needed no edit: the five Alt moves get `route_always` from the
  existing `if chord & int(imgui.Key.mod_alt)` branch, and `Ctrl+Shift+N` / `F5` stay
  `route_global`, which `tests/test_command_routing.py` already pins in both directions.
  `_BINDABLE_KEYS` already carries `f1`–`f12`, and `_STANDALONE_KEYS` already exempts them
  from `chord_needs_modifier`, so `F5` on `TOGGLE_DOCUMENT_PLAY` is bindable and rebindable
  with no table edit.
- **Disjointness, checked independently of the test.** Parsing both vendored docs and scoring
  every spec: vim yields 16 chords, standard 13, and every `GLOBAL` spec is disjoint from the
  union. The single overlap is `CYCLE_COPILOT_LAYOUT` on `Ctrl+H`, which is `COPILOT`-scoped
  and is the one exemption `test_the_only_scoped_chord_a_keymap_owns_is_the_copilot_layout`
  pins by name.
- **The test's chord parser does not restate an existing helper.** `commands.py` has
  `chord_to_str` (int → display string, one-way, documented "display ONLY") and no
  string → int parser; `capture_chord` reads live imgui key state, not text. So
  `_to_chord` / `_MODS` / `_NAMED` in the test have nothing to reuse. It is also the only
  parser of the vendored docs in the repo (`grep -rn "vim_coverage\|standard_keymap"` over
  `shaderbox`, `tests`, `scripts` hits only this module and two comments).
- **`_RESERVED_CHORDS`'s home.** `hotkeys.py` imports `App` and owns the editor drain, so it
  is the dispatch half the map describes; `commands.py` is the leaf registry that never sees
  `App` or `app_state`. A keymap-keyed table that reads
  `app.app_state.editor_settings.keymap` cannot live in the leaf without breaking that.
- **`_apply_editor_settings_to` is the single funnel, and its ordering claim holds.** Two
  callers (`get_session` on lazy-create, `apply_editor_settings` over every open session), two
  triggers (the Settings close funnel and `_handle_escape`'s `was_settings_open` tail). No
  second path sets style. I probed the ordering hazard the comment names: `get_session` calls
  `set_draw_chrome(True)` *before* `_apply_editor_settings_to` runs `set_style`, which would
  be the exact bug the comment warns about — but driving the real `.so`, `ed_draw_chrome`
  reads `True` after both `set_style(STANDARD)` and `set_style(VIM)`, so `draw_chrome` is not
  part of the Chrome that `set_style` replaces and the ordering is safe. `set_chrome_flag` is
  called exactly once in the package, inside the funnel after `set_style`.
- **The keymap setting's placement.** `EditorSettings.keymap` sits with the other editor
  settings, ahead of the numeric block whose comment explains the model-side bounds, and it is
  a `Literal` rather than a bare `str` — so `model_salvage.drop_invalid` costs a hand-edited
  `"emacs"` that one key, which `test_an_unknown_keymap_resets_only_the_keymap` pins on both
  sides (keymap resets, `font_size`/`tab_size`/`current_document_id` survive). The Settings
  combo is the first row of the `label_row` block, gets its width from `label_row`'s
  `set_next_item_width`, and "Keymap" is one word, inside D1's 1-2 word label budget
  (`test_ui_prose_budget` passes).
- **The `_apply_editor_settings_to` docstring count fix.** "Five settings flow through here"
  became "Every editor setting but font_size" — a condition rather than a count, which is what
  `conventions.md`'s "stale counts rot faster than stale prose" bullet asks for.
- **No orphaned `App` state.** Walking every `self.<attr>` assignment in `App` against every
  attribute load across `shaderbox/` and `scripts/smoke.py`, the only never-read attribute is
  `palette_ctx`, which predates this commit (see false trails). The nine region members are
  gone with no residue; `document_tab_select_pending` survives and is read at `ui.py:723`;
  `editor_focus_requested` / `editor_defocus_requested` / `editor_was_ever_focused` all keep
  live readers.
- **The `no_nav_inputs` ternaries did collapse.** `document_settings` and
  `document_preview_grid` both take the flag unconditionally; the two `imgui.WindowFlags_.none
  if <active> else …` expressions are gone, and `nav_flattened` is gone from both
  `tabs/document.py` and `preview_cell`'s deleted `nav_flatten` parameter.
- **`conventions.md`'s three edits.** The 019 nav bullet is deleted whole; the `theme.py`
  SELECT bullet and the inline-editor bullet read as the now. Grepping the file for `region`,
  `CYCLE_REGION`, `no_nav_inputs`, `nav_enable`, `nav_flat` and `019` returns **zero** hits —
  no dangling pointer, no history narration, no changelog residue.
- **The 019 banner and roadmap row.** The banner names 069 W-E and D4, states what went
  (`ActiveRegion`, the cycle chord, the outline, app-wide nav) and what survived; I verified
  each survival claim against the code — `DocumentTab` exists, `focus_document_tab` now does
  exactly "select the panel tab and nothing else" (two assignments, no region call),
  `editor_focused` / `editor_was_ever_focused` are live, and `no_nav_focus` sits on the main
  window (`ui.py:64`) and the chat (`copilot_chat.py:50`) as claimed. The roadmap row flipped
  to `superseded`, which is a real value in the file's own status vocabulary, and its brief
  points at 069 W-E as that vocabulary requires. The Active-context banner moved W-E into
  Landed and names W-G next.
- **`067_custom_editor.md`'s three notes.** All three are dated parentheticals appended to the
  numbered items they qualify (item 7's `no_nav_inputs` meaning, item 11's sixth mapping and
  its ordering rule, item 15's rename and the reserved-set reshape), matching the spec's
  existing in-place-annotation style rather than rewriting shipped history. Item 6 still says
  "Keyboard defocus lives on CYCLE_REGION (Ctrl+`) and the mouse", which is a body sentence the
  notes did not reach — but `hotkeys.py`'s live comment was correctly updated to "Defocus
  lives on the mouse, never on Esc", so the source of truth is right and the spec body is
  historical record, which its own convention permits.
- **New comments state the now.** Reading every added comment line in the diff: no
  bug-we-hit story, no "see <doc> for the saga", no paragraph-length rationale. The
  `_RESERVED_CHORDS` block is the longest and every line of it is a live fact about the
  current table. No `# noqa` / `# type: ignore` / inline import anywhere in the diff.
  Annotations are complete (pyright 0 errors).
- **Test idioms.** Both new modules match the repo's shape: a module docstring stating the
  invariant and pointing at the owning spec (as `test_ui_prose_budget.py` and
  `test_command_routing.py` do), self-invalidating exemptions written out with reasons
  (`_HOST_OWNED`, the `CYCLE_COPILOT_LAYOUT` exemption) rather than silent widening, and
  domain floors plus format canaries so a narrowed domain fails loudly.

---

## False trails

- `ui_primitives.close_cross_button` and `cell_delete_confirm` have no caller anywhere, but
  `git grep` at `2b43f83^` shows both were already caller-less before this commit — pre-existing,
  not left behind by the region removal.
- `ai_docs/features/020_copilot_agent/*` and `064_multistep/design_round/*` mention
  `ActiveRegion`, `_set_region`, `cycle_region` and `nav_flatten` dozens of times, but those are
  historical feature records of what was built at the time, which `conventions.md`'s own
  "this file is not a changelog" split assigns to the spec — correctly left alone.
- The pytest run segfaults in `glfw.get_video_mode` from `App.__init__` via `tests/conftest.py`,
  which is this box having no display, not a defect in the commit.
- `_RESERVED_CHORDS["emacs"]` raises `KeyError`, but the pydantic `Literal` plus
  `model_salvage` make an out-of-domain value structurally unreachable at that call site — not
  a defect.
- `standard_keymap.md` names `F6` as switching keymaps at runtime, which looks like a clash
  with the app's F-key tier — it is the upstream *reference editor's* own binding, not
  something `ed_key` consumes through the `.so`, and no app command uses F6.

---

## Coverage statement

Read end-to-end: `CLAUDE.md`; `ai_docs/conventions.md` (all 965 lines);
`ai_docs/dev_flow.md`'s module map (lines 245-350) and its status/row conventions;
`.claude/skills/imgui-ui/SKILL.md` §6, §8 and §9; every file in `git show --stat 2b43f83`
(25 files); plus, for context the diff needed, `shaderbox/commands.py`,
`shaderbox/hotkeys.py`, `shaderbox/tabs/code.py`, `shaderbox/help_content.py`,
`shaderbox/widgets/cheatsheet.py`, `shaderbox/editor/ffi.py`, `shaderbox/ui_primitives.py`,
`shaderbox/resources/editor/standard_keymap.md`, `tests/test_ui_prose_budget.py`,
`tests/test_command_routing.py`.

Executed as evidence: `make gates` (exit 2, unpiped, captured to a file); `ruff check` and
`ruff format --check` over a pristine `git archive 2b43f83` extract; the five relevant test
modules (548 passed); an AST walk of `App` assignments against package-wide attribute loads;
an independent re-implementation of the doc parse scoring every `CommandSpec` against both
keymaps; a driven run of `test_region_system_is_gone`'s own helpers over the shipped tree to
count containers, flags and Name-resolution; and a live `libeditor.so` probe of whether
`ed_set_style` clears `ed_draw_chrome`.

Not covered (out of this role, or not possible here): runtime behaviour of the moved chords
under a focused editor, which needs the app on a display; the W-E spec's own decision-by-
decision conformance (that is the post-spec review); and the correctness of the vendored
keymap docs themselves against the `.so`, which `abi_probe.py` owns.

---

# Round 2 (closure)

Against `7bd7a1d` (finding 1) and `ce337bc` (findings 2-7). Read via `git show <sha>:<path>`
and a `git archive ce337bc` extract, because W-G is being implemented concurrently in the
working tree. Overall: **PASS**.

| # | Finding | Verdict |
|---|---|---|
| 1 | `make gates` red | **CLOSED** |
| 2 | `dev_flow.md` module map stale | **CLOSED** |
| 3 | keymap vocabulary in three homes | **CLOSED** |
| 4 | comments narrating a gone outline | **CLOSED** |
| 5 | imgui skill § 8 cites `active_region_outline` | **CLOSED** |
| 6 | Tab check blind to wrapper calls | **CLOSED**, and wider than reported |
| 7 | stale user-facing chords | **CLOSED**, with a gate |
| 8 | `ui_regions.py` rename | declined; withdrawn (see below) |

### 1 — CLOSED

`ruff check` and `ruff format --check` over a pristine `git archive ce337bc` extract of
`tests/` and `shaderbox/` both exit 0 ("All checks passed!", "226 files already formatted").

Better than asked: `7bd7a1d` found the root cause I had not — the two modules were UNTRACKED
when the gate ran, and `make check` is `pre-commit run --all-files`, which only sees tracked
files. So ruff never looked at them and the green gate was truthful about the tree it could
see. The fix files the mechanism in `dev_flow.md ### make gates`: "**Stage new files before
running it.** … an UNTRACKED module is skipped by ruff, formatter and pyright alike … Two 069
commits landed exactly that way." That is a rule with a named recurrence (W-B was the first),
which is the bar for writing one down.

### 2 — CLOSED

All three map entries now match the code:

```
dev_flow.md:249  editor sessions, popup-state, exporter panels).        # "nav," dropped
dev_flow.md:269  **`editor/`** — the embedded keymap-selectable code editor, vim or standard per
                 `EditorSettings.keymap` (features 067, 069)
dev_flow.md:344  **`ui_regions.py`** (`DocumentTab` — the settings-panel tab enum, kept out of
                 `commands.py` so the persisted model layer doesn't pull in imgui)
```

`git show ce337bc:ai_docs/dev_flow.md | grep ActiveRegion` returns nothing. The
`editor/` entry now agrees with `conventions.md`'s inline-editor bullet, which was the
disagreement reported.

### 3 — CLOSED

The alias landed and the combo derives from it, matching the `ResultWidgetKind` precedent I
cited:

```
ui_models.py:41   EditorKeymap = Literal["vim", "standard"]
ui_models.py:205  keymap: EditorKeymap = "vim"
settings.py:52    _KEYMAPS: tuple[EditorKeymap, ...] = get_args(EditorKeymap)
```

Executed against the extract: `_KEYMAPS == get_args(EditorKeymap) == ('vim', 'standard')`, so
the combo's option list cannot drift from the persisted domain. `_RESERVED_CHORDS`'s string
keys stay, which is the disposition my own finding argued for.

### 4 — CLOSED

Both sites rewritten to describe the mechanism that still runs, with no outline:

```
ui.py     "the one-shot grab must precede its begin_child, and code.draw clears the flag
           once this has consumed it."
code.py   "ui.py consumed the imgui half (set_next_window_focus before the child), so the
           latch has done its job and is cleared here."
```

The test docstring's wrong count and its two deleted example names are gone:
`grep "panel_flags\|grid_flags\|three of the eight"` over
`ce337bc:tests/test_region_system_is_gone.py` returns nothing.

*Correction to my own round-1 report:* I listed `ui.py:62` ("a 1-item switcher + 2nd outline")
in the same grep. Re-read in context, it describes imgui's own built-in Ctrl+Tab window
switcher and its overlay, not the deleted `active_region_outline` — my grep matched the word,
not the concept. It was correctly left alone. The sibling `commands.py` rationale WAS rewritten
and is now more accurate than what I asked for: "imgui's built-in window-cycle needs
`nav_enable_keyboard`, which is off app-wide (069 W-E D4). `WindowFlags_.no_nav_focus` … keeps
it that way if nav is ever turned back on" — nav-off is the reason Ctrl+Tab is free; the flag
is belt-and-braces.

### 5 — CLOSED

`grep active_region_outline` over `ce337bc:.claude/skills/imgui-ui/SKILL.md` returns nothing.
The § 8 bullet keeps its lesson and loses the dangling symbol ("The same window's own
foreground draw already carried this `any_popup_open()` guard; the focus grab had not.").
§ 9's editor-focus bullet also dropped its stale "Esc / target switch" defocus list for "by an
explicit defocus, or by a tab or document switch", which I had not flagged.

### 6 — CLOSED, and the hole was wider than I reported

I reported one uncovered container (`app_panel`, via `fps_overlay` → `chip_button`). The fix
measured the real extent: **none** of `ui.py`'s six containers writes a widget call inline, so
the syntactic check scored all three of that file's flags clean and deleting all three left
the suite green. My finding named the mechanism correctly and undercounted the blast radius.

Verified by running the falsifiers against the extract, restoring from a copy each time and
diffing the restore against `git show ce337bc:<path>` before the next run:

```
delete ALL THREE ui.py flags   -> 1 failed, 2 passed     (was GREEN before the fix)
delete document_settings only  -> 1 failed, 2 passed
delete chat _WINDOW_FLAGS      -> 1 failed, 2 passed
restored tree                  -> 3 passed
```

The three restores were confirmed byte-identical to the committed blobs before the suite ran.

The walk now follows calls transitively (bare names via imports, `mod.func` via the alias,
`ui_primitives` wrappers derived by AST rather than listed, and callees parked in a
module-level table), stopping at a callee that opens its own top-level window or popup. I
confirmed the deepest claimed route resolves: `widgets/uniform.py` contains `input_text`,
`input_text_multiline`, `drag_int`, `drag_float`, reachable from `document_settings` only via
`_NODE_TABS`.

`_FOCUSABLE` is narrowed from twelve names to the eight text-entry widgets, on a headless
measurement with nav OFF, and `_NAV_ONLY_FOCUSABLE = ("checkbox", "combo", "selectable",
"button")` names the excluded four so a later reader cannot restore them without re-running the
probe. That measurement also corrected the wave spec's own item 1: the grid's flag is earned by
the panel's sliders, not by `preview_cell`'s selectable tiles. I verified the two remaining
green sites are green for the stated reason — neither `tabs/code.py` nor
`widgets/document_grid.py` contains any `_FOCUSABLE` widget — and the docstring now states
measured coverage ("WHAT IT COVERS, measured rather than claimed") instead of implying the
coverage it lacks, which is the honest disposition for a defensive flag.

### 7 — CLOSED, with a gate that fires

All three sites fixed: `help_content.py:155` "Press `Alt+L`", the lib-picker docstring
`"""Shader-library picker (Alt+L).`, and smoke's "Exercise the tab-jump wiring".

The durable half is `test_no_help_prose_quotes_a_chord_the_table_does_not_bind`, which parses
every backticked chord out of every Help section body and requires `COMMAND_SPECS` to bind it
— the generated table followed the specs for free, and a chord typed into a body did not.
Falsified against the extract: restoring `Ctrl+P` in the prose turns it red
(`test_help_content.py:73: AssertionError`), and reverting the string returns 6 passed. The
restore was verified identical to the committed blob.

### 8 — withdrawn

I marked the rename optional and the maintainer declined it as a preference. Per the
late-round rule a preference is a false trail; the docstring already carries the right meaning
and the map entry now agrees with it. Nothing outstanding.

### Coverage

Read via `git show ce337bc:<path>`: `dev_flow.md`, `ui_models.py`, `popups/settings.py`,
`hotkeys.py`, `ui.py`, `tabs/code.py`, `commands.py`, `help_content.py`,
`popups/lib_picker/__init__.py`, `scripts/smoke.py`, `.claude/skills/imgui-ui/SKILL.md`,
`tests/test_region_system_is_gone.py`, `tests/test_help_content.py`, plus both commit messages
and the full `ce337bc` diff of the skill.

Executed: `ruff check` + `ruff format --check` over a pristine `git archive ce337bc` extract;
610 passed / 4 skipped across the seven relevant test modules; the derived-tuple identity for
`_KEYMAPS`; four falsifiers (three flag deletions, one Help-string revert), each restored and
the restore diffed against the committed blob before the next run; and an AST resolution of
the `document_settings` -> `_NODE_TABS` -> uniform-row route.

Not covered, unchanged from round 1: runtime chord behaviour under a focused editor (needs a
display), and the concurrent W-G work in the working tree, which was deliberately not read.
