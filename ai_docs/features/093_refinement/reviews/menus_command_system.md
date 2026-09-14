# Review — the command system (`505c992`)

Anchors: the maintainer's brief ("you must not assume any pre-existing order… design the
whole system!"), row 17 of `00_findings.md` ("good coverage", "a very convenient ui/ux",
"don't overblow the state", "the exact balance"), the code enumeration below, and
`04_menus_inventory.md` §1-8. The design under review is `06_command_system.md`; its author
is not an anchor here.

## Verdict: PARTIAL

The system is real. The table is one designed object, not a rendering of an inherited
grouping: every category is an object, `Tools` is gone with its six verbs re-filed by the
object they act on, and the five consumers all walk `CATEGORY_ORDER` and `COMMAND_SPECS`
rather than authoring a list. Measured against the old table (`505c992^`), the redesign
moved 15 specs between categories, relabelled 11, added 1, and **changed zero chords** —
rule 6 holds exactly, which is the rule most easily broken by a redesign this wide.

Seven findings. Five are table changes; two are gates that do not gate.

---

## 1. Coverage — every verb in the code, classified

Enumerated from `grep -rhno "app\.[a-z_]*("` over `shaderbox/{widgets,popups,tabs,exporters}`
+ `ui.py` (72 distinct `App` methods), every `menu_item_simple` / `confirm_menu_item` label
(24 sites), and every `standard_button` / `primary_button` / `danger_button` label (~78
sites).

### (a) A command in the table — 33 specs

All 33 render in the bar; `tests/test_menus.py::test_the_bar_draws_exactly_the_tables_specs`
pins the label set both ways. Three verbs the inventory's "reachable from exactly one
surface" list named as chord-only — `SAVE`, `CYCLE_CODE_TAB`, `OPEN_PALETTE` — now each have
a menu home. That is the brief's "good coverage" delivered where it was measured as missing.

### (b) An object verb, rightly not a command

The design's "What is NOT a command" section rules these out, and the rule holds on
inspection:

| Verb | Surface | Why not a command |
|---|---|---|
| `Fit`, `Arrange` | graph canvas menu (`pass_graph.py:881,883`) | act on a view the canvas alone holds |
| `Group`, `Dissolve`, `Leave group`, `Open` (box) | node / pass menus | target supplied by the menu |
| `Delete` (pass) | `pass_list.py:94` | object verb; target is the clicked pass |
| `Open`, `Open folder`, `Delete` (document tile) | `document_grid.py:47-52` | ditto; the bar's siblings act on the CURRENT document |
| `New file here`, `New subdirectory`, `Rename`, `Delete`, `Reveal in file manager`, `Insert at caret`, `Open file at declaration`, `Copy name`, `Favorite`/`Unfavorite` | lib-tree menus (`lib_picker/tree.py`) | verbs on a tree row inside one modal |
| resolution mode, aspect chips, canvas fields, uniform sort, pass selector | Document / Uniforms tabs | state controls, not verbs |
| `Render`, `Upload`, `Add to pack`, `New pack`, `Delete pack`, `Connect`, `Clear token`/`Clear credentials` | `exporters/{telegram,youtube}.py`, `tabs/render.py` | exporter verbs, named where they are |
| `Send`, `Stop`, `Revert`, `Allow`/`Deny`, gate answers | `copilot_chat.py` | verbs on one in-flight turn |

The rule is coherent and consistently applied. No misfiling found in this class.

### (c) Neither — reachable by exactly one button, no menu home

Rule 4 says "a verb that exists only as a button or a chord is the defect". Three survive,
and none is named in the "What is NOT a command" section:

**C1. `Clear` — the copilot chat** (`copilot_chat.py:762` → `app.copilot_clear_chat`).
A `danger_button` on the chat header. It is **destructive** (drops the whole conversation),
has no chord, no palette entry, no menu home, and no `confirm_label` — a single click wipes
the chat. Contrast `DELETE_DOCUMENT`, which rule 5 makes a two-click confirm submenu. This is
the sharpest (c): the one verb in the app that both destroys user work and is reachable by
one unconfirmed click.

**C2. `Reset library...` / `Confirm reset`** (`settings.py:199,209`). Destructive (trashes the
shader library), lives only inside the Settings modal, behind a hand-rolled two-step
`lib_reset_armed` flag rather than `confirm_menu_item`. Arguably (b) — a verb on the library
object — but the library has no object menu the way a pass or a document does, so it is
reachable from exactly one place and the design doc does not rule it out.

**C3. `New`, `Open other...`, `Delete` — the Projects modal** (`projects.py:116-149`). Project
lifecycle verbs, reachable only inside the Projects modal. `Delete` is a `danger_button`
gated by an `armed` flag. Defensible as (b) if "project" is read as an object whose menu is
the modal — but the design's category list makes File "the app's files and its settings", and
`New project` sitting outside File while `New document` sits inside it is an asymmetry the
doc does not address.

**Not a finding, checked and cleared:** the copilot `Copilot` toggle chip
(`ui.py:725` → `toggle_copilot_open`) is a second path to `TOGGLE_COPILOT`, not a verb without
a home. `Close` on the chat header sets `is_copilot_open` directly — same verb, and the
toggle covers it.

---

## 2. The six rules, applied to themselves

**Rule 1 — a category is an OBJECT.** Holds. File / Document / Pass / Editor / View / Help;
each names a thing the verbs act on. `View` is the one that is a *state* rather than an
object, but "what is shown" is exactly what View means in every desktop bar, so it is the
convention, not a break. No category is a mode or a toolbox — `Tools` was the break and it is
gone.

**Rule 2 — most-used first within a group.** One violation.

> **F1 — `Document` group 1 leads with `Open script`, not `Open graph`.** The group is
> `Open script` / `Open graph` / `Open folder`. `Open folder` third is right (rarest,
> unbound). Between the first two, the table asserts script is the more used. `Open folder`
> being in this group at all is the weaker claim: the first two open an *editor tab*, the
> third opens a *file manager*. See F5.

Otherwise the ordering is sound: `Add pass` above `Import passes`, `Next pass` above
`Previous pass`, `Help` above `Keyboard cheatsheet`, the four panels in their Ctrl+1..4 order.

**Rule 3 — a label names its object.** Holds, and the 11 relabels are the rule being applied:
`Format` → `Format code`, `Cycle code tab` → `Next code tab`, `Document tab` → `Document
panel`, `Toggle keyboard cheatsheet` → `Keyboard cheatsheet`. No trailing ellipsis survives
(`test_no_command_label_is_respelled_with_an_ellipsis`). Two labels are worth naming:

> **F2 — `Help` under `Help` is a label that does not name its object.** In the bar it reads
> `Help ▸ Help`. In the flat palette it is the bare word `Help`, which names the panel only by
> accident. Rule 3's own examples say a surface-opening verb takes the surface's name — the
> surface is the Help *panel*. `Help panel` would read correctly in all three consumers and
> match the four `… panel` labels already under View.

> **F3 — `Play/stop script` is the only label carrying a slash.** Rule 3 says a toggle says
> `Toggle …`; this one does not, and `Toggle copilot` does. The slash form is the more useful
> of the two here (it says what both states are), so the finding is in the *rule*, not the
> label: rule 3 should name the slash form as the sanctioned shape for a two-state verb whose
> states have names, or the label should become `Toggle script`.

**Rule 4 — every command has a menu home.** Holds for all 33. The converse — every verb has a
command — is what C1/C2/C3 break. The `command_label` half is honoured at all three sites that
use it (`code.py:797`, `document.py:477,480`), and
`test_no_button_respells_a_command_label_in_another_case` pins it.

**Rule 5 — a destructive verb carries `confirm_label`.** One spec carries one:
`DELETE_DOCUMENT`. The two questions asked:

> **F4 — `RESET_DOCUMENT` is destructive and carries no `confirm_label`.** It calls
> `session.reset_document`, which the Document tab itself draws as a `danger_button`
> (`document.py:244`, comment: "Destructive, so the danger tier"). The app's own tier system
> classifies it destructive; the command table does not. So F6 on the bar and the Document
> tab's own button both fire it on one click, while the strictly-recoverable
> `DELETE_DOCUMENT` (moves to trash) takes two. The two are inverted: reset discards
> histories, clock, script and video state with no trash to recover from.

**`QUIT` is not a rule-5 violation, and this is the false trail** — see below.

**Rule 6 — chords are kept, nothing new bound.** Holds exactly. Diffed `505c992^` against
`505c992`: zero chord changes across the 32 shared specs, and the one added spec
(`OPEN_DOCUMENT_DIR`) is unbound, as the doc states. `IMPORT_PASSES` and `OPEN_DOCUMENT_DIR`
are the two unbound specs the doc names. Rebindings persist by `spec.id.value`
(`app.py:1015`), so the 11 relabels cannot orphan a user's saved chord.

---

## 3. Every consumer renders the same system

| Consumer | Walks `CATEGORY_ORDER` | Table order | Same labels | `separator_before` | `confirm_label` | scope |
|---|---|---|---|---|---|---|
| Menu bar (`menus.py:76-90`) | yes | yes | yes | draws it | confirm submenu | per-item `begin_disabled` |
| Palette (`app.py:841-862`) | **no** | insertion order, then fuzzy-sorted | yes, padded | ignored | **ignored** | ignored |
| Cheatsheet (`cheatsheet.py:33-41`) | yes | yes | yes | its own rule per category | n/a | filters by `_is_active` |
| Rebinder (`settings.py:314-334`) | yes | yes | yes | rule per category only | n/a | shows all |
| Help snippet (`help_content.py:74-86`) | yes | yes | yes | dropped entirely | n/a | shows all |

Four of five are faithful. The two divergences:

> **F5 — the palette loses the grouping and can fire a destructive verb on one click.**
> `_register_palette_commands` iterates `COMMAND_SPECS` flat, so `separator_before` has no
> effect (acceptable — a flat search list has no groups) but **`confirm_label` is also
> ignored**: `imcmd.Command.initial_callback` is wired straight to
> `command_callbacks[spec.id]`. Typing "del" + Enter in the palette deletes the current
> document with no confirm, while the same verb on the bar takes two clicks. Rule 5 says a
> destructive verb "renders as a confirm submenu wherever it is a menu item (the bar, a
> tile's menu)" — the palette is not a menu item, so the design is *self-consistent*; the
> finding is that the rule's scope leaves the one consumer where the verb is easiest to fire
> by accident (fuzzy match + Enter) as the one with no guard.
>
> Separately: the upstream `imgui-command-palette` fuzzy-sorts by match score once a query is
> typed, so "the palette reads as the same list as the cheatsheet" (`app.py:842` comment)
> holds only for the empty query. Not a defect — a comment slightly overclaiming.

> **F6 — the Help snippet drops every group.** It filters `spec.default_chord` and joins with
> `\n`, never reading `separator_before`. So Help shows File as five flat rows where the bar
> shows three groups. The cheatsheet has the same flatness but that is deliberate (it
> segments by category rule instead, and only shows what is valid now). For Help — the
> surface a new user reads to *learn* the system — the grouping is the part worth keeping.

**`Open folder` absent from Help while present in the bar: correct.** The section is titled
"Keyboard shortcuts" and its body says "Defaults — every one is rebindable in Settings". A
row with no chord in a shortcuts table would be an empty right-hand column.
`test_every_bound_spec_reaches_the_help_shortcuts` pins the direction that matters (bound ⇒
listed). `Import passes` is absent for the same reason, equally correctly.

---

## 4. The map as a user reads it

Rendered from `COMMAND_SPECS` (`uv run python -c`), 33 specs, 6 menus, 12 groups. Judged menu
by menu:

**File** — correct, and conventional. `New document` / `Save`, then `Projects` / `Settings`,
then `Quit`. A desktop user finds all four where they expect them. Settings under File is the
Windows/Linux convention (macOS puts it under the app menu, which this app has no equivalent
of). No change.

**Document** — one group mixes kinds (F1 / see diff below). `Open script` and `Open graph`
open editor tabs; `Open folder` opens the OS file manager. Otherwise correct: play/reset
together, delete last and confirmed.

**Pass** — correct. Create group, then inspect group, then navigate group. Reads exactly as a
Blender or TouchDesigner user would expect of an object menu.

**Editor** — correct, with one label question. `Format code` / `Next error`, then the tab
group, then `Shader library`. `Shader library` alone in its own group at the bottom is the
right shape (it is a different kind of thing from tab management).

**View** — correct. The four panels in chord order, channel view, the copilot pair, the
palette last. `Command palette` under View is the one placement a VS Code user might look for
elsewhere, but View is where "what is shown" lives and the palette is a surface. No change.

**Help** — correct except F2's label.

**A menu a desktop user would expect that is absent: `Edit`.** And this is right — the app has
no cut/copy/paste/undo at the application level (the editor owns its own, the graph owns its
own). An `Edit` menu with nothing in it would be worse than none. `Window` likewise: the app
is single-window and the panel focus verbs already live under View. **No missing menu, no
menu present without cause.** This is the "don't overblow the state" half of the brief met.

### The table changes, as concrete diffs

```
F1/D1 — Document: Open folder leaves the tab-opening group.
    CommandSpec(CommandId.OPEN_SCRIPT, "Open script", _chord(K.r, K.mod_alt), C.DOCUMENT),
    CommandSpec(CommandId.OPEN_GRAPH,  "Open graph",  _chord(K.g, K.mod_alt), C.DOCUMENT),
-   CommandSpec(CommandId.OPEN_DOCUMENT_DIR, "Open folder", 0, C.DOCUMENT),
    CommandSpec(..TOGGLE_DOCUMENT_PLAY.., separator_before=True),
    CommandSpec(..RESET_DOCUMENT..),
+   CommandSpec(CommandId.OPEN_DOCUMENT_DIR, "Open folder", 0, C.DOCUMENT,
+               separator_before=True),
    CommandSpec(..DELETE_DOCUMENT.., separator_before=True, confirm_label="Move to trash"),
  -> Document reads: [open script, open graph] / [play, reset] / [open folder] / [delete]
     Each group one kind. Open folder sits next to Delete, both "the document as a file".

F4/D2 — Reset document becomes a confirm, matching its own danger_button tier.
    CommandSpec(CommandId.RESET_DOCUMENT, "Reset document", _chord(K.f6), C.DOCUMENT,
+               confirm_label="Reset histories and clock"),

F2/D3 — Help ▸ Help names its surface.
-   CommandSpec(CommandId.HELP, "Help", _chord(K.f1), C.HELP),
+   CommandSpec(CommandId.HELP, "Help panel", _chord(K.f1), C.HELP),

C1/D4 — the copilot's Clear gets a home and a confirm. New enum member CLEAR_COPILOT_CHAT
        under View, in the copilot group, unbound (rule 6 binds nothing new):
    CommandSpec(CommandId.TOGGLE_COPILOT, "Toggle copilot", _chord(K.j, K.mod_alt), C.VIEW,
                separator_before=True),
    CommandSpec(CommandId.CYCLE_COPILOT_LAYOUT, "Next copilot layout", ...),
+   CommandSpec(CommandId.CLEAR_COPILOT_CHAT, "Clear copilot chat", 0, C.VIEW,
+               confirm_label="Clear the conversation"),
    and copilot_chat.py:762's danger_button takes command_label(CLEAR_COPILOT_CHAT)
    and routes through confirm, per rule 4's second sentence.

F5/D5 — the palette honours confirm_label. In _register_palette_commands, a spec carrying
        one gets an initial_callback that opens the same confirm the bar draws, rather than
        the raw verb. (Code, not a table row — listed here because it is rule 5's gap.)
```

C2 and C3 are left as judgement calls for the maintainer: both are defensible as (b) object
verbs, but the design doc should say so explicitly in "What is NOT a command" — currently it
names neither, so a reader cannot tell whether they were considered.

---

## 5. Tests — what they catch and what they do not

`tests/test_menus.py`, 35 tests, all passing at `505c992`. Each claim below was checked by
**mutating the table in a clean worktree of `505c992` and running the suite**, not by reading.

### What the group-structure and contiguity tests catch (verified by breaking them)

| Mutation | Caught by | Result |
|---|---|---|
| `separator_before=True` on `NEW_DOCUMENT` (a category's first spec) | `test_the_groups_are_the_designed_ones` | **1 failed** |
| `SAVE` recategorized to `C.VIEW` but left in the File block | `test_the_table_is_in_menu_order` | **1 failed** |
| `confirm_label` dropped from `DELETE_DOCUMENT` | `test_the_bars_delete_document_is_a_confirm_submenu` | **2 failed** |
| A spec deleted from the table entirely | `test_command_registry_coverage.py::test_every_command_id_has_a_spec` (not `test_menus.py`) | **caught, whole suite** |

So the separator set, the category-contiguity invariant, the confirm submenu and the
enum/table pairing are all genuinely gated.

### What they do not catch

> **F7 — the tests pin WHICH specs carry a separator, never the ORDER of specs within a
> category.** Two mutations, both run against the full 2509-test suite:
>
> - **Intra-category reorder.** Moved `Save` out of File's first group and into the second,
>   so the bar renders `New document / — / Projects, Save, Settings / — / Quit`. Rule 2's
>   "most-used first" is broken and the designed group membership is wrong.
>   → **2509 passed, 4 skipped.** Nothing failed.
> - **Clean category move.** Moved `Shader library` from Editor to View, relocating its row
>   in the table so contiguity still holds. Rule 1's object filing is broken; the verb now
>   renders under the wrong menu.
>   → **2509 passed, 4 skipped.** Nothing failed.
>
> `test_the_groups_are_the_designed_ones` asserts a *set* of ids
> (`behind == {OPEN_PROJECTS, QUIT, …}`), which is invariant under both mutations.
> `test_the_table_is_in_menu_order` asserts only that categories appear contiguously in
> `CATEGORY_ORDER` — it is satisfied by any permutation *within* a category and by any
> category assignment that is reflected in the table's ordering. Together they pin the
> skeleton and leave the design — which verb under which menu, in what order — ungated.
>
> The doc says "`tests/test_menus.py` pins the map's group structure and the table's order".
> The second half is the overclaim: the table's *category blocking* is pinned; its order is
> not.
>
> **The fix that closes it** — pin the map the doc draws, as the doc draws it:
> ```python
> def test_the_map_is_the_designed_one() -> None:
>     """`06_command_system.md`'s map, verbatim: each category's labels in order, grouped.
>     Falsifier: move `Shader library` from Editor to View, or `Save` into File's second
>     group -- both render a different map and both pass every other test."""
>     rendered: dict[str, list[list[str]]] = {}
>     for spec in COMMAND_SPECS:
>         groups = rendered.setdefault(spec.category.value, [[]])
>         if spec.separator_before:
>             groups.append([])
>         groups[-1].append(spec.label)
>     assert rendered == {
>         "File": [["New document", "Save"], ["Projects", "Settings"], ["Quit"]],
>         "Document": [["Open script", "Open graph", "Open folder"],
>                      ["Play/stop script", "Reset document"], ["Delete document"]],
>         "Pass": [["Add pass", "Import passes"], ["Open shader", "Pass settings"],
>                  ["Next pass", "Previous pass"]],
>         "Editor": [["Format code", "Next error"], ["Next code tab", "Close code tab"],
>                    ["Shader library"]],
>         "View": [["Document panel", "Uniforms panel", "Render panel", "Share panel"],
>                  ["Next channel view"], ["Toggle copilot", "Next copilot layout"],
>                  ["Command palette"]],
>         "Help": [["Help panel", "Keyboard cheatsheet"], ["Examples"]],
>     }
> ```
> One assertion replaces the separator-set test and subsumes it: it fails on every mutation
> above, and it is the map in the doc, so the doc and the gate cannot drift apart.

---

## False trails

Four things that look like findings and are not. Each was checked against the code.

**`QUIT` with unsaved work is not a rule-5 violation.** Quitting cannot lose work, so it is
not destructive. `ui.py:162-164` runs `app.save()` then `app.save_imgui_ini()` after the
main loop exits, unconditionally. `request_quit` (`app.py:867`) only sets the glfw
close flag; the save happens on the way out. Adding a `confirm_label` to `QUIT` would put a
confirm in front of a verb that loses nothing — the opposite of the brief's "exact balance".
(The one hole is narrower and is a *different* verb's problem: `shutdown()` does **not**
call `flush_all_dirty_editors`, which `switch_project` does — but that is a save-path
question, not a command-table one, and outside this review.)

**The palette's flat label list is not a consumer drift.** It pads labels to a common width
and appends the chord, which *looks* like a different rendering. It is the same labels in the
same table order; the padding is column alignment. Rule 3's "reads the same in the flat
palette" is met.

**`Open folder` missing from the Help snippet is not an inconsistency.** Covered in §3 — the
section is a *shortcuts* table and the spec is unbound by design.

**The 11 relabels are not a rule-6 break.** Rule 6 is about *chords*, and the persistence
layer keys on `spec.id.value` (`app.py:1015-1017`), not the label — verified by reading
`_merge_effective_bindings`. A user's saved rebinding survives every one of the 11 renames.

---

## Summary

| # | Finding | Closes with |
|---|---|---|
| C1 | `Clear` (copilot chat) — destructive, one unconfirmed click, no menu home | D4 |
| C2 | `Reset library` — destructive, Settings-modal only, not ruled out by the doc | doc: name it in "What is NOT a command" |
| C3 | `New` / `Open other...` / `Delete` (Projects) — modal-only, not ruled out | doc: name them |
| F1 | Document group 1 mixes editor-tab verbs with a file-manager verb | D1 |
| F2 | `Help ▸ Help` does not name its surface | D3 |
| F3 | `Play/stop script` is the only non-`Toggle` toggle — rule 3 needs the shape | doc |
| F4 | `RESET_DOCUMENT` destructive, no `confirm_label`; inverted against `DELETE_DOCUMENT` | D2 |
| F5 | The palette ignores `confirm_label` — the easiest surface to misfire has no guard | D5 |
| F6 | The Help snippet drops every group | read `separator_before` in `_shortcuts_section` |
| F7 | The tests pin the separator SET and category contiguity, never the ORDER — a clean category move and an intra-category reorder both pass all 2509 tests | the map test above |

The three that matter most: **F7** (the gate does not gate the design it claims to pin, shown
by two mutations against the full suite), **C1** (a destructive verb with no home and no
confirm, which is rules 4 and 5 failing on the same verb), and **F4** (the table's one
destructive-verb classification disagrees with the app's own button tier).
