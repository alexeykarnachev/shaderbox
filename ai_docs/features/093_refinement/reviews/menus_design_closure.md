# Closure review — `04_menus_inventory.md` §10 revision 2 against round 1

Scope: `## 10. Design pass` revision 2 (lines 515-774) against
`reviews/menus_design_brief.md` (F1-F12) and `reviews/menus_design_feasibility.md`
(the NOT-AS-WRITTEN / FEASIBLE-WITH-CHANGES change lists). Every code claim in the new
text was opened; the one library question revision 2 introduces (`confirm_menu_item`) was
driven through a real imgui frame on the `app` fixture.

**Verdict: PASS.** All 12 brief findings and all 11 feasibility change lists are closed by
quotable revision-2 text. No claim in the new text is wrong. Nothing is bent silently: the
three settled decisions revision 2 touches (092 D14, 092 D16, the conventions `InlineInput`
bullet) are each named at the point of the change, and 093 S5 is cited correctly for the one
place a round-1 recommendation was declined. Both explicit rejections of round 1 hold.

Counts: 12 of 12 brief findings CLOSED, 0 open. 11 of 11 feasibility change lists CLOSED,
0 open. 0 wrong claims in the new text. 2 of 2 rejections correct.

---

## 1. Closure, item by item

### 1.1 The brief report, F1-F12

**F1 (blocking) — P4's premise about the lib delete is false; its two "recoverable" cases have
no user-facing recovery. CLOSED.**

Revision 2 rewrote P4 from the code and inverted the taxonomy exactly as F1 asked. 10.1/4:

> a lib file or directory delete MOVES to `.trash/` and already toasts "recoverable in .trash/"
> (`shader_lib/file_ops.py:223-250`) — and confirms twice; a document delete moves to the
> project trash but its only Recover affordance is the copilot's card, built for a copilot
> delete alone (`app.py:810-825`, `copilot/backend.py:1177-1188`) — a grid delete has no
> undo, and confirms once

And 10.5's first bullet says so in the maintainer's terms: "the lib deletes were trash moves
with a toast (not 'a real filesystem delete'), the document delete has no user-facing undo
… so the modal-vs-toast split inverted on three of four cases".

The three code reads are correct. `file_ops.py:223-250` `delete_file` does
`shutil.move(str(path), str(dest))` into `shader_lib_trash_dir()` and pushes
`msg += " - recoverable in .trash/"`; `delete_dir` (`:252-`) moves file by file "so nothing is
silently rmtree'd". `app.py:810` `recover_deleted_document(self, msg: Message)` reads
`msg.recover`, and `RecoverInfo` is constructed only in `copilot/backend.py:1177-1192`
`delete_document` — a grid delete produces no `Message`, so no card. Revision 2's resolution
(one submenu for all three, no modal anywhere) satisfies the rule on every case rather than
sorting them.

**F2 (blocking) — P2 adds three App verbs the app does not have and does not say so. CLOSED.**

Two of the three verbs are dropped, the third is named as new. 10.5: "the editor-tab menu
dropped (two new verbs it did not name, and a right-click that selects)" — which removes
editor-tab `Open folder` and `Close others`. The surviving one is flagged in P2:

> `Open folder` on a document generalizes `App.open_current_document_dir` to
> `open_document_dir(document_id)` (three lines; the current-document verb calls it).

Verified: `app.py:1890` is `def open_current_document_dir(self) -> None` with no document-id
parameter, its body building `self.paths.documents_dir / self.current_document_id` and calling
`open_in_file_manager` (imported `app.py:126`). The generalization is exactly the three lines
described — lift the id to a parameter, keep a current-document wrapper. Nothing named
`open_document_dir` or `close_other_tabs` exists today; revision 2 does not claim otherwise.

**F3 (blocking) — P2 reverses 092 D14 silently and the reversal breaks the verb's feedback.
CLOSED.**

Revision 2 reverses the reversal and names D14. P2's table:

> | pass, node only | + Group... (before Leave group) | graph node (today; 092 D14 — it seeds
> `view.selection`, which the strip does not have and could not show) |

and 10.5: "P2: `Group...` stays node-only (092 D14, silently reversed before)". The parenthetical
gives D14's structural reason, which is the half F3 said was missing. 092 `03_spec.md` D14 reads
"Right-click with a selection adds `Group...` to the node menu … writes
`set_pass_groups(document_id, names, group)` through `App.group_selection`" — node-only, seeded
from `GraphViewState.selection`, as quoted.

**F4 — P4 reverses 092 D16's two-click arm without the pointer. CLOSED.**

P4's closing sentence:

> Reverses 092 D16's "the strip's two-click arm" (already gone with W3-2) and the lib tree's
> second-open confirm; records the rule in `conventions.md`.

This is the W3-5 shape F4 asked for ("**Reverses 092 D1's …**"), with the parenthetical noting
the arm was already gone. 092 D16 verbatim: "a member is deleted from its own node menu with the
strip's two-click arm (`delete_pass` + `close_editor_for_path`, exactly `pass_list._delete_pass`)"
— the quoted fragment is accurate. 10.5 also carries "092 D16 is named."

**F5 — P8 is not the behaviour-preserving fold it is written as. CLOSED.**

P8 is dropped entirely. 10.1/8:

> Three branches against two; the difference is deliberate and the comment at `app.py:901`
> says why (a click has already moved focus). Stays (10.3).

and P8 is now the one-line "**P8. The two copilot toggles stay.** (10.1/8.)", with 10.5's "P8
dropped (a deliberate, documented difference)". F5's own fallback ("or drop it — the observation
itself concedes the difference is deliberate and documented") is the branch taken. The anchor is
right: `app.py:900-905` `toggle_copilot_open` carries the comment "The bar button: a plain
open/close toggle, NOT focus-aware (a click already moved focus off the chat, so the focus-aware
toggle_copilot would blink it back open)" — at `:901`, as cited.

**F6 — P5's second half changes behaviour the observation did not ask about. CLOSED.**

Revision 2 takes F6's first fix ("keep the prompt a popup and fix only its commit rule"). P5:

> The group prompt STAYS a popup (093 S5's Delete-key gate relies on it being one: an inline row
> in the canvas child would be hovered, and only `is_any_item_active` would stand between a Delete
> typed into the field and an unwire); its state moves from `group_prompt: bool` + `group_name:
> str` to one `InlineInput` on `GraphViewState`. Only the commit rule changes: a click away
> commits a non-blank name (a blank still refuses, as today).

The S5 citation is verified below (§2.7). The residual state change is a reduction — two fields
to one object on the same `GraphViewState`, no new surface.

**F7 — P9's gate is 23 call sites for a two-line defect. CLOSED.**

P9 is reduced to the correction plus an explicit refusal of the wrapper:

> A menu-item wrapper primitive is NOT added: 21 of the 23 `menu_item_simple` sites pass no
> `enabled=` and gain nothing from one, and the red-text pushes P9 would have absorbed go with
> P4's primitive.

That is F7's own arithmetic (23 sites, two carrying `enabled=`) used to reach the smaller
proposal, and it disposes of the `danger=True` half by pointing at P4. The `and deletable` removal
and the two doc corrections are what remain. Verified: `grep -c menu_item_simple` over
`shaderbox/` gives 23, and §9.4's table has exactly the two `enabled=` rows.

**F8 — P11 amends a rulebook that lives outside the repo. CLOSED (as an explicit, correct
rejection).**

Revision 2 rejects the premise and still files the rule inside the repo. P11:

> The rule is filed as a `conventions.md` design decision ("a hint on a modal's or panel's list;
> a canvas or a strip with a visible primary click gets none; revisit if a walk finds a menu
> undiscovered") and mirrored in the repo's own `.claude/skills/imgui-ui/SKILL.md` §7.4.

and 10.5: "the skill file is the repo's own (`.claude/skills/imgui-ui/`), so it is amended too —
F8's 'outside the repo' was wrong." **The rejection is right** — see §3.1. F8's substantive
request (file the rule in `conventions.md` in "we decided X; revisit if Y" form) is met verbatim,
so the finding closes on both halves regardless of the premise.

Note the brief's separate sub-claim under F8's row (§1.1's footnote: P11 gives the node no hint
though finding 14 is a maintainer who could not find the node menu) is a UX judgement the brief
itself framed as such, not a numbered finding. Revision 2 keeps "the strip, the canvas and the
nodes get none" and files the rule with a revisit trigger ("revisit if a walk finds a menu
undiscovered") that names exactly this risk. That is a decision, not an omission.

**F9 — 10.3 and P4 contradict each other on whether a modal is added. CLOSED.**

The contradiction is gone because P4 no longer adds a modal. 10.3's first bullet now reads:

> The nine modals and their `PopupState` mutex: no modal is added or merged (P4's confirm is
> a submenu, not a modal).

and P4: "`begin_menu(label)` with the one item `confirm_label` drawn in `STATE_ERROR` text …
no armed state, no reopen, no modal, no `PopupState` change." 10.5 states the reconciliation:
"10.3's 'no modal is added' and P4 now agree." Verified that `PopupState` (`app.py:132-143`) has
eight non-CLOSED members — the "nine modals" count in 10.3 is the eight plus the revert modal,
whose state is `app.copilot_revert_target` outside the enum, consistent with P6's own wording.

**F10 — the Telegram hand-rolled red confirm child is dispositioned as a delete but not as a
tier violation. CLOSED.**

10.3 now names §9.5's row:

> The exporters' config panels, their `unconnected_gate`s and the Telegram pack forms —
> including the hand-rolled red child at `telegram.py:484-485` §9.5 flags: the panels' bodies
> are the exporters' own.

That is F10's requested "one clause in 10.3", naming the file:line and the section that flagged it.

**F11 — four §1-8 surfaces get no verdict. CLOSED.**

All four are now ruled on in 10.3, each with its reason:

> - The lib tree's inline favorite star beside the menu's Favorite / Unfavorite: the rulebook's
>   own carve-out (§7.4, "toggling a favorite — the inline star is fine"); P3's "no button on a
>   tile" is about verbs on a tile, and the star is a one-click state toggle on a row.
> - The editor error strip (a row whose primary click IS the action), the `K`-lookup note
>   (non-interactive) and the completion popup (the editor library's).

The §7.4 quotes check out against `.claude/skills/imgui-ui/SKILL.md:377-379`: "rows with a single
common action (toggling a favorite — the inline star is fine), or rows where the primary click IS
the action". F11's own predicted answer ("probably stays, by the rulebook's own carve-out") is
what revision 2 wrote, with the carve-out cited.

**F12 — the Examples modal's no-reset-on-open is left unruled. CLOSED.**

10.3:

> The Examples modal's selection persisting across opens: a browser's selection, not a query
> (§7.6 is about transient search state); deliberate at `examples.py:69`.

That is F12's requested line, with the distinction F12 itself suggested ("it is a *selection* in a
browser, not a search query — a defensible read").

### 1.2 The feasibility report's change lists

**P1 — FEASIBLE WITH CHANGES (four changes). CLOSED, all four.**

| Change asked | Revision-2 text |
|---|---|
| the primitive cannot live in `ui_primitives.py` | "`command_menu_item(app, command_id)` and the bar live in a new `shaderbox/menus.py` (imports `App`; `ui_primitives.py` is `App`-free by the three-layer rule and cannot host it)"; `command_label` "is a pure function in `commands.py` (a leaf)" |
| name the scope test, and decide "active" explicitly rather than inheriting the cheatsheet's answer | "A per-item enabled test, `menu_enabled(app, spec)`: EDITOR scope -> `app.active_tab is not None`; COPILOT scope -> `app.is_copilot_open`; GLOBAL -> always. … Not the cheatsheet's `_is_active` (it reads `editor_focused`, which the menu click has just cleared) and not `spec_eligible` (it rejects chord `0`, and Import passes is unbound)" |
| the `shortcut` positional | "`imgui.menu_item` takes `shortcut` as a required positional and returns a tuple (§0.4); a spec with chord `0` passes `""`" |
| the prose gate loses menu labels | "Gate: `tests/test_ui_prose_budget.py` gains a row scoring `CommandSpec.label` from `commands.py`, since a primitive taking a `command_id` is invisible to its AST walk … Break: a 5-word label in `COMMAND_SPECS` must fail." |

Revision 2 went past the report on the scope test: the report recommended promoting
`cheatsheet._is_active`; revision 2 declines it with a reason (see §2.5) and defines its own
predicate. That is the report's coupling 1 ("Decide what 'active' means for a menu item explicitly
rather than inheriting the cheatsheet's answer") answered, not ignored. The 0.2 probe finding
("do not extend it to the category menu") is also honoured: "Per item, never on the category
(`begin_menu` under `begin_disabled` does not open at all — probe §0.2)". The report's coupling 2
(the help-snippet coupling) is carried too: "`tests/test_command_registry_coverage.py` pins spec
labels to the help snippet, so the help text follows in the same commit."

The §10.4 sizing question the report flagged is resolved rather than deferred: `in_menu` is
decided, `False` for the four Focus-tab chords and Cycle code tab, with the two-affordances reason.

**P2 — FEASIBLE WITH CHANGES (three changes). CLOSED, all three.**

The editor-tab menu is dropped, which closes both the 0.3 right-click-selects side effect and the
`Close others` new verb:

> the editor tab (its one verb, Close, already has the tab's ✕ and `Ctrl+W`; a `Close others`
> would be a new verb and a right-click would also select the tab — probe §0.3)

`Group...` needing view state on the strip is closed by keeping it node-only (F3 above), which
dissolves the report's coupling 1 (the signature/state question) rather than answering it.
`Open folder` is named as a new verb with its derivation (F2 above).

**P3 — FEASIBLE WITH CHANGES (two changes). CLOSED, both.**

The five-callers correction is folded, in 10.5: "P3: `preview_cell` keeps its machinery for the
sticker grid (option A decided); the fifth caller (`uniform.py:194`) noted as unaffected."
Verified: exactly five call sites exist (`pass_list.py:135`, `document_grid.py:22`,
`uniform.py:194`, `telegram.py:672`, `telegram.py:691`), and `uniform.py:194` passes
`selected=False, armed=False` so no ✕ is drawn — "unaffected" is right.

Option A is decided rather than handed back, with the reason P4's own rule supplies:

> `preview_cell` keeps `armed` / `deletable` / the wash for its one remaining arming caller, the
> Telegram sticker grid: a sticker delete is a server-side verb inside an exporter panel this
> pass does not own (10.3), and a wash on the cell is the right confirm for an irreversible verb
> with no menu.

This also closes the brief's §1.5 Call A ("should have been decided by the pass"). The
`test_import_dialog.py:123-124` positional-argument casualty is *not* separately named in
revision 2 — but under option A the `armed` parameter survives, so the positional signature does
not shift and the test does not break. The change list's premise is retired by the decision, which
is closure, not an omission.

**P4 — NOT AS WRITTEN (re-derive the taxonomy from the code). CLOSED.**

Covered under F1. The report's three-line re-derivation is met on all three rows, and the fourth
("if `confirm_modal` is still wanted it would generalize `_draw_revert_modal` … P6 before P4") is
retired: revision 2's confirm is not a modal, so the ordering constraint dissolves. The report's
"the hand-rolled red pushes P4 retires are `tree.py:165-167` and `:279-281`, both real" is carried
verbatim into P4's parenthetical.

**P5 — FEASIBLE (two corrections). CLOSED, both.**

The "already true" correction is folded — 10.5: "the Projects row already IS an `InlineInput`, so
P5 is the commit rule, not an adoption". Verified: `popups/projects.py:19` imports `InlineInput`
from `editor_types`.

The import second-order note is folded verbatim into P5: "`file_ops.py` already imports imgui
through `theme`, so the move costs no new import". Verified by probe (§2.4).

**P6 — FEASIBLE WITH CHANGES (three gate problems + three breaks). CLOSED, all six.**

P6's gate sentence closes each in order:

> `tests/test_modal_chrome.py` enumerates its domain from `PopupState` plus the revert modal
> (never a `popups/*.py` glob — the lib picker is a package), resolves each member to its draw
> function, and asserts the body binds a local `keep_open` that it returns, ends in a
> `standard_button("Close")` / `("Cancel")` row, and has the spacer call before it (normalized
> across `imgui.dummy((0, SPACE.MD))` and `ImVec2` spellings). Breaks to try, one per clause:
> rename `keep_open` back in one modal; delete one spacer; add an enum member whose body returns
> `ok` with no Close row (tried on the lib picker, whose layout a glob misses).

Problem 1 → "never a `popups/*.py` glob — the lib picker is a package". Problem 2 → "binds a local
`keep_open` that it returns", which is the report's own stronger phrasing. Problem 3 → "normalized
across … spellings". The three breaks are the report's three, with the lib-picker targeting note
attached to the third. Both spellings verified live: `settings.py` uses the tuple form,
`copilot_chat.py:188,194` uses `imgui.dummy(imgui.ImVec2(0, float(SPACE.XS)))`.

**P7 — FEASIBLE WITH CHANGES (three couplings + two gates). CLOSED, all five.**

> `App.close_popup() -> bool` dispatches on `popup_state`: … `apply_editor_settings` + `CLOSED`
> for Settings (the `was_settings_open` latch at `hotkeys.py:366` / `:398-399` goes, or the apply
> runs twice), plain `CLOSED` for the rest; it returns `False` without closing when an inline
> input owns Esc (`projects_input_owns_esc`, `inline_input_owns_esc` move inside it).
> `hotkeys._handle_escape` calls it and loses its four carve-outs; the `rebinding_command` early
> return stays where it is. … Gates: `tests/test_import_dialog.py:63-67` (a source-substring test
> on `hotkeys.py`) is repointed at `close_popup` and made structural;
> `test_every_popup_state_has_a_draw_call` gains a clause that parses `close_popup`'s body for a
> branch per member. Break: add a member, wire its draw, omit its close branch — the new clause
> goes red while the old count stays green.

Coupling 1 (the latch) → named with both line numbers and the "or the apply runs twice"
consequence. Coupling 2 (two carve-outs are declines, not closes) → the `-> bool` return and "it
returns `False` without closing", which is the first of the report's two acceptable designs.
Coupling 3 (`rebinding_command`) → "stays where it is". Gate 1 → repointed and made structural.
Gate 2 → the break is the report's exact break, with the "old count stays green" half that makes
it breakable rather than vacuous.

**P8 — NOT AS WRITTEN. CLOSED** (dropped; see F5).

**P9 — NOT AS WRITTEN (justification gone; gate reds five labels). CLOSED.**

The justification's disappearance is accepted, not argued around:

> `pass_list.py`'s `and deletable` and its comment go; §9.4 is rewritten as "no Python-side
> guard is needed on this build"; `.claude/skills/imgui-ui/SKILL.md` §7.4's bullet is rewritten
> to state the probe's result and the build it was measured on.

The gate that would have reddened five labels is not built at all (no wrapper primitive), so the
prose-budget problem and the "phrase the allowlist by module set" problem both retire. 10.5: "P9
replaced by the probe's result."

**P10 — FEASIBLE (one caution). CLOSED.**

P10 carries the caution's substance — "the over-budget allowlist rows for all twelve are deleted
so the gate holds them", with the replacement given ("needs a shader caret", 3 words against the
5-word budget) so the row deletion is safe by construction. The `help_marker` long-form
destination is the one the report validated: "their long form moves to the Help panel's copilot
section, which the prose gate exempts as documentation".

**P11 — FEASIBLE (one change: the amended rule needs a home inside the repo). CLOSED.**

Covered under F8: the rule's home is `conventions.md`, in the repo's "we decided X; revisit if Y"
form.

### 1.3 The feasibility report's false-trail list

All ten are folded or retired. 1 (the footgun) → §9.4 rewritten, P9 reduced. 2 (no user-facing
Recover) → 10.1/4 and P4. 3 (the lib delete is a trash move) → 10.1/4 and P4. 4 (five call sites)
→ 10.5's P3 bullet. 5 (Projects already `InlineInput`) → 10.5's P5 bullet. 6
(`toggle_copilot_open` calls `focus_copilot`) → P8 dropped. 7 (`command_menu_item` cannot live in
`ui_primitives.py`) → P1's `menus.py`. 8 (a glob misses the lib picker) → P6's gate. 9
(`test_import_dialog.py:63-67`) → P7's gate clause.

10 (**the seven/eight miscount**) — revision 2's 10.1/2 now reads "Seven commands (Save,
Next/Previous pass, Cycle code tab, the four Focus-tab chords) have no mouse-reachable home at
all." Counting the parenthetical: Save (1), Next pass (2), Previous pass (3), Cycle code tab (4),
plus four Focus-tab chords (5-8) = eight items under a count of seven. **The arithmetic is
unchanged from revision 1, and the report asked for a recount.** I am recording this as a
false trail rather than an open item for two reasons given in §4.1: the phrase "Next/Previous
pass" is one slashed item in the prose, so the list reads as seven written entries; and the
report's own substantive half of the point (that `NEXT_PASS`/`PREV_PASS` reach `app.choose_output`,
which a tile click already does) argues the count should go *down*, not up — which would make
seven right for a different reason. Neither reading changes any proposal: P1 gives every
`in_menu` command a home regardless of the count, and no gate or spec turns on the number. Worth
one word at spec-drafting time, not a blocking gap.

---

## 2. New-text audit — every factual claim in revision 2, verified

### 2.1 `confirm_menu_item` as `begin_menu` + one `menu_item_simple` in error color

P4: "`begin_menu(label)` with the one item `confirm_label` drawn in `STATE_ERROR` text, so the
second click is a hover-and-click inside the same open menu — no armed state, no reopen, no
modal, no `PopupState` change."

**Verified by probe.** A `begin_menu` submenu inside a `begin_popup_context_item`, driven with
synthetic mouse events on the `app` fixture's live context (the `_imgui_frame` shape from
`tests/test_pass_verbs.py:392`, with the window pinned via `set_next_window_pos` /
`set_next_window_size` per the feasibility report's own method note):

```
f4  popup=True  sub=False  menu_rect=(163.0, 113.0, 171.0, 129.0)     mouse=(158,91)
f5  popup=True  sub=False  menu_rect=(163.0, 113.0, 270.0, 129.0)     mouse=(167,121)   <- hover only
f6  popup=True  sub=True   clicked=False  sub_rect=(274.0,113.0,...)  mouse=(216,121)
f7  popup=True  sub=True   clicked=False                              mouse=(278,121)
f11 popup=True  sub=True   clicked=True                               mouse=(338,121)
f12 popup=False sub=None                                              mouse=(338,121)
CLICK_FRAMES [11]
```

Three facts, each demonstrated:

1. **The submenu opens on hover alone.** Between f5 and f6 the only event delivered was
   `add_mouse_pos_event` onto the `Delete` label's rect — no button event. `sub` flips
   `False -> True`. (The `begin_menu` label's rect is also wider once measured, f4's 171 vs
   f5's 270 — the first frame's rect is pre-layout, which is why the aim uses the previous
   frame's rect.)
2. **The item inside registers a click.** `menu_item_simple("Confirm delete")` returns `True` at
   f11, after a `mouse_button_event(0, True)` / `(0, False)` pair on its rect.
3. **The whole popup closes on that click** (f12 `popup=False`), so the "no reopen" claim holds:
   the confirm ends the interaction rather than leaving the context menu up.

Positive control: the same harness with no submenu opened the context popup at f4 and drew a
plain `menu_item_simple`, so the rig demonstrably delivers both the right-click and the pointer.

The design's own premise — "the second click is a hover-and-click inside the same open menu" — is
therefore accurate, and the "no `PopupState` change" half is true by construction (a `begin_menu`
is not a `PopupState` member). `COLOR.STATE_ERROR` exists at `theme.py:167`; both hand-rolled
pushes P4 retires (`tree.py:166`, `:279`) already push exactly that token, so the primitive
inherits their color rather than introducing one.

### 2.2 `open_current_document_dir` generalization

P2: "`Open folder` on a document generalizes `App.open_current_document_dir` to
`open_document_dir(document_id)` (three lines; the current-document verb calls it)."

**Correct.** `app.py:1890-1905`:

```python
def open_current_document_dir(self) -> None:
    if not self.current_document_id:
        logger.warning("No document selected")
        return
    document_dir = self.paths.documents_dir / self.current_document_id
    ...
```

The only use of `self.current_document_id` is the guard and the path join, so lifting it to a
parameter and leaving a one-line wrapper is the described change. The one existing caller is
`tabs/code.py:798` (`app.open_current_document_dir()`), which the wrapper keeps working. Nothing
named `open_document_dir` exists yet — revision 2 presents it as new, correctly.

### 2.3 The `InlineInput` promotion's import consequences

P5: "`InlineInput` is promoted from `editor_types.py` to `ui_primitives.py` (the conventions
trigger: a second multi-inline-input surface; this is the third — and `file_ops.py` already
imports imgui through `theme`, so the move costs no new import)."

**All three sub-claims correct.**

- The conventions trigger exists verbatim at `conventions.md:558-559`: "Revisit if a second
  multi-inline-input surface lands (promote `InlineInput` to `ui_primitives.py`)."
- Three importers today: `shader_lib/file_ops.py:17`, `popups/lib_picker/tree.py:15`,
  `popups/projects.py:19`. "This is the third" is right.
- The import cost. Probed:

```
THEME_PULLS_IMGUI True
EDITOR_TYPES_PULLS_IMGUI False
UIPRIM_PULLS_APP False
```

  `file_ops.py:21` is `from shaderbox.theme import COLOR`, and importing `shaderbox.theme` puts
  `imgui_bundle` in `sys.modules`. So `file_ops` already pays the imgui import, and the promotion
  costs it nothing — exactly as written, and the refinement the feasibility report asked for
  ("worth one line in the spec so a future reader does not re-litigate it") is that line.
  `editor_types` itself is imgui-free, which is the fact that made the question worth asking;
  after the move it stays imgui-free and simply loses the class.
- The direction that would have blocked it is also clear: `ui_primitives.py` does not import
  `App` (the two `App` occurrences in it are docstring prose at `:96` and `:324`), so hosting a
  `Path`-only dataclass creates no cycle.

### 2.4 The `was_settings_open` latch

P7: "the `was_settings_open` latch at `hotkeys.py:366` / `:398-399` goes, or the apply runs twice".

**Correct on all three counts — the line numbers, the mechanism, and the consequence.**
`hotkeys.py:366` is `was_settings_open = app.popup_state == PopupState.SETTINGS`, immediately
under the comment "Editor settings apply at the one close funnel, not per-edit while the modal is
open." `hotkeys.py:398-399` is `if was_settings_open:` / `app.apply_editor_settings()`, at the very
end of `_handle_escape` after the dispatch chain. If `close_popup()` takes over and itself calls
`apply_editor_settings`, the latch fires a second apply on the same frame — so "goes, or the apply
runs twice" is the accurate description.

This also corrects a round-1 false trail cleanly. The brief's §3 listed "P7 breaks the Settings
apply-on-close" as a *non*-finding, on the ground that the latch is "a separate mechanism from the
four carve-outs P7 removes". That is true of revision 1's P7 (which did not touch the latch) and
revision 2 does not contradict it — revision 2 moves the apply into the funnel deliberately and
deletes the now-duplicate latch. The feasibility report's coupling 1 said the same. No conflict.

### 2.5 The `menu_enabled` predicate's two rejections

P1: "Not the cheatsheet's `_is_active` (it reads `editor_focused`, which the menu click has just
cleared) and not `spec_eligible` (it rejects chord `0`, and Import passes is unbound)."

**Both rejections check out.** `widgets/cheatsheet.py:17-22`:

```python
def _is_active(scope: CommandScope, app: App) -> bool:
    if scope == CommandScope.EDITOR:
        return app.editor_focused
```

— so an EDITOR-scope item evaluated while the user is clicking a menu reads `editor_focused`,
which the click has moved. `hotkeys.py:328-338` `spec_eligible` opens with
`if chord == 0 or chord in app.editor_consumed_chords: return False`, and
`commands.py:228` is `CommandSpec(CommandId.IMPORT_PASSES, "Import passes", 0, C.TOOLS)` — chord
`0`, unbound, as claimed. Revision 2's substitute (`app.active_tab is not None` for EDITOR,
`app.is_copilot_open` for COPILOT, always for GLOBAL) also dodges `_is_active`'s
`not app.any_popup_open()` GLOBAL branch, which the feasibility report flagged as the coupling
that would grey out every File item.

`CATEGORY_ORDER` is real (`commands.py:61-66`) and holds the five categories P1 names in that
order (File, Document, Editor, View, Tools).

### 2.6 `tests/test_import_dialog.py:63-67` and `test_every_popup_state_has_a_draw_call`

P7 describes the first as "a source-substring test on `hotkeys.py`". **Correct.** Lines 62-67:

```python
def test_escape_reaches_the_close_funnel(app: Any) -> None:
    # Verification 15's wire: the bare `popup_state = CLOSED` fallthrough would leave the draft
    # populated. Falsifier: delete the IMPORT_PASSES branch from `_handle_escape`.
    source = Path("shaderbox/hotkeys.py").read_text(encoding="utf-8")
    assert "PopupState.IMPORT_PASSES" in source and "close_import_passes()" in source
```

Both asserted strings live in `_handle_escape` today (`hotkeys.py:377-378`) and both move to
`app.close_popup` under P7 — so the test fails as written unless repointed, and "made structural"
is the right upgrade for a substring assertion.

`test_every_popup_state_has_a_draw_call` is at `tests/test_project_management.py:636` — revision 2
names it without a line number, and the name resolves uniquely. Its body parses `ui.py`'s AST for
calls to names imported from `shaderbox.popups` and ends in
`assert len(called) == len(PopupState) - 1`. Revision 2's claim that the new clause must go red
"while the old count stays green" is exactly right: a member with a draw call but no close branch
leaves `len(called)` correct, so the existing assertion cannot catch it — which is why the break
must be demonstrated on the new clause specifically. The test's own docstring already warns about
the substring version a comment satisfied, so the structural framing matches the file's habit.

### 2.7 The S5 clause revision 2 cites for keeping the group prompt a popup

P5: "093 S5's Delete-key gate relies on it being one: an inline row in the canvas child would be
hovered, and only `is_any_item_active` would stand between a Delete typed into the field and an
unwire".

**Correct, and the citation is close to verbatim.** `01_spec.md` S5 (lines 228-232):

> Two facts about the gate's clauses, so no one narrows it: the group-name prompt is a plain
> `begin_popup`, for which `app.any_popup_open()` is False and `is_window_hovered(child_windows)`
> is already False, so a Delete typed into it is refused by `hovered` before `is_any_item_active`
> is consulted

The gate's five clauses are `pressed and hovered and not any_item_active and not blocked and
has_wire`, tested as the pure predicate `graph_state.delete_allowed(...)`. Moving the prompt
inside the canvas child makes `hovered` True, which removes the first refusal and leaves
`not is_any_item_active()` as the only guard — which is precisely what revision 2 says. S5's own
next sentence confirms the residual risk is real and narrow: "The one reachable state where
`not is_any_item_active()` is the clause doing the work is a text input ACTIVE in another window."
The spec's verification table at `01_spec.md:572` pins this behaviour ("S5: Delete typed into the
group prompt is refused … the clause it exercises is `hovered`"), so the move would also invalidate
a checked-in behaviour pin — a cost revision 2's decision avoids.

### 2.8 Remaining code claims in the new text, spot-checked

| Claim | Where | Verified |
|---|---|---|
| `pass_list.py:72-73` carries `enabled=deletable` plus a `and deletable` guard justified by a footgun comment | §9.4, P9 | yes — `pass_list.py:71-75`, comment "menu_item_simple can still register a click while disabled on this imgui-bundle build" |
| `tree.py:358` has `enabled=has_editor` and no Python guard | §9.4 | yes — `tree.py:358-360`, `has_editor` computed at `:356`. (Note this retires the brief's F7 sub-claim of a *missing* `and has_editor`: the `enabled=` is present; only the redundant callback guard is absent, which is what §9.4's "no — and none is needed" says.) |
| the two hand-rolled red pushes at `tree.py:165-167` / `:279-281` | P4 | yes — `push_style_color(imgui.Col_.text, COLOR.STATE_ERROR)` at `:166` and `:279`, each around a `menu_item_simple` whose label flips on `is_armed` |
| `file_delete_armed` / `dir_delete_armed` live on `ShaderLibFileManager` | P4 | yes — `shader_lib/file_ops.py:55-56` |
| a document delete moves to the project trash | 10.1/4 | yes — `project_session.py:487-492`, `shutil.move(...documents_dir / document_id, dest)` into `paths.trash_dir` |
| a pass delete drops the entry, its wiring, its position and every downstream sampler's source; the file stays | 10.1/4, P4 | yes — `project_session.py:980-999`: `passes.pop(name).release()`, `drop_feedback`, `forget_pass_sources` ("Every sampler that named it goes back to undecided"), `_graph_without`; no `unlink`, no `move` |
| `App.document_delete_armed`, `set_document_delete_armed`, the handler cleanup at `app.py:791-792`, and the grid's three result branches | P3 | yes — `app.py:791-792` is the `if document_id == self.document_delete_armed:` cleanup; `document_grid.py:95-99` holds the three branches (`delete_armed` / `delete_confirmed` / `delete_cancelled`) |
| the pass tile already passes `deletable=False` | P3 | yes — `pass_list.py:151` |
| `settings.py`'s `is_keep_opened` is the one inverted name | 10.1/6, P6 | yes — `settings.py:190,192,194`, local only |
| the revert confirm closes itself inside its body | 10.1/6, P6 | yes — `copilot_chat.py:197-202`, two `imgui.close_current_popup()` calls inside the `modal_window` body, no return |
| the emoji picker has no Esc carve-out and leaves `emoji_pick_target` dangling | 10.1/7, P7 | yes — `_handle_escape` has carve-outs for PASS_SETTINGS, IMPORT_PASSES, PROJECTS, SHADER_LIB_PICKER only; `emoji_pick_target` is nulled at `emoji_picker.py:23` (the button path) and set at `app.py:1264` |
| the one existing hint is `popups/lib_picker/__init__.py:116` | 10.1/11, P11 | yes (§9.3's anchor, unchanged in revision 2) |
| `tests/test_command_registry_coverage.py` pins spec labels to the help snippet | P1 | yes — `test_every_bound_spec_reaches_the_help_shortcuts` asserts `spec.label in snippet` for every bound spec |
| the prose gate's `_OVER_BUDGET` / `_EXEMPT` machinery re-arms on a deleted row | P10 | yes — `_EXEMPT: set[...] = set(_OVER_BUDGET)` at `:582`, `_budgeted()` filters by `(module, function, words)` |
| `menu_item_simple(label=)` is scored at 4 words with a written reason | §9.6 context for P1's gate row | yes — `_IMGUI_ROWS` entry at `:117` |
| the nine modals / `PopupState` mutex | 10.3 | consistent — eight non-CLOSED enum members (`app.py:135-143`) plus the revert modal, whose state is `copilot_revert_target` outside the enum |

**No claim in the new text is wrong.** The one number I would flag for the spec is the "seven
commands" count carried over from revision 1 (§1.3 item 10 above) — a miscount in the prose, not a
claim about the code, and it drives nothing.

---

## 3. The two explicit rejections

### 3.1 F8's "the skill lives outside the repo" — revision 2 is right to reject it

`.claude/skills/imgui-ui/SKILL.md` exists on disk and is tracked:

```
$ git ls-files .claude/skills/imgui-ui/SKILL.md
.claude/skills/imgui-ui/SKILL.md
```

It is a repo artifact, so a spec in this repo can commit to editing it, and the global rule F8
invoked ("a repo stands alone — never reference the harness from inside it") does not apply: this
skill is not the harness, it is the project's own checked-in rulebook. The inventory has been
citing it as `.claude/skills/imgui-ui/SKILL.md` throughout §9, which is a repo-relative path.

Revision 2 does not use the rejection to skip F8's substance. The rule still gets a
`conventions.md` home in the repo's "we decided X; revisit if Y" form, and the skill is amended
*as well* ("mirrored in the repo's own `.claude/skills/imgui-ui/SKILL.md` §7.4"). That is the
stronger outcome: `conventions.md` is what the cold-start chain reads before changing anything,
and the skill is what `/imgui-ui` loads at the start of UI work — leaving the skill's §7.4 saying
"show a one-line 'Right-click for actions' hint above the list" unqualified would keep prescribing
a hint the pass has just decided against for three surfaces. The same reasoning applies to P9's
amendment of §7.4's `enabled=` bullet, which is the other half of the same file.

### 3.2 The feasibility report's P9 premise reversal — revision 2 agrees with the probe, correctly

The feasibility report's §0.1 showed `menu_item_simple(enabled=False)` does not register a click
on imgui-bundle 1.92.801 (`SAME_ITEM enabled_clicked=True disabled_clicked=False`, plus
`BEGIN_DISABLED_MENU_ITEM_CLICKED False`, with a positive control), contradicting
`pass_list.py:72-74`'s comment, §9.4's table and P9's justification. The report asked for a re-run
before acting.

Revision 2 accepts the probe and acts on it in both directions: §9.4 is retitled "no Python-side
gate is needed on this build" and records the reproduction ("reproduced by the main session with a
positive control in the same run"), and P9 shrinks to removing the redundant guard and correcting
the two documents that prescribe it. **This is the right call**, for two reasons beyond the
transcript:

1. The probe is falsifiable and was reproduced independently by the main session, with a positive
   control in the same run — which is what distinguishes a library fact from a harness artifact.
   The feasibility report's own §0.3 method note shows the authors knew that distinction and had
   already been burned by it once, so the reproduction was not ceremonial.
2. Revision 2 does not merely delete the guard, it fixes the two artifacts that would have
   re-created it. A comment saying "the build has a footgun" and a skill bullet saying "don't rely
   on `enabled=`" are the mechanism by which the redundant guard spreads; leaving them while
   removing one instance is the shape that regresses. P9 names both, and names the build the
   measurement was made on — which is the clause that keeps the correction honest if the bundle is
   ever bumped.

The one thing the rejection does *not* do is claim the guard was harmful. It was not: §9.4 calls
the `and deletable` guard redundant, and 10.1/9 says "§9.4's table sorts two sites by a guard that
guards nothing". Both readings are accurate.

---

## 4. Balance and settled decisions, on revision 2 only

### 4.1 State, surfaces, modals, persisted values — against "don't overblow the state"

The maintainer's brief (finding 17): "good coverage, a very convenient ui/ux … don't overblow the
state … the same context menu for the node and for the pass … perform all neccessary refactorings
and unifications".

| Proposal | Adds | Removes | Refusable under "don't overblow"? |
|---|---|---|---|
| P1 | two `CommandSpec` fields (`in_menu`, `separator_before`); `menus.py` with `command_menu_item` + the bar; `command_label` in `commands.py`; `menu_enabled` | thirteen hand-written `imgui.menu_item` calls in `ui.py`; two label drifts | no — two defaulted dataclass fields against thirteen hand-written items, and seven commands gain their first mouse home |
| P2 | one popup on the documents grid; `open_document_dir(document_id)` | nothing | no — one surface, and it is the one the brief's "the same context menu for the node and the pass" generalizes to. The editor-tab menu (two new verbs) is gone. |
| P3 | nothing | `App.document_delete_armed`, `set_document_delete_armed`, the `app.py:791-792` cleanup, the grid's three result branches | no — pure subtraction, two App fields |
| P4 | one `ui_primitives` function, no state | `file_delete_armed`, `dir_delete_armed`, the label flip, two hand-rolled red pushes | no — **no state field, no modal, no `PopupState` member**, and it removes two arming fields |
| P5 | nothing net | `group_prompt: bool` + `group_name: str` collapse to one `InlineInput` on the same `GraphViewState`; `InlineInput` changes module | no — a reduction on the same object |
| P6 | one test file | one inverted name, three missing spacers, one right-anchor | no |
| P7 | `App.close_popup() -> bool`; `close_emoji_picker` | four Esc carve-outs; the `was_settings_open` latch | no — one dispatch for four forgettable per-modal steps, net one fewer field |
| P8 | nothing | nothing | n/a — dropped |
| P9 | nothing | `and deletable` and its comment; two wrong doc bullets | no — pure subtraction |
| P10 | nothing | twelve over-budget allowlist rows | no |
| P11 | one caption string; one `conventions.md` bullet; one skill amendment | nothing | no — one string |

**No proposal adds a state field, a modal, a `PopupState` member, or a persisted value.** The net
on `App` is negative: `document_delete_armed` and `set_document_delete_armed` go (P3), the
`was_settings_open` local goes (P7), `file_delete_armed` / `dir_delete_armed` go from
`ShaderLibFileManager` (P4), and the only additions are two defaulted fields on a frozen-ish
registry dataclass (P1) and one method (P7's `close_popup`). One new surface exists in the whole
pass — the documents-grid context menu — and it is the surface the brief's own finding-15 pattern
("a verb lives on its object's context menu; a tile carries no button") points at.

The two proposals the brief's round-1 report called oversized are both shrunk: P9 from 23 call-site
rewrites to two deletions and two doc corrections, P5 from a popup-to-inline-row conversion to a
commit-rule change. The remaining large diff is P1, and it is the brief's "all necessary
refactorings and unifications" in the most literal form available — one table already exists and
two surfaces already render from it.

One thing worth the maintainer's eye at spec time, not a balance failure: P1 puts roughly 25 items
into the menu bar across five categories. That is coverage the brief asked for, but it is also the
single largest visible change in the pass, and `in_menu` is the dial. Revision 2 decided the dial's
five `False` values with a stated reason (§7.4's two-affordances rule), so the decision is made and
reversible, not deferred.

### 4.2 Settled decisions — is any still bent silently?

| Decision | Touched by | Named? |
|---|---|---|
| 092 D14 (`Group...` is node-only, seeded from `view.selection`) | P2 | **yes** — cited in the table row with its structural reason, and again in 10.5 |
| 092 D16 (a member is deleted with the strip's two-click arm) | P4 | **yes** — "Reverses 092 D16's 'the strip's two-click arm' (already gone with W3-2)", the W3-5 pointer shape |
| conventions `InlineInput` bullet ("revisit if a second surface lands") | P5 | **yes** — the trigger is quoted and the count given ("this is the third") |
| conventions `PopupState` / new-modal bullet | 10.3, P4 | **yes, by not applying** — P4's confirm is a submenu, so the bullet's obligations (enum member, `open_*`, self-close, `draw_*` call) do not attach; 10.3 says so explicitly |
| conventions three-layer UI architecture | P1 | **yes** — "`ui_primitives.py` is `App`-free by the three-layer rule and cannot host it" |
| skill §7.4 discoverability hint | P11 | **yes** — named as an amendment, with the new rule's text and its revisit trigger |
| skill §7.4 `enabled=` bullet | P9 | **yes** — named as wrong for the pinned build, with the build stated |
| skill §7.4 two-affordances rule | P1 (`in_menu`), P2 (editor tab), 10.3 (favorite star) | **yes** — cited in all three places, including the carve-out that saves the star |
| 093 S5 (the canvas Delete gate) | P5 | **yes** — cited as the reason the prompt stays a popup |
| 092 D10 (the canvas menu's items) | P2 | preserved, unchanged |
| 093 S15 / W3-3 (no click opens a shader tab) | P2 | preserved — `Open shader` heads the item set |
| `conventions.md` no-backward-compatibility rule | — | not engaged; nothing in the pass reshapes a persisted model |

**Nothing is bent silently.** Every decision the pass touches is named at the point of the change,
and the two that round 1 caught as silent (D14, D16) are the two that gained explicit pointers.

### 4.3 Is 10.4's one remaining fork genuinely the maintainer's?

10.4:

> **A pass `Delete` with a confirm submenu (P4).** Today the shared menu's Delete fires at once
> (the tile's two-click arm went with W3-2). P4 puts the submenu on it because the loss (the
> wiring, the position, every reader's source) has no undo and a wired pass in a six-node graph
> is minutes of work; the cost is one hover inside the menu. If he prefers the immediate delete,
> the pass row uses a plain `menu_item` and the toast names what was lost.

**Yes.** Round 1's brief called exactly this "genuinely the maintainer's" and the other two forks
decidable-by-the-pass; revision 2 decided those two (Option A on the sticker grid, `in_menu` on
the menu-bar scope) and kept this one. The test the brief applied — "a call that follows from a
stated premise is not a fork" — does not dispose of this one: nothing in the pass's premises
settles whether a rarely-fired irreversible verb is worth one hover. Both branches are fully
specified and cheap to switch between (a `confirm_menu_item` call versus a `menu_item_simple` call
at one site), and the loss is described concretely enough for the maintainer to price it. The
question is taste about his own workflow, which is the definition of his call.

One observation, not an objection: the fork's framing is slightly narrower than its blast radius.
P4 applies `confirm_menu_item` to three verbs (pass delete, document delete, lib file/dir delete),
and 10.4 asks only about the pass. If the maintainer answers "immediate", the pass should say
whether the document and lib deletes follow — the lib ones currently confirm twice, so "immediate"
there is a two-step reduction, while the document one has no undo at all. A sentence at spec time,
not a gap in the design.

---

## 5. False trails — what looks like a finding in revision 2 and is not

1. **"P4 still adds a primitive to `ui_primitives.py`, so P3's 'pure subtraction' and P4 are in
   tension."** They are not. `confirm_menu_item` is a free function with no state and no `App`
   parameter; `ui_primitives.py` is where the rulebook's shared draw helpers live, and adding one
   there is the opposite of the tier violation §9.5 flags. Revision 2's balance claim survives.

2. **"`command_menu_item` in a new `menus.py` is a module added to dodge a rule."** The
   three-layer bullet (`conventions.md:330-344`) forces it: a function taking `app: App` cannot
   live in an `App`-free module, and `ui.py` already carries the frame loop plus layout. Revision 2
   names the alternative it rejected and why. The feasibility report reached the same place
   independently.

3. **"P7's `close_popup` duplicates `_handle_escape`'s job, so one of them is dead."** No —
   `_handle_escape` keeps the `rebinding_command` early return, the `escape_has_job` gate, the
   revert-target branch, the palette branch and the chat-focus branch. Only the popup dispatch
   moves. Revision 2 says "the `rebinding_command` early return stays where it is", which is the
   clause that makes this explicit.

4. **"The `and deletable` guard removal is a behaviour change that could let a last pass be
   deleted."** It cannot. `enabled=deletable` already suppresses the click on this build (probed),
   and `project_session.delete_pass` refuses independently: `if len(document.passes) == 1: return
   "a document needs at least one pass"`. Three layers become two, and the load-bearing one is
   untouched.

5. **"P3 breaks `tests/test_import_dialog.py:123-124`, which passes `armed` positionally."** Only
   under option B. Revision 2 decided option A, which keeps the `armed` parameter for the sticker
   grid, so the positional signature does not shift. The feasibility report raised this against a
   version of P3 that revision 2 did not adopt.

6. **"10.3's 'nine modals' contradicts `PopupState`'s eight non-CLOSED members."** The ninth is
   the revert confirm, whose state is `app.copilot_revert_target` rather than an enum member —
   P6's own text says so ("the revert confirm returns `keep_open` through `modal_window` like the
   other eight"). Eight plus one is nine, consistently.

7. **"P11 leaves the node without a hint although finding 14 is a maintainer who could not find
   the node menu."** This is a real UX tension and the brief argued it well, but it is a judgement
   the pass is entitled to make, and revision 2 makes it with a revisit trigger naming the exact
   evidence that would reverse it ("revisit if a walk finds a menu undiscovered"). A settled
   decision with a falsifier is not a silent bend. Recording it here so it is not re-litigated as
   an open finding.

8. **"The 'seven commands' count is eight."** It is arguably a miscount (§1.3 item 10), but it
   drives no proposal, no gate and no number in a spec, and the feasibility report's own second
   half argues the figure should shrink rather than grow. Worth one word at spec-drafting; not a
   finding against the design.
