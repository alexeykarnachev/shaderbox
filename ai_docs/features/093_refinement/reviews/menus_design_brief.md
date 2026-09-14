# Review — `04_menus_inventory.md` §10 (the design pass) against the brief

Scope: section `## 10. Design pass` only (lines 506-697). Sections 1-9 are treated as verified
inventory; where a proposal turns on one of their facts, the code was opened and the read is
quoted below.

Anchors used, all external to the proposal's author: finding 17 (and 13-15) of `00_findings.md`;
`/imgui-ui` SKILL.md §1, §2, §7; `conventions.md ## Design decisions`; 092 `03_spec.md` D10/D12/
D14/D16; 093 `01_spec.md` S15 and W3-1..W3-6.

**Verdict: PARTIAL.** The pass is well-grounded — nine of the eleven proposals trace to an
inventory observation with a real anchor, the brief's "same context menu for the node and the
pass" is generalized correctly, and 10.3 is an honest not-touched list. But P4 rests on a code
claim that is false, P2 quietly introduces three verbs the app does not have, P5 and P9 are
refactors whose UX payload is small against their blast radius, and two settled decisions are
bent without the pass saying so. 12 findings, 3 of them blocking.

---

## 1. Per-proposal verdicts

### 1.1 Does each proposal follow from a supported 10.1 observation?

| # | Traces to | Inventory line that supports it | Verdict |
|---|---|---|---|
| P1 | 10.1/2, 10.1/3 | §7.1's table + "the menu bar is five hand-written entries"; the label drifts are all in the cross-surface table ("`add pass` (tab button, lowercase) vs. `Add pass` (canvas menu, capitalized) — **label-case drift**") | **supported** |
| P2 | 10.1/1 | §3 "`pass_menu_items` (the shared item list — NOT itself a popup, the caller owns `begin_popup_context_item`)"; §1 for the document grid having no menu | **supported for pass/group; the document and editor-tab rows are NOT** (F2) |
| P3 | 10.1/1, finding 15 | §8 "Pass-tile corner delete-✕ … **No glyph drawn — not a control**"; §8's document-tile and sticker-cell rows | **supported** |
| P4 | 10.1/4 | "Four codings of 'confirm a destructive verb'" + §3's armed lib-tree label flip | **premise false** (F1) |
| P5 | 10.1/5 | §9.1 "new-name input: Enter commits (`:168`), but **deactivate does NOT** … This is a genuine §7.5 divergence"; and the `##graph_group` line under the table | **supported** |
| P6 | 10.1/6 | §9.1's table: `is_keep_opened` at `settings.py:190`; "no spacer" on Examples/Help/Lib Picker; "Close **right-anchored**, the only modal that does" | **supported** |
| P7 | 10.1/7 | §9.2 "Emoji Picker — **no** … a bypass-driven Esc-close leaves the callback dangling" | **supported** |
| P8 | 10.1/8 | §7.1's `TOGGLE_COPILOT` row: "two different `App` methods for the same visible effect" | **supported, but the fold is not behaviour-preserving** (F5) |
| P9 | 10.1/9 | §9.4's two-row table: the lib tree's Insert "**no** — `insert_name(app, fn)` fires with no Python guard" | **supported for the fix; the gate is the over-reach** (F7) |
| P10 | 10.1/10 | §9.6's four bullets, each with a measured word count | **supported** |
| P11 | 10.1/11 | §9.3 "Exactly **one** hint exists anywhere in the app" | **supported as an observation; the rulebook amendment is not the pass's to make** (F8) |

Observation 10.1/11's second clause — "the maintainer found the node menu himself" — is a
restatement of finding 14, which is in `00_findings.md`, not the inventory. It is true but it is
evidence for the *opposite* of P11's conclusion: a menu the maintainer could not find is the case
for a hint on the node, and P11 gives the node none.

### 1.2 Balance — every addition, against "don't overblow the state"

| # | Adds | Removes | "Don't overblow" met? |
|---|---|---|---|
| P1 | one spec field (`separator_before: bool`); one primitive (`command_menu_item`); one helper (`command_label`) | every hand-written `imgui.menu_item` in `ui.py` (13 call sites read at `ui.py:717-751`) | **yes** — one field and two functions against thirteen hand-written items and two label drifts |
| P2 | **three App verbs that do not exist**: a per-document "Open folder", an editor-tab "Open folder", an editor-tab "Close others"; two new popups (documents grid, editor tab bar) | nothing | **no** — F2 |
| P3 | nothing | `armed`, `deletable`, the corner ✕, the `Delete?` wash, `delete_*` results, `cell_delete_confirm`, `close_cross_button`, `App.document_delete_armed`, `set_document_delete_armed` | **yes** — pure subtraction, the cleanest proposal in the pass |
| P4 | one primitive (`confirm_modal`); one modal surface reachable from the lib tree | the armed-label flip, two hand-rolled red pushes | **no** — F1: the surface it adds guards a verb that is already recoverable |
| P5 | one primitive (`name_prompt_row`); `InlineInput` moves module | the `##graph_group` popup; one divergent commit rule | **yes on state, borderline on churn** — F6 |
| P6 | one gate file (`tests/test_modal_chrome.py`) | one inverted name, three missing spacers, one right-anchor | **yes** |
| P7 | one `App.close_popup()` dispatch table; one new close verb (`close_emoji_picker`) | four Esc carve-outs | **yes** — it trades four forgettable per-modal steps for one dispatch a test counts |
| P8 | one parameter (`focus_aware: bool`) | one `App` method | **yes on the count, no on behaviour** — F5 |
| P9 | one primitive (`menu_item`); one gate clause in `test_button_tiers.py`; **23 call-site rewrites** (`grep -c menu_item_simple shaderbox/` = 23) | the `enabled=` footgun at every site | **borderline** — F7 |
| P10 | nothing | four over-budget allowlist rows; ten long help markers | **yes** |
| P11 | one caption string on the documents grid; one amendment to a skill outside the repo | nothing | **no on the skill edit** — F8 |

**Refactor for its own sake?** None is purely gratuitous, but two are weighted wrong:

- **P9** is the clearest case. Its *UX* payload is one bug: the lib tree's "Insert at caret" can
  fire with no editor target (§9.4). That is a two-line fix (`and has_editor`, the shape
  `pass_list.py:75` already uses). P9 instead routes 23 call sites through a new primitive and
  adds a gate. The gate is defensible under "a rule with no gate is a wish" — but the brief asked
  for "all necessary refactorings", and 23 sites is the definition of a wave that is not necessary
  to close the defect it names. The `danger=True` half is genuinely new UX (it is what replaces
  the two hand-rolled red pushes §9.5 flags), so P9 is not empty — it is oversized.
- **P5**'s `InlineInput` promotion is correct and its trigger fires by the conventions bullet's
  own words. Its second half — "the group prompt stops being a floating popup" — is a behaviour
  change the observation does not require. 10.1/5 is about *commit semantics*, not about whether
  the prompt floats. See F6.

### 1.3 Settled decisions reversed or bent

| Decision | Where | Proposal | Says so? |
|---|---|---|---|
| **092 D14** — "Right-click with a selection adds `Group...` to the node menu" | 092 `03_spec.md:268-277` | **P2** moves `Group...` from the node-only branch into the shared item set, so the strip tile gets it | **NO — silent.** P2 says "`Group...` moves from the node-only branch into the shared set so the strip tile gets it too", which describes the move but never names D14 or the reason D14 put it on the node. D14's reason is structural: `Group...` seeds from `GraphViewState.selection`, a canvas-only concept the strip has no equivalent of. P2 says "it seeds the selection with the one pass, as the node does" — which means the strip's Group writes to a *graph view state* the strip does not display. **Finding F3.** |
| **092 D16** — "a member is deleted from its own node menu with **the strip's two-click arm**" | 092 `03_spec.md:293-296` | **P4** removes every arm from the pass delete: "Both fire on the menu click with a notification" | **NO — silent.** D16 is quoted nowhere in §10. It is arguably already superseded (W3-2 deleted `App.pass_delete_armed` and its smoke frames, so the two-click arm is gone from the tile), but a decision that was reversed by an earlier wave needs the pointer, per the repo's own habit — W3-5 is written as "**Reverses 092 D1's 'moved last' clause and D11's double ring**". **Finding F4.** |
| **`/imgui-ui` §7.4** — "Discoverability: show a one-line 'Right-click for actions' hint above the list" | SKILL.md §7.4 | **P11** amends the rule to "on a modal's list; a canvas or a strip with a visible primary click gets none" | **YES, explicitly** — P11 names the amendment. But the skill is fleet-level, outside the repo (F8). |
| **conventions `InlineInput` bullet** — "Revisit if a second multi-inline-input surface lands (promote `InlineInput` to `ui_primitives.py`)" | `conventions.md:551-560` | **P5** promotes it | **YES** — P5 quotes the trigger and argues this is the third surface. Correct. |
| **conventions `popups/*.py` / `PopupState` bullet** — "A new modal popup adds an enum member, its `open_*()`, a self-close to `CLOSED`, **and its `draw_*(app)` call in `ui.py`'s popup block**" | `conventions.md:453-466` | **P4** adds `confirm_modal` as a modal the lib tree opens; 10.3 claims "no modal is added or merged" | **Contradiction.** 10.3's first bullet says the nine modals and their mutex stay as they are and "no modal is added"; P4 adds a confirm modal reachable from inside the lib-picker modal — which is a **nested** modal, the very shape P4's own next sentence says is not the rulebook's ("a nested modal is not"). **Finding F9.** |
| **092 D10** — the canvas menu's items (Add pass, Import..., Fit, Arrange) | 092 `03_spec.md:196-197` | **P2** keeps them and routes two through `command_menu_item` | **no change** — correctly preserved |
| **093 S15 / W3-3** — no click of any count opens a shader tab; the menu's `Open shader` is the gesture | 093 `01_spec.md` W3-3 | **P2** keeps `Open shader` heading the item set | **no change** — correctly preserved |

### 1.4 Coverage of the brief

The brief's phrases, each against the pass:

- **"good coverage"** — P1 is exactly this: seven commands with no mouse home (§7.1: Save, Next/
  Previous pass, Cycle code tab, the four Focus-tab chords) gain one. **Covered.**
- **"a very convenient ui/ux"** — P2, P3, P11. **Covered, with the reservations below.**
- **"don't overblow the state"** — P3 is a net removal of two App fields; P1 adds one spec field.
  **Covered except P2** (F2).
- **"the exact balance the user might need"** — 10.4's third call is precisely this question,
  handed back rather than answered. **See §1.5.**
- **"properly reuse the code (the same context menu for the node and for the pass)"** — P2's
  table is the direct answer, and it generalizes correctly from `pass_menu_items`. **Covered.**
- **"all necessary refactorings and unifications"** — P1, P5, P6, P7, P8, P9. **Covered, P9
  oversized.**
- **"do everything cleanly"** — P6 and P7 each ship a gate with the sweep. **Covered.**

**Gaps — surfaces in §1-8 the pass neither changes nor lists under 10.3:**

1. **The Telegram new-pack / delete-pack inline forms** (§2: "delete: hand-rolled red-tinted
   confirm child, NOT the shared `cell_delete_confirm`") and **`telegram.py:484-485`'s hand-rolled
   `push_style_color`** (§9.5). P4's last sentence says "The Telegram pack delete is out of scope
   (exporter panel)" — so the *delete* is dispositioned, but the hand-rolled red child as a **tier
   / theming violation** (§9.5 flags it separately from §2) is neither fixed nor listed in 10.3.
   **Finding F10.**
2. **The lib tree's inline favorite star** (§3: "the same verb as this menu's Favorite/Unfavorite
   item, reachable without opening the menu"). The rulebook §7.4 names this exact case — "Two
   affordances for the same thing on the same row is the slop signal" — and its own carve-out
   ("rows with a single common action — toggling a favorite — the inline star is fine"). The pass
   never rules on it. It is in neither §10.2 nor 10.3. **Finding F11.** (The right answer is
   probably "stays", by the rulebook's own carve-out — but it must be *said*, because P3's whole
   principle is "a tile carries no button" and a reader will ask why the star survives.)
3. **The `K`-lookup note and the completion popup** (§2) — non-interactive / library-owned, so
   plainly out of scope, but a reader has to derive that. Minor; folded into F11.
4. **The editor error strip** (§2, `code.py:329-374`) — clickable rows, one verb (jump), no menu.
   By §7.4's "rows where the primary click IS the action" it correctly gets none, but again the
   pass does not say so. Minor; folded into F11.
5. **The Examples modal's "no reset on open"** (§9.1: "`app_state.selected_example_id` persists
   across opens (deliberate, `examples.py:69`)") against rulebook §7.6 ("On open: reset transient
   state … A picker that re-opens showing the previous search reads as broken"). The inventory
   marked it *deliberate*; P6 covers chrome only and 10.3 does not mention it. **Finding F12.**

### 1.5 The three calls in 10.4

**Call A — Option A/B on the sticker grid.** *Should have been decided by the pass.* The two
options differ by one thing: whether `preview_cell` keeps its delete machinery for one caller.
The pass already made the equivalent call twice, and made it the same way both times — the
Telegram pack delete is "out of scope (exporter panel)" (P4) and the exporters' config panels are
"the exporters' own and outside this pass" (10.3). Applying the rule it already stated gives A
without a maintainer round-trip.
**Recommendation: A.** One reason: "this pass does not own the exporter panels" is already a
stated premise of this pass in two other places, and a call that follows from a stated premise is
not a fork.

**Call B — pass delete without a confirm.** *Genuinely the maintainer's.* The two readings lead
to materially different work: no-confirm is zero new code, confirm is `confirm_modal` plus a
call site plus a decision about whether the strip and the node both route through it. And the
question is a taste question about an irreversible-in-practice loss (see F1's second half: the
*entry, wiring and position* are gone with no undo, and the toast does not offer one).
**Recommendation: confirm.** One reason: deleting a pass drops its wiring and its graph position
with no undo path anywhere in the app, and a wired-up pass in a 6-node graph is minutes of work,
against a confirm that costs one click on a verb a user fires rarely.

**Call C — every command in the menu bar.** *Should have been decided by the pass.* The pass
already names the alternative and prices it ("an `in_menu` flag on the spec (default `True`),
which is one field and no design"), and it already added one optional spec field in the same
proposal (`separator_before`). Handing back a call whose alternative you have already costed and
whose mechanism you have already introduced is a deferral, not a fork.
**Recommendation: ship the `in_menu` flag, default `True`, and set it `False` on the four
Focus-tab chords and Cycle code tab.** One reason: those five are *view-focus* verbs whose only
honest home is the keyboard — a menu item "Document tab" under a View menu duplicates the tab bar
sitting two inches below it, which is the same two-affordances slop §7.4 names, while Save and
Next/Previous pass have no visible home at all and genuinely need one.

### 1.6 UX judgement

**P2's four item sets.**

- *pass* (Open shader · Settings... · ─ · Group... · Leave group · ─ · Delete). Reads well and
  matches every node-graph tool. One concrete interaction: **right-click a strip tile → Group...**
  — the strip does not draw a selection, so the user gets a name prompt for a group containing one
  pass, with no visual confirmation of what is in it, and the result is invisible on the strip
  (groups are a canvas concept: 092 D4 "Root: every ungrouped pass as a node and one box per
  group"). The verb works; the feedback does not. See F3.
- *group box* (Open · Dissolve). Correct, and matches 092 D16's "a box gets no Delete verb".
- *document* (Open · Open folder · ─ · Delete). The shape is right — the documents grid is the one
  surface a first-timer meets. "Open folder" is a real convenience (it is what `code.py:797`'s
  "Open dir" does for the editor). But the verb does not exist for an arbitrary document
  (`app.py:1890 open_current_document_dir` is current-document-only), which P2 does not flag.
- *editor tab* (Close · Close others · ─ · Open folder). "Close others" is a genuinely useful verb
  in a tab bar and does not exist (`app.py:1636` has `close_tab(index)` only). "Open folder" on a
  **lib** tab means the shader-library folder, not a document dir — the item is ambiguous for one
  of the four tab kinds (`shader` / `script` / `lib` / `graph`), and a `graph` tab has no session
  at all (093 T1). Unflagged.

**P4's destructive rule.** The rule itself — "recoverable fires and toasts, irreversible
confirms" — is the right axis, and it is the one a user can learn. Applied to the code as it
stands it lands wrong in two of its four cases.

*Concrete interaction — delete a pass by mistake.* Right-click a node, the menu opens, `Delete` is
the last item. Under P4 the click fires immediately. What happens: the `PassEntry` is popped, the
feedback history dropped, **every sampler that named it is reset to undecided**
(`project_session.py:997`: "Every sampler that named it goes back to undecided"), the graph entry
is rewritten, and `close_editor_for_path` evicts the tab. The `.glsl` file survives on disk — so
the pass's *code* is recoverable by re-adding a pass and pointing it at the file. Its *wiring*,
its *position*, and every downstream sampler's source are not: there is no undo, no trash entry,
and the toast P4 proposes ("Deleted pass blur") offers nothing to click. Recovery is: add a pass,
re-point it at the file, re-wire every consumer by hand, re-drag it into place. In a six-node
graph that is minutes. **This is not the "recoverable" half of P4's own dichotomy.**

*Concrete interaction — delete a lib file.* The lib tree's delete is P4's flagship example of the
irreversible half. It is not irreversible: `shader_lib/file_ops.py:222-250` moves the file into
`.trash/` and already pushes `"Deleted <name> - recoverable in .trash/"`. **P4's premise is
false**, and under P4's own rule the lib delete belongs in the *toast* branch — where it already
is, minus the armed flip. See F1.

*Concrete interaction — delete a document by mistake.* Under P4 this fires at once. It moves to
`paths.trash_dir` (`project_session.py:488-492`) — recoverable in principle. But the only Recover
affordance in the app is the **copilot chat's** card (`copilot_chat.py:617` → `app.py:810`,
reachable only from a `msg.recover` a copilot turn wrote). A user who deletes from the grid gets
no card and no undo — recovery is a file manager. P4 cites "the copilot's Recover card already
reads it" as if it were a general undo. It is not. See F1.

**P11's hint rule.** The result is defensible for the strip (the `add pass` / `import...` buttons
below it are a visible primary affordance) and for the canvas (right-click-empty-space is the
universal node-editor idiom). It is wrong for the **node**, and the pass's own evidence says so:
finding 14 is the maintainer — the app's author — reporting that he could not find the node's
context menu. *Concrete interaction:* a first-time user opens the graph tab, sees six cards, wants
to open one's shader. W3-3 removed the double-click. The click chooses the output. There is no
button on the card (W3-2). The only path is a right-click nobody told them about. P11's amendment
writes that gap into the rulebook as policy. A node does not need a caption — a one-line dim hint
in the canvas's own empty-state or a tooltip on first hover would do — but "gets none" is the
wrong answer to the one discoverability failure the feature actually recorded.

---

## 2. Findings

**F1 (blocking). P4's premise about the lib delete is false, and its two "recoverable" cases have
no user-facing recovery.** P4: "a lib file / directory delete is a real filesystem delete, so it
becomes a small confirm modal". The code (`shaderbox/shader_lib/file_ops.py:222-250`):

```python
def delete_file(self, path: Path) -> None:
    # Move into `.trash/` (basename + numeric suffix on collision).
    ...
    msg += " - recoverable in .trash/"
    self._notifications.push(msg)
```

`delete_dir` (`:252-300`) is the same, file by file. By P4's own rule both belong in the *fire and
toast* branch — the toast already exists and already says "recoverable". Meanwhile the two verbs
P4 puts in the fire-and-toast branch have no recovery a user can reach: a deleted pass loses its
wiring and position with no undo (`project_session.py:980-999`), and a deleted document's only
Recover affordance is the copilot chat card (`app.py:810`, called from `copilot_chat.py:617` on a
`msg.recover` only a copilot turn writes). **P4 currently has the rule exactly inverted on three
of its four cases.** Fix: apply the rule to the code as measured — lib delete fires and toasts
(dropping the armed flip and the red push, which is the actual §9.5 defect); pass delete confirms
(the wiring loss has no undo); document delete either confirms or the grid gains its own Recover
row, and the pass says which.

**F2 (blocking). P2 adds three App verbs the app does not have, and the proposal does not say
so.** P2's table lists document "Open folder", editor-tab "Close others", and editor-tab "Open
folder". Greps: `app.py:1890 open_current_document_dir` takes no document id; `app.py:1636
close_tab(index)` is the only tab-close verb; nothing named `close_other_tabs` exists. Each is new
`App` code with its own edge cases (a `graph` tab has no session, 093 T1; a `lib` tab's folder is
the library root, not a document dir). "Don't overblow the state" is about fields, and these add
none — but the brief also says "don't overblow", and three new verbs smuggled into a table read as
existing is the shape that gets a spec sized wrong. Fix: name them as new, and say what an "Open
folder" means for each of the four `EditorTabKind` values.

**F3 (blocking). P2 reverses 092 D14 silently, and the reversal breaks the verb's feedback.**
092 D14: "Right-click with a selection adds `Group...` to the node menu" — node-only, seeded from
`GraphViewState.selection`. P2 moves it into the shared set. On the strip there is no selection to
seed, no rubber band, and no box to see afterwards: 092 D4 puts group boxes on the canvas only.
So a strip Group... opens a prompt, writes `set_pass_groups`, and the strip shows nothing changed.
Fix: either say D14 is reversed and how the strip shows the result, or keep `Group...` on the node
branch and say so in the table's "Surfaces that draw it" column.

**F4. P4 reverses 092 D16's two-click arm without the pointer.** D16: "a member is deleted from its
own node menu with the strip's two-click arm". W3-2 already deleted `App.pass_delete_armed`, so
the arm is gone from the tile — but the repo's habit is an explicit pointer (W3-5: "**Reverses 092
D1's 'moved last' clause and D11's double ring**"). §10 names D16 nowhere. Fix: one clause in P4.

**F5. P8 is not the behaviour-preserving fold it is written as.** `app.py:888-905`: `toggle_copilot`
has three branches (closed→open+focus; open&focused→close; open&unfocused→focus), `toggle_copilot_open`
has two (flip; focus on open). `toggle_copilot(focus_aware=False)` must reproduce the second,
which is not a subset of the first — the "open & focused → close" branch has no `focus_aware=False`
counterpart. P8 says "One `App` verb, one label", which reads as a rename. It is a parameterized
merge of two different state machines, and the comment at `:901` explains *why* they differ ("a
click already moved focus off the chat, so the focus-aware toggle_copilot would blink it back
open"). Fix: write P8 as a merge with the two branch tables shown, or drop it — the observation
(10.1/8) itself concedes "the difference is deliberate and documented".

**F6. P5's second half changes behaviour the observation did not ask about.** 10.1/5 is about
commit semantics ("Enter commits, deactivate does not"). P5 fixes that *and* moves the group
prompt from a floating `begin_popup` to a row at the top of the canvas child. The move has a
cost the pass does not price: 093 S5 built the canvas's Delete-key gate around the prompt being a
popup — "the group-name prompt is a plain `begin_popup`, for which `app.any_popup_open()` is False
and `is_window_hovered(child_windows)` is already False, so a Delete typed into it is refused by
`hovered` before `is_any_item_active` is consulted". An inline row *inside* the canvas child is
hovered, so `delete_allowed`'s `hovered` clause no longer refuses it and only
`not is_any_item_active()` stands between a Delete keystroke in the name field and an unwire.
Fix: keep the prompt a popup and fix only its commit rule, or state the `delete_allowed` change
and re-derive S5's clause table.

**F7. P9's gate is 23 call sites for a two-line defect.** `grep -rn menu_item_simple shaderbox/`
= 23 sites across `lib_picker/tree.py`, `pass_list.py`, `pass_graph.py`. The UX defect P9 names
is one of them (`tree.py:359`, the missing `and has_editor` that `pass_list.py:75` already has).
The `danger=True` half is real (it replaces `tree.py:165-167` and `:279-281`'s hand-rolled red).
The `enabled=` wrapper is real prevention. But "the build's disabled-click footgun is dead at
every site" is only true at the *two* sites that pass `enabled=` at all (§9.4's table has exactly
two rows) — the other 21 pass nothing, so the wrapper changes nothing for them. Fix: scope P9 to
the wrapper plus the two `enabled=` sites plus the two red pushes, and price the gate against
21 mechanical rewrites that fix no defect.

**F8. P11 amends a rulebook that lives outside the repo.** P11: "The `/imgui-ui` §7.4 hint rule is
amended to …". `CLAUDE.md`'s own posture is that the repo stands alone; the skill is not a repo
artifact and a spec cannot commit to editing it. The *rule* is fine; its home is `conventions.md`'s
design-decision list, in "we decided X; revisit if Y" form. Fix: file the amendment as a
conventions bullet, and let the skill follow separately if the maintainer wants it fleet-wide.

**F9. 10.3 and P4 contradict each other on whether a modal is added.** 10.3: "The nine modals and
their `PopupState` mutex (conventions): **no modal is added or merged**." P4 adds `confirm_modal`,
opened from inside the lib picker — a nested modal, which P4's own next sentence names as not the
rulebook's shape ("a nested modal is not"). Under `conventions.md:453-466` a new modal owes an
enum member, an `open_*()`, a self-close, and a `draw_*` call in `ui.py`'s popup block, which
`tests/test_project_management.py` counts. Fix: either P4's confirm is not a `PopupState` modal
(then say what it is and how the mutex sees it), or 10.3's first bullet is wrong.

**F10. The Telegram hand-rolled red confirm child is dispositioned as a delete but not as a tier
violation.** §9.5 lists `exporters/telegram.py:484-485` under "Hand-rolled `push_style_color`
**not** covered by that gate" — a theming defect independent of the delete verb. P4 rules the
*verb* out of scope; nothing rules on the *styling*. 10.3's exporter bullet covers "the panels'
bodies", which is arguably enough, but it does not name §9.5's row. Fix: one clause in 10.3.

**F11. Four §1-8 surfaces get no verdict.** The lib tree's inline favorite star (§3 — the one
row in the app with two affordances for one verb, the exact case §7.4 rules on both ways), the
editor error strip (§2), the `K`-lookup note and the completion popup (§2). Each is probably
"stays", and for the star the rulebook's own carve-out says so — but P3's principle is "a tile
carries no button", and a reader who applies it to the star gets the wrong answer. Fix: four
lines in 10.3.

**F12. The Examples modal's no-reset-on-open is left unruled.** §9.1: "**No** —
`app_state.selected_example_id` persists across opens (deliberate, `examples.py:69`)", against
rulebook §7.6's "On open: reset transient state from the previous session". P6 covers chrome only.
Fix: one line in 10.3 saying the persistence is deliberate and why (it is a *selection* in a
browser, not a search query — a defensible read).

---

## 3. False trails — claims that look like findings and are not

- **"P1 makes the table's order a UI fact."** The pass flags this as a cost. It already is one:
  §7 records that the cheatsheet groups by `CATEGORY_ORDER` and the rebinder walks the same table.
  P1's cost line is honest but the cost is already paid.
- **"P6's gate can't read the revert modal, which isn't in `popups/`."** It can — P6 says
  "walks `popups/*.py` **and the revert modal's AST**". And the revert modal already uses the
  wrapper (`copilot_chat.py:183 with modal_window(...)`), so P6's "returns `keep_open` through
  `modal_window` like the other eight" overstates only the *return*, not the wrapper. Not a
  finding; the fix P6 names is the right one.
- **"P3 leaves `preview_cell` with dead delete machinery."** It does not — P3 is explicit that the
  sticker grid is the one remaining caller and that the machinery stays for it, and 10.4's Option
  A/B is exactly that question surfaced rather than buried.
- **"P7 breaks the Settings apply-on-close."** It does not. `hotkeys.py:366,398-399`'s
  `was_settings_open` latch is a separate mechanism from the four carve-outs P7 removes
  (§9.2 records Settings as surviving the bypass *because* of that latch, not because of a
  carve-out).
- **"P10 will break the prose gate by deleting allowlist rows."** Deleting an `_OVER_BUDGET` row
  is exactly what makes the gate hold the shortened string — `_EXEMPT = set(_OVER_BUDGET)`
  (`tests/test_ui_prose_budget.py:582`), so a deleted row re-arms the measurement. P10 has this
  right.
- **"The pass ignores `SAVE` having no mouse home."** It does not — 10.1/2 names it and P1 fixes
  it by construction (a File-category menu rendered from the table gets `Save`).
