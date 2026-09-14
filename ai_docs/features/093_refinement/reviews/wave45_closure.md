# Wave 4 + 5 closure — `d5bf84c` and `06de6e4`

Closes out wave 4's three post-implementation reports against wave 5, audits wave 5 on its
own, and reads the result as a user would.

**Method.** Every finding in the three reports traced to the wave-5 text or code that closes
it. Every break the commit body claims re-run in a worktree at `06de6e4`
(`git worktree add … 06de6e4`), and the named test confirmed red before restoring. The main
checkout was never mutated. Geometry claims measured by driving real frames rather than by
reading the arithmetic.

## Verdict: **PARTIAL** — two findings

Eleven of the twelve report findings are closed, each with the code that closes it and, where
the commit claimed a break, a re-run that goes red for the named reason. The stacked confirm,
the footer trio, both widened gates and all four of the maintainer's items land as described,
and nine modals measure zero scroll overflow.

Two findings:

1. **`06de6e4` does not pass its own gate.** The commit body says "Gates green (`make gates`,
   exit 0)"; the committed tree fails `check` with ruff `I001`. MAJOR — the claim, not the
   defect, is the problem.
2. **The roadmap banner is still stale** — architecture finding 4, accepted in the wave-5
   brief's "every finding accepted", is the one that did not land.

---

## 1. Closure, finding by finding

### `modal_registry_post_correctness.md`

**F1 (major) — a `request_confirm` from inside an open modal replaces it, skipping `on_close`.**
**CLOSED.** Fixed at the open, not at the caller. `app.py:1031-1046`:

```python
def _open_modal(self, modal_id: ModalId, stacked: bool = False) -> None:
    if stacked:
        self.modal_below = self.modal
    else:
        self.modal_below = None
        self._chat_focused_before_popup = self.copilot_focused
    self.modal = modal_id
```

and `popups/registry.py:58-59` restores rather than nulls:

```python
app.modal = app.modal_below
app.modal_below = None
```

Driven through the exact sequence the brief names — picker open, rename armed, confirm
requested from the tree, confirmed, then Esc:

```
after open picker:      modal=shader_lib_picker below=None          chatfocus=True
rename armed: True      owns_esc=True
after request_confirm:  modal=confirm           below=shader_lib_picker chatfocus=True rename=True
after CONFIRM+close:    modal=shader_lib_picker below=None          rename=True file_gone=True chatfocus=True
Esc with rename armed:  returned=False          modal=shader_lib_picker
Esc after rename clear: returned=True           modal=None          below=None
```

The picker is back, its rename still armed (its `on_close` did NOT run), the file is gone, and
the picker's Esc is still declined by `owns_esc`. Break re-run: `request_confirm` opening
unstacked → four tests red, exactly the four the commit names
(`test_a_confirm_from_inside_the_picker_leaves_the_picker_below`,
`test_escape_on_a_stacked_confirm_returns_to_the_picker`,
`test_a_stacked_confirm_leaves_the_chat_focus_capture_alone`,
`test_the_lib_tree_delete_item_asks_from_inside_the_open_picker`). Second break:
`close_modal` writing `None` → the two that assert the picker is back, red.

**F2 (moderate) — the replacement clobbers `_chat_focused_before_popup`.** **CLOSED** by the
same change; the probe above reads `chatfocus=True` at every step, where the report measured
it flipping to `False`. Pinned by `test_a_stacked_confirm_leaves_the_chat_focus_capture_alone`,
red under break 4.

**F3 (gate gap) — the mutex walk covers `popups/` and `widgets/` only.** **CLOSED.** The
domain is now the package tree minus its two owners (`tests/test_modal_chrome.py`):

```python
_MUTEX_OWNERS: frozenset[str] = frozenset({"app.py", "popups/registry.py"})
```

Break re-run — `app.modal = None` beside the Reset button in `tabs/document.py`, the exact
line the report used:

```
AssertionError: tabs/document.py assigns app.modal at [245]; close through `registry.close_modal`
FAILED test_no_module_but_the_owners_writes_the_mutex[tabs/document.py]
1 failed, 148 passed
```

The case count also tells the story: 148 sibling cases where the old walk parametrized far
fewer.

**F4 (trivial) — `lib_picker/__init__.py` keeps a dead `modal_window` import.** **CLOSED.**
The import line is now `modal_footer, modal_footer_height, primary_button, standard_button`.
An AST scan for imported-but-unreferenced names across every module under `popups/` returns
nothing.

**Recorded gap A — `_confirm_dir_delete` has no test.** **CLOSED.** Break re-run:
`_confirm_dir_delete` calling `delete_dir` before the confirm →
`test_the_lib_tree_dir_delete_asks_through_its_menu` red.

**Recorded gap B — the bar's Delete item asserts nothing about its chord hint.**
**CLOSED.** `test_menus.py` now captures the shortcut each `menu_item` is submitted with and
asserts it equals the bound chord. Break re-run: `command_menu_item` submitting `""` →
`test_the_bars_delete_document_is_a_plain_item_that_confirms` red.

### `modal_registry_post_architecture.md`

**1 — the mutex walk misses `tabs/`.** **CLOSED**, same as correctness F3, same break.

**2 — the `_BODIES` table can point a row at another modal's body.** **CLOSED.** Leaves now
derive from the registry, with only dispatcher rows listed by hand and each pinned to its
owning module. Break re-run — the `PASS_SETTINGS` override pointed at `help._draw_body`:

```
AssertionError: pass_settings's override _draw_body lives in shaderbox.popups.help,
but the row's body is declared in shaderbox.popups.pass_settings
FAILED test_every_dispatcher_override_lives_in_its_own_modules_module
```

This is the finding's own suggested fix, landed: derived from `BY_ID`, no second table.

**3 — the dead `modal_window` import and the `F401` ignore.** Import **CLOSED** (see
correctness F4). The report's secondary suggestion — narrowing the `__init__.py` `F401`
ignore to files that actually re-export — was not taken, and was not part of the accepted
brief; the shape that hid the import remains, but nothing is hidden by it today.

**4 — the roadmap banner is stale.** **STILL OPEN.** See §4.

### `modal_registry_post_fidelity.md`

**F1 — the spec's lib-picker `on_close` sentence is stale.** **CLOSED.** R3 now reads
"`lib_picker` -> `App.close_lib_picker` (a new method holding the pre-funnel cleanup …
matching the other rows' shape)".

**F2 — the spec's `switch_project` sentence names the wrong function.** **CLOSED.** The spec
now attributes the write to `_init` and states the `CONFIRM` condition; `app.py:1634-1639`
also clears `modal_below` there, which the stacked change made necessary.

**F3 — `modal_window`'s docstring describes the deleted `is_X_open` shape.** **CLOSED.** The
docstring is rewritten to the registry shape and now shows a `modal_content` / `modal_footer`
body as the worked example.

**Score: 11 closed, 1 still open.**

---

## 2. Wave-5 audit

### The stacked confirm

Verified above. Three details worth stating because they are easy to get wrong and are right
here: a non-stacked open still writes `modal_below = None`, so nothing else moves; the
stacked open skips the focus capture rather than saving and restoring it, which is the
simpler correct thing given the value is already clobbered mid-draw; and `_init` clears
`modal_below` on a project switch, without which a stale pointer would survive into the new
project.

### `modal_footer_height` / `modal_content` / `modal_footer`

**The arithmetic is exact, not approximately right.** Measured by spying the Help modal's
footer against the reservation in the same frame:

```
reserved (modal_footer_height)      = 34.00
occupied (cursor delta over footer) = 34.00
item_spacing.y=4.00  frame_h=18.00  SPACE.MD(8) + 18 + 2*4 = 34.00
window_h=640.00  cursor_y_at_end=636.00  slack_below=4.00
```

Reserved equals occupied. The `2 * item_spacing.y` is right because imgui advances the cursor
after *every* item including the last, so the spacer and the row each contribute a gap —
counting only the two items is what left the old reservation short.

**Every modal measures zero overflow at the app's default size**, including the three the
brief asked about by name, which skip `modal_content` and subtract `modal_footer_height()`
from their own children instead:

| modal | `get_scroll_max_y()` | window h | uses `modal_content`? |
|---|---|---|---|
| help | 0.00 | 640 | yes |
| settings | 0.00 | 968 | yes |
| projects | 0.00 | 420 | yes |
| **examples** | **0.00** | 556 | no — fixed `_size` math |
| **shader_lib_picker** | **0.00** | 1130 | no — two columns take `body_h` |
| **emoji_picker** | **0.00** | 560 | no — grid child takes `scroll_h` |
| pass_settings | 0.00 | 296 | no — auto-resizing, footer alone |
| confirm | 0.00 | 114 | no — auto-resizing, footer alone |
| import_passes | 0.00 | 600 | yes |

The conditional footers the table above cannot reach were driven separately and are also
clean: the Projects armed-delete row, the Projects name-entry row, a Projects error message
in the content region, and `import_passes` on its rejection path — all `scroll_max_y=0.00`.
Every one of these keeps to the one-row contract via `same_line`, which is what the
reservation assumes.

Break re-run — the hand reservation restored in `help.py` (`list_h` plus a hand-written
spacer): three tests red, the three the commit names, and the overflow is
**`AssertionError: assert 12.0 == 0.0`** — the scrollbar quantified at 12px.

### The chrome gate's new clauses

Each of the four clauses the brief lists was checked, and the two with a break available were
broken:

- **Leaves from `BY_ID`** — `_bodies()` builds the table from `MODALS`; broken above.
- **Dispatcher override pinned to its module** — broken above, red.
- **The mutex walk over every package** — broken above, red; domain is now
  `_PKG.rglob("*.py")` minus two named files.
- **No hand measurement** — `test_no_body_measures_the_footer_by_hand` rejects
  `get_frame_height` / `get_frame_height_with_spacing` in any leaf body; red under the
  `help.py` break.

### The canvas without its border (finding 19)

**The commit's claim that the canvas rect is unchanged is true, and measured rather than
argued.** The real graph tab driven through `update_and_draw`, once as landed and once with
`ChildFlags_.borders` forced back:

```
wave5 (as landed): canvas_rect=(24.0, 86.0, 1264.0, 1382.0) zoom=1.000 pan=(-552.0,-573.0)
wave4 (borders):   canvas_rect=(24.0, 86.0, 1264.0, 1382.0) zoom=1.000 pan=(-552.0,-573.0)
DELTA canvas_rect = (0.00, 0.00, 0.00, 0.00)
out_rects identical: True
```

A node's output port lands at `(705.0, 720.0, 719.0, 734.0)` under both, so a hit at a known
point resolves the same. `always_use_window_padding` reproduces the inset `borders` implied,
which is what makes `_draw_canvas`'s `get_cursor_screen_pos` + `get_content_region_avail`
agree with what it assumed before.

### The remaining items

- **`Frame all`** (21) — `pass_graph._canvas_menu`; `grep '"Fit"' shaderbox/` is empty, and
  `06_command_system.md`'s "what is NOT a command" list is updated to match.
- **The hint removal** (22) — gone from both `widgets/document_grid.py` and
  `popups/lib_picker/__init__.py`; `grep -rn "Right-click for actions" shaderbox/` is empty.
  The conventions bullet is reversed in place (the new one states the rule positively, names
  what it reverses, and keeps a revisit trigger), and M6/M12 in `05_menus_spec.md` carry
  reversal pointers.
- **`Documentation`** (20) — `grep -rn "Help panel" shaderbox ai_docs .claude` leaves
  **nothing in `shaderbox/` or `.claude/`**. The `ai_docs/` hits are review records and
  prior-wave inventories describing what those waves did, which is correct per the doc rules
  — except `roadmap.md:35`, which is finding 4 below. The map test reads
  `06_command_system.md` (`tests/test_menus.py:196`) and passes against the updated block.
- **`Insert at caret`** — `insertable`, `_insert_target_ok` and the tooltip are gone from the
  tree; `App.insert_text_at_caret` stays with one caller,
  `popups/lib_picker/filtering.py:92`, exactly as the commit says.
- **Docs** — findings rows 19-22, the wave-5 entry in `01_spec.md`, the conventions bullets
  (hint reversed, registry bullet carrying `modal_below`, both widened gates), `dev_flow.md`'s
  module map (the footer trio, the hint caption removed), the skill's §7.1/§7.2/§7.4, and the
  spec's divergences all landed and read correctly.

---

## 3. User-facing read

**The six menus, rendered from `COMMAND_SPECS`:**

```
File      New document Ctrl+Shift+N · Save Ctrl+S │ Projects Alt+O · Settings Alt+S │ Quit Ctrl+Q
Document  Open script Alt+R · Open graph Alt+G │ Open folder │ Play/stop script F5 ·
          Reset document F6 │ Delete document Alt+D
Pass      Add pass Alt+A · Import passes │ Open shader Alt+C · Pass settings Alt+P │
          Next pass Alt+Right · Previous pass Alt+Left
Editor    Format code Ctrl+Shift+I · Next error F8 │ Next code tab Ctrl+Tab ·
          Close code tab Ctrl+W │ Shader library Alt+L
View      Document/Uniforms/Render/Share panel Ctrl+1..4 │ Next channel view Alt+V │
          Toggle copilot Alt+J · Next copilot layout Ctrl+H · Clear chat │
          Command palette Ctrl+Shift+P
Help      Documentation F1 · Keyboard cheatsheet Alt+/ │ Examples Alt+E
```

**Nothing reads wrong.** `Documentation` sits where `Help panel` did and no longer repeats its
own menu's name, which was the maintainer's objection. `Delete document` and `Reset document`
are plain items carrying their chords, the submenu gone. Two items show no chord (`Open
folder`, `Import passes`, `Clear chat`) — unbound by design, and the bar renders an empty hint
rather than a placeholder.

**The modal footers, as they render:**

| modal | row, left to right | convention |
|---|---|---|
| Documentation | `Close` | ok |
| Settings | `Close` | ok |
| Examples | `Open a copy` (primary) · `Close` | ok |
| Shader Library | `Insert at caret` (primary) · `Close` | ok |
| Emoji | `Close` | ok |
| Import passes | `Import N passes` (primary) · `Cancel` · *rejection text* | ok |
| Pass settings | `Close` | ok |
| Pass draft | `Create` (primary) · `Cancel` | ok |
| Confirm | *verb* (danger) · `Cancel` | ok |
| Projects (verbs) | `Open` (primary) · `New` · `Open other...` · `Delete` (danger) · `Close` | ok |
| Projects (armed) | "Delete to trash?" · `Yes` (primary) · `No` | ok |
| Projects (name) | *input* · verb (primary) · `Cancel` | ok |

**No modal's row differs from the convention** — primary first, dismiss last, in every one.
Two deliberate shapes worth naming rather than filing: the confirm leads with a danger-tier
verb instead of a primary, which is the destructive-confirm tier and matches wave 4's design;
and Projects' armed row leads with text before `Yes`, which is the armed-row pattern the
conventions bullet describes. The `import_passes` rejection message trails the dismiss button
— it is a status string rather than a control, and it stays on the row via `same_line`, so
the footer's one-row contract holds.

---

## 4. Findings

### Finding 1 (major) — `06de6e4` fails its own gate; the commit body claims exit 0

The commit body states: "Gates green (`make gates`, exit 0, smoke passed not skipped)." The
tree as committed does not pass `check`. From a pristine worktree at `06de6e4`, untouched:

```
$ make gates > /tmp/gates.log 2>&1; echo $?
2
== gates: FAILED at check (exit 2); test and smoke not run ==

$ uv run ruff check --select I tests/test_modal_footer.py
I001 [*] Import block is un-sorted or un-formatted
  --> tests/test_modal_footer.py:11:1
Found 1 error.
```

`tests/test_modal_footer.py`, new in this commit, places `from dataclasses import replace`
after the third-party block. With ruff's fix applied and nothing else changed, the gate goes
green end to end:

```
$ make gates > /tmp/gates2.log 2>&1; echo $?
0
== gates: GREEN -- check passed, test passed, smoke passed ==
```

So the defect is one import line and the fix is what ruff already writes. **The finding is the
claim.** `check` runs first and stops the gate, so `test` and `smoke` never ran either — which
means the body's "smoke passed not skipped" describes a run that did not happen on this tree.
The most likely history: the gate was run before the file was staged in its committed form, or
run in a tree where pre-commit had already applied the fix, and the fix was then left
uncommitted. That is consistent with the main checkout, which carries exactly this
one-line reordering as its only uncommitted change — the correction exists, it was simply
never committed.

This is the failure `CLAUDE.md` names twice ("report a state only after running the check that
would disprove it"; the `make gates` rule's "this repo has twice announced a green gate that
was red"). It makes the third.

**Fix:** commit the working tree's `tests/test_modal_footer.py` reordering. One line, no
behavior change, and the gate is green with it.

### Finding 2 (moderate) — the roadmap banner is still stale

Architecture finding 4, inside the set the brief accepted in full, did not land.
`ai_docs/roadmap.md`'s Active-context block still reads:

```
ai_docs/roadmap.md:29  <!-- As of 2026-09-14, 093's menus wave landed and reviewed; ... -->
ai_docs/roadmap.md:30  **Next: his visual review of the menus wave** ...
ai_docs/roadmap.md:33  ... a document tile (`Delete` confirms through a submenu), the name
ai_docs/roadmap.md:35  clause with the Help panel's new section. Each finding is a new ledger
ai_docs/roadmap.md:36  row (18+) in `00_findings.md`, fixed as wave 4 by ...
```

Three separate contradictions with the landed code, and the banner is the cold-start chain's
step 2, which `CLAUDE.md` calls the authoritative "what's next?":

1. "`Delete` confirms through a submenu" — reversed by wave 4; the submenu is gone and
   `confirm_menu_item` is deleted.
2. "the Help panel's new section" — renamed to `Documentation` by wave 5, and this is the one
   live `Help panel` string left anywhere outside historical records.
3. "fixed as wave 4" frames wave 4 as upcoming work, when waves 4 and 5 have both landed.

`01_spec.md` was correctly updated to "waves 1-5 landed", so the two documents now disagree
about where the feature is. A fresh session following the chain reads step 2 and gets a
next-step that is two waves done and a UI shape that no longer exists.

**Fix:** rewrite the Active-context block in full, per its own instruction ("Rewrite this
block IN FULL each time it changes. Do NOT append."), naming waves 4-5 as landed and the
visual review of the confirm modal, the modal footers and the four items as the next input.

---

## 5. Breaks re-run

All seven the commit names, each red for its stated reason, each restored and the tree
verified clean afterwards.

| # | Break | Named test | Result |
|---|---|---|---|
| 1 | `app.modal = None` in `tabs/document.py` beside Reset | `test_no_module_but_the_owners_writes_the_mutex[tabs/document.py]` | **red** |
| 2 | `PASS_SETTINGS` override → `help._draw_body` | `test_every_dispatcher_override_lives_in_its_own_modules_module` | **red** |
| 3 | hand reservation restored in `help.py` | `test_a_modal_with_a_content_region_does_not_overflow[help]` + the two chrome clauses | **red** (overflow = 12.0px) |
| 4 | `request_confirm` opening unstacked | the four stacked-confirm tests | **red** (exactly 4) |
| 5 | `close_modal` writing `None` | the two asserting the picker is back | **red** (exactly 2) |
| 6 | `_confirm_dir_delete` deleting before the confirm | `test_the_lib_tree_dir_delete_asks_through_its_menu` | **red** |
| 7 | `command_menu_item` submitting an empty shortcut | `test_the_bars_delete_document_is_a_plain_item_that_confirms` | **red** |

Each break's test failed for the reason the commit gives, not incidentally — the assertion
messages name the mutated thing. No break produced a wider blast than claimed, which is the
other way a gate can mislead.

---

## 6. False trails

Checked and not findings, recorded so the next reviewer does not re-spend the time.

- **`modal_footer` reserving two `item_spacing.y` gaps looks like one too many.** It is
  exactly right: imgui advances the cursor after every item including the final one, so the
  spacer and the row each contribute. Measured — reserved 34.0 equals occupied 34.0 with 4px
  slack below, and dropping to one gap is what produced the original scrollbar.

- **The three modals that skip `modal_content` look like they escaped the fix.** They did not.
  Examples, the lib picker and the emoji picker each subtract `modal_footer_height()` from
  their own child sizing instead, so they read the same one number; all three measure
  `scroll_max_y = 0.00`. The primitive is the number, not the context manager.

- **`import_passes`' rejection text trailing the `Cancel` button looks like a footer-order
  violation.** It is a status string, not a control, and it stays on the same row via
  `same_line`. The row still ends in the dismiss *button*, and the footer measures zero
  overflow on the rejection path.

- **Projects' three alternative footers look like a dispatcher the chrome gate would miss.**
  The gate lists `projects._draw_verb_row` as the dispatcher's leaf and pins it to the
  module that declares the row; the other two are a name-entry row and an armed-delete row,
  neither of which is the dismiss row the clauses describe. Both were driven anyway and both
  measure zero overflow.

- **The canvas losing `ChildFlags_.borders` looks like it would shift the hit geometry by the
  border's pixel.** Measured under both flag sets: the canvas rect and every port rect are
  identical to 0.01px. `always_use_window_padding` reproduces the inset.

- **`grep -rn "Help panel" ai_docs` returning ~40 hits looks like an incomplete rename.**
  All but one are review records, prior-feature specs and inventories describing what those
  waves did at the time — historical records, correct per the doc rules. `shaderbox/` and
  `.claude/` are clean. The single live one is `roadmap.md:35`, filed as finding 2.

- **`App.insert_text_at_caret` surviving the `Insert at caret` removal looks like dead code.**
  It has one live caller, `popups/lib_picker/filtering.py:92` — the lib picker inserts through
  it, which is what the commit says.

- **The `F401` per-file ignore for `__init__.py` still hides unused imports in
  `lib_picker/__init__.py`.** The shape architecture finding 3 named remains, but an AST scan
  of every module under `popups/` finds no imported-but-unreferenced name today. Narrowing the
  ignore was a suggestion attached to the finding, not part of the finding; the finding itself
  (the dead `modal_window` import) is closed.

---

## 7. Tree state

- Main checkout: untouched by this review. Its one uncommitted change —
  `tests/test_modal_footer.py`'s import reordering — was present before this review began and
  is the subject of finding 1.
- Worktree at `06de6e4`: every mutation restored, `git status --porcelain` empty at the end of
  each break.
- `make gates` on `06de6e4` as committed: **exit 2**, red at `check`. With the one-line import
  fix: **exit 0**, `check passed, test passed, smoke passed` (smoke passed, not skipped).
