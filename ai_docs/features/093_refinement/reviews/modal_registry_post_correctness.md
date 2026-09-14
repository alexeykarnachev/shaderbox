# Post-implementation review — `d5bf84c`, role: CODE CORRECTNESS

Target: `093: wave 4 -- one modal mechanism, the confirm modal`, against
`07_modal_registry_spec.md` R1-R7 and the pre-implementation report
`reviews/modal_registry_pre.md`.

Baseline for every before/after comparison: a worktree at `17d235c` (the spec commit, the
last tree without the registry).

**Read end to end:** `shaderbox/app.py` (`ModalId`, the state fields, `any_popup_open`,
`_open_modal`, `request_confirm`, `clear_confirm`, `open_copilot_revert`, the five
`*_confirmed` verbs, `delete_pass`, `close_pass_settings`, `close_import_passes`,
`close_emoji_picker`, `close_lib_picker`, `reset_projects_state`, `reconcile_popup_focus`,
`_init`'s confirm clear, `switch_project` / `request_project_switch`),
`popups/__init__.py`, `popups/registry.py`, `popups/confirm.py`, `popups/examples.py`,
`popups/help.py`, `popups/settings.py`, `popups/pass_settings.py`, `popups/import_passes.py`,
`popups/emoji_picker.py`, `popups/projects.py`, `popups/lib_picker/__init__.py`,
`popups/lib_picker/tree.py`, `ui.py` (the three `ModalId` reads + the popup block),
`hotkeys.py` (`_handle_escape`, `_dispatch_registry`, `_drain_editor_input`),
`ui_models.py`, `ui_primitives.py` (`modal_window`), `widgets/copilot_chat.py`,
`widgets/pass_list.py`, `widgets/document_grid.py`, `widgets/pass_graph.py`,
`tabs/document.py`, `menus.py`, `commands.py`, `scripts/smoke.py`, `tests/test_confirm.py`,
`tests/test_modal_chrome.py`, and the repointed tests (`test_menus.py`,
`test_render_decoupling_loop.py`, `test_ui_prose_budget.py`, `test_import_dialog.py`,
`test_project_management.py`).

**Skipped, with the reason:** `popups/lib_picker/{search,preview,filtering}.py` and
`popups/emoji_data.py` — the diff does not touch them and none reads `app.modal`, the
registry, or `ConfirmRequest` (`grep` over `shaderbox/` for `\.modal\s*=`, `request_confirm`
and `ModalId` returns nothing in them). `shaderbox/copilot/`, `exporters/`, `editor/` — same
test, no hits, and no path from them into the funnel.

**Gate state, captured unpiped:** `make gates > gates.log 2>&1; echo $?` → `0`;
`== gates: GREEN -- check passed, test passed, smoke passed ==`. The full suite is
`2584 passed, 4 skipped`.

---

## Verdict: PARTIAL

The mechanism is correct on every path I could drive. The imgui stack is balanced on every
branch, the Esc funnel reaches every modal's `on_close`, a declined close leaves `app.modal`
and imgui's popup stack in agreement, the confirm fires exactly once per frame and never on
its appearing frame, the render-plan gates read `ModalId` the way they read `PopupState`,
and every break the commit message claims that I re-ran came back red for the reason it
names.

Four findings. **F1 is a live behavior regression against the baseline** and is the one
worth fixing before this is called done; F2 is its second consequence; F3 is a gate whose
domain is narrower than the invariant it states; F4 is cosmetic.

Counts: 4 findings — 1 major, 1 moderate, 1 gate gap, 1 trivial. 6 false trails.

---

## F1 (major, R2/R5) — a `request_confirm` from inside an open modal REPLACES it, skipping its `on_close`; the lib picker's Delete now closes the picker

`request_confirm` is `self.confirm = request; self._open_modal(ModalId.CONFIRM)`
(`app.py:1034-1037`), and `_open_modal` writes `self.modal = modal_id` with no consultation
of what was open. Every other transition into a modal goes through a surface that is only
reachable with nothing open — except one:

```
shaderbox/popups/lib_picker/tree.py:34   app.request_confirm(   # _confirm_file_delete
shaderbox/popups/lib_picker/tree.py:45   app.request_confirm(   # _confirm_dir_delete
```

Both are called from the lib picker's own context menus (`tree.py:264`, `tree.py:182`),
which draw **inside** `lib_picker._draw_body` — that is, while
`app.modal is ModalId.SHADER_LIB_PICKER`.

Probe (`tests/` throwaway, removed; the picker opened, a rename armed, then the tree's
delete item fired):

```
armed: True
picker open: modal=shader_lib_picker any_popup=True picker=True confirm=False
after request_confirm -> modal: confirm
picker rename still armed (on_close was SKIPPED)?: True
confirm over picker: modal=confirm any_popup=True picker=False confirm=True
after confirm close: modal= None file gone: True
picker state leaked: rename armed = True
```

Two things happen, both new:

1. **The picker is gone.** Before this wave the lib tree's Delete was a
   `confirm_menu_item` submenu inside the context menu and the verb ran in place
   (`17d235c:shaderbox/popups/lib_picker/tree.py:242` —
   `if confirm_menu_item("Delete", "Move to .trash"): app.shader_lib_files.delete_file(path)`),
   so the picker stayed open and the user carried on browsing. Now confirming leaves
   `app.modal is None`: the user who deletes one lib file has to reopen the picker
   (Alt+L) to delete a second. That is a regression in the one flow where deletes come in
   runs.
2. **`SHADER_LIB_PICKER`'s `on_close` never runs.** `close_lib_picker`
   (`reset_inline_state()` + `picker_tag_input_focused = False`) is the cleanup F2 of the
   pre-review had installed for exactly this state, and the replacement path steps around
   the funnel entirely. The probe shows `inline_input_owns_esc()` still `True` after the
   confirm has closed. It self-heals on the next `open_shader_lib_picker` (which calls
   `reset_inline_state()` first), so nothing is corrupted — but the invariant the wave
   states, "every close runs the row's `on_close`", is false on this path.

The spec's own words are the test it fails: R2 calls `close_modal` "the ONE close funnel"
and R3 puts the cleanup "in ONE place". A modal that is replaced is closed without reaching
either.

Severity major because (1) is user-visible and a regression, not a new-feature rough edge.

Nothing in the suite covers it: `test_confirm.py::test_the_lib_tree_delete_asks_through_its_menu`
calls `tree._confirm_file_delete(app, victim)` with **no modal open**, so the replacement
never occurs in the test. That is the "passing for the wrong reason" shape — the test
demonstrates the confirm is asked, and says nothing about the state it was asked from.

Two shapes close it, and the choice is a design call, not mine to make:

- `request_confirm` refuses (or defers) while another modal is open, and the lib tree's
  deletes stay in-place armed rows the way Projects' delete and Settings' library reset
  already do (`07_modal_registry_spec.md`'s own "Out of scope" names a modal over a modal
  as "not the mechanism's shape" — the lib tree is a third instance of that case, and the
  section's Trigger is "a third in-modal confirm").
- Or `request_confirm` remembers the modal it displaced and `close_modal` restores it, at
  which point the funnel must run the displaced row's `on_close` on the way in.

---

## F2 (moderate, R1) — the same replacement clobbers the popup focus-restore capture

`_open_modal` unconditionally writes `self._chat_focused_before_popup = self.copilot_focused`
(`app.py:1027-1032`). On a normal open that is right: the openers run before any window
draws, so `copilot_focused` still holds the true pre-popup value. On the F1 replacement path
it is read while a modal is already up, where `copilot_focused` is `False` by construction —
so the flag the first modal captured is overwritten with `False`.

Probe:

```
after picker open: _chat_focused_before_popup = True
after confirm from inside the picker: _chat_focused_before_popup = False
```

`reconcile_popup_focus` (`app.py:917-929`) then takes the `elif app.editor_was_ever_focused`
branch on the close edge, and a user who had the chat focused before Alt+L lands back in the
editor instead of the chat. `_popup_was_open` itself is edge-correct (`any_popup_open()` is
`True` across the whole picker→confirm→closed run, so the edge fires once, at the right
frame) — the defect is only the captured value.

Same root cause as F1 and fixed by the same change; filed separately because a fix that only
restores the picker would still leave this if it re-enters through `_open_modal`.

---

## F3 (gate gap, R6) — the mutex-write gate walks `popups/` and `widgets/` only, so `tabs/`, `menus.py`, `hotkeys.py` and `ui.py` can write `app.modal` freely

`test_modal_chrome.py::test_no_popup_or_widget_writes_the_mutex` derives its cases from
`_package_sources(("popups", "widgets"))`. The invariant the spec states is broader — R2's
last line is "No module under `popups/` or `widgets/` assigns `app.modal`", but the
pre-review's gap 4 asked for `shaderbox/` minus `app.py` minus `registry.py`, naming this
exact family ("a checker that quietly narrows its own domain") and the reason ("the next
widget that grows a modal re-opens the hole").

Demonstrated by breaking it. One line added to `shaderbox/tabs/document.py`, right beside
the Reset button this very wave rewired:

```python
    if danger_button(label, width=width):
        app.modal = None                 # <- the break
        app.reset_document_confirmed()
```

`uv run pytest tests/test_modal_chrome.py -q` → `64 passed`. The full suite with the break
in place → `2584 passed, 4 skipped`. Restored afterwards; `git status` clean.

`tabs/` is not a hypothetical neighbour: `tabs/document.py` is in this commit's diff, it
holds a destructive control, and it is where the next one would go. The fix is one tuple:
walk every `*.py` under `_PKG` except `app.py` and `popups/registry.py`.

(Not filed: `setattr(app, "modal", None)` also slips past. Every AST gate has that hole and
nobody writes it by accident; the `tabs/` gap is the one a normal edit falls into.)

---

## F4 (trivial) — `popups/lib_picker/__init__.py` imports `modal_window` and no longer uses it

`from shaderbox.ui_primitives import modal_window, primary_button, standard_button`
(`lib_picker/__init__.py:24`); `grep -n modal_window` in that file returns only the import.
Ruff does not catch it because `pyproject.toml`'s `[tool.ruff.lint.per-file-ignores]` has
`"__init__.py" = ["F401"]`, and the lib picker's body happens to live in an `__init__.py`.
Every other popup module dropped the import in this commit.

---

## What I verified holds (each driven, not argued)

**The imgui stack is balanced on every `draw_modal` branch.** `modal_window` is a
`@contextmanager` whose inner `with imgui_ctx.begin_popup_modal(...)` calls `end_popup()`
from `__exit__` and only when `visible`
(`.venv/.../imgui_bundle/imgui_ctx.py:355-357`). `draw_modal`'s early `return` on
`not visible` throws `GeneratorExit` at the yield, which still unwinds that inner `with`, so
the not-visible branch ends with nothing pushed. `close_current_popup()` is called on one
branch only, and only when `close_modal` returned `True` — i.e. inside the popup scope and
after a close that happened. `modal.before(app)` runs outside the scope, which is what the
pass-settings `set_next_window_size_constraints` needs.

**Every path that assigns `app.modal`, and where it lands.**
`grep -rn '\.modal\s*=' shaderbox/` returns hits in `app.py` and `popups/registry.py` alone:

| Site | Reaches `on_close`? |
|---|---|
| `registry.close_modal` (the funnel) | yes, by construction — `modal.on_close(app)` then `app.modal = None` |
| `_open_modal` (every `open_*` + `request_confirm`) | n/a on an open; **F1** is the one case where it is a close in disguise |
| `app.py:583` (`_init`'s dead-pointer recovery → `PROJECTS`) | n/a, nothing was open |
| `app.py:1189` `close_pass_settings`, `:1255` `close_import_passes`, `:1351` `close_emoji_picker`, `:1364` `close_lib_picker` | these ARE the `on_close` values; `close_modal` re-writes `None` after, idempotent, and no caller outside the registry remains (`grep` for each name returns only the `MODAL=` rows) |
| `app.py:1620` (`_init` clearing a `CONFIRM` that closes over the outgoing project) | correct and necessary; probed — after `switch_project` with a confirm pending, `modal=None, confirm=None` |

**Esc.** `hotkeys._handle_escape` → `close_modal(app)` (`hotkeys.py:369-370`), unforced, so
`owns_esc` can decline. Driven end to end by
`test_confirm.py::test_escape_reaches_the_funnel_through_the_hotkey_dispatch`, which I
confirmed goes red when `emoji_picker.MODAL` loses its `on_close`.

**A declined close does NOT desync imgui's stack from `app.modal`** — the pre-review's
unverified invariant 5. Probed: picker open, rename armed, `close_modal(app)` returns
`False`, and on the next two frames `app.modal is SHADER_LIB_PICKER` **and**
`imgui.is_popup_open(picker_label)` is `True`. The shape that would have desynced them
(`close_current_popup()` after a refused close) is not in the code: it is guarded by
`and close_modal(app, forced=True)`.

**The Esc path skipping `close_current_popup()` entirely is also fine.** Probed across
frames: after `close_modal` the popup reads open for the remainder of that frame, then
`draw_modal` returns early (`BY_ID.get(None)` → `None`), `begin_popup_modal` is never
reached, and imgui drops it at end of frame. Opening a different modal immediately after
shows only the new label on the stack, never both.

**The confirm cannot fire twice, and never on its appearing frame.** One `confirmed`
expression feeds one branch (`confirm.py:38-45`). Both breaks re-run:

- dropping `and not imgui.is_window_appearing()` →
  `FAILED test_enter_confirms_but_never_on_the_appearing_frame`;
- splitting the Enter branch off into its own `if` →
  `FAILED test_the_verb_runs_once_when_the_button_and_enter_land_together`.

I also drove the hazard the pre-review's F7 named directly — the same-frame keyboard menu
nav, where the bar phase calls `request_confirm` and the popup phase draws the confirm in
the SAME `new_frame()` with Enter down. `ran == []`, `modal == confirm`. The guard is on the
right frame.

**Esc never runs the verb.** `confirm.MODAL.on_close` is `_on_close` → `app.clear_confirm()`,
which only nulls the field. `test_escape_cancels_and_runs_nothing` asserts the spy is empty.

**No stale target in the `on_confirm` closures.** `delete_pass_confirmed` captures
`document_id` and `name` explicitly, and `delete_pass` re-looks both up
(`app.py:1068-1084`); `session.delete_pass` refuses a missing document, a missing pass, and
a last pass (`project_session.py:980-988`) and the App path toasts the error without calling
`close_editor_for_path`, so a target that vanished between the request and the confirm is a
toast, not a leak. `delete_document_confirmed` captures `document_id`.
`reset_document_confirmed` and `copilot_clear_chat_confirmed` capture BOUND METHODS
(`self.reset_current_document`, `self.copilot_clear_chat`) that re-read
`current_document_id` at confirm time rather than the id the title named — but the window
is unreachable from the UI: `commands.popup_suppresses` returns `True` for every scope
(`commands.py:380-384`) and `_drain_editor_input` bails on `any_popup_open()`
(`hotkeys.py:51`), so no chord, menu item, palette entry or keystroke can change the current
document while the confirm is up. Only a copilot turn could, and `copilot_turn_active`
guards the destructive verbs themselves. Not filed.

**The revert request carries the right `Message`.** `open_copilot_revert(msg)` closes over
`msg` (`app.py:1042-1055`) and
`test_the_revert_glyph_asks_before_reverting` asserts `revert_turn` is not called before the
confirm and once after, with the message's head in the title. `_draw_revert_modal` /
`_draw_revert_body` / `copilot_revert_target` are gone from the tree (the Deletions gate
covers it).

**No resource or session leak through the new verbs.** `pass_list._delete_pass` moved to
`App.delete_pass` byte-for-byte in behavior: the `source.path` is still captured BEFORE
`session.delete_pass` and `close_editor_for_path(doomed)` still runs only on success.
Diffed the two; the only change is `app.` → `self.`. Document delete still routes to
`_delete_document_unguarded` through the busy gate.

**`switch_project`'s deferred close is still correct.** The Projects body returns
`keep_open=False` on Open / Enter / "Open other..." (`projects.py:108, 116, 112`), so the
modal closes through `draw_modal`'s forced funnel in the draw phase and the switch is
consumed next frame by `ui.py:189-198`, outside any popup scope. `close_modal` is never
called from `_tick_frame_state`. The one path that leaves the modal open across a switch is
the row DOUBLE-CLICK (`projects.py:84`, `request_project_switch` with no `return False`) —
identical in the baseline (`17d235c:projects.py:84`), so not a regression. I also probed a
non-`CONFIRM` modal held across a switch (`open_import_passes` then `switch_project`): the
body's `host is None` guard returns `False`, the funnel closes it, three frames render
clean.

**The `ui.py` render-plan gates read `ModalId` the way they read `PopupState`.** All three
reads are `is` comparisons on the same members (`ui.py:166, 168, 512`), and the
pass-settings exception at `:512` is unchanged in meaning. `CONFIRM` is in the mutex, so
`planned_set_mode` returns `(False, False)` behind it — asserted directly by the new
`test_the_confirm_modal_pauses_the_normal_render_set`, which also counts renders over eight
frames and demands zero.

**The prose gate really scores the confirm's call sites.** Enumerated `_budgeted()` for
`call == "ConfirmRequest"`: **21 sites** — five `App` verbs × 3 fields, plus
`tree._confirm_file_delete` and `_confirm_dir_delete` × 3. The `_UNMEASURABLE` row on
`confirm.py::_draw_body` therefore suppresses nothing the walk could read.

**Per-test falsifier check.** Every new test in `test_confirm.py` names a break in its
docstring; I re-ran five of them and each went red for the named reason:

| Break | Test that named it | Result |
|---|---|---|
| appearing-frame guard dropped | `test_enter_confirms_but_never_on_the_appearing_frame` | red |
| Enter branch split from the button | `test_the_verb_runs_once_when_the_button_and_enter_land_together` | red |
| `draw_modal` closing UNFORCED | `test_the_lib_pickers_close_button_works_with_a_rename_armed` | red |
| `emoji_picker.MODAL` loses `on_close` | `test_every_row_that_owns_state_carries_a_cleanup`, `test_the_funnel_runs_each_rows_cleanup[emoji_picker]`, `test_escape_reaches_the_funnel_through_the_hotkey_dispatch`, `test_menus.py::test_each_registry_row_runs_its_own_cleanup[emoji_picker]` | 4 red |
| `help.py` writes `app.modal` | `test_no_popup_or_widget_writes_the_mutex[popups/help.py]` | red |

`test_every_row_that_owns_state_carries_a_cleanup` is the right answer to the
domain-narrowing problem for the parametrized funnel test, and the commit message's break 12
note (it "first reported a domain mismatch instead") shows it was written because of that.

**Two gaps in the tests, below finding threshold but recorded.**
`tree._confirm_dir_delete` has no test at all (`grep -rn '_confirm_dir_delete\|delete_dir' tests/`
returns nothing), while its file-sibling has one; the spec's R5 table names both. And
`test_the_bars_delete_document_is_a_plain_item_that_confirms` promises "a PLAIN item
carrying its chord hint" in its docstring but asserts only the no-submenu half — the `Alt+D`
hint the commit message says the item "gets back" is not pinned anywhere.

---

## False trails

Six things checked that turned out fine, recorded so the next reviewer does not re-derive
them.

1. **`examples`' grid dims computed twice, in two different imgui scopes.** `_size` runs
   before `begin_popup_modal` and `_draw_modal_body` runs inside it, each calling
   `_grid_dims`; the baseline computed it once. Spied both calls over three frames: both
   return `(352.0, 482.0)` every time. `_grid_dims` reads only `get_style()` and
   `SIZE.THUMB_LG`, neither of which the modal pushes. One wasted call per frame, no
   divergence.

2. **`close_current_popup()` never being reached on the Esc path.** Looks like a desync and
   is not — see the probe above. imgui closes the popup itself once `begin_popup_modal` stops
   being called.

3. **The three `close_*` App methods double-writing `app.modal = None`.** They do (once in
   the method, once in `close_modal` after `on_close`), and it is idempotent, which is what
   the pre-review's F3 resolution says. No programmatic caller remains outside the registry —
   `grep` for each of the four names returns only its `MODAL=` row plus one comment.

4. **`reconcile_popup_focus` going stale across the modal→confirm replacement.** Its
   `_popup_was_open` edge reads `any_popup_open()`, which is `True` across the whole run, so
   the edge fires once at the real close. Only the captured VALUE is wrong (F2), not the
   edge.

5. **The Projects double-click leaving the modal open across a switch, with stale rows.**
   Probed: after `switch_project`, `modal=projects` and `projects_rows` still names the
   outgoing project. Identical in the baseline (`17d235c:projects.py:84` has the same
   `request_project_switch` with no `return False`), so not this wave's.

6. **`pass_graph.py` not appearing in the diff despite the spec's file list naming it.** It
   composes `pass_list.pass_menu_items` (`pass_graph.py:82, 1507`), so the delete item was
   rewired by the `pass_list` change and the file needed no edit. Its own `Delete` key
   handling (`pass_graph.py:1348`) is an UNWIRE of the selected wire, not a pass delete —
   out of this wave's scope and unchanged.

---

## Should it land?

**Yes, with F1 answered.** F1 is a real regression in a real flow (deleting more than one
shader-library file in a sitting) and it is the one place the wave's own "one close funnel"
invariant is false, so it wants a decision — refuse the nested confirm and keep the tree's
deletes in-place, or teach `request_confirm` to displace and restore. F2 comes with it. F3 is
one tuple and worth shipping in the same commit as the fix, since F1 is exactly the class of
defect a wider mutex gate would keep out. F4 is one line.
