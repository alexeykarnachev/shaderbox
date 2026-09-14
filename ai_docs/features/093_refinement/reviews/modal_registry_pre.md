# Pre-implementation review — 07_modal_registry_spec.md

Roles A (correctness & design) and B (verification & blast radius), one reviewer.
Read: `CLAUDE.md`, `conventions.md ## Design decisions` (the `popups/*.py`, three-layer,
`InlineInput` and the 093 confirm bullets), `imgui-ui/SKILL.md` §7.1-7.3 / §7.6,
`dev_flow.md` step 4, and the code end to end (`app.py`, `ui.py`, `hotkeys.py`, every
`popups/` module, `widgets/copilot_chat.py`, `widgets/pass_list.py`, `menus.py`,
`commands.py`, `ui_primitives.py`, the four gate tests).

## Verdict: PARTIAL

The shape is right and the wave is worth doing: the five-place roster is real, the
conventions bullet already warns about forgetting one of the five, and the registry
collapses all five into one tuple. Nine findings below block a clean landing; two of
them (F1, F2) are structural and change the spec's design, the rest are missing touches
and gate gaps. None is "should not land".

Counts: 9 findings — 2 design/structural, 4 missing touches, 3 verification gaps.
False trails: 5 things checked that turned out fine.

---

## A. Correctness and design

### F1 (R2/R4, structural) — `ConfirmRequest` on `App` closes the import cycle the spec's own rule forbids

The spec's import rule is explicit and correct as far as it goes:

> `registry.py` imports `App` and every popup module; no popup module imports the
> registry (the popups import `App` only, as today), and `app.py` never imports `popups`

The actual graph that holds today, verified by grep (`grep -n popups shaderbox/app.py`
returns only two comment lines, no import):

```
app.py            -> (nothing in popups)
popups/*.py       -> app.py
popups/registry.py-> app.py, popups/*
ui.py             -> app.py, popups/registry
hotkeys.py        -> app.py, popups/registry
```

Acyclic. But R4 then puts `ConfirmRequest` in `popups/confirm.py` and declares
`App.confirm: ConfirmRequest | None` plus `App.request_confirm(request)`. That is a name
from `popups/confirm.py` used in an annotation inside `app.py` — so `app.py` must
`from shaderbox.popups.confirm import ConfirmRequest`, and `popups/confirm.py` imports
`App`. That is a hard cycle, and the two escapes are both banned: `if TYPE_CHECKING`
(CLAUDE.md, no-`TYPE_CHECKING` rule, "a circular import is a design bug") and a
function-body import (CLAUDE.md, imports at module top only).

The spec never states where `ConfirmRequest` lives beyond the `popups/confirm.py` heading,
so this is an unstated violation rather than a knowing one — which is exactly the class
`dev_flow.md` step 4 asks a reviewer to find.

**What closes it.** `ConfirmRequest` is a pure dataclass with no imgui and no `App` in it
(`title`, `line`, `verb`, `on_confirm: Callable[[], None]`). It belongs in the leaf layer
that `app.py` already imports from — `ui_models.py` is where `ImportDraft` and `PassDraft`
already live and is already an `app.py` import. Spec text that closes it:

> `ConfirmRequest` is a frozen dataclass in `ui_models.py` (the leaf `app.py` already
> imports `ImportDraft` and `PassDraft` from); `popups/confirm.py` holds only the body and
> `MODAL`, and imports the request type from `ui_models`.

Same question applies to `ModalId`: R1 puts it in `app.py`, which is right and cycle-free
(popups already import from `app.py`), so leave that as is.

### F2 (R2/R3, structural) — `lib_picker` is given no `on_close`, and its close cleanup is real

R3's `on_close` table says `examples`, `help`, `lib_picker` — none. That is true for the
first two. It is false for `lib_picker`, whose current wrapper does work before the funnel
(`shaderbox/popups/lib_picker/__init__.py:39-44`):

```python
if not _draw_body(app):
    # A Close reached with an inline input still armed cancels it first: the funnel
    # otherwise declines, leaving a modal the user asked to dismiss on screen.
    app.shader_lib_files.reset_inline_state()
    app.shader_lib_files.picker_tag_input_focused = False
    app.close_popup()
```

That comment names a live defect the cleanup exists to prevent. Under the spec's
`draw_modal`, the body returning `False` calls `close_modal(app)`, which consults
`owns_esc` FIRST — and `lib_picker.owns_esc` is `app.shader_lib_files.inline_input_owns_esc`,
which is true exactly when an inline input is armed. So `close_modal` returns `False`,
`app.modal` stays set, and the Close button becomes dead while a rename input is open.
This is a behavior regression the registry shape introduces, not carries over.

The spec's `draw_modal` also calls `imgui.close_current_popup()` unconditionally after
`close_modal` — including on the `owns_esc` decline — which desyncs imgui's popup stack
from `app.modal`.

**What closes it.** Two clauses:

> `lib_picker`'s `on_close` is a new `App.close_lib_picker` (`reset_inline_state()` +
> `picker_tag_input_focused = False`), so the cleanup the wrapper did by hand moves into the
> registry's one place.
>
> `close_modal`'s `owns_esc` decline is for the Esc path only: `draw_modal` calls
> `close_modal(app, force=True)` when the body returns `False` (a Close click is not an Esc),
> and `imgui.close_current_popup()` runs only when `close_modal` returned `True`.

The `projects` case is the mirror: its wrapper's comment says the body only returns `False`
once its own input is closed, so `owns_esc` cannot refuse it — with `force=True` that
reasoning stops being load-bearing, which is a small gain.

### F3 (R2, missing touch) — the three cleanups that "stop writing the state field" have a caller outside the registry

R2 says an `App` method closing a modal programmatically "sets `self.modal = None` itself",
and R3 says the three `close_*` App methods stop writing the state field and become pure
cleanups. Those two sentences contradict each other for `close_pass_settings` and
`close_import_passes`, which are BOTH the registry's `on_close` AND direct callers:

- `popups/pass_settings.py:60` calls `app.close_pass_settings()` on the body's `False`.
- `popups/import_passes.py:51` calls `app.close_import_passes()` on the body's `False`.
- `app.py:1147` and `app.py:1213` write `popup_state = CLOSED` inside them.

If they stop writing the field (R3), a direct programmatic caller leaves the modal open.
If they keep writing it (R2's "it IS the cleanup owner"), then `close_modal` running
`on_close` and then `app.modal = None` writes `None` twice, which is harmless — but R6's
gate "no `popups/` module assigns `app.modal`" does not catch an `App` method doing it, so
the two rules can coexist without a contradiction only if the spec says which.

Additionally: `create_pass_from_draft` returning `True` is the pass-settings modal's
commit path and today rides `close_pass_settings`; the spec's file list mentions
`close_pass_settings` but not who calls it after the registry lands.

**What closes it.** One sentence resolving the direction:

> The three `close_*` methods stay full closes (`self.modal = None` plus the cleanup) and
> are ALSO the registry's `on_close` values; `close_modal` sets `app.modal = None` after
> `on_close` regardless, so the double write is idempotent and a programmatic caller needs
> no registry import.

### F4 (R1/R5, missing touch) — `settings`' `focus` jump and `pass_settings`' DRAFT mode are behaviors the registry shape must be told to keep

Two current behaviors live in the `draw_*` wrappers the spec deletes, and neither is named:

1. **`pass_settings` DRAFT dispatch** (`popups/pass_settings.py:60`):
   `keep_open = _draw_draft(app) if app.pass_draft is not None else _draw_body(app)`.
   One `ModalId.PASS_SETTINGS` serves two bodies. The spec's `Modal.body` is a single
   callable, so the dispatch must move INSIDE the module-level body. R6 then asserts
   chrome clauses on `MODAL.body`, and if that becomes a dispatcher, the `keep_open` bind
   and the `SPACE.MD` spacer live in `_draw_body` / `_draw_draft`, not in the function the
   gate walks — the chrome gate silently narrows its domain (the exact failure family the
   maintainer's own rulebook calls out). `test_modal_chrome.py` today resolves
   `PASS_SETTINGS` to `pass_settings._draw_body` through `_BODIES`, which is how it dodges
   this; the registry's `MODAL.body` has to point at the same place or the gate goes soft.

2. **`pass_settings`' size constraints** (`set_next_window_size_constraints` +
   `always_auto_resize`) are covered by `Modal.before`, which the spec names. Good.

3. **`settings`' `focus` jump**: `App.open_settings(focus=...)` sets `settings_focus` /
   `settings_mark` before opening, and `_draw_body` consumes them (`popups/settings.py:176`
   `app.settings_focus = ""`). R1 keeps the per-modal `open_*` verbs on `App`, so this
   survives — no change needed, but the spec should say so, because "the per-modal `open_*`
   verbs stay on `App` (they set the modal's own state first, then open)" is the only line
   carrying it and a reader could take `_open_modal(id)` as the whole opener.

**What closes it.**

> `Modal.body` for `PASS_SETTINGS` is a `_draw_modal_body(app)` that dispatches to
> `_draw_draft` / `_draw_body`; `test_modal_chrome`'s chrome clauses walk BOTH leaf bodies
> for that modal, never the dispatcher.

### F5 (R1, missing touch) — `ui.py`'s three `PopupState` reads are three different things, and one of them changes meaning

The spec says "`ui.py`'s three `PopupState` reads become `ModalId` reads". Mechanically true,
but the three are not equivalent and the spec should name what each becomes, because two of
them are render-plan gates with documented invariants:

- `ui.py:173` `importing = app.popup_state == PopupState.IMPORT_PASSES` →
  `app.modal is ModalId.IMPORT_PASSES`.
- `ui.py:175` `examples_planned = app.popup_state == PopupState.EXAMPLES or (...)` →
  `app.modal is ModalId.EXAMPLES or (...)`.
- `ui.py:519` `app.popup_state == PopupState.PASS_SETTINGS and ...` — this one carries a
  three-way agreement comment ("The THIRD read of the plan, and it must agree with the other
  two"). It gates the current document's `render()` behind the pass-settings modal.

The behavioral change the spec does not mention: `any_popup_open()` today is
`popup_state != CLOSED or copilot_revert_target is not None`. Under R1 it becomes
`self.modal is not None` — and the confirm modal joins `ModalId`, so a confirm that was
previously NOT in the mutex on some paths (the revert confirm was, via `copilot_revert_target`)
is now uniformly in it. That is a net simplification and the right direction, but it means
`planned_set_mode` now returns `(False, False)` for `ModalId.CONFIRM`, pausing the document
render behind a confirm — which is what every other non-pass-settings modal already does.
Fine, but worth one line so the post-impl reviewer does not read it as a regression.

**What closes it.** One row in R1:

> The three `ui.py` reads become `app.modal is ModalId.{IMPORT_PASSES, EXAMPLES,
> PASS_SETTINGS}`; `ModalId.CONFIRM` is in the mutex like every other modal, so the render
> plan pauses behind it (the pass-settings exception at `ui.py:519` is unchanged).

### F6 (R2, missing touch) — the project switch's deferred close

`App.request_project_switch` (`app.py:2376`) exists precisely because a popup body must not
release GL textures inside its own draw:

> The modal never switches inside its own draw: a popup body runs AFTER the editor panel
> and the document image have pushed their textures into the draw list, so releasing them
> there leaves imgui rendering freed GL names. `_tick_frame_state` consumes this before
> any drawing (084 D5).

The spec names "the project switch" among the `App` methods that set `self.modal = None`
themselves, which is correct — but the switch's close runs from `_tick_frame_state`,
OUTSIDE the draw phase, where `imgui.close_current_popup()` cannot be called (it asserts
outside a popup scope). Today `switch_project` → `_init` resets state and the popup simply
stops drawing on the next frame, so imgui's own popup stack is closed by
`begin_popup_modal` no longer being reached. Under the registry the same holds, because
`draw_modal` returns early when `BY_ID.get(app.modal)` is `None`. So the behavior survives
— but the spec's `close_modal` is described as "the ONE close funnel: Esc, and a body
returning False", and a reader implementing `switch_project` may reach for `close_modal`,
which would then be called outside a frame. Name the exception.

**What closes it.**

> `switch_project`'s close is `self.modal = None` directly, never `close_modal` — it runs in
> `_tick_frame_state`, outside the draw phase, where `imgui.close_current_popup()` asserts.

### F7 (R4) — Enter-confirms is safe as specified, with one narrow caveat worth pinning

Judged against the installed imgui-bundle 1.92.801. `imgui.is_key_pressed(key, repeat=False)`
is documented in the binding as "was key pressed (went from !Down to Down)?" — edge-
triggered on the transition, not level-triggered on the held state. So:

- **A held Enter does not re-fire.** Any path where the confirm opens on frame N and the
  body first draws on frame N+1 is safe: on N+1 Enter is Down-and-was-Down, not a press.
- **The palette path is such a path.** The palette draws at `ui.py:639-641`, AFTER the popup
  block at `ui.py:629-637`. A palette Enter fires `initial_callback` → `request_confirm` →
  `app.modal = CONFIRM` on frame N, and the confirm body first draws on N+1. Safe.
  (Moot anyway once R5 restores every `in_palette` spec: the palette offers the verb, the
  verb opens the confirm, and the user's Enter is already consumed by the palette.)
- **The menu-bar path is a same-frame path.** `menus.draw_menu_bar(app)` is at `ui.py:555`,
  the popup block at `ui.py:629` — a menu-item click opens the confirm and the body draws
  on the SAME frame. The spec's justification ("the click that opened it was a mouse
  gesture and no Enter is in flight") holds for a mouse click. It does NOT hold for a menu
  item activated by imgui's keyboard nav, where Enter/Space activates the focused item: that
  Enter is a genuine `!Down → Down` transition on that frame, and the confirm's
  `is_key_pressed(enter)` would see it and fire the verb instantly.
- **The chord path is safe.** The two destructive chords are `RESET_DOCUMENT` = F6 and
  `DELETE_DOCUMENT` = Alt+D (`commands.py:158-171`); neither involves Enter, so there is no
  chord whose own keys leave Enter down. `_dispatch_registry` also runs pre-draw, so even a
  hypothetically Enter-bearing chord would open on frame N and draw on N+1.

So the one real hole is keyboard menu-nav, and it is closable in one clause without
reversing the decision.

**What closes it.**

> The confirm body's Enter reads `imgui.is_key_pressed(imgui.Key.enter, repeat=False)` AND
> `not imgui.is_window_appearing()` — the appearing frame is the one a keyboard menu
> activation shares with the press that opened the modal.

If the maintainer prefers, dropping Enter entirely is the one-line reversal the spec's Open
questions section already flags; the guard above is the cheaper keep.

### F8 (R5) — `command_menu_item`'s hint and `document_grid`'s `confirm_label` read

R5 says `CommandSpec.confirm_label` is deleted and `command_menu_item` draws a plain item
for every spec. One call site reads the field for its own copy and is not in the file list's
reason column: `widgets/document_grid.py:52`

```python
if confirm_menu_item("Delete", SPEC_BY_ID[CommandId.DELETE_DOCUMENT].confirm_label):
```

The spec's file list does name `widgets/document_grid.py`, so the file is covered — but the
line's copy ("Move to trash") is the only home for that string, and R5's table gives the
document delete a different line ("Nothing in the app brings it back."). That is a
deliberate rewrite, not a loss; say so, because a post-impl spec-fidelity audit reading
"`confirm_label` gone" against a changed user-visible string needs the intent recorded.

Also: `menus.py` currently imports `confirm_menu_item` from `ui_primitives` (`menus.py:27`)
and `popups/lib_picker/tree.py` imports it too (`tree.py:26`) — both are in the file list.
`tests/test_ui_prose_budget.py` references `confirm_menu_item` as a copy-bearing site; the
spec names the file for the `_UNMEASURABLE` row but not for the `confirm_menu_item`
removal, so the budget gate's domain shrinks by one function and the spec should say the
row is deleted rather than only added.

**What closes it.**

> The document delete's confirm line is REWRITTEN from `Move to trash` to `Nothing in the
> app brings it back.`; `test_ui_prose_budget.py` loses `confirm_menu_item` from its domain
> and gains the `confirm.py::_draw_body` `_UNMEASURABLE` row.

### F9 (Files touched) — the test-file count is nine, not eight

The spec says "every test that sets `popup_state` / `PopupState` (eight files) repointed".
`grep -rn 'popup_state\|PopupState\|close_popup\|copilot_revert_target\|confirm_label\|confirm_menu_item' tests/`
returns **nine** files:

| File | What it touches |
|---|---|
| `tests/test_modal_chrome.py` | `PopupState`, `copilot_revert_target`, `close_popup` — named |
| `tests/test_menus.py` | `PopupState`, `close_popup`, `confirm_label`, `confirm_menu_item` — named |
| `tests/test_project_management.py` | `PopupState`, `close_popup` — named |
| `tests/test_import_dialog.py` | `PopupState`, `close_popup` — named |
| `tests/test_ui_prose_budget.py` | `confirm_menu_item` — named |
| `tests/test_render_decoupling_loop.py` | `popup_state` / `PopupState` — **NOT named** |
| `tests/test_profiling.py` | `popup_state` / `PopupState` — **NOT named** |
| `tests/test_pass_verbs.py` | `popup_state` / `PopupState` — **NOT named** |
| `tests/test_pass_draft.py` | `popup_state` / `PopupState` — **NOT named** |

Four files the spec's list misses, and `test_render_decoupling_loop.py` is the one that
matters most: it drives the render-plan gates that F5 covers, so a mechanical
`popup_state` → `modal` rename there is not enough — its assertions encode which modal
pauses which render set, and `ModalId.CONFIRM` is a new member entering that domain.

**What closes it.** Replace "eight files" with the nine named above, and add:

> `test_render_decoupling_loop.py` gains a case for `ModalId.CONFIRM` pausing the normal
> render set, since a new mutex member changes the plan's domain.

---

## B. Verification and blast radius

### Row-by-row: can the named test fail for exactly the reason the row names?

| Row | Verdict |
|---|---|
| R1/R6 the registry is the roster | **Yes.** `{m.id for m in MODALS} == set(ModalId)` plus a uniqueness assert; the named break (add a member) fails it directly. |
| R2 one draw call | **Yes**, and it is the right replacement for the deleted `test_every_popup_state_has_a_draw_call` (which parses `ui.py`'s import set and call set at `test_project_management.py:624-655`). The named break (re-add `draw_help(app)`) fails the "imports no `draw_*` from `popups`" half. |
| R2 no popup writes the mutex | **Yes** for `popups/`. But see the gap below: it does not cover `widgets/`, and `copilot_chat.py` writes the mutex today (`app.copilot_revert_target = None` at `copilot_chat.py:188`). |
| R2 the close funnel | **Yes for `on_close`. Partly for `owns_esc`** — the row says "arm the input, `close_modal` returns `False` and the modal stays", which is exactly the behavior F2 shows is WRONG for the Close-button path. The row as written would pass on an implementation that has the dead-Close-button bug. |
| R3 chrome | **Yes**, but see F4: if `PASS_SETTINGS`'s `MODAL.body` becomes a dispatcher, this row passes while checking nothing for that modal. |
| R4/R5/R7 the confirm from every surface | **Yes**, and it is the strongest row — a verb spy plus the Esc case per client is a real falsifier. |
| R5 the palette offers every spec | **Yes.** `_palette_command_names` is a live attribute (`app.py:470`) and the filter is one comprehension at `app.py:849-851`. |
| R5 the bar's Delete document is a plain item with its hint | **Yes.** Spying `begin_menu` inside the Document menu for zero calls is decidable; note `draw_menu_bar` itself calls `begin_menu` for the category, so the spy must be scoped to calls INSIDE the already-open Document menu, which the row says. |
| Deletions | **Yes**, and it is the row that catches F9's four missed files automatically — a grep for `PopupState` across `tests/` fails until all nine are repointed. This is the spec's best safety net and is why F9 is a "name them" finding rather than a landing blocker. |

### Invariants the spec states that no row verifies

1. **"`app.py` never imports `popups`"** (R2, stated as the reason the registry is a leaf).
   No row checks it, and it is the invariant F1 shows the spec's own R4 breaks. This is the
   single highest-value missing gate, and it is one AST walk over `app.py`'s `ImportFrom`
   nodes. Without it the cycle can be reintroduced by any later modal that puts a payload
   type in `popups/`.

2. **"the confirm's `on_confirm` runs at most once"**. R7 spies the verb and asserts it "ran
   once", which covers the Enter path for one client. Nothing covers the shape where both
   the `danger_button` click AND the Enter fire in the same frame — the body draws the
   button and reads `is_key_pressed` in the same pass, so a click on `Delete` while Enter is
   also pressed would call `on_confirm()` twice unless the body guards it. Today's revert
   body has the same shape (`primary_button("Revert")` then `standard_button("Cancel")`,
   `copilot_chat.py:203-208`) but no Enter, so this is new surface.
   **Closes it:** `if danger_button(verb) or enter_fired:` as a single branch, plus a row
   asserting the verb spy's call count is exactly 1 when both the click and Enter land.

3. **"a modal's `on_close` runs on the programmatic close path too"**. It does NOT, and the
   spec does not say whether that is intended. R2 says an `App` method that closes a modal
   programmatically "sets `self.modal = None` itself — it IS the cleanup owner". For
   `close_pass_settings` / `close_import_passes` / `close_emoji_picker` that is consistent
   (the method IS the `on_close`). For `settings` → `apply_editor_settings`, `projects` →
   `reset_projects_state`, and (per F2) `lib_picker` → its new cleanup, there is no
   programmatic close path today, so the asymmetry is unobservable. It becomes observable
   the moment one is added. **Closes it:** state the asymmetry as intended —

   > A programmatic close is the cleanup owner by construction: the `App` method that closes
   > a modal IS that modal's `on_close`, so `on_close` cannot be skipped. A modal whose
   > `on_close` is not also an `App` method has no programmatic close path.

4. **"no popup module writes `app.modal`" does not cover `widgets/`.** The revert modal
   moves from `copilot_chat.py` to `popups/confirm.py`, so the immediate instance goes away
   — but the gate's domain should be `shaderbox/` minus `app.py` minus `popups/registry.py`,
   not `popups/` alone, or the next widget that grows a modal re-opens the hole. This is the
   "checker that quietly narrows its own domain" family.

5. **Nothing verifies that `close_modal` and `imgui.close_current_popup()` stay in sync.**
   F2's decline path desyncs them. A frame-driven row asserting that after a declined close
   the modal still draws (its label is still `is_popup_open`) would catch it.

### Diff surface per file, and proportionality

| File | Estimated diff | Proportionate? |
|---|---|---|
| `shaderbox/app.py` | ~150 lines net (`PopupState` → `ModalId`, `close_popup`'s 30-line dispatcher deleted, `_open_popup` → `_open_modal`, `request_confirm`, four `*_confirmed` verbs, `open_copilot_revert` rebuilt, `copilot_revert_target` gone) | **Yes** — `close_popup` alone is a 30-line hand-dispatcher the registry deletes outright. Net negative on `app.py`, which is the point. |
| `shaderbox/popups/registry.py` | ~70 new | Yes |
| `shaderbox/popups/confirm.py` | ~45 new | Yes |
| each of 8 popup modules | ~12 removed, ~10 added (the `draw_*` wrapper becomes a `MODAL = Modal(...)`) | Yes — a near-wash per file, and the size/flags/before move from code into data. |
| `shaderbox/ui.py` | ~14 removed (8 imports + 8 calls), ~2 added; 3 reads retyped | Yes |
| `shaderbox/hotkeys.py` | ~6 lines (the revert branch goes, `close_popup` → `close_modal`) | Yes |
| `shaderbox/widgets/copilot_chat.py` | ~40 removed (`_draw_revert_modal` + `_draw_revert_body`), ~6 added | Yes — pure deletion of a modal that was outside the mechanism. |
| `shaderbox/widgets/pass_list.py` | `_delete_pass` (~17 lines) moves to `App`, the menu item rewires | Yes, and the move is correct: R5's "the verbs own their confirm" needs it on `App` for the palette and chord paths to reach it. |
| `shaderbox/ui_primitives.py` | ~15 removed (`confirm_menu_item`) | Yes |
| `shaderbox/commands.py`, `menus.py`, `document_grid.py`, `pass_graph.py`, `lib_picker/tree.py`, `tabs/document.py` | 3-8 lines each | Yes |
| `tests/` | 9 files repointed + 1 new (`test_confirm.py`, ~150 lines) | Yes |
| Docs (7 files) | moderate | Yes, and the `imgui-ui/SKILL.md` §7.2/§7.3 rewrite is required — §7.2's "the `is_X_open` flag stays on `App` (not on the wrapper) so each modal can do its own per-close cleanup" is the exact sentence the registry supersedes. |

Nothing is out of proportion. The one place where the diff could bloat past its gain is
R5's six-verb table: four of the six verbs (`delete_document_confirmed`,
`reset_document_confirmed`, `copilot_clear_chat_confirmed`, and the lib-file deletes) are
thin wrappers whose only content is building a `ConfirmRequest`. That is the right shape —
it is what makes the confirm identical from every surface — but the spec should say the
wrapper is thin so a post-impl reviewer does not propose collapsing them back into the
call sites.

---

## Should it land?

**Yes, with F1 and F2 fixed in the spec before implementation.** They are the two findings
an implementer would otherwise discover mid-diff and resolve by reaching for a banned
escape (F1 → `TYPE_CHECKING` or an inline import; F2 → a hand-written cleanup at the Close
site, re-creating the exact hand-maintenance the wave exists to delete). F3-F9 are
clarifications an implementer can carry, but each costs a round trip if left implicit.

The minimum edit that makes it landable:

1. R4: `ConfirmRequest` lives in `ui_models.py`.
2. R3: `lib_picker` gets an `on_close`; `draw_modal` forces the close on a body's `False`.
3. R2: one sentence on the double-write direction for the three `close_*` methods.
4. R6: the `app.py`-never-imports-`popups` gate row; the mutex-write gate widened to
   `shaderbox/` minus `app.py` and `registry.py`.
5. R4: the Enter guard against the appearing frame, and a single-branch fire so the verb
   cannot run twice.
6. Files touched: nine test files, named.

---

## False trails

Five things checked that turned out fine, recorded so a later reviewer does not re-derive
them:

1. **`app.py` importing `popups` today.** It does not — `grep -n popups shaderbox/app.py`
   returns two comment lines and no import statement. The spec's "leaf of the popups layer"
   claim is true of the CURRENT code; only R4's `ConfirmRequest` placement breaks it (F1).

2. **A cycle through `registry.py` importing every popup while `ui.py` imports both.**
   Not a cycle: `ui.py` is a consumer of both and nothing imports `ui.py`. `hotkeys.py`
   importing `close_modal` from `popups/registry.py` is likewise fine — `hotkeys.py`
   imports `app.py` already and nothing in `popups/` imports `hotkeys`.

3. **A chord whose own keys leave Enter down when the confirm opens.** Checked every
   destructive spec's `default_chord` in `commands.py`: `RESET_DOCUMENT` is F6,
   `DELETE_DOCUMENT` is Alt+D, `CLEAR_COPILOT_CHAT` is `0` (unbound). None involves Enter,
   and `_dispatch_registry` runs pre-draw anyway. The Enter risk is keyboard MENU nav
   (F7), not chords.

4. **`any_popup_open()` losing the revert confirm from the mutex.** It does not — the
   confirm becomes `ModalId.CONFIRM`, so `self.modal is not None` covers it, and the
   `copilot_revert_target is not None` clause (`app.py:1026-1028`) is correctly deleted
   rather than dropped. Every downstream reader (`hotkeys.py:50`, `:341`, `:370`;
   `ui.py:266`, `:471`, `:588`; `commands.py:78`; `cheatsheet.py:22`; `tabs/code.py:883`,
   `:992`) reads the method, not the field, so they need no touch. The spec is right to
   keep the name.

5. **`_init`'s `first_run` gallery check and the dead-pointer recovery writing the field
   directly.** `app.py:580` (`self.popup_state = PopupState.PROJECTS`) and `app.py:1559`
   (`if first_run and self.popup_state is PopupState.CLOSED`) both become plain `app.modal`
   reads/writes with no behavior change — the mutex reasoning in their comments ("one popup
   at a time", "the single-field popup mutex means only one of them can win") survives the
   rename intact, since `ModalId | None` is still one field. Not a finding; the spec's
   "the three cleanups no longer writing the state" line just does not mention these two,
   and they need no mention.
