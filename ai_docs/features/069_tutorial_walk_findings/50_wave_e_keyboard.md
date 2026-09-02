# 069 W-E: keyboard ownership, regions out, keymap setting, the audit

Implementation spec for wave W-E of feature 069. The parent spec (`01_spec.md § W-E`) fixes the
shape; this file fixes the code. Locked decisions D4, D5 and D7 apply and are not re-opened, and
neither is the audit's rule or any of its outcomes, those live in `02_keybindings.md`, which this
wave implements rather than revisits.

W-F lands before this wave and OWNS the whole editor-style surface in `editor/ffi.py` (W-F § Out
of scope, and § Design decisions item 3): it binds `ed_set_style` / `ed_style`, defines
`class Style(IntEnum): VIM = 0; STANDARD = 1`, and exposes `Editor.set_style` / `Editor.get_style`.
No W-F code calls either. **W-E defines none of that and only calls it**, with W-F's names. This
spec touches `editor/ffi.py` for nothing.

Every citation names a symbol as well as a line, so a shifted line does not invalidate it. Code is
read at HEAD `6a85564`.

## Goal

Three keyboards stop fighting over one machine. The 019 region system comes out whole, no yellow
outline, no `Ctrl+`` cycling, no imgui keyboard nav, none of the nine `App` members that existed to
confine it, so the only focus notion left is the one the editor already owns for its caret. The five
`no_nav_inputs` flags STAY: measurement (§ Design decisions item 1) shows imgui runs basic Tab
traversal whether or not `nav_enable_keyboard` is set, so the flag is not region machinery but the
thing that keeps Tab from walking the panel's sliders and the grid's tiles. What the flags lose is
their region CONDITION, becoming unconditional. The editor's keymap becomes a Setting the user
picks once, vim or standard, applied to every open session through the funnel that already carries
the other five editor settings. And every app chord has exactly one owner in every state, decided by the rule in
`02_keybindings.md` rather than by which surface happens to hold focus, with seven chords moved to
the Alt and F-key tiers to make that true. The result is pinned by a test that reads both keymaps'
chord lists out of the vendored docs, so a re-vendor that grows a keymap turns the gate red instead
of silently shipping a chord that means two things.

## Findings folded

Three, quoted verbatim from `00_findings.md`:

- **#13** (UX feature request, deferred until the editor lib lands it; Code panel / Settings): "when
  I finish the filings, the editor session will already implement the classical keybindings, so we
  integrate them as well. The switch goes in the global Settings — a one-time setting, don't pollute
  the main UI with it."
- **#24** (UX removal, whole app — active-region outline + keyboard nav regions): "remove the active
  zone highlight (the thick yellow boundary). It existed so I could tell which panel arrow keys act
  on; that is overkill. Remove the highlight, the need for active areas, and the arrow-key rotation.
  Everything context-independent. I don't use it and the cost is this coloured boundary noise every
  time I navigate the editor. Rework properly."
- **#26** (DESIGN audit, every keybinding): "we will integrate the standard editor key schema;
  review ALL keybindings so nothing conflicts — standard editor, vim editor, global hotkeys. Clean
  and conflict-less."

## Out of scope

- **The editor's chrome under either keymap** (the status line, the gutter picture, the mode badge,
  the `~` filler, the caret readout that replaces the mode badge under standard) belongs to **W-F**. W-E sets
  the style and W-F's chrome follows it, because the ABI sets keymap and chrome together
  (`ed_set_style`), which is the whole reason the two waves can be independent. W-E touches
  `tabs/code.py` for nothing.
- **A standard-keymap gutter or status design of its own.** The parent spec's Out-of-scope list owns
  this with its own trigger (the maintainer uses standard daily and wants more).
- **`RESET_FEEDBACK`'s command, callback and button** belong to **W-G**. W-E's audit assigns it F6 and W-G
  adds the `CommandSpec`. W-E does not add a command for a callback that does not exist.
- **The Help panel's prose about keyboard navigation**, if any survives. Verified: none does.
  `help_content.py`'s shortcuts section is generated from `COMMAND_SPECS` and its static sections
  never mention regions or arrow nav (checked by reading the module end to end). So there is nothing
  to cut, and the generated table picks the moved chords up with no edit.
- **Rebinding the moved chords back**, or a migration of a user's persisted rebinds. Per the
  repo's no-backward-compatibility rule: `UIAppState.key_bindings` holds `CommandId -> int` and a
  binding for the retired `CYCLE_REGION` id is dropped by `tests/test_retired_ids_pruned.py`'s
  mechanism at load; a user who had rebound a moved command keeps their rebind, which is correct
  because a rebind is a choice. Nothing is migrated and no shim is written.
- **The prose diet on any string this wave touches** belongs to **W-B**. The one new label ("Keymap") is
  written to D1's budget at birth, which is not a diet.

## Design decisions

### 1. The complete deletion list for 019

Everything below goes in one commit. The list is exhaustive: every symbol, every call site, and the
test that fails if one survives.

**One thing is NOT deleted, against the parent spec's bullet: `no_nav_inputs`.** The parent says
"with nav off the flag is inert everywhere, so all four go", and the first draft of this spec
repeated it. **It is refuted by measurement.** imgui's own `ConfigFlags` doc says so
(`.venv/lib/python3.12/site-packages/imgui_bundle/imgui/__init__.pyi`, the `nav_enable_keyboard`
entry: "some features such as basic Tabbing and CtrL+Tab are enabled by regardless of this flag"),
and a headless glfw+imgui rig with `config_flags &= ~nav_enable_keyboard`, two `input_text`s in one
`begin_child`, one synthetic Tab press confirms it:

```
child window_flags = none            ->  a_active=False, b_active=True    Tab MOVED focus
child window_flags = no_nav_inputs   ->  a_active=True,  b_active=False   Tab did NOT move
```

So the flag does live work in the nav-off world and is the only thing suppressing Tab traversal.
Deleting the five uses would not remove a dead flag; it would turn Tab into a focus-traversal key
across the panel's sliders, the grid's `selectable` tiles (a `selectable` IS a Tab stop, unlike an
`invisible_button`) and the chat input. That is new behaviour nobody asked for, in a wave whose
premise is that removal is behaviour-neutral.

**The ruling: the flag stays at every focusable container; only the region CONDITION goes.** The two
conditional forms (`ui.py:750`'s `panel_flags`, `document_grid.py:45`'s `grid_flags`) lose their
`panel_active` / `grid_active` ternary and become unconditional `no_nav_inputs`; the three
unconditional sites (`ui.py:394` editor child, `ui.py:568` copilot bar, `copilot_chat.py:53` chat
window) are untouched. D4's "no imgui keyboard nav" is honoured by keeping Tab from traversing,
which is now the flag's whole job.

**`shaderbox/ui_regions.py`**: `ActiveRegion` deleted. `DocumentTab` stays: it is the persisted
panel tab (`ui_models.py:234 UIAppState.active_document_tab`), which `FOCUS_TAB_*` still drives. The
module docstring names both enums and is rewritten to name one; the sentence explaining why they sit
outside `commands.py` (imgui at module scope in the command table) stays true of `DocumentTab` and
is kept.

**`shaderbox/app.py`**: nine members and one config flag:

| Symbol | Where | Disposition |
|---|---|---|
| `_REGION_CYCLE` | module scope, `:83-87` | deleted |
| `from shaderbox.ui_regions import ActiveRegion, DocumentTab` | `:76` | narrowed to `DocumentTab` |
| `imgui.get_io().config_flags \|= ConfigFlags_.nav_enable_keyboard` | `App.__init__`, `:203` | deleted, with its comment |
| `config_nav_escape_clear_focus_item = False` | `App.__init__`, `:208` | deleted, with its comment; see item 2 |
| `self.active_region` | `App.__init__`, `:379` | deleted |
| `self.region_focus_pending` | `App.__init__`, `:384` | deleted |
| `App.cycle_region` | `:783-785` | deleted |
| `App.focus_move_in_flight` | `:806-811` | deleted |
| `App.region_derive_allowed` | `:813-819` | deleted |
| `App.region_outline_visible` | `:821-829` | deleted |
| `App._yield_editor_to_region` | `:832-837` | deleted |
| `App._set_region` | `:839-853` | deleted |
| `CommandId.CYCLE_REGION: self.cycle_region` | `_build_command_callbacks`, `:525` | deleted |
| the `_set_region(ActiveRegion.PANEL)` line | `App.focus_document_tab`, `:804` | deleted (item 4) |
| the `active_region stays transient` comment | `:1173` | deleted |

`editor_focus_requested`, `editor_defocus_requested`, `editor_focused`, `editor_was_ever_focused`
and `reconcile_popup_focus` all STAY. They are the editor's own focus stop, which #24 explicitly
keeps ("Keep: `editor_focused` … and `copilot_focused`"), and `reconcile_popup_focus` (`:742-754`)
already routes through `editor_focus_requested` with no region involvement.

**`shaderbox/commands.py`**: `CommandId.CYCLE_REGION` and its `CommandSpec` (the `Ctrl+`` entry in
`C.VIEW`). The chord int is not reserved; it becomes free.

**`shaderbox/ui_primitives.py`**: `active_region_outline` deleted entirely (definition at `:17-35`,
including `_REGION_OUTLINE_THICKNESS` at `:5`), and `preview_cell`'s `nav_flatten` parameter
(`:953`), its docstring paragraph (`:965-967`) and the `child_flags |= ChildFlags_.nav_flattened`
branch (`:989-990`). `preview_cell`'s `selectable` (`:1030`) STAYS a `selectable`: its comment says
"so keyboard-nav can land on the cell", which stops being the reason, but the widget stays because
swapping it back to `invisible_button` is a behaviour change nobody asked for and the transparent
`Col_.header*` push already makes it look identical. The comment shrinks to what is now true, the
overlap flag lets the overlay buttons win the click, which is the only non-obvious thing left about
that line.

**`shaderbox/ui.py`**: the import of `active_region_outline` (`:38`), the `ActiveRegion` half of
`:44`, the editor child's derive block and outline (`:402-415`), and the panel's region preamble,
focus grab and derive block:

- `:734-739`: `panel_active`, `focus_panel` and the `region_focus_pending` consume, all three gone;
- **`:747-748`: the `if focus_panel: imgui.set_next_window_focus()` grab.** It sits between the two
  ranges an earlier draft cited and reads `focus_panel`, so leaving it is a `NameError` inside the
  frame. Its sibling grab in `document_grid.py:40` is inside that file's deleted range.
- `:757-764`: the derive block and the outline call.

**All three `no_nav_inputs` flags in this file STAY** (the preamble above): `:394`'s editor child
and `:568`'s copilot bar are already unconditional and are untouched, and `:749-751`'s `panel_flags`
ternary collapses to a plain `window_flags=imgui.WindowFlags_.no_nav_inputs` on the `begin_child`,
keeping `child_flags=ChildFlags_.borders`. `no_nav_focus` at `:65` is untouched (item 2). The parent
spec cites the copilot bar as `:553`, which is the lib-picker menu item; see § Verified premises.

The panel's `document_tab_select_pending` handling (`:742-746`) STAYS: it drives the tab bar's
`set_selected` for `FOCUS_TAB_*` and has nothing to do with regions.

**`shaderbox/widgets/document_grid.py`**: the `active_region_outline` import (`:9`), the
`ui_regions` import (`:12`), the region preamble and derive block (`:38-44`, `:53-60`), the
`nav_flatten: bool = False` parameter of `draw_document_preview_button` (`:21`) and its forward
(`:33`), and the `nav_flatten=True` argument at `:102`. **`:45`'s `grid_flags` ternary collapses to
an unconditional `no_nav_inputs`** rather than going: this is the child the Tab measurement matters
most for, since `preview_cell`'s `selectable` is a real Tab stop and the grid holds one per
document. The `begin_child` keeps its size, borders and the flag.

**`shaderbox/widgets/copilot_chat.py`**: the `active_region_outline` import (`:23`) and the
focus-outline draw at `:150` together with its two-line comment. **`_WINDOW_FLAGS` is untouched**:
`no_nav_inputs` (`:53`) stays for the Tab reason above (the chat's input is a Tab stop), and
`no_nav_focus` (`:51`) stays for the Ctrl+Tab reason in item 2. The block comment at `:46` explains
`no_nav_inputs` as stopping "the nav outline on the programmatically-focused input"; that reason
expires with nav while the flag's job does not, so the sentence is rewritten to name Tab
suppression rather than deleted with the flag. **The chat loses its focus cue.** That is
what #24 asks for ("the copilot chat's focus outline goes too", finding #24's removal shape). The
chat still has a title bar, still reads `copilot_focused`, and still gates `CommandScope.COPILOT`:
only the 2px stroke goes.

**`shaderbox/popups/examples.py`**: the `nav_flatten=True` argument at `:101`.

**`shaderbox/tabs/document.py`**: `ChildFlags_.nav_flattened` at `:257` and the comment above it at
`:252` ("nav_flattened: Tab/arrows reach the sliders without an Enter/Esc window boundary"), which
is false the moment nav is off. The `begin_child` keeps `ChildFlags_.auto_resize_y`.

**`scripts/smoke.py`**: the `ActiveRegion` half of the `ui_regions` import (`:36`), the
`active_region in ActiveRegion` assertion (`:147-149`) and its "Feature 019" comment, and the
`app.cycle_region()` drive at frame 50 (`:272`). The `active_document_tab in DocumentTab` assertion
(`:150-152`) stays, as does the `focus_document_tab(DocumentTab.RENDER)` drive at frame 60.

**Plus a second smoke assertion the first draft missed** (`:214-218`), which would have turned
`make gates` red before any of this wave's own tests ran:

```python
            # Feature 019: nav_enable_keyboard is set in __init__, before any frame —
            # check it here (get_io() reads are frame-context-sensitive mid-loop).
            assert (
                imgui.get_io().config_flags & imgui.ConfigFlags_.nav_enable_keyboard
            ), "nav_enable_keyboard not set"
```

It is **inverted, not deleted**. That is the stronger form: it pins the decision instead of merely
not contradicting it:

```python
            # 069 W-E: nav is OFF app-wide (D4). Checked here for the same reason the
            # old assertion was: get_io() reads are frame-context-sensitive mid-loop.
            assert not (
                imgui.get_io().config_flags & imgui.ConfigFlags_.nav_enable_keyboard
            ), "nav_enable_keyboard is set; D4 removed app-wide nav"
```

**`shaderbox/ui_models.py`** (beyond the `keymap` field of item 4): the comment at `:232-233`,

```python
    # NOT active_region / copilot_focused — those are transient-by-design (focus on
    # launch is a separate UX decision; see todo.md feature-019 deferral).
```

names a deleted concept (so the banned-name test goes red on it) AND points at a `todo.md`
feature-019 deferral that does not exist, verified, `grep -in "region\|keyboard nav\|keymap\|nav_"
ai_docs/todo.md` returns nothing. Rewritten to name `copilot_focused` alone, with the dead pointer
dropped.

**`shaderbox/exporters/youtube.py`**: the comment at `:310-312` says a focus request "lands the nav
outline on the primary Load button". That becomes false the moment `focus_field` stops calling
`set_nav_cursor_visible` (item 2), and no banned token appears in it, so only this list catches it.
The sentence loses its nav-outline clause and keeps what stays true: the credential is loaded by
file pick, and `focus_field`'s one-shot is the caller's. (`ui_primitives.py:470` and
`popups/settings.py:40` also mention `focus_field`, but describe its scroll and one-shot behaviour,
which survives. No edit; named so the sweep is not re-run over them.)

**`shaderbox/hotkeys.py`** (beyond item 5): the Esc comment at `:62` reads "Defocus lives on
CYCLE_REGION and the mouse, never on Esc". `CYCLE_REGION` is gone, so the sentence becomes "Defocus
lives on the mouse, never on Esc". `_handle_escape` itself still needs no change.

**`tests/test_pass_editor_wiring.py`** is covered by item 3.

**The tests that fail if one survives.** Deleting a symbol is checked by the tree compiling, which
`make check`'s pyright pass gives for free on every name above except the string literals. Two
things pyright cannot catch, and two tests cover them: one negative (a retired name came back), one
positive (a flag that must stay went away).

```python
# tests/test_region_system_is_gone.py
# `no_nav_inputs` and `nav_enable_keyboard` are deliberately NOT banned: the first
# STAYS at every focusable container (§ Design decisions item 1: it stops Tab, which
# nav-off does not), and the second is named by smoke.py's inverted assertion.
# `region_` as a bare prefix is also unsafe and is not banned: `get_content_region_avail`
# appears about fifteen times and `scripts/dogfood/judge.py` defines `region_diff`.
# Do not "tighten" this tuple to a prefix.
_BANNED = (
    "ActiveRegion", "active_region", "region_focus_pending", "cycle_region",
    "CYCLE_REGION", "_set_region", "region_derive_allowed",
    "region_outline_visible", "focus_move_in_flight", "_yield_editor_to_region",
    "active_region_outline", "nav_flatten", "nav_flattened",
    "config_nav_escape_clear_focus_item",
)

def test_no_source_file_mentions_the_region_system() -> None:
    hits: list[str] = []
    for path in (*Path("shaderbox").rglob("*.py"), Path("scripts/smoke.py")):
        text = path.read_text()
        for name in _BANNED:
            if name in text:
                hits.append(f"{path}: {name}")
    assert not hits, "\n".join(hits)
```

Note `nav_flatten` is a substring of `nav_flattened`, which is intended: both spellings are retired
and either one hits. The tuple is hand-written rather than derived, which is acceptable because its
domain is a closed list of retired names rather than an enum the code still carries; pyright is the
complement that catches a symbol nobody thought to list.

**The positive test**, which is what finding 1 forces: the flag must not be swept away by a later
reader who reads "nav is off" as "nav flags are dead".

```python
# Every container that HOSTS A FOCUSABLE WIDGET must carry no_nav_inputs on itself
# or on an enclosing container, or Tab traverses it (imgui runs basic Tabbing
# regardless of nav_enable_keyboard).
_FOCUSABLE = (
    "input_text", "input_text_multiline", "input_int", "input_float",
    "drag_int", "drag_float", "slider_int", "slider_float",
    "checkbox", "combo", "selectable", "button",
)
_MODULES = ("ui.py", "widgets/copilot_chat.py", "widgets/document_grid.py")
```

**How the container set is enumerated, so the test cannot pass vacuously.** The enumeration is the
problem here, not the assertion: a test that finds no focusable containers passes while asserting
nothing. Four properties, each measured against the tree at `ccd446b` rather than assumed.

1. **The widget list is validated, not guessed.** Every name in `_FOCUSABLE` must satisfy
   `hasattr(imgui, name)` at test time, so a typo or an imgui rename fails rather than silently
   matching nothing. **imgui-bundle exports no `__all__`** (measured: `hasattr(imgui, "__all__")` is
   `False`), so the check is `hasattr`, not a membership test against a name list. All twelve names
   resolve.
2. **The container set is every `begin_child` / `begin` call with a string-literal first argument in
   the three modules that own them**, found by AST walk. Three modules, not two: an earlier draft
   said "those two modules", which cannot reach `document_grid.py`'s grid child, the one the
   falsifier targets.
3. **A floor assertion pins the count at eight.** Measured at `ccd446b`, the walk finds **eight**
   containers, so `assert len(found) >= 8` before any per-container check:

   ```
   ui.py                     begin       "ShaderBox - UI"          :347   flagged=False
   ui.py                     begin_child "code_editor"             :390   flagged=True
   ui.py                     begin_child "app_panel"               :424   flagged=False
   ui.py                     begin_child "copilot_bar"             :565   flagged=True
   ui.py                     begin_child "control_panel"           :710   flagged=False
   ui.py                     begin_child "document_settings"       :752   flagged=True
   widgets/copilot_chat.py   begin       "Copilot"                 :122   flagged=True
   widgets/document_grid.py  begin_child "document_preview_grid"   :47    flagged=True
   ```

   A floor of five against an actual eight would be slack enough to survive losing
   `document_grid.py` from the walk entirely. Eight leaves none.
4. **A flag reaches its container through a variable, and the walk follows it.** Three of the eight
   pass `window_flags` / `flags` as a NAME, not a literal: `document_settings` via `panel_flags`,
   `document_preview_grid` via `grid_flags` (both ternaries this wave collapses) and `Copilot` via a
   local `flags = _WINDOW_FLAGS | _apply_layout(app)`. A test matching only inline text would score
   all three as unflagged and fail on correct code, so the flag test resolves a `Name` through the
   module's assignments, recursively, to a bounded depth. Verified: with resolution all three read
   flagged; without it, all three read unflagged.

**The rule, with nesting.** A container satisfies it when **it or an enclosing container passes
`no_nav_inputs`**. Ancestry is derived from `with`-statement nesting inside the parsed module: a
container B is inside container A when B's `with` node appears in the AST of A's `with` BODY (the
body, not the whole statement, or a container is its own ancestor and everything passes). Two
containers make this clause necessary rather than theoretical: `ui.py:424` `app_panel` and
`ui.py:710` `control_panel` carry no flag and never should: they are outer wrappers whose focusable
widgets sit in the inner `document_settings` / `document_preview_grid` children that DO carry it,
and flagging the wrappers would be redundant. A rule without the ancestry clause calls both a
violation on correct code. Symmetrically, a focusable call already inside a flagged descendant is
subtracted from the parent's own block before the parent is judged, so a wrapper is not charged for
its child's widgets.

**The honest limit: a container whose widgets are drawn in another function.** The walk is textual
within one module's `with` blocks, so it sees a focusable call only where the call is written. Two
real consequences, both measured, and both stated because the test's value depends on knowing where
it is blind:

- **`copilot_chat.py`'s `Copilot` window has no directly-named focusable call**, its input being drawn
  by a helper called from the block, so deleting `no_nav_inputs` from `_WINDOW_FLAGS` does **not**
  turn this test red. Verified by running the mutation. No automated gate covers the chat's flag;
  manual step 2 is what covers it, which is why that step names the chat as a surface to Tab in.
- **`copilot_chat.py:258`'s `##copilot_history` is a bare `if imgui.begin_child(...)`, not a `with`**,
  so it is outside the container set entirely. It hosts no widget of its own (its
  `input_text_multiline` is the sibling input, drawn outside it) and it sits inside the flagged
  `Copilot` window, so nothing is lost; it is named so the count of eight is not read as a claim
  that eight is every container in the three files.

The grid child is the case the test genuinely covers, and it is the one that matters: its block
calls `imgui.button` and `imgui.checkbox` directly, and the `preview_cell` tiles under it are
`selectable`s, which are Tab stops.

**Falsifier, run rather than asserted:** delete `no_nav_inputs` from `document_grid.py:45`, exactly
what the first draft of this spec prescribed, and the test goes red with
`widgets/document_grid.py:45 "document_preview_grid"`. Reproduced against a mutated copy of the tree.
Without this test that deletion is invisible to every gate: it compiles, pyright passes, the smoke
loop draws fine, and the regression only shows when a user presses Tab and focus walks the document
tiles.

### 2. `nav_enable_keyboard` goes off, and what depends on it

The flag comes off in `App.__init__`. Six things read the world it created; each was checked, and
three need a line changed while three need nothing. **What nav-off does NOT do is make every flag
with `nav` in its name dead.** Two of them are load-bearing afterwards, which is why they are on
this list rather than in item 1's deletions.

**`config_nav_escape_clear_focus_item = False` goes with it.** Its comment says it exists so Esc
does not step the nav cursor out one containment level per press, a behaviour of the nav system.
With nav off, imgui has no nav cursor to clear and the setting has no effect. Leaving it would be a
dead line whose comment describes a mechanism that no longer runs, which is exactly what the repo's
comment rule forbids.

**The Esc glfw filter (`App._install_escape_filter`, `:480-489`) stays exactly as it is.** Its
docstring cites imgui #8059 and the nav-cancel climb, which is what nav-off removes, but the filter
is doing a second job the removal does not touch: it keeps Esc away from imgui while the editor is
focused so the vim keymap owns the modal key, and it swallows a jobless Esc so nothing else reacts.
Both survive. The comment's first clause (naming #8059) is rewritten to state the surviving reason,
per the comment rule; the code is unchanged.

**`ui_primitives.focus_field`'s `set_nav_cursor_visible(True)` (`:522`) goes**, along with the three
docstring sentences explaining it (`:514-517`). With nav off there is no nav cursor and the call is a
no-op, verified by running it: an imgui frame with the flag clear calls `set_nav_cursor_visible`
without error and draws nothing. `set_keyboard_focus_here` itself keeps working with nav off, also
verified by running it (an `input_text` focused on frame 1 reads `is_item_focused` on frame 1 and
`is_item_active` from frame 2 onward, with `nav_enable_keyboard` clear). That is the load-bearing
fact for this whole wave: six sites focus an input programmatically (`lib_picker/tree.py:216,296`,
`lib_picker/search.py:54`, `copilot_chat.py:285`, `widgets/pass_list.py:188`, and `focus_field`
itself) and every one of them keeps working.

**`theme.py:480`'s `style.set_color_(col.nav_cursor, accent_primary)` stays.** It is a style-table
entry, not a behaviour; with nav off nothing draws with it, and deleting one row of a complete
style mapping to save nothing would make the theme's coverage of imgui's colour enum incomplete for
no gain.

**`WindowFlags_.no_nav_inputs` stays at all five sites**, unconditionally. Item 1 carries the
measurement and the ruling; it is repeated in this list because this is where a reader looks for
"what did nav-off kill", and the answer for this flag is "nothing".

**`WindowFlags_.no_nav_focus` stays, on both the main window (`ui.py:65`) and the chat
(`copilot_chat.py:51`).** imgui runs Ctrl+Tab regardless of `nav_enable_keyboard` (its `ConfigFlags`
doc says so explicitly, the same sentence that covers Tab), so the flag is what keeps
`CYCLE_CODE_TAB` from fighting imgui's built-in window-cycle, and `commands.py:128`'s comment
recording exactly that ("Ctrl+Tab is free for us because `WindowFlags_.no_nav_focus` on the main
window (ui.py) suppresses imgui's built-in window-cycle") stays accurate with no edit. It is named
here so nav-off is not read as licence to delete every flag with `nav` in its name.

### 3. `FOCUS_TAB_*` and `_focus_or_add_tab` reduce to open plus focus

`App.focus_document_tab` (`:801-804`) loses its last line and becomes:

```python
    def focus_document_tab(self, tab: DocumentTab) -> None:
        self.active_document_tab = tab
        self.document_tab_select_pending = True
```

The tab still gets selected; `_draw_document_settings` reads `document_tab_select_pending` into
`tab_select_target` and passes `TabItemFlags_.set_selected`, which is the only mechanism that ever
switched the tab. What goes is the region latch that also moved keyboard focus into the panel, which
had no purpose but arrow nav.

`App._focus_or_add_tab` (`:1225-1251`) loses its whole tail. Today it has three behaviours: find or
append the tab, focus the editor when asked, and yield the editor back to a non-editor region when
not asked. The third has no meaning once there are no regions. After:

```python
    def _focus_or_add_tab(self, tab: EditorTab, focus_editor: bool = False) -> None:
        for i, existing in enumerate(self.editor_tabs):
            if existing.path == tab.path:
                self.active_tab_index = i
                break
        else:
            self.editor_tabs.append(tab)
            self.active_tab_index = len(self.editor_tabs) - 1
        self.tab_select_pending = True
        self.editor_was_ever_focused = False
        if focus_editor:
            self.editor_focus_requested = True
```

The `not self.any_popup_open()` branch on the focus request goes too, and this is the one place in
the deletion where dropping a guard needs an argument rather than an assertion. Today the guard
exists because `_set_region(ActiveRegion.EDITOR)` set `editor_focus_requested` AND
`region_focus_pending`, and the latter drives a `set_next_window_focus()` on a background region,
which dismisses an open modal (`/imgui-ui § 8`). With `_set_region` gone, the only thing left is the
`editor_focus_requested` latch, which `ui.py:388` already consumes behind its OWN
`not app.any_popup_open()` gate. So the guard survives, one layer down, at the site that actually
calls `set_next_window_focus`. That is the whole argument; an earlier draft added that
`reconcile_popup_focus` sets the same latch on the popup's CLOSE edge (`app.py:752`), which is a
weaker citation than it reads, since the close edge is the opposite state from the one in question.
The `ui.py:388` evidence carries it alone.

`ensure_shader_tab`, `open_script_for` and the lib-file opener call `_focus_or_add_tab` unchanged;
their signatures do not move.

**`tests/test_pass_editor_wiring.py`** loses both region tests. The parent spec cites `:173`, which
is `def test_a_summon_from_a_non_editor_region_yields_the_editor_back`, confirmed, and it is the
first of two, not the only one. Its sibling
`test_a_summon_from_the_editor_region_keeps_editor_focus` (`:186`) sets `active_region` too. Both
are deleted rather than rewritten, and one test replaces them:

```python
def test_a_summon_does_not_focus_the_editor(app: Any) -> None:
    document_id = _two_pass(app)
    app.editor_focus_requested = False
    app.ensure_shader_tab(document_id, "second")
    assert not app.editor_focus_requested, (
        "summoning a tab must not move keyboard focus; only an explicit "
        "focus_editor=True does"
    )


def test_an_explicit_focus_request_focuses_the_editor(app: Any) -> None:
    document_id = _two_pass(app)
    app.editor_focus_requested = False
    app.ensure_shader_tab(document_id, "second", focus_editor=True)
    assert app.editor_focus_requested
```

The pair preserves what the deleted pair was really pinning, a summon is not a focus move, an
explicit ask is, stated against the latch that survives instead of against the region that does
not. The module's `from shaderbox.ui_regions import ActiveRegion` (`:16`) goes.

### 4. `EditorSettings.keymap`, and everything that must switch with it

**The field.** `shaderbox/ui_models.py`, in `EditorSettings` (`:199-208`):

```python
    keymap: Literal["vim", "standard"] = "vim"
```

`Literal` is already imported (`ui_models.py:5`). It goes at the top of the class, above
`show_whitespace`, because it is the setting that changes what the other five mean rather than a
sibling of them. No `Field` bounds: a `Literal` is its own bound, and `drop_invalid`'s per-key
salvage already turns a hand-edited `"emacs"` into "that one setting resets" via
`validate_assignment`: the same mechanism the numeric bounds comment (`:200-203`) describes, and
the reason the bounds live on the model at all. `EditorSettings` reaches the loader through
`UIAppState`, which `load_model` runs `drop_invalid` over, so nothing new is wired.

**The Settings control.** `shaderbox/popups/settings.py::_draw_body`, in the Editor section, at the
**head of the `label_row` block**: immediately after the `imgui.dummy((0.0, SPACE.SM))` that closes
the three checkboxes, and above Font size.

```python
    label_row(app.font_12, "Keymap", ctrl_w, label_w)
    keymap_idx = _KEYMAPS.index(settings.keymap)
    changed, keymap_idx = imgui.combo("##keymap", keymap_idx, ["vim", "standard"])
    if changed:
        settings.keymap = _KEYMAPS[keymap_idx]
```

with `_KEYMAPS: tuple[Literal["vim", "standard"], ...] = ("vim", "standard")` at module scope so the
index round-trip is typed and the two lists cannot drift apart. `label_row` is the section's existing
label-control primitive (`ui_primitives.py:1088`), used by Font size / Tab size / Line spacing
directly below, so the new row aligns into the same label column with no new helper.

**Why the head of the `label_row` block and not the head of the section.** The Editor section draws
in two idioms: three bare `imgui.checkbox` calls with full-text labels and no label column, then a
`dummy`, then three `label_row` + widget pairs with a right-aligned column. A `label_row` placed
above the checkboxes would make the section read column / no-column / column, and the imgui skill
§ 2 (the authority D1 cites) asks for one alignment idiom per panel. An earlier draft put the combo
first on the argument that the keymap "frames what follows"; that argument is real but the layout
cannot carry it, and the fix for it would be converting the whole Editor section to `label_row`
form, which is W-B's territory and not this wave's.

**Under D1's word budget:** the label is `Keymap`, one word, a noun, inside the 1-2 word budget for
a control label. The items are `vim` and `standard`, which are the names of the things, not
descriptions of them. **No `help_marker`**: D1 says a marker exists only where the label alone is
ambiguous, and a combo whose two visible options are the two keymap names is not. W-B's
`tests/test_ui_prose_budget.py` scores `label_row`'s label argument and will read `"Keymap"` as a
`Constant` of one word with no `FormattedValue`, so the row passes the gate as written.

**The apply path.** `App._apply_editor_settings_to` (`:1381-1394`) gains one line, first, before the
five existing calls:

```python
        editor.set_style(Style.VIM if settings.keymap == "vim" else Style.STANDARD)
```

**"First" is load-bearing, not stylistic** (W-F's pre-implementation review, finding 8).
`ed_set_style` calls `editor_set_keymap` and then replaces the WHOLE `Chrome` from
`chrome_for(style)` (`ffi.odin:940` at the vendored sha), so it resets `LINE_NUMBERS`,
`RELATIVE_NUMBERS` and the status flags to that style's defaults. Measured through the ABI: with
`LINE_NUMBERS` set False by the host, `ed_set_style(h, 0)` leaves it True. So a `set_style` placed
AFTER the existing `set_chrome_flag(ChromeFlag.LINE_NUMBERS, settings.show_line_numbers)` would
silently discard the user's line-numbers setting on every settings apply. `ed_draw_chrome` is NOT
part of `Chrome` and survives the switch, so W-F's `set_draw_chrome(True)` in `get_session` is
unaffected. The same constraint is recorded in `02_keybindings.md` and in W-F's spec.

`Style`, `Editor.set_style` and `Editor.get_style` are all W-F's (its § Design decisions item 3,
which places them beside the existing `Mode` / `Language` / `ChromeFlag` enums and gives
`set_style` the raise-on-false shape `set_palette` and `set_chrome_flag` already use). W-E imports
`Style` from `shaderbox.editor.ffi` alongside the `ChromeFlag` the apply block already imports, and
adds the one call above. **If `Style` is absent when this wave starts, W-F has not landed and W-E
does not begin.** The dependency is a hard edge in the parent spec's § Order, not a soft one.

The call goes in `_apply_editor_settings_to` and not in `get_session` because it IS a setting: it
has a UI control, it changes at runtime, and every open session must follow it. That is the opposite of
W-F's `set_draw_chrome(True)`, which goes in `get_session` precisely because nothing toggles it.
`App.apply_editor_settings` (`:1393-1395`) already loops every session through
`_apply_editor_settings_to`, and `_handle_escape` / the Settings close funnel already call it
(`hotkeys.py:316`, `popups/settings.py:60-63`), so switching the combo and closing the modal
switches every open tab with no new plumbing. The maintainer-visible consequence, and the parent's
manual-verification line, is exactly that: the switch takes effect on modal close, not per keystroke.

**What else must switch with the keymap.** Four things, three in this wave and one already handled:

1. **`_VIM_RESERVED_CHORDS` becomes per-keymap data**, item 5.
2. **`_handle_vim_chord`'s routing**, item 5. Under standard it must not run at all.
3. **The `:`-command handling.** `hotkeys.py::_serve_host_command` drains `ed_take_host_command`
   after every editor input. Under standard there is no command line, so the ABI never produces a
   host command and the drain returns `None` every frame. **It needs no keymap gate**, and adding
   one would be a guard against a value the library cannot produce. Verified from
   `standard_keymap.md`, which lists no `:` binding and states that every unlisted modifier chord
   returns false; and from `vim_coverage.md § Ex commands`, which is the only place the four host
   commands come from. The clipboard handling (`_handle_clipboard`) is likewise keymap-independent
   by the standard doc's own sentence naming Ctrl+C/X/V as the host's, and by vim's register living
   behind `ed_set_register` rather than behind a chord.
4. **The chrome, mode badge, gutter, status line, filler glyph.** W-F's, and it comes for free:
   `ed_set_style` sets keymap AND chrome together (finding #13's verified reading), so a single
   `set_style` call switches both halves. W-E does not touch `tabs/code.py`.
5. **The insert-mode Ctrl+N completion ask** (`hotkeys.py:77-83`) is vim-only in intent: its comment
   calls it "the deliberate completion ask", for a keymap whose Ctrl+N consumes but opens nothing
   under `set_host_completion`. Under standard the keymap owns Ctrl+N itself
   (`standard_keymap.md`: "`Ctrl+Space`, `Ctrl+N`, Opens the completion popup ... with it open,
   moves to the next candidate"). **It needs no keymap gate, verified by probe rather than
   assumed.** Driving the vendored `.so` directly: `ed_set_style(h, 1)` returns True and `ed_style`
   then reads 1; `ed_mode` reads INSERT under standard, as the doc says it always does; and a
   `ed_key(CHAR, CTRL, "n")` returns **consumed=True with `ed_complete_open()` already True**. So
   the branch's own `not editor.complete_open()` conjunct is False on the same frame the keymap
   opened its popup, and `app.editor_completion_requested` is never set under standard. Its comment
   gains one clause naming the self-gate, so a later reader does not add a keymap check that would
   be dead code.

**One consequence of the audit that lands in this same drain, worth stating because it is not
obvious.** `Ctrl+Shift+N`, the new `NEW_DOCUMENT` chord, reaches the editor as
`KeyEvent(CHAR, CTRL|SHIFT, text="N")`, `editor/input.py::_key_char` uppercases under shift, and
`_CHORD_MODS` is ctrl/alt/super, so shift does not suppress the event, it changes its text. The
reserved sets are lowercase, so `_handle_reserved_chord`'s `ch not in _RESERVED_CHORDS[keymap]` is
True for `"N"` and the chord falls through to the registry. Correct as designed, and the
`NEW_DOCUMENT` move depends on it.

### 5. `_VIM_RESERVED_CHORDS` becomes per-keymap data

Today `hotkeys.py:138` holds `frozenset("dufbeyrownphj")`, a hand-typed string, vim-only, consulted
by `_handle_vim_chord` as a fallback after `ed_key` returns unconsumed. Two things change.

**The set becomes a per-keymap mapping**, keyed by the same literal the setting persists:

```python
_RESERVED_CHORDS: dict[str, frozenset[str]] = {
    # Vim: the letters the host still approximates after ed_key declines them.
    # The keymap's own chord list is vim_coverage.md; this is the host's fallback
    # half, which is smaller and is asserted to be a subset of it.
    "vim": frozenset("dufbeyrwnphjm"),
    # Standard consumes every chord it owns inside ed_key and returns false for
    # the rest, so the host approximates nothing.
    "standard": frozenset(),
}
```

Two edits to the letters themselves, both forced by the audit and both recorded in
`02_keybindings.md § _VIM_RESERVED_CHORDS under the rule` (`w` is neither of them: it stays,
see the third bullet and § Review history round 3):

- **`o` is removed.** `vim_coverage.md` lists no `CTRL-O` in either notation (verified by grep for
  `CTRL-O` and `<C-o>`: no match). It was in the host set to consume-noop so `OPEN_PROJECT` could
  not fire mid-insert. After the audit, Ctrl+O is not a vim chord, `OPEN_PROJECT` legitimately owns
  it in all three states, and consuming it would be the host inventing an editor binding.
- **`m` is added.** `vim_coverage.md` lists `CTRL-M` (`+` `CTRL-M` `<CR>`: down, first non-blank) as
  `[x]`. No app command uses Ctrl+M and none may. It is in the set so an unconsumed Ctrl+M mid-insert
  is a noop rather than reaching the registry.
- **`w` stays**, with its NORMAL-mode carve-out intact. Ctrl+W is in NEITHER keymap's list (verified),
  so it is not an editor chord under the rule; what keeps it in the host set is that the host
  IMPLEMENTS insert-mode word-delete itself, in `_delete_word_back`. That is a host behaviour on a
  host-owned chord, which the rule does not govern, and `CLOSE_CODE_TAB` keeps normal-mode Ctrl+W
  exactly as 067 D15 left it.

**The routing gate.** `_handle_vim_chord` is renamed `_handle_reserved_chord` and takes the active
keymap:

```python
def _handle_reserved_chord(app: App, editor: Editor, event: KeyEvent) -> bool:
    if event.code != KeyCode.CHAR or event.mods != KeyMod.CTRL:
        return False
    keymap = app.app_state.editor_settings.keymap
    ch = event.text
    if ch not in _RESERVED_CHORDS[keymap]:
        return False
    ...
```

Under standard the lookup yields the empty frozenset and every chord returns False on the second
test, so the whole vim-approximation body is unreachable without a second branch. That is deliberate:
one gate at the top, expressed as data, rather than an `if keymap == "vim"` wrapped around a
hundred-line body. The call site in `_drain_editor_input` (`hotkeys.py:68`) changes name only.

The docstring block above the set (`hotkeys.py:128-137`) is rewritten. Its current text explains the
vim-only arrangement and the "editor focused → vim wins, unfocused → app wins" resolution, which the
audit replaces; the new text states the two facts a reader needs now, that this is the host's
approximation half and not the keymap's chord list, and that the keymap's list is the vendored doc
the disjointness test reads.

**The subset assertion.** A test asserts every letter in `_RESERVED_CHORDS["vim"]` appears as a
Ctrl chord in `vim_coverage.md`. It is what makes the two edits above impossible to un-make by
hand: the set can be smaller than the doc (the host approximates only some of what the keymap
lists) but never larger, because a letter the doc does not name is the host inventing a binding.
That is the exact defect `o` was.

### 6. The chord moves as `CommandSpec` edits

Seven `default_chord` values change in `shaderbox/commands.py` and one `CommandSpec` is deleted.
Each move is `02_keybindings.md § The moves`; the code is mechanical.

| `CommandId` | Was | Becomes |
|---|---|---|
| `NEW_DOCUMENT` | `_chord(K.n, K.mod_ctrl)` | `_chord(K.n, K.mod_ctrl, K.mod_shift)` |
| `DELETE_DOCUMENT` | `_chord(K.d, K.mod_ctrl)` | `_chord(K.d, K.mod_alt)` |
| `TOGGLE_DOCUMENT_PLAY` | `_chord(K.space, K.mod_ctrl)` | `_chord(K.f5)` |
| `OPEN_SHADER` | `_chord(K.e, K.mod_ctrl)` | `_chord(K.c, K.mod_alt)` |
| `OPEN_SCRIPT` | `_chord(K.r, K.mod_ctrl)` | `_chord(K.r, K.mod_alt)` |
| `TOGGLE_COPILOT` | `_chord(K.j, K.mod_ctrl)` | `_chord(K.j, K.mod_alt)` |
| `OPEN_LIB_PICKER` | `_chord(K.p, K.mod_ctrl)` | `_chord(K.l, K.mod_alt)` |
| `CYCLE_REGION` | `_chord(K.grave_accent, K.mod_ctrl)` | deleted with the id |

Three consequences, each checked rather than assumed:

**`route_flag` starts returning `route_always` for five of them.** `commands.py::route_flag` returns
`route_always` for any chord carrying `mod_alt` and `route_global` otherwise. Five moves land on Alt,
so five commands change routing class. That is the correct direction and is why the Alt tier is the
audit's answer: `route_always` is what makes a chord reach the dispatcher while a text input is
active, and the whole point of moving these is that they must work while the editor is focused. The
two non-Alt moves (Ctrl+Shift+N, F5) keep `route_global`, which is right for them, neither is a verb
one fires mid-typing, and `NEW_DOCUMENT`'s own comment at `app.py:570` already records that a Ctrl
chord routes through an active input.

**`chord_needs_modifier` accepts F5.** `_STANDALONE_KEYS` is `{f1..f12}` and the function exempts
them, so F5 is a legal binding with no registry change. `_BINDABLE_KEYS` already contains every
F-key, so the rebinder can capture F5 too.

**The Help table and the cheatsheet follow with no code change.** `help_content.py::_shortcuts_section`
(`:72-88`) enumerates `COMMAND_SPECS` per category and renders `chord_to_str(spec.default_chord)`;
`widgets/cheatsheet.py::draw` (`:32-40`) does the same over `app.effective_bindings`. Both read the
table, so both show the new chords the frame the table changes.
`tests/test_help_content.py::test_shortcuts_section_lists_every_bound_command` (added by W-C) asserts
each label and chord string appears, so it covers the moved rows automatically.

`CYCLE_REGION`'s deletion also removes the only `C.VIEW` command with no surviving sibling issue:
`FOCUS_TAB_*` ×3 and `TOGGLE_COPILOT` keep the category populated, so
`test_shortcuts_section_covers_every_populated_category` needs no edit.

### 7. The disjointness test, and how it parses the docs

`tests/test_keymap_disjoint.py`, new module. It is the gate D7 owes and the reason the audit table is
a document rather than a promise.

**The parse.** Both docs are markdown and both carry their chords in a structure the file's own
format enforces, so the parse keys on that structure rather than on prose.

*Vim* (`vim_coverage.md`): the chords live in **checklist items**, lines beginning `- [x] ` or
`- [ ] `, whose leading run of backtick-quoted tokens is the key list for that behaviour. Two
notations appear and both are matched: `` `CTRL-X` `` (the motion sections, vim's own `:help`
spelling) and `` `<C-x>` `` (the scrolling and word sections, vim's key-notation spelling). Named
keys appear only in the second form (`<C-Left>`, `<C-Right>`, `<C-Home>`, `<C-End>`).

*Standard* (`standard_keymap.md`): the chords live in the **first cell of a table row**, lines
beginning `| `` `, whose first `|`-delimited cell holds one or more backtick-quoted key names in
`Ctrl+X` / `Ctrl+Shift+Z` / `Shift+Tab` form. Prose outside a table row is never scanned, which is
what keeps the closing paragraph's "Ctrl+X, Ctrl+C and Ctrl+V", a list of chords the editor
explicitly does NOT own, from being read as owned.

```python
_VIM_KEY = re.compile(r"`(?:CTRL-([A-Za-z])|<C-([A-Za-z]|Left|Right|Home|End)>)`")
_STD_KEY = re.compile(r"`((?:Ctrl|Shift|Alt)(?:\+(?:Ctrl|Shift|Alt))*\+[A-Za-z]+)`")


def _vim_chords(text: str) -> set[int]:
    out: set[int] = set()
    for line in text.splitlines():
        if not line.startswith(("- [x]", "- [ ]")):
            continue
        for match in _VIM_KEY.finditer(line):
            out.add(_to_chord(["Ctrl"], match.group(1) or match.group(2)))
    return out


def _standard_chords(text: str) -> set[int]:
    out: set[int] = set()
    for line in text.splitlines():
        if not line.startswith("| `"):
            continue
        for match in _STD_KEY.finditer(line.split("|")[1]):
            *mods, key = match.group(1).split("+")
            out.add(_to_chord(mods, key))
    return out
```

**A `[ ]` row counts.** A chord the keymap has declared and not yet built is still not free: the day
it lands, an app binding on it becomes the collision the audit exists to prevent. So the vim parse
takes both marks, and `Ctrl+Home` / `Ctrl+End` are in the set.

**Chord strings to app chord ints.** `_to_chord(mods, key)` builds the same int `commands._chord`
builds, because it is the only comparison key that means anything (`commands.py`'s module docstring:
"the int is the persistence + comparison key"):

```python
_MODS = {"Ctrl": imgui.Key.mod_ctrl, "Shift": imgui.Key.mod_shift, "Alt": imgui.Key.mod_alt}
_NAMED = {
    "Left": imgui.Key.left_arrow, "Right": imgui.Key.right_arrow,
    "Home": imgui.Key.home, "End": imgui.Key.end,
    "Space": imgui.Key.space, "Tab": imgui.Key.tab,
    "Backspace": imgui.Key.backspace, "Delete": imgui.Key.delete,
}


def _to_chord(mods: list[str], key: str) -> int:
    imgui_key = _NAMED[key] if key in _NAMED else getattr(imgui.Key, key.lower())
    chord = int(imgui_key)
    for mod in mods:
        chord |= int(_MODS[mod])
    return chord
```

A single letter goes through `getattr(imgui.Key, key.lower())`, which is exactly how `commands.py`
builds `K.p` and friends; a named key goes through `_NAMED`, which is small and closed because both
docs together name eight. `_NAMED` raising `KeyError` on a ninth is a feature and is the format-drift
alarm below.

**How it fails when the doc format changes, rather than silently parsing nothing.** This is the
failure the whole test is built to avoid, a checker that quietly narrows its own domain, so it is
guarded three ways, each with its own assertion:

```python
def test_the_vim_doc_still_parses() -> None:
    chords = _vim_chords(_VIM_DOC.read_text())
    assert len(chords) >= 14, f"vim_coverage.md parsed {len(chords)} chords; format changed?"
    assert _to_chord(["Ctrl"], "d") in chords, "the half-page scroll row stopped parsing"


def test_the_standard_doc_still_parses() -> None:
    chords = _standard_chords(_STD_DOC.read_text())
    assert len(chords) >= 12, f"standard_keymap.md parsed {len(chords)} chords; format changed?"
    assert _to_chord(["Ctrl"], "a") in chords, "the select-all row stopped parsing"
```

A floor rather than an equality, so a re-vendor that ADDS a keymap chord does not fail the parse
test, it fails the disjointness test instead, which is where a new collision belongs. The two
sentinel chords are the format canaries: `Ctrl+D` is one of ten chords carried solely in the
`<C-x>` notation (measured: `<C-x>`-only is B, D, E, F, U, Y, Left, Right, Home, End; `CTRL-`-only
is H, J, M, N, P, R; the two notations do not overlap at all), and `Ctrl+A` is likewise carried
solely by a standard table row. A markdown restructure that broke either notation drops its
sentinel and goes red with a message naming the file.

The count floors are 14 and 12 against the 16 and 13 measured at `65264dc`, leaving two and one of
slack so a doc edit that merges two rows is not a false failure while a wholesale format change
still is.

**The disjointness assertion itself:**

```python
def test_no_global_app_chord_belongs_to_either_keymap() -> None:
    owned = _vim_chords(_VIM_DOC.read_text()) | _standard_chords(_STD_DOC.read_text())
    clashes = [
        f"{spec.id.value} on {chord_to_str(spec.default_chord)}"
        for spec in COMMAND_SPECS
        if spec.scope == CommandScope.GLOBAL and spec.default_chord in owned
    ]
    assert not clashes, (
        "these app chords are owned by a focused editor under at least one keymap; "
        "move them to the Alt or F-key tier (ai_docs/features/069_tutorial_walk_findings/"
        f"02_keybindings.md): {clashes}"
    )
```

**Why the scope filter, and why it is exactly one command.** `CommandScope.GLOBAL` is the eligibility
window in which the app fires while the editor is focused. A `COPILOT`-scoped command fires only
when `app.copilot_focused`, and the two focus flags are mutually exclusive, which
`commands.py::scopes_overlap` already encodes and `spec_eligible` already enforces, so the chord
cannot reach the app in a state where the editor could also want it. Exactly one command is
`COPILOT`-scoped: `CYCLE_COPILOT_LAYOUT` on Ctrl+H, which vim lists. Exactly one is `EDITOR`-scoped:
`CLOSE_CODE_TAB` on Ctrl+W, which NEITHER keymap lists, so it would pass the assertion unfiltered
and the filter costs nothing there. A second test pins that this exemption is not a hiding place:

```python
def test_the_only_scoped_chord_a_keymap_owns_is_the_copilot_layout() -> None:
    owned = _vim_chords(_VIM_DOC.read_text()) | _standard_chords(_STD_DOC.read_text())
    excused = {
        spec.id for spec in COMMAND_SPECS
        if spec.scope != CommandScope.GLOBAL and spec.default_chord in owned
    }
    assert excused == {CommandId.CYCLE_COPILOT_LAYOUT}
```

so a future spec that dodges the gate by declaring itself `EDITOR`-scoped turns this red instead.

### 8. The `SELECT`-hue assertion keeps the assertion and loses the rationale

The parent spec leaves this to be decided at implementation by reading `theme.py`. Decided: **the
assertions stay; two sentences of comment change.**

The invariant block is at `theme.py:196-215` (the parent cites `:193`, which is blank, see
§ Verified premises). It is two asserts: `COLOR.SELECT` differs from every accent preset's primary,
and from every `STATE_*` hue. The stated reason for the first is that "SELECT's outline nests inside
accent-outlined regions", the region outline, which this wave deletes.

The assertion survives that because the nesting is not what made it true, only what made it
noticeable. `COLOR.SELECT` still draws as an OUTLINE in three surviving places, each nested inside
something the accent also colours: the document grid's selected-tile border
(`widgets/document_grid.py:94` → `preview_cell`'s `Col_.border`), the examples grid's selected-tile
border (`popups/examples.py:96`, same primitive), and the Telegram sticker grid's selection border
(`exporters/telegram.py:703`). All three sit inside a panel whose surrounding chrome, tab bar,
buttons, focused frame, is accent-coloured, and imgui's `Col_.border` on a focused child is an
accent-adjacent context whether or not a 2px region stroke is drawn on top of it. `SELECT` also
drives the editor's `SELECTION` fill (`theme.py:539`) and the vim VISUAL badge
(`tabs/code.py:27-28`), both of which sit inside a pane whose focus cue is the accent.

So the rule the assertion enforces, a fixed hue with accent-adjacent outline context must not equal
an accent primary, still has instances, and deleting the check would remove a real guard on the
theory that its comment's example expired. Two things change:

- The comment at `theme.py:202-204` ("SELECT's outline nests inside accent-outlined regions") names
  the deleted mechanism. Rewritten to name a surviving one: SELECT outlines a tile inside an
  accent-chromed panel.
- The first assert's failure message ends "...merges with the active-region accent outline when that
  accent is selected". Rewritten to "...merges with the accent chrome around the panel it sits in".

The `conventions.md` entry that owns this rule ("Color roles are SWAPPABLE accent vs FIXED
semantic") carries the same sentence and gets the same correction, see § Docs touched.

## Files touched

| File | What changes |
|---|---|
| `shaderbox/ui_regions.py` | `ActiveRegion` deleted; docstring rewritten for one enum. `DocumentTab` untouched. |
| `shaderbox/app.py` | The fifteen-row deletion table of item 1; `focus_document_tab` and `_focus_or_add_tab` reduced (item 3); `_apply_editor_settings_to` gains the `set_style` call (item 4); the Esc filter's comment rewritten (item 2). |
| `shaderbox/commands.py` | `CommandId.CYCLE_REGION` and its spec deleted; seven `default_chord` values changed (item 6). `route_flag`, `scopes_overlap`, `popup_suppresses`, `capture_chord`, `chord_needs_modifier`, `_BINDABLE_KEYS`, `_STANDALONE_KEYS`: no change, all verified sufficient. |
| `shaderbox/hotkeys.py` | `_VIM_RESERVED_CHORDS` becomes `_RESERVED_CHORDS` keyed by keymap, with `o` out and `m` in; `_handle_vim_chord` renamed `_handle_reserved_chord` and gated on the active keymap; the docstring block rewritten (item 5). Two comments: `:62`'s Esc comment drops `CYCLE_REGION` (item 1), and `:77-83`'s completion-ask comment gains the standard-keymap self-gate clause (item 4, point 5). `_serve_host_command`, `_handle_clipboard`, `spec_eligible`, `_dispatch_registry`, `_handle_escape`: no code change. |
| `shaderbox/ui.py` | The `active_region_outline` import, the `ActiveRegion` import, the editor child's derive block and outline, the panel's region preamble, its `focus_panel` grab at `:747-748` and its derive block (item 1). The three `no_nav_inputs` flags STAY; `:750`'s ternary collapses to unconditional. `no_nav_focus` at `:65` untouched (item 2). |
| `shaderbox/ui_primitives.py` | `active_region_outline` and `_REGION_OUTLINE_THICKNESS` deleted; `preview_cell`'s `nav_flatten` parameter, docstring paragraph and flag branch deleted; the `selectable` comment shortened; `focus_field` loses `set_nav_cursor_visible` and three docstring sentences (items 1, 2). |
| `shaderbox/widgets/document_grid.py` | Both imports, the region preamble and derive block, `draw_document_preview_button`'s `nav_flatten` parameter and its two uses (item 1). `:45`'s `grid_flags` ternary collapses to an unconditional `no_nav_inputs`; the flag stays. |
| `shaderbox/widgets/copilot_chat.py` | The `active_region_outline` import and the focus-outline draw with its comment (item 1). `_WINDOW_FLAGS` untouched: both `no_nav_inputs` and `no_nav_focus` stay; the `:46` comment clause is rewritten to name Tab suppression. |
| `shaderbox/popups/examples.py` | The `nav_flatten=True` argument (item 1). |
| `shaderbox/tabs/document.py` | `ChildFlags_.nav_flattened` and the comment above it (item 1). |
| `shaderbox/ui_models.py` | `EditorSettings.keymap: Literal["vim", "standard"] = "vim"` (item 4); the `:232-233` comment loses its `active_region` mention and its dead `todo.md` feature-019 pointer (item 1). |
| `shaderbox/popups/settings.py` | `_KEYMAPS` at module scope; the Keymap combo at the head of the `label_row` block, above Font size (item 4). |
| `shaderbox/editor/ffi.py` | No change. `Style`, `Editor.set_style` and `Editor.get_style` are W-F's (its § Design decisions item 3); W-E imports `Style` and calls `set_style`. Listed so a reviewer knows the ownership was checked. |
| `shaderbox/theme.py` | Two comment sentences and one assertion message; both asserts and every token unchanged (item 8). |
| `shaderbox/help_content.py` | No change, `_shortcuts_section` reads `COMMAND_SPECS`. Listed so a reviewer knows it was checked. |
| `shaderbox/widgets/cheatsheet.py` | No change, `draw` reads `COMMAND_SPECS` and `effective_bindings`. Listed for the same reason. |
| `scripts/smoke.py` | The `ActiveRegion` import half, the `active_region` assertion, the `cycle_region()` drive; the `nav_enable_keyboard` assertion at `:214-218` INVERTED to assert the flag is clear (item 1). |
| `tests/test_pass_editor_wiring.py` | The `ActiveRegion` import and both region tests deleted; two latch tests added (item 3). |
| `tests/test_region_system_is_gone.py` | New: the banned-name grep AND the positive test that every child hosting a focusable widget still carries `no_nav_inputs` (item 1). |
| `tests/test_keymap_disjoint.py` | New: the parse, the two format canaries, the disjointness assertion, the scope-exemption pin, the reserved-subset assertion (items 5, 7). |
| `shaderbox/exporters/youtube.py` | The `:310-312` comment loses its nav-outline clause (item 1). |
| `tests/test_editor_ffi.py` | Two forced edits the first draft did not foresee, neither visible to pyright: `_drain_app`'s `SimpleNamespace` App stub gains `app_state.editor_settings.keymap`, which `_handle_reserved_chord` now reads; and `test_ctrl_o_is_consumed_noop_while_focused` is INVERTED to `test_ctrl_o_reaches_the_app_while_focused`, since item 5's `o` removal retires exactly the behaviour it pinned. `test_focused_editor_consumes_and_records_chords` is also retargeted (its `CommandId` filter was inert once `OPEN_SCRIPT` left Ctrl+R); the live double-dispatch guard is Ctrl+W under vim NORMAL. The style round-trip still belongs to W-F, which writes the methods (§ Tests). |
| `tests/test_command_routing.py` | No change, `test_no_two_specs_share_a_chord_in_overlapping_scopes` loops `COMMAND_SPECS`. Listed because the seven moves are exactly the kind of half-finished edit it exists to catch. |
| `ai_docs/features/019_keyboard_navigation.md` | The removed-by-069 banner (§ Docs touched). |
| `ai_docs/features/067_custom_editor.md` | The keymap-routing note (§ Docs touched). |
| `ai_docs/conventions.md` | The region entry deleted; the code-editor entry's revisit trigger fired; the SELECT sentence corrected (§ Docs touched). |

## Tests

Each named with its falsifier: the bug that makes it go red.

### `tests/test_region_system_is_gone.py::test_no_source_file_mentions_the_region_system`

Greps every `shaderbox/**/*.py` plus `scripts/smoke.py` for fourteen banned names.

**Falsifier:** leave `tabs/document.py`'s `nav_flattened`, or either `nav_flatten` call-site
argument, or the `active_region stays transient` comment, or `ui_models.py:232`'s `active_region`
mention, or `hotkeys.py:62`'s `CYCLE_REGION`, and the test names the file and the token. Verified to
be a real gap today: pyright cannot see a name inside a comment, and the two `nav_flatten` arguments
would compile fine against a parameter that had merely been defaulted rather than removed.
`no_nav_inputs` and `nav_enable_keyboard` are deliberately NOT in the set (the first stays, the
second is named by smoke's inverted assertion), and `region_` is not banned as a prefix because
`get_content_region_avail` and `scripts/dogfood/judge.py::region_diff` would match it.

### `tests/test_region_system_is_gone.py::test_every_child_hosting_a_focusable_widget_blocks_tab`

The positive half, which finding 1 forces. Walks the AST of `ui.py`, `widgets/copilot_chat.py` and
`widgets/document_grid.py` for `begin_child` / `begin` calls with a string-literal id, asserts at
least **eight** are found (the measured count at `ccd446b`), resolves each container's flag argument
through module assignments when it is a `Name`, and asserts that every container whose own block
calls a focusable widget passes `no_nav_inputs` **on itself or on an enclosing container**.

**Falsifier, run:** delete `no_nav_inputs` from `document_grid.py:45`, which is exactly what the
first draft of this spec prescribed, and it goes red with `widgets/document_grid.py:45
"document_preview_grid"`. Without it that deletion is invisible to every gate: it compiles, pyright
passes, the smoke loop draws fine, and the regression only surfaces when a user presses Tab and
focus walks the document tiles.

**What it does NOT catch, measured:** deleting the flag from `copilot_chat.py`'s `_WINDOW_FLAGS`
leaves the test green, because the chat window's block names no focusable call directly (its input
is drawn in a helper). § Design decisions item 1 states that limit, along with the four non-vacuity
properties and the ancestry rule. The floor of eight is the guard against the walk itself going
vacuous.

### `tests/test_keymap_disjoint.py::test_no_global_app_chord_belongs_to_either_keymap`

**Falsifier:** revert any one of the seven moves. Verified by execution against the vendored docs at
`65264dc`: with the table as it stands at HEAD the assertion collects exactly seven clashes:
`new_document` on Ctrl+N (both keymaps), `delete_document` on Ctrl+D, `open_shader` on Ctrl+E,
`open_script` on Ctrl+R, `toggle_copilot` on Ctrl+J, `open_lib_picker` on Ctrl+P (vim), and
`toggle_document_play` on Ctrl+Space (standard). After the seven edits it collects none. Both
directions were run.

### `tests/test_keymap_disjoint.py::test_the_vim_doc_still_parses` and `::test_the_standard_doc_still_parses`

**Falsifier:** the domain-narrowing failure this whole test exists to prevent. Change
`_VIM_KEY` to match only the `CTRL-X` notation and the vim parse drops from 16 chords to 6, failing
the floor of 14 and the Ctrl+D sentinel. Change `_standard_chords` to scan whole lines instead of
the first table cell and it picks up Ctrl+X / Ctrl+C / Ctrl+V from the closing paragraph, chords
the editor explicitly disclaims, which the disjointness test then reports as three phantom clashes.
Both were run.

### `tests/test_keymap_disjoint.py::test_the_only_scoped_chord_a_keymap_owns_is_the_copilot_layout`

**Falsifier:** move a colliding GLOBAL command to `CommandScope.EDITOR` instead of moving its chord.
The set grows past `{CYCLE_COPILOT_LAYOUT}` and the test names the id. Without this, the scope filter
in the main assertion is a hole a future wave could walk through.

### `tests/test_keymap_disjoint.py::test_every_host_reserved_letter_is_a_vim_chord`

Asserts every letter in `_RESERVED_CHORDS["vim"]` parses out of `vim_coverage.md` as a Ctrl chord,
and that `_RESERVED_CHORDS["standard"]` is empty.

**Falsifier:** re-add `o`. It fails, naming the letter, which is exactly the defect the audit found
by hand and which nothing would have caught. Verified: `grep -o "CTRL-[A-Za-z]" vim_coverage.md`
returns H, J, M, N, P, R and `grep -o "<C-[A-Za-z]>"` returns b, d, e, f, u, y; `o` is in neither
and `m` is in the first.

### `tests/test_editor_ffi.py`: the style round-trip is W-F's

The `set_style` / `get_style` round-trip test belongs to the wave that writes those methods. W-E
adds no test there and asserts no ctypes signature. What W-E pins instead is one layer up: that the
SETTING reaches the call, which is the settings-salvage assertion below plus manual step 5 (three
open tabs all switch). Listed so a reviewer does not read the absence as an untested seam.

### `tests/test_pass_editor_wiring.py::test_a_summon_does_not_focus_the_editor` and `::test_an_explicit_focus_request_focuses_the_editor`

**Falsifier:** the reduction in item 3 dropping the `if focus_editor:` branch entirely (the second
test goes red) or setting `editor_focus_requested` unconditionally (the first does). These replace
two deleted tests that asserted the same pair of behaviours through the region latch.

### `tests/test_persistence_completeness.py`, the keymap field

The module enumerates every persisted model's coverage and fails on an unrostered JSON loader. It
needs no edit for a new field on an existing model, which was checked rather than assumed by reading
it. Listed so a reviewer knows the persistence side was considered. What DOES pin the field's
salvage is one assertion added to the existing editor-settings salvage test: an
`app_state.json` carrying `"keymap": "emacs"` loads with `keymap == "vim"` and every sibling field
intact.

**Falsifier:** declaring the field as `str` instead of `Literal`. The bad value survives validation,
reaches the apply block, and the comparison `settings.keymap == "vim"` silently selects
`Style.STANDARD` for every unrecognised string.

### `make smoke`

The smoke loop keeps running `update_and_draw` with nav off, the region assertions gone and the
`nav_enable_keyboard` assertion inverted. It is the gate on the deletion not having broken a draw
path: `ui.py`, `document_grid.py`, `copilot_chat.py`, `tabs/document.py` and `examples.py` all lose
lines, and the loop draws every one of them.

**Falsifier, and it is not hypothetical:** leaving `scripts/smoke.py:214-218`'s
`assert ... config_flags & nav_enable_keyboard` in place makes the loop fail on its first frame the
moment `app.py:203` is deleted, which turns `make gates` red before any of this wave's own tests
run. The first draft of this spec did not list that assertion; item 1 now inverts it, so the smoke
run pins the decision (nav is off) instead of contradicting it. Two more, both structural: removing
the panel's `panel_flags` local while `:750` still reads it raises `NameError` inside the frame, and
dropping the `focus_panel` grab's definition while `:747` still reads it does the same. All three
are caught in the first frames.

## Manual verification

The parent spec's W-E line, expanded. Each step fails for exactly one reason.

1. **No yellow outline anywhere.** Launch, click the document grid, then the settings panel, then
   the editor, then the copilot chat. No 2px accent stroke appears around any of the four. The
   falsifier for a half-deletion: the grid's outline drew from `document_grid.py:60` and the panel's
   from `ui.py:764`; leaving either shows a stroke on that surface only, so the check must visit all
   four surfaces and not just the editor.
2. **Arrows do nothing outside the editor, and Tab still does not traverse.** With the document
   grid clicked, press every arrow key: no tile selection moves, no focus rectangle appears, no
   slider steps. Then press Tab repeatedly in each of three surfaces: the grid, the settings panel,
   and the copilot chat with its input focused. Focus does not walk from tile to tile, from slider
   to slider, or out of the chat input. **That second half passes because every focusable container
   still carries `no_nav_inputs`, not because nav is off** (imgui runs basic Tabbing regardless;
   § Design decisions item 1), so it is the manual counterpart of the positive test, and it fails
   exactly when a flag was swept away. **The chat is the one that must be checked by hand**: the
   positive test cannot see it (its input is drawn in a helper, so no focusable call appears in the
   window's own block), which makes this step the only cover for that flag. Then click into the editor and press the
   same arrows: the caret moves. The contrast is the point, a step that only checked "arrows do
   nothing" would also pass if the editor had stopped receiving them.
3. **`Ctrl+N` no longer makes a document while the standard keymap is focused.** Settings → Editor →
   Keymap → standard, close the modal, click into the editor, type a few characters, press Ctrl+N.
   The completion popup opens and no document is created. Then press `Ctrl+Shift+N`: a document is
   created. Then click the document grid (defocusing the editor) and press Ctrl+N: nothing happens,
   because the chord has MOVED and is not a fallback. That last press is what distinguishes a real
   move from a focus carve-out.
4. **The same under vim, for the five vim moves.** Keymap → vim, click into the editor, press
   Ctrl+D, Ctrl+E, Ctrl+R, Ctrl+J, Ctrl+P in normal mode. Each does its vim thing (half-page down,
   scroll a line, redo, down a line, up a line) and none opens a document-delete confirm, a shader
   tab, a script tab, the copilot, or the library picker. Then Alt+D, Alt+C, Alt+R, Alt+J, Alt+L
   from inside the editor: each fires its app command without leaving the buffer's text alone
   changed.
5. **Switching keymap in Settings switches every open tab.** Open three editor tabs (a shader, a
   script, a lib file). Settings → Keymap → standard → close. Click into each of the three in turn
   and type a character: each inserts directly with no `i` needed, and Escape does not enter a
   normal mode. Switch back to vim and repeat: each requires `i`. The three-tab shape is the check:
   `_apply_editor_settings_to` runs per session, so a bug that applied the style only to the active
   session passes on one tab and fails here.
6. **The keymap survives a restart.** Set standard, quit, relaunch, open a shader. It is still
   standard. Then hand-edit `app_state.json`'s `editor_settings.keymap` to `"emacs"` and relaunch:
   the app starts, logs one salvage line for that key, and the editor is vim. Nothing else in the
   Editor section resets.
7. **F5 and F6 reach the transport.** With the editor focused and the caret mid-line, press F5: the
   document script plays or stops. Press F6 once W-G lands: the canvas clears. Neither types a
   character into the buffer, which is what `_STANDALONE_KEYS` exists to guarantee.
8. **The cheatsheet and Help show the new chords.** Alt+/ opens the cheatsheet: the Document group
   reads `Ctrl+Shift+N` and `F5`, and there is no "Cycle region" row anywhere. F1 → Keyboard
   shortcuts: the same. This is the consumer check for the generated tables, they read
   `COMMAND_SPECS`, and a reader of the wire is what `dev_flow.md` asks for.
9. **The rebinder still captures.** Settings → the command list → click a chord button for a moved
   command → press Alt+K. It captures and displays `Alt+K`. Escape cancels. This exercises
   `capture_chord` and `chord_needs_modifier` on the moved rows, which the audit changed the defaults
   of but not the machinery for.
10. **Programmatic focus still lands.** Alt+A (add pass): the inline name input has the keyboard
    immediately, with no accent rectangle around it. Alt+L (library picker): the search field has the
    keyboard. Ctrl+J → the chat input has the keyboard. All three used `set_keyboard_focus_here`
    under nav; this is the check that nav-off did not break them. The "no rectangle" half is bought
    by nav-off (there is no nav cursor to draw) rather than by the flag, which is doing the Tab half
    instead. Those are the two jobs `no_nav_inputs` used to do at once, now split between them.

## Verified / corrected premises

Every citation and claim the parent spec's W-E section and finding #24/#26 make, checked against the
tree at `6a85564`. Line numbers are the real ones as of this reading.

| Parent-spec or finding citation | Verdict |
|---|---|
| `nav_enable_keyboard` is ON at `app.py:185` (#24) | **Corrected.** It is `app.py:203`. `:185` is a moderngl `gc_mode` comment, not an io setting at all. |
| Nav is confined via `no_nav_inputs` at `ui.py:379,709` and `document_grid.py:45` (#24) | **Corrected on two of three lines, and the count is five.** The editor child's flag is `ui.py:394`, the panel's conditional is `ui.py:749-751`, and `document_grid.py:45` is correct. #24's list of three misses the copilot bar (`ui.py:568`) and the chat window (`copilot_chat.py:53`); the parent spec's "four" then misses the grid. Five is the census (next row). Confinement is the right word for what these flags did in 019 and the wrong word for what they do after this wave, where they suppress Tab unconditionally rather than confining nav to one region. |
| The four `no_nav_inputs` sites are `ui.py` (editor child, the copilot bar child at `:553`, the panel child) and `copilot_chat.py:53` | **Corrected twice: the count is FIVE, and one line is wrong.** `git grep -n no_nav_inputs` over `shaderbox/**.py` returns six lines, one a comment (`app.py:202`), leaving five flag uses: `ui.py:394` (editor child), `ui.py:568` (copilot bar), `ui.py:750` (panel, conditional), `copilot_chat.py:53` (chat window) and **`document_grid.py:45` (grid child, conditional)**, the fifth, which the parent's list and this spec's first draft both omitted. `copilot_chat.py:53` is exact; `ui.py:553` is `app.open_shader_lib_picker()` inside the Library menu, not a flag. An earlier draft of this spec asserted "four" four times and used it to argue the finding's "three" was wrong; a census number in a spec is read as established, so the number is corrected rather than argued. |
| The flag is inert with nav off, so all of them go (parent spec, and this spec's first draft) | **REFUTED by measurement, and the wave's central premise is reversed.** imgui's `ConfigFlags` doc states that basic Tabbing and Ctrl+Tab run regardless of `nav_enable_keyboard`, and a headless rig with the flag clear confirms it: two `input_text`s in a plain `begin_child` give `a_active=False, b_active=True` after a synthetic Tab (focus MOVED), while the same child flagged `no_nav_inputs` gives `a_active=True, b_active=False` (focus held). So all five flags stay and only the two region CONDITIONS go. See § Design decisions item 1 and § Review history round 1 finding 1. |
| `WindowFlags_.no_nav_focus` is unaffected by nav-off | **Confirmed, and it was missing from this spec entirely.** `ui.py:65` and `copilot_chat.py:51` carry it, and `commands.py:128` records that Ctrl+Tab is free for `CYCLE_CODE_TAB` because of it. Since imgui runs Ctrl+Tab regardless of `nav_enable_keyboard`, that comment stays true and the flag stays load-bearing. Named in item 2 so a later sweep does not delete it on a name match. |
| `active_region_outline` is at `ui_primitives.py:17` (#24) | **Confirmed.** `def active_region_outline(foreground: bool = False) -> None:` at `:17`. |
| It has **six call sites** (parent spec) | **Corrected.** There are **four** draw calls (`ui.py:415`, `ui.py:764`, `document_grid.py:60`, `copilot_chat.py:150`) plus three imports and the definition, eight references. Finding #24's own list names the four correctly (`ui.py:400,723`, `document_grid.py:60`, `copilot_chat.py:150`), with the two `ui.py` lines off by fifteen and forty-one. The parent's "six" appears to have counted imports as sites. The deletion list in item 1 enumerates all eight. |
| The App state machine is `app.py:748–820` (#24) | **Corrected.** The methods run `:783` (`cycle_region`) through `:853` (`_set_region`); `:748-782` is `reconcile_popup_focus`, `copilot_send` and `_copilot_busy_blocked`, none of which are region state. The two attributes are further up, at `:379` and `:384`. |
| The state machine names `region_focus_pending` among its members (#24) | **Confirmed**, and it is the member with the most consumers: `ui.py:737,739`, `document_grid.py:41,43`, plus four writes in `app.py`. |
| There are "derive-from-live-focus blocks in three draw sites" (#24) | **Confirmed.** `ui.py:402-409` (editor), `ui.py:757-762` (panel), `document_grid.py:53-58` (grid). Each is the same four-line shape gated on `region_derive_allowed()`. |
| `CYCLE_REGION` is Ctrl+` at `commands.py:151` (#24) | **Confirmed.** `:151` is `CommandId.CYCLE_REGION,` inside the spec whose `_chord(K.grave_accent, K.mod_ctrl)` is `:153`. |
| `FOCUS_TAB_*` set the PANEL region via `focus_document_tab` (#24) | **Confirmed.** `app.py:801-804`; the last line is `self._set_region(ActiveRegion.PANEL)`. |
| The summon/yield dance in `_focus_or_add_tab` is at `app.py:~820` (#24) | **Corrected.** `_focus_or_add_tab` is `app.py:1225-1251`; `:820` is inside `region_derive_allowed`. The dance itself is `:1246-1251`. |
| `conventions.md:328` carries the region rule (#24) | **Corrected.** The entry ("**App-wide keyboard nav is region-confined (`nav_enable_keyboard` ON).**") begins at `conventions.md:331`. Per the docs rule the durable citation is the entry name, which is what § Docs touched uses. |
| `conventions.md:341` and `theme.py:193` carry the SELECT hue rule (#24) | **Corrected, both.** The conventions entry is "**Color roles are SWAPPABLE accent vs FIXED semantic...**" beginning at `:344`. `theme.py:193` is a blank line; the invariant block runs `:196-215` with the first assert at `:205`. |
| The `SELECT` assertion's stated reason is nesting inside the region outline | **Confirmed**, in two places: the comment at `theme.py:202-204` and the assertion message at `:206-209`, which ends "merges with the active-region accent outline". Item 8 rewrites both and keeps the asserts. |
| One test names the region machinery, `tests/test_pass_editor_wiring.py:173` | **Confirmed as the line, corrected as the count.** `:173` is `def test_a_summon_from_a_non_editor_region_yields_the_editor_back`, but there are TWO such tests: it reads `active_region` at `:178`, and `test_a_summon_from_the_editor_region_keeps_editor_focus` (`:186`) reads it at `:189`. The module's `ActiveRegion` import is `:16`. Item 3 deletes both. |
| The removal keeps `editor_focused` and `copilot_focused` (#24) | **Confirmed as necessary and sufficient.** `editor_focused` gates `CommandScope.EDITOR` in `spec_eligible` (`hotkeys.py:258`) and the editor drain (`:46`); `copilot_focused` gates `CommandScope.COPILOT` (`:260`) and `escape_has_job` (`app.py:498-506`). Neither is written by any region method. |
| `DocumentTab` in `ui_regions.py` stays because it is the persisted panel tab (#24) | **Confirmed.** `ui_models.py:234` persists `active_document_tab: DocumentTab`, mirrored at `app.py:1174` (load) and `:1607` (save). |
| App chords today include "Ctrl+Y" as a collision (#26 lists Ctrl+Y redo vs vim `y`) | **Refuted as an app collision, confirmed as an editor-owned chord.** No `CommandSpec` binds Ctrl+Y. The parent's round-1 review already recorded this ("Ctrl+Y noted as editor-owned, not an app collision"); `02_keybindings.md`'s editor-owned table records it in both keymap columns so nothing lands on it. |
| `_VIM_RESERVED_CHORDS` is at `hotkeys.py:139` and is `d u f b e y r o w n p h j` (#26) | **Corrected on the line, confirmed on the letters.** The `frozenset("dufbeyrownphj")` is `hotkeys.py:138`; `:139` is the blank line after it. The thirteen letters are exact. |
| The vim keymap "reserves" that set | **Refuted as a description of the keymap.** The set is the HOST's approximation half, consulted only after `ed_key` returns unconsumed (`hotkeys.py:68`), and it disagrees with the keymap's own doc in two places: `o` and `w` appear in no row of `vim_coverage.md` (grep for `CTRL-O`, `<C-o>`, `CTRL-W`, `<C-w>`: no matches), and `CTRL-M`, which the doc lists as `[x]`, is absent from the set. Item 5 corrects both. |
| Standard owns "Ctrl+A select all, Ctrl+Z undo, Ctrl+Y / Ctrl+Shift+Z redo, Ctrl+Space / Ctrl+N completion, Ctrl+Left/Right/Home/End, Ctrl+Backspace/Delete, Shift+Tab" (#26) | **Confirmed, all thirteen**, against `shaderbox/resources/editor/standard_keymap.md` at `VERSION` `65264dc`. The parse in item 7 returns exactly that set. |
| "everything else with a modifier is returned to the host" (#26) | **Confirmed verbatim** in the doc's closing paragraph: "Every other key with Ctrl, Alt or Super held -- Ctrl+X, Ctrl+C and Ctrl+V among them -- returns false from `ed_key` and is the host's." |
| "F-keys and Alt are untouched by both keymaps" (parent, open question 4) | **Confirmed by parse.** Neither doc's parsed set contains any `Alt+` chord or any F-key; the vim doc contains no `Alt` token at all and the standard doc names Alt only in the fall-through sentence. |
| The keymap docs "are not in this repo until W-F copies them under `resources/editor/`" (parent, W-E) | **Refuted, they are already vendored.** `shaderbox/resources/editor/vim_coverage.md` and `standard_keymap.md` are committed at HEAD, landed by `4d412fe` ("069 W-F: re-vendor libeditor.so at 65264dc"). W-F's own spec confirms they are byte-identical at `c5c6ae2`, so the tests in item 7 can be written and run before W-F's second re-vendor and stay correct after it. |
| The lists are read "from the editor repo's `docs/`" (parent, W-E) | **Corrected to the vendored copies.** Reading the editor repo would make the test depend on a checkout that does not exist on a fresh clone; the vendored files are in the bundle allowlist path and are what ships. |
| `ed_set_style(h, 0 Vim / 1 Standard)` + `ed_style` set keymap AND chrome together (#13) | **Confirmed as the finding states**, from the editor repo's `ffi/ffi.odin:892,904`. W-F binds both AND defines `Style` / `Editor.set_style` / `Editor.get_style` (its § Design decisions item 3, § Out of scope). An earlier draft of this spec re-defined all three under the name `EditorStyle`; that was a duplicate and is removed. W-E calls them. |
| `editor/ffi.py` binds neither `ed_set_style` nor `ed_style` (#13) | **Confirmed at HEAD `6a85564`.** `grep` for either name in `editor/ffi.py` returns nothing, and no `Style` enum exists. W-F lands all of it; this wave assumes it and defines none of it. |
| The field goes at `ui_models.py:199` and the combo at `popups/settings.py:86` (#13) | **Confirmed as locations.** `EditorSettings` opens at `ui_models.py:199`; `popups/settings.py:86` is `separator_text("Editor")`. The combo goes immediately after it (item 4). |
| `_apply_editor_settings_to` is the funnel, `app.py:1306` (#13) | **Corrected.** It is `app.py:1381-1394`. `:1306` is unrelated. The claim that it is the one place per-editor settings flow is confirmed: it is called from `get_session` (`:1361`, fresh handle) and `apply_editor_settings` (`:1394`, every session), and nothing else sets an editor setting. |
| "Style is a property of the editor STATE per handle, so every open session gets the switch on apply" (#13) | **Confirmed by the code path.** `apply_editor_settings` loops `self.editor_sessions.values()`, and sessions are per-FILE (`conventions.md`'s code-editor entry), so a three-tab document has three handles and all three are told. |
| The host's vim layer must switch off with the keymap: `_handle_vim_chord`, the `:`-command handling, the mode badge, the gutter (#13) | **Confirmed for two, refuted for one, deferred for one.** `_handle_vim_chord` needs the gate (item 5). The `:`-command handling needs none: `_serve_host_command` drains a value the standard keymap cannot produce (item 4, point 3). The badge and gutter are W-F's and follow from `ed_set_style` alone. |
| Removing the region system frees `focus_field`'s nav-cursor call | **Found, not in any parent citation.** `ui_primitives.py:522` calls `set_nav_cursor_visible(True)` and three docstring sentences explain why nav-routed focus needs it. With nav off it is a no-op. Item 2 removes it. This is the one blast-radius item no parent-spec bullet or finding names. |
| `set_keyboard_focus_here` keeps working with `nav_enable_keyboard` off | **Confirmed by execution.** A headless glfw+imgui rig with the flag clear: an `input_text` focused on frame 1 reads `is_item_focused()` True on frame 1 and `is_item_active()` True from frame 2. Six sites depend on this. |
| `set_nav_cursor_visible` and a transparent `selectable` run clean with nav off | **Confirmed by execution** in the same rig, no exception, no assert. So the deletion order does not matter and `preview_cell` needs no widget change. |
| `preview_cell`'s `selectable` exists "so keyboard-nav can land on the cell" (`ui_primitives.py:1030`) | **Confirmed as the stated reason and superseded.** The reason expires with nav; the widget stays (item 1) because `allow_overlap` plus the transparent header colours is what lets the overlay buttons win the click, which is independent of nav. |
| `theme.py:480` maps `col.nav_cursor` to the accent primary | **Confirmed**, and it stays (item 2): a style-table row, not a behaviour. |
| `scripts/smoke.py` asserts `active_region in ActiveRegion` and drives `cycle_region()` | **Confirmed.** The assertion is `:147-149` under a "Feature 019" comment; the drive is `:272` at frame 50. Both go. |
| The chord-uniqueness test covers the moves | **Confirmed.** `tests/test_command_routing.py::test_no_two_specs_share_a_chord_in_overlapping_scopes` compares every pair of `COMMAND_SPECS` and needs no edit; it is what catches a half-finished move that lands two commands on Alt+R. |
| `route_flag` returns `route_always` for Alt chords | **Confirmed** (`commands.py::route_flag`, the `chord & int(imgui.Key.mod_alt)` branch) and confirmed as the reason the Alt tier works: five of the seven moves change routing class as a consequence, in the direction the audit wants. |
| `chord_needs_modifier` and `_BINDABLE_KEYS` accept F5 | **Confirmed.** `_STANDALONE_KEYS` is `{int(K.f1)..int(K.f12)}` and `chord_needs_modifier` returns False for a bare F-key; `_BINDABLE_KEYS` includes `f1`-`f12`. No registry change for F5 or F6. |
| The seven moves are the ones the rule forces | **Confirmed by execution** against the parsed doc sets: exactly seven GLOBAL specs collide, and all eight proposed chords (Ctrl+Shift+N, Alt+D, F5, Alt+C, Alt+R, Alt+J, Alt+L, F6) are absent from both sets. |
| Exactly one non-GLOBAL spec's chord is keymap-owned | **Confirmed by execution.** `CYCLE_COPILOT_LAYOUT` on Ctrl+H (vim). `CLOSE_CODE_TAB` on Ctrl+W is `EDITOR`-scoped but its chord is in neither list, so it needs no exemption. |
| `help_content.py` and `widgets/cheatsheet.py` need no edit for the moves | **Confirmed by reading both.** `_shortcuts_section` (`help_content.py:72-88`) and `cheatsheet.draw` (`:32-40`) each enumerate `COMMAND_SPECS`. Neither hard-codes a chord string. |
| `019_keyboard_navigation.md` is 462 lines and is the subsystem being removed (#24) | **Confirmed as the subject, corrected on the length.** The file is 462 lines by `wc -l`; the claim holds. Its § Goal opens with "**`nav_enable_keyboard` ON**", which is the sentence the banner must contradict. |

Corrected or refuted: **20** (16 corrections, 4 refutations). The four refutations are the vim
keymap "reserving" the host's letter set, Ctrl+Y as an app collision, the keymap docs not yet being
in the repo, and the `:`-command handling needing a keymap gate. One item, `focus_field`'s
nav-cursor call, is a find rather than a check: no parent bullet or finding names it, and it is the
only place in the wave where nav-off silently turns a live call into a no-op.

## Docs touched

- **`ai_docs/features/019_keyboard_navigation.md`**: a banner at the top of the file, above the
  existing intro paragraph, in the shape the repo uses for a superseded spec: three lines naming
  what removed it, when, and what survived, so a reader who greps `ActiveRegion` and lands here does
  not implement it again. The body is not edited and not deleted: it is the record of what was
  built, and `dev_flow.md`'s "resolved entries get deleted in the resolving commit" governs live
  registers, not a feature spec that describes shipped-then-removed work. The `roadmap.md` row for
  019 flips to superseded in the same commit.
- **`ai_docs/conventions.md`**: three edits, all in `## Design decisions`:
  1. The entry **"App-wide keyboard nav is region-confined (`nav_enable_keyboard` ON)"** is deleted
     whole. Its revisit trigger ("Revisit if a region is added/removed or the confinement model
     changes") is what this wave fires, and the model it describes has no instances left.
  2. The entry **"Inline editor state lives on `App`; disk is the source of truth; one libeditor
     instance per opened FILE"** has its revisit trigger fired by D5, the trigger reads "or a
     non-modal keymap ships editor-side", which is exactly what lands here. Its opening sentence
     ("The code editor is the maintainer's own vim-modal library (feature 067)") is rewritten to
     "the maintainer's own library, vim-modal or standard per `EditorSettings.keymap` (features 067,
     069)". The trigger clause about the non-modal keymap is removed, having fired; the entry's other
     two triggers (durable per-tab state, a fourth editable kind) stay.
  3. The entry **"Color roles are SWAPPABLE accent vs FIXED semantic"** has its parenthetical
     "(its outline nests inside the accent's region outline)" replaced with the surviving instance
     ("its outline nests inside the accent-chromed panel it sits in"). The rule and its enforcement
     are unchanged; only the example is.
- **`ai_docs/features/067_custom_editor.md`**: a note where decision 15 states the vim-only
  `_VIM_RESERVED_CHORDS` routing (the paragraph beginning "**Vim-reserved Ctrl chords while focused**",
  whose "The FULL reserved set is `_VIM_RESERVED_CHORDS` = d u f b e y r o w n p h j" is the sentence
  069 supersedes) and a second where decision 3's editor-child flags are listed (the line stating the
  child "gets `no_nav_inputs | no_scrollbar | no_scroll_with_mouse`"). Both notes name 069 W-E and
  say what replaced the mechanism, one clause each; neither rewrites the decision, which is the record
  of what 067 shipped.
- **`ai_docs/roadmap.md`**: the 069 row's status and the Active-context banner, per `dev_flow.md`
  step 9. The banner is rewritten, not appended.
- **`ai_docs/todo.md`**: untouched. It is frozen drain-only and this wave resolves no entry in it
  (checked: `grep -in "region\|keyboard nav\|keymap\|nav_"` returns nothing). What that check
  missed on the first pass is the other direction: `ui_models.py:233` points AT a "todo.md
  feature-019 deferral" that does not exist, so the dead pointer is dropped with the comment
  (item 1's `ui_models.py` row).
- **`.claude/skills/imgui-ui/SKILL.md`**: § 8 carries three bullets written for a nav-on world: the
  `invisible_button`-is-not-a-nav-stop bullet, the `set_keyboard_focus_here`-leaves-a-nav-rectangle
  bullet with its `no_nav_inputs` prescription, and the `set_nav_cursor_visible`-is-the-wrong-fix
  bullet. **They stay as written.** The skill is cross-project and states what imgui does, not what
  ShaderBox configures; a project with nav on still needs all three. What is added is one sentence at
  the head of the pair, naming ShaderBox as a project that runs with `nav_enable_keyboard` OFF, and
  saying precisely what that does and does not buy: no nav cursor and no arrow traversal, but Tab
  and Ctrl+Tab still run, so the `no_nav_inputs` / `no_nav_focus` flags stay load-bearing. That is
  the ShaderBox-specific fact and § 9 is where ShaderBox specifics live, so it goes there with a
  pointer rather than into § 8. **Plus one correction in § 9 itself:** its two-editor-focus-flags
  bullet (`SKILL.md:604`) says `editor_was_ever_focused` is cleared "only by explicit defocus, Esc /
  **arrow nav** / target switch". Arrow nav is what this wave deletes, so `arrow nav` is struck,
  leaving Esc / target switch.

## Open questions

Each carries a robust default, marked as such; none blocks implementation.

1. **Should the copilot chat keep a focus cue of some other kind?** Default, taken: **no cue at
   all.** #24 asks for the outline to go and names the chat's specifically in its removal shape. The
   chat has a title bar, which imgui already renders differently when the window is focused
   (`Col_.title_bg_active` vs `Col_.title_bg`, both mapped in `theme.py`), so a focus signal survives
   without a stroke. Revisit if the maintainer reports losing track of whether the chat has the
   keyboard; the fix would be a title-bar tint, not a re-added outline.
2. **Should `Editor.set_style` be called at session creation as well as in the settings funnel?**
   Default, taken: **the funnel only.** `get_session` calls `_apply_editor_settings_to` on every
   fresh handle (`app.py:1361`), so a session created after a keymap switch gets the style through
   the same one line. Adding a second call site would be two paths to one property, which is what
   the funnel exists to prevent.
3. **Should the disjointness test also assert the app's Alt and F-key chords are unique among
   themselves?** Default, taken: **no, because `test_command_routing.py` already does it.**
   `test_no_two_specs_share_a_chord_in_overlapping_scopes` compares every pair in `COMMAND_SPECS`,
   which is a stronger statement than "the Alt tier is unique" and covers the four new Alt chords.
   Duplicating it in the keymap module would be a second home for one rule.
4. **What should `_RESERVED_CHORDS` be keyed by, the literal, or an enum?** Default, taken: **the
   `Literal["vim", "standard"]` the setting persists.** W-F's `Style` enum exists on the ffi side
   for the ABI's integers, but keying the host table by it would mean converting the persisted string
   at every lookup for no gain, and the two names are the same two names. Revisit if a third keymap
   ships, at which point the enum earns its place on both sides.
5. **Does `Ctrl+M` need a host approximation, or is consume-noop right?** Default, taken:
   **consume-noop**, which is what adding `m` to the reserved set gives. The keymap implements
   Ctrl+M as "down, first non-blank" and `ed_key` consumes it, so the host fallback runs only in the
   modes where the keymap declined, the same position `d`/`f`/`b`/`e`/`y`/`r` are already in. Adding
   a host approximation for a motion the library implements is the thing 067 D15's "future keymap
   growth shadows them automatically" note argues against.

## Review history

**Round 1, pre-implementation review** (`reviews/wave_e_pre.md`, one reviewer, correctness & design
plus verification & blast-radius): parent coverage PASS, audit correctness PASS (the seven moves
re-derived independently from the vendored docs, exact match, plus confirmation that the post-move
table has zero duplicate chords), deletion completeness **FAIL**, keymap design PARTIAL, test
falsifiability PARTIAL, docs PARTIAL. Twelve findings, **all accepted, none rejected.**

**Finding 1 reverses the parent spec's W-E bullet, so it is recorded first and in full.** The parent
says "with nav off the flag is inert everywhere, so all four go", and this spec's first draft
repeated it as its central premise. It is false. imgui's `ConfigFlags` doc states that basic Tabbing
and Ctrl+Tab run regardless of `nav_enable_keyboard`, and the reviewer measured it; I re-ran the
probe rather than relaying it, on a headless glfw+imgui rig with `config_flags &=
~nav_enable_keyboard`, two `input_text`s in one `begin_child`, one synthetic Tab:

```
child window_flags = none            ->  a_active=False, b_active=True    Tab MOVED focus
child window_flags = no_nav_inputs   ->  a_active=True,  b_active=False   Tab did NOT move
```

So `no_nav_inputs` is not region machinery that nav-off makes dead; it is the only thing stopping
Tab from traversing the panel's sliders, the grid's `selectable` tiles and the chat input. Deleting
the five uses would have shipped new behaviour in a wave whose premise is that removal is
behaviour-neutral. **Ruling: the flag stays at every focusable container; only the region CONDITION
goes**, so the two ternaries collapse to unconditional flags and the three unconditional sites are
untouched. D4's "no imgui keyboard nav" is honoured by keeping Tab from traversing, which is now the
flag's whole job. Folded into: § Goal, item 1's preamble and its `ui.py` / `document_grid.py` /
`copilot_chat.py` rows, item 2's list, the banned-token test (`no_nav_inputs` dropped from `_BANNED`,
a positive test added with its three non-vacuity properties), § Files touched, § Tests, § Verified
premises, and manual step 2.

The other eleven:

2. **`no_nav_focus` is load-bearing after nav-off.** Ctrl+Tab also runs regardless of the config
   flag, so the flag on `ui.py:65` and `copilot_chat.py:51` is what keeps `CYCLE_CODE_TAB` off
   imgui's window-cycle, and `commands.py:128`'s comment stays true with no edit. The spec never
   mentioned it, which was correct by accident. Added as item 2's fifth bullet and a premises row,
   so nav-off is not read as licence to delete every flag with `nav` in its name.
3. **W-E and W-F both wrote `Editor.set_style` / `get_style` and a style enum**, under two names.
   Already folded before this review arrived (see the cross-file correction below): W-F owns
   `Style` and both methods, W-E only calls them, `editor/ffi.py` is a no-change row, and the ffi
   round-trip test is returned to W-F. Confirmed the spec text now matches.
4. **Four missed deletion sites**, all verified in the tree: (a) `scripts/smoke.py:214-218`'s
   `nav_enable_keyboard` assertion, which would have turned `make gates` red before any of this
   wave's tests ran, **inverted**, not deleted, so it pins the decision; (b) `ui.py:747-748`'s
   `if focus_panel: set_next_window_focus()`, which sits between the two ranges the draft cited and
   would have been a `NameError`; (c) `ui_models.py:232-233`'s comment, which names `active_region`
   AND points at a `todo.md` feature-019 deferral that does not exist; (d)
   `exporters/youtube.py:310-312`'s "lands the nav outline" comment, which no banned token catches.
5. **The `no_nav_inputs` count is five, not four.** The fifth is `document_grid.py:45`. The draft
   stated four as a verified fact and used it to correct the finding's three; a census number in a
   spec is read as established, so it is corrected in the premises row rather than argued.
6. **Two named falsifiers do not falsify.** (a) The claim that scanning whole lines instead of the
   first table cell would pull Ctrl+X/C/V out of the standard doc's closing paragraph: re-ran it,
   both scans return the identical 13-chord set, because the paragraph writes them unbackticked and
   the regex requires backticks. The restriction is kept as defence against a re-vendor that
   backticks them, and the spec now says it has no falsifier against the doc as vendored rather than
   claiming a red it cannot produce. (b) "`Ctrl+D` is the only chord solely in `<C-x>` notation":
   measured, it is one of ten, with zero overlap between the two notations. The sentinel works; the
   uniqueness claim was wrong and is corrected.
7. **The banned-token grep hits files the deletion list did not name** (`ui_models.py:232`,
   `hotkeys.py:62`'s `CYCLE_REGION` inside the Esc comment). Both added to item 1. The reviewer also
   asked for the `region_`-as-prefix hazard to be written into the test's comment so a later wave
   does not "tighten" it into matching `get_content_region_avail`; done.
8. **A fifth keymap-dependent branch**, `hotkeys.py:77-83`'s insert-mode Ctrl+N completion ask. The
   draft enumerated four and claimed the enumeration was complete. It needs no gate, but "probably
   harmless" is not this spec's standard, so it is settled by probe against the vendored `.so`:
   under standard, `ed_key(CHAR, CTRL, "n")` returns consumed with `ed_complete_open()` already
   True, so the branch's own `not complete_open()` conjunct self-gates. Added as point 5 with the
   measurement, plus the reviewer's related note that `Ctrl+Shift+N` arrives as text `"N"`
   (uppercased by `_key_char` under shift) and so falls through the lowercase reserved sets, which
   the `NEW_DOCUMENT` move depends on.
9. **F-keys never reach `translate_key`.** `_SPECIAL_KEYS` does not list them and `_CHORD_MODS`
   excludes shift-only, so an F-key returns `None` and never enters `editor_key_events`. That is a
   stronger guarantee for the F-key tier than the doc parse gives. Added to `02_keybindings.md`
   note 3.
10. **The Settings combo broke the Editor section's layout idiom.** The draft put a `label_row`
    above three bare checkboxes, making the section read column / no-column / column against the
    imgui skill § 2. Moved to the head of the `label_row` block, above Font size; the "it frames
    what follows" rationale is dropped, since the layout cannot carry it and the alternative
    (converting the whole section) is W-B's.
11. **Two stale doc pointers.** The skill's § 9 two-editor-focus-flags bullet names "arrow nav" as a
    defocus cause, which this wave deletes: struck. And the `todo.md` row's check was right in one
    direction only, missing that a source comment points at a deferral that does not exist.
12. **One clause in the `_focus_or_add_tab` guard-drop argument is weaker than it reads.** The
    citation to `reconcile_popup_focus` setting the latch on the popup's CLOSE edge argues about
    the opposite state; the conclusion holds on the `ui.py:388` evidence alone. Clause dropped.

**Round 2, closure pass** (`reviews/wave_e_pre.md § Round 2`): ten of twelve closed, two reopened.
Both accepted; both were defects in what the round-1 fold WROTE, not in the ruling it recorded.

**F1's positive test had three measured defects, and I re-ran each rather than relaying it.**
(a) The non-vacuity property said `_FOCUSABLE_CALLS` is validated against `imgui.__all__`;
imgui-bundle exports no `__all__` (`hasattr(imgui, "__all__")` is `False`), so the property would
have raised `AttributeError` instead of validating anything. Now `hasattr(imgui, name)`, and all
twelve names resolve. (b) The container walk was described over "those two modules", which cannot
reach `document_grid.py` (the module the falsifier targets), and its floor of five sat against a
real count the reviewer measured at nine. I ran the walk: it finds **eight**, not nine, because
`copilot_chat.py:258`'s `##copilot_history` is a bare `if imgui.begin_child(...)` rather than a
`with`, so a `with`-based walk never sees it. The floor is eight and that container is named as
outside the set. (c) The rule "the container must pass `no_nav_inputs`" fails on correct code:
`ui.py:424` `app_panel` and `ui.py:710` `control_panel` host focusable widgets only through inner
children that carry the flag, so the rule is now "that container **or an enclosing one**", with
ancestry derived from `with`-nesting inside the parsed module.

Two things the round-2 report did not name, both found while building the prototype and both folded:
**a flag reaches three of the eight containers through a variable** (`panel_flags`, `grid_flags`,
and the chat's `flags = _WINDOW_FLAGS | ...`), so the flag test must resolve a `Name` through module
assignments or it fails on correct code, verified in both directions. And **the `document_grid.py:45`
falsifier only fires once the ancestry walk parses each module ONCE**: my first prototype re-parsed
inside the check, so its node-identity ancestry test never matched, every container was skipped, and
the falsifier came back green. That is the same class of defect as (c): a checker silently
narrowing its domain to nothing, caught here only because the falsifier was actually run. It now
goes red with `widgets/document_grid.py:45 "document_preview_grid"`.

**The test's honest limit is stated rather than papered over:** deleting the flag from
`copilot_chat.py`'s `_WINDOW_FLAGS` leaves it GREEN, measured, because the chat window's block names
no focusable call directly (its input is drawn in a helper). The chat's flag is protected by manual
step 2, not by this test, and § Design decisions item 1 says so.

**F6b:** § 7's sentinel sentence still called `Ctrl+D` "the only chord that appears solely in the
`<C-x>` notation", corrected only in the round-1 appendix. It is one of ten (measured: `<C-x>`-only
is B, D, E, F, U, Y, Left, Right, Home, End; `CTRL-`-only is H, J, M, N, P, R; zero overlap). The
sentence now says so, and the sentinel's value is unaffected.

**Cross-file correction from the coordinator, before the review** (a
wave-boundary fix, not a finding against the design).

**W-F owns the editor-style surface in `editor/ffi.py`; W-E only calls it.** The first draft of
this spec defined `class EditorStyle(IntEnum)`, `Editor.set_style` and `Editor.get_style` in its
§ Design decisions item 4 and claimed `editor/ffi.py` in § Files touched. W-F
(`40_wave_f_editor_chrome.md` § Out of scope, § Design decisions item 3) already defines all three,
under the name **`Style`**, and lands first. Two waves defining one symbol under two names is the
kind of collision that survives paper review and fails at merge, so the duplicate is removed:
item 4 now imports `Style` and adds the single `set_style` call, § Files touched marks
`editor/ffi.py` as no-change with the ownership stated, the ffi round-trip test is returned to W-F,
and open question 4 uses W-F's name. **The ordering constraint W-F recorded for W-E** (that
`ed_set_style` replaces the whole `Chrome` from `chrome_for(style)`, so `set_style` must precede
`set_chrome_flag(ChromeFlag.LINE_NUMBERS, ...)` or the user's line-numbers setting is discarded on
every apply) was already folded into item 4's apply block before this correction and is unchanged;
what is new is that the block now names W-F as the owner of the method whose ordering it honours.

The dependency is therefore a hard edge, not a soft one: W-E cannot begin until `Style` exists in
`shaderbox/editor/ffi.py`. The parent spec's § Order already places W-F before W-E for the `.so`;
this adds a second reason, and it is a compile-time one.

**Round 3, post-implementation** (three reviewers: `reviews/wave_e_post_code.md`,
`wave_e_post_arch.md`, `wave_e_post_spec.md`). The gate was RED at the reviewed commit — ruff
rejected both new test modules, which were untracked when `make gates` ran and pre-commit sees
tracked files only. Fixed in `7bd7a1d`; the durable lesson is `git add` a new file BEFORE the
gate, which is now `dev_flow.md`'s. Six findings changed code:

1. **Item 5's literal contradicted its own prose, and the prose won.** The spec prescribed
   `frozenset("dufbeyrnphjm")` while its third bullet said "`w` stays" for the host's insert-mode
   word-delete. Dropping `w` would have made `_delete_word_back` dead code and silently retired a
   067 D15 behaviour, so the landed set is `frozenset("dufbeyrwnphjm")` and the literal above is
   corrected. The subset test carries `w` as `_HOST_OWNED`, one named letter with its reason —
   falsified: emptying `_HOST_OWNED` turns the test red, so the exemption cannot absorb a second
   letter unnoticed. Ctrl+W is in NEITHER keymap's list, so the ownership rule is untouched: the
   app owns the chord (`CLOSE_CODE_TAB`), the host owns a behaviour on it in insert mode only.

2. **The positive Tab test was vacuous on `ui.py`.** It matched only widget calls written inline
   in a `with` body, and none of `ui.py`'s six containers writes one — every widget is drawn in a
   called free function, so deleting all three `ui.py` flags left the suite green. The check now
   follows calls transitively into any module under `shaderbox/`, through `ui_primitives` wrappers
   (derived by AST, not listed) and through callees parked in a module-level table (`_NODE_TABS`
   is the only route from `document_settings` to the uniform sliders), stopping at a callee that
   opens its own top-level window or popup.

3. **The domain was wrong, and the measurement corrected the spec's own claim.** Item 1 argued the
   grid's flag was earned because "`preview_cell`'s `selectable` is a real Tab stop". Measured on a
   headless rig with `nav_enable_keyboard` OFF: Tab lands ONLY on text-entry widgets. `button`,
   `checkbox`, `combo` and `selectable` are nav-on stops and are NOT Tab stops here. So the
   flags are earned by the panel's sliders and the chat's input, not by the tiles. Of the five
   sites, deleting `document_settings`'s or the chat's turns the test red; `code_editor`,
   `copilot_bar` and `document_preview_grid` stay green because none contains a Tab stop today
   (a rendered image, buttons only, a `selectable`). Those three flags are defensive and the test
   starts guarding them the day an input lands there; the test's docstring says so rather than
   implying a coverage it does not have.

4. **`Ctrl+N`'s self-gate comment stated a mechanism that does not run.** It claimed standard's
   Ctrl+N opens the popup on the same frame. It does not: `get_session` sets
   `set_host_completion(True)` on every handle, so BOTH keymaps consume and open nothing, and the
   real gate is the `not editor.complete_open()` conjunct vim already used.

5. **Stale prose the removal left behind**, each rewritten to the now: `app.py`'s "Esc, arrow nav"
   defocus list (the survivors are an explicit defocus and a tab/document switch), the drain's "once
   Esc decides to defocus", the two comments naming an outline read that no longer happens
   (`ui.py`, `tabs/code.py`), `commands.py`'s `no_nav_focus` rationale for Ctrl+Tab (nav-off is
   what frees it now; the flag is the belt-and-braces), the imgui skill's §8 sibling sentence and
   §9 defocus list, `scripts/smoke.py`'s region-cycle comment, and `dev_flow.md`'s module map
   (`ui_regions.py`, `app.py`'s "nav", `editor/`'s "vim-modal").

6. **A user-facing stale chord.** The Help panel still told the user to press `Ctrl+P` for the
   library one commit after the chord became Alt+L — the generated shortcuts table follows
   `COMMAND_SPECS` for free, but a chord typed into a section BODY does not.
   `tests/test_help_content.py` now parses every backticked chord out of every section body and
   requires it to be one the table currently binds; falsified by reverting the string.

Two findings were declined. Renaming `ui_regions.py` to `ui_tabs.py` (arch 8) is a preference the
reviewer itself marked optional, and the module's docstring already carries the correct meaning.
The `_RESERVED_CHORDS` string keys restating the keymap vocabulary (arch 3, third home) stay:
the `Literal` plus `model_salvage` make an out-of-domain key structurally unreachable, which the
reviewer confirmed. The vocabulary's other two homes were merged — `ui_models.EditorKeymap` is
now the alias and `settings.py` derives its combo options with `get_args`.
