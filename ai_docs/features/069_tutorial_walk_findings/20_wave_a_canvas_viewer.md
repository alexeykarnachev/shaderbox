# 069 W-A: Canvas size and the viewer

Implementation spec for wave W-A of feature 069. The parent spec (`01_spec.md § W-A`) fixes the
shape; this file fixes the code. Locked decisions D1, D2 and D11 apply and are not re-opened, nor
are the constraints the parent states as such: the funnel through `Document.set_canvas_size`, the
`W x H` pair of `input_int`s committing on deactivate-after-edit, the presets menu (squares
256 / 512 / 1024 / 2048 plus the named video shapes plus any bound texture's size), the shared
clamp constant, the output pass's disabled size slider with its existing help text, and the
checkerboard plus 1px border built from two `theme.py` greys.

W-C lands before this wave and touches `popups/pass_settings.py` and `ui.py`. Every citation below
names a symbol, so a shifted line does not invalidate it.

## Goal

The canvas size stops being a number only the copilot and a text editor can reach, and stops
disagreeing with itself. A person types `512` and `512` into a pair of fields in the Document tab,
or picks `512 x 512` from a presets menu beside them, and both paths call
`Document.set_canvas_size`. So the field the whole pass graph scales from is the one the UI
writes, every non-output pass follows on the next render, and the gear's size row shows the number
the user just set instead of a stale one. The output pass's percent slider, which the renderer has
always ignored, is disabled rather than pretending. And the viewer says where the canvas is: a
checkerboard behind the preview so a fully transparent output is visible as a shape rather than
vanishing into the panel, and a 1px border so an opaque dark output still reads as having an
extent.

## Findings folded

Five, quoted verbatim from `00_findings.md`:

- **#1** (DEFECT + UX, Before you start): "set the canvas to 512x512 in the Document tab — how?
  there is no 512x512 option, where did you find it?"
- **#2** (ENGINE, found by the agent while checking #1, not reported by the maintainer): the
  ledger's own words — "The Document-tab combo resizes via `render_pass.canvas.set_size()`
  directly ... bypassing `Document.set_canvas_size` ... which is the funnel that updates
  `document.canvas_size`."
- **#3** (ENGINE, = #2 seen live, Before you start): "I tried to change the canvas to 1080x1080,
  then I clicked the settings of the first pass and it still has 1280x960 (100%) -- wtf?"
- **#4** (UX, Pass settings gear): "is it even possible to have the first pass (or the last pass
  actually) at a resolution different from the canvas?"
- **#21** (UX, Viewer — transparent canvas): "when the canvas is transparent (a texture with alpha
  = 0) we can't see the canvas boundary — it blends with the background."

Two clauses the maintainer added to the ledger and this wave is bound by: "Manual JSON editing is
NOT an acceptable workaround (maintainer)" (#3, and D2 generalizes it), and #4's "A control that
does nothing is a UX defect; either disable it for the output pass or hide the row" — the parent
spec picks disable.

**One change in this wave closes no finding: the copilot-turn gate on the first row** (§ Design
decisions item 4a). It is a consistency gap found while reading the panel, not #1/#2/#3/#4/#21. The
row holds the only two document mutations in `tabs/document.py` that lack the
`begin_disabled(app.copilot_turn_active)` every sibling carries, and this wave rewrites that row
anyway. Named here so a later reader does not go looking for the finding it closes.

Two further changes go beyond the parent's W-A bullets but ARE the same defect class as #2, and are
argued where they land: `UIDocument.save` and `_copilot_document_working_view` both read the output
pass's canvas texture where `document.canvas_size` is the authority (§ Design decisions items 1
and 8).

## Out of scope

- **The prose cut on the gear's own strings and the popup's auto-size** (#5, #7, #10, the size
  row's `f"size ({w}, {h})"` label, the `_FORMATS` tooltips, the two size help-marker variants):
  **W-B**. W-A edits `_draw_target` to wrap the output pass's slider in a disabled scope and
  leaves every literal in that function exactly as it stands, including the `f"size ({w}, {h})"`
  label W-B's gate is written to refuse. The two waves therefore do not fight over the same lines,
  and W-B's gate lands after W-A rather than being pre-satisfied by it.
- **The JFA run count and the tutorial's "Before you start" text** (#6, #8): **W-H**. W-A's only
  contract with W-H is that `512 x 512` is reachable from the presets menu, which § Design
  decisions item 5 pins.
- **Absolute per-pass sizes** (a pass larger than the canvas): out of scope for the whole feature
  per the parent's § Out of scope; `TargetConfig.scale`'s `le=1.0` bound stands, and W-A adds no
  way around it.
- **The strip's sublines and default wiring by name** (#19, #37): **W-D**.
- **The rename crash, commit-on-deactivate for the gear's name field, add-pass activation, and the
  first-render sweep** (#9, #17, #18, #25, #28, #36): **W-C**, which lands first. W-A inherits
  `_draw_name`'s new `bool` return and `_draw_body`'s early return without touching either.
- **Pass-qualified scripting, the mouse, and the clear-canvas command** (#22, #23, #29, #30):
  **W-G**.
- **A resolution control on the Render or Share tab.** Those speak `RenderShape`
  (`conventions.md`: "Render output size is ONE named vocabulary"), which is an export concern.
  W-A borrows that table for preset LABELS and DIMS only and writes nothing back to it.

## Design decisions

### 1. The Document tab's resize routes through `Document.set_canvas_size`

`tabs/document.py::draw` today ends its resolution block with

```python
        ui_document.document.render_pass.canvas.set_size((w, h))
```

which is finding #2's root: it resizes the OUTPUT pass's canvas and leaves `document.canvas_size`
holding the previous value, so `Document.render`'s per-pass fixup
(`wanted = entry.target.target_size(self.canvas_size)`) keeps sizing every other pass off the old
dimensions, and `_draw_target`'s `canvas_w, canvas_h = document.canvas_size` shows the old number
in the gear (finding #3).

Every write in this wave goes through one module-level free function in `tabs/document.py`:

```python
def _apply_canvas_size(app: App, ui_document: UIDocument, size: tuple[int, int]) -> None:
    w, h = clamp_canvas_size(size)
    if (w, h) == ui_document.document.canvas_size:
        return
    ui_document.document.set_canvas_size((w, h))
    app.notifications.push(f"Canvas: {w}x{h}")
```

Both the `input_int` pair (item 3) and the presets menu (item 5) call it and nothing else.

**The early return on an unchanged size suppresses a spurious NOTIFICATION, not a reallocation.**
`is_item_deactivated_after_edit()` fires on any deactivate that followed an edit, including one
where the user typed a digit and deleted it again, so without the guard a `Canvas: 512x512` toast
appears every time focus passes through a field the user thought better of. No texture is at risk
either way: `Canvas.set_size` (`core.py::Canvas.set_size`) already opens with
`if size == self.texture.size: return False`, and `Document.set_canvas_size` does nothing else
that allocates, so an unchanged size assigns the field and reallocates nothing. (An earlier draft
justified the return by a reallocation `Canvas.set_size` in fact prevents, and rested a test on
that premise which would have passed with or without the return.)

The notification text is `Canvas: 1080x1080`, which is the `/imgui-ui § 2` notification budget's own
example, one clause, and shorter than today's `f"Canvas resolution changed: {resolution_items[...]}"`.
Cutting it here is not a W-B encroachment: the string is deleted along with the combo that built
it, so there is no surviving site for W-B's gate to measure.

**`UIDocument.save` reads the same field.** `ui_models.py::UIDocument.save` writes
`"canvas_size": list(self.document.render_pass.canvas.texture.size)`, the OUTPUT canvas's texture,
not `document.canvas_size`. That is the same class of bug as #2 (treating the output pass's canvas
as the document's size) and it is why #2 "hides across restarts": the save path reads the one
canvas the broken UI path did resize. It becomes

```python
            "canvas_size": list(self.document.canvas_size),
```

Correct on its own terms once the funnel holds (the two agree after this wave), and correct
independently of it, because `canvas_size` is the field `load_from_dir` reads back
(`document.py::load_from_dir` passes `metadata.get("canvas_size")` into both `Document` and each
`Pass`). Reading the derived value to persist the authoritative one is the shape the funnel exists
to remove, so it goes in this wave rather than being left as a second site for a later reviewer to
find.

### 2. The clamp is one constant, in `pass_graph.py`

`copilot/backend.py` today owns

```python
# Canvas-size clamp for set_canvas_size (feature 052): a sane render-resolution range.
_MIN_CANVAS_PX = 16
_MAX_CANVAS_PX = 4096
```

read only by `CopilotBackend.set_canvas_size`. With the UI gaining a second free-form entry path,
that becomes a cross-cutting guarantee on two callers, and `conventions.md`'s funnel law says the
bracket goes at the shared place rather than being copied. The constants and the clamp move to
`shaderbox/pass_graph.py`, beside `TargetConfig` whose `scale` bound is the sibling constraint on
the same quantity:

```python
# A canvas dimension the render path can actually allocate. Both entry points -- the Document
# tab's W x H fields and the copilot's set_canvas_size -- clamp through here.
MIN_CANVAS_PX: int = 16
MAX_CANVAS_PX: int = 4096


def clamp_canvas_size(size: tuple[int, int]) -> tuple[int, int]:
    w, h = size
    return (
        max(MIN_CANVAS_PX, min(MAX_CANVAS_PX, w)),
        max(MIN_CANVAS_PX, min(MAX_CANVAS_PX, h)),
    )
```

`pass_graph.py` is the right home and not merely a convenient one: it is the GL-free leaf that
already owns the graph's numeric bounds, it imports no imgui and no `Document`, and both callers
already import from it or from a module that does. Putting it on `Document` would make the copilot
backend's clamp depend on a live document; putting it in `theme.py` would make a render limit a
visual token; leaving it in `copilot/backend.py` would make `tabs/document.py` import the copilot.

`backend.py::set_canvas_size` keeps its behaviour and loses its arithmetic:

```python
            w, h = clamp_canvas_size((width, height))
```

`_MIN_CANVAS_PX` / `_MAX_CANVAS_PX` are deleted from `backend.py`;
`tests/test_document_ops.py::test_set_canvas_size_applies_and_clamps` asserts the clamped values
`(4096, 16)` through the public method and needs no edit, which is what makes the move safe to
verify.

### 3. A `W x H` pair of `input_int`s, committing as ONE `set_canvas_size` call

The combo goes. It has no selection state at all: `imgui.combo("##resolution", 0, resolution_items)`
is fed a literal `0` every frame and only its `!= 0` branch does anything, so it is a menu wearing a
combo's clothes, and the current size is item 0 purely so that picking it is a no-op. In its place,
in the same row position (`imgui.same_line(combo_offset)` after the document-name input):

```python
    if not app.canvas_size_editing:
        app.canvas_size_buf = ui_document.document.canvas_size

    imgui.set_next_item_width(_CANVAS_FIELD_W)
    entered_w, buf_w = imgui.input_int(
        "##canvas_w",
        app.canvas_size_buf[0],
        step=0,
        flags=imgui.InputTextFlags_.enter_returns_true,
    )
    active_w = imgui.is_item_active()
    committed_w = entered_w or imgui.is_item_deactivated_after_edit()
    app.canvas_size_buf = (buf_w, app.canvas_size_buf[1])

    imgui.same_line(spacing=float(SPACE.SM))
    imgui.text_colored(COLOR.FG_DIM, "x")
    imgui.same_line(spacing=float(SPACE.SM))

    imgui.set_next_item_width(_CANVAS_FIELD_W)
    entered_h, buf_h = imgui.input_int(
        "##canvas_h",
        app.canvas_size_buf[1],
        step=0,
        flags=imgui.InputTextFlags_.enter_returns_true,
    )
    active_h = imgui.is_item_active()
    committed_h = entered_h or imgui.is_item_deactivated_after_edit()
    app.canvas_size_buf = (app.canvas_size_buf[0], buf_h)

    app.canvas_size_editing = active_w or active_h

    if committed_w or committed_h:
        _apply_canvas_size(app, ui_document, app.canvas_size_buf)
        app.canvas_size_buf = ui_document.document.canvas_size
```

Six properties, each load-bearing:

- **The pending edit lives on `App`, as two fields: `canvas_size_buf: tuple[int, int] = (0, 0)`
  and `canvas_size_editing: bool = False`.** They have to be on `App` and not module globals:
  `tabs/*.py` are free `draw(app)` functions with no state of their own (`conventions.md`: "Tab
  state goes on `App` directly"), and the buffer must survive the frame in which the user types.
  The initial `(0, 0)` is never displayed, because the first frame is by definition a non-editing
  one and overwrites it before the first `input_int` reads it.

- **The buffer MIRRORS the document on every frame in which neither field is active, and holds
  only while one is.** `canvas_size_editing` is the OR of `imgui.is_item_active()` read on the
  line after each `input_int`, and it is written at the END of the row, so the mirror check at the
  TOP of the next frame reads the previous frame's verdict. The frame-by-frame rule, which is what
  makes manual verification item 11 true:

  | Frame | `canvas_size_editing` at the top | What the fields show |
  |---|---|---|
  | Idle, nothing focused | `False` | `document.canvas_size`, re-read this frame |
  | The user clicks into W | `False` (set True at the end of this frame) | `document.canvas_size` |
  | The user is typing in W | `True` | the half-typed buffer, document ignored |
  | Focus leaves W | `True` (set False at the end) | commit fires; buffer re-read from the document |
  | The frame after | `False` | `document.canvas_size`, re-read |
  | The copilot writes the size while nothing is focused | `False` | the NEW `document.canvas_size`, on the next frame, unclicked |

  A `| None` sentinel meaning "not editing" was the drafted shape and is wrong: nothing ever set it
  back to `None`, so after the first frame the document's field was never read again and an
  externally-set size (the copilot's `set_canvas_size`, a `document.json` edit the per-frame sync
  picks up) could never reach the fields. The active-item rule has no such latch, because it is
  re-derived from imgui every frame rather than stored across them.

- **A document switch re-arms the mirror, in `App._on_current_document_changed`.** One line:

  ```python
          self.canvas_size_editing = False
  ```

  beside the `editor_was_ever_focused = False` already there. It does not clear the buffer; it
  clears the flag that would stop the mirror from overwriting it, so the next frame reads the NEW
  document's size and assigns it before either `input_int` is submitted.

  That handler is the reset point, not `App.set_current_document_id`, which an earlier draft named:
  that method is a bare two-line forwarder to `self.session.set_current_document_id(id)` with no
  transient resets in it. `_on_current_document_changed` is where `editor_was_ever_focused` is
  cleared, and `ProjectSession` fires it only when the id ACTUALLY changes, so a no-op re-set of
  the same id does not disturb a field the user is typing in.

  The reset is load-bearing, not belt-and-braces. An earlier draft deleted it, arguing that a
  document switch cannot happen while a field is active because the click that switches is the
  click that deactivates. That is true of the MOUSE and false of the keyboard: `NEW_DOCUMENT` is
  Ctrl+N with `CommandSpec`'s default `CommandScope.GLOBAL`, and `route_flag` returns
  `route_global` for it, with its own comment stating that "an active text input owns all keyboard
  keys and imgui routes only Ctrl-chords through it". `hotkeys.py::spec_eligible` gates on the
  consumed-chord set, the editor and copilot focus flags and an open modal, with no "an input is
  active" term. So Ctrl+N (and the non-modal command palette, same shape) fires with a canvas field
  focused, `create_document_from_example` switches the document, and without this line
  `canvas_size_editing` stays `True` from the previous frame: the fields would show one document's
  half-typed width against another, and a later tab-out would commit that pair to the NEW document.
- **Both fields commit as ONE `set_canvas_size` call.** `_apply_canvas_size` takes the whole pair
  from the buffer, so whichever field the user leaves, the size that lands is
  `(buffer_w, buffer_h)`. **What happens when the user edits W, then clicks into H:** leaving W
  fires `is_item_deactivated_after_edit()` on the W field, which commits the pair
  `(new_w, current_h)` immediately. The canvas resizes on that frame, the fields re-sync to the
  document, and the caret is now in H holding the unchanged height. Editing H and leaving it then
  commits `(new_w, new_h)`. Two resizes, both correct, the second replacing the first; the
  alternative, deferring until BOTH fields are deactivated, would need a "which field was
  touched" ledger and a rule for what happens when the user edits W and never touches H, which is
  the guard-pile shape. One commit per deactivate, each carrying the whole pair, has no
  intermediate state to get wrong.
- **The buffer is re-read from the document after every commit**, so the clamp is visible: typing
  `99999` and tabbing out leaves `4096` in the field, not the rejected number. This is the same
  posture W-C's `_commit_pass_name` takes on a rejected rename (snap the buffer back to the live
  value), and it is why the clamp needs no error notification of its own: the field shows what
  happened.
- **`step=0` suppresses the `-`/`+` spinner buttons.** `input_int`'s default `step=1` draws two
  buttons per field, which would put four buttons in a row that has room for two fields and a
  menu, and each button click is a per-keystroke-equivalent commit the D11 rule exists to avoid.
- **`is_item_deactivated_after_edit()` is read on the line immediately after each `input_int`**,
  before the `same_line` that follows, the item-scoped queries read the LAST submitted item, so
  any intervening widget makes them read the wrong one. Same rule W-C's item 2 states for
  `_draw_name`.

- **Enter is `enter_returns_true` OR-ed with the deactivate query, verbatim as the skill states
  it.** `/imgui-ui § 7.5` at `a246a19`, which W-C rewrote and landed, is now the rule for every
  inline input in this codebase: an input whose commit performs a TRANSACTION "commits on
  `imgui.is_item_deactivated_after_edit()`, with `enter_returns_true` as the Enter shortcut and Esc
  as cancel", with the deactivate query "read on the line IMMEDIATELY after the `input_text` call,
  before any `same_line`". W-C's `_draw_name` implements exactly that
  (`committed or deactivated`), and this row matches it token for token.

  A narrower shape would work here: `input_int` without the flag still deactivates on Enter, so
  the OR's first term is strictly redundant for this widget. It is written anyway, because one
  rule for every inline input is worth more than one saved flag. A reader who implements this row
  from the skill reaches for both terms; a row that silently used only one would read as a
  deviation with no stated reason, and the next person to touch it would have to re-derive that
  `input_int` and `input_text` differ on Enter. The cost is one keyword argument.

**`input_int2` is rejected**, though it exists (`imgui.input_int2(label, v: List[int])` in the
installed stub) and takes exactly the pair this row edits. It returns ONE `changed` flag and
submits ONE item, so `is_item_deactivated_after_edit()` fires only when focus leaves the whole
two-field widget, which makes "edit W, click into H" indistinguishable from "edit W, click
away", and the parent spec's D11 constraint is written per field. It also gives no place to put
the `x` separator. Two `input_int`s is more code and the only shape that can answer the question
D11 asks.

**Word budget (D1).** The row's caption is `Canvas` (1 word, replacing today's
`Resolution` plus its parenthesised uniform-name suffix, a derived value in a label, which D1
forbids and which the `x`-separated pair now carries in the controls). The two fields have
`##`-only ids, so no visible label. No `help_marker`: `Canvas` beside two number fields separated
by `x` is unambiguous, and D1 says a clear label gets no marker at all.

**Layout, and the arithmetic that fixes both numbers.** The cluster must fit the
`SIZE.RES_COMBO_W` width the row already reserves (`theme.py`, `RES_COMBO_W: int = 200`), which is
what the old combo occupied. Budget:

```
_CANVAS_FIELD_W  56
SPACE.SM          4
"x" glyph        ~7   (one char at the default frame font)
SPACE.SM          4
_CANVAS_FIELD_W  56
SPACE.MD          8
_CANVAS_PRESETS_W 64
                ---
                199  <= SIZE.RES_COMBO_W (200)
```

The first draft put `_CANVAS_FIELD_W` at 62, which leaves 53px for the dropdown. A `presets`
preview string is roughly 42px of text plus 8-12px of frame padding, so 53 does not reliably
close. **The field width is what gives:** 56px still holds a four-digit number comfortably (about
26px of digits inside 12px of padding, so 44px of the 56 is used, and 4096 is the largest value
the clamp admits), while 64px is a comfortable dropdown. Both are module constants in
`tabs/document.py` beside the existing row constants rather than `SIZE` tokens, because both are
derived arithmetic against `RES_COMBO_W`, not independent design choices.

`SIZE.RES_COMBO_W` is kept and is what the whole cluster is measured against;
`SIZE.NAME_INPUT_W` and the `combo_offset` computation are unchanged. (If a reviewer prefers
tokens, the counter-argument is `/imgui-ui § 6`'s "a token used by exactly one panel still belongs
in the token bag". This is the one call where the two rules point opposite ways, and the deciding
fact is that both numbers are consequences of `RES_COMBO_W` rather than choices. Promote them if a
second panel ever needs a canvas-dimension field.)

### 4. The presets menu is a `begin_combo` dropdown

Beside the pair, the same dropdown shape the tab already uses one section below for the uniform
sort key:

```python
    imgui.same_line(spacing=float(SPACE.MD))
    imgui.set_next_item_width(_CANVAS_PRESETS_W)
    if imgui.begin_combo(
        "##canvas_presets", "presets", imgui.ComboFlags_.no_arrow_button
    ):
        for label, size in _canvas_presets(ui_document):
            if imgui.selectable(label, False)[0]:
                _apply_canvas_size(app, ui_document, size)
                app.canvas_size_buf = ui_document.document.canvas_size
        imgui.end_combo()
```

`begin_combo` + a `selectable` loop is `tabs/document.py`'s own existing idiom, the uniform sort
control a hundred lines below is exactly this shape, down to the fixed preview string
(`f"Sort by: {...}"`) rather than the selected item's label. Reusing it costs no new pattern and
keeps two dropdowns in the same panel looking alike, which an `open_popup` + `begin_popup` pair
would not: this codebase has no button-opened popup today (every `begin_popup*` site is either
`begin_popup_context_item` or a menu-bar `begin_menu`), so that route would introduce a shape with
one instance.

The preview string is the fixed word `presets`, never the picked entry, the fields beside it are
the readout, and echoing the last-picked preset there would be a second, staler display of the
same number. `selectable(label, False)` is likewise always-unselected: the menu is a set of
actions, not a persisted choice, which is the same reason the old combo was fed a literal `0`.
The difference from the old combo is that this one is honest about it, a `begin_combo` whose
entries are actions reads as a menu, while `imgui.combo` with a hardcoded index reads as a
selection that is silently broken.

`ComboFlags_.no_arrow_button` drops the arrow so the control reads as a button-sized affordance
rather than a wide field. `_CANVAS_PRESETS_W: float = 64.0`, the last term in item 3's row budget
(56 + 4 + 7 + 4 + 56 + 8 + 64 = 199, inside `SIZE.RES_COMBO_W`'s 200).

The buffer is re-synced after a preset the same way it is after a field commit, so a preset picked
while a half-typed number sits in a field replaces it rather than being overwritten by the stale
buffer on the next frame.

### 4a. The whole first row is disabled during a copilot turn

ONE `begin_disabled` pair wraps the document-name input, the `W x H` pair, the `x`, and the presets
dropdown, opening before the row's first `small_caption` and closing after the dropdown:

```python
    imgui.begin_disabled(app.copilot_turn_active)
    ...the captions, the document-name input, the two input_ints, the "x", the presets combo...
    imgui.end_disabled()
```

Every other document MUTATION in this panel already carries this gate: the pass strip
(`widgets/pass_list.py::draw`), the script row and document-play toggle
(`tabs/document.py::_draw_entry_points`), the uniform sliders, and the document grid's tiles. This
row holds the only two controls that do not, and BOTH race a copilot write:

- **The canvas fields** reallocate GL textures. A resize landing mid-turn resizes under a copilot
  tool that has already read the size, and `CopilotBackend.set_canvas_size` runs its own write on
  the main thread through the bridge, so the two can interleave within a frame.
- **The document-name input** writes `ui_document.ui_state.ui_name` every keystroke, and
  `CopilotBackend.rename_document` writes the same field. That is the same mid-turn write race,
  minus the GL. It is gated here rather than deferred: no later wave in `01_spec.md § Workstreams`
  names this input, so "whichever wave next edits it" is a deferral to nowhere, and it is one
  `begin_disabled` pair in a row this wave is rewriting anyway.

Adding the gate brings the row in line with its siblings rather than inventing a rule.

When the gate is on, the fields are dimmed and neither `is_item_active()` nor
`is_item_deactivated_after_edit()` fires on them, so `canvas_size_editing` reads `False` throughout
a turn and the mirror rule keeps the buffer tracking the document. A canvas the copilot sets
mid-turn is therefore already on screen while the gate is still up, not merely when it lifts.

### 5. What the presets menu lists, and where each entry comes from

One module-level free function in `tabs/document.py` builds the list, so the menu draw stays a
loop and the composition is testable without an imgui frame:

```python
def _canvas_presets(ui_document: UIDocument) -> list[tuple[str, tuple[int, int]]]:
```

Three groups, in this order:

1. **Squares**, from a module constant `_SQUARE_PRESETS: tuple[int, ...] = (256, 512, 1024, 2048)`.
   Label `get_resolution_str(None, n, n)` -> `512x512 (1:1)`. This is the group W-H depends on:
   the tutorial's "Before you start" step names `512 x 512`, and `512` being in this tuple is what
   makes that step performable. Every value is inside the clamp (`16 <= n <= 4096`), so no square
   preset can be silently altered on the way through `_apply_canvas_size`.

2. **The named video shapes**, from `render_shape.py`, `MENU_SHAPES` in its declared order,
   skipping `RenderShape.NATIVE` (whose spec has `aspect=None` and `longest_edge=None`; it MEANS
   "the canvas size", so as a canvas preset it is a no-op). For each remaining shape the label is
   its `SHAPE_TABLE[shape].menu_label` (`Short 1080p (9:16)`, `Wide 720p (16:9)`, ...) and the
   dims come from `resolve_dims(shape_to_preset(shape, is_video=False, fps=None, container=None,
   duration_max=None), ui_document.document.canvas_size)`.

   `render_shape.py` is the module that supplies them, and going through
   `shape_to_preset` + `resolve_dims` rather than recomputing `aspect` and `longest_edge` by hand
   is what keeps the vocabulary single-homed (`conventions.md`: "Render output size is ONE named
   vocabulary"). `resolve_dims` also applies `_align`, so a preset can never produce an odd
   dimension a video encoder would reject. `source_size` is only read on the `LONGEST_EDGE` and
   free policies, and `shape_to_preset` emits `FIXED_ASPECT` for every non-`NATIVE` shape, so the
   canvas size passed in is inert here, it is passed because the signature requires it, and the
   six shapes yield the same six sizes whatever the canvas currently is.

   The largest is `SHORT_1440` / `WIDE_1440` at longest edge 2560, inside the 4096 clamp.

3. **Any bound texture's size**, discovered across ALL passes:

```python
    for name, render_pass in ui_document.document.passes.items():
        for uniform_name, value in sorted(render_pass.uniform_values.items()):
            if not isinstance(value, MediaWithTexture) or is_default_image(value):
                continue
            size = value.texture.size
            ...
```

   Three facts make this the right shape:

   - **It reads `uniform_values`, never `get_active_uniforms()`.** The current combo builds its
     texture rows from `render_pass.get_active_uniforms()`, which COMPILES a never-attempted pass
     (`core.py::Pass.get_active_uniforms` calls `compile()` when `program is None` and no error is
     stuck). Doing that across every pass on every frame of the Document tab would compile the whole
     graph the moment the tab draws, which is precisely what 066 D1 ("a pass compiles when something
     first NEEDS its program, never at load") forbids. `uniform_values` is a plain dict populated
     by seeding and by binds; reading it costs nothing and forces nothing.
   - **The `isinstance(value, MediaWithTexture)` filter admits exactly the right values.**
     `uniform_values` can also hold a `moderngl.Buffer` (a uniform block) and plain numbers. A
     `Buffer` is not a `MediaWithTexture`, so it is excluded with no second check. A bound `Video`
     IS one and is admitted, which is right: a bound video's dimensions are as much a user choice
     as an image's, and its `texture.size` is the frame size.
   - **`is_default_image(value)` excludes the shipped placeholder.** `media.py::is_default_image`
     is the sole marker for "this sampler is unbound and holding the seeded default image", and
     `seed_uniform_values` fills every unbound sampler with it, so without this filter EVERY
     document with any sampler would list the default image's dimensions as a canvas preset, which
     is noise, not a choice the user made.
   - **Duplicates collapse, and the current size is not offered.** A `set` of already-emitted sizes
     drops a texture whose dims another entry already lists, and a texture matching
     `ui_document.document.canvas_size` is skipped (picking it would be a no-op that
     `_apply_canvas_size`'s early return would silently swallow, which reads as a dead menu item).
     The label is `get_resolution_str(uniform_name, w, h)` -> `1920x1080 (16:9) - u_image`, reusing
     the helper the old combo used and `widgets/uniform.py` still uses. When two passes bind
     different textures at the same size, the first in `document.passes` insertion order wins the
     label; `sorted()` on the uniform names within a pass makes the choice deterministic rather
     than dependent on seeding order.

**When no texture is bound, the group is simply absent**, the menu shows squares and video shapes
and nothing else. No "no bound textures" placeholder line: D1 caps an empty state at 4 words, and
this one is not an empty state at all, since the menu is never empty (the first two groups are
static). A separator between groups is drawn only when the group that follows is non-empty, so an
unbound document has no trailing rule.

The menu is therefore between 10 and (10 + one per distinct bound texture) items, short enough
that it needs no scroll region.

### 6. The output pass's size slider is disabled

`popups/pass_settings.py::_draw_target` already computes `is_output = name == document.graph.output`
and already branches on it for the size row's help text. The renderer has always ignored the output
pass's `scale` (`document.py::render`'s `if name != output:` guard skips the size fixup for it) and
the help text already says so, but the slider is live, so moving it writes a `TargetConfig.scale`
that changes nothing. Finding #4's "A control that does nothing is a UX defect".

The slider and only the slider is wrapped:

```python
    imgui.begin_disabled(is_output)
    scale_changed, percent = imgui.slider_float(
        f"##scale_{name}",
        target.scale * 100.0,
        5.0,
        100.0,
        "%.0f%%",
    )
    imgui.end_disabled()
    if scale_changed:
        new_target = new_target.model_copy(update={"scale": percent / 100.0})
```

`begin_disabled` / `end_disabled` is the codebase's existing idiom (20 call sites, e.g.
`tabs/render.py`, `popups/examples.py`, `widgets/copilot_chat.py`), a bare pair, not a context
manager, matching every one of them. The `label_row` above and the `same_line` + `help_marker`
below stay OUTSIDE the disabled scope, so the label and the "(?)" keep full contrast and the help
text, which is the explanation for why the slider is dead, stays readable and hoverable. A
`help_marker` inside `begin_disabled` still shows its tooltip, but at reduced alpha, and dimming
the one string that explains the dead control is the opposite of the intent.

`scale_changed` reads `False` for a disabled widget, so the `model_copy` cannot fire and the guard
after it needs no change. Keeping the `if scale_changed:` outside the disabled scope rather than
inside is a readability choice with no behavioural difference.

**The help text is unchanged**, both variants, verbatim, the parent spec says "disabled with the
help text it already has", and W-B owns cutting it.

**The row is disabled, not hidden.** #4 offers both; the parent picks disable. The reason to
prefer it: the row is where the derived resolution is displayed (`size (1080, 1080)`), and hiding
it would remove the output pass's size readout from the only place it appears, the user would lose
the answer to "how big is this pass" precisely on the pass they most want it for.

### 7. Checkerboard and border behind the viewer

In `ui.py::_draw_document_image`, in the with-document branch, between the `img_min` capture and
the `image_with_bg` call:

```python
        img_min = imgui.get_cursor_screen_pos()
        _draw_canvas_backdrop(img_min, image_width, image_height)
        imgui.image_with_bg(...)
```

and after it, the border. One module-level free function in `ui.py`:

```python
def _draw_canvas_backdrop(origin: imgui.ImVec2, width: float, height: float) -> None:
    dl = imgui.get_window_draw_list()
    tile = float(SIZE.CHECKER_TILE)
    dl.add_rect_filled(
        (origin.x, origin.y),
        (origin.x + width, origin.y + height),
        imgui.color_convert_float4_to_u32(COLOR.CHECKER_LIGHT),
    )
    dark = imgui.color_convert_float4_to_u32(COLOR.CHECKER_DARK)
    rows = int(height // tile) + 1
    cols = int(width // tile) + 1
    for row in range(rows):
        for col in range(cols):
            if (row + col) % 2 == 0:
                continue
            x0 = origin.x + col * tile
            y0 = origin.y + row * tile
            dl.add_rect_filled(
                (x0, y0),
                (min(x0 + tile, origin.x + width), min(y0 + tile, origin.y + height)),
                dark,
            )
```

**Draw-list primitives, not a texture.** `add_rect_filled` on the window draw list is the idiom
this codebase already uses for every drawn background (`ui_primitives.py::_code_chip`,
`preview_cell`'s stale wash, `widgets/cheatsheet.py`'s panel), and it needs no GL object, no
lifetime, and no release path. The alternative the finding mentions, "a tiny 2x2 texture at
repeat", would add a moderngl texture to `App` that has to be created after the context, released
on shutdown, and kept out of the export path; that is real machinery for a backdrop.

**Cost.** The preview is capped by the panel, so at the widest realistic layout (roughly 1200 x 900
logical pixels) a 12px tile is about 100 x 75 cells, of which half are drawn: some 3700
`add_rect_filled` calls per frame. That is well inside imgui's per-frame budget for filled rects
(they are two triangles each, batched into one draw call by the shared draw list, with no state
change between them). If it ever measures badly, the fix is a bigger tile, not a texture.

**The `min(...)` on each tile's far corner** clips the last row and column to the image rect, so
the checkerboard never bleeds past the preview into the panel. The finding's own framing is that
the checkerboard marks the canvas EXTENT, which a bleeding edge would destroy.

**The border**, after `image_with_bg` so it draws over the image's outermost pixel row:

```python
        imgui.get_window_draw_list().add_rect(
            (img_min.x, img_min.y),
            (img_min.x + image_width, img_min.y + image_height),
            imgui.color_convert_float4_to_u32(COLOR.BORDER),
            thickness=1.0,
        )
```

`COLOR.BORDER` is the existing token (`_P["bg_2"]`, `#3c3836`) and the finding names it: "a 1px
`COLOR.BORDER`-tier rect". No new token, and `add_rect`'s default `thickness=1.0` is stated
explicitly for the same reason `ui_primitives.py`'s outline states it.

**The two greys are new `theme.py` tokens**, in the neutrals block beside `BG_FRAME` and `BORDER`,
each mapping to a `_P` entry rather than a literal (the theme's own rule: `_P` is the only home for
literal colours):

```python
    # Alpha checkerboard behind the viewer: the canvas's extent must read even when the
    # output is fully transparent. Two adjacent palette greys -- visible as a pattern, quiet
    # enough that an opaque render is not framed by a texture.
    CHECKER_LIGHT: tuple[float, float, float, float] = _P["bg_2"]
    CHECKER_DARK: tuple[float, float, float, float] = _P["bg_1"]
```

`bg_1` (`#282828`) and `bg_2` (`#3c3836`) are one palette step apart, which is the contrast an
alpha checkerboard wants: enough to read as a pattern, not enough to compete with the render on
top of it. They are FIXED roles in `theme.py`'s vocabulary but need no entry in the import-time
`SELECT` invariant, because that assertion covers hues that share spatial context with an accent
OUTLINE, and these are a background fill under an image, with no outline semantics.

`SIZE.CHECKER_TILE: int = 12` joins the SIZE bag. 12 is roughly the size the pattern reads at a
glance without dominating: at 8 the pattern is busy behind a small preview, at 24 a narrow preview
shows two cells and reads as a diagonal split rather than a checkerboard.

**`image_with_bg` does not hide it.** Its `bg_col` parameter defaults to `ImVec4(0, 0, 0, 0)`,
fully transparent, and `ui.py` passes only `tint_col`, so the backdrop drawn immediately before it
shows through exactly where the output's own alpha is below 1. (Finding #21 describes this as "with
the default (window) background"; the observable behaviour it reports is right, the mechanism is
that there is NO background, which is why nothing had to be removed to make the checkerboard
visible. Recorded in § Verified / corrected premises.)

**The empty-state branch is untouched.** `_draw_document_image`'s `else` draws a centred prompt
where no document exists; there is no canvas there to mark the extent of.

**Export is unaffected**, as the finding says: this is draw-list work inside one imgui frame in
`ui.py`, and nothing in `document.py`'s render or export path can see it.

### 8. What happens to every place that reads `render_pass` for the size

Seven sites read the OUTPUT pass's canvas texture where a document size is in question, enumerated
from `grep -n 'render_pass.canvas.texture.size' shaderbox/`. Each is resolved deliberately, not
swept. The test is whether the site is asking "how big is this document" (read the field) or "how
big is the texture I am about to blit or fit to" (read the texture):

| Site | Verdict |
|---|---|
| `tabs/document.py::draw`, `cw, ch = ui_document.document.render_pass.canvas.texture.size` feeding `current_size` | **Deleted with the combo.** The pair reads `ui_document.document.canvas_size` directly. It is the field the user is editing and the field every other pass scales from; reading the derived value to display the authoritative one is #2's shape in miniature. |
| `tabs/document.py::draw`, the `render_pass.get_active_uniforms()` loop building `uniform_resolutions` / `matching_uniforms` / the `resolution_label` suffix | **Deleted with the combo**, replaced by `_canvas_presets`' `uniform_values` scan across ALL passes (item 5). Three gains: it covers every pass rather than only the output (a texture bound on `paint` was invisible to the old combo), it forces no compile, and the `resolution_label`'s `f"Resolution ({current_name})"` derived-value-in-a-label goes with it, which D1 requires anyway. |
| `ui_models.py::UIDocument.save`, `"canvas_size": list(self.document.render_pass.canvas.texture.size)` | **Changed to `document.canvas_size`** (item 1). |
| `ui.py::update_and_draw`, `adjust_size(ui_document.document.render_pass.canvas.texture.size, width=SIZE.PREVIEW_W)` sizing `app.preview_canvas` | **Unchanged, deliberately.** This asks "what aspect is the thing I am about to render into a thumbnail", and the answer is the output canvas's real current size, that is not a document-size question, and after this wave the two agree anyway. Listed so a reviewer knows it was checked and not missed. |
| `ui.py::_draw_document_image`, `image_aspect = np.divide(*ui_document.document.render_pass.canvas.texture.size)` | **Unchanged, deliberately.** Same reasoning: the viewer fits the texture it is about to blit, whose size is the authority on its own aspect. |
| `copilot/backend.py::_copilot_document_working_view`, `canvas=f"{...texture.size[0]}x{...texture.size[1]}"` | **Changed to `document.canvas_size`.** This string tells the MODEL how big the document is, which is a document-size question; it should read the field the copilot's own `set_canvas_size` writes, not the texture that happens to mirror it. One-line change, no behaviour difference once the funnel holds, and it removes the last place a stale output canvas could report a wrong size to the agent. |
| `copilot/backend.py::_render_facts_for`, `cw, ch = document.render_pass.canvas.texture.size`, sizing the probe canvas to match the document's aspect | **Unchanged, deliberately.** Its own comment says it matches the canvas ASPECT so `u_aspect` lays out as the preview does; that is the same fit-the-texture question `ui.py` asks, and the preview it must agree with reads the texture too. Changing one and not the other would be the only way to make them disagree. |

## Files touched

| File | What changes |
|---|---|
| `shaderbox/pass_graph.py` | `MIN_CANVAS_PX` / `MAX_CANVAS_PX` constants and the `clamp_canvas_size` free function, moved from `copilot/backend.py`. |
| `shaderbox/copilot/backend.py` | `_MIN_CANVAS_PX` / `_MAX_CANVAS_PX` deleted; `set_canvas_size` calls `clamp_canvas_size`; `_copilot_document_working_view` reports `document.canvas_size`. |
| `shaderbox/tabs/document.py` | The Resolution combo and its whole build block (`standard_resolutions`, `current_size`, the `get_active_uniforms` sampler scan, `resolution_items` / `resolution_sizes`, `resolution_label`) are replaced by the `W x H` `input_int` pair plus the presets dropdown, inside a `begin_disabled(app.copilot_turn_active)` scope; new module-level `_apply_canvas_size` and `_canvas_presets` free functions, `_SQUARE_PRESETS`, `_CANVAS_FIELD_W` and `_CANVAS_PRESETS_W` constants. |
| `shaderbox/app.py` | Two fields: `canvas_size_buf: tuple[int, int] = (0, 0)` and `canvas_size_editing: bool = False`; `_on_current_document_changed` clears the editing flag so a keyboard document switch re-arms the mirror (§ Design decisions item 3). |
| `shaderbox/popups/pass_settings.py` | `_draw_target` wraps the scale slider in `begin_disabled(is_output)` / `end_disabled`. No string changes. |
| `shaderbox/ui.py` | `_draw_canvas_backdrop` free function; `_draw_document_image` calls it before `image_with_bg` and draws the 1px border after. |
| `shaderbox/theme.py` | `COLOR.CHECKER_LIGHT` / `COLOR.CHECKER_DARK`; `SIZE.CHECKER_TILE`. |
| `shaderbox/ui_models.py` | `UIDocument.save` persists `document.canvas_size` rather than the output canvas's texture size. |
| `shaderbox/document.py` | No change, `set_canvas_size` is the funnel and already correct; its docstring already describes the bug this wave removes from the UI path. Listed so a reviewer knows it was checked. |
| `shaderbox/util.py` | No change, `get_resolution_str` keeps its signature and gains a second caller in `_canvas_presets`. Listed for the same reason. |
| `tests/test_document_graph.py` | The UI-funnel test (below). |
| `tests/test_canvas_presets.py` | New: the preset-composition and clamp tests. |
| `tests/test_document_ops.py` | No change, `test_set_canvas_size_applies_and_clamps` exercises the moved clamp through the public method. Listed for the same reason. |

## Tests

Each named with its falsifier: the bug that makes it go red.

### `tests/test_document_graph.py::test_a_ui_resize_moves_every_pass_together`

The parent spec's stated test: "after a UI resize, every non-output pass's
`target_size(document.canvas_size)` follows on the next render." Sibling of the existing
`test_a_resize_moves_every_pass_together`, which drives `Document.set_canvas_size` directly; this
one drives the function the UI calls. Builds the same three-pass graph (`half` at scale 0.5, `full`
at 1.0, `out` reading both), renders, calls `tabs.document._apply_canvas_size` with a stub carrying
a `notifications` object and the `UIDocument`, renders again, and asserts
`doc.canvas_size == (32, 32)`, `passes["full"].canvas.texture.size == (32, 32)` and
`passes["half"].canvas.texture.size == (16, 16)`.

GL-only, no imgui frame: `_apply_canvas_size` takes `app` solely to push a notification, so the
test passes a minimal stub with a `notifications.push` recorder rather than the `app` fixture.

**Falsifier:** with today's `render_pass.canvas.set_size((w, h))` in place of the funnel call,
`doc.canvas_size` stays `(64, 64)`, `full` stays 64 and `half` stays 32: findings #2 and #3
exactly. All three assertions go red.

### `tests/test_document_graph.py::test_the_ui_resize_clamps_both_ends`

Calls `_apply_canvas_size` with `(99999, 4)` and asserts `doc.canvas_size == (4096, 16)`.

**The test asserts the FIELD and does not render afterwards.** Under the bug it names the document
holds `(99999, 4)`, and a render at that size fails to complete the framebuffer, so a trailing
`doc.render()` would turn a clean assertion failure into a GL error that names nothing. The
assertion is the whole test.

**Falsifier:** the parent spec's constraint is that the clamp applies on BOTH paths. Without the
clamp in `_apply_canvas_size` (the failure mode being "the copilot clamps, the UI does not"), the
document takes `(99999, 4)` and the assertion goes red. This is the test that pins the second half
of the shared constant, the first half being `test_document_ops.py`'s existing clamp assertion.

### `tests/test_document_graph.py::test_an_unchanged_size_pushes_no_notification`

Calls `_apply_canvas_size` with the document's current size and asserts the stub's
`notifications.push` recorder is empty afterwards.

**Falsifier:** without the early return the recorder holds one `Canvas: WxH` entry, and the
assertion goes red. This is the version that actually falsifies: the drafted
`test_an_unchanged_size_does_not_reallocate` asserted texture identity, which holds with OR without
the early return, because `Canvas.set_size` guards the unchanged case itself. A test that is green
under its own named bug verifies nothing, so it is replaced rather than kept alongside.

### `tests/test_canvas_presets.py::test_the_square_presets_include_512`

Builds a single-pass document with no bound media and calls `_canvas_presets`. Asserts
`("512x512 (1:1)", (512, 512))` is in the returned list, and that `(256, 256)`, `(1024, 1024)` and
`(2048, 2048)` are too.

**Falsifier:** W-H's "Before you start" step names 512x512 and is unperformable without it; drop
`512` from `_SQUARE_PRESETS` and this goes red. This is the test that makes W-A's contract with W-H
mechanical rather than a promise in prose.

### `tests/test_canvas_presets.py::test_every_preset_survives_the_clamp`

For every `(label, size)` `_canvas_presets` returns, asserts `clamp_canvas_size(size) == size`.

**Falsifier:** a preset outside `[16, 4096]` would be silently altered on its way through
`_apply_canvas_size`, so the menu would show one number and the canvas would take another. Add an
`8192` square to `_SQUARE_PRESETS` and this goes red before a user ever sees the mismatch. (4096 is
IN range and would not demonstrate it; 8192 is the smallest square that does.)

### `tests/test_canvas_presets.py::test_the_video_shapes_come_from_the_shape_table`

Asserts the preset list contains exactly one entry per non-`NATIVE` member of
`render_shape.MENU_SHAPES`, each labelled with that shape's `SHAPE_TABLE[...].menu_label`, and that
`RenderShape.NATIVE` contributes none. **And asserts the DIMS**, recomputed in the test from
`render_shape` directly:

```python
        expected = resolve_dims(
            shape_to_preset(
                shape, is_video=False, fps=None, container=None, duration_max=None
            ),
            (1, 1),
        )
```

The `(1, 1)` source size is deliberate and is itself an assertion: `shape_to_preset` emits
`FIXED_ASPECT` for every non-`NATIVE` shape, and `resolve_dims`' `FIXED_ASPECT` branch computes
from `aspect` and `longest_edge` alone, so a source size of `(1, 1)` must yield the same dims the
menu produces from the real canvas. If it ever does not, the source-size independence this spec
asserts in item 5 is false and the test says so.

**Falsifier, two halves.** Coverage: adding a member to `MENU_SHAPES` turns it red until the menu
picks it up. Single-homing: hand-rolling the six sizes as literals while still reading
`SHAPE_TABLE[...].menu_label` for the label passes every label-and-membership assertion but goes
red on the dims the moment a literal drifts from the table's `longest_edge`, which is the drift
that would silently make the menu disagree with the Share tab about what "Wide 1080p" means. The
drafted version asserted labels and membership only, so the second half did not falsify.

### `tests/test_canvas_presets.py::test_a_bound_texture_is_offered_and_the_default_image_is_not`

Loads a document whose pass declares a `sampler2D`, renders once so seeding runs, and asserts no
preset names that uniform (the seeded value is the shipped default image). Then binds a real image
of a distinct size to a NON-OUTPUT pass and asserts a preset appears with that size and the
uniform's name in its label.

**Falsifier, two:** without the `is_default_image` filter, the first assertion goes red and every
document with any sampler lists the placeholder's dimensions. With the scan reading only
`document.render_pass` (the shape the old combo had), the second assertion goes red: a texture
bound on any pass but the output stays invisible, which is the parent spec's "any bound texture's
size" read literally.

### `tests/test_canvas_presets.py::test_building_the_presets_compiles_nothing`

Loads a multi-pass document WITHOUT rendering it, asserts every pass has `program is None`, calls
`_canvas_presets`, and asserts every pass still has `program is None`.

**Falsifier:** building the texture group from `get_active_uniforms()`, the natural move, and what
the code being replaced does, compiles each pass on the first Document-tab frame, and every
assertion after the call goes red. This is the test that pins 066 D1 against this wave; without it
the regression is invisible, because a compiled graph renders correctly and merely costs the frame.

### `tests/test_canvas_presets.py::test_no_preset_duplicates_the_current_size`

Sets the canvas to a size that a bound texture also has, and asserts no preset carries that size.

**Falsifier:** without the skip, the menu shows an item whose selection does nothing (the early
return in `_apply_canvas_size` swallows it), which reads as a broken menu.

### The disabled slider, the pair's commit, and the checkerboard: manual

Three things this wave changes have no headless test, each for a stated reason:

- **`begin_disabled`'s effect on the slider.** imgui reports a disabled widget's `changed` as
  `False` inside a real frame, but asserting that headlessly means driving a slider through a
  synthetic frame and trusting the interaction state, which `/imgui-ui § 0` warns reads differently
  headless. What IS pinned is the invariant underneath it (the renderer ignoring the output's
  scale) by the existing `tests/test_document_graph.py` assertion that the output keeps full size
  whatever its scale says. Manual verification item 4.
- **Commit-on-deactivate for the pair.** `is_item_deactivated_after_edit()` needs a real focus
  transition across frames; the same reason W-C states for `_draw_name`. What is pinned headlessly
  is everything the branch calls: `_apply_canvas_size`'s funnel, clamp, and no-op paths.
  Manual verification items 1 and 2.
- **The checkerboard and border.** Pixel output the agent cannot screenshot on this box
  (`/imgui-ui § 9`). Manual verification item 6.

## Manual verification

The parent spec's W-A line ("pick 512x512 from the combo; open a pass gear: size shows 512x512 at
100%; add a pass at 50%: its tile is 256x256 on the next frame; a fully transparent output shows
the checkerboard"), expanded to one falsifiable step per item.

1. **Type a size into the pair.** Open a multi-pass document (the Radiance Cascades example). Click
   into the width field, type `1024`, then click into the height field without pressing Enter.
   Expect: the canvas resizes to 1024 wide on that frame, and the height field still shows the old
   height. **That intermediate resize is intended, not a bug:** each field commits the whole pair
   when focus leaves it (§ Design decisions item 3), so tabbing out of W applies
   `(1024, old_h)` and tabbing out of H then applies `(1024, 1024)`. Type `1024` there and click on
   the panel background. Expect: the canvas is 1024x1024. Fails if either field discards its number
   on click-away (the deactivate branch is not wired) or if the size only lands on Enter.

2. **The clamp is visible.** Type `99999` into the width field and click away. Expect: the field
   shows `4096` and the canvas is 4096 wide. Fails if the field keeps `99999` (the buffer is not
   re-read from the document) or if the app hangs or crashes on the allocation (the clamp is not on
   this path).

3. **Pick 512x512 from the presets menu.** Click `presets`, pick `512x512 (1:1)`. Expect: both
   fields read `512`, and the viewer's preview turns square. Fails if the item is absent (W-H's
   first tutorial step is then unperformable) or if the fields keep the previous numbers while the
   canvas changes (the buffer was not re-synced).

4. **The gear agrees, and the output slider is dead.** With the canvas at 512x512, open the gear on
   the OUTPUT pass. Expect: the size row reads `size (512, 512)` and the percent slider is greyed
   and does not move when dragged, while its `(?)` beside it still shows its tooltip at full
   contrast. This is finding #3's exact repro: fails if the gear shows the size the canvas had
   BEFORE the change.

5. **A non-output pass follows.** With the canvas at 512x512, add a pass and set its size to 50% in
   its gear. Expect: on the next frame its tile in the strip is drawn from a 256x256 texture, the
   gear's own size row reads `size (256, 256)`. Then change the canvas to 1024x1024 and reopen that
   gear. Expect: `size (512, 512)`. Fails if the non-output pass keeps its old dimensions, which is
   finding #2 seen from the pass side.

6. **A transparent output shows the checkerboard.** Open a document whose shader writes
   `fragColor = vec4(0.0)`, or edit one to. Expect: a grey checkerboard fills the preview area
   exactly to the canvas's extent, with a thin border around it, and no checkerboard outside the
   image rect. Fails if the preview is indistinguishable from the panel (the backdrop is not
   drawn), if the checkerboard extends past the image (the per-tile clip is missing), or if it
   covers an OPAQUE render (it is drawn after the image rather than before).

7. **The border reads on an opaque dark output.** Open a document whose shader writes near-black.
   Expect: a 1px border marks the canvas extent against the panel. Fails if the canvas edge is
   still invisible, that is the half of #21 the checkerboard alone does not fix.

8. **A bound texture appears in the menu.** Bind an image of an unusual size (say 1234x567) to a
   sampler on a NON-output pass. Open the presets menu. Expect: an entry reading
   `1234x567 (...) - <uniform name>`. Fails if it is absent (the scan reads only the output pass)
   or if an entry for the shipped default image appears on a document with an unbound sampler (the
   `is_default_image` filter is missing).

9. **Opening the Document tab compiles nothing.** Reopen the app on a six-pass document and let it
   settle on the Document tab. Expect: the same startup behaviour as before the wave, the strip
   fills in over several frames (W-C's sweep) rather than the whole graph compiling on the first
   Document-tab frame. Fails if the first frame after opening the tab visibly stalls, which is the
   `get_active_uniforms` regression the test above pins.

10. **A restart keeps the size.** Set the canvas to 512x512, switch documents (which saves), quit,
    reopen. Expect: the document is 512x512. Fails if it reverts, which would mean
    `UIDocument.save` is persisting a canvas that disagrees with `document.canvas_size`.

11. **A copilot resize reaches the fields unclicked.** With no field focused, ask the copilot to
    set the canvas to 800x600. Expect: the fields read `800` and `600` without any click, while the
    turn is still running. Fails if they keep the old numbers until clicked: that is the
    latched-buffer bug the mirror rule exists to prevent, and it is the step that verifies § Design
    decisions item 3's frame-by-frame table.

12. **The whole first row is dead during a copilot turn.** Start any copilot turn and, while it
    runs, try to type into the width field, into the DOCUMENT-NAME field, and to open the presets
    dropdown. Expect: all three are dimmed and inert, exactly like the pass strip and the script row
    beside them. Fails if any accepts input: a resize landing mid-turn reallocates textures under a
    tool that has already read the size, and a name edit races
    `CopilotBackend.rename_document` on the same field.

13. **A keyboard document switch does not carry a half-typed width across.** Click into the width
    field, type `10` without leaving it, then press Ctrl+N. Expect: the new document appears and its
    canvas fields show ITS OWN size, not `10` and the previous document's height. Fails if the stale
    pair is displayed, which is the case `_on_current_document_changed`'s
    `canvas_size_editing = False` exists for: Ctrl+N is a GLOBAL Ctrl-chord that imgui routes
    through an active text input, so the mouse-only reasoning that a switch always deactivates the
    field does not hold.

14. **A half-typed number is not stolen by the mirror.** Click into the width field and type `12`
    without leaving it, then wait several seconds. Expect: the field keeps showing `12` rather than
    snapping back to the document's width. Fails if the digits vanish, which would mean
    `canvas_size_editing` is not being set from `is_item_active()` and the mirror is overwriting an
    active field.

## Verified / corrected premises

Every citation and claim the parent spec's W-A section and findings #1 #2 #3 #4 #21 make, checked
against the committed tree at `a246a19` (W-C and W-F landed). Line numbers below are the real ones
at that commit; the spec
above cites symbols.

| Parent-spec or finding citation | Verdict |
|---|---|
| The Document tab's combo bypasses `Document.set_canvas_size`; `tabs/document.py:111` is the bypass (#2, W-A bullet 1) | **Confirmed.** `tabs/document.py:111` is `ui_document.document.render_pass.canvas.set_size((w, h))`, inside `tabs.document.draw`. |
| `Document.set_canvas_size` is at `document.py:284` and is the funnel (#2) | **Confirmed.** `document.py:284` is the `def set_canvas_size` line; its body sets `self.canvas_size` then `self.render_pass.canvas.set_size(size)`, and its docstring names the copilot's old version of this exact bug. |
| Intermediate passes size from `document.canvas_size` on every render, `document.py:411` (#2) | **Confirmed.** `:410` is `if name != output:` and `:411` is `wanted = entry.target.target_size(self.canvas_size)`, inside `Document.render`. |
| The combo's list is at `tabs/document.py:53` (#1) | **Confirmed as the block's start.** `:53` is `standard_resolutions = [`; the six literal sizes run to `:59`. The combo itself is `:107-115`. |
| The combo is fed a literal index `0` every frame and has no selection state, `tabs/document.py:109` (W-A bullet 2) | **Confirmed as the claim, corrected as the location.** `:107` is `new_res_idx = imgui.combo("##resolution", 0, resolution_items)[1]`, with `0` a literal, and `:108` is `if new_res_idx != 0:`. The parent cites `:109`, two lines below the call, the claim holds, the line is off by two. |
| The combo's list is built from `render_pass` in a panel that otherwise uses `panel_pass` / `document` (W-A bullet 2) | **Confirmed, and it is the last such row.** `tabs.document.draw` reads `ui_document.document.render_pass` at `:63` (the current size) and `:68`/`:70` (the sampler scan); every uniform row below it goes through `app.panel_pass(app.current_document_id)` (`:31` in `_draw_auto_row`, `:125` in the main loop). |
| The clamp lives at `copilot/backend.py:135` (W-A bullet 2) | **Confirmed as the comment line, corrected as the constants.** `:135` is the comment `# Canvas-size clamp for set_canvas_size (feature 052): a sane render-resolution range.`; the constants are `_MIN_CANVAS_PX = 16` at `:136` and `_MAX_CANVAS_PX = 4096` at `:137`. Their only reader is `CopilotBackend.set_canvas_size` (`:1069`), at `:1079-1080`. |
| "`document.py` (no change; the funnel exists)" (W-A Files) | **Confirmed.** `set_canvas_size` needs no edit: it already writes the field AND resizes the output canvas, which is everything both callers need. |
| The gear's size label is computed from `document.canvas_size`, `pass_settings.py:174` (#3) | **Confirmed as the claim, re-pointed for W-C.** At `a246a19` the line is `pass_settings.py:188` (`canvas_w, canvas_h = document.canvas_size`) inside `_draw_target`, with `w`/`h` derived just below and the label at `:192`. The finding's `:174` was the pre-W-C location. This is why the gear shows the stale number: the field, not the texture, is what it reads. |
| The output pass's slider has no `begin_disabled` for `is_output`, `pass_settings.py:181` (#4) | **Confirmed as the claim, re-pointed for W-C.** At `a246a19`, `is_output` is bound at `pass_settings.py:175` and read only at `:209` (the help-text ternary); the `imgui.slider_float` is `:195`, unguarded. W-C moved these (it restructured `_draw_body` and added `_commit_pass_name`) but did not touch `_draw_target`'s body, which is W-A's only edit in the file. |
| `document.py:410` skips the scale for the output pass, so the output slider changes a stored number with no effect (#4) | **Confirmed.** `:410`'s `if name != output:` guards the whole size fixup, and `tests/test_document_graph.py` already asserts "the OUTPUT keeps full size whatever its own scale says". |
| A pass has no size of its own, only a `scale` in 0-1, `pass_graph.py:61` (#4) | **Confirmed.** `TargetConfig.scale` is `Field(default=DEFAULT_SCALE, gt=0.0, le=1.0)` at `pass_graph.py:61`, with the `le=1.0` bound commented as deliberate. |
| "every non-output pass's `target_size(document.canvas_size)`" names a `Pass` method (W-A bullet 1's test) | **Corrected.** `target_size` is a method of `TargetConfig` (`pass_graph.py:66`), not of `Pass`, `Pass` has no such method. The call in `Document.render` is `entry.target.target_size(self.canvas_size)`. The test's meaning is unchanged; the symbol is not `Pass.target_size`. |
| The preview is `imgui.image_with_bg` at `ui.py:604` "with the default (window) background and no border" (#21) | **Confirmed as the call, corrected as the mechanism, re-pointed for W-C.** At `a246a19` the call is `ui.py:619` inside `_draw_document_image` (defined at `:590`), with `img_min` captured at `:618`, and it passes only `image_size` / `uv0` / `uv1` / `tint_col`. But `bg_col`'s binding default is `ImVec4(0, 0, 0, 0)`, fully TRANSPARENT, not the window background (verified in the installed stub, `imgui_bundle/imgui/__init__.pyi::image_with_bg`). The reported symptom is right (a transparent output is indistinguishable from the panel, because nothing is drawn behind it); the cause is an absent background, not an opaque one. It matters for the fix: nothing has to be removed or overridden for a draw-list backdrop to show through. |
| Two dim greys for the checkerboard come from `theme.py` (#21, W-A bullet 4) | **Corrected, they do not exist yet.** `theme.py`'s `_ColorBag` has `BG_APP` / `BG_SURFACE` / `BG_POPUP` / `BG_FRAME` / `BORDER` and no checkerboard pair. Two new role tokens are added over `_P["bg_1"]` and `_P["bg_2"]`; the parent's constraint is that the call site carries no literal, and it holds. |
| `COLOR.BORDER` exists as a token for the 1px rect (#21) | **Confirmed.** `theme.py`'s `_ColorBag.BORDER` maps to `_P["bg_2"]` (`#3c3836`). |
| `SIZE.PASS_SETTINGS_W/H` is `theme.py:259` (#7, cited here because W-A edits the same function W-B resizes) | **Corrected.** `PASS_SETTINGS_W: int = 440` is `theme.py:259` and `PASS_SETTINGS_H: int = 400` is `:260`. W-A touches neither; W-B deletes `PASS_SETTINGS_H`. |
| `UIDocument.save` persists the output canvas's real size, so #2 hides across restarts, `ui_models.py:360` (#2) | **Confirmed, and it is a second instance of the same class.** `:360` is `"canvas_size": list(self.document.render_pass.canvas.texture.size)` inside `UIDocument.save`. The finding treats it as the reason the bug is invisible across restarts; this wave also changes it, because persisting a derived value in place of the authoritative one is #2's shape. |
| The shipped 512x512 example is 512x512 only because its `document.json` was hand-authored (#1) | **Confirmed.** `shaderbox/resources/document_examples/77a84d27-.../document.json` carries a `canvas_size` key, as do all seven shipped examples. |
| Today's only routes to 512x512 are the copilot tool or editing `document.json` (#1) | **Confirmed.** `grep` for `set_canvas_size` finds exactly two writers: `CopilotBackend.set_canvas_size` and the disk path via `load_from_dir` / `sync_documents_from_disk`. No UI path writes `document.canvas_size` at all. |
| "the named video shapes" are available to list (W-A bullet 2) | **Confirmed, with one correction to what they carry.** `render_shape.py::SHAPE_TABLE` gives each `RenderShape` an `aspect` and a `longest_edge`, NOT concrete dims, the dims come from `render_preset.resolve_dims(shape_to_preset(...), source_size)`. `RenderShape.NATIVE` has `aspect=None` and means "the canvas size", so it cannot be a canvas preset and is skipped. |
| "any bound texture's size" is discoverable (W-A bullet 2) | **Confirmed, with the discovery path corrected.** The old combo used `render_pass.get_active_uniforms()`, which COMPILES a never-attempted pass (`core.py::Pass.get_active_uniforms`, `:228-231`), doing that across every pass would violate 066 D1. `Pass.uniform_values` holds the bound `MediaWithTexture` directly (`core.py:382` type-checks it that way at bind time), so the scan reads that dict and forces no compile. |
| A bound texture is distinguishable from the seeded placeholder | **Confirmed.** `media.py::is_default_image` (`:179`) is the sole marker, and its own docstring states that `seed_uniform_values` fills every unbound sampler with `Image(DEFAULT_IMAGE_FILE_PATH)`. Without the filter every sampler-carrying document would offer the placeholder's size. |
| `get_resolution_str` is the label helper the combo used (#1's list format) | **Confirmed.** `util.py:69`; `tests/test_util.py:54-55` pins both its `- name` and bare forms. Reused unchanged by `_canvas_presets`. |
| A test already exercises the funnel's document-side behaviour | **Confirmed.** `tests/test_document_graph.py::test_a_resize_moves_every_pass_together` drives `Document.set_canvas_size` directly and asserts all three passes follow, it is the sibling the UI-path test is written against, and it is the natural home for the new ones (its `_document` fixture and `_red` helper are what they need). |
| A test already pins the clamp | **Confirmed.** `tests/test_document_ops.py::test_set_canvas_size_applies_and_clamps` asserts `(4096, 16)` from `(99999, 4)` through `CopilotBackend.set_canvas_size`. It exercises the moved constant through the public method, so the move is verified without editing the test. |
| A test already forbids a per-frame reallocation | **Confirmed.** `tests/test_document_graph.py::test_a_scaled_pass_keeps_its_size_across_frames` asserts the target texture object is identical across two renders, with a comment naming the feedback-history loss a reallocation causes. This is why `_apply_canvas_size` returns early on an unchanged size. |
| `begin_disabled` / `end_disabled` is the codebase's disable idiom, in fourteen places | **Confirmed as the idiom, corrected as the count.** `grep -rn begin_disabled shaderbox/` returns **20**, across `popups/settings.py`, `popups/lib_picker/__init__.py`, `popups/examples.py`, `popups/help.py`, `ui.py`, `ui_primitives.py`, `tabs/render.py`, `widgets/copilot_chat.py` and (missed by the first count) `exporters/telegram.py` x2 and `exporters/youtube.py` x2. All are a bare pair rather than a context manager, so the claim the count supports holds. |
| `add_rect_filled` on the window draw list is the codebase's background idiom | **Confirmed.** `ui_primitives.py::_code_chip`, `preview_cell`'s stale wash and its selection frame, and `widgets/cheatsheet.py`'s panel all draw this way, each converting a `COLOR` token via `imgui.color_convert_float4_to_u32`. |
| A button-opened (non-context) popup already exists in this codebase | **Refuted, and it changed the design.** Every `begin_popup*` site is either `begin_popup_context_item` (`widgets/pass_list.py`, three in `popups/lib_picker/tree.py`) or a menu-bar `begin_menu` (`ui.py`), an `open_popup` + `begin_popup` presets menu would be the only one of its shape. `tabs/document.py` already owns a hand-built dropdown for the uniform sort key (`imgui.begin_combo("##uniform_sort_key", f"Sort by: {...}")` + a `selectable` loop + `end_combo`), which is the same panel, the same shape, and a fixed preview string rather than the selection, so the presets menu is that, not a popup. |
| `ComboFlags_.no_arrow_button` and `begin_combo`'s flags parameter exist in this imgui-bundle build | **Confirmed.** `begin_combo(label: str, preview_value: str, flags: ComboFlags = 0)` and `ComboFlags_.no_arrow_button` (`= 1 << 5`, "Display on the preview box without the square arrow button"), both in the installed `imgui_bundle/imgui/__init__.pyi`. `height_large` (~20 items) is also available if the bound-texture group ever makes the list long enough to scroll. |
| `input_int` returns `(changed, value)` and `step=0` suppresses its spinner | **Confirmed.** `input_int(label: str, v: int, step: int = 1, step_fast: int = 100, flags: InputTextFlags = 0) -> Tuple[bool, int]` in the installed stub; the `-`/`+` buttons are drawn only for `step > 0`. `input_int2` also exists and takes the pair directly, but submits ONE item, so per-field deactivation is unavailable through it, rejected in § Design decisions item 3. |
| The Document tab's canvas control is gated on `copilot_turn_active` like its siblings | **Refuted, it is not, today, and neither is its neighbour.** `tabs/document.py`'s only `begin_disabled(app.copilot_turn_active)` pair is in `_draw_entry_points` (`:226` / `:254`), around the script row. BOTH controls above it are ungated: the resolution combo (`:106-115`) and the document-name input (`:102-104`), while every other document mutation in the panel and its widgets is gated (`widgets/pass_list.py::draw`, `widgets/uniform.py::_draw_play_stop`, `widgets/document_grid.py`, and the uniform-slider block). Both race a copilot write on the field they edit (`CopilotBackend.set_canvas_size`, `CopilotBackend.rename_document`), so W-A gates the whole row, not just the canvas half. |
| `copilot/backend.py` has other output-canvas size reads | **Confirmed, two, resolved oppositely.** `_copilot_document_working_view` builds the `canvas=` string the MODEL reads from `document.render_pass.canvas.texture.size`, a document-size question, changed to the field. `_render_facts_for` reads the same texture to match the probe canvas's ASPECT to the preview's, a fit-the-texture question, left alone, since `ui.py::_draw_document_image` reads the texture for the same reason and the two must agree. |
| `App.set_current_document_id` (`app.py:1012`) is where per-document transients are cleared | **Refuted on both halves; the reset itself is kept.** `App.set_current_document_id` is at `app.py:1043` (not `:1012`) and is a bare two-line forwarder to `self.session.set_current_document_id(id)` with no resets in it at all. The reset point is `App._on_current_document_changed` (`app.py:536`), which clears `editor_was_ever_focused` and which `ProjectSession` fires only when the id actually changes, so a no-op re-set cannot disturb a field being typed in. That handler gains `canvas_size_editing = False`. |
| A document switch cannot happen while a canvas field is active, so the reset is dead code | **Refuted, and this reinstated the reset.** True of the mouse, false of the keyboard. `NEW_DOCUMENT` is `_chord(K.n, K.mod_ctrl)` (`commands.py:108`) with `CommandSpec.scope`'s default `CommandScope.GLOBAL` (`commands.py:80`); `route_flag` (`commands.py:233-245`) returns `route_global` for any non-Alt chord and its own comment states that "an active text input owns all keyboard keys and imgui routes only Ctrl-chords through it". `hotkeys.py::spec_eligible` gates on the consumed-chord set, the editor / copilot focus flags and an open modal, with no active-input term, so Ctrl+N dispatches with a canvas field focused and `create_document_from_example` switches the document. Without the reset, a stale `canvas_size_editing == True` would show one document's half-typed width against another and could commit it on tab-out. The command palette is non-modal and has the same shape. |
| W-C's edits to `pass_settings.py`, `ui.py` and `app.py` collide with W-A's | **Refuted, re-checked against the landed commit.** At `a246a19`: in `pass_settings.py` W-C changed `draw_pass_settings`' close branch, `_draw_body`, `_draw_name` and added `_commit_pass_name`, leaving `_draw_target`'s body untouched, which is W-A's only edit there. In `ui.py` W-C changed the tick loop in `update_and_draw`; W-A changes `_draw_document_image` (`:590`). In `app.py` W-C added `close_pass_settings`, `open_pass_settings_for_panel_pass`, `open_add_pass` and two command bindings; W-A adds two fields and one line inside `_on_current_document_changed`, which W-C does not touch. No shared function in any of the three. |
| `/imgui-ui § 7.5` states Enter-commits, so `input_int` may diverge from `_draw_name` | **Refuted, superseded by W-C's landed rewrite.** At `a246a19` § 7.5's Pattern bullet is the D11 rule: an inline input whose commit performs a TRANSACTION "commits on `imgui.is_item_deactivated_after_edit()`, with `enter_returns_true` as the Enter shortcut and Esc as cancel", read "on the line IMMEDIATELY after the `input_text` call, before any `same_line`". W-C's `_draw_name` implements `committed or deactivated`. W-A's first draft omitted the flag, arguing that `input_int` deactivates on Enter anyway (true, and the OR's first term is redundant for this widget). It now carries both terms verbatim: one rule for every inline input in the codebase is worth more than one saved keyword argument. |
| `Canvas.set_size` reallocates on an unchanged size, so `_apply_canvas_size` needs an early return to prevent it | **Refuted.** `core.py::Canvas.set_size` opens with `if size == self.texture.size: return False`, and `Document.set_canvas_size` does nothing else that allocates. The early return is kept for a different and real reason: it suppresses a `Canvas: WxH` notification on a deactivate that changed nothing (§ Design decisions item 1). The test that pinned the false premise asserted texture identity and would have passed either way; it is replaced by `test_an_unchanged_size_pushes_no_notification`. |
| `resolve_dims`' FIXED_ASPECT branch ignores `source_size`, so the canvas size passed in is inert | **Confirmed, and now asserted rather than argued.** `render_preset.py::resolve_dims`' `FIXED_ASPECT` branch computes `w`/`h` from `preset.aspect` and `preset.longest_edge` alone and returns `_align(w), _align(h)`. `test_the_video_shapes_come_from_the_shape_table` recomputes the expected dims with a `(1, 1)` source size, so the independence is a test assertion, not a spec claim. |

Corrected or refuted: **18** of 40 rows (11 corrections, 7 refutations). Six of those verdicts
flipped under review, against a spec that had verified them against `faccf0e`: the
`Canvas.set_size` reallocation premise, the reset point, the two ungated controls, the skill's
Enter rule and the W-C line numbers in round 1, and in round 2 the claim that a document switch
cannot happen while a field is active. Each is recorded above with what replaced it. The last one
is the instructive one: it was not a stale citation but a generalisation from the mouse to "a
document switch", made while deleting the very guard that covered the keyboard path.

## Open questions

Each carries a robust default, marked as such; none blocks implementation.

1. **Should the presets menu offer a "match a bound texture" entry when the texture is on the
   OUTPUT pass and already equals the canvas?** Default, taken: **no**, the current size is
   skipped whatever pass carries it, because selecting it is a no-op the early return swallows.
   Revisit only if a user reads the absence as the texture not being detected; the fix would be to
   show it disabled with a tick, which costs a menu-item state for a case with no action behind it.

2. **Does the checkerboard belong behind the pass-strip thumbnails too?** Default, taken: **no.**
   Finding #21 names the viewer, `preview_cell` already carries its own border and stale wash, and a
   12px checker behind a 112px tile would fight the thumbnail rather than frame it. Revisit if the
   maintainer reports the same "where is the canvas" confusion on the strip; the change would be a
   flag on `preview_cell` calling the same `_draw_canvas_backdrop`, promoted to `ui_primitives.py`.

3. **Where should `_CANVAS_FIELD_W` and `_CANVAS_PRESETS_W` live, module constants or `SIZE`
   tokens?** Default, taken: **module constants in `tabs/document.py`**, because both are
   arithmetic derived from `SIZE.RES_COMBO_W` (two fields plus a separator plus a dropdown inside
   the reserved row width), not independent design choices, and `theme.py` is the home for choices.
   Promote them to `SIZE` the moment a second panel needs a canvas-dimension field.

4. **Should `_apply_canvas_size` save the document?** Default, taken: **no.** The copilot's
   `set_canvas_size` calls `_save_ui_document` because a copilot turn must leave disk consistent
   for its own later reads; the UI path matches every other Document-tab edit (the name field, the
   uniform sliders), which persist on document-switch and shutdown through `UIDocument.save`. Adding
   a save here would write the whole document dir on every deactivate, including the ones that
   changed nothing, and `UIDocument.save` compiles every program-less pass before rebuilding
   the uniform block (066 D1's third puller), so the cost is a graph compile rather than
   merely a file write. Revisit if a canvas size is ever lost on a crash, the same argument would then
   apply to every other field in the tab, so the fix would be a general autosave, not a special case
   here.

5. **Does the presets dropdown need `ComboFlags_.height_large`?** Default, taken: **no.** The list
   is 10 items plus one per distinct bound texture, and imgui's default (`height_regular`, ~8
   visible) scrolls the rest, which is acceptable for a menu whose first ten entries are static and
   ordered. Add the flag if a maintainer check finds the squares scrolled out of view on a document
   with several bound textures, it is one argument.

## Review history

**Round 1, pre-implementation review** (`reviews/wave_a_pre.md`, one reviewer, correctness & design
plus verification & blast radius): parent-bullet coverage **PASS** (all five bullets covered);
locked-decision fidelity **PARTIAL** (D1 and D2 hold; D11's pair trace stated and defensible, but
the buffer's mirror rule was not implemented by the code given); design correctness **FAIL** (three
blocking findings); test falsifiability **PARTIAL** (seven of nine tests went red under their
stated bug, two did not).

Ten findings, all accepted. The rulings on the non-mechanical ones are the main session's, recorded
here because several chose between alternatives the reviewer left open:

| # | Finding | Resolution |
|---|---|---|
| F1 | *(blocking)* The `\| None` buffer never returned to `None`, so an externally-set canvas size could never reach the fields, and the spec's own manual items 11 and 4a's closing sentence were false. | Adopted the reviewer's active-item rule: the buffer mirrors `document.canvas_size` on every frame in which neither field is active, `canvas_size_editing` comes from `is_item_active()` read on the line after each `input_int`, and the buffer loses its `\| None`. Item 3 now carries the frame-by-frame table that makes manual item 11 true. |
| F2 | *(blocking)* `Canvas.set_size` already early-returns on an unchanged size, so the early return had no reallocation to prevent and `test_an_unchanged_size_does_not_reallocate` was green either way. | Kept the early return with its real reason (no notification on a click-away that changed nothing) and replaced the test with `test_an_unchanged_size_pushes_no_notification`, whose falsifier is the recorder holding one entry. |
| F3 | *(blocking)* `App.set_current_document_id` is a bare forwarder at `:1043`, not a reset point at `:1012`. | Both halves corrected. The further ruling made in round 1 (delete the reset entirely) was **refuted in round 2** and reverted; see below. |
| F4 | *(required change)* The document-name input's copilot gate was deferred to "whichever wave next edits that input", which is nowhere. | Lands in W-A. One `begin_disabled(app.copilot_turn_active)` pair now wraps the whole first row; open question 6 deleted, manual item 12 extended, premise row corrected. |
| F5 | The notification fires on a click-away that changed nothing unless the early return is kept for that reason. | Folded into F2's fix. |
| F6 | `test_the_video_shapes_come_from_the_shape_table` asserted labels and membership only, so the single-homing half of its falsifier did not fire. | Added the dims assertion against `resolve_dims(shape_to_preset(...), (1, 1))`, which also turns the spec's source-size-independence claim into an assertion. |
| F7 | `_CANVAS_PRESETS_W` had no value and the row's widths were not shown to fit. | Both numbers fixed with the sum shown: the field width gives, 62 to 56, so 56 + 4 + 7 + 4 + 56 + 8 + 64 = 199 against `SIZE.RES_COMBO_W`'s 200. |
| F8 | The copilot-turn gate closes none of the five findings and was not labelled as scope. | Named in § Findings folded as a consistency fix the wave took on, alongside the two same-class-as-#2 changes. |
| F9 | Two counts wrong: fourteen `begin_disabled` sites (really 20) and `app.py:1012` (really `:1043`). | Both corrected in the prose and the premise rows. |
| F10 | W-C landed `/imgui-ui § 7.5` as the OR of `enter_returns_true` and the deactivate query; W-A's draft took the opposite default without naming the divergence. | Adopted the OR verbatim rather than explaining a divergence. One rule for every inline input; open question 1 deleted. |

Smaller notes, all folded: manual item 1 states that the intermediate resize on tab-out of W is
intended; `test_the_ui_resize_clamps_both_ends` says it asserts the field and does not render;
`test_every_preset_survives_the_clamp`'s demonstration corrected from 4096 (in range) to 8192; open
question 4 gains the fact that `UIDocument.save` compiles every program-less pass; item 5 states
that the `MediaWithTexture` filter admits a bound video and excludes a `moderngl.Buffer`.

All line citations in the three files W-C touched (`popups/pass_settings.py`, `ui.py`, `app.py`)
were re-pointed from `faccf0e` to the committed tree at `a246a19`. The reviewer's four "false
trails" sections confirmed the clamp's new home, the checkerboard draw order, the `input_int2`
rejection and the two fit-the-texture reads left unchanged; none required an edit.

**Round 2, closure** (same reviewer, narrow: does each of F1..F10 have text that closes it): nine
**CLOSED**, one **NOT CLOSED**. The two round-1 rulings that went further than proposed were
re-derived rather than accepted on the ledger's word, and one of them did not survive.

- **F3 NOT CLOSED, and the round-1 ruling is refuted.** The line-number and forwarder halves were
  corrected, but deleting the reset rested on "switching documents cannot happen while a canvas
  field is active", which the command registry contradicts: `NEW_DOCUMENT` is Ctrl+N, GLOBAL by
  `CommandSpec`'s default, and `route_flag`'s own comment says imgui routes Ctrl-chords through an
  active text input, with no active-input term in `hotkeys.py::spec_eligible`'s gate. Type into W,
  press Ctrl+N, and a stale `canvas_size_editing == True` shows one document's half-typed width
  against another. The reviewer's option (i) is taken: `App._on_current_document_changed`
  (`app.py:536`) sets `canvas_size_editing = False`, `app.py` returns to § Files touched as a
  handler edit, decision 3's third bullet and the premise row are rewritten, and manual item 13 is
  the falsifiable step. The lesson is worth keeping: the round-1 ruling generalised from the mouse
  to "a document switch" without checking the keyboard path, which is exactly the
  verify-the-premise failure the project's debugging rules name.
- **F10's ruling upheld.** Round 2 re-derived it and found the OR matches W-C's landed `_draw_name`
  and § 7.5 at `a246a19` token for token, and credited the spec for saying that the first term is
  redundant for `input_int` so a later reader does not delete it as dead code.
- **F1 re-traced against three inputs** (a copilot write mid-turn, a disk sync during typing, a
  W-to-H tab) and closed on all three; the round-2 trace confirms that writing the editing flag
  after both fields and reading the mirror before them is what makes the tab case safe, which the
  spec states as a design property rather than an accident of the sample code.

**Round 2 false trail (b), accepted as an inherent consequence.** If a disk sync replaces the
`Document` while the user is mid-edit in W, the buffer's H half still holds the PRE-sync height, so
the tab-out commits `(new_w, stale_h)` and reverts an externally-set height. This is inherent to any
pending-edit buffer that carries both halves of a pair, and it is the deliberate trade in decision
3's "Both fields commit as ONE `set_canvas_size` call" bullet: suppressing it would need the
touched-field ledger that bullet rejects. Its trigger is a hand edit of `document.json` landing
inside a typing window, which D2 declassifies as a workflow. Recorded, not fixed.

**Post-implementation correction: `_canvas_presets` lost its `app` parameter.** Both rounds carried
the signature `(app: App, ui_document: UIDocument)`, and implementation showed the body never reads
`app` — the composition is a pure function of the document. Neither gate would have caught it
(ruff selects no `ARG` rules; pyright's `include` is `shaderbox` and basic mode does not flag an
unused parameter), so it would have shipped as a dead parameter with an `_ = app` line explaining
nothing. The signature above, the draw-loop call site in decision 4, and the three tests that reach
it through one `_presets` helper all take `ui_document` alone. `_apply_canvas_size` keeps its `app`,
which it genuinely uses for the notification.
