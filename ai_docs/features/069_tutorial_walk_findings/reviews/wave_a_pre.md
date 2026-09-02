# W-A pre-implementation review — canvas size and the viewer

Reviewer role per `dev_flow.md` step 4: correctness & design, plus verification & blast radius.
Artifact under review: `ai_docs/features/069_tutorial_walk_findings/20_wave_a_canvas_viewer.md`.
Anchors: `01_spec.md` (D1, D2, D11, § W-A, § W-H, § Manual verification), `00_findings.md`
#1 #2 #3 #4 #21, and the tree at `faccf0e` plus W-C's uncommitted diff.

## Verdict

| Dimension | Verdict |
|---|---|
| Parent-bullet coverage | **PASS** — all five W-A bullets covered, each with a decision number. |
| Locked-decision fidelity | **PARTIAL** — D1 and D2 hold; D11's pair trace is stated and defensible, but the buffer's stated "mirror the document" rule is not implemented by the code the spec gives (F1). |
| Design correctness | **FAIL** — F1 breaks two manual-verification items the spec itself writes; F2 rests a decision and a test on a false premise about `Canvas.set_size`; F3 names a reset point that does not exist. |
| Test falsifiability | **PARTIAL** — seven of nine named tests go red under their stated bug; `test_an_unchanged_size_does_not_reallocate` is green either way (F2), and `test_the_video_shapes_come_from_the_shape_table` as worded does not falsify the hand-rolled-literals bug it claims (F6). |

Three findings are blocking (F1, F2, F3). One is a required change the main session already ruled
on (F4). The rest are corrections and one scope call.

---

## Findings

### F1 (blocking). The buffer never returns to `None`, so an externally-set canvas size never reaches the fields — and two of the spec's own manual-verification items fail as written

**Claim.** § Design decisions item 3 states `None` means "not editing; mirror the document, which
is what makes the field follow a size set from anywhere else (the copilot's `set_canvas_size`, a
`document.json` edit the per-frame sync picks up, a preset)". The code block given implements no
path back to `None`.

**Evidence.** The block (spec lines 183-205) is the whole of the row's state handling:

```python
    w, h = ui_document.document.canvas_size
    if app.canvas_size_buf is None:
        app.canvas_size_buf = (w, h)
```

`w, h` are read and then used nowhere else in the block. `canvas_size_buf` goes non-`None` on the
first frame the tab draws and, per § Files touched, is set back to `None` only in
`set_current_document_id`. Every other assignment writes a tuple. So after the first frame the
document's field is never read again while the tab stays on one document.

Consequences, traced against the spec's own list:

- **Manual verification item 11** ("Ask the copilot to set the canvas to 800x600. Expect: the
  fields update to `800` and `600` on the next frame without being clicked.") fails. The copilot's
  `set_canvas_size` writes `document.canvas_size` (`copilot/backend.py:1085`); the fields keep
  showing the stale pair indefinitely. The spec names this exact expectation and attributes it to
  "the buffer's `None`-means-mirror rule", which the code does not have.
- **The disk-sync path** fails the same way. `sync_documents_from_disk` replaces the whole
  `Document` (`tests/test_document_dir_sync.py::test_changed_document_json_reloads` pins the
  reload); the buffer survives on `App` and keeps the old numbers.
- **§ Design decisions item 4a's closing sentence** ("a canvas the copilot sets mid-turn is what
  the fields show when the gate lifts") is false for the same reason.

Note that the two *preset* and *field-commit* paths are safe: both re-sync explicitly
(`app.canvas_size_buf = ui_document.document.canvas_size`). It is only the external writers that
are lost.

**Fix (paste-ready).** In § Design decisions item 3, replace the buffer's seeding block and the
`None`-means-mirror paragraph with an active-item rule: the buffer mirrors the document on every
frame in which neither field is being edited, and holds only while one is.

> ```python
>     w, h = ui_document.document.canvas_size
>     editing = app.canvas_size_editing
>     if not editing:
>         app.canvas_size_buf = (w, h)
> ```
>
> where `canvas_size_editing: bool` is set from `imgui.is_item_active()` read on the line after
> each `input_int` and OR-ed across the two fields, and `canvas_size_buf: tuple[int, int]` loses
> its `| None`. A field the user is not typing into therefore shows `document.canvas_size` on
> every frame, which is what makes a copilot write, a disk edit and a preset all visible without a
> click; a field being typed into keeps its half-typed text. The commit branch still re-reads the
> document afterwards so a clamped value replaces the rejected one.

Whatever shape is chosen, the spec must state the frame-by-frame rule that makes item 11 true, and
the reset in § Files touched must move with it (see F3).

### F2 (blocking). `Canvas.set_size` already early-returns on an unchanged size, so the early return in `_apply_canvas_size` has no reallocation to prevent — and the test written to pin it cannot go red

**Claim.** § Design decisions item 1 justifies `_apply_canvas_size`'s early return with "`Canvas.set_size`
reallocates: `set_canvas_size` would drop and rebuild the output texture on a commit that changed
nothing", and § Tests gives
`test_an_unchanged_size_does_not_reallocate` the falsifier "without the early return,
`set_canvas_size` -> `Canvas.set_size` reallocates the texture."

**Evidence.** `shaderbox/core.py:95-101`:

```python
    def set_size(self, size: tuple[int, int]) -> bool:
        if size == self.texture.size:
            return False

        self.release()
        self._init(size)
        return True
```

The identity guard is already inside `Canvas.set_size`. `Document.set_canvas_size`
(`document.py:284-293`) does nothing else that allocates: it assigns the field and calls
`Canvas.set_size`. So with the early return deleted, `doc.render_pass.canvas.texture` is still the
same object, and `test_an_unchanged_size_does_not_reallocate` passes under the bug it names. It is
a green-either-way test, which is the checker-narrowing shape the project's own debugging rules
call the most expensive family.

The cited sibling `test_a_scaled_pass_keeps_its_size_across_frames`
(`tests/test_document_graph.py:469-486`) is about the per-frame resize inside `Document.render`,
which is likewise guarded (`document.py:431`, `if render_pass.canvas.texture.size != wanted:`). It
does not support the claim about `set_canvas_size` either.

**Fix (paste-ready).** Two edits.

> In § Design decisions item 1, replace the reallocation justification with the real one: the early
> return exists so a deactivate that changed nothing pushes no notification — a `Canvas: 512x512`
> toast on every click-away through an untouched field is noise. `Canvas.set_size`
> (`core.py:95-97`) already refuses an unchanged size, so no texture is at risk either way.

> In § Tests, either drop `test_an_unchanged_size_does_not_reallocate` or restate it as
> `test_an_unchanged_size_pushes_no_notification`, asserting the stub's `notifications.push`
> recorder is empty after `_apply_canvas_size` is called with the document's current size.
> Falsifier: without the early return the recorder holds one entry.

### F3 (blocking). `App.set_current_document_id` is a bare forwarder with no transient resets; the real reset point is `App._on_current_document_changed`

**Claim.** § Design decisions item 3: the buffer "is reset to `None` in
`App.set_current_document_id`, beside the other per-document transient resets". § Verified /
corrected premises repeats it as **Confirmed**, citing `app.py:1012`.

**Evidence.** `shaderbox/app.py:1043-1044`, the whole method:

```python
    def set_current_document_id(self, id: str = "") -> None:
        self.session.set_current_document_id(id)
```

There are no other resets there, and it is line 1043, not 1012. The per-document transient reset
point is `App._on_current_document_changed` (`app.py:536-541`), which clears
`editor_was_ever_focused` and opens the new document's shader tab; it is fired from
`ProjectSession.set_current_document_id` (`project_session.py:281-285`) and only when the id
actually changes. That guard matters: putting the reset in the forwarder would also fire it on a
no-op re-set of the same id.

**Fix (paste-ready).** In § Design decisions item 3 and § Files touched, replace both mentions of
`set_current_document_id` with `App._on_current_document_changed` (`app.py:536`), the handler that
already clears `editor_was_ever_focused` and only fires when the id actually changes. Mark the
premise row **Corrected** rather than Confirmed.

### F4 (required change, per the main session's ruling). The document-name input's copilot gate lands in W-A

**Claim.** Open question 6 defers the document-name input's `begin_disabled` to "whichever wave
next edits that input".

**Ruling.** The main session's ruling for this review: it lands in W-A. Same file, same row, one
`begin_disabled` pair. A deferral to "whichever wave next edits that input" is a deferral to
nowhere — no later wave in `01_spec.md § Workstreams` names `tabs/document.py`'s name input.

**Evidence that it is the same defect class.** `tabs/document.py:100-103` draws
`input_text_with_hint("##document_name", ...)` writing `ui_document.ui_state.ui_name` outside any
disabled scope; the only `begin_disabled(app.copilot_turn_active)` in the file is
`_draw_entry_points`' at `:226`/`:254`. The copilot writes the same field:
`CopilotBackend.rename_document` sets `ui_state.ui_name` (pinned by
`tests/test_document_ops.py:127`). So an un-gated name input is a mid-turn write racing a copilot
write on one field, which is the same hazard the row-level gate exists for, minus the GL.

**Fix (paste-ready).** Delete open question 6. In § Design decisions item 4a, extend the gate to
open before the `small_caption` / `input_text_with_hint` pair for the document name and close
after the presets dropdown, so ONE `begin_disabled(app.copilot_turn_active)` /
`end_disabled` pair wraps the whole first row. Add a manual-verification step: during a copilot
turn the document-name field is dimmed and inert alongside the canvas fields. Delete the sentence
in item 4a that says the name input "stays ungated, unchanged", and correct the § Verified /
corrected premises row that ends "the name input is left as it is".

### F5. The `_apply_canvas_size` notification fires on a click-away that changed nothing, unless the early return is kept for that reason

Related to F2 but separate in effect. `is_item_deactivated_after_edit()` fires on any deactivate
that followed an edit, including one where the user typed a digit and deleted it. With the early
return present the notification is suppressed; the spec should say that is the return's purpose
(F2's fix does this). Flagging it separately so the fix is not lost if F2 is resolved by deleting
the return instead.

### F6. `test_the_video_shapes_come_from_the_shape_table` as worded does not falsify the bug it names

**Claim.** § Tests: the test "asserts the preset list contains exactly one entry per non-`NATIVE`
member of `render_shape.MENU_SHAPES`, each labelled with that shape's `SHAPE_TABLE[...].menu_label`",
with the falsifier "hand-rolling the six sizes as literals instead of going through
`shape_to_preset` + `resolve_dims` passes today and drifts the moment `SHAPE_TABLE` gains a tier".

**Evidence.** The assertions as described check labels and membership, not dims. A hand-rolled
implementation that reads `SHAPE_TABLE[shape].menu_label` for the label and hard-codes
`(1080, 1920)` etc. for the dims passes every stated assertion. The second half of the falsifier
(adding a member to `MENU_SHAPES` turns it red) is real, but that is the coverage half, not the
single-homing half.

The dims are checkable without a second source of truth:
`resolve_dims(shape_to_preset(shape, is_video=False, fps=None, container=None, duration_max=None),
any_size)` is a pure function whose FIXED_ASPECT branch (`render_preset.py:46-55`) ignores
`source_size` entirely, as the spec correctly observes.

**Fix (paste-ready).** Add to the test: for every non-`NATIVE` shape, assert the preset's dims
equal `resolve_dims(shape_to_preset(shape, is_video=False, fps=None, container=None,
duration_max=None), (1, 1))`, computed in the test from `render_shape` directly. Falsifier: a
hand-rolled literal that drifts from `SHAPE_TABLE`'s `longest_edge` goes red, and the
source-size-independence the spec asserts is asserted rather than argued.

### F7. `_CANVAS_PRESETS_W` is introduced without a value or an arithmetic, and the row's widths are not shown to fit

§ Design decisions item 3 fixes `_CANVAS_FIELD_W = 62.0` and argues it from "two fields plus the
`x` plus the presets button must fit the `SIZE.RES_COMBO_W` width the row already reserves"
(`SIZE.RES_COMBO_W = 200`, `theme.py:228`). Item 4 then introduces `_CANVAS_PRESETS_W` with only
"sized to the word plus padding" and no number, and § Files touched lists it as a constant to add.
The arithmetic as given: 62 + `SPACE.SM` (4) + the `x` glyph + `SPACE.SM` (4) + 62 + `SPACE.MD` (8)
= about 147 before the dropdown, leaving roughly 53px for a `presets` preview plus frame padding.
That is tight enough that it should be a stated number rather than left to implementation.

**Fix (paste-ready).** In § Design decisions item 4, give `_CANVAS_PRESETS_W` a value and show the
sum against `SIZE.RES_COMBO_W`: `62 + 4 + x_glyph + 4 + 62 + 8 + _CANVAS_PRESETS_W <= 200`. If the
sum does not close, say which of the two numbers gives, or state that the cluster is allowed to
exceed `RES_COMBO_W` and what it is measured against instead.

### F8. Scope call on the two additions beyond the parent's bullets: both belong, one for a stated reason the other lacks

- **`UIDocument.save` reading `document.canvas_size`** (item 1): **same defect class as #2, belongs
  here.** `ui_models.py:360` is `"canvas_size": list(self.document.render_pass.canvas.texture.size)`;
  `document.py:487` and `:503` read `metadata.get("canvas_size")` back into `Document` and each
  `Pass`. Persisting the derived value to restore the authoritative one is #2's shape at the save
  seam, and #2's ledger entry names it explicitly ("Saving is unaffected ... so the bug hides across
  restarts"). Nothing breaks: `tests/test_persistence_completeness.py`'s roster enumerates modules
  containing `json.load` (its `loaders` set is built from `"json.load" in path.read_text()`), which
  the change does not touch, and no test asserts the saved value's provenance. Manual verification
  item 10 is the right check.
- **`_copilot_document_working_view` reading `document.canvas_size`** (item 8's table): **same
  class, belongs here.** `copilot/backend.py:728` builds the `canvas=` string the model reads from
  the output texture; the copilot's own `set_canvas_size` writes the field. Same read-the-derived
  shape, one line, no behaviour change once the funnel holds.
- **The copilot-turn gate on the canvas row** (item 4a): **scope growth, but justified and named as
  such — with a correction.** It is not #2/#3/#4/#21; it is a consistency gap the drafter found
  while reading. The spec argues it from the GL-reallocation hazard, and the row is genuinely the
  odd one out (`tabs/document.py:226` is the file's only gate; `widgets/pass_list.py:147`,
  `widgets/uniform.py:160`, `widgets/document_grid.py:63`/`:87` all carry it). Keep it, but say in
  § Out of scope or § Findings folded that it is a consistency fix the wave took on rather than one
  of the five findings — otherwise a later reader looks for the finding it closes and finds none.
  With F4 folded in, the same sentence covers the name input.

### F9. Two premise-row counts are wrong

- "`begin_disabled` ... (fourteen call sites)" (item 6) and the § Verified row "Fourteen call
  sites". The real count is **20** (`grep -rn "begin_disabled" shaderbox/ | wc -l` → 20, across the
  files the spec lists plus `exporters/telegram.py` ×2 and `exporters/youtube.py` ×2). The claim
  the count supports (it is the codebase's bare-pair idiom) holds; the number does not. Per the
  project's own rule that a number in a doc is read as established, either correct it to 20 or drop
  the count and keep the claim.
- The § Verified row citing `App.set_current_document_id` at `app.py:1012`: it is at `:1043`, and
  the row's verdict is wrong outright (F3).

---

## Coverage, item by item

### Parent W-A bullets

| Parent bullet | Verdict | Wave decision |
|---|---|---|
| Combo routes through `Document.set_canvas_size`; test that every non-output pass follows | **Covered** | Item 1; `test_a_ui_resize_moves_every_pass_together` |
| Resolution control redesign: `W x H` `input_int` pair, D11 commit, clamp from `copilot/backend.py:135` moved to one shared constant | **Covered** with F1 outstanding | Items 2, 3 |
| Presets menu (squares 256/512/1024/2048, named video shapes, any bound texture's size) writing the pair | **Covered** | Items 4, 5 |
| Output pass's size slider disabled with the help text it already has | **Covered** | Item 6 |
| Viewer: checkerboard + 1px border, two greys from `theme.py`, no literal at the call site | **Covered** | Item 7 |

Files: the parent lists `tabs/document.py`, `popups/pass_settings.py`, `ui.py`, `theme.py`,
`document.py` (no change), `copilot/backend.py`, tests. The wave adds `pass_graph.py` (the clamp's
new home), `app.py` (the buffer), `ui_models.py` (the save fix) and `util.py` (no change). Each
addition is argued; `pass_graph.py` as the clamp's home is correct on the funnel law
(`conventions.md:117`) and on import direction — it is GL-free, imports no imgui and no `Document`,
and already owns `TargetConfig.scale`'s bound (`pass_graph.py:61`).

### Findings #1 #2 #3 #4 #21

- **#1** ("there is no 512x512 option, where did you find it?"). **Closed** by item 5 group 1
  (`_SQUARE_PRESETS = (256, 512, 1024, 2048)`) and item 3 (free-form entry). Verified: the current
  combo's list (`tabs/document.py:53-59`) is the six literals the finding names, with no 512
  square, and `grep set_canvas_size` finds only the copilot tool and the disk load path as writers
  of `document.canvas_size` — the finding's "only two routes" claim holds.
- **#2** ("bypassing `Document.set_canvas_size` ... the funnel that updates `document.canvas_size`").
  **Closed** by item 1's `_apply_canvas_size` as the sole write path, plus the two derived-read
  sites in item 8's table. `tabs/document.py:111` is the bypass, verbatim.
- **#3** ("I changed the canvas to 1080x1080, the first pass settings still has 1280x960"). **Closed**
  by the same funnel: the gear reads `document.canvas_size` (`popups/pass_settings.py:190` in the
  W-C working tree, `:174` at `faccf0e`), which the funnel now writes. Manual verification item 4
  is the finding's exact repro. D2's "manual JSON editing is not a workaround" is honoured — the
  free-form pair is the UI route.
- **#4** ("is it even possible to have the first pass at a resolution different from the canvas?" +
  "a control that does nothing is a UX defect; either disable it or hide the row"). **Closed** by
  item 6, disable, which is the parent's pick. Verified: `document.py:429`'s `if name != output:`
  guards the size fixup, and `popups/pass_settings.py` binds `is_output` at `:177` and reads it only
  in the help-text ternary — the slider at `:194-200` is unguarded today. The finding's other half
  (whether scale-only is the right model) is correctly routed to the parent's § Out of scope.
- **#21** ("when the canvas is transparent we can't see the canvas boundary"). **Closed** by item 7,
  both halves: the checkerboard for the transparent case and the border for the opaque-dark case
  the finding also names.

### D1 (word budget)

**Holds.** The row caption becomes `Canvas` (1 word, inside the 1-2 budget); today's
`Resolution` plus its `f" ({current_name})"` suffix (`tabs/document.py:91-93`) is a derived value
in a label, which `/imgui-ui § 2` forbids, and it goes. The fields carry `##`-only ids. The
notification `Canvas: 1080x1080` is the skill's own example for the notification budget. No
`help_marker` is added. The presets entries use `get_resolution_str` (`util.py:69`), which the old
combo already used. Item 6's "the help text is unchanged, both variants, verbatim" correctly leaves
W-B's cut to W-B, and the spec's § Out of scope explains why W-A must not pre-satisfy W-B's gate —
that reasoning is sound: W-B's gate is written to refuse `f"size ({w}, {h})"`
(`pass_settings.py:193`), which W-A leaves standing.

### D2 (no JSON editing)

**Holds.** After the wave, 512x512 is reachable two ways in the UI (typed pair, preset), and every
size in `[16, 4096]` is reachable by typing.

### D11 (commit on deactivate) on a PAIR

The spec **does** say which of the two behaviours it picks and why, so it is not ambiguous:
§ Design decisions item 3 states "Both fields commit as ONE `set_canvas_size` call", then traces
"edit W, tab into H, edit H, click away" as **two** `set_canvas_size` calls — the first carrying
`(new_w, current_h)` on leaving W, the second `(new_w, new_h)` on leaving H — and rejects the
defer-until-both alternative as needing a touched-field ledger. My trace of the given code agrees:

- Frame N (focus leaves W): `input_int` W returns `buf_w = new_w`; `is_item_deactivated_after_edit()`
  True; buffer becomes `(new_w, old_h)`; H draws from `app.canvas_size_buf[1] = old_h`, not
  committed; `committed_w` fires; `_apply_canvas_size(app, ui_document, (new_w, old_h))`; buffer
  re-read as `(new_w, old_h)`. One call, whole pair.
- Frame M (focus leaves H): same on the H side; `_apply_canvas_size(..., (new_w, new_h))`. Second
  call, replacing the first.

Two calls, each carrying the whole pair, each individually correct. Defensible and stated. The one
consequence the spec should add to manual verification item 1 is that the canvas visibly resizes
once on the tab-out of W (item 1 already says "the canvas resizes to 1024 wide on that frame"), so
the reviewer knows the intermediate resize is intended, not a bug.

Enter: the spec's reasoning that `is_item_deactivated_after_edit()` already covers Enter because
Enter deactivates the item is correct for `input_int` without `enter_returns_true`. **But see F10
below on matching W-C.**

### F10. Enter — W-C's in-flight `_draw_name` uses BOTH `enter_returns_true` and the deactivate query; W-A's open question 1 should say why the pair differs

`git diff shaderbox/popups/pass_settings.py` shows W-C's landed shape:

```python
    committed, app.pass_settings_name_buf = imgui.input_text(
        "##pass_settings_name",
        app.pass_settings_name_buf,
        flags=imgui.InputTextFlags_.enter_returns_true,
    )
    # Read on the line after the input: the item-scoped queries see the LAST submitted item.
    deactivated = imgui.is_item_deactivated_after_edit()
    renamed = (
        _commit_pass_name(app, document_id, name)
        if committed or deactivated
        else False
    )
```

W-C also rewrote the imgui skill's § 7.5 to make that the rule: "commits on
`imgui.is_item_deactivated_after_edit()`, with `enter_returns_true` as the Enter shortcut and Esc
as cancel", and "Read the deactivate query on the line IMMEDIATELY after the `input_text` call,
before any `same_line`" (both in the working-tree diff of `.claude/skills/imgui-ui/SKILL.md`).

W-A's open question 1 takes the opposite default (no `enter_returns_true`) and argues it from the
semantics of deactivation. The argument is right for `input_int`: without the flag, Enter
deactivates, so the deactivate query fires. But the skill § 7.5 as W-C rewrites it now states the
OR of the two as the pattern, and a reader of the skill implementing this row will reach for both.
The spec's own note that W-A "inherits `_draw_name`'s new `bool` return ... without touching
either" shows it read W-C; it should read W-C's skill edit too.

**Fix (paste-ready).** In § Design decisions item 3's Enter paragraph and open question 1, name the
skill rule W-C lands (§ 7.5's `committed or deactivated`) and state the one-line reason the pair
diverges from it: `input_text` needs `enter_returns_true` because Enter in a text field does NOT
deactivate it, while `input_int` has no multiline behaviour and Enter deactivates, so the OR's
first term is dead code here. Alternatively, adopt the OR verbatim to match `_draw_name` and the
skill, at the cost of one flag. Either is fine; the spec must not be silent about the divergence
from a rule its sibling wave is writing this week.

### The clamp: one constant, both readers

**Correct as specified.** After the wave:

- UI path: `tabs/document.py::_apply_canvas_size`, first line, `w, h = clamp_canvas_size(size)`
  (spec item 1's code block).
- Copilot path: `copilot/backend.py::set_canvas_size`, `w, h = clamp_canvas_size((width, height))`
  (spec item 2), replacing `copilot/backend.py:1079-1080`'s two `max(min(...))` expressions.

`_MIN_CANVAS_PX` / `_MAX_CANVAS_PX` at `copilot/backend.py:136-137` have exactly one reader today
(`grep` finds `:1079` and `:1080` only), so the move is clean. Both existing clamp tests drive the
copilot's public method — `tests/test_document_ops.py:33` (`(99999, 4)` → `(4096, 16)`) and
`tests/test_document_ops.py:128` (`(320, 99999)` → `(320, 4096)`) — so the move is verified without
touching a test, as the spec claims.

### The presets menu

- **`get_active_uniforms()` compiles a pass.** **Confirmed.** `core.py:225-231`:
  `if self.program is None and not self.compile_unit.error_raw: self.compile()`. The current combo
  calls it (`tabs/document.py:68`), and `conventions.md:200` (066 D1) names it as one of the three
  pullers. Doing it across all passes on every Document-tab frame would compile the graph on the
  tab's first frame.
- **The `uniform_values` + `is_default_image` route forces no compile.** **Confirmed.**
  `uniform_values` is a plain dict; `media.py::is_default_image` is a path comparison on an `Image`.
  Neither touches `program`. One caveat worth adding to the spec: `uniform_values` can also hold a
  `moderngl.Buffer` (uniform blocks, `core.py:330`) and a `Video` (a `MediaWithTexture`). The
  `isinstance(value, MediaWithTexture)` filter handles the buffer correctly and admits a bound
  video, which is right — a bound video's size is as much a user choice as an image's.
- **`RenderShape` / `shape_to_preset` / `resolve_dims` give integer dims for every non-NATIVE
  shape.** **Confirmed.** `shape_to_preset` returns `FIXED_ASPECT` with `aspect` and `longest_edge`
  for every shape whose `spec.aspect is not None` (`render_shape.py`), which is all six non-NATIVE
  members. `resolve_dims`' FIXED_ASPECT branch (`render_preset.py:46-55`) computes from
  `longest_edge` and `aspect` alone — `source_size` is untouched — and returns `_align(w), _align(h)`,
  so both are integers and even. The spec's claim that the canvas size passed in is inert is exact.
  The largest is 2560 (`SHORT_1440` / `WIDE_1440`), inside the 4096 clamp, as claimed.
- **512x512 is reachable from the menu.** **Confirmed by construction**:
  `_SQUARE_PRESETS = (256, 512, 1024, 2048)`, labelled via `get_resolution_str(None, 512, 512)`
  → `512x512 (1:1)` (`util.py:69-75`, `math.gcd(512,512) = 512`). W-H's "Before you start" dependency
  is met, and `test_the_square_presets_include_512` pins it mechanically.
- **`begin_combo` shape.** **Confirmed available**: `begin_combo(label, preview_value, flags=0)` and
  `ComboFlags_.no_arrow_button` (value 32, verified by running the installed bundle), and
  `selectable(label, p_selected, ...) -> Tuple[bool, bool]` so `[0]` is the clicked flag. The
  "same panel already owns this idiom" claim is true: `tabs/document.py:142` is the uniform-sort
  `begin_combo` with a fixed preview string and a `selectable` loop.

### The checkerboard

- **`image_with_bg`'s default `bg_col`.** **Confirmed.** The installed stub
  (`.venv/lib/python3.12/site-packages/imgui_bundle/imgui/__init__.pyi:1516-1528`) declares
  `bg_col: Optional[ImVec4Like] = None` with the documented binding default `ImVec4(0, 0, 0, 0)`.
  `ui.py:619-625` passes only `image_size`, `uv0`, `uv1`, `tint_col`. So the finding's stated
  mechanism ("the default (window) background") is wrong and the spec's correction is right:
  nothing is drawn behind the image, which is why a draw-list backdrop shows through unmodified.
- **The two new COLOR tokens follow `theme.py`'s pattern.** **Confirmed.** `_ColorBag`'s neutrals
  block (`theme.py:124-129`) is `BG_APP` / `BG_SURFACE` / `BG_POPUP` / `BG_FRAME` / `BORDER`, each
  `= _P[...]`, with no checkerboard pair — so the spec's "they do not exist yet" correction is
  right. `CHECKER_LIGHT = _P["bg_2"]` / `CHECKER_DARK = _P["bg_1"]` match the form exactly. The
  claim that they need no `SELECT`-invariant entry is correct: the two import-time assertions
  (`theme.py:196-206`) constrain `COLOR.SELECT` against accent primaries and `STATE_*` hues only;
  a background fill is outside both.
- **Nothing at the call site hard-codes a literal colour or size.** **Confirmed in the spec's code**:
  `_draw_canvas_backdrop` reads `SIZE.CHECKER_TILE`, `COLOR.CHECKER_LIGHT`, `COLOR.CHECKER_DARK`;
  the border reads `COLOR.BORDER`. `add_rect_filled` on the window draw list with a
  `color_convert_float4_to_u32(COLOR.*)` argument is the codebase's idiom
  (`ui_primitives.py:828`, `:846`, `:852`, `:898`, `widgets/cheatsheet.py:78`), and
  `thickness=1.0` is stated explicitly at `ui_primitives.py:818` and `:861`, as claimed.
- **Draw order.** Correct. `img_min = imgui.get_cursor_screen_pos()` is captured, the backdrop is
  appended to the window draw list, then `image_with_bg` appends the image after it and the border
  after that. Draw-list order is submission order, so backdrop under image under border. The cursor
  is untouched by draw-list calls, so `image_with_bg` still starts at `img_min`.
- **Cost.** The ~3700-rect estimate is the right order of magnitude and the "if it measures badly,
  a bigger tile" escape is the right posture. Not a finding.

### The three additions beyond the parent's bullets

Judged in F8 above: `UIDocument.save` and `_copilot_document_working_view` are the same defect class
as #2 and belong here; the copilot-turn gate is justified scope growth that should be labelled as
such. The persistence roster is unaffected — `tests/test_persistence_completeness.py`'s completeness
check builds its `loaders` set from modules containing the literal `"json.load"`, and `ui_models.py`
is already rostered; changing what `save` writes into `meta` touches neither the roster nor any
corruption case.

### The seven closed open questions

| # | Question | Agree? | Evidence |
|---|---|---|---|
| 1 | Enter via an explicit branch? Default: no | **Agree on the mechanism, disagree on the silence** | `input_int` without `enter_returns_true` deactivates on Enter, so the deactivate query fires. But W-C's landed `_draw_name` uses `committed or deactivated` and its skill § 7.5 rewrite makes the OR the rule. See F10 — the divergence must be named. |
| 2 | Offer a bound texture already equal to the canvas? Default: no | **Agree.** Selecting it is a no-op the early return swallows; a menu item with no action reads as broken. |
| 3 | Checkerboard behind the strip thumbnails too? Default: no | **Agree.** #21 names the viewer; `preview_cell` already draws its own frame and stale wash (`ui_primitives.py:818`, `:828`), and `SIZE.PASS_THUMB = 112` (`theme.py:244`) against a 12px tile is 9 cells across, which fights the thumbnail. |
| 4 | `_CANVAS_FIELD_W` / `_CANVAS_PRESETS_W` as module constants or `SIZE` tokens? Default: module constants | **Agree, narrowly.** The skill's "a token used by exactly one panel still belongs in the token bag" (`SKILL.md:227`) does point the other way, and the spec says so honestly. The deciding argument (derived arithmetic vs an independent choice) is the right tiebreak. But see F7: the arithmetic is only half-shown. |
| 5 | Should `_apply_canvas_size` save the document? Default: no | **Agree.** The copilot's `set_canvas_size` calls `_save_ui_document` (`backend.py:1086`) because a turn must leave disk consistent for its own reads; no other Document-tab edit saves. Saving on every deactivate would write the whole document dir, and `save` compiles every program-less pass first (`ui_models.py:368-370`), so the cost is worse than the spec says. Worth adding that sentence. |
| 6 | Gate the document-name input? Default: no, not this wave | **Disagree — required change.** See F4. |
| 7 | `ComboFlags_.height_large`? Default: no | **Agree.** 10 static entries plus one per bound texture; the default scroll is acceptable and the flag is one argument if a check says otherwise. |

### Tests: falsifier by falsifier

| Test | Falsifier goes red? |
|---|---|
| `test_a_ui_resize_moves_every_pass_together` | **Yes.** With `render_pass.canvas.set_size((w, h))` in place of the funnel, `doc.canvas_size` keeps `(64, 64)`, so `document.py:429-430`'s `entry.target.target_size(self.canvas_size)` re-derives `full` at 64 and `half` at 32 on every render. All three assertions fail. Mirrors `test_a_resize_moves_every_pass_together` (`tests/test_document_graph.py:489-518`), whose `_document` fixture and `_red` helper are indeed the right home. |
| `test_the_ui_resize_clamps_both_ends` | **Yes.** Without `clamp_canvas_size` in `_apply_canvas_size`, `doc.canvas_size` becomes `(99999, 4)`. Note the test as written must not render afterwards, or the 99999-wide allocation is the failure rather than the assertion — the spec says as much; make it explicit that the test asserts the field and does not render. |
| `test_an_unchanged_size_does_not_reallocate` | **No.** See F2 — `Canvas.set_size` already guards (`core.py:96`), so the test is green with or without the early return. |
| `test_the_square_presets_include_512` | **Yes.** Dropping `512` from `_SQUARE_PRESETS` removes the tuple entry. Pins W-H's dependency mechanically, which is the point. |
| `test_every_preset_survives_the_clamp` | **Yes**, for a preset outside `[16, 4096]`. The stated way to demonstrate it ("add a `4096`-square preset (8192 is out of range)") is garbled — 4096 is IN range and would not go red. The demonstration is an 8192 square. Correct the parenthetical. |
| `test_the_video_shapes_come_from_the_shape_table` | **Partially.** See F6 — the membership half falsifies, the single-homing half does not until dims are asserted. |
| `test_a_bound_texture_is_offered_and_the_default_image_is_not` | **Yes, both.** Without the `is_default_image` filter, the seeded `Image(DEFAULT_IMAGE_FILE_PATH)` (`core.py:332`) shows up and the first assertion fails. With the scan reading only `document.render_pass`, a texture bound on a non-output pass is invisible and the second fails. |
| `test_building_the_presets_compiles_nothing` | **Yes.** `get_active_uniforms` sets `program` via `compile()` (`core.py:230`), so a `get_active_uniforms`-based scan leaves every pass with a non-`None` program and the post-call assertions fail. This is the test that pins 066 D1 against the wave, and it is the strongest test in the set. |
| `test_no_preset_duplicates_the_current_size` | **Yes.** Without the skip the entry appears; `_canvas_presets` is pure, so the assertion is direct. |

The three "manual, and here is why" items (disabled-slider behaviour, commit-on-deactivate, the
checkerboard) are correctly reasoned: `/imgui-ui § 0` does warn that focus/interaction state reads
differently headless, and manual verification items 4, 1-2 and 6-7 cover them.

### W-C collision check

The spec's § Verified row says the collision is **Refuted**. Re-checked against W-C's actual
uncommitted diff:

- `popups/pass_settings.py`: W-C changes `draw_pass_settings`' close branch, `_draw_body`'s
  structure, `_draw_name`, and adds `_commit_pass_name`. It does not touch `_draw_target`, which is
  W-A's only edit there. **No collision.** W-A's citation of `is_output` and the slider is now at
  `:177` and `:194-200` rather than `:161` and `:181-187`; the spec cites symbols, so it survives.
  Note the `size (w, h)` label is now `:193` and `canvas_w, canvas_h = document.canvas_size` is
  `:190`.
- `ui.py`: W-C changes the tick loop inside `update_and_draw` (adding the pending-pass target
  render); W-A changes `_draw_document_image`. **No collision.**
- `app.py`: W-C adds `close_pass_settings`, `open_pass_settings_for_panel_pass`, `open_add_pass` and
  two command bindings. W-A adds one field and one line to a document-changed handler. **No
  collision**, but F3 relocates W-A's line into `_on_current_document_changed` (`app.py:536`),
  which W-C does not touch either.
- `.claude/skills/imgui-ui/SKILL.md` § 7.5: W-C rewrites it. **This is a real interaction** — see
  F10. Not a merge collision, a rule-consistency one.

---

## False trails (checked, nothing there)

- `_apply_canvas_size`'s notification budget: `Canvas: 1080x1080` is verbatim the skill § 2 table's
  own notification example (`SKILL.md`, the word-budget table). Fine.
- `pass_graph.py` as the clamp's home creating an import cycle: it imports nothing from `document`,
  `core`, or imgui; `tabs/document.py` and `copilot/backend.py` both already reach it transitively.
  Fine.
- `_canvas_presets` reading `MediaWithTexture` catching a `moderngl.Buffer` from a uniform block:
  a `Buffer` is not a `MediaWithTexture`, so the `isinstance` filter excludes it. Fine.
- `input_int2` rejection: the stub confirms `input_int2(label, v: List[int], flags) -> Tuple[bool, List[int]]`
  — one item, one changed flag, so the spec's reason for rejecting it is exact. Fine.
- `SIZE.PASS_SETTINGS_H` ownership between W-A and W-B: W-A touches neither `theme.py:259` nor
  `:260`. Fine.
- `ui.py:261` (`adjust_size(...render_pass.canvas.texture.size, width=SIZE.PREVIEW_W)`) and
  `ui.py:612` (`image_aspect`) left reading the texture: both are fit-the-texture questions, and
  after the funnel holds the two values agree. Correct calls, correctly explained.
- `copilot/backend.py:1812` (`_render_facts_for`) left reading the texture: its own comment ties it
  to matching the preview's aspect, and the preview reads the texture. Correct call.
- `document.py` listed as no-change: `set_canvas_size` (`:284-293`) writes the field and resizes the
  output canvas, which is everything both callers need. Correct.

---

## Coverage statement

**Read end to end:** the wave spec (all 893 lines); `01_spec.md`; `00_findings.md` rows #1 #2 #3 #4
#21 and the header; `shaderbox/tabs/document.py` (draw + `_draw_entry_points`);
`shaderbox/render_shape.py`; `shaderbox/render_preset.py::resolve_dims`; the imgui skill §§ 0-3 and
the W-C-modified § 7.5; `tests/test_persistence_completeness.py`;
`tests/test_document_ops.py`'s two clamp tests.

**Read in the region under review:** `shaderbox/document.py` (`set_canvas_size`, `render`'s fixup,
`load_from_dir`'s two `canvas_size` reads); `shaderbox/core.py` (`Canvas.set_size`,
`get_active_uniforms`, `seed_uniform_values`, `_default_uniform_value`, the sampler bind);
`shaderbox/popups/pass_settings.py::_draw_target`; `shaderbox/ui.py::_draw_document_image` and the
tick loop; `shaderbox/theme.py` (`_P`, `_ColorBag`, the two invariants, the SIZE and SPACE bags);
`shaderbox/copilot/backend.py` (the clamp constants, `set_canvas_size`,
`_copilot_document_working_view`); `shaderbox/ui_models.py::UIDocument.save`;
`shaderbox/media.py::is_default_image`; `shaderbox/util.py::get_resolution_str`;
`shaderbox/app.py` (`set_current_document_id`, `_on_current_document_changed`, the transient
declarations at `:286-305`); `shaderbox/project_session.py::set_current_document_id`;
`tests/test_document_graph.py` (fixture, the two resize tests);
`.venv/lib/python3.12/site-packages/imgui_bundle/imgui/__init__.pyi`
(`image_with_bg`, `input_int`, `input_int2`, `begin_combo`, `selectable`, `add_rect`, `ComboFlags_`),
plus a live check of `ComboFlags_.no_arrow_button` in the installed bundle.

**Skipped, and why:** `shaderbox/widgets/uniform.py`, `document_grid.py`, `pass_list.py` beyond
their `begin_disabled` lines — the spec cites them only as gate precedents, and the grep confirms
the precedent. `exporters/*` — outside W-A's blast radius; opened only to count `begin_disabled`
sites. `ai_docs/features/069_tutorial_walk_findings/10_wave_c_pass_verbs.md` — W-C's own spec is
not this review's anchor; W-C's *code* diff is, and that was read in full for the three shared
files. The 32 findings other than #1 #2 #3 #4 #21.

---

# Round 2 (closure)

Narrow round against the revised `20_wave_a_canvas_viewer.md` (1056 lines). Scope: does each of
F1..F10 have text in the spec that closes it. Two rulings went further than round 1 proposed (F3
deletes the reset entirely; F10 adopts the OR verbatim) and are re-derived rather than accepted on
the ledger's word. Tree re-checked at `a246a19`, which now carries W-C; the working tree holds only
`popups/lib_picker/tree.py` and `tests/test_lib_files.py`, neither in W-A's blast radius.

## Per-finding verdict

| # | Verdict | Closing text | Reason |
|---|---|---|---|
| F1 | **CLOSED** | Item 3's code block (`if not app.canvas_size_editing: app.canvas_size_buf = ui_document.document.canvas_size`), the six-row frame table, and the `App` fields row in § Files touched | The latch is gone; re-traced below against all three inputs. |
| F2 | **CLOSED** | Item 1, "The early return on an unchanged size suppresses a spurious NOTIFICATION, not a reallocation", and `test_an_unchanged_size_pushes_no_notification` | The real reason is stated and cites `Canvas.set_size`'s own guard; the replacement test's falsifier (recorder holds one entry) fires under its named bug. |
| F3 | **NOT CLOSED** | Item 3's third bullet and the premise row at line 972 | The line-number and forwarder halves are corrected. The further ruling rests on "switching documents cannot happen while a canvas field is active", which is false: Ctrl+N is `CommandScope.GLOBAL` and `route_flag` routes it `route_global` while an input is active, by that function's own comment. See below. |
| F4 | **CLOSED** | Item 4a, "ONE `begin_disabled` pair wraps the document-name input, the `W x H` pair, the `x`, and the presets dropdown"; manual item 12 naming the DOCUMENT-NAME field; open question 6 deleted | Gate lands in W-A on the whole row, with the `rename_document` race named as the reason. |
| F5 | **CLOSED** | Item 1's notification paragraph (folded into F2) | The click-away-that-changed-nothing case is the stated reason for the return, so the fix cannot be lost by resolving F2 the other way. |
| F6 | **CLOSED** | `test_the_video_shapes_come_from_the_shape_table`'s "**And asserts the DIMS**" block with `resolve_dims(shape_to_preset(...), (1, 1))` | The single-homing half now falsifies: a hand-rolled literal that drifts from `SHAPE_TABLE`'s `longest_edge` goes red. The `(1, 1)` source size doubles as the source-size-independence assertion, which is better than round 1 asked for. |
| F7 | **CLOSED** | Item 3's "Layout, and the arithmetic that fixes both numbers" budget block; `_CANVAS_PRESETS_W: float = 64.0` in item 4 | Both numbers fixed, the sum shown (56+4+7+4+56+8+64 = 199 <= 200), and which number gives is stated. |
| F8 | **CLOSED** | § Findings folded, "One change in this wave closes no finding: the copilot-turn gate on the first row", plus the paragraph naming `UIDocument.save` and `_copilot_document_working_view` as same-class-as-#2 | A later reader is told which change has no finding behind it. |
| F9 | **CLOSED** | Item 6's "(20 call sites, e.g. ...)" and the premise row at 965; the premise row at 972 carrying `:1043` | Both counts corrected; I re-ran `grep -rn begin_disabled shaderbox/ \| wc -l` -> 20 and confirmed `set_current_document_id` at `app.py:1043`. |
| F10 | **CLOSED** | Item 3's "Enter is `enter_returns_true` OR-ed with the deactivate query, verbatim as the skill states it", and the code block's `flags=imgui.InputTextFlags_.enter_returns_true` on both fields | Matches W-C's landed `_draw_name` (`committed or deactivated`) and § 7.5 at `a246a19` token for token, and states that the OR's first term is redundant for `input_int` so the next reader does not delete it as dead. |

## F1 re-trace against the three inputs

The mechanism to hold in mind: `canvas_size_editing` is written at the END of the row, and the
mirror is read at the TOP, so the mirror acts on the PREVIOUS frame's verdict. The commit branch
also runs at the end and re-reads the buffer from the document itself.

**(a) The copilot sets 800x600 mid-turn while no field is active. CLOSED.** During a turn the row
sits inside `begin_disabled(app.copilot_turn_active)` (item 4a), so `is_item_active()` reads False
on both fields and `canvas_size_editing` stays False. `CopilotBackend.set_canvas_size` writes
`document.canvas_size` on the main thread through the bridge (`backend.py::set_canvas_size`). The
next frame's mirror reads `canvas_size_editing == False` and assigns the new pair before either
`input_int` is submitted, so the fields read `800` / `600` on that frame. Manual item 11's "while
the turn is still running" is now true, and item 4a's closing sentence ("already on screen while
the gate is still up") is the correct strengthening of what round 1 flagged as false.

**(b) The user has typed "10" into W (field active) when a disk sync replaces the Document. WORKS
AS SPECIFIED, with one consequence the spec does not state.** `sync_documents_from_disk`
(`project_session.py:445-490`) puts a `changed` document through `_load_one_document_from_disk`,
which replaces the `Document` object. With W active, `canvas_size_editing` is True, so the mirror
correctly does NOT run and the typed `10` survives -- which is manual item 13's requirement. The
consequence: the buffer's H half still holds the PRE-sync height, so the tab-out commits
`(10, stale_h)` and reverts the externally-set height. This is inherent to any pending-edit buffer
carrying both halves of a pair, it is exactly the trade the spec makes deliberately in item 3's
"Both fields commit as ONE `set_canvas_size` call" bullet, and its trigger is a hand edit of
`document.json` landing inside a typing window -- a path D2 declassifies as a workflow. Not a
defect; listed under false trails.

**(c) The user tabs from W to H. No steal, and no frame where the mirror can act. CLOSED.** Within
the deactivate frame, in submission order:

1. Top of the row: mirror reads the PREVIOUS frame's `canvas_size_editing`, which was True (W was
   active last frame), so the buffer is NOT overwritten. The half-typed width survives into this
   frame's `input_int`.
2. W's `input_int` returns `buf_w = new_w`; `is_item_active()` is False (focus left); the
   deactivate query fires, so `committed_w` is True; buffer becomes `(new_w, old_h)`.
3. H's `input_int` returns `buf_h = old_h` (H was just focused, nothing typed); `active_h` True.
4. `app.canvas_size_editing = active_w or active_h` -> True.
5. `committed_w` fires `_apply_canvas_size(app, ui_document, (new_w, old_h))`, then
   `app.canvas_size_buf = ui_document.document.canvas_size`.

The re-read at step 5 does exactly what the mirror would have done, in the same frame, so there is
no window in which a half-committed pair is either stolen or stale. And on the NEXT frame the
mirror is skipped anyway (`canvas_size_editing` is True from step 4), which is right: H is active.
The ordering that makes this work -- editing flag written after both fields, mirror read before
them -- is stated in the spec ("it is written at the END of the row, so the mirror check at the TOP
of the next frame reads the previous frame's verdict"), so it is a property of the design rather
than an accident of the sample code.

## F3: the claim that a document switch cannot occur while a field is active is false

**Claim under test.** Item 3, third bullet: "switching documents cannot happen while a canvas field
is active (the click that switches is the click that deactivates it), so the very next frame reads
`canvas_size_editing == False` and overwrites the buffer with the new document's size before any
`input_int` reads it." The premise row at line 972 repeats it.

The click half is true. The keyboard half is not.

**Evidence.** `shaderbox/commands.py:108`:

```python
        CommandId.NEW_DOCUMENT, "New document", _chord(K.n, K.mod_ctrl), C.DOCUMENT
```

`CommandSpec.scope` defaults to `CommandScope.GLOBAL` (`commands.py:80`), and `NEW_DOCUMENT`
declares no scope, so it is GLOBAL. `shaderbox/app.py:483-485` binds it to
`create_document_from_example(STARTER_EXAMPLE_ID)`, whose body ends
`self.set_current_document_id(new_document.id)` (`app.py:1665`) -- a document switch.

`route_flag` (`commands.py:233-245`) states the routing fact against itself:

```python
    # ... EXCEPT: an active text input owns all keyboard
    # keys and imgui routes only Ctrl-chords through it -- an Alt-chord (which can never type a
    # character) must route ALWAYS or it is dead while any input is active.
    if chord & int(imgui.Key.mod_alt):
        return imgui.InputFlags_.route_always
    return imgui.InputFlags_.route_global
```

Ctrl+N is a Ctrl-chord, and the comment's own words are that imgui routes Ctrl-chords through an
active input. The dispatcher's gate (`hotkeys.py::spec_eligible`, `:252-262`) checks the consumed-
chord set, the EDITOR / COPILOT focus flags and an open modal -- there is no "an input is active"
term, and a GLOBAL spec passes all three while a canvas field holds focus.

The copilot path the coordinator asks about is closed for a different reason:
`CopilotBackend.switch_document` (`backend.py:1026-1040`) calls `_set_current_document_id`, but a
turn is running whenever it can, and item 4a's gate makes both fields inert for the whole turn, so
`canvas_size_editing` is False throughout. The command-palette path is the same shape as Ctrl+N
(the palette is non-modal, `app.py:330-333`) and inherits the same answer.

**What actually happens under the bug.** Type `10` into W, press Ctrl+N. The command fires,
`ui_documents` gains a document and `current_document_id` moves. `canvas_size_editing` is still
True from the previous frame, so the next frame's mirror is skipped and the fields show `10` and
the OLD document's height against the NEW document. If focus then leaves W -- which the switch does
not itself do -- `_apply_canvas_size` commits that pair to the NEW document. It is narrow (it needs
the chord pressed with a field focused) and self-corrects the frame after focus leaves, but it is
one document's half-typed width landing on another, which is the exact failure the drafted reset
was written to prevent.

**Fix, two options; either closes it.**

> **(i) Keep the reset, at the right handler.** In `App._on_current_document_changed`
> (`app.py:536`, which already clears `editor_was_ever_focused` and fires only when the id actually
> changes), add `self.canvas_size_editing = False`. One line, no new mechanism: it re-arms the
> mirror, which then overwrites the buffer from the new document on the next frame. Restore
> `shaderbox/app.py` to § Files touched as a handler edit.

> **(ii) Keep no reset, and correct the claim.** Replace "switching documents cannot happen while a
> canvas field is active (the click that switches is the click that deactivates it)" with the
> narrower true statement: a MOUSE document switch cannot happen while a field is active, and the
> keyboard paths (Ctrl+N per `commands.py:108`, GLOBAL and `route_global`; the palette) can. Then
> say why the wave accepts it: the stale pair is displayed for as long as focus stays in the field
> and is corrected on the first non-editing frame, and a commit against the new document requires
> the user to leave the field having edited it.

Option (i) is the smaller diff and removes the case rather than documenting it. Whichever is taken,
the sentence as it stands asserts something the command registry contradicts, and the premise row
at line 972 asserts it a second time.

## False trails (round 2)

- **Case (b)'s stale H half.** Real, inherent to a both-halves pending buffer, deliberate per item
  3's one-call bullet, and reachable only by a hand edit of `document.json` inside a typing window.
  A preference, not a defect.
- `canvas_size_buf`'s `(0, 0)` initial value: item 3 states the first frame is non-editing by
  definition and overwrites it before any `input_int` reads it. Traced and true -- `canvas_size_editing`
  starts `False`, so the mirror runs on frame 1.
- The OR's first term being dead for `input_int`: the spec says so itself and gives the reason for
  writing it anyway. Correct on both halves; `input_int` without the flag does deactivate on Enter.
- W-C collision: re-checked at `a246a19`. `_draw_target` is untouched by W-C, `_draw_document_image`
  is untouched, and `app.py`'s W-C additions (`close_pass_settings`,
  `open_pass_settings_for_panel_pass`, `open_add_pass`) do not overlap the two fields W-A adds --
  nor `_on_current_document_changed`, if F3 is closed by option (i).
- The row budget's `~7` for the `x` glyph: an estimate, flagged as one in the spec, and the sum has
  1px of slack against 200. Fine.
