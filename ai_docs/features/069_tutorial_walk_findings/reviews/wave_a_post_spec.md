# 069 W-A — post-implementation review: spec fidelity and architecture

Reviewer role: `dev_flow.md` step 6, spec-fidelity and architecture. Commit under review:
`78bd1bf` ("069 W-A: canvas size fields, presets, viewer backdrop"). Anchors:
`20_wave_a_canvas_viewer.md` (the wave spec), `01_spec.md § W-A` and `§ Locked decisions D1 D2
D11` (the parent), `00_findings.md` #1 #2 #3 #4 #21 (the maintainer's words),
`.claude/skills/imgui-ui/SKILL.md § 1 § 2 § 6 § 7.5`, `ai_docs/conventions.md`, `CLAUDE.md`.

## Verdict

| Dimension | Verdict |
|---|---|
| Wave-spec fidelity | **PARTIAL** — items 1, 2, 3, 4, 4a, 6, 7 and 8's changed rows landed as written. **Item 5's "the current size is not offered" holds for the square and texture groups and NOT for the video-shape group** (finding 1), and item 8's "seven sites, enumerated from a grep" is a count of half the grep's output (finding 4). |
| Parent fidelity | **PASS** — every W-A bullet is satisfied by a cited line; D1, D2 and D11 hold. |
| Findings closure | **PASS** — #1 #2 #3 #4 #21 each closed by a line below; the maintainer's own repro for #1/#3/#4 now produces the right picture. #21's second half (the border) is weakened on a fully transparent output by a colour collision the spec itself prescribed (finding 3). |
| Architecture | **PARTIAL** — `clamp_canvas_size`'s home, `_draw_canvas_backdrop`'s home, the `App` state shape and the `begin_combo` idiom are all right and are argued below. **`SIZE.RES_COMBO_W` is now a token with no code reader** (finding 2). |
| Docs | **PARTIAL** — `help_content.py` and `copilot/prompt.py` carry no stale canvas text, `conventions.md`'s funnel law now describes a true single-writer. **`pass_graph.py`'s own docstring and `dev_flow.md`'s module-map entry both enumerate the module's contents and neither mentions the canvas clamp** (finding 5). |
| D1 budget | **PASS** — every new user-facing string is inside budget, and the one new site W-B's AST gate will actually measure is 1 word. Counts below. |

`make gates` re-run at this commit: **exit 0, check passed / test passed / smoke passed** (captured
unpiped to a file). Working tree clean, so the commit carries no formatter drift.

## Coverage table — wave-spec design decisions

| # | Decision | Status | Evidence |
|---|---|---|---|
| 1 | `_apply_canvas_size` free function; clamp, early return, `set_canvas_size`, notification | **landed as written** | `tabs/document.py:40-47`, verbatim the spec's block. Sole writer of the document's size from the UI; both callers at `:172` and `:182` |
| 1 | `UIDocument.save` persists `document.canvas_size` | **landed as written** | `ui_models.py:360` |
| 2 | `MIN_CANVAS_PX` / `MAX_CANVAS_PX` / `clamp_canvas_size` move to `pass_graph.py`; `backend.py` loses its arithmetic | **landed as written** | `pass_graph.py:47-56`; `copilot/backend.py:1077` (`w, h = clamp_canvas_size((width, height))`), import at `:102`. `grep -rn "_MAX_CANVAS\|_MIN_CANVAS"` over `shaderbox/` and `tests/` returns only the new `pass_graph.py` names — no copy survives |
| 3 | `W x H` `input_int` pair, `step=0`, `enter_returns_true` OR-ed with the deactivate query read on the next line, one commit carrying the whole pair, buffer re-read after commit | **landed as written** | `tabs/document.py:143-173`, token for token against the spec block. `is_item_active()` at `:150` / `:167` and `is_item_deactivated_after_edit()` at `:151` / `:168` are each on the line after their `input_int` and above the `same_line` at `:154` / `:175` |
| 3 | The buffer mirrors the document on every non-editing frame; two `App` fields | **landed as written** | `app.py:290-291` (`canvas_size_buf: tuple[int, int] = (0, 0)`, `canvas_size_editing: bool = False`); the mirror at `tabs/document.py:140-141`, the verdict written at `:169` |
| 3 | `_on_current_document_changed` clears the editing flag | **landed as written** | `app.py:549`, beside `editor_was_ever_focused = False` at `:546` |
| 3 | `_CANVAS_FIELD_W` 56, `_CANVAS_PRESETS_W` 64, module constants | **landed as written** | `tabs/document.py:34-35`. See finding 2 on what they are measured against |
| 4 | `begin_combo` + `selectable` presets, fixed `presets` preview, `no_arrow_button`, buffer re-synced after a pick | **landed as written** | `tabs/document.py:176-185`. Matches the panel's own sort-combo idiom at `:213-220` (same `begin_combo` + `selectable` loop + fixed preview string) |
| 4a | ONE `begin_disabled(app.copilot_turn_active)` wrapping captions, name input, pair, `x` and dropdown | **landed as written** | opens `tabs/document.py:127` before the first `small_caption` at `:129`, closes `:186` after `end_combo`. Matches the sibling gate at `:298`/`:326` |
| 5 | Squares from `_SQUARE_PRESETS`, labelled by `get_resolution_str` | **landed as written** | `tabs/document.py:37`, `:61-66` |
| 5 | Video shapes from `MENU_SHAPES` via `shape_to_preset` + `resolve_dims`, `NATIVE` skipped | **landed as written** | `tabs/document.py:68-78` |
| 5 | Texture group over ALL passes, `uniform_values` not `get_active_uniforms()`, `MediaWithTexture` + `is_default_image` filters | **landed as written** | `tabs/document.py:80-90` |
| 5 | "Duplicates collapse, and the current size is not offered" | **landed with an unreported deviation** — the video-shape loop appends unconditionally and never consults `seen`, so a canvas at a video-shape size gets that shape offered as a dead item | `tabs/document.py:77` appends with no `if size in seen` guard, unlike `:62-64` and `:85-86`. **Finding 1** |
| 6 | `begin_disabled(is_output)` around the scale slider only; `label_row` and `help_marker` outside; help text untouched | **landed as written** | `popups/pass_settings.py:195` / `:203`, with `label_row` at `:192` and `help_marker` at `:206` both outside. `git show` on the file shows two added lines and no string change |
| 7 | `_draw_canvas_backdrop` free function in `ui.py`, called before `image_with_bg`; 1px `COLOR.BORDER` rect after | **landed as written** | `ui.py:590-616` (the function, verbatim including the per-tile `min(...)` clip), `:649` (the call, on the line after `img_min` at `:648`), `:657-662` (the border, after the image) |
| 7 | `COLOR.CHECKER_LIGHT` / `CHECKER_DARK` from `_P`, `SIZE.CHECKER_TILE = 12` | **landed as written** | `theme.py:134-135`, `:257`. Both map to `_P` entries, no literal at the call site |
| 8 | `UIDocument.save` and `_copilot_document_working_view` changed; the fit-the-texture reads left alone | **landed, with the enumeration overstated** | `ui_models.py:360`, `copilot/backend.py:726`. The four "unchanged deliberately" rows are unchanged. But the grep the table claims to enumerate returns 14 sites at `78bd1bf^`, not 7. **Finding 4** |

## Coverage table — tests

| Test | Status | Evidence |
|---|---|---|
| `test_document_graph.py::test_a_ui_resize_moves_every_pass_together` | **landed as written** | `tests/test_document_graph.py:637-662`; three-pass graph, `_apply_canvas_size` through a stub, all three assertions present |
| `test_document_graph.py::test_the_ui_resize_clamps_both_ends` | **landed as written, with a reported deviation in HOW it falsifies** | `tests/test_document_graph.py:667-673`. Demonstrated: stubbing `clamp_canvas_size` to the identity and running the test raises `_moderngl.Error: the framebuffer is not complete (INCOMPLETE_ATTACHMENT)` inside `_apply_canvas_size`, not an assertion failure. **Finding 6** on the spec text and the inline comment that both claim otherwise |
| `test_document_graph.py::test_an_unchanged_size_pushes_no_notification` | **landed as written** | `tests/test_document_graph.py:676-686`; asserts the recorder is empty, which is the version F2 replaced the green-either-way one with |
| `test_canvas_presets.py::test_the_square_presets_include_512` | **landed as written** | `tests/test_canvas_presets.py:93-100`; the exact tuple `("512x512 (1:1)", (512, 512))` plus the other three |
| `test_canvas_presets.py::test_every_preset_survives_the_clamp` | **landed as written** | `tests/test_canvas_presets.py:103-108` |
| `test_canvas_presets.py::test_the_video_shapes_come_from_the_shape_table` | **landed as written** | `tests/test_canvas_presets.py:111-133`; dims recomputed against `resolve_dims(shape_to_preset(...), (1, 1))` at `:123-127`, which is F6's addition |
| `test_canvas_presets.py::test_a_bound_texture_is_offered_and_the_default_image_is_not` | **landed as written** | `tests/test_canvas_presets.py:136-159`; both halves, the bind on the non-output `src` pass at `:154` |
| `test_canvas_presets.py::test_building_the_presets_compiles_nothing` | **landed as written** | `tests/test_canvas_presets.py:162-175` |
| `test_canvas_presets.py::test_no_preset_duplicates_the_current_size` | **landed, and it does not cover the case that is broken** | `tests/test_canvas_presets.py:178-182` sets the canvas to `(512, 512)`, a SQUARE, so it exercises the one group whose guard is present. **Finding 1** |
| The three manual-only items (disabled slider, commit-on-deactivate, checkerboard) | **as specced** — each has a stated reason and a manual step; nothing was silently converted to a headless assertion | spec `§ Tests`, last block |

Both test files pass: `uv run pytest tests/test_canvas_presets.py tests/test_document_graph.py` →
34 passed.

## Parent fidelity — `01_spec.md § W-A` bullet by bullet

| Parent bullet | Satisfied by |
|---|---|
| "The Document tab's Resolution combo routes through `Document.set_canvas_size`" | `tabs/document.py:46`. `grep -rn "canvas.set_size" shaderbox/` leaves no UI-path caller: `document.py:293` (inside the funnel), `:390` and `:439` (the render fixups), `ui.py:264` (the preview canvas), `backend.py:1814` (the probe). `Document.set_canvas_size` is now the single writer of `document.canvas_size` — `grep -rn "canvas_size ="` over `shaderbox/` returns only `document.py:292` |
| "Test: after a UI resize, every non-output pass's `target_size(document.canvas_size)` follows on the next render" | `tests/test_document_graph.py:637-662` |
| "a `W × H` pair of `input_int`s showing `document.canvas_size` (commit on deactivate-after-edit, D11; clamped to the copilot's existing range, moved to one shared constant)" | `tabs/document.py:143-173`; `pass_graph.py:47-56` |
| "plus a presets menu beside it (squares 256 / 512 / 1024 / 2048, the named video shapes, and any bound texture's size) that writes the pair" | `tabs/document.py:176-185` + `:50-92`. All three groups present |
| "Both paths call `Document.set_canvas_size`" | both call `_apply_canvas_size` (`:172`, `:182`), which is the only caller of `set_canvas_size` outside the copilot |
| "Gear: the output pass's size slider is disabled with the help text it already has (#4)" | `popups/pass_settings.py:195-203`; the `help_marker` at `:206-214` is byte-identical to `78bd1bf^` |
| "Viewer: checkerboard behind the preview + 1px border (#21). Two greys from `theme.py`, no literal colours at the call site" | `ui.py:590-616`, `:657-662`; `theme.py:134-135`. No literal in `ui.py` — both colours arrive as `COLOR.*` tokens |
| Files list | Every named file is in the diff, and no file outside it |

**D1 (word budget).** Every new user-facing string, counted:

| String | Site | Words | Budget |
|---|---|---|---|
| `"Canvas"` (caption) | `tabs/document.py:131` | 1 | label 1-2 — **in** |
| `"x"` (separator) | `tabs/document.py:155` | 1 | this is the one new site W-B's AST gate measures (`text_colored` with a `COLOR.FG_DIM` first arg); ≤ 4 — **in** |
| `f"Canvas: {w}x{h}"` (notification) | `tabs/document.py:47` | one clause; the skill § 2 table's own example is `Canvas: 1080x1080` — **in** |
| `"presets"` (combo preview) | `tabs/document.py:178` | 1 | — **in** |
| Preset item labels (`512x512 (1:1)`, `Wide 1080p (16:9)`, `1234x567 (2:1) - u_bound`) | built by `get_resolution_str` / `SHAPE_TABLE[...].menu_label` | 1-3 | menu ITEMS are not in D1's table and not in W-B's gate census (`help_marker`, `set_tooltip`, `separator_text`, `label_row`, `row_label`, `text_colored`+`FG_DIM`); the shape labels are single-homed in `render_shape.py` and shared with the Share tab, so cutting them here would fork the vocabulary — **no overflow, nothing for W-B to refuse** |

The deletion side of D1 also lands: `resolution_label`'s `f"Resolution ({current_name})"` — a
derived value in a label, which D1 forbids — is gone with the combo.

**D2 (no manual JSON editing).** 512x512 is now reachable by two UI paths, and any size in
`[16, 4096]` by a third. `tests/test_canvas_presets.py:96` pins the preset entry.

**D11 (inline inputs commit on deactivate-after-edit, Enter a shortcut).** `tabs/document.py:151`
and `:168`. The skill § 7.5 rule about reading the query on the immediately following line is
honoured at both. `step=0` is load-bearing beyond the visual argument the spec gives: with
`step > 0` imgui's `InputScalar` submits the `+`/`-` buttons after the input, so the last submitted
item — the one `is_item_active()` and `is_item_deactivated_after_edit()` read — would be a button,
not the field. The spec justified `step=0` only by button count and per-keystroke commits; the
stronger reason is that the D11 pattern does not work without it.

## Findings closure

| # | The maintainer's words | Closed by | Would the complaint recur? |
|---|---|---|---|
| #1 | "set the canvas to 512×512 in the Document tab — how? there is no 512×512 option, where did you find it?" | `tabs/document.py:37` (`512` in `_SQUARE_PRESETS`) and `:143-173` (free-form entry). Pinned by `tests/test_canvas_presets.py:96` | **No.** Scenario: open the Document tab, click `presets`, `512x512 (1:1)` is the second item; or type `512` `512` into the pair. Both land 512x512 |
| #2 | (agent) "The Document-tab combo resizes via `render_pass.canvas.set_size()` directly … bypassing `Document.set_canvas_size`" | `tabs/document.py:46`; the direct call is deleted. Pinned by `tests/test_document_graph.py:637-662` | **No.** Scenario: three-pass document, resize in the UI, next render — `half` at scale 0.5 is at half the NEW canvas (asserted). The save-path half is closed too: `ui_models.py:360` now persists the field, so the bug no longer "hides across restarts" |
| #3 | "I tried to change the canvas to 1080x1080, then I clicked the settings of the first pass and it still has 1280x960 (100%) -- wtf?" | Same funnel. The gear reads `document.canvas_size` at `popups/pass_settings.py:188`, which the UI path now writes | **No.** Scenario, verbatim: set 1080x1080 in the pair, open the first pass's gear — `size (1080, 1080)` at 100%, because both the writer and the reader are now the same field |
| #4 | "is it even possible to have the first pass (or the last pass actually) at a resolution different from the canvas?" — plus "A control that does nothing is a UX defect; either disable it for the output pass or hide the row" | `popups/pass_settings.py:195` / `:203` | **No.** Scenario: open the OUTPUT pass's gear — the percent slider is greyed and inert, while the `(?)` beside it stays full-contrast and still explains that the output always draws at full size. The intermediate-pass answer ("yes, smaller") is unchanged and still live |
| #21 | "when the canvas is transparent (a texture with alpha = 0) we can't see the canvas boundary — it blends with the background" | `ui.py:649` (checkerboard) and `:657-662` (border) | **Mostly no.** Scenario: a shader writing `vec4(0.0)` now shows a 12px grey checkerboard filling exactly the image rect, so the extent reads. The finding's SECOND half (the border, for an opaque dark output) works against the panel — `COLOR.BORDER` is `bg_2`, the panel is `bg_0` — but is invisible against the checkerboard's own light squares, which are also `bg_2`. **Finding 3** |

The external anchor for #4 — that the renderer has always ignored the output's `scale` — is
`document.py:459`'s `if name != output:` guard, unchanged by this wave, so the disable matches the
engine rather than asserting a new rule.

## The implementer's three reported deviations

- **(a) `_canvas_presets` lost its `app` parameter.** Confirmed fully clean.
  `tabs/document.py:50` is `def _canvas_presets(ui_document: UIDocument)`, the call at `:180`
  passes one argument, and the spec text at `20_wave_a_canvas_viewer.md:458` and `:389` was
  updated to match. `grep -rn "_ = app\|_ = ui_document\|del app\b" shaderbox/` returns nothing —
  no dead parameter and no suppression anywhere in the package.
- **(b) `test_the_ui_resize_clamps_both_ends` fails inside `_apply_canvas_size`.** Confirmed, and
  the test is still unambiguous — but the spec and the test comment are now wrong about why.
  **Finding 6.**
- **(c) The unused `GL_SAMPLER_2D` import removed.** Confirmed:
  `grep -rn "GL_SAMPLER_2D" shaderbox/` no longer lists `tabs/document.py`, and the eight surviving
  sites all use it. Correct — the import existed only for the deleted `get_active_uniforms` scan.

## Findings

### 1. The presets menu offers a dead item whenever the canvas is at a video-shape size, and the test written to catch that only covers the squares

**Claim.** Wave-spec item 5 states the rule as "Duplicates collapse, and the current size is not
offered … picking it would be a no-op that `_apply_canvas_size`'s early return would silently
swallow, which reads as a dead menu item." Two of the three groups implement it; the video-shape
group does not.

**Evidence.** `tabs/document.py:61-66` (squares) and `:80-90` (textures) both open with
`if size in seen: continue`. The video loop at `:68-78` has no such guard — it appends at `:77` and
only then does `seen.add(size)` at `:78`. Run against the real function:

```
canvas (1920, 1088) -> preset entries equal to the current size: ['Wide 1080p (16:9)']
canvas (1280, 720)  -> preset entries equal to the current size: ['Wide 720p (16:9)']
canvas (512, 512)   -> preset entries equal to the current size: []
```

`tests/test_canvas_presets.py:180` sets the canvas to `(512, 512)` — a square — so the test asserts
the one group whose guard is present and passes over the one that is missing. `1920x1088` is not an
exotic size: it is what "Wide 1080p" itself sets, so the sequence "pick Wide 1080p, reopen the
menu" reaches the state in two clicks.

**Fix.** In `tabs/document.py`, guard the video loop the same way the other two are guarded: after
computing `size`, `if size in seen: continue` before the `presets.append`, keeping the `seen.add`.
Then change `tests/test_canvas_presets.py::test_no_preset_duplicates_the_current_size` to
parametrise over one size per group — a square (512, 512), a video shape (1920, 1088), and a bound
texture's size — so the assertion covers every group rather than the first one.

**State at the time of writing.** The finding is against the commit — `git show
78bd1bf:shaderbox/tabs/document.py` lines 68-78 have the unguarded append. While this review was
being written the guard appeared in the WORKING TREE (`if size in seen: continue` before the
`presets.append`), evidently from the parallel code review. The code fix is therefore already in
hand; what remains from this finding is the test half, which the working tree does not yet carry:
`test_no_preset_duplicates_the_current_size` still exercises only a square canvas, so the guard
that was just added is unpinned and can be removed again without going red.

### 2. `SIZE.RES_COMBO_W` is now a token nothing reads, and the row arithmetic it anchors is unenforced

**Claim.** The spec's item 3 says "`SIZE.RES_COMBO_W` is kept and is what the whole cluster is
measured against", and open question 3 justifies keeping `_CANVAS_FIELD_W` / `_CANVAS_PRESETS_W`
out of `theme.py` on the grounds that they are "arithmetic derived from `SIZE.RES_COMBO_W`". After
the wave, no code derives anything from it.

**Evidence.** `grep -rn "RES_COMBO_W" shaderbox/ tests/ scripts/ --include=*.py` returns three
lines: the definition at `theme.py:234`, and two COMMENT lines at `tabs/document.py:31` and `:33`.
The old `imgui.set_next_item_width(SIZE.RES_COMBO_W)` was deleted with the combo. Nothing
downstream reserves the row width either — `draw` flows the row with `same_line(combo_offset)` and
no width cap — so the `56 + 4 + 7 + 4 + 56 + 8 + 64 = 199 <= 200` sum is an assertion in a comment
about a number no longer in force. This also removes the deciding fact behind open question 3: the
two constants are not derived from anything any more, they are two hardcoded widths, which is the
case `/imgui-ui § 6` ("a token used by exactly one panel still belongs in the token bag") covers.

**Fix.** Delete `SIZE.RES_COMBO_W` from `theme.py` and rewrite the `tabs/document.py:31-33` comment
to state the widths as the choices they now are; or, if the 200px budget is meant to keep binding,
make it bind by computing `_CANVAS_PRESETS_W` from `SIZE.RES_COMBO_W` minus the rest of the row so
a future edit to either number cannot silently break the fit.

### 3. The 1px border is the same colour as the checkerboard's light squares, so on a fully transparent output it is invisible along half its length

**Claim.** Finding #21 asks for two things: a checkerboard so a transparent output reads as a
shape, and "a 1px `COLOR.BORDER`-tier rect around the canvas so the extent reads even when the
output is fully opaque and dark". The spec assigns `COLOR.BORDER` to the rect and `_P["bg_2"]` to
`CHECKER_LIGHT`. Those are the same colour.

**Evidence.**

```
CHECKER_LIGHT (0.2353, 0.2196, 0.2118, 1.0)
BORDER        (0.2353, 0.2196, 0.2118, 1.0)
light == border: True
```

`theme.py:134` is `CHECKER_LIGHT = _P["bg_2"]` and `theme.py:129` is `BORDER = _P["bg_2"]`. So in
the transparent case the border alternates between visible (over a `bg_1` dark square) and
invisible (over a `bg_2` light square) every 12 pixels. The opaque-dark case is fine — the border
is `bg_2` against a `bg_0` panel (`theme.py:442` sets `window_bg` to `_P["bg_0"]`), one palette
step, which is what the finding asked for. So this weakens the belt-and-braces overlap between the
two halves of #21 rather than leaving a case uncovered: the transparent output still reads, via the
checkerboard.

Severity is low and this is the spec's own prescription, not an implementation slip — recorded so
manual step 6/7 is judged against what the code can actually show, and so the fix is on the record
if the maintainer reports the edge as ragged.

**Fix.** If the border is meant to read on every backdrop, give the checkerboard its own darker
light-grey (`_P["bg_1"]` / a new `_P` step) so neither checker tone equals `COLOR.BORDER`; or draw
the border after the image with a token one step brighter than both checker greys.

### 4. Item 8 claims to enumerate a grep that returns twice as many sites as it lists

**Claim.** "Seven sites read the OUTPUT pass's canvas texture where a document size is in question,
enumerated from `grep -n 'render_pass.canvas.texture.size' shaderbox/`. Each is resolved
deliberately, not swept."

**Evidence.** `git grep -n "render_pass.canvas.texture.size" 78bd1bf^ -- shaderbox/ | wc -l` returns
**14**, not 7. The seven the table does not list are `document.py:438`, `document.py:682`,
`exporters/youtube.py:517`, `widgets/details.py:83`, `:85`, `:106`, `widgets/document_grid.py:28`
and `widgets/pass_list.py:112`.

I read each: every one is a fit-the-texture or export-source read of exactly the class the table's
four "unchanged deliberately" rows describe, and `widgets/details.py:83`'s "Presets" full-size
button belongs to the Render tab, which the wave's own § Out of scope excludes. **So there is no
behavioural gap** — the code is right. What is wrong is the audit's completeness claim, which is
the sentence a later reviewer would rely on to skip re-running the grep. The stated method
("enumerated from grep") and the stated result (seven) cannot both be true.

**Fix.** In `20_wave_a_canvas_viewer.md § Design decisions item 8`, change "Seven sites" to the
real count and add the eight missing rows with their one-word verdict (each "unchanged, fit the
texture"), or state plainly that the table lists only the sites where a DOCUMENT-size question was
in play and that the remaining eight were checked and are texture-fit reads.

### 5. `pass_graph.py`'s docstring and `dev_flow.md`'s module map both enumerate the module's contents, and neither now covers the canvas clamp

**Claim.** The clamp's new home is correct — the argument in spec item 2 holds under check — but
both places that describe what `pass_graph.py` contains were left describing the pre-wave module.

**Evidence.** `pass_graph.py:10-11`: "everything here is GL-free pure data: the model, the
topological order, the cycle check and the feedback marking are unit-testable with no context".
`dev_flow.md:213-218`: "leaf, GL-free (feature 065): the `graph.json` model (`PassGraph` /
`PassEntry` / `TargetConfig` / `PassLayout` …) plus the planner — `plan_passes` …
`evaluation_order` … and `assert_plan_invariants`". `MIN_CANVAS_PX` / `MAX_CANVAS_PX` /
`clamp_canvas_size` (`pass_graph.py:47-56`) are none of: not the model, not the order, not the
cycle check, not the feedback marking, and not persisted in `graph.json`. Both descriptions are
closed enumerations, so a reader takes the absence as evidence and either re-derives a clamp
elsewhere (the exact thing the funnel move was for) or files it in the wrong module next time.

The home itself checks out: `pass_graph.py` imports no GL and no imgui (`render_preset.py`, which
`tabs/document.py` now also pulls, is likewise pydantic-only), `copilot/backend.py` importing it at
`:102` creates no cycle, and the alternatives the spec rejects (on `Document`, in `theme.py`, left
in `copilot/backend.py`) are each rejected for a reason that survives reading.

**Fix.** Add the clamp to both enumerations: in `pass_graph.py`'s docstring, extend the "everything
here is GL-free pure data" sentence to name the render-dimension bounds; in `dev_flow.md:213-218`,
append "plus the canvas-dimension bounds (`MIN_CANVAS_PX` / `MAX_CANVAS_PX` / `clamp_canvas_size`),
the shared clamp both the Document tab and the copilot's `set_canvas_size` route through" to the
`pass_graph.py` bullet.

### 6. The spec and the test comment both state a falsification path the code does not take

**Claim.** The wave spec argues at length that `test_the_ui_resize_clamps_both_ends` deliberately
omits a trailing render because a render under the bug "would turn a clean assertion failure into a
GL error that names nothing". The GL error happens anyway, before the assertion, because
`Document.set_canvas_size` allocates immediately.

**Evidence.** `document.py:292-293` is `self.canvas_size = size; self.render_pass.canvas.set_size(size)`,
and `core.py:95-101` `set_size` calls `_init`, which builds the framebuffer at `core.py:87`. So the
allocation is inside `_apply_canvas_size`, not deferred to a render. Demonstrated by stubbing the
clamp to the identity and running the test:

```
_moderngl.Error: the framebuffer is not complete (INCOMPLETE_ATTACHMENT)
  .venv/.../moderngl/__init__.py:2073
FAILED tests/test_document_graph.py::test_the_ui_resize_clamps_both_ends
```

(The texture at 99999x4 itself allocates — this box's `GL_MAX_TEXTURE_SIZE` is 32768 and moderngl
does not raise on the texture — it is the framebuffer attachment that refuses.)

The test is still unambiguous: it fails, only under its named bug, and the traceback points at
`_apply_canvas_size`, which is the function under test. But `20_wave_a_canvas_viewer.md:746-749` and
the comment at `tests/test_document_graph.py:668-669` ("Asserts the FIELD and does not render:
under the bug the document holds (99999, 4), which no framebuffer completes") both assert a
mechanism that is false, and the comment additionally narrates a design deliberation rather than
the code as it is — the shape `CLAUDE.md ## Code rules` asks to keep out of source.

**Fix.** In `tests/test_document_graph.py`, cut the comment's second sentence, leaving
`# Both entry points clamp through the same constant.` In the spec, replace the "asserts the FIELD
and does not render" paragraph with the true statement: without the clamp the allocation inside
`set_canvas_size` raises before the assertion is reached, so the test goes red either way and the
absent render buys nothing.

## False trails

- `_draw_canvas_backdrop` as a free function in `ui.py` rather than `ui_primitives.py`: correct as
  landed. It joins six sibling private `_draw_*` helpers in `ui.py` (`:521`, `:561`, `:579`, `:620`,
  `:696`, `:744`), has exactly one caller, and open question 2 explicitly decided the pass strip
  does NOT get it, naming promotion to `ui_primitives.py` as the change if that is ever revisited.
  The skill § 6 rule fires on a second caller, which does not exist.
- `App.canvas_size_buf` / `canvas_size_editing` as plain attributes: matches W-C's
  `pass_settings_name` / `pass_settings_name_buf` (`app.py:297-298`) exactly — typed attribute in
  `__init__`, reset at the lifecycle point. Consistent.
- The presets `begin_combo`: identical in shape to the panel's own sort combo at
  `tabs/document.py:213-220`, down to the fixed preview string. Not a new pattern.
- `help_content.py`: nothing describes the old Resolution combo. Its only canvas strings are the
  `u_aspect` / `u_resolution` uniform descriptions (`:40-41`) and two shader-anatomy mentions
  (`:108`, `:116`), all still true.
- `copilot/prompt.py`: `:59` ("Each document header shows `canvas WxH` — the render resolution") is
  now MORE accurate, since the header at `backend.py:726` reports the field rather than the output
  texture. `:186` and `:310-335` are structural. No stale text.
- `conventions.md`'s funnel-law entry (`:117-124`) states the rule generically and cites 041 / 028 /
  the `.trash/` filter; it names no canvas site, so it needed no edit, and the claim it embodies is
  now true here — `Document.set_canvas_size` (`document.py:284`) is the single writer.
- `copilot/tools/document_ops.py:22-23` and `:101` carry "16-4096" as prose for the model. Those are
  model-facing descriptions, not a second clamp; the arithmetic is gone from `backend.py`.
- The `App` gate racing a mid-edit turn start: if a turn begins while a field is active the digits
  are dropped silently. Same as every gated sibling in the panel, and the copilot input needs a
  click to send, so the field cannot hold focus at that moment. Not a defect.
- `roadmap.md:29` still says "W-A implementing". `dev_flow.md:154` puts the roadmap row/banner
  flip at the FEATURE's step 9, not per wave. Not a W-A obligation.
- Per-frame cost of the checkerboard: `begin_combo` returns False when closed, so `_canvas_presets`
  does not run per frame; and the backdrop's ~3700 `add_rect_filled` calls at the widest layout are
  the spec's own measured budget with a stated fallback (a bigger tile). Nothing to add.

## Coverage statement

Walked every numbered design decision (1, 2, 3, 4, 4a, 5, 6, 7, 8) against the landed code,
opening each cited symbol in the working tree rather than trusting the diff; every Files-touched
row against `git show --stat`; all nine named tests against `tests/test_canvas_presets.py` and
`tests/test_document_graph.py`; all fourteen manual-verification items against the code paths they
exercise (six of them are pixel/focus manual by design and were checked for whether the code CAN
produce the stated picture, which is how finding 3 surfaced). Parent `01_spec.md § W-A`'s eight
bullets and D1 / D2 / D11 each traced to a line. Findings #1 #2 #3 #4 #21 re-read verbatim from
`00_findings.md:19-22` and `:49` and walked as scenarios.

Six claims were checked by running something rather than reading: the gates (exit 0 unpiped, smoke
passed), the two test files (34 passed), the `_canvas_presets` output at three canvas sizes
(finding 1), the theme colour equality (finding 3), the falsifier of the clamp test with
`clamp_canvas_size` stubbed to the identity (finding 6), and the pre-wave grep count for item 8
(finding 4). Greps run over the whole package for `4096`, `_MIN_CANVAS`/`_MAX_CANVAS`,
`canvas.set_size`, `canvas_size =`, `set_canvas_size`, `render_pass.canvas.texture.size`,
`GL_SAMPLER_2D`, `RES_COMBO_W`, `_ = app`, `enter_returns_true`, and `text_colored(COLOR.FG_DIM`.

Not covered: the rendered frame. This box has no `xdotool` and the agent cannot drive the app, so
manual items 1-14 remain the maintainer's — findings 1 and 3 are the two whose symptoms would show
up there first.
