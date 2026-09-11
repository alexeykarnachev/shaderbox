# 090 pre-implementation review — correctness and design

Reviewer role: `dev_flow.md ## Feature flow` step 4, *correctness & design*. Artifact:
`01_spec.md`. Contract: `02_throttle_and_resolution.md` (D1–D11, premises, resolved questions)
and `ai_docs/conventions.md`.

**Verdict: FAIL** — should not land in this form. Three BLOCKERs, each a place where the spec
tells the implementer to build something the code cannot do or that contradicts its own numbers.

---

## Coverage table — 02's decisions, premises and resolved questions

### Premises (02, "Settled by the maintainer, carried here as premises")

| Premise | Spec passage | Status |
|---|---|---|
| P1 — a document has a resolution mode Auto / Fixed | D1: "`UIDocumentState` gains `resolution_mode: ResolutionMode = ResolutionMode.AUTO`" | COVERED |
| P2 — export resolution is its own concept in both modes, never the live size | Goal: "**Export resolution is its own number in both modes** — never the live size"; *Export* section | COVERED |
| P3 — a document may be throttled on a budget share; its clock stays wall time | D6 + D8 ("`u_time` is wall-clock via `Document.live_time`" carried from research) | COVERED |

### D1–D11

| # | Spec passage that implements it | Status |
|---|---|---|
| D1 | "A `ResolutionMode` StrEnum (`AUTO` / `FIXED`) lands in `render_shape.py` … `Document.canvas_size` keeps its name, type and single writer `set_canvas_size`, but now means *the effective live size*" | **PARTIAL** — see F4: the loader cannot resolve the mode before `Document.__init__` runs, because `Document.load_from_dir` parses `document.json` and calls `_seed_feedback` before `_load_ui_state` ever runs (`ui_models.py::load_document_from_dir`). |
| D2 | "the live size is the largest region displaying the document, computed once per frame before any draw"; `auto_canvas_size(displayed, aspect, previous)`; three recorders named | **CONTRADICTED** — see F1 (the recorders cannot write what the spec says they record) and F2 (the Auto aspect is circular). |
| D3 | "a 5 % dead-band, or 8 consecutive frames at a stable size"; `AUTO_RESIZE_DEAD_BAND = 0.05`, `AUTO_RESIZE_STABLE_FRAMES = 8`, pinned by V3 | COVERED — 02 left the numbers to the implementer and the spec picks them with justifications. |
| D4 | "feedback survives an Auto resize by resampling"; *Feedback resample* five steps; `copy_framebuffer` ruled out | **PARTIAL** — the mechanism and the `copy_framebuffer` rejection are right (I reproduced the measurement, see False trails). But the spec resamples only `self._feedback[name]` and leaves `set_canvas_size`'s own `render_pass.canvas.set_size` — a release-and-reallocate — to blank the LIVE canvas, which the next `begin_frame` swaps into the history. See F3. |
| D5 | "the resolution picker is unchanged; export never reads the live size"; "the default is the largest `MENU_SHAPES` entry at the document's aspect — 1920×1080 for 16:9" | **CONTRADICTED** — see F5: the largest 16:9 `MENU_SHAPES` entry is 2560×1440 (`WIDE_1440`), not 1920×1080; and four of the eight documents the spec hand-edits have aspects (4:3, 1:1) that `MENU_SHAPES` carries no entry for at all, so the stated rule has no answer for them. |
| D6 | "one shared GPU budget, default 50 %; `plan_render_set` is the whole rule; hysteresis is 4 frames"; *The plan function* semantics 1–6 | **PARTIAL** — the arithmetic is right (all eight examples recompute exactly, below), the budget default and the hysteresis window are justified from 088 D2. But `MAX_INTERVAL` is specified only for the degenerate `f = 0` case and does not bind a small nonzero `f` (F7), and the interval gate has no phase offset, so every same-`k` document lands on one frame (F6). |
| D7 | "a per-document `CostRecord` with GPU and CPU fields, recorded always"; `App.document_costs`; "a document with no record yet plans as `k = 1`" | **CONTRADICTED** — D7 says the dict "is written each frame from the profile's `document:` spans", the *Frame integration* step 3 puts that write in `_tick_frame_state`, and the *Profiler* section says it happens "at the render site". Those are three different places, two of them in different functions. And no spelling of it is sound, because `document:` spans are keyed by a non-unique TITLE. See F8. |
| D8 | "the script ticks once per UI frame, unchanged"; "`Document.begin_frame` is **not** called on a skipped frame" | **PARTIAL** — the mechanism is right and matches `document.py::begin_frame`'s swap gate. But `cadence_flow §2` names the feedback-stepping RATE as "the single biggest open question the spec must resolve explicitly", and the spec does not resolve it anywhere. See F9. |
| D9 | (a) "`app.profiler.enabled = app.fps_details_open` goes"; (b) `fps_overlay` gains `document_fps: int \| None`; (c) the panel's `document:` rows carry cost, fps, `k`, share, "colored by share of the budget through the existing `load_color`, so a converged throttled document reads green" | **CONTRADICTED** on (c) — a converged throttled document is by construction at `share/budget ≈ 1.0`, and `theme.py::load_color` returns `STATE_ERROR` at `ratio >= 1.0` and `STATE_WARN` at `>= 0.5`. Green is unreachable for exactly the case the spec says reads green. See F10. (a) and (b) are COVERED. |
| D10 | "Render all keeps its meaning. Every open document still renders" | COVERED |
| D11 | "`is_throttle_documents: bool = True` and `document_gpu_budget: float = Field(default=0.5, ge=0.1, le=1.0)`"; the *Settings* snippet | COVERED — naming matches `is_render_all_documents` / `is_copilot_open`, the snippet matches the existing `label_row` + `drag_int` idiom in `popups/settings.py::_draw_body`, and `Field(ge=…, le=…)` is the constraints-on-the-model rule from `conventions.md`. |

### Resolved questions (02, "Resolved questions")

| # | Spec passage | Status |
|---|---|---|
| 1 — viewer and grid at once: the largest region | D2's `auto_canvas_size(displayed, …)`, one size per document | COVERED (as a rule; the inputs are broken — F1) |
| 2 — feedback on resize: resample, release without a leak, texture-count test | D4 + *Feedback resample* + V4 | PARTIAL (F3) |
| 3 — export size under Auto: the existing picker, default largest list entry at the aspect | D5 | CONTRADICTED (F5) |
| 4 — preview budget: not a constant, one shared budget split automatically | D6, *The plan function* step 4; example 4's note "No preview constant exists" | COVERED |
| 5 — script tick per UI frame; the record carries a CPU field | D7, D8; *Out of scope* "A CPU throttle. D7's record carries the CPU field; no policy reads it." | COVERED |
| 6 — size: large, full review cycle | Size banner: "2 pre-impl reviewers, 3 post plus a spec-fidelity pass, and a sanitization sweep" | COVERED |
| 7 — config: checkbox + budget slider, app-level, nothing else exposed | D11; "Everything else stays a code constant" | COVERED |
| 8 — profiler: GPU spans always on, two-number chip, plan columns on the rows | D9 a/b/c | PARTIAL (F10) |
| Still open — D3's damping numbers and D6's hysteresis | D3 picks 5 % / 8 frames; D6 picks 4 frames, justified against 088 D2's two-frame read lag | COVERED |

---

## The six flagged interpretations

**1. D4's blit via a one-quad draw instead of `copy_framebuffer`.** The interpretation is the one
the code forces, and I verified it rather than taking the spec's word. A standalone `require=460`
context, a 64×64 source half white, copied into a 128×128 destination:

```
err after copy_framebuffer: GL_NO_ERROR
copy_framebuffer white fraction: 0.125      (a rescale gives 0.5)
quad blit white fraction: 0.5
copy far-corner px [127,127]: [0 0 0]       quad: [64 64 64]
```

Exactly the spec's stated numbers. `moderngl.Context.copy_framebuffer` between differently-sized
framebuffers copies 1:1 into a corner, silently. D4 step 3 and V4's far-corner assertion are both
correct as written. **No simpler option exists.**

**2. The three size fields moving into `ui_state`, `canvas_size` leaving the top level.** The
interpretation is right for `fixed_size` / `export_size` / `resolution_mode` — they are
per-document preferences and `UIDocumentState` is where those live, with `drop_invalid`
per-key salvage already covering them (`ui_models.py::_load_ui_state`). **But it is not free**: it
puts the fields on the far side of a load-order wall the spec did not check. See F4.

**3. The cost lookup keyed at the render site because `document:` spans carry titles.** The
premise is correct — `ui.py` opens `app.profiler.cpu(f"document:{document_name}")` with
`document_name = ui_document.ui_state.ui_name`, and 088's post-implementation note says so
explicitly. **But the interpretation is NOT forced, and the simpler option is available**:
`_tick_frame_state` already iterates `app.ui_documents`, so `ui_state.ui_name` is in hand there
too. The refresh can happen at step 3 as the *Frame integration* section says, mapping
`id -> ui_name -> span`, with no render-site write and no extra frame of lag. The spec should
pick one and delete the other two spellings. Neither spelling fixes the title ambiguity (F8).

**4. Displayed sizes recorded on frame N and read on N+1.** Forced, and correct. Every recorder
is inside the draw phase and the resolution resolves at the top of `_tick_frame_state`, so a
same-frame read is impossible without moving layout out of the draw. The spec says so and D2's
`None -> previous` rule handles the first frame. **No simpler option.**

**5. Auto's aspect taken from `fixed_size`.** The interpretation is the only non-circular one, so
it is forced — but the spec does not follow it consistently. See F2.

**6. The copilot's `set_canvas_size` switching the mode to Fixed.** Reasonable and not forced by
the code, but it is the right call: `clamp_canvas_size((width, height))` is an explicit pixel
request, and leaving the document in Auto would have the next frame overwrite it. The `Files
touched` line says it; nothing else in the spec contradicts it. One gap: the spec does not say
what the switch does to `export_size`, which under Auto the picker was writing.

---

## Findings

### BLOCKER F1 — the three recorders cannot record what D2 says they record

**Spec passage.** D2: "The recorders: `ui.py::_draw_document_image` (its already-computed
`image_width`/`image_height`), `widgets/document_grid.py::draw_document_preview_button` (its
`cell_w` square), and `widgets/pass_list.py::_draw_pass_tile` for the OUTPUT pass's tile
(`SIZE.PASS_TILE`)."

**Evidence.** Three separate problems, read off the code:

- `draw_document_preview_button(ui_document, border_color, size, selected, armed, stale)` takes
  no `app`. It cannot write `App.displayed_sizes`. (It *can* reach the id — `UIDocument.id`
  exists — but not the dict.)
- `cell_w` and `SIZE.PASS_TILE` are **not** the size the texture is displayed at.
  `ui_primitives.py::preview_cell` computes `scale = min(avail.x / tw, img_h / th)` and draws at
  `dw, dh = tw * scale, th * scale`, inside a bordered child whose `avail` is already narrower
  than `cell_w` and whose image band is `avail.y - footer_h - chips_h`. For a 9:16 document in a
  168-px square cell the drawn height is the band and the drawn width is far under 168. Recording
  `cell_w` over-reports every non-square document and every cell with a footer or chips — which
  is all of them.
- `_draw_pass_tile` draws the pass's own canvas, and for a non-output pass that canvas is already
  `entry.target.target_size(canvas_size)`. Reading a tile size as a request for the DOCUMENT's
  canvas size is what the spec's own *Out of scope* bullet forbids ("a tile never drives a canvas
  size"), yet D2 names `_draw_pass_tile` as a recorder of exactly that.

**Fix.** Record the size where it is actually computed. `preview_cell` already has `dw, dh`; give
it an optional `on_displayed: Callable[[tuple[int, int]], None] | None = None` (or return the
drawn size on `PreviewCellResult`, which already exists) and let each caller that knows a document
id forward it. `_draw_document_image`'s `image_width`/`image_height` are the one recorder the spec
gets right. Drop `_draw_pass_tile` from the recorder list — it draws a pass, not a document.

### BLOCKER F2 — Auto's aspect is circular as written, and the viewer's layout closes the loop

**Spec passage.** D2: "The aspect under Auto is `fixed_size`'s: the mode changes which pixels are
rendered, never the shape of the picture."

**Evidence.** The rule itself is the right one. But nothing else in the spec is changed to obey
it, and two live sites read the aspect off the LIVE canvas instead:

- `ui.py::_draw_document_image` computes `image_aspect = np.divide(*ui_document.document.render_pass.canvas.texture.size)` and derives `image_width`/`image_height` from it — the same two numbers D2 makes the Auto recorder. So under Auto the loop is: canvas size → viewer aspect → displayed size → canvas size. With the aspect pinned to `fixed_size` this converges, but only if the CANVAS's aspect equals `fixed_size`'s at all times. `auto_canvas_size` clamps through `clamp_canvas_size` (`MIN_CANVAS_PX = 16`, `MAX_CANVAS_PX = 4096`), and a clamp on one axis changes the aspect — at which point the viewer's next layout reads the clamped aspect and the loop no longer closes on `fixed_size`.
- `tabs/document.py::_canvas_presets` calls `resolve_dims(shape_to_preset(...), current)` with
  `current = ui_document.document.canvas_size`. Under Auto that is the live size, so the presets
  list — which D5 says stays "unchanged" — becomes viewport-derived. `resolution_flow.md §2` flags
  this exact row ("presets stop being reproducible across sessions").

**Fix.** State that every aspect read switches to `fixed_size` under Auto: `_draw_document_image`'s
`image_aspect`, `_canvas_presets`'s `current`, and `auto_canvas_size`'s own `aspect` argument. Add
them to *Files touched*. And say what happens when the clamp changes the aspect — either clamp on
the constrained axis and re-derive the other from `fixed_size`, or accept the distortion and say so.

### BLOCKER F3 — the resample preserves the history and loses the live frame

**Spec passage.** *Feedback resample*: "`Document.resample_feedback(name, size)`, called from
`set_canvas_size`'s funnel for every pass holding a history"; steps 1–5 operate on
`self._feedback[name]`.

**Evidence.** `Document.set_canvas_size` is two statements:

```python
self.canvas_size = _as_canvas_size(size) or DEFAULT_CANVAS_SIZE
self.render_pass.canvas.set_size(self.canvas_size)
```

and `core.py::Canvas.set_size` is `self.release(); self._init(size)` — release-then-allocate, the
exact ordering D4's step 5 forbids, applied to the LIVE canvas. So after a resize the output
pass's canvas is a fresh blank texture. `Document._swap_feedback` then exchanges that live canvas
with the history at the next `begin_frame`:

```python
self._feedback[name] = render_pass.canvas   # the blank one
render_pass.canvas = previous               # the resampled one
```

One frame after the resize the resampled content is in the live slot (about to be overwritten by
the draw) and the blank is the history. A self-reading output pass samples black — the drop D4
exists to prevent, one frame later. The spec's step 4 note ("the generation carries over
unchanged") is right and does not help: this is not a generation problem.

The load path has the same shape. `_seed_feedback` computes `expected_size` from
`self.canvas_size` — but `resolution_flow.md §2` already flags that `UIDocument.save` writes
`feedback/<pass>.bin` at whatever the live canvas was at save time. Under Auto that is the last
viewer size, so the persisted size and the next session's `canvas_size` routinely disagree.
"Load at the stored size, then resample" is the right answer and the spec says it; the spec does
not say that `UIDocument.save` must now record the size it wrote, nor that `expected_size` stops
being a match criterion (it becomes only the resample target).

**Fix.** Resample both canvases, or restate the funnel as: resample the history to the new size,
allocate a NEW live canvas at the new size, blit the old live into it, then release both old ones
— never `Canvas.set_size` on a canvas whose content matters. Add `_swap_feedback`'s pairing to the
*Feedback resample* section explicitly, because the two canvases trade places and a rule about one
of them is a rule about neither. And say what `_seed_feedback`'s size check becomes.

### MAJOR F4 — the loader cannot resolve the mode before `Document.__init__`

**Spec passage.** *Data model changes*: "`Document.load_from_dir` no longer reads a top-level
`canvas_size`, and `Document.__init__`'s `canvas_size` parameter keeps its meaning as the initial
effective size, handed in by the loader from the resolved mode."

**Evidence.** `ui_models.py::load_document_from_dir` is:

```python
document, meta = Document.load_from_dir(document_dir)
ui_state = _load_ui_state(meta.get("ui_state", {}), dir_name)
```

`Document.load_from_dir` constructs the `Document` from `metadata.get("canvas_size")`, builds
every `Pass` at `document.canvas_size`, and calls `document._seed_feedback(...)` — all before
`_load_ui_state` has parsed `ui_state` at all. "The loader" that would resolve the mode does not
exist at that point. Moving the three fields into `ui_state` therefore requires either reordering
this function (parse `ui_state` first, pass the resolved size down) or having `load_from_dir`
reach into `metadata["ui_state"]` itself — which is a second reader of the same keys, against the
"one canonical home per concept" rule.

**Fix.** Name the reshaped `load_document_from_dir`: parse `ui_state` from the raw metadata first,
resolve the initial effective size from `resolution_mode` + `fixed_size`, then hand it to
`Document.load_from_dir` as an explicit parameter. Add `ui_models.py::load_document_from_dir` to
*Files touched* — it is not listed.

### MAJOR F5 — D5's export default has no answer for four of the eight documents it edits

**Spec passage.** D5: "the default is the largest `MENU_SHAPES` entry at the document's aspect —
1920×1080 for 16:9 — resolved through `shape_to_preset` + `resolve_dims`, as `_canvas_presets`
already does." *Data model changes*: the two sandbox documents get `"export_size": [1920, 1440]`
because "1280×960 is 4:3, so the export default is the largest wide entry at that aspect".

**Evidence.** `render_shape.py::MENU_SHAPES` is `NATIVE` plus three 9:16 and three 16:9 entries.
There is no 4:3 entry and no 1:1 entry. Computed from `SHAPE_TABLE` through `resolve_dims`:

```
SHORT_1440 -> 1440x2560
WIDE_1440  -> 2560x1440
WIDE_1080  -> 1920x1080
```

Two concrete contradictions:

1. "the largest `MENU_SHAPES` entry at 16:9" is **2560×1440** (`WIDE_1440`), not 1920×1080. The
   parenthetical states the wrong number for the one aspect it gives an example of.
2. The tracked sandbox documents are both `[1280, 960]` — 4:3 — and `MENU_SHAPES` has no 4:3
   entry, so "the largest entry at the document's aspect" has no value. The spec's own answer,
   1920×1440, is not a `MENU_SHAPES` entry at all; it is 4:3 at longest edge 1920, which is a
   different rule (`ResolutionPolicy.LONGEST_EDGE`, not `FIXED_ASPECT`). The shipped examples are
   worse: `1080×1920` (9:16, fine), `1280×960` and `1280×960` (4:3), `1280×1280` and `512×512`
   (1:1), `1600×900` (16:9). Four of six have no entry.

**Fix.** State the rule as what the spec's own hand-edit actually does: the document's aspect at
the largest longest-edge in `SHAPE_TABLE` (2560, or 1920 if that is the intent — pick one), which
is `ResolutionPolicy.LONGEST_EDGE` through `resolve_dims`, not `shape_to_preset`. Then recompute
the eight hand-edit values from that rule and state them, because the two given (`1920×1440`) do
not follow from either reading.

### MAJOR F6 — every same-`k` document renders on the same frame

**Spec passage.** *Frame integration* step 8: "`app.frame_idx % k != 0` is skipped".

**Evidence.** The gate has no per-document phase. Example 3's three previews all get `k = 43`, so
all three render on frames 0, 43, 86 — one frame carrying 15 ms where the plan allotted a 0.35 ms
remainder, and two frames carrying nothing. The current document at `k = 5` coincides with them
every 215 frames, for a ~23 ms frame. The remainder split exists to make previews affordable by
spreading them; a common phase spends the whole spreading on one frame. The measured §3 result the
Goal cites (60 of 300 frames at 40 ms) is a SINGLE-document measurement and does not cover this.

**Fix.** Give each document a phase: `(app.frame_idx + phase[id]) % k != 0`, with `phase` assigned
from the document's index in `displayed` (stable within a frame, ephemeral like the rest). State
it in `plan_render_set`'s output or beside the gate, and add a verification case — a falsifier
that removes the phase and asserts two same-`k` documents land on different frames.

### MAJOR F7 — `MAX_INTERVAL` does not bind the case that needs it

**Spec passage.** *The plan function* step 4: "`f = 0` clamps to `MAX_INTERVAL = 60`, so a
document never stops rendering entirely."

**Evidence.** `MAX_INTERVAL` is applied only to the degenerate `f = 0`. A small nonzero `f` is not
capped, and the formula produces those routinely. Ten previews at 5 ms with example 3's 0.35 ms
remainder: `f = 0.35 × 60 / 50 = 0.42`, so `k = round(60 / 0.42) = 143` — a preview refreshing once
every 2.4 s, well past the cap the spec says exists. The stated guarantee ("a document never stops
rendering entirely") is not delivered by the stated rule.

**Fix.** Apply the cap to the computed `k`, not to `f`: `k = min(MAX_INTERVAL, max(1, round(target_fps / f)))`,
with `f = 0` falling into the same clamp. Add a worked example at ten previews so the cap has a
row that exercises it, and a falsifier that removes the cap.

### MAJOR F8 — the cost record has three homes in the spec and no sound key in the code

**Spec passage.** D7: "written each frame from the profile's `document:` spans". *Frame
integration* step 3: "**Costs refresh** from the last profile into `app.document_costs`."
*Profiler and FPS surfaces*: "`ui.py` writes `app.document_costs[document_id]` at the render site
… not by parsing the tree afterwards".

**Evidence.** Step 3 runs inside `_tick_frame_state`; the render site is in `_update_and_draw`,
after `_tick_frame_state` returns and after step 6 has already run the plan. Those cannot both be
true. The render-site spelling also costs an extra frame: the plan at step 6 of frame N would read
a record written at frame N−1's render site, on top of the ring's own two frames.

Separately, neither spelling is sound. `document:` spans are keyed by `ui_state.ui_name`, and
nothing enforces title uniqueness — `copilot/backend.py::rename_document` and the new-document
paths all assign a free-form name. `profiling.py` handles same-named siblings by ORDINAL in its
ring key (`self._ordinals`), but the published `Span` objects in `children` carry only `name`, so
two documents titled "untitled" are two indistinguishable `Span(name="document:untitled")` and a
by-title lookup picks one arbitrarily. The wrong document then gets the other's cost and is
throttled on it.

**Fix.** Pick one site (step 3 is the better one — the plan reads it in the same function, one
frame fresher, and `ui_state.ui_name` is reachable there through `app.ui_documents`), delete the
other two spellings, and resolve the key. The clean resolution is to make the span carry the id:
open the span as `f"document:{document_id}"` and have `profile_rows_plan` render the title by
looking the id up — but 088 chose the title *because* a uuid does not fit a 280-px panel, so
that is a real trade the spec must make explicitly rather than inherit. The cheap alternative is
to state that a duplicate title costs both documents their record (both plan `k = 1`), which is
fail-soft and matches D7's "nothing is throttled on a cost nobody measured".

### MAJOR F9 — the feedback-stepping rate, which the research calls the biggest open question, is not answered

**Spec passage.** D8 covers `begin_frame` and `_frame`; nothing anywhere addresses what a
throttled feedback pass LOOKS like.

**Evidence.** `cadence_flow.md §2`, the "Feedback stepping RATE" row, verbatim: "This is the
single biggest open question the spec must resolve explicitly, since it's the one place a 'render
less often' policy is not merely slower but qualitatively DIFFERENT (fewer feedback steps ≠ the
same animation played back slower — it is a coarser integration)." It also names the collision:
the maintainer's "no budget/quality cap on a document" constraint from `00_research.md`. The spec
resolves the MECHANISM (`begin_frame` is skipped, which is correct) and never states the
consequence or whether it is accepted.

**Fix.** One paragraph in D8 stating that a throttled feedback pass integrates at the document's
own rate, that this is a visible change in what a trail effect looks like under load, and that it
is accepted because D11's checkbox is the escape. That is a decision the maintainer has arguably
already made by accepting the throttle at all — but the research asked for it in writing and the
review cycle is where it gets written, not assumed.

### MAJOR F10 — "a converged throttled document reads green" is false under the stated formula

**Spec passage.** D9c: "the wall-time share (`cost × doc fps`), colored by share of the budget
through the existing `load_color`, so a converged throttled document reads green."

**Evidence.** `theme.py`:

```python
LOAD_WARN_RATIO: float = 0.5
LOAD_ERROR_RATIO: float = 1.0
def load_color(ratio): ... >= 1.0 -> STATE_ERROR; >= 0.5 -> STATE_WARN; else STATE_OK
```

Convergence means the document is using its budget share, so `share ≈ budget` and
`share / budget ≈ 1.0`. Computed on the spec's own examples:

| example | cost | k | doc fps | share | share/budget | `load_color` |
|---|---|---|---|---|---|---|
| 2 | 100 ms | 12 | 5.0 | 0.500 | **1.00** | STATE_ERROR |
| §3's combined | 40 ms | 5 | 12.0 | 0.480 | **0.96** | STATE_WARN |
| 1 | 6 ms | 1 | 60.0 | 0.360 | 0.72 | STATE_WARN |

Green (`STATE_OK`) needs `share/budget < 0.5`, i.e. a document using less than half its allowance
— which the throttle exists to prevent. The one case that reads green is a document too cheap to
throttle. The stated reading is inverted for exactly the case D9c names.

**Fix.** Either color by share of the FRAME (`load_color(share)`, so a converged document at 0.5
of wall time reads warn and a runaway reads error, which is the honest reading), or say plainly
that a converged throttled document reads warn and that warn here means "at its allowance", which
is what the reader should see. Do not leave a sentence in the spec that the implementer will
implement and the maintainer will then read as a bug.

### MINOR F11 — `ProfileRow` gains two fields the draw cannot use

**Spec passage.** D9c: "`ProfileRow` gains `interval: int` and `share: float`."

**Evidence.** `ui_primitives.py::_profile_rows` reads only `starts_tree`, `depth`, `name`,
`number` and `color`. The same section already specifies the row's `number` as
`f"{cost:.1f} ms  {doc_fps:.0f} fps  x{k}"` and its `color` as `load_color(share / budget)` — so
`k` and `share` are already in the row twice, once as data and once formatted. `ProfileRow`'s own
docstring says it is "decided before any imgui call", i.e. the formatted string IS the contract.
This is the "one canonical home per concept" rule.

**Fix.** Drop the two fields. If a test wants to assert on `k` without parsing a string, assert on
`RenderPlan.intervals`, which is the canonical home and is already pure data.

### MINOR F12 — `ui_models.py::load_document_from_dir` and `tabs/document.py::_canvas_presets` are missing from *Files touched*

`load_document_from_dir` must change for F4; `_canvas_presets` must change for F2. Neither is
listed. The `ui_models.py` line covers "`UIDocumentState`'s three fields, `UIAppState`'s two, and
`UIDocument.save`'s new shape" only.

### MINOR F13 — D5 does not say what the copilot's Fixed switch does to `export_size`

`Files touched` says `copilot/backend.py`'s `set_canvas_size` "writes `fixed_size` and switches
the document to Fixed". Under Auto the picker had been writing `export_size`; after the switch the
same two fields write `fixed_size`. Whether `export_size` is left at its Auto value (so a later
switch back to Auto restores it — probably right) or reset should be one clause in D5.

### MINOR F14 — the D9a probe is specified but its number is a precondition the spec does not gate on

*Pre-implementation measurements* says the always-on query cost "is measured first" and that the
number goes "into this section … before implementation starts", with D7's swap-wait fallback if
it is not negligible. That is the right shape. But nothing in the *Verification* list turns the
missing number red, and the spec is otherwise ready to hand to an implementer. Add the measurement
as an explicit gate — the spec is not implementable until the line is written — or the probe
becomes optional in practice.

---

## Internal consistency

**Names.** Every symbol the spec cites exists at the cited `file::function`, verified by grep:
`clamp_canvas_size`, `MIN_CANVAS_PX`/`MAX_CANVAS_PX` (`pass_graph.py`), `DEFAULT_CANVAS_SIZE`,
`shape_to_preset`/`resolve_dims`/`MENU_SHAPES`/`RenderShape` (`render_shape.py`,
`render_preset.py`), `_canvas_presets` (`tabs/document.py`), `draw_document_preview_button`
(`widgets/document_grid.py`), `_draw_pass_tile`/`SIZE.PASS_TILE`, `_seed_feedback`/
`_feedback_canvas`/`_feedback_generation`/`set_canvas_size`/`render_media`/`_render_media_into`
(`document.py`), `FitPolicy.SCALE_DISTORT`/`RENDER_AT_TARGET`, `TargetConfig.target_size`,
`profiling.gpu_total`, `theme.load_color`, `profile_rows_plan`/`fps_overlay`/`ProfileRow`
(`ui_primitives.py`), `ChannelBlit` (`channel_blit.py`), `media.texture_to_rgba8`. Two naming
slips: the spec writes "`_update_and_draw`'s render block" in *Frame integration* and
"`ui.py::_draw_app_panel`'s `app.profiler.enabled = app.fps_details_open`" in *Profiler* — both
are correct (the line is in `_draw_app_panel`, the render block is in `_update_and_draw`), so no
finding. `resample_feedback` does not exist yet, as intended.

**The worked examples recompute exactly.** All eight, by hand at `period = 16.7`, `budget × period = 8.35`:

| # | recomputation | spec | match |
|---|---|---|---|
| 1 | `6 ≤ 8.35` → `k = 1`, fps 60 | k=1, 60 | ✓ |
| 2 | `ceil(100 / 8.35) = ceil(11.976) = 12`, fps `60/12 = 5.0` | k=12, 5.0 | ✓ |
| 3 | `ceil(40/8.35) = ceil(4.79) = 5`; used `40/5 = 8.0`; remainder `8.35 − 8.0 = 0.35`; `Σcost = 15`; largest `f` with `15f ≤ 0.35 × 60 = 21` → `f = 1.4`; `k = round(60/1.4) = 43` | 5, 0.35, 21, 1.4, 43 | ✓ |
| 4 | `2 ≤ 8.35` → `k=1`; remainder `8.35 − 2 = 6.35`; `Σcost = 20`; `20f ≤ 6.35 × 60 = 381` → `f = 19.05`; `k = round(60/19.05) = 3` | 1, 6.35, 381, 19.05, 3 | ✓ |
| 5 | `4 ≤ 8.35` → `k=1`; remainder `4.35`; `100f ≤ 261` → `f = 2.61`; `k = round(60/2.61) = 23` | 1, 4.35, 261, 2.61, 23 | ✓ |
| 6 | all `k=1` by step 1 | all 1 | ✓ |
| 7 | `budget × period = 16.7`; `ceil(40/16.7) = ceil(2.395) = 3`, fps 20 | 3, 20 | ✓ |
| 8 | no record → `k=1` by step 2 | all 1 | ✓ |

The arithmetic is sound; F6 and F7 are about what the table does not cover, not about what it
computes. One stated-number check: example 5's note "Fixed means the cost stays 100 ms" is
consistent with D10.

**Frame-integration order vs the 084 D5 hazard.** The spec's claim holds. `conventions.md`'s
project-switch bullet states the hazard as "a popup draws AFTER the editor panel and the document
image have pushed their texture handles into the frame's draw list, so releasing them there
leaves imgui rendering freed GL names", and the remedy as consuming the request in
`_tick_frame_state` "before any drawing". `ui.py::_tick_frame_state` opens with exactly that
(`pending = app.pending_project_switch` … `app.switch_project(pending)`), and the whole function
runs inside `_update_and_draw`'s `with app.profiler.cpu("tick")` block, before `imgui.new_frame()`
(which is far below, inside the `with app.profiler.cpu("ui")` block). So steps 4 and 5 —
`set_canvas_size` and the resample, both of which release GL objects — land in the same safe
window the project switch uses. Verified by reading both functions.

**Step ordering 4→5→6.** Sound, and the stated reason ("`k` must be computed from the cost at the
resolution actually rendered") is backed by `cost_and_throttle.md ## False trails`, which measures
the k=12-vs-k=5 gap for the same document at two sizes. The residual lag the spec acknowledges
(the damping window plus the ring's two frames) is real and the 4-frame hysteresis is a
reasonable absorber.

**Step 3's read.** `app.last_profile` is assigned in `update_and_draw` AFTER `_update_and_draw`
returns, so at step 3 it holds frame N−1's value. That is what the spec wants. Consistent.

**Step 8's placement.** `begin_frame` is already called at the tail of `_tick_frame_state`, so
step 8 lands where the spec says. The render block it must agree with is in `_update_and_draw`;
the spec says the whole `if ui_document is not None:` body is skipped together and gives the
reason (`cadence_flow §2`'s pending-sweep row). Correct — I read both call sites.

**`plan_render_set`'s leaf claim.** "imports `math` and `dataclasses` only. No GL, no imgui, no
`App`, no `Document`." The signature takes only `dict[str, CostRecord]`, `str | None`, `list[str]`,
floats, `dict[str, ThrottleState]` and a bool — all primitives and own types. `auto_canvas_size`
takes tuples and a float. `document_export_size` is listed in *Files touched* but never specified
anywhere else in the spec (MINOR, folded into F5's fix — it is presumably D5's default rule). The
leaf claim holds; nothing in the module needs `App`, so the cycle-from-types signal is clean.

---

## Conventions

- **Full annotations.** The three code blocks carry them (`auto_canvas_size`'s signature,
  `plan_render_set`'s, the dataclass fields). OK.
- **No `if TYPE_CHECKING`.** Nothing in the spec needs a forward reference; `render_plan.py` is a
  leaf and `App` holds instances of its types, never the reverse. OK.
- **Imports at top.** Nothing in the spec asks for a function-body import. OK.
- **No `@staticmethod`.** `plan_render_set` and `auto_canvas_size` are module-level free
  functions, which is the rule's positive branch. `resample_feedback` uses `self._feedback` and
  `self._gl`, so it is a genuine method. OK.
- **Leaf-module rule.** `render_plan.py` is declared leaf and its signature honors it (above). It
  needs a line in `dev_flow.md ### Module map`, which *Files touched* includes. OK.
- **No-migration rule.** Nothing in the spec reads an old shape. The wording is explicit and
  correct: "there is no migration and no old-format reader", "`Document.load_from_dir` no longer
  reads a top-level `canvas_size`", and the hand-edit is named per file with `git add projects/dev`
  in the same commit. The two tracked sandbox documents do carry `"canvas_size": [1280, 960]` today
  — verified. This is the rule the spec gets most cleanly right. Its only defect is the VALUE of
  the hand-edit (F5), not the posture.
- **One canonical home per concept.** Violated twice: `ProfileRow`'s two new fields duplicate the
  row's formatted `number` (F11), and the cost-record write is specified in three places (F8).
- **UI through `ui_primitives.py` / `theme.py`.** The *Settings* snippet calls `imgui.checkbox` and
  `imgui.drag_int` directly — which matches the existing `popups/settings.py::_draw_body` exactly
  (`show_cheatsheet`, `global_target_fps` do the same), with `label_row` from `ui_primitives` for
  the label. No hand-rolled `push_style_color`. OK. The chip and panel changes go through
  `fps_overlay` / `profile_rows_plan` / `load_color`, all already in `ui_primitives`/`theme`. OK.
- **Prose budget.** "Throttle documents" (2 words) and "Document GPU budget" (3) are within the
  control-label budget; `tests/test_ui_prose_budget.py` derives its domain by reflection over
  `ui_primitives` signatures, so `label_row`'s label is scored and both pass. The chip's
  `f"{fps} | doc {document_fps}"` adds one literal word. The spec's claim holds.
- **Constraints on the model.** `document_gpu_budget: float = Field(default=0.5, ge=0.1, le=1.0)`
  follows the "new persisted state gets constraints on the model" rule. OK.
- **No raw line numbers in docs.** The spec cites symbols throughout. OK.

---

## False trails — probed, and fine

- **`copy_framebuffer` does not rescale.** I did not take the spec's word; I wrote a 25-line probe
  against a standalone `require=460` context and reproduced the stated numbers exactly (0.125 vs
  0.5 white fraction, `GL_NO_ERROR`, corner copy, far-corner black under copy and non-black under
  the quad). D4 step 3 and V4's far-corner assertion are both correct. **Do not re-litigate.**
- **All eight worked examples.** Recomputed by hand; every number in the table is right, including
  the intermediate remainders and `f` values. **Do not re-litigate the arithmetic** — F6 and F7
  are about coverage, not correctness.
- **The 084 D5 ordering claim.** Read `_tick_frame_state` and `_update_and_draw` end to end. The
  spec's ordering argument is correct and the analogy to the project switch is exact.
- **`app.last_profile`'s one-frame offset.** Checked the assignment site; step 3 reads the previous
  frame, which is what the spec intends. Not a bug.
- **The settings snippet's idiom.** Compared against `popups/settings.py::_draw_body`'s existing
  `Target FPS` and `Show keyboard cheatsheet` rows. It matches. The direct `imgui.checkbox` is not
  a `ui_primitives` violation here.
- **`is_throttle_documents` naming.** Matches `is_render_all_documents` / `is_copilot_open`. Fine.
- **Prose budget for the two settings labels and the chip.** Read the gate's reflection mechanism;
  both labels and the chip are inside budget. Fine.
- **`render_plan.py`'s leaf claim / the cycle-from-types signal.** Checked every parameter type in
  both public signatures. Nothing needs `App` or `Document`. Clean.
- **`MENU_SHAPES` largest entry.** I computed all six through `resolve_dims`' `FIXED_ASPECT` branch
  rather than eyeballing `SHAPE_TABLE`; 2560×1440 is the largest 16:9 and 1440×2560 the largest
  9:16. F5's numbers are measured, not inferred.
- **`load_color`'s thresholds.** Read from `theme.py` (`0.5` / `1.0`) and applied to the spec's own
  examples. F10's table is computed, not asserted.

---

## Coverage statement

**Read fully:** `01_spec.md`; `02_throttle_and_resolution.md`; `research/cost_and_throttle.md`;
`research/resolution_flow.md`; `research/cadence_flow.md §2` and `§4` (sections 1, 3, 5–7 skimmed
via the section index); `conventions.md ## Code rules` and the `## Design decisions` bullets on
the project-switch/084 D5 funnel, no-migration, persistence-evolution, parallel dicts, funnels,
gates, mutation testing, and the `channel_blit` view bullet, plus the `## Known quirks` entries on
`texture_to_rgba8` and one-`update_and_draw`-per-process; `CLAUDE.md`; `dev_flow.md` step 4 and
the `### Module map` head. In code, read end to end: `ui.py::_tick_frame_state`,
`ui.py::_update_and_draw`, `ui.py::update_and_draw`, `ui.py::_draw_document_image`,
`ui.py::_draw_app_panel`'s FPS block, `document.py::Document.__init__` / `set_canvas_size` /
`begin_frame` / `_swap_feedback` / `_seed_feedback` / `_feedback_canvas` / `render_media` /
`load_from_dir`, `core.py::Canvas`, `render_shape.py` in full, `render_preset.py::resolve_dims`,
`tabs/document.py`'s canvas block, `ui_primitives.py::preview_cell` / `profile_rows_plan` /
`_profile_rows` / `fps_overlay` / `ProfileRow`, `widgets/document_grid.py::draw_document_preview_button`,
`widgets/pass_list.py::_draw_pass_tile`, `ui_models.py::UIDocumentState` / `UIAppState` /
`UIDocument.save` / `load_document_from_dir`, `profiling.py`'s module docstring and `Profiler`
state, `popups/settings.py::_draw_body`'s General section, `copilot/backend.py::set_canvas_size`,
`theme.py::load_color`.

**Skimmed:** `00_research.md` (only through the spec's citations of it);
`research/cadence_flow.md §§1,3,5,6,7`; `research/baseline.md` (cited by the D7 fallback, not
independently checked); `ai_docs/features/088_frame_profiler/01_spec.md` (the review-history
section and the `document:<name>` note read in full, the body skimmed);
`tests/test_ui_prose_budget.py` and `tests/test_persistence_completeness.py` (mechanism, not
every row).

**Ran:** one standalone-context probe reproducing the `copy_framebuffer` measurement (written to
the scratchpad, not the repo); the eight worked examples and the `load_color` / `MENU_SHAPES` /
phase-collision arithmetic in `python3`; greps verifying every cited symbol and the two sandbox
`document.json` shapes. **Did not run:** `make gates` (the tree is unmodified and this is a
document review); the D9a always-on-query probe, which the spec itself defers to
pre-implementation.

**Verdict: FAIL** — three BLOCKERs (F1 recorders, F2 circular aspect, F3 the resample loses the
live frame) plus seven MAJORs; the spec should not go to implementation until at least the
BLOCKERs and F5, F8 and F10 are rewritten.

---

# Round 2

Re-review of the revised `01_spec.md` (688 lines). Read the whole spec again from disk, re-ran the
arithmetic, and re-verified every new claim against the code rather than against the drafter's
summary of it.

**Verdict: PASS-WITH-MINORS.** All fourteen round-1 findings are closed, most of them structurally
rather than by patching the sentence — F5 and F13 vanished because the two-size model collapsed to
one field, F11 by deleting the duplicate rather than reconciling it. Two new findings, both MINOR
and both textual: one self-contradiction between a table and the prose under it, and one safety
claim in D4 that is true of the path it names and false of a second caller it does not.

## Round-1 closure table

| id | new passage | verdict |
|---|---|---|
| **F1** (B) recorders can't record what D2 says | D2: "`preview_cell` draws at `dw, dh` after `scale = min(avail.x / tw, img_h / th)`; `cell_w` over-reports every non-square cell… `PreviewCellResult` gains `drawn_size: tuple[float, float]`, and the callers holding a document id record it: `widgets/document_grid.py::draw_document_preview_grid` and `popups/examples.py`'s grid… **Pass-strip tiles are not recorders**" | **CLOSED.** Verified: `PreviewCellResult` today carries only the four click/delete booleans, so `drawn_size` is a genuine addition; `grep -rn draw_document_preview_button` returns exactly two call sites (`document_grid.py:77`, `popups/examples.py:97`), both of which hold a document id, so leaving the function `app`-free works and the spec's stated reason for that choice is accurate. `Out of scope` also gained "a pass-strip tile is not a recorder", so the rule is stated in both places it needs to be. |
| **F2** (B) Auto's aspect is circular | D1: "a clamp that would change the aspect resolves on the CONSTRAINED axis and re-derives the other from `resolution`'s aspect"; D2: "**Every aspect site reads `resolution`'s under Auto, never the live canvas**: `_draw_document_image`'s `image_aspect`, `tabs/document.py::_canvas_presets`'s `current`, `widgets/details.py`'s aspect lock and its two preset-button labels, `copilot/backend.py`'s probe aspect" | **CLOSED**, and the enumeration is right, which I checked rather than assumed. `grep -rn 'np.divide(\*.*texture.size'` returns three sites: `ui.py:679` (named), `details.py:105` (named, the aspect lock), and `core.py:505` — which is `Pass.render`'s `u_aspect` and must stay live, since it describes the canvas being drawn into. Correctly excluded. The two preset buttons are real and were not in my round-1 list: `details.py` computes `full_w, full_h = …canvas.texture.size` and `half_w, half_h = adjust_size(…)` and labels two `standard_button`s with them. `copilot/backend.py:2150` (`cw, ch = document.render_pass.canvas.texture.size` for the probe's aspect) is real. Five sites, all real, none missed. The clamp rule closes the loop I flagged. |
| **F3** (B) the resample loses the live frame | D4: "Resampling the history alone loses the picture one frame later: `set_canvas_size` blanks the LIVE canvas (`Canvas.set_size` is release-then-allocate) and the next `_swap_feedback` trades the blank into the history"; *Feedback resample*'s funnel step 2 replaces the `Canvas.set_size` call on the output pass; V4 asserts "BOTH the live canvas and the history carry content" | **CLOSED.** The diagnosis is restated correctly and the funnel is the right shape — `resample_canvas(old, size) -> Canvas` returning a replacement, callers reassigning, `old.release()` last. Blast F14's refinement (a non-output history resamples to `entry.target.target_size(new)`, not the document size) is a real catch I missed and is now step 3. |
| **F4** (M) the loader can't resolve the mode before `Document.__init__` | *Data model changes*: "`ui_models.py::load_document_from_dir` today calls `Document.load_from_dir(...)` and only then `_load_ui_state(...)` — so the mode is parsed after the `Document` is built, every `Pass` allocated and `_seed_feedback` run. It becomes: read the raw metadata, parse `ui_state` FIRST, resolve the initial effective size…, then pass that size to `Document.load_from_dir` as an explicit parameter." | **CLOSED.** The description of today's order matches `ui_models.py::load_document_from_dir` exactly, and "`load_from_dir` reads no top-level `canvas_size` and no `ui_state` key — one reader per concept" answers the canonical-home half of the finding. Listed in *Files touched*. |
| **F5** (M) D5's export default has no answer for 4 of 8 documents | D1: "There is no second size field — round 1's `export_size` and its derivation rule are gone (closes correctness F5)"; D5: "No default-derivation rule and nothing to compute — a hand-edited document keeps its number (1280×960 stays 1280×960)." | **CLOSED, structurally.** The best fix in the revision: the finding was that a derivation rule had no answer for 4:3 and 1:1, and the answer is to delete the derivation. Every document keeps the number it has, which I verified against all eleven files (table below). The `MENU_SHAPES`-largest-entry arithmetic that was wrong is simply gone. |
| **F6** (M) same-`k` documents share a frame | D6: "Same-`k` documents get a **phase offset** by stable index… the gate is `(app.frame_idx + phase[id]) % k != 0`"; `RenderPlan.phases: dict[str, int]`; semantics (6) `phases[id] = i % interval`; V2a | **CLOSED.** Example 3 now reads "Phases 1, 2, 3 — three frames, not one (F6)", which I recomputed: `displayed` indices 1,2,3 under `i % 43` give 1,2,3 — three distinct frames. V2a's falsifier ("drop the phase → all three share every render frame") is applied at the right layer. |
| **F7** (M) `MAX_INTERVAL` binds only `f = 0` | Semantics (4): "`k = min(MAX_INTERVAL, max(1, round(target_fps / f)))` — the cap binds every `k`, `f = 0` falling into the same clamp"; example 9 | **CLOSED.** Example 9 is the row I asked for, with my own numbers: "ten previews at 5 ms, current 40 ms… `f = 0.42`; uncapped `round(60/0.42) = 143`, **capped to 60** → fps 1.0". Recomputed exactly (below). V1's falsifier "drop the `MAX_INTERVAL` clamp → case 9 returns 143" is the mutation that proves the cap. |
| **F8** (M) three cost-record homes, non-unique title key | D7: "The single write site is **`_tick_frame_state` step 3**… the render site writes nothing (correctness F8 found three specified sites). The title key is closed at the source: render sites open `f"document:{document_id}"` and `profile_rows_plan` renders the title via an `id -> title` map" | **CLOSED**, and it took the harder of the two options I offered rather than the fail-soft one. Step 3 is now the only write site; *Frame integration* step 3 says "The ONE write site (D7)" and the *Profiler* section says "Render sites open `f"document:{document_id}"`… Nothing is matched through a title" — the three spellings are down to one. The 280-px-panel trade I flagged is met head-on by the `id -> title` map, so the panel still shows a name. V10a ("two documents sharing a title… **Falsifier:** match by title → one takes the other's numbers") pins it. |
| **F9** (M) feedback-stepping rate unanswered | D8: "**a throttled feedback pass steps fewer times per wall second, so a trail effect is a coarser integration under load — not the same animation played slower.** This follows from the maintainer's own throttle premise… and is **accepted**. D11's checkbox is the escape." | **CLOSED.** It quotes `cadence_flow §2`'s own framing ("the single biggest open question"), states the consequence in the research's own words, and records the acceptance with its reason. That is the written decision the research asked for. |
| **F10** (M) "converged reads green" is false | D9: "The row color is **not** `load_color`, whose `STATE_ERROR` starts at 1.0 — exactly where a converged document sits… `theme.py::throttle_color(share_ratio: float, frame_over_budget: bool)`: **`STATE_OK` at ≤ 1.0**… **`STATE_WARN` above 1.0**, **`STATE_ERROR` above 1.5 or whenever the UI's frame period exceeds the target**" | **CLOSED.** Recomputed against the new bands: example 2's converged document sits at ratio 1.000 → `STATE_OK`; the §3 combined config at 0.960 → `STATE_OK`; example 1's cheap document at 0.720 → `STATE_OK`. Green is now reachable for exactly the case D9c names, which is what the finding asked for. The `frame_over_budget` clause is a genuine addition — it keeps "inside its allowance while the frame still misses" visible, which neither of my two suggested fixes covered. `theme.py::throttle_color` is a new symbol, correctly listed in *Files touched*. |
| **F11** (m) `ProfileRow`'s two fields duplicate the number | *Profiler*: "`ProfileRow` gains **no** new fields (correctness F11: `k` and `share` are already in the formatted `number`, and `RenderPlan.intervals` is the canonical home for a test to assert on)." | **CLOSED**, by deletion, with the canonical home named. |
| **F12** (m) two files missing from *Files touched* | `ui_models.py` line: "**and `load_document_from_dir`'s reordering** (correctness F4/F12)"; `tabs/document.py` line: "**`_canvas_presets` reads `resolution`** (F2/F12)" | **CLOSED.** Both present, both cross-referenced. |
| **F13** (m) Fixed switch's effect on the second size field | Review history: "**F13 (m)** … → moot; one field now." | **CLOSED**, dissolved by F5's model change. `copilot/backend.py`'s line reads "`set_canvas_size` writes `resolution` and sets the mode to Fixed", and V13 verifies the mode sticks across a `_tick_frame_state`. |
| **F14** (m) the probe gated nothing | *Pre-implementation measurements*: "**Both are done; this section no longer gates starting.**" with the measured numbers and a stated **pass threshold (p95 delta ≤ 0.1 ms)** | **CLOSED.** The measurement was run by the round-1 verification reviewer and the number recorded (p95 delta +0.005 ms across two runs). The section also does the honest thing with a noisy statistic: "The MEDIAN delta changed sign between runs (−0.169, then +0.337 ms), so it sits inside the run-to-run noise floor". D7's CPU-swap-wait fallback is dropped as speculative machinery, which is the right call under the `conventions.md` speculative-machinery bullet now that the premise is measured. |

## Fresh pass over what changed

**The single `resolution` field and its role-follows-mode rule.** Sound, and it removes the whole
class of defect F5 was an instance of. The one subtlety the spec gets right: under Auto, export
must resolve from `resolution` because `resolve_dims`'s `FREE` branch is `w, h = src_w, src_h`
(verified in `render_preset.py`) — so `RenderShape.NATIVE` passes the *source* straight through,
and if the source were the live canvas, NATIVE under Auto would mean "whatever the panel is". The
blast reviewer's F1 is real and the fix ("`render_media` passes `self.resolution` to `resolve_dims`
on BOTH branches") is correct. I confirmed the three consumers it names: `_artifact_matches_shape`
does pass `current_document.document.render_pass.canvas.texture.size` as `resolve_dims`'s source
today (`exporters/youtube.py`), and it gates `upload_enabled`, so under Auto a panel drag would
indeed disarm Upload. The `preset=None` path renders at `resolution` into a scratch canvas and
`resolution_details` stays the Render tab's — which I checked is a real distinction, since
`_render_image` resizes to `details.resolution_details` only when the canvas disagrees.

**The five aspect sites.** Enumerated correctly; see F2 above. Nothing missed, and `core.py`'s
`u_aspect` is correctly left live.

**`PreviewCellResult.drawn_size` flowing to callers with a document id.** Correct, and the
"stays `app`-free" decision is justified by the two call sites both holding an id.

**The resample's coverage and release order.** Correct per canvas, and blast F14's scaled-pass
target is right. See new finding R2 for the one caller whose ordering the safety claim does not
cover.

**Phase offsets, `MAX_INTERVAL`, `document:{id}` spans with an id→title map.** All three verified
above. One note on `phases[id] = i % interval`: `i` is the index in `displayed`, and `displayed` is
`tick_documents`, whose order is stable frame to frame (current first, then `ui_documents` insertion
order) — so the phase is stable, as "by stable index" claims. When the current document changes the
indices shift and a phase jumps, which costs at most one skipped or doubled render. Not a finding.

**`theme.py::throttle_color`.** Verified against the recomputed ratios; see F10.

**D8's accepted coarser stepping.** Stated and accepted; see F9.

**The nine worked examples — recomputed by hand.** `period = 16.7`, `budget × period = 8.35`:

| # | recomputation | spec | match |
|---|---|---|---|
| 1 | `6 ≤ 8.35` → `k = 1`, phase 0, fps 60 | same | ✓ |
| 2 | `ceil(100/8.35) = ceil(11.976) = 12`; fps `60/12 = 5.0` | same | ✓ |
| 3 | `ceil(40/8.35) = 5`; used `40/5 = 8.0`; remainder `0.35`; `Σ = 15`; `15f ≤ 0.35×60 = 21` → `f = 1.4`; `min(60, round(60/1.4)) = min(60, 43) = 43`; phases `1,2,3` | same | ✓ |
| 4 | `2 ≤ 8.35` → `k=1`; remainder `6.35`; `Σ = 20`; `20f ≤ 381` → `f = 19.05`; `min(60, round(60/19.05)) = min(60, 3) = 3`; phases `i%3` | same | ✓ |
| 5 | `4 ≤ 8.35` → `k=1`; remainder `4.35`; `100f ≤ 261` → `f = 2.61`; `min(60, 23) = 23` | same | ✓ |
| 6 | throttle off → all `k=1`, phase 0 | same | ✓ |
| 7 | `budget×period = 16.7`; `ceil(40/16.7) = ceil(2.395) = 3`; fps 20 | same | ✓ |
| 8 | no record → all `k=1` | same | ✓ |
| 9 | `k_cur = 5`, remainder `0.35`; `Σ = 50`; `50f ≤ 21` → `f = 0.42`; `round(60/0.42) = 143`, `min(60, 143) = 60`; fps `60/60 = 1.0` | same | ✓ |

All nine exact, including every intermediate remainder, `f`, and phase.

**The eleven-file table — every value checked against the file.** `git ls-files | grep
'document.json$'` returns exactly eleven, matching the table's paths one for one, and every stated
`resolution` equals the file's current `canvas_size`:

```
e7e00c46 [1280,960]  ec926580 [1280,960]  1901ab60 [960,960]   307598da [1280,960]
0b0d16bb [1080,1920] 53724dbd [1280,960]  73ea2431 [1280,1280] 77a84d27 [512,512]
8d454b7b [1600,900]  f90f5ff9 [1280,960]  bloom_chain [960,960]
```

Blast F10's count correction (eleven, not eight) is right — I missed `projects/documents/`'s two
and the `tests/fixtures/bloom_chain` one in round 1.

**The six test files.** `grep -ln '"canvas_size"' tests/*.py` returns exactly the six the spec
names — `test_feedback_persistence.py`, `test_document_dir_sync.py`, `test_graph_persistence.py`,
`test_canvas_presets.py`, `test_pass_hot_reload.py`, `test_uniform_row_pruning.py`. Not a file
more or fewer.

**V12's silent-failure premise.** Verified end to end: `constants.py` has `DEFAULT_CANVAS_SIZE =
(64, 64)`; `UIDocumentState` carries no `model_config`, so unknown keys are pruned by
`_load_ui_state`'s filter with a warning while a *missing* known key just takes its default with no
warning at all. So a forgotten file loads at 64×64 in silence, exactly as claimed. V12 is the right
gate and "Ships in the same commit as the sweep" matches the global rule that a cleanup ships with
the check preventing its recurrence.

**The xdist correction.** Checked: `pytest.mark.xdist_group` appears in `test_code_panel.py` and
`test_project_management.py` only; `pyproject.toml` registers the marker. Of the four test files
whose text contains `update_and_draw`, `test_model_salvage.py` only mentions it in a comment, so
**three** files drive it — `test_code_panel.py`, `test_project_management.py`, `test_profiling.py` —
and `test_profiling.py` carries no mark. The spec's "three files drive it today and
`test_profiling.py` carries no mark" is exactly right, and the `conventions.md` quirk that says one
named slot-holder is genuinely stale. Correcting a convention bullet this wave is the right
altitude.

**The two existing-test claims.** Both verified.
`test_render_for.py::test_render_media_preset_none_byte_identical` compares two exports *to each
other*, so it passes whichever size they share — the spec's note that "no existing gate catches
this" is correct, and V5a is needed. `test_document_ops.py::test_set_canvas_size_applies_and_clamps`
asserts only the resulting texture size and the clamp, nothing about mode, so V13's extension is
needed too.

**Conventions, re-checked.** Full annotations on every signature (`resample_canvas`,
`apply_damping`, `auto_canvas_size`, `plan_render_set`, `throttle_color`, `fps_overlay`,
`profile_rows_plan`) — OK. No `if TYPE_CHECKING`; `render_plan.py` stays leaf and `apply_damping`'s
move into it *strengthens* that (a pure function over `AutoSizeState` and two tuples, no `App`) —
OK. Imports at top — nothing asks otherwise. No `@staticmethod`: `resample_canvas` uses `self._gl`,
the rest are module-level free functions — OK. No-migration: still clean, and stronger now that
`resolution` keeps each file's existing number rather than deriving a new one; `load_from_dir`
"reads no top-level `canvas_size` and no `ui_state` key" — OK. One canonical home: both round-1
violations are gone (F8's three sites → one, F11's duplicate fields → deleted), and the new text
adds the rule explicitly ("one reader per concept"). UI through `ui_primitives`/`theme`:
`throttle_color` lands in `theme.py` beside `load_color`, `drawn_size` on `PreviewCellResult`, the
settings snippet unchanged and still matching the existing idiom — OK. Prose budget: unchanged
labels — OK. Constraints on the model: `Field(ge=0.1, le=1.0)` retained — OK. Cite-by-symbol: the
spec cites symbols throughout, no line numbers — OK.

## New findings

### MINOR R1 — the hand-edit table and the prose under it disagree on which documents are Fixed

**Spec passage.** The eleven-row table's `mode` column reads `auto` for **every** row, including
`…/77a84d27-…`. Immediately below it: "The JFA and radiance-cascade examples among the six are the
`fixed` ones — their iteration counts follow pixel counts (`resolution_flow §4`); the implementer
identifies them by shader content." *Out of scope* says the same: "The JFA and radiance-cascade
examples are hand-set to Fixed."

**Evidence.** Two problems, one textual and one factual.

The table says `auto` where the prose says `fixed`; an implementer working from the table — which
is the actionable artifact, being the per-file list — ships eleven Auto documents and silently
breaks the one document whose shaders are pixel-dependent. That is the failure the Out-of-scope
bullet exists to prevent.

And "examples" plural is an over-count. The JFA and cascade shaders are two *passes of one
document*: `ls shaderbox/resources/document_examples/*/passes/` shows `77a84d27-…` holding
`cascade.frag.glsl`, `composite.frag.glsl`, `df.frag.glsl`, `jfa.frag.glsl`, `paint.frag.glsl`,
`seed.frag.glsl`, while the other five examples hold a lone `main.frag.glsl`. `resolution_flow §4`
says so in its own count: "2 pixel-dependent document-shader files (JFA + cascade, both in ONE
example document, `77a84d27...`)". So "the `fixed` ones" is one document, not several, and
"the implementer identifies them by shader content" is a search for something the research already
identified by id.

**Fix.** Set the `77a84d27` row's mode to `fixed` in the table and replace the prose with the id:
"`77a84d27-…` is the one `fixed` document — its JFA and radiance-cascade passes have
iteration counts that follow pixel counts (`resolution_flow §4`)." That also lets V12 assert the
mode per file rather than only asserting the key exists.

### MINOR R2 — D4's safety argument covers one caller of `set_canvas_size`, and there are three

**Spec passage.** D4: "Ordering makes it safe: the resize runs in `_tick_frame_state` before the
draw phase, so no imgui draw list holds a released texture — the 084 D5 hazard the project switch
defers for."

**Evidence.** The claim is true of the Auto path, which is step 4/5 of *Frame integration*. It is
not true of every caller of the funnel the resample now lives in. `grep -rn 'set_canvas_size('`
returns three call sites:

- `ui.py`'s new step 4 — inside `_tick_frame_state`, before the draw. Safe, as claimed.
- `copilot/backend.py::set_canvas_size` — reached through `run_on_main`, drained by
  `app.copilot.drain_bridge()` at the head of `_tick_frame_state`. Also before the draw. Safe.
- `tabs/document.py::_apply_canvas_size` — called from the Document tab's W × H commit
  (`if committed_w or committed_h: _apply_canvas_size(...)`), which runs **inside the draw phase**.
  And it runs *after* the viewer: `_draw_app_panel` calls `_draw_document_image` — which pushes
  `render_pass.canvas.texture.glo` into the frame's draw list via `imgui.image_with_bg` — and the
  Document tab's `draw` is one of the tab bodies drawn below it in the same panel.

So a W × H commit releases a texture imgui is already holding for this frame. This hazard exists
today (`Canvas.set_size` already releases the output canvas there), so the spec does not create it
— but D4 widens the blast radius from one canvas to the output canvas plus every feedback history,
and states a blanket safety claim that the reader will take as covering the feature. A reader who
later adds a fourth caller has been told the ordering is handled.

**Fix.** One clause scoping the claim: "…the resize runs in `_tick_frame_state` before the draw
phase for the Auto path and the copilot's bridge. The Document tab's own W × H commit already
resizes inside the draw (pre-existing); the resample inherits that exposure and does not widen the
window per call." If the maintainer would rather close it, the tab's commit sets a pending size
that step 4 consumes — the same shape `pending_project_switch` uses — but that is a scope call, not
something this review asks for.

## False trails — probed in round 2, fine, do not re-spend

- **All nine worked examples.** Recomputed by hand including example 9's new cap row and every
  phase. Exact. Do not re-derive.
- **`throttle_color`'s bands against the spec's own cases.** Computed: 1.000 / 0.960 / 0.720 all
  land `STATE_OK`. The fix genuinely inverts round 1's defect.
- **The eleven-file table's paths and values.** Checked against `git ls-files` and each file's
  JSON. Every path and every number is right.
- **The six test files.** `grep -ln '"canvas_size"'` returns exactly those six. Not five, not seven.
- **The xdist correction and the three `update_and_draw` drivers.** Checked; `test_model_salvage.py`
  is a comment-only mention, so "three" is right and my first grep over-counted.
- **`resolve_dims`'s `FREE` fall-through returning `src_w, src_h`.** Read in `render_preset.py`;
  the blast F1 premise holds and so does the fix.
- **`_artifact_matches_shape` passing the live canvas size.** Read in `exporters/youtube.py`; it
  gates `upload_enabled`, so the disarm-on-panel-move failure is real.
- **The five aspect sites.** Enumerated by grep; five real, `core.py`'s `u_aspect` correctly
  excluded, none missed.
- **`PreviewCellResult`'s current fields and `draw_document_preview_button`'s two callers.** Both
  as the spec describes; the `app`-free decision is sound.
- **`DEFAULT_CANVAS_SIZE = (64, 64)` and `UIDocumentState`'s missing-key silence.** Verified; V12's
  premise is exact.
- **The two existing-test claims** (byte-identical passes either way; the copilot test asserts no
  mode). Both read in full; both correct.
- **`render_plan.py`'s leaf claim after `apply_damping` moved in.** Every parameter is a primitive,
  a tuple or an own dataclass. Still leaf.

**Verdict: PASS-WITH-MINORS** — every round-1 finding closed, most structurally; R1 (table vs prose
on the one Fixed document) should be fixed before implementation since the table is what an
implementer will follow, and R2 is a one-clause scoping of a safety sentence.
