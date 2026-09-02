# 069 W-A post-implementation code review — correctness

Commit under review: `78bd1bf` ("069 W-A: canvas size fields, presets, viewer backdrop").
Role: bugs, races, GL-context lifecycle, imgui frame-order hazards, error handling.

## Verdict

| Area | Verdict |
|---|---|
| Frame order | **PASS** — traced through a real headless imgui frame; the mirror, the OR-commit and Escape all behave. One race, filed as F4. |
| Disabled scope | **PASS** |
| Funnel + clamp | **FAIL** — the early return is defeated on every disk-loaded document (F2). |
| Presets | **FAIL** — hard crash on every disk-loaded document (F1); dead current-size entry at six sizes (F3). |
| Backdrop | **FAIL** — 3.8-9.8 ms/frame, measured, spent every frame (F5). |
| Persistence | **PASS** |
| Tests | **PARTIAL** — 3/3 named falsifiers reproduce red, but the suite is green while F1/F2/F3 are all live. |
| Conventions | **PASS** — one dead token and one stale-format file, both minor (F6, F7). |

`make gates` re-run here: **exit 0, smoke passed** — the commit's claim is accurate. The gates
simply do not reach the crashing path; no test constructs a document through `load_from_dir`.

---

## F1 — `_canvas_presets` raises `TypeError` on every document loaded from disk (CRASH)

**Claim.** Opening the presets dropdown on any real document takes the app down.

**Evidence.** `document.json` stores `canvas_size` as a JSON **list**. `_load_document_metadata`
returns it raw and `Document.__init__` assigns it unconverted (`document.py`,
`self.canvas_size: tuple[int, int] = canvas_size or DEFAULT_CANVAS_SIZE`) — the annotation says
tuple, the runtime value is a list. `_canvas_presets` then does `seen: set[tuple[int, int]] = {current}`:

```
E       TypeError: unhashable type: 'list'
shaderbox/tabs/document.py:59: TypeError
```

Reproduced two ways against the real `app` fixture: calling `_canvas_presets(ui_doc)` directly,
and driving the real `tabs/document.py::draw` for a frame with `begin_combo("##canvas_presets")`
stubbed to report open (what a real click does). Both raise. Every document on disk is affected:

```
projects/dev/documents/e7e00c46-.../document.json  [1280, 960]
projects/dev/documents/ec926580-.../document.json  [1280, 960]
```

This raises inside the imgui frame body, so per `/imgui-ui` §4 it also leaves the imgui stack
unbalanced and surfaces as a confusing downstream assert.

Why the suite misses it: `test_canvas_presets.py::_document` builds `Document(gl=gl, canvas_size=size)`
with a literal **tuple**. No test in the file goes through `load_from_dir`.

**Fix.** Normalize at the funnel: make `Document.__init__` and `Document.set_canvas_size` coerce
to `tuple(...)`, so `canvas_size` is a tuple by construction regardless of what the loader hands
in. Add a preset test built from the `app` fixture (a real disk-loaded document), not a
hand-constructed one.

## F2 — `_apply_canvas_size`'s early return never fires on a disk-loaded document

**Claim.** The "unchanged size" guard is defeated, so a no-op commit pushes a spurious toast.

**Evidence.** Same root as F1: `(w, h) == ui_document.document.canvas_size` compares a tuple to a
list, which is always False. Against the `app` fixture:

```
type: <class 'list'> [1280, 960]
pushes after RE-applying the SAME size: ['Canvas: 1280x960']
type after: <class 'tuple'>
pushes on the second identical apply: []
```

It self-heals after the first commit (because `set_canvas_size` stores the tuple), so it fires
exactly once per document per session — the shape that is hardest to notice. It is user-visible:
typing into W and pressing **Escape** on a freshly loaded document toasts a canvas change that
did not happen. Traced through a real frame sequence (focus W, type `5`, Escape):

```
ESCAPE on a freshly loaded document -> pushes: ['Canvas: 1280x960']
```

**Fix.** The same tuple-normalization as F1 closes this; no separate change needed.

## F3 — the video-shape loop bypasses `seen`, so the current size is offered as a dead entry

**Claim.** At any of the six video-shape sizes, the presets menu lists the size the document is
already at — the exact defect `test_no_preset_duplicates_the_current_size` claims to prevent.

**Evidence.** The square loop and the texture loop both `continue` on `if size in seen`. The
video-shape loop appends unconditionally, adding to `seen` only afterwards
(`tabs/document.py::_canvas_presets`). Measured on real documents:

```
(1280, 720)  -> duplicates of current: [('Wide 720p (16:9)', (1280, 720))]
(720, 1280)  -> duplicates of current: [('Short 720p (9:16)', (720, 1280))]
(2560, 1440) -> duplicates of current: [('Wide 1440p (16:9)', (2560, 1440))]
(512, 512)   -> duplicates of current: []
```

The entry is dead by construction: `_apply_canvas_size` early-returns on it (once F2 is fixed),
so clicking it does nothing at all.

The guarding test uses `size=(512, 512)` — a **square**, which the square loop's own skip already
handles. It passes for a reason other than the behaviour it names.

**Fix.** Add the `if size in seen: continue` guard to the video-shape loop, matching its two
siblings, and re-point the test at a video-shape size (e.g. 1280x720) so it exercises the loop
that was actually broken.

## F4 — a copilot write during an active field is silently reverted, including the untouched half

**Claim.** With W active, a copilot `set_canvas_size` reaches the document but is overwritten when
the user leaves the field — and the H the user never touched is clobbered with a stale value.

**Evidence.** Traced through real frames (focus W, type `5`, then `set_canvas_size((800, 600))`
externally at f4):

```
f3 W=(1280,F,5,active,-) buf=(5, 960) ed=True doc=(1280, 960)
f4 W=(5,F,5,active,-)    buf=(5, 960) ed=True doc=(800, 600)     <- copilot write lands
f5..f7                    buf=(5, 960) ed=True doc=(800, 600)     <- fields still show the old pair
```

The mirror is gated on `if not app.canvas_size_editing`, so while W is active the buffer keeps
`(5, 960)` — the stale 960 for H included. When the user then leaves the field, the pair-commit
writes `clamp(5, 960) = (16, 960)`, reverting the copilot's 800x600 and discarding a value the
user never edited.

`begin_disabled(app.copilot_turn_active)` narrows but does not close this: `begin_disabled`
prevents new interaction, it does not deactivate an already-active item, and a copilot write can
also arrive via the disk-sync path with no turn active.

Scope note: this is a genuine race, not a crash, and it is narrower than F1-F3. Worth deciding
deliberately rather than fixing reflexively.

**Fix.** On commit, write only the field that committed and re-read its sibling from
`document.canvas_size`, so an untouched half can never carry a stale value over an external write.

## F5 — the checkerboard costs 3.8-9.8 ms every frame, measured

**Claim.** The backdrop is a per-frame cost large enough to matter, paid unconditionally.

**Evidence.** `_draw_canvas_backdrop` issues one Python-to-C++ `add_rect_filled` per dark cell,
every frame, for the whole image rect. Counted from `SIZE.CHECKER_TILE = 12`:

```
1600x900:  grid 134x76  =  5,092 add_rect_filled calls/frame
2560x1400: grid 214x117 = 12,519 calls/frame
```

Timed against a real imgui frame (mean of 5):

```
800x600:   1.32 ms/frame
1600x900:  3.83 ms/frame   (23% of a 60fps budget)
2560x1400: 9.76 ms/frame   (58% of a 60fps budget)
```

Every one of those rects is invisible under any opaque shader, which is the common case.

The commit's stated reason for rects over a texture — "no GL object to create, release and keep
out of the export path" — is sound about lifetime and wrong about cost.

**Fix.** Draw the checkerboard once into an offscreen list or a small repeat texture, or emit a
single `add_rect_filled` per **row-run** instead of per cell (halves nothing) — better, gate the
whole backdrop on whether the output actually has alpha below 1, so an opaque render pays nothing.
Correct as drawn otherwise: order is backdrop, image, border; the clip via `min(...)` handles the
partial trailing cell; it covers the image rect only, not the panel.

## F6 — `SIZE.RES_COMBO_W` is now dead

`theme.py:234` defines `RES_COMBO_W: int = 200`; after this commit its only remaining references
in the tree are inside the comment at `tabs/document.py:31-33`. The comment's arithmetic is
correct (56+4+7+4+56+8+64 = 199 <= 200), but a token nothing reads is a token that will drift.

**Fix.** Either use it (`_CANVAS_FIELD_W` etc. derived from it, which is what the comment claims)
or delete it.

## F7 — the committed `test_canvas_presets.py` is not in the repo's own format

`make gates` rewrites the file on every run (ruff splits the `render_shape` import into a
parenthesized block), leaving the tree dirty after a green gate.

**Fix.** Commit the formatted version.

---

## Item-by-item against the brief

2. **`is_item_active()` / `is_item_deactivated_after_edit()` placement** — PASS. Each is read on
   the line immediately after its own `input_int`, before the `same_line`. Confirmed empirically by
   spying on `imgui.input_int` and logging both queries per frame: the flags track the right item
   throughout every trace. `same_line` is not an item and does not disturb the item-scoped queries,
   but here the reads precede it anyway, so the question does not arise.

3. **`begin_disabled` pair** — PASS. One `begin_disabled` / one `end_disabled`, no early return and
   no exception path between them (`_canvas_presets` can raise — F1 — but that unbalances the frame
   regardless of the disabled scope). `end_combo` is inside the `if imgui.begin_combo(...)`, so an
   unbalanced call is structurally impossible. Verified live: 3 frames drawn with
   `copilot_turn_active = True`, no stack imbalance.

4. **`_apply_canvas_size`** — clamp bounds `MIN_CANVAS_PX = 16` / `MAX_CANVAS_PX = 4096` are single-homed
   in `pass_graph.py` and read by both entry points (`tabs/document.py` and
   `copilot/backend.py::set_canvas_size`, which now imports `clamp_canvas_size` instead of its own
   copy) — the funnel is real. The early return is broken (F2). `set_canvas_size` raising is not a
   live risk: the clamp bounds it and `Canvas.set_size` guards the unchanged case. The buffer
   re-read after commit is correct and confirmed in every trace.

5. **`_canvas_presets`** — no compile (falsifier re-run below confirms the pin). A `moderngl.Buffer`
   in `uniform_values` is correctly excluded: `isinstance(value, MediaWithTexture)` does not admit
   it. A bound `Video` is admitted and its `.texture` property decodes a frame on first access —
   correct, though it means opening the dropdown can force a video decode; acceptable, and it is
   the same decode the preview would do. With no bound texture the menu shows squares plus video
   shapes only, which reads fine. Labels are within the D1 budget (`512x512 (1:1)`, `Wide 1080p (16:9)`).
   Dedupe order is squares then shapes then textures, with the shape loop broken (F3).

6. **Checkerboard** — F5.

7. **`UIDocument.save` reading `document.canvas_size`** — PASS. `load_from_dir` reads the same key
   back into `Document(canvas_size=...)`, and per-pass metadata carries only `scale` (derived from
   the document canvas), never its own size — checked against `projects/dev/documents/*/graph.json`.
   No round-trip where the saved field and the output texture disagree.

8. **Tests** — three named falsifiers re-run, each restored and verified with `git diff --quiet`
   before the next command:

   | Falsifier | Result |
   |---|---|
   | drop 512 from `_SQUARE_PRESETS` | `test_the_square_presets_include_512` **FAILED** as claimed |
   | `get_active_uniforms()` inside the scan | `test_building_the_presets_compiles_nothing` **FAILED** as claimed |
   | `seen = set()` (drop the current-size skip) | `test_no_preset_duplicates_the_current_size` **FAILED** as claimed |

   All three reproduce. The third is nonetheless weak: it only reaches the square loop, and the
   suite stays green (6 passed) while the video-shape half of the same rule is live-broken.

9. **Conventions** — PASS. No `# noqa` / `# type: ignore` / `# pyright: ignore`, no inline imports,
   no `Any`, no `@staticmethod`, no hand-rolled `push_style_color`. All colours go through `COLOR.*`
   tokens and the tile size through `SIZE.CHECKER_TILE`. The `app.py` comment "Never a `| None`
   latch -- nothing would clear it" narrates a rejected alternative rather than the code as it is —
   a mild instance of the history-narration rule, not worth a change on its own.

## False trails

- Video-shape presets exceeding the 4096 clamp: they do not — the largest is 2560.
- `Short 1080p` resolving to 1088x1920 rather than 1080x1920: correct, `_align` rounds to the video alignment.
- `input_int(step=0)` rendering stepper buttons: it does not; `step<=0` gives a bare input.
- Double-commit when `enter_returns_true` and `is_item_deactivated_after_edit()` coincide: traced,
  the commit fires exactly once (one toast per Enter, one per click-away).
- Escape leaving a half-typed buffer: it does not — imgui restores the pre-edit value and the
  buffer follows it. (The spurious toast on Escape is F2's early return, not Escape's handling.)
- Ctrl+N mid-edit: `_on_current_document_changed` clears `canvas_size_editing`, so the next frame
  mirrors the new document. Correct as written.
- `begin_combo` inside `begin_disabled` returning True and unbalancing `end_combo`: it does not.
- Backdrop drawing behind the whole panel: it does not — the rect is the image rect exactly.
- The commit's "`make gates` GREEN with smoke passed" claim: re-run here, exit 0, smoke passed. True.

## Coverage

Read end-to-end: `shaderbox/tabs/document.py`, `shaderbox/app.py` (the changed regions plus
`_on_current_document_changed`), `shaderbox/ui.py` (`_draw_canvas_backdrop`, `_draw_document_image`),
`shaderbox/theme.py`, `shaderbox/pass_graph.py`, `shaderbox/popups/pass_settings.py`,
`shaderbox/copilot/backend.py` (the changed regions), `shaderbox/ui_models.py::UIDocument.save`,
`tests/test_canvas_presets.py`, `tests/test_document_graph.py` (the added block). Supporting reads:
`document.py` (`__init__`, `set_canvas_size`, `render`, `load_from_dir`, `_load_document_metadata`),
`core.py` (`Canvas.set_size`, `Pass.get_active_uniforms`, `uniform_values`), `media.py`
(`is_default_image`, `Image.texture`, `Video.texture`), `render_shape.py`, `render_preset.py`
(`resolve_dims`, `_align`), `util.py::get_resolution_str`.

Frame-order claims were established by driving the real `tabs/document.py::draw` in a headless
imgui frame against the `app` fixture (the `tests/test_lib_files.py` rig pattern), spying on
`imgui.input_int` to log the input value, the return, `is_item_active()` and
`is_item_deactivated_after_edit()` per frame — not by reading the code. All probe files were
deleted; `git status --short` is clean.

Not covered: the visual appearance of the checkerboard and border (needs a maintainer `make run`
per `/imgui-ui` §0), and W-B's help-text scope, which this commit explicitly leaves alone.
