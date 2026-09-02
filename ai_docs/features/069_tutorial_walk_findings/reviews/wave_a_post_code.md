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

---

# Round 2 (closure) — against `3910900`

Narrow closure round on the fix-up commit `3910900` ("069 W-A fixes: canvas_size tuple,
per-field buffer, checker"). Scope: F1..F5 only, plus the two minor items F6/F7.

**Overall: PARTIAL.** F1, F2, F3, F5, F6, F7 are CLOSED. F4 is CLOSED for the case it was
filed on and leaves one narrower residual (R2-1). One test-coverage gap (R2-2). Neither is a
crash; both are below the bar of the round-1 findings.

`make gates` re-run: **exit 0, smoke passed**, and the tree stays clean afterwards.

| Finding | Verdict |
|---|---|
| F1 presets crash on a disk-loaded document | **CLOSED** |
| F2 early return defeated | **CLOSED** |
| F3 shape loop ignores `seen` | **CLOSED** |
| F4 stale half reverts an external write | **CLOSED** (residual R2-1) |
| F5 checkerboard per-frame cost | **CLOSED** |
| F6 `SIZE.RES_COMBO_W` dead | **CLOSED** |
| F7 committed file not in repo format | **CLOSED** |

## F1 — CLOSED

`document.py::_as_canvas_size` normalizes at both writers: `Document.__init__`
(`self.canvas_size = _as_canvas_size(canvas_size) or DEFAULT_CANVAS_SIZE`) and
`set_canvas_size` (same coercion). `load_from_dir` now passes `document.canvas_size` — the
already-normalized field — down to each `Pass`, rather than re-reading the raw metadata.

Verified against the real `app` fixture (a disk-loaded document, no normalization in the test):

```
F1 canvas_size type: tuple (1280, 960)
F1 presets built OK: 10 entries
```

Malformed pairs degrade rather than raise, which matches the loader's fail-soft posture:
`[1280]`, `[1280, 960, 4]`, `"1280x960"`, `[1280.5, 960]`, `None`, `{"w": 1}`, `[None, None]`
all return `None` → `DEFAULT_CANVAS_SIZE`. `[true, true]` passes the `isinstance(w, int)` check
(bool subclasses int) and yields `(True, True)`, which `clamp_canvas_size` bounds to `(16, 16)` —
harmless, and not a value a real file produces.

## F2 — CLOSED

The early return now compares tuple to tuple. On a disk-loaded document, re-applying the same
size pushes nothing, and Escape after typing no longer toasts a resize that did not happen:

```
F2 re-applying the SAME size on a disk-loaded doc -> pushes: []
F2 Escape on a freshly loaded doc -> pushes: [] doc: (1280, 960)
```

## F3 — CLOSED

`tabs/document.py` — the shape loop now carries `if size in seen: continue` before `seen.add`
and the append, matching its two siblings. At all four probe sizes, including the three
video-shape ones that were broken:

```
(1280, 720)  -> duplicates of current: []
(720, 1280)  -> duplicates of current: []
(2560, 1440) -> duplicates of current: []
(512, 512)   -> duplicates of current: []
```

The test now covers 1280x720, which exercises the loop that was actually broken.

## F4 — CLOSED, with residual R2-1

Two changes: the mirror is per field (`canvas_w_editing` / `canvas_h_editing` replace the single
`canvas_size_editing`, each half mirrored on its own flag), and each commit pairs its pending
half with `ui_document.document.canvas_size`'s live other half.

The four requested traces, run against the real disk-loaded document through
`tabs/document.py::draw`:

**Copilot write while W is active — the case F4 was filed on. FIXED.**

```
f3 W=(1280,5,active) H=(960,960)   buf=(5, 960) doc=(1280, 960)
f4 W=(5,5,active)    H=(600,600)   buf=(5, 600) doc=(800, 600)   <- write lands; H mirrors AT ONCE
f7 W=(5,-,deact)     H=(600,600)   buf=(16,600) doc=(16, 600)    <- commit keeps the copilot's 600
>>> final doc: (16, 600)
```

Round 1's clobber is gone: the height the user never touched is the copilot's 600, not the
pre-edit 960.

**Tab from W to H. PASS.** Commit fires once at the transition frame, H becomes active in the
same frame, no double toast:

```
f6 W=(5,5,deact=True) H=(960,960,active) buf=(16,960) doc=(16,960) p=['Canvas: 16x960']
f7..f8 unchanged, p unchanged
```

**Escape in W. PASS.** imgui restores the pre-edit value, the buffer follows, no commit, no toast.

**Ctrl+N (document switch) while W is active. Residual — see R2-1.**

### R2-1 — a document switch mid-edit carries the half-typed width into the NEW document

`_on_current_document_changed` clears both editing flags, but the imgui item is still physically
active, so `is_item_active()` reads True again on the very next draw and the W half re-latches the
old document's pending digit. The H half mirrors correctly.

Demonstrated with two documents at different sizes (original 1280x960, other 333x444; type `5`
into W on the original, switch at f4, click away at f6):

```
  [f4 pre] switched; wE=False hE=False buf=(5, 960)
f4 cur=bbbbbbbb buf=(5, 444) wE=True   <- H mirrored to the new doc; W re-latched the stale 5
...
orig  : (1280, 960)   (untouched, correct)
other : (16, 444)     (was 333x444 -- resized by a digit typed into a different document)
pushes: ['Canvas: 16x444']
```

Narrower than the original F4 (which reverted any external write on the untouched axis); this
needs a document switch during an active edit. The original document is correctly left alone.

**Fix.** Clearing the flag is not enough while the item stays active — on a document change also
re-seed `canvas_size_buf` from the incoming document and call
`imgui.set_keyboard_focus_here(-1)` (or clear the active id) so the field is genuinely
deactivated, not merely un-flagged.

### R2-2 — the two halves of the F4 fix are not independently pinned

`tests/test_canvas_fields.py` stays green when the commit-pair change alone is reverted:

```
commit-pair reverted to the buffer pair:            2 passed
per-field MIRROR also reverted (the full R1 bug):   1 failed, 1 passed
```

The per-field mirror is the load-bearing half; the commit-side re-read of the document's live
other half is defensive redundancy the suite does not exercise. Not a code defect — the fix is
correct and over-determined — but a regression that removed only the commit-pair re-read would
ship green.

**Fix.** Add a case where the two disagree at commit time and only the commit-side re-read can
resolve it, or drop the redundant half and let the mirror carry the property alone.

## F5 — CLOSED

One `add_image` of a 2x2 NEAREST/repeat texture (`ui.py::_draw_canvas_backdrop`).

**Cost, re-measured (mean of 50 calls per size):**

| Viewer | Round 1 | Round 2 |
|---|---|---|
| 800x600 | 1.32 ms | 0.0013 ms |
| 1600x900 | 3.83 ms | 0.0010 ms |
| 2560x1400 | 9.76 ms | 0.0010 ms |

Flat in viewer size, as claimed — roughly 3800x cheaper at 1600x900.

**UV math correct.** `pair = 2 * SIZE.CHECKER_TILE`; `uv1 = (w/pair, h/pair)` puts exactly one
2x2 texel pair per 24 px, so one cell lands on 12 px at any viewer size (checked at 1600x900 and
2560x1400: 12.0 px per cell both).

**GL lifetime, all three questions:**

- *Created after the context exists.* `glfw.make_context_current` runs in `App.__init__`
  (app.py:183); `_make_checker_texture()` is called from `_init` (app.py:1131), reached at
  app.py:454 — after. Beside `preview_canvas = Canvas()` on the preceding line, as specified.
- *Released before the context is destroyed.* `App.release` releases it (app.py:1662) under a
  `hasattr` guard, immediately after `preview_canvas.release()`; `ui.py:150` calls `app.release()`
  at shutdown. `_init` calls `self.release()` first, so a project reopen releases the old texture
  before creating the new one — no leak across project switches. The only `glfw.terminate()` is
  the window-creation failure path, before any texture exists. Double release is safe (moderngl's
  `Texture.release` is idempotent; verified directly).
- *Not recreated per frame.* `_make_checker_texture` has exactly one call site. Live check across
  5 drawn frames: `glo` = `[2, 2, 2, 2, 2]`, stable; `size=(2,2)`, `filter=(9728, 9728)` (NEAREST
  both), `repeat_x=repeat_y=True`.

The border colour change is a real fix on the way past: `COLOR.BORDER` is `_P["bg_2"]`, which is
`CHECKER_LIGHT` — the outline vanished along every light square. `COLOR.VIEWER_BORDER` (`bg_4`)
reads against both greys. Not verified visually (needs a maintainer `make run`).

## F6 — CLOSED

`grep -rn "RES_COMBO_W" shaderbox/` returns nothing; the token is deleted from `theme.py` and
replaced by `SIZE.CANVAS_FIELD_W` / `SIZE.CANVAS_PRESETS_W`, both read by `tabs/document.py`.
Remaining hits are in docs and these reviews.

## F7 — CLOSED

`make gates` leaves the tree clean; the previously drifting `test_canvas_presets.py` import block
is committed in the repo's own format.

## Conventions (re-check on the fix-up diff)

PASS. No suppressions, no inline imports, no `Any`, no `@staticmethod`, no hand-rolled style
pushes. The new colour goes through a `_ColorBag` token mapped to a `_P` entry, the two widths
through `SIZE`. `_make_checker_texture` is a module-level free function, correctly not a method.
The round-1 note about the `| None` latch comment is resolved — the comment was rewritten to
describe the per-field rule as it now is, with no rejected-alternative narration.

## False trails (round 2)

- `_as_canvas_size` accepting `[true, true]`: real, bounded to (16,16) by the clamp, not reachable
  from a real file.
- `end_combo` in the wrong window: an artifact of stubbing `begin_combo` to return True in a probe,
  not a defect. The real draw path is balanced.
- The checker texture leaking across project switches: it does not — `_init` releases first.
- The `add_image` UV drifting the cell size at large viewers: it does not, 12.0 px at every size.

## Coverage (round 2)

Read: the whole `3910900` diff for `shaderbox/` end-to-end, plus `tests/test_canvas_fields.py` and
the changed regions of `tests/test_canvas_presets.py`. Re-read `App.__init__` / `_init` / `release`
ordering and `ui.py`'s shutdown call for the F5 lifetime questions.

Verified by execution, not by reading: the F1 crash path and malformed-pair degradation, the F2
early return and the Escape trace, the F3 duplicate scan at four sizes, all four F4 traces plus the
two-document R2-1 case, the F5 timing at three sizes, the texture's `glo`/filter/repeat and its
release idempotence, and two mutation runs establishing R2-2. All probe files deleted;
`git diff --quiet` passes and `git status --short` shows only this review plus two untracked
wave-B files belonging to another agent
(`30_wave_b_prose_diet.md`, `reviews/wave_b_pre.md`).

Not covered: the visual reading of the checkerboard and the new border colour (a maintainer
`make run` call, per `/imgui-ui` §0).
