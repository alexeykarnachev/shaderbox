# 090 — Document throttle and Auto / Fixed resolution

Implements `02_throttle_and_resolution.md` (**D1–D11, locked 2026-09-11 — fixed premises**) after
`00_research.md`'s tiling rejection. Measurements: `research/cost_and_throttle.md`. Code maps:
`research/resolution_flow.md`, `research/cadence_flow.md`. Round-1 reviews:
`reviews/pre_correctness_design.md`, `reviews/pre_verification_blast.md` (both FAIL; triage in
*Review history*).

Size: **high-blast-radius**. 2 pre-impl reviewers (done), 3 post plus a spec-fidelity pass, and a
sanitization sweep.

---

## Goal

A document renders at the size it is displayed at, and one costing more than its share of the
frame renders less often. On the measured 100 ms document the hitch falls from ~105 ms to ~40 ms
and lands on 60 of 300 frames instead of 300, while its own fps holds at 9.5 (`§3`).

- Each document carries a **resolution mode**, Auto or Fixed, and **one** stored width × height.
- Fixed: that number is the live size, exactly today. Auto: the live target follows the largest UI
  region showing the document, and the stored number is the document's **export resolution** —
  what `RenderShape.NATIVE` and `resolve_dims`'s `FREE` fall-through resolve to.
- The Render tab's own W × H still decides what lands on disk, exactly as today.
- Feedback survives an Auto resize by **resampling**, live canvas and history together.
- One **GPU budget** shared by every displayed document decides each one's interval `k`.
- The FPS chip and panel report the plan; the profiler records GPU spans **always**.

---

## Out of scope

- **Tiling.** Rejected. Trigger: none.
- **The GL render thread** (`00_research.md ## Recommendation` item 1) — the only thing that makes
  a document-carrying frame stop blocking the UI. Trigger: the maintainer asks for input latency
  independent of the document after this lands.
- **Resolution-independence shims for pixel-dependent shaders.** One example document,
  `77a84d27-…`, is hand-set to Fixed. Trigger: a document that must be Auto and pixel-exact.
- **A CPU throttle.** D7's record carries the CPU field; no policy reads it. Trigger: a script
  tick measured above a few ms per frame.
- **Per-pass Auto sizing.** A non-output pass keeps scaling through `TargetConfig.target_size`; a
  pass-strip tile is not a recorder. Trigger: a pass whose only consumer is its own tile.
- **Throttling exports and the copilot probe.** Both call `Document.render` outside
  `_tick_frame_state` and take `NULL_PROFILER` (`cadence_flow §3`). Trigger: an export that must
  respect a live budget.

---

## Design decisions

Locked in `02`; restated as they must be implemented. D3's damping, D6's hysteresis and D6's
budget default were the implementer's and are picked here.

**D1 — `ResolutionMode` is persisted per document; ONE stored size; `canvas_size` becomes the
effective live size.** A `ResolutionMode` StrEnum (`AUTO` / `FIXED`) lands in `render_shape.py`,
GL-free. `UIDocumentState` gains exactly two fields:

```python
resolution_mode: ResolutionMode = ResolutionMode.AUTO
resolution: tuple[int, int] = DEFAULT_CANVAS_SIZE
```

`resolution` is the one stored width × height, spelled that way everywhere. Its ROLE follows the
mode: Fixed → the live size; Auto → the export resolution (D5), with the live target following the
display (D2). There is no second size field — round 1's `export_size` and its derivation rule are
gone (closes correctness F5).

`Document.canvas_size` keeps its name, type and single writer `set_canvas_size`, now meaning the
effective live size. Every write funnels through `pass_graph.clamp_canvas_size`, Auto included; a
clamp that would change the aspect resolves on the CONSTRAINED axis and re-derives the other from
`resolution`'s aspect, so the Auto loop closes on the stored aspect (correctness F2).

**D2 — under Auto the live size is the largest displaying region, recorded as the DRAWN image
size.** `preview_cell` draws at `dw, dh` after `scale = min(avail.x / tw, img_h / th)`; `cell_w`
over-reports every non-square cell (correctness F1). So `PreviewCellResult` gains `drawn_size:
tuple[float, float]`, and the callers holding a document id record it:
`widgets/document_grid.py::draw_document_preview_grid` and `popups/examples.py`'s grid write
`App.displayed_sizes[document_id]`. `draw_document_preview_button` stays `app`-free and returns the
result up to its caller (chosen over threading `app` through: two surfaces call it, neither needs
`App`). `ui.py::_draw_document_image` records its `image_width`/`image_height`. **Pass-strip tiles
are not recorders** — they draw a pass, and the current document is in the viewer anyway.

`displayed_sizes` is written in frame N's draw and read at the top of N+1:

```python
def auto_canvas_size(displayed: tuple[int, int] | None, aspect: float,
                     previous: tuple[int, int]) -> tuple[int, int]
```

`None` answers `previous`. **Every aspect site reads `resolution`'s under Auto, never the live
canvas** (correctness F2): `_draw_document_image`'s `image_aspect`,
`tabs/document.py::_canvas_presets`'s `current`, `widgets/details.py`'s aspect lock and its two
preset-button labels, `copilot/backend.py`'s probe aspect. Several surfaces → the largest wins.

**D3 — damping: a 5 % dead-band or 8 stable frames, in a PURE function.** `apply_damping(state:
AutoSizeState, requested, current) -> tuple[int, int] | None` lives in `render_plan.py` and returns
the size to apply or `None`; `_tick_frame_state` applies a non-`None` answer (blast F5 — round 1
put the damping in loop code and tested only the pure half). Below 5 % the cost difference at the
viewer's size is under a tenth of a millisecond (`§1`); 8 frames is ~130 ms at 60 fps, and a drag
emits a new size every frame so the stability clause never fires mid-drag. `AutoSizeState` lives on
`App`, never persisted. `AUTO_RESIZE_DEAD_BAND = 0.05`, `AUTO_RESIZE_STABLE_FRAMES = 8`.

**D4 — an Auto resize resamples EVERY canvas whose size changes.** Resampling the history alone
loses the picture one frame later: `set_canvas_size` blanks the LIVE canvas
(`Canvas.set_size` is release-then-allocate) and the next `_swap_feedback` trades the blank into
the history, so a self-reading pass samples black (correctness F3). The funnel resamples the output
pass's live canvas AND each feedback history, each into a new-size replacement through the one-quad
blit, before any release (*Feedback resample*). A non-output pass's history resamples to
`entry.target.target_size(new)`, never the document size (blast F14). The persisted
`feedback/<pass>.bin` seeds through the same resample: `_seed_feedback`'s size equality stops being
a MATCH criterion and becomes the resample TARGET; dtype and component-count mismatches stay strict
rejections.

Ordering makes it safe, and it must hold for EVERY caller of the funnel, not only the Auto path
(correctness R2 — `grep 'set_canvas_size('` returns three call sites). Two are already before the
draw: step 4, and `copilot/backend.py::set_canvas_size` through the bridge drained at the head of
`_tick_frame_state`. The third, `tabs/document.py::_apply_canvas_size`, runs INSIDE the draw phase
and AFTER `_draw_document_image` pushed `render_pass.canvas.texture.glo` into the frame's draw
list — so a W × H commit would release textures imgui is still holding, across the output canvas
and every history. **So the picker's write does not resize in place:** under Fixed it records
`App.pending_resolution: dict[str, tuple[int, int]]`, which **step 4 of the next
`_tick_frame_state` consumes** through the same resize/resample path as Auto. Every canvas release
then happens before any draw — the shape `pending_project_switch` uses for the same 084 D5
hazard.

**D5 — the picker is unchanged; export resolves from `resolution`; the Render tab still wins.**
`tabs/document.py`'s W × H fields and presets chip stay as they are and write `resolution` in both
modes; the caption reads `Canvas` under Fixed, `Export` under Auto. No default-derivation rule and
nothing to compute — a hand-edited document keeps its number (1280×960 stays 1280×960).

Every export path resolves its source size from `resolution`, not the live canvas: `NATIVE`
lowers to `FREE` + `RENDER_AT_TARGET` and `resolve_dims`'s `FREE` fall-through returns
`source_size` — the live Auto size — and `NATIVE` is the `Field(default=…)` of all three copilot
render tools, YouTube's initial and reset shape, and the input to `_artifact_matches_shape`'s
staleness gate, which under Auto would disarm publish whenever the panel moved (blast F1). So
`render_media` passes `self.resolution` to `resolve_dims` on BOTH branches, and
`_artifact_matches_shape` resolves against the same number.

**The Render tab's `resolution_details` W × H is preserved exactly as today** (blast F2):
`_render_image`'s PIL resize and `_render_video`'s ffmpeg `-s` still take the tab's number, and
`render_media` does NOT overwrite it. The `preset=None` path renders at `resolution`, then resizes
to the tab's number as before.

**D6 — one shared budget, default 50 %; `plan_render_set` is the whole rule; hysteresis 4 frames;
same-`k` documents staggered.** `shaderbox/render_plan.py` (new, leaf) owns the pure function
under *The plan function*. The current document draws on the budget first at `k = ceil(cost /
(budget × frame_period))`; others share the remainder at one common fps. `MAX_INTERVAL = 60` binds
EVERY computed `k`, not only `f = 0` (correctness F7 — ten previews at 5 ms produced `k = 143`,
past a cap the spec claimed). Same-`k` documents get a **phase offset** by stable index, so three
previews at `k = 43` land on three frames, not one (correctness F6): the gate is `(app.frame_idx +
phase[id]) % k != 0`. Intervals and phases are ephemeral.

Budget default **0.5**, 088's knee and the fraction every probe ran at (`§2`). **Hysteresis 4
frames** — the smallest window outlasting the cost input's own two-frame read lag (088 D2), so a
`k` never flips on a number the ring has not finished reporting; ≤66 ms of reaction at 60 fps.
`INTERVAL_HYSTERESIS_FRAMES = 4`. The measurement's optimism is accepted as locked: the achieved
share lands below target (36 % measured vs 50 % predicted, `§2`), erring toward rendering more.

**D7 — a per-document `CostRecord`, recorded always, keyed by id in ONE place.** The single write
site is **`_tick_frame_state` step 3**, where the plan reads it a frame fresher and the id is in
hand; the render site writes nothing (correctness F8 found three specified sites). The title key is
closed at the source: render sites open `f"document:{document_id}"` and `profile_rows_plan` renders
the title via an `id -> title` map passed alongside the plan — nothing is matched through a title,
the panel still shows a name (088 chose the title because a uuid does not fit a 280-px panel), and
two same-titled documents are two rows. The policy reads `gpu_ms` alone, so a CPU throttle later is
a policy change. The record is two frames stale by design, fine for a slow signal. No record →
`k = 1`.

**D8 — the script ticks once per UI frame; a throttled feedback pass integrates at the document's
own rate, accepted.** `session.tick` keeps running over the full set at the UI rate, so integrators
stay smooth and `ctx.frame` keeps meaning the UI frame. `begin_frame` is **not** called on a skipped
frame — its swap gate already keys on whether the pass drew at the previous `begin_frame`, so
calling it less often is exactly right and `Document._frame` becomes document-local.

`cadence_flow §2` calls the feedback-stepping RATE "the single biggest open question the spec must
resolve explicitly"; answered here (correctness F9): **a throttled feedback pass steps fewer times
per wall second, so a trail effect is a coarser integration under load — not the same animation
played slower.** This follows from the maintainer's own throttle premise ("it renders less often,
its clock still on wall time") and is **accepted**. D11's checkbox is the escape.

**D9 — the profiler and FPS surfaces reflect the plan.** (a) GPU spans record **always**:
`app.profiler.enabled = app.fps_details_open` goes, `App` constructs `Profiler(enabled=True)`, and
the panel's open state decides only what is drawn. (b) The chip shows two numbers when the current
document is throttled — `60 | doc 10`; `fps_overlay` gains `document_fps: int | None`, `None` draws
today's single number. (c) `document:` rows carry cost, effective fps, `k` and the wall-time share
(`cost × doc fps`), matched to the plan by id.

The row color is **not** `load_color`, whose `STATE_ERROR` starts at 1.0 — exactly where a
converged document sits, making green unreachable for the case round 1 said reads green
(correctness F10). `theme.py::throttle_color(share_ratio: float, frame_over_budget: bool)`:
**`STATE_OK` at ≤ 1.0** (at or under its allowance is the converged, healthy state), **`STATE_WARN`
above 1.0**, **`STATE_ERROR` above 1.5 or whenever the UI's frame period exceeds the target** — the
last clause because a document inside its allowance while the frame still misses is what the reader
must see. Interval changes log at debug; the row is the event.

**D10 — Render all keeps its meaning.** Every open document renders; under Auto at its displayed
size, under D6 scheduled from the shared remainder. Under Fixed a preview renders at full
resolution and the remainder rule makes it affordable — it renders rarely.

**The Examples popup's set REPLACES the normal one while it is open** (blast R3 — round 2 found the
two sets are mutually exclusive: `ui.py` is `if not any_popup_open(): … elif popup_state ==
EXAMPLES: …`, so no `tick_documents` document renders behind the popup, and `begin_frame` iterates
`tick_documents` only, which would leave an interval computed for an example id read by nothing).
So while the popup is open the plan runs over the popup's six documents AS the displayed set —
recorders from its thumbnails, the remainder rule, and `begin_frame` over that same set beside its
render loop, which takes the `(frame_idx + phase) % k` gate. `current` is the popup's own selection
or `None`; the normal set is not planned that frame. V9a asserts a throttled example is actually
skipped.

**D11 — two settings on `UIAppState`**, beside `global_target_fps`, both properties of the machine:
`is_throttle_documents: bool = True`, `document_gpu_budget: float = Field(default=0.5, ge=0.1,
le=1.0)`. Off means today's behavior. Everything else stays a code constant.

---

## Data model changes

**Before:** `{"canvas_size": [1280, 960], "uniforms": {...}, "ui_state": {...}}`
**After:** `{"uniforms": {...}, "ui_state": {"resolution_mode": "auto", "resolution": [1280, 960], ...}}`

**The load order is reshaped** (correctness F4): `ui_models.py::load_document_from_dir` today calls
`Document.load_from_dir(...)` and only then `_load_ui_state(...)` — so the mode is parsed after the
`Document` is built, every `Pass` allocated and `_seed_feedback` run. It becomes: read the raw
metadata, parse `ui_state` FIRST, resolve the initial effective size from `resolution_mode` +
`resolution`, then pass that size to `Document.load_from_dir` as an explicit parameter.
`load_from_dir` reads no top-level `canvas_size` and no `ui_state` key — one reader per concept.

**App state:** `UIAppState` gains D11's two fields; both defaulted, so `extra='forbid'` rejects
nothing and an older `app_state.json` loads through the salvage path unchanged.

**Ephemeral `App` fields:** `displayed_sizes`, `auto_size_states: dict[str, AutoSizeState]`,
`throttle_states: dict[str, ThrottleState]`, `document_costs: dict[str, CostRecord]`,
`render_plan: RenderPlan | None`.

**The hand-edit — every tracked `document.json`, eleven files** (verified against `git ls-files`;
round 1 listed eight, blast F10). No migration, no old-format reader. Each keeps its current
numbers as `resolution`; mode is `auto` except the two pixel-dependent examples:

| file | `resolution` | mode |
|---|---|---|
| `shaderbox/resources/document_examples/0b0d16bb-…` | `[1080, 1920]` | auto |
| `…/53724dbd-…` | `[1280, 960]` | auto |
| `…/73ea2431-…` | `[1280, 1280]` | auto |
| `…/77a84d27-…` | `[512, 512]` | **fixed** |
| `…/8d454b7b-…` | `[1600, 900]` | auto |
| `…/f90f5ff9-…` | `[1280, 960]` | auto |
| `projects/dev/documents/e7e00c46-…` | `[1280, 960]` | auto |
| `projects/dev/documents/ec926580-…` | `[1280, 960]` | auto |
| `projects/documents/1901ab60-…` | `[960, 960]` | auto |
| `projects/documents/307598da-…` | `[1280, 960]` | auto |
| `tests/fixtures/bloom_chain/` | `[960, 960]` | auto |

`77a84d27-…` is the one `fixed` document — its JFA and radiance-cascade PASSES have iteration
counts that follow pixel counts (`resolution_flow §4`, which names that id). Every other row is
`auto`. V12 asserts the mode per file, not only that the key exists. `projects/` goes in the same
commit as the code.

**Six test files build the old JSON shape** and must move to it, or a document silently loads at
`DEFAULT_CANVAS_SIZE` = (64, 64) (blast F8): `test_canvas_presets.py` (also asserts `canvas_size ==
(1280, 960)` after a load, and has a "malformed canvas_size falls back" case testing a path D1
deletes — that case moves to `_as_canvas_size`, which still guards `Document.__init__`'s
parameter), `test_graph_persistence.py`, `test_pass_hot_reload.py`, `test_uniform_row_pruning.py`,
`test_document_dir_sync.py`, `test_feedback_persistence.py`. A seventh,
`test_youtube_exporter.py`, breaks for a different reason: its `_artifact_matches_shape` stub
exposes only `render_pass.canvas.texture.size` and needs a `resolution` under D5.

---

## The plan function

`shaderbox/render_plan.py` — leaf: `math` and `dataclasses` only. No GL, imgui, `App` or
`Document`. The shape `profile_rows_plan` set (088 D5): the policy as pure data.

```python
@dataclass(frozen=True)
class CostRecord:
    gpu_ms: float
    cpu_ms: float

@dataclass
class ThrottleState:
    interval: int = 1          # the live k
    candidate: int = 1         # the k the last recompute wanted
    agreeing_frames: int = 0   # how long candidate has held

@dataclass
class AutoSizeState:
    requested: tuple[int, int] | None = None
    stable_frames: int = 0

@dataclass(frozen=True)
class RenderPlan:
    intervals: dict[str, int]          # document id -> k
    phases: dict[str, int]             # document id -> frame offset within k
    document_fps: dict[str, float]     # id -> global_target_fps / k

def plan_render_set(
    costs: dict[str, CostRecord],
    current: str | None,
    displayed: list[str],
    budget: float,
    frame_period_ms: float,
    states: dict[str, ThrottleState],
    enabled: bool,
) -> RenderPlan
```

`displayed` is the render set in `_tick_frame_state`'s order; `current` is
`app.current_document_id` when in it. `states` is mutated in place — the hysteresis counters are
the only side effect, and they are ephemeral.

**Semantics.** (1) `enabled=False` → every `k` 1, every phase 0, states reset; today's set exactly.
(2) No `CostRecord`, or `gpu_ms` fits `budget × frame_period_ms` → `k = 1`. (3) The current
document takes the budget first: `k_current = ceil(cost / (budget × period))`. (4) Remainder =
`max(0.0, budget × period − cost_current / k_current)`; others share it at one common fps, the
largest `f ≤ target_fps` with `Σcost_i × f ≤ remainder_ms × target_fps`, each taking `k =
min(MAX_INTERVAL, max(1, round(target_fps / f)))` — the cap binds every `k`, `f = 0` falling into
the same clamp. (5) Hysteresis: the computed `k` becomes `candidate`; equal to the live `interval`
resets the counter, else `agreeing_frames` increments and `interval` moves only at
`INTERVAL_HYSTERESIS_FRAMES`. (6) `phases[id] = i % interval`, `i` the index in `displayed`. (7)
`document_fps[id] = target_fps / interval`.

**Worked examples**, re-derived after the cap and phase changes. `frame_period_ms = 16.7`,
`budget = 0.5` unless stated → budget 8.35 ms; hysteresis converged.

| # | Input | Result |
|---|---|---|
| 1 | current, 6 ms | `6 ≤ 8.35` → `k = 1`, phase 0, fps 60. |
| 2 | current, 100 ms, alone | `ceil(100/8.35) = ceil(11.976) = 12` → `k = 12`, fps 5.0. Measured achieved 3.6 (`§2`); accepted. |
| 3 | current 40 ms + three previews at 5 ms | Current `ceil(40/8.35) = 5`, using `40/5 = 8.0`; remainder `0.35`. `Σcost = 15`; `15f ≤ 0.35 × 60 = 21` → `f = 1.4`; `k = min(60, round(60/1.4)) = 43`. Phases 1, 2, 3 — three frames, not one (F6). |
| 4 | twenty previews at 1 ms, current 2 ms | Current `k = 1`; remainder `6.35`. `Σcost = 20`; `20f ≤ 381` → `f = 19.05`; `k = min(60, 3) = 3`. Phases `i % 3` cycle over the twenty. |
| 5 | a Fixed 100 ms document shown only as a 150 px thumbnail, current something else at 4 ms | Fixed → the cost stays 100 ms (D10). Current `k = 1`; remainder `4.35`; `100f ≤ 261` → `f = 2.61`; `k = min(60, 23) = 23`. |
| 6 | throttle off, input as #3 | Every `k = 1`, phase 0 — exactly today's set. V8's falsifier. |
| 7 | `budget = 1.0`, current 40 ms alone | Budget 16.7 → `ceil(40/16.7) = ceil(2.395) = 3`, fps 20. |
| 8 | cost unknown (first frames), current + two previews | Every `k = 1` (D7). |
| 9 | **ten previews at 5 ms**, current 40 ms — the cap's row (F7) | Current `k = 5`, remainder `0.35`. `Σcost = 50`; `50f ≤ 21` → `f = 0.42`; uncapped `round(60/0.42) = 143`, **capped to 60** → fps 1.0. Uncapped, a preview refreshes once every 2.4 s. |

---

## Frame integration

All in `_tick_frame_state`, before the draw phase. The order is what makes it safe against the 084
D5 hazard: every GL reallocation and release happens while the frame's draw list is still empty,
as the project switch does at the top of the same function.

1. **Existing head** — project switch, lib index, copilot bridge, disk sync, mtime reload.
   Unchanged; the abort path still returns `None` before anything below.
2. **The render set is built** — `tick_documents`, unchanged (066 D1/D2); while the Examples
   popup is open its six documents REPLACE that set for the plan, `begin_frame` and the render
   gate (D10, blast R3).
3. **Costs refresh** from `app.last_profile`'s id-keyed `document:` spans into
   `app.document_costs`. The ONE write site (D7).
4. **Resolution resolves**, consuming `app.pending_resolution` first (the picker's Fixed commit
   from the previous frame's draw, R2): Fixed takes `resolution`; Auto asks `auto_canvas_size` from
   `app.displayed_sizes` (written by the PREVIOUS frame's draw), then `apply_damping`;
   `set_canvas_size` applies a non-`None` answer.
5. **Feedback and the live canvas resample** for every document whose size changed (D4).
6. **The plan runs** → `app.render_plan`.
7. **The script ticks** over the full set (D8).
8. **`begin_frame` runs only for documents the plan admits** — `(app.frame_idx + phase[id]) % k !=
   0` is skipped, so `_frame` holds and feedback does not swap.

`_update_and_draw`'s render block gates on the same answer: the whole `if ui_document is not None:`
body — the output render AND the pending-pass sweep — is skipped together, because they share one
`frame_idx` and splitting them would let the sweep become the document's only measured cost
(`cadence_flow §2`).

Steps 4–5 precede step 6 because `k` must be computed from the cost at the resolution actually
rendered, or it over- or under-throttles by the ratio between the sizes (`cost_and_throttle ##
False trails`, the k=12-vs-k=5 case). The residual lag — damping window plus the ring's two frames
— is what the 4-frame hysteresis absorbs.

---

## Feedback resample

`Document.resample_canvas(old: Canvas, size: tuple[int, int]) -> Canvas`, and the funnel calling
it. Per canvas — allocate, blit, release, never release-then-allocate:

1. Return `old` if `old.texture.size == size`.
2. `new = Canvas(gl=self._gl, size=size, dtype=old.dtype, filter=old.filter, wrap=old.wrap)`.
3. Bind `new.fbo`, bind `old.texture` to unit 0, draw one full-screen triangle sampling it — the
   rescale IS the sampler's linear filter. `copy_framebuffer` cannot do this: measured, it copies
   1:1 into a corner between differently-sized framebuffers (no GL error, a plausible picture, the
   wrong one). The resampler is a small class holding a shader, the shape and rationale of
   `channel_blit.py::ChannelBlit` ("one class holding a shader, since two classes for the same
   one-quad GL lifecycle is the mechanism written twice" — `conventions.md`), and lives beside it.
4. `old.release()` — last, after nothing references it. Return `new`.

**The funnel, in `set_canvas_size`** (D4 — both canvases, or the swap trades a blank):

1. `self.canvas_size = clamped`.
2. Output pass: `self.render_pass.canvas = self.resample_canvas(self.render_pass.canvas,
   self.canvas_size)`, replacing the `Canvas.set_size` call (release-then-allocate on a canvas
   whose content matters).
3. Each history: `self._feedback[name] = self.resample_canvas(self._feedback[name], target)`,
   `target` = `self.canvas_size` for the output pass, `entry.target.target_size(self.canvas_size)`
   otherwise (blast F14 — a full-size history on a `scale=0.5` pass becomes a full-size live canvas
   at the next `_swap_feedback`).
4. `_feedback_generation` carries over **unchanged** — it tracks FORMAT changes and a resample
   changes no format, so bumping it would drop the history the resample preserved.

`_feedback_canvas`'s `elif canvas.texture.size != live.texture.size: canvas.set_size(...)` branch
routes through `resample_canvas` too, so the one resize path in the codebase preserves content.

---

## Export

- **The source size is `resolution`, both branches.** `render_media` passes `self.resolution` to
  `resolve_dims` instead of `self.render_pass.canvas.texture.size`, which is what stops
  `RenderShape.NATIVE` resolving to the live Auto size (blast F1).
  `exporters/youtube.py::_artifact_matches_shape` resolves against the same number, so its
  staleness gate stops disarming publish when the panel moves.
- **The `preset is None or SCALE_DISTORT` branch** mints a scratch `Canvas` at `resolution`,
  released in a `finally`, instead of rendering into the live canvas.
- **`resolution_details` is untouched.** `render_media` does not write it; `_render_image`'s PIL
  resize and `_render_video`'s ffmpeg `-s` still take the Render tab's W × H (blast F2).
- `reset_feedback()` still runs first, so an export starts cold and the resample never runs inside
  one. Non-output passes still size from `entry.target.target_size(self.canvas_size)` while the
  OUTPUT frame is captured at the export size — untested territory today; V6.

`Document` gains `resolution_mode` and `resolution` as plain fields set by the loader and the
Document tab's commit, so `render_media` needs no new parameter.

---

## Profiler and FPS surfaces

**Always-on recording (D9a).** `app.profiler.enabled = app.fps_details_open` goes; `App.__init__`
constructs `Profiler(enabled=True)`. `fps_details_open` decides only whether `fps_overlay` draws
the child. The smoother's feed stays gated on `app.profiler.enabled`, now always true.
`test_profiling.py`'s wire test names that deleted line as its falsifier — its docstring is
rewritten to the new one, and its smoother tests re-checked against always-on (blast F15).

**The span key (D7).** Render sites open `f"document:{document_id}"`; `profile_rows_plan` takes an
`id -> title` map and renders the title. Nothing is matched through a title.

**The chip (D9b).** `fps_overlay` gains `document_fps: int | None`; with the current document's
`k > 1` the label is `f"{fps} | doc {document_fps}"`, else today's `f"{fps} FPS"`.

**The panel (D9c).** `profile_rows_plan(profile, fps, target_fps, plan, titles)` — the two trailing
parameters default to `None`, drawing today's rows. `ProfileRow` gains **no** new fields
(correctness F11: `k` and `share` are already in the formatted `number`, and `RenderPlan.intervals`
is the canonical home for a test to assert on).

The row string is **compact**, because `_profile_rows` gives the name column `avail.x - indent -
_number_width(row.number) - SPACE.SM` out of `SIZE.FPS_PANEL_W = 280`, and a full
`"104.2 ms  5 fps  x12"` is about three times today's `"12.34 ms"` — nothing overflows
(`clipped_caption` clips) but document names would clip hard on exactly the throttled rows the
reader is looking at (blast m3). So a throttled `document:` row reads **`10 fps ×6  72%`** — the
effective fps, the interval, and the share of budget as a percent, with the millisecond cost in the
row's tooltip. An unthrottled row keeps today's `"12.34 ms"`. Color is
`throttle_color(share / budget, frame_over_budget)`.

**`tests/test_ui_prose_budget.py`'s exemption rationale for `_profile_rows` is corrected** (blast
m2): it says a span name is "`document:<the document's title>`", which D7 makes false — the name is
`document:<uuid>` and the title arrives through the `id -> title` map. The site stays valid so the
gate still passes; the sentence is edited, and so is `ui_primitives.py`'s own docstring saying the
same.

---

## Settings

`popups/settings.py::_draw_body`, in **General** under Target FPS:

```python
app.app_state.is_throttle_documents = imgui.checkbox(
    "Throttle documents", app.app_state.is_throttle_documents
)[1]
label_row(app.font_12, "Document GPU budget", ctrl_w, label_w)
percent = imgui.drag_int(
    "##document_gpu_budget", round(app.app_state.document_gpu_budget * 100),
    v_min=10, v_max=100, flags=imgui.SliderFlags_.always_clamp,
)[1]
app.app_state.document_gpu_budget = percent / 100.0
```

Both labels are inside the prose budget; neither takes a `help_marker`.

---

## Files touched

- `shaderbox/render_plan.py` **(new)** — `CostRecord`, `ThrottleState`, `AutoSizeState`,
  `RenderPlan`, `plan_render_set`, `auto_canvas_size`, `apply_damping`, the constants. Leaf.
- `shaderbox/render_shape.py` — `ResolutionMode`.
- `shaderbox/channel_blit.py` — the resampler, beside `ChannelBlit`.
- `shaderbox/document.py` — the two mode fields; `resample_canvas` + the `set_canvas_size` funnel;
  `_feedback_canvas` and `_seed_feedback` routed through it; `render_media`'s source size and
  `SCALE_DISTORT` scratch canvas; `load_from_dir` takes the resolved size, reads no `canvas_size`.
- `shaderbox/ui_models.py` — `UIDocumentState`'s two fields, `UIAppState`'s two, `UIDocument.save`,
  **and `load_document_from_dir`'s reordering** (correctness F4/F12).
- `shaderbox/ui.py` — `_tick_frame_state` steps 3–6 and the phased `begin_frame`; the render block
  gates on the plan; spans keyed by id; `_draw_document_image` records its drawn size and reads
  `resolution`'s aspect; the chip passes `document_fps`; `profiler.enabled` stops following the
  panel.
- `shaderbox/app.py` — the five ephemeral fields plus `pending_resolution: dict[str, tuple[int,
  int]]` (D4/R2); `Profiler(enabled=True)`.
- `shaderbox/ui_primitives.py` — `PreviewCellResult.drawn_size`; `profile_rows_plan` takes plan +
  titles; `fps_overlay` takes `document_fps`.
- `shaderbox/theme.py` — `throttle_color`.
- `shaderbox/tabs/document.py` — the mode control; fields write `resolution`; caption follows the
  mode; **`_canvas_presets` reads `resolution`** (F2/F12); `_apply_canvas_size` records
  `pending_resolution` instead of resizing inside the draw (R2).
- `shaderbox/widgets/details.py` — aspect lock and the two preset-button labels read `resolution`.
- `shaderbox/widgets/document_grid.py`, `shaderbox/popups/examples.py` — each records the
  `drawn_size` its `preview_cell` returns, keyed by document id (D2, D10).
- `shaderbox/popups/settings.py` — D11's two rows.
- `shaderbox/help_content.py` — `u_resolution`'s gloss gains the Auto clause.
- `shaderbox/copilot/backend.py` — `set_canvas_size` writes `resolution` and sets the mode to Fixed
  (an explicit pixel request; leaving it Auto would have the next frame overwrite it); the probe's
  aspect reads `resolution`; the working-set line reports the mode.
- `shaderbox/exporters/youtube.py` — `_artifact_matches_shape` resolves against `resolution`.
- The eleven tracked `document.json` files (table above).
- `tests/test_render_plan.py` **(new)**, `tests/test_document_shapes.py` **(new gate, V12)**; edits
  to `test_canvas_presets.py`, `test_graph_persistence.py`, `test_pass_hot_reload.py`,
  `test_uniform_row_pruning.py`, `test_document_dir_sync.py`, `test_feedback_persistence.py`,
  `test_canvas_fields.py`, `test_render_for.py`, `test_profiling.py`,
  `test_persistence_completeness.py`, `test_document_ops.py`, `test_ui_prose_budget.py` (the
  exemption rationale, m2), and **`test_youtube_exporter.py`** — its `_artifact_matches_shape` test
  stubs a document exposing only `render_pass.canvas.texture.size`, so under D5's fix the stub needs
  a `resolution` (blast, unlisted break).
- `ai_docs/dev_flow.md` module map — one line for `render_plan.py`.
- `ai_docs/conventions.md` — the Auto/Fixed split, the always-on profiler, and a correction to the
  one-`update_and_draw`-per-process quirk: the mechanism is xdist GROUPS (`pytest.mark.xdist_group`
  + `--dist loadgroup`), not one named slot-holder; three files drive it today and
  `test_profiling.py` carries no mark (blast F7).
- `ai_docs/roadmap.md` — the row and the banner.

---

## Verification

Each check names what it asserts and the bug that turns it red. `make gates` is the gate, judged by
its exit code captured unpiped. Tests driving `update_and_draw` get their own
`pytest.mark.xdist_group`.

**A test driving `_tick_frame_state` directly must advance `app.frame_idx` itself each iteration**
(blast R2): the increment is the LAST statement of `_update_and_draw`, not of `_tick_frame_state`,
so a loop that calls the latter twelve times runs twelve iterations at the same index — step 8's
`(app.frame_idx + phase[id]) % k != 0` takes the same branch every time, and a `k = 12` document
renders either 12 times or 0. V3a, V7, V8 and V13 would all pass vacuously. The rig is:

```python
def _drive(app: Any, n: int) -> None:
    for _ in range(n):
        _tick_frame_state(app)
        app.frame_idx += 1
```

Moving the increment into `_tick_frame_state` is NOT the fix: `session.tick` and `begin_frame` both
read `frame_idx` from inside that function, so moving it changes what they see.

- **V1 — the plan's nine examples** (`tests/test_render_plan.py`), each row parametrized on exact
  `k` and phase. **Falsifier:** `ceil`→`floor` → case 2 returns 11; remainder before budget → case
  3's previews take `k = 5`; drop the `MAX_INTERVAL` clamp → case 9 returns 143.
- **V2 — hysteresis holds `k` for 4 frames.** Alternate 8.0 / 8.6 ms: `interval` never moves across
  20 frames; a steady 100 ms moves it on exactly the 4th. **Falsifier:** apply the candidate at once.
- **V2a — same-`k` documents land on different frames.** Three previews at `k = 43`: distinct
  phases, no `frame_idx` admitting more than one. **Falsifier:** drop the phase → all three share
  every render frame (F6).
- **V3 — damping, pure.** `apply_damping` through a 60-frame ramp (764 → 940 px in ~3 px steps): at
  most 4 applied sizes; then one size held 8 frames applies. **Falsifier:** drop the dead-band →
  ~60 applied; drop the stability clause → a sub-5 % final size never lands.
- **V3a — the loop applies the damping.** Drive `_tick_frame_state` through the same ramp via
  `_drive` (advancing `frame_idx` per iteration, or the 8-frame stability clause can never fire —
  blast R2) with a monkeypatched `set_canvas_size` counting calls; assert it matches V3.
  **Falsifier:** call `set_canvas_size` with the raw request → ~60 calls (blast F5).
- **V4 — the texture count is constant and no canvas goes blank** (GL). A document with a
  `scale=0.5` feedback pass and an output feedback pass, rendered, then `set_canvas_size` through
  six sizes, counting live textures via a wrapper on `ctx.texture` and the releases. Assert the
  count after equals before; assert BOTH the live canvas and the history carry content, read via
  `media.texture_to_rgba8`, never `texture.read()[0]`. **Falsifiers:** resample only the history →
  the far-corner pixel of a non-uniform source reads black (F3); release before allocating → the
  blit reads a freed texture; skip the release → the count climbs per resize; swap the quad for
  `copy_framebuffer` → the far corner reads black while a mean check passes; resample the scaled
  pass to the document size → its history comes back at the wrong dims (blast F14).
- **V5 — export never reads the live size, on all three paths.** Parametrized over `preset=None`,
  `shape_to_preset(NATIVE)` and one `FIXED_ASPECT` shape: Auto document, `set_canvas_size` to
  320×180, export, assert the written size each time. **Falsifier:** pass
  `render_pass.canvas.texture.size` as `resolve_dims`'s source → the NATIVE case comes out 320×180
  (blast F1). Plus one case on `_artifact_matches_shape`: an artifact still matches after a live
  resize.
- **V5a — the Render tab's W × H still lands on disk.** Export `preset=None` with
  `resolution_details` 640×480 while `resolution` is 1920×1080; assert the file is 640×480.
  **Falsifier:** overwrite `resolution_details` from `resolution` → it exports 1920×1080. Note
  `test_render_for.py::test_render_media_preset_none_byte_identical` passes either way, so no
  existing gate catches this (blast F2).
- **V6 — a scaled feedback pass inside an off-size export.** Extend `test_render_for.py`'s off-size
  fixture with a `scale=0.5` feedback pass; assert the output is not blank and carries its
  contribution. **Falsifier:** size the feedback canvas from the export target → a mismatched
  texture.
- **V7 — the throttle is WIRED.** Drive `_tick_frame_state` on the `app` fixture via `_drive`
  (advancing `frame_idx`, without which the phase gate never moves and the assertion is unreachable
  either way — blast R2), two documents, a 100 ms `CostRecord` on the current one; across 12 frames
  its `_frame` advanced ~1 time, not 12.
  **Falsifier:** compute the plan and never read it at the render site. The mutation is applied at
  the LOOP, not at `plan_render_set`: a test that hands the condition in cannot discover that
  nothing produces it (`conventions.md`, the mutate-the-wiring law).
- **V8 — throttle off restores today's set.** Same `_drive` rig, `is_throttle_documents = False`:
  every document's `_frame` advances every frame. Without the `frame_idx` bump this passes
  vacuously — it would be green even with the throttle wired wrong (blast R2). **Falsifier:** ignore
  the setting.
- **V9 — the recorders are WIRED** (blast F3, round 1's largest hole: cutting all three was
  silent, since `auto_canvas_size(None, …)` returns `previous`). On the `app` fixture: current
  document to Auto, clear `displayed_sizes`, drive one frame, assert an entry for its id with
  positive dims; drive a second and assert `canvas_size` left its loaded value. Siblings drive the
  document grid and the Examples popup in headless imgui frames (as `test_canvas_fields.py` drives
  the Document tab), asserting each registers its own id. **Falsifier:** delete any recorder's
  write → its case goes red with the dict empty.
- **V9a — a throttled example document is actually skipped in the popup's set** (blast R3). Open
  the Examples popup, plant a 100 ms `CostRecord` on one example, drive frames through `_drive`, and
  assert that example's `Document._frame` advanced fewer times than a cheap sibling's.
  **Falsifier:** render the examples loop without the `(frame_idx + phase) % k` gate — every example
  advances every frame while the plan's intervals are read by nothing, which is the
  defined-not-wired shape the interval gate exists to avoid.
- **V10 — the chip is WIRED.** Monkeypatch `ui.fps_overlay` to capture kwargs (the spy in
  `test_profiling.py` has this shape), plant a 100 ms cost, drive one frame, assert `document_fps`
  is not `None` and equals `global_target_fps // k`. **Falsifier:** pass `document_fps=None`
  unconditionally — every prose gate still passes (blast F6).
- **V10a — the panel's rows match by id.** Two documents sharing a title, different costs: each row
  carries its own `k`. **Falsifier:** match by title → one takes the other's numbers (blast F4).
- **V11 — GPU spans record with the panel closed.** `fps_details_open = False`, four frames,
  `app.last_profile` carries a `document:` child with a non-`None` `gpu_ms`. **Falsifier:** restore
  `app.profiler.enabled = app.fps_details_open`.
- **V12 — every tracked `document.json` is in the new shape** (`tests/test_document_shapes.py`).
  The domain is `git ls-files '*document.json'`, not a directory walk, so a document a local run
  writes under the untracked `projects/documents/` cannot fail the gate red on one box and green in
  CI (blast m1). Assert no top-level `canvas_size`, that each carries `ui_state.resolution_mode` and
  `ui_state.resolution`, and that the mode matches the table — `77a84d27-…` fixed, the rest auto
  (R1). **Falsifier:** leave one file's old key, or flip a mode → the test names that file. Without
  the gate a forgotten document loads at (64, 64) in silence — `_load_ui_state` fills missing keys
  from defaults and never raises (blast F10). Ships in the same commit as the sweep.
- **V13 — the copilot's explicit size sticks.** Extend
  `test_document_ops.py::test_set_canvas_size_applies_and_clamps`: assert `resolution_mode is
  FIXED` and `resolution == (128, 200)`, then drive one `_tick_frame_state` through `_drive` (the
  `frame_idx` bump, blast R2) and assert the size survived. **Falsifier:** leave the mode Auto → the
  next frame overwrites it, the tool silently does nothing and today's assertions all pass (blast
  F9).
- **V14 — the docs and model gates.** `test_roadmap_shape.py`, `test_prose_spelling.py`,
  `test_ui_prose_budget.py` over the two settings labels and the chip,
  `test_persistence_completeness.py` over the reshaped models.
- **Maintainer's eyes:** whether an Auto document reads as sharp at the viewer's size, and whether
  a throttled document at `k = 5` feels like iteration or like lag.

---

## Pre-implementation measurements

**Both are done; this section no longer gates starting.**

**Always-on GPU query cost (D7, D9a) — measured, passes.** 088 measured the ring's READ stall but
never the `begin()`/`end()` pair as a permanent background cost (`cadence_flow §4`). The round-1
verification reviewer ran it: a real `App`, hidden glfw window, six shipped examples with
`is_render_all_documents` on, 120 warm-up frames, then interleaved off/on blocks so driver drift
hits both arms equally.

```
run 1 (300 frames/arm, 6 documents)   OFF p95 13.949 ms   ON p95 13.954 ms   DELTA p95 +0.005 ms
run 2 (400 frames/arm, 6 documents)   OFF p95 13.954 ms   ON p95 13.955 ms   DELTA p95 +0.000 ms
```

The MEDIAN delta changed sign between runs (−0.169, then +0.337 ms), so it sits inside the
run-to-run noise floor — a single run reporting +0.337 ms would read as a real 2 % cost it is not.
The stable number is the **p95 delta, +0.005 ms**, under 0.03 % of the 16.7 ms frame. **Pass
threshold: p95 delta ≤ 0.1 ms**, cleared twenty-fold. D9a's premise holds and **D7's CPU-swap-wait
fallback is dropped** — carrying it would be speculative machinery. The probe belongs under
`probes/` beside the `cost_*` / `throttle_*` siblings; the reviewer ran it from a scratchpad, so
the implementer lands it there or records it as a one-shot.

**`copy_framebuffer` does not rescale — measured twice, independently.** A 64×64 source half white
copied into 128×128 lands 1:1 in a corner: `GL_NO_ERROR`, white fraction 0.125 where a rescale gives
0.5, far corner black; a one-quad draw gives 0.5 with the boundary in the right place. Hence D4 step
3's quad and V4's far-corner assertion.

---

## Open questions for the user

None. D1–D11 are locked; D3's and D6's numbers are picked; the round-1 triage was decided by the
maintainer and the coordinator and is recorded below.

---

## Review history

**Round 1 (pre-implementation, two reviewers, read-only, each with its own probes). Both FAIL;
every finding accepted, none rejected.** Evidence for each lives in the reports; one line here per
finding, id → what changed.

*Correctness & design — 3 BLOCKER (B), 7 MAJOR (M), 4 MINOR (m).*

- **F1 (B)** recorders could not record what D2 said (`draw_document_preview_button` has no `app`;
  `cell_w`/`PASS_TILE` are not the drawn size; `_draw_pass_tile` draws a pass) → D2 rewritten around
  `PreviewCellResult.drawn_size`; pass tiles dropped as recorders.
- **F2 (B)** Auto's aspect was circular → every aspect site reads `resolution`; a clamp resolves on
  the constrained axis; `widgets/details.py` and the probe added to *Files touched*.
- **F3 (B)** the resample kept the history and lost the live frame → D4 and the funnel cover both
  canvases; V4 asserts both.
- **F4 (M)** the loader could not resolve the mode before `Document.__init__` → `load_document_from_dir`
  reordered, parsing `ui_state` first; added to *Files touched*.
- **F5 (M)** D5's export default had no answer for 4 of 8 documents → **closed by the model change**:
  one stored `resolution`, no derivation, existing numbers kept.
- **F6 (M)** same-`k` documents shared a frame → phase offset in `RenderPlan`; example 3 re-derived;
  V2a.
- **F7 (M)** `MAX_INTERVAL` bound only `f = 0` (ten previews → `k = 143`) → the cap binds every `k`;
  example 9 exercises it.
- **F8 (M)** the cost record had three homes and a title key → one site (step 3); spans keyed by id
  with an `id -> title` map for display; V10a.
- **F9 (M)** the feedback-stepping rate was unanswered → D8 states the coarser integration, accepts
  it, names D11's checkbox as the escape.
- **F10 (M)** "converged reads green" was false under `load_color` → `throttle_color`: green ≤ 1.0,
  warn above, error above 1.5 or when the frame misses.
- **F11 (m)** `ProfileRow`'s two new fields duplicated the formatted number → dropped.
- **F12 (m)** two files missing from *Files touched* → added.
- **F13 (m)** the Fixed switch's effect on the second size field → moot; one field now.
- **F14 (m)** the probe was a precondition nothing gated → moot; measured and recorded.

*Verification & blast radius — 3 B, 8 M, 4 m.*

- **F1 (B)** `RenderShape.NATIVE` still read the live size — the default of all three copilot render
  tools, YouTube's initial/reset shape, and `_artifact_matches_shape`'s gate → `resolve_dims` takes
  `resolution` on both branches; V5 over three paths plus the gate.
- **F2 (B)** D5 silently revoked the Render tab's W × H → `resolution_details` explicitly untouched;
  V5a, noting the existing byte-identical test passes either way.
- **F3 (B)** the recorders had no consumer test → V9, the reviewer's sketch adopted with the grid and
  Examples siblings.
- **F4 (M)** panel rows by title vs plan by id → same fix as correctness F8; V10a.
- **F5 (M)** the damping was loop code with only the pure half tested → `apply_damping` moved into
  `render_plan.py`; V3 pure, V3a at the loop.
- **F6 (M)** the two-number chip had no wiring test → V10.
- **F7 (M)** the `update_and_draw` slot is xdist GROUPS, not one holder, and `test_profiling.py`
  carries no mark → the `conventions.md` quirk is corrected this wave; new frame-driving tests get a
  group.
- **F8 (M)** six test files build the old JSON shape → all six named in *Data model changes* and
  *Files touched*.
- **F9 (M)** nothing verified the copilot's Fixed switch → V13.
- **F10 (M)** three tracked `document.json` files missed; a forgotten one defaults to (64, 64) in
  silence → the table is all eleven, verified against `git ls-files`; V12 is the gate.
- **F11 (M)** the Examples popup was a six-document render set with no recorder and no budget → D10
  admits it to the plan with recorders.
- **F12 (m)** the probe's path stated two ways → `probes/`.
- **F13 (m)** no pass threshold → p95 delta ≤ 0.1 ms, stated with the measurement.
- **F14 (m)** `resample_feedback`'s size for a non-output pass → `entry.target.target_size`; V4
  covers a scaled pass.
- **F15 (m)** the smoother claim asserted nowhere → `test_profiling.py`'s smoother tests re-checked
  against always-on.

*False trails both reviewers recorded, so round 2 did not re-spend them:* the `copy_framebuffer`
measurement (reproduced independently); all eight round-1 worked examples recomputed exactly, so
F6/F7 are coverage not arithmetic; the 084 D5 ordering argument; `app.last_profile`'s one-frame
offset is intended; the settings snippet matches the existing idiom; `render_plan.py`'s leaf claim
holds against every parameter type; `gpu_total` over a CPU span is correct; `UIAppState`'s
`extra="forbid"` accepts two defaulted fields; the 44 dogfood-run `document.json` files are
gitignored; the dogfood harness never runs Auto (no `App`), paying only the extra resample.

**Round 2 (same two reviewers, against the revised text). Both PASS-WITH-MINORS**; all 29 round-1
findings confirmed closed, none regressed. Five new items, every one applied:

- **Correctness R1 / blast R1 (regression, introduced by the revision)** — the hand-edit table
  marked all eleven rows `auto` while the prose said the JFA and cascade "examples" are Fixed, and
  "examples" was an over-count: they are two PASSES of one document, `77a84d27-…`
  (`resolution_flow §4` names that id). An implementer working from the table — the actionable
  artifact — would have shipped eleven Auto documents and broken the pixel-dependent one. → that
  row is `fixed`, every other row `auto`, the prose is singular and identifies the document by id,
  *Out of scope* matches, and V12 now asserts the mode per file.
- **Correctness R2** — D4's blanket ordering claim covered the Auto path and the copilot bridge but
  not `tabs/document.py::_apply_canvas_size`, which commits inside the draw phase after
  `_draw_document_image` pushed the texture into the draw list; D4 widened that pre-existing
  exposure from one canvas to the output plus every history. → the picker's Fixed write no longer
  resizes in place: it records `App.pending_resolution`, consumed by step 4 of the next
  `_tick_frame_state` through the same resize/resample path as Auto, so every release precedes any
  draw. Field and step named; `app.py` and `tabs/document.py` updated in *Files touched*.
- **Blast R2** — `app.frame_idx += 1` is the last statement of `_update_and_draw`, not of
  `_tick_frame_state`, so V3a, V7, V8 and V13's second half were unreachable as written: twelve
  iterations at one index make the phase gate take the same branch every time, and V8 would have
  passed even with the throttle wired wrong. → a *Verification* preamble states the `_drive` rig
  (bump `frame_idx` per iteration) with the reason moving the increment is not the fix
  (`session.tick` and `begin_frame` read it from inside), and all four items say so explicitly.
- **Blast R3** — D10 admitted the Examples popup to the plan, but its set is an ALTERNATIVE to
  `tick_documents` (`if not any_popup_open() … elif EXAMPLES`), and `begin_frame` iterates
  `tick_documents` only, so an interval computed for an example id was read by nothing and the
  examples loop still rendered every example every frame. → D10 rewritten: while the popup is open
  its six documents REPLACE the normal set for the plan, `begin_frame` and the `(frame_idx + phase)
  % k` render gate; the normal set is not planned that frame; V9a asserts a throttled example is
  skipped.
- **Blast minors** — m1: V12's domain is `git ls-files '*document.json'`, not a directory walk, so
  an untracked local document cannot fail the gate red on one box and green in CI. m2:
  `test_ui_prose_budget.py`'s exemption rationale (and `ui_primitives.py`'s docstring) said a span
  is `document:<title>`, which D7 makes false — both corrected, the site stays valid. m3: the D9c
  row string was ~3× today's width in a 280-px panel, clipping names on exactly the throttled rows
  being read — the compact form is `10 fps ×6  72%` with the millisecond cost in the tooltip.
- **Unlisted break the blast review found** — `tests/test_youtube_exporter.py`'s
  `_artifact_matches_shape` test stubs a document exposing only `render_pass.canvas.texture.size`,
  which D5's fix stops reading; the stub needs a `resolution`. → added to *Files touched* and to
  the old-shape test list as a seventh file, breaking for a different reason than the six.
