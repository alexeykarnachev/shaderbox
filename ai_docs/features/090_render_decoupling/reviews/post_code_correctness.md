# 090 — post-implementation review: CODE CORRECTNESS

Target: commit `f06d957` ("090: document throttle and Auto/Fixed resolution"), branch `dev`.
Read-only on the tree; every probe ran in a detached worktree at `f06d957` with that worktree
first on `PYTHONPATH`. Box: X11 `:1`, RTX 3090 — the GL tests ran for real, the smoke ran rather
than skipping.

**Verdict: PASS-WITH-MINORS.**

---

## Coverage

`make gates` on the shipped tree: **exit 0**, `== gates: GREEN -- check passed, test passed,
smoke passed ==`, captured unpiped to a file and `$?` read before anything touched it.

**Read end to end (whole file, not the hunks):**

- `shaderbox/render_plan.py` (new, 230 lines)
- `shaderbox/ui.py` (1058) — `_tick_frame_state`, `renders_this_frame`,
  `_refresh_document_costs`, `_resolve_resolutions`, `_update_and_draw`'s three render branches,
  `_draw_document_image`, `_current_document_fps`, `_document_titles`, `_draw_app_panel`
- `shaderbox/document.py` (1071) — `as_canvas_size`, `_clamped_to_aspect`, `resample_canvas`,
  `set_canvas_size`, `begin_frame`, `_swap_feedback`, `reset_feedback`, `drop_feedback`,
  `_seed_feedback`, `newest_frame`, `_feedback_canvas`, `render`, `load_from_dir`, `render_media`
- `shaderbox/channel_blit.py` (123), `shaderbox/render_shape.py` (100)
- `shaderbox/tabs/document.py` (316), `shaderbox/widgets/details.py` (145),
  `shaderbox/widgets/document_grid.py` (113), `shaderbox/popups/examples.py` (129),
  `shaderbox/popups/settings.py` (423)

**Read as the diff plus every function the diff touches and its callers** (the files are 1800–2700
lines and the change is a handful of localized sites in each; I read each changed function whole
and traced its readers):

- `shaderbox/app.py` — `record_displayed_size`, `_init`'s ephemeral block, `Profiler(enabled=True)`,
  `any_popup_open`, `switch_project`
- `shaderbox/ui_primitives.py` — `PreviewCellResult`, `preview_cell`, `ProfileRow`,
  `profile_rows_plan`, `_document_row`, `_plan_tree`, `_measured_row`, `_profile_rows`,
  `fps_overlay`
- `shaderbox/ui_models.py` — `UIDocumentState`, `UIAppState`, `UIDocument.save`,
  `load_document_from_dir`, `load_documents_from_dir`, `_load_ui_state`
- `shaderbox/copilot/backend.py` — `_canvas_line`, `set_canvas_size`, the probe-aspect site,
  `_probe_frame`
- `shaderbox/exporters/youtube.py` — `_artifact_matches_shape`
- `shaderbox/theme.py` — `throttle_color`; `shaderbox/help_content.py` (one string)

**Read as supporting context, unchanged by the diff but load-bearing for the findings:**
`shaderbox/profiling.py` (the ring, `gpu_total`, `headline_ms`, `_read_pending`),
`shaderbox/pass_graph.py` (`clamp_canvas_size`, `MIN/MAX_CANVAS_PX`, `TargetConfig.target_size`),
`shaderbox/project_session.py` (`_load_one_document_from_disk`, `sync_documents_from_disk`).

**Tests — all new and changed files read in full:** `test_render_plan.py` (new),
`test_render_decoupling_loop.py` (new), `test_canvas_resample.py` (new),
`test_document_shapes.py` (new); and the diffs of `test_canvas_fields.py`,
`test_canvas_presets.py`, `test_document_graph.py`, `test_document_dir_sync.py`,
`test_feedback_persistence.py`, `test_graph_persistence.py`, `test_pass_hot_reload.py`,
`test_probe_clock_and_turn_end.py`, `test_profiling.py`, `test_radiance_cascades_example.py`,
`test_render_for.py`, `test_ui_prose_budget.py`, `test_uniform_row_pruning.py`,
`test_youtube_exporter.py`.

**Skipped, and why:** the eleven `document.json` files were not read line by line — they were
exercised instead, by running the app's own loader (`ui_models.load_document_from_dir`) over all
eleven and asserting `resolution`, `resolution_mode` and the resulting `canvas_size` against the
spec's table (all 11 OK, output below). `ai_docs/roadmap.md` and `ai_docs/dev_flow.md` are
documentation and belong to the spec-fidelity pass, not this one; `ai_docs/conventions.md` was
read for the two new Design decisions and the corrected ring/xdist quirks because findings hang
off them.

**Falsifiers re-applied** (five, in the worktree, each restored and the tree verified clean
afterwards): V1's `ceil`→`floor`; V7's render-site gate; V7's step-8 `begin_frame` gate; V4's
one-quad blit → `copy_framebuffer`; V11's `profiler.enabled = fps_details_open`. All five turned
the named tests red. Details under *Falsifier verification*.

---

## Findings

### MAJOR-1 — the pass-settings modal renders a throttled document ungated, while its feedback advance stays gated

`shaderbox/ui.py:502-508`.

`_update_and_draw`'s render block has three branches. The normal one
(`ui.py:461`) and the Examples one (`ui.py:498`) both call `renders_this_frame`. The third does
not:

```python
elif (
    app.popup_state == PopupState.PASS_SETTINGS and current_ui_document is not None
):
    with app.profiler.cpu(f"{_DOCUMENT_SPAN_PREFIX}{app.current_document_id}"):
        current_ui_document.document.render(profiler=app.profiler)
```

Meanwhile `_tick_frame_state` step 8 (`ui.py:326-328`) *does* gate `begin_frame` for that same
document — `tick_documents` under any popup is `[current_document_id]`, and
`planned_documents` is that list when the popup is not EXAMPLES. So the two reads of the plan
disagree: the render site ignores it, the feedback advance honors it.

This is the exact defect class blast R3 found for the Examples popup in round 2 and closed there;
the third branch was missed. The spec never mentions `PASS_SETTINGS` — `grep -n
"PASS_SETTINGS\|pass-settings" 01_spec.md research/cadence_flow.md` returns nothing.

**Demonstration** (probe test, worktree, 12 frames with a 100 ms `CostRecord` planted on the
current document and `_refresh_document_costs` frozen):

```
PASS_SETTINGS open, k=12: renders=12, begin_frames=1 over 12 frames
E  AssertionError: a k=12 document rendered 12 times in 12 frames behind the
   pass-settings modal while begin_frame ran 1 times
```

Consequences, both real: (a) the throttle does nothing behind that modal — the document submits
its full GPU cost on every frame, which is the whole hitch the feature exists to spread, and the
pass-settings modal is exactly where a user sits while adjusting an expensive graph; (b) a
feedback pass draws 12 times into the same canvas between two swaps, so `u_prev` reads the
previous *render* rather than the previous *frame* for 11 of those 12 — D8 specified one
integration step per render at the document's own rate, and this branch delivers 12 renders per
integration step.

The picture does not corrupt (`drawn_frame` is re-stamped to the held `_frame` each time, so the
eventual `begin_frame` still performs exactly one swap — verified separately), so the symptom is
cost and integration rate, not a wrong image.

**Fix** — one line, matching the other two branches:

```python
elif (
    app.popup_state == PopupState.PASS_SETTINGS
    and current_ui_document is not None
    and renders_this_frame(app, app.current_document_id)
):
```

Add a case to `test_render_decoupling_loop.py` beside `test_a_throttled_example_renders_less_often_than_a_cheap_one`,
counting renders (not `_frame`) behind `PopupState.PASS_SETTINGS`; its falsifier is removing the
gate again, which is the probe above.

---

### MINOR-1 — an Auto request outside the canvas bounds never settles, so step 4 re-enters the resize funnel every frame forever

`shaderbox/ui.py:395-403` (`_resolve_resolutions`'s Auto branch) against
`shaderbox/document.py:177-179` (`set_canvas_size`'s `_clamped_to_aspect`).

`apply_damping` compares `requested` against `document.canvas_size`, but `set_canvas_size` stores
the **clamped** value (`clamp_canvas_size`, 16..4096 per axis). When the displayed region resolves
to a size outside that range, `canvas_size` can never equal `requested`, `_past_dead_band` is
therefore true on every frame, and `apply_damping` returns the request every frame — permanently.
`state.stable_frames` is also reset to 0 on every frame, so the damping state is stuck in
"just applied".

**Demonstration** (worktree, `_tick_frame_state` driven with `frame_idx` advanced per iteration,
`set_canvas_size` counted):

```
a region past MAX_CANVAS_PX: region=(5000, 3750) -> 30 set_canvas_size calls / 30 frames,
                             canvas settled at (4096, 3072)
a tile under MIN_CANVAS_PX:  region=(12, 9)      -> 30 set_canvas_size calls / 30 frames,
                             canvas settled at (16, 16)
a small tile (in bounds):    region=(40, 30)     -> 1  set_canvas_size call  / 30 frames
```

And the pure half, showing the class is wider than the two extremes:

```
in bounds              region=(800, 600)   request=(800, 600)   applied=(800, 600)   settles=True
width at max           region=(4096, 3072) request=(4096, 3072) applied=(4096, 3072) settles=True
width over max         region=(4100, 3075) request=(4100, 3075) applied=(4096, 3072) settles=False
tiny                   region=(20, 15)     request=(20, 15)     applied=(21, 16)     settles=False
below min on one axis  region=(300, 10)    request=(13, 10)     applied=(16, 16)     settles=False
16:1 doc in a tall box region=(50, 800)    request=(50, 3)      applied=(256, 16)    settles=False
```

Note the third row of the second block: a `(20, 15)` request applies as `(21, 16)` — the aspect
re-derivation in `_clamped_to_aspect` moved the *width* too. So even a request only barely under
`MIN_CANVAS_PX` on one axis never settles.

**Why MINOR rather than MAJOR:** `resample_canvas` early-returns on `old.texture.size == size`
(`document.py:145-146`), so no GL object is actually allocated, blitted or released on the repeat
frames — measured, a 30-frame run at a stationary over-max request performs **1** texture
allocation, and a 30-frame drag past the clamp performs **1**. The cost is the redundant Python
pass over the output canvas and every history each frame, and the damping state being permanently
disarmed for that document. No leak, no blank canvas, no visual defect.

Reachability is real but narrow: a 5K/8K display with the viewer maximised on a document whose
aspect puts a dimension past 4096, or a very flat document (16:1) in a tall grid tile.

**Fix** — compare the damping's answer against what the funnel will actually store, so the loop
closes. Either clamp the request before damping:

```python
requested = _clamped_to_aspect(
    auto_canvas_size(displayed.get(document_id), aspect, document.canvas_size),
    document.resolution,
)
```

(`_clamped_to_aspect` would need to become public, or a thin `Document.clamped_size(size)` wrapper
added), or have `apply_damping` return `None` when `requested` is what the previous call already
answered and the canvas did not move. The first is the honest one — the requested size should be a
size the document can hold.

The gate: extend `test_render_decoupling_loop.py::test_the_loop_damps_a_drag_ramp_instead_of_resizing_every_frame`
with a parametrized case at `(5000, 3750)` and one at `(12, 9)`, asserting the same
`0 < len(calls) <= 5` bound. Falsifier: revert the clamp and both cases report 30.

---

### MINOR-2 — `ThrottleState` and `AutoSizeState` entries are never dropped for a closed document

`shaderbox/render_plan.py:191` (`states.setdefault`), `shaderbox/ui.py:400`
(`app.auto_size_states.setdefault`), `shaderbox/app.py:1274-1276`.

Both dicts grow with every document id the session ever plans and never shrink. `document_costs`
has the same shape (`ui.py:356`).

**Demonstration:**

```
=== STALE STATE: a document removed never has its ThrottleState dropped ===
  states keys after removal: ['a', 'b', 'c', 'd']
```

(after planning `["c","a","b","d"]` for 8 frames, then `["c","b","d"]` for 8 more).

Bounded by documents-per-session and each entry is three ints, so this is a housekeeping nit, not
a leak that matters. Both dicts reset on a project switch (`app.py:_init`), which is the only
place it could grow without bound. Recording it so the next round does not re-derive it as a
finding.

**Fix** (optional): prune in `plan_render_set` — `for stale in set(states) - set(displayed):
del states[stale]` — and the same in `_resolve_resolutions` for `auto_size_states`. Note this
would change the document-switch behavior: a document dropped from the set for one frame would
lose its hysteresis counter and restart at `interval=1`, which may be worse than the leak. Leaving
it as-is is a defensible call; the point is that it is a call, not an oversight.

---

### MINOR-3 — a `document:` row falls back to a raw uuid for two frames after a document closes

`shaderbox/ui_primitives.py:1450` (`titles.get(document_id, document_id)`).

`_document_titles` is rebuilt from live state every frame (`ui.py:925-941`), so a **rename** is
picked up immediately — no staleness there, which was the brief's question. The reverse case does
bite: `app.last_profile` is two frames behind (088 D2), so a document closed on frame N still has
a `document:<id>` span in the profile drawn on frames N+1 and N+2, while `titles` no longer carries
its id. The row then reads as a bare uuid.

**Demonstration:**

```
  depth=0 name='deadbeef-1111-2222-3333-444444444444' number='4.00 ms' tooltip=''
```

Cosmetic, two frames, in a panel that is only drawn when open. The fallback is the right shape (a
row is better than a crash or a dropped row); a shorter fallback — the first 8 characters, or
`"(closed)"` — would read better in a 280-px panel where a 36-character uuid clips hard. Listed so
the next round does not re-raise it as a match-by-id bug; it is not one.

---

## False trails

Probed, found sound. Recorded so the next round does not re-spend them.

- **The plan's nine worked examples.** Re-derived independently against `plan_render_set`; every
  one matches the spec's table exactly, phases included. `test_render_plan.py` asserts them on the
  exact interval and phase.
- **Edge inputs the spec did not list.** Zero displayed documents → empty plan, no raise. All costs
  zero → every `k = 1`. A cost above `MAX_INTERVAL × budget × period` (5000 ms) → clamped to 60,
  fps 0.998. `budget = 1.0` → example 7's answer. Throttle off at any budget → every `k = 1`, every
  phase 0. `current` not in `displayed` → that document is planned as an "other", which is the
  D10 Examples-popup shape and correct. A duplicate id in `displayed` → the last index wins for
  the phase; harmless, and `tick_documents` cannot produce one.
- **NaN cost.** `math.ceil(nan / x)` is not reached — `current_cost > budget_ms` is `False` for
  NaN, so the interval is 1 and the document is simply not throttled. Fails open. An `inf` cost
  raises `OverflowError`, and `frame_period_ms == 0` raises `ZeroDivisionError`, but neither is
  reachable: `gpu_ms` comes from `query.elapsed` (a `GLuint64`, never negative and never inf) and
  `global_target_fps` is `Field(ge=30, le=240)`.
- **Negative cost.** Unreachable for the same reason. Were it to occur it fails open (the whole set
  goes to `k = 1`), not closed.
- **`intervals` / `document_fps` desync.** Cannot occur — `plan_render_set` fills `intervals`,
  `phases` and `fps` in one loop over `displayed` (`render_plan.py:198-203`), and the
  `enabled=False` branch builds all three from `dict.fromkeys(displayed, ...)`. The chip's
  `round(plan.document_fps.get(id, 0.0))` and `_document_row`'s same-shaped `.get` can only
  produce a 0 from a plan nothing constructs.
- **Phase stability across set changes.** A document removed from the middle shifts every later
  document's phase (`b` moved 2 → 1 when `a` closed) — but this is exactly what the spec specifies
  (`phases[id] = i % interval`, `i` the index in `displayed`), and the interval changes in the same
  frame anyway, so the cadence was going to move regardless. Not a defect.
- **`frame_idx` wrap.** Python ints are unbounded; `(app.frame_idx + phase) % interval` cannot
  wrap. At 60 fps a year is 1.9e9 frames and nothing narrows it to a machine word.
- **Hysteresis across a document switch.** Costs the newly-selected document ~4 plan frames at its
  old "other" interval; measured, the document rendered once in the first 10 frames after the
  switch (~100 ms to the first draw at the new rate, worst case with two 40 ms documents). Within
  the 4-frame window D6 locked; not a defect.
- **`begin_frame` / `_swap_feedback` under throttling.** Verified directly on a GL document with a
  self-reading pass at `k = 5`: the trail advances by exactly one integration step per render
  (R = 12, 25, 38, 50, 63 — 13 per render, i.e. 0.05×255), with no double swap and no missed swap.
  This is D8's accepted coarser integration, working as specified.
- **`_seed_feedback` with no history, a released document, a double `release()`, a
  `reset_feedback()` mid-resize, and a same-size no-op.** All clean: no history → the funnel is a
  no-op over an empty dict; `release()` nulls `_resampler` and a second `release()` survives; a
  same-size `set_canvas_size` keeps the identical texture object and does not even allocate the
  resampler.
- **The resampler's own GL objects.** `CanvasResampler` holds no canvas (the caller owns the
  destination), `Document.release()` releases it and sets it to `None`, and it is allocated lazily
  on the first real resize. `test_canvas_resample.py`'s allocated-minus-released accounting is
  correct and I did not find a way to break it.
- **Texture leak across a resize ramp.** `test_canvas_resample.py` covers six sizes; I additionally
  drove a 30-frame drag past the clamp and a 30-frame stationary over-max request: 1 allocation
  each. No leak.
- **All eleven tracked `document.json` files through the app's own loader.** Each opened, and
  `resolution`, `resolution_mode` and the resulting `canvas_size` all match the spec's table:

  ```
  OK e7e00c46 res=(1280, 960)  mode=auto  canvas=(1280, 960)   passes=1
  OK ec926580 res=(1280, 960)  mode=auto  canvas=(1280, 960)   passes=1
  OK 1901ab60 res=(960, 960)   mode=auto  canvas=(960, 960)    passes=5
  OK 307598da res=(1280, 960)  mode=auto  canvas=(1280, 960)   passes=1
  OK 0b0d16bb res=(1080, 1920) mode=auto  canvas=(1080, 1920)  passes=1
  OK 53724dbd res=(1280, 960)  mode=auto  canvas=(1280, 960)   passes=1
  OK 73ea2431 res=(1280, 1280) mode=auto  canvas=(1280, 1280)  passes=1
  OK 77a84d27 res=(512, 512)   mode=fixed canvas=(512, 512)    passes=6
  OK 8d454b7b res=(1600, 900)  mode=auto  canvas=(1600, 900)   passes=1
  OK f90f5ff9 res=(1280, 960)  mode=auto  canvas=(1280, 960)   passes=1
  OK bloom_chain res=(960, 960) mode=auto canvas=(960, 960)    passes=5

  bad: 0 of 11
  ```

- **A `resolution` of `[0, 0]` in a hand-edited `document.json`** raises
  `moderngl.Error: invalid color attachment` at load. `load_documents_from_dir` catches it per
  document (`ui_models.py:612-615`); `project_session._load_one_document_from_disk` does not. **But
  this is pre-existing, not a 090 regression:** the old loader passed `metadata.get("canvas_size")`
  through the same `_as_canvas_size`, which accepts `[0, 0]` as a valid int pair. Exposure
  unchanged by this diff; out of scope for this review.
- **`pending_resolution` cleared wholesale while only consumed for the planned set.**
  `_resolve_resolutions` (`ui.py:379-380`) empties the dict at the head, then applies entries only
  for `document_ids`. I probed both the "document outside the planned set" case and the "Examples
  popup opens on the consuming frame" case: the first never occurs (a document with a pending
  commit is the current document or a first-render candidate, both in `tick_documents`), and the
  second drops the entry but is **self-healing** — the picker already wrote
  `document.resolution` and `ui_state.resolution` in place, and the Fixed branch re-derives
  `wanted` from `resolution` on every subsequent frame. Verified: canvas reached (256, 144) three
  ticks after the popup closed. The parked entry is belt-and-braces, not the mechanism.
- **The Auto export-size change through the picker.** Settled the live canvas at a 4:3 region,
  then wrote a 16:9 export resolution: the live canvas correctly followed to (800, 450), aspect
  1.7778 matching the stored 1.7778. The `_clamped_to_aspect` + `auto_canvas_size` pair closes on
  the stored aspect as D2 requires.
- **The always-on profiler's query ring.** With `enabled=True` permanently, the ring's only
  eviction rule never fires, so `document:<uuid>` span paths are permanent GL query names.
  Measured: 3 query names per document opened (15 for five documents, none freed on close), and 37
  names total after one visit to the Examples popup with all six shipped examples. Bounded by
  distinct span paths a session visits, which is small. **Already recorded in
  `conventions.md`** ("which the app itself never does since 090 made recording always-on, so the
  ring grows with the span paths a session visits and is bounded by them") — a conscious, documented
  trade, not an oversight.
- **The always-on profiler on the export and probe paths.** Both are unaffected: `render_media`
  and `_probe_frame` call `Document.render` without a `profiler=` argument, taking the
  `NULL_PROFILER` default (`document.py:731`), so an export's passes never land in a live frame's
  tree. The spec's out-of-scope item holds.
- **`_probe_frame`'s aspect from `resolution`.** The probe renders the document at its **live**
  size and box-filters down to a probe size whose aspect now comes from `resolution`. Under Auto
  those two aspects are kept equal by the Auto loop (`auto_canvas_size` derives from
  `resolution`'s aspect, `_clamped_to_aspect` preserves it), so the box filter is not distorting.
  Under Fixed they are equal by definition.
- **The id→title map after a rename.** `_document_titles` is rebuilt every frame from live
  `ui_state.ui_name`; the rename site is `tabs/document.py:180` writing that same field. No
  staleness. (The closed-document case is MINOR-3, a different thing.)
- **The two-number chip when the current document has no plan entry.**
  `_current_document_fps` returns `None` when `plan is None` or `intervals.get(id, 1) <= 1`, and
  `fps_overlay` draws today's single `"{fps} FPS"` on `None`. Correct on every path.
- **`throttle_color`'s bands.** Green at ≤ 1.0, warn above, error above 1.5 or whenever the frame
  misses — matches D9c, and the `frame_over_budget` clause is computed from `profile.cpu_ms >
  budget_ms` in `profile_rows_plan`, which is the UI frame's own period. `budget = 0.0` is guarded
  (`ratio = 0.0`), though `UIAppState`'s `ge=0.1` makes it unreachable.
- **The settings rows.** `document_gpu_budget` round-trips through `round(x * 100)` / `/ 100.0`
  with `v_min=10, v_max=100` and `always_clamp`, matching the model's `ge=0.1, le=1.0`. No way to
  write an out-of-range value from the UI.

---

## Falsifier verification

Five mutations re-applied in the worktree, each confirmed red on the named assertion and then
restored (`git status --porcelain` clean, `git diff --stat` empty after the last restore).

| # | Mutation | Applied at | Result |
|---|---|---|---|
| M1 | `math.ceil` → `math.floor` in the current document's interval | `render_plan.py` (pure) | `test_render_plan.py`: **5 failed, 13 passed** — examples 2, 3, 7, 9 and the phase test |
| M2 | drop `renders_this_frame` from the normal render branch | `ui.py` (the LOOP) | `test_render_decoupling_loop.py`: `test_an_expensive_current_document_advances_once_in_twelve_frames` **failed**, 14 passed |
| M3 | drop the `renders_this_frame` guard on step 8's `begin_frame` | `ui.py` (the LOOP) | same test **failed**, 14 passed — the two halves are separately asserted, as the Implementation notes claim |
| M4 | `CanvasResampler.blit` → `copy_framebuffer` | `channel_blit.py` (GL) | `test_canvas_resample.py`: **2 failed** (`..._keeps_the_live_canvas_and_every_history`, `..._seeded_history_at_another_size_is_resampled_not_dropped`), 1 passed |
| M5 | restore `app.profiler.enabled = app.fps_details_open` | `ui.py` | `test_render_decoupling_loop.py`: `test_gpu_spans_record_while_the_panel_is_closed` and `test_the_costs_refresh_from_the_profile` **failed**, 13 passed |

M2 and M3 are the valuable pair: they confirm the Implementation notes' claim that the render gate
and the feedback gate are two independent reads of the plan, each with its own red test. That is
also what makes MAJOR-1 a real gap rather than a theoretical one — the same two-reads structure
exists on the pass-settings branch with only one of them wired.

---

## Verdict

**PASS-WITH-MINORS** — one MAJOR (a third render branch missing the plan gate, one line to fix,
same class as the round-2 blast R3 finding that was closed for the Examples popup) and three
MINORs (a non-settling Auto request outside the canvas bounds, costing redundant work but no GL
churn; unpruned ephemeral state dicts; a uuid fallback in a closing document's panel row). The
arithmetic, the GL lifecycle, the resample funnel, the export resolution and the eleven
hand-edited documents are all correct, and the verification items' falsifiers do what the
Implementation notes say they do.
