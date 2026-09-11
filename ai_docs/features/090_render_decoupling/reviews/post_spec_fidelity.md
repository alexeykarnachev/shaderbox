# 090 — post-implementation spec-fidelity review

**Artifact:** commit `f06d957` on `dev`, 50 files, +2666/−236.
**Contract:** `01_spec.md` in full, checked against `02_throttle_and_resolution.md`'s locked D1–D11.
**Method:** every row walked against the code; the nine plan examples run through the real
`plan_render_set`; the eleven `document.json` files opened; nine falsifiers re-applied in a
detached worktree at `f06d957` and each watched go red on the assertion the spec names; the
always-on query probe re-run.

**Verdict: PASS-WITH-MINORS.**

`make gates` re-run here: **exit 0**, captured unpiped, `== gates: GREEN -- check passed, test
passed, smoke passed ==`; the smoke RAN (this box has a display), so it is a pass and not a skip.

One MAJOR: a fourth `Document.render` call site — the PASS_SETTINGS branch — reads no interval,
so with that modal open a throttled document renders **24 of 24 frames** while its feedback
advances 2 times. Measured, not argued (below). Two MINORs are bookkeeping: the always-on query
probe was neither landed under `probes/` nor recorded as a one-shot, and two *Files touched*
entries are listed but untouched.

---

## Coverage table

Status: **LANDED** / **PARTIAL** / **MISSING** / **DEVIATED**. A "defined but unwired" mechanism
is MISSING; for each safety the line that READS it is named.

### Design decisions

| Row | Spec passage | Code | Status |
|---|---|---|---|
| D1 — `ResolutionMode` StrEnum, GL-free, in `render_shape.py` | D1 ¶1 | `render_shape.py:13-22` (`AUTO="auto"`, `FIXED="fixed"`) | LANDED |
| D1 — `UIDocumentState` gains exactly two fields, named `resolution_mode` / `resolution`, defaults `AUTO` / `DEFAULT_CANVAS_SIZE` | D1 code block | `ui_models.py:148-155` — field names, types and defaults verbatim | LANDED |
| D1 — no second size field; round 1's `export_size` gone | D1 ¶2 | `grep export_size shaderbox/` → nothing | LANDED |
| D1 — `canvas_size` keeps name/type, single writer `set_canvas_size`, now the effective live size | D1 ¶3 | `document.py:299-303` field, `document.py:388` the one writer; `grep 'canvas_size ='` finds no second writer | LANDED |
| D1 — every write funnels through `clamp_canvas_size`, Auto included | D1 ¶3 | `document.py:388-390` → `_clamped_to_aspect` → `clamp_canvas_size` | LANDED |
| D1 — a clamp that changes aspect resolves on the CONSTRAINED axis, re-derives the other from `resolution`'s aspect (F2) | D1 ¶3 | `document.py:91-106` `_clamped_to_aspect`. Verified: a 900×300 region against a 4:3 stored aspect converges to (400,300), aspect exactly 4/3, no drift over 30 frames | LANDED |
| D2 — `PreviewCellResult.drawn_size: tuple[float,float]` | D2 ¶1 | `ui_primitives.py:1151-1154` field; written at `ui_primitives.py:1276` with `(dw, dh)` — the post-fit size, not `cell_w` | LANDED |
| D2 — recorder: `widgets/document_grid.py::draw_document_preview_grid` | D2 ¶1 | `document_grid.py:92-96`. **Read by** `ui.py:371` (`displayed = app.displayed_sizes`) → `ui.py:385` `auto_canvas_size(displayed.get(id), …)` | LANDED |
| D2 — recorder: `popups/examples.py`'s grid | D2 ¶1 | `examples.py:97-107`. Same reader | LANDED |
| D2 — recorder: `ui.py::_draw_document_image` records `image_width`/`image_height` | D2 ¶1 | `ui.py:830`. Same reader | LANDED |
| D2 — `draw_document_preview_button` stays `app`-free, returns the result up | D2 ¶1 | signature unchanged; both call sites take the return value | LANDED |
| D2 — pass-strip tiles are NOT recorders | D2 ¶1 | `widgets/pass_list.py` untouched by the commit; no `record_displayed_size` there | LANDED |
| D2 — `auto_canvas_size(displayed, aspect, previous)`, `None` answers `previous` | D2 code block | `render_plan.py:96-112`; `None` and a non-positive pair both return `previous` | LANDED |
| D2 — the five aspect sites read `resolution`, never the live canvas (F2) | D2 ¶3 | (1) `ui.py:824` `_draw_document_image`'s `image_aspect`; (2) `tabs/document.py:115` `_canvas_presets`'s `current`; (3) `widgets/details.py:85-86` the aspect lock's `full_w,full_h` + `adjust_size`; (4) `widgets/details.py:106` `draw_media_details`'s aspect; (5) `copilot/backend.py:2171` the probe's `cw,ch`. All five | LANDED |
| D2 — several surfaces → the largest wins | D2 ¶3 | `app.py:1200-1211` `record_displayed_size` keeps the larger area | LANDED |
| D3 — `apply_damping(state, requested, current) -> tuple|None`, PURE, in `render_plan.py` | D3 | `render_plan.py:115-140`; imports are `math` + `dataclasses` only | LANDED |
| D3 — `_tick_frame_state` applies a non-`None` answer (blast F5) | D3 | `ui.py:388-390`. **Read by** the loop, falsifier re-applied and red (below) | LANDED |
| D3 — `AUTO_RESIZE_DEAD_BAND = 0.05`, `AUTO_RESIZE_STABLE_FRAMES = 8` | D3 | `render_plan.py:31`, `render_plan.py:37` — exact | LANDED |
| D3 — `AutoSizeState` on `App`, never persisted | D3 | `app.py:1276` `auto_size_states`; not a pydantic field anywhere | LANDED |
| D4 — the funnel resamples the output pass's LIVE canvas AND each history, before any release | D4 / *Feedback resample* | `document.py:388-404`; `resample_canvas` at `document.py:383-405` allocates → blits → releases | LANDED |
| D4 — a non-output pass's history resamples to `entry.target.target_size(new)` (blast F14) | D4 / funnel step 3 | `document.py:398-403` | LANDED |
| D4 — `_seed_feedback`'s size equality becomes the resample TARGET; dtype/components stay strict | D4 ¶2 | `document.py:552-589`: `stored_size` only checked non-`None`; dtype and `components != 4` still reject; `resample_canvas(canvas, expected_size)` at `document.py:589` | LANDED |
| D4 — `_feedback_canvas`'s resize branch routes through `resample_canvas` | *Feedback resample* ¶last | `document.py:640-642` | LANDED |
| D4 — `_feedback_generation` carries over unchanged | funnel step 4 | not touched in `set_canvas_size`; only `reset_feedback` clears it | LANDED |
| D4/R2 — the picker's Fixed write does NOT resize in place; records `App.pending_resolution` | D4 ¶3 | `tabs/document.py:85-105` parks; **consumed by** `ui.py:377-383` in step 4 | LANDED |
| D4/R2 — the other two `set_canvas_size` callers are already before the draw | D4 ¶3 | step 4 (`ui.py:389`) and the copilot bridge drained at the head of `_tick_frame_state` | LANDED |
| D5 — the picker is unchanged, writes `resolution` in both modes | D5 ¶1 | `tabs/document.py:85-105`, `tabs/document.py:193`, `tabs/document.py:238` | LANDED |
| D5 — caption reads `Canvas` under Fixed, `Export` under Auto | D5 ¶1 | `tabs/document.py:169-175` (caption), `tabs/document.py:98-104` (the notification label) | LANDED |
| D5 — no default-derivation rule; a hand-edited number is kept | D5 ¶1 | nothing computes a default; the loader copies `ui_state.resolution` straight through (`ui_models.py:589-594`) | LANDED |
| D5 — `render_media` passes `self.resolution` to `resolve_dims` on BOTH branches | D5 ¶2 / *Export* | `document.py:1033` (scratch canvas `size=self.resolution`) and `document.py:1044` (`resolve_dims(preset, self.resolution)`) | LANDED |
| D5 — NATIVE / the FREE fall-through / `preset=None` all resolve to `resolution` | D5 ¶2 | verified by running the three export paths: source canvas 640×480 on all of `None`, `native`, and 1280×720 on `wide`, with the live canvas at 320×180 | LANDED |
| D5 — `_artifact_matches_shape` resolves against the same number (blast F1) | D5 ¶2 / *Export* | `exporters/youtube.py:515-518` | LANDED |
| D5 — `resolution_details` preserved exactly; `render_media` does NOT overwrite it (blast F2) | D5 ¶3 | `document.py:1030-1036` — the `preset=None` branch writes nothing to `details`; only the preset branch sets it, as before | LANDED |
| D5 — the `preset=None`/`SCALE_DISTORT` branch mints a scratch `Canvas` released in a `finally` | *Export* ¶2 | `document.py:1027-1039` | LANDED |
| D5 — `Document` gains `resolution_mode` / `resolution` as plain fields | *Export* ¶last | `document.py:304-308` | LANDED |
| D6 — `shaderbox/render_plan.py`, new leaf, `math` + `dataclasses` only | D6 / *The plan function* | `render_plan.py:13-14`; no GL / imgui / App / Document import | LANDED |
| D6 — `k_current = ceil(cost / (budget × period))`; current takes the budget first | D6 / semantics (3) | `render_plan.py:184-188` | LANDED |
| D6 — others share the remainder at one common fps | semantics (4) | `render_plan.py:193-208` | LANDED |
| D6 — `MAX_INTERVAL = 60` binds EVERY `k`, not only `f = 0` (F7) | D6 | `render_plan.py:19` constant; `_clamp_interval` at `render_plan.py:224` applied on BOTH the current path (`:185`) and the others path (`:203`). Example 9 returns 60, not 143 | LANDED |
| D6 — phase offset by stable index; gate is `(frame_idx + phase[id]) % k != 0` (F6) | D6 | `render_plan.py:215` `phases[id] = index % interval`; gate at `ui.py:342` | LANDED |
| D6 — budget default 0.5 | D6 | `ui_models.py:252` `Field(default=0.5, …)` | LANDED |
| D6 — `INTERVAL_HYSTERESIS_FRAMES = 4` | D6 | `render_plan.py:25`; applied in `_settle` at `render_plan.py:227-240` | LANDED |
| D6 — intervals and phases ephemeral | D6 | `app.py:1277` `throttle_states`, `app.py:1279` `render_plan`; neither persisted | LANDED |
| D7 — `CostRecord` with `gpu_ms` + `cpu_ms`; policy reads `gpu_ms` alone | D7 | `render_plan.py:40-49` dataclass; `_cost_of` at `render_plan.py:219-221` reads `gpu_ms` only | LANDED |
| D7 — ONE write site: `_tick_frame_state` step 3; the render site writes nothing | D7 | `ui.py:293` `_refresh_document_costs(app)`, defined `ui.py:347-358`. No other writer of `app.document_costs` in `shaderbox/`. **Read by** `ui.py:296-303` (the `plan_render_set` call) | LANDED |
| D7 — render sites open `f"document:{document_id}"` | D7 / *Profiler* | `ui.py:463`, `ui.py:478`, `ui.py:500`, `ui.py:507` — all four use `_DOCUMENT_SPAN_PREFIX` + id; no title anywhere | LANDED |
| D7 — `profile_rows_plan` renders the title via an `id -> title` map | D7 | `ui_primitives.py:1382-1387` signature; `_document_row` at `ui_primitives.py:1440-1470`; map built at `ui.py:924-940` | LANDED |
| D7 — no record → `k = 1` | D7 / example 8 | `_cost_of` returns 0.0; `render_plan.py:185` keeps `1` when `cost <= budget_ms`. Example 8 verified | LANDED |
| D8 — `session.tick` keeps running over the full set at the UI rate | D8 | `ui.py:311-312` — `tick_documents`, not `planned_documents`, and ungated | LANDED |
| D8 — `begin_frame` is NOT called on a skipped frame | D8 | `ui.py:326-328`. **Read by** step 8's `renders_this_frame`; falsifier re-applied and red | LANDED |
| D9a — `app.profiler.enabled = app.fps_details_open` goes; `Profiler(enabled=True)` | D9a / *Profiler* | the line is gone from `ui.py:958-960`; `app.py:493` constructs `Profiler(enabled=True)` | LANDED |
| D9a — the panel's open state decides only what is drawn | D9a | `ui_primitives.py:1600` — `is_open` gates the child only | LANDED |
| D9a — the smoother's feed stays gated on `profiler.enabled`, now always true | *Profiler* ¶1 | unchanged; `test_profiling.py` re-checked against always-on (its close-the-panel case now asserts `app.profiler.enabled` and a non-empty ring) | LANDED |
| D9b — `fps_overlay` gains `document_fps: int | None`; `None` draws today's single number | D9b | `ui_primitives.py:1564` param; `ui_primitives.py:1583` `f"{fps} FPS" if document_fps is None else f"{fps} | doc {document_fps}"` — the spec's exact `60 | doc 10` shape | LANDED |
| D9b — the chip is passed the current document's rate | D9b | `ui.py:960` `document_fps=_current_document_fps(app)`, defined `ui.py:914-922` | LANDED |
| D9c — `ProfileRow` gains NO new fields (F11) | D9c | one field added, `tooltip` — see *Deviations* row 1; `k` and `share` are indeed not fields | DEVIATED (declared, justified) |
| D9c — the compact row reads `10 fps ×6  72%`, cost in the tooltip (blast m3) | *Profiler* ¶4 | `ui_primitives.py:1464` `f"{document_fps:.0f} fps x{interval}  {ratio*100:.0f}%"`, `tooltip=f"{ms:.2f} ms"`. Tooltip **read by** `ui_primitives.py:1548-1549` | LANDED |
| D9c — an unthrottled row keeps `"12.34 ms"` | *Profiler* ¶4 | `ui_primitives.py:1453-1456` | LANDED |
| D9c — color is `throttle_color(share / budget, frame_over_budget)` | *Profiler* ¶4 | `ui_primitives.py:1467`; `share = ms * document_fps / 1e3`, `ratio = share / budget` | LANDED |
| D9c — `throttle_color`'s three bands: OK ≤ 1.0, WARN above 1.0, ERROR above 1.5 or frame over target (F10) | D9c | `theme.py:339-356`: `THROTTLE_WARN_RATIO = 1.0`, `THROTTLE_ERROR_RATIO = 1.5`; `frame_over_budget` short-circuits to `STATE_ERROR`. Not `load_color` | LANDED |
| D9c — `frame_over_budget` is a real input | D9c | `ui_primitives.py:1407` `profile.cpu_ms > budget_ms` | LANDED |
| D10 — Render all keeps its meaning | D10 ¶1 | `ui.py:250-268` unchanged | LANDED |
| D10 — the Examples popup's set REPLACES the normal one while open (blast R3) | D10 ¶2 | `ui.py:275-287`: `planned_documents`, `planned`, `current_planned` all switch on `examples_open`; the normal set is not planned that frame | LANDED |
| D10 — `begin_frame` over that same set, taking the gate | D10 ¶2 | `ui.py:326-328` iterates `planned_documents` | LANDED |
| D10 — the examples render loop takes the `(frame_idx + phase) % k` gate | D10 ¶2 | `ui.py:497-498`. **Read there**; V9a covers it | LANDED |
| D10 — `current` is the popup's own selection or `None` | D10 ¶2 | `ui.py:283-287` | LANDED |
| D11 — `is_throttle_documents: bool = True` | D11 | `ui_models.py:251`. **Read by** `ui.py:302` (`enabled=`) | LANDED |
| D11 — `document_gpu_budget: float = Field(default=0.5, ge=0.1, le=1.0)` | D11 | `ui_models.py:252` — exact bounds. **Read by** `ui.py:301` and `ui.py:962` | LANDED |
| D11 — both beside `global_target_fps`, everything else a code constant | D11 | `ui_models.py:246-252`; the four constants live in `render_plan.py` | LANDED |

### Data model changes

| Row | Code | Status |
|---|---|---|
| After-shape: no top-level `canvas_size`; `ui_state.resolution_mode` + `ui_state.resolution` | `ui_models.py:396-399` (`save` writes no `canvas_size`); all eleven files verified below | LANDED |
| The load order reshaped: read raw metadata → parse `ui_state` FIRST → resolve size → pass to `load_from_dir` (F4) | `ui_models.py:583-594` | LANDED |
| `load_from_dir` reads no top-level `canvas_size` and no `ui_state` key | `document.py:822-849` — takes `canvas_size` as a parameter, reads neither | LANDED |
| `UIAppState`'s two fields both defaulted, `extra='forbid'` rejects nothing | `ui_models.py:251-252` | LANDED |
| Five ephemeral `App` fields + `pending_resolution` | `app.py:1274-1279` — all six | LANDED |

### The eleven `document.json` hand-edits (each file opened)

| file | expected | found | Status |
|---|---|---|---|
| `…/document_examples/0b0d16bb-…` | `[1080,1920]` auto | `[1080,1920]` auto | LANDED |
| `…/53724dbd-…` | `[1280,960]` auto | `[1280,960]` auto | LANDED |
| `…/73ea2431-…` | `[1280,1280]` auto | `[1280,1280]` auto | LANDED |
| `…/77a84d27-…` | `[512,512]` **fixed** | `[512,512]` **fixed** | LANDED |
| `…/8d454b7b-…` | `[1600,900]` auto | `[1600,900]` auto | LANDED |
| `…/f90f5ff9-…` | `[1280,960]` auto | `[1280,960]` auto | LANDED |
| `projects/dev/documents/e7e00c46-…` | `[1280,960]` auto | `[1280,960]` auto | LANDED |
| `projects/dev/documents/ec926580-…` | `[1280,960]` auto | `[1280,960]` auto | LANDED |
| `projects/documents/1901ab60-…` | `[960,960]` auto | `[960,960]` auto | LANDED |
| `projects/documents/307598da-…` | `[1280,960]` auto | `[1280,960]` auto | LANDED |
| `tests/fixtures/bloom_chain/` | `[960,960]` auto | `[960,960]` auto | LANDED |

Every file: top-level keys exactly `['ui_state','uniforms']` — no `canvas_size` survives anywhere.
`git ls-files '*document.json'` returns exactly these eleven.

### Frame integration, in order

| Step | Spec | Code | Status |
|---|---|---|---|
| 1 — existing head unchanged, abort path still returns `None` first | *Frame integration* 1 | `ui.py:180-228`; the early `return None` precedes everything new | LANDED |
| 2 — render set built by `tick_documents`; examples REPLACE it | 2 | `ui.py:243-287` | LANDED |
| 3 — costs refresh from `document:` spans, the ONE write site | 3 | `ui.py:293` | LANDED |
| 4 — resolution resolves, consuming `pending_resolution` FIRST | 4 | `ui.py:377-383` consumes before the mode branch at `ui.py:384` | LANDED |
| 4 — Fixed takes `resolution`; Auto asks `auto_canvas_size` then `apply_damping` | 4 | `ui.py:384-390` | LANDED |
| 5 — feedback and the live canvas resample for every changed document | 5 | inside `set_canvas_size` (`document.py:388-404`), reached from step 4 | LANDED |
| 6 — the plan runs → `app.render_plan` | 6 | `ui.py:296-304` | LANDED |
| 7 — the script ticks over the FULL set | 7 | `ui.py:311-312` — `tick_documents` | LANDED |
| 8 — `begin_frame` only for documents the plan admits | 8 | `ui.py:326-328` | LANDED |
| The render block gates the output render AND the pending-pass sweep together | ¶after 8 | `ui.py:461` — one `if` covers both `document.render` calls | LANDED |
| Steps 4–5 precede step 6 | ¶after | `ui.py:295` before `ui.py:296` | LANDED |
| **Every render site reads the plan** | implied by 8 + the render-block ¶ | `ui.py:461` (normal), `ui.py:498` (examples), `ui.py:326` (begin_frame) — but **`ui.py:502-508`, the PASS_SETTINGS branch, does not** | **PARTIAL — MAJOR-1** |

### Files touched

| File | Spec says | Found | Status |
|---|---|---|---|
| `render_plan.py` (new) | all five types + three functions + constants | all present, leaf | LANDED |
| `render_shape.py` | `ResolutionMode` | yes | LANDED |
| `channel_blit.py` | the resampler beside `ChannelBlit` | `CanvasResampler` at `channel_blit.py:92-123`, one-quad draw, module docstring updated | LANDED |
| `document.py` | six items | all six | LANDED |
| `ui_models.py` | four items incl. the reordering | all four | LANDED |
| `ui.py` | seven items | all seven | LANDED |
| `app.py` | six fields + `Profiler(enabled=True)` | yes (+ `record_displayed_size`, a helper, not new scope) | LANDED |
| `ui_primitives.py` | three items | all three | LANDED |
| `theme.py` | `throttle_color` | yes | LANDED |
| `tabs/document.py` | five items | all five (+ the `Fixed` toggle, which D5's "the picker is unchanged" does not forbid — D1 needs a mode control somewhere and *Files touched* says "the mode control") | LANDED |
| `widgets/details.py` | aspect lock + two preset labels | yes, plus `draw_media_details`'s aspect (a fifth aspect site D2 names) | LANDED |
| `widgets/document_grid.py`, `popups/examples.py` | each records `drawn_size` by id | yes | LANDED |
| `popups/settings.py` | D11's two rows | `settings.py:90-101` — the spec's snippet verbatim, no `help_marker` | LANDED |
| `help_content.py` | `u_resolution` gains the Auto clause | `help_content.py:42` | LANDED |
| `copilot/backend.py` | three items | `backend.py:1252-1258` (Fixed write), `backend.py:2171` (probe aspect), `backend.py:280-290` (`_canvas_line`) | LANDED |
| `exporters/youtube.py` | `_artifact_matches_shape` | yes | LANDED |
| the eleven `document.json` | table | yes | LANDED |
| `tests/test_render_plan.py` (new), `test_document_shapes.py` (new) | V1/V2/V2a/V3, V12 | both present | LANDED |
| edits to `test_canvas_presets.py`, `test_graph_persistence.py`, `test_pass_hot_reload.py`, `test_uniform_row_pruning.py`, `test_document_dir_sync.py`, `test_feedback_persistence.py`, `test_canvas_fields.py`, `test_render_for.py`, `test_profiling.py`, `test_ui_prose_budget.py`, `test_youtube_exporter.py` | eleven files | all eleven in the stat | LANDED |
| `test_persistence_completeness.py` | listed as edited | **untouched** — the module drives `UIAppState.load` generically over a corruption battery, so the reshaped model needs no edit; V14's claim ("passes against the reshaped models") is what actually holds | DEVIATED (MINOR-2) |
| `test_document_ops.py` | listed as edited; V13 says "extend `test_set_canvas_size_applies_and_clamps`" | **untouched**; V13 landed instead as `test_render_decoupling_loop.py::test_the_copilots_canvas_size_survives_the_next_frame`, which asserts strictly more (mode, resolution, and survival across a driven tick) and needs the frame rig + xdist group that file carries | DEVIATED (MINOR-2) |
| `ai_docs/dev_flow.md` module map | one line for `render_plan.py` | `dev_flow.md:379-384` (plus a `ResolutionMode` clause on `render_shape.py`) | LANDED |
| `ai_docs/conventions.md` | three items | two new Design decisions; the `update_and_draw` quirk rewritten to xdist GROUPS; the ring's eviction rule annotated | LANDED |
| `ai_docs/roadmap.md` | row + banner | the 090 row, the rewritten banner, 088's row corrected | LANDED |
| **Not in *Files touched*, changed anyway** | — | `tests/test_document_graph.py`, `test_probe_clock_and_turn_end.py`, `test_radiance_cascades_example.py`, `test_canvas_resample.py` (new), `test_render_decoupling_loop.py` (new) | DEVIATED — all five declared in *Implementation notes* deviation 5, each a mechanical consequence of a locked decision |

### Verification items

Each row: does the test exist, does it assert what the spec says, does it read the CONSUMER,
and is its recorded falsifier the spec's? **Re-applied** marks a falsifier I re-ran here.

| V | Test | Asserts the spec's claim? | Consumer-facing? | Falsifier | Status |
|---|---|---|---|---|---|
| V1 | `test_render_plan.py`, 9 tests | yes — exact `k` and exact phase per row | pure function, correctly scoped | **Re-applied 3:** `ceil`→`floor` → 5 red incl. example 2 (11 not 12); drop `MAX_INTERVAL` on the common-fps path → example 9 red at 143; `phases=0` → V2a red. All restored | LANDED |
| V1 — the nine examples run through the REAL function | — | **all nine reproduce the spec table exactly**: 1 `k=1`/phase 0/59.88; 2 `k=12`/4.99; 3 `k=5` + three at 43, phases 1,2,3; 4 `k=1` + twenty at 3 cycling `i%3`; 5 `k=1`+23; 6 all 1/all phase 0; 7 `k=3`/19.96; 8 all 1; 9 `k=5` + ten at 60, fps 1.0 | — | — | LANDED |
| V2 | `test_a_cost_hovering_…`, `test_a_steady_cost_moves_…` | 8.0/8.6 never moves across 20 frames; steady 100 ms moves on exactly frame 4 | pure | recorded: apply the candidate at once | LANDED |
| V2a | `test_three_same_interval_previews_…` | distinct phases {1,2,3}; no frame admits >1 | pure | **Re-applied**, red | LANDED |
| V3 | `test_a_drag_ramp_…` + 3 siblings | ≤4 applied across the 764→940 ramp; a held size lands on frame 8; past-band applies at once; unchanged never re-applies | pure | recorded: drop the dead band / drop the stability clause | LANDED |
| V3a | `test_the_loop_damps_a_drag_ramp_…` | drives `_tick_frame_state` via `_drive`, counts `set_canvas_size` calls, bound ≤5 | **yes — counts the loop's calls** | recorded: raw request → ~60 | LANDED |
| V4 | `test_canvas_resample.py::test_a_resize_keeps_…` | allocations−releases constant across six resizes; far-corner > 8 on the live canvas AND each history; trail history (32,32), show (64,64) | reads through `texture_to_rgba8`, asserts on the RESIZE frame via `_assert_resized` | **Re-applied 1 of 5:** `Canvas.set_size` on the live canvas → red on "the live output canvas came back blank at (96,96)". The other four recorded | LANDED |
| V5 | `test_render_for.py::test_an_export_never_reads_…`, 3 params | asserts the file size **and** the canvas each export rendered into (`_record_source_sizes` spy) | **yes — records the source canvas, the thing D5 moves** | **Re-applied:** live-canvas source → `[None]` and `[native]` red. Restored | LANDED |
| V5 (the gate) | `test_an_artifact_still_matches_…` | still matches after a live resize to 320×180 | yes | recorded | LANDED |
| V5a | `test_the_render_tabs_own_size_…` | `resolution_details` 640×480 over `resolution` 1920×1080 → file 640×480 | yes | recorded: overwrite → 1920×1080 | LANDED |
| V6 | `test_a_scaled_feedback_pass_survives_…` | output not blank, trail contributes (near > far+32), **and** trail canvas (32,32) vs show (64,64) | yes — the size is the assertion, since the picture cannot see it | recorded | LANDED |
| V7 | `test_an_expensive_current_document_…` | `k == 12`; `begin_frame` ≤2 in 12 frames; cheap sibling 12; **and** render count ≤2 over 12 `update_and_draw` | **yes — both reads of the plan separately** | **Re-applied 2:** cut only the render-site gate → red; cut only step 8's gate → red. Each half is independently caught | LANDED |
| V8 | `test_the_throttle_off_…` | every document 12/12; every interval 1 | yes, `_drive` bumps `frame_idx` | recorded | LANDED |
| V9 | 4 tests (viewer, grid, examples, no-recorder) | viewer case discriminates on the recorded **SIZE** (> `THUMB_LG`), not the key — the grid tile also records the current document | yes | **Re-applied:** delete the viewer's `record_displayed_size` → red on the size assertion. Others recorded | LANDED |
| V9a | `test_a_throttled_example_renders_less_often_…` | counts **renders**, not `_frame`; expensive < cheap, cheap ≥ 20/24 | yes — the notes record that the first version measured the wrong gate and was fixed | recorded: remove the examples gate | LANDED |
| V10 | `test_the_chip_carries_…` | kwargs spy on `ui.fps_overlay`; `document_fps is not None` and `== round(target_fps / k)` | yes | recorded: `document_fps=None` unconditionally | LANDED |
| V10a | `test_the_panel_rows_match_documents_by_id_…` | two same-titled rows with different numbers; one reads `x5` + tooltip `40.00 ms`, the other ends in `ms` | yes | "covered by construction" — honest: one lookup cannot fill both | LANDED |
| V11 | `test_gpu_spans_record_while_the_panel_is_closed` + `test_profiling.py`'s wire test | `fps_details_open=False`, 4 frames, a `document:` child with a non-`None` `gpu_ms`; the wire test now asserts `enabled` and a non-empty ring | yes | recorded: restore the deleted line | LANDED |
| V12 | `test_document_shapes.py` | domain is `git ls-files '*document.json'`; a census guard (`>= 11`) against an empty parametrization; no top-level `canvas_size`; both keys; **mode per file** | yes | recorded: three breaks on one tracked file | LANDED |
| V13 | `test_the_copilots_canvas_size_survives_the_next_frame` | mode FIXED, `resolution == (128,200)`, survives a driven tick against a competing `displayed_sizes` entry, texture is (128,200) | yes | recorded: leave the mode Auto | LANDED (home moved — MINOR-2) |
| R2 | `test_a_pickers_commit_applies_on_the_next_tick_…` | canvas has NOT moved; `pending_resolution` holds the pair; next tick applies it; the dict is consumed | yes | **Re-applied:** resize in place → red on the first assertion | LANDED |
| Costs (D7 step 3) | `test_the_costs_refresh_from_the_profile` | `document_costs` populated after 6 real frames | yes | recorded: drop `_refresh_document_costs` | LANDED |
| V14 | `test_roadmap_shape.py`, `test_prose_spelling.py`, `test_ui_prose_budget.py`, `test_persistence_completeness.py` | the m2 rationale corrected, an `fps_overlay` budget exemption added for the two-number chip | yes | — | LANDED |

### Implementation notes — deviations

| # | Claim | Verdict |
|---|---|---|
| 1 | `ProfileRow` gained `tooltip` | **Justified.** F11 dropped `k`/`share` *because they duplicated the formatted number*; `tooltip` carries the millisecond cost the compact form deliberately drops, which nothing else in the row can reach. D9c *requires* that tooltip. The tooltip is read at `ui_primitives.py:1548`, so it is wired, not decorative |
| 2 | `profile_rows_plan` gained `budget` | **Justified.** D9c's color is `throttle_color(share / budget, …)` and the pre-existing `budget_ms` is the frame period, a different quantity; it had to arrive from `UIAppState` |
| 3 | `_load_document_metadata` → public `load_document_metadata` | **Justified.** F4's reordering makes `ui_models.py` a second caller; the alternative is a second `json.load` of the same file |
| 4 | `as_canvas_size` public | **Justified.** D1 deletes the raw pair the malformed-pair test exercised; the coercion still guards `Document.__init__`, so the test moved onto it directly |
| 5 | Six test files beyond the list | **Justified and mechanical**, but the note says "six" and names three (`test_document_graph.py`, `test_probe_clock_and_turn_end.py`, `test_radiance_cascades_example.py`). The two new files (`test_canvas_resample.py`, `test_render_decoupling_loop.py`) are the other two the stat shows; the count is loose prose, not a scope change |
| 6 | `test_profiling.py` got an `xdist_group` | **Justified** — the corrected quirk requires it of any frame-driving module |
| — | `make gates` exit 0, smoke RAN, 2153 passed / 4 skipped | **Re-run here: exit 0, green, smoke ran.** The banner line is verbatim |

---

## Findings

### MAJOR-1 — the PASS_SETTINGS render site reads no interval, so the throttle is inert there while the feedback advance is not

**Spec:** *Frame integration*, step 8 and the paragraph after it: "`_update_and_draw`'s render
block gates on the same answer … because they share one `frame_idx` and splitting them would let
the sweep become the document's only measured cost." D6's gate is stated as covering how often a
document draws, and D8 pairs the render with `begin_frame` explicitly ("`begin_frame` is **not**
called on a skipped frame").

**Code:** `shaderbox/ui.py:502-508`. Three of the four `Document.render` call sites in the frame
body consult `renders_this_frame` (`ui.py:461`, `ui.py:498`, and step 8's `ui.py:326`). The
fourth — the pass-settings modal's keep-rendering branch — does not:

```python
elif (
    app.popup_state == PopupState.PASS_SETTINGS and current_ui_document is not None
):
    with app.profiler.cpu(f"{_DOCUMENT_SPAN_PREFIX}{app.current_document_id}"):
        current_ui_document.document.render(profiler=app.profiler)
```

Meanwhile step 8 *does* gate that document: with PASS_SETTINGS open, `any_popup_open()` is true
and `examples_open` is false, so `planned_documents == [current_document_id]` and the plan
computes a real interval for it, which `ui.py:326` then honours.

**Measured**, driving the real `app` fixture with a 100 ms `CostRecord` and PASS_SETTINGS open:

```
interval k=12
PASS_SETTINGS open, k=12: renders=24 begin_frames=2 over 24 frames
```

So the document submits its full GPU cost on every one of 24 frames — the throttle buys nothing
on this path — while its feedback integrates 2 times. That is exactly the disagreement between
the two gates the spec's own paragraph rules out, and it is the pass-settings modal, whose
stated purpose is "watching a wiring/target change land" on a document heavy enough to be
worth watching.

Severity MAJOR rather than BLOCKER: the mechanism is correct on the two paths that carry the
common case, the defect is confined to one modal, and it fails toward rendering more (never a
freeze). No test covers this branch — V7 and V9a cover the other two.

**Fix:** gate it like its two siblings, one line:

```python
elif (
    app.popup_state == PopupState.PASS_SETTINGS
    and current_ui_document is not None
    and renders_this_frame(app, app.current_document_id)
):
```

and add a case to `test_render_decoupling_loop.py` in the shape of V9a — open PASS_SETTINGS,
plant a 100 ms cost, count `render` calls over 24 frames, assert fewer than 24. Its falsifier is
the line above, removed.

### MINOR-1 — the always-on query probe was neither landed under `probes/` nor recorded as a one-shot

**Spec:** *Pre-implementation measurements*, last sentence: "The probe belongs under `probes/`
beside the `cost_*` / `throttle_*` siblings; the reviewer ran it from a scratchpad, so the
implementer lands it there or records it as a one-shot."

**Code:** `ai_docs/features/090_render_decoupling/probes/` contains 34 files; none measures the
profiler-on/profiler-off delta (`grep 'profiler.enabled\|fps_details_open' probes/*.py` hits only
`01_frame_timing.py` and `03_editor_draw_cost.py`, both of which *enable* recording to measure
something else). `git show f06d957 --stat -- '…/probes/*'` is empty. The *Implementation notes*
record no one-shot either. The probe exists only in the implementer's scratchpad.

Neither branch of the spec's "lands it there OR records it as a one-shot" was taken, so the one
number D9a's whole premise rests on has no artifact in the repo.

**Re-run here** (the scratchpad script, 300 frames per arm, 6 documents, interleaved blocks):

```
profiler OFF  median 6.944 ms   p95 7.695 ms
profiler ON   median 6.952 ms   p95 7.683 ms
DELTA         median +0.008 ms  p95 -0.012 ms
```

**The threshold holds**: p95 delta −0.012 ms against a stated pass threshold of ≤ 0.1 ms. (The
absolute frame period is about half the spec's 13.95 ms — a different load state on this box —
but the delta is what the threshold is about, and it is inside noise, sign-flipped from the
spec's +0.005 ms exactly as the spec predicted a single run would be.) D9a's premise is
independently confirmed; only the artifact is missing.

**Fix:** copy the scratchpad script to
`ai_docs/features/090_render_decoupling/probes/always_on_queries.py` (it already imports only
public API and takes a frame count as `argv[1]`), or add one line to *Implementation notes*
recording it as a one-shot with the two runs' numbers. The first is what the spec prefers and
costs one `cp`.

### MINOR-2 — two *Files touched* entries are listed but untouched; V13's home moved

**Spec:** *Files touched*, the tests bullet, lists `test_persistence_completeness.py` and
`test_document_ops.py` among the edits. V13 says "Extend
`test_document_ops.py::test_set_canvas_size_applies_and_clamps`".

**Code:** `git show f06d957 --stat` lists neither file. `test_persistence_completeness.py` drives
`UIAppState.load` through a generic corruption battery and never names a field, so the reshaped
model genuinely needs no edit — V14's weaker claim ("passes against the reshaped models") is the
true one and it holds (gates green). V13's assertions landed in
`test_render_decoupling_loop.py::test_the_copilots_canvas_size_survives_the_next_frame`, which is
the better home: it needs the `_drive` rig and the `xdist_group` that module carries, and it
asserts strictly more than the spec asked (mode, resolution, survival across a tick against a
competing `displayed_sizes` entry, and the texture's own size). `test_document_ops.py`'s existing
clamp test still passes unmodified — verified, 8 passed — because the copilot pre-clamps with
`clamp_canvas_size` before `set_canvas_size`, so `_clamped_to_aspect` sees `clamped == size` and
returns it untouched.

Severity MINOR: no behavior is missing, and this is a documentation drift between the spec's
*Files touched* list and where the work actually landed.

**Fix:** one sentence in *Implementation notes* — V13 landed in `test_render_decoupling_loop.py`
because it needs the frame rig, and `test_persistence_completeness.py` needed no edit because it
enumerates stores rather than fields.

---

## False trails

Recorded so the remaining post-implementation reviewers do not re-spend them.

- **The nine worked examples are arithmetically exact.** All nine re-derived through the real
  `plan_render_set` reproduce the spec's table to the digit, phases included. Any disagreement a
  later reviewer finds is in their harness, not the rule.
- **The Auto loop converges and does not drift.** Driving `auto_canvas_size` + `apply_damping`
  30 frames against a fixed region settles on (764,573) and holds; against a region of a
  different aspect it settles on (400,300) with the stored 4/3 exactly. F2 is closed at the
  arithmetic, not only in prose.
- **The copilot's clamp assertion is not broken by `_clamped_to_aspect`.** `(99999,4)` looks like
  it should become (4096,4096) through the aspect re-derivation, and in isolation it does — but
  the copilot pre-clamps to (4096,16) first, so the funnel's `clamped == size` short-circuit
  fires. `test_document_ops.py` passes unmodified; this is not a latent bug.
- **`test_persistence_completeness.py` and `test_document_ops.py` being unchanged is not a
  missing test.** See MINOR-2 — one is generic by construction, the other's coverage moved
  upward.
- **`ProfileRow.tooltip` is not F11 re-opened.** F11's reason was duplication of the formatted
  number; the tooltip carries what the compact form drops. It is also genuinely read
  (`ui_primitives.py:1548`), not a field nothing displays.
- **The `Fixed` toggle in the Document tab is not a D5 violation.** D5 says "the picker is
  unchanged" — the W×H fields and the presets chip are; D1 requires a mode and *Files touched*
  says "the mode control", so the toggle is the specified addition beside an unchanged picker.
- **`displayed_sizes` being swapped for a fresh dict rather than cleared in place is correct**
  (`ui.py:371-375`): a document that stopped being displayed must fall through to
  `auto_canvas_size(None, …)`'s "keep previous", not answer with a stale region.
- **The always-on cost is genuinely negligible** — independently re-measured here, p95 delta
  −0.012 ms. Only the probe's filing is at issue, never the number.
- **`gates` is green and the smoke really ran**, re-verified unpiped, exit code read before
  anything else. No need to re-run it to check the claim.

---

## Coverage statement

Walked: all eleven decisions D1–D11 and each sub-clause the brief enumerates (the two field names
and defaults; the five aspect sites; the three recorders and their one reader; the two damping
constants; the resample's four per-canvas steps and its release order; NATIVE, the FREE
fall-through and the `preset=None` path; the deferred Fixed write and its consumer; the plan's
seven semantic rules and four constants; the Examples-popup replacement; always-on spans keyed by
id; the chip; the compact row and its tooltip; `throttle_color`'s three bands; the two settings
and their bounds). All eight frame-integration steps in order. All 25 *Files touched* entries plus
the five changed-but-unlisted files. All eleven `document.json` files opened individually. All 24
verification items (V1–V14 with the a/b variants, V9a, and the R2 and costs items the notes add).
All six declared deviations.

Executed: the nine plan examples through the real function; `git ls-files '*document.json'`
against the table; nine falsifiers re-applied in a detached `git worktree` at `f06d957` — the
`MAX_INTERVAL` clamp, `ceil`→`floor`, the phase offset, the render-site gate alone, step 8's gate
alone, the viewer's recorder, the picker's in-place resize, the live-canvas `Canvas.set_size`, and
the export's live-canvas source — each red on the assertion the spec names, then restored (the
worktree ends clean, `git diff --stat` empty); the always-on query probe re-run at 300 frames per
arm; `make gates` re-run unpiped, exit 0; one purpose-built probe of the PASS_SETTINGS branch,
written, run and deleted.

Not executed: the maintainer's-eyes items (whether an Auto document reads sharp, whether `k = 5`
feels like iteration) — those are his by definition. Four of V4's five falsifiers and the majority
of the recorded-but-not-re-applied falsifiers were taken as recorded, since five was the brief's
floor and nine were re-run; every one re-applied matched its record exactly, which is the evidence
the unre-applied records are trustworthy.

## Row counts

| Status | Count |
|---|---|
| LANDED | 103 |
| PARTIAL | 1 |
| DEVIATED | 4 |
| MISSING | 0 |

The single PARTIAL is MAJOR-1. Of the four DEVIATED, three are declared in *Implementation notes*
and justified against the spec's own text; one (`test_persistence_completeness.py` /
`test_document_ops.py`) is MINOR-2's undeclared listing drift.

**Verdict: PASS-WITH-MINORS** — the spec is implemented end to end with every safety wired and
mutation-caught; one render path (the pass-settings modal) reads no interval and must be gated,
and two bookkeeping items want a line each.
