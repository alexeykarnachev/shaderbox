# 090 — Refinement: document throttle and Auto / Fixed resolution

Research and plan for the two levers the maintainer proposed after rejecting tiling
(`00_research.md ## Maintainer decision: no tiling`). Both reduce how much GPU work a document
submits per UI frame, which on this driver is the only thing that shortens the UI's wait. Nothing
is implemented. Reports: `research/resolution_flow.md`, `research/cadence_flow.md`,
`research/cost_and_throttle.md`; probes `probes/cost_*.py`, `probes/throttle_*.py`.

**Settled by the maintainer, carried here as premises:**

- A document has a **resolution mode**: **Auto**, the live render target sized to the displayed
  UI region, or **Fixed**, an explicit width × height for shaders whose logic needs it.
- **Export resolution is its own concept** in both modes (the render preset), never the live size.
- **A document may be throttled**: when its measured cost exceeds a budget share of the UI frame it
  renders less often, its clock still on wall time. This reverses the earlier "no budget on a
  document" constraint, on his own re-decision.

## What was measured

**Cost is linear in pixels down to ~0.5 megapixel**, then a per-draw floor takes over
(`research/cost_and_throttle.md`, heavy shader):

| Size | Cost |
|---|---|
| 1920×1080 | 224.5 ms |
| 1280×720 | 102.6 ms |
| 764×430 (the viewer at a 1920×1080 window, 50/50 split) | 40.3 ms |
| 320×180 | 9.1 ms |

Everything is displayed from the full-resolution texture today: viewer, grid tile and pass-strip
thumbnail all hand the same canvas texture to imgui and scale it (`widgets/pass_list.py`,
`widgets/document_grid.py`, `ui_primitives.py::preview_cell`), and Render all is on by default,
so a document shown only as a thumbnail renders every frame at full resolution.

**A throttle changes how often a hitch happens, never how long it is.** The 100 ms document
rendered every 12th frame leaves 275 of 300 UI frames unblocked; the remaining 25 still cost
101 ms each, because nothing preempts the draw. The share-of-wall-time formula is optimistic:
predicted 5.0 document fps at k=12, measured 3.6, since the frame that carries the document is
101 ms long, not 16.7. **Both levers together** (764×430, k=5 from the measured 40 ms):
240 of 300 frames unblocked, the remaining 60 cost 40 ms instead of 105, and the document's fps
stays at 9.5, the same as today's untouched 1280×720 render at every frame. Resolution shortens
the hitch, the throttle spaces it, and their product is the strongest configuration measured
short of a thread.

## What the code says

**Resolution** (`research/resolution_flow.md`). One field, `Document.canvas_size`, with one
writer, `set_canvas_size`, which resizes the output canvas at once; other passes catch up on
their next render. `u_resolution` and `u_aspect` are computed from the bound canvas every frame
in `Pass.render`, never cached, so a resized target is the resolution the shader sees with no
further plumbing. Two precedents already render at a size other than the live one: the
`RENDER_AT_TARGET` export path (a scratch canvas at preset dimensions) and the copilot's
`_probe_frame`. Two shipped example passes are pixel-dependent, the JFA and the radiance
cascade, whose iteration counts follow pixel counts; the SDF text library is uv-space. No
`gl_FragCoord`, `textureSize` or derivative use exists in shipped resources.

Feedback is the sharp edge: `_seed_feedback` at load and the runtime feedback canvas both match
sizes strictly and, on a mismatch, drop the history and start black; `Canvas.set_size` is
release-and-reallocate. Exports already reset feedback before rendering, so an export at a
different size is a cold start today and stays one. A scaled feedback pass inside an export
still sizes off the LIVE `canvas_size`, an interaction nobody has tested.

The default render preset is `SCALE_DISTORT`, which exports straight from the live canvas
(`document.py`, the `preset is None or preset.fit is FitPolicy.SCALE_DISTORT` branch). Under
Auto the live size is whatever the panel happens to be, so the default export's resolution
would be accidental.

**Cadence** (`research/cadence_flow.md`). One UI frame renders the whole render set; there is
no per-document fps. `u_time` is wall-clock via `Document.live_time`, `begin_frame` is
idempotent by frame number and tolerates being called less often, so a throttle needs no time
plumbing: feedback and iterations simply step at the document's own render rate. Export renders
every frame at its own fps with `u_time = i / fps` and the copilot's probe renders one frame on
demand; neither reads the live cadence or the profiler, so both are unaffected. The script tick
runs once per UI frame with the UI's `dt` and assumes one render per tick, which a throttle
breaks. The natural cost input is 088's GPU timer ring, which records only while the FPS panel
is open (`ui.py`, `app.profiler.enabled = app.fps_details_open`, 088 D4); the always-on cost of
the queries was never measured.

## Design decisions

Locked by the maintainer on 2026-09-11 unless marked *proposed*.

D1 **Resolution mode is a per-document persisted field, `Auto | Fixed(w, h)`**, replacing the
bare `canvas_size` as the user-facing setting; `canvas_size` becomes the effective live size the
mode resolves to. Fixed keeps today's behavior exactly. Default for a new document: Auto.

D2 **Under Auto, the live size is the largest region currently displaying the document**, at
the display's pixel size: the viewer when the document is current, a grid tile or pass-strip
thumbnail otherwise. A document displayed nowhere keeps its last size. The rule is one function
over the frame's layout, computed before the render set and before any draw.

D3 *(proposed)* **Auto resizes are damped.** A new size is applied only when it differs from the
current one by more than a threshold (say 5 % in either dimension) or after the size has been
stable for N frames, so a window drag does not reallocate canvases every frame. The numbers are
the implementer's, pinned by a test that drags a size through a ramp and counts reallocations.

D4 **Feedback under Auto survives a resize by resampling.** On a size change a canvas of the new
size is allocated, the old feedback frame is blitted into it with linear filtering, the old
canvas is released, and the document keeps rendering into the new one from the next frame. The
release goes through the same path `Canvas.set_size` uses today. Ordering makes it safe: the
resize runs before the frame's draw phase, so no imgui draw list references the released texture
(the 084 D5 hazard). A test drags the size through a ramp and asserts the count of live GL
textures is constant. Fixed is unaffected. The persisted `feedback/<pass>.bin` seeds through the
same resample when its stored size differs from the live one.

D5 **The resolution picker stays as it is, list and all.** In Fixed mode it sets the live size,
as today. Under Auto it sets the EXPORT size only, stored per document beside the mode; the
default is the largest list entry with the document's aspect (1920×1080 for 16:9). Export never
reads the live size: `SCALE_DISTORT` takes its dimensions from the preset, falling back to the
Fixed size or the Auto export size. The scaled-feedback-pass-inside-export interaction gets a
test.

D6 **One GPU budget, shared automatically.** A single constant, the fraction of the UI frame all
documents together may occupy (proposed `0.5`, 088's knee). The current document draws on it
first: its interval is `k = ceil(cost / (budget × frame_period))`, `k = 1` when it fits. Every
other displayed document shares what remains at one common fps, the highest at which the sum of
their costs fits the remainder; from that fps each gets its own `k`. One preview gets most of
the remainder, twenty get a little each, and no preview constant exists. The whole rule is a
pure function `plan_render_set(costs, current, displayed, budget, frame_period) -> {doc: k}`,
tested without GL, the shape 088's `profile_rows_plan` set. The intervals are ephemeral, never
persisted. The measurement's optimism (a document-carrying frame is longer than the nominal
period) is accepted: it errs toward rendering more often, never less.

D7 *(proposed)* **The cost input is a per-document cost record with GPU and CPU fields**, the
GPU field from 088's timer span, recorded always and read two frames late as today; the policy
reads only the GPU field now, so a CPU throttle later is a policy change, not plumbing. The
always-on cost of the queries is measured before this lands (a probe over the app with the panel
closed vs open, the number written into the spec); if it is not negligible, the fallback input
is the CPU-side swap wait attributed to the document, which the baseline showed carries the GPU
cost 1:1 in a single-context loop.

D8 **The script ticks once per UI frame, as today.** A throttled document's script keeps
running at the UI rate; its uniform values reach the render on the document's next frame. GPU
cost is the bottleneck being addressed; CPU throttling is a later policy over D7's CPU field.

D9 *(proposed)* **The FPS panel shows the plan**: a document's row carries its effective fps and
`k`, and the budget coloring reads the share over wall time (cost × document fps), so a
converged throttled document reads green, not red.

D10 **Render all keeps its meaning** (every open document renders), but under Auto each renders
at its tile size and under D6 each is scheduled from the shared budget, so the sum that governs
the frame today shrinks on both axes. Under Fixed a preview must render at its full resolution,
and D6's remainder rule is what keeps it affordable: it renders rarely.

## Out of scope (with triggers)

- Tiling, rejected. Trigger: none.
- The GL thread (`00_research.md ## Recommendation` item 1). Trigger: the maintainer asks for
  input latency independent of the document after this refinement lands.
- Resolution-independence shims for pixel-dependent shaders. They use Fixed. Trigger: a
  user-facing document that must be Auto and pixel-exact.
- A CPU throttle. Trigger: a script tick measured above a few ms per frame; it plugs into D6
  through D7's CPU field.

## Files touched (estimate)

`document.py` (mode field, resolve, feedback resample, export size source), `ui_models.py`
(persisted mode + export size), `ui.py` (layout-driven size before the render set, the plan in
the render set), a new pure module for `plan_render_set` (leaf, no GL), `render_preset.py` /
`render_shape.py` (D5), `profiling.py` and the FPS panel (D7, D9), `tabs/` where the resolution
is edited (the mode control beside the unchanged picker), `widgets/pass_list.py` and
`document_grid.py` (report their displayed size), `help_content.py` (`u_resolution` under
Auto), `projects/dev` documents hand-fixed to the new field, tests for D2, D3, D4, D5, D6.

## Sizing

Large, with the full review cycle: the maintainer's call, to be revisited after the discussion.
Per the `dev_flow.md` preamble that means the upper end of the mid range or beyond: 2
pre-implementation reviewers, 3 post-implementation plus a spec-fidelity pass, and a
sanitization sweep, since persisted document state changes shape.

## Resolved questions

Asked and answered 2026-09-11, kept so the spec does not re-open them:

1. Viewer and grid tile at once: the largest region, which is the viewer (D2).
2. Feedback on an Auto resize: resample, keep rendering at the new size, release the old canvas
   without a leak, pinned by a texture-count test (D4).
3. Export size under Auto: the existing picker, unchanged, sets it; default the largest list
   entry at the document's aspect (D5).
4. Preview budget: not a constant; one shared budget split automatically by what is displayed
   (D6).
5. Script tick: per UI frame as today; the cost record carries a CPU field for a later throttle
   (D7, D8).
6. Size: large, full review cycle (Sizing).

Still open, small: D3's damping numbers, D6's budget constant, D9's panel shape. The
implementer proposes them in the spec.
