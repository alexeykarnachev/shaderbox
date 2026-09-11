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

## Design decisions (proposed, for lock)

D1 **Resolution mode is a per-document persisted field, `Auto | Fixed(w, h)`**, replacing the
bare `canvas_size` as the user-facing setting; `canvas_size` becomes the effective live size the
mode resolves to. Fixed keeps today's behavior exactly. Default for a new document: Auto.

D2 **Under Auto, the live size is the largest region currently displaying the document**, at
the display's pixel size: the viewer when the document is current, a grid tile or pass-strip
thumbnail otherwise. A document displayed nowhere keeps its last size. The rule is one function
over the frame's layout, computed on the main thread before the render set.

D3 **Auto resizes are damped.** A new size is applied only when it differs from the current one
by more than a threshold (say 5 % in either dimension) or after the size has been stable for N
frames, so a window drag does not reallocate canvases every frame. The exact numbers are the
implementer's, pinned by a test that drags a size through a ramp and counts reallocations.

D4 **Feedback under Auto survives a resize by resampling, not by restarting.** On a size
change the feedback canvas is drawn into a canvas of the new size with a linear blit before the
old one is released. Restarting black on every panel resize would make Auto unusable for any
feedback document. Fixed is unaffected. The persisted `feedback/<pass>.bin` seeds through the
same resample when its stored size differs from the live one.

D5 **Export never reads the live size.** `SCALE_DISTORT` takes its dimensions from the render
preset; a preset with no explicit size falls back to the document's Fixed size, or, under Auto,
to a per-document export size stored beside the mode (default 1920×1080 at the document's
aspect). The scaled-feedback-pass-inside-export interaction gets a test.

D6 **The throttle is a per-document, ephemeral render interval `k`, recomputed from the
measured document cost**: render when `frame_idx % k == 0`, with `k = ceil(cost / (share ×
frame_period))`, `share = 0.5` (088's knee), `k = 1` for any document under the budget. The
measurement's optimism is accepted: it errs toward rendering the document more often, never
less. No persisted setting; a user who wants a fixed document fps has Fixed resolution and can
lower the app's target fps.

D7 **The cost input is 088's GPU span, recorded always, read two frames late as today.** The
always-on cost is measured before this lands (a probe over the app with the panel closed vs
open, the number written into the spec); if it is not negligible, the fallback input is the
CPU-side swap wait attributed to the document, which the baseline showed carries the GPU cost
1:1 in a single-context loop.

D8 **The script tick runs once per document render, not once per UI frame**, with `dt` equal
to the wall time since the document's previous render. Scripts already receive `context.t` on
wall time; a script that integrates by `dt` keeps its meaning under a throttle.

D9 **The FPS panel shows the throttle**: a document's row carries its effective fps and `k`,
and the budget-share coloring reads the share over wall time (cost × document fps), so a
converged throttled document reads green, not red.

D10 **Render all keeps its meaning** (every open document renders), but under Auto each renders
at its tile size and under D6 each is throttled on its own cost, so the sum that governs the
frame today shrinks on both axes.

D11 **A document that is not current is throttled harder, in both modes.** The throttle's
share is `0.5` for the current document and a `PREVIEW_SHARE` (proposed `0.05`) for every other
displayed document, so `k` follows the same formula with the smaller budget. This is what makes
Fixed affordable: a Fixed document shown only as a thumbnail must render at its full
resolution (its shader needs the size), and rendering it rarely is the only lever left. With
`0.05` at 60 fps: a 1 ms document previews at 30 fps, a 5 ms one at 10 fps, a 100 ms one every
2 s. Under Auto the tile is already cheap and the smaller share is a bonus. A flat preview fps
was considered and rejected: at 10 fps it would waste GPU on cheap documents and still spend
100 % of the GPU on a 100 ms one.

## Out of scope (with triggers)

- Tiling, rejected. Trigger: none.
- The GL thread (`00_research.md ## Recommendation` item 1). Trigger: the maintainer asks for
  input latency independent of the document after this refinement lands.
- Resolution-independence shims for pixel-dependent shaders. They use Fixed. Trigger: a
  user-facing document that must be Auto and pixel-exact.

## Files touched (estimate)

`document.py` (mode field, resolve, feedback resample, export size source), `ui_models.py`
(persisted mode + export size), `ui.py` (layout-driven size before the render set, the
throttle in the render set, the script tick per document render), `render_preset.py` /
`render_shape.py` (D5), `profiling.py` and the FPS panel (D7, D9), `tabs/` where the
resolution is edited (the mode control), `widgets/pass_list.py` and `document_grid.py` (report
their displayed size), `help_content.py` (`u_resolution` under Auto), `projects/dev` documents
hand-fixed to the new field, tests for D2, D3, D4, D5, D6, D8.

## Open questions for the maintainer

1. D2's rule when the document is both current and in the grid: the largest region, or the
   viewer always?
2. D4: resample feedback on resize, or accept a restart? Resampling is the robust default and
   costs one blit per resize.
3. D5: is 1920×1080 at the document's aspect the right default export size under Auto?
4. D6 and D11: `share = 0.5` current / `0.05` preview, or expose them as app settings?
5. D8 changes what a script sees; confirm that a throttled document's script ticking at the
   document's rate is what you want.
6. Sizing: with the thread out of scope this is a mid feature touching the document model, the
   render set, exports and the profiler. Proposed: 2 pre-implementation reviewers, 3
   post-implementation, plus a spec-fidelity pass, since it changes persisted document state.
