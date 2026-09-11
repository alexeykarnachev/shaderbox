# Cost and throttle — two cheaper levers than tiling

Research for feature 090 (render decoupling), after the maintainer rejected tiling
(`00_research.md ## Maintainer decision: no tiling`). Two replacement levers were proposed:
render the live document at the size the viewer actually displays it, instead of a fixed
buffer size; and throttle a heavy document to a lower fps than the UI. This report measures
what each buys, alone and together, with new probes under `../probes/`, prefixed `cost_` and
`throttle_`.

## Setup

Same machine as the rest of 090's research: X11 `:1`, NVIDIA RTX 3090, driver 580.173.02,
OpenGL 4.6, glfw 3.4 (pyGLFW 2.10), moderngl 5.12.0, PyOpenGL 3.1.10, Python 3.12.

**The maintainer's own ShaderBox was running throughout** (`ps aux` showed `uv run python
./shaderbox/ui.py`, PID 45546, using the GPU at a steady baseline) — it was left untouched.
Every probe therefore uses `probes/prio_guard.py`'s baseline-relative guard (samples GPU
utilization, takes the median as the session floor, refuses to measure only if utilization
climbs materially above that floor), not `probes/gpu_clear.py`'s fixed ≤20% threshold, which
could never pass with ShaderBox live. Each run prints `[guard] GPU baseline NN% (ceiling
NN%)` — the baseline measured 31–33% across the three runs, consistent with one live
ShaderBox instance and nothing else contending. `calibrate_heavy.py` had separately confirmed
(per `gpu_preemption.md`) that the calibrated cost is stable under a live background load, so
the baseline does not distort the quantities measured here.

The calibrated heavy shader is `common.HEAVY_FRAG` at `HEAVY_ITERS = 35000`
(100.93 ms at 1280x720, pinned by `calibrate_heavy.py`). The light shader is the same
hash/sin/cos accumulation loop at 3200 iterations (`cost_resolution.py`'s `LIGHT_ITERS`),
which measured 11.4 ms at 1280x720 — close to the requested "~10 ms" and structurally
identical to the heavy shader, so the resolution scaling in §1 is comparable ALU-bound work,
not a different kind of shader. The 30 ms document in §2 uses a third iteration count found by
binary search against the live GPU (`throttle_single.py::find_iters_for_cost`), which measured
9618 iterations at 30.94 ms — printed in that probe's run header rather than assumed.

## Method

**§1 resolution cost** (`probes/cost_resolution.py`): one hidden context, one FBO reused across
sizes, `ctx.finish()` timing (this isolates GPU cost the same way `calibrate_heavy.py` does —
no UI thread involved, no pacing). 7 sizes: 3840x2160, 1920x1080, 1280x720, 960x540, 640x360,
320x180, and 764x430 (the viewer's derived size — see below). 9 samples per size after 3
warmup renders, both HEAVY and LIGHT shaders, 2 full trials.

**The viewer's displayed size** was not measured live (running ShaderBox's own window would
have meant touching the maintainer's instance) but derived from the code:
`shaderbox/ui.py::_draw_document_image` aspect-fits the document into
`avail = imgui.get_content_region_avail()` bounded by `max_image_height = avail.y -
control_panel_min_height - 10` and `max_image_width = avail.x`, taking `min()` on each axis
against the document's own aspect ratio (line 673–681). At a 1920x1080 window: the window
opens `MAXIMIZED` at the monitor's native resolution (`app.py`, `glfw.create_window` sized
from `glfw.get_video_mode`), the left/right split defaults to 50/50
(`ui_models.py::editor_split_fraction`, default `0.5`), so the app (viewer) panel gets roughly
half the window width minus the splitter and editor-min-width guards (`_SPLITTER_W = 6.0`,
`_APP_PANEL_MIN_W = 360.0`) — call it ≈940 px avail width after the menu bar and panel
padding. `control_panel_min_height = SIZE.PANEL_CTRL_MINH = 600`; avail height after the menu
bar is roughly 1040, so `max_image_height ≈ 1040 - 600 - 10 ≈ 430`. For a 16:9 document
(1280x720) `image_width = min(940, 430 × 16/9 ≈ 764)` → height-constrained at **≈764×430**.
This is a derivation from the layout constants, not a pixel-measured screenshot (doing that
live would have required either opening a second window at the maintainer's monitor
resolution while his was running, or attaching to his live window — both avoided); state the
formula if a different window size or split ratio needs re-deriving:
`viewer_h = min(window_h − menu_bar − 10 − 600, ...)`, `viewer_w = min(avail_w, viewer_h ×
aspect)`. The general shape holds regardless of the exact pixels: the viewer is well under the
document's native canvas size whenever the control panel and editor take real screen space,
which they do by construction (`PANEL_CTRL_MINH = 600`, `_APP_PANEL_MIN_W = 360`).

**§2 throttle hitch pattern** (`probes/throttle_single.py`): today's loop shape — one process,
one context, heavy-then-light in one loop (`probe_d_single.py`'s shape) — paced at 60 fps
(`probe_e_paced.py`'s discipline: each UI frame sleeps out the remainder of the 16.7 ms budget,
so what's reported is a deadline miss, not throughput). The document renders only on every
k-th UI frame; other frames present the last-rendered document texture untouched, exactly as a
throttled document would in the app (no new GL work, no sampling change). Per configuration,
300 UI frames after a 5-frame warmup discard: UI period median/p95/max, frames missing the
16.7 ms budget, the document's achieved fps (`doc_renders / wall_time`), and the fraction of
wall-clock time the GPU spent on the document (`sum(doc_frame_costs) / wall_time`). k ∈ {1, 2,
3, 6, 12} for the ~100 ms document (HEAVY_ITERS), k ∈ {1, 2, 3} for the ~30 ms document. 2
trials, both printed.

**Share-of-wall-time policy**: for a document costing `cost` ms, choose the smallest integer k
with `cost / (k × 16.7 ms) ≤ 0.5` — the document gets at most half of wall-clock GPU time.
Computed analytically for costs 10, 30, 60, 100 ms (below), and measured directly for 100 ms
(the k=12 row of `throttle_single.py`, since `100.93/16.7/0.5 ≈ 12.1` rounds to k=12) and for
the viewer-sized ~40 ms document in §3 (k=5, `throttle_combined.py` computed it live from a
measured cost, not the nominal 100 ms figure).

**§3 combined** (`probes/throttle_combined.py`): the same paced single-thread/single-context
loop, but the document renders into a 764×430 target (the derived viewer size) instead of
1280×720, and k is chosen by the share policy from the document's *measured cost at that
size* (40.60 ms measured live at the top of the run, not assumed from §1's table — the two
agree to within 0.3 ms). A 1280×720/k=1 reference config runs in the same script, in the same
process, back to back with the combined config, for a same-run comparison point.

All probes print a `[run header]` line with the GPU baseline, shader parameters, and probe
constants, per the maintainer's instruction.

## §1 — GPU cost vs. resolution

Median of two trials (both reproduced within ~1%); `ms/mpix` = cost ÷ megapixels.

**HEAVY shader (35000 iters, 100.93 ms calibrated at 1280x720):**

| Size | Megapixels | Cost (ms) | ms/mpix |
|---|---:|---:|---:|
| 3840x2160 | 8.294 | 884.49 | 106.64 |
| 1920x1080 | 2.074 | 224.54 | 108.28 |
| 1280x720 | 0.922 | 102.62 | 111.35 |
| 960x540 | 0.518 | 59.90 | 115.54 |
| 764x430 (derived viewer @ 1920x1080 window) | 0.329 | 40.30 | 122.66 |
| 640x360 | 0.230 | 28.99 | 125.81 |
| 320x180 | 0.058 | 9.06 | 157.30 |

**LIGHT shader (3200 iters, ~11.4 ms at 1280x720):**

| Size | Megapixels | Cost (ms) | ms/mpix |
|---|---:|---:|---:|
| 3840x2160 | 8.294 | 83.34 | 10.05 |
| 1920x1080 | 2.074 | 22.37 | 10.79 |
| 1280x720 | 0.922 | 11.42 | 12.39 |
| 960x540 | 0.518 | 6.03 | 11.64 |
| 764x430 | 0.329 | 3.87 | 11.78 |
| 640x360 | 0.230 | 2.82 | 12.24 |
| 320x180 | 0.058 | 0.67 | 11.58 |

**Cost is close to linear in pixel count from 4K down to ~960x540**, for both shaders: the
cost ratio between adjacent sizes tracks the pixel-count ratio within ~5% (e.g. 4K→1080p:
3.94x cost for 4.0x pixels; 1080p→540p: 3.75x cost for 4.0x pixels; 1280x720→764x430: 2.55x
cost for 2.81x pixels). **Below ~960x540 the HEAVY shader goes supralinear in the other
direction — cost per pixel rises** (960x540→320x180: 6.6x cost drop for a 9x pixel drop,
ms/mpix climbing from 115 to 157): a fixed per-draw floor (driver submission, fixed-function
raster setup, `ctx.finish()` round trip) stops shrinking with the framebuffer, so it dominates
an ever-larger share of an ever-smaller total. The LIGHT shader's ms/mpix stays flat across the
whole range (10.0–12.4) because its total cost is already small enough that this floor is a
bigger fraction of it everywhere, not just at the small end.

## §2 — Throttle hitch pattern (today's loop, paced at 60 fps)

**~100 ms document** (HEAVY_ITERS=35000, measured 101–104 ms per frame in this run):

| k | UI period median (ms) | p95 | max | Missed /300 | Doc fps | GPU share of wall |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 103.8 | 105.5 | 106.7 | 300/300 | 9.6 | 99.9% |
| 2 | 51.9 | 103.2 | 104.3 | 150/300 | 8.4 | 86.0% |
| 3 | 0.3 | 102.8 | 103.8 | 100/300 | 7.4 | 75.4% |
| 6 | 0.06 | 102.1 | 102.8 | 50/300 | 5.4 | 55.0% |
| 12 | 0.06 | 101.1 | 103.0 | 25/300 | 3.6 | 36.1% |

**~30 ms document** (measured 9618 iters = 30.94 ms):

| k | UI period median (ms) | p95 | max | Missed /300 | Doc fps | GPU share of wall |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30.4 | 31.6 | 35.1 | 300/300 | 33.2 | 99.7% |
| 2 | 15.1 | 29.6 | 30.5 | 150/300 | 22.0 | 63.3% |
| 3 | 0.3 | 29.0 | 29.5 | 100/300 | 16.2 | 46.0% |

Two things carry across both tables. **The p95 barely moves with k** — it stays pinned near
the document's own cost (102–103 ms for the heavy document at every k, 29–32 ms for the light
one) because whichever frame *does* carry the document render still blocks for the document's
full cost; k only changes how many of the 300 frames are that frame. This is the direct
consequence of `00_research.md`'s headline: the driver does not preempt a draw call, so a
throttled document does not make any *individual* document-carrying frame faster — it makes
document-carrying frames rarer. **Median falls roughly as 1/k** once k is large enough that a
majority of frames carry no document work at all (k≥3 for the heavy document, where median
drops to near 0: 0.3→0.06 ms), because the median then reports the trivial frames. Missed
frames also scale almost exactly as `300/k` in both tables (e.g. heavy k=6 → 50/300 = 300/6),
confirming k directly controls what *fraction* of frames hitch, not how bad each hitch is.

**Share-of-wall-time policy**, costs 10/30/60/100 ms, `k = ceil(cost / 8.35 ms)`:

| Cost | k | Predicted share (cost / (k×16.7ms)) | Doc fps at steady 60 fps UI |
|---:|---:|---:|---:|
| 10 ms | 2 | 29.9% | 30.0 |
| 30 ms | 4 | 44.9% | 15.0 |
| 60 ms | 8 | 44.9% | 7.5 |
| 100 ms | 12 | 49.9% | 5.0 |

The 100 ms row's predicted numbers (k=12, share≈49.9%, doc fps≈5.0) can be checked directly
against the measured k=12 row above: **measured share was 36.1%, not 49.9%, and measured doc
fps was 3.6, not 5.0.** The gap is real and has a mechanical cause: the policy's arithmetic
assumes a document-carrying frame still fits inside its k×16.7 ms slot and the UI free-runs at
60 fps around it, but the document-carrying frame itself takes ~103 ms of *wall* time (it
blocks, per §00's headline), so the achieved period between document renders is `k × (mix of
16.7 ms idle frames and one 103 ms document frame) / k`, not a clean `k × 16.7 ms`. Concretely
at k=12: 11 idle frames at ~16.7 ms plus 1 document frame at ~103 ms totals ~286 ms of wall
time per document render, not the naive 200.4 ms (12×16.7) — hence fps 1000/286≈3.5 (matches
the measured 3.6) rather than the naive 5.0, and share 103/286≈36% (matches the measured
36.1%) rather than the naive 49.9%. **The policy's target share is a floor on k, not an
achieved number** — the real share is always somewhat lower than the target because the
document's own render time inflates the wall-clock denominator beyond `k × budget`. The 30 ms
document's k=3 row shows the same pattern less severely (predicted share for k=3 at 30ms is
30/(3×16.7)=59.9%; if k=4 had been run it would read ≈45% nominal vs. something lower measured,
consistent with the same mechanism at smaller scale).

## §3 — Combined: viewer size + share-policy throttle

Document at the derived viewer size (764×430) instead of 1280×720, k chosen by the share
policy from the size's own measured cost (40.60 ms, matching §1's 40.30 ms table entry within
noise):

| Config | Size | k | UI median | p95 | max | Missed /300 | Doc fps | GPU share |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Combined** | 764x430 | 5 | 0.07 | 39.2 | 39.9 | 60/300 | 9.5 | 36.6% |
| Baseline (today) | 1280x720 | 1 | 105.3 | 107.4 | 110.6 | 300/300 | 9.5 | 99.9% |

Both rows reproduced across 2 trials (combined: 0.07/0.06 ms median, 60/60 missed both trials;
baseline: 105.14/105.47 ms median, 300/300 missed both trials) in the same script, same
process, back to back.

## Interpretation

**Resolution alone** (§1, no throttle) turns the 100 ms document into a 40 ms one at the
derived viewer size — a real ~2.5x cut, close to the pixel-count ratio (2.8x), because cost is
near-linear in this regime. But every document-carrying frame *still* blocks the single loop
for its full cost (40 ms, not the 16.7 ms budget), so resolution alone still misses every frame
at k=1 (this wasn't run standalone, but follows directly from §2's mechanism: any
document-carrying frame in today's loop takes as long as the document costs, full stop).
Resolution changes *what* the blocking frame costs; it does not by itself make frames stop
blocking.

**Throttle alone** (§2) does not shrink any individual hitch — the p95 for the heavy document
stays at 101–106 ms regardless of k, because the document-carrying frame is unchanged. What
throttle buys is *frequency*: at k=12 only 25 of 300 frames (8.3%) carry the hitch instead of
all 300, and the median UI period collapses to sub-millisecond because most frames are free.
In the maintainer's terms: the editor still freezes for ~100 ms when a hitch lands, but a hitch
lands roughly once every 12 UI frames (about every 300 ms at a nominal 40 fps effective rate)
instead of every frame. The document's own fps falls proportionally (9.6 → 3.6 at k=12) — the
"no budget/no quality cap" constraint from `00_research.md` is not violated (every document
frame still completes in full, unshortened), but the document is refreshed less often, which
is a real, visible cost for something the maintainer is actively iterating on.

**Both levers together** (§3) is the best of the measured configurations for hitch frequency
without sacrificing render completeness: the document renders at its full 764×430 viewer
resolution (nothing visually lost — that is already all the pixels the viewer shows), costs
40.6 ms per frame instead of 105 ms, and at the share policy's k=5 only 60 of 300 frames
(20%) carry that 40 ms hitch — one-fifth as often as baseline, and each hitch is 2.5x
shorter than baseline's 105 ms hitch. Document fps holds at 9.5, statistically the same as
baseline's 9.5 — because k=5 was chosen from the *smaller* cost, throttling less aggressively
than the heavy-document k=12 case, so resolution's savings are spent on hitch-shortening
rather than on top of an already-low fps. This is the version of "the terminal's behavior"
that these two levers can reach without a thread: not a 60 fps UI beside a heavy document
(that needs the thread — nothing here changes that a document-carrying frame still blocks the
one loop for its own duration), but a UI that hitches for 40 ms once in five frames instead of
105 ms every frame.

## False trails

**The share-of-wall-time policy's own arithmetic overstates achieved share and underachieves
doc fps** (§2 above) — the formula `cost / (k × 16.7ms)` implicitly assumes the document
render fits inside its slot without inflating the total period, which is false whenever cost
exceeds 16.7 ms (always, for anything worth throttling): the document-carrying frame's wall
time is the document's own cost, not the budget, so it stretches the true period past `k ×
16.7ms`. Anyone implementing the share policy should compute k from a measured or predicted
achieved-share curve, not trust the nominal formula's k to land near the target share.

**Assuming resolution scaling is linear everywhere** — it is not, below ~960x540 for the heavy
shader (§1): ms/mpix climbs from 111 at 1280x720 to 157 at 320x180. A document already small
enough to render fast gets proportionally less benefit from shrinking further, because a fixed
per-draw floor stops shrinking with it. This does not change the viewer-size number in this
report (764x430 sits well inside the near-linear range), but it would matter for a much smaller
UI panel (a pass-strip thumbnail at `SIZE.PASS_TILE = 168` or `PASS_THUMB = 112`, both far
below 320x180 in area) — those are not sized by the viewer path (`_draw_document_image`) and
are out of this report's scope, but the same shader rendered at 112x112 would pay a materially
worse ms/mpix than the numbers above suggest by extrapolation.

**Assuming the viewer's derived size is exact** — it is a derivation from `_draw_document_image`
and the layout constants (`editor_split_fraction=0.5`, `PANEL_CTRL_MINH=600`,
`_APP_PANEL_MIN_W=360`), not a pixel-measured screenshot; the maintainer's own ShaderBox was
running throughout this session and was left untouched rather than used as a measurement
target. The formula is stated in Method so a different window size or split ratio can be
re-derived; the 764x430 figure should be treated as "the right order of magnitude for a
1920x1080 window at the default split," not a pinned constant.

**Assuming k should be computed once from the document's cost at 1280x720** — §3 shows the
share-policy k differs materially depending on which size's cost it is computed from (k=12 at
1280x720's 101 ms vs. k=5 at 764x430's 41 ms for the *same* underlying document). Computing k
from the wrong size either over-throttles (if computed from a larger buffer than actually
rendered) or under-throttles (the reverse) — the two levers are not independent knobs; the
throttle decision must be made after the resolution decision, from the resolution actually
used.

## Verdict — UI frame time recovered for a 100 ms document

- **Resolution alone**: today's loop still blocks once per document frame, so at k=1 it
  recovers none of the *missed-frame count* (still 300/300 missed, by the mechanism above) —
  but it cuts every hitch's *length* from ~105 ms to ~40 ms at the viewer's derived size, a
  63 ms (60%) reduction in how long each blocked frame lasts. It buys duration, not frequency.
- **Throttle alone**: at the share policy's k=12 for the unscaled 100 ms document, 275 of 300
  frames (91.7%) are no longer blocked at all — each of those recovers the full ~101 ms it
  would have cost at k=1, at the price of document fps falling from 9.6 to 3.6. It buys
  frequency, not duration: the 25 frames that still carry the document still cost ~101 ms.
- **Both together**: at the combined configuration's k=5 and 764×430, 240 of 300 frames
  (80%) are unblocked (vs. 0/300 unblocked at today's 1280x720/k=1), and the 60 frames that do
  carry the document cost ~40 ms instead of ~105 ms — a 62% shorter hitch on top of an 80%
  cut in how often it happens, while document fps holds at 9.5 (statistically flat versus
  today's 9.5, because the smaller buffer let k stay at 5 rather than needing 12). This is the
  strongest configuration measured without a render thread: neither lever alone reaches it,
  and the two compose because resolution's savings let the throttle policy choose a gentler k
  for the same wall-time-share target.
