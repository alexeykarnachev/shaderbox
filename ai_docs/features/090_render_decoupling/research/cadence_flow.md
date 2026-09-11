# 090 — Cadence flow: what "how often a document renders" touches

Research for a document-level throttle: when a document's measured GPU cost exceeds a budget
share of the UI frame, render it less often (lower document fps) while its clock keeps
advancing on wall time. This file maps every mechanism that cares how often a document renders,
so `01_spec.md` can state precisely what changes meaning and what must stay pinned. No tiling —
the maintainer rejected it (`00_research.md` "Maintainer decision: no tiling"); this is a
narrower question than 090's thread/tiling design and can land independently of it.

---

## 1. The cadence model today

**One document render = one UI frame, always, for everything in the render set.** There is no
per-document fps. `shaderbox/ui.py::run` (`ui.py:135-153`) drives the whole app at one rate:
`target_fps = app.app_state.global_target_fps` (`ui.py:142`), sleeps out the remainder
(`ui.py:143`), and that is the only frame-rate knob that exists. Every document in the render
set draws exactly once (or twice — see below) per iteration of this loop.

**Who decides a document renders this frame.** `ui.py::_tick_frame_state` (`ui.py:156-266`)
builds `tick_documents`, "the frame's render set (066 D2)" (`ui.py:232`): the current document
always; with "Render all" on (`app.app_state.is_render_all_documents`, default `True`,
`ui_models.py:229`) every document whose `first_render_done` is already `True`
(`ui.py:241-247`); plus **at most one** not-yet-rendered document admitted this frame
(`ui.py:248-258`, the 066 D1/D2 compile-cost throttle — unrelated to a GPU-cost throttle, it
bounds *compile* cost on frame 0, not steady-state render cost). This list is the render set for
every mechanism below: the script tick, `begin_frame`, and the render calls in
`_update_and_draw` (`ui.py:315-363`) all iterate exactly this set.

**The frame number.** `app.frame_idx` (`app.py:1237`, init 0) is a single **UI-loop** counter,
incremented once per `update_and_draw` at `ui.py:539` — not per document, not per document
render. It is passed as `frame` to `session.tick(tick_documents, now, dt, app.frame_idx, ...)`
(`ui.py:260`) and as the frame number to every ticked document's `document.begin_frame(app.frame_idx)`
(`ui.py:265`). So today "frame N" means the same thing for every document simultaneously — the
Nth iteration of the UI loop, whether or not a given document actually drew in it (a document
outside the render set, e.g. behind a modal, simply does not get ticked or `begin_frame`d that
UI frame, and its `_frame` counter — see below — stays where it was).

**`u_time` derivation: wall clock, not frame count.** There is no `u_frame` uniform anywhere in
the engine-driven set — `shaderbox/engine_uniforms.py` lists exactly `u_time`, `u_aspect`,
`u_resolution`, `u_pass_iteration`, `u_pass_iterations` (`engine_uniforms.py:14-20`); no frame
counter is ever written into a shader. `u_time` comes from `Document.live_time()`
(`document.py:417-423`): `process_time() - self.time_origin`, i.e. **wall-clock seconds since
the document was opened or last Reset**, entirely independent of how many times the document has
rendered. `Document.render` resolves it at `document.py:669-670`: `u_time = self.live_time()`
when the caller passes no explicit `u_time`. `Pass.render` (`core.py:472-474`) falls back to
`process_time()` only when a bare `Pass` is drawn with no `u_time` at all (outside a `Document`);
inside a document the resolved value always flows down. **Consequence: `u_time` already
advances on wall time regardless of render cadence** — a document that renders less often will
see `u_time` jump between renders rather than crawl one frame-interval at a time. That is
exactly the throttle's stated intent ("the document's clock still advancing on wall time"), and
it requires no new plumbing — it is what `live_time()` already does. The thing that would newly
depend on cadence is anything that assumes one draw = one fixed `dt` of wall time, which today
is only true because renders happen every UI frame.

**Feedback and iteration stepping.**
- *Frame-to-frame feedback* (a pass reading its own previous frame) swaps at
  `Document.begin_frame(frame)` (`document.py:363-393`), "at most once per frame" and gated on
  `render_pass.drawn_frame` matching the *previous* `self._frame` (`document.py:387-390`): a pass
  that did not draw last frame keeps its history still, rather than alternating every frame it
  didn't draw. `begin_frame` is idempotent on the frame number it's given (`document.py:377-378`:
  `if frame is not None and frame == self._frame: return`) — a second call for the same frame
  number is a no-op, which is exactly the mechanism `conventions.md` calls out (the pass strip
  §"A document is N passes… feedback swaps at the FRAME boundary", quoted in §6).
- *Within-frame iteration* (a pass run N times in one turn, `entry.iterations`) swaps between
  iterations via `self._swap_feedback(name)` inside `Document.render`'s iteration loop
  (`document.py:734-739`), unconditionally on every non-last iteration — this is unrelated to
  the UI frame cadence; it happens fully inside one `render()` call regardless of how often that
  call itself is invoked.
- Both mechanisms are **keyed by the document's own `_frame` counter** (`self._frame`,
  `document.py:302`, advanced inside `begin_frame`), which is a per-document field, not
  `app.frame_idx`. Today the two happen to move in lockstep only because `begin_frame(app.frame_idx)`
  is called every UI frame for every document in the render set (`ui.py:265`) — nothing in
  `Document` itself assumes `self._frame == app.frame_idx`.

**Script tick vs. render.** `_tick_frame_state` runs `app.session.tick(tick_documents, now, dt,
app.frame_idx, mouse=...)` (`ui.py:260`) for the *entire render set*, once, **before** any
document actually renders (`ui.py:315` onward runs after). `ProjectSession.tick`
(`project_session.py:686-711`) computes one `ScriptContext(t=document.live_time(t), dt=dt,
frame=frame, mouse=mouse)` per document and calls `ScriptEngine.tick` — `t`, `dt` and `frame` are
the **same UI-loop values for every document in the set** (`t` differs only via each document's
own `live_time` offset). The script's `update(self, ctx)` therefore runs exactly once per UI
frame per ticked document, today, and a script author writing `self.count += 1` inside `update`
gets one increment per UI frame — this is the load-bearing assumption a throttle would break if
the script tick stayed on the UI cadence while renders thinned out (`document.py:251`'s
"the live path ticks once via `session.tick()` in `ui.py`, so firing it in `render()` would
double-tick the frame" states the current single-tick-per-UI-frame invariant explicitly).
`Document.render` itself never ticks scripts (`self.on_pre_render` fires only from the export
loops, `document.py:295-298,907`).

**"Render all" (`app.app_state.is_render_all_documents`).** A single boolean
(`ui_models.py:229`, default `True`), read once in `_tick_frame_state` (`ui.py:241`). Off, the
render set is the current document alone (plus at most one pending first-render); on, every
open document with `first_render_done` joins the set every frame. It has no per-document
granularity and no relationship to cost — it is purely "how many documents render," not "how
often." A throttle sits at right angles to it: "Render all" decides *which* documents are in
today's set; a throttle would decide, for a document already in that set, whether *this
particular* UI frame is one of the frames it actually draws on.

**Two renders per document per frame, structurally.** For any ticked document,
`_update_and_draw` may call `document.render()` **twice**: once for the output chain
(`ui.py:326`) and, if some pass is still off-chain, once more with `target=pending`
(`ui.py:335-341`, "One never-drawn pass per document per frame draws its own chain"). Both calls
happen inside the same `app.frame_idx`/`self._frame`, which is exactly why `begin_frame`'s
once-per-frame idempotency exists (`conventions.md`, quoted §6) — a naive "render every k-th
frame" throttle must gate *both* calls together, not treat them as two independent render
opportunities.

---

## 2. What changes meaning under a throttle

| item | today's assumption | under a throttle | what the spec must decide |
|---|---|---|---|
| **Render call frequency** | Every ticked document renders every UI frame (`ui.py:315-363`) | A throttled document renders on a subset of UI frames | The gating mechanism: a per-document counter/timer checked before `document.render()` is called, presumably still inside `_tick_frame_state`'s render-set logic or the render loop at `ui.py:315` |
| **`u_time`** | `live_time()` = wall seconds since open/reset (`document.py:417-423`), independent of render count | Unchanged in *formula* — still wall time — but now advances by a larger, irregular Δ between the document's own successive draws | Nothing to decide here structurally (it already does the right thing); the spec must simply confirm no caller assumes a fixed per-render `u_time` delta (none found — `u_time` is always resolved fresh per call) |
| **Script tick cadence** | One `update()` call per UI frame per ticked document (`ui.py:260`, `project_session.py:686-711`), `dt` = UI-frame Δt | Must decide: does the script still tick every UI frame (so `self.count`, integrators, etc. stay smooth) even when the document *doesn't render* that frame, or does it tick only on render frames (so `dt` becomes the throttled document's own frame period)? | **Decide**: ticking on every UI frame (decoupled from render) matches "the document's clock still advancing on wall time" and keeps a script's own integrators smooth; ticking only on render frames means a script sees a bigger `dt` per call, consistent with fewer draws but changes what "one tick" means for any script the maintainer already wrote assuming 1 tick = 1 render (there is no `u_frame`, but scripts DO receive `ctx.frame`, currently = `app.frame_idx`, which under either policy stops meaning "the Nth time this document rendered") |
| **`ctx.frame` (`ScriptContext.frame`, `scripting/context.py:54`)** | = `app.frame_idx`, the UI loop's counter, identical across every document (`project_session.py:706`) | If script tick decouples from render, `ctx.frame` for a throttled document either keeps counting UI frames (cheap, but "frame" no longer means "render") or needs its own per-document counter | **Decide**: is `ctx.frame` still `app.frame_idx`, or does each document need its own frame counter that only advances on its own renders? No such per-document counter exists today (only `Document._frame`, which is feedback-scoped, not script-scoped) |
| **`self._frame` / feedback stepping** | Advances once per UI frame per ticked document via `begin_frame(app.frame_idx)` (`ui.py:265`), which also gates the frame-to-frame feedback swap (`document.py:387-390`) | A throttled document's feedback history would swap only on the UI frames it actually renders — `begin_frame` must be called with the document's OWN advancing frame identity, not skipped-and-caught-up, or the "only a pass that drew last frame has new history" rule (`document.py:387-390`) breaks: skipping `begin_frame` entirely on non-render frames is the correct behavior (the swap gate already keys off whether the pass drew at the PREVIOUS begin_frame call, so calling `begin_frame` less often is exactly right) but the spec must confirm nothing else expects `self._frame` to track `app.frame_idx` 1:1 | **Decide/confirm**: `begin_frame` is simply not called on a throttled document's skipped UI frames (its `_frame` stays put); `Document._frame` becomes a genuinely document-local counter under a throttle rather than accidentally-equal to `app.frame_idx`. Acceptable — nothing reads `Document._frame` against `app.frame_idx` today |
| **Feedback stepping RATE** | A feedback pass advances its history once per UI frame, i.e. at `global_target_fps` | Advances once per the document's *own* render, i.e. fewer times per wall second | **Decide**: is a slower feedback trail (e.g. a trail-blur effect visibly chunkier) acceptable under load? The maintainer's "no budget/quality cap on a document is acceptable" constraint (`00_research.md` line 4) argues against ever *reducing* a document's own visual behavior — a throttle that changes what a feedback effect actually looks like at high load may collide with that constraint. This is the single biggest open question the spec must resolve explicitly, since it's the one place a "render less often" policy is not merely slower but qualitatively DIFFERENT (fewer feedback steps ≠ the same animation played back slower — it is a coarser integration) |
| **Iteration stepping (`entry.iterations`, within one `render()` call)** | Fully contained inside one `render()` call, `document.py:707-739`; unaffected by how often that call happens | Unaffected — an iterated pass still does all N iterations in whichever call actually happens | No decision needed; orthogonal to the throttle |
| **"Render all" render set membership** | Boolean gate on which documents are ticked/rendered at all (`ui.py:232-258`) | Orthogonal — a throttle acts WITHIN the render set, deciding not "is this document rendered" but "is this document rendered THIS frame" | **Decide**: does a throttled document that skips this UI frame still count as "in the render set" for `first_render_done`/pending-first-render bookkeeping, or does skipping it interact with the one-first-render-per-frame admission rule (`ui.py:248-258`)? Likely: a document already past `first_render_done` is the only kind a throttle would ever act on (a not-yet-rendered document must still get its first render to leave the loading state) |
| **The two-renders-per-frame case (output + one pending off-chain pass)** | Both calls happen or neither does, gated by the same `app.frame_idx` (`ui.py:315-341`) | The throttle gate must cover BOTH calls atomically — a document that "skips this frame" skips both its output render and its pending-pass sweep, or the pending-pass sweep silently becomes the document's only per-frame GPU cost measurement point and desyncs from the output's own cost | **Decide**: throttle gate wraps the whole `if ui_document is not None:` block at `ui.py:317-341`, not each `document.render()` call individually |
| **The pass strip / preview tiles (`widgets/pass_list.py`)** | Reads `render_pass.canvas.texture` live every UI frame it draws, showing whatever the last render wrote | Shows the last frame the document actually rendered, held longer (stale by however many UI frames were skipped) | **State, don't newly decide**: this already works correctly for a document that renders less often — a tile just displays an older texture, exactly as it does today when a document is behind a modal and not ticked. No code change implied, but the spec should say so explicitly since "does the pass strip show the last frame" was one of the open questions posed |
| **The FPS chip (`app.global_fps`, `ui.py:145-150`)** | The UI LOOP's own fps (EMA of the whole `run()` iteration period, unaffected by which documents render) | Unchanged — the chip measures the UI loop, and a throttle's entire point is to keep this number at target regardless of document cost | **Confirm**: no change. The chip is not document-scoped and already reports what a throttle is meant to protect |
| **The FPS *panel* ("frame"/"gpu" rows, `profile_rows_plan`)** | `budget_ms = 1e3 / target_fps` is the single UI-frame budget (`ui_primitives.py:1374`); `document:<name>` / `pass:<name>` spans show the cost of whichever documents rendered THIS frame (`ui.py:322-341`) | On a UI frame where a throttled document is skipped, it contributes NO span at all (today: no render call = no `profiler.cpu(f"document:{name}")` wrapper entered) — the panel would show that document's row only on the frames it actually drew, and its GPU-share coloring (§4/§6) would read as if the document cost nothing on skipped frames | **Decide**: does the panel need to show a throttled document's last-known cost on frames it didn't render (so the maintainer isn't confused by a row that flickers in and out), or is "the row is simply absent that frame" acceptable since the tree already reflects "what happened this frame" literally (088's stated design: "each document rendered this frame") |

---

## 3. Export and the copilot: unaffected, with citations

**Export renders every frame at its own fps, stepping time deterministically, entirely outside
the live loop's cadence.** `Document._render_video` (`document.py:837-925`):
- `n_frames = int(details.duration * details.fps)`, `dt = 1.0 / details.fps` (`document.py:902-903`)
  — the export's own fps, unrelated to `global_target_fps` or any live cadence.
- Each frame: `self.on_pre_render(i / details.fps, dt, i)` then `self.begin_frame()` (no frame
  number — `None` means "next frame" per that call's own sequence) then `self.render(i /
  details.fps, canvas=canvas)` (`document.py:906-909`) — `u_time` is **passed explicitly** as
  `i / details.fps`, never read from `live_time()`. This is deterministic by construction: frame
  `i`'s `u_time` is `i / fps` regardless of wall-clock speed, GPU cost, or how the live loop is
  behaving concurrently.
- `_render_image` similarly passes its own `t` (`document.py:809-820`).
- The profiler: both loops call `self.render(...)` with no `profiler=` argument, so it defaults
  to `NULL_PROFILER` (`document.py:652`) — export never touches the live loop's throttle-input
  measurements at all (confirmed again in `conventions.md`'s D3 quote, §6).
- Script ticking during export is injected per-frame via `self.on_pre_render`
  (`document.py:295-298`, fired "ONLY from the export loops… NEVER from render()"), using the
  export's own `i / details.fps` and `dt = 1.0 / details.fps` — completely decoupled from
  `app.frame_idx` or `app.last_tick_time`.
- `export_isolation()` wraps the whole export (`document.py:930-934`) to give the script a fresh
  instance, per `conventions.md`'s "A document restarts through ONE funnel" bullet (§6) — another
  reason export's script state can never leak from or into the live loop's throttled cadence.

**Conclusion: export is structurally unaffected by any live-loop throttle.** It never reads
`global_target_fps`, `app.frame_idx`, `Document._frame`/`begin_frame`'s idempotency-by-live-frame
(it calls `begin_frame(None)` which just means "advance one," self-paced), or the profiler. A
throttle that only changes how often `_update_and_draw`'s render-set loop calls
`document.render()` touches none of this code path. The spec can state this as a fact, not an
intent to preserve.

**The copilot's probe renders one frame on demand, independent of any cadence.**
`_probe_frame` (`copilot/backend.py:148-165`) calls `document.render(u_time=t, target=target)`
directly (`backend.py:158`) with an explicit `t` the caller chose (`copilot/backend.py:2139`
notes the same `t` is shared between the render call and a stamp) — no `profiler=` (defaults
null), no dependency on `app.frame_idx`, `begin_frame`, or the render set. It is a synchronous,
one-shot render call made from wherever the copilot backend runs (main thread today), reusing
`Document.render`'s generic machinery but entering it exactly once per probe, on its own timing.
**Conclusion: the copilot probe is likewise unaffected** — it does not go through
`_tick_frame_state`'s render set or any per-UI-frame gate; a throttle that lives inside
`_tick_frame_state`/the render-set loop in `ui.py` has no surface to intersect with it.

**Where either WOULD be affected**: only if the throttle's chosen implementation moved the
cost-measurement or gating logic into `Document.render` itself (rather than keeping it in
`ui.py`'s call sites) — e.g. if `render()` grew a "should I actually draw this call" gate keyed
off some shared per-document throttle state. That would need an explicit bypass for export and
the probe (mirroring the existing `profiler: Profiler = NULL_PROFILER` pattern, §6's D3 quote:
"what a render reports to is decided by its CALLER, as an explicit parameter"). The natural design
— gate the *call* to `document.render()` in `ui.py`'s render-set loop, not `render()` itself —
avoids this entirely, since export and the probe never go through that loop.

---

## 4. The throttle input: cost measurement today

**How per-document GPU cost is measured today.** The 088 profiler's `document:<name>` CPU span
wraps each `document.render()` call (`ui.py:322-341`); inside it, `pass:<name>` GPU spans (via
`profiler.gpu(f"pass:{name}", count=entry.iterations)`, `document.py:712`) are `GL_TIME_ELAPSED`
queries in the three-deep, path-keyed ring (`profiling.py::Profiler._query_for`,
`profiling.py:227-238`), read two frames late (`profiling.py::_read_pending`,
`profiling.py:250-260`, wired from `begin_frame`'s `self._read_pending((self._frame_index + 1) %
RING_DEPTH)`, `profiling.py:180-188`). A document's total GPU cost for a frame is `gpu_total`
over its `document:<name>` span's subtree (`profiling.py:88-92`, sums every GPU span, which
"cannot nest" so nothing double-counts). This number becomes available to any reader via
`app.last_profile` (set at `ui.py:273`, the RAW `Profiler.last_complete`) or the smoothed
version via `app.profile_smoother.smoothed()` (`app.py:489`, `ProfileSmoother`, `profiling.py:270-336`).

**Recording is only on with the panel open — confirmed.** `ui.py:781`:
`app.profiler.enabled = app.fps_details_open`, executed unconditionally every frame right after
`fps_overlay(...)` returns the panel's open/closed state into `app.fps_details_open`
(`ui.py:771-781`). `Profiler.enabled`'s setter (`profiling.py:174-183`) only *records the wish*
(`self._wanted = value`); it is applied at the next `begin_frame` (`profiling.py:185-186,
198-200`, `_apply_wanted`), and disabling **drops the whole ring** (`profiling.py:200-215`,
`_drop`: `self._ring = {}` etc. — "the ring's ONE eviction rule", also stated in `conventions.md`
D2 quote §6). So today: panel closed → `enabled` is forced `False` every frame → no queries
exist, `app.last_profile` stops updating (frame closes instantly with `complete=True` and an
essentially-empty tree, since no GPU span ever opens — `Profiler.end_frame`,
`profiling.py:190-198`, `if self._gpu_spans: ... else: profile.complete = True`) → **GPU cost
data is unavailable whenever the panel is closed**, which is the normal operating state.

**What would have to change for cost to be available every frame as a throttle input.** The
`app.profiler.enabled = app.fps_details_open` line (`ui.py:781`) is the single point that ties
recording to panel visibility; a throttle needs `enabled` to also be `True` whenever throttling
is active, independent of the panel. The natural fix is an OR: `enabled = app.fps_details_open
or throttle_is_active`. The ring itself, once enabled, behaves identically regardless of why it
is enabled — nothing in `Profiler` assumes the panel is the only consumer (D3's whole point,
quoted §6, is caller-supplied wiring).

**Cost of always-on GPU queries.** 088's spec quotes exact measured numbers for the STEADY-STATE
(panel-open) cost, which is what "always on" would mean if a throttle needs the ring live
continuously: "a three-deep ring read two frames late stalled **0.009 ms**, four-deep 0.007 ms"
under real fragment load (`01_spec.md`, "What was measured before deciding"). The post-implementation
note refines this further: "depth 3 measured under 0.05 ms in every run" and the round-2 reviewer
"re-measured the mechanism as now written… at a 0.013 ms worst read stall with both siblings
reading their own values (5.1 ms and 20.4 ms for a 1:4 load)" (`01_spec.md` review history). So
the READ cost of an always-on three-deep ring is negligible (under 0.05 ms/frame) once
steady-state. What is NOT measured in 088's spec is the cost of the `ctx.query(time=True).mglo.begin()`/`.end()`
pair itself when it wraps EVERY pass of EVERY document EVERY frame at all times — 088 measured
timer-query overhead only in the context of "panel open," i.e. already assuming the cost is
acceptable when a human is watching; it never measured or argued that this is fine as a
permanent, always-on background cost with nobody watching. **This is a real gap the throttle spec
must either re-measure or explicitly accept on the strength of the numbers above** (the read
stall is the part 088 worried about and pinned; the query begin/end pair itself is presumably
cheap — "fifty clears of a 1024-square target read as 0.042 ms" establishes draw-timing
granularity, not query overhead specifically — but no isolated query-pair-overhead number exists
in either doc).

**`app.last_profile` as the throttle's actual read path.** Whatever the throttle uses as its
"measured GPU cost of document X" input, the natural source is `app.last_profile`'s
`document:<name>` subtree total (via `gpu_total`), which is **two frames stale** by 088 D2's own
design (the ring's read-two-frames-late rule) — a throttle reacting to it is inherently reacting
to slightly old information, which is fine for a slow-moving "is this document heavy" signal but
means the very first frame(s) after a document's cost spikes will not yet be throttled.

---

## 5. Candidate throttle policies

UI at 60 fps ⇒ UI frame budget = 16.67 ms. Costs evaluated: 5, 10, 30, 60, 100 ms.

**(a) Explicit per-document target fps setting.** A user- or engine-set `document_fps` field;
render every `round(global_target_fps / document_fps)`-th UI frame. Formula: `k =
round(global_target_fps / document_fps)`, document actually renders at `global_target_fps / k`.
This ignores measured cost entirely — it is a manual knob, not a throttle reacting to load. Doesn't
answer "exceeds a budget share" at all unless something else sets the field from cost. State
needed: one `int`/`float` field per document (`ui_models.py`'s per-document model, e.g.
`UIDocumentState`) — persisted, user-visible. Numbers (assuming, say, a fixed 15 fps setting
regardless of cost): every document renders at 15 fps whether it costs 5 ms or 100 ms — this
policy cannot express "throttle only the expensive ones" without an external cost→fps mapping,
so it is really a manual override, not itself an answer to the maintainer's ask.

**(b) Share-of-wall-time cap: document cost × document fps ≤ share × UI fps budget.** Formula:
`document_fps = min(global_target_fps, floor(share × budget_ms / cost_ms))` where `budget_ms =
1000/global_target_fps` (16.67 ms at 60 fps) and `share = 0.5` (088's existing knee, `theme.py:322`
`LOAD_WARN_RATIO`). At 60 fps UI, `share × budget_ms = 8.33 ms`:

| cost | document_fps = floor(8.33/cost) capped at 60 |
|---|---|
| 5 ms | floor(8.33/5)=1 → 1×60=60 fps (uncapped, cost fits share) |
| 10 ms | floor(8.33/10)=0 → degenerate; needs a floor of 1 render every N frames, not 0 — policy must define a minimum fps (e.g. clamp to ≥1 fps) |
| 30 ms | 8.33/30 ≈ 0.28 → render roughly every 4th UI frame ⇒ ~15 fps effective |
| 60 ms | 8.33/60 ≈ 0.14 → every ~7th frame ⇒ ~8.6 fps |
| 100 ms | 8.33/100 ≈ 0.083 → every ~12th frame ⇒ ~5 fps |

This directly encodes "cost × fps ≤ budget share," which is literally the maintainer's stated
rule. It needs no fixed table, only `share` (could reuse `theme.LOAD_WARN_RATIO = 0.5`) and the
live `global_target_fps`. State needed: the computed `document_fps`/`k` is **derived**, not
stored — recomputed each time a fresh cost sample arrives; could live entirely in `App`/`ui.py`
as ephemeral per-document throttle state (a dict `document_id -> next_render_at_frame_idx` or
similar), not in the persisted `ui_models.py` document model, since it is a live control-loop
output, not a user setting.

**(c) "Every k-th frame" with k from cost/budget.** Essentially the same math as (b) but stated
as an integer stride directly: `k = max(1, ceil(cost_ms / (share × budget_ms)))`. At 60 fps,
`share × budget_ms = 8.33 ms`:

| cost | k = ceil(cost/8.33) | effective document fps = 60/k |
|---|---|---|
| 5 ms | 1 | 60 fps |
| 10 ms | 2 | 30 fps |
| 30 ms | 4 | 15 fps |
| 60 ms | 8 | 7.5 fps |
| 100 ms | 12 | 5 fps |

This is the same underlying formula as (b), just quantized to an integer stride up front rather
than a continuous fps — simpler to implement as "render when `app.frame_idx % k == 0`" (paired
with the document's own `next_render_at` to avoid drift, since `k` can change between
recomputations as cost changes). State needed per document: the current `k` (or
`next_render_frame_idx`) — again ephemeral control-loop state, not a persisted setting; it would
live beside `first_render_done` conceptually (i.e. on `Document`, since it's about the render
loop's own bookkeeping) or on `App`/`ui.py`'s tick-frame-state locals, mirroring where
`tick_documents` itself is built (`ui.py:232-258`) — the natural insertion point for "skip this
document this frame" is right there, in `_tick_frame_state`, since that's already where the
render set is decided.

**Which state lives where.** (b) and (c) both need: the last measured cost (read from
`app.last_profile`/`app.profiler`, §4 — already exists, just needs always-on recording), the
computed stride/fps (ephemeral, recomputed from cost each time a fresh reading lands — belongs
with the render-set decision in `ui.py::_tick_frame_state`, not in `ui_models.py`'s persisted
document state, since it is derived and volatile, not a user choice), and possibly a persisted
`share` knob if the maintainer wants it tunable (that alone would belong in `UIAppState`
alongside `global_target_fps`, `ui_models.py:237`). (a) is the only policy needing genuine
persisted per-document state, and only because it is a manual override rather than a reactive
throttle.

---

## 6. Existing decisions this throttle touches

Quoted from `ai_docs/conventions.md ## Design decisions` (line numbers as read):

> **A document is N passes; a Pass owns everything about ONE shader (feature 065).**
> … **feedback swaps at the FRAME boundary** (`Document.begin_frame(frame)`), never inside
> `render()` — the live loop draws a document twice per frame and the copilot probe twice back
> to back. `begin_frame` takes the frame NUMBER and is idempotent within one, so a second call
> site cannot corrupt the history… Revisit if a pass ever needs to draw more than once per frame
> (MRT, ping-pong within a frame) — then the memo key stops being the pass name.
> (`conventions.md:225-236`, the 065 bullet)

**Honors.** A throttle that simply calls `begin_frame(frame_idx)` less often (skipping it
entirely on frames the document doesn't render) is exactly what this decision's idempotency
property was built to tolerate — `begin_frame` was designed so extra or skipped calls at
different frame numbers behave correctly (`frame == self._frame` short-circuits; the swap gate
checks `drawn_frame != previous_frame`). **Does not need to be revisited** — the "revisit if a
pass draws more than once per frame" clause is about MRT/ping-pong, not throttling frequency.

> A document restarts through ONE funnel, `ProjectSession.reset_document`: histories, clock,
> script instance, and every bound video through the clock.
> (`conventions.md:270-281`, the 089 D-area bullet)

**Honors, unaffected.** Reset is orthogonal to render cadence — it zeroes `time_origin` and
drops history regardless of how often the document had been rendering.

> **What a render reports to is decided by its CALLER, as an explicit parameter (feature 088).**
> `Document.render(..., profiler: Profiler = NULL_PROFILER)`; the live loop passes `app.profiler`
> at each of its render sites and every other caller — exports, the copilot probe, the dogfood
> harness, the smoke script, every test — takes the default and reports nowhere.
> (`conventions.md:1185-1198`, the 088 D3 bullet)

**Extends.** A throttle's cost INPUT is this exact mechanism, read back out via
`app.last_profile`. The throttle does not change who reports to the profiler (still the live
loop, still per the explicit-parameter posture) — it adds a new *consumer* of the profiler's
output (the gating decision in `ui.py`) and, per §4, needs `enabled` decoupled from
`fps_details_open` so the ring runs even with the panel closed. That is an *extension* of D3's
posture (still caller-decided, still explicit), not a reversal — the panel is simply no longer
the only caller-decided reason to enable it.

> **A GPU span is a `GL_TIME_ELAPSED` query, they never nest, and the read is two frames late.**
> … Two frames of margin measured under 0.05 ms in every run. A profiler that adds a frame to the
> frame is not an instrument, so `RING_DEPTH` is 3 with the measurement beside it rather than a
> tunable.
> (`conventions.md:1199-1215`, the 088 D2 bullet)

**Honors, with a caveat.** The two-frame staleness (§4) becomes the throttle's own reaction
lag — a real cost, but the decision itself (three-deep ring, why) is unaffected and should not be
touched. The caveat: 088 never measured or argued for the cost of the ring being **permanently**
open (§4's gap); this decision's "under 0.05 ms in every run" numbers were all gathered with the
panel open (a human actively watching), not as a background always-on cost — the throttle spec
should either re-measure that specific case or note the extrapolation explicitly.

**066 D2, the render-set admission rule** (`ai_docs/dev_flow.md`/`conventions.md` "one DOCUMENT
… and inside each ticked document one PASS off the output chain," `conventions.md:283-297`,
also directly narrated at `ui.py:232-258`): bounds first-render **compile** cost, one
never-rendered document/pass admitted per frame. **Untouched by a throttle** — it only governs
documents/passes that have never rendered; a throttle only ever acts on documents that have
already had their first render (§2's render-set-membership row). The two mechanisms are
independent and can compose without conflict: a document must first clear 066 D2's one-per-frame
admission before a cost-based throttle has anything to measure.

**090's own decisions audit** (`00_research.md`, "Decisions the spec must take," item 6):

> **The profiler across two cadences.** 088's three-deep ring read two frames late assumes one
> loop. The worker profiles its own loop; the FPS panel shows the UI frame beside the document's
> own period. The "GPU share of budget" coloring changes meaning when the document no longer
> shares the UI's budget.

This item was written for 090's THREADED/tiled design (a second GL-thread loop with its own
`begin_frame`/query ring) — tiling was rejected, so there is no second loop and no second
cadence in the throttle's scope; the ring stays one instance, one loop, exactly as 088 built it.
**The throttle does NOT reopen this item** — it is 090's threading design that would have, and
that part was withdrawn. However, the throttle DOES independently change what "GPU share of
budget" *means* in the FPS panel (§2's last table row): today a `document:<name>` row's color
(`load_color(ms/budget_ms)`, `ui_primitives.py:1391-1395`, knees at `LOAD_WARN_RATIO=0.5` /
`LOAD_ERROR_RATIO=1.0`, `theme.py:321-323`) reads as "this document's share of THIS frame's
budget." Under a throttle, a document at "5 fps" is by design spending far more than 50% of any
single UI frame it DOES render in — the whole point is that its cost is amortized across the
frames it skips. The panel showing that document's row in red/error every time it renders (which
is now rarer, e.g. 1 in 12 UI frames per the 100 ms example in §5) would visually read as
"still overshooting," even though the throttle is working exactly as designed and the document's
amortized share is on target. **The spec must decide**: does the panel's per-document coloring
change to reflect amortized share (`cost × document_fps / UI_budget`, which is by construction
≈ `share` once the throttle has converged) rather than raw single-frame share, or does it keep
showing raw share with a caveat that a throttled document's raw number is expected to look
red? This is a genuine open decision, not resolved by any existing text.

---

## 7. False trails and coverage

**False trails ruled out by direct reading:**
- **`u_frame` does not exist.** No frame-counter uniform is ever written to a shader
  (`engine_uniforms.py` enumerates exactly five per-frame engine uniforms, none of them a frame
  index). A throttle proposal that assumes shaders can read "which frame is this" is working from
  a uniform that isn't there — `u_pass_iteration`/`u_pass_iterations` are iteration-local, not
  frame-global, and reset to 0 on every `render()` call regardless of UI frame.
- **`app.frame_idx` is not per-document.** It is tempting to read it as "the current document's
  frame count," but it's a single UI-loop counter shared by every document ticked that frame
  (`ui.py:539`, `ui.py:260`, `ui.py:265`) — a throttle needs its own per-document notion of
  "which of my frames is this," which does not exist yet.
- **The preview-canvas double-render is gone** (088's spec, "Out of scope" / post-implementation
  notes: "the preview canvas was gone before implementation"). The current "two renders per
  document per frame" is the output-chain-plus-one-pending-pass case (`ui.py:322-341`), NOT the
  old preview-then-own-canvas pattern 088 originally designed the path-keyed ring around. Both
  reasons for `begin_frame`'s once-per-frame idempotency (quoted in `conventions.md`) still apply
  to the current two-call shape, so the design intent carries even though the original motivating
  case is gone.
- **The GPU-cost-availability gap is NOT that the profiler is somehow document-unaware.** It
  already attributes cost per document (`document:<name>` spans) and per pass within it
  (`pass:<name>` spans) — the gap is purely that recording is off whenever the panel is closed
  (§4), a single boolean, not a structural limitation of the instrument.
- **Export and the copilot probe do not need a throttle bypass flag.** They never enter the code
  path a throttle would live in (`ui.py`'s render-set loop / `_tick_frame_state`) at all — no
  existing `NULL_PROFILER`-style default is needed unless the throttle logic is pushed down into
  `Document.render` itself, which §3 argues against.

**Coverage statement.** Read end to end, with file:line citations: `shaderbox/ui.py` (`run`,
`_tick_frame_state`, `_update_and_draw`/`update_and_draw`, the render-set loop, `_pump_file_gate`
skimmed as irrelevant); `shaderbox/document.py` (`Document.__init__`'s clock/frame fields,
`render_pass`, `begin_frame`, `_swap_feedback`, `reset`, `live_time`, `reset_feedback`,
`render`, `_render_image`, `_render_video`, `render_media`); `shaderbox/core.py`
(`Pass.render`'s uniform resolution, `u_time`/`u_aspect`/`u_resolution`/`u_pass_iteration(s)`
branches); `shaderbox/engine_uniforms.py` (full file, 23 lines); `shaderbox/help_content.py`
(the engine-uniform glosses and the fps_panel help text, grepped and read in context);
`shaderbox/scripting/engine.py` (`ScriptEngine.tick` signature) and
`shaderbox/scripting/context.py` (`ScriptContext`, `MouseState`, `EXPORT_MOUSE`, read via the
persisted-output preview since the file exceeded a convenient inline read — the `frame` field's
docstring and type were confirmed directly); `shaderbox/project_session.py::tick` (the
per-document wrapper around `ScriptEngine.tick`, read in full); `shaderbox/profiling.py` (entire
file, all of D1–D7's runtime); `shaderbox/ui_primitives.py` (`profile_rows_plan`, `_measured_row`,
`_plan_tree`, `_profile_number`, the FPS-panel row-building path); `shaderbox/theme.py`
(`load_color`, `LOAD_WARN_RATIO`, `LOAD_ERROR_RATIO`); `shaderbox/ui_models.py`
(`UIAppState.global_target_fps`, `is_render_all_documents`; confirmed no per-document fps field
exists anywhere in the file); `shaderbox/media.py` (`Video.update`, wall-time-driven via `t *
fps % n_frames`, confirming video sampling is already a pure function of `u_time`, not of render
count); `shaderbox/copilot/backend.py` (`_probe_target_for`, `_probe_frame`, confirming the
probe's one-shot render); `shaderbox/app.py` (`self.profiler`, `self.last_profile`,
`self.profile_smoother`, `self.frame_idx`, `self.fps_details_open`); `ai_docs/conventions.md`
(full `## Design decisions` section header list, then the 065/088-D2/088-D3 bullets read in
full); `ai_docs/features/090_render_decoupling/00_research.md` (full file) and
`ai_docs/features/088_frame_profiler/01_spec.md` (full file, including review history). Tests
grepped for `target_fps`, `begin_frame`, `drawn_frame`, `u_time`, `iterations`: the cadence-
relevant hits are `tests/test_model_salvage.py` (pins `global_target_fps` bounds and
divide-by-zero safety), `tests/test_profiling.py` (the `profile_rows_plan`/smoother/panel-wiring
tests, including the enabled-at-frame-boundary and ring-drop-on-disable tests read at lines
600-645), and `tests/test_document_reset.py` (`test_a_live_render_counts_from_the_document_clock`,
`test_an_explicit_u_time_ignores_the_document_clock` — both confirm the wall-clock-vs-explicit-
`u_time` split this research relies on in §1/§3). No test today pins a specific render-cadence
relationship between `app.frame_idx` and `Document._frame`, which is consistent with §1's finding
that nothing currently assumes they are the same counter beyond the one call site that happens to
pass the same value.

Not read in full (out of scope for cadence, confirmed by name/grep only): `shaderbox/exporters/video.py`
does not exist as such — export fps/step-time logic lives in `Document._render_video` itself
(§3), not a separate exporters module; the actual `exporters/` package (`base.py`, `registry.py`,
`telegram.py`, `youtube.py`) governs delivery/upload, not frame stepping, and was not read since
it sits downstream of `render_media`'s already-confirmed deterministic stepping.
