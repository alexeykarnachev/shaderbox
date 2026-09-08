# 088 — Frame profiler

A frame-time breakdown behind the FPS chip, built as a general profiling seam rather than a set of
timers: the maintainer plans CPU scripts, multi-turn passes and, later, documents wired into other
documents as black-box nodes, and wants each of those to show up in the same tree without a second
mechanism.

Source: the maintainer's `../TODO`, verbatim: "we have this fps counter at the top right corner of
the rendering, but I also want to know the frame time, i.e the upper bound not capped by fps. Can
we collect this time somehow effectively to estimate the honest full time of the frame. Actually,
we can even provide a little bit of detalization: script time, full rendering time, per-pass
rendering time (all the passes which participate in the current frame)... something else? And we
can show this popup when clicking on the FPS counter (the same way as we do now). This is probably
should be a mid-level feature, because we need carefully and efficiently set up the generalizable
profiling mechanism. I'm planning to work extensively with cpu scripts, with multi-turn passes and
whatnot. So, this shouldn't be just an ad-hoc work-arounds. Make it easilty extandable, i.e later I
want to inroduce cross-documents wiring such that we can implement a document (e.g radiance cascade
for example) and reuse it as a black box node in some other document, so the profiling should
properly reflect this."

Size: **mid, upper end** — a new leaf module, a signature on `Document.render`, spans at six sites
in the frame loop, and a GL-timed test. One pre-implementation reviewer on the seam, one on the
verification.

---

## Goal

Clicking the FPS chip opens the panel it opens today, and the panel shows:

- the honest frame time — the CPU wall of one `update_and_draw` (the sleep the FPS cap adds is
  excluded), and the GPU time the frame's draws took, each as its own number, because the two
  overlap and neither alone is the bound;
- the frame's budget at the current target FPS beside it;
- a tree: the script tick, each document rendered this frame, each pass under it (its iterations
  folded into one row), the editor panel, the UI build and draw, the buffer swap;
- all of it at zero cost while the panel is closed.

---

## What was measured before deciding

Measured on the dev box (RTX 3090, GL 3.3 core, moderngl 5.12) with a throwaway script, since the
design turns on these numbers:

- **`GL_TIME_ELAPSED` queries do not nest.** Two `ctx.query(time=True)` one inside the other:
  the outer reads a garbage value, the inner reads 0, and `ctx.error` is `GL_INVALID_OPERATION`.
  No exception is raised. So an overlapping GPU span is a silent wrong number, and the seam must
  refuse it loudly.
- **Reading a query blocks until the GPU has drained past it, and one frame of pipelining is
  not enough margin.** Reading `elapsed` right after 200 clears blocked 0.4 ms. Reading the
  previous frame's queries cost 0.09-0.16 ms on an IDLE GPU — the first draft's number — but the
  pre-implementation reviewer re-measured under a real fragment load with `swap_buffers` in the
  loop: a two-deep ring read one frame late stalled **22.3 ms** at worst and 6.7-7.1 ms on every
  frame of a lighter run; a three-deep ring read two frames late stalled **0.009 ms**, four-deep
  0.007 ms. A profiler that adds a frame to the frame is not an instrument, so the ring is three
  deep (D2).
- **One query object begun twice in a frame reports only the second block, with no GL error.**
  Measured by the reviewer: `elapsed` 0.0 ms for a real first block. So a ring keyed by a name
  that opens twice per frame — which the current document's preview-then-own-canvas pair does on
  every frame — loses the first render in silence (D2).
- **`moderngl.Query` has no `release()` and no `__del__`.** A query object is a permanent GL name
  for the process, so the ring must be bounded by something other than garbage collection (D2).
- **The cost of a draw is small and measurable:** fifty clears of a 1024-square target read as
  0.042 ms, four sequential spans of 10/20/30/40 clears read 0.007/0.014/0.022/0.029 ms — linear,
  which is what a timer that works looks like.
- **The loop today:** `ui.run` measures `elapsed_time` around `update_and_draw` and then sleeps to
  the target; the FPS it shows is an EMA of the SLEPT period, which is exactly "capped by fps". The
  honest number already exists as a local and is thrown away.
- **What the frame does, in order** (`ui.update_and_draw`): reconcile disk and scripts and tick the
  engine (`_tick_frame_state`); render the CURRENT document into the 200-px preview canvas — the
  whole chain, intermediates at full size; render every document in the tick set into its own
  canvas — the current one AGAIN, plus every other document when "render all" is on (the default);
  then imgui builds the frame, the editor lays out and (when its state moved) redraws its panel,
  the channel blit runs for the viewer, imgui draws, the buffer swaps. The current document's
  chain therefore draws twice per frame; the profiler is what makes that visible (see Out of
  scope).

---

## Out of scope

- **Fixing what the numbers show.** The preview canvas re-render (the pass strip already blits
  the live target instead of re-rendering, 065; the document grid's preview could do the same) and
  every-document-every-frame under "render all" are both real and both stay as they are here: this
  feature is the instrument. Trigger: the maintainer reads the panel and asks.
- **Persisting or exporting a profile.** The panel shows the last completed frame; nothing is
  written. Trigger: the maintainer wants a trace over time (a chart, a CSV).
- **Profiling exports and the copilot probe.** Both render through `Document.render` with the null
  profiler; a per-frame tree is the live loop's. Trigger: an export's per-pass time is asked for.
- **Cross-document nodes themselves.** Unbuilt. D5 says why the tree needs nothing new for them.

---

## Design decisions

### D1 — one leaf module, `shaderbox/profiling.py`, holding the tree and the two span kinds.

`Profiler` builds one `FrameProfile` per frame: `begin_frame()` opens a root, `cpu(name)` and
`gpu(name)` are context managers that push a `Span(name, cpu_ms, gpu_ms, children)` under the
current one, `end_frame() -> FrameProfile` closes the root and hands back the tree. CPU time is
`time.perf_counter` around the block. The module imports `moderngl` for the queries and nothing
else — no imgui, no `App`, no `Document`, so it sits with `core.py`'s leaves and any caller can
take it.

A span's identity is its NAME under its parent, `kind:name` where a kind is ambiguous (`document:
Bloom`, `pass:blur`, `script`, `editor`, `ui`, `swap`). A second span with the same name under the
same parent in one frame is a DISTINCT span with its own sibling ordinal — the tree shows
`pass:blur` under `preview` and again under `document:Bloom`, never a merge — because the current
document's chain is drawn twice per frame today and the panel exists to show exactly that. A pass
drawn N times in its one turn of the order (068 D1) is ONE span whose `gpu(...)` block wraps the
iteration loop; the row shows `x N`.

The profiler resolves the GL context LAZILY, at the first `gpu()` entry of an enabled profiler,
never in `__init__`: `NULL_PROFILER` is built at import time by `document.py`'s default argument,
and `import shaderbox.document` must keep working with no window (the module-map's "live context
before constructing `Canvas` / `Document`" rule is about those constructors, and the profiler
must not extend it to an import).

### D2 — GPU spans are `GL_TIME_ELAPSED` queries in a three-deep ring keyed by PATH, read two frames late, and a nested one raises.

Per span PATH — the parent chain, the name and the sibling ordinal within the frame, so the two
`pass:blur` spans of one frame are two keys — the profiler keeps three `ctx.query(time=True)`
objects. Frame N's block runs inside slot `N % 3`; at frame N+2's `begin_frame` the profiler reads
every slot frame N used and writes the milliseconds into frame N's tree. A `FrameProfile` is
therefore complete two frames after it closes, and the panel draws the last COMPLETE one. Two
frames of margin is what the measurement above says the read needs to stop blocking on the GPU
(0.009 ms against 22.3 ms one frame earlier); the same measurement is why the depth is a named
constant with the number beside it and not a tunable.

The ring is bounded by the panel: disabling the profiler (D4) drops the whole ring dict, and a
query that is never begun again is a handful of GL names until the next open. That is the eviction
rule — one clause, because `Query` cannot be released and a pass rename or a newly opened document
would otherwise grow the ring for the life of the process.

`gpu(...)` asserts no GPU span is open. The GL error for a nested query is silent and the numbers
it produces look plausible; `ui.update_and_draw` also calls `clear_errors()` every frame
(`ui.py`, after the document renders), so `ctx.error` could not serve even as a late check. The
`assert` at the seam is the only guard there is, and the test that pins it nests two spans and
expects the raise. CPU spans nest freely and a GPU span may sit inside a CPU span (a document's
CPU span holds its passes' GPU spans).

### D3 — the profiler reaches `Document.render` as an explicit parameter; the loop passes `app.profiler`.

Two shapes were weighed:

    # A — explicit (chosen)
    document.render(profiler=app.profiler)            # ui.py, each render site
    ...
    with profiler.gpu(f"pass:{name}"):                # Document.render, around the iteration loop
        for iteration in range(entry.iterations): ...

    # B — a module-level active profiler, like shader_lib.index.active
    with profiling.active().gpu(f"pass:{name}"): ...

A is the repo's posture (`ProjectSession`'s injected callbacks, the export isolation factory): what
a render reports to is decided by its caller. B would also capture an export's or the copilot
probe's passes into whichever live frame they ran inside, because those go through the same
`Document.render`, and it would need a reset discipline nobody wants. `Document.render` gains
`profiler: Profiler = NULL_PROFILER`; `Pass.render` is untouched — the document wraps the call,
which is also where the pass's name and iteration count are known. A disabled `Profiler` is the
null object: its context managers cost one attribute read and create no query.

The spans in the loop, each at the call site that already exists:

| span | where |
|---|---|
| `frame` (the root) | a context manager wrapping the WHOLE of `ui.update_and_draw`: `begin_frame` at entry, `end_frame` at exit, so the abort path (`_tick_frame_state` returning `None`) closes an empty root like any other frame and the ring never desynchronises from the frames that drew |
| `tick` (sync, scripts, engine) | `ui.update_and_draw`, around `_tick_frame_state` |
| `script` | inside `tick`, around `session.tick` |
| `preview` | around the preview-canvas render |
| `document:<name>` / `pass:<name>` | around each `document.render()`; the passes from inside |
| `editor` (layout) / `editor:draw` (GPU) | `tabs/code.py` around `layout_following_cursor` and `panel.render` |
| `viewer` (GPU) | `ui._draw_document_image`, around the channel blit on the Alpha / RGB branches only — the Color view blits nothing and gets no row |
| `ui` (build) | around the imgui frame build, `new_frame` to `render` |
| `ui:draw` (GPU) | around `imgui_renderer.render` |
| `swap` | around `glfw.swap_buffers` |

The frame's `cpu_ms` is the root's wall: `update_and_draw` entry to exit, which includes the swap
and excludes the sleep `run` adds after it — so `run` needs no change, and a headless caller of
`update_and_draw` (the smoke script, V6's test) gets a root too. The sum of the root's children is
less than its wall, and the difference draws as `other` so the panel never claims to account for
time it did not measure.

### D4 — recording is on while the panel is open, and the chip keeps its EMA.

`app.profiler.enabled = app.fps_details_open`, set where the overlay's return value is written.
Closed, nothing is timed and no query exists — the maintainer's "efficiently". Open, the first
frame shows the panel with no tree (nothing complete yet) and every frame after shows the previous
one. The chip's `N FPS` stays the slept-period EMA it is: that number answers "am I hitting the
cap"; the panel answers "what would I hit".

### D5 — the panel is `fps_overlay`'s existing child, widened, drawing the tree as indented rows.

`fps_overlay(...)` gains `profile: FrameProfile | None` and `number_font: imgui.ImFont` (the
primitives module never imports `App`; `small_caption(font, text)` is the shape, and `ui.py`
passes `app.font_12`). `SIZE.FPS_PANEL_W` goes 160 -> 280. Each row is ONE `caption_text` whose
text is the label alone, with its number drawn right-aligned at the panel's edge in `number_font`
by a small row helper — so every string the prose-budget gate scores is one or two words, under
its budget of four (`tests/test_ui_prose_budget.py` already scores this overlay's two lines; a
label-plus-number-plus-unit string scored six and failed it in review). The tree indents
`SPACE.MD` per level. Fixed words are `frame`, `gpu`, `budget`, `fps`, `target`, `other`, `x N`;
everything else is a span name, which is data. The first rows:

    frame            4.2 ms
    gpu              9.8 ms
    budget          16.7 ms
    fps              60
    target           60

then the tree. Two numbers on the first lines because CPU and GPU overlap: the honest bound the
maintainer asked for is "the larger of the two, plus whatever they fail to overlap", and showing
both is what lets him see which one he is bound by. No `help_marker` — the words are the
control's names (imgui-ui §2 word budget).

### D6 — the tree is dynamic nesting, which is what a cross-document node needs and nothing more.

A span's parent is whatever span is open when it starts. A document rendered as a node inside
another document's render will run `Document.render(profiler=...)` from inside the outer document's
pass span, and its own `document:` and `pass:` spans nest there by construction — no registry of
documents, no flat list keyed by id, nothing that assumes one level. That is the whole of the
extensibility promise, and it holds because D3 threads the SAME profiler object down rather than
having each document look one up.

### D7 — the null profiler is the default everywhere; the live loop is the only enabler.

`NULL_PROFILER = Profiler(enabled=False)` at module level. `Document.render`'s default, the
headless harness, exports, the smoke script and every test that renders a document all get it for
free. The `enabled` flag is read once per span entry; a disabled span yields without touching the
clock.

---

## Files touched

- `shaderbox/profiling.py` (new) — `Span`, `FrameProfile`, `Profiler`, `NULL_PROFILER`.
- `shaderbox/document.py` — `render(..., profiler=NULL_PROFILER)`, the per-pass GPU span.
- `shaderbox/ui.py` — the root context manager around `update_and_draw`; `tick` / `script` /
  `preview` / `document:` / `ui` / `ui:draw` / `swap` spans; the overlay call passes the last
  profile and `app.font_12`. `run` is untouched.
- `shaderbox/tabs/code.py` — `editor` and `editor:draw` spans.
- `shaderbox/app.py` — `self.profiler`, `self.last_profile`.
- `shaderbox/ui_primitives.py` — `fps_overlay` draws the tree; gains `profile` and
  `number_font`; a right-aligned number row helper.
- `shaderbox/document.py`'s signature change is additive and trailing; the one positional caller
  is its own `_render_video` (`u_time` first), every other caller passes keywords, so no test
  that renders a document changes.
- `shaderbox/theme.py` — `FPS_PANEL_W`.
- `tests/test_profiling.py` (new) — the tree with a fake clock; the GL half on a hidden glfw
  window as `test_editor_ffi.py` does.
- `ai_docs/dev_flow.md` module map — one line for `profiling.py`.
- `ai_docs/conventions.md ## Design decisions` — the seam (D3) and the no-nesting rule (D2).

---

## Verification

- **V1 tree shape:** `cpu("a")` containing `cpu("b")` yields root -> a -> b with `b.cpu_ms <=
  a.cpu_ms` under a fake clock. Falsifier: pop the wrong span and `b` lands under root.
- **V2 nested GPU raises:** `gpu("x")` inside `gpu("y")` raises before any query begins.
  Falsifier: remove the assert and the test sees no raise (and the numbers go wrong in silence).
- **V3 late read:** after frame 1 (a span around 200 clears) and frames 2 and 3's `begin_frame`,
  frame 1's profile carries `gpu_ms > 0`; after frame 2's alone it still carries `None`.
  Falsifier: read one frame late and the "still `None`" assertion fails.
- **V3a same name twice:** one frame with `gpu("pass:a")` twice under different parents, each
  around a distinguishable load (10 and 200 clears), yields two spans whose `gpu_ms` differ by
  the load ratio. Falsifier: key the ring by name and the first reads 0.0.
- **V4 disabled costs nothing:** a `Profiler(enabled=False)` running a thousand spans creates zero
  queries (count `ctx.query` calls through a wrapper) and its `end_frame` returns an empty tree.
  Falsifier: create the query eagerly in `gpu(...)`.
- **V5 the document reports its passes:** a two-pass document rendered with an enabled profiler
  yields `pass:` spans named after `evaluation_order`, one per pass whatever its iteration count;
  rendered twice in one frame (preview canvas, then own canvas) it yields them twice, under two
  parents. Falsifier: wrap per iteration and the count doubles; merge same-name siblings and the
  second render vanishes.
- **V6 the wire exists:** the test drives `update_and_draw` headless for four frames with the
  panel open and asserts `app.last_profile` has a `document:` child with a `pass:` child carrying
  a number. This is the "defined is not wired" check of `dev_flow.md` step 7: cut the
  `profiler=app.profiler` argument at one render site and the child is gone.
- **V7 the abort path:** a frame whose `_tick_frame_state` returns `None` still closes its root;
  the next frame's tree is well-formed. Falsifier: open the root with a bare call instead of the
  context manager and the second frame's root nests under the first.
- **Maintainer's eyes:** the panel's width and the number column; whether the tree reads at a
  glance for a five-pass document.

---

## Open questions for the user

1. **D3 — explicit parameter (A) or module-level active profiler (B)?** Recommended: A.
2. **D4 — record only while the panel is open?** Recommended: yes. Alternative: always on, so the
   first click already shows a tree — costs about 0.1 ms per frame per handful of GPU spans.
3. **Out of scope — the preview-canvas double render.** Recommended: leave it for the numbers to
   argue. Alternative: fold a blit-based preview into this feature.

---

## Review history

**Round 1 (pre-implementation, one reviewer on opus, read-only, with its own GL probes).** Verdict
FAIL; eleven findings, all accepted:

- D2 two-deep ring stalls 22.3 ms under load (measured) → three-deep, read two frames late.
- D2 ring keyed by name discards the current document's preview render (measured 0.0 ms) → keyed
  by path with a sibling ordinal; D1 says two same-named siblings are two spans; V3a and V5 pin it.
- D5 first line scored six words against the gate's four → one label per `caption_text`, numbers
  through a right-aligned helper.
- D3 root site unstated and `run` never runs headless → root is a context manager around
  `update_and_draw`; `run` untouched; V6 rewritten; V7 added for the abort path.
- D1 module-level `NULL_PROFILER` must not resolve a GL context → lazy at first `gpu()`.
- D2 `Query` has no `release()` → the ring drops on disable.
- D5 `fps_overlay` cannot reach `app.font_12` → a `number_font` parameter.
- D3 `viewer` span on the Color view wraps nothing → blit branches only.
- D2 `ctx.error` is cleared every frame → stated; the assert is the only guard.
- V6 grep clause is not a test → deleted.

Rejected: none. False trails the reviewer recorded so round 2 does not re-open them: the nesting
fact holds exactly (outer reads garbage, inner 0, `GL_INVALID_OPERATION`, no exception); timer
linearity holds; the `gpu()` assert cannot fire on any current path (every GPU span is
sequential; the blit runs inside the `ui` CPU span); the trailing keyword default is safe at
every caller; post-swap export and copilot renders open no span because they take the null
default.

**Round 2 (same reviewer, against the patched text).** Verdict PASS: all eleven closed by cited
passages, no new findings. It re-measured the mechanism as now written — a path-keyed three-deep
ring under real fragment load with two same-named spans per frame — at a 0.013 ms worst read
stall with both siblings reading their own values (5.1 ms and 20.4 ms for a 1:4 load), and ran
the prose-budget scorer over every proposed panel string (all at or under four). One false trail
added for a later round: a three-slot ring consumed twice per frame is fine, since each path owns
its own ring and reuse stays three frames apart.
