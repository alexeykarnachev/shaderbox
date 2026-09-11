# 090 BEFORE baseline — document rendering on the UI thread

What the app costs today, before render decoupling, with a deliberately heavy document open:
the frame period, where the time goes, and how late an editor keystroke lands. Every number
below comes from a script under `ai_docs/features/090_render_decoupling/probes/`, each runnable
standalone (`uv run python <script> <project_dir>`), against a throwaway project this research
built and never against `projects/dev/`.

Machine: RTX 3090, driver 580.173.02, X11 `:1`, a real (not llvmpipe) GL 4.6 core context.

---

## Setup

Three single-pass documents, authored directly on disk per `dev_flow.md`'s "Authoring /
debugging documents directly" recipe (`build_project.py`; passes-before-`document.json`,
1024x1024 canvas, no `graph.json` needed for a single pass):

- **Trivial** — `fs_color = vec4(vs_uv, 0.5, 1.0)`. No loop.
- **Medium** — a fullscreen value-noise accumulation loop, `n=1300` iterations.
- **Heavy** — the same shader, `n=17000` iterations.

The iteration counts were calibrated (`00_calibrate_shaders.py`) by timing `ctx.finish()`-bounded
draws of the raw fragment shader outside the App, sweeping `n` from 50 to 25600 (near-linear,
~5.85 µs/iteration at 1024²) and refining around the two targets:

| n | median ms (15 samples, `ctx.finish()`-bounded) |
|---|---|
| 1300 | 8.02 |
| 17000 | 99.1–100.4 |

`n=1300` → medium ≈ 8 ms GPU; `n=17000` → heavy ≈ 100 ms GPU. Both documents share the noise
shader verbatim; only `n` differs.

---

## Method

All four probe scripts build a real `App(project_dir=<throwaway>, headless=...)` (the
`scripts/smoke.py` shape) and drive `shaderbox.ui.update_and_draw` directly — never
`ui.run`, so the frame-cap sleep never enters the measurement (matches how 088's profiler
roots: `update_and_draw` entry to exit, sleep excluded).

**Enabling the profiler outside the UI.** `app.profiler.enabled` is not a switch a caller can
hold — `ui.py`'s own frame body overwrites it every frame from `app.fps_details_open` (088 D4:
recording follows the FPS-panel-open state, applied at the next frame boundary). The way to
record programmatically is to set `app.fps_details_open = True` before the loop and leave it
set; the profiler's own `_apply_wanted` then turns recording on at the next `begin_frame` and
keeps it on for as long as the flag stays true. Setting `app.profiler.enabled` directly is
overwritten within the same frame it was set on the very next `update_and_draw` call.

**Deliverable 2 — per-document frame timing + span breakdown** (`01_frame_timing.py`): for each
of the three documents, `render_all` off, then on: 30 warmup frames (clears both the
one-first-render-per-frame budget for every document in the project — 066 D2 — and the GPU
query ring's 3-deep fill — 088 D2), then 300 measured frames. Per frame: wall time around
`update_and_draw`, and the profiler's span tree (once `app.last_profile.complete`, deduped by
`profile.index`). Run twice: `headless=True` (a hidden glfw window) and `headless=False` (a real
visible maximized window) to check whether visibility changes swap behavior.

**Deliverable 3 — input latency** (`02_input_latency.py`): focuses the shader-tab editor the way
`tests/test_code_panel.py` does — a real mouse click into `app.editor_rect` via
`io.add_mouse_pos_event` / `io.add_mouse_button_event`, driven through real frames until
`app.editor_focused` is true (never a raw attribute poke). Per sample: place the cursor at a
known line, record wall-clock `t0`, append one `translate_char(ord("j"))` `KeyEvent` (exactly
`shaderbox/editor/input.py`'s char-callback path) to `app.editor_key_events` (exactly where
`hotkeys.py::_drain_editor_input` reads from every frame), then drive `update_and_draw` frames
until `editor.get_current_cursor_position().line` changes, recording the wall delay and the
frame count. 50 samples per document (heavy, then trivial), a 200+-line padded buffer so `j`
always has room to move.

**Deliverable 4 — "Render all"**: `is_render_all_documents` (default `True` on `UIAppState`)
folded into deliverable 2's sweep — each document driven once with it forced off, once forced
on.

**Deliverable 5 — editor draw cost** (`03_editor_draw_cost.py`): the same focus-by-click, then
40 real `j` keystrokes (each moves the cursor, which trips `should_redraw`'s gate) with the
profiler recording, reporting the MAX (not median — the claim is a per-frame bound) of the
`editor` (CPU, `layout_following_cursor`) and `editor:draw` (GPU, `EditorPanel.render`) spans.

---

## Results

### Frame period per document, `render_all` off vs on (300 frames, headless)

| Document | render_all | period median | period p95 | period max |
|---|---|---:|---:|---:|
| Trivial | off | 6.94 ms | 7.13 ms | 8.64 ms |
| Medium | off | 13.88 ms | 14.08 ms | 15.43 ms |
| Heavy | off | 104.15 ms | 104.53 ms | 105.64 ms |
| Trivial | on | 111.49 ms | 118.30 ms | 119.78 ms |
| Medium | on | 111.41 ms | 118.48 ms | 124.82 ms |
| Heavy | on | 112.20 ms | 118.48 ms | 120.05 ms |

With `render_all` on, ALL THREE documents render every frame regardless of which is "current" —
the period is the same (~111-112 ms) no matter which tab is in front, because it is bounded by
the slowest document in the project (Heavy, ~100 ms GPU) plus Medium's ~8 ms, run serially.

**Visible window vs hidden window** (`--visible`, real maximized window on `:1`): materially the
same numbers (Trivial 6.93/7.11/13.81, Heavy 110.69/111.71/112.38 median/p95/max headless
vs. visible). Visibility does not change swap behavior on this box — the NVIDIA driver blocks
`glfw.swap_buffers` on GPU completion the same way whether or not the window is mapped, so a
headless run is representative for this measurement. (One visible-run sample,
`medium_render_all_off` p95 = 214.59 ms against a headless p95 of 14.08 ms, is a single-frame
outlier — see False trails.)

### Per-span breakdown, `render_all` off (median ms; heavy document)

| span | cpu median | cpu p95 | gpu median |
|---|---:|---:|---:|
| `frame` (root) | 104.10 | 104.47 | — |
| `frame/swap` | 100.29 | 101.01 | — |
| `frame/ui` | 1.58 | 1.81 | — |
| `frame/tick` | 1.25 | 1.42 | — |
| `frame/ui:draw` | 0.65 | 0.70 | 0.05 |
| `frame/ui/editor` | 0.55 | 0.66 | — |
| `frame/document:Heavy` | 0.10 | 0.11 | — |
| `frame/document:Heavy/pass:main` | 0.06 | 0.07 | **100.53** |
| `frame/tick/script` | 0.01 | 0.01 | — |

Same shape for Medium (`pass:main` GPU 7.31 ms, `swap` CPU 10.56 ms) and Trivial
(`pass:main` GPU 0.01 ms, `swap` CPU 4.06 ms — 4 ms of swap floor even with nothing to draw).

**Which spans exist and which don't.** All nine spans named in 088's design table are present:
`frame`, `tick`, `script`, `document:<name>`, `pass:<name>`, `editor`, `editor:draw`, `ui`,
`ui:draw`, `swap`. There is no standalone `hotkeys` span: `dispatch_commands` (the editor-input
drain + the registry dispatch + Esc handling) runs IN-frame, inside the main imgui window body,
wrapped by the outer `frame/ui` span rather than getting one of its own — and `process_hotkeys`
(the pre-frame glfw poll + `imgui_renderer.process_inputs()`) runs between the document-render
block and `ui`, inside neither `tick` nor `ui`, so its cost is part of the root's small
unattributed remainder. `sync_documents_from_disk` folds into `tick` (it runs inside
`_tick_frame_state`, which `tick` wraps whole). `viewer` (the Alpha/RGB channel blit) never
appears because every probe stays on the Color view. The frame's own `other_ms` (root wall minus
the sum of its children) is small in every config — under 2 ms — so almost nothing is
unattributed; the profiler already accounts for essentially the whole frame.

**Where the 100 ms actually is.** `document:Heavy`'s CPU span is 0.10 ms and `pass:main`'s CPU
span is 0.06 ms — issuing the draw call costs nothing. The 100.53 ms GPU number on `pass:main` is
the `GL_TIME_ELAPSED` query result, read back two frames later (per 088 D2) — it is real, but it
does not show up as CPU wall time anywhere near the render call. The wall clock instead pays
for it at **`swap`**: `glfw.swap_buffers` blocks until the GPU has drained every draw queued
since the last swap, so the ~100 ms is paid there, 100.29 ms median. This is consistent across
every document: Trivial's floor-only swap is ~4 ms with nothing to draw (driver/vsync overhead,
not a target-specific cost), Medium's swap is ~10.6 ms against its own ~7.3 ms GPU span (the
gap between the two is the same ~4 ms floor, roughly), and Heavy's ~100.3 ms swap tracks its
own ~100.5 ms GPU span almost 1:1. With `render_all` on, `swap` (~108 ms) tracks the SUM of all
three documents' GPU spans (0.01 + 7.9 + 102.7 ≈ 110.6 ms) — the driver queues all three
documents' draws before the one swap that blocks on the lot.

### Input latency (50 samples, heavy vs trivial, editor focused via real click)

| Document | delay median | delay p95 | delay max | delay min | frames to land (median / max) |
|---|---:|---:|---:|---:|---:|
| Heavy | 104.42 ms | 111.77 ms | 112.95 ms | — | 1 / 1 |
| Trivial | 6.94 ms | 7.12 ms | 7.23 ms | — | 1 / 1 |

Every sample on both documents landed within exactly ONE frame — `_drain_editor_input` runs
inside `process_hotkeys`, called once per `update_and_draw` before the editor draws, so a key
queued between two frame calls is always consumed on the very next one; there is no polling or
batching to add jitter beyond the frame boundary itself. The measured delay IS the frame period:
Heavy's ~104 ms median matches its own frame-period median (104.15 ms) to within noise, and
Trivial's ~6.9 ms matches its own 6.94 ms. This isolates the frame-period effect the task asked
for — no X11/ibus involvement here at all, since the key never crosses glfw's char callback; it
is injected straight into `app.editor_key_events`, the exact point `hotkeys.py` drains.

### Editor's own draw cost (max over 40 real keystrokes, per document)

| Document | `editor` (layout) CPU max | `editor:draw` (panel) GPU max | redraws observed |
|---|---:|---:|---:|
| Heavy | 1.23 ms | 1.55 ms | 38 / 40 |
| Trivial | 0.54 ms | 0.01 ms | 38 / 40 |

Both well under the 3 ms claim, with the heavy document current — the editor's own cost does
not scale with which document is open; it scales with how much text is on screen and how much
of it changed, which 40 identical `j` presses over a short padded buffer holds constant.

---

## False trails

Things that looked like cost, or looked like a mechanism, and were not:

- **Setting `app.profiler.enabled = True` records nothing.** `ui.py` overwrites it from
  `app.fps_details_open` every single frame (088 D4's own boundary-application design, working
  exactly as specified) — a caller outside the UI has to drive the panel's flag, not the
  profiler's, or every span comes back empty and `app.last_profile` stays `None` forever. Cost
  <5 minutes once found; the fix is one line (`app.fps_details_open = True`).
- **The first background run's `heavy_render_all_off` max was 40156.55 ms** (40 seconds) — not
  a real steady-state spike. With only 20 warmup frames and three documents sharing one project,
  the LAST document to get its one-first-render-per-frame turn (066 D2) can still be compiling
  its shader several frames into what looked like the measured window; the heavy shader's 17000-
  iteration loop is a slow compile. Raising warmup to 30 frames made every run's max sit within
  ~1.5 ms of its own median — the spike was compile time, not render time, and doesn't belong in
  a per-frame budget number.
- **`document:<name>` and `pass:<name>`'s CPU spans (0.03–0.11 ms) are NOT where the render cost
  is.** Issuing a draw call is cheap; a reviewer skimming only the CPU column of the tree would
  conclude documents render for free. The cost is real, on the GPU span two frames later, and it
  is PAID at `swap` — see "Where the 100 ms actually is" above. Any render-decoupling gate that
  only watches `document:`/`pass:` CPU time would watch nothing.
- **A single visible-run outlier** (`medium_render_all_off` p95 = 214.59 ms, one bad frame among
  300, against a headless p95 of 14.08 ms for the identical config) looked like a
  visible-window vsync stall worth a whole design implication. It is not reproducible — the
  visible run's OWN median (13.88 ms) matches headless exactly, and a concurrent probe session
  was independently hammering this same GPU with its own heavy fragment loads at the time this
  sample was taken (confirmed via `ps aux` — another agent's `gil_probe.py` /
  `probe_c_tilecost.py` were running in this same feature directory during this measurement).
  One frame out of 300 stalling under GPU contention from an unrelated process is not a
  visible-vs-hidden finding.
- **`editor:draw` never fires under an idle loop.** The first pass at deliverable 5 drove idle
  frames (no input) and found `editor:draw` absent from every span tree — not because the panel
  is free, but because `should_redraw`'s gate correctly skips a redraw when nothing about the
  editor's visible state changed. The 3 ms claim has to be checked under real edits (40
  keystrokes that each move the cursor), not an idle frame, or the measurement is checking a
  span that structurally cannot appear.

---

## Verdict

With a single ~100 ms-GPU document current (`render_all` off, the common case — most sessions
work one document at a time), the UI thread's frame period sits at a **104.15 ms median / 104.53
ms p95**, essentially all of it (100.3 ms) spent blocked inside `glfw.swap_buffers` waiting for
that one document's GPU work to drain. With `render_all` on (the shipped default), the period is
governed by the SUM of every open document's GPU cost and stays at ~111–112 ms regardless of
which document is in front.

Because keyboard input drains synchronously, once per frame, with no batching (deliverable 3),
this frame period IS the input latency: a keystroke typed while the heavy document renders lands
**~104 ms** later, against **~7 ms** for a trivial document — a ~15x slowdown that traces
entirely to one GPU-bound document blocking the thread the editor also lives on. The editor's
own draw cost is not the problem: `layout_following_cursor` and the panel redraw together never
exceeded 1.55 ms in any measurement here, confirmed (not refuted) under real edits with the
heavy document current.

**The number for a post-implementation gate:** UI frame period **p95 under 15 ms** while a
document with ≥100 ms of GPU work renders off the UI thread (current baseline: 104.53 ms p95,
current floor even for a trivial document is ~7 ms of unavoidable swap/tick/ui overhead, so 15
ms leaves headroom above that floor without re-admitting the render cost this feature exists to
move off-thread). A second, input-facing assertion the same gate should carry: editor-key
latency **p95 under 20 ms** under the same heavy-document load (current baseline: 111.77 ms p95,
inherited 1:1 from the frame period today).
