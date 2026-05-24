# 010 — Outlet-driven render rework (shared renderer + per-outlet concise render UX)

Status: **DONE (2026-05-24).** Renderer refinement landed (shared `RenderPreset` → native
target-size render); the Telegram share-tab UI was prototyped and screenshot-iterated to maintainer
approval over many rounds. Post-session cleanup triaged into feature 011. Follow-up cleanup +
deferred-with-trigger items: `ai_docs/features/011_ui_library_consolidation.md`.
Shape: high-blast-radius feature flow (`ai_docs/dev_flow.md ## Feature flow`) — changes the render
path (`core.py::render_media`), introduces a render-preset value type, refactors the share tab
layout + the Telegram render controls, and touches the `Exporter` contract.

---

## Goal

Kill the render/share duplication and the **silent-truncation lie**: today the share tab has its
own render sub-panel (FPS / Duration / Video sliders) whose values `TelegramExporter.prepare()`
silently overrides (re-fit to ≤512px / ≤3s / ≤30fps), so a 10s duration becomes 3s with no
feedback. Replace it with an **outlet-driven** model:

- **One shared low-level renderer**, refined so it renders the shader **natively at the target
  size/aspect** instead of post-scaling a fixed canvas — driven by a small GL-free **render preset**.
- **Per-outlet concise render UX** on the share tab: each sharing outlet exposes only the controls
  it actually needs (Telegram → essentially just **duration**), and owns its own operations panel
  (pack/grid/add). The generic full render controls stay on the standalone **Render** tab.
- The outlet's hard caps live in **one place** (the preset it declares); `prepare()` becomes
  **verify-not-truncate**.

The abstraction lives at the **renderer/preset layer** (shared), NOT the widget layer (per-outlet,
thin). This is a deliberate proper rework, not a minimal patch — agreed with the maintainer.

### Verified facts this spec rests on (read from code 2026-05-24, not assumed)

| Claim | Verdict | Evidence |
|---|---|---|
| `Node.render(u_time, canvas=...)` already renders into an arbitrary canvas | **Yes** | `core.py:189-190` (`canvas = canvas or self.canvas`) |
| `u_resolution` / `u_aspect` are derived from the *passed* canvas, not the node canvas | **Yes** | `core.py:256-262` (read `canvas.texture.size`) |
| `render_media` ignores that capability — renders into `self.canvas` then post-scales | **Yes** | `_render_image` PIL `.resize` (`core.py:300-303`); `_render_video` ffmpeg `-s WxH` (`core.py:368`) |
| Telegram `prepare()` distorts non-square sources | **No** — `scale=512:-2`/`-2:512`, the `-2` keeps aspect | `telegram.py:471` |
| Render always starts at `u_time=0` (no loop offset) | **Yes** — `_render_video` loops `i/fps` for `i in range(n_frames)` | `core.py:372-374` |
| `Canvas(size=(w,h))` constructs a sized FBO | **Yes** | `core.py:32-51` |
| A live moderngl context exists on the render thread when `render_media` runs | **Yes** | render-thread only; preset is GL-free so it crosses to the worker |

## Out of scope (each with a trigger)

- **Loop offset / trim start** ("which 3s of the loop"). Real gap (render is always `t=0..N`), but an
  *enhancement*, not a correctness bug — defer. *Trigger: first shader whose best window isn't at t=0,
  or a user asks to pick the slice.* Tracked in `todo.md`.
- **Aspect / fit controls for Telegram** (square / letterbox / crop). `prepare()` preserves source
  aspect correctly and Telegram letterboxes into its slot, so no control is needed *for Telegram*.
  The preset's `FitPolicy` machinery is built (the renderer needs it), but no UI exposes it yet.
  *Trigger: a fixed-aspect outlet (YouTube Shorts 9:16) lands, or a user wants a square sticker from a
  non-square shader.*
- **The preset growing beyond render parameters.** The preset carries render params ONLY (size/aspect/
  fit/duration/fps/format/byte-cap). Auth, metadata (title/description/visibility/alt-text), and
  multi-artifact (video + thumbnail) stay on the `Exporter`. *Trigger: a real second concrete outlet —
  validate the preset against its field list on paper before extending it (see Decision 6).*
- **Concrete YouTube / X outlets.** Still disabled stubs (`stubs.py`). This rework must not require
  them; it only must not *block* them. *Trigger: a real export to that platform.*
- **`MediaDetails` on-disk schema change.** The persisted `render_media_details` in node JSON
  (`ui_models.py:152`) must round-trip unchanged — no migration, no version bump from this rework.
  *Trigger: a render param genuinely cannot be expressed without a new MediaDetails field.*

## Design decisions

### Decision 1 — The preset is a GL-free value type in its own leaf module

`shaderbox/render_preset.py` (new leaf, no `App`/GL imports — same rationale as `paths.py` in
`conventions.md`). It carries render intent only:

```python
class ResolutionPolicy(StrEnum):
    FREE          # use the source/canvas size as-is (the Render tab default)
    LONGEST_EDGE  # keep source aspect, cap the longest edge (Telegram: 512)
    FIXED_ASPECT  # force aspect (a,b) + longest_edge -> derive dims (future Shorts 9:16)
    FIXED_DIMS    # exact target_w x target_h

class FitPolicy(StrEnum):
    RENDER_AT_TARGET   # shader rasterizes into the target-sized FBO (preferred, native)
    SCALE_DISTORT      # legacy post-scale (today's behaviour; the FREE default keeps it)
    # LETTERBOX / CROP — built only when a fixed-aspect outlet needs them (out of scope now)

class RenderPreset(BaseModel):   # pydantic, GL-free
    is_video: bool | None = None        # None = caller chooses
    fps: int | None = None              # None = free; set = clamp ceiling
    duration_max: float | None = None   # None = free; set = clamp ceiling
    container: str | None = None        # ".webm"/".mp4"; None = free
    resolution_policy: ResolutionPolicy = ResolutionPolicy.FREE
    longest_edge: int | None = None
    aspect: tuple[int, int] | None = None
    target_w: int | None = None
    target_h: int | None = None
    fit: FitPolicy = FitPolicy.SCALE_DISTORT
    max_bytes: int | None = None        # verify-only ceiling (prepare() asserts)
```

A pure helper `resolve_dims(preset, source_size) -> tuple[int, int]` lives beside it (alignment
rounding to `VIDEO_RESOLUTION_ALIGNMENT` happens here, once). GL-free ⇒ crosses the render/worker
boundary like `RenderedArtifact` (`base.py` thread-affinity docstring). Revisit the field set when
the second concrete outlet validates it (Decision 6).

### Decision 2 — `render_media` renders at the target size; the signature gains an optional preset

```python
def render_media(self, details: MediaDetails, preset: RenderPreset | None = None) -> MediaDetails
```

`preset=None` ⇒ byte-identical to today (FREE + SCALE_DISTORT), so both existing call sites
(`tabs/render.py:49`, the new share path) and the persisted schema are unaffected. For
`FitPolicy.RENDER_AT_TARGET`, `render_media` resolves `(w,h)` via `resolve_dims`, constructs a
**transient `Canvas(size=(w,h))`**, passes it to `self.render(u_time, canvas=target)`, and reads
frames from `target.texture` — no PIL `.resize`, no ffmpeg `-s`. The shader is resolution-
independent, so this produces the right pixels natively instead of stretching a canvas-aspect render.
This is what makes "no resolution slider, just the outlet preset" *correct*.

### Decision 3 — Outlet caps are single-source on the exporter; `prepare()` verifies, not truncates

`TelegramExporter` exposes `render_preset() -> RenderPreset` built from the (now-deleted) `_TG_*`
constants: `is_video=True, container=".webm", fps=30, duration_max=3.0,
resolution_policy=LONGEST_EDGE, longest_edge=512, max_bytes=256*1024, fit=RENDER_AT_TARGET`. The
widget, the renderer, and `prepare()` all read this one object. Because the renderer already produced
≤512/≤30fps/≤duration_max pixels from the same preset, `prepare()`'s job shrinks to the **format**
transcode GL can't do (RGBA→VP9) plus the one thing the renderer can't predict — the encoded byte
size — which it **asserts** (`> max_bytes` → `ExporterValueError`). The `-t`/`scale`/`fps` ffmpeg
args stay only as a defensive belt; the authoritative numbers come from the preset.

### Decision 4 — Per-outlet thin widget + a shared render *helper* (not a generic widget)

The render *controls* are bespoke per outlet (Telegram → one control: duration). The render *glue*
(scratch-path minting, the try/except, prior-artifact cleanup, the file-exists guard — today's
`_render_into_state` / `_make_artifact_path`, ~40 lines) is a **shared free helper**
`render_for(node, preset, scratch_dir) -> RenderedArtifact | None`, NOT duplicated per outlet and NOT
a polymorphic `Widget` (consistent with `conventions.md` "widgets are free functions, no protocol").
The bespoke-ness is the control surface only.

### Decision 5 — Layout: experiment-driven, accordion is the hypothesis

The share tab drops the vertical render-left / telegram-right split. The hypothesis (from the review
round) is a **vertical accordion of outlets** — each outlet a collapsing header, at most one expanded
(mirroring the "one popup open" invariant), the open one gets full height with its sticker grid in a
local `begin_child`; collapsed headers show a one-line status. Rationale: an outlet's content (pack
+ N-row grid + preview + actions) is intrinsically *tall*, which fights a short horizontal band.
**This is a hypothesis to validate by screenshot, not a locked decision** — the maintainer can't see
the glfw window, so the layout is built, screenshotted, and iterated (`[[no-screenshot-driven-dev]]`).
All sizing flows through `theme.py` `SIZE`/`SPACE` tokens (no magic px).

### Decision 6 — Guardrail: the preset is validated against N=2 on paper before it grows

The preset is designed from N=1 (Telegram). Before adding ANY field for a future outlet, write that
outlet's full field list and measure how much lands inside the preset vs outside (auth/metadata/
multi-artifact). If <30% lands inside, the preset is the wrong axis — keep the rework to the shared
`render_for` helper + the concise control set, and don't grow the preset. (`conventions.md` defers
exporter generalization to "the third concrete exporter"; this rework generalizes the *renderer*,
which is real cleanup, while explicitly NOT generalizing the *outlet abstraction* past Telegram.)

## Telegram control set (decided)

- **Duration:** float slider 0.5–3.0s, default 3.0 (NOT 1/2/3 buttons — integer snapping tears a
  loop whose natural period isn't an integer second).
- **Resolution / aspect / fps / quality:** none — derived from the preset; `prepare()` enforces.
- **Emoji:** kept, but it's a per-sticker property on the target panel, not a render control.
- **>256KB recovery:** the duration slider IS the escape hatch (shorter → smaller); the cap error
  message must say so.

## Files touched (anticipated)

- `shaderbox/render_preset.py` (new) — `RenderPreset`, `ResolutionPolicy`, `FitPolicy`, `resolve_dims`.
- `shaderbox/core.py` — `render_media` gains `preset`; `_render_image`/`_render_video` render into a
  transient target canvas under `RENDER_AT_TARGET`.
- `shaderbox/exporters/base.py` — `render_preset()` on the ABC (non-abstract default = FREE preset).
- `shaderbox/exporters/telegram.py` — declare `render_preset()`; `prepare()` → verify-not-truncate;
  delete `_TG_*` constants; concise render controls move into the outlet panel.
- `shaderbox/tabs/share.py` + `share_state.py` — drop the vertical split + `_draw_render_panel`'s
  lying sliders; per-outlet panel; `render_for` helper; per-outlet artifact slot.
- `scripts/smoke.py` — assert the new invariants (preset round-trips, render_media(preset=None)
  unchanged, accordion one-open).

## Test / verification plan

- **Blind-safe (headless):** `resolve_dims` unit cases (square/landscape/portrait → expected dims +
  alignment); `render_media(preset=None)` produces byte-identical output to pre-change; `render_for`
  artifact lifecycle (mint → exists → cleanup); accordion one-open invariant in smoke.
- **Screenshot (maintainer):** the share-tab layout — collapsed/expanded accordion, the concise
  Telegram render control, preview legibility, grid thumb size, status-line affordance.
- `make check` + `make smoke` green at every commit.

## Build order (experiment-driven)

1. `render_preset.py` + `resolve_dims` (pure, unit-tested) — no callers, check stays green.
2. `render_media(preset=None)` param + `RENDER_AT_TARGET` path; verify Render tab non-square output
   has no distortion (blind-safe correctness).
3. `Exporter.render_preset()` default + `TelegramExporter.render_preset()`; `prepare()` →
   verify-not-truncate; delete `_TG_*`.
4. `render_for` helper; share-tab per-outlet artifact slot.
5. Share-tab layout prototype (accordion hypothesis) → screenshot → iterate with maintainer.
6. `/sanitize`: roadmap row + banner, `conventions.md` (preset as a 2nd GL-free cross-boundary value;
   module map), todo loop-offset deferral.
