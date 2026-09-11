# 090 pre-implementation review — verification and blast radius

Reviewer role per `dev_flow.md ## Feature flow` step 4 (*verification & blast-radius*), against
`01_spec.md` at `7f7e67c`, the locked decisions in `02_throttle_and_resolution.md`, and the
verification contract in `dev_flow.md` step 7. Read-only on the tree; the one probe I ran lives in
the session scratchpad, not the repo.

**Verdict: FAIL** — three BLOCKERs. Two of them are the same defect: D5's headline guarantee
("export never reads the live size") is false on the `RenderShape.NATIVE` path, which is the
DEFAULT of every copilot render tool and of the YouTube exporter, and V5 tests the one path the
spec did fix. The third is a persisted-shape hole: three tracked `document.json` files outside the
spec's hand-edit list silently lose their size.

---

## 1. Guarantee-by-guarantee

Guarantees enumerated from *Design decisions* D1–D11 and *Frame integration*. "Consumer?" asks
whether the test reads the mechanism's READER or only its producer (step 7's unwired-is-absent
rule). "Headless?" asks whether it runs under `make test` (standalone context, no display) or
needs the `app` fixture / a display.

| # | Guarantee | Spec test | Consumer? | Falsifier named | Red under that bug ONLY? | Headless? | Verdict |
|---|---|---|---|---|---|---|---|
| G1 | `plan_render_set`'s eight worked rows | V1 | producer only (pure fn — correct here) | yes: `ceil`→`floor`; remainder-before-budget | yes | yes (pure) | OK |
| G2 | hysteresis holds `k` for 4 frames | V2 | producer | yes: apply candidate immediately | yes | yes | OK |
| G3 | D3 damping: dead-band + 8-frame stability | V3 | producer only | yes: drop dead-band / drop stability | yes | yes | **gap — see F5**: V3 drives `auto_canvas_size` plus "the damping", but D3 puts the damping in `_tick_frame_state` step 4, not in the pure function. Nothing tests that the LOOP applies it. |
| G4 | feedback survives a resize; no texture leak | V4 | consumer (`set_canvas_size`) | yes, three of them, incl. the far-corner pixel vs `copy_framebuffer` | yes — the corner-pixel design is the strongest check in the spec | yes (standalone ctx, `texture_to_rgba8`) | OK |
| G5 | a scaled feedback pass sizes from the LIVE canvas inside an off-size export | V6 | consumer | yes | yes | yes | OK |
| G6 | export never reads the live size | V5 | consumer (`render_media`) | yes, for `preset=None` | **NO — see F1.** V5 covers only the `preset is None or SCALE_DISTORT` branch. `RenderShape.NATIVE` lowers to `FREE` + `RENDER_AT_TARGET` (`render_shape.py:71-79`), which the spec leaves *untouched*, and `resolve_dims` under `FREE` returns `source_size` = `self.render_pass.canvas.texture.size` (`document.py:944-945`, `render_preset.py:68-69`) — the live Auto size. | yes | **BLOCKER** |
| G6b | the Render tab's W×H still decides the exported image | none | — | — | — | — | **BLOCKER — F2.** Not stated as a guarantee anywhere, and the spec silently revokes it. |
| G7 | the throttle is WIRED — the render set skips | V7 | consumer (drives `_tick_frame_state`, mutation applied at the LOOP) | yes, and explicitly the loop not the function | yes | **NO** — `_tick_frame_state` needs an `App`; that is the `app` fixture. Spec says "headless" without saying `app`-fixture. Minor (F8). | OK |
| G8 | throttle off restores today's set | V8 | consumer | yes | yes | `app` fixture | OK |
| G9 | GPU spans record with the panel closed | V9 | consumer (`update_and_draw`) | yes | yes | `app` fixture + display; **the named slot-holder is wrong (F7)** | OK-with-minor |
| G10 | the chip shows two numbers | V11 (prose budget only) | **producer only** — `test_ui_prose_budget.py` scores the STRING; nothing asserts `fps_overlay` is CALLED with a non-`None` `document_fps` when `k>1`. | none | no — passes whether or not `ui.py:771` passes the argument | — | **MAJOR — F6** (the exact "defined ≠ wired" shape step 7 names) |
| G11 | the panel's `document:` rows carry `k` and share | none | — | — | — | — | **MAJOR — F4.** And it is not merely untested: `profile_rows_plan` builds rows from span NAMES, which carry the document TITLE (`ui.py:325`, `f"document:{document_name}"`), while `RenderPlan` is keyed by document ID. The spec's own *Profiler and FPS surfaces* section flags this for the cost record and then re-introduces it for the rows. |
| G12 | the copilot's explicit size becomes Fixed | none | — | — | — | — | **MAJOR — F9.** `tests/test_document_ops.py:33-45` exercises `set_canvas_size` and asserts only the clamp; nothing would go red if `backend.py:1223` never wrote `fixed_size` or never flipped the mode. |
| G13 | hand-edited `projects/dev` documents load | V10 (round-trip) + smoke | consumer | yes | round-trip only — V10 saves and reloads a document THE TEST made. Nothing loads the tracked files. | — | **MINOR — F10**: `make smoke` seeds from `document_examples`, never `projects/dev`, so a forgotten sandbox edit is caught by nobody. |
| G14 | the persisted shape drops top-level `canvas_size` | V10 | consumer | yes | yes | `app` fixture or pure | OK |
| G15 | Auto sizes to the largest displaying region | V3 (the pure fn) | **producer only** | none for the recorders | no — nothing asserts `_draw_document_image` / `document_grid` / `pass_list` ever WRITE `displayed_sizes` | — | **MAJOR — F3.** Cut all three recorder lines and every named test still passes; `auto_canvas_size(None, …)` returns `previous`, so the document simply never resizes and no assertion notices. |
| G16 | the always-on query cost is negligible | *Pre-implementation measurements* | — | — | — | — | **MINOR — F11**: the probe the spec names does not exist. Measured below. |

**Guaranteed with no test at all:** G6b, G11, G12, G15 (recorders), plus D9a's smoother claim
("a re-opened panel shows converged numbers") and D8's "`begin_frame` is not called on a skipped
frame" — V7 asserts `_frame` advanced ~1 time in 12, which does cover the latter.

---

## 2. Blast radius — callers and readers *Files touched* does not mention

`Document.canvas_size` has 64 references in `shaderbox/` and `render_pass.canvas` has 23. The
spec's *Files touched* names 15 modules. These are the readers it omits whose behavior changes
when `canvas_size` stops being a stored setting and starts being the panel size.

| file:line | what it reads | what changes under Auto | severity |
|---|---|---|---|
| `shaderbox/render_shape.py:71-79` | `RenderShape.NATIVE` → `FREE` + `RENDER_AT_TARGET` | the spec says NATIVE "now means the document's export size" and in the same section says `RENDER_AT_TARGET` "is untouched". Mutually exclusive. | BLOCKER (F1) |
| `shaderbox/copilot/tools/publish.py:30,44,58` | `shape: RenderShape = Field(default=RenderShape.NATIVE)` | every copilot render/publish call with no explicit shape exports at the live panel size | BLOCKER (F1) |
| `shaderbox/exporters/youtube.py:99,162` | `shape: RenderShape = RenderShape.NATIVE` | same | BLOCKER (F1) |
| `shaderbox/exporters/youtube.py:517` | `_artifact_matches_shape` → `resolve_dims(preset, render_pass.canvas.texture.size)` | a **staleness gate**. Under Auto + NATIVE the expected size moves with the panel, so a rendered artifact stops matching the moment the window is resized and the publish button disarms permanently. | BLOCKER (F1) |
| `shaderbox/document.py:940-942` + `_render_image:818-823` / `_render_video:915-919` | `details.resolution_details` already drives a post-render resize / ffmpeg `-s` | today `preset=None` renders into the live canvas and then **resizes to the user's Render-tab W×H**. D5 replaces that with a scratch canvas at `export_size` and overwrites `resolution_details` — discarding the Render tab's number. | BLOCKER (F2) |
| `shaderbox/widgets/details.py:82-86` | `full_w, full_h = render_pass.canvas.texture.size`; the two preset buttons are LABELED with it | under Auto the Render tab offers a "764x430" preset that changes as the panel moves | MAJOR |
| `shaderbox/widgets/details.py:105` | `aspect = np.divide(*render_pass.canvas.texture.size)` | the W/H aspect-lock follows the panel, not the document | MAJOR |
| `shaderbox/tabs/render.py:34` | `tex = render_pass.canvas.texture` for the tab's preview | cosmetic; the preview follows the live size (acceptable, but unlisted) | MINOR |
| `shaderbox/copilot/backend.py:2150` | `cw, ch = render_pass.canvas.texture.size` — the probe's aspect for `render_facts` | the facts line's sampled aspect becomes a function of the panel geometry, so the same document yields different facts across frames. `_probe_frame` (`backend.py:149-164`) renders "at ITS OWN size" per its own comment — under Auto that is no longer a document property. | MAJOR |
| `shaderbox/copilot/backend.py:861` | `canvas=f"{document.canvas_size[0]}x{document.canvas_size[1]}"` in the working set | the spec DOES say "the working-set line reports the mode", so this is covered — but no test asserts it | MINOR |
| `shaderbox/ui.py:344-356` | the Examples popup renders `app.ui_document_examples` | **a whole render set with no `displayed_sizes` recorder.** `popups/examples.py` draws example previews, and the spec lists recorders only in `_draw_document_image`, `document_grid` and `pass_list`. Under Auto every example keeps `previous` forever — the loaded size. Also unthrottled by design (stated), so the Examples popup renders six full-size documents every frame with no budget. | MAJOR |
| `shaderbox/tabs/document.py:73-80` | `_canvas_presets` reads `current = document.canvas_size` and seeds `seen` with it | under Auto the preset list's dedup key is the panel size, so the list's contents shift as the panel moves. D5 says "the picker is unchanged" — it is unchanged in code and changed in behavior. | MAJOR |
| `shaderbox/popups/pass_settings.py:92,120,183,204` | `document.canvas_size` drives the pass-settings target preview sizes | a scaled pass's displayed target dims now follow the panel | MINOR |
| `shaderbox/project_session.py:915` | `add_pass` mints a `Pass(canvas_size=document.canvas_size)` | a pass added while the panel is small is born small; self-heals on the next render (`document.py:700-702`) | MINOR |
| `scripts/dogfood/harness.py:528-529, 568-569, 594-595, 642-646` | four `saved_size = document.canvas_size` / `set_canvas_size` save-restore pairs | under the spec every `set_canvas_size` now runs `resample_feedback` for every history — so each harness call does two resamples it did not do before. Functionally OK (the harness is headless, no `App`, so Auto never runs), but it is 8 extra blits per render and the spec's *Files touched* does not name `scripts/dogfood/`. | MINOR |
| `tests/test_canvas_presets.py:107,211-232`, `tests/test_document_graph.py:96,306,490-510,640-682`, `tests/test_pass_hot_reload.py:71,152`, `tests/test_uniform_row_pruning.py:141,169`, `tests/test_graph_persistence.py:73,343`, `tests/test_feedback_persistence.py:92` | construct documents from `json.dumps({"canvas_size": [...], ...})` | every one of these builds a document.json in the OLD shape. Under D1 the top-level key is no longer read, so each silently becomes `DEFAULT_CANVAS_SIZE` = **(64, 64)** (`constants.py:29`). `test_canvas_presets.py:215` asserts `canvas_size == (1280, 960)` and goes red; `test_pass_hot_reload.py`'s `[8, 8]` and `test_graph_persistence.py`'s `[16, 16]` silently become 64×64 and may still pass while testing something else. The spec's test-edit list names only `test_feedback_persistence`, `test_canvas_fields`, `test_render_for`, `test_profiling`, `test_persistence_completeness` — **six test files are missing from it.** | MAJOR |
| `tests/test_profiling.py:524-538` | `test_the_wire_and_the_abort_path`'s docstring names "drop `app.profiler.enabled = app.fps_details_open` and `last_profile` stays None" as a falsifier | D9a **deletes that line**. The existing test's stated falsifier becomes the new intended behavior; the docstring must be rewritten or the test documents a lie. The spec does list `test_profiling.py` as edited, but not this. | MAJOR |
| `shaderbox/ui.py:241`, `widgets/document_grid.py:51,87` | `is_render_all_documents` — 3 readers, all listed or unaffected | fine | — |
| `shaderbox/ui.py:142,229,775`, `popups/settings.py:80-82` | `global_target_fps` — 4 readers | `ui.py:142` and `:229` are the frame pacer and the script `dt`; neither changes. Fine. | — |

---

## 3. The persisted-shape change — every `document.json` in the repo

51 `document.json` files exist. 44 are under `scripts/dogfood/runs/`, which `.gitignore` covers
("Dogfood run artifacts … regenerable junk") — correctly out of scope. Of the seven that remain:

| file | tracked | top-level `canvas_size` | spec hand-edits it? |
|---|---|---|---|
| `shaderbox/resources/document_examples/53724dbd-…/document.json` | yes | `[1280, 960]` | yes ("the six shipped examples") |
| `…/73ea2431-…/document.json` | yes | `[1280, 1280]` | yes |
| `…/f90f5ff9-…/document.json` | yes | `[1280, 960]` | yes |
| `…/0b0d16bb-…/document.json` | yes | `[1080, 1920]` | yes |
| `…/8d454b7b-…/document.json` | yes | `[1600, 900]` | yes |
| `…/77a84d27-…/document.json` | yes | `[512, 512]` | yes |
| `projects/dev/documents/e7e00c46-…/document.json` | yes | `[1280, 960]` | yes (named explicitly) |
| `projects/dev/documents/ec926580-…/document.json` | yes | `[1280, 960]` | yes (named explicitly) |
| **`projects/documents/1901ab60-8d6f-4de0-b598-ca35ff5c3664/document.json`** | **yes** | `[960, 960]` | **NO** |
| **`projects/documents/307598da-de4f-4133-beaa-354e901d6a2b/document.json`** | **yes** | `[1280, 960]` | **NO** |
| **`tests/fixtures/bloom_chain/document.json`** | **yes** | `[960, 960]` | **NO** |

The example count and the `projects/dev` pair are right. Three tracked files are missed.

**What loading a forgotten one does — silent default, not a crash.** Traced:
`Document.load_from_dir` (`document.py:754-756`) reads `metadata.get("canvas_size")`; under D1 that
read is deleted, so the stale top-level key is simply never consulted. `_load_ui_state`
(`ui_models.py:543-565`) then prunes unknown `ui_state` keys with a `logger.warning` and fills
every MISSING key from its model default — so `resolution_mode` defaults to `AUTO`, `fixed_size`
to `DEFAULT_CANVAS_SIZE` = **(64, 64)**, `export_size` to `(1920, 1080)`. No `ValidationError`
(`UIDocumentState` deliberately does not set `extra="forbid"` — `tests/test_model_kwargs.py:88-108`
pins that), no crash, no user-visible error. A 960×960 fixture becomes a 64×64 document whose
aspect is wrong, which is exactly the "plausible picture, the wrong one" shape the spec's own D4
rationale warns about.

Consequences of the three misses:
- `tests/fixtures/bloom_chain` is loaded by `test_lazy_compile.py:31` and `test_default_wiring.py:40`.
  Both would run against a 64×64 bloom chain. Whether they go red depends on whether they assert
  sizes; either way they stop testing what they name.
- `projects/documents/` is the second tracked project dir (committed in `6e628e1` / `2e865b8`). It is
  not `projects/dev/`, so `CLAUDE.md`'s sandbox rule does not name it — but it IS tracked, so the
  no-migration rule's "fix the files by hand" applies to it identically.

---

## 4. The pre-implementation measurement — specified, absent, and now measured

**Specified well enough to run? Partly.** The spec names the file
(`ai_docs/features/090_render_decoupling/probes/…`), the invocation
(`xvfb-run -a uv run python scripts/probe_always_on_queries.py --frames 300` — note the two paths
disagree: the prose says `probes/`, the command line says `scripts/`), the method (300 frames off,
300 on, same document set, median and p95 of `update_and_draw` taken from outside) and where the
number goes. It does **not** state a pass threshold: "if it is not negligible against the 16.7 ms
budget" is not decidable. Give it a number.

**The probe does not exist.** `ls ai_docs/features/090_render_decoupling/probes/ | grep -i 'always_on\|query'` → empty;
`scripts/probe_always_on_queries.py` is absent (`scripts/` holds only `token_probe.py`).

**I wrote and ran it.** Real App, hidden glfw window on this box's display, all six shipped
examples seeded and `is_render_all_documents` on, 120 warm-up frames, then interleaved
off/on blocks so driver drift hits both arms equally.

```
run 1 (300 frames/arm, 6 documents)
  profiler OFF  median  7.396 ms   p95 13.949 ms   mean  9.624
  profiler ON   median  7.227 ms   p95 13.954 ms   mean  9.700
  DELTA         median -0.169 ms   p95 +0.005 ms   mean +0.076

run 2 (400 frames/arm, 6 documents)
  profiler OFF  median  7.311 ms   p95 13.954 ms   mean  9.609
  profiler ON   median  7.648 ms   p95 13.955 ms   mean  9.783
  DELTA         median +0.337 ms   p95 +0.000 ms   mean +0.174
```

**The median delta changes sign between runs (−0.169 ms, +0.337 ms), so the cost is inside the
run-to-run noise floor.** The stable number is the p95 delta: **+0.005 ms and +0.000 ms**, i.e.
under 0.03 % of the 16.7 ms frame. D9a's premise holds and D7's CPU-swap-wait fallback is not
needed. Record this in the spec's *Pre-implementation measurements* section, with the sign-flip
noted — a single run reporting `+0.337 ms` would read as a real 2 % cost and it is not one.

---

## Findings

### BLOCKER

**F1 — "export never reads the live size" is false on the `RenderShape.NATIVE` path, which is the default everywhere.**
*Evidence.* `render_shape.py:71-79`: `NATIVE`'s `ShapeSpec.aspect` is `None`, so `shape_to_preset`
returns `ResolutionPolicy.FREE` + `FitPolicy.RENDER_AT_TARGET`. `document.py:944-945`:
`resolve_dims(preset, self.render_pass.canvas.texture.size)`. `render_preset.py:68-69`: the `FREE`
fall-through is `w, h = src_w, src_h`. So `RENDER_AT_TARGET` + `FREE` = the live canvas size. The
spec's *Export* section says in one bullet that "`RenderShape.NATIVE` now means the document's
export size" and in the next that "`RENDER_AT_TARGET` is untouched"; leaving it untouched is
exactly what makes the first sentence false. `NATIVE` is the `Field(default=...)` of all three
copilot render tools (`copilot/tools/publish.py:30,44,58`) and the YouTube exporter's initial and
reset shape (`youtube.py:99,162`), and it drives `_artifact_matches_shape`'s staleness gate
(`youtube.py:511-519`) — which under Auto disarms publish whenever the panel moves.
*Fix.* Route the `FREE` source size through the document's export size rather than the live canvas:
in `render_media`, resolve the source as `self.export_size if self.resolution_mode is AUTO else
self.fixed_size` and pass THAT to `resolve_dims`, for both branches. Then V5 must be parametrized
over `preset=None`, `shape_to_preset(NATIVE)`, and one `FIXED_ASPECT` shape, asserting the written
image size in each — the falsifier being "pass `render_pass.canvas.texture.size` as the source and
the NATIVE case comes out at the live size". Add the YouTube staleness gate to the same check.

**F2 — D5 silently revokes the Render tab's chosen export resolution.**
*Evidence.* Today `preset=None` hands the live canvas to `_render_media_into`, and
`_render_image` (`document.py:818-823`) resizes the PIL image to `details.resolution_details`,
while `_render_video` (`document.py:915-919`) passes `-s WxH` to ffmpeg when the canvas differs.
Those dims come from `widgets/details.py::draw_resolution_details`, the Render tab's W/H drag-ints
plus its two preset buttons — a number the user set. D5 replaces the branch with a scratch canvas
at `export_size` and "writes the dims into `details.resolution_details` on a `model_copy(deep=True)`
as the `RENDER_AT_TARGET` branch does" — i.e. overwrites the user's number with the document's.
The spec never names this as a change; it reads as if `preset=None` currently exports at the live
size, which it does not.
*Fix.* Decide and state it. Either (a) the Render tab's `resolution_details` keeps winning and
`export_size` is only the FALLBACK when the tab has no number — then D5's export default rule
seeds `render_media_details.resolution_details` instead of overriding it; or (b) the tab's fields
are the `export_size` editor and the two concepts are merged, in which case `UIDocumentState`
already persists `render_media_details` and `export_size` is redundant — say so and drop the third
field. Whichever, add the guarantee "the Render tab's W×H is what lands on disk" to *Verification*
with the falsifier "overwrite `resolution_details` from `export_size` and a 640×480 tab setting
exports at 1920×1080". Note that `test_render_for.py::test_render_media_preset_none_byte_identical`
passes under either, so no existing gate catches this.

**F3 — the three `displayed_sizes` recorders have no consumer test; cutting all three is silent.**
*Evidence.* D2 names three writers (`ui.py::_draw_document_image`,
`widgets/document_grid.py::draw_document_preview_button`, `widgets/pass_list.py::_draw_pass_tile`)
and one reader (`auto_canvas_size`). V3 drives `auto_canvas_size` directly with a synthesized ramp.
Per D2, `None` — displayed nowhere — answers `previous`. So deleting every recorder leaves each
document at its loaded size forever, with no exception, no log and no failing assertion: V1/V2/V3
are pure-function tests, V4 calls `set_canvas_size` by hand, V7/V8 plant costs. This is precisely
step 7's named failure mode ("a spec'd safety … named, commented, even given config, but never
CONNECTED"), and the spec applies the mutate-the-wiring law to V7 while omitting it here.
*Fix.* A wiring test on the `app` fixture, riding the `update_and_draw` slot alongside V9:

```python
def test_the_viewer_records_the_size_it_displays_the_document_at(app: Any) -> None:
    # Falsifier: delete _draw_document_image's displayed_sizes write and the dict stays empty,
    # so every Auto document keeps its loaded size forever with nothing to notice it.
    document_id = app.current_document_id
    app.ui_documents[document_id].ui_state.resolution_mode = ResolutionMode.AUTO
    app.displayed_sizes.clear()
    update_and_draw(app)                      # one frame: the draw writes
    recorded = app.displayed_sizes.get(document_id)
    assert recorded is not None, "no surface recorded a displayed size for the current document"
    assert recorded[0] > 0 and recorded[1] > 0
    update_and_draw(app)                      # next frame: step 4 reads it
    assert app.ui_documents[document_id].document.canvas_size != _LOADED_SIZE
```

Add a sibling asserting `pass_list`'s OUTPUT tile and the grid cell each register their own id
when the viewer does not (drive `draw_document_preview_grid` in a headless imgui frame the way
`tests/test_canvas_fields.py` drives the Document tab).

### MAJOR

**F4 — the panel's plan columns cannot be keyed: `profile_rows_plan` sees titles, `RenderPlan` holds ids.**
*Evidence.* The render sites build `f"document:{document_name}"` from `ui_state.ui_name`
(`ui.py:325,340,355,362`); `_plan_tree` (`ui_primitives.py:1398-1405`) walks `span.children` and
carries only `child.name`. `RenderPlan.intervals` is `dict[str, int]` keyed by document id. The
spec's own *Profiler and FPS surfaces* section identifies this exact hazard for the cost record
("a `document:` span carries the document's TITLE rather than its id — 088's post-implementation
note") and then specifies `profile_rows_plan(profile, fps, target_fps, plan)` matching rows to plan
entries with no stated key. Two documents may share a title.
*Fix.* Pass the panel a `dict[str, int]` keyed by the same TITLE the span carries, built at the
render site where both are in hand — or stamp the id into the span name and strip it for display.
State the choice in D9c and give V11 a case with two same-titled documents.

**F5 — D3's damping lives in the frame loop and only the pure function is tested.**
*Evidence.* D3 places `AutoSizeState` on `App` and *Frame integration* step 4 puts the
dead-band/stability decision in `_tick_frame_state`. V3 "drives `auto_canvas_size` plus the damping
through a 60-frame drag ramp" — but if the damping is loop code, V3 can only reach it by driving
the loop, and the spec says the plan function's half "runs with no GL".
*Fix.* Either move the damping into `render_plan.py` as a pure
`apply_damping(state, requested, current) -> tuple[int,int] | None` (then V3 is honest and pure),
or keep it in the loop and add a second falsifier at the loop ("remove the dead-band from step 4
and a 60-frame ramp calls `set_canvas_size` ~60 times" — count the calls with a monkeypatched
`set_canvas_size`). The spec should say which.

**F6 — the two-number chip has no wiring test.**
*Evidence.* V11 lists `tests/test_ui_prose_budget.py` "over … the two-number chip". That gate
scores authored STRINGS (`test_ui_prose_budget.py:235` is a span-name entry), not call arguments.
Nothing asserts `ui.py:771`'s `fps_overlay(...)` receives `document_fps=<k-derived int>` when the
current document's `k > 1`.
*Fix.* Monkeypatch `ui.fps_overlay` to capture kwargs — `tests/test_profiling.py:508-522` already
has exactly this spy (`_capture_the_overlays_profile`) — plant a 100 ms `CostRecord` on the current
document, drive one frame, assert the captured `document_fps` is not `None` and equals
`global_target_fps // k`. Falsifier: pass `document_fps=None` unconditionally and the label is
today's single number while every prose gate still passes.

**F7 — V9's "this rides the existing holder of that slot" names a slot that four files already hold.**
*Evidence.* `conventions.md:1037-1045` says only one test per process may drive
`ui.update_and_draw` and names `tests/test_code_panel.py::test_ctrl_tab_focuses_…`. In fact
`test_code_panel.py:107`, `test_project_management.py:491` and `test_profiling.py` (nine call sites)
all drive it. They coexist via xdist process isolation: `test_code_panel.py:19` and
`test_project_management.py:36` carry `pytestmark = pytest.mark.xdist_group(...)`, and `make test`
runs `-n 8 --dist loadgroup`. **`test_profiling.py` carries no `xdist_group` mark** — it survives
on scheduling luck.
*Fix.* V9 should say which file it lands in and give that file an `xdist_group` of its own; and the
quirk's text in `conventions.md` should be corrected to describe the xdist-group mechanism rather
than naming one stale slot-holder. This is a doc fix the spec can carry (it already edits
`conventions.md`).

**F8 — six test files construct old-shape `document.json` and are not in the edit list.**
*Evidence.* `tests/test_canvas_presets.py:107,229`; `tests/test_document_graph.py` (via
`Document(gl=gl, canvas_size=size)` — that constructor keeps its meaning, so those are fine, but
`:107`-style JSON writes are not); `tests/test_pass_hot_reload.py:71,152` (`[8, 8]`);
`tests/test_uniform_row_pruning.py:141,169` (`[64, 64]`); `tests/test_graph_persistence.py:73,343`
(`[16, 16]`); `tests/test_feedback_persistence.py:92` (listed — good).
`tests/test_canvas_presets.py:215` asserts `canvas_size == (1280, 960)` after loading a
`{"canvas_size": [1280, 960]}` document; under D1 that read is gone and the assertion goes red.
`test_canvas_presets.py:221-232` ("a malformed canvas_size falls back to the default") tests a
loader path D1 deletes outright.
*Fix.* Add all five to *Files touched* and say for each whether it moves to `ui_state.fixed_size`
or is deleted with the path it tested. `_as_canvas_size` (`document.py:71-89`) still guards the
`Document.__init__` parameter, so the coercion tests can move rather than die.

**F9 — nothing verifies the copilot's explicit size flips the mode to Fixed.**
*Evidence.* `tests/test_document_ops.py:33-45` asserts only the clamped dims returned by
`backend.set_canvas_size`. If `backend.py:1223-1243` writes `fixed_size` but never sets
`resolution_mode = FIXED`, the very next frame's Auto step overwrites the size the copilot just
set — the tool silently does nothing — and every test still passes.
*Fix.* Extend `test_set_canvas_size_applies_and_clamps` with
`assert ui_state.resolution_mode is ResolutionMode.FIXED` and
`assert ui_state.fixed_size == (128, 200)`, then drive one `_tick_frame_state` and assert the size
survived. Falsifier: leave the mode at Auto and the second assertion fails on the next frame.

**F10 — three tracked `document.json` files outside the hand-edit list; a forgotten one defaults to 64×64 in silence.**
Full inventory and the traced load behavior in §3.
*Fix.* Add `projects/documents/1901ab60-…`, `projects/documents/307598da-…` and
`tests/fixtures/bloom_chain/` to *The hand-edit*, each with its own aspect's export default
(960×960 → the largest square/`MENU_SHAPES` entry at 1:1; 1280×960 → 1920×1440 as the spec already
computes). And add the gate that makes the class non-recurring, in the same wave per the
file-the-lesson rule: a test that walks every tracked `document.json` and asserts none carries a
top-level `canvas_size` and each carries the three `ui_state` keys —

```python
_ROOTS = [REPO / "shaderbox/resources/document_examples",
          REPO / "projects/dev/documents", REPO / "projects/documents",
          REPO / "tests/fixtures"]

def test_every_tracked_document_json_is_in_the_new_shape() -> None:
    # Falsifier: leave one file's top-level "canvas_size" in place and this names that file.
    # Without it a forgotten document loads at DEFAULT_CANVAS_SIZE (64, 64) in silence —
    # _load_ui_state fills missing keys from defaults and never raises.
    for root in _ROOTS:
        for path in root.rglob("document.json"):
            meta = json.loads(path.read_text())
            assert "canvas_size" not in meta, f"{path} still carries the retired top-level key"
            state = meta.get("ui_state", {})
            for key in ("resolution_mode", "fixed_size", "export_size"):
                assert key in state, f"{path} is missing ui_state.{key}"
```

**F11 — the Examples popup is a render set with no size recorder and no budget.**
*Evidence.* `ui.py:344-356` renders every `app.ui_document_examples` entry whose first render is
done, under its own `document:` span. `popups/examples.py` displays them through `preview_cell`.
The spec's recorder list omits the examples grid, and *Frame integration* explicitly exempts the
Examples popup from throttling ("a popup's render set is one document the user is staring at" —
but it is six, not one).
*Fix.* Either add the examples grid as a fourth recorder and admit the popup's set to the plan, or
state in D2/D10 that example documents are always Fixed at their shipped size and never Auto — and
say which in one sentence, because today's spec implies Auto (the default) applies to them and then
never sizes them.

### MINOR

**F12 — the probe's own path is stated two ways.** Prose says
`ai_docs/features/090_render_decoupling/probes/`; the command line says
`scripts/probe_always_on_queries.py`. Pick one (the `probes/` dir, matching the `cost_*` /
`throttle_*` siblings).

**F13 — the always-on measurement has no pass threshold.** "Not negligible against the 16.7 ms
budget" is not decidable. Propose: **p95 delta ≤ 0.1 ms** (0.6 % of the frame), which the measured
+0.005 ms clears by twenty-fold.

**F14 — `resample_feedback`'s size argument is unspecified for a non-output pass.**
`set_canvas_size` (`document.py:352-361`) resizes ONLY `self.render_pass.canvas`; every other pass
resizes lazily inside `render` (`document.py:699-702`). D4 says `resample_feedback` is called "from
`set_canvas_size`'s funnel for every pass holding a history" without saying what `size` a non-output
pass gets. If it is `canvas_size`, a `scale=0.5` feedback pass's history becomes full-size while its
live canvas is still half-size — and `_swap_feedback` (`document.py:392-404`) then EXCHANGES them,
putting a full-size canvas into `render_pass.canvas`. Silent wrong picture. The `_feedback_canvas`
route is safe (it resamples to `live.texture.size`, `document.py:561-562`), so the fix is to make
the funnel use the same rule: `entry.target.target_size(new_size)` per pass, or simply not resample
non-output passes from the funnel at all and let `_feedback_canvas` handle them on their next
render. State it in D4 and let V4 cover a scaled feedback pass, not only the output one.

**F15 — D9a's smoother claim is asserted nowhere.** "a re-opened panel shows converged numbers
instead of climbing from cold" — `ui.py:276-279` feeds the smoother only while `app.profiler.enabled`,
which becomes permanently true, so the claim follows. But `tests/test_profiling.py:607` currently
asserts the smoother "stays empty while the panel is open"-adjacent behavior; check that file's
smoother tests against the new always-on state when editing it.

---

## False trails

Things that look like findings and are not — checked and cleared, so the next reviewer does not
re-spend the time:

- **`gpu_total` over a CPU span.** D7 reads `gpu_ms` from `profiling.gpu_total(span)` where the
  `document:` span is a `profiler.cpu(...)` span. That is correct: `gpu_total`
  (`profiling.py:91-96`) sums the whole subtree, and the `pass:` children ARE GPU spans
  (`document.py:713`). No finding.
- **`UIAppState`'s `extra="forbid"` and D11's two new fields.** `ui_models.py:261` does set
  `extra="forbid"`, and the spec's claim that two defaulted fields reject nothing is right —
  `model_salvage.drop_unknown` runs before construction (`model_salvage.py:115`). No finding.
- **`UIDocumentState` raising on the three new keys.** It does not set `extra="forbid"`
  (deliberately, pinned by `tests/test_model_kwargs.py:88-108`), and `_load_ui_state` prunes
  unknown keys and salvages per key. The new fields load cleanly from old files — the problem is
  the SILENCE of the default, not a raise. Folded into F10.
- **`create_document_from_example` losing the new fields.** `app.py:2061-2070` loads the example
  through `load_document_from_dir` and re-saves; `ui_state` carries through intact. Fine, provided
  the examples are edited (they are listed).
- **`Canvas.set_size` leaking on the resample path.** `core.py:158-164` releases before
  re-initializing, which is the ordering D4 forbids — but D4's `resample_feedback` mints a NEW
  `Canvas` and releases the old last, never calling `set_size`. The order is right as specified.
- **The dogfood harness breaking under Auto.** `scripts/dogfood/harness.py` drives `ProjectSession`
  with no `App`, so `displayed_sizes` is never written and Auto never runs there. Only the extra
  resample cost applies (listed as MINOR in §2).
- **`clamp_canvas_size` being bypassed under Auto.** D1 says every write funnels through it,
  including Auto, and `MIN_CANVAS_PX=16` / `MAX_CANVAS_PX=4096` (`pass_graph.py:58-59`) bound a
  one-pixel panel. Correctly specified.
- **44 dogfood-run `document.json` files.** `.gitignore` covers `scripts/dogfood/runs/`; they are
  regenerable. Out of scope, correctly.

---

## Coverage statement

**Read end-to-end:** `01_spec.md` (554 lines), `02_throttle_and_resolution.md` (206),
`dev_flow.md ## Feature flow` steps 1–9, `Makefile`, `tests/conftest.py`, `scripts/smoke.py`
(header + skip path), `shaderbox/render_shape.py`, `shaderbox/render_preset.py::resolve_dims`.

**Read in the relevant part:** `shaderbox/document.py` (`__init__`, `set_canvas_size`,
`begin_frame`, `_swap_feedback`, `_seed_feedback`, `_feedback_canvas`, `render`'s pass loop,
`load_from_dir`, `_render_image`, `_render_video`, `render_media`, `_as_canvas_size`,
`_load_document_metadata`), `shaderbox/ui.py` (`_tick_frame_state`, `_update_and_draw`'s render
block, `_draw_document_image`, the fps-overlay site), `shaderbox/ui_models.py` (`UIDocumentState`,
`UIAppState`, `UIDocument.save`, `_load_ui_state`, `load_document_from_dir`),
`shaderbox/ui_primitives.py` (`ProfileRow`, `profile_rows_plan`, `_plan_tree`, `fps_overlay`,
`preview_cell`), `shaderbox/profiling.py` (`Profiler.__init__`, `gpu_total`, `headline_ms`),
`shaderbox/core.py::Canvas`, `shaderbox/tabs/document.py` (`_apply_canvas_size`, `_canvas_presets`),
`shaderbox/tabs/render.py`, `shaderbox/widgets/details.py`, `shaderbox/widgets/document_grid.py`,
`shaderbox/widgets/pass_list.py`, `shaderbox/copilot/backend.py` (`_probe_frame`,
`set_canvas_size`, the working-set line, `render_image`, `render_video`),
`shaderbox/exporters/youtube.py` (`_artifact_matches_shape`, the shape defaults),
`shaderbox/app.py` (`Profiler(enabled=False)`, `create_document_from_example`),
`shaderbox/project_session.py::add_pass`, `shaderbox/pass_graph.py` (clamp + `target_size`),
`ai_docs/conventions.md ## Known quirks` (the `update_and_draw` slot, the `texture_to_rgba8` rule),
`tests/test_render_for.py`, `tests/test_profiling.py` (the wire test), `tests/test_canvas_fields.py`,
`tests/test_model_kwargs.py`, `tests/test_persistence_completeness.py`.

**Deliberately not read:** `00_research.md` and the twelve `research/*.md` reports — the spec's
measurements are cited, not re-derived, and re-measuring the throttle numbers is not this role.
The copilot prompt/tool-schema layer beyond the resolution-touching tools. The editor, scripting
engine, telegram exporter internals and shader-lib — no `canvas_size` surface in them (verified by
the grep counts above, 64 and 23 references enumerated in full).

**Ran:** the 51-file `document.json` inventory with a tracked/untracked split; `canvas_size`,
`set_canvas_size`, `render_pass.canvas`, `profiler.enabled`, `is_render_all_documents`,
`global_target_fps`, `document:`, `resolution_details`, `shape_to_preset`, `preview_cell` and
`update_and_draw` greps across `shaderbox/`, `tests/`, `scripts/`; and the always-on GPU-query
probe twice (300 and 400 frames per arm) on this box's real display.

**Verdict: FAIL** — F1, F2 and F3 are blocking. F1 and F2 are the same root: D5 was specified
against one export branch and the app has three, with the untouched one carrying every default.
F3 is the spec's own unwired-mechanism law applied to V7 and not to D2.

---

# Round 2

Re-review of the revised `01_spec.md` (688 lines) against the code. Same rules: read-only, nothing
asserted that was not run. The drafter accepted all fifteen of my round-1 findings and recorded the
triage in *Review history*; the fixes are substantive, not cosmetic, and several are better than
what I proposed (the `export_size` field removed entirely rather than patched; `apply_damping`
lifted into the pure module).

**Verdict: PASS-WITH-MINORS** — every round-1 finding is closed. Three new findings, all narrow and
all fixable in the spec text: one REGRESSION the revision introduced (R1, the "fixed" examples),
one test-design defect that would make four verification items unreachable as written (R2,
`frame_idx`), and one gap in the new D10 (R3, the Examples popup has no `begin_frame` or render
gate). None is a redesign; R1 and R2 are one-line corrections.

## Round-1 closure table

| id | round-1 finding | the passage that closes it | status |
|---|---|---|---|
| **F1 (B)** | `NATIVE` / the `FREE` fall-through read the live size; default of all three copilot tools, YouTube's shape and `_artifact_matches_shape` | D5: "`render_media` passes `self.resolution` to `resolve_dims` on BOTH branches, and `_artifact_matches_shape` resolves against the same number." *Export* repeats it. Verified: `resolve_dims` has exactly two live-size call sites, `document.py:944` and `youtube.py:515`, and the spec names both; `tabs/document.py:94`'s third is covered by "`_canvas_presets` reads `resolution`". V5 is parametrized over `preset=None`, `NATIVE` and one `FIXED_ASPECT`, plus a `_artifact_matches_shape` case. **Bonus:** the same one-line fix also covers Telegram, whose `LONGEST_EDGE` preset (`telegram.py:290-299`) reads `src_w`/`src_h` at `render_preset.py:61-66` — a path neither report named. | **CLOSED** |
| **F2 (B)** | D5 silently revoked the Render tab's W × H | D5: "**The Render tab's `resolution_details` W × H is preserved exactly as today**… `render_media` does NOT overwrite it." *Export*: "`resolution_details` is untouched." V5a asserts 640×480 lands while `resolution` is 1920×1080, and carries my note that `test_render_media_preset_none_byte_identical` passes either way. | **CLOSED** |
| **F3 (B)** | the three recorders had no consumer test | V9, adopting my sketch plus grid and Examples siblings, with the falsifier "delete any recorder's write → its case goes red with the dict empty" and my rationale quoted ("`auto_canvas_size(None, …)` returns `previous`"). | **CLOSED** |
| **F4 (M)** | panel rows by title, plan by id | D7: render sites open `f"document:{document_id}"`; `profile_rows_plan` takes an `id -> title` map. V10a: two same-titled documents, each row carries its own `k`. Verified the reader surface: only `test_profiling.py:549,575,599` match on `startswith("document:")` (prefix-only, unaffected) and `:849,856` use a synthetic `document:one` through the 3-arg call that still defaults. | **CLOSED** |
| **F5 (M)** | damping in loop code, only the pure half tested | D3 moves it into `render_plan.py` as `apply_damping(...) -> tuple[int,int] \| None`; V3 pure, V3a at the loop with a call-counting monkeypatch. | **CLOSED** (but see R2 — V3a's loop driver has the `frame_idx` problem) |
| **F6 (M)** | the chip had no wiring test | V10: monkeypatch `ui.fps_overlay`, assert `document_fps == global_target_fps // k`; falsifier "pass `document_fps=None` unconditionally — every prose gate still passes". | **CLOSED** |
| **F7 (M)** | the `update_and_draw` quirk names a stale slot-holder; `test_profiling.py` has no mark | *Files touched*: `conventions.md` corrected to "the mechanism is xdist GROUPS… three files drive it today and `test_profiling.py` carries no mark", and *Verification*'s preamble: "Tests driving `update_and_draw` get their own `pytest.mark.xdist_group`." | **CLOSED** |
| **F8 (M)** | six test files build the old JSON shape | All six named in *Data model changes* and *Files touched*, with the `(64, 64)` consequence and the disposition of `test_canvas_presets.py`'s two doomed cases (the assertion moves; the malformed case moves to `_as_canvas_size`, which does still guard `Document.__init__` — verified at `document.py:71-89`, `274-281`). | **CLOSED** |
| **F9 (M)** | nothing verified the copilot's Fixed switch | V13 extends `test_document_ops.py::test_set_canvas_size_applies_and_clamps` with the mode and the survive-one-tick assertion. | **CLOSED** (R2 applies to its second half) |
| **F10 (M)** | three tracked `document.json` missed | **Re-ran the inventory.** `git ls-files '*document.json'` returns exactly 11 files; the spec's table lists exactly those 11, byte-for-byte the same paths and the same current sizes (`[1080,1920]`, `[1280,960]`, `[1280,1280]`, `[512,512]`, `[1600,900]`, `[1280,960]`, `[1280,960]`×2, `[960,960]`, `[1280,960]`, `[960,960]`). **None missing, none spurious.** V12's four walk roots cover all eleven with no gap (checked by classifying every `git ls-files` entry). | **CLOSED** — see R1 for the mode column and m1 for the walk root |
| **F11 (M)** | the Examples popup: six documents, no recorder, no budget | D10: "**The Examples popup's six documents join the plan** as displayed documents with their own recorders"; *Frame integration* step 2 adds its set; `popups/examples.py` is in *Files touched* and V9 has an Examples sibling. | **CLOSED in intent, INCOMPLETE in mechanism — R3** |
| **F12 (m)** | the probe path stated two ways | *Pre-implementation measurements*: "The probe belongs under `probes/`…". | **CLOSED** |
| **F13 (m)** | no pass threshold | "**Pass threshold: p95 delta ≤ 0.1 ms**, cleared twenty-fold." My two runs are quoted with the p95 figures (+0.005, +0.000 ms) and — correctly — the sign-flip caveat on the median, so nobody later reads +0.337 ms as a real 2 % cost. | **CLOSED** |
| **F14 (m)** | `resample_feedback`'s size for a non-output pass | D4 and the funnel: "`target` = `self.canvas_size` for the output pass, `entry.target.target_size(self.canvas_size)` otherwise", with my `_swap_feedback` rationale quoted. V4 covers a `scale=0.5` pass and names the falsifier. | **CLOSED** |
| **F15 (m)** | the smoother claim asserted nowhere | *Profiler and FPS surfaces*: "`test_profiling.py`'s wire test names that deleted line as its falsifier — its docstring is rewritten to the new one, and its smoother tests re-checked against always-on." | **CLOSED** |

**Nothing REGRESSED from round 1's list.** The one regression (R1) is in new material.

## Fresh pass over *Verification* (18 items)

| item | consumer or producer | falsifier named | red for one reason only | headless or fixture |
|---|---|---|---|---|
| V1 plan's nine examples | producer (pure fn — right) | 3, incl. "drop `MAX_INTERVAL` → case 9 returns 143" | yes | headless, pure |
| V2 hysteresis | producer | yes | yes | headless, pure |
| V2a phase offsets | producer | "drop the phase → all three share every render frame" | yes | headless, pure |
| V3 damping, pure | producer (correct — D3 made it pure) | 2 | yes | headless, pure |
| V3a the loop applies damping | **consumer** | "call `set_canvas_size` with the raw request → ~60 calls" | yes | `app` fixture — **R2** |
| V4 texture count + no blank canvas | **consumer** (`set_canvas_size`) | 5, incl. the far-corner-vs-mean distinction and the scaled-pass dims | yes; the far-corner design is still the sharpest check here | headless standalone ctx |
| V5 export on three paths | **consumer** | "pass `render_pass.canvas.texture.size` as the source → the NATIVE case comes out 320×180" | yes | headless standalone ctx |
| V5a Render tab's W × H | **consumer** | "overwrite `resolution_details` from `resolution` → exports 1920×1080" | yes | headless standalone ctx |
| V6 scaled feedback in an off-size export | **consumer** | yes | yes | headless standalone ctx |
| V7 the throttle is wired | **consumer**, mutation at the LOOP (explicitly) | yes | yes | `app` fixture — **R2** |
| V8 throttle off | **consumer** | yes | yes | `app` fixture — **R2** |
| V9 the recorders are wired | **consumer** | "delete any recorder's write → its case goes red with the dict empty" | yes, per recorder | `app` fixture + headless imgui frames |
| V10 the chip is wired | **consumer** (kwarg spy) | yes | yes | `app` fixture |
| V10a rows match by id | **consumer** | "match by title → one takes the other's numbers" | yes | headless (pure `profile_rows_plan`) |
| V11 spans record with the panel closed | **consumer** | "restore `app.profiler.enabled = app.fps_details_open`" | yes | `app` fixture + display |
| V12 every tracked `document.json` | **consumer** (the files themselves) | "leave one file's old key → the test names it" | yes | headless, pure file walk — **m1** |
| V13 the copilot's size sticks | **consumer** | "leave the mode Auto → the next frame overwrites it… today's assertions all pass" | yes | `app` fixture — **R2** on the second half |
| V14 docs and model gates | producer | — (gate-of-gates) | n/a | headless |

Producer-only items are all pure-function tests where the function IS the deliverable; every
behavioral guarantee now has a consumer test. That was round 1's structural complaint and it is
answered.

## New findings

### R1 — REGRESSION (MAJOR): "the JFA and radiance-cascade **examples**" are two passes of ONE document, and the table marks all eleven `auto`

*Evidence.* *Data model changes* says: "Each keeps its current numbers as `resolution`; mode is
`auto` except the two pixel-dependent examples", the table then lists all eleven rows with
`mode = auto`, and the prose below says "The JFA and radiance-cascade examples among the six are
the `fixed` ones … the implementer identifies them by shader content." The spec's own cited source
says the opposite. `research/resolution_flow.md:170-171`:

> **Count: 2 pixel-dependent document-shader **files** (JFA + cascade, **both in ONE example
> document**, `77a84d27...`, the bloom-chain-style multi-pass example), 0 pixel-dependent in the
> shipped [library]

Confirmed against the tree: `shaderbox/resources/document_examples/77a84d27-…/passes/` holds
`paint / seed / jfa / df / cascade / composite` — six passes of one document (`constants.py:20`
names it "Radiance Cascades (iterated passes)"). A grep for pixel-dependent shader content across
all six example dirs returns exactly one directory, `77a84d27`. So:
- "the two pixel-dependent examples" is one example;
- "identifies them by shader content" sends the implementer to find a second document that does not
  exist, and the most likely wrong answer is to also mark `73ea2431` (Media Input) or the bloom
  fixture, neither of which is pixel-dependent;
- every row in the table says `auto`, so the table and the prose contradict each other and the
  table is what an implementer copies.

*Fix.* One row and one sentence: set `77a84d27-…`'s mode column to **`fixed`**, leave the other ten
`auto`, and replace the prose with "the radiance-cascade example (`77a84d27`) is the one `fixed`
row — its JFA and cascade passes are the repo's only pixel-dependent document shaders
(`resolution_flow §4`)". Then V12 can assert the count: exactly one tracked document is `fixed`,
which makes the fact self-checking rather than a sentence someone has to re-derive.

### R2 — MAJOR: `app.frame_idx` is incremented outside `_tick_frame_state`, so V7, V8, V3a and V13's second half are unreachable as written

*Evidence.* Four items drive `_tick_frame_state` in a loop and assert on per-frame behavior:

- V7: "Drive `_tick_frame_state` … across 12 frames its `_frame` advanced ~1 time, not 12."
- V8: "Same rig, `is_throttle_documents = False`" (every document advances every frame).
- V3a: "Drive `_tick_frame_state` through the same ramp with a monkeypatched `set_canvas_size`."
- V13: "then drive one `_tick_frame_state` and assert the size survived."

But `app.frame_idx += 1` is the **last statement of `_update_and_draw`**, at `shaderbox/ui.py:539`
— not in `_tick_frame_state`, whose body ends at `ui.py:267` with `return tick_documents`. The only
other writer is `app.py:1237`'s `self.frame_idx = 0`. So a loop calling `_tick_frame_state(app)`
twelve times runs every iteration at the SAME `frame_idx`, and step 8's gate
`(app.frame_idx + phase[id]) % k != 0` takes the same branch all twelve times: a `k = 12` document
either renders 12 times (if the frozen index happens to satisfy the gate) or 0 times. "Advanced ~1
time, not 12" is reachable under neither. V8's "advances every frame" passes vacuously for the same
reason — it would pass even if the throttle were wired wrong, which is exactly the theater step 7
forbids. V3a's ramp likewise never advances, so the 8-frame stability clause can never fire.

This is not hypothetical: `tests/test_profiling.py:441` monkeypatches `_tick_frame_state` precisely
because driving it standalone is awkward, and no test in the suite calls it directly today (grep:
only `test_project_management.py:476,605-607` mention it in prose).

*Fix.* State it in the *Verification* preamble, one sentence: **a test driving `_tick_frame_state`
directly must advance `app.frame_idx` itself each iteration**, because the increment lives in
`_update_and_draw`. Then V7's rig is:

```python
def _drive(app, n):
    # frame_idx is bumped at the END of _update_and_draw (ui.py:539), not inside
    # _tick_frame_state — a direct driver must advance it or every iteration re-runs
    # the same frame and the phase gate never moves.
    for _ in range(n):
        _tick_frame_state(app)
        app.frame_idx += 1
```

Alternatively move the increment into `_tick_frame_state` — but that is a behavior change to the
frame loop for a test's convenience, and `frame_idx` is also read by `session.tick` at `ui.py:260`
and by `begin_frame` at `:265`, both inside `_tick_frame_state`, so moving it would change what
those two see. Keep the loop as it is and fix the test rig.

### R3 — MAJOR: D10 admits the Examples popup to the plan, but its render path has neither a `begin_frame` nor a throttle gate

*Evidence.* D10 now says "**The Examples popup's six documents join the plan** as displayed
documents with their own recorders, thumbnails scheduled by the remainder rule", and *Frame
integration* step 2 says the render set is "`tick_documents`, unchanged, plus the Examples popup's
set when open". Two mechanical problems:

1. **"Plus" is wrong — the two sets are mutually exclusive.** `ui.py:319-356`:
   `if not app.any_popup_open(): for document_id in tick_documents: …` / `elif app.popup_state ==
   PopupState.EXAMPLES: … for ui_document in app.ui_document_examples.values(): …`. When the popup
   is open, **no** `tick_documents` document renders. So the plan's `displayed` list is the examples
   set *instead of*, not in addition to, the render set — and `current` (D6's budget-first document)
   has no meaning there, since the current document is not rendering at all.
2. **Step 8 cannot reach them.** `begin_frame` has exactly one live call site,
   `ui.py:265`, iterating `tick_documents`; example documents never receive it (verified: the only
   other `begin_frame` calls are `document.py:816`'s export path and the iteration swap). Step 8
   gates "`begin_frame` … only for documents the plan admits" over that one loop. An interval
   computed for an example id is therefore read by nothing — the "defined is not wired" shape, and
   V9's Examples sibling only tests the RECORDER, not the throttle.

The consequence today is mild — no shipped example holds a feedback history (checked every
`graph.json`: zero self-reading passes across all six), so the missing `begin_frame` costs nothing
yet — but the throttle gate is genuinely absent: the examples loop at `ui.py:353-356` renders every
non-pending example every frame with no interval check, which is the exact cost D10 claims to have
brought under the budget.

*Fix.* Either (a) say the examples set REPLACES `tick_documents` in the plan while the popup is
open, give the examples render loop the same `(frame_idx + phase) % k` gate and a `begin_frame`
call beside it, and give V9's Examples sibling a throttle assertion; or (b) keep round 1's
exemption and say plainly that the Examples popup is unthrottled because it is transient and
modal — which is defensible, and cheaper. What cannot stand is D10's claim that they are
"scheduled by the remainder rule" with nothing reading the schedule. If (b), drop the Examples
recorder too, or Auto sizing for examples has a writer and no consumer.

### m1 — MINOR: V12 walks directories, so an untracked stray under `projects/documents/` fails the gate

V12 walks four directory roots. The invariant it states is about TRACKED files ("every tracked
`document.json` is in the new shape", and the table was "verified against `git ls-files`"). A
document created by a local run under `projects/documents/` — an untracked dir the app itself
writes — would be walked and would fail, red on someone's box and green in CI. Filter the walk to
`git ls-files` output, or restrict the roots to the two that cannot receive runtime writes
(`shaderbox/resources/document_examples/`, `tests/fixtures/`) and list the four project files
explicitly. I verified the four roots cover all eleven tracked files today, so this is about
future false reds, not a present hole.

### m2 — MINOR: the prose-budget allowlist carries a rationale the id-keyed spans falsify

`tests/test_ui_prose_budget.py:232-237` exempts `ui_primitives._profile_rows` with the reason
"a profiler span's own name (`pass:blur`, `document:<the document's title>`)". Under D7 a span name
is `document:<uuid>` and the title arrives through the `id -> title` map. The exemption's SITE stays
valid so the gate still passes (the value is free text, not an assertion — the module's "neither
list can rot" rule checks that the site exists, `test_ui_prose_budget.py:15-20`), but the sentence
becomes false. One-word edit; `ui_primitives.py`'s own docstring at `:1436` ("a document's own
title") needs the same.

### m3 — MINOR: D9c's row string is ~3× today's width in a 280-px panel

`_profile_rows` (`ui_primitives.py:1440-1458`) computes the name column as
`avail.x - indent - _number_width(row.number) - SPACE.SM`, and `SIZE.FPS_PANEL_W = 280`
(`theme.py:276`). Today's number is `"12.34 ms"`; D9c's is `"104.2 ms  5 fps  x12"`. `room` is
guarded by `max(0.0, room)` and `clipped_caption` clips, so nothing asserts or overflows — document
names just clip hard on throttled rows. D7 already shows the drafter is tracking this budget ("a
uuid does not fit a 280-px panel"), so this is a note, not an objection: either widen
`FPS_PANEL_W` or drop the `ms` unit on document rows.

## False trails (round 2) — checked, cleared, do not re-spend

- **`PreviewCellResult.drawn_size` cannot be computed.** It can: `preview_cell` already binds
  `dw, dh = tw * scale, th * scale` at `ui_primitives.py:1252`. Note only that `dw/dh` bind
  **inside** `if texture_glo is not None and min(texture_size) > 0`, so the field needs a default
  (`(0.0, 0.0)`) for the no-texture path — the drafter's `drawn_size: tuple[float, float]` on a
  `@dataclass` with defaults handles it as long as one is given.
- **Both recorder call sites lack the document id.** They have it:
  `document_grid.py:69` iterates `(id, ui_document)`, `examples.py:95-99` has
  `ui_document_example.id`. `examples.py` calls `draw_document_preview_button(...).clicked` without
  binding the result — a one-line change, not a design problem.
- **`resolution` collides with an existing attribute.** No `.resolution` attribute exists on
  `Document`, `UIDocument` or `UIDocumentState` today (grep excluding `resolution_details` /
  `resolution_policy` returns nothing). `media.ResolutionDetails` and
  `render_preset.ResolutionPolicy` are unrelated type names. The name is free.
- **`throttle_color` duplicates `load_color`.** It does not: `load_color`
  (`theme.py:326-332`) returns `STATE_ERROR` at `ratio >= LOAD_ERROR_RATIO = 1.0`
  (`theme.py:323`), and a converged throttled document sits at exactly share/budget ≈ 1.0 —
  correctness F10 was right, and a separate function with `STATE_OK` at ≤ 1.0 is the correct fix,
  not a duplicate.
- **`self.render_pass.canvas = …` in the funnel is invalid.** `render_pass` is a property returning
  a `Pass` (`document.py:328-337`); assigning to `.canvas` sets the Pass's attribute, which is
  exactly what `_swap_feedback` already does at `document.py:403-404`.
- **`ProfileRow` needs the two new fields after all.** It does not: `_profile_rows` reads only
  `depth`, `name`, `count`, `number`, `color`, `starts_tree`. Correctness F11's removal is right.
- **D1's aspect-preserving clamp is hand-waving.** Ran it: on a stored 1080×1920 (9:16) document, a
  naive clamp at an 8000×8000 panel gives (4096, 4096), aspect 1.0000 against a target of 0.5625;
  D1's constrained-axis rule gives (2304, 4096), aspect 0.5625 exactly. At the lower bound, naive
  gives (16, 16) / 1.0000 and D1's rule (16, 28) / 0.5714. The rule does what it claims.
- **The id-keyed spans break existing profiler tests.** `test_profiling.py:549,575,599` match on the
  `"document:"` PREFIX only; `:849,856` build a synthetic `document:one` and call the 3-argument
  `profile_rows_plan`, which keeps working with the two new parameters defaulting to `None`.
- **`_seed_feedback`'s new resample contract needs a new caller.** It has one:
  `load_from_dir` already calls `document._seed_feedback(document_dir, _feedback_rows(metadata))`
  at `document.py:806`, after the passes exist, so the reordered loader's resolved size is in place
  by then.
- **The dogfood harness or the shipped examples hold feedback that a throttle would stall.** No
  shipped example has a self-reading pass (checked all six `graph.json` wirings: zero self-reads),
  and the harness runs without an `App`, so Auto and the plan never engage there.

## New blast-radius reader the revision introduces

One file, and it is a hard break rather than a behavior shift:

| file:line | what it does | why the revision breaks it | severity |
|---|---|---|---|
| `tests/test_youtube_exporter.py:205-236` | builds a stub document whose ONLY attribute chain is `render_pass.canvas.texture.size` (`class _Canvas: class texture: size = document_canvas`), then drives `_artifact_matches_shape` and `resolve_dims` against it | D5 changes `_artifact_matches_shape` to resolve against `current_document.document.resolution`. The stub has no `resolution` attribute, so the test raises `AttributeError` rather than failing an assertion. It is **not** in *Files touched*, whose test list names eleven files and not this one. | MAJOR (add to the list; the edit is two lines on the stub) |

Everything else the revision renames was checked and needs no unlisted edit: `drawn_size` is new
(no readers), `throttle_color` is new (no readers), `resolution` has no name collision, and the
`document:` span readers are the three prefix matches and one synthetic already cleared above.

## Coverage statement (round 2)

**Read end-to-end:** the revised `01_spec.md`, all 688 lines, including the full *Review history*.

**Re-read in the code to check a specific claim:** `git ls-files '*document.json'` (the eleven, and
every one classified against V12's four roots); `ui_primitives.py` (`preview_cell`'s draw math at
`:1242-1262`, `PreviewCellResult` at `:1137-1141`, `_profile_rows` at `:1433-1458`, `_number_width`,
`profile_rows_plan`); `theme.py` (`load_color`, `LOAD_WARN_RATIO` / `LOAD_ERROR_RATIO`, `STATE_*`,
`FPS_PANEL_W`); `ui.py` (`_tick_frame_state`'s extent and its `return` at `:267`, the render block
at `:316-362`, the examples branch, `frame_idx` at `:260,265,539`); `document.py` (`render_pass`
property, `set_canvas_size`, `_swap_feedback`, `_feedback_canvas`, `_seed_feedback`,
`load_from_dir`'s `_seed_feedback` call, `render_media`, `_as_canvas_size`);
`render_preset.py::resolve_dims` and every one of its call sites; `render_shape.py::shape_to_preset`;
`exporters/youtube.py` (`_artifact_matches_shape` and its one caller at `:460`);
`exporters/telegram.py::render_preset`; `widgets/document_grid.py` and `popups/examples.py`'s grid
loops; `tests/test_youtube_exporter.py`, `tests/test_ui_prose_budget.py`,
`tests/test_feedback_persistence.py`, `tests/test_document_dir_sync.py`, `tests/test_profiling.py`.

**Ran:** the tracked-file inventory and its classification against V12's roots; a grep for
pixel-dependent shader content across all six example dirs; a parse of all six example `graph.json`
wirings for self-reading passes; a numeric check of D1's aspect-preserving clamp at both bounds;
greps for readers of `document:` spans, `resolve_dims`, `.resolution`, `begin_frame` and
`frame_idx`.

**Not re-run:** the always-on query probe (round 1's two runs are quoted correctly in the spec,
including the sign-flip caveat) and the `copy_framebuffer` measurement (both reviewers reproduced
it independently in round 1).

**Verdict: PASS-WITH-MINORS** — all fifteen round-1 findings closed; R1 (the `fixed` example is one
document, not two) and R2 (`frame_idx` lives outside `_tick_frame_state`, so four V-items are
unreachable as written) are one-line spec corrections; R3 needs D10 to either wire the Examples
popup's throttle or drop the claim.
