# Post-implementation review — 090 Revision 1 ("Auto owns only an aspect")

Diff under review: `26caad9` *090: revision 1, Auto owns only an aspect*, 41 files
(+1327 / −374). Branch `dev`, tree otherwise as committed.

**Verdict: PASS-WITH-MINORS.** The mechanics land as the maintainer specified them, the
removed machinery is genuinely gone, `export_source_size` is the single export seam, and every
falsifier I re-applied went red on the test the spec names. One MAJOR: `conventions.md`'s own
090 decision bullet — the file the cold-start chain routes a change through — still states the
replaced model in four particulars, so the next session reads a contradiction of the shipped
code. Two MINORs besides.

`make gates` run unpiped, exit code captured before anything else: **0**, smoke RAN (not
skipped) — `check passed, test passed, smoke passed`.

---

## What I read

**Read end to end:** `shaderbox/render_shape.py` (204 lines), `shaderbox/tabs/document.py`
(444), plus the complete commit diff of `shaderbox/document.py`, `ui_models.py`, `app.py`,
`ui.py`, `render_plan.py`, `ui_primitives.py`, `widgets/details.py`,
`widgets/document_grid.py`, `popups/examples.py`, `help_content.py`, `copilot/backend.py`,
`exporters/youtube.py`, `theme.py`, and around each hunk enough surrounding code to judge it
(`_resolve_resolutions` whole, `render_media` whole, `create_document_from_example` whole,
`_clamped_to_aspect`, `resample_canvas`, `segmented_choice`, `toggle_button`, `chip_button`,
`small_caption`). The spec's `## Revision 1` section in full, the unchanged decisions it
depends on, the `conventions.md` 090 bullets, `dev_flow.md`'s module map, the imgui skill's
§4 (SetCursorPos).

**Tests read in full:** `test_canvas_fields.py`, `test_document_shapes.py`, plus every
revision-1 hunk of `test_render_decoupling_loop.py`, `test_render_plan.py`,
`test_canvas_resample.py`, `test_render_for.py`, `test_document_dir_sync.py`.

**Skipped:** the unchanged bulk of `copilot/backend.py` (2711 lines — I read the three changed
regions and grepped every export/render path), `ui_primitives.py` outside the two changed
regions and the four button tiers, `exporters/youtube.py` outside `_artifact_matches_shape`,
`app.py` outside the four changed regions, and
`ai_docs/features/090_render_decoupling/design/aspect_picker.html` (213 lines, a sketch, not
shipped code). `test_youtube_exporter.py` and `test_probe_clock_and_turn_end.py` were read at
their changed lines only.

All demonstrations below ran in a throwaway `git worktree` first on `PYTHONPATH`; the worktree
is removed and the shared tree was never mutated.

---

## Coverage table

| # | Clause | Verdict | Evidence |
|---|---|---|---|
| 1 | **R1** Auto stores an aspect, no size; reduced integer ratio | LANDED | `ui_models.py:163` `aspect: tuple[int,int] = DEFAULT_ASPECT`; `document.py:308`; `render_shape.py:49-73` `reduce_aspect` reduces by gcd and clamps AFTER, with the 1280×960→25:24 bug named in the docstring |
| 1 | **R1** live canvas = viewer region fitted to the aspect | LANDED | `ui.py:395` `document.clamped_size(fit_to_aspect(region, document.aspect))`; `render_shape.py:110-125` |
| 1 | **R1** EVERY Auto document, current or not | LANDED | `ui.py:380-403` loops `document_ids` with no current-document branch; pinned by `test_render_decoupling_loop.py:298` |
| 1 | **R1** `App.displayed_sizes` → single `App.viewer_region` | LANDED | `app.py:1284`; written once, `ui.py:841` |
| 1 | **R1** default `(16, 9)` for a new document | DEVIATED (MINOR-1) | `render_shape.py:29` is `(16,9)`, but a new document is a copy of the starter example, which carries `aspect [4,3]` — probe below |
| 2 | **R2** Fixed stores `resolution`; pair not shown/editable under Auto | LANDED | `tabs/document.py:362-365` branches the whole control; `_draw_canvas_fields` docstring states the rule |
| 3 | **R3** Auto→Fixed seeds from live canvas | LANDED | `tabs/document.py:73-80`; test at `test_render_decoupling_loop.py:594` |
| 3 | **R3** Fixed→Auto seeds aspect from the reduced ratio | LANDED | `tabs/document.py:81-84`; test at `:616` |
| 3 | **R3** Auto→Fixed when the region is not yet known | LANDED | probe below: seeds `(720,540)`, the loader's aspect-fitted size — never `(0,0)` nor the stale pair |
| 4 | **R4** `export_source_size` is the ONE export seam | LANDED | `document.py:391-401`; four call sites, no bypass — grep below |
| 4 | **R4** copilot `set_canvas_size` switches to Fixed and seeds the aspect | LANDED | `copilot/backend.py:1262-1270` |
| 5 | **R5** segmented `Auto \| Fixed` control | LANDED | `ui_primitives.py:126-146` `segmented_choice`; called `tabs/document.py:58` |
| 5 | **R5** aspect chips + custom ratio writing the REDUCED pair | LANDED | `tabs/document.py:99-154`; `_apply_aspect:90` reduces; test at `test_canvas_fields.py:199` |
| 5 | **R5** variant-A row 2: plain readout, `font_12`, no prose | LANDED | `tabs/document.py:211-226`; `small_caption(app.font_12, text)` and nothing else |
| 5 | **R5** tweak 2 — Fixed readout is the ASPECT | LANDED | `tabs/document.py:220-221`; test `test_a_fixed_documents_readout_is_the_aspect_of_its_pair` |
| 5 | **R5** caption `Aspect` / `Canvas` | LANDED | `tabs/document.py:338-343` |
| 8 | **Readout rule** gcd, 1 % snap, else reduced integers | LANDED | `render_shape.py:84-101`; probe reproduces every case the spec names |
| 6 | **R6** `u_resolution` help text; copilot working-set line | LANDED | `help_content.py:42`; `copilot/backend.py:286-297` |
| 7 | **R7** eleven `document.json` files, both keys, agreeing | LANDED | all 11 verified by probe; `77a84d27` alone is Fixed; gate at `test_document_shapes.py:73` |
| 3 | **Removed machinery gone** | LANDED | grep below: zero hits in `shaderbox/` and `tests/` |
| 9 | **Throttle untouched** | LANDED | falsifier re-applied, red |
| 9 | **Resample untouched** | LANDED | falsifier re-applied, red (2 tests) |
| 10 | `02_throttle_and_resolution.md` points at the revision | LANDED | superseded-in-part block at its head, lines 9-15 |
| 10 | `dev_flow.md` module map names only existing symbols | LANDED | all 11 named symbols resolve — check below |
| 10 | `conventions.md` bullets no longer mention the largest region | **MISSING** | **MAJOR-1** — `conventions.md:754-779` untouched |
| 10 | shader-lab template has the new keys | LANDED | `SKILL.md:144` adds `"aspect": [9, 16]`, and the legend rewritten |
| 11 | No comments narrating the change | LANDED (one borderline) | see MINOR-2 |

---

## Findings

### MAJOR-1 — `conventions.md`'s 090 decision bullet still states the model revision 1 replaced

`ai_docs/conventions.md:754-779`, the bullet headed **"A document stores ONE size, and its MODE
says what that number means (feature 090)"**, was not touched by this commit. The only
conventions edit in the diff is three lines inside the *throttle* bullet (`:798-800`). The
decision bullet now contradicts the shipped code in four particulars:

- `:754` — "A document stores ONE size" — under Auto it stores an aspect and the pair is inert.
- `:756-757` — "Under AUTO the live canvas follows **the largest UI region showing the
  document**" — the rule this revision deleted. `grep -rn "largest" shaderbox/ tests/` returns
  nothing; this line is the last statement of it in the repo outside the feature's own history.
- `:757-759` — "the pair is the document's EXPORT resolution — what `RenderShape.NATIVE` …
  resolve to" — inverted. Under Auto `NATIVE` now resolves to the LIVE canvas
  (`document.py:398-400`).
- `:770-773` — "**Every ASPECT reader reads `resolution`, never the live canvas**" and "**Every
  EXPORT path resolves from `resolution`**" — both superseded: aspect readers now call
  `shape_aspect()` (`ui.py:834`, `widgets/details.py:107`) and export paths call
  `export_source_size()`.

Why this is MAJOR and not a doc nit: `CLAUDE.md`'s cold-start chain routes every *change* through
`conventions.md ## Design decisions` precisely so a settled decision is not re-derived or
violated, and the file's own framing is "we decided X; revisit if Y". A reader following the
chain today is handed the pre-revision model as the settled one. The spec's own "Files changed"
section claims `ai_docs/conventions.md` (the 090 bullets) was updated — so the commit believes
it did this.

**Fix.** Rewrite `:754-779` to the revision-1 model, keeping the four "consequences a change
must not break" that still hold (the two-places/two-writers pairing, `set_canvas_size` as the
single writer with its resample-before-release, the deferred `pending_resolution` write) and
replacing the two that inverted:

- heading → **"A document is shaped by an ASPECT under Auto and sized by a PAIR under Fixed
  (feature 090, revision 1)."**
- the largest-region sentence → the live canvas is `App.viewer_region` fitted to `aspect`, one
  size source for every Auto document, current or not.
- "Every ASPECT reader reads `resolution`" → reads `Document.shape_aspect()`.
- "Every EXPORT path resolves from `resolution`" → resolves through
  `Document.export_source_size()` — the live canvas under Auto, the pair under Fixed.
- keep a `Revisit if` clause.

### MINOR-1 — a new document opens 4:3, not the 16:9 the spec's R1 names

R1 says the stored aspect is "`(16, 9)` for a new document", and
`test_a_new_document_opens_wide_and_sizes_itself_to_the_viewer`
(`test_render_decoupling_loop.py:638`) is cited as the maintainer's bug pinned as a test. It
does not pin the clause it appears to. The test creates the document and then *assigns*
`document.resolution_mode = AUTO` and `document.aspect = DEFAULT_ASPECT` before asserting — so
what it proves is that `fit_to_aspect` is wired, not that a new document arrives 16:9. (It also
writes `created = app.create_document_from_example(...)` and branches on
`isinstance(created, str)`, but that method returns `None` — `app.py:2103` — so the branch is
always the fallback.)

What a real new document does, probed through the actual creation path:

```
PROBE mode= auto aspect= (4, 3) resolution= (1280, 960) canvas= (720, 540)
PROBE after drive canvas= (533, 400)   16:9 would be (711, 400)   4:3 would be (533, 400)
```

Every "new document" is a copy of `STARTER_EXAMPLE_ID` (`app.py:618`, `ui.py:707`,
`document_grid.py:46`, `popups/examples.py:84` — there is no other constructor), and the
starter example UV Mango ships `aspect [4, 3]`. `DEFAULT_ASPECT = (16, 9)` is therefore only
the *model* default, reached by a document whose JSON omits the key — which
`test_document_shapes.py` now forbids for every tracked file.

**Severity.** MINOR, not MAJOR: the maintainer's actual complaint was *square at 64×64*, and
that is fixed — a new document is 4:3 at the viewer's size, wide and correctly scaled. The
`DEFAULT_ASPECT` falsifier is also live (see below). But the roadmap's "Awaiting his eyes" line
in the working tree already promises him "a new document opening **16:9** at the viewer's
size", so he will look for 16:9 and see 4:3.

**Fix — one of two, his call.** Either (a) set the starter example's `aspect` to `[16, 9]`
(and its `resolution` to a 16:9 pair, hand-edited per the no-migration rule) so the promise
holds; or (b) correct R1 and the roadmap line to say a new document opens at the starter
example's own aspect. Whichever is chosen, strengthen the test to assert the mode and aspect
the document *arrives* with instead of assigning them, and drop the dead
`isinstance(created, str)` branch.

### MINOR-2 — one added comment narrates the change rather than the now

`ui.py:434-435`: "…and a new document can **no longer** open square because nothing had
recorded a region for it yet." The clause explains the defect that was, in the tense the code
rule excludes ("never narrate development history"). The two sentences before it describe the
mechanism as it is and carry the whole meaning.

**Fix.** End the docstring at "…rather than a separately-sized render." The bug story is
already in the commit message and in the spec's Revision 1 section, which is where the rule
sends it.

Everything else I flagged on the first pass was a false positive — `render_shape.py:58`
("which turned a 1280×960 document into 25:24"), `render_plan.py:10` ("since revision 1"),
`tabs/document.py:158` ("exactly as before 090"), `ui_models.py:600` ("revision 1") — each
states a live constraint or names a decision's home, which is what the rule permits.

---

## Demonstrations

**Removed machinery is gone.** `grep -rn 'displayed_sizes|record_displayed_size|auto_canvas_size|drawn_size|largest region|largest-region|largest UI region'` over `--include=*.py --include=*.md --include=*.json`, excluding the feature's own research/reviews/spec history: **zero hits in `shaderbox/` and `tests/`**. The surviving hits are `conventions.md:757` (MAJOR-1), `roadmap.md:50` (a duplicate 090 row — a false trail, see below), the spec's own superseded D1/D2 text, and `02_throttle_and_resolution.md`, which carries the "superseded in part" banner at its head.

**`export_source_size` is the only export size source.** All four non-test call sites: `document.py:1058` (`render_media`, the funnel every export goes through — Render tab, Share scratch, copilot tools, per its own comment at `:1047`), `copilot/backend.py:2182` (the render-facts probe), `widgets/details.py:85` (the Render tab's presets), `exporters/youtube.py:519` (`_artifact_matches_shape`). `resolve_dims` has exactly three callers in `shaderbox/` — `document.py:1072` and `youtube.py:518`, both fed from `export_source_size()`, and `tabs/document.py:298`, which is the Fixed-only canvas-presets list, where the pair IS the live canvas. Of the remaining `.resolution` reads, every one is a write, a Fixed-mode read, a seed, or the readout. No bypass.

**Mutation — `export_source_size` ignoring the mode** (return `self.resolution` always):
```
FAILED test_render_for.py::test_an_auto_export_renders_at_the_live_size[None]
FAILED test_render_for.py::test_an_auto_export_renders_at_the_live_size[native]
FAILED test_render_for.py::test_the_publish_gate_resolves_from_the_same_source_the_export_does
3 failed, 22 passed
```
Three red, exactly as the spec's breaks-tried list claims.

**Mutation — the maintainer's bug, `DEFAULT_ASPECT = (1, 1)`:**
```
FAILED test_render_decoupling_loop.py::test_a_new_document_opens_wide_and_sizes_itself_to_the_viewer
1 failed, 25 deselected
```
Live, though see MINOR-1 for what it does and does not pin.

**Mutation — drop the 1 % preset snap in `aspect_label`** (iterate no presets):
```
FAILED test_render_plan.py::test_a_size_is_named_by_its_aspect[size3-16:9]     (+ 30:17)
1 failed, 40 passed
```
Exactly one case, 1920×1088 — the alignment case the snap exists for, as the spec states.

**Readout rule, probed directly:**
```
(1280,720) -> 16:9   (1280,960) -> 4:3     (1920,1088) -> 16:9  [reduced (30,17)]
(1214,683) -> 16:9   (512,512)  -> 1:1     (1080,1920) -> 9:16
(607,341)  -> 16:9   (1000,333) -> 1000:333            (1600,900) -> 16:9
```
gcd reduction, 1 % snap, else reduced integers — every case the spec names, including the
`1214×683 → 16:9` collision it flags as a known conflict with the brief. `reduce_aspect`
degrades `(0,5)→(1,5)`, `(-3,4)→(1,4)`, and clamps AFTER reducing.

**Throttle untouched** — removed the `MAX_INTERVAL` cap (`render_plan.py:212` → `max(1, value)`):
```
FAILED test_render_plan.py::test_the_interval_cap_binds_every_document_not_only_the_zero_fps_case
1 failed, 34 passed
```

**Resample untouched** — dropped the one-quad blit in `resample_canvas`:
```
FAILED test_canvas_resample.py::test_a_resize_keeps_the_live_canvas_and_every_history
FAILED test_canvas_resample.py::test_a_seeded_history_at_another_size_is_resampled_not_dropped
2 failed, 2 passed
```

**Mode switch, both directions, including the first frame.** Auto→Fixed with `viewer_region = None` (nothing drawn yet):
```
PROBE pre-switch canvas= (720, 540)
PROBE seeded resolution= (720, 540)   ui_state= (720, 540)
PROBE after ticks canvas= (720, 540)  resolution= (720, 540)
```
It seeds the live canvas — which on frame one is the loader's `fit_to_aspect(INITIAL_AUTO_REGION, aspect)` (`ui_models.py:41-46, 600-610`), a shaped size rather than `(0,0)` or the stale pair. The stale `(111,222)` never appears. Fixed→Auto on odd sizes:
```
(1214,683)  -> aspect (1000,563)    ratio 1.77745 vs 1.77620
(1920,1088) -> aspect (30,17)       ratio 1.76471 vs 1.76471   (exact)
(1237,919)  -> aspect (1000,743)    ratio 1.34603 vs 1.34590
(4096,17)   -> aspect (250,1)       ratio 240.94  vs 250.00
```
The `MAX_ASPECT_TERM = 1000` rescale is lossy on a coprime pair. I checked whether it is
visible by re-fitting each aspect into its own original region: the round trip is off by at
most **one pixel** on one axis (`(1214,683)→(1213,683)`; `(4096,17)→(4096,16)`). Sub-visible,
and the bound exists for a stated reason (`render_shape.py:42-46` — the terms live in
hand-edited JSON). **Not a finding.**

**The eleven `document.json` files.** All 11 tracked files carry `resolution_mode`, `aspect`
and `resolution`; every aspect is the exact gcd reduction of its pair; `77a84d27` is the sole
`fixed`, matching `_FIXED_DOCUMENTS` at `test_document_shapes.py:29`:
```
e7e00c46 auto [4,3]  [1280,960]      ec926580 auto [4,3]  [1280,960]
1901ab60 auto [1,1]  [960,960]       307598da auto [4,3]  [1280,960]
0b0d16bb auto [9,16] [1080,1920]     53724dbd auto [4,3]  [1280,960]
73ea2431 auto [1,1]  [1280,1280]     77a84d27 fixed[1,1]  [512,512]
8d454b7b auto [16,9] [1600,900]      f90f5ff9 auto [4,3]  [1280,960]
bloom_chain (fixture) auto [1,1] [960,960]
```
The gate is real: `test_git_tracks_the_documents_this_gate_expects` guards against the
parametrize list coming back empty, which is the failure mode that would make the whole file a
green no-op.

**UI against `/imgui-ui`.** The segmented control is `ui_primitives.segmented_choice`
(`:126-146`), built from the existing `toggle_button` tier — so the accent-fill/standard-frame
states come from the tier system, not from a call-site push. `grep push_style_color|push_style_var`
over `tabs/document.py` and `widgets/details.py`: **no hits**. The one `push_style_var` inside
`segmented_choice` zeroes horizontal item spacing to join the strip and pops it — correct, and
it is in the primitive where it belongs. The readout uses `small_caption(app.font_12, …)`,
which pushes `font.legacy_size` per the known-quirk. **SetCursorPos:** `_draw_size_readout:225`
sets cursor *X only* to `control_x`, a position left of and behind items already submitted on
the row above, then immediately submits text — it cannot extend the window boundary, so §4's
assert is not reachable. Ids are per document: `imgui.push_id(ui_document.id)` at
`tabs/document.py:356` wraps the whole cluster and pops at `:369`, and
`test_a_document_switch_mid_edit_does_not_resize_the_new_document` pins it. The
`begin_disabled`/`end_disabled` pairing across `_draw_document_reset` is unchanged and balanced.

**`dev_flow.md` module map.** All eleven symbols it now names resolve: `reduce_aspect`,
`aspect_of`, `fit_to_aspect`, `aspect_label`, `ASPECT_PRESETS`, `DEFAULT_ASPECT`
(`render_shape.py`), `plan_render_set`, `apply_damping`, `AutoSizeState`, `CostRecord`,
`document_span_name` (`render_plan.py`). It names no removed symbol.

---

## False trails

- **The `MAX_ASPECT_TERM` rescale losing the ratio.** `reduce_aspect((1214,683))` returns
  `(1000,563)`, not the exact `(1214,683)`, which reads as a correctness hole on Fixed→Auto.
  Measured: the round trip costs at most one pixel on one axis. Dropped.
- **`INITIAL_AUTO_REGION = (960, 540)` as a magic placeholder.** It is a real first-frame
  allocation and its docstring says so; `test_an_auto_document_opens_at_its_aspect_not_at_its_stored_pair`
  (`test_canvas_resample.py`) pins that a 16:9 document with a *square* stored pair opens 16:9,
  which is the property that matters. Dropped.
- **`_apply_canvas_size` still pushing a notification labelled "Export"**
  (`tabs/document.py:267-272`). The `else` branch is unreachable from the UI — the function is
  called only from `_draw_canvas_fields` / `_draw_canvas_presets`, both Fixed-only. Dead
  branch, not a defect; a one-line simplification at most, and below the bar for a finding.
- **Function-body imports in the new tests** (`from shaderbox.tabs.document import _apply_aspect`
  and friends). `CLAUDE.md`'s imports-at-top rule is a code rule; the repo's existing tests use
  this shape throughout for GL-module and private-symbol imports, and `make gates` (ruff +
  pyright) accepts it. Not a regression this commit introduced.
- **Two `| 090 | render_decoupling |` rows in `roadmap.md`.** Real, and the first still
  describes "Auto follows the largest region displaying it". But both rows are present in
  `HEAD` *and* in `0a4cae6`, its parent — the duplicate predates this diff. Out of scope here;
  worth a one-line delete in the next sweep.
- **`ai_docs/roadmap.md` is uncommitted in the working tree.** Its diff rewrites the Active-context
  banner and the second 090 row for revision 1 — someone's in-flight sanitize. Not produced by
  this review (read-only apart from this file), and not part of the commit under review. Noted
  because MINOR-1 interacts with its new "a new document opening 16:9" promise.

---

## Coverage statement

Eleven review items, all exercised. Eight demonstrated by a mutation re-applied in an isolated
worktree (the maintainer's bug falsifier, `export_source_size` ignoring the mode, the
`aspect_label` snap, the throttle's `MAX_INTERVAL`, the resample blit) or by a probe run against
the real app fixture (the new-document creation path, the first-frame Auto→Fixed seed, the
Fixed→Auto odd-size reduction, the readout rule across ten sizes, all eleven `document.json`
files). Three checked by reading plus exhaustive grep rather than mutation — the removed
machinery's absence, the export-path bypass sweep, and the UI-versus-skill audit — since
absence and a style rule have no mutation that demonstrates them.

Not covered: the rendered pixels. There is no WM on this box and the smoke test is headless, so
"does Auto read sharp at the viewer's size" and "does the variant-A row look right" remain the
maintainer's to judge — which is what the roadmap's *Awaiting his eyes* line already says.

**Verdict: PASS-WITH-MINORS** — the revision's mechanics are correct and gated; fix MAJOR-1
(`conventions.md`'s stale 090 decision bullet) before the next session reads it as settled.
