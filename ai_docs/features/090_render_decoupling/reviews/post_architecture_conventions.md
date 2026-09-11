# 090 — post-implementation review: architecture and conventions

Reviewer scope: module boundaries, import direction, duplication, dead code, comment and
docstring discipline, naming against the spec, the UI edits against `/imgui-ui`, the staged
`projects/` documents, the six declared deviations, and what the implementation learned that is
still only in the spec.

Diff reviewed: `f06d957` (50 files). Read-only on the tree except this file; two mutations were
applied to a file, watched, and restored with `git diff --quiet` verification before anything
else ran (recorded under *Findings*).

---

## Coverage

**Read end to end (every changed non-test source file):** `shaderbox/render_plan.py` (new, whole
file), `shaderbox/render_shape.py`, `shaderbox/channel_blit.py`, `shaderbox/document.py` (the
changed regions plus `set_canvas_size`, `resample_canvas`, `_seed_feedback`, `_feedback_canvas`,
`load_from_dir`, `render_media` in full context), `shaderbox/ui_models.py`,
`shaderbox/ui.py` (`_tick_frame_state`, `renders_this_frame`, `_refresh_document_costs`,
`_resolve_resolutions`, `_update_and_draw`'s render block, `_draw_document_image`,
`_current_document_fps`, `_document_titles`, `_draw_app_panel`), `shaderbox/app.py`,
`shaderbox/ui_primitives.py` (`PreviewCellResult`, `preview_cell`, `ProfileRow`,
`profile_rows_plan`, `_document_row`, `_plan_tree`, `_profile_rows`, `fps_overlay`,
`toggle_button`), `shaderbox/theme.py`, `shaderbox/tabs/document.py`,
`shaderbox/widgets/details.py`, `shaderbox/widgets/document_grid.py`,
`shaderbox/popups/examples.py`, `shaderbox/popups/settings.py`, `shaderbox/help_content.py`,
`shaderbox/copilot/backend.py` (the three changed sites), `shaderbox/exporters/youtube.py`.

**New tests read in full:** `tests/test_document_shapes.py`. **Read in the regions that carry
the verification claims:** `tests/test_render_decoupling_loop.py` (V7/V9/V9a/V10/V10a/V11/V13
and the R2 case), `tests/test_render_plan.py` (headers and the off/no-record cases).

**Skipped, with reason:** the bodies of `tests/test_canvas_resample.py` and the twelve edited
test files — the correctness reviewer owns test-content fidelity, and this review needed only
whether a convention or a gate was violated, which the greps and two mutations answered
directly. The eleven `document.json` files were machine-checked for shape and mode rather than
read as prose.

**Contracts read in full before the code:** `CLAUDE.md`, `ai_docs/conventions.md` (all three
sections), `ai_docs/dev_flow.md ### Module map`, `.claude/skills/imgui-ui/SKILL.md` (the prose
budget and button-tier sections), `ai_docs/features/090_render_decoupling/01_spec.md` (all 918
lines).

**Gate:** `make gates` run unpiped, redirected to a file, `$?` read first — **exit 0**,
`== gates: GREEN -- check passed, test passed, smoke passed ==`. `uv run ruff check` clean;
`uv run pyright` 0 errors, 7 warnings (all pre-existing upstream stub gaps). Working tree clean
at start and at end.

---

## Findings

### MAJOR 1 — `throttle_color`'s bands are not gated; reverting it to the exact bug F10 named leaves the suite green

`shaderbox/theme.py::throttle_color` exists only because correctness F10 found that
`load_color`'s knees are wrong for this row: `load_color` returns `STATE_ERROR` at ratio 1.0,
which is precisely where a converged throttled document sits, so green was unreachable for the
case round 1 claimed reads green. `throttle_color` inverts that — `STATE_OK` at ≤ 1.0,
`STATE_WARN` above, `STATE_ERROR` above 1.5 or whenever the frame misses.

Nothing asserts those bands. `tests/test_theme.py` tests `load_color`'s three knees and
`profile_rows_plan`'s row order; `tests/test_render_decoupling_loop.py::test_the_panel_rows_
match_documents_by_id_not_by_title` asserts the row's `number` and `tooltip` and never its
`color`; `tests/test_render_plan.py` is about intervals.

**Demonstrated.** I replaced `throttle_color`'s whole body with `return load_color(share_ratio)`
— reintroducing F10 verbatim, so every converged document's row turns red — and ran the full
suite: `2153 passed, 4 skipped`. Restored, `git diff --quiet shaderbox/theme.py` clean.

This is the repo's own gate law from `conventions.md ## Design decisions` ("A GATE is the one
kind of code whose correctness a normal run cannot show"), applied to the colour policy rather
than to a checker: the function reads as enforcing a decision and enforces nothing. Every other
verification item in this feature was mutation-tested and named in the spec's *Implementation
notes*; this one has no entry there, which is consistent with it never having been broken.

**Fix.** Add three assertions beside `tests/test_theme.py`'s `load_color` case, in the same
shape:

```python
def test_throttle_color_reads_a_converged_document_as_healthy() -> None:
    # F10: load_color's STATE_ERROR starts at 1.0, exactly where a converged throttled
    # document sits. Falsifier: return load_color(share_ratio) and this goes red at 1.0.
    assert throttle_color(1.0, False) is COLOR.STATE_OK
    assert throttle_color(1.2, False) is COLOR.STATE_WARN
    assert throttle_color(1.6, False) is COLOR.STATE_ERROR
    assert throttle_color(0.2, True) is COLOR.STATE_ERROR
```

The last line is the clause the spec singles out ("a document inside its allowance while the
frame still misses is what the reader must see") and is the half most likely to be dropped by a
later edit.

### MAJOR 2 — the `document:` span key is a wire contract defined twice

`_DOCUMENT_SPAN_PREFIX = "document:"` is declared in **two** modules:

- `shaderbox/ui.py:63` — the writer's home: `ui.py:463`, `ui.py:478`, `ui.py:500`, `ui.py:507`
  open the spans, and `ui.py:353`/`ui.py:355` parse them back in `_refresh_document_costs`.
- `shaderbox/ui_primitives.py:1381` — the reader's home: `ui_primitives.py:1484` detects a
  document row and `ui_primitives.py:1449` slices the id back out.

This is one concept — the format of the key joining a profiler span to a document id — with two
homes, which `conventions.md`'s one-canonical-home rule is about. The failure is concrete rather
than stylistic: change the prefix in `ui.py` alone and the panel silently stops recognising
document rows, falling back to `_measured_row` with the raw `newprefix:<uuid>` as the row name.
No test would catch that, because both new tests construct their spans with the literal
`"document:aaa"` rather than through either constant.

It is not a leaf-order problem — `ui_primitives.py` sits below `ui.py` and cannot import it —
but `render_plan.py` is a leaf **both** modules already import (`ui.py:35`, `ui_primitives.py:14`)
and is already the canonical home for every other piece of this policy (`MAX_INTERVAL`,
`INTERVAL_HYSTERESIS_FRAMES`, `AUTO_RESIZE_*`, `RenderPlan`). The key belongs there.

**Fix.** Move the constant to `shaderbox/render_plan.py` beside the other constants, export it,
and import it in both consumers — deleting both local declarations. Two small helpers make the
slicing single-homed too and remove the three duplicated `span.name[len(...):]` expressions:

```python
# render_plan.py — the profiler span key that joins a measurement to a document (090 D7).
DOCUMENT_SPAN_PREFIX: str = "document:"


def document_span_name(document_id: str) -> str:
    return f"{DOCUMENT_SPAN_PREFIX}{document_id}"


def document_id_of_span(name: str) -> str | None:
    """The document a span measures, or None where the span measures something else."""
    if not name.startswith(DOCUMENT_SPAN_PREFIX):
        return None
    return name[len(DOCUMENT_SPAN_PREFIX) :]
```

### MINOR 1 — `Document.resolution` / `resolution_mode` mirror `UIDocumentState`'s, with the pairing held only by convention

The two fields live on both `UIDocumentState` (`ui_models.py:154-155`, the persisted authority —
`UIDocument.save` dumps `ui_state` and nothing else) and `Document` (`document.py:307-308`, a
cache so `render_media` needs no new parameter, which the spec states as the reason). Four write
sites pair them by hand: `ui_models.py:594-595`, `ui.py:388-389`, `tabs/document.py:52-53` and
`97-98`, `copilot/backend.py:1255-1258`. Nothing structurally prevents a fifth site from writing
one half.

I checked whether the pairing is at least incidentally gated. It is: dropping the `ui_state`
half of `tabs/document.py::_apply_canvas_size`'s paired write reddens
`tests/test_document_graph.py::test_the_ui_resize_clamps_both_ends` (`At index 0 diff: 64 !=
4096`). Restored, `git diff --quiet` clean. So today's sites are covered and this is a MINOR,
not a MAJOR — the risk is a *future* writer, not a live defect.

The repo's structural-impossibility law would prefer one home, but every alternative here is
worse than the duplication: making `Document` read `UIDocumentState` inverts the layering
(`ui_models` imports `document`, not the reverse), and threading `resolution` through
`render_media` re-admits the per-caller parameter the export funnel exists to avoid. **No change
requested.** The right response is the `conventions.md` bullet below, which names the pairing as
the contract a fifth writer must honour — the cheapest available enforcement for a rule whose
only alternative is a layering inversion.

### MINOR 2 — the `Fixed` toggle's tooltip restates its own label

`shaderbox/tabs/document.py:58-59` hovers `"Fixed canvas size"` on a button already labeled
`Fixed`. The `/imgui-ui` skill's budget table says an icon or button tooltip is *"the control's
NAME, nothing else"* — the examples are `Pass settings`, `Copy`, `Delete`, all cases where the
control carries an icon and the tooltip supplies the name it lacks. Here the name is on the
button, so the tooltip is the label plus a gloss.

It is inside the 5-word budget and `test_ui_prose_budget.py` passes it, so this is a judgement
call rather than a gate violation. Two defensible fixes: drop the tooltip (the toggle's label
and its accent fill already carry the state, per the skill's "the style carries the state"), or
keep it and make it name what the *other* position means, which is the genuinely ambiguous half.
My preference is dropping it.

---

## Checks that came back clean

- **`render_plan.py` is a true leaf.** `uv run python -c "import shaderbox.render_plan"`
  succeeds in isolation; its only imports are `math` and `dataclasses`
  (`render_plan.py:14-15`). No cycle-from-types: nothing in it is annotated against `App`,
  `Document` or any GL type, and the three consumers (`ui.py`, `app.py`, `ui_primitives.py`)
  all sit above it. Placement matches the `dev_flow.md ### Module map` entry added in the same
  commit, which describes the module accurately.
- **No upward imports introduced.** `ui_primitives.py` gained `render_plan` and `theme`
  symbols only; `document.py` gained `channel_blit` and `render_shape`, both leaves below it;
  `ui_models.py` gained `render_shape`. `channel_blit.py` gained nothing.
- **No `if TYPE_CHECKING`, no `@staticmethod`/`@classmethod`, no `from __future__ import
  annotations`, no new suppression.** `grep` over every changed file returns nothing; ruff and
  pyright are clean with no new allowlist entry in `conventions.md ## Known quirks`.
- **Imports at module top only.** The two sanctioned lazy seams are untouched; no new
  function-body import in any changed source file. (`tests/test_render_decoupling_loop.py` uses
  in-function imports, which is the test-side norm here and outside the rule's scope.)
- **Dead code from the model change: none found.** No reader of the top-level `canvas_size`
  key survives in `shaderbox/` (the only hits are the unrelated `set_canvas_size` tool name and
  `test_document_shapes.py`'s assertion that the key is *absent*). `export_size` — round 1's
  rejected second field — appears nowhere in the tree. `load_color` keeps two live call sites
  (`ui_primitives.py:1436`, `1455`, the unthrottled rows) so it is not bypassed. No unused
  import, no unreachable branch.
- **Every aspect reader reads `resolution`.** `ui.py:827`, `widgets/details.py:106`,
  `tabs/document.py:117`, `copilot/backend.py:2171` — the four sites F2 named. The one
  remaining `np.divide(*canvas.texture.size)` is `core.py:505`, which computes the engine's
  `u_aspect` uniform from the live canvas, correctly: the shader must know the pixels it is
  drawing into, which under Auto *is* the live size.
- **The effective-size resolution, the clamp and the displayed-size recording each have one
  home.** `_resolve_resolutions` (`ui.py:361`) is the sole resolver; `_clamped_to_aspect`
  (`document.py:96`) is the sole aspect-preserving clamp and funnels into `pass_graph.
  clamp_canvas_size`; `App.record_displayed_size` (`app.py:1200`) is the sole recorder, called
  by all three surfaces, and owns the largest-wins rule so no call site repeats it. Three
  `set_canvas_size` callers remain and all are pre-draw, as D4 requires.
- **Comments state the now.** The narration greps (`no longer`, `used to`, `was`,
  `previously`, `we changed`) return only pre-existing lines outside this diff. The forty `090`
  mentions are all ≤1-line spec pointers in the established `088 D2` / `066 D1` / `084 D5`
  idiom that `conventions.md ## Code rules` explicitly sanctions, and each names a
  still-true invariant rather than a change. The one phrase that reads historically —
  `document.py:534`, "stops being a MATCH criterion and becomes the resample TARGET" — is
  inside a docstring explaining what the code does now and is borderline at worst.
- **Docstrings.** The wrapped summary on `throttle_color` matches 121 existing instances in
  the package, so it is the repo norm, not a deviation. Every new public symbol carries one.
- **Naming matches the spec exactly**: `resolution`, `resolution_mode`, `pending_resolution`,
  `displayed_sizes`, `document_costs`, `ThrottleState`, `CostRecord`, `RenderPlan`,
  `AutoSizeState`, `auto_canvas_size`, `apply_damping`, `plan_render_set`, `MAX_INTERVAL`,
  `INTERVAL_HYSTERESIS_FRAMES`, `AUTO_RESIZE_DEAD_BAND`, `AUTO_RESIZE_STABLE_FRAMES`. No
  drift.
- **The UI edits follow the skill.** The mode control is `toggle_button` (`tabs/document.py:50`)
  — the skill's own stateful on/off tier, used identically at `ui.py:759`; no hand-rolled
  `push_style_color` at any call site in this diff; all colour comes from `theme.py` tokens via
  `throttle_color` / `load_color`. The Settings rows copy the `label_row` + `drag_int` +
  `checkbox` idiom sitting three lines above them verbatim. No new `set_cursor_pos` call, so
  the SetCursorPos assert is not in play. The panel row's compact form is the m3 fix and keeps
  the 280-px column intact, with the millisecond cost moved to a tooltip.
- **`projects/` documents are staged and consistent.** Working tree clean; all eleven tracked
  `document.json` files carry `ui_state.resolution_mode` + `ui_state.resolution`, none carries
  a top-level `canvas_size`, and the modes match the spec table exactly — `77a84d27-…` fixed
  at `[512, 512]`, the other ten auto at their prior numbers. `tests/test_document_shapes.py`
  is a well-built gate: its domain is `git ls-files` (the m1 fix) and it carries a census guard
  (`test_git_tracks_the_documents_this_gate_expects`) against the empty-parametrization false
  green, which is the exact shape the repo's gate law warns about.
- **`help_content.py`.** `"canvas size in pixels; follows the view under Auto"` — one clause
  added to an existing gloss, factually correct under both modes, within budget.
- **The six deviations are each forced and none widened scope.** (1) `ProfileRow.tooltip` is
  required by D9c's own text and carries what the compact number drops, so it is a
  clarification of F11 rather than a reversal, as claimed. (2) `profile_rows_plan`'s `budget`
  parameter is arithmetically necessary — D9c's ratio needs the wall-time budget, and the
  existing `budget_ms` is the frame period. (3) `load_document_metadata` public: exactly two
  callers (`document.py:840`, `ui_models.py:588`), and the alternative is a second `json.load`
  of the same file, which is the concept written twice. (4) `as_canvas_size` public: three
  internal callers plus the relocated malformed-pair test; the coercion still guards
  `Document.__init__`. (5) Three extra test files, each a mechanical consequence of a locked
  decision. (6) `test_profiling.py` gained an `xdist_group`, which the corrected quirk requires
  of any frame-driving module. All six are recorded in the spec's *Deviations from the spec*
  section, which is where `conventions.md` says per-feature mechanics belong.
- **The xdist-group mechanism is real, not just documented.** Four modules carry
  `pytest.mark.xdist_group` and `Makefile:41` runs `-n 8 --dist loadgroup`, so the corrected
  quirk describes the tree as it is.

---

## Promote to `conventions.md`

Both items are durable and generic; both are currently only in the spec, where the next reader
of an unrelated feature will not find them.

**1 → `## Known quirks` (a moderngl footgun with a workaround — the section's definition).**
The `copy_framebuffer` measurement is a library fact, not a 090 decision, and today lives only
inside a 090 Design-decisions bullet ("`copy_framebuffer` between differently-sized framebuffers
copies 1:1 into a corner") and in three source comments. A future author reaching for it to
downscale a thumbnail, a preview or an export buffer will not be reading 090's bullet. Proposed
text:

> - **`moderngl`'s `copy_framebuffer` does not RESCALE — it copies 1:1 into a corner.** Between
>   two differently-sized framebuffers it returns `GL_NO_ERROR` and leaves a plausible picture
>   that is the wrong one: measured twice independently, a 64x64 source half white copied into
>   128x128 gives a white fraction of 0.125 where a rescale gives 0.5, with the far corner
>   black. There is no flag; the call has no filter or rectangle parameters to give it. A resize
>   that must preserve content draws ONE QUAD instead, sampling the source texture so the
>   rescale is the sampler's own linear filter — `channel_blit.py::CanvasResampler` is that
>   mechanism, and `Document.resample_canvas` is its one consumer. A mean-brightness assertion
>   passes under the broken version, so a test for a rescale asserts the FAR CORNER. Revisit if
>   moderngl ever exposes `glBlitFramebuffer`'s filter argument.

**2 → `## Known quirks`, appended to the existing xdist-group bullet.** The quirk was rewritten
this wave and states the mechanism correctly, but it does not say the one thing that makes it
enforceable: nothing checks that a frame-driving module declared a group. Deviation 6 is
evidence — `test_profiling.py` drove frames without a mark and it took a review to notice.
Proposed sentence to append:

> Nothing enforces the mark: a module that drives frames without one is a latent crash that
>   surfaces only when the scheduler happens to pair it with a sibling, so the mark is added with
>   the module rather than after the first red run.

**Not promoted, deliberately.** The 5 % dead band, the 8-frame stability window, the 4-frame
hysteresis, `MAX_INTERVAL = 60` and the 0.5 budget are all 090 policy numbers with their
rationale already on the constants in `render_plan.py`; `conventions.md`'s own preamble excludes
per-feature mechanics. The `+0.005 ms` always-on query measurement is likewise already captured
in the 090 bullet and in the amended 088 ring bullet.

---

## False trails

Recorded so the next reviewer does not re-spend them.

- **`Document.resolution` duplicating `UIDocumentState.resolution` looks like a
  one-canonical-home violation and is not one worth fixing.** The layering forbids the obvious
  repair (`ui_models` imports `document`; the reverse would cycle) and the parameter-threading
  alternative re-admits the per-caller shape the export funnel exists to remove. I mutation-
  tested the pairing rather than arguing about it: dropping half a paired write reddens an
  existing test. Filed as MINOR 1 with no code change requested.
- **`_measured_row` and `_document_row` both formatting `f"{ms:.2f} ms"` with
  `load_color(ms / budget_ms)` reads as duplication.** It is the deliberate unthrottled
  fall-through: `_document_row` must produce today's row when `plan is None` or `interval <= 1`,
  and routing that through `_measured_row` would mean passing the resolved title as the span
  name, which is exactly the title-keyed coupling D7 removed. Two lines, correct as written.
- **`render_media`'s new scratch `Canvas` on the `preset=None` / `SCALE_DISTORT` branch looks
  like a leak.** It is released in a `finally`, mirroring the `target` canvas on the branch
  directly below it (`document.py:1032-1042` and `1049-1053`). Same idiom, same shape.
- **`_resolve_resolutions` swapping `app.displayed_sizes` and `app.pending_resolution` for fresh
  dicts looked like it could strand a recorder's write.** It cannot: the recorders run in the
  draw phase, strictly after this function, so they write into the fresh dict that the *next*
  frame reads. Both swaps carry a comment saying why, and both reasons check out.
- **The always-on profiler looked like it might have left `profiler.enabled` writable from a
  second site.** The single assignment `app.profiler.enabled = app.fps_details_open` is deleted
  at `ui.py` and the constructor is `Profiler(enabled=True)`; no other writer exists, and
  `test_profiling.py` names the deleted line as its falsifier.
- **The `88 D2` two-frame cost lag against the 4-frame hysteresis window.** Verified
  arithmetically against `plan_render_set`'s `_settle`, which requires four *agreeing* frames
  before moving `interval` — strictly longer than the lag, as D6 claims. Not a finding.
- **`u_aspect` still reading the live canvas at `core.py:505`.** Correct by design: the shader
  needs the aspect of the pixels it is drawing into, which under Auto is the live size. Not one
  of F2's four sites.

---

## Verdict

**PASS-WITH-MINORS.** The architecture is sound: `render_plan.py` is a genuine leaf whose import
isolation I verified by running it, the layering is unviolated, no dead code survives the model
change, the comments state the now, the naming matches the spec symbol for symbol, the UI edits
go through the sanctioned primitives and tokens, all eleven documents are staged in the declared
shape behind a well-built gate, and all six deviations are forced, minimal and filed where the
conventions say. Two items to land before close-out: the missing `throttle_color` band test
(MAJOR 1 — demonstrated by mutation, the suite stays green while the bug F10 named is back in
the tree) and the twice-declared `document:` span key (MAJOR 2), plus the two
`conventions.md ## Known quirks` promotions above.
