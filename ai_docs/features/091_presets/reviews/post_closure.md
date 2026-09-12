# 091 closure review — the three post-implementation reports against `7ac0b30`

Anchor: `reviews/post_code_correctness.md` (F1–F6), `reviews/post_architecture_conventions.md`
(F1–F4 plus its §3 housing judgements), `reviews/post_spec_fidelity.md` (F1–F3 plus its
PARTIAL items 3 and 19), each closed against `7ac0b30` ("091: close the post-implementation
review"). Read-only: the baseline breaks ran in `git worktree add /tmp/091-closure 7ac0b30`,
removed at the end (`git worktree list` shows only the main tree). Four temporary probe files
were written under `tests/` and deleted; `git status --short` afterwards shows only the two
pre-existing uncommitted edits described below.

## Verdict: PASS

Every finding in the three reports is closed at `7ac0b30`, and both of the two real defects are
closed in the code, not only in the docs — each re-derived here from the code and each gate
broken and watched go red. `make gates` is green.

```
$ make gates > /tmp/g.log 2>&1; echo $?
0
== gates: GREEN -- check passed, test passed, smoke passed ==
```

The check stage passed on the **first** run; no hook rewrote anything. `git status --short`
reports two modified files, and both predate the run rather than being written by it —
`ai_docs/conventions.md` and `scripts/smoke.py` carry mtime `11:43:10`, twelve seconds before
`/tmp/g.log`'s `11:43:22`, and both diffs are prose, which no formatter hook produces. They are
the implementer's own unfinished edits, not a gate artifact. Neither changes behaviour:

- `scripts/smoke.py` — the comment above the stamp loop, updated to match the three-pass stamp
  that `7ac0b30` already committed ("Two adjacent passes and one apart" for the landed
  `("paint", "seed", "df")`). The committed comment still says "Two non-adjacent passes".
- `ai_docs/conventions.md` — the import-by-copy bullet gains the loop rejection the fix commit
  added to the code and to the spec. At `7ac0b30` that bullet describes `plan_import` as
  returning "every write or the rejection" without naming the loop check.

**The one thing to do before this feature is closed out:** commit those two edits. They are the
doc half of the loop fix, written and not staged. Nothing in this review depends on them —
both findings they touch are closed by the committed spec text — but a tree sitting ahead of
its own commit with the decision's `conventions.md` sentence uncommitted is the state the
next session reads as stale.

## Closure table

### `post_code_correctness.md`

| # | Verdict | What closes it at `7ac0b30` |
|---|---|---|
| F1 — the dialog's defaults can build a cycle, and the import accepts it | **CLOSED** | `pass_import.plan_import` builds the post-import wiring and rejects a loop; re-derived below. The fix is the report's own recommended one (`plan_passes` over the merged wiring), not the narrow variant. |
| F2 — a part-way failure leaves orphan passes that resurrect on the next load | **CLOSED** | `ProjectSession.import_passes` copies every value before the first write and unwinds the write loop; re-derived below. |
| F3 — the commit does not pass `make check` though the spec says it was green | **CLOSED** | `make gates` exits 0 at `7ac0b30`, check green on the first run (above). `01_spec.md ## Implementation notes` now reads: *"The implementation commit (`4dc1423`) was committed with two files the formatter had not yet rewritten and one unsorted import block, so `make check` was red at that commit while the note here claimed green; the post-implementation review caught it and the fix commit that follows is the one the gate is green at, exit code read unpiped."* The claim now names the commit it is about, which is what made the old sentence expensive. |
| F4 — `CopilotCapabilities.set_pass` declares `group` with no default | **CLOSED** | `copilot/capabilities.py::CopilotCapabilities.set_pass` now reads `group: str \| None = None, /`, matching `backend.py::CopilotBackend.set_pass`. F4's own probe re-run: a module calling `caps.set_pass("doc", "main", 1, "f1", 1.0, True, False, False, "")` against the Protocol gives `uv run pyright` → `0 errors, 0 warnings, 0 informations`. |
| F5 — `preview_cell(bordered=False, border_color=…)` silently drops the border | **CLOSED**, and stronger than the report's suggested fix | `ui_primitives.py::preview_cell` takes `child_flags=imgui.ChildFlags_.borders if bordered or border_color is not None else …always_use_window_padding`, so the colour now has a border to tint rather than asserting the combination away. The docstring gained "An explicit `border_color` still draws its border either way." Measured, three cells in one real frame — bordered / unbordered / unbordered-with-`border_color` — content origins `[(8.0, 8.0), (8.0, 8.0), (8.0, 8.0)]`, so the border came back with no padding shift. The call site followed: `widgets/pass_list.py::_draw_pass_tile` now passes `bordered=tint is None`, dropping the `or border is not None` disjunct it no longer needs, which is also the expression D7's Deviations entry records. |
| F6 — `_draw_group` shows an empty group picker | **CLOSED** | `popups/pass_settings.py::_draw_group` wraps the combo in `imgui.begin_disabled(not existing)` / `imgui.end_disabled()`, the sibling shape F6 named. |

### `post_architecture_conventions.md`

| # | Verdict | What closes it at `7ac0b30` |
|---|---|---|
| F1 — `make check` RED and the spec says green | **CLOSED** | Same as code-correctness F3 above: gate exit 0, spec sentence corrected. |
| F2 — `tests/test_pass_import.py::_plan` carries a `# type: ignore` outside the allowlist | **CLOSED**, by the fix F2 prescribed | `_plan` now has explicit keyword parameters with the fixture defaults (`source_wiring: Wiring = _BLOOM`, … `host_output: str = "final"`) and forwards them positionally; `substitutions: Mapping[str, str] \| None = None` carries the `{"scene": "main"}` default at the call rather than in the signature, since a mutable default cannot sit there. No call site changed (all fourteen already passed keywords). `grep -rn "type: ignore\|noqa\|pyright: ignore"` over `tests/test_pass_import.py`, `tests/test_import_dialog.py`, `shaderbox/pass_import.py`, `shaderbox/popups/import_passes.py`, `shaderbox/widgets/pass_list.py`, `shaderbox/document.py` returns nothing. |
| F3 — `ImportDraft.rejection` written and read by nothing in production | **CLOSED**, by F3's own first option | `popups/import_passes.py::_draw_body` now gates the button on the field: `imgui.begin_disabled(bool(draft.rejection))`, and the red text reads `draft.rejection` rather than the local `plan`. So the docstring's claim ("so the Import button and a test read the same thing") is now true, and the field has a production consumer. |
| F4 — `group_tint` inserted mid-way through theme.py's SELECT invariant block | **CLOSED** | `theme.py`: the `def group_tint` moved out of the assert region down to the function area (it now sits immediately above `apply_theme`). The finding was specifically the *function* splitting the block — "accurate for the asserts and not for the function" — and the asserts are where they were. The `conventions.md` sentence F4 flagged ("the import-time assert beside the SELECT invariant") is now literally true. |
| §3 — `offered_entry_points` housed above its layer | **CLOSED** | Moved to `shaderbox/document.py::offered_entry_points`, the home §3 argued for. `popups/import_passes.py` imports it from `document` (in one sorted block with `document_dir_of`), and the popup no longer imports `project_session` at all. `tests/test_pass_verbs.py` imports it from `document`. `compile_pending_passes` stayed in `project_session.py`, which §3 itself called "a legitimate home". `dev_flow.md`'s module map and the spec's Deviations bullet both follow, naming `document.offered_entry_points`. |
| §3 — `host_readers_of` is logic, not state | **CLOSED**, by §3's own suggested signature | `pass_graph.py::readers_of(wiring: Wiring, fed: str) -> set[tuple[str, str]]` landed beside `entry_points`, and `app.py::App.host_readers_of` shrank to `return readers_of(ui_document.document.effective_wiring(), fed)` — the one-line delegation §3 described. `dev_flow.md` names `pass_graph.readers_of` as the source of the default handovers. |

### `post_spec_fidelity.md`

| # | Verdict | What closes it at `7ac0b30` |
|---|---|---|
| F1 — the check gate is red on the commit | **CLOSED** | Same as above. |
| F2 — D4's "a plan that copies nothing" is unreachable; an undocumented seventh rejection fires | **CLOSED** | `01_spec.md` D4 now lists the rejection that actually fires: *"…or a substitution of the source's OUTPUT pass ("is the output and stays"), which is what a single-pass source's one pass is; the dialog disables that combo, so the rejection is the verb's own guard."* Re-measured: `plan_import({'a':{}}, 'a', 'g', {'a':'main'}, set(), {'main':{}}, 'main')` → `"'a' is the output and stays"`, and the "nothing to import" branch is reachable only with `source_output=""` (a hand-broken `graph.json`), which the spec no longer claims the dialog can produce. The existing test `test_replacing_the_only_pass_rejects` asserting `"output" in plan` now matches what the spec names, which is what made it a PARTIAL. The dialog half holds too: `_draw_entry_points` wraps the output row's combo in `begin_disabled(is_output)` with a `"the output, stays"` caption. |
| F3 — break-table row 1 overcounts (3 claimed, 2 red) | **CLOSED** | The spec's falsifier table row now reads `` `test_pass_import.py` (2 tests) ``. |
| Item 3 PARTIAL on D4(d) | **CLOSED** | Folded into F2 above: the item's message and the spec's now agree. |
| Item 19 PARTIAL — the smoke never draws a multi-tile run | **CLOSED** | `scripts/smoke.py` stamps `("paint", "seed", "df")`. Measured over `pass_graph.group_runs` in both of Radiance Cascades' orders: uncompiled (name-sorted) `[['cascade'],['composite'],['df'],['jfa'],['paint','seed']]`, compiled (topological) `[['paint','seed'],['jfa'],['df'],['cascade'],['composite']]` — grouped run lengths `[1, 2]` and `[2, 1]`. The old two-name stamp gave `[1, 1]` in both orders, which is the gap the item named. And the smoke genuinely ran rather than skipping: `make smoke` exits 0 with `smoke: OK (200 frames, 7 documents)` and three `Document 'Radiance Cascades' saved` lines, one per `set_pass_group`. Verification 19's own text still names `paint` and `df`; the Deviations entry carries the correction ("`paint` and `seed` are adjacent in both of Radiance Cascades' strip orders and `df` is apart from them, so one frame draws a two-tile run and a split run"), which is the repo's recorded shape for a landed deviation. |

Nothing in the three reports is DECLINED — every finding was acted on.

## Re-derivation 1: the loop

Derived from the code, then driven through a real headless `App`.

`plan_import` validates a handover pair only against `host_wiring[host_pass][uniform] in fed`
— that the host sampler reads a pass being replaced. Nothing in that check looks at whether
`host_pass` is itself upstream of the bundle. The shape that closes the ring: two source entry
points fed by two host passes where one reads the other. Then the bundle reads both host
passes, and the default handover makes the downstream host pass read the bundle.

At `7ac0b30` the guard is in `pass_import.plan_import`, after `handed` is assembled:

```python
merged: dict[str, dict[str, str]] = {
    name: dict(reads) for name, reads in host_wiring.items()
}
for host_pass, rows in handed.items():
    merged[host_pass].update(rows)
for name in copied:
    merged[renames[name]] = dict(sources.get(renames[name], {}))
_, errors = plan_passes(merged)
if errors:
    return f"a loop through '{errors[0].pass_name}': uncheck a handover"
```

**Pure probe**, the report's exact shape — host `main → mid` with `mid` the output, source
entry points `a`, `b` and output `mix`, handovers left at the dialog's D6 default
(`readers_of(host_wiring, fed)` unioned over the fed passes):

```
source entry points: ['a', 'b']
DIALOG DEFAULT subs: {'a': 'main', 'b': 'mid'} handovers: [('mid', 'u_main')]
plan_import -> "a loop through 'fx_mix': uncheck a handover"

wiring the unchecked import would leave: {'main': {}, 'mid': {'u_main': 'fx_mix'},
                                          'fx_mix': {'u_a': 'main', 'u_b': 'mid'}}
plan_passes errors on it: [('fx_mix', 'passes form a cycle: fx_mix -> mid -> fx_mix. ...'),
                           ('mid', 'pass is not ordered: an input is on a cycle.')]
```

Two controls, because a cycle check is the kind of guard that can close the defect by
over-rejecting: with the loop-closing handover unchecked the same selection returns an
`ImportPlan`; and a handover onto `mid` when only `a` is fed returns an `ImportPlan` with
`handovers={'mid': {'u_main': 'fx_mix'}}` — the legitimate insertion still plans.

**Headless-App probe**, the `tests/conftest.py::app` fixture's route, frames pumped the way
`tests/test_import_dialog.py::_pump` pumps them (`imgui.new_frame()` →
`import_passes.draw_import_passes(app)` → `imgui.end_frame()`). A `mid` pass built as a bare
`Pass` over the starter document and made the output, a two-root source document synced from
disk, both compiled:

```
HOST wiring: {'main': {}, 'mid': {'u_main': 'main'}} output: mid
SOURCE wiring: {'bg': {}, 'fg': {}, 'mix': {'u_bg': 'bg', 'u_fg': 'fg'}} output: mix
DIALOG DEFAULT subs: {'fg': 'main', 'bg': 'mid'} handovers: [('mid', 'u_main')]
draft.rejection after 3 pumped frames: "a loop through 'fx_mix': uncheck a handover"
verb result: ImportResult(error="a loop through 'fx_mix': uncheck a handover", notes=())
host passes after the rejected verb: ['main', 'mid']
rejection with the handover unchecked: ''
```

So the rejection reaches `ImportDraft.rejection` in a drawn frame — which is what
`begin_disabled(bool(draft.rejection))` reads, so the Import button is dead — the programmatic
route rejects with the same message, and the host gained no pass and no file
(`passes_dir.glob("fx_*")` empty). Unchecking the one handover clears the rejection in the next
pumped frame, so the message's advice is the fix it names.

**Gate broken.** In the pinned worktree, deleting the three-line `plan_passes` rejection:
`tests/test_pass_import.py::test_a_handover_onto_a_feeding_pass_is_a_loop_and_rejects` →
`1 failed, 9 passed`, failing on

```
AssertionError: ImportPlan(renames={'mix': 'bloom_mix'}, sources={'bloom_mix': {'u_fg': 'a', 'u_bg': 'b'}},
                           output='bloom_mix', handovers={'b': {'u_a': 'bloom_mix'}}, becomes_output=True)
```

which is the cycle itself (`b` reads `bloom_mix`, `bloom_mix` reads `b`) coming back as a valid
plan. Restored; the worktree's 091 suite is `127 passed`.

One thing the fix does *not* cover, and correctly: `Document.graph_errors` is still surfaced
nowhere (`grep -rn graph_errors shaderbox/` hits only `document.py`'s own field, getter and two
writers). F1 named that as the reason the cycle was silent rather than as a finding of its own,
and a cycle is now unreachable through the feature; a hand-edited `graph.json` can still
produce one quietly, which is a pre-091 property of the renderer and not this feature's.

## Re-derivation 2: the torn import

`ProjectSession.import_passes` end to end at `7ac0b30`, every write in order and what each
failure path leaves:

1. `ui_documents.get(document_id)` → returns `ImportResult(error=…)` on a miss. **No write.**
2. `compile_pending_passes(source_document)`, `compile_pending_passes(host)` — compile only;
   `Pass.compile` swallows every exception into `compile_unit.errors` and returns, so neither
   raises. **No write to disk or to the graph.**
3. The broken-host-pass handover guard → returns. **No write.**
4. `plan_import(...)` → returns on a rejection (this is where the loop above lands). **No write.**
5. **The copy phase**, new in `7ac0b30`: every copied pass's `uniform_values` run through
   `_copied_uniform_value` into a local `copies: dict[str, dict[str, Any]]`, inside
   `try / except Exception`. This is where a bound `Image`/`Video` whose file is gone raises
   (`FileNotFoundError` from PIL, `ValueError` from `Video`). On failure: every value already
   copied is `try_to_release`d, and it returns
   `ImportResult(error=f"could not copy a bound asset: {e}")`. **Nothing has been written** —
   no pass file, no `host.passes` entry, no graph change. The raise no longer escapes into the
   frame loop; it comes back as `ImportResult.error`, which is what F2 asked for.
6. **The write loop**, inside `try / except OSError`, per copied pass: `path.write_text` (the
   file, appended to `written`), `ShaderSource.load(path)`, `Pass(...)`,
   `render_pass.uniform_values = copies.pop(source_name)` (the pre-made copy, so the fragile
   work is behind it), `compile()`, the plan's `PassSource` rows, then `built[host_name]` and
   `entries[host_name]` — **two local dicts, not `host.passes`**. On an `OSError`: every
   `built` pass is `release()`d, every un-popped `copies` value is `try_to_release`d, every
   `written` path is `unlink(missing_ok=True)`d, and it returns
   `ImportResult(error=f"could not write a pass file: {e}")`. **The host is untouched** — the
   orphan-pass resurrection F2 described needed `host.passes[host_name] = render_pass` inside
   the loop, and that assignment is gone. `OSError` is the right width here: the loop's only
   other raisers are `ShaderSource.load` (`read_text`/`lstat`, `OSError` subclasses) and
   `compile()`, which cannot raise at all.
7. `host.passes.update(built)` — the first host mutation, after the loop cannot fail.
8. `host.graph = host.graph.with_passes(...)`, then the handover rows
   (`values[uniform] = PassSource(read)`, no `try_to_release` — the sampler reads the fed pass
   so its value is a source, never a texture), then the `ui_uniforms` `setdefault` merge.
9. One `save_ui_document(ui_document)` — the host's, never the source's.

**Probed, both failure paths**, against a real headless `App`, comparing
`document.passes` / `graph.passes` / `graph.output` / the `passes/` glob before and after:

```
COPY PHASE (a shipped example's bound u_image.png renamed away):
  result: could not copy a bound asset: [Errno 2] No such file or directory: '.../u_image.png'
  before: {'document.passes': [], 'graph.passes': [], 'graph.output': 'main', 'files': []}
  after : {'document.passes': [], 'graph.passes': [], 'graph.output': 'main', 'files': []}

WRITE PHASE (Path.write_text raising OSError on the 6th of 6 passes):
  write_text calls: ['g_cascade...', 'g_composite...', 'g_df...', 'g_jfa...', 'g_paint...', 'g_seed...']
  result: could not write a pass file: injected: disk full
  before: {'document.passes': [], 'graph.passes': [], 'graph.output': 'main', 'files': []}
  after : {'document.passes': [], 'graph.passes': [], 'graph.output': 'main', 'files': []}
  after a later save: {..., 'files': []}       # the resurrection F2 described does not happen
```

Five files were written and unwound. The later `save_ui_document` is the step F2 showed
committing the orphans; it now finds nothing to commit.

**No double-release on the unwind.** Instrumenting `Pass.release` and
`project_session.try_to_release` through the write-phase failure: `Pass.release count: 5
unique: 5` (the five `built` passes, once each, = n-1 for a 6-pass source) and zero non-`None`
`try_to_release` calls — the `copies.pop` in the loop is what keeps the consumed values out of
the unwind's second sweep.

**Gate broken.** In the pinned worktree, reverting to the pre-fix shape — the copy phase
deleted, `_copied_uniform_value` called inside the write loop, the `except OSError` unwind
removed — `tests/test_pass_verbs.py::test_a_torn_import_writes_nothing` goes red with
`FileNotFoundError: [Errno 2] No such file or directory: '.../u_image.png'`, i.e. the
exception escaping the verb uncaught, which is the defect. Restored.

## False trails

Settled by the three reports and not re-checked here: the `##` id tail on this imgui build
(ids hash the whole string, so the handover checkboxes do not collide); `_copied_uniform_value`
over the whole value domain, including `Image(value.texture)` being an independent copy and the
empty-path case the D5 prohibition does not cover; `compile_pending_passes` disturbing neither
the Examples popup nor the planned set; the group outline's clip, paint order and font; the
`always_use_window_padding`/`borders` equivalence; the tab read-back's self-healing; the
render-count-per-popup-state matrix; `set_pass(..., group)` through the tool in every D9 case;
`plan_import`'s other ten edge cases; the source being unmutated on disk and in memory;
`ui_uniforms` merging against the save-funnel prune; `ImportDraft` not being persisted;
`PassEntry.group`'s per-key salvage; the local layout constants not being theme-token
violations; the grid duplication between `examples.py` and `import_passes.py`; `graph_errors`
having no UI consumer (pre-091).

Two from those lists are now no longer merely-fine but actively resolved, worth naming so they
are not read as still-open: the handover-site `try_to_release` that code-correctness false
trail 2 measured as dead defensive code is **deleted** at `7ac0b30`, and the D6 spec paragraph
states why ("a host sampler that READS the fed pass, so its value is a source … never a bound
texture"); and the `offered_entry_points` / `host_readers_of` housing judgements, which the
architecture report raised as sub-findings, are both acted on.

New to this review, and not a finding: the source document's `mix` pass in the headless loop
probe resolved `u_fg`/`u_bg` with **no explicit rows**, by the name rule alone, once it
compiled — so the two-root loop shape needs nothing hand-wired and is as reachable as F1 said.
The probe's first attempt looked like a false negative (`draft.rejection == ''`) and was my
fixture, not the code: the shaders declared `in vec2 v_uv; out vec4 f;` where this repo's
vertex stage exports `vs_uv` / `fs_color`, so `mix` never linked and answered an empty wiring.
Worth recording for the next probe author — a pass that fails to compile reads as a root, which
silently removes the edge a wiring probe is about.
