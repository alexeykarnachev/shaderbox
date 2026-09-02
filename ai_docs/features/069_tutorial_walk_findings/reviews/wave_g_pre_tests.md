# W-G pre-implementation review: verification & blast radius

Anchor: `60_wave_g_scripting.md` at the working tree, code at `0ce84f8`. Every count below was run.

## Verdict

| Dimension | Verdict |
|---|---|
| Test-budget accuracy | **PASS.** Every line/test/call-site figure the spec states is exact. |
| Falsifiability | **PARTIAL.** Eleven of thirteen named tests go red under their stated mutation. Two do not: the orphan-console test cites an idiom the repo does not have, and the `set_uniform` gate test contradicts the fixture it must run on. |
| Seam coverage | **PASS on the seven, FAIL on the Files-touched line.** All seven `.render_pass` arguments are real and each has a test that fails if left unchanged. The Files-touched row says "four". |
| On-disk claims | **PASS.** Verified independently: nine `document.json` files, every persisted value `[]` or absent, no `projects/dev/documents/*/scripts/`. |
| Blast radius | **FAIL.** Three readers outside Files touched, one of them inside `make gates`. |

## Findings

### 1. `scripts/smoke.py:317` calls `set_uniform_stopped` with the old arity, and smoke is a gate

`make gates` runs check -> test -> **smoke**. `scripts/smoke.py` is not in Files touched, and it holds
the 048 stopped-skip canary:

```
scripts/smoke.py:317:  app.session.set_uniform_stopped("script_document", "u_a", True)
scripts/smoke.py:306:  driven = engine.script_driven_uniforms("script_document")
scripts/smoke.py:322:  assert "u_a" in engine.script_driven_uniforms("script_document")
```

Under item 6 `set_uniform_stopped` becomes `(document_id, pass_name, name, stopped)`, so `:317` is a
`TypeError` at runtime, and `:306`/`:322` compare a name against a set of pairs, so the assertion at
`:306` fails before `:317` is even reached. The spec declares gates green in no section, but a wave
whose Files-touched list omits a gate file cannot reach a green gate.

Fix, pasteable: add a row `| `scripts/smoke.py` | the 048 stopped-skip canary passes the pass name to
`set_uniform_stopped` and compares `("main", "u_a")` against the pair-shaped driven set. |` to Files
touched, and state in § Tests that the smoke canary is the third place (beside `test_script_engine.py`
and the dogfood check) where the pair re-key must land or `make gates` fails at the smoke step.

### 2. `scripts/dogfood/verify_script_engine.py` asserts the driven set by name, twice

Not in Files touched. Two assertions break on the pair shape:

```
:96   driven = h.session.get_script_driven_uniforms("scripted")
:97   assert driven == {"u_wave"}, ...
:145  assert h.session.script_engine.script_driven_uniforms("scripted") == set()
```

`:97` is an equality against a name set and goes red the moment `get_script_driven_uniforms` returns
pairs. (`:145` compares against the empty set and survives by accident, which is worse: it keeps
passing while its sibling is broken.) The prompt names `scripts/dogfood/` as a generator over live
code, and this is the file that is.

Fix: add `| `scripts/dogfood/verify_script_engine.py` | the driven-set assertion becomes
`{("main", "u_wave")}`; the drop_document assertion is shape-agnostic and needs no edit. |` to Files
touched.

### 3. `test_the_orphan_warning_no_longer_reaches_the_console` cites an idiom that does not exist

The spec: "Drive an orphan key with `caplog` (loguru's propagation shim, **as the repo's other log
assertions use**)". There are none. Over the whole repo:

```
$ grep -rn "caplog" . --include=*.py --include=*.toml --include=*.cfg | grep -v .venv
(no output)
$ grep -rn "logger\|loguru\|propagate\|LogCapture" tests/*.py
(no output)
```

Zero `caplog`, zero loguru references, zero logger assertions in the suite; `tests/conftest.py`
contains no logging setup. This repo logs through loguru, which does **not** propagate to the stdlib
`logging` tree that pytest's `caplog` fixture reads, so a `caplog`-based test asserting "no WARNING
record" passes vacuously whether or not the `logger.warning` at `engine.py:587` is still there. That
is an unwired mechanism: the test as described cannot go red under its own stated falsifier ("keep the
`logger.warning`").

Fix: replace the test's mechanism with one that reads a real sink. Paste: "`test_the_orphan_warning_no_longer_reaches_the_console`: install a loguru sink for the duration
(`handle = logger.add(records.append, level='WARNING')`, removed in a fixture teardown), drive an
orphan key, and assert the sink captured nothing from `shaderbox.scripting.engine` while
`script_status().soft_errors` carries the row. The repo has no existing log-assertion idiom, so this
test introduces one; `caplog` is not it, because loguru does not propagate into the stdlib logging
tree pytest reads and a `caplog` assertion would pass with the `logger.warning` still in place."

### 4. The `set_uniform` gate change contradicts the fixture `test_script_driven_reject.py` is built on

Item 7 says the gate "asks `(output_pass_name, name) in driven`". The output pass name is
`document.graph.output_pass` (`document.py:286`), reached through
`self._get_ui_documents()[document_id].document`. But the gate at `backend.py:879` fires **before**
`target` is resolved at `:887`, and the test's stub says so explicitly:

```
tests/test_script_driven_reject.py:22-24
    ui_documents = {
        "n0": object()
    }  # the document only needs to EXIST; the reject returns before .document
```

`test_set_uniform_rejects_script_driven` passes a bare `object()`. Under the change the gate must
touch `.document.graph.output_pass`, so that test raises `AttributeError` before reaching its
assertion. The spec says of this file: "the assertions ... are unchanged in meaning", and names only
"both fakes change shape". The fixture, not the assertion, is what breaks.

This also means the spec's added test ("`set_uniform` does NOT reject a uniform driven on a NON-output
pass") cannot be written against the existing `_stub` helper either.

Fix: paste into the `test_script_driven_reject.py` entry — "Both stubs gain a real
`document.graph.output_pass`, because the gate now resolves the output pass name before asking the
driven set and the current `object()` placeholder (whose comment reads 'the reject returns before
.document') stops being valid. Either move the driven-set gate below the `target =` resolution at
`backend.py:887`, or hoist the document lookup above it; the spec takes the second, so the stub grows
`document=SimpleNamespace(graph=SimpleNamespace(output_pass='main'), render_pass=...)`."

### 5. Open question 3's claimed imgui id collision does not exist

The spec asserts "two passes declaring the same uniform name currently produce one id for two rows,
which is an imgui state bug the pass prefix fixes as a side effect."

The Document panel draws exactly one pass per frame. `App.panel_pass` (`app.py:606-617`) returns a
single `Pass`; `tabs/document.py:257-263` loops `sorted_hashes` from that one pass's
`active_uniform_hashes` and calls `draw_ui_uniform` per row. Two passes' rows of the same uniform name
are therefore never submitted in the same frame, and `play_stop_toggle`'s
`small_button(f"{label}##play_stop_u_{name}")` (`ui_primitives.py:208`) never sees a duplicate id.

The id change is still correct (it keeps per-pass imgui state, e.g. the button's held/hovered state,
from carrying across a panel-pass switch), but the justification as written is a bug claim that no
line supports, and no test or manual step pins it. Under the "unwired mechanisms count as absent"
rule, a fix for a non-existent collision has no falsifier.

Fix: rewrite open question 3's default as — "**no concern**, the id is invisible. The panel draws one
pass per frame (`app.py::panel_pass`, `tabs/document.py:257`), so two same-named rows are never
submitted together and there is no live collision to fix; the prefix keeps a row's imgui state from
carrying across a panel-pass switch, which is a correctness nicety, not a bug fix. Named here so a
reviewer does not read the id change as either cosmetic or as closing a defect."

### 6. `Document.has_feedback` over `_feedback` is render-history, not a document property

Item 11 gates the Clear button on "`Document` exposes that as a property over its `_feedback` map,
`has_feedback: bool`."

`self._feedback` is populated **lazily inside `render()`**: the only writer is `_feedback_canvas`
(`document.py:410`), reached from `render_document` at `:478` when a pass's graph input names the pass
itself. So on a freshly loaded feedback document, `_feedback` is empty until the first frame that
actually draws that pass — the button is absent on frame 0 and appears on frame 1. Worse,
`Document.release()` clears `_feedback` (`:301-302`) and `drop_feedback` pops entries (`:382-383`), so
the button can vanish mid-session on a target-format change.

The document-level fact is in the graph: a pass whose `graph.passes[name].inputs` maps some uniform to
`name` itself, which is the exact condition `:474` tests.

Fix, pasteable: "`has_feedback` is a property over the GRAPH, not over `_feedback`:
`any(source == name for name, entry in self.graph.passes.items() for source in entry.inputs.values())`.
`_feedback` is populated lazily by `_feedback_canvas` during `render()` and cleared by `release()` /
`drop_feedback`, so a property over it reads False on a feedback document's first frame and can flip
back mid-session; the graph is the document's own statement of what feeds back and is stable from
load."

### 7. The Files-touched row says four `.render_pass` seams; the spec's own premises table says seven

`shaderbox/project_session.py`'s row reads "the **four** `.render_pass` seams pass the document".
Premises row 2 correctly says seven and names them. Verified at `0ce84f8`:

```
$ git show 0ce84f8:shaderbox/project_session.py | grep -n render_pass
:430  reload   (_resolve_scripts)
:511  reload   (_load_one_document_from_disk)
:555  tick_export (_make_export_isolation closure)
:577  reload   (reload_scripts)
:599  tick     (ProjectSession.tick)
:683  reload   (write_script_source)
:687  dry_run  (write_script_source)
```
plus `:328` / `:482` (`.source.path` on delete, not engine arguments), `:640`
(`_scriptable_uniforms_for`, item 9's own seam), and `:514/:773-826` (`render_pass` as a local, not
`.render_pass`). Seven engine arguments across five methods, exactly as the premises table says.

Seam-to-test mapping, all covered: `:599` and `:555` by `test_script_engine_gl.py`'s tick/export
calls; `:683`/`:687` by `test_script_dry_run.py` and `test_copilot_script_tools.py`'s `write_script`;
`:430` and `:511` and `:577` are the three `reload` paths — `test_script_engine.py` calls `reload` at
most of its 77 sites and `scripts/dogfood/verify_script_engine.py:92` exercises `reload_scripts`
(`:577`) against a live session. The two the spec singles out as prose-sweep misses (`:430`, `:511`)
are covered structurally rather than by a named test: `reload`'s parameter type changes, so a site
left passing `.render_pass` is a pyright error at `make check`, which is the first gate step. That is
a real falsifier and worth stating.

Fix: change the Files-touched row to "the **seven** `.render_pass` seams pass the document", and add
one sentence to § Tests: "The three `reload` seams (`:430`, `:511`, `:577`) have no dedicated test;
their falsifier is `make check` — `reload`'s parameter becomes `ScriptTarget`, so a site still passing
`.render_pass` (a `Pass`) fails pyright at the first gate step."

### 8. `_pass_views`'s driven filter — the wave's headline fix — has no existing test to edit

The spec's `test_copilot_script_tools.py` entry says the added test goes in that file. It has zero
`_pass_views` coverage: its thirteen tests all build `ScriptWriteResult` / `ScriptReadResult` payloads
and assert the rendered strings. The one place the `<driven by script.py>` marker is asserted is
`tests/test_working_set.py:127,137` — and that test builds a `WorkingSetView` **by hand** with a
pre-made `uniforms=[...]` list, with its own comment saying so:

```
tests/test_working_set.py:119-120
    # marker, not a phantom value (feature 043 D6/D7a). The marker itself is built backend-side
    # (_format_uniforms); here we verify the working-set RENDER carries the script fields it's given.
```

So `_format_uniforms`'s `elif u.name in driven` branch (`backend.py:257`) and `_pass_views`'s
document-scoped `driven` binding (`:738`) are untested today, in any file. The spec's added test is
therefore not "one added test" to an existing group but the first test of that code path, and it needs
a `_pass_views` fixture (a two-pass `Document` with a graph) that no test file currently has.

`tests/test_working_set.py` appears in no section of the spec — not Files touched, not Tests, not the
premises table.

Fix: paste into the `test_copilot_script_tools.py` entry — "The added `_pass_views` test is the FIRST
test of that code path: `_format_uniforms`'s driven branch (`backend.py:257`) and `_pass_views`'s
driven binding (`:738`) have no coverage today. `tests/test_working_set.py:127` asserts the marker
STRING but builds `WorkingSetView.uniforms` by hand (its own comment: 'the marker itself is built
backend-side'), so it neither exercises nor breaks on the filter. The new test needs a two-pass
`Document` fixture with a graph, which this file does not have; state where it comes from."

### 9. `test_a_stale_string_stopped_set_drops_to_empty` must construct its own stale file — the spec says this, and it is right

Flagged as verified rather than as a defect. The spec's § 13 claim ("no example persists a non-empty
list, so this test must construct one") is correct, and I re-ran the salvage rather than trusting it:

```
$ uv run python -c "... drop_invalid(S, {'stopped_uniforms': ['u_x'], 'all_stopped': True}, ...)"
WARNING | Ignoring invalid document 'x'.stopped_uniforms (1 error(s))
after salvage: {'all_stopped': True}
constructed: stopped_uniforms=[] all_stopped=True
```

and the mixed case the spec calls out:

```
mixed in:  [{"pass_name":"paint","name":"u_a"}, "u_x"]
mixed after: {'all_stopped': True}     <- the valid pair is lost too
```

Both the drop-to-`[]` and the whole-list (not element-level) clarification are exactly as the spec
states, and `all_stopped` survives. The test as described goes red under its falsifier (migration code
reinterpreting a bare string). No change needed.

### 10. Manual verification step 3 has no falsifiable observation for the re-entry rule

Steps 1, 2, 4, 5, 6 and 7 each name something a maintainer can see be wrong. Step 3's last clause —
"Drag off the canvas edge mid-stroke and back on: the stroke ends at the edge and restarts where the
cursor re-entered, with no straight line across the gap" — is the only one whose mechanism
(`script_mouse_inside`, item 10) has no other check, and "no straight line across the gap" is only
observable if the user drags a visibly long distance off-canvas. Item 10 also changes existing
behavior that `context.py:9-10` documents ("Outside the canvas the live value clamps to the last
in-bounds position") by adding an else branch that did not exist — I confirmed `ui.py:658-659` has no
else today.

Fix: paste as step 3's last clause — "Drag off the LEFT edge mid-stroke, move to the RIGHT edge
off-canvas, and re-enter there: the stroke must end at the left edge and restart at the right, with no
stamp along the line between them. A full-width streak is the `script_mouse_inside` flag being unread.
Note this adds an else branch to the hit test that does not exist at HEAD, so `context.py`'s
'clamps to the last in-bounds position' comment is rewritten in the same edit."

## False trails

- `test_script_engine.py`'s budget: 1088 lines / 61 tests / 77 engine call sites — all three exact; the 62 `uniform_values[` assertion sites and 9 name-keyed `script_driven_uniforms` assertions match "every test body edited at the assertion line, roughly a dozen edited more deeply".
- Every other test file's engine-call count is exact: gl 15, dry_run 12, export_wiring 1, copilot_script_tools 0, driven_reject 0, api_doc 0.
- `test_script_engine_gl.py` shows 34 `render_pass` hits vs the spec's "fifteen call sites" — the other 19 are setup and assertions on a real one-pass `Document`, which stay valid; the spec's "most assertions read unchanged" is right.
- The on-disk table is correct file by file: five examples `[]`, `1c4f8a20` and `77a84d27` key-absent, both dev documents `[]`. `projects/dev/documents/` has no `scripts/`.
- `projects/dev/trash/` DOES hold `script.py` files (five dirs), but trash is untracked (`git ls-files projects/dev` returns 8 paths, none under trash), so "the wave changes no bytes under `projects/dev`" holds for the repo.
- `EngineNode` has no reader outside `scripting/engine.py`, `scripting/__init__.py` and one test comment — the retirement's blast radius is as small as the spec says.
- `MouseState` / `EXPORT_MOUSE` readers are exactly the eight files the spec accounts for; no unnamed reader.
- `script_stub_for` and `_scriptable_uniforms_for` have no reader outside `project_session.py`, `scripting/`, and `test_script_engine.py`.
- The api_doc pins are accurate: `test_summary_lists_every_ctx_field_and_the_mouse_subfields` loops `MouseState.__dataclass_fields__` (`:104-105`) and the freeze caveat is a substring assertion (`:112-116`) that survives a longer gloss.
- Command premises hold: `_STANDALONE_KEYS` is F1-F12 (`commands.py:288`), `chord_needs_modifier` exempts them (`:316`), F1 and F8 are existing bare-F bindings, F6 collides with nothing.
- The `app.py:1414-1435` and `tabs/document.py:253` refutations are both correct: the wrappers are `:1489`/`:1496`, and `tabs/document.py` has exactly one stopped call (`:333`, `set_document_all_stopped`, arity unchanged).
- The `conventions.md` line cites (`:289`, `:308`, `:314-316`, `:705`, `:711`), the 065 D12 quote and the 068 D7 retraction all read as the spec quotes them.

## Coverage statement

Opened at `0ce84f8`: all seven named test files (line/test/call-site counts run, not estimated),
`project_session.py`, `scripting/engine.py`, `context.py`, `errors.py`, `api_doc.py`,
`copilot/backend.py`, `widgets/uniform.py`, `tabs/document.py`, `tabs/code.py`, `ui.py`,
`ui_primitives.py`, `ui_models.py`, `model_salvage.py`, `document.py`, `commands.py`, `app.py`,
`tests/test_working_set.py`, `tests/conftest.py`, `scripts/smoke.py`,
`scripts/dogfood/verify_script_engine.py`, `scripts/dogfood/harness.py`.

Ran rather than read: the nine `document.json` files parsed for `stopped_uniforms`; the salvage
behaviour executed against a real `list[StoppedKey]` field (both the stale-string and the mixed-list
case); `git ls-files projects/dev`; repo-wide greps for `caplog`, `EngineNode`, `last_driven`,
`stopped_uniforms`, `MouseState`, `EXPORT_MOUSE`, `_scriptable_uniforms_for`, `script_stub_for`, and
the five stopped/driven session methods.

Premises spot-check: 24 of the 40 rows opened at the cited symbol. All 24 read as the spec states,
including all six refutations and the two the spec calls out as work-changing (the seven seams, the
zero-byte hand edit). The drafter's corrections are accurate; the defects above are in what the spec
does NOT cite, not in what it does.

---

# Round 2 (closure)

Narrow re-check of the ten round-1 findings against the folded spec. Code re-read at `d2ade88`
(W-F landed since round 1); the lines this round turns on are untouched by W-F.

**Overall: PASS.** Ten of ten closed. One follow-on gap is filed as finding 8a below, created by the
fold rather than surviving it, and it is the only thing standing between this spec and a green gate.

| # | Round-1 finding | Verdict |
|---|---|---|
| 1 | `scripts/smoke.py` arity + gate | **CLOSED** |
| 2 | `scripts/dogfood/verify_script_engine.py` name assertion | **CLOSED** |
| 3 | orphan-console test cites a non-existent `caplog` idiom | **CLOSED** |
| 4 | gate change contradicts the `object()` stub | **CLOSED** |
| 5 | open question 3's imgui id collision does not exist | **CLOSED** |
| 6 | `has_feedback` over `_feedback` | **CLOSED** |
| 7 | Files-touched says four seams, premises say seven | **CLOSED** |
| 8 | `_pass_views` has no existing test to edit | **CLOSED**, with follow-on 8a |
| 9 | stale-string test constructs its own file | **CLOSED** (was verified, not a defect) |
| 10 | manual step 3 has no falsifiable observation | **CLOSED** |

## Per-finding closure

### 1 — CLOSED

§ Files touched now carries the row:

> `| scripts/smoke.py | the 048 stopped-skip canary passes the pass name to set_uniform_stopped and compares ("main", "u_a") against the pair-shaped driven set. **This file is inside `make gates`.** |`

and § Tests § "The three `reload` seams and the two script files inside the gates" states the failure
order I filed: "`:306`'s assertion fails before `:317`'s `TypeError` is even reached, and the smoke
step of `make gates` goes red." The exact edits are named (`:306`, `:317`, `:322`), the gate
membership is bolded, and the "third place beside `test_script_engine.py` and the dogfood check"
framing survived. Nothing left open.

### 2 — CLOSED

§ Files touched:

> `| scripts/dogfood/verify_script_engine.py | the driven-set assertion becomes {("main", "u_wave")}; the drop_document assertion compares against the empty set and is shape-agnostic. |`

and § Tests keeps the point that mattered — the `:145` sibling "would survive by accident, which is
worse: it keeps passing while `:97` is broken." Both gate files now have rows with exact edits, as
asked.

### 3 — CLOSED, mechanism confirmed

The test section now reads:

> Install a loguru sink for the duration (`handle = logger.add(records.append, level="WARNING")`,
> removed in a fixture teardown), drive an orphan key, and assert the sink captured nothing from
> `shaderbox.scripting.engine`

**Confirmed it goes red with the `logger.warning` present.** The mechanism, end to end:
`engine.py:587` calls `logger.warning(...)` on the module-level loguru logger imported at `engine.py`'s
top. `logger.add(sink, level="WARNING")` registers `sink` as a loguru handler; loguru dispatches every
record at or above WARNING to every registered handler synchronously, in the emitting thread, at the
call. `records.append` therefore receives one `loguru._handler.Message` (a `str` subclass carrying
`.record`) whose `record["name"]` is the emitting module's `__name__`, i.e. `shaderbox.scripting.engine`.
So with the `logger.warning` in place the assertion "captured nothing from `shaderbox.scripting.engine`"
sees exactly one matching record and fails. With it deleted, `records` stays empty and the assertion
holds. The test is falsifiable under its own stated mutation, which the `caplog` version was not.

The spec also states the reason in-line ("loguru ... does not propagate into the stdlib `logging` tree
pytest's `caplog` reads, so a `caplog` assertion would pass vacuously"), which is what stops the
idiom being re-introduced by the implementer. My round-1 grep result is preserved verbatim as the
evidence ("zero `caplog`, zero `loguru` references anywhere under `tests/`, and `conftest.py` sets up
no logging").

### 4 — CLOSED, stub shape confirmed against the gate's new position

Item 7 now moves the gate:

> The gate MOVES below the `target = self._get_ui_documents()[document_id].document` bind at `:887`
> and asks `(pass_name_of(target.render_pass.source.path), name) in driven`.

and the test section says both stubs grow:

> `SimpleNamespace(graph=SimpleNamespace(output_pass="main"), render_pass=SimpleNamespace(source=SimpleNamespace(path=Path("passes/main.frag.glsl"))))`

**The shapes agree.** The gate as now specified reads `target.render_pass.source.path` and nothing
else from the document, and the stub supplies exactly that chain. Resolved for real:

```
$ uv run python -c "from pathlib import Path; from shaderbox.paths import pass_name_of; print(repr(pass_name_of(Path('passes/main.frag.glsl'))))"
'main'
```

(`paths.py:26`, `shader_path.name[: -len(PASS_SHADER_SUFFIX)]` with `PASS_SHADER_SUFFIX = ".frag.glsl"`
at `:13`.) So the stub's pass name is `"main"` and matches a driven pair `("main", "u_wave")` — the
reject fires as the existing assertion expects, and the added non-output test (`("paint", "u_x")`
against an output of `main`) does not.

The `graph=SimpleNamespace(output_pass="main")` half of the stub is **unused** by the gate as
specified, since the gate went with `pass_name_of` rather than `graph.output_pass`. It is harmless
and consistent with `Document.render_pass` (`document.py:286` resolves through `graph.output_pass`),
so it is a redundancy, not a contradiction — and by the late-round rule, trimming it is a preference,
not a finding. The stale comment at `:22-24` is explicitly named for removal, which was the actual
defect.

### 5 — CLOSED

Open question 3 now states there is no live collision and names the settling line:

> `tabs/document.py:261-262` loops that one pass's `sorted_hashes` through the only `draw_ui_uniform`
> call site in the codebase, so two passes' rows of the same uniform name are never submitted in the
> same frame

Re-verified at `d2ade88`: `draw_ui_uniform` has one call site, inside the `sorted_hashes` loop under
`begin_child("ui_uniforms")`, over `app.panel_pass`'s single `Pass` (`app.py:606-617`). The fold also
records that the two reviewers split and which reading won, which is the right disposition for a
disputed claim. What the prefix buys is stated accurately (state not carrying across a panel-pass
switch, "a correctness nicety rather than a bug fix").

### 6 — CLOSED, and the replacement is verified

Item 11 now reads:

> `Document` exposes that as `has_feedback: bool` computed from the GRAPH, never from `_feedback`:
> `bool(plan_passes(self.graph)[0].feedback)`.

Confirmed the target exists and means what the spec says. `plan_passes` is `pass_graph.py:248` and
returns `tuple[PassPlan, list[GraphError]]`, so `[0]` is the plan. `PassPlan.feedback` is
`pass_graph.py:224`, `feedback: set[str]`, and the class docstring at `:217-219` defines it as exactly
the self-edge: "`reads` excludes the self-edge (that is `feedback`)". It is filled from the graph
alone, so it is stable from load and independent of render history.

The spec also names the failure mode I filed and adds one I did not: "`reset_feedback` ITSELF" empties
`_feedback`, "so a property over it would hide the button on an unrendered document and hide it again
the instant the user clicks it." That second half is correct — `document.py:363-366` clears
`_feedback` — and is a sharper statement of the defect than my round-1 version.

### 7 — CLOSED

§ Files touched now reads "the **seven** `.render_pass` seams pass the document (`:430`, `:511`,
`:555`, `:577`, `:599`, `:683`, `:687`)", matching the premises table. § Tests gained the section I
asked for:

> Their falsifier is **`make check`**, the first gate step: `reload`'s parameter becomes
> `ScriptTarget`, so a site still passing `.render_pass` (a `Pass`) is a pyright error.

That is a real falsifier for the three untested `reload` seams, and the spec draws the conclusion that
makes it load-bearing ("a prose-driven sweep that fixes only the three seams the parent names still
cannot ship").

### 8 — CLOSED, with follow-on 8a

The test section now states it is the first test of that path, accounts for `test_working_set.py`
correctly (asserts the marker string over a hand-built list, needs no edit) and quotes that file's own
comment as the reason. That closes the finding as filed.

### 8a — NEW, created by the fold: the sourced fixture makes the new test GL-gated

The fold answers "where does the two-pass fixture come from" with:

> it takes the `_document` helper from `tests/test_document_graph.py`, which already builds exactly
> that ... lifted into a shared helper or imported directly.

`_document` (`tests/test_document_graph.py:79-98`) takes `gl: moderngl.Context`, constructs real
`Pass` objects, calls `render_pass.compile()` and asserts the compile is clean. Its context comes from
that file's module-scoped `gl_ctx` fixture (`:64-74`), which **skips** when no standalone context is
available:

```
tests/test_document_graph.py:72-74
        ctx = moderngl.create_standalone_context()
    except Exception as e:
        pytest.skip(f"no standalone GL context available: {e}")
```

`tests/test_copilot_script_tools.py` is GL-free by construction: its imports are
`ScriptView`/`ScriptWriteResult`, the tool registry and `tests/_caps.minimal_caps`, and every test
drives the tool layer over fake caps. Importing `_document` moves the wave's headline test behind a GL
skip, and a skipped test is not a pass — the same rule `CLAUDE.md` states for the smoke step.

It is also unnecessary. `_pass_views` reads only `document.graph`, `document.passes`,
`render_pass.source.text`, `render_pass.compile_unit.errors`, `render_pass.get_active_uniforms()` and
`render_pass.uniform_values` (`backend.py:730-757`, `_format_uniforms` at `:241-262`) — every one
satisfiable by a `SimpleNamespace`, which is the idiom `tests/test_script_driven_reject.py` already
uses on this exact backend method (`__get__`-bound onto a light stub, per its own docstring: "the
reject branch fires before any GL, so the real set_uniform method is bound onto a light stub").

Fix, pasteable, replacing the last sentence of the `test_copilot_script_tools.py` entry: "The two-pass
fixture is a `SimpleNamespace` document, not `tests/test_document_graph.py`'s `_document`: that helper
takes a `moderngl.Context` and compiles, and its `gl_ctx` fixture skips without a standalone context,
which would put the wave's headline test behind a GL skip in a file that is GL-free by construction.
`_pass_views` reads only `graph`, `passes`, `source.text`, `compile_unit.errors`,
`get_active_uniforms()` and `uniform_values`, so a stub supplies all of them — the same
`__get__`-onto-a-light-stub idiom `tests/test_script_driven_reject.py` already uses on this backend."

### 9 — CLOSED

Filed as verified, not as a defect; the fold records it that way ("Filed as verified, not as a defect;
no change"). My round-1 execution of the salvage (both the stale-string and the mixed-list case) stands.

### 10 — CLOSED

Manual step 3 is now the traverse I specified:

> Then drag off the LEFT edge mid-stroke, move to the RIGHT edge off-canvas, and re-enter there: the
> stroke must end at the left edge and restart at the right, with no stamp along the line between
> them. A full-width streak is the `script_mouse_inside` flag being unread.

and item 10 states the else branch is new, that `context.py:8-10`'s "clamps to the last in-bounds
position" comment is rewritten in the same edit, and adds two cases I did not file: the else must
cover both `hit is None` and `hit[2] is False`, and `script_mouse_inside`/`down` clear at the top of
the with-document branch so a document closed mid-drag cannot latch `down=True`. Both are correct
against `ui.py`'s current shape (no else at HEAD).

## Re-verified on-disk claims (the fold changed them)

The disk enumeration moved from nine files to eleven, on the design reviewer's finding 5. Checked
independently rather than relying on either reviewer:

```
$ git ls-files | grep -c "document.json$"
11
```

Seven under `shaderbox/resources/document_examples/`, two under `projects/documents/`
(`1901ab60-...`, `307598da-...`), two under `projects/dev/documents/`. Both newly-named files parse
with `"stopped_uniforms": []`. My round-1 count of nine was the miss, and the fold is right. The
zero-byte and no-salvage-line conclusions survive the correction, and manual step 6 now says "all
eleven tracked `document.json` files".

`projects/documents/*/` holds `document.json`, `graph.json`, `passes/` and no `scripts/`; no
`scripts/*.py` is tracked outside the top-level `scripts/` tool directory.

## False trails (round 2)

- The `graph=SimpleNamespace(output_pass="main")` half of the reject stub is unused by the gate as
  specified. Redundant, not wrong, and trimming it is a preference.
- Manual step 1's new cold-open paragraph (play/stop buttons appearing one pass per frame) is a
  design-reviewer fold, outside my ten, and reads correctly against item 2.
- The review-history table records my round-1 falsifiability verdict as PARTIAL (findings 3, 4); with
  both closed it would now read PASS. Restating a superseded verdict in a history section is not a
  defect.

## Coverage statement (round 2)

Re-read in the folded spec: § Files touched (all 28 rows), § Tests (every entry naming one of my ten,
plus the new `reload`-seams section), § Design decisions items 7, 10, 11, § Manual verification, §
Open questions 3 and 4, § Review history.

Re-verified in code at `d2ade88` rather than from round-1 notes: `pass_graph.py:214-226` and `:248`
(`PassPlan.feedback`, `plan_passes`), `paths.py:13,26` plus a live `pass_name_of` call,
`tests/test_document_graph.py:64-98` (`gl_ctx`, `_document`), `tests/test_copilot_script_tools.py`
imports, `tests/test_script_driven_reject.py:22-24`, `copilot/backend.py:730-757` and `:241-262`,
`tabs/document.py:257-263`, `app.py:606-617`, `document.py:286,363-366`, `git ls-files | grep
document.json` and both `projects/documents/` files parsed.
