# W-G post-implementation spec-fidelity audit

Anchor: `60_wave_g_scripting.md` (the wave spec, two reviewers + two closure rounds), `01_spec.md § W-G`,
`01_design_scripting.md`, `00_findings.md` rows 22/23/29/30. Commit under review: `928c231`, 38 files,
HEAD of `dev`. Every design decision (1..14), every Files-touched row, every Tests row, every Manual
step, and every "Verified / corrected premises" row that named a change was walked. The worked example
and both misspelling cases were EXECUTED against the real engine.

## Verdict

| Dimension | Verdict |
|---|---|
| Wave-spec fidelity (code) | **PASS** — all 14 decisions landed; deviations are shape-only and behaviour-preserving |
| Design-doc fidelity | **PASS** — B1, bare-key broadcast, dict-value invariant, orphan handling all land |
| Parent fidelity | **PASS** — every W-G bullet satisfied; D3 and D7's mouse half honoured; no builtin `u_mouse` |
| Findings closure (#22 #23 #29 #30) | **PASS** — each closed by a cited line; none would recur |
| On-disk claims | **PASS** — zero bytes changed, all eleven `document.json` load clean, no migration code |
| **Spec-record fidelity** | **FAIL** — the wave spec was not touched by the commit; eleven landed deviations are recorded only in the commit message (finding 1) |

The code is faithful. The documentation of what actually landed is not: `60_wave_g_scripting.md` still
describes shapes the implementation deliberately and correctly departed from, and a reader who opens the
spec after `/clear` is handed instructions that contradict the code in eleven places.

## Coverage table — design decisions

| # | Decision | Status | Evidence |
|---|---|---|---|
| 1 | `EngineNode` → `ScriptPass` + `ScriptTarget` | **Landed, deviated** | `shaderbox/scripting/engine.py:102` `ScriptPass`, `:119` `ScriptTarget`. Deviation: `passes` is a read-only `Mapping` property, not `dict` (`:126`), and `script_ready` a read-only property, not a bare `bool` (`:112`). Both forced by pyright variance. `EngineNode` gone repo-wide (grep: zero hits). `__init__.py` exports updated (`shaderbox/scripting/__init__.py`, `ScriptPass` / `ScriptTarget` in both the import and `__all__`). |
| 2 | Two-phase routing + `script_ready` truth table | **Landed** | `engine.py:696-697` the two comprehensions, `:699` broadcasts loop, `:732` blocks loop. `_active_by_pass` (`:562-576`) omits a not-ready pass entirely. `Pass.script_ready` = `self.program is not None or bool(self.compile_unit.error_raw)` (`shaderbox/core.py:230-237`) — the NEGATION form round 2 reopened, not the inverted one. |
| 3 | `ScriptError.pass_name` + three messages | **Landed** | `shaderbox/scripting/errors.py:17` `pass_name: str = ""`. Messages verbatim: `engine.py:393` orphan, `:395` sampler, `:713` bare-key, `:738` unknown pass with `sorted(document.passes)`. |
| 4 | `coerce_one` rejects a dict | **Landed** | `shaderbox/scripting/behavior.py:259-268`, before `normalize_output` at `:269`. |
| 5 | `(pass, name)` keys end to end | **Landed** | Every row of the spec's table: `errors` 3-tuple (`engine.py:292`), `last_driven`/`last_skipped` (`:162,167`), `last_good` (`:157`), `warned` REMOVED, `script_driven_uniforms -> set[tuple]` (`:298`), `ScriptStatus.soft_errors` 3-tuple (`:69`), `ScriptProbe` all four fields (`:83-86`), `tick(stopped=frozenset[StoppedKey])` (`:428`). `_drop_script` UNPACKS (`:375`). `_freeze` takes the document (`:129`). |
| 6 | `StoppedKey` + signatures | **Landed, deviated** | Deviation: lives in `shaderbox/scripting/keys.py:12`, not `ui_models.py`; `frozen=True` in the class args, not `model_config`. Both justified in the file's own docstring. Six signatures: `project_session.py:712,720,731,759,605,638`. `App` wrapper `app.py:1413`. |
| 7 | Copilot readers filtered per pass | **Landed** | `_driven_on` helper `backend.py:241`; `_pass_views` uses it at `:785`; the two output-pass formatters at `:670` and `:743-747`; the `set_uniform` gate MOVED below the `target =` bind, `:920-932`. `ScriptWriteResult` dotted form `:1987-1997`. `_motion_verdict`/`_uniform_changes` take the pair. |
| 8 | Console warning out, shader-tab strip in | **Landed, deviated** | Zero `logger.warning` / `warned` / `warn=` in `engine.py`. `_script_errors_for` prefixes the pass (`tabs/code.py:123-138`); `_script_errors_for_pass` added (`:140-158`); concatenated at `:498`. Deviation: the spec's `_to_pass_errors(edited)` does not exist — the implementation uses `edited.compile_unit.errors` directly, same ordering. Deviation: `_draw_error_strip` gained a `tab` parameter (`:223`) to reach `open_script_for` at `:246`. |
| 9 | Stub, one block per pass | **Landed, cosmetic deviation** | `script_stub_for(uniforms_by_pass)` `engine.py:234`; the per-pass loop `:250-260`; `(no scriptable uniforms)` `:258`; bare `return {}` fallback `:269`. `_scriptable_uniforms_for` per-pass `project_session.py:638-656`. `_script_import_line` over the union `engine.py:245`. Cosmetic: `engine.py:251` emits `# 'paint': {` (repr, single quotes); spec and design note both show `# "paint": {`. |
| 10 | `MouseState.down` / `prev_*` + the `ui.py` fill | **Landed** | `scripting/context.py:15,18-19`; `EXPORT_MOUSE` `:26`. The fill `ui.py:660-670`; `script_mouse_inside` on `App` `app.py:1072`. The clear runs BEFORE both branches (`ui.py:612-617`), which is stronger than the spec's else-branch and closes the closed-mid-drag case the spec named. |
| 11 | `RESET_FEEDBACK`, F6, button, callback | **Landed** | `commands.py:24` enum, `:125-130` spec on `_chord(K.f6)` in `C.DOCUMENT` beside `TOGGLE_DOCUMENT_PLAY`. Callback `app.py:528`, method `app.py:1430`. Button `ui.py:715-723`, ghost, top-left, `SPACE.MD` inset, tooltip through `_hint`. `Document.has_feedback` over `plan_passes(self.graph)[0].feedback` (`document.py:357-363`). |
| 12 | Prompt block regenerates | **Landed** | `api_doc.py:72-76` gloss, `:122-127` contract bullet. Caveat contiguity VERIFIED by execution: `"FROZEN at 0.5,0.5 on export and in the headless probe"` is present as one substring. `prompt.py` / `prompt_context.py` untouched, as the spec predicted. `capabilities.py` comments only. |
| 13 | On-disk, hand edits, no migration | **Landed** | `git show 928c231 --stat -- '*.json' 'projects/'` is EMPTY. `git ls-files \| grep -c document.json$` = 11; nine hold `[]`, two omit the key. All eleven executed through `_load_ui_state`: every one yields `stopped_uniforms == []`. Zero migration/compat code anywhere in the diff. |
| 14 | Docs | **Landed** | `conventions.md` scripting entry and PLAY/STOP entry both rewritten as specified; `065/01_spec.md:136-139` supersession; `068/01_spec.md:88-92` retraction lifted; `help_content.py:139-140` one sentence. Plus `dev_flow.md`'s `scripting/` entry, which the spec did not list. |

## Coverage table — the seven seams and the gate files

| Seam | Status | Evidence |
|---|---|---|
| `_resolve_scripts` → `reload` | Landed | `project_session.py:428` |
| `_load_one_document_from_disk` → `reload` | Landed | `:509` |
| `_make_export_isolation` → `tick_export` | Landed | `:554` |
| `reload_scripts` → `reload` | Landed | `:575` |
| `ProjectSession.tick` → `tick` | Landed | `:598` |
| `write_script_source` → `reload` | Landed | `:692` |
| `write_script_source` → `dry_run` | Landed | `:697` |
| **Verification** | — | `grep -n "script_engine\.\(tick\|tick_export\|dry_run\|reload\)" shaderbox/project_session.py` returns exactly these seven lines, and no `.render_pass` appears in any of their argument lists |
| `scripts/smoke.py` (in `make gates`) | Landed | `:301` `("main", "u_a")` in the bind canary, `:311` the new `set_uniform_stopped` arity, `:318` the pair-shaped driven check |
| `scripts/dogfood/verify_script_engine.py` | Landed | `:97-99` `{("main", "u_wave")}` |
| `scripts/dogfood/harness.py` (spec MISSED it) | Landed | the `tick_export` call takes the document; the `dry_run` call likewise; `probe.per_key_errors` / `orphan_keys` unpack three elements |
| `tests/test_motion_verdict.py` (spec MISSED it) | Landed | 13 tests, all pair-keyed, green |

## Coverage table — tests

| File | Spec row | Status |
|---|---|---|
| `test_script_engine.py` | routing table + 7 named tests | **All 8 landed.** `test_the_routing_table` `:1195` with exactly the six spec'd rows (`:1160-1191`); `:1217`, `:1233`, `:1251`, `:1272`, `:1297`, `:1332`, `:1355`. 75 tests, green. |
| `test_script_engine_gl.py` | drop `.render_pass`, +1 two-pass broadcast | Landed. `test_a_broadcast_reaches_both_passes_on_the_gpu` `:338`. 7 tests. Deviation: the GPU test asserts absolute pixels per sample time, not a difference — recorded in the commit message, not the spec. |
| `test_script_dry_run.py` | pairs + 1 added | Landed. `test_dry_run_reports_the_pass_in_orphan_keys` `:282`. 11 tests. |
| `test_export_script_wiring.py` | document handed + non-output pass + EXPORT_MOUSE | Landed. `:93`, `:136`. 3 tests. |
| `test_copilot_script_tools.py` | dotted strings + `_pass_views` sibling test | Landed. `test_pass_views_marks_driven_only_on_the_pass_that_declares_it` `:216`, built on a `SimpleNamespace` as round 2's finding 8a required (no GL skip). 14 tests. |
| `test_script_driven_reject.py` | fixture grows a real document + 1 added | Landed. `test_set_uniform_does_not_reject_a_uniform_driven_on_another_pass` `:85`. 3 tests. |
| `test_script_api_doc.py` | 2 added, 2 no-edit pins | Landed. `:175`, `:184`. 12 tests. |
| `test_ui_models.py` | `test_a_stale_string_stopped_set_drops_to_empty` | Landed `:18`, plus three siblings `:29,:41,:50`. |
| `test_command_routing.py` | F6 standalone assertion | Landed `:60`. |
| `test_script_error_strip.py` | NOT in the spec | New file, 4 tests, first coverage of `tabs/code.py`'s two adapters. |
| `test_document_graph.py` | NOT in the spec | `test_has_feedback_reads_the_graph_not_the_allocation_cache` `:687`. |
| `test_persistence_completeness.py` | no edit | Confirmed untouched. |

**Full suite under xvfb: 1599 passed, 4 skipped.** `make check` passes. `make gates` on this box reports
RED at the smoke step (exit 139) — a segfault in `glfw.get_video_mode`, reproduced IDENTICALLY at the
parent commit `928c231^` in a clean worktree, so it is this box's display environment and not the commit.
The smoke step is therefore UNVERIFIED here, not failed.

## Findings

### 1. The wave spec records none of the eleven landed deviations (severity: high)

**Claim.** `928c231` does not touch `60_wave_g_scripting.md` — `git show --stat 928c231 -- ai_docs/features/069_tutorial_walk_findings/`
returns nothing, and `git log --oneline -1 -- 60_wave_g_scripting.md` is `453c4f3`, the lock commit two
commits earlier. `928c231` is HEAD, so nothing follows it either. The spec still instructs:

- `StoppedKey` "placed in `ui_models.py` immediately above `UIDocumentState`" (spec `:353`) — it is in
  `scripting/keys.py:12`.
- "`model_config` carries `frozen=True`" (spec `:363-364`) — it is a class arg (`keys.py:12`).
- `class ScriptPass: uniform_values; script_ready: bool` (spec `:135-137`) — `script_ready` is a property (`engine.py:111-112`).
- `class ScriptTarget: passes: dict[str, ScriptPass]` (spec `:118`) — it is a read-only `Mapping` property (`engine.py:125-126`).
- `_to_pass_errors(edited)` in the `errors` expression (spec `:466`) — that function does not exist; `tabs/code.py:497` uses `edited.compile_unit.errors`.
- `_draw_error_strip`'s signature is unchanged in the spec — it gained `tab`.
- The Files-touched table lists no `tests/test_motion_verdict.py` and no `scripts/dogfood/harness.py`;
  both are in the diff, the latter carrying two more `.render_pass` seams (an eighth and ninth) against
  the spec's headline count of seven.
- The Tests section does not name `tests/test_script_error_strip.py` or the `test_document_graph.py`
  addition; both shipped.
- `_load_ui_state` (the `ui_models.py:495` split) appears nowhere in the spec.
- The GL test's rework (absolute pixel per sample time) appears nowhere in the spec.

**Evidence.** Each of the eleven is verifiable by opening the spec line cited above and the code line
cited in the coverage tables; the two disagree in every case. The commit message carries all eleven,
which is why the code is right — but a commit message is not where the next session looks for the spec.

**Why it matters here specifically.** The spec's own § Verified / corrected premises exists precisely
because "a citation is verified by opening it", and round 2's post-mortem says the rule "was applied to
the round-1 claims but not to the round-1 fixes". The same gap has now repeated one level down: applied
to the spec's claims, not to the implementation's corrections of them.

**Fix.** Add a `## Landed deviations` section to `60_wave_g_scripting.md` listing the eleven with the
one-clause reason each (the commit message already has the prose), and correct the four inline code
blocks (`ScriptPass`, `ScriptTarget`, `StoppedKey`'s placement and `frozen=True` form, the `errors`
expression) so a reader copying from the spec writes what is in the tree.

### 2. The parent spec's refuted salvage-line expectation is still stated as fact, with no pointer to its correction (severity: medium)

**Claim.** `01_spec.md:325` says "First launch after W-G logs one salvage line per stale `projects/dev`
document — expected, and the verification list says so", and `01_spec.md:432-435`'s manual-verification
line repeats it: "The first launch logs one salvage line per `projects/dev` document whose
`stopped_uniforms` predates the pair shape — expected once, gone after the hand-edit."

The wave spec refutes both (§ Verified / corrected premises, and item 13): both dev documents hold `[]`,
so nothing is stale and the first launch logs nothing. I confirmed this by executing `_load_ui_state`
over all eleven tracked files — every one returns `stopped_uniforms == []` with no salvage line.

**Evidence.** `grep -n "60_wave_g\|wave_g" ai_docs/features/069_tutorial_walk_findings/01_spec.md`
returns nothing: the parent has no pointer to the wave spec at all, so a reader arriving at the parent's
manual step will look for a log line that cannot appear and read its absence as a defect.

**Fix.** Amend `01_spec.md:325` and the W-G manual-verification bullet to say the salvage line does not
appear because every tracked `document.json` holds `[]`, and cite
`60_wave_g_scripting.md § Manual verification` step 6 for the deliberate way to exercise the path once.

### 3. The `script_ready` inversion is caught only by GL-gated tests (severity: low)

**Claim.** Design finding 4 — reopened in round 2 because the fold wrote the expression inverted — is the
wave's most-discussed correctness hazard, and the spec answers it with a truth table plus "read the table,
not the expression". No GL-free test pins it.

**Evidence, demonstrated.** Replacing `core.py:237` with the inverted form
(`self.program is None and not self.compile_unit.error_raw`) and running the GL-free suite gives
`75 passed, 7 skipped` — fully green with the bug in. The same mutation under `xvfb-run` gives
`6 failed, 1 passed` in `tests/test_script_engine_gl.py` alone. So the defect is caught, but only on a box
with a GL context; on a display-less box (the case the commit message itself calls out when justifying
the `_load_ui_state` split, "a persistence rule verified behind a GL skip is a rule nothing checks on a
display-less box") the inversion ships green.

`test_a_not_yet_compiled_pass_is_held_not_errored` (`tests/test_script_engine.py:1251`) uses a fake whose
`script_ready` is set by hand, so it pins the ENGINE's use of the flag and not `Pass`'s definition of it.

**Fix.** Add a GL-free test in `tests/test_script_engine.py` (or beside the `has_feedback` test) that
builds a `Pass` with no context and asserts the three truth-table rows directly: never-attempted →
`script_ready is False`; `compile_unit.error_raw` set → True; `program` set → True. The property reads
only two attributes, so a light stub suffices and no GL context is needed.

### 4. The stub emits single-quoted pass keys where both the spec and the design note show double (severity: cosmetic)

**Claim.** `script_stub_for` builds each block header as `f"            # {pass_name!r}: {{\n"`
(`engine.py:251`), so a document with a `paint` pass emits `#     'paint': {`. The wave spec's item 9
snippet and `01_design_scripting.md`'s Decision-B1 snippet both show `# "paint": {`.

**Evidence.** `engine.py:251` and `:255` use `!r` for the pass name and each uniform name. The rendered
comment is valid Python either way and `test_the_stub_has_one_block_per_pass` asserts the pass name
appears, not its quoting, so nothing is broken. It is a divergence between the shipped stub and the two
documents a user or agent may read alongside it.

**Fix.** Either change `engine.py:251,255` to emit double quotes, or update the two snippets to single
quotes. The first is the smaller edit and matches the repo's JSON-adjacent reading of the stub.

## False trails

- *`tabs/code.py`'s "Clickable toggle (F6)" collides with the new `RESET_FEEDBACK` chord.* It does not — that F6 is feature-047 shorthand for a numbered decision, not a key.
- *`is_uniform_stopped` wrapping `state.stopped_uniforms` in `set()` per call is a bug.* It is redundant (a list `in` works on `StoppedKey`'s `__eq__`) but correct; the call is per row per frame over a list that is empty in every shipped document.
- *A broadcast whose only declaring pass holds the name as a sampler reports "no pass declares X" rather than the sampler message.* Checked: this is what the spec's item 2 says ("every pass for which `_binding_reject` returns `None`"), so it is designed, not a slip.
- *`_script_render_line` in `backend.py` is a change outside the spec's Files-touched clause for that file.* It is a required consequence of `ScriptProbe.samples` becoming pair-keyed; the spec's clause for `backend.py` names the probe re-key and this is inside it.
- *The initial `make gates` RED is the commit's.* It is not: `glfw.get_video_mode` segfaults identically at `928c231^` in a clean worktree on this box.
- *The `test_the_orphan_warning_no_longer_reaches_the_console` failure I first saw is real.* It was stale `__pycache__` bytecode from my own mutation runs; after clearing caches the test is green on a clean tree.

## Coverage statement

Read end to end: all 38 changed files (via `git show 928c231 -- <path>` for each), the wave spec in full
(1443 lines), `01_design_scripting.md` in full, `01_spec.md § W-G` plus its manual-verification and
open-questions sections, and `00_findings.md` rows 22/23/29/30 verbatim.

Executed, not merely read: the worked example and both misspelling cases against the real `ScriptEngine`
through a temp-dir script and a two-pass fake (results match the spec's stated outcome exactly, including
the three driven pairs, the `pass_name` on each error, and the `""` pass on the bare-key orphan); all
eleven tracked `document.json` files through `_load_ui_state`; the api_doc caveat-contiguity substring;
the full test suite under `xvfb` (1599 passed, 4 skipped); `make check`; `make gates` at both `928c231`
and `928c231^` to attribute the smoke segfault; and three falsifier mutations (loop-order swap →
precedence test red; `script_ready` inversion → GL-free green but GL red; both restored, tree verified
clean by `git status`).

Not verified: the seven manual-verification steps, which require the app on a real display (paint stroke
continuity, the re-entry traverse, the Clear button's appearance, export determinism, the copilot round
trip), and the smoke gate step, which segfaults on this box independently of the commit.

---

# Round 2 (closure) — against `a873ace`

Narrow closure round on the four findings above, plus a re-walk of the twelve `## Landed deviations`
rows against the diff of `928c231` + `a873ace`. Read via `git show a873ace:<path>` throughout: W-D is
being implemented concurrently and the working tree carries its in-flight edits.

**Overall: PASS.** All four findings closed. The twelve rows are complete and each is supported by the
diff. Round 3's routing change is a real correctness fix, not a preference — I reproduced the defect it
names against `928c231` and confirmed the fix against `a873ace`.

## Finding verdicts

| # | Finding | Verdict | Line |
|---|---|---|---|
| 1 | Wave spec records no landed deviations | **CLOSED** | `60_wave_g_scripting.md:1372` `## Landed deviations`, twelve rows at `:1382-1393`; the four copyable code blocks corrected — `script_ready` as a `@property` (`:139-142`, and item 1's block at `:95-99`), `passes` as a read-only `Mapping` (`:99`), `class StoppedKey(BaseModel, frozen=True)` (`:366`), and `else edited.compile_unit.errors` (`:491`) with `_to_pass_errors` gone from the block. |
| 2 | Parent's refuted salvage-line claim stands with no pointer | **CLOSED** | `01_spec.md:327-328` now reads "**no salvage line appears** … `stopped_uniforms`" citing `60_wave_g_scripting.md` item 13 and its § Manual verification step 6; the W-G manual bullet at `:435-436` likewise. Both sites now reference the wave spec, which the parent had never cited. |
| 3 | `script_ready` inversion caught only by GL-gated tests | **CLOSED** | `tests/test_script_engine.py:1454` `test_script_ready_matches_its_truth_table` and `:1468` `test_script_ready_never_compiles`, both binding `Pass.script_ready.fget` onto a two-attribute `SimpleNamespace`. **Falsified:** with the property monkeypatched to the inverted form in a scratchpad `conftest.py` (working tree untouched), `DISPLAY= pytest -k script_ready` gives `2 failed`; unpatched the same GL-free file gives `80 passed`. The hazard is now red on a display-less box. |
| 4 | Stub emits single-quoted pass keys | **CLOSED** | `engine.py:253` `blocks += f'            # "{pass_name}": {{\n'` — double quotes, matching the bare-key example in the same block and the design note. |

## The twelve landed-deviation rows, re-walked

Each checked against the combined diff. None missing; none overstated.

| Row | Claim | Supported by |
|---|---|---|
| 1 | `StoppedKey` in `scripting/keys.py`, not `ui_models.py` | `scripting/keys.py:12`; `ui_models.py` imports it |
| 2 | `frozen=True` in the class args | `keys.py:12` `class StoppedKey(BaseModel, frozen=True)` |
| 3 | `script_ready` a read-only property | `engine.py:112` |
| 4 | `passes` a read-only `Mapping` | `engine.py:126` |
| 5 | `_to_pass_errors` never existed | `tabs/code.py:497` `else edited.compile_unit.errors` |
| 6 | `_draw_error_strip` takes the `tab` | `tabs/code.py:223`, reaching `open_script_for` at `:246` |
| 7 | **nine** seams plus a tenth stale key | `harness.py`'s two; and the tenth confirmed by diff: `928c231:verify_script_engine.py:160-162` asserts the two-tuple `("scripted", "script.py")`, `a873ace:169-175` the three-tuple `("scripted", "", "script.py")`. The wave's own re-key had missed it. |
| 8 | `test_motion_verdict.py` in the blast radius | 13 tests, pair-keyed, green |
| 9 | `verify_script_engine.py` is NOT in `make gates` | `Makefile`'s `gates` runs `check`, `test`, `scripts/smoke.py` only. Its 065 breakage also fixed: `928c231:34` wrote `shader.frag.glsl` at the document root; `a873ace:38` writes `passes/` + `pass_shader_name("main")`. **Ran it: exits 0**, printing "animated across t, deterministic at fixed t, export-isolated, broken script froze". Dead for four features, now live. |
| 10 | `test_script_error_strip.py` + the `test_document_graph.py` row shipped | 4 and 1 tests |
| 11 | `_load_ui_state` split; GPU test absolute-pixel; round-3 `gl_ctx.finish()` | `ui_models.py:495`; `test_script_engine_gl.py` diff carries the `finish()` |
| 12 | Broadcast gives the block phase's three-way answer; `dry_run` compiles first | `engine.py:573-577` `_active_by_pass -> tuple[..., set[str]]`; the broadcast branch's `if not_ready: continue`, then `_broken_pass_for`, then the pass-free row. `dry_run` pre-compiles at `:492-494` with the 066-D1 distinction stated. |

### Row 12 is a defect fix, not a preference — demonstrated

I ran the same three-case probe against both commits with a two-pass fake.

At `928c231` (extracted via `git archive`), all three cases collapsed to one answer:

```
A cold/not-ready      -> soft: [('', 'u_wave', "no pass declares 'u_wave' (orphan key)")]
B one broken          -> soft: [('', 'u_wave', "no pass declares 'u_wave' (orphan key)")]
C genuinely homeless  -> soft: [('', 'u_wave', "no pass declares 'u_wave' (orphan key)")]
```

Case A is the orphan-that-clears-a-frame-later the hold exists to prevent, landing on frame 0 of a
never-rendered document via the one addressing form 069 introduced. Case B named neither the pass nor
the failure and, being pass-free, reached no shader tab.

At `a873ace` the three separate correctly:

```
A cold/not-ready      -> soft: []   driven: []                      (HELD)
B one broken          -> soft: [('paint', 'u_wave', "... — pass 'paint' does not compile")]
C genuinely homeless  -> soft: [('', 'u_wave', "no pass declares 'u_wave' (orphan key)")]
```

**This is a defect my round-1 audit missed.** I walked the broadcast branch and read it as matching
item 2's wording ("a bare key that NO pass declares is a soft error"), without asking what "no pass
declares" means while a pass has not compiled yet. The block phase distinguishes the two with a named
lookup; the broadcast phase had nothing to look up. Reading the branch against the spec's sentence
passed it; running frame 0 of a cold document would have caught it.

**Falsified:** replacing `if not_ready:` with `if False:` in an isolated `git archive` copy turns
`test_a_broadcast_is_held_for_a_not_yet_compiled_pass` red (`1 failed, 2 passed`).

## Verification run

- Full suite on a clean tree at `a873ace` under xvfb: **1605 passed, 4 skipped** (was 1599 at `928c231`; six new tests).
- The worked example and both misspelling cases re-executed: byte-identical to round 1, so row 12 is behaviour-preserving for the all-ready case.
- `scripts/dogfood/verify_script_engine.py`: exits 0.
- Two falsifiers run in isolated copies, working tree never modified.
- `make gates` at `a873ace` is not measurable here: W-D began editing `pass_graph.py`, `pass_list.py`, `pass_settings.py`, `document.py`, `backend.py` and `project_session.py` mid-run, and every failure in the log sits in those files (`test_copilot_passes.py:153`, `test_lazy_compile.py:246`, `test_pass_hot_reload.py:118`) — none on W-G's surface. The clean-tree suite run above is the valid measurement.

## False trails this round

- *The stub's double-quote change alters behaviour.* It does not; the emitted comment is valid Python either way. It was filed as cosmetic and closed as cosmetic.
- *Row 9 contradicts the wave spec's Files-touched table, which calls `verify_script_engine.py` a gate file.* It does, and that is the row's point — the deviation is recorded rather than silently corrected.
- *The `## Landed deviations` preamble undercounts by saying "rows 1-11 landed with the implementation commit".* Checked: row 7's tenth-seam clause and row 11's `gl_ctx.finish()` clause are both round-3 work folded into rows that began at `928c231`, and the preamble names row 12 as round 3's. The split is accurate at row granularity.
