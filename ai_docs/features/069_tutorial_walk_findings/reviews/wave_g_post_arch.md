# W-G post-implementation review - architecture & conventions

Commit under review: `928c231` ("069 W-G: address passes from the document script"), 38 files.
Reviewer role: `dev_flow.md` step 6 architecture pass - module boundaries, where things live,
duplication, protocol shape, docs that should have moved, prompt tier discipline.

## Verdict

| Area | Verdict |
|---|---|
| Boundaries (`scripting/keys.py`, the import graph) | **PASS** |
| Protocol (`ScriptPass` / `ScriptTarget`) | **PASS** |
| Duplication (`_driven_on`, the two error adapters, the stub generator) | **PASS** |
| Copilot prompt (tier, token delta, pin) | **PASS** |
| Docs (`conventions.md`, `dev_flow.md`, 065 D12, 068 D7, `help_content.py`, the design note) | **PARTIAL** - one stale mechanism name in the `dev_flow.md` entry this commit edited |
| Dogfood script (`verify_script_engine.py`) | **PARTIAL** - a pre-existing 065 breakage the wave re-keyed without fixing, correctly disclosed |
| Conventions (comments, suppressions, annotations, test idioms) | **PASS** |

Gate on the restored tree, unpiped: `check passed`, `test passed` (1599 tests), `smoke` exits 139
under `xvfb`/llvmpipe - an environment artifact of this reviewer's shell, not a code fault. The
implementer's own log (`scratchpad/gates_wg_main.log`) shows all three green on the real display.

---

## Findings

### 1. `dev_flow.md` still lists "orphan-warn" as an `engine.py` responsibility, in the same entry this commit rewrote

**Claim.** The commit deletes the console `logger.warning` and the `warned` dedup set outright, and
the commit message says so. The `dev_flow.md` `scripting/` entry that this commit edited - three
lines above - still enumerates `engine.py` as doing "resolve/orphan-**warn**".

**Evidence.**

```
$ grep -n "logger\|warn" shaderbox/scripting/engine.py
(no output)

$ git show 928c231^:shaderbox/scripting/engine.py | grep -c "logger\|warn"
8
```

and `ai_docs/dev_flow.md`, in the `scripting/` bullet:

```
  `engine.py` (`ScriptEngine`: per-document registry, `(path,mtime)` cache + cached source, resolve/orphan-
  warn, `tick`, `dry_run`, `reset`, `fresh_behavior_for`, `script_stub_for`, `script_status`,
  `is_scriptable`), `keys.py` (`StoppedKey`, ...
```

The `keys.py` clause immediately after it is this commit's own addition, so the line was edited and
the stale half was carried over. The module has no `loguru` import at all any more; the whole
concept the word names is gone, replaced by the strip rows the same commit built.

**Severity.** Low but exactly the class `conventions.md ## Code rules` calls out: a doc naming a
mechanism the reader cannot find. It is the module map, so it is the first place a new reader looks
to learn what `engine.py` does.

**Fix.** In `ai_docs/dev_flow.md`, change the `engine.py` clause from `resolve/orphan-warn` to
`resolve/orphan-report` (or just `resolve`), since an orphan is now a strip row, not a log line.

---

### 2. `scripts/dogfood/verify_script_engine.py` has produced a zero-pass document since 065, and W-G re-keyed it without fixing that

**Claim.** The implementer's disclosure is accurate: the script seeds `shader.frag.glsl` at the
document root, which 065's loader stopped reading. The document is rejected outright at load, so the
script has been dead - not degraded - since 065, and the wave applied the pair re-key on top.

**Evidence.** Run it on the tree under review:

```
$ uv run python scripts/dogfood/verify_script_engine.py
... ERROR | shaderbox.ui_models:load_documents_from_dir:543 -
    Skipping unreadable document 'scripted': scripted: no readable pass file
AssertionError: expected u_wave script-driven on main, got set()
```

Root cause, `shaderbox/document.py::Document.load_from_dir`:

```python
for shader_path in sorted(
    (document_dir / PASSES_DIR_NAME).glob(f"*{PASS_SHADER_SUFFIX}")
):
```

`PASSES_DIR_NAME = "passes"`, `PASS_SHADER_SUFFIX = ".frag.glsl"` (`shaderbox/paths.py`), against
`scripts/dogfood/verify_script_engine.py::_seed_scripted_document`, which writes
`(document / "shader.frag.glsl")` at the document root and no `graph.json`. `load_from_dir` then
raises `ValueError(f"{document_dir.name}: no readable pass file")`.

`git log -L 34,34` dates the breakage to `19b9c52` ("065: retire 'node' for 'document'"), which
renamed `nodes/` to `documents/` and left the shader path at the root while 065 moved pass files
into `passes/`.

**Reachability.** Not in `make gates` - the `gates` target runs `check`, `test`,
`scripts/smoke.py`, and nothing else (`Makefile:67-`). Not referenced anywhere in
`.claude/skills/dogfood/SKILL.md` either (`grep -n "verify_" .claude/skills/dogfood/SKILL.md`
returns nothing). Its only callers are its own docstring and historical feature specs. So no gate
was ever red because of it, which is why it went unnoticed for four features.

**Judgement.** Leaving it broken is defensible for this wave but should not stand past it. The
"a defect found in a wave is fixed in that wave" rule bites hardest on defects the wave *creates*;
this one predates it and is in a dev script outside every gate. Against that: the wave *touched*
this file - it applied the pair re-key at line 97 - so it took ownership of a file it could see was
dead, and the fix is two lines. The cost of leaving it is that the next scripting wave inherits a
canonical check that has never run, and the disclosure lives only in a commit body.

**Fix.** In `scripts/dogfood/verify_script_engine.py::_seed_scripted_document`, write the shader to
`document / "passes" / "main.frag.glsl"` (creating `passes/`) instead of `document /
"shader.frag.glsl"`, so the document loads with one pass named `main` - which is what the file's own
`assert driven == {("main", "u_wave")}` already expects. No other line needs to change.

---

### 3. `keys.py` is the right home and `StoppedKey` the right shape - but the name is the loosest thing in the change

**Verdict: PASS, with a naming note only.**

The boundary argument holds and is checkable. The engine takes `frozenset[StoppedKey]` per tick
(`ScriptEngine.tick`), and `ui_models` persists `list[StoppedKey]` on `UIDocumentState`. A shared
type is required, and it cannot live in `ui_models`:

```
$ grep -n "^from shaderbox" shaderbox/ui_models.py
...
shaderbox/ui_models.py:21:from shaderbox.document import Document
```

`ui_models` imports the concrete `Document`, which the `scripting/` package is forbidden to reach
(`conventions.md`: "imports no imgui/glfw and no concrete `Document` type"). The import graph after
the change is clean in the required direction:

```
$ grep -rn "^from\|^import" shaderbox/scripting/*.py
# only: stdlib, moderngl, OpenGL.GL, pydantic,
#       shaderbox.paths, shaderbox.uniform_coerce, and scripting's own leaves
```

No `Document`, no imgui, no glfw, no `App`, and `keys.py` imports only `pydantic` - a true leaf.
`ui_models` imports `scripting.keys` (one direction), and nothing in `scripting/` imports
`ui_models`, so nothing cycles. `make check` (pyright + ruff) passes, which is the mechanical proof.

**On the home vs. `outputs.py` / `context.py`.** Neither is the right host. `outputs.py`'s docstring
scopes it to "typed return values a behavior's `update` hands back" - `Vec2/3/4`, `Array`, `Text`,
all of which normalize into `coerce_uniform_value`. `StoppedKey` is not a return value and never
reaches coercion. `context.py` holds "the one read-only object every behavior's `update` receives" -
`StoppedKey` is neither received by a script nor visible to one. Both would widen a
tightly-scoped module's stated role; a new leaf is the smaller change, and the `dev_flow.md` module
map now names it.

**The naming note.** `keys.py` is the weakest word in the change - it says "some keys" while the
module holds exactly one key type. `stop_key.py`, or `stopped.py`, would say what is in it. The file
docstring compensates, and the map entry names the symbol, so this is a preference, not a defect;
I am not asking for a rename, only recording that `keys.py` is the one name here that will need a
second look when the second key type arrives.

**`frozen=True` in the class args.** Verified as the commit claims: it is required, not stylistic.
`class StoppedKey(BaseModel, frozen=True)` makes the generated `__hash__` visible to pyright, which
`model_config = ConfigDict(frozen=True)` does not; `_stopped_for` builds a `frozenset` of these and
`_write_one` does a set membership test, so an invisible `__hash__` would be a pyright error.

---

### 4. The protocol is minimal and adapter-free; `Document` and `Pass` satisfy it structurally

**Verdict: PASS.**

**No adapter or wrapper class was added.** `ProjectSession` hands `ui_document.document` straight to
`ScriptEngine.tick` / `dry_run` / `reload` / `tick_export`; nothing in the diff wraps, shims or
projects it. The two members that had to change shape are exactly the two the commit message names,
and both changes are forced rather than chosen:

- `passes` is `Mapping[str, ScriptPass]`, not `dict`. A mutable `dict` is invariant in its value
  type, so `Document.passes: dict[str, Pass]` would not satisfy `dict[str, ScriptPass]`. `Mapping`
  is covariant in its value, so it does. This is also the honest contract: the engine reads
  `document.passes.get(...)` and `.items()` and never inserts a pass.
- `script_ready` is a read-only protocol property, so `Pass` can satisfy it with `@property`.

**Every protocol member has a reader - no dead surface.** All three `ScriptPass` members and the one
`ScriptTarget` member are read inside `engine.py`:

```
$ grep -n "\.script_ready\|get_active_uniforms()\|\.uniform_values\b\|\.passes\b" shaderbox/scripting/engine.py
140,141,146   document.passes.get / render_pass.uniform_values   (_freeze)
572,573,574   get_active_uniforms / passes.items / script_ready   (_active_by_pass)
602,605,624,634  render_pass.uniform_values                       (_write_one)
723,733,734,766  document.passes                                  (_tick_script)
```

`script_ready`'s semantics are the subtle part and they are right. `Pass.script_ready` is
`self.program is not None or bool(self.compile_unit.error_raw)` - the exact negation of the guard
`Pass.get_active_uniforms` tests before compiling (`if self.program is None and not
self.compile_unit.error_raw`). So `script_ready` is False on precisely the states where calling
`get_active_uniforms` would compile, which is what 066 D1 forbids from inside the frame loop. A
never-attempted pass is absent from `_active_by_pass` and its keys are held silently; a
compile-FAILED pass is present-but-empty and its keys take the orphan path. The distinction matters:
a failed attempt is never retried, so holding on `program is None` would silence those keys for the
life of the source.

`ScriptPass` and `ScriptTarget` are exported from `scripting/__init__.py` with no importer outside
the package. That is not new surface: `EngineNode` was exported the same way with the same zero
outside importers (`git grep -n EngineNode 928c231^`), so the change preserves an existing pattern
rather than adding to it.

---

### 5. No duplication introduced; the four `_driven_on` sites are one helper, and the two error adapters are genuinely different rules

**Verdict: PASS.**

**`_driven_on` vs `uniform_is_driven`.** These are not two paths to one answer. `_driven_on`
(`copilot/backend.py`) *projects* the document-scoped pair set down to the names on one pass, because
`_format_uniforms(render_pass, driven: set[str])` formats one pass and keeps its name-keyed
signature. `ProjectSession.uniform_is_driven` is a *membership test* on one pair. Different return
types, different questions:

```python
def _driven_on(driven: set[tuple[str, str]], pass_name: str) -> set[str]:
    return {name for pass_, name in driven if pass_ == pass_name}
```

```python
def uniform_is_driven(self, document_id: str, pass_name: str, name: str) -> bool:
    return (pass_name, name) in self.script_engine.script_driven_uniforms(document_id)
```

Both read the single source (`ScriptEngine.script_driven_uniforms`); neither reimplements the other.
The four `_driven_on` call sites all pass through the one helper rather than inlining the
comprehension, which is the shape the funnel rule wants. Correct as landed.

**`_script_errors_for` vs `_script_errors_for_pass`.** These encode two different UI rules, not one
rule twice. The script tab shows *every* soft error whatever pass it names, each prefixed
`pass.uniform`; a shader tab shows *only* the errors naming its own pass, unprefixed, and carries the
SCRIPT path so the click lands where the fix is. A bare-key error (pass `""`) appears on the script
tab and on no shader tab, because a bare key is a claim about the whole document. Merging them would
need a mode flag on one function that switches filter, prefix and path together - three behaviours,
which is a fork wearing a parameter. Both read the one `ScriptStatus` from the engine's
`script_status`; neither re-derives an error. `tests/test_script_error_strip.py` pins all three rules
with named falsifiers.

**The stub generator vs `_scriptable_uniforms_for`.** One producer, one consumer, no second path:

```python
# project_session.py — the only place uniforms are filtered for a stub
def _scriptable_uniforms_for(self, document_id: str) -> dict[str, list[moderngl.Uniform]]:
    return {pass_name: [u for u in render_pass.get_active_uniforms()
                        if is_scriptable(u) and u.name not in ENGINE_DRIVEN_UNIFORMS]
            for pass_name, render_pass in document.passes.items()}
```

Both call sites (`create_script`, `read_script_source`) feed that one dict into `script_stub_for`,
which owns the per-pass block emission. The engine-owned filter lives on the session side (where
`ENGINE_DRIVEN_UNIFORMS` is reachable without breaking the headless boundary) and the formatting
lives in the engine - a clean producer/format split.

---

### 6. The prompt block is in the right tier, the delta is small, and both new claims are pinned

**Verdict: PASS.**

**Tier.** The SCRIPT API block rides `PromptBlock("project_context", Volatility.RARE, ...)`
(`copilot/prompt.py`), composed into `prompt_context.py`'s `script_api` field, exactly as
`api_doc.py`'s own docstring states ("the copilot prompt's RARE tier carries (feature 059 D3)"). The
right tier: the block changes only when the engine's contract changes, which is what RARE means, and
the change did not move it.

**Token delta, measured.**

```
$ # old summary rendered from 928c231^:shaderbox/scripting/api_doc.py, new from HEAD
old chars 1250   new chars 1628   delta +378
old words  192   new words  263   delta  +71
```

Roughly +95 tokens on a 1628-char block, in the tier that caches best. The spend buys the grammar
the agent cannot infer - that a dict value means a pass block, and that a pass block beats a bare
key - plus the `down`/`prev_*` meanings a bare field list would leave as names without semantics.
Both are pipeline facts, not model-competence patches, so they clear the guard bar in
`conventions.md` ("would a strictly better model still need it?" - yes: this grammar is ours, and
nothing else in the prompt states it).

**Both claims are pinned.** `tests/test_script_api_doc.py` gains two tests, each with an explicit
falsifier naming the edit that turns it red:

- `test_the_contract_bullet_states_both_addressing_forms` asserts `"EVERY pass declaring it"`,
  `"{pass: {uniform: value}}"` and `"WINS over a bare key"`.
- `test_the_mouse_gloss_states_the_button_and_the_previous_position` asserts `"LMB"` and
  `"PREVIOUS cursor position"`, and its comment correctly notes the pre-existing field-list pin
  already covers the bare names, so this test covers only the meanings.

**`_pass_views` reads the pair-keyed set correctly.** It projects per pass:

```python
uniforms=_format_uniforms(render_pass, _driven_on(driven, name)),
```

so a uniform driven only on `paint` no longer shows a phantom driven-marker on `composite`; the
sibling pass shows its real value. The two `WorkingSetView` sites do the same, projecting to
`pass_name_of(document.render_pass.source.path)` since that listing formats the output pass.
`tests/test_copilot_script_tools.py::test_pass_views_marks_driven_only_on_the_pass_that_declares_it`
pins it.

The `_dotted` helper is display-only and says so in its comment ("never parsed back, and never a key
grammar"), and `ScriptWriteResult`'s docstring repeats the point for the agent-facing field. That
distinction matters - a dotted string the agent might try to use as a dict key would be a silent
orphan - and it is stated in both places a reader lands.

---

### 7. Docs: the three `conventions.md` edits read as the now; 065 D12 and 068 D7 are correctly superseded

**Verdict: PASS** (the one exception is finding 1, in `dev_flow.md`).

**`conventions.md`.** Grepped for the specific stale-claim shapes the parent asked about:

- No `render_pass` claim survives in either scripting entry - the file's only `render_pass` mention
  is in the structural-impossibility law, about a dirty-signal example, unrelated to scripting.
- `EngineNode` appears nowhere in `conventions.md` or `dev_flow.md`. Its remaining hits are all in
  feature specs (041, 069's own spec and review files), which are historical records and correctly
  left alone.
- "name-keyed" survives in the scripting entry only in its *contrastive* form - "a name-keyed set
  would freeze it on every pass at once" - which is the justification for the pair, not a claim
  about the present. Its other two hits are in unrelated entries (029's tool registry, the
  parallel-dicts smell).
- The freeze-granularity, error-key and PLAY/STOP paragraphs all now read `(document_id, pass, name)`
  / `(pass, name)` / `frozenset[StoppedKey]`, and state the `""` absent-pass marker explicitly.

The 069 D3 routing rule is folded into the existing script bullet rather than added as a new one,
which is right: it is the same decision refined, and the file's own preamble forbids a changelog.

**065 D12.** The supersession is placed directly under the decision it retires and is honest about
what survives: "The addressing hole this decision names is real and is closed by the dict's shape
rather than by a file per pass; the per-pass file was never implemented." That last clause is
verifiable and true - `git log` shows no per-pass script file ever landed.

**068 D7.** "Retraction lifted by 069 (W-G)" names both triggers the original retraction recorded (a
script bound to the output pass, and a mouse with no button) and points at W-H for the rewrite. One
cosmetic note, not a defect: the new paragraph sits *above* the older "*Superseded during the review
round (`ac747d6`)*" line, so a reader meets the lift before the supersession it lifts. The facts are
all present and correct; only the reading order is slightly out of sequence.

**`help_content.py`.** The added sentence is 21 words in a body paragraph (not a label, tooltip or
`help_marker`), so it is outside `tests/test_ui_prose_budget.py`'s domain and correctly so. It states
both forms in the user's vocabulary without the engine's terms: "A script drives a uniform on every
pass that declares it, or names a pass to drive only that one."

**`01_design_scripting.md`.** Consistent with what landed, no "landed as" note needed. Its "Decision:
B1" section plus the "Bare keys broadcast (maintainer, after B1)" section pin four rules, and all
four are in the code: value-type dispatch (`broadcasts` / `blocks` split on `isinstance(v, dict)`);
a broadcast no pass declares is the orphan error (`errors[(document_id, "", name)]`); a pass block
wins over a broadcast (the two-phase ordering); the stub lists each pass's uniforms commented under
its block (`script_stub_for`). Its header still reads "Status: research, no code", which was accurate
when written and is the normal state for a design note whose decision has shipped into the spec.

**`dev_flow.md`.** The `scripting/` entry gains `keys.py` and the broadcast/pass-block rule, and
swaps `EngineNode` for `ScriptTarget`. The `ProjectSession` entry needed no edit and got none -
correctly: nothing about session ownership of the engine changed, only the type it hands it.

---

### 8. Conventions: comments state the now, no suppressions, annotations complete, test idioms match

**Verdict: PASS.**

**Suppressions.** One `# type: ignore` exists in the touched files
(`ui_models.py:78`, `gl_type=uniform.gl_type`). It is pre-existing (`git show 928c231^` shows it at
line 77) and explicitly allowlisted in `conventions.md ## Known quirks` ("`moderngl.Uniform.gl_type`
- not in moderngl's stub (`uniform_coerce.py`, `ui_models.py`, `util.py`)"). No `# noqa`, no
`# pyright: ignore`, no inline import anywhere in the diff.

**Comments state the now.** Spot-checked every substantial new comment block. They name the
non-obvious *current* fact and stop: `script_ready`'s comment states the invariant and the 066 D1
constraint, not the debugging story; `_active_by_pass` states why an unready pass is absent rather
than empty; `has_feedback` states why `_feedback` is the wrong source (it is an allocation cache);
`_drop_script` states why the pair is unpacked. `_write_one`'s `driven.add` carries a five-word
trailing comment. None narrate development history, and the one place that would have been tempting
- the dict rejection in `coerce_one` - states the invariant it defends rather than the routing bug
it prevents.

**Annotations.** Complete on every new signature: `_driven_on`, `_dotted`, `_uniform_changes`,
`_script_errors_for_pass`, `_load_ui_state`, `script_ready` on both `Pass` and the protocol,
`has_feedback`, `reset_current_document_feedback`, and every widened parameter list. `make check`
(pyright basic, 0 errors) is the mechanical proof and it passes.

**Test idioms.** `tests/test_script_engine.py`'s `_FakePass` / `_FakeDocument` and
`tests/test_script_driven_reject.py`'s `types.SimpleNamespace` stubs are the same family, chosen by
what each needs. The engine tests need a *mutable* `uniform_values` dict and a callable
`get_active_uniforms` across many ticks, so a small class is the readable form; the reject test needs
a read-only slice for one call, so `SimpleNamespace` is. Both files use `SimpleNamespace` for the
`moderngl.Uniform` stand-in (`_u` / `_uniform`), which is the repo-wide idiom. `_FakeDocument`'s
one-pass shorthand (a bare list builds `{"main": ...}`) is what let most existing tests keep their
`{"u_x": ...}` scripts unchanged under the broadcast rule - a good sign the routing rule is
backward-compatible by design rather than by patching.

`tests/test_ui_models.py` drives `_load_ui_state`, the newly extracted GL-free half, which is the
right call: the loader builds a real `Document` and would need a GL context, and the file's docstring
says exactly that. The no-migration rule is pinned by a test whose falsifier is "any migration code
in this path" - mechanical rather than a promise, which is what `CLAUDE.md`'s hard rule wants.

---

## False trails

- **A second `logger.warning` left behind after the orphan-warn deletion.** None: `grep -n "logger\|warn" shaderbox/scripting/engine.py` returns nothing on the whole module.
- **`stopped_uniforms` shape drift in the tracked `document.json` files.** Checked all eleven: nine hold `[]`, two omit the key, none holds a `list[str]`. The commit message's count and claim are exact.
- **`_freeze` mishandling a bare-key `("", name)` pair.** It cannot receive one: `_freeze` is called only with `drove_last = set(last_driven)`, and a bare orphan enters `last_skipped`, never `last_driven`.
- **F6 colliding with the error strip's expand toggle.** The `(F6)` in `tabs/code.py` is 047's *finding label*, not a chord; `errors_expanded` is toggled by a click and bound to no key. `RESET_FEEDBACK` is the only F6 chord.
- **A stale soft error surviving a script that starts crashing.** Real, and reproduced - a `u_typo` orphan error stays on the strip while the new script raises, because the behavior-level branch returns before the stale-clear. But it reproduces identically on `928c231^`, so it is pre-existing, not a W-G regression, and out of scope here.
- **The stub generator duplicating the engine-owned filter.** One implementation, in `ProjectSession._scriptable_uniforms_for`; `script_stub_for` receives the already-filtered dict.
- **`set_uniform`'s gate deviating from the spec's "stays name-keyed within that pass".** It asks `(pass_name_of(output pass), name)` against the document-scoped pair set, which *is* name-keyed within that pass. Consistent.

---

## Coverage

**Read end-to-end:** `shaderbox/scripting/{keys,engine,context,errors,outputs,__init__}.py`,
`shaderbox/scripting/api_doc.py` (diff + rendered output), the full diffs of `shaderbox/{app,ui,ui_models,core,document,commands,help_content}.py`,
`shaderbox/copilot/{backend,capabilities}.py`, `shaderbox/tabs/code.py`, `shaderbox/widgets/uniform.py`,
`shaderbox/scripting/behavior.py`, and the relevant regions of `shaderbox/project_session.py`
(`_stopped_for`, `uniform_is_driven`, `is_uniform_stopped`, `set_uniform_stopped`,
`set_document_all_stopped`, `get_script_driven_uniforms`, `_scriptable_uniforms_for`,
`create_script`, `read_script_source`, `write_script_source`).
`shaderbox/core.py::Pass.{script_ready,get_active_uniforms}` and
`shaderbox/document.py::{has_feedback,load_from_dir}`.

**Docs read:** project `CLAUDE.md`; `ai_docs/conventions.md` in full; `ai_docs/dev_flow.md`'s module
map; `.claude/skills/copilot-llm-agent-design/SKILL.md` triggers; `069/01_design_scripting.md` in
full; `069/01_spec.md` W-G section; the 065 D12 and 068 D7 supersession edits.

**Tests read:** `tests/test_script_error_strip.py`, `tests/test_ui_models.py`,
`tests/test_script_api_doc.py`'s two new tests, `tests/test_script_driven_reject.py`,
`tests/test_script_engine.py`'s fixtures and fakes.

**Executed:** the full gate under `xvfb` twice (`check passed`, `test passed` - 1599 tests;
`smoke` 139, environment); the targeted 112-test scripting/UI subset; a rendered-output diff of
`script_api_summary()` old vs new for the token measurement; `scripts/dogfood/verify_script_engine.py`
(reproduced the zero-pass failure); a live probe of the stale-soft-error question against both
`HEAD` and a `928c231^` worktree.

**Not covered:** the interactive behaviour of the mouse capsule and the `Clear` ghost button (both
need a display and a hand on the cursor; they are `wave_g_post_code`'s and the maintainer's);
GPU-level correctness of the broadcast write beyond the one GL test the wave added; the W-H tutorial
rewrite the 068 D7 note promises.

**Tracked files edited by this review:** none. One pre-existing uncommitted mutation in
`shaderbox/scripting/engine.py` (a `[:1]` slice on the broadcast target list, left over from an
earlier session's falsification round) was found mid-review and restored with `git checkout --`;
the tree is clean and matches `928c231`. Every measurement above was re-run on the restored tree.
