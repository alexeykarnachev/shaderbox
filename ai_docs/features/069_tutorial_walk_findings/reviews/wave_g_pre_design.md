# W-G pre-implementation review — correctness & design

Artifact: `ai_docs/features/069_tutorial_walk_findings/60_wave_g_scripting.md`.
Code read at HEAD `0ce84f8` (`git show 0ce84f8:<path>`); the working tree carries W-F's uncommitted
edits and was not used as evidence. Anchors: `01_spec.md` D3 / D7 / § W-G / § Out of scope /
§ Manual verification, `01_design_scripting.md`, `00_findings.md` rows 22 23 29 30,
`02_keybindings.md`'s `RESET_FEEDBACK` row, `.claude/skills/copilot-llm-agent-design/SKILL.md`.

## Verdict

| Dimension | Verdict |
|---|---|
| D3 fidelity | **PASS** |
| Key propagation `(pass, name)` | **PARTIAL** — finding 3 (a call site the spec pins to a line where the value it needs is not yet in scope) |
| Persistence | **PARTIAL** — finding 5 (the disk enumeration misses two tracked documents) |
| Mouse | **FAIL** — finding 1 (the api_doc gloss rewrite breaks the test the spec claims survives unedited) |
| Command | **FAIL** — finding 2 (`has_feedback` is specified over a lazily-populated map, so the button is absent on exactly the documents it is for) |
| Docs | **PASS** |

The spec is unusually strong on premise-checking: its § Verified / corrected premises re-derived 40
parent citations and corrected or refuted 17, and every row I re-checked against the code was
right, including the four that change the work (the seven `.render_pass` seams, the `app.py:1489`
relocation, the `tabs/document.py:253` refutation, the zero-byte disk edit). The findings below are
in the residue that check did not cover: two are wrong-by-execution, three are gaps.

---

## Findings

### 1. The new mouse gloss breaks `test_the_mouse_gloss_carries_the_frozen_at_center_caveat`, which item 12 asserts survives unedited. FAIL, mouse.

**Claim.** Item 12 says the freeze-caveat test "asserts the substring
`f"FROZEN at {at} on export and in the headless probe"`, which survives verbatim inside the longer
sentence", and § Tests repeats it: the test "also needs no edit (the caveat substring survives)".
The gloss the same item specifies does not contain that substring.

**Evidence.** The test at `tests/test_script_api_doc.py:109-115` builds
`caveat = f"FROZEN at {at} on export and in the headless probe"` and asserts it is `in`
`_CTX_GLOSS["mouse"]` and `in _flat(script_api_summary())`. The spec's replacement gloss reads
`... down = LMB over the canvas -- FROZEN at 0.5,0.5 with down=False on export and in the headless
probe`. The words `with down=False` sit between `at 0.5,0.5` and `on export`, so the asserted
substring is not present and both assertions go red. The test is not a line-number pin that a
shifted line invalidates; it is a literal substring built from `EXPORT_MOUSE`, so the break is
mechanical and certain.

**Fix (paste).** Replace item 12's first bullet's gloss with one that keeps the caveat contiguous:
`(`, the field list, `; x/y and prev_x/prev_y in 0..1 y-up, down = LMB over the canvas -- FROZEN at
{at} on export and in the headless probe, where down is False and prev equals x/y)`. § Tests then
correctly reads: `test_the_mouse_gloss_carries_the_frozen_at_center_caveat` needs no edit because
the caveat is contiguous in the new sentence, which is a constraint on the rewrite, not an
observation about it.

**Note on the second half of the same claim.** The other no-edit pin IS right:
`test_summary_lists_every_ctx_field_and_the_mouse_subfields` (`:101-106`) loops
`MouseState.__dataclass_fields__` and asserts `` f"`{name}`" `` is in the summary, and `_MOUSE_FIELDS`
(`api_doc.py:61`) renders every field name in backticks, so `down` / `prev_x` / `prev_y` are covered
with no edit. Verified by reading both.

### 2. `Document.has_feedback` over `_feedback` is false on exactly the documents the button is for. FAIL, command.

**Claim.** Item 11 says "`Document` exposes that as a property over its `_feedback` map,
`has_feedback: bool`", and gates the `Clear` button on it. `_feedback` is an allocation cache, not a
declaration, so the property answers "has this document already rendered a feedback frame", not "does
this document have a feedback pass".

**Evidence.** `document.py:246-250` states it directly: "A feedback pass's previous frame. Allocated
**on demand by the first frame that needs one**". `_feedback_canvas` (`:387-411`) is what inserts a
key, called from `render` (`:478`) only when an input names its own pass. So on a freshly opened
document, before its first render, `_feedback` is `{}` and the button is hidden on a document that
plainly has a feedback pass. Worse, `Document.reset_feedback` itself clears the map (`:365`), so the
button **hides itself the moment it is clicked** and does not come back until the next rendered frame
re-allocates. That is the control disappearing on use.

The declared fact exists and needs no render: `plan_passes` computes `PassPlan.feedback: set[str]`
(`pass_graph.py:224, 263`) from the self-edge in `PassEntry.inputs` — `pass_graph.py:94` documents
"an entry naming its own pass is feedback".

**Fix (paste).** Replace item 11's sentence with: `Document` exposes `has_feedback: bool` computed
from the GRAPH, not from `_feedback`: `bool(plan_passes(self.graph)[0].feedback)`. The `_feedback`
map is an allocation cache filled by the first frame that needs one and emptied by `reset_feedback`
itself, so a property over it would hide the button on an unrendered document and hide it again the
instant the user clicks it.

### 3. The `set_uniform` gate's pass-qualified check is specified at a line where the output pass name is not in scope. PARTIAL, key propagation.

**Claim.** Item 7 says the gate at `backend.py:879` "asks `(output_pass_name, name) in driven`".
Nothing named `output_pass_name` exists at that point in the function.

**Evidence.** `set_uniform`'s `_on_main` (`backend.py:862`) resolves `document_id` at `:863`, then the
engine-owned check (`:873`), then the driven check (`:879`). The document object is first bound at
`:887` (`target = self._get_ui_documents()[document_id].document`), and the output pass name is
reachable only from it (`target.graph.output` / `target.render_pass`). So the change as written does
not compile. The verdict it encodes is right; the placement is not.

**Fix (paste).** In item 7, replace the `set_uniform` bullet's second sentence with: the gate moves
BELOW the `target = ...` bind at `:887` (the output pass name is unreachable above it) and asks
`(pass_name_of(target.render_pass.source.path), name) in driven`. Moving it past the
`uniform is None` resolution is also the better order: a name that is not on the output pass at all
should get the "no active uniform" answer, not the script-driven one.

### 4. A permanently-broken pass is silently held forever by `script_ready`. PARTIAL, D3 fidelity.

**Claim.** Item 2 says a not-ready pass "is absent from `active_by_pass` and its keys are held, not
errored: a key naming a not-yet-compiled pass records no soft error and drives nothing this tick",
and open question 2 defends the protocol member over catching the compile. The design is right for a
pass that has not compiled YET. It is wrong for a pass that has compiled and FAILED, and the spec
does not separate the two.

**Evidence.** `core.py:230-239`: `get_active_uniforms` compiles on demand, and "A FAILED attempt is
not retried — its errors stick in `compile_unit` until `invalidate()` resets it". So after a failed
compile `program is None` permanently, and any `script_ready` defined as "the program is built" stays
False for the life of that source. Under item 2's rule the script's keys for that pass are then held
silently forever: no strip row on the script tab, no strip row on the pass tab, and the driven set
never contains the pair, so its play/stop button never appears. The user sees a uniform that simply
does nothing, which is the same "silently inert" failure the whole wave exists to remove (#29, and
065 D15's "nothing may fail silently").

The spec also mis-states the sibling condition: it says the property returns "whether its program is
built (the same condition `first_render_done` already tracks against)". They are different — `Pass`
sets `first_render_done` on a render ATTEMPT (`core.py:167, 202`, read at `ui.py:262, 306`), not on a
successful compile.

**Fix (paste).** In item 2, split the not-ready case in two. `ScriptPass.script_ready` is False only
while the pass has NEVER ATTEMPTED a compile (`program is None and not compile_unit.error_raw` — the
same pair `get_active_uniforms` itself tests before compiling, so the protocol member and the lazy
compile agree by construction). A pass whose compile FAILED is ready-but-empty: its active map is
empty, so its keys take the ordinary orphan path and the user reads
`pass 'paint' has no active uniform 'u_brush' (orphan key)` on the strip beside the compile error
that caused it. Only the never-attempted pass is held silently, and that state lasts frames, not
forever. Drop the "the same condition `first_render_done` already tracks against" clause: that flag
is set on a render ATTEMPT, not on a compile.

### 5. The disk enumeration misses two tracked `document.json` files. PARTIAL, persistence.

**Claim.** Item 13's table enumerates nine files (seven examples, two `projects/dev` documents) and
concludes "The wave changes no bytes under `projects/dev`" and that first launch logs nothing. Two
further tracked documents are absent from the table.

**Evidence.** `git ls-tree -r 0ce84f8 --name-only | grep document.json` returns eleven files. The two
not in the spec's table are `projects/documents/1901ab60-8d6f-4de0-b598-ca35ff5c3664/document.json`
(five passes) and `projects/documents/307598da-de4f-4133-beaa-354e901d6a2b/document.json`. Both carry
`"stopped_uniforms": []` under `ui_state`, so the CONCLUSION survives untouched — still zero bytes,
still no salvage line. But the spec's own standard for the row is that "the claim that no edit is
needed is only worth anything if someone looked", and these two were not looked at. `projects/` is
tracked exactly as `projects/dev/` is (only `projects/*/media/`, `renders/` and the copilot archives
are gitignored), and `1901ab60` is a five-pass document, the shape most likely to hold a real stop
set.

**Fix (paste).** Add two rows to item 13's table for
`projects/documents/1901ab60-.../document.json` (5 passes) and
`projects/documents/307598da-.../document.json`, both `[]`, both "none needed", and change the
enumeration sentence to eleven `document.json` files across `shaderbox/resources/document_examples/`,
`projects/` and `projects/dev/` — the tracked set is `git ls-tree -r HEAD --name-only | grep
document.json`, not the two directories the parent named.

### 6. A click on a script-error row in a shader tab does not open the script. PARTIAL, docs/UX of item 8.

**Claim.** Item 8 says "a click on such a row opens the script tab at the line. This falls out of
passing `script_path_for` as the row's path instead of `tab.path`; no strip change is needed." The
path is necessary and not sufficient: nothing switches the active tab.

**Evidence.** `_draw_error_strip` sets `app.editor_jump_request = JumpRequest(err.path, err.line, 0)`
(`tabs/code.py:218`) and stops there. `_consume_jump` (`:182-193`) discards a request whose path is
not the CURRENT tab's: "A request for a different file is stale (one editor only); clear it". Nothing
else reads `editor_jump_request` to change tabs (`grep -rn editor_jump_request shaderbox/` returns
`code.py`, `app.py` rename/close bookkeeping, `lib_picker/filtering.py`, `widgets/uniform.py`). Both
existing cross-file jumps open the file FIRST and then set the request: `filtering.py:97-98`
(`app.open_shader_lib_file(fn.file)` then the request) and `widgets/uniform.py:69-72`
(`if path != app.current_editor_path: app.open_shader_lib_file(path)` then the request). Today the
strip needs neither because a script tab's rows carry `tab.path`, which is already current. Making
the row point elsewhere is what introduces the case.

**Fix (paste).** In item 8, replace "no strip change is needed" with: `_draw_error_strip`'s click
branch opens the target before latching the jump, following the two existing cross-file jumps
(`widgets/uniform.py:69-72`, `popups/lib_picker/filtering.py:97-98`) — for a script row that is
`app.open_script_for(tab.document_id)`, then the `JumpRequest`. Without it `_consume_jump`
(`tabs/code.py:187`) discards the request as stale on the very next frame, because the active tab is
still the shader.

### 7. The first frames of a multi-pass document drop broadcasts silently, and the spec does not say so. PARTIAL, D3 fidelity.

**Claim.** Item 2 says a held pass "is retried next tick". True, and the parent asked specifically
whether such values "are applied later or dropped". They are DROPPED for that frame — the tick writes
only what this tick produced, and there is no queue. The spec never says it, and on a fresh multi-pass
document the wait is several frames, not one.

**Evidence.** `ui.py:241` runs `session.tick(...)` BEFORE `document.render()` (`:253`), and the
first-render sweep at `:258-267` elects exactly ONE never-drawn pass per document per frame
("One never-drawn pass per document per frame draws its own chain"). Nothing else compiles a
non-output pass. So on a six-pass document opened cold, pass N first has a program around frame N,
and a bare-key broadcast aimed at it is dropped on frames 0..N-1. The user-visible consequence is not
the dropped value (the next tick writes it) but the DRIVEN set: `driven` is built per tick from what
routed, `script_driven_uniforms` reads `last_driven` (`engine.py:250`), and `widgets/uniform.py:187`
gates the play/stop button on it — so the row's play button appears a few frames late, one pass at a
time. Worth a sentence because the spec's own § Manual verification step 1 has the maintainer looking
at exactly this on a two-pass document.

**Fix (paste).** Add to item 2, after the `script_ready` paragraph: a held pass's values for that tick
are DROPPED, not queued — the next tick recomputes and writes them, and nothing accumulates. On a cold
multi-pass document the wait is one frame per pass, because `ui.py`'s first-render sweep compiles one
never-drawn pass per document per frame, so a six-pass document's last pass starts being driven around
frame six and its play/stop button appears then. This is the same budget every tile in the grid already
waits on (066 D2) and needs no new mechanism, but it is what the maintainer will see on step 1.

### 8. `_drop_script`'s loop needs an unpack, not only a wider tuple. Cosmetic, key propagation.

**Claim.** Item 5: "`_drop_script`'s cleanup loop and `drop_document`'s prefix filter (`k[0] ==
document_id`) both keep working on the wider tuple with no change beyond the tuple width."

**Evidence.** True of `drop_document` (`engine.py:472-474`, `k[0] == document_id` is width-agnostic).
Not true of `_drop_script` (`:323-325`), which iterates `for stale in scripts.last_driven |
scripts.last_skipped` and then builds `self.errors.pop((document_id, stale), None)`. With pair
elements that composes `(document_id, (pass, name))`, a two-tuple whose second element is a tuple —
never equal to the three-tuple key it means to pop, so every per-key error survives a script deletion
and the strip keeps showing errors for a script that no longer exists. It needs
`for pass_name, name in ...` and `(document_id, pass_name, name)`.

**Fix (paste).** In item 5, replace the sentence with: `drop_document`'s prefix filter (`k[0] ==
document_id`) is width-agnostic and needs no change; `_drop_script`'s cleanup loop (`engine.py:323`)
must UNPACK the pair (`for pass_name, name in scripts.last_driven | scripts.last_skipped`) and pop
`(document_id, pass_name, name)` — iterating the pairs as opaque elements would compose
`(document_id, (pass, name))` and pop nothing.

---

## D3 fidelity, item by item

Every clause of D3 is present and correctly bound, and the two-phase shape is the right call.

| D3 clause | In the spec | Verdict |
|---|---|---|
| Nested dict per pass | Item 2, "Pass block" | Correct |
| Bare key broadcasts to every pass declaring it | Item 2, "Bare key" | Correct |
| Pass block wins over broadcast | Item 2, "Pass block beats broadcast" | Correct, and pinned by a test whose second half varies insertion order — the one falsifier that distinguishes two phases from one loop plus a precedence test |
| Unknown pass / uniform = strip error | Item 3's three messages | Correct |
| Dict-value invariant asserted in `coerce_one` | Item 4 | Correct; `coerce_one(value, uniform, error_name)` at `behavior.py:251` takes the insertion cleanly, and the `ScriptError(name, kind, message)` positional form matches `errors.py` |

Traced by hand against the six cases the parent named:

- **(a) bare key declared by no pass.** Broadcast phase, lookup across every ready pass returns
  nothing, one error `no pass declares 'u_brsh' (orphan key)` with `pass_name=""` under
  `(doc, "", name)`. The `""` pass makes the stale-clear poppable without knowing which passes were
  consulted, which is the reason the spec gives and it holds.
- **(b) nested block for a real pass naming a uniform it does not declare.** The block's per-key path
  runs `_binding_reject` against that pass's active map, error `pass 'composite' has no active
  uniform 'u_b'`, `pass_name="composite"`. Correct and it reaches the composite shader tab.
- **(c) a bare key AND a nested block for the same uniform, where the nested pass does not declare
  it.** Broadcasts run FIRST and unconditionally, so every other declaring pass is written before the
  block is even looked at; the block then errors on its own pass alone. The broadcast survives. This
  is the case where a one-pass-with-precedence-test implementation would differ, and the spec's phase
  order gets it right.
- **(d) a dict under a nested block (double nesting).** `{"paint": {"u_brush": {"x": 0.5}}}` reaches
  `coerce_one` with a `dict`, which item 4 rejects with the grammar message rather than a shape hint
  about a float. Correct, and the reason to put the assert in the atom rather than in the tick.
- **(e) a pass whose program has not compiled.** Findings 4 and 7 above.
- **worked example.** Re-derived independently: three driven pairs, `composite` untouched by
  `u_brush`, strip empty. Matches.

One clause I checked for and did not find a hole in: the spec asserts the dispatch is "the value's
type and nothing else. Not 'is the key a pass name'". `normalize_output` (`outputs.py:169-178`)
passes an unknown type through unchanged and returns only `list` / `str` / the original for the known
ones, so no legal uniform value normalizes to a mapping. The invariant holds today by inspection, and
item 4 is what converts inspection into a gate.

## `(pass, name)` end to end

Enumerated independently rather than read off the spec.

- **The seven engine-facing `.render_pass` seams** are exactly the spec's:
  `project_session.py:430` (`_resolve_scripts` → `reload`), `:511`
  (`_load_one_document_from_disk` → `reload`), `:555` (`_make_export_isolation`'s closure →
  `tick_export`), `:577` (`reload_scripts` → `reload`), `:599` (`tick` → `tick`), `:683`
  (`write_script_source` → `reload`), `:687` (`write_script_source` → `dry_run`). Five methods, seven
  arguments. The parent named three.
- **Six further `.render_pass` occurrences are NOT engine-facing** and correctly stay: `:328` and
  `:482` read `source.path` for editor-session bookkeeping on delete, `:640` is
  `_scriptable_uniforms_for`'s stub scan (item 9 reshapes it for its own reason, not as a seam),
  `:514-516` iterates `passes.values()` already, and `:773-826` are pass create/rename. The spec does
  not list these as seams and is right not to.
- **`widgets/uniform.py`'s four call sites** confirmed at `:187` (`uniform_is_driven`), `:188`
  (`is_uniform_stopped`), `:302` (auto-stop-on-grab) and `:169` (inside `_draw_play_stop`), with
  `panel_pass` bound at `:177` and `Pass` carrying `source.path` not a name, so
  `pass_name_of(panel_pass.source.path)` is the right derivation. `tabs/document.py` has none of
  these calls; the parent's `:253` cite is a blank line. Both re-verified.
- **`app.py` wrappers** at `:1489` / `:1496`, not the parent's `:1414-1435` (which is
  `get_current_session_if_exists` / `flush_current_editor`). `toggle_current_document_play` (`:1503`)
  is a third caller and needs no edit since `set_document_all_stopped`'s arity is unchanged. Confirmed.
- **`DocumentScripts.last_driven` / `last_skipped`** at `engine.py:130` / `:134`, plus `last_good`
  (`:125`) and `warned` (`:137`) which the parent does not name. The spec catches all four.
- **Copilot.** `_pass_views` at `:730` with the driven set bound at `:738` and the loop at `:740`;
  the three other `_format_uniforms` callers at `:647`, `:720-721` and (the same) single-pass listing
  path — all four confirmed to pass a document-scoped set. The spec's correction (three more sites
  wrong by the same mechanism) is right, and `_format_uniforms(render_pass, driven: set[str])`
  (`:241`) keeping its name-keyed signature with a per-pass set built at each call site is the
  minimal shape. `ScriptProbe` consumers `_uniform_changes` (`:300`) and `_motion_verdict` (`:308`)
  index samples by name, as stated.
- **`ScriptStatus.soft_errors` has exactly one consumer**, `tabs/code.py:133`. The working-set view
  carries only the sentinel (`backend.py:711-713`), so the widened tuple does not leak into the
  copilot. Verified by grep; the spec does not claim otherwise but does not state it either, and it
  is what makes item 8's change safely local.

Finding 3 is the one site left mis-specified; finding 8 is the one site left under-specified.

## Persistence

- `StoppedKey` as a frozen `BaseModel` is correct and hashable. Executed: a frozen pydantic model
  hashes, so `frozenset[StoppedKey]` works as the engine parameter.
- **The zero-bytes claim is right.** Every persisted `stopped_uniforms` in the tree is `[]` (two
  examples omit the key entirely and default). Executed `drop_invalid` against a real
  `list[StoppedKey]` field: a stale `["u_x"]` logs one
  `Ignoring invalid document 'x'.stopped_uniforms (1 error(s))` and drops to `[]` with the sibling
  `all_stopped` surviving, and `model_salvage.py` needs no edit. The spec's clarification that this
  is WHOLE-LIST and not element-level salvage is also right and I reproduced it: a mixed
  `[{"pass_name":..,"name":..}, "u_y"]` loses the valid pair too, because the element branch
  (`model_salvage.py:74-76`) passes a non-dict element through untouched and the top-level
  `validate_assignment` (`:91`) then rejects the list.
- **The parent's manual-verification line must be rewritten**, and the spec is right to say so: no
  document on disk is stale, so the first launch logs nothing. The spec's § Manual verification step
  6 replaces it with a deliberate way to exercise the path once, which is the right shape — it keeps
  the fail-soft path verified rather than assumed.
- `test_persistence_completeness.py` needs no edit: its roster is the four app-data stores, it
  exempts `document.py` by name (`:106`), and `ui_models.py` is already rostered (`:101`), so a
  nested model changes nothing the completeness grep asserts. Confirmed.
- Finding 5 is the gap: two tracked files outside the enumerated directories.

## Mouse

- `MouseState` gaining `down` / `prev_x` / `prev_y` with `x`, `y` first is correct: `EXPORT_MOUSE`
  and `ui.py:659` are the only two construction sites and both are positional-friendly, and
  `EngineContext.mouse` defaults via `field(default_factory=...)` so the bare-clock sites are
  untouched (`context.py:21-27`).
- The `ui.py` fill lands inside the existing hit-test at `:654-659` with no second hit-test.
  `item_normalized_mouse` (`ui_primitives.py:403-428`) already ANDs
  `is_window_hovered(child_windows)` into `inside`, so a popup over the canvas suppresses both
  position and `down` — the spec's claim, verified in the helper's body.
- The re-entry design is right and the spec is honest that the naive version draws a line across a
  gap the user did not draw. `App.script_mouse_inside` beside `script_mouse` (`app.py:1140`) is the
  minimal carrier. One thing the spec's snippet leaves implicit: the else branch has to cover BOTH
  `hit is None` (imgui's invalid-mouse sentinel, `ui_primitives.py:419-420`) and `hit[2] == False`,
  and `_draw_document_image`'s no-document branch (`ui.py:660-680`) never reaches either, so a
  document closed mid-drag leaves `down=True` latched until the next document draws. Cheap fix worth
  a clause: clear `script_mouse_inside` and `down` at the top of the with-document branch rather than
  only in the hit test's else.
- `EXPORT_MOUSE(0.5, 0.5, False, 0.5, 0.5)` with `down=False` and prev == current is exactly what
  #22 and the parent ask for, and the added export test's falsifier ("default `down=True`, or wire
  the live cursor into the export context") is the right pair.
- `api_doc.py` regenerating from `MouseState.__dataclass_fields__`: confirmed at `:61`, and the
  spec's diagnosis of WHY the stub cannot be read by `api_doc.py` is right —
  `test_api_doc_reaches_only_for_the_gl_free_half_of_the_package` (`:156-164`) parses `api_doc.py`'s
  own imports by AST, and `script_stub_for` lives in `engine.py`, which imports `moderngl`.
  **On the parent's question — is authored prose pinned by a test the right shape, or should the
  grammar sentence be generated from a GL-free source?** Authored prose is right here, and the
  existing module already answers the question: `api_doc.py`'s docstring states the split ("Names,
  signatures and field types come from the code; the semantics beside them are authored"), and every
  gloss in the file (`_CTX_GLOSS`, `_VALUE_SHAPE_GLOSS`, `_VEC_OPERATOR_GLOSS`) is authored prose
  joined to the code by a test. The D3 grammar is semantics, not a name or a type — there is no
  GL-free source that HOLDS it, because the routing lives in `engine.py` behind moderngl, and the
  alternative (lifting a grammar string into `context.py` so `api_doc` can import it) would put prose
  in a module about the ctx dataclass purely to satisfy an import rule. The precedent that settles it
  is `test_every_stub_kind_type_name_has_a_value_shape_gloss` (`:118-125`), which joins `api_doc` to
  `_stub_kind` by READING ITS SOURCE with `ast` rather than importing it — a test-time join,
  deliberately not an import. The spec's shape is the same one, and it is the house pattern.
  Finding 1 is about the execution of that shape, not the shape.

## The `RESET_FEEDBACK` command

- **The chord is right and free.** `02_keybindings.md:140` assigns F6 with note 3's reasoning (rule
  3 forces Alt or an F-key because the verb must survive editor focus; F6 sits beside F5, where W-E
  moves `TOGGLE_DOCUMENT_PLAY`). Only F1 (`HELP`, `commands.py:186`) and F8 (`JUMP_NEXT_ERROR`,
  `:141`) are bound today. `_STANDALONE_KEYS` (`:287-289`) is F1-F12 and `chord_needs_modifier`
  (`:313-317`) exempts them, so no registry change. Worth naming a false alarm: `tabs/code.py:221`
  says "Clickable toggle (F6): expand to all errors" — that is 047's FINDING F6, not a chord; nothing
  binds a key to `errors_expanded` (`grep -rn errors_expanded` returns only `app.py:439, 1273` and
  `code.py:199, 223, 228`). No collision.
- **`Document.reset_feedback` is callable as-is.** `document.py:357-368` releases every history
  canvas, clears both maps and resets `_frame = -1`. **On the parent's question of a pass
  mid-iteration**: nothing is in flight when the callback runs. The command fires from
  `App._build_command_callbacks` during the frame's input phase, while `render()` — the only thing
  that iterates (`document.py:472-500`) and the only caller of `_swap_feedback` besides `begin_frame`
  — has already returned for the previous frame and has not begun for this one. Within `render`, a
  pass's iterations run to completion inside one call (`:472`), so there is no yield point a command
  could land in. `_frame = -1` is the detail that makes the next `begin_frame` a clean start rather
  than a same-frame no-op (`:306-307` returns early when `frame == self._frame`). The spec's "called
  as-is, with no change" holds; I would add the one clause about `_frame = -1` being why, since the
  parent asked.
- The export path already calls it inside `export_isolation` (`document.py:695-699`), so an export
  starting from black is unchanged by this wave.
- **Open question 4 (hidden vs disabled): hidden is right**, and the reason the spec gives (a
  permanently-disabled control on the primary viewing surface, on most documents) is the stronger
  half. But finding 2 makes the gate itself wrong, and with `_feedback` as the source the answer
  would be neither hidden nor disabled but *flickering*, which no reading of the question sanctions.
- Placement: `ui.py:693-699` anchors the FPS chip to the preview's top-right via
  `fps_overlay(anchor_x=cursor_pos.x + image_width, ...)`, so the top-left mirror is the same
  mechanism and the corners cannot collide. `ghost_button` exists (`ui_primitives.py:97`). One word,
  `Clear`, is within D1.
- `help_content.py`'s shortcuts section enumerating every bound command, pinned by
  `test_shortcuts_section_lists_every_bound_command`: the spec's "appears there with no edit" is the
  claim; I did not re-run the test, so treat that one as unverified by me.

## The shader-tab soft-error strip

- `_script_errors_for` at `tabs/code.py:130` and the `errors` selection at `:517-523` are as cited,
  and today a shader tab gets `edited.compile_unit.errors` and nothing else. Confirmed.
- Compile errors first, script errors after, is the right order given `_MAX_ERROR_ROWS` caps the
  visible rows.
- The markers do NOT leak: `_apply_markers` (`:160-165`) filters the fingerprint to
  `err.path == current_path`, so a script error carrying the script path adds no line-fill to the
  shader editor. The spec does not claim this, but it is the thing that would have broken, and it
  holds for free.
- The sentinel staying on the script tab only is right: it belongs to no pass.
- Finding 6 is the gap in the click path.
- **The console warning goes.** `engine.py:587`'s `logger.warning`, the `warned` set (`:137`), the
  `warn: bool` parameter and its two `warn=False` call sites (`tick_export` at `:405`, `dry_run` at
  `:456`) are all as described. This is #29's own last sentence and the wave closes it.

## The stub

- `_scriptable_uniforms_for` returning a per-pass mapping is the right fix and the spec's reason for
  why the compile cost is acceptable is correct: its two callers are `create_script`
  (`project_session.py:652`) and `read_script_source` (`:669`), both user or agent triggered, never
  the frame loop, so 066 D1 is not in play.
- **The single-pass question, which the parent asked specifically: the spec answers it and answers it
  right.** § Tests states it as a design consequence, not a convenience: "Scripts returning
  `{"u_x": ...}` need NO edit at all under the broadcast rule: a single-pass document that declares
  `u_x` receives it. That is the reason the broadcast rule keeps this rewrite finite". So the nested
  shape is never forced, and a one-pass script keeps working unchanged. What the spec does NOT say is
  what the STUB emits for a one-pass document — item 9's template shows the bare-key rule as a
  comment line and then per-pass blocks, so a one-pass document gets one block header plus the
  bare-key line, which is more ceremony than the case needs but is consistent and self-explaining.
  Acceptable; flagging only because the parent asked whether the nested shape is forced, and the stub
  is the only place a reader could conclude that it is.
- **The tutorial's paint step against the nested form**: out of this wave's scope by the spec's own
  statement (the 068 D7 line says "The tutorial's paint step is rewritten against them in 069 W-H"),
  and D3's broadcast rule means the paint step could legitimately use either form. Since `paint` is
  a non-output pass whose `u_brush` no other pass declares, the bare key is sufficient and is the
  simpler teaching shape. Worth one sentence in the 068 D7 supersession line so W-H does not
  default to the nested form for a case that does not need it.
- The empty-pass block emitting `#     (no scriptable uniforms)` rather than being omitted is the
  right call (the user learns the pass exists), and `_script_import_line` taking the union across
  passes is necessary — `script_stub_for` derives it from `kinds` (`engine.py:209`), so a per-pass
  list would drop a `Vec2` that only one pass needs.
- `ast.parse` in the stub test is a real falsifier for a separate defect class (comment text that is
  not valid Python), not padding.

## Docs

- `conventions.md`'s scripting entry begins at `:288`, "A broken script is **error-as-data**" is at
  `:308`, "PLAY/STOP is document-scoped + name-keyed model state" at `:314`, `EngineNode` named at
  `:299`, `stopped_uniforms: list[str]` at `:316`, freeze granularity `(document_id, name)` at
  `:310`. Every clause the spec proposes to rewrite is where it says, and the three surgical edits
  are the right granularity — the entry's lazy-row argument, its list-not-set reason and its
  auto-stop-on-grab clause are all still true after the change and correctly stay.
- 065 D12's supersession line and 068 D7's retraction-lifted line are both accurate against those
  specs, and leaving each body intact is right: D12's diagnosis is why D3 exists, and D7's two
  reasons are the record of why the retraction was right then.
- `help_content.py:137-139` is the sentence cited and one added sentence fits D1's budget.
- **The copilot prompt block — the SKILL's tier rules.** The grammar sentence lands in the RARE tier
  via `script_api_summary()` → `prompt_context.py:99` → `_context_block` (`prompt.py:38-53`) →
  `PromptBlock("project_context", Volatility.RARE, ...)` (`:435`). That is the right tier under the
  SKILL's §4 rule "a rule's HOME follows WHEN it fires: pre-action stays STATIC": the grammar governs
  what the model writes BEFORE any tool result exists, and the RARE block is a system message present
  on every turn in the cacheable prefix, so it is available pre-action exactly as a STATIC rule would
  be. It is not a per-turn result gloss and must not become one. Token cost: the change replaces one
  clause of an existing bullet with two sentences and extends one gloss — on the order of 40 tokens
  added to a block that already carries the project map, the lib catalogue, the example catalogue and
  the conventions, and it sits in the cached prefix so it is re-billed at the cache rate. Under the
  SKILL's §6 that is not a cost decision worth measuring. The spec's claim that `prompt.py` and
  `prompt_context.py` need no edit is right: both just render whatever the generator returns, which
  is the property 059 D3 built the generator for.
- `copilot/capabilities.py` changing only in field comments: consistent with the tool signatures
  staying pass-free, which is right — `read_script` / `write_script` address the document, and the
  routing verdict comes back in the RESULT (the dotted `pass.uniform` display form), which is the
  SKILL's "enrich an existing tool's result rather than add a tool" rule applied correctly.

## The four closed open questions

1. **A bare key says nothing about the passes it skipped.** Agree. The design note states the rule
   and the spec's reason is the decisive one: a brush uniform declared only on `paint` is the
   INTENDED case, so an informational row per skipped pass would fire on every frame of the wave's
   own motivating example and train the user to ignore the strip. The revisit trigger (a user reports
   a broadcast not reaching a pass they believed declared it) is falsifiable and the proposed fix (a
   hover readout on the row, not a strip row) keeps the strip for real errors.
2. **`script_ready` as a protocol member rather than catching the compile.** Agree with the choice,
   disagree with its stated condition — finding 4. The member is right for the reason given (the
   engine never triggers a compile, so 066 D1 holds by construction rather than by exception
   handling, and swallowing a compile error inside the script path would hide a real shader failure).
   The defect is that "the program is built" conflates never-attempted with failed, and `core.py:236`
   already provides the two-part condition that separates them.
3. **The `_draw_play_stop` imgui id change.** Agree, and it is under-sold rather than over-sold. Two
   passes declaring the same uniform name produce `f"u_{name}"` twice today (`widgets/uniform.py:168`),
   which is one imgui id for two rows — a real state bug (the two toggles share hover/active state),
   not just a latent one. The pass prefix fixes it as a side effect. Naming it so a reviewer does not
   read the id change as cosmetic is exactly right.
4. **Hidden rather than disabled on a no-feedback document.** Agree on the answer. The reason is
   sound and matches #4's rule. But see finding 2: the gate as specified does not compute the
   predicate the question is about.

## False trails

- **F6 vs the error strip's "(F6)" comment** (`tabs/code.py:221`) — that is feature 047's finding F6, not a chord; nothing binds a key to `errors_expanded`. No collision.
- **`_apply_markers` leaking script line numbers into the shader editor** — it filters on `err.path == current_path` (`:164`), so a script-path row adds no marker to the shader. Safe as specified.
- **`normalize_output` needing a dict guard** — the spec puts the rejection in `coerce_one` and says `outputs.py` is unchanged; correct, since `normalize_output` passes unknowns through by design (`outputs.py:172-173`) and the atom is where the invariant is load-bearing.
- **`set_document_all_stopped` needing a pass** — the parent lists it among the signatures that gain one; the spec says it does not and is right, it is document-wide by definition and only the type it clears changes.
- **`model_salvage.py` needing element-level support for `StoppedKey`** — executed; no edit needed, and the whole-list drop is the correct outcome for a stop set.
- **`test_persistence_completeness.py` being in the blast radius** — its roster is the four app-data stores and it exempts `document.py` by name (`:106`). No edit.
- **`_freeze` needing a per-pass variant** — the spec's single signature taking the document and resolving each pair is right; the sink variant staying a flat dict keyed by pairs preserves the dry-run isolation exactly.
- **`tabs/document.py` needing an edit** — it has no stop/driven call at all; every one is in `widgets/uniform.py`. The parent's `:253` is a blank line.

## Coverage statement

Read in full at `0ce84f8`: `scripting/engine.py`, `scripting/context.py`, `scripting/api_doc.py`,
`scripting/outputs.py` (the normalize path), `scripting/behavior.py` (`coerce_one`),
`scripting/errors.py`, `project_session.py` (every `render_pass` occurrence, enumerated by grep and
classified individually), `widgets/uniform.py` (`_draw_play_stop` + `draw_ui_uniform`),
`tabs/code.py` (`_script_errors_for`, `_apply_markers`, `_consume_jump`, `_draw_error_strip`, the
`errors` selection), `ui.py` (`_draw_document_image`'s hit test, `_draw_app_panel`'s anchoring, the
tick/first-render ordering), `ui_primitives.py::item_normalized_mouse`, `document.py`
(`_feedback` lifecycle, `reset_feedback`, `render`'s iteration loop, `render_media`),
`pass_graph.py` (`PassPlan.feedback`, `PassEntry`), `core.py::Pass.get_active_uniforms`,
`ui_models.py` (the stopped field), `model_salvage.py::drop_invalid`, `commands.py` (F-keys,
`_STANDALONE_KEYS`, `chord_needs_modifier`), `app.py` (`panel_pass`, the two wrappers,
`script_mouse`, `open_script_for`), `copilot/backend.py` (`_pass_views`, the three other
`_format_uniforms` callers, the `set_uniform` gate, the probe render sites, `_motion_verdict`),
`copilot/prompt.py` (the tier machinery and `_context_block`), `help_content.py`,
`tests/test_script_api_doc.py`, `tests/test_persistence_completeness.py`.

Executed rather than reasoned: the `drop_invalid` salvage against a real `list[StoppedKey]` field
(finding evidence for the persistence section, including the mixed-list case and the frozen model's
hashability); the eleven-file `document.json` enumeration and each file's `stopped_uniforms` value
(finding 5); the seven test files' line/test counts (all seven of the spec's figures match exactly).

Also verified after the first pass: `_shortcuts_section` (`help_content.py:72-93`) builds its
snippet by enumerating `COMMAND_SPECS` per category, and
`tests/test_help_content.py:43-51` asserts each bound spec's label AND rendered chord are in that
snippet — so the spec's "appears there with no edit" holds, and the test goes red if the new spec
is added without a `default_chord`. 065 D12 (`065_pass_graph/01_spec.md:133-140`) and 068 D7
(`068_radiance_cascades/01_spec.md:87-95`) read directly: D12's body is the addressing-hole
diagnosis the supersession line preserves, and D7's two stated reasons are exactly the two things
this wave removes, so both proposed lines are accurate.

Nothing in the coverage list above is second-hand.

---

# Round 2 (closure)

Narrow re-read of `60_wave_g_scripting.md` (now 1376 lines) against the eight round-1 findings, plus
the two re-traces and the one contested open question the coordinator named. Only the folded text was
read; no new surface reviewed.

## Verdict: PARTIAL

Seven of eight CLOSED. Finding 4 is NOT CLOSED: the prose is right and the code expression beside it
is its exact negation, so the spec as written specifies the opposite of what it argues for. One
line, mechanical, and it must be fixed before implementation because the expression is what an
implementer will copy.

| # | Finding | Verdict |
|---|---|---|
| 1 | api_doc gloss breaks the caveat test | **CLOSED** |
| 2 | `has_feedback` over `_feedback` | **CLOSED** |
| 3 | `set_uniform` gate out of scope at `:879` | **CLOSED** |
| 4 | `script_ready` holds a broken pass forever | **NOT CLOSED** — the expression is inverted |
| 5 | Disk enumeration missed two documents | **CLOSED** |
| 6 | Script-error row does not open the script | **CLOSED** |
| 7 | First frames drop broadcasts silently | **CLOSED** |
| 8 | `_drop_script` needs an unpack | **CLOSED** |

## Per finding

### 1. CLOSED

`:673-679`. The gloss is now `... down = LMB over the canvas -- FROZEN at {at} on export and in the
headless probe, where down is False and prev equals x/y`, and the fold adds the rule that generated
it: "**Keeping the caveat contiguous is a constraint on the rewrite, not an observation about it** …
any wording that inserts words between `at 0.5,0.5` and `on export` turns it red. The new-field
clauses therefore go AFTER the caveat, never inside it." The asserted substring
`FROZEN at 0.5,0.5 on export and in the headless probe` is contiguous in the new sentence, so
`test_the_mouse_gloss_carries_the_frozen_at_center_caveat` (`tests/test_script_api_doc.py:109-115`)
passes unedited. Naming the constraint rather than only the corrected string is the stronger fix: it
survives the next rewording.

### 2. CLOSED, re-traced in both states

`:652-659` and the files table at `:864`. `has_feedback` is now
`bool(plan_passes(self.graph)[0].feedback)`, with `_feedback` explicitly ruled out as the source and
all three emptiers named (`release()`, `drop_feedback`, and `reset_feedback` itself).

Re-traced as asked, by execution rather than by reading:

- **Freshly loaded feedback document, nothing rendered.** `plan_passes` fills `feedback` at
  `pass_graph.py:274` from `entry.inputs` alone (`if source == name: feedback.add(name)`), which is
  persisted `graph.json` data. Ran it against the three tracked documents that declare a self-edge:
  `1c4f8a20` → `{'trail'}`, `77a84d27` → `{'cascade', 'jfa'}`, `1901ab60` → `{'trail'}`, and the
  single-pass `0b0d16bb` → `set()`. All computed with no GL context, no `Document`, no render. So the
  button is present on frame 0 of a feedback document and absent on a non-feedback one, which is the
  state the `_feedback` version got wrong.
- **Immediately after `reset_feedback`.** `document.py:357-368` touches `_feedback`,
  `_feedback_generation` and `_frame`, and nothing else. `self.graph` is untouched, so
  `plan_passes(self.graph)[0].feedback` is byte-identical before and after the click and the button
  stays. The self-hiding behaviour is gone.

The premises table also gains a row for it (`:1237`), which is the right place: the defect was a
premise error, not a design error.

### 3. CLOSED

`:398-406`. The gate now MOVES below the `target = ...` bind at `:887` and asks
`(pass_name_of(target.render_pass.source.path), name) in driven`, with the reason stated ("nothing
naming the output pass is in scope above that bind, so the check as written at `:879` does not
compile") and the secondary benefit kept (past `uniform is None`, so a name that is not on the output
pass gets the "no active uniform" answer). Matches the fix sentence exactly.

### 4. NOT CLOSED — the expression is the negation of the prose

`:137-142`. The fold gets the DESIGN right and the CODE wrong, and they contradict each other in the
same paragraph.

The prose is correct throughout: the protocol comment reads "False only while the pass has NEVER
ATTEMPTED a compile"; the body splits never-attempted (held, no error) from failed (ready-but-empty,
ordinary orphan path); `core.py:232` is quoted for why a failed attempt is never retried; and the
premises table (`:1219`) states the split cleanly. That is the whole of my finding, folded.

The expression beside it is inverted. `:142` reads: "`Pass` satisfies `script_ready` with a property
over `program is None and not compile_unit.error_raw`, the same pair `get_active_uniforms` itself
tests before compiling (`core.py:236`)". That pair IS `get_active_uniforms`'s *would-compile* test
(`core.py:236`: `if self.program is None and not self.compile_unit.error_raw: self.compile()`) — it
is True exactly when the pass has never attempted, which is precisely when `script_ready` must be
False. Executed the truth table over the three reachable states:

| Pass state | `program is None and not error_raw` | `script_ready` the prose requires | agree? |
|---|---|---|---|
| never attempted | `True` | `False` | no |
| compile FAILED | `False` | `True` | no |
| compiled OK | `False` | `True` | no |

Wrong in all three. Implemented as written, the polarity flips the whole guard: a never-attempted
pass reads ready and the tick calls `get_active_uniforms()` on it, which compiles it from inside the
script tick — the exact 066 D1 violation `script_ready` exists to prevent, and on a six-pass document
it compiles all six on the first tick, which is the cost open question 2 defends the member against.
Meanwhile a compiled-OK pass reads not-ready and is held forever, so nothing is ever driven. My
round-1 fix text carried the same expression, so this is my error propagated, not a drafting slip.

**Fix (paste).** In item 2, replace the sentence at `:142` with: `Pass` satisfies `script_ready` with
a property returning `self.program is not None or bool(self.compile_unit.error_raw)` — the NEGATION
of the would-compile test `get_active_uniforms` runs at `core.py:236`, so the protocol member is
False on exactly the states that call would compile and the two agree by construction. Equivalently
and more directly: `not (self.program is None and not self.compile_unit.error_raw)`.

Everything else in item 2 stands. This is one expression, and the paragraph around it already argues
for the right behaviour.

### 5. CLOSED

`:714-731`. The table now carries eleven rows including
`projects/documents/1901ab60-.../document.json` (5 passes) and `.../307598da-.../document.json`, both
`[]`, and the conclusion is restated as "eleven files verified and zero bytes changed". The fold also
records why they were missed ("outside the two directories the parent named") and that `projects/` is
tracked exactly as `projects/dev/` is. The premises row (`:1189`) carries the same correction. The
zero-bytes verdict is unchanged, which is what I expected; the enumeration is now complete.

### 6. CLOSED

`:455-465`. The path change is kept and explicitly demoted to "necessary and not sufficient", with
`_consume_jump`'s discard quoted and both cross-file precedents cited, and the fix is
`app.open_script_for(tab.document_id)` before latching. The premises table gains a refutation row
(`:1205`). The `_apply_markers` note is retained as the thing that does NOT need fixing, which keeps
the boundary of the change visible.

### 7. CLOSED

`:159-165`, plus manual verification at `:1132-1133`. "**A held pass's values for that tick are
DROPPED, not queued.** The next tick recomputes and writes them and nothing accumulates", followed by
the one-frame-per-pass consequence with `ui.py:258-267` cited and the play/stop button's late
appearance named. Folding it into manual verification too is right — it is a thing the maintainer
sees, not a thing the code does wrong.

### 8. CLOSED

`:306-312`. `drop_document`'s filter is stated as width-agnostic and `_drop_script`'s loop is given
the unpack (`for pass_name, name in ...`) with the popped key spelled out and the failure mode
explained (a two-tuple containing a tuple never equals the three-tuple key). Premises row at `:1206`.

## Open question 3 — I concede

`:1262-1275` resolves it against me and is right. Read `tabs/document.py:261-262`:

```python
        for hash in sorted_hashes:
            draw_ui_uniform(app, ui_uniforms[hash])
```

`sorted_hashes` derives from `active_uniform_hashes`, built at `:212` by looping
`app.panel_pass(app.current_document_id).get_active_uniforms()` — ONE pass. `App.panel_pass`
(`app.py:606-617`) returns a single `Pass`. And this is the only `draw_ui_uniform` call site in the
codebase. So two passes' rows of the same uniform name are never submitted in the same frame, and
`play_stop_toggle`'s `f"{label}##play_stop_{id_}"` (`ui_primitives.py:208`) never sees a duplicate id
within a frame. My round-1 claim that "the two toggles share hover/active state" required both rows
to be live at once, which the draw loop makes impossible.

I withdraw it. The tests reviewer is right, and the spec's resolution states the residual value
correctly: the prefix keeps a row's imgui state from carrying across a panel-pass switch, which is a
correctness nicety, not a defect fixed. The spec's phrasing ("neither cosmetic nor as closing a
defect") is the accurate reading, and it now cites the line that decides it rather than either
reviewer's assertion.

## False trails this round

None raised. Per the late-round rule I checked each remaining disagreement for whether it was a
preference and dropped it if so: the stub's ceremony on a one-pass document (item 9 emits a block
header plus the bare-key line where a bare list would do) is a taste call with no defect behind it,
and I am not restating it as a finding. Same for the `ui.py` else-branch clause I raised in round 1's
mouse section — the spec's `script_mouse_inside` carries it, and my preferred placement is a
preference.

## What I verified this round

Executed: the `script_ready` truth table over the three reachable `Pass` states (finding 4);
`plan_passes` against `1c4f8a20`, `77a84d27`, `1901ab60` and `0b0d16bb` with no GL, in the
freshly-loaded state (finding 2). Read at `0ce84f8`: `core.py:230-247`, `pass_graph.py:265-279`,
`document.py:357-368`, `tabs/document.py:207-263`, `ui_primitives.py:200-212`,
`tests/test_script_api_doc.py:109-115`. Read in the folded spec: items 2, 5, 7, 8, 11, 12, 13, the
files table, the premises rows added this round, open questions 2 and 3, and the round-1 fold
summary at `:1310-1338`.
