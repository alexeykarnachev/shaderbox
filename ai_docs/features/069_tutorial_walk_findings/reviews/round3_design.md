# 069 — pre-impl review, round 3: the five rewritten bullets

Reviewer role: design/implementability, third round. **Scope is narrow by instruction**: the five
bullets rewritten after round 2 — W-A's "Resolution control redesign", W-B's "Gate", W-C's "First
render of every pass", W-D's "Default wiring by name is a RESOLUTION rule", W-G's "Driven / stopped
state". Nothing else in the spec was reviewed. Read-only except this file.

Anchor: the code on `dev` at `1767483`. Round 2's "False trails" are taken as verified and were not
re-probed. Every claim below names the file I opened; Task 3 is demonstrated by a script that ran.

---

## Task 1 — closure of round 2's checklist

### Round 2 Task 1(a) — W-C `render(target=…)`

Round 2 listed **four undecided points plus two uncovered `output` reads (sites 2 and 4)** and one
unstated export fact.

**(a) site 2 — the early-out guard.** **Closed.**

> "(1) `target` (or the graph output when None) feeds the early-out guard, `plan_for_output`, and
> the cycle fallback `order = [output]` — all three, not two"

Names the guard explicitly. Maps to `document.py:392` (`if output is None or output not in
self.passes: … return`).

**(a) site 4 — the cycle fallback.** **Closed.** Same sentence, "and the cycle fallback
`order = [output]` — all three, not two". Maps to `document.py:397-400`.

**(a) point 2 — target alone vs the whole ancestor chain.** **Closed.**

> "(2) a target draws its WHOLE ancestor chain (a pass alone would sample black inputs and show a
> wrong picture), but skips any pass already drawn this frame — `Pass.drawn_frame: int` set by
> `render` against the document's frame counter — so shared ancestors draw once per frame and an
> iterated feedback pass never advances twice"

The 21-vs-6 draw blow-up round 2 measured is answered by the per-frame skip, not by narrowing the
chain. Correct direction; see Task 2(b) for the one hole in *"the document's frame counter"*.

**(a) point 3 — does `canvas is None → first_render_done = True` apply to a `target=` call?**
**Closed differently.** The spec does not re-guard the document flag; instead it splits the two
flags apart:

> "(4) `Pass.first_render_done` is set by `render` on every pass it draws; `Document.first_render_done`
> keeps its meaning (output drawn, `canvas is None and target is None`)."

`canvas is None and target is None` is a strictly narrower condition than today's `canvas is None`
(`document.py:389-390`), so the hazard round 2 named — a `render(target=intermediate)` admitting the
document into the 066 D2 "already rendered" set before its output ever drew — cannot fire. This
closes the point better than the guard round 2 asked for, because it is one added conjunct at one
line rather than a new rule.

**(a) point 4 — `begin_frame` / iterated-pass double-swap.** **Closed.** The `drawn_frame` skip in
decision (2) is stated in exactly those terms: "an iterated feedback pass never advances twice".
`document.py:449`'s `_swap_feedback` between iterations now runs at most once per pass per frame,
because a second `render(target=…)` for a pass already drawn skips the whole iteration loop.

**(a) — who writes `Pass.first_render_done`, the two non-equivalent candidates.** **Closed.**

> "`Pass.first_render_done` is set by `render` on every pass it draws"

Picks the first of round 2's two candidates by name (inside `Document.render`, not the `ui.py`
gate), which is the one that collapses the sweep instead of running it for `len(passes)` frames.

**(a) — export not threaded.** **Closed.**

> "Export is untouched: `target` is not threaded through `_render_image` / `_render_media_into`, by
> intent."

This is the sentence round 2 asked for verbatim.

**(a) — the `_graph_errors` side effect (round 2's site 7).** **Still open** — but see the note.
Round 2 flagged that `render(target=name)` overwrites document-scoped `_graph_errors` with a
partial plan's errors, called it "happens to be harmless today" because `plan_for_output`
(`pass_graph.py:362`) returns whole-graph errors, and asked for a pin. The rewritten bullet says
nothing about `_graph_errors`. **This is a preference, not a defect**: the code is correct today and
the spec's own decision (1) routes `target` into `plan_for_output`, whose error half is
target-independent by construction (`plan_passes(graph)` is called on the whole graph before
`_order_for` narrows the order). I record it as unclosed for completeness, not as a blocker.

**Verdict for 1(a): 7 of 8 closed (1 closed differently), 1 left open by choice and harmless.**

---

### Round 2 Task 1(b) — W-G pass-qualified stopped/driven

Round 2 named **four unnamed consumers, one wrong example count, and one undecided persistence
shape**.

**(b1) `App.set_uniform_stopped` / `App.set_document_all_stopped` (`app.py:1414-1435`).** **Closed.**

> "and so do their `App` wrappers (`App.set_uniform_stopped` / `set_document_all_stopped`,
> `app.py:1414-1435`)"

Verified against the file: both wrappers exist at those lines and forward to the session.

**(b2) `tabs/document.py:253`.** **Closed.**

> "and the panel's call site (`tabs/document.py:253`)"

Verified: `app.set_document_all_stopped(document_id, playing)` at `tabs/document.py:253`. Round 2
asked the spec to *state a decision* about whether document-level stop goes pass-qualified; the spec
names the site as one that changes. That is a decision, stated.

**(b3) `copilot/backend.py:881` — the `set_uniform` gate, and the corrected motivation.**
**Closed.**

> "In `copilot/backend.py` the site this genuinely fixes is `_pass_views` (`:740`), which loops
> every pass against one document-scoped driven set; the `set_uniform` gate (`:889`) resolves
> against `render_pass` only and stays name-keyed within that pass."

Verified both: `backend.py:735-750` loops `for name in sorted(document.passes)` while holding one
`driven = self._get_script_driven_uniforms(full_id)` (`:740`); the `set_uniform` path resolves
`target.render_pass.get_active_uniforms()` (`:890-892`), output-pass only. The spec now picks round
2's second option (the gate hardcodes the output pass, stays name-keyed) and drops the wrong
motivation. Both halves of round 2's finding are folded.

**(b4) `ProjectSession.tick` (`:598`) and the `stopped` type / `last_driven` / `last_skipped`.**
**Closed.**

> "the `EngineNode` `stopped: frozenset[str]` parameter type and `DocumentScripts.last_driven` /
> `last_skipped` become `(pass, name)`-keyed; the live tick (`project_session.py:598`) is the fourth
> `.render_pass` seam and the one whose protocol type changes"

Names the site, calls it the fourth seam (answering round 2's F3 in the same clause), and names the
type change and both `DocumentScripts` fields. Round 2 also named `ScriptStatus.driven_count`
(`engine.py:272`) — that follows from `last_driven` changing shape and is not separately required.

**(b5) the example count.** **Closed.**

> "Seven shipped examples exist; five persist `stopped_uniforms` and are hand-edited; the two that
> do not are exactly the multi-pass ones."

This is round 2's correction restated, including its point (the two documents where a pass-qualified
key means anything have nothing to edit).

**(b6) the persistence shape decision.** **Closed.**

> "Persisted shape: `stopped_uniforms: list[StoppedKey]` with `StoppedKey(pass: str, name: str)` a
> `BaseModel` — element-level salvage, and a stale `list[str]` fails `validate_assignment` and drops
> to `[]` under the existing `drop_invalid` policy (`model_salvage` needs no change)."

Picks one of round 2's three candidates (`list[PairModel]`), and states the reason (element-level
salvage) round 2 said had to be decided. I verified the salvage claim against
`model_salvage.py:60-83`: `drop_invalid`'s nested branch tests
`get_args(field.annotation)` for a `BaseModel` subclass, which `list[StoppedKey]` satisfies and
`list[list[str]]` does not — so element-level salvage is real for the chosen shape. A stale
`list[str]` file: each element is a `str`, not a `dict`, so the per-element loop `kept.append(item)`s
it unchanged, and the final `validate_assignment` then rejects the whole field and drops it to `[]`.
Exactly as the spec says.

**Verdict for 1(b): 6 of 6 closed.**

---

### Round 2 Task 1(c) — W-D default wiring (round 2 verdict: FAIL)

Round 2's FAIL had three legs: **the hot-reload seam does not exist**; **`"(none)"` is
indistinguishable from never-wired**; **the copilot `edit_shader` asymmetry**. Plus the dangling-edge
third state.

**(c1) the non-existent hot-reload seam.** **Closed differently — the design is replaced, not
patched.**

> "Default wiring by name is a RESOLUTION rule, not a stored edge. Two facts force that: the
> hot-reload seam (`watch.py::_reload_pass_if_changed`) has no graph and no compile — it
> `invalidate()`s and the samplers are unknowable there (forcing a compile inverts 066 D1)"

The spec now cites round 2's own finding as the reason the stored-edge shape is abandoned. Round 2's
three-way choice (i)/(ii)/(iii) disappears with the seam: there is no wire to write anywhere, so
there is no seam to pick. Verified `watch.py:36`'s `release_program` → `core.py:196-217`
`invalidate()` sets `self.program = None` with no recompile, and `core.py:225` carries the "Lazy
compile (066 D1)" comment the spec cites.

**(c2) `"(none)"` indistinguishable from never-wired.** **Closed.**

> "`PassGraph.with_input(…, "")` deletes the key, so an explicit "(none)" is byte-identical to
> never-wired and any rule keyed on absence would re-wire against the user's choice. So: (1)
> `with_input(name, "")` STORES the empty string — absent key = never decided, `""` = explicitly
> none"

This is round 2's option (a) ("a sentinel `""` value that `with_input` would have to stop deleting")
taken verbatim. Verified the code it changes: `pass_graph.py:163-166`,
`if producer: inputs[uniform] = producer` / `else: inputs.pop(uniform, None)`.

**(c3) the copilot `edit_shader` asymmetry.** **Closed.**

> "the copilot's `edit_shader` path gets the same behaviour with no code, since resolution happens
> at render"

Correct by construction: `_copilot_persist_shader` writes no graph, and under a resolution rule it
does not need to. The asymmetry round 2 measured was a property of the stored-edge design and dies
with it.

**(c4) the third state — an edge naming a pass that no longer exists.** **Closed differently.** The
spec does not name the dangling-edge case, but the rule's phrasing settles it:

> "(2) the effective input of a sampler with an ABSENT key is `u_<x>` → pass `<x>` when such a pass
> exists (`u_prev` → the pass itself), else black"

The rule fires on an **absent** key only. A dangling `cascade.u_df -> "df"` is a *present* key, so
`effective_inputs` returns it unchanged and `Document.render`'s existing "reads BLACK" fallback
(`document.py:434-439`) handles it exactly as today. Round 2's "not stated" becomes "stated by the
rule's domain". Worth one clarifying clause at impl, but not a gap.

**(c5) round 2 Task 3's texture-capture hazard (folded into this bullet).** **Closed.**

> "(3) a sampler whose `uniform_values` entry is a user-bound texture (`MediaWithTexture`) is never
> auto-wired — `core.py:373`'s `inputs.get(name, uniform_values.get(name))` would otherwise let a
> pass named `image` silently replace the PNG in the `Media Input` example (`73ea2431…`)"

Verified `core.py:373` is exactly `value = inputs.get(uniform.name, self.uniform_values.get(uniform.name))`
and `core.py:382` is the `isinstance(value, MediaWithTexture)` branch. The guard is one condition at
the one seam, as round 2 asked.

**(c6) the duplicated `_sampler_names` helper (round 2's drift smell).** **Still open.** Round 2
noted `popups/pass_settings.py:114-119` and `copilot/backend.py:268` are two copies today and a
third would be a drift smell. The spec's `effective_inputs(entry, samplers, passes)` takes `samplers`
as a *parameter*, so it adds no third copy — but it also does not say the two existing copies
converge. **Preference, not a defect**: the rule as written adds nothing; consolidating the existing
pair is out of this bullet's scope.

**Verdict for 1(c): 5 of 6 closed (3 closed differently), 1 open and a preference. The FAIL is
lifted.**

---

### Round 2 Task 2 — the W-B gate (round 2 verdict: FAIL)

Round 2 demanded **three additions**.

**(2a) treat `ast.JoinedStr` as measurable, scoring each `FormattedValue` as one word.** **Closed.**

> "It scores `Constant` strings AND `JoinedStr` (each `FormattedValue` counts one word —
> `pass_settings.py:128`'s "(?)" tooltip is an implicit concatenation with an f-string and parses as
> `JoinedStr`)"

Names the mechanism, the scoring rule, and the site round 2 measured.

**(2b) add `label_row` / `row_label`, with "no `FormattedValue` in a label" as the budget.**
**Closed.**

> "asserts a `label_row` / `row_label` label carries no `FormattedValue` at all (a label is fixed
> text; `pass_settings.py:178`'s `f"size ({w}, {h})"` is the overflow #7 reported and the gate must
> refuse it)"

Both functions named, the budget stated as round 2 specified, and the target site named.

**(2c) decide what happens to a `Name` argument.** **Closed** — with the *third* option, which round
2 did not offer but which subsumes both:

> "A `Name` argument is resolved to its assignment in the enclosing function when that is a string;
> otherwise the site is listed in a pinned allowlist in the test (the `ui_primitives` shared helpers
> that forward a caller's text — their callers are the measured sites), so an unmeasured site is a
> deliberate entry, not a blind spot."

Round 2's own framing ("a gate that silently skips a third of its domain is the *checker that
quietly narrows its own domain* shape") is answered by the pinned allowlist: the skip is enumerated,
so it cannot grow silently. Round 2 also asked the test to "assert the count of sites it measured" —
that half is **still open**; the allowlist pins the *unmeasured* set but the spec does not pin the
measured count. See Task 3 for why that matters less than round 2 thought, and Finding R3-1 for why
it matters more.

**Verdict for Task 2: 3 of 3 closed, 1 sub-clause (measured-count assertion) open. The FAIL is
lifted — see Task 3 for the demonstration and for one census error the rewrite inherited.**

---

### Round 2 Task 3 — D9 wiring vs the shipped examples

Round 2's one unclosed item was the **texture-capture guard**. **Closed** — quoted at (c5) above.

Round 2's secondary note ("the producer-naming trailing comments in `blur.frag.glsl:9`,
`trail.frag.glsl:10-11`, `composite.frag.glsl:8-10` become redundant under D9; worth one line in
W-D") is **still open** — the rewritten W-D bullet does not mention the comments. **Preference, not a
defect**: the comments stay correct after the rename; only their usefulness changes.

---

### Round 2 Task 4 — F1–F5

**F1 — the combo's literal index `0`.** **Closed.**

> "today's combo is fed a literal index `0` every frame (`tabs/document.py:109` — it has no
> selection state, it is a menu) … It becomes one control with state: a `W × H` pair of `input_int`s
> showing `document.canvas_size`"

The control gains state by holding `document.canvas_size` rather than by making the combo's index
stateful — which is the cleaner of the two answers round 2 offered ("the combo's index becomes
state, or the free entry replaces the combo's current-size slot"). Verified `tabs/document.py:109`
is `imgui.combo("##resolution", 0, resolution_items)` and `:110` is `if new_res_idx != 0`.

**F2 — the list is built from `render_pass`, not the document.** **Closed.**

> "and its list is built from `render_pass` (the last output-keyed row in a panel that otherwise
> uses `panel_pass` / `document`) … a `W × H` pair of `input_int`s showing `document.canvas_size`"

Verified `tabs/document.py:62` (`cw, ch = ui_document.document.render_pass.canvas.texture.size`) and
`:68` (`for uniform in ui_document.document.render_pass.get_active_uniforms()`). The rewrite moves
the *displayed size* to `document.canvas_size`; the presets menu still says "any bound texture's
size", which is still a `render_pass`-sourced list. That is acceptable — bound textures live in
`uniform_values`, which is per-pass by design (`core.py` docstring) — but note the spec does not say
*which* pass's textures feed the presets. **Preference**: one clause at impl ("the panel pass's
bound textures").

**F3 — the fourth `.render_pass` script seam.** **Closed.** Quoted at (b4) — "the live tick
(`project_session.py:598`) is the fourth `.render_pass` seam and the one whose protocol type
changes."

**F4 — the orphan error needs the pass, not just a pass-qualified `last_driven`.** **Closed.**

> "The engine's soft error for an orphan records the PASS it named (a new field on the error, not
> only the key), so a shader tab can show the errors that name its pass."

"a new field on the error, not only the key" is exactly the distinction round 2 drew.

**F5 — the first-launch salvage warning nobody expects.** **Closed**, in two places:

> "First launch after W-G logs one salvage line per stale `projects/dev` document — expected, and
> the verification list says so."

and in `## Manual verification`:

> "The first launch logs one salvage line per `projects/dev` document whose `stopped_uniforms`
> predates the pair shape — expected once, gone after the hand-edit."

**Verdict for Task 4: 5 of 5 closed.**

---

### Task 1 verdict: **PASS**

| Round 2 item | Closure |
|---|---|
| 1(a) site 2 early-out | closed |
| 1(a) site 4 cycle fallback | closed |
| 1(a) target-alone vs ancestor chain | closed |
| 1(a) `Document.first_render_done` interaction | closed differently (narrower guard) |
| 1(a) `begin_frame` / iterated double-swap | closed |
| 1(a) who writes `Pass.first_render_done` | closed |
| 1(a) export not threaded | closed |
| 1(a) `_graph_errors` side effect | still open — **preference** |
| 1(b) App wrappers | closed |
| 1(b) `tabs/document.py:253` | closed |
| 1(b) `set_uniform` gate + corrected motivation | closed |
| 1(b) `tick` / `stopped` type / `last_driven` | closed |
| 1(b) example count | closed |
| 1(b) persistence shape | closed |
| 1(c) hot-reload seam | closed differently (design replaced) |
| 1(c) `"(none)"` vs never-wired | closed |
| 1(c) `edit_shader` asymmetry | closed |
| 1(c) dangling edge | closed differently (out of the rule's domain) |
| 1(c) texture capture | closed |
| 1(c) duplicated `_sampler_names` | still open — **preference** |
| 2 JoinedStr | closed |
| 2 `label_row` / `row_label` | closed |
| 2 `Name` handling | closed (allowlist); measured-count assertion still open |
| 3 producer comments | still open — **preference** |
| 4 F1–F5 | all closed |

24 items: **20 closed, 3 closed differently, 3 open of which 3 are preferences, 1 sub-clause open**
(the measured-count assertion, which Task 3 shows matters).

---

## Task 2 — implementability of the two redesigns

### (a) W-D — the resolution rule

**Verdict: PARTIAL.** Implementable, and the four questions asked have concrete answers — but one of
them (the lazy-compile interaction) changes the draw order between frame 1 and frame 2 in a way the
spec does not state, and the gear's combo needs a third item the spec's own wording does not supply.

**Where `effective_inputs(entry, samplers, passes)` gets `samplers` for an uncompiled pass.**

`samplers` is a parameter, so the answer is "from whoever calls it", and the callers the spec names
differ:

- **`Document.render`** (`document.py:428`, today `for uniform, source_name in entry.inputs.items()`)
  — by the time the loop runs, `Pass.render` has been reached and `core.py:355-356`'s lazy compile
  has run, so `get_active_uniforms()` (`core.py:225-231`) returns the real sampler list. Except that
  `render` must resolve inputs *before* calling `render_pass.render`, so it must call
  `get_active_uniforms()` itself — which forces the compile one statement earlier in the same call.
  Harmless; the compile happens on the same frame either way.
- **The planner** (`plan_for_output` / `plan_passes`, `pass_graph.py:232-307`) — `pass_graph.py` is
  **GL-free pure data by its own module docstring** ("everything here is GL-free pure data … the
  cycle check … unit-testable with no context"). It holds `PassGraph`, not `dict[str, Pass]`, so it
  cannot call `get_active_uniforms()` at all. The samplers must be handed in from `Document.render`,
  which means the planner signature grows a `samplers_by_pass: dict[str, list[str]]` argument (or
  `plan_passes` gains one). The spec says the planner "must see auto edges to order the draw and to
  detect cycles" but does not say how the GL-free module gets a GL-derived fact. **This is the one
  structural decision the bullet leaves to the implementer**, and it is not free: `plan_for_output`
  has other callers (`evaluation_order`, tests) that have no samplers.

**Does the planner see no auto edges until the first render, and is that acceptable?**

Yes, it sees none, and **the draw order does change between frame 1 and frame 2** for a document
whose auto edges are load-bearing.

Trace: at load nothing is compiled (066 D1). On frame 1, `Document.render` resolves passes in
`order`. To build `order` it calls `plan_for_output` **before** any pass has rendered — so at that
moment `get_active_uniforms()` has not been called on any pass unless the resolution code calls it.
Two outcomes:

1. **If `Document.render` compiles every pass up front to gather samplers** (calling
   `get_active_uniforms()` on all of `self.passes` before planning): the planner sees the auto edges
   on frame 1, the order is right from the first frame — and 066 D1's laziness is spent, because a
   document with six passes compiles all six on its first render rather than only the output chain.
   That is what 066 D2's one-document-per-frame budget already assumes ("a first render pays the
   document's pass compiles"), so it is defensible — but it is a change to *which* passes compile:
   today only the output chain does, and a pass on a dead branch never compiles at all.
2. **If it gathers samplers only from passes already compiled**: on frame 1 the planner sees only
   explicit edges, so a pass whose *only* inbound edge is an auto edge is not in `order` and does
   not draw. On frame 2 it is compiled (or still is not — it never drew, so nothing compiled it),
   which is a fixpoint: **a pass reachable only through an auto edge never enters the order, because
   the fact that would put it there is produced by the compile that only entering the order would
   trigger.** That is not "order changes between frames"; it is an auto edge that never resolves.

So option 2 is not viable and option 1 is the only implementable reading. **The spec must say
`Document.render` gathers samplers for every pass before planning, and that this makes the first
render of a document compile every pass rather than the output chain.** The consequence is real and
measurable (the RC example's six passes), and it interacts with W-C's first-render sweep, which
exists precisely to spread those compiles over frames — under W-D they all land on frame 1 anyway.
**This is the one finding in Task 2 I would call a defect rather than a preference.**

(Note the interaction runs the other way too, and favours the spec: W-C's sweep draws every pass, so
under option 1 the compiles W-C was spreading are already paid. The two bullets should be read
together at impl.)

**Does storing `""` in `graph.json` survive the per-key salvage on load?**

**Yes**, verified by running it:

```
PassGraph(output='a', passes={'a': PassEntry(inputs={'u_b':''}), 'b': PassEntry()})
  → g.passes['a'].inputs == {'u_b': ''}
  → g.model_dump()['passes']['a'] == {'inputs': {'u_b': ''}, 'target': {...}, 'iterations': 1}
```

`PassEntry.inputs` is `dict[str, str]` (`pass_graph.py:83`) with no constraint on the value, so `""`
validates. `load_graph`'s per-entry salvage (`document.py:101-120`) runs `drop_unknown` +
`drop_invalid` + `model(**row)` on each entry; a `{"u_b": ""}` inputs dict passes all three. The
`_reject_unnamed_pass` validator (`pass_graph.py:126-131`) rejects a *pass named* `""`, never an
input *value* of `""`. **No salvage change needed** — the spec's silence here is correct.

**One consequence the spec does not state**: `plan_passes` classifies a `""` value as an *unresolved
input*. Ran it:

```
plan.unresolved_inputs == {'a': {'u_b': ''}}
```

because `pass_graph.py:262` tests `if source not in known` and `""` is never a known pass name. That
is the *right* ordering behaviour (an explicit none contributes no edge), but it lands the sampler in
a field whose docstring reads "names the inputs pointing at a pass that does not exist, which read
black" (`pass_graph.py:201-203`). Today `unresolved_inputs` has **no production consumer** — grep
finds it only in `pass_graph.py` and `tests/test_pass_graph.py:154,177` — so nothing surfaces it to
the user and nothing breaks. **Preference, not a defect**: one clause in W-D saying an explicit `""`
reads as unresolved-and-that-is-fine, or a filter in `plan_passes`, either works.

**Do `_graph_renamed` / `_graph_without` need to treat `""` specially?**

**No.** Ran both:

```
g = passes={'a': PassEntry(inputs={'u_b':'', 'u_c':'b'}), 'b': PassEntry()}
_graph_renamed(g,'b','z').passes['a'].inputs  → {'u_b': '', 'u_c': 'z'}
_graph_without(g,'b',{'a':…}).passes['a'].inputs → {'u_b': ''}
```

`_graph_renamed` (`project_session.py:141-155`) maps `new if src == old else src`; `""` never equals
a pass name because `PassGraph` forbids an empty pass name. `_graph_without`
(`project_session.py:126-139`) filters `src != removed`; same argument. Both preserve `""` untouched,
which is the wanted semantics (an explicit un-wire survives a rename or a delete elsewhere). **The
spec is right to say nothing.**

**What the gear's combo shows for the three states.**

Today (`popups/pass_settings.py:141-146`):

```python
choices = [_UNWIRED, *sorted(document.passes)]      # _UNWIRED = "(none)"
current = entry.inputs.get(uniform, "")
index = choices.index(current) if current in choices else 0
```

- **absent key** → `current == ""` (the `.get` default), `"" not in choices`, → index 0 → shows
  `(none)`.
- **stored `""`** → `current == ""` → **the same index 0** → shows `(none)`.
- **explicit `"df"`** → index of `df` → shows `df`.

So **the two states the redesign exists to distinguish render identically under the current widget**,
and the spec's own requirement —

> "(4) the gear shows an absent key as "auto: x" (or "auto: none") with the explicit choices beside
> it"

— is not reachable by changing the sentinel alone. The combo needs a **third item**: the list becomes
`[f"auto: {resolved}", _UNWIRED, *passes]`, `current == ""` maps to the `(none)` slot, an absent key
maps to the `auto:` slot, and the write-back maps slot 0 → "delete the key", slot 1 → `""`, slot n →
the pass name. `with_input`'s two-branch signature (`producer` truthy/falsy) cannot express three
outcomes, so **`with_input` needs a third state too** — either a separate `without_input(consumer,
uniform)` or a sentinel argument. The spec says `with_input(name, "")` STORES `""` but does not say
what *deletes* a key any more. **Defect-sized gap, but a small one**: one added method and a
three-item combo, both mechanical. Worth a sentence because "explicit choices beside it" reads as
though the existing two-item list still works.

Verified `_UNWIRED = "(none)"` at `popups/pass_settings.py:31` and the write-back
`producer = "" if picked == 0 else choices[picked]` at `:147`.

### (b) W-C — `render(target)` against the code

**Verdict: PARTIAL.** All four decisions map to concrete lines. `Pass.drawn_frame` against "the
document's frame counter" is **not well-defined for one live path**.

Read `document.py:386-455` in full again. The mapping:

| Decision | Concrete line(s) | Maps cleanly? |
|---|---|---|
| (1) `target` feeds the early-out guard, `plan_for_output`, and the cycle fallback | `:392` (`if output is None or output not in self.passes`), `:395` (`plan_for_output(self.graph, output)`), `:397-400` (`if not order: order = [output]`) | **Yes** — three sites, exactly as the spec enumerates. Each reads the local `output`, so the implementation is one rebinding: `resolved = target or self.graph.output_pass`, then three substitutions. |
| (2) whole ancestor chain, skipping passes already drawn this frame via `Pass.drawn_frame` | the `for name in order:` loop head at `:401`; `_order_for` (`pass_graph.py:379-392`) already walks `plan.reads` transitively, so the chain comes for free | **Yes** for the chain; the skip is a new `if` at the top of the loop body. See below for `drawn_frame`. |
| (3) a target sizes by its own scale and never receives the external canvas; the two `name == output` comparisons use the graph output | `:409` (`if name != output:` — the full-size exemption) and `:426` (`target = canvas if (name == output and last) else None`) | **Yes**, and the "unchanged" is the load-bearing half: both must keep reading `self.graph.output_pass`, not the new `target` parameter. The spec says so. **Naming collision worth flagging**: `:426`'s local is already called `target`, so a parameter of that name shadows it inside the iteration loop. Mechanical, but it is exactly the kind of rename that goes wrong silently. |
| (4) `Pass.first_render_done` set by `render` on every pass it draws; `Document.first_render_done` = `canvas is None and target is None` | `:389-390` gains the conjunct; a new assignment inside the `for name in order:` loop | **Yes.** `core.py:150-168` (`Pass.__init__`) has neither field, so both `first_render_done` and `drawn_frame` are new — the spec's Files list names `core.py` for `Pass.first_render_done` but **not for `Pass.drawn_frame`**. One-word omission. |

**`Pass.drawn_frame` against "the document's frame counter" — the definition has a hole.**

The counter is `Document._frame` (`document.py:239`, `self._frame: int = -1`), advanced only by
`begin_frame` (`:295-313`). `begin_frame` has exactly one production caller: `ui.py:246`,

```python
for document_id in tick_documents:
    app.ui_documents[document_id].document.begin_frame(app.frame_idx)
```

— over `tick_documents`, which is built from `app.ui_documents` (`ui.py:214-243`). **The examples
popup path renders documents that are never in that set.** `ui.py:298-309`:

```python
elif app.popup_state == PopupState.EXAMPLES:
    pending_example = next((ui_document for ui_document in app.ui_document_examples.values()
                            if not ui_document.document.first_render_done), None)
    for ui_document in app.ui_document_examples.values():
        if ui_document.document.first_render_done or ui_document is pending_example:
            ui_document.document.render()
```

`app.ui_document_examples` is a **separate dict** (`project_session.py:200`, surfaced at
`app.py:967-968`); nothing calls `begin_frame` on its documents. Verified by grep: the only
`begin_frame` call sites in `shaderbox/` are `ui.py:246` and the two export loops
(`document.py:527`, `:619`). So for every example document, `self._frame` stays **`-1` forever**.

Consequence for the spec's rule: if `render` writes `render_pass.drawn_frame = self._frame` and skips
when `drawn_frame == self._frame`, then on an example document **every pass is skipped from its
second `render()` call onward, permanently** — `-1 == -1` on frame 2, frame 3, and every frame the
popup is open. The examples grid would freeze after one frame. (Today the same documents render every
frame with no skip, which is why nobody has noticed the frozen `_frame`.)

Three ways out, none stated:
- initialise `Pass.drawn_frame = -2` (or `None`) and make the skip test `drawn_frame == self._frame
  and self._frame >= 0` — smallest change, but it encodes "-1 means no counter" at a second site;
- have `render` advance nothing but compare against a counter it increments itself when `_frame` is
  untouched;
- call `begin_frame()` (the no-argument form, which `document.py:311-312` treats as "the next
  frame") on example documents in `ui.py:308` — which is arguably the real bug: an example document
  with a feedback pass currently never swaps its history either.

**The third is a behaviour fix beyond this bullet's scope**; the first is the one-line answer. Either
way the spec's phrase "the document's frame counter" needs the qualifier that a document rendered
without `begin_frame` has none. **Defect, small.**

Also worth stating explicitly for the implementer: **`drawn_frame` must be written for the passes in
`order`, i.e. inside the same loop, and the skip must be tested before the size-fixup at `:409-411`**
— a skipped pass that still gets `canvas.set_size()` applied is harmless, but a skipped pass that
runs the iteration loop is not, and the two live in the same loop body.

**One thing the bullet gets right that is easy to get wrong**, so it deserves a note rather than a
finding: decision (2)'s skip must **not** apply to the output-chain render itself, or a
`render(target=x)` earlier in the frame would suppress the output's own draw and the preview would
go stale. Re-reading the bullet — "skips any pass already drawn this frame … so shared ancestors draw
once per frame" — the skip *is* meant to apply to the output render too, and that is correct: the
preview render at `ui.py:265` (`render(canvas=app.preview_canvas)`) and the own-canvas render at
`ui.py:301` are both output renders in the same frame, and the second **must not** be skipped,
because it draws into a different canvas. So the skip cannot be unconditional on `drawn_frame`; it
must exempt the pass that is receiving an external `canvas`. **That is a fifth decision the bullet
does not state, and it is a real hazard**: with the skip as literally written, `ui.py:301`'s
`document.render()` would find every pass already `drawn_frame == frame` from the preview render at
`:265` and draw nothing — the pass thumbnails in the strip would go permanently black.

Concretely: `ui.py:265` renders the whole output chain into `preview_canvas`; every pass in the chain
gets `drawn_frame = frame`. Then `ui.py:301` renders the same document with `canvas=None`, which is
the render that fills each pass's **own** canvas — the texture `_draw_pass_tile` blits
(`widgets/pass_list.py:112-113`). Under the skip, that second render is a no-op. **This is the most
serious finding in this review.** The fix is one condition (skip only when `canvas is None and target
is not None`, i.e. only the first-render sweep participates in the skip), but the spec as written
says "skips any pass already drawn this frame" without qualification.

### Task 2 verdict: **PARTIAL**

1. **(a) defect** — the planner is GL-free and cannot obtain samplers; the spec does not say
   `Document.render` gathers them for every pass before planning, nor that doing so makes a
   document's first render compile every pass rather than only the output chain.
2. **(a) defect, small** — the gear needs a three-item combo and `with_input` needs a third outcome
   ("delete the key"); with the current two-item list an absent key and a stored `""` both render as
   `(none)`, defeating the redesign's whole point.
3. **(a) preference** — an explicit `""` lands in `plan.unresolved_inputs`, whose docstring says
   "pointing at a pass that does not exist". No production consumer; cosmetic.
4. **(b) defect, serious** — "skips any pass already drawn this frame" as written makes
   `ui.py:301`'s own-canvas render a no-op after `ui.py:265`'s preview render, blanking every pass
   thumbnail. The skip must be scoped to the first-render sweep.
5. **(b) defect, small** — "the document's frame counter" is `-1` forever on example documents
   (`ui.py:298-309` renders them; nothing calls `begin_frame` on `ui_document_examples`), so a naive
   `drawn_frame == self._frame` skip freezes the examples grid after one frame.
6. **(b) preference** — `Pass.drawn_frame` is not in the Files list beside `Pass.first_render_done`;
   and `document.py:426`'s existing local `target` shadows the new parameter name.

Confirmed clean: all four numbered decisions map to concrete lines; `_order_for` already gives the
ancestor chain for free; `render_media` / `_render_image` / `_render_media_into` need no edit (round
2's false trail, re-confirmed only to the extent of the spec's new "not threaded" sentence).

---

## Task 3 — the W-B gate as rewritten, demonstrated

**Verdict: PASS on the four target strings; PARTIAL on the bullet, for a census error it inherited.**

### By AST shape, before running anything

| Ledger item | Site | AST shape of the measured argument | Rewritten gate catches it? |
|---|---|---|---|
| #5, the "(?)" Reads tooltip | `popups/pass_settings.py:128` | `imgui.set_tooltip(...)` with three implicitly-concatenated parts, the middle an f-string → one `ast.JoinedStr` | **Yes** — by the new "scores `Constant` strings AND `JoinedStr`" clause |
| #5, the empty-Reads line | `popups/pass_settings.py:135` | `imgui.text_colored(COLOR.FG_DIM, "nothing — declare a sampler2D uniform to read another pass")` → `Attribute` + `Constant` | **Yes** — already caught by round 2's gate |
| #7, the size label | `popups/pass_settings.py:178` | `label_row(app.font_12, f"size ({w}, {h})", _CTRL_W, _ROW_LABEL_W)` → `label_row`, arg 1 a `JoinedStr` with two `FormattedValue`s | **Yes** — by the new `label_row`/`row_label` clause *and* its no-`FormattedValue` rule |
| #10, the gear tooltip | `widgets/pass_list.py:98` | `imgui.set_tooltip("Pass settings — what it reads, what it draws into")` → `Constant` | **Yes** — already caught |

(Note the spec cites the empty state as `pass_settings.py:136`; the `imgui.text_colored(` call opens
at **:134** and its string literal is on **:135**. `ast` reports the *call*'s lineno, so a gate
report will say `:134`. Cosmetic citation drift, no effect on the gate.)

### The script, and its census

Wrote a 60-line `ast` walk implementing the gate exactly as the bullet specifies — the four named
functions plus `label_row`/`row_label`, `text_colored` gated on a `COLOR.FG_DIM` first arg,
`Constant` and `JoinedStr` both measured with each `FormattedValue` counting one word, a `Name`
resolved to a string assignment in scope, everything else routed to the allowlist, and
"no `FormattedValue` in a `label_row`/`row_label` label" asserted. Budgets from D1: label 1-2 words,
`help_marker` ≤ 8, empty state ≤ 4; `set_tooltip` scored at 8 and `separator_text` at 4 as the
nearest D1 readings (D1 gives no explicit number for either, which is itself worth a clause in W-B).

Ran against `shaderbox/`:

```
measured=62  allowlist=26  flagged=11

FLAG text_colored+FG_DIM shaderbox/exporters/telegram.py:761      9 words, fv=True,  limit=4
FLAG set_tooltip         shaderbox/popups/help.py:84             15 words, fv=False, limit=8
FLAG set_tooltip         shaderbox/popups/lib_picker/__init__.py:140  11 words, fv=False, limit=8
FLAG help_marker         shaderbox/popups/pass_settings.py:108   25 words, fv=False, limit=8
FLAG label_row           shaderbox/popups/pass_settings.py:178    6 words, fv=True,  limit=2   ← #7
FLAG help_marker         shaderbox/popups/pass_settings.py:205   23 words, fv=False, limit=8
FLAG help_marker         shaderbox/popups/pass_settings.py:215   29 words, fv=False, limit=8
FLAG help_marker         shaderbox/popups/pass_settings.py:244   53 words, fv=False, limit=8
FLAG set_tooltip         shaderbox/popups/pass_settings.py:128   31 words, fv=True,  limit=8   ← #5 tooltip
FLAG text_colored+FG_DIM shaderbox/popups/pass_settings.py:134   10 words, fv=False, limit=4   ← #5 empty state
FLAG set_tooltip         shaderbox/widgets/pass_list.py:98       10 words, fv=False, limit=8   ← #10
```

**Census: 62 sites measured, 26 in the allowlist, 11 flagged. All four ledger strings are flagged.**
Round 2's FAIL ("catches two of the four") is demonstrably lifted.

The 26 allowlist entries are the ones the bullet predicts: six `ui_primitives.py` `set_tooltip`
forwards (`:211`, `:447`, `:842`, `:1161`, `:1252`) plus `row_label:1096` (the helper forwarding its
own parameter), the `_FORMATS[...]` `Subscript` at `pass_settings.py:172`, the `IfExp` at `:191`, the
emoji-picker `Attribute`s, and a dozen `text_colored` `Name`/`BinOp`/`Call` arguments. Every one is
a variable or an expression at the call site, so the bullet's "their callers are the measured sites"
reading holds for the `ui_primitives` cluster and the allowlist is the honest home for the rest.

### The one problem the rewrite inherited: the census in the bullet is wrong

The bullet states:

> "Census at writing: `help_marker` 7 sites, `set_tooltip` 19, `separator_text` 10,
> `text_colored`+`FG_DIM` 11."

The first three check out (`grep` gives 7 / 19 / 10). **The fourth is wrong by roughly 3x.** An `ast`
walk over `shaderbox/` finds **63 `text_colored` calls, 31 of them with a `COLOR.FG_DIM` first
argument** — not 11. (`grep -rn "text_colored(\s*COLOR.FG_DIM"` gives 27, missing the four where the
call opens on its own line; the AST count of 31 is the authoritative one.) The number `11` came from
round 2's table, where it was a grep artifact, and the rewrite copied it.

This matters because of the sub-clause Task 1 left open: round 2 asked that "the test should assert
the count of sites it measured so the domain cannot shrink unnoticed", and the rewrite did not fold
it. A written census of 11 against a real 31 is precisely the *checker that quietly narrows its own
domain* shape — if the test is ever seeded from the spec's number, two-thirds of the empty-state
idiom drops out of the gate silently. **Defect**: fix the number to 31 and add the measured-count
assertion (my run says 62 measured / 26 allowlisted, which is the pair the test should pin).

### Task 3 verdict: **PARTIAL**

1. **PASS** — all four ledger strings are flagged by the gate as rewritten; demonstrated by a run,
   not argued.
2. **Defect** — the bullet's `text_colored`+`FG_DIM` census says 11; the real count is 31 by AST (27
   by naive grep). Copied from round 2's table.
3. **Defect, small** — the measured-count assertion round 2 asked for is not in the bullet; without
   it the allowlist pins the unmeasured set but nothing pins the measured set.
4. **Preference** — D1 gives no word budget for `set_tooltip` or `separator_text`; I scored them at
   8 and 4 to run the gate. W-B should state both numbers, or the gate's authors will pick their own.
5. **Preference** — the bullet cites the empty state as `:136`; `ast` reports the call at `:134`.

---

## False trails — probed, fine, do not re-check

- **`""` breaking `graph.json` load.** Constructed and dumped a graph with `inputs={'u_b': ''}`:
  validates, round-trips, and survives `load_graph`'s per-entry `drop_unknown` + `drop_invalid` +
  `model(**row)` (`document.py:101-120`). `PassEntry.inputs` is `dict[str, str]` with no value
  constraint (`pass_graph.py:83`), and `_reject_unnamed_pass` (`:126-131`) guards a pass *name*, not
  an input value. No salvage change needed, as the spec assumes.
- **`_graph_renamed` / `_graph_without` needing a `""` special case.** Ran both against a graph
  carrying `{'u_b': '', 'u_c': 'b'}`; both preserve `''` untouched and rewrite only the real name.
  The reason is structural (no pass may be named `""`), so it cannot regress by accident.
- **`unresolved_inputs` surfacing an explicit `""` to the user.** Grepped every reference: only
  `pass_graph.py:201/208/307` and `tests/test_pass_graph.py:154,177`. No production consumer, no UI
  path. Cosmetic only.
- **`plan_for_output` overwriting `_graph_errors` with a partial plan's errors under `target=`.**
  Re-confirmed round 2's own conclusion from `pass_graph.py:355-362`: `plan_for_output` calls
  `plan_passes(graph)` on the WHOLE graph and only then narrows with `_order_for`, so the error list
  is target-independent by construction. Harmless, and structurally so.
- **`with_input` being unable to store `""` for a model reason.** It is a plain `if producer:`
  branch (`pass_graph.py:163-166`); nothing in the model resists the change. The gap I raise in Task
  2(a) is about the *third* outcome (deleting a key), not about storing `""`.
- **`begin_frame` double-advance under W-C.** Round 2's false trail, re-confirmed for the one thing
  I added: `document.py:307-309` is idempotent per frame number, and `ui.py:246` is the only
  per-frame caller. My Task 2(b) finding is the opposite shape — a document that never gets
  `begin_frame` at all — not a double-advance.
- **`_sampler_names` gaining a third copy under W-D.** `effective_inputs(entry, samplers, passes)`
  takes samplers as a parameter, so the rule adds no copy. The two existing copies
  (`popups/pass_settings.py:114-119`, `copilot/backend.py:268`) are unchanged by this bullet.

---

## Verdicts

| Task | Verdict | Count |
|---|---|---|
| 1 — closure of round 2's checklist | **PASS** | 24 items: 20 closed, 3 closed differently, 3 open (all preferences) + 1 open sub-clause (the measured-count assertion, which Task 3 shows should be folded) |
| 2 — implementability of W-D and W-C as redesigned | **PARTIAL** | 6 items: 4 defects (one serious — the unqualified skip blanks every pass thumbnail), 2 preferences |
| 3 — the W-B gate, demonstrated | **PARTIAL** | 5 items: the gate flags all four ledger strings (62 measured / 26 allowlisted / 11 flagged); 2 defects (the wrong `FG_DIM` census, the missing count assertion), 2 preferences, 1 citation drift |

**Recommendation.** The two redesigns are sound and round 2's two FAILs are lifted. Three sentences
should land before implementation, in descending order of cost:

1. **W-C**: scope the `drawn_frame` skip to the first-render sweep only. As written it applies to
   every render, and `ui.py:265`'s preview render would suppress `ui.py:301`'s own-canvas render —
   every pass thumbnail goes black. One condition (`canvas is None and target is not None`).
2. **W-D**: say that `Document.render` gathers samplers for every pass before planning (the planner
   is GL-free and cannot get them itself), and that this makes a document's first render compile
   every pass rather than only the output chain — which is what makes the auto edges visible to the
   planner at all.
3. **W-D**: the gear's combo becomes three-item and `with_input` gains a way to delete a key; with
   the current two-item list an absent key and a stored `""` are indistinguishable on screen, which
   is the exact failure the sentinel exists to fix.

Plus two small ones: `Pass.drawn_frame` needs the "no `begin_frame`" qualifier (the examples popup
renders documents whose `_frame` is `-1` forever), and W-B's `text_colored`+`FG_DIM` census should
read 31, not 11, with the measured-site count asserted in the test.
