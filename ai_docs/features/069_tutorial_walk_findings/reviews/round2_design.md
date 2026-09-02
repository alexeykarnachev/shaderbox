# 069 — pre-impl review, round 2: design & implementability

Reviewer role: design/implementability, second round, against the spec **as patched after round 1**
(`01_spec.md`). Read-only. Anchors: the code on `dev` at `1767483`, `00_findings.md`,
`01_design_scripting.md`, `CLAUDE.md`, `ai_docs/conventions.md`, `ai_docs/dev_flow.md`.

Round 1's two "False trails" sections are taken as verified and were not re-probed. Where round 1's
conclusion is *contradicted* by something I opened, it is flagged explicitly and the file is named.

Every claim below names the file I opened. Line numbers are today's; each is paired with a symbol
so the citation survives an edit.

**The spec was edited on disk mid-review.** I re-read W-B, W-C, W-D and W-G against the current
text before finalising. Three deltas landed: D3 gained the "no uniform value is ever a `dict`"
invariant; W-B's gate gained a scope disclaimer about runtime-built strings; W-D's strip-tune bullet
gained detail; W-G's tick list gained `reload` and its Files list gained `projects/dev`. Task 2's
verdict is written against the **new** W-B text (the disclaimer changes the reading, not the
outcome — see Task 2's opening). W-C and W-D's default-wiring bullet are unchanged.

---

## Task 1 — implementability of the three highest-stakes changes

### (a) W-C — `Document.render(target=...)`

**Verdict: PARTIAL.** The signature change is implementable, but the spec's one sentence describes a
substitution that does not correspond to the code's actual `output` reads, and it leaves four
decisions unstated.

The spec (W-C, "First render of every pass") says:

> `Document.render` gains `target: str | None = None` — it substitutes for `graph.output_pass` in
> `plan_for_output` AND in the two `name == output` comparisons (the full-size exemption and the
> external-canvas rule …)

**Every branch in `Document.render` (`shaderbox/document.py:386-455`) that reads `output`, and
whether the spec's sentence covers it:**

| # | Site | Code | Spec covers? |
|---|---|---|---|
| 1 | `document.py:391` | `output = self.graph.output_pass` — the binding itself | Implied ("substitutes for `graph.output_pass`"), but see D1 below |
| 2 | `document.py:392` | `if output is None or output not in self.passes: … return` — the early-out that records `plan_passes` errors and draws nothing | **NOT covered.** Under `target=`, this guard must test the *target*, not the output. A target that vanished mid-frame (renamed, deleted between the frame gate's pick and the render) would otherwise fall through to a plan for a pass that is gone. |
| 3 | `document.py:395` | `planned, self._graph_errors = plan_for_output(self.graph, output)` | Covered |
| 4 | `document.py:397-400` | `if not order: order = [output]` — the cycle/unreachable fallback | **NOT covered.** With `target=` this must be `[target]`, else a first-render pass sitting on a cycle silently draws the *output* instead. Silent: same signature, wrong pass drawn. |
| 5 | `document.py:409` | `if name != output:` — the full-size exemption (an intermediate is sized by `entry.target.target_size(canvas_size)`) | Covered by the spec's clause "a first-render target sizes by its own scale like an intermediate" |
| 6 | `document.py:426` | `target = canvas if (name == output and last) else None` — the external-canvas rule | Covered by the spec's clause "never receives an external canvas" |
| 7 | `document.py:392-395` | `self._graph_errors = …` — **the side effect**, not a branch | **NOT covered.** `_graph_errors` is document-scoped state that the UI reads. A `render(target=name)` for a first-render sweep would overwrite the document's graph errors with the errors of a *partial* plan. `plan_for_output` returns the whole graph's errors (`pass_graph.py:362`, `plan, errors = plan_passes(graph)`), so this happens to be harmless today — but nothing in the spec pins that, and it is exactly the kind of coupling that goes wrong when `_order_for` is touched. |

**What an implementer must still decide, that the spec does not say:**

1. **Whether `target` also substitutes at site 2 and site 4.** Both are `output` reads. The spec
   says "the two `name == output` comparisons", which are sites 5 and 6 only. Sites 2 and 4 are
   `output`-vs-nothing reads and are not mentioned. Site 4 is a correctness hazard (wrong pass
   drawn, silently).
2. **Whether `target=` renders only the target or the target *and all its ancestors*.**
   `plan_for_output` → `_order_for` (`pass_graph.py:379-392`) walks `plan.reads[name]` transitively,
   so `render(target="composite")` draws **the entire chain feeding composite**, not composite
   alone. For the six-pass RC example, a per-frame "one not-yet-drawn pass" sweep would draw
   1 + 2 + 3 + 4 + 5 + 6 = up to 21 pass-draws across six frames rather than 6 — and the spec's own
   test ("N frames later every pass has rendered once") passes either way, so the test does not
   distinguish them. The alternative (draw the target alone, ancestors black) produces a *wrong*
   first thumbnail, which is the finding #36 the wave is fixing. Neither is stated.
3. **Whether `canvas is None → self.first_render_done = True` (`document.py:389-390`) applies to a
   `target=` call.** Today the document flag is set on ATTEMPT of an own-canvas render, and its
   comment says explicitly it must not be set for "a probe/export into a foreign canvas … the pass
   canvases (what the grid tile shows) [stay] unwritten". A `render(target=intermediate)` is an
   own-canvas render that leaves the OUTPUT's canvas unwritten — the mirror case. If it sets
   `first_render_done`, `ui.py:228/235/304/309` (the 066 D2 frame budget) will admit the document
   into the "already rendered" set before its output ever drew, and `widgets/document_grid.py:110`
   / `popups/examples.py:103` will drop the stale wash on a tile that is still black. The spec
   says nothing.
4. **Whether `begin_frame` interacts.** `ui.py:246` calls `begin_frame(app.frame_idx)` once per
   frame per document, and `document.py:307` makes a repeat call for the same frame a no-op — so a
   second `render(target=…)` in the same frame is safe for feedback. But an *iterated* target with
   `entry.iterations > 1` calls `_swap_feedback` between iterations (`document.py:449`), and a
   first-render sweep therefore advances a self-reading pass's history N-1 times outside the frame
   swap. For `jfa` (11 runs after W-H) that is 10 extra swaps on its first frame. Not stated.

**How `Pass.first_render_done` gets set — the spec does not say, and the two candidates differ:**

`shaderbox/core.py:150-168` (`Pass.__init__`) has no such field; the spec's `## Files touched` names
`core.py` for it, so it is new. Nothing in the spec says who writes it. The two options are not
equivalent:

- **Inside `Document.render`** — mirrors `Document.first_render_done` (`document.py:389-393`, set
  inside `render`). It would be set for every pass in `order`, so a first-render sweep marks the
  whole ancestor chain at once (which is *correct*, and would collapse the 21 draws in point 2 to
  6). But `Document.render` has no notion of "this call is a first-render sweep", so it would also
  be set by the preview render (`ui.py:265`, `canvas=app.preview_canvas`) and by the copilot probe
  (`copilot/backend.py:1818/1826`, `canvas=self._probe_canvas`) — both of which leave the pass
  canvases unwritten for the OUTPUT pass, the exact hazard `Document.first_render_done`'s
  `canvas is None` guard exists to prevent. So it needs the same `canvas is None` guard, and the
  spec does not name one.
- **By `ui.py`'s frame gate** — the spec's own wording ("`ui.py`'s frame gate then draws … tracked
  by `Pass.first_render_done`") reads this way. Then a pass drawn as an *ancestor* of the output
  chain is never marked, and the sweep re-renders passes that have in fact been drawn every frame
  since load — the sweep would run for `len(passes)` frames instead of terminating early. Harmless
  but wasteful, and it makes the "steady state draws only the output chain" test the only thing
  standing between the sweep and a permanent per-frame extra render.

**Export path — `document.py:641-660` (`render_media`) is NOT affected, but for a reason the spec
does not state.** `render_media` enters `self.export_isolation()` then `self.reset_feedback()`, then
funnels into `_render_image` (`document.py:520-528`) / `_render_media_into` (`document.py:619-620`),
both of which call `self.render(u_time=…, canvas=canvas)` with **no** `target`. With
`target: str | None = None` defaulting to `None → graph.output_pass`, every export call is
byte-identical. The same holds for `app.py:1375` (`document.render()` after an in-app save) and the
copilot probe. **What the spec should say and does not:** that `target` is *not* threaded through
`render_media`, `_render_image` or `_render_media_into`, i.e. export never renders a non-output
target. Without that sentence an implementer plausibly adds `target` to `_render_image` "for
symmetry", and `reset_feedback()` at `document.py:650` then wipes feedback history for a
first-render sweep that has nothing to do with export.

---

### (b) W-G — pass-qualified stopped/driven state

**Verdict: PARTIAL.** The five session methods the spec names are the right five, but it misses
three consumers and one persistence fact, and the `model_salvage` question has a non-obvious answer.

**What I read:** `project_session.py` `_stopped_for` (`:604-614`), `is_uniform_stopped` (`:706-712`),
`set_uniform_stopped` (`:714-725`), `set_document_all_stopped` (`:727-736`),
`uniform_is_driven` (`:700-704`), `get_script_driven_uniforms` (`:738-741`), `tick` (`:580-602`);
`ui_models.py` `UIDocumentState.stopped_uniforms` (`:163`) and `load_document_from_dir`
(`:492-521`); `widgets/uniform.py` (`:150-171`, `:187-188`, `:296-303`);
`copilot/backend.py`'s four `_get_script_driven_uniforms` call sites (`:640`, `:723`, `:740`, `:881`).

**Consumers the spec does not name:**

1. **`App.set_uniform_stopped` / `App.set_document_all_stopped` (`app.py:1414-1435`).** These are
   the *only* callers `widgets/uniform.py` and `tabs/document.py:253` use — they forward to the
   session methods. A pass parameter on the session methods forces the same parameter through both
   App wrappers, plus `app.py:1435` (`self.set_document_all_stopped(document_id, playing)`, the
   play/stop-all path). `app.py` is in W-G's file list, but for `commands.py`/`RESET_FEEDBACK`
   reasons; these three methods are not mentioned.
2. **`tabs/document.py:253`** — `app.set_document_all_stopped(document_id, playing)`, the
   document-level play/stop button. `tabs/document.py` is **not** in W-G's file list at all (it is
   in W-A's and W-B's). Document-level stop is pass-*un*qualified by design, so this may be a
   no-change site — but that is a decision the spec should state, because the alternative reading
   ("everything goes pass-qualified end to end") makes it a change.
3. **`copilot/backend.py:881` — the `set_uniform` reject gate — cannot be made pass-qualified as
   written, because `set_uniform` has no pass.** Six lines later (`:889-892`) it resolves the
   uniform as `target.render_pass.get_active_uniforms()` — the **output pass only**. So the spec's
   stated motivation ("today's name-only test would reject `composite.u_x` because `paint.u_x` is
   driven") is half-right about the bug and wrong about the fix: the tool can only ever address the
   output pass, so the pass to test against is always `document.graph.output_pass`. Either
   `set_uniform` grows a pass argument (a copilot-tool schema change, nowhere in the spec, and the
   `copilot-llm-agent-design` skill's tool-count rule applies) or the gate hardcodes the output
   pass. Not stated. **The three other sites (`:640`, `:723`, `:740`) all feed `_format_uniforms`
   against `document.render_pass`** (`backend.py:243-265`), so they too are output-pass-only —
   except `_pass_views` (`:740`), which loops over **every** pass (`backend.py:735-750`,
   `for name in sorted(document.passes)`) while holding a single document-scoped `driven` set. That
   one is a genuine correctness bug today and the only one of the four that a pass-qualified set
   actually fixes. The spec treats all four as one item.
4. **`ProjectSession.tick` (`:598`) passes `ui_document.document.render_pass`** as the `EngineNode`,
   and `_stopped_for` returns a document-scoped `frozenset[str]` consumed at `engine.py:375`
   (`stopped: frozenset[str]`). The spec names the `EngineNode`→`Document`-shaped protocol change,
   but not that `stopped`'s **type** changes from `frozenset[str]` to a pass-qualified shape at the
   `engine.py:370` / `engine.py:397` (`tick_export`) / `engine.py:521` (`_tick_script`'s
   `last_driven`) boundary, nor that `DocumentScripts.last_driven: set[str]` (`engine.py:130`) and
   `last_skipped` (`:132`) — the fields `script_driven_uniforms` (`:250-255`) reads — become
   pass-qualified too. `ScriptStatus.driven_count` (`:272`) counts them.

**Persistence — the spec's count is wrong.** W-G says "the five shipped examples' `document.json`
files that persist `"stopped_uniforms": []` are hand-edited". There are **seven** example
directories under `shaderbox/resources/document_examples/`, and exactly **five** carry
`stopped_uniforms`. The two that do **not** are `1c4f8a20-…` (Bloom Chain) and `77a84d27-…`
(Radiance Cascades) — i.e. **the only two multi-pass examples, the only two where a pass-qualified
key means anything.** Their `ui_state` blocks hold only
`['description','render_media_details','ui_name','ui_uniforms']`. So the five hand-edits are all
no-op shape churn on single-pass documents, and the two documents that would exercise the new shape
have nothing to edit. The spec's Files list does name
`projects/dev` `document.json`s, and two live documents there (`ec926580-…`, `e7e00c46-…`) persist
`"stopped_uniforms": []` — so that half is covered. (Nine more copies sit under `projects/dev/trash/` and
`projects/dev/copilot/checkpoints/`; every value in the repo is `[]`, verified by
`grep -rn stopped_uniforms --include=*.json . | grep -v '\[\]'` → no hits.)

**Does `model_salvage` need a change for a list-of-pairs field? No — but not because it handles the
shape.** `UIDocumentState` does **not** go through `load_model` or `drop_unknown`. `ui_models.py`
`load_document_from_dir` (`:498-514`) hand-filters unknown keys against `UIDocumentState.model_fields`
and then calls **`drop_invalid` only** (`:514`). `drop_invalid` (`model_salvage.py:49-100`) has two
branches: a nested-`BaseModel` descent for `list[Model]`, and a final per-key
`validate_assignment`. `list[list[str]]` / `list[tuple[str, str]]` hits neither the nested branch
(`get_args(list[list[str]])` yields `list[str]`, which is not a `BaseModel` subclass, so the loop
`continue`s) nor any special case — it falls to `validate_assignment`, which pydantic evaluates
against the whole field. A file carrying the OLD `["u_x", "u_y"]` therefore fails validation *as a
whole* and the key is dropped to its default `[]`, with one `logger.warning`. **That is the correct
fail-soft outcome and needs no code change** — and it is also, in effect, a migration by omission,
which is what the no-migration rule wants. **What the spec must decide and does not:** whether
`stopped_uniforms` becomes `list[list[str]]` (JSON-native, salvage-transparent) or
`list[tuple[str, str]]` (pydantic coerces from a JSON list, same salvage behaviour) — and whether
the per-*element* salvage that `drop_invalid` gives a `list[Model]` is wanted here. A `list[PairModel]`
would get element-level salvage (one bad pair dropped, the rest kept); a bare `list[list[str]]`
gets all-or-nothing. The comment on `ui_models.py:161-162` pins the reason the field is a list at
all ("`model_dump()` -> `json.dump` … raises on a Python set") — a `set[tuple[...]]` is still
excluded, and the spec's "list of `[pass, name]` pairs" is consistent with that. State which.

---

### (c) W-D — default wiring by name

**Verdict: FAIL.** The `add_pass` half is implementable. The hot-reload half has **no seam that
exists**: the file the spec points at has neither the graph nor the session, and the pass's uniforms
are not knowable at the moment it fires.

**`add_pass` (`project_session.py:762-785`).** The seam is clean and unambiguous: after
`render_pass.compile()` (`:779`) and before `document.graph.with_passes(…)` (`:781-783`), the
sampler names are available via `render_pass.get_active_uniforms()` filtered on
`gl_type == GL_SAMPLER_2D` (the helper already exists as `popups/pass_settings.py::_sampler_names`,
`:114-119`, and again as `copilot/backend.py::_sampler_uniform_names`, `:268` — **two copies today;
a third here would be a drift smell under `conventions.md`'s "two parallel name-keyed dicts"
rule**). The new pass is created from `PASS_STUB`, so in practice it declares no samplers and the
rule is a no-op on `add_pass` — the maintainer's own verification step ("declare
`uniform sampler2D u_df;` in a new pass — it is wired") is a **hot-reload** scenario, not an
`add_pass` one.

**Hot reload — where would the rule live?** The spec says "`add_pass` / hot reload of a pass whose
sampler `u_<x>` names an existing pass `<x>` … pre-fills the edge". Reading
`shaderbox/watch.py::_reload_pass_if_changed` (`:19-61`):

- It takes `(app: App, name: str, render_pass: Pass)`. It has `app`, so `app.session` is reachable —
  but it does **not** know the `document_id`; `name` here is the *document display name* passed
  down from `reload_document_if_changed`'s caller (`ui.py:199`, which passes the loop's document
  key — so in fact it *is* the id, but the parameter is named `name` and used only in log lines,
  and `reload_document_if_changed` never touches `ui_document.document.graph`).
- More decisively: the reload path calls `render_pass.release_program(new_text)` (`:36`), which
  calls `invalidate()` (`core.py:196-217`) — it **releases the program and sets it to `None`**. It
  does not recompile. The next compile happens lazily inside `Pass.render()` (`core.py:355-356`) or
  `get_active_uniforms()` (`core.py:225-231`), one or more frames later, on the render thread.
  **So at the moment the hot reload fires, the pass's new sampler set is unknowable without forcing
  a compile there** — and forcing one in `watch.py` inverts the 066 D1 lazy-compile decision
  (`core.py:225`, "Lazy compile (066 D1): nothing compiles at load"), which the spec does not
  propose.

So the implementer faces an undecided choice the spec does not name: **(i)** force a compile inside
`watch.py` and wire there (violates 066 D1's laziness, and `watch.py` currently imports neither
`ProjectSession` nor `PassGraph`); **(ii)** put the rule in `Document.render`, after the per-pass
lazy compile, where the samplers are finally known — but `render` is on the headless core and the
graph is immutable (`PassEntry` is `frozen=True`, `pass_graph.py:81`) and `save_ui_document` lives
on the session, so a render-time wire either does not persist or reaches back into the session from
the core; or **(iii)** a per-frame reconcile in `ui.py` after the render, which is a fourth
mechanism. Each has a different answer to "does the auto-wire get saved to `graph.json`?".

**What "unwired" means against `PassEntry.inputs` — the spec does not say, and the two readings
differ.** `PassEntry.inputs: dict[str, str]` (`pass_graph.py:83`) is keyed by *this* pass's sampler
name. Reading the code:

- **Absent key** is the only representation of unwired in the model. `Document.render`
  (`document.py:428-439`) iterates `entry.inputs.items()`; a sampler with no key is simply not in
  the loop and falls through to `Pass.render`'s `uniform_values` (`core.py:373`,
  `inputs.get(name, self.uniform_values.get(name))`).
- **`"(none)"` is a UI string only, and picking it is INDISTINGUISHABLE from never-wired — which
  breaks the spec's own "explicit choice always wins" clause.** `popups/pass_settings.py:31`,
  `_UNWIRED = "(none)"`, used as `choices[0]` (`:141`); `_draw_inputs` (`:145`) maps index 0 to
  `producer = ""`; `wire_pass_input` (`project_session.py:847-865`) passes it to
  `PassGraph.with_input`, which at `pass_graph.py:163-166` does
  `if producer: inputs[uniform] = producer` `else: inputs.pop(uniform, None)` — it **deletes the
  key**. So after a user deliberately sets `cascade.u_df` to `(none)`, `entry.inputs` is byte-
  identical to a pass that was never wired. **Any rule keyed on "the key is absent" therefore
  re-wires against the user's explicit un-wire, on the next hot reload, silently.** For the RC
  example that is `u_df`, `u_prev`, `u_scene` — every unwire a reader makes while following the
  tutorial would spring back. This is not a wording gap; it is a missing model state. The spec must
  either (a) say the rule fires ONCE per sampler and records that it did (a new field, or a
  sentinel `""` value that `with_input` would have to stop deleting), or (b) scope the rule to
  `add_pass` and the *first* compile of a newly-declared sampler only — which is what the
  maintainer's verification step actually describes ("declare `u_df` in a new pass — it is wired"),
  and which needs a "samplers I have already seen" set somewhere. Neither exists today.
- A third state exists and is not considered: `entry.inputs` may name a pass that **no longer
  exists** (`pass_graph.py:76-78`, "An input naming a pass that does not exist is not an error: it
  reads black"). Is that "unwired" for the rule's purposes? A pass renamed from `df` to `sdf` leaves
  `cascade.u_df -> "df"` dangling; under one reading the rule re-wires `u_df` to nothing (no pass
  `df`), under another it leaves the dangling edge. Not stated.

**The copilot's `edit_shader` path does NOT hit the same seam.** `copilot/backend.py::_copilot_persist_shader`
(`:2042-2054`) is "the shared tail of every source edit" and does
`render_pass.release_program(new_text)` → `render_pass.compile()` → `write_text` →
`_sync_editor_from_disk`. Two differences from `watch.py`: it **does** compile eagerly (so the new
samplers ARE knowable right there), and it takes `render_pass: Pass` **only** — no document, no
graph, no session, and its docstring says so explicitly ("Takes the PASS, not the document"). So an
agent that adds `uniform sampler2D u_df;` via `edit_shader` gets **no** auto-wire under the spec as
written, while the same edit typed in the app's editor (which routes through `app.py:1375`'s save
path, then the graph-less `release_program`) gets one or not depending on which of (i)/(ii)/(iii)
above is chosen. **That asymmetry is a decision the spec does not make.** It matters: the copilot
writes shaders far more often than the maintainer types them, and W-D's own "Rule stated in … the
copilot prompt's pass block" tells the agent the rule exists.

Note also: the write path also lands on disk (`:2051`), so `watch.py` sees a changed mtime on the
next frame and reloads it again — meaning whichever seam is chosen, a copilot edit passes through
it twice. Round 1 did not look at this path.

---

## Task 2 — the W-B gate as specified

**Verdict: FAIL.** The gate as written catches **two of the four** strings the ledger complains
about. Round 1's coverage report concluded "W-B's `ast` gate can actually see the strings it
targets" — that is true for the two it names by their function, and false for the other two.

The spec's newly-added disclaimer —

> Strings built at runtime (f-strings, the vim status line's mode badge, the notification texts)
> are outside an `ast` literal walk; the gate covers the literal surfaces, and the review of each
> UI wave covers the rest by eye — stated so nobody reads the gate as total.

— is honest about the gate's limit but **does not resolve the defect, and its examples point away
from where the misses actually are.** The three it names (vim status line, notifications, generic
f-strings) are cosmetic omissions. The two real misses are `pass_settings.py:128` and
`pass_settings.py:178` — i.e. **finding #5's own "(?)" tooltip and finding #7's own size label**,
the two of the four ledger strings whose defect is *specifically* that a runtime-substituted value
overflows its box. So the disclaimer's "the review covers the rest by eye" is doing the load-bearing
work for exactly the class the ledger filed, and the gate is left guarding the strings that were
never the hard case. Per the repo's "a rule with no gate is a wish", the wave would ship the cut and
the check would not prevent the recurrence. All three fixes below are small and keep the gate an
`ast` walk — none of them requires running the UI.

**Census.** `grep` over `shaderbox/**/*.py` (raw call counts), cross-checked with an `ast` walk that
classifies the measured argument:

| Function | Call sites | Measured arg is a `str` literal | Not measurable by a literal walk |
|---|---|---|---|
| `help_marker` | 7 | 4 | **3** |
| `imgui.set_tooltip` | 19 | 9 | **10** |
| `imgui.separator_text` | 10 | 9 | **1** |
| `imgui.text_colored` | 63 (**11** with `COLOR.FG_DIM` first arg) | 11 | **20** (non-`FG_DIM` sites the gate skips by design, plus f-strings/variables among them) |

**The non-literal sites** (file:line, AST node of the argument the gate would measure):

*`help_marker`* — `popups/pass_settings.py:172` (`Subscript`: `_FORMATS[…][2]`);
`popups/pass_settings.py:191` (`IfExp`: the two size-help variants);
`popups/settings.py:307` (`Name`).

*`imgui.set_tooltip`* — `popups/emoji_picker.py:65` (`Attribute`);
**`popups/pass_settings.py:128` (`JoinedStr` — see below)**; `tabs/code.py:676` (f-string);
`tabs/document.py:242` (`Name`); `ui_primitives.py:211`, `:447`, `:842`, `:1161`, `:1252` (`Name` ×5);
`widgets/copilot_chat.py:679` (f-string).

*`imgui.separator_text`* — `popups/emoji_picker.py:54` (`Attribute`).

*`imgui.text_colored` with a non-literal second arg* — `exporters/telegram.py:761`;
`popups/lib_picker/search.py:60`, `:69`; `popups/lib_picker/tree.py:211`, `:338`;
`popups/settings.py:142`, `:298`; `tabs/code.py:206`, `:393`, `:405`, `:433`;
`tabs/document.py:43`, `:200`; `ui_primitives.py:699`, `:1045`, `:1053`;
`widgets/copilot_chat.py:394`, `:421`, `:571`, `:577`.

Note the `ui_primitives.py` cluster: five of the nineteen `set_tooltip` calls pass a `Name` because
`ui_primitives` is the shared draw layer — the tooltip *text* is the caller's literal, handed in as
a parameter. **The gate as specified measures the call site, so every tooltip that reaches the
screen through a `ui_primitives` helper is invisible to it**, including all of `pass_list.py`'s
button tooltips except the one raw `imgui.set_tooltip` at `:98`.

**The four ledger strings, one by one:**

| Ledger item | Site | Shape | Gate catches it? |
|---|---|---|---|
| #5, the "(?)" Reads tooltip | `popups/pass_settings.py:128` | `imgui.set_tooltip(` + **three implicitly-concatenated parts, the middle one an f-string** (`f"fills it ({_UNWIRED} leaves it black). …"`) → Python parses the whole concatenation as a single **`ast.JoinedStr`**, not a `Constant` | **NO** |
| #5, the empty-Reads line | `popups/pass_settings.py:136` | `imgui.text_colored(COLOR.FG_DIM, "nothing — declare a sampler2D uniform to read another pass")` | **YES** |
| #7, the size label | `popups/pass_settings.py:178` | `label_row(app.font_12, f"size ({w}, {h})", _CTRL_W, _ROW_LABEL_W)` — **`label_row`, not one of the four functions**, and an f-string besides | **NO** |
| #10, the gear tooltip | `widgets/pass_list.py:98` | `imgui.set_tooltip("Pass settings — what it reads, what it draws into")` | **YES** |

So the gate as written misses **exactly the two findings whose defect the ledger describes as
overflow of a variable-length string** — which is the class the gate exists to prevent recurring.
A cut made in W-B would land, and nothing would stop the next f-string tooltip from being written.

**What the gate must match instead** (three additions, each demonstrated above):

1. **Treat `ast.JoinedStr` as measurable.** Walk its `values`; count words in the `Constant` parts
   and score each `FormattedValue` as one word (a substituted value is one visual token: `(none)`,
   `1080×1080`). This is what makes `pass_settings.py:128` visible, and it also brings the ten
   `set_tooltip` f-strings and four `help_marker`/`text_colored` f-strings into the budget. Without
   it the gate's own coverage of `set_tooltip` is 9/19.
2. **Add `label_row` / `row_label` to the function set, measuring the `label` argument** —
   `ui_primitives.py:1088-1097` (`label_row(font, label, item_width, label_w)`), 17 call sites. This
   is the only way #7 is pinned, and D1's own text ("label 1-2 words … derived values in the
   control, never the label") is a rule *about labels*, which the four named functions do not draw.
   The budget for this one is not a word count but "no `FormattedValue` in a label", which is the
   literal statement of D1's last clause and is decidable from the AST.
3. **Decide what happens to a `Name` argument.** Nine of the 40-odd measurable sites pass a variable
   (mostly into `ui_primitives`). Either the gate resolves module-level `Final` constants (feasible:
   `_FORMATS` at `pass_settings.py` is a module constant, and resolving it covers `help_marker(…)`
   at `:172`, one of #5's eight complained-about markers), or it **asserts they are absent** —
   i.e. the rule becomes "a tooltip/help string is a literal at its call site", which is a positive
   requirement, enforceable, and makes the gate's own domain equal to its call-site census. A gate
   that silently skips a third of its domain is the "checker that quietly narrows its own domain"
   shape; the spec should state which of the two it is, and the test should assert the count of
   sites it measured so the domain cannot shrink unnoticed.

---

## Task 3 — D9 default wiring against the shipped examples

**Verdict: PARTIAL.** The rule wires every sampler in both multi-pass examples after W-D's rename —
that half is sound. The accidental-capture half is real, is not hypothetical, and the spec has no
guard for it.

Enumerated from all seven `document_examples/*/graph.json` + every `sampler2D` declaration in their
`passes/*.glsl`.

**`77a84d27-…` Radiance Cascades** (passes: `paint`, `seed`, `jfa`, `df`, `cascade`, `composite`):

| Pass | sampler (today) | after W-D rename | names an existing pass? | rule wires it? |
|---|---|---|---|---|
| `seed` | `u_scene` | `u_paint` | yes (`paint`) | **yes** |
| `jfa` | `u_seed` | `u_seed` (unchanged) | yes (`seed`) | **yes** |
| `jfa` | `u_prev` | `u_prev` | self (`u_prev` → itself) | **yes** |
| `df` | `u_jfa` | `u_jfa` (unchanged) | yes (`jfa`) | **yes** |
| `cascade` | `u_scene` | `u_paint` | yes (`paint`) | **yes** |
| `cascade` | `u_df` | `u_df` (unchanged) | yes (`df`) | **yes** |
| `cascade` | `u_prev` | `u_prev` | self | **yes** |
| `composite` | `u_light` | `u_cascade` | yes (`cascade`) | **yes** |
| `composite` | `u_scene` | `u_paint` | yes (`paint`) | **yes** |

9/9. Note `u_seed`, `u_jfa`, `u_df` already satisfy D9 today — the spec's rename list correctly
names only the three that do not.

**`1c4f8a20-…` Bloom Chain** (passes: `scene`, `bright`, `blur`, `trail`, `composite`):

| Pass | sampler (today) | after W-D rename | names an existing pass? | rule wires it? |
|---|---|---|---|---|
| `bright` | `u_src` | `u_scene` | yes (`scene`) | **yes** |
| `blur` | `u_src` | `u_bright` | yes (`bright`) | **yes** |
| `trail` | `u_src` | `u_scene` | yes (`scene`) | **yes** |
| `trail` | `u_prev` | `u_prev` | self (`trail` reads `trail`) | **yes** |
| `composite` | `u_lit` | `u_scene` | yes (`scene`) | **yes** |
| `composite` | `u_glow` | `u_blur` | yes (`blur`) | **yes** |
| `composite` | `u_trail` | `u_trail` (stays) | yes (`trail`) | **yes** |

7/7. The spec's Bloom mapping matches the `graph.json` exactly. **One thing the spec's shader-side
list misses:** `blur.frag.glsl:9`, `trail.frag.glsl:10-11` and `composite.frag.glsl:8-10` carry
trailing comments naming the producer (`// filled by \`bright\``, `// filled by \`scene\``, …). An
exact-token rename of the uniform leaves those comments *correct* (they name passes, not uniforms) —
but `composite.frag.glsl:8`'s `u_lit // filled by \`scene\`` becomes `u_scene // filled by \`scene\``,
which is now redundant noise, and under D9 the whole comment convention is superseded by the naming
rule. W-H generates the tutorial's code blocks from these files (D8), so the comments ship. Worth
one line in W-D.

**Accidental capture of a user-bound TEXTURE — the hazard is real and unguarded.**

The three single-pass examples that declare samplers are the test:
`73ea2431-…/passes/main.frag.glsl:5-6` declares `uniform sampler2D u_image;` and
`uniform sampler2D u_video;`, and its `document.json` binds both to files
(`media/main/u_image.png`, `media/main/u_video.mp4`, `ui_uniforms` `input_type: "texture"`).
Today no capture is possible: the document has one pass, named `main`, and there is no pass named
`image` or `video`.

**But the mechanism that makes capture harmful is already in place.** `core.py:373`:

```python
value = inputs.get(uniform.name, self.uniform_values.get(uniform.name))
```

A graph-supplied `inputs` entry **wins over** the user's bound texture, and `Pass.render`'s docstring
(`core.py:342-345`) states the split as a deliberate design ("the graph owns those bindings, the
pass owns the ones the user set"). So a `u_<x>` edge auto-created for a sampler that already holds a
bound image silently replaces the picture with a pass render — no error, no notification; the gear's
Reads combo would show it as wired (which is what the spec asks for), and the Document tab's texture
row would still show the bound file (`tabs/document.py:65-75` reads `uniform_values`, which is
untouched). Two rows of the same UI disagreeing, with no warning.

**The repro that produces it under D9 needs no contrivance:** a user with a media-input document
(the shipped `Media Input` example is the starting point the copilot's `create_document(example=…)`
offers) adds a second pass and names it `image` or `video` — plausible names for a pass in exactly
that document. The next hot reload/`add_pass` wires `main.u_image` to it, and the bound PNG stops
being read. The spec's "explicit choice always wins" clause does not cover this: the explicit choice
here is a *texture binding in `uniform_values`*, not a graph edge, and the rule as written only
consults `PassEntry.inputs`.

**What the spec must add:** the rule skips a sampler whose `uniform_values` entry is a
`MediaWithTexture` (the type `core.py:382` tests) — i.e. a user-bound image/video is itself an
explicit choice. That is one condition and it is decidable at the same seam that reads the samplers.

---

## Task 4 — five items a fresh read finds that round 1 did not

**Verdict: PARTIAL** for the spec (five items, each a real gap; none of them fatal to the design).

**F1 — the Resolution combo is fed a hardcoded current-index of `0`, so W-A's "route through
`set_canvas_size`" fixes the funnel but not the control.**
`tabs/document.py:109`: `new_res_idx = imgui.combo("##resolution", 0, resolution_items)[1]` — the
selected index is the literal `0`, and `resolution_items[0]` is always the *current* size
(`:77-78`). The combo therefore cannot display a picked preset as selected, and `:110`'s
`if new_res_idx != 0` is the only reason it works at all. W-A ("the Document tab's Resolution combo
routes through `Document.set_canvas_size`") describes replacing `:111`'s
`render_pass.canvas.set_size((w, h))` — correct, and round 1 agreed — but adding two `input_int`s
beside a combo whose selection state is a literal will produce a control that resets its display
every frame. **Spec line concerned:** W-A bullet 1-2. One sentence: the combo's index becomes state,
or the free entry replaces the combo's current-size slot.

**F2 — the resolution list is built from `render_pass` (the output pass), not the document.**
`tabs/document.py:65-74` iterates `ui_document.document.render_pass.get_active_uniforms()` to collect
bound-texture sizes, and `:59` reads `render_pass.canvas.texture.size` as the current size. Under
065 the document owns `canvas_size` (`document.py:284-293`), and `panel_pass` (`app.py:571-582`)
already establishes that the Document tab edits the **panel** pass, not the output. So the
Resolution row is the one place in that tab still keyed to the output pass. W-A's file list names
`tabs/document.py`; its text does not name this. **Spec line concerned:** W-A, "the combo's fixed
list gains … AND a free `W × H` entry".

**F3 — `_scriptable_uniforms_for` is not the only `.render_pass` script seam; `_stopped_for` and
`ProjectSession.tick` are two more, and one of them is in W-G's own sentence as if it were
`_scriptable_uniforms_for`'s sibling.** `project_session.py:598` hands
`ui_document.document.render_pass` to `script_engine.tick` as the `EngineNode`. W-G says the tick
callers "stop passing `.render_pass`" — correct — but the same paragraph says
`_scriptable_uniforms_for` is "the sibling of the 068 D7 defect", implying two sites. There are
four: `:598` (live tick), `_scriptable_uniforms_for` (`:742-752`), the export pre-render closure,
and `write_script_source`'s dry-run. Round 1 found the last two; the live tick at `:598` is named
only obliquely ("every caller in `project_session.py` — `tick`, …"), and it is the one whose
*protocol type* changes. **Spec line concerned:** W-G bullet 1.

**F4 — `ScriptStatus.driven_count` and the strip's soft-error keying are pass-blind, and W-G asks
for a per-pass shader-tab strip without saying what keys it.** `engine.py:272` builds
`driven_count=len(document.last_driven)` from a document-scoped set; `project_session.py:625`
(`script_has_error`) keys errors by `(document_id, DOCUMENT_SCRIPT_BASENAME)`. W-G's "add the same
strip on a SHADER tab whose pass the script names wrongly (today only the script tab shows them)"
requires the engine to record *which pass* an orphan key named — a third field on the error record,
not just a pass-qualified `last_driven`. The spec names `tabs/code.py:130` as the existing strip but
not the engine-side error shape that would feed a per-pass one. **Spec line concerned:** W-G,
"add the same strip on a SHADER tab".

**F5 — `load_document_from_dir` bypasses `drop_unknown`, so W-G's shape change gets a
double-warning, and the spec's "hand-edited, NO migration code" leaves a log line nobody expects.**
`ui_models.py:501-505` logs `Ignored invalid UIDocumentState keys …` for unknown keys, then `:514`'s
`drop_invalid` logs `Ignoring invalid document '<id>'.stopped_uniforms` for a *known* key whose value
no longer validates. Every `projects/dev` document and every user document on the maintainer's disk
that still carries the old `list[str]` shape will emit the second warning once, on first load after
the wave. That is the intended fail-soft behaviour and needs no code — but the spec's manual-
verification list for W-G ("a script returning `{"paint": {…}, "u_time_scale": 0.5}` drives both…")
does not mention that the first launch after the wave logs a warning per stale document, which will
read as a defect during the walk. One line in W-G's verification list. **Spec line concerned:**
W-G, "the five shipped examples' `document.json` files … are hand-edited, NO migration code".

---

## False trails — probed, fine, do not re-check

- **`Document.first_render_done` vs a new `Pass.first_render_done`.** Confirmed independently of
  round 1: `document.py:238` is document-scoped with five readers (`ui.py:228/235/304/309`,
  `document_grid.py:110`, `examples.py:103`, `test_lazy_compile.py:86/88`); `core.py:150-168` has no
  such field. No shadowing. The *interaction* I raise in Task 1(a) point 3 is about the
  **document** flag's `canvas is None` guard, not about a name clash.
- **`render_media` / `_render_image` / `_render_media_into` under W-C.** All three call
  `self.render(…, canvas=canvas)` with no positional third argument
  (`document.py:528`, `:620`, `:646-650`). A keyword-only `target` with a `None` default is
  source-compatible with every one of the 20 `render(` call sites I enumerated across
  `shaderbox/` and `tests/`. No call site needs editing.
- **`PassEntry` immutability blocking the auto-wire.** `pass_graph.py:81` is
  `model_config = ConfigDict(frozen=True)`, but `PassGraph.with_input` already exists and is the
  sanctioned mutator (`project_session.py:863`). Frozen-ness is not the obstacle in Task 1(c) —
  *reachability of the graph from `watch.py`* is.
- **`begin_frame` double-advance under a W-C first-render sweep.** `document.py:307-309` makes a
  repeat call for the same `frame` a no-op, and `ui.py:246` is the only per-frame caller. A second
  `render()` in one frame does not advance history. (The *iteration*-level `_swap_feedback` at
  `document.py:449` is a separate matter and is raised in Task 1(a) point 4.)
- **The `u_light_*` / `u_glow_*` prefix trap the spec names.** Verified the two files:
  `8d454b7b-…/passes/main.frag.glsl:56-60` declares `u_light_*` uniforms and
  `0b0d16bb-…/passes/main.frag.glsl:12-13` declares `u_glow_*`, and both documents persist those
  values in `document.json`. Neither declares a `sampler2D`. The spec's exact-token warning is
  correct and sufficient; no further audit needed.
- **`stopped_uniforms` values in the repo.** Every occurrence in every `.json` under the repo is the
  empty list — `grep -rn stopped_uniforms --include=*.json . | grep -v '\[\]'` returns nothing. So
  the hand-edit is a pure shape change with no data to translate, in all 18 files that carry the key.
- **The copilot has no `wire_pass_input` / pass tools.** Re-confirmed for Task 1(c) specifically:
  `copilot/backend.py` has no call to `wire_pass_input`, `add_pass` or `set_pass_target`. So the
  auto-wire rule has no agent-facing tool to mirror it, and the asymmetry I raise in Task 1(c) is
  about `edit_shader`'s *side effect*, not about a missing tool.

---

## Verdicts

| Task | Verdict | Count |
|---|---|---|
| 1 — implementability of the three highest-stakes changes | **PARTIAL** | (a) 4 undecided + 1 uncovered branch pair; (b) 4 unnamed consumers + a wrong example count; (c) **FAIL** — the hot-reload seam does not exist as described, and "unwired" has no representation that survives an explicit un-wire |
| 2 — the W-B gate as specified | **FAIL** | catches 2 of the 4 ledger strings; 3 required changes to the match set |
| 3 — D9 wiring vs the shipped examples | **PARTIAL** | 16/16 samplers wire correctly; 1 unguarded texture-capture hazard |
| 4 — fresh findings | **PARTIAL** | 5 items, F1–F5 |

**Recommendation.** W-D's default-wiring bullet and W-B's gate bullet both need a rewrite before
implementation — the first because it names a seam that has neither the graph nor the pass's
uniforms, the second because it misses the two strings whose defect it exists to prevent. W-C's
`target=` and W-G's pass-qualification are implementable as designed; each needs the enumerated
decisions written into the spec so an implementer does not make them silently. Task 3's texture-
capture guard is one condition and should ship with the rule, not after it.
