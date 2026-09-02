# 069 W-D - post-implementation spec-fidelity and architecture review

Reviewed commit: `f18a7d3` ("069 W-D: a sampler's name is its default wire"), 25 files changed
(the commit's own stat line says 25; the task brief said 26). Anchors: `70_wave_d_wiring_naming.md`,
`01_spec.md § W-D / D9 / D12`, `00_findings.md` #19 and #37, `ai_docs/conventions.md`,
`ai_docs/dev_flow.md`, `.claude/skills/copilot-llm-agent-design/SKILL.md`.

W-H is editing the tree concurrently (`ai_docs/features/068_radiance_cascades/build_tutorial.py`,
`help_content.py`, RC's `document.json` / `graph.json` / `jfa.frag.glsl` / `paint.frag.glsl`), so
every W-D file was read via `git show f18a7d3:<path>` or from the diff. Nothing tracked was edited
by this review.

## Verdict

| Dimension | Verdict |
|---|---|
| Wave-spec fidelity | **PARTIAL** - every design decision landed, but the spec was never updated: it has no post-implementation Review-history section and six landed deviations go unrecorded (F2). One decision-7 behaviour is defeated by call ordering (F1). |
| Parent fidelity | **PARTIAL** - every W-D bullet of `01_spec.md` is satisfied by landed code, but the refuted "api-lock tests for examples updated" claim still stands uncorrected in the parent at `01_spec.md:184` (F3). |
| Findings closure | **PASS** - #19 closed as option B, #37 closed whole (rule + rename + default wiring), and neither complaint recurs. |
| Renames | **PASS** - all 18 tokens, all 6 comments, all 4 prefix traps intact, no single-pass example touched, `projects/dev` untouched, the D9 gate exists and was mutation-tested red. |
| Architecture | **PASS** - `pass_graph.py` stays GL-free, `effective_inputs` is pure, all six consumers read one graph, no seventh planner site exists. Two small placement notes (F5, F6). |
| Docs | **PARTIAL** - `conventions.md`, `dev_flow.md`, `help_content.py` and the prompt all landed correctly; two new module docstrings narrate development history against the repo's own comment rule (F4), and the roadmap banner is stale (F7). |

## Coverage table

### Design decisions

| # | Decision | Status | Evidence |
|---|---|---|---|
| 1 | `effective_inputs` is the one resolution, and it is pure | **landed, with a recorded-nowhere deviation** | `shaderbox/pass_graph.py:225-277`. Signature matches the spec exactly. Body is a loop, not the spec's dict comprehension over `samplers`, and it seeds `resolved` from every truthy stored edge first. See F2.1 - the deviation is **necessary**, not cosmetic. |
| 1a | `_auto_source` strips `u_`, maps `u_prev` to `consumer` | landed | `pass_graph.py:229-240`. Verified: `effective_inputs(PassEntry(), ["u_prev"], {"cascade","prev"}, "cascade") == {"u_prev": "cascade"}`. |
| 1b | A name without the `u_` prefix gets no auto edge | landed | `pass_graph.py:236-237`; `effective_inputs(PassEntry(), ["df"], {"df"}, "edge") == {}` (run). |
| 2 | Sampler names from the COMPILED program, never `get_active_uniforms()` | landed | `document.py:436-448`. Comment matches the spec's text verbatim except "W-C's first-render sweep" became "The live loop's first-render sweep" - correct, since a shipped repo must not name a wave. Pinned by `test_an_uncompiled_pass_contributes_no_auto_edge_and_compiles_nothing` (passes). |
| 3 | `Document` builds one effective graph per render; the planner sees it | landed | `document.py:452-476` (`effective_graph`), `:494` (`resolved_graph` bound once), `:498`, `:500`, `:517`. `self.graph` still what is saved - confirmed by the worked example: after 8 frames `document.graph.passes["edge"].inputs == {}`. |
| 3a | The five named consumers all read `effective_graph` | landed, **plus a sixth** | `document.py:494` (binder+planner), `widgets/pass_list.py:154` (both strip calls), `copilot/backend.py:782`, `popups/pass_settings.py:160`, and `document.py:377` (`has_feedback`). A grep over `plan_passes(` / `plan_for_output(` / `evaluation_order(` in `shaderbox/` returns no seventh call site outside `pass_graph.py` itself. |
| 4 | The media-bound exclusion lives at the resolution seam | landed | `document.py:222-227` (`_is_user_bound`), `:466-470` (the `bound` argument). `core.py` untouched (0 lines in the diff). |
| 5 | `with_input("")` stores; `without_input` deletes | landed | `pass_graph.py:177-192` and `:194-201`. Pinned through disk by `test_a_stored_empty_string_stays_black_across_a_reload` (passes). |
| 5a | `ProjectSession.unwire_pass_input` | landed | `project_session.py:892-903`; same validate-save-return-`""` shape as its siblings. |
| 6 | The gear's combo has three kinds of item | landed | `popups/pass_settings.py:152-183`. `choices = [f"auto: {auto or 'none'}", _UNWIRED, *sorted(document.passes)]` at `:164`; read by stored state at `:165-173`; write by position at `:176-181`. |
| 7 | The copilot's pass view resolves the same way | **landed but defeated by ordering** | `copilot/backend.py:780-792`, `_input_row` at `:272-284`. The three row texts are exactly as specified. **F1**: `effective_graph()` is called before the loop that compiles each pass, so on a not-yet-compiled name-wired document every auto sampler is reported to the model as `(nothing; reads BLACK)`. |
| 8 | `add_pass` stores no default wiring (no change) | landed as no-change | `project_session.py` diff carries only `unwire_pass_input` and the `wire_pass_input` docstring. |
| 9 | The naming rename: 18 tokens, exact-token | landed | See the rename table below. |
| 10 | `graph.json` hand-edits, nothing else | landed | Both `graph.json` diffs are key renames only; no `""` added; no `document.json` touched; `projects/dev` absent from the diff. |
| 11 | The strip tune, and the strip plans the effective graph | landed | `widgets/pass_list.py:100-112` (sublines and the error line gone; `PassEntry` import dropped at `:18`), `:154-158` and `:163` (both planner calls take `resolved`). `preview_cell` keeps its `sublines` parameter. Spacing untouched (`theme.py` and `ui_primitives.py` are 0 lines of the diff), as decision 11 requires. |
| 12 | The rule's two written homes | landed | `help_content.py:175-179` (four sentences, first in the `pass_settings` body); `copilot/prompt.py:65-67`, inside `_SYSTEM_PROMPT` (defined at `:47`), which `build_prompt` maps to `Volatility.STATIC` at `:436`. |

### Worked example, executed against the real engine

Run on a tmp copy of the RC example, adding an `edge` pass exactly as the spec's script describes:

| Spec frame | Spec says | Engine says | Match |
|---|---|---|---|
| 0, the add | stub compiles, `_sampler_names` returns `[]`, effective `{}`, viewer black | `F0 samplers: [] eff: {}` | yes |
| 1, save with `u_df` | `release_program` clears `program`, effective carries no auto edge | `F1 program is None: True eff: {}` | yes |
| 2, first render after reload | "draws `edge` with an unresolved `u_df` - **black, one frame**" | `F2 max px: 255`, and `uniform_values["u_df"]` is the seeded `Image` with `is_default_image() == True` | **no** - see F8 |
| 3, the edge appears | effective `{"u_df": "df"}`, order `paint, seed, jfa, df, edge`, tile shows the distance field | `F3 eff: {'u_df': 'df'}`; `F3 order: ['paint', 'seed', 'jfa', 'df', 'edge']`; `F3 max px: 154` | yes |
| never | `graph.json` untouched | after 8 frames `document.graph.passes["edge"].inputs == {}` | yes |

The gear's three states are pinned by `test_the_gear_shows_three_distinct_states`, which asserts the
full item list on both automatic outcomes: `(0, ["auto: df", "(none)", "df", "edge"])` and, after a
rename that removes `df`, `(0, ["auto: none", "(none)", "edge", "field"])`. That test needs the
`app` fixture, which segfaults in this reviewer's shell inside `glfw.get_video_mode`
(`tests/conftest.py:61` -> `shaderbox/app.py:150`). Confirmed environmental, not W-D: the untouched
`tests/test_canvas_fields.py` crashes identically at the same frame. Read as code rather than run.

### Files-touched rows

| Row | Status |
|---|---|
| `shaderbox/pass_graph.py` | landed; the promised "no new imports beyond `collections.abc`" holds exactly (`pass_graph.py:28`). |
| `shaderbox/document.py` | landed; all three imports as promised (`OpenGL.GL.GL_SAMPLER_2D`, `MediaWithTexture` + `is_default_image`, `effective_inputs`). |
| `shaderbox/project_session.py` | landed. |
| `shaderbox/popups/pass_settings.py` | landed. |
| `shaderbox/widgets/pass_list.py` | landed. |
| `shaderbox/copilot/backend.py` | landed (see F1). |
| `shaderbox/copilot/prompt.py` | landed. |
| `shaderbox/help_content.py` | landed. |
| Both examples' dirs | landed. |
| **`tests/test_pass_graph.py`** | **not touched.** Correctly so - it asserts nothing about the old `""` semantics (its only `with_input` use, at `:303`, wires a real name). But there is now no `PassGraph`-level unit test of `without_input`; it is covered only through `ProjectSession`. |
| **`tests/test_document_graph.py`** | **not touched.** Correctly so - it contains no `with_input` / `effective` reference at all. |
| `ai_docs/conventions.md` | landed - one bullet in the pass-graph entry, as decision 12's open question 4 defaulted. |
| `ai_docs/dev_flow.md` | landed - all three module-map edits present. |
| `tests/test_ui_prose_budget.py` | landed - the `_UNMEASURABLE` reason is now `"the pass's own name"`. |

### Tests

| Test | Status | Evidence |
|---|---|---|
| `test_effective_inputs_over_every_state` | landed, all nine rows plus `u_prev` plus the no-prefix case | `tests/test_default_wiring.py:66-92`; passes |
| `test_the_planner_orders_an_auto_edge` | landed, both halves (order and cycle) | `:95-133`; passes |
| `test_u_df_beside_df_renders_without_the_gear` | landed, both assertions (edge and pixels), plus a third that `graph.json` stays empty | `:156-172`; passes |
| `test_a_stored_empty_string_stays_black_across_a_reload` | landed, through disk as required | `:175-194`; passes |
| `test_an_uncompiled_pass_contributes_no_auto_edge_and_compiles_nothing` | landed, asserts on compile state | `:197-215`; passes |
| `test_the_gear_shows_three_distinct_states` | landed | `:238-282`; not run here (environmental, above) |
| `test_examples_resolve.py::test_every_example_input_uniform_names_its_source` | landed | `tests/test_examples_resolve.py:39-52`; **mutation-tested**: reverting Bloom's `composite.u_scene` to `u_lit` makes it fail with `+ u_lit`, and the file was restored |
| `test_every_multi_pass_example_compiles_every_pass` | landed, parametrised over both examples | `tests/test_default_wiring.py:288-303`; passes |
| `test_unwiring_stores_an_explicit_none_and_unwire_forgets_it` | landed, renamed in place, three assertions live and after reload | `tests/test_pass_verbs.py:186-203` |
| `test_a_pass_s_wiring_is_shown_including_what_is_unwired` (extended) | landed | `tests/test_copilot_passes.py:157-166` |
| `test_a_pass_wired_only_by_its_uniform_name_is_shown_as_filled` (new case) | landed | `tests/test_copilot_passes.py:169-180`. **Passes only because it compiles the pass before reading the working set** (`:177-178`), which masks F1. |
| `test_the_strip_draws_no_sublines` | landed | `tests/test_pass_verbs.py:497-515` |
| `test_an_auto_wired_ancestor_is_not_washed_stale` | landed | `:518-548` |
| `test_the_strip_orders_a_name_wired_document_topologically` | landed, with the alphabetical-disagreeing names the spec required | `:551-568` |
| `test_a_u_prev_pass_has_feedback_without_a_stored_edge` | **not in the spec** - added with the `has_feedback` deviation | `tests/test_default_wiring.py:306-328`; passes |

### Manual verification steps

Steps 1-8 are maintainer-in-the-app work and were not performed here. Steps 3, 4, 5 and 8 have
automated proxies that were run or read: the rename (the D9 gate plus both compile tests), the cold
auto wire (the worked-example script above), the three combo states (the gear test), and the
copilot's view (`test_a_pass_wired_only_by_its_uniform_name_is_shown_as_filled` - but see F1, which
step 8 would surface if the maintainer opens the chat before the passes have rendered). Step 4's
promise "within a frame or two the viewer shows the distance field" is confirmed exactly. Step 4's
**intermediate** frame does not match the spec's prose - see F8.

### The rename table: all 18 sites

Every row of decision 9's table is present in the diff. Verification beyond the diff:

- **No old token survives anywhere in the examples tree.** `grep -rnE '\bu_(src|lit|glow|light|scene)\b' shaderbox/resources/document_examples/` returns nine hits, all of them the NEW `u_scene` in Bloom (three `graph.json` keys, three declarations, three reads). Zero `u_src`, `u_lit`, `u_glow`, `u_light`.
- **All four prefix traps intact.** `u_light_ambient` / `u_light_sky_key` / `u_light_moon_key` / `u_light_cool_color` / `u_light_warm_color` still in `8d454b7b…/passes/main.frag.glsl:56-60` with their `document.json` values at `:55`, `:62`, `:371`, `:395`; `u_glow_strength` / `u_glow_radius` in `0b0d16bb…/passes/main.frag.glsl:12-13` with values at `document.json:12-13`; `u_light_radius` in RC's own `paint.frag.glsl:20` and read at `:35`; `u_trail_mix` in Bloom's `composite.frag.glsl:13`.
- **The six comments.** `grep -rni "illed by"` over the examples tree returns exactly one line: `1c4f8a20…/passes/trail.frag.glsl:11`, the `u_prev` comment the spec keeps. RC's `jfa.frag.glsl` and `cascade.frag.glsl` prose comments were left alone as specified.
- **No single-pass example touched**; **`projects/dev` absent from the commit**; **`tests/shader_lib_api_lock.json` not regenerated**.

## Findings

### F1 - `_pass_views` resolves before it compiles, so the model is told an auto-wired sampler reads BLACK

**Claim.** Decision 7 exists to stop one specific defect: "the copilot would be told a sampler reads
BLACK while the renderer fills it - a false fact on the channel the model reads, which § 4 of the
copilot skill names as the worst kind of prompt defect." The landed code still produces that fact,
because of the order of two calls.

`copilot/backend.py:782` binds `resolved = document.effective_graph()` **before** the loop. Inside
the loop, `:791` calls `_sampler_uniform_names(render_pass)`, which is
`render_pass.get_active_uniforms()` (`backend.py:286-291`) - and that **compiles** the pass. So on
the first working-set read of a document whose passes have not rendered yet, `effective_graph()`
sees `program is None` for every pass, returns no auto edges, and then the row builder compiles each
pass and finds samplers the now-stale `resolved` knows nothing about.

**Evidence.** On a copy of Bloom with every stored edge removed (`without_input`), so it is wired
by name alone:

```
all programs None: True
effective for composite: {}
sampler names (after compile): ['u_blur', 'u_scene', 'u_trail']
-> rows would say 'nothing; reads BLACK' for all of: ['u_blur', 'u_scene', 'u_trail']
effective AFTER the compile: {'u_blur': 'blur', 'u_scene': 'scene', 'u_trail': 'trail'}
```

The last line is the renderer's answer. The third line is what the model is told. That is the false
fact, verbatim.

`test_a_pass_wired_only_by_its_uniform_name_is_shown_as_filled`
(`tests/test_copilot_passes.py:169-180`) does not catch it because it calls
`document.passes["composite"].compile()` at `:178`, one line before `read_working_set()` - so the
program exists by the time `effective_graph()` runs. The test asserts the right thing about a state
it arranges past the bug.

**Fix.** In `_pass_views`, gather the sampler names for every pass first and resolve afterwards:
build `per_pass = {name: _sampler_uniform_names(document.passes[name]) for name in sorted(document.passes)}`
before the `resolved = document.effective_graph()` line, then read rows from `per_pass[name]` inside
the loop; and change the test's falsifier so the pass is left uncompiled before `read_working_set()`.
Verified: with the gather hoisted, the same Bloom document yields
`FIX: effective for composite: {'u_blur': 'blur', 'u_scene': 'scene', 'u_trail': 'trail'}`.

### F2 - Six landed deviations are unrecorded: the spec has no post-implementation Review-history section

**Claim.** `70_wave_d_wiring_naming.md` is byte-identical to its state at `453c4f3` ("069: lock the
W-E, W-G, W-D specs; W-F closure rounds"). `git log --oneline -- ai_docs/features/069_tutorial_walk_findings/70_wave_d_wiring_naming.md`
returns exactly one commit, and `git diff f18a7d3 HEAD -- ai_docs/features/069_tutorial_walk_findings/`
is empty. Its `## Review history` carries Round 1 only. Six deviations landed and none is written
down. Under this repo's own rule that a decision living only in chat is lost on `/clear`, each is a
finding on its own; the missing section is what makes them one finding.

**Evidence, deviation by deviation.**

1. **`effective_inputs` carries every stored edge through unconditionally.** The spec (decision 1)
   says "The body is a dict comprehension over `samplers`". The landed body
   (`pass_graph.py:271-277`) instead seeds `resolved = {u: src for u, src in entry.inputs.items() if src}`
   and only then loops the samplers. This is **necessary, not stylistic**: under the spec's literal
   comprehension an uncompiled pass (`samplers == []`) would return `{}` and lose every edge
   `graph.json` holds, so a freshly loaded document would plan with no edges at all. Confirmed:
   `effective_inputs(PassEntry(inputs={"u_a": "a"}), [], {"a"}, "b")` returns `{'u_a': 'a'}` under
   the landed code. The implementer documented it inside the docstring
   (`pass_graph.py:265-267`, "Every stored edge is carried through whether or not `samplers` names
   it"), which is the right place for the mechanism - but the spec still describes a shape the code
   does not have. Second-order consequence, also unrecorded: a stored edge on a uniform the program
   does not declare survives into the effective graph
   (`effective_inputs(PassEntry(inputs={"u_ghost": "df", "u_df": "df"}), ["u_df"], {"df"}, "edge")`
   returns both keys). Harmless - `core.py` binds by declared uniform - but it is a behaviour the
   spec's shape would not have had.
2. **`has_feedback` is a sixth consumer.** `document.py:377` now plans the effective graph. The spec
   names five consumers twice (decision 3's closing paragraph, and premise 16 which calls
   `_pass_views` "a fifth consumer"). Correct and load-bearing: `has_feedback` gates the Clear
   canvas button, and a `u_prev` pass wired by name alone was invisible to it. Pinned by
   `test_a_u_prev_pass_has_feedback_without_a_stored_edge`, itself an unrecorded seventh test.
   `conventions.md:215-216` already names all six, so the DOC is right and only the spec is behind.
3. **An explicit black bind at the render seam.** `document.py:538-546`. The spec asserts the
   behaviour ("the next render binds the 1x1 black texture") in the worked example but names no
   mechanism, and decision 4 explicitly says `core.py` keeps binding what the document hands it.
   The landed dict comprehension over `stored_inputs` is the mechanism, and it is required: without
   it a `""` sampler is absent from `inputs` and `core.py:381` falls through to the seeded default
   photo. Correct.
4. **`_pass_views` hoists `effective_graph()` out of the loop.** The spec's snippet puts
   `document.effective_graph()` inside the per-pass loop; the code binds it once at `:782`. The
   hoist is right for cost and is what F1 is about.
5. **A shim in `tests/test_copilot_script_tools.py`'s stub** (`+3` lines): the fake document gains
   `effective_graph=lambda: types.SimpleNamespace(passes={})`. Necessary, since `_pass_views` now
   calls it.
6. **Two module docstrings gained paragraphs** (`widgets/pass_list.py`, `popups/pass_settings.py`) -
   see F4 for why they are also a rule violation.

**Fix.** Add a `**Post-implementation, `f18a7d3`**` subsection to
`70_wave_d_wiring_naming.md ## Review history` recording all six, with deviation 1 restated as the
decided shape (the loop that carries stored edges through) rather than as a departure, and correct
decision 1's body sentence and decision 3's "five consumers" to six in place.

### F3 - The parent spec's refuted api-lock claim still stands

**Claim.** `01_spec.md:184` still reads "Rule stated in the Help panel's pass section and the
copilot prompt's pass block; **the api-lock tests for examples updated.**" The wave spec's premise
15 refutes it outright ("There is no example-keyed api-lock elsewhere ... no lock file is
regenerated"), and the implementation confirms the refutation:
`tests/shader_lib_api_lock.json` is absent from `f18a7d3`'s file list.

The wave spec records the correction, so the knowledge is not lost - but the parent is the document
a later reader of § W-D opens first, and it names a job that does not exist and never did. Nothing
in the parent points at the wave spec's correction; § W-D's text is unchanged since the wave spec
was written.

**Fix.** In `01_spec.md § W-D`, replace the clause with what the wave actually shipped as the
rename's gate: "verified by both multi-pass examples compiling every pass plus a D9 naming gate
over every shipped `graph.json` (`70_wave_d_wiring_naming.md § Verified / corrected premises` 15) -
no api-lock file is involved."

### F4 - Two new module docstrings narrate development history

**Claim.** `ai_docs/conventions.md:30-32` states the rule: "When a comment IS warranted, it states
what's non-obvious about the code as it is NOW - **never narrates development history.** Banned:
the bug-we-hit story, the why-we-changed-it backstory". Both new docstring paragraphs are that
story.

**Evidence.**

`shaderbox/widgets/pass_list.py`, module docstring:

> A tile is a picture and a name, nothing else: the wiring lines **it used to carry** were ellipsized
> to nothing at this width, and the error line **was a second spelling** of the red border already
> drawn.

`shaderbox/popups/pass_settings.py`, module docstring:

> ... Three stored states, three distinct readings -- a combo that **showed** one label for a working
> wire and a black one **is what this replaced.**

The first clause of each is the rule-compliant half ("A tile is a picture and a name"; "The combo
carries two synthetic items ahead of the names"). The trailing clauses are the backstory, and both
belong to the commit message, which already carries them nearly word for word.

**Fix.** Truncate each paragraph at its statement of the current mechanism: drop everything from
"the wiring lines it used to carry" onward in `pass_list.py`, and drop the final sentence
("Three stored states, three distinct readings -- ... is what this replaced.") in
`pass_settings.py`, keeping the two sentences that describe what the two synthetic items ARE.

### F5 - `_is_user_bound` is imported privately across a package boundary

**Claim.** `popups/pass_settings.py:24` does `from shaderbox.document import _is_user_bound`. The
spec sanctions it ("that predicate lives in `document.py` beside its only other caller; the gear
imports it from there, which is allowed"). It works, and the reasoning about `media.py` is correct:
`media.py` imports moderngl (`shaderbox/media.py:11`), so it cannot host the predicate without
costing `pass_graph.py` its GL-freedom, and `pass_graph.py` genuinely stays GL-free (its only
imports are `collections.abc`, `dataclasses`, `typing` and `pydantic`, `pass_graph.py:28-32`).

The note: a leading-underscore name crossing a module boundary says "private" and is then read by
someone else, and there is exactly one other instance in the package
(`popups/lib_picker/tree.py:24` imports `_ellipsize` from `ui_primitives`). Two instances is not a
pattern, and this one is now the second.

**Fix.** Rename it `is_user_bound` in `document.py` and update both call sites. One line each,
and it makes the shared predicate say what it is.

### F6 - `_black_texture`'s lifetime, checked and clean

**Claim.** No defect; recorded because the task asked. `_black_texture` (`document.py:391-395`) is
pre-existing (not in the diff) and unchanged: one lazily-built 1x1 RGBA texture cached on
`self._black`, per document. W-D adds one new call site (`:543`) beside the existing one (`:557`).
It is the same texture object, so the new explicit-none binding costs no allocation, and its
release follows the document's existing path. Correct home.

### F7 - The roadmap banner is stale

**Claim.** `ai_docs/roadmap.md:29` reads "As of 2026-09-02 (069 in progress: W-C, W-A, W-B, W-F and
W-E landed; W-G next)" and `:42` reads "**W-G next** ... then W-D, W-H". W-G landed at `f5c7446`
("069: W-G closure rounds") and W-D landed at `f18a7d3`. The banner is the "what's next?" the cold
start chain routes every session through, and it now points two waves behind.

This is arguably W-H's sweep to make, since W-H is the last wave and is in flight. Recorded so it
is not assumed done.

**Fix.** Update the banner to name W-H as the only wave left, in whichever wave closes out.

### F8 - The spec's worked example promises one frame of black; the engine shows one frame of the shipped photo

**Claim.** The spec's frame 2 says "So frame 2 draws `edge` with an unresolved `u_df` - black, one
frame", and manual step 4 and decision 2 both repeat the "one frame from black per pass" cost. What
actually happens on that frame is the shipped default photograph.

**Evidence.** On a fresh RC copy with an `edge` pass declaring `u_df`, output set to `edge`, one
`begin_frame` + `render`:

```
frame0 (first render, no program at plan time) max: 255 mean: 111.8
uniform_values for u_df: <shaderbox.media.Image object ...>
is default image: True
```

The mechanism: on the frame the pass first compiles, `effective_graph()` ran while `program` was
still `None`, so `entry.inputs` is empty; `core.py:381`'s
`inputs.get(uniform.name, self.uniform_values.get(uniform.name))` then falls through to
`_default_uniform_value`'s seeded `Image(DEFAULT_IMAGE_FILE_PATH)`. The same fallthrough is
permanent, not one-frame, for a sampler that never resolves: a pass declaring
`uniform sampler2D u_nosuch` with no pass called `nosuch` renders the photo forever
(`u_nosuch (no such pass) -> max px: 255, mean: 111.8` after 8 frames), where D3 says an unfilled
input reads black.

**This is pre-existing, not a W-D regression** - the identical script against `f18a7d3~1` in a
throwaway worktree gives the identical numbers. But W-D is what makes it matter: before this wave a
sampler with no edge was a state the user had constructed in the gear, and now the naming rule
creates a large new population of samplers that legitimately resolve to nothing (every non-`u_<pass>`
name, every `u_<pass>` naming a pass that does not exist, every sampler on a pass that has not
compiled yet). The commit message names this exact failure as the reason for the explicit-none bind
("left unbound it fell through to the seeded default photo, which is exactly the
mis-wire-shows-a-picture failure 065 D3 exists to prevent") - the wave applied that reasoning to the
`""` case and stopped there.

**Fix.** Extend the existing binding at `document.py:538-546` to cover every declared sampler that
the effective graph does not fill, not only the stored-`""` ones - which needs the sampler-name list
the effective graph already computed, so the natural shape is for `effective_graph` (or a sibling)
to hand `render` the per-pass sampler names alongside the entries. If the maintainer would rather
keep the seeded photo as the deliberate "no texture chosen" affordance, then decision 2's "one
frame from black" and the worked example's frame 2 should say "one frame of the default image"
instead, so the spec stops asserting a behaviour the engine does not have.

## False trails

- **`make gates` is RED (exit 2) in this reviewer's shell, and it is not W-D.** The segfault is `glfw.get_video_mode` (`shaderbox/app.py:150` via `tests/conftest.py:61`) inside `tests/test_canvas_fields.py`, a module W-D does not touch; the same crash reproduces there directly. Every W-D test that does not need the `app` fixture was run and passes (8/8 in `test_default_wiring.py` with the gear test deselected; 561 passed across `test_examples_resolve.py`, `test_ui_prose_budget.py`, `test_copilot_script_tools.py`, `test_pass_graph.py`).
- The gear's `auto: df` / `auto: none` / `(none)` labels are outside the prose-budget gate's domain entirely, not merely unreadable by its AST walk: the gate scores four explicit `imgui.*` rows (`help_marker`, `set_tooltip`, `separator_text`, `text_colored`) plus derived `ui_primitives` signatures, and `imgui.combo` is in neither list. The spec's parenthetical reasoning was more elaborate than it needed to be; its conclusion is right.
- D1's word budget is a **UI**-string rule, so `_pass_views`'s `(none; reads BLACK)` is not in scope - it is prompt text on the model's channel, where the extra clause buys the model the difference between "respect this" and "fill this". Within the D1 spirit regardless: `auto: df` is two words plus a name, `(none)` is one.
- The copilot prompt sentence is genuinely STATIC-tier and cheap: 175 characters, roughly 43 tokens, inside `_SYSTEM_PROMPT` (`prompt.py:47`) which `build_prompt` renders as `Volatility.STATIC` (`:436`). It is paid once per conversation and does not move the prefix cache.
- `tests/test_pass_graph.py` and `tests/test_document_graph.py` appear in Files-touched but are untouched. Checked: neither asserts the old `with_input("")` semantics, so neither needed an edit. The only residue is that `PassGraph.without_input` has no direct unit test, only `ProjectSession` coverage.
- 065's D15 ("wiring is a closed-set selector over existing pass names", `065_pass_graph/01_spec.md:174-177`) does not contradict the new combo and needs no superseded line: the set of PASS choices is still closed, and D15's six verbs are all still present (`unwire` is now two verbs, which extends rather than contradicts it). `dev_flow.md`'s `pass_settings.py` entry is the place a reader looks for the combo's shape, and W-D updated it. 065's D3 (`01_spec.md:58-63`) uses `u_src` as an illustrative name, not a convention claim, so it does not conflict with D9 either. Nothing in `067_custom_editor.md` touches the strip or the gear.
- `assert_plan_invariants` was not weakened: it audits `plan.reads` against the graph it was handed, so feeding it the effective graph keeps the audit honest. The cross-frame concern decision 3 spends a section on is real and correctly answered - a grep confirms every planner entry point takes one graph per call, and `unresolved_inputs` has no consumer in `shaderbox/` outside `pass_graph.py` itself.

## Coverage statement

Read end-to-end: the full wave spec (1157 lines), `01_spec.md § W-D` plus D9 and D12, the complete
`f18a7d3` diff across all 25 files, `pass_graph.py`'s new region, `document.py:219-227`, `:371-377`,
`:391-395`, `:435-476` and `:490-560`, `copilot/backend.py:270-292` and `:770-800`,
`popups/pass_settings.py:1-185`, `widgets/pass_list.py:1-175`, `project_session.py:865-905`,
`tests/test_default_wiring.py` in full, the four other test diffs, `tests/test_ui_prose_budget.py:1-260`,
`conventions.md ## Code rules` and the pass-graph entry, `dev_flow.md`'s three edited module-map
entries, `065_pass_graph/01_spec.md` D3 and D15, and `067_custom_editor.md` by grep.

Executed against the real engine: the spec's worked example frame by frame (frames 0-7); the
unresolved-auto-sampler case; the same case against `f18a7d3~1` in a throwaway worktree to
establish F8 is pre-existing; the `_pass_views` ordering trace of F1 and its fix; the
stored-edge-carry-through behaviour of deviation 1; a mutation test of the D9 gate (reverted
`u_lit`, went red, restored). Run: `test_default_wiring.py` (8 of 9), `test_examples_resolve.py`,
`test_ui_prose_budget.py`, `test_copilot_script_tools.py`, `test_pass_graph.py`, `test_pass_verbs.py`
and `test_copilot_passes.py` to the extent the missing display allows.

Not verified: the eight manual in-app steps (maintainer's, and this shell has no usable display);
`test_the_gear_shows_three_distinct_states` and the three `app`-fixture strip tests, which were read
as code instead; whether F8's fix has a visual cost the maintainer would object to.
