# W-D pre-implementation review

Reviewer role per `dev_flow.md` step 4: correctness & design, verification & blast-radius.
Artifact: `70_wave_d_wiring_naming.md`. Code read at `d2ade88` via `git show`, not the working
tree (a W-F fix-up is uncommitted in `editor/`, `tabs/code.py`, `theme.py`).

## Verdict

| Dimension | Verdict |
|---|---|
| Parent coverage | **PASS** |
| Resolution-rule correctness | **PARTIAL** — F2 (the `bound` predicate reads a dict the seam does not guarantee is populated) |
| Fixpoint and invariants | **FAIL** — F1 (two planner calls in `pass_list.py` keep the raw graph; the strip washes auto-wired passes grey and orders them wrong) |
| Renames | **PARTIAL** — F3 (one `// Filled by` comment missing from the table), F4 (the cited safety net does not compile GLSL) |
| Strip tune | **PARTIAL** — F5 (the prose gate's `_UNMEASURABLE` entry for `_draw_pass_tile` goes red) |
| Test falsifiability | **PARTIAL** — F1, F5 and F6 each name a test the spec does not have |
| Docs | **PARTIAL** — F7 (`dev_flow.md`'s `pass_list.py` module-map entry becomes false) |

Seven findings. F1 is the one that must be fixed before a line is written; the rest are each a
paste-sized sentence.

---

## Findings

### F1. The strip plans the RAW graph twice, so an auto-wired pass is washed grey and mis-ordered

**Claim.** Decision 3 names five consumers of the resolution (`Document.render`, the planner via
`plan_for_output`, the gear, the strip, `_pass_views`) and routes four of them through
`effective_graph`. The strip is listed as a consumer, but decision 11 gives it only a REMOVAL (the
sublines go) and never hands it the effective graph. `widgets/pass_list.py` calls the planner twice
on `document.graph`, and both calls decide what the user sees:

```
d2ade88:shaderbox/widgets/pass_list.py:36:    order = [n for n in plan_passes(graph)[0].order if n in known]
d2ade88:shaderbox/widgets/pass_list.py:158:        set(evaluation_order(document.graph, output)) or {output}
```

Line 158 computes `live`; line 169 passes `name not in live` as `stale` to `_draw_pass_tile`, which
forwards it to `preview_cell`, which "washes toward grey (desaturated…) and the footer dims"
(`ui_primitives.py` docstring at `d2ade88`).

**Evidence — the worked example is the demonstration.** Take the spec's own § Worked example:
frame 3, pass `edge` is the output, its `u_df` resolved by name to `df`, and `graph.json` holds
`"edge": {"inputs": {}}`. The strip runs `evaluation_order(document.graph, "edge")` on the RAW
graph. `_order_for` walks `plan.reads["edge"]`, which is `set()` because the raw entry has no
inputs, so `live == {"edge"}`. `paint`, `seed`, `jfa` and `df` are all drawn every frame by the
renderer (which planned the EFFECTIVE graph) and all four tiles are washed grey with the stale
corner tick, telling the user "was drawn, is no longer live" about four passes the renderer just
drew. The wash is exactly the signal W-C's decision 7 spent a whole section making honest
(`10_wave_c_pass_verbs.md`: "after the sweep a pass outside the output chain shows a real picture
under a grey wash"), and W-D silently makes it lie.

Line 36 is the same defect on the tile ORDER: `_strip_order` documents itself as "producers left of
consumers", and on the raw graph a document wired entirely by name has no edges at all, so
`plan_passes` returns the passes in sorted-name order and `paint, seed, jfa, df, cascade, composite`
becomes `cascade, composite, df, jfa, paint, seed`. Both examples keep explicit edges after the
rename, so they are unaffected; a document a user builds by name alone gets the alphabetical strip.

**Why this is not caught by anything the spec ships.** `test_the_strip_draws_no_sublines` (§ Tests)
monkeypatches `preview_cell` and asserts on `sublines`; it captures kwargs, so `stale` is available
to it, but nothing asserts on it. No test in the spec drives the strip on a name-wired document.

**Fix (paste).** In decision 11, after the sublines bullet: "`draw` and `_strip_order` both plan the
EFFECTIVE graph, not `document.graph`: `resolved = document.effective_graph()` once at the top of
`draw`, passed to `_strip_order(document.passes, resolved)` and to
`evaluation_order(resolved, output)`. Without it an auto-wired ancestor is outside `live` and takes
the stale wash while the renderer is drawing it every frame, and a document wired entirely by name
loses its topological tile order to sorted-name order." Add to § Tests a case on
`test_the_strip_draws_no_sublines`'s rig: a two-pass document `a`, `b` where `b` declares `u_a` with
no stored edge, output `b`, asserting `_draw_pass_tile` is reached with `stale=False` for `a`.
Falsifier: plan the raw graph and `a` comes back `stale=True`.

---

### F2. `_is_user_bound` reads `uniform_values`, which a freshly compiled pass has not yet filled

**Claim.** Decision 4's exclusion is keyed on `render_pass.uniform_values.get(uniform)` being a
non-default `MediaWithTexture`. Decision 2 gathers samplers from `render_pass.program` directly. The
two are not populated by the same event: `Pass.compile()` sets `self.program` and does NOT call
`seed_uniform_values`.

**Evidence.** `core.py::compile` at `d2ade88` ends by assigning `self.program`, `self.vbo`,
`self.vao` and writing the engine tables — no seed call. Seeding happens in exactly three places
(`git grep seed_uniform_values`): `core.py:239` inside `get_active_uniforms` (only on the branch
where IT triggered the compile), `core.py:377` at the top of `Pass.render`, and `ui_models.py:446`
at save. So after `project_session.add_pass`'s bare `render_pass.compile()` (`project_session.py:779`),
`program` is set and `uniform_values` is `{}`.

**The consequence is benign, and that is the point — it is benign by accident.** `bound` comes back
empty, so the auto edge is computed and the media exclusion does not fire on that pass until
something seeds it. On the add-pass path there is no media to protect, so nothing breaks. But the
spec asserts the exclusion holds wherever the resolution runs, and § Manual verification step 7
("A bound texture is not stolen") exercises exactly the window where it does not: the user adds a
pass called `image` to the Media Input example. There the OUTPUT pass `main` has been rendered and
seeded, so its `u_image` is protected — the ordering works out. The design is correct on every path
that exists today and rests on a coincidence the spec does not name.

**Fix (paste).** In decision 4, after the `_is_user_bound` snippet: "`compile()` does not seed —
`seed_uniform_values` runs at the top of `Pass.render`, inside `get_active_uniforms`'s own compile
branch, and at save (`core.py`). A pass compiled but never rendered therefore has an empty
`uniform_values` and contributes an empty `bound`, so its samplers auto-wire. That is correct
(nothing is bound yet, so nothing can be stolen) and it is why `_sampler_names` reads the program
rather than `uniform_values.keys()`: the two are populated by different events and only the program
is set by `compile()`." Add the same sentence as a comment-free note in decision 2's prose.

---

### F3. One `// Filled by` comment is missing from decision 9's list

**Claim.** Decision 9's "The two comments that become wrong" enumerates five comments to delete:
`blur.frag.glsl:9`, `trail.frag.glsl:10`, and three on `composite.frag.glsl:8-10`. It misses a
sixth, on `bright.frag.glsl`.

**Evidence.**

```
$ grep -rn "illed by" 1c4f8a20*/passes/*.glsl
bright.frag.glsl:9:// Filled by `scene` -- see the pass list's inputs, or graph.json. An unfilled input reads black.
composite.frag.glsl:8:uniform sampler2D u_lit;    // filled by `scene`
composite.frag.glsl:9:uniform sampler2D u_glow;   // filled by `blur`
composite.frag.glsl:10:uniform sampler2D u_trail; // filled by `trail`
blur.frag.glsl:9:uniform sampler2D u_src;  // filled by `bright`
trail.frag.glsl:10:uniform sampler2D u_src;   // filled by `scene`
trail.frag.glsl:11:uniform sampler2D u_prev;  // filled by `trail` -- itself, i.e. the previous frame
```

`bright.frag.glsl:9` is a whole-line comment above the decl (capital F, and it carries a second
clause the others do not), which is why a grep for the inline pattern misses it. Under D9 it becomes
`u_scene` filled by `scene` — the uniform name says it, so it goes the same way as the other five.
Its trailing "An unfilled input reads black" is engine prose that the Help panel and the copilot
prompt block both now carry, so it goes with the line.

**Fix (paste).** In decision 9's comments paragraph, change "five" to "six" and add
"`bright.frag.glsl:9` (`// Filled by \`scene\` -- …`, a whole-line comment above the decl, capital F,
which an inline-pattern grep misses)".

---

### F4. `test_examples_resolve_clean` cannot catch a half-renamed shader; the pair that can is not the pair the spec names

**Claim.** § Tests' last entry says `test_examples_resolve_clean` "catches a shader whose declaration
was renamed but whose read was not (the resolver walks the flattened source)". It does not. It runs
`resolve_usage` — the `SB_*` INCLUDE resolver — and never invokes GLSL.

**Evidence.** `tests/test_examples_resolve.py:29-36` at `d2ade88`:

```python
def test_examples_resolve_clean() -> None:
    index = ShaderLibIndex.build(SHADER_LIB_SEED_DIR)
    shaders = sorted(DOCUMENT_EXAMPLES_DIR.glob("*/passes/*.frag.glsl"))
    for path in shaders:
        _, _, _, errors = resolve_usage(ShaderSource.load(path), index)
        assert not errors, ...
```

`resolve_usage` reports unresolved `SB_*` calls. `texture(u_src, …)` against a decl renamed to
`u_scene` is not an `SB_` name; the resolver is indifferent to it and the test passes. The failure
would surface only at `gl.program(...)` link time.

**What actually covers it.** Two tests, and between them the coverage is complete but uneven:

- `tests/test_radiance_cascades_example.py::test_every_pass_compiles_and_the_graph_is_clean` — real
  compile of all six RC passes plus `graph_errors == []`. Covers RC fully, including the
  `graph.json`-not-renamed half (a stale edge lands in `unresolved_inputs`, not `graph_errors`, so
  the second assertion does NOT catch that; the pixel tests at `:71` and `:81` do).
- `tests/test_lazy_compile.py::test_every_pass_renders_once_within_n_frames` — loads Bloom and
  drives the sweep over `len(document.passes)` frames, so every Bloom pass reaches
  `Document.render` and compiles. That is Bloom's only whole-document compile coverage.

The one the spec might reach for instead does not help:
`tests/test_example_library.py::test_shipped_examples_read_clean_without_joining_working_set` asserts
`len(v.errors) == 0` per example, but `read_shaders` compiles `document.render_pass` only
(`copilot/backend.py:632`) — the OUTPUT pass. Bloom's `bright`, `blur`, `trail` and `scene` are never
touched by it.

**A real gap this exposes.** `test_every_pass_renders_once_within_n_frames` asserts `first_render_done`
and the no-double-election property. It does NOT assert `compile_unit.errors == []`, and the stamps
are written on ATTEMPT (W-C decision 7, deliberately), so a Bloom pass that fails to compile still
gets stamped and the test stays green. So Bloom has no assertion that its non-output passes compile.
A half-renamed `bright.frag.glsl` goes red in nothing.

**Fix (paste).** Replace § Tests' claim with: "`test_examples_resolve_clean` resolves `SB_*` includes
and is indifferent to a uniform name, so it does NOT catch a half-rename; it is run as a regression
check, not as this rename's gate. RC is covered by
`test_radiance_cascades_example.py::test_every_pass_compiles_and_the_graph_is_clean` (a real compile
of all six) plus its two pixel assertions (which catch a `graph.json` left unrenamed, since the pass
then reads black). Bloom has no equivalent: `test_lazy_compile.py::test_every_pass_renders_once_within_n_frames`
drives every Bloom pass through `Document.render` but asserts only the stamps, and the stamps are
written on ATTEMPT. So this wave adds one assertion to `tests/test_default_wiring.py`:
`test_every_multi_pass_example_compiles_every_pass` — load RC and Bloom, render `len(passes)` sweep
frames each, assert `compile_unit.errors == []` for every pass. Falsifier: rename a decl and leave
its `texture()` read, and it goes red naming the pass and the linker message."

---

### F5. Dropping `sublines` at the call site turns the prose gate's `_UNMEASURABLE` entry into a failure

**Claim.** Decision 11 keeps `preview_cell`'s `sublines` parameter (correctly — decision 17's
reasoning about the derived domain holds) but removes the ARGUMENT at `pass_list.py:117`. That is the
half decision 11 does not trace, and it makes `tests/test_ui_prose_budget.py` go red.

**Evidence.** The gate has a rot check that fires on an entry naming no live site:

```python
@pytest.mark.parametrize("key", sorted(_UNMEASURABLE), ids=lambda k: f"{k[0]}::{k[1]}")
def test_every_unmeasurable_entry_still_names_a_real_site(key: tuple[str, str]) -> None:
    assert key in {site.key for site in _UNREADABLE}, (
        f"{key} no longer names an unmeasurable site; delete the entry."
    )
```

and the entry:

```python
("shaderbox/widgets/pass_list.py", "_draw_pass_tile"): "the pass's name and wiring",
```

`_draw_pass_tile` reaches the collector through exactly two `preview_cell` arguments
(`git show d2ade88:shaderbox/widgets/pass_list.py | grep -n "footer=\|sublines="` → `116:footer=name`,
`117:sublines=sublines`). Both are `ast.Name` nodes. `_resolve` handles them differently:

- `footer=name` — `name` is a function PARAMETER, not an assignment in the body, so `candidates` is
  empty and `_resolve` returns None → `UNMEASURABLE`.
- `sublines=sublines` — the body does `sublines.append("has compile errors")`, which is an
  `ast.Call` on an attribute, not an `ast.AugAssign`, so the AugAssign guard does not fire; the
  assignment is a `ListComp`, which is not in `_resolve`'s `(Constant, JoinedStr, IfExp)` filter, so
  `candidates` is again empty → `UNMEASURABLE`.

Both are unreadable today, so the entry names a real site and the rot check passes. Remove the
`sublines=` argument and `footer=name` remains, still `UNMEASURABLE`, so the entry SURVIVES and the
rot check still passes. The gate stays green. Decision 11 is therefore correct on the outcome and
wrong on the reasoning it gives: it argues from "removing the PARAMETER would change the derived
rows", which is true but is not the risk, and never checks the allowlist entry the removal actually
touches.

**The reason to state it anyway.** The entry's written reason becomes half-false — "the pass's name
and wiring" when there is no wiring left — and this repo's rule is that an allowlist entry carries a
reason that stays true. A reviewer of 070 (which puts the wiring back, in the graph view) will read
a reason that describes a state the code left.

**Fix (paste).** Add to decision 11: "`tests/test_ui_prose_budget.py`'s `_UNMEASURABLE` entry
`(\"shaderbox/widgets/pass_list.py\", \"_draw_pass_tile\")` survives the removal — `footer=name`
still reaches the collector as an unresolvable `ast.Name` (a function parameter has no assignment in
the body), so `test_every_unmeasurable_entry_still_names_a_real_site` stays green. Its reason is
updated from \"the pass's name and wiring\" to \"the pass's own name\" in the same commit, since the
wiring half is what this wave removes."

---

### F6. `unwire_pass_input` is specified without its validation rule, and the gear's stale branch can call `wire_pass_input` with a name it just rejected

**Claim, part one.** Decision 5 gives `unwire_pass_input` a docstring and says it "validates the
document and the consumer the same way". `wire_pass_input` at `d2ade88:project_session.py:847-865`
validates three things, and the third is the one that changes meaning:

```python
        if producer and producer not in document.passes:
            return f"no such pass '{producer}'"
```

The `if producer` guard is what lets `""` through today. Under decision 5 `""` is a stored value
rather than a deletion, and this line still lets it through unchanged — correct, and worth stating,
because a reader reworking `wire_pass_input`'s docstring (which decision 5 asks for) is one line
above it.

**Claim, part two — the real gap.** Decision 6's index logic ends:

```python
        else:
            index = choices.index(stored) if stored in choices else 1
```

with the prose "Picking any item then rewrites the key to something valid, so the stale name cannot
survive a visit to the gear." That is only true if the user picks. A stale name displayed as
`(none)` at index 1 and never touched stays stale, which is the correct behaviour and is what open
question 2 decides. But `choices` in the new design is `[auto_label, _UNWIRED, *sorted(passes)]`
(three kinds), while today it is `[_UNWIRED, *sorted(passes)]`. The spec gives the index-READ logic
and the store-per-position table but never says how `choices` is built, and the two must agree
positionally or every selection writes the wrong pass. The `stored in choices` test in the last
branch is also now wrong: `choices` contains the auto label and `(none)` as well as pass names, so a
pass literally named `auto: df` — impossible, `_PASS_NAME_RE` rejects the colon and the space
(`project_session.py:113`: `^[A-Za-z_][A-Za-z0-9_]*$`) — cannot collide, but a pass named exactly
whatever `_UNWIRED` holds could. Worth one line.

**Fix (paste).** In decision 6, before the index snippet: "`choices = [auto_label, _UNWIRED,
*sorted(document.passes)]`, with `auto_label = f\"auto: {auto or 'none'}\"` where `auto` is
`effective_inputs(entry_without_this_key, [uniform], names, name, bound).get(uniform, \"\")`. The
selection writes by POSITION against this list: 0 → `unwire_pass_input`, 1 →
`wire_pass_input(..., \"\")`, else `wire_pass_input(..., choices[picked])`. A pass name can never
collide with either of the first two labels: `_PASS_NAME_RE` (`project_session.py`) admits only
`[A-Za-z_][A-Za-z0-9_]*`, which excludes the colon, the space and the parentheses." And in
decision 5: "`wire_pass_input`'s `if producer and producer not in document.passes` guard is
unchanged and is what admits `\"\"` — under the new meaning it admits an explicit none rather than a
deletion request, which is exactly the same line doing a different job."

---

### F7. `dev_flow.md`'s `pass_list.py` module-map entry becomes false

**Claim.** § Files touched names `ai_docs/conventions.md` for the D9 bullet and lists nothing under
`dev_flow.md`. Two of the three module-map entries the review brief asks about stay true; one does
not.

**Evidence.** `ai_docs/dev_flow.md:204-208`:

> **`widgets/pass_list.py`** — the Document tab's pass strip (feature 065): one live `preview_cell`
> thumbnail per pass in `plan_passes` topological order, blind to the output choice.

Under F1's fix the order comes from `plan_passes` of the EFFECTIVE graph, not `document.graph`. The
sentence is not wrong about the function, but it is the sentence a reader consults to learn which
graph the strip plans, and after W-D that is the question with a new answer. The other two:

- `pass_graph.py` (`:213-221`) lists the edit verbs as "`with_passes` / `with_input` / `with_target` /
  `with_output`". `without_input` joins them, so the list is incomplete but not false.
- `pass_settings.py` (`:209-212`) says "one pass's input wiring (closed-set combos over the
  document's own pass names)". After decision 6 the set is the pass names PLUS two synthetic items;
  "closed-set" survives, the parenthetical narrows.

**Fix (paste).** Add to § Files touched: "**`ai_docs/dev_flow.md`** — the module map's
`widgets/pass_list.py` entry says the strip orders by `plan_passes` and washes off-plan tiles; both
now read the effective graph, so the entry says which. `pass_graph.py`'s verb list gains
`without_input`; `pass_settings.py`'s 'closed-set combos over the document's own pass names' gains
the two synthetic items."

---

## Coverage the brief asked for, answered

**`effective_inputs` over the full matrix.** Traced by hand over all nine cells plus `u_prev`; the
spec's § Tests table matches the design's own branches on every row, and the branches are total (an
absent key, `""`, and a non-empty string exhaust `dict.get`'s codomain). Two cases the spec leaves
open and both are decided correctly in prose elsewhere: a sampler not named `u_*` at all
(`_auto_source` strips the `u_` prefix — on a sampler called `tex` there is no prefix to strip, and
the spec does not say whether it yields `tex` or nothing; either is safe since a pass named `tex`
would then auto-wire, which is arguably right, but it is undecided — see the note below), and a pass
named `prev` beside a `u_prev` sampler, which decision 1 decides explicitly and correctly.

*The one genuinely undecided cell.* `_auto_source("tex", consumer)` — a sampler whose name lacks the
`u_` prefix. D9 says input uniforms ARE named `u_<pass>`, so the rule has nothing to say; the safe
reading is "no prefix, no auto edge". State it: "`_auto_source` returns nothing for a name without
the `u_` prefix — D9's rule is about `u_<pass>` names and a bare `tex` is outside it."

**GL-freedom of the signature.** Confirmed sound. `effective_inputs(entry, samplers, passes,
consumer, bound)` takes `bound` as a `Collection[str]` computed by the caller, so `pass_graph.py`
imports nothing from `core.py` or `media.py` — decision 1 says so and it is the right split.
Answering the brief's sub-question directly: **`media.py` DOES import GL** (`import moderngl` at
`d2ade88:shaderbox/media.py:11`), so had `effective_inputs` needed `is_default_image` itself,
`pass_graph.py` would have lost the GL-free property its module docstring pins ("everything here is
GL-free pure data … importable from anywhere without a cycle"). The `bound` parameter is what avoids
that, and the spec's reasoning for it is correct.

**The five consumers.** `git grep "\.inputs" d2ade88 -- shaderbox/` returns eleven sites. Nine are
correctly handled: `document.py:476` (decision 3), `copilot/backend.py:743` (decision 7),
`pass_settings.py:143` (decision 6), `pass_list.py:103` (deleted by decision 11), `pass_graph.py:179`
(inside `with_input`), `pass_graph.py:269` / `:349` / `:367` (the planner and the invariants, which
by design operate on whatever graph they are handed), and `copilot/prompt.py:330-331` (reads
`pass_view.inputs`, a list of formatted strings, not a `PassEntry`). Two are correct as-is:
`project_session.py:130` and `:143` (the delete and rename rewrites, which operate on STORED edges
and must, since an auto edge is not stored). Decision 5's audit of those two is right on both counts.
**The planner calls in `pass_list.py` are the miss** — F1.

**The `""` change.** Every `with_input` caller enumerated. In `shaderbox/`: exactly one
(`pass_settings.py:149`, via `wire_pass_input`), which decision 6 rewrites. In `tests/`: six call
sites, of which two pass `""` — `test_copilot_passes.py:148` and `test_pass_verbs.py:177` — and the
spec rewrites both (§ Tests, the `test_unwiring_*` and `test_copilot_passes` entries). No other
caller expects deletion. Confirmed by inspection that neither example nor either `projects/dev`
document needs a `""` value: RC and Bloom wire every sampler they declare, and both dev documents are
single-pass with `"inputs": {}` and one sampler (`u_video`) that names no pass. The spec's premise 14
is exactly right, verified independently.

**The compiled-passes fixpoint.** The mechanism is sound and 066 D1 is preserved: `_sampler_names`
reads `render_pass.program` and returns `[]` when it is None, so nothing compiles. Traced over frames
0..N of a six-pass document under W-C's sweep (`ui.py:294-311`, which landed at `d2ade88`):

- Frame 0, fresh load. Every `program is None`, so `effective_graph()` returns the raw graph.
  `plan_for_output` orders the raw output chain; the sweep elects one pending pass and draws its
  raw chain. Compiles happen inside `Pass.render`.
- Frames 1..5. Each frame more passes have programs, so more auto edges appear and the order can
  GROW. The brief asks whether that breaks `assert_plan_invariants` or double-draws. **It does
  neither, and the reason is structural rather than lucky:** the invariants are asserted on ONE
  plan against the graph THAT plan was built from, inside `plan_for_output`, within a single call.
  There is no cross-frame invariant to violate. Draw-once is per-order (`len(plan.order) ==
  len(set(plan.order))`), also per-call. And the sweep's skip is keyed on `drawn_frame == self._frame`,
  which is per-frame, so a pass whose auto edge appears mid-sequence is drawn once in the frame the
  edge appears and skipped by the sweep in that same frame. The spec's claim holds. Worth one added
  sentence saying WHY (the invariants are per-call, so a between-frame order change is outside their
  domain by construction), because the spec asserts the conclusion without the reason.
- The hot-reload seam. Traced: `watch.py::_reload_pass_if_changed` → `release_program(text)` →
  `invalidate()`, which sets `program = None` AND (per W-C, `core.py` at `d2ade88`) clears
  `first_render_done`, re-admitting the pass to the sweep. So its auto edges DO vanish for a frame
  and its dependents read black for that frame. The spec states this (§ Worked example frames 1-2,
  and decision 2's "one frame from black per pass"), correctly, and the worked example's frame
  accounting matches the code.

**The renames.** Every one of the eighteen `graph.json` keys and every decl/read line in the table
was opened and matched. All correct, line numbers included: RC `seed` 17/20, `cascade` 25/44,
`composite` `u_light` 13/18 and `u_scene` 14/22; Bloom `bright` 10/17, `blur` 9/16+23, `trail`
10/20, `composite` `u_lit` 8/18 and `u_glow` 9/19. The third prefix trap is confirmed:
`77a84d27…/passes/paint.frag.glsl:20` declares `uniform float u_light_radius = 0.035;` and reads it
at `:35`, inside one of the two multi-pass examples, so file-scoping the replace is genuinely
insufficient and the spec's correction 13 is right. The other three traps confirmed at their cited
lines (`8d454b7b…:56-60` and `:460-461`, `0b0d16bb…:12-13` and `:164`/`:167`, `1c4f8a20…:13`).
Both `document.json` files carry `"uniforms": {}`, so the rename strands nothing — premise verified
independently, not taken from the spec. No single-pass example is touched. Simulated the new D9 gate
over all seven shipped examples: sixteen edges, nine fail today, zero fail after the table's
renames — so `test_every_example_input_uniform_names_its_source` goes red today and green after,
exactly as § Tests claims.

**The strip tune.** `SIZE.PASS_THUMB` is 112 in `theme.py`, confirmed. `step = SIZE.PASS_THUMB +
SPACE.MD` and `same_line(spacing=SPACE.MD)` agree at `pass_list.py:163` and `:171`, and neither reads
the card height — the spec's "unchanged" is right. The `imgui.dummy((0, SPACE.SM))` at `:172` sits
before `add pass`, as described. The `sublines`-in-signature-vs-call-site distinction is handled in
F5.

**The four closed open questions.** Agree with all four defaults. Q1 (`auto: <x>` names the pass):
agree — the distinction between "the name found a pass" and "the name found nothing" is the one the
combo exists to make visible, and both labels are inside § 2's budget for a control's own item text.
Q2 (no fourth item for a stale name): agree, and the evidence is stronger than the spec claims —
`_graph_without` and `_graph_renamed` rewrite every edge (`project_session.py:130`, `:143`), so the
state is reachable only by hand-editing, which `PassGraph`'s own docstring forbids ("The user edits
it through the panel, never by hand"). Q3 (do not mark auto edges for the copilot): agree, and it is
the right call under the skill's home rule — the naming rule is pre-action and belongs in STATIC,
which is where decision 12 puts it. Q4 (one bullet in the pass-graph entry): agree; the resolution
rule is a property of the same mechanism the entry already describes.

**Docs.** The Help paragraph is documentation body prose, which § 2 of the imgui skill exempts from
the word budget explicitly ("Anything longer than the budget is documentation and goes to the Help
panel or the tutorial") — the spec cites this correctly, and `help_content.py`'s `pass_settings`
section is body-only with no new key, so `tests/test_help_content.py` is unaffected. The copilot
sentence belongs in `_SYSTEM_PROMPT` at `Volatility.STATIC` (`copilot/prompt.py:24-28` defines the
tiers; the pass block's existing "an unfilled one reads BLACK, it is not an error" is at
`prompt.py:64-65`, inside `_SYSTEM_PROMPT`), and the skill's rule "a rule's HOME follows WHEN it
fires: pre-action stays STATIC" puts it exactly there. Decision 12's reasoning matches. The
`conventions.md` bullet placement is fine. `dev_flow.md` is F7.

**Findings 19 and 37 against the maintainer's verbatim words.** #19 asked for the graph view OR "at
least tune the current visuals, it is awful", and named the specific defect: "'u_prev <- …' gets
cut" and "the `<-` reads as a cheap workaround". Decision 11 removes both the truncation and the
arrow, which is option B as the maintainer ranked it, with A deferred to 070 by the maintainer's own
call in the parent's Out of scope. The complaint cannot recur while the sublines are gone. #37 asked
for one rule applied to both examples and the tutorial, and named the pay-off ("the gear can
DEFAULT-wire by name"). Decision 9 lands the rule and the rename; decisions 1-6 land the default
wiring, and go further than the finding asked (resolution at render time rather than a pre-filled
gear, which is what makes the copilot and hot-reload paths correct for free). The finding named a
third home for the rule (the add-pass stub's comment); § Out of scope declines it with a reason that
is checkable and correct — `PASS_STUB` (`project_session.py:102-110`) is six lines declaring no
sampler, so a comment about sampler naming there would document a uniform the stub does not have.
The complaint could recur only via a new example reintroducing a role name, which the new D9 gate
test blocks.

---

## False trails

- `PassEntry` frozen (`model_config = ConfigDict(frozen=True)`) — `model_copy(update=...)` works on
  frozen pydantic models, so `effective_graph`'s per-entry copy is fine.
- `with_passes` resetting `version` / `output` / `layout` — it passes only `_PASSES_FIELD` into
  `model_copy` when `output`/`layout` are None, so the effective graph keeps all three.
- `plan_passes` treating a source `""` as a cycle — it lands in `missing`, not `deps`, and
  `PassGraph._reject_unnamed_pass` guarantees `"" not in known`. Decision 5's audit is right.
- `assert_plan_invariants` going vacuous on the effective graph — its `expected` set is recomputed
  from the graph it is handed, so handing it the effective graph audits the effective edges. The
  spec's claim is correct.
- `_pass_views` calling `effective_graph()` once per pass inside the loop (decision 7's snippet
  reads `document.effective_graph()` under `for name in sorted(document.passes)`) — O(n²) on a
  handful of passes on a copilot turn that costs seconds; not worth a finding, though hoisting it
  above the loop is free.
- The examples popup double-drawing under the sweep — `ui.py:311`'s branch calls `render()` with no
  `target`, so the skip cannot fire; W-C already reasoned this and it still holds.
- `SPACE.SM` between wrapped rows changing — it is `style.item_spacing.y`, set in `theme.py`, and
  nothing in decision 11 touches it.

## Coverage statement

Read at `d2ade88`: `pass_graph.py` whole, `document.py` (`__init__`, `begin_frame`, `render`,
`graph_errors`), `core.py` (`__init__`, `get_active_uniforms`, `compile`, `seed_uniform_values`,
`_default_uniform_value`, `render`), `media.py` (imports, `is_default_image`), `project_session.py`
(`PASS_STUB`, `_PASS_NAME_RE`, `_graph_without`, `_graph_renamed`, `add_pass`, `delete_pass`,
`wire_pass_input`), `popups/pass_settings.py::_draw_inputs` + `_sampler_names`,
`widgets/pass_list.py` whole, `ui_primitives.py::preview_cell`, `copilot/backend.py` (`_pass_views`,
`read_shaders`, `_sampler_uniform_names`), `copilot/prompt.py` (tiers + pass block),
`help_content.py::pass_settings`, `ui.py` (the document-render block and the sweep),
`tests/test_ui_prose_budget.py` whole, `tests/test_examples_resolve.py` whole,
`tests/test_example_library.py`, `tests/test_lazy_compile.py` (the sweep tests),
`tests/test_radiance_cascades_example.py` (test names). Both multi-pass examples' `graph.json`,
every pass shader in both, all seven examples' `document.json`, both `projects/dev` documents and
their shaders, the two prefix-trap single-pass examples. Anchors: `01_spec.md` (§ W-D whole, locked
decisions, review history, out of scope), `00_findings.md` rows 19 and 37 verbatim,
`10_wave_c_pass_verbs.md` decision 7, `.claude/skills/imgui-ui/SKILL.md` §§ 2-3,
`.claude/skills/copilot-llm-agent-design/SKILL.md` (the tier/home rules), `ai_docs/dev_flow.md`
module map, `CLAUDE.md`.

Two claims verified by running code rather than reading: the D9 gate's before/after over all seven
examples' `graph.json` (nine failures today, zero after the table), and the `"illed by"` comment
census in both examples. Not verified by execution: the imgui-frame behaviour of the three-state
combo (no rig run), and the pixel outcomes of the manual steps.

---

# Round 2 (closure)

Narrow round: for each of F1..F7, cite the new text and rule CLOSED / NOT CLOSED. Both new strip
tests were checked by replicating `plan_passes` / `_order_for` / `_strip_order` over each test's
own fixture and running the named mutation; decision 6's `choices` was checked by enumerating
every index of the read and the write. Code re-read at `d2ade88`. Preferences are out of scope
this round.

## Per-finding verdicts

### F1 — CLOSED

Decision 11 is retitled "The strip tune, **and the strip plans the effective graph**" and carries a
bolded paragraph: "`resolved = document.effective_graph()` once at the top of `draw`, passed to
`_strip_order(document.passes, resolved)` and to `evaluation_order(resolved, output)`", with both
halves of the defect written out — the stale wash at `:158` and the tile order at `:36`. The strip
is now stated as "a real consumer of `effective_inputs` rather than a removal", which is the
sentence decision 3's five-consumer list needed. One detail the drafter added that I had not asked
for and that is right: "`_strip_order`'s signature is unchanged (it already takes a `graph`
parameter), so `tests/test_pass_verbs.py`'s import of it stays valid; only the argument the caller
passes moves."

**Both named mutations confirmed red**, by replicating the planner over each fixture.

`test_an_auto_wired_ancestor_is_not_washed_stale` — fixture `a` (no inputs), `b` declaring `u_a`
with no stored edge, output `b`:

```
effective  live=['a', 'b']  -> stale(a) = False     <- the assertion
raw        live=['b']       -> stale(a) = True      <- the falsifier
```

`test_the_strip_orders_a_name_wired_document_topologically` — fixture `zeta` / `alpha` (`u_zeta`) /
`mid` (`u_alpha`), nothing wired:

```
effective -> ['zeta', 'alpha', 'mid']    <- the assertion, matches the spec verbatim
raw       -> ['alpha', 'mid', 'zeta']    <- the falsifier, matches the spec verbatim
disagree at every position: True         <- the spec's own decidability claim, verified
```

The spec's stated reason for the name choice ("with names like `a`, `b`, `c` the two orders
coincide and the test would pass under the bug") is correct: the fixture is chosen so alphabetical
and topological order share no position, so the assertion cannot pass by coincidence.

### F2 — CLOSED

Two sites, as asked. Decision 2 gains "**Why the program and not `uniform_values.keys()`**": "The
two are populated by different events, and only the program is set by `compile()`… `Pass.compile()`
assigns `program`, `vbo`, `vao` and writes the engine tables, and does NOT seed". It names the
producer of the window precisely — "which is exactly what `project_session.add_pass`'s bare
`render_pass.compile()` produces" — which is the line I traced. Decision 4 gains "**The seeding
window, and why it is correct rather than merely harmless**", which is the framing the finding
asked for: "nothing is bound yet, so there is nothing to steal, and the first render seeds before
it binds", and ties manual step 7 to the ordering that makes it valid.

### F3 — CLOSED

Decision 9 now reads "**The six comments that become wrong**" (was five), lists
`bright.frag.glsl:9` first, and states the census method: "the reason the census here was taken
with `grep -rn \"illed by\"` rather than a pattern anchored to the declaration line". It also
records the two properties that made the line invisible to the original pass — the capital F and
the second clause. Re-verified: `grep -rn "illed by"` over both examples returns exactly seven
lines, six that go and `trail.frag.glsl:11`'s `u_prev` comment that stays.

### F4 — CLOSED

§ Tests carries a bolded correction, "**What the rename's safety net actually is, corrected**",
which states plainly that the earlier claim "**does not**" hold and gives the reason
(`resolve_usage` is the `SB_*` include resolver; `texture(u_src, …)` is not an `SB_` name). The
uneven coverage is written out per example, including the sub-point that RC's `graph_errors`
assertion does not catch an unrenamed `graph.json` (a stale edge lands in `unresolved_inputs`) and
the pixel tests do; and that `test_shipped_examples_read_clean_without_joining_working_set` does
not help because `read_shaders` compiles `document.render_pass` only.
`test_every_multi_pass_example_compiles_every_pass` closes the Bloom gap and asserts on
`compile_unit.errors`, which is the assertion `test_every_pass_renders_once_within_n_frames` lacks
because W-C stamps on ATTEMPT. Premise 15 was updated in the same pass to stop citing the resolve
test as part of the answer.

### F5 — CLOSED

Decision 11 traces the allowlist entry through the collector — "`footer=name` and
`sublines=sublines`, both unresolvable `ast.Name` nodes… Dropping the `sublines=` argument leaves
`footer=name` - a function PARAMETER, so there is no assignment in the body for `_resolve` to read"
— reaches the right conclusion (the gate stays green), and then does the part that was missing:
the reason string is updated to "the pass's own name" **in the same commit**, and
`tests/test_ui_prose_budget.py` is added to § Files touched for that edit. The drafter also
recorded the superseded reasoning rather than silently swapping it, both in decision 11 and in
premise 17 ("the derived rows were never the risk, since they come from the signature and do not
move").

### F6 — CLOSED

Decision 6 now gives the construction:

```python
choices = [f"auto: {auto or 'none'}", _UNWIRED, *sorted(document.passes)]
```

with `auto` computed from `effective_inputs` on the entry with this key removed, and the write
stated as by-position against exactly that list.

**Read and write agree on every index**, verified by enumerating both directions over a seven-pass
document, for `auto="df"` and `auto=""`:

| stored | index | label at index | write | round trip |
|---|---|---|---|---|
| absent | 0 | `auto: df` | `unwire_pass_input` | absent — OK |
| `""` | 1 | `(none)` | `wire(…, "")` | `""` — OK |
| `"df"` | 4 | `df` | `wire(…, "df")` | `"df"` — OK |
| `"jfa"` | 6 | `jfa` | `wire(…, "jfa")` | `"jfa"` — OK |
| `"ghost"` | 1 | `(none)` | `wire(…, "")` | `""` — by design |

and on the write side every index 2..N writes the pass whose label sits at that index (seven of
seven). The `ghost` row is the only non-identity and it is the decided behaviour of open question 2;
it fires only on an actual pick, since imgui returns `changed` only then, so a displayed stale name
is not rewritten by display alone. `_UNWIRED` is confirmed to be the literal `"(none)"`
(`pass_settings.py:31`), so the table's position 1 and the code's position 1 are the same string.
The collision argument is verified against the real regex: `_PASS_NAME_RE` is
`^[A-Za-z_][A-Za-z0-9_]*$` (`project_session.py:113`), which admits neither the colon, the space,
nor the parenthesis, so no pass name can shadow either synthetic label and `stored in choices`
cannot hit falsely. Decision 5's added sentence about `wire_pass_input`'s unchanged
`if producer and producer not in document.passes` guard is present and correct.

### F7 — CLOSED

§ Files touched gains an `ai_docs/dev_flow.md` bullet naming all three edits, with the
`pass_list.py` one carrying the reason ("the entry is the sentence a reader consults to learn which
graph the strip plans, so it says which"), plus a separate `tests/test_ui_prose_budget.py` bullet
for F5's reason string. The `pass_settings.py` edit correctly keeps "closed-set" and narrows only
the parenthetical.

## The two rulings

**No `u_` prefix, no auto edge — sound.** Decision 1: "`_auto_source` returns nothing for a bare
`tex` or `noise`: D9's rule is about `u_<pass>` names, so a name outside that shape is outside the
rule and the only honest answer is to leave it undecided." This is the conservative branch and the
one a user can predict from the rule they were taught, which is the stated argument and the right
one. It also keeps the function total over the sampler-name domain with no third outcome.

**The per-call invariant reasoning — correct as written.** Decision 3's new paragraph states it
exactly: `assert_plan_invariants` runs inside `plan_for_output` and recomputes `expected` from the
graph it was handed; the draw-once check is `len(plan.order) == len(set(plan.order))`, also
per-call; draw-once across a frame is W-C's `drawn_frame == self._frame` skip. Verified against
`pass_graph.py:340-372` and `document.py:440-455` at `d2ade88`. The framing the drafter added —
"the conclusion without the reason invites a future reader to add a cross-frame guard that has
nothing to guard" — is why the paragraph earns its place.

## False trails this round

None raised. Two things I checked and did not report: `_pass_views` still calls
`document.effective_graph()` inside its per-pass loop (a preference, and it was named a false trail
in round 1), and decision 11's `PassEntry`-import bullet still says "check at edit time" rather
than deciding, which is correct for a bullet whose answer depends on the final diff.

## Round 2 verdict

**PASS.** Seven of seven findings CLOSED, both rulings sound. The FAIL dimension from round 1
(fixpoint and invariants) is closed by decision 11's effective-graph paragraph plus two tests whose
mutations I confirmed red by replicating the planner. Every dimension now passes: parent coverage,
resolution-rule correctness, fixpoint and invariants, renames, strip tune, test falsifiability,
docs. No finding is escalated and nothing new was opened.

**Method note.** Closure was judged against the spec's new text and against `d2ade88`, not against
round 1's report. The two strip tests and the combo index table were re-derived from the code
rather than read back from the spec's own claims about them.
