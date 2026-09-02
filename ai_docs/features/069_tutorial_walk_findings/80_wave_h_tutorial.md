# 069 W-H: the tutorial rewrite

Implementation spec for wave W-H of feature 069, the last one. The parent spec (`01_spec.md § W-H`)
fixes the shape; this file fixes the code and the prose. Locked decisions **D8** and **D9** apply as
CONSTRAINTS and are not re-opened: pass steps follow one template, pass cards and pass code are
GENERATED from the shipped example by `build_tutorial.py`, and an input uniform is named after the
pass it reads.

W-H is the verification of everything above it: the maintainer walks the tutorial again, in the app,
and every step must produce its picture with no control that does not exist. So this spec's job is
to leave the implementer no prose to invent. Every sentence that changes is in § The prose table,
old text beside new; every card is derived rather than typed; the one shader edit is written out.

Every citation names a symbol or a marker, not a line, per `conventions.md ## Code rules`.

## Dependencies on waves that have not landed

W-H is last in the parent's § Order and depends on three waves. Two of them are unlanded at the time
this spec is written, and this spec is written against their locked specs rather than against code.

| Wave | Status at writing | What W-H consumes | Where it is fixed |
|---|---|---|---|
| **W-A** canvas presets | **landed**, verified in code: `tabs/document.py::_SQUARE_PRESETS` is `(256, 512, 1024, 2048)` and `tests/test_canvas_presets.py` exists | the `512x512 (1:1)` entry of `_SQUARE_PRESETS`, which is what makes "Before you start" performable | `20_wave_a_canvas_viewer.md § 5`, group 1 |
| **W-E** keyboard audit | **in flight** (uncommitted at `41bce30`; `tests/test_keymap_disjoint.py` is untracked) | the chord every tutorial sentence quotes, `OPEN_SCRIPT` above all | `02_keybindings.md § The moves`, `50_wave_e_keyboard.md § 6` |
| **W-G** scripting | **not started** | the nested `{pass: {uniform: value}}` contract, `MouseState.down` / `prev_x` / `prev_y`, `RESET_FEEDBACK` on F6 | `60_wave_g_scripting.md § 2, § 10, § 11` |
| **W-D** naming | **not started** | the renamed uniforms in the RC example (`u_paint`, `u_cascade`) that every generated card and code block quotes | `70_wave_d_wiring_naming.md § 9` |

**W-H must not start before W-D and W-G have landed**, and the reason is mechanical rather than
procedural: the generator reads `graph.json` and `passes/*.frag.glsl` at build time, so building
against un-renamed files produces cards naming `u_scene` and a tutorial that contradicts the shipped
example the reader compares against. The build test in § Tests fails loudly in that state (D9's
naming assertion), which is the intended guard rather than a hoped-for one.

## Goal

A reader who has never seen radiance cascades opens `tutorial.html`, follows it top to bottom in the
app, and ends with the six-pass document the Radiance Cascades example ships. Every pass step names
its pass, states every target setting in one card in one place, and pastes one complete shader file.
No step names a control that does not exist, no step depends on a note four paragraphs earlier, and
no card can drift from the example, because the cards and the code are read out of the example at
build time rather than typed.

The two things a reader could not do before this wave become possible: the canvas reaches 512x512
through a control (W-A), and the scene is painted with the mouse through a document script (W-G),
which is the goal 068 lost when D7 was retracted.

## Findings folded

Nine, quoted verbatim from `00_findings.md`. Every one is a text or generator change; none needs
engine code except #6's shader edit.

- **#1** (DEFECT + UX, Before you start): "set the canvas to 512x512 in the Document tab — how?
  there is no 512x512 option, where did you find it?"
- **#6** (DEFECT (example source), Step 4 jfa): "`jfa.frag.glsl:17` in the shipped example says a
  short run count makes 'the pass settings panel warn'. No such warning exists" — the finding's own
  verified column, the maintainer's half being the walk that surfaced it. The finding continues:
  "the spec should decide whether the run count should be derivable from the canvas in-shader (e.g.
  via `u_pass_iterations` being a maximum and an early-out) rather than a hand-set number."
- **#8** (DEFECT, Before you start / Step 1): "the tutorial says 'Each step below adds one pass:
  click add pass, name it' and then 'Add a pass called paint:' — WE ALREADY HAVE THE DEFAULT FIRST
  PASS. Should I delete it or what?"
- **#20** (DEFECT (text), source not located): "'Add scripts/script.py to the document' — we don't
  add this manually. We just hit Ctrl+R in the editor and the script is created."
- **#27** (DEFECT, Step 3 seed and every pass step): "for the seed pass I don't see which resolution
  it should be — the tutorial only gives the 32-bit reasoning. And sampling?"
- **#31** (DEFECT (structure), whole tutorial): "Very inconsistent. It mimics a structure but has no
  real flow. When I create a pass I want all its main parameters — shape, datatype, name — in one
  place; instead they are scattered, differ per section, some sections skip the size. The attention
  blocks mimic order but it's theater."
- **#33** (DEFECT, Step 6 Sphere marching and step 2 Naive GI): "'Replace the naive inner loop's
  fixed step with a jump by the distance field' — replace WHERE?"
- **#34** (DEFECT, Steps 7-8 Radiance cascades / The merge): "this is not the whole shader code???"
- **#35** (DEFECT (naming), Step headings vs pass names): "you call the step 'The merge', but the
  shader is called 'cascade', and you mention that word only once."

#20's location was never found (the parent's open question 1). It lands as the build-time check the
maintainer chose as the default, stated positively: every script mention carries the chord, and no
"add ... script" instruction exists anywhere in the tutorial or Help.

## Out of scope

- **Re-rendering the six stage images.** `img/*.png` are the six pass renders and the passes they
  picture do not change under this wave, so the existing PNGs stand. The one that COULD have changed
  is `paint.png`, and it does not: the analytic scene stays in the shipped example (§ Design
  decision 7), and the script the tutorial teaches drives a brush ON TOP of it rather than replacing
  it. *Trigger to re-render: a change to any shipped `passes/*.frag.glsl` that alters what the pass
  draws.*
- **A seventh pass in the shipped example.** The example keeps its six passes; the tutorial's paint
  step teaches a script that drives `paint`'s existing uniforms, and the example ships without the
  script (§ Design decision 7). *Trigger: a maintainer decision that the shipped example should be
  interactive.*
- **The naive-GI pass as a real buildable pass.** #33 offered two fixes and named which is honest
  ("The first is the honest one given the example ships without it"). W-H takes it: naive GI and
  sphere marching become interludes that build nothing. *Trigger: none; the example would have to
  grow the pass first.*
- **A tutorial for the Bloom Chain example.** W-D renames its uniforms and nothing teaches it.
  *Trigger: a second tutorial is asked for.*
- **Publishing the tutorial anywhere.** 068's "Tutorial lives in `ai_docs/` and is never shipped"
  stands unchanged; `build.sh` does not see `ai_docs/`.
- **A generator for the Help panel.** `help_content.py` gets two hand edits (§ The prose table); it
  is not templated from anything. *Trigger: a third consumer of the same facts.*

## Design decisions

### 1. The generator reads the example, and the body file holds only prose and markers

`build_tutorial.py` grows from an image splicer into a three-marker templater. The three markers,
all in `tutorial_body.html`:

| Marker | Replaced by | Source |
|---|---|---|
| `{{CARD:<pass>}}` | the pass card table (§ The generated card) | `graph.json` |
| `{{CODE:<pass>}}` | the complete shader in a `<pre><code>` block | `passes/<pass>.frag.glsl` |
| `{{IMG:<pass>}}` | a base64 data URI (unchanged from today) | `img/<pass>.png` |

The module gains one constant naming the example, and the six pass names come from the graph rather
than from a typed tuple:

```python
EXAMPLE_ID = "77a84d27-2e5b-406d-8011-ee1cb1a9587c"
EXAMPLE_DIR = (
    pathlib.Path(__file__).resolve().parents[3]
    / "shaderbox"
    / "resources"
    / "document_examples"
    / EXAMPLE_ID
)
```

`parents[3]` from `ai_docs/features/068_radiance_cascades/build_tutorial.py` is the repo root; the
path is asserted at build time by the `graph.json` read failing loudly if it is wrong, which is
better than a guard that prints a friendlier message and continues.

**The generator never imports `shaderbox`.** It reads `graph.json` with `json.load` and takes the
defaults it needs as three module constants (§ decision 3), rather than importing `TargetConfig`.
Two reasons, and the second is the load-bearing one: the script runs from `ai_docs/`, outside the
package's own tree, so an import would make a documentation build depend on the app's import graph
being healthy; and a `TargetConfig` import would silently re-import `pydantic` and `moderngl`
transitively. The cost is that the three defaults are duplicated, and § decision 3 states how the
test keeps them honest.

The build order matters: cards and code are spliced BEFORE images, because a code block could in
principle contain the string `{{IMG:` and a card never can. In practice neither does, and the order
is fixed anyway so the property does not depend on the shader files' current content.

```python
def build() -> None:
    graph = json.loads((EXAMPLE_DIR / "graph.json").read_text(encoding="utf-8"))
    body = (HERE / "tutorial_body.html").read_text(encoding="utf-8")
    for name in graph["passes"]:
        body = body.replace(f"{{{{CARD:{name}}}}}", _card_html(name, graph))
        body = body.replace(f"{{{{CODE:{name}}}}}", _code_html(name))
    for name in graph["passes"]:
        body = body.replace(f"{{{{IMG:{name}}}}}", _data_uri(name))
    (HERE / "tutorial.html").write_text(body, encoding="utf-8")
```

The image loop is separate rather than folded into the first, because `img/` and `passes/` need not
hold the same set forever and the failure modes differ (a missing PNG is a `FileNotFoundError`, a
missing marker is a silent no-op the test catches).

### 2. The card's data model: one row per gear control, in the gear's own order

A card is a `<table class="card">` with a fixed row set. The order is the parent's, and it is the
order the gear draws its controls in, so a reader filling the gear reads the card top to bottom:

| Row | `graph.json` source | Rendered as |
|---|---|---|
| name | the pass key | `<code>jfa</code>` |
| reads | `passes[name].inputs`, sorted by uniform name | `<code>u_seed</code> from <b>seed</b>` per line, or `nothing` when empty |
| format | `passes[name].target.dtype` | the gear's own label (§ decision 3) |
| size | `passes[name].target.scale` | `100%` (percent, no decimals) |
| smooth | `passes[name].target.filter_linear` | `on` / `off` |
| repeat | `passes[name].target.wrap` | `on` / `off` |
| runs | `passes[name].iterations` | the integer |

**Format sits above size** because that is the order `popups/pass_settings.py` draws them in under
the `Draws into` separator, and the card's whole claim is that a reader fills the gear by reading it
top to bottom. Transposing two adjacent rows on all six cards is a small cost paid six times.

**The card's `smooth` and `repeat` labels are the gear's CHECKBOX captions, not its label column's
`sampling` and `edges`**, because the checkbox is the control the reader clicks. That is the one
place the card deliberately does not echo the gear's label column, and it is written down here so an
implementer told to "match the gear" does not pick the other one.

**Every row is always present**, including the ones whose value is the default. That is #31's
demand read literally ("with 'default' written where the default holds, so the card is complete even
when nothing changes"), and it is what makes the card the single place a reader looks.

A row whose value equals the gear's default is suffixed with the word `default` in a dimmed span:

```html
<tr><td>size</td><td>100% <span class="dfl">default</span></td></tr>
```

The suffix is a marker on the value, never a replacement for it. A card reading `size default` would
make the reader open the gear to find out what the default IS, which is the round trip the card
exists to remove.

`u_prev` reads a special line, because D9 makes it the one input whose source is not a sibling:

```html
<code>u_prev</code> from <b>jfa</b> (itself, last run)
```

The parenthetical is generated, not typed: it is emitted when `inputs[uniform] == name`. The words
"last run" rather than "last frame" are correct for an ITERATED pass and are what the engine does
(068 D5, the per-iteration swap); on a pass with `iterations: 1` the emitted words are
`(itself, last frame)`. Both examples of that exist in this document: `jfa` and `cascade` are
iterated, and neither example ships a feedback pass at `iterations: 1`, so the second branch is
exercised only by the unit test. It is written anyway rather than left to a future reader, because a
generator that is right for one branch and absent for the other is the shape that produces a wrong
card the day someone adds the pass.

### 3. The three defaults are duplicated into the generator, and one test pins them

```python
_DEFAULT_SCALE = 1.0
_DEFAULT_DTYPE = "f2"
_DEFAULT_FILTER_LINEAR = True
_DEFAULT_WRAP = False
_DEFAULT_ITERATIONS = 1
```

These mirror `pass_graph.py`'s `DEFAULT_SCALE`, `DEFAULT_DTYPE`, `DEFAULT_FILTER_LINEAR`,
`DEFAULT_WRAP` and `PassEntry.iterations`'s field default. **The duplication is deliberate and
gated** (§ Tests, `test_the_generator_defaults_match_the_engine`): the test imports both and
compares them, so a change to `pass_graph.py`'s defaults turns the suite red with the tutorial named
as the thing to rebuild. That is the shape the repo already uses for a doc-derived fact, and it costs
one assertion.

The default set is what makes `filter_linear` the interesting row: `DEFAULT_FILTER_LINEAR` is
**True**, and five of the RC example's six passes set it False. So five cards read `smooth off` with
no `default` marker and one (`composite`) reads `smooth on default`. That is #27's grievance
answered exactly: "smooth must be switched off by hand on five of six passes, and the tutorial says
so for only two" becomes six cards each stating it.

The dtype labels come from the gear's own vocabulary so the card names the control's own words:

```python
_DTYPE_LABELS = {"f1": "8-bit", "f2": "16-bit float", "f4": "32-bit float"}
```

These are the `_FORMATS` labels in `popups/pass_settings.py`. W-B cut their TOOLTIPS to one clause
each and left the labels alone; the implementer re-reads them at implementation time and matches
whatever they say, since a card naming a format the combo does not is exactly the defect class this
wave exists to close. The test in § Tests asserts the three keys are the `DTYPES` tuple, so a fourth
dtype cannot land without the card growing a label for it.

### 4. The body file's structure: six pass steps, four interludes, two bookends

`tutorial_body.html` is rewritten. The numbered steps become the six passes and nothing else, which
is #35's rule ("every pass step's heading IS the pass name, with the concept as a subtitle") and
#31's ("numbers 1-6 are the six passes").

| Position | Kind | Heading | Body |
|---|---|---|---|
| — | bookend | `Radiance Cascades in ShaderBox` | lede, the source-article credit, the table of contents, **Before you start** |
| 1 | pass step | `paint` — the scene to light | card, produces, code, explanation, then the paint-script subsection |
| — | interlude | Naive global illumination | concept only, builds nothing |
| 2 | pass step | `seed` — positions, not distances | card, produces, code, explanation |
| 3 | pass step | `jfa` — the jump flood | card, produces, code, explanation, the run-count formula |
| 4 | pass step | `df` — one line of real work | card, produces, code, explanation |
| — | interlude | Sphere marching | concept only, points at `cascade`'s `march` |
| — | interlude | The cascade idea | why direction and position trade off |
| 5 | pass step | `cascade` — the cascade stack | card, produces, code (all 109 lines), explanation in three parts: packing, march, merge |
| — | interlude | The merge | the one trap, folded INTO step 5's explanation, not a step |
| 6 | pass step | `composite` — what you look at | card, produces, code, explanation |
| — | bookend | Verifying it | the oracle, why a plausible render proves nothing |
| — | bookend | What you built / Things to try | unchanged in kind |

Ten headings become six numbered plus four unnumbered, and the interludes carry an explicit opening
sentence #33 asks for by name: **"Nothing to build here."** It is the first thing in every interlude,
in bold, so a reader skimming for the next pass cannot mistake an idea for a step.

The merge is the one structural judgement call. #34 asks for the cascade pass to appear whole and
#31 lists the merge among the interludes. Both are satisfied by making the merge the third part of
step 5's EXPLANATION rather than a separate interlude: the code is already on the page above it
(spliced whole), so the explanation refers to its lines rather than re-quoting a fragment, which is
what #34's fix says in as many words ("then the explanation walking its sections (packing, march,
merge) by referring to the lines above rather than re-quoting them"). An interlude between the code
and its own explanation would separate them for no gain.

The per-step body order is fixed and identical for all six, per D8:

```html
<h2 id="jfa"><span class="step">3</span><code>jfa</code> — the jump flood</h2>
{{CARD:jfa}}
<p class="produces">...one sentence: what this pass produces...</p>
<figure><img src="{{IMG:jfa}}" ...><figcaption>...</figcaption></figure>
{{CODE:jfa}}
<p>...the explanation...</p>
```

The picture sits between the "produces" sentence and the code, rather than after the code as it does
today, because a reader deciding whether their own pass is right looks at the picture immediately
after being told what to expect. The parent's template lists "what it produces + picture" as one
item, which this ordering is.

### 5. The step's lead-in is self-sufficient, and step 1 renames rather than adds

#8 is the sharpest structural finding: an instruction that depends on a note four paragraphs earlier.
The fix is a rule the implementer applies to all six, not a patch to one:

**Every pass step opens with the verb the reader performs, naming the pass, with no back-reference.**
Steps 2-6 open `Add a pass called <code>seed</code>`, the phrase D8 forbids for step 1 only.
Step 1 opens with the rename, because the starter document already has a pass:

> **Rename the starter's pass.** A new document arrives from the UV Mango starter with one pass
> called `main`. Open its gear (`Alt+P`), type `paint` in the name field, click elsewhere to commit,
> and replace its shader with the code below. Do not delete it; steps 2 to 6 are the ones that add a
> pass.

Three things in that paragraph are wave-dependent facts rather than prose choices, and each is
verified in § Verified / corrected premises: the starter IS UV Mango and its one pass IS called
`main`; the gear chord IS `Alt+P` (W-C landed it, W-E's audit confirmed it); and the name field
commits on deactivate-after-edit, which is W-C's D11, so "click elsewhere" is the instruction and
Enter is the shortcut.

### 6. The paint step, written against W-G's contract

This is the wave's one piece of genuinely new teaching, and it is what lifts 068 D7's retraction. The
shipped example stays analytic (§ decision 7); the tutorial adds a subsection to step 1 that teaches
the reader to drive it with the mouse.

The subsection sits after step 1's explanation, under an `<h3>`, and reads in this order:

1. **The uniforms.** The reader adds three lines to `paint.frag.glsl` and one line to its `main()`.
2. **The script.** `Alt+R` creates and opens `scripts/script.py`; the complete script is pasted over
   the stub.
3. **What happens.** LMB paints, hover does not, and the stroke is a capsule rather than blobs.
   There is nothing to clear, and § The paint script says why.

The shader addition, written out because the reader pastes it:

```glsl
uniform vec2 u_brush;        // where the cursor is, 0..1
uniform vec2 u_brush_prev;   // where it was last frame
uniform float u_brush_down;  // 1 while the left button is held
```

and, inside `main()`, immediately before the final `fs_color = vec4(0.0);`:

```glsl
    // Distance to the SEGMENT from last frame's cursor to this one, so a fast drag paints a
    // continuous stroke instead of one disc per frame.
    vec2 ab = u_brush - u_brush_prev;
    float h = clamp(dot(p - u_brush_prev, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
    if (u_brush_down > 0.5 && length(p - (u_brush_prev + ab * h)) < 0.02) {
        fs_color = vec4(4.0, 3.3, 2.0, 1.0);
        return;
    }
```

**Why three floats and not one `Vec3`.** The script returns a `Vec2` for each cursor and a plain
`float` for the button, which are the shapes `outputs.py` already carries and `coerce_one` already
accepts. Packing the button into a third component of a `Vec3` would save a uniform and cost the
reader the naming that makes the shader readable.

**Why `u_brush_prev` and not a feedback pass.** `paint` at `iterations: 1` reading itself would make
the scene persistent, which is a different and larger teaching (and would need `F6` to be the only
way back to an empty canvas). The capsule is the smaller thing that answers #22's (a) half, and the
`Clear canvas` command answers #23. A reader who wants a persistent canvas is pointed at "Things to
try", where it is one sentence.

### 7. The shipped example keeps no script, and the tutorial says so once

The RC example ships six passes, no `scripts/` directory, and an analytic `paint.frag.glsl`. W-H does
not add a script to it. Three reasons, in order of weight:

1. **The example is the finished artifact and the tutorial is the path to it** (068 D6). A reader who
   opens the example to compare against their own document should see the same six passes, not a
   seventh state their tutorial-built document lacks.
2. **An export must be deterministic.** `EXPORT_MOUSE` freezes the cursor at the centre with
   `down=False` (W-G § 10), so a script-driven example would render a fixed brushless frame in an
   export and a different picture live. The example is used in `make smoke` and in the examples
   popup, both of which render it.
3. **`paint.frag.glsl`'s own comment already explains the analytic choice**, and W-D leaves it alone.
   Its second paragraph is the one that goes stale (§ The prose table): it states 068 D7's retraction
   as a present fact, and W-G lifts that retraction.

So the tutorial's paint subsection opens with one sentence naming the difference, and it is the only
place the difference is stated:

> The shipped example draws its scene analytically and ships no script, so an exported video is the
> same every time. Everything below is yours to add on top.

### 8. The JFA shader derives its offset from the canvas, and `iterations` becomes 12

This is #6, and it is the wave's one change to shipped engine-adjacent source. The parent fixes the
formula; this decision fixes the whole edit.

`passes/jfa.frag.glsl`'s `main()` opens with:

```glsl
    float offset = exp2(ceil(log2(max(u_resolution.x, u_resolution.y))) - 1.0 - u_pass_iteration);
    // Runs past the end of the chain have nothing left to spread; pass the buffer through.
    if (offset < 1.0) {
        fs_color = u_pass_iteration < 0.5 ? texture(u_seed, vs_uv) : texture(u_prev, vs_uv);
        return;
    }
```

replacing today's `float offset = pow(2.0, u_pass_iterations - u_pass_iteration - 1.0);`.

**Three properties, each checked rather than assumed.**

- The offset no longer reads `u_pass_iterations` at all, so the run count stops changing what any
  single run does. It is derived from the CANVAS, which is what makes one number correct at every
  canvas rather than at one. Measured: at 512 the offsets are 256, 128 ... 1 over runs 0 to 8; at
  4096 they are 2048 down to 1 over runs 0 to 11. `ceil(log2(max side))` is 9, 10, 11 and 12 at 512,
  1024, 2048 and 4096, the last being `MAX_CANVAS_PX`.
- **A surplus run is a pass-through, not a no-op.** The early return copies the previous run's buffer
  forward rather than leaving the target untouched, and it must: the pass ping-pongs between
  iterations (068 D5), so a run that writes nothing leaves the target holding the state from TWO runs
  ago and the chain goes backwards. The `u_pass_iteration < 0.5` branch is not reachable at any
  canvas the clamp permits (`MIN_CANVAS_PX` is 16, so run 0's offset is at least 8); it is written
  because the early return must be total. A reader who sets `iterations` to 1 on a pass whose first
  run is surplus would otherwise get an unwritten target, and a branch that is right for one case
  and absent for the other is the shape that produces a wrong render the day the bound moves.
- **`u_pass_iterations` becomes unused in this shader and its declaration goes.** Leaving a declared
  uniform the shader never reads would make it inactive at link time and absent from
  `get_active_uniforms`, which is harmless but is a declaration that lies. The engine still binds
  `u_pass_iteration`, which the shader does read.

`graph.json`'s `jfa.iterations` becomes **12**, which is `ceil(log2(MAX_CANVAS_PX))` and therefore
covers every canvas the app can make. **This is a deliberate deviation from the parent's 11**
(`01_spec.md § W-H` derived that number from the largest square preset); § Review history records
the reason. The reachable set is not the preset list: W-A shipped free-form width and height fields
beside the presets combo, both committed through `clamp_canvas_size`, whose ceiling is
`MAX_CANVAS_PX` = 4096. A reader who types 4096 into one field performs no JSON edit and breaks no
rule, and at 11 runs their chain would be one run short with no warning, which is the exact failure
#6 exists to close.

The cost is surplus runs at smaller canvases: three at 512, each a full-screen pass-through draw in
a pass that already runs nine times. That is what buys one number being right at every canvas
instead of at four presets. Measured offsets: at 4096 the runs are 2048, 1024 ... 1 with none
surplus; at 512 they are 256 down to 1 over runs 0-8, with runs 9, 10 and 11 surplus.

The comment block at the top of `jfa.frag.glsl` loses its final paragraph (the "panel warns" claim,
which is #6's defect) and its "ONE shader, run 9 times" opener becomes 12. Both are in § The prose
table.

**The `document.json` description also says 9** and is corrected in the same edit; it is the string
the examples popup shows.

### 9. The build test lives in `tests/` and imports the generator

`tests/test_tutorial_build.py` imports `build_tutorial` from `ai_docs/features/068_radiance_cascades/`
by path, because `ai_docs/` is not a package:

```python
_BUILD = importlib.util.spec_from_file_location(
    "build_tutorial",
    pathlib.Path(__file__).resolve().parents[1]
    / "ai_docs" / "features" / "068_radiance_cascades" / "build_tutorial.py",
)
```

The alternative (shelling out to `uv run python build_tutorial.py` and reading the output file) was
rejected: it would make the test write `tutorial.html` as a side effect, so a failing test would leave
a half-built artifact in the tree, and it could not assert on the intermediate card HTML. The import
gives the test `_card_html` and `_code_html` directly, and `build()` is exercised once through a
`tmp_path` output.

The test is GL-free and needs no display, so it runs under `make check`'s sibling `make test` on any
box. It reads the SHIPPED example rather than a fixture, which is the whole point: it is the gate
that catches an example edited without a tutorial rebuild.

### 10. `tutorial.html` is regenerated and committed

The generated file is tracked today and stays tracked. The wave's last act is running
`uv run python ai_docs/features/068_radiance_cascades/build_tutorial.py` and committing the result
with the body and the generator. A test asserting the tracked `tutorial.html` matches a fresh build
was considered and rejected: it would turn every example edit into a red suite until someone reruns a
script, which is a worse failure than a stale HTML file that the build test already proves is
rebuildable. The § Manual verification step is the reader opening the regenerated file.

## The generated card (worked example: `jfa`)

What `{{CARD:jfa}}` becomes, given the RC `graph.json` after W-D's rename and this wave's
`iterations: 12`. Rendered as text, values first:

```
name     jfa
reads    u_seed from seed
         u_prev from jfa (itself, last run)
format   32-bit float
size     100%                        default
smooth   off
repeat   off                         default
runs     12
```

and as the HTML the generator emits:

```html
<table class="card">
  <tr><td>name</td><td><code>jfa</code></td></tr>
  <tr><td>reads</td><td><code>u_seed</code> from <b>seed</b><br>
      <code>u_prev</code> from <b>jfa</b> (itself, last run)</td></tr>
  <tr><td>format</td><td>32-bit float</td></tr>
  <tr><td>size</td><td>100% <span class="dfl">default</span></td></tr>
  <tr><td>smooth</td><td>off</td></tr>
  <tr><td>repeat</td><td>off <span class="dfl">default</span></td></tr>
  <tr><td>runs</td><td>12</td></tr>
</table>
```

Reading the rows against `graph.json`: `dtype` is `f4` against a `f2` default, so no marker; `scale`
is `1.0` and `_DEFAULT_SCALE` is `1.0`, so `default`; `filter_linear` is `false` against a `True`
default, so no marker, which is the row #27 says every step must carry; `wrap` is `false` and matches,
so `default`; `iterations` is `12` against `1`, so no marker.

The other five cards, in one table so the implementer can check the generator's output at a glance
(every value read from the shipped `graph.json`, with W-D's renames and this wave's `iterations`):

| Pass | reads | format | size | smooth | repeat | runs |
|---|---|---|---|---|---|---|
| `paint` | nothing | 16-bit float default | 100% default | off | off default | 1 default |
| `seed` | `u_paint` from `paint` | 32-bit float | 100% default | off | off default | 1 default |
| `jfa` | `u_seed` from `seed`; `u_prev` from `jfa` (itself, last run) | 32-bit float | 100% default | off | off default | 12 |
| `df` | `u_jfa` from `jfa` | 16-bit float default | 100% default | off | off default | 1 default |
| `cascade` | `u_df` from `df`; `u_paint` from `paint`; `u_prev` from `cascade` (itself, last run) | 16-bit float default | 100% default | off | off default | 6 |
| `composite` | `u_cascade` from `cascade`; `u_paint` from `paint` | 8-bit | 100% default | on default | off default | 1 default |

The `reads` rows are sorted by uniform name, which is why `cascade` lists `u_df` before `u_paint`
before `u_prev`. Sorting rather than preserving `graph.json`'s key order is deliberate: JSON key order
is an artifact of how the file was written and would make two semantically identical graphs produce
different cards.

## The paint script (complete)

What the tutorial ships in step 1's paint subsection, written against W-G's nested contract and
`MouseState`'s new fields. It is pasted over the stub `Alt+R` creates.

```python
from shaderbox.scripting import ScriptBehavior, Ctx, Vec2


class Brush(ScriptBehavior):
    """Paint into the scene with the left mouse button."""

    def update(self, ctx: Ctx) -> dict:
        return {
            "paint": {
                "u_brush": Vec2(ctx.mouse.x, ctx.mouse.y),
                "u_brush_prev": Vec2(ctx.mouse.prev_x, ctx.mouse.prev_y),
                "u_brush_down": 1.0 if ctx.mouse.down else 0.0,
            },
        }
```

Six things about it, each a teaching point the tutorial's surrounding prose makes:

- **The key is `"paint"`, not the output pass.** That is the whole of D3 in one line: the script
  addresses a pass by NAME, so it drives `paint` while the reader is looking at `composite`. The
  tutorial says so explicitly, because it is exactly what 068 could not do.
- **The value under `"paint"` is a dict, which is what makes it a pass block.** A bare
  `"u_brush": ...` at the top level would broadcast to every pass declaring `u_brush`, which here is
  only `paint` and would work, and the tutorial says that too, in one sentence, as the second half of
  the grammar. The block form is taught first because it is the form that scales.
- **`ctx.mouse.down` is a bool and the uniform is a float**, so the script converts. GLSL has no
  bool uniform in this engine's coercion set, and `1.0 if ... else 0.0` is clearer than relying on a
  bool coercing.
- **`prev_x` / `prev_y` come from the engine**, not from `self`. A script keeping its own previous
  position would miss the re-entry reset W-G's fill performs (a cursor that leaves the canvas and
  returns gets `prev == current`, so no line is drawn across the gap).
- **No `__init__` and no `self` state.** The script is stateless because the engine now carries the
  one piece of state a brush needs. That is worth the tutorial naming, since 041's whole premise is
  that state is why CPU scripting exists.
- **The import line matches what `script_stub_for` emits**, name for name and in order:
  `_script_import_line` always leads with `ScriptBehavior, Ctx` and appends only the output types the
  document's uniforms reference, so a document whose only shaped uniform is a `vec2` emits exactly
  `from shaderbox.scripting import ScriptBehavior, Ctx, Vec2`. W-G § 9 widens the annotation source
  to every pass rather than the output pass alone. The bare `-> dict` return annotation is the stub's
  too, and the tutorial keeps it rather than writing a precise nested type, because the engine's own
  stub is what a reader compares against.

The tutorial states the result immediately after: hold the left button to paint, and hover does
nothing. Then the sentence that stops a reader looking for a clear key:

> There is nothing to clear: `paint` redraws from scratch every frame, so releasing the button leaves
> the analytic scene and the next frame has no brush in it. `F6` (Clear canvas) and its `Clear`
> button appear only once a pass reads itself, which is why they belong to the persistent-canvas
> suggestion in Things to try rather than to this step.

**This corrects the spec's own first draft**, which taught `F6` here. `Document.reset_feedback`
releases `_feedback`'s canvases, and `_feedback` is filled only for an input whose source is the
pass itself; `paint` at `iterations: 1` reads nothing and reads no self-input, so the command is a
literal no-op on the document step 1 builds. W-G § 11 also draws the `Clear` button only when
`plan_passes(graph)[0].feedback` is non-empty, and the tutorial's document has no feedback pass
until `jfa` in step 3 — so at step 1 there is no button on the preview to press. Teaching a chord
that does nothing, beside a button that is not there, is the class of defect this whole wave exists
to remove.

## Files touched

| File | Change |
|---|---|
| `ai_docs/features/068_radiance_cascades/build_tutorial.py` | `EXAMPLE_DIR` + `EXAMPLE_ID`; the five default constants and `_DTYPE_LABELS`; `_card_html`, `_code_html`, `_reads_html`; `build()` reads `graph.json` and splices cards and code before images. |
| `ai_docs/features/068_radiance_cascades/tutorial_body.html` | Rewritten per § decision 4 and § The prose table: six pass steps with `{{CARD:}}` / `{{CODE:}}` markers, four interludes, the paint subsection, the CSS gains `.card` / `.dfl` / `.produces`. Every pasted shader body is DELETED (the markers replace them). |
| `ai_docs/features/068_radiance_cascades/tutorial.html` | Regenerated (§ decision 10). |
| `shaderbox/resources/document_examples/77a84d27-.../passes/jfa.frag.glsl` | The canvas-derived offset + the surplus-run pass-through; `u_pass_iterations` declaration removed; the header comment's run count and the "panel warns" paragraph (§ decision 8, § The prose table). |
| `shaderbox/resources/document_examples/77a84d27-.../passes/paint.frag.glsl` | Header comment only: the second paragraph's 068 D7 clause, which `{{CODE:paint}}` splices into step 1 above the paint subsection that contradicts it (§ The prose table, F7). No GLSL changes. |
| `shaderbox/resources/document_examples/77a84d27-.../graph.json` | `jfa.iterations` 9 → 12. |
| `shaderbox/resources/document_examples/77a84d27-.../document.json` | the description's "9 runs" → "12 runs". |
| `shaderbox/help_content.py` | the `your_uniforms` script sentence names the chord (§ The prose table). |
| `ai_docs/features/068_radiance_cascades/01_spec.md` | the history and CORRECTION callouts the tutorial sheds land here (§ The prose table, the four "moved to 068 spec" rows). |
| `tests/test_tutorial_build.py` | New. § Tests. |
| `ai_docs/roadmap.md` | the 069 row and the banner; the 068 row's "tutorial is being rewritten under 069 W-H" clause becomes the finished state. |

**Not touched, each checked:** `shaderbox/resources/document_examples/77a84d27-.../passes/` other
than `jfa` and `paint`'s header (W-D owns the renames, and no other shader's CODE changes under this
wave); `img/*.png` (§ Out of scope, and `jfa`'s picture at 12 runs is identical to 9 because the
surplus runs copy forward); `oracle.py` (the numerical check is 068's and the merge is unchanged);
`build.sh` (`ai_docs/` is not bundled); `copilot/prompt.py` (W-D wrote the naming rule into the pass
block already).

## Tests

`tests/test_tutorial_build.py`, GL-free, no display. Each names the falsifier: the change that turns
it red.

### `test_every_pass_has_a_card_and_a_code_block`

Load the shipped `graph.json`, read `tutorial_body.html`, and assert `{{CARD:<name>}}` and
`{{CODE:<name>}}` both appear for every key in `graph["passes"]`.

*Falsifier:* delete the `{{CARD:df}}` marker from the body and it goes red naming `df`. This is the
parent's "every pass in the example has a card and a code block", asserted on the SOURCE rather than
the output, so the error message points at the file the implementer edits.

### `test_no_marker_survives_the_build`

Run `build()` against a `tmp_path` output and assert `"{{" not in html`.

*Falsifier:* add a `{{CARD:naive}}` marker for a pass the example does not have and it goes red; the
build's `for name in graph["passes"]` loop cannot replace it. This is the check that catches the
likeliest editing mistake, a marker for a pass that was renamed.

### `test_a_card_states_every_row_and_marks_the_defaults`

Call `_card_html("jfa", graph)` and assert the seven row labels appear in the gear's order (name,
reads, format, size, smooth, repeat, runs), that `32-bit float` appears without a `dfl` span, and
that `size` and `repeat` carry one. Then
`_card_html("composite", graph)` and assert `smooth` carries `on` and a `dfl` span.

*Falsifier:* make `_card_html` omit a row whose value is the default and the `size` assertion goes
red. That is #31's demand ("with 'default' written where the default holds, so the card is complete
even when nothing changes"), and it is the one property a generator would plausibly get wrong by
being helpful.

### `test_a_code_block_is_the_whole_file`

Assert `_code_html("cascade")` contains `#version 460 core`, contains the final `}` of the shipped
file, and that the number of lines between them equals the file's own line count.

*Falsifier:* splice a fragment (say from the first `void main()`) and the `#version` assertion goes
red. This is #34 asserted: "this is not the whole shader code???" cannot recur while the block starts
at the file's first line.

### `test_the_code_block_escapes_html`

Assert `&lt;` appears in `_code_html("jfa")` (its loop has `x <= 1.0`) and that a bare `<` does not
appear inside the emitted `<code>` element.

*Falsifier:* drop the `html.escape` call and the shipped `jfa` shader's `uv.x < 0.0` silently opens a
tag in the rendered page. This is the failure the hand-written body avoided by hand-escaping, and a
generator that forgets it produces a page that renders wrong without erroring.

### `test_no_hand_written_fragment_names_an_absent_uniform`

Collect every `<pre><code>` block in `tutorial_body.html` that is NOT a `{{CODE:}}` marker (the
interlude fragments and the paint subsection's GLSL and Python), extract every `u_[a-z_]+` token, and
assert each one appears in some shipped `passes/*.frag.glsl` or is one of the brush uniforms this
wave's step 1 adds (`u_brush`, `u_brush_prev`, `u_brush_down`).

*Falsifier:* leave `u_scene` in the naive-GI or sphere-marching fragment after W-D's rename and it
goes red naming the token. The generated blocks need no such test because they ARE the shipped files;
this is the guard for the hand-written GLSL that generation cannot reach, which is the gap F3 found.
The brush allowlist is written into the test rather than derived, because those three uniforms exist
only in the reader's document and never in a shipped file.

### `test_every_script_instruction_carries_the_chord`

Strip every `<pre><code>...</code></pre>` block from `tutorial_body.html`, split what remains into
sentences, and assert the chord on the sentences that are INSTRUCTIONS about the script rather than
on every sentence naming one:

```python
_INSTRUCTION_VERBS = (
    "add", "adds", "adding", "added",
    "create", "creates", "creating", "created",
    "make", "makes", "making",
    "open", "opens", "opening", "opened",
    "write", "writes", "writing",
    "hit", "hits", "hitting",
    "press", "presses", "pressing", "pressed",
)
```

A sentence is asserted on when it contains `script` (case-insensitive) AND matches
`\b<verb>\b` for at least one verb in that set. It must then contain `Alt+R` or `Script → open`.
A sentence mentioning the script descriptively is not an instruction and is not asserted on.

**Both word boundaries are required, and that is the whole of the matching rule.** A leading
boundary alone (`\badd`) matches `addresses`, and this spec's own § The paint script contains "the
script **addresses** a pass by NAME" — a descriptive sentence about D3 that would be asserted on and
go red, since it carries no chord and should not. Verified by running both predicates over the three
sentences that matter: with `\badd` the D3 sentence is asserted on (wrongly); with `\badd\b` it is
not, while "press Ctrl+R and the script is created" is still asserted on (correctly, via both
`press` and `created`) and § decision 7's "ships no script, so an exported video is the same every
time" is asserted on by neither.

**The inflections are enumerated rather than stemmed.** A stemmer would re-open the same class of
error from the other side (it maps `addresses` to `address`, but also `creating` and `created` to
`create` only if it is a good one), and the set is small enough to write out. The past forms are
included where they read as instructions in the passive voice a tutorial actually uses ("the script
is **created** for you"); `makes` has no past form in the list because "made" in a tutorial sentence
about a script is descriptive rather than instructive.

**The verb set is enumerated here rather than invented at implementation time**, because a checker
whose domain is chosen while making it pass is the failure class the parent's #20 default exists to
prevent. Widening the set, or relaxing either boundary, is a spec edit.

The predicate is what makes the test compatible with the wave's own required prose: § decision 7
mandates "The shipped example draws its scene analytically and ships no script, so an exported video
is the same every time", which names the script, carries no instruction verb, and is correctly not
asserted on. Under a bare "every occurrence of the word" rule that sentence goes red, and the
implementer's only ways out are weakening the checker or deleting mandated prose.

*Falsifier:* write "press Ctrl+R and the script is created" and it goes red naming the sentence,
since it carries `press`, `created` and `script` and neither `Alt+R` nor `Script → open`. Drop the
chord from the paint subsection's `Alt+R` sentence and it goes red the same way. And the falsifier
for the boundary rule itself: relax the match to `\b<verb>` and the suite goes red on § The paint
script's own "the script addresses a pass by NAME", which is the false positive round 2 caught.

The sentence split is on `.`, `!`, `?` and `</p>` / `</li>` boundaries. Code blocks are excluded
because the script's own source contains the word in its filename and in the class docstring, and
neither is an instruction.

### `test_no_add_the_script_instruction_anywhere`

Over `tutorial_body.html` and `help_content.py`'s rendered section bodies, assert no match for a
regex of the shape `add[^.]{0,40}script` (case-insensitive), and none for `scripts/script.py`
preceded by a creation verb.

*Falsifier:* restore the sentence #20 reports ("Add scripts/script.py to the document") in either file
and it goes red. The finding never located the sentence, so this test is the standing guard rather
than a regression test for a known site, and it covers Help as well as the tutorial because the
parent's default names both.

### `test_the_generator_defaults_match_the_engine`

Import `pass_graph` and assert `_DEFAULT_SCALE == DEFAULT_SCALE`, `_DEFAULT_DTYPE == DEFAULT_DTYPE`,
`_DEFAULT_FILTER_LINEAR == DEFAULT_FILTER_LINEAR`, `_DEFAULT_WRAP == DEFAULT_WRAP`,
`_DEFAULT_ITERATIONS == PassEntry().iterations`, and `set(_DTYPE_LABELS) == set(DTYPES)`.

*Falsifier:* flip `DEFAULT_FILTER_LINEAR` to False in `pass_graph.py` and it goes red. That is the
gate § decision 3 owes for duplicating the constants, and the `DTYPES` half is what stops a fourth
format landing without a card label.

### `test_the_jfa_run_count_covers_every_reachable_canvas`

Read the example's `graph.json` and assert its `jfa.iterations` clears two bounds, each imported
rather than typed:

```python
runs = graph["passes"]["jfa"]["iterations"]
assert runs >= math.ceil(math.log2(max(_SQUARE_PRESETS)))     # tabs/document.py
assert runs >= math.ceil(math.log2(MAX_CANVAS_PX)), (         # pass_graph.py
    f"jfa runs {runs} is short of ceil(log2(MAX_CANVAS_PX={MAX_CANVAS_PX})); "
    "the canvas fields reach the clamp, not only the presets"
)
```

Two assertions rather than one because the two bounds can move independently and the preset list is
no longer the reachable set (§ decision 8): a canvas reaches `MAX_CANVAS_PX` through W-A's width and
height fields with no preset involved. The failure message names the clamp so a reader who raised it
is told what to recompute.

*Falsifier:* set `iterations` back to 9 and both assertions go red. Set it to 11, the parent's
number, and the first passes while the second goes red naming `MAX_CANVAS_PX`, which is exactly the
gap F6 found and the reason this wave ships 12. Raise `MAX_CANVAS_PX` to 8192 without raising the
count and it goes red again. The parent asks for this check "if the build can check it against the
example's `graph.json`". It can, and this is it.

### `test_the_tutorial_names_no_chord_the_command_table_does_not_have`

For every `Ctrl+X` / `Alt+X` / `F<n>` string in `tutorial_body.html`, assert the chord is the
`default_chord` of some `CommandSpec` in `COMMAND_SPECS`, rendered through the same formatter the
Help shortcuts table uses.

*Falsifier:* leave a `Ctrl+R` in the body after W-E moved `OPEN_SCRIPT` to `Alt+R` and it goes red
naming `Ctrl+R`. **This is the test that catches the parent's own stale instruction** (§ Verified /
corrected premises, item 3), and it is the general form of the specific check the parent asked for:
rather than pinning one chord's spelling, it pins every chord the tutorial quotes to the live
registry, so the next audit that moves a chord turns this red instead of leaving the tutorial wrong.

## Manual verification

The maintainer walks the tutorial end to end in the app, from a new document. One falsifiable step
per tutorial step; each fails for exactly one reason.

1. **Before you start.** `Ctrl+Shift+N` makes a document; the Document tab's presets menu lists
   `512x512 (1:1)`; picking it sets both fields to 512 and the pass gear then shows `512x512` at
   100%. *Falsifier: the preset is absent, or the gear still reads 1280x960 (W-A's funnel bug).*
2. **Step 1, `paint`.** The starter's `main` renames to `paint` from the gear with no crash, and the
   pasted shader renders two bright discs, a thin wall and a round blob on black. *Falsifier: the
   rename crashes (W-C), or the viewer stays black because 16-bit float was not set.*
3. **The paint script.** `Alt+R` creates and opens `scripts/script.py` (the tab appears with the stub
   in it, not an error); pasting the § The paint script code and holding the left button over the
   canvas paints a continuous stroke; hovering with no button paints nothing; a fast drag leaves a
   line rather than spaced discs; releasing the button leaves the analytic scene with no brush
   residue. *Falsifier: hover paints (`down` not wired), or the stroke is discs (`prev_x` not
   wired), or the script's `"paint"` key errors on the strip (the pass block did not route, W-G).*
4. **Step 2, `seed`.** Added as a pass, wired `u_paint` from `paint` with the gear untouched: the
   name does the wiring (W-D). Format set to 32-bit float. The viewer shows faint red-green dots
   where the scene was solid. *Falsifier: the gear shows `auto: none` for `u_paint`, meaning the name
   rule did not fire.*
5. **Step 3, `jfa`.** Runs set to 12 and the card said 12; the viewer shows a smooth red-green field
   filling the canvas with visible Voronoi seams. Then change the canvas to 1024x1024 from the
   presets, and again to 4096 by typing it into the width and height fields, and the field stays
   correct at both. *Falsifier: at 1024 or 4096 the field goes blocky or the seams break, which is
   the canvas-derived offset not working, or the count being short of the clamp.*

6. **Clear canvas, now that a feedback pass exists.** With `jfa` in the document, the `Clear` ghost
   button is drawn at the preview's top-left and `F6` fires it; the flood field blanks for one frame
   and rebuilds. *Falsifier: no button (W-G's `has_feedback` gate reading `_feedback` rather than the
   graph), or `F6` does nothing. This check belongs here and not at step 3's script, where `paint`
   has no feedback and both the button and the command are correctly inert.*
7. **Step 4, `df`.** Grey field, black at the walls and emitters, brightening outward. *Falsifier:
   uniformly white, meaning `u_jfa` reads black.*
8. **Step 5, `cascade`.** The complete 109-line shader pastes as one block and compiles; runs 6; the
   viewer shows a bright unclamped directional field. The explanation's three parts each point at
   code visible above them without re-quoting it. *Falsifier: the code block is a fragment, which is
   #34 recurring.*
9. **Step 6, `composite`.** Two coloured lights casting soft shadows past the wall, the walls solid.
   This is the finished picture, and it matches the shipped example opened beside it. *Falsifier: the
   maintainer's document and the example differ visibly, meaning a card stated a setting the example
   does not have.*
10. **The cards.** Every one of the six pass steps has a card, every card has all seven rows, and no
   step states a target setting anywhere except in its card. *Falsifier: a settings sentence in the
   prose, which is #27 and #31 recurring.*
11. **Chords.** Every chord the tutorial quotes fires in the app: `Ctrl+Shift+N`, `Alt+P`, `Alt+A`,
    `Alt+R`. The list is derived from the body at walk time, not typed here, so a chord added to the
    prose is walked without this step being edited. `F5` and `F6` are deliberately absent: the
    tutorial quotes neither (`TOGGLE_DOCUMENT_PLAY`'s label is "Play/stop document script", not a
    document transport, and quoting it beside the paint step would confuse the script with the
    render; `F6` is out per F2). *Falsifier: any quoted chord does nothing, meaning the tutorial
    quotes a pre-audit chord.*
12. **Interludes.** The four unnumbered sections each open with "Nothing to build here" and the
    reader never looks for a pass to add in one. *Falsifier: #33's "replace WHERE?" is askable again.*

## Verified / corrected premises

Everything this spec asserts about the repo, opened at `41bce30` (`git show 41bce30:<path>` where the
working tree differs). Six premises the parent stated are corrected.

**The tutorial's source files.** `ai_docs/features/068_radiance_cascades/` holds `01_spec.md`,
`build_tutorial.py`, `oracle.py`, `tutorial_body.html`, `tutorial.html`, and `img/` with six PNGs
named for the six passes. The parent's "`tutorial_body.html` (or whatever source file exists)" hedge
resolves: `tutorial_body.html` is the source and `tutorial.html` is generated by `build_tutorial.py`,
whose only current job is splicing `{{IMG:<name>}}` markers for a hard-coded tuple of six names.

**The example.** Id `77a84d27-2e5b-406d-8011-ee1cb1a9587c`, `ui_name` "Radiance Cascades",
`canvas_size` `[512, 512]`, output `composite`, six passes in `graph.json` insertion order: **paint,
seed, jfa, df, cascade, composite**. That is also the tutorial order. `passes/` holds exactly six
`.frag.glsl` files, no `scripts/` directory, and `document.json` carries no `stopped_uniforms` key.

**The current `iterations`.** `jfa` 9, `cascade` 6, every other pass 1. `jfa.frag.glsl`'s header
comment says "run 9 times" and `document.json`'s description says "the jump flood builds a distance
field in 9 runs, and the cascade stack computes light in 6". All three say 9 and all three change.

**The gear defaults**, from `pass_graph.py`: `DEFAULT_SCALE = 1.0`, `DEFAULT_DTYPE = "f2"`,
**`DEFAULT_FILTER_LINEAR = True`**, `DEFAULT_WRAP = False`, `PassEntry.iterations` default 1,
`DTYPES = ("f1", "f2", "f4")`. The `filter_linear` default being True is what makes five of six cards
carry an unmarked `smooth off`, and it is the fact #27 turns on.

**The starter.** `constants.py::STARTER_EXAMPLE_ID` is `EXAMPLE_ORDER[0]` =
`53724dbd-8efb-4c09-8c7d-28d626a066e7`, "UV Mango", one pass called `main`, `canvas_size`
`[1280, 960]`, `dtype` `f1`, `filter_linear` true. The tutorial's "Before you start" claim that a new
document "arrives with one pass called `main` at 1280x960" is **correct today** and stays correct.

**`u_resolution` is engine-driven.** `core.py::ENGINE_DRIVEN_UNIFORMS` holds `u_time`, `u_aspect`,
`u_resolution`, `u_pass_iteration`, `u_pass_iterations`. So § decision 8's formula has its input
bound with no wiring and no UI row, and dropping `u_pass_iterations` from `jfa` costs nothing.

**No test builds or checks the tutorial today.** `grep -rn tutorial tests/` returns two incidental
mentions: `tests/test_keymap_disjoint.py` cites this feature's folder in a docstring and an error
message, and `tests/test_canvas_presets.py` says the squares are "W-H's first tutorial step". Neither
touches `build_tutorial.py`. `tests/test_tutorial_build.py` is new.

### Corrected premises

**1. The parent says "every mention of the script says `Ctrl+R`". It is `Alt+R` after W-E.**
`02_keybindings.md § The moves` row 5 moves `OPEN_SCRIPT` from Ctrl+R to Alt+R, forced by rule 1
under vim (`u CTRL-R`, the undo tree), and `50_wave_e_keyboard.md § 6`'s table lists the code change
`_chord(K.r, K.mod_ctrl)` → `_chord(K.r, K.mod_alt)`. At `41bce30` `commands.py` still has Ctrl+R,
because W-E is in flight. The parent's W-H bullet and its open question 1 were both written before
the audit ran. **W-H writes `Alt+R`**, and § Tests'
`test_the_tutorial_names_no_chord_the_command_table_does_not_have` pins the tutorial to the registry
rather than to either spelling, so this class of staleness cannot recur.

**2. Four other chords in the tutorial's prose are also stale.** The same audit moves
`NEW_DOCUMENT` Ctrl+N → **Ctrl+Shift+N** (the "Before you start" paragraph quotes Ctrl+N),
`TOGGLE_DOCUMENT_PLAY` Ctrl+Space → **F5**, `OPEN_SHADER` Ctrl+E → **Alt+C**, and adds
`RESET_FEEDBACK` on **F6**. `ADD_PASS` (Alt+A) and `OPEN_PASS_SETTINGS` (Alt+P) are confirmed
unchanged by the audit's own § Where the rule contradicted the parent spec.

**3. The parent's JFA formula reads `u_pass_iterations`; the corrected one must not.** The parent
writes `offset = exp2(ceil(log2(max(u_resolution.x, u_resolution.y))) - 1.0 - u_pass_iteration)`,
which is correct and canvas-derived, and its point is exactly that `u_pass_iterations` drops out.
Today's shader is `pow(2.0, u_pass_iterations - u_pass_iteration - 1.0)`, which is iteration-count-
derived, and that is why raising the count from 9 to 12 under the OLD formula would break 512 rather
than fix 2048 (the first two runs would jump 1024 and 512 texels on a 512 canvas and find nothing).
So the count change and the formula change are one edit and must not be split. Arithmetic verified:
at 512 the offsets run 256 down to 1 over runs 0-8 with runs 9-10 surplus; at 2048 they run 1024 down
to 1 over runs 0-10 with none surplus.

**4. The parent does not mention the surplus-run pass-through's interaction with ping-pong.** Its
wording ("returns its input unchanged when `offset < 1.0`") is right, and the reason it must be a
COPY rather than an early `return` with no write is 068 D5: an iterated self-reading pass swaps its
ping-pong between runs, so a run that writes nothing leaves the target two runs stale. § decision 8
states it.

**5. The parent lists `help_content.py` under W-H's #20 check but not under Files touched.** Its W-H
bullet says "every sentence in the tutorial and the Help panel that mentions the document script says
that `Ctrl+R` (or Script → open) creates it", while its Files-touched line names only the three
tutorial files, the example's `jfa.frag.glsl` and the test. `help_content.py` has exactly one script
sentence (in `your_uniforms`, pointing at "the Script entry point in the Document tab"), and W-G adds
a second about the grammar. Both are in § The prose table and `help_content.py` is in § Files touched.

**6. The parent's #33 fix is stated as two options; this spec picks the one the finding itself names
as honest.** The finding offers the interlude route or making naive GI a real optional pass, and adds
"The first is the honest one given the example ships without it". § Out of scope records the choice
rather than leaving the implementer to re-decide it.

## The prose table

Every sentence that changes. Old text is the current `tutorial_body.html` (or the named file); new
text is what replaces it. Rows marked **moved to 068 spec** leave the tutorial entirely and land in
`068/01_spec.md` under the decision they belong to.

### Before you start

| Old | New |
|---|---|
| "`Ctrl+N` makes a new document from the starter example, so it arrives with one pass called `main` at 1280x960." | "`Ctrl+Shift+N` makes a new document from the UV Mango starter, so it arrives with one pass called `main` at 1280x960." |
| "Rename that pass to `paint` and paste step 1 over it, and set the canvas to 512x512 in the Document tab (the cascade intervals below are tuned for it)." | "Set the canvas to 512x512 first: the Document tab's **presets** menu, beside the width and height fields, has `512x512 (1:1)`. The cascade intervals below are tuned for it. Step 1 renames the starter's pass; steps 2 to 6 each add one." |
| "Each step after this adds one pass: click **add pass**, name it, paste the shader, then open its gear to wire its inputs and set its target." | "Each pass step opens with a **card**: the pass's name, what it reads, and every target setting, with `default` marked where the gear's default already holds. Set the card, paste the shader, done. An input named `u_<pass>` wires itself to the pass of that name, so the card's `reads` rows are usually nothing to do." |
| "The finished thing ships as the **Radiance Cascades** example — open that if you get stuck, or to compare." | unchanged |

### Step 1, `paint`

| Old | New |
|---|---|
| `<h2 id="draw"><span class="step">1</span>A scene to light</h2>` | `<h2 id="paint"><span class="step">1</span><code>paint</code> — the scene to light</h2>` |
| "Add a pass called `paint`:" | "**Rename the starter's pass.** A new document arrives with one pass called `main`. Open its gear (`Alt+P`), type `paint` in the name field, click elsewhere to commit, and replace its shader with the code below. Do not delete it; steps 2 to 6 are the ones that add a pass." |
| the pasted shader body (a hand-copy of `paint.frag.glsl` minus its comment header) | `{{CODE:paint}}` |
| "In the pass's settings: set the format to **16-bit float** and turn **smooth** off. It reads nothing, so it has no inputs to wire." | deleted; the card states all seven rows. |
| the note "**Why 16-bit float, and why not smooth.** An emitter is brighter than white ..." | kept, moved to after the card, retitled "**Why the card says 16-bit float and smooth off.**" — it explains a card row rather than instructing. |
| the warn block "**Why this is not painted with the mouse.** The obvious design ... was built here and **renders black**. Two independent reasons ... the script engine binds to a document's **OUTPUT** pass ... `ctx.mouse` carries position only, no buttons ..." | **moved to 068 spec** (D7, which already records both reasons). The tutorial replaces it with the paint subsection: "**Paint it with the mouse.** The shipped example draws its scene analytically and ships no script, so an exported video is the same every time. Everything below is yours to add on top." + § decision 6's three parts. |

### Interlude, naive GI (was step 2)

| Old | New |
|---|---|
| `<h2 id="naive"><span class="step">2</span>Naive global illumination</h2>` | `<h2 id="naive">Naive global illumination</h2>` (no `step` span) |
| (no such sentence) | new opening: "**Nothing to build here.** This is the idea the cascade pass is an optimisation of; the shipped example has no naive pass." |
| the aside "IF you build this stage ... then delete it happily at step 7" | deleted; the interlude's opening sentence says it once, up front. |
| the illustrative `raymarch()` fragment's `vec4 light = texture(u_scene, vs_uv);` and `vec4 hit = texture(u_scene, uv);` | `u_paint` in both. W-D renames `u_scene` to `u_paint` in every RC shader, and these fragments illustrate a pass that does not exist, so they carry no `{{CODE:}}` marker and the generator cannot reach them. Left alone they teach a uniform name the shipped example does not have, which is #35's complaint recurring in the one place generation does not cover. |

### Step 2, `seed` (was step 3)

| Old | New |
|---|---|
| `<h2 id="seed"><span class="step">3</span>The seed pass</h2>` | `<h2 id="seed"><span class="step">2</span><code>seed</code> — positions, not distances</h2>` |
| "This pass writes the starting state. Add `seed`, wired to **paint**:" | "Add a pass called `seed` (`Alt+A`)." (the wiring is a card row, and `u_paint` wires itself. **This is the one step that names the chord**; steps 3 to 6 read "Add a pass called `x`, as before".) |
| the pasted shader body | `{{CODE:seed}}` |
| the warn "**Set this target to 32-bit float.** It stores coordinates, not colours. At 8-bit a UV quantizes to 1/255 ... at 16-bit it is subtly wrong. This is the one pass where `f4` genuinely earns its cost." | kept as an explanation after the card, retitled "**Why the card says 32-bit float.**" — the imperative goes because the card carries the instruction. |

### Step 3, `jfa` (was step 4)

| Old | New |
|---|---|
| `<h2 id="jfa"><span class="step">4</span>The jump flood</h2>` | `<h2 id="jfa"><span class="step">3</span><code>jfa</code> — the jump flood</h2>` |
| "At 512x512 that is `ceil(log2(512)) = 9` passes — and this is exactly what ShaderBox's **Runs per frame** is for. One shader, nine runs." | "The offset is derived from the canvas, not from the run count: run `i` jumps `exp2(ceil(log2(max side)) - 1 - i)` texels, so the chain is complete after `ceil(log2(max side))` runs and any run past that has nothing left to spread and copies its input forward. The card says **12**, which is `ceil(log2(4096))` and therefore covers every canvas the app can make. One shader, twelve runs." |
| "Wire `u_seed` → **seed**, `u_prev` → **jfa** (itself), format **32-bit float**, smooth **off**, and set **Runs per frame** to **9**." | deleted; the card states all of it. |
| the pasted shader body | `{{CODE:jfa}}` |
| the note "**Why the shader reads two different inputs.** ... run 0 reads the *seed*, and every later run reads what the run before it wrote." | unchanged |
| the note "**The engine hands over the index, not the offset.** ... That is why there is no `u_jfa_offset`" | unchanged |
| the warn "**Resize changes the answer.** 9 runs spans 512px; a 1024px canvas needs 10. Come up short and it still renders — just with a subtly wrong distance field and no error anywhere. Nothing warns you: an engine-side check was built and then retracted, because it cannot tell a base-2 jump flood from the base-4 cascade stack in the next pass, and a check assuming one is wrong for the other. The count is yours to keep right." | replaced by: "**Resize freely.** The offset formula follows the canvas, so the run count covers every canvas the app can make: 12 runs are `ceil(log2(4096))`, and 4096 is the largest side the width and height fields accept. Raise it only if you raise that limit. A short chain would still render, just with a subtly wrong distance field and no error anywhere, which is why the number covers the whole range rather than the size you happen to be using." The retraction history (**moved to 068 spec**, D3, which already records it), and with it the old paragraph's premise: there is no longer a reachable canvas that needs the reader to add a run. |

### Step 4, `df` (was step 5)

| Old | New |
|---|---|
| `<h2 id="df"><span class="step">5</span>The distance field</h2>` | `<h2 id="df"><span class="step">4</span><code>df</code> — one line of real work</h2>` |
| "One line of real work. Add `df`, wired to **jfa**:" | "Add a pass called `df`, as before." |
| the pasted shader body (which differs from the shipped file: the tutorial reformats the ternary across three lines and the shipped file keeps it on two) | `{{CODE:df}}` — the drift this row documents is exactly what D8's generation removes. |

### Interlude, sphere marching (was step 6)

| Old | New |
|---|---|
| `<h2 id="march"><span class="step">6</span>Sphere marching</h2>` | `<h2 id="march">Sphere marching</h2>` |
| "Now the payoff. Replace the naive inner loop's fixed step with a jump by the distance field:" | "**Nothing to build here.** This is the loop the cascade pass runs; you will paste it as part of `cascade.frag.glsl` in step 5, in its `march` function. Compare it against the naive walk above." (#33's "replace WHERE?" answered by naming the file and the function) |
| the loop fragment's `if (dist &lt; EPS) { radiance += texture(u_scene, uv); break; }` | `u_paint`, same reason as the naive-GI fragment above. This one matters most: the new interlude text tells the reader to compare the fragment against `cascade.frag.glsl` by name, so a reader who does finds the illustration and the shipped file disagreeing on the uniform. |

### New interlude, the cascade idea (was the first half of step 7)

| Old | New |
|---|---|
| `<h2 id="cascades"><span class="step">7</span>Radiance cascades</h2>` and its "How a level is packed" `<h3>` | `<h2 id="idea">The cascade idea</h2>`, opening "**Nothing to build here.**", carrying the direction-versus-position trade-off and the packing explanation. The packing CODE fragment is deleted: it reappears in step 5's complete shader. |

### Step 5, `cascade` (was steps 7 and 8)

| Old | New |
|---|---|
| `<h2 id="merge"><span class="step">8</span>The merge</h2>` and the 7-line packing fragment and the ~28-line merge fragment | `<h2 id="cascade"><span class="step">5</span><code>cascade</code> — the cascade stack</h2>`, then `{{CARD:cascade}}`, then `{{CODE:cascade}}` (all 109 lines, #34), then one explanation in three parts headed "Packing", "The march" and "The merge", each referring to the code above. |
| (no such line) | new lead-in: "Add a pass called `cascade`, as before. This is the pass the whole feature is for, and it is the only one longer than a screen. Paste it whole and read the explanation underneath." |
| the SINGLE `CORRECTION` warn block ("**CORRECTION** **Do the bilinear blend by hand, and keep the target ...**") and the lede's "Two corrections to the published code are marked **CORRECTION** where they come up — both were verified numerically, not by eye." | **moved to 068 spec** (§ What reading the primary sources changed, item 1, which already carries the corrected merge and the 30.3% figure). The tutorial's "The merge" explanation keeps the FACT (blend by hand across four probes, sampling the same slot in each, because letting the sampler do it blends different directions) without the CORRECTION label or the article's-bug history. The lede's sentence becomes: "The merge is the step that is easiest to get wrong while still looking right; the explanation in step 5 says how, and `oracle.py` beside this file is what proves it." **The tutorial has exactly ONE `CORRECTION` callout, not two** — the word appears twice, once in this warn block and once in the lede sentence claiming there are two of them, so the lede is wrong in the current tutorial and this row is what fixes it. |

### Step 6, `composite` (was step 9)

| Old | New |
|---|---|
| `<h2 id="composite"><span class="step">9</span>Composite</h2>` | `<h2 id="composite"><span class="step">6</span><code>composite</code> — what you look at</h2>` |
| (the step states nothing about size, format or smooth — #27's sharpest instance, since this is the one pass with smooth ON and 8-bit) | `{{CARD:composite}}` states all seven rows. |
| the pasted shader body | `{{CODE:composite}}` |

### The bookends

Four live sections change and are easy to miss because none of them is a step. Two of them state
facts this wave makes false.

| Old | New |
|---|---|
| the lede: "Build real-time 2D global illumination from nothing, one pass at a time. **Paint light and walls with the mouse**; watch light bounce, mix colour and cast soft shadows ..." | "Build real-time 2D global illumination from nothing, one pass at a time. Build a scene of lights and walls, then paint into it with the mouse in step 1; watch light bounce, mix colour and cast soft shadows ..." The old clause was false in 068, which is why the warn block existed; it becomes true only for a reader who reaches step 1's paint subsection, so the lede says where. |
| the ToC's ten-item `<ol>` mixing passes and concepts, whose item 1 reads "**A canvas you can draw on** — feedback, and why a pass reads itself" and item 4 "The jump flood — **one shader, nine runs**" | six numbered entries, each `<code>name</code> — concept`, **reusing each step's own subtitle verbatim** so the two cannot drift: `paint` — the scene to light, `seed` — positions not distances, `jfa` — the jump flood, `df` — one line of real work, `cascade` — the cascade stack, `composite` — what you look at. The four interludes follow, unnumbered, under an "Along the way" subheading. Item 1's old text is wrong twice over after this wave: `paint` has no feedback (`graph.json`'s `paint.inputs` is `{}`) and its heading is no longer "A canvas you can draw on". |
| "What you built": the table's `jfa` **Runs** cell `<b>9</b>` | `<b>12</b>` |
| "What you built": the table's `paint` **Reads** cell `itself` | `nothing`. `paint` reads no input at all, and this cell is the same per-pass-settings drift #27 and #31 report, surviving in the summary table because nothing generated it. |
| "What you built": "Six shaders, **19 draws a frame**, no noise and no temporal accumulation." | "Six shaders, **22 draws a frame**, no noise and no temporal accumulation." (1 + 1 + 12 + 1 + 6 + 1 = 22 under `iterations: 12`.) |
| "Things to try": "Resize the canvas to 1024 and look at `df` — **nine runs no longer span it**, and nothing tells you; the field just goes subtly wrong at range." | "Raise `jfa`'s runs above 12 and nothing changes: the extra runs copy their input forward. Lower it to 4 and the field goes blocky at range, which is the same experiment from the other side." The old bullet instructs the reader to observe a defect this wave removes, and it contradicts Manual verification step 5, which asks the maintainer to resize and expect the field to stay correct. |
| "Things to try": (no such bullet) | a new bullet carrying the persistent-canvas suggestion the paint subsection defers to it: "Wire `paint`'s `u_prev` to `paint` itself and stop clearing every frame, so the brush accumulates. Now `F6` (Clear canvas) and its `Clear` button have something to do." |

The three "Things to try" bullets not named above (`cascade` runs, output-pass switching, `smooth` on
`cascade`) are unchanged and correct after this wave.

### The table of contents

Covered by the bookends table above.

### Non-tutorial files

| File | Old | New |
|---|---|---|
| `paint.frag.glsl` header, second paragraph | "Built analytically from SDFs and u_time, so it carries no state and needs no script. The engine's script engine binds to a document's **OUTPUT** pass, so a script could not reach a brush uniform declared here anyway -- and a scene that redraws itself every frame is what lets the two lights drift and the shadows follow." | "Built analytically from SDFs and u_time, so it carries no state and the shipped example needs no script: a scene that redraws itself every frame is what lets the two lights drift and the shadows follow." The removed clause states 068 D7's retraction as present fact, and W-G lifts it. This matters more than a stale comment usually would, because `{{CODE:paint}}` splices this header verbatim into step 1, directly above a subsection whose entire content is a script reaching a brush uniform declared in this file. |
| `jfa.frag.glsl` header | "ONE shader, run 9 times. Set 'Runs per frame' to 9 in this pass's settings ..." | "ONE shader, run 12 times. The offset follows the canvas, so 12 runs cover every canvas size the app allows." |
| `jfa.frag.glsl` header | "Each run samples 8 neighbours at a HALVING offset -- 256 texels away, then 128, 64 ... 1 ... After ceil(log2(512)) = 9 runs every texel holds the UV of its nearest solid texel." | "Each run samples 8 neighbours at a HALVING offset derived from the canvas -- at 512 that is 256 texels, then 128, 64 ... 1. After ceil(log2(max side)) runs every texel holds the UV of its nearest solid texel; the runs past that copy their input forward." |
| `jfa.frag.glsl` header | "Resize note: 9 runs spans 512px. A 1024px canvas needs 10, and the pass settings panel warns you when the number no longer reaches -- because a short chain still renders, just wrong." | deleted. The panel does not warn (#6); the offset now follows the canvas, so the note's premise is gone rather than merely wrong. |
| `document.json` description | "the jump flood builds a distance field in 9 runs, and the cascade stack computes light in 6" | "the jump flood builds a distance field in 12 runs, and the cascade stack computes light in 6" |
| `help_content.py`, `your_uniforms` | "document can carry a Python script that drives its uniforms — see the Script entry point in the Document tab" | "a document can carry one Python script that drives its uniforms; `Alt+R` creates and opens it, or use the Script row's open button in the Document tab" |
| `068/01_spec.md` D7 | (W-G appends its "retraction lifted by 069" line) | W-H appends nothing further to D7 but files the tutorial's warn-block text under it as the record of what the tutorial used to say. |
| `068/01_spec.md` D3 | (carries the retraction reasoning already) | gains the tutorial's "Resize changes the answer" paragraph as the record, plus one sentence: "069 W-H made the offset canvas-derived in the shipped shader and set the count to `ceil(log2(MAX_CANVAS_PX))`, so it is now correct at every canvas the app can make rather than at 512 alone." |

## Open questions

Two, each with a robust default already chosen, marked so the maintainer can overrule without
reopening the design.

1. **Does the shipped example gain the brush script?** *Default: no* (§ decision 7). The tutorial
   teaches the script and the example stays analytic, so an export is deterministic and the example a
   reader opens matches the six passes they built. Overruling this means adding a seventh state to
   the example, re-rendering `paint.png`, and accepting that the example's export shows an unpainted
   scene while the live document shows a painted one.

2. **Do the four interludes keep their place in the reading order, or move to an appendix?**
   *Default: keep them inline*, unnumbered, each opening "Nothing to build here." The alternative
   (all four at the end, so steps 1-6 are uninterrupted) makes the build faster to follow and the
   explanation worse: sphere marching read after `cascade` is read after the code it explains. #31
   asks for them "between pass steps", which is what the default does.

## Review history

### Round 2, closure check

All ten round-1 findings confirmed CLOSED, with **one residual on F8**, accepted and fixed.

**The residual.** Round 1's fix specified the instruction predicate as a verb from an enumerated set
appearing in the sentence, and the natural reading of "contains the verb" is a prefix match
(`\badd`). That matches `addresses`, and § The paint script's own explanation of D3 says "the script
**addresses** a pass by NAME" — a descriptive sentence carrying no chord. So the checker as specified
would go red on prose this spec mandates, which is the same shape as the original F8 defect: a test
whose domain is wider than its intent, closing on the implementer at the moment they try to make it
pass.

**The fix**, in § Tests: the match is `\b<verb>\b` with BOTH boundaries, and the inflections are
enumerated explicitly (`adds`, `adding`, `added`, `creates`, `creating`, `created`, and the rest)
rather than left to a prefix match or a stemmer to derive. Verified by running both predicates over
the three sentences that decide it: the D3 sentence is asserted on under `\badd` and not under
`\badd\b`; "press Ctrl+R and the script is created" is asserted on under both, so the test keeps its
teeth; and § decision 7's "ships no script, so an exported video is the same every time" is asserted
on by neither, which was round 1's own reason for the predicate.

The boundary rule is now written as part of the specified matcher rather than left implicit, and
relaxing either boundary is called out as a spec edit, so the next reader cannot re-derive the prefix
match as an equivalent simplification.

### Round 1, pre-implementation review (`reviews/wave_h_pre.md`)

One reviewer, correctness & design plus verification & blast-radius. Verdict: parent coverage PASS,
dependencies PASS, prose completeness **FAIL**, and PARTIAL on the generator design, JFA correctness,
paint-script fidelity and test falsifiability. **All ten findings accepted and folded**; none was
rejected and none escalated. What each changed:

| # | Finding | Folded as |
|---|---|---|
| F1 | Four live sections change with no prose-table row: the lede, the ToC, "What you built" (whose table hard-codes `jfa` runs 9 and a false `paint` reads `itself`, and whose "19 draws a frame" is stale), "Things to try" (whose resize bullet instructs the reader to observe a defect this wave removes, contradicting Manual verification step 5) | § The prose table gains `### The bookends` with a row each, plus the new persistent-canvas bullet the paint subsection defers to |
| F2 | The tutorial taught `F6` in step 1, where `paint` has no feedback pass, so `reset_feedback` is a literal no-op and W-G draws no `Clear` button at all | § decision 6 and § The paint script drop `F6` and say why; Manual verification step 3 loses the claim and a new step 6 checks `F6` once `jfa` exists |
| F3 | The two interlude GLSL fragments name `u_scene`, which W-D renames, and no marker or test reaches hand-written code | two prose-table rows renaming them to `u_paint`, plus `test_no_hand_written_fragment_names_an_absent_uniform` |
| F4 | The card's row order (size before format) contradicts the spec's own stated reason for it: the gear draws format first | § decision 2, § The generated card and the card test reorder to name · reads · format · size · smooth · repeat · runs; the checkbox-caption label rule written down |
| F5 | The early return's seed branch was justified by `max side <= 2`, unreachable under `MIN_CANVAS_PX` = 16 | § decision 8 replaces the rationale with the totality argument; the branch itself stays |
| F6 | `iterations: 11` covers the square presets but not `MAX_CANVAS_PX` = 4096, which W-A's own width and height fields reach | **Ruled: 12.** See below |
| F7 | `paint.frag.glsl`'s header states 068 D7's retraction as fact and is spliced verbatim above the subsection that contradicts it | a prose-table row under Non-tutorial files; the file moves out of Files-touched's "Not touched" list |
| F8 | `test_every_script_mention_carries_the_chord` goes red on prose § decision 7 mandates | restated as `test_every_script_instruction_carries_the_chord` over an enumerated instruction-verb set |
| F9 | The prose table moves a "second CORRECTION callout" that does not exist; the tutorial has one, and the lede's "Two corrections" is itself wrong | the two Step 5 rows merged into one that also fixes the lede |
| F10 | Manual step 10 lists `Alt+A` and `F5`, which the rewritten tutorial quotes nowhere, and `F5`'s command is "Play/stop document script" rather than a transport | step 10 lists only the chords the body quotes and says the list is derived at walk time; `Alt+A` is added to step 2's lead-in explicitly |

### The F6 ruling: `iterations` is 12, a deviation from the parent's 11

`01_spec.md § W-H` says "the shipped `iterations` becomes 11 (correct through 2048, the largest W-A
preset)". **This wave ships 12 instead, deliberately.** The parent derived its number from the preset
LIST, and the preset list is not the reachable set: W-A shipped free-form width and height fields
beside the presets combo, both committed through `clamp_canvas_size`, whose ceiling is
`MAX_CANVAS_PX` = 4096. `ceil(log2(4096))` is 12. A reader who types 4096 into one field has edited
no JSON and broken no rule, and at 11 runs their distance field would be one run short with no
warning anywhere, which is precisely the failure #6 exists to close.

The cost of the correction is three surplus runs at 512 rather than two, each a pass-through draw in
a pass that already runs nine times. That buys a count that is right at every canvas the app can
make rather than at four sizes, so the number stops being a thing the reader has to re-derive when
they resize.

Two consequences recorded so they are not re-derived: the "resize past 2048 and add a run" paragraph
the parent asks for becomes "the run count covers every canvas the app can make; raise it only if you
raise the clamp", since there is no longer a reachable canvas that needs a raise; and
`test_the_jfa_run_count_covers_every_reachable_canvas` asserts against `MAX_CANVAS_PX` as well as the
preset list, naming the clamp in its failure message, so raising either bound forces the count to be
reconsidered.
