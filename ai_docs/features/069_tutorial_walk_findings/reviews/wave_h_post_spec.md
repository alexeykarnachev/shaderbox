# 069 W-H — post-implementation review: spec fidelity and prose

Commit under review: `fdc7841` ("069 W-H: generate the tutorial from the example").
Anchors: `80_wave_h_tutorial.md` (rounds 1 and 2, PASS), `01_spec.md § W-H / D8 / D9 / Open
questions 1`, `00_findings.md` rows 1, 6, 8, 20, 27, 31, 33, 34, 35, and the landed app.

## Verdict

| Dimension | Verdict |
|---|---|
| Wave-spec fidelity | **PASS** — every design decision landed; three deviations, two of them improvements, all unrecorded |
| Parent fidelity | **PASS** — every W-H bullet satisfied; the 12-vs-11 deviation is recorded and correct |
| Findings closure | **PASS** — all nine closed by a named tutorial line |
| App-noun fidelity | **PARTIAL** — one control name the tutorial instructs the reader to set does not exist in the gear |
| Prose coherence | **PASS** — one sentence directs the reader at a control that is not openable |

The wave is sound. The generation is real, the cards are derived, the tests bite, and the
maintainer's nine complaints are each answered by a line I can point at. Two findings below
would be met during the walk; the rest are records that need updating.

## Coverage

### Design decisions

| # | Decision | Status | Evidence |
|---|---|---|---|
| 1 | Three-marker templater; body holds prose and markers only | **Landed** | `build_tutorial.py` `_card_html` / `_code_html` / `_data_uri`; `build()` splices cards and code before images, over `graph["passes"]` rather than a typed tuple; `EXAMPLE_ID` / `EXAMPLE_DIR` as specified; no `import shaderbox` |
| 2 | Card row set in the gear's draw order | **Landed** | Every card emits name · reads · format · size · smooth · repeat · runs. Verified against `popups/pass_settings.py`: `separator_text("Reads")` then `separator_text("Draws into")` with `label_row` "format", "size", "sampling", "edges", then `separator_text("Runs")` with `label_row("runs")`. `smooth` / `repeat` are the checkbox captions, not the `sampling` / `edges` label column, as decision 2 mandates |
| 2 | `u_prev` parenthetical generated from `inputs[uniform] == name` | **Landed** | `_reads_html` emits `(itself, last run)` for `iterations > 1`, `(itself, last frame)` otherwise; both branches written |
| 3 | Five defaults + `_DTYPE_LABELS` duplicated, one test pins them | **Landed** | Constants present; `test_the_generator_defaults_match_the_engine` compares all five plus `set(_DTYPE_LABELS) == set(DTYPES)`. `_DTYPE_LABELS` values match `_FORMATS`' labels in `pass_settings.py` character for character |
| 4 | Six pass steps, four interludes, two bookends | **Landed, with a deviation** | Six `<h2>` carry `<span class="step">` 1..6, headings are the pass names; `naive`, `march`, `idea` are unnumbered and open "Nothing to build here". **`verify` is a fifth such section** — see finding 3 |
| 4 | Per-step body order identical for all six | **Landed** | Machine-checked: all six are CARD → produces → figure → CODE, picture before code as decision 4 requires |
| 4 | The merge folded into step 5's explanation | **Landed** | `<h3>Packing` / `<h3>The march` / `<h3>The merge` inside step 5, each referring to the code above without re-quoting it |
| 5 | Step 1 renames, steps 2-6 add; every step self-sufficient | **Landed** | Step 1: "Rename the starter's pass. … Do not delete it; steps 2 to 6 are the ones that add a pass." Step 2 names `Alt+A`; steps 3-6 read "as before". No step back-references a note |
| 6 | Paint step: uniforms, script, what happens | **Landed** | The three GLSL declarations and the `main()` insert are **byte-identical** to the spec's blocks; the script is **byte-identical** to § The paint script. All six teaching points appear in the surrounding prose |
| 7 | Example keeps no script; the tutorial says so once | **Landed** | `<h3>Paint it with the mouse` opens with the mandated sentence verbatim; no `scripts/` in the example |
| 8 | Canvas-derived offset, surplus pass-through, `u_pass_iterations` dropped, `iterations` 12 | **Landed** | `jfa.frag.glsl` matches the spec's GLSL exactly; the `u_pass_iterations` declaration is gone; `graph.json` `jfa.iterations` 9 → 12; `document.json` description 9 → 12 |
| 9 | Test in `tests/`, imports the generator by path | **Landed** | `tests/test_tutorial_build.py` uses `importlib.util.spec_from_file_location`; `build()` gained an `out` parameter so the `tmp_path` build writes nothing to the tree |
| 10 | `tutorial.html` regenerated and committed | **Landed** | 793 lines changed in the tracked file; a fresh `build()` over a `tmp_path` produces no surviving marker |

### The worked `jfa` card, character by character

The spec's § The generated card prints `u_seed` above `u_prev`. The generated card prints
`u_prev` above `u_seed`:

```
  <tr><td>reads</td><td><code>u_prev</code> from <b>jfa</b> (itself, last run)<br>
      <code>u_seed</code> from <b>seed</b></td></tr>
```

**Judgement: the implementer is right and the spec's sample was wrong.** § decision 2 states the
rule ("`passes[name].inputs`, sorted by uniform name") and § The generated card restates it below
the summary table ("The `reads` rows are sorted by uniform name, which is why `cascade` lists
`u_df` before `u_paint` before `u_prev`"). `u_prev` sorts before `u_seed`. The worked sample and
the summary table's `jfa` row both preserved `graph.json`'s insertion order instead, which is the
one thing the stated rule exists to reject ("JSON key order is an artifact of how the file was
written"). Sorting is what landed, on all six cards, and `cascade` and `composite` match the
spec's own summary rows exactly. Every other value on every other card matches `graph.json`.

### The complete paint script

Byte-compared against § The paint script after HTML-unescaping: **identical**, including the
docstring, the `-> dict` annotation and the trailing comma. The import line
`from shaderbox.scripting import ScriptBehavior, Ctx, Vec2` matches what
`_script_import_line` emits (`ScriptBehavior, Ctx` always, then the referenced output types) for
a document whose only shaped uniform is a `vec2`.

### The JFA change

| Claim | Status | Evidence |
|---|---|---|
| 12 runs | Landed | `graph.json`; `test_the_jfa_run_count_covers_every_reachable_canvas` asserts against both `max(_SQUARE_PRESETS)` and `MAX_CANVAS_PX` |
| The formula | Landed | `exp2(ceil(log2(max(u_resolution.x, u_resolution.y))) - 1.0 - u_pass_iteration)`, matching the parent and the wave spec |
| Pass-through, not no-op | Landed | The early return copies `u_seed` or `u_prev` forward, which per-iteration ping-pong requires |
| The arithmetic | Verified independently | At 4096, offsets 2048 → 1 with none surplus; at 2048 one surplus; at 1024 two; at 512 three; at `MIN_CANVAS_PX` = 16, run 0's offset is 8, so the seed branch is indeed unreachable and written for totality |

Note for the record: § Corrected premises item 3 says "at 512 … runs 9-10 surplus" (two). Three
is correct, and § decision 8, § Review history and the commit message all say three. The stale
half-sentence is in the spec, not in what landed.

### The prose table

| Section | Rows | Status |
|---|---|---|
| Before you start | 4 | All landed. **One sentence was added that no row lists** — finding 2 |
| Step 1 `paint` | 6 | All landed; the warn block moved to 068's spec |
| Interlude, naive GI | 4 | All landed; `u_scene` → `u_paint` in both fragment lines |
| Step 2 `seed` | 4 | All landed |
| Step 3 `jfa` | 6 | All landed; the "Resize changes the answer" warn replaced by "Resize freely" |
| Step 4 `df` | 3 | All landed |
| Interlude, sphere marching | 3 | All landed; the fragment now reads `u_paint` |
| New interlude, the cascade idea | 1 | Landed; the packing fragment deleted, reappearing inside step 5's complete shader |
| Step 5 `cascade` | 2 | Both landed; the CORRECTION callout and the lede's "Two corrections" sentence are gone, the fact kept in the merge explanation |
| Step 6 `composite` | 3 | All landed |
| The bookends | 7 | All landed: the lede, the ToC (six numbered plus an "Along the way" list), `jfa` runs 9 → 12, `paint` reads `itself` → `nothing`, 19 → 22 draws, the resize bullet replaced, the persistent-canvas bullet added |
| Non-tutorial files | 7 | All landed: `paint.frag.glsl`'s second paragraph, three `jfa.frag.glsl` header rows, `document.json`'s description, `help_content.py`'s `your_uniforms`, and the four 068-spec landings |

**The three deleted wiring sentences the table does not list**, judged one by one:

1. `<p>This pass writes the starting state. Add <code>seed</code>, wired to <b>paint</b>:</p>` —
   **listed**, in the Step 2 table's second row.
2. `Wire <code>u_scene</code> → <b>paint</b>, <code>u_df</code> → <b>df</b>, <code>u_prev</code> →
   …` (the `cascade` wiring) — **not listed** as its own row. Correct to delete: the Step 5 row
   replaces the whole of steps 7 and 8 with `{{CARD:cascade}}` plus `{{CODE:cascade}}`, and the
   card carries all three reads. The reader loses nothing; the sentence also named `u_scene`,
   which W-D renamed.
3. `Wire <code>u_light</code> → <b>cascade</b>, <code>u_scene</code> → <b>paint</b>, format …`
   (the `composite` wiring) — **not listed**. Correct to delete for the same reason, and the
   Step 6 table's second row ("the step states nothing about size, format or smooth") anticipates
   the card taking over the step's settings prose. It also named `u_light`, which no shipped
   shader declares after W-D.

All three deletions are right. The table under-specified two of them; nothing wrong landed.

### Tests

| Test | Present | Matches the spec |
|---|---|---|
| `test_every_pass_has_a_card_and_a_code_block` | yes | asserts on the SOURCE body, as specified |
| `test_no_marker_survives_the_build` | yes | builds to `tmp_path`, asserts `"{{" not in html` |
| `test_a_card_states_every_row_and_marks_the_defaults` | yes | asserts the seven labels in order, `jfa`'s format without `dfl`, size and repeat with it, `composite`'s smooth `on` with it |
| `test_a_code_block_is_the_whole_file` | yes | stronger than specified: unescapes and compares the whole file, not only the line count |
| `test_the_code_block_escapes_html` | yes | as specified |
| `test_no_hand_written_fragment_names_an_absent_uniform` | yes | brush allowlist written in, as specified |
| `test_every_script_instruction_carries_the_chord` | yes | `_INSTRUCTION_VERBS` is the spec's tuple verbatim; `_is_script_instruction` uses `\b{verb}\b` with both boundaries |
| `test_no_add_the_script_instruction_anywhere` | yes | covers the body and every `help_sections()` body |
| `test_the_generator_defaults_match_the_engine` | yes | all five plus the `DTYPES` set |
| `test_the_jfa_run_count_covers_every_reachable_canvas` | yes | both assertions, the clamp named in the message |
| `test_the_tutorial_names_no_chord_the_command_table_does_not_have` | yes | every `Ctrl/Alt/Shift+X` and `F1`-`F12` in the body pinned to a live `CommandSpec` |

All eleven pass (`uv run pytest tests/test_tutorial_build.py -q` → 11 passed). `make gates` is
RED on this box, at `tests/test_canvas_fields.py`, with a glfw `get_video_mode` segfault inside
the `app` fixture — a GL-context failure of this sandbox, reproducible on the file alone and
unrelated to anything W-H touched. The gate is not green here and I cannot green it; the
tutorial tests themselves are GL-free and pass.

### Findings closure

| # | The maintainer's words | Closed by | Would it recur on the walk? |
|---|---|---|---|
| 1 | "there is no 512x512 option, where did you find it?" | "Set the canvas to 512×512 first: the Document tab's **presets** menu, beside the width and height fields, has `512x512 (1:1)`." Verified: `_SQUARE_PRESETS` is `(256, 512, 1024, 2048)` and `get_resolution_str(None, 512, 512)` renders exactly `512x512 (1:1)`; the combo is drawn after the two `input_int` fields on the same line | No |
| 6 | "`jfa.frag.glsl:17` … says a short run count makes 'the pass settings panel warn'. No such warning exists" | The paragraph is deleted; the offset is canvas-derived; the count covers `MAX_CANVAS_PX` | No, for `jfa` — **but see finding 1**, which is the same class in the sibling shader |
| 8 | "WE ALREADY HAVE THE DEFAULT FIRST PASS. Should I delete it or what?" | "Do not delete it; steps 2 to 6 are the ones that add a pass", and "Before you start" says "Step 1 renames the starter's pass; steps 2 to 6 each add one" | No |
| 20 | "we don't add this manually. We just hit Ctrl+R" | "Press `Alt+R`: the document's `scripts/script.py` is written for you and opens in a tab, with a stub in it." Two standing guards: the instruction test and the no-"add … script" test over the body and Help | No |
| 27 | "for the seed pass I don't see which resolution … And sampling?" | `seed`'s card states size `100% default` and smooth `off`; all six cards carry all seven rows | No |
| 31 | "when I create a pass I want all its main parameters … in one place" | The card, first thing after every step's lead-in, in the gear's own order, every row present | No |
| 33 | "'Replace the naive inner loop's fixed step …' — replace WHERE?" | "**Nothing to build here.** This is the loop the cascade pass runs; you will paste it as part of `cascade.frag.glsl` in step 5, in its `march` function." | No |
| 34 | "this is not the whole shader code???" | `{{CODE:cascade}}` splices all 109 lines; `test_a_code_block_is_the_whole_file` compares the unescaped block against the file | No |
| 35 | "you call the step 'The merge', but the shader is called 'cascade'" | Step 5's heading is `cascade` — the cascade stack; the merge is an `<h3>` inside it; every step heading is its pass name, and the ToC reuses each subtitle verbatim | No |

### Parent-spec bullets (`01_spec.md § W-H`)

| Parent bullet | Status |
|---|---|
| Template per pass step (heading = pass name, card, produces + picture, complete shader, explanation) | Satisfied, all six identically. The parent's card order lists "size · format"; the wave spec's F4 transposed it to format-before-size to match the gear, and the gear does draw format first (`pass_settings.py` "format" at the `Draws into` separator, "size" after it). The wave's order is right |
| Concept sections become unnumbered interludes; steps 1-6 are the six passes | Satisfied |
| `build_tutorial.py` generates cards and splices code; a test asserts every pass has both and no `{{` survives | Satisfied |
| "Before you start" says 512×512 via the new preset; rename `main` → `paint` (no "Add"); defaults stated once; history/CORRECTION callouts move to the 068 spec | Satisfied; all four 068-spec landings verified in the `01_spec.md` diff |
| Script mentions say `Ctrl+R` | **Superseded, and the supersession is recorded.** The wave spec's § Corrected premises item 1 states the parent's `Ctrl+R` was written before W-E's audit and that W-H writes `Alt+R`; `commands.py` binds `OPEN_SCRIPT` to `_chord(K.r, K.mod_alt)`. The parent's own text still says `Ctrl+R` in both § W-H and § Open questions 1 and points at nothing; the wave spec carries the correction, which is what the review brief allows |
| JFA: canvas-derived offset, `< 1.0` returns its input, `iterations` becomes 11, "panel warns" goes | Satisfied except the count, which is **12 by a recorded deliberate deviation** (§ The F6 ruling, § decision 8, and the commit message each give the reason and the measurement). The deviation is correct: `MAX_CANVAS_PX` is 4096 and `clamp_canvas_size` is what both canvas fields commit through, so 11 leaves a reachable canvas one run short |
| Files | All five named files touched, plus the four the wave spec added (`paint.frag.glsl`, `graph.json`, `document.json`, `help_content.py`) and the roadmap |

**D8** (one template, cards and code generated): satisfied. **D9** (`u_<pass>`, feedback
`u_prev`): every card's reads row matches `graph.json`, and the "Before you start" claim that an
input named `u_<pass>` wires itself is true for every input in the example, `u_prev` included
(`pass_graph.py::_auto_source` returns the consumer for `u_prev`). **Open question 1**: closed as
the parent's default said, positively, by two tests.

### Manual verification and app nouns

Every chord the tutorial quotes is live: `Ctrl+Shift+N` (`NEW_DOCUMENT`), `Alt+P`
(`OPEN_PASS_SETTINGS`), `Alt+A` (`ADD_PASS`), `Alt+R` (`OPEN_SCRIPT`), `F6` (`RESET_FEEDBACK`,
labelled "Clear canvas"). Every gear label the cards use is the gear's own word. The
`512x512 (1:1)` string is what `get_resolution_str` renders. `MouseState` carries `x`, `y`,
`down`, `prev_x`, `prev_y`. The `Clear` ghost button exists over the preview's top-left and is
gated on `document.has_feedback`, which reads the effective graph rather than `_feedback` — so
the tutorial's claim that it "appears only once a pass reads itself" is exactly true. The name
field commits on `is_item_deactivated_after_edit`, so "click elsewhere to commit" is right.

Two nouns do not check out: findings 1 and 2 below.

## Findings

### 1. Step 5 tells the reader to set a control called "Runs per frame". The gear says "Runs".

**Claim.** The tutorial instructs the reader to set a named control that does not exist, which is
the exact defect class #6 reported and this wave exists to close.

**Evidence.** `tutorial.html:721`, inside step 5's `{{CODE:cascade}}` block, which the reader
pastes and reads:

```
// Set &quot;Runs per frame&quot; to 6. Run 0 is the COARSEST level (this shader reverses the index, so
```

The source is `shaderbox/resources/document_examples/77a84d27-…/passes/cascade.frag.glsl:9`.

The app, `shaderbox/popups/pass_settings.py`:

```python
    imgui.separator_text("Runs")

    label_row(app.font_12, "runs", _CTRL_W, _ROW_LABEL_W)
```

`git log -S'Runs per frame' -- shaderbox/popups/pass_settings.py` returns `ccd446b`
("069 W-B: cut UI prose to budget and gate it"), and `git show ccd446b~1:…/pass_settings.py`
line 253 reads `imgui.separator_text("Runs per frame")`. So W-B renamed the control four waves
before this one, and two sites in the RC example still name the old label.

The second site is `document.json:9`, in the string the examples popup shows: "Open a pass
settings to see its Runs per frame." **This commit edited that same string** (9 runs → 12 runs)
and left the stale label one clause later.

This is not a near miss of the wave's own guard. `test_the_tutorial_names_no_chord_the_command_table_does_not_have`
pins chords; nothing pins control labels, and `{{CODE:}}` splicing carries a shipped shader's
prose into the tutorial verbatim — which is precisely why W-H had to hand-edit `jfa.frag.glsl`'s
and `paint.frag.glsl`'s headers. `cascade.frag.glsl`'s header was in the "Not touched, each
checked" list, checked for uniform renames rather than for control names.

**Fix.** In `cascade.frag.glsl` line 9, replace `Set "Runs per frame" to 6.` with
`Set "runs" to 6 in this pass's gear.`, and in `document.json`'s description replace
`Open a pass settings to see its Runs per frame.` with `Open a pass's settings to see its runs.`,
then rerun `build_tutorial.py` and commit the regenerated `tutorial.html`.

### 2. "Then open the Passes strip" names an action the app does not offer.

**Claim.** A reader is told to open something that is always open, in the paragraph whose whole
job is getting them set up before step 1.

**Evidence.** `tutorial.html`, the "Before you start" note:

> Step 1 renames the starter's pass; steps 2 to 6 each add one. **Then open the Passes strip.**
> Each pass step opens with a card: …

`shaderbox/widgets/pass_list.py::draw`:

```python
    small_caption(app.font_12, "Passes")
```

`small_caption` is `push_font` + `text_colored` (`ui_primitives.py:657`). There is no
collapsing header, no toggle, no window. `tabs/document.py:306` calls `pass_list.draw`
unconditionally on every Document-tab frame. The strip is a caption over a row of tiles, always
drawn.

The sentence is also **not in § The prose table**. Its "Before you start" row supplies the
replacement text for the third old sentence, and that text runs "Each pass step opens with a
**card**: …" with no "Then open the Passes strip." in front of it. The implementer added a
sentence the table does not carry, and the added sentence is the one that is wrong.

**Fix.** Delete `Then open the <b>Passes</b> strip.` from `tutorial_body.html`'s "Before you
start" note and rerun the build; the surrounding text needs no other change.

### 3. "Verifying it" became a fifth "Nothing to build here" section, unrecorded.

**Claim.** The structure that landed is better than the one specified, and nothing says so.

**Evidence.** § decision 4's table lists `Verifying it` under **bookend**, and manual
verification step 12 reads "The **four** unnumbered sections each open with 'Nothing to build
here'". What landed:

```
-<h2 id="verify"><span class="step">10</span>Verifying it</h2>
+<h2 id="verify">Verifying it</h2>
+<p><b>Nothing to build here.</b> This is how you find out whether what you built is right.</p>
```

and the ToC lists it as a fourth entry under "Along the way", beside the three interludes.

The change is right — a numbered step 10 that builds nothing is #31's and #33's complaint
verbatim — but the spec now describes a tutorial with four such sections while the tutorial has
five, so a maintainer walking manual step 12 counts wrong.

**Fix.** In `80_wave_h_tutorial.md`, change § decision 4's `Verifying it` row from `bookend` to
`interlude` and manual step 12's "The four unnumbered sections" to "The five unnumbered
sections".

### 4. Manual verification step 11 says the tutorial quotes no `F6`. It quotes it twice.

**Claim.** The step that checks every chord fires omits one the tutorial actually quotes, so a
maintainer following it does not press `F6` and does not learn whether the sentence is true.

**Evidence.** Manual step 11:

> Every chord the tutorial quotes fires in the app: `Ctrl+Shift+N`, `Alt+P`, `Alt+A`, `Alt+R`.
> … `F5` and `F6` are deliberately absent: the tutorial quotes neither.

`grep -on 'F6' tutorial_body.html` returns lines 253 and 687 — the "There is nothing to clear"
note and the persistent-canvas "Things to try" bullet. Both are **mandated**: § decision 6's
closing paragraph writes the first ("`F6` (Clear canvas) and its `Clear` button appear only once
a pass reads itself"), and § The prose table's bookends row writes the second ("Now `F6` (Clear
canvas) and its `Clear` button have something to do"). So the tutorial is right and step 11's
parenthetical is stale — it survived F2's rewrite of the step, which removed `F6` as an
*instruction* at step 1 while the two *descriptive* mentions stayed by design.

Manual step 6 does exercise `F6`, so the walk is covered; only step 11's list is wrong.

**Fix.** In manual step 11, replace the `F5`/`F6` parenthetical with: "`F6` appears twice
descriptively (the nothing-to-clear note and the persistent-canvas suggestion) and is pressed at
step 6, not here; `F5` is deliberately absent, since `TOGGLE_DOCUMENT_PLAY`'s label is 'Play/stop
document script' rather than a document transport."

### 5. The spec's worked `jfa` card and its summary row disagree with the spec's own sorting rule.

**Claim.** The implementer resolved a contradiction in the spec correctly, and the spec still
carries the losing half, so the next reader of § The generated card sees a card the generator
cannot produce.

**Evidence.** § The generated card prints `u_seed` above `u_prev` in both the text sample and
the HTML sample, and its summary table's `jfa` row reads
"`u_seed` from `seed`; `u_prev` from `jfa` (itself, last run)". Eight lines below the table the
same section states the rule that decides it: "The `reads` rows are sorted by uniform name".
`u_prev` < `u_seed`. The generator sorts; the samples do not.

**The implementer's call is right.** § decision 2's row for `reads` says
"`passes[name].inputs`, sorted by uniform name", the rule is restated with its rationale
(insertion order is "an artifact of how the file was written"), and `cascade`'s summary row
already lists `u_df`, `u_paint`, `u_prev` — which is sorted order, and which the generated card
reproduces exactly. Only `jfa`'s two samples kept `graph.json`'s insertion order.

**Fix.** In § The generated card, swap the two `reads` lines in the text sample and in the HTML
sample, and swap the `jfa` summary row's two entries, so all three show `u_prev` before `u_seed`.

### 6. `document.json`'s description still says "Open a pass settings to see its Runs per frame".

Folded into finding 1's fix, listed separately because it is a different file with a different
reader: the examples popup shows this string to anyone browsing examples, tutorial or no
tutorial. This commit edited the same sentence and left the label.

## False trails

- **The `u_pass_iterations` at the tutorial's line 540** is `cascade`'s, not `jfa`'s; `cascade.frag.glsl` declares it three times and reads it, so the explanation is correct.
- **"Runs per frame" in `jfa.frag.glsl`** — already gone; the prose table's row removed the whole sentence that carried it.
- **The card's `smooth` / `repeat` labels not matching the gear's `sampling` / `edges` column** — deliberate, written down in § decision 2, and right: those are the checkbox captions the reader clicks.
- **Step 6's "and set it as the document's output"** — redundant rather than wrong: `pass_list.py:215` sets every newly added pass as output, so `composite` is already the output by the time the reader reads it. Harmless, and it makes the step self-sufficient for a reader who clicked another tile.
- **`make gates` RED** — a glfw `get_video_mode` segfault in `tests/test_canvas_fields.py`'s `app` fixture, reproducible on that file alone in this sandbox, untouched by W-H.
- **The uncommitted `shaderbox/document.py` and `tests/test_default_wiring.py` in the tree** — a concurrent W-D follow-up, not this commit's.
- **The `20×`, `1364`, `30.3%`, `3.65%` figures** — unchanged by this wave and out of its scope; the wave's prose table does not touch them.
- **`test_every_script_instruction_carries_the_chord` being weaker than "every mention"** — deliberate, specified, and the round-2 residual; the enumerated verb tuple in the test is the spec's tuple verbatim, and both word boundaries are present.

## Coverage statement

Read end to end: the commit's eleven changed files; the generated `tutorial.html` in full (all
1023 lines, images stripped); `tutorial_body.html`; `build_tutorial.py`; `tests/test_tutorial_build.py`;
`80_wave_h_tutorial.md` in full including both review rounds; `01_spec.md § W-H`, D8, D9 and
§ Open questions; the nine `00_findings.md` rows verbatim. Checked against the landed app:
`commands.py` (every `CommandSpec` and its chord), `popups/pass_settings.py` (the gear's draw
order, every separator and label and checkbox caption, `_FORMATS`), `tabs/document.py`
(`_SQUARE_PRESETS`, the presets combo, the width and height fields, the Script row),
`util.py::get_resolution_str`, `widgets/pass_list.py` (the strip, the add path, the context
menu), `ui.py` (the `Clear` ghost button and its gate), `document.py::has_feedback`,
`pass_graph.py` (the defaults, `DTYPES`, `MIN/MAX_CANVAS_PX`, `clamp_canvas_size`,
`_auto_source`), `scripting/__init__.py`, `scripting/context.py`, `scripting/engine.py`
(`script_stub_for`, `_script_import_line`), `help_content.py`, and the example's `graph.json`
plus all six `passes/*.frag.glsl`. Machine-verified: the paint script and both brush GLSL blocks
byte-for-byte against the spec; every card value against `graph.json`; the per-step element order
on all six steps; the ToC subtitles against the step headings; the 22-draw arithmetic; the JFA
offset table at 16, 512, 1024, 2048 and 4096. Ran `uv run pytest tests/test_tutorial_build.py`
(11 passed) and `make gates` (RED at an unrelated GL fixture).

Not verified: the rendered pictures against a running app, which needs the maintainer's walk;
`oracle.py`'s numbers, unchanged by this wave.

---

# Round 2 (closure) — against `7440c1d`

Narrow closure round on `7440c1d` ("069 W-H fixes: gate the generated file too"), read via
`git show 7440c1d:<path>`, with the regenerated `tutorial.html` re-read for app nouns.

## Verdicts

| # | Finding | Verdict | Line |
|---|---|---|---|
| 1 | Step 5 instructs "Runs per frame"; the gear says "runs" | **CLOSED** | `cascade.frag.glsl:9` now reads `// Set "runs" to 6 in this pass's gear.`; `tutorial.html:721` carries it as `Set &quot;runs&quot; to 6 in this pass&#x27;s gear.` |
| 6 | `document.json`'s description carried the same stale label | **CLOSED** | The description now ends "Open a pass's settings to see its runs." `git grep 'Runs per frame' 7440c1d -- shaderbox/ ai_docs/features/068_radiance_cascades/` returns nothing |
| 2 | "Then open the Passes strip" names an action the app does not offer | **CLOSED** | The sentence is gone from `tutorial_body.html` and from the generated file; the "Before you start" note now runs "…steps 2 to 6 each add one. Each pass step opens with a **card**: …" with no seam |
| 3 | `Verifying it` became a fifth "Nothing to build here" section, unrecorded | **CLOSED** | Recorded as **four**, and the count is correct — see the judgement below |
| 4 | Manual step 11 said the tutorial quotes no `F6` | **CLOSED** | Step 11 now reads "`F6` appears twice descriptively (the nothing-to-clear note and the persistent-canvas suggestion) and is pressed at step 6, not here". Verified: the body's only `F6` sites are lines 253 and 687, and manual step 6 is where it is pressed |
| 5 | The worked `jfa` card contradicted the spec's own sorting rule | **CLOSED** | The text sample, the HTML sample and the summary row all show `u_prev` above `u_seed`. Byte-compared: the spec's HTML sample is **identical** to `_card_html("jfa", graph)`'s output in the committed `tutorial.html` |

## The FOUR-vs-five judgement

**The implementer's reading is right, and it is not a dodge.** § decision 4's table carries five
rows marked `interlude`, but one of them, `The merge`, has the body cell "the one trap, folded
INTO step 5's explanation, **not a step**". It is an `<h3>` inside step 5, not an `<h2>` section:
the generated file's `<h2>` list is `paint, naive, seed, jfa, df, march, idea, cascade,
composite, verify, What you built`, and `The merge` is not among them. So the sections that build
nothing are four — `naive`, `march`, `idea`, `verify` — which is exactly the count of
"Nothing to build here" in the body (lines 261, 435, 462, 612).

The amended text does not merely change a number; it names the four and explains the fifth row:
decision 4's counting sentence now reads "six numbered plus four unnumbered sections that build
nothing — Naive global illumination, Sphere marching, The cascade idea and **Verifying it**",
with a parenthetical stating that the table's fifth `interlude` row is a row rather than a
section. Manual step 12 enumerates the same four by name, so a maintainer walking it counts what
the page shows.

## Also landed, verified

Not my findings, but they change the gate this wave leaves behind and I checked them:

- **`test_the_committed_tutorial_is_a_fresh_build`** closes the one real hole in the wave's own
  premise. Before it, the tracked `tutorial.html` had no freshness gate: a body edit that was
  never rebuilt, or an edit to the generated file alone, both left the suite green. Mutation-tested
  here — appending `<!-- drift -->` to the committed file turns it red with the rebuild command in
  the message; restored, and `git status` on the directory is clean.
- **`_DTYPE_LABELS` is now pinned to the mapping**, not the key set, so the one card value that is
  a copied string cannot drift into naming a format the combo does not show.
- **`test_a_card_resolves_the_same_reads_the_engine_does`** drives every subset of each pass's
  stored keys removed and compares against the engine's own `effective_inputs`. This matters for
  D9: an absent key is the preferred on-disk state, and a card built from stored keys alone would
  print `nothing` for an edge the engine binds. The commit message records that the first version
  of this test proved nothing (every sampler carries an explicit key today, so both rules agreed
  trivially and two falsifiers went green) — the subset drive is what gave it teeth.
- **`_BODY_CHORD`** replaced `\w+` after the final `+` with "anything but the closing tag", so
  `Alt+/` — a chord `COMMAND_SPECS` actually binds — is checked rather than skipped.

## App nouns, re-checked against the regenerated file

Every quoted UI string in the whole file (code blocks included) is now `"runs"`, which matches
`pass_settings.py`'s `label_row(app.font_12, "runs", …)` under `separator_text("Runs")`. The
lowercase bolded nouns in the prose are `presets` and `smooth`, both verified in round 1 against
`tabs/document.py`'s presets combo and the gear's checkbox caption. The five distinct chords the
body quotes — `Ctrl+Shift+N`, `Alt+P`, `Alt+A`, `Alt+R`, `F6` — are each a live `CommandSpec`
default. No noun was introduced that the app does not have.

`uv run pytest tests/test_tutorial_build.py` → **13 passed**.

## Overall

**PASS.** All six findings closed, each by a line I can point at, and none by weakening a check.
The three fixes that touch shipped source (`cascade.frag.glsl`, `document.json`, the deleted
sentence) are the minimum change that removes the symptom. The freshness gate added alongside
closes the wave's own remaining drift path, which was the premise the whole wave rests on: the
generated file is now provably what the generator produces.

The only thing left is the maintainer's walk (`80_wave_h_tutorial.md § Manual verification`),
which no review can stand in for. `make gates` remains RED on this box at
`tests/test_canvas_fields.py`'s glfw `get_video_mode` segfault, unrelated to W-H and unchanged by
this commit.
