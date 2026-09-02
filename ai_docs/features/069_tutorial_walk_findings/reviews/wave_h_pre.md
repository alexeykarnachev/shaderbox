# 069 W-H pre-implementation review (correctness & design + verification & blast-radius)

Anchor: `80_wave_h_tutorial.md` (948 lines) against the repo at `2b43f83`, the parent
`01_spec.md § W-H` / D8 / D9 / open question 1 / Manual verification, `00_findings.md` rows
1 6 8 20 27 31 33 34 35, and the locked siblings `20_wave_a_canvas_viewer.md`,
`70_wave_d_wiring_naming.md`, `60_wave_g_scripting.md`, `02_keybindings.md`.

## Verdict

| Dimension | Verdict |
|---|---|
| Parent coverage (§ W-H bullets, D8, D9, the nine findings) | **PASS** |
| Generator design (card model, marker splice, body structure) | **PARTIAL** — the card's row order contradicts its own stated rule (F4); nothing else structural |
| JFA correctness (formula, pass-through, ping-pong, arithmetic) | **PARTIAL** — the formula, the copy-forward and the 512/1024/2048 arithmetic are all correct and verified; the degenerate-branch rationale is wrong (F5) and 4096 is reachable (F6) |
| Paint script fidelity to W-G | **PARTIAL** — the script itself is exactly right against W-G § 2, § 9, § 10; the `F6` claim wrapped around it is false (F2) |
| Prose completeness (every changed sentence in the table) | **FAIL** — four live sections change and have no row: the lede, the ToC, "What you built", "Things to try" (F1); plus the interlude code fragments (F3) and `paint.frag.glsl`'s header (F7) |
| Test falsifiability | **PARTIAL** — nine of ten falsifiers go red as claimed; `test_every_script_mention_carries_the_chord` goes red on the spec's own mandated sentence (F8) |
| Dependencies stated | **PASS** — W-D and W-G named as blocking with a mechanical reason and a build-test guard; the wave table is stale about W-E but harmlessly so (T3) |

Ten findings, six of them things a reader walking the tutorial hits directly.

---

## Findings

### F1 (severity: highest). Four live tutorial sections change under this wave and have no prose-table row; two of them state facts the wave makes FALSE.

**Claim.** § The prose table opens "Every sentence that changes." Four sections change and are absent
or waved through.

**Evidence**, all from the current `tutorial_body.html` at `2b43f83`:

1. **"What you built"** (`:648-660`), which § decision 4 lists as a bookend "unchanged in kind" and
   the prose table gives no row. Its table hard-codes `jfa` **runs 9** (`<td><b>9</b></td>`), which
   § decision 8 changes to 11; and its `paint` row reads `<td>itself</td>` under **Reads**, which
   is wrong today (`graph.json`'s `paint.inputs` is `{}`) and is the exact per-pass-settings drift
   #27 and #31 report. Its closing paragraph says "Six shaders, **19 draws a frame**"; at
   `iterations: 11` the count is 1+1+11+1+6+1 = **21**.
2. **"Things to try"** (`:664-672`), same status. Its last bullet reads "Resize the canvas to 1024
   and look at `df` — **nine runs no longer span it**, and nothing tells you; the field just goes
   subtly wrong at range." Under § decision 8 the offset follows the canvas and 11 runs DO span
   1024, so this bullet instructs the reader to observe a defect the wave removes. Manual
   verification step 5 asks the maintainer to do exactly this resize and expect the field to stay
   correct, so the tutorial and the verification step contradict each other in the same document.
3. **The lede** (`:70-72`): "Build real-time 2D global illumination from nothing, one pass at a
   time. **Paint light and walls with the mouse**; watch light bounce…". That sentence was false in
   068 (which is why `:179-187`'s warn block exists) and becomes true only for a reader who reaches
   step 1's new paint subsection. The prose table's only lede row is the CORRECTION sentence.
4. **The table of contents** (`:85-97`). § The prose table's `### The table of contents` row says
   "a ten-item `<ol>`… → six numbered pass entries…" and stops there, so the entries' own
   descriptions are unspecified. Item 1 today reads "**A canvas you can draw on** — feedback, and
   why a pass reads itself", which is wrong twice over after this wave: `paint` has no feedback
   (`graph.json` `paint.inputs` is `{}`) and the new heading is "`paint` — the scene to light".
   Item 4 reads "The jump flood — **one shader, nine runs**".

**Fix (paste-able).** Add to § The prose table, under a new `### The bookends` subsection: the
"What you built" table's `jfa` runs cell becomes `11`, its `paint` **Reads** cell becomes `nothing`,
and the "19 draws a frame" clause becomes "21 draws a frame"; "Things to try"'s resize bullet
becomes "Resize the canvas past 2048 and look at `df` — 11 runs no longer span it, and the number is
yours to raise"; the lede's "Paint light and walls with the mouse" becomes "Build a scene of lights
and walls, then paint into it with the mouse in step 1"; and the ToC's ten items are replaced by the
six pass entries `<code>name</code> — concept` reusing each step's own subtitle verbatim, with the
four interludes unnumbered under "Along the way".

---

### F2 (severity: high). The tutorial teaches `F6` as the way to clear the brush, but W-G hides the button and makes the command a no-op on the document the tutorial builds.

**Claim.** § The paint script's closing paragraph ("`F6` (Clear canvas) resets the feedback
history") and Manual verification step 3 ("`F6` clears") both fail on the reader's document.

**Evidence.**
- `Document.reset_feedback` (`shaderbox/document.py:357-367`) releases `self._feedback`'s canvases
  and clears the generation map. `self._feedback` is filled only by `_feedback_canvas`, which
  `render()` calls only for an input whose `source_name == name` (`document.py:476-478`). The
  tutorial's `paint` at `iterations: 1` reads nothing and reads no self-input, so `_feedback` is
  empty and `reset_feedback` is a literal no-op: **nothing on screen changes.**
- `60_wave_g_scripting.md § 11` states the button "is drawn only when the current document has at
  least one feedback pass", computed as `bool(plan_passes(self.graph)[0].feedback)`. The tutorial's
  document has no feedback pass through step 4 (`jfa` is the first), so at the point step 1 teaches
  `F6` there is **no `Clear` button on the preview at all**, and W-G's own open question 4 confirms
  hidden-not-disabled is the taken default.
- The W-H spec half-notices this ("The third is the one that needs saying, because `paint` at
  `iterations: 1` does not accumulate") but then keeps `F6` in the three-fact result and in a
  falsifiable manual step, where "*Falsifier: hover paints…*" implies `F6` clearing is a checkable
  pass condition.
- `RESET_FEEDBACK` is also absent from `COMMAND_SPECS` at `2b43f83` (verified: `commands.py:100-183`
  has no such entry), so Manual verification step 10's `F6` cannot fire until W-G lands. That part
  is only an ordering note, not a defect.

**Fix (paste-able).** In § The paint script, replace the `F6` sentence with: "There is nothing to
clear: `paint` redraws from scratch every frame, so releasing the button leaves the analytic scene
and the next frame has no brush in it. `F6` (Clear canvas) and its `Clear` button appear only once a
pass reads itself, which is why they belong to the persistent-canvas suggestion in Things to try
rather than to this step." In § Manual verification step 3, drop "`F6` clears" and replace the
observation with "releasing the button leaves the analytic scene with no brush residue", and move
the `F6` check to a step that runs after `jfa` exists.

---

### F3 (severity: high). The two interludes carry hand-written GLSL fragments naming `u_scene`, which W-D renames, and no marker or test covers them.

**Claim.** § The prose table's interlude rows change only the headings, the opening sentence and one
deleted aside. The code inside them is untouched and goes stale under W-D.

**Evidence.** Inside the naive-GI interlude and the sphere-marching interlude the fragments read:

- `tutorial_body.html:198` `vec4 light = texture(u_scene, vs_uv);`
- `:212` `vec4 hit = texture(u_scene, uv);`
- `:458` (sphere-march fragment) `if (dist &lt; EPS) { radiance += texture(u_scene, uv); break; }`

`70_wave_d_wiring_naming.md § 9`'s rename table moves `u_scene` → `u_paint` in every RC shader and
in `graph.json`. These fragments illustrate a pass that does not exist, so they carry no
`{{CODE:x}}` marker; `test_no_marker_survives_the_build` and
`test_a_code_block_is_the_whole_file` both operate on generated blocks and see nothing here. A
reader who compares the sphere-march interlude against `cascade.frag.glsl` (which the new interlude
text tells them to do by name) finds the interlude saying `u_scene` and the shipped file saying
`u_paint`, which is #35's naming complaint recurring in the one place the generator cannot reach.

**Fix (paste-able).** Add two rows to § The prose table's interlude sections: in the naive-GI
interlude's illustrative `raymarch()` fragment and in the sphere-marching interlude's loop fragment,
every `u_scene` becomes `u_paint` and every `u_df` stays, so the illustration reads in the same
vocabulary as the pass it points at; and add a test
`test_no_hand_written_fragment_names_a_renamed_uniform` asserting that no `<pre><code>` block in
`tutorial_body.html` outside a `{{CODE:}}` marker contains a `u_` token absent from every shipped
`passes/*.frag.glsl`.

---

### F4 (severity: medium). The card's row order is not the gear's order, contradicting the spec's own stated reason for choosing it.

**Claim.** § decision 2 says "The order is the parent's, and it is the order the gear draws its
controls in, so a reader filling the gear reads the card top to bottom." The gear draws format
before size; the card puts size before format.

**Evidence.** `shaderbox/popups/pass_settings.py` at `2b43f83`, in draw order:
`separator_text("Pass")` `:78` → `label_row(… "name" …)` `:94` → `separator_text("Reads")` `:134` →
per-input rows `:145` → `separator_text("Draws into")` `:163` → `label_row(… "format" …)` **`:165`**
→ `label_row(… "size" …)` **`:177`** → `label_row(… "sampling" …)` `:194` with the checkbox
`smooth` `:195` → `label_row(… "edges" …)` `:199` with the checkbox `repeat` `:200` →
`separator_text("Runs")` `:221` → `label_row(… "runs" …)` `:223`.

So the gear is name · reads · **format · size** · smooth · repeat · runs, and the card in
§ decision 2 and in § The generated card is name · reads · **size · format** · smooth · repeat ·
runs. This is post-W-B (the roadmap banner records W-B as landed), so it is the final order. It is a
small defect with a real cost: the card exists to be read top-to-bottom while filling the gear, and a
reader doing that transposes two adjacent rows on all six cards.

The row LABELS also diverge: the gear's label column reads `sampling` and `edges` (the words
`smooth` and `repeat` are the checkbox captions beside them). The card's `smooth` / `repeat` labels
match the captions, which is defensible, but the spec should say so rather than leaving an
implementer to "match the gear" and pick the label column.

**Fix (paste-able).** In § decision 2 and § The generated card, swap the `size` and `format` rows so
the order is name · reads · format · size · smooth · repeat · runs, matching
`pass_settings.py`'s draw order (`format` at the `Draws into` separator, then `size`); and add one
sentence: "The card's `smooth` and `repeat` labels are the gear's CHECKBOX captions, not its label
column's `sampling` / `edges`, because the checkbox is the control the reader clicks."

---

### F5 (severity: medium). The JFA early-return's `u_pass_iteration < 0.5` branch is justified by a canvas size the clamp cannot produce.

**Claim.** § decision 8's second bullet says the seed branch is "for the degenerate case of a canvas
so small that run 0 is already surplus (`max side <= 2`, reachable at the 16px clamp floor)". Run 0
is never surplus at any reachable canvas.

**Evidence.** `pass_graph.py:51` `MIN_CANVAS_PX: int = 16` and `clamp_canvas_size` (`:55-60`) floors
both dimensions at 16. At `max side = 16`, `ceil(log2(16)) = 4` and run 0's offset is
`exp2(4 - 1 - 0) = 8`, which is `>= 1`. Computed across the reachable range: run 0's offset is
`2^(ceil(log2(max side)) - 1)`, which is `>= 8` for every `max side >= 16`. `max side <= 2` is
therefore unreachable, and the parenthetical's own evidence contradicts its own claim.

The branch is still CORRECT and should stay — it is what makes the early return total rather than
partial — but the stated reason is false, and a false reason in a locked spec is what a later reader
deletes the branch on.

**Fix (paste-able).** Replace the parenthetical with: "The `u_pass_iteration < 0.5` branch is not
reachable at any canvas the clamp permits (`MIN_CANVAS_PX` is 16, so run 0's offset is at least 8);
it is written because the early return must be total — a reader who sets `iterations` to 1 on a pass
whose first run is surplus would otherwise get an unwritten target, and a branch that is right for
one case and absent for the other is the shape that produces a wrong render the day the bound moves."

---

### F6 (severity: medium). `iterations: 11` is correct at every square PRESET and incorrect at canvas sizes W-A's own fields reach.

**Claim.** The spec repeatedly frames 11 as "correct through 2048, which is the largest square in
W-A's `_SQUARE_PRESETS`", and the test bounds `iterations` by
`ceil(log2(max(_SQUARE_PRESETS)))`. W-A shipped free-form width and height fields alongside the
presets combo, and those reach 4096.

**Evidence.** `shaderbox/tabs/document.py:152-184` draws two `imgui.input_int` fields (`##canvas_w`,
`##canvas_h`) committed through `_apply_canvas_size` → `clamp_canvas_size`, whose ceiling is
`MAX_CANVAS_PX = 4096` (`pass_graph.py:52`). `ceil(log2(4096)) = 12`, so a reader who types 4096 —
one field entry, no JSON edit, D2-compliant — gets a chain that is one run short with no warning,
which is the exact failure mode #6 exists to close.

The spec's prose does cover it once (the replacement "Resize past 2048 and add a run" paragraph), so
this is not a hole in the teaching; it is a hole in the TEST bound and in the framing. Bounding by
`max(_SQUARE_PRESETS)` pins the assertion to the preset list, which the spec argues for, but the
preset list is not the reachable set any more.

**Fix (paste-able).** Change `test_the_jfa_run_count_covers_every_square_preset` to
`test_the_jfa_run_count_covers_every_reachable_square` and assert
`graph["passes"]["jfa"]["iterations"] >= math.ceil(math.log2(max(_SQUARE_PRESETS)))` alongside a
second assertion naming `MAX_CANVAS_PX` in its failure message, so raising either the preset list or
the clamp ceiling forces the count to be reconsidered rather than only the preset list; and reword
§ decision 8's "correct through 2048, the largest square in W-A's `_SQUARE_PRESETS`" to "correct
through 2048, the largest square preset; the width and height fields reach `MAX_CANVAS_PX` (4096),
which needs 12 and is what the resize paragraph tells the reader."

---

### F7 (severity: medium). `paint.frag.glsl`'s header comment states 068 D7's retraction as present fact, is spliced verbatim into step 1 by `{{CODE:paint}}`, and no row in the prose table changes it.

**Claim.** § decision 7 point 3 says "`paint.frag.glsl`'s own comment already explains the analytic
choice, and W-D leaves it alone. Its second paragraph is the one that goes stale
(§ The prose table)." § The prose table has no `paint.frag.glsl` row, and § Files touched lists
`passes/` other than `jfa` under **Not touched, each checked**.

**Evidence.** `passes/paint.frag.glsl:9-12` reads:

> Built analytically from SDFs and u_time, so it carries no state and needs no script. The
> engine's script engine binds to a document's **OUTPUT** pass, so a script could not reach a
> brush uniform declared here anyway -- and a scene that redraws itself every frame is what
> lets the two lights drift and the shadows follow.

`60_wave_g_scripting.md § 2` replaces exactly that binding: a pass block addresses a pass by name and
`paint` is driven while `composite` is the output. So under `{{CODE:paint}}` the tutorial pastes
"a script could not reach a brush uniform declared here anyway" directly above a subsection whose
whole content is a script reaching a brush uniform declared there. That is #33's "replace WHERE?"
failure shape in a new place: the reader is given two contradicting instructions on one screen.
Neither W-D nor W-G touches this comment (verified: `70_wave_d…` names `paint.frag.glsl` only for
the `u_light_radius` prefix trap; `60_wave_g…` names it not at all).

**Fix (paste-able).** Add a `passes/paint.frag.glsl` row to § The prose table's `### Non-tutorial
files`: the second paragraph's "so it carries no state and needs no script. The engine's script
engine binds to a document's OUTPUT pass, so a script could not reach a brush uniform declared here
anyway -- and" becomes "so it carries no state and the shipped example needs no script: a scene that
redraws itself every frame is what"; and move `passes/paint.frag.glsl` out of § Files touched's
"Not touched" list into the touched table.

---

### F8 (severity: medium). `test_every_script_mention_carries_the_chord` goes red on the sentence § decision 7 mandates.

**Claim.** The test is specified as: strip code blocks, then for each remaining occurrence of
"script" (case-insensitive) assert its enclosing sentence contains `Alt+R` or `Script → open`, with
sentences split on `.` `!` `?` `</p>` `</li>`. § decision 7 mandates a sentence that fails it.

**Evidence.** The mandated opening of the paint subsection, quoted verbatim from § decision 7:

> The shipped example draws its scene analytically and ships no script, so an exported video is the
> same every time.

Under the spec's own split rule that is one sentence, it is prose rather than a code block, it
contains "script", and it contains neither `Alt+R` nor `Script → open`. § Out of scope's
"the example ships without the script (§ Design decision 7)" and § decision 6's "3. **What happens.**
… `F6` clears" narration are the same shape. The test therefore cannot be written as specified
without either the wave's own required prose failing it or the implementer quietly weakening it,
and a checker that gets narrowed at implementation time to fit is the failure class the parent's #20
default exists to prevent.

The intent is sound and worth keeping: #20 is about an INSTRUCTION to add a script, not about every
noun. The predicate needs to name that.

**Fix (paste-able).** Restate the test as: strip `<pre><code>` blocks, then for each sentence that
contains "script" AND an imperative verb from a fixed set (`add`, `create`, `make`, `open`, `write`,
`hit`, `press`), assert the sentence contains `Alt+R` or `Script → open`; a sentence mentioning the
script descriptively (no verb from the set) is not an instruction and is not asserted on. Name the
verb set in the spec so it is enumerated rather than invented at implementation time, and keep the
falsifier as "write 'press Ctrl+R and the script is created' and it goes red naming the sentence".

---

### F9 (severity: low). The prose table names a "second `CORRECTION` callout" that does not exist.

**Claim.** § The prose table's Step 5 section has a row reading "the second `CORRECTION` callout and
the lede's 'Two corrections to the published code are marked **CORRECTION**…'".

**Evidence.** `grep -c CORRECTION tutorial_body.html` returns **2**, and both are located: `:80`
inside the lede's `why` block, and `:524` opening the one `warn` block. There is exactly ONE
CORRECTION callout, and the row above it in the same table already moves it. So the row instructs
the implementer to move a block that does not exist, while the lede sentence it also covers is real.
The lede's own "Two corrections" claim is therefore itself wrong in the current tutorial, which is
a small finding the wave gets for free.

**Fix (paste-able).** Merge the two Step 5 rows into one: "the single `CORRECTION` warn block
(`:523-531`) and the lede's 'Two corrections to the published code are marked CORRECTION where they
come up — both were verified numerically, not by eye' → **moved to 068 spec**; the lede's sentence
becomes 'The merge is the step that is easiest to get wrong while still looking right; the
explanation in step 5 says how, and `oracle.py` beside this file is what proves it.'"

---

### F10 (severity: low). Manual verification step 10 lists chords the tutorial has no stated reason to quote, and one whose label the spec misnames.

**Claim.** Step 10 asks the maintainer to fire `Ctrl+Shift+N`, `Alt+P`, `Alt+A`, `Alt+R`, `F5`, `F6`.
Two of the six have no home in the rewritten tutorial as specified, and `F5`'s command is not
"play the document" in the sense a reader would infer.

**Evidence.**
- The prose table's new text quotes `Ctrl+Shift+N` (Before you start), `Alt+P` (step 1's rename),
  and `Alt+R` (the paint subsection and `help_content.py`). It quotes **`Alt+A` nowhere**: the new
  step lead-ins are "Add a pass called `seed`", with no chord. And it quotes **`F5` nowhere**.
- `F6` is covered by F2 above.
- `commands.py:118-123` at `2b43f83`: `TOGGLE_DOCUMENT_PLAY`'s label is **"Play/stop document
  script"**, not a document transport. A tutorial step telling a reader to press F5 to "play" would
  be quoting a command that toggles the SCRIPT, which is a live and confusing distinction on the one
  step where the reader has just written a script.
- This is not a correctness hole in `test_the_tutorial_names_no_chord_the_command_table_does_not_have`
  (that test asserts one direction only, which is the right direction). It is a manual step that
  cannot be performed as written.

**Fix (paste-able).** Replace step 10's chord list with "every chord the tutorial quotes fires in the
app: `Ctrl+Shift+N`, `Alt+P`, `Alt+R`" and add "the list is derived from the body at walk time, not
typed here"; and if `Alt+A` is meant to appear in the step lead-ins, add it to § The prose table's
per-step rows explicitly rather than leaving it to the implementer.

---

## What the spec gets right, demonstrated

These were checked against code rather than accepted, and each holds.

- **`DEFAULT_FILTER_LINEAR` is `True`** (`pass_graph.py:39`), so five of six RC passes carry an
  unmarked `smooth off` and `composite` alone reads `smooth on default`. Verified against
  `graph.json`: `filter_linear` is `false` on paint, seed, jfa, df, cascade and `true` on composite.
  The other four defaults are `DEFAULT_SCALE = 1.0` (`:41`), `DEFAULT_DTYPE = "f2"` (`:38`),
  `DEFAULT_WRAP = False` (`:40`), `PassEntry.iterations` default 1 (`:113`), and
  `DTYPES = ("f1","f2","f4")` (`:33`) — every default the spec states is exact.
- **The six card rows in § The generated card's summary table are correct** against `graph.json`
  read with W-D's renames applied: every `scale` 1.0, dtypes f2/f4/f4/f2/f2/f1, `wrap` false
  throughout, `iterations` 1/1/11/1/6/1. `cascade`'s sorted reads really are `u_df`, `u_paint`,
  `u_prev`.
- **`_DTYPE_LABELS` matches the gear verbatim.** `pass_settings.py:36-40` `_FORMATS` is
  `("f1","8-bit",…), ("f2","16-bit float",…), ("f4","32-bit float",…)`.
- **The ping-pong argument for the copy-forward is exactly right.** `document.py:472-500`: the
  iteration loop calls `_swap_feedback(name)` after every non-last run, and `_swap_feedback`
  (`:343-354`) exchanges `self._feedback[name]` with `render_pass.canvas`. A run that returns
  without writing therefore leaves the target holding the state from two runs earlier and the chain
  goes backwards. A `discard`-style early return would have been a silent corruption.
- **The JFA arithmetic is correct at all four presets.** Computed: `ceil(log2(side))` is 8, 9, 10, 11
  at 256, 512, 1024, 2048; at 512 the offsets run 256 down to 1 over runs 0-8 with 9 and 10 surplus;
  at 2048 they run 1024 down to 1 over runs 0-10 with none surplus. The spec's numbers match.
- **`u_pass_iterations` can be dropped from `jfa` at no cost.** `core.py:37`'s
  `ENGINE_DRIVEN_UNIFORMS` binds it, `core.py:420` writes it only when the program declares it, and
  `cascade.frag.glsl:29,58,81` is the only other reader in the example.
- **The paint script is exactly W-G's contract.** `Vec2.__new__(cls, x, y)` (`outputs.py:76-78`)
  takes two floats; `coerce_one` (`behavior.py`) accepts a `Vec*` through `normalize_output`;
  `_resolve_behavior_class` (`behavior.py`) accepts a class named `Brush` because it falls back to
  the first `ScriptBehavior` subclass. **The import line is verified by running the generator**:
  `script_stub_for([u_brush(vec2), u_brush_down(float)])` emits
  `from shaderbox.scripting import ScriptBehavior, Ctx, Vec2` — exactly the spec's line, in order.
  `MouseState.down` / `prev_x` / `prev_y` are used the way `60_wave_g… § 10` defines them
  (prev is the previous in-bounds FRAME sample, `prev == current` on the first down frame after a
  re-entry, so the first frame stamps a zero-length capsule which the shader's `max(dot(ab,ab),1e-6)`
  guard handles). The `1.0 if ctx.mouse.down else 0.0` conversion is required, not defensive: there
  is no bool in the coercion set.
- **The `graph.json` field access is safe.** `ui_models.py:407` writes
  `self.document.graph.model_dump()` with no `exclude_defaults`, so `iterations` and all four
  `target` keys are always present on every pass; the generator's direct indexing cannot KeyError on
  an app-written file. Confirmed by reading all seven shipped `graph.json` files.
- **`W-A`'s preset is present and the wording is right.** `tabs/document.py:31`
  `_SQUARE_PRESETS = (256, 512, 1024, 2048)`; `_canvas_presets` (`:44-86`) emits
  `get_resolution_str(None, 512, 512)`, which `util.py:69-75` renders as exactly `512x512 (1:1)`;
  and the presets combo does sit beside the width and height fields (`:151-196`). The starter is
  1280x960, so 512x512 is not filtered out by the `seen` set.
- **`test_a_code_block_is_the_whole_file`'s targets check out.** `cascade.frag.glsl` is **109 lines**
  and ends `fs_color = vec4(acc * 0.25, 1.0);\n}`. `jfa.frag.glsl:38` has `x <= 1.0`, so
  `test_the_code_block_escapes_html`'s `&lt;` assertion has a real subject.
- **The prose table's old text is accurate.** Ten rows spot-checked verbatim against
  `tutorial_body.html`: the `Ctrl+N` sentence (`:101`), the 512x512 sentence (`:103`), the
  "Each step after this" sentence (`:104-106`), step 1's heading (`:111`), "Add a pass called
  `paint`:" (`:118`), the "Why this is not painted with the mouse" warn (`:179-187`), the seed
  heading (`:241`) and its lead-in (`:249`), the seed 32-bit warn (`:267-270`), the jfa "nine runs"
  paragraph (`:295-297`), the jfa wiring sentence (`:341-344`), the "Resize changes the answer" warn
  (`:353-359`), the df heading (`:370`) and lead-in (`:372`). Every quotation is exact.
- **The `df` drift the spec documents is real.** The tutorial (`:381-383`) breaks the ternary across
  three lines with an inline comment; the shipped file (`df.frag.glsl:22`) keeps it on one. That is
  precisely the drift D8's generation removes.
- **#20's regex is clean today.** `re.compile(r'add[^.]{0,40}script', re.I)` over
  `tutorial_body.html` and `help_content.py` returns zero matches, so
  `test_no_add_the_script_instruction_anywhere` is a standing guard rather than a red-on-arrival
  test, exactly as the spec says.
- **The `help_content.py` old text is exact.** `:137-139` reads "document can carry a Python script
  that drives its uniforms — see the Script entry point in the Document tab".
- **The two closed open questions: I agree with both.**
  Question 1 (the example gains no brush script): the parent's own § W-G says "the tutorial's paint
  step is written against D3", scoping the script to the TUTORIAL and never to the example, and its
  § W-H bullet lists only the three tutorial files plus `jfa.frag.glsl` and the test under Files.
  The decisive independent evidence is `60_wave_g… § 10`: `EXPORT_MOUSE` keeps `down=False`, so a
  script-driven example renders one picture live and a different one in `make smoke` and in every
  export, and the example is rendered by both. § decision 7's ordering of the three reasons is right
  as well: D6's "the example is the finished artifact" is the one that would hold even if export
  were deterministic.
  Question 2 (interludes stay inline): #31 asks for them "between pass steps" in as many words, and
  the alternative puts sphere marching after the code it explains. No disagreement.
- **The dependency gate is real, not hoped-for.** Building against un-renamed files would emit a
  `u_scene` card, and `test_a_card_states_every_row_and_marks_the_defaults` plus D9's naming
  assertion catch it. The spec is right that the guard is mechanical.

## False trails (checked, not findings)

- `tests/test_tutorial_build.py` importing `_SQUARE_PRESETS` from `tabs/document.py` pulls
  `shaderbox.app` and glfw — but `env -u DISPLAY uv run python -c "from shaderbox.tabs.document import _SQUARE_PRESETS"` succeeds, and `tests/test_canvas_presets.py:34` already imports from that module. No display is needed for the import.
- The generator KeyError-ing on a `graph.json` that omits `iterations` or a `target` key: impossible
  for an app-written file (`model_dump()` with no `exclude_defaults`), verified across all seven
  shipped graphs.
- `parents[3]` in § decision 1 resolving to the wrong root: `ai_docs/features/068_radiance_cascades/build_tutorial.py` → parents[0] `068_radiance_cascades`, [1] `features`, [2] `ai_docs`, [3] repo root. Correct.
- The spliced `{{CODE:cascade}}` block being 109 lines of `<pre>` inside a `<figure>`-adjacent flow:
  the existing body already pastes complete files this way for five of six passes; nothing new.
- The `u_prev` "(itself, last frame)" branch being dead code: it is unexercised by the shipped
  example but the spec says so explicitly and writes it anyway, which is the right call.
- `_card_html` sorting reads by uniform name rather than `graph.json` key order: deliberate, argued,
  and the RC example's key order happens to agree for every pass except `cascade` (`u_paint` before
  `u_df` on disk, `u_df` before `u_paint` sorted). No defect.
- The spec's wave table calling W-E "in flight (uncommitted at `41bce30`)": W-E landed at `2b43f83`
  (`commands.py:125` is `_chord(K.r, K.mod_alt)`, `:110` is `_chord(K.n, K.mod_ctrl, K.mod_shift)`,
  `:121` is `_chord(K.f5)`, `:124` is `_chord(K.c, K.mod_alt)`). Every chord the spec's § Corrected
  premises 1 and 2 predicts is now in the registry, so the prediction was right and the table is
  merely stale about the status. Not worth a spec edit.
- `Alt+P` opening the gear on the starter's single pass: `OPEN_PASS_SETTINGS` is
  `_chord(K.p, K.mod_alt)` (`commands.py:171-175`) and `App.panel_pass` falls back to the output
  pass, so step 1's rename instruction works on a one-pass document.

## Coverage statement

Read end to end: `80_wave_h_tutorial.md` (all 948 lines); `01_spec.md` §§ Locked decisions, W-G, W-H,
Order, Files touched, Manual verification, Open questions; `00_findings.md` rows 1, 6, 8, 20, 27, 31,
33, 34, 35 verbatim plus the ledger's header; `60_wave_g_scripting.md` §§ 2, 9, 10, 11, Manual
verification, Open questions, Verified premises; `70_wave_d_wiring_naming.md` §§ 9, 10, 11;
`ai_docs/features/068_radiance_cascades/tutorial_body.html` (all 674 lines) and `build_tutorial.py`;
`dev_flow.md` §§ make gates / make check / make test / make smoke.

Code opened at `2b43f83`: `shaderbox/pass_graph.py` (defaults, `clamp_canvas_size`, `PassEntry`),
`shaderbox/document.py` (`render`'s iteration loop, `_swap_feedback`, `reset_feedback`,
`_feedback_canvas`), `shaderbox/tabs/document.py` (`_SQUARE_PRESETS`, `_canvas_presets`, the canvas
fields), `shaderbox/popups/pass_settings.py` (`_FORMATS`, the gear's full draw order),
`shaderbox/commands.py` (`COMMAND_SPECS` in full), `shaderbox/core.py:37`
(`ENGINE_DRIVEN_UNIFORMS`), `shaderbox/scripting/{behavior,context,engine,outputs}.py`
(`ScriptBehavior`, `MouseState`, `script_stub_for`, `_script_import_line`, `coerce_one`, `Vec2`),
`shaderbox/app.py::open_script_for`, `shaderbox/ui_models.py:406-407`, `shaderbox/help_content.py`,
`shaderbox/util.py::get_resolution_str`, all six RC `passes/*.frag.glsl`, the RC `graph.json` and
`document.json`, and all seven shipped `graph.json` files.

Executed rather than reasoned: the offset table at 16/256/512/1024/2048 for `iterations` 11; the
`add…script` regex over the tutorial body and `help_content.py`; `script_stub_for` against a fake
vec2+float uniform set, to read the emitted import line; a headless import of `_SQUARE_PRESETS`;
`wc -l` on every RC shader.

Not covered: the six `img/*.png` renders (§ Out of scope, and the passes they picture are unchanged
except `jfa`, whose picture at 11 runs is identical to 9 because runs 9 and 10 copy forward);
`oracle.py`'s numbers; the CSS the body file gains; W-B's landed prose edits beyond the gear layout.

---

# Round 2 (closure)

Narrow closure round against the folded `80_wave_h_tutorial.md` (1102 lines, up from 948), read at
the working tree over `91a687b`. Scope: did each of F1..F10 close, with the new text cited. The
main-session ruling on F6 (`iterations` 12) is a CONSTRAINT here, not a question, and is checked for
arithmetic and for propagation rather than re-argued.

## Overall: **PASS**

Ten of ten findings closed, one with a narrow residual (F8) that is a defect in the folded predicate
rather than a reopening of the finding. No fold introduced a new inconsistency, and the F6 ruling
propagated to all fourteen sites that carry the number.

| Finding | Verdict |
|---|---|
| F1 four unrowed bookend sections | **CLOSED** |
| F2 `F6` taught where it is inert | **CLOSED** |
| F3 interlude fragments naming `u_scene` | **CLOSED** |
| F4 card row order vs the gear | **CLOSED** |
| F5 unreachable degenerate rationale | **CLOSED** |
| F6 count short of the reachable ceiling | **CLOSED** (ruling verified: 12 covers 16..4096; 512 carries exactly three copy-forward runs) |
| F7 `paint.frag.glsl` header spliced against itself | **CLOSED** |
| F8 script-mention predicate | **CLOSED with a residual** — the predicate accepts every mandated descriptive sentence and rejects the falsifier, but `\badd` matches `address*`, which the spec's own D3 teaching sentence uses |
| F9 the non-existent second CORRECTION | **CLOSED** |
| F10 manual step 10's chords | **CLOSED** |

---

## Per-finding closure

### F1 — CLOSED

§ The prose table gains `### The bookends` (`:1008-1025`) with seven rows. Each of the four sections
I named now has text:

- the lede's "**Paint light and walls with the mouse**" → "Build a scene of lights and walls, then
  paint into it with the mouse in step 1", with the reason recorded ("it becomes true only for a
  reader who reaches step 1's paint subsection, so the lede says where");
- the ToC → six numbered entries **reusing each step's own subtitle verbatim** so the two cannot
  drift, plus the four interludes under "Along the way". The row also names why item 1's old text was
  wrong twice over;
- "What you built"'s `jfa` **Runs** cell `9` → `12`; its `paint` **Reads** cell `itself` → `nothing`,
  with the row explicitly identifying it as #27/#31 drift "surviving in the summary table because
  nothing generated it";
- "19 draws a frame" → "**22 draws a frame**". Recomputed: 1+1+12+1+6+1 = **22**. Correct under the
  12 ruling;
- the "Things to try" resize bullet is replaced by "Raise `jfa`'s runs above 12 and nothing changes:
  the extra runs copy their input forward. Lower it to 4 and the field goes blocky at range", which
  is the same experiment stated in terms the wave makes true, and the row names the contradiction
  with Manual verification step 5 that forced it;
- a NEW "Things to try" bullet carries the persistent-canvas suggestion, which is what F2's fold
  defers to. The two folds are wired to each other rather than left dangling.

The section closes by naming the three bullets it did NOT change and asserting they stay correct,
which is the completeness claim the original table lacked. `### The table of contents` (`:1026-1028`)
now reads "Covered by the bookends table above" rather than carrying a second half-specification.

### F2 — CLOSED

Three coordinated edits.

§ decision 6's part 3 (`:331-332`) drops `F6` from the three-fact result: "LMB paints, hover does
not, and the stroke is a capsule rather than blobs. There is nothing to clear, and § The paint script
says why."

§ The paint script (`:578-596`) replaces the old sentence with the block quote I proposed, and then
adds a paragraph headed "**This corrects the spec's own first draft**" that states the mechanism:
`reset_feedback` releases `_feedback`, `_feedback` is filled only for a self-sourced input, `paint`
at `iterations: 1` has none, and W-G § 11 draws the button only when `plan_passes(graph)[0].feedback`
is non-empty. It ends "Teaching a chord that does nothing, beside a button that is not there, is the
class of defect this whole wave exists to remove", which files the reasoning at the right altitude.

Manual verification loses the claim at step 3 (`:786-791`, now ending "releasing the button leaves
the analytic scene with no brush residue") and gains a **new step 6** (`:802-806`) that checks `F6`
where it is live: "With `jfa` in the document, the `Clear` ghost button is drawn at the preview's
top-left and `F6` fires it; the flood field blanks for one frame and rebuilds", with the falsifier
naming W-G's `has_feedback` gate reading `_feedback` rather than the graph. The step states why it
sits there and not at step 3. That is a stronger close than I asked for: it converts a false check
into a real one for W-G's own gate.

### F3 — CLOSED

Two prose-table rows plus a test.

The naive-GI interlude row (`:947`) renames `vec4 light = texture(u_scene, vs_uv);` and
`vec4 hit = texture(u_scene, uv);` to `u_paint`, with the reason ("they carry no `{{CODE:}}` marker
and the generator cannot reach them"). The sphere-marching row (`:985`) renames the loop fragment's
`texture(u_scene, uv)` and adds the sharper point: "the new interlude text tells the reader to
compare the fragment against `cascade.frag.glsl` by name, so a reader who does finds the illustration
and the shipped file disagreeing on the uniform."

`test_no_hand_written_fragment_names_an_absent_uniform` (`:672-683`) collects every `<pre><code>`
block that is not a `{{CODE:}}` marker, extracts `u_[a-z_]+` tokens, and asserts each appears in some
shipped `passes/*.frag.glsl` or is one of the three brush uniforms. The allowlist is written into the
test with its reason stated ("those three uniforms exist only in the reader's document and never in a
shipped file"), so the exception is enumerated rather than discovered. Falsifier is exact: leave
`u_scene` after W-D's rename and it goes red naming the token.

### F4 — CLOSED, and propagated to every dependent site

§ decision 2's row table (`:168-176`) is now name · reads · **format · size** · smooth · repeat ·
runs, and the spec explains rather than merely asserts: "**Format sits above size** because that is
the order `popups/pass_settings.py` draws them in under the `Draws into` separator" (`:178-180`),
which matches `pass_settings.py:165` (`format`) preceding `:177` (`size`).

The label question I raised is answered in writing (`:182-186`): "The card's `smooth` and `repeat`
labels are the gear's CHECKBOX captions, not its label column's `sampling` and `edges` … written
down here so an implementer told to 'match the gear' does not pick the other one."

Propagation checked at all three dependent sites: the worked `jfa` card's text rendering and its
HTML both put `format` before `size` (`:485-508`); the five-card summary table's column order is
`reads | format | size | smooth | repeat | runs` (`:516-523`) and its per-row values are unchanged
and still correct against `graph.json`; and
`test_a_card_states_every_row_and_marks_the_defaults` (`:642-647`) now asserts "the seven row labels
appear in the gear's order (name, reads, format, size, smooth, repeat, runs)". A fold that changed
the design and not the test would have left the test asserting the old order.

### F5 — CLOSED

§ decision 8's second bullet (`:415-421`) drops the `max side <= 2` claim entirely and substitutes
the totality argument: "The `u_pass_iteration < 0.5` branch is not reachable at any canvas the clamp
permits (`MIN_CANVAS_PX` is 16, so run 0's offset is at least 8); it is written because the early
return must be total." Re-verified: `MIN_CANVAS_PX` is 16 (`pass_graph.py:51`) and run 0's offset at
16 is `exp2(ceil(log2(16)) - 1) = 8`. The branch stays, which is what I asked for; only the false
reason went.

### F6 — CLOSED. Ruling arithmetic verified independently.

I recomputed rather than accepting the drafter's numbers.

**Coverage.** For every integer side from 16 to 4096 inclusive, `ceil(log2(side)) <= 12`. The set of
canvases needing more than 12 runs is **empty**. Spot values: `ceil(log2(16))=4`, `ceil(log2(17))=5`,
`ceil(log2(512))=9`, `ceil(log2(1024))=10`, `ceil(log2(2048))=11`, `ceil(log2(4095))=12`,
`ceil(log2(4096))=12`. So 12 is exactly `ceil(log2(MAX_CANVAS_PX))` and not one more than needed.

**The surplus at 512.** Offsets `2^(9-1-i)` for i in 0..11: 256, 128, 64, 32, 16, 8, 4, 2, 1, then
0.5, 0.25, 0.125. Runs 0-8 are real (offset >= 1); runs 9, 10 and 11 take the `offset < 1.0` branch.
**Exactly three copy-forward runs**, as § decision 8 (`:441-444`) and the ruling (`:1093-1095`) both
state. At 4096 the offsets are 2048 down to 1 with **zero** surplus, also as stated.

**Propagation.** The number moved everywhere it needed to, checked site by site: § decision 8's
heading and body (`:388`, `:430-444`); the worked card's text and HTML and the summary table
(`:481`, `:490`, `:504`, `:519`); § Files touched's `graph.json` and `document.json` rows (`:606`,
`:607`); the prose table's `jfa` step lead-in (`:964`, "The card says **12**, which is
`ceil(log2(4096))`"); the `jfa.frag.glsl` header rows (`:1034-1035`); the `document.json` description
row (`:1037`); the 068 D3 record row (`:1041`); "What you built"'s runs cell and draw count
(`:1018`, `:1020`); Manual verification step 5 (`:797`). Every surviving `11` in the file is a
deliberate reference to the parent's superseded number (`:432`, `:1076`, `:1082`, `:1084`, `:1089`)
or the test falsifier that pins the deviation (`:757`).

**The resize paragraph.** The prose table row (`:970`) now reads "**Resize freely.** The offset
formula follows the canvas, so the run count covers every canvas the app can make: 12 runs are
`ceil(log2(4096))`, and 4096 is the largest side the width and height fields accept. Raise it only if
you raise that limit." That is the ruling's wording, and the row records that the old paragraph's
premise is gone rather than merely restated.

**The test.** `test_the_jfa_run_count_covers_every_reachable_canvas` (`:738-761`) asserts against
both bounds with both imported. Falsifier claims re-derived: 9 fails both (9 < 11 and 9 < 12); **11
passes the first and fails the second** naming `MAX_CANVAS_PX`, which is the gap F6 found; 12 passes
both; raising `MAX_CANVAS_PX` to 8192 makes 12 fail the second. Every falsifier the spec claims is
true. Keeping the preset assertion alongside the clamp assertion is currently redundant (11 <= 12),
and the spec says so and justifies it: "the two bounds can move independently". That is a reason, not
a preference, so it stands.

The deviation is recorded where a future reader will look: § Review history's `### The F6 ruling`
(`:1082-1100`) states the parent's number, the reason it was derived from the wrong set, the cost in
surplus draws, and the two consequences. § decision 8 also flags it inline ("**This is a deliberate
deviation from the parent's 11**"). Nothing in the tree now claims 11.

### F7 — CLOSED

§ The prose table's `### Non-tutorial files` gains a `paint.frag.glsl` header row (`:1033`) with old
and new text. The removed clause is exactly the one I quoted ("The engine's script engine binds to a
document's **OUTPUT** pass, so a script could not reach a brush uniform declared here anyway -- and"),
and the replacement keeps the analytic rationale ("so it carries no state and the shipped example
needs no script: a scene that redraws itself every frame is what lets the two lights drift and the
shadows follow"). The row states the splice consequence that made it urgent.

§ Files touched gains the file (`:605`) with the scope pinned: "Header comment only … **No GLSL
changes.**", and the "Not touched" list (`:614-615`) is amended to "other than `jfa` and `paint`'s
header … no other shader's CODE changes under this wave". The exclusion list and the inclusion list
now agree, which they did not before.

### F8 — CLOSED, with one residual

**Closed.** `test_every_script_instruction_carries_the_chord` (`:685-716`) enumerates
`_INSTRUCTION_VERBS = ("add", "create", "make", "open", "write", "hit", "press")` in the spec, and
asserts only on sentences carrying both `script` and a verb from that set. The spec states why the
set is enumerated here ("a checker whose domain is chosen while making it pass is the failure class
the parent's #20 default exists to prevent. Widening the set is a spec edit") and names the
compatibility case explicitly.

**I ran the predicate** over every script-naming sentence the folded spec mandates for the body:

| Sentence | Asserted on | Result |
|---|---|---|
| "The shipped example draws its scene analytically and ships no script, so an exported video is the same every time." (§ decision 7, `:363-364`) | no | correctly exempt |
| "There is nothing to clear: `paint` redraws from scratch every frame …" (§ The paint script, `:583`) | no (no "script" token) | exempt |
| "`F6` (Clear canvas) and its `Clear` button appear only once a pass reads itself …" (`:584-586`) | no | exempt |
| "`Alt+R` creates and opens `scripts/script.py`; the complete script is pasted over the stub." (§ decision 6, `:329-330`) | yes (`create`, `open`) | **passes** — carries `Alt+R` |
| "a document can carry one Python script that drives its uniforms; `Alt+R` creates and opens it …" (`help_content.py` row, `:1038`) | yes (`create`, `open`) | **passes** — carries `Alt+R` |
| "press Ctrl+R and the script is created" (the spec's own falsifier) | yes (`press`, `create`) | **goes RED**, as claimed |

So the predicate accepts every mandated descriptive sentence, passes both real instructions, and
rejects the falsifier. That is the closure.

**Residual (new, narrow).** `\badd` matches `address`, `addresses`, `addressing`, `addressed`. The
spec's own D3 teaching point (`:556-558`) is "the script **addresses** a pass by NAME, so it drives
`paint` while the reader is looking at `composite`", and it says "The tutorial says so explicitly",
so a near-identical sentence lands in the body. It contains `script`, matches `\badd`, and carries no
chord, so it goes red on a sentence that is not an instruction. `pressure` (matching `press`) and
`openly` (matching `open`) are the same shape but do not appear in any mandated text. This is a
defect in the predicate the fold introduced, not a reopening of F8: the enumerated-verb design is
right and the fix is one line.

*Fix as a sentence the spec author can paste:* "The verb match is `\b<verb>\b` (both boundaries), not
`\b<verb>`, so `addresses` in the D3 teaching sentence and `pressure` are not read as `add` and
`press`; the inflections that ARE instructions (`creates`, `opens`, `makes`) are matched by adding
them to the set explicitly rather than by a prefix match."

### F9 — CLOSED

The two Step 5 rows are merged into one (`:997`). It names the callout in the singular ("the SINGLE
`CORRECTION` warn block"), carries the lede sentence, gives the replacement lede text I proposed, and
adds the diagnosis: "**The tutorial has exactly ONE `CORRECTION` callout, not two** — the word
appears twice, once in this warn block and once in the lede sentence claiming there are two of them,
so the lede is wrong in the current tutorial and this row is what fixes it." Re-verified against the
body: `grep -c CORRECTION` is 2, at `:80` (lede) and `:524` (the one block).

### F10 — CLOSED

Manual verification step 11 (`:825-830`) lists only `Ctrl+Shift+N`, `Alt+P`, `Alt+A`, `Alt+R`, adds
"The list is derived from the body at walk time, not typed here", and states why `F5` and `F6` are
absent, naming `TOGGLE_DOCUMENT_PLAY`'s real label ("Play/stop document script, not a document
transport") and pointing at F2 for `F6`.

`Alt+A` is no longer an orphan: the Step 2 prose row (`:954`) now reads "Add a pass called `seed`
(`Alt+A`)" with the rule stated — "**This is the one step that names the chord**; steps 3 to 6 read
'Add a pass called `x`, as before'". So the chord the manual step walks is quoted in the body, which
is what `test_the_tutorial_names_no_chord_the_command_table_does_not_have` needs to have something to
check. Verified against `commands.py:176`: `ADD_PASS` is `_chord(K.a, K.mod_alt)`.

## False trails this round (a preference is a false trail)

- The `test_the_jfa_run_count_covers_every_reachable_canvas` preset assertion being redundant while
  `MAX_CANVAS_PX` > `max(_SQUARE_PRESETS)`: true today, and the spec gives the reason it stays (the
  bounds move independently). Not a finding.
- Step 5's manual check now asking the maintainer to type 4096 into the canvas fields: slower to
  walk than the preset-only version, but it is the check the 12 ruling earns. Not a finding.
- Three surplus full-screen draws at 512 rather than two: the cost the ruling accepts, stated in two
  places. Not a finding.
- The card's `format`/`size` transposition being invisible to a reader who fills the gear by name
  rather than by position: possible, and irrelevant — the spec's stated claim is top-to-bottom
  reading, and the fold now makes that claim true.
- § decision 6 part 1 still saying the reader "adds three lines to `paint.frag.glsl`" while F7 edits
  that file's header: different halves of the file, no collision. The reader edits their own copy;
  F7 edits the shipped one.

## Coverage statement, round 2

Read in full: the folded `80_wave_h_tutorial.md` §§ Dependencies, Findings folded, Out of scope,
Design decisions 1-10, The generated card, The paint script, Files touched, Tests (all ten), Manual
verification (all twelve steps), Verified / corrected premises, The prose table (all thirteen
subsections), Open questions, Review history including `### The F6 ruling`.

Executed rather than reasoned: coverage of `ceil(log2(side)) <= 12` over every integer side from 16
to 4096; the offset table at 16, 17, 512, 1024, 2048, 4095, 4096 with the real/surplus split; the
draw count 1+1+12+1+6+1; the two test-bound falsifiers at 9, 11, 12 and at a raised clamp; the F8
verb predicate over the seven mandated sentences plus the falsifier; a `\b<verb>` collision scan over
the enumerated set. Re-read from code: `pass_graph.py:51-52` (`MIN_CANVAS_PX` 16, `MAX_CANVAS_PX`
4096), `popups/pass_settings.py:165,177` (format above size), `commands.py:176` (`ADD_PASS` = Alt+A),
and `grep -c CORRECTION` over `tutorial_body.html`.

Not re-checked this round because round 1 verified them and no fold touched them: the five gear
defaults, the six card values against `graph.json`, the ping-pong argument, the paint script's W-G
fidelity, the stub's import line, `cascade.frag.glsl`'s 109 lines, the ten prose-table old-text
quotations.
