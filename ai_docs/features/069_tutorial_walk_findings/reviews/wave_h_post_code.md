# W-H post-implementation review — code correctness

Commit under review: `fdc7841` ("069 W-H: generate the tutorial from the example").

**Verdict**

| Area | Verdict |
|---|---|
| Generator | PARTIAL |
| JFA shader | PASS |
| Example | PASS |
| Tests as checkers | PARTIAL |
| Help | PASS |
| Conventions | PASS |

`make gates` is GREEN at `fdc7841` in an isolated worktree (exit 0, captured unpiped; check
passed, test passed 1629/4 skipped, smoke **passed** — not skipped). The committed
`tutorial.html` regenerates **byte-identical** (md5 `caf13f53a6944ddf1a395826c796b305` both
sides). Every load-bearing number in the commit message reproduced exactly.

Nothing here blocks the wave. Three findings are open gates on generated-vs-source drift — the
same defect class W-H exists to close, left open on three surfaces the wave did not reach.

---

## Findings

### 1. The committed `tutorial.html` — the file a reader opens — has no freshness gate

**Claim.** `test_no_marker_survives_the_build` builds into `tmp_path` and asserts on that. No
test, hook or make target compares the build's output to the committed `tutorial.html`, so the
generated artifact can drift from its own generator and the suite stays green. This is the
generated-drifts-from-source class the wave was built to close, still open on the output side.

**Evidence.** In an isolated worktree at `fdc7841`, editing the committed output alone:

```
$ sed -i 's|<td>runs</td><td>12</td>|<td>runs</td><td>9</td>|' .../tutorial.html
$ pytest tests/test_tutorial_build.py -q
11 passed in 0.68s
```

And the mirror case — a body edit that is never rebuilt ships a tutorial missing it:

```
$ printf '<p>An entirely new paragraph that never reached tutorial.html.</p>\n' >> tutorial_body.html
$ pytest tests/test_tutorial_build.py -q
11 passed in 0.52s
$ grep -c "entirely new paragraph" tutorial.html
0
```

`grep -rn "tutorial.html" tests/ Makefile .pre-commit-config.yaml` returns exactly one hit,
`tests/test_tutorial_build.py:76`, which is the `tmp_path` name. The file is in sync at this
commit (verified byte-identical above); nothing keeps it so.

**Fix.** Add a test that builds to `tmp_path` and asserts the result equals the committed
`ai_docs/features/068_radiance_cascades/tutorial.html` byte for byte, so a body or example edit
without a rebuild turns the suite red.

---

### 2. The card's `format` row can print a label the gear does not show

**Claim.** `test_the_generator_defaults_match_the_engine` gates `_DTYPE_LABELS` with
`set(_BUILD._DTYPE_LABELS) == set(DTYPES)` — the KEY set only. The label TEXT is a hand-copied
duplicate of `pass_settings.py::_FORMATS` and nothing compares it, so the one card row whose
value is a copied string is the one row that can drift. That is precisely finding #27/#31's
defect surviving in the fix for it.

**Evidence.** In the isolated worktree, drifting a label while leaving keys untouched:

```
$ sed -i 's|"f2": "16-bit float"|"f2": "half float"|' build_tutorial.py
$ pytest tests/test_tutorial_build.py -q
11 passed in 0.66s
```

The tutorial would then tell a reader to pick "half float" from a combo whose entries are
`8-bit` / `16-bit float` / `32-bit float`. (Today they do match — I compared
`_DTYPE_LABELS` against `{c: lbl for c, lbl, _ in _FORMATS}` and got `EQUAL: True`.)

**Fix.** Strengthen that assertion to compare the mapping, not the key set:
`assert _BUILD._DTYPE_LABELS == {code: label for code, label, _ in _FORMATS}`, importing
`_FORMATS` from `shaderbox.popups.pass_settings`.

---

### 3. The card's `reads` row reads the raw graph, so a name-resolved edge renders as "nothing"

**Claim.** `_reads_html` iterates `entry.get("inputs", {})` straight out of `graph.json`, but
069 D9 makes an ABSENT key the preferred on-disk state ("a resolved edge never reaches disk,
because it would then be indistinguishable from a chosen one"). A pass wired by the name rule
alone therefore prints `reads: nothing` while the engine binds the texture. The engine's own
answer to this question is one imported pure function, `pass_graph.effective_inputs`, which
`Document.effective_graph` already makes the single source all six consumers read.

**Evidence.** Deleting the `u_jfa` key from `df` — which D9's name rule resolves anyway:

```
generator card    : <tr><td>reads</td><td>nothing</td></tr>
engine resolves to: {'u_jfa': 'jfa'}
$ pytest tests/test_tutorial_build.py -q      # after the same deletion in graph.json
11 passed in 0.68s
```

Every sampler in today's example carries an explicit key (verified: the six `.frag.glsl` files
declare `u_paint`, `u_seed`/`u_prev`, `u_jfa`, `u_df`/`u_prev`, `u_cascade`, all present in
`graph.json`), so the rendered tutorial is correct at this commit. The gap is structural and
silent.

**Fix.** Have the generator resolve the reads row through the engine, either by importing
`pass_graph.effective_inputs` or by asserting in the test that every declared sampler of every
example pass has an explicit `graph.json` key, so a future D9-shaped save turns the suite red
instead of silently blanking a card row.

---

### 4. The chord test cannot see a punctuation-key chord, though its W-E twin can

**Claim.** `test_the_tutorial_names_no_chord_the_command_table_does_not_have` matches
`(?:Ctrl|Alt|Shift)(?:\+(?:Ctrl|Alt|Shift))*\+\w+`. `\w` excludes punctuation, so `Alt+/` — a
chord `COMMAND_SPECS` actually binds — is invisible to the checker, as is a stale `Ctrl+;`. The
already-solved twin shipped by W-E, `test_help_content.py::_PROSE_CHORD`, uses
`` `((?:Ctrl|Alt|Shift)\+[^`]+|F[0-9]{1,2})` `` and does catch it; its own comment names `Alt+/`
as a case it handles. The new checker narrowed the domain relative to the sibling it mirrors.

**Evidence.**

```
Alt+/           caught=False      <- bound by a live CommandSpec
Ctrl+;          caught=False
F13             caught=False
Ctrl+Tab        caught=True
Ctrl+Shift+Z    caught=True
```

The five chord forms the body uses today (`Alt+A`, `Alt+P`, `Alt+R`, `Ctrl+Shift+N`, `F6`) are
all caught and all resolve against `COMMAND_SPECS`, so the test is correct on current content.

**Fix.** Widen the modifier pattern to the sibling's shape — accept a non-space run after the
final `+` rather than `\w+` — so a punctuation-key chord is checked rather than skipped.

---

### 5. Spec D2 and two source comments still state the retired offset formula

**Claim.** The commit replaced the run-count-derived offset with a canvas-derived one, and added
a W-H note at `01_spec.md` line 76 saying so. Three earlier sites still present the old formula
as the current one, and a reader hitting D2 first gets the retired answer.

**Evidence.**

- `ai_docs/features/068_radiance_cascades/01_spec.md:55` — "JFA's offset is
  `pow(2.0, u_pass_iterations - u_pass_iteration - 1.0)`, one line", stated in the present tense
  21 lines above the note that corrects it.
- `shaderbox/core.py:367` — "the shader's own `pow(2.0, iterations - iteration - 1.0)` is one
  line", in `Pass.render`'s docstring.
- `shaderbox/pass_graph.py:104` — "`u_pass_iterations` so one shader can be the whole chain --
  JFA's halving offset".

The shipped shader computes
`exp2(ceil(log2(max(u_resolution.x, u_resolution.y))) - 1.0 - u_pass_iteration)`. The cascade
pass does still use the `u_pass_iterations - 1.0 - u_pass_iteration` shape
(`cascade.frag.glsl:58`), so the general point survives; only the attribution to JFA is stale.

**Fix.** In `01_spec.md` D2, attribute the `pow`/`exp2(iterations - iteration - 1)` example to
the cascade stack rather than to JFA, and do the same in `core.py`'s docstring and
`pass_graph.py`'s comment.

---

### 6. `_hand_written_code_blocks`'s splice filter is unreachable

**Claim.** The filter `[b for b in blocks if "{{CODE:" not in b]` never excludes anything: the
six `{{CODE:x}}` markers each sit alone on their own line with no wrapping `<pre><code>` (the
generator supplies the wrapper), so no marker is ever inside a matched block.

**Evidence.** `total <pre><code> blocks in body: 6` / `classified hand-written: 6`; the marker
lines are bare (`grep -B2 -A2 "{{CODE:jfa}}"` shows blank lines on both sides). The direction is
safe — over-inclusion, and the six blocks it does scan are exactly the hand-written ones — so
this is a cosmetic dead branch, not a hole.

**Fix.** Leave it; it costs nothing and guards the shape a future body edit might introduce.

---

## False trails

- **Unannotated locals in `build_tutorial.py`** (`raw`, `line`, `when`, `body`, `source`,
  `target`) against CLAUDE.md's "full type annotations on all variables" — measured, the package
  itself leaves 65% of local assignments bare (2644 of 4076), so the generator matches actual
  repo practice. Not a finding.
- **`ai_docs/` escapes `make check`** — confirmed (`extend-exclude = ["ai_docs/"]`, pyright
  `include = ["shaderbox"]`), but running both tools on the generator by hand gives ruff "All
  checks passed", `ruff format --check` "1 file already formatted", pyright "0 errors". The
  duplicated-defaults concern the implementer flagged is genuinely gated by
  `test_the_generator_defaults_match_the_engine` (falsified: drifting `_DEFAULT_DTYPE` to `"f1"`
  turns it red). Restating them is acceptable — the module documents why it will not import
  `shaderbox`, and the test closes the loop for the values. Only the label TEXT is ungated
  (finding 2).
- **`test_default_wiring.py::test_an_unresolved_sampler_renders_black` fails in the live tree** —
  not this commit. It passes at `3d635ef` and at `fdc7841` in a clean worktree; the live tree
  carries another reviewer's uncommitted `M tests/test_default_wiring.py` plus a stray
  `tests/test_zz_probe2.py`.
- **`oracle.py` and the 3.65% relMAE** — the oracle does not read the example
  (`grep "document_examples\|graph.json\|jfa" oracle.py` returns nothing), so the run-count and
  offset change cannot touch it. No re-run needed.
- **`jfa.png` staleness** — the commit touches no file under `img/`, and the surplus runs are
  provably a no-op (below), so the standing image is correct.
- **`document.json`'s change** — one line, the `ui_state.description` string, `9 runs` to
  `12 runs`. It is a field `UIDocument.save` writes and nothing else in the file moved.
- **`_is_script_instruction` over-matching** — probed nine sentences; it rejects "a script
  addresses a pass by NAME" and "the script readdresses the pass", accepts "press Ctrl+R and the
  script is created" and "now add a script to the document". Behaves exactly as the commit
  message claims.

---

## Coverage

**Generator.** Regenerated and diffed against the committed file (byte-identical). Rendered all
six cards and compared every row against `graph.json` by hand: format, size, smooth, repeat and
runs match every pass, and the `default` marks match `pass_graph.py`'s `DEFAULT_*` and
`PassEntry().iterations`. Confirmed `_DTYPE_LABELS` equals `_FORMATS`'s mapping today. Verified
each `{{CODE:x}}` block is the whole file (`html.unescape(inner) == source.rstrip()` for all six)
with correct escaping and no raw `<` (jfa has 7 `<`, all escaped). Confirmed no marker text
exists inside the example, so a splice cannot inject one. Ran ruff, `ruff format --check` and
pyright on the generator manually. Confirmed the seven card rows follow the gear's draw order in
`_draw_target` / `_draw_repeat`.

**JFA.** Drove the shipped `jfa.frag.glsl` headless over a corner seed with the same ping-pong
`document.py`'s loop performs, at 512 / 1024 / 2048 / 4096 and 11 / 12 runs. Reproduced the
commit's exact figure: at 11 runs and 4096, `12582912 / 16777216` texels unreached; at 12 runs
every size completes with 0 unreached. Confirmed surplus runs are a no-op — 9 vs 12 at 512 is
`np.array_equal` True, max abs diff `0.0`; 11 vs 12 at 2048 likewise. Mutation-tested the
copy-forward branch: replacing it with `discard` leaves all 262144 texels unreached at 512x12,
so the branch is load-bearing exactly as 068 D5's per-iteration swap requires. Read the
iteration loop and `_swap_feedback` to confirm the swap target the copy feeds.

**Example.** `graph.json` `iterations: 12` confirmed; `document.json`'s single-line change
reviewed; the paint header rewrite reviewed against 068 D7's lifted retraction. Ran
`tests/test_default_wiring.py` (incl. `test_every_multi_pass_example_compiles_every_pass`) and
`tests/test_examples_resolve.py` — green in the clean worktree. Verified the tutorial's taught
script compiles and every `ctx.mouse` field it reads (`x`, `y`, `prev_x`, `prev_y`, `down`)
exists on the live `MouseState`.

**Tests.** Read all eleven end to end. Ran **eight** falsifiers, each restored and verified
restored (`git checkout` + `git diff --quiet`) in an isolated worktree, never the live tree:
jfa iterations 12→11 (red), an unbound chord `Ctrl+E` (red), a chordless script instruction
(red), a fragment naming an absent uniform (red), a dropped `{{CARD:df}}` (red), a drifted
`_DEFAULT_DTYPE` (red), `MAX_CANVAS_PX` raised to 8192 (red), a mistyped `{{CARD:nosuchpass}}`
(red). Also relaxed the trailing `\b` in the verb pattern and confirmed the suite goes red on
the mandated "addresses a pass by NAME" prose, reproducing the commit message's claim. Probed
each checker's domain: the chord regex against nine chord shapes (hole found, finding 4), the
script predicate against nine sentences (clean), the hand-written-block scan against all six
`<pre><code>` in the body (all seen), and confirmed `MAX_CANVAS_PX` / `_SQUARE_PRESETS` are
imported rather than restated. Ran three green-stays-green probes for gaps: a stale
`tutorial.html`, an unrebuilt body, a drifted label, a dropped resolvable edge (findings 1, 2,
3).

**Help.** Reviewed the `your_uniforms` chord change; `tests/test_help_content.py` green (9
passed), including W-E's `test_no_help_prose_quotes_a_chord_the_table_does_not_bind`.

**Not covered.** The rendered tutorial's visual layout in a browser, and the maintainer's own
walk (`80_wave_h_tutorial.md § Manual verification`), which the roadmap already names as the
remaining verification.

---

# Round 2 (closure)

Commit under review: `7440c1d` ("069 W-H fixes: gate the generated file too"), read via
`git show 7440c1d:<path>` and exercised in an isolated worktree at that commit. The live tree
was never touched.

**Overall: PASS.**

| Finding | Verdict | The line |
|---|---|---|
| 1. `tutorial.html` had no freshness gate | **CLOSED** | `tests/test_tutorial_build.py::test_the_committed_tutorial_is_a_fresh_build` |
| 2. `format` row could print a label the gear does not show | **CLOSED** | `assert {code: label for code, label, _ in _FORMATS} == _BUILD._DTYPE_LABELS` |
| 3. `reads` row read the raw graph | **CLOSED** | `build_tutorial.py::_resolved_inputs` + `test_a_card_resolves_the_same_reads_the_engine_does` |
| 4. Chord test blind to a punctuation-key chord | **CLOSED** | `_BODY_CHORD = re.compile(r"<code>((?:Ctrl\|Alt\|Shift)\+[^<]+\|F[0-9]{1,2})</code>")` |
| 5. Three sites stated the retired offset formula | **CLOSED** | `01_spec.md` D2, `core.py::Pass.render` docstring, `pass_graph.py::PassEntry.iterations` comment |
| 6. Unreachable splice filter | **NOT CLOSED, as advised** | left in place per Round 1's recommendation |

`make gates` is GREEN at `7440c1d` (exit 0, captured unpiped; check passed, test passed, smoke
**passed**). `tests/test_tutorial_build.py` is 11 tests to 13. The committed `tutorial.html`
regenerates **byte-identical** (md5 `c5e087f0c31c979f482d30ee4a6b68ff` both sides).

## The three green-stays-green probes, re-run

Each was `11 passed` at `fdc7841`. Each now goes red, restored and verified restored
(`git checkout` + `git diff --quiet`) after every mutation.

**Finding 1, both directions.** Editing the committed output alone
(`<td>runs</td><td>12</td>` to `9`): `FAILED test_the_committed_tutorial_is_a_fresh_build`,
`1 failed, 12 passed`. Appending a paragraph to the body without rebuilding: same test red,
`1 failed, 12 passed`. Both halves of the class are now gated.

**Finding 2.** Drifting `"f2": "16-bit float"` to `"half float"` with the key set untouched:
`FAILED test_the_generator_defaults_match_the_engine` (and the freshness test, correctly, since
the output no longer matches). `2 failed, 11 passed`.

**Finding 3** needed a different probe than Round 1's, because the Round 1 probe measured the
defect and the defect is gone. Dropping `u_jfa` from `df`'s stored inputs now renders
`<code>u_jfa</code> from <b>jfa</b>` instead of `nothing`, and the suite is correctly green —
that is the fix working, not a miss. So I falsified the generator's new rule instead, twice:

- reverting `_resolved_inputs` to stored-keys-only (`for uniform in []`), which is exactly the
  original defect: `FAILED test_a_card_resolves_the_same_reads_the_engine_does`;
- breaking the self-read branch (`_FEEDBACK_UNIFORM = "u_never_matches"`): same test red.

The parity test's non-vacuousness is measurable: it drives **21 cases across the six passes, 15
of them with at least one key dropped** (paint 1, seed 2, jfa 4, df 2, cascade 8, composite 4).
The commit's account of its first version being vacuous is consistent with what I measured — with
every key present both rules agree trivially, and the `for uniform in []` mutation above would
have passed such a test.

## Finding 4, falsified end to end

The regex now catches all eight shapes I probed at Round 1, including the three that were
invisible (`Alt+/`, `Ctrl+;`, `F13`). End to end in the body:

- `<code>Ctrl+;</code>` (unbound, punctuation — the exact shape that was invisible at `fdc7841`):
  `FAILED test_the_tutorial_names_no_chord_the_command_table_does_not_have`;
- `<code>Alt+/</code>` (bound): the chord test **passes**, so the widening did not turn a real
  chord into a false positive;
- `<code>Ctrl+E</code>` (unbound, plain): still red, so the original coverage did not regress.

(In all three the unrelated `test_every_script_instruction_carries_the_chord` also fires, because
the substitution removes `Alt+R` from a script sentence. Expected, and identical across the
three, so it does not confound the comparison.)

## Finding 5, verified as true rather than merely changed

`grep -rn "pow(2.0"` over `01_spec.md`, `core.py` and `pass_graph.py` now returns nothing. The
replacements are factually correct against the shipped shaders: `cascade.frag.glsl:59` is
`float level = u_pass_iterations - 1.0 - u_pass_iteration;`, which is the shape all three sites
now attribute to the cascade stack, and `jfa.frag.glsl:27` is the canvas-derived
`exp2(ceil(log2(max(...))) - 1.0 - u_pass_iteration)`. D2 also keeps a pointer to the W-H note
rather than deleting the history.

## Two fixes beyond my findings, checked

Both come from the sibling review and both are real, so I verified rather than took them on
report.

- `cascade.frag.glsl`'s header said `Set "Runs per frame" to 6`. The gear draws
  `separator_text("Runs")` and `label_row(..., "runs", ...)`, so the label was stale, and
  `{{CODE:cascade}}` splices that header verbatim into step 5 — a shipped shader's own prose
  riding into the tutorial. `grep -rn "Runs per frame"` over `shaderbox/` and the feature
  directory now returns nothing. `document.json` carried the same label and is fixed with it.
- "Then open the **Passes** strip" is gone. `tabs/document.py:306` calls `pass_list.draw(...)`
  with no guard, so the strip is always drawn and there was nothing to open.

Both example-directory edits are comment and description prose only, no GLSL logic, and
`git diff fdc7841 7440c1d -- .../passes/jfa.frag.glsl` is empty. I re-ran the JFA probe anyway
and got the Round 1 figures unchanged: 11 runs at 4096 leaves `12582912 / 16777216` unreached,
12 runs completes at 4096 and at 512.

## Coverage and non-findings

Read the full diff of `build_tutorial.py`, `tests/test_tutorial_build.py`, `01_spec.md`,
`core.py`, `pass_graph.py`, `tutorial_body.html` and both example files. Ran the tutorial suite
(13 passed), eight mutations across the five findings, the JFA spot-check, and `make gates`.

Per the late-round rule I am not raising preferences. Two things I looked at and am explicitly
not filing: `_resolved_inputs` restates D9's name rule rather than importing it, which is forced
by the module's stated decision not to import `shaderbox` and is closed by the parity test that
drives the engine's own `effective_inputs` (both falsifiers bite); and `_SAMPLER_RE` parses
declarations by regex rather than from a compiled program, which the docstring justifies and
which the same parity test covers, since a missed sampler would drop a row the engine resolves.
