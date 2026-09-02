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
