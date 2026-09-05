# 079 — the final review

A convergence loop over everything the feature landed (`0e17e64..HEAD`, eleven commits, 63 files).
Each round's verdicts are written here so the loop survives a context reset.

## Reviewers

Four per round: three sonnet for the bulk enumeration, one opus where the judgement is the
deliverable.

| Agent | Model | Anchor |
|---|---|---|
| Decision coverage | sonnet | the spec's `## Locked decisions`, D1..D13, each demonstrated by execution |
| Deleted-symbol sweep | sonnet | the explicit list of symbols deleted or renamed in this feature |
| Doc-to-code fidelity | sonnet | `CLAUDE.md`, `conventions.md`, `dev_flow.md`, the `imgui-ui` skill |
| Behaviour verification | opus | the eleven commit messages' own claims, each probed |

The non-self-authored anchors: the vendored editor library (W-C's measurement is against its real
lexer output), and `make gates`' exit code.

## Round 1

### Deleted-symbol sweep (sonnet) — FAIL, one live break, fixed

`.claude/skills/shader-lab/render_document.py` imported and called `normalize_output` from the
deleted `shaderbox.scripting.outputs`. Verified independently: the import raises
`ModuleNotFoundError`, and running the script reached that line before any argument parsing.

The reason the gates missed it is the finding under the finding: pytest runs over `tests/`, which
imports the package but never the standalone helpers under `.claude/skills/`, `scripts/` and
`dogfood/`. Those are invoked by hand, so a symbol they depend on can be deleted with every gate
green.

Fixed at the root rather than the instance: the script now passes the value straight to
`coerce_uniform_value`, mirroring `behavior.py::coerce_one` after the wrapper types went, and
`tests/test_out_of_suite_scripts.py` imports every `shaderbox.*` module all 35 such scripts name.
Mutation-tested by reintroducing the bad import — the gate names the script and the module.

Everything else the sweep found was historical narration in `ai_docs/` (legitimate — a past event
does not drift) or an unrelated word. `_ENTRY_LABEL_W` is gone everywhere; no `ghost_button(` call
site or deleted-tier `button(` usage remains in code.

### Doc-to-code fidelity (sonnet) — CLAUDE.md PASS, imgui-ui PASS, conventions FAIL, dev_flow PARTIAL

Six stale claims, every one verified against the real file before fixing:

| Where | Was | Now |
|---|---|---|
| `conventions.md` script-value rule | "a bare number for a scalar; `Vec2/3/4`/`Array`/`Text` for the shaped kinds" | plain Python, shaped against the live uniform, no wrapper types |
| `conventions.md` import claim | "a real `from shaderbox.scripting import Vec2, Vec3, …` all work" | the narrowed `__import__` and the three names it admits |
| `conventions.md` exec-seam bullet | "the exec globals are the REAL builtins" + the old injected list | the real builtins with ONE substitution, `__import__` |
| `dev_flow.md` scripting map | `outputs.py` (deleted) | `api_doc.py` (real, and the docstring gate's subject) |
| `dev_flow.md` copilot map | no `tools/passes.py` | its three lazy tools listed |
| `behavior.py::ScriptBehavior` docstring | the wrapper types — and this is what `K` SHOWS the user | the plain-Python contract |

The last one matters most: it is not a comment but a live user-facing string, and it contradicted
its own module docstring 38 lines above.

What held: every `CLAUDE.md` claim (no `@staticmethod`, no `TYPE_CHECKING`, no `__future__`, the
skills list), the imgui-ui skill's tier table and its pinned imgui-bundle version, the button-tier
convention, the docstring gate, `StoppedKey`'s home, and the copilot eager/lazy map apart from the
one omission.

### A stray probe left in `tests/`

One reviewer wrote `tests/_zz_probe_d9_test.py` despite the prompt saying not to, and it failed the
gate. Removed. Its subject was D9 (the error strip's placement), the one decision with no gate
because only one App per process may drive `update_and_draw` — so its claim is worth reading in the
behaviour agent's report rather than dismissing with the file.

### Decision coverage (sonnet) — PASS, all 13 covered

Every locked decision demonstrated by execution rather than by reading. Two worth naming:

- **D9** (the error strip below the image) is the one decision with no permanent gate, because only
  one App per process may drive `update_and_draw`. The agent probed it in a real frame anyway and
  measured the strip landing at the image's true bottom, one `cell_h` below where the interaction
  button ends. That is the geometry the fix intended.
- **D8** (the viewer at full alpha) has no direct render test either; the mechanism is a
  `push_style_var` wrapping the viewer inside the outer disabled scope, unambiguous in source.

Its one FAIL — `test_editor_ffi.py` — was the in-flight re-vendor, not a 079 regression. It read
the concurrent working-tree churn as "two other Claude sessions"; it was this session, mid-vendor.

## Editor re-vendor: `6d526c6` -> `38cadbc`

W-C's remaining half is closed. The editor session reframed the ask: `gl_FragColor`, `texture2D`
and `textureCube` are not builtins in the wrong color, they are names 460 core REMOVED, so
dropping them from the lexer's builtin list is a correctness fix rather than a new precedence
rule. Measured after vendoring: 30 orange glyphs for `gl_FragColor` plus a user's `fragColor` twice,
`gl_FragCoord` still a builtin.

Two corrections to this session's own earlier work:
- **`gl_FragData` was never a builtin.** Re-measured on the OLD library, it recolored fine. The
  claim that both names were affected came from reading `_BUILTIN_OUTPUTS` rather than measuring
  each name.
- **`abi_probe.py` is a sixth vendored artifact** the handover list omitted; the signature-table
  gate fails without it. Reported back.

### Behaviour verification (opus) — PARTIAL: three real, one false positive

Probed every behavioural claim the eleven commit messages make. Three held up under my own
re-verification and are fixed; one was a timing artifact.

**REAL — the import hint advised a deleted type.** `_import_hint` fell back to `'Vec3'` when the
exception carried no `name`, and it appended itself to the import GATE's own message — so a wrong
import produced advice that recreates the error: "not Vec3 -- import it: `from shaderbox.scripting
import Vec3`". The fallback outlived the type it named. The hint now fires on a `NameError` alone,
for the name the error actually carries.

**REAL — a bare sampler key went silent.** D5 keeps sampler/block keys errors, and W-E's `_binds`
helper collapsed "not declared" and "declared but a sampler" into one falsy answer on the broadcast
path. `{'u_tex': 1.0}` was silent while `{'main': {'u_tex': 1.0}}` errored. The broadcast branch now
separates the two: nowhere declared is silent (D5's first half), declared-and-refused is an error on
that pass (D5's second half). Gated.

**REAL — `dev_flow.md`'s step 7 survived.** The commit claimed to remove the manual-check apparatus
and removed four incidental mentions while leaving the actual step standing. Step 7 is now the
verification-design guidance it mostly already was, with the go-and-look framing gone.

**FALSE POSITIVE — `gl_FragColor`.** The agent measured 12 orange glyphs and called the commits and
the banner wrong. It was reading a tree where the new `.so` had been copied in mid-run but the docs
had not caught up. Settled from the primary source rather than either measurement: `gl_FragColor`
IS in `src/lex_glsl.odin`'s builtin list at `6d526c6` and is NOT at `38cadbc`. Both measurements
are correct for their own binary. The stale roadmap banner it noticed was real and is rewritten.

**Also noted, not a defect:** the W-L geometry figure (`open` at x=50) is font-metric dependent —
56 on the agent's read, decomposing exactly as the label's width plus one `SPACE.MD`. The spec now
states the decomposition rather than the machine-specific number.

## Round 2

### Fix verification (sonnet) — PASS, all four hold

The bare sampler key errors on both paths and an undeclared key stays silent; the import hint
returns a string only for a NameError on one of the three importable names, and "" for every
gate message and every deleted name; the editor's 100 exports match the binding and the vendored
signature table in both directions, with zero argtype mismatches.

Two residual observations, both checked:

- **`abi_probe.py` cannot run from its vendored location** — its `ROOT` is the upstream repo's
  layout, one directory higher than ours. Not a defect: it is vendored as the reference table the
  signature gate PARSES, and that gate already says so in as many words, stopping its exec above
  the `CDLL` line for exactly this reason. Left alone.
- **`dev_flow.md`'s Maintainer habits still said to "confirm the behavior in the actual app (run
  it)"** — an instruction to the agent to go and look, which is both the gating that went and
  something this box cannot do. Rewritten to demand a probe that would fail if the claim were
  false.

### Script-engine contract audit (opus) — PARTIAL: four real findings

The full bare-vs-block matrix, every cell probed. The contract itself is consistent — the one
asymmetry that survives is deliberate and statable: a bare key that finds a home on one pass while
another declares the name as a sampler is silent, because a bare key is a claim about the DOCUMENT
and it found its home; naming that pass explicitly is a claim about the pass and errors. Four
things were wrong, all fixed:

- **A stale error lived forever across a bare/block rewrite.** The pass-free slot
  `(document_id, "", key)` holds two namespaces — a bare uniform name no pass declares, and a
  block naming a pass that does not exist. Rewriting `{"blur": {...}}` as `{"blur": 1.0}` touched
  the pair from the bare path, which suppressed the stale-clear, while never writing the slot to
  overwrite it. The strip and the copilot's probe both kept reporting "no pass named 'blur'" for a
  script that no longer names a pass. The clear now asks whether this tick RE-RECORDED the error,
  not whether the key was touched. Gated, mutation-tested.
- **`None` did not mean "stays manual".** The stub, the copilot's API block and 059's spec all say
  it does; the value reached coercion, failed as "not a number" and counted as driven — a red row
  for the one gesture meaning "leave this one to me". Dropped beside the engine-owned drop, so
  both paths get it. One test had pinned the old behavior with a comment calling it deliberate;
  three user-facing surfaces outweigh a test comment, and the value is unchanged either way.
- **`scripts/dogfood/harness.py` crashed on any orphan key** — it unpacked `orphan_keys` as
  3-tuples after D5 reshaped them to pairs. Two more errors in the same file predate this feature:
  `render_video` called with a width and height after the API took a `RenderShape`, and a samples
  annotation that never followed 069's pass-qualified keys.
- **`smoke.py`'s feedback canary armed nothing.** It set `PassEntry(inputs=…)`, a field 072
  removed — and pydantic drops an unknown field silently. The canary passed anyway because the
  shader declares `u_prev`, which self-wires under the default `AutoSource`; mutation-testing
  `begin_frame` confirms the canary is genuinely live. The inert line is gone.

**The root cause under three of those: pyright did not check `scripts/` or `.claude/skills/`.**
`include = ["shaderbox"]`, and those helpers are invoked by hand rather than imported by `tests/`,
so breakage accumulated invisibly. The checked set now covers both. Excluded with reasons: the
dogfood run artifacts (recorded model output, evidence rather than code) and three scripts whose
only findings are upstream-stub imprecision on working code.

Also amended: D5's "a pass that does not compile stays an error" was the stale half of a
disagreement with the code, which is silent there deliberately — the pass's own compile error is
the thing to read.

## Round 3

### Fix verification (sonnet) — PARTIAL: my round-2 fix had a regression

Claims 2, 3 and 4 held. Claim 1 half-held, and the half that failed was one I had introduced.

The round-2 stale-clear asked whether a key was REWRITTEN this tick, computed as a set difference
against the rows present before the tick. A re-recorded error is not "rewritten" by that
definition — it was there before and it is there after — so a row the script keeps earning was
popped right after being written. Probed directly, it did not merely vanish: it OSCILLATED, on
for tick 1, off for tick 2, on for tick 3.

The agent's coverage probe is worth recording: it verified pyright's widened set by injecting an
undefined name into all nine non-excluded scripts inside a disposable `git worktree` and watching
the error count go 0 -> 9, one per file, then confirmed the three excluded files report
`filesAnalyzed: 0` when passed explicitly. That is the difference between "the config says so" and
"the checker does so".

**The fix, third attempt and the right shape:** the document's per-key rows are DELETED at the top
of the tick and rebuilt by the phases below. What a tick writes is what stands; a key the script
fixed is simply not written again. No question about touched-ness or rewritten-ness arises, which
is what made the first two attempts subtly wrong. The behavior-level sentinel is not a per-key row
and is left alone. Both halves are now gated — a row that persists across four ticks of an
unchanged script, and a row that clears across a bare/block rewrite — and the mutation fails both.

### Unreached-code hunt (opus) — PARTIAL, and it found the worst one

**The round-2 pyright widening never took effect.** `.pre-commit-config.yaml`'s hook read
`uv run pyright shaderbox` — a path argument, which OVERRIDES `[tool.pyright] include`. So the
config named `scripts/` and `.claude/skills/`, the gate stayed green, and nothing checked them.
Proven both ways: an injected type error in `scripts/token_probe.py` is invisible to
`pyright shaderbox` and caught by `pyright` with no argument. The hook now passes no path, and
`make check` catches the injected error.

That is the worst failure shape in this whole review — a rule that reads as enforced, passes its
gate, and enforces nothing. It is also the second time this feature shipped an inert mechanism
(the smoke canary's `PassEntry(inputs=…)` was the first), which is why both now have a gate that
was mutation-tested rather than a comment claiming they work.

**A generated artifact had drifted from its generator.** `shader_lib/text/glyphs.glsl` still said
`Node.compile()`, the name 065 renamed — in shipped library text the copilot reads. Nobody re-ran
`scripts/gen_glyphs.py` after the rename and nothing compared them. Regenerated, and
`tests/test_generated_artifacts.py` now asserts both of that generator's outputs match what it
renders, naming the fix command on failure. Mutation-tested.

**The shader-lab skill stated a confident falsehood.** It said `SB_fbm` "was REMOVED" and told the
reader to hand-write one; `SB_fbm` ships in `noise/fbm.glsl`, and the skill's inventory also
omitted `SB_domain_warp`, `SB_hash22`, `SB_hash31`, `SB_tri_wave`. A recited inventory goes stale
in silence, so the skill now names the grep that is always right instead of a list.

**`063/rc_proof.py` does not import** — it reads `shaderbox.core.Node`. Left that way on purpose:
063's own `17_direction.md` calls it "evidence, not a foundation", and rewriting its imports to
today's names would make it a claim about an engine it never measured. It now says so in its
docstring, so a reader does not mistake a frozen record for a runnable script.

Its false-trail section is the most useful part of the report: `render_document.py` runs in both
modes now, `068/oracle.py` still passes at 3.65% relMAE, `build_tutorial.py` reproduces the
committed HTML byte for byte, every `scripts/**` module resolves every attribute it accesses,
and the three shell scripts are `bash -n` clean with every staged path present.

## Convergence

Three rounds, each finding real defects, and two of them finding defects in the previous round's
fix. That is the loop working as intended rather than a sign of thrash: the stale-clear took three
attempts because the first two asked the wrong question, and the pyright widening took two because
the first was inert.

## Round 4

### Fix verification (sonnet) — PASS

All four round-3 fixes bite. Each broken in a disposable worktree, each gate goes red naming the
offending file: a type error under `scripts/` and one under `.claude/skills/` both fail
`make check`; hand-editing either generated artifact fails its parametrized case with the fix
command; a bogus import fails exactly one of the standalone-script cases; and a soft error
persists across all ten consecutive ticks, clearing the tick after the fix, with two bad keys
resolving independently.

Its own false trail is the useful part: its first tick probe looked like an engine defect until it
found the probe had skipped `eng.reload()` — `tick()` does not recompile a changed file. It
caught its own bug instead of filing it.

### Inert-mechanism hunt (opus) — PARTIAL, and it caught the gate itself

**`make gates` reported GREEN for every auto-fixable ruff violation.** The target retried `check`
on ANY first-run failure, reasoning that hooks rewrite files — but `ruff --fix` and `ruff-format`
ARE the file-rewriting hooks, so an auto-fixable violation was silently repaired and run 2 passed.
Reproduced: appending an unformatted function to `shaderbox/paths.py` leaves `make gates` at exit
0, GREEN, with the file rewritten in the tree.

This is why an unformatted file shipped earlier today, and why the commit that fixed that instance
said "`make check` does catch it" — true of `make check`, false of `make gates`, which is the gate
the rules mandate. The instance was closed and the mechanism was not.

The retry now fires only when the tree is UNCHANGED — pyright's env bootstrap, the case that
earned it — and a rewrite fails the gate telling the reader to review and stage. Three cases
verified: a clean tree stays green, an auto-fixable violation fails naming the rewrite, a
non-auto-fixable error still fails. Getting the detection right took three attempts (porcelain
alone, then a diff stat that happened to be identical, then hashing the diff content), which is
the same lesson as the stale-clear: compare what actually changed, not a summary of it.

**A second `PassEntry(inputs=…)`.** `TargetConfig(persist=True)` in `test_pass_verbs.py` names a
field 7188717 removed; pydantic dropped it. `extra="forbid"` is not available on these twelve
models — they are the persisted shapes, and forbidding extras would make an older `document.json`
fail to load, which the persistence posture rules out — so `tests/test_model_kwargs.py` guards the
CALL SITES instead, walking every `Model(field=…)` in `shaderbox/`, `tests/`, `scripts/` and
`dogfood/`. Mutation-tested: it names file, line, model and keyword.

Its false trail is worth keeping: seven guard tests mutated at their source all went red, every
dataclass raises on an unknown kwarg (only pydantic is silent), no model uses an alias, 91
settings fields all trace to a consumer, and all 16 env vars are accounted for.

## Convergence

Four rounds. Each found real defects, and three found defects in the previous round's fix — the
stale-clear took three attempts, the pyright widening two, the gates retry three. The pattern
worth keeping is not "review until clean" but "a fix is a claim, and a claim needs the same
demonstration as the bug it closes."
