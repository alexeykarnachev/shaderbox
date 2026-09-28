# 101 — Instancing research record

RESEARCH RECORD, not a plan. The source of evidence for features 102-105. One session's solo reading plus four adversarial agents,
every claim checked against the code and the measurements re-run. Nothing here is
designed and nothing is scheduled.

**This file holds four features' worth of material, and the split is DONE** -- 102, 103,
104 and 105 cite it for evidence (`## The split`). It was written whole because the
dependency order only settled at the end of the research: the on-ramp this feature was
originally filed for turned out to sit ON TOP of an engine contract that is not yet
correct, so documenting it first would have documented a broken foundation.

**This file is now a RECORD, not a plan.** Nothing here is scheduled; the four specs carry
the work.

Provenance, since it decides how much to trust each line: claims marked MEASURED were
produced by running code, and the agent reports re-ran every one of them. Claims marked
READ cite a file and line. Anything else is reasoning and is labelled as such -- the
session that wrote this shipped one confident assertion (additive blending) that had not
been tested, so the two registers are kept apart on purpose.

## Maintainer decisions (SETTLED — carry these as premises, never as options)

- **D-A. The copilot gets FULL integration.** It must understand the whole mechanism,
  write instanced passes, and its tools must carry populations. Explicitly NOT the
  smaller reading of "merely stop recommending the old technique".
- **D-B. Highlighting must be a GENERAL, REUSABLE mechanism.** The editor library gets
  reused in other projects, so the answer is a host pushing position-based spans it
  computed itself -- not a shaderbox-specific workaround, not word-table-only, and not
  teaching the Odin lexer Python by name.
- **D-C. Editor-repo work is planned HERE and implemented by a separate editor Claude
  session**, which receives the requirements this feature produces. The same split the
  vendored `graph_canvas` binary already uses.
- **D-D. Blend mode becomes a per-pass CHOICE with the options people actually use**
  (additive, alpha, opaque, and the usual others), surfaced in the graph, the pass
  panel, and every other place a pass's configuration is visible. Reverses the hardcoded
  `ONE, ONE` of feature 100 D7.
- **D-E. Feature numbering is the implementing session's call.**

## What was REJECTED during research, and why

Recorded so the next session does not re-propose them.

- **A timer before reporting a population that reaches a pass with no `flat in`.** The
  session proposed "stay silent for ~1s, then error", and the maintainer asked why a
  second mechanism was needed when one exists. It is not: `engine.py:969` already skips
  a key the pass does not declare, SILENTLY, and the script-engine bullet of
  `conventions.md ## Design decisions` states the rule --
  writing the script before the shader is a normal authoring step (079 D5). The
  proposal would have added an inconsistent second rule for a case already ruled on.
  **What survives is a different question, stated in I1 below**: `@` is the LOUD
  namespace, so which of the two existing rules `@instances` should follow.
- **Documenting `sb_instanced` as a user-facing builtin.** An earlier draft of this spec
  asked for it. It is engine-internal (I-facts below) and declaring it is refused.
- **Widening the lexer's `PYTHON_BUILTINS`.** MEASURED: across 13 real document scripts,
  6 builtins appear in 43 of 1128 identifier occurrences (3.8%). Buys almost nothing.

---

# Part 1 — The instancing CONTRACT

The engine's defects. None of these is a documentation problem, and the on-ramp in
Part 3 should not land before they are fixed: it would teach a mechanism that misreports
its own state.

## The state machine is EIGHT states, not three

An earlier reading of this session gave three (populated / fullscreen / refused). MEASURED
by an agent driving each state and reading the rendered frame on a 64² canvas:

| # | State | Reached by | Drawn result |
|---|---|---|---|
| a | fields + valid columns, N>0 | normal | one quad per entity (mean R/A `0.017 / 0.069`) |
| b | fields + no population | `not instances`, `core.py:662` | **FULLSCREEN** (`0.198 / 0.788`) |
| c | fields + bad columns | `_INVALID_POPULATION`, `core.py:621` | previous frame held |
| d | fields + REFUSED sentinel | `core.py:657` | previous frame held, **and returns before `_instances_error` is touched, so a stale message survives** |
| e | fields + count exactly 0 | zero-length columns validate clean | **canvas CLEARED to black** (`0.0 / 0.0`) |
| f | compile failed with `InstancedError` | `core.py:415-424` | `program=None`; re-enters `compile()` every frame |
| g | recompile failed on NEW source | early return at `core.py:423` | **old `entity_fields` AND old program keep drawing**: strip shows the new error, canvas shows the old shader |
| h | population to a pass with NO fields | `core.py:662` | silently ignored, forever (see I1) |

States d, e, g, h were missed by the first reading. They matter because any signal design
has to distinguish them: (b) and (e) are two DIFFERENT wrong pictures that both leave the
error strip empty.

## The defects

- **I1. A population reaching a pass with no `flat in` is accepted, stored, and dropped
  every frame, with no signal, indefinitely.** READ, `engine.py:836-842`: the comment
  justifies the silence with "the pass has not compiled ... ordinary on frame one", and
  nothing ever escalates. MEASURED: a document whose `swarm` pass declares no `flat in`
  and whose script sends `@instances` to it reports `soft_errors: []` after a full compile
  and two renders, holds `['pos','radius']` in `pending_instances`, and draws the plain
  fullscreen gradient.
  **ANSWERED by D-F, and answered by generalising rather than by choosing.** Two rules
  existed and disagreed here -- the orphan rule (079 D5, `engine.py:969`) skipped a key
  naming no declared uniform SILENTLY, while the reserved-vocabulary rule made an
  unrecognised `@` key a hard error. D-F collapses both into one: every key that does not
  land WARNS, to the logs and the notifications, with no branch per case. So this is not
  "instancing gets its own handling"; instancing stops being a special case, and 079 D5's
  silence -- which is what hid this defect and is what let two rules disagree -- goes with
  it. The implementing wave should find ONE site, not one per case.
- **I2. State (b) has no signal anywhere.** READ: word-boundary grep for instancing
  vocabulary across `shaderbox/` finds nothing in `tabs/`, `panels/`, `popups/`,
  `copilot/`, `help_content.py`, `profiling.py`, `ui_models.py`. Not the error strip
  (fed by `compile_unit.errors` and the engine's soft errors -- (b) produces neither),
  not the uniform panel (an entity field is a vertex ATTRIBUTE, so `get_active_uniforms`
  has no row for it), not the graph canvas, not the profiler (`document.py:989` counts
  iterations, never instances), not logging.
- **I3. Zero entities clears the canvas to black, silently.** MEASURED: zero-length
  columns pass `validate_population` cleanly (`counts.pop()` on `{0}` returns 0), so
  `mode.value = True`, blending is enabled, and `vao.render(instances=0)` draws nothing
  over a cleared target. Visually distinct from (a) and (b); textually identical to both.
- **I4. `_instances_error` is dead state.** READ: assigned `core.py:666`, cleared
  `core.py:668`, **read by nothing** -- the only three occurrences in the repo are those
  and the declaration at `core.py:282`. It is not logged either. It also goes STALE: the
  `REFUSED_POPULATION` branch returns at `core.py:661` before reaching either assignment,
  so a message from an earlier bad-dtype frame survives into a later refusal (MEASURED:
  case (d) printed case (c)'s message).
  **This falsified the `@instances` bullet of `conventions.md ## Design decisions`**, which
  said "`Pass` can only log". It does not log. ALREADY FIXED -- that bullet now records the
  dead field and points here; nothing more is owed.
- **I5. A failed recompile keeps the old fields AND the old program.** READ,
  `core.py:423` returns before `self.entity_fields = fields` at `core.py:472`. MEASURED: a
  pass compiled with `pos`/`radius` and then given source declaring only `velocity` still
  reports `entity_fields == ['pos','radius']` with a live program and renders the old
  population. Intentional (`core.py:381-382` documents preserving the previous program on
  a failed compile) but absent from the three-state model.
- **I6. Additive blending is hardcoded and excludes most entity looks.** READ,
  `core.py:643`: `enable(BLEND); blend_func = ONE, ONE`, unconditional, no model field, no
  uniform, no escape hatch. MEASURED: two overlapping entities each writing a constant
  `vec4(0.5,0,0,1)` produce R values `{0.0, 0.498, 0.996}` -- the opaque colour doubles in
  the overlap. The shader's alpha does nothing, because the source factor is `ONE` and not
  `SRC_ALPHA`, so alpha accumulates like colour rather than weighting coverage. **A dark
  entity on a light field is impossible outright**: `core.py:633` clears unconditionally
  before every draw, so the destination starts black and `ONE, ONE` can only add. A later
  pass cannot recover it either -- the sum destroys which entity was on top, so depth
  order, occlusion and per-entity opaque identity are gone. Spec 100 D7 reads in full
  *"Additive, no depth, no per-frame sort. Order-independent and the natural look for
  glowing entities; a sort costs CPU per frame for no GPU gain"* -- sound for what it
  argues, and silent about the rest of the range. **D-D reverses it.**
- **I7. `f1` target plus additive is an unguarded trap.** READ, `pass_graph.py:43`:
  `TargetDtype = Literal["f1","f2","f4"]`, and **six of seven shipped examples set
  `"dtype": "f1"` in their `graph.json`**. So 8-bit-plus-additive saturation is reachable
  from the most likely starting point, copying an example. NOTE: an earlier refutation in
  this session said the trap could not be reached because `DEFAULT_DTYPE` is `"f2"`
  (`pass_graph.py:46`). That is true only for a pass created fresh with no explicit
  target; the refutation was stated at the wrong width. **D-D probably dissolves this**:
  the trap exists because blending is not a choice.

## The shape of I1–I5, and one proposal to pressure-test

Five symptoms, one cause: **`Pass.render` is where the population is judged, and it can
only return.** No caller reads a verdict, which is exactly why `_instances_error` has
nowhere to go.

The fix, adopted as D-G: the draw returns a per-frame outcome —

    InstancedOutcome = drew(n) | fullscreen | empty | refused(why) | not_instanced

— which one surface reads. That would close I2, I3, I4 (the dead field becomes the return
value), give I5 something to say, and make I1 observable so a rule can be applied to it.

**ADOPTED (D-G).** A large diff is justified when the symptom is provably one instance of a
systemic class and the fix is a shared primitive applied across every instance; five symptoms
with one cause is that shape. The test that keeps it honest during implementation: the sibling
call sites must be UNCHANGED except for the fix, which a reviewer or a trace has to show. If
that cannot be shown the generalisation was speculation -- but the default is the shared root.

---

## What D-F costs beyond instancing

D-F is not an instancing change. It reverses 079 D5 for EVERY script key, so the
implementing wave touches the whole script-engine reporting path and must expect fallout
the research did not measure:

- **The `skipped` / `driven` / `orphan_keys` split** (`engine.py:89-96`) was built around
  silence being a normal outcome. With every non-landing key warning, `orphan_keys` stops
  being a quiet category and the copilot's `ScriptProbe` (which already reports orphans)
  changes shape alongside C7.
- **`conventions.md ## Design decisions` carries 079 D5 in the script-engine bullet and
  the `@instances` bullet carries the reserved-vocabulary rule.** Both must be rewritten to
  the single rule in the same wave, or the docs will state the behaviour the code no longer
  has. This is the "docs are living" rule, and it is the largest doc edit in the wave.
- **Tests pin the silence.** `tests/test_script_engine.py` and `tests/test_instances_routing.py`
  assert quiet outcomes for orphan keys (the routing file's own docstring calls the silence
  "deliberately SILENT (079 D5)"). Those assertions INVERT rather than relax -- each one is a
  place the new rule must be seen to fire, which is also how the new behaviour gets gated.
- **Notification volume is the risk to watch.** "Warn on every non-landing key, every frame"
  is a per-tick event on a path that runs at frame rate. The warning needs to be
  edge-triggered (on the key's state CHANGING) rather than level-triggered, or an author
  mid-edit gets a notification per frame. The research did not measure this; the wave must.

---

# Part 2 — The COPILOT (premise D-A)

MEASURED baseline: word-boundary grep for `instanced|entity_fields|@instances|vs_quad|flat in`
across all of `shaderbox/copilot/` (8767 lines) returns **zero substantive hits** -- the only
matches are `flat cartoon`/`flat cel` prose and unrelated `isinstance` calls. The copilot does
not know the feature exists at any layer.

- **C1. The probe reports the SHIPPED WORKING example as broken.** MEASURED: `dry_run` on
  Entity Flock returns `driven: set()`, `samples: [(0.0,{}),(0.5,{}),(1.0,{})]`, producing
  the verdict verbatim: *"drives 0 uniforms (update returned an empty dict / only orphan
  keys). Nothing animates and every uniform stays manual."* (`backend.py:405-412`).
  `ScriptProbe.driven` is `set[tuple[str,str]]` of (pass, uniform) and has no slot for a
  population. **The copilot's own feedback loop tells it correct work is broken**, which is
  worse than silence: the agent will "fix" what is right. This is D-A's hardest blocker.
- **C2. The generated SCRIPT API doc asserts something FALSE.** READ, `api_doc.py:20-23`:
  *"Every value is PLAIN PYTHON ... there is no wrapper type to learn"*, rendered as
  *"Legal value shapes, all PLAIN PYTHON (there are no wrapper types)"* over five shapes,
  none of them numpy and none of them `@instances`. The engine REQUIRES numpy under
  `@instances` (`validate_population` rejects a non-`ndarray`, `instanced.py:276-277`). A
  copilot following the doc literally cannot write an instanced script.
- **C3. The gate pinning that doc cannot see the gap.** READ: `test_script_api_doc.py`'s
  `test_every_stub_kind_type_name_has_a_value_shape_gloss` asserts
  `returned <= set(_VALUE_SHAPE_GLOSS)`, where `returned` is parsed from `_stub_kind`'s AST
  and `_stub_kind` dispatches on `moderngl.Uniform`. `@instances` is a reserved key, not a
  uniform, so it is **outside the gate's domain by construction**. The gate reads as "the
  doc covers everything the engine accepts" and enforces "the doc covers every uniform
  shape" -- the checker-narrows-its-own-domain shape. Fixing C2 without fixing C3 leaves
  the next drift unguarded, and the new gate must be BROKEN and watched to fail.
- **C4. The prompt recommends the REPLACED technique by name.** READ, `prompt.py:142-144`:
  *"HEAVY stateful compute (a cloth Verlet sim, particles, a boids flock) -- step the CPU
  state each frame and push the result as an ARRAY uniform"*. Asked for a particle system
  today, the copilot builds the thing feature 100 replaced. Under D-A this is not a deleted
  sentence but an instancing BLOCK on the order of the existing SCRIPTING section: the
  `flat in` declaration, `pos`/`radius` reserved and clip-space, `vs_quad`'s range, the
  f4/i4/u4 dtype rule, C-contiguity, the `(N, components)`/`(N,)` shapes, the
  every-column-needs-a-field bijection, and the blend mode D-D introduces. Plus the
  watershed it must now state: a POPULATION of similar things -> instancing; a small fixed
  vector of parameters -> an array uniform.
- **C5. `ShaderView` carries no entity fields.** READ, `capabilities.py:60-68`:
  `document_id, name, listing, uniforms, errors`. The copilot infers a pass entirely from
  these, so an instanced pass reads as a fullscreen pass with odd inputs, and `edit_shader`
  on it is blind.
- **C6. The dry-run discards populations, and a stated invariant is in the way.** READ,
  `engine.py:881-887`: the population write is guarded by `if values_sink is None`, and
  `values_sink` is set only by `dry_run` (`engine.py:545`) -- the copilot's only synchronous
  script feedback (`project_session.py:869`). The guard exists to protect the 063 dry-run
  isolation ruling, so this is a DESIGN change against a stated invariant, not a wiring fix.
  Likely shape (reasoning, unverified): the probe REPORTS the population without WRITING
  it, since validation already runs and the statistics exist before the write is skipped.
- **C7. `ScriptWriteResult` has no population channel.** READ, `capabilities.py:278-297`:
  `driven` is a list of uniform names, and a population is not a uniform. With no channel,
  `tools/script.py:88`'s `if not result.driven` branch emits the loud no-op on correct work.
- **C8. A population can never cross the tool boundary as data — settled by arithmetic.**
  READ, `base.py:49`: tool args are pydantic `ToolArgs` rendered to JSON Schema, so a call
  is JSON. 20 000 entities x 4 f4 columns is ~320 KB, orders of magnitude past any tool-call
  budget. **So the design is fixed: the copilot writes the GENERATOR (`write_script` /
  `edit_script`, which it already has) and reads back STATISTICS** -- count, per-column
  dtype/shape/min/max/mean, and the validation verdict. The work is in the RESULT types,
  not in a new argument type.
- **C9. `probe_render` has no vocabulary for the instanced case.** READ, `inspect.py:43`:
  it renders, so it sees an instanced pass, but cannot report "instanced, drew N" versus
  "drew fullscreen" -- state (b) reaches the agent as a picture with no label.
- **C10. `add_pass` / `set_pass` do not warn on the I7 combination.** READ,
  `passes.py:109,128`: `set_pass` sets the target dtype with nothing said about `f1` plus
  instancing. Probably dissolved by D-D.

---

# Part 3 — The ON-RAMP (discoverability)

What the original 101 was filed for. It should land AFTER Part 1, so that what it teaches
is true. Ten surfaces were inventoried by the first pass; the agents verified all ten as
real, corrected two, and added five more.

- **D1. The help modal's FIRST section states invariants instancing breaks.** READ,
  `help_content.py:114-119`: *"Every document is one fragment shader. ShaderBox draws a
  full-screen quad and runs your `main()` once per pixel. Three things are fixed: the
  `#version` line, the `vs_uv` input, and a single `vec4` output."* For an instanced pass
  two of those three are false -- the draw is one quad per ENTITY, and `vs_uv` is not the
  coordinate to reach for. **This is wrong text, not missing text**, in the section a new
  user reads first, and it was absent from the original inventory.
- **D2. `vs_quad` is documented nowhere user-facing.** READ: it appears in app code only at
  `instanced.py:37` and in the one example shader. `help_content.py:119` names only "the
  `vs_uv` input"; `prompt_context.py:21` opens *"read `in vec2 vs_uv`"*. It is the one name
  an author cannot guess, and guessing wrong is silent -- reaching for `vs_uv` to shape an
  entity draws a canvas-wide vignette of hard squares with no error (`instanced.py:34-37`).
  (`glsl_docs.VARIABLES` is the wrong home: it is `gl_*`-only, and `vs_uv` is not in it.)
- **D3. Shader autocomplete has no notion of a `flat in` field**, and `vs_quad` is offered
  nowhere. READ, `intel/index.py`: `classes()` enumerates five kinds, none an entity field.
  CORRECTION to the original item: it said `SCRIPT_UNIFORM` candidates are built "from a
  script's returned literals"; `index.py:158-165` builds them from shader DECLARATIONS. The
  conclusion stands, the stated mechanism was wrong.
- **D4. The script stub teaches only the uniform path.** READ, `engine.py:250`:
  `script_stub_for` GENERATES the stub from introspected scriptable uniforms -- it is not a
  static template. So a document whose only pass is instanced gets `(no scriptable
  uniforms)` and `return {}`. That stub is the first thing both a human (`Alt+R`) and the
  copilot (`tools/script.py:113-119`) see for a script-less document.
- **D5. Additive blending is undocumented.** READ: `core.py:643`, rationale only in spec
  100 D7 and a comment inside the example's own shader; absent from `conventions.md`, help
  and the prompt. Under D-D this becomes "document the new per-pass choice" rather than
  "document the constant".
- **D6. The uniform panel and the graph canvas do not mark an instanced pass.** READ:
  zero instancing hits in `shaderbox/tabs/uniforms.py` (118 lines) or
  `shaderbox/widgets/pass_graph.py` (754 lines, the canvas -- note the same-named
  `shaderbox/pass_graph.py` is the MODEL and is also clean).
  Cheap to close: `entity_fields` already sits on `RenderPass` (`core.py:276`), so both are
  reads rather than new plumbing.
  **The correct SCALE, established this session:** instancing is a property of a PASS
  (`entity_fields(unit.flattened)` at `core.py:407`, stored at `core.py:472`) -- not of a
  script (one script feeds many passes; `@instances` sits inside a pass block) and not of a
  document (instanced and fullscreen passes mix freely). It is read from the FLATTENED
  source, so **a `flat in` spliced in from a `lib:` include counts and the open tab may not
  reveal it** (`core.py:403-405`). CAPABILITY is per-pass and fixed at compile; DRAW STATE
  is per-frame. A static badge belongs on the pass surfaces; a live count belongs where
  per-frame state already shows.
- **D7. The README never mentions instancing.** READ, `README.md:33-37`: the scripting
  bullet says *"a physics step, an integrator"* -- exactly the workload -- and stops at
  uniform-driving. This is the itch.io-facing pitch.
- **D8. Script autocomplete knows nothing of `@instances`.** READ, `intel/python.py` (154
  lines): zero hits. Ranked last by the original inventory; completion inside a dict literal
  is the fiddliest of these.
- **D9. Help content's engine-uniform section lacks `vs_quad`.** Its gate walks
  `ENGINE_DRIVEN_UNIFORMS` (`test_help_content.py`). **`sb_instanced` must NOT be added**:
  it is written straight onto the program at `core.py:629`, is outside
  `ENGINE_DRIVEN_UNIFORMS` (so the gate could never have caught it), exists only inside the
  GENERATED vertex stage, and is in `instanced.RESERVED_NAMES` (`instanced.py:42`) so an
  author declaring it is refused. Documenting it would invite exactly that.
- **D10. The example sorts LAST in the browser.** READ, `constants.py:21`: Entity Flock is
  the last of seven in `EXAMPLE_ORDER`. One line, zero risk -- and worth more than "free":
  the flock's `document.json` description (*"Twenty thousand entities steered in numpy and
  drawn in one call..."*, rendered at `popups/examples.py:111-120`) is **the single best
  in-app explanation of the feature**, and it is gated behind selecting the last gallery
  item.

**Ordering.** The original inventory claimed "ordered by harm" and that item 1 was "the only
item that does damage rather than withholding help". Both are false: D1 is wrong text, I1/I2
draw wrong pictures, and C1 produces a wrong agent verdict. Re-order by *wrong text / wrong
picture / wrong verdict first*, or drop the claim and say what the ordering is.

## CHECKED AND REFUTED — do not re-raise

- **A hand-made instanced pass does not risk a white frame from a default target.**
  `pass_graph.py:46` `DEFAULT_DTYPE = "f2"`. (Narrowed by I7: true for a FRESH pass only.)
- **`getattr(render_pass, "entity_fields", ())` at `engine.py:836` is harmless.** MEASURED
  twice, independently: a typo'd pass name carrying `@instances` errors correctly with
  *"no pass named 'swrm' in this document (passes: swarm)"*, because `@instances` is popped
  out of the block at `engine.py:814` but the block key remains, so the
  `pass_name not in document.passes` check at `engine.py:938` still fires. Note the reason:
  a SECOND loop catches it, not the getattr itself. (The separate I1 case is not caught by
  that loop, because there the pass name is valid.)
- **The error strip's messages are good.** `instanced.py:246-300` produces actionable text
  (``nothing declares {x} -- add a `flat in` for it``, ``is {dtype}, expected {y} -- numpy
  casts silently``, ``is not contiguous -- use np.ascontiguousarray``). Once an author is ON
  the path the diagnostics teach well; the gap is purely discovery.
- **Iterations + additive do not discard accumulation wrongly.** `core.py:633` clears
  unconditionally, but iterations chain through `u_prev` feedback, which spec 100 names as
  the sanctioned accumulation route.
- **Blend state does not leak into later passes.** `core.py:644` disables immediately.
- **VAO rebuild is not a per-frame cost** (`core.py:682-683`, buffers double on growth), and
  **dead-stripped attributes are handled** (`core.py:695-701`).
- **Command palette, menus, settings, hotkeys, export** carry no instancing vocabulary, and
  correctly so -- none is a place a rendering technique should be taught.
- **`LIB_FUNCTION`/`SCRIPT_UNIFORM` sharing slot 6 is NOT drift.** An earlier draft
  suspected it; `syntax_colors.py:45-52` documents it as a deliberate colour choice (078
  D2/D12), and `test_every_documented_builtin_draws_in_the_builtin_slot` pins it.

## Also noted

**Nothing outside tests passes `instances=` to `Pass.render`** (`test_instances_routing.py:450`):
a real frame reads `pending_instances`. The parameter is a test-only seam, and any new route
(an off-thread producer, a copilot tool) will be tempted by it -- the validation asymmetry
between the two routes is what the REFUSED sentinel papers over.

---

# Part 4 — The HIGHLIGHTING subsystem (premise D-B)

Reported by the maintainer through its symptom: `self`, `__init__` and method names draw
plain in a `script.py`. The research says this is a half-exported library feature, and the
fix is a general mechanism rather than a Python special case.

## What the symptom actually is

MEASURED on the shipped flock script (123 lines): **302 of 316 identifier occurrences draw
plain (95.6%), and `self` is 41 of them** -- the most frequent word in the file. (Counted
the way `word_classes_apply` walks the buffer: regex word runs over code with strings and
comments blanked. Python's own `tokenize` gives 300/314; the difference is method, and the
lexer-shaped count is the right one for this question.)

READ, `~/src/editor/src/lex_python.odin`: `PYTHON_KEYWORDS` is complete and correct --
`class`, `def`, `None`, `True` all colour. The gap is everything that is not a keyword.
Python has no `self` keyword, no dunder rule, and no notion that the name after `def` is a
definition rather than a use.

## The mechanism, corrected

An earlier draft of this spec said the host pushes classifications through
`highlight_set_ranges`. **That was wrong and it changed the whole design space.**

- **READ: `highlight_set_ranges` does NOT cross the C ABI.** It exists at
  `src/highlight.odin:38` and is a public PACKAGE procedure (also called from
  `ui/main.odin:175,405`), but it is not exported; `ffi/ffi.odin:542` calls it internally to
  publish the combined spans. Correction to an earlier phrasing in this session: "Odin-
  internal" overstated it -- what it is not is *exported*, which makes exposing it cheaper
  than that phrasing implies.
- **READ: the host's only channel is `ed_set_word_class(h, word, class)` plus
  `ed_clear_word_classes`** (`ffi/ffi.odin:216,230`) over `src/word_class.odin`. There is no
  `ed_set_span`. `ed_class_at` is a READ, not a feed.
- **READ: `word_classes_apply` (`word_class.odin:138`) is EXACT-word, whole-buffer and
  POSITION-BLIND**, filling only identifiers no lexer span already covers (`:184`), and
  skipping digit-led runs (`:171`).
- **READ: `language_override` does not cross the ABI either** (`language.odin:36,55`) --
  it takes an Odin proc with a default allocator parameter. The contrast proves the barrier
  is the proc pointer, not policy: `language_override_path` DOES cross (`ffi.odin:984`),
  because it takes only a string and an int.

**So the cases split in two, and this is the load-bearing consequence:**

| reachable by the word table | needs positions |
|---|---|
| `self` / `cls`, dunders, a fixed vocabulary | the name after `def`/`class`, decorators, annotations |

The table can say "`self` is a builtin variable everywhere". It cannot say "the `update`
after `def` is a definition and the `update` in a call is not".

## Why this is FINISHING a stated design, not adding one

READ, and stronger than the session first put it:

- `language.odin:46-47` names the intended hosts: *"a tree-sitter grammar, an LSP's
  semantic tokens, its own compiler's output"*.
- `language.odin:28-29`: *"`None` is a first-class choice rather than an absence: ... a host
  that pushes its own spans through highlight_set_ranges and wants nothing overwriting
  them."*
- `ffi/README.md:946-949` repeats it **as a promise to a C host, in the ABI's own contract
  document**: *"Language `None` is a choice, not an absence. It turns highlighting off and
  leaves spans a host pushed itself alone."*

No ABI call can keep that promise. It is a documented-but-unimplemented feature.

READ, `ffi/README.md:919-922`, the library naming its own gap: *"The classification is
LEXICAL, so it is by spelling, not position: in `float mix; ... mix(1.0, 2.0, mix)` every
`mix` reads as the builtin."*

## Why shaderbox's Python side gets nothing today

Narrower and more concrete than "the host half does not exist":

- **READ, `tabs/code.py:1036`: `if tab.kind != "script": _glsl_index_for(app, editor, tab)`.**
  A script tab never builds an index and so never calls `_feed_classes` (`code.py:405`).
  Corroborated negatively: `build_glsl_index` has one app call site, and
  `set_word_class`/`clear_word_classes` have one each. No second feed path exists.
- **READ: the table is per editor session** (a field on the ffi session struct,
  `ffi.odin:29`, init at `:92`), and `_feed_classes` clears first, so a script tab's table is
  simply empty -- nothing leaks in from a shader tab.
- **CORRECTION to an earlier draft's "two halves" table**: it conflated a kind's SLOT with
  its HALF. `classes()` (`intel/index.py:67-81`) returns FIVE kinds -- `ENGINE_UNIFORM`,
  `SCRIPT_UNIFORM`, `PASS_SAMPLER`, `LIB_FUNCTION`, `OUTPUT_VARIABLE` -- so `LIB_FUNCTION`
  and `SCRIPT_UNIFORM` ARE host-fed and merely draw in slot 6, the lexer's builtin green.
  Membership in `classes()` decides the half, and every one of its five kinds is GLSL.

## The blocking constraint: ZERO free syntax slots

MEASURED by counting the enum and the map. Nine syntax slots exist -- `SYNTAX_1..SYNTAX_9`
(`shaderbox/editor/ffi.py:105-119`), backed by `Theme.syntax: [10]Color`
(`~/src/editor/src/theme.odin:53`) with index 0 reserved for "no class", so **9 is a library
ceiling, not a shaderbox convention**. Four are lexer-owned (2 string, 3 comment, 4 number,
5 operator). Five are host-assigned in `_KIND_SLOT` (`shaderbox/syntax_colors.py`, NOT `intel/`) at 1, 6, 7, 8, 9.

    9 total − 4 lexer-owned − 5 host-assigned = 0 free

The four requested distinctions (`self`/`cls`, constructor, method definition, annotation)
need four slots against zero. **Under D-B the answer is to extend `Theme.syntax`**: a span
mechanism that cannot express more classes than GLSL happens to need is not general. Extending
is cheap in the library (`theme.odin:175-182` already maps slots by name) but touches `Slot`,
`editor_palette`, `_KIND_SLOT`, and the enum test's hardcoded `0 <= slot <= 9`
(`tests/test_intel_sources.py:149-164`).

NOTE a correction inside this session: on first hearing that the library reserves slots 7/8/9
for hosts, the session reported three slots as available. They are available in the LIBRARY and
already spoken for in THIS host.

## What an exported span API must satisfy

Requirements for the editor session, each with the evidence behind it:

- **Versioning.** `highlight_set_ranges` stores the version (`highlight.odin:52`) and
  `highlight_class_at` returns 0 when it differs from the buffer's (`:59-62`). The version
  is the BUFFER's, not a host serial -- and a C host cannot read it: `ed_revision`
  (`ffi.odin:398`) is *not* `buffer.version` (`ffi.odin:47-52` adds a base to stay monotonic
  across `ed_set_text`). So the API must take the version the host BELIEVES it computed
  against and **return accepted/stale**, because an async host (jedi on a worker) will
  routinely answer for text that has changed.
- **Codepoint snapping** is solved (`highlight.odin:44-49`, end snaps UP) but only when the
  buffer is passed -- the `b: ^Buffer = nil` default silently disables it. Pinned at the Odin
  level by `test_span_boundaries_snap_outward`, so the ABI wrapper needs its OWN test that it
  did not forget the argument.
- **Ordering.** Not required by the highlighter (`highlight.odin:63-69` scans without early
  exit, by design, pinned by `test_unsorted_spans_still_resolve`), and `word_classes_apply`
  defends itself by sorting a copy (`word_class.odin:157,192-195`). But overlap resolution is
  first-match-wins in scan order, so for an unsorted set INSERTION order decides -- document
  it or sort on ingest.
- **Ownership is simpler than the word table's.** `Span` is POD (`{start, end: Pos, class: u32}`)
  and `highlight_set_ranges` copies on append, so the host's buffer may die on return -- unlike
  `word_classes_set`, which must clone its string keys (`word_class.odin:44-46`) and pays for it
  with the `key_of` scan (`:115-123`). **Do not put a host string in a span.**
- **`Pos` is a BYTE offset** (`src/types.odin:22`), while every other ABI call that names a
  location takes line/col and converts via `ffi_pos_at_column` (`ffi.odin:421,1014,1760`). No
  precedent exists for byte offsets crossing the boundary. **OPEN: byte offsets (matching `Span`,
  cheap) or line/col (matching every existing call, 2 conversions per span).**
- **Follow the established ABI pattern for bulk data.** Two exist: pointer+cap returning a count
  (`ed_primitives`, `ffi.odin:645`, chosen because per-item ctypes crossing measured ~0.55 us and
  ~1.1 ms/frame), and begin/push-one (`ed_complete_*`, `ffi.odin:160-165`, chosen because "the FFI
  surface inverts control: the host PUSHES"). They agree on
  `ed_set_spans(h, spans: [^]FFI_Span, count: i32, version: u64) -> bool` -- one crossing, a flat
  ABI-owned struct following `FFI_Primitive`'s precedent of not exposing the internal layout,
  copied immediately. **Do not invent a callback or a retained host array.**

## What would make a naive span API WRONG

Three obstacles, all in the FFI layer rather than the core:

- **The retokenize path would overwrite host spans every frame the buffer moves.**
  `ffi.odin:542` is unconditional inside its block and `highlight_set_ranges` CLEARS first
  (`highlight.odin:39`). Host spans must live in their own `Handle` field and be MERGED into
  the span set inside that block -- **exporting `highlight_set_ranges` as-is is wrong.**
- **The gate at `ffi.odin:531` would make host spans inert** under exactly `Language.None`
  with an empty word table -- the one configuration the README promises them in. **This exact
  bug already shipped once for the word table**: the comment at `ffi.odin:522-527` records it,
  *"gating the whole block on a language made word classes silently inert in the one
  configuration the document promises them in. Measured: class 7 on a word read back as 0
  under None and 7 under GLSL."* The first clause needs `|| host_span_count > 0`.
- **`hl_active` would leave them undrawn even if stored.** Computed identically at
  `ffi.odin:224`, `:235` and `:957`, and `ed_class_at` short-circuits on it (`:1006`). Three
  copy-pasted sites -- **the duplicate to REMOVE into one procedure, not to gate.**
- **Copy `wc_revision`'s model** (`ffi.odin:30,24-28,533`): a host re-feeding while the text
  stands still must reach the screen on the next FRAME, not the next keystroke. A span push
  needs the same revision pair.
- **Not an obstacle:** the `Highlighter` itself. No lexer-specific cache, and `:63-65`
  explicitly anticipates "an app-supplied set".

## jedi: already present, and it answers every case

MEASURED, and the numbers were re-run by an agent (the first pass's cold figure did not
reproduce and is corrected here):

- jedi is a declared dependency (`pyproject.toml:46`), in-process
  (`intel/python.py:20-26`), and serialized on ONE thread (`worker.py:53-62,92`) with a `WARM`
  kind (`worker.py:23`) and per-kind coalescing carrying a revision (`worker.py:33,66`).
- `Script.get_names(all_scopes=True, definitions=True, references=True)` returns 303 names on
  the flock script and answers **all five cases with line/column**: `self` as `type=param` at
  each definition and `type=statement` at each use, `update` as `type=function` with
  `is_definition()` true at 13:8, `property` as the decorator's own name, annotations as
  statements.
- Cost: **median 7.2 ms warm** on the 123-line script, **30.4 ms at 4x size**, **~37 ms cold**
  (Script() 29 ms + get_names 7.4 ms), plus a one-time **~42 ms `import jedi`**. An earlier
  figure in this session said 45.8 ms cold; it does not reproduce.

## The timing problem, which has no analogue today

The word-class feed is **synchronous on the frame thread** -- `_glsl_index_for` builds the
index inline and `_feed_classes` pushes in the same call, so "index rebuilt" and "classes fed"
are one instant and the buffer cannot move between them. jedi is the opposite:
`_python_candidates` (`code.py:512`) submits and returns `()`, and `_pump_python`
(`code.py:533`) lands the answer frames later.

- **Colour would lag text by roughly one to three frames while typing.** Qualitatively unlike
  completion latency, which is invisible because the popup simply is not there yet; here the
  user sees WRONG colour, briefly, on every keystroke.
- **`PythonRequest.matches` (`worker.py:38`) requires caret agreement** and `_pump_python:543`
  drops on failure. A span result must NOT be dropped because the caret moved -- spans are a
  property of the text. A span request needs a revision-only predicate.
- **On a revision mismatch, two policies, and the library forces the choice.** Either drop and
  re-request (colour falls to plain during a fast burst) or push against the current version
  (visibly smeared spans on multi-line edits, since the library offers no offset adjustment).
- **A mitigation worth keeping:** put `self`/`cls`/dunders on the WORD TABLE (instant, no jedi,
  position-blind but correct for those names) and spans only for the genuinely positional
  distinctions. That halves the latency exposure -- and it is the permanent right home for that
  half, not a stopgap.

## The word table SURVIVES, and the reason is sharper than "two jobs"

READ, `word_class.odin:19-20` and `ffi.odin:232-233`. The durable argument is an asymmetry the
session first missed: **a name-keyed fact is edit-invariant; a position-keyed fact is not.**
`word_classes_apply` re-runs against the new buffer positions on every retokenize, so a fixed
vocabulary survives every edit for free. `highlight_class_at:59` discards a whole span set on any
version change, so a span-fed host must recompute and re-push on EVERY keystroke. Using spans for
engine uniforms would mean recomputing a constant list every frame the buffer moves.

## GLSL should stay on the word table

READ: **there is no GLSL parser.** `intel/glsl.py` is nine regexes over comment-stripped text,
and `index.py:130-131` calls its locals scan "regex-shaped and body-blind". Nothing in
`pyproject.toml` provides one (`pyparsing` is transitive via `uv.lock` only), and the library's
`lex_glsl.odin` is a hand-written tokenizer emitting the 7 `Token_Class` members. `classes()`
returns `dict[str, SymbolKind]` -- one kind per NAME, positions absent by type.

GLSL would GAIN declaration-vs-use, scoping, and the `float mix; mix(...)` case. It would LOSE
the word table's edit-invariance and inherit the staleness problem, and the spans would have to
come from widening the same regexes to emit offsets -- more code, same job, more ways to be
wrong. None of the gains is a current complaint.

## `Token_Class` bounds the lexer route

READ, `language.odin:13-21`: exactly 7 members (None, Keyword, String, Comment, Number, Operator,
Builtin), deliberately language-neutral -- *"adding a language is adding a lexer and a case"*.
So teaching the Odin lexer about `self` would colour it `Builtin`, the same as `len`: it cannot
tell a builtin variable from a constructor from a method name. **CAVEAT recorded so the next
session is not misled by the number**: 7 is the LEXER's vocabulary, not the palette's. The theme
has 9 host-visible slots (above), which is what lets a host-fed route express distinctions the
lexer cannot. D-B chooses the host route, so `Token_Class` need not change.

## The enum gate, verified by breaking it

READ + MEASURED: `tests/test_intel_sources.py:145` `test_every_kind_has_a_color` walks
`for kind in SymbolKind:` and enforces four things per kind -- a 4-component colour, a rank, a
slot in `0..9`, and (for a slotted kind) that the popup colour IS the slot's palette entry. An
agent broke it: adding `PY_SELF` with no colour fails with `KeyError` at `syntax_colors.py:20`;
with a colour and rank but no slot it fails at `syntax_colors.py:77`. Both halves bind.
**Its blind spot:** it walks the enum but has no coverage that a kind is ever PRODUCED. A kind
with a colour and a slot that nothing emits passes clean.

## Corrections to earlier statements in this session

- Cold jedi is ~37 ms, not 45.8 ms.
- "Only `min` and `abs` are used from the lexer's builtins" is FALSE even in the file it cited
  (`dict` appears at line 48, in the `-> dict` annotation). MEASURED across 13 real scripts: six
  builtins (`float` 11, `min` 8, `abs` 8, `dict` 8, `max` 7, `len` 1), 43 of 1128 identifier
  occurrences, 3.8%. The conclusion survives; the evidence sentence did not.
- The flock script is 123 lines, not 124.
- Worker coalescing is keyed on `request.kind` and `_next` (`worker.py:89`) pops in INSERTION
  order, not by priority. A span request would be a fourth kind sharing that dict.

---

# The split — DONE

The four features are written. This file stays as the RESEARCH RECORD they cite for
evidence and measurements; it is not itself a plan any more.

| Feature | Covers | Gates on |
|---|---|---|
| **102** script_reporting_contract | I1–I7, via D-F and D-G | nothing |
| **103** copilot_instancing | C1–C10 | 102's outcome seam |
| **104** instancing_onramp | D1–D10 | 102 and 103 |
| **105** semantic_highlighting | Part 4; requirements in `02_editor_requirements.md` | nothing — parallel |

**Why this order.** I1–I5 and C1–C3 are one class -- *the system knows something is wrong
and tells no one, or tells the opposite*. Documentation cannot fix either, and shipping the
on-ramp first would teach a mechanism that misreports its own state.

**105 runs in parallel** because it shares no code with the other three. Its editor-side
requirements are written and are handed to the editor session when that feature starts (D-C),
not before.

## Open questions

**None blocking.** Every question this research raised has been answered:

- **I1's boundary** and the **`InstancedOutcome` shape** -> D-F and D-G.
- **The span ABI's coordinate unit, and every other editor-side technical choice** -> the
  editor session's, via `02_editor_requirements.md`. Do not decide it here.
- **The 063 dry-run ruling and populations** (C6) -> **the correct solution, not a bend.**
  063 protects one property: a `dry_run` leaves the live document byte-identical. Reporting
  is not writing, and `validate_population` already runs before the write is skipped
  (`engine.py:881`), so the statistics exist at that point. The probe REPORTS the population
  -- count, per-column dtype/shape/range, the validation verdict -- and still writes nothing.
  That satisfies 063 as stated rather than weakening it. The implementing wave must gate the
  isolation itself (a `dry_run` over an instanced document leaves `pending_instances`
  untouched), because that is the invariant a reporting path could silently break.
- **Which of D1–D10 are in** -> **all of them.** The list is ten items, none large, and the
  set is what makes the feature discoverable; shipping a subset leaves an author on a partial
  on-ramp, which is the state the feature exists to end. Order by the corrected ranking
  (wrong text, then wrong picture, then wrong verdict, then silence), so D1 and D2 lead.
