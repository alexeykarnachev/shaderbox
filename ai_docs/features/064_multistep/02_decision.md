# 064 — The UI/UX decision, and why

**Status: DECIDED (maintainer: "proceed in the most logical and structurally correct and robust
way"). Supersedes nothing; `00_scenario.md` R1-R10 remain the anchor.**

Four UI/UX proposals were produced against `00_scenario.md` (saved verbatim in `design_round/`),
then judged by three independent reviewers on three lenses. **The three lenses picked three
different winners, and that disagreement is the finding** — it says the authoring SEAM and the
authoring SURFACE are separable, and should be separated.

| Lens | Ranking | Winner's reason |
|---|---|---|
| Requirement fidelity | **B > C > D > A** | B's claims all survive tracing to full scenario scale; A's two "costs nothing" cells break |
| Grain fit | **B > D > A > C** | Only B rated NATIVE; it extends seams `conventions.md` names as its own revisit triggers |
| Build cost / ship odds | **A >> D > B >> C** | A is ~half the next-shortest path and the only one where a working cascade exists BEFORE any UI |

## THE REWORK: steps are named, not commented (supersedes the seam chosen below)

**Read this before the rest of the file.** Everything below chose proposal A's seam, in which a step
was declared by a comment on its sampler — `uniform sampler2D u_blur;  // step, scale: 0.5, f2`.
**That seam was replaced after it was built.** The rest of the document is kept as the record of how
the four proposals were judged, which is still the reasoning behind everything EXCEPT the seam.

**Why it changed.** The maintainer had already answered this question before the design round ran:

> "this should be a special uniform name and type like for example: `uniform sampler2D
> u_pass0_blablabla` or `u_pass_blablabla` something like this. **We don't use comments for a
> semanticly bearing stuff.** When a pass defined in the code we just provide the default
> parameters, the user can change them later in the editor."

The design round then produced four proposals, the judges compared them, and A's comment rider was
adopted from that comparison — without noticing it contradicted a decision already taken. The
instruction was not re-read; the agents' framing won.

**The instruction was also the better design, and the evidence is what the rider cost.** A comment is
not part of the language, so it cannot be checked. To stop an ordinary English comment from breaking
a shader, the rider needed near-miss detection (`// stp`), transposition handling (`setp`), orphan
inference, reserved-name checks, and a scoping rule so `// stop` on a texture sampler did not refuse
the compile — two of which shipped as real regressions and were found by post-impl review. A name
deletes that apparatus rather than hardening it: `step_spec.py` went 460 -> 292 lines and the entire
typo error class went with it.

**The rider had also inverted the instruction's second half.** Parameters lived in the shader and the
panel was read-only; the instruction said defaults in code, changed later in the editor. Now a
step's target (scale, format, filter, edge, persist) is `UINodeState.step_configs` with working
defaults, edited by combos in the Steps rows and saved to `node.json`.

**Current design, in one line:** a sampler named `u_step_<name>` is a step, its body is
`void step_<name>(out vec4 o)`, and its target is configured in the panel. See `03_engine_spec.md`
D1/D2.

**The process lesson, which is the durable part:** a decision the maintainer already made is not
re-opened by a later exploration of the same question. The design round should have been given that
answer as a constraint rather than as one option among four. Filed at fleet level, not here, because
it is not about shaders.

## The decision

**Build A's seam as the engine. Build B's or D's surface on top of it, later, as a separate
feature.** *(The seam was later replaced — see the rework section above. The surface half of this
decision stands and shipped.)* They are not rivals — both B and D say so themselves:

- B: *"if a text form is wanted later, the chips writing that comment line is the natural
  serialization."*
- D: *"the slot row and target combos could be a VIEW OVER such a syntax, and if the backend phase
  picks that seam this design sits on top unchanged."*

So A is their backend. Ship A's seam and the surface becomes a pure-UI increment with no engine
risk left in it. **C is the only proposal that cannot be reached this way** — a board that edits a
comment syntax must round-trip-write the user's shader text.

### Why the seam decision goes to A even though A ranked last on two lenses

The deciding fact is structural and no designer weighed it: **the codebase is singular everywhere.**
`node.source` and `node.compile_unit` are reached from across the codebase, several of those in
`copilot/backend.py`;
`EditorTabKind` (11 string-compared sites, so pyright catches nothing when a member is added),
`watch.py::reload_node_if_changed` hardcoding `sources[0]` as "the root shader", and `copilot/address.py` carrying
three address kinds with **no slot for a step**.

B, C and D each multiply files and pay 150-300 lines of pure plumbing in the highest-blast-radius
files, with no user-visible result. **A multiplies only `program`** — referenced at 4 sites.

And the copilot axis, which the design round under-priced: post-058 the copilot is the code-shipper.
Under A it authors a whole multi-step effect with **zero new tools**, because `write_shader` already
writes the one file that contains the steps. Under C it cannot author one at all.

### Why C is dead

Not a judgement call — **falsified by experiment.** `imgui_node_editor` hard-asserts on
`BeginChild` inside its canvas. Reproduced:

**Corrected later, and the correction matters for any future graph UI:** the library PRINTS a
notice naming three widgets as incompatible, but only `BeginChild` actually asserts —
`BeginListBox` and `InputTextMultiline` return cleanly and simply do nothing, which is harder to
notice, not easier. And `ed.suspend()` / `ed.resume()` is a sanctioned escape: a `begin_child`
wrapped in that pair runs with no assert (verified). What it costs is the canvas's coordinate
space — content drawn that way is an overlay ON the canvas, not a widget inside a node body that
pans and zooms with the graph. So the ban is on child windows living INSIDE a zoomed node, not on
child windows near a graph.

```
RuntimeError: IM_ASSERT( false && "ImGui::BeginChild should not be called inside a
node editor canvas" )   ---   imgui.cpp:6789
```

`preview_cell` IS a `begin_child`; `draw_ui_uniform`'s `"text"` input IS
`input_text_multiline`. C's whole "reuse the existing primitives, native on day one" premise
crashes the frame, and the fix is forking the two most-reused UI primitives in the codebase. Its
~1100-line estimate was low by 40-60% before counting those forks.

(The binding itself is real and complete — 126 symbols, a working demo, and it renders correctly
inside ShaderBox's nested child/tab structure. C's own verification was honest. The canvas
restriction is what kills it, and C flagged it as an unknown rather than testing it.)

## Corrections to `00_scenario.md` established by the round

1. **A's "N files fragment the SourceMap" is FALSE.** `SourceMap` is `dict[int, Path]` — multi-file
   by construction — and `shader_lib/resolver.py` already interns every contributing source and
   emits `#line N M` per lib function. Multi-file error locality in one compile unit is the
   machinery that ALREADY SHIPS (feature 032). A's one-file constraint pays for a solved problem.
   Its conclusion survives on different grounds (the singular `compile_unit` plumbing).
2. **A's "every step's uniforms are active on the ONE program" is FALSE, measured.** Compiling
   `#define SB_STEP N` variants of one source, each variant exposes only ITS branch's uniforms
   (`['a_pos','u_k']` vs `['a_pos','u_scene']`). Since `UINode.save` prunes UI rows against the live
   program's uniform set, "the live program" is undefined with N programs. **The panel must present
   the UNION across variants with per-variant write dispatch.** This is A's largest unpriced item
   and a real data-loss risk on save — the class commit `b25d9e3` was written to prevent.
3. **The three owed fixes were already landed** in `a4758cd`; the scenario's "grep returns nothing"
   for `gc_mode` is stale. The fourth (the false `u_time` comment) landed with them.
4. **Two further prerequisites the round surfaced**, landed in `e2cbb03`:
   - `texture_to_pil` read every texture as 8-bit, so an `f2` target (8192 bytes where
     `frombytes` expects 4096) returned a plausible-looking WRONG image. Silent corruption,
     unreachable only while `Canvas` was always `f1`.
   - `preview_cell` had no stale mark, so with "Render all" off the node grid showed photographs of
     the past with nothing saying so. A truthfulness bug in shipped code.

## The gap NO proposal handles: R7 x R9

**All four cover R7's storage and all four fail its display.** In the Living Scene the HDR steps and
the watchable steps are THE SAME STEPS — the cascades are float because 8-bit is fatal, and
bright-extract exists to isolate values above a threshold.

`grep -rn "tonemap\|exposure\|hdr" shaderbox/` returns **nothing**. `dl.add_image` had no tint
argument until `e2cbb03`; `image_with_bg` is a straight blit. So today: focus cascade level 4 and
the preview shows **white**. Bright-extract: **white**. The float format that makes the effect
possible is what makes it unviewable.

Two proposals (A §8.4, D §10.5) named the gap and filed it as a known cost; two did not see it; all
four marked R7 COVERED.

**Refined after the engine landed, by measurement.** The MAIN preview is NOT affected: it draws the
node's `f1` canvas, which the user's own `main()` has already tonemapped — a step writing `7.0`
composited through `main()` shows a correct 7..216 gradient, not white. The gap opens only where a
float STEP TARGET is displayed RAW, which is precisely what a step strip or contact sheet does.

So the view transform is a prerequisite **of the surface feature**, not a debt owed now, and the
surface feature must carry it or every step worth debugging previews as pure white. The export half
landed in `e2cbb03` (`texture_to_rgba8`, which tonemaps out-of-range values rather than truncating
the buffer).

## What ships in which order

1. **`e2cbb03`** — Canvas dtype/filter/wrap, dtype-aware export, the stale mark. DONE.
2. **The engine seam (A's shape)** — step declarations, multi-variant compile, the scheduler
   (topological order + memoization + implicit ping-pong), per-step targets, the union-uniform
   panel. Verified working: `#define` variants compile and introspect per-variant, and a `#line`
   restore keeps error lines exact — but `resolve_usage`'s no-lib FAST PATH emitted no `#line` at
   all, so injecting a define there shifts every error by one. **3-line fix, owed.**
3. **The live HDR view transform** — the R7xR9 gap above.
4. **The authoring surface** — B's list or D's sheet, decided with a working cascade on screen.

Cascades is the acceptance test, not the first milestone.

## Post-impl review: what three reviewers found

Three adversarial reviewers (spec fidelity, correctness hunt, regression sweep) ran against the
landed engine. Every finding was reproduced before it was fixed. The engine core — variant
compilation, driver-sourced edges, memoized topological evaluation, ping-pong, the union, the D11
predicate — came back faithful, each verified on a real GL context rather than read.

**Four real defects, all fixed:**

1. **`make check` was RED and being reported green.** Eight ruff errors sat behind a grep of
   pyright's summary line that never looked at the exit code. The most expensive kind of process
   failure — a gate that appears to pass. Now checked by exit code.
2. **`persist` was a no-op.** `invalidate()` preserved the target and the next `compile()` freed it
   unconditionally; since every real flow is invalidate-then-compile, the flag never once worked
   while the Help panel advertised it. The fix depends on ordering: `self.steps` is assigned before
   the release, so the persist set comes from the new declarations.
3. **Export did not start cold.** A feedback step carried however long the app had been open, so the
   same node exported twice differed (measured 13 vs 255). `export_isolation` already re-instantiates
   a stateful SCRIPT for exactly this reason; a step target is the same class of state.
4. **Two regressions in the step diagnostics.** `// stop` on a texture sampler is an ordinary English
   comment and `void step_march(vec2 p)` an ordinary helper name; both refused the whole compile.
   The checks helped an author writing a chain and broke everyone who was not, so they are now scoped
   to a shader actually authoring steps.

**The process finding, which matters more than any of them.** A memoization bug — a shared ancestor
re-rendered once per consuming path, the exact defect the deleted DAG shipped — reached `dev` and
survived four commits under a green 870-test suite. It came from a reviewer mutation-testing in the
live worktree while the implementer committed, and it was caught by *rendering the effect and reading
the printed evaluation order*, not by the suite.

Two mechanisms now close that class:

- **The invariant is asserted on every plan the test module builds**, not in one case. The diamond
  test asserted `order.count("base") == 1` — one name, one case — while eleven other tests built
  plans and never checked for duplicates. Reintroducing the bug now turns 8 of 12 red where it
  previously turned 0.
- **A mutation test verifies its own restore before anything else runs.** A green suite that predates
  a file write proves nothing about what is on disk.

Worth stating why this class hides: a duplicated step still renders the *correct picture*, just N
times. It reads as "slow", not "wrong", so no visual check finds it. Only the order itself shows it.
