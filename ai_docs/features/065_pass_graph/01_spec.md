# 065 — The pass graph: specification

**Status: DRAFT, awaiting plan-lock.** Anchor for facts: `00_facts.md` (six-agent round, verified by
hand where load-bearing). Predecessor: `064_multistep/` — built, reverted (`34f6d19`), kept for the
record of what was tried and what it cost.

## What ships

A **document** holds several **passes** forming a DAG. Each pass is its own `.glsl` file with its own
`main()`, its own render target, and its own uniforms. One pass is the document's output. A project
holds several documents, as today it holds several nodes.

## The naming decision

Today "node" means BOTH the document you open and save AND the thing that renders. Every attempt to
add a second render unit was pushed back by that collision — in 064 passes could not be nodes, so
they became functions inside one file.

| Concept | Word | Notes |
|---|---|---|
| What you open, save, share, export | **document** | `Document`, `documents/<id>/`. Zero prior uses of the identifier in the package. |
| One shader, one target, one draw | **pass** | `Pass`, `passes/<name>.glsl`. Matches Shadertoy/ISF/SHADERed; `render_pass` where a bare `pass` would be the Python keyword. |
| An edge | **input** | A pass's named input, filled by another pass. Not "wire" (implies a spatial UI we may not build), not "channel" (implies Shadertoy's positional slots, which the survey names as a documented footgun). |
| A pass reading itself | **feedback** | Its previous frame. |

**"Node" is retired from the domain entirely.** Not repurposed to mean "pass" — a word that has meant
two things for the project's whole history will keep dragging the old meaning along, and the survey
shows only the spatial-patch tools (TouchDesigner, VVVV, freska) call a per-draw unit a node, which
is exactly the collision we are removing. `NodeTab`, `TreeNode`, `EngineNode` are unrelated
namesakes and stay.

**"Step" is unavailable**: spent by 064, and `copilot/state.py::StepRecord` already means a tool
call's progress marker.

## D-decisions

**D1. A pass is a file.** `documents/<id>/passes/<name>.glsl`, an ordinary fragment shader with its
own `main()`. It compiles alone, so it gets its own `CompileUnit`, its own `SourceMap`, and its own
errors with correct file and line — which is what "genuine separate shaders" means and what the
one-file design could not give.

**D2. The graph is engine-level machinery in `graph.json`, never comments.** Maintainer decision,
carried as a fixed premise: *"we don't use comments for semantically bearing stuff."* 064's rework
measured the cost of the alternative — a comment cannot be checked, so it needed near-miss
detection, transposition handling, orphan inference and a scoping rule, two of which shipped as
regressions. `graph.json` is app-written derived state, like `node.json` today; the user edits it
through the UI, never by hand.

It holds: which passes exist, which pass fills which input of which pass, each pass's target
configuration, which pass is the output, and layout if a spatial UI ever wants it (kept in a separate
key so losing layout never costs the effect — freska's split, which it got right).

**D3. A pass declares inputs as ordinary sampler uniforms; the graph says what fills them.** A pass
writes `uniform sampler2D u_src;` and `graph.json` records `blur.u_src <- bright`. Binding is BY
NAME — the survey's clearest negative finding is SHADERed's positional slots, where the same slot is
`posTex` in one shipped example and `clr` in another. An unfilled input is not an error; it reads
black, so a half-built graph stays usable while you build it (freska's silent-no-op rule, which it
got right).

**D4. Each pass owns its uniforms.** Maintainer decision: *"shared uniforms are the coincidence and
the specific issue feature, not the generalisation."* No sharing mechanism. Separate files give this
for free, and it removes the union-uniform problem 064 hit — where "the live program" was undefined
across N variants and `UINode.save` risked pruning tuned values away.

**D5. Evaluation is a topological sort with memoization; each pass draws at most once per frame.**
Ported from 064, where the logic was correct. The bug class is documented twice in this repo's own
history: an earlier deleted DAG and 064 both shipped a shared ancestor re-rendering once per
consuming path, and **tests caught it neither time** — a duplicated pass renders the CORRECT picture,
just N times, so it reads as slow rather than wrong. **The invariant is asserted on every plan the
test module builds, not in one test.**

**D6. A pass reading itself is feedback, with implicit double-buffering.** The engine hands it the
previous frame; the user never manages a pair. Every surveyed tool that supports feedback makes it
implicit. The self-edge is excluded from the cycle check but kept in the model — the trap that sinks
a naive implementation. The swap is tied to a FRAME, not a render CALL: the live loop renders the
current document twice per frame and the copilot probe renders twice back to back.

**D7. A non-self cycle is a loud error, reported per pass.** Not a hang.

**D8. Recompile only the pass that changed.** Measured: ~1.3 ms per program, so 16 passes is a ~21 ms
hitch if everything rebuilds on every keystroke-save. Separate files make per-pass recompilation
natural; this is designed in from the start rather than retrofitted.

**D9. Per-pass target configuration is document state with working defaults**, edited in the panel,
saved in `graph.json`. Defaults: full canvas size, **`f2`** (063 measured `f1` saturating at 255 on
the first accumulate pass where `f2` reached 7.0), linear, clamp (moderngl defaults to repeat, which
is wrong for a feedback border). A pass renders correctly before anyone opens the panel.

**D10. Export renders the output pass, and starts cold.** `export_isolation` already re-instantiates
a stateful script per export so a render does not depend on how long the app has been open; a
feedback target is the same class of state and 064 had to learn this the hard way (the same document
exported twice differed).

**D11. The copilot addresses a pass through the ADDRESS SCHEME, not through new tools.**
`copilot/address.py` is explicitly "the single round-trip parse/build point" and has two prefixed
kinds today. A fourth kind naming a pass is a small contained change that every existing tool
inherits. This matters because the tool budget is already over its own threshold — 31 tools, 18
eager, against a ">16 eager" trigger the roadmap records as blown at 24. **A design needing new tools
is expensive; one extending the address scheme is nearly free.**

**D12. One script per document, not per pass.** The scripting engine keys `dict[node_id, NodeScripts]`
with a single `behavior` per entry and reaches the engine only through a two-member protocol. A
script drives uniform VALUES over time; splitting it per pass would multiply the surface for no
stated need, and 048 already spent five feature numbers converging on one-script-per-unit. Revisit
trigger: a real effect wants two passes driven by genuinely independent CPU state.

**D13. `Document` and `Pass` are separate types.** The facts round found only THREE methods fusing
shader-identity with canvas-identity — `render`, `compile`, `get_active_uniforms` — so the split is
clean rather than a rewrite. A `Pass` owns source, program, target, uniforms. A `Document` owns the
passes, the graph, the output choice, the script, and export. **`canvas` travels DOWN with the pass**,
not up: each pass needs its own target.

**D14. One bad pass file must not cost the document.** `load_nodes_from_dir` today catches any load
exception and skips the whole directory. With N pass files, a document loads with the passes it can
read and reports the ones it cannot.

## What is deliberately NOT in this feature

- **A spatial graph editor.** The graph is edited as a list of passes with their inputs. A canvas UI
  is a separate feature, and `imgui_node_editor` is available if it is ever wanted — with a verified
  constraint: only `BeginChild` hard-asserts inside its canvas (`BeginListBox` and
  `InputTextMultiline` silently do nothing, which is worse), and `ed.suspend()`/`ed.resume()` is a
  sanctioned escape at the cost of the canvas coordinate space.
- **Cross-document composition.** A document's graph is self-contained. The maintainer wants the door
  left open, so nothing in the format forbids it later; nothing implements it now.
- **Migration.** Maintainer decision: build from scratch. Existing projects are the dev sandbox and
  shipped examples, both authored by hand in the new shape.
- **MRT, 3D, per-object simulation.** Out, as in 064.

## Files

New: `shaderbox/document.py` (the `Document`/`Pass` model + the graph), `shaderbox/pass_graph.py`
(topological order, cycle detection, feedback), `shaderbox/widgets/pass_list.py` (the panel).

Reshaped: `core.py` (`Node` splits into `Pass` + what `Document` owns), `ui_models.py`,
`project_session.py`, `watch.py`, `paths.py`, `copilot/address.py` + the tool surface, `tabs/node.py`
-> the document panel, `editor_types.py` (a `pass` tab kind).

Untouched, verified design-independent: the exporters (they consume a `RenderedArtifact` — bytes,
GL-free), `shader_lib/` and the resolver, `shader_errors.py`, `model_salvage.py`,
`ui_primitives.py`/`theme.py`, the scripting engine's protocol seam.

## Verification

Each check fails for exactly one reason.

1. A document with one pass behaves exactly as a node does today, including its editor tab.
2. A two-pass chain renders B-reads-A, not A alone.
3. Order is topological on a diamond, and the shared ancestor draws ONCE. Asserted on every plan.
4. A feedback pass reads its own previous frame and accumulates across frames.
5. A non-self cycle reports an error per pass and does not hang.
6. An error in pass 2 lands in the strip with pass 2's file and line, and click-to-jump works.
7. Editing one pass recompiles only that pass.
8. An unfilled input reads black; the document still renders.
9. A malformed pass file costs that pass, not the document.
10. Export renders the output pass, and two exports of a feedback document are identical.
11. The copilot can author a two-pass document with no new tools.
12. Save/reload round-trips the graph, the target configs, and every pass's uniforms.
