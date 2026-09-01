# 065 — The pass graph: specification

**Status: LANDED — all nine stages; checks 13-16 owed (a display, and a dogfood run).**
Plan-locked before implementation and reviewed by three agents (completeness, adversarial design, cold-start);
D10b, D11, D12, D15 and D16 were added or corrected in response, and the verification list was made
falsifiable. ALL NINE STAGES ARE LANDED; checks 13-16 need a display and a dogfood run. Anchor for facts: `00_facts.md` (six-agent round, verified by
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
is exactly the collision we are removing. `TreeNode` (the lib picker's filesystem tree) and
`EngineNode` (the scripting protocol) are unrelated namesakes and stay. `NodeTab` was listed here
as a namesake too and turned out not to be one: its `NODE` member IS the document tab, so it is
domain — stage 5 renamed it `DocumentTab.DOCUMENT`.

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

**D10. Export renders the output pass, and starts cold.** `export_isolation` already
re-instantiates a stateful script per export so a render does not depend on how long the app has
been open; a feedback target is the same class of state and 064 had to learn this the hard way (the
same document exported twice differed).

**D10b. Viewing an intermediate pass is separate from choosing the output, and goes through a
tonemap.** Added during review. Exactly one pass is the output, but a debugging user needs to LOOK
at pass 4 without changing which pass the document exports — otherwise the only way to inspect is to
mutate saved document state and change it back.

So: a **view selection** in session state (never `graph.json` — a debug view is not a document
edit), defaulting to the output. And displaying a float target goes through Reinhard + sRGB, because
D9 makes `f2` the default and **a float target blitted raw is pure white** — 064 measured exactly
this, and its own post-mortem ruled that the view transform belongs to whichever feature ships the
surface. 065 is that feature.

**D11. The copilot addresses a pass through the address scheme — AND the working set becomes
document-grouped.** The address half is contained: `copilot/address.py` is the single round-trip
parse/build point, so a fourth kind is a small change every tool inherits. That matters because the
tool budget is over its own threshold — 31 tools, 18 eager, against a ">16 eager" trigger the
roadmap records as blown at 24.

**But the address scheme alone is insufficient, and the first draft stopped there.** Three verified
blockers, all in `00_facts.md` and all dropped from the first draft:

- **The working set caps at SIX members** (`copilot_working_set_max_nodes`, enforced by an eviction
  loop in `project_session.py`). If a PASS is a member, an 8-pass document cannot fit at all — the
  model would evict passes of the very document it is editing. The fix is not a bigger number: one
  member becomes one DOCUMENT, rendering its passes as sub-sections, which is the shape
  `WorkingSetView` already uses for its optional script listing.
- **`node_tree()` is flat** — `(id, name, has_errors, is_current)`. The model would never be SHOWN
  that a document has passes, so it could not construct a pass address it has never seen. The entry
  gains a pass list.
- **Errors are one flat list per member.** Per-pass errors need a per-pass slot, or the model edits
  a pass blind and cannot attribute what comes back.

The headline still holds — no new tools — but this is a shape change to `WorkingSetView` and
`NodeTreeEntry`, not a contained change to a 43-line module.

**D12. One script per PASS, keyed `(document, pass)`.** Corrected during review; the first draft
said one script per document and that was wrong.

The engine resolves a script's returned dict against a flat name map --
`active = {u.name: u for u in node.get_active_uniforms()}` -- and rejects an unmatched key as an
"orphan key". With D4 making every uniform pass-scoped, a document-scoped script has **no way to
name which pass's `u_ray_count` it drives**. That is the same addressing hole D11 fixes for the
copilot, left open in the other subsystem.

The 048 citation in the first draft was also used wrongly. 048 converged on *one script per unit,
bound by existence* -- and under D1 the unit that owns uniforms is the PASS. Keeping 048's number
while changing what the unit is meets the number and discards the form.

So: `scripts/<pass>.py`, bound by existence exactly as today, keyed `(document_id, pass_name)` in
the engine's dict. `EngineNode` is satisfied by a `Pass`, which is the protocol's natural fit --
`uniform_values` plus `get_active_uniforms()` are pass-level under D4. The engine stays the
shallowest subsystem in the codebase; it just keys on a pair.

Revisit trigger: a real effect wants ONE piece of CPU state driving uniforms across several passes,
and repeating the script per pass is genuinely worse than a shared one.

**D13. `Document` and `Pass` are separate types.** The facts round found only THREE methods fusing
shader-identity with canvas-identity — `render`, `compile`, `get_active_uniforms` — so the split is
clean rather than a rewrite. A `Pass` owns source, program, target, uniforms. A `Document` owns the
passes, the graph, the output choice, the script, and export. **`canvas` travels DOWN with the pass**,
not up: each pass needs its own target.

**D14. One bad pass file must not cost the document.** `load_nodes_from_dir` today catches any load
exception and skips the whole directory. With N pass files, a document loads with the passes it can
read and reports the ones it cannot.

**D15. The panel's six verbs, and wiring as a closed set.** Added during review — the first draft
specified the engine to the millimetre and the surface in one sentence, which is the same imbalance
that produced 064's rejection.

The pass list supports exactly six verbs: **add** a pass (creates `passes/<name>.glsl` from a stub
and an entry in `graph.json`), **delete**, **rename**, **set output**, **wire an input**, **unwire**.

**Wiring is a closed-set selector over existing pass names**, never a free-text field. This is
KodeLife's model, it makes SHADERed's positional footgun impossible, and it means an input can never
name a pass that does not exist.

**Rename rewrites every edge that references the pass, renames the file, and re-points any open
editor tab** (tab identity is the PATH). Without that, D3's silent-black rule — which is what keeps a
half-built graph usable — would make a rename fail invisibly. The graceful-degradation rule and the
silent-breakage risk are the same rule, so rename must be transactional.

**D16. Media and textures are namespaced by pass.** `UINode.save` writes `media/<uniform>.*` and
`textures/<uniform>.bin` flat per directory, and its orphan sweep deletes any asset no surviving
uniform references. With D4 making uniforms pass-scoped, two passes both binding `u_tex` would
overwrite each other's file and the sweep would delete the survivor's asset — **silent data loss**,
and a direct consequence of D4 that the first draft did not price. Assets live under
`media/<pass>/<uniform>.*` and `textures/<pass>/<uniform>.bin`, and the sweep scopes to one pass.

## Open questions for the user

None blocking. Two judgement calls made during review that are worth overruling if you disagree:

- **D12 flipped to one script per PASS.** The first draft said per document; the engine's flat
  name-resolution makes that unaddressable under D4. If you would rather have one script driving
  several passes, the mechanism is pass-qualified keys (`{"blur.u_radius": 0.5}`) — say so and I will
  spec that instead.
- **D10b adds a view selection** the first draft omitted. If you would rather debug by re-pointing
  the output, it is fewer moving parts — but it makes a read into a document write.

## The on-disk shape

```
projects/<project>/documents/<uuid>/
    graph.json              app-written; the graph, the targets, the output choice
    passes/<name>.glsl      one ordinary fragment shader per pass, its own main()
    scripts/<pass>.py       optional, one per PASS (D12), bound by existence
    media/<pass>/<uniform>.*        a sampler's bound media
    textures/<pass>/<uniform>.bin   a raw texture bound to a sampler
```

**Media and textures are namespaced BY PASS** (D16). Today they are flat per node, keyed by uniform
name; with N passes in one directory, two passes reusing a uniform name would overwrite each other
and the orphan sweep would delete the survivor's file.

`graph.json`:

```json
{
  "version": 1,
  "output": "composite",
  "passes": {
    "scene":     { "inputs": {},
                   "target": {"scale": 1.0, "dtype": "f2", "filter_linear": true,
                              "wrap": false, "persist": false} },
    "bright":    { "inputs": {"u_src": "scene"},
                   "target": {"scale": 0.5, "dtype": "f2", "filter_linear": true,
                              "wrap": false, "persist": false} },
    "trail":     { "inputs": {"u_src": "scene", "u_prev": "trail"},
                   "target": {"scale": 1.0, "dtype": "f2", "filter_linear": true,
                              "wrap": false, "persist": true} },
    "composite": { "inputs": {"u_lit": "scene", "u_glow": "bright", "u_trail": "trail"},
                   "target": {"scale": 1.0, "dtype": "f1", "filter_linear": true,
                              "wrap": false, "persist": false} }
  },
  "layout": {"scene": {"x": 0, "y": 0}, "bright": {"x": 200, "y": -60}}
}
```

Reading it: `inputs` maps THIS pass's sampler uniform name to the pass that fills it, so
`"u_prev": "trail"` inside `trail` is feedback. `output` names the pass the preview and export show.
`layout` is cosmetic and separate, so losing it never costs the effect — freska's split, which it
got right. A pass file with no entry in `passes` gets defaults; an entry naming a file that does not
exist is reported and skipped.

**Everything in `graph.json` is derived, app-written state**, exactly as `node.json` is today. It is
never hand-authored, so per-key salvage (`model_salvage.py`) applies: a malformed pass entry costs
that pass, never the document.

## Implementation order

Each stage leaves the tree green and is verifiable on its own.

1. **`pass_graph.py`** — DONE. The graph model plus topological order, cycle detection and feedback
   marking. GL-free, pure data, fully unit-testable. **The memoization invariant is asserted on every
   plan the test module builds** (D5) AND inside `evaluation_order`, the function the renderer calls;
   this bug has shipped twice in this repo and tests caught it neither time. Landed with a mutation
   battery: each guarantee's falsifier is a mutant the suite kills.
2. **`Pass`** — DONE. Split out of `Node`: source, program, target, uniforms, compile, draw, in
   `core.py`; `Node` moved to `document.py` with the document concerns and holds one `Pass` as
   `render_pass`. Verified by rendering a bare pass headlessly (`tests/test_pass_render.py`), and
   verification check 1 passes byte-exactly against the pre-split engine. An unconfigured pass keeps
   the 8-bit canvas: `TargetConfig`'s `f2` (D9) is what a pass IN A GRAPH gets, and applying it
   universally would reformat every document's canvas that the export path reads as 8-bit.
3. **`Document`** — DONE. `Node` in `document.py` owns `passes` + the `PassGraph`, draws them in
   dependency order, resolves `render_pass` to the output, and owns feedback. Engine checks 1-6 and
   8-9 pass. Two decisions the implementation forced: an unresolved input binds an explicit BLACK
   texture rather than being left unbound (leaving it unbound falls through to the sampler's default
   photo, so a mis-wire showed an image instead of nothing); and `begin_frame(frame)` takes the frame
   NUMBER, so a second call for the same frame is a no-op. Call-count alone was unfalsifiable —
   because the preview render passes its own canvas, an extra swap per render cancelled out and a
   wrong call site passed every gate.
4. **Persistence** — DONE. `graph.json` load/save with per-key salvage, `passes/<name>.frag.glsl`,
   and `media/<pass>/` + `textures/<pass>/`. Checks 10-12 pass. Two bugs the round-trip caught, both
   silent: `drop_unknown` walks a nested model's FIELD names, so running it over `passes` read every
   pass NAME as an unknown key and pruned the whole graph to empty — the keyed-dict fields are now
   held out of the whole-model pass and salvaged per entry; and `UINode.save` skipped a pass with no
   program, which is every pass off the OUTPUT's path, so their tuned uniforms were written away as
   `{}` — save now compiles a pass that has not drawn, and keeps what is on disk if it cannot.
5. **The rename** — DONE. `node` -> `document` across the package, on-disk paths, and the tests, in
   its own commit. `Document` / `document_id` / `ui_documents` / `documents/<id>/document.json`;
   `tabs/document.py`, `widgets/document_grid.py`, `copilot/tools/document_ops.py`; the copilot's
   `rename_document` / `duplicate_document` / `import_document`. Namesakes that stayed: `EngineNode`
   (a scripting protocol whose subject is now a `Pass`), the lib picker's filesystem tree nodes, and
   imgui's own `tree_node_ex`. `NodeTab` did NOT stay — its `NODE` member is the document tab, so it
   is domain, not a namesake, and became `DocumentTab.DOCUMENT`.
6. **Editor and hot reload** — DONE. `watch.py` polls EVERY pass, not just the output's, and
   identifies a root by PATH rather than by index 0. Check 7 passes, counted. Two things the spec
   expected that turned out not to be needed: a `pass` tab KIND (a pass tab is a shader tab keyed
   by its own path, which is what `EditorTab` already was — `ensure_shader_tab` just takes an
   optional pass name), and work to make recompilation per-pass (the stage-2 split gave it, since
   each `Pass` owns its program). What DID need work was the editor-sync callback, which keyed on
   `document_id` and so could only ever address one file per document; it keys on the PATH now,
   which is what an editor session was always keyed by.
7. **The panel** — DONE (headlessly; the visual pass is owed to the maintainer). `widgets/
   pass_list.py` in the Document tab: a row per pass, a right-click menu, an inline add/rename, a
   closed-set input combo per sampler, and the target controls. D15's six verbs live on
   `ProjectSession` (`add_pass` / `delete_pass` / `rename_pass` / `set_output_pass` /
   `wire_pass_input` + `set_pass_target`), each mutating the live document AND saving, so
   `passes/` and `graph.json` cannot disagree with the screen; `tests/test_pass_verbs.py` drives
   them headlessly and reloads from disk to prove it. Rename is transactional as D15 demands, and
   the tab-repointing half rides a new `on_pass_renamed` hook. Smoke drives the panel's non-default
   paths (the settings modal, both inline inputs) because none of them draw by default and
   none can be screenshotted here.
8. **The copilot** — DONE, no new tools. The address kind is a SUFFIX (`<id>#<pass>`) rather than
   a prefix, so a bare id stays valid and means the output pass — every tool that predates the
   graph keeps its meaning. `WorkingSetView.passes` renders a multi-pass document as sub-sections
   the way SCRIPT already did, and a pass address in the working set collapses to its DOCUMENT so
   an 8-pass document cannot evict its own passes. Three things D11 did not name, each found by a
   test rather than by reading: `has_errors` on the project map read only the OUTPUT pass, so a
   broken pass nothing draws reported the document clean; the unstick's `_last_clean` /
   `_broken_streak` were keyed by document id, which now aliases across a document's passes, and
   are keyed by working-set address; and `_format_uniforms` / `_sampler_binding` took a Document
   when uniforms are pass-scoped (D4), so they take a Pass. Check 16 still needs `/dogfood`.
9. **Content** — DONE. The five existing examples and the dev sandbox were re-authored in the new
   shape at stage 4 (by hand, no migration), and stage 9 added "Bloom Chain": a five-pass document
   exercising a chain, a fan-in, a half-scale target, feedback, and an output that is not the last
   pass authored. Authoring it caught `TargetConfig.scale` being DEFINED and read by nothing —
   a knob the panel could set and the file could persist while the renderer ignored it. The
   document applies it (it owns the canvas size), the output keeps full size, and two tests pin
   both halves.

Stages 1-4 are the engine and can be built and tested with no UI at all. That is deliberate: 064's
lesson is that the surface should be judged against a working engine, not designed alongside one.

## What happens to today's data

**Nothing is migrated.** Maintainer decision: build from scratch.

- The five shipped examples in `shaderbox/resources/document_examples/` are re-authored by hand as
  single-pass documents. (A sixth, "Render Steps", shipped 064's syntax and was deleted with the
  rest of the residue.) They are the app's first-run content and the examples browser's contents,
  so they cannot simply be dropped.
- `projects/dev/` is the maintainer's sandbox; its nodes are re-authored or discarded at the
  maintainer's discretion, and the tree is committed in the same wave per the sandbox rule.
- **An empty project must start.** Verify this early — `App._init` seeds a starter example on first
  run, and that path must work with the new format before anything else can be tested by hand.

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
`project_session.py`, `watch.py` (its "index 0 is the root" branch early-RETURNS, so it is
wrong-by-construction with N roots, not merely in need of generalising), `paths.py`,
`copilot/address.py` + `backend.py`'s working set + `node_tree`, `tabs/node.py` -> the document
panel, `editor_types.py` (a `pass` tab kind — string-compared at ~11 sites with no exhaustiveness
check, so the new kind needs a guard that fails loudly), `app.py` (`current_node_id` is reached at
~89 sites and every one must decide document-or-pass), `copilot/checkpoint.py` + `revert.py` (the
snapshot path writes one shader file and needs the pass loop), `scripting/engine.py` (keys on a
pair per D12 — NOT untouched, as the first draft claimed).

Untouched, verified design-independent: the exporters (they consume a `RenderedArtifact` — bytes,
GL-free, and the one-node assumption lives at the `render_job.py` CALL SITE), `shader_lib/` and the
resolver, `shader_errors.py`, `model_salvage.py`, `ui_primitives.py`/`theme.py`.

**Scale, honestly:** 871 case-insensitive `node` mentions across 56 package files, with
`copilot/backend.py` (442), `project_session.py` (259) and `app.py` (179) heaviest. The rename
itself is mechanical; the expensive part is the ~10 sites where "node" is load-bearing SEMANTICS —
current-node, checkpoint unit, script key, working-set member, export unit — each of which must
decide document-or-pass.

## Review notes — the maintainer's hands-on pass

The engine was built and verified headlessly; the UI was too, which is the wrong way round and
this section is the consequence. Every entry below came from the maintainer OPENING THE APP, and
none of them could have been caught here (no window manager on the dev box — `/imgui-ui` §0).
Treat it as an in-progress list, not a closed one: the review rounds recorded below settled the
strip's shape (commit `c464d01` counts eight hands-on rounds), and `### Still unseen` names what
has had no human frames yet.

### Where the UI lives

- **`shaderbox/widgets/pass_list.py`** — the strip itself. A horizontal strip of
  `preview_cell` tiles (one per pass, showing that pass's own live target scaled down by imgui —
  no second render), the rename input and `add pass`. Right-click a tile for settings / rename /
  set-as-output / delete.
- **`shaderbox/popups/pass_settings.py`** — the `Reads` and `Draws into` controls, as a modal in
  the `PopupState` mutex. Opens from the selected tile's sliders-glyph overlay, the context menu,
  and automatically on `add pass` — the one moment the choices are actually made.
- **`shaderbox/tabs/document.py`** calls it, above the `Entry points` section that still holds the
  Script row.
- **`shaderbox/tabs/code.py::tab_label`** names a pass in its editor tab, and
  `_pass_for_tab` resolves the tab's own pass for the header and error strip.
- The six verbs it calls are `ProjectSession.add_pass` / `delete_pass` / `rename_pass` /
  `set_output_pass` / `wire_pass_input` / `set_pass_target` — all headless, all tested in
  `tests/test_pass_verbs.py`.

### What came back, and what it cost

1. **"`open` opens the same shader no matter which pass I click."** THREE consumers resolved "the
   current document's shader" as `document.render_pass` — the OUTPUT — rather than the active
   tab's own path: `tabs/code.py`'s session lookup for a tab with no session yet,
   `App.get_current_session`, and `App.flush_current_editor`'s save branch. Each was correct for
   the project's whole history, because a document had exactly one shader file until stage 2.
   Fixed in `fa3d9b0`; `tests/test_pass_editor_wiring.py` pins each consumer to the tab's PATH.
2. **Tabs were indistinguishable.** Every pass of a document read `<document> (shader)`. Now
   `<document> (scene)`, taken from the tab's path so a rename carries it (`2e5f713`).
3. **"What the fuck are all these selectors and sliders, what the fuck is `f1`/`f2`."** Correct,
   and the sharpest lesson of the feature: I had put the DATA MODEL on screen. `f1`/`f2`/`f4` are
   moderngl dtype strings and scale/smooth/tile are `TargetConfig` field names, dumped as four
   unlabelled widgets under every pass. Also too tall — a column of full-width rows. Now a
   thumbnail strip (`4640726`) whose controls name what they change about the PICTURE, with a `?`
   explaining each (`efb5d26`). A test pins the format labels against `pass_graph.DTYPES` so a
   raw dtype string cannot come back.

**Round two (after the strip rework):**

4. **"The `Reads`/`Draws into` UI looks like shit — shit ton of vertical space for settings used
   once at pass setup."** Correct again, and the direction was theirs: the block became a MODAL
   (`popups/pass_settings.py`), opened from a sliders-glyph overlay on the selected tile, the
   tile's context menu, and automatically when a pass is created. The strip now spends its
   vertical budget on thumbnails only. This also dissolved their first-launch finding — the
   selected pass showed no `Reads`/`Draws into` until an explicit tile click, because
   `pass_expanded` started empty while the selection border tracked the active tab; there is no
   expanded state to be empty any more.
5. **Tiles too small.** Their number, not a guess: 1.5-2x. `SIZE.PASS_THUMB` 64 -> 112.
6. **The scrollbar landed on the uniforms sub-pane, not the Document tab.** The `ui_uniforms`
   child had a fixed size (the remaining space) and scrolled internally, so the tab itself never
   scrolled. `ChildFlags_.auto_resize_y` makes the child grow to its content and the whole
   `document_settings` panel scroll as one surface.
7. **The size slider read 0%-1%.** `%.0f` was formatting the RAW 0.05-1.0 scale. The slider now
   runs over 5-100 (the number a person reads) and the model keeps the 0-1 scale.

**Round three (on the modal):**

8. **"Put name editing in the settings."** The inline rename input under the strip and the
   context menu's Rename are gone; the modal's first row is the name (Enter commits, and
   `_on_pass_renamed` re-points the modal's own target so it keeps drawing through its own
   rename).
9. **Tooltip and help noise.** The tile tooltip's "click to open; right-click for actions" line
   is deleted, and the Reads section carries ONE `?` on its caption (how rows appear, how to add
   one) instead of the same sentence under every combo.
10. **The size slider's text overflowed it.** Only `%.0f%%` lives inside the slider now; the
    derived resolution moved into the row label — `size (960, 540)` — and `_ROW_LABEL_W` grew to
    fit it.
11. **"Clicking another pass preview should fully switch to it."** The main viewer now shows a
    per-document **viewed pass** (`UIDocumentState.viewed_pass`, persisted; "" = follow the
    output). A tile click opens the pass AND views it; "Set as output" resets viewed to "" so
    the viewer tracks the output — which is exactly the rare "look at A while tweaking B" tool,
    because the uniforms panel does NOT follow the viewer: it follows `App.panel_pass` (the
    active shader tab's pass, then viewed, then output). `Document.view_pass` (runtime, synced
    per frame in `ui.py`) extends the render plan so a viewed pass the output does not need
    still draws (pinned in `test_document_graph.py`); rename-follow is pinned in
    `test_pass_verbs.py`. The current document also keeps rendering behind the pass-settings
    modal — watching a wiring change land is the modal's whole point.
12. **The strip orders topologically, output last** (`_strip_order`, pinned in
    `test_pass_verbs.py`): producers left of consumers via `plan_passes`, unplannable passes
    appended by name, and the output moved to the end so the strip reads left-to-right as "how
    the final picture is built". It was alphabetical.
13. **The modal speaks the Settings dialect**: `separator_text` sections (Pass / Reads / Draws
    into) and the shared `help_marker` "(?)" instead of a bespoke bare `?`; the Reads section's
    help hovers on its own separator title.
14. **"Sometimes a tile click selected the code field, sometimes not."** The tab's TextEditor
    auto-grabs keyboard focus on its FIRST render (/imgui-ui section 8), so the panel kept focus
    only when the summoned tab had rendered before. Fixed at the shared root:
    `App._focus_or_add_tab` yields the editor back (`_yield_editor_to_region`) whenever a
    non-editor region owns focus — every summoner (tile click, document switch, OPEN_SHADER /
    OPEN_SCRIPT, startup restore) now keeps the summoning surface focused; entering the editor
    is its own gesture. Skipped while a popup is open (the region latch would force-close the
    modal); `select_document`'s local GRID guard became redundant and was removed. Pinned in
    `test_pass_editor_wiring.py`.

**Round five (the strip's meaning settles):**

15. **Click = set as output; ONE highlight; the viewed-pass mechanism is gone.** Round three's
    `viewed_pass` split the viewer from the output and the maintainer rejected it in use: a tile
    click now sets the graph OUTPUT (and opens the editor tab), the viewer simply follows the
    output, and the accent border has one meaning — the picked/output pass (error red still
    overrides; an open tab is shown by the tab bar, not a second border colour). The
    look-at-A-while-tweaking-B case rides the TAB BAR: tabs never touch the output and the
    uniforms panel follows the active tab (`App.panel_pass`). Removed with it: the
    `UIDocumentState.viewed_pass` field, `Document.view_pass` + its render-plan extension and
    test, and the context menu's now-redundant "Set as output".
16. **The strip order is output-independent.** Setting a pass as output was re-shuffling the
    tiles (the round-four "output last" move). `_strip_order` is now pure `plan_passes` topo
    order — deterministic and blind to the output — so picking a different output never moves a
    tile. Pinned in `test_pass_verbs.py`.
17. **Off-plan passes freeze visibly.** Passes the current output does not need never render
    (that was always the engine's rule); their tiles now say so: the image washes toward
    mid-grey (`COLOR.STALE_WASH` over the image — 0.4*pixel + 0.6*grey, desaturation, after a
    multiplicative dark tint "worked strange") and the footer dims (`preview_cell(stale=...)`).
    The corner tick and the tooltip line the first cut used were both rejected as noise; the
    treatment is shared with the document grid's stale thumbnails.

### Still unseen, and the first thing to ask about

Nothing below has been looked at by a human. Ask before assuming any of it is fine:

- **The modal, whole.** `popups/pass_settings.py` was built after the maintainer's last look:
  its size (`SIZE.PASS_SETTINGS_W/H`, 440x400), the sliders-glyph overlay button, the "reads
  nothing" caption for a sampler-less pass, the name row, and open-on-create have had zero human
  frames.
- **The click-sets-output switch, live** — including whether rewriting `graph.json` on every
  tile click feels right in use, and the stale dim on off-plan tiles.
- **The 112px tiles at real panel widths** — whether the strip wraps sensibly and whether a long
  pass name truncates badly in a tile footer. `SIZE.PASS_THUMB` has exactly two consumers, both
  in `pass_list.py`, so a correction is one token edit.
- **Spec checks 13-15**, which need a display: an error in pass 2 landing in the strip with pass
  2's file and line and click-to-jump working; a rename re-pointing an open tab (the engine half
  is tested, the TAB half only smoke-verified); the six shipped examples loading with the browser
  populated.
- **Check 16**, which needs `/dogfood` and real API cost: the copilot authoring a two-pass
  document and seeing per-pass errors.

### What headless verification CAN and CANNOT catch here

`make smoke` drives the panel's real draw paths through 200 frames — including the five-tile strip
(it switches to Bloom Chain mid-run, since the single-pass fixture never draws more than one tile),
the armed delete wash, the settings modal and both inline inputs. That catches an unbalanced imgui
ID stack, a SetCursorPos assert, a None-deref. Measured, it does NOT catch: a leading `same_line`
on the first tile, an unbalanced `indent`, or anything about size, spacing, wrapping or wording.
Those are exactly the class that has to go to the maintainer.

## Verification

Each check fails for exactly one reason, and each names its falsifier.

**Engine (headless, no UI needed):**

1. **A one-pass document renders the same pixels as today's node for the same shader at the same
   `u_time`.** Falsifier: any difference. Note the target format changes to `f2` (D9), so the
   comparison is on the tonemapped output, not the raw target.
2. **A two-pass chain shows B-reads-A.** Falsifier: the output equals A's own image.
3. **A diamond orders topologically and the shared ancestor draws ONCE.** Asserted on EVERY plan the
   test module builds, not in one test — this bug has shipped twice here and reads as slow, not
   wrong.
4. **A feedback pass accumulates across frames**, and the swap happens once per frame even though
   the live loop renders twice. Falsifier: the value advances at 2x, or not at all.
5. **A non-self cycle reports an error per pass and does not hang.**
6. **An unfilled input reads black and the document still renders.** Falsifier: an exception, or the
   default image appearing.
7. **Editing one pass recompiles ONLY that pass.** Falsifier: a compile counter shows more than one.
   The observable is a count, so the test must count rather than eyeball.
8. **Two exports of a feedback document are byte-identical**, with live frames rendered in between.
9. **Export renders the OUTPUT pass.** Falsifier: build a document whose output is not the
   last-authored pass; picking the wrong one gives a visibly different image.

**Persistence:**

10. **`graph.json` round-trips** the graph, every target config, and every pass's uniforms.
11. **A malformed pass file costs that pass, not the document** — and a malformed `graph.json` entry
    costs that entry, not the document.
12. **Two passes binding a sampler of the same name keep separate media**, and the orphan sweep
    deletes neither (D16).

**UI (manual, needs a display):**

13. **An error in pass 2 lands in the strip with pass 2's file and line**, and click-to-jump works.
14. **Rename rewrites every edge**, renames the file, and re-points an open tab. Falsifier: an edge
    still names the old pass, and per D3 it silently reads black.
15. **The six shipped examples still load and render**, and the examples browser is populated. This
    is the highest-probability breakage of the whole wave.

**Copilot (needs the dogfood harness and real API cost — not a green-tree check):**

16. **The copilot authors a two-pass document with no new tools**, and can see per-pass errors in
    its working set. Runs under `/dogfood`, not `make test`.

## Fixtures that do not exist yet

Checks 2-9 need a hand-authored multi-pass document checked into `tests/`. The dev sandbox has no
candidate to promote — `e0bdb82` removed the 064 residue an earlier draft named here — so the
fixture is authored from scratch alongside stage 3.
