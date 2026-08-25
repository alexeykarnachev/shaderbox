# Proposal C — "The Chain Board" (bias: GRAPH-FIRST)

> Produced by a design agent anchored to `00_scenario.md` R1-R10, the freska source fetched
> first-hand from GitHub, the installed imgui-bundle package, and the real UI code. Verbatim agent
> deliverable, saved for the judging round.
>
> **One correction applied by the maintainer session:** the agent's closing note claims
> `00_scenario.md`'s owed-fix item 3 is "half-stale" because `core.py` already passes `dtype`. It
> does not predate the agent — that line is from commit `a4758cd`, landed in THIS session an hour
> before the agent read the tree. Both sides (loader + save) are now fixed and covered by
> `tests/test_raw_texture_round_trip.py`. The agent read a patched tree and mistook the patch for
> the original state. Its underlying point (R6-Hold promotes that latent defect to a blocker)
> stands.

## 1. The core idea

Every node gains a second view of itself — a **Chain Board**, a zoomable canvas where the node's
draws appear as boxes wired together, containing by default exactly one box holding the shader that
is there today. Each box is one draw: a fragment shader file, a live thumbnail, an output
resolution, and its own auto-generated uniform sliders drawn on the box face. Add a step by
right-clicking empty canvas; wire it by dragging from one box's output dot to another's named input
dot — and **the wire's name IS the `sampler2D` uniform name in the receiving shader**, so the picture
on the board and the text in the editor are the same fact seen twice.

**What makes it native rather than a bolted-on second app:** the board is not a new document type, it
is a second layout of the panel that already exists. Today's node panel shows one resolution combo,
one entry-point row, and a list of uniform sliders (`tabs/node.py:96-172`). The board shows N of
those, positioned, with wires. A one-step node's board is visually a single card carrying the same
widgets in the same order.

## 2. Where it lives: a fourth node tab

One row added to `_NODE_TABS` (`ui.py:594-598`), one member added to `NodeTab`
(`ui_regions.py:18-22`), one new `tabs/chain.py` with a `draw(app)` matching the existing signature.
That is the whole seam — `Ctrl+digit` tab jumping picks it up for free because it iterates
`_NODE_TABS`.

**Why a tab, not a mode toggle on the preview:** the big image is the node's RESULT and stays the
result. A mode toggle there would make the board compete with the thing it exists to produce — you'd
lose sight of the output exactly when wiring it. The tab bar is already where ShaderBox says
"different facet of the same node" (Node = parameters, Render = export, Share = publish);
"Chain = structure" is the same kind of statement.

**Why not a fourth `ActiveRegion`:** `ActiveRegion` has three members walked by the cycle-region
command. A fourth changes keyboard muscle memory for every user for a feature most nodes never use.

## 3. The board

```
 [Chain] tab body
| [+ Step]  [Fit]  [Watch: final v]        (1 step . 512x512 . 0.3 ms)   |  toolbar
|                                                                        |
|      /- image ---------------------\                                   |
|      |  +----------------+         |                                   |
|      |  | ## thumbnail ##|  (o) out|                                   |
|      |  +----------------+         |                                   |
|      |  512x512 . rgba8            |                                   |
|      | [drag] u_scale  --O---      |   <- the SAME row draw_ui_uniform  |
|      | [color] u_tint  ####        |      produces today                |
|      \----------------------------/                                    |
```

**That is a default node.** One box named `image` (Shadertoy's word for the final step, which the
survey shows every file-based tool converging on), no wires, no inputs. **A user who never opens the
Chain tab never learns a new concept.**

Two steps:

```
 /- blur_h ---------------\        /- image -----------------\
 | +----------+           |        | +----------+            |
 | | #thumb#  |    (o) out|--------|-(*) u_src  |    (o) out |
 | +----------+           |        | +----------+            |
 | 256x256 . rgba16f  1/2 |        | 512x512 . rgba8          |
 | [drag] u_sigma --O--   |        | [drag] u_gain ---O--     |
 \------------------------/        \-------------------------/
```

The wire lands on a pin **labelled `u_src`**, because `u_src` is a `sampler2D` in
`image.frag.glsl`. **Input pins are not authored in the board — they are introspected**, exactly as
uniform rows are today: `get_active_uniforms()` (`core.py:244-252`) enumerates the live program and
`node.py:67-76` already filters by `gl_type == GL_SAMPLER_2D`. A box's input pins ARE that filtered
list. Delete the `uniform sampler2D u_src;` line and the pin (and its wire) disappear on hot-reload.

**This is the single most important design commitment.** The board is a *view of introspected program
state*, not a document that can disagree with the code. ShaderBox already builds the entire uniform
panel this way (`node.py:124-136` creates rows lazily from live uniforms; `ui_models.py:303-315`
prunes rows the program no longer has). The board inherits the identical lifecycle.

**Files:** `steps/<name>.frag.glsl` beside the existing `shader.frag.glsl`; `node.json` gains a
`chain` block. `node.json` staying app-written is preserved — the survey's warning ("hand-authored
pass declarations would turn it into a file the user must edit") is satisfied **because the board is
the editor for it.** The user never hand-edits chain JSON; they drag boxes. That is the graph-first
answer to that specific concern, and arguably a better one than any text syntax gives, since a text
form is hand-authored by definition.

**Layout vs semantics stored separately, following freska exactly.** `freska.json` (fetched) holds
ONLY positions/view/selection, never a pin or link. Losing the layout loses arrangement, never the
effect.

## 4. Requirement coverage

| # | UI element | User action |
|---|---|---|
| R1 | board canvas; `[+ Step]` + right-click-empty-canvas | right-click -> "Add step" -> box appears at cursor with a starter shader; rename inline |
| R2 | size chip on the box face: `512x512 . rgba8` | click -> `[Same as node / 1/2 / 1/4 / 1/8 / custom]`. Default for a box fed by exactly one wire is **"Same as input"** — freska's own rule (`render_texture = LoadRenderTexture(frame.width, frame.height)`) AND the survey's convergence #1 |
| R3 | multiple input pins, one per `sampler2D` | declare two samplers -> the box grows two labelled pins on hot-reload -> drag a wire into each |
| R4 | one output pin carries many wires; the canvas is 2-D so **a branch is drawn as a branch** | drag a second wire from `lit` out to `bright_extract` in. The bloom chain visibly hangs off to the side — precisely the scenario's phrasing |
| R5 | a wire from a box's out pin **back to its own in pin**, dashed, with a `loop` badge | **freska explicitly REJECTS same-node links** (`if (start_pin->node_id == end_pin->node_id) return false;`); ShaderBox must INVERT that rule for texture pins, and that inversion is the whole of R5's UI |
| R6 | the self-wire's context menu: `Persist: [Every frame / Hold / Warm N]` + a `[Reset]` button + a `state dot` in the header | §6 |
| R7 | format half of the size chip | **an amber `8-bit` marker** appears on a box wired into by a float box while itself 8-bit — the measured-fatal case made visible at the exact place it happens |
| R8 | **the WIRE's** context menu: `Filter`, `Wrap` | **siting it on the wire, not the box, is deliberate:** filtering is a property of THE READ, not of the stored image, and a box feeding two consumers can legitimately be read linear by one and nearest by the other |
| R9 | (a) every box carries a live thumbnail, always; (b) `Watch: [final v]` + click any thumbnail routes it to the big preview | **the requirement this bias serves best** — no other design gives all eight cascade levels visible simultaneously |
| R10 | `draw_ui_uniform` rows **on the box face** + the Node tab retargeting to the focused step | ownership is **positional and unambiguous** — the slider is physically on the box |

## 5. The freska lessons (source read first-hand)

**Lifted:**
1. **`PinKind`'s three-way split** — a parameter and an edge are different mechanisms wearing the
   word "input".
2. **Binding by pin NAME** (`GetShaderLocation(shader, pin.name)`) — the same introspection-driven
   binding `core.py:359-404` does. ShaderBox's version is stronger because the pin list is DERIVED
   FROM the program rather than declared beside it.
3. **Size derived from the input.**
4. **A step with no ready input is a silent no-op, not an error** — this matters because a graph is
   ALWAYS half-built during authoring; treating that as an error would make the board hostile for the
   entire time you use it.
5. **Layout-only persistence.**
6. **The whole interaction skeleton from `app.cpp`** — including the `Suspend()`/`Resume()` sandwich
   around every popup, "a non-obvious requirement freska demonstrates and I would otherwise have got
   wrong".
7. **The output-pin thumbnail** — ShaderBox has a better version already in `preview_cell`. "This is
   the concrete reason the board can look native on day one."

**Defect 1 — broken evaluation order.** `Graph::update()` iterates `unordered_map` order (arbitrary,
unstable across runs) and propagates links AFTER all updates, so a value crosses one edge per frame:
an N-step chain lags N-1 frames, nondeterministically. **How the UI makes the fix checkable rather
than promised:** each box header carries its evaluation index (`1 blur_h`, `2 blur_v`, `3 image`), so
a wrong sort is VISIBLY wrong — *freska's bug was invisible because nothing displayed the order*. A
non-self cycle is refused at drag time with red-wire feedback; the `Watch:` combo lists steps in
evaluation order, asserting it in a second place. A subtler trap the board must not hide: **a
self-loop contributes NO ordering constraint** and must be excluded from the cycle check while
included in ordering as "reads the previous frame".

**Defect 2 — inputs by hard index** (`pins[0]._texture`, with his own unactioned TODO).
**Avoided structurally, not by discipline: there is no pin table.** Pins are computed each frame from
the introspected sampler list; a wire is `(from_step, to_step, to_uniform_name)` — three strings,
nothing positional anywhere. User-visible consequence: **rename a sampler and its wire drops,
visibly, on hot-reload**, with a notification `Wire dropped: image.u_src no longer exists`. Strictly
better than SHADERed, which the survey singles out for the opposite failure ("rename the RT and
nothing breaks, reorder the slots and everything breaks") — **silent breakage-by-reorder is worse
than loud breakage-by-rename.**

**Is `PinKind::MANUAL` right for ShaderBox?** *The distinction is right. The mechanism is wrong, and
adopting it literally would be a regression.* Freska's `MANUAL` pin is hand-declared, hand-ranged,
hand-typed C++ (`Pin::create_float(MANUAL, "exposure", -1.0, -1.0, 10.0)`) — a declaration beside the
shader, in a different language, that can silently disagree with it: the exact KodeLife drift the
survey names. **ShaderBox already HAS `MANUAL`, and its version is derived rather than declared**
(`ui_models.py:54`, `71`, `99`). The board adopts freska's TAXONOMY and rejects its CONSTRUCTION.
One genuine gap freska's version exposes: its pins carry min/max, so its sliders have sensible
ranges; ShaderBox's introspection cannot know a range, hence `drag_float` fallback. The board doesn't
worsen this but doesn't fix it — and range declarations are refused precisely because they'd
reintroduce the drift.

## 6. R6 — three visible values on the feedback wire

| Value | Meaning | For |
|---|---|---|
| **Every frame** (default) | reads last frame; contents not saved; reload starts black | trails, most feedback |
| **Hold** | reads last frame; **contents saved with the node and restored on load** | the smoke sim |
| **Warm N** | reads last frame; on load, runs N frames headless before the first visible frame | sims that look wrong cold but shouldn't bloat the node dir |

"Everything follows from which value is set, so the answer to what happens on save/revert/export is
always VISIBLE ON THE WIRE, not buried."

- **Save/reload.** `Hold` writes the texture into the node dir. **R6 is what promotes the latent
  `textures/` mkdir defect to a blocker** — it is the first thing to make that code path live. The
  board makes the cost legible: the header reads `hold . 1.0 MB`, so nobody is surprised by a fat
  node directory.
- **Copilot revert.** Because `Hold` state is written by `UINode.save()`, it is INSIDE the
  persistence boundary — a revert restores the smoke along with everything else. "That is not
  incidental; it is the reason `Hold` is defined as 'saved with the node' rather than 'kept in GL
  memory'." `Every frame` state is deliberately outside the boundary: reconstructible by waiting, so
  restoring it buys nothing. One UI consequence named rather than hidden: after a revert on a `Hold`
  box the sim jumps backward in time, so the notification says `Reverted . smoke restored to
  checkpoint state` — "a silent time-jump is the confusing version".
- **Export.** `render_media` already enters `export_isolation` so a stateful script ticks from a
  FRESH instance, "structural, so no export caller can forget to isolate". **R6 state must follow the
  same rule for the same reason, and the fact that ShaderBox already made this exact decision for
  scripts is the strongest available argument for what to do.** So export starts cold, and the Render
  tab gains `Warm-up frames: [0]`, shown only when the chain has a feedback wire. For a `Hold` box, a
  `[x] Start from saved state` checkbox seeds the export from the persisted texture — "capture it as
  it looks right now" without capturing whatever transient the live app was mid-way through.

## 7. The naming collision — worse than "a bit awkward"

"Node" in ShaderBox is not casual: `UINode`, `Node`, `node.json`, `nodes/<uuid>/`, `NodeTab`,
`node_grid.py`, the copilot tools (`create_node`, `switch_node`, `import_node`), and the prompt says
it repeatedly and load-bearingly ("the user authors `.frag.glsl` 'nodes'", "In replies, call nodes by
NAME", "a bare/demonstrative reference = the CURRENT node"). **The copilot damage is the serious
part:** an LLM reading a prompt where "node" means two nested things will conflate them, and the
demonstrative-reference rule becomes genuinely ambiguous.

**The proposal: the word "node" never enters the graph vocabulary.**

| Concept | Word | Never say |
|---|---|---|
| the existing unit | **node** — unchanged, every existing usage stands | — |
| the graph inside a node | **chain** | graph, patch, network, DAG |
| one box = one draw | **step** | node, pass, stage |
| a connection | **wire** | link, edge, connection |
| a texture input socket | **input** (named by its uniform) | pin, port, slot, channel |
| the canvas | **the chain board** | the graph, the patch |
| a self-referencing wire | **feedback wire** | loop, recursion, ping-pong |

*"A node contains a chain of steps wired together on the board."* Nothing nests a word inside itself.
**The practical argument for these words: the collision is avoided without renaming a single existing
identifier.** Help panel needs one amendment ("Every node is one fragment shader — or, when you need
several draws, a chain of them"). The copilot block appears in the working set ONLY for nodes that
actually have more than one step. One honest residual: users who've seen TouchDesigner will SAY
"node" meaning "step", so the prompt needs one disambiguating line.

## 8. Cost, and what it makes worse

**The imgui-bundle binding is present, complete, and importable — verified, not assumed.**
`imgui_node_editor.pyi` (1244 lines of stubs), `imgui_node_editor_ctx.py` (Pythonic `with` wrappers),
a runtime import executed via `uv run python` returning **126 public symbols**, and **a complete
247-line working Python demo shipped in the package** (`demos_node_editor/demo_node_editor_basic.py`)
whose settings JSON is shape-identical to `freska.json`. Every function freska's `app.cpp` calls
exists under a snake_case name, plus extras this design wants (`get_ordered_node_ids`,
`screen_to_canvas`, `flow`, `show_pin_context_menu`, theming enums). `ed.Config` exposes
`force_window_content_width_to_node_width` — the fix for sliders inside nodes, documented verbatim in
the stub.

> The agent's own framing: *"the answer to 'what happens if imgui-bundle has no node-editor binding'
> is: it does have one, so that contingency does not arise. If it had been missing, the honest answer
> would have been that this proposal is not viable — hand-rolling a pan/zoom canvas with bezier
> wires, hit-testing, box dragging, marquee selection and per-node clipping is a multi-week project
> on its own, and I would have withdrawn the design rather than cost it."*

Two integration frictions: ShaderBox drives glfw/imgui by hand (no immapp), so
`ed.create_editor(config)` must be called manually in `App.__init__` — freska does exactly this with
the same raw-loop shape, so the pattern is proven outside immapp. And: **one shared editor context**,
with layout persisted by ShaderBox into `node.json` rather than by the library into its own file, so
that duplicating or exporting a node carries its board arrangement.

**Cost estimate: ~900-1000 lines of UI**, plus `chain_model.py` (~200 lines of pure, GL-free,
unit-testable logic: `Step`, `Wire`, `Chain`, topo sort, cycle check, self-loop rule).
**Plus theming, budgeted explicitly as "the single largest 'feels bolted on' risk, and a
visual-polish cost that no functional test catches"** — `imgui_node_editor` has its own
`StyleColor`/`StyleVar` enums, so `theme.py` needs a mapping pass or the board will look like a
different application dropped into the panel.

**What it makes genuinely worse:**

- **(a) There is no text form.** The agent engages the survey's objection directly rather than routing
  around it: *the survey's strongest objection does not land here* — it objects to an INTER-node
  document model, and this chain is strictly intra-node, so a node directory still holds everything
  and copies as a unit. **The objection that DOES land** is VVVV's "no support for multiple passes in
  shader code": **you cannot see the whole effect by reading files.** Someone `cat`-ing the node
  directory sees eight shaders and a JSON blob. *"That is a real and permanent loss and the strongest
  argument against this proposal."* Counterpoint the agent raises: this design introspects MORE than
  a text form does — the inputs are ordinary `sampler2D` declarations understood by every tool that
  reads GLSL, whereas a pragma design invents a micro-syntax the survey concedes "no other tooling
  understands"; the only thing stored outside the shader is what genuinely cannot live in it.
- **(b) Discoverability** — a user who never opens the Chain tab never learns steps exist.
- **(c) Copilot authoring is spatial, and an LLM cannot drag.** Three new tools (`add_step`,
  `wire_steps`, `set_step_target`) would be needed — a real budget item under tool-count discipline,
  where a text form would need zero because the copilot already has `write_shader`. Defensible first
  cut: ship the board copilot-READABLE but not copilot-WRITABLE.
- **(d) Eight-step chains are a lot of dragging.** Mitigated by `Ctrl+D` duplicate carrying chips
  forward, and drag-from-out-pin-onto-empty-canvas creating + auto-wiring a box in one gesture.
- **(e) Two places to look for a slider** (box face vs Node tab). Intentional, but it is two places.
- **(f) The board occupies a panel that is already tight** — narrower than the code editor, so a wide
  chain needs panning even at fit-zoom.

## 9. Self-verdict

R1-R10 all **COVERED**. *"Which obliges me to name where that verdict is thinnest, because a clean
sweep from the author is weak evidence."*

1. **Least confident: R10's ergonomics inside a node box.** The mechanism is sound (it IS
   `draw_ui_uniform`), but a ~460px row inside eight boxes on a panel narrower than the code editor
   will be cramped. Relying on `force_window_content_width_to_node_width` plus narrowed widths,
   **not run**. If box-face sliders prove unusable, R10 degrades to "click the box, use the Node
   tab" — still covered, materially less good than the journey claims. **First thing to prototype.**
2. **Theming** — most likely to overrun, most likely to leave the board looking like a second app,
   which is exactly the failure this design must avoid. No functional test catches it.
3. **R7's amber `8-bit` marker** — the rule is the agent's invention, not measured, and could
   false-positive on a deliberate tonemap-to-8-bit final step (correct code the marker would nag
   about). Should ship as a tooltip rather than a warning until there's a real rule.
4. **Did not read `render.py` / `share.py` / the exporter worker** — the Render-tab warm-up control's
   placement is an assumption.
5. **Unverified: whether `preview_cell` (which opens its own child window) works inside
   `ed.begin_node()`.** If child windows are illegal there, box thumbnails need a plainer
   `imgui.image` and lose the selection border and delete-X, weakening the "already looks native"
   claim.

## 10. False trails

- **A mode toggle on the big preview.** Flipping the result into the board means losing sight of the
  output at the exact moment you're wiring it.
- **A fourth `ActiveRegion`.** Changes keyboard muscle memory for everyone for a feature most nodes
  never use.
- **A floating/dockable board window.** ShaderBox has no docking; it would be the first, needing its
  own focus/z-order/persistence rules — "precisely the bolted-on second app failure the brief warns
  about".
- **Generic numbered pins wired by index.** Freska's documented defect 2 AND SHADERed's defect. "The
  named-uniform pin is not a refinement of this — it is what makes the drift class structurally
  impossible."
- **Declaring pins and parameter ranges in `node.json`, freska-style.** Reintroduces a declaration
  that can disagree with the shader, and throws away introspection ShaderBox already does better.
- **Storing chain topology in the node-editor settings file.** Both freska and the binding's own demo
  store ONLY layout. "Layout is disposable, topology is the effect. Merging them would make a lost
  layout file destroy the effect."
- **One editor context per node.** Layout would not travel with a duplicated or exported node.
- **A board spanning ALL nodes (a project-level patch).** Out of scope per the scenario, condemned by
  the survey, "and it would have made the naming collision unresolvable, since the boxes really would
  be nodes".
- **Auto-layout.** "It deletes the reason to be graph-first. Spatial memory — bloom hangs off to the
  side, the cascade staircases down — is the payload, and auto-layout reshuffles it whenever a wire
  changes." Kept as `[Fit]` plus a deliberate one-shot `Tidy`.
- **Wires carrying non-texture values (float, color), freska-style.** *"Nothing in the ten
  requirements asks for it... per that document's own rule, a proposal that cannot name the
  requirement it serves is invention, and gets cut. Cutting my own."*
- **Per-step `#ifdef PASS_N` branches with the board as a read-only visualiser.** A read-only board is
  a diagram, not an editor — graph-FLAVOURED rather than graph-first, carrying the full cost for none
  of the authoring benefit. **"Worth recording that this is the strongest NON-graph option and the
  one a text-first proposal should build on."**
- **Making the board the only view, deleting the Node tab's uniform list.** Would degrade the
  single-step case — a regression for the majority of nodes to serve the minority.
