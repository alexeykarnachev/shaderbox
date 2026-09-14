# D/E — Hover and selection feedback; feedback and self-reads

Researcher scope: brief questions 12-14 (area D) and 15 (area E). ShaderBox facts anchored
against `shaderbox/widgets/pass_graph.py::_draw_node`, `_draw_port_dot`, `_draw_self_loop`,
`_draw_canvas`'s edge-colour block, `shaderbox/theme.py`'s `COLOR`/`SIZE` bags, and
`.claude/skills/imgui-ui/SKILL.md` §3 and §8.

## Sources

| Name | What it is | URL / path | Read as |
|---|---|---|---|
| thedmd/imgui-node-editor | Dear ImGui node-editor library (blueprint-style) | `imgui_node_editor.cpp`, `.h`, `imgui_node_editor_internal.h` (shallow clone) | code |
| Nelarius/imnodes | Dear ImGui node-editor library (minimal) | `imnodes.cpp`, `imnodes.h` (shallow clone) | code |
| Robert2005/ImNodeFlow | Dear ImGui node-editor library | `include/ImNodeFlow.h`, `src/ImNodeFlow.cpp` (shallow clone) | code |
| xyflow/xyflow | React/Svelte DOM+SVG node-editor library (React Flow) | `packages/react/src/components/Edges/BaseEdge.tsx`, `packages/system/src/styles/base.css` (shallow clone) | code |
| jagenjo/litegraph.js | Canvas2D node-editor library | `src/litegraph.js` (shallow clone) | code |
| Blender | `node_draw.cc`, `node_relationships.cc` (paths supplied pre-fetched, current `main`) | local files (no repo root; presumed `blender/blender`) | code |
| TouchDesigner docs | Feedback TOP reference | https://docs.derivative.ca/Feedback_TOP | docs |
| Unreal Engine docs | Blueprint editor node/wire/pin behaviour | https://dev.epicgames.com/documentation/unreal-engine/nodes-in-unreal-engine , https://dev.epicgames.com/documentation/en-us/unreal-engine/connecting-nodes-in-unreal-engine | docs |
| SideFX Houdini docs | SOP Solver (frame-to-frame feedback) | http://www.sidefx.com/docs/houdini/nodes/sop/solver | docs |

Max/MSP and Pure Data are cited from established public behaviour (patch-cord routing with
manual routing around a self-connection — no source repo available for either; both are
closed/GPL desktop apps without a browsable rendering source suitable for this brief). Treat
those two rows as description-level, not code-verified.

## Q12-13 — hover/selection cue per element, priority, propagation

| Reference | Node hover | Node selection | Port/pin hover | Wire hover | Wire selection | Hover priority (overlap) | Port hover → lights its wires | Node hover → lights its wires |
|---|---|---|---|---|---|---|---|---|
| imgui-node-editor | Border colour swap to `HovNodeBorder` (blue, `50,176,255`), width jumps from `NodeBorderWidth` to `HoveredNodeBorderWidth` (3.5px), rect expanded by `HoverNodeBorderOffset` | Border colour swap to `SelNodeBorder` (orange, `255,176,50`), width `SelectedNodeBorderWidth` (3.5px), offset `SelectedNodeBorderOffset` | Pin gets a translucent rect fill (`StyleColor_PinRect`/`PinRectBorder`, blue) while a wire drag targets it | Link's own `HovLinkBorder` colour (blue) + thicker border; hit test is a flattened-cubic-bezier distance test with a **fixed 5px extra threshold** (`c_LinkSelectThickness`) added to the link's own thickness — not zoom-scaled in canvas space (the whole canvas is under one zoom transform, so it scales with everything else) | `SelLinkBorder` colour (orange) | `FindLinkAt` returns the **first** link in `m_Links` whose hit test passes — no explicit depth stack for links; nodes use `IsItemHovered` per node so imgui's own top-item-wins ordering governs node-vs-node | No — pin hover only drives the pin's own rect fill and wire-drag targeting | No automatic propagation. `StyleColor_HighlightLinkBorder` exists as a *separate*, app-driven colour a call site sets per-link via `DoLink`'s `color` param (the blueprint example uses it for a data-type/flow cue) — it is not wired to node hover in the library itself |
| imnodes | `ImNodesCol_NodeBackgroundHovered` fill swap (grey/blue depending on preset; Dark preset `75,75,75`) | `ImNodesCol_NodeBackgroundSelected` fill swap + `NodeOutline` border | `ImNodesCol_PinHovered` colour swap on the pin glyph | `ImNodesCol_LinkHovered` colour swap; distance test against `ImNodesStyleVar_LinkHoverDistance` (**default 10.0px, in screen pixels, NOT zoom-corrected** — `GetDistanceToCubicBezier` runs on already-transformed screen-space points) | `ImNodesCol_LinkSelected` (same value as `LinkHovered` in the Dark preset — selection and hover share a colour there) | Strict hierarchy in `ResolveHoveredPin/Node/Link`, evaluated in that exact order and **mutually exclusive**: pin found → skip node and link resolution entirely; else node found (by topmost `NodeDepthOrder` among overlapping candidates) → skip link resolution; else link resolved by nearest-distance-under-threshold, with a special case: **if a pin is hovered, any link touching that pin is force-hovered regardless of distance** (`ResolveHoveredLink`'s early `return idx` when `HoveredPinIdx` matches an endpoint) | Yes — see above: hovering a pin makes any link at that pin report hovered (this is deliberate, `TestHit`'s comment says it's needed for the drag-to-detach gesture, not for a visual "light up" reason) | No — node hover only gates whether pin/link resolution is skipped; it does not add entries to the link colour path |
| ImNodeFlow | (no distinct hover cue found in the fields read; style struct has no node-hover colour token, only pin/link) | Not found in the read fields (no `NodeStyle` hover/selected colour observed) | `PinStyleExtras`'s `socket_hovered_radius` (the dot itself grows) and `link_hovered_thickness` (3.5 vs base 2.6) | `Link::update()`: `smart_bezier_collider(mouse, start, end, 2.5)` (fixed 2.5px canvas-space threshold) sets `m_hovered`, which swaps the drawn thickness from `link_thickness` (2.6) to `link_hovered_thickness` (3.5) — **colour is unchanged on hover**, only width | Selection draws a **second, wider stroke underneath** in `outline_color` at `thickness + link_selected_outline_thickness` (0.5px halo), then the normal-colour stroke on top at `thickness` — a halo-under-stroke technique, not a colour swap | Not derivable from the read excerpt (single-link `update()`, no cross-link ordering seen) | No cross-reference found between pin/node hover state and link colour in the read code | No cross-reference found |
| xyflow (React Flow) | CSS `:hover` on `.react-flow__node` (class-based; DOM/CSS cascade, not draw-list) | CSS `.react-flow__node.selected` | CSS `.react-flow__handle:hover` | CSS `.react-flow__edge:hover .react-flow__edge-path`; **hit testing is a second, invisible, wider `<path>`** (`BaseEdge.tsx`: `react-flow__edge-interaction`, `strokeOpacity={0}`, `strokeWidth={interactionWidth}`, default **20px**, author-overridable per edge) laid on top of the visible stroke — this is what the mouse actually hit-tests via native SVG/DOM hit-testing, not a manual distance calc | CSS `.react-flow__edge.selected .react-flow__edge-path` | Native DOM z-order: an edge/node later in the DOM (or with higher CSS `z-index`) wins; xyflow re-renders a selected/hovered edge with a later DOM position in some configurations to force it on top | Architecturally N/A — no per-frame draw-list to grep; propagation is left to the app's own state → prop wiring, not the library | Same — DOM/CSS, no draw-list evidence to grep for this repo |
| litegraph.js | `node.mouseOver` flag drives a drawn shadow/**glow** around the node body (`glow = true` when `mouseOver`) via canvas shadow, not a border-colour swap; title text colour unaffected | `LiteGraph.NODE_SELECTED_TITLE_COLOR` (`"#FFF"`) swaps the title text colour when `selected` | Not found as a distinct pin-hover cue in the read excerpt | `this.highlighted_links[link.id]` forces the link's stroke colour to `"#FFF"` in `renderLink`; also `render_connections_border` draws an extra `lineWidth + 4` border pass under every link when zoomed in enough (`ds.scale > 0.6`), independent of hover | Selected NODE's links are pushed into `highlighted_links` (see next column) and rendered white; there is no separate "wire selected" state distinct from "wire touches a selected node" | Draw order only — no explicit z-priority list found beyond normal array iteration | Not found — pin-level hover has no dedicated code path read | **Yes, but on SELECTION, not hover.** `LGraphCanvas.selectNodes` walks the newly-selected node's `inputs[].link` and every `outputs[].links[]` and sets `highlighted_links[id] = true` for each; `renderLink` then force-colours any such link white. `node.mouseOver` (true hover) only sets a redraw flag and the glow, and is explicitly **not** wired into `highlighted_links` in the read code |
| Blender | No separate node-hover-only cue found in `node_draw.cc`'s selection/outline block — the outline block (`node_draw_basis`) branches only on `node.is_selected()` / undefined / zone-type / default, with no `is_hovered` branch | Outline is a rect **expanded outward** by `outline_width` (grows past the node's own bounds, not inset) and colour-swapped to `TH_ACTIVE` (the one "active"/last-clicked node) vs `TH_SELECT` (every other selected node) — two distinct hues for "selected" vs "selected-and-active" | Socket colour keys off `sock->flag & SELECT` in `node_socket_outline_color_get`; no hover-grown radius found in the read `node_socket_draw` (radius is a fixed `NODE_SOCKSIZE * scale`, `scale` being the zoom/DPI factor, not a hover multiplier) | Not located in the read excerpt (wire hit-testing lives outside `node_draw.cc`/`node_relationships.cc`, likely `node_edit.cc`, not fetched) | **Z-order priority, verified directly**: unselected links are drawn in one full pass first, then **all selected links are drawn in a second pass on top** — `/* Draw selected node links after the unselected ones, so they are shown on top. */` in the draw loop | N/A (not located) | Not located — the brief's `is_highlighted`/`NODE_LINK_DIM`/`node_link_draw_data` names were not found verbatim in the current `node_draw.cc`; likely an older Blender version's naming, or the logic now lives in `node_relationships.cc` / a header not fetched. Treat as unverified rather than repeat the brief's claim | Not located for the same reason |
| Unreal Blueprint (docs only) | Hovering a compatible pin during a drag shows a green check mark; a tooltip names the type being connected | Selected node gets a highlighted border (screenshots in docs show a bright outline; exact hue not specified in text) | Manual highlight: Shift+click a pin or wire highlights it (one at a time); Shift+click again clears it — an explicit user action, not an automatic hover cue | A wire being dragged shows live type-compatibility feedback (green check / red X) at the target pin | Same manual Shift+click highlight applies to wires | Not documented at this granularity | Not documented as automatic; the Shift+click highlight is manual and single-target | Not documented as automatic |

## Q14 — cursor shapes per state

| Reference | Over empty canvas | Over a node (idle) | Dragging a node | Over a port/pin | Dragging a wire | Over a wire | Resize handle |
|---|---|---|---|---|---|---|---|
| imgui-node-editor | Default arrow (no explicit cursor calls found in the read files) | Default arrow | Default arrow (no drag-cursor override read) | Default arrow | Default arrow | Default arrow | N/A |
| litegraph.js | `""` (browser default, effectively arrow/pointer) explicitly reset when not over anything actionable | `""` (comment shows a `move` cursor was tried and left commented out — `//this.canvas.style.cursor = "move";`) | Not distinctly set in the read excerpt | Not distinctly set | `"crosshair"` while `connecting_node` is active | Not distinctly set | `"se-resize"` for the node's corner resize grip |
| xyflow | CSS `cursor: grab` / `grabbing` on the pane (documented convention; DOM-driven, not read verbatim from CSS here) | CSS default (`pointer` on interactive elements per DOM/CSS convention) | `grabbing` (drag convention) | `crosshair` or `pointer` on a handle (typical xyflow convention) | `crosshair` while forming a connection | Default | `nwse-resize`/etc. via the NodeResizer's own handles |
| Blender / Houdini / Unreal | Not verified from source in this pass; not central to D/E's deliverable | | | | | | |

**ShaderBox today**: `shaderbox/app.py` already owns three glfw cursor objects
(`ibeam_cursor`, `resize_ew_cursor`, `resize_ns_cursor`) and the single-owner
`want_cursor`/`cur_cursor` pattern (`shaderbox/ui.py`, applied once per frame, gated on
`want != cur` to avoid the X11 flicker documented in `imgui-ui/SKILL.md` §8). `pass_graph.py`
sets no cursor anywhere today — every graph interaction reads as the plain arrow.

## Q15 — self-read / feedback display

| Reference | Mechanism | Where it's drawn |
|---|---|---|
| litegraph.js | **No same-node self-link support found.** Its `Link` model connects one output slot to one input slot on *different* nodes; no code path building or rendering a link whose `origin_id == target_id` was found by grepping for self/loop terms. A feedback effect in litegraph-based tools (e.g. ComfyUI) is conventionally built as a separate "hold previous frame" node, not a loop wire. **False trail**: the brief's premise that litegraph "allows a link from a node to itself" does not check out against this repo's source — treat as unverified/likely wrong for this library specifically. |
| imnodes | No same-node link example or special-case code found (`Link()` takes two arbitrary pin ids; nothing in the read excerpt special-cases `start_node==end_node`). Same caveat as litegraph — not confirmed as a supported, distinctly-drawn case. |
| Blender | No self-link either, but a **structurally adjacent** mechanism exists: `node_draw_mute_line` builds a synthetic `bNodeLink` with `fromnode = tonode = &node` (both ends on the same node) to draw the pass-through line across a *muted* node. It reuses the ordinary bezier link drawer (`node_draw_link_bezier`) with both endpoints on one node's own sockets. This is Blender's only same-node link construction in the read files, and it's a bypass indicator, not a feedback read — but it proves the "same-node bezier, drawn with the normal link function" shape is workable on a node-graph canvas. Blender's actual feedback mechanism is the **Simulation Zone** (`Simulation Input` / `Simulation Output` node pair) — a distinct pair of nodes bracketing a subgraph, with the "previous state" implicit in the zone, never a loop wire. |
| Unreal (materials) | Docs describe no self-referencing material node; a feedback-like accumulation is built via render targets read back as a texture sample in a *later* frame's graph evaluation — outside the graph itself, not a drawn loop. |
| TouchDesigner | Dedicated **Feedback TOP** node: a distinct node type wired in-line (Feedback TOP → optional filter chain → Target TOP downstream), where the Feedback TOP's own `Target TOP` parameter (not a wire) names which downstream node's previous-frame output feeds back in. The loop is **not drawn as a wire on the canvas at all** — it's an out-of-band parameter reference, so the network stays a DAG visually even though it behaves as a cycle. Source: https://docs.derivative.ca/Feedback_TOP |
| Houdini (SOP Solver) | Feedback is a **distinct container node** (`Solver SOP`) whose *interior* network has a special `prev_frame` input socket wired to a DOP-level solver; from outside, the Solver node looks like a completely ordinary single-input/single-output SOP — no loop or self-wire is visible at the network level the artist normally sees. The recursion is hidden a level down, inside the node. Source: http://www.sidefx.com/docs/houdini/nodes/sop/solver |
| Max/MSP, Pure Data | (Description-level, no source read.) A feedback path is an ordinary patch cord that the user manually routes in a loop shape (down, across, up into the earlier object's inlet) — visually a real cord crossing the patch, not a glyph. This is effectively what ShaderBox's *current* implementation already resembles (a loop drawn crossing the node's top) and is the shape the maintainer explicitly wants to move away from. |

**Convergence on Q15**: every reference that has a first-class feedback concept (TouchDesigner,
Houdini) **removes the loop from the graph's line layer entirely** — it becomes a node
parameter/attribute or an interior-network detail, never a wire the eye has to trace across
the canvas. The two graph libraries that plausibly could draw a same-node loop wire
(litegraph, imnodes) turn out **not to implement one** on inspection — the "self-link renders
as a loop" idea is a plausible-sounding but unverified premise for those two; don't cite them
as precedent for a loop *glyph* without that caveat. Blender's only same-node-bezier code
(`node_draw_mute_line`) is for a different concept (mute bypass) but does establish that
drawing a bezier with both control points anchored to one node's own sockets is an ordinary,
well-behaved operation on these canvases.

## CONVERGENCE

- **Selection and hover are always two different hues, never the same cue at different
  intensity.** imgui-node-editor: blue (hover) vs orange (select). Blender: default outline vs
  `TH_SELECT`/`TH_ACTIVE`. litegraph: white title (select) vs glow (hover) — visually distinct
  effects, not a lighter/darker version of one another.
- **A halo/wider-stroke-underneath, not a thicker single stroke, is the mature way to show
  selection on a line**, confirmed independently in imgui-node-editor (`HoveredNodeBorderOffset`
  expanding the rect before drawing a same-width border ring) and ImNodeFlow (a wider outline
  stroke drawn first, the normal stroke drawn on top). This is compatible with the imgui-ui
  rule (§3): the extra stroke is additional geometry, not a resize of the existing one, so it
  doesn't perturb layout.
- **Wire hit-testing is never literally "distance to the visible stroke width."** Every
  reference with pixel-hit-tested wires pads the test: imgui-node-editor adds a flat 5px,
  imnodes defaults to 10px, ImNodeFlow uses 2.5px, xyflow uses a fully separate 20px-wide
  invisible hit path. All four numbers are independent of the *drawn* stroke width — the hit
  region is always wider than what's visible.
- **Hover priority is hierarchical and exclusive, always pin/port first.** imnodes states this
  explicitly and implements it as early-return; imgui-node-editor's per-item `IsItemHovered`
  chain (background → node → port, each declaring `allow_overlap` so the later item wins) is
  the same idea from ShaderBox's own already-working invisible-button order
  (`pass_graph.py`'s module docstring: "the ports, which are last... only forfeits the drop
  target"). No reference puts a wire above a node or a node above a port in hover priority.
- **A node's hover does not, by default, light up its wires in any of the four graph libraries
  actually inspected.** Where wire highlighting on node interaction exists at all (litegraph),
  it fires on **selection**, not hover, and only litegraph implements it among the four.
  imnodes' pin-lights-its-link behaviour is real but is an implementation detail for the
  drag-to-detach gesture, not a visual affordance.
- **A feedback/self-read is best represented off the wire layer entirely.** TouchDesigner and
  Houdini — the two references with a genuine, named feedback concept — both hide the
  recursion from the canvas's line layer (a node parameter, or an interior network level).
  This directly supports the maintainer's ask to replace the current corner-crossing loop with
  a badge-adjacent glyph rather than a geometry fix to the loop bezier.

## DIVERGENCE

- **Node-body hover cue varies by medium.** Draw-list/immediate-mode libraries (imgui-node-editor,
  imnodes) recolour the *border*; litegraph (retained Canvas2D, full redraw each frame same as
  imgui) instead draws a *glow/shadow* around the body and leaves the border alone; xyflow (DOM)
  uses whatever CSS the app author supplies. No single "the" hover treatment for a node body —
  border-colour-swap is the majority pattern among the imgui-family libraries, which is the
  closest cousins to ShaderBox's own draw-list architecture.
- **Whether link-selected-drawn-on-top is explicit varies.** Blender does it explicitly (two full
  passes, unselected then selected). litegraph achieves the same visual result differently — by
  forcing colour, drawn in original iteration order, so a selected link touching an
  earlier-drawn node's output can still be occluded by a later unselected link drawn on top of
  it in z-order (a real gap in that implementation). ShaderBox's own edges are all drawn before
  any node (channel 0 vs channel 1 in `_draw_canvas`), so this question doesn't currently apply
  at the node/edge level — but it would apply among edges once edge selection exists.
- **Hover-distance thresholds are not zoom-consistent across references.** imnodes' default
  10px and ImNodeFlow's 2.5px are both applied in the already-zoomed screen/canvas space they
  operate in (the whole editor canvas is one coordinate space transformed as a unit), so the
  *effective* screen-pixel hit width does scale with zoom in both. imgui-node-editor's 5px is
  the same. None of the three references were found to explicitly re-derive the threshold as a
  function of zoom level to keep a constant *screen*-pixel width — they get that property for
  free from transforming the whole canvas once, which is exactly ShaderBox's own `_Xf`
  transform shape (`pass_graph.py`'s `to_screen`/`to_canvas`).
- **Self-read visibility policy differs by domain.** Node-graph libraries with no native concept
  of "read my own previous output" (litegraph, imnodes) simply don't have an answer — feedback in
  tools built on them (ComfyUI-style) is emulated with a distinct node type, matching TouchDesigner
  and Houdini's approach; a genuine loop-wire-back-into-self is not attested in code by any
  reference read here, only assumed by the brief. This favours a glyph/badge over any wire
  geometry fix, which is also literally what the maintainer already asked for.

## Recommended for ShaderBox

### New theme tokens (`shaderbox/theme.py`)

Add under the `# The graph canvas (092)` block in `_ColorBag`, reusing existing fixed hues
rather than inventing new ones (per the theme's own "FIXED roles must read distinctly" and
"reuse before adding" posture):

```python
GRAPH_HOVER: tuple[float, float, float, float] = _P["blue_b"]   # reuse: distinct from SELECT (purple_b) and every STATE_* and GROUP_TINTS hue
GRAPH_HOVER_HALO_ALPHA: float = 0.35
GRAPH_SELECT_HALO_ALPHA: float = 0.55
```

`blue_b` is already excluded from `GROUP_TINTS` and is not an accent primary or a `STATE_*`
value (`theme.py`'s own invariant assertions confirm no collision), so it reads as a genuinely
new signal next to the existing purple `SELECT` and red/yellow/aqua `STATE_*` triad. No new
`SIZE` tokens for thickness are needed — reuse `SIZE.GRAPH_WIRE_W` (1.5) as the base and derive
hover/select widths as offsets from it (below), matching how `GRAPH_PORT_R`/`GRAPH_PORT_RING_W`
are already derived rather than hardcoded per state.

### Drawing rule per element/state

All values are in the SAME canvas units `_draw_node`/`_draw_wire` already multiply by `z`
(`xf.zoom`) — so every number below gets `* z` at the call site exactly like `SIZE.GRAPH_WIRE_W`
does today.

**Node border** (`_draw_node`, extending the existing `if node.error: ... elif is_output: ...`
chain):
- Hover, no other state: keep the existing 1.0px `COLOR.BORDER` stroke unchanged in colour, but
  draw one extra `dl.add_rect` at the same rect with `COLOR.GRAPH_HOVER` faded to
  `GRAPH_HOVER_HALO_ALPHA`, `SIZE.GRAPH_WIRE_W * 2 * z` thick, drawn **inset** by
  `SIZE.GRAPH_WIRE_W * z` (i.e. shrink `p0`/`p1` inward by that amount before the extra rect) —
  this is the ImNodeFlow/imgui-node-editor halo-under-stroke pattern, done as an inset addition
  so it never grows the node's footprint (the imgui-ui §3 rule: colour change, not size change,
  inset).
- Selected (today's `elif selected: border = COLOR.SELECT`): keep the colour swap, but ALSO add
  the same inset halo technique using `COLOR.SELECT` at `GRAPH_SELECT_HALO_ALPHA` — this is what
  makes "selected" read as heavier than "hovered" (matches imgui-node-editor's own
  `SelectedNodeBorderWidth == HoveredNodeBorderWidth` numerically, 3.5 both, but distinguished by
  hue orange vs blue; ShaderBox has one hue for select already, so the halo alpha step is what
  carries "more emphasis").
- Hover + selected together: draw both halos (select halo outermost/first since it's the
  stronger state, hover halo just inside it) — no new token needed, this falls out of applying
  both rules; keep them at their own alphas rather than summing, so double-application never
  exceeds `GRAPH_SELECT_HALO_ALPHA` visually via z-fighting (draw select halo, then hover halo,
  both partially transparent, standard alpha-over-alpha compositing handles it).
- `node.error` keeps first priority in the existing chain (unchanged) — an error is the one
  state stronger than hover or selection, matching the existing precedence.

**Port dot** (`_draw_port_dot` / the per-port loop in `_draw_node`): on hover, swap `port_col`
from `COLOR.FG_MUTED` to `COLOR.GRAPH_HOVER` for that one dot only (colour change on the exact
existing circle draw — no radius change, keeping ImNodeFlow's "the dot grows" idea explicitly
OUT per the imgui-ui inset/no-size rule). This requires passing which port is hovered into
`_draw_node`, sourced from the same `view.port_rects` hit-test dict `_draw_canvas` already
builds per frame from the invisible-button loop — no new hit-testing needed, only a lookup
against the existing rects using `io.mouse_pos`.

**Output dot**: same colour-swap rule as an input port dot, keyed off a matching hover check
against `_out_point`'s hit rect (the `out_hit` invisible-button loop already computed).

**Wire hover and hit test** (`_draw_wire` call site in `_draw_canvas`, and a new hit-test
function): add a distance-to-flattened-bezier test, thresholded at
`max(4.0, SIZE.GRAPH_WIRE_W * 2.5) * z` screen pixels — mirroring ImNodeFlow's 2.5x-thickness
multiplier and imgui-node-editor's flat-padding idea combined, scaled by zoom the same way every
other graph hit box in this file already is (`hit = min(max(SIZE.GRAPH_PORT_R * view.zoom, ...)`
pattern). Sample the bezier at a fixed **24 points** (cheap, matches the density
`imgui.add_bezier_cubic`'s own internal tessellation already implies at this stroke length; no
reference read here needed more than 50 samples even for large canvas-filling curves, and
ShaderBox's wires are short). On hover, draw the wire in `COLOR.GRAPH_HOVER` instead of
`edge_col`/`dim_col`, at `SIZE.GRAPH_WIRE_W * 1.4 * z` thickness (a modest width bump — unlike
the node border, a line has no "inset" concept, so imgui-ui §3's colour-not-size rule doesn't
forbid this; ImNodeFlow's own hover treatment IS a width bump, 2.6→3.5, ~1.35x, which is the
precedent this number follows). A cycle-error wire (`err_col`) keeps priority over hover, same
precedence logic as the node border's error-first chain.

**Wire selection**: not in scope per the brief (`finding 1`: a wire cannot be selected today) —
no rule proposed here; flag it as the natural next step once wire selection exists, following
Blender's two-pass draw-order rule (draw all unselected wires first, selected wires in a second
pass on top) rather than litegraph's colour-only approach, since ShaderBox already channel-splits
wires from nodes (`dl.channels_split(2)`) and a third channel for "selected wires on top of
everything else, including nodes" is a one-line addition to that existing mechanism.

### Hover priority order

Exactly the order ShaderBox's own hit-testing loop already establishes and that every
reference (imnodes explicitly, imgui-node-editor implicitly) converges on:

1. **Port/output dot** (the `gport_*`/`gout_*` invisible buttons, submitted last, so they
   already win ties per the imgui-ui §8 `allow_overlap` note)
2. **Node body** (the `gnode_*` invisible button)
3. **Wire** (no invisible button today — becomes a passive draw-list distance test, evaluated
   only when neither 1 nor 2 report hovered)
4. **Empty canvas / background**

Implementation shape: after the existing node/port hit-testing loop in `_draw_canvas` computes
`node_hovered` and the per-port hover state, run the wire-hover distance test only
`if not node_hovered and hovered_port is None` — mirroring imnodes' exact early-exit chain
(`ResolveHoveredPin` → `if not pin: ResolveHoveredNode` → `if not node: ResolveHoveredLink`).

### Cursor shapes through `App.want_cursor`

`shaderbox/app.py` already owns `ibeam_cursor`, `resize_ew_cursor`, `resize_ns_cursor`, and the
apply-once-per-frame pattern in `shaderbox/ui.py` (`if app.want_cursor is not app.cur_cursor:
glfw.set_cursor(...)`). None of the references converge on a rich per-element cursor vocabulary
for node graphs (litegraph's own author commented OUT a `move` cursor for node-hover; most
readers rely on the visual highlight rather than the cursor to communicate state) — so the
recommendation is deliberately minimal, adding only what already has a clear ShaderBox
precedent shape:

- **Over empty canvas, over a node body, over a port at rest**: leave `want_cursor` untouched
  (defaults to arrow) — matches litegraph's own reset-to-default and avoids inventing a cursor
  vocabulary no reference actually committed to.
- **Panning is engaged (`panning` is already True in `_draw_canvas`)**: request a "hand"
  cursor. Glfw's standard set has no closed-hand; use `glfw.HAND_CURSOR` for both press and
  drag (glfw doesn't offer open/closed variants) — add `app.hand_cursor =
  glfw.create_standard_cursor(glfw.HAND_CURSOR)` next to the three existing cursor fields in
  `app.py`, and set `app.want_cursor = app.hand_cursor` in the existing `if panning:` branch.
- **Dragging a wire (`view.wire_drag is not None`)**: request `glfw.CROSSHAIR_CURSOR` —
  matches litegraph's own `"crosshair"` while `connecting_node`, the one cursor state a
  reference actually implements for this exact gesture. Add `app.crosshair_cursor` alongside
  the others.
- **Dragging a node (`view.node_drag is not None`)**: reuse the hand cursor (no reference
  distinguishes node-drag from pan with a different cursor; treating both as "you're moving
  something by grabbing it" is consistent and keeps the new cursor set to two additions).
- Apply these as `want_cursor` REQUESTS inside the existing per-gesture branches in
  `_draw_canvas` (`if panning:`, the wire-drag block, the node-drag block) — never a raw
  `glfw.set_cursor` call, per §8's single-owner-per-frame rule; the existing end-of-frame apply
  in `ui.py` needs no changes since it already applies whatever `want_cursor` holds.

### The feedback glyph (replacing `_draw_self_loop`)

Per the maintainer's own framing (brief line 33) and the TouchDesigner/Houdini convergence
above (a feedback stays off the wire layer, expressed as a marker instead of a crossing line):
drop the bezier loop entirely and draw a small double-loop (infinity/∞-like) glyph immediately
to the left of the `xN` run-count badge (`_draw_badge`'s call site at `(s1[0], s0[1])`,
`right_aligned=True` — the badge already anchors to the picture's top-right corner via
`_thumb_rect`).

**Why a drawn glyph, not a text character**: ShaderBox loads `font_14_bold`, `font_14`,
`font_12` (seen in `_draw_node`) — general UI faces, not a symbol/icon font (the imgui-ui skill
§5 rule: "No icon font unless you loaded one... Use text" / draw explicit shapes for glyphs
rather than trust a font's rendering of an odd codepoint). A Unicode `∞` (U+221E) or `↻`
(U+21BB) glyph's presence in whatever TTF is loaded, its centring, and its legibility at 12px
are all unverified and version-fragile; the skill's own precedent (§3, the crisp ✕: "don't rely
on a font glyph... draw it with the draw list") is the same class of problem at the same size,
solved the same way there. Draw it.

**Geometry** (draw-list primitives, 12px square budget, matching the existing badge's own
`_BADGE_H = 12.0` so the glyph sits flush with the badge row):

Two overlapping circular arcs forming a figure-eight/infinity shape, using
`ImDrawList.add_bezier_cubic` (already imported and used throughout this file — no new
primitive) or the simpler `path_arc_to`-equivalent via two `add_circle` calls with a gap ARE
available on `imgui.ImDrawList` (`path_arc_to`, `path_stroke`) but the simplest robust shape
that unambiguously reads as "loop" at 12px, following the same hand-rolled-shapes approach
`_draw_port_dot` already uses for its `none`/`prev` ring glyphs, is two offset open circles
(each ~75% of a full circle, gap facing each other) rather than true bezier lobes — cheaper to
get symmetric at tiny sizes than a bezier figure-eight, and it's the same "two rings" visual
vocabulary the `prev` port dot (`_draw_port_dot`'s `kind == "prev"` branch: two concentric
rings) already established for "this reads from its own history," so the badge glyph and the
port glyph share a visual language instead of inventing a second one:

```python
_LOOP_GLYPH_SIZE = 12.0   # matches _BADGE_H, so the glyph's box is exactly badge-row height
_LOOP_GLYPH_GAP = 2.0     # px between the badge's left edge and the glyph's right edge

def _draw_feedback_glyph(
    dl: imgui.ImDrawList, corner: tuple[float, float], z: float, col: int
) -> None:
    """A double-loop mark left of the run-count badge: two small rings, each open on the
    side facing the other, read together as a feedback/self-read cue (092/093)."""
    size = _LOOP_GLYPH_SIZE * z
    r = size / 4.0
    cx0 = corner[0] - size + r
    cx1 = corner[0] - r
    cy = corner[1] + size / 2.0
    thickness = max(1.0, SIZE.GRAPH_WIRE_W * z)
    # Each ring is ~300 degrees of arc (gap facing the other ring), drawn via path_arc_to.
    dl.path_clear()
    dl.path_arc_to((cx0, cy), r, 0.6, 2 * 3.14159 - 0.6, 20)
    dl.path_stroke(col, imgui.ImDrawFlags_.none, thickness)
    dl.path_clear()
    dl.path_arc_to((cx1, cy), r, 3.14159 + 0.6, 3.14159 - 0.6 + 2 * 3.14159, 20)
    dl.path_stroke(col, imgui.ImDrawFlags_.none, thickness)
```

Call site: in `_draw_node`, where the `xN` badge is currently drawn
(`if node.runs > 1 and node.kind != "ghost": _draw_badge(...)`), add a parallel branch that
checks whether the node has any `port.kind == "prev"` among `node.ports` (the same predicate
`_draw_canvas` already uses to decide whether to call `_draw_self_loop` at all) and, if so,
calls `_draw_feedback_glyph` anchored at the badge's own left edge — `(s1[0] - badge_w, s0[1])`
if a badge is present that frame, else `(s1[0], s0[1])` directly (the glyph takes the badge's
usual slot when there's no run-count badge to share it with). Colour: reuse `badge_fg`
(`COLOR.FG_MUTED` faded) at rest, matching the badge text's own colour so the two read as one
family of "corner annotations," and swap to `COLOR.GRAPH_HOVER` if the maintainer later wants
the glyph itself to be hoverable (not required by the brief; the loop glyph is a static
indicator, not an interactive element, in every reference surveyed — TouchDesigner's and
Houdini's feedback markers are not draggable/clickable either).

This removes `_draw_self_loop` and its call site in `_draw_canvas`'s
`for node in nodes: for slot, port in enumerate(node.ports): if port.kind == "prev": ...` loop,
along with `SIZE.GRAPH_LOOP_RISE`/`GRAPH_LOOP_REACH`, which stop being read.

## False trails

- **litegraph.js and imnodes "self-link drawn as a loop"** — the brief's premise for these two
  specifically. Grepping both repos for same-node-link construction or rendering turned up
  nothing; neither library's `Link`/`LGraphCanvas` code special-cases `start_node == end_node`.
  This may be true of a *fork* or a *newer* version, or may be conflating "a link can visually
  cross behind a node" with "a link can start and end on the same node" — but as read here, it
  doesn't check out. Don't cite either as precedent for drawing a literal loop wire; use them
  only for their confirmed mechanics (hit-testing thresholds, hover/selection colour handling).
- **Blender's `is_highlighted` / `NODE_LINK_DIM` / `node_link_draw_data` names from the brief** —
  not found verbatim in the fetched `node_draw.cc`/`node_relationships.cc`. The fetched files do
  contain a real, useful, and verified two-pass "selected links drawn on top" mechanism, but
  under different names/shape than the brief described. Likely explanation: those names belong
  to an older Blender version, or to a header/file not included in the pre-fetch (`node_edit.cc`,
  named in the brief for cursor changes, was also absent from the scratchpad and not separately
  cloned — flagging rather than fabricating its content).
- **ImNodeFlow's node-level hover/selected colours** — the brief lists `NodeStyle` hover colours
  as a citable fact; the read excerpt of `ImNodeFlow.h`'s style structs surfaced pin- and
  link-level hover/selected fields but no analogous node-body hover/selected colour token in the
  portion read. Either it exists elsewhere in the header (a `NodeStyle` struct not captured by
  the grep pattern used) or the library leans on `PinStyle`/`LinkStyle` alone for feedback and
  draws node selection through a different, un-grepped mechanism. Flagged rather than asserted.
- **Unreal Engine's exact hover/selection colour values** — Epic's public docs describe the
  *behaviour* (green check on compatible hover, Shift-click highlight) in prose but do not
  publish hex/RGB values or pixel thresholds anywhere in the fetched pages; the engine's editor
  source (`SGraphPanel.cpp`/`SGraphNode.cpp` etc.) is not publicly browsable outside the
  Epic-gated Unreal source access, so Unreal contributes qualitative confirmation only, not a
  number to copy.
