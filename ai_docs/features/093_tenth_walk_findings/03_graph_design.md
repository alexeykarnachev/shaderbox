# 093 — The graph editor's redesign: the design record

The maintainer's verdict (`02_research_brief.md`, verbatim) was that the graph editor "feels very
cheap": cusped beziers, uncertain controls, cards too small for their own names, nothing
highlighting on hover. Six researchers answered areas A-G against primary sources. This record is
what the research settled, the decisions G1-G18 a coder can implement blind, what deliberately
stays as it is, the forks left for the maintainer, the false trails consolidated, and the
falsifier per guarantee.

Every decision below assumes the settled move of `01_research_graph_tab.md`: the graph is a third
editor-tab kind, one per document, so the canvas gets the full editor pane rather than the
Document tab's leftovers. The card-width arithmetic in G11 is computed against that pane, not
against today's cramped child.

**What was checked against the primary source, not the report.** The reports' numbers are
load-bearing, so five contested constants were re-read from the clones under the scratchpad's
`refs/` rather than trusted as relayed: imgui-node-editor's `easeLinkStrength` (the `sin` ease is
`strength * sin(π/2 · halfDistance/strength)` with the argument in `[0, π/2]`, so the result is
non-negative — A's proof stands), its `SourceDirection = (1,0)` / `TargetDirection = (-1,0)` and
`LinkStrength = 100.0f` (`imgui_node_editor.h`), its `c_LinkSelectThickness = 5.0f; // canvas
pixels` and `Link::TestHit`'s `ImProjectOnCubicBezier(..., 50)` against `m_Thickness +
extraThickness` (`imgui_node_editor.cpp`), imnodes' `LinkThickness(3.f),
LinkLineSegmentsPerLength(0.1f), LinkHoverDistance(10.f) ... PinHoverRadius(10.f)` (`imnodes.cpp`
style ctor), and ImNodeFlow's `smart_bezier` in full (`src/ImNodeFlow.inl`) — which confirms A's
reading exactly: `p22` takes the un-flipped `delta` in every case and `p11`'s sign flips only past
`p2.x < p1.x - 50.f`, so the 0-50px backward band folds. The font advance (0.545898 × em,
monospace both weights) is taken from C's measurement and used unmodified; the fit-to-view
arithmetic was recomputed here because C's table omitted `_FIT_MARGIN` (see G11).

---

## 1. What the research settled

**A — wire geometry.** Four of six code references (imgui-node-editor, imnodes, litegraph.js,
Blender) are cusp-proof not by special-casing the backward direction but by construction: the
control-point offset is always non-negative and always points along its own endpoint's fixed
outward axis, so `cp0.x = a.x + offset` and `cp1.x = b.x - offset` can never cross, because
crossing needs `2·offset ≤ b.x - a.x` and the right side is negative whenever the consumer is left
of the producer (`A_wire_geometry.md` Q2, the structural proof; verified against
`imgui_node_editor.cpp`'s `GetCurve`/`easeLinkStrength` and `imgui_node_editor.h`'s fixed
`SourceDirection`/`TargetDirection`). The two references that compute a *signed* offset (xyflow,
ImNodeFlow) are the only two needing a backward branch, and ImNodeFlow's ships broken — its
hardcoded 50px flip leaves a 0-50px band where both control points pull the same way and the curve
folds, the same failure shape as ShaderBox's bus. No reference routes wires around node bodies;
all rely on z-order and let the user move nodes.

**B — mouse controls.** ShaderBox's bindings already match the convergent schema. Pan lives off
the plain left button in every 2D-canvas widget library read (imnodes and ImNodeFlow default to
middle-drag; Houdini's Space+LMB is the same hold-a-key shape as our Alt+drag), wheel zooms about
the cursor everywhere, left-drag on empty canvas rubber-bands wherever pan is not on the left
button, and pressing a filled input to re-grab its wire with no modifier is litegraph.js's default
(`allow_reconnect_links = true`) — which is exactly what `_draw_canvas` does today
(`B_mouse_controls.md` Q6-Q8). The one genuine gap: every reference that binds delete at all binds
it as a **bare, unmodified key** read locally against hover-and-selection, never through a global
rebindable keymap (imgui-node-editor's `ImGuiKey_Delete`, ImNodeFlow's
`IsKeyPressed(Delete) && !IsAnyItemActive()`, litegraph's `keyCode == 46 or 8`, Blender's
`X`/`Delete`, Unreal's `Delete`).

**C — the node card.** Card sizing splits cleanly by rendering substrate, not by taste: every
Dear-ImGui reference auto-sizes because `BeginGroup`/`EndGroup` makes it nearly free there, xyflow
fixes at 150px because CSS boxes do, and litegraph.js — the only raw-canvas sibling — measures text
itself rather than accept clipping (`C_node_card.md` Q11 and its DIVERGENCE). ShaderBox draws on a
bare `ImDrawList` with no layout pass, so the reachable answer is Houdini's: a fixed width budget
plus an explicit shorten policy ("Shorten Long Node Names" + "Maximum Node Name Width",
`G_ux_guidance.md` area C). No reference documents silent unguarded overflow as a choice — where
it is unhandled, auto-width already prevented the case. ShaderBox's `GRAPH_PORT_R = 4` already
matches the converged 4-4.67px dot radius.

**D/E — hover and feedback.** Hover priority is hierarchical and exclusive everywhere, always
port first: imnodes implements it as a literal early-return chain (`ResolveHoveredPin` → else
`ResolveHoveredNode` → else `ResolveHoveredLink`), imgui-node-editor gets the same order from its
reverse per-node scan with each node's pins checked before its body. Selection and hover are always
two different hues, never one cue at two intensities. A wider stroke drawn *underneath* is the
mature way to mark a line, confirmed independently in imgui-node-editor and ImNodeFlow. Wire hit
regions are always padded well past the drawn stroke (5px flat, 10px, 2.5px, a separate 20px
invisible path) (`D_E_hover_feedback.md` Q12-13, CONVERGENCE). On feedback: every reference with a
first-class feedback concept — TouchDesigner's Feedback TOP, Houdini's Solver SOP — removes the
loop from the line layer entirely, and the two libraries the brief assumed draw a self-loop
(litegraph, imnodes) turn out not to implement one at all (`D_E_hover_feedback.md` Q15 and its
false trails). The maintainer's badge-glyph instinct is the converged answer.

**F — machinery on a draw list.** All four references hand-roll wire hit-testing (a curve has no
natural `InvisibleButton` shape) while node/pin hit-testing splits 1-2: imgui-node-editor keeps
nodes and pins inside ImGui's own `ButtonBehavior`, imnodes and ImNodeFlow hand-roll those too.
ShaderBox sits on imgui-node-editor's side and should stay there. Channel splitting is the
canonical ImGui answer to layering, confirmed upstream. ShaderBox's `push_font(font,
legacy_size * z)` is already the crisp-text strategy ImNodeFlow reaches a nested `ImGuiContext`
for — F's job there was to confirm it, not replace it. F's headless probe settles the one
mechanical question the 5-channel scheme depends on: **paint order follows channel index, not call
order**, read correctly through the index buffer (the vertex buffer stays in submission order and
reading it is the trap that made the first probe read as a null result) (`F_imgui_machinery.md`,
the probe section).

**G — the written guidance.** Every mature editor gives the user a cheap explicit way to bend a
wire's path without changing what it connects — Blender's reroute node, Houdini's dots, Godot's
`VisualShaderNodeReroute`, Unreal's double-click-wire, Max/MSP's segmented cords — and the sources
treat that as *complementary to*, never a substitute for, fixing the curve math
(`G_ux_guidance.md` area B). Purchase's result, quoted with citation through the Kobourov survey,
ranks edge crossings far above bend count and curve smoothness as a readability factor, at a graph
scale (16 vertices, 18-28 edges) directly comparable to a pass graph.

---

## 2. Decisions

Coordinate spaces are named explicitly throughout. **Canvas space** is zoom-1 units, the space
positions and `node_size` live in. **Screen space** is post-`_Xf.to_screen` pixels. Where a
threshold is given in screen pixels it does not scale with zoom unless the formula says so.

### G1. We decide the wire is one cubic bezier, pin to pin, with a branch-free non-negative offset

**Rule.** For every edge, forward or backward, in screen space:

```
p0 = xf.to_screen(out_point)          # a
p3 = xf.to_screen(port_point)         # b
dx, dy = p3[0] - p0[0], p3[1] - p0[1]
dist   = sqrt(dx*dx + dy*dy)
offset = max(SIZE.GRAPH_WIRE_MIN_OFF * z, SIZE.GRAPH_WIRE_BOW * dist)
cp0 = (p0[0] + offset, p0[1])
cp1 = (p3[0] - offset, p3[1])
dl.add_bezier_cubic(p0, cp0, cp1, p3, col, max(1.0, SIZE.GRAPH_WIRE_W * z))
```

No branch on `dx`'s sign, no threshold, no second curve family, no bus.

**The algebraic reason it is cusp-proof.** A fold requires the two control points to cross in x,
i.e. `p0.x + offset ≤ p3.x - offset`, i.e. `2·offset ≤ dx`. `offset` is a `max` of two
non-negative terms, so it is non-negative; in the backward case `dx < 0`, so the right-hand side is
negative and the inequality can never hold, for any `dist` and any `GRAPH_WIRE_BOW`. The guarantee
is structural, not tuned: it does not depend on the constants' values.

**Source.** `A_wire_geometry.md` Q1/Q2 and its recommendation; the primary is
`imgui_node_editor.cpp::ed::Link::GetCurve` with `imgui_node_editor.h`'s
`SourceDirection = (1,0)` / `TargetDirection = (-1,0)` (re-read here), corroborated in
`imnodes.cpp::GetCubicBezier` (offset `0.25 × link_length`, non-negative, applied to the
canonicalized output), litegraph's `renderLink` (fixed `RIGHT`/`LEFT` axes) and Blender's
`calculate_inner_link_bezier_points` (offset built from `abs()`).

**Alternatives rejected.** (a) *Keep the bus and clamp its descent*, which `G_ux_guidance.md`
recommends at the reasoning layer by analogy to Blender's slope clamp. Rejected: the clamp treats
the symptom while the bus remains two independently-anchored cubics converging on a shared y, and
the brief names that convergence as the cusp's structural cause. A's proof shows the whole failure
mode disappears when the topology switch does. (b) *xyflow's signed offset with a `sqrt` backward
branch.* Rejected as unnecessary — it exists only because xyflow measures the offset along the
pin's own signed axis, a choice we are not making. (c) *ImNodeFlow's sign-flip past a fixed
backward gap.* Rejected and recorded as a cautionary: re-read here from `src/ImNodeFlow.inl`, it
folds inside its own 0-50px dead band, which is the very defect we are removing. (d) *Blender's
`clamp_factor` flatness correction and Rete's `max(vertical/2, |dx|)`.* Deferred, not rejected:
they are the one place the six references genuinely disagree (they pull in opposite directions),
and neither addresses the reported defect. Add one only if a screenshot shows a long near-horizontal
wire reading conspicuously flat.

**Tokens.** `SIZE.GRAPH_WIRE_BOW: float = 0.40` (new; between imnodes/litegraph's 0.25 and today's
0.45, and it now multiplies the full Euclidean distance rather than `dx` alone, so 0.40 of `dist`
is close to today's felt curvature on a typical run). `SIZE.GRAPH_WIRE_MIN_OFF: int = 24` (new;
takes over the value of the module-local `_MIN_DIRECT_DX`, which stops being read).

**Code.** `_draw_wire` loses its `bus_y` parameter and its entire second branch; `_BEZIER_BOW` and
`_MIN_DIRECT_DX` are deleted from `pass_graph.py` in favour of the tokens. In `_draw_canvas`'s
edge loop the `backward` / `bus` / `bottom` computation goes away entirely, and the call becomes
`_draw_wire(dl, xf, a, b, col, hovered, selected)`. The in-flight wire drawn in the `view.wire_drag`
block uses the same formula, so the wire being aimed and the wire once dropped are the same shape.

### G2. We decide the backward case has no rule of its own

**Rule.** There is no backward case in the code. `dx < 0` produces `cp0` right of the output and
`cp1` left of the input, which is the S-curve shape, from the same two lines as every other edge.
No `if backward:` appears anywhere in `pass_graph.py`.

**Source.** `A_wire_geometry.md` Q2: four of six references have no backward branch and are
provably fold-free; the two that branch are the two that needed to.

**Alternatives rejected.** Any explicit S-curve rule with clamps. A threshold that switches curve
*topology* is the exact failure shape twice demonstrated (ImNodeFlow's 50px band; our own bus). A
formula with no threshold has no boundary to fall into.

**Code.** Deletes the `backward = b[0] < a[0] + _MIN_DIRECT_DX` line in `_draw_canvas`.

### G3. We decide the bus is removed, and with it finding 5

**Rule.** `SIZE.GRAPH_BUS_CLEAR` and `SIZE.GRAPH_BUS_STEP` stop being read and are deleted from
`theme.py`. `_Edge.span` stops being read by the drawing code (it may stay on the dataclass only if
another decision reads it; nothing here does, so it goes too).

**Source.** `A_wire_geometry.md`'s "replace the bus entirely" and its comparison table; the brief's
own diagnosis that the maintainer's screenshot IS the bus route.

**What it changes beyond geometry.** Finding 5 in `00_findings.md` — the long `paint → raymarch`
wire that should ride a bus under the row, with no bus line visible, and `_fit` framing the nodes'
bounding box while the bus sits below it — is dissolved rather than fixed: with no bus, no wire
leaves the nodes' bounding box, so `_fit`'s box is correct by construction and there is nothing
below the fitted view to miss. The finding's row is closed as "removed with the bus", not as a
separate repair.

**Code.** `_draw_canvas`'s `bottom = max(...)` and the whole `bus` expression are deleted.

### G4. We decide wire hit testing flattens 24 segments in screen space against a floored threshold

**Rule.** One hand-rolled pass per frame over `picture.edges`, after the node/port
`invisible_button` loop has run:

```
threshold = max(float(SIZE.GRAPH_WIRE_HIT_MIN), SIZE.GRAPH_WIRE_W * 2.0 * z)   # SCREEN px
for each edge:
    p0, cp0, cp1, p3 = the same four screen points G1 draws
    bounds = bbox(p0, cp0, cp1, p3) expanded by threshold
    if mouse not in bounds: continue                       # cheap reject
    pts = [cubic(p0, cp0, cp1, p3, i/24) for i in 0..24]   # 25 points, 24 segments
    d = min(point_to_segment_distance(mouse, pts[i], pts[i+1]) for i in 0..23)
    if d <= threshold and d < best: best, view.hovered_wire = d, edge
```

Screen space, a fixed 24 subdivisions, the single nearest wire under the threshold wins.

**The coordinate-space call (A vs F).** F recommends "5.0 canvas-space units, zoom-stable by
construction, exactly like the references"; A recommends `max(6.0, GRAPH_WIRE_W * 2.0) * zoom`
screen pixels. **A's space is right and both reports' thresholds are wrong at the low end.** F's
"zoom-stable by construction" reasoning is imported from imgui-node-editor, where it is true for a
reason that does not hold here: there the *entire canvas including the stroke width* passes through
one vertex rescale, so `c_LinkSelectThickness = 5.0f // canvas pixels` (re-read here at
`imgui_node_editor.cpp:143`) tracks the drawn line at every zoom, and `TestHit` compares against
`m_Thickness + extraThickness` — the link's own drawn thickness IS part of the threshold.
ShaderBox does not rescale vertices; `_draw_wire` computes `thickness = max(1.0, GRAPH_WIRE_W * z)`
with a **1px floor**, so at zoom 0.25 the line is drawn 1px wide while F's 5 canvas units resolve to
1.25 screen px and A's formula to 1.5 — a 1.5px reach for a 1px line is not clickable by a human.
The drawn stroke has a screen-pixel floor, so the hit threshold needs one too, which is precisely
what this codebase already does for ports: `hit = min(max(GRAPH_PORT_R * zoom, GRAPH_HIT_MIN),
GRAPH_PORT_ROW * zoom / 2)` floors at `GRAPH_HIT_MIN = 7` screen px for exactly this reason
(092 D8). So the wire threshold gets the same shape, a `max` against a screen-pixel floor:

| zoom | drawn stroke (px) | F: 5 canvas units | A: `max(6, W·2)·z` | **G4: `max(6, W·2·z)`** |
|---|---|---|---|---|
| 0.25 | 1.00 | 1.25 | 1.50 | **6.00** |
| 0.5 | 1.00 | 2.50 | 3.00 | **6.00** |
| 1.0 | 1.50 | 5.00 | 6.00 | **6.00** |
| 1.5 | 2.25 | 7.50 | 9.00 | **6.00** |
| 2.5 | 3.75 | 12.50 | 15.00 | **7.50** |

6px is below the port's own 7px floor by design: a port must outrank a wire on overlap (G6), and a
threshold no larger than the port's keeps the two from competing for the same pixel. The upper end
tracks the stroke so a fat wire at high zoom stays proportionate.

**The subdivision count (24, fixed).** imgui-node-editor passes 50 explicitly at its call site;
imnodes is adaptive at `LinkLineSegmentsPerLength = 0.1` (one segment per 10px of length, re-read
here from the style ctor); ImNodeFlow inherits the library default 100; Blender uses a small fixed
count. A fixed 24 is one code path rather than two, and for ShaderBox's typical 100-400px screen-
space runs it lands inside imnodes' own adaptive 10-40 range, so it is not a looser test than the
reference that tunes hardest for cheapness.

**Source.** `A_wire_geometry.md` Q5 and its recommendation (the flatten-and-minimize algorithm
shape, shared by four references); `F_imgui_machinery.md`'s hit-testing table (the same algorithm,
plus the bounding-box pre-check); `D_E_hover_feedback.md`'s CONVERGENCE that the hit region is
always wider than the drawn stroke. The primaries are `imgui_node_editor.cpp::Link::TestHit`
(re-read: bbox-contains reject, then `ImProjectOnCubicBezier(..., 50)`, compared to
`m_Thickness + extraThickness`) and `imnodes.cpp::ResolveHoveredLink` (flatten, per-segment
`ImLineClosestPoint`, keep the minimum under `LinkHoverDistance`).

**Alternatives rejected.** (a) *One `invisible_button` per wire.* Rejected per
`F_imgui_machinery.md`: a curve has no rect, and a bounding-box button is either far too permissive
on a long shallow curve or needs per-segment buttons, which reintroduces the overlap-order problem
inside one wire. All four ImGui-family references bypass the item system for links specifically.
(b) *litegraph's single midpoint box.* Rejected: it cannot support click-anywhere-on-the-wire, which
is finding 1. (c) *Adaptive subdivision by length.* Rejected for being two behaviours where one
suffices.

**Tokens.** `SIZE.GRAPH_WIRE_HIT_MIN: int = 6` (new, screen px, the floor).
`SIZE.GRAPH_WIRE_HIT_SEGS: int = 24` (new).

**Code.** A new module-level pure helper in `pass_graph.py` — `_wire_hit(mouse, p0, cp0, cp1, p3,
threshold, segs) -> float | None` returning the distance or `None` — called from a loop in
`_draw_canvas` that writes `view.hovered_wire: tuple[str, int, str, int] | None` (the edge's
`src_key, src_slot, dst_key, dst_slot`) onto `GraphViewState`.

### G5. We decide a wire is selectable, and carries both a mid-curve ✕ and the bare Delete

**Rule.** `GraphViewState` gains `selected_wire: tuple[str, str] | None` — the `(consumer,
sampler)` the wire terminates at, which is the identity the unwire verb already takes, not a view
key that a rebuild invalidates.

- A left click (press and release with no drag past G13's lock) on a hovered wire with no node or
  port hovered sets `view.selected_wire` to that edge's `(owner, sampler)` and clears
  `view.selection` unless Shift is held; a click on empty canvas clears both. A node click clears
  `selected_wire`; selecting a wire clears the node selection. The two selections are exclusive —
  one Delete key, one unambiguous target.
- **The mid-curve ✕** draws only on the selected wire, never on a merely hovered one. Geometry: the
  cubic's point at `t = 0.5`, which for a bezier is `(p0 + 3·cp0 + 3·cp1 + p3) / 8`. A filled circle
  of radius `SIZE.GRAPH_WIRE_X_R * z` in `COLOR.BG_APP` (so the wire does not read through it), a
  1px ring in `COLOR.SELECT`, then the ✕ itself as two `add_line` calls across the circle at
  `0.5 · r` half-extent, thickness `max(1.0, SIZE.GRAPH_WIRE_W * z)`, in `COLOR.SELECT`. Two lines
  drawn with the draw list, never a font glyph — `imgui-ui/SKILL.md` §3 states this rule for exactly
  this mark ("don't rely on a font glyph centered by button text-align... Draw it with the draw
  list — two `add_line` calls for an ✕").
- The ✕ is clicked through one `invisible_button` submitted **after** every node and port button, at
  `(centre - h, centre - h)` with `h = max(SIZE.GRAPH_WIRE_X_R * z, float(SIZE.GRAPH_HIT_MIN))` —
  last in the chain, so it wins the overlap by submission order per `imgui-ui/SKILL.md` §8
  ("`set_next_item_allow_overlap()` goes on the item submitted FIRST, not the one on top"; the
  ports already declare nothing and "only forfeit the drop target", and the ✕ takes that last rung).
  Clicking it calls `App.unwire(document_id, consumer, sampler)`.
- **Bare Delete/Backspace**, read locally in `_draw_canvas` with
  `imgui.is_key_pressed(imgui.Key.delete)` (and `imgui.Key.backspace` — both names verified present
  on this build's `imgui.Key`), gated on three conditions: the
  canvas child is hovered (`hovered`, the existing `is_window_hovered(child_windows)` read), no
  imgui item is active (`not imgui.is_any_item_active()`), and not `blocked`. With
  `selected_wire` set it unwires that wire; with `view.selection` non-empty and no wire selected it
  is **not** bound (see G17 — deleting passes by key is an open question, not a decision).

**Source.** `B_mouse_controls.md` Q6's Delete row and its "changed one thing" paragraph — the bare
key read locally against hover-and-selection is what every reference that binds delete does
(`imgui_node_editor.cpp`'s `DeleteItemsAction` on `ImGuiKey_Delete` with no modifier; ImNodeFlow's
`IsKeyPressed(Delete) && !IsAnyItemActive()`; litegraph's `processKey` on 46/8), and
`commands.chord_needs_modifier` refuses a bare non-F key as a registry chord anyway, which is the
same constraint from our own side. The ✕ is the maintainer's own words in finding 1 ("by clikcing
on the cross which we should render at the center of the bezier line").

**Alternatives rejected.** (a) *Delete through the command registry.* Refused by
`chord_needs_modifier`, and every reference does it locally regardless — the fix is not to work
around the registry but to not use it. (b) *A ✕ on hover rather than on selection.* Rejected: a
mark that appears under the cursor on every wire the mouse crosses is noise on a dense canvas, and
it would need its own hit target competing with the wire's. Selection is the gate. (c) *Alt+click a
wire to delete it*, imgui-node-editor's shortcut. Rejected as a third path to one write with no
affordance; the ✕ is the discoverable one and Delete the accelerator.

**Tokens.** `SIZE.GRAPH_WIRE_X_R: int = 7` (new, canvas units, the ✕ badge's radius).

**Code.** New `_draw_wire_x` and a `selected_wire` field on `GraphViewState`; a Delete read and a
✕ button submitted last in `_draw_canvas`; the write goes through the existing `App.unwire` verb,
so the no-write gate is unaffected.

### G6. We decide the hover model: port, then node, then wire, then background — exclusive

**Rule.** Exactly one element is hovered per frame, resolved in this order, each step short-
circuiting the rest:

1. an input port dot or an output dot (the `gport_*` / `gout_*` buttons)
2. a node body (the `gnode_*` button)
3. a wire (G4's distance pass)
4. the canvas background

Implementation shape: the existing node/port `invisible_button` loop already computes
`node_hovered` and per-port hover through imgui's own item system, and its submission order already
encodes 1-over-2 (`imgui-ui/SKILL.md` §8, measured on this build). So run G4's wire pass, then
consult it as `view.hovered_wire if (not node_hovered and hovered_port is None) else None` at the
end of the frame. Nothing else changes about the existing chain.

**The cue per element, and the halo-under-stroke rule.**

| Element | Hovered | Selected | Hovered + selected |
|---|---|---|---|
| Node border | extra rect in `COLOR.GRAPH_HOVER` at `GRAPH_HOVER_HALO_ALPHA`, `GRAPH_WIRE_W * 2 * z` thick, drawn **inset** by `GRAPH_WIRE_W * z`; the existing 1px `BORDER` stroke unchanged | today's `COLOR.SELECT` border swap, **plus** the same inset halo in `COLOR.SELECT` at `GRAPH_SELECT_HALO_ALPHA` | both halos, select drawn first (outermost), hover just inside it, each at its own alpha |
| Input port dot | `port_col` swaps `FG_MUTED` → `COLOR.GRAPH_HOVER` for that one dot; **radius unchanged** | n/a (a port is not selectable) | n/a |
| Output dot | same colour swap, keyed off the `gout_*` hit | n/a | n/a |
| Wire | stroke in `COLOR.GRAPH_HOVER` at `GRAPH_WIRE_W * 1.4 * z`, over a halo (below) | stroke in `COLOR.SELECT`, over a halo, plus the mid-curve ✕ (G5) | selected wins the stroke colour; no third treatment |

**The halo-under-stroke rule.** A highlighted wire is drawn as **two strokes: first a wider, dimmer
one, then the normal-width crisp one on top** — never a single thickened line. The halo is
`GRAPH_WIRE_W * 3.0 * z` wide at the state's halo alpha; the crisp stroke is the normal width at
full alpha. This is ImNodeFlow's technique (`link_selected_outline_thickness`, outline stroke first,
normal stroke over it) and imgui-node-editor's (`c_LinkChannel_Selection`, a wider border stroke on
its own channel), independently arrived at, and it is what keeps a highlight from reading as a
thicker wire. The halo goes in its own channel *below* the wire channel (G12) so one wire's halo
never paints over a neighbouring wire's crisp stroke.

**No size change anywhere.** No dot grows on hover, no node grows, no card lifts. `imgui-ui/SKILL.md`
§3's first rule is that a highlight changes *colour, never size*, and a thicker border must be drawn
inset rather than straddling the edge; ImNodeFlow's growing `socket_hovered_radius` is explicitly
not adopted for that reason (`D_E_hover_feedback.md`'s port-dot recommendation says so).

**Source.** `D_E_hover_feedback.md` Q12-13 and its hover-priority section; the primaries are
`imnodes.cpp::ResolveHoveredPin/Node/Link` (the literal early-return chain) and
`imgui_node_editor.cpp::BuildControl` (reverse node scan, each node's pins before its body, links
consulted only `if (nullptr == hotObject)`). The halo pattern is `ImNodeFlow.h`'s style fields and
imgui-node-editor's selection channel.

**Alternatives rejected.** (a) *Hovering a node lights its wires.* Rejected: none of the four
libraries inspected does this on hover — litegraph does it on **selection** only
(`selectNodes` → `highlighted_links`), and imnodes' pin-lights-its-link is an implementation detail
for drag-to-detach, not a visual affordance (`D_E_hover_feedback.md` CONVERGENCE). (b) *A glow or
drop shadow around the node body*, litegraph's choice. Rejected: it is a retained-Canvas2D idiom,
and the border-colour swap is the majority pattern among the draw-list cousins. (c) *Hover cue
identical to selection at a lower intensity.* Rejected: every reference uses two different hues,
never one at two strengths.

**Tokens.** `COLOR.GRAPH_HOVER = _P["blue_b"]` (new). `blue_b` is already excluded from
`GROUP_TINTS` and is not an accent primary; note it currently equals `COLOR.TAG`, which is a
different surface entirely (a tag chip in the library, never drawn on the canvas), so no canvas cue
collides — but the invariant block in `theme.py` should gain `GRAPH_HOVER` alongside `GRAPH_EDGE`
in `_GROUP_TINT_EXCLUSIONS` so a future group tint cannot become the hover colour.
`COLOR.GRAPH_HOVER_HALO_ALPHA: float = 0.35` and `COLOR.GRAPH_SELECT_HALO_ALPHA: float = 0.55`
(new). No new `SIZE` token for thickness: hover and halo widths derive from `GRAPH_WIRE_W`, matching
how `GRAPH_PORT_RING_W` already derives rather than hardcoding per state.

**Code.** `_draw_node` gains `hovered: bool` and `hovered_port: int | None` parameters;
`_draw_port_dot`'s colour argument becomes per-dot rather than one `port_col` for the node;
`_draw_wire` gains the halo pass. The hover state must be computed **before** the draw or read from
the previous frame — finding 2 notes that the hit rects are submitted after the picture, so a
same-frame hover cue is a frame late. Resolution: keep last frame's hover on `GraphViewState`
(`view.hovered_node`, `view.hovered_port`, `view.hovered_wire`) and read it at draw time, writing
the fresh values in the hit-test pass for the next frame. A one-frame lag on a hover highlight at
60fps is invisible; restructuring the draw to run after the hit test is not, because the ports'
positions come from the same geometry the draw needs.

### G7. We decide the cursor changes in three states, through `App.want_cursor`

**Rule.** Requests only, into the single-owner `app.want_cursor` field, inside the existing
per-gesture branches in `_draw_canvas`; never a raw `glfw.set_cursor` call
(`imgui-ui/SKILL.md` §8's single-owner rule, and `ui.py` already applies `want_cursor` once at
end-of-frame gated on change).

| State | Cursor |
|---|---|
| panning (`panning` is True) | `app.hand_cursor` |
| dragging a node (`view.node_drag is not None`) | `app.hand_cursor` |
| dragging a wire (`view.wire_drag is not None`) | `app.crosshair_cursor` |
| over a port, a node, a wire, or empty canvas at rest | nothing requested (arrow) |

Two new cursor objects beside the existing three in `App.__init__`:
`self.hand_cursor = glfw.create_standard_cursor(glfw.HAND_CURSOR)` and
`self.crosshair_cursor = glfw.create_standard_cursor(glfw.CROSSHAIR_CURSOR)`.

**Source.** `D_E_hover_feedback.md` Q14 and its cursor section. The one cursor a reference actually
implements for a graph gesture is litegraph's `"crosshair"` while `connecting_node`; litegraph's
author left a `move` cursor for node-hover **commented out** in his own source, which is the
strongest available signal that a rich per-element cursor vocabulary is not wanted here.

**Alternatives rejected.** A cursor per hovered element (pointer over a node, crosshair over a
port, something over a wire). Rejected: no reference commits to one, and it competes with G6's
highlights for the same message while adding X11 flicker risk for no information gain.

**Tokens.** None; cursors are glfw objects on `App`, not theme tokens.

### G8. We decide the feedback loop becomes a glyph beside the run-count badge

**Rule.** `_draw_self_loop` and its call site are deleted, along with `SIZE.GRAPH_LOOP_RISE` and
`SIZE.GRAPH_LOOP_REACH`, which stop being read. In their place, in `_draw_node`, a node with any
port whose `kind == "prev"` draws a double-ring mark in the badge row at the picture's top-right:
at `(s1[0] - badge_w - GRAPH_FB_GAP * z, s0[1])` when an `xN` badge is present that frame, else at
`(s1[0], s0[1])`.

**Geometry, as draw-list primitives at the badge's size.** The mark occupies a
`SIZE.GRAPH_FB_SIZE * z` square, where `GRAPH_FB_SIZE` equals the badge height `_BADGE_H = 12.0`,
so the glyph sits flush with the badge row. Two open rings, each `r = size / 4`, centred at
`cx0 = right - size + r` and `cx1 = right - r`, both at `cy = top + size / 2`; each ring is drawn
with `path_arc_to` over roughly 300° with its gap facing the other ring, then `path_stroke` at
`max(1.0, SIZE.GRAPH_WIRE_W * z)`:

```python
dl.path_clear()
dl.path_arc_to((cx0, cy), r, 0.6, 2 * pi - 0.6, 20)
dl.path_stroke(col, imgui.ImDrawFlags_.none, thickness)
dl.path_clear()
dl.path_arc_to((cx1, cy), r, pi + 0.6, 3 * pi - 0.6, 20)
dl.path_stroke(col, imgui.ImDrawFlags_.none, thickness)
```

Colour: `badge_fg` (`COLOR.FG_MUTED` faded by the node's alpha), so the mark reads as one family
with the `xN` badge rather than a second vocabulary.

**Why two rings and not a font character.** Two reasons, both rules we already hold. First,
`imgui-ui/SKILL.md` §3 and §5: don't rely on a font glyph for a small mark — `∞` (U+221E) or `↻`
(U+21BB) in the loaded UI faces is unverified for presence, centring and legibility at 12px, and
the skill's own precedent is the crisp ✕ drawn with the draw list. Second, the double ring is
already this canvas's sign for "reads its own history": `_draw_port_dot`'s `kind == "prev"` branch
draws two concentric rings for the feedback port (092 D11). The badge mark and the port dot then
share one visual language instead of inventing a second.

**Source.** The maintainer's finding 3 verbatim ("draw a little double looped arrow (where you draw
\"xN\" at top right of the node card)"); `D_E_hover_feedback.md` Q15's convergence that
TouchDesigner and Houdini — the two references with a genuine feedback concept — both remove the
recursion from the line layer entirely (a node parameter, an interior network), plus its finding
that neither litegraph nor imnodes implements a self-loop wire at all, so there is no precedent to
copy for a loop and the glyph is the converged shape.

**Alternatives rejected.** (a) *Fix the loop's bezier so it clears the picture's corner.* Rejected:
the maintainer's objection is that the wire is there at all ("the edge is overlapped by the node, it
looks unpleasant"), and the references agree the recursion belongs off the line layer. (b) *A true
bezier figure-eight.* Rejected as harder to keep symmetric at 12px than two arcs.

**Tokens.** `SIZE.GRAPH_FB_SIZE: int = 12` and `SIZE.GRAPH_FB_GAP: int = 2` (new); delete
`GRAPH_LOOP_RISE` and `GRAPH_LOOP_REACH`.

**Code.** New `_draw_feedback_glyph` in `pass_graph.py`; `_draw_node`'s badge block gains the
branch; `_draw_canvas`'s `for node ... if port.kind == "prev": _draw_self_loop(...)` loop is
deleted.

### G9. We decide arrowheads: none

**Rule.** No arrowhead on any wire, at either end or the midpoint.

**Source.** `A_wire_geometry.md` Q4: arrowheads are opt-in and **off by default** in every
reference that has the concept — imgui-node-editor's `PinArrowSize`/`PinArrowWidth` both default to
`0`, litegraph's `render_connection_arrows` defaults `false`, xyflow's SVG markers are opt-in;
imnodes has no arrowhead code at all (a repo-wide grep for "arrow" returns nothing). A plain
undecorated curve is the converged default look.

**Alternatives rejected.** A midpoint arrowhead (litegraph's placement when enabled). Rejected:
direction is already legible from this canvas's own filled-disc-versus-ring port-dot distinction
(092 D11, `_draw_port_dot`), and the mid-curve slot is now the selected wire's ✕ (G5) — an
arrowhead there would collide with the one mark that has to be unmistakable.

**Code.** No change; this decision records that none is added.

### G10. We decide the wire's endpoints attach at the dot centres, with no stub

**Rule.** The curve starts exactly at `_out_point(...)` and ends exactly at `_port_point(...)`, as
today. No radial offset, no straight stub.

**Source.** `A_wire_geometry.md` Q4: every reference attaches at the bare endpoint by default;
imgui-node-editor offsets by `radius + arrowSize` only when an arrow is configured, which G9 says it
is not.

**Code.** No change.

### G11. We decide the card is 128 wide, with an ellipsis rule on the name and every port label

**Rule and tokens.**

| Token | Current | Decided | Why |
|---|---|---|---|
| `GRAPH_NODE_W` | 108 | **128** | See the fit arithmetic below |
| `GRAPH_THUMB` | 80 | **96** | Grows with the card, stays centred; leaves 8px of card either side at `GRAPH_PAD = 8`, proportionally what today's 80-in-108 gives |
| `GRAPH_PAD` | 6 | **8** | imnodes' and imgui-node-editor's converged padding exactly (`C_node_card.md` CONVERGENCE) |
| `GRAPH_NAME_H` | 18 | **20** | The same ~2px slack over a 14px bold face that 18 gives today, at the new card's scale |
| `GRAPH_PORT_ROW` | 16 | **18** | Still under Blender's `NODE_DY` (20px widget unit) and litegraph's `NODE_SLOT_HEIGHT` (20) |
| `GRAPH_PORT_TOP` | 4 | 4 | No reference argues for a change |
| `GRAPH_PORT_R` | 4 | 4 | Already the converged 4-4.67px radius; the complaint was never the dot |
| `GRAPH_ROUNDING` | 6 | 6 | Inside the converged 3-12px band |
| `GRAPH_GAP_X` | 64 | 64 | Unchanged; nothing argues to move it |

Node height at zoom 1 becomes `8 + 96 + 20 + 8 = 132` with no ports, `+ 4 + n×18` per port row.

**The ellipsis rule.** Both the name and every port label go through
`shaderbox.ui_primitives._ellipsize`, the app's existing binary-search-the-longest-prefix helper
that `preview_cell`'s footer already uses — the same fix, not a new one.

- Name budget: `(p1.x - p0.x) - 2 * GRAPH_PAD * z`, measured in the pushed `font_14_bold` at
  `legacy_size * z`, i.e. computed in the same pushed-font scope as the `calc_text_size` that
  positions it.
- Port-label budget: `node_width * z - (2 * GRAPH_PORT_R * z + 2 * z) - GRAPH_PAD * z`, matching
  where `_draw_node` already places the label at `center[0] + 2*r + 2*z`.

**The width call (128, against C's 136 and the fit numbers).** C recommends 136. The fork is real
and the arithmetic decides it. The maintainer's correction stands: `distance_field` and
`u_distance_field` are **real names from his own document**, not mock data (C's report treats them
as illustrative, taken from `00_mock_panel.html` — but that mock was built from his screenshot, so
they are content, and the ellipsis rule is load-bearing rather than a guard against a hypothetical).
At the shipped font's measured advance (0.545898 × em, monospace):

| Name | 14px bold | 12px regular |
|---|---|---|
| `distance_field` | 107.0 | — |
| `u_distance_field` | — | 104.8 |
| `composite` | 68.8 | — |
| `u_cascade` | — | 59.0 |

| `GRAPH_NODE_W` | name budget (14b) | port-label budget (12) | `distance_field` fits | `u_distance_field` fits |
|---|---|---|---|---|
| 108 (today) | 92 | 90 | no | no |
| **128** | **112** | **110** | **yes (107.0)** | **yes (104.8)** |
| 136 | 120 | 118 | yes | yes |
| 150 | 134 | 132 | yes | yes |

128 is the smallest of the candidates where both of the maintainer's own longest strings fit
uncut, which is the actual requirement he stated ("the uniform names and the node names don't fit
fully"). Everything above 128 buys headroom for strings nobody has written, at a real cost:

**Fit-to-view for a six-column chain** (`6 × W + 5 × GAP_X`, plus `2 × _FIT_MARGIN` where
`_FIT_MARGIN = SPACE.LG = 16`, which `_fit` adds on each side — C's table omitted it, so C's
numbers run ~1.5% optimistic and are recomputed here):

| `GRAPH_NODE_W` | fitted width | fit zoom, 740px pane | fit zoom, 1225px pane |
|---|---|---|---|
| 108 (today) | 1000 | 0.740 | 1.225 |
| **128** | **1120** | **0.661** | **1.094** |
| 136 | 1168 | 0.634 | 1.049 |
| 150 | 1252 | 0.591 | 0.978 |

At 128 the wide pane still fits the whole chain above zoom 1 with the most headroom of any
candidate that solves the stated problem, and the narrow pane loses the least. 150 is the point
where the wide pane drops below 1.0, so it is out on its own numbers; 136 clears both bars too and
is a legitimate taste call, which is why it goes to the maintainer as G-Q1 rather than being
decided here on a difference no argument settles.

**Source.** `C_node_card.md` Q9-Q11 and its recommended table (the token values, the padding
convergence, the `_ellipsize` precedent, the font measurement); `G_ux_guidance.md` area C for the
reasoning layer — Houdini's explicit "Shorten Long Node Names" + "Maximum Node Name Width" policy
is the one primary source that documents overflow as a first-class decision, and no source anywhere
defends silent unguarded clipping.

**Alternatives rejected.** (a) *True auto-width to the longest label*, which every Dear-ImGui
reference does. Rejected on the substrate argument: those get it nearly free from
`BeginGroup`/`EndGroup`, and `_draw_node` has no layout pass — auto-width means a `calc_text_size`
sweep over the name and every port label of every node before the draw, under a pushed font at the
current zoom, which is a separately-scoped change and would also make `node_size` (shared with
`rank_layout` and Arrange) depend on fonts. (b) *xyflow's wrap-instead-of-clip.* Rejected: a wrapped
port label breaks the one-row-per-port invariant that `_port_point` and `node_size` both compute
from. (c) *No ellipsis, rely on the wider card.* Rejected by the maintainer's own correction: the
long names are real, and a future or user-authored pass can always exceed any fixed budget — today
`_draw_node` calls `dl.add_text` with no width check and draws straight past the card edge onto the
canvas.

**Code.** `theme.py`'s `SIZE` block; `node_size` in `graph_state.py` picks the new values up
unchanged; `_draw_node`'s name draw and its port-label draw each gain an `_ellipsize` call;
`_thumb_rect` and `_port_point` need no edit (they already read the tokens).

### G12. We decide the channel layout is five, and z-order brings selected and dragged to front

**Rule.** `dl.channels_split(5)` once per canvas frame, replacing today's `channels_split(2)`:

| Channel | Content |
|---|---|
| 0 | wire halos (G6's wider dimmer stroke) |
| 1 | wires (the crisp stroke), and the selected wire's ✕ |
| 2 | nodes — everything `_draw_node` emits |
| 3 | the in-flight wire being dragged, so it is never occluded while being aimed at a port |
| 4 | the rubber band and the snap guides, always on top |

`channels_merge()` stays one call at the end. The existing `channels_set_current(0)` and `(1)` call
sites keep their meaning (wires, then nodes) and shift to `(1)` and `(2)`.

**Why this is safe.** `F_imgui_machinery.md`'s headless probe settles it directly: with three rects
submitted in a fixed order and routed to channels in ascending then descending order across frames,
**paint order follows channel index, not call order**, stable across frames and across a 5-channel
split. The probe also records the trap — reading `vtx_buffer` shows submission order in both frames
and looks like a null result, because `ImDrawListSplitter::Merge` reorders `CmdBuffer`/`IdxBuffer`
and never touches `VtxBuffer`; the index buffer is what answers the question.

**Bring-to-front.** Reorder `nodes = list(picture.nodes.values())` once per frame before the draw
loop, sorted ascending by `(is_selected, is_being_dragged)`, so a selected or dragged node draws
last within channel 2 and occludes its neighbours. **Not on hover** — merely passing the cursor
over a dense cluster must not reshuffle paint order, which matches imgui-node-editor (selection
changes colour and channel, never `m_Nodes` order; no `BringToFront` exists in that codebase).
`picture.nodes` is rebuilt every frame by `_build_view`, so there is no persistent order to
maintain and no per-node channel-pair swap needed — imnodes' `DrawListSortChannelsByDepth` machinery
exists because imnodes assigns channels per node and must keep them consistent; we redraw from
scratch.

**Source.** `F_imgui_machinery.md`'s channel-layout recommendation, its layering table (the primary
being `imgui_node_editor.cpp`'s named channel constants and `imnodes.cpp`'s
`DrawListGrowChannels`/`NodeDepthOrder`), and its own probe; `D_E_hover_feedback.md`'s note that
Blender draws selected links in a second pass on top (`"/* Draw selected node links after the
unselected ones, so they are shown on top. */"`).

**Alternatives rejected.** (a) *A channel pair per node*, imnodes' scheme. Rejected: it buys
independent background/foreground layering between overlapping nodes, which this canvas does not
need (a node is one opaque card), at the cost of an O(n²) per-frame reorder. (b) *Reorder on hover
as well as selection.* Rejected as flicker.

**Tokens.** None.

### G13. We decide the click-versus-drag threshold is 4 screen pixels, named, read at every site

**Rule.** `SIZE.GRAPH_DRAG_LOCK_PX: float = 4.0`, screen pixels, **not** zoom-scaled, passed
explicitly as `lock_threshold` to every `imgui.is_mouse_dragging(imgui.MouseButton_.left, ...)`
call in `_draw_canvas`. There are four such sites today — the rubber band, the node-body drag, the
port-press-becomes-node-drag, and the output-dot wire drag — and all four currently inherit imgui's
global 6px default, which is tuned for buttons and text selection.

**The value (4, against F's 3).** B recommends 4px, F recommends 3px. Both reason from the same
evidence — imgui's 6px default is too coarse, the references' 1px (imgui-node-editor's literal
`IsMouseDragging(button, 1)` at every call site) and 0px (imnodes' `IsMouseDragging(0, 0.0f)`) are
too eager for a mouse app — and land one pixel apart with no measurement separating them. **4 is
the call**, for the reason B gives and F does not weigh: this canvas's click semantics are
load-bearing in a way the references' are not. A click on a node runs `pick_pass`, which changes
which pass the *whole app* is editing; a click that is misread as a 2px drag instead starts a
`NodeDrag` and writes a position on release. Both references that debounce tighter than imgui also
hand-roll their own hit-testing and gain nothing from imgui's native click/drag disambiguation,
which `pass_graph.py` does benefit from (`F_imgui_machinery.md` says this itself while still
recommending 3). Between two unmeasured neighbours, take the one that errs toward "this was a
click", since a wrong click is inert and a wrong drag writes. Neither report claims a measurement;
if the maintainer's hand disagrees, it is one token.

**Source.** `B_mouse_controls.md` Q7's threshold row and its recommendation;
`F_imgui_machinery.md`'s click-vs-drag table. Primaries: `imgui_node_editor.cpp`'s literal `1`
lock_threshold at `DragAction::Accept`/`SelectAction`/`SizeAction`/`CreateItemAction`;
`imnodes.cpp`'s `IsMouseDragging(0, 0.0f)`; xyflow's `XYDrag.ts` comment that the threshold is
measured in client pixels "for consistent drag threshold behavior across zoom levels".

**Alternatives rejected.** (a) *Leave the global 6px default.* Rejected: it is a named constant we
never chose, and the maintainer asked about the control scheme specifically. (b) *Scale it with
zoom.* Rejected — every reference agrees the threshold is screen-space and zoom-independent, xyflow
says so in a source comment.

**Tokens.** `SIZE.GRAPH_DRAG_LOCK_PX: float = 4.0` (new).

### G14. We decide the pan/zoom/select bindings are kept, and only Delete is added

**Kept, unchanged:**

| Gesture | Binding |
|---|---|
| Pan | middle-drag, or Alt + left-drag |
| Zoom | wheel about the cursor, `zoom *= 1.1 ** wheel`, clamped `[0.25, 2.5]` |
| Select | left click (clears unless Shift) |
| Rubber band | left-drag from empty canvas that is not a pan |
| Node drag | left-drag on a node body, or on a port with no wire to grab |
| Wire from an output | left-drag from an output dot |
| Re-plug | plain left press on a filled input re-grabs at the producer end, no modifier |
| Drop on empty | cancels a fresh drag, disconnects a re-grabbed one |
| Context menu | right-click release, node menu or canvas menu |
| Fit / Arrange | canvas menu items, no keybinding |

**Changed: one thing.** The bare `Delete` / `Backspace` accelerator of G5, read locally and gated
on canvas-hover + no-active-item + a selected wire.

**Source.** `B_mouse_controls.md`'s recommendation table and its "kept nearly everything"
paragraph: every current binding already matches the convergent shape across all eight references.
`G_ux_guidance.md` area B names the one real fork — xyflow's map-convention (drag pans) versus the
design-tool convention (drag selects, pan on a modifier) — and records that ShaderBox is already
squarely in the design-tool camp, so a redesign must not "fix" it into the other by accident. The
maintainer's own doubt about the controls is answered: the bindings are not the problem.

**Alternatives rejected.** (a) *ImNodeFlow's header-only node drag.* Rejected: our cards have no
header region and adding one is a layout change. (b) *Unreal's filtered create-node menu on a drop
into empty space.* Rejected: it needs a searchable palette keyed by pin type, far larger than this
feature, and none of the imgui-family references do it. (c) *Keys for Fit and Arrange.* Deferred —
Houdini's `H` and Unreal's `Home` are real precedents, but both come from editors with a far denser
hotkey surface; wait for the maintainer to ask. (d) *Escape cancels a live band/wire/node drag.*
Not decided here: no reference requires it, and if wanted it is a `_draw_canvas`-local read using
the existing `press_blocked` machinery, never a rung in the global Escape ladder.

### G15. We decide there is no routing around nodes

**Rule.** Wires draw straight from pin to pin and pass under node cards by channel order (G12).
No obstacle avoidance, no bundling, no waypoints computed by the app.

**Source.** `A_wire_geometry.md` Q3: none of the six code references does automatic obstacle
avoidance, and neither documented commercial editor does; the universal answer is "draw the bezier,
rely on z-order, let the user move nodes". imgui-node-editor's PR #119 exists precisely because the
plain bezier draws through node bodies, and its fix is a 16-waypoint routed path — shipped as a PR,
not as the library's default. `G_ux_guidance.md` adds the one exception: Max/MSP's "Route Patcher
Cords" is a user-invoked one-shot command, not continuous avoidance.

**Alternatives rejected.** (a) *Edge bundling*, Holten's technique. Rejected on the paper's own
stated preconditions, read in full: bundling needs a compound graph with an underlying tree
hierarchy plus a separate adjacency layer, and pays off with large numbers of adjacency edges. A
pass graph is a small flat DAG with a handful of edges. (b) *xyflow's `smoothstep` orthogonal-ish
path.* Rejected: it is a curve-family choice keyed only on the endpoints' facing, not on other
nodes, and its own source comment concedes "it's not as good as a real orthogonal edge routing".

### G16. We decide reroute/waypoint nodes are not now, with a trigger

**Rule.** No reroute node, no waypoint, no dot. **Trigger to revisit:** the maintainer reports a
wire he cannot read on a real document *after* G1-G3 land — that is, after the bus and the cusp are
gone. Until then the complaint on record is about curve quality, and the curve is being fixed.

**Source.** `G_ux_guidance.md` area B: the reroute/dot/straighten family is the strongest
convergence in the whole research pass — Blender's reroute node, Houdini's pinned and unpinned dots,
Godot's `VisualShaderNodeReroute`, Unreal's double-click-a-wire, Max/MSP's segmented cords — and
every source treats a manual waypoint as **complementary to** the curve math, never a substitute.
`A_wire_geometry.md` Q3 confirms the same from the code side, and names the cost: a new node kind
with its own port, its own persistence and its own layout participation.

**Why not now anyway.** Two reasons. It is a new entity in a data model that 092 settled and this
feature does not reopen (a reroute is a node with a position, a pass-shaped thing that is not a
pass), and `NO backward-compatibility / migration code` means the `graph.json` reshape it implies
is hand-fixed in `projects/dev/` — worth doing for a feature the maintainer has asked for, not for
one the research inferred. And the research's own convergence is that a waypoint is the answer for
*genuinely bad geometry*; we are about to remove the only source of genuinely bad geometry on this
canvas. Deciding it now would be deciding it against a defect that no longer exists.

### G17. We decide Delete does not delete passes from the canvas

**Rule.** The bare Delete key deletes a **wire** (G5) and nothing else. Deleting a pass stays where
it is: the node's context menu, through `pass_menu_items`, with the strip's two-click arm and its
`len(document.passes) > 1` gate (092 D10, D16).

**Source.** `B_mouse_controls.md`'s Delete recommendation routes the key through the same verb the
menu uses; it does not argue for extending the key to passes. 092 D16 is explicit that deleting a
pass is an armed, two-click gesture because it is N writes with no undo.

**Alternatives rejected.** Binding Delete to the node selection as well. Rejected: it would make one
bare keystroke destroy several passes with no arm, which is the opposite of the deliberate friction
092 chose. Whether the node menu should *also* carry a Delete beside the wire's is G-Q2.

### G18. We decide `_draw_wire`'s signature carries the state, not a colour the caller pre-picked

**Rule.** `_draw_wire(dl, xf, a, b, col, halo_col, halo_alpha)` — the caller resolves the colour
precedence and passes both strokes. The precedence, unchanged in spirit from today's node-border
chain: **error, then selected, then hovered, then dim, then normal.** A cycle-error wire stays
`STATE_ERROR` even while hovered, exactly as `node.error` already outranks `selected` in
`_draw_node`'s border chain.

**Source.** `D_E_hover_feedback.md`'s wire recommendation ("a cycle-error wire keeps priority over
hover, same precedence logic as the node border's error-first chain") and the existing chain in
`_draw_node`.

**Tokens.** None beyond G6's.

---

## 3. What stays as it is

A reviewer should be able to see the redesign's edge. None of the following is touched:

- **The data model.** `PassEntry.position`, `graph.json`, `Document.effective_wiring()` as the one
  source of edges, `node_ports` as the one source of ports (092 D1, D6). No new persisted field, no
  `version` bump, no migration — and per the project's hard rule, if anything had reshaped the
  format the `projects/dev/` files would be hand-fixed in the same wave rather than migrated.
- **The box and ghost model.** One box per group at the root with boundary-edge ports, a tab per
  group with outside passes as ghosts, the bundle output, the hollow unread output dot (092 D4, D5).
  Settled in 092 and explicitly out of review per the brief.
- **Every gesture through an `App` verb, and the no-write gate.** `drop_wire`, `unwire`,
  `commit_node_drag`, `group_selection`, `dissolve_group`, `arrange_graph`, `pick_pass`. The new
  writes this feature adds — the ✕ and the Delete key — both go through the existing `App.unwire`,
  so `test_the_widget_makes_no_session_write_of_its_own` keeps passing unchanged (092 D10, D12).
- **Positions.** Written only by a placement: a drag's release through `set_pass_positions`, or
  Arrange. `rank_layout` for a pass never placed. `update()` returns nothing to write; `commit()` is
  the only thing that does (092 D6, D13). The redesign adds no new writer.
- **The compile seam.** `compile_pending_passes(document)` called per canvas frame, the same seam
  091 uses, so a never-compiled pass still gets its ports (092 D1).
- **The Arrange layout.** `rank_layout` and its rank/tiebreak rules, unchanged. It reads
  `node_sizes`, which picks up G11's new width automatically, and nothing else about it moves.
- **Node/port hit testing through imgui's item system.** The `invisible_button` +
  `set_next_item_allow_overlap` chain stays exactly as documented in the module docstring and
  `imgui-ui/SKILL.md` §8. Only the wire gets a hand-rolled pass, which is what all four ImGui-family
  references do (`F_imgui_machinery.md` DIVERGENCE).
- **Text at zoom.** `push_font(font, max(4.0, font.legacy_size * z))` stays. It is already the
  crisp-text strategy ImNodeFlow builds a nested `ImGuiContext` to reach, and it avoids
  imgui-node-editor's bitmap-scaled blur. F confirms rather than replaces it.
- **The port-dot shape language.** Filled disc, hollow ring, ring-with-core, double ring, square
  (092 D11, D15). `GRAPH_PORT_R = 4` already matches the converged reference radius.

---

## 4. Open questions for the maintainer

Only genuine forks. Everything the research could settle is settled above.

**G-Q1. Card width: 128 or 136?** Both clear the bar; the research cannot separate them.
- **128 (decided above, G11).** The smallest width where `distance_field` (107.0px at 14 bold) and
  `u_distance_field` (104.8px at 12) both fit uncut. Six-column fit: 0.661 at 740px, 1.094 at
  1225px.
- **136 (C's recommendation).** ~12px more headroom per label for names nobody has written yet.
  Fit: 0.634 and 1.049.
- *Recommendation: 128.* It solves the stated problem with the most fit-to-view headroom left over,
  and 136's extra width buys only hypothetical strings. One token either way — worth eyeballing once
  it renders.

**G-Q2. Should the node menu also carry Delete for a wire?** The ✕ is the wire's discoverable
affordance and Delete its accelerator; neither is in a menu.
- **No menu entry (assumed above).** The ✕ is already discoverable and the wire has no menu of its
  own today — adding one means a right-click hit test on a curve, a fourth rung in the overlap chain.
- **A "Disconnect" item on the *consumer node's* menu**, listing its wired samplers. Discoverable
  from the keyboard-free path, no new hit test, but a submenu on a menu that is already four items.
- *Recommendation: no menu entry for now.* Revisit if the ✕ turns out to be hard to hit at low zoom
  — which G4's 6px screen floor is specifically designed to prevent.

**G-Q3. Click-versus-drag: 4px or 3px?** G13 decides 4 and says why, but neither report measured it
and it is a feel question your hand answers in seconds.
- **4px (B, decided).** Errs toward "that was a click", so a click that runs `pick_pass` is not
  misread as a 2px drag that writes a position.
- **3px (F).** Slightly more responsive to a deliberate small nudge.
- *Recommendation: ship 4, try 3 in the same sitting.* It is one token read at four call sites.

**G-Q4. Does the hover halo want the blue, or a desaturated neutral?** G6 picks `blue_b` because it
is theme-legal and distinct from `SELECT` (purple) and every `STATE_*`. But `blue_b` is also
`COLOR.TAG`, on a different surface. If the canvas ends up reading as too colourful with a blue
hover, a neutral near `FG_SECONDARY` at the same alphas is the fallback and needs no other change.
- *Recommendation: ship blue.* Every reference uses a distinct hue for hover rather than a
  brightness step, and a neutral halo against a grey wire is the one combination most likely to read
  as nothing at all.

---

## 5. False trails, consolidated

So the spec round does not re-walk them.

**Sources that looked right and were not:**
- **`retejs/connection-plugin`** carries no curve geometry — `presets/classic.ts` and `flow/*` are
  interaction state machines. The path formula is in the separate `retejs/render-utils`
  (`classicConnectionPath`). (A)
- **Unreal's documented S-curve/tangent formula does not exist.** Epic publishes no implementable
  curve rule; the engine's Slate source (`SGraphNode`/`SGraphPin`) needs an Epic-linked GitHub
  account to clone. Every Unreal claim in this research is behavioural, from docs. (A, C)
- **Blender's `node_link_bezier_handles`/`node_link_bezier_points` are not in `node_draw.cc`.** The
  handle math is `calculate_inner_link_bezier_points` in `drawnode.cc`. (A)
- **Blender's `is_highlighted` / `NODE_LINK_DIM` / `node_link_draw_data`** were not found verbatim
  in the current `node_draw.cc` — older names, or a file not fetched. What *is* verified there is
  the two-pass "selected links drawn on top" comment. (D/E)
- **Bret Victor's "Up and Down the Ladder of Abstraction"**, read in full: about interactive
  parametric sliders, not node UIs. No Victor essay on node-graph UX was found. (G)
- **Holten's hierarchical edge bundling**, read in full: its own preconditions (a compound graph
  with a tree hierarchy plus an adjacency layer; many adjacency edges) do not hold for a small flat
  DAG. Named in the brief as worth citing "if only to say it is not for us" — confirmed. (G)
- **Purchase's crossing-minimization result** is real and correctly cited, but answers a *layout*
  question (how to arrange nodes) not a *rendering* one. ShaderBox's layout is manual. Adjacent, not
  actionable. (G)
- **imgui-node-editor's Python binding as a source of machinery.** The `.pyi` exposes only the
  high-level node/pin/link API plus read-back; `FindLinkAt`, `BringToFront` and the channel
  constants are unreachable from Python. Adopting it means replacing the whole model, not borrowing
  one mechanism. (F)
- **ocornut's `imgui_demo.cpp` "Custom rendering" canvas**: its entire technique is already what
  `_draw_canvas`'s background/pan/band code does, function for function. (F)

**Premises in the brief that did not survive contact with the code:**
- **"litegraph/imnodes allow a link from a node to itself, drawn as a loop."** Neither implements
  one; no code path special-cases `origin_id == target_id`. Do not cite either as precedent for a
  loop glyph. (D/E)
- **"Blender's backward-link formula is direction-blind, therefore it folds."** Plausible-sounding
  (an unconditional `abs()` reads like a missing branch) and disproved by working the algebra: a
  non-negative offset on fixed axes cannot cross. Worth flagging because it is the exact shape of
  reasoning that reads as correct without the arithmetic. (A)
- **ImNodeFlow's `smart_bezier` as a pattern to adapt.** Read in full (and re-read here), it is the
  closest analogue to our own bug, not a fix to copy. Cited as a cautionary. (A)
- **imnodes' author's own blog describing a recursive hierarchical subdivision for link hit
  testing** does not match the shipped code, which is a single-pass per-segment scan. The blog is
  intent; the code is ground truth — a small instance of "a relayed source is not the source"
  applying even to an author writing about his own library. (G)

**Access failures, recorded rather than filled in from memory:** `docs.blender.org` 403s the
WebFetch tool but serves `curl` cleanly (use `curl`); `docs.unrealengine.com` 403s both, and the
Wayback snapshot of the pre-migration cheat sheet is what matched the live client-rendered page;
Adobe's helpx `interface-overview.html` and "Graph Creation Etiquette" 403 consistently, hit
independently by two researchers; `puredata.info` serves a bot challenge; Xu et al.'s curved-edge
user study is paywalled everywhere and its finding is a secondary paraphrase, not a verified quote.

---

## 6. Verification sketch

Per decision that states a guarantee: the falsifier, and who runs it. The repo's frame-driven
pattern is `tests/test_graph_view.py::_frames` plus `io.add_mouse_pos_event` /
`io.add_mouse_button_event` against `view.port_rects` — real imgui frames, synthetic input, no
display needed for the parts that do not read pixels.

| Decision | Guarantee | Falsifier | Kind |
|---|---|---|---|
| G1 | The curve never cusps, for any endpoints | Pure: over a grid of `(dx, dy)` covering backward, forward, near-zero and long runs, plus `dist = 0`, assert `cp0.x > cp1.x` whenever `dx < 0` and assert the offset is non-negative for every input. The claim is algebraic, so the test is an exhaustive-ish sweep of a pure function, not a rendering check. Extract the control-point computation as a pure `_wire_points(a, b, z) -> (p0, cp0, cp1, p3)` so it is callable without a draw list. | pure |
| G1 | ...including at the old bus boundary | Pure: sweep `dx` across `±_MIN_DIRECT_DX` in 1px steps and assert the four control points are continuous — no jump anywhere, which is what a topology switch would produce. This is the test that would have caught ImNodeFlow's 50px band. | pure |
| G3 | No wire leaves the nodes' bounding box, so `_fit` frames everything | Pure: for a built `_View`, assert every wire's four control points lie inside `_bbox(nodes)` expanded by the bow. Closes finding 5 by construction rather than by inspection. | pure |
| G4 | A click within the threshold of a wire hits it; beyond it does not | Pure: `_wire_hit` against a known cubic — a point on the curve returns ~0, a point at `threshold - 1` returns a hit, at `threshold + 1` returns `None`. Plus the floor: assert the threshold at `zoom = 0.25` is still 6.0, which is the whole point of the A-vs-F resolution. | pure |
| G4 | ...and the 24-segment flattening does not miss a real hit | Pure: mutation-style — walk 200 points along the true cubic, assert every one reports a distance under the threshold. A subdivision count too low shows up as a point mid-segment reading over. | pure |
| G5 | Selecting a wire and pressing Delete unwires exactly that sampler | Frame-driven: draw frames, read the wire's mid-curve point from the view state, click it, assert `view.selected_wire`, send Delete, assert the session saw one `set_sampler_source` to `NoSource` on that `(consumer, sampler)` and nothing else. Mirrors `test_a_wire_dropped_on_a_drawn_port_writes_that_port`. Needs `view.wire_rects` or the ✕ centre exposed on the view state the way `port_rects` already is — add it for the same stated reason ("so a headless test can aim a drop where a user would"). | frame-driven |
| G5 | Delete does not fire while a text field has focus | Frame-driven: open the group-name popup, type, send Delete, assert no write. The `not is_any_item_active()` gate is the thing under test. | frame-driven |
| G6 | Exactly one element is hovered, and the order is port > node > wire | Frame-driven: park the mouse on a port that a wire passes near and assert `hovered_port` is set while `hovered_wire` is `None`; move onto the node body away from any port and assert node-not-wire; move onto a wire in open canvas and assert wire. Three positions, one assertion each — this is the "one reason" test, and it is also the mutation test for the early-exit chain (break the chain's order and the first case flips). | frame-driven |
| G6 | Nothing changes size on hover | Pure: assert `node_size` and the port hit extent are functions of `(port_count, box, zoom)` alone — no hover parameter reaches them. Cheap, and it is the structural version of `imgui-ui` §3's rule. | pure |
| G6 | The halo never covers the crisp stroke | Maintainer's eyes. Channel order is asserted mechanically by G12's test; whether the result reads right at 0.25 and 2.5 zoom is a look. |
| G7 | The cursor is requested once per frame and only by the topmost surface | Frame-driven: assert `app.want_cursor` is the hand while panning and `None` at rest. The X11 flicker itself is the maintainer's eyes; the single-owner discipline is testable. | frame-driven |
| G8 | The feedback glyph replaces the loop and sits in the badge row | Maintainer's eyes for the shape at 12px. Testable part: assert `GRAPH_LOOP_RISE`/`GRAPH_LOOP_REACH` are gone from `theme.py` and `_draw_self_loop` from `pass_graph.py` — a grep-style guard that the removal actually happened rather than the new glyph being added beside the old loop. | pure + eyes |
| G11 | A name longer than the budget ellipsizes and never draws past the card | Pure: `_ellipsize("u_distance_field", budget)` at the 108 budget returns something ending `...` and measuring under budget; at the 128 budget returns the string unchanged. Both directions matter — the second is the one that proves the width decision, and it is stated against the maintainer's own two real names. | pure |
| G11 | Fit-to-view still frames a six-column chain above zoom 1 in a wide pane | Pure: build six chained passes, call `_fit` with a 1225×N avail, assert `view.zoom > 1.0`. This is the number the width decision turned on, so it earns a pin. | pure |
| G12 | Paint order follows channel index | Already established by F's headless probe against this exact imgui-bundle build; the repo's own guard is the weaker but sufficient one: assert the five `channels_set_current` calls happen in the documented order and that `channels_merge` is called once. Re-run F's probe on an imgui-bundle bump. | pure + probe |
| G12 | A selected node draws over its neighbours | Frame-driven for the ordering (assert the sorted node list puts the selected one last); the maintainer's eyes for whether it looks right when two cards overlap. | frame-driven + eyes |
| G13 | A 3px mouse move is a click, a 5px move is a drag | Frame-driven: press on a node, move 3px, release — assert `pick_pass` ran and `set_pass_positions` did not. Repeat with 5px and assert the opposite. This is the falsifier that makes the threshold a gate rather than a number in a file; it fails if `lock_threshold` is omitted at any of the four call sites, which is the actual defect it guards. | frame-driven |
| G14 | Every kept binding still works after the rewrite | The existing `tests/test_graph_view.py` suite is the falsifier — it was written against these bindings and must pass unchanged. A binding test that needs editing to stay green is the signal that something was changed silently. | frame-driven |
| G18 | An error wire stays red while hovered | Pure: the colour-precedence resolution extracted as a pure function over `(on_cycle, selected, hovered, dim)`, asserted over all sixteen combinations. A checker that quietly narrows its domain is the expensive family; enumerate the four booleans rather than spot-check three cases. | pure |

**One note on the gates themselves.** Three of the rows above are new *gates*, not new tests: G13's
threshold, G6's hover order, and G4's zoom floor each exist to prevent a regression, and a gate that
has never been broken is a wish. Each should be landed by first breaking the guarded thing —
omitting `lock_threshold` at one of the four sites, reversing two rungs of the hover chain, dropping
the `max(6.0, ...)` floor — watching the named check fail, then restoring it. The commit says which
break was tried.
