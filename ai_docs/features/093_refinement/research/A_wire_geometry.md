# Area A — wire geometry and routing

Research for feature 093 (the tenth walk on the graph editor), answering questions 1-5 of
`ai_docs/features/093_refinement/02_research_brief.md`. Every claim below is checked
against source code fetched into a scratchpad clone or single-file fetch, never against a README,
and every disputed claim below was independently re-derived with the actual arithmetic before being
stated as fact (see the note after Q2).

## Sources

| Name | What it is | URL | What was read |
|---|---|---|---|
| thedmd/imgui-node-editor | Dear ImGui node-editor extension, draw-list based (the canonical reference for a hand-drawn node editor) | github.com/thedmd/imgui-node-editor | Code: `imgui_node_editor.cpp`, `imgui_node_editor.h`, `imgui_node_editor_internal.h`, `imgui_bezier_math.h`/`.inl` |
| Nelarius/imnodes | Minimal Dear ImGui node-editor widget | github.com/Nelarius/imnodes | Code: `imnodes.cpp` |
| Fattorino/ImNodeFlow | Dear ImGui node-editor, "smart bezier" links | github.com/Fattorino/ImNodeFlow | Code: `src/ImNodeFlow.inl` (`smart_bezier`, `smart_bezier_collider`), `include/ImNodeFlow.h` (vendors its own copy of imgui-node-editor's `imgui_bezier_math.inl`, confirmed byte-identical in structure) |
| xyflow/xyflow | React Flow / Svelte Flow core, SVG-rendered flow-diagram library | github.com/xyflow/xyflow | Code: `packages/system/src/utils/edges/bezier-edge.ts`, `smoothstep-edge.ts`, `straight-edge.ts`; `packages/react/src/components/Edges/BaseEdge.tsx` |
| jagenjo/litegraph.js | Canvas2D node-editor library (used by ComfyUI and others) | github.com/jagenjo/litegraph.js | Code: `src/litegraph.js` (`LGraphCanvas.prototype.renderLink`, the mouse-move link-picking loop) |
| retejs/render-utils | Rete.js's render-agnostic connection-path helper package | github.com/retejs/render-utils | Code: `src/connection.ts` (`classicConnectionPath`, `loopConnectionPath`) |
| retejs/connection-plugin | Rete.js's connection/interaction plugin | github.com/retejs/connection-plugin | Code: `src/presets/classic.ts`, `src/flow/*` — false trail, no curve-geometry code here (see below) |
| blender/blender | Blender's node editor (shader/geometry/compositor) | github.com/blender/blender | Code: `source/blender/editors/space_node/drawnode.cc` (fetched individually via the GitHub raw API — the repo is too large to shallow-clone usefully for two files) |

Unreal Engine's material graph is closed source; a web search of Epic's official docs
(dev.epicgames.com) for an S-curve/tangent formula returned nothing — no formula is documented
publicly. What Epic's own docs do say (*Organizing a Material Graph in Unreal Engine*, UE 5.8
docs): "Reroute nodes provide a way to modify the path of the wire between two Material expression
nodes." That is Epic's documented answer to Q3 (routing), cited as a convergence point below; no
Q1/Q2/Q4/Q5 claim is made for Unreal since no primary artifact could be read.

## Q1 — the forward-wire control-point rule

| Reference | File / function | Rule (quoted) | Constants / clamps |
|---|---|---|---|
| imgui-node-editor | `imgui_node_editor.cpp` `ed::Link::GetCurve()` | `cp0 = m_Start + m_StartPin->m_Dir * startStrength; cp1 = m_End + m_EndPin->m_Dir * endStrength;` where `m_Dir` is a **fixed** unit vector assigned per pin kind (`imgui_node_editor.cpp`: `m_CurrentPin->m_Dir = kind==Output ? editorStyle.SourceDirection : editorStyle.TargetDirection`), never derived from the other endpoint. `startStrength`/`endStrength` come from `easeLinkStrength(a,b,strength)`: `distance=len(b-a); halfDistance=distance*0.5; if (halfDistance<strength) strength = strength*sin(PI*0.5*halfDistance/strength);` | `LinkStrength=100.0f` (`imgui_node_editor.h`, `Style` ctor default); `SourceDirection=(1,0)`, `TargetDirection=(-1,0)` (output always points `+x`, input always `-x`). The `sin` ease is a soft clamp: it shrinks the strength continuously as the two pins get closer than `2*strength` apart, never a hard `min()` |
| imnodes | `imnodes.cpp` `GetCubicBezier()` | after `ImSwap(start,end)` when `start_type==Input` (so `start` is always canonicalized to the output): `link_length = sqrt(LengthSqr(end-start)); offset = (0.25*link_length, 0); P1 = start+offset; P2 = end-offset;` | Fixed `0.25` of the full Euclidean distance (not just `dx`); no minimum, no clamp |
| ImNodeFlow (`smart_bezier`) | `src/ImNodeFlow.inl` | `distance = sqrt(dx*dx+dy*dy); delta = distance*0.45f;` (forward branch; two backward-only adjustments follow, Q2) | `0.45` fraction of Euclidean distance; vertical component of the offset is explicitly zeroed |
| xyflow (`calculateControlOffset` via `getControlWithCurvature`) | `bezier-edge.ts` | `function calculateControlOffset(distance, curvature) { if (distance >= 0) return 0.5*distance; return curvature*25*Math.sqrt(-distance); }`, called per-endpoint as `getControlWithCurvature({pos, x1,y1,x2,y2,c})`, e.g. for `Position.Right`: `x1 + calculateControlOffset(x2-x1, c)` | `curvature` default `0.25` (`getBezierPath` default parameter); forward offset is `0.5 * signed distance along the pin's own facing axis`, not full Euclidean length |
| litegraph.js (`SPLINE_LINK`) | `src/litegraph.js` `LGraphCanvas.prototype.renderLink` | `dist = distance(a,b)` (Euclidean); per `start_dir`/`end_dir` switch: `RIGHT: +0.25*dist`, `LEFT: -0.25*dist` (similarly `UP`/`DOWN` on y), fed into `ctx.bezierCurveTo(a.x+start_offset_x, ..., b.x+end_offset_x, ..., b.x, b.y)` | Fixed `0.25` of Euclidean distance; `start_dir`/`end_dir` default to `RIGHT`/`LEFT` and are **not** derived from the sign of `dx` |
| Blender (`calculate_inner_link_bezier_points`) | `drawnode.cc` | `dist_x = math::distance(x0,x3)` (Blender's float `distance` is `abs(a-b)`, always ≥0); `dist_y = math::distance(y0,y3)`; `slope = safe_divide(dist_y,dist_x)`; `clamp_factor = min(1.0, slope*(4.5-0.25*curving))`; `handle_offset = curving*0.1*dist_x*clamp_factor`; `points[1].x = points[0].x + handle_offset; points[2].x = points[3].x - handle_offset;` (`points[0]`=from/output, `points[3]`=to/input) | `curving` is a user theme int (`TH_NODE_CURVING`, default `4` in the shipped theme); `curving==0` degenerates to a dead-straight line (`points[1]`/`points[2]` linearly interpolated at 1/3, 2/3); `clamp_factor` is capped at `1.0` and **shrinks** as the link approaches horizontal (`slope→0`) — the opposite of a minimum-offset floor, an anti-flatness correction none of the draw-list libraries above have |
| Rete (`classicConnectionPath`) | `render-utils/src/connection.ts` | `vertical = abs(y1-y2); hx1 = x1 + max(vertical/2, abs(x2-x1))*curvature; hx2 = x2 - max(vertical/2, abs(x2-x1))*curvature;` | `curvature` is a caller-supplied parameter, no default in this package; the `max(vertical/2, |dx|)` term is unique to Rete among these seven — it *grows* the offset from vertical separation so a near-vertical link doesn't go flat, the opposite problem from Blender's `clamp_factor` |

## Q2 — the backward case (consumer left of producer): the S-curve and the cusp

| Reference | Special-casing? | Mechanism (quoted) | Does it avoid the fold? |
|---|---|---|---|
| imgui-node-editor | No branch | `m_Dir` is fixed per pin **kind**, so `cp0` always offsets `+x` from the output and `cp1` always offsets `-x` into the input, whichever pin is physically on which side | **Yes, by construction** — proven below |
| imnodes | No branch | `offset.x = 0.25*link_length ≥ 0` always added to the (canonicalized) output and subtracted from the input | **Yes, by construction** — proven below |
| litegraph.js | No branch | `dist*0.25 ≥ 0` (Euclidean distance) always added to the output's fixed `RIGHT` axis, subtracted from the input's fixed `LEFT` axis | **Yes, by construction** — proven below |
| Blender | No branch | `handle_offset ≥ 0` (built from `abs(x0-x3)`) always added to the output's x, subtracted from the input's x, no sign check on which x is larger | **Yes, by construction** — proven below, despite reading at first glance like a bug |
| xyflow | **Yes, explicit branch** | `calculateControlOffset`: `distance>=0 → 0.5*distance`; `distance<0 → curvature*25*sqrt(-distance)`. `distance` is the *signed* distance along the pin's own facing axis (e.g. `x2-x1` for a `Right`-facing source), so it goes negative exactly when the link runs backward relative to that pin | Yes — sub-linear (`sqrt`) growth for the backward case, continuous at `distance=0` (`0.5*0 == curvature*25*sqrt(0) == 0`), by an explicit rule rather than a structural guarantee |
| ImNodeFlow | **Yes, two-stage branch, and it is broken** | `smart_bezier`, quoted in full: `float delta = distance*0.45f; if (p2.x < p1.x) delta += 0.2f*(p1.x-p2.x); ImVec2 p22 = p2 - ImVec2(delta,0); if (p2.x < p1.x - 50.f) delta *= -1.f; ImVec2 p11 = p1 + ImVec2(delta,0);` | **No, not for a 0-50px backward gap.** `p22` is computed with the un-flipped (still-positive) `delta` in all cases; `p11`'s sign only flips once the backward gap exceeds a hardcoded **50px** threshold. So for `0 < p1.x-p2.x <= 50`, both `p11` and `p22` are still offset in the *same* direction (both effectively pulled toward the same side), which is exactly the "two control points converging instead of diverging" shape that folds. Cited as a cautionary, shipped example of the same bug class as ShaderBox's own bus |

**A structural proof, not a re-read of the reports.** Three of the four "no branch" rows above
looked, on a first pass through two independently-spawned sub-agents' drafts, like they might
*not* actually avoid the fold for Blender specifically ("its formula is direction-blind, so a
backward link bows outward on both sides and can fold") — that claim was checked here by hand
before being accepted or rejected, because it contradicts the same reports' own claim that Blender
uses the same "fixed side, non-negative offset" mechanism as imgui-node-editor. Let `x0` = output x,
`x3` = input x, `offset ≥ 0` in all four cases (imgui-node-editor's `startStrength`/`endStrength`
after the `sin` ease is still non-negative; imnodes', litegraph's and Blender's are built from
`abs()`/Euclidean distance). In every one of the four, the output's control point is `x0 + offset`
and the input's is `x3 - offset`. A fold requires the two control points to cross:
`x0 + offset ≤ x3 - offset`, i.e. `2*offset ≤ x3 - x0`. In the backward case `x3 < x0`, so the
right-hand side is negative; since `offset ≥ 0`, the left-hand side is non-negative, so the
inequality **can never hold** — crossing is mathematically impossible for any offset magnitude,
however large `curving`/`0.25`/`0.45` makes it. Verified numerically for Blender (output at
`x=500`, input at `x=100`, `curving` swept 1-10: the output control point stays right of the input
control point at every value; see the arithmetic that produced this table for the reasoning, not a
citation). This is **the single most load-bearing finding of this report**: four of six references
are cusp-proof not because they special-case direction, but because their offset is structurally
incapable of making the control points converge. xyflow and ImNodeFlow are the only two that
compute a *signed* offset and therefore need (and in ImNodeFlow's case, botch) an explicit backward
branch.

## Q3 — routing around nodes / bundling

| Reference | Routing/avoidance? | What it does instead |
|---|---|---|
| imgui-node-editor | None (`grep -i "obstacle\|avoid" imgui_node_editor.cpp imgui_node_editor.h` — no hits beyond unrelated words) | Draws every link on its own draw-list channel (`c_LinkChannel_Links`, plus a wider copy on `c_LinkChannel_Selection` for hover/selected), so separation is z-order, not path-finding; the user moves nodes to declutter |
| imnodes | None found | Same: links on their own channel before nodes, straight bezier, no occlusion logic |
| litegraph.js | None (repo-wide grep for "obstacle"/"avoid"/"route" — only unrelated hits in comments) | Draws over/under by node registration order; the three render *styles* (straight/linear/spline) are a visual choice, not routing |
| ImNodeFlow | None | Same, draws on top, no node-bounding-box awareness in `smart_bezier` |
| xyflow | None automatic; `smoothstep-edge.ts` gives an *orthogonal-looking* path via a fixed one-or-two-bend heuristic, and says so in its own comment: *"With this function we try to mimic an orthogonal edge routing behaviour. It's not as good as a real orthogonal edge routing, but it's faster and good enough as a default"* | A fixed-bend heuristic keyed only on the two endpoints' facing directions, not on other nodes' positions — a curve-family choice, not obstacle avoidance |
| Blender | **User-placed Reroute nodes**, not automatic routing | `node.is_reroute()` gates several paths in `node_relationships.cc` (always link-eligible even on a type mismatch; mute status propagated through it both directions; a link can be "inserted into" one on drop) — Blender's entire routing story is a real node the user drops onto a wire to bend it |
| Unreal (docs only) | Not documented as automatic | Epic's own docs name Reroute nodes as the path-shaping mechanism (quoted above) — same manual answer as Blender |

**Convergence:** none of the six code references, and neither documented commercial editor, does
automatic obstacle avoidance. The universal answer is "draw the bezier straight, rely on z-order,
let the user rearrange nodes"; Blender's and Unreal's only addition is a manual tool (a real node
the artist places), not an algorithm.

## Q4 — arrowheads, thickness, endpoint attachment

| Reference | Endpoint attachment | Arrowhead | Thickness / zoom scaling |
|---|---|---|---|
| imgui-node-editor | Exactly at the pin's pivot point by default; when `m_SnapLinkToDir` (default true) and an arrow/radius is configured, the curve start snaps outward by `radius+arrowSize` along `m_Dir` — with both at their default `0`, it is the bare pin center | Opt-in per pin (`PinArrowSize`/`PinArrowWidth` style vars, both default `0`, i.e. off); drawn via `ImDrawList_AddBezierWithArrows` when non-zero | `m_Thickness` set from the `LinkThickness` style var; widened by a fixed screen-space amount when selected/hovered, drawn on a separate channel so the wide border sits over the normal link. Thickness lives in the same coordinate space as the rest of the canvas, which is itself zoom-transformed (`imgui_canvas.cpp`), so it scales with zoom implicitly, not by an explicit per-call multiply |
| imnodes | Exactly at the pin's position, no stub | Not found (`grep -i arrow imnodes.cpp` — no hits); imnodes draws no arrowheads | `Style.LinkThickness` default `3.0f`; imnodes has no built-in canvas zoom, so this is a constant in screen pixels always |
| litegraph.js | Exactly at the port's canvas position (`ctx.moveTo(a[0],a[1])` / bezier to `b`) | Off by default (`render_connection_arrows=false`); when enabled, drawn at the link's **midpoint**, not at either endpoint | `connections_width` (canvas px); the whole `ctx` is under the canvas's `ds.scale` transform when `renderLink` runs, so thickness scales with zoom the same way any Canvas2D stroke under a transform does |
| ImNodeFlow | Exactly at the pin's point (`pinPoint()`), no stub | Not found in `smart_bezier` | `link_thickness`/`link_hovered_thickness`, plain `AddBezierCubic` parameters; no explicit zoom-scaling code found in the files read |
| xyflow | SVG path starts/ends exactly at the handle's `sourceX,sourceY`/`targetX,targetY`, no stub | Opt-in via SVG `marker-start`/`marker-end` refs, off by default | `stroke-width` is a CSS/SVG property; the whole edge lives inside a CSS-transformed `<g>`, so stroke and path coordinates scale together with viewport zoom automatically, no manual scaling code |
| Blender | Exactly at the socket's `runtime->location` (`socket_link_connection_location`), with a documented fan-out offset only for multi-input sockets | Not found in `drawnode.cc`'s link-geometry functions; not resolved as a "no arrowheads anywhere" finding — flagged as a gap since the GPU draw call for the link itself lives in a section of the same file not captured by this research's greps | Not resolved from the file read — the draw call's thickness argument is threaded from a caller not traced; flagged as a gap, not a finding of "no scaling" |

## Q5 — hit testing

| Reference | Mechanism | Samples / segments | Pixel threshold | Zoom-scaled? |
|---|---|---|---|---|
| imgui-node-editor | `Link::TestHit`: bounding-box reject (`GetBounds().Expand(extraThickness)`, contains-check) then `ImProjectOnCubicBezier(point, P0,P1,P2,P3, 50)` | **50**, passed explicitly at the call site (the function's own default, per its declaration in `imgui_bezier_math.h`, is `subdivisions=100`) | `result.Distance <= m_Thickness + extraThickness` — the link's own draw thickness IS the threshold, no separate constant | Yes, implicitly: hit-testing runs in the same canvas-space coordinates as drawing, and the whole canvas is zoom-transformed, so the threshold scales with zoom the same way the line does |
| imnodes | `GetDistanceToCubicBezier`: bounding-rect pre-check (expanded by hover distance) then walks straight sub-segments, `ImLineClosestPoint` per segment, keeps the minimum | `num_segments = max(int(link_length * LinkLineSegmentsPerLength), 1)`, `LinkLineSegmentsPerLength` default `0.1` — one segment per 10px of link length, adaptive | `LinkHoverDistance` default `10.0f` px | imnodes has no built-in camera zoom, so this is a literal, constant screen-pixel threshold |
| litegraph.js | **Not a curve-distance test.** The hover-pick loop tests only a fixed box around the link's precomputed midpoint (used for its hover tooltip) | 0 — no curve flattening at all | A small fixed box (single digits of px) around one point per link, not a full-curve test | Not meaningfully zoom-scaled; the box itself is a constant regardless of zoom |
| ImNodeFlow | `smart_bezier_collider`: recomputes the same control points as `smart_bezier`, then `ImProjectOnCubicBezier(p, p1,p11,p22,p2).Distance < radius` — the imgui-node-editor math, vendored, at its **unoverridden default** of 100 subdivisions | 100 (default, unlike imgui-node-editor's own call site which explicitly passes 50) | `radius` passed by the caller; the call site read passes a small literal | Not resolved from the files read whether the caller scales the radius by zoom |
| xyflow | **No math-based hit test.** A second, invisible SVG `<path>` with the identical `d` attribute, `strokeOpacity={0}`, `strokeWidth={interactionWidth}`, relies on the browser's native SVG `pointer-events` hit-testing | N/A — delegated to the renderer | `interactionWidth` default (documented, tens of px) in SVG user-space units | The invisible path sits in the same CSS-transformed group as the visible edge, so its width scales with pane zoom exactly like the visible line |
| Blender | Flattens the bezier via a forward-difference evaluator into a small fixed number of points, then `dist_squared_to_line_segment_v2` per segment | A small fixed segment count (not adaptive to link length) | A DPI-scaled constant compared directly against a **squared** distance — a looseness in Blender's own source, noted as observed rather than re-derived further | DPI/UI-scale-aware, not node-editor-zoom-aware — a fixed threshold in *view* space regardless of how far the user has zoomed the node tree |

## Convergence

- **The S-curve for a backward link is not a special case in four of six code references
  (imgui-node-editor, imnodes, litegraph.js, Blender).** It falls out for free, and is
  *mathematically guaranteed* (not just typical), from a formula whose control-point offset is
  always non-negative and always points along the endpoint's own fixed outward axis — proven above,
  not merely observed. This is the single most load-bearing finding for ShaderBox: the fix is not
  "detect backward and switch to an S formula," it is "use an offset that can never be negative and
  never changes which way it points, and the fold becomes impossible by construction."
- **No reference routes wires around node geometry, code or documented.** All seven (six code, plus
  Unreal's docs) draw the curve straight from pin to pin and rely on z-order/channel layering or
  manual reroute placement, never automatic obstacle avoidance.
- **Hit testing on an immediate-mode/canvas renderer (imgui-node-editor, imnodes, ImNodeFlow,
  Blender) converges on "flatten to a polyline, take the minimum distance to a segment, compare to a
  small pixel threshold."** Sample counts differ (50, adaptive-by-length, 100, a small fixed count)
  but the algorithm shape is identical across all four. SVG-based xyflow skips this by delegating to
  the browser's own path hit-testing.
- **Arrowheads are opt-in and off by default everywhere they exist** (imgui-node-editor, xyflow,
  litegraph.js); a plain undecorated curve is the converged default look.

## Divergence

- **xyflow and ImNodeFlow are the only two references with an explicit backward branch**, and for
  the same underlying reason: both compute a *signed* offset from the endpoint's own facing axis
  rather than an always-non-negative one, so both need a sign-aware rule to avoid a fold that the
  other four references never risk in the first place. xyflow's `sqrt` term is continuous at the
  sign change; ImNodeFlow's hardcoded 50px cutoff is not, and visibly fails to avoid the fold inside
  that 0-50px band (proven above) — a shipped, maintained example of the same bug class as
  ShaderBox's own bus.
- **litegraph.js and xyflow are the two outliers on hit testing, for opposite reasons.** xyflow
  delegates entirely to native SVG hit-testing (no app-side curve math at all, because SVG affords
  it for free); litegraph.js also skips curve math, but because it only cares about a hover tooltip
  at one cached point — a real functional gap if the intent is "click anywhere on the wire to select
  it," which litegraph.js does not support.
- **Curvature-vs-flatness correction is not universal.** Only Blender (`clamp_factor`, shrinking the
  offset as a link approaches horizontal) and Rete (`max(vertical/2, |dx|)`, growing the offset for
  near-vertical links) treat the perpendicular separation as a first-class input to the offset
  formula; imgui-node-editor, imnodes, litegraph.js and ImNodeFlow's forward branch all size the
  offset from the horizontal run alone (or full Euclidean length, dominated by `dx` in a left-to-
  right graph) and ignore vertical separation.

## False trails

- **retejs/connection-plugin** looked like the right repo by name (Rete's own "classic" connection
  preset, handles drag-to-connect) but contains no bezier/curve-geometry code — `presets/classic.ts`
  and `flow/*` are pure interaction-state machines. The actual path formula lives in the separate
  `retejs/render-utils` package (`classicConnectionPath`); go there directly rather than following
  the brief's literal "rete-render-utils / the classic connection path" phrase into the plugin repo.
- **Unreal Engine's documented S-curve/tangent rule** does not exist as a citable formula: a web
  search of Epic's own docs for the terms the brief names returns nothing implementable. What is
  documented (Reroute nodes as the path-shaping mechanism) is cited above under Q3; no Q1/Q2/Q4/Q5
  claim is made for Unreal since no formula could be verified against a primary artifact.
- **Blender's `node_link_bezier_handles`/`node_link_bezier_points`, as the brief names them, are not
  in `node_draw.cc`.** The actual handle math (`calculate_inner_link_bezier_points`) lives in
  `drawnode.cc`; `node_draw.cc` only calls the evaluated-points function. The brief's function names
  are correct; its file attribution needed correcting during this research.
- **A first-pass reading of Blender's backward-link formula as "direction-blind, therefore it folds
  like ShaderBox's bus"** is itself a false trail, caught and disproved by direct arithmetic in the
  note after Q2 — worth flagging explicitly since it is the kind of claim that reads as plausible
  (an unconditional `abs()` sounds like a missing branch) without actually working the algebra.
- **ImNodeFlow's `smart_bezier` looked like a good pattern to adapt** at first glance — a
  maintained, purpose-built "smart" curve for exactly this problem. Reading it in full instead
  surfaces it as the closest analogue to ShaderBox's own bug: an asymmetric backward-case patch that
  visibly produces a same-direction-control-points fold for gaps between 0 and 50px. Cited above as
  a cautionary example, not adapted.

## Recommended for ShaderBox

**Principle: replace the bus entirely.** Draw one cubic bezier from output dot to input dot for
every edge, forward or backward, with an offset formula that is structurally incapable of folding —
matching the proven mechanism of imgui-node-editor / imnodes / litegraph.js / Blender — rather than
xyflow's or ImNodeFlow's signed-branch approach. A branch-free, always-non-negative offset needs no
tuning to avoid a cusp; a signed-branch rule can (ImNodeFlow proves it does, in a maintained
library) get the branch wrong. The maintainer's cusp is the bus's fault, stated explicitly in the
brief; removing the bus removes the cusp's structural cause, not just its visible symptom.

**One formula, forward and backward, no branch:**

```
dx = b.x - a.x           # a = output dot, b = input dot, screen space (post zoom/pan)
dy = b.y - a.y
dist = sqrt(dx*dx + dy*dy)
offset = max(MIN_OFFSET, CURVE_FRAC * dist)

cp0 = (a.x + offset, a.y)     # always points +x from the output, whatever side b is on
cp1 = (b.x - offset, b.y)     # always points -x into the input, whatever side a is on
```

- `offset ≥ 0` always (it is `max` of two non-negative terms) and always points the same way
  relative to its own endpoint — this is the exact property proven above to make a fold
  mathematically impossible, at any `dist` and any `CURVE_FRAC`. No clamp is needed on the upper end
  for the same reason: an unbounded offset can produce a wide bow but never a crossing.
- `CURVE_FRAC = 0.4`: between imnodes/litegraph's `0.25` and ShaderBox's current `0.45`
  (`_BEZIER_BOW`); ShaderBox's own recent value is close to imgui-node-editor family's territory
  already, so keep it in that neighbourhood rather than importing a new constant untested against
  this codebase's node spacing. Tune by eye once implemented — every reference's constant is a
  hand-picked value, not a derived one.
- `MIN_OFFSET = 24.0 * zoom` (screen px) — reuse ShaderBox's existing `_MIN_DIRECT_DX` value so
  already-tuned short-run cases don't visibly jump, and so a near-zero-length link doesn't collapse
  toward a straight line.
- No `dy`-based flatness correction (Blender's `clamp_factor`, Rete's `max(vertical/2, ...)`):
  worth adding only if a real screenshot later shows a near-horizontal long link looking
  conspicuously flat — not needed for a first cut, and it is the one place the six references
  genuinely disagree on whether it's worth the complexity (Blender and Rete do it in opposite
  directions; imgui-node-editor, imnodes, litegraph, ImNodeFlow's forward branch don't do it at
  all).
- This single formula **is** the backward-case rule too: for `dx < 0`, `cp0` still sits at
  `a.x + offset` (right of the output) and `cp1` still sits at `b.x - offset` (left of the input) —
  both handles point outward past their own endpoint toward the far side, which is exactly the
  S-curve shape, with no `if backward:` anywhere in the drawing code and no threshold that a
  near-boundary case can fall into (contrast ImNodeFlow's 50px cutoff, proven above to fold).

**Hit testing:**

```
subdivisions = 24                                     # fixed, not adaptive
threshold_px = max(6.0, SIZE.GRAPH_WIRE_W * 2.0) * zoom   # screen space, scales with zoom
```

Flatten the same four points (`a, cp0, cp1, b`) into 24 straight sub-segments with the standard
cubic bezier point formula, take the minimum point-to-segment distance (imnodes' and Blender's
shared algorithm shape), compare to `threshold_px`. Precede it with a bounding-rect pre-check
(min/max of the four points, expanded by `threshold_px`) as a cheap reject, matching
imgui-node-editor's `TestHit` shape. A fixed 24 is a middle ground between imnodes' adaptive
`length/10` (which for ShaderBox's typical 100-400px screen-space runs lands in the 10-40 range
anyway) and Blender's small fixed count; pick fixed over adaptive so the hit test is one code path,
not two. Do the distance comparison in screen space (post `_Xf.to_screen`, which is what
`threshold_px` above already assumes) rather than canvas space, matching imgui-node-editor/imnodes:
a fixed-canvas-unit threshold would make wires nearly unclickable at `GRAPH_ZOOM_MIN=0.25` and
oversized at `GRAPH_ZOOM_MAX=2.5`. Do not use litegraph's single-midpoint-box shortcut — it fails
the brief's own finding 1 ("a wire cannot be selected... except by grabbing it off its port"),
which every other reference in this set treats as a first-class interaction to support.

**Endpoints, arrowheads, thickness:**

- Attach exactly at the existing `_out_point`/`_port_point` dot centers — every reference read
  attaches at the bare endpoint by default (imgui-node-editor only offsets when an arrow is
  explicitly configured, which ShaderBox isn't adding). No stub needed.
- No arrowhead. Off by default in every reference that has the concept at all (imgui-node-editor's
  `PinArrowSize` defaults to `0`; imnodes and litegraph draw none by default); direction is already
  legible from ShaderBox's existing filled-circle-vs-ring port-dot distinction (`_draw_port_dot`).
  Adding one is extra visual weight for disambiguation ShaderBox doesn't need.
- Keep `SIZE.GRAPH_WIRE_W * zoom` as today — already the zoom-scaling pattern every reference
  either does explicitly (Blender) or gets for free from a zoom-transformed canvas/SVG group
  (imgui-node-editor, litegraph, xyflow). No change needed here.

**Routing around nodes: not worth it** for a canvas of at most a dozen nodes. Every one of the six
code references and Unreal's documented behavior converges on "don't" — none does automatic
obstacle avoidance, and the two production tools with any node-avoidance UX at all (Blender,
Unreal) both use a manual, user-placed Reroute node rather than an algorithm, which is a much
bigger feature (a new node kind, its own port, its own persistence) than this brief is scoped to.
Keep the existing draw-wires-under-nodes order (`dl.channels_split(2)` in `_draw_canvas` already
gets this right) and let the maintainer's existing drag-to-reposition handle any visual overlap —
exactly what every reference does instead of routing.

## What our current rule gets wrong, against the recommendation

| Current (`_draw_wire`, `_BEZIER_BOW`, `_MIN_DIRECT_DX`, `_draw_canvas`'s `backward`/`bus`) | Recommended | Why it matters |
|---|---|---|
| A backward or multi-rank edge is rerouted onto a shared horizontal "bus" line via two extra cubics whose control points are fixed at `GRAPH_GAP_X/2`, independent of the two edges' actual endpoints | One cubic, always, pin to pin, with a branch-free offset formula | The bus is the cusp's actual structural cause per the brief: two independently-anchored cubics converging on the same `bus_y` line can (and visibly do) cross. Removing the bus removes the failure mode; the four cusp-proof references above never needed a bus in the first place |
| Offset is `max(_MIN_DIRECT_DX*z, dx*0.45)` — computed from the **horizontal run only**, and only on the direct (non-bus) branch; the bus branch uses a different, unrelated offset rule (`GRAPH_GAP_X/2`) | `offset = max(MIN_OFFSET, CURVE_FRAC*dist)`, `dist` = full Euclidean distance, the same formula for every edge whatever its span or direction | Two different offset rules for "forward" vs "bus" edges is itself a seam that produces visually inconsistent curvature at the boundary; a single formula removes the seam, matching every reference read here (each has exactly one offset rule, not two) |
| `backward = b[0] < a[0] + _MIN_DIRECT_DX`, a hard threshold that switches curve **topology** (single cubic → three-segment bus) | No topology switch: the same single-cubic formula handles `dx<0` by construction | A threshold-based switch between two different curve shapes is exactly the failure shape ImNodeFlow's 50px cutoff has (proven above to fold inside its own dead zone); a branch-free formula has no such zone to fall into |
| No hit test on a wire at all today; the brief's own finding notes "a wire cannot be selected or deleted except by grabbing it off its port" | 24-segment flatten + point-to-segment distance, `max(6, GRAPH_WIRE_W*2)*zoom` threshold, zoom-scaled | Four of six references (imgui-node-editor, imnodes, ImNodeFlow, Blender) treat click-to-select-a-wire as a first-class interaction with this exact mechanism shape; ShaderBox's own finding 1 is a gap every one of them already closes |
