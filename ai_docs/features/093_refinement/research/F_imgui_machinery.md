# F — Machinery on an immediate-mode draw list

Area F answers question 16: how editors built on Dear ImGui's draw list hit-test wires, layer
their draw calls, handle zoom without blurring text, debounce click vs drag, and what each gave
up. Read as CODE, not README, per the brief; every claim below is a function/struct/constant name
found in the cloned source, checked with grep and targeted `sed -n` reads, never guessed from
training data.

## Sources

| Name | What it is | Path (cloned shallow) | Read as |
|---|---|---|---|
| thedmd/imgui-node-editor | The reference full-featured ImGui node editor (Blueprints-style); C++, ships a Python binding via imgui-bundle | `refs/imgui-node-editor/` | code: `imgui_node_editor.cpp`, `imgui_node_editor_internal.h`, `imgui_canvas.cpp/.h`, `imgui_bezier_math.h/.inl` |
| Nelarius/imnodes | A minimal, single-header-style ImGui node editor; no zoom | `refs/imnodes/` | code: `imnodes.cpp`, `imnodes_internal.h`, `imnodes.h` |
| Fattorino/ImNodeFlow | A node editor that runs the canvas as its own nested ImGui context (not just a transformed draw list) | `refs/ImNodeFlow/` | code: `src/ImNodeFlow.cpp`, `src/ImNodeFlow.inl`, `src/context_wrapper.h`, `include/ImNodeFlow.h` |
| imgui-bundle's `imgui_node_editor` Python binding | The Python surface actually available to ShaderBox if it ever adopted imgui-node-editor | `.venv/lib/python3.12/site-packages/imgui_bundle/imgui_node_editor.pyi`, `demos_python/demos_node_editor/demo_node_editor_basic.py` | code (stub + demo) |
| `shaderbox/widgets/pass_graph.py` | ShaderBox's own hand-rolled canvas, the baseline every recommendation is adapted onto | repo | code |
| `.claude/skills/imgui-ui/SKILL.md` §3/§4/§8 | The traps ShaderBox has already paid for on this exact draw-list-canvas shape | repo | doc |

`ocornut/imgui`'s own `imgui_demo.cpp` "Custom rendering" canvas example was not cloned
separately — its shape (`InvisibleButton` + `IsItemHovered` + `io.MouseDelta` + `PushClipRect`,
no channels, no zoom) is exactly what `pass_graph.py`'s background/pan/rubber-band handling
already does; it is folded into CONVERGENCE rather than given its own row.

## Q16 — hit testing, layering, zoom, click-vs-drag, overlap traps, what each gave up

### Hit testing

| | Node | Pin | Link/wire | Distance threshold | Scales with zoom? | Priority when overlapping |
|---|---|---|---|---|---|---|
| imgui-node-editor | The PER-FRAME interaction resolver is `EditorContext::BuildControl`, not `FindNodeAt` (that function is used exactly once, by `NavigateAction::HandleZoom`'s "zoom to hovered object" logic, not general hover/click). `BuildControl` walks `m_Nodes` in REVERSE (`m_Nodes.rbegin()` — last-submitted/topmost node first) and for each live node emits a real hand-rolled item (`emitInteractiveAreaEx`→`invisibleButtonEx`, a local clone of `InvisibleButton` built on the SAME `ImGui::ButtonBehavior`) over `node->m_Bounds` (or, for a group node, over each `NodeRegion` sub-rect) | Each node's pins are checked BEFORE that node's own body, in the same reverse per-node loop (`for (auto pin = node->m_LastPin; pin; pin = pin->m_PreviousPin) checkInteractionsInArea(pin->m_ID, pin->m_Bounds, pin)`) via the same `invisibleButtonEx` item mechanism, rect-based on `pin->m_Bounds` | Links are NOT real items — checked only AFTER the full node/pin loop, and only `if (nullptr == hotObject)` (i.e. only when no node or pin claimed hover that frame): `hotObject = FindLinkAt(mousePos)`, which iterates `m_Links` and calls `Link::TestHit(point, extraThickness)`: cheap `ImRect.Contains` pre-check, then `ImProjectOnCubicBezier(point, P0,P1,P2,P3, 50)` (analytic closest-point, 50 subdivisions), passes when `result.Distance <= m_Thickness + extraThickness` | `c_LinkSelectThickness = 5.0f` **canvas-space pixels** (declared `static const float`, comment `// canvas pixels`) | Yes — canvas pixels are pre-zoom local space; the whole canvas is later rescaled by `Canvas::LeaveLocalSpace`, so a fixed local threshold becomes zoom-proportional screen pixels automatically | `checkInteractionsInArea`'s callback sets `hotObject` on `!hotObject && IsItemHovered(...)` — i.e. FIRST-HIT-WINS within the loop, and because nodes are walked in REVERSE submission order with each node's pins checked immediately before that node's body, the effective priority is: **topmost-submitted node's pins, then that node's body, then the next-topmost node's pins/body, ... then links last of all** (links "steal" a click from the background afterward if `hotLink` is set — see `BuildControl`'s post-loop background-click-stealing block). So imgui-node-editor DOES use ImGui's own item system (`ButtonBehavior`) for nodes and pins — only links bypass it |
| imnodes | `ResolveHoveredNode(depth_stack)`: candidates come from `NodeIndicesOverlappingWithMouse` (populated during node submission via rect containment); if >1 candidate, picks the one with the largest index into `editor.NodeDepthOrder` (i.e. the node drawn most recently / highest in z) | `ResolveHoveredPin`: squared-distance to `pin.Pos` vs `PinHoverRadius^2`, smallest distance wins, occluded pins (`occluded_pin_indices`) skipped | `ResolveHoveredLink`: **pin-priority first** — if a pin is already hovered, any link touching that pin is immediately reported hovered (`return idx`) without distance testing, so link-detach-by-drag works; otherwise nearest-bezier by `GetClosestPointOnCubicBezier` over a **manually flattened polyline**, segment count = `LinkLineSegmentsPerLength * curve length` (adaptive, not fixed) | `PinHoverRadius = 10.0f`, `LinkHoverDistance = 10.0f` (both `ImNodesStyle` defaults, screen-space local-canvas pixels) | Yes, same reasoning as above — imnodes canvas coordinates are local/grid space until draw, so a fixed local constant is zoom-proportional; **but imnodes has no canvas zoom at all** (see Zoom row), so this is moot for it in practice | Pin-hover explicitly outranks link-hover (a link connected to the hovered pin is reported even if the mouse isn't geometrically closest to it); among overlapping nodes, highest depth-stack index (topmost by z) wins |
| ImNodeFlow | `BaseNode::isHovered()`: plain `ImGui::IsMouseHoveringRect` over the node's padded rect | No dedicated pin hit test found beyond the node body / a per-pin small `InvisibleButton`-adjacent widget (not centrally resolved like the other two — no `ResolveHoveredPin` equivalent) | `Link::update()` calls `smart_bezier_collider(mouse, start, end, 2.5)`, which rebuilds the SAME bezier control points as the draw call (`smart_bezier`) and calls `ImProjectOnCubicBezier(p, p1,p11,p22,p2)` with the library DEFAULT subdivision count (100, not overridden) | Fixed `2.5f` (function-call literal, not a style token) | Local-canvas-space again (drawn inside `ContainedContext`, see Zoom row), so it self-scales with the nested-context zoom | No explicit multi-candidate priority machinery — `m_hovering` is set to whichever pin's own per-frame update ran last that frame (iteration order of an `unordered_map`), so overlap resolution is incidental, not designed |

### Layering (channels, z-order)

| | Channel count & layout | Node draw order | Bring-to-front on hover/select? |
|---|---|---|---|
| imgui-node-editor | Fixed layer budget declared as named constants in `imgui_node_editor.cpp`: `c_UserLayersCount=5` (content/grid/hints etc, channels 0-4), then `c_BackgroundChannelCount=1` (selection rect), then `c_LinkChannelCount=4` (`c_LinkChannel_Selection`, `_Links`, `_Flow`, `_NewLink`), THEN a per-node block of `c_ChannelsPerNode=5` (`c_NodeBaseChannel`, `c_NodeBackgroundChannel`, `c_NodeUserBackgroundChannel`, `c_NodePinChannel`, `c_NodeContentChannel`) allocated dynamically per node (`m_Channel + c_NodeBaseChannel` etc, so N nodes cost 5N channels on one `ImDrawListSplitter`/`m_PinSplitter`) | `m_Nodes` is a plain vector; no code path was found that reorders it on hover — nodes are appended once (`m_Nodes.push_back`) and never resorted by `FindNodeAt`/`TestHit`/draw | **Not by hover.** No `BringToFront` function exists in this codebase (grepped `m_Nodes.(erase|insert|push_back)` and `BringToFront` — no hits beyond the initial push). Selection changes color/border only (`m_IsSelected` flag read at draw time), not draw order |
| imnodes | `DrawListSet`/`DrawListAddNode` grow the splitter by exactly 2 channels per node (`ImDrawListGrowChannels(..., 2)`): channel 0 is the canvas grid, then each node gets `[background, foreground]` at `1+2*submission_idx` / `+1`, and `DrawListAppendClickInteractionChannel` adds one final top channel for the selection box + in-flight link | `editor.NodeDepthOrder`, a separate `ImVector<int>` independent of submission order | **Yes, explicitly.** `BeginNodeSelection` moves the just-selected node to the back of `NodeDepthOrder` (`depth_stack.erase(elem); depth_stack.push_back(node_idx)` — vector's back = topmost), then `EndNodeEditor` calls `DrawListSortChannelsByDepth`, which walks `NodeDepthOrder` and calls `DrawListSwapSubmissionIndices` to physically swap each out-of-order node's background+foreground channel PAIR into position — an O(n²) reorder run once per frame, only when depth order changed (`start_idx` early-out compares against `NodeIdxSubmissionOrder` first) |
| ImNodeFlow | Only **2** channels total, shared across every node (`draw_list->ChannelsSplit(2)` once per frame in `ImNodeFlow::update`, `ChannelsSetCurrent(1)` for "Foreground" inside each node's `update()`) — background/foreground is per-CALL, not per-node, so two nodes cannot occlude each other's background independently | `m_nodes` is `std::unordered_map<NodeUID, shared_ptr<BaseNode>>`; iteration order is hash order, not insertion or interaction order | **No.** No reordering code was found; a selected/hovered node draws wherever the hash-map iteration happens to place it that frame. This is a real regression relative to the other two — flagged in "what it gave up" below |

### Zoom and text crispness

| | Mechanism | Text handling at zoom | Cost stated in source |
|---|---|---|---|
| imgui-node-editor | `ImGuiEx::Canvas` draws everything at **1:1 local coordinates** (nodes, text, everything submitted between `EnterLocalSpace`/`LeaveLocalSpace` uses UNSCALED positions and the DEFAULT font size). `Canvas::LeaveLocalSpace` (called from `End()`/`Resume()`) then walks the RAW `ImDrawList::VtxBuffer` from `m_DrawListStartVertexIndex` to the current index and does `vertex->pos = vertex->pos * m_View.Scale + m_ViewTransformPosition` in place — a genuine post-hoc **vertex-buffer rescale**, plus the same affine transform applied to every `CmdBuffer[i].ClipRect`. Also scales `_FringeScale` (`fringeScale *= m_View.InvScale` on enter, restored on leave) — ImGui's anti-aliasing feather-width compensation, added upstream specifically "for sharp rendering while zooming" (confirmed in the repo's own `docs/TODO.txt`, now merged since ImGui 1.80) | Text is rasterized ONCE at the font's baked size and then scaled as part of the same vertex-buffer rescale as every other primitive — it is bitmap-scaled, not re-rasterized. At high zoom this blurs exactly like scaling a screenshot; the fringe-scale trick keeps edges from acquiring extra soft AA halo but does not add resolution | No explicit perf number, but the mechanism itself is the cost: it must walk and rewrite every vertex emitted inside the canvas scope EVERY frame, and it needs `Suspend()`/`Resume()` bracketing (see overlap traps) any time ImGui-native chrome (a tooltip, a popup, a `BeginChild`) must escape the transformed space, because vertices already inside a suspended-and-resumed sub-draw would otherwise get double-transformed |
| imnodes | **No zoom support in the main editor at all.** `imnodes.h`'s only zoom-adjacent constant is `ImNodesStyleVar_LinkLineSegmentsPerLength` (bezier tessellation density, unrelated to view scale); the README documents zoom only for the separate MiniMap overlay ("The mini-map can be zoomed and scrolled. Editor nodes will track the panning of the mini-map accordingly") — an independent, smaller, non-interactive-content view, not the main canvas. This is a genuine, acknowledged limitation, not an oversight — the project's own open issues (referenced in the README) discuss community zoom forks/patches that never landed upstream | N/A — text is always drawn at native resolution because there is never a scale transform to fight | Cost is the flip side: by refusing zoom, imnodes never needs vertex rescaling, `Suspend`/`Resume`, or fringe-scale compensation — its draw code is a plain `ImDrawList` user throughout |
| ImNodeFlow | `ContainedContext` (in `src/context_wrapper.h`) runs the canvas as **its own nested `ImGuiContext`** (`m_ctx`), not just a transformed region of the outer one. `ContainedContext::begin()` calls `ImGui::SetFontRasterizerDensity(roundf(m_scale * 100.0f) / 100.0f)` BEFORE switching context — this is Dear ImGui 1.92's dynamic-glyph-loading density control (the same mechanism ShaderBox's own `push_font(font, size*zoom)` exploits, per SKILL.md §8), so text is genuinely **re-rasterized at the current zoom density**, not bitmap-scaled. After the nested context finishes its own `new_frame`/`render`, `ContainedContext::end()` calls `AppendDrawData(cmd_list, origin, scale)` (in `context_wrapper.h`), which blits the inner context's already-rendered `ImDrawList`s into the outer window's draw list, multiplying vertex positions by `(scale, origin)` — i.e. it STILL does a vertex-buffer rescale for geometry, but text stays crisp because the glyphs were rasterized at the right density before the rescale, so the rescale is closer to 1:1 for glyph quads | Crisp — genuinely re-rasterized per current zoom (rounded to 2 decimal digits of density to bound texture churn) | The comment in `CopyIOEvents` states the explicit trade-off: it copies `InputEventsTrail` (already-processed events) rather than the live `InputEventsQueue` to avoid double-processing, "at the cost of exactly one frame of input latency inside the inner context" — a stated, deliberate 1-frame input lag as the price of a fully separate nested context |

**This is the single most load-bearing convergence/divergence point for ShaderBox**, because ShaderBox already does neither of the two "blurry" strategies: `pass_graph.py::_draw_node` calls `imgui.push_font(font, max(4.0, font.legacy_size * z))` — i.e. it re-selects/re-rasterizes the font at the CURRENT zoomed pixel size every frame, the same family as ImNodeFlow's `SetFontRasterizerDensity`, not imgui-node-editor's vertex-rescale-only approach. This is already the crisp-text strategy; area F's job is to confirm it, not propose replacing it.

### Click vs drag threshold

| | Mechanism | Pixel threshold | State machine |
|---|---|---|---|
| imgui-node-editor | Every drag-initiating call site uses `ImGui::IsMouseDragging(button, 1)` — the second argument is ImGui's `lock_threshold` override, hard-coded to `1.0f` local-canvas pixels everywhere (`DragAction::Accept`, `SelectAction`, `SizeAction`, `CreateItemAction` all use the literal `1`) — NOT `io.MouseDragThreshold` (default 6px) and not zero | `1.0f` canvas-local px (becomes ~1×zoom screen px after the vertex rescale) | `EditorAction::AcceptResult` returns `True`/`False`/`Possible` from each action's `Accept(Control)`; `DragAction::Accept` requires BOTH `control.ActiveObject` (an item is already the active ImGui item, i.e. mouse pressed while over it) AND `IsMouseDragging(button, 1)`; a plain click with no movement past 1px never transitions out of "possible" into `DragAction` at all, so `is_item_clicked`-equivalent logic downstream sees a click, not a drag |
| imnodes | `GImNodes->LeftMouseDragging = ImGui::IsMouseDragging(0, 0.0f)` — lock_threshold explicitly `0.0f`, i.e. ANY movement counts as "dragging" from imnodes' own point of view | `0.0f` — no debounce at the drag-detection primitive itself | The debounce instead lives in WHICH interaction starts: `ImNodesClickInteractionType_` is `None` until a press is matched to a specific target (`BeginNodeSelection`/`BeginLinkSelection`/`BeginLinkCreation`/panning/box-select) on the PRESS frame; a plain click with zero movement is resolved as a `None`→selection transition on release with the click-interaction type staying `Node`/`Link` rather than ever promoting to a moved-position drag, because node/link position updates only apply `GetIO().MouseDelta`, which is exactly zero for a non-moving mouse — so "click" vs "drag" falls out of whether any nonzero delta was ever integrated, not out of a separate threshold check |
| ImNodeFlow | Drag start is gated on `onHeader && mouseClickState` (a fresh single-use click over the node's header specifically, not anywhere on the body) which sets `m_dragged = true`; from then on every frame while `m_dragged` the node's target position is nudged by `m_inf->getScreenSpaceDelta()` (`IO.MouseDelta / scale()`) with no separate distance gate at all — dragging begins the instant the click lands on the header, and a genuine zero-movement click is a drag of magnitude zero (harmless because positions are then snapped to grid via `round(pos/step)*step`, so a same-pixel click doesn't visibly move the node) | None (0px, same-frame) | No explicit state machine — a single boolean `m_dragged` per node plus a handler-level `m_draggingNode` flag that other nodes read to know a drag is in flight (so hovering another node mid-drag doesn't start a second drag) |

### Overlap and cursor-position traps, and how each avoids them

| | Overlap handling | Cursor/SetCursorPos trap |
|---|---|---|
| imgui-node-editor | For NODES and PINS: uses a hand-rolled clone of ImGui's own item primitive (`invisibleButtonEx`, built on the same `ImGui::ButtonBehavior`) rather than the real `InvisibleButton` call, submitted in REVERSE node order with `set_next_item_allow_overlap`-equivalent behavior achieved for free by the first-hit-wins scan (topmost node's pins checked first, so a pin never loses to the body under it — no explicit overlap flag needed because the scan order alone encodes priority). For LINKS: genuinely bypasses the item system — `FindLinkAt` is a pure geometric pass over `m_Links`, consulted only when the node/pin scan found nothing, so links never compete for `invisibleButtonEx` overlap at all | `Canvas::Suspend()`/`Resume()` exist specifically so a `BeginChild`, tooltip, or popup can be opened FROM WITHIN the local (zoomed) coordinate space and have its own content positioned in normal SCREEN space rather than being doubly-transformed — `Suspend` calls `LeaveLocalSpace()` (undoing the vertex/clip transform bookkeeping so subsequent normal ImGui calls aren't touched by `LeaveLocalSpace`'s rescale a second time on `End()`), `Resume()` calls `EnterLocalSpace()` again. Both assert `m_DrawList->_Splitter._Current == m_ExpectedChannel` — i.e. Suspend/Resume must not straddle a channel switch, a documented ordering trap in the source comments themselves ("please make sure you do not interleave channel splitter with canvas") |
| imnodes | ALSO bypasses `InvisibleButton`/native item hover for node/pin/link detection (`ResolveHoveredNode`/`Pin`/`Link` as above); it does use plain ImGui widgets for node CONTENT inside `BeginNode`/`EndNode` the same way node-editor does. `NodeIndicesOverlappingWithMouse` is populated by comparing the mouse against each node's screen rect directly during submission — no overlap-order flag equivalent to ShaderBox's `set_next_item_allow_overlap` is needed because there's no competing ImGui item chain at all | No `Suspend`/`Resume` equivalent was found — imnodes has no zoom, so there is no local-space vs screen-space distinction to escape from; ordinary ImGui popups/children work unmodified |
| ImNodeFlow | Node/pin/link hover again bypasses `InvisibleButton` (`IsMouseHoveringRect` direct geometry checks); BUT because the canvas is a fully separate nested `ImGuiContext`, anything that must render OUTSIDE the zoomed nested context (a tooltip anchored to screen space, a dropped-link popup) is opened AFTER `m_context.end()` restores the outer context — i.e. ImNodeFlow's answer to the "escape the transform" problem is "do it in the outer context, sequenced after `end()`", not a `Suspend`/`Resume` pair inside one context | The dropped-link popup (`ImGui::OpenPopup("DroppedLinkPopUp")`) is explicitly opened only in `on_free_space()` (no node or link hovered) — same shape as ShaderBox's own canvas-background right-click-with-no-node-hovered rule in `pass_graph.py::_draw_canvas`, independently arrived at |

### What each gave up

| | Give-ups |
|---|---|
| imgui-node-editor | Bitmap-scaled text at zoom (blurs at extremes, mitigated only by fringe-scale AA compensation, not resolution); must rewrite every emitted vertex every frame inside the canvas (a real per-frame CPU cost proportional to visible geometry); Suspend/Resume discipline is fragile — the source's own comments flag unresolved edge cases around channel-splitter interleaving (`#FIXME: This condition is not enough to avoid when user choose to use channel splitter... More investigation is needed`); the Python binding (imgui-bundle) exposes only the high-level opinionated API (`begin_node`/`begin_pin`/`link`/`query_new_link`) plus read-back (`get_hovered_node/pin/link`, `get_current_zoom`, `screen_to_canvas`/`canvas_to_screen`) and `suspend()`/`resume()` — it does NOT expose `FindNodeAt`/`FindLinkAt`, the `ImDrawListSplitter` channel constants, or `BringToFront`, so adopting it from Python means adopting its full node/pin/link/style model wholesale, not borrowing just the hit-testing or channel machinery piecewise |
| imnodes | No zoom, full stop — the maintainers accepted this as a permanent scope boundary rather than build the vertex-rescale/font-density machinery the other two need; `BeginNodeTitleBar` measures via a plain `ImGui::BeginGroup()` (no absolute-position tricks), which is simple but means title-bar layout can't do anything a group can't |
| ImNodeFlow | No bring-to-front / z-order control for nodes at all (hash-map iteration order); shares only 2 draw-list channels across ALL nodes rather than one pair per node, so two overlapping nodes cannot each have their own independent background-vs-foreground layering — a link or highlight meant to sit "under this node but over that one" has no channel to live in; a stated 1-frame input-event latency for anything happening inside the nested context, by design |

## CONVERGENCE

All three bypass ImGui's native item system for LINK hit testing specifically — links are never
real `InvisibleButton`s in any of the three, always a hand-rolled per-frame distance pass over the
stored link list. Node and pin hit testing diverges more than it first appears: imgui-node-editor
actually DOES use ImGui's own `ButtonBehavior` for nodes and pins (`BuildControl`'s
`invisibleButtonEx`, a local clone of `InvisibleButton`, not a bypass of the item system, just not
the literal `ImGui::InvisibleButton` entry point); imnodes and ImNodeFlow hand-roll node/pin
hit-testing too (plain rect/distance checks against `ImGui::IsMouseHoveringRect` or stored bounds,
no `ButtonBehavior` involved). So the real convergence is narrower than "all three bypass the item
system": all three bypass it for LINKS, and node/pin hit-testing is split 1-2 rather than
unanimous. `pass_graph.py`'s approach (layering `invisible_button` hit rects for nodes AND ports
with the allow-overlap chain) sits closest to imgui-node-editor's node/pin half — see DIVERGENCE
for where the two still differ.

All three that render links use an **analytic distance-to-bezier test** (`ImProjectOnCubicBezier`
or an equivalent closest-point walk over the SAME control points used to draw the curve), never a
naive fixed grid of sample points unrelated to the curve shape, and all three express the hover
threshold as a small constant in the SAME coordinate space the curve itself lives in (so it scales
with zoom automatically, when zoom exists) rather than a hard screen-pixel constant.

All three that draw links under/behind nodes use an `ImDrawListSplitter` (or, for ImNodeFlow, the
plain two-channel form of the same primitive) rather than draw-order tricks — channel-based
layering, not z-sorting the draw calls by hand, is the converged answer to "wires under nodes."

Two of three (node-editor, imnodes) give overlapping nodes their OWN per-node channel pair so a
node's own background/pin/content layers never bleed into a neighbor's; the one that shares
channels across all nodes (ImNodeFlow) is also the one with no z-order control — the two gaps are
the same root cause.

## DIVERGENCE

**Bypassing ImGui's item system vs staying inside it — but only for LINKS, all three ways.**
ShaderBox's `pass_graph.py` stays inside ImGui's own hit-test machinery (`invisible_button` +
`set_next_item_allow_overlap` chain, documented in the module docstring and SKILL.md §8) for
nodes AND ports, so hover/active/clicked read as ordinary ImGui item queries and interoperate with
popups, drag detection, and focus the way the rest of the app already does. imgui-node-editor does
the same for nodes and pins (`BuildControl`'s `invisibleButtonEx`, built on the real
`ImGui::ButtonBehavior`) — the two projects converge there. imnodes and ImNodeFlow diverge further:
both hand-roll node AND pin hit-testing too (rect/distance checks, no `ButtonBehavior`), not only
link hit-testing. So the real fork is narrower than "ShaderBox vs the three": it's "node/pin hit
testing through the item system" (ShaderBox, imgui-node-editor) vs "node/pin hit testing hand-rolled
too" (imnodes, ImNodeFlow) — and ALL FOUR hand-roll link hit-testing, because a curve has no natural
`InvisibleButton` shape. This is not a mistake on ShaderBox's side to fix for nodes/pins; the
"recommended" section below keeps `pass_graph.py` on the item-system side for nodes/ports (matching
imgui-node-editor) and adds a hand-rolled pass for wires only (matching all four references),
because a hand-rolled hit-test pass for nodes/ports would also have to hand-roll everything ImGui's
item system currently gives `pass_graph.py` for free (activity tracking across frames, drag-delta
bookkeeping via `is_mouse_dragging`, popup-blocking interaction via `is_window_hovered`).

**Zoom strategy: none / vertex-rescale-only / font-density-plus-vertex-rescale.** imnodes opts out
entirely; imgui-node-editor rescales vertices and accepts bitmap-blurred text; ImNodeFlow
re-rasterizes text at the zoomed density (a full extra nested ImGuiContext to do it) and
vertex-rescales the rest. ShaderBox already re-rasterizes text at zoom too, but WITHOUT a nested
context — it pushes a font at `size * zoom` directly inside the same context, which is cheaper
than ImNodeFlow's approach and gets the same crisp-text result, at the cost (already paid, per
SKILL.md §8) of imgui-bundle 1.92 baking one glyph atlas entry per integer pixel size touched.

**Click-vs-drag threshold: near-zero (1px or 0px) vs ImGui's global default (6px).** Both
reference projects that debounce at all use a threshold far tighter than ImGui's own
`io.mouse_drag_threshold` default of 6.0px — 1px for node-editor, 0px (no debounce at the
primitive, resolved by "did MouseDelta integrate to nonzero") for imnodes. `pass_graph.py`
currently calls `imgui.is_mouse_dragging(imgui.MouseButton_.left)` with NO explicit
`lock_threshold` argument at every one of its four drag-initiation sites (node drag, wire drag,
port-press-as-node-drag, rubber band), so it inherits the global 6px default — tuned for ordinary
buttons and text selection, not for a canvas where "did I mean to nudge this node 2px or click
it" is a live UX question the maintainer explicitly raised ("What will be the most convenient
mouse control schema?").

## Recommended for ShaderBox

Numbers and rules only, written to be implemented blind against `pass_graph.py`'s existing shape
(one child, one `ImDrawList`, `invisible_button` hit rects, positions and colors from
`shaderbox/theme.py`'s `SIZE`/`COLOR` token bags).

**Channel layout — 5 channels, in this order, split once per canvas frame:**

1. `0` — wire halo (a new layer, wider+dimmer stroke under the wire itself, for the hover/selection
   glow finding 2/12 of the brief asks for elsewhere; drawing it in its own channel BELOW the wire
   channel means a halo never paints over a neighboring wire's crisp stroke).
2. `1` — wires (the existing `_draw_wire` calls, both plain and bus-routed, plus `_draw_self_loop`).
3. `2` — nodes (the existing node body/picture/name/badges/ports — everything in `_draw_node`).
4. `3` — the in-flight wire (the wire currently being dragged from a port, drawn AFTER every node so
   it is never occluded while the user is aiming it at a target port).
5. `4` — the rubber-band rect (drawn last so the selection marquee is always on top of everything,
   matching the existing draw order where the band rect is emitted after the node loop).

This replaces today's `dl.channels_split(2)` (wires=0, nodes=1) with a 5-way split; every existing
`channels_set_current(0)`/`(1)` call site keeps its meaning, two new channels are inserted around
it. `channels_merge()` stays a single call at the end — the probe below confirms this is safe.

**Hit-testing order and the wire-hover-among-overlaps rule:** keep `invisible_button` +
`set_next_item_allow_overlap` for nodes and ports (this already works, is inside ImGui's own item
system, and SKILL.md §8 already documents its one sharp edge). For WIRES specifically — which
today have no hit target at all (finding 1) — do NOT add one `invisible_button` per wire (a wire
is a curve, not a rect; a bounding-box button would be either too permissive on long shallow
curves or would need per-segment buttons, which reintroduces the overlap-order problem inside a
single wire). Instead, adopt the reference convergence directly: each frame, after nodes are
placed, do ONE hand-rolled pass over `picture.edges` computing distance from the mouse (in CANVAS
space, i.e. `xf.to_canvas(mouse)`, so the threshold is zoom-stable) to the same bezier control
points `_draw_wire` already computes, using `imgui.ImVec2` closest-point-on-segment over a
flattened sample of the curve (`ImProjectOnCubicBezier` is not exposed to Python, so flatten:
sample the curve at a fixed number of points scaled by its length, e.g.
`max(8, int(chord_length / 20))` segments, then closest-point-on-each-segment — the imnodes
approach, adaptive density, cheaper than imgui-node-editor's flat 50 for short wires). Threshold:
`5.0` canvas-space units (matches imgui-node-editor's `c_LinkSelectThickness`), i.e.
`5.0 / view.zoom` is NOT needed — compute distance in canvas space directly against a 5.0 canvas-
unit threshold, so it is zoom-stable by construction, exactly like the references. Track only the
SINGLE closest wire under the threshold (`smallest_distance` pattern, both node-editor and
imnodes) as `view.hovered_wire`; do this pass BEFORE the node invisible-buttons in submission
order but let a node/port hover WIN when both are hit that frame (ports are drag targets and must
never be shadowed by a wire passing near them) — i.e. compute wire-hover first, then let the
existing node/port loop overwrite `node_hovered`/port hover state unconditionally, and only trust
`view.hovered_wire` when `not node_hovered and drop_target is None` at the end of the frame. This
gives pins > nodes > wires priority on overlap, matching imnodes' explicit "pin hover outranks
link hover" rule (needed there for link-detach-by-drag; needed here so hovering a wire never
steals a click meant for the port it terminates at).

**Click-vs-drag threshold:** pass an explicit `lock_threshold` to every
`imgui.is_mouse_dragging(imgui.MouseButton_.left, lock_threshold=...)` call in `_draw_canvas`
(there are four call sites today) instead of relying on the global 6px default. Use **3.0 screen
pixels** — tighter than ImGui's 6px default (so a deliberate small node nudge registers promptly,
matching the maintainer's stated wish for the control scheme to feel considered rather than
generic), but not as aggressive as the references' 1px/0px (which assume their OWN hand-rolled hit
testing and gain nothing from ImGui's native click/drag disambiguation that `pass_graph.py`
already benefits from — going to 0px would make `is_item_clicked`-based clicks on nodes/ports fire
inconsistently against the drag branch that runs in the same frame). 3px is a single named
constant in `theme.py`'s `SIZE` bag (e.g. `SIZE.GRAPH_DRAG_LOCK_PX = 3.0`), read at all four sites,
so the graph's feel can be tuned in one place without touching four call sites.

**Z-order rule for hovered/selected nodes:** bring-to-front by REORDERING `nodes = list(picture.
nodes.values())` once per frame before the draw loop, not by touching the channel scheme. Sort
key: `(is_selected, is_being_dragged)` ascending, so selected/dragged nodes draw (and therefore
occlude) last within the shared node channel — this is cheap (list is already rebuilt every frame
from `_build_view`, so no persistent ordering state to maintain) and sidesteps imnodes' heavier
per-node-channel-pair-swap machinery, which ShaderBox does not need because `pass_graph.py`
already redraws its whole node set from scratch every frame rather than diffing against a
persistent channel assignment. This intentionally does NOT reorder on hover alone (only
select/drag) — matching imgui-node-editor's choice (selection changes color, not z-order, on
hover) and avoiding a distracting flicker where merely passing the mouse over a dense cluster
keeps reshuffling paint order.

**Text-at-zoom rule:** keep the current `imgui.push_font(font, max(4.0, font.legacy_size * z))`
pattern in `_draw_node` exactly as-is — it already matches the crisp-text strategy (re-rasterize
at zoomed size) that ImNodeFlow independently arrives at via `SetFontRasterizerDensity`, and it
avoids imgui-node-editor's blur-at-extreme-zoom trade-off entirely. No change recommended here;
this is a confirmation, not a new rule. One caveat carried forward from SKILL.md §8: this bakes
one glyph-atlas entry per INTEGER pixel size touched, so continuous zoom (0.25-2.5 range, per
`SIZE.GRAPH_ZOOM_MIN`/`MAX`) will bake a distinct atlas entry near-continuously while a zoom drag
is in flight — acceptable (this already happens today) but worth knowing if atlas growth is ever
profiled.

## Headless probe: does `channels_split`/`channels_merge` keep draw order stable when a rect
## moves between channels across frames?

Ran a standalone hidden-glfw + bare ImGui-context probe (no App, no test fixture — a 64×64 hidden
window, `imgui.create_context()`, a real `GlfwRenderer` so the dynamic-glyph-loading backend flags
1.92 asserts on are set correctly per SKILL.md §8) at
`/tmp/claude-.../scratchpad/probe_channels_final.py`. Three colored 10×10 rects (A red at x=0, B
green at x=20, C blue at x=40) are submitted in the FIXED call order A, B, C every frame inside
`dl.channels_split(5)` / per-rect `channels_set_current(ch)` / `dl.channels_merge()`, but the
CHANNEL each rect targets is varied per frame:

- frame 0: `A→ch2, B→ch3, C→ch4` (ascending — channel order matches submission order)
- frame 1: `A→ch4, B→ch3, C→ch2` (reversed — channel order now OPPOSES submission order)

**First read attempt was wrong and is worth recording as a trap.** Reading
`imgui.get_draw_data().cmd_lists[0].vtx_buffer` directly in submission order gives `[A, B, C]` in
BOTH frames, regardless of channel assignment — which looks like "channel routing has no effect."
It doesn't have no effect; that read is the wrong buffer. `ImDrawListSplitter::Merge`
(`imgui_draw.cpp` in the bundled real Dear ImGui source under
`refs/imgui-node-editor/external/imgui/`) reorders `CmdBuffer` and `IdxBuffer` by channel index —
it never touches `VtxBuffer`, which stays in submission order for the lifetime of the frame. The
buffer that answers "what does the GPU actually paint, in what order" is the **index** buffer, read
through each `ImDrawCmd`'s `IdxOffset`/`ElemCount`, not the vertex buffer scanned start to end.

Re-reading via `idx_buffer` (walking it 6 indices at a time — two triangles per `add_rect_filled`
quad — and looking up each first index's vertex color) gives the correct paint order:

```
frame 0: A->ch2 B->ch3 C->ch4 (ascending, matches submission)
  vtx_buffer order (submission order):     [red@x=0, green@x=20, blue@x=40]
  idx_buffer/paint order (channel order):  [red@x=0, green@x=20, blue@x=40]
frame 1: A->ch4 B->ch3 C->ch2 (reversed, opposes submission)
  vtx_buffer order (submission order):     [red@x=0, green@x=20, blue@x=40]
  idx_buffer/paint order (channel order):  [blue@x=40, green@x=20, red@x=0]
```

In frame 1 the paint order is `[C, B, A]` — exactly the channel-index order (`ch2, ch3, ch4` →
`C, B, A`), the reverse of submission order. This is the correct, positive result: **paint order
follows channel index, not call order**, and it does so on every frame regardless of which order
the calls happened in that frame. A second probe (`probe_channels_final.py`) confirmed the same
holds with a 5-channel split and a single quad moved between channel 2 and channel 4 across three
frames (submitted BEFORE the fixed per-channel quads every time, to rule out "last write wins"):
paint order placed the moving quad at exactly the channel-index position expected, stable across
all three frames. This is the guarantee the 5-channel scheme above depends on and it holds.

## False trails

- **ocornut/imgui's `imgui_demo.cpp` "Custom rendering" canvas example**, named in the brief as a
  reference, was not cloned or read as a separate source: its entire technique (an
  `InvisibleButton` sized to the canvas, `IsItemHovered`/`IsItemActive` for pan/drag,
  `io.MouseDelta` for the pan offset, `PushClipRect` to bound drawing) is already exactly what
  `_draw_canvas`'s background-button/pan/rubber-band code does, function-for-function. Cloning it
  would have re-derived a pattern already implemented and correct; it is folded into CONVERGENCE
  instead of given its own row.
- **imgui-node-editor's Python binding as a drop-in for wire hit-testing or channel machinery**:
  looked promising (it's already a dependency-free import via imgui-bundle, already installed in
  `.venv`), but the `.pyi` stub confirms it exposes only the high-level opinionated node/pin/link
  API plus read-back queries (`get_hovered_node/pin/link`, `get_current_zoom`) — none of
  `FindNodeAt`/`FindLinkAt`/`BringToFront`/the channel constants are reachable from Python. Using
  it would mean replacing `pass_graph.py`'s entire node/port/wire model with imgui-node-editor's
  own `NodeId`/`PinId`/`LinkId` system, not borrowing one mechanism — out of scope for what
  question 16 asked and a much larger change than the maintainer's "the graph editor... I don't
  like the current implementation" framing calls for (a polish pass, not a library swap).
- **litegraph.js and xyflow**, present in the shared scratchpad's `refs/` from a sibling
  researcher's clone, were not read for this area — both are DOM/canvas-2d or SVG based (not
  Dear ImGui / immediate-mode draw-list), so they answer area A/B/D questions about wire routing
  and UX conventions but have no machinery relevant to question 16's ImGui-specific hit-testing/
  channel/zoom/drag-threshold mechanics.
