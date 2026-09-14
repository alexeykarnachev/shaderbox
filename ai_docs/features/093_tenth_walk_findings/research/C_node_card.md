# C — Node card layout and sizing

Research for 093 area C: questions 9-11 of `ai_docs/features/093_tenth_walk_findings/02_research_brief.md`.
Scope is the card's own geometry (width, height, picture, label overflow) — not wire routing (A),
mouse schema (B), or hover feedback (D), covered by other areas' reports.

## Sources

| Name | What it is | URL / path | Read as |
|---|---|---|---|
| thedmd/imgui-node-editor | Dear ImGui node-editor library; blueprints-example is its flagship sample | `examples/blueprints-example/blueprints-example.cpp`, `imgui_node_editor.h` (cloned to scratchpad) | code |
| Nelarius/imnodes | Minimal Dear ImGui node-editor extension | `imnodes.h`, `imnodes.cpp` (cloned) | code |
| Fattorino/ImNodeFlow | Dear ImGui node-editor with per-node/per-pin style objects | `include/ImNodeFlow.h`, `src/ImNodeFlow.inl` (cloned) | code |
| xyflow/xyflow | React/Svelte node-graph library (React Flow / Svelte Flow) | `packages/react/src/components/Nodes/DefaultNode.tsx`, `packages/system/src/styles/style.css` (cloned) | code |
| jagenjo/litegraph.js | Canvas2D JS node-graph library (used by ComfyUI and others) | `build/litegraph.core.js` (`LiteGraph.NODE_*` constants, `LGraphNode.prototype.computeSize`) (cloned) | code |
| Blender | `blender/blender` repo, node editor | `source/blender/editors/space_node/node_intern.hh`, `node_draw.cc`, `DNA_node_types.h` (GitHub mirror, fetched) | code |
| Unreal Engine | Epic dev docs, Blueprint node anatomy/connections pages | `dev.epicgames.com/documentation/...anatomy-of-a-blueprint-in-unreal-engine`, `...connecting-nodes-in-unreal-engine` | docs (no numeric spec found) |
| TouchDesigner | Derivative docs, OP Viewer / node viewer pages | `docs.derivative.ca/OP_Viewer_TOP`, `docs.derivative.ca/Viewer` | docs (no numeric spec found) |
| Adobe Substance 3D Designer | Interface overview / node reference docs | `experienceleague.adobe.com/.../nodes-reference-for-substance-graphs` | docs (no numeric spec found; interface-overview page returned 403) |
| ShaderBox itself | The app's existing content-driven answer for a long label | `shaderbox/ui_primitives.py::_ellipsize`, `::preview_cell` | code |

Houdini is dropped — see False trails.

## Q9 — card widths/heights, font sizes, long-label handling

| Reference | Width rule | Height rule | Font size | Long-label handling |
|---|---|---|---|---|
| imgui-node-editor (blueprints-example) | **Auto**: no width constant anywhere in the sample; node content is laid out with `ImGui::BeginHorizontal`/`Spring()` (a flex-row helper) and the node's rect is the content's item rect plus `Style.NodePadding` (default `(8,8,8,8)`, `imgui_node_editor.h`) | Auto, from content rows | App-default ImGui font; pin icon fixed `m_PinIconSize = 24` px (`blueprints-example.cpp`) | No ellipsis in the sample — a long label simply widens the node, because width is derived from it |
| imnodes | **Auto**: `node.Rect = GetItemRect(); node.Rect.Expand(node.LayoutStyle.Padding)` (`imnodes.cpp`) — the rect wraps a `BeginGroup`/`EndGroup` around the title bar + every attribute row | Auto, same mechanism | App-default ImGui font; `PinCircleRadius = 4.f` (default style, `imnodes.cpp`) | No ellipsis anywhere in the library — same as above, the node grows |
| ImNodeFlow | **Auto**: `m_size = ImGui::GetItemRectSize()` after `BeginGroup()`/`EndGroup()` around the header + body (`ImNodeFlow.inl`); `NodeStyle::padding = (13.7, 6, 13.7, 2)` LTRB, corner radius e.g. 6.5 for the cyan preset (`ImNodeFlow.h`) | Auto | App-default ImGui font | No ellipsis; auto-width absorbs any label length |
| xyflow (React/Svelte Flow) `DefaultNode` | **Fixed**: `width: 150px` (`.xy-flow__node-default`, `style.css`) | Auto (`padding: 10px` around the label, no fixed height) | `font-size: 12px` | No `text-overflow`/`white-space: nowrap` rule on the default node class — a long label **wraps** onto more lines inside the fixed-width box rather than clipping |
| litegraph.js | **Auto to content, with a floor**: `LGraphNode.prototype.computeSize` sets `size[0] = max(input_width + output_width + 10, title_width, LiteGraph.NODE_WIDTH)`, and `× 1.5` if the node has widgets (`litegraph.core.js`); `NODE_WIDTH = 140`, `NODE_MIN_WIDTH = 50` | `size[1] = slot_start_y + rows * NODE_SLOT_HEIGHT`; `NODE_TITLE_HEIGHT = 30`, `NODE_SLOT_HEIGHT = 20` | `NODE_TEXT_SIZE = 14` | No ellipsis needed — width is computed from the longest of title / summed input+output label widths, per node, every time size is (re)computed |
| Blender node editor | **User-resizable, per node**: `bNode.width` is a float field the user drags (`DNA_node_types.h`: "Custom width and height controlled by users"); `NODE_WIDTH(node)` macro is `node.width * UI_SCALE_FAC`, not a constant (`node_intern.hh`) | Mostly auto: `NODE_DY = U.widget_unit` (the 20px-at-100%-UI-scale row height), height is computed from row count | Header/labels use the UI's default font at the panel's UI scale; socket radius `NODE_SOCKSIZE = 0.25 * U.widget_unit` (5px at 100%) | The header/panel label is drawn through `uiDefBut(ButtonType::Label, …, width = draw_bounds.xmax - draw_bounds.xmin - left_padding, …)` — a real button widget clipped to that width, i.e. Blender's normal label-clip/ellipsis behavior applies; it does not grow the node |
| Unreal Blueprint | Content-driven (from the docs and public screenshots: node width visibly grows with the longest pin name or the node title) | Content-driven, one row per pin | Not documented numerically | Not documented numerically; from public screenshots, pins keep short pin names in full and the node widens — no visible ellipsis in the common case |
| TouchDesigner | Fixed small tile with an optional live "cook" thumbnail overlay (`docs.derivative.ca/Viewer`, `OP_Viewer_TOP`); no pixel numbers published | Fixed tile | Not documented numerically | Long operator names visibly truncate in the network-pane tile per the docs' screenshots; no numeric spec |
| Substance Designer | Square node = live thumbnail; a size dropdown scales the whole graph's thumbnails together (from search results on the interface; the interface-overview page itself returned HTTP 403 to a direct fetch, so this line is a secondary citation, not the primary page) | Thumbnail height + one name line below | Not documented numerically | Not documented numerically |
| ShaderBox's own `preview_cell` (strip tile) | Fixed: `SIZE.PASS_TILE = 168` outer, `SIZE.PASS_THUMB = 112` thumb (`theme.py`) | `cell_h = cell_w + footer_h + chips_h` (`ui_primitives.py::preview_cell`) | Footer font is caller-supplied (the strip passes its own name font) | **Ellipsis**: `label = _ellipsize(footer, avail.x)` (`ui_primitives.py`) — binary-searches the longest prefix + `"..."` that fits `calc_text_size`, the app's one existing precedent for a long label on a fixed-width tile |

## Q10 — picture placement, port placement

| Reference | Picture / preview | Port dot placement | Port label placement |
|---|---|---|---|
| imgui-node-editor blueprints | No thumbnail concept (blueprint nodes are logic, not media) | `ax::Widgets::Icon`, 24×24, at the left/right edge of each pin row, drawn via the pin's own horizontal layout group | Beside the icon, inside the node, drawn as normal text in the row |
| imnodes | No thumbnail | `PinCircleRadius = 4.f`, drawn at the pin's attribute-row anchor on the node's left/right edge, offset by `PinOffset` (default 0) | Inside the node, beside the circle, part of the same `BeginInputAttribute`/text/`EndInputAttribute` group |
| ImNodeFlow | No thumbnail | Pin circle (`PinStyle::socket_radius`, e.g. 4.f) on the node's left/right edge | Inside the node, next to the socket, same content group that drives auto-width |
| xyflow | No thumbnail | `.xy-flow__handle`: 6×6px circle, absolutely positioned on the node's border (top/bottom or left/right per `Position`) | No text label by default — `Handle` is just the dot; a labeled variant is left to the consumer's own node component |
| litegraph.js | No thumbnail (it is a logic/data graph, not media-first) | One dot per slot row on the left (inputs) / right (outputs) edge, row height `NODE_SLOT_HEIGHT = 20` | Beside the dot, inside the node, in the same row whose width fed `computeSize` |
| Blender | No general thumbnail (image/texture nodes have a dedicated large preview sub-panel, not a small always-on picture) | Socket circle (`NODE_SOCKSIZE`) sits ON the node's left/right border, half in/half out | Inside the node body, in the socket's row, left-aligned for inputs / right-aligned for outputs |
| Unreal Blueprint | No thumbnail | Pin icon on the node's edge, color-coded by type (docs: "pins are color coded to reflect the type of connection") | Beside the pin icon, inside the node |
| TouchDesigner | **Yes** — the node IS a live preview by default in most panes (`docs.derivative.ca/Viewer`); this is the strongest "picture-first" precedent among the references | Small connector nubs on the node's top/bottom edge (network view), not inside a labeled row | No persistent per-connector text label in the compact network view; the full name reads on hover/selection |
| Substance Designer | **Yes** — the node's body IS a live square thumbnail of that node's output, name below it | Small port triangles/dots on the node's left/right edge | Not shown inline in the compact view (tooltip/selection reveals full port names); closest analog to ShaderBox's picture-first, name-below card |
| ShaderBox today | `GRAPH_THUMB = 80` square, centered under the top padding, above the name (`_thumb_rect`, `pass_graph.py::_draw_node`) | `GRAPH_PORT_R = 4` dot on the node's LEFT edge only (inputs); outputs are dots on the right edge of the picture, not the card (`_port_point`, `_out_point`) | Beside the dot, inside the card, one 16px row per port (`GRAPH_PORT_ROW`), drawn with no ellipsis (`_draw_node`'s `dl.add_text(..., label)` — plain, unclipped) |

## Q11 — auto-width vs fixed width, and the fit-to-view trade-off

**What the references do.** Every Dear-ImGui-based editor (imgui-node-editor, imnodes, ImNodeFlow) computes node width from an `ImGui::BeginGroup()`/`EndGroup()` around the actual content and adds a fixed padding — there is no width constant to tune in any of the three. litegraph.js is a hybrid: auto-to-content with a floor (`NODE_WIDTH = 140`) and a widget multiplier (`×1.5`). Blender lets the USER resize each node and only floors/auto-fits the height. xyflow is the one reference with a genuinely fixed width (`150px`) and no per-node override in the default node — it accepts wrapping instead.

**The trade-off for a canvas whose fit-to-view zoom is already under 1.** Auto-width-to-content (the majority answer among the closest analogs — imgui-node-editor, imnodes, ImNodeFlow, litegraph's floor-plus-grow) makes each node exactly as wide as it needs to be, so short names/ports (which is most of what ShaderBox ships today — see the measurement below) stay compact and only a genuinely long label costs width. A single fixed width sized for the longest name (xyflow's approach) either clips/wraps short names' surrounding padding wastefully or, if sized for the WORST case, makes every node — including `df`/`jfa` one-letter-shorter nodes — pay the same tax, which widens the six-node chain and pushes fit-to-view further below 1. Per-node auto-width is the better fit for a canvas that is already zoom-constrained; a single FIXED width close to what the references converge on (blueprint/imnodes/ImNodeFlow padding + typical content, litegraph's 140, xyflow's 150) is the fallback if auto-width is more implementation than this wave wants, and is what is recommended below given ShaderBox's node is drawn on a raw `ImDrawList` (no `BeginGroup` layout pass to piggyback on) — auto-width would mean computing `calc_text_size` for the name and every port label before the draw, which is a bigger, separately-scoped change than a wave-sized number bump.

## CONVERGENCE

- **No reference besides ShaderBox's own `preview_cell` and, per its docs, Substance Designer's node ellipsizes a node's own title/pin text down to fit a box smaller than the content** — every Dear ImGui node editor and litegraph.js instead SIZES THE BOX to the content (auto-width), and xyflow instead lets the text WRAP. Clipping a name silently, with no affordance to see the rest, is not what any of the closest analogs do.
- **A small fixed circle (4-4.67px radius, ~3-4px BOTH in imnodes' `PinCircleRadius=4.f` and ImNodeFlow's `socket_radius=4.f`, or xyflow's 6×6px square-ish handle) is the converged port-dot size** — ShaderBox's `GRAPH_PORT_R = 4` already matches this exactly.
- **Padding around content converges near 6-13px**: imnodes 8px, imgui-node-editor 8px, ImNodeFlow ~13.7/6px LTRB, Blender's row height (`NODE_DY`) is the 20px widget unit. ShaderBox's `GRAPH_PAD = 6` sits at the tight end of this band.
- **A row-per-port with one dot + one label is universal** among every non-thumbnail reference (imgui-node-editor, imnodes, ImNodeFlow, litegraph.js, Blender, Unreal) — ShaderBox's per-port row is the standard shape, not an outlier.
- **A live-thumbnail-first node body is rare but has real precedent**: TouchDesigner (network-pane live preview) and Substance Designer (square thumbnail + name below) are the two references that put a picture where ShaderBox puts one, and Substance's "thumbnail, then name below, then ports separately" shape is the closest analog to ShaderBox's node overall.

## DIVERGENCE

- **Fixed vs. auto width splits cleanly by rendering substrate.** Every Dear ImGui-on-immediate-mode reference (imgui-node-editor, imnodes, ImNodeFlow) auto-sizes because `BeginGroup`/`EndGroup` makes it nearly free in that framework. xyflow (a retained-mode DOM/CSS renderer) is fixed because CSS boxes default to a fixed inline width and wrapping is cheap there. litegraph.js (canvas2D, so neither framework gives it layout for free) chose to measure text itself (`compute_text_size`) rather than accept clipping — it pays the measurement cost explicitly instead of adopting a fixed box. ShaderBox is also raw-canvas (an `ImDrawList`), so litegraph.js's "measure then decide" posture is the nearest architectural sibling, not the Dear-ImGui-group trick (ShaderBox's node draw has no `BeginGroup` layout pass).
- **Thumbnail-first vs. logic-first is a domain split, not a taste split.** Data/logic graphs (Blueprint, imnodes/imgui-node-editor's own examples, litegraph.js) never show a picture because there usually isn't one cheap to render per node; texture/compositing tools (TouchDesigner, Substance Designer) do, because a live per-node preview IS the point of authoring visually. ShaderBox is in the second camp by its own domain (`document_examples` passes ARE images), so the picture-first shape is domain-correct, not a stylistic choice to reconsider.

## Recommended for ShaderBox

A coder-implementable geometry at zoom 1, changing only the `SIZE.GRAPH_*` tokens in `shaderbox/theme.py` (consumed by `node_size` in `shaderbox/widgets/graph_state.py` and the draw math in `shaderbox/widgets/pass_graph.py`) — **fixed width**, per the substrate argument above (no `BeginGroup`-style auto-layout exists on this draw list; true auto-width is a separately-scoped change, not a wave-sized number bump):

| Token | Current | Recommended | Why |
|---|---|---|---|
| `GRAPH_NODE_W` | 108 | **136** | +26%, "a little bit larger, not super large" per the maintainer's own words; sits between litegraph's `NODE_WIDTH=140` and the imgui-family's small-padded-content nodes; leaves `u_distance_field` (104.8px at 12px monospace, computed below) inside the card with room for the dot and padding |
| `GRAPH_THUMB` | 80 | **96** | Grows with the card, stays centered, keeps the same `GRAPH_PAD` margin on both sides (136 − 2×8 pad ≈ 120, thumb 96 leaves 12px breathing room each side vs. today's 108 − 2×6 = 96, thumb 80 leaves 8px — a proportionally similar margin) |
| `GRAPH_PAD` | 6 | **8** | Matches imnodes' and imgui-node-editor's converged 8px padding exactly |
| `GRAPH_NAME_H` | 18 | **20** | Headroom for `font_14_bold` at the (unchanged) 14px name size; keeps the same ~2px slack the current 18px gives a 14px font |
| `GRAPH_PORT_ROW` | 16 | **18** | A hair more breathing room per port row; still well under Blender's `NODE_DY` (20px widget unit) and litegraph's `NODE_SLOT_HEIGHT` (20px), which both this and the current 16 sit under |
| `GRAPH_PORT_TOP` | 4 | 4 (unchanged) | No reference gives a reason to change the name-to-first-port gap |
| `GRAPH_PORT_R` | 4 | 4 (unchanged) | Already matches the converged 4-4.67px reference radius; the maintainer's complaint was never about the dot |
| `GRAPH_ROUNDING` | 6 | 6 (unchanged) | Sits inside the converged 3-12px band (xyflow 3, imnodes 4, ImNodeFlow 6.5, imgui-node-editor 12); no reason to move it |
| Name overflow | none (unclipped) | **`_ellipsize` at `_draw_node`'s name draw**, budget = `(p1.x - p0.x) - 2*GRAPH_PAD` in `font_14_bold`, using the app's existing `shaderbox.ui_primitives._ellipsize` (the same function `preview_cell`'s footer already uses) | No reference silently overflows text past the card edge the way ShaderBox does today; ellipsis-on-overflow is the app's own established answer (`preview_cell`), so this is the same fix, not a new one |
| Port-label overflow | none (unclipped) | **`_ellipsize`** at each port-label draw in `_draw_node`, budget = `(node width) - (2*GRAPH_PORT_R + label_x_gap) - GRAPH_PAD` | Same reasoning; a long sampler name should clip with `...`, never draw past the card's right edge into the canvas |

**Geometry at zoom 1 with these numbers** (`node_size(port_count=1, box=False)`, `shaderbox/widgets/graph_state.py::node_size`): width **136px**, height `8 + 96 + 20 + 8 + 4 + 1×18` = **154px** (vs. today's 108×120 for one port). A no-port node (a source pass) is `8+96+20+8` = **132px** tall (vs. today's 100).

**Fits the shipped examples without ellipsis.** Measured with `fontTools` against the actual font ShaderBox ships (`shaderbox/resources/fonts/Anonymous_Pro/AnonymousPro-Regular.ttf` / `-Bold.ttf`, confirmed monospace: advance width = 0.545898 × em for both weights) — the longest real pass name across every `shaderbox/resources/document_examples/*/passes/*.glsl` is `composite` (9 chars, 68.8px at 14px bold) and the longest real sampler is `u_cascade` (9 chars, 59.0px at 12px). Both fit inside the recommended 136px card (with ~104px and ~120px of label budget respectively, after padding and the port dot) with no ellipsis triggered — today's 108px card already fits them too; the brief's `distance_field` / `u_distance_field` clipping example is illustrative (from `00_mock_panel.html`'s mock data, not a shipped asset), not a defect against the current example set. The ellipsis rule above is still worth adding, because it guards the case a future or user-authored pass DOES have a longer name — today's silent overflow-past-the-card-edge is real and unguarded (`_draw_node` calls `dl.add_text` with no width check on either the name or the port-label draw), it is simply not yet triggered by the shipped examples.

**Fit-to-view trade-off, six-node chain** (`GRAPH_GAP_X = 64`, unchanged — no reference or finding argues for changing the gap):

| Width | Chain width (6 × W + 5 × 64) | Fit zoom, 740px panel | Fit zoom, 1225px panel |
|---|---|---|---|
| 108 (current) | 968 | 0.764 | 1.265 |
| **136 (recommended)** | **1136** | **0.651** | **1.078** |
| 150 (xyflow's fixed width) | 1220 | 0.607 | 1.004 |
| 160 | 1280 | 0.578 | 0.957 |

At 136px the six-node chain still fits ABOVE zoom 1 in a 1225px panel (1.078) and the 740px panel's fit zoom drops from 0.764 to 0.651 — a real cost, but the chain was already sub-1 at 108px, so this does not cross a new threshold; it makes an existing "zoomed out a bit" state a bit more zoomed out. 150 (xyflow's number) is the point where the wider panel's fit crosses below 1.0, which is the strongest quantitative argument for stopping short of it — 136 keeps both panel sizes on the correct side of that line while still delivering a size increase the maintainer will see.

## False trails

- **Houdini** — dropped from Q9-11 despite being named in the brief. No official Houdini docs page publishes numeric node-tile dimensions (font size, padding, socket radius), and its node/network editor is a general DAG view rather than a picture-first tile the way TouchDesigner's or Substance's is, so it would have contributed only a qualitative "content-driven, resizable" data point already covered more concretely by Blender and litegraph.js. Spending a fetch cycle confirming that absence was not worth carrying into the table.
- **Adobe's `interface-overview.html`** returned HTTP 403 to a direct fetch (likely bot-blocked); the Substance Designer row above is sourced from the search snippet and the Experience League node-reference pages instead, and is flagged as a secondary citation rather than a primary-page read.
- **Unreal Engine's per-file C++ pin/node layout source** (Slate widgets, `SGraphNode`/`SGraphPin`) was not cloned or fetched — Epic's engine source requires a GitHub account linked to Epic's org to clone, which is out of scope for a research pass; the Unreal row above is qualitative, from public docs pages only, and should not be read as code-verified the way the Dear ImGui and JS references are.
