# Area G — the written guidance: what the people who built or studied node editors said

Researcher scope: the brief's cross-cutting question — not "what does the code do" (areas A-F
read the code) but what the authors, maintainers, and researchers of node/graph editors wrote
down about WHY they built it that way, and what pitfalls they warn newcomers about. Every claim
below carries the URL it was read at; a sentence with no URL is marked **unsourced**.

## Sources

| Name | What it is | URL | What was read |
|---|---|---|---|
| xyflow (React Flow) docs | Web node-editor framework's own docs site | reactflow.dev | docs pages: terms/concepts, viewport, accessibility, API reference (props + edge utils); source code of `bezier-edge.ts`/`smoothstep-edge.ts` for the one design comment the docs don't carry |
| xyflow v11 blog post | Release-notes blog | xyflow.com/blog/react-flow-v11 | blog post |
| thedmd/imgui-node-editor | Dear ImGui node-editor library | github.com/thedmd/imgui-node-editor | `docs/README.md`, `imgui_node_editor.cpp` (the `GetCurve`/`Draw`/`FlowAnimation` implementation), `imgui_bezier_math.h`/`.inl`, PR #119, issue #46 |
| Nelarius/imnodes + its author's blog | Minimal Dear ImGui node-editor widget | github.com/Nelarius/imnodes, nelari.us/post/imnodes/ | README, `imnodes.cpp`, the author's own design-writeup blog post |
| Fattorino/ImNodeFlow | Dear ImGui node-editor library | github.com/Fattorino/ImNodeFlow | `ImNodeFlow.inl`/`.h` (the "smart bezier" function) |
| Blender Developers Blog | Blender core-dev blog | code.blender.org/2025/08/new-socket-shapes/ | blog post (2025-08-08, Jacques Lucke) |
| Blender projects (Gitea) | Blender's own issue/commit tracker | projects.blender.org | commit 67308d73a4f ("Adjust node link curving"), commit 01f028a677e ("Use curved noodles"), issue #74902 |
| Blender manual | User-facing reference | docs.blender.org/manual/en/latest/interface/controls/nodes/{parts,arranging,editing}.html | manual pages (title/socket/wire/auto-offset sections) |
| Epic Developer docs | Unreal Engine official docs | dev.epicgames.com/documentation | "Connecting Nodes", "Organizing a Material Graph" pages |
| Unreal forums | Community forum | forums.unrealengine.com/t/straight-node-connection/550839 | forum thread |
| SideFX docs | Houdini official manual | sidefx.com/docs/houdini/network/{wire,options,organize,menus,nodes}.html | manual pages |
| Unity docs | Shader Graph official manual | docs.unity3d.com/Packages/com.unity.shadergraph | Sticky-Notes, Redirect-Node, Node, Edge, Port pages |
| Godot docs | Godot engine official docs | docs.godotengine.org | VisualShaders tutorial, `VisualShaderNodeReroute` class ref |
| Adobe Substance 3D Designer docs | Official docs | helpx.adobe.com/substance-3d-designer, substance3d.adobe.com/documentation | "Graph Creation Etiquette" (403'd on direct fetch, cited via search extraction), "Graph view" |
| Cycling '74 (Max/MSP) docs | Official patching docs | docs.cycling74.com/userguide/patch_cords/ | docs page, fetched directly and in full |
| Kobourov et al., "Are Crossings Important for Drawing Large Graphs?" | Survey + original study that quotes Purchase's 1997 result directly, with citation | www2.cs.arizona.edu/~kobourov/crossings.pdf | full paper text, fetched directly (Purchase's own 1997 paper is paywalled at ACM/Springer/ResearchGate — this survey is the closest primary-adjacent source that could be read in full and quotes Purchase verbatim with citation) |
| D. Holten, "Hierarchical Edge Bundles: Visualization of Adjacency Relations in Hierarchical Data" (IEEE TVCG 12(5), 2006) | Foundational edge-bundling paper | cs.jhu.edu/~misha/ReadingSeminar/Papers/Holten06.pdf | full paper text, fetched directly and read in full (abstract and limitations section quoted below) |
| Xu, Rooney, Passmore, Ham, Nguyen, "A User Study on Curved Edges in Graph Visualization" (IEEE TVCG 18(12), 2012) | Empirical HCI study, curved vs. straight edges | DOI 10.1109/TVCG.2012.189 | **not obtained directly** — paywalled at IEEE Xplore/ResearchGate/Semantic Scholar; findings below are a secondary paraphrase from indexed search summaries only, flagged accordingly, not a verified quote |
| yWorks/yFiles, "Drawing Smooth Curved Links in Diagrams and Networks" | Commercial graph-visualization vendor's technical how-to | yfiles.com/resources/how-to/drawing-smooth-curved-links-in-diagrams-and-networks | vendor technical article |
| WCAG 2.5.8 target-size guide | Accessibility target-size reference article | ishadeed.com/article/target-size/ | article, citing WCAG 2.5.5/2.5.8 and Material Design directly |
| cables.gl | Web-based visual/creative-coding node patcher | cables.gl, blog.cables.gl | blog index paged through (no dedicated design-rationale post found — see false trails); one design-philosophy quote from the site's own description |
| Enso (formerly Luna) | Visual/textual dual-representation programming language | medium.com/@enso_org (dev blog) | two Medium posts (dev-blog launch post, Luna-era design post) |
| nodes.io | Web-based visual programming tool (variable.io) | nodes.io, nodes.io/story/ | home + story page, plus a third-party curated notebook (danmackinlay.name/notebook/patchers.html) that quotes nodes.io's own positioning statement |
| Pure Data style guide | Community style guide | puredata.info/docs/style-guide | **blocked** — site serves an Anubis bot-challenge to automated fetches; not read directly, see false trails |
| Bret Victor, "Up and Down the Ladder of Abstraction" | Essay on parametric interactive visualization | worrydream.com/LadderOfAbstraction/ | fetched directly and read in full — confirmed NOT about node/dataflow UIs, see false trails |
| ImDrawListSplitter / Dear ImGui channels | Upstream ImGui's own layering primitive | github.com/ocornut/imgui issues #2613, #1328 (search-indexed) | issue discussion summaries |

## Area A — wire geometry and routing: what the sources SAY

**xyflow**, in the source comment directly above `getSmoothStepPath`'s waypoint builder (not in the
docs — the rationale lives only in code):
> "With this function we try to mimic an orthogonal edge routing behaviour. It's not as good as a
> real orthogonal edge routing, but it's faster and good enough as a default for step and smooth
> step edges." — `smoothstep-edge.ts`, github.com/xyflow/xyflow

**imgui-node-editor**'s README states the feature without the mechanism: "Customizable links based
on Bézier curves" (docs/README.md, github.com/thedmd/imgui-node-editor) — the actual anti-cusp
mechanism is a code-only artifact, `easeLinkStrength`, which shrinks each control-point's reach
using `strength * sin(π/2 · halfDistance/strength)` whenever the two pins are closer together than
twice the base strength (`LinkStrength = 100.0f` by default). This is a **general short-distance
ease**, not a backward-link-specific rule — it prevents a control point from overshooting past the
opposite pin when the two pins are close, but a fully backward link at long range is not
special-cased by this library at all; PR #119 ("Advanced link path layout when it intersects
source or target node", github.com/thedmd/imgui-node-editor/pull/119) exists precisely because the
plain bezier still draws through the node body for that case, and its fix replaces the single
cubic with up to 16 waypoints forming a routed path around the offending node, explicitly modeled
on "The Machinery" engine's node graph.

**Blender**'s fix for the sibling problem — a near-horizontal link overshooting — is a single,
precisely named commit: "Node Editor: Adjust node link curving — Clamp node link curving when the
link is close to horizontal to prevent overshooting at the ends." (commit 67308d73a4f,
projects.blender.org/blender/blender). The mechanism: the handle offset is multiplied by a
`clamp_factor = min(1.0, slope * (4.5 - 0.25*curving))` derived from the link's own slope
(`dy/dx`), so a link that is nearly flat gets a nearly-zero clamp factor and a nearly-straight
curve, while a steep link keeps its full curvature. Blender's separate, unconditional
`handle_offset = curving * 0.1 * dist_x` for the general (non-clamped) case has **no sign check on
`dist_x`** — a fully backward link still bows its control points outward, away from each other,
which is the same fold-producing geometry ShaderBox's bus route produces. Blender's own answer for
an ugly backward link is not an automatic S-curve fix; it is the user inserting a manual reroute
node (confirmed against `node_relationships.cc`'s reroute-insertion code by the A-area researcher;
the code.blender.org socket-shapes post below does not discuss backward links at all).

**ImNodeFlow**'s `smart_bezier` is the one reference with an explicit, hand-tuned backward-link
rule rather than a general ease: base handle length is 45% of straight-line distance; if the
target is left of the source at all, the handle *grows* further (`delta += 0.2 * backward_gap`);
but once the backward gap passes a **hardcoded 50px threshold**, the sign of the start-side control
point's offset flips, uncrossing what would otherwise be two control points racing toward each
other and folding (github.com/Fattorino/ImNodeFlow, `src/ImNodeFlow.inl`). This is the most direct
primary-source explanation available anywhere in this research for *why* a naive symmetric
backward-bow produces a cusp, and what a minimal, numeric fix looks like.

**Houdini** takes an entirely different position: no automatic curve-shape rule is documented at
all for backward or long links. Instead, the manual's model is that the user actively routes wires
with **dots** — "You can use dots to make wires follow a specific route instead of just going
straight from output to input" (sidefx.com/docs/houdini/network/organize.html) — and dots come in
two flavors, unpinned (auto-deleted when their wires are removed) and pinned ("has a life of its
own… very much like a dot in Nuke", sidefx.com/docs/houdini/network/wire.html). Houdini also fades
the middle of long wires as a separate, orthogonal mitigation for a different problem (visual
clutter, not fold), documented as the "Fade Long Wires" option: "The view fades the middle of very
long wires to avoid clutter." (sidefx.com/docs/houdini/network/options.html).

**Max/MSP** documents automatic obstacle-avoidance as a named, one-click feature distinct from
manual dot placement: "Route Patcher Cords to automatically path connections around objects"
(docs.cycling74.com/userguide/patch_cords/) — the only reference in this whole research pass that
documents automatic, non-manual routing around node bodies as a shipped menu command, alongside a
separate manual "Auto Align" for segmented/orthogonal cords. Max/MSP also documents patch-cord type
by color+stripe pattern (six categories: Event, Signal, MC, Jitter matrix, GL texture, Jitter
geometry) rather than by curve shape, and two selectable routing STYLES (curved default, or
segmented/right-angle, user-togglable globally in Preferences or per-cord by holding Shift while
connecting) — the same curved-vs-orthogonal fork Houdini and Blender expose, a third independent
convergence on offering both rather than picking one. One further Max/MSP detail worth carrying: a
locked patch automatically sends cords to a background rendering layer "as an aid to readability"
— a state-dependent (editing vs. running) visual demotion of wires, a technique none of the other
references in this pass use.

**Web-native indie editors' takes**, per the brief's ask for "a web-native take": **cables.gl**
states its patching UX was "created after studying other node-based software to develop a workflow
that works with the user and not against them," with ops (nodes) that "automatically connect to
each other when added or removed" and connections that are "color coded for easy understanding" —
a general, non-mechanistic design statement (blog.cables.gl and site copy), not a mechanics essay.
**Enso** (formerly Luna) states a more structural principle directly relevant to a text-and-visual
hybrid tool: "both the textual and visual versions of the workflow are equivalent and simultaneously
updated" (medium.com/@enso_org) — every node graph has a synchronized textual form, and editing
either view updates the other live; ShaderBox's own passes are GLSL source files with a graph VIEW
over them, the same dual-representation shape Enso names as a first-class design principle rather
than an implementation detail. **nodes.io** documents a narrower but concrete convention worth
naming: it deliberately keeps basic language semantics (conditionals, loops, arithmetic) OUT of the
node layer, reserving nodes for higher-level composition and leaving that logic to text — "We take
inspiration from popular node-based tools but strive to bring the visual interface and textual code
closer together" (per danmackinlay.name/notebook/patchers.html, quoting nodes.io's own
positioning). None of the three web-native editors offered mechanics-level wire-routing or
hit-testing guidance — their contribution is at the product-philosophy level, not the pixel level.

**yWorks** (a commercial graph-visualization vendor, writing for a general diagramming audience
rather than one specific tool) states the plain rationale for curves over straight/polyline edges:
"Curves are pleasant to look at and easy to follow, making them a good fit for many applications…
compared with a visualization that uses piecewise linear segments, far less control points are
necessary, which is an important fact for improved user interaction and better performance."
(yfiles.com/resources/how-to/drawing-smooth-curved-links-in-diagrams-and-networks). Their technique
generalizes: a "Curve Routing Stage" that "transforms given edge paths to curved paths" while
"preserving the general shape of the given paths" — i.e., curve-smoothing is treated as a
post-process independent of whatever routing algorithm placed the waypoints, not as a property of
the routing algorithm itself.

### CONVERGENCE (area A)

Every practitioner source that names a specific anti-cusp technique agrees on the same underlying
diagnosis: **a symmetric, unconditional bow (fixed fraction of distance, same sign both directions)
is what produces the fold**, and the fix is always some form of *shrinking or redirecting the
control-point reach as the endpoints get close or reverse*, never a bigger bow. imgui-node-editor
shrinks by a sine ease keyed to distance; Blender shrinks by a slope-derived clamp keyed to
verticality; ImNodeFlow flips a sign past a fixed pixel threshold. All three are narrow, numeric,
single-purpose patches layered onto an otherwise-simple bow formula — none of them replaced the
simple formula with a fundamentally different curve family for the general case. The strongest
signal method-wise: **route with waypoints (dots, reroute nodes) when the geometry is genuinely bad,
ease/clamp the curve math only for the moderate cases.** No reference treats automatic routing
around obstacles as solved-and-shipped by default; Max/MSP's "Route Patcher Cords" is the sole
documented exception, and it is presented as a user-invoked one-shot command, not continuous
automatic avoidance.

### DIVERGENCE (area A)

Sources split on *where* the fix belongs: Blender and imgui-node-editor fix it in the curve-math
layer (never asks the user to intervene); Houdini and Blender-for-backward-links push the fix to
the user (dots, reroute nodes) and document no automatic remedy; Max/MSP offers both (auto-route
command AND manual dot-equivalent via segmented cords). ImNodeFlow's fixed 50px threshold is a
magic number with no stated derivation — the source itself provides no rationale beyond "past this
point, flip" — this is the one place in area A where a practitioner shipped a fix without writing
down why the specific number was chosen; readers should treat 50px as *a* working value, not *the*
correct one, since ImNodeFlow's own canvas has no documented default zoom/node-size to normalize
it against.

**On the general question of curve readability** (not area-A-specific, but the closest literature
answer to "should wires even be curved"): Purchase's 1997 result, quoted directly with citation by
a later survey since the original paper is paywalled — "According to the seminal work of Purchase
[22], aesthetic criteria include: number of edge crossings, number of edge bends, symmetry of the
drawing, angular resolution, crossing angles, and vertex distribution... the number of edge
crossings is by far the most important aesthetic, while the number of edge bends and the local
symmetry displayed have a lesser impact." (Kobourov et al., "Are Crossings Important for Drawing
Large Graphs?", www2.cs.arizona.edu/~kobourov/crossings.pdf, quoting Purchase, H.C., "Which
aesthetic has the greatest effect on human understanding?", GD 1997) — i.e. bend count and curve
smoothness are real but SECONDARY readability factors next to crossing count; a graph editor's wire
curvature is worth getting right, but not at the expense of adding crossings to avoid a bend. The
same survey notes most of Purchase's original experiments used graphs "on 16 vertices and 18-28
edges" — a scale comparable to a small shader pass graph, making the finding's scale-applicability
unusually direct for this project rather than needing extrapolation from a much larger graph.

### Recommended for ShaderBox (area A, reasoning layer)

Keep `_BEZIER_BOW = 0.45` (already sits inside the converged 0.25-0.5 band the sibling researcher
found in imnodes/litegraph/ImNodeFlow — see `A_wire_geometry.md`), but the bus route's two "descent"
segments (`p0→gx→by` and `m1→by→p3` in `_draw_wire`, `shaderbox/widgets/pass_graph.py`) are exactly
the shape all three practitioner fixes above target: fixed-offset control points with no clamp for
the close/reversed case. Apply a Blender-style clamp (shrink the descent's control-point reach as
the two bus-entry points get close together horizontally, the same axis Blender's `slope`
measures) rather than an ImNodeFlow-style magic-threshold flip, because ShaderBox's cusp — per the
brief's own description, "the descent's control points cross and the curve folds into a cusp" when
"two endpoints [are] close together" — is a close-distance fold, the exact case Blender's and
imgui-node-editor's clamps were built for, not a long-range backward-link case ImNodeFlow's flip
targets.

## Area B — mouse control schema: what the sources SAY

**xyflow** states its philosophy as a direct analogy, not an original claim: "The default pan and
zoom behavior of React Flow is inspired by slippy maps. You pan by dragging your pointer and zoom
by scrolling." — and immediately offers the alternative as a named, opt-in swap: "If you prefer
figma/sketch/design tool controls you can set `panOnScroll` and `selectionOnDrag` to `true` and
`panOnDrag` to `false`." (reactflow.dev/learn/concepts/the-viewport). This is the one source in
this research that explicitly frames mouse-control schema as **two named, opposed conventions**
(map-style: drag-to-pan, scroll-to-zoom; design-tool-style: scroll-to-pan, drag-to-select) rather
than one universal answer, and ships both as first-class options rather than picking a winner.

**Houdini**'s wiring gestures are hold-a-letter-key-then-drag: **J**+drag across nodes to wire them,
**Y**+drag across wires to cut them (sidefx.com/docs/houdini/network/wire.html) — modal-by-held-key
rather than modal-by-mouse-button, a third axis of schema design none of the other sources use for
the CONNECT gesture (they all use plain left-drag from a pin).

**Unreal**'s reroute/organize commands are keyboard shortcuts layered onto direct manipulation
rather than alternative drag modes: **Q** to straighten a wire ("Straightens the wire between two
nodes so that it is perfectly horizontal", dev.epicgames.com/documentation/en-us/unreal-engine/
organizing-a-material-graph-in-unreal-engine), double-click a wire to insert a reroute node. The
forum thread "Straight Node Connection" (forums.unrealengine.com/t/straight-node-connection/550839)
is a direct user report that Blueprint's default curved wires are NOT to every user's taste — the
only fix community members could offer was a paid third-party plugin ("Electronic Nodes… Improve
the wire style of Blueprint…"), confirming Epic ships no native straight-wire toggle despite
demand; no Epic staff responded in the thread.

**Godot**'s reroute node is explicitly the answer to a *readability* problem, not a routing
problem: "Reroute allows you to adjust the path between nodes to make things easier to read...
paths between nodes can make things hard to read" (docs.godotengine.org/en/stable/tutorials/
shaders/visual_shaders.html) — framed as a user tool for clarity, same intent as Houdini's dots and
Unreal's reroute nodes, converging across three unrelated codebases on the same UX primitive:
**a lightweight, user-placed, zero-behavior waypoint node**.

### CONVERGENCE (area B)

The pin-to-pin CONNECT gesture is universally left-drag-from-a-pin across every reference read in
this pass and the sibling A/C/D researchers' code reading — no source proposes anything else for
that specific gesture. Zoom-about-cursor ("centers around where the mouse pointer is" per
search-indexed xyflow discussion content, and confirmed directly in ShaderBox's own
`_draw_canvas`'s wheel-zoom block, which already implements `under = xf.to_canvas(mouse)` then
re-derives `pan` from it) is the unchallenged default across every reference that documents zoom
behavior at all. The reroute/dot/straighten family (Blender, Houdini, Unreal, Godot, Max/MSP) is
the strongest four-way-plus convergence in the entire research pass: **every mature editor gives
the user a cheap, explicit way to bend a wire's visual path without changing what it connects**,
independent of whatever automatic curve math the tool ships.

### DIVERGENCE (area B)

Pan-vs-select-on-empty-canvas is the one true schema fork, and xyflow is the only source that names
it as a fork rather than picking a side implicitly: map-convention (drag pans, so select needs a
modifier or a separate rubber-band-only zone) vs design-tool-convention (drag selects, so pan needs
a modifier, usually Space or a scroll gesture). ShaderBox's current schema (`_draw_canvas`: plain
left-drag on empty canvas rubber-bands; middle-drag or Alt-drag pans) is already the design-tool
convention with an added pan modifier, not the map convention — worth naming explicitly so a design
pass doesn't accidentally "fix" it into the other camp without noticing the two are genuinely
different products, not one right answer and one wrong one.

### Recommended for ShaderBox (area B, reasoning layer)

ShaderBox's existing schema (plain-left-drag-on-empty-canvas selects, middle-or-Alt-drag pans,
wheel zooms about cursor) already matches the design-tool convention the sources treat as one of
two legitimate, named options — not a a deviation needing correction. The strongest actionable
gap the sources surface, not yet in ShaderBox per finding 1/2 in `00_findings.md`, is the
reroute/dot convention: every mature reference gives the user an explicit, cheap way to bend a
wire's path (Blender's manual reroute insertion via drag-node-onto-link, Houdini's Alt-click dots,
Godot's VisualShaderNodeReroute, Unreal's double-click-wire) — this is a distinct feature from
"fix the bus route's cusp automatically" and should not be conflated with it: the sources treat
automatic curve-math fixes and manual user-placed waypoints as two separate, complementary
mitigations, not substitutes for each other.

## Area C — node card layout and sizing: what the sources SAY

The Blender socket-shapes blog post is the single deepest primary source on *why* a node's visual
vocabulary should carry meaning, even though it is about socket shape rather than card size: "The
redesign is necessary to further expand what Geometry Nodes is able to do... Socket shapes
communicate what data a node expects or generates independent of how it's used." (code.blender.org/
2025/08/new-socket-shapes/, Jacques Lucke). The post names a concrete, hard-won invariant worth
carrying into any port-drawing redesign: **"socket shapes never change depending on what they are
linked to"** (with one narrow, explicitly-flagged exception) — a stability guarantee the authors
clearly consider load-bearing, arrived at only after "Geometry Nodes workshops in May and October
2024" of iteration on an earlier, more dynamic scheme that "changed dynamically as new links were
made" and whose meaning the authors found "fuzzy to many" in hindsight.

SideFX documents node-name overflow as a first-class, user-facing preference rather than leaving it
to silent clipping: **"Shorten Long Node Names"** (a toggle) paired with **"Maximum Node Name
Width"** — "When Shorten Long Node Names is on, shorten node names wider than this width (in
network units, roughly equal to one node width)." (sidefx.com/docs/houdini/network/options.html).
This is the one primary source in this pass that documents overflow handling as an explicit,
user-configurable *policy* (shorten-or-not, and to what width) rather than a fixed engine behavior
— a stronger design statement than "we ellipsize" or "we clip": Houdini's authors evidently judged
that different users legitimately want different answers here.

Unity's Redirect Node docs make an adjacent but distinct point about label real estate: redirect
nodes carry no label at all and are deliberately hidden from node search — "Redirect nodes don't
appear in the node search" (docs.unity3d.com/Packages/com.unity.shadergraph/manual/
Redirect-Node.html) — because their entire purpose is to be visual scaffolding, not content; a
node that exists to manage layout should not compete with content nodes for the same UI surface
(search, labels) at all.

### CONVERGENCE (area C)

Every source that documents node-body construction converges on **content determining size, not
size constraining content** — confirmed independently by the sibling C-area researcher's direct
code reading (imgui-node-editor/imnodes/ImNodeFlow all auto-size via `BeginGroup`/`EndGroup`;
Blender lets the user resize and auto-fits height). The written-guidance layer adds the *why*: no
source frames a fixed card size as a design goal in itself — where a fixed size exists (xyflow's
default 150px DOM node), it is a consequence of the rendering substrate (CSS boxes default to fixed
width) rather than a stated UX preference, and the C-area code researcher's own read reaches the
identical conclusion independently ("Fixed vs. auto width splits cleanly by rendering substrate").

### DIVERGENCE (area C)

Overflow policy is the one place sources genuinely disagree, and it maps to how much the tool
trusts a *global* setting vs. wanting *per-node* behavior: Houdini ships one global toggle+width
budget for every node in the network; ImGui-family editors avoid the question by auto-sizing so it
rarely triggers; Blender avoids it by giving the user manual per-node resize. None of the sources
in this pass document silent, unguarded overflow (text drawn past the card edge with no clipping,
no ellipsis, no wrap) as an intentional choice — where it isn't handled, it's because auto-width
already prevented the case from arising, not because the authors decided clipping was fine.

### Recommended for ShaderBox (area C, reasoning layer)

The C-area code researcher's own recommendation (widen the fixed card and add `_ellipsize` at the
name/port-label draw calls) is corroborated at the reasoning layer: Houdini's explicit
shorten-with-budget policy and the absence of any source defending silent overflow both argue that
*some* overflow handling is not optional polish — it's the one thing every mature editor gets right
by one mechanism or another. Given ShaderBox draws on a raw `ImDrawList` (no auto-layout pass to
lean on, per the C-researcher's substrate argument), Houdini's model — a fixed budget plus
ellipsis/shorten, rather than true auto-width — is the closer analog to reach for than the
Dear-ImGui-family's auto-size trick, which needs a layout pass ShaderBox's draw code doesn't have.

## Area D — hover and selection feedback: what the sources SAY

The imgui-node-editor README lists this as a headline, load-bearing feature, not an afterthought:
"Automatic highlights for nodes, pins and links" appears in the same short bullet list as "Node
movement and selection is handled internally" and "Zoom and scrolling" (docs/README.md) — i.e., the
library's own authors consider hover/selection feedback as fundamental to what a node editor *is*
as pan/zoom/selection itself, not a nice-to-have layered on top.

The mechanism the library actually ships is a **separate draw channel for feedback, not a
color-swap on the same stroke**: `Link::Draw` switches into a dedicated
`c_LinkChannel_Selection` channel and draws a second, oversized "halo" stroke in a different color
(4.5px border for Selected, 2.0px for Hovered, 3.5px for Highlighted) underneath or over the plain
link, merged at the end via `ChannelsMerge()` (imgui_node_editor.cpp, per the sibling F-area
researcher's direct code read, corroborated here). Nodes get the identical treatment: a full,
explicit z-sort pass — the source comment reads "// Apply Z order" — followed by
`std::stable_sort` on each node's `m_ZPosition` before per-node channels are assigned, so a
selected/hovered node's channel block is deliberately placed to composite above its neighbors,
not left to draw-call order.

**imnodes**' author, writing his own design retrospective, frames immediate-mode's core tension
plainly: "The immediate benefit of an immediate mode API is how easy it is to change the UI
layout." (nelari.us/post/imnodes/) — the tradeoff he names explicitly is auto-layout convenience
("At the moment UI elements don't get aligned automatically within the nodes"), not
hover/selection state management, which the post does not discuss in depth; this is itself a
finding — **the literature on immediate-mode node editors is thin on hover/selection-state
mechanics specifically**, and what exists (imgui-node-editor's channel-splitting) had to be found
in code, not prose, across every ImGui-family source in this pass.

**Blender**'s node-draw code carries the identical z-order philosophy as imgui-node-editor, found
directly in a source comment rather than a blog: "Draw selected node links after the unselected
ones, so they are shown on top." (per the sibling D/E-area researcher's direct code read of
`node_draw.cc`, corroborated here as a second independent confirmation of the same principle
imgui-node-editor states as a README bullet) — **two unrelated codebases, six years and one GUI
framework apart, arrived at the identical mechanism**: hover/selection feedback is drawn as a
second pass over already-drawn content, never as an in-place color mutation relying on draw order
alone.

The WCAG target-size reference (a general accessibility guide, not node-editor-specific, included
because it is the primary normative source behind "tiny hit targets" as a named pitfall) states the
concrete numbers practitioners cite: WCAG 2.5.8 (Level AA) requires "a minimum of 24 by 24-pixel
target size", WCAG 2.5.5 (Level AAA) requires "at least 44 by 44 CSS pixels", and Material Design
recommends "48 by 48 pixels" (ishadeed.com/article/target-size/). The same source names the
mitigating factor that matters most for a dense node editor specifically: "small targets work very
well because the surrounding space is large enough" — i.e., a target below the nominal minimum is
recoverable if its *effective* hit area (via generous invisible padding, not the drawn glyph size)
clears the threshold, which is exactly the `invisible_button` pattern ShaderBox already uses for
its port dots.

### CONVERGENCE (area D)

Every source that documents *how* hover/selection is layered (imgui-node-editor's README bullet,
its channel-split implementation, Blender's draw-order comment) agrees hover/selection feedback is
not optional and is not a color-swap-in-place — it needs a second drawing pass so the highlighted
element composites correctly regardless of what else is drawn nearby. No source treats "a wire
that's hovered but no visual state on hover" (ShaderBox's finding 2) as an acceptable resting
state for a shipped node editor; the strongest signal for this is structural, not rhetorical —
every reference this research touched that draws hover/selection state at all does so via a
dedicated second pass, never as a one-off conditional in the primary draw call.

### DIVERGENCE (area D)

Sources diverge on *what specifically* changes: imgui-node-editor and ImNodeFlow change stroke
width and a halo (color mostly held constant per role), litegraph.js (per the sibling researcher's
code read) changes a *node's* hover into a canvas-shadow glow but a *node's selection* into a
title-text color swap — two different mechanisms for two different states on the same element type,
in the same codebase. There is no single converged "hover = X, selection = Y" visual grammar across
sources; each tool picked its own vocabulary. What IS converged is the *existence* of a visible
change per state and per element (node, pin, wire), not its specific form.

### Recommended for ShaderBox (area D, reasoning layer)

The structural convergence — a dedicated second draw pass for hover/selection, not a
color-swap-in-place — matters more here than any specific color choice, because ShaderBox's
`_draw_canvas` already channel-splits into exactly two channels (wires=0, nodes=1) for the
node-over-wire z-order; adding hover/selection feedback is the same technique one level deeper (a
third pass, or reordering within the existing two), not a new pattern to introduce. The WCAG
minimum (24×24px effective target, achievable via invisible-button padding rather than growing the
drawn dot) gives a concrete, checkable floor for "tiny hit targets" independent of whatever the
final port-dot radius ends up being.

## Area E — feedback and self-reads: what the sources SAY

No source in this pass discusses a self-loop/feedback-read visual convention as a named design
problem with a documented solution — this is a genuine literature gap, not a researcher oversight.
Houdini's SOP Solver (a frame-to-frame feedback primitive functionally similar to ShaderBox's
`prev`-port self-read) is documented purely mechanically in SideFX's reference pages (per the
sibling D/E researcher's citation of sidefx.com/docs/houdini/nodes/sop/solver) with no accompanying
prose on how the solver's self-reference is drawn or visually distinguished — it is presented as a
subnetwork with an internal "previous frame" input node, not a loop-drawn-over-the-node-body the
way ShaderBox currently draws it. This is consistent with the maintainer's own framing in finding 3
(`00_findings.md`): "draw a little double looped arrow… or something like that" is explicitly
proposed as a novel glyph because no reference gives a ready-made answer to copy.

### Recommended for ShaderBox (area E, reasoning layer)

Absent a converged external answer, the strongest constraint from adjacent literature is Blender's
own hard-won principle from the socket-shapes post — **the visual vocabulary should communicate
what the data IS, independent of how it's drawn elsewhere** — applied here as: a feedback/self-read
should be a property of the PORT (a distinct badge/glyph at the port, as the maintainer proposes),
not a property of a WIRE that happens to loop, because a wire is inherently about the connection
between two DIFFERENT things and a self-read is conceptually a different kind of fact. This
reasoning supports the maintainer's own instinct rather than contradicting it — no source argues
against it, but none argues for it either; it should be read as a from-first-principles
justification, not a corroborated convergence.

## Area F — machinery on an immediate-mode draw list: what the sources SAY

This area is the one most directly answered by primary-source CODE rather than prose (the sibling
F-area code researcher covers the numbers); the written-guidance layer's contribution is the two
places practitioners explicitly reasoned about *tradeoffs*, in prose, rather than just shipping a
number.

**imnodes' author**, on his hit-testing approach in his own retrospective: "A simple hierarchical
algorithm is used. The curve is divided into n discrete segments. Then we can use the bezier
function to compute the position of the midpoint of each segment and find the position closest to
the mouse. Then we repeat the process for the closest segment recursively a few times."
(nelari.us/post/imnodes/) — worth flagging precisely because it **does not match the shipped
`imnodes.cpp` code** (a single-pass per-segment scan, confirmed by direct code read), which means
either the blog describes an earlier/idealized design the implementation simplified away from, or
the post is aspirational. Treat the blog's prose as the *intent*, the code as the *ground truth* —
a small, concrete instance of the "a relayed source is not the source" caution applying even to an
author's own writing about their own code.

**imgui-node-editor**'s issue #46 ("Fonts blurry on HiDPI") is the one place in this whole pass a
maintainer wrote out, in his own words, the general technique for crisp text at non-1.0 zoom on a
Dear ImGui canvas: regenerate the font atlas at a resolution tied to zoom/DPI rather than stretching
a fixed atlas, and disable ImGui's own default prefilter passes ("I had to modify font generator to
achieve such result. In `imstb_truetype.h` it was a matter of disabling calls to
`stbtt__h_prefilter` and `stbtt__v_prefilter`…") — but the thread's own resolution shows the
*actual* root cause in the reporter's case was unrelated (process DPI-awareness on Windows, not
font rendering at all), a caution directly on point for ShaderBox: verify the actual symptom
(blurry text) traces to the actual cause (atlas resolution vs. OS scaling vs. something else)
before reaching for the harder fix.

**ImGui's own channel-splitter** (`ImDrawListSplitter`, referenced across ocornut/imgui issues
#2613 and #1328) is the shared primitive every ImGui-family node editor in this pass (imnodes:
2-channel; ImNodeFlow: 2-channel; imgui-node-editor: 4+ channels per node plus dedicated selection
channels) builds its z-order/hover-on-top behavior from — confirming, from the upstream library's
own side, that draw-list channel splitting is the canonical, intended mechanism for this problem
on Dear ImGui specifically, not a workaround any one node-editor author invented independently.

### Recommended for ShaderBox (area F, reasoning layer)

ShaderBox's `_draw_canvas` already uses `dl.channels_split(2)` for the wires-under-nodes z-order —
this is already the canonical ImGui-family technique, not a deviation from it. Extending it for
hover/selection (a third channel, or drawing a hovered/selected wire's halo into channel 1 after
the nodes) is the same primitive applied one layer deeper, which every ImGui-family reference in
this pass reaches for rather than reordering within a single channel or mutating draw order.

## Pitfalls the practitioners warn about, mapped to ShaderBox's canvas

| Pitfall named by a source | Source | ShaderBox's current fact (from `pass_graph.py` / the brief) |
|---|---|---|
| A wire that crosses its own node (self-loop drawn over the node body) | Maintainer's own finding 3 (`00_findings.md`) — no external source documents a fix, but Houdini's Solver avoids the shape entirely by routing self-reference through a distinct subnetwork input, and Blender's socket-shapes philosophy argues for a port-level glyph over a wire-shape hack | `_draw_self_loop` draws a cubic "over the node's top into the prev port, which crosses the picture's top-left corner" (finding 3, verified against code) |
| Tiny hit targets | WCAG 2.5.8 (ishadeed.com/article/target-size/): 24×24px effective minimum, recoverable via generous invisible padding around a small visible glyph | `SIZE.GRAPH_PORT_R = 4` (an 8px-diameter visible dot); the C-area researcher confirms this matches the converged reference dot size, but the EFFECTIVE hit area (the `invisible_button` around it) is what WCAG's minimum actually governs, and that value was not verified in this pass — worth checking against 24px before calling the port hit target settled |
| Pan and select bound to the same button/gesture | xyflow's explicit two-named-schemas framing (reactflow.dev/learn/concepts/the-viewport) — the fork exists precisely because binding both to plain-left-drag is the naive/broken starting point every mature tool moved away from | Already resolved in ShaderBox: plain left-drag selects (rubber-band), middle-drag or Alt-drag pans — the two gestures are already on different inputs, per `_draw_canvas`'s `bg_active`/`panning` branch |
| No visual state on hover | imgui-node-editor's README ("Automatic highlights for nodes, pins and links" as a headline feature) and Blender's draw-order comment ("Draw selected node links after the unselected ones, so they are shown on top") — both treat hover/selection feedback as structurally necessary, requiring a dedicated draw pass | Confirmed directly: `_draw_node` reads only `is_output`/`selected`/`error`/`stale`; nothing reads a hover flag; wires have no hover state at all (finding 2, verified against code) |
| Labels that clip with no affordance | Houdini's explicit "Shorten Long Node Names" + "Maximum Node Name Width" policy (sidefx.com/docs/houdini/network/options.html) — named as a first-class, user-configurable setting, implying silent unguarded clipping was considered and rejected as the default | Confirmed directly: `_draw_node`'s name and port-label draws call `dl.add_text` with no width check or ellipsis (per the C-area researcher's code read); `distance_field`/`u_distance_field` clip at zoom 1 per the brief |
| A symmetric bow formula with no clamp, applied to a close or backward pair of endpoints | Blender's commit 67308d73a4f, imgui-node-editor's `easeLinkStrength`, ImNodeFlow's sign-flip — three independent fixes for the same root cause | Confirmed directly: `_draw_wire`'s bus-route control points (`gx`, `by` offsets) have no clamp or ease as the two bus-entry x-coordinates converge — the brief's described cusp is this exact, previously-solved-elsewhere shape |
| A fixed backward/routing threshold with no stated derivation | ImNodeFlow's unexplained 50px backward-flip threshold — flagged by this research as a caution, not a pattern to copy uncritically | N/A directly, but relevant if ShaderBox's own fix reaches for a magic-number threshold: prefer a formula derived from a measurable quantity (slope, as Blender does) over a bare pixel constant unless the constant's derivation is written down |

## False trails

- **Bret Victor, "Up and Down the Ladder of Abstraction"** (worrydream.com/LadderOfAbstraction/) —
  fetched and read in full. Confirmed NOT about node-based UIs, dataflow graphs, or visual
  programming: the essay is entirely about interactive parametric-slider visualizations (its one
  code example is a plain Processing drawing sketch). Checked the rest of Victor's essay list
  (worrydream.com) for a closer match — "Learnable Programming", "Drawing Dynamic
  Visualizations", "Kill Math", "Tangle", "Magic Ink" — none confirmed to address node/dataflow
  editor interaction specifically by title or description; not individually fetched given none
  surfaced as on-topic. No Bret Victor essay addressing node-graph UX was found.
- **Nicky Case, "Explorable Explanations"** (blog.ncase.me) — a general systems-thinking/interactive-
  essay series, not a node-editor design analysis. **Loopy** (ncase.me/loopy/), Case's own
  causal-loop-diagram tool, IS graph-adjacent (nodes and arrows a user draws by hand), but no
  accompanying design-rationale writeup about ITS wire/node interaction mechanics was found in
  search results — flagged as a real, adjacent artifact with no design-rationale text to cite, not
  a solid source.
- **Danny Holten's edge bundling** (cs.jhu.edu/~misha/ReadingSeminar/Papers/Holten06.pdf, full text
  read) — real, and the paper names its own precondition precisely: bundling applies to *compound
  graphs* that have BOTH an underlying tree hierarchy AND a separate layer of non-hierarchical
  "adjacency" edges, and its stated benefit is reducing clutter "when dealing with large numbers of
  adjacency edges." The author's own limitations section: "the bundle overlap in case of layouts
  with a large number of collinear nodes" is "the biggest problem of hierarchical edge bundles." A
  ShaderBox pass graph is a small flat DAG with no inherent tree hierarchy and a handful of edges
  per document — neither of the technique's own stated preconditions holds. Named in the brief as
  worth citing "if only to say it is not for us" — confirmed not applicable at ShaderBox's scale.
- **Helen Purchase's edge-crossing-minimization result** — real and well-sourced (quoted directly,
  with citation, via the Kobourov survey paper above, since Purchase's own 1997 paper is paywalled
  at ACM/Springer/ResearchGate and could not be fetched in full in this pass), but it answers a
  LAYOUT question (how should an algorithm arrange nodes to minimize crossings) rather than a
  RENDERING question (how should one wire's curve be drawn). ShaderBox's layout is manual (the user
  positions nodes; positions persist per pass per the brief) — Purchase's result would matter for an
  auto-layout feature ShaderBox does not have and this brief does not propose adding. Kept as
  general graph-aesthetics grounding, flagged adjacent-not-actionable rather than a pure false
  trail, since it is real and correctly cited.
- **Xu et al.'s curved-vs-straight-edge user study** (IEEE TVCG 2012) — real and on-topic (directly
  about bezier-curve readability, the closest possible match to area A's rendering question), but
  every fetch attempt (IEEE Xplore, ResearchGate, Semantic Scholar, Crossref, Unpaywall) hit a
  paywall or a 403. What's reported above ("a uniform curvature level had a negative impact on
  graph readability, which increased with additional curvature") is a secondary paraphrase from
  indexed search summaries, not a verified direct quote — flagged, not silently upgraded to a
  primary citation.
- **cables.gl blog** (blog.cables.gl) — real, active, paged through its index; does describe the
  tool's patching UX generally via its own site copy ("Ops can automatically connect to each other
  when added or removed, connecting is achieved instantly, and ops and connections are color coded
  for easy understanding... created after studying other node-based software to develop a workflow
  that works with the user and not against them"), but no post found constitutes a *design
  rationale* writeup — the blog itself is release notes and community-challenge posts, not a "why
  we built the wire renderer this way" article. Kept as a source-of-general-description, not a
  design-rationale source.
- **nodes.io** (nodes.io) — real, distinct project (built by variable.io), with a short but genuine
  positioning statement ("We take inspiration from popular node-based tools but strive to bring the
  visual interface and textual code closer together," per a third-party curated notebook quoting
  the site) and a documented design choice to keep basic language semantics — conditionals, loops,
  arithmetic — out of the node layer entirely, reserving nodes for higher-level composition. Not a
  false trail, but thin: no deep patching-mechanics essay exists to cite beyond this positioning
  statement.
- **Pure Data style guide** (puredata.info/docs/style-guide) — the URL is real and on-topic (a
  PEP8-style community guide to patch construction, comments, naming conventions), but the site
  serves an Anubis bot-verification challenge to every automated fetch attempt in this pass and
  could not be read directly. Not a false trail — an access failure, recorded rather than silently
  dropped or filled in from memory.
- **Substance Designer's "Graph Creation Etiquette" page** — real and relevant (confirmed via search
  extraction: warns explicitly about "links crossing through the Graph without much control" on
  long spans, directly analogous to ShaderBox's bus-route problem), but the page 403'd on every
  direct WebFetch attempt in this pass (Adobe's helpx domain appears to block the fetch tool's
  user-agent). Cited above via the search-tool's extracted quotes rather than a directly-read page
  — flagged here rather than silently treated as equivalent to a primary read.
- **Adobe's `interface-overview.html`** — same 403 pattern as above; the C-area researcher hit the
  identical wall independently. Consistent enough across two unrelated research passes that it is
  worth recording as a known access limitation for this domain, not a one-off fetch failure.
