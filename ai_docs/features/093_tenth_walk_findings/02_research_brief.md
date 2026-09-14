# 093 — Research brief: what a solid graph editor does

The maintainer's verdict on the graph view as shipped (2026-09-14, verbatim): "the graph editor
is important and large part of the ui/ux and the application as a whole. And I don't like the
current implementation, it feels very cheap. For example, there are some corner cases with these
bezier curves, there are some sharp angles appear sometimes, wrong curvature, strange angle
clampings and so on. Also, I'm not sure about the controls... What will be the most convenient
mouse control schema? Also, the size of the cards could be a little bit larger for the default
zoom level, since the uniform names and the node names don't fit fully. And as I already stated
in the previous todo list: the highlights on hover of different elements." He asked for research
before design: articles, tutorials, repositories, reference implementations, the machinery and
the UI/UX of graph editors, and a design that is solid rather than quick.

This brief anchors that research. Every researcher answers the questions below for its area,
against named primary sources (a repository's code, a library's documentation, an article), and
delivers a decision record, not a reading list.

## What the canvas is today (the facts a researcher needs, verified 2026-09-14)

- `shaderbox/widgets/pass_graph.py`, hand-drawn on one imgui `ImDrawList` in one child; hit
  testing by `invisible_button`s (background, node bodies, port dots); zoom 0.25 to 2.5 about the
  cursor; middle-drag or Alt-drag pans; left-drag on empty canvas rubber-bands; a node is 108 wide
  with an 80px picture, a 14px bold name and one 16px row per input port in the 12px font, so a
  name like `distance_field` and a port like `u_distance_field` clip at zoom 1.
- Wires: `_draw_wire` draws one cubic bezier from the output dot to the port dot with both control
  points offset horizontally by `max(24, 0.45 * dx)`; when the consumer is left of the producer or
  more than one rank away, the wire is routed on a "bus" line under the whole picture through two
  more cubics whose control points are fixed at half the column gap. **The maintainer's
  screenshot of an anomaly is this bus route with two endpoints close together: the descent's
  control points cross and the curve folds into a cusp.** There is no obstacle avoidance and no
  routing; the bus is a heuristic.
- Feedback (a pass reading its own previous frame) draws a loop from the output dot over the node's
  top into the `prev` port, which crosses the picture's corner; the maintainer wants a glyph beside
  the run-count badge instead (finding 3 in `00_findings.md`).
- Nothing highlights on hover (finding 2). A wire cannot be selected or deleted except by grabbing
  it off its port (finding 1).
- Selection: click, shift-click, rubber band; drag moves the selection with snap-to-neighbour
  guides; a wire drags from an output dot into a port or is grabbed off a filled port; drop on empty
  canvas from a grabbed wire disconnects. Every write goes through an `App` verb and is refused for
  a cycle or a media-bound port before anything is written. Positions persist per pass.
- Groups: a box at the root with boundary-edge ports, a tab per group with outside passes as
  ghosts. This model is settled (092) and is not under review here.
- Stack constraints that are settled: imgui-bundle 1.92.801 on Python, draw-list only (no
  `imgui_node_editor`), a display the agent cannot see (visual calls are the maintainer's).

## Questions, by area

### A. Wire geometry and routing
1. How do the reference editors draw a wire between an output on the right of one node and an
   input on the left of another: the curve family (cubic bezier, two quadratics, orthogonal with
   rounded corners, straight), and the exact control-point rule as a function of the endpoints'
   distance and direction (horizontal offset, vertical component, minimum offset, clamps)?
2. What do they do when the consumer is LEFT of the producer (a backward wire): the S-curve rule,
   its clamps, and how they avoid the cusp our bus produces?
3. Does any of them route around nodes or bundle wires? If so, at what cost and with what
   algorithm; if not, what do they do instead (draw on top, dim under, let the user move nodes)?
4. Arrowheads, thickness, the endpoint's attachment (does the curve end exactly at the dot's
   centre, at its rim, with a short straight stub?).
5. Hit testing a wire: distance to a flattened polyline, the sample count, the threshold in screen
   pixels, and whether the threshold scales with zoom.

### B. Mouse control schema
6. For each reference editor, the full binding table: pan (which button, which modifier), zoom
   (wheel about the cursor? step size?), select (click, shift, ctrl, alt), rubber band (which
   button, from empty space only?), node drag, wire drag (from an output only? from an input to
   re-plug? drag a wire's end off a port to disconnect? alt-drag to detach?), delete (key, menu,
   both), duplicate, multi-select move, context menu.
7. Which conflicts do they resolve, and how: pan vs rubber band on empty space, click vs drag
   threshold, a press that starts on a port vs on the node body under it, a drop on empty space
   from a wire (disconnect vs cancel vs "create node here" popup).
8. Which schema do the strongest node editors converge on, and where do they differ by domain?

### C. Node card layout and sizing
9. Card widths and heights in the references at their default zoom; font sizes; how long labels
   are handled (ellipsis, wrap, auto-width to the longest label, a tooltip on hover).
10. Where the picture (preview thumbnail) sits, if any; where ports sit relative to the picture and
    the name; the port label's placement (inside the card beside the dot, outside the card).
11. Auto-width vs fixed width: what the references do, and what the trade-off is for a canvas
    whose fit-to-view zoom is under 1.

### D. Hover and selection feedback
12. Exactly what changes on hover for a node, a port, a wire: color, thickness, glow, a lift, the
    cursor shape; and on selection; and on both.
13. How the hovered wire is disambiguated when several overlap; whether hovering a port highlights
    the wires attached to it; whether hovering a node highlights its wires.
14. Cursor shapes per state (over a port, over a wire, dragging a node, dragging a wire, over
    empty canvas).

### E. Feedback and self-reads
15. How the references show a node reading its own previous output (a feedback or delay): a loop
    drawn where, a glyph, a distinct port, a separate "delay" node?

### F. Machinery on an immediate-mode draw list
16. For editors built on Dear ImGui specifically (imgui_node_editor, imnodes, ImNodeFlow, ImFlow,
    others): how they hit-test wires, how they layer (channels, z-order on hover, selected on
    top), how they avoid the SetCursorPos and overlap traps, how they debounce click vs drag, how
    they render text at fractional zoom, and what they gave up.

## Deliverable shape (each researcher)

A markdown report under `ai_docs/features/093_tenth_walk_findings/research/`, named for the area,
with: a sources table (name | what it is | URL or path | what was read: code / docs / article);
one section per question with the answer per reference in a table where a table fits; the
CONVERGENCE (what the strongest references agree on) and the DIVERGENCE (where and why they
differ); a "recommended for ShaderBox" section that states the rule in a form a coder can
implement blind (numbers, formulas, thresholds), derived from the references and adapted to the
facts above; and a false-trails section (sources that looked relevant and were not, with why).
Cite the primary artifact; a claim about a repository is checked against its code, not its README.
