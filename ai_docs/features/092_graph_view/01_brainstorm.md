# 092 — graph view: the brainstorm record

Status: **brainstorming, no spec yet.** This file is the durable trace of the design conversation
so far: what is fixed, what is leaning, what is open, and the corner cases still to be traced.
The sketches are `00_mock.html` (three rounds; round 3 on top, rounds 2 and 1 under folds).

## Where it comes from

- 069 #19 asked for the wiring to be visible as a graph. 070 brainstormed six layouts and chose the
  strip with chips, closing the graph view *as the strip's replacement* and the `imgui_node_editor`
  question with it.
- 091 (presets) reopened it as an **opt-in second view** for a document that has outgrown the strip,
  **hand-drawn on the imgui draw list, no `imgui_node_editor`** (the maintainer's reason: a future
  port). 091 also rejected folding a group on the strip: a folded group must be convex in the DAG,
  and that rule set was too much for the strip. `conventions.md` records "a group is a label on the
  pass entry, and nothing folds", with no revisit trigger; this feature is the deliberate reopen.
- 072 D9 deleted `PassGraph.layout`; no node position is stored anywhere today.
- 091 deferred: a cross-project presets folder, export a group as a document, nested groups (a
  source's inner labels are dropped on import), a group rename verb.

## What the code gives us (facts, verified in this session)

- `Document.effective_wiring()` -> `pass -> uniform -> pass`, what the binder binds. A sampler's
  source is a VALUE on the sampler (`PassSource` / `NoSource` / `AutoSource`, 072); the name rule
  (`u_<pass>`) decides an undecided one; a compiled pass answers over the samplers its program
  declares, an uncompiled one over its explicit rows only.
- `pass_graph.plan_passes(wiring)` -> topological order, per-pass reads, the feedback set, and
  `GraphError`s (a cycle per pass, a read of a pass that does not exist). `strip_order` is stable
  across output changes. `entry_points` (a pass reading no other) and `readers_of` are 091's.
- `PassEntry` = `target`, `iterations`, `group: str` (pattern-checked label, "" = none). A group
  exists while a pass carries its name; there is no group entity. `group_tint` hashes the name to
  one of four theme tints; `group_runs` cuts the strip order into runs of consecutive members.
- The strip (`widgets/pass_list.py`): one `preview_cell` child window per pass; click = set the
  output AND open the pass in the editor; gear overlay -> settings modal; context menu: Settings,
  Delete, Leave group; delete-✕ arms an in-cell confirm; off-plan passes dim; compile error = red
  border; output = accent border; frozen under `copilot_turn_active`.
- The editor's tabs: `EditorTab(path, kind, document_id)` records, a native imgui tab bar
  (`tabs/code.py::_draw_tab_row`) with reorder, close, unsaved dot, error tint, and the
  select-pending latch for programmatic switches. `ui_primitives.text_tab_row` is the lighter
  color-only selector the Uniforms tab uses.
- Largest real document: Radiance Cascades, six passes in a chain SIX ranks deep with two long
  edges from `paint` and two feedback passes (run counts 12 and 6). The bloom fixture is five
  passes; as an imported group in a host it is seven passes in five ranks. Depth, not fan-out,
  is what the picture has to absorb.
- The Document tab sits right of the document grid (which takes width/2.6 of the app panel); the
  app panel can go down to 360 wide. No window manager on the dev box: headless drivers catch
  crashes only, every visual call is the maintainer's.

## Fixed by the maintainer in this conversation

1. **Hand-drawn on `ImDrawList`**, nodes as draw-list pictures with one `invisible_button` per node
   for hit testing and the context-menu anchor. Zoom-to-fit is the reason a graph beats the strip
   once a document outgrows it, and child-window nodes cannot zoom.
2. **The node is the port-list node** (round 2, variant A): picture, name, then one input port per
   sampler under it; the output dot on the right of the picture; a feedback read is a `prev` port.
   Edges end at a port's dot and carry no label.
3. **Tabs instead of folding.** The root tab shows every ungrouped pass as a node and every group as
   one **box**; a group's tab shows its members as nodes and the outside passes they touch as
   dimmed, dashed **ghosts** at the border (ports kept). Double-click a box or right-click > Open
   enters; Escape goes up. No fold flag is stored; the root always shows boxes.
4. **A box's ports are the group's boundary edges**: a member sampler that reads an outside pass or
   nothing is an input port; a member read from outside, plus the bundle's own output, is an output
   port. This is what removes 091's convexity rule: the box is never a node the planner orders. A
   non-convex group draws a back edge at the root and its leaking pass as a ghost on both sides of
   the group tab. Nothing is refused.
5. **Positions are the pass's; arrange is a verb.** A `position` on the pass entry (optional, loads
   without migration); the first sight of a document runs the rank layout once and stores it;
   "Arrange" rewrites positions; snapping is a drag helper (align to a neighbour's column/row),
   never a regime. A new pass lands under the cursor, unconnected. The box sits at its members'
   bounding box; dragging it translates the members.
6. **Wiring is a drag from an output dot into a port** and writes `PassSource(name)` into that
   sampler, the same write the uniforms-panel row makes; dropping on empty space writes `NoSource`.
   A drop that would make a cycle is refused before anything is written, by planning the
   hypothetical wiring with the pure planner (091's pattern). A node's output into its own port is
   feedback, allowed.
7. **The strip stays as it is.** The graph is a second view; 091's outlines are untouched.
8. **Groups earn their place only because import makes them.** Interactive grouping is: multi-select
   on the canvas, right-click > Group, a name; the label is written onto each selected pass through
   the existing per-pass write. No rename of the passes (091's prefix is an import-time collision
   guard). Any subset may be a group, so no eligibility highlighting.

## Leaning, not yet decided

- **Merged ports on a box** (round 3 B): slots fed by the same outside pass fold into one port
  (`scene ×3`); unfilled slots fold by sampler name; a wire dropped on a merged port rewrites every
  slot behind it. Versus one port per slot named `member.sampler` (round 3 A).
- **Nesting as a path label** (`post/bloom`): the root shows one box per top-level segment, a group
  tab shows its direct passes and one box per sub-group; boundary ports computed the same way at
  every depth; the tab row becomes a breadcrumb; import prefixes inner labels instead of dropping
  them; the strip outlines by full path. Reverses 091's no-nesting deferral (its trigger: "a source
  with two bundles worth keeping apart").
- **The graph as a third editor-tab kind** (`EditorTab.kind = "graph"`, keyed by a group path
  instead of a file), in the editor pane's own tab bar. The maintainer plans a zen mode / a pane
  swap later, so the canvas should be a pane-agnostic widget either way.
- **Inverse verbs**: Dissolve (clear the label on every member), Rename (retype on the members).
  **Save a group as a document** (091's deferred export half). Picking a box as the output picks
  the bundle's output; the box's picture follows the document output when it is a member.
- **Staging**: (1) the read-only canvas with boxes and tabs; (2) drag and wiring; (3) group verbs
  and nesting. Each a mid feature.

## Corner cases the maintainer named, still to trace

- Removing an intermediate member of a group: what the root box shows afterwards (its ports, its
  picture), what the group tab shows, what the wiring does today (a read of a vanished pass binds
  black and is no chip).
- Error visualization: colored ports or edges for a compile error, a cycle, a read of a missing
  pass, a sampler the program no longer declares, a black (unfilled) input.
- Wrong connections: is one even possible when every port is a `sampler2D`? Target dtype / scale
  mismatch between producer and consumer, feedback into an iterated pass, a self-read, a wire that
  would close a cycle, a wire into a ghost.
- "And so on": the maintainer expects more of these than he listed.
