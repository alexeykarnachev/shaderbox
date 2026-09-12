# 092 — The graph view

Status: **spec locked after pre-implementation round 1; implementing W1.** Sketches: `00_mock.html` (round 3 is the
picture). Record of the design conversation: `01_brainstorm.md`. The review round that produced
the constraints and decisions below: `02_triage.md` and `reviews/brainstorm_*.md`. Every
"Fixed" item of the brainstorm and every default of the triage is locked here; nothing is
re-opened.

## Goal

A second, opt-in view of a document's passes beside the strip: a node canvas, hand-drawn on the
imgui draw list, where each pass is a node with one input port per sampler and one output dot,
each read is a wire into a port, and each group is one **box** at the root whose ports are the
group's boundary edges and which opens into its own **tab** showing the members with the outside
passes they touch as **ghosts**. The picture is what the wiring is: no fold, no group entity,
no second source of truth. Two waves land it:

- **W1, the read-only canvas:** nodes, boxes, ghosts, the root and group tabs, pan, zoom, fit,
  arrange, click and double-click, the node and canvas context menus, the error cues.
- **W2, the interaction:** drag a node or a box, drag a wire, rubber-band and shift-click
  selection, Group and Dissolve, and the two guards the round found missing in the flat model
  (a rename that would close a cycle, a pass named like a group).

070 closed the graph view as the strip's replacement and this is not that: the strip is
untouched. 091 rejected folding on the strip because a folded group must be convex in the DAG;
this design never orders a box, so the rule does not arise, and `conventions.md`'s "nothing
folds" entry gets its revisit pointer (D20).

## Out of scope

- **Nesting** (path labels `post/bloom`). Breaks the import prefix (a slash in a pass filename),
  collides `group_tint('post')` with `group_tint('post/bloom')`, needs the three group validators
  collapsed. No scenario in the review needed it. Trigger: a source with two bundles worth keeping
  apart, 091's own.
- **The graph as an `EditorTab.kind`** in the editor pane's tab bar. `path` is the tab's identity
  in five places and a group is not a `Path`. The canvas is written pane-agnostic (D2) so the move
  is a mount change. Trigger: the zen mode / pane swap the maintainer plans.
- **A wire drop that writes the `sampler2D` declaration into the shader.** The canvas cannot build
  a chain because a port exists only when the shader declares a sampler (`PASS_STUB` declares
  none); making the drop author GLSL is a different feature. Trigger: the maintainer asks for it
  after using W2.
- **Keyboard verbs on the canvas** (Escape up a tab, Delete). Escape's ladder has no graph rung and
  a bare Delete cannot be a registry chord. The tab row is the way up; delete is on the menu.
  Trigger: the first session where the mouse-only canvas is the complaint.
- **`duplicate_pass`**, hence duplicating a box. Trigger: the maintainer wants two blooms.
- **Save a group as a document**, the presets folder, a group rename verb: 091's deferrals, their
  triggers unchanged.
- **A cycle cue outside the graph.** The strip's grey wash stays unexplained; the graph is the
  first surface that names a cycle. Trigger: a user asks why the strip went grey.
- **Undo.** Grouping is N writes and Dissolve is its inverse (D16); a wire drop overwrites as the
  sampler row's combo does. Trigger: none new.
- **Nodes at the strip's tile size**, a header-bar node, ports beside the picture: sketched in
  round 2, not chosen. Trigger: the compact node proves unreadable in use.

## Design decisions

**D1. Ports come from the compiled program; edges come from the wiring.** A node's input ports
are `sampler_names(pass)` in the order that function returns (the compiled program's own
iteration order), with every port whose resolved source is the consumer itself moved to the
end: a self-read is the feedback port and reads bottom-most whether it sits on `u_prev` or on
any other sampler explicitly sourced to its own pass. Its edges are `Document.effective_wiring()`. A port is therefore never drawn for a sampler the program no
longer declares, which is what stops `set_sampler_source` (which does not validate the uniform)
writing a dead row. A never-compiled pass has no ports, so **the canvas compiles what it draws** through the
seam 091 already uses for the same reason, `project_session.compile_pending_passes(document)`,
called once on a document's first canvas frame (bounded, the largest document is six passes;
the render loop's own sweep only brings the output's chain online, so an off-plan pass would
otherwise never get ports). The per-node port list is a pure function,
`pass_graph.node_ports(declared, values, wiring_row, name) -> list[Port]`, over
`sampler_names(pass)`, the pass's sampler values and its wiring row; the canvas only calls it.

**D2. Where it lives.** The Document tab's Passes caption row gets a `strip | graph`
`segmented_choice`; the choice is `UIAppState.passes_view: PassesView` (`ui_regions.PassesView`,
`STRIP` default), an app-level preference like `channel_view`. `tabs/document.py` draws the
caption, the toggle and the `add pass` / `import...` row itself and dispatches the body to
`pass_list.draw` or `pass_graph.draw`; `pass_list.draw` loses its caption and its buttons and
draws only tiles. The canvas widget is `widgets/pass_graph.py::draw(app, document_id)`, a leaf
that fills the child it is handed and never positions a sibling; its per-document view state is
`widgets/graph_state.py::GraphViewState` (pan, zoom, scope, selection, the in-flight drag, the
one-shot `fitted` flag), held in `App.graph_views: dict[str, GraphViewState]` and created on
first use. It is transient UI state, not persisted, and the lazy-row rule does not apply since
nothing off-draw writes it. `App.graph_views` is dropped per document in
`App.forget_render_state(document_id)`, beside `pending_resolution` / `auto_size_states` /
`throttle_states` / `document_costs`, the funnel that already exists for every ephemeral
per-document entry. The `add pass` / `import...` row moves to `tabs/document.py` INSIDE its own
`begin_disabled(app.copilot_turn_active)`: the bracket in `pass_list.draw` wraps those buttons
today, and moving them out of it without re-establishing it would silently unfreeze them during
a copilot turn.

**D3. The scope and the tab row.** `GraphViewState.scope` is `""` (the root) or a group name.
A `text_tab_row` at the canvas top lists the document's display name (the root) then every group
name in strip order of its first member; clicking sets the scope. The scope is revalidated every
frame: a name no pass carries falls back to `""` (the last member can leave from inside the tab).
Double-click on a box enters it; the root tab is the way up (no Escape, out of scope). A scope
change refits the view once. `text_tab_row` keys and returns by NAME, so the root cannot be the
empty string there: the row is built as `[root_label, *group_names]` where `root_label` is the
document's `ui_name` or `document` when it is empty (the Document tab's input can be cleared),
and a click is mapped back to a scope by INDEX, never by the returned string, since a document
named like a group would otherwise be ambiguous.

**D4. What each scope shows.** Root: every ungrouped pass as a node and one box per group.
Group `g`: every member as a node, plus one **ghost** per outside pass that feeds a member or
reads one; a pass that does both is drawn twice (feeder left, reader right) so no edge runs
backward through the members. Ghosts are dimmed (`GRAPH_GHOST_ALPHA`), dashed, keep their ports,
carry no badge; clicking a ghost sets the scope to `""` and selects the pass. Groups never draw a
tinted region: the box is the group's picture.

**D5. A box's interface** (pure, `pass_graph.group_boundary`). Inputs: every `(member, sampler)`
whose source is outside the group or is unfilled, one port per slot, labelled by the sampler
name when unique across the box, `member.sampler` otherwise (round 3 A, triage D1). Outputs:
every member read from outside, plus the **bundle output** always, drawn hollow when nothing
outside reads it. Bundle output (`pass_graph.bundle_output`): the document output when it is a
member; else the first member in strip order read from outside; else the last member in strip
order that no member reads; else the last member. The box's picture is the bundle output's live
texture; the box's tint is `group_tint`; its badge says `N passes`; clicking it picks the
bundle output (`pick_pass`); the accent border shows when the document output is a member.
A group split across the DAG (the same label on disconnected passes) is one box at its members'
bounding box; Arrange pulls the members together.

**D6. Positions.** `PassEntry.position: tuple[GraphCoord, GraphCoord] | None = None`, where
`GraphCoord = Annotated[float, Field(allow_inf_nan=False, ge=-MAX_GRAPH_COORD, le=MAX_GRAPH_COORD)]`
(`MAX_GRAPH_COORD = 100_000.0`), canvas space at zoom 1, the node's top-left. `None` means never
placed. `pass_graph.rank_layout(wiring, names, groups, sizes, placed) -> dict[str, tuple[float,
float]]` is pure and returns a position for every name in `names` (rank = longest path from a
root over the non-self edges, cycle members rank 0; within a rank, group members adjacent, then
by the mean position of predecessors, then strip order; columns left to right at `GRAPH_GAP_X`,
rows at `GRAPH_GAP_Y`, each column centred on the tallest; `sizes` is each node's width and
height, since a port row adds height; `placed` is the already-stored positions, read only for
the predecessor tiebreak). The canvas calls it every frame with `names` = the passes whose
`position is None` and uses the result for those alone; Arrange calls it with `names` = every
pass of the DOCUMENT and `placed = {}` and writes the whole result. A placed pass keeps its
stored position, and an unplaced one may overlap it until Arrange. A position is written by a
drag (once, on release, W2) or by Arrange (once for every pass, W1) through one verb,
`ProjectSession.set_pass_positions(document_id, positions)`, one save. Every creator leaves
`None` (`add_pass`, the copilot's tool, `import_passes` which strips the source's positions).
The box has no position: it is its members' bounding box, and dragging it translates the
members. `graph.json`'s `version` does not bump. `load_graph`'s per-entry salvage carries the
field: a corrupt position costs that position, nothing else.

**D7. Drawing.** One `begin_child` (`##pass_graph`, borders, `no_scrollbar |
no_scroll_with_mouse`) of the tab's remaining height, at least `SIZE.GRAPH_MIN_H`. Everything
inside is draw-list geometry under one transform, `screen = origin + (canvas - pan) * zoom`,
with `channels_split(3)`: 0 wires, 1 nodes and ports, 2 the foreground (an in-flight wire, the
rubber band). A node is a rounded rect (`BG_SURFACE` fill, `BORDER` line), the pass's live
texture via `add_image_rounded` at `GRAPH_THUMB * zoom` (the same blit `preview_cell` does, no
second render), the name under the picture (`font_14_bold` when live, `FG_DORMANT` when
off-plan, pushed at `size * zoom`), a `×N` badge on the picture when `iterations > 1`, then one
port row per input (`GRAPH_PORT_ROW * zoom` high: a dot at the node's left edge, the label in
`font_12`), and the output dot at the picture's right mid. Node width `GRAPH_NODE_W`, a box
`GRAPH_NODE_W + GRAPH_BOX_EXTRA_W`. A wire is `add_bezier_cubic` from the output dot to the
port's dot with control points offset by 45% of the horizontal distance (30 minimum); a wire
whose consumer sits left of its producer, or that spans more than one rank, rides a bus below
the row (`GRAPH_BUS_STEP` per extra rank); feedback is a small loop from the output dot over the
node's top-right into the `prev` port. Fonts are pushed at fractional sizes (1.92 bakes a face
per integer size, crisp at any zoom). Zoom is clamped to `[GRAPH_ZOOM_MIN, GRAPH_ZOOM_MAX]` =
`[0.25, 2.5]`. `COLOR.GRAPH_EDGE` is a fixed role and joins `_GROUP_TINT_EXCLUSIONS` beside the
state hues, so a wire can never be drawn in a group's tint.

**D8. Hit testing.** One `invisible_button` covering the child is the canvas background,
submitted first with `set_next_item_allow_overlap()`; then one `invisible_button` per node body,
each with `set_next_item_allow_overlap()`; then (W2) one per port. The chain is what lets the
later item win; a missed level makes the canvas inert. A port's hit box is
`max(GRAPH_PORT_R * zoom, GRAPH_HIT_MIN)` in screen pixels while the drawn dot keeps scaling.
Every context menu on the canvas opens with `begin_popup_context_item(None)` (the previous
item), never an explicit id, which on one shared child fires on a right-click anywhere. The
canvas menu opens by hand: right-click released over the background with no node hovered.

**D9. Pan, zoom, fit, arrange.** Middle-drag on the canvas, or Alt+left-drag, pans
(`io.mouse_delta`); the wheel zooms about the cursor (the canvas point under the cursor is
invariant). Fit sets pan and zoom from the visible nodes' bounding box with `SPACE.LG` margin,
zoom capped at 1; it runs once when a document's canvas first draws at a nonzero size, on every
scope change, and from the canvas menu. Arrange (canvas menu) is `App.arrange_graph(document_id)`: it writes `rank_layout` over every
pass of the DOCUMENT, never the visible subset, so the verb means the same thing at the root and
inside a group tab, into `position` through `set_pass_positions`, one save, and then fits the
current scope. Left-drag on empty canvas is
the rubber band (W2) and does nothing in W1.

**D10. Click and menus (W1).** Click a node: `pick_pass(document_id, name, focus_editor=False)`,
the strip's verb. Double-click a node: the same with `focus_editor=True`. Click a box: pick the
bundle output; double-click: enter. Right-click a node: the strip's item set, extracted from
`pass_list._draw_context_menu` into `pass_list.pass_menu_items(app, document_id, name)` so the
two surfaces cannot drift (Settings, Delete, Leave group); the strip keeps its own
`begin_popup_context_item` with its explicit id, which is safe there because each tile is its
own window. Right-click a box: Open, and Dissolve (W2). Right-click empty canvas: Add pass,
Import..., Fit, Arrange. `pass_menu_items` keeps the strip's two gates (Delete only while
`len(document.passes) > 1`; Leave group only while the entry carries one). A ghost's right-click
offers the same item set, since a ghost is a real pass and the verbs name the pass, not the
scope. A box's menu is its own (Open; Dissolve in W2) and never `pass_menu_items`, since a box is
not a pass. The whole canvas sits under `begin_disabled(app.copilot_turn_active)` like the strip;
the draw list still paints the live pictures.

**D11. The error language.** Port dots carry state by shape: filled disc = wired; hollow ring =
unfilled (black by default, not a fault); ring with a filled centre = `NoSource`; double ring =
`prev`; a media-bound sampler draws a small square dot (D15). Node border: `STATE_ERROR` when
`compile_unit.errors`, else `ACCENT_PRIMARY` when the pass is the document output, else
`BORDER`; a never-compiled pass has a dashed border. A node the output does not need (the
strip's `live` set, `evaluation_order(wiring, output) or {output}`) dims its name and its edges.
Exactly ONE pass per cycle carries the culprit message (`plan_passes` records
`_cycle_message(trail, name)` on the back-edge visit only; every other member of the loop gets
"pass is not ordered: an input is on a cycle."), and that one message names the whole loop
(`passes form a cycle: cascade -> paint -> composite -> cascade`). So
`pass_graph.cycle_edges(errors)` parses the trail out of each culprit message and returns the
set of consecutive `(producer, consumer)` pairs on it; every edge in that set is `STATE_ERROR`.
Victims get nothing on the node and nothing on their edges. The cue reads
`plan_passes(document.effective_wiring())[1]` computed by the canvas each frame, never
`Document.graph_errors`, which is refreshed only inside `Document.render` and so is stale for a
document the 090 throttle skipped this frame; the planner is pure and the cost is nothing. A box carries
`STATE_ERROR` when any member has a compile error, and nothing for non-convexity. No cue for
dtype / scale / filter / wrap. A compile-erroring pass's picture is its last good frame; the
node dims it (`preview_cell`'s docstring names a corner tick that its code never draws, so
there is no mark to reuse).

**D12. Wires (W2).** Press on an output dot and release over an input port writes
`set_sampler_source(document_id, consumer, sampler, PassSource(producer))`; release over empty
canvas writes nothing (the drag came from an output). Press on a **filled** input port grabs its
wire: release over another input port moves it (`PassSource` there, `NoSource` on the original),
release over empty canvas writes `NoSource` on the original. Every drop lands through ONE verb, `App.drop_wire(document_id, consumer, sampler, source)
-> str`, which decides the refusal and then makes the write; the widget never calls
`set_sampler_source` itself, so the refusal cannot be bypassed by a second drop path. A drop
that would close a cycle is refused before any write: `pass_graph.refuse_drop(wiring, consumer,
sampler, producer) -> str` builds the hypothetical wiring (`wiring_with`), runs `plan_passes`,
and returns the planner's message on any error (the culprit need not be an endpoint) or `""`;
the verb toasts the message through `notifications`. A node's output into its own port is feedback, allowed. A drop on a
media-bound port is refused ("bound to media; unbind on the Uniforms tab", D15). A drop onto a
port wired by the name rule materializes an explicit `PassSource`, since the drop writes one
anyway. A ghost's ports are drop targets and drag sources like any other, since a ghost is a
real pass drawn dimmed (the write names the pass, the tab is only a viewport). The wire in flight
is a bezier from the source to the cursor on channel 2.

**D13. Drag (W2).** Press on a node body and move: the node follows `io.mouse_delta / zoom`;
release writes its position through `set_pass_positions`, one save. The drag is a pure state
machine in `widgets/graph_state.py` (`NodeDrag.begin(names, positions)`, `.update(delta)`,
`.commit() -> dict[str, position]`): `update` returns nothing to write and `commit` is the only
thing that does, so "one save per gesture" is asserted on the machine, and
`App.commit_node_drag(document_id)` is the one caller of `set_pass_positions` from a drag. A box drags every member
by the same delta. A drag moves every selected node together when the pressed node is selected.
During a drag, snapping aligns the moving node's left edge or top edge to any other visible
node's within `GRAPH_SNAP_PX` screen pixels, and draws the guide line on channel 2.

**D14. Selection and Group (W2).** `GraphViewState.selection: set[str]` of pass names (a box
selects its members). Left-drag on empty canvas draws the rubber band from
`io.mouse_pos - get_mouse_drag_delta` and selects every node whose rect intersects it on
release; shift-click toggles one; click on empty clears. Right-click with a selection adds
`Group...` to the node menu: a popup with a name input (Enter or Create commits, Cancel or Esc
cancels, per imgui-ui 7.5) that writes `set_pass_groups(document_id, names, group)` through `App.group_selection`, a new
session verb that validates once (the group pattern, and D17's collision) and saves once, and
writes nothing on a refusal.
Selecting a box and grouping it with others rewrites its members to the new label (flat
labels, so the old group dissolves into the new one). Dissolve on a box menu is
`set_pass_groups(members, "")`. `set_pass_group` stays for the modal and the copilot and calls
the batched verb with one name.

**D15. A media-bound sampler's port.** `sampler_names` lists it and the wiring does not (a bound
texture is not a `PassSource`), so it draws as a port in a media state (a square dot, no wire)
and a drop on it is refused (D12): releasing the bound texture from a drag is the one write the
canvas must not make silently. `wired_pass` answers `None` identically for a bound texture, a
`NoSource` and an unresolvable `AutoSource`, so the port states of D11 are not derivable from
the wiring: the canvas reads `Pass.uniform_values[sampler]` and branches by type exactly as
`widgets/uniform.py` does (`MediaWithTexture | moderngl.Texture` = media, `NoSource` = ring with
a filled centre, a `PassSource | AutoSource` resolving to a pass = filled disc, otherwise a
hollow ring); a sampler whose resolved pass is the consumer itself is the feedback port,
whatever its name.

**D16. Dissolve is required.** It is the inverse of Group, which is N writes with no undo. A box
gets no Delete verb; a member is deleted from its own node menu with the strip's two-click arm
(`delete_pass` + `close_editor_for_path`, exactly `pass_list._delete_pass`).

**D17. One namespace for passes and groups.** One function decides a group name,
`pass_graph.group_name_error(group, pass_names) -> str` (the pattern, then the collision), and
every entry point that writes a group calls it: `set_pass_groups` (hence `set_pass_group`, the
modal and the copilot), and `pass_import.plan_import` (a bundle imported under a name a host
pass carries would otherwise draw two root entities with one name). `_pass_name_error` gains
the mirror check, so `add_pass` and `rename_pass` reject a pass named like an existing group.
The root keys nodes and boxes by name. Messages: the existing "a group name starts with a
letter ..." for the pattern (tests pin it), and "a pass and a group cannot share a name" for
the collision.

**D18. A rename plans before it moves.** `rename_pass` computes the wiring the document would
have after the rename through `Document.wiring_if_renamed(old, new)`: it re-keys `self.passes`
under the new name AND applies `rename_pass_sources(old, new)` to a copy of each pass's sampler
values, reads `effective_wiring()`, then restores both in a `finally` (a compile raising inside
the read must not leave the document re-keyed). Both halves are needed because `rename_pass`
itself does both: a key swap alone drops every explicit `PassSource(old)` row. `rename_pass`
refuses when `plan_passes` over that wiring reports any error, with the planner's message. Today a rename can create a name-rule edge that closes a
loop with no check; the drag has the guard (D12), so the rename gets the same one.

**D19. What the copilot sees: nothing new.** No tool accepts or reports a position; the pass
table stays flat; no groups paragraph in the prompt; no `group_passes` tool. The dogfood
harness needs nothing: the layout and the boundary are pure functions in `pass_graph.py`.

**D20. Docs.** `conventions.md`'s "A pass GROUP is a label ... and nothing folds (feature 091)" keeps its
own revisit trigger (a group-level fact no member can hold, which 092 does not create: the box
is derived from its members every frame and stores nothing) and gains one sentence scoping the
no-folding half to the STRIP: the graph view contracts a group to a box whose ports are its
boundary edges, and since the box is never a node the planner orders, convexity is not a rule
there either. A new entry records D6 (a position is
written only by a placement, never by a draw) and D1 (ports from the program, edges from the
wiring). The imgui-ui skill §8 gets the two canvas rules of D8 (the allow-overlap chain; the
context-item id). `dev_flow.md`'s module map gets `widgets/pass_graph.py` and
`widgets/graph_state.py`. The Help panel's Passes section gets one sentence: a port exists
because the shader declares a sampler. 070's spec gets its pointer to this feature. The import dialog
(`popups/import_passes.py`) gets one line saying the source's own groups are flattened under
the new one (triage S15; 091 drops inner labels silently today).

## Files touched

- `shaderbox/pass_graph.py`: `PassEntry.position` + `GraphCoord` + `MAX_GRAPH_COORD`,
  `PassGraph.with_positions`, `rank_layout`, `graph_ranks`, `node_ports` (+ `Port`, its state),
  `group_boundary` (+ `Boundary`, `BoxPort`), `bundle_output`, `wiring_with`, `refuse_drop`,
  `cycle_edges` (the `(producer, consumer)` pairs a culprit message names), `group_name_error`.
- `shaderbox/document.py`: `wiring_if_renamed` (re-key plus `rename_pass_sources` on copies,
  restored in a `finally`).
- `shaderbox/project_session.py`: `set_pass_positions`, `set_pass_groups`, `set_pass_group`
  routed through it, `_pass_name_error` taking the group names, the D18 guard in `rename_pass`
  before the file moves, `import_passes` writing `"position": None` into every copied entry
  (the `model_copy(update=...)` line that copies the source's entry verbatim today). The block
  comment over the verbs counts eight, not six.
- `shaderbox/pass_import.py`: `plan_import` rejects the group through `group_name_error`.
- `shaderbox/popups/import_passes.py`: the flatten line.
- `shaderbox/ui_regions.py`: `PassesView`, `PASSES_VIEW_LABELS`.
- `shaderbox/ui_models.py`: `UIAppState.passes_view`.
- `shaderbox/theme.py`: `SIZE.GRAPH_*`, `COLOR.GRAPH_EDGE` (joins `_GROUP_TINT_EXCLUSIONS`),
  `COLOR.GRAPH_GHOST_ALPHA`; the zoom clamp lives beside the sizes, `MAX_GRAPH_COORD` beside the
  model it bounds.
- `shaderbox/widgets/graph_state.py` (new): `GraphViewState`, `NodeDrag` (pure), `WireDrag`.
- `shaderbox/widgets/pass_graph.py` (new): the canvas.
- `shaderbox/widgets/pass_list.py`: caption and buttons move out; `pass_menu_items` extracted;
  `_draw_pass_tile`'s `set_tooltip("Pass settings")` stays (the prose gate requires the module
  to remain in its walk).
- `shaderbox/tabs/document.py`: the caption row with the toggle, the dispatch, the button row
  under its own `begin_disabled`.
- `shaderbox/app.py`: `graph_views`, `graph_view_for`, the eviction in `forget_render_state`,
  the gesture verbs `arrange_graph`, `drop_wire` (W2), `commit_node_drag` (W2),
  `group_selection` (W2), `dissolve_group` (W2).
- `shaderbox/help_content.py`: one sentence.
- `scripts/smoke.py`: after the frame-42 group stretch and before the frame-48 return: frame 43
  switches `app_state.passes_view` to GRAPH and snapshots the multi-pass document's
  `graph.model_dump()`; 45 asserts the view fitted and sets the scope to `smoke_group`; 46 asserts
  the scope survived a frame and sets a scope no pass carries; 47 asserts the scope fell back to
  the root, that the graph dump is unchanged (a draw wrote nothing), that `arrange_graph` saved
  exactly once (a `_count_saves` helper beside `_arm_feedback_canary`) and left no pass
  unplaced, then switches back to STRIP. `make gates` reports a display-less smoke as skipped,
  so on the dev box this stretch does not run; the pure tests are the gate.
- `tests/test_pass_graph.py`: `rank_layout` puts producers left of consumers (falsifier: rank by
  insertion order), is deterministic over dict order, places only the names asked for (two of
  five placed, three returned), keeps cycle members at rank 0, keeps group members adjacent;
  `group_boundary` over the bloom, non-convex, generator, one-member, split and reads-nothing
  shapes, each asserting the full (inputs, outputs) pair (falsifier for the last: drop the "or
  unfilled" clause); `test_a_terminal_box_still_has_its_bundle_output_port` (a group nothing
  outside reads: exactly one output port, the bundle output; falsifier: make it conditional on an
  outside reader, the mutations review's case 2); `bundle_output` follows its four branches in
  order and always returns a member; `node_ports` builds from the program's samplers, never from
  a stored row the program no longer declares (falsifier: build from `uniform_values` keys), and
  classifies the media, `NoSource`, wired, unfilled and feedback states; `refuse_drop` refuses
  `a->b->c` + `c -> a.u_c` with the trail in the message, allows the diamond `a -> c.u_a2`, the
  unrelated `loner -> a.u_l` and a self-read (falsifier: refuse only when the culprit is an
  endpoint); `cycle_edges` returns the consecutive pairs of a culprit message; `position` bounds
  (NaN, +/-inf, +/-1e30 raise; a pair is accepted; falsifier: a bare `tuple[float, float]`);
  `group_name_error` over the pattern and the collision.
- `tests/test_graph_state.py` (new): `NodeDrag.update` returns nothing to write and `commit`
  returns every moved name once (falsifier: write on update); the scope revalidation helper
  falls back to the root for a label no pass carries.
- `tests/test_graph_persistence.py`: `position` round-trips (extend
  `test_a_graph_round_trips_every_field` with a placed entry, so the name stays true) and a
  corrupt position costs only itself through `load_graph`.
- `tests/test_pass_verbs.py`: `set_pass_positions` saves once for the whole set; a position
  survives every other verb (target, iterations, group, output, rename); `set_pass_groups` saves
  once, and a group named like a pass is refused with nothing written; `add_pass` refuses a name
  a group carries; every group-writing entry point (`set_pass_groups`, `set_pass_group`,
  `plan_import`) refuses the same collision, and `add_pass` / `rename_pass` the mirror
  (falsifier: the check on `set_pass_groups` alone); `rename_pass` refuses the cycle it would
  create, asserting the planner's message AND that the old file still exists and the new does
  not (applied at `ProjectSession.rename_pass`, never at `wiring_if_renamed`; the mutation: with
  the guard removed the rename succeeds and the plan reports the cycle);
  `wiring_if_renamed` leaves `document.passes` keys and every `uniform_values` identity as it
  found them, also after a call that raises; `import_passes` leaves every position `None`
  (falsifier: the current `model_copy` line); no `ProjectSession` public method but
  `set_pass_positions` names a position parameter (reflection); the two `pass_list.draw` tests
  keep asserting what they name after the caption and buttons move out.
- `tests/test_graph_view.py` (new, the `app` fixture): `App.drop_wire` refuses the cycle drop
  and writes nothing, accepts the legal drop and writes the `PassSource`, refuses a drop on a
  media-bound sampler and keeps the bound texture (falsifier: call `set_sampler_source`
  unconditionally, which runs `try_to_release`), writes `NoSource` on the original for a
  grab-and-drop-on-empty; `arrange_graph` saves once and leaves no pass unplaced;
  `group_selection` on a box plus a plain pass rewrites the members to the new label;
  `dissolve_group` clears every member with one save; `widgets/pass_graph.py`'s source contains
  no call to `set_sampler_source`, `set_pass_positions` or `set_pass_groups` (the App verbs are
  the only writers).
- `tests/test_ui_regions.py` (new): every `PassesView` has a label, each label is within the
  control budget, the default is STRIP and the choice persists through `UIAppState` (the
  `ChannelView` trio's shape).
- `tests/test_theme.py`: `GRAPH_EDGE` is in the mirrored exclusion set (falsifier:
  `GRAPH_EDGE = GROUP_TINTS[2]` imports clean today).
- `tests/test_button_tiers.py`: `("widgets/pass_graph.py", "invisible_button")` joins
  `_NOT_A_VERB` ("the canvas, node and port hit rects: a hit rect, no label") in the SAME commit
  as the widget, since `test_every_listed_exception_still_exists` fails on an entry without a
  site.
- `tests/test_ui_prose_budget.py`: `menu_item_simple` joins `_IMGUI_ROWS` at the button-label
  budget (a menu item is an action phrase like a button), so the canvas menus and the strip's
  Settings / Delete / Leave group enter the gate; the `strip | graph` toggle reads its labels
  from `PASSES_VIEW_LABELS`, which the walk cannot resolve, so the caption function in
  `tabs/document.py` joins `_UNMEASURABLE` with that reason (the `ui.py` entry for
  `CHANNEL_VIEW_LABELS` is the precedent). `ImDrawList.add_text` stays outside the gate.
- `tests/test_copilot_pass_tools.py`: re-run; `set_pass_group` as a wrapper must return the
  same "group name" string for a bad pattern. `tests/test_default_wiring.py`: re-run; its two
  renames run through the name-rule fixture the D18 guard plans over.
- `ai_docs/conventions.md`, `ai_docs/dev_flow.md`, `.claude/skills/imgui-ui/SKILL.md`,
  `ai_docs/roadmap.md`, `ai_docs/features/070_pass_reads/01_spec.md` (the pointer).

## Open questions for the user

None. The maintainer took the triage's defaults for all eleven decisions.

## Manual verification (the maintainer's, no window manager here)

Preamble: items marked (bloom) need the bloom chain, which is a TEST FIXTURE and not a
shipped example. Copy `tests/fixtures/bloom_chain/` into a new
`projects/dev/documents/<uuid>/` and open it once; every (bloom) item then runs against
it. Items 7 and 10 need document throttling OFF (Settings > Throttle documents), because
a throttled document does not re-plan and the cycle cue would lag.

**W1 -- the read-only canvas**

1. Radiance Cascades, `graph`: SIX nodes are drawn, one per pass.
2. The same view: `paint`'s two long wires ride the bus under the row rather than
   crossing the nodes between them.
3. The same view: `jfa` and `cascade` each draw a self-loop into their `prev` port, and
   no other node does.
4. The same view: `jfa` carries `x12` and `cascade` carries `x6`; no other node carries
   a badge.
5. The same view: `composite` alone carries the accent border.
6. Fit from the canvas menu: every node is inside the panel and none is clipped.
7. Wheel over `df`: the point of the picture under the cursor stays under the cursor
   through a zoom in and back out.
8. Middle-drag on empty canvas: the whole picture translates and nothing is selected.
9. Click `seed`: the viewer switches to `seed` and the accent border moves to it.
10. The same click: `seed`'s shader tab is at the front of the editor's tab row, and the
    editor does NOT take keyboard focus (type a letter -- it does not land in the buffer).
11. Double-click `seed`: the editor takes keyboard focus (the same letter lands).
12. After clicking `seed`: every node the output does not need is dimmed, and so are its
    wires.
13. Break `jfa`'s shader (delete a semicolon): `jfa`'s border turns red.
14. With `jfa` broken: `jfa`'s picture still shows its last good frame and carries the
    stale mark.
15. Fix `jfa`: the border returns to normal within a second.
16. On the Uniforms tab, point `paint`'s sampler at `composite`: on the canvas, the two
    wires of the loop turn red and no NODE turns red.
17. Unwire it: both wires return to normal.
18. `strip | graph` on Fire (single pass): one node, no input ports.
19. `strip | graph` on Media Input: one node with two square media ports and no wires.
20. (bloom) The root shows the `bloom` box with the badge `4 passes`.
21. (bloom) The box's input ports are three, one per member sampler reading `scene`, each
    labelled `member.u_scene`.
22. (bloom) The box has exactly one output port, into `final`.
23. (bloom) Double-click the box: the bloom tab draws its four members as solid nodes.
24. (bloom) In the bloom tab: `scene` is a dashed dim ghost on the LEFT and `final` a
    dashed dim ghost on the RIGHT.
25. (bloom) Click the `scene` ghost: the scope returns to the root and `scene` is
    selected.
26. (bloom) The root tab is labelled with the document's own name.
27. (bloom) From the bloom tab's node menu, `Leave group` on all four members: the tab
    closes itself and the root shows four plain nodes.
28. (bloom) Group three of them again, then delete the member the box's picture comes
    from: the box keeps exactly one output port, drawn hollow, and its picture changes to
    another member rather than going blank. (mutations case 2 -- the one case the review
    called a rule break.)
29. (bloom) Put a group label on a pass that reads nothing and is read by nothing: the
    badge counts it, no port appears for it, and the group tab shows it as a lone node.
    (mutations case 4.)
30. Group a single pass: the box draws, it has that pass's own ports, and its tab holds
    one node. (mutations case 6 -- confirm this reads as intended and not as a bug.)
31. Delete the document output while it is inside a box: the accent border moves to a
    root-level node chosen by insertion order, not by the graph. Confirm the jump is
    tolerable. (mutations case 3.)
32. Label two passes on opposite sides of the chain with the same group name: ONE box
    draws at their bounding box and encloses non-members visually; Arrange pulls the
    members together. (mutations case 5 / triage D9.)
33. Right-click a node: the menu carries Settings, Delete and (when grouped) Leave group,
    and nothing else.
34. Right-click empty canvas: the menu carries Add pass, Import..., Fit, Arrange -- NOT
    the node menu. (The `begin_popup_context_item(None)` rule; an explicit id opens the
    last node's menu from anywhere.)
35. `Add pass` from the canvas menu: the new node draws with ZERO input ports and a
    dashed border until it compiles. Type `uniform sampler2D u_seed;` into its shader and
    save: one port appears. (scenarios 8 -- the canvas is a view of wiring, not a
    construction surface.)
36. Start a copilot turn with the canvas open: every control is disabled and the node
    pictures keep updating live.

**W2 -- the interaction**

37. Drag `df` to the right and release: it stays.
38. Switch to another document and back: `df` is still where it was dropped.
39. Restart the app: `df` is still where it was dropped. (The only item that proves D6
    reached disk.)
40. `git diff projects/dev` after 37: exactly one `graph.json` changed, and only `df`'s
    position within it. (One save per drag.)
41. Drag a node until its left edge is within a few pixels of `cascade`'s: a guide line
    draws and the node snaps.
42. Arrange from the canvas menu: every node moves to the rank layout.
43. `git diff projects/dev` after 42: one `graph.json`, every entry carrying a position.
44. Drag from `paint`'s output dot onto `composite`'s `u_cascade` port: the wire moves and
    `composite`'s picture changes.
45. (bloom) Drag from a downstream pass's output onto an upstream pass's port so the drop
    would close a loop: a notification carries the planner's cycle message.
46. After 45: `git diff projects/dev` is empty -- the refusal wrote nothing.
47. Press `cascade`'s `u_df` port and release on empty canvas: the port goes hollow with a
    filled centre and `df`'s node dims.
48. Drop a wire from `df` back onto that port: it fills again.
49. Drop a wire on Media Input's `u_image` port: refused with "bound to media; unbind on
    the Uniforms tab", and the Uniforms tab still shows the bound texture. (The one
    canvas gesture that could destroy user data.)
50. Rubber-band `seed`, `jfa`, `df`; right-click; `Group...`; type `sdf`; Enter: one box
    appears.
51. `git diff projects/dev` after 50: one `graph.json`, three entries changed, one save.
52. Double-click the `sdf` box: its tab shows the three members with `paint` and
    `cascade` as ghosts.
53. Dissolve from the box menu: three plain nodes again.
54. Group a selection under a name an existing pass carries: refused with "a pass and a
    group cannot share a name", and nothing is written.
55. From the gear, rename a pass to an existing group's name: refused with the same
    message, and the pass's file is not renamed on disk.
56. (bloom) Rename a pass so the name rule closes a loop -- with `fx_bright` reading
    `u_bright2` by the name rule, rename `blur` to `bright2`: refused with the planner's
    cycle message, and `ls projects/dev/documents/<id>/passes/` still shows `blur`.
57. (bloom) In the bloom tab, drag a member's output onto a GHOST's port: the write lands
    on a pass the tab does not contain. Confirm this reads as intended. (mutations 14.)
58. (bloom) Rename a pass that a member reads BY THE NAME RULE: the edge disappears and
    the box gains an input port, with no other change. (mutations 8.)
59. Escape while the scope is a group tab: nothing happens (out of scope by design); the
    root tab is the way up.
```

---

## Review history

**Pre-implementation round 1 (2026-09-12): two reviewers, both PARTIAL, no should-not-land.**
Reports: `reviews/pre_correctness_design.md`, `reviews/pre_verification_blast.md`. Folded in:
D11's cycle cue rewritten (the planner names one culprit per cycle whose message carries the
trail, so `cycle_edges` parses it; the old "both endpoints are culprits" rule reddened nothing),
and the cue computed by the canvas rather than read from `graph_errors` (stale under the
throttle); `rank_layout`'s contract stated and Arrange scoped to the whole document; D18's
`wiring_if_renamed` applying `rename_pass_sources` too, under a `finally`; the port list as a
pure `node_ports`; the port-state source named (`uniform_values` by type, as the sampler row);
the compile seam named (`compile_pending_passes`, 091's); the root tab keyed by index with a
fallback label; `graph_views` evicted in `forget_render_state`; the copilot-freeze bracket kept
on the moved buttons; ghost and box menus stated; `GRAPH_EDGE` in the tint exclusions; every
gesture routed through one App verb so its refusal is testable headlessly (`drop_wire`,
`commit_node_drag` over a pure `NodeDrag`, `group_selection`, `dissolve_group`,
`arrange_graph`); D17 widened to one `group_name_error` funnel that `plan_import` also calls;
triage S15 (the dialog's flatten line) restored; the prose-budget line made true
(`menu_item_simple` enters the gate, the toggle's labels are listed unmeasurable); the test
list rewritten with a falsifier per invariant; the smoke stretch specified; the manual list
rewritten so each item fails for one reason, with the bloom preamble (the fixture is not a
shipped example) and the twelve cases the brainstorm reviews implied. The D11 stale-mark
sentence was corrected by the implementer: `preview_cell`'s docstring names a corner tick its
code never draws. Rejected: nothing. False trails both reviewers recorded stand.
