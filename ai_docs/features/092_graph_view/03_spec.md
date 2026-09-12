# 092 — The graph view

Status: **spec, awaiting pre-implementation review.** Sketches: `00_mock.html` (round 3 is the
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
are `sampler_names(pass)` in declaration order with `prev` (a self-read) last, its edges are
`Document.effective_wiring()`. A port is therefore never drawn for a sampler the program no
longer declares, which is what stops `set_sampler_source` (which does not validate the uniform)
writing a dead row. A never-compiled pass has no ports, so **the canvas compiles what it draws**:
on a document's first canvas frame, every pass whose `program is None` with no recorded errors
is compiled (bounded, the largest document is six passes); the first-render sweep of 066 D1 is
left as it is for the strip.

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
nothing off-draw writes it.

**D3. The scope and the tab row.** `GraphViewState.scope` is `""` (the root) or a group name.
A `text_tab_row` at the canvas top lists the document's display name (the root) then every group
name in strip order of its first member; clicking sets the scope. The scope is revalidated every
frame: a name no pass carries falls back to `""` (the last member can leave from inside the tab).
Double-click on a box enters it; the root tab is the way up (no Escape, out of scope). A scope
change refits the view once.

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
placed: `pass_graph.rank_layout` places the unplaced passes every frame (rank = longest path from
a root over the non-self edges, cycle members rank 0; within a rank, group members adjacent, then
by the mean position of predecessors, then strip order; columns left to right at
`GRAPH_GAP_X`, rows at `GRAPH_GAP_Y`, each column centred on the tallest). A placed pass keeps its
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
`[0.25, 2.5]`.

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
scope change, and from the canvas menu. Arrange (canvas menu) writes `rank_layout` over every
visible pass into `position` through `set_pass_positions` and fits. Left-drag on empty canvas is
the rubber band (W2) and does nothing in W1.

**D10. Click and menus (W1).** Click a node: `pick_pass(document_id, name, focus_editor=False)`,
the strip's verb. Double-click a node: the same with `focus_editor=True`. Click a box: pick the
bundle output; double-click: enter. Right-click a node: the strip's item set, extracted from
`pass_list._draw_context_menu` into `pass_list.pass_menu_items(app, document_id, name)` so the
two surfaces cannot drift (Settings, Delete, Leave group); the strip keeps its own
`begin_popup_context_item` with its explicit id, which is safe there because each tile is its
own window. Right-click a box: Open, and Dissolve (W2). Right-click empty canvas: Add pass,
Import..., Fit, Arrange. The whole canvas sits under `begin_disabled(app.copilot_turn_active)`
like the strip; the draw list still paints the live pictures.

**D11. The error language.** Port dots carry state by shape: filled disc = wired; hollow ring =
unfilled (black by default, not a fault); ring with a filled centre = `NoSource`; double ring =
`prev`; a media-bound sampler draws a small square dot (D15). Node border: `STATE_ERROR` when
`compile_unit.errors`, else `ACCENT_PRIMARY` when the pass is the document output, else
`BORDER`; a never-compiled pass has a dashed border. A node the output does not need (the
strip's `live` set, `evaluation_order(wiring, output) or {output}`) dims its name and its edges.
The edges between two passes that `graph_errors` both name as cycle culprits (`message`
starting with "passes form a cycle") are `STATE_ERROR`; victims get nothing. A box carries
`STATE_ERROR` when any member has a compile error, and nothing for non-convexity. No cue for
dtype / scale / filter / wrap. A compile-erroring pass's picture is its last good frame; the
node draws the strip's stale mark on it.

**D12. Wires (W2).** Press on an output dot and release over an input port writes
`set_sampler_source(document_id, consumer, sampler, PassSource(producer))`; release over empty
canvas writes nothing (the drag came from an output). Press on a **filled** input port grabs its
wire: release over another input port moves it (`PassSource` there, `NoSource` on the original),
release over empty canvas writes `NoSource` on the original. A drop that would close a cycle is
refused before any write: `pass_graph.wiring_with(wiring, consumer, sampler, producer)` builds
the hypothetical wiring, `plan_passes` runs, and any error refuses with the planner's message
through `notifications`. A node's output into its own port is feedback, allowed. A drop on a
media-bound port is refused ("bound to media; unbind on the Uniforms tab", D15). A drop onto a
port wired by the name rule materializes an explicit `PassSource`, since the drop writes one
anyway. A ghost's ports are drop targets and drag sources like any other, since a ghost is a
real pass drawn dimmed (the write names the pass, the tab is only a viewport). The wire in flight
is a bezier from the source to the cursor on channel 2.

**D13. Drag (W2).** Press on a node body and move: the node follows `io.mouse_delta / zoom`;
release writes its position through `set_pass_positions`, one save. A box drags every member
by the same delta. A drag moves every selected node together when the pressed node is selected.
During a drag, snapping aligns the moving node's left edge or top edge to any other visible
node's within `GRAPH_SNAP_PX` screen pixels, and draws the guide line on channel 2.

**D14. Selection and Group (W2).** `GraphViewState.selection: set[str]` of pass names (a box
selects its members). Left-drag on empty canvas draws the rubber band from
`io.mouse_pos - get_mouse_drag_delta` and selects every node whose rect intersects it on
release; shift-click toggles one; click on empty clears. Right-click with a selection adds
`Group...` to the node menu: a popup with a name input (Enter or Create commits, Cancel or Esc
cancels, per imgui-ui 7.5) that writes `set_pass_groups(document_id, names, group)`, a new
session verb that validates once (the group pattern, and D17's collision) and saves once.
Selecting a box and grouping it with others rewrites its members to the new label (flat
labels, so the old group dissolves into the new one). Dissolve on a box menu is
`set_pass_groups(members, "")`. `set_pass_group` stays for the modal and the copilot and calls
the batched verb with one name.

**D15. A media-bound sampler's port.** `sampler_names` lists it and the wiring does not (a bound
texture is not a `PassSource`), so it draws as a port in a media state (a square dot, no wire)
and a drop on it is refused (D12): releasing the bound texture from a drag is the one write the
canvas must not make silently.

**D16. Dissolve is required.** It is the inverse of Group, which is N writes with no undo. A box
gets no Delete verb; a member is deleted from its own node menu with the strip's two-click arm
(`delete_pass` + `close_editor_for_path`, exactly `pass_list._delete_pass`).

**D17. One namespace for passes and groups.** `set_pass_groups` rejects a group named like an
existing pass; `add_pass` and `rename_pass` reject a pass named like an existing group. The
root keys nodes and boxes by name. Message: "a pass and a group cannot share a name".

**D18. A rename plans before it moves.** `rename_pass` computes the wiring the document would
have after the rename (`Document.wiring_if_renamed(old, new)`, which swaps the pass under the
new key, reads `effective_wiring()`, and restores) and refuses when `plan_passes` reports a
cycle, with the planner's message. Today a rename can create a name-rule edge that closes a
loop with no check; the drag has the guard (D12), so the rename gets the same one.

**D19. What the copilot sees: nothing new.** No tool accepts or reports a position; the pass
table stays flat; no groups paragraph in the prompt; no `group_passes` tool. The dogfood
harness needs nothing: the layout and the boundary are pure functions in `pass_graph.py`.

**D20. Docs.** `conventions.md` "A pass GROUP is a label ... and nothing folds" gets the revisit
pointer: the strip stays flat; the graph view contracts a group to its boundary edges and never
orders the box, so convexity is not a rule anywhere. A new entry records D6 (a position is
written only by a placement, never by a draw) and D1 (ports from the program, edges from the
wiring). The imgui-ui skill §8 gets the two canvas rules of D8 (the allow-overlap chain; the
context-item id). `dev_flow.md`'s module map gets `widgets/pass_graph.py` and
`widgets/graph_state.py`. The Help panel's Passes section gets one sentence: a port exists
because the shader declares a sampler. 070's spec gets its pointer to this feature.

## Files touched

- `shaderbox/pass_graph.py`: `PassEntry.position` + `GraphCoord` + `MAX_GRAPH_COORD`,
  `PassGraph.with_positions`, `rank_layout`, `graph_ranks`, `group_boundary` (+ `Boundary`,
  `BoxPort`), `bundle_output`, `wiring_with`, `cycle_message_for` (the refusal text from a plan).
- `shaderbox/document.py`: `wiring_if_renamed`.
- `shaderbox/project_session.py`: `set_pass_positions`, `set_pass_groups`, `set_pass_group`
  routed through it, the D17 checks in `add_pass` / `rename_pass` / `set_pass_groups`, the D18
  guard in `rename_pass`, `import_passes` stripping positions.
- `shaderbox/ui_regions.py`: `PassesView`, `PASSES_VIEW_LABELS`.
- `shaderbox/ui_models.py`: `UIAppState.passes_view`.
- `shaderbox/theme.py`: `SIZE.GRAPH_*`, `COLOR.GRAPH_EDGE`, `COLOR.GRAPH_GHOST_ALPHA`.
- `shaderbox/widgets/graph_state.py` (new): `GraphViewState`, `WireDrag`, `NodeDrag`.
- `shaderbox/widgets/pass_graph.py` (new): the canvas.
- `shaderbox/widgets/pass_list.py`: caption and buttons move out; `pass_menu_items` extracted.
- `shaderbox/tabs/document.py`: the caption row with the toggle, the dispatch, the button row.
- `shaderbox/app.py`: `graph_views`, `graph_view_for`, `open_pass_settings` unchanged.
- `shaderbox/help_content.py`: one sentence.
- `scripts/smoke.py`: a stretch with the graph view on for the multi-pass document, a group
  scope, Arrange, and back.
- `tests/test_pass_graph.py`: `rank_layout` (producers left of consumers, deterministic,
  group adjacency, only unplaced names placed, cycle members not lost), `group_boundary` and
  `bundle_output` over the bloom shape, the non-convex shape, a generator box, a one-member
  group, a split group, a member whose sampler reads nothing; `wiring_with` + the cycle refusal;
  `position` bounds (NaN, inf, 1e30 rejected; a pair accepted).
- `tests/test_graph_persistence.py`: `position` round-trips and a corrupt position costs only
  itself through `load_graph`.
- `tests/test_pass_verbs.py` (or the existing verb tests): `set_pass_positions` saves once and
  survives every other verb; `set_pass_groups` saves once and rejects D17; `rename_pass`
  refuses the cycle it would create (a mutation test: with the guard removed the rename
  succeeds and the plan reports the cycle); `import_passes` leaves positions `None`.
- `tests/test_button_tiers.py`: `widgets/pass_graph.py` `invisible_button` allowlisted (the
  canvas and node hit rects).
- `tests/test_ui_prose_budget.py`: the canvas menu labels are within budget; nothing to allow.
- `ai_docs/conventions.md`, `ai_docs/dev_flow.md`, `.claude/skills/imgui-ui/SKILL.md`,
  `ai_docs/roadmap.md`, `ai_docs/features/070_pass_reads/01_spec.md` (the pointer).

## Open questions for the user

None. The maintainer took the triage's defaults for all eleven decisions.

## Manual verification (the maintainer's, no window manager here)

1. Radiance Cascades, `graph`: six nodes in six columns, `paint`'s two long wires on the bus
   under the row, `prev` loops on `jfa` and `cascade`, `×12` and `×6` badges, `composite` in the
   accent border. Fit shows the whole chain at the panel's width.
2. Wheel over `df`: the node under the cursor stays under it. Middle-drag pans. Fit from the
   canvas menu restores.
3. Click `seed`: the viewer switches to `seed`, the shader tab opens, the border moves, the
   later nodes dim with their wires. Double-click `seed`: the editor takes focus.
4. Bloom Chain imported into a host with a `scene` and a `final`: the root shows the `bloom` box
   with `4 passes`, one input port per member sampler reading `scene` (named `member.u_scene`),
   one output into `final`. Double-click the box: the bloom tab, `scene` a ghost on the left,
   `final` a ghost on the right, both dashed and dim. Click the `scene` ghost: back at the root
   with `scene` selected. The root tab is labelled with the document's name.
5. From the bloom tab's node menu, `Leave group` on every member: the tab closes itself and the
   root shows four plain nodes.
6. Break `jfa`'s shader: its border goes red, its picture keeps the last frame with the stale
   mark. Fix it: the border returns.
7. Wire `paint`'s sampler to read `composite` on the Uniforms tab: the two wires of the loop go
   red on the canvas; the strip goes grey as before. Unwire: they return.
8. `strip | graph` on a single-pass document: one node, no ports for Fire, two square media
   ports for Media Input.
9. (W2) Drag `df` to the right and release: it stays after a document switch and a restart. A
   drag near `cascade`'s column snaps with a guide. Arrange puts it back.
10. (W2) Drag from `paint`'s output dot onto `composite`'s `u_cascade` port: the wire moves, the
    picture changes. Drag from `composite`'s output onto `paint`'s port (Fire has none; use the
    bloom host): refused with the cycle message, nothing changes on disk.
11. (W2) Press `cascade`'s `u_df` port and release on empty canvas: the port goes hollow and
    `df`'s node dims. Drop a wire from `df` back.
12. (W2) Rubber-band `seed`, `jfa`, `df`; right-click, Group..., `sdf`: one box appears; its tab
    shows the three with `paint` and `cascade` as ghosts. Dissolve from the box menu: three nodes
    again. Group a selection under a name that is a pass: refused. Rename a pass to a group's
    name from the gear: refused.
13. (W2) Rename `blur` to `bright2` in the bloom host so that `fx_bright.u_bright2`... (a rename
    that closes a loop): refused with the cycle message and the file is not moved.
14. (W2) Drop a wire on Media Input's `u_image` port: refused, the texture stays.

## Review history

(pre-implementation review pending)
