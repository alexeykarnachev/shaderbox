# 093 — Tenth walk findings

Status: **in progress; wave 1 spec written, pre-implementation review next.** Five findings
filed (`00_findings.md`). The maintainer's verdict on the shipped canvas ("feels very cheap")
sent the walk into research first: `02_research_brief.md` is the brief, `research/` holds six
area reports against primary sources, `03_graph_design.md` is the design record (G1-G18, what
stays, the four forks G-Q1..Q4, the false trails, a verification sketch), and
`01_research_graph_tab.md` records the tab machinery the graph moves into (mock C of
`00_mock_panel.html`, his pick). On the four forks he said: "It is hard to answer without the
actual visuals. Let's implement everything, and then we'll tune." So every fork ships at the
record's recommendation and is tuned by him against the rendered result.

This file is the implementation contract for wave 1. The record carries the rationale and the
sources; where this spec refines a "Code" paragraph of the record, the refinement is stated
under `## Refinements over the record` with its reason, so a reviewer can see the edge.

Source: `../TODO` (2026-09-13) and the screenshot of the control panel he took with it, verbatim
in the ledger.

## How this walk runs

- A batch of findings he reports in one message is one **wave**: research each finding against
  the code, file it in the ledger with what the code does and why, fix the small ones together,
  run `make gates` once, commit once, update this spec's wave list and the ledger's "Landed in"
  column in that commit.
- A finding whose fix is a feature by the `dev_flow.md` size preamble (a new module, a real
  behavior change, a reshape of a persisted model) gets its own numbered feature with the usual
  flow; the ledger row points at it and this spec lists it under the wave as delegated.
- A visual call stays his: no window manager here, so a fix to how something looks is described
  (what changed, where to look) and he judges it in the next batch.
- A settled decision (092's D-list, `conventions.md`) stays settled; a finding that shows one
  wrong reverses it explicitly in that feature's Review history, never silently.

## Goal

The graph editor becomes a surface the maintainer would call solid: one cusp-proof curve family
for every wire and no bus; a wire that can be selected, and deleted by its mid-curve ✕ or the
Delete key; an exclusive hover model (port, node, wire, background) with a distinct hover hue
drawn as a halo under the stroke; the feedback loop replaced by a glyph in the badge row; a card
wide enough for his own longest names with an ellipsis for anything longer; a named
click-versus-drag threshold; a cursor for the three gestures. And the canvas moves out of the
Document tab's leftovers into the editor pane as a third tab kind, one graph tab per document,
so it gets the pane's whole height. Findings 1, 2, 3 and 5 close in this wave; finding 4 is
delegated (below).

## Out of scope

- **Finding 4, the control panel's composition** (the document grid's dead space, the stacked
  rows). Its own feature with its own mock; mock A's toolbar consolidation is separable from the
  tab move and stays there. Trigger: the maintainer's next batch after this wave renders, when he
  can judge the panel with the graph gone from it.
- **Reroute / waypoint nodes** (G16). Trigger: he reports a wire he cannot read on a real
  document AFTER this wave lands, i.e. after the bus and the cusp are gone.
- **Routing around nodes, edge bundling** (G15): none of the references does it; z-order and the
  user moving nodes is the converged answer. No trigger; a request from him reopens it.
- **Delete for passes from the canvas** (G17): a pass delete stays the node menu's armed
  two-click through `pass_menu_items` (092 D10, D16). Trigger: he asks for the key.
- **A menu entry to disconnect a wire** (G-Q2, shipped without): the ✕ and Delete are the two
  paths. Trigger: he reports the ✕ hard to hit at low zoom despite the 6px screen floor.
- **Keys for Fit and Arrange, Escape cancelling a live gesture** (G14): not asked for. Trigger:
  he asks.
- **Auto-width cards** (G11 alt. a): a layout pass over every label under a pushed font, and it
  would make `node_size` depend on fonts. Trigger: a real document whose names ellipsize at 128
  in a way he objects to after seeing it.
- **Blender's flatness correction / Rete's vertical term for a long near-horizontal wire** (G1
  alt. d): the references disagree and neither addresses the defect. Trigger: a screenshot of a
  wire reading conspicuously flat.
- **Tab persistence across restarts**: tabs are not persisted today (only the current document's
  shader tab reopens), and the graph tab follows that rule.

## Design decisions

Numbered `T` for the tab move and `S` for the spec's own calls; the canvas rules themselves are
the record's **G1-G18, adopted as locked constraints by reference** (a reviewer checks the
implementation against the record's Rule paragraphs; this spec does not restate them). The four
forks are closed here as S1.

**S1. The forks ship at the record's recommendation.** `GRAPH_NODE_W = 128` (G-Q1), no menu
entry for a wire (G-Q2), `GRAPH_DRAG_LOCK_PX = 4.0` (G-Q3), `COLOR.GRAPH_HOVER = _P["blue_b"]`
(G-Q4). Each is one token or one absent item, tuned by the maintainer after he sees it; a change
he asks for is a token edit, not a re-lock.

**T1. The graph is an editor-tab kind.** `EditorTabKind` gains `"graph"`. A document's graph tab
is `EditorTab(path=app.paths.graph_json_for(document_id), kind="graph", document_id=document_id)`:
the path is a real per-document file, so every path-keyed pass-through (`_focus_or_add_tab`'s
dedupe, `_tab_id_suffix`, `close_editor_for_path`, `_close_tab_for_path`) works unchanged, and
the pass-rename hook, which re-keys by pass path, never matches it. A graph tab has no
`EditorSession`: `is_tab_dirty` finds none and answers False, `is_current_editor_dirty` likewise,
`formatter_for("graph")` is `None` (no table entry), `jump_to_next_error` and
`format_current_editor` return on the missing session, `_drain_editor_input` returns on it, and
the flush paths skip it. `_on_document_deleted` filters by `document_id` and so drops it;
`forget_render_state` already drops `graph_views`. This reverses 092 D2; 092's spec records the
reversal in its Review history and D2 gets a one-line pointer at the overruled passage, the way
083 D5 carries 085's.

**T2. The dispatch is one branch in `tabs/code.py::draw`, right after the tab row.** When the
active tab's kind is `"graph"`, a `_draw_graph_tab(app, tab)` runs and `draw` returns before the
session fetch. The branch does the pane's focus bookkeeping the text body does, and nothing
else of it: consumes `editor_focus_requested` when no popup is open (ui.py has already issued
the `set_next_window_focus`), sets `app.editor_focused` from
`imgui.is_window_focused(FocusedFlags_.child_windows)` so `Ctrl+W` closes the graph tab and
`Ctrl+Tab` cycles onto it (they gate on that flag), consumes `editor_defocus_requested` as the
text body does, and sets `app.editor_errors = []` so a stale error list from a previous tab does
not drive `F8`. Then `pass_graph.draw(app, tab.document_id)`. `draw_chrome` needs no branch: its
non-shader path already prints `tab_label`, and the dirty read is False.

**T3. The canvas fills its host.** `pass_graph.draw(app, document_id)` loses its `height`
parameter: the scope row draws, then `begin_child("##pass_graph", size=(0, 0), ...)` takes the
rest of the host's content region. `SIZE.GRAPH_MIN_H` stops being read and is deleted. The
widget still positions no sibling and measures none (092 D2's leaf rule, kept).

**T4. Opening it.** `App.open_graph_for(document_id, focus_editor=False)`, the sibling of
`open_script_for`: `_focus_or_add_tab(EditorTab(...graph...), focus_editor=focus_editor)`,
frozen mid-copilot-turn like its sibling. `CommandId.OPEN_GRAPH`, label `"Open graph"`, chord
`Alt+G` (unbound today; `Alt+G` is no vim or standard-keymap chord), category `C.EDITOR`, the
same default scope as `OPEN_SHADER` / `OPEN_SCRIPT`, handler
`lambda: self.open_graph_for(self.current_document_id, focus_editor=True)`. The registry
coverage test then requires the label in the help shortcuts, which the generated section
provides.

**T5. The Document tab keeps the strip and gets an `open` for the graph, like the Script row.**
`tabs/document.py::_draw_passes` draws `_entry_row_label(graph_active, "Passes")` -- the accent
tick marks a graph tab of THIS document as the editor's active tab, exactly as the Script row's
tick does -- then `standard_button("open##entry_graph")` (tooltip `"Open the pass graph"`)
calling `app.open_graph_for(document_id, focus_editor=True)`, then the strip, then the
`add pass` / `import...` row, each bracket in `begin_disabled(app.copilot_turn_active)` as
today. The `strip | graph` `segmented_choice` goes; `PassesView` and `PASSES_VIEW_LABELS` are
deleted from `ui_regions.py`; `UIAppState.passes_view` is deleted from `ui_models.py`
(`load_model`'s per-key salvage drops the retired key from any saved `app_state.json`; the
sandbox's `projects/dev/app_state.json` carries no such key, verified). `tests/test_ui_regions.py`
tested only the retired enum and is deleted. `scripts/smoke.py`'s graph frames open the tab
through `open_graph_for` instead of the preference (T6).

**T6. The smoke exercises the tab.** Frame 43 calls `app.open_graph_for(multi)` where it set
`passes_view = GRAPH`; frames 45-47 keep their asserts (fitted, scope survival, scope
revalidation, draw-writes-nothing, Arrange saves once); frame 47 ends with
`app.close_editor_for_path(app.paths.graph_json_for(multi))` and asserts no tab of kind
`"graph"` remains, exercising the session-less teardown; frame 48's document switch is
unchanged. The editor-drew assertion at the tail still holds because the shader tab draws every
other frame of the run.

**S2. Hover and selection of a wire share one identity: the `(owner, sampler)` the wire
terminates at.** That pair identifies a wire uniquely in the whole document -- a sampler has one
source (072) -- and it is the identity `App.unwire` takes, so `GraphViewState.hovered_wire` and
`selected_wire` are both `tuple[str, str] | None` and compare directly. `_Edge` gains `owner`
and `sampler` (the consumer pass and its sampler; for a box edge the member behind the box slot,
for a ghost-reader edge the reader itself). `selected_wire` is revalidated every canvas frame
against `picture.edges` and cleared when no drawn wire carries it (an unwire, a pass delete, a
scope change).

**S3. Hover is read one frame late and written fresh each frame.** The draw runs before the hit
rects (the ports' positions come from the same geometry), so `_draw_canvas` reads last frame's
`view.hovered_node: str | None` (a node key), `hovered_port: tuple[str, int] | None` and
`hovered_out: tuple[str, int] | None` (node key, slot) and `hovered_wire` at draw time, and the
hit-test section writes this frame's values at its end, resolved exclusively in G6's order: a
port or output dot (`is_item_hovered()` on the `gport_*` / `gout_*` buttons; during a wire
drag the drop target counts as the hovered port), else a node body, else the nearest wire under
G4's threshold, else nothing. Every field is written every frame, `None` included, so a mouse
that left the canvas leaves no stale cue.

**S4. Selection happens on press, wire and node alike; a click is resolved against the drag
lock.** Today a node's `_click` fires on `is_item_clicked` (press) and a drag may follow, so a
drag also picks. G13's falsifier requires the opposite: a press that becomes a drag must not
`pick_pass`. So a node's click fires on the release frame -- `is_item_deactivated()` while
`is_mouse_released(left)`, with `view.node_drag is None and view.wire_drag is None` (the drag
blocks run after the node loop, so on the release frame a drag that happened is still set) and
`not blocked` -- and the double-click stays on `is_mouse_double_clicked` as today. The background
press is recorded as `bg_pressed = is_item_clicked(left)` on the background button and acted on
AFTER the node loop, when this frame's node and port hover are known: with a wire under G4's
threshold and no node or port hovered, `selected_wire` becomes that wire and `view.selection`
clears unless Shift is held; with no wire, both clear unless Shift is held. Selecting a node
clears `selected_wire`; a rubber band's release clears it too. The two selections are exclusive
so one Delete has one target.

**S5. Delete and Backspace are read locally, once, in `_draw_canvas`.** Fired when
`imgui.is_key_pressed(imgui.Key.delete) or imgui.is_key_pressed(imgui.Key.backspace)`, the
canvas child is hovered (`hovered`, the existing `is_window_hovered(child_windows)`), no imgui
item is active (`not imgui.is_any_item_active()`), not `blocked`, and `selected_wire` is set;
the write is `app.unwire(document_id, *view.selected_wire)`, its refusal toasted as `_drop`'s
is. With a node selection and no wire, the key does nothing (G17). Both key names are present on
this build's `imgui.Key` (verified).

**S6. The ✕ lives on the top channel and its button is the last submitted.** G12's channel 1
would let a card that covers the wire's midpoint hide the ✕ while its button, submitted last,
still won the click there -- a mark that wins a click must be the mark that is visible. So the
channels are: 0 wire halos, 1 wires (crisp strokes), 2 nodes, 3 the in-flight wire, 4 the
overlays (the selected wire's ✕, the rubber band, the snap guides). `channels_split(5)` once at
the top of `_draw_canvas`, `channels_merge()` once at its end, after the guides; the split stays
open across the hit-test section, which submits only invisible buttons and popups (a popup is
its own window with its own draw list, so the canvas list's splitter is untouched). The ✕ is
drawn per G5 (a `BG_APP` disc, a 1px `SELECT` ring, two `add_line`s) and clicked through one
`invisible_button` at `(centre - h, centre - h)` with `h = max(GRAPH_WIRE_X_R * z,
float(GRAPH_HIT_MIN))`, submitted after every node, port and output button; its rect is exposed as
`view.x_rect` for the headless test, the way `port_rects` is.

**S7. Bring-to-front orders the buttons too.** The nodes are drawn in ascending
`(is_selected, is_being_dragged)` order (G12), and the hit-test loop submits the node buttons in
the SAME order, so the card that paints on top is the one whose button, submitted later, wins
the overlap. One list, sorted once per frame, used by both loops.

**S8. The fit frames the wires as well as the nodes.** A backward wire's S-curve bulges past the
producer's right edge and the consumer's left edge by a fraction of its offset, which the
fit-margin does not cover, so `_fit` takes the nodes AND the edges and frames the union of the
nodes' bounding box and every wire's four control points (a cubic lies inside its control
polygon's hull, so the four points bound the curve). The control points are computed in canvas
space through the same `wire_points` at zoom 1 -- the offset and the distance both scale
linearly with zoom, so the canvas-space points are the screen-space ones divided by the zoom.
This is what closes finding 5 by construction; G3's "no wire leaves the nodes' bounding box" is
not literally true of an S-curve and is replaced by "no wire leaves the fitted view". Note also
that `_fit` clamps the zoom at 1.0 (`min(1.0, ...)`), so G11's fit-zoom table describes the
unclamped ratio; the verifiable claim is that a six-column chain fits a 1225px pane at zoom 1.0
and a 740px pane above 0.6.

**S9. The pure geometry lives in `graph_state.py`, importable without imgui.** New free
functions beside `NodeDrag` and `node_size`:
- `wire_points(a, b, zoom) -> tuple[Point, Point, Point, Point]`: G1's formula over two screen
  points, returning `(p0, cp0, cp1, p3)`. No branch on the sign of `dx`.
- `bezier_point(p0, cp0, cp1, p3, t) -> Point`.
- `wire_hit_threshold(zoom) -> float`: `max(float(SIZE.GRAPH_WIRE_HIT_MIN),
  SIZE.GRAPH_WIRE_W * 2.0 * zoom)` (G4, screen px).
- `wire_hit(mouse, points, threshold, segs) -> float | None`: the bounding-box reject expanded by
  the threshold, then the minimum point-to-segment distance over `segs` segments, or `None` when
  over the threshold.
- `class WireState(StrEnum)`: `ERROR`, `SELECTED`, `HOVERED`, `DIM`, `NORMAL`, and
  `wire_state(on_cycle, selected, hovered, dim) -> WireState` in G18's precedence (error first).
  The widget maps a state to its stroke color and halo; the precedence is the pure part.
`_draw_wire(dl, points, col, halo_col)` in the widget draws the halo (`GRAPH_WIRE_W * 3.0 * z`
at the state's halo alpha, channel 0) when `halo_col` is not `None`, then the crisp stroke
(channel 1) -- a hovered stroke is `GRAPH_WIRE_W * 1.4 * z` per G6, a selected or normal one
`GRAPH_WIRE_W * z`, all floored at 1.0. The caller resolves the state; `_draw_wire` never reads
it (G18). The in-flight wire uses the same `wire_points`, on channel 3, in `ACCENT_PRIMARY`.

**S10. The badge row hosts the feedback glyph.** `_draw_badge` returns the pill's width so
`_draw_feedback_glyph` can sit `GRAPH_FB_GAP * z` left of an `xN` badge when one is drawn that
frame and flush at the picture's top-right otherwise (G8). The glyph is two open arcs through
`path_arc_to` + `path_stroke(col, thickness)` -- `path_stroke`'s flags default to `0`, so no
`ImDrawFlags_` constant is needed. Ghost nodes draw no badges today and draw no glyph.

**S11. The ellipsis helper goes public.** `ui_primitives._ellipsize` is renamed `ellipsize`
(its four call sites inside `ui_primitives` follow) so the canvas imports it without reaching
for a private name. The name budget and the port-label budget are G11's, measured inside the
same pushed-font scope as the `calc_text_size` that positions the text.

**S12. Theme tokens.** `SIZE`: `GRAPH_NODE_W 108 -> 128`, `GRAPH_THUMB 80 -> 96`,
`GRAPH_PAD 6 -> 8`, `GRAPH_NAME_H 18 -> 20`, `GRAPH_PORT_ROW 16 -> 18`; new `GRAPH_WIRE_BOW:
float = 0.40`, `GRAPH_WIRE_MIN_OFF: int = 24`, `GRAPH_WIRE_HIT_MIN: int = 6`,
`GRAPH_WIRE_HIT_SEGS: int = 24`, `GRAPH_WIRE_X_R: int = 7`, `GRAPH_FB_SIZE: int = 12`,
`GRAPH_FB_GAP: int = 2`, `GRAPH_DRAG_LOCK_PX: float = 4.0`; deleted `GRAPH_BUS_STEP`,
`GRAPH_BUS_CLEAR`, `GRAPH_LOOP_RISE`, `GRAPH_LOOP_REACH`, `GRAPH_MIN_H`. `COLOR`: new
`GRAPH_HOVER = _P["blue_b"]`, `GRAPH_HOVER_HALO_ALPHA: float = 0.35`,
`GRAPH_SELECT_HALO_ALPHA: float = 0.55`; `GRAPH_HOVER` joins `_GROUP_TINT_EXCLUSIONS` and the
import-time invariants gain `GRAPH_HOVER != SELECT`, `!= STATE_ERROR`, `!= GRAPH_EDGE` (three
cues that meet on one wire). The module-local `_MIN_DIRECT_DX` and `_BEZIER_BOW` in the widget
are deleted in favor of the tokens. The comment block over the `SIZE.GRAPH_*` tokens is
rewritten for the tokens as they are (no bus, no loop).

**S13. The drag lock is passed at every site.** `imgui.is_mouse_dragging(MouseButton_.left,
SIZE.GRAPH_DRAG_LOCK_PX)` at the four sites (the rubber band, the node body, the port press that
moves the node, the output dot), and `imgui.get_mouse_drag_delta(MouseButton_.left,
SIZE.GRAPH_DRAG_LOCK_PX)` where the band anchor is derived, so the anchor and the lock agree.
No other `is_mouse_dragging` call in the module may omit the threshold (the test greps for it).

**S14. Cursors.** `App.__init__` creates `hand_cursor` (`glfw.HAND_CURSOR`) and
`crosshair_cursor` (`glfw.CROSSHAIR_CURSOR`) beside the three existing ones; `_draw_canvas`
requests `app.want_cursor = app.hand_cursor` while `panning` or `view.node_drag is not None`, and
`app.crosshair_cursor` while `view.wire_drag is not None` (G7). Requests only; `ui.py` applies
once per frame on change.

## Refinements over the record

Where the record's "Code" paragraphs and this spec differ, this spec wins, for these reasons:

| Record | Spec | Why |
|---|---|---|
| G4: `hovered_wire` is the edge's `(src_key, src_slot, dst_key, dst_slot)` | S2: `(owner, sampler)`, the same identity as `selected_wire` and `App.unwire` | One identity for hover, selection and the verb; comparable without a lookup |
| G5: a wire is selected by "press and release with no drag past the lock" | S4: on press, as a node is | The node path selects on press today and a band may follow either; the drag lock is what a CLICK (pick_pass) is resolved against, and that is where G13's falsifier bites |
| G5/G12: the ✕ on channel 1 with the wires | S6: on the top channel with the band and guides | The button that wins the click must be the mark that is visible |
| G3: "no wire leaves the nodes' bounding box" | S8: the fit frames nodes and wire control points | A backward S-curve bulges past both cards; framing the control polygon is the construction that actually closes finding 5 |
| G11's fit-zoom table | S8's note: `_fit` clamps at 1.0 | The table is the unclamped ratio; the test asserts the clamped fact |
| G4/G18: helpers as module-level functions in the widget | S9: in `graph_state.py` | The pure tests import no imgui and need no `app` fixture |
| G8: `path_stroke(col, ImDrawFlags_.none, thickness)` | S10: `path_stroke(col, thickness)` | The binding's signature is `(col, thickness=1.0, flags=0)` (verified) |

## Waves

**Wave 1 (this spec): the graph editor's redesign and its move into the editor pane.** Findings
1, 2, 3, 5 closed; finding 4 delegated to its own feature. One feature flow: pre-implementation
review (two reviewers), implementation as one diff, post-implementation review to convergence
(three reviewers per round, a spec-fidelity audit among them), sanitize, the maintainer's
hands-on pass.

## Files touched

- `shaderbox/editor_types.py` -- `EditorTabKind` gains `"graph"`.
- `shaderbox/tabs/code.py` -- `tab_label` for the graph kind; the `_draw_graph_tab` branch (T2).
- `shaderbox/tabs/document.py` -- `_draw_passes` per T5; the `PassesView` import and
  `GRAPH_MIN_H` read go.
- `shaderbox/app.py` -- `open_graph_for`, the `OPEN_GRAPH` handler, the two cursors.
- `shaderbox/commands.py` -- `CommandId.OPEN_GRAPH` and its spec.
- `shaderbox/ui_regions.py` -- `PassesView` / `PASSES_VIEW_LABELS` deleted.
- `shaderbox/ui_models.py` -- `UIAppState.passes_view` deleted.
- `shaderbox/theme.py` -- S12.
- `shaderbox/ui_primitives.py` -- S11.
- `shaderbox/widgets/graph_state.py` -- S9's functions and `WireState`; the new
  `GraphViewState` fields (`hovered_node`, `hovered_port`, `hovered_out`, `hovered_wire`,
  `selected_wire`, `x_rect`, `wire_mids: dict[tuple[str, str], Point]` -- each drawn wire's
  screen-space midpoint this frame, for the headless test to aim a click at).
- `shaderbox/widgets/pass_graph.py` -- the canvas: G1-G3 (`_draw_wire` rewritten, bus and
  `_draw_self_loop` deleted, the edge loop simplified), G4-G6 (the wire pass, the hover reads
  and writes, the halos, the per-dot colors), G7 (cursor requests), G8 (`_draw_feedback_glyph`),
  G11 (ellipsis), G12/S6/S7 (five channels, the sorted node list), G13/S13 (the lock), S2-S5,
  S8, T3 (`draw(app, document_id)`). The module docstring is updated for the hover model, the
  wire pass and the channel layout.
- `scripts/smoke.py` -- T6.
- `tests/test_graph_state.py`, `tests/test_graph_view.py`, `tests/test_theme.py`, new
  `tests/test_graph_tab.py`; `tests/test_ui_regions.py` deleted.
- Docs, same wave: `ai_docs/features/092_graph_view/03_spec.md` (D2's pointer + Review history
  entry for the reversal), `ai_docs/conventions.md` (the 092 bullet: the canvas's home is the
  editor tab, the wire is one bezier by construction, the hover model; the "revisit the canvas's
  home" clause resolved), `ai_docs/dev_flow.md` module map (`widgets/pass_list.py`,
  `widgets/pass_graph.py`, `tabs/document.py` and `tabs/code.py` entries),
  `shaderbox/help_content.py` (the "graph view" phrase becomes the graph tab), `00_findings.md`
  (the "Landed in" column), `ai_docs/roadmap.md` (row + banner), this file's status.

## Verification

The falsifier per guarantee, from the record's sketch, made concrete. Pure tests import
`graph_state` / `theme` only; frame-driven ones use the `app` fixture and
`tests/test_graph_view.py::_frames` with `io.add_mouse_*` / `io.add_key_event`, aiming through
`view.port_rects`, `view.wire_mids`, `view.x_rect` and `view.canvas_rect`.

| Guarantee | Test | Kind |
|---|---|---|
| G1: no cusp for any endpoints | `wire_points` over a grid of `(dx, dy)` covering backward, forward, near-zero and long runs plus `dist = 0`, at zooms 0.25 / 1 / 2.5: the offset is non-negative everywhere, and `cp0.x > cp1.x` whenever `dx < 0` | pure |
| G1: continuity across the old bus boundary | sweep `dx` over `[-48, 48]` in 1px steps at fixed `dy`: each control point moves by less than 2px per step (a topology switch is a jump) | pure |
| G4: the threshold and its floor | `wire_hit_threshold(0.25) == 6.0`, `(1.0) == 6.0`, `(2.5) == 7.5`; `wire_hit` on a known cubic: a point on the curve returns `< 0.5`, at `threshold - 1` a hit, at `threshold + 1` `None`; the bounding-box reject returns `None` for a far point | pure |
| G4: 24 segments miss no real hit | 200 points along the true cubic of a 400px S-curve each report a distance under `threshold` | pure |
| G18: an error wire stays red while hovered | `wire_state` over all sixteen `(on_cycle, selected, hovered, dim)` combinations against the precedence table | pure |
| G6: nothing changes size on hover | `inspect.signature(node_size).parameters` is exactly `(port_count, box)`; `wire_hit_threshold` takes `zoom` alone | pure |
| G8/G3: the loop and the bus are gone | `SIZE` has none of `GRAPH_LOOP_RISE`, `GRAPH_LOOP_REACH`, `GRAPH_BUS_STEP`, `GRAPH_BUS_CLEAR`, `GRAPH_MIN_H`; the widget's source contains neither `_draw_self_loop` nor `bus_y` | pure |
| S13: the lock is passed everywhere | every `is_mouse_dragging(` and `get_mouse_drag_delta(` in the widget's source carries `GRAPH_DRAG_LOCK_PX` | pure |
| G11: the ellipsis and the width | frame-driven (a pushed font is needed): `ellipsize("u_distance_field", budget_at_108)` ends in `...` and measures under budget; at the 128 budget the string is unchanged; same for `distance_field` in the bold face | frame-driven |
| S8: the fit frames every wire | build the chain `a -> b -> c` plus a backward read (`a` reading `c`, wired directly through `set_sampler_source` on the session, so the cycle path is exercised too), `_fit` on a 800x600 avail, then every wire's four control points at zoom 1 through `_Xf.to_screen` lie inside `view.canvas_rect` | app fixture, no frames |
| G11: a six-column chain fits | six chained passes, `_fit` with `avail = (1225, 600)`: `view.zoom == 1.0`; with `(740, 600)`: `0.6 < view.zoom < 1.0` | app fixture, no frames |
| G5: select a wire and Delete it | open the graph tab, frames, click `view.wire_mids[("c", "u_src")]`, assert `selected_wire == ("c", "u_src")` and `selection == set()`, send `Key.delete`, assert exactly one `set_sampler_source(..., "c", "u_src", NoSource())` and nothing else | frame-driven |
| G5: the ✕ unwires | select the wire as above, click the centre of `view.x_rect`, assert the same single write | frame-driven |
| G5: Delete is inert while a text field is active | select a wire, open the group prompt (`view.group_prompt = True`), frames, send `Key.delete`, assert no `set_sampler_source` call | frame-driven |
| G6: exclusive hover in order | park the mouse on `port_rects[("c", "u_src")]`'s centre: `hovered_port` set, `hovered_wire is None`, `hovered_node is None`; on the node body away from any port: `hovered_node` set, others `None`; on `wire_mids[("b", "u_src")]`: `hovered_wire` set, others `None`; off the canvas: all `None`. Break to try: swap the port and node rungs -- the first case flips | frame-driven |
| G13: 3px is a click, 5px is a drag | press on a node's body, move 3px, release: `pick_pass` ran, `set_pass_positions` did not; press, move 5px, release: `set_pass_positions` ran, `pick_pass` did not. Break to try: omit `lock_threshold` at the node-body site -- the 5px case then reads imgui's 6px default and stays a click | frame-driven |
| G7: the cursor is requested by the gesture | during a middle-drag pan `app.want_cursor is app.hand_cursor` on the frame before `ui.py` resets it (read through a probe frame that stops before the apply, or by asserting `app.cur_cursor` after the frame); at rest `app.cur_cursor is None` | frame-driven |
| G14: every kept binding still works | the existing `tests/test_graph_view.py` passes with only the `passes_view -> open_graph_for` substitution; a test needing any other edit is a silent binding change | frame-driven |
| G12: the five channels | the widget's source calls `channels_split(5)` once and `channels_merge()` once | pure |
| T1-T6: the tab | `tab_label` reads `"<name> (graph)"`; `open_graph_for` twice yields one tab and it is active; `is_tab_dirty` False; `formatter_for("graph") is None`; `format_current_editor` and `jump_to_next_error` return without error on a graph tab; `close_editor_for_path` removes it; `_on_document_deleted` removes it and keeps a lib tab; `command_callbacks[CommandId.OPEN_GRAPH]` exists; three frames with the graph tab active leave `view.fitted` True and `app.editor_errors == []` | app fixture + frames |
| Theme | `test_group_tints_are_stable_and_collide_with_nothing` adds `GRAPH_HOVER` to its excluded set and asserts it differs from `SELECT`, `STATE_ERROR`, `GRAPH_EDGE` | pure |

**Gates that must be broken before they are believed.** G13's lock, G6's order and G4's floor
are gates, not tests: each is landed by breaking the guarded thing (omitting `lock_threshold`
at one site; reversing the port and node rungs; dropping the `max(6.0, ...)` floor), watching
the named test go red, restoring it. The implementation commit's body says which break was
tried for each. `make gates` green with the smoke run (not skipped) before the commit, judged by
its exit code captured unpiped.

**The maintainer's eyes (no gate can run these):** the halo never reads as a thicker wire at
0.25 and 2.5 zoom; the feedback glyph's shape at 12px; a selected card over a neighbor; the
blue hover against the grey wire; the 128 card with his own names; the 4px lock under his hand.
Each of the four forks is then a token edit.

## Open questions for the user

None. The four forks ship at the record's recommendation by his instruction and are tuned
against the rendered result.

## Review history

(pre-implementation round 1 pending)
