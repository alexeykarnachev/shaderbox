# 093 — Refinement: the graph editor

Status: **waves 1-6 landed, the editor's half of wave 6 re-vendored at `5a56ccf`. Next is his
visual review in the running app, whose findings open the next wave in the ledger.** The waves and what each landed are the `## Waves`
section below, newest first; the findings are `00_findings.md`. The maintainer's verdict on the shipped canvas ("feels
very cheap") sent the walk into research first: `02_research_brief.md` is the brief, `research/`
holds six area reports against primary sources, `03_graph_design.md` is the design record
(G1-G18, what stays, the four forks G-Q1..Q4, the false trails, a verification sketch), and
`01_research_graph_tab.md` records the tab machinery the graph moves into (mock C of
`00_mock_panel.html`, his pick). On the four forks he said: "It is hard to answer without the
actual visuals. Let's implement everything, and then we'll tune." So every fork ships at the
record's recommendation and is tuned by him against the rendered result.

This file is the implementation contract for wave 1. The record carries the rationale and the
sources; where this spec refines a "Code" paragraph of the record, the refinement is stated
under `## Refinements over the record` with its reason, so a reviewer can see the edge. Review
reports live in `reviews/`.

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
  would make `node_size` depend on fonts. Trigger: a real document whose names ellipsize at 136
  in a way he objects to after seeing it.
- **Blender's flatness correction / Rete's vertical term for a long near-horizontal wire** (G1
  alt. d): the references disagree and neither addresses the defect. Trigger: a screenshot of a
  wire reading conspicuously flat.
- **The box width.** `GRAPH_BOX_EXTRA_W` stays 40, so a box is 176 wide around a 116 picture.
  The maintainer's eyes; a token if he objects.

## Design decisions

Numbered `T` for the tab move and `S` for the spec's own calls; the canvas rules themselves are
the record's **G1-G18, adopted as locked constraints by reference** (a reviewer checks the
implementation against the record's Rule paragraphs; this spec does not restate them). The four
forks are closed here as S1.

**S1. The forks ship at the record's recommendation, with the width and the hue corrected by
measurement.** No menu entry for a wire (G-Q2), `GRAPH_DRAG_LOCK_PX = 4.0` (G-Q3). For G-Q1 the
record's 128 rests on a font advance of 0.545898 em; measured in a rig frame on the shipped
faces, the advance is 7.0px at 12px and 8.0px at 14px bold, so `u_distance_field` is 112px
against 128's 110px port-label budget (it ellipsizes) and `distance_field` is 112px against
128's 112px name budget (zero slack). The requirement he stated -- his own two names fit uncut
-- is met by **`GRAPH_NODE_W = 136`** (budgets 118 and 120), C's original recommendation, which
the fit-clamp check survives (1168px fitted width: 1.049 at 1225px, 0.634 at 740px). The same
6.5508px-per-character estimate once shipped a truncating check in
`tests/test_pass_settings_layout.py`; the ellipsis row below measures in a rig frame for that
reason. For G-Q4 the
record's `blue_b` is refused by the code: `blue_b` IS the `blue` accent preset's primary
(`theme._ACCENTS["blue"]`), so under that accent a hovered wire would read as the in-flight wire
and a hovered node's halo as the output border -- the collision the import-time invariant block
exists to prevent, and the record's G6 sentence "is not an accent primary" is wrong. Every
remaining chromatic palette hue is an accent primary, a state hue, a group tint or `SELECT`, so
the hover takes the record's own fallback, a neutral: `COLOR.GRAPH_HOVER = _P["fg_0"]` (the
`FG_TITLE` brightness, a clear step above the `gray` wire and the `fg_4` dots at the halo
alphas), with the invariant `GRAPH_HOVER not in _accent_primaries` beside the three `!=` ones
(S12). Each fork is one token or one absent item, tuned by the maintainer after he sees it; a
change he asks for is a token edit, not a re-lock.

**T1. The graph is an editor-tab kind.** `EditorTabKind` gains `"graph"`. A document's graph tab
is `EditorTab(path=app.paths.graph_json_for(document_id), kind="graph", document_id=document_id)`:
the path is a real per-document file, so every path-keyed pass-through (`_focus_or_add_tab`'s
dedupe, `_tab_id_suffix`, `close_editor_for_path`, `_close_tab_for_path`) works unchanged, and
the pass-rename hook, which re-keys by pass path, never matches it. A graph tab has no
`EditorSession`: `is_tab_dirty` finds none and answers False, `is_current_editor_dirty` likewise,
`formatter_for("graph")` is `None` (no table entry), `jump_to_next_error` and
`format_current_editor` return on the missing session, `_drain_editor_input` returns on it, and
the flush paths skip it. Two edits are REQUIRED for the claim to hold, not implied: `tab_label`
gains a `"graph"` branch returning `f"{document_name} (graph)"` before the multi-pass fallthrough
(which would call `pass_name_of` on `graph.json`); and
`widgets/uniform.py::_locate_uniform_declaration` switches from the creating
`app.get_current_session()` to `get_current_session_if_exists()` (it already handles `None` on
both branches) -- it is the one creating call site, reachable from the Uniforms panel while a
graph tab is active, and it would otherwise open a GLSL editor session over `graph.json` and make
the graph tab dirty-capable. `_on_document_deleted` filters by `document_id` and so drops the tab;
`forget_render_state` already drops `graph_views`. This reverses 092 D2; 092's spec records the
reversal in its Review history and D2 gets a one-line pointer at the overruled passage, the way
083 D5 carries 085's.

**T2. The dispatch is one branch in `tabs/code.py::draw`, right after the tab row and before the
`ui_document is None` guard** (a graph tab of a non-current document must still draw). When the
active tab's kind is `"graph"`, `_draw_graph_tab(app, tab)` runs and `draw` returns before the
session fetch. The branch does the pane's focus bookkeeping the text body does, and nothing
else of it: consumes `editor_focus_requested` when no popup is open (ui.py has already issued
the `set_next_window_focus`), sets `app.editor_focused` from
`imgui.is_window_focused(FocusedFlags_.child_windows)`, consumes `editor_defocus_requested` as
the text body does, and sets `app.editor_errors = []` so a stale error list from a previous tab
does not drive `F8`. Then `pass_graph.draw(app, tab.document_id)`. Setting `editor_focused` is
what makes `Ctrl+W` close the graph tab and `Ctrl+Tab` cycle onto it; it also makes every
`CommandScope.EDITOR` spec dispatchable on the tab, which today is those two plus
`FORMAT_BUFFER`, whose handler returns on the missing session. `draw_chrome` needs no branch: its
non-shader path already prints `tab_label`, and the dirty read is False.

**T3. The canvas fills its host.** `pass_graph.draw(app, document_id)` loses its `height`
parameter: the scope row draws, then `begin_child("##pass_graph", size=(0, 0), ...)` takes the
rest of the host's content region. `SIZE.GRAPH_MIN_H` stops being read and is deleted. The
widget still positions no sibling and measures none (092 D2's leaf rule, kept).

**T4. Opening it.** `App.open_graph_for(document_id, focus_editor=False)`, the sibling of
`open_script_for`: `_focus_or_add_tab(EditorTab(...graph...), focus_editor=focus_editor)`,
frozen mid-copilot-turn like its sibling. `CommandId.OPEN_GRAPH`, label `"Open graph"`, chord
`Alt+G` (unbound today; no vim or standard-keymap chord uses Alt), category `C.EDITOR` with the
default `GLOBAL` scope exactly as `OPEN_SHADER` / `OPEN_SCRIPT` are declared, handler
`lambda: self.open_graph_for(self.current_document_id, focus_editor=True)`. The registry
coverage test then requires the label in the help shortcuts, which the generated section
provides.

**T5. The Document tab keeps the strip and gets an `open` for the graph, like the Script row.**
`tabs/document.py::_draw_passes` draws `_entry_row_label(graph_active, "Passes")`, where
`graph_active = _entry_tab_active(app, document_id, "graph")` -- a new free predicate over
`app.active_tab` (kind and document match) that the Script row's inline `script_active` also
moves onto, so the tick rule has one home and a test can call it -- a VISUAL
change from today's `small_caption(app.font_12, "Passes")`: the label moves to the ambient font
with frame-padding alignment, and the accent tick marks a graph tab of THIS document as the
editor's active tab, exactly as the Script row's tick does -- then
`standard_button("open##entry_graph")` (tooltip `"Open the pass graph"`) calling
`app.open_graph_for(document_id, focus_editor=True)`, then the strip, then the
`add pass` / `import...` row, each bracket in `begin_disabled(app.copilot_turn_active)` as
today. The `strip | graph` `segmented_choice` goes; `PassesView` and `PASSES_VIEW_LABELS` are
deleted from `ui_regions.py`; `UIAppState.passes_view` is deleted from `ui_models.py`
(`load_model`'s per-key salvage drops the retired key from any saved `app_state.json`; the
sandbox's `projects/dev/app_state.json` carries no such key, verified). `tests/test_ui_regions.py`
is deleted: its three tests cover only the retired enum (its docstring's `ChannelView` claim has
no test behind it; `tests/test_channel_view.py` covers that enum). `scripts/smoke.py`'s graph
frames open the tab through `open_graph_for` instead of the preference (T6).

**T6. The smoke exercises the tab.** Frame 43 calls `app.open_graph_for(multi)` where it set
`passes_view = GRAPH`; frames 45-47 keep their asserts (fitted, scope survival, scope
revalidation, draw-writes-nothing, Arrange saves once); frame 47 ends with
`app.close_editor_for_path(app.paths.graph_json_for(multi))` in place of the `passes_view =
STRIP` reset and asserts no tab of kind `"graph"` remains, exercising the session-less
teardown. The tail's `get_current_session_if_exists() is not None` assertion is about the tab
ACTIVE at the end, and it holds because frame 48's `set_current_document_id(canary_id)` runs
`_on_current_document_changed` -> `ensure_shader_tab`, which focuses a shader tab; a later edit
to frames 47 or 48 must keep that true.

**S2. Hover and selection of a wire share one identity: the `(owner, sampler)` the wire
terminates at.** That pair identifies a wire uniquely in the whole document -- a sampler has one
source (072) -- and it is the identity `App.unwire` takes, so `GraphViewState.hovered_wire` and
`selected_wire` are both `tuple[str, str] | None` and compare directly. `_Edge` gains `owner`
and `sampler` (the consumer pass and its sampler; for a box edge the member behind the box slot,
for a ghost-reader edge the reader itself) and loses `span`, which nothing reads once the bus is
gone. `selected_wire` is revalidated every canvas frame through a pure
`graph_state.revalidated_wire(selected, edges) -> tuple[str, str] | None` (the shape of
`revalidated_scope`) and so clears when no drawn wire carries it (an unwire, a pass delete, a
scope change).

**S3. Hover is read one frame late for the DRAW, same-frame for the selection, and written
fresh each frame.** The picture is drawn before the hit rects (the ports' positions come from
the same geometry), so `_draw_canvas` reads last frame's `view.hovered_node: str | None` (a
node key), `hovered_port: tuple[str, int] | None` and `hovered_out: tuple[str, int] | None`
(node key, slot) and `hovered_wire` at draw time. The wire distance pass (G4) runs AFTER the
node/port/output button loop and BEFORE the background press is acted on (S4), so this frame's
hover is fully known at the point a selection is decided; the pass writes the four fields at the
end of that step, resolved exclusively in G6's order: a port or output dot (`is_item_hovered()`
on the `gport_*` / `gout_*` buttons; during a wire drag the drop target counts as the hovered
port), else a node body, else the nearest wire under G4's threshold, else nothing. Every field
is written every frame, `None` included, so a mouse that left the canvas leaves no stale cue.

**S4. Selection happens on press, wire and node alike; a click is resolved on release against
the drag lock.** Today a node's `_click` fires on `is_item_clicked` (press) and a drag may
follow, so a drag also picks (measured: an 8px drag today runs both `pick_pass` and
`set_pass_positions`). G13's falsifier requires the opposite. So a node's click fires on the
release frame -- `is_item_deactivated() and is_item_hovered() and is_mouse_released(left)`, with
`view.node_drag is None and view.wire_drag is None` (the drag blocks run after the node loop, so
on the release frame a drag that happened is still set; `is_mouse_dragging` itself is already
False on that frame, which is why the gate rests on the drag object) and `not blocked` -- and
the double-click stays on `is_mouse_double_clicked` as today. `is_item_hovered()` is there
because `is_item_deactivated()` is also True when the release lands off the item. A move and a
release arriving in ONE frame read as a click (imgui resets the drag before the frame runs);
today's press-time click does the same, and the frame-driven tests keep the move and the release
in separate frames. The background press is recorded as `bg_pressed = is_item_clicked(left)` on
the background button and acted on AFTER the node loop and the wire pass, gated on `not
blocked`: with a wire under the threshold and no node or port hovered, `selected_wire` becomes
that wire and `view.selection` clears unless Shift is held; with no wire, both clear unless Shift
is held. Selecting a node clears `selected_wire`; a rubber band's release clears it too. The two
selections are exclusive so one Delete has one target.

**S5. Delete and Backspace are read locally, once, AFTER the hit-test section and the drag
blocks, before `_canvas_menu`.** The position matters: on a release frame `is_any_item_active()`
is True at the top of `_draw_canvas` and False at the bottom, and `is_window_hovered` the
reverse (measured), so a read at the top is dead on exactly the frame a user who just clicked
the wire presses the key. The gate: `imgui.is_key_pressed(imgui.Key.delete) or
imgui.is_key_pressed(imgui.Key.backspace)`, the canvas child hovered (`hovered`, the existing
`is_window_hovered(child_windows)`), `not imgui.is_any_item_active()`, `not blocked`, and
`selected_wire` set; the write is `app.unwire(document_id, *view.selected_wire)`, its refusal
toasted as `_drop`'s is. With a node selection and no wire, the key does nothing (G17). Two
facts about the gate's clauses, so no one narrows it: the group-name prompt is a plain
`begin_popup`, for which `app.any_popup_open()` is False and `is_window_hovered(child_windows)`
is already False, so a Delete typed into it is refused by `hovered` before `is_any_item_active`
is consulted; and while a press is HELD on the canvas `hovered` is likewise already False
(measured: dropping `is_any_item_active` alone still refuses the held case). The one reachable
state where `not is_any_item_active()` is the clause doing the work is a text input ACTIVE in
another window while the mouse rests over the canvas -- the Document tab's name field, the
chat's input. Neither is reachable headlessly (the `app` fixture has no OpenRouter key, so the
chat draws its connect gate and no input; measured), so the gate is a pure predicate,
`graph_state.delete_allowed(pressed, hovered, any_item_active, blocked, has_wire) -> bool`,
tested over its whole domain, and `_draw_canvas` feeds it the five live reads in one line; the
live scenario (type Delete into the name field with the mouse over the canvas: nothing happens)
is the maintainer's check. Both key names are present on this build's `imgui.Key` (verified).

**S6. The ✕ lives on the top channel and is hit-tested by hand on the press.** Submission order
cannot make it win: an item without `allow_overlap` submitted earlier beats a later overlapper
(measured), the ports declare nothing so the drop target keeps working, and a short wire's
midpoint lands inside its consumer port's 7px box. So the ✕ is not an item. On
`imgui.is_mouse_clicked(left)` with `hovered` and `not blocked`, if the mouse lies inside
`view.x_rect`, the canvas calls `app.unwire(document_id, *view.selected_wire)` and sets
`view.press_blocked = True`, the existing latch that keeps this press from becoming any other
gesture (a node click, a port grab, a band) until the button comes up. Two orderings make that
true, and both are required. First, the check runs at the TOP of `_draw_canvas`, inside the
press bookkeeping and BEFORE `blocked = frozen or view.press_blocked` is computed (it reads last
frame's `x_rect` and `hovered`, which is computed just above the picture), so the same frame's
`blocked` already carries the latch and the background button's `is_item_clicked` on that frame
is ignored by S4's `not blocked` gate. Second, the latch's clear (`if not mouse_down:
view.press_blocked = False`, today at the top of the frame) moves to the END of `_draw_canvas`,
so on the RELEASE frame `blocked` is still True and the release-time node click of S4 is
refused; the next frame is clean. Without the second ordering a ✕ over a card (the normal case
under G15: a wire runs under a node) would both unwire the sampler and, on release, choose the
node as the output (measured). `x_rect` is written every frame the selected wire is drawn (`(centre - h, centre - h,
centre + h, centre + h)` with `h = max(GRAPH_WIRE_X_R * z, float(GRAPH_HIT_MIN))`) and `None`
otherwise. The channels: 0 wire halos, 1 wires (crisp strokes), 2 nodes, 3 the in-flight wire,
4 the overlays (the ✕ drawn per G5 -- a `BG_APP` disc, a 1px `SELECT` ring, two `add_line`s --
the rubber band, the snap guides). `channels_split(5)` once at the top of `_draw_canvas`,
`channels_merge()` once at its end after the guides; the split stays open across the hit-test
section, which submits only invisible buttons and popups (a popup is its own window with its own
draw list; measured to leave the canvas splitter untouched).

**S7. Bring-to-front orders the buttons too.** The nodes are drawn in ascending
`(is_being_dragged, is_selected)` order (G12) -- a card in flight paints above a merely selected
one, which is what a drag needs -- and the hit-test loop submits the node buttons in
the SAME order, so the card that paints on top is the one whose button, submitted later, wins
the overlap. One list, sorted once per frame, used by both loops, and exposed as
`view.node_order: list[str]` (node keys, this frame) so a test can assert the selected node is
last.

**S8. The fit frames the wires as well as the nodes, by the curve, not its control polygon.**
A backward wire's S-curve bulges past the producer's right edge and the consumer's left edge
(21.3px at the decided tokens against a 16px `_FIT_MARGIN`), so G3's "no wire leaves the nodes'
bounding box" is false. Framing the four control points would over-frame it (on the spec's own
three-column chain the hull inflates the fitted width from 605 to 957px and drops the fit zoom
to 0.84 where the curve needs none), so `_fit` takes the nodes AND the edges and frames the union
of the nodes' bounding box and, per wire, the 25 points `bezier_point(..., i / 24)` over its
canvas-space cubic -- `wire_points(a, b, 1.0)` on the canvas-space endpoints, exact because the
offset and the distance both scale linearly with zoom, so the canvas-space curve is the
screen-space one divided by the zoom. This is what closes finding 5 by construction. `_fit`
clamps the zoom at 1.0 (`min(1.0, ...)`), so G11's fit-zoom table describes the unclamped ratio.

**S9. The pure geometry lives in `graph_state.py`, importable without imgui.** New free
functions beside `NodeDrag` and `node_size`:
- `wire_points(a, b, zoom) -> tuple[Point, Point, Point, Point]`: G1's formula over two screen
  points, returning `(p0, cp0, cp1, p3)`. No branch on the sign of `dx`.
- `bezier_point(p0, cp0, cp1, p3, t) -> Point`.
- `wire_hit_threshold(zoom) -> float`: `max(float(SIZE.GRAPH_WIRE_HIT_FLOOR),
  SIZE.GRAPH_WIRE_W * 2.0 * zoom)` (G4, screen px).
- `wire_hit(mouse, points, threshold, segs) -> float | None`: the bounding-box reject expanded by
  the threshold, then the minimum point-to-segment distance over `segs` segments, or `None` when
  over the threshold.
- `class WireState(StrEnum)`: `ERROR`, `SELECTED`, `HOVERED`, `DIM`, `NORMAL`, and
  `wire_state(on_cycle, selected, hovered, dim) -> WireState` in G18's precedence (error first).
  The widget maps a state to its stroke color and halo; the precedence is the pure part.
- `revalidated_wire(selected, edges)` (S2), `delete_allowed(...)` (S5).
`_draw_wire(dl, points, zoom, col, halo_col)` in the widget draws the halo (`GRAPH_WIRE_W * 3.0 * z`
at the state's halo alpha, channel 0) when `halo_col` is not `None`, then the crisp stroke
(channel 1) -- a hovered stroke is `GRAPH_WIRE_W * 1.4 * z` per G6, a selected or normal one
`GRAPH_WIRE_W * z`, all floored at 1.0. The caller resolves the state; `_draw_wire` never reads
it (G18's rule kept; G18's seven-parameter signature superseded, see the Refinements table). The
in-flight wire uses the same `wire_points`, on channel 3, in `ACCENT_PRIMARY`.

**S10. The badge row hosts the feedback glyph.** `_draw_badge` returns the pill's width so
`_draw_feedback_glyph` can sit `GRAPH_FB_GAP * z` left of an `xN` badge when one is drawn that
frame and flush at the picture's top-right otherwise (G8). The glyph is two open arcs through
`path_arc_to` + `path_stroke(col, thickness)` -- `path_stroke`'s flags default to `0`, so no
`ImDrawFlags_` constant is needed. Ghost nodes draw no badges today and draw no glyph.

**S11. The ellipsis helper goes public.** `ui_primitives._ellipsize` is renamed `ellipsize`,
and every reader follows: its four call sites inside `ui_primitives`, `popups/lib_picker/tree.py`,
`tests/test_pass_settings_layout.py`, `tests/test_anchored_note.py`. The name budget and the
port-label budget are G11's, measured inside the same pushed-font scope as the `calc_text_size`
that positions the text.

**S12. Theme tokens.** `SIZE`: `GRAPH_NODE_W 108 -> 136` (S1), `GRAPH_THUMB 80 -> 96`,
`GRAPH_PAD 6 -> 8`, `GRAPH_NAME_H 18 -> 20`, `GRAPH_PORT_ROW 16 -> 18`; new `GRAPH_WIRE_BOW:
float = 0.40`, `GRAPH_WIRE_MIN_OFF: int = 24`, `GRAPH_WIRE_HIT_FLOOR: int = 6` (named so it
cannot be misread for the port floor `GRAPH_HIT_MIN = 7`; a comment names the pair and the
6-under-7 reason), `GRAPH_WIRE_HIT_SEGS: int = 24`, `GRAPH_WIRE_X_R: int = 7`, `GRAPH_FB_SIZE:
int = 12`, `GRAPH_FB_GAP: int = 2`, `GRAPH_DRAG_LOCK_PX: float = 4.0`; deleted `GRAPH_BUS_STEP`,
`GRAPH_BUS_CLEAR`, `GRAPH_LOOP_RISE`, `GRAPH_LOOP_REACH`, `GRAPH_MIN_H`. `COLOR`: new
`GRAPH_HOVER = _P["fg_0"]` (S1), `GRAPH_HOVER_HALO_ALPHA: float = 0.35`,
`GRAPH_SELECT_HALO_ALPHA: float = 0.55`; `GRAPH_HOVER` joins `_GROUP_TINT_EXCLUSIONS` and the
import-time invariants gain `GRAPH_HOVER not in _accent_primaries` (the one that bites -- it is
red for `blue_b`) and `GRAPH_HOVER != SELECT`, `!= STATE_ERROR`, `!= GRAPH_EDGE` (three cues that
meet on one wire). The module-local `_MIN_DIRECT_DX` (seven read sites, one of them the
in-flight wire) and `_BEZIER_BOW` in the widget are deleted in favor of the tokens. The comment
block over the `SIZE.GRAPH_*` tokens is rewritten for the tokens as they are (no bus, no loop).

**S13. The drag lock is passed at every site.** `imgui.is_mouse_dragging(MouseButton_.left,
SIZE.GRAPH_DRAG_LOCK_PX)` at the four sites (the rubber band, the node body, the port press that
moves the node, the output dot), and `imgui.get_mouse_drag_delta(MouseButton_.left,
SIZE.GRAPH_DRAG_LOCK_PX)` where the band anchor is derived, so the anchor and the lock agree.
No other `is_mouse_dragging` call in the module may omit the threshold (the test greps for it).

**S14. Cursors.** `App.__init__` creates `hand_cursor` (`glfw.HAND_CURSOR`) and
`crosshair_cursor` (`glfw.CROSSHAIR_CURSOR`) beside the three existing ones; `_draw_canvas`
requests `app.want_cursor = app.hand_cursor` while `panning` or `view.node_drag is not None`, and
`app.crosshair_cursor` while `view.wire_drag is not None` (G7). Requests only -- `glfw.set_cursor`
never appears in the widget; `ui.py` applies once per frame on change and resets `want_cursor`
to `None`, so a test reads `app.cur_cursor` after the frame.

**S15. A single click on a node chooses the output; only a double-click opens its shader tab.**
(AMENDED by W3-3: no click of any count opens the tab; the context menu's `Open shader` does.)
Today `_click` calls `pick_pass`, which is `ensure_shader_tab` + `set_output_pass`, and
`ensure_shader_tab` ACTIVATES the shader tab (measured: one 3px click appends a tab and moves
`active_tab_index`). Inside the editor pane that evicts the graph tab on every click and the
canvas stops drawing, so the strip's rule cannot be the canvas's. `App.pick_pass` splits into
`App.choose_output(document_id, name)` (the `set_output_pass` half with its toast) and
`pick_pass` = `ensure_shader_tab` + `choose_output`, unchanged for the strip, the uniforms row
and the copilot (which calls neither). `choose_output` also clears the Uniforms tab's explicit
pick (`set_panel_pass(document_id, "")`), which `ensure_shader_tab` did for today's click and
083 states as the rule ("a pick retires an older explicit one"); without it a persisted pin
would keep the panel on another pass after a canvas click (measured). The canvas's `_click`
calls `choose_output` (a ghost click stays as it is: back to the root, the ghost selected);
`_double_click` keeps `pick_pass(..., focus_editor=True)`, which is the deliberate "open this
pass" gesture and switches the pane to the shader tab. The Uniforms panel follows: with the pin
cleared, `panel_pass` falls to the output when no shader tab of the document is active, so the
clicked node's uniforms are the ones shown. This reverses the click half of
092 D10 ("Click a node: `pick_pass(...)`"); 092's spec records it beside D2's reversal.

## Refinements over the record

Where the record's "Code" paragraphs and this spec differ, this spec wins, for these reasons:

| Record | Spec | Why |
|---|---|---|
| G6/G-Q4: `GRAPH_HOVER = blue_b`, "not an accent primary" | S1: `fg_0`, plus the accent-primary invariant | `blue_b` is the blue accent's primary; the record's premise is false (measured) |
| G4: `hovered_wire` is the edge's `(src_key, src_slot, dst_key, dst_slot)` | S2: `(owner, sampler)`, the same identity as `selected_wire` and `App.unwire` | One identity for hover, selection and the verb; comparable without a lookup |
| G5: a wire is selected by "press and release with no drag past the lock" | S4: on press, as a node is; the node CLICK moves to release | The drag lock is what a click (an output choice) is resolved against, and that is where G13's falsifier bites |
| G5/G12: the ✕ on channel 1, an item submitted last that "wins the overlap" | S6: on the top channel, hand hit-tested on the press, latching `press_blocked` | An earlier item without `allow_overlap` beats a later one (measured); the ports must keep declaring nothing for the drop target |
| G3: "no wire leaves the nodes' bounding box" | S8: the fit frames nodes and the sampled curves | A backward S-curve bulges 21px past both cards (measured); the control polygon over-frames by 58%, the sampled curve does not |
| G11's fit-zoom table | S8's note: `_fit` clamps at 1.0 | The table is the unclamped ratio |
| G11/G-Q1: 128, from a 0.545898 em advance | S1: 136 | The advance is 7.0 / 8.0px (measured in a rig frame); at 128 `u_distance_field` ellipsizes and `distance_field` has zero slack |
| G4/G18: helpers as module-level functions in the widget | S9: in `graph_state.py` | The pure tests import no imgui and need no `app` fixture |
| G18: `_draw_wire(dl, xf, a, b, col, halo_col, halo_alpha)` (and G1's `..., hovered, selected`) | S9: `_draw_wire(dl, points, zoom, col, halo_col)` | It takes the points `wire_points` returns; the rule that the caller resolves the state is kept |
| G8: `path_stroke(col, ImDrawFlags_.none, thickness)` | S10: `path_stroke(col, thickness)` | The binding's signature is `(col, thickness=1.0, flags=0)` (verified) |
| G14: "Select -- left click, kept unchanged" | S15: a click chooses the output without opening the shader tab | Inside the pane the old click evicts the graph tab (measured) |
| G4: `GRAPH_WIRE_HIT_MIN` | S12: `GRAPH_WIRE_HIT_FLOOR` | Two floors one pixel apart under near-identical names invite a transposition nothing would catch |

## Waves

**Wave 6: his walk of waves 3-5, findings 23-30 (2026-09-15).** Seven host fixes in one
commit; finding 27 (the vim view-scroll commands) is the editor library's, sent to the editor
session with the measurements the ledger row carries and re-vendored from its `5a56ccf` in a
second commit (no export delta; `ed_scroll_max` now means the last line at the top row, and
the editor's `ed_set_text` resets the offset to 0). Two settled decisions reversed on his call, each
with its pointer: 090 D8 (the script ticked at the UI rate; a tick is now a render, with the
document's own `dt` and `frame` and the cursor's previous position anchored per tick -- the
gaps in his drawing script's stroke were the frames between renders) and 065's single-pass
`(shader)` tab label (every shader tab names its pass). The confirm lost its consequence line
(`ConfirmRequest.line` deleted, seven builders), the document tile its `Open` item, the Python
completion its in-string file paths and the `type` protocol's `mro`, and the shape vocabulary
gained the 4:3 `STANDARD_960/1440/1920` tiers on the shared longest-edge ladder. Modal sizing
became one declaration, `Modal.sizing: ModalSizing` (RESIZABLE / FIXED / AUTO) resolved in
`modal_window`, with `Modal.before` and `fixed_size` deleted: the confirm's persisted 132px
was the scrollbar he saw. Gates broken and restored: the chrome clause (the confirm declared
RESIZABLE fails it), the footer test's AUTO domain (a row leaving AUTO fails the list check)
and its overflow read (the AUTO branch degraded to first-use seeding overflows 32px and 236px),
the tick test (ticking `tick_documents` instead of the rendered set ticks twelve times in
twelve frames). Docs in the same commit: `conventions.md` (the confirm, popups, tab-label,
shape and throttle bullets), 090's D8 pointer, `06_command_system.md`'s non-command list,
`dev_flow.md`'s module map, the repo's `/imgui-ui` §7.2, the script API glosses.

**Wave 5: a stacked confirm, the modal footer, and his four items (2026-09-14).** Two halves,
one commit. (A) Wave 4's three post-implementation reports, every finding accepted. A
`request_confirm` fired from inside an open modal used to REPLACE it, skipping its `on_close`
and re-capturing the chat focus while a modal was up -- the lib tree's two deletes are the one
such caller, and confirming one left the picker gone with its armed rename leaked.
`App._open_modal(id, stacked=False)` is the general fix: a stacked open records
`App.modal_below` and leaves the focus capture alone, and `close_modal` restores that modal
instead of writing `None`, so a modal below keeps its state and its `on_close` runs only when
IT closes. The mutex-write gate now walks the whole package tree minus its two owners
(`tabs/document.py` held a destructive control and was outside the old tuple); the chrome
gate derives each row's leaf bodies from `BY_ID` and pins every dispatcher override to the
module that declares its row. (B) Findings 19-22. `ui_primitives` gains
`modal_footer_height` / `modal_content` / `modal_footer`, one number every modal's
reservation reads, and every body adopts them -- the Help modal's scrollbar was its content
and its footer measuring the room apart. The graph canvas loses its border (the pane's frame
is the frame), `Fit` becomes `Frame all`, `Insert at caret` leaves the reading surface, the
Help modal becomes `Documentation`, and the `Right-click for actions` caption goes from both
the documents grid and the lib tree -- REVERSING M6/M12 and the conventions bullet that put
it there.

**Wave 4: one modal mechanism, and the confirm modal (2026-09-14).**
`07_modal_registry_spec.md`'s R1-R7, one commit, after one pre-implementation review
(`reviews/modal_registry_pre.md`, nine findings, all folded). The roster that had been
hand-maintained in five places became one registry: `ModalId` on `App` (`app.modal`, `None`
is closed), one `Modal(...)` constant per popup module, and `popups/registry.py` deriving
`draw_modal` (ui.py's one popup call), `close_modal` (the Esc and Close funnel, with each
row's `on_close` cleanup and `owns_esc` decline) and every gate from `MODALS`. A Close the
user clicked is a FORCED close, which is what the lib picker's dead Close button under an
armed rename had been missing. `popups/confirm.py` landed on it as the first new client, and
every destructive verb now routes through an `App` method that builds a
`ui_models.ConfirmRequest` naming its target and its consequence -- the pass and document
tiles, the bar's `Delete document` / `Reset document`, the Document tab's Reset button, the
lib tree's two deletes, the chat's Clear and its revert glyph. That REVERSED M5's confirm
submenu on the maintainer's call and deleted `confirm_menu_item`, `CommandSpec.confirm_label`
and the copilot's own revert modal with it; the palette offers every `in_palette` spec again.
`app.py` imports nothing from `shaderbox.popups` and a gate walks its imports to keep it that
way -- the cycle a payload type in the popups layer would close has only banned escapes. Docs
in the same commit: `conventions.md` (the popups bullet rewritten around the registry, the M5
bullet reversed), `dev_flow.md`'s module map, the repo's `/imgui-ui` §7.1-7.4,
`06_command_system.md`'s rule 5 and its map, and 05's M5 / M8 / M9 / M12 pointers.

**The menus wave: finding 17, the whole spec (2026-09-14).** `05_menus_spec.md`'s M1-M14, one
commit. The menu bar became a render of `COMMAND_SPECS` (`shaderbox/menus.py`, new: the
`App`-facing `draw_menu_bar` / `command_menu_item` / `menu_enabled`; `commands.command_label`
is the pure one-spelling accessor every button that opens a command's surface now takes). Each
object kind got ONE item-set function its every surface draws -- the pass set already shared,
the group box's and the document grid's are new -- and the grid tile lost its armed corner ✕,
as the pass tile did in W3-2. A destructive menu verb confirms through
`ui_primitives.confirm_menu_item`'s submenu, which retired the lib tree's armed flip and its
two hand-rolled red pushes (the submenu itself is reversed by wave 4, above; the armed flip
stays retired). `InlineInput` moved into `ui_primitives.py` beside the one
`name_input_row` every name-entry row now draws through (the trigger the conventions bullet
named, met a third time). `App.close_popup` became the single Esc/Close funnel, so
`hotkeys._handle_escape` lost its four carve-outs and the `was_settings_open` latch (wave 4
moved that funnel into the registry). Modal
chrome is pinned by `tests/test_modal_chrome.py`, the bar and the item sets by
`tests/test_menus.py`; the copilot limits' copy moved into one `COPILOT_LIMIT_ROWS` table
(`copilot/config.py`) whose short `hint` Settings shows and whose long `explanation` the Help
panel's new copilot section prints. Docs in the same commit: `conventions.md` (four new
design decisions), `dev_flow.md`'s module map, the repo's `/imgui-ui` §7.1 and §7.4, and 092's
D14 / D16 pointers.

Post-implementation: three opus reviewers (correctness, architecture, spec fidelity;
`reviews/menus_post_*.md`), fourteen findings, one fix commit (the group prompt's runaway
width, a bar `Delete document` without its submenu, four verb wirings no test gated, `Group`'s
place, the rest small), then a closure round that re-ran every mutation: PASS. Then the
maintainer's verdict on the bar's grouping -- it had rendered the cheatsheet's, which was
never designed -- and `06_command_system.md`: the command table designed as one system, six
categories by object, groups most-used first, labels naming their object, every command in
the bar, chords kept; one opus review of it (`reviews/menus_command_system.md`) found the
tests pinning the skeleton and not the map (fixed: the test parses the doc's map), the
reset/delete confirms inverted against recoverability (`Reset document` confirms; a
confirming verb leaves the palette), and the chat's one-click `Clear` with no home (now the
`Clear chat` command).

**Wave 3: his second hands-on batch, findings 12-17 (2026-09-14).** Five fixes in one commit;
finding 17 (every menu, popup and modal, reviewed and redesigned) is delegated to
`04_menus_inventory.md` -> `05_menus_spec.md` in this directory and runs the feature flow.

- **W3-1 (finding 12). The picture's top inset equals its side inset.** `SIZE.GRAPH_THUMB_INSET
  = 10` replaces `GRAPH_PAD` above the picture in `_thumb_rect`, `_port_point` and `node_size`;
  `GRAPH_PAD` stays the name row's gap. A theme invariant asserts `GRAPH_NODE_W - GRAPH_THUMB ==
  2 * GRAPH_THUMB_INSET` (break tried: `9` fails the import with the invariant's message).
- **W3-2 (finding 15). The tile is a picture, a name and its chips.** The gear overlay
  (`tune_icon_button`, deleted with no caller left) and the armed corner ✕ (`App.pass_delete_armed`,
  its rename follow-through and its smoke frames, deleted) go; `preview_cell` gets
  `armed=False` and no overlay. Both verbs stay on the context menu, which already had them.
- **W3-3 (finding 13). No click opens a shader tab.** The tile click and the uniforms row's
  source preview call `App.choose_output`; the canvas's double-click on a pass does nothing
  beyond the click (a box's double-click still enters the group). **Amends S15**: the
  double-click was the "open this pass" gesture; the context menu is now. `pick_pass` keeps
  its two callers that create or step (`add pass`, `Alt+Left`/`Alt+Right`), where the editor
  following is the point.
- **W3-4 (finding 14). `Open shader` heads `pass_menu_items`**, calling `ensure_shader_tab(
  document_id, name, focus_editor=True)`, so the tile and the node get it together.
- **W3-5 (finding 16). A self-read grows no port.** `node_ports` skips a sampler whose source is
  the pass itself; `PortKind` loses `"prev"`, `_draw_port_dot` its double ring, `group_boundary`
  its skip, `_pass_node` its `prev` label. **Reverses 092 D1's "moved last" clause and D11's
  double ring**; the feedback read itself is untouched (069 D9 still writes `u_prev` down as
  reading yourself, and the strip's `prev` chip still says so). Revisit when the uniforms
  extension he mentioned shows every uniform of a node.
- **W3-6 (finding 17).** Delegated: `04_menus_inventory.md` (§1-9 the inventory, three opus
  readers over two rounds to convergence; §10 the design pass, two opus reviews and a PASS
  closure round, `reviews/menus_*.md`) -> `05_menus_spec.md` (M1-M14), implemented and reviewed
  as its own wave above.

Tests: `test_node_ports_classify_every_state_and_skip_the_self_read` (the self-read absent);
`test_a_double_click_on_a_pass_leaves_the_pane_on_the_graph` (the tab kind stays `graph`, the
output is chosen); `test_a_node_grows_one_row_per_port_and_a_box_is_wider` follows the token;
`test_an_armed_delete_follows_a_rename` deleted with the latch. Docs in the same commit: 092's
D1, D10 and D11 pointers; S15's verification row; `conventions.md`'s graph bullet (the click
sentence and the ports sentence); `dev_flow.md`'s `pass_list.py` and `pass_graph.py` entries;
the help text on ports; the ledger; the roadmap banner.

**Wave 2: his first hands-on batch, findings 6-11 (2026-09-14).** Six decisions, one wave,
one commit, two post-implementation reviewers; two of them reverse wave-1 rules and say so.

- **W2-1 (finding 6). The graph's `open` sits on its own row directly under the Script row.**
  `_draw_entry_points` draws, under the Script row at the row spacing that row uses,
  `_entry_row_label(graph_active, "Graph")` + `standard_button("open##entry_graph")` (tooltip
  `"Open the pass graph"`, same handler). Not on the Script row itself: at the narrow panel
  (`_APP_PANEL_MIN_W` minus the grid) the settings child offers about 194px and the combined row
  needs about 276, and `same_line` clips rather than wraps (measured). The Passes row goes back to the plain
  `small_caption(app.font_12, "Passes")` over the strip with no tick and no button; T5's
  entry-row shape moves with the button. `_entry_tab_active` stays the one predicate.
- **W2-2 (finding 7). Open tabs survive a restart.** `UIAppState` gains `editor_tabs:
  list[TabRecord]` and `active_tab_path: str = ""`, where `TabRecord` (in `editor_types.py`,
  a pydantic model beside `EditorTab`: `path: str`, `kind: EditorTabKind`, `document_id: str`)
  is the persisted shape of one tab. `App.save` mirrors the live list through a pure
  `tab_records(tabs) -> list[TabRecord]`; `App._init`, after the documents are loaded and
  before today's `ensure_shader_tab` fallback, restores through a pure
  `tabs_from_records(records, document_ids, exists) -> list[EditorTab]` that drops a record
  whose path no longer exists or whose `document_id` names no loaded document (a lib tab has
  `""` and passes), then sets `active_tab_index` to the restored tab whose path equals the
  persisted `active_tab_path: str` (else 0 -- a positional index would land on the wrong file
  once an earlier record is dropped) and `tab_select_pending`;
  the fallback runs only when nothing was restored. Sessions are not restored -- the draw
  creates one lazily as it does for any tab today. No migration: an absent key is the default
  `[]`; the sandbox's `app_state.json` picks the key up as drift and is `git add`ed.
- **W2-3 (finding 8). The feedback glyph goes; the `xN` badge stays.** `_draw_feedback_glyph`
  and `GRAPH_FB_SIZE` / `GRAPH_FB_GAP` are deleted; `_draw_badge` returns to returning
  nothing if no reader of its width remains; the `prev` port's double ring stays. Reverses
  wave 1's G8 on the maintainer's own verdict after seeing it; the record's G8 gets a pointer.
- **W2-4 (finding 9). An input pin is not a drag source.** A press on any input port moves the
  node, exactly as an unfilled port's press does today; `WireDrag.grabbed` is deleted with the
  re-grab branch and `_drop`'s two `grabbed` paths, so a wire is removed only by its ✕ or
  Delete and replaced only by dropping a new wire from an output onto the port (`drop_wire`
  overwrites the source). Reverses 092 D12's re-grab half and the record's G14 "re-plug" row,
  against the convergent schema the research recorded (every reference detaches by dragging
  the wire off the input); his objection is the ghost that does not say "release removes".
  Trigger to revisit: he asks for a drag-to-detach again after living with the ✕.
  `test_a_press_that_spans_a_copilot_turn_never_becomes_a_gesture` and the crosshair-cursor
  test start their wire from an OUTPUT dot instead; a new test presses a filled input, drags
  6px and asserts `node_drag` is set, `wire_drag` is `None`, and no write followed.
- **W2-5 (finding 10). The picture grows into the margin.** `GRAPH_THUMB 96 -> 116` and
  `GRAPH_PAD 8 -> 4`: the side gap goes 20 -> 10 and the top gap 8 -> 4, both halved as asked.
  The name and label budgets follow the tokens (128 and 122 -- the label formula subtracts the
  dot, its inset and `GRAPH_PAD`) and the ellipsis test's slack
  assertions still hold.
- **W2-6 (finding 11). A bottom margin under the last port row.** `SIZE.GRAPH_PORT_BOTTOM: int
  = 7`, added to `node_size` when the card has ports, so the last label's text bottom clears
  the border by the same 10px its left inset gives it (`PORT_ROW / 2 - 6 + 7`);
  `test_a_node_grows_one_row_per_port_and_a_box_is_wider` includes the term.

Tests: `TabRecord` round trip through `UIAppState.save` / `load`; `tabs_from_records` drops a
vanished path and an unknown document and keeps a lib tab; an `app`-fixture test sets
`app.app_state.editor_tabs` to a shader, a script and a graph record and calls the restore
verb, asserting the three tabs and the active index; the W2-4 test above; the token tests
follow the values. Docs in the same commit: 092's D12 gets a `REVERSED by 093 W2-4` pointer;
the record's G8 and G14 get one-line pointers; `conventions.md`'s graph bullet loses the
re-grab clause and gains the input-pin rule; `dev_flow.md`'s `tabs/document.py` and
`pass_graph.py` entries; the ledger's "Landed in" column; `ai_docs/roadmap.md` banner.

**Wave 1: the graph editor's redesign and its move into the editor pane.** Findings
1, 2, 3, 5 closed; finding 4 delegated to its own feature. One feature flow: pre-implementation
review (two reviewers, to convergence), implementation as one diff, post-implementation review
to convergence (three reviewers per round, a spec-fidelity audit among them), sanitize, the
maintainer's hands-on pass.

## Files touched

- `shaderbox/editor_types.py` -- `EditorTabKind` gains `"graph"`.
- `shaderbox/tabs/code.py` -- `tab_label`'s graph branch; the `_draw_graph_tab` branch (T2).
- `shaderbox/tabs/document.py` -- `_draw_passes` per T5; the `PassesView` import and
  `GRAPH_MIN_H` read go.
- `shaderbox/app.py` -- `open_graph_for`, the `OPEN_GRAPH` handler, the two cursors,
  `choose_output` split out of `pick_pass` (S15).
- `shaderbox/commands.py` -- `CommandId.OPEN_GRAPH` and its spec.
- `shaderbox/ui_regions.py` -- `PassesView` / `PASSES_VIEW_LABELS` deleted.
- `shaderbox/ui_models.py` -- `UIAppState.passes_view` deleted.
- `shaderbox/theme.py` -- S12.
- `shaderbox/ui_primitives.py`, `shaderbox/popups/lib_picker/tree.py` -- S11.
- `shaderbox/widgets/uniform.py` -- `_locate_uniform_declaration` uses the non-creating getter
  (T1).
- `shaderbox/widgets/pass_list.py` -- its docstring's "shared with the graph view" clause about
  the add / import row is corrected (the graph has its own tab).
- `shaderbox/widgets/graph_state.py` -- S9's functions, `WireState`, `revalidated_wire`; the new
  `GraphViewState` fields (`hovered_node`, `hovered_port`, `hovered_out`, `hovered_wire`,
  `selected_wire`, `x_rect`, `node_order`, `wire_mids: dict[tuple[str, str], Point]` -- each
  drawn wire's screen-space midpoint this frame, for the headless test to aim a click at).
- `shaderbox/widgets/pass_graph.py` -- the canvas: G1-G3 (`_draw_wire` rewritten, bus and
  `_draw_self_loop` deleted, the edge loop simplified, `_Edge.span` gone), G4-G6 (the wire pass,
  the hover reads and writes, the halos, the per-dot colors), G7 (cursor requests), G8
  (`_draw_feedback_glyph`), G11 (ellipsis), G12/S6/S7 (five channels, the sorted node list),
  G13/S13 (the lock), S2-S5, S8, S15, T3 (`draw(app, document_id)`). The module docstring is
  updated for the hover model, the wire pass, the hand-tested ✕ and the channel layout.
- `scripts/smoke.py` -- T6.
- `tests/test_graph_state.py`, `tests/test_graph_view.py` (gains
  `pytestmark = pytest.mark.xdist_group("gl_frames_graph_view")` -- it drives frames today with
  no group, green by luck of the worker split), `tests/test_theme.py`,
  `tests/test_pass_settings_layout.py`, `tests/test_anchored_note.py` (the rename), new
  `tests/test_graph_tab.py` with `xdist_group("gl_frames_graph_tab")`; `tests/test_ui_regions.py`
  deleted.
- Docs, same wave: `ai_docs/features/092_graph_view/03_spec.md` (D2's and D10's pointers +
  Review history entry for the two reversals), `ai_docs/conventions.md` (the 092 bullet: the
  canvas's home is the editor tab, the wire is one bezier by construction, the hover model, the
  click chooses the output; the "revisit the canvas's home" clause resolved; the editor-tab
  bullet's "a 4th editable `kind` lands" trigger gains the clause that a non-editable kind --
  the graph -- does not fire it, since it has no session), `ai_docs/dev_flow.md` module map
  (`widgets/pass_list.py`, `widgets/pass_graph.py`, `tabs/document.py` and `tabs/code.py`
  entries), `shaderbox/help_content.py` (the "graph view" phrase becomes the graph tab),
  `00_findings.md` (the "Landed in" column), `ai_docs/roadmap.md` (row + banner), this file's
  status.

## Verification

The falsifier per guarantee, from the record's sketch, made concrete. Pure tests import
`graph_state` / `theme` only; frame-driven ones use the `app` fixture and
`tests/test_graph_view.py::_frames` with `io.add_mouse_*` / `io.add_key_event`, aiming through
`view.port_rects`, `view.wire_mids`, `view.x_rect` and `view.canvas_rect`. Three mechanics every
frame-driven test obeys: a key event queued in the same batch as a mouse-button event reaches
`is_key_pressed` one frame LATER than the button (measured), so a `_frames` call separates
them; a move and a release in one frame read as a click (S4), so they are separate frames too;
and a test that needs the canvas to have drawn asserts `view.canvas_rect != (0, 0, 0, 0)` on
that frame before reading anything else. A measurement that needs a font (`ellipsize`,
`calc_text_size`) runs inside `imgui.new_frame()` / `imgui.begin("rig")` / `push_font` /
`imgui.end()` / `imgui.end_frame()` on the `app` fixture, the shape of
`tests/test_pass_settings_layout.py::test_the_auto_name_column_fits_every_engine_uniform`;
outside a frame `calc_text_size` segfaults the process (measured).

| Guarantee | Test | Kind |
|---|---|---|
| G1: no cusp for any endpoints | `wire_points` over a grid of `(dx, dy)` covering backward, forward, near-zero and long runs plus `dist = 0`, at zooms 0.25 / 1 / 2.5: the offset is non-negative everywhere, and `cp0.x > cp1.x` whenever `dx < 0` | pure |
| G1: continuity across the old bus boundary | sweep `dx` over `[-48, 48]` in 1px steps at `dy = 40`: each control point moves by less than `1.0 + 2 * GRAPH_WIRE_BOW` px per step (the endpoint's own 1px plus the offset's change; a topology switch is a jump of tens of px). Measured today's worst step at these tokens: 1.31px | pure |
| G4: the threshold and its floor | `wire_hit_threshold(0.25) == 6.0`, `(1.0) == 6.0`, `(2.5) == 7.5`; `wire_hit` on a known cubic: a point on the curve returns `< 0.5`, at `threshold - 1` a hit, at `threshold + 1` `None`; the bounding-box reject returns `None` for a far point. Break to try: drop the floor -- `(0.25)` reads 0.75 | pure |
| G4: the flattening misses no real hit (regression guard) | 200 points along the true cubic of a 400px backward S-curve each report a distance under `threshold`; this discriminates a count of 6 or below (worst error 9.5px at 6, 2.7 at 8), so it guards against a catastrophic count, not the choice of 24 | pure |
| G18: an error wire stays red while hovered | `wire_state` over all sixteen `(on_cycle, selected, hovered, dim)` combinations against the precedence table | pure |
| S2: a stale wire selection clears | `revalidated_wire(("c", "u_src"), edges)` returns it while an edge carries the pair and `None` once none does; `None` in gives `None` out | pure |
| G6: nothing changes size on hover | `inspect.signature(node_size).parameters` is exactly `(port_count, box)`; `wire_hit_threshold` takes `zoom` alone | pure |
| G8/G3/S12: the loop, the bus and the module constants are gone | `SIZE` has none of `GRAPH_LOOP_RISE`, `GRAPH_LOOP_REACH`, `GRAPH_BUS_STEP`, `GRAPH_BUS_CLEAR`, `GRAPH_MIN_H`; the widget's source contains none of `_draw_self_loop`, `bus_y`, `_MIN_DIRECT_DX`, `_BEZIER_BOW`, `glfw.set_cursor` | pure |
| S13: the lock is passed everywhere | an `ast` walk over the widget: every `is_mouse_dragging(` and `get_mouse_drag_delta(` call carries `GRAPH_DRAG_LOCK_PX` as a real argument, and the site count is the expected one (a token in a comment satisfies nothing) | pure |
| G12: the five channels in order | the widget's source calls `channels_split(5)` once and `channels_merge()` once, and the five named channel constants (`_CH_HALO`, `_CH_WIRE`, `_CH_NODE`, `_CH_INFLIGHT`, `_CH_OVERLAY`) are strictly ascending -- paint order follows the index, so the constants' order IS the layering; the call sites are not in source order (the overlay channel is entered from `_draw_wire_x` early in the file) | pure |
| G11: the ellipsis pins the width | inside a rig frame with `app.font_12` pushed at its `legacy_size`: `ellipsize("u_distance_field", budget)` at the 128-card port-label budget (110) ends in `...`, and at the decided card's budget (`GRAPH_NODE_W - 2 * GRAPH_PORT_R - 2 - GRAPH_PAD`, 118 at 136) returns the string unchanged with at least 4px of slack; with `app.font_14_bold`: `distance_field` against the name budget (`GRAPH_NODE_W - 2 * GRAPH_PAD`, 120 at 136) unchanged with at least 4px of slack. This is the width decision's pin -- red at 128 (measured: 112px against 110 and 112), green at 136 | rig frame |
| S8: the fit frames every wire | build `a -> b -> c` plus a backward read (`a` reading `c` through `app.session.set_sampler_source`, so a cycle wire is present), `_fit` with `avail = (520, 200)` -- small enough that the 1.0 clamp does not centre slack around the content (at 800x600 the break below stays green, measured); the fitted window in canvas space is `(pan.x, pan.y, pan.x + 520 / zoom, pan.y + 200 / zoom)`, and every wire's 25 sampled canvas-space curve points lie inside it. Break to try: fit the nodes alone -- 8 of 75 points land outside (measured) | app fixture, no frames |
| `_fit`'s clamp (regression check, not a width pin) | six chained passes, `_fit` with `avail = (1225, 600)`: `view.zoom == 1.0`; with `(740, 600)`: `0.6 < view.zoom < 1.0`. Green at 108 and 136 alike; the width is pinned by the ellipsis row | app fixture, no frames |
| G5: select a wire and Delete it | open the graph tab, frames, click `view.wire_mids[("c", "u_src")]`, assert `selected_wire == ("c", "u_src")` and `selection == set()`, release, frames, send `Key.delete`, frames, assert exactly one `set_sampler_source(..., "c", "u_src", NoSource())` and nothing else | frame-driven |
| G5: the ✕ unwires, and the press is nothing else | select the wire as above, press at the centre of `view.x_rect`, hold two frames, release, frames: exactly the one unwire write; `band_anchor is None` and `node_drag is None` throughout; `set_output_pass` was never called and `view.selection` is unchanged. Then place a node (through `app.session.set_pass_positions`) so its body covers the selected wire's midpoint but not the port the wire ends at -- centred on the midpoint the body also covers the port at 136, so shift it 50px or more toward the producer (measured) -- and repeat: the same single write and still no `set_output_pass` -- the release-frame node click is refused because the latch clears at the end of the frame. Break to try: clear the latch at the top of the frame as today -- the covered case chooses the output | frame-driven |
| S5: Delete is refused while a press is held on the canvas | select a wire, press and hold on empty canvas, send `Key.delete`, frames: no write; release, frames, send `Key.delete`: the write. A behavior pin; the clause it exercises is `hovered` (measured: `is_window_hovered` is False for the whole held press) | frame-driven |
| S5: the Delete gate over its whole domain | `delete_allowed` over all 32 combinations of its five booleans: True exactly when `pressed and hovered and not any_item_active and not blocked and has_wire`. The `any_item_active` clause's live scenario (a text input active in another window) is the maintainer's check | pure |
| S5: Delete typed into the group prompt is refused | select a wire, set `view.group_prompt = True` (a one-shot the first frame consumes; do not re-assert it), frames, send `Key.delete`, frames: no write. Behaviour pin; the clause it exercises is `hovered` | frame-driven |
| S5: Delete is refused during a copilot turn | select a wire, `app.copilot.state.in_flight = True`, frames, send `Key.delete`: no write (the `not blocked` clause) | frame-driven |
| G6: exclusive hover, one rung per sequence, no click in any | (a) park on `port_rects[("c", "u_src")]`'s centre: `hovered_port` set, the other three `None`; (b) park on a node body away from any port and wire: `hovered_node` set, others `None`; (c) park on `wire_mids[("b", "u_src")]` in open canvas: `hovered_wire` set, others `None`; (d) place `c` (through `app.session.set_pass_positions`) so its body covers `wire_mids[("b", "u_src")]`, park there: `hovered_node` set and `hovered_wire is None`; (e) off the canvas: all `None`. Break to try: swap the node and wire rungs (d flips). The port-over-node rung is enforced by imgui's item chain, not by the resolution order -- at a port dot the node button's own hover is already False because the later port button takes the overlap (measured) -- so swapping those two rungs changes nothing and is not a gate; the test's docstring says so | frame-driven |
| S3: the hover fields are exactly the ones written | the set of `GraphViewState` fields whose name starts with `hovered_` is exactly `{hovered_node, hovered_port, hovered_out, hovered_wire}`, and sequence (e) above leaves each `None` -- a fifth field nobody wires is caught | pure + frame-driven |
| S4: the selections are exclusive | after selecting the wire, click a node: `selected_wire is None` and `selection == {node}`; select the wire again, rubber-band over empty canvas and release: `selected_wire is None` | frame-driven |
| G13/S15: 3px is a click, 5px is a drag | press on a node's body, move 3px, release: `set_output_pass` ran once, `set_pass_positions` did not, and `app.active_tab.kind == "graph"` still; press, move 5px, release: `set_pass_positions` ran, `set_output_pass` did not. Break to try: omit `lock_threshold` at the node-body site -- the 5px case reads imgui's 6px default and stays a click (measured today: 5px is a click, 8px a drag) | frame-driven |
| S15 as amended by W3-3: no click opens the shader tab | double-click a pass node: `app.active_tab.kind == "graph"` still and the pass is the output | frame-driven |
| S7: the selected node draws and hit-tests last, a dragged one above it | select `a`, frames: `view.node_order[-1] == "p:a"`; with `b` selected, drag `a` and read `node_order` mid-drag: `[-1] == "p:a"` | frame-driven |
| G7: the cursor follows the gesture | during a middle-drag pan, after the frame, `app.cur_cursor is app.hand_cursor`; at rest, on a frame where `view.canvas_rect != (0, 0, 0, 0)`, `app.cur_cursor is None` | frame-driven |
| G14: every kept binding still works | the existing `tests/test_graph_view.py` passes with only the `passes_view -> open_graph_for` substitution and the new `pytestmark`; a failure that traces to a binding is a silent change, a failure that traces to the tab being inactive is a test-mechanics bug (the tab is opened before any copilot-turn simulation, since `open_graph_for` is frozen during one) | frame-driven |
| T1-T6: the tab | `tab_label` reads `"<name> (graph)"` on a multi-pass document; `open_graph_for` twice yields one tab and it is active; `is_tab_dirty` False; `formatter_for("graph") is None`; `format_current_editor` and `jump_to_next_error` return without error on a graph tab; `close_editor_for_path` removes it; `_on_document_deleted` removes it and keeps a lib tab; `command_callbacks[CommandId.OPEN_GRAPH]` exists; three frames with the graph tab active leave `view.fitted` True and `app.editor_errors == []` | app fixture + frames |
| T1: the Uniforms panel creates no session over `graph.json` | with the graph tab active, inside a rig frame, call `widgets.uniform._locate_uniform_declaration(app, "u_src")` directly (it runs only on a hover or click of a uniform's name, which four frames of the focused tab never inject -- measured), then assert `app.paths.graph_json_for(document_id) not in app.editor_sessions`. Break to try: the creating `get_current_session()` -- a session appears at the graph path | rig frame |
| T5: the tick predicate | with document A's graph tab active, `_entry_tab_active(app, A, "graph")` is True, `_entry_tab_active(app, B, "graph")` False, `_entry_tab_active(app, A, "script")` False; the drawn tick is the maintainer's eyes, as the Script row's is | app fixture |
| Theme | `test_group_tints_are_stable_and_collide_with_nothing` imports `_GROUP_TINT_EXCLUSIONS` and asserts `GRAPH_HOVER` is in it, `GRAPH_HOVER not in {primary for primary, _, _ in _ACCENTS.values()}`, and `GRAPH_HOVER` differs from `SELECT`, `STATE_ERROR`, `GRAPH_EDGE`. Break to try: `blue_b` -- the accent clause goes red | pure |

**Gates that must be broken before they are believed.** G13's lock, G6's order and G4's floor
are gates, not tests: each is landed by breaking the guarded thing (omitting `lock_threshold`
at one site; reversing the node and wire rungs; dropping the
`max(6.0, ...)` floor), watching the named test go red, restoring it. The implementation
commit's body says which break was tried for each. `make gates` green with the smoke run (not
skipped -- it passes on this box without `xvfb-run`, measured) before the commit, judged by its
exit code captured unpiped. Every new frame-driving test module declares its own `xdist_group`
(`pyproject.toml` states the rule: the imgui font atlas is process-global).

**The maintainer's eyes (no gate can run these):** the halo never reads as a thicker wire at
0.25 and 2.5 zoom; the feedback glyph's shape at 12px; a selected card over a neighbor; the
neutral hover against the grey wire; the 136 card with his own names and the 176 box; the 4px
lock under his hand; the Passes row's label in the ambient font with its tick. Each of the four
forks is then a token edit.

## Open questions for the user

None. The four forks ship at the record's recommendation by his instruction and are tuned
against the rendered result; the one correction (S1's hue) follows from a measured collision.

## Review history

**Pre-implementation round 1 (2026-09-14): two reviewers on opus, both PARTIAL, no
should-not-land.** Reports: `reviews/pre_correctness_design.md` (11 findings),
`reviews/pre_verification_blast.md` (9 findings + 21 row verdicts). Every finding was
demonstrated by a probe or a quoted line and all were folded in; none was rejected. The two
design calls they forced: **S15** (a node click chose the output through `pick_pass`, which
activates the shader tab and so evicted the graph tab on every click -- the click now chooses
the output only, the double-click opens the tab; reverses 092 D10's click half) and **S6**
(submission order cannot make the ✕ win over a port that declares no overlap -- the ✕ is
hit-tested by hand on the press and latches `press_blocked`). S1's hue changed from `blue_b`
(the blue accent's primary) to `fg_0`. Pinned positions: the wire pass between the button loop
and the background-press action (S3), the Delete read after the hit-test section (S5). Folded
rows: the fit's containment target restated in canvas space and the fit itself framing the
sampled curve rather than the control polygon (S8); the theme row given the accent-primary
clause that actually bites; the six-column fit row demoted to a clamp check with the width
pinned by the ellipsis row; the 24-segment row labelled a regression guard; the hover-order
row split per rung with a node-over-wire position; the cursor row reading `cur_cursor`; the
channel row checking the assignment order; the `_ellipsize` readers outside `ui_primitives`
added to Files touched; `tab_label`'s branch and `uniform.py`'s creating getter named as
required edits; `_Edge.span`'s removal named; `GRAPH_WIRE_HIT_MIN` renamed; xdist groups for
both graph frame modules; S2's revalidation, S3's field enumeration, S4's exclusivity, S5's
`blocked` clause, S7's order and S14's no-raw-`set_cursor` rule each given a row.

**Pre-implementation round 2 (2026-09-14): the same two reviewers, both PARTIAL, seven new
findings, all demonstrated and folded; every round-1 item confirmed CLOSED by quoted text.**
Round-2 sections in the same two reports. The width fork moved: the font advance is 7.0 / 8.0px
in a rig frame, so 128 truncates the very names it was widened for; **136** (S1). `choose_output`
clears the Uniforms pin, or a persisted pin outlives a canvas click (S15). The ✕ latch needed two
orderings -- the check above `blocked`'s computation and the clear at the frame's END -- since
S4's release-time click would otherwise fire on a ✕ over a card (S6). The `is_any_item_active`
clause's held-press falsifier was shown redundant with `hovered`; its reachable case is a text
input active in another window, built through the chat's focus (S5). The fit row's break was
green at 800x600 (the clamp centres slack) and bites at 520x200. `_locate_uniform_declaration`
runs only on a hover or click, so its row calls it directly. `graph_active` becomes the callable
`_entry_tab_active`, shared with the Script row (T5).

**Pre-implementation round 3 (2026-09-14): both reviewers PASS.** Every round-2 item confirmed
CLOSED by quoted text; S6's two orderings traced against `_draw_canvas` (the same-frame
background press and the release-frame node click both refused, the copilot-turn test's last
stanza intact); the 136 width recomputed against the record's own G11 table (1.049 / 0.634,
unchanged). One demonstrated residual folded: the chat input cannot be focused in the fixture
(no OpenRouter key, so no input is drawn), so the Delete gate is now the pure `delete_allowed`
tested over its 32 combinations and the live text-input scenario is the maintainer's check; the
✕-over-a-card row names the 50px offset that covers the midpoint without covering the port.

**Post-implementation round 1 (2026-09-14): three reviewers on opus -- correctness PASS (one
FRAGILE), architecture PARTIAL (one violation, three smells), fidelity PARTIAL (five rows);
`make gates` exit 0 with the smoke run, reproduced by two of the three.** Reports:
`reviews/post_code_correctness.md`, `reviews/post_architecture_conventions.md`,
`reviews/post_spec_fidelity.md`. Accepted and fixed in `ccdf6d1`: the ledger's literal
`commit <sha>`; a comment narrating the change and a stale one naming the deleted self-loop;
the bring-to-front key, which put a merely selected card over one in flight (now
`(is_being_dragged, is_selected)`, S7 amended); the drag-lock source gate made an `ast` walk,
since a token in a comment satisfied the text window (demonstrated); and the spelling gate's
roster, which held neither `centre` nor `neighbour` and so passed fifty-two British spellings
across thirteen files -- the roster grows and every site it then flags is fixed in the same
commit, which also rebuilt the 068 tutorial page since its body is on the gate's roster. Spec
text corrected here: `_draw_wire` carries `zoom`; the channel row describes the shipped check
(the five constants ascending, since paint order follows the index); the port-over-node hover
rung is enforced by imgui's item chain, so its swap is struck from the breaks. Rejected: the
1.4x hovered stroke (the record's G6 prescribes it); the doubled-ghost `port_rects` collision
(pre-existing, and the live drop reads `drop_target`).

**Post-implementation round 2 (2026-09-14): all three reviewers PASS.** Each round-1 item closed
by a quoted line; the bring-to-front fix re-driven (`node_order` mid-drag ends on the dragged
card); the `ast` lock gate shown red on a commented token where the old text window stayed
green; the spelling roster and sweep in one commit with zero hits left outside the gate's
exclusions; `make gates` exit 0 with the smoke run. Wave 1 is closed; the maintainer's hands-on
pass is the next input.

**Wave 2 post-implementation round 1 (2026-09-14): two reviewers on opus -- correctness PARTIAL
(one defect, two fragile), fidelity PARTIAL (seven text rows); gate exit 0 with the smoke run.**
Reports: `reviews/post_w2_code_correctness.md`, `reviews/post_w2_fidelity_conventions.md`.
Fixed in the follow-up commit: the Graph label and button on the Script row clipped at the
narrow panel (own row under it, W2-1 amended); the active tab persisted by position landed on
the wrong file once an earlier record was dropped (by path, W2-2 amended); two comments
narrating the change; `centring` added to the spelling roster with its one live site. Corrected
here: the spec's Status fragment, the label budget (122, an arithmetic slip), and the research
readout behind W2-4, which had overstated the convergence -- Houdini is click-based, imnodes
opt-in, ImNodeFlow without the gesture; the claim now names the references that draw a wire
from a filled input. Rejected: a value pin on `GRAPH_THUMB` / `GRAPH_PAD` (tuning values, his
eyes). The banner regained wave 1's "still unseen" list.
