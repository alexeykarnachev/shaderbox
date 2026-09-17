# 094 — The graph is the panel

## Status

**Locked after four review rounds** (ten reviewers; the log is `02_review_log.md`, the commit
sequence is `03_work_order.md`). Nothing implemented yet.

The rounds earned their keep: round 1 found that `App.panel_pass` could not be deleted as drafted,
round 2 that two of round 1's fixes had never been written to the file, round 3 that per-pass render
as specified would have produced BLACK exports, and round 4 that round 3's own node-width decision
had not been carried into four downstream passages. The recurring lesson, recorded because it will
recur: **when a decision is rewritten, re-read every decision and check that cites it.**

Supersedes the framing in `00_mock.html`, which asked how to make the documents grid denser. The
answer this spec takes is that the grid, the four-tab settings panel and the pass strip all go: the
lower half of the app panel becomes the graph, and everything those surfaces held moves onto the
graph, onto the canvas's toolbelt, or is deleted.

---

## Goal

**One surface answers every question about a document's passes.** Today six do: the pass strip, the
graph (an editor tab), the Uniforms tab, the Document tab, the Render tab and the Share tab. The
same pass is drawn as a tile in the strip, as a node in the graph, and as a pass selector in
Uniforms; its uniforms live on a tab that has no idea which pass you are looking at.

After this feature:

- The app's layout is **unchanged**: editor left, rendering canvas top-right, and under it the
  region that is today the control panel.
- That region is **the graph canvas**, edge to edge. It is no longer an editor tab.
- A node carries its pass's **uniforms**, live and editable, at close-up zoom.
- **Render** and **Share** are modes of a node: the view centres on it, the node grows, the rest of
  the canvas dims. The existing `tabs/render.py` and `tabs/share.py` bodies are reused.
- The documents selector is a **dropdown on the graph's breadcrumb**, with small live previews.
- The canvas shape and the channel view move into **the fps chip's menu** — one chip on the canvas,
  reading the current document's rate (D13a). The frame profiler's panel is deleted (D13b).
- `Render all documents` is **deleted**; the existing throttle already answers it.

The measure of success is subtractive: `ui_regions.DocumentTab`, `tabs/document.py`,
`tabs/uniforms.py`, `widgets/pass_list.py`, `widgets/document_grid.py`'s grid, four `FOCUS_TAB_*`
commands and `CommandId.OPEN_GRAPH` all stop existing, and nothing the user could do before becomes
impossible.

---

## Out of scope

Each deferral names the trigger that would reopen it.

- **A second exporter's Share layout.** Share is specified against the one available exporter
  (telegram). *Trigger:* a second exporter becomes available — then re-measure the focused node's
  Share body, which is an accordion and may need its own scroll.
- **Restoring an at-a-glance documents view.** The dropdown replaces the grid; no thumbnail strip is
  built. *Trigger:* the maintainer reports that picking a document by name-plus-click is slower than
  the grid was.
- **Node auto-layout on focus exit.** Focus never moves a node, so nothing needs re-arranging.
  *Trigger:* a focused node's fixed size is found to overlap in a way that survives leaving the mode
  (it cannot, by D7, but a bug here would show as a node drawn at panel size on the normal canvas).
- **Editing a uniform's bounds or input type on the node.** The node draws the value control only;
  configuration stays where D12 puts it. *Trigger:* the maintainer asks to retype a uniform without
  opening the focused node.
- **Group boxes carrying uniforms.** A group box is not a pass and has no uniforms; it keeps the
  body it has today. *Trigger:* none — a box has no uniform set to show.
- **A drag-time render throttle for the splitter.** The resolution path already damps
  (`apply_damping`); the splitter adds no new one. *Trigger:* dragging the splitter on a heavy
  document is measured to stall, which would mean the damping does not cover a continuous resize.

- **Three things the deleted surfaces made EASY that the graph makes merely possible.** Stated as
  accepted costs rather than discovered later, each with the trigger that reopens it.
  (1) *Comparing two passes' pictures.* The strip drew every pass in a wrapping grid, all at once;
  the graph spreads them over a canvas, so two passes the user wants side by side are wherever the
  layout put them. D4e removes the SIZE half of this — a 220px node picture is larger than the
  strip's 168px tile — but not the adjacency half.
  *Trigger:* the maintainer reports comparing passes is slower than it was.
  (2) *Switching the output by clicking a picture.* The tile's whole 168px was the click target; a
  node's click is EXCLUSIVE with its drag, so a tremor during the click commits a position change to
  disk instead. *Trigger:* a mis-drag is reported, or the click-vs-drag threshold needs tuning.
  (3) *Reading a pass's inputs as names.* The strip drew one chip per source pass under each tile;
  on the graph that fact lives in wire geometry, and a pass with four samplers on a busy canvas is a
  tracing exercise. The ports do carry the SAMPLER names, which is the other half.
  *Trigger:* a document whose wiring is routinely read rather than edited.

- **Touching the throttle's algorithm.** `plan_render_set` is not modified; only the set handed to
  it changes (D15). *Trigger:* the dropdown's previews are measured to starve the current document,
  which would mean the `others` branch is wrong, not the set.

---

## Design decisions

Numbered, lock-in only. Open questions are in their own section below.

### The region

**D1 — The app's layout does not change.** `ui.py` keeps the editor/app split, its splitter, and
inside the app panel the rendering canvas above and a region below. Only that lower region's
*content* changes.

**D1a — A horizontal splitter is ADDED between the rendering canvas and the graph.** This is new
behaviour, not a restatement: today the canvas's height is *derived*, not dragged. `ui.py::_draw_document_image`
computes it as the panel's width at a fixed aspect, capped by the room above the control panel's
minimum:

```python
box_height = max(min(avail.y - control_panel_min_height - 10, box_width / VIEWER_BOX_ASPECT), 100.0)
```

and `ViewerGeometry`'s docstring says the box "moves only when the splitter moves" — meaning the
*vertical* splitter, indirectly through width. There is no horizontal one, so the canvas cannot be
made taller or shorter on purpose.

The new splitter mirrors `_draw_splitter` exactly, on the other axis: an `invisible_button` of
`_SPLITTER_W` across the panel's width, `resize_ns_cursor` while hovered or active, and a drag that
accumulates `mouse_delta.y / total_height` into a persisted fraction. It replaces the derived
`box_height` with `canvas_split_fraction × panel_height`, clamped the way the existing one is.

Three consequences to get right, each of which is a verification below:

- **`VIEWER_BOX_ASPECT` stops deciding the height** and the fixed-aspect box goes with it. The
  picture is still fitted and centred inside whatever box the splitter gives (that half of
  `_draw_document_image` is unchanged); what changes is where the box's height comes from.
  `ViewerGeometry`'s docstring, which states the derivation, is rewritten.
- **`SIZE.PANEL_CTRL_MINH` stops being the cap.** The clamp becomes the splitter's own bounds, so
  the minimum is a splitter limit rather than a control-panel constant. The token is then read by
  nothing and is deleted with the control panel.
- **This resizes every Auto document, live.** `app.viewer_region` is assigned from this box
  (`ui.py::_draw_document_image`) and is what every Auto-mode document renders at next frame (090, revision 1). So
  dragging the splitter is not a layout preference — it changes render resolution, which
  `_resolve_resolutions` then reallocates canvases for. That is correct and wanted, but it means a
  drag must not thrash — and the existing damping does NOT cover it. `_past_dead_band` returns the
  request immediately whenever an axis moves more than 5%, so only a NOISY change is damped, not a
  continuous one: on a 400px canvas, 5% is 20px, which an ordinary drag crosses in a few frames.
  Each application resizes the canvas and resamples every feedback history.

  So the applied size is LATCHED to the drag: held while the splitter's `invisible_button` is
  active, applied once on release. That is the same press-latch shape `App.update_splitter_drag`
  already uses for the vertical splitter, and it makes the deferral about heavy documents moot
  rather than merely unlikely.

**D1c — A region-size change clears `fitted`, or the graph is cropped with no way back.**
`GraphViewState.fitted` is a one-shot: `_fit` runs once at the first nonzero size, then never again
unless the scope changes or `Frame all` is clicked. That was right for an editor tab, whose pane
only resized when the vertical splitter moved. In its new home the region's height is a DRAGGED
control, so every drag changes the canvas under a camera that will not re-fit — the nodes are simply
cropped, with no cue and no recovery except a context-menu item the user has to know about.

`_draw_canvas` already reads `avail` every frame. When it changes by more than a threshold, clear
`fitted`. Three lines, and it makes the splitter read as resizing a view rather than cropping one.

**D1b — The split fraction persists on `app_state`, with its constraint on the model.**
`canvas_split_fraction: float = Field(default=…, ge=0.0, le=1.0)` beside `editor_split_fraction`,
which carries exactly that shape (`UIAppState.editor_split_fraction`). Per the locked persisted-model decision, a
knob that moves into a file gets its constraint on the field, never a check at the call site. The
default reproduces today's derived height at a typical window size, so the first launch after this
feature looks unchanged.

**D2a — The copilot-turn freeze must not disable the breadcrumb, and TWO brackets stand in the
way.** `pass_graph.draw` wraps the whole widget — `_tab_row` included — in
`begin_disabled(app.copilot_turn_active)`. But `ui.py` ALREADY wraps the entire app panel in another
one, and imgui's own binding says a nested `begin_disabled` cannot re-enable: *"a single
BeginDisabled(True) in the stack is enough to keep everything disabled."* So moving the inner bracket
changes nothing — an implementer following an earlier draft of this decision would watch check 14
fail with no idea where to look.

Both brackets move. `_draw_app_panel` takes over the bracketing per region rather than wrapping
itself: the rendering canvas and the graph's canvas child stay frozen (the outer bracket's own
comment says it exists to freeze the uniform sliders), while the breadcrumb's document SELECTION
stays live. A user who cannot leave a document during a long copilot turn is stuck, and switching
documents mid-turn is already supported.

Note the coupling this must not break: `_draw_document_image` pushes `StyleVar_.alpha = 1.0` around
the preview precisely because the outer bracket scales alpha for everything in the panel. Whatever
replaces the outer bracket keeps that working.

**D2 — The lower region is one child hosting `pass_graph.draw`.** `_draw_app_panel`'s two-column
body (`draw_document_preview_grid` beside `_draw_document_settings`) is replaced by a single call.
The graph widget keeps its current contract: it fills the host's content region, positions no
sibling and measures none (`widgets/pass_graph.py::draw`'s docstring). This is what makes the change
small — the widget already draws into an arbitrary box.

**D3 — The graph stops being an editor tab.** `tabs/code.py::_draw_graph_tab`, the `"graph"` tab
kind, `App.open_graph_for` and `CommandId.OPEN_GRAPH` are deleted. The graph is always on screen, so
a command to summon it names a surface that cannot be absent. The tab-kind branch in
`tabs/code.py::_tab_title` goes with it.

Consequence to handle, not a side note: `_draw_graph_tab` also owns the graph pane's focus
bookkeeping (`editor_focused`, `editor_was_ever_focused`, clearing `editor_errors`). The graph in
its new home is **not** part of the editor's focus domain — it must not set `editor_focused`, or
`Ctrl+W` would try to close it and `F8` would read a cleared error list. The new host sets none of
those flags.

### The node

**D4 — A node draws its pass's uniforms under the name, at close-up zoom only.** The rows are the
pass's own active uniforms, which is what makes the Uniforms tab's pass selector unnecessary: the
node *is* the selection.

**D4a — The target becomes an ARGUMENT; `panel_pass` survives as the chord-time resolver.** This is
the feature's real cost, and the first draft priced it as a table row. Three reused bodies resolve
their target from global state today, and the whole point of the feature is to make the target a
specific node:

- `widgets/uniform.py::draw_ui_uniform` takes no pass. It calls `App.panel_pass` internally and
  reads the result at roughly fourteen further sites, including the write-back. Two nodes' rows
  cannot be drawn in one frame with this signature — worse, a node would silently show and WRITE
  another pass's values.
- `widgets/uniform.py::_locate_uniform_declaration` walks `App.panel_pass(...).compile_unit.sources`
  for the jump-to-declaration and the editor hover bridge. It needs the same argument, and the
  first draft named only `draw_ui_uniform`.
- `App.open_shader_for_panel_pass` and `App.open_pass_settings_for_panel_pass` resolve `Open shader`
  and `Pass settings` the same way.

The conventions' pass-resolution decision LOCKS this and names its own escape: *"Revisit if a
surface needs a pass the user did NOT pick, which wants its own argument rather than a fourth
tier."* A node is a pass the user did not pick in that sense — it is the pass being drawn. So:

1. `draw_ui_uniform`, `_locate_uniform_declaration`, `uniform_name_label` and `_draw_auto_block`
   each gain an explicit `render_pass: Pass` parameter; every internal `panel_pass` read becomes that
   argument. `_draw_auto_block` is in the list because it resolves through `panel_pass` too — hosted
   on a focused node it would otherwise show the PANEL pass's auto values inside another pass's
   body.
2. The two command resolvers keep using `App.panel_pass`. A chord fired with no node under the
   cursor still needs "the pass being worked on" inferred, which is exactly what the tier chain is
   for.
3. `ui_state.panel_pass` and `App.panel_pass` therefore SURVIVE, minus the Uniforms-tab writer.
   `App.choose_output` and `App.ensure_shader_tab` keep recording the pick, so the chord tier keeps
   being fed.
4. The convention is amended in this wave: its list of resolving surfaces names "the uniforms
   panel", which this feature deletes, and the resolution is by ARGUMENT where a surface names its
   own pass, by `panel_pass` where a chord must infer one.

**D5a — `_fit` never zooms IN, so the interactive level must be reachable.** `_fit` clamps
`zoom = min(1.0, avail/w, avail/h)`, so `Frame all`, a scope change and every first fit land at
zoom 1.0 or below — a node at most `GRAPH_NODE_W` wide. The "near" level where rows are interactive
is therefore only reachable by deliberate wheeling, every time.

Two consequences the thresholds must respect: the near threshold sits at or below zoom 1.0, so a
freshly-fitted graph shows rows rather than requiring a wheel; and the row's control width is
budgeted against `GRAPH_NODE_W × zoom`, not against `UNIFORM_CTRL_W`. At zoom 1.0 that is 240px
(D4e) — about 182px of control after the name column — which is what makes D4's compact table (no
texture row, no text row, no array row) a width decision rather than a taste one.

**D5 — Three levels of detail, keyed on zoom.** The node sheds what has stopped being readable
rather than shrinking it:

| Level | Shows | Rationale |
|---|---|---|
| **near** | picture, name, ports + labels, uniform rows | the only level that accepts input |
| **mid** | picture, name, ports (no labels, no rows) | the name is drawn at a floor size, so it stays legible while the node shrinks |
| **far** | picture and ports only | text at this scale is a smudge that costs a font push per node |

The name's font at **mid** does not scale below a floor — that is the entire reason the level
exists. The two thresholds are tuning constants, not spec'd values (see Open questions).

**D4e — `GRAPH_NODE_W` goes 136 → 240, because the rows must be readable at zoom 1.0.** `_fit`
clamps at 1.0 and never zooms in, so zoom 1.0 is where realistic documents land and where the rows
have to work. At 136 the control column is ~86px, giving a `vec3` 29px per component and a `vec4`
22px — against roughly 38px to render "0.000". Vector uniforms are most of what a shader declares,
so the common case was the unreadable one.

At 240 the control column is ~182px: `vec3` 61px per component, `vec4` 46px. Both clear the text
width with margin for the drag grab.

What moves with it: `GRAPH_THUMB` becomes `240 - 2 × GRAPH_THUMB_INSET` = 220, preserving the
theme's own assertion that the node width is the thumb plus two insets.

`GRAPH_BOX_EXTRA_W` (40) needs looking at, and the earlier draft's "nothing else reads the width"
was wrong: `node_size(port_count, box=True)` returns `GRAPH_NODE_W + GRAPH_BOX_EXTRA_W`, so a group
box goes 176 → 280 while `_thumb_rect` still centres a 220px thumb in it — 30px of inset per side
against the card's own 10, so the box's picture stops reading as inset by one rule. The extra shrinks
to keep the box's inset proportional.

`GRAPH_GAP_X` (64) and `GRAPH_GAP_Y` (20) need **no** change, and the spec says so rather than
leaving it as judgement: both are read only by `rank_layout` as gaps BETWEEN boxes, and nothing
derives them from the node width. `GRAPH_WIRE_MIN_OFF` (24) is likewise a canvas-unit floor
independent of it.

`tests/test_graph_tab.py`'s width pin goes stale: its docstring says "red at 128 …, green at 136"
and it asserts against a hard-coded 110.0 label budget. At 240 it passes while pinning nothing — a
gate that cannot fail. It is re-anchored to the new width or deleted, in this wave.

The picture gets bigger too, which answers the first accepted cost above: a 220px node thumb against
the strip's 168px tile means comparing two passes' pictures is no longer a downgrade.

**D4d — `get_uniform_hash` gains the pass name, or two nodes share one row's config.**
`UIDocumentState.ui_uniforms` is keyed by `get_uniform_hash`, which hashes
`name_arraylength_dimension_gltype` — **the pass is not in the key**. Invisible today because exactly
one pass's rows are drawn per frame; D4 draws every visible node's rows at once, and `UIUniform`
carries `input_type`, which is user-set configuration.

So with `uniform vec3 u_tint` in both `blur` and `composite`, setting blur's row to `color` sets
composite's too, and the other node silently redraws as a swatch. The codebase already namespaced the
VALUES by pass for exactly this reason — `ui_models.py`'s save comment says a flat layout would have
two passes' `u_tex` overwrite each other — and left the UI config flat because nothing needed it.

The hash takes a pass name. THREE sites follow it, not one: `UIDocument.save`'s stale-row prune,
which enumerates live rows across passes; the row-building helper (D10a); and
**`ProjectSession.import_passes`**, which copies `source.ui_state.ui_uniforms` by key into the host
while `plan.renames` may rename the pass — under a pass-keyed hash every imported row would land
under the SOURCE's pass name and be dropped by the next save's prune.
`tests/test_pass_verbs.py::test_a_merged_ui_row_survives_the_save` pins that behaviour and goes red;
`tests/test_uniform_row_pruning.py` also calls `get_uniform_hash`. Both join the `tests/` row.

**The data fix reaches the SHIPPED examples, not only the dev sandbox.** Six
`shaderbox/resources/document_examples/*/document.json` files carry 100 `ui_uniforms` rows between
them (28, 59, 9, 3, 1, 0) and five carry `uniform_sort_key`/`uniform_sort_desc`. `drop_unknown`
handles the sort fields; the hash re-key does NOT self-heal — a stranded row is silently dropped by
`UIDocument.save`'s prune and re-created at defaults, losing every `input_type` the example ships
with. Each row is re-keyed by hand under the hash the new function produces for its declaring pass;
where the owning pass is genuinely ambiguous the row is DELETED, since a regenerated default is a
cosmetic loss and a wrong key is a silent one.

The check that catches a bad re-key is a row COUNT comparison before and after a load+save, not
"it loaded" — the prune deletes what it cannot match, which makes a wrong key look like a clean
save.

**D4b — The rows join the overlap chain explicitly, and they do NOT enter `node_size`.** Two
mechanics that decide whether this works at all:

- **Submission order, and the rows must not overlap a port at all.** `set_next_item_allow_overlap()`
  goes on the item submitted FIRST, so the canvas is a chain: the background declares it, then each
  node body, then the ports come last. The ports deliberately declare NOTHING — the code comment says
  why ("the ports must keep declaring nothing or the drop target dies"). A row submitted before a
  port and declaring overlap therefore loses to that port wherever the two intersect: the row's drag
  would be dead there, not the port's. So the rows are laid out CLEAR of the port column rather than
  relying on the chain to arbitrate. Order: background → node body → rows → ports → output dots, with
  the rows' rects disjoint from the ports'.

- **The rows are submitted OUTSIDE the channel split.** `_draw_canvas` wraps the node drawing in
  `dl.channels_split(...) … channels_merge()`, and a real imgui widget emits into
  `get_window_draw_list()` — the same draw list that is split. A `drag_float` submitted while a
  channel is current lands in that channel and is re-ordered by the merge, which tears the widget
  (frame in one channel, text in another). The node PICTURES are draw-list calls and belong inside
  the split; the rows are widgets and are submitted after `channels_merge()`, in the same pass as
  the existing hit-test `invisible_button`s, which is where every other imgui item on this canvas
  already lives.
- **The rows draw OUTSIDE the node's canvas-space box.** `node_size` is the single source of truth
  for the rank layout, `App.arrange_graph`, `_bbox`/`_fit` and the hit rects — and `_port_point` /
  `_out_point` re-derive port positions from the same constants independently rather than reading
  `node.size`. If rows entered `node_size`, three sites would have to agree, and under D5's LOD
  ladder the box would become ZOOM-DEPENDENT, which is D7's own prohibition applied to ordinary
  nodes. The box is what the graph lays out; the rows are what it shows.

**D4c — The sort control returns as an icon on the node's card, and the key lives on `App`.** The
maintainer's call: rather than deleting the Uniforms tab's sort with the tab, a small sort glyph on
the card cycles the same `UniformSortKey` values, so ordering a long uniform list survives.
Declaration order (`"code"`) stays the default — the driver's own order is not it, and
`sort_uniform_hashes` is what produces it.

**The key is `App.uniform_sort_key` / `uniform_sort_desc`, not `GraphViewState`.** D18 says the sort
becomes per-SESSION rather than per-document, and `GraphViewState` is per-document (and dropped by
`App.forget_render_state`), so putting it there would make it per-document again under a decision
that says otherwise. On `App` it is one ordering for every node in the session, which is what a
glyph cycling three values should mean, and it persists nowhere — matching D18's deletion of the
persisted pair.

**D6 — Uniform rows scroll inside the node; 8–10 visible.** The ceiling is on the DRAWN extent, not
on `node_size` — D4b keeps the rows out of the layout box, so the box never grows and there was never
a wall for it to become. What the scroll prevents is a twenty-uniform pass painting over its
neighbours, which `Arrange` cannot see and cannot route around. The scroll offset lives on the
node's view state (per pass, per document), so it survives a pan, a zoom and a scope change — it
belongs to the node, not to the camera.

**The wheel needs a disambiguation rule, because it is unconditionally the zoom today.** The canvas
child sets `no_scroll_with_mouse` and `_draw_canvas` reads `io.mouse_wheel` gated only on the child
being hovered — no per-item check. Gating the zoom on "not over a row rect" leaks through an open
node menu (`is_mouse_hovering_rect` ignores popup blocking), and a per-node `begin_child` collides
with the shared draw list the channel split depends on. So: **wheel zooms, Shift+wheel scrolls the
hovered node's rows.** One modifier, no draw-list change.

One thing that does NOT come free: `view.hovered_node` is written from the node's own
`invisible_button`, sized `node_size` — and D4b puts the rows outside that box, so `hovered_node` is
None exactly when the pointer is on a row. The rows therefore publish their own hover through
`row_rects` (D19's second seam, which check 5(b) needs anyway): the scroll target is the node whose
row block contains the pointer, read from the same rects the tests aim through. No second hit item,
so the ports' "declare nothing" rule is untouched.

**D7 — A focused node grows in SCREEN units; its canvas footprint never changes.** Growing it in
canvas units would make the graph's stored layout different after leaving: `Arrange` would see a
giant node and wires would route around a box that is not really there. Focus is a *view* state, so
nothing about it survives leaving it. The node's `pos` and `size` on the pass entry are not written
by entering or leaving a mode.

### Render and Share

**D8 — Render and Share are modes of a node, entered from its context menu.** `Render…` and
`Share…` are items on the node's existing context menu (`pass_menu_items`). There are no mode chips
on the node: a focused node already says which mode it is in by what it is showing. Esc, or a click
on the scrim, leaves.

**D8a — Esc needs a new job, or it will not close the focus mode.** `App.escape_has_job`
(`App.escape_has_job`) returns True only for an open popup, the palette, a focused editor or a focused chat,
and `App`'s glfw key callback SWALLOWS the key at the GLFW layer when it returns False — imgui never sees the
press. A focused node is none of the four, so the exit the maintainer named would silently not work.

Two edits, both small and both load-bearing: a fifth clause on `escape_has_job` (a node is focused),
and a branch in `hotkeys._handle_escape` placed in precedence order — after the modal and the
palette, since a confirm opened from a focused node's menu must answer Esc first.

**And a third case the precedence does not cover: an imgui popup.** `App.any_popup_open` is
`self.modal is not None` and knows nothing about imgui popups, but the node context menu, the
documents dropdown and the group prompt are all imgui popups. With one of those open over a focused
node, `escape_has_job` is True via the new clause, the modal branch is False, and the focus branch
fires — while imgui independently closes the popup on the same press. One Esc dismisses two things.
The focus branch is therefore gated on `not imgui.is_any_popup_open()` as well, so a popup answers
Esc first and the focus survives to answer the next one.

**D9 — Entering a mode focuses the node: centre, grow, dim.** The view animates (or cuts — Open
question) to centre that node; the node draws at a fixed screen size large enough for the mode's
body; a scrim dims the rest of the canvas.

**D9c — The focused node takes a size PER MODE, not one constant.** The three bodies differ too
much for one number: Render is a file row plus size/length/fps, Share is an exporter panel with a
preview and its own controls, Uniforms is N full rows at `UNIFORM_CTRL_W` (320) plus the auto block.
One constant sized for the largest wastes most of the panel on the smallest. Three tokens, each
measured against its own body, all capped by the panel's content region — a focused node never
exceeds the region it is drawn in, whatever the token says.

**D9b — Entering focus WRITES the camera directly and cancels any in-flight gesture.** `_fit` cannot
serve this: it clamps `zoom = min(1.0, …)` so it never zooms in, and it frames every node rather than
one. So entry writes `pan`/`zoom` itself, after saving the previous three fields — `pan`, `zoom` and
`fitted`, the third because `_fit` re-runs on it and a restored camera with `fitted` cleared would be
re-framed on the next frame.

Entry also cancels `node_drag`, `wire_drag` and `band_anchor`. A drag surviving into focus would run
its release branch behind the scrim and commit a pass position from a gesture the user can no longer
see — and the position written would be some other node's, so D7's footprint check would not catch
it.

Focus is dropped when the scope changes. A focused pass outside the open group has no node under the
scrim, and `revalidated_scope` and the focus revalidation are otherwise independent.

**D9a — The focused pass is revalidated every frame, like every other name the view holds.** The
canvas already does this for its siblings: `view.selection &= set(document.passes)`
(`pass_graph.draw`), `revalidated_scope`, `revalidated_wire`. The focused pass gets the same
treatment, because `App.delete_pass` touches no graph view and `_on_pass_renamed` (`App._on_pass_renamed`)
rewrites editor tabs and the settings target but nothing in `graph_views`. Without it, deleting or
renaming the focused pass strands the view: a scrim over a canvas with no node to click and — before
D8a — no working Esc. A rename is reachable from the focused node's own `Pass settings…`.

Focus is also dropped when the document switches. The view is keyed per document so it would
otherwise survive, restoring a scrim on return to a document the user left in Render mode.

**D10 — The scrim is ONE rect on its own draw channel, and it eats clicks.** Not an alpha pass over
every node and wire: that would be N draws and would leak through overlapping shapes. The canvas
already splits its draw list into ordered channels (`_CH_HALO`, `_CH_WIRE`, `_CH_NODE`,
`_CH_INFLIGHT`, `_CH_OVERLAY`); the scrim is a new channel between the graph and the focused node,
filled at `COLOR.GRAPH_DIM_ALPHA` — the token already exists.

**The scrim leaves on RELEASE, not on press, and sets `press_blocked`.** A press-time exit leaves the
button still down on the next frame with focus cleared, where the freshly-hovered background can
start a rubber band from a click the user made on a scrim. The canvas's existing latch is the fix and
the precedent: set `view.press_blocked` on the exiting press, which clears only once the button comes
up.

Eating clicks is what makes it read as modal rather than decorative: without it, a click lands on a
dimmed node behind the scrim and two things are being edited at once. While a node is focused the
canvas refuses every background gesture (pan, rubber band, node press, wire drag), and a click on
the scrim leaves the mode. This is a refusal in `_draw_canvas`'s gesture branches, gated on the
focus state — the same shape as the existing `frozen` (copilot-turn) refusal, which is the
precedent to follow.

**Two mechanics the first draft understated.** `dl.channels_split(5)` (`pass_graph._draw_canvas`) is a
hard-coded count: adding the scrim means 6 and renumbering every constant above it, and an
off-by-one paints the scrim OVER the focused node. And refusing canvas GESTURES does not stop the
uniform rows on the DIMMED nodes behind the scrim — those are real imgui widgets, not
`invisible_button` hit-tests, so a drag on one still moves a value. Dimmed nodes must draw their
rows non-interactively (or not at all) while a node is focused; the gesture refusal alone is not
enough, and the verification below asserts the widget, not the gesture.

**D11 — The Render and Share bodies are REUSED, not reimplemented.** `tabs/render.py::draw` and
`tabs/share.py::draw` (plus `widgets/details.py`, `tabs/share_state.py`, the exporter panels and the
`RenderControl` seam) are the bodies. What may change is their *layout* — `tabs/render.py` places a
preview column beside a controls column, which is wrong for a node-shaped box and becomes one
column. No control, no setting and no behaviour is redesigned; the exporter contract in
`exporters/base.py` is untouched.

The focused node's picture is the pass's own live target, which is what the Render body's preview
column was showing — so the preview is the node's picture, drawn once, not twice.

**D12 — Render targets the node's pass, by threading the `target` parameter that already exists.**
`Document.render` ALREADY takes `target: str | None`, resolved against `graph.output_pass`, and its
docstring states the intent: *"`target` draws that pass and its ancestor chain instead of the graph
output's… The graph output still decides which pass keeps full size."* So per-pass render is a
parameter threaded through four existing signatures — `render_media` → `_render_media_into` →
`_render_image` / `_render_video`, each of which already calls `self.render(...)`.

**But `target` and `canvas` do not compose today, and that is the one core change this feature
makes.** `Document.render`'s blit is gated on the GRAPH OUTPUT, not on the resolved target:

```python
resolved = target if target is not None else self.graph.output_pass
output = self.graph.output_pass
...
if canvas is not None and name == output:
    self._blit_into(render_pass.canvas, canvas)
```

So `render(canvas=scratch, target="blur")` draws blur's chain into the passes' own canvases and
blits NOTHING into the export scratch — a black file. `target` was built for the live preview, which
passes no canvas; the export path always passes one. The fix is one condition — `name == resolved` —
and it is correct for both callers, since `resolved` IS the output when `target` is None. But it is
an edit to the core render loop, so `document.py` carries it explicitly and check 8 is what proves
it.

**A third site is gated on the output and needs the same treatment:** `_render_video` calls
`self.render_pass.restart_video_uniforms()`, and `render_pass` IS the output. Exporting a non-output
target would restart the OUTPUT's video uniforms and leave the target pass's own — and its
ancestors' — mid-playback, so the export would start from wherever the live preview left the clip.
Non-deterministic, and invisible in a still frame. The restart follows the target's chain.

**The export renders the target pass at ITS OWN size.** `render_media` sizes its scratch from
`self.render_pass.canvas`'s dtype/filter/wrap and `export_source_size()` — both OUTPUT properties,
while a non-output pass renders at `canvas_size_for(name)` (its `scale` applied). The rule already
has one home, `Document.canvas_size_for`, whose docstring says the output keeps full size and every
other pass takes its scale. Rendering a scaled pass at the output's size would either letterbox or
silently upscale, so the export reads `canvas_size_for(target)` and the target pass's own canvas
properties. A target that IS the output resolves to exactly today's behaviour.

**No output mutation, and explicitly NOT a temporary output switch.** An earlier draft proposed
switching the output inside `export_isolation()`; that is wrong on three counts and is recorded here
so it is not re-proposed. Things read the output mid-encode (`_render_video` calls
`render_pass.restart_video_uniforms()`; `render_media` sizes its scratch from
`render_pass.canvas`'s dtype/filter/wrap). The natural writer, `Document.set_output_pass`, resizes
by design — it calls `conform_canvases()` and resamples the promoted pass to full document size, so
a switch-and-restore reallocates every canvas twice per export and destroys the off-output pass's
scale. And it would collide with the canvas-ownership decision, which exists precisely because
output-role changes carry a resize obligation. `target=` has none of these properties: it mutates
nothing.

A pass that IS the output renders exactly what today's Render tab rendered.

**D12a — Share stays DOCUMENT-scoped.** Not symmetric with Render, deliberately.
`tabs/share_state.py`'s `TabState.outlets` is keyed by exporter id alone — one `OutletRenderState`
for the whole app, holding the current artifact and its freshness. Focusing Share on node A,
rendering, then focusing node B and clicking Publish would publish A's artifact while
`artifact_is_fresh` read True. Making Share per-pass means re-keying that state AND changing the
exporter ABC (`exporters/base.py::draw_target_panel` takes a `UIDocument`), which contradicts D11's
"the exporter contract is untouched".

Sharing a document's OUTPUT is also what sharing means — a sticker pack wants the finished frame,
not an intermediate pass. So `Share…` on any node shares the document's output, and the menu item
reads `Share document…` so the target is named rather than assumed. `render_job.render_for` is
unchanged.

**D10a — Somebody must still build `ui_uniforms`, and the deleted tab was the only site.**
`tabs/uniforms.py`'s loop is what calls `UIUniform.from_uniform` for a hash not yet seen and
`snap_input_type()` on every row — two writes to persisted document state, and the only place they
happen. Delete the tab without rehoming that loop and a freshly-compiled pass has no `UIUniform`
entries at all, so the node draws no rows and the focused mode draws none either.

The loop moves to a pure helper beside the row drawing, taking the pass explicitly (D4a's rule), and
both the node's compact rows and the focused node's full rows call it. It also owns the
`sort_uniform_hashes` call, which is what produces declaration order (D18), and the auto/active
split that routes auto uniforms to `_draw_auto_block`.

**D11a — The reused bodies are bracketed, because they run inside a split draw list.**
`tabs/share.py::draw` already wraps its whole body in a try/except; `tabs/render.py::draw` does not.
Hosted on a node, both run between `channels_split` and `channels_merge` inside the canvas child, so
an exception unwinds past both and leaves the draw list and the imgui window stack unbalanced —
surfacing as an unrelated assert at some later `end_child`. The host brackets each body in a
try/except, and the merge sits in a `finally`.

`widgets/details.py`'s file picker is a main-thread spin loop (`pfd_block` busy-waits until the
dialog is ready), so it blocks inside that split. Blocking is tolerable — it is what the picker does
today — but it is another path that can raise, which is what the bracket is for.

**D11b — A deferred render captures the PASS, and drops if it vanished.** The defer is a two-frame
latch that fires after the swap, so two frames elapse between the click and the encode. There are
exactly two `render_defer.submit` sites, and the one that becomes per-pass is
**`tabs/render.py::_draw_render_button::_run_render`** — Share's (`tabs/share.py::_do_render`) stays
document-scoped per D12a and has no pass to capture. Both already capture `document_id` rather than
the object, with a comment saying why: a delete or project switch in between releases that
document's GL program. A per-pass
render adds a second thing that can vanish, and the copilot can delete passes. So the closure
captures the pass NAME too and re-resolves both at fire time, dropping the request when either is
gone.

**D11c — Share's outlet state stays shared, and the focus does not own it.** `share.update(app)` runs
every frame outside any focus state, and `_draw_outlet`'s closure captures the outlet by reference
from the frame the button was clicked. Leaving focus therefore does not cancel an in-flight render or
publish, and must not: the operation belongs to the document's outlet, which is what D12a makes it.
The focused node is a host for the panel, never its owner.

### What moves off the tabs

**D13 — The canvas control moves into the fps chip's menu, verbatim.** `tabs/document.py`'s
`canvas_choice_groups` / `canvas_choice_label` / `_apply_canvas_choice` move to the toolbelt
unchanged. There is **no `Auto | Fixed` mode row**: 093 deleted that toggle because a mode whose two
positions edited different things (a ratio, a pixel pair) put a control on the row that changed
meaning under it. What exists is one list grouped by aspect, each group's first row `Auto`, the
fixed sizes of that ratio under it.

The toolbelt also carries the channel view (`Color` / `Alpha` / `RGB`), which vacates the rendering
canvas's top-left corner. `CommandId.CYCLE_CHANNEL_VIEW` keeps its chord, so the menu is a fallback
rather than the only path.

There is no separate toolbelt chip: the fps chip IS the opener (D13a). Target fps, throttle on/off
and the GPU budget stay in the Settings modal — they are global, not per-canvas, and this feature
does not move them.

**D13c — The canvas READOUT survives, as the menu's header.** `_draw_canvas_caption`
(`tabs/document.py`) shows the half of the fact the control does not carry: under Auto the
control names a RATIO so the caption shows the live pixel size; under Fixed it names a SIZE so the
caption shows the aspect. Deleting the Document tab would take it, and under Auto it is the ONLY
place the actual pixel count appears — which D1a makes more load-bearing, not less, since dragging
the splitter changes that number continuously.

It becomes the first row of the fps chip's menu, above the aspect groups: `Canvas 1920x1080` under
Auto, `Canvas 16:9` under Fixed. Same function, same two branches, new host.

**D13a — ONE chip, reading the CURRENT DOCUMENT's fps, and it opens the canvas menu.** Not
`58 | doc 30`, not a separate gear beside it, not an eye or any second glyph: a single pill reading
`N FPS`. Clicking it opens the menu that carries the channel view and the resolution picker (D13).
So the canvas has exactly one affordance.

The number is the DOCUMENT's rate, never the UI's. `_current_document_fps` (`ui.py::_current_document_fps`) returns
`None` when the document renders every frame, and today the chip falls back to the UI fps in that
case — which is what produced the two-number label. With one number the fallback must be a number
too, or the chip blanks on the common case: when the plan's interval is 1 the document renders
every frame, so its rate is the frame rate.

**It must be the MEASURED rate, not the target.** `RenderPlan.document_fps` holds
`target_fps / interval`, and `target_fps` round-trips to `app_state.global_target_fps` — the
SETTING. So reading the plan at interval 1 would make the chip read `60 FPS` on a machine actually
rendering at 30, which is the opposite of what a frame-rate chip is for. Today's two-number label
hid this because it fell back to `app.global_fps`, the measured value. So: at interval 1 the chip
reads `round(app.global_fps)`; above 1 it reads the plan's `document_fps`, which is the rate the
throttle is deliberately holding it to. `_current_document_fps` is rewritten to always return an
int, and its `| None` return goes.

**D13b — The frame profiler's PANEL is deleted; the profiler itself keeps measuring.** The panel is
`fps_overlay`'s unfolding half plus `profile_rows_plan`, `_profile_rows`, `_measured_row`,
`_document_row`, `_plan_tree`, `SIZE.FPS_PANEL_W`, `App.fps_details_open` and `App.profile_smoother`.

What must NOT be deleted: `shaderbox/profiling.py` and every `profiler=` thread through the render
path. `_refresh_document_costs` (`ui.py::_refresh_document_costs`) reads `app.last_profile`'s `document:<id>` spans and
is the ONE write site for the costs `plan_render_set` throttles on (090 D7) — so deleting the
measurement would silently disable the throttle, with no test naming it. The chip loses its panel;
the engine keeps its instrumentation.

*Deferred, with its trigger:* the profiler gets a new home ON the graph — frame times visualised over
the nodes that spend them. *Trigger:* a feature is opened for it; explicitly NOT this one.

**D14 — The breadcrumb's first crumb carries the document's whole verb set.** With the grid gone,
`document_menu_items` has no host — and six `CommandId.DOCUMENT` verbs lose their only surface:
`OPEN_SCRIPT` (Alt+R), `OPEN_DOCUMENT_DIR`, `TOGGLE_DOCUMENT_PLAY`, `RESET_DOCUMENT`,
`DELETE_DOCUMENT`, `NEW_DOCUMENT` (Ctrl+N). The crumb names the document, so it owns them: a LEFT
click opens the documents dropdown (D16), a RIGHT click opens that document's menu — the existing
`document_menu_items`, which already takes a `document_id` rather than acting on the current
document, plus a rename (the field the Document tab held).

Each row of the dropdown carries the same right-click menu, so a verb reaches any document, not only
the open one — which is exactly what the grid's tiles did.

**D14a — Every document HAS a script; it is just OFF by default.** "Does this document have a
script?" is internal bookkeeping and stops being a question the user is asked. The chip is therefore
unconditional and has exactly two states — running or stopped — with no third "absent" state and no
special first-click behaviour.

The fiction is one step from what the core already believes. `create_script`
(`ProjectSession.create_script`) writes a generated stub built from the document's own scriptable
uniforms, and the copilot's `read_script` already answers a missing script with that stub rather
than an error (`ProjectSession.read_script_source`, returning `(text, is_stub)`). So "always there, off"
describes the model; only the UI was exposing the disk state.

| Chip | Click | Play/stop glyph |
|---|---|---|
| `script` | opens the script tab, creating the file on first touch | stopped (dim) or running (accent) |

Two consequences to implement rather than discover:

- **`session.has_script` stops being a UI gate — at THREE sites, not one.** It remains a disk fact
  the engine and the copilot read, but no chip, glyph or menu item is conditioned on it. The sites:
  `tabs/document.py::_draw_script_toggle` (file deleted anyway);
  **`App.toggle_current_document_play`**, whose `if not has_script: return` makes
  `TOGGLE_DOCUMENT_PLAY` a silent no-op on a document whose stub was never written — exactly the
  third state D14a says does not exist, and its own comment names the coupling it preserved ("matching
  the button, which only renders for a present script") to a button this feature deletes; and
  `popups/import_passes.py::_has_script`, which must be checked and left alone or changed
  deliberately. `App.open_script_for`'s lazy-create gate is correct and stays. The Document tab's old
  `if not app.session.has_script(document_id): return` (`tabs/document.py::_draw_script_toggle`) has no successor.
- **The play/stop is meaningful before anything is on disk.** `all_stopped` is `UIDocumentState`,
  not a property of the file, so a document whose stub has never been written can still be
  "stopped" — and starting it creates the stub the same way clicking the chip does. The toggle
  therefore calls `create_script` on the same lazy path `open_script_for` uses, or the first play
  runs nothing and silently does so.

The click is `open_script_for(document_id, focus_editor=True)`, unchanged — it already creates
lazily (`App.open_script_for`). The PLAY path must refuse the same way: `open_script_for` returns
early during a copilot turn, and so does `App.set_document_all_stopped`, so a first play mid-turn
must refuse outright rather than create the stub and then fail to start it. Both affordances are frozen mid-copilot-turn, which `open_script_for`
refuses internally (`App.open_script_for`); the chip's own bracket must match, or it looks live while
refusing.

**D15 — `Render all documents` is deleted, and the throttle is left alone.** It was the switch that
kept the grid's thumbnails alive. `plan_render_set` already gives `current` the budget first at
`k = ceil(cost / (budget × period))` and hands the others only the remainder at one common fps — a
proportional answer to the same question, with a hard on/off sitting in front of it.

Deleted: `app_state.is_render_all_documents`, its checkbox, the `stale` branch that read it, and its
branch in `ui.py`'s render-set computation. The render set becomes: **the current document, plus at
most one document awaiting its first render, plus the documents the open dropdown is showing.**

**D19 — The feature publishes four OBSERVATION SEAMS, because four of its checks name values
nothing can read.** A gate that cannot be broken is not a gate, and `conventions.md` says a new one
is done only when the guarded thing has been broken and the gate named it. These four are not test
scaffolding — each is a value the app already computes and then discards:

- **`App.planned_documents`** — today a local in `_tick_frame_state`. Check 3 asserts the dropdown
  ADDS to the planned set rather than replacing it, and cannot see the list. Published as a field
  written once per frame, beside `App.render_plan`, which is already published for exactly this
  reason.
- **`GraphViewState.row_rects`** — where each node's uniform rows landed on screen, keyed
  `(pass, uniform)`. The view already publishes `port_rects`, `out_rects` and `canvas_rect` with the
  stated reason "so a headless test can aim a drop where a user would"; the rows are the fourth such
  surface and check 5(b) — the scrim eating WIDGET input, not just gestures — needs it.
- **`App.viewer_region` is the splitter's observable.** `box_height` is a local and stays one;
  check 16 drops its drag half (unobservable) and keeps the persistence round-trip, while check 17
  owns the drag through `viewer_region`, which is the value that actually reaches the renderer.
- **The Esc seam.** `escape_has_job()` is directly testable; "the press clears it" is not, since no
  test in this repo drives `imgui.is_key_pressed`. Check 12 splits into two: the clause, and
  `hotkeys._handle_escape`'s branch called directly.

**D16a — A dropdown row carries the two states the grid tile carried.** The grid's tile showed a
compile error (`STATE_ERROR` border) and a staleness wash for a document whose picture is a
photograph of the past (`document_grid.draw_document_preview_grid`). Both survive on the row: an error tints the
row's name, and a document whose `first_render_done` is False — or which has not yet had its turn
since the dropdown opened — draws its thumbnail dim. The second is what makes the open dropdown
honest while the previews fill in one first-render per frame.

`is_render_all_documents` is gone from that condition (D15): a row is stale iff it has not rendered
since the dropdown opened, which is a fact about the frame, not about a setting.

**D16 — The documents selector is a dropdown on the breadcrumb, with live previews rendered only
while it is open.** 48×27 (16:9) thumbnails, one per document, letterboxed like every other preview
cell. A document in the list renders while the list is open and not otherwise.

**The open state must be an App field, not imgui's.** `_tick_frame_state` — the whole render-set
computation, `plan_render_set` and every `document.render()` call — runs BEFORE
`imgui.new_frame()`. So nothing in the planning path can read an imgui popup's open state, and the
first draft's "the dropdown's documents join `planned_documents`" is not expressible as written. The
Examples popup is the precedent for the SET, not for the trigger: it keys off `app.modal`, an App
field set by a command handler outside the frame.

So the dropdown sets `App.documents_dropdown_open` during its draw, and the NEXT frame's planning
reads it. The consequence is accepted and stated rather than discovered: the set is one frame stale
in both directions — the frame the dropdown opens renders no previews, and the frame it closes
renders one extra set. A thumbnail appears on the second frame, which is the same one-frame lag the
throttle already has everywhere else.

**The documents join BOTH lists.** `tick_documents` and `planned_documents` are different lists with
different consumers: the render loop and `_rendering_this_frame` iterate `tick_documents`, while
`plan_render_set` and the `begin_frame` advance take `planned_documents`. Joining only the latter
would give a dropdown document an interval and a feedback-history advance with no render — a
feedback pass advancing without drawing into it.

The precedent is the Examples popup, which plans its own set while up and admits one first render
per frame so opening it never stalls on compiling everything at once. **One difference, and it is
the point:** Examples *replaces* the normal set (`planned_set_mode`); this dropdown **adds** to it,
because the rendering canvas above stays live while the maintainer picks. So the dropdown's
documents join `planned_documents` while `current_planned` stays the current document — which is
exactly the priority asked for: the current document keeps its rate, the thumbnails share the
remainder.

**D17 — The pass strip is deleted as a surface; `pass_menu_items` is kept.** `widgets/pass_list.py`
exists to draw tiles the graph now draws as nodes. Its `pass_menu_items` is already shared with the
graph node (092 D10, so the two surfaces cannot drift) and moves to a module the node imports.
`Add pass` and `Import passes` are already on the canvas's own context menu (`_canvas_menu`).

**D18 — Deleted commands and state.** `FOCUS_TAB_DOCUMENT` / `_UNIFORMS` / `_RENDER` / `_SHARE`
(they focus tabs that no longer exist) and `OPEN_GRAPH`. `ui_regions.DocumentTab`,
`ui_state.active_document_tab` — which has THREE sites, not one: the model field, the live `App`
attribute restored in `_init` (so it runs on every project switch) and its mirror-back in
`App.save` — `App.document_tab_select_pending`, `App.focus_document_tab`, and
`ui_state.uniform_sort_key` / `uniform_sort_desc` (the sort combo was the Uniforms tab's).

**NOT deleted** (reversing the first draft): `ui_state.panel_pass` and `App.panel_pass` survive as
the chord-time pass resolver — see D4a.

**Declaration order is not free.** `get_active_uniforms()` yields GL's driver-defined order;
`sort_uniform_hashes(..., "code", False)` (`tabs/uniforms.py::draw`) is what PRODUCES declaration
order. The node keeps calling it — with the key D4c's card icon cycles, defaulting to `"code"` —
so rows never land in driver order. What goes is the persisted `uniform_sort_key` /
`uniform_sort_desc` pair and the Uniforms tab's combo; the ordering itself survives, and the sort
becomes per-session rather than per-document.

**D18a — Alt+G's per-document use is preserved.** `OPEN_GRAPH` is deleted as a command, but
`document_grid.py`'s menu used it for ANOTHER document's graph ("open that document's graph"), which
the dropdown's row-selection does not offer. Selecting a document in the dropdown IS showing its
graph, since the graph is always on screen for the current document — so the verb collapses into the
selection rather than being lost.

`ui_regions.py` keeps `ChannelView` and loses `DocumentTab`. Its module docstring, which explains
why `DocumentTab` lives outside the imgui-evaluating command table, is rewritten for what remains.

---

## What the surfaces look like

### The graph's overlays

| Surface | Position | What |
|---|---|---|
| rendering canvas | top-right | **ONE chip**, `N FPS` — the current document's rate. Click opens the menu carrying the channel view and the resolution picker (D13a). Nothing else is on the canvas. |
| graph | top-left | the breadcrumb: `document ▾ / group`, then the **script chip** immediately after the first crumb |
| graph | top-right | nothing |
| graph | bottom-right | nothing — kept clear for the "Rendering…" cue and notifications |

**D14b — The script chip sits on the graph canvas, beside the breadcrumb.** The maintainer named the
surface ("on the graph view canvas somewhere: a little script icon with play/stop"), and the reason
he named it is that clicking the icon must OPEN the script, not only toggle it — an affordance a
context menu cannot be. It goes beside the first crumb because both answer questions about the same
object: the crumb says which document, the chip says whether its script runs.

The chip carries both verbs (D14a): the label opens the script tab, the glyph toggles play/stop. It
is drawn after the crumb and before any scope crumbs, and it degrades to the glyph alone when the
crumb row is tight — the breadcrumb grows a crumb per open group, so the row has a variable budget.

The play/stop ALSO stays on the document's right-click menu (D14), because a menu is where a verb is
discoverable by convention and where its chord hint lives. Two routes to one verb is the existing
pattern for every document verb, not a duplication.

The breadcrumb today is `_tab_row`, which answers *scope* (root → group). Adding the documents
dropdown to the first crumb keeps that: the crumb still switches scope when the graph is inside a
group, and its caret opens the documents list. Scope tabs after the first are unchanged.

### The node's uniform rows

A uniform row on a node is the value control alone. This is the part of the feature with the most
existing surface behind it, and the spec must be explicit: `draw_ui_uniform` handles **seven** input
types (`UIUniformInputType`'s members), several of which cannot shrink to a node row.
`SIZE.UNIFORM_CTRL_W` is 320 and `SIZE.GRAPH_NODE_W` is 240 after D4e — a full row still does not
fit, which is why the table below is a width decision and not a taste one.

| Input type | On the node | Why |
|---|---|---|
| `drag` (1–4 dim) | a compact drag control per component, value shown | the common case; this is what tuning means |
| `color` (3–4 dim) | a swatch that opens the picker | `color_edit` already collapses to a swatch |
| `auto` | **no row** | identical on every pass (`u_time`, `u_resolution`); the Uniforms tab already segregated these into their own block |
| `texture` | **no row** | the sampler is already a PORT on the node, and its source is the wire; drawing a thumbnail would say it twice |
| `buffer` | a name and size, no control | its only control is `Randomize`, which is not a tuning act |
| `array` | name, `(n/cap)`, no control | the control is a comma-separated text field; it does not fit |
| `text` | name, `(n/cap)`, no control | the control is a 72px multiline box |
| a driven uniform | its play/stop glyph where the value sits | the script state is watched while tuning |

The four rows with no control are **not hidden** — they are drawn dim, showing name and state, so
the node tells the truth about what the pass declares.

**The auto uniforms keep a home.** `auto` gets no node row, but the Uniforms tab's `_draw_auto_block`
(`tabs/uniforms.py::_draw_auto_block`) shows every engine-driven uniform with its LIVE value, and those hashes are
routed AROUND `draw_ui_uniform` (`tabs/uniforms.py::draw`) — so "reuse the tab's body" does not carry it
unless `_draw_auto_block` moves too. It moves, into the focused node's uniforms mode. (The first
draft also claimed auto values are "identical on every pass"; that is wrong —
`format_auto_value` reads the PASS's own slot, `tabs/uniforms.py::_draw_auto_block`.)

**The uniform-name jump and hover bridge survive.** `_locate_uniform_declaration`
(`widgets/uniform.py::_locate_uniform_declaration`) makes a uniform's name clickable — it jumps to the declaration — and
hovering marks the line in the editor (`app.editor_hover_line`, `app.code_hovered_uniform`). It is
a capability on every row today and the first draft neither preserved nor named it. It lives on the
focused node's full rows, where the name column exists; a compact node row has no name column wide
enough to click, so the bridge is not on the node itself. Their controls are reached by focusing the
node (below).

**A focused node shows the FULL uniform rows** — `draw_ui_uniform` verbatim, at
`SIZE.UNIFORM_CTRL_W`, with the input-type chip, the texture combos and previews, the multiline text
box, the `Randomize` button. This is the third mode, and it is what makes the compact rows
acceptable: everything the Uniforms tab could do is one right-click away, on the pass it belongs to.
So the node has a third focus mode, **uniforms** — but it is not a peer of Render and Share and must
not be presented as one. The maintainer named uniforms among the buttons no longer needed and asked
for Render and Share on the context menu. This mode exists because the full rows had to go
SOMEWHERE: it is the only home for the input-type chip, the texture source combo with its previews
and video filters, the multiline text box, `Randomize`, the auto block and the jump/hover bridge.
Framed correctly: the node shows what fits, and `Uniforms…` opens what does not.

### The node's context menu

The existing `pass_menu_items`, plus the three modes:

```
Open shader              Ctrl+E
Pass settings…           Alt+P
Group…
─────────────────────
Uniforms…
Render…
Share…
─────────────────────
Leave group                     (only when grouped)
─────────────────────
Delete                          (disabled when it is the last pass)
```

`Set as output` is **not** added — choosing the output is a click on a pass's picture elsewhere
today (`App.choose_output`), and this feature does not change that verb.

### The toolbelt menu

```
16:9
  Auto                    ●
  Wide 720p
  Wide 1080p
1:1
  Auto
  512x512
  1024x1024
…one group per aspect the document can take…
─────────────────────
VIEW
  Color                   ●
  Alpha
  RGB                     Ctrl+Shift+C
```

The aspect groups and their rows are whatever `canvas_choice_groups` returns for the document —
including a group per media-bound texture's aspect, which is how a document adopts the shape of an
image it samples. The `VIEW` rows are `ChannelView`'s three members.

### The breadcrumb's dropdown

```
DOCUMENTS
  [thumb] RC 2            ●
  [thumb] RC 1
  [thumb] blur test
─────────────────────
New document             Ctrl+N
Projects…                Alt+O
```

Right-clicking a row opens that document's menu (`document_menu_items` + rename + the script
play/stop), which is where D14 sends the Document tab's remains.

---

**D20 — Three ordering constraints, stated as sequence rather than as hazards.** Each is a window in
which the app is broken in a way no test names:

1. **D4a before D4, and D4d before D4.** Draw the rows before the target is an argument and every
   node shows — and WRITES — the panel pass's values; draw two passes' rows before the hash carries
   the pass name and they share one `input_type`. Both look right, so nothing surfaces either.
2. **D2a before D3.** Delete the graph tab while both `begin_disabled` brackets still stand and the
   graph is permanently on screen AND fully frozen for every copilot turn, with D16 having made the
   breadcrumb the only document switcher.
3. **D16 before D15.** Delete `is_render_all_documents` before the dropdown exists and no non-current
   document renders at all, with the grid already gone.

## Files touched

| File | Change |
|---|---|
| `shaderbox/ui.py` | a horizontal `_draw_canvas_splitter` mirroring `_draw_splitter`; `box_height` read from the fraction, not `VIEWER_BOX_ASPECT`; `ViewerGeometry`'s docstring rewritten; `_draw_app_panel`'s lower half becomes one graph child; the render-set branch loses `is_render_all_documents`; the dropdown's documents join `planned_documents`; `_draw_document_settings` and `_NODE_TABS` deleted; the channel chip moves to the toolbelt |
| `shaderbox/widgets/pass_graph.py` | uniform rows on the node; the LOD ladder; the focus mode (scrim channel, camera save/restore, screen-unit node, gesture refusal); `_node_menu` gains three items; `_tab_row` gains the documents dropdown |
| `shaderbox/widgets/graph_state.py` | `GraphViewState` gains: focused pass + mode, the saved camera (`pan`/`zoom`/`fitted`), per-pass uniform scroll offsets, and **`row_rects`** (D19's seam, read by the row hover and by check 5(b)) |
| `shaderbox/widgets/uniform.py` | a compact node row per input type, beside the existing full row |
| `shaderbox/widgets/document_grid.py` | grid → dropdown rows; `document_menu_items` kept |
| `shaderbox/tabs/document.py` | `canvas_choice_*` moves to the toolbelt; the rest deleted |
| `shaderbox/tabs/uniforms.py` | deleted (its body becomes the focused node's uniforms mode) |
| `shaderbox/tabs/render.py` | body re-laid-out as one column; hosted by the focused node |
| `shaderbox/tabs/share.py` | body re-laid-out as one column; hosted by the focused node |
| `shaderbox/tabs/code.py` | `_draw_graph_tab`, the `"graph"` tab kind and its title branch deleted |
| `shaderbox/widgets/pass_list.py` | deleted; `pass_menu_items` moves out |
| `shaderbox/ui_regions.py` | `DocumentTab` deleted; docstring rewritten |
| `shaderbox/commands.py` | `FOCUS_TAB_*` ×4 and `OPEN_GRAPH` deleted |
| `shaderbox/ui_models.py` | `UIAppState`: `is_render_all_documents`, `active_document_tab` deleted, `canvas_split_fraction` added. `UIDocumentState`: `uniform_sort_key`/`uniform_sort_desc` deleted — and `_reset_out_of_range_values`'s clause reading them, plus `UniformSortKey` if it goes dead. `panel_pass` KEPT (D4a). |
| `shaderbox/app.py` | `open_graph_for`, `document_tab_select_pending` deleted; `escape_has_job` gains a clause; `panel_pass` KEPT (D4a) |
| `shaderbox/theme.py` | `GRAPH_NODE_W` 136→240 and `GRAPH_THUMB` 116→220 (D4e), with the width/inset assertion re-checked; `GRAPH_BOX_EXTRA_W` shrunk to keep a group box's inset proportional; `GRAPH_GAP_X`/`GRAPH_GAP_Y` unchanged (nothing derives them from the node width); a `GRAPH_LOD_*` threshold pair; three focused-node sizes (D9c); `PANEL_CTRL_MINH` deleted |
| `shaderbox/ui_models.py` (splitter) | `canvas_split_fraction` added with its `ge`/`le` constraint |
| `shaderbox/document.py` | `render_media` gains a target pass (D12) |
| `shaderbox/render_job.py` | unchanged — Share stays document-scoped (D12a) |
| `shaderbox/hotkeys.py` | `_handle_escape` gains the focus-mode branch (D8a) |
| `scripts/smoke.py` | rewritten off `DocumentTab` / `open_graph_for` (part of `make gates`) |
| `tests/` | `test_region_system_is_gone`, `test_pass_strip_layout`, `test_uniforms_tab`, `test_graph_tab`, `test_menus`, `test_pass_verbs`, `test_pass_navigation`, `test_graph_view`, `test_canvas_presets`, `test_canvas_fields`, **`test_theme`** (four `profile_rows_plan` sites — its throttle-band suite; decide at implementation time whether the bands survive independently of the deleted row builder, and say so in the commit), **`test_default_wiring`** (a `draw_ui_uniform` call site, D4a), **`test_graph_tab`** (its width pin, stale under D4e), **`test_pass_verbs`** and **`test_uniform_row_pruning`** (the `get_uniform_hash` re-key, D4d), **`test_render_decoupling_loop`** (sets `is_render_all_documents`), **`test_model_salvage`** (uses `active_document_tab` as its retired-enum fixture — it needs a different retired key, since the point of the fixture is a key no field claims) |
| `shaderbox/ui_primitives.py` | `profile_rows_plan`, `_profile_rows`, `_measured_row`, `_document_row`, `_plan_tree` deleted; `fps_overlay` loses its unfolding half and its `profile`/`plan`/`titles`/`budget`/`is_open` parameters (D13a/D13b) |
| `shaderbox/menus.py` | `menu_enabled`'s EDITOR predicate revisited — a graph tab used to satisfy `app.active_tab is not None` (check 15) |
| `ai_docs/conventions.md` | THREE amendments: (1) the pass-resolution decision — drop "the uniforms panel" from its resolving surfaces, add the by-argument rule; (2) the item-set-function decision — `pass_menu_items`' new home, AND its `document_grid.document_menu_items` (a grid tile)` clause, since D16 turns the tile into a dropdown row; (3) **the viewer-box decision** — D1a replaces its height derivation, its `PANEL_CTRL_MINH` cap and its "only the splitter moves that boundary" clause, and adds a second writer of the size every Auto document renders at. Its "the graph view lives in the editor pane" clause also goes (D3). |
| `shaderbox/copilot/` | three user-facing strings naming "the Share tab" and one naming "the pass strip" reworded |
| `shaderbox/resources/document_examples/**` | SIX `document.json` files: `ui_uniforms` re-keyed per D4d (100 rows), `uniform_sort_key`/`uniform_sort_desc` dropped from five |
| `projects/dev/**` | THREE files, hand-edited (no migration code — hard rule): `app_state.json` drops `is_render_all_documents` and `active_document_tab`; both `documents/*/document.json` drop `uniform_sort_key` and `uniform_sort_desc` from their `ui_state`. (An earlier note in the review log said the document files were clean; they are not.) |

---

## How correctness is decided

Every check below is one the repo can run; none asks the maintainer to go and look.

1. **The deleted surfaces are gone, and nothing reads them.** The grep is scoped, or it cannot be
   evaluated: over `shaderbox/`, `tests/`, `scripts/` and `dogfood/` only — `ai_docs/` keeps prose
   hits by design. Symbols: `DocumentTab`, `is_render_all_documents`, `active_document_tab`,
   `OPEN_GRAPH`, `FOCUS_TAB_`, `pass_list`. (`panel_pass` is NOT in this list — D4a keeps it.)
   *Falsifier:* reintroduce one reader; the check names it.

2. **The render set no longer contains a non-current document at rest.** With the dropdown closed
   and two documents open, `tick_documents` is the current document alone (plus a pending first
   render). *Falsifier:* restore the `is_render_all_documents` branch; the assert fires. The test must seed a
   SECOND document and warm it to `first_render_done`, or the restored branch admits nothing and the
   falsifier is inert against the one-document fixture default.

3. **The dropdown ADDS to the planned set rather than replacing it.** With the dropdown open,
   `App.planned_documents` (D19's first seam) contains the current document AND the listed ones. *Falsifier:* make the
   dropdown replace the set (the Examples shape); the current document is absent and the assert
   fires.

   The first draft also asserted "the current document's interval is unchanged" as the priority
   guarantee. That clause is **defective and removed**: `current_interval` is computed from
   `current_cost` alone (`plan_render_set`), never from `others`, so it is identical under both
   shapes and the assert could not fail for the reason its falsifier named. The priority guarantee
   is structural — `current` is passed as `current_planned` — and is covered by this check's
   membership assert plus the existing `plan_render_set` suite.

4. **A focused node's canvas footprint is unchanged.** Enter render mode on a node, leave, assert
   the pass entry's stored position and size are byte-identical. *Falsifier:* grow the node in
   canvas units; the assert fires.

5. **A focused node's scrim blocks BOTH gestures and widgets.** Two separate families, because
   refusing canvas gestures does not reach imgui items:
   (a) per gesture — pan, rubber band, node press, wire drag — one assert each, since a single
   "gestures are refused" test can pass for four reasons and verifies none;
   (b) a uniform drag on a DIMMED node behind the scrim does not change its value — aimed through
   `GraphViewState.row_rects` (D19's second seam), the way the existing gesture tests aim through
   `port_rects`.
   *Falsifier (a):* remove the focus gate from one branch; that gesture's assert fires.
   *Falsifier (b):* draw the dimmed nodes' rows interactively; the value moves and the assert fires.

6. **The camera and the scope are restored on leave.** Enter focus from a known pan/zoom/scope,
   leave, assert all three are back — `fitted` included, since `_fit` re-runs on it. *Falsifier:*
   drop the restore; the assert fires.

7. **Every input type has a node row, enumerated from the TYPE.** The domain is
   `get_args(UIUniformInputType)` — **seven** members (`UIUniformInputType`), read the way
   `ui_models.py`'s input-type validator already reads it. NOT `valid_input_types()`, which is an instance method
   returning the types valid for ONE uniform (at most two) and would silently cover two of seven —
   the exact checker-narrows-its-own-domain failure this item exists to prevent. The driven
   play/stop is a separate assert: it is a STATE orthogonal to the input type, not an eighth member.
   *Falsifier:* add a member to the Literal with no node row; the test names it.

8. **Render on a non-output node renders THAT pass, and writes no document state.** Focus render
   mode on a non-output pass, encode to a temp path, and assert the written frame matches that
   pass's target rather than the document's output. Then assert `document.graph.output` is
   BYTE-IDENTICAL to what it was before — not "restored", never written. D12 threads `target=`
   precisely so there is nothing to restore; an implementation that switches the output and puts it
   back would pass a restore assert while violating the decision.
   *Falsifier (target):* call `render_media` without the target; the frames match the output and
   the pixel compare fires. The assert compares PIXELS, not a path or a size — a size assert passes
   for the wrong pass whenever two passes share a canvas.
   *Falsifier (no-write):* implement it as a switch-and-restore; a spy on `Document.set_output_pass`
   fires and the check goes red. The spy is what makes this falsifiable — a correct restore is
   indistinguishable from no write by value alone.

9. **Share targets the document's output, and its outlet state is not confused by focus.** Focus
   Share on node A, render, focus node B, assert the artifact and its freshness flag still describe
   the DOCUMENT's output (D12a) rather than either node. *Falsifier:* key the outlet per pass
   without re-keying `TabState.outlets`; the flag reads fresh for the wrong artifact.

10. **A deferred render captures the pass, not just the document.** Submit a render from a focused
    node, delete that pass before the deferred fire, assert the render is dropped rather than
    encoding a released target. *Falsifier:* capture only `document_id`, as
    `tabs/render.py::_draw_render_button::_run_render` does today; the encode runs against a freed
    pass.

11. **The focused pass is revalidated.** Delete the focused pass and assert the focus clears;
    rename it and assert the focus follows or clears rather than stranding; switch documents and
    assert the focus clears. *Falsifier:* omit the revalidation; a scrim survives over a canvas
    with no node.

12. **Esc leaves the focus mode — two separate asserts.** (a) `escape_has_job()` is True while a
    node is focused. (b) `hotkeys._handle_escape`, called directly with the clause present, clears
    the focus. Split because with the clause omitted both halves fail for the SAME reason, which is
    the one-reason rule. *Falsifier:* omit the `escape_has_job` clause; the key is swallowed
    at the GLFW layer (`App`'s glfw key callback) and the assert fires. This is the check for a requirement the
    maintainer named explicitly and that nothing else would catch.

13. **The graph does not join the editor's focus domain.** With a shader tab OPEN (so the flag is
    not trivially False), draw the graph and assert `app.editor_focused` still reflects the editor
    and `editor_errors` is not cleared. *Falsifier:* keep `_draw_graph_tab`'s bookkeeping in the new
    host; the assert fires. The "with a tab open" precondition is what makes this falsifiable —
    `editor_focused` is written False on the no-tab path too (`tabs/code.py::draw`), so the naive
    version passes whenever no shader is open.

14. **The breadcrumb stays live during a copilot turn.** `begin_disabled` writes no App field, so
    the observable is the CONSEQUENCE: with `copilot_turn_active` True, a synthetic click on a
    dropdown row changes `app.current_document_id`, while a synthetic press on a node writes no
    position. *Falsifier:* leave `begin_disabled` wrapping the whole widget; the dropdown is
    dead and the assert fires.

15. **`menu_enabled` names its intended value, not today's.** With no shader tab open, assert
    EDITOR-scoped items are DISABLED — the intended post-feature behaviour, stated as a value rather
    than as a comparison against a baseline this feature deliberately moves. The blast radius is
    three commands (`FORMAT_BUFFER`, `JUMP_NEXT_ERROR`, `CLOSE_CODE_TAB`), not the whole bar: every
    pass and document command is `GLOBAL` and unaffected — a graph tab used to satisfy
    `app.active_tab is not None` (`menus.menu_enabled`). *Falsifier:* delete the graph tab kind without
    revisiting the predicate; every editor menu item greys in a fresh session.

16. **The split fraction round-trips.** `app_state.canvas_split_fraction` survives a save/load.
    *Falsifier:* omit the field from the model; the reload reverts to the default. The drag's EFFECT
    is check 17's, not this one's — `box_height` is a local and nothing can read it, so an assert
    naming it could not fail.

17. **The splitter's drag reaches the Auto render size.** With a document in Auto mode, drag the
    splitter and assert `app.viewer_region` — and the document's resolved canvas size next frame —
    followed it. *Falsifier:* assign `viewer_region` from the old derived box; the size does not
    move.

18. **The fps chip always shows a number.** With the document unthrottled (interval 1), assert the
    chip's label is a number rather than blank. *Falsifier:* keep `_current_document_fps`'s
    `None` return; the unthrottled case — the common one — draws an empty chip.

19. **The profiler keeps measuring after its panel is deleted.** Assert `app.last_profile` carries
    `document:<id>` spans and that `app.document_costs` is non-empty after a frame. *Falsifier:*
    delete `profiling.py`'s threading with the panel; the costs empty, `plan_render_set` throttles
    on nothing, and the assert fires. This guards the one deletion that would disable the throttle
    silently.

20. **Smoke covers the new panel.** The old smoke asserted a `DocumentTab` member drew every frame;
    its successor asserts the graph child drew and that a synthetic click on a node selected it, so
    the gate still exercises the lower panel rather than silently covering nothing. The rewrite of
    the files that read deleted symbols is work, not a check — it lives in the Files-touched table.
    *Falsifier:* stub the graph's draw to return early; smoke goes red.

21. **Two nodes' rows write to their OWN passes.** D4a is the feature's largest change and nothing
    checked it. The failure it names — "a node would silently show and WRITE another pass's values" —
    is invisible on screen, since both nodes draw plausible rows. Draw two nodes' rows in one frame,
    write through node B's row, assert pass A's `uniform_values` is unchanged and B's moved.
    *Falsifier:* leave `draw_ui_uniform` resolving through `panel_pass`; both rows write the same
    pass and A's value moves.

22. **`node_size` is invariant across zoom and across a pass gaining uniforms.** D4b keeps the rows
    out of the layout box; if they enter it, the box becomes zoom-dependent and wires land wrong at
    some zooms — which reads as imprecision, not as a bug. `node_size` is importable and pure.
    *Falsifier:* add the row block's height to it; the invariance assert fires.

23. **A raise inside a reused body does not unbalance the frame.** D11a brackets the Render and Share
    bodies because they run between `channels_split` and `channels_merge`; an escape leaves both the
    draw list and the window stack unbalanced, and surfaces as an unrelated assert at some later
    `end_child`. Patch `tabs/render.py::draw` to raise, drive a frame with a node focused in render
    mode, assert the frame completes. *Falsifier:* drop the bracket; the frame raises.

24. **`TOGGLE_DOCUMENT_PLAY` works on a document with no script on disk.** D14a's central claim is
    that `has_script` stops being a UI gate; the named failure is a SILENT no-op. Assert the toggle
    flips `all_stopped` and creates the stub. *Falsifier:* keep
    `App.toggle_current_document_play`'s `if not has_script: return`; the state does not move.

25. **A dropdown row is stale iff it has not rendered since the dropdown opened.** D16a removes
    `is_render_all_documents` from that condition. *Falsifier:* wire the dim to `first_render_done`
    alone; a document that has rendered once but not since the dropdown opened reads fresh, and the
    "honest while previews fill in" property is lost with no visible symptom.

26. **`make gates` green**, judged by exit code captured unpiped.


---

## Open questions for the user

1. **The two LOD thresholds** — RESOLVED at implementation time, and the proposal was wrong.
   Keyed on ZOOM (the draft's 0.75 / 0.4) the rows would never appear for anything but a short
   chain: measured after D4e, a 3-pass chain fits at zoom 1.00, a 6-pass at **0.50** and a
   10-pass at **0.30**, because `_fit` clamps at 1.0 and never zooms in. What decides a row is
   whether its control is readable, which is `GRAPH_NODE_W × zoom` — so the thresholds are
   SCREEN-WIDTH tokens: `GRAPH_LOD_ROWS_PX = 200` (a vec4 component gets ~38px there, the width
   of "0.000") and `GRAPH_LOD_NAME_PX = 96`. A 6-pass chain then reads as `mid` rather than
   losing its rows silently.

2. **Does entering focus animate the camera or cut to it?** A cut is less code and may read as a
   jump. Proposal: cut for the first implementation, add easing only if it reads badly.

3. **Does the focused node's uniforms mode replace the compact rows, or sit above them?** Proposal:
   replace — the focused node shows the full rows and nothing else, so there is one uniform list on
   screen, not two spellings of the same list.

4. **Does the uniform sort control's loss matter?** D18 keeps declaration order and deletes the
   user-facing sort (name / type). It is a capability the maintainer did not ask to lose, and a
   pass with many uniforms is exactly where sorting helped. Proposal: ship without it and restore a
   sort on the focused node's full rows if it is missed.

---

## Review history

(To be filled by the pre-implementation review.)
