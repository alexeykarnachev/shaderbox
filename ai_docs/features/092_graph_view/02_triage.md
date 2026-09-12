# 092 — triage of the brainstorm review round

Five opus reviewers, one angle each, anchored to `01_brainstorm.md` and `00_mock.html` round 3,
every claim probed against the code (`reviews/brainstorm_*.md`). This file is what survived:
the claims re-verified by hand, the constraints the code settles, the decisions that need the
maintainer, and the trails not to walk again. It does not re-open anything in `01_brainstorm.md`
"Fixed by the maintainer".

## Verified by hand after the reports

- `Document.graph_errors` has no reader outside `document.py`: a cycle is invisible in the app
  today. The strip's grey wash (`or {output}` fallback) is the only symptom, unexplained.
- `PASS_STUB` declares no `sampler2D`: a pass added on the canvas has no input port until its
  shader declares one. A port exists because a sampler is declared in the shader text.
- Every pass verb on `ProjectSession` ends in `save_ui_document`; nothing in the widgets writes
  `graph.json` from a draw.
- `_GROUP_PATTERN` and `PASS_NAME_RE` are the same pattern: no slash, and a pass and a group may
  share a name.

## Corrections to the brainstorm record

- `extra='forbid'` is NOT on `PassGraph` / `PassEntry`. What keeps `graph.json` loadable is
  `document.load_graph`'s per-entry `drop_unknown` / `drop_invalid`. A new field on `PassEntry`
  is tested through `load_graph`, never through the generic `load_model` (which reads every pass
  NAME as an unknown key and prunes `passes` to `{}`).
- "A read of a pass that does not exist" is not a `GraphError` the app can produce: `wired_pass`
  drops it before the planner sees it. The only planner error is a cycle (culprit + victims).
- "The first sight of a document runs the rank layout once and stores it" would be a draw-time
  write, which the save-funnel and lazy-row conventions forbid. Replaced below (S3).

## Settled by the code and the reports (constraints for the spec, no decision needed)

- **S1. Ports come from the compiled program, edges from the wiring.** `effective_wiring` drops
  unfilled samplers, so the port list is `sampler_names(pass)` and the edge list is the wiring.
  Consequence: a never-compiled pass has no ports; the canvas compiles what it draws (bounded:
  the largest document is six passes) or admits one per frame like the first-render sweep.
  Consequence: a port for a sampler the program no longer declares is never drawn, which is what
  stops `set_sampler_source` (which does not validate the uniform name) writing a dead row.
- **S2. No wrong connection exists.** Every port is a `sampler2D`, every source is one of the
  document's own passes, dtype / scale / filter / wrap are invisible to a consumer (probed on a
  real context). The one refusal is a cycle, decided by `plan_passes` over the hypothetical
  wiring before any write; refuse on `errors != []`, since the culprit need not be an endpoint.
- **S3. Positions: `PassEntry.position: tuple[float, float] | None = None`, canvas space,
  bounded (no NaN / inf, a magnitude clamp through one funnel).** `None` means never placed; the
  pure `rank_layout` in `pass_graph.py` places the `None` ones every frame. A position is written
  only by a drag (once, on release) or by Arrange (once for the set). Every creator without a
  cursor (the copilot's `add_pass`, `import_passes`, the strip's `add pass`) leaves `None`. An
  import strips the source's positions. The box has no position: it is its members' bounding
  box, and dragging it translates the members.
- **S4. The box's picture:** the document output when it is a member, else the member read from
  outside, else the last unread member in strip order. The box's output ports: every member read
  from outside, plus the bundle output always (a hollow dot when nothing reads it). This is the
  resolution of the one rule break the mutations review found (deleting the output member left a
  box with a picture and no dot) and of the prose-vs-mock disagreement (N ports vs one).
- **S5. The graph adds no write path.** Click = `pick_pass`; gear / menu = `open_pass_settings`;
  wire drop = `set_sampler_source` (overwrites without asking, as the combo does); Group / Leave /
  Dissolve = `set_pass_group`; delete = `delete_pass` + `close_editor_for_path`. Multi-pass verbs
  write once for the whole set (a batched write or a save-suppressing scope), never once per pass.
- **S6. The canvas is `invisible_button` + `is_item_active` + `io.mouse_delta` + `io.mouse_wheel`
  + one draw list**, probed end to end on this stack (22 scenarios, PASS). Rules the probe paid
  for: `set_next_item_allow_overlap()` goes on the item submitted FIRST (canvas background, then
  node body, then ports / overlays); `begin_popup_context_item(None)`, never an explicit id, on a
  shared canvas; `WindowFlags_.no_scroll_with_mouse` on the child; `channels_split` for wires
  under nodes; the port hit radius floors at a screen-pixel size while the drawn dot scales;
  `push_font(font, 14 * zoom)` is crisp at any zoom (1.92 bakes a face per integer size).
- **S7. The copilot sees nothing new.** No tool accepts or reports a position; the pass table
  stays flat (a box has no address); no `group_passes` tool; no groups paragraph in the prompt.
- **S8. `graph.json`'s `version` does not bump** for an additive defaulted field (091 added
  `group` without a bump; nothing reads the stamp).
- **S9. The headless side needs only the pure half:** `rank_layout` and the boundary-port
  computation in `pass_graph.py`, testable with two dicts and no window.
- **S10. `Dissolve` is required, not leaning:** grouping is N saved writes with no undo, and
  Dissolve is its inverse. A box gets no Delete verb (Dissolve plus per-node delete with the
  strip's two-click arm). `delete_pass` and `import_passes` remain the two one-way verbs.
- **S11. `set_pass_group` and `rename_pass` reject a pass/group name collision**: one namespace,
  since the root keys nodes and boxes by name.
- **S12. `rename_pass` plans the post-rename wiring and rejects a cycle**, the same guard the
  drag has; a rename can create a name-rule edge today with no check.
- **S13. A drop onto a port that is name-wired materializes an explicit `PassSource`** (free: the
  drop writes one anyway). No visual distinction between name-rule and explicit edges in the
  first landing.
- **S14. The canvas freezes under `copilot_turn_active`** like the strip; the draw list still
  paints the live pictures.
- **S15. The strip and the import dialog are untouched, except one line in the dialog** saying
  the source's own groups are flattened (091 drops inner labels silently today).

## Decisions for the maintainer

**D1. Box ports: one per sampler slot, or merged by what feeds them?** (round 3 A vs B)
- (a) One port per slot, labelled by the sampler, `member.sampler` when not unique. Honest; a
  drop rewrites exactly one sampler; three `scene` wires into the bloom box.
- (b) Merged by source (`scene ×3`), unfilled slots merged by sampler name. Reads as a black
  box; a drop rewrites every slot behind it, so a legal drop on one slot is refused when a
  sibling slot would cycle, and the port hides which; two members cannot be wired to different
  sources without entering the group.
- (c) Merged for display, expanded on drop (the port opens into its slots when a wire hovers).
Recommendation: (a) for the first landing; (c) if the cascades case (three members reading
`paint`) proves noisy in use.

**D2. Where the graph lives.**
- (a) A `strip | graph` toggle on the Document tab's Passes row, the root/group tabs as a
  `text_tab_row` inside the canvas, the canvas written pane-agnostic. Zero edits to `EditorTab`,
  the tab bar, `is_tab_dirty`, `close_tab`, `code.draw`.
- (b) A third `EditorTab.kind = "graph"` in the editor pane's tab bar. `path` is the tab's
  identity in five places and a group is not a `Path`; `Ctrl+W` is editor-scoped; `Ctrl+Tab`
  would cycle onto a bufferless tab.
Recommendation: (a) now; (b) when the zen mode / pane swap arrives, by moving the widget.

**D3. Nesting.**
- (a) Stay flat (091's deferral stands). Reparenting a box into a group dissolves it into the
  outer label, and the dialog's new line says so on import.
- (b) Path labels (`post/bloom`), graph-only nesting. Breaks the import prefix (a slash in a
  pass filename), collides `group_tint('post')` with `group_tint('post/bloom')`, needs the three
  group validators collapsed into one, and makes a parent box with no direct members exist.
Recommendation: (a). No scenario in the walk needed (b).

**D4. Pan versus rubber-band on empty canvas** (one left-drag today does both).
- (a) Left-drag = rubber band, middle-drag = pan (Nuke, Blender).
- (b) Left-drag = pan, Shift+left-drag = rubber band.
- (c) Left-drag = rubber band, Alt+drag = pan.
Recommendation: (a), with wheel zoom about the cursor and a Fit verb.

**D5. Dragging from an input port.**
- (a) Grab and carry: pressing a filled port detaches its wire; release on another port moves
  it, release on empty writes `NoSource` on the original.
- (b) Input ports are drop targets only; Disconnect on the port's menu.
- (c) Both.
Recommendation: (a); a drop on empty already means `NoSource`.

**D6. Keyboard on the canvas.** Escape has an owner ladder (`_handle_escape` + `escape_has_job`)
with no graph rung; a bare Delete cannot be a registry chord (`chord_needs_modifier`).
- (a) No new chords: the tab row is the way up, delete is on the context menu (the strip's
  shape), `Alt+Left/Right` keep stepping the output pass.
- (b) An Escape rung ("up one tab, else clear selection") plus `Alt+Delete` as a registry
  command in the document scope.
Recommendation: (a) for the read-only landing, (b) decided before drag-and-wire lands.

**D7. A media-bound sampler's port.** `sampler_names` lists it, the wiring does not. A wire
dropped on it would release the bound texture.
- (a) A port in a distinct "media" state (a small thumbnail dot); a drop on it is refused.
- (b) A port in the media state; a drop overwrites, as the combo does.
- (c) No port, matching the strip's chips.
Recommendation: (a).

**D8. A single-pass document.** Five of six shipped examples and both sandbox documents.
- (a) The graph toggle is always available and never the default.
- (b) Hidden until the document has two passes.
Recommendation: (a).

**D9. A group split across the DAG** (the same label on disconnected passes): one box at the
bounding box, enclosing non-members visually; Arrange pulls members together. Or one box per
connected run (the strip's answer, which breaks "a group is one box"). Recommendation: one box.

**D10. Does a cycle get a surface outside the graph?** The graph is the first place a cycle is
ever explained. Either that is the answer, or the strip gets a one-word badge in the same wave.
Recommendation: the graph only, in 092; the strip badge is its own small fix if it is wanted.

**D11. Staging.** (1) The read-only canvas: nodes, boxes, ghosts, tabs, click / double-click /
menu, fit, arrange, the error cues. (2) Drag, wire, rubber-band, Group / Dissolve. (3) Nothing
else is queued; nesting, duplicate-pass, save-group-as-document, the presets folder stay
deferred with their triggers.

## The error language (from the wiring review, adopted)

Shape carries port state, one hue carries fault. Filled disc = wired; hollow ring = unfilled
(black, not a fault); ring with a black centre = `NoSource`; double ring = `prev`; media state
per D7. `STATE_ERROR` on a node border = compile error (the strip's rule, output accent
overridden), with the stale tick on the picture since it is the last good frame. `STATE_ERROR`
on the edges of a cycle, never on victim nodes. Off-plan nodes dim with their edges. A
never-compiled node has a dashed border. A box carries `STATE_ERROR` when any member has one,
and nothing at all for non-convexity. No cue for dtype / scale / filter / wrap.

## False trails, consolidated (do not re-walk)

- Non-convex groups: handled by the boundary rule with no special case, probed.
- Feedback under rename, feedback into an iterated pass, a pass reading the document output:
  nothing new, existing machinery.
- Per-node `begin_child`, `begin_drag_drop_source` for wires, `push_clip_rect` on the canvas,
  `is_mouse_hovering_rect` for hover, integer zoom steps for crisp text: all wrong tools.
- A `group_passes` copilot tool, positions in the copilot's view, a `version` bump, a stored box
  position (it would make the group an entity, 091's revisit trigger), a nested strip outline.
- Compile-error propagation to consumers: none by design; the consumer binds the last good frame.
- The tutorial walked in the graph, the shader-lab workflow: neither changes the design.
