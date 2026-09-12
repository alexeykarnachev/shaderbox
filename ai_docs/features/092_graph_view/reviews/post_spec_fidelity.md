# 092 — post-implementation spec-fidelity audit

Auditor: spec-fidelity, post-implementation. Date: 2026-09-12.

## Coverage

Read end to end: `ai_docs/features/092_graph_view/03_spec.md` (all 557 lines: Goal, Out of scope,
D1–D20, Files touched, Manual verification, Review history).

Diff: `git show --stat` on both commits, then every changed file read end to end —
`shaderbox/pass_graph.py` (the whole 092 section plus `_cycle_message`/`PassEntry`),
`shaderbox/widgets/pass_graph.py` (all of it), `shaderbox/widgets/graph_state.py` (all of it),
`shaderbox/app.py` (the 092 hunks: `graph_views`, `graph_view_for`, `forget_render_state`,
`arrange_graph`, `drop_wire`, `unwire`, `commit_node_drag`, `group_selection`, `dissolve_group`),
`shaderbox/project_session.py` (the 092 hunks: `_pass_name_error`, `add_pass`, `rename_pass`,
`set_pass_positions`, `set_pass_group`/`set_pass_groups`, `import_passes`),
`shaderbox/document.py` (`wiring_if_renamed`), `shaderbox/pass_import.py` (`plan_import`),
`shaderbox/tabs/document.py` (`_draw_passes`), `shaderbox/widgets/pass_list.py`
(`pass_menu_items`, `draw`), `shaderbox/theme.py`, `shaderbox/ui_regions.py`,
`shaderbox/ui_models.py`, `shaderbox/help_content.py`, `shaderbox/popups/import_passes.py`,
`scripts/smoke.py` (the frames 43–48 stretch), and every test file the two commits touched
(`test_pass_graph.py`, `test_graph_state.py`, `test_graph_view.py`, `test_ui_regions.py`,
`test_graph_persistence.py`, `test_pass_verbs.py`, `test_theme.py`, `test_button_tiers.py`,
`test_ui_prose_budget.py`, `test_canvas_fields.py`) — test BODIES read, not names.

Also read to check claims rather than trust them: `ui_primitives.text_tab_row` /
`segmented_choice` / `fade`, `App.pick_pass`, `project_session.compile_pending_passes`,
`pass_list._delete_pass` and the tile's arm path, `ai_docs/conventions.md` (the "nothing folds"
entry), `ai_docs/dev_flow.md` (the module map), `ai_docs/roadmap.md` (the banner),
`.claude/skills/imgui-ui/SKILL.md`, `ai_docs/features/070_pass_reads/01_spec.md`, and
`shaderbox/copilot/**` (grepped for `position` / `graph` / `group`).

Not covered: the running app (no window manager here; every pixel-level clause is judged from the
draw code and is marked as such), and the `make gates` full run (the named test set was run — see
the count below).

Test run: `uv run python -m pytest tests/test_pass_graph.py tests/test_graph_state.py
tests/test_graph_view.py tests/test_ui_regions.py tests/test_graph_persistence.py -q
-p no:cacheprovider` → **88 passed in 1.40s**.

## D1 — ports from the program, edges from the wiring

| Clause | Verdict | Evidence |
|---|---|---|
| Input ports are `sampler_names(pass)` in the compiled program's order | LANDED | `pass_graph.node_ports` iterates `declared`; `widgets/pass_graph._build_view` passes `sampler_names(render_pass)` |
| Every port whose resolved source is the consumer itself moved to the END (any sampler, not only `u_prev`) | LANDED | `node_ports` collects `feedback` separately when `source == name`, returns `ports + feedback`; pinned by `test_node_ports_classify_every_state_and_put_feedback_last` |
| Edges are `Document.effective_wiring()` | LANDED | `_build_view` opens with `wiring = document.effective_wiring()`; `draw` computes the same |
| No port for a sampler the program no longer declares | LANDED | `node_ports` never reads `values` keys; `test_node_ports_come_from_the_program_never_the_stored_rows` |
| Pure `node_ports(declared, values, wiring_row, name) -> list[Port]`; the canvas only calls it | LANDED | signature matches the spec exactly; the widget's only construction site is in `_build_view` |
| The compile seam `project_session.compile_pending_passes(document)` | LANDED | imported and called in `widgets/pass_graph.draw` and in `App.arrange_graph` |
| Called **once on a document's first canvas frame** | DEVIATED | `draw` calls `compile_pending_passes(document)` unconditionally every frame. `GraphViewState.compiled` — the field the spec's own D2 list names for this ("the one-shot `fitted` flag" and, in the state file's comment, "The compile seam ran for this document (092 D1)") — exists in `graph_state.py` and is **read nowhere** (`grep -rn "\.compiled" shaderbox/` returns nothing). The docstring argues the per-frame call is "a no-op once every pass has been attempted", which is true for the loop body but still walks every pass and rebuilds a sorted list of failures per frame. Unrecorded. |

## D2 — where it lives

| Clause | Verdict | Evidence |
|---|---|---|
| `strip \| graph` `segmented_choice` on the Passes caption row | LANDED | `tabs/document._draw_passes` → `segmented_choice("##passes_view", …)` with `PASSES_VIEW_LABELS` |
| `UIAppState.passes_view: PassesView`, `STRIP` default, app-level preference | LANDED | `ui_models.UIAppState.passes_view`; `ui_regions.PassesView` / `PASSES_VIEW_LABELS` |
| `tabs/document.py` draws the caption, the toggle and the `add pass` / `import...` row and dispatches the body | LANDED | `_draw_passes` does exactly this and branches to `pass_graph.draw` / `pass_list.draw` |
| `pass_list.draw` loses its caption and its buttons, draws only tiles | LANDED | both hunks removed in the W1 diff; `small_caption` / `standard_button` dropped from its imports |
| `widgets/pass_graph.py::draw(app, document_id)` a leaf filling the child it is handed | DEVIATED | `draw` does not merely fill: it draws the tab row itself, then **computes its own child height** — `height = max(float(SIZE.GRAPH_MIN_H), avail.y - reserve)` where `reserve = imgui.get_frame_height() + 2 * SPACE.SM` reserves room for the add/import row the *sibling* (`tabs/document.py`) draws after it. That is a leaf encoding its sibling's height, which is what "never positions a sibling" was written to prevent; it is also the coupling the "graph as an `EditorTab.kind`" out-of-scope note says is avoided because the canvas is "written pane-agnostic". Unrecorded. |
| `GraphViewState` in `widgets/graph_state.py` (pan, zoom, scope, selection, in-flight drag, one-shot `fitted`) | LANDED | `graph_state.GraphViewState` carries all of them plus `guides`, `group_prompt`, `group_name`, `band_anchor`, `wire_drag`, `compiled` |
| Held in `App.graph_views: dict[str, GraphViewState]`, created on first use | LANDED | `App.graph_views` in `__init__`; `App.graph_view_for` creates on miss |
| Transient, not persisted | LANDED | nothing in `save_ui_document` / `UIAppState` names it |
| Evicted per document in `App.forget_render_state` | LANDED | `self.graph_views.pop(document_id, None)` beside `document_costs` |
| The `add pass` / `import...` row inside its OWN `begin_disabled(app.copilot_turn_active)` | LANDED | `_draw_passes` opens a second `begin_disabled(app.copilot_turn_active)` around the two `standard_button`s |
| The caption + toggle also bracketed | LANDED | a first `begin_disabled` wraps `small_caption` + `segmented_choice` |

## D3 — the scope and the tab row

| Clause | Verdict | Evidence |
|---|---|---|
| `GraphViewState.scope` is `""` or a group name | LANDED | `graph_state.GraphViewState.scope: str = ""` |
| `text_tab_row` at the canvas top: root label then every group in strip order of its first member | LANDED | `_tab_row` builds `[root_label, *groups]`; `groups` comes from `group_names_in_order(order, groups)`, and `order` is `strip_order` |
| Scope revalidated every frame; a name no pass carries falls back to `""` | LANDED | `draw` → `view.scope = revalidated_scope(view.scope, set(group_names))`; `graph_state.revalidated_scope`; smoke frames 46–47 assert it under the real loop |
| Double-click on a box enters it; the root tab is the way up | LANDED | `_double_click` sets `view.scope = node.group` for `kind == "box"`; no Escape handler anywhere (`grep` for graph in the key registry is empty) |
| A scope change refits once | LANDED | every scope write (`_tab_row`, `_double_click`, `_click` on a ghost, the box menu's `Open`) sets `view.fitted = False` |
| Root label = `ui_name` or `document` when empty | LANDED | `_tab_row`: `ui_document.ui_state.ui_name.strip() or _ROOT_FALLBACK_LABEL`, `_ROOT_FALLBACK_LABEL = "document"` |
| A click mapped back to a scope by INDEX, never by the returned string | DEVIATED | `_tab_row` does `index = labels.index(clicked)` — `list.index` on the RETURNED STRING, which is exactly the ambiguity the clause forbids. `text_tab_row` returns a name, so a document named `bloom` with a group `bloom` builds `labels = ["bloom", "bloom"]` and `labels.index("bloom")` always answers 0: the group tab becomes unclickable. The comment above the line claims the index mapping is in force. The fix is for `text_tab_row` to return the index (or for the row to be keyed by a disambiguated label). Unrecorded, and it is a live defect, not a wording slip. |

## D4 — what each scope shows

| Clause | Verdict | Evidence |
|---|---|---|
| Root: every ungrouped pass as a node, one box per group | LANDED | `_build_view`'s root branch skips `groups[name]`-carrying passes and emits one `_Node(kind="box")` per `group_names_in_order` |
| Group `g`: every member as a node | LANDED | `members = [name for name in order if groups[name] == scope]` |
| One ghost per outside pass that feeds a member or reads one | LANDED | `feeders` and `readers` sets in the scoped branch |
| A pass that does BOTH is drawn twice (feeder left, reader right) | LANDED | the two ghost loops use distinct keys `g:in:{name}` and `g:out:{name}`, so a pass in both sets yields two `_Node`s at `left - w - gap` and `right + gap` |
| Ghosts dimmed (`GRAPH_GHOST_ALPHA`) | LANDED | `_draw_node`: `alpha = COLOR.GRAPH_GHOST_ALPHA if node.kind == "ghost" else 1.0`, applied through `fade` to fill, border, name, ports, picture |
| Ghosts dashed | LANDED | `_draw_node`: `if node.kind == "ghost" or node.uncompiled: _dashed_rect(...)` |
| Ghosts keep their ports | LANDED | `_pass_node(..., ports[name], ...)` for both ghost loops |
| Ghosts carry no badge | LANDED | the `×N` badge is gated `node.runs > 1 and node.kind != "ghost"`; the `N passes` badge is `kind == "box"` only |
| Clicking a ghost sets the scope to `""` and selects the pass | LANDED | `_click`: `view.scope = ""`, `view.fitted = False`, `view.selection = {node.name}` |
| Groups never draw a tinted region | LANDED | no region fill; the tint is the box's own `add_rect_filled` at `GROUP_FILL_ALPHA` |

## D5 — a box's interface

| Clause | Verdict | Evidence |
|---|---|---|
| Inputs: every `(member, sampler)` whose source is outside the group or is unfilled, one port per slot | LANDED | `pass_graph.group_boundary` skips `kind == "wired" and source in inside`, and skips `prev`; everything else becomes a slot. `test_group_boundary_over_a_member_whose_sampler_reads_nothing` pins the "or unfilled" half |
| Labelled by the sampler name when unique across the box, `member.sampler` otherwise | LANDED | `counts[port.sampler] == 1` decides; `test_group_boundary_over_the_bloom_shape` (three `u_scene` → qualified) and `test_group_boundary_over_a_one_member_group` (unique → bare) |
| Outputs: every member read from outside, plus the bundle output ALWAYS | LANDED | `outputs = [m for m in members if m in read_outside or m == bundle]`; `test_a_terminal_box_still_has_its_bundle_output_port` |
| Drawn hollow when nothing outside reads it | LANDED | `_build_view` computes `read_outside` again for the box node and emits `outputs=((m, m not in read_outside), …)`; `_draw_node` draws `add_circle` for a hollow slot, `add_circle_filled` otherwise |
| `bundle_output`'s four branches in order | LANDED | `pass_graph.bundle_output` is document-output → first read-from-outside → last unread-by-members → last member; `test_bundle_output_follows_its_four_branches_in_order` walks all four and asserts totality |
| The box's picture is the bundle output's live texture | LANDED | `render_pass = document.passes[bundle]`; `texture_glo=render_pass.canvas.texture.glo` |
| The box's tint is `group_tint` | LANDED | `_draw_node`: `tint = group_tint(node.group) if node.kind == "box" else None` |
| Badge says `N passes` | LANDED | `label = f"{len(node.members)} passes"` |
| Clicking it picks the bundle output (`pick_pass`) | LANDED | `_click` → `app.pick_pass(document_id, node.bundle if node.kind == "box" else node.name, focus_editor=False)` |
| The accent border shows when the document output is a member | LANDED | `_draw_canvas`: `is_output = node.name == output if node.kind != "box" else output in node.members` |
| A split group is ONE box at the members' bounding box | DEVIATED (partial) | The box key is one per group name (`f"b:{group}"`), so a split group draws once — landed. Its position is `x = min(positions[m][0] …), y = min(positions[m][1] …)`, the bounding box's **top-left only**: its SIZE is `node_size(len(box_ports), True)`, a fixed node width, so the box does not span its members' bounding box and does not visually enclose the non-members between them. Manual item 32 asserts the enclosure ("encloses non-members visually"), which this geometry cannot produce. Whether the fixed-size box is the better picture is a design call; either way the spec's "at its members' bounding box" and manual item 32 do not describe the code. Unrecorded. |

## D6 — positions

| Clause | Verdict | Evidence |
|---|---|---|
| `PassEntry.position: tuple[GraphCoord, GraphCoord] \| None = None` | LANDED | `pass_graph.PassEntry.position` |
| `GraphCoord = Annotated[float, Field(allow_inf_nan=False, ge=-MAX, le=MAX)]`, `MAX_GRAPH_COORD = 100_000.0` | LANDED | both in `pass_graph`; `test_a_position_is_bounded_on_the_model` parametrizes NaN, ±inf, ±1e30 |
| `rank_layout` pure, returns a position for every name in `names` | LANDED | `pass_graph.rank_layout`; `test_rank_layout_places_only_the_names_asked_for` |
| Rank = longest path from a root over non-self edges, cycle members rank 0 | LANDED | `pass_graph.graph_ranks` (`1 + max(deps)`, `setdefault(name, 0)` for the unordered); `test_rank_layout_keeps_cycle_members` asserts `graph_ranks == {a:0,b:0,c:0}` |
| Within a rank: group members adjacent, then mean position of predecessors, then strip order | LANDED | `sort_key` returns `(anchor, bary, strip_index)`; `test_rank_layout_keeps_group_members_adjacent` |
| Columns left to right at `GRAPH_GAP_X`, rows at `GRAPH_GAP_Y`, each column centred on the tallest | LANDED | the `x += widest + gap_x` loop then the second centring pass over `tallest` |
| `sizes` is each node's width and height | LANDED | parameter present and read for both axes |
| `placed` read ONLY for the predecessor tiebreak | LANDED | `placed` appears only inside `sort_key`'s `ys` |
| `rank_layout`'s signature | DEVIATED (cosmetic) | The spec writes `rank_layout(wiring, names, groups, sizes, placed)`; the code takes two more required parameters, `gap_x` and `gap_y`. Defensible (the gaps come from `SIZE`, keeping `pass_graph` free of theme), and every call site passes `SIZE.GRAPH_GAP_X/Y`. Unrecorded. |
| The canvas calls it every frame with `names` = passes whose `position is None`, using the result for those alone | LANDED | `_positions`: `unplaced = [name for name in document.passes if name not in stored]`, returns `{**stored, **laid, **overrides}` |
| Arrange calls it with every pass of the DOCUMENT and `placed = {}` and writes the whole result | LANDED | `App.arrange_graph`: `list(document.passes)`, `{}`, then `set_pass_positions(document_id, laid)` |
| A position written by a drag (once, on release) or by Arrange, through `ProjectSession.set_pass_positions`, one save | LANDED | `set_pass_positions` is the single writer of `position`; its only callers are `App.arrange_graph` and `App.commit_node_drag`. `test_set_pass_positions_saves_once_for_the_whole_set`, `test_commit_node_drag_writes_once_and_only_the_moved_passes`, smoke frame 47 |
| Every creator leaves `None` (`add_pass`, the copilot's tool, `import_passes` strips) | LANDED | `add_pass` constructs `PassEntry()`; `import_passes` now `model_copy(update={"group": group, "position": None})`; `test_import_passes_leaves_every_position_none` |
| The box has no position; dragging it translates the members | LANDED | no `position` on a box node; `_drag_names` returns `list(node.members)` for a box |
| `graph.json`'s `version` does not bump | LANDED | `version` untouched in both diffs; `test_a_graph_round_trips_every_field` still writes `"version": 2` |
| `load_graph`'s per-entry salvage carries the field | LANDED | `test_a_corrupt_position_costs_that_position_and_nothing_else` proves `iterations`/`group` survive a rejected position |
| `with_positions` validates rather than `model_copy`s | LANDED | `PassGraph.with_positions` → `PassEntry.model_validate`; falsifier asserted in `test_a_legal_position_is_accepted_and_carried_by_the_funnel` |

## D7 — drawing

| Clause | Verdict | Evidence |
|---|---|---|
| One `begin_child("##pass_graph", borders, no_scrollbar \| no_scroll_with_mouse)` | LANDED | `draw`'s `imgui.begin_child` with exactly those flags |
| Of the tab's remaining height, at least `SIZE.GRAPH_MIN_H` | LANDED (with the D2 caveat above) | `height = max(float(SIZE.GRAPH_MIN_H), avail.y - reserve)` |
| One transform `screen = origin + (canvas - pan) * zoom` | LANDED | `_Xf.to_screen` / `to_canvas` |
| `channels_split(3)`: 0 wires, 1 nodes and ports, 2 the foreground (in-flight wire, rubber band) | DEVIATED | `_draw_canvas` calls `dl.channels_split(2)` and merges after the node pass. The in-flight wire, the rubber band and the snap guides are drawn AFTER `channels_merge()`, i.e. later in the same merged list, which paints them on top correctly — so the picture matches, the mechanism does not. Unrecorded. |
| Node: rounded rect, `BG_SURFACE` fill, `BORDER` line | LANDED | `_draw_node`'s `add_rect_filled(fade(COLOR.BG_SURFACE, alpha))` and the `BORDER` fallback in the border ladder |
| The live texture via `add_image_rounded` at `GRAPH_THUMB * zoom`, no second render | LANDED | `add_image_rounded(imgui.ImTextureRef(node.texture_glo), …)` from `_thumb_rect`, aspect-fit; no `render()` call in the widget |
| The name under the picture, `font_14_bold` when live, `FG_DORMANT` when off-plan, pushed at `size * zoom` | LANDED | `font = app.font_14_bold if not node.stale else app.font_14`; `name_color = … COLOR.FG_DORMANT if node.stale …`; `push_font(font, max(4.0, font.legacy_size * z))` |
| A `×N` badge on the picture when `iterations > 1` | LANDED | `if node.runs > 1 and node.kind != "ghost"`, label `f"x{node.runs}"` (ASCII `x`, not `×` — the prose/emoji-font constraint makes this the safe spelling; the manual items say `x12`/`x6` too) |
| One port row per input, `GRAPH_PORT_ROW * zoom` high, dot at the node's left edge, label in `font_12` | LANDED | `_port_point` steps by `SIZE.GRAPH_PORT_ROW` at `node.pos[0]`; the label loop runs under `push_font(app.font_12, …)` |
| The output dot at the picture's right mid | LANDED | `_out_point` returns `(pos.x + size.x, (y0+y1)/2)` for a single output; multi-output boxes step within the thumb's vertical span |
| Node width `GRAPH_NODE_W`, a box `GRAPH_NODE_W + GRAPH_BOX_EXTRA_W` | LANDED | `graph_state.node_size(port_count, box)`; `test_a_node_grows_one_row_per_port_and_a_box_is_wider` |
| A wire is `add_bezier_cubic` from the output dot to the port's dot, control points 45% of the horizontal distance (30 minimum) | LANDED | `_draw_wire`: `c = max(30.0 * z, dx * 0.45)` |
| A wire whose consumer sits left of its producer, or that spans more than one rank, rides a bus below the row (`GRAPH_BUS_STEP` per extra rank) | LANDED | `_draw_canvas` computes `backward = b[0] < a[0] + 24` and `bus = bottom + 16 + max(0, span-2)*GRAPH_BUS_STEP + (GRAPH_BUS_STEP if backward else 0)`, passed when `backward or span > 1`; `_draw_wire`'s bus branch draws enter-bezier → line → exit-bezier |
| Feedback is a small loop from the output dot over the node's top-right into the `prev` port | LANDED | `_draw_self_loop`, called for every `port.kind == "prev"`, control points at `node top - 12*z` |
| Fonts pushed at fractional sizes | LANDED | `push_font(font, max(4.0, font.legacy_size * z))` in both font pushes |
| Zoom clamped to `[0.25, 2.5]` | LANDED | `SIZE.GRAPH_ZOOM_MIN/MAX = 0.25/2.5`; clamped in both `_fit` and the wheel handler |
| `COLOR.GRAPH_EDGE` a fixed role joining `_GROUP_TINT_EXCLUSIONS` | LANDED | `theme.GRAPH_EDGE` added to the set beside `FAVS`; the module-level assert plus `test_group_tints_are_stable_and_collide_with_nothing` (with the stated falsifier) |

## D8 — hit testing

| Clause | Verdict | Evidence |
|---|---|---|
| One `invisible_button` covering the child, submitted FIRST, with `set_next_item_allow_overlap()` | LANDED | `_draw_canvas`: `set_cursor_screen_pos(origin)`, `set_next_item_allow_overlap()`, `invisible_button("##graph_bg", …)` |
| Then one `invisible_button` per node body, each with `set_next_item_allow_overlap()` | LANDED | the node loop calls `set_next_item_allow_overlap()` before each `##gnode_{key}` |
| Then (W2) one per port | DEVIATED | The port and output hit rects are submitted (`##gport_{key}_{slot}`, `##gout_{key}_{slot}`) but **without `set_next_item_allow_overlap()`**, and — worse — the node body button that precedes them in the same iteration does not re-declare overlap for them. The chain the spec describes ("a missed level makes the canvas inert") is broken at its last rung: the node body is submitted first and, on this imgui build, wins an overlapping hover unless the LATER item is reached — in practice the port rects sit inside the node rect, so the ports' `is_item_hovered()` / `is_item_active()` depend on imgui's last-submitted-wins for non-overlap-declared items rather than on the declared chain the spec pins. This is the one clause the spec singles out as the failure mode. Unrecorded. |
| A port's hit box is `max(GRAPH_PORT_R * zoom, GRAPH_HIT_MIN)` in screen pixels, drawn dot keeps scaling | LANDED | `hit = max(SIZE.GRAPH_PORT_R * view.zoom, float(SIZE.GRAPH_HIT_MIN))` for the rect; `r = SIZE.GRAPH_PORT_R * z` for the drawn dot |
| Every context menu on the canvas opens with `begin_popup_context_item(None)` | LANDED | `_node_menu`'s only popup call is `imgui.begin_popup_context_item(None)`, with the reason in a comment |
| The canvas menu opens by hand: right-click released over the background with no node hovered | LANDED | `if bg_hovered and not node_hovered and imgui.is_mouse_released(right): imgui.open_popup("##graph_canvas_menu")`, drawn by `_canvas_menu`'s plain `begin_popup` |

## D9 — pan, zoom, fit, arrange

| Clause | Verdict | Evidence |
|---|---|---|
| Middle-drag on the canvas, or Alt+left-drag, pans (`io.mouse_delta`) | LANDED | `panning = bg_active and (is_mouse_down(middle) or io.key_alt)`; the bg button is submitted with `mouse_button_left \| mouse_button_middle` so both activate it |
| The wheel zooms about the cursor (the canvas point under the cursor is invariant) | LANDED | the wheel block takes `under = xf.to_canvas(mouse)` before the zoom change and solves `view.pan` so `under` maps back to `mouse`; `xf` rebuilt after |
| Fit from the visible nodes' bounding box with `SPACE.LG` margin, zoom capped at 1 | LANDED | `_fit`: `_FIT_MARGIN = float(SPACE.LG)`, `zoom = min(1.0, avail.x / w, avail.y / h)` then clamped into the D7 range |
| Runs once when a document's canvas first draws at a nonzero size | LANDED | `if not view.fitted: _fit(...)`; `_fit` returns early on `not nodes or avail.x <= 0 or avail.y <= 0` and only then sets `fitted = True`; smoke frame 45 asserts it fired |
| On every scope change | LANDED | every scope assignment sets `fitted = False` (four sites) |
| From the canvas menu | LANDED | `_canvas_menu`'s `Fit` sets `view.fitted = False` |
| Arrange is `App.arrange_graph(document_id)`, over every pass of the DOCUMENT | LANDED | `App.arrange_graph` lays out `list(document.passes)` |
| …into `position` through `set_pass_positions`, one save | LANDED | one call; smoke frame 47 asserts `saves.call_count == 1` and no pass left unplaced |
| …and then fits the current scope | LANDED | `self.graph_view_for(document_id).fitted = False` as the last statement |
| Left-drag on empty canvas is the rubber band (W2) | LANDED | the `band_anchor` block, gated `not panning` |

## D10 — click and menus

| Clause | Verdict | Evidence |
|---|---|---|
| Click a node: `pick_pass(document_id, name, focus_editor=False)` | LANDED | `_click`'s tail |
| Double-click a node: the same with `focus_editor=True` | LANDED | `_double_click`'s tail |
| Click a box: pick the bundle output | LANDED | `_click` passes `node.bundle` for a box |
| Double-click a box: enter | LANDED | `_double_click` sets the scope |
| Right-click a node: the strip's item set, extracted into `pass_list.pass_menu_items(app, document_id, name)` | LANDED | `pass_list.pass_menu_items` exists and is called by BOTH `pass_list._draw_context_menu` and `widgets/pass_graph._node_menu` |
| The strip keeps its own `begin_popup_context_item` with its explicit id | LANDED | `_draw_context_menu` still uses `f"##pass_menu_{name}"` |
| Right-click a box: Open, and Dissolve | LANDED | `_node_menu`'s `kind == "box"` branch: `Open` then `Dissolve` |
| Right-click empty canvas: Add pass, Import..., Fit, Arrange | LANDED | `_canvas_menu`'s four `menu_item_simple`s, with a separator |
| `pass_menu_items` keeps the strip's two gates (Delete only while `len(passes) > 1`; Leave group only while the entry carries one) | LANDED | both gates moved verbatim, including the "gated in Python, not by `enabled=`" comment |
| A ghost's right-click offers the same item set | LANDED | `_node_menu`'s `else` branch runs for `kind in ("pass", "ghost")` |
| A box's menu is its own and never `pass_menu_items` | LANDED | the `if node.kind == "box"` branch returns without calling it |
| The whole canvas under `begin_disabled(app.copilot_turn_active)`; the draw list still paints the live pictures | LANDED | `draw` brackets the tab row AND the child; the draw-list calls are unaffected by `begin_disabled` |
| `Group...` on the node menu | LANDED (see D14) | `_node_menu`'s `if node.kind == "pass" and imgui.menu_item_simple("Group...")` |

Note, not a spec violation but worth the reader's eye: the canvas node menu's item set is
`pass_menu_items` **plus `Group...`**, so a canvas node's menu is a strict superset of the
strip's. D14 asks for exactly that ("Right-click with a selection adds `Group...` to the node
menu"), so the two surfaces are intentionally not identical — D10's "the strip's item set" and
D14's "adds" are consistent once read together.

## D11 — the error language

| Clause | Verdict | Evidence |
|---|---|---|
| filled disc = wired | LANDED | `_draw_port_dot`'s `"wired"` → `add_circle_filled` |
| hollow ring = unfilled | LANDED | the `else` branch → `add_circle` |
| ring with a filled centre = `NoSource` | LANDED | `"none"` → `add_circle` + `add_circle_filled(r*0.45)` |
| double ring = `prev` | LANDED | `"prev"` → two `add_circle`s |
| media-bound sampler draws a small square dot | LANDED | `"media"` → `add_rect_filled` |
| Node border `STATE_ERROR` when `compile_unit.errors`, else `ACCENT_PRIMARY` when the pass is the document output, else `BORDER` | DEVIATED | The ladder in `_draw_node` is `error → is_output → selected (COLOR.SELECT) → tint → BORDER`. Two rungs the spec's precedence does not name: the **selection** border (W2's need, unavoidable, but unrecorded) and the **group tint** border on a box (D5 says the box's tint is `group_tint`, so this is consistent with D5 but not with D11's stated ladder). The precedence is the spec's phrase "node borders and their precedence", so the extra rungs belong in the Review history. Unrecorded. |
| A never-compiled pass has a dashed border | LANDED | `node.uncompiled = render_pass.program is None and not render_pass.compile_unit.errors` → `_dashed_rect` |
| A node the output does not need (`evaluation_order(wiring, output) or {output}`) dims its name and its edges | LANDED | `live = set(evaluation_order(wiring, output)) or {output}`; `stale` on the node dims name + picture; `_Edge.dim` dims the wire via `dim_col` |
| Exactly ONE pass per cycle carries the culprit message; `cycle_edges(errors)` parses the trail and returns consecutive `(producer, consumer)` pairs; every edge in the set is `STATE_ERROR` | LANDED | `pass_graph.cycle_edges` splits on `_CYCLE_PREFIX` then `" -> "` and emits `pairwise` reversed into `(producer, consumer)`; `_build_view` sets `on_cycle=(source, name) in cycle_pairs`; `_draw_canvas` picks `err_col` first. `test_cycle_edges_are_the_pairs_of_the_culprit_message` asserts the one-culprit premise AND the exact pair set |
| Victims get nothing on the node and nothing on their edges | DEVIATED | True for a plain node (the node border reads `compile_unit.errors` only, never `culprits`) and true for edges. **Not** true for a box: `_build_view` sets the box's `error=any(compile errors) or any(m in culprits for m in members)`, and `culprits` is the set of passes whose message starts with the cycle prefix — the CULPRIT, i.e. the one member the planner named. So a box turns red for a cycle member, where D11 says a box carries `STATE_ERROR` "when any member has a compile error" and says victims get nothing. Defensible as a design choice (a cycle inside a box would otherwise be invisible at the root), but it is not what the spec says, and manual item 16 ("the two wires of the loop turn red and no NODE turns red") is written against the spec's rule. Unrecorded. |
| The cue reads `plan_passes(document.effective_wiring())[1]` computed by the canvas each frame, never `Document.graph_errors` | LANDED | `_build_view`: `errors = plan_passes(wiring)[1]`; `grep graph_errors shaderbox/widgets/pass_graph.py` is empty |
| A box carries `STATE_ERROR` when any member has a compile error | LANDED | the first disjunct of the box's `error=` |
| Nothing for non-convexity | LANDED | no convexity computation anywhere |
| No cue for dtype / scale / filter / wrap | LANDED | none drawn |
| A compile-erroring pass's picture is its last good frame; the node dims it | LANDED | the texture is the pass's own canvas (never cleared on a failed compile); `picture_alpha = alpha * (0.5 if node.error or node.stale else 1.0)` |

## D12 — wires

| Clause | Verdict | Evidence |
|---|---|---|
| Press on an output dot, release over an input port writes `set_sampler_source(…, PassSource(producer))` | LANDED | the `##gout_` loop starts a `WireDrag(producer=member, start=…)`; `_drop` → `App.drop_wire` → `session.set_sampler_source(..., PassSource(producer))` |
| Release over empty canvas from an output writes nothing | LANDED | `_drop`'s `target is None` path returns without a write when `wire.grabbed is None` |
| Press on a FILLED input port grabs its wire | LANDED | the `##gport_` loop starts a `WireDrag(..., grabbed=(owner, sampler))` gated on `port.kind == "wired" and port.source is not None` |
| Release over another input port moves it (`PassSource` there, `NoSource` on the original) | LANDED | `_drop`: `drop_wire(...)` then, on success and `grabbed is not None`, `app.unwire(document_id, *wire.grabbed)` |
| Release over empty canvas writes `NoSource` on the original | LANDED | `_drop`'s tail: `app.unwire(document_id, *wire.grabbed)`; `App.unwire` writes `NoSource()`; `test_unwire_writes_black_by_decision` |
| Every drop lands through ONE verb `App.drop_wire(document_id, consumer, sampler, source) -> str` | DEVIATED (signature) + LANDED (funnel) | The funnel is real and pinned (`test_the_widget_makes_no_session_write_of_its_own` greps the widget source for `set_sampler_source`). The signature differs: the code is `drop_wire(self, document_id, producer, consumer, sampler) -> str` — a producer NAME in a different parameter order, not a `source` object. A second verb, `App.unwire(document_id, consumer, sampler)`, carries the `NoSource` half; the spec routed both through `drop_wire`'s `source` parameter and named no `unwire`. Two verbs where the spec has one, and the widget calls both. Unrecorded. |
| A drop that would close a cycle is refused BEFORE any write, via `pass_graph.refuse_drop(wiring, consumer, sampler, producer)` building `wiring_with` and running `plan_passes`, returning the planner's message on ANY error | LANDED | `refuse_drop` exactly as specified; `App.drop_wire` returns the refusal before touching the session; `test_drop_wire_writes_the_read_and_refuses_the_loop` asserts `write.call_count == 0` |
| The verb toasts the message through `notifications` | DEVIATED (location) | `App.drop_wire` **returns** the refusal and does not toast; the widget's `_drop` does `app.notifications.push(error)`. The spec says "the verb toasts". Testability is unharmed (the test asserts the returned string), and the other 092 verbs are inconsistent with each other here: `arrange_graph` / `commit_node_drag` toast internally, `group_selection` / `dissolve_group` toast internally AND return. Unrecorded. |
| A node's output into its own port is feedback, allowed | LANDED | `refuse_drop` plans a self-read as feedback; `test_drop_wire_writes_the_read_and_refuses_the_loop` asserts `drop_wire(..., "c", "c", "u_src") == ""` |
| A drop on a media-bound port is refused ("bound to media; unbind on the Uniforms tab") | LANDED | `App.drop_wire`: `if value is not None and not isinstance(value, SamplerSource): return "bound to media; unbind on the Uniforms tab"`; `test_drop_wire_refuses_a_media_bound_port_and_keeps_the_texture` asserts the texture survives and `glo != 0` |
| A drop onto a port wired by the name rule materializes an explicit `PassSource` | LANDED | the write is unconditionally `PassSource(producer)`; no branch reads the prior value's kind |
| A ghost's ports are drop targets and drag sources like any other | DEVIATED | Drop TARGET: landed — the `##gport_` loop runs for every node including ghosts and sets `drop_target`. Drag SOURCE from a ghost's INPUT port: landed (same loop). Drag source from a ghost's OUTPUT dot: **absent** — the `##gout_` loop opens with `if node.kind == "ghost": continue`, so a ghost's output dot has no hit rect at all. Manual item 57 ("drag a member's output onto a GHOST's port") still works, but the mirror gesture the clause promises ("drag sources like any other") does not. Unrecorded. |
| The wire in flight is a bezier from the source to the cursor on channel 2 | LANDED (picture) / DEVIATED (channel) | drawn with `add_bezier_cubic` in `ACCENT_PRIMARY` after `channels_merge()` — see D7's channel deviation |

## D13 — drag

| Clause | Verdict | Evidence |
|---|---|---|
| Press on a node body and move: the node follows `io.mouse_delta / zoom` | LANDED | `view.node_drag.update(io.mouse_delta.x / view.zoom, io.mouse_delta.y / view.zoom)` |
| Release writes its position through `set_pass_positions`, one save | LANDED | `App.commit_node_drag` → one `set_pass_positions`; `test_commit_node_drag_writes_once_and_only_the_moved_passes` asserts `saves.call_count == 1` |
| A pure state machine in `graph_state.py`: `NodeDrag.begin(names, positions)`, `.update(delta)`, `.commit()` | DEVIATED | `NodeDrag` has `update`, `current` and `commit`; there is **no `begin` classmethod** — construction is `NodeDrag(origin={...})` at the call site (and a `begin` would have been a `@classmethod`, which this repo's code rules forbid outside genuine alternate constructors, so the omission is right). Unrecorded. |
| `update` returns nothing to write; `commit` is the only thing that does | LANDED | `update` returns `None`; `test_a_drag_writes_nothing_until_commit_and_then_every_moved_name_once` asserts `drag.update(...) is None` twice |
| `App.commit_node_drag(document_id)` is the one caller of `set_pass_positions` from a drag | LANDED | grep: `set_pass_positions` has exactly two callers, `arrange_graph` and `commit_node_drag` |
| A box drags every member by the same delta | LANDED | `_drag_names` returns `list(node.members)` for a box; one shared `delta` |
| A drag moves every selected node together when the pressed node is selected | LANDED | `_drag_names`: `if node.name in view.selection: return sorted(view.selection)` |
| Snapping aligns the moving node's left or top edge to any other VISIBLE node's within `GRAPH_SNAP_PX` SCREEN pixels | DEVIATED | `_snap` uses `threshold = SIZE.GRAPH_SNAP_PX / view.zoom` and then compares CANVAS-space distances (`abs(n.pos[0] - current[0])`), which is the correct conversion and does yield a screen-pixel threshold — that half is right. What deviates: it snaps only the drag's `primary = next(iter(drag.origin))` node, and it re-applies the correction to `drag.delta` **cumulatively every frame** while the node stays within the threshold (the delta is mutated, then `current()` is recomputed from it next frame), so a node held near a guide keeps absorbing the same correction rather than resting on it. Unrecorded; this is a behaviour defect, not only a wording one. |
| Draws the guide line on channel 2 | LANDED (picture) / DEVIATED (channel) | `view.guides` drawn after `channels_merge()` — see D7 |

## D14 — selection and Group

| Clause | Verdict | Evidence |
|---|---|---|
| `GraphViewState.selection: set[str]` of pass names; a box selects its members | LANDED | `selection: set[str]`; `_click` uses `set(node.members)` for a box; `_draw_canvas` marks a box selected when `set(node.members) & view.selection` |
| Left-drag on empty canvas draws the rubber band from `io.mouse_pos - get_mouse_drag_delta` | LANDED | `view.band_anchor = (io.mouse_pos.x - delta.x, io.mouse_pos.y - delta.y)` with `delta = get_mouse_drag_delta(left)` |
| Selects every node whose rect intersects it on release | LANDED | the release block's AABB test, `picked |= set(node.members) if box else {node.name}` |
| Shift-click toggles one | LANDED | `_click(..., extend=io.key_shift)` → `view.selection ^= names` |
| Click on empty clears | LANDED | `if imgui.is_item_clicked(left) and not io.key_shift: view.selection.clear()` |
| `Group...` on the node menu with a name popup (Enter or Create commits, Cancel or Esc cancels) | LANDED | `_group_prompt`: `input_text` with `enter_returns_true`, `primary_button("Create")`, `standard_button("Cancel")`; Esc closes because it is a plain `begin_popup` (imgui's own dismissal) |
| Writes `set_pass_groups(document_id, names, group)` through `App.group_selection`, a new session verb validating once and saving once | LANDED | `App.group_selection` → `session.set_pass_groups(document_id, sorted(view.selection), group)`; `ProjectSession.set_pass_groups` validates once via `group_name_error` then loops `with_group` and saves once |
| Writes nothing on a refusal | LANDED | `set_pass_groups` returns before touching `document.graph`; `test_set_pass_groups_saves_once_and_refuses_a_pass_name` asserts the groups are unchanged after a refusal, and `test_group_selection_and_dissolve_are_one_write_each` asserts it through the App verb |
| Selecting a box and grouping it with others rewrites its members to the new label | LANDED | flat `with_group` per name; `test_group_selection_and_dissolve_are_one_write_each` groups `a,b` then `a,b,c` and asserts `{"trio"}` |
| `Dissolve` on a box menu is `set_pass_groups(members, "")` | LANDED | `App.dissolve_group` collects members from the entries and calls `set_pass_groups(..., "")` |
| `set_pass_group` stays for the modal and the copilot and calls the batched verb with one name | LANDED | `set_pass_group` is now a one-line delegate to `set_pass_groups`; the copilot reaches it through `pass_set_group=self.set_pass_group` in the capabilities wiring |
| The prompt is reachable with a selection | DEVIATED (minor) | The spec says "Right-click **with a selection** adds `Group...`"; the code shows `Group...` on every `kind == "pass"` node menu regardless of selection, and materializes a one-name selection when the pressed node is not selected (`if node.name not in view.selection: view.selection = {node.name}`). That is a superset of the specified behaviour and matches manual item 50's flow; it also makes manual item 30 ("group a single pass") reachable without a rubber band. Unrecorded. |

## D15 — a media-bound sampler's port

| Clause | Verdict | Evidence |
|---|---|---|
| Draws as a port in a media state (a square dot, no wire) | LANDED | `node_ports` classifies a non-`SamplerSource` value as `"media"`; `_draw_port_dot`'s `"media"` → `add_rect_filled`; no edge is emitted because `port.source is None` |
| A drop on it is refused (D12) | LANDED | `App.drop_wire`'s media branch, asserted with its falsifier in `test_drop_wire_refuses_a_media_bound_port_and_keeps_the_texture` |
| The canvas reads `Pass.uniform_values[sampler]` and branches by type exactly as `widgets/uniform.py` does | LANDED | `node_ports(declared, values, wiring_row, name)` branches `NoSource` → `"none"`, `PassSource \| AutoSource` → `"unfilled"`, else `"media"`, with the wiring row deciding `"wired"`/`"prev"` first; `test_node_ports_classify_every_state_and_put_feedback_last` walks all five |
| A sampler whose resolved pass is the consumer itself is the feedback port, whatever its name | LANDED | the `source == name` branch precedes every other |

## D16 — Dissolve is required

| Clause | Verdict | Evidence |
|---|---|---|
| Dissolve exists as the inverse of Group | LANDED | `App.dissolve_group` + the box menu item |
| A box gets no Delete verb | LANDED | `_node_menu`'s box branch has exactly `Open` and `Dissolve` |
| A member is deleted from its own node menu with `delete_pass` + `close_editor_for_path`, exactly `pass_list._delete_pass` | LANDED | `pass_menu_items`'s `Delete` calls `pass_list._delete_pass`, the same function the strip uses |
| "…with the strip's two-click arm" | DEVIATED (the spec is wrong, not the code) | **Answering the audit's question directly: the canvas's node-menu Delete has no confirm step — and neither does the strip's.** The arm is on the strip's TILE, not on its menu: `preview_cell` returns `delete_armed` / `delete_confirmed`, which `_draw_pass_tile` turns into `app.pass_delete_armed = name` and then the wash. `pass_list.pass_menu_items`'s `Delete` — the code the strip's own context menu has always run, and the code the canvas now shares — calls `_delete_pass` immediately. So the two surfaces do NOT drift (D10's goal is met), and the canvas is exactly as safe as the strip's menu; but D16's sentence describes an arm that the menu path never had. The canvas has no tile and so no place for the wash. Unrecorded; the spec sentence is what needs correcting. |

## D17 — one namespace for passes and groups

| Clause | Verdict | Evidence |
|---|---|---|
| `pass_graph.group_name_error(group, pass_names) -> str` (the pattern, then the collision) | LANDED | exactly that order in `group_name_error` |
| `set_pass_groups` calls it | LANDED | `ProjectSession.set_pass_groups` → `group_name_error(group, document.passes)` |
| …hence `set_pass_group`, the modal and the copilot | LANDED | `set_pass_group` delegates; the copilot's `pass_set_group` capability points at `set_pass_group` (`project_session.py`'s capabilities block); the modal's path is `pass_list`'s `Leave group` / the settings popup, both on `set_pass_group` |
| `pass_import.plan_import` calls it | LANDED | `plan_import` replaced its inline `PASS_NAME_RE` check with `group_name_error(group, host_names)` |
| `_pass_name_error` gains the mirror check, so `add_pass` and `rename_pass` reject a pass named like an existing group | LANDED | `_pass_name_error(name, existing, graph)` with `any(entry.group == name …)`; both callers updated to pass `document.graph` |
| The root keys nodes and boxes by name | LANDED | `_build_view` keys `p:{name}` and `b:{group}` — namespaced prefixes, so the dict itself could not collide, but the DRAWN identity is the bare name, which is what the rule protects |
| Messages: the existing pattern message, and "a pass and a group cannot share a name" | LANDED | both strings verbatim; `test_group_name_error_covers_the_pattern_and_the_namespace`, `test_every_group_writing_entry_point_shares_one_validator` |

## D18 — a rename plans before it moves

| Clause | Verdict | Evidence |
|---|---|---|
| `Document.wiring_if_renamed(old, new)` re-keys `self.passes` under the new name | LANDED | `document.wiring_if_renamed` rebuilds the dict in place, preserving insertion order (with the reason in a comment) |
| AND applies `rename_pass_sources(old, new)` to a copy of each pass's sampler values | DEVIATED (mechanism) | It applies the rewrite **in place** to the live `uniform_values` and records `rewritten` to undo it, rather than to a copy. The observable contract (the document is as it was found, and after a raise) is what the test asserts and it holds; but "to a copy of each pass's sampler values" is not what the code does, and the in-place version is the one that can be observed by another thread mid-call. Unrecorded. |
| Reads `effective_wiring()` | LANDED | inside the `try` |
| Restores both in a `finally` | LANDED | the `finally` undoes `rewritten` then restores the original key order |
| `rename_pass` refuses when `plan_passes` over that wiring reports any error, with the planner's message | LANDED | the guard in `ProjectSession.rename_pass`; it prefers the cycle message and falls back to `loop[0].message` |
| The guard runs BEFORE the file moves | LANDED | the guard sits between `_pass_name_error` and the "Transactional (D15)" block; `test_rename_pass_refuses_the_cycle_it_would_create` asserts `old_path.exists()` and that the new shader path does not |
| `wiring_if_renamed` leaves the document as it found it, also after a raise | LANDED | `test_wiring_if_renamed_leaves_the_document_as_it_found_it` asserts keys and value identity, then repeats under a patched `effective_wiring` that raises |

## D19 — what the copilot sees: nothing new

| Clause | Verdict | Evidence |
|---|---|---|
| No tool accepts or reports a position | LANDED | `grep -rniE "position" shaderbox/copilot/` returns four hits, all prose in `prompt.py` / `prompt_context.py` / a `CopilotLayout` comment — none a tool field. `test_no_session_verb_but_one_accepts_a_position` pins the structural form at the session boundary |
| The pass table stays flat | LANDED | `copilot/tools/passes.py` unchanged by both commits |
| No groups paragraph in the prompt | LANDED | `prompt.py` untouched by both commits; no `graph`/`group` paragraph added |
| No `group_passes` tool | LANDED | `grep group_passes shaderbox/` empty |
| The dogfood harness needs nothing | LANDED | no harness file in either diff |

One consequence worth naming, since it is a behaviour change the copilot sees and D19 says
"nothing new": the copilot's existing `pass_set_group` now routes through `set_pass_groups` and
therefore through `group_name_error`, so a copilot call that names a group colliding with a pass
is newly refused. D17 asks for exactly this ("hence `set_pass_group`, the modal and the
copilot"), so it is intended — it is a widened refusal, not a new capability, and
`tests/test_copilot_pass_tools.py` was re-run per the Files-touched note.

## D20 — docs (the sanitize checklist)

Every item below is **ABSENT**. The spec's own status line says the post-implementation review is
pending, so this is expected; the list is the checklist the sanitize step owes.

| Doc edit owed | State |
|---|---|
| `conventions.md`'s "A pass GROUP is a label … and nothing folds (feature 091)" keeps its revisit trigger and gains one sentence scoping the no-folding half to the STRIP (the graph contracts a group to a box; the box is never a node the planner orders, so convexity is not a rule there) | ABSENT — the entry at `conventions.md` line ~822 is verbatim 091's; no 092 sentence |
| A new `conventions.md` entry recording D6 (a position is written only by a placement, never by a draw) | ABSENT |
| A new `conventions.md` entry recording D1 (ports from the program, edges from the wiring) | ABSENT |
| `imgui-ui` skill §8 gains D8's two canvas rules (the allow-overlap chain; the context-item id) | ABSENT — `.claude/skills/imgui-ui/SKILL.md` has no 092 mention and no canvas section |
| `dev_flow.md`'s module map gains `widgets/pass_graph.py` | ABSENT |
| `dev_flow.md`'s module map gains `widgets/graph_state.py` | ABSENT |
| `dev_flow.md`'s `pass_graph.py` entry describing the new pure half | ABSENT — the entry still stops at `clamp_canvas_size` |
| The Help panel's Passes section gains one sentence: a port exists because the shader declares a sampler | **LANDED** (the one D20 item that shipped) — `help_content.py`: "On the graph view a port exists because the shader declares a sampler, so a new pass has none until you add one." |
| 070's spec gets its pointer to this feature | ABSENT — `grep 092 ai_docs/features/070_pass_reads/01_spec.md` is empty |
| The import dialog gets one line saying the source's own groups are flattened under the new one | **LANDED** — `popups/import_passes.py`: `caption_text("groups flattened")`, gated on the source carrying any group |
| `roadmap.md` row + Active-context banner for 092 | ABSENT — the banner still reads "Next: nothing is claimed; the graph view he reopened is the candidate"; no 092 row |

## Files touched — the named tests

Read by body, not by name. "Falsifier" = the test's own comment or construction names the mutation
the spec asked it to exclude.

| Spec's test name | Exists? | Asserts the invariant? | Carries the falsifier? |
|---|---|---|---|
| `rank_layout` puts producers left of consumers | yes — `test_rank_layout_puts_producers_left_of_consumers` | yes: every `(producer, consumer)` of the bloom wiring compared on x | yes, and it is EXERCISED: the wiring is reversed (`dict(reversed(...))`) so insertion order would fail |
| …is deterministic over dict order | yes — `test_rank_layout_is_deterministic_over_dict_order` | yes: two dict orders → equal dicts | n/a (the test IS the falsifier) |
| …places only the names asked for (two of five placed, three returned) | yes — `test_rank_layout_places_only_the_names_asked_for` | partially: asserts `set(laid) == {"b_bright","b_blur"}` (two asked, two returned). The spec's shape was "two of five placed, three returned"; the test passes `placed` for two and asks for two, so the "returned ⊂ asked" invariant is pinned but the stated arithmetic is not | yes, stated ("return a position for every name, which would overwrite a drag") |
| …keeps cycle members at rank 0 | yes — `test_rank_layout_keeps_cycle_members` | yes: both members returned AND `graph_ranks == {a:0,b:0,c:0}` | yes ("lay out `plan.order` alone") |
| …keeps group members adjacent | yes — `test_rank_layout_keeps_group_members_adjacent` | yes: the two bloom members are contiguous in a rank shared with an ungrouped `side` | yes ("sort a column by strip order alone") |
| `group_boundary` over bloom / non-convex / generator / one-member / split / reads-nothing, each asserting the full (inputs, outputs) pair | yes — six tests, one per shape | mostly: bloom, non-convex, generator, one-member and reads-nothing assert inputs; outputs asserted in bloom, non-convex, generator, one-member, split. **The split test asserts `outputs` and `bundle` but not `inputs`**, so "the full pair" is not asserted for that shape | yes for the reads-nothing case ("drop the 'or unfilled' clause") |
| `test_a_terminal_box_still_has_its_bundle_output_port` | yes, by that exact name | yes: exactly one output, the bundle | yes ("make it conditional on an outside reader", naming mutations case 2) |
| `bundle_output` follows its four branches in order and always returns a member | yes — `test_bundle_output_follows_its_four_branches_in_order` | yes: all four branches, plus a totality loop asserting membership | implicit (the branch order IS the assertion) |
| `node_ports` builds from the program's samplers, never a stored row | yes — `test_node_ports_come_from_the_program_never_the_stored_rows` | yes: a dead `u_gone` row grows no port | yes ("build from `uniform_values` keys") |
| …classifies media, `NoSource`, wired, unfilled and feedback | yes — `test_node_ports_classify_every_state_and_put_feedback_last` | yes: all five kinds in one ordered assertion, plus feedback last and both `source` endpoints | implicit |
| `refuse_drop` refuses `a->b->c` + `c -> a.u_c` with the trail; allows the diamond, the unrelated, a self-read | yes — `test_refuse_drop_refuses_every_cycle_and_allows_every_legal_drop` | yes: all four cases, and the trail substring `a -> c -> b -> a` | yes, and EXERCISED: a `longer` wiring is built so the culprit is a third pass, not an endpoint |
| `cycle_edges` returns the consecutive pairs of a culprit message | yes — `test_cycle_edges_are_the_pairs_of_the_culprit_message` | yes: the exact three-pair set, plus `cycle_edges([]) == set()`, plus the one-culprit premise asserted with its own message | the premise assertion (`len(culprits) == 1`) is the falsifier for D11's rewrite |
| `position` bounds (NaN, ±inf, ±1e30 raise; a pair accepted) | yes — `test_a_position_is_bounded_on_the_model` (parametrized, 5 cases) + `test_a_legal_position_is_accepted_and_carried_by_the_funnel` | yes: all five rejections; acceptance, field preservation, and survival through `with_group` | yes ("a bare `tuple[float, float]`", and separately "`model_copy(update=…)`, which skips the field's bounds") |
| `group_name_error` over the pattern and the collision | yes — `test_group_name_error_covers_the_pattern_and_the_namespace` | yes: empty, bad pattern, collision, clean | implicit |
| `test_graph_state.py`: `NodeDrag.update` returns nothing, `commit` returns every moved name once | yes — `test_a_drag_writes_nothing_until_commit_and_then_every_moved_name_once` | yes: `update(...) is None` twice, accumulated delta, each name once | yes ("return positions from `update`") |
| …the scope revalidation helper falls back to the root | yes — `test_a_scope_no_pass_carries_falls_back_to_the_root` | yes: live, dead, empty | implicit |
| `test_graph_persistence.py`: `position` round-trips (extend `test_a_graph_round_trips_every_field`) | yes — the existing test now places `composite` at `(120.0, -8.5)` and asserts it back | yes | n/a |
| …a corrupt position costs only itself | yes — `test_a_corrupt_position_costs_that_position_and_nothing_else` | yes: three entries (out-of-range, wrong type, good), siblings' `iterations`/`group` preserved | yes ("validate the entry whole — `iterations` goes too") |
| `test_pass_verbs.py`: `set_pass_positions` saves once for the whole set | yes — `test_set_pass_positions_saves_once_for_the_whole_set` | yes: `saves[0] == 1` for three passes, reload round-trip, plus the two refusal paths | yes ("loop a single-position verb — the count becomes three") |
| …a position survives every other verb (target, iterations, group, output, rename) | yes — `test_a_position_survives_every_other_pass_verb` | yes: all five named verbs | yes (names the `with_target` rebuild bug) |
| …`set_pass_groups` saves once, and a group named like a pass is refused with nothing written | yes — `test_set_pass_groups_saves_once_and_refuses_a_pass_name` | yes: one save, the refusal, groups unchanged after, plus the pattern refusal and the missing-pass refusal | yes ("validate per pass inside the loop, leaving a partial write behind") |
| …`add_pass` refuses a name a group carries | yes — `test_add_pass_and_rename_refuse_a_name_a_group_carries` | yes, both `add_pass` and `rename_pass`, plus the pass survives | implicit |
| …every group-writing entry point refuses the same collision | yes — `test_every_group_writing_entry_point_shares_one_validator` | yes: `set_pass_groups`, `set_pass_group`, `plan_import`, all three asserted against the same string | yes ("the check on `set_pass_groups` alone") |
| …`rename_pass` refuses the cycle it would create, asserting the planner's message AND that the old file still exists and the new does not | yes — `test_rename_pass_refuses_the_cycle_it_would_create` | yes: all three, plus the in-memory dict unchanged | yes ("delete the guard — the rename succeeds and the plan reports the cycle") |
| …`wiring_if_renamed` leaves keys and every `uniform_values` identity as found, also after a raise | yes — `test_wiring_if_renamed_leaves_the_document_as_it_found_it` | yes: keys, value, the rewritten row observed inside the call, then the raising path | yes ("drop the restore") |
| …`import_passes` leaves every position `None` | yes — `test_import_passes_leaves_every_position_none` | yes: the source is deliberately placed first, then every copied entry asserted `None` | yes ("the entry copy that carried the source's coordinates verbatim") |
| …no `ProjectSession` public method but `set_pass_positions` names a position parameter (reflection) | yes — `test_no_session_verb_but_one_accepts_a_position` | yes: `inspect.getmembers` + `signature`, exact list equality | the reflection IS the falsifier |
| …the two `pass_list.draw` tests keep asserting what they name after the move | PARTIAL | `test_canvas_fields.py`'s two readout tests were retargeted (`captions[-1]` → `captions[-2]`) with the reason in the docstring, so they still assert the readout. But these are the CANVAS-FIELD tests, not "the two `pass_list.draw` tests": no test in the repo drives `pass_list.draw` and asserts the tiles alone, so the clause's own subject is unverified. See Findings |
| `test_graph_view.py`: `drop_wire` refuses the cycle and writes nothing | yes — `test_drop_wire_writes_the_read_and_refuses_the_loop` | yes: the message, `write.call_count == 0`, the value not written, plus feedback allowed | yes ("refuse only when the culprit is an endpoint") |
| …accepts the legal drop and writes the `PassSource` | yes — same test (`_chain` asserts two legal drops, then the `PassSource("b")` value) | yes | n/a |
| …refuses a media-bound sampler and keeps the bound texture | yes — `test_drop_wire_refuses_a_media_bound_port_and_keeps_the_texture` | yes: the message, no write, `is bound`, `glo != 0` | yes ("call `set_sampler_source` unconditionally, which runs `try_to_release`") |
| …writes `NoSource` on the original for a grab-and-drop-on-empty | PARTIAL — `test_unwire_writes_black_by_decision` | asserts `App.unwire` writes `NoSource`. The GESTURE (`_drop` with `target is None` and a `grabbed` wire) is not driven; the widget's `_drop` is untested | implicit |
| …`arrange_graph` saves once and leaves no pass unplaced | **ABSENT from `test_graph_view.py`** | the invariant IS asserted, but only in `scripts/smoke.py` frame 47, which `make gates` reports as SKIPPED without a display. On the dev box nothing checks it. See Findings | n/a |
| …`group_selection` on a box plus a plain pass rewrites the members to the new label | yes — `test_group_selection_and_dissolve_are_one_write_each` | yes: `{a,b}` → `pair`, then `{a,b,c}` → `trio`, plus a refusal leaving it unchanged | implicit |
| …`dissolve_group` clears every member with one save | yes — same test | yes: one save, every group `""` | implicit |
| …the widget's source contains no call to `set_sampler_source`, `set_pass_positions` or `set_pass_groups` | yes — `test_the_widget_makes_no_session_write_of_its_own` | yes: source grep for all three | the grep IS the falsifier |
| `test_ui_regions.py`: every `PassesView` has a label | yes — `test_every_passes_view_has_a_label` | yes, set equality | implicit |
| …each label within the control budget | yes — `test_every_passes_view_label_is_within_the_control_budget` | yes, `len(split()) <= 2` | implicit |
| …the default is STRIP and the choice persists through `UIAppState` | yes — `test_the_default_is_the_strip_and_the_choice_persists` | yes: default, save, reload | implicit |
| `test_theme.py`: `GRAPH_EDGE` is in the mirrored exclusion set | yes — `test_group_tints_are_stable_and_collide_with_nothing` extended, plus a new `GRAPH_EDGE != STATE_ERROR` line | yes | yes, stated ("`GRAPH_EDGE = COLOR.GROUP_TINTS[2]` imported clean before it joined") |
| `test_button_tiers.py`: `("widgets/pass_graph.py", "invisible_button")` joins `_NOT_A_VERB` in the SAME commit | yes | yes — and it is in commit `f04821a`, the widget's own commit, as required | n/a |
| `test_ui_prose_budget.py`: `menu_item_simple` joins `_IMGUI_ROWS` at four words | yes | yes: `("menu_item_simple", "label", 0, 4)` with the qualifier rationale in a comment | n/a |

## The smoke stretch — spec vs `scripts/smoke.py`

| Spec | Code | Verdict |
|---|---|---|
| Frame 43 switches `app_state.passes_view` to GRAPH and snapshots `graph.model_dump()` | `if frame_idx == 43:` sets `PassesView.GRAPH` and `graph_before = json.dumps(..., sort_keys=True)` | LANDED |
| Frame 43 also closes the pass-settings popup left open by 42 | the `PopupState.CLOSED` / `pass_settings_name = ""` lines moved from 48 into 43 | LANDED (a necessary consequence of inserting the stretch; not in the spec's wording, but harmless and required) |
| Frame 45 asserts the view fitted and sets the scope to `smoke_group` | `assert view.fitted, …` then `view.scope = "smoke_group"` | LANDED |
| Frame 46 asserts the scope survived a frame and sets a scope no pass carries | `assert view.scope == "smoke_group"` then `view.scope = "no_such_group"` | LANDED |
| Frame 47 asserts the scope fell back to the root | `assert view.scope == ""` | LANDED |
| Frame 47 asserts the graph dump is unchanged (a draw wrote nothing) | `assert graph_after == graph_before` with the D6 message | LANDED |
| Frame 47 asserts `arrange_graph` saved exactly once, via a `_count_saves` helper beside `_arm_feedback_canary` | The assertion is there (`saves.call_count == 1`) but it uses `unittest.mock.patch.object`, not a `_count_saves` helper; `scripts/smoke.py` gained a `from unittest import mock` import and defines no `_count_saves`. (A `_count_saves` helper DOES exist — in `tests/test_pass_verbs.py`, where the spec's Files-touched line for that file also asks for one.) | DEVIATED (mechanism only; the invariant is asserted) |
| Frame 47 asserts Arrange left no pass unplaced | `assert all(entry.position is not None …)` | LANDED |
| Frame 47 switches back to STRIP | `app.app_state.passes_view = PassesView.STRIP` | LANDED |
| Frame 48 returns to the canary document | unchanged `app.set_current_document_id(canary_id)` | LANDED |
| Frame 44 | no branch — the spec names none either (it is the free frame that lets the canvas draw once before the fit is checked) | LANDED |

Caveat the spec itself states and that holds: `make gates` reports a display-less smoke as
**skipped**, so none of the above runs on this box. It is not a pass.

## Out of scope — did anything land anyway?

| Out-of-scope item | Landed? |
|---|---|
| Nesting (path labels `post/bloom`) | No. `_GROUP_PATTERN` still forbids a slash; `group_names_in_order` is flat |
| The graph as an `EditorTab.kind` | No. `PassesView` is an app-state enum, not a tab kind; `EditorTab` untouched. **But** see D2's deviation: the widget reserving height for its sibling's button row is the coupling this item's rationale ("the canvas is written pane-agnostic so the move is a mount change") was protecting |
| A wire drop that writes the `sampler2D` declaration into the shader | No. `App.drop_wire` writes only `uniform_values` through `set_sampler_source`; nothing touches `ShaderSource` |
| Keyboard verbs on the canvas (Escape up a tab, Delete) | No. No key handling in the widget; `grep graph` over the key registry is empty. The `input_text`'s `enter_returns_true` in the Group popup is a widget-local commit, not a canvas verb |
| `duplicate_pass` / duplicating a box | No. `grep duplicate_pass shaderbox/` is empty |
| Save a group as a document, the presets folder, a group rename verb | No. No such verb; `set_pass_groups` is a label write, not a rename |
| A cycle cue outside the graph | No cue was added to the strip. **But** `pass_menu_items`, now shared, means any future menu change lands on both — not a violation, a noted coupling |
| Undo | No undo stack; every verb overwrites |
| Nodes at the strip's tile size / header-bar node / ports beside the picture | No. The node is picture-over-name-over-port-rows at `GRAPH_NODE_W`, the round-3 shape |

Nothing out of scope landed. The single item worth the maintainer's eye is the D2 height coupling,
which weakens (without breaching) the pane-agnostic property the `EditorTab.kind` deferral rests on.

## Deviations to record in the spec's Review history

No deviation is recorded today — the Review history ends at the pre-implementation round. The
block below is replacement text for a new paragraph appended to `## Review history`. Each entry
states what shipped, so the spec describes the code.

> **Post-implementation round (2026-09-12): one reviewer, spec fidelity. FINDINGS.**
> Report: `reviews/post_spec_fidelity.md`. The implementation deviations, recorded so the spec
> describes what shipped:
>
> - **D1's compile seam runs every frame, not once.** `widgets/pass_graph.draw` calls
>   `compile_pending_passes(document)` unconditionally; `GraphViewState.compiled` was added for
>   the one-shot and is read nowhere. The call is a no-op once every pass has been attempted,
>   but it walks the passes and builds a sorted failure list per frame.
> - **D3's tab row maps a click by the returned STRING, not by index.** `_tab_row` does
>   `labels.index(clicked)`, which is the ambiguity the clause forbids: a document whose
>   `ui_name` equals a group name makes that group's tab unreachable. Fixed in code, not here.
> - **D5's box is a fixed-size node at its members' top-left, not spanning their bounding
>   box.** The box's position is `min(x), min(y)` over the members and its size is
>   `node_size(ports, box=True)`, so a split group draws one box that does not visually
>   enclose the non-members between its members. Manual item 32's "encloses non-members
>   visually" is rewritten to match.
> - **D6's `rank_layout` takes `gap_x` and `gap_y`** as explicit parameters, keeping
>   `pass_graph` free of `theme`; every caller passes `SIZE.GRAPH_GAP_X/Y`.
> - **D7 splits the draw list into 2 channels, not 3.** Wires are channel 0, nodes channel 1,
>   and the foreground (in-flight wire, rubber band, snap guides) is drawn after
>   `channels_merge()` — later in the merged list, so it paints on top.
> - **D8's allow-overlap chain stops at the node body.** The port and output hit rects are
>   submitted without `set_next_item_allow_overlap()`, so the last rung of the chain the
>   clause names is missing. Fixed in code, not here.
> - **D11's border precedence has two more rungs**, `COLOR.SELECT` for a selected node (W2's
>   need) and the group tint for a box (D5's rule), between `ACCENT_PRIMARY` and `BORDER`.
> - **D11's box reddens for a cycle culprit among its members**, not only for a member's
>   compile error, so a loop inside a group is visible at the root. Manual item 16's "no NODE
>   turns red" holds at the root only while no cycle member is grouped.
> - **D12 is two verbs, not one.** `App.drop_wire(document_id, producer, consumer, sampler)`
>   takes a producer name in that order and writes the `PassSource`; `App.unwire(document_id,
>   consumer, sampler)` writes the `NoSource`. The widget calls both and toasts the returned
>   refusal itself; `drop_wire` returns rather than toasts.
> - **D12's ghosts are drop targets and input-port drag sources, but not output-dot drag
>   sources**: the output hit-rect loop skips `kind == "ghost"`.
> - **D13's `NodeDrag` has no `begin`**; it is constructed at the call site
>   (`NodeDrag(origin=...)`), since a `begin` classmethod is outside this repo's
>   alternate-constructor exception. `update` / `current` / `commit` are as specified.
> - **D13's snapping corrects only the drag's first name** and re-applies the correction to
>   the accumulated delta each frame it stays inside the threshold. Fixed in code, not here.
> - **D16's "the strip's two-click arm" was never on the strip's MENU**: the arm is on the
>   tile's ✕ (`preview_cell`'s `delete_armed` / `delete_confirmed`), while
>   `pass_list.pass_menu_items`'s Delete calls `_delete_pass` immediately. The canvas shares
>   that menu, so the two surfaces match; the sentence is corrected to say so.
> - **D18's `wiring_if_renamed` rewrites the sampler values in place and undoes them in the
>   `finally`**, rather than working on copies. The observable contract (the document as it
>   was found, also after a raise) is asserted.
> - **The smoke stretch counts saves with `mock.patch.object`**, not a `_count_saves` helper;
>   the helper of that name lives in `tests/test_pass_verbs.py`.
> - **`App.arrange_graph`'s "saves once, leaves no pass unplaced" has no headless test.** It
>   is asserted only in the smoke stretch, which `make gates` skips without a display. The
>   spec's `test_graph_view.py` list names it; it is owed.
> - **D20 landed two of eleven items** (the Help sentence, the import dialog's flatten line);
>   the other nine are the sanitize step's checklist.

Manual-verification replacements, to be applied in the same edit:

> - Item 16 becomes: "…the two wires of the loop turn red and no root-level NODE turns red; a
>   box whose member is the planner's culprit does turn red."
> - Item 32 becomes: "…ONE box draws at their members' top-left corner; Arrange pulls the
>   members together."

## False trails

Things that look like findings and are not. Each was checked to the primary artifact.

- **`channels_split(2)` looks like a lost foreground layer.** It is not: the in-flight wire, the
  rubber band and the guides are all submitted after `channels_merge()`, which places them last
  in the merged command list, i.e. on top. The picture matches D7; only the mechanism differs.
- **`App.unwire` looks like a second write path bypassing the refusal funnel.** It is not a
  bypass: `unwire` writes `NoSource`, which can neither close a cycle (removing a read cannot
  create one) nor release a bound texture (a bound value never reaches `unwire` — only a
  `kind == "wired"` port can be grabbed). The one gesture that can destroy data still goes
  through `drop_wire`'s two guards.
- **`_drop` calling `drop_wire` then `unwire` looks like two saves per gesture.** It is two
  saves, but D12 specifies exactly that for the move case ("`PassSource` there, `NoSource` on
  the original") and the one-save rule in the spec is D13's, about positions.
- **The node menu showing `Group...` without a selection looks like it contradicts D14's "with
  a selection".** It is a deliberate superset that makes manual item 30 (group a single pass)
  reachable, and the code materializes the one-name selection before opening the prompt.
- **`x12` rather than `×12` on the badge looks like a D7 miss.** The spec writes `×N` in prose
  and the manual items write `x12` / `x6`; the ASCII spelling is the one the manual list pins
  and the one the font can draw.
- **`test_no_session_verb_but_one_accepts_a_position` looks like a D19 test filed in the wrong
  file.** It is at the right altitude: it pins the SESSION boundary the copilot reaches through,
  which is the structural reason no tool can be handed a coordinate.
- **`_pass_name_error` gaining a `graph` parameter looks like an unrecorded signature change.**
  The spec's Files-touched line asks for exactly this ("`_pass_name_error` taking the group
  names"); it takes the `PassGraph` rather than a name set, which is the same information.
- **`plan_import` dropping its `PASS_NAME_RE` import looks like a lost pattern check.** The
  pattern check moved inside `group_name_error`, which `plan_import` now calls; the pattern
  message is unchanged and `test_every_group_writing_entry_point_shares_one_validator` covers it.
- **The block comment "the pass graph's six verbs (D15)" that the spec asked to be recounted to
  eight.** It was not recounted to eight — it was rewritten to carry no number at all ("the pass
  graph's verbs (D15; 091 and 092 added theirs)"), which is strictly better: a count in a comment
  is the kind of number that goes stale. Not a finding.

## Verdict

**FINDINGS.**

### Must change in code

1. **D3 — `_tab_row` maps a click by string.** `labels.index(clicked)` re-introduces the exact
   ambiguity the clause forbids; a document named like one of its groups makes that group's tab
   unreachable. Either have `text_tab_row` return the clicked index, or disambiguate the labels
   before the lookup. (`widgets/pass_graph._tab_row`, `ui_primitives.text_tab_row`.)
2. **D8 — the allow-overlap chain stops short of the ports.** Add
   `set_next_item_allow_overlap()` before each `##gport_` and `##gout_` `invisible_button`, so
   the chain the clause pins is complete rather than relying on submission order.
   (`widgets/pass_graph._draw_canvas`.)
3. **D13 — snapping re-applies its correction every frame and covers only the first dragged
   name.** Compute the snap offset from the un-snapped delta each frame rather than folding it
   into `drag.delta`, and consider all moving nodes' edges rather than `next(iter(drag.origin))`.
   (`widgets/pass_graph._snap`.)
4. **`arrange_graph` has no headless test.** The spec's `test_graph_view.py` list names "saves
   once and leaves no pass unplaced"; it exists only in the display-gated smoke stretch. Add it
   to `tests/test_graph_view.py` beside the other App-verb tests.
5. **D1 — make the compile seam the one-shot the spec describes**, or delete the unused
   `GraphViewState.compiled` field. A declared-and-never-read state field is the kind of thing a
   later reader trusts.
6. Smaller, at the maintainer's discretion: the grab-and-drop-on-empty GESTURE (`_drop` with no
   target) is untested — only `App.unwire` is; and a ghost's output dot has no drag rect, which
   D12's "drag sources like any other" promises.

### Must be recorded in the spec (Review history + two manual items)

The sixteen entries and the two manual-item rewrites in the "Deviations to record" section above.
The D20 doc checklist (nine items still owed, two landed) is the sanitize step's, not a code
finding.
