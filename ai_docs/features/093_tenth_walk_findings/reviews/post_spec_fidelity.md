# 093 wave 1 — post-implementation spec-fidelity audit (commit f012074)

Anchors in order: the maintainer's verbatim findings (`00_findings.md` rows 1, 2, 3, 5) and his
verdict quoted in `02_research_brief.md`; the spec `01_spec.md` (T1-T6, S1-S15, Refinements,
Verification, Files touched); the record `03_graph_design.md` §2 (G1-G18) and §3. Every row is
demonstrated by a quoted line, a test run, or a probe; nothing is asserted from reading alone
where a probe was cheap. Probes live in the session scratchpad and touched no tracked file.

Test runs behind this report (GL env, unpiped):
`tests/test_graph_view.py tests/test_graph_state.py tests/test_graph_tab.py tests/test_theme.py
tests/test_canvas_fields.py tests/test_pass_settings_layout.py tests/test_anchored_note.py`
— **74 passed**. `ruff check` / `ruff format --check` clean (`make gates` is another reviewer's).

---

## Non-LANDED rows first

### 1. `00_findings.md`'s "Landed in" column still says `commit <sha>` — F, DEVIATES

The spec requires the ledger's Landed-in column filled in the same commit. What landed:

```
| 1 | UX | the graph view, a wire | ... | wave 1, commit <sha> |
```

Rows 1, 2, 3 and 5 all carry the literal placeholder `<sha>`, not `f012074`. A chicken-and-egg
artifact (the sha is unknown while the commit is being written), but the ledger now reads as a
pointer to nothing, and `dev_flow.md`'s own rule is that the ledger is updated in the wave's
commit. **Fix:** amend the four cells to `wave 1, commit f012074` (an amend, or the next wave's
commit if the maintainer prefers not to rewrite).

### 2. `_draw_wire`'s signature carries `zoom`, which the spec's S9 does not — B/C, DEVIATES

Spec S9: "`_draw_wire(dl, points, col, halo_col)` in the widget draws the halo … then the crisp
stroke". Shipped (`shaderbox/widgets/pass_graph.py:601`):

```python
def _draw_wire(
    dl: imgui.ImDrawList,
    points: tuple[Position, Position, Position, Position],
    zoom: float,
    col: int,
    halo_col: int | None,
) -> None:
```

A five-parameter signature. The extra `zoom` is *necessary* — the body needs it for
`max(1.0, SIZE.GRAPH_WIRE_W * zoom)` and the `* 3.0 * zoom` halo, and `points` alone cannot
supply it — so this is the spec's own literal claim being wrong, not the code being wrong. The
behaviour S9 specifies (halo first at `WIRE_W * 3.0 * z`, crisp `* 1.4 * z` when hovered,
`* z` otherwise, all floored at 1.0; the caller resolves the state) is met verbatim at
lines 612-619. **Fix:** one-line correction to S9's signature in the spec, or leave it as a
recorded refinement; no code change.

### 3. The channel row's stated mechanic is not the mechanic implemented — D, DEVIATES

Verification row: "the first occurrence of each `channels_set_current(k)` literal for k in 0..4
appears in ascending source order (halos, strokes, nodes, in-flight, overlays)". The shipped
source order is **0, 1, 4, 2, 3, 4**, because `_draw_wire` and `_draw_wire_x` are defined above
`_draw_canvas`:

```
615:        dl.channels_set_current(_CH_HALO)      # 0
618:    dl.channels_set_current(_CH_WIRE)          # 1
627:    dl.channels_set_current(_CH_OVERLAY)       # 4   <- breaks ascending order
1110:    dl.channels_set_current(_CH_NODE)         # 2
1339:        dl.channels_set_current(_CH_INFLIGHT) # 3
```

The row as written would be RED. The implementation replaced the literals with named constants
and `test_the_five_channels_carry_the_layers_in_order` asserts the constants' VALUES
(`layers == [0, 1, 2, 3, 4]`), that all five are used, that no sixth is reached for, and that
`channels_split(5)` / `channels_merge()` each appear once. That pins the layering decision more
directly than source order did (paint order follows the index, which is exactly what the
constants encode). **Fix:** amend the Verification row to describe what is actually asserted;
no code change.

### 4. A stale comment names the deleted self-loop — C (G8), DEVIATES

`shaderbox/widgets/pass_graph.py:316`, inside the scoped-view edge build:

```python
                if source is None or source == name:
                    continue  # a self-read is the loop drawn on the node
```

G8 deleted `_draw_self_loop`; a self-read is now the badge-row glyph, not a loop. The line is
untouched by the diff (it predates the commit), but the commit is what made it false, and the
project's comment rule is "state what's non-obvious about the code AS IT IS NOW — never narrate
development history". The root-scope twin at line 415 carries no such comment. **Fix:** one-word
edit (`the glyph in the badge row`) or drop the comment.

### 5. The drag-lock gate's 4-line window can be satisfied by a comment — D, minor gate weakness

`test_the_drag_lock_is_passed_at_every_site` asserts `GRAPH_DRAG_LOCK_PX` appears within four
lines *below* each `is_mouse_dragging(` occurrence. Line 1183 is a COMMENT mentioning
`is_mouse_dragging`, and the window that follows it happens to contain the token from the next
real site. A future call site placed just above a commented mention would pass while omitting
the threshold. The break the commit reports (omit `lock_threshold` at the node-body site) did go
red, so the gate works today on the four real sites; this is a robustness note, not a false
green. **Fix (optional):** match `is_mouse_dragging(imgui.MouseButton_` so comments are excluded.

Everything else below is LANDED.

---

## A. The maintainer's four findings, in HIS terms

| # | His words | What the code does now | Verdict |
|---|---|---|---|
| 1 | "select an edge by clicking on it (in order to delete: by pressing del or by clikcing on the cross which we should render at the center of the bezier line)" | A wire is selected by a background press resolved against this frame's wire-distance hover (`pass_graph.py:1315-1322`, `if bg_pressed and not blocked: if view.hovered_wire is not None: view.selected_wire = view.hovered_wire`). Delete AND Backspace unwire it (`:1402-1412`, through `delete_allowed` + `app.unwire`). The ✕ is drawn at the cubic's `t=0.5` point (`bezier_point(*points, 0.5)`, `:1084`) and hand hit-tested on the press (`:1003-1018`). `test_a_wire_is_selected_by_a_click_and_deleted_by_the_key` and `test_the_wires_own_badge_unwires_it_and_the_press_is_nothing_else` pass. | **LANDED** — all three affordances he named, at the place he named |
| 2 | "Higlight elements: nodes, pins, edges on hover." | Probed live (scratchpad `test_probe_hover_draw.py`, passed): with the mouse on a port dot, `_draw_port_dot` receives `_u32(COLOR.GRAPH_HOVER)`; on a wire, `_draw_wire` receives `(hover_col, hover_halo)`; on a node body, `_draw_node` receives `hovered=True` for `p:c` and draws the inset halo (`:772-787`). All three kinds, one frame late by design (S3). | **LANDED** — nodes, pins and edges each change colour on hover |
| 3 | "No need to show the feedback edge … Instead, draw a little double looped arrow (where you draw \"xN\" at top right of the node card)" | Probed live (scratchpad `test_probe_fb.py`, passed): a pass whose only sampler is `u_prev` builds `edges: []` — no feedback wire at all (`:315-316`, `if source is None or source == name: continue`) — and `_draw_feedback_glyph` fires 4 times over 4 frames at the picture's top-right (`:841-848`, positioned `s1[0] - (badge_w + GRAPH_FB_GAP * z …), s0[1]`). Two open arcs via `path_arc_to` + `path_stroke(col, thickness)` in `badge_fg` (`:709-731`). | **LANDED** — the edge is gone, the double-ring mark sits where the `xN` badge sits |
| 5 | (his observation: the long `paint -> raymarch.u_paint` wire should ride a bus, no bus is visible; and `_fit` frames the nodes, not the bus) | Dissolved, not repaired, exactly as G3 says: there is no bus (`GRAPH_BUS_STEP` / `GRAPH_BUS_CLEAR` deleted from `SIZE`, `bus_y` absent from the widget — asserted by `test_the_loop_the_bus_and_the_module_constants_are_gone`), every wire is one cubic, and `_fit` now frames the union of the cards and 25 sampled points per wire (`:525-560`). `test_the_fit_frames_every_wire_not_only_the_cards` asserts all 75 points inside the fitted window and that ≥8 leave the cards' box, so the row still discriminates. | **LANDED** |

---

## B. T1-T6 and S1-S15

| Item | Symbol(s) | Literal claims checked | Verdict |
|---|---|---|---|
| **T1** tab kind | `editor_types.EditorTabKind = Literal["shader", "script", "lib", "graph"]`; `App.open_graph_for` (`app.py:1669`) building `EditorTab(path=self.paths.graph_json_for(document_id), kind="graph", document_id=document_id)`; `tabs/code.tab_label`'s graph branch at `:77-80` placed BEFORE `multi_pass = …`; `widgets/uniform._locate_uniform_declaration` now `app.get_current_session_if_exists()` | Both REQUIRED edits present and in the stated order. `test_opening_the_graph_twice_focuses_one_tab_and_it_is_session_less` asserts `is_tab_dirty is False`, `is_current_editor_dirty() is False`, `formatter_for("graph") is None`, `format_current_editor()` / `jump_to_next_error()` raise nothing, and no session at the graph path. `test_the_graph_tab_closes_and_dies_with_its_document` covers `close_editor_for_path` and `_on_document_deleted` keeping the lib tab | **LANDED** |
| **T2** dispatch | `tabs/code.draw:906-911` — `tab = app.active_tab` then `if tab is not None and tab.kind == "graph": _draw_graph_tab(app, tab); return`, above `current_path` and above the `ui_document is None` guard | `_draw_graph_tab` (`:876-897`) consumes `editor_focus_requested` under `not app.any_popup_open()`, sets `app.editor_errors = []`, sets `editor_focused` from `is_window_focused(FocusedFlags_.child_windows)`, consumes `editor_defocus_requested` with `set_window_focus(None)` — the same four moves the text body makes at `:992-996` / `:1105-1111`. `test_the_graph_tab_draws_and_leaves_the_error_list_empty` pins the error clear | **LANDED** |
| **T3** fills its host | `pass_graph.draw(app, document_id)` (`:938`) — no `height`; `begin_child("##pass_graph", size=imgui.ImVec2(0.0, 0.0), …)` (`:962`); `SIZE.GRAPH_MIN_H` deleted (`grep` finds it only in the test that asserts its absence) | Signature, size and token deletion all literal | **LANDED** |
| **T4** opening it | `App.open_graph_for(document_id, focus_editor=False)` with `if self.copilot_turn_active or document_id not in self.ui_documents: return`; `CommandSpec(CommandId.OPEN_GRAPH, "Open graph", _chord(K.g, K.mod_alt), C.EDITOR)`; handler `lambda: self.open_graph_for(self.current_document_id, focus_editor=True)` | Label, chord `Alt+G`, category, default GLOBAL scope (no scope argument, as `OPEN_SHADER`/`OPEN_SCRIPT`), frozen mid-turn — all literal. `test_the_open_graph_command_is_registered` | **LANDED** |
| **T5** Document tab row | `tabs/document._draw_passes:453-462`; `_entry_tab_active(app, document_id, kind)` at `:383-389` shared with the Script row (`script_active = _entry_tab_active(app, document_id, "script")`, `:422`); `standard_button("open##entry_graph")` + `set_tooltip("Open the pass graph")`; `pass_list.draw` below; `PassesView` / `PASSES_VIEW_LABELS` gone from `ui_regions.py`; `UIAppState.passes_view` gone; `tests/test_ui_regions.py` deleted | Every named element present, in the spec's order. `projects/dev/app_state.json` carries no `passes_view` key (`grep -c` = 0), as the spec claimed. `test_the_entry_tick_marks_this_documents_tab_of_this_kind` covers the predicate over both documents and both kinds | **LANDED** |
| **T6** smoke | `scripts/smoke.py` frame 43 `app.open_graph_for(multi)`; frame 47 `app.close_editor_for_path(app.paths.graph_json_for(multi))` + `assert not any(t.kind == "graph" for t in app.editor_tabs)`; frames 45-47 asserts untouched; frame 48's `set_current_document_id(canary_id)` unchanged | Literal. (The smoke itself is `make gates`' to run; the commit body reports it ran, 200 frames, not skipped) | **LANDED** |
| **S1** forks | `GRAPH_NODE_W: int = 136`; `GRAPH_DRAG_LOCK_PX: float = 4.0`; `GRAPH_HOVER = _P["fg_0"]`; no wire menu entry (`_node_menu` carries only Open/Dissolve/`pass_menu_items`/Group...) | `test_the_card_is_wide_enough_for_the_maintainers_own_longest_names` measures in a rig frame and asserts the 128 budget (110) still CUTS `u_distance_field` while 136's does not, with ≥4px slack on both budgets — the width's pin, red at 128 | **LANDED** |
| **S2** one wire identity | `graph_state.WireId = tuple[str, str]`; `_Edge` gains `owner` / `sampler` and `wire_id` property (`:122-138`); no `span` field anywhere; `revalidated_wire` called each frame at `:1030` | Box edges pass the member (`bp.member` owners), ghost-reader edges pass the reader `name` (`:329-341`) — both as S2 states. `test_a_wire_selection_no_drawn_edge_carries_clears` | **LANDED** |
| **S3** hover timing | Draw reads `view.hovered_*` (`:1075-1109`); the wire pass runs at `:1286-1303`, AFTER the button loop and BEFORE the background press action at `:1315`; the four fields written unconditionally at `:1309-1313` | Order verified by line numbers; `test_exactly_one_thing_is_hovered_and_the_rungs_are_in_order` rung (e) proves every field is written `None` off-canvas; `test_the_hover_fields_are_exactly_the_four_that_are_written` pins the field set | **LANDED** |
| **S4** press/release | Node click at `:1186-1194`: `is_item_deactivated() and is_item_hovered() and is_mouse_released(left) and view.node_drag is None and view.wire_drag is None and not blocked` — every clause the spec names, in that shape. `bg_pressed = imgui.is_item_clicked(imgui.MouseButton_.left)` at `:1140`, acted on at `:1315-1322`. Shift branches present; `_click` clears `selected_wire` on all three paths; the band's release sets `view.selected_wire = None` (`:1373`) | `test_a_wire_selection_and_a_node_selection_are_exclusive` covers node-click and band-release | **LANDED** |
| **S5** Delete | Read at `:1400-1412`, after the hit-test section, after the drag blocks, after `channels_merge()`, BEFORE `_canvas_menu` (`:1420`). Gate is the pure `delete_allowed(pressed, hovered, any_item_active, blocked, has_wire)` fed five live reads in one call; both `Key.delete` and `Key.backspace` | `test_the_delete_gate_is_true_on_exactly_one_of_its_thirty_two_states` over all 32; plus three behaviour pins (held press, group prompt, copilot turn), all passing | **LANDED** |
| **S6** the ✕ | Hand hit-test at `:1003-1018`, inside the press bookkeeping and ABOVE `blocked = frozen or view.press_blocked` (`:1019`); latch clear moved to the END (`:1426-1427`, `if not mouse_down: view.press_blocked = False`). `x_rect` half is `max(SIZE.GRAPH_WIRE_X_R * view.zoom, float(SIZE.GRAPH_HIT_MIN))` (`:1086`), written every frame the selected wire draws and `None` otherwise (`:1073`). Channels 0-4 exactly as listed; `channels_split(5)` once at `:1052`, `channels_merge()` once at `:1396` after the guides | Both required orderings present. `test_the_badge_wins_over_a_card_that_covers_the_wire` is the gate for the second, and the commit body records the break (`set_output_pass called once`) | **LANDED** |
| **S7** bring-to-front | One list sorted once at `:1096-1105` by `(selected, dragged)`, `view.node_order = [node.key for node in nodes]` (`:1109`), and the hit-test loop at `:1163` iterates the SAME `nodes` list | `test_the_selected_card_draws_and_hit_tests_last` asserts `view.node_order[-1] == "p:a"` | **LANDED** |
| **S8** the fit | `_fit(view, nodes, picture, avail)` samples `GRAPH_WIRE_HIT_SEGS + 1` points per edge through `_wire_canvas_points` (which calls `wire_points(..., 1.0)`) and unions them with `_bbox(nodes)`; `zoom = min(1.0, avail.x / w, avail.y / h)` | The 1.0 clamp is literal. `test_a_wires_canvas_points_are_its_screen_points_divided_by_the_zoom` pins the exactness the sampling rests on | **LANDED** |
| **S9** pure geometry | `graph_state.py` carries `wire_points` (no branch on `dx`'s sign), `bezier_point`, `wire_hit_threshold`, `wire_hit`, `WireState` + `wire_state`, `revalidated_wire`, `delete_allowed` — all importable without imgui (`test_graph_state.py` imports no imgui) | `_draw_wire`'s signature carries an extra `zoom` — see **non-LANDED #2**. Every other claim literal: halo `* 3.0 * z` on channel 0, hovered stroke `* 1.4 * z`, floors at 1.0, in-flight wire through `wire_points` on channel 3 in `ACCENT_PRIMARY` (`:1339-1347`) | **DEVIATES** (signature only) |
| **S10** feedback glyph | `_draw_badge` returns `w` (`:706`); `_draw_feedback_glyph` placed left of the badge by `GRAPH_FB_GAP * z` when one drew, flush otherwise; `path_stroke(col, thickness)` — two arguments, as the Refinements table requires; ghosts excluded (`node.kind != "ghost"`) | Probed live; glyph fires | **LANDED** |
| **S11** `ellipsize` public | Renamed in `ui_primitives.py`; four internal call sites, `popups/lib_picker/tree.py`, `tests/test_pass_settings_layout.py`, `tests/test_anchored_note.py` all follow. `grep -rn "_ellipsize" shaderbox tests scripts` → **no hits** | Budgets measured inside the pushed-font scope: name at `:822` between `push_font(font, …)` and `pop_font()`; port label at `:859-860` inside the `font_12` scope | **LANDED** |
| **S12** tokens | Every value literal in `theme.py`: `136 / 96 / 8 / 20 / 18`; new `GRAPH_WIRE_BOW 0.40`, `GRAPH_WIRE_MIN_OFF 24`, `GRAPH_WIRE_HIT_FLOOR 6` (with the comment naming the 6-under-7 pair), `GRAPH_WIRE_HIT_SEGS 24`, `GRAPH_WIRE_X_R 7`, `GRAPH_FB_SIZE 12`, `GRAPH_FB_GAP 2`, `GRAPH_DRAG_LOCK_PX 4.0`; deleted `GRAPH_BUS_STEP / GRAPH_BUS_CLEAR / GRAPH_LOOP_RISE / GRAPH_LOOP_REACH / GRAPH_MIN_H`; `GRAPH_HOVER = _P["fg_0"]`, the two halo alphas, `GRAPH_HOVER` in `_GROUP_TINT_EXCLUSIONS`, and all four invariants (`not in _accent_primaries`, `!= SELECT / STATE_ERROR / GRAPH_EDGE`). The `SIZE.GRAPH_*` comment block is rewritten with no bus and no loop | `_MIN_DIRECT_DX` / `_BEZIER_BOW` gone (test-asserted) | **LANDED** |
| **S13** lock at every site | Four `is_mouse_dragging(..., SIZE.GRAPH_DRAG_LOCK_PX)` (`:1154`, `:1196`, `:1242`, `:1279` — band, node body, port press, output dot) and one `get_mouse_drag_delta(..., SIZE.GRAPH_DRAG_LOCK_PX)` (`:1158`) | See **non-LANDED #5** for the gate's window shape; the four sites themselves are correct | **LANDED** (gate note only) |
| **S14** cursors | `app.py:276-277` creates both; `_draw_canvas` requests `hand_cursor` under `node_drag` (`:1325`) and `panning` (`:1332`), `crosshair_cursor` under `wire_drag` (`:1334`); `"glfw" not in source` asserted by `test_the_loop_the_bus_and_the_module_constants_are_gone` | `test_the_cursor_follows_the_gesture` reads `app.cur_cursor` after the frame, and asserts `None` at rest on a frame where `canvas_rect != (0,0,0,0)` | **LANDED** |
| **S15** click chooses the output | `App.choose_output(document_id, name)` opens with `self.set_panel_pass(document_id, "")` then the `set_output_pass` half with its toast; `pick_pass = ensure_shader_tab + choose_output`; `_click` calls `choose_output` (`:1516`); `_double_click` keeps `pick_pass(..., focus_editor=True)`; the ghost click still resets scope and selects the ghost | The Uniforms-pin clear is present and is the round-2 addition. `test_three_pixels_is_a_click_and_five_is_a_drag` also asserts `app.active_tab.kind == "graph"` after the click; `test_a_double_click_opens_the_passs_shader_tab` asserts `kind == "shader"` and the pass's path | **LANDED** |

---

## C. G1-G18 against the code (spec Refinements applied)

| Rule | Evidence | Verdict |
|---|---|---|
| **G1** one cubic, non-negative offset | `graph_state.wire_points`: `offset = max(SIZE.GRAPH_WIRE_MIN_OFF * zoom, SIZE.GRAPH_WIRE_BOW * math.hypot(dx, dy))`, returning `a, (a[0]+offset, a[1]), (b[0]-offset, b[1]), b`. `test_no_pair_of_endpoints_folds_the_curve` sweeps a 9×9 grid at three zooms | **LANDED** |
| **G2** no backward case | `grep -n "backward" pass_graph.py` hits only the module docstring's prose; no `if dx <` / `if backward` anywhere | **LANDED** |
| **G3** bus removed | Tokens deleted, `bus_y` absent; **spec S8 overrides** the rule's clause "no wire leaves the nodes' bounding box" — the fit frames the sampled curves instead, which is what the shipped `_fit` does | **LANDED** (per spec, which is the reference here) |
| **G4** hit test | `wire_hit_threshold` = `max(float(GRAPH_WIRE_HIT_FLOOR), GRAPH_WIRE_W * 2.0 * zoom)`; `wire_hit` does the bbox reject expanded by the threshold then the min point-to-segment distance over 24 segments; the loop at `:1286-1303` keeps the nearest under threshold. **Spec S2 overrides** the identity (`(owner, sampler)`, not the 4-tuple) and **S12** the token name (`_HIT_FLOOR`, not `_HIT_MIN`) — both as shipped | **LANDED** |
| **G5** selectable wire, ✕ + Delete | See finding 1 and S4/S5/S6 above. **Spec S4 overrides** "press and release with no drag" (selection is on press) and **S6** overrides the ✕-as-last-item mechanism (hand hit-test) — both as shipped. The ✕ geometry is the rule's: `BG_APP` disc, 1px `SELECT` ring, two `add_line`s at `0.5 * r`, thickness `max(1.0, GRAPH_WIRE_W * z)` | **LANDED** |
| **G6** exclusive hover, halos, no size change | Resolution at `:1309-1313`; per-dot colour swap (`hover_col if slot == hovered_port else port_col`, `:877`) with `r` unchanged; node halo inset by `GRAPH_WIRE_W * z`, select outermost then hover (`:772-787`); wire hover stroke `* 1.4 * z` over a halo. **Spec S1 overrides** `blue_b` → `fg_0`. `test_nothing_on_the_canvas_changes_size_on_hover` pins `node_size(port_count, box)` and `wire_hit_threshold(zoom)` signatures | **LANDED** (see D for the rung-enforcement wording) |
| **G7** three cursors | S14 above | **LANDED** |
| **G8** feedback glyph | Finding 3 above; **spec S10 overrides** `path_stroke`'s arity | **LANDED** |
| **G9** no arrowheads | `grep -in "arrow" shaderbox/widgets/pass_graph.py` → no hits. Nothing added | **LANDED (no code)** |
| **G10** endpoints at dot centres, no stub | `wire_points(xf.to_screen(_out_point(src, …)), xf.to_screen(_port_point(dst, …)), zoom)` — the bare endpoints, no radial offset anywhere in the widget | **LANDED (no code)** |
| **G11** card width + ellipsis | **Spec S1 overrides** 128 → 136. Every token of the record's table matches except `GRAPH_NODE_W`; `GRAPH_PORT_TOP 4`, `GRAPH_PORT_R 4`, `GRAPH_ROUNDING 6`, `GRAPH_GAP_X 64` unchanged as the table says. Name budget `(p1.x - p0.x) - 2 * GRAPH_PAD * z`, label budget `node.size[0] * z - label_x - GRAPH_PAD * z` with `label_x = 2 * r + 2 * z` — both the record's formulas | **LANDED** |
| **G12** five channels + bring-to-front | Constants 0-4 in the rule's order; sort by `(is_selected, is_being_dragged)`, NOT hover; `picture.nodes` rebuilt each frame so no persistent order | **LANDED** (row mechanic deviates — non-LANDED #3) |
| **G13** 4px lock, named, every site | S13 above; the commit body records the break (`5px did not read as a drag`, `set_pass_positions` 0 calls) | **LANDED** |
| **G14** bindings kept, only Delete added | `test_graph_view.py`'s pre-existing rows (drop-wire, media refusal, unwire, commit-drag, group/dissolve, arrange, copilot-turn press) all pass unchanged apart from the `open_graph_for` substitution and the new `pytestmark`; pan is still middle-or-Alt (`:1144-1146`), zoom `_ZOOM_STEP ** wheel` clamped, band, re-plug, drop-on-empty, `begin_popup_context_item(None)` menus, Fit/Arrange in the canvas menu with no key. **Spec S15 overrides** the "Select — left click, kept unchanged" line | **LANDED** |
| **G15** no routing | `grep -in "rout\|avoid\|bundl"` in the widget → no hits. Wires pass under cards by channel order alone | **LANDED (no code)** |
| **G16** no reroute node | `NodeKind = Literal["pass", "box", "ghost"]` — unchanged, no fourth kind; no waypoint, no dot | **LANDED (no code)** |
| **G17** Delete does not delete passes | The only key read is `delete_allowed(...)` gated on `view.selected_wire is not None`; with a node selection and no wire it does nothing. Pass delete stays in `pass_menu_items` (`_node_menu:1544`) | **LANDED (no code)** |
| **G18** state precedence | `wire_state(on_cycle, selected, hovered, dim)` — error, selected, hovered, dim, normal; the caller maps the state to `(col, halo)` and `_draw_wire` never reads it. **Spec S9 overrides** the signature. `test_an_error_wire_stays_red_however_it_is_touched` over all 16 combinations | **LANDED** |

### §3 "What stays as it is"

`git show f012074 -- shaderbox/pass_graph.py shaderbox/document.py shaderbox/project_session.py`
→ **empty**. So: the data model, `PassEntry.position`, `graph.json`, `effective_wiring()`,
`node_ports`, the box/ghost model, `rank_layout`, the compile seam and the position-writing
verbs are all untouched at the source. In the widget: every gesture still routes through an
`App` verb (`app.unwire`, `app.drop_wire`, `app.commit_node_drag`, `app.group_selection`,
`app.dissolve_group`, `app.choose_output`, `app.pick_pass`) — the two new writes (✕, Delete) both
call `app.unwire`, so `test_the_widget_makes_no_session_write_of_its_own` still passes.
`push_font(font, max(4.0, font.legacy_size * z))` unchanged. `_draw_port_dot`'s five shapes
unchanged. The node/port `invisible_button` + `set_next_item_allow_overlap` chain unchanged.
**LANDED — every bullet.**

---

## D. The Verification table, row by row

Every row has a test; all 74 pass. Rows whose mechanic differs from the written row:

| Row | Test (file::name) | Match | Break recorded in the commit body? |
|---|---|---|---|
| G1 no cusp | `test_graph_state.py::test_no_pair_of_endpoints_folds_the_curve` | grid, three zooms, `offset >= 0`, `cp0.x > cp1.x` when `dx < 0` — as written (adds `dist = 0` via the `0.0, 0.0` pair) | n/a |
| G1 continuity | `::test_the_curve_is_continuous_across_the_bus_boundary_it_replaced` | `dx` over `[-48, 48]` at `dy = 40`, bound `1.0 + 2 * GRAPH_WIRE_BOW` — literal | n/a |
| G4 threshold + floor | `::test_the_hit_threshold_is_floored_in_screen_pixels` | `6.0 / 6.0 / 7.5` exactly; adds the `FLOOR < HIT_MIN` clause | **yes** — "assert 0.75 == 6.0" |
| G4 flattening guard | `::test_the_flattening_misses_no_real_hit_on_a_long_backward_curve` | 200 points, labelled a regression guard as the row says | n/a |
| G18 precedence | `::test_an_error_wire_stays_red_however_it_is_touched` | all 16 | n/a |
| S2 revalidation | `::test_a_wire_selection_no_drawn_edge_carries_clears` | all three cases including `None` in | n/a |
| G6 no size change | `::test_nothing_on_the_canvas_changes_size_on_hover` | `inspect.signature` on both, literal | n/a |
| G8/G3/S12 removals | `::test_the_loop_the_bus_and_the_module_constants_are_gone` | five tokens + four symbols + `glfw` — all named | n/a |
| S13 lock | `::test_the_drag_lock_is_passed_at_every_site` | window-based (non-LANDED #5) | **yes** |
| G12 channels | `::test_the_five_channels_carry_the_layers_in_order` | **DEVIATES** — see non-LANDED #3 | n/a |
| G11 ellipsis | `test_graph_tab.py::test_the_card_is_wide_enough_for_the_maintainers_own_longest_names` | rig frame, both fonts, the 110 budget cutting and the 118/120 budgets not, ≥4px slack — literal, and it asserts the 128 case would be red | n/a |
| S8 fit | `::test_the_fit_frames_every_wire_not_only_the_cards` | `avail = (520, 200)`, 75 points, `_fitted_window` in canvas space — literal; adds a `stray >= 8` clause so the row keeps discriminating | **yes** — "a point at x=557.15 outside a fitted window ending at 552.0" |
| fit clamp | `::test_the_fit_never_zooms_past_one_and_shrinks_for_a_narrow_pane` | `(1225,600) → 1.0`; `(740,600) → 0.6 < z < 1.0` — literal | n/a |
| G5 select + Delete | `test_graph_view.py::test_a_wire_is_selected_by_a_click_and_deleted_by_the_key` | `wire_mids` click, `selected_wire`, `selection == set()`, one `set_sampler_source(..., NoSource())`; key and mouse in separate `_frames` batches via `_press_key`; `canvas_rect` asserted first in `_open_graph` | n/a |
| G5 the ✕ | `::test_the_wires_own_badge_unwires_it_and_the_press_is_nothing_else` + `::test_the_badge_wins_over_a_card_that_covers_the_wire` | one write, `output.call_count == 0`, `band_anchor`/`node_drag` None throughout, selection unchanged; the covered case uses the spec's **50px** shift | **yes** — "the release-frame node click fired through the latch" |
| S5 held press | `::test_delete_is_refused_while_a_press_is_held_on_the_canvas` | literal, both halves | n/a |
| S5 gate domain | `test_graph_state.py::test_the_delete_gate_is_true_on_exactly_one_of_its_thirty_two_states` | all 32 | n/a |
| S5 group prompt | `::test_delete_typed_into_the_group_prompt_is_refused` | one-shot `group_prompt`, not re-asserted, as the row requires | n/a |
| S5 copilot turn | `::test_delete_is_refused_during_a_copilot_turn` | literal | n/a |
| G6 rungs | `::test_exactly_one_thing_is_hovered_and_the_rungs_are_in_order` | all five sequences (a)-(e); (d) uses `set_pass_positions` + the 50px shift | **yes** for node-vs-wire — "rung (d), assert None == 'p:c'" |
| S3 field set | `::test_the_hover_fields_are_exactly_the_four_that_are_written` + (e) above | literal | n/a |
| S4 exclusivity | `::test_a_wire_selection_and_a_node_selection_are_exclusive` | node click and band release both covered | n/a |
| G13/S15 3px vs 5px | `::test_three_pixels_is_a_click_and_five_is_a_drag` | move and release in separate frames; adds the `active_tab.kind == "graph"` clause | **yes** — "5px did not read as a drag" |
| S15 double-click | `::test_a_double_click_opens_the_passs_shader_tab` | `kind == "shader"` and the pass's path | n/a |
| S7 node order | `::test_the_selected_card_draws_and_hit_tests_last` | literal | n/a |
| G7 cursor | `::test_the_cursor_follows_the_gesture` | `cur_cursor` read after the frame; rest-state asserted on a frame where `canvas_rect != (0,0,0,0)`; Alt+left substituted for middle-drag with the reason stated | n/a |
| G14 kept bindings | the pre-existing rows of `test_graph_view.py`, all green with the `open_graph_for` substitution and the new `pytestmark` | literal | n/a |
| T1-T6 | `test_graph_tab.py` rows 1-6 | every clause of the row has an assertion | n/a |
| T1 uniforms panel | `::test_the_uniforms_panel_opens_no_editor_over_the_graphs_own_file` | called directly inside a rig frame, as round 2 decided | n/a |
| T5 tick predicate | `::test_the_entry_tick_marks_this_documents_tab_of_this_kind` | all three cases | n/a |
| Theme | `test_theme.py::test_group_tints_are_stable_and_collide_with_nothing` | imports `_GROUP_TINT_EXCLUSIONS`, asserts membership, the accent-primary clause and the three `!=` — literal | n/a |

**Mechanics, all three obeyed.** `_open_graph` asserts `view.canvas_rect != (0,0,0,0)` before any
read; `_press_key` queues the key in its own `_frames` batch; `_drag_node` puts the move and the
release in separate frames with a comment naming the reason.

### The sixth break: is the implementer's claim true?

**Yes — verified by probe, not by argument.** I loaded a MUTATED copy of `pass_graph.py` whose
resolution puts the node rung FIRST (`view.hovered_node = node_hovered`, then the dot rungs
gated on it), installed it in place of the real module for one test, and ran the spec's row (a)
— the mouse parked 2px inside the card on the port `("c", "u_src")`. Result:

```
RAW (node_hovered, port, out, wire) at the port dot: (None, ('p:c', 0), None, ('c', 'u_src'))
fields: ('p:c', 0) None None None
1 passed
```

`node_hovered` is already `None` at the port dot, so the swapped resolution produces the same
four fields and row (a) stays green. imgui hands the overlap to the `gport_*` button submitted
last; the resolution's port-over-node rung has nothing left to decide there. The commit body
reports exactly this and says so honestly ("the rung is enforced by the submission chain").

**Should the spec's row and G6 be amended?** Yes, and the test already carries the corrected
wording in its docstring: *"The port-over-node rung is NOT breakable from the resolution chain …
What (a) pins is the submission chain's OUTCOME."* The Verification row still reads "Breaks to
try: swap the port and node rungs (a flips)", which is false, and G6's implementation paragraph
("its submission order already encodes 1-over-2") is actually correct as written — it is the
spec's break list that is wrong. **Fix:** strike "swap the port and node rungs (a flips)" from
the row and replace it with "(a) pins the submission chain's outcome; the resolution's port rung
is redundant with it". The test needs no change.

---

## E. Files touched

Every file the spec lists is in the commit, and every file in the commit is listed or implied.
Two files are in the commit and not in the spec's list — both named by the implementer and both
warranted:

- **`tests/test_canvas_fields.py`** — `_captions` was anchored by an END-relative index
  (`captions[-2]`) to a caption row that T5 retires. Re-anchored to `captions[2]` from the front,
  with the reason in the docstring. A consequence of T5, not scope creep; the alternative is a
  red test.
- **`tests/test_ui_prose_budget.py`** — S11 renames `_ellipsize` to `ellipsize`, and the budget
  walk's domain is *public* copy-bearing helpers, so three call sites entered the domain
  (`anchored_note`, `preview_cell`, `pass_graph._draw_node`). Each is added to `_UNMEASURABLE`
  with a reason naming the string as data, not authored copy. Warranted and minimal.

`tests/test_ui_regions.py` is deleted, as T5 requires. `shaderbox/ui_regions.py` keeps its
`StrEnum, auto` import — still used by `DocumentTab` / `ChannelView`; ruff is clean.

**LANDED.**

---

## F. The required doc edits

| Required | Quoted | Verdict |
|---|---|---|
| 092 D2 pointer | "**D2. Where it lives.** REVERSED by 093 T1-T5: the canvas is a third editor-tab kind … What follows describes the shipped 092 shape." | **LANDED** |
| 092 D10 click-half pointer | "(REVERSED by 093 S15 for the CLICK only: inside the editor pane `pick_pass` activates the shader tab … The menus below stand.)" | **LANDED** |
| 092 Review-history entry | "**Reversed by 093 wave 1 (2026-09-14): D2 and D10's click half.**" … "Everything else in this spec stands, D10's menus included." | **LANDED** |
| `conventions.md` rewritten 092 bullet | "**The graph view is a second picture of the same wiring, it lives in the editor pane, and it stores one thing (features 092, 093).**" — carries the tab home, the one-bezier construction, the exclusive hover model and the click-chooses-the-output rule, and the "revisit the canvas's home" clause is replaced by "Revisit if a gesture needs state the canvas cannot rebuild from the document each frame" | **LANDED** — all four required facts plus the resolved clause |
| `conventions.md` editor-tab trigger clause | "a 4th EDITABLE `kind` lands -- a non-editable kind does not fire it, since it brings no session and so no dirty state, no formatter and no flush (the graph is the first)." | **LANDED** |
| `dev_flow.md` four module-map entries | `widgets/pass_list.py` ("since 093 that caption row is an entry-point row like the Script one"); `widgets/pass_graph.py` ("a second picture of the same passes in its own editor tab (`draw(app, document_id)`…)"); the `tabs/` bullet covering `document.py` ("its two entry-point rows (Script, Passes) each carry an `open`") and `code.py` ("since 093 -- a branch above the session fetch that hands a `graph` tab to `pass_graph.draw`") | **LANDED** — plus a fifth, `widgets/graph_state.py`, listing the pure geometry |
| `help_content.py` phrase | "On the graph tab a port exists because the shader declares a sampler" | **LANDED** |
| `pass_list.py` docstring | "The caption row and the add / import row are the Document tab's; the graph has its own editor tab and draws neither (093 T5)." | **LANDED** |
| ledger "Landed in" | `wave 1, commit <sha>` on rows 1/2/3/5; row 4 `— (delegated to its own feature)` | **DEVIATES** — non-LANDED #1 |
| roadmap row + banner | Row: `| 093 | tenth_walk_findings | in progress |`. Banner: "As of 2026-09-14, 093 wave 1 is implemented and awaits its post-implementation review." + "**Next: the post-implementation review of 093 wave 1**" + the four tuning forks under "Awaiting his eyes" | **LANDED** |
| spec Status line | "Status: **wave 1 implemented, post-implementation review next.**" | **LANDED** |

---

## False trails — checked and found landed exactly

- **`_Edge.span` leaking on.** `grep -n "span" pass_graph.py` → no hits. Gone as S2 requires.
- **A `backward` branch surviving in the wire draw.** Only the docstring's prose uses the word.
- **A self-loop edge still being built.** Probed: a `u_prev`-only pass builds `edges: []`.
- **The ghost/box edge owners.** Box edges carry `bp.member`, ghost-reader edges carry the
  reader's own `name` — S2's exact wording, at `:329-341` and `:396-404`.
- **`projects/dev/app_state.json` carrying the retired key.** `grep -c passes_view` → `0`, as the
  spec claimed, so no sandbox hand-fix was owed.
- **`_ellipsize` left behind anywhere.** Zero hits across `shaderbox`, `tests`, `scripts`.
- **`GRAPH_MIN_H` still read.** Only in the test asserting its absence.
- **The record's §3 "what stays" being quietly touched.** `git show f012074 --` on
  `pass_graph.py`, `document.py`, `project_session.py` is empty.
- **T2's focus bookkeeping diverging from the text body's.** Compared line for line against
  `tabs/code.py:992-996` and `:1105-1111`: the same four moves, plus the `editor_errors` clear
  T2 adds on purpose.
- **A fifth `is_mouse_dragging` site without the lock.** Four real sites, all carrying it.
- **`ruff` / `ruff format`.** Clean (314 files formatted, all checks passed).

---

VERDICT: PARTIAL — 5 non-LANDED rows: the ledger's `<sha>` placeholder (F, the only one the
maintainer would notice); S9's `_draw_wire` signature (spec text wrong, code right); the channel
Verification row's mechanic (spec text wrong, test stronger); the stale "loop drawn on the node"
comment at `pass_graph.py:316`; and the drag-lock gate's comment-permeable window. Four of the
five are doc/spec corrections, one is a one-line comment edit; none is a behaviour defect, and
all four of the maintainer's findings are met in his own terms.
