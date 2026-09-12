# 092 — post-implementation spec-fidelity audit, round 2

Auditor: spec-fidelity, round 2. Date: 2026-09-12. Anchor: round 1's report
(`reviews/post_spec_fidelity.md`), closed item by item against the fix commit `ed1c28b`
(`092: fold the post-implementation round 1`) and the docs commit before it.

## Coverage

Read: round 1's report end to end (all 575 lines); the spec's `## Design decisions` (D1–D20),
`## Files touched`, `## Manual verification` items 16 and 32, and the whole `## Review history`
including the new "Post-implementation round 1" paragraph; `shaderbox/widgets/pass_graph.py`
(`draw`, `_draw_canvas`, `_build_view`, `_tab_row`, `_snap`, `_drop`, `_group_prompt`,
`_draw_node`'s border ladder, the three `invisible_button` levels), `shaderbox/widgets/graph_state.py`
(all of it), `shaderbox/app.py` (`arrange_graph`, `drop_wire`, `unwire`, `commit_node_drag`,
`group_selection`, `dissolve_group`, `leave_group`), `shaderbox/project_session.py`
(`_pass_name_error`, `set_pass_groups`), `shaderbox/pass_graph.py` (`namespace_error`,
`group_name_error`), `shaderbox/pass_import.py` (`plan_import`), `shaderbox/document.py`
(`wiring_if_renamed`), `shaderbox/tabs/document.py` (`_draw_passes`),
`shaderbox/widgets/pass_list.py`, `shaderbox/ui_primitives.py` (`text_tab_row`),
`shaderbox/theme.py`; and the five doc targets of D20 by their actual text.

Test run: `uv run python -m pytest tests/test_pass_graph.py tests/test_graph_state.py
tests/test_graph_view.py tests/test_ui_regions.py tests/test_graph_persistence.py
tests/test_pass_import.py -q -p no:cacheprovider` → **101 passed in 1.80s** (round 1's
narrower set was 88; `test_pass_import.py` is new to this invocation and the round-1 fixes
added tests to `test_graph_view.py` / `test_graph_state.py` / `test_pass_graph.py`).

Not covered: the running app (no window manager), and the full `make gates` (out of this
auditor's remit; the named set above is the evidence).

## (a) The six code items

| # | Round 1's item | Verdict | Evidence |
|---|---|---|---|
| 1 | **D3 — `_tab_row` maps a click by string** (`labels.index(clicked)`; a document named like a group makes that group's tab unreachable) | **CLOSED** | `widgets/pass_graph._tab_row` now makes the root label distinct before drawing: `if not root_label or root_label in groups: root_label = next(label for label in _ROOT_LABELS if label not in groups)`, with `_ROOT_LABELS = ("document", "root", "all")`. `labels` therefore holds no duplicate and `labels.index(clicked)` is unambiguous. The comment above it states the mechanism truthfully ("The row keys and answers by NAME, so the root's label is made distinct from every group's before it is drawn"). `ui_primitives.text_tab_row` keeps its `-> str \| None` signature — the fix is at the caller, which is the cheaper of the two round-1 options. |
| 2 | **D8 — the allow-overlap chain stops short of the ports** | **CLOSED** | `_draw_canvas` now calls `imgui.set_next_item_allow_overlap()` before all four levels: `##graph_bg`, `##gnode_{key}`, `##gport_{key}_{slot}` and `##gout_{key}_{slot}`. Round 1's own false-trail note ("the allow-overlap chain") is preserved in the spec. |
| 3 | **D13 — snapping re-applies its correction every frame and covers only the first dragged name** | **CLOSED (first half) / OPEN (second half, by design)** | The re-application is gone: `NodeDrag` gained a separate `snap: Position` field; `_snap` sets `drag.snap = (0.0, 0.0)` at entry, recomputes the offset from `drag.raw()` (which is `origin + delta`, the un-corrected delta), and assigns `drag.snap` fresh. `NodeDrag.current()` = `raw() + snap`, `commit()` = `current()`. The docstring states the invariant: "`delta` is never corrected, so a node leaves a guide as soon as the cursor does; `snap` is recomputed from `raw()` every frame." The second half — "consider all moving nodes' edges rather than `next(iter(drag.origin))`" — was **not** taken: `_snap` still does `primary = next(iter(drag.origin))`. Round 1 phrased it as "and consider…", a suggestion inside a must-change item; the fix commit made the behaviour defect go away and left the scope. See (e) D13 for the spec/code agreement. |
| 4 | **`arrange_graph` has no headless test** | **CLOSED** | `tests/test_graph_view.py::test_arrange_graph_saves_once_and_leaves_no_pass_unplaced` exists and asserts all three: `saves.call_count == 1`, `all(entry.position is not None for entry in document.graph.passes.values())`, and `app.graph_view_for(document_id).fitted is False`. It runs headless (no display gate) and is in the 101-passing set. |
| 5 | **D1 — make the compile seam the one-shot, or delete the unused `GraphViewState.compiled`** | **CLOSED (by the second branch)** | The field is gone: `grep "compiled" shaderbox/widgets/` returns nothing, and `graph_state.GraphViewState` has no such field. `draw` keeps the unconditional `compile_pending_passes(document)` with the reason in a comment ("A no-op once every pass has been attempted, so calling it per frame costs nothing"), and the per-frame call is recorded in the spec's Review history. Round 1 offered either branch; the maintainer took the one that removes the dead state. |
| 6 | **Smaller, at discretion: the grab-and-drop-on-empty GESTURE untested; a ghost's output dot has no drag rect** | **OPEN (both, deliberately)** | The gesture: `_drop`'s `target is None` branch is still driven by no test — `test_unwire_writes_black_by_decision` still covers only `App.unwire`. The ghost output: `_draw_canvas`'s output loop still opens `if node.kind == "ghost": continue`. Both were marked discretionary by round 1 and the second is recorded as a deviation in the spec (see (b)). Not a finding. |

## (b) The sixteen deviations

Each row quotes the sentence from the spec's new `## Review history` paragraph
("Post-implementation round 1 (2026-09-12)") that describes what shipped, or cites the code
that made the deviation moot.

| # | Round 1's deviation | State | Quote / citation |
|---|---|---|---|
| 1 | D1's compile seam runs every frame, not once | **RECORDED** | "the compile seam runs every frame (a no-op once every pass is attempted) rather than once" |
| 2 | D3's tab row maps a click by the returned STRING | **FIXED in code** | Recorded as fixed, not as a standing deviation: "the tab row mapped a click by name (the root label is now made distinct from every group's before drawing)". Code: `_tab_row`'s `_ROOT_LABELS` fallback. |
| 3 | D5's box is a fixed-size node at its members' top-left | **RECORDED** | "D5's box is a fixed-size node at its members' top-left" |
| 4 | D6's `rank_layout` takes `gap_x` / `gap_y` | **UNRECORDED** | No sentence in the Review history names `gap_x` / `gap_y`. Round 1 itself graded this "DEVIATED (cosmetic)" and called it defensible; the spec's D6 still writes the five-parameter signature. Low value, but it is the one deviation of the sixteen with no line anywhere. See Findings. |
| 5 | D7 splits 2 channels, foreground after the merge | **RECORDED** | "D7 splits two channels and draws the foreground after the merge" |
| 6 | D8's allow-overlap chain stops at the node body | **FIXED in code** | Recorded as part of the hit fix: "the port rects declare allow-overlap". Code: the four `set_next_item_allow_overlap()` sites. |
| 7 | D11's border precedence has two more rungs | **RECORDED** | "D11's border ladder has `SELECT` and the group tint between the accent and the plain border" |
| 8 | D11's box reddens for a cycle culprit among its members | **RECORDED** | "…and a box reddens for a cycle culprit among its members" |
| 9 | D12 is two verbs, not one | **RECORDED** | "D12 is two verbs, `App.drop_wire` and `App.unwire`, the widget toasting the refusal they return" — this one sentence also carries deviation 10 below |
| 10 | D12's verb returns rather than toasts | **RECORDED** | same sentence: "the widget toasting the refusal they return" |
| 11 | D12's ghosts are not output-dot drag sources | **RECORDED** | "a ghost's output dot is not a drag source" |
| 12 | D13's `NodeDrag` has no `begin` | **RECORDED** | "`NodeDrag` is constructed directly (no `begin`)" |
| 13 | D13's snapping re-applies its correction each frame | **FIXED in code** | "the snap folded its correction into the drag's delta, so a node held at a guide never left it (now a separate offset recomputed from the raw drag each frame, `NodeDrag.snap`)". Code: `NodeDrag.snap` / `raw()` / `current()`. |
| 14 | D16's "the strip's two-click arm" was never on the menu | **RECORDED** | "D16's confirm step lives on the strip's tile ✕, not on the shared menu's Delete, so the canvas matches the strip" |
| 15 | D18's `wiring_if_renamed` rewrites in place | **RECORDED** | "`wiring_if_renamed` rewrites in place and restores" |
| 16 | The smoke stretch counts saves with `mock.patch.object`, not a `_count_saves` helper | **UNRECORDED** | No sentence names the smoke's save-counting mechanism. The invariant is asserted; only the named helper differs, and it now also has a headless twin (item (a)4). Cosmetic. See Findings. |
| — | (round 1's seventeenth bullet) `arrange_graph` has no headless test | **FIXED in code** | `test_arrange_graph_saves_once_and_leaves_no_pass_unplaced`. Correctly absent from the deviation paragraph, since it is no longer true. |
| — | (round 1's eighteenth bullet) D20 landed two of eleven | **FIXED in docs** | See (d): all eleven are now landed except the roadmap banner, which the sanitize step owns. Correctly absent. |

Fourteen of sixteen are closed (eleven recorded verbatim, three fixed in code). The two left
unrecorded — `rank_layout`'s two extra parameters and the smoke's save-counting mechanism — are
both cosmetic and both were graded so by round 1.

## (c) The two manual items

| Item | Proposed rewrite (round 1) | Shipped text | Verdict |
|---|---|---|---|
| 16 | "…the two wires of the loop turn red and no root-level NODE turns red; a box whose member is the planner's culprit does turn red." | "On the Uniforms tab, point `paint`'s sampler at `composite`: on the canvas, the two wires of the loop turn red and no root-level NODE turns red; a box whose member is the planner's culprit does turn red." | **REWRITTEN, verbatim** |
| 32 | "…ONE box draws at their members' top-left corner; Arrange pulls the members together." | "Label two passes on opposite sides of the chain with the same group name: ONE box draws at its members' top-left corner; Arrange pulls the members together. (mutations case 5 / triage D9.)" | **REWRITTEN** (`its` for `their`, the case reference kept — the substance is the proposal) |

## (d) The D20 checklist — eleven doc edits

Read from the actual doc text, not from the commit message.

| # | Doc edit owed | State | Evidence |
|---|---|---|---|
| 1 | `conventions.md`'s "nothing folds (091)" keeps its revisit trigger and gains one sentence scoping the no-folding half to the STRIP | **LANDED** | The entry now ends: "That no-folding half is about the STRIP: the graph view (092) contracts a group to a box whose ports are its boundary edges, and since the box is never a node the planner orders, convexity is not a rule there either. Revisit if a group-level fact appears that no member can hold…" — the sentence added, the revisit trigger intact |
| 2 | A new `conventions.md` entry recording D6 (a position is written only by a placement, never by a draw) | **LANDED** | In the new "The graph view is a second picture of the same wiring, and it stores one thing (feature 092)" entry: "**A position is written only by a placement -- a drag's release or Arrange -- through `ProjectSession.set_pass_positions`, never by a draw**" |
| 3 | A new `conventions.md` entry recording D1 (ports from the program, edges from the wiring) | **LANDED** | Same entry: "Ports come from the COMPILED program (`sampler_names`, through `pass_graph.node_ports`) and edges from `effective_wiring()`, two sources of truth on purpose: the wiring drops an unfilled sampler, so a port list built from it would have no dot to drop on, and a port list built from stored rows would draw a sampler the program no longer declares." Both D6 and D1 landed as one entry rather than two; the checklist says "A new entry records D6 … and D1", so one entry is what D20 itself asks for |
| 4 | `imgui-ui` skill §8 gains D8's two canvas rules | **LANDED** | §8's tail carries both: "**`set_next_item_allow_overlap()` goes on the item submitted FIRST, not the one on top.** … a canvas of stacked `invisible_button`s is a chain … (ShaderBox `widgets/pass_graph.py::_draw_canvas`)" and "**`begin_popup_context_item(str_id)` with an explicit id fires on a right-click ANYWHERE in the window.** … on one shared canvas child, a per-node menu must anchor with `None`, and the canvas's own menu is opened by hand" |
| 5 | `dev_flow.md`'s module map gains `widgets/pass_graph.py` | **LANDED** | "**`widgets/pass_graph.py`** — the graph canvas (feature 092), the Document tab's second view of the same passes…" |
| 6 | `dev_flow.md`'s module map gains `widgets/graph_state.py` | **LANDED** | "**`widgets/graph_state.py`** — the canvas's per-document transient state (`GraphViewState`: pan, zoom, scope, selection, the drag machines) held in `App.graph_views` and dropped in `forget_render_state`…" |
| 7 | `dev_flow.md`'s `pass_graph.py` entry describing the new pure half | **LANDED** | The entry no longer stops at `clamp_canvas_size`: "Since 092 also the graph canvas's pure half: `PassEntry.position` (bounded, validated through `with_positions`), `rank_layout`, `node_ports`, `group_boundary`, `bundle_output`, `refuse_drop`, `cycle_edges`, and the one namespace passes and groups share (`namespace_error` behind `group_name_error` and the session's pass-name check)." The `pass_list.py` entry was updated in the same sweep ("Since 092 the caption, the `strip \| graph` toggle and the `add pass` / `import...` row are the Document tab's") |
| 8 | Help panel's Passes section: a port exists because the shader declares a sampler | **LANDED** (already in round 1) | `help_content.py`: "On the graph view a port exists because the shader declares a sampler, so a new pass has none until you add one." |
| 9 | 070's spec gets its pointer | **LANDED** | `070_pass_reads/01_spec.md`: "The graph view returned in 092 as an opt-in SECOND view beside the strip, not as its replacement (`ai_docs/features/092_graph_view/03_spec.md`); the decision below stands for the strip." |
| 10 | The import dialog's flatten line | **LANDED** (already in round 1) | `popups/import_passes.py`: `caption_text("groups flattened")` |
| 11 | `roadmap.md` row for 092 | **LANDED** | Row 092 exists: "\| 092 \| graph_view \| in progress \| A second, opt-in view of a document's passes beside the strip…" with the spec path and the sibling docs |
| — | `roadmap.md` Active-context **banner** | **AS EXPECTED** | The banner is rewritten at the sanitize step and stays as is for now. It currently reads "As of 2026-09-12, 092 is implemented in two waves, post-implementation round 2 in flight, gates green with the smoke run; the whole canvas awaits his eyes." and "**Next: nothing is claimed.** 092 is the graph view…" — already 092-aware, so round 1's "the banner still reads 'the graph view he reopened is the candidate'" is stale. No finding |

**Eleven of eleven landed.** The banner is out of scope for this round by instruction.

## (e) Re-walking the decisions the fixes touched

Spec text vs code, after the fixes.

| Decision | Agree? | What each says |
|---|---|---|
| **D3 — the tab row** | **NO (the spec is wrong)** | D3 says "a click is mapped back to a scope by INDEX, never by the returned string, since a document named like a group would otherwise be ambiguous." The code maps by the returned string (`index = labels.index(clicked)`) and removes the ambiguity a different way — by making the root label distinct before drawing. The behaviour the clause protects is delivered; the mechanism it names is not the one that shipped. The Review history's fix sentence describes the code ("the root label is now made distinct from every group's before drawing"), so the fix is on the record, but D3's own sentence still asserts an index mapping the code does not do. **Replacement for D3's last clause** (the spec is the wrong side): "…and the root label is made distinct from every group's before the row is drawn (`_ROOT_LABELS`, the first of `document` / `root` / `all` no group carries), so the click maps back by name without ambiguity, which a document named like a group would otherwise create." |
| **D8 — ports allow overlap, the hit clamp** | **NO (the spec is wrong, on the clamp only)** | Allow-overlap: agree. D8 says "then one `invisible_button` per node body, each with `set_next_item_allow_overlap()`; then (W2) one per port", and all four levels now declare it — the chain the clause pins is complete. The hit box: D8 says "A port's hit box is `max(GRAPH_PORT_R * zoom, GRAPH_HIT_MIN)` in screen pixels while the drawn dot keeps scaling." The code is `min(max(SIZE.GRAPH_PORT_R * view.zoom, float(SIZE.GRAPH_HIT_MIN)), SIZE.GRAPH_PORT_ROW * view.zoom / 2.0)` — a ceiling of half the row pitch the spec does not name, which is the round-1 fix (below zoom 0.875 the floor overlapped the row below and a drop landed on the wrong sampler). The Review history records it ("the hit box is now clamped to half the pitch"), so the deviation is on the record; D8's own formula is stale. **Replacement for D8's hit-box sentence**: "A port's hit box is `max(GRAPH_PORT_R * zoom, GRAPH_HIT_MIN)` in screen pixels, capped at half the row pitch (`GRAPH_PORT_ROW * zoom / 2`) so two rows can never share a press, while the drawn dot keeps scaling." |
| **D12 — the cancel path; a ghost's output dot** | **NO (the spec is silent on the cancel path)** | The cancel path is new code with no clause: `_draw_canvas` opens with `released_elsewhere = not imgui.is_mouse_down(left) and not imgui.is_mouse_released(left)` and clears `node_drag` / `wire_drag` / `band_anchor` / `guides` on that or on `app.copilot_turn_active`, with the reason in a comment ("A gesture whose release the canvas did not see … is cancelled, never resumed: a stray later click must not write"). D12's own text names no cancellation. It IS in the Review history ("a gesture whose release the canvas did not see survived and a later stray click committed it … now cancelled at the top of the canvas frame, also under a copilot turn, which `begin_disabled` does not cover for raw `io` reads"), so the fact is durable — but a reader of D12 alone would not know the rule. **Suggested addition to D12** (a sentence, since the rule spans the drag as well as the wire): "A gesture whose release the canvas never saw is cancelled at the top of the frame, never resumed — the left button neither down nor released this frame, or a copilot turn in progress, which `begin_disabled` does not cover for raw `io` reads." The ghost's output dot: D12 says "A ghost's ports are drop targets and drag sources like any other"; the code still skips the output loop for a ghost. Recorded ("a ghost's output dot is not a drag source"), so the spec describes what shipped through the Review history — the D12 sentence itself remains an overstatement, and this is the one deviation round 1 already accepted as recorded-not-fixed. |
| **D13 — the snap as an offset; the drag from an unfilled port** | **NO (the spec is wrong, twice)** | The snap: D13 says "snapping aligns the moving node's left edge or top edge to any other visible node's within `GRAPH_SNAP_PX` screen pixels". The code aligns only `primary = next(iter(drag.origin))` — one node of the moving set, not "the moving node" as a set — and carries the alignment on a separate `NodeDrag.snap` recomputed from `raw()` each frame, the shape the whole clause turns on. Neither the scope nor the offset mechanism is in D13's wording. The offset IS in the Review history ("a separate offset recomputed from the raw drag each frame, `NodeDrag.snap`"); the single-node scope is in neither. **Replacement for D13's snap sentence**: "During a drag, snapping aligns the first dragged node's left or top edge to any other visible node's within `GRAPH_SNAP_PX` screen pixels, as a separate offset (`NodeDrag.snap`) recomputed from the un-corrected delta every frame — so a node leaves a guide as soon as the cursor does — and draws the guide line." The drag from an unfilled port: new behaviour with no clause. `_draw_canvas`'s port loop ends `elif pressed and node.kind != "ghost":` starting a `NodeDrag`, with the comment "An unfilled port is not a wire to grab: the press moves the node." Recorded in the Review history ("a press on an unfilled port was dead (it starts a node drag)"), absent from D13's own text. **Suggested addition to D13**: "A press that drags from a port with no wire to grab moves the node, so no part of a node's surface is a dead press." |
| **D14 — blank name, empty selection** | **NO (the spec is silent)** | D14 says the popup's "Enter or Create commits". The code refuses to commit a blank name or an empty selection: `_group_prompt`'s `if (committed and name and view.selection and app.group_selection(...) == "")`, with the comment "A blank name would mean 'no group' to the verb, which is Dissolve, not Create." Recorded in the Review history ("`Create` with a blank name dissolved the selection's group (blank and empty-selection are not committable)"). **Suggested addition to D14**: "A blank name or an empty selection is not committable: the blank would mean 'no group' to the verb, which is Dissolve, not Create." |
| **D17 — `plan_import`'s mirror** | **NO (the spec is narrower than the code)** | D17 names one function, "`pass_graph.group_name_error(group, pass_names) -> str` (the pattern, then the collision)", and says `plan_import` calls it. The code split the predicate: `namespace_error(candidate, pass_names, group_names)` is the shared half, `group_name_error(group, pass_names)` is the pattern then `namespace_error(group, pass_names, ())`, and `plan_import` calls **both** — `group_name_error(group, host_names)` for the group's half and `namespace_error(new, (), host_groups)` per copied pass for the mirror the round-1 fix added. `_pass_name_error` calls `namespace_error(name, (), {entry.group …})`. So the "one function both directions call" is `namespace_error`, not `group_name_error`. Recorded in the Review history ("`plan_import` checked only the group's half of the namespace (it now rejects a copied pass named like a host group, through one `namespace_error` both directions call") and in `dev_flow.md` ("`namespace_error` behind `group_name_error` and the session's pass-name check"). D17's own text is stale. **Replacement for D17's first sentence**: "One predicate decides the shared namespace, `pass_graph.namespace_error(candidate, pass_names, group_names) -> str`, and both directions call it: `group_name_error(group, pass_names)` checks the pattern then delegates, and `_pass_name_error` delegates for the mirror. Every entry point that writes a group calls `group_name_error` — `set_pass_groups` (hence `set_pass_group`, the modal and the copilot) and `pass_import.plan_import`, which also calls `namespace_error` per copied pass so a bundle cannot land a pass named like a host group." |
| **D10 — `Leave group` through `App.leave_group`; Dissolve of nothing** | **NO (the spec is silent on both)** | `Leave group`: D10 says `pass_menu_items` "keeps the strip's two gates (Delete only while `len(document.passes) > 1`; Leave group only while the entry carries one)" and says nothing about the write path. The round-1 fix routed it through `App.leave_group` (`app.py`, docstring: "the strip's and the canvas's shared menu item, routed here so every canvas write is an App verb"), and `widgets/pass_list.py` now calls `app.leave_group(document_id, name)`; the no-write gate `test_the_widget_makes_no_session_write_of_its_own` was widened to walk both modules and to forbid `"set_pass_group("` as well. Recorded in the Review history ("`Leave group` was a session write reached from the canvas (now `App.leave_group`, and the no-write gate also reads the strip's module)"). Dissolve of nothing: `App.dissolve_group` now returns `""` before any write when `not group or not members`, pinned by `test_dissolve_of_nothing_and_leave_group_write_as_they_should` (`saves.call_count == 0`, "a dissolve of nothing saved"); D16's "Dissolve on a box menu is `set_pass_groups(members, "")`" says nothing about the empty case. Recorded ("`dissolve_group` of nothing saved"). **Suggested addition to D10**: "`Leave group` writes through `App.leave_group`, not `set_pass_group` at the menu, so the shared menu makes no session write of its own from either surface — pinned by the no-write gate, which walks `widgets/pass_graph.py` and `widgets/pass_list.py` alike." |

Six of the seven decisions carry a live spec/code disagreement. In every case the CODE is right
and the SPEC's clause is what needs the edit; every one of them is already described truthfully
in the Review history paragraph, so the spec as a whole describes what shipped — the stale
sentence is in the D-clause, which a reader reaches first.

## (f) The "Files touched" test list — round 1's two gaps

| Test round 1 found missing | Exists? | Asserts what it names? |
|---|---|---|
| `arrange_graph` saves once and leaves no pass unplaced, headless | **yes** — `tests/test_graph_view.py::test_arrange_graph_saves_once_and_leaves_no_pass_unplaced` | **yes**: `saves.call_count == 1`, `all(entry.position is not None …)`, plus `fitted is False` (D9's "then fits the current scope", which the spec's line does not name and the test asserts anyway) |
| The widget no-write gate covering the shared menu | **yes** — `tests/test_graph_view.py::test_the_widget_makes_no_session_write_of_its_own`, widened | **yes**: it now loops `for module in (pass_graph, pass_list)` and greps each module's source for `set_sampler_source`, `set_pass_positions`, `set_pass_groups` and `set_pass_group(` — four forbidden names over two modules, where round 1 found three over one. The comment states why ("The strip's menu is shared with the canvas, so it is held to the same rule"). The grep is its own falsifier |

Round 1's other PARTIAL rows were not in this round's remit and are unchanged: the
grab-and-drop-on-empty gesture is still untested (item (a)6), and the "two `pass_list.draw`
tests" clause still has no test that drives `pass_list.draw` and asserts the tiles alone.

Test count: **101 passed in 1.80s** over the six named files.

## False trails

Checked to the primary artifact and found not to be findings.

- **"The roadmap banner still names the graph view as an unclaimed candidate."** Round 1 said
  so; it is no longer true. The banner reads "As of 2026-09-12, 092 is implemented in two waves,
  post-implementation round 2 in flight…". Quoting round 1 without re-reading `roadmap.md` would
  have produced a false ABSENT.
- **`conventions.md` appears to have one 092 entry where D20 asks for two.** D20's own wording is
  "A new entry records D6 … and D1", singular. One entry covering both is what was asked.
- **`_tab_row` still calling `labels.index(clicked)` looks like item 1 unfixed.** It is not: the
  duplicate that made `index` ambiguous is removed before the list is built, so the lookup is
  total. The remaining gap is the spec's wording, not the behaviour.
- **`_snap` still reading `next(iter(drag.origin))` looks like item 3 unfixed.** The defect
  round 1 named ("re-applies its correction every frame", "a node held at a guide never left
  it") is gone — the correction lives on `NodeDrag.snap`, recomputed from the un-corrected
  `raw()`. Only the suggested widening of scope was declined.
- **`GraphViewState.compiled` looks deleted without the one-shot landing.** That is the fix, not
  a miss: round 1 offered "make it the one-shot, **or** delete the unused field", and the
  per-frame call is recorded as a deviation.
- **`namespace_error` looks like an unrecorded new public function in `pass_graph.py`.** It is
  the round-1 fix for `plan_import`'s half-check, recorded in the Review history and in
  `dev_flow.md`'s module map. The finding is D17's stale sentence, not the function.
- **Two `set_pass_groups` calls per Group-then-Dissolve gesture look like a broken one-save
  rule.** The one-save rule is per verb; `set_pass_groups` saves once for the whole set, which
  `test_group_selection_and_dissolve_are_one_write_each` asserts.
- **`test_pass_import.py` inflating the count from 88 to 101 looks like thirteen new tests.**
  It does not: this round's command adds a sixth file round 1 did not run, alongside the tests
  the fix commit added. The number is the command's, not a delta.

## Verdict

**Code: PASS.** All six round-1 code items are closed or closed-by-the-offered-alternative;
the two left open (the untested grab-and-drop-on-empty gesture, the ghost's output dot) were
marked discretionary by round 1, and the second is a recorded deviation. 101 tests pass over
the six named files.

**Spec: FINDINGS.** Two of substance and two cosmetic.

1. **Six D-clauses now contradict the code they describe** — D3's index mapping, D8's hit-box
   formula, D13's snap sentence, D17's one-function namespace, and the silences in D12 (the
   cancel path), D13 (the drag from an unfilled port), D14 (blank name / empty selection) and
   D10 (`Leave group`'s verb, Dissolve of nothing). Each is described truthfully in the Review
   history, so the spec as a document is honest; but a reader hits the D-clause first, and four
   of the six assert a mechanism that is not the one in the code. Replacement text for each is
   in section (e); every replacement edits the SPEC, since the code is right in all six.
2. **D6's `rank_layout(gap_x, gap_y)` is the one deviation recorded nowhere** — not in the
   Review history, not in the D6 signature. One clause: add `gap_x`/`gap_y` to D6's signature,
   or one line to the Review history.
3. Cosmetic: the smoke stretch's `mock.patch.object` vs the named `_count_saves` helper is also
   unrecorded; it matters less now that the invariant has a headless test.
4. Cosmetic: D2's "`draw(app, document_id)`" is still the two-parameter signature; the code is
   `draw(app, document_id, height)`, the round-1 fix for the sibling-height coupling (the
   docstring now says "The caller sizes it; the widget measures no sibling"). Recorded as a fix
   in the Review history ("the widget takes its height from the tab"), not as a signature.

Manual items 16 and 32: both rewritten as proposed. D20: eleven of eleven landed; the banner is
the sanitize step's and is already 092-aware.
