# 093 wave 2 (1974aa0) — spec fidelity and conventions

Read-only audit of commit `1974aa0` against, in order: the maintainer's verbatim findings 6-11
(`00_findings.md`), the spec's Wave 2 block (`01_spec.md` W2-1..W2-6 + its Tests/Docs
paragraphs), `CLAUDE.md`'s code rules, `conventions.md ## Code rules` + the graph bullet in
`## Design decisions`, and `/imgui-ui` §1-§3 + §8. Every changed file read end to end. The gate
was run once, at the end.

**Non-clean rows come first in each section.** `SPECULATION` marks anything not demonstrated.

---

## Non-clean rows

| # | Row | Verdict | The quoted line |
|---|---|---|---|
| N1 | C. comment rule — history narration, `ui_models.py:272` | **VIOLATION** | `# the current document's shader tab exactly as before the key existed.` — the rule is "state what's non-obvious about the code AS IT IS NOW — never narrate development history". "before the key existed" is a fact about the repo's past, not about the field. The sentence survives the cut: "An absent key is an empty list, which falls back to the current document's shader tab." |
| N2 | C. comment rule — temporal marker, `pass_graph.py::_drop` | **SMELL** | `how a wire is replaced now that an input pin is not a drag source (093 W2-4).` — "now that" narrates the change. Same class as N1, milder: strike "now that" → "which is how a wire is replaced, an input pin not being a drag source (093 W2-4)". |
| N3 | D. the spec's Status line | **VIOLATION** (inherited, not fixed by this wave) | `01_spec.md` lines 3-5 read `Status: **wave 2 in progress ... wave 1 landed (f012074,` / `ccdf6d1).**` / `maintainer's hands-on pass, whose findings open wave 2.**` — the third line is an orphaned tail of the *replaced* sentence, left by `b6cb229`'s rewrite (`git show b6cb229 -- 01_spec.md` shows the old line's head removed and its tail kept). It reads as a second, unterminated Status clause with a stray `**`. Also: the Status still says "wave 2 **in progress**" after the wave landed; per the Docs paragraph this commit owns the spec's wave list, so it owns the Status too. |
| N4 | A/D. finding 9's research readout in the ledger row | **DEVIATES** | The row asserts `Blender and Houdini all detach by dragging the wire off the input with a ghost to the cursor`. `research/B_mouse_controls.md` Q6 puts Houdini at `Click the connected input, then click empty space to disconnect; or click-drag it elsewhere to re-plug`, and Q7's "Drop on empty from a wire" column gives Houdini `n/a (click-based, not drag-and-drop-to-empty)`. Houdini is a CLICK schema, so it does not belong in the drag-with-ghost list — which is the one claim the reversal is argued against. The row also says `imnodes and ImNodeFlow leave the gesture out`; the research has imnodes at `EnableLinkDetachWithDragClick` (opt-in per-pin flag), i.e. present-but-off, not absent. ImNodeFlow's `Not modeled` is accurate. |
| N5 | C/D. "every reference" overclaimed in two docs and the commit body | **DEVIATES** | `conventions.md`: `That goes against every reference the research read -- all of them detach by dragging the wire off its input`. The research's Q8 scopes it precisely: `Every reference **that draws wires from a filled input** treats "press the filled input" as ...` — which by its own Q6 table excludes imnodes-by-default and ImNodeFlow. Same sentence in the roadmap banner (`against every reference the research read`) and the commit body (`every reference detaches by dragging the wire off its input`). The correct form is the research's own: every reference that offers the gesture at all. |
| N6 | B. W2-5's label budget, spec says 126 | **the SPEC is wrong; the code is right** | Code: `label_budget = node.size[0] * z - label_x - SIZE.GRAPH_PAD * z` with `label_x = 2 * r + 2 * z`. At z=1: `136 - (2*4 + 2) - 4 = 122`. The spec's `(128 and 126)` drops the trailing `GRAPH_PAD` term (136 - 10 = 126). 128 for the name budget is right (`136 - 2*4`). The test's independent formula agrees with the code: `label_budget = float(SIZE.GRAPH_NODE_W - 2 * SIZE.GRAPH_PORT_R - 2 - SIZE.GRAPH_PAD)` = 122. So: **the spec's 126 is corrected to 122**; no code change is owed. |
| N7 | D. the roadmap banner dropped the "Still unseen" list | **SMELL** | Wave 1's banner carried `Still unseen: 091's outline, the import dialog, the gear's group row; 090's `Auto | Fixed` control.` Wave 2's rewrite drops it with nothing said. Those four are unrelated to wave 2 and there is no evidence in this commit that he saw them, so this is a silent loss of the one list that tracked un-eyeballed work across features. (The banner is a rewrite-in-full block, so dropping is the mechanism by which it happens; naming it so the maintainer can confirm.) |
| N8 | C. American spelling — a roster prefix gap, not a wave-2 regression | **CLEAN for this diff; SMELL for the gate** | `tests/test_prose_spelling.py`'s `_BRITISH_WORDS` has `"centre"`, matched as `\bcentre`, which does not match `centring`. Live hits in tracked, non-excluded files: `shaderbox/pass_graph.py:646` `# every column is measured, so the vertical centring is applied in a second pass.` (introduced by `f04821a`, 092 — **not this diff**). `git log -1 -S centring -- shaderbox/pass_graph.py` → `f04821a`. `centrali*` / `colouri*`: no hits outside `shaderbox/resources/editor/` (excluded by design). The gate has a hole it does not know about; wave 2 did not widen it. |
| N9 | B. Tests paragraph — one undisclosed test-mechanics change | **SMELL** | `test_a_press_that_spans_a_copilot_turn_never_becomes_a_gesture` gained `_let_the_double_click_lapse(app)` at line 260. Forced, not optional: the wire now starts at an OUTPUT dot, and a second press on that same dot inside imgui's double-click window would read as the double-click that opens the shader tab. The commit body discloses the `out_rects` deviation by name but not this one; the helper is pre-existing (line 305) so nothing new was invented. |

---

## A. His six findings, in his terms

| # | His words | What the code does at the place he pointed at | Verdict |
|---|---|---|---|
| 6 | "Put open graph button to the document panel (new the script button)" (voice: NEAR the script button) | `tabs/document.py::_draw_entry_points` now ends the Script row with `imgui.same_line(spacing=float(SPACE.LG))` / `_entry_row_label(graph_active, "Graph")` / `if standard_button("open##entry_graph"):` — one row, the two `open`s side by side with a 16px gap. `_draw_passes` is back to `small_caption(app.font_12, "Passes")` with no tick and no button. The button he asked to move is literally the same widget id (`open##entry_graph`) and handler (`app.open_graph_for(document_id, focus_editor=True)`), relocated. | **LANDED** |
| 7 | "the opened graph tab is not remembered after the application restart (probably other tabs as well. We should preserve the opened tab after restart)." | `UIAppState` gains `editor_tabs: list[TabRecord] = []` + `active_tab_index: int = 0`; `App.save` writes `self.app_state.editor_tabs = tab_records(self.editor_tabs)`; `App._init` restores through `tabs_from_records(...)` and sets `active_tab_index` clamped + `tab_select_pending = True`. His "probably other tabs as well" is met in full: the round trip covers `shader`, `script`, `graph` and `lib` (`test_the_open_tabs_round_trip_through_the_app_state`), and `test_the_saved_tabs_reopen_on_a_fresh_app` drives a real second `App` over the same project dir and asserts the graph tab is among what comes back (`assert "graph" in saved_kinds and "script" in saved_kinds`). | **LANDED** |
| 8 | "the icon, representing a node's self feedback loop is ugly, remove it. Let's for now keep only this \"xN\" tooltip." | `_draw_feedback_glyph`, `_FB_RING_R`, `_FB_GAP_ARC`, `_FB_ARC_SEGS`, `SIZE.GRAPH_FB_SIZE` and `SIZE.GRAPH_FB_GAP` are gone, as is the `import math` they were the last reader of. The `xN` pill he asked to keep stays: `_draw_badge(dl, (s1[0], s0[1]), True, f"x{node.runs}", z, badge_bg, badge_fg)`. `_draw_badge` drops its `-> float` return (`badge_w` was its only reader). The `prev` port's own double ring (092 D11) is untouched — a port, not the mark he objected to. | **LANDED** |
| 9 | "Let's disallow dragging by the input pin. Currently, I can drag an input pin and detouch an edge. I don't like this behavior... I specifically don't like that when I starting to drag the edge, the yellow ghost appears... we need to check how do others do this..." | The port-press branch is now unconditional on the node: `if pressed and node.kind != "ghost":` → `view.node_drag = NodeDrag(...)`, with the comment `# An input port is not a drag source at all (093 W2-4): its press moves the node, filled or not.` `WireDrag.grabbed` is deleted from the dataclass; `_drop` is down to `if target is None: return` + one `app.drop_wire`. Both removal paths he named survive: the mid-curve ✕ (`pass_graph.py:970 error = app.unwire(document_id, *view.selected_wire)`) and Delete (`:1360`, same verb). His "check how do others do this" was answered in the ledger row and the reversal is recorded as his call against the research — see N4/N5 for where the readout overstates it. | **LANDED** (the code); the readout **DEVIATES** |
| 10 | "Let's make the render previews a little bit larger. Decrease the gap between the canvas previews and the node's card border by 50% of the current gap." | `GRAPH_THUMB: int = 116` (was 96), `GRAPH_PAD: int = 4` (was 8). Side gap `(136 - 116) / 2 = 10.0`, was `(136 - 96) / 2 = 20` — halved. Top gap 4, was 8 — halved. Both halves of what he asked for: the picture larger AND the gap halved, since the card width is fixed so the two are the same move. | **LANDED** |
| 11 | "let's add alittle bit of a margin at the top: between the last uniform name and the bottom border, unless the gap is already the same as the left and right ones... it seems like it a little bit smaller." (the BOTTOM margin) | `SIZE.GRAPH_PORT_BOTTOM: int = 7`, added in `graph_state.node_size` under `if port_count:`. The last label's text bottom now clears the border by `GRAPH_PORT_ROW / 2 - 6 + 7 = 18/2 - 6 + 7 = 10.0`, against the label's left inset `2 * SIZE.GRAPH_PORT_R + 2 = 10` — his "unless the gap is already the same as the left and right ones" is exactly what was matched. A card with no ports pays neither pad, pinned by the new assertion `assert h0 == float(SIZE.GRAPH_PAD + SIZE.GRAPH_THUMB + SIZE.GRAPH_NAME_H + SIZE.GRAPH_PAD)`. | **LANDED** |

### Finding 9's reversal and trigger, where the spec says they go

| Pointer | Verdict | The line |
|---|---|---|
| 092 D12's pointer | **LANDED** | `092_graph_view/03_spec.md`: `**D12. Wires (W2).** REVERSED in part by 093 W2-4: an input pin is no longer a drag source at all (its press moves the node), so the re-grab half below is gone ... The output-dot half stands. What follows describes the shipped 092 shape.` The framing sentence is the right move — the paragraph below still describes the grab, and is now explicitly historical. |
| the record's G14 pointer | **LANDED** | `03_graph_design.md`: `**The "Re-plug" row is REVERSED by wave 2 (finding 9).** ... Trigger to revisit: he asks for drag-to-detach after living with the ✕. Every other row here stands.` Trigger present and in the "he asks for X" form the ledger's own convention uses. |
| the record's G8 pointer (Docs paragraph asks for it too) | **LANDED** | `**REVERSED by wave 2 (finding 8), on the maintainer's verdict after seeing it rendered: "the icon ... is ugly, remove it. ..."**` |
| the conventions bullet | **LANDED** (with N5's overclaim) | `An INPUT PIN is not a drag source at all: its press moves the node, filled or not, so a wire leaves only by that badge or the Delete key and is replaced by dropping a new one from an output onto the port (`drop_wire` overwrites). ... revisit if he asks for drag-to-detach after living with the badge.` Input-pin rule present, trigger present; the re-grab clause is gone (`grep -n "re-grab\|re-plug\|grabbed" ai_docs/conventions.md ai_docs/dev_flow.md` → no hits in either). No history narration in the bullet. |
| the ledger row's research readout | **DEVIATES** | See **N4**. |

---

## B. Every claim in W2-1..W2-6 and the Tests/Docs paragraphs

| Spec claim | Symbol | Verdict |
|---|---|---|
| W2-1: `_draw_entry_points` draws on the Script row after the play/stop toggle and a `SPACE.LG` gap | `imgui.same_line(spacing=float(SPACE.LG))` at `document.py:448`, after the `if present:` toggle block | **LANDED** |
| W2-1: `_entry_row_label(graph_active, "Graph")` + `standard_button("open##entry_graph")` | both, `document.py:449-451` | **LANDED** |
| W2-1: tooltip `"Open the pass graph"`, same handler | `imgui.set_tooltip("Open the pass graph")`; `app.open_graph_for(document_id, focus_editor=True)` | **LANDED** |
| W2-1: the Passes row goes back to plain `small_caption(app.font_12, "Passes")`, no tick, no button | `_draw_passes` body is `small_caption(app.font_12, "Passes")` between the two disable brackets | **LANDED** |
| W2-1: `_entry_tab_active` stays the one predicate | one definition (`document.py:383`), two call sites (`:423` script, `:424` graph) | **LANDED** |
| W2-2: `editor_tabs: list[TabRecord]`, `active_tab_index: int = 0` | `ui_models.py:273-274` | **LANDED** |
| W2-2: `TabRecord` in `editor_types.py`, a pydantic model beside `EditorTab`, `path: str` / `kind: EditorTabKind` / `document_id: str` | `class TabRecord(BaseModel)` at `editor_types.py:27`, fields exactly as specced, declared directly under `EditorTab` | **LANDED** |
| W2-2: `App.save` mirrors through a pure `tab_records(tabs) -> list[TabRecord]` | `self.app_state.editor_tabs = tab_records(self.editor_tabs)` (`app.py:2191`); `tab_records` takes `Iterable[EditorTab]` and touches no global | **LANDED** |
| W2-2: `App._init`, after the documents load and before `ensure_shader_tab`, restores through `tabs_from_records(records, document_ids, exists)` | `app.py:1477-1481`, the block sits after `seed_starter_document` and above the fallback; signature `(Sequence[TabRecord], frozenset[str], Callable[[Path], bool])` | **LANDED** |
| W2-2: drops a record whose path is gone or whose `document_id` names no loaded document; a lib tab has `""` and passes | `if not exists(path): continue` / `if record.document_id and record.document_id not in document_ids: continue` — the `and` short-circuit is what lets `""` through | **LANDED** |
| W2-2: sets `active_tab_index` clamped to the list and `tab_select_pending` | `max(0, min(self.app_state.active_tab_index, len(self.editor_tabs) - 1))`; `self.tab_select_pending = True` | **LANDED** |
| W2-2: "the fallback runs only when nothing was restored" | **the nesting is equivalent to the stated rule.** The code is `if not self.editor_tabs:` wrapping the existing `if current_document_id ... elif self.ui_documents:` pair, rather than a third `elif`. Behaviorally identical: a non-empty `editor_tabs` reaches neither `ensure_shader_tab` nor `set_current_document_id`, which is the whole content of the rule. A literal third `elif` would have required the restore to become the chain's FIRST branch with a `pass` body (`if self.editor_tabs: pass / elif ...`), which is strictly worse to read. The spec named a rule, not a syntax; the rule holds. Pinned from both sides: `test_the_saved_tabs_reopen_on_a_fresh_app` (restore wins) and `test_a_project_with_no_saved_tabs_still_opens_its_shader` (`assert len(fresh.editor_tabs) == 1` / `kind == "shader"`). | **LANDED** (not a deviation) |
| W2-2: sessions are not restored | nothing in the restore block touches sessions; the docstring says `No session is restored — the draw creates one lazily, as it does for a tab opened by hand.` | **LANDED** |
| W2-2: no migration; an absent key is `[]` | `editor_tabs: list[TabRecord] = []` is a plain pydantic default — no reader for an old shape, no shim | **LANDED** |
| W2-2: "the sandbox's `app_state.json` picks the key up as drift and is `git add`ed" | **DEVIATES, and the commit body corrects the spec.** `grep -o editor_tabs projects/dev/app_state.json` → no hit; `git status --short` → clean. The commit body states why and it is right: the smoke and the tests use throwaway project dirs, so nothing in this wave writes the sandbox's state file — it picks the key up on his next real launch+quit. The spec's sentence was a prediction about a write that does not happen here. No hard-rule breach (there is no unstaged `projects/dev/` drift to leave). | **spec DEVIATES; code CLEAN** |
| W2-3: `_draw_feedback_glyph` and `GRAPH_FB_SIZE` / `GRAPH_FB_GAP` deleted | all three gone, plus `_FB_RING_R` / `_FB_GAP_ARC` / `_FB_ARC_SEGS` and the now-unused `import math` | **LANDED** |
| W2-3: `_draw_badge` returns to returning nothing | `) -> None:` and the `return w` removed | **LANDED** |
| W2-3: the `prev` port's double ring stays | `_draw_port_dot` untouched by this diff | **LANDED** |
| W2-4: a press on any input port moves the node, exactly as an unfilled port's press does today | `if pressed and node.kind != "ghost":` — the `port.kind == "wired"` guard is gone, so the one branch serves both | **LANDED** |
| W2-4: `WireDrag.grabbed` deleted with the re-grab branch and `_drop`'s two `grabbed` paths | field gone from the dataclass; `_drop` has one write and no `grabbed` reference; `grep -rn grabbed shaderbox/` shows only unrelated focus-comment hits | **LANDED** |
| W2-4: a wire removed only by its ✕ or Delete | `pass_graph.py:970` (badge press) and `:1360` (`# ---- Delete unwires the selected wire ...`), both `app.unwire(document_id, *view.selected_wire)` | **LANDED** |
| W2-4: replaced only by dropping from an output onto the port, `drop_wire` overwrites | `_drop`'s single `app.drop_wire(...)`; pinned by `test_a_wire_dropped_on_a_filled_port_overwrites_its_source` (`u_src` goes `PassSource("b")` → `PassSource("a")`) | **LANDED** |
| W2-4: the copilot-turn and crosshair-cursor tests start their wire from an OUTPUT dot | both now read `view.out_rects[("p:a", 0)]` where they read `view.port_rects[...]` | **LANDED** |
| W2-4: a new test presses a filled input, drags 6px, asserts `node_drag` set, `wire_drag` None, no write | `test_an_input_pin_moves_the_node_and_never_carries_its_wire`: `_park(app, (start[0] + 6.0, start[1] + 6.0))`, `assert view.node_drag is not None`, `assert view.wire_drag is None`, `assert write.call_count == 0` over a `mock.patch.object(app.session, "set_sampler_source")` | **LANDED** |
| W2-4: trigger to revisit recorded | in G14, the conventions bullet and the commit body | **LANDED** |
| W2-5: `GRAPH_THUMB 96 -> 116`, `GRAPH_PAD 8 -> 4`; side 20 -> 10, top 8 -> 4 | tokens as stated; arithmetic re-derived above | **LANDED** |
| W2-5: "the name and label budgets follow the tokens (128 and 126)" | 128 ✓; **126 is wrong, the code and the test both give 122** | **DEVIATES — see N6; the spec is corrected to 122** |
| W2-5: the ellipsis test's slack assertions still hold | `test_the_card_is_wide_enough_for_the_maintainers_own_longest_names` passes in the gate; its budgets are computed from the tokens, so they moved with them | **LANDED** |
| W2-6: `SIZE.GRAPH_PORT_BOTTOM: int = 7`, added in `node_size` when the card has ports | `theme.py:339`; `graph_state.node_size` under `if port_count:` | **LANDED** |
| W2-6: clears the border by the same 10px the left inset gives (`PORT_ROW / 2 - 6 + 7`) | `18/2 - 6 + 7 = 10.0` = `2 * GRAPH_PORT_R + 2 = 10` | **LANDED** |
| W2-6: `test_a_node_grows_one_row_per_port_and_a_box_is_wider` includes the term | `assert h2 == h0 + SIZE.GRAPH_PORT_TOP + 2 * SIZE.GRAPH_PORT_ROW + (SIZE.GRAPH_PORT_BOTTOM)` — the literal `4.0` was replaced by the token, which is the better form | **LANDED** |
| Tests: `TabRecord` round trip through `UIAppState.save` / `load` | `test_the_open_tabs_round_trip_through_the_app_state`, all four kinds + `active_tab_index == 2` | **LANDED** |
| Tests: `tabs_from_records` drops a vanished path and an unknown document, keeps a lib tab | `test_a_record_whose_file_or_document_is_gone_is_dropped` — and it also pins a dedup (`if any(tab.path == path for tab in kept)`) the spec never asked for; harmless and asserted | **LANDED** |
| Tests: an `app`-fixture test sets the three records and calls the restore verb, asserting the tabs and the active index | **DEVIATES, and the deviation is stronger than the spec.** There is no "call the restore verb" test; `test_the_saved_tabs_reopen_on_a_fresh_app` drives a real `App.save()` then constructs a second `App(project_dir=project_dir)` and asserts `[t.path for t in fresh.editor_tabs] == saved_paths`, the kinds, `fresh.active_tab.path == shader_path` and `fresh.tab_select_pending is True`. That exercises the whole restart path the finding is about, not a hand-set state through one verb. Accept. | **DEVIATES (upward)** |
| Tests: the token tests follow the values | `test_a_node_grows_one_row_per_port_and_a_box_is_wider` gained the two pads; no token literal left stale (`grep -rn "GRAPH_THUMB\|GRAPH_PAD\|GRAPH_PORT_BOTTOM" shaderbox/` — every reader goes through `SIZE.`) | **LANDED** |
| Tests (extra, unspecced): the malformed-row test | `test_one_malformed_tab_record_costs_only_itself` — real, not decorative: `model_salvage.drop_invalid` genuinely descends a `list` of nested models (`elif isinstance(value, list): ... kept.append(item)`), so `["/a", "/c"]` survives a `"kind": "not_a_kind"` sibling and `global_target_fps == 90` proves the file was not lost | **LANDED** |
| Tests (extra, unspecced): `GraphViewState.out_rects` exposed | disclosed in the commit body with its alternative (recomputing widget geometry in test code). It is written in the same `_draw_canvas` pass as `port_rects` and cleared beside it (`view.out_rects = {}`), so the two cannot drift | **LANDED** |
| Docs: 092 D12 pointer / the record's G8 and G14 pointers / the conventions bullet / `dev_flow.md`'s two entries / the ledger's "Landed in" / the roadmap banner | all present — see section D | **LANDED**, with N3, N4, N5, N7 |

---

## C. Conventions over the diff

| Rule | Verdict | Evidence |
|---|---|---|
| Comment rule | **VIOLATION** (N1) + **SMELL** (N2) | Every added comment/docstring enumerated below. |
| American spelling | **CLEAN** for the diff | `tests/test_prose_spelling.py` green (part of the gate). Diff-scoped grep for `centring` / `centrali` / `colouri`: no hits in this commit's added lines. The one `centring` in tracked source predates it (N8). The `centres` in `00_findings.md` row 10 sits under `ai_docs/features/`, which the test excludes by design ("they quote what was said at the time"). |
| Full type annotations, no `from __future__ import annotations` | **CLEAN** | `tab_records(tabs: Iterable[EditorTab]) -> list[TabRecord]`, `tabs_from_records(records: Sequence[TabRecord], document_ids: frozenset[str], exists: Callable[[Path], bool]) -> list[EditorTab]`, locals annotated (`kept: list[EditorTab] = []`). No `__future__` line added. |
| Imports at module top only | **CLEAN** | `git show 1974aa0 -- shaderbox/ \| grep -E '^\+ +(import\|from .* import)'` → nothing. The four new imports (`Callable, Iterable, Sequence`, `pydantic.BaseModel`, `tab_records, tabs_from_records`, `TabRecord`) are all at module top. |
| No `@staticmethod` / `@classmethod` | **CLEAN** | none added; `tab_records` / `tabs_from_records` are module-level free functions, which is the rule's positive form. |
| No suppressions | **CLEAN** | `git show 1974aa0 \| grep '^+.*\(noqa\|type: ignore\|pyright: ignore\)'` → nothing. |
| No `if TYPE_CHECKING:` | **CLEAN** | no hit in the diff. |
| Tokens only in `theme.py` | **CLEAN** | `GRAPH_THUMB` / `GRAPH_PAD` / `GRAPH_PORT_BOTTOM` defined in `theme.py::SIZE` and read only through `SIZE.`; the deleted `GRAPH_FB_SIZE` / `GRAPH_FB_GAP` left no orphan reader. `_BADGE_PAD` / `_BADGE_INSET` / `_FB_*` are draw-list geometry constants private to the widget, pre-existing pattern. No `push_style_color(Col_.button, …)` added; the Graph `open` uses the `standard_button` tier, correct per `/imgui-ui` §1 for "an ordinary verb — the default tier (\"open\", ...)". |
| `/imgui-ui` §3 (a state cue changes color, never size) | **CLEAN** | the Graph tick is the same `_entry_row_label` draw-list `add_line` the Script tick uses, with the comment `presence and color only, never size (/imgui-ui §3)`. It lands at `pos.x - _ENTRY_TICK_W` = 4px left of the label, inside the 16px `SPACE.LG` gap — no overlap with the play/stop button, and it perturbs no content size. |
| `/imgui-ui` §8 (version-pinned quirks) | **CLEAN** | no binding workaround added or removed; `imgui.same_line(spacing=...)` and `set_tooltip` are the shapes already used on the same row. |
| `TabRecord`'s placement + the import graph | **CLEAN** | `TabRecord` is a `BaseModel` in `editor_types.py`, imported by `ui_models.py` (`from shaderbox.editor_types import TabRecord`). `editor_types.py` remains a leaf: its only `shaderbox` imports are `shaderbox.editor.ffi` and `shaderbox.shader_source`, and its transitive closure is `{constants, editor.ffi, editor_types, shader_source}` — `ui_models` is NOT reachable from it, so the new edge adds no cycle. No `App` reference anywhere in the file (`grep -n 'App\b' shaderbox/editor_types.py` → nothing). |
| `tests/test_persistence_completeness.py`'s roster | **CLEAN** | the roster is per-MODULE, not per-field: `rostered = {"ui_models.py", "integrations.py", "tags.py", "favorites.py"}`, and `ui_models.py` is already in it, so the new field inherits the corruption battery rather than needing a row. The test is green in the gate, and its completeness half (`test_the_roster_covers_every_module_that_loads_a_persisted_store`) would have named a new loader module — none was added. |
| UI prose budgets | **CLEAN** | `tests/test_ui_prose_budget.py` green (703 passed across the four doc/prose gates; the 4 skips are its own named exemptions, all pre-existing: `help_marker`, `markdown_text`, `modal_window`, `parse_markdown_lines`). The two new strings are `"Graph"` and `"Open the pass graph"`. |
| No backward-compat / migration code | **CLEAN** | the absent key is a pydantic default; no old-format reader, no shim, no throwaway script. |
| `projects/dev/` never left unstaged | **CLEAN** | `git status --short` → empty. |

### Every added or changed comment and docstring

| Site | Verdict |
|---|---|
| `app.py` restore block, `# The tabs the last session had open (093 W2-2), before the fallback below: ...` | CLEAN — states the rule and the ordering as they are; the feature tag is the repo's convention. |
| `app.py` fallback comment, `... Open it here, unless the restore above already put tabs back.` | CLEAN — the added clause describes the guard that is now there. (The surrounding paragraph's "load() restores ... so `_on_current_document_changed` never fires" predates this diff.) |
| `app.py::unwire` docstring, `The wire into `consumer.sampler` removed, by its mid-curve ✕ or the Delete key (093 W2-4)` | CLEAN — replaced a now-false description ("grabbed off ... and dropped on empty canvas") with the current one. Exactly the right edit. |
| `editor_types.py::TabRecord` docstring, `A separate shape rather than making `EditorTab` a model: the live tab is a frozen dataclass ...` | CLEAN — a design rationale for a non-obvious duplication, not a history of the change. |
| `editor_types.py::tab_records` docstring | CLEAN — one line. |
| `editor_types.py::tabs_from_records` docstring, `Dropping rather than repairing is the point: a tab pointing at a deleted pass would eat its own edits ...` | CLEAN — states why the code is shaped this way, in the present tense. |
| `document.py::_draw_entry_points` header comment, `... Both sit on ONE row, so the two summoners are side by side.` | CLEAN — describes the row as it now is. |
| `document.py` `# No section caption: a document has exactly one script (048) and one graph, ...` | CLEAN. |
| `document.py::_draw_passes` `# A plain caption over the strip: the graph's summoner lives on the entry-point row beside the script's (093 W2-1) ...` | CLEAN — a cross-reference to where the button went, which a reader of this function genuinely needs. |
| `theme.py` `# The gap between the name and the first port row, and the one under the LAST port row so its label clears the border by what its left inset gives it` | CLEAN. |
| `ui_models.py` `# ... exactly as before the key existed.` | **VIOLATION — N1.** |
| `graph_state.py::WireDrag` docstring, `An OUTPUT dot is the only source (093 W2-4): a press on an input port moves the node ...` | CLEAN — the invariant, present tense. |
| `graph_state.py` `# And where each OUTPUT dot's hit rect landed ... so this is where a headless test starts one.` | CLEAN — names the reader, which is the non-obvious part of a field only tests read. |
| `pass_graph.py::_draw_badge` docstring (trimmed) | CLEAN — dropped the clause about the deleted glyph. |
| `pass_graph.py` `# An input port is not a drag source at all (093 W2-4): its press moves the node, filled or not.` | CLEAN. |
| `pass_graph.py::_drop` docstring, `... which is how a wire is replaced now that an input pin is not a drag source (093 W2-4).` | **SMELL — N2.** |
| test comments/docstrings (falsifier lines in all five new tests) | CLEAN — the repo's stated test convention is that each test names its falsifier, and each does (`Falsifier: restore the `port.kind == "wired"` branch -- ...`, `Falsifier: skip the restore block and the fresh app opens only the current document's shader tab`, etc.). None narrates a review round. |

No comment anywhere in the diff narrates a review round or a reviewer.

---

## D. Docs

| Item | Verdict | Evidence |
|---|---|---|
| 092 D12's pointer | **LANDED** | quoted in section A. |
| the record's G8 pointer | **LANDED** | quoted in section A. |
| the record's G14 pointer | **LANDED** | quoted in section A. |
| the conventions bullet: re-grab clause gone | **LANDED** | `grep -n "re-grab\|re-plug\|grabbed" ai_docs/conventions.md ai_docs/dev_flow.md` → no hits in either file (the six repo-wide hits are unrelated focus comments in `ui_primitives.py`, `popups/`, `app.py`, `copilot_chat.py`). |
| the conventions bullet: input-pin rule present | **LANDED** | quoted in section A. |
| the conventions bullet: no history narration | **LANDED** | it states the rule and the maintainer's reason, not "we used to allow X". The "every reference" overclaim is N5, a different defect. |
| the conventions bullet: prose | **SMELL** | the insert left a two-word orphan line: `... after living with the badge. A` / `single CLICK on a node chooses the output ...`. Same in `dev_flow.md`: `a plain `Passes` caption. `code.py` is the` and an over-long line at `... lives in `pass_graph.py`.`. Cosmetic — the repo wraps at ~100 and these rows do not. |
| `dev_flow.md` entry 1 (`widgets/pass_graph.py`) | **LANDED** | `wire drag FROM AN OUTPUT DOT ONLY (an input pin's press moves the node -- a wire is removed by its mid-curve ✕ or Delete and replaced by a drop onto the port)` |
| `dev_flow.md` entry 2 (`tabs/document.py`) | **LANDED** | `ONE entry-point row carries both summoners side by side (Script, then Graph), with the strip under a plain `Passes` caption.` |
| the ledger's "Landed in" for 6-11 | **LANDED** | all six rows read `wave 2 (<sha>)`; rows 1-5 keep `wave 1 (f012074)` / the delegation dash. The `<sha>` placeholder is the maintainer's to fill after this round, as briefed. |
| the roadmap banner: ≤200 words | **LANDED** | `tests/test_roadmap_shape.py` green; it measures `len(re.sub(r"<!--.*?-->", "", _banner(), flags=re.DOTALL).split())` against `BANNER_WORD_CEILING = 200`, i.e. comment-stripped. |
| the roadmap banner: date-stamped | **LANDED** | `<!-- As of 2026-09-14, 093 wave 2 has landed and awaits his eyes. -->`, matching the commit date. |
| the roadmap banner: rewritten not appended | **LANDED** | the wave-1 paragraph is replaced, not kept above the new one; the gate's anti-append check passes. |
| the roadmap banner: content | **SMELL (N7)** + **DEVIATES (N5)** | the dropped "Still unseen" list, and `against every reference the research read`. |
| the spec's Status line | **VIOLATION (N3)** | the orphaned third line, and "in progress" for a landed wave. |

---

## False trails

Four things that looked like findings and are not:

- **`centring` in `shaderbox/pass_graph.py:646`.** A real hole in the spelling roster's prefix
  matching (`\bcentre` misses `centring`), but `git log -1 -S "centring" -- shaderbox/pass_graph.py`
  → `f04821a`, feature 092. Not wave 2's, and the file is not even in wave 2's diff (that one is
  `shaderbox/widgets/pass_graph.py`). Filed as N8 for the gate's sake, not against this commit.
- **`centres` in the ledger's row 10.** Under `ai_docs/features/`, which
  `test_prose_spelling.py::_EXCLUDED_PREFIXES` excludes on purpose. Not a violation.
- **The `_init` fallback's nesting.** The implementer flagged it as a possible deviation. It is
  not one: the nesting satisfies the spec's stated rule exactly, and a literal third `elif` would
  have needed a `pass` branch. See section B.
- **`_ENTRY_TICK_W = float(SPACE.SM)` as a token outside `theme.py`.** It is an alias reading a
  theme token, not a new magic number, and it predates this diff.

One more that is a finding but not the one it looks like: **the spec's `126`** reads as a code
bug and is a spec arithmetic slip. The code, the test and the re-derivation all give 122.

---

## E. The gate

Run once, at the end, unpiped to a file:

```
TMPDIR=.../scratchpad make gates > .../gates_w2.log 2>&1; echo EXIT=$?
EXIT=0
```

The log's `== gates:` lines:

```
== gates: check ==
== gates: check passed ==
== gates: test ==
== gates: test passed ==
== gates: smoke ==
== gates: smoke passed ==
== gates: GREEN -- check passed, test passed, smoke passed ==
```

**The smoke word is `passed`, not `skipped`** — the display was available and the smoke ran. Exit
code 0. The commit body's gate claim is accurate.

---

## Verdict

Every one of his six findings landed at the place he pointed at, and every W2 claim is either
met or deviates in the implementer's favor. What is left is four doc defects and one comment-rule
violation — all text, none of them a reason to hold the code:

- **N3** the spec's Status line (an orphaned sentence fragment inherited from `b6cb229`, plus a
  stale "in progress"), **N4** Houdini and imnodes misreported in the ledger's research readout,
  **N5** "every reference" overclaimed in `conventions.md`, the roadmap banner and the commit
  body, **N7** the roadmap's dropped "Still unseen" list, **N1** one comment narrating history.
- **N6** is the spec's own arithmetic, corrected here to 122; no code change is owed.

N4 and N5 matter more than their size: the reversal of a convergent industry schema is justified
in three documents by a claim about the research that the research does not support at that
strength, and the research's own careful scoping ("every reference *that draws wires from a
filled input*") is the fix, already written.

VERDICT: PARTIAL
