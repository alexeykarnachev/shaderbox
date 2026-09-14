# 093 wave 1 — post-implementation review: code correctness

Commit `f012074`. Read-only on the shared tree; probes under a scratchpad, one `git worktree` at
`44976d3` for the gate mutation. Every claim below is a probe output, a quoted line, or a test
run. `make gates` was not run (another reviewer owns that log path); the five named gate tests,
`tests/test_graph_view.py`, `tests/test_graph_state.py`, `tests/test_graph_tab.py`,
`tests/test_theme.py`, `tests/test_canvas_fields.py` and `scripts/smoke.py` were run here.

## Findings

**F1. A card in flight draws and hit-tests UNDER a merely-selected neighbour — `FRAGILE`.**
`pass_graph.py:1098-1108` sorts one list ascending by

```python
    nodes.sort(
        key=lambda n: (
            selected_of[n.key],
            bool(set(n.members) & dragging) if n.kind == "box" else n.name in dragging,
        )
    )
```

A dragged-but-unselected card scores `(False, True)`; a selected-but-still card scores
`(True, False)`, which sorts higher. Driven for real — select `b`, then press-drag `a` (a drag on
an unselected node leaves the selection alone, `_drag_names` returns `[node.name]`):

```
selection={b}: node_order: ['p:a', 'p:main', 'p:b']
mid-drag: node_drag origin: ['a']  selection: ['b']  node_order: ['p:main', 'p:a', 'p:b']
VERDICT: the card in flight is NOT last -- last is p:b
```

So the card under the user's cursor can be occluded by a stationary one, and its button is
submitted earlier, so the selected card wins their overlap. The code matches S7's literal key
(`(is_selected, is_being_dragged)` ascending) and G12's prose does not say which of the two wins,
so this is spec-faithful; it is filed as FRAGILE because the behaviour contradicts G12's stated
purpose ("a selected **or** dragged node draws last and occludes its neighbours") in a reachable
state, and the fix is one key swap. Cosmetic, and a visual call.

**F2. `view.port_rects` collides on a doubled ghost — `CORRECT` behaviour, misleading test aid.**
In a group scope a pass that both feeds a member and reads another is drawn twice (092 D4,
`g:in:x` and `g:out:x`), and both write `port_rects[(owner, sampler)]`
(`pass_graph.py:1225`, keyed on `node.owners[slot]`), so the LAST ghost submitted wins:

```
group nodes: ['g:in:x', 'g:out:x', 'p:m1', 'p:m2']
port_rects key collisions: {('x','u_other'): ['g:in:x','g:out:x'], ('x','u_src'): ['g:in:x','g:out:x']}
  g:in:x u_other dot at screen 176,755
  g:out:x u_other dot at screen 976,755
live port_rects[('x','u_other')] -> (969, 748)   # the g:out one
```

The live drop is unaffected: it reads `drop_target`, computed in the loop, not `port_rects`. Proved
by spying `_drop` on a drag aimed at the LEFT ghost's dot — `_drop saw target: ('x','u_src','none')`
and `x.u_src` became `PassSource('m2')`. The keying and the ghost doubling are both `44976d3`
code, unchanged by this commit (`git show 44976d3:…/pass_graph.py` line 995 is the same
expression), so it is neither introduced nor a regression. A headless test that aims through
`port_rects` at a doubled ghost would aim at the wrong dot; no such test exists.

No `DEFECT` found.

## 1. The frame walk of `_draw_canvas`, and every field

Numbered as the code runs (line numbers from `shaderbox/widgets/pass_graph.py`):

1. **968-1002 press bookkeeping.** `origin`/`avail`/`io`; `mouse_down`, `released_elsewhere`,
   `frozen`. A press held into a turn latches `press_blocked = True` (997). A gesture whose
   release the canvas did not see, or a turn, clears `node_drag`, `wire_drag`, `band_anchor`,
   `guides` (999-1002).
2. **1003 `hovered`** = `is_window_hovered(child_windows)`.
3. **1006-1020 the badge hit-test, by hand**, reading LAST frame's `x_rect` and this frame's
   `selected_wire`, gated on `not (frozen or view.press_blocked)`. On a hit: `app.unwire(...)`,
   toast on refusal, `press_blocked = True`.
4. **1021 `blocked`** = `frozen or view.press_blocked` — so step 3's latch is already in it.
5. **1022-1023** `port_rects = {}`, `canvas_rect` written.
6. **1024-1029** `overrides` from the drag; `picture = _build_view(...)`; `selected_wire`
   revalidated against this frame's drawn edges.
7. **1030-1031** `_fit` when `not view.fitted` (sets `zoom`, `pan`, `fitted`).
8. **1032 `xf`**; **1036-1047** wheel zoom about the cursor, `xf` rebuilt.
9. **1052 `channels_split(5)`**; **1062-1063** `wire_mids = {}`, `x_rect = None`.
10. **1064-1094 the wire loop** — per edge: state from `hovered_wire` (last frame) and
    `selected_wire` (this frame), stroke + halo, `wire_mids[wire_id] = centre`, and for the
    selected wire `x_rect` + `_draw_wire_x`.
11. **1096-1109** the sort, `node_order` written; **1110-1130** `_draw_node` per card, reading
    `hovered_node` / `hovered_port` / `hovered_out` from LAST frame.
12. **1132-1161 the background button**: `bg_hovered`, `bg_active`, `bg_pressed`, `panning`
    (pan writes `view.pan`), the rubber band's `band_anchor`.
13. **1163-1287 the node loop** (same order as the draw): node button → double-click →
    release-click → node drag; then the port buttons (writing `port_rects`, `drop_target`,
    `wire_drag` / `node_drag`); then the output dots (`wire_drag`).
14. **1291-1308 the wire distance pass** (only when `hovered`).
15. **1310-1313** the four hover fields written, exclusively, every frame, `None` included.
16. **1315-1323 the background press**, decided against THIS frame's `hovered_wire`.
17. **1325-1351** the drag in flight: `node_drag.update`, `guides = _snap(...)`, commit on
    release; the cursor requests; the in-flight wire and `_drop`.
18. **1352-1395** the band's overlay and its release (writes `selection`, clears
    `selected_wire`, `band_anchor`); the guides.
19. **1396 `channels_merge()`**.
20. **1398-1412 Delete/Backspace**, read here and once, through `delete_allowed`.
21. **1414-1421** the canvas context menu on a right-click release with no node hovered.
22. **1422-1423** `_canvas_menu`, `_group_prompt`.
23. **1425-1426** `if not mouse_down: press_blocked = False` — the latch clears at the END.

Per field: `hovered_node` / `hovered_port` / `hovered_out` / `hovered_wire` — written once at
step 15, read at step 11 (draw) and step 16 (the background press reads `hovered_wire` AFTER
step 15, i.e. this frame's). `selected_wire` — written at 6 (revalidation), 16, 18; read at 3,
10, 20. `x_rect` — cleared at 9, written at 10; read at 3 (last frame's, intentional, S6).
`wire_mids` — cleared and written at 9-10; read by tests only. `node_order` — written at 11.
`press_blocked` — written at 1, 3, 23; read at 3, 4. `node_drag` / `wire_drag` — cleared at 1,
set at 13, read at 6, 11, 13, 17. `band_anchor` — cleared at 1 and 18, set at 12, read at 12, 18.
`port_rects` — cleared at 5, written at 13. `canvas_rect` — written at 5.

**Reads that precede their write:** exactly one, `x_rect` at step 3, and it is S6's design ("it
reads last frame's `x_rect` and `hovered`"). Probed for the risk it creates — a press aimed at
where the badge USED to be after the view moved:

```
x_rect before pan: (737.0, 750.5, 751.0, 764.5)  after one frame: (437.0, 550.5, 451.0, 564.5)
press at the STALE badge position: writes= 0  selected_wire= None
press at the CURRENT badge position after a pan: writes= 1
```

The rect is rewritten every frame the wire is drawn, so the window is one frame wide and the
press at the old spot is refused. `hovered_node` shows the same one-frame character across a
topology change and never lights a wrong card: `p:c` → (scope entry) `None` → `g:out:c`.

**Fields not written on some path.** `_draw_canvas` runs only when `begin_child` returns True,
and `_fit` returns early on `avail.x <= 0 or avail.y <= 0` WITHOUT setting `fitted`, so a
zero-size frame refits on the next sized one — correct. On an empty/one-pass document every field
is still written: `wire_mids == {}`, `x_rect is None`, `node_order == ['p:main']`,
`fitted is True`, `canvas_rect` set, `port_rects == {}`. On a scope change `selected_wire`
clears the moment no drawn edge carries it (`('c','u_src')` became `None` when `c` went inside a
box). `forget_render_state` drops `graph_views` and the next frame rebuilds one
(`canvas_rect` non-zero after).

## 2. Every gesture, driven

Each row is a real frame drive with `mock.patch.object(app.session, …)` wraps. "—" = no session
write.

| Gesture | Session writes | View left as |
|---|---|---|
| Click a node body (3px) | `set_output_pass` ×1 | `selection={'b'}`, `selected_wire=None`, tab still `graph`, `panel_pass=''` (the pin cleared) |
| 3px press | `set_output_pass` ×1, `set_pass_positions` ×0 | tab `graph` |
| 5px press | `set_pass_positions` ×1, `set_output_pass` ×0 | — |
| Shift-click a second node | none | `selection={'a','b'}`; a third shift-click → `{'a'}` (XOR) |
| Rubber band over everything | none | `selection={'a','b','c','main'}`, `band_anchor=None`, `selected_wire=None` |
| Output dot → empty port | `set_sampler_source('d','u_src',PassSource('a'))` ×1 | `wire_drag=None` |
| Output dot → media-bound port | ×0 (refused, toasted) | value still the bound texture, `glo=22` (not released) |
| Output dot → empty canvas | ×0 | `wire_drag=None` |
| Re-grab, dropped back on the same port | ×0 | value still `PassSource('b')` |
| Re-grab, dropped on empty canvas | `set_sampler_source('c','u_src',NoSource())` ×1 | — |
| Click a wire | none | `selected_wire=('c','u_src')`, `selection=set()` |
| Delete | `set_sampler_source(...NoSource())` ×1 | — |
| Backspace | ×1 | — |
| ✕ over open canvas | `set_sampler_source` ×1, `set_output_pass` ×0 | `selected_wire=None`, `x_rect=None`, `press_blocked=False`, `selection` unchanged |
| ✕ over a covering card | ×1, `set_output_pass` ×0 | the release-frame node click refused by the latch |
| Double-click a pass | — | `app.active_tab.kind == 'shader'`, path = the pass's |
| Double-click a box | — | `scope='G'`, `selection={'b','c'}`, `fitted=True` |
| Click a box | `set_output_pass` ×1 (the bundle, `c`) | `selection={'b','c'}` |
| Click a ghost | ×0 | `scope=''`, `selection={'a'}`, `fitted=True` |
| Middle-drag pan | — | `pan` moved, `cur_cursor is hand_cursor` |
| Alt+left pan | — | `pan (0,0) → (-30,-20)`, hand cursor |
| Wheel +1 | — | `zoom 1.0 → 1.1` (×1.1000) and the canvas point under the cursor unchanged to 1e-3 |
| Wheel clamps | — | 2.5 max, 0.25 min; a wheel off the canvas leaves zoom alone |
| Right-click a node | — | only the node menu opened (4 frames); canvas menu 0 |
| Right-click the background | — | only the canvas menu (4 frames); node menu 0 |
| Scope into a group with a wire selected and back | ×0 on a Delete there | `selected_wire` cleared by `revalidated_wire` when the edge went inside the box |
| Delete while a press is held | ×0 | the `hovered` clause refuses it; after the release, ×1 |
| Delete with a node selection only | ×0, `passes` unchanged | G17 held |
| Press held across a copilot turn | `set_output_pass` ×0, `set_pass_positions` ×0 | `press_blocked=True` during, `False` after the release |
| Multi-select drag (2 cards) | `set_pass_positions` ×1, `save_ui_document` ×1 | both positions written, `node_drag=None`, `guides=[]` |

## 3. Geometry, recomputed

All `CORRECT`.

- `wire_points` over `dx ∈ [-600,600]` step 7 × `dy ∈ [-400,400]` step 11 × zoom
  {0.25, 1, 2.5}: **0 violations** of (offset ≥ 0), (cp offsets symmetric), (`cp0.x > cp1.x`
  whenever `dx < 0`), (no fold `2·offset ≤ dx` in the backward case).
- `dist = 0`: offset is `GRAPH_WIRE_MIN_OFF * z` exactly (6.0 / 24.0 / 60.0), the curve
  degenerates to the endpoint and `bezier_point(t=0.5)` returns it.
- `bezier_point(…, 0.5)` equals G5's closed form `(p0+3cp0+3cp1+p3)/8` to 0.0.
- `wire_hit_threshold`: 6.0, 6.0, 6.0, 6.0, 7.5 at zooms 0.25 / 0.5 / 1 / 1.5 / 2.5 — G4's table
  verbatim.
- `wire_hit`: on-curve 0.023; at `threshold − 1` along the normal, 4.977 (a hit); at
  `threshold + 1`, `None`; a point at (5000,5000), `None` (the bbox reject).
- `_point_segment_distance` with a zero-length segment returns the point distance (5.0 for
  (3,4) vs (0,0)) — no division by zero.
- Flattening at 24 segments over a 400px backward S-curve, 200 true-cubic samples: worst
  distance **0.677** against a 6.0 threshold. The spec's discrimination claim holds — 4 segs
  23.4, 6 segs 8.5, 8 segs 1.2, 24 segs 0.68.
- `_fit` over the sampled curves: the shipped test frames every wire; with the sampling removed
  in the base worktree a point lands at x=557.15 outside a window ending at 552.0 (see §5).
- `_wire_x`: `half = max(GRAPH_WIRE_X_R·z, GRAPH_HIT_MIN)` = 7.0 / 7.0 / 17.5 at zoom
  0.25 / 1 / 2.5; arm `0.5·half`, thickness `max(1, 1.5z)`. Draw and hit use the same `half`.
- `_draw_feedback_glyph`: `r = size·0.25` = G8's `size/4`; both arcs sweep 291.2°, gap facing the
  other ring — the record's `(0.6, 2π−0.6)` and `(π+0.6, 3π−0.6)` produce the same two sweeps
  from the code's `a_min + 2π − 2·0.6` form.
- Ellipsis budgets at 136, measured in a rig frame on the shipped faces: advance **7.0** at 12px
  and **8.0** at 14px bold; `u_distance_field` 112px against the 118 label budget (6px slack) and
  ellipsizes at the 128-card's 110 (`'u_distance_f...'`); `distance_field` 112 against the 120
  name budget (8px slack). Degenerate budgets (0, −5, 1, empty string) all return `''`, never a
  raise.
- `node_size`: 0 ports (136, 132) = `8+96+20+8`; 1 port (136, 154) = `+4+18`; a box (176, 132).
- The font floor `max(4.0, legacy_size·z)` makes text LARGER than proportional below z≈0.333
  while the budget scales linearly — probed at five zooms, the ellipsis still keeps every label
  inside the card (z=0.25: `'u_distance_...'` 28.0px against a 29.5 budget). Cuts more, never
  overflows.

## 4. Lifecycle and errors

All `CORRECT`.

- Graph tab API surface: `tab_label` → `"UV Mango (graph)"`; `is_tab_dirty` False;
  `is_current_editor_dirty` False; `formatter_for("graph") is None`; `format_current_editor()`
  and `jump_to_next_error()` return without error; `editor_errors == []`; no session at
  `graph_json_for(did)`.
- `widgets.uniform._locate_uniform_declaration(app, "u_src")` inside a rig frame with the graph
  tab active returns `None` and creates **no** session at the graph path (T1's break would).
- `open_graph_for` twice → one graph tab, and it is active.
- `close_editor_for_path(graph_json_for(did))` removes it and leaves the shader tab.
- Deleting the document drops the tab (`_on_document_deleted` filters by `document_id`).
- Renaming the document re-labels through `tab_label` (a shader tab of the renamed document read
  `"renamed one (main)"`).
- Deleting every pass but one while the tab is active: three deletes, no crash, tab still
  `graph`, `node_order == ['p:main']`, `wire_mids == {}`.
- Changing the current document moves focus to that document's shader tab (T6's premise:
  `_on_current_document_changed` → `ensure_shader_tab`); the graph tab survives in the tab list.
- A graph tab of a non-current document still draws (`canvas_rect` non-zero after the switch).
- A compile error on `b`: `_Node.error True`, `uncompiled False`, the wire still drawn.
- A cycle forced into the model: both edges flagged `on_cycle True`, the canvas keeps drawing,
  `fitted` stays True.
- `App.unwire`'s refusal path: `unwire(did,"nosuch","u_src")` → `"no such pass 'nosuch'"`,
  which the two call sites (`_draw_canvas`'s badge, the Delete read) push to
  `app.notifications`. `unwire` itself is unchanged by this commit.
- `choose_output`'s refusal path: a bogus name leaves `graph.output` alone (`main`) and pushes
  nothing (the `ui_document.document.graph.output == name` / missing-document guard returns).
- The pin clearing: `set_panel_pass(did,"b")` then `choose_output(did,"c")` leaves
  `panel_pass == ''` (measured in the node-click row above) — S15's rule.

## 5. The gate claims

From the working tree: `tests/test_graph_view.py` 23 passed; the five named files together
**66 passed** (`test_graph_view.py`, `test_graph_state.py`, `test_graph_tab.py`,
`test_theme.py`, `test_canvas_fields.py`). The five named gate tests by name: **5 passed**.
`scripts/smoke.py`: exit 0, `smoke: OK (200 frames, 7 documents)` — the body's claim, reproduced.

The mutation, in the base worktree only (`git worktree add … 44976d3`, then
`git checkout f012074 -- shaderbox tests scripts`, then the break, then restore). Removing the
sampled-curve loop from `_fit`:

```
FAILED tests/test_graph_tab.py::test_the_fit_frames_every_wire_not_only_the_cards
E   AssertionError: (('a','u_src'), 1, (557.1512796973191, 56.45066550925927))
E   assert (557.1512796973191 <= 552.0)
```

The commit body says "a point at x=557.15 outside a fitted window ending at 552.0" — the same
point, the same window. After restoring, `tests/test_graph_tab.py` is 11 passed there again.

Also verified without a mutation: S13's grep gate covers all four `is_mouse_dragging(` /
`get_mouse_drag_delta(` sites (lines 1154, 1158, 1196, 1242, 1279) and is not fooled by the
prose mention at line 1183 (backticked, no paren, so it does not match the gate's needle);
`channels_split(5)` and `channels_merge()` appear once each; the widget source contains none of
`_draw_self_loop`, `bus_y`, `_MIN_DIRECT_DX`, `_BEZIER_BOW`, `glfw.set_cursor`, `if backward`;
`GRAPH_MIN_H`, `PassesView`, `passes_view` and `PASSES_VIEW_LABELS` are gone from `shaderbox/`,
`tests/` (except the retired-token assertion list), `scripts/` and `projects/dev/`.

## False trails — probed and found fine

- **A stale `x_rect` after the view moves.** It is rewritten every frame the wire draws; a press
  at the old position writes nothing (§1).
- **Both context menus on one right-click.** My first probe counted 4 canvas-menu frames on a
  node right-click; isolated from a clean start, node → node menu only, background → canvas menu
  only. The 4 were my own unclosed earlier popup.
- **A `wire_mids` key collision.** No drawn-edge `wire_id` collision exists in the doubled-ghost
  case (`{('m1','u_src'): [...], ('x','u_other'): [...]}`, one each) — a sampler has one source,
  so one edge terminates at it.
- **A drop on the LEFT ghost dot being lost to the `port_rects` collision.** Not lost; the empty
  write I first saw was the cycle refusal (`'passes form a cycle: m1 -> x -> m1'`). With a legal
  producer the drop wrote (`_drop saw target: ('x','u_src','none')`, `x.u_src = PassSource('m2')`).
- **`node_order` instability with nothing selected.** `list.sort` is stable, so ties keep
  `picture.nodes` insertion order exactly (`['p:a','p:b','p:c','p:main']` both ways).
- **The hover key surviving a scope change onto the wrong card.** It resolves to `None` on the
  first frame the key is gone, then to the new key.
- **The low-zoom font floor overflowing a card.** Cuts more, never overflows, at five zooms.
- **`_fit` leaving `fitted` True on a zero-size frame.** It returns before the write, so the
  next sized frame fits.
- **`App.unwire` answering `''` for a sampler that does not exist.** Reachable only through a
  `(consumer, sampler)` pair the canvas never produces (every pair comes from a drawn edge), and
  `unwire` is `44976d3` code, untouched here.

## What was skipped, and why

- `make gates` — another reviewer owns its shared log path (instructed).
- Everything the spec assigns to the maintainer's eyes: the halo reading as a thicker wire, the
  glyph's shape at 12px, the neutral hover against the grey wire, the 4px lock under his hand.
  No gate can run these and no probe substitutes for them.
- The `any_item_active` clause's live scenario (a text input active in another window while the
  mouse rests over the canvas) — the spec already records it as unreachable headlessly and
  assigns it to the maintainer; `delete_allowed` is pinned over its whole 32-case domain by the
  shipped pure test.
- Pixel comparison of the rendered canvas (no window manager here).

## Round 2 — closure against `ccdf6d1` (spec text `db93fe1`)

**F1 is CLOSED.** The key is now `nodes.sort(key=lambda n: (dragged_of[n.key], selected_of[n.key]))`
(read from disk, line 1101), both flags from one `_touches(node, names)` helper that answers for a
box's members. Driving the same real gesture as round 1 — select `b`, then press-drag the
unselected `a`:

```
selection={b}: node_order: ['p:a', 'p:main', 'p:b']
mid-drag: node_drag origin: ['a']  selection: ['b']  node_order: ['p:main', 'p:b', 'p:a']
VERDICT: the card in flight is LAST (on top)
```

Round 1 measured `['p:main', 'p:a', 'p:b']` here. The card under the hand now paints and
hit-tests last. The shipped gate is real: reverting the key in a scratch worktree turns
`test_the_card_in_flight_outranks_a_merely_selected_one` red with `assert 'p:b' == 'p:a'` —
the commit body's quoted failure, reproduced.

**The three suites are green:** `tests/test_graph_view.py`, `tests/test_graph_state.py`,
`tests/test_graph_tab.py` → **53 passed** under the GL env.

**No rename changed a key.** The sweep's sites in `pass_graph.py` are one local variable
(`centre` → `center`, subscripts included), `_draw_wire_x`'s parameter of the same name, and
prose. Proved rather than skimmed: an `ast` diff of every string `Constant` between `f012074`
and disk gives, as only-in-old / only-in-new / spelling-related — `pass_graph.py` 1/2/2 (both
hits the SAME docstring), `test_graph_view.py` 0/1/0, `test_graph_state.py` 3/3/0,
`test_graph_tab.py` 1/1/2 (one docstring). Every spelling-related string change is a docstring.
No `##id` (`##gnode_`, `##gport_`, `##gout_`, `##graph_bg`, `##graph_canvas_menu`,
`##graph_group`), no `NodeKind` `Literal` member, no port `kind` string, no `port_rects` /
`wire_mids` tuple element moved. Nothing found.

**The drag-lock gate is a real gate.** `_drag_call_sites()` walks the `ast` and reads each call's
unparsed arguments, with a site count of 5. In a scratch worktree at `ccdf6d1`, replacing the
node-body site with a bare call and the token in a comment on the next line:

```
            and imgui.is_mouse_dragging(imgui.MouseButton_.left)
            # the lock is SIZE.GRAPH_DRAG_LOCK_PX
```

→ `FAILED tests/test_graph_state.py::test_the_drag_lock_is_passed_at_every_site`
`AssertionError: ('is_mouse_dragging', 1189, ['imgui.MouseButton_.left'])`.

Replaying the OLD `f012074` four-line text-window logic on that same broken tree:
`failures: []` — it accepted the comment, which is precisely the hole the rewrite closes.
Restored; `tests/test_graph_state.py` 18 passed.

F2 (the doubled-ghost `port_rects` collision) stands as round 1 filed it; the commit body leaves
it on record for the reason I measured — it cannot reach a write. No new finding.

VERDICT: PASS
