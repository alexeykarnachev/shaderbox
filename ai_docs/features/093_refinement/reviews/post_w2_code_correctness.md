# 093 wave 2 — post-implementation review: CODE CORRECTNESS

Commit `1974aa0`, anchored on the maintainer's verbatim findings 6-11 (`00_findings.md`) and the
spec's `## Waves` W2-1..W2-6. Every verdict is a probe output, a quoted line or a test run; each
break-test names the line broken and the test that caught it. Probes in
`scratchpad/test_probe_{tabs,drops2,idxshift2,pin,row2,row3,live,outrects,sess,gj,geom}.py` and
`scratchpad/{geom,zoomcheck,badge}.py`. Baseline: the six graph / persistence / theme test files —
**114 passed**; `make check` exit 0 (7 pre-existing stub-gap warnings).

## Findings

### F1 — DEFECT: the entry row overflows the panel at the narrow split (W2-1)

Putting both summoners on one row makes the row wider than the settings child at the minimum
panel width. Measured in the LIVE draw (`test_probe_live.py`, `editor_split_fraction = 1.0`, so
the panel sits at `_APP_PANEL_MIN_W`):

```
LIVE entry-row region: x0=2350.0 avail=194.0   (the row needs ~276 with the play toggle)
```

The row's own width, drawing the real `_entry_row_label` + `standard_button` +
`play_stop_toggle` sequence in a constrained child (`test_probe_row3.py`): `row_width=260.0`, so
`avail=205` overruns by 55px and `avail=264` clears by 4.

With no script present the row is 208px and still overruns by 3px at `avail = 205`
(`test_probe_row2.py`). `imgui.same_line` never wraps, so the Graph label and its `open` are
clipped at the child's right edge — the button the maintainer asked to be moved there is the
first to disappear. The arithmetic: `_APP_PANEL_MIN_W = 360.0` (`ui.py:67`) minus the grid's
`control_panel_width / 2.6 = 138.5` (`ui.py:1016`) leaves 221.5 for the `document_settings` child
and 194-205 of content inside its borders; the row needs 276.

Wave 1 did not have this: the graph `open` was on its own `_draw_passes` row. The fix is a layout
one (wrap the Graph pair below when `avail` is short, or tighten the `SPACE.LG` gap) — not a token
edit. It bites only the narrow split, which the splitter reaches by hand; the default
`editor_split_fraction` 0.5 gives a wide panel.

### F2 — CORRECT: tab persistence round-trips end to end (W2-2)

The full chain over four tabs — two shader (the current document's `main` plus a second pass
`b`), its script via `open_script_for`, its graph via `open_graph_for`
(`test_probe_tabs.py::test_probe_full_roundtrip`):

```
JSON  editor_tabs: [ {passes/main.frag.glsl, shader, 5372…}, {passes/b.frag.glsl, shader, …},
                     {scripts/script.py, script, …}, {graph.json, graph, …} ]   active_tab_index: 2
FRESH tabs: the same four, in order, kinds [shader, shader, script, graph]
FRESH active_tab_index: 2  tab_select_pending: True
```

`App.save` mirrors at `app.py:2191-2192`; the field is `ui_models.py:273`; the restore is
**inline in `_init`**, `app.py:1478-1487`, NOT a callable method — reachable only by constructing
a second `App` over the same project dir, which is how both the shipped test and my probes drive
it. That is the honest driver (a restart is the finding), but the restore cannot be unit-tested on
its own; only its two pure halves in `editor_types.py` can.

The graph tab survives, which is finding 7 itself: `graph.json` is written by `save_ui_document`
and present for every seeded document (`test_probe_gj.py`:
`after restart kinds: ['shader','graph'] graph survived: True`).

### F3 — CORRECT: every drop rule fires, and the fallback still runs

Records written straight into `app_state.json` (an `App.save()` would overwrite them from the
live list — the trap my first probe fell into), then a fresh `App` (`test_probe_drops2.py`):

```
CASE 'gone path only':     tabs=[('main.frag.glsl','shader')] idx=0   <- dropped, fallback fired
CASE 'unknown doc only':   tabs=[('main.frag.glsl','shader')] idx=0   <- dropped, fallback fired
CASE 'dup path':           tabs=[('main.frag.glsl','shader')] idx=0   <- deduped to one
CASE 'lib tab, no doc id': tabs=[('main.frag.glsl','lib','')] idx=0   <- kept, "" passes the clause
CASE 'graph tab':          tabs=[('graph.json','graph')] idx=0
CASE 'two good, idx 99':   idx=1  <- clamped to len-1     CASE 'two good, idx -5': idx=0  <- clamped to 0
CASE 'empty':              tabs=[('main.frag.glsl','shader')] idx=0   <- ensure_shader_tab fired
CASE 'one gone one good':  tabs=[('graph.json','graph')] idx=0
```

The dedup is `if any(tab.path == path for tab in kept): continue` (`editor_types.py`), keyed on
`path` alone, so two records with one path collapse to the FIRST — right, since
`_focus_or_add_tab` is also path-keyed (`app.py:1571`), so a duplicate path was never live.

Lazy sessions hold across the restore (`test_probe_sess.py`): `sessions right after _init: []`,
one appears per tab as it draws, and the graph tab creates none (`graph.json session created:
False` on all three; `editor_errors: []`). T1's rule survives W2-2.

### F4 — FRAGILE: the restored active index is positional, so a drop before it mis-lands

`active_tab_index` is saved as an index into the PRE-drop list and restored by clamping only
(`app.py:1484-1486`), so dropping an earlier record shifts every later tab down one and the index
selects the wrong one. Four records, `records[0]`'s file gone, the active tab at index 1
(`test_probe_idxshift2.py`):

```
surviving tabs: ['b.frag.glsl', 'c.frag.glsl', 'graph.json']
restored index: 1 => active: c.frag.glsl
EXPECTED active (what he was looking at): b.frag.glsl
```

The spec says "sets `active_tab_index` clamped to the list" — exactly what shipped, so this is no
spec violation, and it only shows when a record was dropped (a pass or document deleted between
sessions). Remapping is two lines (carry the active RECORD's identity, or have
`tabs_from_records` return the surviving indices). Filed fragile because nothing tests it and the
next reader will assume the index means what it meant at save time.

### F5 — CORRECT: `model_salvage` costs one bad row, never the file

Probed over six corruption shapes (`test_probe_tabs.py::test_probe_malformed`), each with an
unrelated `global_target_fps: 90` sitting beside the tabs:

```
MALFORMED 'wrong kind':         tabs=[('/a','shader'),('/c','lib')] fps=90   <- the bad row alone
MALFORMED 'missing path':       tabs=[('/a','shader'),('/c','lib')] fps=90
MALFORMED 'path is int':        tabs=[('/a','shader'),('/c','lib')] fps=90
MALFORMED 'extra key':          tabs=[('/a','shader')]              fps=90   <- drop_unknown pruned it
MALFORMED 'row is scalar':      tabs=[]                             fps=90   <- whole field lost
MALFORMED 'whole field is str': tabs=[]                             fps=90   <- whole field lost
```

`drop_invalid`'s list branch descends per element and drops only the one that still fails
`nested(**item)`, so the three dict-shaped corruptions cost their own row. The two that cost the
whole field are its stated design (a non-dict element is `kept.append`ed, then the field-level
`validate_assignment` rejects the list and pops the key) — pre-existing, degrading to `[]` so the
fallback fires, with the sibling key surviving throughout. No new defect.

**The roster rule is satisfied trivially.** `test_persistence_completeness.py`'s completeness
half enumerates MODULES that call `json.load` against `rostered = {"ui_models.py", …}` —
`editor_tabs` is a new FIELD on the already-rostered `app_state` store, so it inherits the whole
battery with no roster edit. All 4 stores x 9 corruptions + 4 absent-file cases pass.

### F6 — CORRECT: the input-pin rule, all five behaviors (W2-4)

Frames driven with both session writes mocked (`test_probe_pin.py`):

```
press a FILLED port, drag 6px, release: node_drag True, wire_drag None,
                                        set_sampler_source 0, set_pass_positions 1, c.u_src still PassSource('b')
output dot -> FILLED port:              set_sampler_source 1 [('c','u_src',PassSource('a'))], c.u_src now PassSource('a')
output dot -> EMPTY canvas:             set_sampler_source 0, c.u_src unchanged
the ✕ / Delete:                         1 write each, [('c','u_src',NoSource())] / [('b','u_src',NoSource())]
```

**No path can still build a `WireDrag` with a grab.** `grep -rn "WireDrag(" shaderbox/ tests/
scripts/` → two sites: `pass_graph.py:1235`, inside the `node.outputs` loop
(`WireDrag(producer=member, start=_out_point(node, slot))`), plus one test assignment. The field
is gone from the dataclass; `grep -rn grabbed` over the same trees finds only two unrelated prose
matches. **`_drop` cannot unwire**: its body is `if target is None: return`, one
`app.drop_wire(...)`, a notification — no `app.unwire` call. `app.unwire`'s only callers are
`pass_graph.py:970` and `:1360` (Delete and the ✕), plus one test.

**Break-test.** Restoring the `port.kind == "wired"` grab branch in a worktree at `1974aa0` turned
`test_an_input_pin_moves_the_node_and_never_carries_its_wire` red with exactly the message the
commit body claims: `AssertionError: the press did not move the node` (`1 failed, 2 passed`).

### F7 — CORRECT: the card geometry at every zoom (W2-5, W2-6)

Recomputed from the shipped tokens (`scratchpad/geom.py`, `zoomcheck.py`, `badge.py`): the side
gap `(136-116)/2` is **10.0** (asked: 20 -> 10), the top gap `GRAPH_PAD` is **4** (asked: 8 -> 4),
and the last label's text bottom clears the border by **10.0** at z=1 — the same 10 its left inset
`2*R+2` gives it. The derivation: text bottom `= centre + font/2 = centre + 6`, card bottom
`= centre + ROW/2 + PORT_BOTTOM = centre + 16`. Measured at 1, 3 and 5 ports: 10.0 each.

The picture stays inside the card at every zoom, slack scaling linearly:

```
z=0.25 ports=3: card=(0,0,34,52.25)  thumb=(2.5,1,31.5,30)  inside=True slack=(L2.5,T1,R2.5)
z=1.0  ports=3: card=(0,0,136,209)   thumb=(10,4,126,120)   inside=True slack=(L10,T4,R10)
z=2.5  ports=3: card=(0,0,340,522.5) thumb=(25,10,315,300)  inside=True slack=(L25,T10,R25)
```

A 0-port card pays neither port pad: `node_size(0, False) = (136, 144) = 4+116+20+4` — correct,
no label to clear. The `xN` badge sits inside the picture at all three zooms (`badge.py`).

**The trailing `GRAPH_PAD` (122 vs the spec's 126) does not matter for the two names.** In a rig
frame (`test_probe_geom.py`), `u_distance_field` measures **112.0px** in `font_12` — it fits the
code's 122 (10px slack) and the spec's 126 (14px) alike, and the ellipsis test's `>= 4.0`
assertions hold at 122. A longer name cuts at either (`u_previous_frame_tex` is 140px). In
`font_14_bold`, `distance_field` is 112.0 against the 128 name budget.

**Break-test.** Dropping `SIZE.GRAPH_PORT_BOTTOM` from `node_size` turned `test_a_node_grows_one_
row_per_port_and_a_box_is_wider` red: `assert 184.0 == (((144.0 + 4) + (2 * 18)) + 7)`.

### F8 — FRAGILE: no test pins the new token VALUES

Reverting `GRAPH_THUMB` to 96 and `GRAPH_PAD` to 8 in a worktree at `1974aa0` left the ellipsis
row and the `node_size` row **both green** (`2 passed, 29 deselected`). Every test reference
recomputes from the tokens (`grep -rn "GRAPH_THUMB\|GRAPH_PAD" tests/` → 4 derived expressions, no
literal), and `GRAPH_NODE_W` is unchanged at 136, so the name budget merely shifts 128 -> 120 and
`distance_field` at 112px still fits. The two numbers he measured by eye can be silently
reverted.

Not a should-not-land: W2-5 promises only that the budgets follow the tokens and the slack
assertions hold (both true), and the spec names the card's look as the maintainer's eyes with no
gate claimed. But the commit body's "the two token rows" reads as a value pin and is not one.
`assert (SIZE.GRAPH_NODE_W - SIZE.GRAPH_THUMB) / 2 == 10 and SIZE.GRAPH_PAD == 4` would fix it.

### F9 — CORRECT: the glyph removal is complete (W2-3)

`grep -rn "_draw_feedback_glyph\|GRAPH_FB_SIZE\|GRAPH_FB_GAP\|_FB_RING_R\|_FB_GAP_ARC\|
_FB_ARC_SEGS" shaderbox/ tests/ scripts/` → **zero hits**. The only survivors are in `ai_docs/` —
the wave-1 spec and design record (history, both now carrying the REVERSED pointer) and the wave-1
reviews. No live reader.

`_draw_badge`'s return is unread: the signature is `-> None` (`pass_graph.py:679`), the body ends
at `dl.add_text(...)` with no `return w`, and both call sites (`:803`, `:805`) discard. `badge_w`
is gone from `_draw_node`.

The `prev` port keeps its double ring, untouched, at `_draw_port_dot`:

```python
    elif kind == "prev":
        dl.add_circle(center, r, col, 0, ring)
        dl.add_circle(center, r * _PREV_INNER, col, 0, ring)
```

`_Port.source` did not become dead — still read at six edge-building sites (`:313, 331, 334,
338, 339, 416`).

### F10 — CORRECT: the Script row's two `open`s and the one tick predicate (W2-1)

Both buttons dispatch as specified: `open##entry_script` -> `open_script_for(document_id,
focus_editor=True)` with the create-if-absent tooltip and `STATE_ERROR` text on a script error;
`open##entry_graph` -> `open_graph_for(document_id, focus_editor=True)`, tooltip `"Open the pass
graph"`. The `open##entry_graph` id moved out of `_draw_passes` with the button, so no collision —
the Passes row is `small_caption(app.font_12, "Passes")` alone.

`_entry_tab_active` stays the single predicate, read twice at `document.py:423-424`:
`active is not None and active.kind == kind and active.document_id == document_id`. Correct per
kind — a graph tab of document A makes `("A","graph")` True and both `("B","graph")` and
`("A","script")` False, which `test_graph_tab.py`'s T5 row asserts and which passes. Both `open`s
sit inside the one `begin_disabled(app.copilot_turn_active)` bracket, and `open_graph_for` also
refuses mid-turn on its own (`app.py:1693`).

### F11 — CORRECT: `out_rects` lives exactly as `port_rects` does, and the three rewritten tests still bite

`out_rects` is cleared one line after `port_rects` (`pass_graph.py:975-976`) and written in the
same node loop (`:1218`, beside `:1173`). Probed (`test_probe_outrects.py`):

```
out_rects keys: [('p:a',0),('p:b',0),('p:c',0),('p:main',0)];  rewritten every frame (new dict): True
same keys next frame: True;  after switching away -> out_rects stale 4, port_rects stale 2
```

Identical lifecycle, including the pre-existing staleness when the tab is not drawn (both dicts
are cleared inside `_draw_canvas`). No new hazard: the tests read them only on a frame where
`canvas_rect != (0,0,0,0)`.

**The three rewritten tests still test their docstrings.**

- `test_a_press_that_spans_a_copilot_turn_never_becomes_a_gesture` — its subject is the
  `press_blocked` latch, unchanged by W2-4; only the gesture's START moved to an output dot.
  Break-tested: replacing `view.press_blocked = True` with `pass` turned it **red**
  (`1 failed, 3 passed`). The added `_let_the_double_click_lapse(app)` is test mechanics, not a
  weakening — the re-arm assertion still runs after it.
- `test_the_cursor_follows_the_gesture` — the crosshair half drives an output dot now; its three
  assertions and the pan half are unchanged. Still the G7 gate.
- `test_an_input_pin_moves_the_node_and_never_carries_its_wire` — new, its named falsifier
  reproduces (F6). `write.call_count == 0` covers the whole gesture, and the
  `uniform_values["u_src"] == PassSource("b")` line checks the state, not just the mock.

## False trails

- **The restore skipping `ensure_python_worker`'s warm-up.** `open_script_for` queues the jedi
  warm-up (`app.py:1686`); the restore does not. Dismissed: `tabs/code.py:535` and `:633` call
  `ensure_python_worker()` at the completion and lookup sites, and it queues `WARM` before
  returning the worker — a restored script tab warms on its first completion, on the worker
  thread. No user-visible change.
- **`_Port.source` going dead with the grab branch.** Still read at six sites; not dead.
- **`shutdown()` clearing `editor_tabs` before a save.** It does (`app.py:2221`), but `save()`
  runs before it in the frame-loop tail. Probed: the round trip survives.
- **`open_graph_for`'s `document_id not in self.ui_documents` guard vs. the restore.** The restore
  builds `EditorTab`s directly, so the guard does not apply — but `tabs_from_records`' document
  clause covers the same case, which the "unknown doc" probe drops.
- **My first drop probe restoring `main.frag.glsl` in every case.** Self-inflicted: I called
  `app.save()` after hand-setting `app_state.editor_tabs`, and `save()` re-mirrors from the LIVE
  list. Re-run writing `app_state.json` directly (F3) — recorded because the same trap would make
  a future test green for the wrong reason.

## Summary

| # | Verdict | Subject |
|---|---|---|
| F1 | **DEFECT** | W2-1: the one-row entry points overflow the settings child by 55-66px at the minimum panel; the Graph `open` is what clips |
| F2 | CORRECT | W2-2: the four-tab round trip, disk shape and active index |
| F3 | CORRECT | W2-2: all five drop/clamp rules + the `ensure_shader_tab` fallback + lazy sessions |
| F4 | FRAGILE | W2-2: the restored active index is positional, so a drop before it selects the wrong tab |
| F5 | CORRECT | W2-2: one malformed row costs itself; the roster rule needs no edit |
| F6 | CORRECT | W2-4: the pin press, the overwrite, the empty drop, the ✕ and Delete; no grab construction survives; break-tested |
| F7 | CORRECT | W2-5/W2-6: both gaps halved, 10px clearance, picture inside at 0.25/1/2.5; 122 vs 126 is moot for both names |
| F8 | FRAGILE | W2-5: reverting the tokens to 96/8 leaves every test green — no value pin |
| F9 | CORRECT | W2-3: zero live readers of the deleted symbols, the badge return unread, the `prev` ring intact |
| F10 | CORRECT | W2-1: both `open`s dispatch right, `_entry_tab_active` is still the one predicate |
| F11 | CORRECT | `out_rects` matches `port_rects`' lifecycle; all three rewritten tests still bite |

One defect, two fragilities, eight correct. F1 is a layout regression W2-1 introduced in a
reachable state, worth a row edit before the maintainer's hands-on pass: the button he asked to be
moved is the one that vanishes at the narrow split. Nothing here touches a write path or a
persistence guarantee.

VERDICT: PARTIAL
