# 093 wave 1 — pre-implementation review: verification and blast radius

Read: `01_spec.md` in full, `03_graph_design.md` §1-2 and §6, `CLAUDE.md`, `dev_flow.md`,
`conventions.md`, the nine code modules and the eleven test modules the brief names. Every
finding is a probe output, a grep result or a quoted line. Probes live under the scratchpad
(`test_probe*.py`, `probe_fit.py`, `probe_prose.py`), run against today's code through the
repo's own `app` fixture.

---

## Findings, worst first

**F1 (BLOCKER for two Verification rows). A node click calls `pick_pass`, which OPENS
AND ACTIVATES the pass's shader tab — so any node gesture under T1 destroys the graph
tab's own activeness mid-test.** `pass_graph._click` (`pass_graph.py:1209`) ends with
`app.pick_pass(...)`; `App.pick_pass` (`app.py:1886`) begins with
`self.ensure_shader_tab(document_id, name, focus_editor=focus_editor)`, which is
`_focus_or_add_tab` on the pass's shader path. Probed (`test_probe6.py`):

```
TABS_BEFORE (0, 'main.frag.glsl', 1) ['main.frag.glsl']
3px move: pick_pass 1 set_pass_positions 0
TABS_AFTER 1 c.frag.glsl ['main.frag.glsl', 'c.frag.glsl']
```

One 3px click on a node appended a tab and moved `active_tab_index` 0 → 1. Under T2 the
graph body draws only while `app.active_tab.kind == "graph"`, so the frame after ANY node
click the canvas stops drawing entirely: `_draw_canvas` never runs, `view.port_rects`,
`view.canvas_rect`, `view.wire_mids`, `view.x_rect` and every hover field go stale, and no
later frame of the same test reaches the canvas. This is not a test-mechanics quibble — it
is a live behavior question the spec does not decide: **does clicking a node inside the
graph tab yank the user out of the graph tab?** Neither S4, T1-T6 nor G14 names it, and the
record's G14 table lists "Select — left click (clears unless Shift)" as *kept, unchanged*.
Affected rows: `G13: 3px is a click, 5px is a drag` and `G6: exclusive hover in order`
(whose first assert parks on a port, but whose node-body rung needs a node press). Decide
it in the spec (a graph-tab `_click` that picks the pass without summoning its tab, or an
explicit "a node click switches you to the shader tab" behaviour the tests then encode).

**F2 (BLOCKER for one row). `view.canvas_rect` is written ONLY inside `_draw_canvas`, so
the S8 row's "no frames" kind cannot read it.** One writer:

```
shaderbox/widgets/pass_graph.py:858:    view.canvas_rect = (origin.x, origin.y, origin.x + avail.x, origin.y + avail.y)
```

A `GraphViewState` that never drew carries `(0.0, 0.0, 0.0, 0.0)`, so the containment is
false for every point. Deriving the rect from the same `avail` the test handed `_fit` makes
the assertion a tautology over `_fit`'s own output instead. Restate it against the fitted
window in CANVAS space (`view.pan` + `avail / view.zoom`), which `_fit` does set.

**F3. The `GRAPH_HOVER` theme row PASSES WITHOUT THE FIX — three times over.** The record
picks `GRAPH_HOVER = _P["blue_b"]`, and:

```
blue_b  = (0.5137254901960784, 0.6470588235294118, 0.596078431372549, 1.0)
TAG     = (0.5137254901960784, 0.6470588235294118, 0.596078431372549, 1.0)
equal    True
blue_b in exclusions already: True
blue_b vs SELECT/ERROR/EDGE distinct: True
```

`GRAPH_HOVER == COLOR.TAG`, and `COLOR.TAG` is already in `_GROUP_TINT_EXCLUSIONS`
(`theme.py:240`). So adding `GRAPH_HOVER` to that set changes it by nothing — the
future-group-tint guarantee the record claims for it is already carried by TAG; and the three
`!=` asserts are true today and stay true for any hue that is not one of those exact three,
so they only catch a wrong implementation that picks SELECT, STATE_ERROR or GRAPH_EDGE. Keep
them as invariants, not as this guarantee's falsifier. Separately,
`tests/test_theme.py:119-130` re-types the exclusion set by hand instead of importing
`_GROUP_TINT_EXCLUSIONS`, so the two can diverge and nothing catches it — two edits for one
guarantee, and the spec names one.

**F4. `_ellipsize` has three readers OUTSIDE `ui_primitives` that S11's rename breaks and
the Files-touched list does not name.** S11 says "its four call sites inside
`ui_primitives` follow". Grep:

```
shaderbox/popups/lib_picker/tree.py:24:from shaderbox.ui_primitives import _ellipsize, context_menu_style, standard_button
shaderbox/popups/lib_picker/tree.py:350:  ... sep + _ellipsize(first_doc_line, avail))
tests/test_pass_settings_layout.py:20:from shaderbox.ui_primitives import _ellipsize
tests/test_pass_settings_layout.py:91:        kept[name] = _ellipsize(name, float(SIZE.AUTO_NAME_W))
tests/test_anchored_note.py:15:from shaderbox.ui_primitives import _ellipsize, anchored_note
tests/test_anchored_note.py:53:    cut = _ellipsize(long_value, wrap)
```

Three files missing from `## Files touched`. `make check` (pyright) catches all three, so
this costs a round rather than a shipped break — but the spec's list is the contract, and
`tests/test_anchored_note.py` / `tests/test_pass_settings_layout.py` are not in it either.

**F5. `imgui.calc_text_size` between frames SEGFAULTS the process, so the ellipsis row's
"frame-driven" kind is mandatory, not a preference.** Calling it after `a.shutdown()`-less
teardown but outside `new_frame`:

```
Fatal Python error: Segmentation fault
  File ".../test_probe.py", line 134 in test_probe_calc_text_size_outside_frame
```

(the line was `imgui.calc_text_size("u_distance_field")`.) The spec already marks the row
frame-driven; flagging it because the record's §6 sketch marks the same falsifier `pure`, so
an implementer following the record crashes the suite rather than failing a test.

**F6. `imgui` defers a key event queued in the same batch as a mouse-button event to the
FOLLOWING frame.** Probed (`test_probe4.py`), each line one frame after the queue:

```
A_delete_alone        (pressed=True,  down=True,  bksp=False, any_active=False, hovered=True)
B_backspace_alone     (pressed=False, down=False, bksp=True,  any_active=False, hovered=True)
C_release_frame       (False, False, False, False, True)
C_delete_after        (True,  True,  False, False, True)
D_same_batch          (False, False, False, False, True)
D_next                (True,  True,  False, False, True)
```

`add_key_event(Key.delete, True)` reaches `is_key_pressed(delete)` inside `_draw_canvas` on
the next frame (A), and `Key.backspace` likewise (B) — S5's two key names are confirmed live.
But D shows a mouse release and a key queued together put the key a frame later, so any
key-driven row written `release+key, _frames(app, 1), assert` is red. Interleave a `_frames`
call — which the spec's prose already implies, so it is an implementer note, not a defect.

**F7. S5's two gates read OPPOSITE answers at the top and the bottom of `_draw_canvas`, and
the spec does not say where the read sits.** Probed (`test_probe3.py`), same frame, both
measured around the real `_draw_canvas` call:

```
RELEASE+DELETE: start_active_hovered=(True, False)  end_active_hovered=(False, True)
HELD+DELETE:    start_active_hovered=(True, False)  end_active_hovered=(True, False)
```

On the frame a click's release lands, `is_any_item_active()` is **True at the top** (last
frame's background-button active id has not been released yet) and **False at the bottom**;
`is_window_hovered(child_windows)` is **False at the top** and **True at the bottom**. S5
names both conditions and says only "read locally, once, in `_draw_canvas`". Placed at the
top, Delete is dead on every frame that is also a release frame — exactly the frame a user
who just clicked the wire presses it. Pin the position in the spec (after the hit-test
section, before `_canvas_menu`), or the gate silently narrows its own domain.

**F8. The `G11: a six-column chain fits` row distinguishes nothing.** Reproducing `_fit`'s
arithmetic at both widths (`probe_fit.py`, `SPACE.LG = 16.0`, `GRAPH_GAP_X = 64`, six nodes
+ five gaps + 2×margin):

```
W=128 avail=1225.0: raw 1.0938 clamped 1.0000
W=128 avail=740.0:  raw 0.6607 clamped 0.6607
W=108 avail=1225.0: raw 1.2250 clamped 1.0000
W=108 avail=740.0:  raw 0.7400 clamped 0.7400
```

Both asserts (`zoom == 1.0` at 1225; `0.6 < zoom < 1.0` at 740) are **green at the old 108
too**. The row is presented as "the number the width decision turned on, so it earns a
pin", and it pins nothing about the width. Either drop the width claim from it and keep it
as a fit-clamp regression check, or assert the boundary — the widest `avail` at which the
clamp still bites is `1120.0` for W=128 versus `1000.0` for W=108, which does separate them.
(The y term is not binding: node height at the new tokens is 132 + 22 = 154, so
`600/186 = 3.2`.)

**F9. The `G4: 24 segments miss no real hit` row discriminates only down to 8 segments.**
Worst flatten error over 200 true points of a 400px backward S-curve versus the threshold
of 6.0 (`probe_fit`-style arithmetic):

```
segs=  2 worst=  24.621 passes_at_6px=False
segs=  4 worst=  24.621 passes_at_6px=False
segs=  6 worst=   9.477 passes_at_6px=False
segs=  8 worst=   2.666 passes_at_6px=True
segs= 24 worst=   0.906 passes_at_6px=True
```

So the row catches a catastrophically low count and lets 8, 12 and 16 through. Fine as a
regression guard; say so rather than calling it the pin on 24.

---

## A. Row by row

Legend as the brief defines it. `wire_points`, `wire_hit`, `wire_hit_threshold`,
`wire_state`, `ellipsize`, `WireState` do not exist yet; for those rows "RUNNABLE" means
the mechanism the test needs is specified unambiguously and I confirmed the arithmetic.

| # | Row | Verdict | Evidence |
|---|---|---|---|
| 1 | G1: no cusp for any endpoints | **RUNNABLE** | Pure function over floats, no imgui. The algebra holds unconditionally: `offset = max(24z, 0.40·dist) ≥ 0`, so `cp0.x - cp1.x = dx - 2·offset < 0` requires `dx > 0`. Probed the backward instance: `p0.x=512 cp0.x=717.15 cp1.x=-205.15 p3.x=0`, `cp0.x > cp1.x` True. |
| 2 | G1: continuity across the old bus boundary | **RUNNABLE**, thin margin | Sweeping `dx ∈ [-48, 48]` at `dy=40`, the max per-step control-point move is **1.306px** against the row's `< 2px` bound. It holds, but it is a 35% margin on a bound that also moves with `GRAPH_WIRE_BOW`: raising the bow past ~0.6 turns this red on a token edit, not on a topology switch. Name the bound as `2 * GRAPH_WIRE_BOW + ε` rather than a literal 2. |
| 3 | G4: the threshold and its floor | **RUNNABLE** | `max(6.0, 1.5·2·z)` computes 6.0 / 6.0 / 7.5 at z = 0.25 / 1.0 / 2.5 — matches the row exactly. The `threshold ± 1` and bbox-reject halves are pure geometry over the same 4 points. |
| 4 | G4: 24 segments miss no real hit | **PASSES WITHOUT THE FIX** | See F9. Green at `GRAPH_WIRE_HIT_SEGS = 8`; the wrong implementation it lets through is any halving of the count down to 8. |
| 5 | G18: an error wire stays red while hovered | **RUNNABLE** | Pure `StrEnum` + one function over four bools; 16 combinations is the whole domain, which is the right shape. |
| 6 | G6: nothing changes size on hover | **RUNNABLE** | `inspect.signature(node_size).parameters` is `('port_count', 'box')` today (`graph_state.py:101`) — confirmed by `tests/test_graph_state.py` already importing it. The second half needs `wire_hit_threshold` to exist. |
| 7 | G8/G3: the loop and the bus are gone | **RUNNABLE** | All five tokens are live and greppable (`theme.py:328,329,338,340,341`); `_draw_self_loop` at `pass_graph.py:562,906`; `bus_y` occurs 3× (`:539,547,554`). A source-grep test is the right kind here. |
| 8 | S13: the lock is passed everywhere | **RUNNABLE** | Four `is_mouse_dragging(` sites today (`:942, 968, 1009, 1042`), one `get_mouse_drag_delta(` (`:946`), all without a threshold. A grep over `Path(pass_graph.__file__).read_text()` is the established pattern (`test_the_widget_makes_no_session_write_of_its_own`). |
| 9 | G11: the ellipsis and the width | **RUNNABLE, frame-driven mandatory** | See F5 — `calc_text_size` between frames segfaults, so the row's own "frame-driven (a pushed font is needed)" is load-bearing and the record's `pure` marking is wrong. The budgets must be measured inside the same `push_font(app.font_14_bold, legacy_size * z)` scope, per G11's own rule. Prior art for exactly this shape: 069 wave B replaced a font-advance estimate with a headless-frame `_ellipsize` measurement, falsified at 19 (red) and 18 (green). |
| 10 | S8: the fit frames every wire | **NOT RUNNABLE AS WRITTEN** | See F2 — `view.canvas_rect` is `(0,0,0,0)` without a frame. The premise is sound and strong: probed, a backward wire's control points sit **205px outside** the node bbox on each side against a `_FIT_MARGIN` of 16.0. `_fit` and `_Xf` do run outside a frame (`_fit(v, nodes, ImVec2(800,600))` → `zoom 1.0, pan (-240.0, -223.0)`, no crash), so only the containment target needs restating. |
| 11 | G11: a six-column chain fits | **PASSES WITHOUT THE FIX** | See F8. |
| 12 | G5: select a wire and Delete it | **RUNNABLE** | Probed (`test_probe8.py`) that both wire midpoints of the `a→b→c` chain land on open background inside the canvas and inside no node: `edge p:a->p:b mid=(2080.0, 1167.0) inside_nodes=[] in_canvas=True`, `edge p:b->p:c mid=(2252.0, 1194.5) inside_nodes=[] in_canvas=True`. Delete reaches `is_key_pressed` (F6-A). Caveat F6-D on batching and F7 on where the read sits. |
| 13 | G5: the ✕ unwires | **RUNNABLE** | Same aiming mechanism as `port_rects`, which `test_a_wire_dropped_on_a_drawn_port_writes_that_port` already proves works. `view.x_rect` must be written every frame the selected wire is drawn and cleared otherwise, or a stale rect aims the click at nothing — worth stating in S6 beside the `port_rects` parallel. |
| 14 | G5: Delete is inert while a text field is active | **RUNNABLE** | Probed (`test_probe5.py`): with `view.group_prompt = True` and three frames, `is_any_item_active()` is **True at both the top and the end** of `_draw_canvas` and `is_key_pressed(delete)` is True — so the gate does what it claims. One mechanism note the spec should carry: `app.any_popup_open()` is **False** for this popup (probe printed `popup: False`), so `blocked` does not cover it and `not is_any_item_active()` is the only thing standing between the prompt and an unwire. Also `view.group_prompt` is a one-shot (`pass_graph.py:1253-1255` sets it False on the first frame) — the test must not re-assert it. |
| 15 | G6: exclusive hover in order | **TWO REASONS** | (a) The port and off-canvas rungs are clean. (b) The node-body rung is reachable only by parking the mouse on a card — which is fine for hover, but the row's stated break ("swap the port and node rungs — the first case flips") is only checked at the port position, and a swap of the *node and wire* rungs is not covered. (c) F1: if any earlier assert in the same test clicks a node, `pick_pass` switches the tab away and every later read is stale. Split the row so each rung has its own frame sequence with no click in it. |
| 16 | G13: 3px is a click, 5px is a drag | **RUNNABLE, and the named break WORKS — but see F1** | Probed on today's code: `3px → pick_pass 1, set_pass_positions 0`; `5px → pick_pass 1, set_pass_positions 0`; `8px → pick_pass 1, set_pass_positions 1`. So today's effective lock is imgui's 6px default and the 5px case reads as a click, exactly as the row's break predicts: **omitting `lock_threshold` at the node-body site turns the 5px assertion red.** The gate is real. Two riders: the 8px probe shows today BOTH `pick_pass` and `set_pass_positions` fire on one gesture, which is the S4 defect — good, the spec is well-motivated; and F1 means the test must tolerate (or the spec must prevent) the tab switch that `pick_pass` causes. |
| 17 | G7: the cursor is requested by the gesture | **NOT RUNNABLE AS WRITTEN (first half); TWO REASONS (second half)** | `ui.py:661` is `app.want_cursor = None`, unconditionally, after the apply — so `app.want_cursor` read after `update_and_draw` is always `None`. Probed: `want_cursor after frame: None cur_cursor: None`; and with the canvas requesting a cursor, `after canvas requested ibeam: cur_cursor is ibeam? True`. So the parenthetical alternative ("or by asserting `app.cur_cursor` after the frame") is the only one that runs; strike the `want_cursor` phrasing. The middle-drag pan itself works headlessly (`pan changed by middle-drag: True (-144.0,-109.5) → (-124.0,-89.5)`). `at rest app.cur_cursor is None` has **two reasons**: nothing requested a cursor, or the canvas did not draw at all (a tab switch, a zero-size child). Anchor the at-rest assert to a frame where the canvas provably drew (`view.canvas_rect != (0,0,0,0)`). |
| 18 | G14: every kept binding still works | **TWO REASONS** | The substitution is mechanical (4 sites in `tests/test_graph_view.py:191,213,223,259`), and the suite is green today (`27 passed` over `test_graph_view/test_graph_state/test_ui_regions/test_theme`). But a red run after the substitution has two causes the row conflates: a genuine binding change, OR the tab machinery (a `pick_pass` tab switch per F1, `open_graph_for` frozen mid-copilot-turn interacting with `test_a_press_that_spans_a_copilot_turn_never_becomes_a_gesture`, which drives `copilot.state.in_flight` for three frames — T4 says `open_graph_for` is "frozen mid-copilot-turn like its sibling", so a re-open inside that test's turn window is a no-op). State which failures count as binding changes. |
| 19 | G12: the five channels | **PASSES WITHOUT THE FIX** | `channels_split(5)` + one `channels_merge()` says nothing about the channel ASSIGNMENT, which is the whole content of G12/S6 — the wrong implementation it lets through is any permutation of the five (halos over strokes, ✕ under nodes). The record's own sketch asked for more: "assert the five `channels_set_current` calls happen in the documented order"; the spec dropped that half. Restore it (a source-order grep of the `channels_set_current(n)` literals is enough and is the same kind as row 7). |
| 20 | T1-T6: the tab | **RUNNABLE** (nine sub-asserts; two riders) | Confirmed present and unchanged: `formatter_for` is `_FORMATTERS.get(tab_kind)` (`formatting.py:118`) so `formatter_for("graph")` is `None` for free; `is_tab_dirty` is `editor_sessions.get(tab.path)` → None → False (`app.py:1737`); `_on_document_deleted` filters `t.document_id != document_id or t.kind == "lib"` (`app.py:781-785`) so a graph tab of that document IS dropped and a lib tab survives; `close_editor_for_path` → `_close_tab_for_path` (`app.py:1612`); `jump_to_next_error` returns on `get_current_session_if_exists() is None` (`app.py:869`); `format_current_editor` likewise (`app.py:1767`); `command_callbacks` completeness is already gated by `test_every_command_id_has_a_handler`. `Alt+G` is free — no `K.g` in `COMMAND_SPECS`, and the vendored `standard_keymap.md` contains no `Alt+` chord at all, so `test_keymap_disjoint` stays green. `"Open graph"` reaches the help snippet automatically: `_shortcuts_section` generates from `COMMAND_SPECS` (`help_content.py:76-84`). Riders: (a) `tab_label` has no `"graph"` branch and would fall through to the multi-pass `pass_name_of(tab.path)` path on `graph.json` — the spec names the edit in Files-touched but the Verification row states the expected string `"<name> (graph)"` without saying it is a new branch; (b) `app.editor_focused = True` for a graph tab makes every `CommandScope.EDITOR` spec dispatchable on it (`hotkeys.py:334`), which is deliberate for `Ctrl+W`/`Ctrl+Tab` but silently also arms `Alt+C`/`Alt+R`/vim-ish editor commands against a tab with no session — worth one sentence in T2 and one assert. |
| 21 | Theme | **PASSES WITHOUT THE FIX** | See F3. |

### The three named gates, traced

- **G13's lock.** The named break (omit `lock_threshold` at the node-body site) **does**
  turn the named test red: today, with no threshold anywhere, a 5px move reads as a click
  (`5px move: pick_pass 1, set_pass_positions 0`) and only 8px reads as a drag. So the
  5px-is-a-drag assertion is red without the lock and green with it. Real gate.
- **G6's order.** The named break (reverse the port and node rungs) flips the row's FIRST
  case only — the mouse parked on a port centre would report `hovered_node` instead of
  `hovered_port`. That is a genuine red. But the wire rung is not guarded by the break as
  described: reversing *node and wire* leaves the port case green, and the row's node-body
  position ("away from any port") is also away from the wire, so a node↔wire swap may pass.
  Add a position where a node body and a wire overlap, or the gate's domain narrows.
- **G4's floor.** Dropping `max(6.0, ...)` gives `1.5·2·0.25 = 0.75` at z=0.25 against the
  row's asserted `6.0` — unambiguously red. Real gate, and the cheapest of the three.

---

## B. Blast radius

### Deleted / renamed symbols and every reader

| Symbol | Readers found | In `## Files touched`? |
|---|---|---|
| `PassesView` | `ui_regions.py:49`, `ui_models.py:39,273`, `tabs/document.py:30,455,463`, `tests/test_ui_regions.py:7,11,21-24`, `tests/test_graph_view.py:16,191,213,223,259`, `scripts/smoke.py:37,289,340` | **Yes**, all six files |
| `PASSES_VIEW_LABELS` | `ui_regions.py:56-59`, `tabs/document.py:30,458`, `tests/test_ui_regions.py:7,11,15` | Yes |
| `UIAppState.passes_view` | `ui_models.py:273`; readers via `app.app_state.passes_view` at `tabs/document.py:456,461,463`, `tests/test_graph_view.py` ×4, `scripts/smoke.py` ×2, `tests/test_ui_regions.py` ×3 | Yes. Salvage claim verified: `projects/dev/app_state.json` has **no** `passes_view` key (`False ['active_document_tab', ... 'telegram_default_pack']`), and `load_model` is per-key fail-soft (`ui_models.py:296-299`), so the retired key needs no migration — consistent with the no-backcompat rule. `tests/test_persistence_completeness.py`'s `("retired keys", ...)` case already covers the shape generically; no edit needed there. |
| `SIZE.GRAPH_BUS_STEP` | `theme.py:328`, `pass_graph.py:899,900` | Yes |
| `GRAPH_BUS_CLEAR` | `theme.py:338`, `pass_graph.py:898` | Yes |
| `GRAPH_LOOP_RISE` | `theme.py:340`, `pass_graph.py:568` | Yes |
| `GRAPH_LOOP_REACH` | `theme.py:341`, `pass_graph.py:571,572` | Yes |
| `GRAPH_MIN_H` | `theme.py:329`, `tabs/document.py:468` | Yes (T5/T3 both name it) |
| `_MIN_DIRECT_DX` | `pass_graph.py:72,547,548,555,556,895,1061` — **seven** sites, one of them the in-flight wire at `:1061` | Yes (module-local). The spec's S12 says "deleted in favor of the tokens"; the Verification table has **no row** for it (see §C). |
| `_BEZIER_BOW` | `pass_graph.py:74,548,1061` | Yes, same |
| `_draw_self_loop` | `pass_graph.py:562,906` | Yes |
| `_ellipsize` → `ellipsize` | `ui_primitives.py:26,391,577,1319,1741` (5 sites, not 4) **plus** `popups/lib_picker/tree.py:24,350`, `tests/test_pass_settings_layout.py:20,91`, `tests/test_anchored_note.py:15,53` | **NO** — see F4. Three files missing, and S11's "four call sites" undercounts the in-module ones by one (`:26` is the def; `:391, 577, 1319, 1741` are four calls — so "four" is right for calls and the def is a fifth line). |
| `pass_graph.draw`'s `height` | one caller, `tabs/document.py:470`; docstring at `pass_graph.py:790-791` names it | Yes |
| `_Edge.span` | `pass_graph.py` (the dataclass and the bus expression) — G3 says it goes | The spec does not name `_Edge.span`'s removal in Files-touched or Verification; S2 names only the two ADDED fields. Minor. |

### Added symbols — collision check

Grepped `shaderbox/ tests/ scripts/ dogfood/` for each; **zero hits for every one**:
`ellipsize` (as a bare name), `WireState`, `wire_points`, `OPEN_GRAPH`, `hand_cursor`,
`crosshair_cursor`, `wire_mids`, `x_rect`, `hovered_wire`, `selected_wire`, `hovered_node`,
`hovered_port`, `hovered_out`. No collisions.

`"graph"` as a tab kind: the only occurrence of the literal string `"graph"` anywhere in
`shaderbox/ tests/ scripts/ dogfood/` is `ui_regions.py:58` — the `PASSES_VIEW_LABELS`
entry being deleted. Nothing compares `tab.kind` to it, and `document.graph` /
`graph_json_for` / `graph_views` are attribute names, not that literal. Clean.

`SIZE.GRAPH_WIRE_HIT_MIN = 6` versus the existing `GRAPH_HIT_MIN = 7` (`theme.py:325`):
two similarly-named screen-pixel floors one apart, one read for ports and one for wires.
The record justifies the 6-under-7 relationship deliberately; the names invite a
transposition at a call site that nothing would catch. Consider `GRAPH_WIRE_HIT_MIN` →
something that cannot be misread for the port floor, or a comment naming the pair.

Binding signatures verified against the installed stub rather than memory:
`path_stroke(self, col, thickness=1.0, flags=0)` and
`path_arc_to(self, center, radius, a_min, a_max, num_segments=0)`
(`.venv/.../imgui/__init__.pyi:10558,10562`), so S10's Refinements row is right; pyright
passes a file using `path_arc_to` + `path_stroke(col, 2.0)` + `channels_split(5)` /
`channels_set_current(4)` / `channels_merge()` at **0 errors**, so no suppression is needed.
`imgui.Key.delete` and `imgui.Key.backspace` both exist (probe A/B fired on them).

### Docs quoting a number the spec changes

| Doc | Quote | Named by the spec? |
|---|---|---|
| `ai_docs/dev_flow.md:211` | "the caption, the `strip | graph` toggle and the `add pass` / `import...` row are the Document tab's" | Yes (module map named) |
| `ai_docs/dev_flow.md:217` | "`widgets/pass_graph.py` — the graph canvas (feature 092), the Document tab's second view of" | Yes |
| `ai_docs/conventions.md:856-857` | "Revisit the canvas's home when the editor pane can host it (the planned pane swap)" | Yes — and the clause is real; I verified the wording |
| `ai_docs/conventions.md:483-485` | "Revisit if a tab needs durable per-tab state ... **or a 4th editable `kind` lands**" | **No.** A graph kind is a 4th `kind` (non-editable, which is arguably why the clause does not bite) — the spec should either say why it does not, or amend it. |
| `shaderbox/widgets/pass_list.py:178` | docstring: "the add / import row are the Document tab's (092 D2), shared with the graph view" | **No.** T5 keeps the row on the Document tab, so the sentence stays half-true ("shared with the graph view" is now wrong — the graph has its own tab). One-line fix, unnamed. |
| `shaderbox/help_content.py:180` | "or none, in the gear. On the graph view a port exists because the shader" | Yes (help_content named) |
| `shaderbox/theme.py:311` | comment "The graph canvas (092 D7): a compact node, its picture, one port row, the dot and the" | Yes (S12 says the comment block is rewritten) |
| `shaderbox/widgets/pass_graph.py:790-791` | `draw`'s docstring names the `height` parameter and "the caller sizes it" | Yes (T3) |
| `shaderbox/widgets/pass_graph.py:10-14` | module docstring's hit-order paragraph (background → nodes → ports) — S6/S7 add two rungs after the ports | Yes (the spec says "the module docstring is updated") |

The specific numbers 108, 80, 6, 18, 16 appear **only** in `theme.py` (`:315-341`) and in
092's own review files, which are historical records and correctly left alone. No live doc
quotes the literals. `tests/test_graph_state.py:58-59` reads `SIZE.GRAPH_NODE_W` and
`SIZE.GRAPH_PORT_ROW` symbolically, so it survives the token bump — good.

---

## C. What nothing verifies

Each item below is a spec guarantee with no Verification row and no existing test.

| Guarantee | Deserves a check? | Why / how |
|---|---|---|
| **S2's revalidation of `selected_wire`** ("cleared when no drawn wire carries it — an unwire, a pass delete, a scope change") | **Yes, cheap.** | Exactly the shape of `revalidated_scope`, which HAS a pure test (`test_a_scope_no_pass_carries_falls_back_to_the_root`). Make it a free function `revalidated_wire(selected, edges) -> tuple|None` in `graph_state.py` and pin it pure. Without it, a stale `selected_wire` plus a Delete calls `unwire` on a sampler no longer drawn — `App.unwire` would refuse, but the ✕ would still be drawn at a stale `x_rect`. |
| **S3's "every field is written every frame, `None` included"** | **Yes, and it is the one most likely to rot.** | This is the "checker that quietly narrows its own domain" family. Enumerate it from the dataclass: assert the set of hover fields on `GraphViewState` is exactly the four the canvas writes, then drive one frame with the mouse off-canvas and assert all four are `None`. The row-15 "off the canvas: all `None`" assert covers the *off-canvas* case but not the *a fifth field was added and nobody wired it* case. |
| **S4's exclusivity** (selecting a node clears the wire; a band's release clears the wire) | **Yes.** | Two asserts on top of the row-12 sequence: after selecting the wire, click a node → `selected_wire is None`; after a band release → `selected_wire is None`. Both are the reason "one Delete has one target", and both are currently unasserted. Blocked on F1 for the node half. |
| **S5's `not blocked` gate** | **Yes, and the existing pattern is ready.** | `blocked = frozen or view.press_blocked`. `test_a_press_that_spans_a_copilot_turn_never_becomes_a_gesture` already drives `app.copilot.state.in_flight`; add three frames with a wire selected, `in_flight = True`, Delete → no `set_sampler_source`. One `with mock.patch.object` block. |
| **S7's shared order** (the node buttons submitted in the same sorted order the draw uses) | **Yes, and a source check is not enough.** | The row-19 replacement I propose covers `channels_set_current`; S7 is about ONE sorted list used by two loops. A structural check: assert the module contains exactly one `sorted(` over the node list and that both loops iterate the same local name — brittle. Better: expose `view.node_order: list[str]` (written once per frame) and assert a selected node is last in it, then assert the overlap click lands on it. The record's own sketch asked for "assert the sorted node list puts the selected one last", which the spec dropped. |
| **T2's `app.editor_errors = []`** | Covered — the T1-T6 row's last clause asserts it. | No gap. |
| **T5's tick** (`_entry_row_label(graph_active, "Passes")` lights when THIS document's graph tab is active) | **Maintainer's eyes is defensible, but a cheap assert exists.** | `graph_active` is a pure predicate over `app.active_tab`; assert it directly (open the graph tab of document A, switch to B, assert the tick predicate is False). The drawn tick itself is a look. The Script row's equivalent has no test either, so this is consistent — say so rather than leaving it silent. |
| **T6's close assertion** | Covered — T6 states it and the T1-T6 row asserts `close_editor_for_path` removes it. | But note the smoke's frame 47 currently ends with `app.app_state.passes_view = PassesView.STRIP` (`smoke.py:340`); T6 replaces that with the close. Make sure frame 48's document switch still finds the canary document current — it does (`smoke.py:342-344` is independent). |
| **S12's `_MIN_DIRECT_DX` / `_BEZIER_BOW` deletion** | **Yes, one greppable line.** | Row 7 greps for `_draw_self_loop` and `bus_y` but not for these two, and `_MIN_DIRECT_DX` has **seven** read sites including the in-flight wire at `:1061` — the single likeliest place for one to survive the rewrite, because it is 150 lines away from `_draw_wire` and outside the edge loop. Add both names to row 7's source-grep list; it costs nothing and guards the one straggler. |
| **The `is_any_item_active()` gate's interplay with the canvas's own invisible buttons** | **Probed, and the answer is: it is safe at the END of `_draw_canvas` and broken at the TOP.** | See F7 and row 14. On the frame Delete is pressed one frame after a click-and-release, `is_any_item_active()` is False at the end (probe: `end_active_hovered=(False, True)`) — so the gate does not spuriously suppress Delete. On the frame the release itself lands, the top reads True. While the button is still HELD, both reads are True and Delete is correctly suppressed (`HELD+DELETE: end=(True, False)`). **So: no, the background button is not ACTIVE at the end of the frame after a click — but it IS at the top of the release frame.** Pin the read position. |
| **G8's "ghost nodes draw no glyph"** (S10's last sentence) | No — leave to the maintainer's eyes. | A negative about a draw with no state to read. |
| **S14's "requests only, never a raw `glfw.set_cursor`"** | **Yes, one grep.** | The single-owner rule is exactly the kind a source grep gates: assert `glfw.set_cursor` does not appear in `pass_graph.py`. `ui.py:659` is the one legitimate caller. |

---

## D. The gate

**`make smoke` PASSES on this box** — run once, unpiped, with the gate's own env:

```
$ SHADERBOX_SMOKE_SKIP_EXIT=87 uv run python scripts/smoke.py; echo EXIT=$?
... smoke: OK (200 frames, 7 documents)
EXIT=0
```

So a skip is not the outcome here and `xvfb-run` is not needed. The spec's requirement
("`make gates` green with the smoke run (not skipped)") is achievable as stated; judge it by
`make gates > /tmp/g.log 2>&1; echo $?` per `CLAUDE.md`.

**The xdist group for `tests/test_graph_tab.py`.** `pyproject.toml:91-96` states the rule
("Each module that drives real frames carries its OWN `xdist_group`, so no two of them
share a process: the imgui font atlas is process-global"), and the four existing
frame-driving modules declare one: `tests/test_code_panel.py:19` is
`pytestmark = pytest.mark.xdist_group("gl_frames_code_panel")`, plus
`gl_frames_render_decoupling`, `gl_frames_projects`, `gl_frames_profiling`. A new
frame-driving `tests/test_graph_tab.py` must therefore carry
`pytestmark = pytest.mark.xdist_group("gl_frames_graph_tab")` — the spec does not say so,
and the omission is invisible under `-n 8 --dist loadgroup` until an unlucky worker split.

**Related gap, pre-existing and now made worse.** `tests/test_graph_view.py` carries **no
`pytestmark`** despite driving real frames in two tests (`_frames` → `update_and_draw`). It
is green today by luck of the split. This wave adds a second frame-driving graph module and
extends the first, which raises the odds of the two landing in one worker. Give
`test_graph_view.py` a group in this wave (`gl_frames_graph_view`), or put the new tests
IN it rather than in a new module. This is a real gate defect the spec should name.

**`tests/test_ui_prose_budget.py` accepts all three new strings.** Ran the module's own
scorer on each (`probe_prose.py`, importing `_score` and `_visible` from the test module):

```
'"Open graph"'           score= 2  visible= Open graph
'"open##entry_graph"'    score= 1  visible= open
'"Open the pass graph"'  score= 4  visible= Open the pass graph
```

Against the budgets: `standard_button`'s `label` is scored at `_BUTTON_LABEL_BUDGET = 3`
(so `"open"` at 1 passes), `set_tooltip`'s `text` at 5 (so `"Open the pass graph"` at 4
passes), and none of the three contains a `_CLAUSE_JOINERS` member (`";"`, `" — "`,
`" -- "`). `"Open graph"` is a `CommandSpec.label`, which is **not in the gate's domain at
all** (the domain is `ui_primitives` signatures plus four explicit `imgui.*` rows), so it is
unscored either way. No `_OVER_BUDGET` or `_UNMEASURABLE` entry is needed.

**`tests/test_ui_regions.py`'s deletion loses nothing measurable.** Its docstring claims it
also covers "The `ChannelView` trio's shape", but the file's three tests are all
`PassesView` (`:10, 14, 19`) — there is no `ChannelView` assertion in it. So deleting it
drops no coverage of the surviving enums. Worth one sentence in T5 so a reviewer does not
have to re-derive it.

---

## False trails — checked, fine

- **`Alt+G` collision.** No `K.g` anywhere in `COMMAND_SPECS`; the vendored
  `standard_keymap.md` contains **no** `Alt+` chord at all, and the vim doc only yields
  `CTRL-`/`<C-` forms, so `test_keymap_disjoint` cannot see it. `chord_needs_modifier`
  accepts it (it has `mod_alt`). Free and safe.
- **`C.EDITOR` as a scope.** The spec writes "category `C.EDITOR`, the same default scope as
  `OPEN_SHADER` / `OPEN_SCRIPT`". `CommandSpec` (`commands.py:80-90`) has `category` at
  position 4 and `scope: CommandScope = CommandScope.GLOBAL` at 5; `OPEN_SHADER` and
  `OPEN_SCRIPT` (`:147-148`) pass only the category. So `C.EDITOR` + default GLOBAL scope is
  exactly the siblings' shape, and the command is NOT gated on `editor_focused`. Correct as
  written.
- **`_fit` needing a frame.** It takes `avail: imgui.ImVec2` but makes no imgui call; probed
  outside a frame at `zoom 1.0, pan (-240.0, -223.0), fitted True` with `_Xf.to_screen`
  working. The "app fixture, no frames" kind is right for the fit rows (modulo F2's
  containment target).
- **`formatter_for("graph")`, `_on_document_deleted`, the `xdist_group` marker,
  `path_stroke`'s "(private API)" label, `projects/dev/app_state.json` carrying
  `passes_view`, the wire midpoint being unclickable, `GRAPH_WIRE_BOW` folding at
  `dist = 0`.** Each checked; each fine, and each already cited in §A/§B/§D above where it
  supports a verdict.

---

## Verdict

Three of the twenty-one Verification rows are green against today's code or a plausible
wrong implementation (F3 theme, F8 six-column fit, row 19 five channels), one more discriminates
only down to a third of its stated number (F9), two are not runnable as written (F2 the fit's
containment target, row 17's `want_cursor` read), and one live behaviour question is
undecided in a way that breaks two frame-driven rows (F1: a node click summons the pass's
shader tab and evicts the graph tab). None of that is a design objection — G1's algebra, G4's
floor, G13's lock and S8's premise all check out numerically, and the two gates I could trace
end-to-end (G13's lock, G4's floor) do go red on their named break. The fixes are row
rewrites plus one spec sentence each for F1 and F7, not a redesign.

**VERDICT: PARTIAL** — land after deciding F1 (what a node click does to the graph tab)
and F7 (where in `_draw_canvas` the Delete read sits), and after rewriting the four rows
named in F2, F3, F8 and row 19 so each can only pass for the reason it names.

---

# Round 2

Re-read `01_spec.md` from disk at `3f75d78` in full (499 lines). Every round-1 finding is
addressed by quoted spec text; two of the fixes introduced a new defect and one older claim
the revision inherited turns out to be measurably false. Probes: `test_r2a.py`..`test_r2f.py`
under the scratchpad.

## New findings

**N1 (BLOCKER). The card is too narrow for the very names it was widened for — the record's
font advance is wrong, and it is the same error this repo already fixed once.** S1 ships
`GRAPH_NODE_W = 128` on G11's table, which gives `distance_field` as 107.0px at 14 bold and
`u_distance_field` as 104.8px at 12, from "the shipped font's measured advance (0.545898 ×
em, monospace)". Measured in a rig frame against the real rasterized faces (`test_r2f.py`):

```
font_12:      legacy_size=12.0  advance/char=7.0000  ratio=0.583333
    'u_distance_field'   =  112.00  (16 chars)
font_14_bold: legacy_size=14.0  advance/char=8.0000  ratio=0.571429
    'distance_field'     =  112.00  (14 chars)
```

The advance is **7.0 at 12px and 8.0 at 14px, not 6.55 and 7.64**. So at `GRAPH_NODE_W = 128`
with `GRAPH_PAD = 8`:

| Name | Record says | Measured | Budget at 128 | Fits? |
|---|---|---|---|---|
| `distance_field` @14 bold | 107.0 | **112.00** | `128 - 2*8 = 112.0` | tie — `_ellipsize` uses `<=`, so kept, with **0.0px** of slack |
| `u_distance_field` @12 | 104.8 | **112.00** | `128 - (2*4+2) - 8 = 110.0` | **NO — truncates** |

Probed end-to-end through the real helper (`test_r2e.py`):

```
W=  128 port_budget= 110.0 u_distance_field measures 112.00 -> 'u_distance_f...'
        name_budget= 112.0 distance_field   measures 112.00 -> 'distance_field'
```

So the Verification row's own pin — "at the 128 budget returns the string unchanged... This
is the width decision's pin -- red at 108, green at 128" — is **red at 128 for the port-label
half**. The goal section's promise ("a card wide enough for his own longest names") is not met
by 128, and the maintainer's two real names are exactly the two that fail.

This is not a new class. `tests/test_pass_settings_layout.py:74-77` carries the same lesson in
its docstring: *"Measured against the real rasterized face inside a frame, never a hard-coded
em ratio: the first version of this check assumed 6.5508px per character where the 12px face
advances 7.0, so it passed a 19-character name that visibly truncates."* `0.545898 × 12 =
6.5508` — the record's ratio **is** that already-refuted number. The fix is a width, not a
test edit. Smallest multiples of 4 that clear both at `PAD = 8`:

```
W= 128 PAD=8: name_budget= 112.0 (need >112.0: NO)  port_budget= 110.0 (need >112.0: NO)
W= 132 PAD=8: name_budget= 116.0 (OK)               port_budget= 114.0 (OK)
W= 136 PAD=8: name_budget= 120.0 (OK)               port_budget= 118.0 (OK)
```

**136 is C's original recommendation**, which the record overrode using the bad arithmetic
("The width call (128, against C's 136 and the fit numbers). The fork is real and the
arithmetic decides it"). The arithmetic decided it wrongly, so C's number stands. Set
`GRAPH_NODE_W = 136` (or 132 for the tightest fit with ~2px of slack; 136 gives 6-8px and
survives a font bump), re-run the fit-clamp row's numbers, and keep the ellipsis row as the
pin — it then bites for the right reason. The fit still clears a six-column chain at 136:
`6*136 + 5*64 + 32 = 1168`, so `1225/1168 = 1.049` still clamps to 1.0, and `740/1168 = 0.634`
is still inside `0.6 < zoom < 1.0`, so that row's asserts survive the change unedited.

**N2. S5's named break does not turn its own test red.** The row says: *"Break to try: drop
`not is_any_item_active()` -- the held case writes"*, and S5's prose says *"the clause `not
is_any_item_active()` is what refuses the key while a press is HELD on the canvas ... and that
is its falsifier."* Measured at S5's pinned read position with a press held on empty canvas,
evaluating all four gate variants on the same frame (`test_r2c.py`):

```
HELD-PRESS gate outcomes (1 = it would write): {'full': 0, 'no_active': 0, 'no_hovered': 0, 'neither': 1}
```

Dropping `is_any_item_active()` alone still yields **0** writes, because `hovered` is
*independently* False while the press is held — `is_window_hovered(child_windows)` reads False
for the whole drag (`test_r2b.py`: `HELD {'any_active': True, 'hovered': False, ...}`). Only
dropping **both** clauses lets the held case through. So this gate passes whether or not the
clause it names is present, which is the exact family `CLAUDE.md` calls the most expensive.
Two ways out: name the break as "drop both `hovered` and `not is_any_item_active()`" and say
the two clauses are redundant for THIS case; or find a case where they separate — a press held
on a node while the mouse stays inside the child would be one if `hovered` recovered, but it
does not (measured), so the honest statement is that `hovered` is the clause that refuses the
held press and `is_any_item_active()` is belt-and-braces. Either way the row must not claim a
falsifier it does not have.

**N3. The T1-T6 row's last clause PASSES WITHOUT THE FIX.** The row asserts *"no
`editor_sessions` key at the graph path after the Uniforms tab has been focused for those
frames (T1's non-creating getter)"*. But `_locate_uniform_declaration` is reached only past
`widgets/uniform.py:67`:

```
    if not (clicked or imgui.is_item_hovered()):
        return
    located = _locate_uniform_declaration(app, name)
```

Probed (`test_r2d.py`) with the Uniforms tab focused for four frames and no mouse over a name:

```
UNIFORMS_TAB_FOCUSED_NO_MOUSE: _locate calls 0 get_current_session calls 0
```

So the row is green with or without T1's edit. The underlying risk is real — driving the
creating getter on a graph-path tab does make a session:

```
get_current_session on graph.json created a session? True -> editor_sessions now has graph.json: True
```

— so T1's edit is needed; the test just does not exercise it. **The input the test must
inject**: park the mouse on the uniform-name cell, whose imgui id is `uname_<name>`
(`widgets/uniform.py:59`, `clickable_label(..., id_=f"uname_{name}")`). There is no rect
exposed for it the way `port_rects` is, so either expose one, or drive
`_locate_uniform_declaration(app, "<a uniform of the document>")` directly with a graph tab
active and assert no `editor_sessions` key appears — a unit call on the function the edit
touches, which is the cheaper and more direct falsifier.

**N4. T5's tick predicate is not a nameable predicate today, so the row cannot be called as
written.** The row says *"the Passes row's `graph_active` is True for A and False for B (the
predicate is pure over `app.active_tab`)"*. Its model, `script_active`, is computed **inline**
inside `_draw_entry_points` (`tabs/document.py:414-418`):

```
    script_active = (
        active is not None
        and active.kind == "script"
        and active.document_id == document_id
    )
```

and no `*_active` helper exists anywhere in `tabs/document.py` or `app.py` (grep returns only
`close_active_tab`, `set_active_tab`, `_reanchor_active_tab`, `_active_tab`). So the row is
runnable only if the implementation extracts the predicate — e.g. a free
`tab_active_for(tab: EditorTab | None, kind: EditorTabKind, document_id: str) -> bool` in
`tabs/document.py` that both rows call. Say so in T5, or the row becomes a test nobody can
write and gets dropped at implementation time. (Extracting it also removes the copy-paste
between the Script and Passes rows, which is the reason to do it anyway.)

## 1. Closure, item by item

### Round-1 findings

| # | Closing spec text (quoted) | Status |
|---|---|---|
| **F1** `pick_pass` evicts the graph tab | **S15**: "`App.pick_pass` splits into `App.choose_output(document_id, name)` (the `set_output_pass` half with its toast) and `pick_pass` = `ensure_shader_tab` + `choose_output`, unchanged for the strip, the uniforms row and the copilot. The canvas's `_click` calls `choose_output`". | **CLOSED**, and verified: `set_output_pass` (`project_session.py:1045-1054`) writes `document.graph` and calls `save_ui_document`, firing no `on_*` callback; `save_ui_document` (`:435-450`) is a disk write plus an mtime rebaseline. Probed the split (`test_r2a.py`): `BEFORE (0, 1, ['main.frag.glsl'])` → `3px: set_output_pass 1 set_pass_positions 0` → `AFTER (0, 1, ['main.frag.glsl'])`, `canvas_rect nonzero after: True`, `output pass now: c`. No tab side effect. The Uniforms claim also holds by code: `App.panel_pass` (`app.py:756-761`) requires `tab.kind == "shader"` and falls to `document.render_pass` otherwise, so with a graph tab active the panel shows the output — probed `panel_pass default: ''` unchanged after `set_output_pass('c')`. |
| **F2** `canvas_rect` unreadable without a frame | **S8** row: "the fitted window in canvas space is `(pan.x, pan.y, pan.x + 800 / zoom, pan.y + 600 / zoom)`, and every wire's 25 sampled canvas-space curve points lie inside it." | **CLOSED.** `_fit` writes `pan` and `zoom` outside a frame (probed round 1: `zoom 1.0, pan (-240.0, -223.0)`), and the target is no longer `canvas_rect`. The "Break to try: fit the nodes alone" clause makes it a real check. Also closed the harder half I did not ask for: S8 now frames the **sampled curve** rather than the control polygon, with the over-framing measured ("inflates the fitted width from 605 to 957px"). |
| **F3** theme row passes without the fix | **S1**: "`blue_b` IS the `blue` accent preset's primary (`theme._ACCENTS["blue"]`) ... the record's G6 sentence 'is not an accent primary' is wrong ... `COLOR.GRAPH_HOVER = _P["fg_0"]`". **Theme row**: "imports `_GROUP_TINT_EXCLUSIONS` ... `GRAPH_HOVER not in {primary for primary, _, _ in _ACCENTS.values()}` ... Break to try: `blue_b` -- the accent clause goes red". | **CLOSED, and the revision found more than I did.** My round 1 checked `blue_b` against TAG and the exclusion set but **not** against `_accent_primaries`; the spec is right: `blue_b in accent primaries: True`. And `fg_0` is **not** already in `_GROUP_TINT_EXCLUSIONS` (`fg_0 in _GROUP_TINT_EXCLUSIONS already: False`), so the edit is now a real change rather than a no-op. The named break bites exactly once: `fg_0 -> ALL GREEN`; `blue_b -> RED at: not in accent primaries`. The hand-typed-set half is closed too ("imports `_GROUP_TINT_EXCLUSIONS`"). |
| **F4** `_ellipsize` readers outside `ui_primitives` | **S11**: "every reader follows: its four call sites inside `ui_primitives`, `popups/lib_picker/tree.py`, `tests/test_pass_settings_layout.py`, `tests/test_anchored_note.py`." Files-touched adds `shaderbox/popups/lib_picker/tree.py`, `tests/test_pass_settings_layout.py`, `tests/test_anchored_note.py`. | **CLOSED.** All three named; the count now matches the grep. |
| **F5** `calc_text_size` segfaults between frames | **Verification preamble**: "A measurement that needs a font (`ellipsize`, `calc_text_size`) runs inside `imgui.new_frame()` / `imgui.begin("rig")` / `push_font` / `imgui.end()` / `imgui.end_frame()` on the `app` fixture, the shape of `tests/test_pass_settings_layout.py::test_the_auto_name_column_fits_every_engine_uniform`; outside a frame `calc_text_size` segfaults the process (measured)." | **CLOSED**, and the cited pattern is real and green (`3 passed`). The row's kind is now "rig frame". |
| **F6** key-after-mouse batching | **Verification preamble**: "a key event queued in the same batch as a mouse-button event reaches `is_key_pressed` one frame LATER than the button (measured), so a `_frames` call separates them". The G5 row now reads "click ..., release, frames, send `Key.delete`, frames, assert". | **CLOSED.** |
| **F7** where the Delete read sits | **S5** title: "read locally, once, AFTER the hit-test section and the drag blocks, before `_canvas_menu`", with the reason: "on a release frame `is_any_item_active()` is True at the top of `_draw_canvas` and False at the bottom, and `is_window_hovered` the reverse (measured), so a read at the top is dead on exactly the frame a user who just clicked the wire presses the key." | **CLOSED** (the position is pinned and the measurement recorded). The clause-attribution inside it is **N2**, a new defect. |
| **F8** six-column fit pins nothing | Row renamed "`_fit`'s clamp (regression check, not a width pin)" with "Green at 108 too; the width is pinned by the ellipsis row". | **CLOSED as stated** — the row no longer claims to pin the width. But the pin it hands off to is **N1**, which is red at 128. |
| **F9** 24 segments discriminate only to 8 | Row renamed "the flattening misses no real hit (regression guard)" with "this discriminates a count of 6 or below (worst error 9.5px at 6, 2.7 at 8), so it guards against a catastrophic count, not the choice of 24". | **CLOSED**, and the numbers quoted match mine exactly (9.477 at 6, 2.666 at 8). |

### Round-1 row verdicts other than RUNNABLE

| Round-1 row | Closing text | Status |
|---|---|---|
| Row 2 continuity, thin margin | "each control point moves by less than `1.0 + 2 * GRAPH_WIRE_BOW` px per step (the endpoint's own 1px plus the offset's change ...). Measured today's worst step at these tokens: 1.31px" | **CLOSED** — the bound is now derived from the token, so a bow change moves the bound with it. |
| Row 4 → F9 | above | CLOSED |
| Row 10 → F2 | above | CLOSED |
| Row 11 → F8 | above | CLOSED (but see N1) |
| Row 15 hover, TWO REASONS | Row split into five lettered sequences, "one rung per sequence, no click in any", with "(d) place `c` (through `app.session.set_pass_positions`) so its body covers `wire_mids[("b", "u_src")]`" and "Breaks to try: swap the port and node rungs (a flips); swap the node and wire rungs (d flips)". | **CLOSED.** The "no click in any" clause removes the F1 contamination and the (d) position gives the node↔wire swap a falsifier. Geometry verified below. |
| Row 17 cursor, NOT RUNNABLE + TWO REASONS | **S14**: "`ui.py` applies once per frame on change and resets `want_cursor` to `None`, so a test reads `app.cur_cursor` after the frame." Row: "during a middle-drag pan, after the frame, `app.cur_cursor is app.hand_cursor`; at rest, on a frame where `view.canvas_rect != (0, 0, 0, 0)`, `app.cur_cursor is None`". | **CLOSED** on both halves — the `want_cursor` phrasing is gone and the at-rest assert is anchored to a frame that provably drew. |
| Row 18 G14, TWO REASONS | "a failure that traces to a binding is a silent change, a failure that traces to the tab being inactive is a test-mechanics bug (the tab is opened before any copilot-turn simulation, since `open_graph_for` is frozen during one)" | **CLOSED** — the two causes are now named and separated, and the `open_graph_for`-frozen interaction I flagged is called out. |
| Row 19 channels, PASSES WITHOUT THE FIX | "the first occurrence of each `channels_set_current(k)` literal for k in 0..4 appears in ascending source order (halos, strokes, nodes, in-flight, overlays)" | **CLOSED** — the assignment order is now checked. |
| Row 20 T1-T6 riders (a) `tab_label`, (b) `editor_focused` arming EDITOR scope | **T1**: "Two edits are REQUIRED for the claim to hold, not implied: `tab_label` gains a `"graph"` branch returning `f"{document_name} (graph)"` before the multi-pass fallthrough (which would call `pass_name_of` on `graph.json`)". **T2**: "it also makes every `CommandScope.EDITOR` spec dispatchable on the tab, which today is those two plus `FORMAT_BUFFER`, whose handler returns on the missing session." | **CLOSED** on both. T2's enumeration is correct — `COMMAND_SPECS` has `OPEN_SHADER`/`OPEN_SCRIPT` at default GLOBAL scope, so the EDITOR-scope set really is small. |
| Row 21 theme → F3 | above | CLOSED |

### Section C items

| Item | Closing text | Status |
|---|---|---|
| S2 revalidation | **S2**: "revalidated every canvas frame through a pure `graph_state.revalidated_wire(selected, edges) -> tuple[str, str] \| None` (the shape of `revalidated_scope`)"; row "S2: a stale wire selection clears". | **CLOSED** as a pure row, exactly the `revalidated_scope` shape I proposed. |
| S3 every field written | Row "S3: the hover fields are exactly the ones written ... the set of `GraphViewState` fields whose name starts with `hovered_` is exactly `{...}` ... a fifth field nobody wires is caught". | **CLOSED** — enumerated from the dataclass, which is the anti-narrowing shape. |
| S4 exclusivity | Row "S4: the selections are exclusive ... after selecting the wire, click a node: `selected_wire is None` ...; select the wire again, rubber-band over empty canvas and release: `selected_wire is None`". | **CLOSED.** |
| S5 `not blocked` | Row "S5: Delete is refused during a copilot turn ... `app.copilot.state.in_flight = True` ... no write (the `not blocked` clause)". | **CLOSED.** |
| S7 shared order | **S7**: "exposed as `view.node_order: list[str]` (node keys, this frame) so a test can assert the selected node is last"; row "S7: the selected node draws and hit-tests last ... `view.node_order[-1] == "p:a"`". | **CLOSED** via the `node_order` field I proposed. |
| T2 `editor_errors = []` | Already covered in round 1; still in the T1-T6 row. | CLOSED |
| T5 tick | Row "T5: the tick predicate". | **STILL OPEN — N4**: the predicate does not exist as a callable thing. |
| T6 close assertion | **T6**: "frame 47 ends with `app.close_editor_for_path(...)` in place of the `passes_view = STRIP` reset ... The tail's `get_current_session_if_exists() is not None` assertion ... holds because frame 48's `set_current_document_id(canary_id)` runs `_on_current_document_changed` -> `ensure_shader_tab`". | **CLOSED**, and it answers the frame-48 question I raised with the mechanism rather than an assertion. |
| S12 `_MIN_DIRECT_DX` deletion | Row "the widget's source contains none of `_draw_self_loop`, `bus_y`, `_MIN_DIRECT_DX`, `_BEZIER_BOW`, `glfw.set_cursor`". | **CLOSED** — all four names plus S14's `glfw.set_cursor` rule in one grep row. |
| `is_any_item_active()` vs the canvas's own buttons | **S5**: "the clause `not is_any_item_active()` is what refuses the key while a press is HELD on the canvas (the background, a node or a port button is active), and that is its falsifier"; row "S5: Delete is refused while a press is held on the canvas". | **STILL OPEN — N2**: the row exists and the behaviour is right, but the named falsifier is not one. |
| S14 no raw `set_cursor` | In the S12/G8 grep row (above) and S14's "`glfw.set_cursor` never appears in the widget". | **CLOSED.** |

### Section D items

| Item | Closing text | Status |
|---|---|---|
| Smoke passes, not skipped | "`make gates` green with the smoke run (not skipped -- it passes on this box without `xvfb-run`, measured)". | **CLOSED.** |
| New module's xdist group | Files touched: "new `tests/test_graph_tab.py` with `xdist_group("gl_frames_graph_tab")`"; gates paragraph: "Every new frame-driving test module declares its own `xdist_group`". | **CLOSED.** |
| `test_graph_view.py` has no group | Files touched: "`tests/test_graph_view.py` (gains `pytestmark = pytest.mark.xdist_group("gl_frames_graph_view")` -- it drives frames today with no group, green by luck of the worker split)". | **CLOSED**, with my reason quoted. |
| Prose budget accepts the three strings | Not restated in the spec — correctly, since it was a clean result needing no change. | CLOSED (no action was required). |
| `test_ui_regions.py` deletion loses nothing | **T5**: "its three tests cover only the retired enum (its docstring's `ChannelView` claim has no test behind it; `tests/test_channel_view.py` covers that enum)". | **CLOSED**, and verified: `tests/test_channel_view.py` exists and imports `CHANNEL_VIEW_LABELS`, `ChannelView`, `next_channel_view`. |
| `conventions.md:483` "4th editable kind" | Files touched: "the editor-tab bullet's "a 4th editable `kind` lands" trigger gains the clause that a non-editable kind -- the graph -- does not fire it, since it has no session". | **CLOSED.** |
| `pass_list.py` docstring | Files touched: "`shaderbox/widgets/pass_list.py` -- its docstring's "shared with the graph view" clause about the add / import row is corrected". | **CLOSED.** |

## 2. The rewritten Verification table

| # | Row | Verdict | Note |
|---|---|---|---|
| 1 | G1 no cusp | RUNNABLE | Unchanged from round 1. |
| 2 | G1 continuity | RUNNABLE | Bound now token-derived (`1.0 + 2 * GRAPH_WIRE_BOW = 1.8` against a measured 1.31). |
| 3 | G4 threshold + floor | RUNNABLE | Break added and it bites: `(0.25)` reads 0.75 without the floor. |
| 4 | G4 flattening regression guard | RUNNABLE | Honestly labelled; numbers match my measurement. |
| 5 | G18 wire_state ×16 | RUNNABLE | |
| 6 | S2 stale wire clears | RUNNABLE | Pure, `revalidated_scope`'s shape; `revalidated_scope` has a green test of exactly this form. |
| 7 | G6 nothing changes size | RUNNABLE | `signature(node_size).parameters` is `('port_count', 'box')` today. |
| 8 | G8/G3/S12 removals | RUNNABLE | Five names, all greppable today; `_MIN_DIRECT_DX` has 7 sites, `bus_y` 3. |
| 9 | S13 the lock | RUNNABLE | 4 + 1 sites today, all thresholdless. |
| 10 | G12 five channels in order | RUNNABLE | The assignment order is now the assertion. |
| 11 | G11 the ellipsis pins the width | **NOT RUNNABLE AS WRITTEN** | **N1.** Measured: `u_distance_field` is 112.00 against a 110.0 port budget at 128 → `'u_distance_f...'`. The row's "green at 128" half is red. Red at 108 holds. |
| 12 | S8 the fit frames every wire | RUNNABLE | Target restated in canvas space; `_fit`/`_Xf` run frameless (probed round 1). Break clause present. |
| 13 | `_fit`'s clamp | RUNNABLE | Correctly demoted. Survives N1's width change unedited (`1225/1168 = 1.049` clamps; `740/1168 = 0.634`). |
| 14 | G5 select + Delete | RUNNABLE | Frame order fixed per F6. Midpoints land on open background (probed round 1). |
| 15 | G5 the ✕ unwires | **RUNNABLE** — probed | The new clause "the press started no band and no drag (`band_anchor is None`, `node_drag is None`)" holds. Splicing S6's latch in at the top of `_draw_canvas` (where the check is pinned, before the background button) and then dragging 160px (`test_r2b.py`): `PRESS_FRAME ['LATCHED', ('end', True, None, False, False, [])]`, `DRAG_FRAME_1/2/4 ('end', True, None, False, False, [])`, `AFTER_RELEASE ('end', False, None, ...)`. The unarmed control DOES start a band on the same gesture: `UNARMED_BAND ('end', False, (2324.0, 1375.0), ...)`. So `press_blocked` set before the background button really does survive the button's own `is_item_clicked`/`is_item_active` reads for that frame and every later frame of the press, `selection` stays empty, and the latch clears on release. |
| 16 | S5 Delete refused while a press is held | **PASSES WITHOUT THE FIX** | **N2.** The behaviour is right, but dropping the named clause alone leaves the test green (`no_active: 0`); `hovered` refuses it independently. |
| 17 | S5 Delete in the group prompt | RUNNABLE | Correctly re-attributed to `hovered` and labelled "Behaviour pin"; probed round 1 (`any_active` True, and now known `hovered` is False too). The one-shot caveat is carried. |
| 18 | S5 Delete during a copilot turn | RUNNABLE | The `blocked` path is the one `test_a_press_that_spans_a_copilot_turn...` already drives. |
| 19 | G6 exclusive hover (a)-(e) | **RUNNABLE** — geometry verified | (d) is satisfiable at the decided tokens. The `a->b` wire (key `("b", "u_src")`, the consumer being `b`) runs `(128.0, 56.0) -> (192.0, 145.0)`, offset 43.85, midpoint **(160.0, 100.5)**. Placing `c` at `(96.0, 23.5)` puts its 128×154 body over that point; `c`'s own port is **93.4px** away and its output **67.4px** (both far outside the 7px port box), and the `b->c` wire passes **26.9px** from it (outside the 6px wire threshold). Crucially the midpoint does not move: `wire_points` reads only the two endpoints, and `a` and `b` are untouched. So (d) isolates the node↔wire rung. |
| 20 | S3 hover fields enumerated | RUNNABLE | |
| 21 | S4 selections exclusive | RUNNABLE | Unblocked by S15 (the node click no longer switches tabs). |
| 22 | G13/S15 3px vs 5px | **RUNNABLE** — probed | The added clause `app.active_tab.kind == "graph"` holds under the split: `test_r2a.py` shows a 3px click with the `choose_output` half leaves `active_tab_index` 0 and the tab list length 1. `set_output_pass` fires no callback and `save_ui_document` touches no tab state. The break still bites (measured today: 5px is a click, 8px a drag). |
| 23 | S15 double-click opens the shader tab | RUNNABLE | `_double_click` keeps `pick_pass(..., focus_editor=True)` → `ensure_shader_tab`, which is what the round-1 probe measured doing exactly this. |
| 24 | S7 node_order | RUNNABLE | |
| 25 | G7 cursor | RUNNABLE | Both halves fixed (`cur_cursor`, canvas-drew anchor). |
| 26 | G14 kept bindings | RUNNABLE | Causes separated. |
| 27 | T1-T6 the tab | **TWO REASONS** on the last clause | **N3.** Eight sub-asserts are RUNNABLE (each verified against the code in round 1 and unchanged). The ninth — "no `editor_sessions` key at the graph path after the Uniforms tab has been focused" — passes without T1's edit, because `_locate_uniform_declaration` is gated on a click/hover of the name (`uniform.py:67`) and never runs with no mouse there: probed `_locate calls 0 get_current_session calls 0`. The two reasons a green result has: the edit landed, or nothing called the function. |
| 28 | T5 the tick predicate | **NOT RUNNABLE AS WRITTEN** | **N4.** `graph_active`'s model `script_active` is inline in `_draw_entry_points`; no `*_active` predicate exists to call. |
| 29 | Theme | RUNNABLE | The accent clause is the one that bites; break verified (`blue_b -> RED at: not in accent primaries`). |

## 3. Blast radius of the additions

Grepped `shaderbox/ tests/ scripts/ dogfood/` for each new name — **zero pre-existing hits**
for `choose_output`, `revalidated_wire`, `node_order`, `GRAPH_WIRE_HIT_FLOOR`, `bezier_point`
(and `GRAPH_WIRE_HIT_MIN`, so the rename leaves no stale reader). No collisions.

- **`App.choose_output`.** No name collision. Every surviving `pick_pass` caller is still
  correct under S15's "unchanged for the strip, the uniforms row and the copilot": `app.py:1154`
  (a newly added pass — "the editor tab, the viewer and the gear all follow it", so it wants
  the tab), `app.py:2027` (the step-through-passes hotkey, `focus_editor=self.editor_focused`),
  `widgets/pass_list.py:166` (a strip tile), `widgets/uniform.py:339` (the panel's source
  thumbnail, whose comment says "its click does what the strip's tile does"). Only
  `pass_graph.py:1209` (`_click`) moves to `choose_output`; `pass_graph.py:1227`
  (`_double_click`) stays. That is 1 of 6 call sites changed, matching S15 exactly. One
  behaviour note the spec already owns: `set_output_pass` returns `""` or an error string and
  S15 keeps "the toast", so the refusal path is preserved.
- **`revalidated_wire`.** New pure function beside `revalidated_scope`, same arity shape, its
  own row. No reader to update.
- **`node_order`.** New `GraphViewState` field; nothing reads the name today. Like
  `port_rects` and `canvas_rect` it is rebuilt per frame, so the round-1 F2 hazard (a
  no-frames test reading a default) applies — the S7 row correctly says "select `a`, frames"
  rather than calling `_fit` alone.
- **`GRAPH_WIRE_HIT_FLOOR`.** The rename answers my round-1 note verbatim ("named so it cannot
  be misread for the port floor `GRAPH_HIT_MIN = 7`; a comment names the pair and the 6-under-7
  reason"). No existing reader of either spelling outside `theme.py` and the port hit-box line.
- **The `xdist_group` additions.** Adding a module-level `pytestmark` to
  `tests/test_graph_view.py` pins **all ten** tests to one worker, not just the two that drive
  frames — including `test_the_widget_makes_no_session_write_of_its_own`, a pure source grep
  that needs no fixture at all. Acceptable, and measured: the whole module runs in **1.83s**
  (slowest test 0.30s, and seven of the ten already pay a ~0.06-0.19s `app`-fixture setup), so
  the serialisation costs a fraction of a second against a suite that runs `-n 8`. The
  alternative — marking only the two frame tests — would split the module across workers and
  is exactly what `pyproject.toml:91-96` warns against. Module-level is the right granularity
  and matches all four existing frame modules.
- **`widgets/uniform.py`.** One-line getter swap; `_locate_uniform_declaration` already
  handles `None` on both branches (`uniform.py:86` `if session is not None`, `:91`
  `active_path = session.source.path if session is not None else None`), so the change is
  behaviour-preserving for every existing case. Verified the risk it removes is real
  (`get_current_session` on a graph-path tab creates a session). Its test coverage is **N3**.
- **`widgets/pass_list.py`.** Docstring only (`:178`). No code reader.
- **`conventions.md`.** Two clauses: the 092 bullet (the canvas's home, the click choosing the
  output) and the editor-tab bullet's "4th editable `kind`" trigger. Both quoted lines exist
  where the spec says — `conventions.md:856-857` for the first, `:483-485` for the second. No
  other convention names `passes_view` or the strip/graph toggle.

## Verdict

N1 is the one that matters: the card ships at a width that truncates both of the maintainer's
real names, the row meant to pin the width is red at 128 for the port-label half, and the
error is the same hard-coded em ratio (`0.545898 × 12 = 6.5508`) that
`tests/test_pass_settings_layout.py`'s own docstring records as already having shipped a
truncating check once. The fix is a token (`GRAPH_NODE_W = 136`, C's original number) plus
re-reading the ellipsis row's expectation off the measurement, and the fit-clamp row survives
it unedited. N2 and N3 are gates that pass whether or not the thing they name is present —
cheap to restate, expensive to leave. N4 needs one sentence in T5. Everything else from round
1 is closed by quoted text, two of the closures (S8's sampled curve, S1's accent-primary
collision) go further than what I asked for, and the twenty-nine-row table is otherwise sound:
twenty-five RUNNABLE, and the two rows whose new mechanics I probed (S6's latch, G6's (d)
geometry) both behave as specified.

**VERDICT: PARTIAL** — land after setting `GRAPH_NODE_W` from the measured advances (N1),
restating S5's held-press falsifier and T1's session-creation falsifier so each can only pass
for its own reason (N2, N3), and naming the tick predicate as an extracted function (N4).
