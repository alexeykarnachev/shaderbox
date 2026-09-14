# 093 wave 1 — pre-implementation review: correctness and design

Reviewed `01_spec.md` against `03_graph_design.md`, `00_findings.md`, `01_research_graph_tab.md`,
`CLAUDE.md`, `conventions.md`, `imgui-ui/SKILL.md` §3/§4/§8, and the code the spec changes read
end to end. Read-only on the repo. Every finding below is a quoted line or a probe output; six
probes were run headlessly against imgui-bundle 1.92.801 and are pasted where they decide a claim.

Baseline before review: `uv run pytest tests/test_graph_view.py tests/test_theme.py
tests/test_ui_regions.py -q` → `21 passed in 1.86s`.

---

## Findings

### CONTRADICTION 1 — S12/G-Q4: `GRAPH_HOVER = _P["blue_b"]` IS an accent primary

The record's G6 states: "`blue_b` is already excluded from `GROUP_TINTS` and **is not an accent
primary**". It is one. Probe:

```
$ uv run python -c "import shaderbox.theme as t; print([k for k,(p,_,_) in t._ACCENTS.items() if p==t._P['blue_b']])"
['blue']
blue_b in accents? True
```

`theme.py:474-475` in `apply_theme` writes `COLOR.ACCENT_PRIMARY = primary` at runtime, so with
the **blue accent selected** `COLOR.GRAPH_HOVER == COLOR.ACCENT_PRIMARY`. Both are drawn on this
canvas, on adjacent cues:

- `pass_graph.py:654` — `border = COLOR.ACCENT_PRIMARY` (the OUTPUT node's border)
- `pass_graph.py:1067` — `_u32(COLOR.ACCENT_PRIMARY)` on the in-flight wire

So under the blue accent a hovered wire reads as the in-flight wire, and a hovered node's halo
reads as the output node's border — the exact "two cues merge under some accent" failure the
import-time invariant block exists to prevent (`theme.py:225-227`: "a FIXED hue ... may not equal
any accent preset's primary").

S12's proposed invariants (`GRAPH_HOVER != SELECT`, `!= STATE_ERROR`, `!= GRAPH_EDGE`) all **pass**
and none of them catches this. A gate that passes on the one collision that exists is the "checker
that quietly narrows its own domain" class. The invariant `GRAPH_HOVER not in _accent_primaries`
is the one that bites, and it is red as written.

Fix is one token plus one assertion line. G-Q4's own fallback ("a neutral near `FG_SECONDARY`")
is available, or another `_P` hue no accent uses.

### CONTRADICTION 2 — S6: the ✕ does NOT win the overlap by submission order

S6: "clicked through one `invisible_button` ... submitted after every node, port and output
button" and the Refinements row "The button that wins the click must be the mark that is visible."
The ports **declare nothing**, so the ✕ submitted after a port loses the press to that port.
Probe (`probe3.py`, a port button first with no `allow_overlap`, then an overlapping ✕-sized
button):

```
mouse in the OVERLAP of port(first, no allow_overlap) and x(later):
  {'port_hov': True, 'port_click': False, 'x_hov': True, 'x_click': False}
pressed there:
  {'port_hov': True, 'port_click': True, 'x_hov': False, 'x_click': False}
```

The port wins. This is the repo's own documented rule — `pass_graph.py:11-14`: "each declaring
`set_next_item_allow_overlap()` so the later item wins (**imgui gives an overlapping hit to the
EARLIEST item unless it allows it**), then the ports, which are last and declare nothing" — and
`imgui-ui/SKILL.md` §3: "For the overlay's click to win over the cell beneath it: call
`imgui.set_next_item_allow_overlap()` *before* the cell's (invisible) button". §8's "then the
ports and overlay buttons come last and win" is true only because the earlier rungs declare it.

Reachable in practice: the ✕ sits at `t = 0.5` of a wire that terminates **at a port**, so a
short wire's midpoint lands inside the consumer port's hit box (`hit` is floored at
`GRAPH_HIT_MIN = 7` screen px, `pass_graph.py:996-999`, and `GRAPH_WIRE_X_R = 7` gives the ✕ the
same reach).

Fix: `set_next_item_allow_overlap()` on every port and output button (the last rung before the
✕), which the docstring says costs them "only the drop target" — but the drop target is
load-bearing and already pinned by `tests/test_graph_view.py::test_a_wire_dropped_on_a_drawn_port_writes_that_port`,
whose stated falsifier is exactly that flag. So the correct fix is the opposite direction: submit
the ✕ **before** the ports and give the ✕ `allow_overlap` — or keep the ✕ last and accept that a
wire whose midpoint lands on a port is unwired by Delete, not by the ✕. Either way S6's stated
reason does not hold and the design must choose deliberately.

### GAP 3 — S4/S3: the frame position of the wire-distance pass is unstated, and the two sections disagree

S3: the hit-test section "writes this frame's values **at its end**", resolved port → node →
"the nearest wire under G4's threshold".
S4: the background press is "acted on **AFTER the node loop**, when this frame's node and port
hover are known: **with a wire under G4's threshold** and no node or port hovered, `selected_wire`
becomes that wire".

"A wire under G4's threshold" is either this frame's distance pass (which S3 puts at the END of
the section, after the point S4 acts) or last frame's `view.hovered_wire`. The spec never says
which. Reading last frame's value makes a click select the wire the mouse was over on the
**previous** frame — at the 4px lock and 60fps that is a real mis-selection on a moving cursor.
The spec must state that the wire distance pass runs between the node loop and the `bg_pressed`
action, and that `bg_pressed` is resolved against that same-frame result. S3's "one frame late"
applies to the DRAW only, not to the selection decision.

### GAP 4 — S5's named falsifier cannot go red

Verification row: "G5: Delete is inert while a text field is active | select a wire, open the
group prompt (`view.group_prompt = True`), frames, send `Key.delete`, assert no
`set_sampler_source` call" — and G5 names "the `not is_any_item_active()` gate is the thing under
test". Probe (`probe4.py`, a child window with a `begin_popup` holding a focused `input_text`):

```
baseline, mouse inside child:            {'child_hov': True,  'any_active': False, 'del': False}
with the popup open + input focused:     {'child_hov': False, 'any_active': True,  'del': False}
delete pressed while popup up:           {'child_hov': False, 'any_active': True,  'del': True}
```

`is_window_hovered(child_windows)` is already **False** while the popup is up (the stub docstring
says so: "is current window hovered and hoverable (e.g. **not blocked by a popup/modal**)?",
`imgui/__init__.pyi:844-845`). So the `hovered` clause alone refuses the key; deleting
`is_any_item_active()` leaves the test green. Per the spec's own "Gates that must be broken before
they are believed", this row is a wish.

The break that DOES exercise `is_any_item_active()` is an active item **inside** the canvas child
(no popup) — there is none today, so either the clause is unreachable-and-defensible (say so and
drop it from the gate list) or the test needs a different break, e.g. drop the `hovered` clause
and assert the key still refuses.

### GAP 5 — T1/T2: `tab_label` falls through to `pass_name_of(graph.json)`

`tabs/code.py:57-78`: the chain is `lib` → `script` → else `suffix = pass_name_of(tab.path) if
multi_pass else "shader"`. A graph tab on a multi-pass document takes the else and calls
`paths.pass_name_of(Path(".../graph.json"))`. The spec's Files-touched lists "`tab_label` for the
graph kind", so the branch is intended — but T1's prose ("every path-keyed pass-through works
**unchanged**") reads as if no edit were needed, and the Verification row only asserts the output.
State the branch in T2 as a required edit, not an implied one.

### GAP 6 — T1: `App.get_current_session()` (the CREATING variant) is reachable on a graph tab

T1 enumerates the non-creating readers (`is_tab_dirty`, `is_current_editor_dirty`,
`format_current_editor`, `jump_to_next_error`, `_drain_editor_input`, the flush paths) and is
correct on every one — all six go through `get_current_session_if_exists()`
(`app.py:869, 1278, 1732, 1767, 1790`, `hotkeys.py:53`). It misses the one creating call site:

```
$ grep -rn 'get_current_session()' shaderbox/
shaderbox/widgets/uniform.py:85:    session = app.get_current_session()
```

`uniform.py:82-98::_locate_uniform_declaration` runs on a click or hover of a uniform's name in
the Uniforms panel — a different panel from the editor pane, live while a graph tab is active.
`get_current_session()` → `get_session_for_path(graph.json)` → `ShaderSource.load` →
`Editor(json_text)` with `language_for_path` falling back to GLSL ("unknown falls back to GLSL —
host policy", `editor/ffi.py:452-456`). `graph.json` exists on disk for every document
(`find projects/dev -name graph.json` returns two), so this does not raise — it silently creates
a GLSL editor session over a JSON file, which then appears as a dirty-capable session keyed at
`graph.json`, and `is_tab_dirty` on the graph tab stops answering False. T1's "A graph tab has no
`EditorSession`" becomes false after one uniform hover.

Fix: `_locate_uniform_declaration` should use `get_current_session_if_exists()` (it already
handles `session is None` on both branches), or `open_graph_for` must be excluded there by kind.

### GAP 7 — S8: framing the control polygon over-frames the fit by ~58% when a backward wire exists

S8's correction of G3 is **right** — the geometry confirms an S-curve leaves both cards. Worked
at `GRAPH_NODE_W = 128`, `GRAPH_WIRE_BOW = 0.40`, `GRAPH_WIRE_MIN_OFF = 24`, producer output
`(128, 60)` → consumer port `(-192, 200)`:

```
control pts ((128.0, 60.0), (267.71, 60.0), (-331.71, 200.0), (-192.0, 200.0))
curve max x 149.27   producer right edge 128.0   bulge right of producer: 21.27
curve min x -213.27  consumer left edge  -192.0  bulge left  of consumer: 21.27
FIT_MARGIN=16; right bulge > 16? True
```

So G3's "no wire leaves the nodes' bounding box" is false and `_FIT_MARGIN = 16`
(`pass_graph.py:70`) does not cover the 21.3px bulge. **S8's diagnosis is SOUND.**

But the chosen construction is far looser than the defect requires. On the spec's own
verification chain (`a -> b -> c` plus a backward read, three columns at W=128/GAP=64):

```
nodes bbox         (0.0, 0.0, 512.0, 132.0)
true curve bbox    (-30.3, 56.0, 542.3, 124.0)
ctrl-pt bbox       (-206.6, 56.0, 718.6, 124.0)
nodes only         w= 544.0  fit zoom(800x600)=1.000
nodes+true curve   w= 604.6  fit zoom(800x600)=1.000
nodes+ctrl pts     w= 957.2  fit zoom(800x600)=0.836
```

The hull inflates the fitted width 604.6 → 957.2 and drops the fit zoom to 0.836 where the true
curve needs no zoom-out at all. G11's six-column numbers survive (a forward chain's offsets stay
inside `GAP_X`, so all three constructions give 1.000 at 1225px and 0.661 at 740px — the two
tests do not conflict), so this is a quality finding, not a correctness one. S9 already ships
`bezier_point`; sampling it at a handful of `t` values, or taking the cubic's x/y extrema, closes
finding 5 without costing 12% of the fit zoom.

S8's zoom-division claim is verified exactly:

```
z=0.25: screen/z == wire_points(canvas,1.0) ? True
z=1.0 : True
z=2.5 : True
```

### GAP 8 — S4 has no `is_item_hovered()` clause, and a same-frame move+release reads as a click

Probe (`probe.py` / `probe2.py`, an `invisible_button` with `allow_overlap`):

```
A: press, move 3px, release
  press              {'node_active': True,  'node_clicked': True, 'node_deact': False, 'node_dragging4': False}
  moved 3px          {'node_active': True,  'node_clicked': False,'node_deact': False, 'node_dragging4': False}
  release after 3px  {'node_active': False, 'node_clicked': False,'node_deact': True,  'node_dragging4': False, 'mouse_released': True}

B: press, move 20px, release
  moved 20px         {'node_active': True,  'node_deact': False, 'node_dragging4': True}
  release after 20px {'node_active': False, 'node_deact': True,  'node_dragging4': False, 'mouse_released': True}
```

S4's claim "**on the release frame a drag that happened is still set**" is **SOUND**, but for a
reason S4 does not state: `is_mouse_dragging` is itself **False** on the release frame (both
cases), so the gate rests entirely on `view.node_drag` surviving. It does —
`app.py:1960-1966::commit_node_drag` sets `view.node_drag = None`, and that runs in the drag block
at `pass_graph.py:1049-1053`, **after** the node loop at `:948-1047`. Verified structurally.

Two residuals:

1. `is_item_deactivated()` is True on the release frame **even when the release lands far off the
   item**:
   ```
   G: press on node, drag to (500,500), release outside
     dragged off   {'hov': False, 'act': True,  'deact': False}
     release OUTSIDE {'hov': False, 'act': False, 'deact': True, 'rel': True}
   ```
   S4's gate carries no `is_item_hovered()`. Add it — it costs nothing and closes the case.

2. When the move and the release arrive in the **same** frame, imgui's release processing has
   already reset the drag:
   ```
   D: move 30px AND release queued into ONE frame
     move+release SAME frame {'act': False, 'deact': True, 'drag4': False, 'rel': True, 'dd': (0.0, 0.0)}
   ```
   `get_mouse_drag_delta(left, 4.0)` is `(0.0, 0.0)` and `is_mouse_dragging` False, so a fast
   flick reads as a click and `pick_pass` fires. Today's `is_item_clicked`-on-press does the same,
   so S4 is not a regression — but G13's "5px is a drag" gate only holds when the move and the
   release land in separate frames, which the existing `_frames` shape does satisfy. Note it in
   the test so a future rewrite does not collapse the frames.

### GAP 9 — T6: the smoke's tail reads the CURRENT session, not "some session drew"

`scripts/smoke.py:426-427`: `editor_session = app.get_current_session_if_exists()` then
`assert editor_session is not None, "smoke: no editor session after the loop"`. T6's reassurance
("the editor-drew assertion at the tail still holds because the shader tab draws every other
frame of the run") is true but for the wrong reason: the assert is about the **active tab at the
end**, not about accumulated draws. It survives because frame 47 closes the graph tab and frame
48's `app.set_current_document_id(canary_id)` runs `_on_current_document_changed` →
`ensure_shader_tab` (`app.py:699-710`), which focuses a shader tab. Say that, so a later edit to
frame 47 or 48 does not quietly break the tail.

Also: the spec says "Frame 43 calls `app.open_graph_for(multi)` where it set `passes_view = GRAPH`"
— frame 43 is correct (`smoke.py:286-289`).

### CONVENTION 10 — G18 vs S9: two different `_draw_wire` signatures, unreconciled

G18 (adopted by reference as a locked constraint): `_draw_wire(dl, xf, a, b, col, halo_col,
halo_alpha)`. S9: `_draw_wire(dl, points, col, halo_col)` — four parameters, no `xf`, no
`halo_alpha`. G1's Code paragraph gives a third: `_draw_wire(dl, xf, a, b, col, hovered, selected)`.
The Refinements table has no row for this. S9's shape is the better one (it takes the points S9's
own `wire_points` returns, and G18's rule that the caller resolves the state is preserved), but
the spec must say it supersedes G18's signature, or a reviewer checking the implementation
against "the record's Rule paragraphs" reports a deviation.

### CONVENTION 11 — T5 changes the Passes caption's widget without saying so

`tabs/document.py:449-451` draws `small_caption(app.font_12, "Passes")`. T5 replaces it with
`_entry_row_label(graph_active, "Passes")`, which is a different widget:
`align_text_to_frame_padding()` + `text_colored(COLOR.FG_DIM, label)` + `same_line(SPACE.MD)`
(`document.py:383-402`), in the **ambient** font rather than `font_12`. That is the right call for
consistency with the Script row's tick, but it is a visual change the spec presents as a
mechanical one. Name it so the maintainer's eyes pass knows to look.

---

## Per-decision verdicts

| id | verdict | evidence |
|---|---|---|
| S1 | SOUND | `GRAPH_NODE_W`/`GRAPH_DRAG_LOCK_PX` are one token each; G-Q2 ships as an absent item. `GRAPH_HOVER`'s value is CONTRADICTION 1. |
| T1 | GAP | `_on_pass_renamed` matches `tab.path == old_path` exactly (`app.py:733-735`) — `graph.json` never matches: verified. `_on_document_deleted` filters `t.document_id != document_id or t.kind == "lib"` (`app.py:781-785`): drops it. `forget_render_state` pops `graph_views` (`app.py:804`). `close_editor_for_path` pops with a `None` default (`app.py:1612-1620`). `is_tab_dirty` reads `editor_sessions.get(tab.path)` → None → False (`app.py:1737-1742`). `formatter_for("graph")` returns None (no table row). `flush_current_editor`, `format_current_editor`, `jump_to_next_error`, `_drain_editor_input` all use `get_current_session_if_exists`. The flush-all loop skips non-dirty tabs (`app.py:2227-2231`). **Gaps 5 and 6.** |
| T2 | GAP | `draw` returns early on `ui_document is None` BEFORE the session fetch (`code.py:877-883`), so the branch must sit after `_draw_tab_row` and before that guard, or a graph tab of a non-current document dies. The spec says "right after the tab row" — compatible, but the interaction is unstated. `editor_focus_requested` is consumed at `code.py:947-952` gated on `not any_popup_open()`, matching T2. `draw_chrome`'s non-shader path prints `tab_label` and reads `is_current_editor_dirty` (`code.py:795-799`) — False for a graph tab: verified. Gap 5. |
| T3 | SOUND | `SIZE.GRAPH_MIN_H` is read only at `document.py:468`; deleting both is consistent. `begin_child(size=(0,0))` takes the host's region. |
| T4 | SOUND | `Alt+G` unbound (`[s.id for s in COMMAND_SPECS if s.default_chord == int(K.g)\|int(K.mod_alt)]` → `[]`), `chord_needs_modifier` accepts it. `OPEN_SHADER`/`OPEN_SCRIPT` are `CommandScope.GLOBAL`, category `Editor` — matches "the same default scope". `tests/test_command_registry_coverage.py::test_every_bound_spec_reaches_the_help_shortcuts` reads the generated section: satisfied. |
| T5 | CONVENTION | `drop_unknown` in `model_salvage.py:115` retires the key. `projects/dev/app_state.json` carries no `passes_view` (probed: `'passes_view' in d` → False). `tests/test_ui_regions.py` tests only the retired enum (its docstring claims `ChannelView` too, but no test does — `ChannelView` is covered by `tests/test_channel_view.py`), so deleting it loses nothing. Tooltip "Open the pass graph" = 4 words, under the 5-word budget. **Finding 11.** |
| T6 | GAP | Frame 43 is the `passes_view = GRAPH` site: verified. `_check_invariants` (`smoke.py:139-156`) is tab-agnostic. **Gap 9.** |
| S2 | SOUND | `(owner, sampler)` is unique per drawn edge in any one scope: `node_ports` builds one port per declared sampler from a dict-keyed `wiring_row`, so one sampler has one source; the three edge-construction sites (members `:295-306`, readers `:307-321`, root/box `:391-419`) never share an owner within a scope (a name is member XOR reader; a feeder ghost gets no incoming edge). `App.unwire` → `set_sampler_source` is document-flat (`project_session.py:1071-1081`), so a ghost or box owner works. `_Edge` gaining `owner`/`sampler` is derivable at every site: `ports[name][slot].sampler` for sites 1-2, `node.owners[slot]` for a box. |
| S3 | GAP | The one-frame-late DRAW is correct and necessary — the picture is drawn at `:882-916`, the hit rects submitted at `:918-1047`. "Every field written every frame, `None` included" is the right shape. **Gap 3** is the selection read, not the draw. |
| S4 | GAP | **Gaps 3 and 8.** The core claim (a drag is still set on the release frame) is verified structurally and by probe. |
| S5 | SOUND / GAP | `imgui.Key.delete` = 522, `imgui.Key.backspace` = 523 on this build: verified. Only Escape is filtered at the glfw layer (`app.py:605-615`); Delete always reaches imgui. `hovered` = the existing `is_window_hovered(child_windows)` at `:867`. **Gap 4** is the gate's falsifier, not the gate. |
| S6 | CONTRADICTION | **Contradiction 2** for the click. The channel mechanics are verified: a 5-way split held open across `invisible_button` submissions and a `begin_popup` merges in index-buffer order `ff0000ff → ff00ff00 → ffffffff` for channels 1/2/4 (`probe6.py`), so paint order follows channel index and the popup does not disturb the canvas splitter. Note the split must now be merged AFTER the guides — today `channels_merge()` is at `:917`, before the hit test, and the band/guides/in-flight wire are drawn post-merge and win by call order (`:1049-1124`). Moving the merge is required, and the spec says so. |
| S7 | SOUND | The sorted list used by both loops is correct, and an earlier node losing its hover to a later overlapping one is fine because `node_hovered` is an OR over the loop (`:948`, `:962-963`). Probe F: an item declaring `allow_overlap` yields its hover to a later overlapper (`{'a_hov': False, 'b_hov': True}` in the overlap, `{'a_hov': True}` outside it), which is exactly the mechanism S7 relies on. |
| S8 | GAP | **Gap 7.** The diagnosis is right and the record's G3 is wrong; the construction is loose. The `_fit` clamp note is correct: `zoom = min(1.0, avail.x / w, avail.y / h)` at `pass_graph.py:497`. |
| S9 | CONVENTION | The five functions and `WireState` are pure and imgui-free — `graph_state.py` imports only `document`, `pass_graph`, `theme`, so the module stays importable without imgui (`theme.py` does import imgui, but it already does today and `graph_state` already reads `SIZE`). `wire_hit_threshold(zoom)` matching G4's table: `max(6.0, 1.5*2*0.25)=6.0`, `max(6.0, 3.0)=6.0`, `max(6.0, 7.5)=7.5` — correct. **Finding 10.** |
| S10 | SOUND | `path_stroke(self, col, thickness=1.0, flags=0)` (`imgui/__init__.pyi:10558`) and `path_arc_to(center, radius, a_min, a_max, num_segments=0)` (`:10562`): the Refinements row is verified. `_draw_badge` returning its width is a pure additive change; it already computes `w` at `:613`. |
| S11 | SOUND | `_ellipsize` has exactly four in-module call sites (`ui_primitives.py:391, 577, 1319, 1741`) plus the definition at `:26`; the rename is mechanical. |
| S12 | CONTRADICTION | **Contradiction 1.** The token deletions are consistent: `GRAPH_BUS_STEP`/`GRAPH_BUS_CLEAR` read only in `_draw_canvas`'s bus expression (`:901-906`), `GRAPH_LOOP_RISE`/`GRAPH_LOOP_REACH` only in `_draw_self_loop` (`:565-571`), `GRAPH_MIN_H` only at `document.py:468`. `_MIN_DIRECT_DX = 24.0` and `_BEZIER_BOW = 0.45` at `:71-75` become the tokens. Unreviewed: `GRAPH_BOX_EXTRA_W` stays 40 while the base grows 108 → 128, so a box is 168 wide holding a 96 thumb — worth a look but not a defect. |
| S13 | SOUND | Exactly four `is_mouse_dragging` sites (`:942, 968, 1009, 1042`) and one `get_mouse_drag_delta` (`:946`), matching the claim. `is_mouse_dragging(button, lock_threshold=-1.0)` and `get_mouse_drag_delta(button=0, lock_threshold=-1.0)` confirmed (`:3120`, `:3125`), and the docstring states the default comes from `io.MouseDraggingThreshold`. |
| S14 | SOUND | Three cursors exist at `app.py:273-275`; `want_cursor`/`cur_cursor` at `:279-280`. `ui.py:658-661` applies once on change and resets `want_cursor = None` — so the test must read `app.cur_cursor` after the frame, which the spec already allows. |

### Refinements table

| row | verdict | evidence |
|---|---|---|
| G4 → S2 identity | SOUND | Unique, comparable, and the identity `unwire` takes. See S2 above. |
| G5 → S4 select on press | GAP | The reason given is right (the node path selects on press today, `:961-965`), but the rule needs `is_item_hovered()` and a stated wire-pass position. Gaps 3, 8. |
| G5/G12 → S6 ✕ on the top channel | CONTRADICTION | The channel move is right and verified; the click-order reason is false. Contradiction 2. |
| G3 → S8 fit frames the wires | SOUND (diagnosis) / GAP (construction) | The bulge is 21.3px against a 16px margin: measured. The hull over-frames 58%. Gap 7. |
| G11 fit table → S8's clamp note | SOUND | `min(1.0, ...)` at `:497`. Recomputed: 1120px fitted width → 1.000 at 1225px, 0.661 at 740px, matching G11's table. |
| G4/G18 → S9 in `graph_state.py` | SOUND | Pure, imgui-free, no `app` fixture needed. |
| G8 → S10 `path_stroke(col, thickness)` | SOUND | Signature verified in the stub. |

---

## Coverage — what was checked

Frame order walked end to end at `pass_graph.py:827-1124`: mouse-state reconcile (`:840-857`) →
`port_rects`/`canvas_rect` reset (`:858-859`) → `_build_view` (`:861`) → `_fit` (`:862-863`) →
wheel zoom (`:867-880`) → `channels_split(2)` + wires + self-loops + nodes + `channels_merge`
(`:882-917`) → bg button, pan, selection-clear, band start (`:918-947`) → node/port/output loop
(`:948-1047`) → node drag update + commit (`:1049-1053`) → wire drag draw + drop (`:1054-1071`) →
band draw + release (`:1072-1101`) → guides (`:1102-1118`) → canvas menu + group prompt
(`:1120-1124`). Writes-vs-reads for every state the spec touches are in the per-decision table.

Probes run (all on imgui-bundle 1.92.801, `MESA_GL_VERSION_OVERRIDE=4.6`): `is_item_clicked` on
press; `is_item_deactivated` on release and on release-off-the-item; `is_mouse_dragging` on the
release frame and on a same-frame move+release; allow-overlap hover forfeit; a no-allow-overlap
first item beating a later overlapper; `is_window_hovered(child_windows)` and
`is_any_item_active()` under an open non-modal popup; a 5-way `channels_split` held open across
item submissions and a popup, read through the index buffer.

Tab pass-throughs each cited to a line in the per-decision T1 row. Enumerated the `tab.kind` /
`current_editor_path` / `editor_tabs` surface with
`grep -rn 'tab\.kind|active_tab\b|current_editor_path|editor_tabs' shaderbox/ scripts/ tests/`;
the only misbehaving path found is Gap 6 (`uniform.py:85`). `popups/help.py:100` and
`tabs/document.py:413-417` both gate on `tab.kind == "shader"` / `== "script"`, so a graph tab
falls through harmlessly. `panel_pass` (`app.py:756-757`) gates on `tab.kind == "shader"`: safe.

Conventions: all new symbols are module-level functions (no `@staticmethod`), imports stay at
top, tokens live only in `theme.py`, labels are within budget, and the no-session-write pin
(`tests/test_graph_view.py:162-173`, a source grep for `set_sampler_source` /
`set_pass_positions` / `set_pass_groups` / `set_pass_group(`) is unaffected because the ✕ and
Delete both route through `App.unwire`. The 092 conventions bullet's "Revisit the canvas's home
when the editor pane can host it" clause is the one T1/T5 resolves, and the spec lists that doc
edit.

## What was skipped, and why

- **The record's G1-G18 as rules.** Adopted by reference; I checked only the ones the spec refines
  or that a spec decision depends on (G1's algebra, G3, G4's threshold table, G5, G6's order,
  G8's primitives, G11's fit arithmetic, G12, G13, G18).
- **G11's font measurement (0.545898 × em).** Taken from the record; verifying it needs a pushed
  font in a live frame and the spec already routes that budget through a frame-driven test.
- **The six research reports under `research/`.** The record is the contract the spec is written
  from; re-deriving the reports' primaries is out of this review's scope.
- **The rendered look.** No window manager here; every visual call in the spec's "maintainer's
  eyes" list stays his.

## False trails — checked and fine, do not re-check

- **`_on_pass_renamed` matching a graph tab.** It compares `tab.path == old_path` against a pass
  file path; `graph.json` cannot collide. T1 is right.
- **`_on_document_deleted` keeping a graph tab.** The filter is `document_id != document_id or
  kind == "lib"`, so a graph tab with that id is dropped. T1 is right.
- **`formatter_for("graph")`.** No table row, returns None, and `format_current_editor` returns on
  `get_current_session_if_exists() is None` before ever reaching it. Doubly safe.
- **The flush-all loop KeyError-ing on a graph tab.** `app.py:2227-2231` skips non-dirty tabs, and
  a graph tab is never dirty.
- **`drop_unknown` choking on the retired `passes_view` key.** `model_salvage.load_model` drops
  unknown keys before construction; the sandbox carries no such key anyway.
- **`imgui.Key.delete` / `backspace` presence.** Both exist (522 / 523).
- **`path_stroke`'s flags position.** `(col, thickness=1.0, flags=0)` — S10's refinement is right
  and G8's `ImDrawFlags_.none` positional is the wrong one.
- **Whether a popup breaks the open channel split.** It does not (probe6); a popup is its own
  window with its own draw list, as S6 says.
- **Whether Delete is swallowed before imgui.** Only Escape is (`app.py:605`).
- **`_fit`'s 1.0 clamp.** Present at `:497`; S8's note is accurate.
- **G11's six-column fit numbers under S8's new fit.** Recomputed all three constructions: 1.000
  at 1225px and 0.661 at 740px in every case, because a forward chain's control offsets stay
  inside `GAP_X = 64`. The two Verification rows do not conflict.
- **Deleting `tests/test_ui_regions.py` losing `ChannelView` coverage.** It does not;
  `tests/test_channel_view.py` covers it.
- **`Alt+G` colliding with a vim or standard chord, or being refused by the registry.** Unbound,
  and `chord_needs_modifier` accepts it.
- **S2's `(owner, sampler)` colliding across the box / ghost / member edge shapes.** It cannot
  within one scope; reasoned over all three construction sites.

---

VERDICT: PARTIAL
