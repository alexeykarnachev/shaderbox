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

---

# Round 2

Re-read `01_spec.md` at commit `3f75d78` in full from disk. Baseline re-run before reviewing:
`uv run pytest tests/test_graph_view.py tests/test_theme.py tests/test_uniforms_tab.py
tests/test_pass_settings_layout.py -q` → `26 passed in 2.37s`. Two new probes (`probe7`,
`probe8`) and two `app`-fixture experiments were run; both experiment files were deleted after
reading.

## Part 1 — closure of the eleven round-1 findings

| # | round-1 finding | closing text | verdict |
|---|---|---|---|
| 1 | `GRAPH_HOVER = blue_b` is an accent primary | **S1**: "For G-Q4 the record's `blue_b` is refused by the code: `blue_b` IS the `blue` accent preset's primary (`theme._ACCENTS["blue"]`) … the record's G6 sentence 'is not an accent primary' is wrong. … the hover takes the record's own fallback, a neutral: `COLOR.GRAPH_HOVER = _P["fg_0"]` … with the invariant `GRAPH_HOVER not in _accent_primaries`". **S12**: "the import-time invariants gain `GRAPH_HOVER not in _accent_primaries` (the one that bites -- it is red for `blue_b`)". **Theme row**: "Break to try: `blue_b` -- the accent clause goes red". | CLOSED |
| 2 | the ✕ does not win the overlap by submission order | **S6**: "Submission order cannot make it win: an item without `allow_overlap` submitted earlier beats a later overlapper (measured), the ports declare nothing so the drop target keeps working, and a short wire's midpoint lands inside its consumer port's 7px box. So the ✕ is not an item." | CLOSED — the false premise is gone and the ports keep their no-flag state, so `test_a_wire_dropped_on_a_drawn_port_writes_that_port` is untouched. The replacement mechanism carries two new defects of its own (findings 12 and 13 below). |
| 3 | the wire pass's frame position was unstated | **S3**: "The wire distance pass (G4) runs AFTER the node/port/output button loop and BEFORE the background press is acted on (S4), so this frame's hover is fully known at the point a selection is decided". **S4**: "acted on AFTER the node loop and the wire pass". Title now reads "one frame late for the DRAW, same-frame for the selection". | CLOSED |
| 4 | S5's falsifier could not go red | **S5**: "the group-name prompt is a plain `begin_popup`, for which `app.any_popup_open()` is False and `is_window_hovered(child_windows)` is already False, so a Delete typed into it is refused by `hovered` before `is_any_item_active` is consulted; the clause `not is_any_item_active()` is what refuses the key while a press is HELD on the canvas … and that is its falsifier." Two rows now exist: the held-press row carries "Break to try: drop `not is_any_item_active()` -- the held case writes", and the prompt row is relabelled "Behaviour pin; the clause it exercises is `hovered`". | CLOSED — and the relocation of the read (S5's "AFTER the hit-test section and the drag blocks") is a second correction I had not asked for, justified by its own measurement. |
| 5 | `tab_label` falls through to `pass_name_of(graph.json)` | **T1**: "Two edits are REQUIRED for the claim to hold, not implied: `tab_label` gains a `"graph"` branch returning `f"{document_name} (graph)"` before the multi-pass fallthrough (which would call `pass_name_of` on `graph.json`)". Files touched now reads "`tab_label`'s graph branch". | CLOSED |
| 6 | `get_current_session()` reachable on a graph tab | **T1**: "`widgets/uniform.py::_locate_uniform_declaration` switches from the creating `app.get_current_session()` to `get_current_session_if_exists()` (it already handles `None` on both branches) -- it is the one creating call site … it would otherwise open a GLSL editor session over `graph.json` and make the graph tab dirty-capable." The T1-T6 row gains "no `editor_sessions` key at the graph path after the Uniforms tab has been focused for those frames". | CLOSED — the fix, the reason and a gate. |
| 7 | S8's control-polygon hull over-frames | **S8**: "by the curve, not its control polygon … Framing the four control points would over-frame it (on the spec's own three-column chain the hull inflates the fitted width from 605 to 957px and drops the fit zoom to 0.84 where the curve needs none), so `_fit` … frames the union of the nodes' bounding box and, per wire, the 25 points `bezier_point(..., i / 24)`". | CLOSED — construction verified below; its test row does not yet bite (finding 14). |
| 8 | S4 lacked `is_item_hovered()`; same-frame move+release | **S4**: "`is_item_deactivated() and is_item_hovered() and is_mouse_released(left)` … `is_item_hovered()` is there because `is_item_deactivated()` is also True when the release lands off the item. A move and a release arriving in ONE frame read as a click (imgui resets the drag before the frame runs); today's press-time click does the same, and the frame-driven tests keep the move and the release in separate frames." The Verification preamble repeats the rule as a standing mechanic. | CLOSED |
| 9 | the smoke tail reads the current session | **T6**: "The tail's `get_current_session_if_exists() is not None` assertion is about the tab ACTIVE at the end, and it holds because frame 48's `set_current_document_id(canary_id)` runs `_on_current_document_changed` -> `ensure_shader_tab`, which focuses a shader tab; a later edit to frames 47 or 48 must keep that true." | CLOSED |
| 10 | G18 vs S9: two `_draw_wire` signatures | **S9**: "(G18's rule kept; G18's seven-parameter signature superseded, see the Refinements table)", and the table gains the row "G18: `_draw_wire(dl, xf, a, b, col, halo_col, halo_alpha)` (and G1's `..., hovered, selected`) → S9: `_draw_wire(dl, points, col, halo_col)`". | CLOSED |
| 11 | T5's caption-widget change was unflagged | **T5**: "draws `_entry_row_label(graph_active, "Passes")` -- a VISUAL change from today's `small_caption(app.font_12, "Passes")`: the label moves to the ambient font with frame-padding alignment". The maintainer's-eyes list gains "the Passes row's label in the ambient font with its tick", and a T5 row pins the predicate. | CLOSED |

Eleven of eleven CLOSED. No round-1 finding is still open.

## Part 2 — the new and rewritten decisions

### S15 — GAP: the Uniforms panel does not follow the clicked node when an explicit pin is set

The split itself is safe for every caller. `pick_pass` is `ensure_shader_tab` + the
`set_output_pass` half (`app.py:1883-1892`), and the five production callers that keep it are
`create_pass_from_draft` (`app.py:1154`), `step_output_pass` (`app.py:2027`),
`pass_list.py:166`, `uniform.py:339`, and the canvas's `_double_click` (`pass_graph.py:1227`).
Each wants the tab to come forward, so leaving them on `pick_pass` is right. The copilot calls
neither verb (`grep -rn 'pick_pass\|choose_output\|set_output_pass' shaderbox/copilot/` → no
hits), so the spec's phrase "unchanged for the strip, the uniforms row and the copilot" names one
surface that was never a caller and omits the two `app.py` internal ones — cosmetic, both keep
`pick_pass`.

The panel claim is where it breaks. S15 asserts: "The Uniforms panel follows: `panel_pass` falls
to the output when no shader tab of the document is active, so the clicked node's uniforms are the
ones shown." `panel_pass` (`app.py:743-761`) is a three-rung chain, and rung 1 is an explicit
pin that outranks the output:

```python
chosen = ui_document.ui_state.panel_pass
if chosen and chosen in document.passes:
    return document.passes[chosen]
```

Today a canvas click reached that pin, because `pick_pass` → `ensure_shader_tab` →
`self.set_panel_pass(document_id, "")` (`app.py:1568`), whose own comment says "Opening a pass in
the editor is itself a pick, so it retires an older explicit one (083)". `choose_output` alone
does not clear it. Measured on the `app` fixture:

```
after pick_pass('b'):      panel_pass ui_state = ''    panel_pass resolves to b
after choose_output('c'):  panel_pass ui_state = 'a'   panel_pass resolves to a
document.graph.output = c
VERDICT: panel_pass follows the clicked node? False
```

With no pin the claim holds exactly (`no pin, no shader tab: output = c  panel_pass -> c`), so the
gap is scoped to the explicit-pin case — but that pin is **persisted**
(`ui_models.py:177 panel_pass: str = ""`, and `tabs/uniforms.py:35-38`: "picking one here pins it
until a shader tab is opened, which retires the pick"). So a user who once used the Uniforms
tab's pass row gets a canvas whose clicks change the output while the panel stays put, across
restarts. That is a behaviour regression the split introduces, not a pre-existing one.

Fix: `choose_output` calls `set_panel_pass(document_id, "")` — which also keeps the 083 rule
("a pick retires an older explicit one") true of the canvas's pick, since choosing the output IS
a pick. One line, and it deserves a row: with a pin set, a canvas click leaves
`ui_state.panel_pass == ""`.

### S6 as rewritten — GAP: the mid-frame latch is invisible to this frame's `blocked`, and the press still reaches the background item

`press_blocked` has exactly four sites (`graph_state.py:91`, `pass_graph.py:846, 850, 856`), and
the reconcile block is:

```python
840  mouse_down = imgui.is_mouse_down(imgui.MouseButton_.left)
844  if not mouse_down:
846      view.press_blocked = False
847  elif frozen:
850      view.press_blocked = True
856  blocked = frozen or view.press_blocked
```

`blocked` is a **local**, computed once at `:856`, before the hit-test section. S6 writes
`view.press_blocked = True` after that, so `blocked` stays False for the rest of that frame at
all four gesture-start gates (`:946` band, `:973` node drag, `:1013` port press, `:1046` output
dot). Two consequences, one benign and one not.

**Benign.** All four gates also require `is_mouse_dragging(left, 4.0)`, which probe7 shows is
False on the press frame:

```
PRESS frame:            {'bg_clicked': True,  'bg_active': True,  'drag4': False, 'mouse_clicked': True}
held, frame 2:          {'bg_clicked': False, 'bg_active': True,  'drag4': False}
moved 40px while held:  {'bg_clicked': False, 'bg_active': True,  'drag4': True}
```

`drag4` first turns True on a later frame, by which time `blocked` has been recomputed from the
latched `press_blocked` and correctly suppresses everything. So "keeps this press from becoming …
a port grab, a band" holds, and the stale local is harmless for those.

**Not benign, two claims.**

1. "**The check runs FIRST in the hit-test section, before the background button, so the press is
   consumed before any item reads it.**" Ordering a hand test first does not stop imgui routing
   the press to an item — the ✕ is not an item, so nothing is consumed. probe7 shows
   `bg_clicked: True` on that same press frame, and the background's selection-clear at `:938`
   reads no `blocked`:
   ```python
   938  if imgui.is_item_clicked(imgui.MouseButton_.left) and not io.key_shift:
   939      view.selection.clear()
   ```
   So a ✕ click on a wire over empty canvas also clears the node selection. Minor, but the
   sentence as written is false and the `:938` line needs the `bg_pressed` treatment S4 gives the
   rest of that button (defer it, gate it on `not blocked` or on the ✕ not having fired).

2. "**a node click**" is listed among what the latch prevents. It is not, because S4 moved the
   node click to the **release** frame and `:844-846` clears the latch at the top of that frame,
   before `:856` recomputes `blocked`. And where a wire runs under a card — which G15 says is the
   normal case — the node, not the background, takes the press (probe8, a bg with
   `allow_overlap` and a node submitted after it, covering the point):
   ```
   PRESS over the node:  {'bg_active': False, 'nd_hov': True, 'nd_active': True,  'nd_deact': False, 'nd_clicked': True}
   RELEASE:              {'bg_active': False, 'nd_hov': True, 'nd_active': False, 'nd_deact': True,  'rel': True}
   ```
   So on the release frame S4's gate is `is_item_deactivated()` True, `is_item_hovered()` True,
   `is_mouse_released()` True, `node_drag`/`wire_drag` None, `not blocked` True → `_click` runs
   and `choose_output` fires. **One ✕ click on a wire crossing a card both unwires the sampler and
   changes the document's output pass.**

   Fix: the latch must survive the release frame, or the ✕ must set a one-shot the node click also
   reads. The smallest correct shape is to clear `press_blocked` on the release frame only *after*
   the gesture gates have run, or to have the ✕ set a separate `press_consumed` flag that
   `:844-846` does not touch and the node-click gate reads. Either way the verification row
   ("assert … that the press started no band and no drag (`band_anchor is None`, `node_drag is
   None`)") does not cover it: add `set_output_pass` was not called, and `document.graph.output`
   is unchanged.

The channel half of S6 is sound and was already probed in round 1 (a 5-way split held open across
item submissions and a `begin_popup` merges in index-buffer order `ff0000ff → ff00ff00 →
ffffffff` for channels 1/2/4).

### S1's `fg_0` — SOUND

Probed against every cue drawn on the canvas and against the theme block:

```
fg_0                       (0.9843, 0.9451, 0.7804, 1.0)
in accents?                False        == SELECT?  False
== STATE_ERROR?            False        == GRAPH_EDGE? False
in GROUP_TINTS?            False
FG_TITLE == fg_0           True
GRAPH_EDGE == FG_DIM       (0.5725, 0.5137, 0.4549, 1.0)
```

All four proposed invariants pass, `fg_0` is not in `_ACCENTS` under any accent, and it is a clear
brightness step above the `gray`/`FG_DIM` wire and the `FG_MUTED` dots at 0.35/0.55 alpha, which
is what G-Q4's fallback asked for.

The `FG_TITLE` identity is **a legitimate reuse of a neutral, not a collision the theme block
should refuse.** `FG_TITLE` has exactly one canvas use — `pass_graph.py:695`, the node's name
text inside the card — while `GRAPH_HOVER` is a border **inset** (G6) and a wire stroke, so the
two never share a pixel. The block's own rule permits precisely this: "a FIXED hue … may not equal
any accent preset's primary, nor another fixed hue **it shares spatial context with**"
(`theme.py:219-221`), and its cited precedent for allowing a share is the same shape — "State
colors are status TEXT, not nested outlines, so they may share an accent hue" (`:225-227`). An
invariant `GRAPH_HOVER != FG_TITLE` would be the domain-narrowing kind: it would forbid the one
safe neutral the palette has left, since S1 already establishes every chromatic hue is taken.

### S3/S4's pinned positions — SOUND, no read precedes its write

The sequence as now written:

```
 1  press_blocked reconciled; blocked computed (LOCAL, :840-856)
 2  port_rects = {}, canvas_rect written
 3  _build_view, _fit, wheel zoom
 4  channels_split(5); draw wires (reads last frame's hovered_*, selected_wire, x_rect written here), draw nodes (node_order)
 5  S6: the ✕ hand hit-test on is_mouse_clicked inside x_rect        -> writes press_blocked
 6  background button; bg_pressed = is_item_clicked
 7  node/port/output loop, sorted per S7                             -> node_hovered, drop_target, node_order; node CLICK on release
 8  S3: the wire distance pass                                       -> writes hovered_node/port/out/wire (this frame)
 9  S4: act on bg_pressed                                            -> reads this frame's hovered_wire, node_hovered
10  drag blocks (node_drag commit, wire_drag drop, band release)
11  guides
12  S5: the Delete read
13  channels_merge(); _canvas_menu; _group_prompt
```

Field by field: `x_rect` written at 4, read at 5 — after. `bg_pressed` written at 6, read at 9 —
after. `node_hovered` written at 7, read at 9 — after. `hovered_wire` written at 8 and read at 9
for the selection — after; its read at 4 is last frame's, which S3 now names as intentional for
the draw only. `node_order` is read at 4 and at 7, so the sort must happen once before 4, which
S7 states ("One list, sorted once per frame, used by both loops"). The only stale read in the
whole sequence is `blocked` against a `press_blocked` written at step 5 — S6's defect above, not
S3/S4's.

One edge checked and safe: on a frame where S2's `revalidated_wire` clears `selected_wire`, step 4
draws no ✕ and writes `x_rect = None` (the spec says "`None` otherwise"), so step 5's
inside-the-rect guard skips and there is no `unwire(*None)`.

### S8's sampled-curve fit — SOUND (construction); GAP (its verification row's break does not bite)

The construction is exact enough. 25 points at `i/24` against the true curve sampled at 2001
points, on the spec's own three-column chain with a backward read:

```
sampled@24  union=(-29.4, 0.0, 541.4, 132.0)  w=602.8  fitzoom(800x600)=1.0000
true(2000)  union=(-30.3, 0.0, 542.3, 132.0)  w=604.6  fitzoom(800x600)=1.0000
ctrl pts    union=(-206.6, 0.0, 718.6, 132.0) w=957.2  fitzoom(800x600)=0.8358
max under-coverage per side (px): 0.916  0.0  0.916  0.0   (against a 16px _FIT_MARGIN)
```

Under-coverage is 0.92px at worst, absorbed by the margin, and the sampled fit recovers the full
zoom the hull threw away. The spec's own figures (605 → 957px, 0.84) reproduce exactly.

**But the row's named break is green.** The row is "`_fit` with `avail = (800, 600)` … every wire's
25 sampled canvas-space curve points lie inside it. Break to try: fit the nodes alone -- the
backward wire's bulge lands outside." At 800×600 the zoom clamps to 1.0 and the pan centres the
content, so the visible window (-144 … 656 in canvas x) swallows the 21px bulge under **both**
implementations:

```
     avail |        correct (nodes+curve) |          broken (nodes only)
(800, 600) | zoom=1.000 outside=0         | zoom=1.000 outside=0
(600, 400) | zoom=0.995 outside=0         | zoom=1.000 outside=0
(560, 300) | zoom=0.929 outside=0         | zoom=1.000 outside=4
(520, 200) | zoom=0.863 outside=0         | zoom=0.956 outside=8
(460, 150) | zoom=0.763 outside=0         | zoom=0.846 outside=8
```

Fix: one number — the row needs an `avail` at or below `(560, 300)`. At `(520, 200)` the break
puts 8 of 75 points outside, which is unambiguous.

### T2's guard ordering — SOUND

`code.py:871-882` is `_draw_tab_row(app)` → `tab = app.active_tab` → the guard
`if tab is None or current_path is None or ui_document is None: return`, and `ui_document` is
keyed on `app.current_document_id` (`:873`), not on `tab.document_id`. So a graph tab of a
non-current document — or any tab while every document is deleted and the id is `""` — would hit
that guard and never draw. T2's revision ("right after the tab row and **before** the
`ui_document is None` guard (a graph tab of a non-current document must still draw)") is exactly
right, and the branch is safe on its own: `pass_graph.draw` guards its own id
(`pass_graph.py:791-793`, `ui_document = app.ui_documents.get(document_id); if ui_document is
None: return`), and testing "the active tab's kind is `graph`" implies the tab is not None.

T2's added sentence about scope is also correct: `CommandScope.EDITOR` specs dispatchable on the
tab are `CYCLE_CODE_TAB`, `CLOSE_CODE_TAB` and `FORMAT_BUFFER`, and `format_current_editor`
returns on `get_current_session_if_exists() is None` (`app.py:1767-1772`) before reaching
`formatter_for`.

## Round-2 finding summary

Three new findings, all demonstrated:

| # | decision | class | one line |
|---|---|---|---|
| 12 | S15 | GAP | `choose_output` does not clear the persisted `ui_state.panel_pass`, so with a pin set the panel does not follow the clicked node (measured: resolves to `a` while the output is `c`). One line to fix. |
| 13 | S6 | GAP | The mid-frame `press_blocked` write is invisible to this frame's local `blocked`, and it is cleared at the top of the release frame — so a ✕ over a card runs `_click` → `choose_output` on release (probe8), and the background's `:938` selection-clear fires on the ✕ press (probe7). "the press is consumed before any item reads it" is false. |
| 14 | S8 | GAP | The row's named break ("fit the nodes alone") is green at `avail = (800, 600)` because the 1.0 clamp centres the content; it bites at `(560, 300)` and below. One number. |

SOUND with nothing to add: **S1** (`fg_0` passes every invariant and the `FG_TITLE` identity is a
legitimate neutral reuse the theme block's own rule permits), **S3/S4**'s pinned sequence (no read
precedes its write; the only stale read belongs to finding 13), **S8**'s construction (0.92px
worst under-coverage against a 16px margin), **T2**'s guard ordering.

Nothing marked SPECULATION — every claim above is a quoted line, a probe, or an arithmetic
result reproduced here.

## Round-2 false trails — checked and fine

- **The `pick_pass` split breaking a caller.** All five keepers want the tab forward; the copilot
  calls neither verb. The split is behaviour-preserving for every one.
- **`choose_output` needing its own refusal path.** `pick_pass`'s tail already guards
  `ui_document is None or output == name` and toasts `set_output_pass`'s error; lifting that whole
  tail is the split.
- **S6's `unwire(*None)` on a stale `x_rect`.** The spec writes `x_rect = None` when no selected
  wire is drawn, and the inside-the-rect guard skips on `None`.
- **`is_mouse_dragging` leaking a gesture on the ✕'s press frame.** `drag4` is False on that
  frame (probe7), so all four gates are inert regardless of the stale `blocked`.
- **`fg_0` colliding with a group tint, `SELECT`, `STATE_ERROR`, `GRAPH_EDGE` or an accent.** None
  of them; probed.
- **S8's 25 samples missing the curve's extremum.** 0.92px worst case against a 16px margin.
- **T2 drawing a graph tab with `tab is None`.** The branch's own predicate excludes it, and
  `pass_graph.draw` guards its document id.
- **S11's reader list being incomplete.** Exactly the four external sites named:
  `popups/lib_picker/tree.py:24,350`, `tests/test_pass_settings_layout.py:20,91`,
  `tests/test_anchored_note.py:15,53`.
- **The xdist claim.** `tests/test_graph_view.py` carries no `pytestmark` today while
  `pyproject.toml:91-93` states the per-module rule — "green by luck of the worker split" is
  accurate.
- **The rig-frame shape S-Verification cites.** Real, at
  `tests/test_pass_settings_layout.py:86-90` (`new_frame` / `begin("rig")` / `push_font` /
  `calc_text_size`).

VERDICT: PARTIAL

---

# Round 3

Re-read `01_spec.md` at `ad428cc`. Baseline: `tests/test_graph_view.py tests/test_theme.py` → 18 passed.

## Closure

**12 (S15) — CLOSED.** "`choose_output` also clears the Uniforms tab's explicit pick
(`set_panel_pass(document_id, "")`), which `ensure_shader_tab` did for today's click and 083
states as the rule ('a pick retires an older explicit one'); without it a persisted pin would
keep the panel on another pass after a canvas click (measured)."

**13 (S6) — CLOSED, both halves.** "First, the check runs at the TOP of `_draw_canvas`, inside
the press bookkeeping and BEFORE `blocked = frozen or view.press_blocked` is computed … so the
same frame's `blocked` already carries the latch and the background button's `is_item_clicked` on
that frame is ignored by S4's `not blocked` gate. Second, the latch's clear (`if not mouse_down:
view.press_blocked = False`, today at the top of the frame) moves to the END of `_draw_canvas`,
so on the RELEASE frame `blocked` is still True and the release-time node click of S4 is refused."

Traced against `_draw_canvas` as it exists:

- *Same-frame background press.* The ✕ check lands above `:856`, so `blocked` is True for the
  whole frame. S4 already moved the selection clear off `:938` into a deferred block "acted on
  AFTER the node loop and the wire pass, gated on `not blocked`", so that block is refused. The
  `is_item_clicked` at `:938` itself is retired by S4, not merely ignored — refused.
- *Release-frame node click.* With the clear moved to the end, frame R computes
  `blocked = frozen or view.press_blocked` → True at `:856`, and S4's node-click gate carries
  `not blocked` → refused. The end-of-frame `if not mouse_down` then clears it; frame R+1 is
  clean. Refused.
- *The copilot-turn test's last stanza* ("the latch clears with the button: the next press is a
  gesture again", `tests/test_graph_view.py:249-256`) is unaffected. The stanza is
  `add_mouse_button_event(0, False)` → `_frames(app, 2)` → reposition → `_frames(app, 2)` →
  press. Both post-release frames run the end-clear (`mouse_down` False in each), so
  `press_blocked` is False well before the press on a later frame, and `view.wire_drag is not
  None` still holds. More generally the clear runs at the end of *every* frame where the button
  is up, i.e. every frame between a release and the next press — so no new press can ever see a
  stale latch.

**14 (S8) — CLOSED.** "`_fit` with `avail = (520, 200)` -- small enough that the 1.0 clamp does
not centre slack around the content (at 800x600 the break below stays green, measured) … Break
to try: fit the nodes alone -- 8 of 75 points land outside (measured)." Re-measured at the new
`GRAPH_NODE_W = 136`: correct → zoom 0.826, 0/75 outside; broken → zoom 0.915, **8/75** outside.
The stated count reproduces at the new width.

## S1's new width (136)

SOUND; nothing the record's arithmetic or the box width contradicts. Recomputed from the
measured 7.0px / 8.0px advances:

```
W=136  name budget 120  port-label budget 118
  distance_field   14b: 112.0 <= 120  fits
  u_distance_field 12 : 112.0 <= 118  fits
W=128  budgets 112 / 110 -> u_distance_field at 112 EXCEEDS 110 (ellipsizes); distance_field has zero slack
fitted width 6*136+5*64+2*16 = 1168   z@1225 = 1.049   z@740 = 0.634
```

Both figures match the record's own G11 table row for 136 exactly, so the fit arithmetic is not
contradicted — it is the row the record already published and rejected only on taste. The
record's stated bound ("150 is the point where the wide pane drops below 1.0, so it is out on its
own numbers") still holds and 136 sits inside it: 1225/1168 = 1.049 > 1.0. The `_fit`-clamp
verification row passes at 136 (`avail=(1225,600)` → zoom exactly 1.0; `(740,600)` → 0.6336,
inside `0.6 < z < 1.0`), and its own text already says "Green at 108 and 136 alike".

The box at 176 (`GRAPH_BOX_EXTRA_W` stays 40) contradicts nothing: it is named in Out of scope
("a box is 176 wide around a 96 picture. The maintainer's eyes; a token if he objects"), it keeps
the box wider than a pass card as the boundary-port labels and the "N passes" badge need, and it
is not read by the fit rows (both use pass nodes). The side margin grows 20 → 40px around the
thumb on a box, which is a look, not a correctness matter — already routed to his eyes.

No new finding: nothing here could be demonstrated as wrong.

VERDICT: PASS
