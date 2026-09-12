# 092 post-implementation review — code correctness, round 3

Closing out round 2 (`reviews/post_code_correctness_r2.md`, N1 and N2) against the fix commit
`c203784` ("092: fold the post-implementation round 2"). Every row was re-probed against the
real frame loop: a headless `App` driven through `shaderbox.ui.update_and_draw`, with
`io.add_mouse_*_event` supplying the gesture and `view.port_rects` / `view.canvas_rect` — the
per-frame records the fix added — supplying the aim. The two mutation tests ran in a detached
`git worktree` at `HEAD` in the scratchpad, restored byte-identically and removed afterwards;
no file in the repo was modified except this report.

## Gate result

| Gate | Command | Exit |
|---|---|---|
| `make gates` (check → test → smoke) | `make gates > log 2>&1; echo $?` | **0** — GREEN |

The smoke **ran**; it did not skip. The log's last line is
`== gates: GREEN -- check passed, test passed, smoke passed ==`, and the target's own
"stdout is not a terminal" warning is present, confirming the code came from the unpiped
redirect rather than from a pipe.

---

## Closure table

| # | Round 2 finding | Verdict | Observed now |
|---|---|---|---|
| N1 | The port allow-overlap killed every drop at every zoom | **CLOSED** | 12 of 12 cells green — 3 zooms × 4 slots, press, hover and drop all resolving to the same slot and the write landing. Table below |
| N2 | The copilot cancel re-armed the gesture in the same frame | **CLOSED for a press during the turn / OPEN as N3 for the frame the turn ENDS** | A press begun during a turn arms nothing on any of four surfaces and writes nothing on release. A gesture already dragging when the turn begins is correctly cancelled on every turn frame — and then rebuilt on the first frame after the turn, which is **N3** |

### N1, in full

Four-port consumer `b` (`u_p u_q u_r u_s`) fed from `a`. Each cell restores the positions and
the wiring, refits the view at the zoom, presses on the port's recorded rect and reads which
`gport` button imgui makes `is_item_active()`, then carries a wire from `a` over the same rect
and reads the `target` argument `_drop` receives:

```
zoom=0.5 slot=u_p: press=u_p hover=u_p wrote=PassSource(name='a') stray=- OK
zoom=0.5 slot=u_q: press=u_q hover=u_q wrote=PassSource(name='a') stray=- OK
zoom=0.5 slot=u_r: press=u_r hover=u_r wrote=PassSource(name='a') stray=- OK
zoom=0.5 slot=u_s: press=u_s hover=u_s wrote=PassSource(name='a') stray=- OK
zoom=1.0 slot=u_p: press=u_p hover=u_p wrote=PassSource(name='a') stray=- OK
zoom=1.0 slot=u_q: press=u_q hover=u_q wrote=PassSource(name='a') stray=- OK
zoom=1.0 slot=u_r: press=u_r hover=u_r wrote=PassSource(name='a') stray=- OK
zoom=1.0 slot=u_s: press=u_s hover=u_s wrote=PassSource(name='a') stray=- OK
zoom=2.0 slot=u_p: press=u_p hover=u_p wrote=PassSource(name='a') stray=- OK
zoom=2.0 slot=u_q: press=u_q hover=u_q wrote=PassSource(name='a') stray=- OK
zoom=2.0 slot=u_r: press=u_r hover=u_r wrote=PassSource(name='a') stray=- OK
zoom=2.0 slot=u_s: press=u_s hover=u_s wrote=PassSource(name='a') stray=- OK
failures: 0
```

`stray=-` is the check that no OTHER slot of the four took the write. Round 2's `u_p, u_p,
u_q, u_r` press collapse and its `AutoSource()` drop are both gone.

### N2's closed half, in full

A press made **during** a turn (`app.copilot.state.in_flight = True`, which `update_and_draw`
reconciles into `copilot_turn_active` each frame) arms nothing, on every press surface the
canvas has, and the release after the turn writes nothing. The control is a live drag with no
turn, which does arm and does commit:

```
wired-port:    armed_during_turn=(None, None, None)  post_release node=None wire=None  write=NONE
unfilled-port: armed_during_turn=(None, None, None)  post_release node=None wire=None  write=NONE
node-body:     armed_during_turn=(None, None, None)  post_release node=None wire=None  write=NONE
empty-canvas:  armed_during_turn=(None, None, None)  post_release node=None wire=None  write=NONE
control (no turn): armed=True wrote=True
violations: 0
```

The snapshot compared before and after each gesture covers the wiring, the positions and the
groups, so "write=NONE" is the whole document model, not one field.

---

## Mutation results — one gate holds, one does not

| Mutation | Expected | Observed |
|---|---|---|
| Re-add `set_next_item_allow_overlap()` before the port `invisible_button` in `_draw_canvas` | drop test red | **RED** — `test_a_wire_dropped_on_a_drawn_port_writes_that_port` fails, 9 others pass |
| Restore | green | **GREEN** — 10 passed, worktree diff vs `HEAD` empty |
| Remove `not frozen` from the port press branch | freeze test red | **GREEN — the gate does not fire** |
| Remove `not frozen` from **all nine** guard sites | freeze test red | **GREEN — the gate does not fire** |
| Neuter only `imgui.begin_disabled(app.copilot_turn_active)` in `pass_graph.draw` | — | GREEN |
| Neuter only `imgui.begin_disabled(app.copilot_turn_active)` around `_draw_app_panel` in `ui.py` | — | GREEN |
| Neuter **all three** layers at once | — | **RED** — `test_a_gesture_cannot_start_during_a_copilot_turn` fails |
| Restore all | green | **GREEN** — 10 passed, worktree diff vs `HEAD` empty |

The drop test is a real gate: it goes red on exactly the regression it names, and green again
on restore.

The freeze test is not. It passes on three redundant layers, and the one the commit message
credits it with pinning is the one it cannot detect. Traced to the cause: the graph canvas is
already inside `imgui.begin_disabled(app.copilot_turn_active)` twice — once in `pass_graph.draw`
around the tab row and the child, and once in `ui.py` around the whole `app_panel` child that
contains the document tab. A disabled item never becomes hovered or active, so during a turn no
canvas item reaches the press branches at all, with or without the `frozen` guards. Measured
under the both-`begin_disabled`-neutered, all-`frozen`-removed mutant, at the port's own rect:

```
hover during turn (no button): n_items=7 active=[] hovered=[]
after press during turn:       n_items=7 active=[] hovered=[]
after move during turn:        n_items=7 active=[] hovered=[]
```

and the identical sequence with the turn off, same mutant, same coordinates:

```
hover during turn (no button): n_items=7 active=[] hovered=[('##gport_p:b_0', False, True)]
after press during turn:       n_items=7 active=[('##gport_p:b_0', True, True)]  hovered=[...]
after move during turn:        n_items=7 active=[('##gport_p:b_0', True, False)] hovered=[]
```

So the test measures the disable wrapper, which predates this feature. The `frozen` guards are
not wrong — they are a correct second line — but nothing in the suite would notice their
removal, and the commit message states the test carries them.

---

## N3. REAL — the frame the turn ENDS rebuilds the cancelled gesture, and the release commits

**Symptom.** The cancel is right during the turn and wrong on the frame after it. While
`frozen` is true the top-of-frame block clears `view.node_drag` / `view.wire_drag` every frame
and the press branches are blocked, so nothing survives. On the first frame after the turn
`frozen` goes false while the mouse button is **still down**: the port or node button is still
`is_item_active()`, `is_mouse_dragging` is still true, and the press branch's
`view.wire_drag is None` / `view.node_drag is None` guards are satisfied by the cancel itself.
A fresh gesture is constructed from a press the user made before the turn, and the user's
release commits it.

**Demonstration, instrumented at the item.** Grab `b.u_src` (reading `a`), run three turn
frames, end the turn, and spy on the first post-turn frame:

```
armed wire:                     WireDrag(producer='a', start=(172.0, 177.0), grabbed=('b','u_src'))
during turn wire:               None
first post-turn frame, ACTIVE:  [('##gport_p:b_0', True, True)]
wire_drag now:                  WireDrag(producer='a', start=(172.0, 177.0), grabbed=('b','u_src'))
mouse down: True  dragging: True  drag_delta: (120.0, 100.0)
b u_src after release:          NoSource()
```

The wire the user was holding is destroyed: `PassSource('a')` becomes `NoSource()` on a release
the user intended for a drop that the turn interrupted.

**It needs no motion after the turn.** Four configurations, node drag and wire drag, with and
without a mouse move on the post-turn frame — all four re-arm, and each writes:

```
node move_after_turn=True:  rearmed_after_turn=(True, False)  pos None->(252.0, 115.0)
node move_after_turn=False: rearmed_after_turn=(True, False)  pos (252.0,115.0)->(252.0,115.0)
wire move_after_turn=True:  rearmed_after_turn=(False, True)  wire PassSource(name='a')->NoSource()
wire move_after_turn=False: rearmed_after_turn=(False, True)  wire PassSource(name='a')->NoSource()
```

The node case commits `(252.0, 115.0)`, a coordinate no gesture asked for: the motion before
the turn and the motion during it are both discarded, and what survives is whatever one frame's
`io.mouse_delta` the rebuilt drag accumulated.

**Reach.** Only a gesture already dragging when the turn begins. A gesture that starts inside a
turn never arms (the closed half of N2 above), so this is the narrow residue of round 2's N2,
moved one frame later by the `frozen` guards rather than eliminated.

**Fix.** The same latch round 2 proposed, with the window extended by one frame: cancellation
must hold until the button next comes up, not until `frozen` next goes false. A flag on
`GraphViewState` set when a turn cancels a gesture and cleared on
`imgui.is_mouse_released(left)`, ANDed into the three press branches alongside `not frozen`.

---

## The new-risk checks

The hover flag `HoveredFlags_.allow_when_blocked_by_active_item` was the round's specific
worry. Each half was probed and each is clean.

- **A port under a POPUP.** The flag does not see through one. The installed stub is explicit
  that it relaxes only the active-item block — a popup is `allow_when_blocked_by_popup`
  (`1 << 5`), an overlapping item is `allow_when_overlapped_by_item` (`1 << 8`), and neither is
  passed. Demonstrated rather than argued: with the canvas menu **forced open while the wire is
  already in flight** (`_canvas_menu` wrapped to re-`open_popup` every frame), the drop resolves
  to nothing and writes nothing — `popup_open_during_flight=True wire_alive=None target=None
  wrote NoSource()->NoSource()`. The wire is cancelled outright, which is the conservative
  outcome.
- **Is a menu open during a flight even reachable?** No, by construction, and this is worth
  recording because it makes the above a belt-and-braces result. Opening either menu takes a
  right-click; starting a wire takes a left-press; and the left-press dismisses any open popup
  before the wire exists. Probed both menus: the canvas menu (right-click on empty canvas) and
  the node menu (right-click on a node body) both open — `canvas menu open: True`, `node menu
  open: True` — and both are already closed by the time the wire is in flight
  (`popup_alive_mid_drag=False`), with the drop then landing correctly on the aimed port. The
  only way to hold a popup open across a flight is the forced instrumentation above, and even
  that writes nothing.
- **A port under another NODE's body.** Unchanged, and the flag is not what decides it. The
  submission order per frame is `##graph_bg`, then for each node `##gnode_<key>`, its
  `##gport_*`, its `##gout_*`. A port declares no allow-overlap, so a LATER node's body cannot
  take a hit the port already owns — swept 45 offsets of a second node against a port and found
  zero points where a plain hover at the port's centre resolved to anything but that port. In
  the other direction a node body DOES declare allow-overlap, so a later port wins, which is the
  designed precedence.
- **And the flag does not change the arbitration.** Two nodes placed 10 canvas-units apart so
  their port rects genuinely overlap (`b` at 2373-2386, `c` at 2381-2394 in x). Three sample
  points, each compared against what a plain hover with no flag reports as the owner:

  ```
  region    plain-hover owner   drop target            b                    c
  b-only    ##gport_p:b_0       ('b','u_src','none')   PassSource(name='a')  NoSource()
  overlap   ##gport_p:c_0       ('c','u_src','none')   NoSource()            PassSource(name='a')
  c-only    ##gport_p:c_0       ('c','u_src','none')   NoSource()            PassSource(name='a')
  ```

  The drop target equals the plain-hover owner in every region. The flag relaxes the
  active-item block and nothing else.

---

## False trails (probed, and fine)

- **My own first N1 probe, twice.** The first version reported all 12 cells failing and would
  have been filed as "the drop is still dead". Both failures were the harness. The press phase
  presses an unfilled port, which by design moves the NODE — so each iteration walked `b` 40 px
  down and right, and by the fourth slot `u_s`'s rect had slid below the canvas's bottom edge
  (`rect y 1408-1422` against `canvas bottom 1405`); a clipped port has no hover. The second
  version fixed that and then failed at zoom 2.0 for the same reason with a hand-picked pan.
  Recorded because the shape is generic: a frame-driven probe that mutates state between cells
  reports the widget's behaviour plus its own drift, and the fix is to restore the positions and
  refit per cell, then assert the aimed rect is inside `canvas_rect` before aiming at it.
- **`rebuilt_on_turn_frames=[]`.** The first N2 run showed the gesture surviving the turn and
  writing on release, which looks exactly like round 2's N2 unfixed. It is not: the per-frame
  instrumentation shows zero rebuilds on any turn frame and one on the frame after, which is a
  different defect with a different fix window. Worth separating, because "N2 is not fixed"
  would have been the wrong verdict — the guards do work, for exactly as long as `frozen` is
  true.
- **The `_ROOT_LABEL` suffix loop.** `_tab_row` now appends `_1`, `_2`, … until the label is not
  a group name. The loop terminates for any finite group set and the label is distinct by
  construction, which is what round 1's finding 6 asked for. No probe needed beyond reading it;
  recorded so the change is not mistaken for unreviewed.
- **The token substitutions.** `SIZE.GRAPH_LOOP_RISE`, `GRAPH_LOOP_REACH`, and the two
  `GRAPH_WIRE_W` / `GRAPH_PORT_RING_W` replacements of literal `1.5` / `1.2` are value-identical
  to what they replaced (12, 28, 1.5, 1.2). Nothing to probe; they cannot change behaviour.
- **`view.port_rects` as a test seam.** It is written every frame before the node loop and
  cleared at the top, so a stale entry cannot survive a frame in which the port was not drawn —
  confirmed by the scope switch in the overlap probe, where the dict held exactly the ports the
  frame submitted. A test aiming at it is aiming where the user would.

---

## Verdict

**FINDINGS.**

Both round-2 findings are closed on what they were raised for — N1 outright across every zoom
and slot, N2 for a press made during a turn. Two things remain:

- **N3 (real)** — a gesture already in flight when a turn starts is cancelled correctly for the
  turn's duration and then rebuilt on the first frame after it, because `frozen` goes false
  while the button is still down and the press branch's own guard is satisfied by the cancel.
  The release commits: a node lands on a coordinate no gesture asked for, and a grabbed wire
  becomes `NoSource()`. Round 2's proposed latch is still the fix; its release condition must be
  the button coming up, not the turn ending.
- **The freeze gate does not gate.** `test_a_gesture_cannot_start_during_a_copilot_turn` passes
  with every `frozen` guard deleted, and with either `begin_disabled` neutered; only removing
  all three layers turns it red. It is pinning the pre-existing disable wrapper, not the guards
  the commit message credits it with. Whatever shape the N3 fix takes, its test must be broken
  once against the latch alone — with the disable wrappers intact — before it is believed.

The drop gate, by contrast, is a real one: broken and restored in this round, red on exactly its
own regression.
