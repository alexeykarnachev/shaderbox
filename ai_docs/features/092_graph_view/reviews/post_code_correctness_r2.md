# 092 post-implementation review — code correctness, round 2

Closing out round 1 (`reviews/post_code_correctness.md`, findings 1-12) against the fix commit
`ed1c28b` ("092: fold the post-implementation round 1"). Every row below was re-probed with the
headless `App` driven through `shaderbox.ui.update_and_draw`, the same driver round 1 used (a spy
on `_draw_canvas` captures the child's screen origin, `io.add_mouse_*_event` supplies the
gesture), or with a pure-function probe over the GL-free modules.

Two baselines were used, both as detached `git worktree` checkouts in the scratchpad and both
removed afterwards: `57b137e` (pre-fix W2) to confirm a probe shape still reproduces the old
behaviour there, and a pristine `ed1c28b` to confirm a new finding is in the commit rather than in
another reviewer's concurrent working-tree edits. No repo file outside this report was modified;
the one mutation test below restored its file byte-identically and is verified as such.

## Gate results

| Gate | Command | Exit |
|---|---|---|
| `make gates` (check → test → smoke) | `make gates > log 2>&1; echo $?` | **0** — GREEN |

The smoke **ran**; it did not skip. The log's last line is
`== gates: GREEN -- check passed, test passed, smoke passed ==`, and the target's own
"stdout is not a terminal" warning is present, confirming the exit code was read from the
unpiped redirect rather than from a pipe.

The green gate is the headline caveat of this round. Finding N1 below breaks wire-dropping
completely, and `make gates` is green anyway: the suite exercises `App.drop_wire`,
`pass_graph.refuse_drop` and `pass_graph._drop` as pure verbs, and nothing drives the imgui hover
that feeds `_drop`'s `target` argument. The gate is blind to the class.

---

## Round 1, item by item

| # | Round 1 | Verdict | Observed now |
|---|---|---|---|
| 1 | `_snap` latches the drag; node pinned to the guide, commit is the guide's coordinate | **CLOSED** | Node aligned on a guide, cursor moved 117 canvas px: `b` tracks the cursor and commits `(114.0, 200.0)`. Band behaves: `raw.x ≤ 6.0` (the threshold) holds `shown.x = 0.0` on the guide, `raw.x = 9.0` releases it and `shown.x` equals `raw.x` from there on |
| 2 | A gesture orphaned by the canvas not drawing survives; a later stray click commits | **CLOSED** | Node drag: after the STRIP round trip `node_drag` is `None`, a no-button move leaves it `None`, a stray click leaves `b` at `(0.0, 300.0)`. Wire drag: `wire_drag` `None` after the round trip, `b.u_src` stays `PassSource(name='a')` through a stray click on empty canvas (round 1 wrote `NoSource()` here) |
| 3 | A wire drop lands on the wrong sampler below zoom 1 | **PRESS CLOSED / DROP REGRESSED** | Press at zoom 0.4 now resolves each of the four slots to itself (`u_p→u_p … u_s→u_s`; round 1 gave `u_p, u_p, u_q, u_r`). The drop half is worse than before, not better — see **N1**: nothing is written at any zoom |
| 4 | Every canvas gesture writes through the copilot-turn bracket | **CLOSED for the write / OPEN as N2** | With `copilot.state.in_flight` driven (the flag `ui.py` re-derives each frame), a node drag released under the turn leaves `b` at `(0.0, 300.0)` and a wire drag released under the turn leaves `b.u_src = PassSource(name='a')`. Nothing is written during the turn. The gesture is not actually cancelled, though — see **N2** |
| 5 | `Create` with a blank name silently dissolves the selection's group | **CLOSED** | Blank name + live selection: `group_selection` calls `[]`, groups unchanged `{a: 'g', b: 'g'}`; empty selection + good name: calls `[]`, groups unchanged; control (good name + selection): `['newg']`, `a` becomes `newg`. The popup stays open across four frames on the blank Create |
| 6 | `labels.index(clicked)` defeats D3's tab disambiguation | **CLOSED** | Document renamed `twin` with a group `twin`: labels `['document', 'twin']`, distinct, `active='document'`; index 0 → scope `''`, index 1 → scope `'twin'` |
| 7 | `plan_import` enforces only half of D17's one namespace | **CLOSED** | A copied pass named like a host group is rejected: `'a pass and a group cannot share a name'`. Same for the prefixed form (`g_bloom` vs host group `g_bloom`). The group's own half still fires (group named like host pass `h`). Control with no host group returns an `ImportPlan` |
| 8 | A feedback port inside a group tab draws a spurious second wire | **CLOSED** | `a` with `u_src = PassSource('a')`, grouped into `g`: in scope `'g'` the edges are `p:a 0 → p:b 0` only, self edges `0`, and the self-read is carried by the single `prev`-kind port on `p:a` that `_draw_self_loop` draws. Round 1 had an extra `p:a 0 → p:a 0 SELF` edge routed to the bus |
| 9 | Pressing an unwired port row is a dead zone | **CLOSED** | Press on `b`'s `unfilled` `u_src` port: `node_drag=NodeDrag(origin={'b': (0.0, 300.0)}, …)`, `wire_drag=None`, and the release commits `b` to `(40.26, 340.26)` |
| 10 | Two ghosts in a group tab can overlap exactly | **CLOSED** | Two feeders hand-placed at the same `y=100`, one with three ports so its size differs: `g:in:f1 (228, 100) size (108, 162)`, `g:in:f2 (228, 282) size (108, 130)` — overlapping pairs `[]`. With `f1` also reading a member it gets two entries in two columns (`g:in:f1` and `g:out:f1`), still zero overlaps |
| 11 | The no-session-write gate does not cover the menu the widget shares | **CLOSED** | `Leave group` is now `App.leave_group`; the gate reads both `pass_graph` and `pass_list`. Mutation-tested: injecting `app.session.set_pass_groups(...)` into `_draw_canvas` fails the test (`rc 1`), restoring it passes (`rc 0`), file byte-identical afterwards |
| 12 | `dissolve_group` on a name with no members still saves | **CLOSED** | `dissolve_group("") → '' saves=0`, `dissolve_group("nope") → '' saves=0`, control `dissolve_group("g") → '' saves=1` |

Ten of twelve closed outright. Finding 3 closed on its press half and regressed on its drop half;
finding 4's write is closed but its cancel is not, and both carry forward as new findings.

---

## New findings

### N1. BLOCKER — a wire can no longer be dropped on any port, at any zoom

**Symptom.** The fix added `imgui.set_next_item_allow_overlap()` immediately before each port's
`invisible_button` in `_draw_canvas`. That call marks an item as *overlappable by a later item* —
the installed stub says so in as many words: "allow next item to be overlapped by a subsequent
item. Typically useful with InvisibleButton() … covering an area where subsequent items may need to
be added." The ports are the LAST items submitted in the chain, so there is no subsequent item to
yield to; the flag only makes them forfeit their own hover. `drop_target` is assigned from
`imgui.is_item_hovered()` on exactly those buttons, so it is never set, and `_drop` always takes
its no-target branch.

**Demonstration.** Drag `a`'s output dot onto `b.u_p`, five frames of dwell on the target before
release, identical probe on both trees:

```
zoom=0.5 tree=head: drag a.out -> b.u_p => u_p=AutoSource()
zoom=1.0 tree=head: drag a.out -> b.u_p => u_p=AutoSource()
zoom=2.0 tree=head: drag a.out -> b.u_p => u_p=AutoSource()
--- baseline 57b137e ---
zoom=0.5 tree=base: drag a.out -> b.u_p => u_p=PassSource(name='a')
zoom=1.0 tree=base: drag a.out -> b.u_p => u_p=PassSource(name='a')
zoom=2.0 tree=base: drag a.out -> b.u_p => u_p=PassSource(name='a')
```

Traced to the frame. On the release frame the baseline hovers three overlapping ports and takes the
last (round 1's wrong-slot bug); HEAD hovers none at all:

```
### HEAD ###     drops: [None]                      all port hovers: []
### BASE ###     drops: [('b', 'u_r', 'unfilled')]  all port hovers: [(11, '##gport_p:b_0'),
                                                                     (11, '##gport_p:b_1'),
                                                                     (11, '##gport_p:b_2')]
```

Isolated to the one call by suppressing only the `allow_overlap` that precedes a `gport` button,
everything else on HEAD unchanged:

```
MODE=ctrl:   release-frame port hovers=[]                       drops=[None]                       wrote_u_q=AutoSource()
MODE=noao:   release-frame port hovers=[(11, '##gport_p:b_1')]  drops=[('b', 'u_q', 'unfilled')]   wrote_u_q=PassSource(name='a')
```

Reproduced on a pristine `ed1c28b` worktree, so it is in the commit and not in another reviewer's
concurrent edits to the working tree.

**Fix, verified.** Delete that one `set_next_item_allow_overlap()` before the port
`invisible_button` (the node-body and background ones are correct and must stay — they are the
earlier items in the chain). On a worktree with only that line removed, every slot at every zoom
resolves correctly, and the pitch clamp's press resolution is untouched:

```
zoom=0.4 hit_half=3.20 pitch=6.40
  slot 0 (u_p) drop_target=('b','u_p','unfilled') wired=['u_p'] OK
  slot 1 (u_q) drop_target=('b','u_q','unfilled') wired=['u_q'] OK
  slot 2 (u_r) drop_target=('b','u_r','unfilled') wired=['u_r'] OK
  slot 3 (u_s) drop_target=('b','u_s','unfilled') wired=['u_s'] OK
ALL OK                      (same at zoom 1.0 and 2.5)

press on slot 0 (u_p) -> active ['##gport_p:b_0'] grabbed ('b','u_p')
press on slot 1 (u_q) -> active ['##gport_p:b_1'] grabbed ('b','u_q')
press on slot 2 (u_r) -> active ['##gport_p:b_2'] grabbed ('b','u_r')
press on slot 3 (u_s) -> active ['##gport_p:b_3'] grabbed ('b','u_s')
```

That closes finding 3 on both halves at once — the pitch clamp fixes the press, removing the flag
fixes the drop.

**And the gate should learn this class.** `make gates` is green with wire-dropping dead. A gate
that drives a real press-move-release onto a port through `update_and_draw` and asserts the sampler
took the aimed producer is the check that would have caught it; whatever shape it takes, it must be
broken once (remove the flag's removal, watch it go red) before it is believed.

### N2. REAL — the copilot cancel clears the gesture and the same frame re-arms it; the write lands after the turn

**Symptom.** The cancel block at the top of `_draw_canvas` sets `view.node_drag = None` /
`view.wire_drag = None` when `app.copilot_turn_active`. Further down the same frame, the node's and
the port's press branches are still reached — the item is still `is_item_active()` and
`is_mouse_dragging` is still true, because the button never came up — and their `view.node_drag is
None` / `view.wire_drag is None` guards are now *satisfied by the cancel itself*. A fresh gesture is
constructed every turn frame. Nothing is written during the turn (finding 4's write is genuinely
closed), but the gesture is alive when the turn ends, and the user's release then commits it.

**Demonstration, node drag.** Press and drag `b`, start a turn, keep dragging, end the turn,
release. Exactly one `NodeDrag` is constructed per turn frame, its `delta` frozen at one frame's
motion:

```
drag armed, delta (129.32, 129.32)
  turn frame 1: gnode active=True node_drag=(64.7, 0.0)
  turn frame 2: gnode active=True node_drag=(64.7, 0.0)
  turn frame 3: gnode active=True node_drag=(64.7, 0.0)
turn over: node_drag (64.66, 0.0)
committed: (64.66, 300.0)      origin was (0.0, 300.0)
```

`b` moved to a coordinate no gesture asked for: the 129 px the user dragged before the turn are
discarded, the motion during the turn is discarded, and what survives is one arbitrary frame's
delta from whichever frame last rebuilt the drag.

**Demonstration, wire drag — this one destroys a wire.** Grab `b.u_src` (reading `a`), start a
turn, end it, release on empty canvas:

```
wire armed:   WireDrag(producer='a', start=(0.0, 422.0), grabbed=('b', 'u_src'))
  turn frame 0: wire_drag WireDrag(producer='a', …, grabbed=('b','u_src'))
  turn frame 1: wire_drag WireDrag(producer='a', …, grabbed=('b','u_src'))
  turn frame 2: wire_drag WireDrag(producer='a', …, grabbed=('b','u_src'))
turn over:    wire_drag WireDrag(producer='a', …, grabbed=('b','u_src'))
after release on empty canvas: b u_src = NoSource()
```

Instrumented to the construction, one rebuild per frame, the cancel visibly undone within the frame:

```
armed at frames: [7]
turn frame 0: WireDrag constructions this frame = 1; pre=True post=True
turn frame 1: WireDrag constructions this frame = 1; pre=True post=True
turn frame 2: WireDrag constructions this frame = 1; pre=True post=True
```

Note this is reached only when the gesture was already dragging when the turn began (the item must
already be active). A drag that *starts* inside a turn never arms — probed, `node_drag` stays
`None` and nothing is written.

**Fix.** Make the cancel sticky for as long as the button stays down, rather than a per-frame
clear that the same frame can undo: latch the cancellation (a flag on `GraphViewState` cleared when
the button next comes up) and gate the two press branches on it, so a gesture cancelled by a turn
cannot be rebuilt until the user releases and presses again.

**Severity note.** Both the round-1 report and the spec's Review history state the copilot guard as
done. The write it was aimed at is done; the cancel it claims is not, and the wire case still loses
a wire the user never meant to cut — the same symptom finding 2 was raised for, reached by a
different route.

---

## The specific new-risk checks, all clean

These were the places the round-1 fixes could plausibly have broken something. Each was probed and
each is fine.

- **The cancel block versus the release frame.** The task's worry does not materialise. The guard
  is `not is_mouse_down(left) and not is_mouse_released(left)`; on a release frame
  `is_mouse_released` is true, so the second conjunct is false and the cancel does not fire.
  Measured across a whole gesture: normal drag frame `down=True rel=False cancel=False`, release
  frame `down=False rel=True cancel=False`, idle frame `down=False rel=False cancel=True`. A
  normal drag commits: `b` moves off `(0.0, 300.0)` to `(161.66, 380.83)` on the release.
- **The port hit clamp with one port, and at zoom 2.5.** The clamp is
  `min(max(GRAPH_PORT_R·z, GRAPH_HIT_MIN), GRAPH_PORT_ROW·z/2)`. It never falls below the drawn
  dot radius at any zoom, so a click on the visible dot always lands: zoom 0.25 `hit=2.00 dot=1.00`,
  0.4 `3.20/1.60`, 0.875 `7.00/3.50`, 1.0 `7.00/4.00`, 2.5 `10.00/10.00`, 4.0 `16.00/16.00`. With a
  single port the pitch term is the same constant, so the clamp applies harmlessly — it bounds the
  box at half a row even though no second row exists, which costs nothing because the bound still
  covers the dot.
- **The output-dot hit at 1, 2, 5 and 8 outputs.** `out_hit` never reaches half the step, so two
  output dots can never share a press: 8 outputs at zoom 0.4 gives `out_hit=1.78` against
  `step=3.56`; at zoom 2.5 `out_hit=10.00` against `step=22.22`. Round 1's "5 outputs already
  overlap at zoom 1.0" is gone.
- **The badge helper at zoom 0.25 and 2.5.** The label stays inside the pill horizontally at every
  zoom — `calc_text_size` does track the pushed font, so text and pad scale together: zoom 0.25
  `text_w=16.00 pill_w=17.50`, zoom 2.5 `text_w=128.00 pill_w=143.00`, fits in all four cases
  measured. Vertically there is one sub-pixel artifact at zoom 0.25 only: the font size floors at
  `4.0` while the pill height is `12·0.25 = 3.0`, giving a `-0.50` px offset. Half a pixel of glyph
  above a 3 px pill at the minimum zoom is not a defect worth a change.
- **Ghost stacking when a ghost both feeds and reads, and when sizes differ.** Both columns start
  at the members' top and advance by each ghost's own height, so a pass that does both gets two
  entries in two columns (`g:in:f1` at `x=228`, `g:out:f1` at `x=572`) and a feeder whose size
  differs still stacks without overlap. Zero overlapping pairs in both configurations.
- **`_snap` when `still` is empty.** Every node dragged: `guides=[]`, `snap=(0.0, 0.0)`, `current()`
  returns the raw offsets — the `default=None` on both `min()` calls carries it. An empty node list
  behaves the same.
- **`namespace_error` on a rename to itself.** `rename_pass` returns `""` at `if new == old` before
  it ever calls `_pass_name_error`; confirmed positionally in the function's source (the early
  return precedes the check). A pass renamed to its own name cannot collide with a group of that
  name, because the check is never reached.

---

## False trails (probed, and fine)

- **An empty `NodeDrag.origin`.** `_snap` does raise `StopIteration` at
  `next(iter(drag.origin))` on an empty origin — but it is unconstructible. `draw` runs
  `view.selection &= set(document.passes)` every frame before `_draw_canvas`, so a stale selection
  cannot reach `_drag_names`; a box's members come from `strip_order` over `document.passes`; and
  `picture.positions` covers every pass (`set(doc.passes) <= set(pic.positions)` measured True).
  Both construction sites therefore always yield at least one entry. Not a finding.
- **The one-step offset at the head of a drag.** A drag commits the cursor's position minus one
  step of motion (117 px of cursor travel commits 114) because the frame on which
  `is_mouse_dragging` first fires is the frame the drag is constructed on, so that frame's
  `io.mouse_delta` is not accumulated. Present on the baseline too, inherent to arming on a
  dragging predicate, and invisible at the scale of a real gesture. Not the snap latch, which is
  separately confirmed closed.
- **`pass_list.py`'s remaining `session.` call.** `_delete_pass` calls
  `app.session.delete_pass(...)` directly, and the no-write gate's forbidden list
  (`set_sampler_source`, `set_pass_positions`, `set_pass_groups`, `set_pass_group(`) does not name
  it. That is deliberate scope, not an oversight: the gate exists for the 092 graph writes, and
  delete has its own editor-teardown path that predates this feature. Worth knowing the gate is
  narrower than its name suggests; not a defect in this diff.
- **Round 1's own false trails.** Not re-probed, and they stand: `wiring_if_renamed`'s restore on
  every path, the wheel-zoom invariance, the `_drop`/`unwire` order (brute-forced over every
  acyclic 4-pass graph), the allow-overlap chain on the background and node bodies, the position
  bounds rejection, the group prompt's open/close lifecycle. Nothing in this diff touches their
  mechanisms.

---

## Verdict

**FINDINGS.**

Round 1 is genuinely closed on ten of twelve items, and the two that are not are both in the
gesture block the round-1 verdict identified as the single shared root. Two must be fixed:

- **N1 (blocker)** — one stray `set_next_item_allow_overlap()` on the port items means a wire
  cannot be dropped on a port at any zoom. The fix is deleting that line, and it closes finding 3's
  drop half as well; verified in a worktree across three zooms and all four slots. It also wants a
  gate, because `make gates` is green today with the feature's central gesture dead.
- **N2 (real)** — the copilot cancel is undone within the same frame by the press branch it just
  unblocked, so a gesture in flight when a turn starts survives the turn and commits on the release
  after it, once turning a grabbed wire into `NoSource()`. The write during the turn is correctly
  prevented; the cancel needs to latch until the button comes up.

Both are regressions of the round-1 fix rather than survivals of the round-1 findings, and both sit
in `_draw_canvas`'s gesture block — one line to remove, one latch to add.
