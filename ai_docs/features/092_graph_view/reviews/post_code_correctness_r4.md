# 092 post-implementation review — code correctness, round 4 (closure)

Closing out round 3's single finding (`reviews/post_code_correctness_r3.md`, N3) and its
companion observation about the freeze test, against the fix commit `25c7d1e` ("092: fold the
post-implementation round 3"), which is `HEAD` on `dev`. Every row below was re-probed with the
round-3 harness unchanged: a headless `App` driven through `shaderbox.ui.update_and_draw`, with
`io.add_mouse_*_event` supplying the gesture and `view.port_rects` supplying the aim. The
mutations ran in a detached `git worktree` at `HEAD` in the scratchpad, restored byte-identically
(`git status --short` empty, `git diff HEAD` empty) and the worktree removed; the repo itself was
never modified, and no file in it except this report.

This is a closure round: the brief was to confirm, not to manufacture. Nothing new is filed.

## Gate result

| Gate | Command | Exit |
|---|---|---|
| `make gates` (check → test → smoke) | `make gates > log 2>&1; echo $?` | **0** — GREEN |

The smoke **ran**; it did not skip. The log's last line is
`== gates: GREEN -- check passed, test passed, smoke passed ==`, and the target's own
"stdout is not a terminal" warning is present above it, confirming the code came from the
unpiped redirect rather than from a pipe.

---

## Closure table

| # | Round 3 finding | Verdict | Observed now |
|---|---|---|---|
| N3 | A gesture cancelled by a turn was rebuilt on the first frame after it, and the release committed | **CLOSED** | Wire and node, both: nothing re-arms on any post-turn frame, and the release writes nothing. Values below |
| — | The freeze test pinned the `begin_disabled` wrappers, not the guards | **CLOSED** | Replaced by `test_a_press_that_spans_a_copilot_turn_never_becomes_a_gesture`, which is a live gate on **both** halves of the latch — each half mutated separately, each turning it red |

### N3, in full

The round-3 probe exactly: press before the turn, drag so the gesture arms, turn on, turn off
with the button still down, **no motion after the turn**, then release. `press_blocked` is read
on every post-turn frame alongside the two drag slots.

Wire case — grab `b.u_src` (reading `a`):

```
N3-wire armed before turn:        WireDrag(producer='a', start=(172.0, 177.0), grabbed=('b','u_src'))
N3-wire during turn:              ('None', 'None')
N3-wire post-turn frames (wire, node, press_blocked):
                                  [(False, False, True), (False, False, True),
                                   (False, False, True), (False, False, True)]
N3-wire b.u_src after release:    PassSource(name='a')
N3-wire model changed:            False
```

Node case — grab `b`'s body:

```
N3-node armed before turn:        NodeDrag(origin={'b': (172.0, 55.0)}, delta=(50.0, 40.0), snap=(0.0, 0.0))
N3-node during turn:              None / None
N3-node post-turn frames (wire, node, press_blocked):
                                  [(False, False, True), (False, False, True),
                                   (False, False, True), (False, False, True)]
N3-node positions after release:  {'main': None, 'a': None, 'b': None, 'c': None}
N3-node model changed:            False
```

Round 3's two symptoms are both gone. The wire that became `NoSource()` stays `PassSource('a')`;
the node that landed on `(252.0, 115.0)` — a coordinate no gesture asked for — stays at
`position None`, never committed. `press_blocked` reads `True` on every post-turn frame, which is
the latch doing the work: the mechanism is observed, not inferred from the outcome.

"Model changed: False" compares a snapshot of the whole document — every pass's uniform values,
every graph entry's position and group — before the gesture and after the release. It is the
whole document model, not one field.

---

## Mutation outcomes

The prompt asked for one mutation; the second was added because the test makes two claims and
only one of them was covered by the first.

| Mutation in `_draw_canvas` | Expected | Observed |
|---|---|---|
| Remove `view.press_blocked = True` (the `elif frozen` branch's body) | red | **RED** — `test_a_press_that_spans_a_copilot_turn_never_becomes_a_gesture` fails at `assert view.wire_drag is None and view.node_drag is None, "the press re-armed"`, with `WireDrag(producer='a', start=(172.0, 177.0), grabbed=('b','u_src'))` rebuilt. 9 others pass |
| Restore | green | **GREEN** — 10 passed, worktree diff vs `HEAD` empty |
| Remove `view.press_blocked = False` (the `if not mouse_down` branch's body — the latch never clears) | red | **RED** — the same test fails at its second half, `assert view.wire_drag is not None`, with `press_blocked=True` still latched. 9 others pass |
| Restore | green | **GREEN** — 10 passed, worktree diff vs `HEAD` empty |

Each mutation fails **exactly one** test, and it is the one that names the behaviour. The first
mutation reproduces round 3's N3 verbatim, down to the same `WireDrag` repr — so the test is
pinned to the defect it was written for, not to a neighbouring symptom. The second shows the
"next press is a gesture again" half is not decoration: a latch that never released would pass
the first half and be caught here.

This answers round 3's closing demand directly. The old test stayed green with all nine `frozen`
guards deleted, because three `begin_disabled` layers made the canvas unreachable during a turn.
The new test's window is the frames **after** the turn ends, when `copilot_turn_active` is false
and every `begin_disabled` is therefore inert — so the wrappers cannot carry it, and only the
latch can. The mutations confirm that: the wrappers were left untouched in both, and the test
went red anyway.

---

## The two edge checks

### A turn that starts and ends with the button UP

`press_blocked` stays False throughout, and the next press is a full gesture:

```
EDGE-A press_blocked during turn (button up): False
EDGE-A press_blocked after turn:              False
EDGE-A wire after the next press:             WireDrag(producer='a', start=(172.0, 177.0), grabbed=('b','u_src'))
```

This is the branch order in `_draw_canvas` doing what it reads as. `mouse_down` is computed
first, and `if not mouse_down: view.press_blocked = False` is tested **before** `elif frozen`, so
a turn with no press held cannot latch anything — the `elif` is unreachable while the button is
up. The latch's cost to the common case is zero: a turn the user was not mid-gesture through
leaves the canvas exactly as it found it.

### A release and a re-press both inside the turn

Sequence: press, arm, turn on, release during the turn, press again during the turn, turn ends
with that second press still down.

```
EDGE-B press_blocked after the turn saw the press: True
EDGE-B press_blocked after the release in-turn:    False
EDGE-B press_blocked after the re-press in-turn:   True
EDGE-B press_blocked after the turn ended:         True
EDGE-B wire after motion post-turn:                None
EDGE-B re-press blocked until its own release:     True
EDGE-B press_blocked after the real release:       False
EDGE-B the next fresh press arms:                  True
```

**What the code does.** The latch tracks the *current* button hold, not the turn. On the in-turn
release `not mouse_down` fires and clears it; on the in-turn re-press `elif frozen` fires and sets
it again. So the re-press inherits a fresh latch of its own and is blocked until **its** release,
which is one gesture's worth of blocking past the turn's end. The user's next press after that
release arms normally.

**Is it acceptable?** Yes, and for a stronger reason than the latch. That re-press was made while
the canvas was inside `begin_disabled(app.copilot_turn_active)`, so the port button never became
active and no gesture could have started from it regardless. Demonstrated rather than argued: the
same sequence run with the latch cleared **by hand** the instant the turn ends — the one thing
that could block the re-press removed — still arms nothing and still writes nothing:

```
EDGE-B control, latch cleared by hand, wire: None
EDGE-B control, b.u_src: PassSource(name='a')
```

So the latch is belt-and-braces here, not the load-bearing part, and it is blocking a press that
was already dead. The behaviour it produces is also the right one on its own terms: a press the
user made into a frozen canvas got no visual feedback, so treating its release as a no-op and
starting fresh on the next press is what the user's mental model expects. No finding.

---

## What was probed and is fine

- **The latch's branch order.** `if not mouse_down` before `elif frozen` in `_draw_canvas` is
  what makes EDGE-A free; the clear wins over the set on any frame where both could apply, which
  is the correct precedence — a button that is up can have nothing held across anything.
- **`released_elsewhere` still computes the same thing.** The fix hoisted `mouse_down` out of the
  expression rather than changing it: `not mouse_down and not is_mouse_released(left)` is the
  round-2 expression with its first term named. The cancel block it gates is unchanged, and the
  during-turn rows above (`('None', 'None')`, `None / None`) confirm the turn still cancels.
- **The nine guard sites all read `blocked`.** `blocked = frozen or view.press_blocked` widens
  every site round 3 examined rather than a subset — the band anchor, the node body, the wired
  port, the unfilled port, the output dot. Both N3 cases above exercise different sites (port and
  node body) and both hold, so the widening is not pinned on one path.
- **The rest of the suite is undisturbed.** Both mutations left the other 9 tests green, so the
  latch is not entangled with the drop gate, the cycle refusal, the media refusal, or the commit
  paths. The drop gate from round 3 was not re-broken this round; it was verified there and
  nothing in this commit touches the hover flag or the port submission.

## False trails (probed, and fine)

- **My snapshot called every gesture a write.** The first N3 run reported "model changed: True" on
  both cases and, read naively, would have been filed as "N3 is not fixed — the release still
  writes". It was the harness: my whole-document snapshot included `main.u_time`, the running
  clock, which advances every frame whether or not anything is touched
  (`0.16773594400001457 -> 0.285977143999844`). The wiring, the positions and the groups were
  identical in both snapshots. Recorded because the shape is generic and round 3 hit its own
  version of it: a "nothing changed" assertion over a live model must exclude the model's clock,
  or it reports the passage of time as a defect.
- **EDGE-B's `press_blocked=True` after the turn looked like the latch over-reaching.** It is not
  the latch that blocks that re-press — the control above shows the re-press arms nothing with the
  latch cleared by hand. Worth separating, because "the latch blocks a press the turn never saw"
  would have been a wrong finding about a correct mechanism.
- **The comment on `press_blocked` in `graph_state.py`.** It states what the field is for in
  present tense ("A press the copilot turn saw held down may not become a gesture even after the
  turn ends; the latch clears when the button comes up") rather than narrating the bug. That is
  the repo's comment rule. No change wanted.

---

## Verdict

**PASS.**

N3 is closed on both the wire case and the node case, with the latch observed set on every
post-turn frame and the document unchanged through the release. The freeze-gate problem is closed
too, and closed properly: the replacement test was broken twice, once per half of the latch, with
the `begin_disabled` wrappers left intact both times — which is exactly the falsification round 3
said had to happen before the new test was believed. Its window sits after the turn ends, where
the wrappers are inert, so it cannot be carried by them.

Both edge cases behave correctly and for defensible reasons: a turn with the button up costs
nothing, and a re-press inside a turn is blocked by the disable wrapper before the latch is even
consulted.

`make gates` is green at exit 0 with the smoke run. No finding is filed this round.
