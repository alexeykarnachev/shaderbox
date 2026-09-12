# 092 post-implementation review — architecture and conventions, round 3

Closing round 2's two findings (`post_architecture_conventions_r2.md`, A and B) against the fix
commit `c203784` ("092: fold the post-implementation round 2"), re-doing the literal scan one last
time, and judging the two things the fix added that round 2 never saw: `GraphViewState.port_rects`
/ `canvas_rect`, and the two frame-driven tests.

Gate state, captured unpiped: `make check` -> **0** (0 errors, 7 warnings, all
`reportMissingModuleSource` on upstream stub gaps). `uv run pytest tests/test_graph_view.py -q` ->
**0**, 10 passed.

## Closure table

| # | R2 severity | Verdict | What closes it |
|---|---|---|---|
| A | REAL | **CLOSED** | All six sites converted, and the two new tokens exist. `SIZE.GRAPH_LOOP_RISE = 12` and `SIZE.GRAPH_LOOP_REACH = 28` are in `theme.py` under one comment naming both ("A feedback loop's rise above the node and its horizontal control reach"); `_draw_self_loop` reads `SIZE.GRAPH_LOOP_RISE * z` for `top` and `SIZE.GRAPH_LOOP_REACH * z` at both control points. The three bare strokes now read their tokens: `_draw_node`'s `thickness = SIZE.GRAPH_WIRE_W if (node.error or is_output or selected) else 1.0`, `_draw_node`'s hollow output dot `dl.add_circle(center, r, out_col, 0, SIZE.GRAPH_PORT_RING_W)`, and `_draw_canvas`'s in-flight wire `max(1.0, SIZE.GRAPH_WIRE_W * view.zoom)`. Grepping the two widget modules for `GRAPH_WIRE_W` / `GRAPH_PORT_RING_W` returns five reader sites and no survivor: `_draw_wire`'s `thickness`, `_draw_self_loop`'s stroke, `_draw_port_dot`'s `ring`, `_draw_node`'s border and its output ring, `_draw_canvas`'s in-flight wire. The one-spelling-per-visual-decision property R2 asked for holds. |
| B | REAL | **CLOSED** | `_ROOT_LABELS` the three-element tuple is gone; `_ROOT_LABEL = "document"` is a single string, and `_tab_row` appends a counting suffix until the label is free: `root_label = _ROOT_LABEL; n = 1; while root_label in groups: root_label = f"{_ROOT_LABEL}_{n}"; n += 1`. The loop terminates because `groups` is finite, so some `document_n` is always free — no `next`, no `StopIteration`, and the distinctness `labels.index(clicked)` depends on is now a property of the construction rather than of a lucky tuple. This is exactly the "generated label counting up" R2 named as the cheap correct fix. |

Round 2 filed no NITs beyond finding C, which was recorded as no-change and stays that way:
`plan_import`'s one-sided namespace check is unchanged and still correct for the stated reason
(the import strips the source's groups, so there are no incoming group names to collide).

## The literal-number scan, re-done

`widgets/graph_state.py` — **clean**, and unchanged in substance from round 2. Every number is
`0.0` / `1.0` (an origin, an identity zoom, a delta accumulator) or a tuple index. `node_size`
reads `SIZE.GRAPH_NODE_W`, `GRAPH_BOX_EXTRA_W`, `GRAPH_PAD`, `GRAPH_THUMB`, `GRAPH_NAME_H`,
`GRAPH_PORT_TOP`, `GRAPH_PORT_ROW`. The two fields the fix added contribute one literal each:
`canvas_rect`'s `(0.0, 0.0, 0.0, 0.0)` default, which is an empty rect, and `port_rects`'s
`default_factory=dict`. Neither is a design number.

`widgets/pass_graph.py` — **no bare number or color remains that is not a token or a named module
constant.** The three R2 flagged and the self-loop's two are gone (above). Everything the scan
still hits falls in one of these classes, each re-checked this round:

| Class | Sites | Verdict |
|---|---|---|
| Named module constants | `_ZOOM_STEP`, `_MIN_DIRECT_DX`, `_BEZIER_BOW`, `_NONE_CORE`, `_PREV_INNER`, `_MEDIA_HALF`, `_BADGE_PAD`, `_BADGE_H`, `_BADGE_INSET`, `_FIT_MARGIN`, `_CYCLE_PREFIX`, `_ROOT_LABEL` | correct; `_FIT_MARGIN` is token-derived (`float(SPACE.LG)`) and `_ROOT_LABEL` is now one string rather than R2's tuple |
| Arithmetic halving / doubling | `/ 2.0` in `_thumb_rect`, `_out_point`, `_fit`, `_port_point`, `_draw_wire`'s `gx`; `2 * hit`, `2 * out_hit`, `2 * dash`, `2 * _BADGE_PAD`, `2 * _FIT_MARGIN` | fine — each doubles or halves a quantity that is itself a token or a constant |
| Degenerate-size floors | `max(1.0, avail.x)`, `max(1.0, p1[0] - p0[0])`, `max(4.0, font.legacy_size * z)`, `max(1.0, SIZE.GRAPH_WIRE_W * z)` | fine — a floor against a zero or sub-pixel rasterized size, not a design number |
| Structural counts and indices | `channels_split(2)` / `channels_set_current(0/1)`, `pop_style_color(1)`, `add_circle`'s `0` segment count, `add_image_rounded`'s `(0, 1)` / `(1, 0)` UVs, `max(0, edge.span - 2)`, `len(node.outputs) <= 1`, `node.runs > 1`, `edge.span > 1`, `(slot + 1)`, `alpha = ... else 1.0` | fine — a channel count, a stack depth, a UV corner, rank arithmetic, an opaque alpha |
| `2 * r + 2 * z` | the port label's offset from its dot | fine, unchanged from R2's judgement — one term of a spacing expression whose dominant term is the token-derived radius |
| `(x1 - x0) / 2` style centering in `_draw_badge` | `(_BADGE_H * z - imgui.get_font_size()) / 2` | fine — vertical centering |

Colors: **clean**. Every color reaches through `COLOR.*`, `fade(...)`, `group_tint(...)` or
`_u32(...)` of one of those. The picture tint is `_u32(fade(COLOR.WHITE, picture_alpha))`; the one
tuple-shaped expression, `(*tint[:3], COLOR.GROUP_FILL_ALPHA * alpha)`, is token-derived and matches
`pass_list.py`. No hex, no raw 3- or 4-tuple.

`theme.py`'s two new tokens carry integer values (`GRAPH_LOOP_RISE: int = 12`,
`GRAPH_LOOP_REACH: int = 28`), matching the neighbouring `GRAPH_BUS_CLEAR: int = 16` and the
canvas-unit convention the rest of the `GRAPH_*` block uses. Both are multiplied by `z` at the
only reader, so the int/float split is consistent with `GRAPH_GAP_X` / `GRAPH_BUS_STEP`.

## (3) `GraphViewState.port_rects` and `canvas_rect` — the right home, and not speculative

**Verdict: fine as transient UI state, correctly placed, and not the lazy-row trap.**

Three tests the conventions apply, each answered from the code rather than from the shape:

**Is this the lazy-row trap — a derived value cached on a model where a later reader mistakes it
for truth?** No, and the structural reason is that `GraphViewState` is *already* declared as
frame-scoped, non-persisted state, by its own module docstring: "Transient: nothing here is
persisted, and nothing off-draw writes it." The two fields sit beside `guides` (the snap guides
this frame, cleared by the same cancel block), `band_anchor` (a screen-space press point, so
already a screen-space frame value) and `fitted`. A screen-space rect on this dataclass is the
same kind of thing its neighbours are, not a new kind. The trap needs a *second* reader that
could be wrong about freshness; grepping `shaderbox/` for `port_rects` and `canvas_rect` returns
exactly two writers (`_draw_canvas`'s clear-and-rebuild and the per-port assignment) and zero
readers outside the widget — `app.py`, `ui.py` and `tabs/` never touch either. Only the two tests
read them, and they read on the frame after a draw.

**Is the rebuild honest?** Yes, and it is placed where it cannot go stale within a drawn frame:
`view.port_rects = {}` and `view.canvas_rect = (...)` sit at the top of `_draw_canvas`, above
`_build_view`, so every frame that draws the canvas discards the previous frame's aim before
computing this one's. The clear is unconditional — it is not inside the `released_elsewhere or
frozen` cancel block — so a frozen frame still rebuilds a truthful map, which is what the
copilot-turn test depends on when it reads `view.port_rects[("b", "u_src")]` after four frames.

**Could a stale value be read?** Only when `begin_child` returns false (the child is clipped or
collapsed), since `_draw_canvas` is the sole writer and `draw` calls it under `if child_open`. In
that state `port_rects` holds the last drawn frame's rects. That is harmless *because* nothing in
`shaderbox/` reads them: no shipped code path can act on a stale rect. Recorded rather than filed
— adding a clear in the `else` branch would be defending against a reader that does not exist,
which is the same speculative machinery the test below rejects.

**The speculative-machinery test.** It passes on the strict reading: the fields have a real,
present consumer (`test_a_wire_dropped_on_a_drawn_port_writes_that_port` and
`test_a_gesture_cannot_start_during_a_copilot_turn`), not a hypothetical future one. The
conventions' objection is to machinery built for a use that has not arrived; this is machinery
built for a use that arrived in the same commit and that caught a live regression — the drop being
dead at every zoom. The alternative shapes are all worse: a module-level dict in `pass_graph.py`
would be global state keyed by nothing; recomputing the rect in the test would duplicate
`_port_point`, `_Xf` and the `hit` clamp, so the test would agree with a copy of the widget's
geometry rather than with the widget, which is precisely the bug class it exists to catch (R1's
finding was a hit box overlapping the row pitch — a test computing its own rect would have
reproduced the wrong rect and stayed green).

Two smaller conventions checks on the new fields: both carry a comment stating what they are and
why, in the present tense, with no development-history narration ("rebuilt every draw, so a
headless test can aim a drop where a user would" / "The canvas child's screen rect this frame, for
the same reason"). The comment does name the test as the reason, which is a fact about the code as
it is now (there is no other reader), not a bug story. Both are fully annotated;
`port_rects` uses `field(default_factory=dict)` as the mutable default requires. No
`@staticmethod`, no `TYPE_CHECKING`, no suppression.

## (4) The two new tests

**One-reason: yes for both, with one qualification on the first.**

`test_a_gesture_cannot_start_during_a_copilot_turn` is cleanly one-reason. It sets
`app.copilot.state.in_flight`, presses inside a port rect, drags well outside, and asserts three
things that are one fact from three angles: no wire drag, no node drag, and — after the turn ends
and the button releases — the wiring unchanged and every `entry.position` still `None`. The last
two are the *consequence* being absent, not a second reason, which is the right shape for a
"nothing happened" test. Its comment states the regression it falsifies in two lines and names the
mechanism ("the turn's cancel ran, then the same frame's press re-armed the drag").

`test_a_wire_dropped_on_a_drawn_port_writes_that_port` is one-reason on its assertion (the drop
writes `PassSource("a")` into `c.u_src`) and carries a stated falsifier in its first comment:
re-adding `set_next_item_allow_overlap()` on the port rects makes it red. The qualification: it
holds three intermediate asserts — `("c", "u_src") in view.port_rects`, `view.wire_drag is not
None` mid-drag, and `view.wire_drag is None` after release. Read strictly these are three
different failure causes. Read as the test's own scaffolding they are diagnostic guards that turn a
confusing final-assert failure into a located one, each with a message or an obvious meaning, and
the middle one ("the wire in flight was cancelled mid-drag") is the difference between "the drop
did not land" and "the drag never survived to the drop". That is a reasonable trade and matches
what the repo's other frame-driven tests do. Not a finding.

**The comment about a press outside any window: a FACT, and it is imgui's own wording.** The test
says "a press outside any window is owned by the application and imgui reports no hover for the
rest of the drag", and presses at `canvas_rect`'s bottom-right corner minus six pixels to stay
inside the child. Verified against the installed binding's stubs rather than from memory —
`.venv/lib/python3.12/site-packages/imgui_bundle/imgui/__init__.pyi`, the `IO` struct:

```
mouse_down_owned: ...  # Track if button was clicked inside a dear imgui window or over None
                       # blocked by a popup. We don't request mouse capture from the
                       # application if click started outside ImGui bounds.
```

"We don't request mouse capture from the application if click started outside ImGui bounds" is the
same statement the comment makes, in imgui's voice. The comment is a restatement of documented
behavior, not a guess about why the test needed tuning.

Two neighbouring claims checked from the same primary source while there, both also facts:
- `HoveredFlags_.allow_when_blocked_by_active_item` — the stub's comment reads "Return True even
  if an active item is blocking access to this item/window. Useful for Drag and Drop patterns."
  That is exactly what the port-hover comment in `_draw_canvas` asserts, down to the drag-and-drop
  framing.
- `set_next_item_allow_overlap()` — "allow next item to be overlapped by a subsequent item."
  Confirms the direction the round-2 fix reasoned from (the flag makes an item overlappable by a
  LATER one, so the last rung gains nothing and loses its own hover), and confirms the module
  docstring's rewritten sentence and the imgui-ui skill's §8 rule.

## (5) The spec after HEAD

**No mismatch.** Every sentence checked against the code at HEAD.

The Review-history round-2 paragraph is accurate on all seven of its claims: the allow-overlap
removal (two `set_next_item_allow_overlap()` calls remain, on the background and the node bodies,
none on ports or output dots), the drag-and-drop hover flag, the `frozen` guard on all three
gesture starts (rubber band, node drag, both port and output-dot wire presses), the two
frame-driven tests, the `port_rects` / `canvas_rect` recording, the tokens, and the suffix loop
("appends a numeric suffix until no group carries it (the three-name tuple could raise
`StopIteration`)"). The parenthetical naming `StopIteration` is a statement about the code that was
replaced, inside a paragraph whose whole job is recording what changed, so it reads correctly.

The rewritten D-clauses match the code:
- **D3** — "the root label is made distinct from every group's before the row is drawn (`document`,
  with a numeric suffix appended until no group carries the label)" is now literally what
  `_tab_row` does. This sentence was written for the old tuple and is correct for the new loop;
  no edit needed.
- **D8** — "then (W2) one per port" is correct and, notably, does NOT claim the ports declare the
  flag; the allow-overlap sentence covers only the background and the node bodies, which is the
  code. The hit-box clause matches `min(max(GRAPH_PORT_R * zoom, GRAPH_HIT_MIN), GRAPH_PORT_ROW *
  zoom / 2.0)`.
- **D10** — `pass_menu_items`'s two gates (`len(document.passes) > 1`, the entry carrying a group)
  and the `App.leave_group` routing are all present in `pass_list.pass_menu_items`.
- **D12** — the two verbs, the widget toasting the refusal, and the cancel-at-top-of-frame sentence
  naming the copilot turn all match `App.drop_wire` / `App.unwire` and `_draw_canvas`'s cancel
  block.
- **D13** — `NodeDrag.snap` as a separate offset recomputed from the raw drag matches
  `graph_state.NodeDrag`. See the false trails on `NodeDrag.begin`.
- **D14** — the blank-name and empty-selection gates, and `App.group_selection`, match.
- **D17** — `namespace_error(candidate, pass_names, group_names)` with three callers
  (`group_name_error`, `project_session._pass_name_error`, `pass_import.plan_import`) and
  `NAMESPACE_COLLISION` defined once, all as the clause states.

The roadmap's 092 row is `in progress` and its banner says "post-implementation round 2 in flight",
which is the state this round is closing rather than a stale claim.

## False trails

Chased this round and found sound. Recorded so a later round does not re-spend the time.

- **`port_rects` going stale when `begin_child` returns false is not a defect.** `_draw_canvas` is
  the only writer and `draw` guards it with `if child_open`, so a collapsed canvas leaves last
  frame's rects in place — but grepping all of `shaderbox/` finds zero readers outside the widget,
  so no shipped path can act on them. Adding a clear in the `else` branch would defend against a
  reader that does not exist.
- **D13's `NodeDrag.begin(names, positions)` is not a new mismatch.** `graph_state.NodeDrag` has no
  `begin`; it is constructed directly. This was already recorded in round 1's Deviations paragraph
  ("`NodeDrag` is constructed directly (no `begin`)"), which is the spec's own convention for this
  class — the body describes the design, the deviation register describes what shipped. Same
  category as round 2's finding-7 false trail. Do not re-file.
- **The round-1 Review-history sentence "the port rects declare allow-overlap" is not stale text.**
  It is inside the paragraph recording what round 1 changed, and round 2's paragraph immediately
  below records that the same call was removed and why. A history paragraph describes its own
  round; rewriting it would erase the record that makes the round-2 paragraph legible.
- **`_BADGE_PAD` / `_BADGE_H` / `_BADGE_INSET` as module constants rather than `SIZE.*` tokens
  remains a judgement call, not a missed conversion.** Unchanged from round 2's recorded reasoning;
  they parameterize one private helper's internal geometry and no second reader can exist.
- **`GRAPH_LOOP_RISE` / `GRAPH_LOOP_REACH` being `int` while `GRAPH_WIRE_W` / `GRAPH_PORT_RING_W`
  are `float` is not an inconsistency.** The stroke widths are sub-pixel by nature; the loop's rise
  and reach are canvas-unit distances, matching `GRAPH_BUS_CLEAR`, `GRAPH_GAP_X`, `GRAPH_PORT_ROW`
  and the rest of the geometry block, and both are multiplied by `z` at their only reader.
- **The drop test's three intermediate asserts are not a one-reason violation.** They are located
  diagnostics on the path to one assertion, each of which would otherwise surface as a confusing
  final-assert failure.
- **The gate is genuinely green**, `make check` exit 0 captured unpiped, and
  `tests/test_graph_view.py` 10 passed.

## Verdict

**PASS.**

Both round-2 findings are closed at the code level, by the fix each one named: the four stroke
widths and the self-loop's two numbers all read tokens now, with `SIZE.GRAPH_LOOP_RISE` and
`SIZE.GRAPH_LOOP_REACH` added for the latter; the `StopIteration` path is gone, replaced by a
suffix loop that cannot exhaust. The literal scan over both widget modules turns up nothing that is
not a token, a named module constant, or arithmetic over one — the third consecutive round narrowing
this, and the first to end empty.

The two things the fix added stand up: `port_rects` / `canvas_rect` are transient screen-space
state on a dataclass whose docstring already declares itself transient, sitting beside two
neighbours of the same kind, with a real consumer and no reader that could misread their freshness;
and the two tests read as one-reason, with the drop test's imgui comment restating the binding's own
documented `MouseDownOwned` behavior rather than guessing at it.

No spec sentence contradicts the code at HEAD, so no replacement text is offered. The architecture
verdict from rounds 1 and 2 is unchanged: the pure half in `pass_graph.py`, the per-document state
in `graph_state.py`, one namespace with one predicate, every canvas write an App verb, and the
gate that pins it now covering both modules that draw the shared menu.
