# 092 post-implementation review — code correctness

Angle: bugs, races, GL-context and resource lifecycle, error handling, frame-timing and
draw-order behaviour, geometry off-by-ones, and every place a gesture writes something the
spec forbids. Diff under review: `f04821a` (W1) and `57b137e` (W2).

Every runtime claim below was produced by a probe: either a headless `App` driven through
`shaderbox.ui.update_and_draw` with synthetic mouse events (the `tests/conftest.py` `app`
fixture's construction, plus a spy on `_draw_canvas` to capture the child's screen origin),
or a pure-function probe over the GL-free modules. Probe transcripts are quoted inline.

## Gate results

| Gate | Command | Exit |
|---|---|---|
| `make check` (ruff + pyright) | `make check` | **0** (0 errors, 7 pre-existing `reportMissingModuleSource` warnings) |
| Test suite | `uv run python -m pytest tests -q -p no:cacheprovider` | **0** — 2360 passed, 4 skipped, 71.22 s |
| Smoke | `uv run python scripts/smoke.py` | **0** — `smoke: OK (200 frames, 7 documents)` (this box DOES have a display, so the 092 stretch ran) |

The smoke's 092 stretch is a real gate, not a decorative one. Mutation-tested: with
`revalidated_scope` changed to `return scope`, the smoke fails at
`frame 47: a scope no pass carries survived` and exits 1. The break was restored and
`git status` confirms no code file in `shaderbox/`, `scripts/` or `tests/` is modified.

## Coverage

Read end to end: `shaderbox/pass_graph.py`, `shaderbox/widgets/pass_graph.py`,
`shaderbox/widgets/graph_state.py`, `shaderbox/ui_regions.py`, `shaderbox/theme.py` (the
`GRAPH_*` block and `_GROUP_TINT_EXCLUSIONS`), `shaderbox/widgets/pass_list.py`
(`pass_menu_items` and the caption/button removal), `shaderbox/tabs/document.py`
(`_draw_passes`), `shaderbox/app.py` (`graph_views`, `graph_view_for`, `arrange_graph`,
`drop_wire`, `unwire`, `commit_node_drag`, `group_selection`, `dissolve_group`,
`forget_render_state`, `_on_document_deleted`), `shaderbox/project_session.py`
(`_pass_name_error`, `add_pass`, `rename_pass`, `set_pass_positions`, `set_pass_group`,
`set_pass_groups`, `set_sampler_source`, `import_passes`, both document-delete paths),
`shaderbox/document.py` (`wiring_if_renamed`, `effective_wiring`, `_reads_of`, the
`_feedback` bookkeeping), `shaderbox/pass_import.py` (`plan_import`),
`shaderbox/popups/import_passes.py`, `shaderbox/help_content.py`, `shaderbox/ui_models.py`,
`scripts/smoke.py` (frames 42-48 and the loop's draw-then-assert order),
`tests/test_graph_view.py`, plus the spec `03_spec.md` end to end and imgui-ui SKILL §3/§4/§8.

Read as diff hunks only, not end to end, with the reason: `tests/test_pass_graph.py`,
`tests/test_pass_verbs.py`, `tests/test_graph_state.py`, `tests/test_graph_persistence.py`,
`tests/test_ui_regions.py`, `tests/test_theme.py`, `tests/test_button_tiers.py`,
`tests/test_ui_prose_budget.py` — the architecture/conventions reviewer owns test-shape and
falsifier coverage; my angle is the runtime behaviour of the production code, and where a
test's claim mattered to a finding I verified the behaviour directly rather than trusting the
test.

Skipped entirely: `shaderbox/ui_primitives.py` beyond `text_tab_row` (unchanged by this
diff), the copilot tool surface (D19 adds nothing), and the export/share paths (untouched).

---

## Findings

### 1. BLOCKER — `_snap` latches the drag permanently; the node stops dead at the first guide and the committed position is wrong

**Symptom.** `_snap` writes its correction back into `drag.delta`. Once the moving node's
edge lands on a guide, every subsequent frame re-measures the distance as `0 <= threshold`
and re-corrects to zero offset, wiping the `io.mouse_delta` that `update()` just added. The
node is pinned to the guide for the rest of the gesture, and the release commits the guide's
coordinate, not the cursor's.

**Evidence.** Headless drag of pass `b` (start `(0, 200)`) with `a` placed at `(0, 0)`, so
`b` begins already aligned on x. Mouse dragged right in 3 screen-px steps:

```
step 10: mouse canvas dx=30  b.x=0.0 drift=-30.0  guides=[('v', 0.0)]
step 20: mouse canvas dx=60  b.x=0.0 drift=-60.0  guides=[('v', 0.0)]
step 39: mouse canvas dx=117 b.x=0.0 drift=-117.0 guides=[('v', 0.0)]
committed: {'a': (0.0, 0.0), 'b': (0.0, 200.0)}      <- b never moved at all
```

Second probe, node starting unaligned, to show the latch closing mid-drag:

```
step 90 : want dx=270 b.x=267.0 drift=-3.0 guides=[]
step 100: want dx=300 b.x=300.0 drift=+0.0 guides=[('v', 300.0)]   <- snaps to a.x
committed b: (300.0, 200.0)     (the cursor asked for x=327)
```

The only escape is a single frame whose motion exceeds `2 * GRAPH_SNAP_PX / zoom`
(6 screen px at any zoom), so a fast flick escapes and a deliberate drag never does. Since
`rank_layout` puts every column at one x and every row at one y, an unplaced document's nodes
all start mutually aligned — the default state is a drag that cannot move.

**Fix.** Keep the snap out of the drag's accumulated state: leave `NodeDrag.delta` as the raw
mouse accumulation and apply the guide offset as a separate display/commit adjustment
computed fresh each frame from the unsnapped `current()`, so the un-snap happens as soon as
the raw position leaves the band.

### 2. BLOCKER — a drag, wire or band orphaned by the canvas not drawing survives, resumes with no button held, and the next unrelated click commits a write

**Symptom.** `view.node_drag` / `view.wire_drag` are only cleared by a release handled inside
`_draw_canvas`. Any frame where the canvas does not draw — the `strip | graph` toggle, the
Document tab losing focus, a modal covering the pane, a document switch — swallows the
release. The state persists in `GraphViewState`; when the canvas draws again, `update()`
keeps accumulating `io.mouse_delta` with the button up, and the first
`is_mouse_released(left)` from any later click commits.

**Evidence, node drag.** Press on `b`'s body, drag, switch `app_state.passes_view` to
`STRIP`, release there, switch back, move the mouse, click once elsewhere:

```
drag in flight: NodeDrag(origin={'b': (0.0, 300.0)}, delta=(110.6, 110.6))
after release under STRIP:  node_drag still present, positions unchanged
after a mouse MOVE with no button down: delta=(368.7, 368.7)   <- still accumulating
after a stray click: {'b': (368.70748299319735, 568.7074829931973)}   <- committed
```

**Evidence, wire drag (worse: it destroys a wire).** Same sequence grabbing `b.u_src`:

```
wire in flight: WireDrag(producer='a', start=(300.0, 122.0), grabbed=('b', 'u_src'))
after release under STRIP: wire_drag still present, b u_src = PassSource(name='a')
after a stray click:       wire_drag None, b u_src = NoSource()
```

The stray click landed on empty canvas, so `_drop` took the grabbed-onto-empty branch and
wrote `NoSource()` over a wire the user never touched.

**Fix.** At the top of `_draw_canvas`, before the gesture blocks, cancel any in-flight
gesture whose button is no longer down: `if not imgui.is_mouse_down(MouseButton_.left):
view.node_drag = None; view.wire_drag = None; view.band_anchor = None; view.guides = []` (or
commit the node drag there if a silent discard is the wrong call).

### 3. BLOCKER — a wire drop lands on the wrong sampler at zoom below 1

**Symptom.** The port hit box is `max(GRAPH_PORT_R * zoom, GRAPH_HIT_MIN)` — a 7 px *screen*
floor — while the port ROW spacing scales with zoom (`GRAPH_PORT_ROW * zoom`). Below zoom
0.875 the 14 px hit boxes overlap. The port `invisible_button`s carry no
`set_next_item_allow_overlap()`, so the press goes to the earliest-submitted (topmost) slot,
while `drop_target` is assigned in the same loop and so ends up holding the LAST hovered
(bottom-most) slot. Press and drop therefore resolve to different, both-wrong, ports.

**Evidence.** A pass with four samplers at zoom 0.4. Press:

```
press on slot 0 (u_p) -> active ['##gport_p:b_0'] grabbed ('b', 'u_p')
press on slot 1 (u_q) -> active ['##gport_p:b_0'] grabbed ('b', 'u_p')
press on slot 2 (u_r) -> active ['##gport_p:b_1'] grabbed ('b', 'u_q')
press on slot 3 (u_s) -> active ['##gport_p:b_2'] grabbed ('b', 'u_r')
```

Drop, dragging `a`'s output onto the port drawn for `u_q`:

```
aimed at u_q (slot 1); wrote: {'u_p': AutoSource(), 'u_q': AutoSource(),
                               'u_r': PassSource(name='a'), 'u_s': AutoSource()}
```

`u_r` was rewired. This is a silent wrong write to the document — the most serious member of
the hit-testing family, because nothing in the UI says the drop went somewhere else.

The same arithmetic hits a box's output dots, whose spacing is `GRAPH_THUMB / (n+1) * zoom`:
at 5 outputs they already overlap at zoom 1.0, at 8 outputs the spacing is 8.9 px against a
14 px hit box.

**Fix.** Clamp the hit half-size so it cannot exceed half the row pitch —
`min(max(GRAPH_PORT_R * zoom, GRAPH_HIT_MIN), GRAPH_PORT_ROW * zoom / 2)` for inputs and the
equivalent against the output step — and make press and hover agree by resolving both from
one nearest-port computation rather than from two different loop positions.

### 4. REAL — every canvas gesture writes through the copilot-turn bracket

**Symptom.** D10 puts the whole canvas under `begin_disabled(app.copilot_turn_active)`, but
`begin_disabled` only gates *items*. The drag update, the drag commit, the wire drop and the
band release all read `io.mouse_delta` / `is_mouse_released` directly in `_draw_canvas` and
run regardless, so a gesture already in flight when a copilot turn starts still writes the
document mid-turn.

**Evidence, position write:**

```
drag in flight: True
under copilot turn: node_drag True delta (134.6, 134.6)
mouse moved under disabled: delta (336.5, 336.5)
released under copilot turn: positions {'b': (336.5, 636.5)}   <- set_pass_positions ran
```

**Evidence, sampler write:**

```
wire in flight: True
released under copilot turn: b u_src = NoSource()
```

**Fix.** Cancel rather than commit when `app.copilot_turn_active` — the same one-line guard
finding 2 wants, extended with `or app.copilot_turn_active`.

### 5. REAL — `Create` on the Group... prompt with an empty name silently dissolves the selection's group

**Symptom.** `_group_prompt` commits on `primary_button("Create") or entered` and closes when
`app.group_selection(...) == ""`. `group_name_error("")` returns `""` by design — the empty
label is how `set_pass_group` means "no group" — so an empty input is accepted, saved, and the
popup closes. The user pressed Create and got a Dissolve.

**Evidence.**

```
before: {'a': 'g', 'b': 'g'}
group_selection with an EMPTY name -> ''
after : {'a': '', 'b': ''}
```

An empty selection is likewise accepted (`group_selection` with `selection=set()` returns
`""`) and produces a no-op document save.

**Fix.** In `_group_prompt`, treat a blank `view.group_name.strip()` as not-committable (do
not call the verb, keep the popup open), and skip the call when `view.selection` is empty.

### 6. REAL — D3's tab-row disambiguation is defeated by `labels.index(clicked)`

**Symptom.** D3 requires the click to be mapped back to a scope "by INDEX, never by the
returned string, since a document named like a group would otherwise be ambiguous". `_tab_row`
does `index = labels.index(clicked)` — a *name* lookup that returns the first match. When the
document's `ui_name` equals a group name, clicking the group tab resolves to index 0, the
root. `text_tab_row` compounds it: its selectable id is `f"{name}##{id_}_{name}"`, so the two
identical labels share one imgui id. And `active = labels[scopes.index(view.scope)]` makes
both tabs highlight, so the row cannot even show which scope is open.

**Evidence.** Document renamed to `twin` with a group `twin`:

```
tab row labels: ['twin', 'twin']  active: twin
mapping a click on 'twin' -> index 0 -> scope ''
selectable ids: ['twin##graph_scope_twin', 'twin##graph_scope_twin']
```

**Fix.** Have `text_tab_row` return the clicked INDEX (or accept explicit per-item ids), and
map with that; failing that, disambiguate the root label in `_tab_row` when it collides.

### 7. REAL — `plan_import` enforces only half of D17's one namespace

**Symptom.** D17 requires one namespace for passes and groups, with `plan_import` among the
entry points. `plan_import` checks the *group* name against host pass names
(`group_name_error(group, host_names)`) but never checks the imported *pass* names against the
host's existing group labels — the mirror `add_pass` and `rename_pass` both enforce.
`host_wiring` carries only pass names, so the group labels never reach the function.

**Evidence.** Host has pass `h` in group `bloom`; a source whose pass is named `bloom` is
imported with no group prefix:

```
plan_import creating a pass named like a host GROUP -> ACCEPTED, renames={'bloom': 'bloom'}
add_pass('bloom')           -> 'a pass and a group cannot share a name'
rename_pass('h'->'bloom')   -> 'a pass and a group cannot share a name'
```

The canvas survives the resulting state (keys are prefixed `p:` and `b:`, so the root's node
dict does not collide — I drew both scopes without error), so this is a spec-compliance hole
rather than a crash, but it is the exact two-root-entities-one-name state D17 exists to
prevent.

**Fix.** Pass the host's group labels into `plan_import` and reject a `renames` value that
lands on one.

### 8. REAL — a feedback port inside a group tab draws a spurious second wire

**Symptom.** `_build_view`'s root branch skips a self-read
(`if source is None or source == name: continue`); the group-scope branch has only
`if source is None: continue`. A member with a feedback port therefore gets a normal `_Edge`
from its own output slot to its own port slot, ON TOP of the `_draw_self_loop` that
`port.kind == "prev"` already draws.

**Evidence.** Pass `a` with `u_src = PassSource('a')`, grouped into `g`:

```
scope=''  -> node b:g []                (no edge; the root branch skipped it)
scope='g' -> edge p:a 0 -> p:a 0 SELF
             edge p:a 0 -> p:b 0
```

The self-edge's `dx` is `-GRAPH_NODE_W = -108`, so `_draw_wire` takes the `backward` branch
and routes it down to the bus below the whole graph and back — a long stray line across the
canvas, in addition to the correct loop over the node's top.

**Fix.** Add `or source == name` to the group branch's skip, matching the root branch.

### 9. NIT — pressing an unwired port row is a dead zone

**Symptom.** A port's `invisible_button` takes the press (the node body never activates), but
the port's drag branch is gated on `port.kind == "wired"`, so a press on an `unfilled`,
`none` or `media` port does nothing at all — the node cannot be dragged from that 14x14 px
region, and no other gesture starts there.

**Evidence.** Positions reset to unplaced before each gesture so the geometry is identical:

```
body (picture):            active=['##gnode_p:b']   node_drag=yes wire=no
ON the UNFILLED port dot:  active=['##gport_p:b_0'] node_drag=no  wire=no
```

**Fix.** When the pressed port is not `wired`, start a node drag instead (the same
`_drag_names` / `NodeDrag` construction the body branch uses).

### 10. NIT — two ghosts in a group tab can overlap exactly

**Symptom.** Feeder ghosts all share one x (`left - size - gap`) and take their y from the
pass's own position, so two feeders placed at the same y draw at the same rect; the later one
wins every hit and the wires visually collapse.

**Evidence.** `f1` and `f2` both hand-placed at y=100:

```
g:in:f1 pos (128.0, 100.0) size (108.0, 110.0)
g:in:f2 pos (128.0, 100.0) size (108.0, 110.0)
overlapping ghosts: [('g:in:f1', 'g:in:f2')]
```

Under the default rank layout the ghosts separate (`(0,0)` and `(0,130)`), so this needs
hand-placed positions — hence NIT.

**Fix.** Stack the ghost column by index rather than inheriting the source pass's y.

### 11. NIT — the widget's no-session-write gate does not cover the menu it shares

**Symptom.** `test_the_widget_makes_no_session_write_of_its_own` greps only
`widgets/pass_graph.py`'s own source. The canvas's node menu calls
`pass_list.pass_menu_items`, which calls `app.session.set_pass_group(...)` directly — a
session write reached from the canvas that the gate cannot see. Harmless today (that verb now
routes through the validated `set_pass_groups`), but the gate reads as enforcement broader
than it is.

**Fix.** Extend the grep to the transitive helpers the widget calls, or move the `Leave group`
write behind an `App` verb like every other canvas write.

### 12. NIT — `dissolve_group` on a name with no members still saves

**Symptom.** `dissolve_group(document_id, "")` collects every ungrouped pass and issues a
`set_pass_groups(members, "")`, and `dissolve_group(document_id, "nope")` issues
`set_pass_groups([], "")` — both return `""` and save the document once for no change.

**Evidence.** `dissolve_group("") -> '' saves 1` and `dissolve_group("nope") -> '' saves 1`.

Not reachable from the UI (the box menu always passes a live group name), so NIT.

**Fix.** Return early when `members` is empty, or when `group` is falsy.

---

## False trails (probed, and fine)

- **`wiring_if_renamed`'s in-place re-key.** Restores on every path. Probed the normal return
  and a forced raise inside `effective_wiring`: key order restored, `id(document.passes)`
  unchanged, every `uniform_values` dict identity preserved, every value restored. The other
  holder of `passes` in the same object, `Document._feedback`, is keyed independently and is
  never touched; `Document.render` and the strip both read `passes` only inside their own
  synchronous frames, and nothing yields between the re-key and the restore. One cosmetic
  divergence, not a defect: the real `rename_pass` does `pop` + reinsert, which moves the pass
  to the END of the dict, while `wiring_if_renamed` preserves position — so the *planned*
  wiring's iteration order differs from the post-rename order. Cycle existence is
  order-independent, so the guard's verdict is unaffected.
- **The wheel-zoom pan correction.** Exactly invariant. The canvas point under the cursor
  reproduces to 0.000e+00 for wheel `+1`, `-1` and `+3`, and a zoom already at the clamp
  leaves `pan` untouched rather than drifting.
- **The drop/unwire order in `_drop`.** `drop_wire` (new edge) runs before `unwire` (old
  edge), so the refusal is planned against a wiring that still holds the old edge. Brute-forced
  every wire move over all acyclic 4-pass / 2-sampler graphs — **4 157 664 moves, 0
  order-dependent refusals**. Structurally sound: removing an edge OUT of the producer cannot
  break a path INTO the producer.
- **`_out_point` with an empty `outputs`.** The `len(outputs) <= 1` branch handles 0 without
  dividing by zero, and `outputs` is never empty anyway — `group_boundary` always includes the
  bundle, and `bundle_output` is total over a non-empty member list. `bundle_output([])` does
  raise `IndexError`, but `_build_view` derives groups from `group_names_in_order`, so an
  empty member list is unconstructible.
- **`bundle_output` / the box when every member is off-plan or erroring.** Probed a group
  whose two members both fail to compile and that nothing outside reads: one output port (the
  bundle), `error=True`, `stale=True`, a valid `texture_glo`; both the root and the group tab
  drew without error.
- **An edge referencing a node key not in `view.nodes`.** Probed the non-convex shape
  (`g = {a, c}` with ungrouped `b` between them) and a member reading a pass in another group
  (`h = {d}`, `c` reads `d`). All three scopes: zero dangling keys, zero out-of-range slots,
  all three drew. Both `src_of` and the box-slot lookup return `None` and `continue` when the
  endpoint is not exposed.
- **`graph_ranks` KeyError on `ranks[source]`.** `graph_ranks` does
  `ranks.setdefault(name, 0)` for every key of `wiring`, `effective_wiring` keys every pass,
  and `wired_pass` resolves only to existing passes. Unreachable.
- **`arrange_graph`'s compile ordering.** `compile_pending_passes` runs before
  `sampler_names`. Probed with a pass whose program was never built: `samplers []` before,
  `samplers ['u_src']` after, one save, no pass left unplaced.
- **The layout shifting under a drag.** `_positions` applies `overrides` AFTER `rank_layout`
  and never feeds them into the barycentre tiebreak, so dragging one unplaced node leaves the
  other unplaced nodes exactly where they were (`c/d moved? False`).
- **`_fit` at a zero-size child.** Declines (`fitted` stays False, so a later frame retries)
  rather than dividing by zero; also declines an empty node list; clamps to
  `GRAPH_ZOOM_MIN` on an extreme bounding box without raising.
- **The rubber band leaving the child.** Starting a band and dragging the cursor far outside
  the child, then releasing there, selects correctly and clears `band_anchor`; no stuck state
  over the following frames.
- **Deleting the dragged pass mid-drag.** `commit_node_drag` filters `moved` to
  `name in document.passes`, so the release writes nothing and no `no such pass` error is
  raised. Document deletion evicts `graph_views` through `forget_render_state` on both delete
  paths (`_delete_document_unguarded` and the disk-sync `removed` loop, which both funnel into
  `_on_document_deleted`), and `draw` returns early on a missing `ui_document`.
- **The port-over-body allow-overlap chain.** Correct per imgui-ui §8: the chain declares
  `set_next_item_allow_overlap()` on the background and on each node body, and the ports come
  last. Verified — a press on a wired port gives `active=['##gport_p:b_0']` and grabs the wire
  while `node_drag` stays `None`.
- **A draw writing the graph.** Six frames of the graph view over a fresh multi-pass document
  left every `position` at `None`; the smoke's frame-47 dump comparison is the standing gate
  and it is a real one (mutation-tested above).
- **Position bounds.** `set_pass_positions` rejects `inf` and `1e30` with
  `a position is out of range: 1 value(s) rejected` and leaves the model untouched; a drag
  commit of `1e9` is refused the same way, so an out-of-range gesture cannot corrupt
  `graph.json`.
- **The group prompt's open/close lifecycle.** `open_popup` fires exactly once
  (`group_prompt` flips True->False on the frame it opens and stays False), and the one opener
  (`_node_menu`) resets `view.group_name = ""`, so no stale buffer reaches a reopen through any
  reachable path.
- **Every group-writing entry point.** `set_pass_group`, `group_selection`, `dissolve_group`
  and the copilot's `pass_set_group` all funnel through `set_pass_groups` ->
  `group_name_error`; `add_pass` and `rename_pass` carry the mirror check via
  `_pass_name_error(name, passes, graph)`. Only `plan_import` has the gap (finding 7).
- **The canvas making a direct session write.** `grep` over
  `widgets/pass_graph.py` and `widgets/graph_state.py` for `session.` returns nothing; the
  widget's only `app.*` calls are `arrange_graph`, `commit_node_drag`, `dissolve_group`,
  `drop_wire`, `group_selection`, `unwire`, `pick_pass`, `open_add_pass`,
  `open_import_passes`, plus read-only state. The one indirect write is finding 11.
- **The smoke stretch's asserts exercising what they claim.** The loop is
  `update_and_draw` FIRST, then the `if frame_idx ==` block — so frame 45 sets the scope after
  its draw, frame 46's draw consumes it, and frame 46's block asserts it survived that draw.
  Genuine. (The spec's Files-touched names a `_count_saves` helper beside
  `_arm_feedback_canary`; the code uses an inline `mock.patch.object` instead. Equivalent
  assertion, so a spec/code wording divergence rather than a defect — worth a spec touch-up.)

---

## Verdict

**FINDINGS.** Must be fixed: **1, 2, 3** (blockers — a drag that cannot move and commits the
wrong coordinate; orphaned gestures that write on an unrelated later click, including
destroying a wire; a wire drop that lands on the wrong sampler below zoom 1), and **4, 5, 6,
7, 8** (real — writes during a copilot turn; Create-with-empty-name dissolving a group;
D3's tab disambiguation not actually implemented; D17 enforced on only one side of the
import; a spurious duplicated wire in every group tab containing a feedback pass).

Findings 9-12 are NITs and can ride the same wave or a later one.

Findings 1, 2, 3 and 4 share one root: the gesture state machine's lifecycle is driven
entirely from inside `_draw_canvas` with no entry guard and no cancel path, and the hit
geometry is computed twice from two different loop positions. A single pass over
`_draw_canvas`'s gesture block — one cancel guard at the top, one nearest-port resolution
shared by press and hover, and the snap moved out of `NodeDrag.delta` — closes all four.
