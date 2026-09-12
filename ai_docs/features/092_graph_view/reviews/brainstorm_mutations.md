# 092 review — the graph under every mutation of the flat model

Angle: what the root box and the group tab show after each mutation the code can perform, and
where the boundary-edge rule set (fixed item 4) gives one answer versus a fork.

The rule set under test, restated from `01_brainstorm.md`:

- A box's **input port** = a member sampler reading an outside pass or nothing.
- A box's **output port** = a member read from outside, plus the bundle's own output.
- The **bundle output** is the mock's `bundleOutput`: the document output if it is an unread
  member, else the first unread member some outside pass reads, else the last unread member,
  else the last member.
- A **ghost** in a group tab = an outside pass a member reads, or an outside pass that reads a
  member; a pass that does both appears twice.

Every probe below ran with `uv run python` over `pass_graph` / `pass_import` (GL-free), building
a wiring dict of the shape `Document.effective_wiring` returns and calling the real planner. The
group is a dict `pass -> group` standing in for `PassEntry.group`.

---

## 1. Delete a member that other members read

**(a)** `ProjectSession.delete_pass` pops the pass, calls `Document.drop_feedback`,
`Document.forget_pass_sources` (every `PassSource` naming it becomes `AutoSource`) and
`_graph_without`. The reader's sampler then re-resolves through `wired_pass` with an
`AutoSource`, so the **name rule decides it again**. Probed on a bloom-shaped group
(`scene → b_bright → b_blur → b_comp → final`, the middle three grouped):

```
before: plan ['scene','b_bright','b_blur','b_comp','final']
delete b_blur:
  plan  ['scene','b_bright','b_comp','final']     -- still plans, no GraphError
  boundary bloom: in [('b_bright','u_scene','scene'), ('b_comp','u_scene','scene')]
                  out [('final','u_b_comp','b_comp')]
```

The chain is severed silently. `b_comp.u_blur` is now an `AutoSource` named `u_blur` matching no
pass, so `wired_pass` returns `None` and it is **not in the wiring at all** — no edge, no chip on
the strip, black at bind time. `b_bright` keeps producing into nothing.

**(b)** The root box is **unchanged**: same two input ports, same one output port, same picture.
Nothing at the root level tells the maintainer that the group's interior fell apart. The group
tab shows the severing: `b_bright` with no outgoing edge, `b_comp` with a port that has no wire.

**(c)** One answer for the ports. A fork on **whether the root box may look identical after a
destructive edit** — see Decision 1.

## 2. Delete the member that IS the bundle output

**(a)** Same `delete_pass` path. `bundleOutput` re-derives from scratch on the next frame.

```
before: bundle out = b_comp
delete b_comp:
  bundle out = b_blur           -- the picture jumps to a different member's texture
  boundary bloom: in [('b_bright','u_scene','scene')]   out []
  final's reads = {}            -- final is now an entry point
```

`final.u_b_comp` was rewritten to `AutoSource("u_b_comp")`, which names nothing, so **the box's
only output port disappears** and `final` becomes a root reading black.

**(b)** The box loses its output dot entirely, its picture silently becomes `b_blur`'s, and the
edge to `final` vanishes. A box with zero output ports is drawable but has no meaning under the
rule set — the rule says "plus the bundle's own output", and the bundle output always exists, so
this is a rule contradiction in the mock's own terms: `bundleOutput` returns `b_blur` yet
`b_blur` is not an output port because nothing outside reads it.

**(c)** Fork — Decision 2: is the bundle output always an output port (so the box always has at
least one dot), or only when something outside reads it?

## 3. Delete the document output while it is inside a box

**(a)** `_graph_without` picks the replacement: `output = graph.output if graph.output != removed
else next(iter(kept), "")` — the **first key of `document.passes`**, which is insertion order (the
order passes were added or loaded), not topological order and not group-aware. Probed:

```
passes inserted as [final, scene, b_bright, b_comp], output = b_comp (in group 'bloom')
delete b_comp -> new output 'final'      (an UNGROUPED pass)
```

**(b)** The document output leaves the box. Under the rule set the box's picture then falls to
`bundleOutput`'s next branch, and the accent border moves to a root-level node. The maintainer
sees the preview change to an unrelated pass with no explanation.

**(c)** This is a pre-existing wart, not a graph-view one. But the graph view makes it visible in
a way the strip did not: on the strip the accent border just moves one tile; on the graph the
output jumps out of a box. One answer is available (leave it), so not a fork — noted as context
for Decision 2.

## 4. Delete a member with no wires at all (the disconnected member)

**(a)** A pass can carry a group label while reading nothing and being read by nothing:
`set_pass_group` validates only the name pattern. Probed with a `lonely` pass labelled `bloom`:

```
boundary bloom: in [('b_bright','u_scene','scene'), ('b_comp','u_scene','scene')]
                out [('final','u_b_comp','b_comp')]
group_runs: [['scene'], ['b_bright','b_blur','b_comp'], ['final'], ['lonely']]
```

**(b)** `lonely` contributes **no port at all** — it is invisible in the box's boundary set. The
box's `count` badge says 4 members; the ports account for 3. The strip already shows this
honestly as a second run (`group_runs` is by adjacency, 091 D7). The graph's box hides it.

**(c)** One answer: the member count badge is the disclosure, and the group tab shows the
disconnected node. No fork, but it is the case that makes the "a box is its boundary edges" rule
incomplete as a *description* of the group — see the verdict.

## 5. The same group name on two disconnected passes

**(a)** A group is a label (091 D1); nothing checks connectivity. Two passes on opposite sides of
the DAG carrying `bloom` are one group.

**(b)** The root draws **one box** spanning both. Under fixed item 5 the box sits at its members'
bounding box, so it is a large rectangle enclosing unrelated root-level nodes that are not its
members. The strip's answer (two runs) has no graph analogue: a box cannot be two boxes without
becoming a group entity, which 091 D1 forbids.

**(c)** Fork — Decision 3: one box at the bounding box (swallowing non-members visually), one box
drawn as a discontinuous region, or "Arrange" is required to pull members together.

## 6. A group left with one member

**(a)** Reachable by deleting or ungrouping the others. Probed with only `b_comp` labelled:

```
boundary bloom: in [('b_comp','u_blur','b_blur'), ('b_comp','u_scene','scene')]
                out [('final','u_b_comp','b_comp')]
bundle out = b_comp
```

**(b)** The box's ports are **exactly the member's own ports**, and its picture is the member's
texture. The box is a node wearing a box's chrome, plus a tab that contains one node and two
ghosts. Correct under the rule set, and useless.

**(c)** One answer under the rules as written (draw it). A fork on whether a one-member box
auto-dissolves — Decision 4.

## 7. A group left with zero members

**(a)** Nothing remembers it. Probed: with every `PassEntry.group` cleared, no entry names
`bloom` anywhere in `PassGraph`. There is no group table by 091 D1.

**(b)** The tab vanishes mid-session. If the maintainer is **inside** that tab when the last
member leaves (deleted the last member, or set its group to `""` from the tab's own context
menu), the active tab has no content and no identity to restore.

**(c)** One answer: fall back to the root tab. Not a fork, but it is a concrete implementation
requirement the rule set does not state: **the active tab key must be revalidated every frame
against the live label set**, because the group can evaporate under it.

## 8. Rename a pass read by the NAME RULE versus by an explicit `PassSource`

**(a)** `rename_pass` calls `Document.rename_pass_sources`, which rewrites **only**
`PassSource` values. An `AutoSource` resolving by name is not touched. Probed:

```
AutoSource 'u_blur' on 'comp', passes {blur,bright,comp}        -> 'blur'
rename blur -> blur2; same AutoSource 'u_blur'                  -> None      (edge GONE)
explicit PassSource('blur') rewritten to PassSource('blur2')    -> 'blur2'   (edge KEPT)
```

Two passes drawn identically on the graph behave oppositely under a rename. This is by design
(072 / 069 D9) and 091 D4 already records the same asymmetry for import.

**(b)** After a rename the edges that were name-rule edges **disappear from the picture with no
other change**. Inside a box, this can convert an internal edge into nothing, which changes the
box's input ports: the reader's sampler now reads nothing, so it becomes an input port that was
not one before. A rename adds ports to a box.

**(c)** Fork — Decision 5: does the graph distinguish a name-rule edge from an explicit one?

## 9. Rename a pass *to* a name some sampler resolves by the name rule

**(a)** The inverse of case 8. Probed:

```
AutoSource 'u_scene' on 'b_blur', after some pass is renamed TO 'scene' -> 'scene'
```

A **new edge appears** from a rename that touched neither endpoint's wiring.

**(b)** A box can gain an input port, an output port, or an internal edge because an unrelated
pass was renamed. A cycle can also appear this way, and nothing checks for one: the pre-plan
cycle guard of fixed item 6 covers drag-wiring only, not `rename_pass`.

**(c)** Fork — Decision 6: does `rename_pass` re-plan and warn when the rename creates a cycle?

## 10. Feedback under rename

**(a)** `u_prev` resolves to the consumer itself in `_auto_source`, so it is name-independent.
Probed: `wired_pass(AutoSource(), "u_prev", "rc_jfa2", {"rc_jfa2"})` → `rc_jfa2`.

**(b)** The `prev` port survives every rename. No issue.

**(c)** No fork. (False trail — listed below.)

## 11. A member whose sampler reads a pass that no longer exists

**(a)** `wired_pass(PassSource("gone"), …)` returns `None` — 065 D3, the half-built-graph rule.
The sampler is **absent from `effective_wiring`**, so the planner, the strip's chips and any
graph built on the wiring all see nothing rather than a broken edge.

**(b)** Under the boundary rule, "reads nothing" **is** an input port ("a member sampler that
reads an outside pass **or nothing**"). So the box shows an unfilled input port — which is the
right answer. But: building that port list requires `sampler_names(render_pass)` from
`document.py`, **not** `effective_wiring`, because the wiring drops unfilled samplers entirely.
Probed: `wired_pass(NoSource(), …)` → `None`, `wired_pass(AutoSource(), "tex", …)` → `None` (no
`u_` prefix, so the name rule says nothing).

This is the most concrete implementation constraint in this review: **the graph's port list has
a different source of truth than its edge list.** Ports come from the compiled program's
samplers; edges come from the wiring. An **uncompiled** pass has `sampler_names == []`, so its
node has no ports at all until it compiles — the same hole 091 D3 and D6 both had to close by
compiling before planning.

**(c)** One answer for the port, one requirement for the data source. No fork.

## 12. A pass leaving a group while an outside pass reads it

**(a)** `set_pass_group(doc, name, "")` — one field write, no wiring change at all. The context
menu's "Leave group" in `pass_list._draw_context_menu` already does this.

**(b)** At the root the pass becomes its own node; the box loses whatever ports that member
contributed. If the departing member was the bundle output, the box's picture changes and an
output port moves from the box onto the new node. Every edge is preserved; only the grouping
moved. This is the cleanest case in the review — the rule set gives exactly one answer and it is
the right one.

**(c)** No fork.

## 13. A non-convex group (a path between two members leaves the group)

**(a)** No code refuses it — `set_pass_group` validates a name, nothing else. Probed:

```
a -> g1 -> mid -> g2 -> out,  group 'grp' = {g1, g2}
box in-ports:  [('g1','u_a','a'), ('g2','u_mid','mid')]
box out-ports: [('mid','u_g1','g1'), ('out','u_g2','g2')]
```

`mid` is both a source into the box and a reader of it.

**(b)** Exactly what fixed item 4 promises: a back edge at the root, `mid` as a ghost on both
sides in the group tab (the mock's `groupView` builds the twin, suffixed `·`). The rule set
handles this without a special case. Confirmed as designed.

**(c)** No fork.

## 14. A wire dragged onto a ghost in a group tab

**(a)** A ghost is a real outside pass drawn dimmed. Fixed item 6 says a drop writes
`PassSource(name)` into the target sampler. Dropping a member's output onto a ghost's port
therefore writes onto a pass that is **not a member of the tab you are in**. Probed:

```
scene -> m1 -> m2, group {m1,m2}, ghost 'scene'
drop m2 onto ghost scene.u_m2:
  cycle m1 -> scene -> m2 -> m1; every pass unplannable; order []
```

The cycle guard of fixed item 6 catches this one (it plans the hypothetical wiring), so the drop
is refused. The mock's own caption says ghost "ports kept so a wire can still be dragged across
the boundary", so this is an intended interaction.

**(b)** Two distinct cases the rule set does not separate: a drop **from** a ghost **into** a
member (fills a box input port — unambiguous, mutates a member), and a drop **from** a member
**into** a ghost (mutates a non-member from inside a group tab). The second is an edit to a pass
the current tab does not contain.

**(c)** Fork — Decision 7: may a drop inside a group tab write onto a ghost?

## 15. A merged port rewritten by one drop (round 3 B, still "leaning")

**(a)** The leaning rule: "a wire dropped on a merged port rewrites every slot behind it". Fixed
item 6's cycle guard plans **the hypothetical wiring**, so it must plan all N rewrites at once.
Probed the case where only one slot would cycle:

```
a -> g1, a -> g2, g1 -> out; 'a' merged behind one port of box {g1,g2}
drop 'out' on the merged port (rewrites g1.u_a=out AND g2.u_a=out):
  cycle g1 -> out -> g1; refused
rewriting only g2.u_a=out would be legal: order ['a','g1','out','g2']
```

**(b)** A merged port is **all-or-nothing**: a legal drop on one slot is refused because a
sibling slot behind the same port would cycle. The maintainer gets a refusal with no way to see
which slot caused it, because the port hides the slots.

**(c)** Fork — Decision 8: this is an argument against merged ports (round 3 B) that round 3 did
not have. Either merged ports keep the all-or-nothing rule and the refusal message names the
offending slot, or merging is display-only and a drop expands to the slot list.

## 16. 091 D6 handover, then the host pass is deleted

**(a)** Probed the full D6 path with the real `plan_import`:

```
host: scene -> grade (output);  source: sc -> bright -> comp
plan_import(substitutions={'sc':'scene'}, handovers=[('grade','u_scene')])
 -> renames {bright: bloom_bright, comp: bloom_comp}
    sources {bloom_bright:{u_sc:'scene'}, bloom_comp:{u_bright:'bloom_bright'}}
    handovers {grade:{u_scene:'bloom_comp'}}
after import: order ['scene','bloom_bright','bloom_comp','grade']
```

Now delete the substituted host pass `scene` — `forget_pass_sources` rewrites
`bloom_bright.u_sc` to `AutoSource("u_sc")`, matching no pass:

```
after: {grade:{u_scene:'bloom_comp'}, bloom_bright:{}, bloom_comp:{u_bright:'bloom_bright'}}
bloom_bright is now an entry point; the box's INPUT port list is []
```

**(b)** The box loses its input port and becomes a source with no input. The bundle still renders
(reading black), so the picture goes dark with no error. `bloom_bright` reverts to what
`plan_import` explicitly materialized it away from: 091 D4 wrote an explicit `PassSource` to
survive the prefix, and the delete undoes exactly that write.

**(c)** One answer under the rules: an input port with no wire. No fork, but note the
asymmetry — an imported group's boundary is the one place the model *did* materialize explicit
wires, and a single delete throws that away without a trace.

## 17. Picking a box as the document output

**(a)** "Picking a box as the output picks the bundle's output" (leaning). But
`PassGraph.output` is a **pass name**, and `set_output_pass` guards `if name not in
document.passes`. Probed: `PassGraph(output="bloom")` where `bloom` is a group and not a pass
constructs fine and `output_pass` returns `None` — a stale output that renders nothing. So the
box pick **must** resolve to a member before the write; the model cannot hold a box as output.

**(b)** After the pick, the accent border is on the box, and `bundleOutput`'s first branch ("the
document output if it is an unread member") then pins the box picture to that member — stable.
But every later mutation re-derives `bundleOutput` from scratch while `graph.output` stays
pinned to the member that was picked, so the two can drift apart.

**(c)** One answer (resolve to the member, write that). No fork.

## 18. A pass named the same as a group

**(a)** `PASS_NAME_RE` and `_GROUP_PATTERN` are the **same** pattern (091 D1: "the name obeys
`PASS_NAME_RE`"). Nothing forbids a pass called `bloom` while a group is called `bloom`. Probed:

```
passes {bloom (ungrouped), b1 (group bloom), b2 (group bloom)}
root nodes+boxes: ['bloom', 'bloom']   -> duplicate key
```

**(b)** The root view has two entities with one name. The mock keys passes by name throughout
(`byName`, `L.nodes[p.n]`, `G[p.n]`), so this collapses two nodes into one: positions, ports and
edges all collide. In the mock's `rootView` the box is pushed into the same `passes` array as
the node, so `byName` returns whichever comes first.

**(c)** Fork — Decision 9: reject the collision in `set_pass_group` / `rename_pass`, or key the
graph by something other than the name.

## 19. Positions under add, delete, rename, import

**(a)** No positions exist today — 072 D9 deleted `PassGraph.layout`. Fixed item 5 adds
`position` to the pass entry. Tracing each verb against that:

- `add_pass` builds `PassEntry()` — the new entry carries the **default** position, so every new
  pass lands at the same spot unless `add_pass` learns the cursor. Fixed item 5 says "a new pass
  lands under the cursor", so `add_pass` gains an argument, and **the copilot's `add_pass` tool
  has no cursor** (`copilot/tools/passes.py`, `caps.add_pass(document, name)`).
- `delete_pass` — `_graph_without` drops the entry and its position with it. Correct.
- `rename_pass` — `_graph_renamed` re-keys the entry, carrying the position. Correct, free.
- `set_pass_target` / `set_pass_iterations` / `set_pass_group` — all go through
  `model_copy(update=…)` on the existing entry, so the position rides along. Correct, free.
- `import_passes` — builds each entry as
  `source_document.graph.passes.get(source_name, PassEntry()).model_copy(update={"group": group})`.
  It therefore copies **the source document's positions verbatim**, which are coordinates in the
  source's canvas and will overlap whatever the host already has there.

**(b)** An imported group lands on top of the host's existing nodes. The rule set says nothing
about where an import goes.

**(c)** Fork — Decision 10: where does an imported group land?

## 20. Undo — which mutations are one-way

**(a)** The app has no undo. Auditing the verbs for reversibility:

| mutation | reversible by hand? |
|---|---|
| `set_pass_group` | yes — retype the label |
| `set_output_pass` | yes |
| `set_pass_target` / `set_pass_iterations` | yes |
| `rename_pass` | yes, **except** name-rule edges (case 8): renaming back restores them, but any edge severed in between and re-resolved is not tracked |
| `set_sampler_source` | yes, unless it overwrote a bound texture — `try_to_release(values.get(uniform))` **frees it** |
| `delete_pass` | **no** — the file is deleted, and every `PassSource` naming it across the document is flattened to `AutoSource` by `forget_pass_sources`; re-adding the pass does not restore those wires |
| `import_passes` | **no** — no inverse verb exists; undoing it is N deletes plus re-wiring every handover |

**(b)** The graph makes two of these one-click: Dissolve (safe, reversible) and delete-node
(destructive). The brainstorm gives the delete no confirm step; the strip has one
(`preview_cell`'s `delete_armed` → `delete_confirmed` two-click arm, `pass_list._delete_pass`).

**(c)** Fork — Decision 11: what is the confirm step on the canvas, and is there one for
deleting a **box** (N passes at once, the single most destructive gesture the graph would add)?

---

## Decisions the maintainer must make

**1. May the root box look unchanged after a member was deleted?** (case 1) A destructive edit
inside a group produces an identical box at the root. Options: (a) accept it — the box is its
boundary and the boundary did not change; (b) a member whose output nothing reads gets a marker
that propagates to the box (a "dead member" dot on the box corner); (c) the box shows a count of
unwired input ports, so a severed interior reads as `bloom · 1 unfilled`.

**2. Is the bundle output always an output port?** (case 2) Options: (a) yes — the box always has
one output dot, even when nothing outside reads it (matches the fixed rule's wording "plus the
bundle's own output"); (b) only when something outside reads it — a terminal box has no output
dot, which is honest but leaves the box's picture unexplained; (c) yes, and draw it differently
when unread (hollow dot).

**3. What does a group split across the DAG draw as?** (case 5) Options: (a) one box at the
members' bounding box, swallowing unrelated nodes inside it visually; (b) one box per connected
component of the members, with the name repeated (the strip's `group_runs` answer, but it
breaks "a group is one box"); (c) one box, and the layout pass is allowed to move non-members
out of its rectangle.

**4. Does a one-member group stay a box?** (case 6) Options: (a) yes, uniform rule, no special
case; (b) a one-member group draws as a plain node with the group tint, no tab; (c) leaving the
second-to-last member offers to dissolve.

**5. Does the graph distinguish a name-rule edge from an explicit `PassSource`?** (cases 8, 9)
This is the difference between an edge that survives a rename and one that does not. Options:
(a) no — an edge is an edge, and the rename surprise is accepted (matches the strip, which shows
one chip either way); (b) a name-rule edge draws dashed or thinner, so the maintainer can see
which wires are fragile; (c) dragging a wire onto an already-name-wired sampler materializes it
to an explicit `PassSource` (fixed item 6 already writes one, so this is free for any edge the
maintainer touches).

**6. Does `rename_pass` check for the cycle a rename can create?** (case 9) Fixed item 6 guards
drag-wiring with a hypothetical plan; the same hazard exists on rename and is unguarded today.
Options: (a) leave it — the renderer's cycle fallback handles it and the graph draws the error;
(b) `rename_pass` plans the post-rename wiring and rejects, matching the drag guard; (c) it
plans and warns but proceeds.

**7. May a drop inside a group tab write onto a ghost?** (case 14) Options: (a) yes — a ghost's
ports are live, the edit is honest, and the cycle guard already covers the dangerous case;
(b) no — a ghost is read-only, and a wire into the group is dragged at the root instead; (c) yes
from a ghost into a member (filling a box input), no from a member into a ghost.

**8. Merged ports: all-or-nothing, or expand?** (case 15) The probe shows a merged port can
refuse a drop that is legal for one of its slots. Options: (a) keep the merge and the
all-or-nothing rewrite, with the refusal naming the offending slot; (b) merged ports are
display-only: a drop expands the port and the maintainer picks the slot; (c) drop the merge,
one port per slot (round 3 A).

**9. May a pass and a group share a name?** (case 18) The two patterns are identical and nothing
checks. Options: (a) reject in `set_pass_group` and `rename_pass` — one namespace, one message;
(b) allow, and key the graph by `(kind, name)` instead of the name; (c) allow, and let the box
win (the mock's current behaviour, which is an accident).

**10. Where does an imported group land on the canvas?** (case 19) `import_passes` copies the
source's entries verbatim, so positions come along from another document. Options: (a) strip
positions on import and run the rank layout for the new passes only, packed to the right of the
host's bounding box; (b) keep the source's relative layout and translate the whole cluster to a
free spot; (c) land it under the cursor like a new pass, with the source's relative layout kept.

**11. What is the confirm step for a destructive canvas gesture?** (case 20) Options: (a) the
strip's two-click arm, per node, and a box delete is not offered at all — you delete members
from inside the tab; (b) the two-click arm on a node and a modal naming the N members for a box;
(c) no box delete verb, only Dissolve (which is reversible) plus per-node delete.

---

## False trails

- **Feedback under rename.** `u_prev` resolves through `_auto_source`'s consumer branch, not by
  pass name, so a `prev` port survives every rename with no work. Probed, non-issue.
- **The cycle guard being insufficient for drag-wiring.** `plan_passes` reports every pass on the
  loop *and* every pass downstream of it, and returns an empty order for the lot, so a
  hypothetical plan detects the cycle regardless of where the drop lands. Fixed item 6's rule is
  sound as written. (It is the mutations that *do not* go through it — rename, case 9 — that are
  the hole.)
- **Non-convex groups needing a special case.** Probed the leaking-pass shape; the boundary rule
  produces a back edge at the root and a twin ghost in the tab with no extra rule. Fixed item 4's
  claim holds exactly as stated.
- **`set_pass_group` corrupting the wiring.** It writes one string field and nothing else; every
  edge is untouched. Grouping and ungrouping are pure view changes over the same DAG. This is the
  property the whole feature rests on and it is real.
- **Positions surviving the existing verbs.** Every `PassGraph` mutation except `_graph_without`
  goes through `model_copy(update=…)` or carries the entry object, so a new `position` field
  rides them with no edit — the same argument 091 D1 made for `group` and it checks out. Only
  `add_pass` (builds a fresh `PassEntry()`) and `import_passes` (copies the source's entry) need
  work.
- **`GraphError` on a read of a missing pass.** The module docstring of `pass_graph` mentions "a
  read of a pass that does not exist" as a `GraphError`, but `plan_passes` never emits one:
  `wired_pass` returns `None` first, so the read is absent from the wiring entirely and the
  planner has nothing to complain about. Every error the planner emits is a cycle or a
  descendant of one. The graph view cannot get a missing-read error from the planner and must
  derive it from `sampler_names` versus the wiring (case 11).

---

## Verdict

The boundary-edge rule set is **complete for every mutation that moves a wire or a label, and
incomplete for deletion**: case 2 breaks it — deleting the member that is the bundle output
leaves a box whose picture comes from a member that is not an output port, so the rule's own two
clauses ("a member read from outside" and "plus the bundle's own output") disagree about whether
the box has an output dot.
