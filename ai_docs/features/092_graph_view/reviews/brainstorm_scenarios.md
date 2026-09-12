# 092 brainstorm review — the user scenarios, end to end

Angle: walk each scenario as a sequence of actions on the surfaces this product actually has, and
compare each departure from node-editor precedent. Anchors: `01_brainstorm.md` (items 1-8 under
*Fixed by the maintainer* are treated as settled and are never re-litigated below), `00_mock.html`
round 3, `091_presets/01_spec.md`, `070_pass_reads/01_spec.md`, `069/00_findings.md` #19,
`conventions.md ## Design decisions`, and the code (`pass_graph.py`, `project_session.py`,
`document.py`, `widgets/pass_list.py`, `widgets/uniform.py`, `editor_types.py`,
`copilot/tools/passes.py`, `help_content.py`).

The surfaces the verbs can land on today, named once so the sequences below can be terse:

- **the strip** — `widgets/pass_list.py`: click a tile (sets the output AND opens the pass in the
  editor), the gear overlay, the context menu (Settings / Delete / Leave group), `add pass`,
  `import…`.
- **the settings modal** — `popups/pass_settings.py`: name, group (with a combo of existing
  groups), target, runs.
- **the sampler row** — `widgets/uniform.py`, in the Uniforms tab: a `grouped_combo` per
  `sampler2D` writing `PassSource` / `NoSource` through `session.set_sampler_source`. This is the
  only place wiring is edited today.
- **the import dialog** — `popups/import_passes.py`: two tabs (This project / Examples), a card
  grid, the group name, one combo per entry point, the handover checkboxes.
- **the editor tabs** — `tabs/code.py::_draw_tab_row` over `EditorTab(path: Path, kind, document_id)`.
- **the copilot** — `add_pass` / `set_pass` (which learned `group` in 091 D9) / `delete_pass`.
- **the session verbs** — `add_pass`, `delete_pass`, `rename_pass`, `set_output_pass`,
  `set_sampler_source`, `set_pass_target`, `set_pass_group`, `import_passes`,
  `set_pass_iterations`. There is **no `duplicate_pass`** (only `duplicate_document`).

One corpus fact governs half of what follows, so it is stated once: of the six shipped examples,
**five are single-pass**, and four of those five (`Fire`, `UV Mango`, `Night City`,
`Text Rendering`) declare **zero `sampler2D` uniforms**. The fifth (`Media Input`) declares two,
both bound to files, neither a pass read. Both `projects/dev/` documents are single-pass `main`.
The only multi-pass documents in the repo are Radiance Cascades (6 passes, 5 ranks) and the
`tests/fixtures/bloom_chain/` fixture (5 passes, 4 ranks).

---

## The maintainer's scenarios

### 1. Import a prebuilt node (a document) — COVERED

1. Strip → `import…` (or the palette's `IMPORT_PASSES`). → the import dialog.
2. Examples tab → the Radiance Cascades card. The dialog compiles the source's and the host's
   program-less passes and derives `entry_points`.
3. The group field prefills `radiance` from `group_slug`.
4. One row: `paint ← [theirs (copy it) | main | …]`. Pick `keep` or a host pass; if fed, the
   handover checkboxes appear.
5. `Import 6 passes`. → six entries carrying `group="radiance"`, six files under `passes/`.
6. The strip draws a flush outline around each run of consecutive members.
7. **The graph view's contribution:** open the graph tab; the six members are one box with
   whatever boundary ports the import left, at the bounding box of six positions the first-sight
   rank layout just wrote.

Verdict COVERED. Nothing in the sequence needs a new surface, which is exactly the brainstorm's
item 8 ("groups earn their place only because import makes them") holding up. Worth stating
plainly: **this is the one scenario where the graph view pays for itself with no new verbs at
all** — the box already exists in the data the moment the dialog closes.

### 2. "Do I want to modify them? Maybe? Maybe not." — COVERED, with one asymmetry

The design's answer is that you never have to decide: the box is one node until you enter it, and
entering it is non-destructive. That is right, and it matches every precedent (see the table).
The asymmetry: **the root tab already lets you modify the bundle without entering it**, because
fixed item 6 makes a port a drop target and round 3 B lets a wire dropped on a merged port rewrite
every slot behind it. So "I don't want to modify them" is not a mode the user is in; it is a
description of what they happen not to have done. No gap — but it means the "black box" framing
should not be promised anywhere in the UI (a tooltip, the Help panel), because the box is not
sealed.

### 3. RMB on the composite node → open it in a separate tab — DECISION

1. Right-click the box → `Open` (or double-click). Brainstorm item 3.
2. The group's tab draws members as nodes and the outside passes they touch as ghosts.
3. Escape goes up.

Two things in that sequence have no surface, and they are separable.

**3a. Where the tab lives — DECISION.** The brainstorm leans toward
`EditorTab.kind = "graph"`, keyed by a group path. `EditorTab` is
`@dataclass(frozen=True)` with `path: Path`, and `path` is the `EditorSession` key and the imgui
`##id` — a group path is not a file and has no `EditorSession`. Also `tab_label` derives the label
from the document name, and `kind` selects the error tint. Options:

- **(i) A synthetic `Path`** (`document_dir/graph/<group>`). Cheapest; costs a `Path` that names
  nothing on disk, which the file-path row in the bottom bar and `Open dir` both read.
- **(ii) Widen `EditorTab` to `path: Path | str`** and let a graph tab carry the group path.
  Honest; costs a union every consumer of `path` has to answer, and `editor_types.py` is a leaf
  every tab site imports.
- **(iii) The graph's own tab strip inside the graph pane** (`ui_primitives.text_tab_row`, which
  the Uniforms tab already uses), with ONE `EditorTab` for the whole graph view. The group tabs
  are then the graph widget's own state, not the editor's. This is what the mock draws (`.gtab`),
  it is what Nuke does, and it keeps `EditorTab` untouched.

(iii) is the one that matches the mock and costs nothing structurally; name it in the spec rather
than leaving it to implementation.

**3b. Escape — DECISION.** `hotkeys.py::_handle_escape` is already a contended funnel: the
`IMPORT_PASSES` branch, the other popups, open name inputs, the vim layer, and a glfw-level
filter that swallows a jobless Escape (`App._install_escape_filter`). Adding "go up one group
level" makes Escape's meaning depend on which pane has focus, and the graph canvas has no focus
notion (019's regions were removed by 069 W-E). Options: Escape only while the graph pane is
hovered; a breadcrumb click as the only way up (no chord); or a dedicated chord in the Alt tier.
The mock's breadcrumb already draws the way up, so the chord may be unnecessary.

### 4. Group already-existing passes, "for the sake of refactoring" — COVERED, one real hole

1. Rubber-band or shift-click several nodes on the canvas.
2. Right-click → `Group`, type a name.
3. `set_pass_group` runs once per selected pass. The label lands; the box appears at the next
   frame's bounding box.

Covered by item 8, and a new selection state is the only new machinery. The hole is **undo**:
grouping seven passes is seven independent saved writes, and the app has no undo outside the
editor buffer. Mis-grouping is repaired by `Leave group` on each tile, one at a time, from the
strip's context menu. Minimum: the canvas context menu needs `Dissolve` (brainstorm already lists
it under *Leaning*) so a wrong group is one action to undo rather than N. **Promote Dissolve out
of "leaning" — it is the inverse of the only new verb the feature adds.**

Second, smaller: after `Group`, the pass names are unchanged (item 8 is explicit that grouping
does not rename, and 091's prefix was an import-time collision guard). So an interactively-made
group's members are `bright`, `blur`, `trail` while an imported group's are `bloom_bright`,
`bloom_blur`. Both are correct; the spec should say so once, because the box's merged-port names
(round 3 A's `member.sampler`) read very differently in the two cases.

### 5. Could the groups be nested? — DECISION

The brainstorm's *leaning* is a path label (`post/bloom`). Walked as actions:

1. Select three members of `post` → `Group` → type `bloom`.
2. Their `group` becomes `post/bloom`; `PassEntry.group`'s pattern (`_GROUP_PATTERN`) must admit
   `/`, which it does not today.
3. The root shows one box per top-level segment; `post`'s tab shows its direct passes plus a
   `bloom` box; boundary ports compute identically at each depth.
4. The strip's `group_runs` cuts by adjacency on the **full path**, so `post/bloom` and
   `post/grade` are two different groups and two different outlines — the strip shows no nesting
   at all, just two labels. That is a real cost: the two views disagree about what a group is.

Options:

- **(i) Stay flat (091's deferral stands).** The scenario "add an already-grouped node to my
  current group" then resolves by the import-time rule 091 already has: inner labels are dropped.
  One depth, one box, the strip and the graph agree.
- **(ii) Path labels, graph-only nesting.** The graph nests; the strip draws one outline per
  distinct full path. Cheap in the model (one regex change), and the disagreement above is the
  price.
- **(iii) Path labels with the strip outlining by top-level segment.** The two views agree; costs
  `group_runs` a second grouping key and the strip a nested outline, which is the shape 091
  rejected for the strip.

The scenario set below does not produce a case that needs (ii) or (iii). The one that comes
closest is #6.

### 6. "What if I add an already-existing grouped node to my current group?" — GAP

This is the scenario with no surface at all today, in two distinct forms.

**6a. Import a source that itself carries groups.** 091 drops the inner labels
(`Out of scope: Nested groups`). So: import a document whose passes carry `bloom`, under group
`post` → every copied pass gets `group="post"` and the inner `bloom` is gone. The user who
built the source as two bundles loses that structure silently. The dialog says nothing.
**GAP: the import dialog has no line saying the source's own groups are being flattened.** That
is a one-line fix to `popups/import_passes.py` (the modal already carries a "the script is not
imported" note, so the surface exists), and it is worth landing whether or not nesting ever does —
today the flattening is invisible.

**6b. Drag an existing group's box into another group on the canvas.** No verb. `set_pass_group`
takes one pass and one label; there is no "reparent these N passes" operation, and with flat
labels the result would be indistinguishable from dissolving the inner group into the outer one.
**GAP: no reparent verb.** Under option 5(i) the honest answer is that this action does not exist
and the box should not be a drop target for another box; under 5(ii) it is a string prefix
rewrite across the members. Decide 5 first; 6b follows from it.

### 7. "Am I missing anything?" — the group I/O question the brainstorm did not put as a question

Fixed item 4 derives a box's ports from boundary edges. Blender, Houdini and Unreal all put
explicit input/output **nodes inside** the group and the group's ports are those nodes'. This is
the single largest departure in the design and it deserves being stated as what it buys and costs,
per the two users:

- **For the user who never opens the box** (scenario 2, the common one): derived ports are
  strictly better. There is nothing to author, nothing to keep in sync, and a bundle imported from
  a document that never anticipated being a group still has a sensible interface. Blender's
  equivalent — `Ctrl+G` on a selection — has to *auto-detect* boundary sockets and build the
  Group Input node from them, which is exactly this computation, done once and then frozen as
  data. Deriving it every frame is the same answer without the staleness.
- **For the user who opens it** (scenario 3): derived ports are worse in one specific way. Inside
  a Blender group, the Group Input node is a first-class node you can drag a wire from — you can
  give the group a new input by dragging from the Group Input's empty socket. The ghosts in this
  design are dimmed, dashed, **not editable** (mock round 3 C says so, though it keeps their ports
  "so a wire can still be dragged across the boundary"). So from inside the group there is no way
  to say "this group should take one more input"; you say it by wiring a member's sampler to
  something outside, and the port appears. That is discoverable only by trying it.

There is also a concrete inconsistency to settle. Item 4 says "a member read from outside, **plus
the bundle's own output**, is an output port" — plural is possible. The mock's `bundleOutput()`
returns exactly **one** name, with a tiebreak (`unread[unread.length - 1]`) when several members
are unread from outside. So a group with two unread members silently shows one output port and the
other member's picture never becomes the box's. **DECISION: one output port (which one, when
several members qualify?) or N.** N is the honest answer and matches every precedent; the box's
*picture* still needs one choice, and "the member that is the document output if one is, else the
first in strip order" is a rule the user can predict.

---

## The scenarios he did not name

### 8. Build an effect from scratch on the canvas, no shader written — GAP (the deepest one)

1. Graph tab on a fresh document. One node, `main`, zero ports (four of six examples are exactly
   this).
2. Right-click the canvas → `Add pass` → it lands under the cursor, unconnected, dashed until it
   compiles (item 5 says so). `session.add_pass` writes `PASS_STUB`:

   ```
   #version 460 core
   in vec2 vs_uv;
   out vec4 fs_color;
   void main() { fs_color = vec4(0.0, 0.0, 0.0, 1.0); }
   ```

3. The new node has **zero input ports**, because it declares no `sampler2D`.
4. To wire anything into it the user must open its shader, type
   `uniform sampler2D u_main;`, and save. Only then does a port exist to drop a wire on.

So the canvas cannot build a chain. Every other node editor in the comparison lets you drop a node
and wire it immediately, because a node's ports are declared by its *type*. Here **a pass is a
shader file and its ports are text inside it** — which is the product's whole thesis, not a
defect. But it means the graph view is a view of wiring that already exists, plus a faster way to
re-point it; it is not a construction surface. Two honest options:

- **(i) Say so.** The canvas's `Add pass` opens the new pass's shader immediately (which is what
  069 #28 already decided for the strip's add: activate, then the gear). The user types a sampler
  and the port appears. No new model.
- **(ii) The drop creates the sampler.** Dragging from a node's output dot onto *empty space* on
  another node (not onto a port) appends `uniform sampler2D u_<source>;` to that pass's shader and
  writes the `PassSource`. This is the only way the canvas becomes a build surface, and it makes
  the graph view write GLSL — a large new power, and a large new failure mode (where in the file?
  what if the pass does not compile?).

Recommend (i) and say it in the spec, because (ii) is a different feature wearing this one's
clothes. Either way the spec should state the rule once: **a port exists because a sampler is
declared in the shader.** The Help panel's `Passes` section already teaches the naming rule
(`u_blur` reads `blur`) and is the place for it.

### 9. A tiny document — the 1-pass case — DECISION

Five of six shipped examples and both dev-sandbox documents are one pass. The graph tab on any of
them shows one node, usually with zero ports, at whatever position the rank layout wrote.

`Media Input` is the interesting one: two `sampler2D`s, both bound to files. `effective_wiring`
answers `{}` for it (a media-bound value is not a `PassSource`, and `wired_pass` returns `None`),
so a wiring-derived node has **no ports** while the sampler rows in the Uniforms tab show two.
The graph and the panel disagree about how many inputs the pass has.

DECISION, three parts:
- **Does the graph tab open at all for a single-pass document?** Options: always (consistent, and
  mostly useless); only when `len(passes) > 1`; always, but it is never the tab that opens by
  default. The third is the one that costs nothing and surprises nobody.
- **Does a media-bound sampler get a port?** Options: no port (wiring-derived, consistent with
  `_reads` and the strip's chips, which also show nothing for a media bind); an unfilled port
  (consistent with the panel, and a drop target for re-pointing it at a pass — which would
  silently discard the bound texture); a port with the media thumbnail. 070 already decided the
  strip's chips show only pass reads, so "no port" is the decision that keeps the two views
  consistent; if ports are shown, the wire-drop-discards-a-texture case needs an answer.
- Note this is also the first case where "one port per sampler slot" (round 3 A) and the wiring
  disagree, independent of groups.

### 10. A big document — 20+ passes — COVERED by intent, unmeasured

Zoom-to-fit is fixed item 1's whole justification, and the mock's layout engine measures natural
width per document. But the largest document that exists is six passes, and the mock's stats table
is computed over those two. The brainstorm's own note — "Depth, not fan-out, is what the picture
has to absorb" — is a measurement over a corpus of two. No gap in the design; a gap in the
evidence. If the spec wants a number, the cheap probe is to generate a 20-pass chain into
`projects/_lab/` and measure natural width, rather than reasoning about it.

### 11. Duplicating a box — GAP

1. Right-click a box → `Duplicate`. **No such verb.** `duplicate_document` exists;
   `duplicate_pass` does not.
2. Today the user re-runs the import dialog against the same source (scenario 12) or hand-copies
   files.

This is a real hole and it is not group-specific: duplicating a single *pass* has no verb either.
It is out of 092's scope, but the box makes the absence conspicuous — a bundle is exactly the
thing a user wants two of (two blurs at different radii). Name it as out of scope with a trigger,
or add `duplicate_pass` as the primitive and let a box duplicate be N of them plus a group rename.

### 12. Importing the same preset twice — GAP (a rejection the user cannot act on)

1. Import the bloom fixture under group `bloom`. Members `bloom_scene` … `bloom_composite`.
2. Import it again. The group field prefills `bloom` again (`group_slug` is deterministic).
3. `plan_import` rejects: "a copied name already among the host's passes."
4. The user retypes the group as `bloom2`. Import succeeds. Members `bloom2_bright` etc.

The rejection is correct and the dialog shows it live (091 D10 recomputes the plan every frame).
But the user has to *guess* that the group name is the fix — the message names the colliding pass,
not the field to change. Small, and squarely in 091's surface rather than 092's, but it is a
scenario the graph view makes common: two bloom boxes side by side is the obvious thing to want.
**Minimum fix in 092's spec or as a 091 follow-up: when the rejection is a name collision and the
group name is the prefix, say "rename the group".** (The prefill could also disambiguate, but a
silent `bloom2` prefill would be worse — the user would not notice they made a second group.)

### 13. A preset that expects a scene and gets a generator — DECISION (a type question the ports hide)

1. Import the bloom fixture, feed its entry point `scene` with the host's `paint`.
2. `paint` is `f2` at scale 1.0; the bloom members are `f2` at scale 0.5 except `composite` (`f1`).
3. Nothing checks anything. Every port is a `sampler2D`; every wire is legal.

The brainstorm names this under corner cases ("target dtype / scale mismatch between producer and
consumer"). Walked as a scenario, the answer is that **there is no wrong connection in this
model** — which is a genuine difference from Substance Designer (typed color/grayscale ports that
refuse) and from Unreal (typed float/vector pins). What can be wrong is *quality*: feeding an `f1`
(0-1 clamped) pass into a bloom that needs values above 1 produces a dull result and no error
anywhere. Options: no signal (consistent with the engine's "the author knows" posture, which
`PassEntry.iterations`' comment states explicitly); a dim annotation on the edge when the producer
is `f1` and the consumer is `f2`/`f4`; a port color by dtype. The second is the only one that
costs nothing and says something true. This decision belongs in the spec because it is the
closest thing this design has to a type system, and the answer "we have none, deliberately" should
be written down rather than left to be re-asked.

### 14. Swapping the wiring of two boxes — COVERED

1. Drag from `a`'s output dot onto `c`'s port that currently reads `b`. The drop writes
   `PassSource("a")` over the old value — the same write the sampler row makes.
2. Drag from `b`'s output onto the port that read `a`.

Covered by item 6, and the cycle refusal (plan the hypothetical wiring first) covers the case
where the swap would close a loop. One thing to state: **the drop overwrites, it does not ask.**
The sampler row's combo has the same semantics, so this is consistent; worth one sentence in the
spec because a drag feels more destructive than a combo pick and there is no undo.

### 15. The 068 tutorial walked in the graph instead of the strip — FALSE TRAIL, mostly (see below)

Kept here rather than in False trails because one finding survives: the tutorial's per-pass card
(069 #31's template: name · reads · size · format · smooth · repeat · runs) is a **pass** card,
and every field on it except `reads` is edited in the settings modal. The graph view carries only
`reads`. So a tutorial walked in the graph would send the reader to the gear for five of six
fields on every step. That is not an argument against the graph view; it is an argument that the
node's context menu must reach `Settings` (which item 3's right-click menu does — the mock's node
menu draws it). Confirm that is in the spec; otherwise the graph is a view you have to leave for
every parameter.

### 16. The copilot builds a chain while the user watches the canvas — GAP (small, real)

1. The user types "add a bloom chain" in the chat.
2. `copilot_turn_active` goes true. The strip's `add pass` / `import…` are `begin_disabled`, and
   `open_import_passes` refuses through `_copilot_busy_blocked`.
3. The copilot calls `add_pass` three times and `set_pass` to group them.
4. Each new pass has **no `position`** — item 5 says a new pass lands under the cursor, and there
   is no cursor. The canvas must do something.

**GAP: the position rule has no branch for a pass created without a cursor.** Three creators exist
today: the strip's `add pass`, the copilot's `add_pass`, and `import_passes` (six at once). Only
the canvas one has a cursor. Options: passes with no position are laid out by the rank layout at
the next sight (which is also the first-sight rule, so it is one code path, not two); or the
canvas places them in a free column to the right. The first is simpler and already specified for
first sight — just say it covers "any pass without a position", not only "a document seen for the
first time".

Also: the graph canvas needs the same `copilot_turn_active` freeze the strip has, or the user
drags a node while the copilot deletes it. One line, easy to forget.

### 17. Keyboard-only use — DECISION

Today `Alt+Left` / `Alt+Right` (`NEXT_PASS` / `PREV_PASS` → `step_output_pass`) walk passes in
**strip order**, which is `strip_order(names, wiring)` — a stable topological order, not a spatial
one. In a graph where the user has dragged nodes freely, "next" means the next topologically, not
the next to the right. That is defensible (it is what the strip means) but it should be a
deliberate answer, not an accident: either the chords keep their meaning in the graph, or the
graph gets no keyboard traversal at all and is honestly a mouse surface. Note that 069 W-E removed
the app-wide keyboard nav regions (019), so there is no focus machinery to hang a spatial
traversal on. Recommend: the existing chords keep working unchanged, the graph highlights the
current pass, and no new keyboard verbs. Say it, so the absence reads as a decision.

### 18. The shader-lab workflow producing a preset — FALSE TRAIL for 092, one line for the spec

The `/shader-lab` skill builds each step as a **new document** in `projects/_lab/<slug>/`, and
promotes a result by hand (move the document dir, or copy it into `document_examples/`). A lab
document therefore arrives as a document, which the import dialog's "This project" tab already
lists — provided the lab project is the open project. A cross-project presets folder is 091's
named deferral with its own trigger ("the maintainer wants a bundle in a second project"), and the
lab workflow is precisely the thing that fires that trigger, since `projects/_lab/` is a different
project from `projects/dev/`. That is worth one sentence in 092's spec as *not this feature's
problem*, with the pointer — it does not change the graph view's design at all.

### 19. Save a group as a document (the export half) — DECISION, and it is where boundary ports bite

091 defers it; the brainstorm lists it under *Leaning*. Walked:

1. Right-click a box → `Save as document`.
2. The members' files and entries are written into a new document dir.
3. **What happens to a member sampler that read an outside pass?** In the host it is a
   `PassSource("scene")` naming a pass the new document does not have. 091's out-of-scope entry
   says "boundary samplers as explicit black rows", i.e. `NoSource`.
4. The new document is then re-importable, and `entry_points` on it answers exactly the passes
   whose boundary samplers were blacked out — so the round trip closes: what was an input port
   becomes an entry point, and the import dialog asks about it again.

That round trip is the strongest argument in the whole design for derived boundary ports: **the
input port and the entry point are the same concept computed from the same wiring**, once at the
box and once at the dialog. It is worth writing down explicitly, because it is the thing the
explicit-group-I/O-node precedent would break — Blender's Group Input node is a fourth
representation of the same fact. If the export half is ever built, the derived design gets it
nearly free; an explicit-ports design would have to author the I/O nodes on export and consume
them on import.

One decision inside it: a member whose boundary sampler is blacked out loses a *bound texture*
too, if the sampler held media rather than a pass. Say what happens (copy the media, as
`import_passes` does through `document_dir_of`, or drop it).

---

## Prior art, where it decides something here

| Design choice (092) | Precedent | What it does there | What it buys or costs here |
|---|---|---|---|
| Ports derived from boundary edges, no explicit group I/O nodes | **Blender** node groups | `Ctrl+G` auto-detects boundary sockets and materializes a Group Input / Group Output node **inside** the group; from then on the interface is authored data you can reorder, rename, add to, and give defaults | Buys: nothing to author, nothing to keep in sync, an imported bundle that never anticipated being a group still has an interface; and the export round trip (#19) reuses `entry_points` instead of inventing a fourth representation. Costs: from inside the group there is no handle to say "this group takes one more input" — you say it by wiring a member outward and the port appears. No port ordering, no port renaming, no per-port default. |
| Same | **Houdini** subnets | Explicit `subinput`/`suboutput` nodes inside; the subnet's ports are theirs, indexed | Same trade. Houdini's indexing is what lets a subnet's interface be stable while its innards change; derived ports here re-derive on every edit, so a port can appear or vanish as a member's shader is edited. That is honest but means the box's shape is not stable across a shader edit — worth stating. |
| Same | **Unreal** material functions | Explicit `FunctionInput` / `FunctionOutput` nodes, each with a name, a type, a sort priority and a default | The sort priority is the thing derived ports cannot express: here port order falls out of member order and sampler order. For a six-slot bundle that is arbitrary. Low cost at this size; a real one if bundles get large. |
| Same | **Substance Designer** | Explicit input/output nodes; inputs are typed (color/grayscale) and a wrong connection is refused | This design refuses nothing (#13): every port is a `sampler2D`. That is the correct call for a product where a pass is a shader file, but it means the ports carry no information beyond "a texture goes here". |
| Tabs instead of folding; the root always shows boxes | **Nuke** groups | Double-click a Group opens its contents in a **separate tab** of the node graph pane; the group node stays one node in the parent graph, never expands in place | Direct precedent, and the closest match to this product. Buys: kills 091's convexity rule outright, because the box is never a node the planner orders — a non-convex group just draws a back edge (mock round 3 D). Costs: nothing found. This is the strongest borrowed decision in the design. |
| Same | **TouchDesigner** COMPs | Enter a COMP and the whole network view *replaces* itself; a breadcrumb path bar is the way up and the way back | The breadcrumb, not a tab strip, is TD's answer — and it is what nesting (#5) would force here too. If nesting is ever adopted, the tab strip becomes a breadcrumb, which is a UI change the spec should name in advance rather than discover. |
| Same | **Blender**, **Houdini** | Tab enters the group *in place* — the editor's context changes, the layout does not open a second tab | The in-place version needs a strong "you are inside X" signal (Blender paints a header path, Houdini a path bar). The tab version gets that signal for free from the tab itself, which is why Nuke's answer is the right one for a product whose editor pane already has a tab bar. |
| No fold (a group never collapses in the parent view) | **Unreal** collapsed graphs | A collapsed graph is a node you can also *expand back* in place, restoring the members to the parent graph | Buys: 091's convexity rule stays dead, and there is no second representation to keep consistent. Costs: at the root you cannot see a group's innards *in context* — you either see the box or you switch tabs and lose the surroundings. Every precedent that offers expand-in-place pays for it with the convexity problem, so this is a good trade; but note the ghosts exist exactly to buy back some of that lost context. |
| A group is a label, not an entity | **All five** | Every one of them has a group/subnet/function *node* — a real object with a name, a position, a comment, and an interface | Buys: `PassEntry.group` and nothing else, no member list to keep in step, and `conventions.md`'s own revisit trigger stays unfired. Costs: a group cannot carry a fact of its own. The two facts this scenario walk found that want to live on the group are the **box's collapsed/expanded-tab state** (not needed — item 3 removes it) and the **box's own position** (item 5 removes it by making the box the bounding box). So the label survives this scenario set intact — see the closing line. |
| Positions per pass, the box at the bounding box | **All five** | The group node has its own position; its members have positions in their own inner space, independent of the parent layout | Buys: no group entity (above), and dragging a box is N translations of real data. Costs: two real ones. (a) The box's size is not the user's to set — a group with one far-flung member draws a huge box, and the only fix is to move the member. (b) Members share one coordinate space with non-members, so the group tab's layout and the root tab's layout are the same positions seen twice; a layout that reads well at the root may read badly inside the group and there is no second set of positions to fix it with. Both are acceptable at six passes; both get worse with size. |
| Ghosts at the border, not editable | **Blender** Group Input node | Is a real node: draggable, and its empty socket is a live drag source for adding an input | The ghost is strictly a viewer. Since a ghost's port is still a drop target (mock round 3 C), a ghost is half-editable already — the spec should say which half, because "not editable" and "a wire can be dragged across the boundary" are in tension in the mock's own caption. |

---

## What the maintainer is missing

Ranked by how much each changes the spec.

1. **The canvas cannot build a chain, because a pass's ports are text in its shader file**
   (scenario 8). `PASS_STUB` declares no sampler, so a new node has zero input ports until the
   user opens the shader and types one. Every precedent lets you wire a freshly dropped node
   immediately. This is not a defect — it is what "a pass IS a shader file" means — but the spec
   must say which of the two answers it takes (add-pass opens the shader, or a drop authors the
   sampler declaration), and the Help panel's `Passes` section is where the rule gets written for
   users. Changes the framing of the whole feature: it is a *view of* wiring plus a re-pointing
   tool, not a construction surface.

2. **Four of six shipped examples have zero sampler2D uniforms; five of six are single-pass**
   (scenario 9). The graph tab on most of what a user opens is one node with no ports. The spec
   needs the open-or-not rule and the media-bound-sampler rule, and the feature's value should be
   stated against the corpus that exists (two multi-pass documents in the whole repo) rather than
   against an imagined one.

3. **The box's output-port count is inconsistent between the brainstorm and the mock**
   (scenario 7). Item 4's prose admits several output ports; `bundleOutput()` in the mock returns
   exactly one, with a last-element tiebreak. Pick N, and give the box's *picture* a predictable
   rule.

4. **Nothing tells the user their imported groups were flattened** (scenario 6a). 091 drops a
   source's inner labels silently. One line in `popups/import_passes.py`, worth landing whether or
   not nesting happens.

5. **A pass created without a cursor has no position** (scenario 16). Three of the four creation
   paths have no cursor (copilot `add_pass`, `import_passes`, and the strip's `add pass`). Extend
   the first-sight rank layout to cover "any pass with no position" rather than only a
   first-seen document. Also: the canvas needs the strip's `copilot_turn_active` freeze.

6. **`Dissolve` is the undo for the only new verb the feature adds** (scenario 4). It sits under
   *Leaning*; grouping is N independent saved writes with no undo, so the inverse is not optional.

7. **Escape is already a contended funnel** (scenario 3b). "Escape goes up" needs a focus story
   the app does not have since 019 was removed, or the breadcrumb becomes the only way up.

8. **`EditorTab.path` is a `Path` and a group path is not a file** (scenario 3a). The
   graph-as-a-third-tab-kind lean collides with the dataclass, the `EditorSession` key and the
   `##id`. The graph's own inner tab strip (`ui_primitives.text_tab_row`) sidesteps it and matches
   both the mock and Nuke.

9. **No `duplicate_pass`, hence no duplicating a box** (scenario 11). Conspicuous once bundles are
   first-class; out of scope, but name it with a trigger.

10. **There is no wrong connection, and that should be written down** (scenario 13). The design
    has no type system by choice; say so, and decide whether a dtype mismatch gets a dim
    annotation or nothing.

11. **A second import of the same preset rejects on a name the user did not choose** (scenario 12).
    The message should name the group field as the fix.

---

## False trails

Scenarios tested and dropped, with why, so they are not re-walked.

- **The tutorial walked in the graph instead of the strip** (#15). The 068 tutorial is generated
  per pass by `build_tutorial.py` from the shipped `graph.json`, and 069 #31's card carries six
  fields of which the graph shows one (`reads`). The graph changes nothing about the tutorial. The
  one surviving item — the node's context menu must reach `Settings` — is already in the mock's
  node menu; verify it stays in the spec and drop the rest.
- **The shader-lab workflow producing a preset** (#18). The lab produces documents in a separate
  project; the import dialog's "This project" tab reaches them only when that project is open, and
  the cross-project presets folder is 091's deferral with its own trigger. Nothing here changes
  092's design.
- **Feedback as a wiring hazard.** Tested whether the canvas needs to refuse a self-read or a
  feedback into an iterated pass. It does not: item 6 already allows a node's output into its own
  port, `PassEntry.iterations`' own comment records that the engine does not second-guess the
  author's run count, and an iterated self-reader ping-pongs between iterations by design. No new
  rule.
- **Wrong connections as a type error.** Tested against Substance Designer's typed ports. Every
  port here is a `sampler2D` and every wire is legal; the only real mismatch is dtype/scale
  *quality*, which is #13, a decision rather than a scenario. Dropped as a scenario.
- **A group that is not convex.** The mock's round 3 D already walks it and nothing is refused —
  this is the design working as intended, not a scenario with a gap. Recorded here so the next
  reviewer does not re-open 091's convexity argument: item 4 dissolved it.
- **Output selection through the box.** "Picking a box as the output picks the bundle's output"
  is in *Leaning* and walks cleanly: click the box → `set_output_pass(bundle output)`, the same
  verb a tile click makes. No gap, no decision. It does depend on #3's answer for which member the
  bundle output is.

---

## Does the scenario set justify groups at all?

In his own terms — "groups earn their place only because import makes them" (item 8) — yes, and
only just. Import creates them whether or not anything else does (#1), and the box is the only way
seven copied passes stay one thing on a canvas. Everything else the scenarios asked of groups is
either already free (#2, #14), a small verb (#4's Dissolve), or does not exist yet and does not
need to (#5 nesting, #6b reparent, #11 duplicate). The label survives the whole walk without ever
needing to become an entity, which is `conventions.md`'s revisit trigger staying unfired. The
harder question the walk raises is not about groups: it is whether a graph view of a corpus where
five of six documents are one node earns a second view at all — and the answer there is that it
earns it for the two documents that have a chain, which are exactly the two the import verb exists
to move around.
