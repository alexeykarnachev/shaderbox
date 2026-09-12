# 092 brainstorm review — the persisted model and the copilot's view

Angle: what the graph view ADDS to `graph.json` and to the copilot's view of a document, and what
it must keep out. Every claim below is anchored to a symbol or to a probe run in this session
(`uv run python`, output pasted verbatim).

---

## 1. `PassEntry.position` — shape, default, who writes it

### What the code already settles

**The salvage carries a new optional field for free, and it is the entry-level salvage that does
it, not the generic one.** `document.load_graph` runs `drop_unknown` + `drop_invalid` PER ENTRY
(over `_keyed_entry_fields()`), then pops the keyed dicts out before the whole-model pass. The
comment in `load_graph` says why, and a probe confirms the failure it prevents:

```
$ uv run python -c "... drop_unknown(PassGraph, {'version':2,'output':'a','passes':{'a':{...}}}, 'graph.json') ..."
WARNING | shaderbox.model_salvage:drop_unknown:33 - Ignoring unknown graph.json.passes key: a
after drop_unknown: {"version": 2, "output": "a", "passes": {}}
```

A pass NAME reads as an unknown field name to the generic walker. `load_graph` is the only reason
`graph.json` survives at all, and a `position` field inherits that machinery with no new code —
`_keyed_entry_fields()` enumerates from `PassGraph.model_fields`, so nothing has to be remembered.

**A missing `position` and a corrupt `position` both degrade to the same thing.** Probe, with a
hypothetical `position: tuple[float, float] | None = None` on a `PassEntryV3` copy:

```
old file ->        ... iterations=3 group='fx' position=None
corrupt position ->... iterations=3 group='fx' position=None
```

`drop_invalid` validates the key alone and drops it; `iterations` and `group` beside it survive.
This satisfies "a persisted model salvages per KEY" and "a retired or malformed field costs the
user THAT setting, never the rest of the file" with zero bespoke code.

**`extra='forbid'` is NOT on this model, and the brainstorm's framing of it should be corrected.**

```
PassGraph extra config: None
PassEntry extra config: None
strict construct ok: target=TargetConfig(...) iterations=1 group=''   # from {'version':3, ..., 'position':[10.0,20.0]}
```

`UIAppState` carries `model_config = {"extra": "forbid"}`; `PassGraph` and `PassEntry` carry
nothing, so pydantic's default `ignore` applies — and `load_graph`'s `drop_unknown` already logs
and prunes the unknown key before construction. So **a file from a NEWER build loads clean on an
older one, loses the position, and the older build then writes it away on the next save.** That is
the correct fail-soft behavior under the loadability posture; it is not the `extra='forbid'`
loudness the posture names as the default for app-read state. Worth flagging: the convention
("`extra='forbid'` … is the default for any state read back by the app") and `PassGraph` are
already out of step today, and 092 is not the feature to reconcile them — but the review should
not let the spec claim `extra='forbid'` protection it does not have.

**Bounds belong on the model.** `TargetConfig.scale` is `Field(gt=0.0, le=1.0)`, `iterations` is
`Field(ge=1, le=MAX_ITERATIONS)`, and each carries a comment saying `graph.json` type-checks
nothing. A bare `tuple[float, float]` does NOT inherit that:

```
  position=[nan, 0.0] -> (nan, 0.0)
  position=[1e+30, 1e+30] -> (1e+30, 1e+30)
```

A NaN position propagates into every draw-list coordinate and every bounding-box `min`/`max` the
box computation does; `1e30` puts a node outside any reachable pan. Both need a model-level bound
(`Field(allow_inf_nan=False)` plus a magnitude clamp, the same shape as `MIN_CANVAS_PX` /
`MAX_CANVAS_PX` and `clamp_canvas_size` — a `clamp_position` funnel both entry points reach).

### Shape and space

The brainstorm fixes "positions are the pass's" (Fixed #5), and the code agrees that this is the
only place it can go: `PassEntry` is frozen, and every mutation funnels through
`PassGraph.with_passes` / `model_copy`, which the 091 bullet notes is exactly why `group` needed no
per-callsite work. A `position` rides the same funnel.

Which space: the position is a CANVAS-space coordinate, not a screen one. Zoom-to-fit is the stated
reason a graph beats the strip (Fixed #1), so the stored number must be zoom- and pan-invariant.
The layout function should be dimensionless-ish, in units of one node's nominal width, so a theme
size change does not reshuffle a stored layout — `theme.py` owns size tokens and they move.

**The group-box position question the brainstorm asks is already answered by Fixed #4 + #5, and the
answer is "nowhere".** Fixed #5 says the box sits at its members' bounding box and dragging it
translates the members; Fixed #4 says the box is never a node the planner orders. So the box has NO
stored position: it is derived, every frame, from the members' positions. This is the right call and
it matches "A pass GROUP is a label on the pass entry" — a stored box position would be a
group-level fact no member can hold, which is precisely the 091 revisit trigger ("Revisit if a
group-level fact appears that no member can hold — then it becomes an entity"). **Storing a box
position turns `group` into an entity and reopens 091's whole rejected design.** Keep it derived.

The one real consequence: entering a group tab and laying out its members INSIDE that tab means the
members' positions are now serving two pictures at once — the root's bounding box and the tab's
canvas. If the tab's canvas is the same coordinate space (the tab just pans/zooms to the members'
bbox and draws the outside passes as ghosts at the border), there is one set of coordinates and no
problem. If the tab were to re-lay-out members independently, each pass would need a position per
tab — a `dict[group_path, position]` on the entry, which is a parallel name-keyed dict keyed by
something the user can rename, i.e. the exact drift smell the conventions name twice ("Two parallel
name-keyed dicts…", and the `ui_uniforms` prune rule). **One coordinate space, the tab is a
viewport** is the shape the conventions push toward.

### The first-sight write — the important finding

The brainstorm says "the first sight of a document runs the rank layout once and stores it". Taken
literally that is a **draw-time write**, and two conventions land on it from opposite directions:

- "derived state is pruned at the SAVE funnel … never in the draw loop that lazily CREATES the
  rows (that is the lazy-row trap two bullets down)".
- "A model flag whose ROW is created lazily-on-draw cannot be set programmatically — the writer
  must eager-create. … any programmatic or HEADLESS mutation that runs before that draw silently
  no-ops … it only bites smoke/dogfood/copilot/any off-draw caller."

That second bullet is this exact situation with the nouns changed. If `position` is created on
first draw of the graph view, then: `add_pass` from the copilot cannot give the new pass a
position; the dogfood harness never has one; and a document opened, edited by the copilot and saved
without the graph tab ever being drawn round-trips with positions for some passes and not others.

Also, nothing in the UI writes `graph.json` from a draw today. Every one of the six pass verbs in
`project_session.py` (`add_pass`, `delete_pass`, `rename_pass`, `set_pass_target`,
`set_pass_iterations`, `set_pass_group`) ends in `self.save_ui_document(ui_document)`, under the
comment "Every one mutates the live document AND saves, so `passes/` and `graph.json` never
disagree with what is on screen." The strip (`widgets/pass_list.py`) contains no `save_ui_document`
call at all — a grep over `widgets/pass_list.py` and `tabs/*.py` returns nothing.

**So: a draw-time write is not acceptable, and the layout must be a verb.** The concrete shape that
fits the code:

- `pass_graph.rank_layout(wiring, names) -> dict[str, tuple[float, float]]` — pure, GL-free, in
  `pass_graph.py` beside `strip_order` / `entry_points` / `group_runs`, all of which are already
  pure functions over a `Wiring`. Unit-testable with no context, importable from anywhere.
- A `layout_passes(document_id)` verb on `ProjectSession`, in the same block as the other six,
  ending in `save_ui_document`. "Arrange" calls it. First sight calls it too — but as a deferred
  action off the draw, the same way the project switch is "DEFERRED out of the draw" per its own
  convention bullet.
- `add_pass` seeds the new pass's position at the write seam (the "writer must eager-create" fix),
  not on the next draw. "A new pass lands under the cursor" (Fixed #5) is then a UI-supplied
  argument to the verb when the UI has a cursor, and a layout-derived default when it does not
  (the copilot's `add_pass`, the dogfood harness).

The cleanest resolution of "first sight" is to make it **not a write at all**: a pass with
`position=None` is laid out by `rank_layout` on the fly every frame, and nothing is stored until
the user drags or presses Arrange. `position` then means "the user placed this one", the default
is honest (`None` = never placed, not "never drawn"), and there is no first-sight write to schedule
off the draw. This is strictly simpler and loses nothing the maintainer named.

### `projects/dev/` by hand

The sandbox currently holds two documents, both single-pass:

```
$ ls projects/dev/documents | wc -l
2
$ for f in projects/dev/documents/*/graph.json; do ...; done
1 projects/dev/documents/ec926580-.../graph.json
1 projects/dev/documents/e7e00c46-.../graph.json
```

So the hand-edit obligation from the no-backward-compat rule is nearly free here: a defaulted
optional `position` needs no sandbox edit at all (absent = default, the field is additive). The
rule bites only if the shape CHANGES later — a tuple becoming a dict, a per-tab map — at which
point the sandbox files are hand-edited or regenerated through load+save and `git add
projects/dev` in the same wave. No migration function, no old-format reader. Since the sandbox has
no multi-pass document, the feature's own fixtures live in `tests/` and in the shipped examples
(the Radiance Cascades example dir), which are the real multi-pass artifacts that will grow
positions.

### Verdict — item 1

- Shape (pair, canvas space, per-pass only): **SETTLED-BY-CODE.** `PassEntry` is frozen and every
  mutation funnels through `with_passes`; the box position is derived because storing it would trip
  091's "then it becomes an entity" trigger.
- Default and salvage: **SETTLED-BY-CODE.** `load_graph`'s per-entry `drop_unknown`/`drop_invalid`
  handles absent, corrupt, and newer-build files; probes above.
- Bounds on the field: **SETTLED-BY-CODE** in the sense that the convention is explicit ("New
  persisted state gets constraints on the model") and the probe shows NaN/1e30 pass unbounded today
  — so the spec must add them; there is nothing for the maintainer to decide.
- The first-sight write: **CONFLICTS** — a draw-time write collides with "derived state is pruned
  at the SAVE funnel, never in the draw loop" and with the lazy-row-on-draw bullet. The resolution
  is a verb, and the recommended one is "store nothing until the user places a node".
- `extra='forbid'`: **OPEN (small)** — the brainstorm should not assume it; `PassGraph` does not
  set it (probe above).

---

## 2. `group` as a PATH (`post/bloom`)

### What the pattern allows today

```
_GROUP_PATTERN = ^([A-Za-z_][A-Za-z0-9_]*)?$
  PassEntry(group='post/bloom') -> REJECTED
  PassEntry(group='post')       -> ACCEPTED
  PassEntry(group='')           -> ACCEPTED
  PassEntry(group='post_bloom') -> ACCEPTED
  PassEntry(group='a/b/c')      -> REJECTED
```

A slash is rejected at the model, so nesting is a real model change, not a convention someone can
adopt informally. The comment above `PASS_NAME_RE` states the reason the pattern is what it is: "A
pass name is a FILENAME and a graph key… A group name is a pass-name prefix and a border label, so
it obeys the same rule."

### What each consumer does with a slash

**`pass_import.plan_import` — hard reject, and the message is the right one.**

```
group='post/bloom' -> a group name starts with a letter and holds letters, digits and underscores
group='post'       -> {'blur': 'post_blur', 'comp': 'post_comp'}
```

`plan_import` checks `PASS_NAME_RE.match(group)` explicitly, independently of `_GROUP_PATTERN`, and
then builds `prefix = f"{group}_"` and `renames = {name: f"{prefix}{name}"}`. **This is the
load-bearing one: the group name is a PASS-NAME PREFIX, and a pass name is a FILENAME.**
`post/bloom` as a prefix produces a pass named `post/bloom_blur`, which is a path with a directory
separator in it, written to `passes/post/bloom_blur.frag.glsl` by `paths.pass_shader_for`. The
loader enumerates `(document_dir / PASSES_DIR_NAME).glob(f"*{PASS_SHADER_SUFFIX}")` — non-recursive
— so such a pass would write and never load back. `_pass_name_error` / `PASS_NAME_RE` would also
reject the resulting name at `add_pass`.

So nesting-as-a-path forces a **decoupling of the group label from the import prefix**: the label
may be `post/bloom` while the prefix stays `bloom` (the last segment) or `post_bloom` (the path
flattened). The brainstorm's line "import prefixes inner labels instead of dropping them" is
describing a change to `plan_import`'s prefix rule, and the spec must say which of the two it is.

**`group_slug` — unaffected, and it already can't produce a path.**
`group_slug` takes the FIRST WORD of a display name and `re.sub(r"[^A-Za-z0-9_]", "_", ...)`. A
slash in a document name becomes an underscore. `tests/test_pass_graph.py::test_group_slug_is_the_first_word_made_legal`
asserts `PASS_NAME_RE.match(group_slug(name))` for every case, so `group_slug` stays a single
segment no matter what nesting the label allows — which is correct, since it prefills the IMPORT
dialog's group field (`popups/import_passes.py`: "marks the tiles and prefixes the passes").

**`theme.group_tint` — silently wrong under paths, in two ways.**

```
group_tint('post')       index = 1
group_tint('post/bloom') index = 1
group_tint('post/blur')  index = 3
group_tint('fx')         index = 0
group_tint('bloom')      index = 3
```

`group_tint` hashes the WHOLE string (`zlib.crc32(name.encode()) % 4`). Two consequences: (a)
`post` and `post/bloom` collide at index 1 — a parent box and a child box drawn the same hue, which
is the one thing the theme invariant assert exists to prevent between a tint and every other cue
(`assert not set(COLOR.GROUP_TINTS) & _GROUP_TINT_EXCLUSIONS`, and
`assert len(set(COLOR.GROUP_TINTS)) == len(COLOR.GROUP_TINTS)` — "two group tints are the same hue,
so two groups would look alike"). The assert cannot see this collision because both sides are the
same token. (b) `post/bloom` and `post/blur` get unrelated hues (1 and 3), so siblings of one
parent read as unrelated groups. Neither is a bug in `group_tint`; both are the consequence of
feeding it a composite key. Under nesting the spec must say what `group_tint` is called WITH — the
full path (today's collisions), the first segment (siblings share a hue, which may be what the box
picture wants), or the last (the current behavior for a flat label). Four hues over a nested
namespace is also a real crowding problem the 091 bullet's revisit trigger ("Revisit if a fifth
palette hue appears") anticipates from the other side.

**`group_runs` — works unmodified, exactly by accident of being a string compare.**

```
order=['a','b','c','d'], groups={'a':'post/bloom','b':'post/bloom','c':'post/blur','d':''}
group_runs -> [['a','b'], ['c'], ['d']]
```

It compares labels for equality, so full-path labels cut runs at the LEAF level: `post/bloom` and
`post/blur` are two runs even though both are under `post`. The brainstorm says "the strip outlines
by full path", which is what this already does — but note it means the strip shows no parent outline
at all. A parent-level outline would need `group_runs` to take a depth or a key function, which is a
second cut of the same order and a second outline nesting on the tiles. That is UI work 091
deliberately did not take on; keeping the strip at leaf-level outlining is the no-change option and
the one the "the strip stays as it is" fix (#7) points at.

**The copilot's pass table — passes the string through verbatim.** `backend._pass_table` emits
`f"{f', group {entry.group}' if entry.group else ''}"`, so a path label renders as `group
post/bloom`. Pinned by `tests/test_copilot_pass_tools.py::test_set_pass_group_lands_and_echoes`,
which asserts the literal `"glow: runs 1, target f2 x1, linear, group fx"` and
`_graph(app, document_id)["passes"]["glow"]["group"] == "fx"`. A path label needs no format change;
the test needs a path case added, and `set_pass_group`'s own `PASS_NAME_RE.match(group)` check in
`project_session.py` must be relaxed in lockstep with `_GROUP_PATTERN` or the copilot will reject
what the UI accepts — **there are two independent validators of the same field today**
(`_GROUP_PATTERN` on the model, `PASS_NAME_RE` in `set_pass_group`, and a third in `plan_import`),
and a nesting change that updates one leaves the others silently narrower. That is a checker
quietly narrowing its own domain; the spec should collapse them to one `is_legal_group(label)` in
`pass_graph.py` that all three call.

### Path string vs tuple

Store the STRING. Reasons from the code, not from taste:

- `PassEntry` is a pydantic model serialized straight to JSON by `json.dump(...model_dump())`. A
  tuple round-trips as a list, and `drop_invalid` then has to validate a sequence rather than a
  pattern — losing the single `Field(pattern=...)` that today makes the whole legality question one
  regex the model enforces for free.
- `group_runs` compares labels by equality; a string compare is that, a tuple compare is also that,
  but every other consumer (`group_tint`'s `.encode()`, the table's f-string, the import prefix)
  takes text.
- A tuple invites an empty-tuple-vs-`("",)` ambiguity where `""` today is unambiguously "no group",
  checked by `if entry.group` at three call sites.

A path string with a documented separator, one `is_legal_group` validator, and helpers
(`group_segments(label) -> list[str]`, `group_parent(label)`) in `pass_graph.py` beside
`group_runs` is the shape that matches everything already there.

### Nesting's effect on boundary ports and the two-disconnected-passes case

Fixed #4 computes a box's ports from membership: a member sampler reading an outside pass is an
input port. Under nesting, "member" has to mean "member at or below this path" for the root's box
and "direct member or member of a sub-box" for a tab. That is a prefix test on the label
(`label == path or label.startswith(path + "/")`), pure, and it composes at every depth as the
brainstorm claims. Nothing in `plan_passes` changes — the box is still never a planner node, which
is what keeps 091's convexity rule dead.

**The "same name typed on two disconnected passes" case gets WORSE under nesting, and the code says
so.** Today a group is "a label on the pass entry… a group exists while a pass carries its name",
and `group_runs` cutting by ADJACENCY is the strip's honest answer: two disconnected passes labelled
`fx` draw as two runs, and nothing claims they are one thing. The graph view's box, by contrast,
claims exactly that: one box, one bounding box, one set of boundary ports. So the graph view ALREADY
introduces the question 091 avoided, at depth 0, before nesting. Fixed #4 answers it ("A non-convex
group draws a back edge at the root and its leaking pass as a ghost on both sides… Nothing is
refused"), which is consistent. Nesting multiplies the cases (a non-convex group whose parent is
also non-convex) but adds no new KIND. The one new kind nesting does add: a label `post/bloom` where
no pass carries `post` — an implied intermediate box with no direct members. The box-is-derived rule
answers it (the box exists because a descendant path implies it), but the spec must say it out loud,
because it is the first group that exists with zero passes carrying its name, which contradicts the
091 bullet's sentence verbatim.

### Verdict — item 2

- What the pattern allows: **SETTLED-BY-CODE.** `_GROUP_PATTERN` rejects every slash form (probe).
- The import prefix under a path: **CONFLICTS** — `plan_import`'s `prefix = f"{group}_"` produces an
  illegal pass name (a filename with a separator) that `PASS_NAME_RE`, `_pass_name_error` and the
  non-recursive `passes/` glob all independently reject. Nesting requires deciding the prefix rule
  separately from the label.
- `group_tint` under a path: **OPEN** — three defensible keys (full path / first segment / last
  segment), with a measured parent-child collision on the full-path option.
- `group_runs` / the strip: **SETTLED-BY-CODE** at leaf level (probe); a parent-level outline is new
  UI the "strip stays as it is" fix argues against.
- String vs tuple: **SETTLED-BY-CODE** — string, because `Field(pattern=...)` is the single
  validator and every consumer takes text.
- Three independent group validators: **CONFLICTS** with the structural-impossibility law and the
  narrowing-checker family; collapse to one function in the same wave.
- The implied parent box with no members: **OPEN** — it contradicts "a group exists while a pass
  carries its name" as written.

---

## 3. The copilot

### Should the model see positions? No.

Straight from the actor model in `.claude/skills/copilot-llm-agent-design/SKILL.md`: the model has
exactly two reliable behaviors — it copies text verbatim, and it is blind outside its token stream.
A coordinate is named in the skill's own list of what the model must SYNTHESIZE rather than copy
("ask it to *describe* a location, a count, a coordinate, an intent… and it will be imprecise in
ways no guard can fully catch"). The whole line/anchor arc (020·14 → 036 → 038 → 039) was deleting
exactly this class of addressing.

Apply the guard test the conventions state ("A copilot tooling/prompt GUARD earns its place only if
a strictly BETTER model would still need it", inverted for an affordance): **would a strictly better
model produce a better shader if it knew where the nodes sit?** No. Node positions change nothing
about what renders. They are a property of the picture the user looks at, not of the document's
behavior. Feeding them costs tokens on every `_pass_table` echo and every working-set rebuild, for
a fact with no downstream use — the "permanent prompt tax" the same bullet names.

The corollary is the one worth writing into the spec: **`add_pass` must not ask the model for a
position, and `set_pass` must not accept one.** If it did, the model would synthesize coordinates
and the graph would fill with overlapping nodes at plausible-looking numbers. The position for a
copilot-created pass comes from the engine's own `rank_layout` (or stays `None` and is laid out on
sight, per item 1's recommendation) — which is the same answer as "a tool-side normalizer at the
parse boundary, never a standing prompt rule" applied one level up.

### Should the model see boxes or the flat passes? The flat passes.

`_pass_table` today lists one row per pass with `group X` appended, and `prompt.py`'s working-set
renderer emits `=== PASS <name> (edit as: <id>#<name>) ===` sub-sections. Both are FLAT, and both
are addressing surfaces: `<id>#<name>` is how an edit lands. A box is not addressable — there is no
`<id>#post/bloom` to edit, because a box has no source. Rendering boxes in the model's view would
give it a noun it cannot act on, and the address-kind convention is explicit about the shape a new
addressable thing takes ("A new addressable copilot SOURCE kind gets a `<kind>:` prefix + rides the
EXISTING read/grep, never a parallel tool") — a box is not a source, so it gets no address.

The group label on the flat row is the right amount: it is a fact the model can copy verbatim into
a `set_pass(group=...)` call, and it is already there. Under nesting the row becomes `group
post/bloom`, which needs no format change.

### Does the graph view change what `set_pass` accepts? Only if nesting lands.

`_SetPassArgs.group` is `str | None` with the description "the group the pass belongs to (a label
on the strip); '' leaves the group". Two things follow:

- **If nesting lands, that description is stale in a load-bearing way** — "a label on the strip"
  stops being the whole truth, and the model has no way to know a slash is legal. The description
  is where it learns the field's domain (corollary 2: blind outside the stream). It becomes
  something like "the group the pass belongs to; `a/b` nests b inside a; '' leaves the group".
- **The validator split is the real change.** `set_pass_group` in `project_session.py` checks
  `PASS_NAME_RE.match(group)` and returns "a group name starts with a letter and holds letters,
  digits and underscores" — a message the model reads verbatim. If `_GROUP_PATTERN` is widened and
  this is not, the copilot rejects a label the UI accepts, with a message that is then false. The
  `tests/test_copilot_pass_tools.py` case asserting `"group name" in bad.error` pins the message but
  not its truth.

Positions: no change. `set_pass` stays as it is.

### Is a `group_passes` tool warranted? No.

The tool-count bullet is direct: "Tool count must not grow casually… every tool's description is
re-billed on every iteration (it dilutes attention from the load-bearing rules). Prefer enriching an
existing tool's result over a new tool." `set_pass(group=...)` already does the whole job, one pass
per call. A `group_passes(names, group)` tool would be N calls collapsed into one — a latency saving
on an operation the model performs approximately never, since **groups are, by the maintainer's own
Fixed #8, a thing IMPORT makes**, and import is not a copilot tool at all. The model has no workflow
that produces a multi-pass group from scratch.

The one argument for it — "the copilot should be able to tidy an imported bundle" — is speculation
about a workflow nobody has run, and the speculative-machinery test ("is REMOVING it churn?") says
add it when a dogfood trace shows the model wanting it.

### What the prompt's pass block should say about groups

Today it says nothing about groups at all. `prompt.py`'s multi-pass paragraph covers passes,
addresses, `u_<pass>` and `u_prev`, and stops there. That is correct and should stay correct even
if nesting lands, for the reason the prompt-home bullet gives ("a rule's HOME follows WHEN it
fires"): a group affects nothing the model does — not compilation, not wiring, not rendering. It is
a label the strip and the graph draw. The model learns the field exists from `set_pass`'s schema,
which is where a per-tool fact belongs ("A static per-tool fact is a `ToolDefinition` field").

Adding a groups paragraph to the static prompt would be prompt tax paid on every request for a
concept that steers no decision. The one thing that WOULD earn a line is if a group ever gained
semantics the render depends on — which Fixed #4 explicitly prevents by keeping the box out of the
planner.

### Verdict — item 3

- Positions in the model's view: **SETTLED-BY-CODE** (by the skill's actor model + the guard test);
  the spec should write "no tool accepts or reports a position" as an explicit non-goal, because
  the natural implementation adds it.
- Boxes vs flat passes: **SETTLED-BY-CODE** — `_pass_table` and the working-set renderer are
  addressing surfaces and a box has no address.
- `set_pass` changes: **OPEN, conditional on nesting** — the `group` description and the
  `set_pass_group` validator/message both need the same widening, in the same wave.
- `group_passes` tool: **SETTLED-BY-CODE** — the tool-count bullet and the speculative-machinery
  test both say no.
- The prompt's pass block: **SETTLED-BY-CODE** — no groups paragraph; the schema carries it.

---

## 4. The dogfood / headless harness

The convention is short and it decides most of this: "The dogfooding station records; it never
judges… An observer of the copilot engine reads the `TraceLog` listener seam
(`CopilotSession.trace_listeners`), never the plain-text transcript."

The graph view is UI — `ImDrawList`, `invisible_button`, a tab bar — and `ProjectSession` is "the
headless project + copilot core" with `App` owning one and forwarding. So the split is already
drawn: **nothing about drawing the graph belongs in the headless engine.** The harness renders
shaders, not panels.

What the headless side DOES need is the pure half, and only because item 1's verb structure puts it
there anyway:

- `rank_layout(wiring, names) -> dict[str, tuple[float, float]]` in `pass_graph.py`, pure and
  GL-free, beside `strip_order` / `entry_points` / `readers_of` / `group_runs`. Every one of those
  is already a pure function over a `Wiring` that the strip consumes and tests exercise without a
  context. A layout function joins that set naturally, and `pass_graph.py`'s own docstring makes the
  promise it has to keep: "everything here is pure data… unit-testable with no context and
  importable from anywhere without a cycle."
- The boundary-port computation (Fixed #4) is the same kind of function: membership by label prefix,
  ports from the wiring. Pure, GL-free, testable with two dicts. Putting it in `pass_graph.py`
  rather than in the canvas widget is what lets a test assert a non-convex group's ports without a
  window — and the dev box has no window manager, so "every visual call is the maintainer's"
  (brainstorm, "What the code gives us").

That is the whole headless requirement, and note it is not a dogfood requirement — it is a
TESTABILITY requirement that the dogfood harness inherits for free. The harness needs no new
capability, no new trace event, and no judgement about whether a layout is good. A layout the
harness could score would be exactly the standing checker the convention forbids.

One thing to keep out explicitly: **no layout assertion in the smoke test beyond "it did not
crash".** `make gates` reports a skipped smoke as skipped when there is no display; a graph-view
smoke canary would be skipped on the dev box and therefore prove nothing, while reading as
coverage. The pure functions are where the gate has teeth.

### Verdict — item 4

**SETTLED-BY-CODE.** The headless engine needs the layout and the boundary-port computation as pure
functions in `pass_graph.py` — which item 1's verb structure requires independently — and nothing
else. No harness change, no trace event, no judging.

---

## 5. `graph.json`'s `version`

### What the repo's practice actually is

`GRAPH_JSON_VERSION = 2`, and `version: int = GRAPH_JSON_VERSION` on `PassGraph`. Three facts,
each checked:

1. **Nothing reads it.** A grep for `.version` across `shaderbox/` and `tests/` (excluding imgui /
   sys / `__version__`) returns nothing. It is written by `model_dump()` and never consulted — not
   by `load_graph`, not by `Document.load_from_dir`, not by any test. `tests/test_graph_persistence.py`
   mentions `"version": 2` only as a field in a fixture it constructs.
2. **It was bumped once, 1 → 2, in `5357b1f` "Land 070, 072 and the 073 host waves"** — the wave
   that reshaped a sampler's source into a VALUE on the sampler (`PassSource`/`NoSource`/
   `AutoSource`) and deleted `PassGraph.layout`. That is a change of MEANING: the same keys, read
   differently.
3. **091 added `group` and did NOT bump it.** `GRAPH_JSON_VERSION` is untouched since `5357b1f`;
   `PassEntry.group` landed after it. So the repo's practice, read from the only two data points
   available, is: **an additive defaulted field does not bump; a reinterpretation of existing data
   does.**

That practice is exactly what the persistence-evolution posture prescribes: "Adding a field: make
it defaulted + loose-typed-optional + fail-soft on load (no migration)… the App-state version stamp
exists only to fail-soft-reset a foreign file, not to walk it forward." And the App state doesn't
even carry a stamp — `UIAppState` has `model_config = {"extra": "forbid"}` and no `version` field,
which is the posture's "make a stale key LOUD" half implemented without a number.

### Does a position field bump it?

No. `position` is additive, defaulted, and absent-means-default; nothing about the existing keys
changes meaning. Bumping to 3 would write a number nobody reads to mark a change nobody needs to
detect — and worse, it would look like a migration boundary, which is the thing the no-back-compat
rule says to delete on sight.

The honest observation to put in the spec: **the stamp is dead weight today.** It is write-only, it
has no reader, and there is no `load_and_migrate` ladder for it to feed (the conventions say
explicitly there is none). Under "structural impossibility over guard-piles" and the
speculative-machinery test ("is REMOVING it churn?"), removing it is close to zero churn —
`drop_unknown` would prune the key out of every existing file on load, silently, and the sandbox's
two `graph.json` files are hand-editable in one pass. That is a decision for the maintainer, not
something 092 should do on its own; but 092 should NOT bump it, and should not add the reader that
would justify it.

### Verdict — item 5

- Practice: **SETTLED-BY-CODE.** Bumped once for a reinterpretation (`5357b1f`); not bumped for
  091's additive `group`. Nothing reads it.
- A position field: **SETTLED-BY-CODE — does not bump.**
- Whether the stamp should exist at all: **OPEN**, and out of 092's scope unless the maintainer
  wants it swept.

---

## Decisions the maintainer must make

### D1. What does "first sight lays out the document" mean on disk?

A draw-time write collides with two conventions (item 1). Three options:

**(a) Lay out lazily, store nothing.** `position` means "the user placed this one".

```python
class PassEntry(BaseModel):
    position: tuple[float, float] | None = None   # None = never placed; rank_layout decides
```
The canvas calls `rank_layout(wiring, names)` each frame for the `None` ones, which is a
topological sort over at most a dozen passes. Nothing is written until a drag or Arrange.
No first-sight write to schedule, no lazy-row trap, and the default is honest.

**(b) Lay out on entry, write through a verb, deferred out of the draw.** The graph tab's open
handler queues `session.layout_passes(document_id)`, which ends in `save_ui_document` like the other
six verbs. Every pass always has a position on disk. Costs a deferred-action path and makes opening
a tab a mutation.

**(c) Lay out at the write seam.** `add_pass` seeds a position for the new pass; a document whose
passes predate the feature gets positions on the next verb that touches it. Partial state on disk
is normal and permanent.

(a) is the recommendation: it is the only one where opening a view writes nothing.

### D2. Does nesting land, and if so what is the import prefix?

`plan_import` builds `prefix = f"{group}_"`, so a `post/bloom` label yields the illegal pass name
`post/bloom_blur` (item 2). If nesting lands, pick one:

- **Last segment:** label `post/bloom`, passes named `bloom_blur`. Collision risk across two
  sub-groups of different parents with the same leaf name — and collision is exactly what the
  prefix exists to prevent (091 D2: "the prefix is an import-time collision guard").
- **Flattened path:** label `post/bloom`, passes named `post_bloom_blur`. No collision, long names,
  and the pass name no longer matches the label the user typed.
- **Label and prefix decouple entirely:** the import dialog keeps asking for a single-segment
  prefix and separately offers a parent to nest under. Two fields in
  `popups/import_passes.py` where there is one today.

### D3. What key does `group_tint` get under nesting?

Measured collisions (item 2): full path gives `group_tint('post') == group_tint('post/bloom')`
(both index 1) and `post/bloom` vs `post/blur` at 1 vs 3.

- **First segment:** all children of `post` share the parent's hue. Reads as "one family", loses
  sibling distinction.
- **Last segment:** today's behavior for a flat label; siblings differ, a child may collide with an
  unrelated top-level group.
- **Full path:** what falls out if nothing is done; has the measured parent-child collision above.

Four hues total either way, which the 091 bullet already flags as the palette's limit.

### D4. Does the spec collapse the three group validators?

`_GROUP_PATTERN` (model), `PASS_NAME_RE.match` in `project_session.set_pass_group`, and
`PASS_NAME_RE.match` in `pass_import.plan_import` all independently decide what a legal group is.
Widening one leaves the others narrower, silently. Options: collapse to one `is_legal_group()` in
`pass_graph.py` that all three call (recommended, and it is the "single FUNNEL" law); or leave them
and pin the agreement with a test that asserts all three accept the same set.

---

## False trails

- **"`extra='forbid'` protects `graph.json` from a newer build's key."** It does not —
  `PassGraph.model_config.get('extra')` is `None` (probe). `load_graph`'s `drop_unknown` does the
  work, logging and pruning. The behavior is right; the attribution in the brainstorm's framing is
  wrong, and it matters because a spec that assumes `forbid` may skip the salvage test.

- **"The generic `model_salvage.load_model` can load `graph.json`."** It cannot: `drop_unknown`
  reads every pass NAME as an unknown key and prunes `passes` to `{}` (probe output above). This is
  why `document.load_graph` exists as a separate function with the per-entry loop. Any new field on
  `PassEntry` must be tested through `load_graph`, not through `load_model`.

- **"The box needs a stored position because dragging it moves it."** Fixed #5 already says
  dragging the box translates the members, so the members' positions ARE the box's position. Storing
  one would create the group-level fact that 091's revisit trigger says turns `group` into an
  entity — reopening the design 091 rejected, in a feature that explicitly reopens only the FOLDING
  half.

- **"A `group_passes` tool would be symmetric with the UI's Group verb."** Reflex symmetry is named
  as the anti-pattern in the addressing bullet ("picks its addressing by RISK, not by reflex
  symmetry") and the tool-count bullet forbids the growth. `set_pass(group=...)` covers it.

- **"The copilot should see the graph layout so it can reason about the pipeline's structure."** It
  already sees the structure — the working-set block lists every pass with its samplers and what
  each reads (`<- <pass>`), which IS the wiring. Positions add geometry, not structure.

- **"Bump `version` to 3 for the new field."** 091 added `group` without bumping; nothing reads the
  stamp; the posture forbids the migration ladder a bump would imply.

- **"The strip needs a nested outline if nesting lands."** `group_runs` already cuts by full-label
  equality, which is leaf-level outlining and is what the brainstorm describes. A parent outline is
  new strip UI, against Fixed #7.

- **"`group_slug` needs a path-aware variant."** It prefills the import dialog from a DOCUMENT
  NAME's first word and is pinned to produce a legal single segment
  (`test_group_slug_is_the_first_word_made_legal`). A document name has no nesting to derive.

---

## Verdicts at a glance

| Item | Verdict |
|---|---|
| 1 — `PassEntry.position` shape / default / salvage | SETTLED-BY-CODE (per-entry salvage, frozen model, derived box position) |
| 1 — the first-sight WRITE | CONFLICTS (draw-time write vs the save-funnel + lazy-row rules); resolve via D1 |
| 1 — bounds on the field | SETTLED-BY-CODE (convention explicit; NaN/1e30 measured unbounded — the spec must add them) |
| 2 — what the pattern allows | SETTLED-BY-CODE (`_GROUP_PATTERN` rejects every slash form) |
| 2 — the import prefix under a path | CONFLICTS (illegal pass name / filename); resolve via D2 |
| 2 — `group_tint` key | OPEN; resolve via D3 |
| 2 — `group_runs` / the strip | SETTLED-BY-CODE (leaf-level, unmodified) |
| 2 — string vs tuple | SETTLED-BY-CODE (string) |
| 2 — three validators | CONFLICTS (single-funnel law); resolve via D4 |
| 2 — an implied parent box with no members | OPEN (contradicts "a group exists while a pass carries its name") |
| 3 — positions in the copilot's view | SETTLED-BY-CODE (no) |
| 3 — boxes vs flat passes | SETTLED-BY-CODE (flat) |
| 3 — `set_pass` changes | OPEN, conditional on nesting (description + validator together) |
| 3 — a `group_passes` tool | SETTLED-BY-CODE (no) |
| 3 — the prompt's pass block | SETTLED-BY-CODE (no groups paragraph) |
| 4 — the headless harness | SETTLED-BY-CODE (pure `rank_layout` + boundary ports in `pass_graph.py`; nothing else) |
| 5 — `version` bump | SETTLED-BY-CODE (no bump); whether the stamp should exist at all is OPEN and out of scope |
