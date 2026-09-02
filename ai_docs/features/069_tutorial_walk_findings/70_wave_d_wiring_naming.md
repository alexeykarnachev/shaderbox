# 069 W-D - Naming, default wiring, strip tune

Implementation spec for wave W-D of feature 069. The parent spec (`01_spec.md § W-D`) fixes the
shape; this file fixes the code. Locked decisions D9 and D12 apply and are not re-opened, nor is
the parent's four-point RESOLUTION-rule design: `""` stores an explicit none, an absent key
resolves by name through one pure `effective_inputs`, a media-bound sampler is never auto-wired,
and the gear's combo has three kinds of item. What this file decides is the exact code shape of
each of those, and the exact tokens the rename moves.

W-C, W-A, W-B and W-F have landed at `d2ade88`. Every citation below names a symbol rather than a
line, except where a line is the finding itself.

## Goal

An input uniform's name says what fills it, and saying it is enough. A person who writes
`uniform sampler2D u_df;` in a pass that sits beside a pass called `df` gets `df`'s texture on the
next frame without opening the gear, because the name IS the wire. A person who wants that same
sampler black says so once, in the gear, and it stays black across a reload instead of being
re-wired by the rule that filled it in the first place. Both shipped multi-pass examples are
renamed to the one convention, so the tutorial the next wave generates from them never has to
teach a wiring step that the engine already performed. And the pass strip stops trying to draw the
graph in truncated text under a thumbnail: it is a name and a picture, and the wiring lives in the
gear until feature 070 draws it properly.

## Findings folded

Two, quoted verbatim from `00_findings.md`:

- **#19** (UX design, Passes strip - the cards): "The input mappings don't fit the card ('u_prev
  <- …' gets cut). The `<-` reads as a cheap workaround. We need a better representation:
  visualise the passes as a directed graph — smaller, clean square previews with a small
  pre-render, connected with arrows. Or at least tune the current visuals, it is awful."
- **#37** (DEFECT (convention), Pass input naming, tutorial + both examples): "very inconsistent
  naming: sometimes the input uniform matches the pass name, sometimes not (u_scene ← paint)."

#19 lands as its option B. #37 lands whole: the rule, both examples renamed to it, and the
default-wiring pay-off the finding names ("the gear can DEFAULT-wire by name").

## Out of scope

- **The graph view of the strip (#19 option A).** Feature 070, per the parent spec's Out of scope,
  and its direction is already fixed there: thumbnails as nodes in evaluation order, edges
  labelled by pass name under D9, feedback as a loop mark, read-only, drawn with `draw_list`
  lines and arrowheads on a plain child - **not** `imgui_node_editor`. W-D leaves the strip a
  wrapping row of `preview_cell` tiles and changes only what each tile shows. Trigger: 070 opens
  once 069 lands.
- **The tutorial's own wiring prose.** W-H regenerates the pass cards and pass code from the
  renamed example files; W-D changes no file under `ai_docs/features/068_radiance_cascades/` and
  no tutorial text. The parent's order makes W-D the wave before W-H for exactly this reason.
- **A UI for the naming rule beyond one sentence.** The rule is stated in the Help panel's pass
  section and the copilot prompt's pass block. The add-pass stub gets no comment: `PASS_STUB` is
  six lines of black-fill GLSL with no sampler in it, so a comment about sampler naming there
  would document a uniform the stub does not declare. (#37 suggests the stub as a third home;
  this is a deliberate departure, recorded here.)
- **Renaming uniforms in the five single-pass examples.** They declare no sampler that names a
  pass, so D9 has nothing to say about them, and their tuned values persist in `document.json`
  keyed by uniform name - a rename there is pure risk with no pay-off.
- **A layout field for the strip.** `PassGraph.layout` stays unused; 070 owns it.

## Design decisions

### 1. `effective_inputs` is the one resolution, and it is pure

New free function in `shaderbox/pass_graph.py`, beside the planner and importable anywhere:

```python
def effective_inputs(
    entry: PassEntry,
    samplers: Sequence[str],
    passes: Collection[str],
    consumer: str = "",
    bound: Collection[str] = (),
) -> dict[str, str]:
    """Which pass fills each of `consumer`'s samplers, stored edges and name defaults together.

    Three states per sampler, and they are distinct on purpose:

    - a name in `entry.inputs` -> that pass, when it exists (a stale name is left in place; the
      planner reports it as unresolved and the renderer binds black);
    - `""` in `entry.inputs`   -> nothing, explicitly. The user picked "(none)" and the default
      rule must not undo it;
    - no key at all            -> undecided, so the NAME decides: `u_<x>` fills from pass `<x>`
      when a pass called `<x>` exists, and `u_prev` from `consumer` itself.

    A sampler named in `bound` never auto-wires: its value is a texture the user bound, and the
    name rule would silently replace it.

    GL-free: `samplers` are NAMES, so nothing here compiles, binds or touches a context.
    """
```

The body carries every stored edge through first (`{u: src for u, src in entry.inputs.items() if
src}`) and then loops `samplers`, applying the name rule to the keys the entry does not decide and
skipping those in `bound`; `_auto_source(uniform, consumer)` strips the `u_` prefix and maps
`u_prev` to `consumer`. A sampler that resolves to nothing is simply absent from the returned
dict, which is exactly the shape `PassEntry.inputs` already has and every consumer already
handles.

**Carrying stored edges through is required, not stylistic.** `samplers` comes from a COMPILED
program, so a freshly loaded document passes `[]` for every pass -- and a body that only looped
`samplers` would return `{}` there, losing every edge `graph.json` holds and planning a document
with no wiring at all. The consequence to know: a stored edge on a uniform the program does not
declare survives into the effective graph. That is harmless, because `core.py` binds by declared
uniform, and it is the same tolerance `unresolved_inputs` already extends to a stale name.

Two things it deliberately does NOT do. It does not read `PassGraph` - it takes one entry and the
set of pass names, so a caller that has only a subgraph (the planner building an effective graph
below) can call it per entry without circularity. And it does not consult the sampler's value -
`bound` is passed in by whoever can see `uniform_values`, which keeps `pass_graph.py` free of any
import from `core.py` or `media.py`.

`u_prev` is not special-cased as a literal beyond the `consumer` substitution: D9 states the
feedback exception as a name (`u_prev` reads yourself), so it is `_auto_source` returning
`consumer` for that one uniform. That means a pass called `prev` beside a `u_prev` sampler
resolves to the SELF pass, not to `prev`. The feedback exception wins, because it is the one D9
writes down; a document that wants pass `prev` wires it explicitly in the gear.

**A sampler whose name lacks the `u_` prefix gets no auto edge.** `_auto_source` returns nothing
for a bare `tex` or `noise`: D9's rule is about `u_<pass>` names, so a name outside that shape is
outside the rule and the only honest answer is to leave it undecided. The alternative - treating
`tex` as naming a pass `tex` - would extend D9 past what it says on the strength of a guess, and
it is the branch a user cannot predict from the rule they were taught. This is the one cell the
matrix left open; it is decided here rather than in the code's behaviour.

### 2. `Document` gathers sampler names from COMPILED passes only, and never compiles to do it

`Pass.get_active_uniforms()` compiles a never-attempted pass (066 D1's third puller). Calling it
per pass per frame to discover samplers would compile the whole document on frame one and invert
that decision. So `Document` gets a private accessor that reads the program only if it is already
there:

```python
    def _sampler_names(self, render_pass: Pass) -> list[str]:
        # Compiled passes only: get_active_uniforms() COMPILES a never-attempted pass (066 D1),
        # so asking it here would compile the whole document on frame one. W-C's first-render
        # sweep is what brings each pass online, one per frame.
        program = render_pass.program
        if program is None:
            return []
        return [
            name
            for name in program
            if isinstance(program[name], moderngl.Uniform)
            and getattr(program[name], "gl_type", None) == GL_SAMPLER_2D
        ]
```

An uncompiled pass therefore contributes no auto edges, and its own auto edges do not exist yet
either. W-C's sweep draws one never-rendered pass per document per frame, and a render compiles
what it draws, so a six-pass document is fully online within six frames of the render set
admitting it - the same fixpoint W-C already ships and pins. The observable consequence is stated
plainly: **the draw order may change between frame 1 and frame N as passes come online**, and a
pass whose auto input has not resolved yet reads black for those frames. That is one frame from
black per pass, the same cost any lazy first render already pays, and 066 D1 holds unchanged.

Opening a pass's gear also compiles it, via `_sampler_names` in `popups/pass_settings.py`, which
still calls `render_pass.get_active_uniforms()`. That is correct there and stays: the user asked
to see this pass's inputs, so paying its compile is what they requested. The gear is the one
place a compile is triggered by the wiring UI.

**Why the program and not `uniform_values.keys()`.** The two are populated by different events,
and only the program is set by `compile()`. `Pass.compile()` assigns `program`, `vbo`, `vao` and
writes the engine tables, and does NOT seed: `seed_uniform_values` runs at the top of
`Pass.render`, inside `get_active_uniforms`'s own compile branch, and at save. So a pass that was
compiled but never rendered - which is exactly what `project_session.add_pass`'s bare
`render_pass.compile()` produces - has a program full of samplers and an empty `uniform_values`.
Reading the program is what makes `_sampler_names` correct in that window; reading
`uniform_values` would return nothing there and the auto edges would not appear until the first
render. Decision 4's `bound` set is on the other side of the same window, and is treated there.

### 3. `Document` builds an effective graph once per render, and the planner sees it

`Document.render` today calls `plan_for_output(self.graph, resolved)` and then, inside the loop,
reads `entry.inputs` to bind textures. Both must see the same resolved wiring, or the planner
orders a draw the binder does not perform. One method produces it:

```python
    def effective_graph(self) -> PassGraph:
        """`self.graph` with every sampler's effective source filled in (069 D9).

        The planner must see the auto edges or it cannot order the draw, and it cannot detect a
        cycle a name default creates. Built from COMPILED passes only, so it grows as the sweep
        brings passes online rather than compiling them to find out.
        """
        names = set(self.passes)
        entries: dict[str, PassEntry] = {}
        for name, entry in self.graph.passes.items():
            render_pass = self.passes.get(name)
            if render_pass is None:
                entries[name] = entry
                continue
            samplers = self._sampler_names(render_pass)
            bound = [
                uniform
                for uniform in samplers
                if _is_user_bound(render_pass.uniform_values.get(uniform))
            ]
            entries[name] = entry.model_copy(
                update={
                    "inputs": effective_inputs(entry, samplers, names, name, bound)
                }
            )
        return self.graph.with_passes(entries)
```

`render` binds `resolved_graph = self.effective_graph()` once, at the top, and uses it for
`plan_for_output`, for `plan_passes` on the early-out branch, and for the per-pass `entry` inside
the loop. `self.graph` keeps its meaning everywhere else: it is what is SAVED, so `graph.json`
still holds only what the user decided and never the resolved edges. That separation is the point
- a resolved edge written to disk would be indistinguishable from a chosen one the next time the
rule changed.

The invariants come along for free. `assert_plan_invariants` audits `plan.reads` against
`graph.passes[name].inputs.values()` of the graph it was given, so handing it the effective graph
keeps the audit honest rather than making it vacuous; a name default that closes a loop is
reported as a cycle by the same `_cycle_message` path an explicit edge would be; and an auto edge
to a pass that exists is by construction never "unresolved", so `unresolved_inputs` continues to
mean only "an explicit name points at a pass that is gone".

`self._graph_errors` is assigned from the effective graph's plan, which is what the strip's error
border and the copilot's error rows should show: an auto-wired cycle is a real defect the user
must see.

**Why a draw order that grows between frames breaks nothing.** Decision 2 says the order may
change from frame 1 to frame N as passes come online. The reason that is safe is structural, not
lucky, and it is worth writing down because the conclusion without the reason invites a future
reader to add a cross-frame guard that has nothing to guard. **Every invariant in `pass_graph.py`
is asserted per-call, on ONE plan, against the graph THAT plan was built from.**
`assert_plan_invariants` runs inside `plan_for_output`, recomputing its `expected` set from the
graph it was handed, so it audits whatever wiring that call saw; the draw-once check is
`len(plan.order) == len(set(plan.order))`, also per-call. **There is no cross-frame invariant to
violate.** Draw-once across a frame is enforced separately, by W-C's `drawn_frame == self._frame`
skip, which is per-frame by construction - so a pass whose auto edge first appears in frame k is
drawn once in frame k and skipped by the sweep in that same frame. A between-frame order change
is outside the domain of every assertion in the module, by design rather than by accident.

The SIX consumers of the resolution are therefore: `Document.render` (the binder), the planner
via `plan_for_output` on the effective graph, the gear (decision 6), the strip's two planner calls
(decision 11 - the tile order and the stale wash), `copilot/backend.py::_pass_views` (decision 7),
and `Document.has_feedback`, which gates the Clear canvas button and would otherwise miss a
`u_prev` pass wired by name alone. All six read the same `effective_graph`, which is what makes
"the renderer draws it" and "the strip says it is live" the same claim.

### 4. The media-bound exclusion lives at the resolution seam, not the render seam

`core.py`'s `render` binds `value = inputs.get(uniform.name, self.uniform_values.get(uniform.name))`
(`shaderbox/core.py:381` at `d2ade88`) - `inputs` wins. So an auto edge for a sampler holding a
user-bound `Image`/`Video` would silently replace that media with a pass's texture, and nothing
on screen would say why the PNG vanished. The exclusion is applied where the edge is COMPUTED, in
`effective_graph`'s `bound` argument, rather than by weakening the `inputs`-wins rule in
`core.py`: `core.py` must keep binding what the document handed it, because an EXPLICIT wire to a
media-bound sampler is a legitimate thing to do (the user overrode their own texture on purpose).
Only the automatic edge yields.

"User-bound" is not "is a `MediaWithTexture`": `Pass._default_uniform_value` seeds EVERY unbound
sampler with `Image(DEFAULT_IMAGE_FILE_PATH)`, so every sampler in a compiled pass holds one.
`media.py::is_default_image` exists for exactly this distinction and is what the predicate uses:

```python
def _is_user_bound(value: object) -> bool:
    return isinstance(value, MediaWithTexture) and not is_default_image(value)
```

`document.py` already imports both names' module, so this adds `is_default_image` and
`MediaWithTexture` to the existing `shaderbox.media` import.

**The seeding window, and why it is correct rather than merely harmless.** `compile()` does not
seed - `seed_uniform_values` runs at the top of `Pass.render`, inside `get_active_uniforms`'s own
compile branch, and at save. A pass compiled but never rendered therefore has an empty
`uniform_values`, contributes an empty `bound`, and every one of its samplers auto-wires. That is
the right answer, not a gap the ordering happens to cover: nothing is bound yet, so there is
nothing to steal, and the first render seeds before it binds. The window closes on the pass's own
first `render`, which is the same event that would have exposed the defect if there were one.
Manual step 7 exercises the case that matters - the OUTPUT pass of the Media Input example has
rendered and seeded long before the user adds a pass called `image`, so its `u_image` is protected
when the new pass appears.

### 5. `with_input("")` stores the empty string; `without_input` deletes the key

`PassGraph.with_input` today pops the key when `producer` is falsy, which makes an explicit
"(none)" byte-identical to never-decided. Under decision 1 those are two different answers, so:

```python
    def with_input(self, consumer: str, uniform: str, producer: str) -> "PassGraph":
        """Fill `consumer`'s `uniform` from `producer`, or store an explicit none when empty.

        `""` is a DECISION, not an absence: `effective_inputs` fills an absent key from the
        uniform's name (069 D9) and must not undo a user who chose nothing. `without_input`
        is how a key goes back to undecided.
        """
        entry = self.passes.get(consumer, PassEntry())
        inputs = {**entry.inputs, uniform: producer}
        return self.with_passes(
            {**self.passes, consumer: entry.model_copy(update={"inputs": inputs})}
        )

    def without_input(self, consumer: str, uniform: str) -> "PassGraph":
        """Forget `consumer`'s decision about `uniform`, so the name rule decides again."""
        entry = self.passes.get(consumer, PassEntry())
        if uniform not in entry.inputs:
            return self
        inputs = {u: src for u, src in entry.inputs.items() if u != uniform}
        return self.with_passes(
            {**self.passes, consumer: entry.model_copy(update={"inputs": inputs})}
        )
```

Both keep the `model_copy` funnel `with_passes` documents, for the reason it documents: a field
added to `PassEntry` tomorrow survives the edit.

The three existing edge-rewriting helpers need no change, and each is checked rather than assumed:

- `plan_passes` classifies a source `""` as `missing` (it is not in `known`, since `PassGraph`
  rejects a pass named `""`), so an explicit none shows up in `unresolved_inputs` if it ever
  reaches the planner. It does not: `effective_inputs` drops it, so the effective graph the
  planner sees carries no `""` value at all. `plan_passes` on the RAW graph - which nothing in
  the render path does after this wave - would report it, and `assert_plan_invariants` would not
  fire, because its `expected` set filters `source in graph.passes`.
- `project_session.py::_graph_without` keeps `src != removed`, and `""` is never a removed pass
  name, so an explicit none survives a sibling's deletion.
- `project_session.py::_graph_renamed` maps `src == old`, and `""` never equals a valid pass name,
  so an explicit none survives a rename.

`ProjectSession.wire_pass_input` gains a sibling verb rather than overloading its empty string:

```python
    def unwire_pass_input(self, document_id: str, consumer: str, uniform: str) -> str:
        """Return `consumer`'s `uniform` to undecided, so the name rule fills it again."""
```

It validates the document and the consumer the same way, calls `graph.without_input`, saves, and
returns `""` or an error string - the shape every one of the graph verbs has. `wire_pass_input`
keeps its signature and its docstring's closed-set claim; only the meaning of its empty
`producer` changes, from "delete the key" to "store an explicit none", and its docstring says so.

Its validation is unchanged, including the line that carries the change:
`if producer and producer not in document.passes: return f"no such pass '{producer}'"`. The
`if producer` guard is what admits `""` today and what admits it after this wave - the same line
doing a different job, admitting an explicit none rather than a deletion request. A reader
reworking the docstring sits one line above it, which is why it is named here.

### 6. The gear's combo has three kinds of item, and each stores something different

`_draw_inputs` in `popups/pass_settings.py` builds one combo per sampler. Today: two kinds of
item, and an index that falls to `0` for both an absent key and a stored `""`, so the two states
are indistinguishable on screen (`choices.index(current) if current in choices else 0`, with
`current = entry.inputs.get(uniform, "")` and `""` never in `choices`).

The new item list, per sampler, in this order:

| Position | Label | Stores | Shown when |
|---|---|---|---|
| 0 | `auto: <x>` or `auto: none` | deletes the key (`without_input`) | always |
| 1 | `(none)` | `""` (`with_input(..., "")`) | always |
| 2..N | each pass name, sorted | that name (`with_input`) | always |

`<x>` is `effective_inputs`'s answer for this sampler with the key removed - the pass the name
rule would pick - so `auto: df` on a `u_df` beside a pass `df`, and `auto: none` on a `u_df` with
no such pass, on a media-bound sampler, or on a pass whose program has not compiled yet.
`choices` is built to match that table position for position, because the read and the write
index the same list:

```python
        auto = effective_inputs(
            entry.model_copy(update={"inputs": {u: s for u, s in entry.inputs.items() if u != uniform}}),
            [uniform], names, name, bound,
        ).get(uniform, "")
        choices = [f"auto: {auto or 'none'}", _UNWIRED, *sorted(document.passes)]
```

and the write is by POSITION against exactly that list: 0 calls `unwire_pass_input`, 1 calls
`wire_pass_input(..., "")`, and anything else calls `wire_pass_input(..., choices[picked])`. The
`auto` value is `effective_inputs`'s answer for this sampler with its key removed, which is what
the rule WOULD pick, whatever is stored now.

A pass name can never collide with either synthetic label: `_PASS_NAME_RE` in
`project_session.py` is `^[A-Za-z_][A-Za-z0-9_]*$`, which admits no colon, no space and no
parenthesis, so neither `auto: df` nor `(none)` is a reachable pass name. That is what makes
indexing this mixed list safe, and it is the reason the last branch below can test membership
against it without a false hit.

Selection is by the STORED state, not by the effective one:

```python
        stored = entry.inputs.get(uniform)
        if stored is None:
            index = 0
        elif stored == "":
            index = 1
        else:
            index = choices.index(stored) if stored in choices else 1
```

The last branch is the stale case - an explicit name whose pass is gone. It selects `(none)`,
which is what such a wire renders as, and the strip's own error path is not involved because a
stale explicit name is not a graph ERROR, it is an unresolved input that reads black. Picking any
item then rewrites the key to something valid, so the stale name cannot survive a visit to the
gear.

Three states, three distinct readings, and the round trip holds: pick `auto: df`, the key is
deleted and the combo shows `auto: df`; pick `(none)`, `""` is stored and the combo shows
`(none)`; pick `df`, `"df"` is stored and the combo shows `df`. The label `auto: <x>` carries a
derived value inside the control rather than in the label column, which is what § 2 of the imgui
skill requires of a derived value, and it is two words plus a name - inside the budget the prose
gate scores, and not a string the gate's AST walk can see in any case (it is an f-string built
per sampler, listed in a `combo`'s items, not in any helper the gate's domain covers).

The gear needs the same `bound` information the renderer has, so `_draw_inputs` computes it from
`render_pass.uniform_values` with the same `_is_user_bound` predicate. That predicate lives in
`document.py` beside its only other caller; the gear imports it from there, which is allowed -
`popups/` already imports `app.py`, which imports `document.py`.

### 7. The copilot's pass view resolves the same way

`copilot/backend.py::_pass_views` builds the `inputs:` rows the working set shows, reading
`entry.inputs` raw. Left alone, the copilot would be told a sampler reads BLACK while the
renderer fills it from a pass - a false fact on the channel the model reads, which § 4 of the
copilot skill names as the worst kind of prompt defect. So it reads the effective graph:

```python
            effective = document.effective_graph().passes.get(name)
            wired = effective.inputs if effective is not None else {}
```

and the row text distinguishes the two silent states, since the model can act on the difference:

- filled: `u_df <- df` (unchanged; whether it was chosen or auto is not the model's business,
  and adding "auto" would invite it to reason about the mechanism instead of the picture)
- explicit none in `entry.inputs`: `u_df <- (none; reads BLACK)`
- nothing at all: `u_df <- (nothing; reads BLACK)` (unchanged text)

The distinction costs one branch and answers the question a model would otherwise ask by editing:
whether an empty input is a choice it should respect or a gap it should fill.

`edit_shader` needs no change at all, which is the pay-off the parent names: a copilot that adds
`uniform sampler2D u_blur;` to a pass gets it wired on the next render, because resolution happens
at render time and nothing in the edit path stores an edge. The same holds for `write_shader`,
`watch.py`'s hot reload, and a user editing the file in the inline editor - all three land new
sampler names with no graph write, and all three are correct by construction under this design.
That is the whole reason the parent rejected a stored-edge shape.

### 8. `project_session.add_pass` stores no default wiring

`add_pass` today writes `PassEntry()` - an empty entry, every sampler undecided. Under decision 1
that is already the right thing, and the correct change is **no change**: an empty entry means
"the name decides", which is what D9 asks for. The parent's Files-touched line names
`project_session.py` for "default wiring", and this wave's answer is that the default wiring
needs no code there, because it is a resolution rule and not a stored edge. What
`project_session.py` does gain is `unwire_pass_input` (decision 5).

`add_pass` calls `render_pass.compile()` eagerly, so a pass added through the UI is online in the
frame it is created and its auto edges resolve immediately. That is not a change either; it is
worth stating because it means the "black for a frame" cost of decision 2 is paid only on a
document REOPEN, never on the add-pass path the tutorial walks.

### 9. The naming rename: exact-token, per file, with the collision check performed

D9: an input uniform is named after the pass it reads, `u_<pass>`; feedback is `u_prev`.

**Collision check.** Under D9 a collision is two inputs of ONE pass reading the SAME source, which
would want the same name. Both examples' `graph.json` were read at `d2ade88` and enumerated: RC's
`cascade` reads `paint`, `df` and itself; `composite` reads `cascade` and `paint`; every other
pass reads at most one source. Bloom's `composite` reads `scene`, `blur` and `trail`; `trail`
reads `scene` and itself; every other pass reads one. **No pass in either example reads one source
through two samplers, so no rename collides.** Two DIFFERENT passes each naming a `u_scene` (Bloom's
`bright`, `trail` and `composite` after the rename) is fine - they are separate programs with
separate uniform namespaces.

**The rename table.** Every token, per file. `graph.json` keys move with the shaders.

| Example | File | Old token | New token | Sites |
|---|---|---|---|---|
| RC `77a84d27…` | `passes/seed.frag.glsl` | `u_scene` | `u_paint` | decl L17, read L20 |
| RC `77a84d27…` | `passes/cascade.frag.glsl` | `u_scene` | `u_paint` | decl L25, read L44 |
| RC `77a84d27…` | `passes/composite.frag.glsl` | `u_scene` | `u_paint` | decl L14, read L22 |
| RC `77a84d27…` | `passes/composite.frag.glsl` | `u_light` | `u_cascade` | decl L13, read L18 |
| RC `77a84d27…` | `graph.json` | `seed.inputs.u_scene` | `u_paint` | 1 key |
| RC `77a84d27…` | `graph.json` | `cascade.inputs.u_scene` | `u_paint` | 1 key |
| RC `77a84d27…` | `graph.json` | `composite.inputs.u_scene` | `u_paint` | 1 key |
| RC `77a84d27…` | `graph.json` | `composite.inputs.u_light` | `u_cascade` | 1 key |
| Bloom `1c4f8a20…` | `passes/bright.frag.glsl` | `u_src` | `u_scene` | decl L10, read L17 |
| Bloom `1c4f8a20…` | `passes/blur.frag.glsl` | `u_src` | `u_bright` | decl L9, reads L16 L23 |
| Bloom `1c4f8a20…` | `passes/trail.frag.glsl` | `u_src` | `u_scene` | decl L10, read L20 |
| Bloom `1c4f8a20…` | `passes/composite.frag.glsl` | `u_lit` | `u_scene` | decl L8, read L18 |
| Bloom `1c4f8a20…` | `passes/composite.frag.glsl` | `u_glow` | `u_blur` | decl L9, read L19 |
| Bloom `1c4f8a20…` | `graph.json` | `bright.inputs.u_src` | `u_scene` | 1 key |
| Bloom `1c4f8a20…` | `graph.json` | `blur.inputs.u_src` | `u_bright` | 1 key |
| Bloom `1c4f8a20…` | `graph.json` | `trail.inputs.u_src` | `u_scene` | 1 key |
| Bloom `1c4f8a20…` | `graph.json` | `composite.inputs.u_lit` | `u_scene` | 1 key |
| Bloom `1c4f8a20…` | `graph.json` | `composite.inputs.u_glow` | `u_blur` | 1 key |

Already conforming, renamed nowhere: RC `jfa.u_seed`, `jfa.u_prev`, `df.u_jfa`, `cascade.u_df`,
`cascade.u_prev`; Bloom `composite.u_trail`, `trail.u_prev`.

**The four prefix traps.** A blind substring replace corrupts all four; the rename is exact-token
(whole identifier, word-boundary) and per-file:

1. `8d454b7b…/passes/main.frag.glsl` declares `u_light_ambient`, `u_light_sky_key`,
   `u_light_moon_key`, `u_light_cool_color`, `u_light_warm_color` (L56-60, read L460-461, named
   in a comment at L19). A `u_light` → `u_cascade` replace across the examples tree destroys five
   uniforms whose tuned values persist in that document's `document.json`.
2. `0b0d16bb…/passes/main.frag.glsl` declares `u_glow_strength` and `u_glow_radius` (L12-13, read
   L164, L167). A `u_glow` → `u_blur` replace destroys both.
3. **Inside the RC example itself**, `passes/paint.frag.glsl` declares `u_light_radius` (L20, read
   L35). This one is the trap the parent's warning does not name: limiting the replace to "the two
   multi-pass examples" is NOT sufficient protection, because the collision is inside one of them.
   Only exact-token matching saves it.
4. `1c4f8a20…/passes/composite.frag.glsl` declares `u_trail_mix` (L13, read L20) beside the
   `u_trail` sampler. `u_trail` is not renamed, so nothing moves here - but a future re-run of the
   same sweep would hit it, and it belongs in the record.

**The six comments that become wrong.** Bloom's shaders carry comments naming the source pass:
`bright.frag.glsl:9`, `blur.frag.glsl:9` (`// filled by \`bright\``), `trail.frag.glsl:10`
(`// filled by \`scene\``), `composite.frag.glsl:8-10` (three of them), and
`trail.frag.glsl:11`'s `// filled by \`trail\` -- itself, i.e. the previous frame`. Under D9 the
uniform name says what the comment says, so the first six go; `u_prev`'s stays, because "itself,
i.e. the previous frame" is the one fact the name does not carry. RC's `jfa.frag.glsl:7` and
`cascade.frag.glsl:10` prose comments mention `u_prev` and stay unchanged.

`bright.frag.glsl:9` is the one an inline-pattern grep misses, and it is the reason the census
here was taken with `grep -rn "illed by"` rather than a pattern anchored to the declaration line:
it is a WHOLE-LINE comment sitting above the decl, with a capital F, and it carries a second
clause the other five do not - `// Filled by \`scene\` -- see the pass list's inputs, or
graph.json. An unfilled input reads black.` The trailing sentence is engine prose that the Help
panel's new paragraph and the copilot prompt's pass block both now carry, so it goes with the
line rather than being kept as the one surviving comment.

**`document.json` needs no edit in either example.** Both were read: neither carries a `uniforms`
entry keyed by any renamed name - RC's and Bloom's tuned values are on scalar uniforms
(`u_exposure` and friends), and a sampler's value is skipped at save when it holds the default
image. The rename therefore strands nothing. A tuned value that DID collide would be hand-edited
per the no-migration rule, not migrated.

### 10. `graph.json` hand-edits, and what does not need one

Two edits, both by hand, no migration code, per `CLAUDE.md`'s hard rule and
`conventions.md ## Design decisions`:

- **Both examples' `graph.json`**: the eighteen key renames in the table above. Nothing else moves
  - no `""` value is added to either, because every sampler in both is either explicitly wired or
  conforms to the name rule, and there is no sampler in either example that must read black.
- **`projects/dev/documents/`**: **no edit needed, verified rather than assumed.** Both dev
  documents (`e7e00c46…`, `ec926580…`) are single-pass with one pass called `main` and
  `"inputs": {}`. `e7e00c46…` declares `uniform sampler2D u_video`, and there is no pass called
  `video`, so `effective_inputs` returns nothing for it and `core.py`'s
  `inputs.get(name, uniform_values.get(name))` falls through to the seeded default image exactly
  as it does today. The wave still runs `git add projects/dev` for whatever sandbox drift the run
  produces, per the standing rule.
- **The five single-pass examples**: no edit. `73ea2431…` (Media Input) is the interesting one and
  was checked: it is single-pass (`main`) with `u_image` and `u_video` bound to real media in its
  `document.json`. Neither name matches a pass, so the media exclusion of decision 4 is not what
  protects it here - the absence of a pass called `image` is. The exclusion protects the general
  case (a document with a pass named `image` beside a bound `u_image`), and this example is the
  nearest thing the repo ships to that shape rather than an instance of it.

### 11. The strip tune, and the strip plans the effective graph

`widgets/pass_list.py::_draw_pass_tile`:

- The `wired` / `sublines` computation goes, and with it the `PassEntry` import if nothing else in
  the module uses it (`_strip_order` uses `PassGraph` and `plan_passes`; check at edit time).
  `preview_cell` is called with no `sublines` argument, so it defaults to `()`.
- The `"has compile errors"` subline goes. The error state is already carried by
  `border = COLOR.STATE_ERROR if errors else ...`, which is drawn before and independent of the
  sublines, so removing the text loses no signal - the border was always the primary marker and
  the line was a second spelling of it.
- `preview_cell` LOSES its `sublines` parameter, with its docstring paragraph, its term in
  `footer_h` and its render loop. `pass_list.py` was its only caller in `shaderbox/`, so the
  parameter would otherwise be surface with no consumer that still has to be taught and
  maintained; 070's graph view is not a `preview_cell`. `tests/test_ui_prose_budget.py` derives
  its scored domain from the live signature, so the `sublines` row leaves with the parameter
  rather than needing an edit. (This reverses the draft's decision to keep it; round 2's code F4
  is why.)

**`draw` and `_strip_order` both plan the EFFECTIVE graph, not `document.graph`.** This is the
half of decision 3 the strip needs, and it is a real consumer of `effective_inputs` rather than a
removal: `resolved = document.effective_graph()` once at the top of `draw`, passed to
`_strip_order(document.passes, resolved)` and to `evaluation_order(resolved, output)`. Both calls
decide what the user sees, and both are wrong on the raw graph:

- **The stale wash.** `live` is computed at `pass_list.py:158` from `evaluation_order`, and
  `name not in live` is handed to `_draw_pass_tile` as `stale`, which `preview_cell` renders as a
  desaturating wash plus a dimmed footer. On the raw graph the worked example's frame 3 gives
  `live == {"edge"}`, because the raw entry for `edge` has no inputs at all, so `paint`, `seed`,
  `jfa` and `df` are all washed grey while the renderer draws all four every frame. That wash is
  the signal W-C's decision 7 spent a section making honest ("a pass outside the output chain
  shows a real picture under a grey wash"), and planning the raw graph here makes it lie.
- **The tile order.** `_strip_order` documents itself as "producers left of consumers". On the raw
  graph a document wired entirely by name has no edges, so `plan_passes` returns sorted-name order
  and `paint, seed, jfa, df, cascade, composite` becomes `cascade, composite, df, jfa, paint,
  seed`. Both shipped examples keep explicit edges after the rename and are unaffected; a document
  a user builds by name alone gets the alphabetical strip.

`_strip_order`'s signature is unchanged (it already takes a `graph` parameter), so
`tests/test_pass_verbs.py`'s import of it stays valid; only the argument the caller passes moves.

**`tests/test_ui_prose_budget.py`'s `_UNMEASURABLE` entry survives the removal.**
`("shaderbox/widgets/pass_list.py", "_draw_pass_tile")` reaches the gate's collector through two
`preview_cell` arguments, `footer=name` and `sublines=sublines`, both unresolvable `ast.Name`
nodes. Dropping the `sublines=` argument leaves `footer=name` - a function PARAMETER, so there is
no assignment in the body for `_resolve` to read, so it stays unmeasurable - and
`test_every_unmeasurable_entry_still_names_a_real_site` stays green. The entry's written REASON
does not survive: it reads "the pass's name and wiring", and the wiring half is what this wave
removes, so it becomes "the pass's own name" in the same commit. (This corrects the reasoning an
earlier draft of this decision gave: the risk was never the derived rows, which come from
`ui_primitives`' signatures and do not move, but the allowlist entry whose reason the removal
falsifies. This repo's rule is that an allowlist entry carries a reason that stays true, and a
reviewer of 070 would otherwise read a reason describing a state the code left.)

**The spacing, re-measured.** `preview_cell` computes
`cell_h = cell_w + (line_h if footer else 0) + line_h * len(sublines)`. With sublines gone, the
card is `SIZE.PASS_THUMB` (112) plus one text line, and it shrinks by one to four lines depending
on how many inputs the pass had - so the tallest tile in a row no longer sets the row height from
its wiring. Two consequences and one non-change:

- Horizontal: `step = SIZE.PASS_THUMB + SPACE.MD` and `same_line(spacing=SPACE.MD)` already agree,
  and neither depends on the card height. Unchanged.
- Vertical between wrapped rows: imgui's `style.item_spacing.y`, set to `SPACE.SM` (4) in
  `theme.py`. With the cards now uniform in height, 4px between rows reads as a grid rather than
  as slack, which is what the maintainer's "tune the current visuals" asks for. Unchanged, and the
  reason it is unchanged is that the sublines were what made it look wrong.
- Below the strip: the `imgui.dummy((0, SPACE.SM))` before `add pass` stays. It separates the
  strip from a button, which is a zone break, not slack under a footer.

`SIZE.PASS_THUMB` stays 112 - the maintainer chose that number in 065 round two, and #19's option
B changes what a tile SHOWS, not how big it is.

### 12. The rule's two written homes

**Help panel**, `help_content.py`, the `pass_settings` section. One paragraph, inserted first in
the body so it precedes the target controls:

> An input uniform is named after the pass it reads: `u_blur` reads the pass called `blur`, and
> the gear fills it in for you. `u_prev` is the one exception - it reads this pass's own previous
> frame. Pick a different source, or none, in the gear.

Four sentences, in a Help body, where § 2 of the imgui skill exempts documentation from the word
budget ("Anything longer than the budget is documentation and goes to the Help panel or the
tutorial, where a reader chose to read"). `tests/test_help_content.py` runs over the sections; the
addition is body prose, not a new key, so no test structure changes.

**Copilot prompt**, `copilot/prompt.py`, the pass block. The existing sentence ends "an unfilled
one reads BLACK, it is not an error." One sentence is appended:

> A sampler named `u_<pass>` is filled from that pass automatically, so naming an input after the
> pass it reads is how you wire it; `u_prev` reads the pass's own previous frame.

It belongs in the STATIC block and nowhere else, per the copilot skill's home rule: it governs
what the model does BEFORE acting (choosing a uniform name while writing a shader), and a rule
parked on a tool result would be absent on the turn a shader is first written. It is one clause
of fact, on the tier that already carries the pass-block facts, so the prefix cache is unaffected
by construction - the block's volatility rank does not change.

## Worked example: a fresh pass declaring `u_df` beside a pass `df`

The Radiance Cascades example is open; its passes are `paint`, `seed`, `jfa`, `df`, `cascade`,
`composite`, and `composite` is the output. The user clicks `add pass`, types `edge`, and commits.

**Frame 0 - the add.** `add_pass` writes `passes/edge.frag.glsl` from `PASS_STUB`, calls
`render_pass.compile()` (so `edge` is online immediately), stores `PassEntry()` with empty
`inputs`, and saves. W-C's D10 then activates it: the editor opens `edge`, the graph output
becomes `edge`, and the gear opens on it. The stub declares no sampler, so `_sampler_names`
returns `[]`, `effective_inputs` returns `{}`, and the gear's Reads section shows
`no sampler2D uniforms`. The viewer shows opaque black, which is what the stub draws.

**Frame 1 - the user types `uniform sampler2D u_df;` in the editor and saves.**
`watch.py::_reload_pass_if_changed` sees the mtime change, calls `render_pass.release_program(text)`,
and updates the source. It does not compile, does not see the graph, and stores no edge - which
is precisely why the resolution rule is a rule and not a stored edge. `edge.program` is now None,
so `_sampler_names(edge)` returns `[]` and the effective graph carries no auto edge for it. The
tile is black.

**Frame 2 - the first render after the reload.** `edge` is the output, so `Document.render` draws
it whether or not the sweep elects it; `Pass.render` compiles per call while the program is
missing. Once the program exists, `_sampler_names` returns `["u_df"]` on the NEXT call to
`effective_graph`, which is at the top of the next `render`. So frame 2 draws `edge` with an
unresolved `u_df` - black, one frame.

**Frame 3 - the edge appears.** `effective_graph` sees `u_df` as an absent key on a compiled pass,
`_auto_source` strips `u_` to `df`, a pass called `df` exists, and the edge `edge.u_df <- df` is
in the effective graph. `plan_for_output(effective, "edge")` therefore orders `paint, seed, jfa,
df, edge` - the whole ancestor chain of `df`, which none of the previous frames drew because
`edge` had no inputs. The loop binds `df`'s canvas texture into `u_df`, and the tile shows the
distance field. **The user never opened the gear.** Total cost: `u_df` is black for frames 1 and 2
after the save, then correct forever.

**What the gear shows in each of the three states.** The user opens `edge`'s gear on frame 4:

- **Absent key (where they are now).** Items: `auto: df`, `(none)`, `cascade`, `composite`, `df`,
  `edge`, `jfa`, `paint`, `seed`. Selected: index 0, `auto: df`. `graph.json` holds
  `"edge": {"inputs": {}, ...}`.
- **They pick `(none)`.** `wire_pass_input(doc, "edge", "u_df", "")` stores `""`; the combo now
  shows `(none)` at index 1. `effective_inputs` returns `{}` for `edge`, so the next render binds
  the 1x1 black texture and the tile is black. Reopening the document reloads `{"u_df": ""}` and
  it is STILL black - the rule does not re-wire what the user un-wired. `graph.json` holds
  `"inputs": {"u_df": ""}`.
- **They pick `jfa`.** `wire_pass_input(doc, "edge", "u_df", "jfa")` stores `"jfa"`; the combo
  shows `jfa`. The name says `df` and the wire says `jfa`, and the wire wins - an explicit
  decision always beats the default. The tile shows the jump-flood buffer. `graph.json` holds
  `"inputs": {"u_df": "jfa"}`.
- **Back to auto.** They pick `auto: df` again; `unwire_pass_input` deletes the key and the
  entry is `{"inputs": {}}` once more.

## Files touched

- **`shaderbox/pass_graph.py`** - `effective_inputs` + its `_auto_source` helper;
  `with_input` stores `""`; new `without_input`. No new imports beyond `collections.abc`.
- **`shaderbox/document.py`** - `_sampler_names`, `_is_user_bound`, `effective_graph`; `render`
  plans and binds from the effective graph. Imports `effective_inputs` from `pass_graph`,
  `is_default_image` + `MediaWithTexture` from `media`, `GL_SAMPLER_2D` from `OpenGL.GL`.
- **`shaderbox/project_session.py`** - `unwire_pass_input`; `wire_pass_input`'s docstring says
  what an empty producer now means. `add_pass` unchanged (decision 8).
- **`shaderbox/popups/pass_settings.py`** - `_draw_inputs`'s three-state combo; the `bound`
  computation.
- **`shaderbox/widgets/pass_list.py`** - `_draw_pass_tile` drops `wired` / `sublines`; `draw`
  binds `document.effective_graph()` once and feeds it to both planner calls; `_strip_order`'s
  signature is unchanged (`tests/test_pass_verbs.py` imports it), only its argument moves.
- **`shaderbox/copilot/backend.py`** - `_pass_views` reads the effective graph; the explicit-none
  row text.
- **`shaderbox/copilot/prompt.py`** - one sentence in the pass block.
- **`shaderbox/help_content.py`** - one paragraph in the `pass_settings` section.
- **`shaderbox/resources/document_examples/77a84d27-…/`** - `graph.json` (4 keys) +
  `passes/seed.frag.glsl`, `passes/cascade.frag.glsl`, `passes/composite.frag.glsl`.
- **`shaderbox/resources/document_examples/1c4f8a20-…/`** - `graph.json` (5 keys) +
  `passes/bright.frag.glsl`, `passes/blur.frag.glsl`, `passes/trail.frag.glsl`,
  `passes/composite.frag.glsl`.
- **Tests** - `tests/test_pass_graph.py`, `tests/test_document_graph.py`,
  `tests/test_pass_verbs.py`, `tests/test_copilot_passes.py`, and a new
  `tests/test_default_wiring.py`.
- **`ai_docs/conventions.md`** - the D9 naming rule and the resolution posture join the pass-graph
  entry (one bullet, phrased as what the mechanism IS, not as a wave's state).
- **`ai_docs/dev_flow.md`** - three module-map edits. The `widgets/pass_list.py` entry says the
  strip orders by `plan_passes` and washes off-plan tiles; both now read the EFFECTIVE graph, and
  the entry is the sentence a reader consults to learn which graph the strip plans, so it says
  which. `pass_graph.py`'s edit-verb list (`with_passes` / `with_input` / `with_target` /
  `with_output`) gains `without_input`. `pass_settings.py`'s "closed-set combos over the
  document's own pass names" gains the two synthetic items - the set is still closed, the
  parenthetical narrows.
- **`tests/test_ui_prose_budget.py`** - the `_UNMEASURABLE` reason for
  `("shaderbox/widgets/pass_list.py", "_draw_pass_tile")` becomes "the pass's own name"
  (decision 11).

Not touched, each checked: `tests/shader_lib_api_lock.json` and `tests/test_examples_resolve.py`
(the lock pins shipped `SB_*` signatures, and the rename touches no `SB_` name - see § Verified
premises); `projects/dev/documents/*/graph.json`; the five single-pass examples; `watch.py`;
`core.py`; `theme.py`; `ui_primitives.py`.

## Tests

Each names the falsifier - the input that goes red under the bug it exists for.

### `tests/test_default_wiring.py::test_effective_inputs_over_every_state`

The nine-cell matrix the parent asks for: `(absent, "", explicit)` × `(pass exists, not)` ×
`(media-bound)`. Table-driven, GL-free, no context.

| stored | `u_df` vs passes | bound | expected |
|---|---|---|---|
| absent | `df` exists | no | `{"u_df": "df"}` |
| absent | no `df` | no | `{}` |
| absent | `df` exists | yes | `{}` |
| `""` | `df` exists | no | `{}` |
| `""` | no `df` | no | `{}` |
| `""` | `df` exists | yes | `{}` |
| `"jfa"` | `df` exists | no | `{"u_df": "jfa"}` |
| `"jfa"` | no `df`, `jfa` exists | no | `{"u_df": "jfa"}` |
| `"jfa"` | `df` exists | yes | `{"u_df": "jfa"}` |

Plus `u_prev` on a pass `cascade` resolving to `"cascade"` and not to a pass called `prev`.

Falsifier: make the `""` branch fall through to the name rule (the single change from the old
`with_input` semantics) and rows 4 and 6 go red. Make `bound` advisory and rows 3 and 6 go red.

### `tests/test_default_wiring.py::test_the_planner_orders_an_auto_edge`

Build a `PassGraph` of `a` (no inputs) and `b` (empty `inputs`), pass `b`'s samplers as
`["u_a"]` through `effective_inputs`, plan the resulting graph for output `b`, and assert the
order is `["a", "b"]` and `plan.reads["b"] == {"a"}`. Then a second graph where the auto edge
closes a loop (`a` declaring `u_b`, `b` declaring `u_a`) and assert `plan_passes` returns a cycle
error naming both.

Falsifier: hand `plan_for_output` the RAW graph instead of the effective one and the order is
`["b"]` with `a` never drawn - which is the exact defect the parent's "the planner must see auto
edges to order the draw and to detect cycles" is about. The cycle half goes red the same way: the
raw graph has no edges at all, so no cycle is reported.

### `tests/test_default_wiring.py::test_u_df_beside_df_renders_without_the_gear`

GL. Copy the RC example to a tmp dir, add a pass file `edge.frag.glsl` declaring
`uniform sampler2D u_df;` and writing `texture(u_df, vs_uv)`, load the document, set the output to
`edge`, render enough frames for the sweep to bring it online, and assert two things: the
effective graph carries `edge.u_df -> df`, and the rendered texture is not uniformly black.
`graph.json` is never written and the gear is never opened.

Falsifier: revert `effective_graph` to return `self.graph` and the assertion on the edge goes red
immediately; revert only the binding half (plan from effective, bind from raw) and the pixel
assertion goes red while the edge assertion passes, which is why both are asserted.

### `tests/test_default_wiring.py::test_a_stored_empty_string_stays_black_across_a_reload`

GL. Same document. `wire_pass_input(doc, "edge", "u_df", "")`, save, reload from disk, assert
`graph.passes["edge"].inputs == {"u_df": ""}` on the reloaded document, assert
`effective_graph().passes["edge"].inputs == {}`, render, and assert the pass is black.

Falsifier: restore `with_input`'s `inputs.pop(uniform, None)` and the reloaded entry is `{}`, the
name rule fills `u_df` from `df`, and the black assertion goes red. This is the single test that
pins the `""`-stores-explicit-none half of the design, and it must go through disk - an in-memory
assertion would pass under a `with_input` that stored `""` but a `model_dump` that dropped it.

### `tests/test_default_wiring.py::test_an_uncompiled_pass_contributes_no_auto_edge_and_compiles_nothing`

GL. Load Bloom fresh (066 D1: nothing compiles at load), assert every `program is None`, call
`effective_graph()`, and assert (a) it equals the raw graph entry-for-entry, and (b) every
`program` is STILL None. Then render one frame and assert the passes the output chain drew now
contribute their samplers.

Falsifier: swap `_sampler_names`'s `render_pass.program` read for
`render_pass.get_active_uniforms()` and (b) goes red on the first call - which is the 066 D1
inversion the parent's design exists to avoid, and the reason this test asserts on compile state
rather than on edges alone.

### `tests/test_default_wiring.py::test_the_gear_shows_three_distinct_states`

Through a real imgui frame (the rig `tests/test_pass_settings_layout.py` uses). Drive
`_draw_inputs` on a pass with one sampler in each of the three stored states and capture the
combo's selected index and item list. Assert index 0 with a leading `auto: ` item on absent, index
1 on `""`, and the pass's index on an explicit name; assert the item at 0 reads `auto: df` when a
pass `df` exists and `auto: none` when it does not.

Falsifier: today's `choices.index(current) if current in choices else 0` scores absent and `""`
identically at 0, so the `""` row goes red under it. This is the finding round 3 raised, and it is
unobservable without reading the index.

### `tests/test_examples_resolve.py::test_every_example_input_uniform_names_its_source`

New, GL-free, beside the existing example checks. For every shipped example's `graph.json`, assert
every `inputs` key either equals `f"u_{source}"` or is exactly `"u_prev"` with `source` equal to
the consuming pass. This is the D9 gate: it fails on a future example that reintroduces a role
name, and it fails today until the rename lands.

Falsifier: revert one `graph.json` key (say Bloom's `composite.u_scene` back to `u_lit`) and it
goes red naming that pass and that key.

### `tests/test_default_wiring.py::test_every_multi_pass_example_compiles_every_pass`

**What the rename's safety net actually is, corrected.** An earlier draft of this section claimed
`test_examples_resolve_clean` catches a shader whose declaration was renamed but whose read was
not. **It does not.** That test runs `resolve_usage`, the `SB_*` INCLUDE resolver, and never
invokes GLSL: `texture(u_src, ...)` against a decl renamed to `u_scene` is not an `SB_` name, the
resolver is indifferent to it, and the test passes. The failure surfaces only at link time. The
test stays in the suite and this wave runs it as a regression check, but it is not this rename's
gate.

What genuinely covers the two examples is uneven, so the gap gets a test:

- **RC is covered.** `tests/test_radiance_cascades_example.py::test_every_pass_compiles_and_the_graph_is_clean`
  really compiles all six passes; a half-renamed shader fails to link and it goes red. A
  `graph.json` left unrenamed is caught not by its `graph_errors` assertion (a stale edge lands in
  `unresolved_inputs`, not `graph_errors`) but by the two pixel assertions in the same module, since
  the pass then reads black.
- **Bloom is not.** `tests/test_lazy_compile.py::test_every_pass_renders_once_within_n_frames`
  drives every Bloom pass through `Document.render`, but asserts only `first_render_done` and the
  no-double-election property - and W-C writes those stamps on ATTEMPT, deliberately, so a Bloom
  pass that fails to compile is still stamped and the test stays green. A half-renamed
  `bright.frag.glsl` goes red in nothing today.
  (`tests/test_example_library.py::test_shipped_examples_read_clean_without_joining_working_set`
  does not help either: `read_shaders` compiles `document.render_pass` only, so Bloom's four
  non-output passes are never touched.)

So this wave adds the assertion. Load RC and Bloom, render `len(document.passes)` sweep frames
each, and assert `compile_unit.errors == []` for every pass in both.

Falsifier: rename a declaration and leave its `texture()` read on the old name, and it goes red
naming the pass and carrying the linker's undeclared-identifier message. It is red for exactly one
reason: it asserts on compile errors, not on pixels, so a shader that links but draws the wrong
thing is a different test's job.

**And what is NOT touched:** `test_shader_lib_api_lock` in the same module as the resolve test
pins the shipped `SB_*` signatures against `tests/shader_lib_api_lock.json`. It reads the seed
shader library, not the examples, and the rename touches no `SB_` identifier, so the lock file is
not regenerated by this wave.

### `tests/test_pass_verbs.py::test_unwiring_is_an_empty_producer` - rewritten in place

Today it asserts `inputs == {}` after `wire_pass_input(..., "")`, on the live document and after a
reload. Both assertions invert to `{"u_src": ""}`, and a third is added: after
`unwire_pass_input(...)` the entry is `{}` again, live and after a reload. The test's name becomes
`test_unwiring_stores_an_explicit_none_and_unwire_forgets_it`.

Falsifier: it is its own falsifier - it goes red the moment `with_input` changes, which is the
point of editing it rather than deleting it.

### `tests/test_copilot_passes.py::test_a_pass_s_wiring_is_shown_including_what_is_unwired` - extended

Its existing `"u_src <- (nothing; reads BLACK)"` assertion moves to the explicit-none text
(`"u_src <- (none; reads BLACK)"`, since the test reaches that state through
`wire_pass_input(..., "")`), and a new case asserts that a pass declaring `u_scene` beside a pass
`scene`, with no stored edge, shows `u_scene <- scene`.

Falsifier: leave `_pass_views` reading `entry.inputs` raw and the new case shows
`(nothing; reads BLACK)` while the renderer fills it - a false fact in the working set, which is
the defect the change exists for.

### `tests/test_pass_verbs.py::test_the_strip_draws_no_sublines`

Through the imgui frame rig, drive `pass_list.draw` on a two-pass document with a wired input and
assert `preview_cell` was called with `sublines == ()` for every tile - monkeypatch
`pass_list.preview_cell` and capture its kwargs, the same shape
`tests/test_pass_settings_layout.py` uses to capture `_draw_body`.

Falsifier: restore the `sublines = [f"{uniform} <- {src}" ...]` line and it goes red naming the
tile. Asserting on the ARGUMENT rather than on a rendered rect is what makes it fail for exactly
one reason: a geometry assertion would also move if `SIZE.PASS_THUMB` or the font changed.

### `tests/test_pass_verbs.py::test_an_auto_wired_ancestor_is_not_washed_stale`

The stale half of F1, on the same rig. A two-pass document `a` (no inputs) and `b` declaring
`uniform sampler2D u_a;` with NO stored edge, output `b`. Render enough frames for both to
compile, then drive `pass_list.draw` with `_draw_pass_tile` monkeypatched to capture its `stale`
argument per pass. Assert `stale is False` for `a`.

Falsifier: plan the raw graph in `draw` (that is, revert `evaluation_order(resolved, output)` to
`evaluation_order(document.graph, output)`) and `a` comes back `stale=True` - the exact defect,
the strip telling the user a pass is dead while the renderer draws it every frame. It fails for
one reason because it asserts on the boolean the wash is computed from, not on pixels.

### `tests/test_pass_verbs.py::test_the_strip_orders_a_name_wired_document_topologically`

The order half of F1, and GL-free apart from needing compiled programs for the sampler names.
Three passes `zeta` (no inputs), `alpha` declaring `u_zeta`, `mid` declaring `u_alpha`, none of
them wired in `graph.json`. Assert `_strip_order(document.passes, document.effective_graph())`
returns `["zeta", "alpha", "mid"]`.

Falsifier: pass `document.graph` instead and it returns `["alpha", "mid", "zeta"]` - sorted-name
order, because the raw graph has no edges at all. The pass names are chosen so that alphabetical
and topological order disagree on every position, which is what makes the assertion decidable;
with names like `a`, `b`, `c` the two orders coincide and the test would pass under the bug.

## Manual verification (the maintainer, in the app)

The parent's W-D line is "the strip shows name + thumbnail, nothing truncated; declare
`uniform sampler2D u_df;` in a new pass - it is wired." Expanded into steps that each fail for one
reason:

1. **The strip.** Open the Radiance Cascades example. Every tile is a square picture with one
   centred name under it and nothing else. No `u_x <- y` text anywhere, no `has compile errors`
   line, and no ellipsis. The rows sit evenly: with the cards now all the same height, the gaps
   between rows are uniform rather than set by whichever pass had the most inputs.
2. **The error border still speaks.** Break one pass's shader (delete a semicolon, save). Its tile
   takes the red border and no text appears under the name. Fix it; the border goes.
3. **The rename landed.** Open the gear on `composite`. The Reads rows read `u_cascade` and
   `u_paint`, both showing their pass by name. Open Bloom Chain's `composite`: `u_scene`,
   `u_blur`, `u_trail`. Both examples render exactly as they did before - RC shows bounced light,
   Bloom shows the glow and the trail.
4. **The auto wire, cold.** Add a pass called `edge` to the RC document. Its gear opens with
   `no sampler2D uniforms`. Close it, type `uniform sampler2D u_df;` into the editor above
   `void main()`, make the body `fs_color = texture(u_df, vs_uv);`, and save. Within a frame or two
   the viewer shows the distance field. **The gear is not opened and `graph.json` is not touched.**
5. **The gear's three states, one sampler.** Open `edge`'s gear. The `u_df` row shows `auto: df`.
   Pick `(none)`: the viewer goes black. Pick `jfa`: the viewer shows the jump-flood buffer. Pick
   `auto: df`: the distance field is back. Each of the three is visibly a different item in the
   list, which is the half today's combo cannot express.
6. **The explicit none survives a reload.** With `u_df` set to `(none)` and the viewer black,
   switch to another document and back (which saves and reloads). Still black, and the gear still
   reads `(none)` rather than `auto: df`.
7. **A bound texture is not stolen.** Open the Media Input example, add a pass called `image`, and
   confirm the output pass's `u_image` still shows the shipped PNG rather than the new pass's black
   canvas. (This is the one manual step that exercises decision 4 on a real document; the example
   does not ship in that shape, so the pass has to be added by hand.)
8. **The copilot's view.** Open the chat on the RC document with `edge` present and unwired, and
   ask it to describe what `edge` reads. It says `df`, because the working set's `inputs:` row says
   so - not "nothing".

## Verified / corrected premises

Every citation the parent spec makes about W-D, opened at `d2ade88` and marked. Nine confirmed,
five corrected, one refuted.

**Confirmed.**

1. `PassGraph.with_input(…, "")` deletes the key today -
   `if producer: inputs[uniform] = producer else: inputs.pop(uniform, None)`. The parent's premise
   for the `""` redesign holds exactly.
2. `watch.py::_reload_pass_if_changed` has no graph and performs no compile: it calls
   `render_pass.release_program(new_text)` on the root path and `render_pass.invalidate()` on a lib
   path. Nothing in the module imports `PassGraph`. The hot-reload seam is as described.
3. `Pass.get_active_uniforms()` compiles a never-attempted pass (`if self.program is None and not
   self.compile_unit.error_raw: self.compile()`). Using it to gather sampler names would invert
   066 D1, as the parent states.
4. The gear's combo falls to index 0 for a stored `""`: `choices = [_UNWIRED, *sorted(passes)]`,
   `current = entry.inputs.get(uniform, "")`, `index = choices.index(current) if current in choices
   else 0`, and `""` is not in `choices`. Absent and explicit-none are indistinguishable on screen.
5. `_draw_pass_tile` builds `sublines = [f"{uniform} <- {src}" ...]` and appends
   `"has compile errors"`, and `preview_cell` ellipsizes each to the cell width
   (`_ellipsize(sub, avail.x)`). #19's truncation complaint is reproduced by reading the code.
6. `SIZE.PASS_THUMB` is 112 in `theme.py`.
7. RC's `graph.json` wires `seed.u_scene <- paint`, `cascade.u_scene <- paint`,
   `composite.u_scene <- paint`, `composite.u_light <- cascade`, and already conforms on
   `jfa.u_seed`, `df.u_jfa`, `cascade.u_df`, and both `u_prev`s. The parent's RC mapping is right.
8. Bloom's `graph.json` wires `bright.u_src <- scene`, `blur.u_src <- bright`,
   `trail.u_src <- scene`, `composite.u_lit <- scene`, `composite.u_glow <- blur`,
   `composite.u_trail <- trail`. The parent's Bloom mapping, corrected in round 1, is right.
9. `8d454b7b…/passes/main.frag.glsl` declares five `u_light_*` uniforms at lines 56-60, and
   `0b0d16bb…/passes/main.frag.glsl` declares `u_glow_strength` / `u_glow_radius` at lines 12-13.
   The prefix trap is real and both citations are accurate.

**Corrected.**

10. **`core.py:373` is `core.py:381` at `d2ade88`.** The line
    `value = inputs.get(uniform.name, self.uniform_values.get(uniform.name))` sits at 381; line 373
    is inside `Pass.render`'s signature region. The parent's claim about the line's BEHAVIOUR is
    exactly right - `inputs` wins over `uniform_values`, so an auto edge would replace bound media
    - only the number drifted. This spec cites the symbol.
11. **The media exclusion cannot test `isinstance(value, MediaWithTexture)` alone.**
    `Pass._default_uniform_value` seeds EVERY unbound sampler with
    `Image(self._DEFAULT_IMAGE_FILE_PATH)`, so every sampler in a compiled pass holds a
    `MediaWithTexture`. A naive exclusion would therefore disable auto-wiring entirely.
    `media.py::is_default_image` exists for exactly this distinction and is what decision 4 uses.
    The parent's design is right; its implementation needs this one extra term.
12. **`73ea2431…` is not an instance of the media-collision shape.** The parent writes that an auto
    rule "would otherwise let a pass named `image` silently replace the PNG in the `Media Input`
    example (`73ea2431…`)". That example is SINGLE-PASS - one pass called `main` - so no pass is
    named `image` or `video` and the rule could never fire there as shipped. The exclusion is still
    required, for the general case and for a user who adds such a pass (manual step 7 constructs
    it); the example is the nearest shipped shape, not an instance.
13. **A third prefix trap sits inside the RC example itself.** `77a84d27…/passes/paint.frag.glsl`
    declares `uniform float u_light_radius` at line 20 and reads it at line 35. The parent's
    warning scopes the rename to "the two multi-pass examples" as the protection against the
    `u_light_*` trap - but one of those two multi-pass examples contains the trap. Exact-token
    matching, not file scoping, is what makes the rename safe, and this spec says so.
14. **`projects/dev` needs no `""` hand-edit.** The parent's job description asks for
    "`graph.json` hand-edits for both examples and any `projects/dev` document (`""` where an
    explicit none is meant)". Both dev documents were opened: `e7e00c46…` and `ec926580…` are
    single-pass with `"inputs": {}` and no sampler that names a pass (`e7e00c46…`'s `u_video` has
    no pass called `video`). No edit is needed in either, and none of the shipped examples needs a
    `""` either - every sampler in both multi-pass examples is meant to be filled.

**Refuted.**

15. **The `SB_*` api-lock test is not affected by this wave.** The parent writes "the api-lock
    tests for examples updated". `tests/test_examples_resolve.py::test_shader_lib_api_lock` pins
    `{name: signature}` of every shipped `SB_*` function against
    `tests/shader_lib_api_lock.json`; it reads the seed shader library, not the examples, and the
    rename touches no `SB_` identifier. There is no example-keyed api-lock elsewhere - a grep for
    `document_examples` across `tests/` returns ten modules, none of which asserts an example's
    uniform NAMES. So no lock file is regenerated. What the parent's intent maps onto is the
    RC compile test plus the new `test_every_multi_pass_example_compiles_every_pass` and the new
    D9 gate in § Tests. (The resolve-clean test is NOT part of that answer - round 1's F4 showed
    it runs the `SB_*` include resolver and never invokes GLSL, so it cannot see a uniform rename
    at all; § Tests carries the correction.)

**Two premises this spec adds that the parent does not state.**

16. `copilot/backend.py::_pass_views` reads `entry.inputs` raw to build the working set's `inputs:`
    rows. It is a consumer of the resolution beyond the four the parent names (render, planner,
    gear, strip), and left alone it would tell the model a sampler reads BLACK while the renderer
    fills it. Decision 7 covers it. (`has_feedback` is a sixth, found at implementation time -
    see § Landed deviations.)
17. `widgets/pass_list.py` is the ONLY caller of `preview_cell`'s `sublines` in `shaderbox/`
    (verified by grep at `d2ade88`), and `tests/test_ui_prose_budget.py` derives its scored domain
    from `ui_primitives`' live signatures, scoring `sublines` at 4 words. So decision 11 keeps the
    parameter and removes only the caller's use of it. **Round 1's F5 corrected the reasoning
    behind that:** the derived rows were never the risk, since they come from the signature and do
    not move. What the removal actually touches is the gate's `_UNMEASURABLE` allowlist entry for
    `_draw_pass_tile`, which SURVIVES (`footer=name` is still an unresolvable `ast.Name`) while its
    written reason, "the pass's name and wiring", stops being true. Decision 11 now traces that and
    updates the reason.

## Open questions

Each carries the robust default this wave implements if the maintainer says nothing.

1. **Should `auto: <x>` say the pass name, or just `auto`?** The three-state combo needs the two
   automatic outcomes to be distinguishable (`auto: df` vs `auto: none`), because "the name found
   a pass" and "the name found nothing" are different states the user acts on differently. But a
   plain `auto` for both is shorter and the effective source is visible in the picture. **Default:
   name the pass** (`auto: df` / `auto: none`), because a combo that shows the same label for a
   working wire and a black one is the defect this decision exists to fix. Marked: cosmetic, one
   f-string, reversible.

2. **Does a stale explicit name deserve its own combo item?** An input wired to a pass that was
   deleted outside the transactional verbs (a hand-edited `graph.json`) selects `(none)` today
   under decision 6, which loses the information that a name is stored. A fourth item kind
   (`missing: ghost`) would show it. **Default: no fourth item.** The transactional delete and
   rename already rewrite every edge, so this state is reachable only by hand-editing a file the
   docs say not to hand-edit; `unresolved_inputs` still carries it for the planner, and the pass
   reads black either way. Marked: adds a state to a control that just gained two.

3. **Should the copilot see which inputs are auto?** Decision 7 shows `u_df <- df` whether the
   edge was chosen or derived, on the argument that the mechanism is not the model's business.
   The counter-argument: a model that knows an edge is automatic knows it can change the wire by
   renaming the uniform, which is cheaper than a graph tool it does not have. **Default: do not
   mark it.** The prompt block's new sentence already teaches the naming rule, so the model can
   derive the mechanism from the name it can see; marking every row would spend tokens on every
   turn to restate what one static sentence says once. Marked: one branch in `_pass_views`,
   reversible after a dogfood run shows the model mis-wiring.

4. **Where does the D9 rule live in `conventions.md`?** This wave folds it into the existing
   pass-graph entry as one bullet rather than opening a new entry, on the grounds that it is a
   property of the same mechanism. If the maintainer would rather the resolution rule stand as its
   own entry (it is a render-time rule, not a model rule), it moves. **Default: one bullet in the
   pass-graph entry**, phrased as what the mechanism IS rather than as what this wave changed.
   Marked: doc placement only.

## Landed deviations

What the implementation does that this document did not say, each with why it was forced. The
decisions above are edited in place where the change is the decided shape; this section is the
record of what MOVED, so a reader of the spec and a reader of the code see the same design.

1. **`effective_inputs` carries stored edges through unconditionally.** Decision 1 said "a dict
   comprehension over `samplers`", which returns `{}` for an uncompiled pass and would lose every
   edge `graph.json` holds on a freshly loaded document. Decision 1 now carries the real shape and
   the reason. Second-order: a stored edge on a uniform the program does not declare survives into
   the effective graph, which is harmless because `core.py` binds by declared uniform.

2. **`Document.has_feedback` is a sixth consumer.** It gates the Clear canvas button and planned
   the RAW graph, so a pass declaring `u_prev` with no stored edge was invisible to it and the
   button never appeared. Decision 3's list and `conventions.md` both say six.
   `test_a_u_prev_pass_has_feedback_without_a_stored_edge` pins it.

3. **Every declared sampler is bound BLACK before the resolved edges overwrite it.** Decision 4
   said `core.py` keeps binding what the document hands it, and the worked example asserts the
   `(none)` case renders black - but an unbound sampler falls through to `uniform_values`, which
   holds the seeded default PHOTO. Two surfaces this wave shipped assert otherwise (the gear's
   `auto: none`, the copilot's `reads BLACK`), and 065 D3 says an unfilled input reads black. So
   `Document.render` seeds `inputs` from `sampler_names(render_pass)` with the 1x1 black texture
   and lets resolved edges overwrite. This covers the stored-`""` case, the name-matched-nothing
   case and the no-`u_`-prefix case with one rule. The fall-through was pre-existing, not a W-D
   regression; W-D is what made a large population of samplers legitimately resolve to nothing.

4. **`_pass_views` resolves AFTER the compile that discovers the samplers.**
   `_sampler_uniform_names` goes through `get_active_uniforms`, which compiles - so resolving
   first sees `program is None` for every pass and reports every name-wired sampler as BLACK. The
   sampler names are gathered for all passes first, then the graph resolves once. (On the live
   path `_copilot_document_working_view` already compiles before it calls here, so the defect is
   reachable only through a direct `_pass_views` call; the test asserts there.)

5. **`preview_cell`'s `sublines` parameter is deleted, not kept.** Decision 11 kept it for 070 and
   for the prose-budget gate's derived row. With the strip no longer passing it the parameter had
   zero production callers, which is speculative surface that must be taught and maintained; 070's
   graph view is not a `preview_cell`. The gate's derived row follows the signature, so dropping
   the parameter drops the row.

6. **`_sampler_names` is a module-level `sampler_names`, and `_is_user_bound` is `is_user_bound`.**
   The first took `self` and never used it (a free function by the `@staticmethod` rule); the
   second is read by `popups/pass_settings.py`, and a name two modules share is not private.
   `popups/pass_settings.py` keeps its OWN `_sampler_names`, which calls `get_active_uniforms` on
   purpose - opening the gear is a user asking to see this pass's inputs, so it pays the compile.

7. **`_pass_views` binds `effective_graph()` once above the loop**, where decision 7's snippet put
   it inside. Same answer, N program scans instead of N².

8. **A shim in `tests/test_copilot_script_tools.py`.** Its hand-built `SimpleNamespace` document
   gained `effective_graph`, since `_pass_views` now calls it.

9. **Two module docstrings** (`widgets/pass_list.py`, `popups/pass_settings.py`) gained a paragraph
   stating what the surface now IS. Their first drafts narrated the change instead and were
   rewritten.

## Review history

**Round 1, pre-implementation review** (`reviews/wave_d_pre.md`, opus - judgement against the code
was the deliverable). Parent coverage PASS; resolution-rule correctness, renames, strip tune, test
falsifiability and docs each PARTIAL; **fixpoint and invariants FAIL**. Seven findings, all
accepted as written and folded above. Each was re-verified against `d2ade88` before folding rather
than taken from the report.

- **F1 (the FAIL).** `widgets/pass_list.py` plans the RAW graph twice - `evaluation_order` at
  `:158` for the stale wash and `plan_passes` inside `_strip_order` at `:36` for the tile order -
  and the spec had listed the strip as a consumer of the resolution while giving it only a
  removal. On the worked example's own frame 3 that washes four passes grey while the renderer
  draws them every frame, and a document wired entirely by name loses its topological tile order
  to sorted-name order. Decision 11 now hands both calls the effective graph, and the strip is a
  real consumer rather than a deletion. Two tests added:
  `test_an_auto_wired_ancestor_is_not_washed_stale` and
  `test_the_strip_orders_a_name_wired_document_topologically`.
- **F2.** `compile()` does not call `seed_uniform_values`, so a compiled-but-never-rendered pass
  has an empty `uniform_values` and an empty `bound`. The behaviour was already right; the spec
  asserted it without naming the window. Decision 2 now states why `_sampler_names` reads the
  program (the two are populated by different events), and decision 4 states why the empty `bound`
  is correct rather than merely harmless.
- **F3.** A sixth `Filled by` comment, `bright.frag.glsl:9` - a whole-line comment above the decl
  with a capital F, which an inline-pattern grep misses. Decision 9 now says six, names it, and
  says why the census was taken with `grep -rn "illed by"`.
- **F4.** `test_examples_resolve_clean` runs the `SB_*` include resolver and never invokes GLSL,
  so it cannot catch a half-renamed shader - the spec's claim that it did was wrong. RC is covered
  by its own compile test plus two pixel assertions; Bloom was covered by nothing, because
  `test_every_pass_renders_once_within_n_frames` asserts stamps and W-C writes those on ATTEMPT.
  `test_every_multi_pass_example_compiles_every_pass` closes it.
- **F5.** The `_UNMEASURABLE` allowlist entry survives the `sublines=` removal (`footer=name` is
  still an unresolvable `ast.Name`), so the gate stays green - but its written reason, "the pass's
  name and wiring", becomes half-false. The spec's original reasoning argued from the derived
  rows, which was true and was not the risk. Decision 11 now traces the allowlist entry and
  updates its reason in the same commit.
- **F6.** Decision 6 gave the index-READ logic and the store-per-position table but never said how
  `choices` is built, and the two must agree positionally or every selection writes the wrong
  pass. Now specified, with the `_PASS_NAME_RE` argument for why a pass name can never collide
  with either synthetic label. Decision 5 also names `wire_pass_input`'s unchanged
  `if producer and ...` guard as the line that admits `""` under its new meaning.
- **F7.** `dev_flow.md`'s module map gains three edits; § Files touched had named only
  `conventions.md`.

**The one undecided cell, closed.** The reviewer's matrix trace found `effective_inputs` total over
`(absent, "", explicit)` and correct on `u_prev`, with one cell genuinely open: a sampler whose
name lacks the `u_` prefix. Decided in decision 1 as no prefix, no auto edge - D9's rule is about
`u_<pass>` names, and extending it past what it says would give a user a branch they cannot predict
from the rule they were taught.

**The fixpoint reasoning, added.** The spec asserted that a draw order growing between frames is
safe; the reviewer traced frames 0..N of a six-pass document under W-C's sweep and confirmed it,
supplying the reason the spec lacked - every invariant in `pass_graph.py` is asserted per-call on
one plan against the graph that plan was built from, so there is no cross-frame invariant to
violate, and draw-once within a frame is W-C's per-frame `drawn_frame` skip. Decision 3 now carries
it, so a later reader does not add a guard with nothing to guard.

**Confirmed against the code by the reviewer, independently of this spec:** all eighteen rename
keys and every decl/read line in the table; the third prefix trap inside RC's own
`paint.frag.glsl:20`; both `document.json` files carrying `"uniforms": {}`, so the rename strands
nothing; both `projects/dev` documents needing no `""`; the GL-freedom of `effective_inputs`'s
signature (and that `media.py` imports moderngl, so the `bound` parameter is what preserves
`pass_graph.py`'s GL-free property rather than a stylistic choice); and the new D9 gate simulated
over all seven examples' `graph.json` - nine edges fail today, zero after the table. The four
closed open questions were reviewed and all four defaults agreed with.

**Nothing rejected**, and no finding escalated to the maintainer.

**Round 2, post-implementation** (`reviews/wave_d_post_code.md`, correctness; `wave_d_post_spec.md`,
spec fidelity and architecture. Commit under review `f18a7d3`). Both ran `make gates` unpiped and
both report GREEN. Resolution rule, effective graph, planner-across-frames, gear, renames and
`has_feedback` all PASS in the code review; findings closure and renames PASS in the spec review.
Eight findings between them, all accepted; nothing escalated.

- **The FAIL, found twice** (code F1, spec F8). A sampler with an ABSENT key that resolves to
  nothing rendered the shipped 960x1280 photograph, while the gear said `auto: none` and the
  copilot said `reads BLACK` about the same sampler in the same frame. The wave applied the
  black-bind reasoning to the `""` case and stopped there. The maintainer's ruling: an unresolved
  sampler reads black exactly like a stored `""`. Landed deviation 3.
- **Code F3.** `test_u_df_beside_df_renders_without_the_gear` asserted only `max(rgb) > 0`, so the
  render-seam falsifier (plan the RAW graph) stayed GREEN - the photo it fell through to is also
  non-black. This is the checker-narrows-its-own-domain shape: the test's subject was "the
  distance field rendered" and it verified "something non-black rendered". It now compares the
  `edge` canvas against `df`'s own canvas texel for texel, and the falsifier goes red.
  `test_an_unresolved_sampler_renders_black` covers the other half.
- **Code F2 / spec F1.** `_pass_views` ordering. Landed deviation 4.
- **Code F4.** `preview_cell`'s `sublines` had zero production callers. Landed deviation 5.
- **Code F5 / spec F5.** `_sampler_names` took an unused `self`; `_is_user_bound` crossed a module
  boundary under a private name. Landed deviation 6.
- **Spec F2 / F4 / F3 / F7.** This spec had no post-implementation record (now § Landed
  deviations and this round), two docstrings narrated development history (rewritten), the parent
  spec's refuted "api-lock tests for examples updated" claim still stood (corrected in place at
  `01_spec.md § W-D`, pointing at premise 15), and the roadmap banner was two waves behind
  (rewritten).
- **Spec F6.** `_black_texture`'s lifetime checked and clean: one lazily-built 1x1 texture per
  document, released on the document's own path, so the new binding allocates nothing.

**Confirmed by the reviewers, independently of this spec:** the twelve-cell resolution matrix plus
seven extra cases (a bare `u_`, a bound `u_prev`, `u_edge` on pass `edge`, a stale edge on an
uncompiled pass); a six-pass twelve-frame planner trace with `assert_plan_invariants` on every
frame, converging at f3 with no pass drawn twice and one predicted out-of-order turn at f2; the
media-exclusion collision case constructed by adding a pass named `image` to the Media Input
example (the user's PNG survives); `""` surviving both `_graph_renamed` and `_graph_without`;
`effective_graph` measured at 0.021 ms for six passes, ~0.5% of a 60 fps budget across a frame's
four calls; and all eighteen rename tokens with the three prefix traps intact.
