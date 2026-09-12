# 092 review — the semantics of a connection, and the visual language of errors

Scope: the maintainer's two questions. *How do we visualize errors?* and *what happens when we
connect wrong data to a wrong pin — is a wrong connection even possible here?* Answered from the
code, with `uv run python` probes over `pass_graph` / `document` / a standalone GL context where
the answer is not readable off a branch.

## The three facts the rest of this hangs on

**1. `wired_pass` cannot emit a name that is not a pass.** Its `passes` argument comes from
`Document.effective_wiring`'s `names = set(self.passes)`, and every return path is either `None`
or a name it just membership-tested against that set. Probed: over `passes={'a','b'}`, the outputs
of `PassSource("ghosty")`, `PassSource("a")` and `AutoSource()` are `{None, 'a'}` — a subset of
`names | {None}`. So the wiring handed to `plan_passes` is closed by construction.

The consequence for the graph: **`GraphError`'s second documented kind — "a read of a pass that
does not exist" — is unreachable through the app.** `plan_passes` does `reads[name]` on a dep it
never keyed, so a wiring with a dangling value raises `KeyError` rather than reporting; probed on
`{"b": {"u_a": "ghosty"}}` → `KeyError: 'ghosty'`. The only `GraphError`s the app can produce are
the two cycle messages. A missing source is not an error anywhere — it is an *absence* from the
wiring, which is the black read.

**2. Every `sampler2D` accepts every texture, and GL says nothing.** Probed on a real context
(RTX 3090, `moderngl.create_context(standalone=True)`): the same `uniform sampler2D u_src` was fed
an `f1` 64×64, an `f2` 8×8 and an `f4` 256×256 in turn, sampled and doubled, with no error raised
and a correct value out each time. `texture()` takes normalized coordinates, so **scale is
invisible to the consumer**: a producer at `scale=0.25` (32×32) feeding a consumer at 128×128
yields the producer's exact color, `graph_errors == []`. `Pass.render`'s sampler branch does
`texture.use(location=texture_unit)` with no format question asked — there is no branch in the
engine that could reject a texture.

**3. `Document.graph_errors` has no reader.** `grep -rn graph_errors --include=*.py` over
`shaderbox/` returns `document.py` and nothing else; the only other hits are four tests. It is
recomputed on every `Document.render` (both the early-return path and `plan_for_output`) and
displayed nowhere. **A cycle is completely invisible in the app today.** That is the single
largest finding here, and it is what makes the graph view's error language worth designing rather
than retrofitting.

---

## The state table

`cite:` names the function that decides. "Shown today" = the strip (`widgets/pass_list.py`) and
the sampler row (`widgets/uniform.py::draw_ui_uniform`, `input_type == "texture"`).

### Port states (a sampler on a consumer node)

| # | State | Engine behavior (cited) | Shown today | Proposed cue |
|---|---|---|---|---|
| P1 | **Filled by a pass** — value is `PassSource(p)`, `p` exists | `wired_pass` → `p`; `Document.render` puts `passes[p].canvas.texture` in `inputs`; `Pass.render` binds it. Probed: wiring `{'cons': {'u_src': 'src'}}` | Row shows the pass's live thumbnail + its name, clickable to pick that pass. Strip: a chip naming `p` | Solid dot, filled, edge drawn. No color beyond the neutral port fill — the normal case must be the quiet one |
| P2 | **Filled by the name rule** — value is `AutoSource()`, uniform is `u_<pass>` and that pass exists | `_auto_source` → `uniform[2:]`; same bind as P1. Probed: `wired_pass(AutoSource(), 'u_a', 'b', {'a','b'})` → `'a'` | Identical to P1 — the combo shows the resolved pass name, nothing says it was inferred rather than chosen | **The one place a distinction is worth drawing.** Same solid dot, but the *edge* is drawn at reduced alpha or with a fine dash: a resolved edge never reaches disk (`conventions.md`: "A resolved edge never reaches disk, because it would then be indistinguishable from a chosen one"), so the picture should carry the same distinction the persistence does. No hue — this is not a fault |
| P3 | **Unfilled, undecided** — `AutoSource()`, name matches no pass (`u_nope`, or bare `tex`) | `wired_pass` → `None`, no wiring entry; `Pass.render` falls to `self._black_texture()` — an explicit 1×1 `\x00\x00\x00\xff`. Probed: `wired_pass(AutoSource(), 'u_nope', 'b', …)` → `None` | Row: `_draw_texture_preview(..., None, "no source")` — an empty bordered slot captioned *no source*. Strip: no chip | Hollow dot (ring, not disc), no edge. **Not `STATE_WARN`** — black is the documented default, not a fault (`conventions.md`: "An unfilled pass input reads BLACK … Revisit if a 'no input' state ever needs to be visibly distinct") |
| P4 | **Unfilled, decided** — `NoSource()` | Same bind as P3 — `wired_pass` returns `None` for a `NoSource` unconditionally | Row: combo reads `none`, same empty slot. Indistinguishable from P3 in the picture; only the combo's text differs | Hollow dot with a **filled black centre** — a decision, not an absence. This is the distinction the panel already makes in text and the picture does not make at all |
| P5 | **Explicit source naming a pass that no longer exists** — `PassSource("vanished")` | `wired_pass` → `None` (membership test fails), so **it is exactly P3**: reads black, no wiring entry, no error. Probed end-to-end: set `u_src = PassSource("vanished")`, render → `effective_wiring() == {'src': {}, 'cons': {}}`, `graph_errors == []`, `sampler_source('cons','u_src') is None` | Row: combo snaps to `none` (`index = 1 + passes.index(resolved) if resolved in passes else 0`), empty slot. **The stored name is silently unreadable in the UI.** Strip: no chip | `STATE_WARN` hollow dot + a short stub edge going nowhere, labelled with the dead name. This is the one unfilled state that is genuinely a *fault*: the user asked for something and got nothing, and today nothing tells them. Note both verbs that should prevent it exist (`forget_pass_sources` / `rename_pass_sources`), so P5 only arises from a hand-edit or an import — rare, and worth flagging loudly when it happens |
| P6 | **A stored row for a sampler the compiled program no longer declares** | `_reads_of`: when `declared` is non-empty, the candidate list is *the declared samplers only* — a stale row is not iterated. Probed: removed `u_gone` from the source, `release_program`, render → `uniform_values` still has the `u_gone` key, `sampler_names(cons) == ['u_src']`, `effective_wiring()['cons'] == {'u_src': 'src'}`, `sampler_source('cons','u_gone') is None` | **Invisible everywhere.** The uniforms panel iterates `get_active_uniforms()`, so no row draws; the strip's chips come from the wiring, so no chip. The row persists to `document.json` and returns if the sampler comes back | **No port, and no cue.** The graph should draw ports from `sampler_names` exactly as the panel draws rows from `get_active_uniforms` — one rule, one source of truth. A stale row is dormant state, not an error; surfacing it would invent a problem the engine does not have. (It *does* create one drag hazard — see C7 below) |
| P7 | **Feedback read** — `PassSource(self)`, or `u_prev` under `AutoSource` | `plan_passes` puts the name in `feedback` and contributes **no ordering edge**; `Document.render` binds `self._feedback_canvas(name).texture`, the previous frame. Probed: `{'trail': {'u_prev':'trail','u_src':'src'}}` → order `['src','trail']`, `feedback == {'trail'}`, invariants OK | Strip: a `prev` chip (`FEEDBACK_CHIP`). Row: the thumbnail is the history texture via `input_texture`, captioned with the pass's own name | The `prev` port the brainstorm already fixed (D2), with its **own dot shape** — a half-filled or double-ring dot. `STATE_INFO` is available and semantically right ("this is a different kind of read"), but see the Decisions section: a fourth hue in the port vocabulary may be one too many |
| P8 | **A never-compiled consumer** | `_reads_of`: `declared` is `[]`, so candidates are the **explicit `PassSource` rows only**. Probed: a pass whose samplers are all `AutoSource` reports `effective_wiring()['acc'] == {}` before its first render, and `{'u_prev': 'acc', 'u_src': 'src'}` after | Strip: the tile draws, no chips. Row: the uniforms panel *forces* a compile via `get_active_uniforms`, so opening the panel resolves it | **The graph's real hazard.** A node would draw with **zero ports** until something compiles it — and unlike the strip, the graph is a picture *of the ports*. See Decision 1 |

### Node states (a pass)

| # | State | Engine behavior (cited) | Shown today | Proposed cue |
|---|---|---|---|---|
| N1 | **Compile error on this pass** | `Pass.compile` on failure keeps the *previous* program (or `None`), stores `compile_unit.errors`, and returns. `Pass.render` retries the compile per call, then `if not self.program or not self.vao: return` — the pass **does not draw**, and its canvas keeps whatever was last drawn into it. Probed: broke `src`'s source, rendered → `program is None`, `compile_unit.errors` populated, and the canvas still reads `[1.0, 0.2, 0.2, 1.0]` — the last good frame | Strip: `border = COLOR.STATE_ERROR` on the tile. Editor: the error strip + the `F8` jump | `STATE_ERROR` node border, exactly as the strip. Plus a **stale mark on the node's picture** — the tile's existing corner tick — because the picture is a *lie* in this state: it is the last good frame, and nothing today says so |
| N2 | **Compile error on a producer this node reads** | Nothing propagates. The consumer binds the producer's canvas (last good frame, or black if it never drew) and renders normally. No error, no flag | Nothing. The consumer's tile looks healthy and its picture may be correct-looking but stale | Nothing on the consumer's border. Optionally the edge from the erroring producer dims. Propagating red would make one typo paint the whole document red, which destroys the cue's meaning |
| N3 | **On a cycle (the culprit)** | `plan_passes` → `GraphError(name, "passes form a cycle: a -> b -> a. …")`, pass left out of `order`. Probed: `{'a': {'u_b':'b'}, 'b': {'u_a':'a'}}` → `order == []`, two errors | **Nothing.** `graph_errors` has no UI reader. The strip's `live` set falls back to `{output}` ("`or {output}` mirrors the renderer's cycle fallback"), so every non-output pass dims and the output draws alone — the user sees a document go grey with no explanation | `STATE_ERROR` on the **edges of the cycle**, not the nodes — the cycle is a property of the loop, and the message already names the trail. Node border unchanged so it does not compete with N1. This is the state with the biggest gap between what the engine knows and what the UI shows |
| N4 | **Not ordered — an input is on a cycle (the victim)** | `plan_passes` → `GraphError(name, "pass is not ordered: an input is on a cycle.")`, also left out of `order`. Probed on a 5-pass graph with one 2-cycle upstream: `order == []`, one culprit + four victims — the whole document | Same as N3: nothing, plus the grey wash | Nothing on the node. The culprit's red edges are the explanation, and marking four victims for one fault repeats N2's mistake at a larger scale. If a badge is wanted, `STATE_WARN` on the victim — never red |
| N5 | **Off-plan** — a live, compiling pass the output does not need | `_order_for` walks back from the target only, so an unreached pass is simply not in the returned order. Probed: `{'base':{}, 'out':{'u_base':'base'}, 'side':{'u_out':'out'}}`, target `out` → `['base','out']`; `side` never draws | Strip: `stale=True` → dimmed footer, dimmed chips, the corner tick. Its picture is the last frame it drew | The same stale treatment on the node: dim the picture and the name, keep the corner tick. No hue |
| N6 | **Never compiled** | `Pass.program is None` and `compile_unit.errors` empty. The live loop admits one such pass per frame (066 D1's first-render sweep) | Strip: an ordinary tile with an empty (black) picture, no border, no chips | Dashed node border (round 3 F already sketches "dashed until it compiles"), zero or partial ports. Neutral, not a hue — this state resolves itself within a frame or two |
| N7 | **Output pass** | `graph.output_pass`; `Document.render` keeps it full size and may redirect its last iteration into an external canvas | Strip: `ACCENT_PRIMARY` border; error red overrides it | `ACCENT_PRIMARY` border, and error red overrides — repeat the strip's rule verbatim, including its precedence |
| N8 | **An iterated pass reading itself** | `Document.render` runs `for iteration in range(entry.iterations)` inside one turn in the order, calling `_swap_feedback(name)` between iterations — so iteration N reads iteration N−1, not the previous *frame*. Probed: `iterations=4`, self-read + a `0.25` source → first pixel `[1.0, 0, 0, 4.0]`, i.e. four accumulations inside one frame | Strip: the `prev` chip, plus the run-count badge in the pass-settings modal | The `prev` port plus the run-count badge the mock already has. **The badge and the `prev` port together are the whole story** — no extra cue. The only thing worth adding is that the self-edge is drawn as a small loop on the node rather than routed on the bus |
| N9 | **A pass reading the document's output** | Not a special case: the planner has no notion of "output" when ordering — output is only `_order_for`'s start node. If the output does not read back, it is an ordinary edge and the reader is simply off-plan (N5). If it does, it is a cycle (N3). Probed both: `{'a':{'u_out':'out'},'out':{'u_a':'a'}}` → cycle; `side` reading `out` → `order == ['base','out','side']`, no error, `side` off-plan for target `out` | N5's stale wash, or N3's invisible cycle | Whatever N5 / N3 already say. **No dedicated cue** — this state does not exist as a distinct thing in the engine, and inventing one in the picture would be the graph asserting a rule the renderer does not have |

### Edge states

| # | State | Engine behavior | Proposed cue |
|---|---|---|---|
| E1 | Ordinary edge (P1) | The bind | Solid stroke, neutral |
| E2 | Name-rule edge (P2) | The bind, but never persisted | Same stroke at lower alpha, or fine dash |
| E3 | Feedback self-edge (P7/N8) | No ordering constraint | A small loop on the node, not a bus route. The mock's `loop` toggle already offers this as an option |
| E4 | Edge on a cycle (N3) | No draw at all | `STATE_ERROR` stroke, and only these |
| E5 | Edge into an off-plan consumer (N5) | Bound but never executed | Dimmed with its consumer |
| E6 | Boundary edge crossing a box / to a ghost | Nothing — a group is a label (`PassEntry.group`), invisible to the planner | Whatever the box's port style is. A ghost is dimmed + dashed per D3, which is a *scope* cue, not a state cue — keep it visually distinct from E2's dash (see Decision 3) |

### Box states (a group at the root tab)

A group is a label on a pass entry and nothing more — `group_runs` is adjacency-based, there is no
group entity, and **the planner never sees a group**. So every box state is a *derived* statement
about its members, and none of them is an engine state.

| # | State | Derivation | Proposed cue |
|---|---|---|---|
| B1 | A member has a compile error | `any(passes[m].compile_unit.errors for m in members)` | `STATE_ERROR` on the box border **and** on the specific boundary port that carries the broken member's output, if any. Entering the tab is what shows which member |
| B2 | The bundle's output pass has a compile error | The box's picture is that pass's canvas — the last good frame | B1's border plus the stale mark on the box's picture, same as N1 |
| B3 | The box's picture has no output to show | The bundle output is ambiguous or missing | Hollow picture, `STATE_WARN` — but see Decision 4: "which member is the bundle's output" is not yet decided, and the error language cannot be finished ahead of it |
| B4 | A member is on a cycle | Members' `GraphError`s | `STATE_ERROR` border. The cycle may be entirely inside the box, so entering the tab is the only place the red edges can be drawn |
| B5 | Non-convex group (mock D) | Nothing — nothing is refused (D4) | **No cue at all.** D4 fixed that nothing is refused; a warning colour would re-litigate a settled decision in the visual language |
| B6 | A group whose members are all off-plan | Every member stale | Dimmed box, same as N5 |

---

## The connection verdict list

Every connection the drag can make: from an output dot (a pass's canvas) to an input port (one
`sampler2D` on one pass). The drop calls the same write the panel row makes —
`ProjectSession.set_sampler_source(document_id, pass_name, uniform, PassSource(producer))` (D6).

That function's guards are the whole refusal surface, and there are exactly three:

```
if ui_document is None:            return f"no such document '{document_id}'"
if pass_name not in document.passes:   return f"no such pass '{pass_name}'"
if isinstance(source, PassSource) and source.name not in document.passes:
                                   return f"no such pass '{source.name}'"
```

It does **not** validate `uniform`. That is the only gap.

| # | Connection | Verdict | Basis |
|---|---|---|---|
| C1 | Producer → a sampler on a different pass, no cycle | **ALLOWED-AND-CORRECT** | The ordinary bind |
| C2 | A pass's own output → its own port | **ALLOWED-AND-CORRECT** (feedback) | D6 fixed it; `plan_passes` adds it to `feedback` with no ordering edge. Probed: hypothetically dropping `b → b.u_prev` on a 4-pass base yields zero errors |
| C3 | A drop that would close a cycle | **REFUSED**, by the pure planner on the hypothetical wiring, before any write | The check: copy the wiring, set `hypo[consumer][uniform] = producer`, `plan_passes(hypo)`, refuse if any error. Probed: dropping `c → a.u_c` on `a→b→c` returns `('a', 'passes form a cycle: a -> c -> b -> a')` plus two victims; the diamond drop `a → c.u_a2` and the unrelated drop `loner → a.u_l` both return `[]`. **Refusing on `errors != []` rather than on "the culprit is one of my two endpoints" is the right test** — the victims' presence is what makes the whole-list check equivalent, and the culprit is not necessarily an endpoint |
| C4 | Producer → a port on a pass with a compile error | **ALLOWED-AND-CORRECT** | The write is to `uniform_values`, which needs no program. `set_sampler_source` never asks about `program`. The row persists and takes effect when the pass compiles |
| C5 | An erroring producer → any port | **ALLOWED-AND-CORRECT** | The consumer binds the producer's canvas — its last good frame, or black. N2's cue (or absence of one) covers it |
| C6 | Producer → a port on a **never-compiled** consumer | **ALLOWED-AND-CORRECT**, and it is the *good* path | An explicit `PassSource` is honoured before any compile (`_reads_of`'s uncompiled branch; 066 D1). But the port must exist to be dropped on — see Decision 1 |
| C7 | Producer → a port for a sampler the program no longer declares | **ALLOWED, and it writes a dead row.** `set_sampler_source` does not check the uniform name; `_reads_of` then ignores the row because `declared` excludes it | **This is the only way the drag can write something that does nothing.** The fix is not a guard in the session write — it is to draw ports from `sampler_names` (P6), so the port is not there to drop on. A stale port cannot be dropped on if it is never drawn |
| C8 | Producer → a **ghost's** port (a ghost is an outside pass shown at a group tab's border) | **ALLOWED-AND-CORRECT** *if the ghost is the consumer* — the write targets a real pass by name, and the tab is only a viewport. D3 already says ghosts keep their ports "so a wire can still be dragged across the boundary" | Note the ghost is not editable per D3 but its ports are drop targets; those two statements need reconciling in the spec. The engine has no opinion — a group is a label |
| C9 | Producer → a **box's merged port** (round 3 B) | **ALLOWED-AND-CORRECT**, expanding to N writes | D-leaning: "a wire dropped on a merged port rewrites every slot behind it". Each expanded write is C1 or C3 individually, so the cycle check must run on the **whole hypothetical set**, not per slot — a set that is fine one-at-a-time can close a cycle collectively only if one of the slots does, so per-slot checking is actually sufficient here; still, plan once with all N applied and refuse atomically, because a partial write is the worse failure |
| C10 | A drop on empty space | **ALLOWED-AND-CORRECT** — writes `NoSource()` | D6. Note this is P4, a *decision*, and it survives a reload where `AutoSource` would be re-decided by the name rule |
| C11 | Producer → a port whose consumer is currently off-plan | **ALLOWED-AND-CORRECT** | Off-plan is about the output, not about validity |
| C12 | Producer at `dtype=f1` → a consumer that expects HDR | **ALLOWED-AND-CORRECT, and not flagged** | See below |
| C13 | Producer at `scale=0.25` → a full-size consumer | **ALLOWED-AND-CORRECT, and not flagged** | See below |
| C14 | Producer with `filter_linear=False` / `wrap` → a consumer that assumes otherwise | **ALLOWED-AND-CORRECT, and not flagged** | See below |
| C15 | A drop onto a pass that was deleted mid-drag | **REFUSED** by `pass_name not in document.passes` / `source.name not in document.passes` — the two guards that already exist | The session write is the backstop; the canvas need not duplicate it |

### The dtype / scale / filter / wrap verdict: **nothing**, not a warning

Not an error, not a warning, not a cue. Three reasons, in order of weight:

1. **The engine does not have the concept.** There is no branch in `core.py` that compares a
   producer's `TargetConfig` to anything a consumer declares, because a consumer declares nothing
   — a `sampler2D` is a `sampler2D`. `Pass.render`'s sampler branch is three lines: pick a texture,
   `texture.use(location=…)`, assign the unit. Probed on a real context across all three dtypes and
   three sizes: correct values, zero errors.

2. **There is no "expects" to mismatch against.** `TargetConfig` describes how a pass's *own*
   target is allocated. A consumer has no target-format expectation to violate; the only thing it
   could be measured against is the *reader's* own `TargetConfig`, which describes where it writes,
   not what it reads. A warning would be comparing two unrelated quantities.

3. **Every combination is a real technique.** A quarter-scale blur feeding a full-size composite is
   the bloom fixture. An `f1` LDR stage feeding an `f2` accumulator is deliberate. `NEAREST` on a
   data texture feeding a `LINEAR` consumer is how JFA works. `wrap` is a per-effect choice. Flagging
   any of these would fire on the repo's own examples, and a warning that fires on correct documents
   trains the user to ignore the colour.

The one genuine hazard is real but is **not a property of the connection**: an `f1` target clamps.
Probed — writing `vec4(7.0)` into an `f1` target stores `255`, into `f2`/`f4` stores `7.0`; and the
self-accumulating chain at `f1` pins at `255` after 12 iterations while the same chain at `f2`
reaches `4.0`. That is `pass_graph.py`'s own documented reason for `DEFAULT_DTYPE = "f2"` ("063
measured f1 saturating at 255 on the FIRST accumulate pass where f2 reached exactly 7.0"). It is a
property of the producer's *own* target, visible with no consumer at all, so if it deserves a cue
it belongs on the node's target badge — never on the edge.

---

## Decisions the maintainer must make

**1. What does a never-compiled node's port list look like?** (P8, C6 — the one that blocks the
picture, not the palette.) `effective_wiring` answers over explicit rows only for an uncompiled
pass, so a pass whose samplers are all name-wired reports zero ports. Probed: the same pass reports
`{}` before its first render and `{'u_prev': 'acc', 'u_src': 'src'}` after. The strip survives this
because its tile is a picture with chips *under* it; the graph cannot, because the node *is* the
port list.

- **(a) Draw zero ports, dashed border, let the sweep fill it in.** Honest, costs nothing, and the
  state resolves within a frame or two of the pass being on screen. But a user who opens the graph
  on a cold document sees nodes pop ports as they compile, and cannot wire a pass that shows no port.
- **(b) The graph compiles what it draws** — call `get_active_uniforms()` on every visible node once
  when the canvas opens. Full picture immediately; inverts 066 D1's budget for exactly one surface,
  and the Radiance Cascades document is six passes, so the cost is bounded and one-off.
- **(c) One per frame, like the first-render sweep** — the canvas admits one never-compiled node per
  frame. Keeps 066 D1's shape exactly; the picture fills in over six frames instead of one.

**2. Does the graph get a fourth semantic hue, or does the port vocabulary stay shape-only?**
The palette is already spent: four `GROUP_TINTS`, `ACCENT_PRIMARY` (output) and `SELECT`
(selection), all guarded by `theme.py`'s import-time collision assert. `STATE_ERROR` is the one hue
the strip already spends on a node. The question is whether ports get colour at all.

- **(a) Shape-only for ports, colour reserved for nodes and edges.** Solid disc = filled, hollow ring
  = unfilled, black-centred ring = `NoSource`, double ring = `prev`. One hue in the whole port
  vocabulary: `STATE_WARN` for P5's dead name. Cheapest, survives an accent swap by construction,
  and keeps `STATE_ERROR` meaning exactly one thing.
- **(b) Ports carry `STATE_*` freely.** More legible at a glance; risks `STATE_WARN`'s yellow sitting
  next to `GROUP_TINTS`' `yellow_n` on a grouped node — the assert forbids the tints colliding with
  the state hues by *identity*, not by perceptual distance, and `yellow_b` vs `yellow_n` on adjacent
  small dots is exactly the case the `STATE_INFO` comment already worries about ("blue_b reads too
  close to the aqua STATE_OK … most visible where both sit as adjacent status squares").
- **(c) Shape-only, plus `STATE_INFO` for the `prev` port specifically**, since feedback is a
  *kind* of read rather than a fault and `STATE_INFO` is the theme's existing "this is machinery"
  colour (it is what `_draw_auto_block` uses for engine-driven uniforms).

**3. Two dashes, two meanings — how are they kept apart?** A dash is already spoken for three times:
ghosts are "dimmed, dashed" (D3), a never-compiled node is "dashed until it compiles" (mock F), and
E2 above wants a dash for the name-rule edge. Ghost-dash is a *scope* statement, compile-dash is a
*state* statement, name-rule-dash is a *provenance* statement.

- **(a) Dash means scope only** (ghosts). Never-compiled uses low alpha; the name rule uses low alpha.
- **(b) Dash on strokes means provenance; dimming means scope; a hairline border means uncompiled.**
- **(c) Drop the name-rule distinction entirely** — accept that a resolved edge and a chosen edge look
  identical in the picture, as they do in the panel today. Simplest; loses the one cue that would
  explain why an edge vanished after a rename.

**4. Which member is a box's output, and what does the box show when there isn't one?** (B2/B3.)
The brainstorm says "the box's picture is the bundle output's live texture" and, separately,
"picking a box as the output picks the bundle's output" — but a group is a label with no entity, so
nothing stores which member that is. Candidates: the member no other member reads; the member the
document output reads; the member that *is* the document output when one is inside. On the
non-convex case (mock D) and on a group with two unread members, these disagree — and the error
language for B3 cannot be written until this is settled, because "no bundle output" is either an
impossible state or a routine one depending on the answer.

**5. Should `graph_errors` reach the UI outside the graph view?** A cycle is invisible today in
every surface; the graph view would become the only place the app explains a document that has gone
grey. Either the graph is the answer (and a user without it keeps the silent failure), or a
notification / strip badge lands in the same wave. This is a scope question, not a design one.

---

## False trails

Things probed that turned out to be non-issues, recorded so the next reader does not re-probe them.

- **"An explicit source naming a pass that no longer exists" as a `GraphError`.** The class's
  docstring and the brainstorm both list "a read of a pass that does not exist" as one of the two
  errors `plan_passes` reports. It is unreachable: `wired_pass` filters by membership, so such an
  edge never enters the wiring. Feeding one in directly raises `KeyError` rather than reporting.
  **The only `GraphError`s the app can produce are the two cycle messages.** The graph view's error
  language therefore has exactly one engine-reported fault class to render, not two.

- **Target dtype / scale / filter / wrap mismatch.** Probed on a real GL context across all three
  dtypes and a 4× size difference: no error, correct values, `graph_errors == []`. There is no
  expectation to violate. Verdict: nothing — see the section above.

- **"A wrong pin type."** Every port is a `sampler2D` and every output is a `moderngl.Texture`.
  There is no second port type in the model, so type-compatibility checking has no domain. If a
  `PassSource` ever grows a field (the conventions entry names a channel swizzle or a mip level as
  the revisit trigger), that is when a port could gain a type — not before.

- **Non-convex groups.** D4 settled that nothing is refused, and the planner cannot see a group at
  all (`PassEntry.group` is a label; `group_runs` works on adjacency in an already-computed order).
  There is no engine state to visualize and nothing for the error language to say.

- **A pass reading the document's output.** Not a distinct state. Either it closes a cycle (N3) or
  it is an ordinary edge whose reader is off-plan (N5). Probed both. The planner has no notion of
  "output" while ordering.

- **Feedback into an iterated pass.** Fully specified by existing machinery: `_swap_feedback`
  between iterations, `begin_frame` once per frame, `target_generation` dropping a history that
  predates a format change. Probed: four iterations of a self-accumulating pass produce four
  accumulations in one frame. The `prev` port and the run-count badge together say everything; no
  new cue is needed.

- **Compile-error propagation to consumers.** There is none, by design — the consumer binds the
  producer's last good frame and renders normally. Verified by rendering with a broken producer:
  `program is None`, errors populated, canvas still holding `[1.0, 0.2, 0.2, 1.0]`.

---

**Can a wrong connection exist? No.** Every port is a `sampler2D`, every output is a texture, the
source is chosen from a closed set of the document's own pass names, the only structurally invalid
drop — a cycle — is refused by the pure planner before anything is written, and every remaining
"mismatch" (dtype, scale, filter, wrap) is a combination the engine binds without complaint and the
repo's own examples depend on.

The one exception is not a wrong connection but a connection to a port that should not have been
drawn: `set_sampler_source` does not validate the uniform name, so a drop on a port for a sampler
the program no longer declares writes a row `_reads_of` will ignore — fixed by drawing ports from
`sampler_names`, not by adding a guard.
