# Proposal B — "The Step List" (bias: LIST-FIRST)

> Produced by a design agent anchored to `00_scenario.md` R1-R10, the primary-source playground
> survey, and the real UI code. Verbatim agent deliverable, saved for the judging round.

## 1. The core idea

The Node tab grows a **Steps** list above the uniform rows: an ordered stack of rows, one per draw,
executed strictly top-to-bottom, each row showing a live thumbnail, its size, and format chips.
Branching is not drawn as wires — it is read off each row's **`reads:` chips**, which name the
earlier steps that row samples, plus a thin **spine gutter** on the left drawing the same
information as vertical connector lines. Each row expands in place to reveal that step's own
auto-generated uniform controls — the exact `draw_ui_uniform` rows that exist today
(`widgets/uniform.py:173`).

**The load-bearing claim:** the list IS the execution order, and the `reads:` chips ARE the graph.
Order is authored (drag rows); dependency is authored (pick a source in a combo). The list is honest
because an ordered list is what the GPU actually does — a topological sort the user wrote down.

## 2. Where it sits

Inside `tabs/node.py::draw`, between `_draw_entry_points` (`node.py:116`) and the uniform-sort /
uniform-list block (`node.py:120-172`). **No new tab, no new panel, no new window** — deliberate,
because `Ctrl+1/2/3` are taken (`commands.py:134-140`) and a fourth tab would mean the list can
never sit beside the uniform rows, which is what makes R10 feel like one surface.

New file `widgets/step_list.py` exposing `draw_step_list(app)`, mirroring the existing widget
layering (`widgets/uniform.py`, `widgets/node_grid.py` are pure `draw(app)` free functions).

## 3. The collapsed row

```
|  Steps  3                                    [ Add step ]                          |
|  .-------------------------------------------------------------------.            |
|  | |  [thumb] 1 scene      full   rgba8   -                     [v]  |            |
|  | |  [thumb] 2 blurX      1/2    rgba16f  reads: (scene)       [v]  |            |
|  | |  [thumb] 3 out    *   full   rgba8   reads: (blurX)(scene) [v]  |            |
|  '-------------------------------------------------------------------'            |
|   ^spine gutter                                                                    |
```

| Cell | What | Built from |
|---|---|---|
| spine gutter | 12px column, draw-list lines only | `add_line`, the `node.py:186-192` entry-point tick idiom |
| `[thumb]` | live 44px square of that step's target | `preview_cell` (`ui_primitives.py:923`), same as the grid |
| `scene` | the step's NAME — the identifier other steps read | `clickable_label` (`ui_primitives.py:1177`) |
| `full`/`1/2` | size chip, ratio-of-output default | `chip_button` (`ui_primitives.py:164`) |
| `rgba16f` | format chip | `chip_button` |
| `reads: (a)(b)` | the dependency cells — **this is the branching** | `pill_button` (`ui_primitives.py:139`) |
| `*` | **view pin**: this step's output is what the big preview shows | `pill_button(color=ACCENT_PRIMARY)` |
| `[v]` | expand caret | `ghost_button` |

## 4. The expanded row

```
|  |  |  Name     [ blurX_______ ]        Order  [^] [v]       |
|  |  |  Source   [ open ]  blur.frag.glsl                     |
|  |  |  Size     ( full )( 1/2 )( 1/4 )( abs.. )              |
|  |  |  Format   ( rgba8 )( rgba16f )( rgba32f )              |
|  |  |  Filter   ( nearest )( linear )   Edge ( clamp )( repeat )
|  |  |  Reads                                                 |
|  |  |    u_src      <- [ scene       v ]                     |
|  |  |    u_prev     <- [ (self)      v ]   "own last frame"  |
|  |  |  Uniforms                                              |
|  |  |    [drag] u_radius     [====o========]                 |
```

Three things to notice:

1. **The Uniforms block IS `draw_ui_uniform`** — same chip/name/control layout
   (`uniform.py:173-311`), same jump-to-declaration, same play/stop pill, same auto-stop-on-grab.
   R10 is not a new mechanism; it is the existing one addressed per-step.
2. **The Reads block is a table of `sampler2D` uniform name -> source step.** The left cell is a
   REAL active sampler introspected off this step's program — the same `GL_SAMPLER_2D` test
   `node.py:67-68` already does. The right cell is a combo listing every step ABOVE this one,
   `(self)`, `(file...)`, `(none)`. **`(self)` is the whole of R5.**
3. **`Source [open]`** summons the step's shader into the editor tab bar — the exact idiom
   `_draw_entry_points` uses (`node.py:221` -> `app.ensure_shader_tab`).

`EditorTabKind` gains a fourth value `"step"`; `EditorTab` gains `step_id`. `tab_label`
(`code.py:20`) gains one branch. The error strip and click-to-jump then serve a step's GLSL error
with ZERO new machinery, because a step's shader is a real file going through the same
`resolve_usage -> CompileUnit -> parse_shader_errors` path. This is the deliberate inverse of the
ergonomics Blocker 1 ("a typo on line 12 surfaces as line 18"). **One file per step is what makes
error locality free.**

On disk: `nodes/<id>/steps/<name>.frag.glsl`; the ordered list + chips serialize into `node.json`'s
`ui_state`.

## 5. Requirement coverage

| # | Requirement | UI element | User action |
|---|---|---|---|
| R1 | several steps | the Steps list | `[ Add step ]` appends a row + creates the file |
| R2 | differing resolutions | per-row size chip, collapsed-visible | click chip -> popup; ratio-of-output default (survey convergence #1) |
| R3 | one step reading several | expanded Reads table, one row per declared sampler; collapsed shows `reads: (a)(b)(c)` | declare `uniform sampler2D u_near;` -> a Reads row appears (introspected, not configured) -> pick source. Nothing caps the count |
| R4 | branching order | the **spine gutter**: list stays linear (= execution order), gutter draws incoming edges as vertical runs | nothing extra — branching is the emergent shape of Reads pointing at non-adjacent rows |
| R5 | self-read previous frame | `(self)` in every Reads combo + a `prev` badge | pick `(self)`. No pair to name, create or alternate |
| R6 | forever-state | `persist` toggle chip + per-step `[ reset ]` | toggle it; §8 |
| R7 | values outside [0,1] | per-row format chip, **collapsed-visible by design** — 8-bit measured fatal, so the wrong value must be readable at a glance, not buried | click, pick `rgba16f` |
| R8 | filter + wrap | expanded Filter/Edge chip groups; defaults `linear` + `clamp` | two clicks |
| R9 | view any intermediate | **(a)** every row's live 44px thumbnail, 0 clicks; **(b)** the view pin `*` retargets the big preview | glance, or 1 click |
| R10 | uniforms per step | expanded row's Uniforms block | expand, drag. A cascade-count slider costs one `uniform float` and one USE of it — no decoy, no no-op multiply, no second file |

## 6. The Living Scene, as the list actually draws it

```
 Steps  15                                                        [ Add step ]
| |    [t]  1  scene           full    rgba16f  -                                [v] |
| |-.                                                                                |
| | |  [t]  2  c5              1/32    rgba16f  reads: (scene)                    [v] |
| | |  [t]  3  c4              1/16    rgba16f  reads: (scene)(c5)                [v] |
| | |  [t]  4  c3              1/8     rgba16f  reads: (scene)(c4)                [v] |
| | |  [t]  5  c2              1/4     rgba16f  reads: (scene)(c3)                [v] |
| | |  [t]  6  c1              1/2     rgba16f  reads: (scene)(c2)                [v] |
| | |  [t]  7  c0              full    rgba16f  reads: (scene)(c1)                [v] |
| |-'                                                                                |
| |    [t]  8  lit             full    rgba16f  reads: (scene)(c0)                [v] |
| |-.                                                                                |
| | |  [t]  9  bright          1/2     rgba16f  reads: (lit)                      [v] |
| | |  [t] 10  bloomA          1/4     rgba16f  reads: (bright)                   [v] |
| | |  [t] 11  bloomB          1/8     rgba16f  reads: (bloomA)                   [v] |
| | |  [t] 12  bloomC          1/16    rgba16f  reads: (bloomB)                   [v] |
| |-'                                                                                |
| |    [t] 13  trails    prev  full    rgba16f  reads: (lit)(self)                [v] |
| |    [t] 14  smoke  P  prev  1/2     rgba32f  reads: (self)                     [v] |
| |    [t] 15  out       *     full    rgba8    reads: (trails)(bloomC)(smoke)    [v] |
```

**Gutter rule** (deliberately simple, therefore honest): a vertical run connects a step to the LAST
row that reads it; a run spanning non-adjacent rows indents one level (the `|-.` / `|-'` brackets),
which makes the cascade block and bloom block read as BLOCKS; a `(self)` read draws a small closed
loop glyph. Nothing else — no crossing lines, no routing, no layout algorithm. **The gutter is a
reading aid over information the `reads:` chips already state in text; if it is ever ambiguous, the
text is authoritative.** That subordination is the safety valve: a spatial hint that can be ignored
costs nothing when it degrades; a spatial *representation* that degrades is a broken UI.

### Readability, honestly assessed by the agent

**What works.** Of the 15 rows, 12 read exactly one predecessor and 11 of those read the row
immediately above. The genuinely branching content is FOUR facts: `scene` is read by many, `lit`
twice, `out` merges three, two rows read themselves. A graph canvas would spend its entire area
drawing the 11 boring edges to make the 4 interesting ones visible; the list draws the 11 boring
ones as ADJACENCY (free, zero ink) and spends its ink on the 4. The `reads:` column is scannable as
a column — `(scene)` repeating six times IS the fan-out, seen at a glance; in a node graph that is
six edges at six y-positions, the picture that requires panning.

**What genuinely fails.** `out`'s three-source merge is the one place the list under-serves:
`reads: (trails)(bloomC)(smoke)` names three rows at distances 2, 3 and 1 above, and the convergence
point is the least legible thing on screen. **A graph would draw this one node better.** Mitigation
is partial: hovering a `reads:` pill highlights the source row. Interrogation is worse than seeing.

**And it does not fit.** 15 rows at `ROW_HEIGHT = 22` plus 44px thumbnails is ~700px against
`PANEL_CTRL_MINH = 600` minus the uniform rows below. **The list scrolls** — and a scrolling list is
one where you cannot see the whole structure at once, precisely the property being claimed as its
strength. Mitigation: a `compact` toggle (16px thumbs, no format chip) -> ~330px. It fits, is
uglier, and loses R9's zero-click glance.

**Agent's verdict on its own crux:** honest and adequate; better than a graph on the 90% case;
worse on the single hardest element (3-way merge among 15 rows); needs compact mode to fit at all.
"I would ship it, and I would not claim the merge case is solved."

## 7. What this makes HARD

1. **Authoring order fights authoring sequence.** A list is a topological sort the USER maintains.
   The natural way to build is "make the final picture, then add the thing that feeds it" — which
   inserts producers ABOVE, against the list's growth direction. Mitigations (insert-above-the-pin
   default, `Order [^][v]`, drag-reorder) reduce but do not remove it: the user can always author a
   step reading something below it, and the UI must show an error state. **A graph never has this
   problem because it has no order to violate.**
2. **A three-way merge is interrogated, not seen.** The design's honest hole.
3. **Bulk edits are click-heavy.** 8 formats or 7 descending sizes = 8-16 discrete actions.
   A row context-menu `Apply format to all steps below` covers the uniform case; the DESCENDING case
   has no bulk affordance and the agent deliberately proposes none ("a generate-N-steps wizard is
   exactly the kind of feature that looks helpful and then constrains everything after"). Cascades
   cost ~20 clicks to set up. That is the price.
4. **Two places show uniforms** — expanded rows vs the node-level list — a discoverability tax.
5. **No chord.** Reachable by `Ctrl+1` then scrolling. Thin for a headline feature, but spending
   `Ctrl+4` costs more.
6. **The panel is frozen mid-copilot-turn** (`begin_disabled(app.copilot_turn_active)`,
   `ui.py:337-340`). Pre-existing and correct, but a 15-row list with live thumbnails going grey is
   a far more visible freeze than three greyed sliders. Expect "the UI is broken" reports.
7. **The view pin is transient state that can confuse** — pin `c3`, walk away, come back to a
   black-ish intermediate and read it as "my shader broke". The `viewing: c3` caption is the whole
   defence and should be prominent.
8. **Scales badly past ~20 steps.** Nothing in the scenario needs 30, but this design has a ceiling
   and a graph does not.

## 8. Naming

User-facing word: **step**. It is the word `00_scenario.md` itself uses; it carries ORDER in its
ordinary meaning (the thesis of a list-first design) where "pass" is order-neutral and "stage"
implies a fixed pipeline the user does not author; it collides with nothing user-facing.

| Concept | Word | Code | Never say |
|---|---|---|---|
| the document | **node** | `UINode` | — |
| one draw inside it | **step** | `Step`, `EditorTabKind="step"` | pass, sub-node, layer, stage |
| the ordered set | **the Steps list** | `ui_state.steps` | pipeline, chain, graph, DAG |
| what a step samples | **reads** | `Step.reads: dict[str,str]` | inputs, channels, ports, wires |
| own last frame | **`(self)`** | `SELF_READ` | ping-pong, double buffer |
| a step's picture | **output** | `Step.target` | buffer, RT, attachment |
| the retarget control | **view pin** (`*`) | `viewed_step_id` | stage-to-render, isolate |

Two deliberate bans. **"Graph"/"DAG" banned from user-facing text** — the UI does not show one, and
promising one invites the expectation of a canvas (a DAG was built here and deleted).
**"Channel" banned** — it implies Shadertoy's positional `iChannel0..3`, and the survey's clearest
negative finding is SHADERed's positional binding (same slot named `posTex` in one example and `clr`
in another). The Reads table is keyed on the sampler's real GLSL name for exactly that reason.

## 9. R6 — forever-state, answered

Governing constraint (from `05_node_model.md`, quoted in the scenario): a node's mutable state is
`UINode.save()` output + `script.py`; anything unreachable escapes persistence AND revert.

**`persist` is a per-step opt-in, OFF by default.** Three questions, three different answers:

| Event | `persist` off (default) | `persist` on |
|---|---|---|
| save / reload | cold, warms live | picture written to `steps/<name>.bin` WITH dtype; resumes as it was |
| copilot revert | cold; **named in the revert notice** | restored from the checkpoint |
| export | cold + N warm-up frames | **cold + N warm-up frames (identical)** |

- **Off-by-default is right** because 14 of the Living Scene's 15 steps genuinely are transient, and
  the ergonomics doc measured the alternative ("2 MB of transient cascade data per save, for a
  texture regenerated in `__init__` anyway"). The `persist` tooltip shows the live cost:
  `Save this step's picture with the node (8.3 MB)`. **A cost the user cannot see is a cost they
  will resent.**
- **R6 makes the missing-`dtype` fix a hard blocker**, not just an R7 nicety: an `rgba32f` buffer
  cannot round-trip without it.
- **Revert must say so out loud.** `RevertResult` gains `cold_steps: list[str]`; the notice reads
  `Reverted scene. 1 step restarted cold: smoke.` Silent state loss is the failure mode the
  reliability doc catalogued for the script route.
- **Export always cold, overruling convenience.** `core.py:187-198` documents `export_isolation` as
  structural "so no export caller can forget to isolate"; a stateful STEP is the same class as a
  stateful SCRIPT, and giving it the opposite policy means two kinds of state with contradictory
  export semantics. Reproducibility wins: exporting twice must produce the same video. So a
  `persist` step's expanded row gains `Warm-up [120] frames`, default 0, and the Render tab shows
  `+120 warm-up frames` so a long export does not look like a hang.

The export row being identical in both columns is deliberate: `persist` answers "does my session
survive a restart", not "what does the video look like", because the video must not depend on how
long the app has been open.

## 10. Self-verdict

R1-R10 all **COVERED**, with R4 carrying an explicitly named weak case rather than a clean pass.

Least confident, in the agent's own order:

1. **The spine gutter.** Cannot be verified without seeing it (the dev box cannot screenshot the
   app). Whether 15 rows of it reads as STRUCTURE or as HATCHING is an aesthetic call for the
   maintainer. **The design survives the gutter being cut** — the `reads:` chips are authoritative
   by construction — so low-risk, but the largest uncertainty.
2. **Whether 15 rows fit.** The arithmetic was done from tokens, not a rendered frame, and the node
   grid takes `control_panel_width / 2.6` horizontally, unchecked against the row's chip widths.
   **The row may not fit horizontally**, in which case `reads:` truncates and the collapsed row
   loses R3-at-a-glance. "The most likely thing to be wrong."
3. **The insert-above-the-pin default**, invented while writing the journey — "exactly when a rule
   is least tested". Right for producers, wrong for consumers, real ratio unknown.
4. **Export always cold** — confident it is CORRECT, less confident it is what the maintainer wants,
   since "capture the smoke as it looks right now" is a legitimate creative request being refused.
   Worth contesting.
5. **The pinned-step uniform mode** — solves the two-places-for-uniforms tax elegantly, but the
   bottom uniform list silently changes what it shows based on a pin elsewhere on screen. "That is a
   mode, and modes are how UIs lie." May be better dropped.

## 11. False trails

- **A fourth `Passes` node-settings tab.** Cheapest to build, rejected because it puts the list and
  the uniform rows on DIFFERENT tabs — optimising the thing that was not broken, when the direction
  doc names parameters and error locality as where friction lives.
- **A node-graph canvas.** Rejected on three independent grounds: the survey's "node graphs fight
  hardest... a different product, not a feature", and both graph tools REFUSE to put multipass in the
  shader ("prepare the passes as individual TextureFX and plug them together in a patch") which
  ShaderBox cannot do having no patch; a DAG was built here and deleted; and §6 argues the list wins
  the 90% case.
- **glslViewer-style pure inference.** "The model that fits ShaderBox's grain best" per the survey,
  rejected as the UI answer because it HAS no UI — no list, no thumbnail, nowhere to hang
  R2/R7/R8/R9/R10, and it exposes no per-pass resolution/format/filter/wrap at all. Kept from it:
  name-based binding, and that a read cannot drift from its declaration.
- **The line-comment micro-syntax** (the survey's named hybrid). Rejected as the PRIMARY surface
  because it makes every R2/R7/R8 change a text edit in a file the user must open, when the point of
  ShaderBox is "drag a slider". **Not rejected as an idea** — the chips are a GUI over exactly this
  information, and if a text form is wanted later, the chips writing that comment line is the
  natural serialization.
- **Per-step editor tabs as the ONLY surface.** Free error locality, but makes R9 and R4 impossible:
  a tab bar has no room for thumbnails or a gutter. The step tabs exist here, just not alone.
- **A tree widget with steps nested under consumers.** Free from imgui, would draw the blocks as real
  subtrees. Rejected: a step read by two consumers has no single parent, so it must be duplicated (a
  lie) or hoisted (destroying the nesting). **A tree can only represent a graph that is a tree, and
  this one is not.** The gutter's block-indent is the honest half — it SUGGESTS nesting without
  CLAIMING it, and degrades to a flat list when the structure does not cooperate.
- **A Steps column in the node grid.** Rejected: the grid's contract is one cell = one node; a click
  that sometimes selects a node and sometimes a step is the two-affordances-for-one-thing slop
  signal.
- **Auto-deriving list order from the reads (engine-computed topological sort).** Would eliminate
  cost #1 entirely. Rejected: it makes the order DERIVED, and a list whose rows move when you edit an
  unrelated shader is disorienting; it destroys the honesty claim (the list is the execution order
  BECAUSE the user wrote it); and a cycle would need reporting with no natural home. "I would rather
  show an error on a backward read than silently rearrange the user's rows."
