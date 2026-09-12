# 092 post-implementation review — architecture and conventions

Angle: where things live, and whether they duplicate something that already exists. Diff under
review: `f04821a` (W1) and `57b137e` (W2).

## Coverage

Read end to end:

- `shaderbox/widgets/pass_graph.py` (new, the whole file)
- `shaderbox/widgets/graph_state.py` (new, the whole file)
- `shaderbox/pass_graph.py` — the 092 additions (`GraphCoord` / `MAX_GRAPH_COORD`,
  `PassEntry.position`, `PassGraph.with_positions`, `group_name_error`, `graph_ranks`,
  `rank_layout`, `Port` / `node_ports`, `BoxPort` / `Boundary` / `group_boundary`,
  `bundle_output`, `wiring_with`, `refuse_drop`, `cycle_edges`, `_CYCLE_PREFIX`)
- `shaderbox/app.py` — the added block (`graph_views`, `graph_view_for`, `arrange_graph`,
  `drop_wire`, `unwire`, `commit_node_drag`, `group_selection`, `dissolve_group`, the
  `forget_render_state` eviction)
- `shaderbox/project_session.py` — `_pass_name_error`, `add_pass`, `rename_pass`,
  `set_pass_positions`, `set_pass_group` / `set_pass_groups`, `import_passes`'s
  `model_copy` line, `set_sampler_source` (unchanged, read as the funnel question),
  `compile_pending_passes`
- `shaderbox/document.py` — `wiring_if_renamed`, `sampler_names`, `rename_pass_sources`
- `shaderbox/tabs/document.py::_draw_passes` and its caller `_draw_entry_points`
- `shaderbox/widgets/pass_list.py` — `pass_menu_items`, `_draw_context_menu`, `draw`
- `shaderbox/pass_import.py::plan_import`, `shaderbox/popups/import_passes.py`,
  `shaderbox/theme.py` (the `GRAPH_*` block and `_GROUP_TINT_EXCLUSIONS`),
  `shaderbox/ui_regions.py`, `shaderbox/ui_models.py`, `shaderbox/help_content.py`
- `tests/test_graph_view.py`, `tests/test_graph_state.py`, and the added cases of
  `tests/test_pass_verbs.py`; the `scripts/smoke.py` stretch
- Anchors: `ai_docs/conventions.md` in full, `.claude/skills/imgui-ui/SKILL.md` in full,
  `ai_docs/dev_flow.md`'s Module map, `03_spec.md` in full
- Comparison reading (not in the diff): `shaderbox/ui_primitives.py` symbol table plus
  `preview_cell`, `text_chip`, `_glyph_button`, `close_cross_button`, `step_squares`,
  `segmented_choice`, `text_tab_row`; `shaderbox/widgets/uniform.py` end to end;
  `shaderbox/tabs/share_state.py`

Skipped, with the reason: `tests/test_pass_graph.py`, `tests/test_graph_persistence.py`,
`tests/test_theme.py`, `tests/test_ui_regions.py`, `tests/test_button_tiers.py`,
`tests/test_ui_prose_budget.py` — read only for their new case names and falsifier comments,
not line by line; they are the correctness reviewer's ground, and this angle needed only
whether a test name over-promises. `shaderbox/core.py`, `shaderbox/ui.py` — untouched by the
diff. The `00_mock.html`, `01_brainstorm.md`, `02_triage.md` and the pre-implementation
reviews — read for what the spec locked, not re-litigated.

Gate state, captured unpiped: `make gates > /tmp/g.log 2>&1; echo $?` -> `0`, and the log's
last line reads `GREEN -- check passed, test passed, smoke passed`. The smoke actually ran
here; it is not a skip.

## Every literal number and color in the two widget modules

`widgets/graph_state.py` first — it is short and almost clean:

| Literal | Site | Verdict |
|---|---|---|
| `(0.0, 0.0)` | `NodeDrag.delta`, `GraphViewState.pan` defaults | fine — an origin, not a size |
| `1.0` | `GraphViewState.zoom` default | fine — the identity zoom |
| `4.0` | `node_size`, the gap above the first port row | **should be a token or a named local**: the same `4.0` is re-derived in `pass_graph._port_point`, and the two must agree or every port dot sits off its own row. See finding 1. |

`widgets/pass_graph.py`:

| Literal | Site | Verdict |
|---|---|---|
| `_ZOOM_STEP = 1.1` | module constant, used in the wheel handler | named local — correct, a gesture rate is not a theme token |
| `_ROUNDING = 6.0` | module constant, node corner radius | **should be a token** (`SIZE.GRAPH_ROUNDING`): a corner radius is exactly the "magic px" §6 puts in `theme.py`. See finding 2. |
| `_FIT_MARGIN = float(SPACE.LG)` | module constant | token-derived — correct |
| `_CYCLE_PREFIX`, `_ROOT_FALLBACK_LABEL` | module constants | named locals — correct |
| `4.0` | `_port_point`, the same first-row gap as `node_size` | **should be the shared constant of finding 1** |
| `2.0` | `_thumb_rect`, `_out_point`, `_fit`, several `/ 2.0` centerings | fine — arithmetic halving, not a design number |
| `2 * _FIT_MARGIN` | `_fit` | fine |
| `4.0 * z` | `_draw_node`, the dash length passed to `_dashed_rect` | **should be a token** (`SIZE.GRAPH_DASH`): it is the visual length of a dash, a look decision. See finding 2. |
| `2 * dash` | `_dashed_rect`'s stride | fine — it encodes "dash and gap are equal", a stated relation |
| `1.5` / `max(1.0, 1.5 * z)` | `_draw_wire`, `_draw_self_loop`, the in-flight wire, the node border | **should be a token** (`SIZE.GRAPH_WIRE_W`); it appears four times. The `1.0` floor is fine (a hairline). See finding 2. |
| `1.2` | `_draw_port_dot`, four `add_circle` thickness args | **should be a token** (`SIZE.GRAPH_PORT_RING_W`) — four repeats of one stroke width |
| `24` (`24 * z`, `dx >= 24 * z`, `b[0] < a[0] + 24`) | `_draw_wire`, `_draw_canvas`'s `backward` test | **should be a token or a named local**; it is also **inconsistent by construction** — two of the three are zoom-scaled and the `backward` test is not. See finding 3. |
| `30.0 * z` | `_draw_wire`, the bezier control floor; `30.0 * view.zoom` in the in-flight wire | **should be a named local** — it is D7's "30 minimum", written twice |
| `0.45` | `_draw_wire` and the in-flight wire, the control-point fraction | **should be a named local** — D7's "45% of the horizontal distance", written twice |
| `12 * z` | `_draw_self_loop`, how far the loop rises over the node | **should be a token** (`SIZE.GRAPH_LOOP_RISE`) |
| `28 * z` | `_draw_self_loop`, twice, the loop's horizontal control reach | **should be a named local** |
| `0.45` | `_draw_port_dot`, the `NoSource` centre-dot radius fraction | **should be a named local** — and note it is the SAME number as the wire's 45%, meaning two unrelated design facts share one magic value. See finding 4. |
| `0.5` | `_draw_port_dot`, the `prev` inner-ring fraction | **should be a named local** (same cluster) |
| `0.8` | `_draw_port_dot`, the `media` square half-extent fraction | **should be a named local** (same cluster) |
| `0.5` | `_draw_node`'s `picture_alpha` dim factor for an error or stale node | **should be a token** — every other alpha in this file is a `COLOR.*_ALPHA` (`GRAPH_GHOST_ALPHA`, `GROUP_FILL_ALPHA`). See finding 5. |
| `2.0 * z` | `_draw_node`, `add_image_rounded`'s corner radius | **should be a token**; it is the picture's own rounding beside `_ROUNDING` for the node |
| `6 * z`, `2 * z`, `14 * z`, `3 * z`, `5 * z`, `3 * z` | `_draw_node`'s two badges (pad, inset, height, corner radius, text offsets) | **six magic px in one block, and the block is written twice** — see finding 6, the badge duplication |
| `max(4.0, …)` | `_draw_node`'s two `push_font` size floors | fine — a floor against a degenerate rasterized size |
| `2 * r + 2 * z` | `_draw_node`, the port label's offset from its dot | **should be a named local or `SPACE`-derived** |
| `2` | `channels_split(2)` and `channels_set_current(0/1)` | fine — a channel count, and note the spec's D7 said three channels; see finding 7 |
| `16.0` | `_draw_canvas`, the bus's clearance under the lowest node | **should be a token** (`SIZE.GRAPH_BUS_CLEAR`) beside `GRAPH_BUS_STEP`, which IS one |
| `max(0, edge.span - 2)` | `_draw_canvas`, the bus step multiplier | fine — a rank arithmetic offset |
| `0.35` | `_draw_canvas`'s `dim_col` | **should be a token** (`COLOR.GRAPH_DIM_ALPHA`) — the off-plan dimming of a wire is a theme decision, and `GRAPH_GHOST_ALPHA` is its sibling that DID become one |
| `0.12`, `0.8` | `_draw_canvas`, the rubber band's fill and border alpha over `COLOR.SELECT` | **should be tokens** (`COLOR.BAND_FILL_ALPHA` / `COLOR.BAND_EDGE_ALPHA`) |
| `0.7` | `_draw_canvas`, the snap guides' alpha over `COLOR.SELECT`, twice | **should be a token** |
| `0.0`, `1.0` | the band's `add_rect` rounding and thickness | fine |
| `max(1.0, avail.x)` etc. | the hit rects' degenerate-size floors | fine |
| `2 * hit` | the port and output hit rects | fine — a diameter from a radius |

Color literals:

| Literal | Site | Verdict |
|---|---|---|
| `(1.0, 1.0, 1.0, picture_alpha)` | `_draw_node`'s `add_image_rounded` tint | **should be `fade(COLOR.WHITE, picture_alpha)`** — `COLOR.WHITE` exists in `theme.py` and `preview_cell` already uses it for exactly this argument (`color_convert_float4_to_u32(COLOR.WHITE)`). A raw white tuple is the one hardcoded color in the file. See finding 8. |
| `(*tint[:3], COLOR.GROUP_FILL_ALPHA * alpha)` | `_draw_node`'s box fill | token-derived, and it matches `pass_list.py`'s two sites — correct |

Everything else in both modules reaches a color through `COLOR.*` and `fade(...)`, which is
right.

## Findings

### 1. REAL — `4.0`, the port row's top gap, is written twice and nothing ties the two

`widgets/graph_state.py::node_size` adds `4.0 + port_count * SIZE.GRAPH_PORT_ROW` to a node's
height; `widgets/pass_graph.py::_port_point` starts the first dot at `y0 + 4.0 + slot *
SIZE.GRAPH_PORT_ROW + SIZE.GRAPH_PORT_ROW / 2.0`. One decides the box, the other decides where
the dots go inside it, and they agree only because someone typed the same number in two files.
Change one and every port dot drifts off its own row, silently — the picture stays plausible,
which is the failure mode the "two parallel things that must stay in lockstep" bullet in
`conventions.md ## Design decisions` names. `tests/test_graph_state.py::test_a_node_grows_one_row_per_port_and_a_box_is_wider`
pins `node_size`'s half and cannot see `_port_point`'s.

Fix: one `SIZE.GRAPH_PORT_TOP` (or a module constant in `graph_state.py` that `pass_graph.py`
imports beside `node_size`), read by both sites.

### 2. REAL — four look decisions sit as magic px in the widget instead of `theme.py`

`_ROUNDING = 6.0` (the node's corner radius), `4.0 * z` (the dash length), the `1.5` wire
thickness at four sites, the `1.2` port-ring stroke at four sites, `2.0 * z` (the picture's
own rounding), `16.0` (the bus clearance). imgui-ui §6 is unambiguous: "no hardcoded hex or
magic px anywhere else… A token used by exactly one panel still belongs in the token bag." The
feature already added fifteen `SIZE.GRAPH_*` tokens, so the bag is the obvious home and the
omission reads as accident rather than decision — `GRAPH_BUS_STEP` is a token while the
clearance it is added to is not.

Fix: `SIZE.GRAPH_ROUNDING`, `GRAPH_DASH`, `GRAPH_WIRE_W`, `GRAPH_PORT_RING_W`,
`GRAPH_THUMB_ROUNDING`, `GRAPH_BUS_CLEAR`, in the existing `GRAPH_*` block with its comment
extended.

### 3. REAL — the `24` that decides a wire's shape is zoom-scaled in one place and not in the other

`_draw_wire` takes the direct bezier only when `dx >= 24 * z` — a screen-space test.
`_draw_canvas` decides whether a wire is `backward` with `b[0] < a[0] + 24`, on CANVAS
coordinates, with no `zoom` in sight. The two are the same threshold in intent (is the
consumer far enough right of the producer to draw a plain curve) and disagree at every zoom
but 1.0: at zoom 0.25 the canvas test admits wires the draw routine then reroutes onto the
bus, and at 2.5 the reverse. The number is also unnamed at all three sites.

Fix: one named constant in canvas units (`_MIN_DIRECT_DX`), and decide the space once —
`_draw_wire`'s test becomes `dx >= _MIN_DIRECT_DX * z` reading the same constant, so the two
cannot disagree.

### 4. NIT — the port-dot proportions are unnamed, and `0.45` collides with the wire's `0.45`

`_draw_port_dot`'s `r * 0.45` (the `NoSource` centre), `r * 0.5` (the `prev` inner ring) and
`r * 0.8` (the media square) are the D11 error language expressed as three bare fractions.
They are local to one function so they are not tokens, but they should be named: `r * 0.45`
carries no hint that it means "a filled centre, visibly smaller than the ring". The collision
with `_draw_wire`'s `0.45` (the bezier's horizontal control fraction) is the reason to name
them rather than leave it — a reader grepping `0.45` finds two unrelated design facts.

Fix: `_NONE_CORE`, `_PREV_INNER`, `_MEDIA_HALF` as module constants beside `_ROUNDING`, and
`_BEZIER_BOW = 0.45` for the wire's two sites.

### 5. REAL — the stale/error picture dim is a bare `0.5` where every sibling alpha is a token

`_draw_node` computes `picture_alpha = alpha * (0.5 if node.error or node.stale else 1.0)`.
`COLOR.GRAPH_GHOST_ALPHA` was added to `theme.py` for exactly this class of decision (how far
a node fades), and `COLOR.GROUP_FILL_ALPHA` is its 091 precedent. The dim wire alpha (`0.35`),
the rubber band's two (`0.12`, `0.8`) and the guide's (`0.7`) are the same class. Five alphas
in the widget, one in the theme.

Fix: `COLOR.GRAPH_STALE_ALPHA`, `GRAPH_DIM_ALPHA`, `GRAPH_BAND_FILL_ALPHA`,
`GRAPH_BAND_EDGE_ALPHA`, `GRAPH_GUIDE_ALPHA` beside `GRAPH_GHOST_ALPHA`.

### 6. REAL — the badge is drawn twice, and imgui-ui §6 says don't repeat a widget

In `_draw_node`, the `xN` run-count badge and the `N passes` member-count badge are two
copies of the same nine lines: measure the label, `add_rect_filled` a `badge_bg` pill with
`6 * z` of horizontal pad and `14 * z` of height at `3 * z` rounding, `add_text` the label in
`badge_fg`. The only differences are the anchor corner (`s1[0] - w - 2*z` vs `s0[0] + 2*z`)
and a one-pixel text offset that differs for no stated reason (`s1[0] - w + z` vs
`s0[0] + 5*z`). imgui-ui §6: "A draw block appearing twice… is extracted to a `ui_primitives`
free function and called from both. Two copies drift." They have already drifted by that
pixel.

This one genuinely belongs in `ui_primitives.py`, not merely in a local helper: a draw-list
badge at an arbitrary screen point is a primitive the repo does not have and the next canvas
will want. `text_chip` is NOT it — it draws at the imgui CURSOR and calls `imgui.text_colored`,
so it cannot be used inside a `channels_split` draw-list block at a computed point.

Fix: `ui_primitives.drawlist_badge(dl, anchor, align, label, scale)` (or a local
`_draw_badge` in the widget if the maintainer wants to wait for the second consumer), called
from both sites with the anchor as the only argument that differs.

### 7. NIT — the widget splits two draw-list channels where D7 specifies three

D7: "`channels_split(3)`: 0 wires, 1 nodes and ports, 2 the foreground (an in-flight wire, the
rubber band)." The implementation calls `channels_split(2)`, merges before the hit-test block,
and then draws the in-flight wire, the rubber band and the snap guides straight onto the
merged list. The result is the same paint order in practice (the foreground work all happens
after `channels_merge()`), so this is not a defect — but the spec's three-channel sentence is
now false, and the next reader who adds a foreground element mid-picture will look for channel
2 and not find it.

Fix: one sentence in D7 saying the foreground is drawn after the merge rather than on a third
channel. Cheaper than changing the code, and the code is the better shape.

### 8. REAL — a raw white tuple where `COLOR.WHITE` exists and the sibling site already uses it

`_draw_node` passes `_u32((1.0, 1.0, 1.0, picture_alpha))` to `add_image_rounded`.
`preview_cell` — the primitive whose blit this deliberately mirrors, per the module
docstring — passes `imgui.color_convert_float4_to_u32(COLOR.WHITE)` to `add_image`. The
theme bag has `WHITE` for this. This is the file's only hardcoded color and it sits next to
the one place that proves the token was the intended spelling.

Fix: `_u32(fade(COLOR.WHITE, picture_alpha))`.

### 9. NIT — `_u32` is a one-line alias for an imgui call, not a shared primitive

`_u32(color)` wraps `imgui.color_convert_float4_to_u32`. `ui_primitives.py` calls that
function verbatim at a dozen sites and has never abbreviated it. So the alias is not
duplicating a `ui_primitives` helper — it is introducing a second spelling of a call the repo
already spells one way. Given how many times `pass_graph.py` calls it, the abbreviation earns
its place locally; the alternative worth considering is promoting `_u32` to `ui_primitives`
and sweeping the dozen existing sites, which is a separate cleanup and not this feature's job.

Verdict: leave it, but do not let a second module grow its own copy — if a third canvas wants
it, it goes to `ui_primitives`.

### 10. NIT — `_dashed_rect` and `_draw_port_dot` are correctly private, not misplaced

Checked against the §6 "don't repeat a widget" rule and the `conventions.md` free-function
rule. Neither duplicates anything in `ui_primitives.py`: the closest relatives are
`close_cross_button` / `_glyph_button` (which submit a real imgui button and draw inside its
rect — a different thing entirely) and `faint_hline` (a single line at the cursor). Both new
helpers are module-level free functions with full annotations, each used from exactly one
place, and neither has a second consumer in sight. Correct as private. Only the badge (finding
6) has two call sites and therefore a claim on `ui_primitives`.

### 11. REAL — `drop_wire`'s media check is a per-caller bracket; the uniforms combo releases a bound texture silently today

This is the finding with the largest blast radius.

`App.drop_wire` refuses a drop on a media-bound sampler with "bound to media; unbind on the
Uniforms tab", because `ProjectSession.set_sampler_source` runs `try_to_release(values.get(uniform))`
before writing — so the write would destroy the user's texture. The spec (D15) calls this "the
one write the canvas must not make silently", and `tests/test_graph_view.py::test_drop_wire_refuses_a_media_bound_port_and_keeps_the_texture`
pins it with the right falsifier.

But the canvas is not the only path. `widgets/uniform.py::draw_ui_uniform`, the texture branch,
computes `index = file_item` when the current value is a bound texture, and on the next combo
pick calls `app.session.set_sampler_source(...)` unconditionally with a `NoSource()` or a
`PassSource`. `set_sampler_source`'s own docstring even states the behavior as intended: "A
texture the user bound is written by its own pickers… into the same slot, so choosing a source
here replaces it." So the uniforms panel's combo does today exactly what the canvas was
forbidden from doing — releases the bound texture with no warning and no undo.

Two readings, and they point opposite ways:

- If silently replacing a bound texture from the combo is INTENDED (the docstring says so, and
  a combo is a deliberate per-sampler choice the user is looking at), then the canvas's refusal
  is a canvas-specific affordance decision — a drag is a coarse gesture and should not destroy
  data — and the code is right where it is. The spec's D15 argues this case.
- If it is NOT intended, then the guarantee "a bound texture is never released without the user
  asking" is cross-cutting, and `conventions.md`'s funnel law applies verbatim: "A cross-cutting
  guarantee is enforced at the single FUNNEL, not per-caller… A SECOND fix of the same bug at a
  sibling site is the trigger to move to the funnel."

I cannot settle which the maintainer wants, and a reviewer should not: the two surfaces differ
in gesture coarseness, which is a real design axis. What the review CAN say is that the
asymmetry is undocumented. `App.drop_wire`'s docstring explains why the canvas refuses; nothing
in `set_sampler_source` or in `conventions.md` records that the combo deliberately does not.
The next person who notices the combo's behavior will read it as the bug the canvas already
fixed and will "fix" it at the funnel, which silently changes the panel.

Fix, minimal and behavior-free: one sentence in `set_sampler_source`'s docstring (or the 092
`conventions.md` entry D20 already owes) saying the release is deliberate for an explicit
per-sampler pick and refused for a canvas drag, and why. If the maintainer instead wants the
guarantee, the check moves onto `set_sampler_source` with an explicit `allow_release` flag the
combo passes, and `drop_wire` drops its copy.

### 12. REAL — `_pass_name_error` and `group_name_error` are two checks of one namespace, not the single funnel D17 asks for

D17: "One function decides a group name, `pass_graph.group_name_error(group, pass_names)`…
`_pass_name_error` gains the mirror check." That is what landed, and it is two functions
holding two halves of one invariant:

- `pass_graph.group_name_error(group, pass_names)` returns "a pass and a group cannot share a
  name" when `group in pass_names`.
- `project_session._pass_name_error(name, existing, graph)` returns the same string when
  `any(entry.group == name for entry in graph.passes.values())`.

They are mirror images with separately written predicates, separately written message
literals, and different argument shapes (one takes a name collection, the other a `PassGraph`).
The invariant they jointly enforce — pass names and group names occupy one namespace — is not
expressed anywhere as one thing. Two failure modes follow: the message strings can drift apart
(they are two literals today), and a third entry point can call one and forget the other, which
is exactly how `_TOOL_VERBS` + `_GATE_PROMPTS` got into `conventions.md` as the drift-smell
example.

`tests/test_pass_verbs.py::test_every_group_writing_entry_point_shares_one_validator` pins the
group direction across its three entry points. Nothing pins that the two directions agree on
the message, and nothing structurally prevents a fourth writer of either kind.

Fix: one `namespace_error(candidate, pass_names, group_names) -> str` in `pass_graph.py` that
both directions call with their own two arguments, so the collision predicate and its message
exist once. `_pass_name_error` keeps its pattern and its `'{name}' already exists` check and
delegates the collision; `group_name_error` keeps the group pattern and delegates the same.

### 13. NIT — `wiring_if_renamed` is on `Document`, and that is right

Checked, because the spec puts it there and a mutate-then-restore method on a live GL-holding
object invites a "should this be pure?" objection. It cannot be pure: `effective_wiring()`
reads `self.passes` (a dict of live `Pass` objects carrying `uniform_values`) and the name rule
resolves against the pass set, so computing the post-rename wiring means presenting a
post-rename `Document` to it. Reconstructing that as a pure function would mean duplicating
`effective_wiring`'s resolution over a hypothetical name set — a second implementation of the
rule, which is worse.

The mutation is correctly bracketed: the re-key and every `PassSource` rewrite are undone in a
`finally`, insertion order is preserved (the docstring says why — strip order and the output
fallback read it), and `tests/test_pass_verbs.py::test_wiring_if_renamed_leaves_the_document_as_it_found_it`
asserts the restore including after a raising call. This matches the conventions bullet "a
serialize routine must not MUTATE what it serializes" in spirit: it does mutate, and it proves
the restore.

Verdict: right altitude, right guard. No change.

### 14. REAL — `sampler_names`'s docstring is now stale

`document.py::sampler_names` says: "Reads the program rather than `get_active_uniforms()`, which
COMPILES a never-attempted pass (066 D1) — asking that here would compile the whole document on
frame one. The live loop's first-render sweep is what brings each pass online, one per frame."

That last sentence was true when the strip was the only consumer. It is false now: the graph
canvas calls `compile_pending_passes(document)` on every frame it draws, which compiles EVERY
never-attempted pass at once — the sweep is no longer the only thing bringing passes online,
and the "one per frame" bound does not hold for a document whose graph view is open.

On whether that per-frame call is acceptable against 066 D1: it is. 066 D1's rule is "a pass
compiles when something first NEEDS its program" and its cost concern is the frame-0 stampede.
The canvas genuinely needs every pass's program (a port exists only when the program declares
the sampler), the cost is paid once per document because `compile_pending_passes` no-ops when
`program is not None or compile_unit.errors`, and the view is opt-in and per-document. 091's
`plan_import` uses the same seam for the same reason. The comment at the call site says this
correctly. The only defect is the stale claim in `sampler_names`.

Fix: replace the final sentence with something that survives a second consumer — the sweep
brings passes online incrementally for the render path, and a consumer needing every program at
once calls `compile_pending_passes`.

### 15. NIT — `GraphViewState.compiled` is dead

`widgets/graph_state.py::GraphViewState` carries `compiled: bool = False` with the comment "The
compile seam ran for this document (092 D1): every pass has ports thereafter." Nothing reads or
writes it: the widget calls `compile_pending_passes` unconditionally every frame, which is the
right shape given the no-op cost. It is a field documenting an optimization that was
deliberately not taken.

Per `conventions.md`'s speculative-machinery bullet — the test is "is REMOVING it churn?" — a
model-visible field on a state object that no code reads is teach-and-maintain surface with no
consumer. Delete it.

### 16. NIT — `_draw_node`'s `fill` has a dead branch

```python
fill = (
    (*tint[:3], COLOR.GROUP_FILL_ALPHA * alpha)
    if tint is not None
    else fade(COLOR.BG_SURFACE, alpha)
)
dl.add_rect_filled(p0, p1, _u32(fade(COLOR.BG_SURFACE, alpha)), _ROUNDING * z)
if tint is not None:
    dl.add_rect_filled(p0, p1, _u32(fill), _ROUNDING * z)
```

The `else` arm is computed and never used — `fill` is only read under `if tint is not None`,
and the unconditional base fill re-derives `fade(COLOR.BG_SURFACE, alpha)` inline. Harmless,
but it reads as if the base fill were the ternary's job, which it is not.

Fix: drop the ternary, keep the base `add_rect_filled`, and inline the tint color in the
guarded call.

### 17. NIT — `_View` / `_Node` / `_Edge` belong where they are

The question was whether these three dataclasses should move to `graph_state.py`. They should
not. `graph_state.py` holds state that OUTLIVES a frame (pan, zoom, scope, selection, the
in-flight drags) — its docstring says so — while `_Node` / `_Edge` / `_View` are rebuilt from
scratch by `_build_view` every frame and thrown away. Moving them would put a per-frame derived
picture in the module whose stated job is per-document persistence-adjacent state, and would
make `graph_state.py` import `Port` from `pass_graph`, which it does not today. They are also
private to the one module that builds and draws them.

Verdict: correct placement, no change.

### 18. NIT — the App verbs are at the right altitude, with one exception worth naming

Checked each against the three-layer rule (`app.py` owns imgui-bound state and forwards project
ops to `ProjectSession`) and the "every gesture routed through one App verb" decision the
pre-implementation round locked:

- `arrange_graph`, `commit_node_drag`, `group_selection`, `dissolve_group`, `drop_wire`,
  `unwire` — all read `self.ui_documents` / `self.graph_views` (App-owned UI state), decide,
  and forward the write to exactly one `self.session` verb. Correct: the refusal logic needs
  the view state, which the session does not have and must not learn.
- `graph_view_for` — the lazy-create accessor, mirroring the `auto_size_states` /
  `throttle_states` idiom, evicted in `forget_render_state` beside its siblings. Correct.
- `arrange_graph` is the one that carries real computation on App rather than forwarding: it
  rebuilds `sizes` by calling `node_ports` per pass and then `rank_layout`. That same
  computation exists in `widgets/pass_graph.py::_build_view`. Two sites deriving the same
  `sizes` dict from the same three inputs is a smaller instance of finding 1's shape. It is a
  NIT rather than a REAL because a drift here changes only the Arrange layout's spacing, not
  correctness — but a `graph_state.node_sizes(document, wiring)` helper called by both would
  remove it.

`GraphViewState` living on `App` as `graph_views: dict[str, GraphViewState]` follows the
`share_state.py` precedent correctly: the conventions bullet is "Tab state goes on `App`
directly; a state-only sibling module may hold its dataclass to keep `app.py`
import-cycle-free", and that is exactly the shape — the dataclass in `widgets/graph_state.py`,
the dict on `App`. Nothing on `GraphViewState` should be on `App` directly: every field is
per-document, and a per-document dict on App keyed by document id with an eviction in
`forget_render_state` is the established idiom for exactly that.

### 19. NIT — `pass_menu_items`'s extraction is behavior-preserving for the strip

Compared the extracted body against the pre-diff `_draw_context_menu`. Item order, the
`enabled=deletable` plus the Python-side `and deletable` gate (with its `/imgui-ui §7.4`
pointer comment intact), the `len(document.passes) > 1` predicate, the group predicate reading
`document.graph.passes.get(name, PassEntry()).group`, and the `set_pass_group(document_id,
name, "")` call with its notification push — all identical. The only change is that `document`
is now resolved inside `pass_menu_items` rather than in the caller, which is the same lookup.
The strip's own `begin_popup_context_item(f"##pass_menu_{name}")` with its explicit id is
retained, as D10 requires.

Verdict: byte-identical behavior for the strip. No change.

### 20. NIT — the Document tab's `_draw_passes` respects both the row rule and the copilot bracket

The copilot-turn bracket: D2 warned that moving the add/import row out of `pass_list.draw`
without re-establishing the bracket would silently unfreeze it. `_draw_passes` opens
`begin_disabled(app.copilot_turn_active)` three separate times — around the caption and toggle,
then the body dispatch runs outside (each of `pass_list.draw` and `pass_graph.draw` opens its
own), then a third around the add/import row. Every control is covered, and the draw-list
pictures keep painting live inside `begin_disabled`, which is what manual item 36 checks. The
three-bracket shape is slightly noisier than one bracket around the whole function, but it is
necessary: the body must NOT be inside the parent's bracket or the nested `begin_disabled` in
each widget would still be correct but the split would be invisible to a reader.

The label-control row: imgui-ui §2's `label_row` rule governs a fixed-width label COLUMN where
several rows must align their controls. The Passes caption is a section heading with one
control beside it (`small_caption` + `same_line(spacing=SPACE.LG)` + `segmented_choice`), which
is the same shape `_draw_entry_points` uses for its Script row directly above. Not a
label-control row, so the fixed-column rule does not apply.

Verdict: both correct. No change.

### 21. NIT — docstrings follow PEP 257 / Google shape; two are borderline

Checked every new docstring against the `conventions.md` bullet (summary line prescribing the
effect as a command, ending in a period; a blank line; then elaboration; `Args:` / `Returns:`
spelled exactly).

None of the new functions use `Args:` / `Returns:` headings, which is consistent with the rest
of the repo — the gate (`tests/test_script_api_doc.py`) covers the script API surface, and
these are internal helpers whose one-paragraph form matches their neighbors in
`pass_list.py` / `ui_primitives.py`.

Two deviate from the summary-line-then-blank-line shape:

- `_positions` in `pass_graph.py` — its summary runs to two lines with no blank before the
  elaboration, and the summary is a noun phrase ("Every pass's canvas position: …") rather
  than a command.
- `_snap` — same shape, a noun-phrase summary wrapping onto the second line.

Several others share the noun-phrase opening (`node_size`, `group_names_in_order`,
`bundle_output`, `Boundary`). That style is pervasive in `pass_graph.py` already
(`wired_pass`, `strip_order`), so calling it a violation would indict the module rather than
the diff. The specific thing worth fixing is the missing blank line in `_positions` and
`_snap`, where a multi-line summary runs straight into the elaboration.

Verdict: NIT, fix the two.

### 22. NIT — no comment narrates history; the comment budget is respected

Grepped every added comment across both commits for the banned shapes (the bug-we-hit story,
the why-we-changed-it backstory, a "see <doc> for the saga", paragraph-length rationale).
Found none. The comments that exist state a non-obvious present fact and each earns its line:
the `allow_overlap` chain and the `begin_popup_context_item(None)` rule in the module
docstring; "Mapped back by INDEX: a document named like a group would make the name ambiguous"
in `_tab_row`; "Gated in Python, not by `enabled=`" with its `/imgui-ui §7.4` pointer (carried
over, not new); "the seam 091 uses before it plans. A no-op once every pass has been
attempted" at the `compile_pending_passes` call; "Re-keyed IN PLACE and in the same order" in
`wiring_if_renamed`; "Validated, not `model_copy`-ed: a copy skips the field's bounds" in
`with_positions`. The `# ---- section ----` banners are the sanctioned kind.

One borderline: `_draw_canvas`'s `# ---- wheel zoom about the cursor, read before the transform
is used ----` — the trailing clause is an ordering constraint, which is exactly what the rule
admits. Fine.

Verdict: clean. No change.

### 23. NIT — no suppression, no inline import, no `Any` on a real-typed param in the source diff

Grepped the full two-commit diff for `# type: ignore`, `# noqa`, `# pyright: ignore`, and
function-body imports: zero in `shaderbox/`. Every new function and dataclass field carries a
full annotation; no `from __future__ import annotations`; no `if TYPE_CHECKING`; no
`@staticmethod` or `@classmethod` (`NodeDrag.update` / `.current` / `.commit` and
`_Xf.to_screen` / `.to_canvas` all use `self`).

`Any` appears only in the new test files, as `app: Any` on the shared fixture. That is the
repo-wide convention — 285 occurrences of `app: Any` across `tests/`, zero of `app: App`,
and `tests/conftest.py::app` itself is annotated `-> Iterator[Any]`. Not a finding.

`make check` (ruff + pyright) is green, which is consistent.

### 24. NIT — one test name promises more than it asserts

`tests/test_graph_view.py::test_the_widget_makes_no_session_write_of_its_own` reads the
widget's source text and asserts the three strings `set_sampler_source`, `set_pass_positions`
and `set_pass_groups` do not appear. The name claims the widget makes NO session write; the
assertion covers three named methods.

The widget does in fact reach the session, through `pass_menu_items`, which calls
`app.session.set_pass_group(document_id, name, "")` for "Leave group". That is correct
behavior — it is the strip's own shared menu, deliberately unchanged by D10 — but it means the
test's name is false as written while its assertion is true. A reader trusting the name would
conclude the widget cannot touch the session at all.

The other new test names hold up: each `test_drop_wire_*`, `test_unwire_*`,
`test_commit_node_drag_writes_once_and_only_the_moved_passes` and
`test_group_selection_and_dissolve_are_one_write_each` assert exactly what they say, each with
a falsifier comment naming the mutation.

Fix: rename to `test_the_widget_routes_every_graph_write_through_an_app_verb`, which is the
invariant the three strings actually pin.

## The doc edits D20 still owes

None of D20 has landed, which is expected before the sanitize step. For the record, the
complete outstanding list, checked against the tree:

- **`ai_docs/conventions.md`** — the "A pass GROUP is a label… and nothing folds (feature 091)"
  entry is unchanged; it still owes the sentence scoping the no-folding half to the STRIP. And
  the new entry recording D6 (a position is written only by a placement, never by a draw) and
  D1 (ports from the program, edges from the wiring) does not exist.
- **`.claude/skills/imgui-ui/SKILL.md` §8** — neither canvas rule is present: not the
  allow-overlap submission chain (background first, then each node, each declaring
  `set_next_item_allow_overlap()`), not the `begin_popup_context_item(None)` rule for a shared
  child. Both are live footguns the next canvas will hit.
- **`ai_docs/dev_flow.md`'s Module map** — `widgets/pass_graph.py` and
  `widgets/graph_state.py` have no entry. Confirmed by reading the map end to end.
- **`ai_docs/dev_flow.md`'s `widgets/pass_list.py` entry is now STALE**, not merely
  incomplete. It says: "`add pass` opens the settings modal on `App.pass_draft` (078),
  `import...` opens the import dialog on `App.import_draft` (091)". Both buttons moved to
  `tabs/document.py::_draw_passes` in this diff. The entry also does not mention that the
  caption moved or that the strip is now one of two views.
- **`shaderbox/help_content.py`** — this one DID land (the Passes section gained "On the graph
  view a port exists because the shader declares a sampler, so a new pass has none until you
  add one."), as did the import dialog's "groups flattened" line in
  `popups/import_passes.py`. Two of D20's items are done.
- **`ai_docs/features/070_pass_reads/01_spec.md`** — the pointer to 092 is not there.
- **`ai_docs/roadmap.md`** — the Active-context banner and the 092 row.

Beyond D20's own list, findings 11 and 14 each ask for a doc edit that D20 did not anticipate:
the deliberate asymmetry between the combo's texture release and the canvas's refusal, and
`sampler_names`'s now-false closing sentence.

## False trails

Things this review chased and found sound — recorded so the next reviewer does not re-spend
the time:

- **`pass_menu_items`'s extraction is not a behavior change for the strip.** Diffed item by
  item against the pre-extraction body; identical including the `enabled=` plus Python gate
  and the `§7.4` comment.
- **`wiring_if_renamed` on `Document` is not a layering violation.** It must see live passes to
  reach `effective_wiring`; a pure version would duplicate the name rule.
- **`compile_pending_passes` per canvas frame does not invert 066 D1.** It no-ops after the
  first attempt per pass, the canvas genuinely needs every program, and the seam is 091's.
- **Nothing on `GraphViewState` belongs on `App` directly.** Every field is per-document and
  the per-document-dict-with-eviction idiom is the established one.
- **`_Node` / `_Edge` / `_View` do not belong in `graph_state.py`.** Per-frame derived data
  versus per-document state; moving them would also add an import.
- **`_dashed_rect` / `_draw_port_dot` / `_u32` do not duplicate a `ui_primitives` helper.**
  Checked `close_cross_button`, `_glyph_button`, `text_chip`, `faint_hline`, `step_squares`,
  `gauge_bar`, `cell_delete_confirm`. Only the badge (finding 6) has a real claim.
- **`text_chip` cannot be reused for the node badges.** It draws at the imgui cursor and calls
  `imgui.text_colored`; the badges are draw-list geometry at a computed screen point inside a
  channel split.
- **The `app: Any` annotations in the new tests are the repo convention, not a typing gap.**
- **The Passes caption row is not a label-control row.** It is a section heading plus one
  control, the same shape as the Script row directly above it.
- **The copilot-turn bracket was not lost when the buttons moved.** Three brackets in
  `_draw_passes` cover the caption, the body (via each widget's own) and the button row.
- **The button-tier rule is not violated.** Every labelled verb on the canvas goes through
  `primary_button` / `standard_button` / `menu_item_simple`; the raw `invisible_button` calls
  are hit rects and are allowlisted in `tests/test_button_tiers.py` in the same commit, as D20
  required.
- **No migration code anywhere in the diff.** `position` defaults to `None`, `graph.json`'s
  version does not bump, `load_graph`'s per-entry salvage carries the field, and
  `import_passes` strips the source's positions with `model_copy(update={... "position":
  None})` rather than reading an old shape.
- **D19 holds: the copilot sees nothing new.** No tool signature changed, no prompt block
  gained a groups or positions paragraph, and `set_pass_group` survives as a one-name wrapper
  over `set_pass_groups` so the copilot's existing call is unchanged.
- **The gates are genuinely green**, exit code 0 captured unpiped, smoke ran rather than
  skipped.

## Verdict

**FINDINGS.**

Must be fixed before this feature closes: **1, 2, 3, 5, 6, 8, 11, 12, 14**, plus the D20 doc
edits (the conventions entry, the imgui-ui §8 canvas rules, the module map's two new entries
and the stale `pass_list.py` entry, the 070 pointer, the roadmap).

Finding 11 is the one that needs a maintainer decision rather than an edit: whether the
uniforms combo's silent texture release is intended. Either answer is defensible; what is not
defensible is leaving the asymmetry unrecorded.

Findings 4, 7, 9, 10, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 are NITs — worth sweeping
together with the token work of findings 2 and 5, since they touch the same lines.

Nothing in this diff should not have landed. The architecture is right: the pure half is in
`pass_graph.py` and tested without a window, the state is in `graph_state.py` with the drag as
a pure machine, every gesture routes through one App verb, the widget is a leaf that positions
no sibling, and the one save per gesture is asserted under the real frame loop. The findings
are, with the exception of 11 and 12, a token-and-duplication sweep rather than a design
problem.
