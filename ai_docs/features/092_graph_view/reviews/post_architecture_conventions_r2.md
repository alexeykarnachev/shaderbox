# 092 post-implementation review — architecture and conventions, round 2

Closing round 1 (`post_architecture_conventions.md`, findings 1-24) item by item against the fix
commit `ed1c28b` ("092: fold the post-implementation round 1") and the D20 doc commit folded into
it. Angle unchanged: where things live, whether anything duplicates what already exists.

Gate state, captured unpiped: `make check` -> **0**. `make gates > /tmp/g.log 2>&1; echo $?` -> **0**,
last line `gates: GREEN -- check passed, test passed, smoke passed`; the smoke ran, it is not a skip.

## Closure table

| # | R1 severity | Verdict | What closes it (or what is still there) |
|---|---|---|---|
| 1 | REAL | **CLOSED** | `SIZE.GRAPH_PORT_TOP` in `theme.py`, with the comment naming both readers. `graph_state.node_size` uses `SIZE.GRAPH_PORT_TOP + port_count * SIZE.GRAPH_PORT_ROW`; `pass_graph._port_point` uses `SIZE.GRAPH_PORT_TOP + slot * SIZE.GRAPH_PORT_ROW + ...`. No `4.0` remains at either site. |
| 2 | REAL | **PARTIAL — see new finding A** | Five of six landed: `SIZE.GRAPH_ROUNDING` (read in `_draw_node` as `rounding`), `GRAPH_DASH` (`_dashed_rect` call), `GRAPH_THUMB_ROUNDING` (`add_image_rounded`), `GRAPH_BUS_CLEAR` (`_draw_canvas`'s bus), `GRAPH_WIRE_W` / `GRAPH_PORT_RING_W` (defined and read in `_draw_wire`, `_draw_self_loop`, `_draw_port_dot`). Still bare, and now WORSE than in round 1 because the token exists beside them: `1.5`/`1.0` in `_draw_node`'s `thickness`, `1.2` in `_draw_node`'s hollow output dot, `1.5` in `_draw_canvas`'s in-flight wire. |
| 3 | REAL | **CLOSED** | `_MIN_DIRECT_DX = 24.0` with the comment fixing the coordinate space ("canvas units… One constant, read in one coordinate space"). Read at four sites: `_draw_wire`'s `dx >= _MIN_DIRECT_DX * z`, its `c` floor, its two bus elbows, and `_draw_canvas`'s `backward = b[0] < a[0] + _MIN_DIRECT_DX` — the canvas test is canvas-space and the screen test multiplies by `z`, so they agree at every zoom. The in-flight wire reads it too. |
| 4 | NIT | **CLOSED** | `_NONE_CORE = 0.45`, `_PREV_INNER = 0.5`, `_MEDIA_HALF = 0.8`, `_BEZIER_BOW = 0.45`, each a module constant under a comment saying what it is a fraction of. `_draw_port_dot` and both bezier sites read them; the `0.45` collision is now two named facts. |
| 5 | REAL | **CLOSED** | Five alphas added to `_ColorBag` beside `GRAPH_GHOST_ALPHA`: `GRAPH_STALE_ALPHA`, `GRAPH_DIM_ALPHA`, `GRAPH_BAND_FILL_ALPHA`, `GRAPH_BAND_EDGE_ALPHA`, `GRAPH_GUIDE_ALPHA`. Each read once in `pass_graph.py`; no bare alpha remains in the widget. |
| 6 | REAL | **CLOSED** | `_draw_badge(dl, corner, right_aligned, label, z, bg, fg)` with `_BADGE_PAD` / `_BADGE_H` / `_BADGE_INSET` beside it; both the `x{runs}` and the `{n} passes` sites call it, and the drifted one-pixel text offset is gone — the text sits at `x0 + _BADGE_PAD * z` in both. Kept local rather than promoted to `ui_primitives`, which R1 explicitly offered as the maintainer's option. |
| 7 | NIT | **ACCEPTED-AS-IS** | Code unchanged (`channels_split(2)` + foreground after `channels_merge()`, which R1 itself called the better shape). The spec's D7 body and D12/D13 still read "channel 2", but the Review history's Deviations paragraph now states it: "D7 splits two channels and draws the foreground after the merge". Not worth a further edit — the deviation register is where the spec records this class, and re-writing D7's body would erase what was designed. |
| 8 | REAL | **CLOSED** | `_u32(fade(COLOR.WHITE, picture_alpha))` in `_draw_node`'s `add_image_rounded`. No raw color tuple remains in either widget module; the only tuple-shaped color is `(*tint[:3], COLOR.GROUP_FILL_ALPHA * alpha)`, which is token-derived. |
| 9 | NIT | **ACCEPTED-AS-IS** | `_u32` unchanged and still private. R1's own verdict was "leave it"; nothing in the fix commit grew a second copy, and no third canvas exists. |
| 10 | NIT | **CLOSED (no change needed)** | `_dashed_rect` / `_draw_port_dot` still private free functions; `_draw_badge` joined them with the same shape. Re-checked `ui_primitives.py` for a badge-shaped primitive that appeared since round 1: none. |
| 11 | REAL | **ACCEPTED-AS-IS (maintainer's recorded default)** | Behavior unchanged on both surfaces; the asymmetry is now recorded in two places, as R1 asked. `set_sampler_source`'s docstring: "…so choosing a source here replaces it -- and releases it. That is deliberate for the panel's combo, a per-sampler pick the user is looking at; the graph canvas refuses a wire dropped on a media-bound port before reaching here (092 D15), because a drag is a coarser gesture than a pick. The asymmetry is by design." The 092 `conventions.md` entry carries the same sentence. The spec's Review history names it as the maintainer's default on the one design fork. |
| 12 | REAL | **CLOSED** | `pass_graph.namespace_error(candidate, pass_names, group_names)` and `NAMESPACE_COLLISION`, with the docstring stating the single-predicate intent. Three callers, each passing its own half: `group_name_error` (`namespace_error(group, pass_names, ())`), `project_session._pass_name_error` (`namespace_error(name, (), {…entry.group…})`), and `pass_import.plan_import` (`namespace_error(new, (), host_groups)`). The message literal exists once. `plan_import` gained `host_groups` as a required parameter, and both call sites pass it. |
| 13 | NIT | **CLOSED (no change needed)** | `wiring_if_renamed` unchanged on `Document`; R1's verdict stands and nothing in the fix moved it. |
| 14 | REAL | **CLOSED** | `document.sampler_names`'s closing sentence now reads: "The render path brings passes online incrementally through its first-render sweep; a consumer that needs every program at once (the import dialog, the graph canvas) calls `compile_pending_passes` first." Survives a third consumer. |
| 15 | NIT | **CLOSED** | `GraphViewState.compiled` is gone; `grep compiled shaderbox/widgets/graph_state.py` hits only the word inside `ports_of`'s docstring. |
| 16 | NIT | **CLOSED** | The `fill` ternary is gone. `_draw_node` now does one unconditional `add_rect_filled(… fade(COLOR.BG_SURFACE, alpha) …)` and one guarded tint fill with the color inlined. |
| 17 | NIT | **CLOSED (no change needed)** | `_View` / `_Node` / `_Edge` still private to `pass_graph.py`. `graph_state.py` gained `ports_of` / `node_sizes` but no per-frame picture type. |
| 18 | NIT | **CLOSED** | The `sizes` duplication R1 named is gone: `graph_state.ports_of(document, wiring)` and `graph_state.node_sizes(ports)` are the one derivation, called by `App.arrange_graph` and by `pass_graph._build_view`. `app.py` dropped its `node_ports` / `sampler_names` / `node_size` imports for `node_sizes, ports_of`. |
| 19 | NIT | **CLOSED (no change needed)** | `pass_menu_items` unchanged except the `Leave group` body, which is finding 24's territory and now calls `app.leave_group`. |
| 20 | NIT | **CLOSED (no change needed)** | `_draw_passes`'s three `begin_disabled` brackets intact; the function gained only the height reservation. |
| 21 | NIT | **CLOSED** | Both named docstrings fixed. `_positions`: summary "Place every pass on the canvas without writing anything." (one line, a command, period), blank line, elaboration. `_snap`: "Set the drag's snap offset and return the guides to draw (092 D13)." same shape. |
| 22 | NIT | **CLOSED (no change needed)** | Re-grepped every comment the fix commit added for the banned shapes. The cancel block's comment — "A gesture whose release the canvas did not see (the view switched, a modal covered it, a copilot turn began) is cancelled, never resumed: a stray later click must not write." — states what the code does as it is now and why the cancel is unconditional. No bug-we-hit narration; the story is in the commit message and the spec's Review history, where it belongs. Same for the port hit box's comment and `App.leave_group`'s docstring. |
| 23 | NIT | **CLOSED (no change needed)** | `git show HEAD -- shaderbox/` grepped for `# type: ignore`, `# noqa`, `# pyright: ignore`: zero. Grepped added lines for a function-body import: zero. No `from __future__ import annotations`, no `if TYPE_CHECKING`, no `@staticmethod` / `@classmethod` in the new code. |
| 24 | NIT | **CLOSED — differently than proposed** | The name was kept and the ASSERTION was broadened instead, which makes the name true. `test_the_widget_makes_no_session_write_of_its_own` now loops over `(pass_graph, pass_list)` and forbids four spellings: `set_sampler_source`, `set_pass_positions`, `set_pass_groups`, `set_pass_group(`. The one escape R1 found (`pass_menu_items`'s direct `app.session.set_pass_group`) was closed by routing it through the new `App.leave_group`, so the module really does make no session write. Stronger than the rename R1 suggested. |

## The literal-number table, re-done

`widgets/graph_state.py` — **clean**. Every number in it is `0.0` / `1.0` (an origin, an identity
zoom, a delta accumulator) or an index. `node_size`'s height reads `SIZE.GRAPH_PAD`,
`GRAPH_THUMB`, `GRAPH_NAME_H`, `GRAPH_PORT_TOP`, `GRAPH_PORT_ROW`; its width `GRAPH_NODE_W` and
`GRAPH_BOX_EXTRA_W`. No verdict-worthy literal remains.

`widgets/pass_graph.py`:

| Literal | Site | Verdict |
|---|---|---|
| `_ZOOM_STEP = 1.1` | module constant | named local — correct, a gesture rate |
| `_MIN_DIRECT_DX = 24.0` | module constant, four readers | named local — correct, and the coordinate space is now stated |
| `_BEZIER_BOW = 0.45` | module constant, two readers | named local — correct |
| `_NONE_CORE` / `_PREV_INNER` / `_MEDIA_HALF` | module constants, `_draw_port_dot` | named locals — correct |
| `_FIT_MARGIN = float(SPACE.LG)` | module constant | token-derived — correct |
| `_CYCLE_PREFIX`, `_ROOT_LABELS` | module constants | named locals — correct (but see finding B for `_ROOT_LABELS`'s length) |
| `_BADGE_PAD = 3.0`, `_BADGE_H = 12.0`, `_BADGE_INSET = 2.0` | module constants, `_draw_badge` | named locals — acceptable. They are the internal proportions of one private helper with one geometry, and unlike a corner radius or a stroke width no other module can want them. Arguable against §6's "a token used by exactly one panel still belongs in the token bag"; recorded as a judgement call, not a defect, since promoting three constants that parameterize a single private draw helper adds a `theme.py` edit for no second reader. |
| **`1.5` / `1.0`** | `_draw_node`'s `thickness = 1.5 if (node.error or is_output or selected) else 1.0` | **should be `SIZE.GRAPH_WIRE_W` (the token now exists) and a `1.0` hairline floor** — finding A |
| **`1.2`** | `_draw_node`'s hollow output dot, `dl.add_circle(center, r, out_col, 0, 1.2)` | **should be `SIZE.GRAPH_PORT_RING_W`** — the token was created in this very commit for this exact stroke and three of its four sites were converted; this one was missed. Finding A |
| **`1.5`** | `_draw_canvas`'s in-flight wire, `max(1.0, 1.5 * view.zoom)` | **should be `SIZE.GRAPH_WIRE_W`** — the same wire width as `_draw_wire`, which reads the token two hundred lines up. Finding A |
| **`12 * z`** | `_draw_self_loop`, the loop's rise over the node | **should be a token or a named local** — unchanged from round 1, where it was listed as `SIZE.GRAPH_LOOP_RISE`. Finding A |
| **`28 * z`** (twice) | `_draw_self_loop`, the loop's horizontal control reach | **should be a named local** — unchanged from round 1. Finding A |
| `2.0` / `/ 2.0` / `2 *` | `_thumb_rect`, `_out_point`, `_fit`, `_port_point`, the hit rects' `2 * hit`, `2 * dash` | fine — arithmetic halving and doubling, and `2 * dash` encodes the stated dash-equals-gap relation |
| `max(4.0, …)` | the two `push_font` size floors | fine — a floor against a degenerate rasterized size |
| `2 * r + 2 * z` | the port label's offset from its dot | still a bare `2 * z`, but it is one term of a spacing expression whose other term is `r` (token-derived); reading it as an independent design number over-reads it. fine |
| `2` | `channels_split(2)` / `channels_set_current(0/1)` | fine — a channel count (finding 7) |
| `max(0, edge.span - 2)` | the bus step multiplier | fine — rank arithmetic |
| `0.0`, `1.0`, `0`, `1` | rounding args, thickness args, `add_image_rounded` UVs, `pop_style_color(1)`, segment counts | fine |
| `max(1.0, avail.x)` etc. | degenerate-size floors | fine |

Colors: **clean**. `_draw_node`'s tint is `_u32(fade(COLOR.WHITE, picture_alpha))`; every other
color reaches through `COLOR.*` / `fade(…)` / `group_tint(…)`. The one tuple expression,
`(*tint[:3], COLOR.GROUP_FILL_ALPHA * alpha)`, is token-derived and matches `pass_list.py`.

## The D20 doc edits

All six landed and read correctly.

- **`ai_docs/conventions.md`, the 091 bullet** — gained the scoping sentence: "That no-folding
  half is about the STRIP: the graph view (092) contracts a group to a box whose ports are its
  boundary edges, and since the box is never a node the planner orders, convexity is not a rule
  there either." It answers the exact question R1 raised (why 091's convexity rule does not bind
  092) and preserves the original "Revisit if…" clause. Correct.
- **`ai_docs/conventions.md`, the new 092 bullet** — present, and it carries what the altitude
  asks for: the two-sources-of-truth rationale (a wiring-built port list would have no dot to
  drop on; a stored-row port list would draw a sampler the program no longer declares), the box
  and the ghost tab, `PassEntry.position` bounded through `with_positions`, the bolded
  write-only-by-a-placement rule, the six App verbs by name, the one namespace, and a
  "Revisit if" clause on the canvas's home. The **media-asymmetry sentence is there**: "One
  asymmetry is deliberate: the uniforms panel's combo replaces (and so releases) a bound texture
  on a pick, while the canvas refuses a wire dropped on a media-bound port before the write -- a
  pick is a per-sampler choice the user is looking at, a drag is a coarser gesture." That is the
  minimal behavior-free edit finding 11 asked for.
- **`ai_docs/dev_flow.md`, the `pass_list.py` entry** — no longer claims the buttons. The stale
  "`add pass` opens the settings modal on `App.pass_draft` (078), `import...` opens the import
  dialog on `App.import_draft` (091)" is replaced by "Since 092 the caption, the `strip | graph`
  toggle and the `add pass` / `import...` row are the Document tab's
  (`tabs/document.py::_draw_passes`), drawn over whichever view is on." It also now names
  `pass_menu_items` as the item set shared with the graph's node menu. Correct.
- **`ai_docs/dev_flow.md`, the two new widget entries** — `widgets/pass_graph.py` and
  `widgets/graph_state.py` both present in the module map, in the widgets block. The
  `pass_graph.py` (pure) entry also gained its new sentence naming the canvas's pure half
  including `namespace_error`. Correct.
- **`.claude/skills/imgui-ui/SKILL.md` §8** — both canvas rules present. The allow-overlap rule
  states the chain in the right direction ("goes on the item submitted FIRST, not the one on
  top") and names the failure mode the wrong order produces ("each drag pans the canvas, which
  reads like a coordinate bug and is not"). The `begin_popup_context_item` rule states the
  binding's own docstring and explains why the strip's tiles get away with an explicit id (each
  tile is its own child window). Both cite `widgets/pass_graph.py::_draw_canvas` as the measured
  instance, which is the skill's own convention. Correct.
- **`ai_docs/features/070_pass_reads/01_spec.md`** — the pointer is in the status paragraph:
  "The graph view returned in 092 as an opt-in SECOND view beside the strip, not as its
  replacement (`ai_docs/features/092_graph_view/03_spec.md`); the decision below stands for the
  strip." Correct — it is scoped, it does not rewrite 070's rejection, and it names the file.
- **`ai_docs/roadmap.md`** — the 092 row is present at the head of the table with status
  **`in progress`**, as this round requires. The banner is untouched, correctly deferred to the
  sanitize step.

## New code the fixes added, against the conventions

- **`_draw_badge` + its three constants** — a private module free function, full annotations, a
  one-line PEP-257 summary ("A small pill with a word on it at a picture's corner, in the current
  font."). No `@staticmethod`. See the literal table for the constants' judgement call.
- **`ports_of` / `node_sizes` in `graph_state.py`, and the layering question** — **no cycle, no
  layering problem.** Verified by import rather than by reading: `uv run python -c "import
  shaderbox.widgets.graph_state"` succeeds standalone, and `make check` (pyright) is clean. The
  direction is `app.py -> widgets/graph_state.py -> {document.py, pass_graph.py, theme.py}`;
  `document.py` imports no `widgets` module (grepped: zero hits) and `pass_graph.py` imports no
  `widgets` module either. `graph_state.py` still imports nothing from `widgets/pass_graph.py`,
  so R1's finding 17 reasoning survives. The module's docstring still says it holds the canvas's
  per-document state, and `ports_of` / `node_sizes` are derivations over a document rather than
  state — a mild widening of the module's stated job, but the alternative homes are worse:
  `pass_graph.py` (pure) cannot import `Document`, and `app.py` was where the duplication was.
  Correct as placed.
- **`namespace_error` / `NAMESPACE_COLLISION` in `pass_graph.py`** — right altitude. `pass_graph.py`
  is the pure module both `project_session.py` and `pass_import.py` already import, and the
  constant plus the function give the invariant one name and one message. The docstring states
  the intent ("Both directions call this, so the predicate and its message exist once"), which is
  the fact a future fourth caller needs. The `Collection[str]` parameters let each caller pass its
  own half with `()` for the other, which reads slightly oddly at the call sites but keeps one
  signature. Correct.
- **`App.leave_group`** — correct altitude. It reads no App-owned UI state, so it is a pure
  forward to `self.session.set_pass_group` plus the notification push — the same shape as
  `dissolve_group`'s tail. Its reason for existing is structural rather than computational and the
  docstring says so ("routed here so every canvas write is an App verb"), which is exactly the
  invariant `test_the_widget_makes_no_session_write_of_its_own` now enforces over both modules.
  One caller in `shaderbox/` (`pass_list.pass_menu_items`) plus the test.
- **The cancel block's comment** — states the present rule and the reason the cancel is
  unconditional, in two lines. No development history. Within budget.
- **`_ROOT_LABELS` fallback** — see finding B.
- **Docstrings** — every new or edited docstring (`_draw_badge`, `ports_of`, `node_sizes`,
  `namespace_error`, `App.leave_group`, `_positions`, `_snap`, `set_sampler_source`'s addition,
  `sampler_names`'s addition, `draw`'s height sentence) opens with a command-form summary ending
  in a period, and every multi-paragraph one has the blank line. None uses `Args:` / `Returns:`,
  consistent with the repo's internal-helper style.
- **Suppressions / inline imports** — zero added, verified by grepping the diff for
  `# type: ignore`, `# noqa`, `# pyright: ignore`, and for indented `from` / `import` lines.

## New findings

### A. REAL — four stroke widths in `pass_graph.py` stayed bare while their token was being created

The finding-2 sweep created `SIZE.GRAPH_WIRE_W = 1.5` and `SIZE.GRAPH_PORT_RING_W = 1.2` and
converted most sites, and missed these:

- `_draw_node`: `thickness = 1.5 if (node.error or is_output or selected) else 1.0` — the node
  border's emphasized width, the same `1.5` the token holds.
- `_draw_node`: `dl.add_circle(center, r, out_col, 0, 1.2)` for a box's hollow output dot — the
  ring stroke `GRAPH_PORT_RING_W` was created for, and the three sibling rings inside
  `_draw_port_dot` DO read it. One drawing of a ring reads the token, another two hundred lines
  down does not.
- `_draw_canvas`: the in-flight wire's `max(1.0, 1.5 * view.zoom)` — `_draw_wire` spells the same
  thing `max(1.0, SIZE.GRAPH_WIRE_W * z)`.

This is a worse state than round 1's, not merely an unchanged one: in round 1 the number was
magic everywhere and consistently so; now the same visual decision has two spellings, and
retuning the wire width via the token silently leaves the in-flight wire and the node border
behind. That is the lockstep failure mode `conventions.md` names, in the file the sweep just
touched.

Separately and unchanged from round 1: `_draw_self_loop`'s `12 * z` (the loop's rise) and its two
`28 * z` (the horizontal control reach) are still bare. R1 listed both; neither was converted nor
recorded as accepted.

Fix: read `SIZE.GRAPH_WIRE_W` at the two `1.5` sites and `SIZE.GRAPH_PORT_RING_W` at the `1.2`
site; give the self-loop's two numbers a token (`GRAPH_LOOP_RISE`) and a module constant
(`_LOOP_REACH`), or record in the spec's deviation paragraph that they were judged not worth it.

### B. REAL — `_ROOT_LABELS`'s fallback can raise `StopIteration`, in code the fix added

`_tab_row` picks the root tab's label:

```python
root_label = ui_document.ui_state.ui_name.strip()
if not root_label or root_label in groups:
    root_label = next(label for label in _ROOT_LABELS if label not in groups)
```

`_ROOT_LABELS = ("document", "root", "all")`, and a group name is validated only against
`PASS_NAME_RE = ^[A-Za-z_][A-Za-z0-9_]*$` — so all three are legal group names. A document whose
name is blank (or equals one of its group names) and which carries groups named exactly
`document`, `root` AND `all` exhausts the generator and `next` raises `StopIteration` inside the
draw, taking the frame down. Demonstrated:

```
>>> next(l for l in ('document','root','all') if l not in ['document','root','all'])
StopIteration
```

Contrived, yes — three specific group names at once. But this is new code written by the fix for
finding-class "the tab row mapped a click by name", it is an unhandled exception rather than a
wrong picture, and the escape costs one token: `next((… ), _ROOT_LABELS[0])` with a disambiguating
suffix, or simply a `default=` that accepts a duplicate label since the row keys by index-of-click
against `labels` and `labels.index(clicked)` would then return the first match — which is why the
fix chose distinct names in the first place. The cheap correct fix is a generated label
(`f"{_ROOT_LABELS[0]}_{n}"` counting up) rather than a fixed three-element tuple.

Severity REAL rather than NIT because it is a crash, not a cosmetic defect, and because the fix
is one line.

### C. NIT — `plan_import`'s namespace check runs over the renamed set only

`plan_import` now does:

```python
for new in sorted(renames.values()):
    collision = namespace_error(new, (), host_groups)
```

which is the right direction (a copied pass may not be named like a host group) and closes
finding 12's half of the gap. The mirror — the imported passes' own GROUP labels against the
host's PASS names — is not checked here, but `import_passes` strips the source's groups (the
import dialog's own "groups flattened" line says so), so there are no incoming group names to
collide. Recorded so the next reader does not read the one-sided check as an oversight. No change.

## False trails

Chased this round and found sound; recorded so round 3 does not re-spend the time.

- **`graph_state.py` importing `document.py` is not a cycle and not a layering inversion.**
  Verified by a standalone import and by pyright, not by reading: `document.py` imports no
  `widgets` module, `pass_graph.py` (pure) imports no `widgets` module, and `app.py` importing
  `graph_state` is the same direction it already imported it in.
- **`ports_of` / `node_sizes` living in `graph_state.py` is not a misfile.** The two alternatives
  are worse — `pass_graph.py` is pure and cannot see `Document`, and `app.py` is where the
  duplication was. Finding 18 is genuinely closed by this, not merely moved.
- **Finding 24 is closed even though the test name was not changed.** The assertion was broadened
  instead (two modules, four spellings) and the one real escape was routed through `App.leave_group`,
  so the name became true. Do not re-file this as "the rename never happened".
- **Finding 7's spec sentence was not edited in D7's body, and that is deliberate.** The Review
  history's Deviations paragraph records it; the spec's convention is that the body describes the
  design and the deviation register describes what shipped.
- **`_BADGE_PAD` / `_BADGE_H` / `_BADGE_INSET` as module constants rather than `SIZE.*` tokens is
  a judgement call, not a missed conversion.** They parameterize one private helper's internal
  geometry, and no second reader can exist for them. Do not re-file as finding 2.
- **The `2 * z` in the port label's offset (`2 * r + 2 * z`) is not a magic px.** It is one term of
  a spacing expression whose dominant term is the token-derived radius.
- **`namespace_error`'s `()` arguments at two of three call sites are not a smell.** Each caller
  genuinely knows only one half of the namespace at that point, and one signature is the point of
  the funnel.
- **`App.leave_group` forwarding with no computation is not a pointless wrapper.** It exists so the
  no-session-write gate can be a source-text assertion over the whole module, which is what makes
  the D12 invariant testable without a window.
- **The gates are genuinely green**, exit 0 captured unpiped on both `make check` and `make gates`,
  and the smoke ran rather than skipping.

## Verdict

**FINDINGS** — two, both small and both one-line fixes.

- **A (REAL)** — four stroke widths (`1.5` twice, `1.2` once, plus the self-loop's `12 * z` and
  two `28 * z`) stayed bare while `SIZE.GRAPH_WIRE_W` and `SIZE.GRAPH_PORT_RING_W` were created
  for exactly them in the same commit. Two spellings of one visual decision.
- **B (REAL)** — `_ROOT_LABELS`'s three-element fallback can raise `StopIteration` in the draw.

Round 1's nine must-fixes: **8 CLOSED** (1, 2 partially — the token bag landed, three sites were
missed; 3, 5, 6, 8, 12, 14), **1 ACCEPTED-AS-IS** on the maintainer's recorded default (11). All
fifteen NITs are closed, closed-as-no-change, or accepted with a stated reason. All six D20 doc
edits landed and read correctly, the roadmap row is `in progress` as this round requires, and
`make check` exits 0.

The architecture verdict is unchanged and strengthened: the pure half is in `pass_graph.py`, the
per-document state in `graph_state.py`, the one namespace has one predicate, every canvas write is
an App verb now including `Leave group`, and the test that pins it covers both modules that draw
the shared menu.
