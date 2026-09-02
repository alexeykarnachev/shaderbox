# W-D post-implementation code review (correctness)

Commit under review: `f18a7d3` "069 W-D: a sampler's name is its default wire", 26 files.
Read end-to-end; every claim below was produced by running code, not by reading it.

## Verdict

| Area | Verdict |
| --- | --- |
| Resolution rule (`effective_inputs` / `_auto_source`) | **PASS** |
| Effective graph (`effective_graph` / `_sampler_names` / `_is_user_bound`) | **PASS** |
| Render seam | **PARTIAL** — F1 |
| Planner across frames | **PASS** |
| Gear (three-state combo) | **PASS** |
| Strip | **PARTIAL** — F4 (dead parameter) |
| Renames | **PASS** |
| `has_feedback` | **PASS** |
| Tests | **PARTIAL** — F3 (one falsifier does not go red) |
| Conventions | **PARTIAL** — F5 |

`make gates` is GREEN by exit code, captured unpiped: `EXIT=0`, check passed, test passed,
smoke **passed** (not skipped). Full suite 1629 passed, 4 skipped.

---

## F1. A genuinely unwired sampler reads the shipped PHOTO, while the gear and the copilot both say it reads black

**Severity: highest.** This is the exact failure 065 D3 exists to prevent, and it is the one
case the commit message claims to have closed.

**Claim.** The commit message says an explicit none "binds the 1x1 black texture at the render
seam -- left unbound it fell through to the seeded default photo, which is exactly the
mis-wire-shows-a-picture failure 065 D3 exists to prevent." That is true for the `""` case and
untrue for the ABSENT-and-unresolved case, which is the more common one: a sampler whose name
matches no pass, or whose name has no `u_` prefix, still falls through to the photo.

**Evidence.** A pass declaring `uniform sampler2D u_nosuchpass` added to the Radiance Cascades
example, rendered until every pass is online:

```
stored key   effective      what the pass actually reads
absent, name matches nothing   {}   max_rgb=255, 41319 unique colours  <- the shipped photo
absent, no u_ prefix (`tex`)   {}   max_rgb=255, 41319 unique colours  <- the shipped photo
explicit ""                    {}   max_rgb=0                          <- black, correct
stale explicit name            {u_nosuchpass: ghostpass}  max_rgb=0    <- black, correct
```

The uniform's value at that moment is `Image` with `is_default_image=True`, and the render is
the 960x1280 photo scaled to the canvas. Three surfaces disagree about the same sampler in the
same frame:

```
GEAR would show:  auto: none   (selected, index 0)
COPILOT row:      u_nosuchpass <- (nothing; reads BLACK)
ACTUAL render:    max_rgb=255, 41319 unique colours
```

The spec is unambiguous about which is right (`01_spec.md` § W-D, decision 2): "the effective
input of a sampler with an ABSENT key is `u_<x>` -> pass `<x>` when such a pass exists (`u_prev`
-> the pass itself), **else black**." The wave doc notices the fall-through at its
`projects/dev` check ("falls through to the seeded default image exactly as it does today") and
treats it as out of scope, but the wave simultaneously shipped a gear label that asserts
`auto: none` and a copilot row that asserts `reads BLACK` for that state. Before the commit no
surface made the claim; now two do, and both are false.

**Fix.** In `Document.render`'s input-binder, bind `self._black_texture()` for every sampler in
`self._sampler_names(render_pass)` that the effective graph leaves unresolved, not only for
those whose stored entry is `""` — i.e. seed the `inputs` dict from the sampler list and let the
resolved edges overwrite it, so "unresolved" and "explicit none" reach the same black texture.

## F2. `_pass_views` reads the effective graph BEFORE the loop that compiles the passes, so the first working-set read of a name-wired document reports every input as black

**Claim.** `copilot/backend.py::_pass_views` hoists `resolved = document.effective_graph()`
above the per-pass loop, but the loop body calls `_sampler_uniform_names(render_pass)`, which
goes through `Pass.get_active_uniforms()` and COMPILES the pass. So the sampler rows exist
because of a compile that happened after the graph was resolved, and `effective_graph` saw
`program is None` for every one of them.

**Evidence.** Bloom with every stored edge stripped, so the name rule alone carries it, read
twice with no render in between:

```
call 1 (nothing compiled):
   blur      ['u_bright <- (nothing; reads BLACK)']
   bright    ['u_scene <- (nothing; reads BLACK)']
   composite ['u_scene <- (nothing; reads BLACK)', 'u_blur <- (nothing; reads BLACK)', ...]
   trail     ['u_scene <- (nothing; reads BLACK)', 'u_prev <- (nothing; reads BLACK)']
call 2 (now compiled):
   blur      ['u_bright <- bright']
   bright    ['u_scene <- scene']
   composite ['u_scene <- scene', 'u_blur <- blur', 'u_trail <- trail']
   trail     ['u_scene <- scene', 'u_prev <- trail']
```

This is precisely the false fact the commit message says the change exists to remove ("telling
the model it reads BLACK while the renderer fills it is a false fact"). It is narrower than the
message implies: it self-heals on the second read, and a document the user has been looking at
is already compiled. It bites the copilot's first turn on a freshly opened, never-rendered
document — which is a normal opening move.

**Fix.** Resolve the graph after the sampler names are known: collect
`_sampler_uniform_names(render_pass)` for every pass first (that is the call that compiles),
then call `document.effective_graph()` once, then build the rows.

## F3. `test_u_df_beside_df_renders_without_the_gear` does not go red when `render` plans the RAW graph

**Claim.** The commit message lists "the raw-graph planner" among the falsifiers that were run
and went red. It goes red for the STRIP (`test_an_auto_wired_ancestor_is_not_washed_stale`),
but not for the RENDER seam, which is the load-bearing half.

**Evidence.** Replacing `resolved_graph = self.effective_graph()` with `resolved_graph =
self.graph` in `Document.render` and running the two suites the wave added:

```
tests/test_default_wiring.py tests/test_pass_verbs.py -q  ->  40 passed
```

The whole suite stays green. The reason is the assertion's shape: the test asserts
`int(np.asarray(image)[:, :, :3].max()) > 0` with the message "the auto-wired distance field
rendered black". Under the mutation `u_df` is unbound, falls through to the seeded photo (F1),
and the photo is also non-black. The two outcomes are trivially separable by any statistic that
looks at the picture rather than at its maximum:

```
correct code (the distance field):  max_rgb=154,  155 unique colours,   mean [44.6 44.6 44.6]
mutated code (the shipped photo):   max_rgb=255, 41319 unique colours,  mean [126.7 114.1 94.9]
```

This is the checker-narrows-its-own-domain shape: the test's stated subject is "the auto-wired
distance field rendered", and it verifies only "something non-black rendered".

**Fix.** Assert against the producing pass rather than against brightness: compare the `edge`
canvas to `document.passes["df"].canvas` texture contents, or assert the render is greyscale
(`r == g == b` everywhere), which the distance field satisfies and the photo does not.

## F4. `preview_cell`'s `sublines` parameter now has zero callers

**Claim.** The wave removed the strip's `sublines` argument, which was the parameter's only
caller in the app. The parameter, its docstring paragraph, and its layout arithmetic remain.

**Evidence.** Every non-definition occurrence in the tree:

```
tests/test_pass_verbs.py:498  def test_the_strip_draws_no_sublines(...)
tests/test_pass_verbs.py:510  captured.append(kw.get("sublines", ()))
tests/test_ui_prose_budget.py:47  "sublines": 4,
```

No production call site passes it. Under the speculative-machinery rule the test is "is REMOVING
it churn?" — it is a parameter on a shared UI primitive that must be taught and maintained, with
no named near-future consumer (the wiring display is 070's graph view, which is not a
`preview_cell`). The prose-budget roster entry keeps it alive as measurable surface too.

**Fix.** Delete the `sublines` parameter, its docstring paragraph, its term in `footer_h`, and
its render loop from `ui_primitives.py::preview_cell`; drop its entry from
`test_ui_prose_budget.py`'s widths map, and rewrite `test_the_strip_draws_no_sublines` to assert
the call passes no such kwarg.

## F5. `_sampler_names` takes `self` and never uses it; `_is_user_bound` is a private name imported across modules

Two conventions items, both small, both in `## Code rules`.

**Claim and evidence.** `Document._sampler_names(self, render_pass)` contains zero `self.`
references (verified by grep over its body). The rule: "A method that doesn't use `self` isn't a
method — make it a module-level free function." Separately, `popups/pass_settings.py` line 24
does `from shaderbox.document import _is_user_bound` — a leading-underscore name reached from
another module. The wave doc argues the import direction is legal, which it is; the naming is
what is off, since a name two modules share is not private.

**Fix.** Move `_sampler_names` to a module-level `def _sampler_names(render_pass: Pass) ->
list[str]` in `document.py` and call it unqualified. Rename `_is_user_bound` to `is_user_bound`
now that it has a second module reading it.

---

## Areas that came out clean, and what was run to establish it

**The resolution rule.** The full matrix executed independently of the test file — (absent,
`""`, explicit) x (pass `<x>` exists, does not) x (media-bound, unbound), twelve cells — matches
the spec in every cell. An explicit name always wins, including over the bound exclusion and
including when its pass is gone; `""` always yields nothing; the name rule fires only on an
absent key for an unbound sampler naming an existing pass. Beyond the matrix: `u_prev` resolves
to the consumer even when a sibling pass is literally called `prev`; a bound `u_prev` yields
nothing; a name with no `u_` prefix yields nothing; a bare `u_` yields nothing; `u_edge` on pass
`edge` resolves to itself (name-driven feedback, consistent with the `u_prev` rule); and a
stored edge on an UNCOMPILED pass (`samplers=[]`) carries through unconditionally, including a
stale one, which is the implementer's stated contract.

**The effective graph never compiles.** Reverting `_sampler_names` to `get_active_uniforms()`
turns `test_an_uncompiled_pass_contributes_no_auto_edge_and_compiles_nothing` and
`test_a_u_prev_pass_has_feedback_without_a_stored_edge` red — the 066 D1 inversion is genuinely
pinned. Cost per frame measured on the six-pass Radiance Cascades example: `effective_graph()`
is called exactly once per consumer (`render` once, the strip's `draw` once, `has_feedback`
once, `_pass_views` once) and performs N program scans per call, one per pass, never per-pass
inside a loop. Timed at 0.021 ms for six passes, so roughly 0.084 ms across a live frame's four
calls, 0.5% of a 60 fps budget.

**The media exclusion at the seam.** Constructed the case the shipped examples do not contain: a
pass literally named `image` added to the Media Input example beside its bound `u_image`. The
effective graph for `main` stays `{}` and the render keeps the user's PNG (132493 unique
colours), rather than the new pass's black. The exclusion fires on `is_default_image`, not on
`isinstance` alone, as claimed.

**The black texture's GL lifetime.** `Document._black` is `None` at construction, allocated once
lazily in `_black_texture()`, and released in `Document.release()` where it is also set back to
`None`. Nothing allocates per frame and nothing allocates per bind.

**The planner across frames.** A six-pass name-wired Radiance Cascades under the sweep, traced
per frame with `Pass.render` instrumented and `assert_plan_invariants` asserted on every frame's
plan against the graph that plan was built from:

```
f0: composite, cascade
f1: df, paint, cascade, composite, jfa
f2: seed, jfa, df, paint, cascade, composite
f3: paint, seed, jfa, df, cascade, composite   <- converged, stable f3..f11
```

No pass draws twice in a frame (the repeats in the raw trace are `jfa` x9 and `cascade` x6,
which match those passes' `iterations` in `graph.json` — 068 D1's draw-N-times-inside-one-turn,
not a double draw). No pass is skipped once online. The one out-of-order turn is at f2, where
`seed` draws before `paint`: `seed` compiled during f2's own draw, so its `u_paint` edge existed
only in the post-render effective graph and the plan that ordered f2 never had it. That is the
one frame from black the spec's worked example predicts, and it self-corrects at f3.

**The gear.** Read `_draw_inputs`: positions and stores agree. Index 0 is `auto: <x>` and is
written by `unwire_pass_input` (deletes the key); index 1 is `(none)` and is written by
`wire_pass_input(..., "")` (stores `""`); indices 2+ are the sorted pass names and store the
name. The `choices.index(...)` on the mixed list is safe because `_PASS_NAME_RE` admits no
colon, space or parenthesis, so neither synthetic label collides with a pass name. The stale
case (an explicit name whose pass is gone) displays at index 1, `(none)` — verified — which
renders the same black the stale wire renders, so the display is honest; re-picking index 1
returns `changed=False` and leaves the stale key in `graph.json`, which is cosmetic since both
states render identically.

**Transactional rename and remove pass `""` through.** Executed against a graph carrying
`{"u_none": "", "u_src": "src", "u_x": "gone"}`:

```
rename src->source: {'u_none': '', 'u_src': 'source', 'u_x': 'gone'}
delete src        : {'u_none': '', 'u_x': 'gone'}
```

`_graph_renamed` maps `""` to itself (no valid pass name equals `""`); `_graph_without` drops
only keys whose source matched the removed pass, so `""` survives. Deleting `src` returns
`u_src` to UNDECIDED rather than to an explicit none, so the name rule re-wires if a pass called
`src` reappears — consistent with the design's "absent means the name decides".

**The strip.** Both planner calls in `pass_list.draw` read one `effective_graph()` hoisted above
the loop (`live` and `_strip_order` share `resolved`). The two new tests pin the stale wash and
the topological order, and the order test's names (`zeta`, `alpha`, `mid`) disagree with
alphabetical in every position, so a sorted-name fallback cannot pass by accident. Card spacing
follows automatically: `footer_h` is `line_h * (1 + len(sublines))` and `len(sublines)` is now 0.

**The renames.** All eighteen tokens landed and no role name survives in either multi-pass
example (`u_src`, `u_lit`, `u_glow`, `u_light` all absent from both). The three prefix traps are
intact: `u_light_radius` inside the Radiance Cascades example's own `paint.frag.glsl`, and the
`u_light_*` / `u_glow_*` families in the unrelated single-pass examples, values and all. Both
`graph.json`s carry the new keys; neither `document.json` was touched (the commit's stat
confirms). The new gate `test_every_example_input_uniform_names_its_source` derives its domain
from the shipped graphs by glob, so a new example defaults into it.

**`has_feedback`.** Reads `plan_passes(self.effective_graph())[0].feedback`. Pinned by
`test_a_u_prev_pass_has_feedback_without_a_stored_edge`, which asserts both halves — false while
nothing has compiled, true after the sweep, with `graph.json` still storing nothing. Removing
the `u_prev` branch from `_auto_source` turns it red.

**Falsifiers re-run.** Four mutations, each restored and each restore verified with `git diff
--quiet` before anything else ran:

| Mutation | Result |
| --- | --- |
| `with_input("")` pops the key again | RED (3 tests) — as claimed |
| `_sampler_names` uses `get_active_uniforms()` | RED (2 tests) — as claimed |
| drop the `bound` exclusion in `effective_inputs` | RED (1 test) — as claimed |
| drop the `u_prev` branch in `_auto_source` | RED (2 tests) — as claimed |
| `Document.render` plans `self.graph` | **GREEN** — F3 |
| `pass_list.draw` plans `document.graph` | RED (1 test) — as claimed |

**Conventions.** No `# noqa`, no `# type: ignore`, no `# pyright: ignore`, no inline import, no
`Any`, no `@staticmethod`, no `if TYPE_CHECKING` anywhere in the added source. Comments state
what is non-obvious about the code as it is now and point at their canonical home (069 D9, 066
D1, 065 D3) rather than narrating the change; none is a development-history block. The prose
budget passes, and the `_draw_pass_tile` roster entry was updated to match its shortened card.

---

## False trails

- The `jfa` x9 and `cascade` x6 repeats in the render trace looked like a memoization break; they
  are those passes' `iterations` values in `graph.json`, drawn inside one turn per 068 D1.
- `_pass_views` calling `get_active_uniforms()` (which compiles) looked like a fresh 066 D1
  inversion; it predates this commit and is the copilot path's existing posture — the defect it
  causes here is ordering, not compiling, which is F2.
- The gear showing `(none)` for a stale explicit name looked like a wrong-state display; both
  states render black, so the label is honest and re-picking it is a no-op that costs nothing.
- The Media Input example looked exposed to auto-wiring of its bound `u_image`; it is protected
  by the absence of a pass called `image`, and the exclusion itself was verified separately by
  constructing that pass.
- `projects/dev` looked like it needed a hand edit for the new `""` convention; both sandbox
  documents are single-pass with `"inputs": {}`, so nothing there changes.
- `effective_graph()` being called four times a frame looked like a per-frame cost worth
  flagging; measured at 0.5% of a 60 fps budget on the largest shipped document.

## Coverage

Read every file in `f18a7d3` end-to-end, plus the unchanged `pass_graph.py` planner,
`core.Pass.render`'s binder, `media.is_default_image`, `project_session`'s `_graph_renamed` /
`_graph_without` / `delete_pass` / `rename_pass`, `ui_primitives.preview_cell`, `ui.py`'s
`has_feedback` call site, `01_spec.md` § D9 and § W-D, and `70_wave_d_wiring_naming.md`.
Executed: the twelve-cell resolution matrix plus seven extra cases; a four-state render-seam
trace on a real document; the media-exclusion collision case; a six-pass twelve-frame planner
trace with per-frame invariant assertions and a per-turn ordering check; the copilot two-call
read; the rename/remove `""` round trip; a cost measurement; six mutation falsifiers with
verified restores; `make gates` unpiped (EXIT=0, smoke passed) and the full suite (1629 passed).
Not executed: the interactive gear (its combo was exercised through the wave's own imgui-frame
rig rather than by clicking), and the tutorial/W-H surface, which is a different wave.
