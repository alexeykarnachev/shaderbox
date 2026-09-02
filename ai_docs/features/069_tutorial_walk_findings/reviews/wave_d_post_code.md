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

---

# Round 2 (closure) — against `3d635ef`

Narrow closure round on `3d635ef` ("069 W-D fixes: an unfilled input reads black"), 12 files,
read via `git show 3d635ef:<path>`. Nothing tracked was edited beyond this file; every probe and
every mutation ran in the scratchpad and was restored with the restore verified.

## Overall: **FAIL**

All five round-1 findings are CLOSED. The fix for F1 introduced a new defect of the same class
and of higher severity than the one it closed: the shipped Media Input example now renders fully
black. It is a one-line fix, and the suite cannot see it.

| Finding | Verdict |
| --- | --- |
| F1 unresolved sampler reads the photo | **CLOSED** (regression R1 below) |
| F2 `_pass_views` resolves before the compile | **CLOSED** |
| F3 raw-graph render falsifier stays green | **CLOSED** |
| F4 `sublines` has zero callers | **CLOSED** |
| F5 `_sampler_names` takes an unused `self`; `_is_user_bound` shared while private | **CLOSED** |
| R1 a user-bound texture is overwritten with black | **NEW, blocking** |

---

## R1 (new, blocking). The black seed overwrites a USER-BOUND texture, and the shipped Media Input example renders black

**Claim.** `Document.render` now seeds `inputs` with the black texture for every sampler the
pass DECLARES:

```python
inputs: dict[str, moderngl.Texture] = {
    uniform: self._black_texture()
    for uniform in sampler_names(render_pass)
}
```

`Pass.render` resolves each sampler as `inputs.get(name, uniform_values.get(name))`, so a seeded
entry SHADOWS `uniform_values` — including a texture the user bound. The effective graph
deliberately produces no edge for a user-bound sampler (that is the 069 D9 media exclusion), so
nothing overwrites the seed and the user's image is replaced by black.

**Evidence.** The shipped Media Input example (`73ea2431-13f6-41e4-b923-04d846b678b0`),
UNMODIFIED, copied to a temp dir and rendered until every pass is online:

```
u_image: Image, is_default_image=False, user_bound=True
u_video: Video, is_default_image=False, user_bound=True
effective inputs for `main`: {}
RENDER: max_rgb=0   unique_colours=1
```

Bisected across the two commits with only `shaderbox/document.py` swapped, same probe:

```
f18a7d3:  max_rgb=255  unique=131266     <- the user's PNG and video
3d635ef:  max_rgb=0    unique=1          <- fully black
```

This is strictly worse than F1. F1 showed the wrong picture for a sampler the user had not
wired; R1 discards a picture the user explicitly chose, on a shipped example whose entire subject
is media input. The wave's own decision 3 exists to prevent exactly this ("a sampler whose
`uniform_values` entry is a user-bound texture is never auto-wired ... would otherwise let a pass
named `image` silently replace the PNG in the `Media Input` example"); the exclusion was applied
in `effective_inputs` and then bypassed at the seam that came after it.

**Why the suite is green.** No test renders the Media Input example and inspects its pixels.
`test_raw_texture_round_trip.py` and `test_video_frame_stepping.py` reach into that directory for
a media FILE, not for a document render. The round-1 media-exclusion probe that would have caught
it asserted on `effective_graph`, which is still correct — the defect is downstream of it.

R1 is independent of the frame-0 compile guard the concurrent spec review added to
`Document.render` after this round began: re-run against the working tree carrying that guard,
the Media Input example still renders `max_rgb=0, unique=1`. The two defects sit on the same
seed and neither fix substitutes for the other.

**Fix.** Exclude user-bound samplers from the seed, the same predicate the graph already uses:

```python
inputs: dict[str, moderngl.Texture] = {
    uniform: self._black_texture()
    for uniform in sampler_names(render_pass)
    if not is_user_bound(render_pass.uniform_values.get(uniform))
}
```

Verified: with that one line, Media Input renders `max_rgb=255, unique=131266` again, the
`u_nosuchpass` probe still renders `max_rgb=0`, and `test_default_wiring.py` +
`test_pass_verbs.py` + `test_copilot_passes.py` are 57 passed. The accompanying test is the one
the suite lacks: render the shipped Media Input example and assert its output is not uniformly
black.

---

## F1 — CLOSED

`Document.render` seeds every declared sampler black and lets resolved edges overwrite, so the
three states reach the same texture by one rule. The four-state seam probe, re-run verbatim:

```
                              round 1        round 2
absent, name matches nothing  max_rgb=255 -> max_rgb=0
absent, no u_ prefix (`tex`)  max_rgb=255 -> max_rgb=0
explicit ""                   max_rgb=0   -> max_rgb=0
stale explicit name           max_rgb=0   -> max_rgb=0
```

The three surfaces that disagreed now agree on the same sampler in the same frame:

```
GEAR:     auto: none   (index 0)
COPILOT:  u_nosuchpass <- (nothing; reads BLACK)
RENDER:   max_rgb=0, 1 unique colour
```

Pinned: reverting the seed to the stored-`""`-only form turns
`test_an_unresolved_sampler_renders_black` red.

## F2 — CLOSED

`_pass_views` gathers `_sampler_uniform_names` for every pass (the call that compiles) before
resolving the graph once. The round-1 two-read probe, first read on a name-wired never-compiled
Bloom, all programs `None` going in:

```
call 1:  blur ['u_bright <- bright']  bright ['u_scene <- scene']
         composite ['u_scene <- scene', 'u_blur <- blur', 'u_trail <- trail']
         trail ['u_scene <- scene', 'u_prev <- trail']
```

Correct on the FIRST read, where round 1 reported every row as `(nothing; reads BLACK)`. Pinned:
moving the sampler gather back below the resolve turns
`test_pass_views_resolves_after_the_compile_that_finds_the_samplers` red. The implementer's note
that `read_working_set` already compiles before reaching `_pass_views` is correct, so the test
asserting on `_pass_views` directly is the right level.

## F3 — CLOSED

`test_u_df_beside_df_renders_without_the_gear` now compares `edge`'s canvas against `df`'s texel
for texel (`abs diff max <= 1`) and keeps a non-black check as a separate assertion. The round-1
falsifier that stayed green:

```
Document.render plans self.graph  ->  1 failed, 40 passed
FAILED tests/test_default_wiring.py::test_u_df_beside_df_renders_without_the_gear
```

Red now, where round 1 was 40 passed. The failure output shows the texel comparison rejecting the
grey field against the photo, which is the mechanism the finding asked for.

## F4 — CLOSED

`sublines` is gone from `preview_cell`'s signature, docstring, `footer_h` term and render loop,
and from `test_ui_prose_budget.py`'s widths map. The only occurrence left in the tree is the
guard assertion in `test_the_strip_draws_a_picture_and_a_name_only`:

```
tests/test_pass_verbs.py:517:  assert "sublines" not in kwargs, kwargs
```

The rewritten test also asserts each tile's `footer` is a real pass name, so it checks what a
tile DOES carry rather than only what it does not.

## F5 — CLOSED

`Document._sampler_names` is now the free function `document.sampler_names(render_pass)` with no
`self`, and `_is_user_bound` is `is_user_bound`, both with docstrings. `pass_settings.py` imports
the public name. `pass_settings.py` keeps its own local `_sampler_names` that goes through
`get_active_uniforms()`; that is a different function on purpose (opening the gear is a user act
that should bring the pass online, where the render path must not compile), it predates this
commit, and it is single-module private, so it is correctly named.

---

## False trails, round 2

- The two `sampler_names` functions in `document.py` and `pass_settings.py` looked like a
  duplicate left behind by the rename; they differ in the one respect that matters (compiles
  versus does not) and both are correct where they sit.
- The black seed running per iteration rather than per pass looked like wasted work; it is a dict
  comprehension over a handful of names inside a loop that already issues a draw call, and the
  round-1 measurement put the whole resolution path at 0.5% of a 60 fps budget.
- `is_default_image` still returning True for an unresolved sampler's stored value looked like a
  leftover; the fix binds over the seeded value rather than changing what is seeded, which is what
  keeps the media exclusion expressible at all.

## Coverage, round 2

Read every source and test file in `3d635ef` end-to-end. Re-ran the round-1 seam probe, the
three-surface agreement probe, the `_pass_views` two-read probe, the media-exclusion probe, and
the twelve-frame planner trace with per-frame invariant assertions (unchanged: converges at f3,
no double draw, no skipped draw). Ran four mutations with verified restores: the raw-graph render
plan (now red), the stored-only black seed (red), the `_pass_views` resolve order (red), and the
candidate R1 fix (green on all three suites plus both probes). Bisected R1 across `f18a7d3` and
`3d635ef` with only `document.py` swapped. `make gates` unpiped: EXIT=0, check passed, test
passed, smoke passed. Not executed: the interactive gear and the manual in-app steps, which need
a display this shell does not have.

---

# Round 3 — against `3d36794`

Narrow closure round on `3d36794` ("069 W-D: black holds from frame 0, media survives"), 4 files,
read via `git show 3d36794:<path>`. W-H's fix-up is in flight in the working tree; nothing tracked
was edited beyond this file, no `git stash` was used, and every mutation was restored with the
restore verified.

## Overall: **PASS**

| Item | Verdict |
| --- | --- |
| R1 user-bound texture overwritten with black | **CLOSED** |
| Residue: one frame of photo before the seed knows the samplers | **CLOSED** |
| 066 D1 (no eager compile of an off-chain pass) | **INTACT** |
| Round-1 F1..F5 | still closed |

---

## R1 — CLOSED

The seed now carries the same predicate the graph excludes with:

```python
inputs: dict[str, moderngl.Texture] = {
    uniform: self._black_texture()
    for uniform in sampler_names(render_pass)
    if not is_user_bound(render_pass.uniform_values.get(uniform))
}
```

The Media Input bisect, re-run across all three commits with only `shaderbox/document.py`
swapped and the same probe, on the UNMODIFIED shipped example:

```
f18a7d3:  max_rgb=255  unique=131266     <- the user's PNG and video
3d635ef:  max_rgb=0    unique=1          <- the regression
3d36794:  max_rgb=255  unique=131266     <- restored, texel-identical to pre-regression
```

The restored numbers match `f18a7d3` exactly, so the fix returns the example to its prior pixels
rather than to some other non-black state. Pinned: dropping the `is_user_bound` clause turns
`test_a_user_bound_texture_survives_the_black_seed` red. That test renders the shipped example
and asserts both a non-black maximum and more than 1000 distinct colours, so a flat fill of any
brightness fails it — the failure mode the round-2 finding described.

The narrower probe from round 1 also still holds: a pass literally named `image` added beside a
bound `u_image` produces no auto edge and renders `max_rgb=255, unique=132493`, so the exclusion
is intact at both the resolution seam and the bind seam.

## The one-frame residue — CLOSED

A program-less pass now compiles immediately before its input seed is built, so
`sampler_names(render_pass)` is non-empty on the pass's very first frame:

```python
if render_pass.program is None:
    render_pass.compile()
```

The `u_nosuchpass` probe, per frame rather than at the settled state:

```
frame  0: max_rgb=0 unique=1
frame  1: max_rgb=0 unique=1
...
frame 10: max_rgb=0 unique=1
WORST across all frames: max_rgb=0
```

Black from frame 0, where the seed previously had nothing to name on the first frame and let the
photo through. Pinned: removing the pre-seed compile turns
`test_an_unresolved_sampler_renders_black` red, and that test is now frame-indexed with the frame
number in its message, so it fails ON the offending frame rather than reporting a settled state.

## 066 D1 — INTACT

The pre-seed compile fires inside the `for name in order` loop, so it can only reach a pass the
plan already selected for this frame. Verified with a tripwire rather than by reading: Bloom with
`scene` as the output, so `bright` / `blur` / `trail` / `composite` are off the chain, every
off-chain pass given a `compile` that raises, nothing compiled going in:

```
output=scene  on-chain: ['scene']  off-chain: ['blur', 'bright', 'composite', 'trail']
render() completed with NO off-chain compile  -> 066 D1 intact
off-chain passes whose compile was called: none
on-chain programs after frame 0:  {'scene': True}
off-chain programs after frame 0: {'blur': False, 'bright': False, 'composite': False, 'trail': False}
```

The four off-chain stubs stayed silent and their programs are still `None` after the frame.

The guard also adds no compile that was not already happening. Counting `Pass.compile` calls per
frame over the six-pass Radiance Cascades example, before and after:

```
f18a7d3:  frame 0 -> 6 calls, frames 1..7 -> 0 each
3d36794:  frame 0 -> 6 calls, frames 1..7 -> 0 each
```

Identical, and six calls for six passes means each compiles once. The guard moves the compile a
few lines earlier within the same pass's turn; `Pass.render` would have issued it moments later,
which is what the code comment claims and what the count confirms.

## Regression sweep

Everything that was PASS in round 1 or CLOSED in round 2 was re-run against this tree:

- the four-state seam probe: `max_rgb=0` on all four (stored `""`, name matches nothing, no `u_`
  prefix, stale explicit name);
- the media exclusion with a colliding pass name: `max_rgb=255, unique=132493`;
- the twelve-frame planner trace with per-frame `assert_plan_invariants`: converges at f3, no
  double draw, no skipped draw, no out-of-order turn after the transition;
- `sublines` has zero occurrences in `shaderbox/`; `sampler_names` and `is_user_bound` are both
  free public functions;
- `test_pass_views_resolves_after_the_compile_that_finds_the_samplers` plus the two wiring suites:
  43 passed.

`make gates` unpiped: **EXIT=0**, check passed, test passed, smoke **passed**.

## False trails, round 3

- Six compiles on frame 0 looked like the eager-compile 066 D1 forbids; the count is identical
  before and after this commit, and those are `Pass.render`'s own compiles of passes already in
  `order` — the D1 rule is about compiling a pass nothing is drawing, which the tripwire shows
  does not happen.
- The pre-seed compile looked like it might double-compile a pass whose compile FAILS, since a
  failed attempt sticks; it is guarded on `program is None` and `Pass.render` re-checks the same
  condition, so a broken pass is attempted once per frame exactly as before.
- My round-1 `copilot_gap2.py` probe still prints the black rows; it hardcodes the old
  resolve-before-gather order to reproduce the original defect, so its output describes the
  script, not the shipped code — the shipped path is covered by the test above.

## Coverage, round 3

Read all four files in `3d36794` end-to-end. Ran: the Media Input bisect across three commits
with only `document.py` swapped; the `u_nosuchpass` probe frame-indexed from frame 0; an
off-chain compile tripwire on four passes; a per-frame compile count compared against the
pre-fix commit; the round-1 seam, media-exclusion and planner probes; two mutations with verified
restores (the `is_user_bound` clause and the pre-seed compile, each turning its own test red);
and `make gates` unpiped. Not executed: the interactive gear and the manual in-app steps, which
need a display this shell does not have.
