# 091 pre-implementation review — correctness & design

**Verdict: PARTIAL.** Implementable after six spec edits. Nothing in D1–D11 is structurally
unbuildable; the defects are (a) a named anchor that does not exist in the repo, (b) two places
where the spec asserts "nothing else moves" and the code says otherwise, (c) three copy rules
with a hole the plan's own wording opens. Ranked by cost to discover during implementation:

1. **E1 — "Bloom Chain" is not a shipped example.** The six examples are Fire, UV Mango, Media
   Input, Radiance Cascades, Night City, Text Rendering; only RC is multi-pass. D2, D3 and
   verification items 3 and 4 are all written against a document that does not exist.
2. **E2 — D7's "`tiles_per_row` is untouched" is false at some panel widths,** and the mock it
   cites changes the in-group gap from 8px to 4px.
3. **E3 — D7's "no border of its own" cannot be done with `border_color`;** `preview_cell` draws
   its border as imgui's own `ChildFlags_.borders`.
4. **E4 — D5's copy list omits `NoSource` / `AutoSource`,** and `Image(value.texture)` silently
   drops the media's `file_details`.
5. **E5 — D11 needs a fourth render branch,** not a third; `ui.py` already has three.
6. **E6 — the out-of-scope "graph view (092)" contradicts a settled decision** (070: "no graph
   view ... the `imgui_node_editor` question is closed with it").

Verified green baseline before reviewing: `make gates` exit 0 (check + test + smoke all passed,
exit code captured unpiped).

---

## Per-decision table

| # | Verdict | Evidence |
|---|---------|----------|
| **D1** `PassEntry.group: str = ""` | **CONSISTENT** | `PassEntry` is a frozen pydantic model with every field defaulted (`shaderbox/pass_graph.py:100-120`), and `load_graph`'s per-entry salvage enumerates `_keyed_entry_fields()` from `PassGraph.model_fields` rather than a hand list (`shaderbox/document.py:109-163`), so the new field inherits the salvage with no edit. Probed: a `graph.json` carrying `"group": "bloom"` **today** logs `Ignoring unknown graph.json.passes.a key: group` and loads the rest — so the field is additive in both directions. The lockstep-dicts bullet (`conventions.md:997`) is satisfied: one field on the one entity, no second name-keyed dict. `_graph_renamed` / `_graph_without` carry the whole `entry` object (`project_session.py:137-149`), so rename keeps and delete drops the group with no edit, exactly as "Files touched" claims. |
| **D2** group name is the prefix, `group_slug` | **INCONSISTENT (E1)** on the worked example; the rule itself is CONSISTENT | The function is right: probed `group_slug` against every shipped `ui_name` — `Radiance Cascades`→`radiance`, `UV Mango`→`uv`, `Media Input`→`media`, `2D SDF`→`g_2d`; all match `_PASS_NAME_RE` (`project_session.py:125`). But `Bloom Chain` is not a document in this repo: `shaderbox/resources/document_examples/` holds six dirs whose `ui_name`s are Fire, UV Mango, Media Input, **Radiance Cascades**, Night City, Text Rendering, and five of the six are single-pass (`graph.json` `passes` keys). The spec's three pinned inputs therefore pin one real name, one invented name and one name from no document at all. Two further gaps the implementer has to answer: an empty `ui_name` falls back to the **dir name** (`ui_models.py:_load_ui_state`, `filtered_ui_state.setdefault("ui_name", dir_name)`), so a document whose name was never set slugs its UUID — probed `77a84d27-8011-ee1cb1a9587c` → `g_77a84d27_8011_ee1cb1a9587c`, valid but 28 chars, which is past the 21-char budget D2 itself derives; and `group_slug("")` → `g_`, which passes `_PASS_NAME_RE` and is therefore silently accepted as a group name. |
| **D3** entry points = roots of the compiled source wiring | **CONSISTENT** (the compile-safety claim is **demonstrated**) | Probed the exact prescription on the real RC document loaded from resources: before compiling, `effective_wiring()` answers `{cascade:{}, composite:{}, df:{}, jfa:{}, paint:{}, seed:{}}` — RC's `document.json` carries **no explicit sampler rows at all**, so its entire wiring is name-rule (069 D9) and is invisible until the programs exist. That makes D3's compile requirement load-bearing, not belt-and-braces: without it every source would read as all-entry-points. After `p.compile()` on each pass with no program and no errors, the wiring is the full six-pass DAG and `entry_points` answers exactly `['paint']` — one root, as D3 claims. **The state question is answered: nothing the source document owns is disturbed.** Measured before/after the compile loop: `document.first_render_done` False→False, `Pass.first_render_done` all False→all False, `drawn_frame` all −1→all −1, `feedback_passes()` []→[]. That holds by construction, not by luck: the three flags are written only in `Document.render` (`document.py:775,800-801`) and `_feedback_canvas`, and `Pass.compile` (`core.py:332-415`) touches only `compile_unit`, `program`, `vbo`, `vao`. D3's parenthetical about `_bring_chain_online` is also right — it walks only the target's chain (`document.py:718-738`), so on a source whose bundle has a branch the output does not read, it would leave that branch's wiring unknown. Also checked the three shapes D3's verification names: `entry_points({'main':{}})`→`['main']`, `entry_points({'main':{'u_prev':'main'}})`→`['main']` (self-read still a root), a two-input compositor→`['bg','fg']`. |
| **D4** substituted entry point is not copied; readers re-pointed | **UNDERSPECIFIED** | The `ImportPlan` shape and the rejection set are implementable and the `plan_import` signature is GL-free as claimed (`Wiring` is `Mapping[str, Mapping[str, str]]`, `pass_graph.py:197`). The unanswered question is **how a copied pass's samplers that the wiring did NOT fill reach `import_passes`**. D4's last sentence says they "keep their value", but `ImportPlan.sources` is typed `dict[str, dict[str, str]]` — pass names only — so it carries no channel for a `NoSource`, an `AutoSource` or a bound texture, and `import_passes` (D5) is told to "overwrite the wired samplers with the plan's `PassSource` rows". **The answer that follows from the code:** the unfilled values ride the *uniform-value copy* of D5, not the plan — they already live in `sp.uniform_values` and a `dict` copy carries them; the plan's job is only the wired subset. D4 should say that explicitly, because the two sentences as written read as two different mechanisms. Second gap: D4 says an `AutoSource` "may catch a host pass by name, which is the rule working as designed" — but the prefix makes that *more* likely to fire wrongly, not less: a copied `bloom_seed` declaring `uniform sampler2D u_paint` with no row will bind the **host's** `paint` if the host has one, since `_auto_source` strips `u_` and looks the bare name up in the host's pass set (`pass_graph.py:218-228`). The spec should state whether that is wanted or whether every undecided sampler of a copied pass is materialized to `NoSource()` to close it. |
| **D5** `import_passes` executes and saves | **UNDERSPECIFIED / INCONSISTENT in three places (E4)** | **(a) The value-type roster has a hole.** Probed every shipped example's live `uniform_values` after compile + seed: the types present are `float`(61), `tuple`(34), `AutoSource`(9), `int`(1), `Image`(1), `Video`(1). D5 names scalars, tuples, `Buffer`, `Image`, `Video`, raw `Texture` — it does **not** name `NoSource` or `AutoSource`, which are 9 of the 107 values in the shipped set and are exactly the ones D4 says must survive. They are immutable frozen dataclasses (`pass_graph.py:205-220`), so the copy rule is "by reference, they are values"; the spec has to say so, or an implementer reading "copy uniform values — scalars and tuples by value, a Buffer via…" reasonably concludes a source value is not a uniform value and drops it. The full domain to cover is `UniformValue` in `core.py:165-173` plus the three `SamplerSource` members. **(b) `Image(value.texture)` loses the media's identity.** Probed: `Image(img.texture)` round-trips the pixels (960×1280 both sides, distinct objects) but its `details.file_details` comes back `path='' size=0`, because `Image.__init__` only builds `FileDetails.from_file_path` for a `PathLike` src (`media.py:113-140`). The media panel reads those fields. The copy that keeps them is `Image(source_dir / row["file_path"])` — which is also why `source_dir` is in the signature. **(c) `source_dir` is redundant as a parameter:** `document.document_dir_of(source)` already derives it from a pass file's path, and its docstring says it exists so exactly one place knows the depth (`document.py:1092-1099`). Passing it separately is a second source of truth for the same fact. **(d) The save-interaction question is answered and the answer is clean — no edit needed.** Probed the whole D5 sequence end to end on a real host (starter) + real source (Media Input, the one example with a bound `Image` and `Video`): copy the pass file, build the `Pass`, copy the uniform values, merge the 3 source `ui_uniforms` rows the host lacked, `UIDocument.save`. Result: all 4 rows survive the prune (`ui_models.py:462-480` prunes against `get_uniform_hash` over every pass's *live* uniforms, and the copied pass compiles inside `save`, so its rows are live), and the asset sweep wrote and kept `media/mi_main/u_image.png` and `media/mi_main/u_video.mp4` — the sweep is per-pass-dir and keyed on the freshly-built metadata (`ui_models.py:500-525`), so a pass dir that did not exist before the import is created and referenced in the same pass. **One caveat D5 should record:** `get_uniform_hash` is name+shape only, **not** pass-qualified (`util.py:78-85`), so a merged source row for `u_scale` silently merges with the host's own `u_scale` row of the same shape — the host's wins under `setdefault` and the source's input-type/range is lost. That is the correct precedence, but it is a real limit on "a sampler's `texture` input type and a drag's range come along", and the spec claims it unconditionally. |
| **D6** the bundle's output takes over the fed pass's role | **UNDERSPECIFIED (the release path)** | The rule and the per-reader picker follow the maintainer's verbatim words (f) and are expressible. The answer to the review's release question is **yes, it matters, and the spec does not say it.** `set_sampler_source` calls `try_to_release(values.get(uniform))` before writing (`project_session.py:1008-1010`), and `try_to_release` calls `.release()` on anything that has one (`util.py:100-104`) — so a host sampler currently holding a bound `Image`/`Video`/`Texture` must be released when the handover row overwrites it, or it leaks for the life of the session (`Pass.release` only frees what `uniform_values` still holds, `core.py:286-297`). D6 says `import_passes` "writes the handover rows as explicit `PassSource`s on the HOST passes" with no mention of the release. Two routes, and the spec should name one: call `session.set_sampler_source` per handover (gets the release free, but it **saves on every call** — N+1 saves for N handovers, where the six verbs each save exactly once), or write `uniform_values` directly inside `import_passes` and call `try_to_release` on the old value first, saving once at the end. The second is what "executes the plan and saves, like the six pass verbs" implies. Second gap: a handover pair can only name a host sampler the *host wiring* filled, and the host wiring for a host pass that has not compiled is its explicit `PassSource` rows only (`document.py:689-717`) — so the checkbox list is incomplete for an uncompiled host pass whose wire is name-rule. D3 compiles the **source**; nothing in the spec compiles the host. The answer that follows from the code: compile the host's program-less passes too when the dialog opens, the same loop D3 prescribes. |
| **D7** one flush outline per run | **INCONSISTENT on two claims (E2, E3)** | **(a) "`tiles_per_row` is untouched, since gaps do not change" is false twice over.** The mock it cites sets the in-group gap to **4px** against the strip's 8px (`00_mock.html:121` `.flush .grp .strip { gap:4px }` vs `:11` `--gap:8px`), and its own note says so: "the members … sit 4px apart so the outline reads as one thing". `SPACE.SM` is 4 and `SPACE.MD` is 8 (`theme.py:308-310`), and the strip passes `SPACE.MD` to both `tiles_per_row` and `same_line` (`pass_list.py:170-180`) — if the group gap narrows to match the mock, the per-row arithmetic is no longer a single gap value and `tiles_per_row` **is** touched. **(b) The 2px-outside outline clips at some widths even with the gap unchanged.** `tiles_per_row` charges no trailing gap, so a full row's span can come within a pixel of the content edge: computed for the real tokens, `avail=700` → `n=4`, `span=696`, **slack 4.0px** — a rect drawn 2px outside the run plus its own stroke lands on or past the edge, and the label rides 8px *above* the first tile's top, outside the strip's own vertical extent entirely. **(c) `bordered: bool = True` does not remove the border as written.** Read `preview_cell` end to end (`ui_primitives.py:1213-1353`): the tile's border is `child_flags=imgui.ChildFlags_.borders` on the `begin_child` (line 1268), and `border_color` only pushes `Col_.border` over it (1259-1261). So `bordered=False` has to drop the flag from the `child_flags` expression, not pass a colour — and the spec's parenthetical "(the accent output border and the red error border still win)" then needs the flag kept whenever `border_color is not None`, which is a different predicate from `bordered`. **(d) The drawability question is answered: yes.** Each tile is its own child window (1265-1271), but the parent draw list is reachable from the strip's own scope — `imgui.get_window_draw_list()` is used that way in a dozen places including `ui_primitives.py:942-1141` — and the run's rect is computable from the tiles' screen positions without entering a child. The **order** is the part to get right: drawn after the tiles it paints over them, so the outline must either be drawn before the run's tiles from positions predicted by the same arithmetic `same_line` uses, or via a channel split (the pattern the imgui-ui skill documents as the alternative to a child). The spec says neither. |
| **D8** group tints are theme tokens, stable hash | **CONSISTENT** | The six named hues exist in `_P` and none is an accent primary: `_ACCENTS` primaries are `yellow_b`, `aqua_b`, `orange_b`, `blue_b` (`theme.py:88-100`), and the six proposed are `blue_b`, `purple_b`, `green_n`, `aqua_n`, `orange_n`, `blue_n`. **`blue_b` collides** — it is the `blue` accent's primary *and* `COLOR.TAG` (`theme.py:172`). So the tuple as written fails the assert the spec itself proposes, which is the assert working. Drop `blue_b` (five tints, or substitute `purple_n`/`green_b`). The invariant's shape matches the existing one (`theme.py:196-216`), the `zlib.crc32` choice is right (`hash()` is PYTHONHASHSEED-salted), and the conventions bullet this rides (`conventions.md:440-449`) says a new fixed role with accent-adjacent *outline* context gets added to the assertion — which is exactly what a group outline is. |
| **D9** group editable by hand + by the copilot | **UNDERSPECIFIED** | The modal row is straightforward (`pass_settings.py` already has the `name` row to copy, `_draw_name` at 141-160, and `_apply_entry` is the write seam at 126-138). The copilot half has three mechanical consequences the spec does not name, all verified: `CopilotCapabilities.set_pass` is a **positional-only** protocol signature (`capabilities.py:458-470`, note the `/`), so a `group` parameter must be appended there, in `backend.set_pass` (`backend.py:1385-1421`), in `tools/passes.py`'s `_SetPassArgs` + handler, **and** in the five existing test call sites that pass all nine arguments positionally (`tests/test_copilot_pass_tools.py:49,55,76,80,85`) — those break on an inserted parameter and merely lengthen on an appended one. **The two tests the review asked about are both safe:** `test_the_pass_verbs_are_mutating_with_delete_lazy_and_gated` (`tests/test_copilot_pass_tools.py:102-118`) asserts only `eager`/`mutating`/`gate_policy` and executes `add_pass` with one arg, so a new optional `set_pass` field does not touch it; the only schema lock is `tests/test_tool_registry.py:38-40` asserting `additionalProperties is False` on every args model, which a `Field(default=None)` on a `ToolArgs` subclass keeps. What the spec must decide: `_pass_table` gains a column on every row or a suffix on grouped rows only — the spec says "prints `group <name>` on grouped rows", which is the cheaper of the two and is what it should say in the table's own format string (`backend.py:1289-1292`). |
| **D10** one modal in the `PopupState` mutex | **CONSISTENT, with one hard requirement the spec half-states** | The enum-collision question is answered: **one place enumerates `PopupState` exhaustively**, `tests/test_project_management.py:636-667`, which ASTs `ui.py` for calls to names imported from `shaderbox.popups` and asserts `len(called) == len(PopupState) - 1`. Today that is 7 calls against 8 members (`ui.py:26-32` imports, 614-620 calls). So adding `IMPORT_PASSES` **fails that test** until `draw_import_passes(app)` is both imported and called in `ui.py`'s popup block — which the spec's "Files touched" does list, so this is a gate doing its job, not a defect. No other site enumerates the enum (grepped every `PopupState.` reference across `shaderbox/` and `tests/`: 10 in `app.py`, 4 in `hotkeys.py`, 3 in `ui.py`, one per popup module, and the test files above, all naming individual members). The conventions bullet (`conventions.md:453-466`) states the four-part checklist the spec satisfies. `ImportDraft` on `App` beside `PassDraft` matches "open/closed state lives on `App`". |
| **D11** the planned set treats `IMPORT_PASSES` as `EXAMPLES` | **INCONSISTENT (E5)** | The structure the spec describes does not exist. `ui.py`'s render chain is **already three branches** — `if not any_popup_open(): … elif EXAMPLES: … elif PASS_SETTINGS and renders_this_frame(current): …` (`ui.py:437-515`) — so D11 adds a **fourth**, not "a third render block" the spec says it avoids. And D11's rule needs both halves: a `IMPORT_PASSES`-with-Examples-tab branch that mirrors the EXAMPLES block, *and* a `IMPORT_PASSES`-with-project-tab branch that renders the ordinary set (which is what the `if not any_popup_open()` block does, and a popup is open). A copy-paste of the EXAMPLES block duplicates 20 lines: the `pending_example` one-per-frame election, the `first_render_done or is pending` admission, the `renders_this_frame` interval gate and the profiler span — the third textual copy of the same budget rule. The funnel bullet (`conventions.md:167-173`) names the second copy at a sibling site as the trigger to move to a funnel, and this is the third. The planned-set half (`_tick_frame_state`, `ui.py:267-285`) is a cleaner fit: `examples_open` becomes a predicate over two states and `planned`/`planned_documents`/`current_planned` follow, which is one edit. |

## "Out of scope" bullets

| Bullet | Verdict |
|---|---|
| cross-project presets folder | **CONSISTENT** with the maintainer's (b) verbatim ("no need for the cross project yet. just make sure that the code is generalizable enough"). The generalizability claim is real, not asserted: a `list[(label, dict[str, UIDocument])]` tab source is satisfied today by `session.ui_documents` and `session.ui_document_examples`, both already `dict[str, UIDocument]` (`project_session.py:508`, `load_documents_from_dir`). |
| export a group as a document | **CONSISTENT**, no code contradicts it. |
| folding a group | **CONSISTENT** with (c) verbatim ("too many corner cases… too many nuances"), and the rejection reason the spec records (convexity in the DAG) is the technically right one. |
| **the graph view (092)** | **INCONSISTENT (E6).** 070 closed this, not deferred it: "Decision: **no graph view.** The strip is the one view of the passes… The `imgui_node_editor` question is closed with it" (`ai_docs/features/070_pass_reads/01_spec.md:27-28`), after a six-layout brainstorm where every arc layout was rejected as "messy" and degrading at 480px. The spec cites "092's graph view" twice as an existing plan with a trigger ("Trigger: 091 has landed"), and the roadmap names no 092 — grepped `ai_docs/roadmap.md`, neither 091 nor 092 has a row or a banner mention. Anchor (h) is the maintainer preferring **hand-drawn rendering over `imgui_node_editor`** *if* such a view happens, which is a note on a hypothetical, not a commitment that reopens 070. The spec must not assert a follow-on feature as the reason a 091 decision is safe ("092's graph view draws a group as a region and needs no contraction" is the stated justification for the fold rejection). |
| copilot `import_passes` tool / `preset:` prefix | **CONSISTENT**; deferring the tool while teaching `group` to the table and `set_pass` matches the actor model (one fact the model reads matches `graph.json`). |
| group rename verb | **CONSISTENT** with the speculative-machinery bullet (`conventions.md:157-166`) — surface that must be taught gets cut at N=1. |
| nested groups | **CONSISTENT**; dropping inner labels is the only answer a single `str` field can give. |
| importing the source's script | **CONSISTENT**; a document has one script (048) and `ProjectSession._resolve_scripts` keys the engine per document id (`project_session.py:526-538`), so there is no per-group script slot to import into. |
| importing feedback seeds | **CONSISTENT**; `_feedback` is populated only by `_seed_feedback` at load and by `_feedback_canvas` at render (`document.py:554-670`), so a freshly built `Pass` has no history and "starts black" is what the code already does — no code needed for this bullet. |

---

## Should-not-land findings

**F1 — the spec's worked example, its prefill test and two of its eleven verification items are
written against a document that is not in the repo.** Evidence: `ls
shaderbox/resources/document_examples/` is six dirs; their `ui_name`s are Fire, UV Mango, Media
Input, Radiance Cascades, Night City, Text Rendering; five are single-pass per their `graph.json`
`passes` keys. Verification item 3 says "the Bloom Chain example imported into the starter
document with `scene` substituted by `main`" and item 4 builds on it, so the feature's only
end-to-end test and its only rendered-output test are both unimplementable as written. Why it
matters beyond a name: the *only* multi-pass source available is RC, whose one entry point
`paint` is a pass with **no samplers at all**, so a test built on it exercises the
single-entry-point path and never the substitution-with-readers path the feature exists for.
Either the test constructs its multi-pass source in the fixture (the honest option — `add_pass`
plus `write_text` builds a three-pass bloom shape in a dozen lines), or the feature ships a
Bloom Chain example and says so in "Files touched", which it currently does not.

**F2 — D7's three strip claims do not survive contact with `preview_cell` and
`tiles_per_row`.** Evidence in the table (E2, E3): the cited mock narrows the in-group gap to
4px while the spec says gaps do not change; the 2px-outside outline has 4px of slack at
`avail=700` (computed with the real `SIZE.PASS_TILE=168` / `SPACE.MD=8`); and the border the
spec turns off with a parameter is an imgui `ChildFlags_.borders` flag, not a pushed colour
(`ui_primitives.py:1268`). As written, an implementer either follows the mock and breaks the
"untouched" claim, or follows the spec and ships an outline that does not read as one thing.

**F3 — D6's handover writes have no stated release, and the leak is silent.** Evidence:
`project_session.py:1008-1010` releases the old value on every sampler write for exactly this
reason; a handover row that bypasses it over a bound `Image` or `Video` strands a texture (and,
for a `Video`, an open `cv2.VideoCapture`) with no reference to free it — `Pass.release` can only
free what `uniform_values` still holds (`core.py:286-297`). Nothing observable fails; the
frame count is identical.

**F4 — D11 as written licenses a third copy of the one-first-render-per-frame budget.**
Evidence: `ui.py:480-500` is already the second copy of the rule stated at 258-267, and the
conventions funnel bullet names the second sibling fix as the trigger to move to the funnel
(`conventions.md:172`). The spec's own framing ("without a third render block") shows the author
expected two branches where there are three.

**F5 — `GROUP_TINTS` as listed fails the assert D8 proposes.** `blue_b` is the `blue` accent's
primary (`theme.py:99`) and `COLOR.TAG` (`theme.py:172`).

---

## Spec edits proposed

**Edit 1 — D2, D3, and verification 3/4: replace the Bloom Chain anchor.** In D2, replace the
three pinned inputs with names that exist plus a synthetic: "(`Radiance Cascades` → `radiance`,
`Media Input` → `media`, `2D SDF` → `g_2d`)". Add after the `group_slug` sentence:

> A document whose `ui_name` was never set falls back to its directory name
> (`ui_models._load_ui_state`), so the slug of a UUID-named document is long but valid; the
> dialog prefills it and the user retypes it.

In D3, replace "Bloom Chain and Radiance Cascades have exactly one (`scene`, `paint`)" with:

> Radiance Cascades — the one multi-pass example shipped — has exactly one (`paint`); a
> single-pass document's one pass is both entry point and output.

In Verification 3, replace the subject:

> **`import_passes` end to end** (`tests/test_pass_verbs.py`): a three-pass source built in the
> fixture (`scene` → `bright` → `composite`, `bright.u_src` and `composite.u_scene` reading
> `scene` by name) imported into the starter document under group `bloom` with `scene`
> substituted by `main`; reload from disk; assert the two files under `passes/`, the two entries
> with `group == "bloom"`, the rows `bloom_bright.u_src == {"pass": "main"}` and
> `bloom_composite.u_scene == {"pass": "main"}`, the output `bloom_composite`, and that the
> source document's passes still hold their own textures after `release()` of the host (D5's
> copy). The source is built rather than taken from the shipped set because the only multi-pass
> example (Radiance Cascades) has a sampler-free entry point and so exercises no substitution.

**Edit 2 — D7, the three strip claims.** Replace the sentences from "`tiles_per_row` is
untouched" through the `bordered` parenthetical with:

> Member tiles draw with the group tint as `bg_color` at low alpha and no border of their own:
> `preview_cell` gains `bordered: bool = True`, which drops `ChildFlags_.borders` from the
> child's flags — the border is imgui's own child border, not the pushed `Col_.border`, so a
> colour cannot turn it off. A tile with an explicit `border_color` (the accent output border,
> the red error border) keeps the flag, so those still win.
>
> The run's geometry is reserved, not drawn over: the strip already knows each tile's screen
> rect before it draws, so the outline is emitted to the parent draw list **before** the run's
> tiles, and the label's fill after. The run keeps the strip's `SPACE.MD` gap — the mock's 4px
> in-group gap is dropped, because a per-run gap makes `tiles_per_row`'s single-gap arithmetic
> wrong and the outline reads as one thing at 8px with the members' own borders gone. The
> outline is drawn **inset** by 1px rather than 2px outside: `tiles_per_row` charges the last
> tile no trailing gap, so a full row's slack is as little as 4px (168/8 tokens at a 700px
> panel) and an outside rect clips. The label sits on the top border inside the first tile's
> top, on a `BG_SURFACE` fill. `tiles_per_row` is then genuinely untouched.

**Edit 3 — D5, the copy roster and the two mechanics.** Replace "copy uniform values — scalars
and tuples by value, …" with:

> copy uniform values over the whole of `core.UniformValue` plus the three `SamplerSource`
> members, with no default branch: scalars and tuples by value; a `PassSource` / `NoSource` /
> `AutoSource` by reference (frozen dataclasses, so the reference IS the value, and these are the
> values D4 leaves alone); a `moderngl.Buffer` via `gl.buffer(buf.read())`; an `Image` re-opened
> from its file under the source's `media/<source pass>/`, never `Image(value.texture)` — a
> texture-built `Image` comes back with an empty `file_details`, so the media panel loses the
> path and size; a `Video` re-opened from `details.file_details.path`; a raw `moderngl.Texture`
> via `gl.texture(size, components, data=tex.read(), dtype=...)`. The source directory the files
> are read from is `document.document_dir_of(source)`, so `import_passes` takes no `source_dir`
> parameter — that function exists so one place knows the `passes/` depth.

And append to the `ui_uniforms` sentence:

> `get_uniform_hash` is keyed by name and shape, not by pass, so a merged row whose name and
> shape the host already uses keeps the HOST's row; the source's input type and range are lost
> in that one case, which is the right precedence and the limit of the merge.

**Edit 4 — D6, the release and the host compile.** Append to the paragraph ending "moves the
output when the flag is set":

> A handover row overwrites a host sampler whose old value may be a bound `Image` / `Video` /
> `Texture`, so each write releases it first (`try_to_release`, the same call
> `set_sampler_source` makes) — `import_passes` writes `uniform_values` directly and saves once
> at the end rather than calling `set_sampler_source` per row, which saves per call. The host's
> program-less passes are compiled when the dialog opens, the same loop D3 runs on the source:
> an uncompiled host pass's wiring is its explicit rows only, so without it the reader
> checkboxes miss every name-rule wire.

**Edit 5 — D11, the branch count.** Replace D11's body with:

> **D11 — while the dialog is open, the planned render set is the open tab's documents plus the
> current document.** The predicate `_tick_frame_state` already computes as `examples_open`
> becomes "the Examples popup, or the import dialog with its Examples tab active", and
> `planned` / `planned_documents` / `current_planned` follow it unchanged (090 D10). The render
> chain is **not** given a fourth branch: `ui.py`'s `elif EXAMPLES:` block becomes
> `elif <that same predicate>:`, so the one-example-per-frame first-render election, the
> `renders_this_frame` gate and the profiler span have one home rather than a third copy —
> `ui.py` already carries two copies of that budget rule and the funnel bullet names the second
> sibling as the trigger. With the project tab active the import dialog renders the ordinary
> set, which means the `if not any_popup_open():` branch grows the same predicate's negation.

**Edit 6 — out of scope, the graph view.** Replace both mentions. In the fold bullet:

> **Folding a group** into one tile. Rejected in chat: a folded group must be convex in the DAG,
> which turned into a rule set the UI could not carry. Trigger: none — a group that is not
> contiguous simply draws as two runs (D7), which is the behaviour folding would have had to
> special-case.

and delete the "**The graph view** (092). Trigger: 091 has landed." bullet entirely, or replace
it with:

> **A spatial view of the graph.** Closed by 070, not deferred: "no graph view. The strip is the
> one view of the passes… the `imgui_node_editor` question is closed with it"
> (`ai_docs/features/070_pass_reads/01_spec.md`). 091 adds no pressure to reopen it — a group is
> a label on the strip. Trigger: the maintainer reopens 070's decision.

**Edit 7 (one line) — D8, drop the colliding tint.** Replace the six hues with five:
`purple_b`, `green_n`, `aqua_n`, `orange_n`, `blue_n`. `blue_b` is the `blue` accent's primary
and `COLOR.TAG`, so the tuple as written fails the assert D8 itself proposes.

---

## False trails — probed and fine, do not re-check

- **`effective_wiring()` on an uncompiled pass, and whether D3's compile disturbs the source.**
  Both answered by measurement, not reasoning: the wiring is the explicit-rows-only subset
  before compiling (empty for RC, which has no rows) and the full DAG after, and the compile
  loop leaves `Document.first_render_done`, every `Pass.first_render_done`, every `drawn_frame`
  and `feedback_passes()` exactly as they were. D3 is safe as written.
- **`UIDocument.save`'s prune + asset sweep against D5's merged rows and copied assets.** Ran
  the full D5 sequence on a real host and a real source with a bound `Image` and `Video`: all
  merged rows survived the prune, both assets were written under `media/<new pass>/` and kept.
  No edit needed; the only caveat is the name+shape hash collision noted in Edit 3.
- **`PopupState.IMPORT_PASSES` colliding with an exhaustive enumeration.** Exactly one site
  enumerates the enum (`tests/test_project_management.py:636-667`, AST-parsing `ui.py` for
  popup draw calls against `len(PopupState) - 1`), and the spec already lists the `ui.py` draw
  call that satisfies it. No other site counts members.
- **`set_pass(group=)` against the pass-verb gating test and the schema locks.**
  `test_the_pass_verbs_are_mutating_with_delete_lazy_and_gated` asserts only eager/mutating/gate
  and calls `add_pass` with one argument, so it is untouched; the only schema assertion is
  `additionalProperties is False` (`tests/test_tool_registry.py:38-40`), which a defaulted
  `ToolArgs` field keeps. The real cost is the five positional `backend.set_pass(...)` calls in
  `tests/test_copilot_pass_tools.py` and the positional-only protocol signature — append the
  parameter, never insert it.
- **`load_graph`'s salvage riding `PassEntry.group` for free.** Verified by running today's
  loader against a `graph.json` that already carries the key: it logs the unknown key and loads
  the rest, so the field is additive in both directions and `GRAPH_JSON_VERSION` staying 2 is
  right.
- **`entry_points` on the three shapes D3's verification names** (RC, a self-reading single
  pass, a two-input compositor): all three answer what the spec claims.
- **`group_slug` on every shipped `ui_name`**: all valid against `_PASS_NAME_RE`, including the
  UUID-fallback and empty-name edge cases.
- **Whether the D7 outline is reachable from the parent draw list at all.** Yes —
  `imgui.get_window_draw_list()` from the strip's scope, the pattern `ui_primitives.py` uses in
  a dozen places. The open question was never reachability but draw ORDER, which Edit 2 settles.
- **`make gates` baseline**: exit 0, captured unpiped, check + test + smoke all passed — so
  every failure this review names is about the spec, not a pre-existing red.
