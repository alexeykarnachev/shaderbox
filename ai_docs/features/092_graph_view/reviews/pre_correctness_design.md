# 092 pre-implementation review — correctness & design

Anchors read in full: `03_spec.md`, `02_triage.md`, `01_brainstorm.md`, all five
`reviews/brainstorm_*.md`, `conventions.md ## Design decisions`, `dev_flow.md` (Feature flow
step 2 + the module map), `.claude/skills/imgui-ui/SKILL.md` §3/§4/§6/§7.4/§8. Code read:
`pass_graph.py` whole; `document.py` (`load_graph`, `_keyed_entry_fields`, `sampler_names`,
`effective_wiring`, `_reads_of`, `graph_errors`, `drop_feedback`, `forget_pass_sources`,
`rename_pass_sources`); `project_session.py` (the six verbs, `import_passes`, `_graph_without`,
`_graph_renamed`, `_pass_name_error`, the capability injection block); `widgets/pass_list.py`;
`tabs/document.py`; `ui_regions.py`; `ui_models.py` (`UIAppState`, `UIDocumentState`,
`_load_ui_state`, `load_document_from_dir`); `theme.py`; `ui_primitives.py` (`preview_cell`,
`segmented_choice`, `text_tab_row`, `context_menu_style`); `app.py` (`pick_pass`,
`open_pass_settings`, `open_add_pass`, `open_import_passes`, `forget_render_state`,
`__init__`'s per-document dicts); `copilot/tools/passes.py`; `copilot/backend.py`
(`_pass_table`, `set_pass`, `_configure_pass`); `tests/test_button_tiers.py`,
`tests/test_ui_prose_budget.py`, `tests/test_graph_persistence.py`.

Verdict up front: **PARTIAL** — three items must change before implementation (D11's cycle-edge
rule, D6/D9's Arrange scope + `rank_layout` contract, D18's hypothetical-wiring definition), and
five more are under-specified enough that a blind coder would guess. No item contradicts a locked
triage constraint or a brainstorm "Fixed" item, and the spec carries **no** migration, compat or
graceful-evolution language (grepped: the single `backward` hit is the geometry sentence in D4).

---

## Decision table

| # | Verdict | Evidence |
|---|---|---|
| **D1** | (a) + (d) | Consistent with triage S1 verbatim and with the wiring review's case 11 ("the graph's port list has a different source of truth than its edge list"). `sampler_names` reads `render_pass.program` and returns `[]` when `program is None` — confirmed by reading it. The "compiles what it draws" branch matches S1's first consequence and does not disturb `Document._bring_chain_online`. **Under-specified twice.** (i) "`sampler_names(pass)` in declaration order" — `sampler_names` iterates the moderngl `Program`, which is not guaranteed source order; the coder must know whether to sort, to trust program order, or to re-derive from the source text. (ii) "with `prev` (a self-read) last" conflates two different things. Probed: `wired_pass(PassSource('trail'),'u_anything','trail',{'trail'}) -> 'trail'` — a self-read can sit on ANY sampler, not only `u_prev`; and `wired_pass(AutoSource(),'u_prev','trail',{'trail'}) -> 'trail'` — `u_prev` is itself a declared sampler that is already in `sampler_names`. So "`prev` last" is either "sort the `u_prev` entry to the end" or "every self-reading port to the end", and the two differ on a pass with `u_anything = PassSource(self)`. |
| **D2** | (a) + (d) | Matches triage D2 default (a) and the interaction review's §4 recommendation. `UIAppState` carries `model_config = {"extra": "forbid"}` and already holds `channel_view: ChannelView` beside `active_document_tab` — an enum-valued view preference is precedent, and `ui_regions.py`'s docstring states exactly why `PassesView` belongs there (a leaf with no imgui, persisted by `ui_models`). App-level vs per-document is the right altitude: `channel_view` is the sibling and it is app-level; the per-document half is `GraphViewState`, which is correctly NOT persisted. The lazy-row disclaimer is sound — nothing off-draw writes `GraphViewState`. **Gap:** `App.graph_views` is a per-document-id dict and `App.forget_render_state(document_id)` is the documented funnel that drops every ephemeral per-document entry when a document goes away (`pending_resolution`, `auto_size_states`, `throttle_states`, `document_costs` all pop there). `graph_views` is not named in D2 or in Files touched, so it would leak an entry per deleted/closed document. |
| **D3** | (a) + (d) | Scope revalidation per frame answers the mutations review's case 7 ("the active tab key must be revalidated every frame against the live label set") exactly. Tab-row root label reachable: `UIDocumentState.ui_name` exists, and `_load_ui_state` does `filtered_ui_state.setdefault("ui_name", dir_name)`, so a loaded document always has a non-empty name **unless the user clears the input** — `tabs/document.py` writes `ui_document.ui_state.ui_name = imgui.input_text_with_hint("##document_name", "document name", …)` with no non-empty guard, so `""` is reachable and would render an unclickable zero-width root tab. **Structural gap:** `ui_primitives.text_tab_row(id_, names, active) -> str | None` keys by NAME and returns the clicked NAME. The root's key is `""` (D3's own scope value) but its LABEL is the document name, so the widget as written cannot express "root" — and if the document is named `bloom` while a group is named `bloom`, the row has two identical entries and the return value is ambiguous. D17 forbids a pass/group collision but says nothing about a document/group one. |
| **D4** | (a) | Matches brainstorm Fixed #3 and #4 and the mutations review's case 13 (non-convex probed: `mid` is a ghost on both sides, no special case). The feeder-left/reader-right twin is the mock's `·` suffix made explicit. "Groups never draw a tinted region" resolves cleanly against `pass_list._draw_group_outline`, which stays untouched (S15). |
| **D5** | (a) | Implementable and **total on every shape the mutations review probed** — I wrote `bundle_output` exactly as D5 words it and ran it: `bloom -> b_comp (first-read-outside)`; `non-convex {g1,g2} -> g1`, out-ports `['g1','g2']`; `generator -> gen2`; `one-member -> b_comp`; `split {x,y} -> x`, out-ports `['x','y']`; `deleted-output -> b_blur (last-unread-by-members)` with out-ports `['b_blur']` — which is exactly the mutations review's case-2 rule break, now closed; `group-is-output -> g2 (doc-output)`; `two-unread-terminal -> m2`; `all-members-read (a cycle inside the group) -> m2 (last-member)`. D5's fourth branch ("else the last member") is an addition over triage S4's three, and it is the branch that makes the function total — without it the `all-members-read` shape returns nothing. A strict refinement of S4, not a contradiction. "strip order" is unambiguous: `strip_order(names, wiring)` takes the name SET and plans the whole wiring, so "filter the document order" and "order the members" coincide (probed on two shapes, identical results). |
| **D6** | (a) on the model, (b)-adjacent on the verb scope, (d) on `rank_layout` | **The pydantic annotation works.** Probed `tuple[GraphCoord, GraphCoord]` with `GraphCoord = Annotated[float, Field(allow_inf_nan=False, ge=-1e5, le=1e5)]`: `[nan,0]` → `finite_number`; `[inf,0]` → `finite_number`; `[1e30,1e30]` → `less_than_equal`; `[100001,0]` → `less_than_equal`; `[10,20]` → OK; `['a',1]` → `float_parsing`; `[1.0]` → `missing`; `[1,2,3]` → `too_long`. This closes the persistence review's measured gap (a bare `tuple[float,float]` accepted NaN and 1e30). **The salvage carries it.** Probed through the real `document.load_graph` with a `position` field monkeypatched onto the entry model: entries with NaN, 1e30 and `"garbage"` positions each logged `Ignoring invalid graph.json.passes.<n>.position` and loaded with `position=None` while `iterations` and `group` survived beside them; the clean entry kept `(10.0, 20.0)`. Exactly what conventions' "a retired or malformed field costs the user THAT setting, never the rest of the file" requires, with no bespoke code — `_keyed_entry_fields()` enumerates from `PassGraph.model_fields`. No version bump matches S8 and the repo's practice (091 added `group` without one). **Under-specified:** `rank_layout`'s signature is never given, and D6 and D9 disagree about its domain — D6 says it "places the unplaced passes every frame", D9 says Arrange "writes `rank_layout` over every visible pass". A function that places only the unplaced ones cannot produce Arrange's full layout, and one that places all of them cannot be called per frame for the unplaced subset without clobbering. Also unstated: what `rank_layout` takes (a wiring? plus the group map, since "within a rank, group members adjacent"? plus the already-placed positions, since "then by the mean position of predecessors" needs them?) and what it returns. |
| **D7** | (a) + (d) | Every draw-list call named exists on this binding — probed `ImDrawList`: `add_image_rounded`, `channels_split`, `channels_set_current`, `channels_merge`, `add_bezier_cubic`, `add_circle`, `add_circle_filled`, `add_rect`, `add_text` all present. `no_scroll_with_mouse` matches the interaction review's measured requirement (416 px of scroll without it) and `preview_cell` already sets it on its own child. The fractional-font claim is the probe's (`get_font_baked().size` tracked the ask). **Gap:** the transform is given as `screen = origin + (canvas - pan) * zoom` while the interaction review's probe used `origin + (p + pan) * zoom` — opposite sign on `pan`. Either is fine internally but the fit/zoom-about-cursor arithmetic in D9 must use the same convention, and the spec states only one of the two places. Minor and self-correcting at impl time; noted so it is not read as a transcription of the probe. |
| **D8** | (a) | Matches the interaction review's two measured FAILS verbatim and triage S6. The three-deep chain is the probe's own finding (`allow_overlap on bg = False` → `node_hov False, node_act False`; `= True` → `node_hov True`). `begin_popup_context_item(None)` is the binding's documented "associate the popup to previous item" form; the imgui skill §8 has no entry for either rule yet, which is why D20 adds them. The port-hit floor answers the probe's 3.5-px-at-zoom-0.5 qualification. `set_next_item_allow_overlap` exists on this build (probed). One imgui-skill §4 interaction the spec handles implicitly: the canvas uses `set_cursor_screen_pos` before every `invisible_button`, and each one covers the moved-to position, so the SetCursorPos assert cannot fire. |
| **D9** | (c) on Arrange's scope, (a) otherwise | Pan/rubber-band split matches triage D4 default (a) plus Alt as the interaction review's option C — both, which is more than either but contradicts nothing. Fit-once-on-first-nonzero-size is the right shape (a zero-size child on the first frame would produce an infinite zoom). **Conflicts with D6.** "Arrange … writes `rank_layout` over every VISIBLE pass" — inside a group tab the visible set is the members plus ghosts, so Arrange from a group tab would rewrite the ghosts' (non-member) positions, or not, and the spec does not say. Worse, S5 and the interaction review both require a multi-pass verb to write once for the whole set; "every visible pass" makes the written set depend on the current scope, which means Arrange is a different operation at the root than in a tab. This needs one sentence. |
| **D10** | (a) + (d) | `pick_pass(document_id, name, focus_editor)` exists on `App` with that exact signature and is what `pass_list._draw_pass_tile` calls. The extraction target is safe: `pass_list._draw_context_menu` today is `with context_menu_style(): if imgui.begin_popup_context_item(f"##pass_menu_{name}"): <three items> imgui.end_popup()`. Lifting the three items into `pass_menu_items(app, document_id, name)` and leaving the strip's `context_menu_style()` + `begin_popup_context_item(<id>)` + `end_popup()` in place is behaviour-preserving — the items read only `app`, `document_id`, `name` and `document.passes` / `document.graph.passes`, and `_delete_pass` is already a module-level function. **Under-specified:** `_draw_context_menu` gates Delete on `len(document.passes) > 1` and shows `Leave group` only when the entry carries a group. The canvas's box menu ("Open, and Dissolve") and the ghost's menu are not said to use `pass_menu_items` at all — a ghost is a real pass, so does right-clicking one offer Settings/Delete/Leave group on a pass outside the current scope? D4 says a ghost click changes scope; it does not say what a ghost right-click does. |
| **D11** | **(b) — this rule fires on nothing** | The cue rule reads: "The edges between two passes that `graph_errors` both name as cycle culprits (`message` starting with 'passes form a cycle') are `STATE_ERROR`". Probed `plan_passes` on three shapes: 2-cycle `{a↔b}` → culprits `['a']`, all errors `['a','b']`; 3-cycle `{a→b→c→a}` → culprits `['a']`, all `['a','b','c']`; two independent 2-cycles → culprits `['a','x']`. **`_cycle_message` is emitted exactly once per cycle** (`failures.setdefault(name, GraphError(name, _cycle_message(trail, name)))` fires on the back-edge visit only; every other member gets "pass is not ordered: an input is on a cycle."). So "two passes that BOTH carry the culprit message" is never true of the two endpoints of a cycle edge, and **zero edges would be reddened** on the manual-verification step 7 shape. Probed that shape directly: `{paint→composite→cascade→paint, df→paint}` yields one culprit, `cascade`, whose message is `passes form a cycle: cascade -> paint -> composite -> cascade`. The trail inside that one message names the whole loop and is the workable input. The triage's own wording ("`STATE_ERROR` on the edges of a cycle") is right; the spec's operationalisation of it is not. Everything else in D11 checks out: the strip's `live` set expression is copied verbatim from `pass_list.draw`; `STATE_ERROR`-over-`ACCENT_PRIMARY` precedence matches `_draw_pass_tile`'s `COLOR.STATE_ERROR if errors else COLOR.ACCENT_PRIMARY if is_output else None`. |
| **D12** | (a), and the fixed-item-6 "contradiction" is not one | Fixed item 6 reads "**Wiring is a drag from an output dot into a port** … dropping on empty space writes `NoSource`". Read in its own sentence, "dropping on empty space" is the terminal clause of a sentence whose subject is the output→input drag — but the write it names (`NoSource` on a sampler) is only expressible when the drag STARTED at an input port, because a drag from an output has no sampler to write to. `set_sampler_source(document_id, pass, uniform, source)` requires a `(pass, uniform)` pair; a drop from an output onto empty canvas supplies neither. **D12's reading is the correct one and the only implementable one**, and it is also what triage D5(a) locked ("release on empty writes `NoSource` on the original", i.e. on the input port the drag grabbed). Fixed item 6 is imprecise, not contradicted. Recommend one clause in D12 saying so, so the next reader does not re-litigate it. The cycle refusal matches S2 ("refuse on `errors != []`, since the culprit need not be an endpoint") — and `plan_passes` bears that out: on `{paint→composite→cascade→paint}` the culprit is `cascade`, not either endpoint of the edge that closed it. |
| **D13** | (a) | `io.mouse_delta / zoom` is the probe's measured rule (40,20 screen → 40,20 graph at zoom 1; → 20,10 at zoom 2). Write-on-release matches the interaction review's cross-cutting note ("a per-frame `save_ui_document` on every drag frame would write the document file ~60×/s"). Snapping matches brainstorm Fixed #5 ("snapping is a drag helper … never a regime"). |
| **D14** | (a) | `set_pass_groups` on `ProjectSession` is the right altitude and satisfies the funnel law: **the copilot reaches the same validator**, because `project_session.py` injects `pass_set_group=self.set_pass_group` into the capability bundle, `backend._configure_pass` calls `self._pass_set_group(document_id, name, group)`, and `copilot/tools/passes.py::set_pass` forwards `args["group"]` into `caps.set_pass`. So routing `set_pass_group` through the batched verb (as D14 says) puts the D17 collision check on the copilot's path for free — no second validator. This is also what avoids the persistence review's "three independent group validators" finding at the two sites 092 touches (`plan_import`'s own `PASS_NAME_RE.match(group)` stays a third, untouched and out of scope). Save-once matches S5. |
| **D15** | (a) + (d) | Correct as far as it goes: probed `wired_pass(<a bound texture>, 'u_image', 'main', {...}) -> None`, so a media-bound sampler is in `sampler_names` and absent from the wiring, exactly as D15 says. **Under-specified:** `wired_pass` returns `None` identically for a bound texture, a `NoSource`, and an unresolvable `AutoSource`. D11 asks the port to draw four distinguishable states (hollow ring / ring-with-centre / double ring / square) and **no decision names where that classification comes from** — it cannot come from the wiring. The answer exists in the tree: `widgets/uniform.py` branches on `isinstance(current_value, PassSource | AutoSource)` / `NoSource` / `MediaWithTexture \| moderngl.Texture` over `Pass.uniform_values`. The spec must say the canvas reads the same value, and `document.py` / `pass_graph.py` are where a shared classifier would live if one is wanted. |
| **D16** | (a) | Matches triage S10 verbatim. `pass_list._delete_pass` exists as a module-level function capturing the doomed path before `session.delete_pass`, then `app.close_editor_for_path(doomed)` — reusable as named. `delete_pass` itself refuses the last pass (`if len(document.passes) == 1`), which the strip's menu mirrors with `deletable`. |
| **D17** | (a) | Matches triage S11 and the mutations review's case 18 (probed there: two root entities keyed `bloom`). `_pass_name_error(name, existing)` is the one place `add_pass` and `rename_pass` validate a name and it takes only `existing: dict[str, Pass]` — adding the group check means either widening that helper or checking beside it; either is a one-line decision, not a gap. The message is a single clause and within the app's prose habits. |
| **D18** | (b)/(d) — the stated mechanism does not compute what it claims | S12 locks the guard; D18 implements it. The claim under test: "`Document.wiring_if_renamed(old, new)`, which swaps the pass under the new key, reads `effective_wiring()`, and restores". **Swap-and-restore is safe** as regards state: I read `effective_wiring` and `_reads_of` and confirmed neither references `self._feedback`, `self.graph`, or `self._graph_errors` — they read only `self.passes`, each `Pass.uniform_values`, and `sampler_names`. So re-keying `self.passes` and restoring it is sufficient to produce a wiring, and the feedback dict (keyed by name, popped only by `drop_feedback`) is untouched. **But the wiring it produces is not the post-rename wiring.** `rename_pass` also calls `document.rename_pass_sources(old, new)`, which rewrites every explicit `PassSource(old)` to `PassSource(new)`. A key swap alone leaves those rows naming a pass that no longer exists, so `wired_pass` drops them: probed, `{a:{}, b:{u_x: PassSource('a')}}` renamed `a→a2` gives `{'a2': {}, 'b': {}}` under swap-only (the edge GONE) versus `{'a2': {}, 'b': {'u_x': 'a2'}}` with `rename_pass_sources` applied. I then brute-forced every 3-pass graph over {no edge, name-rule edge, explicit edge} on all ordered pairs, for three rename targets, comparing the cycle verdict of swap-only against swap+`rename_pass_sources`: **0 divergences** in the verdict (dropping edges can only remove cycles, and an explicit edge's endpoints do not change meaning under a rename, so any cycle it closes already exists). So the guard is not *wrong* today — but it is stated as computing "the wiring the document would have after the rename", which it does not, and a coder implementing the sentence literally ships a function whose name is a lie and whose correctness rests on an argument nobody wrote down. The fix is one clause. The guard's real target is confirmed reachable: probed `{scene, bright(u_scene=PassSource(scene), u_final=Auto), comp(u_bright=PassSource(bright))}` clean before; renaming `comp→final` makes `bright.u_final` resolve by the name rule and the plan reports `passes form a cycle: bright -> final -> bright` — the mutations review's case 9, now guarded. |
| **D19** | (a) | Matches triage S7 and the persistence review's item 3 verbatim. `backend._pass_table` emits one flat row per pass with `group X` appended and nothing positional; no tool signature changes. The guard test from the copilot skill ("would a strictly better model produce a better shader if it knew where the nodes sit?") is applied and answered in the persistence review; D19 records the answer rather than re-deriving it, which is right. |
| **D20** | (a) + (d) | Every doc target exists: `conventions.md`'s "A pass GROUP is a label … and nothing folds (feature 091)" bullet is there with its revisit trigger; `dev_flow.md ### Module map` lists `widgets/pass_list.py` and `pass_graph.py` and would take the two new entries; `.claude/skills/imgui-ui/SKILL.md` §8 is the version-pinned-quirks section and both D8 rules are quirks of this binding (neither is present today — grepped); `help_content.py` has a `title="Passes"` block; `ai_docs/features/070_pass_reads/01_spec.md` exists at that path. **Gap:** D20 says the 091 bullet "gets the revisit pointer", but conventions' own form is "we decided X; revisit if Y" — the 091 bullet's existing trigger is "Revisit if a group-level fact appears that no member can hold". 092 does not fire that trigger (it stores nothing group-level — D5's box is derived, D6's box has no position), so what the bullet needs is not a *revisit* but a *scope clarification*: the no-folding decision was about the STRIP. The edit text is given below so the coder does not have to invent it. |

---

## Files touched — line by line

| Line | Verdict | Note |
|---|---|---|
| `pass_graph.py`: `PassEntry.position` + `GraphCoord` + `MAX_GRAPH_COORD`, `PassGraph.with_positions`, `rank_layout`, `graph_ranks`, `group_boundary` (+ `Boundary`, `BoxPort`), `bundle_output`, `wiring_with`, `cycle_message_for` | (a), one gap | All GL-free and pure, which matches the module's docstring promise and S9. `with_positions` joins `with_passes`/`with_target`/`with_output`/`with_group` correctly (the module's stated funnel). `cycle_message_for` is the one symbol with no stated contract — given the D11 finding it is also the one that matters most: it should extract the trail from the single culprit's message, not assume one message per member. |
| `document.py`: `wiring_if_renamed` | (d) | See D18. Also unstated: whether it is a method on `Document` (the spec's phrasing) and whether it restores under a `try/finally` — it must, or an exception inside `effective_wiring` (a compile raising) leaves `self.passes` re-keyed and the document corrupt in memory. |
| `project_session.py`: `set_pass_positions`, `set_pass_groups`, `set_pass_group` routed through it, the D17 checks in `add_pass`/`rename_pass`/`set_pass_groups`, the D18 guard in `rename_pass`, `import_passes` stripping positions | (a) | Correct altitude (all six existing verbs live there and each ends in `save_ui_document`). `import_passes` stripping is a one-line change to the existing `.model_copy(update={"group": group})` → `update={"group": group, "position": None}`, which is where the persistence review's case-19 finding lands. The block comment above the verbs says "the pass graph's SIX verbs (D15)" and would become eight — worth updating in the same edit. |
| `ui_regions.py`: `PassesView`, `PASSES_VIEW_LABELS` | (a) | Exactly the shape `ChannelView` + `CHANNEL_VIEW_LABELS` already has in that file. |
| `ui_models.py`: `UIAppState.passes_view` | (a) | `extra: forbid` is on the model, so the field is loud rather than silently dropped — the posture's own default. |
| `theme.py`: `SIZE.GRAPH_*`, `COLOR.GRAPH_EDGE`, `COLOR.GRAPH_GHOST_ALPHA` | (a), one gap | The token bag is the right home (imgui skill §6: "A token used by exactly one panel still belongs in the token bag"). `GRAPH_GHOST_ALPHA` is a float on `COLOR` — the precedent is `GROUP_FILL_ALPHA: float = 0.10`, so that is consistent. **Gap:** `theme.py` carries import-time asserts that `COLOR.GROUP_TINTS` collides with no state hue, accent or `SELECT`. A new `COLOR.GRAPH_EDGE` is a new fixed role; the spec does not say whether it joins `_GROUP_TINT_EXCLUSIONS`. D11 puts `STATE_ERROR` on edges, so `GRAPH_EDGE` and `STATE_ERROR` must be distinguishable, and a `GRAPH_EDGE` that happened to equal a group tint would make a wire read as a box border. One line either way. |
| `widgets/graph_state.py` (new): `GraphViewState`, `WireDrag`, `NodeDrag` | (a) | A state-only sibling module is sanctioned precedent — conventions' `tabs/*.py` bullet names `tabs/share_state.py` for exactly this ("to keep `app.py` import-cycle-free"). Necessary here for the same reason: `App` must hold `graph_views` while `widgets/pass_graph.py` annotates `app: App`. |
| `widgets/pass_graph.py` (new): the canvas | (a) | A leaf free function `draw(app, document_id)` matching `pass_list.draw`'s signature — the `widgets/*.py` convention exactly. |
| `widgets/pass_list.py`: caption and buttons move out; `pass_menu_items` extracted | (a) | See D10. Moving `small_caption(app.font_12, "Passes")` and the two `standard_button`s out leaves `draw` starting at `begin_disabled` — note the `imgui.begin_disabled` / `end_disabled` pair currently BRACKETS the buttons, so whichever module keeps them must keep the bracket; if the buttons move to `tabs/document.py` the disable must move with them or the copilot freeze silently stops covering `add pass` / `import...`. That is the exact "unwired mechanism counts as ABSENT" shape dev_flow step 7 warns about. Worth one clause in D2. |
| `tabs/document.py`: the caption row with the toggle, the dispatch, the button row | (a) | `segmented_choice(id_, options, selected) -> int` already used twice in this file (`_draw_resolution_mode`); `pass_list.draw(app, document_id)` is called at one site at the tail of `draw`. |
| `app.py`: `graph_views`, `graph_view_for`, `open_pass_settings` unchanged | (a), one gap | See D2 — `forget_render_state` is not named. |
| `help_content.py`: one sentence | (a) | The `title="Passes"` block exists. |
| `scripts/smoke.py`: a stretch with the graph view on | (a), noted | The persistence review's caveat stands and the spec does not repeat it: `make gates` reports a skipped smoke as **skipped, which is not a pass** (CLAUDE.md says so too), and the dev box has no display. So this stretch is crash-coverage on the maintainer's machine only. Fine as long as nothing else relies on it — and nothing does, since the pure functions carry the tests. |
| `tests/test_pass_graph.py` | (a) | Good falsifier list. The `position` bounds cases (NaN, inf, 1e30 rejected; a pair accepted) are the ones I probed green above, so they are known-writable. |
| `tests/test_graph_persistence.py` | (a), one addition | `test_a_graph_round_trips_every_field` already exists in that file and is the natural home for the round-trip half; the corrupt-position half is a sibling of the existing `test_a_malformed_graph_entry_costs_that_entry_not_the_document`. Both confirmed present by name. |
| `tests/test_pass_verbs.py` | (a) | The rename mutation test is stated with its falsifier ("with the guard removed the rename succeeds and the plan reports the cycle"), which is what conventions' gate bullet demands. Note it must be applied at the SITE THAT COMPUTES THE CONDITION — i.e. drive `ProjectSession.rename_pass`, not `wiring_if_renamed` directly (conventions: "Mutate the WIRING, not the renderer"). Say so, since the obvious test is the pure one. |
| `tests/test_button_tiers.py`: `widgets/pass_graph.py` `invisible_button` allowlisted | (a) | Verified against the test: `_NOT_A_VERB` is keyed `(module, call_name)`, so ONE entry covers the canvas background, every node body and every port. The reason string must be of the sanctioned kind ("the call draws something that is not a button with a word on it") — it is. `test_every_listed_exception_still_exists` means the entry cannot be added before the call site. |
| `tests/test_ui_prose_budget.py`: "the canvas menu labels are within budget; nothing to allow" | **(b) — inert** | Probed the gate's own domain: `sorted({r[0] for r in _SCORED})` contains 34 entries and **`menu_item_simple` is not among them**. The scored `imgui.*` rows are exactly `help_marker`, `set_tooltip`, `separator_text`, `text_colored`; everything else is derived from `ui_primitives` signatures. So no canvas menu label ever reaches this gate, and the line claims coverage that does not exist — the "a spec'd safety shipped as a no-op" shape. Either drop the line or add `menu_item_simple` as an `_IMGUI_ROWS` entry (which would also pull in `pass_list`'s existing Settings / Delete / Leave group, all within budget). |
| docs (`conventions.md`, `dev_flow.md`, the imgui skill, `roadmap.md`, `070/01_spec.md`) | (a) | All five paths verified to exist. |

**Section set vs dev_flow step 2:** the required minimum is *Goal / Out of scope (each deferral
with a trigger) / Design decisions (numbered, lock-in only) / Files touched / Open questions for
the user*. All five present; every one of the eight Out-of-scope bullets carries an explicit
Trigger; Design decisions are numbered D1-D20 with no open question mixed in; Open questions is
"None" with the reason. **PASS on the section set.** The two extra sections (Manual verification,
Review history) match 091's spec shape.

---

## EDITS the spec needs

Each is replacement text a coder can paste.

**1. D11 — the cycle-edge rule fires on nothing. Replace** the sentence

> The edges between two passes that `graph_errors` both name as cycle culprits (`message` starting with "passes form a cycle") are `STATE_ERROR`; victims get nothing.

**with:**

> Exactly ONE pass per cycle carries the culprit message (`plan_passes` does
> `failures.setdefault(name, GraphError(name, _cycle_message(trail, name)))` on the back-edge
> visit; every other member of the loop gets "pass is not ordered: an input is on a cycle."), and
> that one message names the whole loop: `passes form a cycle: cascade -> paint -> composite ->
> cascade`. So `pass_graph.cycle_message_for(errors)` parses the trail out of each culprit message
> and returns the set of consecutive `(producer, consumer)` pairs on it; every edge in that set is
> `STATE_ERROR`. Victims get nothing on the node and nothing on their edges.

**2. D6 + D9 — `rank_layout`'s contract and Arrange's scope. Replace** in D6

> `pass_graph.rank_layout` places the unplaced passes every frame

**with:**

> `pass_graph.rank_layout(wiring, names, groups, placed) -> dict[str, tuple[float, float]]` is
> pure and returns a position for every name in `names`; `placed` is the already-stored positions,
> read only for the "mean position of predecessors" tiebreak. The canvas calls it every frame with
> `names` = the passes whose `position is None` and uses the result for those alone; Arrange calls
> it with `names` = every pass of the DOCUMENT and `placed = {}`, and writes the whole result.

**and replace** in D9

> Arrange (canvas menu) writes `rank_layout` over every visible pass into `position` through `set_pass_positions` and fits.

**with:**

> Arrange (canvas menu) writes `rank_layout` over every pass of the DOCUMENT — not the visible
> subset, so the verb means the same thing at the root and inside a group tab — into `position`
> through `set_pass_positions`, one save, and then fits the current scope.

**3. D18 — say what the hypothetical wiring is. Replace**

> `rename_pass` computes the wiring the document would have after the rename (`Document.wiring_if_renamed(old, new)`, which swaps the pass under the new key, reads `effective_wiring()`, and restores) and refuses when `plan_passes` reports a cycle, with the planner's message.

**with:**

> `rename_pass` computes the wiring the document would have after the rename through
> `Document.wiring_if_renamed(old, new)`: it re-keys `self.passes` under the new name AND applies
> `rename_pass_sources(old, new)` to a copy of each pass's sampler values, reads
> `effective_wiring()`, then restores both in a `finally` (a compile raising inside the read must
> not leave the document re-keyed). Both halves are needed because `rename_pass` itself does both:
> a key swap alone drops every explicit `PassSource(old)` row, so the hypothetical wiring would be
> missing edges the real rename keeps. `rename_pass` refuses when `plan_passes` over that wiring
> reports any error, with the planner's message. (The guard's target is the name-rule edge a
> rename creates: renaming `comp` to `final` makes a sibling's `u_final` resolve where it resolved
> to nothing before.)

**4. D2 — name the eviction funnel and keep the copilot freeze on the buttons. Append** to D2:

> `App.graph_views` is dropped per document in `App.forget_render_state(document_id)`, beside
> `pending_resolution` / `auto_size_states` / `throttle_states` / `document_costs` — the funnel
> that already exists for every ephemeral per-document entry. The `add pass` / `import...` row
> moves to `tabs/document.py` INSIDE its own `begin_disabled(app.copilot_turn_active)`; the
> bracket currently in `pass_list.draw` wraps those buttons, and moving them out of it without
> re-establishing it would silently unfreeze them during a copilot turn.

**5. D3 — the root tab's key and its label collision. Append** to D3:

> `text_tab_row` keys and returns by NAME, so the root cannot be the empty string there: the row is
> built as `[root_label, *group_names]` where `root_label` is the document's `ui_name` or
> `"document"` when it is empty (`UIDocumentState.ui_name` defaults to the directory name on load
> but the Document tab's input can be cleared), and a click is mapped back to a scope by INDEX, not
> by the returned string — a document named the same as a group would otherwise be ambiguous.

**6. D15 + D11 — say where a port's state is read from. Append** to D15:

> `wired_pass` answers `None` identically for a bound texture, a `NoSource` and an unresolvable
> `AutoSource`, so the four port states of D11 are not derivable from the wiring. The canvas reads
> `Pass.uniform_values[sampler]` and branches by type exactly as `widgets/uniform.py` does:
> `MediaWithTexture | moderngl.Texture` → media (square), `NoSource` → ring with a filled centre,
> `PassSource | AutoSource` resolving to a pass → filled disc, otherwise → hollow ring. A sampler
> whose resolved pass is the consumer itself is the feedback port (double ring), whatever its name
> — `u_prev` is the common case, not the definition.

**7. D1 — port order. Replace**

> A node's input ports are `sampler_names(pass)` in declaration order with `prev` (a self-read) last

**with:**

> A node's input ports are `sampler_names(pass)` in the order that function returns (the compiled
> program's own iteration order), with every port whose resolved source is the consumer itself
> moved to the end — a self-read is the feedback port and reads bottom-most whether it sits on
> `u_prev` or on any other sampler explicitly sourced to its own pass.

**8. D10 — the ghost's and the box's menus. Append** to D10:

> `pass_menu_items` keeps the strip's two gates (Delete only while `len(document.passes) > 1`;
> Leave group only while the entry carries one). A GHOST's right-click offers the same item set —
> a ghost is a real pass and the verbs name the pass, not the scope. A BOX's menu is its own
> (Open; Dissolve in W2) and never `pass_menu_items`, since a box is not a pass.

**9. D20 — the 091 bullet's edit is a scope clarification, not a revisit. Replace**

> `conventions.md` "A pass GROUP is a label ... and nothing folds" gets the revisit pointer: the strip stays flat; the graph view contracts a group to its boundary edges and never orders the box, so convexity is not a rule anywhere.

**with:**

> `conventions.md`'s "A pass GROUP is a label … and nothing folds (feature 091)" keeps its own
> revisit trigger (a group-level fact no member can hold — which 092 does not create: the box is
> derived from its members every frame and stores nothing) and gains one sentence scoping the
> no-folding half to the STRIP: the graph view (092) contracts a group to a box whose ports are its
> boundary edges, and since the box is never a node the planner orders, convexity is not a rule
> there either.

**10. Files touched — drop the inert prose-budget line. Replace**

> `tests/test_ui_prose_budget.py`: the canvas menu labels are within budget; nothing to allow.

**with:**

> `tests/test_ui_prose_budget.py`: add `("menu_item_simple", "label", 0, 2)` to `_IMGUI_ROWS` so
> menu labels enter the gate at all — they are outside it today (the scored `imgui.*` calls are
> `help_marker`, `set_tooltip`, `separator_text`, `text_colored` only), which means the canvas menu
> AND the strip's existing Settings / Delete / Leave group are unmeasured. All of them are within
> the 2-word budget, so nothing needs allowing.

**11. Files touched — `test_pass_verbs.py`, name the mutation's layer. Append** to that line:

> The rename mutation is applied at `ProjectSession.rename_pass` (the site that computes the
> condition), never at `wiring_if_renamed` — a test that drives the pure helper with a wiring it
> supplies cannot discover that the verb never calls it.

**12. Theme — one clause in D7 or the Files-touched line:**

> `COLOR.GRAPH_EDGE` is a fixed role and joins `_GROUP_TINT_EXCLUSIONS` beside the state hues, so a
> wire can never be drawn in a group's tint.

---

## False trails (checked, fine — do not re-check)

- **The pydantic bounds.** `Annotated[float, Field(allow_inf_nan=False, ge=…, le=…)]` inside a
  `tuple[...]` on a frozen pydantic-v2 model DOES reject NaN, ±inf, 1e30 and an out-of-range
  magnitude, and also a wrong arity and a non-numeric string. Probed all eight cases. The
  constraint propagates into the tuple element; no `model_validator` is needed.
- **The per-entry salvage.** A new `position` on `PassEntry` rides `load_graph`'s existing
  `drop_unknown`/`drop_invalid` loop with zero new code, because `_keyed_entry_fields()` enumerates
  from `PassGraph.model_fields`. Probed with a real corrupt file: three bad positions dropped
  individually, siblings' `iterations`/`group` intact, the clean entry's position kept.
- **`bundle_output` totality.** Probed over nine shapes including all six the mutations review
  named; single-valued and defined everywhere. The mutations review's case-2 rule break (a box with
  a picture and no output dot) is genuinely closed by "the bundle output always, hollow when
  unread".
- **"strip order" ambiguity in D5.** `strip_order(names, wiring)` plans the whole wiring and filters
  to `names`, so ordering the members alone and filtering the document order give the same list.
  Probed on two shapes.
- **`_reads_of` / `effective_wiring` state dependencies.** Neither touches `self._feedback`,
  `self.graph` or `self._graph_errors`. A swap-and-restore of `self.passes` is therefore sufficient
  and cannot strand a feedback canvas (`drop_feedback` is the only thing that pops that dict).
- **D18 producing a FALSE PASS.** It cannot. Brute-forced every 3-pass graph over {no edge,
  name-rule edge, explicit edge} on all six ordered pairs × three rename targets, comparing the
  cycle verdict of swap-only against swap+`rename_pass_sources`: zero divergences. The edit above
  is about the function meaning what it says, not about a live hole.
- **The imgui surface.** `add_image_rounded`, `channels_split`/`_set_current`/`_merge`,
  `add_bezier_cubic`, `add_circle`/`_filled`, `add_rect`, `add_text` and
  `set_next_item_allow_overlap` all exist on this build. Nothing in D7/D8 needs a workaround.
- **The copilot reaching the D17 validator.** It does, through
  `pass_set_group=self.set_pass_group` → `backend._pass_set_group` → `_configure_pass` →
  `tools/passes.py::set_pass`. No second validator, no parallel path.
- **`_NOT_A_VERB` granularity.** One `(module, call_name)` entry covers every `invisible_button` in
  the new widget; the canvas does not need three.
- **Migration / compat language.** Grepped the spec for `migrat|backward|back-compat|compat|graceful|deprecat|legacy|old-format`: one hit, the word "backward" in D4's "no edge runs backward through the members". Nothing to delete.
- **Guard-pile smell.** I looked for the copilot skill's "a guard a better model would not need" and
  found none: D19 adds no tool, no prompt paragraph and no position field; D12's cycle refusal and
  D18's rename refusal are both structural (a pure plan over a hypothetical wiring, refused before
  any write), which is the sanctioned shape, not a second wave of validation over what an actor
  meant.
- **`version` bump.** Correctly absent. 091 added `group` without one; nothing reads the stamp.

---

## Verdict

**PARTIAL.** The design is sound, matches every locked item of `02_triage.md` and every "Fixed by
the maintainer" item of `01_brainstorm.md`, satisfies dev_flow step 2's section set, and carries no
migration or compat language. Three items must be edited before implementation:

1. **D11's cycle-edge rule** — as written it reddens zero edges, because `plan_passes` emits the
   culprit message once per cycle, not once per member (probed on three shapes). Manual-verification
   step 7 would fail. Edit 1.
2. **D6/D9's `rank_layout` contract and Arrange's scope** — the two decisions describe
   incompatible domains for the same function, and "every visible pass" makes Arrange mean
   different things at the root and in a group tab. Edit 2.
3. **D18's `wiring_if_renamed`** — the stated mechanism does not produce the post-rename wiring
   (probed: swap-only loses every explicit edge naming the old pass). No live hole, but the
   function would be named for something it does not do and would lack its `finally`. Edit 3.

Five further items are under-specified enough that a blind coder guesses: the port-state source
(D15/D11, edit 6), the root tab's key and label (D3, edit 5), the `graph_views` eviction and the
copilot-freeze bracket on the moved buttons (D2, edit 4), the port ordering rule (D1, edit 7), and
the ghost/box menus (D10, edit 8). Edits 9-12 are corrections to the docs and test lines, one of
which (edit 10) removes a claim of coverage the gate does not provide.

None of this is a should-not-land finding.
