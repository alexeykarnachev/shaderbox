# 092 pre-implementation review — verification design and blast radius

Anchors read end to end: `03_spec.md`, `02_triage.md`, all five `reviews/brainstorm_*.md`,
`dev_flow.md` "Feature flow" step 7, `conventions.md`'s four gate laws, `Makefile`,
`scripts/smoke.py`, and the tests + code named below. Every claim is a symbol the grep found or
a file I opened; no line numbers, per the repo's citation rule.

**Baseline, run once before judging:**
`uv run python -m pytest tests/test_pass_graph.py tests/test_graph_persistence.py -q` →
**49 passed in 0.54s**. Green before the feature.

---

## Part 1 — Blast radius, line by line of "Files touched"

Each row greps the symbol for READERS (not definitions) and says whether the spec accounts for
them.

### 1.1 `widgets/pass_list.py` — the caption and buttons move out

| Reader of `pass_list.draw` / the moved lines | Effect | Spec accounts for it |
|---|---|---|
| `tabs/document.py` (the sole app call site) | The spec names it. | YES |
| `tests/test_pass_verbs.py` — `test_*` around `pass_list.preview_cell` and `pass_list._draw_pass_tile`, both driving `_imgui_frame(lambda: pass_list.draw(app, document_id))` | These monkeypatch `preview_cell` / `_draw_pass_tile` and call `draw` directly. Removing the caption and buttons does not break them (they assert on chips and tile arguments), but they will now exercise a `draw` that no longer draws the verbs — so **nothing in the suite drives the moved `add pass` / `import...` buttons any more**. | NO — not named, and the spec's "Files touched" does not list `tests/test_pass_verbs.py` for this half |
| `tests/test_ui_prose_budget.py::test_the_walk_finds_the_known_call_sites` asserts `"shaderbox/widgets/pass_list.py" in modules` | Survives: `_draw_pass_tile` keeps `imgui.set_tooltip("Pass settings")` and `preview_cell(footer=…)`, both scored. **Verified by reading the file** — the module stays in the walk. | Not named, but harmless. Worth one sentence so the implementer does not strip the tooltip too. |
| `_OVER_BUDGET` / `_UNMEASURABLE` entry `("shaderbox/widgets/pass_list.py", "_draw_pass_tile")` | Unaffected — `_draw_pass_tile` is not what moves. | OK |
| `scripts/smoke.py` | Does not call `pass_list.draw`; it drives the whole frame. Unaffected by the move itself. | OK |

**Also unaccounted:** the two `standard_button` labels `"add pass"` and `"import..."` are MEASURED
prose-budget sites today, keyed by `(module, function)` = `("shaderbox/widgets/pass_list.py",
"draw")`. Moving them to `tabs/document.py` re-keys them. They are not in `_OVER_BUDGET` or
`_UNMEASURABLE`, so the move is silent — but the spec should say the move is checked, because
`tabs/document.py` is already asserted present in the walk.

### 1.2 `set_pass_group` becoming a wrapper over `set_pass_groups`

Readers of `set_pass_group` (grep, not definition):

| Reader | What it pins | Spec accounts |
|---|---|---|
| `ProjectSession` capability table (`pass_set_group=self.set_pass_group`) → the copilot's `set_pass` | The copilot's whole group path. | Partly — D14 says "stays for the modal and the copilot" |
| `App` (the pass-settings modal's group commit, and the draft-apply path — two separate call sites) | Both must keep the same error-string contract. | NO — the spec names "the modal" as one site; there are **two** in `app.py` |
| `widgets/pass_list.py::_draw_context_menu` (`Leave group`) | Moves into `pass_menu_items`. | YES |
| `scripts/smoke.py` at frame 42 (`set_pass_group(multi, grouped, "smoke_group")`) | The strip's split-run draw path. | NO — not named. It must keep working through the wrapper, and it is the one place a wrapper regression shows headlessly. |
| `tests/test_pass_verbs.py` (three asserts: `""` on success, non-empty on `"2bad"`, `""` on clearing) | The message contract. | NO |
| `tests/test_copilot_pass_tools.py::test_set_pass_group_lands_and_echoes` — asserts `"group name" in bad.error` **and** the literal table row `"glow: runs 1, target f2 x1, linear, group fx"` | **This is the message pin the prompt asked about.** `project_session.set_pass_group` returns the literal `"a group name starts with a letter and holds letters, digits and underscores"`. If the wrapper routes through `set_pass_groups` and that verb returns a differently-worded refusal (or D17's new `"a pass and a group cannot share a name"` for a case the old code accepted), this test goes red. | NO — the spec does not mention it |

There is a **third** independent validator the persistence review already named: `_GROUP_PATTERN`
on `PassEntry`, `PASS_NAME_RE.match(group)` in `set_pass_group`, and `PASS_NAME_RE.match(group)`
in `pass_import.plan_import` (its message pinned by `tests/test_pass_import.py` and
`tests/test_import_dialog.py`, both asserting `"group name"`). D17 adds a **fourth** rule (the
namespace collision) to only one of them. That is the "checker that quietly narrows its own
domain" shape: `set_pass_groups` will reject `bloom` when a pass is named `bloom`, and
`plan_import` will still accept it, producing exactly the duplicate root key case 18 of the
mutations review demonstrated. **The spec does not name `pass_import.plan_import` in Files
touched at all.**

### 1.3 A new field on `PassEntry`

| Reader | Effect | Spec accounts |
|---|---|---|
| `tests/test_graph_persistence.py::test_a_graph_round_trips_every_field` | Constructs `PassEntry(...)` with explicit kwargs and asserts `load_graph(path) == graph`. A defaulted `position=None` round-trips **vacuously** — the test's name claims "every field" and it would not notice a `position` that never persisted. This is a narrowed-domain test that the feature makes worse. | NO |
| `tests/test_pass_graph.py::test_the_spec_schema_round_trips` | Builds `PassGraph(**data)` from a literal dict with no `position` and asserts `PassGraph(**graph.model_dump()) == graph`. Passes either way. | NO |
| `tests/test_pass_graph.py::test_graph_edits_preserve_fields_they_do_not_name` | The funnel test. `with_target` / `with_output` / `with_group` all `model_copy`, so a `position` rides free — **but this test only asserts `iterations`**. It will stay green with a `with_positions` that resets `position` on every other entry. | NO |
| `tests/test_model_kwargs.py::test_no_call_site_names_a_field_its_model_dropped` + `test_the_walk_reaches_the_models_that_persist` (which names `PassEntry` explicitly) | A `PassEntry(position=…)` kwarg typo is caught. Free. | OK |
| Every `PassEntry(...)` construction in the codebase — I counted them: **all keyword-only**, none positional, in both `shaderbox/` and `tests/`. | A new field is safe from positional breakage. | Confirmed; the spec's silence is correct here |
| `model_dump()` comparisons — `tests/test_copilot_pass_tools.py::_graph(app, document_id)["passes"]["glow"]["group"]` indexes one key, does not compare whole dicts. `tests/test_canvas_presets.py` and `tests/test_tutorial_build.py` read `.iterations` / `.target`. | No whole-dict equality on `PassEntry.model_dump()` anywhere. Safe. | OK |
| `scripts/build_tutorial.py` via `tests/test_tutorial_build.py` (`PassEntry().iterations == _BUILD._DEFAULT_ITERATIONS`) | Unaffected. | OK |
| `load_graph`'s per-entry `drop_unknown` / `drop_invalid` over `_keyed_entry_fields()` | Enumerated from `PassGraph.model_fields`, so the salvage is inherited. **Confirmed by reading `load_graph`.** The spec's D6 claim is true. | YES |

### 1.4 `rename_pass` gaining a refusal (D18)

Callers, all of which now have a new failure mode:

| Caller | Spec accounts |
|---|---|
| `popups/pass_settings.py` (the gear's rename commit — pushes the error string) | Partly (D18 names "with the planner's message") |
| `app.py` (the second rename path) | NO — two call sites, not one |
| `tests/test_pass_verbs.py` — six renames, one asserting `"already exists"`, one asserting a non-empty error for `"no spaces"` | Any of these that happens to close a cycle would now fail. Reading them: none does (they rename in single- or two-pass fixtures with no back edge). Safe, but unverified by the spec. | NO |
| `tests/test_default_wiring.py` — renames `→ "df"` then `"df" → "field"`, in a fixture built specifically to exercise the **name rule** (`u_df` beside `df`) | This is the single highest-risk caller: D18's guard computes the post-rename wiring through the name rule, which is exactly what this fixture manipulates. A guard that mis-plans here turns a passing test red. | NO — `test_default_wiring.py` is not in Files touched |
| `tests/test_copilot_passes.py`, `tests/test_pass_editor_wiring.py`, `tests/test_pass_verbs.py` (the `_graph_renamed` position-carrying assert) | Each renames once. | NO |

### 1.5 `add_pass` gaining a refusal (D17: a pass named like a group)

| Caller | Risk |
|---|---|
| `copilot/tools/passes.py::add_pass` → `capabilities.add_pass` → `backend.add_pass` | A new refusal string the model reads verbatim. `tests/test_copilot_pass_tools.py` drives `backend.add_pass("", "glow", …)`. Low risk, but the copilot-design skill's rule is that a new refusal message is a per-tool fact. **Not named in the spec.** |
| `scripts/smoke.py` frame 20 (`add_pass(…, "smoke_pass")`) and frame 42 (`set_pass_group(multi, grouped, "smoke_group")`) | **These two collide in principle**: the smoke creates a group literally named `smoke_group` while a pass is named `smoke_pass` — different strings, so no collision today. But the smoke's own `add_pass` at frame 20 runs on `app.current_document_id` while frame 42 groups a *different* document. No live break; worth a line so the implementer does not "fix" the smoke by renaming. |
| `PASS_STUB` callers / `app.open_add_pass` | The modal's create mode (078 D5) draws a draft; a name that is a group must be refused **in the draft preview**, not only on commit, or the user types a name and the modal accepts it into a refusal. Spec does not say which. |

### 1.6 `UIAppState.passes_view`

- `UIAppState` carries `model_config = {"extra": "forbid"}` — **confirmed by reading `ui_models.py`**.
- `tests/test_persistence_completeness.py` drives `UIAppState.load` against nine corruptions
  including `("retired keys", '{"a_retired_key": true, …}')`. A new field inherits the battery
  through `load_model`'s per-key salvage. **No roster edit needed** (the roster is by MODULE:
  `ui_models.py` is already in `_STORES`). The spec does not claim otherwise; fine.
- **The gap:** `tests/test_channel_view.py` carries the precedent pair —
  `test_every_view_has_a_label` (`set(CHANNEL_VIEW_LABELS) == set(ChannelView)`) and
  `test_the_default_is_color_and_the_choice_persists` (save → load round trip). The spec adds
  `PassesView` + `PASSES_VIEW_LABELS` with **no equivalent test**. A label dict that misses a
  member draws an empty segment in `segmented_choice` and nothing catches it.

### 1.7 `theme.py` tokens

- The import-time asserts in `theme.py` are: `SELECT` vs accent primaries; `GROUP_TINTS` vs
  `_GROUP_TINT_EXCLUSIONS` (= accent primaries ∪ {STATE_OK, STATE_WARN, STATE_ERROR, STATE_INFO,
  SELECT, TAG, FAVS}); `GROUP_TINTS` internally distinct; `SELECT` vs the STATE hues.
  `tests/test_theme.py::test_group_tints_are_stable_and_collide_with_nothing` mirrors the same set.
- **The gap:** `COLOR.GRAPH_EDGE` is a new fixed hue that the canvas draws *adjacent to* group
  tints (a box's border) and *adjacent to* `STATE_ERROR` (a cycle edge). It is not in
  `_GROUP_TINT_EXCLUSIONS`, so `GRAPH_EDGE == GROUP_TINTS[2]` or `GRAPH_EDGE == STATE_ERROR`
  imports clean and ships. The theme's own invariant comment says a fixed hue may not equal
  "another fixed hue it shares spatial context with" — a wire beside a box border is exactly that.
- `GRAPH_GHOST_ALPHA` is an alpha, not a hue; no assert applies. `SIZE.GRAPH_*` are sizes; none
  of the theme asserts covers sizes. `GRAPH_ZOOM_MIN/MAX` and `MAX_GRAPH_COORD` live in D7/D6 —
  the spec puts the zoom clamp in `theme.py` implicitly (`[0.25, 2.5]` are written as tokens) but
  `MAX_GRAPH_COORD` in `pass_graph.py`; that split is unstated.

### 1.8 The button-tier allowlist

`tests/test_button_tiers.py` has three tests. Two matter:

- `test_no_site_hand_rolls_a_button_outside_the_primitives` — walks `imgui.*button*` calls
  across `shaderbox/**`. Every `invisible_button` in `widgets/pass_graph.py` lands here. **The
  spec's allowlist entry is required.**
- `test_every_listed_exception_still_exists` — "an allowlist that outlives its site is a rule
  nobody is following any more". **Ordering constraint the spec does not state:** adding
  `("widgets/pass_graph.py", "invisible_button")` to `_NOT_A_VERB` in a commit where the file does
  not yet exist (or has no `invisible_button` yet) turns this test RED. The allowlist entry and
  the canvas must land in the same commit, and the reason string must match the file's shape
  ("a hit rect, no label").
- The canvas also submits **port** hit rects (D8, W2) — the allowlist is keyed by `(module, call
  name)`, so one entry covers all of them. Correct as spec'd.

### 1.9 The prose-budget gate — the spec's claim is a no-op

The spec says: *"`tests/test_ui_prose_budget.py`: the canvas menu labels are within budget;
nothing to allow."* Reading the test:

- `_SCORED` = `_IMGUI_ROWS` + `_derived_rows()`. `_IMGUI_ROWS` is exactly four entries:
  `help_marker`, `set_tooltip`, `separator_text`, `text_colored`. `_derived_rows()` reflects over
  **`ui_primitives` functions only**.
- **`imgui.menu_item_simple` is not in the domain.** Grepped: the string `menu_item` does not
  appear in `test_ui_prose_budget.py` at all. Every canvas menu label (`Add pass`, `Import...`,
  `Fit`, `Arrange`, `Open`, `Dissolve`, `Group...`, `Leave group`, `Settings`, `Delete`) is
  **unmeasured**, today and after this feature.
- `ImDrawList.add_text` (the node name, the port label, the `N passes` badge, the `×N` badge) is
  likewise outside the domain.
- What IS scored on the canvas: any `text_tab_row` / `small_caption` / `standard_button` /
  `set_tooltip` the widget calls, plus whatever `tabs/document.py` gains.
- **The `strip | graph` toggle:** if the two option labels are read out of `PASSES_VIEW_LABELS`
  (the `CHANNEL_VIEW_LABELS` precedent), the argument to `segmented_choice` is a subscript
  expression the walk cannot resolve → the site becomes UNREADABLE →
  `test_every_unmeasurable_site_is_listed` **fails** unless `("shaderbox/tabs/document.py",
  "<the enclosing function>")` is added to `_UNMEASURABLE` with a reason. The existing precedent
  is the `("shaderbox/ui.py", …)` entry for `CHANNEL_VIEW_LABELS`. So "nothing to allow" is
  wrong in one direction and vacuous in the other.

### 1.10 `test_roadmap_shape.py`

- `test_every_feature_row_has_a_status_from_the_vocabulary` and
  `test_every_feature_row_points_at_a_spec_that_exists` — the 092 row must carry a status from
  `STATUSES` and a `` `ai_docs/features/092_graph_view/03_spec.md` `` pointer that resolves. It
  does today. Fine.
- `test_both_intel_rosters_name_every_module` is scoped to `shaderbox/intel/*.py` only —
  **`widgets/pass_graph.py` and `widgets/graph_state.py` are NOT gated by it.** D20's promise
  that `dev_flow.md`'s module map gains the two modules is therefore an ungated prose edit. Not a
  blocker; worth knowing that nothing fails if it is forgotten. (Listed under False trails.)
- `test_the_banner_stays_within_its_stated_budget` and the date stamp — the Active-context banner
  rewrite at step 9 must stay in budget. Standard.

### 1.11 Readers the spec does not touch at all but that the feature changes

1. **`Document.graph_errors` is refreshed ONLY inside `Document.render`** — the two write sites
   are the `resolved is None` early return and the `plan_for_output` line. `ui.py`'s render loop
   gates `document.render(...)` on `renders_this_frame(app, document_id)`, which is the 090 D11
   throttle. **A throttled document's `graph_errors` is stale**, so D11's cycle-edge cue can show
   a cycle that was just unwired, or miss one just wired, for as long as the throttle skips that
   document. `tests/test_document_graph.py::test_a_cycle_reports_per_pass_and_still_draws_the_output`
   is the only behavioural pin on the content of `graph_errors`, and it renders directly.
   **The spec does not name this.** Either the canvas computes its own `plan_passes` over
   `effective_wiring()` (pure, free, and correct every frame — the same call D12's `wiring_with`
   makes), or D11 has a visible lag nothing tests.
2. **D1's "the canvas compiles what it draws"** is a GL side effect from a draw function. Every
   other compile in the app runs inside `Document.render` / `_bring_chain_online` / the
   first-render sweep, all under the render plan. `sampler_names` is explicitly documented as
   "reads the program rather than `get_active_uniforms()`, which COMPILES a never-attempted pass
   (066 D1) — asking that here would compile the whole document on frame one." D1 does exactly
   that, from the draw, for the visible document. It is bounded, and the spec argues the bound —
   but it inverts the 066 D1 budget **and** does GL work for a document the render plan may have
   skipped this frame. Nothing in Files touched names `document.py`'s `sampler_names` docstring,
   which will then be stale prose (the "docs are living" rule).
3. **`pass_import.plan_import`** — D17's namespace rule has no branch there (§1.2).
4. **The import dialog's flatten line.** Triage S15 says the strip and the dialog are untouched
   *"except one line in the dialog"* saying the source's own groups are flattened. The spec has
   **no such line**: `popups/import_passes.py` is absent from Files touched and the flatten note
   is absent from Out of scope. A triage constraint has been dropped silently.
   `tests/test_import_dialog.py` is therefore also correctly absent — but the constraint is lost.
5. **`text_tab_row` keys its items by NAME** (`imgui.selectable(f"{name}##{id_}_{name}")`) and
   returns the clicked name. D3's row lists "the document's display name (the root) then every
   group name". A document display name is free text (spaces, empty, emoji) and can equal a group
   name — two rows with one id, and `clicked` is ambiguous. D17 does not cover the
   document-name/group-name collision, only pass/group.

---

## Part 2 — Verification design, per locked decision that states a guarantee

Format: **invariant** / **falsifier** (the input or mutation that must turn a named test red) /
**the line that READS the mechanism** / **does Files touched contain a test that fails for
exactly that reason?**

### D1 — ports from the program, edges from the wiring

- **Invariant:** a node draws one input port per name in `sampler_names(pass)` (plus `prev`), and
  never a port for a stored `uniform_values` row whose sampler the program no longer declares.
- **Falsifier:** build a pass, compile it, remove the `sampler2D` from its source, `release_program`,
  re-compile; the stored row survives (`_reads_of` proves this) — assert the port list is
  `sampler_names(...)` and does NOT contain the dead uniform. Mutation: make the port list read
  `uniform_values` keys instead; the test must go red.
- **Reader:** the canvas's port loop in `widgets/pass_graph.py`. This is UI code, so the assert
  must be on a **pure function** that returns the port list, not on the draw.
- **Verdict: MISSING.** Files touched lists `group_boundary`, `bundle_output`, `rank_layout`,
  `wiring_with` as pure and tested — but the per-node port list is not named as a function at
  all. It exists only inside the draw, where nothing can assert it. This is the C7 hazard the
  wiring review called "the only way the drag can write something that does nothing", and the
  spec's own answer to it ("a stale port cannot be dropped on if it is never drawn") is
  unverifiable as designed.
- **D1's second half** ("the canvas compiles what it draws") has no named test either; it is a
  side effect of a draw. At minimum it needs a pure `passes_to_compile(document) -> list[str]`
  the canvas calls, so the *selection rule* is testable without a window.

### D5 — the boundary and the bundle output

- **Invariant (boundary):** `group_boundary` returns one input port per `(member, sampler)` whose
  source is outside the group or unfilled, and one output per member read from outside, plus the
  bundle output always.
- **Falsifier (inputs):** the non-convex shape from mutations case 13 — `a → g1 → mid → g2 → out`
  with `grp = {g1, g2}` — must yield inputs `[(g1,u_a,a), (g2,u_mid,mid)]` and outputs
  `[(mid,u_g1,g1), (out,u_g2,g2)]`. Mutation: drop the "or is unfilled" clause and the
  severed-chain case (mutations case 1) loses a port.
- **Falsifier (bundle output, the one the mutations review says BREAKS the rule):** delete the
  member that is the bundle output (case 2). The review demonstrated the contradiction: the
  bundle falls to `b_blur`, which nothing outside reads, so under a "read from outside" rule the
  box has zero output ports while its picture is `b_blur`'s. D5 resolves it ("plus the **bundle
  output** always, drawn hollow when nothing outside reads it") — **so the falsifier is: a group
  whose members nothing outside reads must still return exactly one output port, and it must be
  the bundle output.** Mutation: make the bundle output conditional on an outside reader; the
  test goes red.
- **Reader:** the box draw + `pick_pass` on a box click.
- **Verdict: PARTIAL.** Files touched names "`group_boundary` and `bundle_output` over the bloom
  shape, the non-convex shape, a generator box, a one-member group, a split group, a member whose
  sampler reads nothing." That covers the shapes. It does **not** name the terminal-box case
  (nothing outside reads any member) — which is the exact case the review's verdict called the
  rule break. Add it by name.
- **Second gap:** D5 says "clicking it picks the bundle output (`pick_pass`)". Mutations case 17
  proved `PassGraph(output="bloom")` constructs fine and `output_pass` returns `None` — a stale
  output rendering nothing. The invariant is *the box pick resolves to a member name before the
  write*. No test named. Falsifier: click a box, assert `graph.output in document.passes`.

### D6 — positions written only by placement, stripped on import

- **Invariant A (bounds):** `PassEntry(position=(nan, 0))`, `(inf, 0)`, `(1e30, 1e30)` all raise;
  a legal pair is accepted. **Files touched names it. GOOD** — and this is the one the
  persistence review measured as unbounded today.
- **Invariant B (salvage):** a corrupt `position` in `graph.json` costs that position and nothing
  else. **Named. GOOD** — and correctly routed through `load_graph`, not `load_model` (the
  persistence review's false trail).
- **Invariant C (no draw-time write):** *nothing writes `position` from a draw.* This is the
  conventions collision the persistence review flagged as CONFLICTS. **The falsifier is the one
  the spec does not have:** render N frames of the graph view over a document with every
  `position=None` and assert `graph.json`'s mtime (or the entries' `position`) is unchanged.
  Mutation: have the canvas call `set_pass_positions` for the unplaced set on first sight; the
  test must go red. **MISSING** — the spec has no such test, and `scripts/smoke.py` is the only
  place a draw loop runs.
- **Invariant D (every creator leaves `None`):** Files touched names `import_passes` only
  ("`import_passes` leaves positions `None`"). The spec's D6 text names three creators —
  `add_pass`, the copilot's tool, `import_passes`. **The copilot path is untested**:
  `capabilities.add_pass` → `session.add_pass` builds `PassEntry()`, which defaults to `None`, so
  it is correct by construction *today* — but the moment `add_pass` takes an optional position
  (which D6 does not ask for and D19 forbids), the copilot path is the one that regresses. State
  the invariant as "no `ProjectSession` entry point accepts a position except
  `set_pass_positions`" and test it by signature reflection, which is the structural-impossibility
  form the conventions prefer over a per-caller assert.
- **Invariant E (one save per Arrange):** see D14 below — same mechanism.
- **Reader for D:** `import_passes`'s `source_document.graph.passes.get(source_name,
  PassEntry()).model_copy(update={"group": group})` — the line that must gain `"position": None`.
  **Name this line in the spec**; mutations case 19 showed it copies the source's entry verbatim,
  which is where a foreign coordinate enters.

### D8 — the overlap chain

- **Invariant:** the background `invisible_button` carries `set_next_item_allow_overlap()`, and
  so does each node body; a missed level makes the canvas inert.
- **Falsifier:** the interaction review MEASURED it — `allow_overlap on bg = False` →
  `{'node_hov': False, 'node_act': False}`. Removing one call silently turns every node drag into
  a pan.
- **Reader:** the draw order in `widgets/pass_graph.py`.
- **Verdict: UNVERIFIABLE headlessly, and the spec does not say so.** No test can assert hit
  priority without synthetic mouse input, which this box does not have (imgui-ui §0; no
  `xdotool`). What the smoke CAN prove: the canvas draws N frames without an assert, a released
  texture binding, or a popup-state crash. What it CANNOT prove: that a click reaches the node
  rather than the background. **This must be a manual item with a single failure reason** (it is
  currently inside manual item 2, mixed with pan and Fit).

### D12 — the cycle refusal and the media refusal

- **Invariant (cycle):** `wiring_with(wiring, consumer, sampler, producer)` + `plan_passes` → any
  error refuses the drop before `set_sampler_source` is called.
- **Falsifier:** the wiring review's measured case — `a→b→c`, drop `c → a.u_c` returns
  `('a', 'passes form a cycle: a -> c -> b -> a')` plus two victims; the diamond drop
  `a → c.u_a2` and the unrelated `loner → a.u_l` both return `[]`. **All three must be in the
  test**, because a guard that refuses everything passes a one-case test. The refusal test must
  be on `errors != []`, not on "the culprit is an endpoint" — the review proved the culprit need
  not be an endpoint.
- **Reader:** the drop handler. **This is the mutate-the-wiring law's exact shape:** a test that
  calls `wiring_with` + `plan_passes` directly proves the pure function, and says nothing about
  whether the drop handler calls it. The conventions' 086 precedent is literally this — "a
  loader's refusal was required by the spec, claimed by its verification step, and exercised only
  through the helper it called, so gutting it stayed green."
- **Verdict: PARTIAL.** Files touched names "`wiring_with` + the cycle refusal" in
  `test_pass_graph.py` — the pure half. **Nothing drives the consumer.** Since the drop handler
  is draw code, the honest fix is to put the decision in a pure function the handler's only job
  is to call — e.g. `pass_graph.refuse_drop(wiring, consumer, sampler, producer) -> str` — and
  test *that*, then assert by AST/grep that the widget has no `set_sampler_source` call outside
  the branch that checked it. Say which.
- **Invariant (media):** a drop on a media-bound port is refused. **Falsifier:** a sampler whose
  `uniform_values` holds a texture; the drop must not call `set_sampler_source` (which would run
  `try_to_release(values.get(uniform))` and free the texture — the one irreversible write in
  mutations case 20's table). **No test named. MISSING**, and this is the highest-consequence
  refusal in the feature: it is the only canvas gesture that destroys user data.

### D13 — one save per drag

- **Invariant:** a node drag writes `position` exactly once, on release, through
  `set_pass_positions`, and that is one `save_ui_document`.
- **Falsifier:** count saves. The measurable form: monkeypatch `session.save_ui_document` (or
  count `graph.json` writes) across a synthetic drag of K frames and assert the count is 1, not K.
  Mutation: write the position per frame; the count becomes K.
- **Reader:** the release branch in the drag handler.
- **Verdict: PARTIAL.** Files touched says "`set_pass_positions` saves once and survives every
  other verb" — that tests the VERB (one call → one save), which is not the guarantee. The
  guarantee is about the *gesture*, and the gesture is draw code. A verb-level save count passes
  whether the handler calls it once or sixty times. **This is the mutate-the-wiring law again.**
  The testable surface: make the drag state (`NodeDrag` in `graph_state.py`) a pure state machine
  with `begin/update/commit`, where `commit` is the only thing that returns positions, and assert
  `update` returns nothing to write.

### D14 — one save per Group

- **Invariant:** `set_pass_groups(document_id, names, group)` validates once and saves once for N
  passes.
- **Falsifier:** group four passes, assert one `save_ui_document`. Mutation: loop
  `set_pass_group` per name → four saves.
- **Verdict: NAMED and CORRECT** ("`set_pass_groups` saves once and rejects D17"). This one is
  well-formed: the verb *is* the guarantee, unlike D13.
- **Sub-gap:** "Selecting a box and grouping it with others rewrites its members to the new label
  (flat labels, so the old group dissolves into the new one)" — a behaviour with no named test,
  and the one the scenarios review called GAP 6b (no reparent verb). One case in the same test.

### D17 — one namespace

- **Invariant:** `set_pass_groups` rejects a group named like an existing pass; `add_pass` and
  `rename_pass` reject a pass named like an existing group; message
  `"a pass and a group cannot share a name"`.
- **Falsifier (the domain, not one instance):** the checker's domain is **four** entry points —
  `set_pass_groups`, `add_pass`, `rename_pass`, and `pass_import.plan_import`. A test that drives
  three of four is the narrowing-checker family. Mutations case 18 demonstrated the failure with
  `passes {bloom, b1(group bloom), b2(group bloom)}` → root keys `['bloom', 'bloom']`.
- **Verdict: PARTIAL and NARROWED.** Files touched names `set_pass_groups` rejecting D17 and
  `rename_pass` refusing the cycle — it does **not** name `add_pass` rejecting D17 (which D17's
  own text requires), and `plan_import` is not in the design at all. Enumerate the entry points
  from a single `is_legal_group` / `namespace_error` funnel and assert every one of them calls it,
  which is the single-funnel law's form.

### D18 — a rename plans before it moves

- **Invariant:** `rename_pass` refuses when `plan_passes(Document.wiring_if_renamed(old, new))`
  reports a cycle, with the planner's message, and the file is not moved.
- **Falsifier:** mutations case 9's shape — an `AutoSource` `u_scene` on a member, rename some
  pass **to** `scene`, closing a loop. The test must assert both the refusal string AND that
  `paths.pass_shader_for(old)` still exists on disk (the rename is transactional: the file
  `.replace()` happens *before* `rename_pass_sources`, so a guard added after the move leaves a
  half-renamed document).
- **Reader:** the new early-return in `rename_pass`, before `render_pass = document.passes.pop(old)`.
- **Verdict: NAMED, and the spec even names the mutation** ("a mutation test: with the guard
  removed the rename succeeds and the plan reports the cycle"). **This is the best-formed
  verification item in the spec.** Two additions: (a) the on-disk assert above, because "refuses"
  and "does not move the file" are two reasons the same test could pass; (b) `wiring_if_renamed`
  "swaps the pass under the new key, reads `effective_wiring()`, and restores" — a mutate-and-restore
  on the LIVE `Document`. The conventions' law "a mutation test verifies its own restore before
  anything else runs" applies to production code here: an exception between swap and restore leaves
  the document keyed under the new name with the old file. Make it a copy, or assert the restore.

### D9, D10, D11 — the UI-only guarantees

**What `scripts/smoke.py` can prove headlessly:** that the canvas draws N frames without raising,
without an imgui assert (a mismatched `channels_split`/`merge`, a `begin_child` without `end_child`,
a `push_font` without a pop, a `SetCursorPos` assert), without binding a released texture, and
without leaving a popup open across frames; and that a state flag the frames set has the value the
frames expect (`app.app_state.passes_view`, `app.graph_views[id].scope`, `.fitted`, and the
`position` values after an Arrange).

**What it cannot prove** (imgui-ui §0 — no window manager, no synthetic input on this box): that a
click lands on the node rather than the background (D8), that the wheel zoom is cursor-anchored
(D9), that a wire's bezier goes where it looks like it goes (D12), that anything is legible, that
a colour reads as distinct (D11), or that a context menu opened for the right item (D8's
`begin_popup_context_item(None)` rule). A smoke assertion on any of those would read as coverage
and prove nothing — the persistence review's own warning ("no layout assertion in the smoke test
beyond 'it did not crash'"), which stands, with the exception that a **state flag** the frames
set programmatically IS a real assert.

**The exact smoke stretch the spec should add** — in the style of the existing `smoke_group`
stretch at frame 42, placed after it and before the frame-48 return to the canary. Frames 42-47
already belong to the multi-pass document, so the graph stretch reuses that selection:

```python
                # The graph canvas (092): the second view of the same passes. Drawn on the
                # MULTI-pass document selected at frame 42, so the canvas has nodes, a box
                # (`smoke_group`), wires and a group tab to draw rather than one lone node.
                # None of it can be screenshotted on this box, so the frame loop executing
                # every branch IS the check -- plus the three state asserts below, which are
                # the parts a draw-time write or a lost scope would break silently.
                if frame_idx == 43:
                    app.app_state.passes_view = PassesView.GRAPH
                    # D6: opening the view must WRITE NOTHING. Captured before the first
                    # canvas frame; compared at 47. Falsifier: have the canvas call
                    # set_pass_positions for the unplaced set on first sight.
                    graph_before = json.dumps(
                        app.ui_documents[multi].document.graph.model_dump(), sort_keys=True
                    )
                if frame_idx == 45:
                    # D9: the one-shot fit ran on the first canvas frame at a nonzero size.
                    view = app.graph_view_for(multi)
                    assert view.fitted, (
                        "frame 45: the graph view never fitted -- the first-draw fit is "
                        "gated on a size the child never reports"
                    )
                    # D3: entering a group tab. The scope is revalidated every frame, so a
                    # scope no pass carries falls back to the root.
                    view.scope = "smoke_group"
                if frame_idx == 46:
                    view = app.graph_view_for(multi)
                    assert view.scope == "smoke_group", (
                        "frame 46: the group scope did not survive a frame -- the "
                        "revalidation dropped a live label"
                    )
                    # D3: a scope no pass carries must fall back to the root within a frame.
                    view.scope = "no_such_group"
                if frame_idx == 47:
                    view = app.graph_view_for(multi)
                    assert view.scope == "", (
                        "frame 47: a scope no pass carries survived -- the last member can "
                        "leave from inside the tab and the tab must close itself"
                    )
                    assert json.dumps(
                        app.ui_documents[multi].document.graph.model_dump(), sort_keys=True
                    ) == graph_before, (
                        "frame 47: drawing the canvas WROTE the graph (092 D6) -- a position "
                        "is written by a placement, never by a draw"
                    )
                    # D9: Arrange is a placement, so it DOES write -- and exactly once.
                    saves = _count_saves(app)
                    app.arrange_graph(multi)
                    assert _count_saves(app) == saves + 1, (
                        "frame 47: Arrange saved more than once -- D6 writes every position "
                        "through one set_pass_positions call"
                    )
                    assert all(
                        entry.position is not None
                        for entry in app.ui_documents[multi].document.graph.passes.values()
                    ), "frame 47: Arrange left a pass unplaced"
                    app.app_state.passes_view = PassesView.STRIP
```

`_count_saves` is a small helper beside `_arm_feedback_canary` that wraps
`app.session.save_ui_document` with a counter (or reads a counter the session already keeps). The
frame-47 stretch is the only place in the repo where D6's "a draw writes nothing" and D13/D14's
"one save per gesture" can be observed against the real loop rather than against the verb.

**Caveat the spec must state:** `make gates` reports a display-less smoke as **skipped**, which
is not a pass. On the dev box this entire stretch does not run. So the smoke is the *regression*
net for the maintainer's machine and CI, not the gate that proves the feature — the pure
functions are.

---

## Part 3 — The manual-verification list

### 3.1 The blocking finding: five of fourteen items are unrunnable

Items **4, 5, 10, 12, 13** all say "the bloom host" / "Bloom Chain imported into a host". I
checked: the bloom chain is `tests/fixtures/bloom_chain/` — a **test fixture**, not a shipped
example. `shaderbox/resources/document_examples/` holds six documents; five are single-pass
(`main.frag.glsl` only) and the sixth is Radiance Cascades (`cascade, composite, df, jfa, paint,
seed`). Grepping `bloom_chain` under `shaderbox/` returns nothing. **The maintainer cannot open
the bloom chain from the app.** Every item that depends on it is unrunnable as written.

Fix: either (a) each such item begins "copy `tests/fixtures/bloom_chain/` into
`projects/dev/documents/<new uuid>/` and open it" — one preamble sentence, stated once at the top
of the list; or (b) rewrite them against Radiance Cascades, which has the shapes: `seed/jfa/df` is
a groupable run, `paint` and `cascade` are the outside passes that become ghosts, `paint` has a
sampler that can close a loop with `composite`.

### 3.2 Item-by-item

| # | Fails for exactly one reason? | Observable without a WM? | Verdict |
|---|---|---|---|
| 1 | **No — six claims in one item** (node count, column count, bus routing, two `prev` loops, two badges, the accent border, and Fit) | Yes (the maintainer has a display) | REWRITE — split |
| 2 | **No — three gestures** (wheel anchoring, middle-drag pan, Fit) | Yes | REWRITE — split |
| 3 | **No — four effects of one click** plus a double-click | Yes | REWRITE — split; and it silently also asserts `set_output_pass` fired |
| 4 | No — box ports + port labelling + tab entry + two ghosts + ghost click + root tab label, **and the document is unreachable** | Blocked | REWRITE + preamble |
| 5 | Yes (one gesture, one outcome), **document unreachable** | Blocked | Preamble only |
| 6 | **No — two reasons**: the border could stay red because the fix did not recompile, or because the border does not read `compile_unit.errors`. Also asserts the stale mark. | Yes | REWRITE — split the stale mark out |
| 7 | Yes in shape, but the strip half ("the strip goes grey as before") is a **second** system. **And it is the one item that cannot be trusted**: `graph_errors` is refreshed only in `Document.render`, so on a throttled document the red edges lag. | Yes | REWRITE — add "with document throttling off (Settings)" |
| 8 | **No — two documents** (a single-pass Fire, and Media Input) | Yes | REWRITE — split |
| 9 | **No — four claims** (drag persists, survives a document switch, survives a restart, snaps with a guide, Arrange undoes it) | Yes | REWRITE — split; the restart half is the only one that proves D6 persisted |
| 10 | **No — two opposite outcomes in one item**, and "nothing changes on disk" is the load-bearing half with no stated way to check it | Yes | REWRITE — say `git diff projects/dev` |
| 11 | Yes | Yes | KEEP |
| 12 | **No — five claims** (group, tab, ghosts, dissolve, two refusals) | Partly blocked | REWRITE — split |
| 13 | Truncated in the spec ("so that `fx_bright.u_bright2`... (a rename that closes a loop)") — **the item does not say what to type** | Blocked | REWRITE — the sentence is unfinished |
| 14 | Yes, and it is the most valuable item in the list (the one irreversible write) | Yes | KEEP — promote it |

### 3.3 Cases the reviews' lists imply and the spec's manual list lacks

From the mutations report's 20 cases, the scenarios report's 19, and the wiring report's 15-row
connection table, these are the ones with no manual item and no named test:

1. **Delete the member that IS the bundle output** (mutations 2 — the review's own verdict, "the
   rule set is incomplete for deletion"). The box's picture jumps and the output port must still
   exist, drawn hollow.
2. **Delete the document output while it is inside a box** (mutations 3): `_graph_without` picks
   `next(iter(kept))` — insertion order, not topological — so the accent border jumps out of the
   box onto an unrelated node.
3. **A member with no wires at all** (mutations 4): the badge says `N passes`, the ports account
   for N−1. The one case that makes "a box is its boundary edges" incomplete as a description.
4. **A one-member group** (mutations 6): a node wearing a box's chrome. D5 draws it; nobody has
   looked at it.
5. **The last member leaves while you are inside its tab** (mutations 7): D3 promises the
   fallback; the smoke stretch above is the only place it is checked, and the maintainer should
   see it once.
6. **A rename that severs a name-rule edge inside a box** (mutations 8): the box GAINS an input
   port because the reader's sampler now reads nothing. D5 covers it; no item exercises it.
7. **A drop from a member onto a GHOST's port** (mutations 14 / wiring C8): D12 says ghosts are
   drop targets "like any other". This edits a pass the current tab does not contain. The
   maintainer should see it once and decide he likes it.
8. **A drop onto a never-compiled consumer** (wiring C6): the good path — an explicit
   `PassSource` is honoured before any compile — but only if the port exists to drop on, which
   D1 says it does not. So the canvas cannot wire a cold pass. That is the design; confirm it
   reads as intended rather than as a bug.
9. **`Add pass` from the canvas menu** (scenarios 8): `PASS_STUB` declares no sampler, so the new
   node has zero input ports and the canvas cannot build a chain. The single largest framing
   consequence of the feature and no manual item shows it.
10. **The copilot's turn over an open canvas** (scenarios 16 / wiring S14): `begin_disabled`
    freezes the canvas while the draw list still paints live pictures.
11. **A cycle entirely INSIDE a box** (wiring B4): red on the box border at the root, red edges
    only after entering the tab.
12. **A group split across the DAG** (mutations 5 / triage D9): one box at the bounding box,
    swallowing non-members. Arrange pulls them together. The only D9 decision with a visible cost.

---

## Edits to the spec — paste-ready

### E1. Replace the `tests/test_pass_graph.py` bullet in "Files touched"

```
- `tests/test_pass_graph.py`:
  - `test_rank_layout_puts_producers_left_of_consumers` — every edge's producer has a
    strictly smaller x. Falsifier: rank by insertion order; the bloom chain goes red.
  - `test_rank_layout_is_deterministic_over_dict_order` — two shuffled wirings give the
    same dict. Falsifier: iterate an unsorted set inside the rank walk.
  - `test_rank_layout_places_only_the_unplaced` — a wiring where two of five carry a
    position: the returned dict has exactly the other three keys. Falsifier: return a
    position for every name, which would silently overwrite a drag on the next Arrange.
  - `test_rank_layout_keeps_cycle_members` — the two-cycle shape from
    `test_a_two_pass_cycle_is_an_error_per_pass_and_does_not_hang`: both members are
    placed at rank 0 rather than dropped. Falsifier: place only `plan.order`.
  - `test_rank_layout_puts_group_members_adjacent` — within a rank, members of one group
    are contiguous. Falsifier: sort a rank by strip order alone.
  - `test_group_boundary_over_the_bloom_shape` / `_over_the_non_convex_shape` /
    `_over_a_generator_box` / `_over_a_one_member_group` / `_over_a_split_group` /
    `_over_a_member_whose_sampler_reads_nothing` — the six shapes, each asserting the
    full (inputs, outputs) pair. Falsifier for the last: drop the "or is unfilled"
    clause and the severed-chain shape (mutations case 1) loses an input port.
  - `test_a_terminal_box_still_has_its_bundle_output_port` — a group nothing outside
    reads: exactly one output port, and it is `bundle_output(...)`. Falsifier: make the
    bundle output conditional on an outside reader — the case the mutations review
    called the rule break (its case 2).
  - `test_bundle_output_follows_its_four_branches_in_order` — the document output when a
    member; else the first member read from outside; else the last member no member
    reads; else the last member. Four wirings, one per branch. Falsifier: reorder two
    branches and the split-group shape answers a different member.
  - `test_the_port_list_is_the_programs_samplers_not_the_stored_rows` — D1, driven
    through the pure `node_ports(...)` the canvas calls (see the new bullet in
    `pass_graph.py`). A pass with a stored `u_gone` row the program no longer declares
    yields no `u_gone` port. Falsifier: build the port list from `uniform_values` keys.
  - `test_wiring_with_refuses_every_cycle_and_allows_every_legal_drop` — three measured
    cases from the wiring review: `a->b->c` with `c -> a.u_c` REFUSED (message names the
    trail), the diamond `a -> c.u_a2` ALLOWED, the unrelated `loner -> a.u_l` ALLOWED,
    and a self-read ALLOWED (feedback). Falsifier: refuse only when the culprit is one
    of the drop's two endpoints — the diamond stays green and the real case regresses,
    because `plan_passes` reports victims as well as culprits.
  - `test_a_position_is_bounded_on_the_model` — NaN, +/-inf, 1e30 and -1e30 each raise;
    `(12.0, -34.5)` is accepted. Falsifier: type the field `tuple[float, float] | None`
    with no `Field`, which is what the persistence review measured as unbounded today.
```

### E2. Add a `pass_graph.py` bullet for the port list (D1's testable surface)

```
- `shaderbox/pass_graph.py`: ... plus `node_ports(declared, wiring_row, name) -> list[Port]`
  — the per-node port list as a PURE function over `sampler_names(pass)` and the pass's
  wiring row, so D1's "a port is never drawn for a sampler the program no longer
  declares" is asserted without a window. The canvas's only job is to call it; a port
  built inside the draw is a rule nothing can check.
```

### E3. Replace the `tests/test_pass_verbs.py` bullet

```
- `tests/test_pass_verbs.py`:
  - `test_set_pass_positions_saves_once_for_the_whole_set` — four positions, one
    `save_ui_document`. Falsifier: loop a single-position verb; the count becomes four.
  - `test_a_position_survives_every_other_pass_verb` — set a position, then
    `set_pass_target`, `set_pass_iterations`, `set_pass_group`, `set_output_pass` and a
    rename; the position is unchanged after each. Falsifier: rebuild the entry from
    `PassEntry()` in any of them (the `with_target` regression
    `test_graph_edits_preserve_fields_they_do_not_name` was written for).
  - `test_set_pass_groups_saves_once_and_rejects_a_pass_name` — four passes into one
    group: one save; a group named like an existing pass is refused with
    "a pass and a group cannot share a name" and NOTHING is written (assert every
    entry's group is unchanged). Falsifier: validate per pass inside the loop, which
    leaves a partial write behind the refusal.
  - `test_add_pass_refuses_a_name_a_group_already_carries` — D17's other half, which the
    spec's own D17 requires and its test list omits.
  - `test_every_group_writing_entry_point_shares_one_validator` — enumerate
    `set_pass_groups`, `set_pass_group`, `add_pass`, `rename_pass` and
    `pass_import.plan_import`; each must refuse the same collision. Falsifier: add the
    D17 check to `set_pass_groups` alone (which is what the spec currently describes) —
    `plan_import` then still imports a bundle under a group name a pass already carries,
    and the root draws two entities with one name (mutations case 18).
  - `test_rename_pass_refuses_the_cycle_it_would_create` — mutations case 9's shape: an
    `AutoSource` `u_scene` on a member, rename a pass TO `scene`. Assert (a) the refusal
    carries the planner's cycle message, and (b) `paths.pass_shader_for(old)` still
    exists and `pass_shader_for(new)` does not — because "refused" and "did not move the
    file" are two reasons one assert could pass, and `rename_pass` moves the file before
    it rewrites the sources. The mutation: delete the guard; the rename succeeds and
    `graph_errors` reports the cycle. Applied at `ProjectSession.rename_pass`, the site
    that computes the condition, not at `wiring_if_renamed`.
  - `test_wiring_if_renamed_leaves_the_document_as_it_found_it` — the swap/read/restore
    is a mutation of the LIVE document: assert `document.passes` keys and every
    `uniform_values` identity are unchanged after the call, and after a call that raises.
    Falsifier: drop the restore.
  - `test_import_passes_leaves_every_position_none` — import a source whose entries carry
    positions; every copied entry's position is `None`. Falsifier: the current line,
    `source_document.graph.passes.get(source_name, PassEntry()).model_copy(update={"group": group})`,
    which copies the source's coordinates verbatim (mutations case 19).
  - `test_no_session_verb_but_one_accepts_a_position` — reflect over `ProjectSession`'s
    public methods; only `set_pass_positions` names a position parameter. D19's
    structural form: the copilot cannot be handed a coordinate it would synthesize.
```

### E4. Add the missing test files to "Files touched"

```
- `tests/test_ui_regions.py` (or beside `tests/test_channel_view.py`):
  `test_every_passes_view_has_a_label` (`set(PASSES_VIEW_LABELS) == set(PassesView)`),
  `test_every_passes_view_label_is_within_the_control_budget` (<= 2 words), and
  `test_the_default_is_the_strip_and_the_choice_persists` (save -> load). The
  `ChannelView` trio, which is the precedent this field copies.
- `tests/test_theme.py`: `COLOR.GRAPH_EDGE` joins `_GROUP_TINT_EXCLUSIONS` in `theme.py`
  and the mirrored set in `test_group_tints_are_stable_and_collide_with_nothing`. A wire
  is drawn against a box border and beside a `STATE_ERROR` edge, so it shares spatial
  context with both. Falsifier: set `GRAPH_EDGE = COLOR.GROUP_TINTS[2]` — today that
  imports clean.
- `tests/test_pass_verbs.py` (the strip half): the two `pass_list.draw` tests keep
  driving a `draw` that no longer submits the caption or the buttons; confirm they still
  assert what they name, and that `_draw_pass_tile`'s `set_tooltip("Pass settings")`
  stays, since `test_ui_prose_budget.py::test_the_walk_finds_the_known_call_sites`
  requires `widgets/pass_list.py` to remain in the walk.
- `tests/test_copilot_pass_tools.py`: `test_set_pass_group_lands_and_echoes` pins both
  the literal table row and `"group name" in bad.error`. Re-run it after
  `set_pass_group` becomes a wrapper -- the wrapper must return the SAME string for an
  illegal pattern and the new string only for the namespace collision.
- `tests/test_default_wiring.py`: two renames run through the name-rule fixture
  (`u_df` beside `df`). D18's guard plans over exactly that rule, so this file is the
  one most likely to go red from a correct-looking guard.
```

### E5. Replace the prose-budget bullet

```
- `tests/test_ui_prose_budget.py`: the `strip | graph` toggle reads its labels from
  `PASSES_VIEW_LABELS`, which the walk cannot resolve, so `tabs/document.py`'s caption
  function joins `_UNMEASURABLE` with the reason -- the `("shaderbox/ui.py", ...)` entry
  for `CHANNEL_VIEW_LABELS` is the precedent. Note for the record: the canvas's MENU
  labels and its draw-list text are outside this gate's domain entirely -- `_IMGUI_ROWS`
  covers `help_marker` / `set_tooltip` / `separator_text` / `text_colored` and nothing
  else, and `_derived_rows()` reflects only over `ui_primitives`. `imgui.menu_item_simple`
  and `ImDrawList.add_text` are unscored, so "within budget" for the canvas menu is an
  author's promise, not a gate.
```

### E6. Add the ordering constraint to the button-tier bullet

```
- `tests/test_button_tiers.py`: `("widgets/pass_graph.py", "invisible_button")` joins
  `_NOT_A_VERB` with "the canvas, node and port hit rects -- a hit rect, no label". It
  must land in the SAME commit as the widget: `test_every_listed_exception_still_exists`
  fails on an allowlist entry whose site does not yet exist.
```

### E7. Add the D11 freshness decision to the spec's D11

```
The cue reads `plan_passes(document.effective_wiring())[1]`, computed by the canvas, not
`Document.graph_errors` -- which is refreshed only inside `Document.render` and is
therefore stale for a document the 090 D11 throttle skipped this frame. The planner is
pure and the largest document is six passes, so recomputing costs nothing and the cue
cannot lag the wiring.
```

### E8. Add the dropped triage constraint

Either restore S15 to Files touched:

```
- `shaderbox/popups/import_passes.py`: one line saying the source's own groups are
  flattened (triage S15; 091 drops inner labels silently today).
```

or move it to Out of scope with a trigger. Silently dropping a triage constraint is the
failure mode `dev_flow.md` step 4 exists to catch.

### E9. The smoke stretch

Add it to the `scripts/smoke.py` bullet verbatim from Part 2's code block above, plus:

```
  and a `_count_saves(app)` helper beside `_arm_feedback_canary`. The stretch's THREE
  asserts are the parts no pure test can reach: the first-draw fit ran, a dead scope
  falls back to the root within one frame, and drawing the canvas wrote nothing to
  `graph.json`. `make gates` reports a display-less smoke as SKIPPED, which is not a
  pass -- on the dev box this stretch does not run, so it is the regression net for the
  maintainer's machine, not the gate that proves D6.
```

### E10. The rewritten manual-verification list

```
## Manual verification (the maintainer's, no window manager here)

Preamble: items marked (bloom) need the bloom chain, which is a TEST FIXTURE and not a
shipped example. Copy `tests/fixtures/bloom_chain/` into a new
`projects/dev/documents/<uuid>/` and open it once; every (bloom) item then runs against
it. Items 7 and 10 need document throttling OFF (Settings > Throttle documents), because
a throttled document does not re-plan and the cycle cue would lag.

**W1 -- the read-only canvas**

1. Radiance Cascades, `graph`: SIX nodes are drawn, one per pass.
2. The same view: `paint`'s two long wires ride the bus under the row rather than
   crossing the nodes between them.
3. The same view: `jfa` and `cascade` each draw a self-loop into their `prev` port, and
   no other node does.
4. The same view: `jfa` carries `x12` and `cascade` carries `x6`; no other node carries
   a badge.
5. The same view: `composite` alone carries the accent border.
6. Fit from the canvas menu: every node is inside the panel and none is clipped.
7. Wheel over `df`: the point of the picture under the cursor stays under the cursor
   through a zoom in and back out.
8. Middle-drag on empty canvas: the whole picture translates and nothing is selected.
9. Click `seed`: the viewer switches to `seed` and the accent border moves to it.
10. The same click: `seed`'s shader tab is at the front of the editor's tab row, and the
    editor does NOT take keyboard focus (type a letter -- it does not land in the buffer).
11. Double-click `seed`: the editor takes keyboard focus (the same letter lands).
12. After clicking `seed`: every node the output does not need is dimmed, and so are its
    wires.
13. Break `jfa`'s shader (delete a semicolon): `jfa`'s border turns red.
14. With `jfa` broken: `jfa`'s picture still shows its last good frame and carries the
    stale mark.
15. Fix `jfa`: the border returns to normal within a second.
16. On the Uniforms tab, point `paint`'s sampler at `composite`: on the canvas, the two
    wires of the loop turn red and no NODE turns red.
17. Unwire it: both wires return to normal.
18. `strip | graph` on Fire (single pass): one node, no input ports.
19. `strip | graph` on Media Input: one node with two square media ports and no wires.
20. (bloom) The root shows the `bloom` box with the badge `4 passes`.
21. (bloom) The box's input ports are three, one per member sampler reading `scene`, each
    labelled `member.u_scene`.
22. (bloom) The box has exactly one output port, into `final`.
23. (bloom) Double-click the box: the bloom tab draws its four members as solid nodes.
24. (bloom) In the bloom tab: `scene` is a dashed dim ghost on the LEFT and `final` a
    dashed dim ghost on the RIGHT.
25. (bloom) Click the `scene` ghost: the scope returns to the root and `scene` is
    selected.
26. (bloom) The root tab is labelled with the document's own name.
27. (bloom) From the bloom tab's node menu, `Leave group` on all four members: the tab
    closes itself and the root shows four plain nodes.
28. (bloom) Group three of them again, then delete the member the box's picture comes
    from: the box keeps exactly one output port, drawn hollow, and its picture changes to
    another member rather than going blank. (mutations case 2 -- the one case the review
    called a rule break.)
29. (bloom) Put a group label on a pass that reads nothing and is read by nothing: the
    badge counts it, no port appears for it, and the group tab shows it as a lone node.
    (mutations case 4.)
30. Group a single pass: the box draws, it has that pass's own ports, and its tab holds
    one node. (mutations case 6 -- confirm this reads as intended and not as a bug.)
31. Delete the document output while it is inside a box: the accent border moves to a
    root-level node chosen by insertion order, not by the graph. Confirm the jump is
    tolerable. (mutations case 3.)
32. Label two passes on opposite sides of the chain with the same group name: ONE box
    draws at their bounding box and encloses non-members visually; Arrange pulls the
    members together. (mutations case 5 / triage D9.)
33. Right-click a node: the menu carries Settings, Delete and (when grouped) Leave group,
    and nothing else.
34. Right-click empty canvas: the menu carries Add pass, Import..., Fit, Arrange -- NOT
    the node menu. (The `begin_popup_context_item(None)` rule; an explicit id opens the
    last node's menu from anywhere.)
35. `Add pass` from the canvas menu: the new node draws with ZERO input ports and a
    dashed border until it compiles. Type `uniform sampler2D u_seed;` into its shader and
    save: one port appears. (scenarios 8 -- the canvas is a view of wiring, not a
    construction surface.)
36. Start a copilot turn with the canvas open: every control is disabled and the node
    pictures keep updating live.

**W2 -- the interaction**

37. Drag `df` to the right and release: it stays.
38. Switch to another document and back: `df` is still where it was dropped.
39. Restart the app: `df` is still where it was dropped. (The only item that proves D6
    reached disk.)
40. `git diff projects/dev` after 37: exactly one `graph.json` changed, and only `df`'s
    position within it. (One save per drag.)
41. Drag a node until its left edge is within a few pixels of `cascade`'s: a guide line
    draws and the node snaps.
42. Arrange from the canvas menu: every node moves to the rank layout.
43. `git diff projects/dev` after 42: one `graph.json`, every entry carrying a position.
44. Drag from `paint`'s output dot onto `composite`'s `u_cascade` port: the wire moves and
    `composite`'s picture changes.
45. (bloom) Drag from a downstream pass's output onto an upstream pass's port so the drop
    would close a loop: a notification carries the planner's cycle message.
46. After 45: `git diff projects/dev` is empty -- the refusal wrote nothing.
47. Press `cascade`'s `u_df` port and release on empty canvas: the port goes hollow with a
    filled centre and `df`'s node dims.
48. Drop a wire from `df` back onto that port: it fills again.
49. Drop a wire on Media Input's `u_image` port: refused with "bound to media; unbind on
    the Uniforms tab", and the Uniforms tab still shows the bound texture. (The one
    canvas gesture that could destroy user data.)
50. Rubber-band `seed`, `jfa`, `df`; right-click; `Group...`; type `sdf`; Enter: one box
    appears.
51. `git diff projects/dev` after 50: one `graph.json`, three entries changed, one save.
52. Double-click the `sdf` box: its tab shows the three members with `paint` and
    `cascade` as ghosts.
53. Dissolve from the box menu: three plain nodes again.
54. Group a selection under a name an existing pass carries: refused with "a pass and a
    group cannot share a name", and nothing is written.
55. From the gear, rename a pass to an existing group's name: refused with the same
    message, and the pass's file is not renamed on disk.
56. (bloom) Rename a pass so the name rule closes a loop -- with `fx_bright` reading
    `u_bright2` by the name rule, rename `blur` to `bright2`: refused with the planner's
    cycle message, and `ls projects/dev/documents/<id>/passes/` still shows `blur`.
57. (bloom) In the bloom tab, drag a member's output onto a GHOST's port: the write lands
    on a pass the tab does not contain. Confirm this reads as intended. (mutations 14.)
58. (bloom) Rename a pass that a member reads BY THE NAME RULE: the edge disappears and
    the box gains an input port, with no other change. (mutations 8.)
59. Escape while the scope is a group tab: nothing happens (out of scope by design); the
    root tab is the way up.
```

---

## False trails (do not re-walk)

- **`test_both_intel_rosters_name_every_module` gating the new widgets.** It globs
  `shaderbox/intel/*.py` only. `widgets/pass_graph.py` and `widgets/graph_state.py` in
  `dev_flow.md`'s module map are an ungated prose edit — worth doing, but no test enforces it and
  none should be invented for it.
- **`extra='forbid'` protecting `graph.json` from a stray `position` key.** `PassGraph` and
  `PassEntry` set no `extra` (the persistence review measured `None`). `load_graph`'s per-entry
  `drop_unknown`/`drop_invalid` does the work. A test written against `load_model` instead would
  prune `passes` to `{}` because a pass NAME reads as an unknown field.
- **A `graph.json` version bump.** 091 added `group` without one; nothing in `shaderbox/` or
  `tests/` reads `.version`. The spec is right to say it does not bump.
- **Positional `PassEntry(...)` breakage from the new field.** I checked every construction site in
  `shaderbox/` and `tests/`: all keyword-only. Nothing to guard.
- **`test_persistence_completeness.py`'s roster needing a new entry for `passes_view`.** The
  roster is keyed by MODULE (`ui_models.py`), already present; a new field on `UIAppState`
  inherits the corruption battery with no edit.
- **A smoke assertion on layout quality, node placement or colour.** `make gates` reports a
  display-less smoke as skipped, so such a canary is skipped on the dev box while reading as
  coverage — the persistence review's own warning, and it stands. The three state asserts in E9
  are exempt because they read a flag the frames set, not a picture.
- **Non-convex groups needing a refusal or a cue.** Probed in two reviews; the boundary rule
  handles it with no special case and D11 explicitly gives it no cue. Settled.
- **A `group_passes` copilot tool / positions in the copilot's view.** D19 and the triage S7
  settle it; the copilot-design skill's tool-count rule is the reason. Do not re-open.
- **`Document.graph_errors` being the natural source for D11.** It looks right and it is stale
  under the throttle — see E7. This is the trap, not the answer.

---

## Verdict

**PARTIAL.** The spec's design is sound and its decision set is complete; what is under-built is
the verification half, and one manual-list item is unrunnable as written. Named items, in the
order they would bite:

1. **The manual list's five (bloom) items reference a test fixture the app cannot open** (items
   4, 5, 10, 12, 13). Blocking for the maintainer's own pass. Fix = E10's preamble.
2. **Manual item 13 is an unfinished sentence** — it does not say what to rename. Blocking.
3. **D12's media refusal has no test and no manual item in a runnable form.** It is the only
   canvas gesture that irreversibly destroys user data (`set_sampler_source` calls
   `try_to_release` on the bound texture). E3 + E10 item 49.
4. **D13's "one save per drag" is tested at the verb, not at the gesture** — the exact shape the
   conventions' "mutate the WIRING, not the renderer" law names, with two in-repo precedents (086).
   E3 + E9.
5. **D12's cycle refusal is tested at the pure function only**; nothing drives the consumer. Same
   law. E3.
6. **D17 narrows its own domain**: four entry points decide what a legal group name is; the spec
   adds the rule to one. E3's `test_every_group_writing_entry_point_shares_one_validator`.
7. **D6's "a position is written only by a placement" has no falsifier anywhere** — the one
   invariant whose violation is a convention breach, and the only place it can be observed is the
   smoke stretch in E9.
8. **`COLOR.GRAPH_EDGE` escapes the theme collision assert.** E4.
9. **The prose-budget claim is a no-op and the `_UNMEASURABLE` entry the toggle needs is missing** —
   the second half will turn the gate red at implementation time. E5.
10. **D11 reads a stale `graph_errors` under the document throttle.** E7.
11. **Triage S15 (the import dialog's flatten line) was dropped without a trigger.** E8.

None of these is a design defect and none warrants SHOULD-NOT-LAND. Items 1, 2 and 3 should be
fixed in the spec before implementation begins; 4-7 are the difference between a feature that is
verified and one that reads as verified.
