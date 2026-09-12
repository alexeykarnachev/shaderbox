# 091 post-implementation review — architecture and conventions

Commit under review: `4dc1423` ("091: import another document's passes as a group"), branch `dev`.
Read-only review; the working tree was restored byte-for-byte after every gate run (see F1's
method note).

## Verdict: FINDINGS (1 blocking, 2 minor, 1 cosmetic)

### F1 — BLOCKING. `make check` is RED at this commit, and the spec says it is green.

`shaderbox/popups/import_passes.py` (the import block) and `shaderbox/pass_import.py` +
`tests/test_import_dialog.py` (formatting) fail the repo's own lint gate:

```
$ uv run ruff check --no-fix shaderbox tests scripts
shaderbox/popups/import_passes.py:15:1: I001 [*] Import block is un-sorted or un-formatted
Found 1 error.
$ uv run ruff format --check shaderbox tests scripts
Would reformat: shaderbox/pass_import.py
Would reformat: tests/test_import_dialog.py
```

`make gates` therefore stops at the first stage:

```
$ make gates > /tmp/gates.log 2>&1; echo $?
2
== gates: FAILED at check (exit 2); test and smoke not run ==
```

All three are 091's own new files; the rest of the tree is clean (307 files already formatted,
0 ruff errors elsewhere). The specific deviations: `import_passes.py` has
`from shaderbox.pass_import` and `from shaderbox.project_session` placed before
`from shaderbox.document`; `pass_import.py`'s `taken` ternary and two `test_import_dialog.py`
comprehensions/calls are wrapped the way a human wraps them, not the way `ruff format` does.

The compounding half is the claim. `ai_docs/features/091_presets/01_spec.md`, the new
`## Implementation notes` section, states:

> Landed 2026-09-12; `make gates` green (check, test, smoke), exit code read unpiped.

That is the repo's most-named failure class, written into a committed doc where the next reader
takes it as established (`conventions.md ## Design decisions`: "A gate is checked by its EXIT
CODE" — the bullet whose own history is feature 064 reporting `make check` green while it was
red). `make test` IS green at this commit (2262 passed, 4 skipped, exit 0), so the claim is
half-true, which is what makes it expensive.

**Fix:** `uv run ruff check --fix shaderbox tests scripts && uv run ruff format shaderbox tests scripts`,
re-run `make gates` unpiped, and correct the spec sentence to what the run actually reported.
The three files' content is unaffected — this is whitespace and import order only.

*Method note:* ruff's pre-commit hooks write to the tree. After each `make check` / `make gates`
run I restored every touched file with `git show 4dc1423:<path> > <path>` and verified
`git diff --stat` empty and `git status --short` clean — per `conventions.md`'s mutation rule
(restore, then verify, before anything else runs), and never via `git stash` / `git checkout`.

### F2 — MINOR. `tests/test_pass_import.py::_plan` carries a `# type: ignore` outside the allowlist.

```python
def _plan(**overrides: object) -> ImportPlan | str:
    args: dict[str, object] = {...}
    args.update(overrides)
    return plan_import(**args)  # type: ignore[arg-type]
```

`conventions.md ## Code rules` bans suppressions outside the `## Known quirks` allowlist, and
that allowlist is explicitly "upstream stub gaps only" (`moderngl.Uniform.gl_type`,
`create_standalone_context(backend=)`, `openai`'s TypedDict params). This one is the test's own
`dict[str, object]`-splat, not an upstream gap — exactly the "new markers outside this list are
a design smell; fix the design" case.

**Fix:** give `_plan` explicit keyword parameters with the fixture defaults
(`def _plan(*, source_wiring: Wiring = _BLOOM, source_output: str = "composite", group: str = "bloom", substitutions: Mapping[str, str] | None = None, ...)`) and forward them positionally. Every
call site already passes keywords, so no test body changes.

### F3 — MINOR. `ImportDraft.rejection` is written and read by nothing in production.

`shaderbox/ui_models.py::ImportDraft.rejection` is set once, in
`shaderbox/popups/import_passes.py::_draw_body` (`draft.rejection = plan if isinstance(plan, str) else ""`),
and read only by `tests/test_import_dialog.py::test_the_plan_is_recomputed_every_frame`. Its own
docstring justifies it as "so the Import button and a test read the same thing" — but the button
does not read it: `_draw_body` gates `begin_disabled` and the error text on the local `plan`
variable. So the field is state on a persisted-adjacent model that exists for one assertion.

This is the speculative-machinery bullet's "is REMOVING it churn?" test with the answer "no, and
removing it costs one test line". Scored against the rule's reconciling axis, it is surface with a
maintain cost (a field on the draft model that every future reader must decide whether to set) and
no consumer.

**Fix:** either have the button read `draft.rejection` (making the docstring true, one field, one
source) or drop the field and let the test drive `_plan(draft, source, host)` directly. The first
is the smaller diff and matches the stated intent.

### F4 — COSMETIC. `group_tint` was inserted mid-way through theme.py's SELECT invariant block.

`shaderbox/theme.py` has a banner comment ("Theme-portability invariant (enforced at import)")
over a block of `assert COLOR.SELECT ...` statements. 091 inserted `_GROUP_TINT_EXCLUSIONS`, its
two asserts, AND the `def group_tint(...)` function between the first SELECT assert and the second
(`assert COLOR.SELECT not in {COLOR.STATE_OK, ...}`), so a function definition now splits one
invariant block in half. The conventions bullet describes the placement as "the import-time assert
beside the SELECT invariant", which is accurate for the asserts and not for the function.

**Fix:** move `def group_tint` below the last assert, leaving the assert block contiguous. No
behavior change (module-level asserts and a def at module scope are order-independent here).

---

## Coverage table

### 1. `conventions.md ## Code rules`

| Rule | Verdict | Evidence |
|---|---|---|
| Full type annotations on params and variables | COVERED | Every new def is annotated, including locals that need it (`segments: list[tuple[str, imgui.ImVec2, imgui.ImVec2]]`, `open_segment: tuple[...] | None`, `sources: dict[str, dict[str, str]]`, `handed: dict[str, dict[str, str]]`). `_copied_uniform_value(gl: moderngl.Context, value: Any) -> Any` uses `Any` legitimately — it dispatches over the untyped `uniform_values` heterogeneous store, the same `Any` every sibling value-handler in `project_session.py` uses. `make check` reports 0 pyright errors. |
| No `from __future__ import annotations` | COVERED | `git show 4dc1423 \| grep "from __future__"` → no hits. |
| Imports at module top only | COVERED | Grepped all 26 changed `.py` files for `^[ \t]+\(import \|from .* import \)`: zero hits (the two `theme.py` matches are inside its module docstring's usage example, pre-existing). `tests/test_import_diet.py` passes. |
| American spelling | COVERED | Grepped the whole diff for `colour\|behaviour\|centre\|initialise\|organis\|normalis`: no hits. `tests/test_prose_spelling.py` passes. |
| Comments state what is non-obvious NOW, no development-history narration | COVERED | Read every new comment. Each names a live non-obvious fact, not a story: `pass_graph.py`'s `PASS_NAME_RE` comment ("a pass name is a FILENAME and a graph key") is a constraint; `pass_list.py`'s `_GROUP_INSET` ("a full row can end flush with the panel's edge, so a rect outside the tiles would clip") is a geometry fact; `import_passes.py::_draw_tabs`'s comment is a ≤3-line imgui-behavior fact with the sanctioned `/imgui-ui §8` pointer shape; `project_session.py::_copied_uniform_value`'s comment states the ownership invariant (`Pass.release` frees what it holds). `theme.py`'s `GROUP_TINTS` comment explains why four and why `aqua_n` is safe — a palette fact, not a changelog. No "the bug we hit" blocks anywhere in the diff. |
| Comments not restating the code | COVERED | The closest call is `ui.py::planned_set_mode`'s docstring, which names the two return values — but it also carries the non-obvious reason (the project tab's cards blit live canvases, so the set must not pause), which is the part a reader cannot get from the signature. |
| No `@staticmethod` / `@classmethod` | COVERED | No hits in the diff. `compile_pending_passes`, `offered_entry_points`, `_copied_uniform_value`, `group_tint`, `entry_points`, `group_slug`, `group_runs`, `_existing_groups` are all module-level free functions — the rule's prescribed shape. |
| No `if TYPE_CHECKING:` | COVERED | No hits in the diff. |
| No suppressions beyond the allowlist | **VIOLATED** | See F2: `tests/test_pass_import.py:33` `# type: ignore[arg-type]`. One new marker; no `# noqa` or `# pyright: ignore` anywhere in the diff. |
| UI prose budgets | COVERED | `tests/test_ui_prose_budget.py` passes, and the gate's reflection does reach the new module (its `_UNMEASURABLE` gained a reasoned `import_passes.py::_draw_description` entry for the source document's own description). I dumped the scored sites to confirm rather than trusting the green: every new fixed string is measured and inside budget — `separator_text("Entry points")` 2/2, `primary_button("Import … passes")` 3/3, `help_marker("marks the tiles and prefixes the passes")` 7/8, `help_marker("the tile's group label")` 4/8, `label_row("group")` 1/2, `standard_button("import...")` 1/3, and nine `caption_text` lines at 2-4 of 4. By eye they read as control copy, not documentation. One judgement call worth naming, not a finding: `imgui.text_colored(COLOR.STATE_WARN, f"{name} does not compile; copied as is")` is unscored (the gate scores `text_colored` only at `FG_DIM`) and carries a `;` clause joiner, which the gate rejects in measured copy — it is a dynamic diagnostic rather than fixed copy, which is the category the gate's design exempts, so it stands. |
| No raw line numbers or file-length counts in docs | COVERED | Grepped the `ai_docs/` half of the diff for `\.py:[0-9]+`, `\.md:[0-9]+`, `N L)`, `(NNN lines`: no hits. Every new citation is a symbol (`group_runs`, `pass_import.plan_import`, `preview_cell(bordered=False)`, `PassEntry.group`, `ProjectSession.import_passes`). |

### 2. `conventions.md ## Design decisions` — the bullets that touch 091

| Rule | Verdict | Evidence |
|---|---|---|
| Three-layer UI: `app.py` owns state, no drawing | COVERED, with a judgement noted in §3 below | No imgui call in any of the nine new `App` methods. |
| `ui.py` orchestrates | COVERED | `ui.py` gained `planned_set_mode` (a layout/scheduling decision composing siblings, which is what the bullet puts there) and one `draw_import_passes(app)` line in the popup block. |
| `popups/*.py` shape: free `draw(app)`, state on `App` as one `PopupState` field | COVERED | `popups/import_passes.py::draw_import_passes(app: App) -> None` is a free function; every helper is a module-private free function taking `app`/`draft`; no classes. `PopupState.IMPORT_PASSES` is a new enum member, `open_import_passes()` sets it via `_open_popup`, the body self-closes to `CLOSED` through `close_import_passes()`, and the `draw_import_passes(app)` call landed in `ui.py`'s popup block — the step the bullet calls out as the one that gets forgotten. `tests/test_project_management.py` (which counts the block's calls against the enum) passes. |
| `widgets/*.py` shape: free functions taking `app`, no ABC | COVERED | `pass_list.py` gained `_draw_group_outline(app, group, lo, hi)` and a `group: str` parameter on `_draw_pass_tile` — free functions, no protocol, no shared return shape. |
| Button tiers: every new button through a tier | COVERED | The two new buttons are `standard_button("import...")` (pass_list) and `primary_button(label)` + `standard_button("Cancel")` (the modal's action row). `tests/test_button_tiers.py` passes. The `imgui.checkbox` / `imgui.selectable` calls in the new code are not labelled verbs — a checkbox is a state toggle and a selectable is a combo row, both outside the four-tier count, matching `pass_settings.py`'s existing `imgui.checkbox("smooth##…")`. |
| Theme tokens: no hardcoded colors or magic px outside `theme.py` | COVERED | Every color in the new code is a token: `COLOR.SELECT`, `COLOR.FG_DIM`, `COLOR.FG_SECONDARY`, `COLOR.STATE_ERROR`, `COLOR.STATE_WARN`, `COLOR.BG_SURFACE`, and the new `COLOR.GROUP_TINTS` / `COLOR.GROUP_FILL_ALPHA` — both added to `_ColorBag` mapping `_P` entries, never literals, exactly as the color-roles bullet requires. The alpha composite `(*tint[:3], COLOR.GROUP_FILL_ALPHA)` reads both halves from tokens. On the numbers I was asked to check specifically: `_GROUP_INSET` / `_GROUP_ROUNDING` / `_GROUP_LABEL_PAD` (pass_list) and `_POPUP_W` / `_POPUP_H` / `_COMBO_W` / `_ROW_LABEL_W` / `_CTRL_W` / `_GRID_COLS` / `_GRID_ROWS` (import_passes) are module-private named constants, which is exactly how every sibling holds such numbers — `emoji_picker.py` (`_GRID_COLS`, `_CELL`, `_POPUP_W/H`), `help.py` (`_POPUP_W/H`, `_LIST_W`), `examples.py` (`_GRID_COLS`, `_GRID_MAX_ROWS`, `_DESC_SLOT_H`), `projects.py` (`_POPUP_W/H`, `_PATH_X`), and `pass_settings.py`'s own pre-existing `_ROW_LABEL_W = 110.0` / `_CTRL_W = 168.0`, which import_passes reuses by value. `SPACE.SM` / `SPACE.MD` / `SIZE.THUMB_LG` / `SIZE.PASS_TILE` are taken from tokens wherever a token exists. No new number belongs in `theme.py`: these are one-surface layout, not cross-surface design tokens, which is the line the sibling modules already draw. |
| The funnel rule (one save per verb) | COVERED | Each session verb saves exactly once: `set_pass_group` ends in one `save_ui_document`; `import_passes` writes every pass, the graph and the handovers and then saves ONCE at the end (the docstring and the commit both state this, and it is what keeps the source — possibly a read-only shipped example — unwritten). `App.create_pass_from_draft` calling `add_pass` then `set_pass_target` then `set_pass_iterations` then `set_pass_group` is N saves, but that is the pre-existing shape this commit extended by one branch, and `_apply_entry`'s own comment states the intent ("one per changed field, so the document follows every control the frame it moves"). Not a new violation. |
| The lockstep-dicts smell | COVERED, and deliberately avoided | `PassEntry.group` is a field on the entity, not a `dict[name, group]` beside `dict[name, PassEntry]` — the bullet's prescribed remedy applied on the first try. The new conventions bullet states it as the decision ("there is no group table or member list to keep in step with it"), and `tests/test_pass_verbs.py::test_the_group_survives_rename_and_goes_with_delete` plus `test_graph_persistence.py::test_a_group_round_trips_and_an_absent_key_reads_empty` are the invariant tests that every entry mutation carries the field. |
| Speculative machinery | **VIOLATED (minor)** | See F3: `ImportDraft.rejection` is written by the popup and read by no production code. Everything else earns its place: `ImportResult.notes` is consumed by `App.import_passes_from_draft`; `ImportPlan.handovers` / `.becomes_output` are consumed by `ProjectSession.import_passes`; `tab_select_pending` mirrors the existing `document_tab_select_pending` idiom; `preview_cell(bordered=)` has its one caller and its own falsifier test. |
| `ProjectSession` headless invariant (no imgui context) | COVERED | No imgui or glfw import in `project_session.py` or `pass_import.py` (the only matches are the word "imgui" inside explanatory comments). `tests/test_import_diet.py` passes. `ImportResult`, `compile_pending_passes`, `offered_entry_points` and `import_passes` are all GL/imgui-context-free; `import_passes` touches `moderngl` only through `host.gl`, which is the core's existing surface. |
| Copilot conventions for a new tool argument | COVERED | `group: str | None = None` on `_SetPassArgs` with a model-facing description, threaded `capabilities.py` (Protocol) → `tools/passes.py` (handler) → `backend.py::set_pass` → `_configure_pass` → the injected `pass_set_group` callback → `ProjectSession.set_pass_group`. It is an argument on the EXISTING `set_pass` tool rather than a new tool, so the tool count does not grow. Addressing is right per the document-addressing bullet: setting a group is reversible and project-internal, so it takes the explicit document id and does not switch. The reject path is a domain reject — `set_pass_group` returns a message string that `_configure_pass` returns and the handler turns into `(False, "error: …")`, never a bare string through the generic path. `None` means "leave it" and `""` means "leave the group", distinguished correctly (the `if group is not None` guard), and `test_copilot_pass_tools.py::test_set_pass_group_lands_and_echoes` pins all three cases plus the working-set table echo. |
| `/imgui-ui` §7 modal conventions | COVERED | Chrome (§7.1): bottom action row, `primary_button` left + `standard_button` right, `imgui.dummy((0, SPACE.MD))` above it, primary at content width. Labeling (§7.1): "Cancel" is correct — this is a form whose commit mutates state, which is the rule's stated condition for Cancel over Close (and `pass_settings.py` keeps "Close" for its view-only edit mode, so the two read as one system). Wrapper (§7.2): `modal_window(_LABEL, (_POPUP_W, _POPUP_H))` is used, not a hand-rolled `begin_popup_modal`; `is_popup_open`/`first_use_ever` come from the wrapper. `keep_open` (§7.3): `_draw_body(app) -> bool` returns `keep_open` with the right polarity, both close paths set the same local, and the cleanup (`app.close_import_passes()` + `imgui.close_current_popup()`) runs at the wrapper call site after the body returns False, not inside the body. Reset on open/close (§7.6): `open_import_passes` assigns a fresh `ImportDraft()` — every transient field resets by construction rather than by a reset method that can miss one; `close_import_passes` nulls the draft, so nothing dangles. Escape reaches that funnel (`hotkeys.py::_handle_escape` gained the `IMPORT_PASSES` branch) rather than the bare `popup_state = CLOSED` fallthrough, with `test_escape_reaches_the_close_funnel` pinning it. §7.5's focus rule is satisfied the other way round, and deliberately: the module docstring records that NO field auto-focuses, because a focused `input_text` would write its buffer back over the programmatic group prefill — so there is no every-frame `set_keyboard_focus_here`. |

### 3. Module boundaries

| Question | Verdict | Evidence |
|---|---|---|
| `pass_import.py` imports `pass_graph` only | COVERED | Its entire import list is `collections.abc`, `dataclasses`, and `from shaderbox.pass_graph import PASS_NAME_RE, Wiring, entry_points`. No GL, no imgui, no `document`, no `core`. The dev_flow claim ("leaf, GL-free … Imports `pass_graph` only") is literally true. |
| `pass_graph.py` stays GL-free and imgui-free | COVERED | Imports after 091: `re`, `collections.abc`, `dataclasses`, `typing`, `pydantic`. The new `re` import is the only addition and is stdlib. `PASS_NAME_RE` moving here from `project_session.py` is a net improvement — the pattern is a graph-key constraint, and `project_session` now imports it rather than owning a private copy that `pass_import` would have had to duplicate or reach across a layer for. |
| Is `project_session` the right home for `offered_entry_points`? | FINDING, sub-blocking (fold into F3's class) | `offered_entry_points(document: Document) -> list[str]` and `compile_pending_passes(document: Document) -> list[str]` are free functions whose every argument is a `Document` — they read `document.passes[*].program` and `document.effective_wiring()` and touch no session state at all. Their non-test consumers are `popups/import_passes.py` (the first) and `App.select_import_source` + `ProjectSession.import_passes` (the second). **The layering rule I applied:** a free function over one entity belongs in that entity's module unless it needs a second collaborator; `document.py` already hosts exactly this shape (`document_dir_of(document)`, `sampler_names(render_pass)`), already imports `pass_graph`, and is already imported BY `project_session`, so the move is cycle-free in the direction it needs to go. Keeping them in `project_session.py` makes a popup import the session module for a pure `Document` query, which is the weaker arrangement. This is a judgement, not a rule violation — the conventions name no home for Document-scoped free helpers, and `project_session.py` is a legitimate home for `compile_pending_passes` on the grounds that `import_passes` is its only production caller. The one I would actually move is `offered_entry_points`, whose sole production caller is the popup. |
| Which of `App`'s nine new methods are state, which are logic? | MIXED, see below | |

`App` gained nine methods (the task said seven; the diff shows nine). Against the three-layer
rule ("`app.py` owns all imgui-bound state", leaves are pure draw functions):

- **State / lifecycle, correctly on `App`** (4): `open_import_passes` (the busy guard + the
  `PopupState` transition + draft construction — the `open_*()` helper the popups bullet
  mandates), `close_import_passes` (the mutex reset), `commit_pass_group` (owns
  `pass_settings_group_buf`, the App-held modal buffer, and the toast-and-restore on refusal),
  `import_passes_from_draft` (the verb call plus `self.notifications` — the UI reaction).
- **State, correctly on `App`** (2): `import_sources` and `import_source` — both read
  `self.ui_documents` / `self.ui_document_examples` / `self.current_document_id`, which is App's
  own state, and neither can be a free function without taking `app` anyway.
- **Draft mutation, correctly on `App`** (2): `select_import_source` and
  `set_import_substitution` mutate `self.import_draft` and call `self.ui_documents`-scoped
  helpers. These hold real cross-field logic (the reseed-and-clear on source change; the
  remove-old-handovers/add-new set algebra), and `test_import_dialog.py` drives them headlessly
  through `App` — which is the right test seam precisely because they are App state transitions.
  Defensible as landed.
- **Pure logic, the one I would move** (1): `host_readers_of(fed: str) -> set[tuple[str, str]]`
  is a pure inverse-wiring query — `{(name, uniform) for name, reads in wiring.items() for
  uniform, read in reads.items() if read == fed}` — whose only `self` use is fetching the current
  document's wiring. Its natural home is `pass_graph.py` as
  `readers_of(wiring: Wiring, name: str) -> set[tuple[str, str]]`, beside `entry_points(wiring)`
  which 091 just added there: same module, same input type, same purity, and then it is unit-
  testable with a dict and no `App`. `App.host_readers_of` would shrink to one line fetching the
  wiring and delegating. Minor, and not a rule violation — but it is the one method of the nine
  whose body is logic rather than state.

### 4. Duplication

| Candidate | Verdict |
|---|---|
| The group field in the settings modal's edit and create modes | NOT duplicated — already extracted. `popups/pass_settings.py::_draw_group(app, id_, group, existing)` is one helper called from both `_draw_draft` (create) and `_draw_body` (edit), exactly mirroring how `_draw_target` and `_draw_repeat` already serve both modes in that file. `_existing_groups(graph)` is likewise one helper. Nothing to extract. |
| The grid in `popups/examples.py` vs `popups/import_passes.py` | ACCEPTABLE, and the shared part is already extracted. Both call `widgets/document_grid.py::draw_document_preview_button` — the one primitive that wraps `preview_cell` with a document's texture, name and border — so the actual cell drawing is shared. What repeats is ~8 lines of grid arithmetic: a `cell_h` from `SIZE.THUMB_LG + get_text_line_height_with_spacing()`, a `grid_h` from rows × cell_h + spacings, and the `if (i + 1) % _GRID_COLS != 0 and i != len(...) - 1: same_line() else: spacing()` wrap. They differ in the parts that matter — `examples.py` derives its row count from the item count and caps it at `_GRID_MAX_ROWS` with a scrollbar, computes a `grid_w` to size the whole modal, and composes a `_DESC_SLOT_H` description pane below; `import_passes.py` fixes two rows and sizes to `0.0` width inside an already-sized modal. This is the repo's "don't abstract from N=1" case rather than its "extract to ui_primitives" case: the conventions' speculative-machinery bullet prescribes writing the second consumer's full field list and checking whether <30% lands inside, and here the genuinely common residue is the two lines of `% _GRID_COLS` wrap logic. A `preview_grid` primitive would have to take rows-or-auto, a width mode, a cap, and an optional detail pane to serve both — more surface than it saves. Leave it; the trigger for extraction is a third caller. |
| `App.host_readers_of` vs anything in `document.py` | NOT duplicated. `document.py` has no inverse-wiring query (`Document.sampler_source` answers the forward direction for one uniform; `effective_wiring` returns the forward map). The only adjacent comprehension is `popups/import_passes.py::_draw_entry_points`'s `readers = sorted(name for name, reads in wiring.items() if root in reads.values())`, which looks similar but has a different subject (the SOURCE document's wiring, not the host's) and a different shape (pass names only, not `(pass, sampler)` pairs) — it feeds a `readers: a, b` caption while `host_readers_of` feeds the handover checkbox set. Two genuinely different questions; not a duplication. The §3 note about moving `host_readers_of` to `pass_graph` stands on purity, not on duplication. |

### 5. The new conventions bullets and dev_flow module-map entries

Read all five new/edited doc blocks against the code.

| Claim | Verdict |
|---|---|
| `conventions.md`: "A pass GROUP is a label on the pass entry, and nothing folds" | COVERED — accurate and at the right altitude. `PassEntry.group` is the whole model as claimed; "every existing entry mutation carries it because they all go through `model_copy` or carry the entry object" is verifiable (`with_group`, `with_target`, `with_output` all `model_copy`; `_graph_renamed` and `_graph_without` carry the entry object) and pinned by `test_the_group_survives_rename_and_goes_with_delete`. `group_runs` cuts by adjacency as stated. Form is right: "we decided X (a label, nothing folds); revisit if a group-level fact appears that no member can hold". |
| `conventions.md`: "Import is by COPY, decided as a pure plan over two wirings" | COVERED — accurate. The "both AFTER every pass has compiled, since a never-compiled pass answers its explicit rows only and every pass then reads as a root" clause matches `compile_pending_passes` and is the non-obvious fact worth filing. The entry-point/substitution/handover description matches `plan_import` exactly. "Values are COPIED, never shared: `Pass.release` frees what it holds" matches `_copied_uniform_value`. "The source is never saved" matches the single `save_ui_document(ui_document)` on the host. Revisit trigger is concrete (a second document-shaped source needing something `UIDocument` lacks). |
| `conventions.md`: "Group tints are theme tokens picked by a stable hash" | COVERED with one wording drift. `group_tint` indexes `COLOR.GROUP_TINTS` by `zlib.crc32` as stated; the exclusion set and the pure `tests/test_theme.py` gate both exist as described, and the parenthetical reason ("an import-time assert cannot be tripped from a test") is the honest justification for having both. The drift: "the import-time assert beside the SELECT invariant" describes the asserts correctly but the commit also put the `group_tint` FUNCTION inside that assert block (F4). Fix F4 and the sentence becomes exactly true. |
| `dev_flow.md`: the `pass_list.py` strip entry | COVERED — accurate. "the fill on the parent draw list, the outline and label on the foreground list clipped to the strip, since the tiles are child windows that paint over their parent" matches `_draw_group_outline` line for line (`parent.add_rect_filled`, `fg.push_clip_rect(parent.get_clip_rect_min(), …)`, `fg.add_rect`, `fg.add_text`). "member tiles keep their padding through `preview_cell(bordered=False)`" matches the `ChildFlags_.always_use_window_padding` branch. Cited by symbol, no line numbers. |
| `dev_flow.md`: the `popups/import_passes.py` and `pass_import.py` entries | COVERED — accurate. Two tabs, the group name that also prefixes, one combo per entry point, handover checkboxes, "the plan is recomputed every frame", "no field is auto-focused", "executes through `ProjectSession.import_passes`" — all verified against the code. `pass_import.py`'s "leaf, GL-free … Imports `pass_graph` only" is literally true. The `pass_settings.py` entry's added "group (091)" is correct. |

Altitude: all three conventions bullets are generic future-constraining rules with revisit
triggers, in the right section (`## Design decisions`, not `## Code rules` or `## Known quirks`),
and none is a current-state snapshot. The per-feature mechanics correctly stayed in the spec's
`## Implementation notes` rather than migrating up. One altitude note on the spec itself: the
`## Implementation notes` "Deviations from the spec" list and the falsifier table are exactly the
feature-record content the conventions bullet says belongs there rather than in `conventions.md` —
correct placement, undermined only by F1's false gate claim inside it.

### 6. `tests/` style

| Rule | Verdict |
|---|---|
| Docstring stating the falsifier | COVERED for the two new files. `test_pass_import.py`'s module docstring names the falsifier class explicitly ("above all skipping the materialization, which leaves a bundle whose every inter-pass edge is gone while only its feedback survives, silently"); `test_import_dialog.py`'s names its subjects and the constraint that shaped the design ("A second frame-driving App in one process hits the torn-down font atlas, so the Escape wire is asserted … on the dispatch's source"). Per-test falsifiers are in the body comments, which matches the sibling style (`test_pass_verbs.py`, `test_pass_graph.py`, `test_theme.py` all do per-test `# Falsifier:` comments), and the additions to existing files follow each file's own convention including the `# --- 091 ---` section banners that `test_pass_graph.py` and `test_pass_verbs.py` already use. |
| Helpers at top | COVERED. `test_pass_import.py`: `_BLOOM`, `_HOST`, `_plan` before the first test. `test_import_dialog.py`: `_multi_pass_example` at top; `_pump` sits mid-file immediately before the one test that uses it, which is the same placement `test_pass_verbs.py` uses for `_imgui_frame`-adjacent helpers. Additions to `test_pass_verbs.py` put `_BLOOM_FIXTURE`, `_CONST_HALF`, `_HALVE_SCENE`, `_load_bloom`, `_rows` at the head of the new 091 section rather than the file top — consistent with how that file's earlier sections are organized. |
| No in-function imports | COVERED. Grepped all six touched test files for indented `import` / `from … import`: zero hits. |
| Sibling-style source reads | COVERED. `test_import_dialog.py::test_escape_reaches_the_close_funnel` reads `Path("shaderbox/hotkeys.py")` — the same cwd-relative source-read assertion style as `tests/test_button_tiers.py`, and the docstring explains why it asserts on the source rather than driving a second App (the font-atlas constraint). |
| `Any`-typed fixtures | COVERED. `app: Any`, `monkeypatch: Any` matches every sibling app-fixture test in the repo (`conftest.py::app` is itself `-> Iterator[Any]`). |
| Suppressions | **VIOLATED** — F2, in `test_pass_import.py`. |

---

## False trails

Four things I suspected and disproved; recording them so the next reviewer does not re-spend the
time.

1. **The handover checkboxes' `##handover` id suffix is NOT an ID collision.**
   `popups/import_passes.py::_draw_handovers` draws `imgui.checkbox(f"{pair[0]}.{pair[1]}##handover", on)`
   — every checkbox in the row ending in the same literal suffix, and inverted from the repo's usual
   `f"label##{unique}"`. I expected colliding IDs (two checkboxes sharing state). Tested it on the
   installed imgui-bundle: `get_id("grade.u_main##handover")` and `get_id("final.u_grade##handover")`
   are `0x4e67c07e` and `0x440eb7a7` — distinct. `##` suppresses DISPLAY of the suffix but the whole
   string is hashed; `###` is the form that discards the prefix. No bug, and no finding. The unusual
   ordering is cosmetic only (it keeps the visible label `pass.sampler` free of the suffix).

2. **`ImportDraft` is a `@dataclass`, not a persisted pydantic model — so it needs no salvage.**
   I checked whether the per-KEY fail-soft persistence bullet applied. It does not: `ImportDraft`
   lives only in `App.import_draft` for the life of the open modal, never reaches disk, and
   `tests/test_persistence_completeness.py` is correctly not tripped. `PassEntry.group` IS persisted,
   and it does the right thing — a `Field(default="", pattern=_GROUP_PATTERN)` constraint ON THE
   MODEL (the bullet's prescribed shape, not a call-site check), with an absent key reading `""` and
   a malformed one costing that field alone. `test_a_group_round_trips_and_an_absent_key_reads_empty`
   pins both halves.

3. **`ui_uniforms` row merging does not trip the lazy-row trap or the derived-state prune.**
   `ProjectSession.import_passes` does `ui_document.ui_state.ui_uniforms.setdefault(key, row.model_copy())`,
   and I expected the save-funnel prune to drop rows keyed to passes that do not exist under the new
   names. It does not: `get_uniform_hash` keys on name-and-shape only, never on the pass, so a merged
   row survives the rename by construction. `test_a_merged_ui_row_survives_the_save` pins exactly
   this, with the right falsifier named ("re-key the merged row by the new pass name (a hash nothing
   computes) and the prune drops it").

4. **The local `_POPUP_W` / `_ROW_LABEL_W` / `_GROUP_INSET` constants are not theme-token
   violations.** I opened this expecting magic px outside `theme.py`. Every sibling popup and widget
   holds its one-surface layout numbers as module-private named constants — `emoji_picker.py`,
   `help.py`, `examples.py`, `projects.py`, and `pass_settings.py`'s own `_ROW_LABEL_W = 110.0` /
   `_CTRL_W = 168.0`, which import_passes reuses by the same value for visual consistency. The theme
   bullet governs COLOR roles and the shared `SIZE`/`SPACE` tokens, both of which the new code uses
   wherever a token exists. Not a finding.
