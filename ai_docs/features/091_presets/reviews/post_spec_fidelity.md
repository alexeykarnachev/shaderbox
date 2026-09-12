# 091 post-implementation review — spec fidelity

Anchor: `ai_docs/features/091_presets/01_spec.md` against commit `4dc1423` (dev). Every claim
below was measured against that commit, with the tree clean; the worktree used for part C was
removed and the main tree verified at `4dc1423` afterwards.

While this audit was being written another reviewer began editing the working tree — among
other things reformatting the three files of F1 and moving `offered_entry_points` out of
`project_session.py`. Those edits are **not** in `4dc1423`, so the findings stand as findings
about the commit; F1 in particular may already be fixed in the tree by the time this is read,
and the check to re-run is `make gates` on whatever is committed next.

## Verdict: FINDINGS

Every locked decision landed or has a recorded deviation; every verification item has a test
that asserts what the item says; all six break-table rows go red. Three findings, none of
them a missing decision.

| # | Severity | Finding |
|---|---|---|
| F1 | **Major** | `make gates` is RED at `check` on this commit: three 091 files fail `ruff format` / import-sort. The Implementation notes say "`make gates` green (check, test, smoke), exit code read unpiped." |
| F2 | Minor | D4's rejection "a plan that copies nothing (a single-pass source whose one pass is substituted)" is unreachable by that route, and an undocumented seventh rejection (`'<x>' is the output and stays`) is what actually fires. |
| F3 | Trivial | The break table's first row claims 3 red tests in `test_pass_import.py`; 2 go red. |

`make test` is green (2262 passed, 4 skipped). Every cited test file passes. Nothing from
*Out of scope* landed.

### F1 — the check gate is red on the commit (Major)

`make gates > /tmp/g.log 2>&1; echo $?` → **exit 2**, failing at `check`, before test and
smoke run:

```
ruff (legacy alias)......................................................Failed
ruff format..............................................................Failed
- files were modified by this hook
== gates: FAILED at check (exit 2); test and smoke not run ==
```

Reproduced against a pristine `git worktree` of `4dc1423`, so it is the commit and not a local
artifact:

```
$ ruff format --check shaderbox/ tests/
Would reformat: shaderbox/pass_import.py
Would reformat: tests/test_import_dialog.py
2 files would be reformatted, 296 files already formatted
$ ruff check shaderbox/ tests/
... help: Organize imports        # shaderbox/popups/import_passes.py
Found 1 error.
```

The three files are 091's own: `shaderbox/pass_import.py` (the `taken` ternary at line 69),
`shaderbox/popups/import_passes.py` (imports at 18–21 — `document_dir_of` and
`offered_entry_points` out of alphabetical order), `tests/test_import_dialog.py` (four
over-long lines). All are whitespace/ordering only; no behaviour changes. The first `make
gates` run of the audit reported `pyright ... Failed / files were modified` and a clean
`git status` — that is the hook's own run-order artifact; the second run surfaced the real
ruff rewrite. Left unfixed: this audit is read-only.

The hard rule in `CLAUDE.md` is that `make gates` decides done, judged by the exit code
captured unpiped. The Implementation notes assert that was done and green; it was not.

### F2 — one D4 rejection is unreachable, and an undocumented one fires instead (Minor)

D4 lists five rejections. `plan_import` has **seven** returns, and the mapping is not the
listed one. `shaderbox/pass_import.py:60`:

```python
if source_output in substitutions:
    return f"'{source_output}' is the output and stays"
```

This guard sits *before* the `if not copied:` check at line 64, so for D4's named case — a
single-pass source whose one pass is substituted — line 64 can never be reached:
`PassGraph.output_pass` falls back to the only pass of a single-pass document
(`pass_graph.py:196`), so that pass is always `source_output` and the guard fires first.
Measured:

```
plan_import({"a":{}}, "a", "g", {"a":"main"}, …)  →  "'a' is the output and stays"
plan_import({"a":{}}, "",  "g", {"a":"main"}, …)  →  "nothing to import: every pass is replaced"
```

Line 64 is reachable only when `source_output` is not a key of `source_wiring` (an
`output_pass` of `None` passed as `""`), which the dialog cannot produce for a single-pass
source. The test for D4(d),
`tests/test_pass_import.py::test_replacing_the_only_pass_rejects`, asserts `"output" in plan`
— i.e. it pins the *guard's* message, not the one the spec names, so the item is **PARTIAL**:
the "nothing to import" branch has no test and no dialog route.

The guard itself is coherent UI — `popups/import_passes.py:196` wraps an output entry point's
combo in `begin_disabled`, captioned "the output, stays" — but it is a behavioural rule the
spec never states, and it has a consequence D3 does not draw out: D3 says "a single-pass
document's one pass is both entry point and output", so a single-pass source can never be fed
at all, only copied. Not recorded under *Deviations*.

### F3 — break-table row 1 overcounts (Trivial)

The row "`plan_import` never fills `sources` (no materialization)" claims "`test_pass_import.py`
(3 tests)". Re-applying the break in a worktree gives **2**:
`test_every_wired_sampler_is_explicit_under_the_new_names` and
`test_a_fed_entry_point_is_not_copied_and_its_readers_point_at_the_host`. The third candidate,
`test_a_source_pass_with_no_edges_still_plans`, asserts `"bloom_blur" not in plan.sources`,
which stays true under the break. The named third red test
(`test_a_rendered_import_reads_its_bundle`, output reads black) did go red. The break is
caught; only the count is wrong.

---

## A. Per-decision audit

Every sentence of D1–D11 making a claim about the code. "Rec." = recorded under
`## Implementation notes ### Deviations`.

### D1 — `PassEntry.group` is the group

| Claim | Verdict | Evidence |
|---|---|---|
| `PassEntry.group: str = ""` one field on the existing entry | LANDED | `pass_graph.py:124` `group: str = Field(default="", pattern=_GROUP_PATTERN)` |
| `graph.json` gains one key per grouped pass, nothing else | LANDED | `test_graph_persistence.py::test_a_group_round_trips_and_an_absent_key_reads_empty` |
| No group table / member list / group object | LANDED | no second structure anywhere; `grep parent_group\|subgroup` empty |
| Name obeys `PASS_NAME_RE` | LANDED | the pydantic `pattern=_GROUP_PATTERN` (`^([A-Za-z_][A-Za-z0-9_]*)?$`, i.e. the regex or empty) plus `set_pass_group`'s explicit check, `project_session.py:1096` |
| `PASS_NAME_RE` moves from `project_session.py` to `pass_graph.py` | LANDED | `pass_graph.py:55`; `project_session.py` imports it back and `_pass_name_error` uses it |
| …loses its underscore | LANDED | `_PASS_NAME_RE` → `PASS_NAME_RE` |
| `group_slug` and `plan_import` both validate against it | LANDED | `pass_import.py:51`; `group_slug` guarantees a match — pinned by `test_group_slug_is_the_first_word_made_legal`'s final loop |
| Field rides `load_graph`'s per-entry salvage unchanged | LANDED | `load_graph` untouched in the diff; `test_a_group_round_trips…` drives the loader |
| Every existing mutation carries the group with no edit | LANDED | `test_the_group_survives_rename_and_goes_with_delete` (rename keeps `fx`, survives reload) |
| Two sites compare against `PassEntry()` defaults and must not be read as "is this entry default" | LANDED | `app.py:1121,1125` compare `.target` and `.iterations` per field; the new group branch at 1130 is `if entry.group:`, not an entry comparison |
| `PassGraph.with_group(name, group)` | LANDED | `pass_graph.py:181` |

### D2 — the group name is the prefix

| Claim | Verdict | Evidence |
|---|---|---|
| Group `bloom` names copies `bloom_<source name>` | LANDED | `pass_import.py:66-67`; `test_import_copies_the_bundle_under_the_group_and_feeds_it` asserts `bloom_bright` … |
| An empty group imports under own names, no group | LANDED | `pass_import.py:66` `prefix = "" if not group`; `test_an_empty_group_copies_under_bare_names` |
| `group_slug(name: str) -> str` in `pass_graph.py` | LANDED | `pass_graph.py:471` |
| first word, lowercased | LANDED | `words[0].lower()` |
| chars outside `[A-Za-z0-9_]` → `_` | LANDED | `re.sub(r"[^A-Za-z0-9_]", "_", …)` |
| `g_` prepended when the result doesn't start with a letter or underscore | LANDED | `pass_graph.py:476-477`; `group_slug("2D SDF") == "g_2d"` asserted |
| `preset` for an empty name | LANDED | `pass_graph.py:474-475`; `group_slug("") == "preset"` asserted |
| `Bloom Chain`→`bloom`, `Radiance Cascades`→`radiance`, `2D SDF`→`g_2d`, `""`→`preset` | LANDED | all four in `test_group_slug_is_the_first_word_made_legal` |
| Dialog prefills from the source's display name through `group_slug` | LANDED | `app.py:select_import_source` → `group_slug(source.ui_state.ui_name)`; `test_the_draft_resets_on_source_change_and_on_close` |

The `g_` branch is reached only via the non-letter path, since the `_`-start case already
matches `^[A-Za-z_]`; behaviour is as specified either way.

### D3 — entry points are the roots of the effective wiring

| Claim | Verdict | Evidence |
|---|---|---|
| `entry_points(wiring)` in `pass_graph.py` | LANDED | `pass_graph.py:459` |
| Passes with no source other than themselves | LANDED | `all(source == name for source in reads.values())` |
| A self-read is feedback, not an input | LANDED | `test_entry_points_are_the_passes_reading_no_other_pass` pins `{"acc": {"u_prev": "acc"}} → ["acc"]` |
| Pure, GL-free | LANDED | the whole test module is fixture-free |
| Bloom fixture one root (`scene`), RC one (`paint`) | LANDED | asserted for bloom; RC verified by hand — `effective_wiring` after compile gives `paint` the only root (its shader has no samplers) |
| A single-pass document's one pass is both entry point and output | LANDED | `entry_points({"lone": {}}) == ["lone"]`; but see F2 — it is then un-feedable |
| The wiring is the source's after every pass has been compiled | LANDED | `compile_pending_passes`, `project_session.py:150`, called on source and host before planning |
| RC carries no explicit sampler rows, so before compiling its wiring is empty | LANDED | verified: RC's `document.json` has zero `{"pass": …}` rows; its `graph.json` names none |
| Compiling touches only compile_unit/program/vbo/vao | LANDED (inherited) | `compile_pending_passes` calls `render_pass.compile()` only; recorded as a round-1 false trail, not re-measured |
| A source pass whose compile fails is not offered as an entry point | LANDED | `offered_entry_points`, `project_session.py:165`; `test_a_broken_source_pass_is_imported_as_is_and_named` asserts `== ["scene"]` |
| …is copied as it is with its explicit rows | LANDED | same test: `bloom_blur` lands in `document.passes` |
| …and the import result names it | LANDED | same test: `any("blur" in note for note in result.notes)` |

### D4 — a substituted entry point is not copied

| Claim | Verdict | Evidence |
|---|---|---|
| `ImportPlan` frozen dataclass with `renames`/`sources`/`output`/`handovers`/`becomes_output` | LANDED | `pass_import.py:19-32`, all five fields, types as specced |
| New leaf module `pass_import.py` importing `pass_graph` only | LANDED | `pass_import.py:16` is the sole project import |
| `plan_import` signature as specced | LANDED | `pass_import.py:35-43`, seven parameters in the spec's order |
| `host_wiring` key set is the host's pass names | LANDED | `host_names = set(host_wiring)`, `pass_import.py:53` |
| Returning `str` is the rejection | LANDED | `-> ImportPlan \| str` |
| Rejection: a copied name already among the host's passes | LANDED, reachable | `pass_import.py:68-74`; `test_a_colliding_name_rejects_and_names_it` |
| Rejection: a group name failing `PASS_NAME_RE` | LANDED, reachable | line 51; `test_rejections_name_what_was_wrong` |
| Rejection: a substitution naming a non-entry-point | LANDED, reachable | line 57; same test |
| Rejection: a substitution naming a host pass that doesn't exist | LANDED, reachable | line 59; same test |
| Rejection: a handover pair not reading a fed pass | LANDED, reachable | line 92; `test_a_handover_rewires_that_host_sampler_and_leaves_the_rest` |
| Rejection: a plan that copies nothing (single-pass source, its pass substituted) | **DEVIATED, not recorded** | **F2** — line 64 exists but that route hits line 61's undocumented guard first; no test on line 64, no *Deviations* entry |
| `sources` carries only the WIRED subset | LANDED | lines 78-86 iterate `source_wiring[name].items()` only |
| …to the renamed pass | LANDED | `test_every_wired_sampler_is_explicit…`: `bloom_blur → {"u_bright": "bloom_bright"}` |
| …to the host pass when substituted | LANDED | same test: `bloom_composite.u_scene == "main"` |
| …a self-read written, not skipped | LANDED | same test: `bloom_trail.u_prev == "bloom_trail"` |
| Name-rule wiring does not survive the prefix, so materializing is not optional | LANDED | proved by break-table row 1 going red (`test_a_rendered_import_reads_its_bundle` reads 0) |
| Samplers the wiring did NOT fill are not the plan's business | LANDED | no `else` branch; `test_a_source_pass_with_no_edges_still_plans` asserts no `sources` row |
| **AutoSource paragraph**: an undecided sampler rides D5's value copy; a bound texture copied, `NoSource` stays a decision, `AutoSource` stays undecided; an undecided `u_paint` may catch a host `paint` by name | LANDED | `_copied_uniform_value` (`project_session.py:185`) returns frozen source objects unchanged (`return value`), and only the plan's rows are overwritten (`project_session.py:1158-1160`). No materialize-to-black anywhere — grep for a `NoSource`/black default on copy finds none. Asserted indirectly only; no dedicated test, which the spec does not ask for |

### D5 — `ProjectSession.import_passes` executes the plan and saves

| Claim | Verdict | Evidence |
|---|---|---|
| `ImportResult` frozen dataclass, `error: str = ""`, `notes: tuple[str, ...] = ()` | LANDED | `project_session.py:144-150` |
| `import_passes(document_id, source, group, substitutions, handovers)` | DEVIATED, **recorded** | source is a `UIDocument`, not a `Document` — first *Deviations* bullet, with the reason (the merged `ui_uniforms` rows live on `UIDocument.ui_state`) |
| **Order: compile source's and host's program-less passes** | LANDED | `project_session.py:1129-1130`, the first two statements after the document lookup |
| **…then plan (reject before touching anything)** | LANDED | `plan_import` at 1138, `return` at 1147, all writes after |
| "plan before the first write" ordering | LANDED | no `write_text` / `passes[…] =` / `graph =` precedes the `isinstance(plan, str)` return |
| For each copied pass write its text to `passes/<host name>.frag.glsl` | LANDED | lines 1151-1154 |
| Build a `Pass` with the source entry's target and compile it | LANDED | lines 1158-1164 `target=entry.target`, then `render_pass.compile()` |
| Copy the entry with `group` set | LANDED | `model_copy(update={"group": group})`, line 1155-1157 |
| **Value roster**: scalars/tuples by value | LANDED | `_copied_uniform_value` falls through to `return value` |
| …`PassSource`/`NoSource`/`AutoSource` by reference | LANDED | same fall-through (frozen dataclasses) |
| …`moderngl.Buffer` via `gl.buffer(buf.read())` | LANDED | `project_session.py:199` |
| …`Image` re-opened from its file, never `Image(value.texture)` | LANDED | `MediaWithTexture` branch re-opens from `details.file_details.path` via `media_class_for`; the `Image(value.texture)` fallback fires only when the path is empty, which is the no-file case the spec's prohibition does not cover |
| …`Video` re-opened from `details.file_details.path` | LANDED | same branch, `media_class_for` returns `Video` for a video suffix |
| …raw `moderngl.Texture` via `gl.texture(size, components, data=tex.read(), dtype=…)` | LANDED | lines 192-197, all four arguments |
| …with no default branch | DEVIATED (harmless) | there *is* a fall-through `return value`, which is what carries the scalars/tuples/source-objects the spec wants by value. The spec's "no default branch" means no silent catch-all for an unknown type; the roster is covered by explicit branches plus this one, so the effect matches. Not recorded |
| Source dir is `document_dir_of(source)`, the one place knowing the `passes/` depth | LANDED | used at `project_session.py:1182` (the log line); the media path comes from `file_details.path`, which is already absolute, so the depth knowledge is not needed twice |
| Overwrite the wired samplers with the plan's `PassSource` rows | LANDED | lines 1158-1160 |
| Write the handover rows on the host passes | LANDED | lines 1172-1176 |
| …releasing each overwritten value first | LANDED | `try_to_release(values.get(uniform))` before each write, both loops |
| Set the output when `becomes_output` | LANDED | `output=plan.output if plan.becomes_output else None`, line 1170 (`with_passes` leaves the output alone on `None`) |
| **Save the HOST's `UIDocument` once** | LANDED | exactly one `save_ui_document(ui_document)` at line 1179 |
| Never the source's | LANDED | `test_import_leaves_the_shipped_example_byte_identical` compares every file's bytes |
| Copying rather than sharing keeps `Pass.release` from freeing a source texture | LANDED | `_copied_uniform_value`'s docstring and branches; the spec's item-5 falsifier |
| **`ui_uniforms` rows merged for hashes the host does not have** | LANDED | `project_session.py:1177-1178` |
| …and where the host already has a row of that name and shape, the HOST's row wins | LANDED | `setdefault` is exactly host-wins |
| `get_uniform_hash` keyed by name and shape, so a copied `u_amount` keeps its row with no re-keying | LANDED | `test_a_merged_ui_row_survives_the_save` reads the row back under the source's own `key` |

### D6 — the bundle's output takes over the fed pass's role

| Claim | Verdict | Evidence |
|---|---|---|
| Feeding an entry point with a host pass is an INSERTION | LANDED | the two-scenario test pair in `test_pass_verbs.py` |
| Every host sampler that read the fed pass now reads the bundle's output | LANDED | `test_import_hands_the_fed_passs_readers_to_the_bundle`: after reload `grade.u_main == PassSource("bloom_composite")` |
| If the fed pass was the document output, the bundle's output becomes the output | LANDED | `test_import_copies_the_bundle_under_the_group_and_feeds_it`: `graph.output == "bloom_composite"` |
| `becomes_output` only when a fed pass is the host output | LANDED | `test_the_output_moves_only_when_the_fed_pass_was_the_output`; and the insertion test asserts `output == "grade"` stays |
| Takeover is explicit and per reader | LANDED | `handovers: set[tuple[str,str]]`; per-pair checkbox at `import_passes.py:240` |
| Under a fed entry-point row, the dialog lists that pass's host readers as `(pass, sampler)` checkboxes | LANDED | `_draw_handovers`, lines 232-245 |
| **…all on by default** | LANDED | `app.set_import_substitution` does `draft.handovers \|= self.host_readers_of(host_pass)`; `test_the_draft_resets_on_source_change_and_on_close` asserts `draft.handovers == app.host_readers_of("main")` |
| With no readers and the fed pass being the output, the line reads "`X` becomes the output" | LANDED (placement differs) | `_draw_handovers` prints nothing when `not readers and fed == host_output`; the line is emitted by `_draw_entry_points:217-218` as a separate note under "output: X" rather than inline on the row. Cosmetic; the text the spec names is on screen |
| Nothing is shown for `keep` | LANDED | `if fed:` guards the whole handover block, `import_passes.py:211` |
| **Two fed entry points each get their own line; both hand over to the same bundle output** | LANDED | the per-`root` loop calls `_draw_handovers` per fed row; measured on the plan: two substitutions → `handovers={'r': {'u_p': 'g_mix'}, 's': {'u_q': 'g_mix'}}`, both values the one `output`. (Edge not covered: when one host pass feeds *two* entry points, the checkbox label `f"{pair[0]}.{pair[1]}##handover"` is drawn twice, an imgui ID collision. The plan is still correct — measured — and D6 does not rule on this shape.) |
| Plan carries the result as `handovers` and `becomes_output` | LANDED | `pass_import.py:95-100` |
| Readers come from `host_wiring`; an uncompiled host pass answers explicit rows only | LANDED | `test_import_hands_the_fed_passs_readers_to_the_bundle` asserts `app.host_readers_of("main") == set()` before the import, with `grade` program-less |
| Host's program-less passes compiled when the dialog opens | LANDED | `app.select_import_source` calls `compile_pending_passes(host.document)` |
| …and again inside `import_passes` before planning | LANDED | `project_session.py:1130` |
| **A handover naming a host pass whose compile fails is rejected with a message naming it** | LANDED | `project_session.py:1131-1136`; `test_a_handover_on_a_broken_host_pass_is_rejected_by_name` asserts `"grade" in error and "compile" in error` |
| **A handover row releases the old value before overwrite** (`try_to_release`) | LANDED | `project_session.py:1175` |
| …writes `uniform_values` directly, saving once at the end rather than `set_sampler_source` per row | LANDED | direct `values[uniform] = PassSource(read)`; no `set_sampler_source` call in `import_passes` |

### D7 — the strip draws a group as one flush outline

| Claim | Verdict | Evidence |
|---|---|---|
| Tiles keep their order (`strip_order`) and their `SPACE.MD` gap | LANDED | `pass_list.py:200,213` — `same_line(spacing=float(SPACE.MD))` unchanged |
| Consecutive tiles of one group on one row form a run | LANDED | `group_runs` + the per-row `open_segment` split at `pass_list.py:210-212` |
| `group_runs(order, groups)` in `pass_graph.py`, pure, by adjacency, never by name | LANDED | `pass_graph.py:487`; `test_group_runs_cut_by_adjacency_not_by_name` |
| A group split by an outside pass is two runs | LANDED | same test: `["a"],["b","c"],["d"],["e"]` |
| The mock's 4px in-group gap is dropped | LANDED | no second gap constant; only `SPACE.MD` is used |
| `tiles_per_row` genuinely untouched (one caller) | LANDED | unchanged in the diff; one call site at `pass_list.py:199` |
| **Per run, the parent draw list gets a rounded rect inset by 1px** | DEVIATED, not recorded | the **inset** rect (`_GROUP_INSET = 1.0`) is the *outline*, drawn on the **foreground** list; the parent list gets the *fill* at full run extents with no inset (`pass_list.py:256`). The docstring explains why (child windows paint over their parent), and the spec's own reason for the inset — "an outside rect clips" on a flush row — is satisfied for the line. Not in *Deviations* |
| …in the group's tint | LANDED | `group_tint(group)` → `line`, `pass_list.py:254-255` |
| **The group name on the top border at the run's left** | LANDED | `pass_list.py:262-269`: `x = lo.x + 2*_GROUP_LABEL_PAD + _GROUP_ROUNDING`, `y = lo.y - text_size.y/2` — vertically centred on the top edge, horizontally inset from the left |
| …on a `BG_SURFACE` fill so it reads over the line | LANDED | `add_rect_filled(..., COLOR.BG_SURFACE)` at 264-268, emitted before the text |
| …inside the first tile's top rather than above the strip | LANDED | `y = lo.y - h/2` straddles the tile's top edge |
| The rect is emitted BEFORE the run's tiles so the tiles paint over it | DEVIATED, not recorded | the fill is emitted **after** the whole strip (the `for group, lo, hi in segments` loop runs past the tile loop, `pass_list.py:238-239`) — the positions are taken from `get_item_rect_min/max` of drawn tiles rather than precomputed. The tiles still show over it because a child window's own draw commands precede the later parent-list command in a different channel; the visible result is the faint fill in the gaps, as the docstring says. Not in *Deviations* |
| The label's fill is emitted after the run | LANDED | both label rects are in `_draw_group_outline`, after the tiles |
| A run that wraps is two runs with the label on each | LANDED | the `elif open_segment is not None:` row-break at 210-212 closes a segment at each row start; both reach `_draw_group_outline` |
| Member tiles draw with the group tint as `bg_color` at low alpha | LANDED | `pass_list.py:131` `bg = (*tint[:3], COLOR.GROUP_FILL_ALPHA)`, passed as `bg_color` |
| **The fill alpha** | LANDED | `theme.py:185` `GROUP_FILL_ALPHA: float = 0.10`; used for both the tile bg and the run fill |
| …and no border of their own | LANDED | `bordered=` drops `ChildFlags_.borders` |
| `preview_cell` gains `bordered: bool = True` | LANDED | `ui_primitives.py:1229` |
| …which drops `ChildFlags_.borders` from the child's flags | LANDED | `ui_primitives.py:1272-1274` |
| `bordered=False` passes `ChildFlags_.always_use_window_padding` instead | LANDED | same lines |
| …and measures byte-identical to `borders` | LANDED | `test_an_unbordered_tile_keeps_its_padding` asserts equal origins, `(8,8)` both |
| No hand-rolled pad; the cards keep their size | LANDED | no pad arithmetic added to `preview_cell` |
| **The strip passes `bordered=border is not None`** | DEVIATED, not recorded | it passes `bordered=tint is None or border is not None` (`pass_list.py:148`). The spec's literal expression would unborder **every** tile, grouped or not; the landed form unborders only grouped tiles without an accent/error border, which is what the surrounding prose describes ("Member tiles draw … no border of their own", "the accent output border and the red error border still win on a grouped tile"). Behaviour matches the intent; the expression does not match the spec's sentence, and it is not in *Deviations* |

### D8 — group tints are theme tokens picked by a stable hash

| Claim | Verdict | Evidence |
|---|---|---|
| `COLOR.GROUP_TINTS`, **four hues** | LANDED | `theme.py:178-184`, exactly four entries |
| `purple_n`, `green_b`, `yellow_n`, `aqua_n` | LANDED | the four, in that order |
| `group_tint(name) -> color` indexes by `zlib.crc32(name.encode()) % len(...)` | LANDED | `theme.py:241-244` |
| …never by `hash()` | LANDED | break-table row 4 red on 3 consecutive runs |
| **An import-time assert beside the `SELECT` invariant** | LANDED | `theme.py:223-236`, immediately after the `SELECT` assert |
| …pins no tint equals an accent primary, any `STATE_*` hue, `COLOR.SELECT`, `COLOR.TAG`, `COLOR.FAVS` | LANDED | `_GROUP_TINT_EXCLUSIONS = _accent_primaries \| {STATE_OK, STATE_WARN, STATE_ERROR, STATE_INFO, SELECT, TAG, FAVS}` — all seven named members plus the primaries |
| Accent **actives** deliberately outside the set | LANDED | `_accent_primaries` only; `aqua_n` (an active) is a tint and the assert passes |
| The GATE is a pure test over the tuple | LANDED | `test_group_tints_are_stable_and_collide_with_nothing` |
| **The widened assert set** (no duplicate) | LANDED | a second assert, `len(set(...)) == len(...)`, `theme.py:234-236`; mirrored in the test |

### D9 — the group is editable by hand and by the copilot

| Claim | Verdict | Evidence |
|---|---|---|
| The modal gains a `group` row under `name` | LANDED | `pass_settings.py:120-124` (edit mode), drawn after `_draw_name` |
| …an input with a combo of the document's existing groups | LANDED | `_draw_group` + `_existing_groups(graph)`, `pass_settings.py:175-201` |
| Committing writes `session.set_pass_group(document_id, name, group)` | LANDED | `app.commit_pass_group` → `set_pass_group` |
| A seventh verb, validated by `PASS_NAME_RE` or empty, saved like the others | LANDED | `project_session.py:1088-1101`: the check, then `save_ui_document`; registered on the backend as `pass_set_group` |
| **In create mode the row edits `draft.entry.group`** | LANDED | `pass_settings.py:89-93`: `draft.entry = draft.entry.model_copy(update={"group": group})` |
| `create_pass_from_draft` applies it through `set_pass_group` after `add_pass` | LANDED | `app.py:1130-1133`, after the `add_pass` call and beside target/iterations |
| **The tile context menu gains `Leave group` on a grouped tile** | LANDED | `pass_list.py:79-84` — guarded by `.group and imgui.menu_item_simple("Leave group")`, so it only shows for a grouped pass; calls `set_pass_group(…, "")` |
| `set_pass` gains `group: str \| None` (None = keep, `""` = leave) | LANDED | `tools/passes.py:49-52`; `backend.py:1360-1363` `if group is not None:` |
| **APPENDED to the positional-only signature** in `capabilities.py`, `backend.py`, `tools/passes.py` | LANDED | `capabilities.py:469` last before `/`; `backend.py:1404` last parameter of `set_pass`; `tools/passes.py:91` last in the forwarding call |
| The five positional test call sites lengthen | LANDED | `test_copilot_pass_tools.py` calls pass 10 positionals; full suite green |
| **`_pass_table` appends `, group <name>` to a grouped row** | LANDED | `backend.py:1294`; `test_set_pass_group_lands_and_echoes` asserts `"glow: runs 1, target f2 x1, linear, group fx"` and that `group` is absent after `group=""` |

### D10 — the import dialog is one modal in the `PopupState` mutex

| Claim | Verdict | Evidence |
|---|---|---|
| `PopupState.IMPORT_PASSES` | LANDED | `app.py:126` |
| Drawn by a new `popups/import_passes.py` with the module's own early-return guard | LANDED | `import_passes.py:46-47` |
| Imported and called in `ui.py`'s popup chain | LANDED | `ui.py:29,634`; gate re-verified red when the call is removed |
| Opened from an `import…` button beside `add pass` | LANDED | `pass_list.py:243-245`, `same_line` after `add pass`, inside the same `begin_disabled` |
| …and from a palette command `IMPORT_PASSES` | LANDED | `commands.py:44,226` |
| Chordless, palette only | LANDED | `CommandSpec(CommandId.IMPORT_PASSES, "Import passes", 0, C.TOOLS)` — chord `0` |
| A `COMMAND_SPECS` row AND an `app.command_callbacks` handler | LANDED | `commands.py:226`; `app.py:662`. Registry gate verified red with the row removed |
| **`open_import_passes` refuses while `app.copilot_turn_active`, through `_copilot_busy_blocked`** | LANDED | `app.py:1144-1145`; `test_the_busy_guard_refuses_the_palette_route` asserts CLOSED + the notification |
| **Escape reaches `close_import_passes` through its own branch in `hotkeys.py`** | LANDED | `hotkeys.py:379-380` `elif app.popup_state == PopupState.IMPORT_PASSES:`; `test_escape_reaches_the_close_funnel` |
| Transient state is an `ImportDraft` dataclass on `App`, in `ui_models.py` beside `PassDraft` | LANDED | `ui_models.py:637-652`, immediately before `PassDraft` |
| …the active tab, selected source id, group buffer | LANDED | `examples_tab`, `source_id`, `group_buf` |
| …`substitutions: dict[str, str]` keyed by entry point | LANDED | field 5 |
| …`handovers: set[tuple[str, str]]` | LANDED | field 6 |
| **…and `rejection: str`, the plan's message as of the last drawn frame (empty when valid), stored so a test can read what the button read** | LANDED | `ui_models.py:651`; written once per frame at `import_passes.py:80` from the same `plan` the button's `begin_disabled` uses; read by `test_the_plan_is_recomputed_every_frame` |
| A tab row `This project \| Examples` | LANDED | `import_passes.py:108` |
| A card grid through `draw_document_preview_button` | LANDED | `import_passes.py:135` |
| **…the current document excluded from the project tab** | LANDED | `app.import_sources`: `if document_id != self.current_document_id` |
| **The selected source's description** | LANDED | `_draw_description`, `import_passes.py:150-155`, from `source.ui_state.description`; exempted in `test_ui_prose_budget.py`'s `_UNMEASURABLE` with a reason |
| The `group` field | LANDED | `import_passes.py:74-77` |
| One row per entry point, `<name> ← [combo]` with `theirs (copy it)` first and the host's passes after | LANDED (label shortened) | `import_passes.py:186-204`; `_KEEP = "theirs"` is drawn first, host passes after. The spec's parenthetical "(copy it)" is not in the string — a label trim, not a behaviour change |
| …the entry point's readers listed beside it in the 12px face | LANDED | `caption_text(f"readers: …")` at 210 (`caption_text` is the 12px helper) |
| …and under a row fed by a host pass the D6 checkbox line | LANDED | `if fed: _draw_handovers(...)` at 211-212 |
| A note naming the bundle's output and whether it becomes the document's | LANDED | `import_passes.py:216-218` |
| **A note that the source's script is not imported** | LANDED | `import_passes.py:221-222`, shown when `_has_script(source)` finds `scripts/script.py` |
| `Import N passes` (disabled with the rejection in red while `plan_import` rejects) | LANDED | `import_passes.py:84-98`: `begin_disabled(isinstance(plan, str))`, label `f"Import {len(plan.renames)} passes"`, `text_colored(COLOR.STATE_ERROR, plan)` |
| …and `Cancel` | LANDED | `import_passes.py:94` |
| **No field takes keyboard focus on open or on selection** | LANDED | no `set_keyboard_focus_here` anywhere in the module (grep clean); the module docstring states the rule |
| The group field is focused by a click | LANDED | imgui default for an unfocused `input_text` |
| Selecting a source compiles its program-less passes and the host's | LANDED | `app.select_import_source` calls `compile_pending_passes` on both |
| …resets the group buffer to D2's slug | LANDED | `group_buf = group_slug(source.ui_state.ui_name)` |
| …the substitutions to `keep` and the handovers to empty | LANDED | `draft.substitutions = {}`, `draft.handovers = set()`; `test_the_draft_resets_on_source_change_and_on_close` |
| Picking a host pass resets that entry point's handovers to all of that pass's readers | LANDED | `app.set_import_substitution` removes the previous fed pass's readers and unions the new pass's; same test |
| **The plan is recomputed each frame the dialog draws, never cached on selection** | LANDED | `_plan(draft, source, host)` is called inside `_draw_body`, no memo; `test_the_plan_is_recomputed_every_frame` |

One extra field landed: `ImportDraft.tab_select_pending` — recorded in *Deviations* ("The tab
bar needs a one-shot `set_selected`"), with the reason and how it was found.

### D11 — the planned render set follows the open tab

| Claim | Verdict | Evidence |
|---|---|---|
| **Two predicates, not one and its negation** | LANDED | `ui.py:167-177` `planned_set_mode` returns `(examples_planned, import_project_tab)` |
| `examples_planned` is "the Examples popup, or the import dialog with its Examples tab active" | LANDED | `ui.py:173-175`, exactly that disjunction |
| `import_project_tab` is "the import dialog with its project tab active" | LANDED | `ui.py:176` `importing and not examples_planned` |
| `_tick_frame_state`'s `examples_open` becomes `examples_planned` | LANDED | the local is gone; `planned_documents`/`planned`/`current_planned` all read `examples_planned` (`ui.py:291-299`) |
| …and they follow it unchanged (090 D10) | LANDED | the three expressions are otherwise byte-identical in the diff |
| The render chain gets NO new branch | LANDED | `elif app.popup_state == PopupState.EXAMPLES:` → `elif examples_planned:`, `ui.py:496`; no added branch |
| The `if not any_popup_open():` branch takes `or import_project_tab` | LANDED | `ui.py:470` |
| …never `not examples_planned` | LANDED | the negation appears nowhere in either gate |
| **The same `or import_project_tab` on `_tick_frame_state`'s own `if not any_popup_open():` gate** | LANDED | `ui.py:265-266` — the second of the **two** required sites |
| Both gates carry the predicate | LANDED | break-table row 3: removing it from the `_tick_frame_state` gate alone turns `test_the_import_dialog_plans_the_open_tabs_documents` red |

---

## B. Per-item verification audit

All files below were run: `test_pass_import.py`, `test_pass_graph.py`,
`test_pass_strip_layout.py`, `test_theme.py`, `test_graph_persistence.py` (69 passed);
`test_pass_verbs.py`, `test_import_dialog.py`, `test_copilot_pass_tools.py` (56 passed);
`test_render_decoupling_loop.py`, `test_project_management.py`,
`test_command_registry_coverage.py`, `test_ui_prose_budget.py` (665 passed, 4 skipped).

| # | Test | Verdict |
|---|---|---|
| 1 | `test_pass_graph.py::test_entry_points_are_the_passes_reading_no_other_pass` | **FULL** — all five shapes (bloom, the standalone self-read `{"acc":{"u_prev":"acc"}}`, RC-shaped via `{"lone":{}}`, the two-input compositor, a no-entry pass); the falsifier (`[]` for `acc`) is the asserted value |
| 2 | `test_pass_graph.py::test_group_slug_is_the_first_word_made_legal` | **FULL** — the four D2 inputs, plus a loop asserting `PASS_NAME_RE` matches every output, which is the item's falsifier |
| 3 | `test_pass_import.py`, 8 tests | **PARTIAL on (d)** — (a) `test_every_wired_sampler_is_explicit…` incl. `PassSource("bloom_trail")`; (b) `test_a_fed_entry_point_is_not_copied…`; (c) `test_a_colliding_name_rejects_and_names_it`; (e) `test_an_empty_group_copies_under_bare_names`; (f) `test_a_handover_rewires…` incl. the unchecked case; (g) `test_the_output_moves_only_when…`; (h) `test_a_source_pass_with_no_edges_still_plans`. **(d)** `test_replacing_the_only_pass_rejects` asserts `"output" in plan`, i.e. the output guard's message, not "nothing to import" — see F2. Falsifier for (a) verified red (break 1) |
| 4 | `test_pass_verbs.py::test_a_rendered_import_reads_its_bundle` | **FULL** — DEVIATED home, recorded: lives in `test_pass_verbs.py` under the `app` fixture, not `test_document_graph.py`'s `gl_ctx`, with the reason. Host `main` renders 0.5 red, a `halve` source, output asserted `60 <= red <= 68` (a quarter of 255). The named falsifier (drop materialization → black) verified red: `assert 60 <= 0` |
| 5 | `test_pass_verbs.py::test_import_copies_the_bundle_under_the_group_and_feeds_it` + `::test_import_hands_the_fed_passs_readers_to_the_bundle` + `::test_a_handover_on_a_broken_host_pass_is_rejected_by_name` | **FULL** — append case: the bloom fixture copied into `tmp_path`, loaded with `load_document_from_dir`, the four files beside the host's, the four `group == "bloom"` entries, all three named rows (`bloom_bright.u_scene=={"pass":"main"}`, `bloom_composite.u_blur`, `bloom_trail.u_prev`), output `bloom_composite`, reload via `_reload`. Insertion case: `grade` built as a bare `Pass` never through `add_pass` — **5(i)'s exact route** — with `assert app.host_readers_of("main") == set()` pinning that the reader is invisible uncompiled; after reload `grade.u_main == PassSource("bloom_composite")` and output still `grade`. **5(ii)**: the broken-host-pass rejection names it. Falsifier (i) verified red (break 6). The source-textures-after-`release()` assertion of item 5 is **not** present — covered instead by item 6's byte-identity test and `_copied_uniform_value`; a gap in letter, not in effect |
| 6 | `test_pass_verbs.py::test_import_leaves_the_shipped_example_byte_identical` | **FULL** — `rglob("*")` byte map before/after, which is stronger than the item's three named files |
| 7 | `test_pass_verbs.py::test_a_merged_ui_row_survives_the_save` | **FULL** — a non-default `input_type="text"` on the source's `u_threshold` row, keyed by `get_uniform_hash`, read back after `_reload` under the same key |
| 8 | `test_pass_verbs.py::test_a_broken_source_pass_is_imported_as_is_and_named` | **FULL** — both falsifiers: `offered_entry_points(...) == ["scene"]` (break 5 turns it `['blur','scene']`, the item's exact wording) and `any("blur" in note …)`; the other passes land |
| 9 | `test_pass_verbs.py::test_the_group_survives_rename_and_goes_with_delete` + `test_graph_persistence.py::test_a_group_round_trips_and_an_absent_key_reads_empty` | **FULL** — split by fixture as the item says; rename keeps `fx` through a reload, delete drops the pass, an absent key reads `""`, and a `ValidationError` on `group="2bad"` as a bonus |
| 10 | `test_pass_strip_layout.py::test_group_runs_cut_by_adjacency_not_by_name` | **FULL** — all three shapes; the named falsifier (name-keyed merge) is what the split assertion forbids |
| 11 | `test_theme.py::test_group_tints_are_stable_and_collide_with_nothing` | **FULL** — (a) three names pinned by **value** at literal indices (`is COLOR.GROUP_TINTS[3]` etc.); (b) disjointness from accent primaries (via `_ACCENTS`), four `STATE_*`, `SELECT`, `TAG`, `FAVS`, plus the no-duplicate assert. (b)'s falsifier is `purple_b == SELECT`, which the `SELECT` member of the set catches. Break 4 red on 3 runs |
| 12 | `test_copilot_pass_tools.py::test_set_pass_group_lands_and_echoes` | **FULL** — all three clauses: `group fx` in the echoed table, `group=""` clears it (asserted absent from `glow`'s row), `"2bad"` is an error not a no-op |
| 13 | `test_command_registry_coverage.py` | **FULL** — verified red in a worktree with the `COMMAND_SPECS` row removed: `assert set(SPEC_BY_ID) == set(CommandId)` names `CommandId.IMPORT_PASSES` |
| 14 | `test_project_management.py::test_every_popup_state_has_a_draw_call` | **FULL** — verified red with `draw_import_passes(app)` removed from `ui.py`: "imported from shaderbox.popups but never called in ui.py: ['draw_import_passes']". (The gate reads `shaderbox/ui.py` by **relative** path, so it must be run with cwd inside the tree under test — my first attempt parsed the main tree and passed spuriously) |
| 15 | `test_import_dialog.py::test_the_draft_resets_on_source_change_and_on_close` + `::test_escape_reaches_the_close_funnel` | **FULL** — DEVIATED home (the item names `test_pass_draft.py`; it landed in the new `test_import_dialog.py`, not recorded). Every clause: select → slug + empty decisions; set handovers; select a second source → handovers empty and the second slug; re-pick a host pass → that pass's readers; `close_import_passes()` → draft `None`; reopen → initial state. The `hotkeys.py` substring assert is its own test, the shape the item names |
| 16 | `test_import_dialog.py::test_the_busy_guard_refuses_the_palette_route` | **FULL** — DEVIATED home (same). `copilot_turn_active = True` → `popup_state` CLOSED, draft `None`, `"locked"` in the notification |
| 17 | `test_import_dialog.py::test_the_plan_is_recomputed_every_frame` | **FULL** — DEVIATED home (the item names `test_pass_settings_layout.py`). Group `2bad` → pumped frames → `"group name" in draft.rejection`; buffer to `ok` with no source change → pumped frames → `rejection == ""`. Pumps twice rather than once, which the `tab_select_pending` deviation explains |
| 18 | `test_import_dialog.py::test_an_unbordered_tile_keeps_its_padding` | **FULL** — DEVIATED home (same). Spies `get_cursor_screen_pos`, asserts both cells' content origins equal **and** `!= (0,0)`. Break 2 red with `assert (8.0, 8.0) == (0.0, 0.0)` — the item's measured numbers exactly |
| 19 | `scripts/smoke.py` (frame 42) | **PARTIAL** — the stamp lands by name on `paint` and `df` (recorded in *Deviations*), and I verified the non-adjacency claim holds in **both** orderings: uncompiled (name-sorted) `['cascade','composite','df','jfa','paint','seed']`, compiled (topological) `['paint','seed','jfa','df','cascade','composite']`. So the **split-run** path and the unbordered member tiles execute. But both runs are then length **1**, so the smoke never draws a multi-tile run — the outline spanning two or more tiles, which is what D7 is for, is exercised only by the maintainer's manual walk. Also not verified by me: smoke did not run here (`make gates` stopped at `check`; a display-less run reports *skipped*, which is not a pass) |
| 20 | `test_render_decoupling_loop.py::test_the_import_dialog_plans_the_open_tabs_documents` | **FULL** — Examples tab: examples render, the other project document does not; project tab: the other document renders. Break 3 red with the item's own message ("the project tab did not render the other document"). The "pending-first election runs" half is asserted as the render count rather than the election directly — covered, in effect |

Items 15–18's homes differ from the spec's named files (all four consolidated into the new
`tests/test_import_dialog.py`). Not listed under *Deviations*; the tests themselves are
present and assert what the items say, so this is a bookkeeping gap.

---

## C. The break table

Each break applied in `git worktree add /tmp/091-audit HEAD`, the named test run from inside
the worktree, then reverted; the worktree was removed at the end and the main tree verified
clean at `4dc1423`. **All six go red.**

| # | Break | Named test | Result |
|---|---|---|---|
| 1 | `plan_import` never fills `sources` | `test_pass_import.py` (3 tests) + `test_a_rendered_import_reads_its_bundle` | **RED** — but **2** of the `test_pass_import.py` tests, not 3 (F3). The render test red with `assert 60 <= 0`, i.e. black, as claimed |
| 2 | `preview_cell(bordered=False)` passes `ChildFlags_.none` | `test_an_unbordered_tile_keeps_its_padding` | **RED** — `assert (8.0, 8.0) == (0.0, 0.0)`, the claimed origin |
| 3 | `_tick_frame_state`'s gate without `or import_project_tab` | `test_the_import_dialog_plans_the_open_tabs_documents` | **RED** — "the project tab did not render the other document" |
| 4 | `group_tint` by `hash()` | `test_group_tints_are_stable_and_collide_with_nothing` | **RED** on 3 consecutive runs, confirming "red on essentially every run" rather than flaky |
| 5 | `offered_entry_points` keeps a broken pass | `test_a_broken_source_pass_is_imported_as_is_and_named` | **RED** — `assert ['blur', 'scene'] == ['scene']`, the spec's exact falsifier |
| 6 | `import_passes` does not compile the host | `test_import_hands_the_fed_passs_readers_to_the_bundle` | **RED** — `"'grade.u_main' does not read a replaced pass"`, the rejection the uncompiled wiring forces |

A caveat on method, since it nearly produced a false PASS: `PYTHONPATH=/tmp/091-audit` alone
does **not** win against the project's editable install (`python -c "import
shaderbox.pass_import"` resolves to the main tree). What makes the worktree win is pytest
collecting from `/tmp/091-audit/tests/`, which inserts the worktree's rootdir ahead of it —
confirmed by a probe module placed in the worktree's `tests/` printing
`/tmp/091-audit/shaderbox/pass_import.py`. Every red above was produced that way. Tests that
read a source file by **relative** path (item 14's AST gate, item 15's `hotkeys.py` substring)
are immune to both and must be run with cwd inside the tree under test.

---

## D. Out of scope — nothing landed

| Item | Checked | Result |
|---|---|---|
| A cross-project presets folder | `grep -rn presets shaderbox/ scripts/` | Absent. The only hits are pre-existing canvas-size presets (`tabs/document.py`, `theme.py`'s accent presets) |
| Export a group as a document | `grep export_group\|export_passes\|group_to_document` | Absent |
| Folding a group | `grep -ni fold` in `pass_list.py`, `pass_graph.py` | Absent. `group_runs` draws the split case the fold would have special-cased |
| A second graph view | no new view module in the diff | Absent |
| A copilot `import_passes` tool | `grep import_passes shaderbox/copilot/` | Absent. Only `set_pass`'s `group` parameter landed, which D9 scopes in |
| A `preset:` read prefix | `grep '"preset:'` | Absent |
| A group rename verb | `grep rename_group\|set_group_name` | Absent. `set_pass_group` is per pass |
| Nested groups | `grep parent_group\|subgroup\|nested` in `pass_graph.py` | Absent. `group: str` is flat |
| Importing the source's script | the dialog's `_has_script` only **prints** "script not imported" | Correct; no script copy in `import_passes` |
| Importing feedback seeds | no `_feedback` write in `import_passes` | Absent |

Docs promised in *Files touched* all landed: `conventions.md` gains the three bullets (the
label-on-the-entry with the fold rejection, import-by-copy with substitution and insertion,
stable-hash tints), each in "we decided X; revisit if Y" form; `dev_flow.md ### Module map`
gains `pass_import.py`, `popups/import_passes.py` and the strip's group outline;
`roadmap.md` has the 091 row and banner.

---

## False trails

Things that looked like findings and were not — do not re-check these.

- **`git status` clean after a red `make gates`.** The first `check` run reported `pyright …
  Failed / files were modified by this hook` with a clean tree. That is a pre-commit
  run-order artifact, not a code rewrite. The **second** run surfaced the real failure (ruff,
  F1). A single `make check` run is not enough to characterise which hook modified what.
- **`test_every_popup_state_has_a_draw_call` passing with the draw call deleted.** My first
  break run used `PYTHONPATH` with cwd in the main repo; the gate reads `shaderbox/ui.py` by
  relative path, so it parsed the unbroken main tree. Run from inside the worktree it is red.
  The gate is sound.
- **`_copied_uniform_value`'s `return value` fall-through reading as the "no default branch"
  D5 forbids.** It is what carries scalars, tuples and the three frozen `SamplerSource`
  members by value, which D5 explicitly wants. Not a hole.
- **`Image(value.texture)` appearing in `_copied_uniform_value`, which D5 says "never".** It
  is the fallback for a `MediaWithTexture` with an **empty** path — no file to re-open, so the
  prohibition's reason (losing `file_details`) does not apply. The path-bearing case re-opens
  from the file via `media_class_for`.
- **The group outline's fill not being inset.** The 1px inset belongs to the outline, which is
  where the spec's stated reason (a flush row clipping an outside rect) bites. The fill is
  clipped by the parent window anyway.
- **`ImportDraft.tab_select_pending` as an undeclared field.** It is the first
  *Deviations* bullet, with its cause and how it was found.
- **Two fed entry points.** The plan handles both the different-host-pass and same-host-pass
  shapes correctly (measured). Only the UI's checkbox ID would collide in the
  one-pass-feeds-two-entry-points case, which no decision rules on.
- **D1's "two sites compare against `PassEntry()`'s field defaults".** `app.py:1121,1125`
  compare `.target` and `.iterations` field-wise, never the whole entry, so the new field
  cannot make a default entry read as non-default.
