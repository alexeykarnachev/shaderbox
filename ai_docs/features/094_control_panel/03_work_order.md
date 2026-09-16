# 094 — work order

The commit sequence for `01_spec.md`, produced by the round-4 readiness reviewer and verified
against the tree. Fourteen commits. Each leaves `make gates` green except C10, which names its own
atomic unit.

Judge every gate by its exit code captured unpiped: `make gates > /tmp/g.log 2>&1; echo $?`. A
skipped smoke is not a pass.

---

## The sequence

| # | What | Files | Spec | Checks it makes passable |
|---|---|---:|---|---|
| C1 | `render(canvas=, target=)` composes; `render_media` takes a target | 2 | D12 | 8 |
| C2 | The four observation seams (`planned_documents`, `row_rects`) | 3 | D19 | — (makes 3 and 5b expressible) |
| C3 | `draw_ui_uniform` and friends take the pass | 6 | D4a | 21 (assertable at C8) |
| C4 | `get_uniform_hash` takes the pass name; the data re-key | 16 | D4d | — (makes 7, 21 meaningful) |
| C5 | The horizontal splitter | 6 | D1a/D1b/D1c | 16, 17 |
| C6 | One fps number; the profiler panel dies | 7 | D13a/D13b | 18, 19 |
| C7 | Esc learns the focus mode; the focus state lands | 3 | D8a, D9b, D6 | 12a, 12b |
| C8 | The node draws uniform rows | 6 | D4/D4b/D4c/D4e/D5/D5a/D6/D10a | 7, 21, 22 |
| C9 | The focus mode: scrim, camera, refusals | 7 | D7/D9–D11c/D12a | 4, 5a, 5b, 6, 9, 10, 11, 23 |
| **C10** | **The graph becomes the panel** | **17** | D2/D2a/D13/D13c/D14–D17 | 14, 20, 24, 25 |
| C11 | The commands and the tab kind go | 10 | D3/D18/D18a | 1, 13, 15 |
| C12 | `Render all documents` is deleted | 5 | D15, D16 | 2, 3 |
| C13 | The region-system test's anchor | 1 | — | — |
| C14 | Conventions, copilot strings, roadmap | 6 | — | 26 |

Totals: 28 source files, 19 tests, 7 data files, 3 modules deleted.

## The three ordering constraints (D20), plus one

1. **D4a and D4d before D4** (C3, C4 before C8). Rows drawn before the pass is an argument write the
   wrong pass; two passes' rows drawn before the hash carries the pass name share one `input_type`.
   Both look right on screen.
2. **D2a with D3** (inside C10, before C11). Deleting the graph tab while both `begin_disabled`
   brackets stand leaves the graph permanently on screen AND frozen for every copilot turn, with the
   breadcrumb as the only document switcher.
3. **D16 before D15** (C10 before C12). Deleting `is_render_all_documents` before the dropdown exists
   means no non-current document renders at all, with the grid already gone.
4. **C11 and C12 are a PAIR.** Between them the commands are gone and the dropdown exists while
   `is_render_all_documents` still gates the set — gates green, behaviour wrong, which is worse
   than red.

## C10 is atomic

`pass_list.py` is imported by `pass_graph.py` and four test files; `tabs/document.py` and
`tabs/uniforms.py` by `ui.py`'s `_NODE_TABS` and three test files; `scripts/smoke.py` reads
`DocumentTab`, `active_document_tab`, `open_graph_for` and the `"graph"` tab kind, and smoke is the
third leg of `make gates`. The minimum unit is: delete the three modules + rewrite `_draw_app_panel`
+ relocate `pass_menu_items` and `canvas_choice_*` + rewrite smoke's sweep + re-point seven test
files. Any split leaves an `ImportError` at collection, which reds every gate at once.

If work stops mid-C10, recover with `git reset --hard` to C9 rather than a forward fix.

## Risk order, and the headless signal for each

1. **C10.** Half its failure modes are invisible to pytest and only smoke sees them. Signals in
   order: `make check` catches a half-relocated import in seconds; `test_menus` + `test_graph_view`
   catch a wrong bracket as check 14; the smoke leg must read `smoke passed`, never `smoke skipped`.
   **The silent one:** `_draw_document_image` pushes `StyleVar_.alpha = 1.0` because the outer
   bracket scales alpha. Move the bracket without it and the preview renders dim — no headless
   signal at all. Grep for it and say in the commit body that it was re-checked.
2. **C8.** Three mechanics that fail silently (channel, overlap, `node_size`). Run check 22 first —
   it is pure and fast. `test_graph_view` is the port-drag canary: rows overlapping ports kill the
   drop target exactly as the code comment warns. **Measure the fitted zoom** for a 3-pass and a
   6-pass document after the width change and compare against the near-LOD threshold before
   trusting the proposed 0.75 — a taller node in a half-height region may fit below it.
3. **C4.** The only commit that edits data a user would lose. Compare row COUNTS before and after a
   load+save, not "it loaded" — the prune deletes what it cannot match, so a wrong key looks clean.
4. **C6.** Write check 19 BEFORE the deletion, watch it pass, delete, watch it still pass, then try
   its falsifier (drop ONE `profiler=` argument) and restore it. An unbroken gate is a wish, and this
   one guards a deletion.
5. **C9.** No headless signal for paint order. Proxy: assert the channel constants are a contiguous
   range and that `channels_split`'s argument equals `max + 1`.
6. **C12.** Check 2's falsifier is INERT unless a second document is seeded and warmed to
   `first_render_done`. Run the falsifier explicitly; if it does not fire, the fixture is wrong.

## Rollback boundaries

**Shippable** — the tree works, nothing is lost, the feature is merely unfinished: after C1, C2,
**C3** (the best stopping point — a pure refactor banking the hardest prerequisite), C4, C5, C6, C7,
**C8** (the largest user-visible win, graph still a tab), C9, C11, C12, C13, C14.

**Do not stop:** inside C4 (rows stranded between the signature change and the re-key), inside C10
(the tree does not import), between C11 and C12 (green and wrong).

## Two things to report rather than decide silently

1. **The LOD thresholds against the new region.** 0.75/0.4 were reasoned before D4e widened the node.
   Measure the fitted zoom for a 3-pass and a 6-pass document at C8 and report both numbers.
2. **`tests/test_theme.py`'s throttle-band tests.** Whether they die with `profile_rows_plan` or
   re-point at a surviving band function depends on whether the thresholds exist independently. A
   five-minute read at C6; state the call in the commit body either way.
