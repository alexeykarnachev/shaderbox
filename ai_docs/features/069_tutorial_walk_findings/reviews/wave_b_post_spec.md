# 069 W-B — post-implementation review: spec fidelity and architecture

Reviewing `ccd446b` ("069 W-B: cut UI prose to budget and gate it") against
`30_wave_b_prose_diet.md`, `01_spec.md § W-B` + D1, and `00_findings.md` #5 #7 #10 #32.

## Verdict

| Dimension | Verdict |
|---|---|
| Wave-spec fidelity | **PASS** |
| Parent fidelity (`01_spec.md § W-B`, D1) | **PASS** |
| Findings closure (#5 #7 #10 #32) | **PASS** |
| Census (re-run independently) | **PASS** — every figure reproduces exactly |
| Architecture | **PARTIAL** — one enumerable domain hole in `_SCORED` (finding 2) |
| Docs | **PARTIAL** — D1 is not discoverable from any in-repo doc (finding 3) |

Overall: **PARTIAL**. The wave lands its spec almost cell for cell and the census
reproduces under an independent collector. Two things hold it short of PASS: the commit
is red at `make check` on a ruff SIM102 in its own new test file (finding 1), and the
gate carries the exact defect class its § 1 claims four self-checks against — three
`_SCORED` rows are positionally reachable but read by keyword only (finding 2).

---

## Census: re-run independently

I wrote a throwaway collector from the spec's §§ 2-5 prose (not from the shipped test),
in the scratchpad, and ran it against clean `git archive` checkouts of `ccd446b` and
`ccd446b^`. It implements the same call set, the same positional-or-keyword read, the
same `Constant` / `JoinedStr` / `IfExp` / sequence scoring, the same `AugAssign` guard
and worst-candidate `Name` resolution, and the same clause joiners.

| Figure | Commit message claims | My collector | Verdict |
|---|---|---|---|
| sites, pre-cut | 106 | 106 | matches |
| measurable, pre-cut | 71 | 71 | matches |
| over budget, pre-cut | 17 | 17 | matches |
| clause failures, pre-cut | 9 | 9 | matches |
| distinct violating, pre-cut | 18 | 18 | matches |
| sites, post-cut | 103 | 103 | matches |
| measurable, post-cut | 68 | 68 | matches |
| unmeasurable, post-cut | 35 | 35 | matches |
| violations outside the exemptions, post-cut | 0 | 0 | matches |

Allowlist reconciliation, both directions: the 29 keys my collector produces as
unmeasurable are **exactly** `_UNMEASURABLE`'s 29 keys (`in gate, not in mine: []`;
`in mine, not in gate: []`). My 5 distinct post-cut violations are exactly
`_OVER_BUDGET`'s 5 keys, at exactly the recorded counts (telegram 9, youtube 4, help 15,
lib_picker 11, settings 3). No row on either side is unaccounted for.

The one discrepancy I hit was mine, not the commit's: a first run against the live
working tree scored 104 sites and an extra `ui_primitives.py::badge_chip` key. That
function is a probe another agent had transiently in the tree; it is absent from
`ccd446b` and from the tree now. Re-running against a clean checkout gave 103.

The spec's own census tables also reconcile: parsing every `_UNMEASURABLE` verdict cell
out of §§ The census yields the same 29 keys as the shipped dict, modulo the
`_draw_auto_row` -> `_draw_auto_block` rename § 10 mandates. So the spec's tables were
generated as § The census claims, not transcribed.

---

## Coverage: design decisions

| # | Decision | Status | Evidence |
|---|---|---|---|
| 1 | Gate is one new test module on the `test_worker_daemon_contract.py` idiom | landed | `tests/test_ui_prose_budget.py:304` `_collect()` module-level, `:366` `_SITES`, `:393` parametrize, `:380` floor test |
| 2 | Call set = six + keyword-taking helpers; positional OR keyword read; neither = UNMEASURABLE | landed | `:35-52` `_SCORED` carries all 16 rows of the spec table verbatim; `:321-331` reads positional then keyword; `:356` `_supplies` distinguishes omitted from unreadable. **Partial**, see finding 2 |
| 2a | Match by `Name.id` or `Attribute.attr`, module qualifier unchecked | landed | `:243-248` `_call_name` |
| 2b | Label helpers read at index **1**, pinned by `inspect.signature` | landed | `:39-40` index 1; `:530` `test_the_label_helpers_are_read_at_the_right_argument` asserts param 0 is `font` |
| 3 | Score over `Constant` / `JoinedStr` / `IfExp` worst branch / sequence worst; clause check | landed | `:166-190` `_score`; `:54` `_CLAUSE_JOINERS` = `";"`, `" — "`, `" -- "`; `:402` clause test |
| 4 | `Name` resolved to a straight-line string assignment; `AugAssign` -> unmeasurable; multiple -> worst | landed | `:210-240` `_resolve`; `:222-228` AugAssign guard; `:240` `max(..., key=score)` |
| 5 | Two dicts, keyed `(module, function)`, in the test module, both anti-rot tested | landed | `:58` `_UNMEASURABLE`, `:117` `_OVER_BUDGET` (key carries the count); `:442` + `:451` the two anti-rot tests |
| 6 | Label carries no `FormattedValue` at all, as its own assertion | landed | `:415` `test_no_label_carries_an_interpolation`, separate from `:394` |
| 7 | `always_auto_resize \| no_scrollbar`, height `0.0`, `SIZE.PASS_SETTINGS_H` deleted | landed | `popups/pass_settings.py:49-51`; `theme.py` diff removes `PASS_SETTINGS_H`, `PASS_SETTINGS_W: int = 440` stays |
| 8 | Label `size`, slider format `%.0f%% · {w}x{h}`, ASCII `x` | landed | `pass_settings.py:167` `label_row(app.font_12, "size", ...)`, `:176` `f"%.0f%% · {w}x{h}"` |
| 9 | `Reads (?)` -> `Reads`, header tooltip block deleted, empty state four words | landed | `pass_settings.py:124` `separator_text("Reads")`; the `is_item_hovered` / `set_tooltip` block is gone from the diff; `:127` `"no sampler2D uniforms"` |
| 9a | New `HelpSection` carrying the four facts | landed, **with the recorded key deviation** | `help_content.py:170-188`, `key="pass_settings"`, `title="Passes"`, body verbatim from the spec's snippet |
| 9b | `(see Runs per frame)` -> `(see Runs)` | landed | `help_content.py:42` |
| 10 | `_draw_auto_row` -> `_draw_auto_block`, vertical, fixed `SIZE.AUTO_NAME_W`, through `uniform_name_label`, guard stays, call moves below the `dummy` | landed | `tabs/document.py:95` rename; `:104-110` fixed column + `same_line(AUTO_NAME_W + SPACE.MD)`; `:241-245` the `dummy` now precedes the guarded call, `same_line(spacing=XL)` gone |
| 11 | Unambiguous markers deleted, survivors keep the what-clause, tooltips become names | landed | `sampling` and `edges` markers deleted outright (diff); survivors are 6, 7 and 7 words |

## Coverage: census replacement strings

Every replacement string in §§ The census landed **verbatim**. Checked by exact-literal
grep over `shaderbox/`, one hit each:

`names its shader file and its wires` / `no sampler2D uniforms` /
`share of the canvas, output always full` / `redraws per frame, each reading the last` /
`Pass settings` / `Open the document script` / `Create the document script` /
`Stop the whole script` / `Resume the whole script` / `Whole script is stopped` /
`Stop this uniform` / `Resume this uniform` / `clamps to 0-1, the smallest` /
`holds values above 1, the default` / `full precision, twice the memory`.

`separator_text` labels: `Pass`, `Reads`, `Draws into`, `Runs` — the two unchanged rows
unchanged, the two cut rows cut. No drift on any cell.

## Coverage: files touched

| Spec row | Status |
|---|---|
| `popups/pass_settings.py` | landed, every clause of the row |
| `widgets/pass_list.py` | landed (`:98`) |
| `tabs/document.py` | landed |
| `widgets/uniform.py` | landed (`:161-167`, three branches) |
| `help_content.py` | landed; the spec's Files row and its "IS touched" paragraph both say so |
| `theme.py` | landed; row names both `PASS_SETTINGS_H` deletion and `AUTO_NAME_W` addition |
| `tests/test_ui_prose_budget.py` | landed |
| `.claude/skills/imgui-ui/SKILL.md` | landed (§ 2, 5 lines) |
| `ui_primitives.py` **not** touched | held — the diff does not touch it |

## Coverage: tests

All 14 spec'd tests exist; `pytest tests/test_ui_prose_budget.py` is 222 passed. The
spec's one naming split: `test_every_allowlist_entry_still_names_a_real_site` ships as
two tests (`test_every_unmeasurable_entry_still_names_a_real_site:442`,
`test_every_over_budget_entry_still_names_a_real_site:451`), which is the same contract
parametrized per dict rather than one test over both — not a deviation of substance.

| Test | Status | Line |
|---|---|---|
| `test_the_walk_finds_the_known_call_sites` | landed, floor 60 as spec'd | `:380` |
| `test_every_measured_site_is_within_budget` | landed | `:394` |
| `test_a_keyword_supplied_argument_is_scored` | landed | `:465` |
| `test_an_argument_supplied_by_neither_position_nor_keyword_is_unmeasurable` | landed | `:470` |
| `test_no_scored_string_joins_a_second_clause` | landed | `:402` |
| `test_an_ifexp_argument_scores_its_worst_branch` | landed | `:475` |
| `test_no_label_carries_an_interpolation` | landed | `:415` |
| `test_the_label_helpers_are_read_at_the_right_argument` | landed | `:530` |
| `test_a_name_rebound_by_augassign_is_not_resolved` | landed | `:480` |
| `test_no_site_is_both_measured_and_unmeasurable_listed` | landed | `:431` |
| `test_every_unmeasurable_site_is_listed` | landed | `:423` |
| `test_every_allowlist_entry_still_names_a_real_site` | landed as two | `:442`, `:451` |
| `test_the_format_tooltips_are_within_the_help_budget` | landed, also asserts the clause check (spec asked count only — a strengthening) | `:546` |
| `test_the_auto_name_column_fits_every_engine_uniform` | landed, bound carried as `AUTO_NAME_W // (12*1118/2048)` = 19 rather than a bare literal, as § spec'd | `:556` |

I exercised the collector directly on the shapes the spec names, through the real module:

- `help_marker(*parts)` -> one row, UNMEASURABLE (not skipped).
- `help_marker(**kw)` -> one row, UNMEASURABLE.
- `gauge_bar(a, b)` (optional tooltip omitted) -> no row. Correct per § 2.
- `clickable_label('a b c', tooltip='w x y z v u')` -> two rows, 3 and 6.
- `preview_cell(sublines=['ok', 'a b c d e f'])` -> one row scoring 6 (worst element).

## Coverage: manual steps

Twelve steps, all requiring a live window. This box has no display (`make gates` reports
smoke as skipped), so I verified each step's *code precondition* from the diff rather
than the rendered frame. Steps 1-2 (auto-resize): the flag reaches `begin_popup_modal`
through `modal_window`'s `flags` parameter (`ui_primitives.py:306`), and
`always_auto_resize` is "Resize every window to its content every frame" in the installed
stub (`imgui_bundle/imgui/__init__.pyi:3270`). Steps 3-4 (format string): landed at
`:176`, inside `begin_disabled` as spec'd. Steps 5-7 (markers): three survivors (name,
format, size), the two deletions confirmed absent. Step 8: `:98`. Steps 9-11 (block):
landed. Step 12 (entry-point tooltip): landed, two branches.

**The rendered frame remains unverified** — that is the spec's own position (§ "The
gear's auto-resize and the size row's format: manual"), and step 2 in particular tests a
precedence claim (`always_auto_resize` vs an ini-restored rect) the spec explicitly flags
as not traced to the stub. It stays open for the maintainer.

## Findings closure

**#5** ("doesn't fit the default popup size … Strip this text 2x … the text under (?) is
absurd"): closed by `pass_settings.py:127` (empty state 10 -> 3 words),
`:124` + the deleted `set_tooltip` block (header tooltip 31 -> 0), and the six marker
cuts. **Recurrence:** blocked. A new over-budget `help_marker` in this file is red at
`test_every_measured_site_is_within_budget`, and the maintainer's "this is a class, not
one string" reading is what the gate encodes.

**#7** ("'size (1080, 1080)' doesn't fit … overlaps with the scroll bar"): closed on both
halves — the label at `:167`, the scrollbar at `:49-51`. **Recurrence:** the label half is
blocked twice over (word count and `test_no_label_carries_an_interpolation`, which fires
even on a one-word `f"{n}"`). The scrollbar half is **not** gated — no test asserts the
flag, by the spec's own reasoning that a window rect exists only in a live frame. A future
edit could drop `always_auto_resize` silently. That is a knowing trade, recorded, not a
defect.

**#10** ("'Pass settings' is enough"): closed at `:98`, the maintainer's words verbatim.
**Recurrence:** blocked, `set_tooltip` budget 5.

**#32** ("don't fit the panel. Align them vertically, under the sort picker (outside the
sorting)"): closed by `tabs/document.py:95-112` + `:241-245`. All three clauses of the
maintainer's sentence are satisfied: vertical, under the sort picker, and outside the
sorting (`auto_hashes` is still partitioned away before `sort_uniform_hashes`).
**Recurrence:** partly blocked — `test_the_auto_name_column_fits_every_engine_uniform`
catches the exact regression that caused #32 (068 added two names to a row sized for
three), at the right bound and against the right failure mode (`clickable_label`
ellipsizes, so an over-wide name is a silent truncation, not an overlap). The panel-width
half is not gated, which the spec says.

---

## Findings

### 1. The commit is red at `make check`: a ruff SIM102 in its own new test file

**Claim.** `ccd446b` does not pass `make gates`. `make check` fails on the new gate
module, so the wave shipped a red tree — and the project rule is "Run `make gates` before
declaring anything done", judged by the exit code captured unpiped.

**Evidence.** Run against a clean `git archive` of `ccd446b`:

```
SIM102 Use a single `if` statement instead of nested `if` statements
   --> .../at_head/tests/test_ui_prose_budget.py:328:21
328 | /                     if arg is None and (index is None or len(node.args) <= index):
329 | |                         # A call that omits an optional parameter carries no string.
330 | |                         if not _supplies(node, parameter, index):
    | |_________________________________________________________________^
331 |                               continue
Found 1 error.
```

`make gates > log 2>&1; echo $?` gives **2**, with
`== gates: FAILED at check (exit 2); test and smoke not run ==`. So `test` and `smoke`
never ran in the gate at all. (The failure is in the commit, not in a sibling agent's
working-tree edit: it reproduces from the archive of `ccd446b` itself.)

**Fix.** Collapse the two `if`s at `_collect`'s keyword branch into one condition —
`if arg is None and (index is None or len(node.args) <= index) and not _supplies(node, parameter, index): continue`
— or lift the whole test into a small helper, then re-run `make gates` and read `$?`
before anything else touches the log.

### 2. Three `_SCORED` rows are positionally reachable but read by keyword only, so a positional caller produces no row at all

**Claim.** The gate's § 2 rule is "positionally at the call's index for that parameter;
when the call has fewer positional arguments than that index, from the keyword". For five
rows the shipped `_SCORED` sets `index=None`, meaning keyword-only. Three of those five
parameters are **not** keyword-only in the real signature, so a caller passing them
positionally is silently skipped: no score, no exemption, no parametrized id. That is
exactly the "checker that quietly narrows its own domain" family the module's § 1 claims
four self-checks against, and the same shape round 1 found and this wave was written to
close.

**Evidence.** Enumerating `_SCORED`'s `index=None` rows against `inspect.signature`:

```
play_stop_toggle.tooltip   index=None  positionally-reachable=False  (keyword-only, safe)
clickable_label.tooltip    index=None  positionally-reachable=False  (keyword-only, safe)
draw_copyable_text.tooltip index=None  positionally-reachable=True   at index 3
preview_cell.footer        index=None  positionally-reachable=True   at index 8
preview_cell.sublines      index=None  positionally-reachable=True   at index 9
```

Demonstrated through the real module:

```
draw_copyable_text('p','v',None, tooltip='a b c d e f g')  -> [('label',1), ('tooltip',7)]
draw_copyable_text('p','v',None,        'a b c d e f g')  -> [('label',1)]
```

The second call carries a seven-word tooltip against a five-word budget and the gate
returns **no tooltip row**. Note the contrast with `gauge_bar`, whose `tooltip` is
correctly given `index=2` and is caught positionally (`gauge_bar('i',0.5,'a b c d e f g',10.0)`
-> `[('tooltip',7)]`).

This is **latent, not live**: no call site in `shaderbox/` today passes any of the three
positionally (`draw_copyable_text` calls pass 1 positional; every `preview_cell` call
passes 0). So the census figures are unaffected and no shipped string escapes. But the
hole is the kind the wave exists to close, and unlike an aliased import it is not "outside
the walk by construction" — the floor test cannot catch it, because the walk still finds
plenty of sites.

**Fix.** Give the three rows their real positional index —
`("draw_copyable_text", "tooltip", 3, 5)`, `("preview_cell", "footer", 8, 2)`,
`("preview_cell", "sublines", 9, 4)` — and pin the whole class with a test that reads each
`_SCORED` row's index back off `inspect.signature`, asserting that a row with `index=None`
names a genuinely keyword-only parameter and that a row with an index names the parameter
at that index. That generalizes `test_the_label_helpers_are_read_at_the_right_argument`
from three helpers to all sixteen rows, which is where it should have been.

### 3. D1 is enforced by a repo test but stated only in the skill, so the rule is undiscoverable from inside the repo

**Claim.** The task asks where D1 lives in this repo's docs and whether it points at
`tests/test_ui_prose_budget.py`. It lives **nowhere** in this repo's docs. Grepping
`ai_docs/conventions.md`, `ai_docs/dev_flow.md` and `CLAUDE.md` for `word budget`,
`ui_prose`, `help_marker` returns zero matches on the rule. The only statement of D1 is
`.claude/skills/imgui-ui/SKILL.md § 2`, and the only in-repo trace is the test's own
docstring, which cites the skill.

**Evidence.**

```
$ grep -ni 'word budget|prose|help_marker|ui_prose' ai_docs/conventions.md   # no rule hit
$ grep -ni 'word budget|ui_prose|help_marker' ai_docs/dev_flow.md            # (nothing)
$ grep -ni 'word budget|ui_prose' CLAUDE.md                                  # (nothing)
```

`CLAUDE.md` does route all UI work through `/imgui-ui`, so a session following the cold
start chain reaches § 2. But a reader whose only artifact is the repo — a fresh clone, CI,
anyone without the skill — hits `tests/test_ui_prose_budget.py` failing and has no in-repo
statement of the rule it enforces beyond the test's own docstring. This is the mirror of
the global rule the task cites: a cleanup ships its check, and the check should be
discoverable from the rule it enforces. The skill->test pointer landed (finding: correct
and well phrased). The test->rule direction is covered by the docstring. What is missing
is any repo-side statement of the rule at all.

**Fix.** Add one bullet to `ai_docs/conventions.md ## Code rules` stating the UI-string
budget in a sentence and naming `tests/test_ui_prose_budget.py` as its gate, so the rule
and its check are both reachable from the repo alone.

### 4. The roadmap banner still says W-B is unlanded

**Claim.** `ai_docs/roadmap.md:29` reads "069 in progress: W-C and W-A landed, W-B spec in
review", and `:38` reads "**W-B next** (`30_wave_b_prose_diet.md`, pre-review converging)".
Both are stale as of `ccd446b`.

**Evidence.** The banner's own comment is dated 2026-09-02, the same day as the commit;
W-C and W-A each got a "**W-C is DONE**" / "**W-A is DONE**" sentence with their commit
hashes, so the convention for a landed wave is established and W-B has not received it.

**Severity: low, and arguably not yet due.** `dev_flow.md` step 9 ("Done") puts the
roadmap flip after the review pass, which is this document. Recording it so the step is
not skipped.

**Fix.** After the reviews close, replace the "W-B next" clause with a
"**W-B is DONE** (`ccd446b`: prose cut to budget, the AST gate, gear auto-resize, the
engine-uniform block)" sentence and update the banner date comment.

---

## Architecture

**The gate belongs in `tests/`, and W-B follows the repo's convention.** `make check` is
`uv run pre-commit run --all-files` and nothing else (`Makefile:24`) — ruff, ruff format,
pyright. It hosts no repo-invariant checks of any kind. Every pure-Python invariant test
in this repo lives in `tests/` under `make test`: `test_document_dir_layout.py` (the bare
`"passes"` literal ban), `test_worker_daemon_contract.py` (the AST idiom this gate copies),
`test_help_content.py`. The AST-walk-plus-parametrize shape, with a per-site test id and a
failure naming file and line, is what pytest gives for free and what a `scripts/` checker
would have to rebuild. Moving it would be inventing a second convention for a repo that
already has one. **Correct as landed.**

**`_draw_auto_block` is right in `tabs/document.py`.** It is a layout function for one
region of one tab: it reads `app.panel_pass(app.current_document_id)`, pushes the tab's
font, and lays out against `SIZE.AUTO_NAME_W`. `widgets/uniform.py` holds the reusable row
primitives (`uniform_name_label`, `draw_ui_uniform`) that many callers share; the block is
a caller of one of them, not a sibling. Moving it would put a `tabs/`-specific composition
into the shared widget module. The conventions rule it must satisfy — "One shared row
primitive per row-KIND, not per-kind special-case rows"
(`conventions.md:390`) — is satisfied by the block routing through `uniform_name_label`
rather than a bare `text_colored`, which the spec's § 10 argues at length and the code
does. **Correct as landed.**

**The `theme.py` token changes follow the file's pattern.** `AUTO_NAME_W: int = 128` sits
in the `SIZE` class beside `CANVAS_FIELD_W` / `CANVAS_PRESETS_W`, with a two-line comment
saying what it is for and naming the test that bounds it — the same shape those two
neighbours carry. `PASS_SETTINGS_H` is deleted rather than kept at an unused value, and
its sole reader went with it, so pyright would have caught a miss. The spec's § 7 argument
against a module-level `_AUTO_NAME_W` constant (W-A's fix-up promoted its two local
constants to `SIZE` and deleted the comment stating the opposite rule) is the right read
of the precedent. **Correct as landed.**

**The new `HelpSection` matches the panel's voice and sits sensibly.** Reading
`shader_skeleton` and `your_uniforms`: both address the reader in second person, use
`` `code` `` for identifiers and `**bold**` for UI surfaces, run two to four short
paragraphs separated by `\n\n`, and explain a mechanism rather than narrating a decision.
`pass_settings` does all four — `` `smooth` ``, `` `repeat` ``, `**Runs**`, `**size**`,
four paragraphs, mechanism-first. Order: it lands fifth, after `shader_library` and before
`_shortcuts_section()`, which is right — the four content sections run
shader -> engine uniforms -> your uniforms -> library -> passes, roughly increasing scope,
with the keyboard reference last as a reference rather than a concept. `test_help_content.py`
needs no change exactly as § 9a predicted (`test_sections_are_well_formed` asserts
`len >= 5`, unique keys, and non-empty `key`/`title`/`body`; a sixth section satisfies all
three). The section omits `snippet`, so the default `insertable=True` is inert — the panel
gates the Insert button on `section.snippet and section.insertable` (`popups/help.py:75`).

**The skill § 2 pointer states behaviour, not the wave's story.** The five added lines say
what the test does ("walks the package's AST, scores every call carrying authored copy,
and fails an over-budget string or one joining a second clause") and what a writer must do
when their string does not fit ("goes into one of its two allowlists with a written
reason"). No wave number, no finding number, no before/after counts, nothing that dates.
It reads as a permanent property of the rule, which is the right altitude for a skill.
**Correct as landed.**

**Docs.** The two `dev_flow.md` module-map entries survive the layout changes.
`popups/pass_settings.py` (`:209`) describes the modal by what it contains — "one pass's
input wiring …, its target controls, and the rename row" — all still true; the entry says
nothing about size, scrolling or prose. `tabs/document.py` has no module-map entry of its
own (only `document.py`, the model, at `:197`, and `tabs/` collectively at `:325`), so
there is nothing to go stale. Neither needs an edit.

---

## False trails

- **The `_UNMEASURABLE` "invented entry" deviation.** The implementer reported one was removed by the anti-rot test; parsing every verdict cell out of the spec's census tables yields exactly the 29 keys the shipped dict carries, so no invented entry is traceable to the spec as it now stands — the spec's own § 5 example dict is elided with `...`. Either the spec was corrected in the same wave, or the entry came from a draft. Not a fidelity gap either way.
- **`_settings_overlay` missing from `_UNMEASURABLE`.** The census names the gear tooltip's site as `_settings_overlay:98`, a nested function, and no such key exists in the dict. Correct: the site is measurable at 2 words after the cut, so it needs no exemption.
- **`badge_chip` as a 30th unmeasurable key.** My first census run picked it up; it is a probe another agent had transiently in the working tree, absent from `ccd446b`.
- **`SPACE.LG` left unused after the block reshape.** Still used elsewhere in the repo; the import is not dead.
- **The `_FORMATS` test asserting more than the spec asked.** The spec asked for a word count on the tooltip and the label; the test also runs the clause check. A strengthening, not a drift.
- **`test_every_allowlist_entry_still_names_a_real_site` split into two tests.** Same contract, parametrized per dict, both falsifiers preserved.
- **`insertable=True` on a snippet-less section.** Inert: the Insert button is gated on `section.snippet` being non-empty.

---

## Coverage statement

I read `ccd446b` in full — all ten changed files end to end, including the 567-line gate
and the 1375-line wave spec — plus `01_spec.md § W-B` and D1, the four findings verbatim
in `00_findings.md`, `conventions.md`, `dev_flow.md`'s module map and `make gates` rules,
`.claude/skills/imgui-ui/SKILL.md` §§ 2 and 7, `help_content.py` and `popups/help.py` in
full, `tests/test_help_content.py`, and the `Makefile`.

I re-derived the census independently rather than trusting the commit message: a
throwaway collector written from the spec's §§ 2-5 prose, run against clean `git archive`
checkouts of `ccd446b` and `ccd446b^`, reconciled in both directions against both
allowlists. I exercised the shipped collector directly on six argument shapes, read the
five keyword-taking helpers' real signatures with `inspect.signature`, and confirmed the
`always_auto_resize` semantics against the installed stub rather than from memory. I ran
`pytest tests/test_ui_prose_budget.py` (222 passed) and `make gates` unpiped, reading `$?`
(exit 2), and reproduced the lint failure from the commit's own archive to rule out a
sibling agent's working-tree edit.

**Not verified:** all twelve manual steps, which need a live window this box does not
have. I checked each step's code precondition from the diff instead and say so above. In
particular the step-2 precedence question — whether `always_auto_resize` overrides an
ini-restored rect — remains open, as the spec itself flags.
