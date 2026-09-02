# W-B pre-implementation review — prose diet, gear layout, engine-uniform block

Reviewer role per `dev_flow.md` step 4: correctness & design AND verification & blast-radius.
Artifact: `30_wave_b_prose_diet.md`. Code read at `78bd1bf` (a W-A fix-up is uncommitted in the
working tree; every code claim below was re-run against `git archive 78bd1bf`).

Census re-run independently with a throwaway `ast` collector implementing §§ 2-4 of the spec
verbatim, over `shaderbox/**/*.py` at `78bd1bf`.

## Verdict

| Dimension | Verdict |
|---|---|
| Parent coverage (#5 #7 #10 #32, D1, § W-B bullets) | **PASS** |
| D1 fidelity | **PARTIAL** — three replacements are semicolon-joined two-clause strings |
| Gate domain completeness | **FAIL** — keyword-argument call sites are outside the walk entirely, and that is where the codebase's remaining over-budget authored tooltips live |
| Census accuracy | **PARTIAL** — every aggregate matches exactly; eight of thirteen named allowlist keys name functions that do not exist |
| Test falsifiability | **PARTIAL** — nine of eleven falsifiers hold; the auto-name-column bound is off by one and its stated failure mode is wrong |

The aggregates reproduce exactly. My collector returns **89 sites** (`text_colored`+FG_DIM 32,
`set_tooltip` 19, `label_row` 17, `separator_text` 10, `help_marker` 7, `row_label` 4), **63
measurable / 26 unmeasurable**, and **12 over budget** — every number in the spec's totals line, and
the same twelve sites. The spec's three design-changing refutations all hold (evidence in False
trails). What fails is not the arithmetic; it is the domain.

---

## Findings

### 1. The gate never reads a keyword argument, and that is where the surviving over-budget tooltips are (FAIL — gate domain)

**Claim.** The gate as specified scores `node.args[idx]`. Every one of the four `ui_primitives`
forwarders it allowlists is called by its callers **with a keyword**, so the "the CALLERS are the
measured sites" reason written beside each `_UNMEASURABLE` entry is false: the callers are not
measured, they are invisible. Three authored tooltips over the 5-word budget survive the gate green.

**Evidence.** `widgets/uniform.py:161-165` binds and forwards:

```
    tooltip = (
        "Can't play a single uniform while the whole script is stopped"   # 11 words
        if document_stopped
        else "Stop the script driving this uniform (edit it by hand)"     # 10 words
        if playing
        else "Resume the script driving this uniform"                     #  6 words
    )
    if play_stop_toggle(f"u_{name}", playing, tooltip=tooltip):
```

`widgets/uniform.py` contributes **zero rows** to my 89-site census — I checked by filtering the
collector output for that path. `tabs/document.py:318-325` does the same with
`tooltip="Stop the whole script (freeze all uniforms)"` (7 words). `widgets/uniform.py:52-58` calls
`clickable_label(..., tooltip="Jump to declaration")`. `popups/lib_picker/preview.py:30` calls
`tooltip="Click to copy file path"` (5). These are authored UI copy of exactly the kind #10 is
about, and the gate's parametrization never generates an id for any of them.

I ran the mutation directly: a probe file containing `help_marker(text="…14 words…")` produced a
row with no argument node at all (`len(node.args) <= idx`), so the site scores neither a count nor
UNMEASURABLE. Under the spec as written it falls out of both dicts' assertions silently — the
"checker that quietly narrows its own domain" family, and the one the spec's own § 1 says it
carries four self-checks against.

**Fix (paste).** § 2 gains: "The scored argument is read positionally at the call's index, and when
the call has fewer positional arguments, from the keyword whose name is the parameter at that index
(`text` for `help_marker`, `label` for `label_row` / `row_label`). A call that supplies the scored
argument by neither is UNMEASURABLE, never skipped." § 2's call set gains `play_stop_toggle`,
`clickable_label`, `clipped_caption`, `gauge_bar`, `draw_copyable_text` and `preview_cell`'s
`footer` / `sublines` at their `tooltip` / `label` / `text` keyword, each at the 5-word tooltip
budget, and the four `_UNMEASURABLE` forwarder entries are then deleted rather than kept — a
forwarder measured at its callers needs no exemption. The census gains the three
`widgets/uniform.py` strings as cuts or `_OVER_BUDGET` entries with a reason.

### 2. Eight of the thirteen named allowlist keys name functions that do not exist (FAIL — the allowlist is dead on arrival)

**Claim.** The allowlist is keyed `(module path, enclosing function name)` and
`test_every_allowlist_entry_still_names_a_real_site` asserts every key names a real site. Eight of
the keys the census tables spell out do not. The gate goes red on its first run for reasons that
have nothing to do with any UI string.

**Evidence.** `grep -c "^def <name>(" shaderbox/ui_primitives.py` at `78bd1bf`:

| Spec's key | Reality |
|---|---|
| `ui_primitives.py::usage_bar` | no such function; the real one is `gauge_bar` |
| `ui_primitives.py::copyable_label` | no such function; the real one is `draw_copyable_text` |
| `ui_primitives.py::copy_path_label` (`"Click to open + copy"`) | no such function; the real one is `draw_link` |
| `ui_primitives.py::clickable_label` `text_colored` FG_DIM | `clickable_label` contains **no** `text_colored` call (read `ui_primitives.py:1222-1252`) |
| `ui_primitives.py::row_label` as a gate site | its body calls `small_caption`, which is **not** in the gate's six; `row_label` is a *called* site, never a *containing* one |
| `popups/settings.py::_draw_copilot` | the real one is `_draw_copilot_config` (`settings.py:279`) |
| `popups/emoji_picker.py::draw` and `::_draw_grid` | the file defines only `draw_emoji_picker`, `_draw_body`, `_pick`; both sites are in `_draw_body` |
| `popups/lib_picker/tree.py::_draw_inline_input` | the real one is `_draw_inline_new_input` |
| `tabs/code.py::draw` (the vim message) | the real one is `draw_chrome` |

Two real unmeasurable sites are missing from the spec entirely: `ui_primitives.py::preview_cell`
has **two** `text_colored(COLOR.FG_DIM, …)` sites (`:1045` `label`, `:1053` `text`), neither listed.
The totals still land on 26 because the invented rows and the missed rows happen to cancel; that is
the strongest argument for re-deriving the tables from a run rather than transcribing them.

**Fix (paste).** Regenerate every census table's Site column mechanically from the collector output
(`(path, enclosing function name)` printed by the collector), not by hand, and state in § 5 that
the allowlist's initial contents are the collector's own output pasted in.

### 3. `test_the_auto_name_column_fits_every_engine_uniform` passes on a name that does not fit (PARTIAL — a checker that admits its own defect)

**Claim.** The test asserts every engine uniform name is at most **20 characters**, "the width
`_AUTO_NAME_W` was chosen against". At 12px in the app's font, 20 characters is 131.0px against a
128.0px column. The test's own bound admits an overflowing name.

**Evidence.** `app.get_font` loads `AnonymousPro-Regular.ttf` with no glyph-range argument; it is
monospace, advance 1118/2048 em = 0.5459, so one character at 12px is 6.5508px. 19 chars = 124.5px
(fits); 20 chars = 131.0px (does not). `128 / 6.5508 = 19.54`.

The stated failure mode is also wrong. The block draws names through `uniform_name_label` →
`clickable_label`, whose body calls `_ellipsize(label, width)` (`ui_primitives.py:1240`,
`:46-59`). An over-wide name is therefore **truncated to `u_pass_iteratio...`**, not overlapped —
silent data loss, which is worse than the overlap the test claims to prevent and is invisible to
every manual step in the spec.

**Fix (paste).** Change the bound to 19 characters and restate the falsifier: "a 20-character
engine uniform name renders ellipsized inside `clickable_label`, silently losing its tail; the
character bound is `_AUTO_NAME_W / 6.5508` for the 12px monospace face, floored."

### 4. Renaming `separator_text("Runs per frame")` to `Runs` breaks a live cross-reference in `help_content.py` (PARTIAL — blast radius)

**Claim.** The cut is described as touching only `pass_settings.py`. It also invalidates a string in
a file the Files table does not list.

**Evidence.** `help_content.py:42`:
`"u_pass_iteration": ("float", "which run this is, 0-based (see Runs per frame)")`. That parenthetical
names the gear heading the reader is being sent to. After the cut the heading reads `Runs` and the
pointer names a section that no longer exists. `tests/test_help_content.py` asserts coverage of the
uniform set, not the wording, so nothing catches it.

**Fix (paste).** Add `shaderbox/help_content.py` to the Files table: `ENGINE_UNIFORM_DOCS["u_pass_iteration"]`'s
text becomes `"which run this is, 0-based (see Runs)"`, so the Help panel's pointer keeps naming a
heading that exists.

### 5. Three of the six surviving replacements are two clauses joined by a semicolon (PARTIAL — D1 fidelity)

**Claim.** D1 and § 2 both say `help_marker` is **one clause**, and § 2 states the mechanical test:
"A string with an em-dash-joined second clause is over budget by construction." A semicolon-joined
second clause is the same shape with different punctuation, and the word count does not see it.

**Evidence.** `share of the canvas; output is always full` (8 words, two independent clauses),
`clamps to 0-1; smallest` (4), `holds values above 1; the default` (6),
`full precision; twice the memory` (5). Four of the seven strings the wave ships carry the pattern.
The gate cannot refuse them because § 3 scores words only, so D1's "one clause" half is unenforced
by the very test whose purpose is enforcing D1.

**Fix (paste).** Either add to § 3 "a scored `Constant` or `JoinedStr` containing `;` or ` — ` fails
the clause half of D1, asserted alongside the count", and rewrite the four strings to one clause; or
state in § Design decisions 11 that D1's clause half is deliberately left to review-by-eye and say
why the count is the only assertable half.

### 6. The gate matches call names without saying how, so an alias and a method call disagree (PARTIAL — gate domain)

**Claim.** § 2 says the gate "reads by name" and the falsifier for
`test_the_walk_finds_the_known_call_sites` mentions "the `Attribute` branch that matches
`imgui.set_tooltip`", but nothing states the matching rule. Two mutations then have undefined
answers.

**Evidence.** I ran both against my collector (`ast.Name → id`, `ast.Attribute → attr`):
`from ui_primitives import help_marker as hm` then `hm("…10 words…")` produced **no row** — the
alias is invisible under any name-based rule, and the spec does not say so. `imgui.help_marker("…14
words…")` produced a row scoring 14 under an `attr`-based rule and no row under a strict
`imgui.`-prefix rule. Today the codebase has no alias (`grep` for
`import .* as (hm|help_marker)` returns nothing) and the six calls appear only as bare
`help_marker(` / `label_row(` / `row_label(` and `imgui.`-prefixed `set_tooltip` /
`separator_text` / `text_colored` — so both mutations are theoretical today, but the gate's answer
should be written down rather than left to the implementer.

The same silence covers a same-named local helper: `tabs/document.py:257` defines
`_entry_row_label`, which an `endswith`-style rule would sweep in and an exact-name rule would not.

**Fix (paste).** § 2 gains one sentence: "A call matches when `func` is an `ast.Name` whose `id` is
one of the six, or an `ast.Attribute` whose `attr` is one of the six — the module qualifier is not
checked, so an aliased import is out of the walk by construction and `test_the_walk_finds_the_known_call_sites`'s
floor is what catches an import style the walk cannot see."

### 7. `_entry_row_label` has one caller, not two, and its allowlist reason therefore does not hold (PARTIAL — census)

**Claim.** The `tabs/document.py` census row reads "allowlisted because both callers pass a literal;
the callers are measured."

**Evidence.** `grep -n "_entry_row_label" shaderbox/tabs/document.py` at `78bd1bf` returns exactly
two lines: the definition at `:257` and one call at `:302` (`_entry_row_label(script_active, "Script")`).
The Shader entry-point row does not use the helper. One caller, and it passes its literal
positionally to a *non-gate* function, so nothing measures `"Script"` either — the argument reaches
`text_colored` only inside `_entry_row_label`, where it is the unmeasurable parameter.

**Fix (paste).** Rewrite the reason as: "allowlisted because its one caller (`_draw_entry_points`)
passes the literal `"Script"` to a helper outside the gate's call set; the string is 1 word and is
verified by eye here."

### 8. § 2's heading promises a keyword surface the section never defines, and § 12 does not exist (PARTIAL — spec integrity)

**Claim.** Two dangling references, both pointing at the hole finding 1 describes.

**Evidence.** The heading of § Design decisions 2 reads "six call names, **one keyword-argument
surface**, and the budget each carries"; the section body has no occurrence of the word "keyword"
(`grep -n keyword` over the file returns that heading and nothing else). Line 506 reads "the gate
does not see them through the call; **§ 12** below says how it does" — the numbered sections stop at
11 (`grep -n "^### [0-9]"`). The `_FORMATS` mechanism it means is
`test_the_format_tooltips_are_within_the_help_budget` in § Tests.

**Fix (paste).** Repoint the `_FORMATS` sentence at `§ Tests, test_the_format_tooltips_are_within_the_help_budget`,
and make the § 2 heading's keyword clause real by folding in finding 1's rule.

### 9. `_AUTO_NAME_W`'s "W-A's precedent" justification is invalidated by the W-A fix-up now in the working tree (PARTIAL — a census row moving under the spec)

**Claim.** § Design decisions 10 argues `_AUTO_NAME_W: float = 128.0` stays a module constant
"beside `_CANVAS_FIELD_W` and `_CANVAS_PRESETS_W` (W-A's precedent, and W-A's open question 3 states
the rule: arithmetic and layout-local numbers stay module constants)". The uncommitted W-A fix-up
deletes both constants and the comment stating that rule, promoting them to `SIZE.CANVAS_FIELD_W`
and `SIZE.CANVAS_PRESETS_W` in `theme.py`.

**Evidence.** `git diff shaderbox/tabs/document.py` removes the block
`_CANVAS_FIELD_W: float = 56.0` / `_CANVAS_PRESETS_W: float = 64.0` together with its "so they stay
module constants" comment; `git diff shaderbox/theme.py` adds `CANVAS_FIELD_W: int = 56` and
`CANVAS_PRESETS_W: int = 64` to `SIZE` (and a `COLOR.VIEWER_BORDER` token). The named precedent will
not exist when W-B is written.

The string census itself does **not** move: I ran the collector over the working tree and got the
same 89 rows with an identical `(path, function, call, argument)` key set, so no census row changes.

**Fix (paste).** Replace the precedent sentence with the choice on its own terms, or follow the
fix-up and make it `SIZE.AUTO_NAME_W: int = 128` in `theme.py` beside `CANVAS_FIELD_W`; either way
drop the citation of a comment that no longer exists.

### 10. Question 4's premise is false: `help_content.py` has no pass section, and neither concept is covered (PARTIAL — a fact going nowhere)

**Claim.** Open question 4 takes the default "nowhere in the app; **the Help panel and the tutorial
already own them**". The Help panel does not.

**Evidence.** `grep -i "smooth|filter|repeat|wrap|clamp|tiling|edges|sampling" shaderbox/help_content.py`
returns **zero** matches. `help_sections()` returns five sections — `shader_skeleton`,
`engine_uniforms`, `your_uniforms`, `shader_library`, `shortcuts` — and **none is about passes at
all**. So deleting the `sampling` and `edges` markers removes the only place in the app where
texture filtering and wrap mode are explained, and the spec's justification for the deletion asserts
a coverage that does not exist.

Per the reviewer's ruling: W-B adds the two facts to `help_content.py`'s data in this wave rather
than deferring to W-H. Because there is no pass section to add to, the smallest honest shape is a
new `HelpSection` rather than an edit.

**Fix (paste).** § Files gains `shaderbox/help_content.py`, and `help_sections()` gains a
`key="passes"` section whose body carries one sentence each: "`smooth` blends between pixels when
another pass reads this one — the right choice for a blur or an upscale; off gives hard pixel
edges." and "`repeat` wraps a read past the edge to the far side for tiling; off clamps to the edge
pixel, which is what a feedback trail wants." `tests/test_help_content.py` covers the uniform set
only, so no test changes.

### 11. `always_auto_resize` versus `modal_window`'s `Cond_.first_use_ever` and a persisted `imgui.ini` (PARTIAL — verification)

**Claim.** § Design decisions 7 says the height "is ignored" with auto-resize on, so `0.0` is
honest. The size is seeded through `set_next_window_size(..., Cond_.first_use_ever)`
(`ui_primitives.py:328-329`), and `app.py:174-177` points imgui's ini at
`app_data_dir()/"imgui.ini"` with `save_imgui_ini()` called from `ui.py:149`. A developer box
already carries a saved 440x400 rect for the `Pass settings` popup.

**Evidence.** The claim that auto-resize wins over a restored ini rect is the one assertion in § 7
not traced to the installed stub — the stub text quoted ("Resize every window to its content every
frame") is about the resize, not about precedence over `.ini`-restored size. Manual step 2
("open the gear on `cascade`, then on `seed`; the popup is visibly shorter") is the right falsifier,
but it will be run on a box whose ini already has the old rect, which is exactly the case that
could pass or fail for the wrong reason.

**Fix (paste).** Manual step 2 gains a precondition: "delete `imgui.ini` from the app-data dir
first, then repeat the step with it in place — both must show the height following the content, or
the flag is not winning over the restored rect."

### 12. The block's placement contradicts the file's current order without saying so (minor)

**Claim.** § 10 says the block "sits after the sort row's `dummy`". Today the call is **before** it.

**Evidence.** `tabs/document.py:226-231` at `78bd1bf`: `if auto_hashes:` / `same_line(spacing=XL)` /
`_draw_auto_row(...)`, then `imgui.dummy((0, SPACE.MD))`. The spec's `_draw_auto_block` snippet also
drops the `if auto_hashes:` guard, which today prevents a bare `same_line` with nothing after it.

**Fix (paste).** State the move explicitly: "the `if auto_hashes:` guard stays, the
`same_line(spacing=XL)` inside it goes, and the whole guarded call moves below the existing
`imgui.dummy((0, SPACE.MD))`."

---

## Mutation table (the coverage the review owes)

| # | Mutation | Gate as specified | Named by the spec? |
|---|---|---|---|
| a | over-budget `Constant` at a fresh `help_marker` site | **RED** | yes, `test_every_measured_site_is_within_budget` |
| b | over-budget `JoinedStr`, each `FormattedValue` = 1 | **RED** (verified: the Reads tooltip scores 31 this way) | yes, § 3 |
| c | `label_row` label containing a `FormattedValue` | **RED**, twice (count 6 > 2, and the interpolation assertion) | yes, § 6 + `test_no_label_carries_an_interpolation` |
| d | a site removed from `_UNMEASURABLE` without becoming measurable | **RED** | yes, `test_every_unmeasurable_site_is_listed` |
| e | allowlist entry whose site no longer exists | **RED**, both dicts (`_OVER_BUDGET` sharper: the key carries the count) | yes, `test_every_allowlist_entry_still_names_a_real_site` |
| f | new site reaching `set_tooltip` through a `Subscript` | **RED** (unmeasurable and unlisted) | yes |
| f' | ... through a `Name` bound in an **enclosing** function | **RED** (the resolver walks only the nearest `FunctionDef`; an outer binding does not resolve, so the site is unmeasurable and unlisted) | § 4 says "does not resolve" for module scope; the nested-function case is not named but lands the same way |
| g1 | `help_marker(text=...)` keyword call | **GREEN — MISS.** No positional arg at the index; the site produces neither a score nor UNMEASURABLE | **no** → finding 1 |
| g2 | alias `from ui_primitives import help_marker as hm` | **GREEN — MISS** | **no** → finding 6 |
| g3 | `imgui.set_tooltip` vs bare `set_tooltip` | matching rule undefined; both plausible | **no** → finding 6 |
| g4 | call inside a lambda or a comprehension | **RED** — `ast.walk` reaches it and the nearest `FunctionDef` ancestor is found (verified on a probe: both scored) | not named, but correct |
| g5 | string built with `+`, `%` or `.format` | **RED** as unmeasurable-and-unlisted (verified on a probe: `BinOp` and `Call` args both returned UNMEASURABLE) | not named, but correct |
| h | a site in a file the walk does not visit | walk is `shaderbox/**/*.py`, so `popups/lib_picker/` and `exporters/` **are** covered — my run returns rows from both (`lib_picker/search.py`, `lib_picker/tree.py`, `lib_picker/preview.py`, `lib_picker/__init__.py`, `exporters/telegram.py`) | yes, § 1 |

Three misses, all on the same axis: the gate matches a *shape of call*, and every call shape that
does not put the string in a positional slot under a bare or `imgui.`-qualified name is outside it.
`g1` is the one with live instances (finding 1); `g2` and `g3` are latent.

## The budgets

The five-word tooltip bound is consistent with D1 and § 2 and the spec says where it came from: the
census's own gap between the longest passing tooltip and the shortest failing one. I re-measured
that gap and it is real but **narrower** than the spec states. The spec says "the longest passing
tooltip is 5 words, the shortest failing one is 10". My run: longest passing is 5
(`"Click to open + copy"`, in `draw_link`), shortest failing is **10** among the *cut* sites but
**7** once finding 1's keyword sites are in scope (`"Stop the whole script (freeze all uniforms)"`).
The threshold is still in empty space at 5; the sentence "the exact threshold is not load-bearing"
survives, but the numbers behind it change once the domain is fixed.

`separator_text` at 2 is consistent with § 2's table ("section title 1-2 words") and its own
examples, and open question 6 states the trade honestly including the argument against. I agree with
2: "per frame" restates the `runs` label directly below it. The one consequence the spec misses is
finding 4.

The `help_marker` <= 8, label <= 2, empty state <= 4 bounds are D1's own numbers, unchanged.

## The cuts

Every replacement is in budget (I counted all ten). Each keeps the fact the control needs:

- `name` → `names its shader file and its wires` (7): keeps both facts #5's original carried that
  are not recoverable from the label. The dropped half ("Enter applies") is D11/W-C's commit rule,
  which W-C already made an app-wide behaviour rather than a per-control one.
- `size` → `share of the canvas; output is always full` (8): keeps the output-pass exception, which
  is the fact finding #4 is about and the reason the row is disabled at all.
- `runs` → `redraws per frame, each reading the last` (7): keeps the chaining fact. Drops the
  `u_pass_iteration` / `u_pass_iterations` names, which `help_content.py:42-43` documents.
- `sampling` / `edges` markers deleted: rule 1 (label unambiguous) applies, **but** see finding 10 —
  the facts land nowhere.
- `_FORMATS` f1/f2/f4: each keeps the choosing criterion (clamping, above-1, precision-vs-memory).

The three `_OVER_BUDGET` entries are each justified by something other than difficulty. Two
(`popups/help.py::_draw_body` 15, `popups/lib_picker/__init__.py::_draw_body` 11 — both confirmed at
those counts by my run) explain a **disabled state**, which a control's name structurally cannot
carry; cutting them leaves a greyed button with no reason. The third
(`exporters/telegram.py::_draw_status_slot` 9) is a derived stat line whose 9 is five
`FormattedValue`s plus four authored words; the entry says the scoring rule is wrong for stat lines
rather than that the string is fine. All three name a revisit condition. I agree with all three.

## Gear layout

- **Jitter (§ 3).** Holds. The width stays fixed at `SIZE.PASS_SETTINGS_W = 440` and only the height
  follows content, so nothing changes width across passes — open question 3 reaches this conclusion
  for the right reason (with `always_auto_resize` an unfixed width would follow the widest Reads
  combo and jitter as the user picks).
- **Flags named exactly.** Yes: `imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_scrollbar`,
  at the `modal_window` call site, height `0.0`. `modal_window`'s signature is
  `(label, size, flags=0, fixed_size=False)` and its body passes `flags` straight to
  `begin_popup_modal` (`ui_primitives.py:336`) — the spec's "no `ui_primitives.py` edit needed"
  correction to the parent's Files line is right. Both flags are already used in this codebase
  (`ui_primitives.py:352`, `ui.py:58`, `popups/examples.py`).
- **`SIZE.PASS_SETTINGS_H` goes.** Confirmed safe: `grep -rn PASS_SETTINGS_H shaderbox/ tests/`
  returns `theme.py:276` and `pass_settings.py:62`, nothing else.
- **Open issue:** the `Cond_.first_use_ever` + persisted `imgui.ini` interaction — finding 11.

**The `·` glyph is safe.** U+00B7 is present in `AnonymousPro-Regular.ttf` (checked with fontTools
against `getBestCmap()`) and sits inside imgui's default Basic-Latin + Latin-1 range, which is what
`app.get_font` gets since it passes no `glyph_ranges` (`app.py:1174-1178`). It already renders in the
app at `tabs/code.py:212` (`f"Line {n}  ·  {msg}"`). The spec's citation is off by a file — it says
`exporters/telegram.py`'s artifact line, which does carry one, but `tabs/code.py` is the older and
plainer proof.

## The engine-uniform block

The name column's width **is** derived and pinned: `_AUTO_NAME_W: float = 128.0` with
`test_the_auto_name_column_fits_every_engine_uniform` asserting the character bound. It is stable
across documents because it is a constant and because `ENGINE_DRIVEN_UNIFORMS` is a frozenset in
`core.py:36-39`, not per-document data — the five names that reach the block are the same in every
document. That is the right design; the bound is off by one (finding 3) and the constant's home
argument is stale (finding 9).

`uniform_name_label` is kept, which preserves the code↔panel bridge
(`widgets/uniform.py:62-72`, the `is_item_hovered` branch calling `_locate_uniform_declaration`) and
avoids re-committing feature 008's special-case-row mistake that `conventions.md` names. Manual step
10 is the correct falsifier for it. The `label_row` refusal is right for the stated reason: it calls
`set_next_item_width` for a widget that never comes.

## Tests

Nine of eleven falsifiers hold as described. Two do not:

- `test_the_auto_name_column_fits_every_engine_uniform` — finding 3.
- `test_every_allowlist_entry_still_names_a_real_site` — it holds *as a mechanism*, and finding 2 is
  the demonstration: eight of the keys the spec hands it are stale, so it goes red on first run.
  That is the test working, and the spec's tables failing it.

`test_the_walk_finds_the_known_call_sites`'s floor of 60 against today's 89 is the right shape (a
count that tracks the census trains the reader to bump it). Note it does **not** catch finding 1:
the keyword sites were never in the 89, so the floor was always measured against the narrowed
domain.

`test_a_name_rebound_by_augassign_is_not_resolved` is the right call as a unit test — I confirmed
the codebase's only `Name`-to-string binding is `widgets/details.py:47-49`
(`text = "No output file selected"` then `text += f" ({', '.join(extensions)})"`), outside the
gate's call set, so a tree-level assertion would be green with or without the branch.

## False trails

- The three design-changing refutations all hold. `row_label(font, label, label_w=...)` at
  `ui_primitives.py:1078` and `label_row(font, label, item_width, label_w=...)` at `:1088` take
  `font` first (`small_caption(font, text)` at `:693` likewise) — argument index 1 is correct, and a
  collector reading index 0 does report 21 sites as `app.font_12`.
- `_FORMATS` really is reached through an `ast.Subscript`
  (`help_marker(_FORMATS[_FORMAT_CODES.index(target.dtype)][2])`, `pass_settings.py:186`), invisible
  to any call-site walk; the direct-import test is the only way to reach it.
- The `_FORMATS` menu-label bound of <= 2 holds today: `8-bit` (1), `16-bit float` (2),
  `32-bit float` (2).
- The walk **does** cover `popups/lib_picker/` and `exporters/` recursively — I got rows from five
  files across those two trees.
- `help_content.py` really does contain zero calls to any of the six gate call names, so its
  exemption is by construction, as claimed. (Its prose problem is finding 10, a different thing.)
- The 26-unmeasurable and 12-over-budget aggregates both reproduce exactly; the reconciliation
  table's account of the `text_colored` 31→32 delta as W-A's `"x"` separator is correct
  (`tabs/document.py:155`).
- A disabled slider rendering its format string is stated as inspection-plus-manual-check with a
  named fallback, which is the honest handling; I did not verify it and neither did the spec, and
  manual step 4 is the right falsifier.
- Open questions 1, 2, 3, 5 and 6: I agree with all five defaults. Question 4 is finding 10.

## Coverage statement

Read: `30_wave_b_prose_diet.md` in full; `01_spec.md` § Locked decisions D1, § W-B, § Manual
verification, § Review history round 3; `00_findings.md` #5 #7 #10 #32 verbatim;
`.claude/skills/imgui-ui/SKILL.md` §§ 2, 3, 5; project `CLAUDE.md`. Opened at `78bd1bf`:
`ui_primitives.py` (`modal_window`, `row_label`, `label_row`, `small_caption`, `clickable_label`,
`_ellipsize`, `preview_cell`, `help_marker`), `popups/pass_settings.py`, `widgets/pass_list.py`,
`tabs/document.py`, `widgets/uniform.py`, `widgets/details.py`, `help_content.py`, `theme.py`,
`core.py`, `app.py` (font loading, ini path), `popups/settings.py`, `popups/emoji_picker.py`,
`popups/lib_picker/tree.py`.

Ran: an independent AST collector implementing §§ 2-4 over `shaderbox/` at `78bd1bf` (89 rows,
diffed row by row against every census table); the same collector over the working tree (identical
row set, so the W-A fix-up moves no census row); a mutation probe file exercising keyword calls,
aliases, `imgui.`-qualified calls, lambdas, comprehensions and `+` / `%` / `.format` strings; a
fontTools cmap check on `AnonymousPro-Regular.ttf` for U+00B7 and the monospace advance width.

Not verified: whether `always_auto_resize` overrides an `.ini`-restored rect (finding 11 turns it
into a manual precondition); whether `begin_disabled` suppresses a slider's format string (manual
step 4, with a named fallback).

---

# Round 2 (closure)

Narrow closure round on the folded spec at `3910900`. Method for F1 and F2: I rewrote my collector
from the folded rules (16-row call table read positionally or by keyword, `IfExp` worst branch
recursive, `List`/`Tuple` worst element, the clause check over `;` / ` — ` / ` -- `, the `AugAssign`
guard) and diffed my row set against the spec's regenerated tables **mechanically** — parsing the
106 markdown rows out of § The census and comparing them as multisets against my collector's output.

**Aggregates reproduce exactly:** 106 sites, 71 measurable, 35 unmeasurable, 18 distinct violating
sites (17 over the word budget, 9 failing the clause check).

**Row-set diff: identical.** 106 spec rows, 106 collector rows, zero mismatches on
`(file, function, call, parameter)`. **Per-row value diff: zero mismatches** on word count, clause
verdict and the `+interp` flag — every `--` in the spec's W column is a row my scorer also returned
UNMEASURABLE, and every number matches.

## Per-finding verdicts

| # | Verdict | Closing text, and what I checked |
|---|---|---|
| F1 | **CLOSED** | § 2 now reads the argument "positionally at the call's index ... when the call has fewer positional arguments, from the keyword whose name is the parameter at that index" and states "A call that supplies the scored argument by neither is UNMEASURABLE, never skipped." The 16-row table adds the seven helpers. My collector, built from that table, returns the 17 rows the reconciliation predicts, including all five keyword sites (`play_stop_toggle` x2, `clickable_label`, `preview_cell` x2). `widgets/uniform.py` now contributes 5 rows where it contributed 0. The three tooltips I demonstrated — 11 words at `_draw_play_stop`, 7 at `_draw_entry_points`, and the `clickable_label` 3 — all appear, and the two over-budget ones are cut. The four forwarder exemptions are deleted rather than kept, which is the right resolution: their stated reason is now true. |
| F2 | **CLOSED** | § 5 opens "Both dicts' initial contents are the collector's own output pasted in, never a hand transcription, and the same is true of every census table below", and § The census repeats "Generated, not transcribed." Every one of the eight wrong keys is now right in the tables: `gauge_bar` :842, `draw_copyable_text` :1161, `draw_link` :1189, `_draw_copilot_config` :298/:307, `emoji_picker::_draw_body` :54/:65, `lib_picker/tree::_draw_inline_new_input` :211, `code.py::draw_chrome` :405. Both previously-missing `preview_cell` `text_colored` rows (:1045, :1053) are present. The invented `clickable_label` `text_colored` row and the invented `row_label`-as-container row are both gone. |
| F3 | **CLOSED** | `test_the_auto_name_column_fits_every_engine_uniform` now asserts **19**, carries `floor(SIZE.AUTO_NAME_W / 6.5508)` at the assertion so the arithmetic is visible, and restates the falsifier as silent ellipsis truncation rather than overlap. The spec correctly flags the font metric as a relayed number it did not re-derive. |
| F4 | **CLOSED** | § 9b added, and `shaderbox/help_content.py` is in the Files table: `ENGINE_UNIFORM_DOCS["u_pass_iteration"]` becomes `"which run this is, 0-based (see Runs)"`. |
| F5 | **CLOSED** | § 3 gains `clause_ok(text) -> ";" not in text and " — " not in text and " -- " not in text`, asserted beside the count, plus `test_no_scored_string_joins_a_second_clause`. I re-ran count and clause over all 18 replacement strings (the 17 plus the extra `play_stop_toggle` branch): **zero failures**. The four semicolons are gone — `share of the canvas, output always full` (7), `clamps to 0-1, the smallest` (5), `holds values above 1, the default` (6), `full precision, twice the memory` (5). |
| F6 | **CLOSED** | § 2 states the matcher: `ast.Name.id` or `ast.Attribute.attr` in the set, module qualifier unchecked, an aliased import out of the walk by construction with the floor test as the backstop, and exact-name matching keeping `_entry_row_label` out. That is a decidable rule, and my collector implements it and lands on the same 106 rows. |
| F7 | **CLOSED** | The premises table records "**Refuted, one.**" and the regenerated row carries the real shape at `_entry_row_label` :287 as `_UNMEASURABLE`. |
| F8 | **CLOSED** | `grep` for "§ 12" over the file returns only the round-1 fold table's description of the old defect; the `_FORMATS` sentence now points at `test_the_format_tooltips_are_within_the_help_budget`. § 2's heading promise is real. |
| F9 | **CLOSED** | `SIZE.AUTO_NAME_W: int = 128` in `theme.py` beside `CANVAS_FIELD_W`, with the stale "W-A's precedent" citation replaced by the fix-up's own settlement of the question. |
| F10 | **CLOSED** | § 9a ships `HelpSection(key="passes")` in this wave. It goes past what I asked for: reading the surviving markers found **four** dropped facts, not the two I drafted. The `test_help_content.py` claim checks out — `test_sections_are_well_formed` asserts `len(sections) >= 5` (line 27), so a sixth section satisfies it, and no test asserts prose or an exact count. |
| F11 | **CLOSED** | Manual step 2 is now a two-run check: delete `imgui.ini`, run, then repeat with it in place; both must show the height following content. It also names the reason — the stub's quoted text is about the resize, not about precedence over a restored rect. |
| F12 | **CLOSED** | § 10 and the Files row both state the move exactly: the `if auto_hashes:` guard stays, the `same_line(spacing=XL)` inside it goes, and the guarded call moves below the existing `imgui.dummy((0, SPACE.MD))`. |

## The new `passes` HelpSection

Help prose, not UI copy, so the gate does not score it and D1's word budget does not apply. Against
D1's spirit — one fact, plainly stated — all four hold:

1. `smooth` (25 words): one fact (what filtering does between pixels) plus its use case. The
   semicolon joins the on and off states of one control, which is the fact, not a second one.
2. `repeat` (26 words): same shape, wrap versus clamp, with the feedback-trail case naming why the
   off state is the interesting one.
3. **Runs** (31 words): one fact, and it is the only place in the app connecting `u_pass_iteration` /
   `u_pass_iterations` to what sets their count — `ENGINE_UNIFORM_DOCS` names them but not that.
4. **size** (30 words): one fact with its consequence ("a quarter of the pixels") and the output-pass
   exception, which is finding #4's subject and which the 7-word marker cannot hold.

Each states one fact plainly. Sentence 3 is the longest and earns it. The markdown-lite vocabulary
(`**bold**`, `` `code` ``) matches what `ui_primitives.markdown_text` parses.

## Two stale leftovers from the fold

Neither is a finding against the design; both are text the fold did not sweep, and both would
mislead an implementer reading the Files section alone.

1. **The Files section contradicts itself on `help_content.py`.** The table's row adds the file
   (cross-reference plus the `passes` section), and the paragraph below the table still reads
   "`help_content.py` is not touched: verified to contain zero calls to any of the six gate call
   names." Fix: delete that sentence, or restate it as "`help_content.py` contains zero calls to any
   gate call name, so its prose is outside the gate; this wave edits it for the reasons in §§ 9a-9b."
2. **The `theme.py` Files row does not mention the token it adds.** It reads
   "`SIZE.PASS_SETTINGS_H` deleted. `PASS_SETTINGS_W` stays" while § 10 introduces
   `SIZE.AUTO_NAME_W: int = 128` in that file. Fix: append "`SIZE.AUTO_NAME_W: int = 128` added
   beside `CANVAS_FIELD_W`."

## False trails

None raised. Everything I checked this round either closed or is one of the two text leftovers
above; I record no preferences.

## Overall

**PASS.** All twelve findings closed. The census is now generated rather than transcribed, and an
independent re-derivation of it from the folded rules agrees with the spec on every one of the 106
rows and every per-row value. The two leftovers are one-sentence text fixes in the Files section,
not design defects, and nothing in the wave's implementation depends on either.
