# 069 W-B — Prose diet, gear layout, engine-uniform block

The third wave of feature 069 (`01_spec.md § W-B`), folding findings #5 #7 #10 #32 and enforcing
locked decision D1. It lands as one commit: the cut and the gate that pins it ship together, because
a word budget with no check is a wish, and this repo has the receipts (`conventions.md`, the
`make gates` bullet in `dev_flow.md`).

Written against `3910900` (W-C, W-A and W-A's fix-up all landed and committed; W-A left the gear's
size help text untouched by design, for this wave). Round 1 of pre-implementation review returned
gate-domain FAIL and census PARTIAL; all twelve findings are folded, and every census table below is
now the collector's own output rather than a hand transcription. § Review history records the round.

---

## Goal

After this wave, every fixed UI string in `shaderbox/` fits the D1 budget, and a new one that does
not cannot reach a commit without either being cut or being written into an allowlist with a reason
beside it. The pass-settings gear sizes to its own content and never scrolls; its size row carries
the derived resolution inside the slider rather than in the fixed label column; the engine-driven
uniforms stack vertically under the sort row instead of overflowing the panel on one line.

Concretely, the three surfaces the maintainer named:

- The gear (`popups/pass_settings.py`) loses **219 words net** — 196 across its call sites (six help
  markers, one section tooltip, one empty state, two headings, one label) and 23 more in the
  `_FORMATS` table — and gains a height that follows its content.
- The strip's gear icon (`widgets/pass_list.py`) hovers "Pass settings", nothing else.
- The Document tab's engine uniforms (`tabs/document.py`) become a block, one per line.
- Four authored tooltips that reach their helper by KEYWORD (`widgets/uniform.py` x3,
  `tabs/document.py` x1) are cut. They were invisible to the gate as first specified, which is the
  hole round 1 found: the gate read positional arguments only, so the helper was allowlisted as a
  forwarder and its callers were never measured.

## Findings folded (verbatim)

**#5 — UX, Pass settings gear, Reads section.** The maintainer's words:

> "'nothing — declare a sampler2D uniform to read another …' doesn't fit the default popup size.
> Strip this text 2x. Pure textual noise, stop writing long passages everywhere. Also the text under
> (?) is absurd. Compact this crap."

The ledger's own verification adds the rule this wave enforces: "The maintainer's rule for the spec:
**every in-UI string is one short clause**; a help marker explains the control in one sentence at
most; the empty-state line must fit the popup width (target: half the current length, e.g. 'no
sampler2D uniforms'). This is a class, not one string."

**#7 — UX, Pass settings gear, Draws into / size row.** The maintainer's words:

> "text 'size (1080, 1080)' doesn't fit (in the settings). it overlaps with the scroll bar"

The ledger's fix shape: "The popup is a fixed 440×400 … and its content is taller than that, so it
grows a vertical scrollbar at all — a settings popup that needs scrolling is itself the defect; it
should size to its content. … the derived resolution does not belong in the label column — show it
as the slider's own format string (e.g. '100% · 1080×1080') … label stays the one word 'size'."

**#10 — UX, Passes strip, gear icon.** The maintainer's words:

> "When hovering the pass settings icon I don't want to see this text passage. 'Pass settings' is
> enough."

The ledger: "Same bar as #5 — an icon tooltip is the control's NAME, nothing more. The spec's prose
sweep should grep every `set_tooltip(` / `help_marker(` and cut each to its name or one clause."

**#32 — UX, Document tab, engine uniforms row.** The maintainer's words:

> "`u_pass_iteration: …  u_pass_iterations: …  u_resolution: […]` don't fit the panel. Align them
> vertically, under the sort picker (outside the sorting)."

The ledger's fix: "drop the `same_line` before `_draw_auto_row`; draw the engine uniforms as a
vertical block under the sort row, one uniform per line (`label_row`-style fixed name column, dim
value), still outside the sorted list and still read-only; keep the code↔panel hover bridge on the
names."

**D1, quoted in full from `01_spec.md § Locked decisions`:**

> **D1. UI strings have a word budget** (`.claude/skills/imgui-ui/SKILL.md § 2`): label 1-2 words;
> icon tooltip = the control's name; `help_marker` one clause, <= 8 words, only where the label is
> ambiguous; empty state <= 4 words; derived values in the control, never the label. Enforced by a
> test (W-B).

## Out of scope (each naming its owning wave)

- **Runtime-built text** — the vim status line (W-F), notification bodies, error strings from the
  compiler and the script engine, the copilot's chat rendering. The gate measures fixed authored
  text only; a string assembled per frame from live values has no stable word count to assert. The
  parent says so: "Runtime-built text (the vim status line, notifications) stays outside it, and the
  UI waves' reviews cover those by eye." The one notification this wave touches at all is W-A's
  `Canvas: {w}x{h}`, which is already inside the skill's one-clause budget (it is § 2's own example)
  and is measured here only as a premise check, not as a cut.
- **The tutorial's prose** (`tutorial_body.html`) — **W-H**. Finding #5 says the same
  over-explaining "exists in the tutorial … and likely in Help panel content".
- **`help_content.py`'s prose is outside the GATE but not outside this WAVE.** It contains zero
  calls to any gate call name, so it is outside the walk by construction rather than by an exemption
  anyone must remember, and § 2 exempts it as documentation a reader chose to open. But round 1
  showed the Help panel does not yet carry the facts this wave's cuts delete, so W-B *writes* two
  entries there (§ Design decisions 9a, 9b) while measuring none of it.
- **The gear's rows themselves** — which controls exist, what they wire, the three-state input combo.
  Wiring semantics are **W-D**; the rename crash and the commit rule are **W-C**, landed.
- **`SIZE.PASS_SETTINGS_W`.** The width stays 440. Only the height token goes; a modal that
  auto-resizes vertically still needs a width, or every row's control column would follow its
  longest label.
- **The copilot Settings hints** (`popups/settings.py::_draw_copilot_config`'s `_COPILOT_LIMITS` hints). Two-to-three-sentence
  `help_marker` hints, well over budget, and they are the one place in the app where the reader is
  spending real money on the answer. They go in the allowlist with that reason; cutting them is a
  copilot-UX question, not a prose-diet one, and `copilot-llm-agent-design` owns it.

---

## Design decisions

Numbered, code-level. Every one is a constraint from the parent spec or D1, resolved to an exact
shape here.

### 1. The gate is one test module, `tests/test_ui_prose_budget.py`, built on the AST idiom already in the repo

`tests/test_worker_daemon_contract.py` is the shape to copy, and this wave copies it rather than
inventing a second one. Its three moving parts are exactly what this gate needs:

- a module-level collector that walks `shaderbox/**/*.py` with `ast.parse` and returns a list of
  tuples;
- a `@pytest.mark.parametrize` over that collector, so each site is its own test id and a failure
  names the file and line;
- a first test asserting the collector found the known sites, so an enumeration that silently
  matched nothing cannot pass every assertion vacuously. Its comment names the family: "the
  'checker that narrows its own domain' family". That failure mode is the one `~/.claude`'s
  debugging discipline calls the most expensive bug family, and it is why this gate carries four
  self-checks rather than one.

The module is new. It does not extend `test_worker_daemon_contract.py`, which owns a different
invariant.

### 2. What the gate scores: a call set, an argument read positionally OR by keyword, and a budget each

The parent fixes six call names: `help_marker`, `set_tooltip`, `separator_text`, `label_row`,
`row_label`, and `text_colored` with a `COLOR.FG_DIM` first argument. **Round 1 showed that set is
not the gate's domain, it is a subset of it.** Every `ui_primitives` helper that takes authored copy
takes it by KEYWORD from its callers, and a gate reading positional arguments only sees neither the
helper (whose argument is a parameter) nor the caller (whose argument is in a keyword slot). Four
over-budget authored tooltips sat in that hole, including three in `widgets/uniform.py`, a file that
contributed **zero rows** to the first census. So the call set is the six plus every
`ui_primitives` function whose signature takes a `tooltip` / `label` / `text` / `footer` /
`sublines` parameter carrying authored copy, each read at that parameter.

**How a call matches.** A call matches when `node.func` is an `ast.Name` whose `id` is in the set,
or an `ast.Attribute` whose `attr` is in the set. The module qualifier is **not** checked, so
`imgui.set_tooltip` and a bare `set_tooltip` both match, and an aliased import
(`from ui_primitives import help_marker as hm`) is out of the walk **by construction** —
`test_the_walk_finds_the_known_call_sites`'s floor is what catches an import style the walk cannot
see, not the matcher. Matching is by exact name, so `tabs/document.py`'s local `_entry_row_label`
is not swept in by resemblance to `row_label`.

**How the scored argument is read.** Positionally at the call's index for that parameter; when the
call has fewer positional arguments than that index, from the keyword whose name is the parameter at
that index. **A call that supplies the scored argument by neither is UNMEASURABLE, never skipped.**
The first specification skipped it — a probe file containing `help_marker(text="...14 words...")`
produced no row at all, scoring neither a count nor an exemption, which is the
"checker that quietly narrows its own domain" family this gate's § 1 claims four self-checks
against. A genuinely optional parameter that the call omits contributes no row, which is different
and correct: there is no string to measure.

The table, one row per (call, parameter) the gate scores:

| Call | Parameter | Index | Budget | D1 clause |
|---|---|---|---|---|
| `help_marker` | `text` | 0 | <= 8 words | "`help_marker` one clause, <= 8 words" |
| `imgui.set_tooltip` | `text` | 0 | <= 5 words | "icon tooltip = the control's name" |
| `imgui.separator_text` | `label` | 0 | <= 2 words | "section title 1-2 words" (§ 2's table) |
| `label_row` | `label` | **1** | <= 2 words, no `FormattedValue` | "label 1-2 words" + "derived values in the control" |
| `row_label` | `label` | **1** | <= 2 words, no `FormattedValue` | same |
| `imgui.text_colored` (arg 0 `COLOR.FG_DIM`) | `text` | 1 | <= 4 words | "empty state <= 4 words" |
| `play_stop_toggle` | `tooltip` | keyword-only | <= 5 words | tooltip |
| `clickable_label` | `label` | 0 | <= 2 words | label |
| `clickable_label` | `tooltip` | keyword | <= 5 words | tooltip |
| `clipped_caption` | `text` | 0 | <= 4 words | a dim readout, the empty-state budget |
| `gauge_bar` | `tooltip` | 2 | <= 5 words | tooltip |
| `draw_copyable_text` | `label` | 0 | <= 2 words | label |
| `draw_copyable_text` | `tooltip` | keyword | <= 5 words | tooltip |
| `draw_link` | `label` | 0 | <= 2 words | label |
| `preview_cell` | `footer` | keyword | <= 2 words | a tile's one-line label |
| `preview_cell` | `sublines` | keyword | <= 4 words each | a tile's dim second line |

`preview_cell`'s `sublines` is a sequence; the gate scores each element and reports the worst, so
one long subline among short ones cannot hide. W-D removes the sublines the strip passes today, and
the parameter stays scored so a later caller cannot reintroduce a passage there.

**The four forwarder exemptions are deleted, not kept.** The first draft allowlisted
`play_stop_toggle`, `clipped_caption`, `gauge_bar` and the rest with the reason "their callers are
the measured sites", which was false while the callers were invisible. With the callers in scope the
reason becomes true and the exemption becomes unnecessary: a helper whose own argument is a
parameter is still UNMEASURABLE at its definition site (`ui_primitives.py::play_stop_toggle`'s
`set_tooltip(tooltip)`), and that entry stays, but it now means what it says.

**The argument index for the three label helpers is 1, not 0**, and getting this wrong is the single
most likely way to ship a gate that measures nothing. Their signatures are
`row_label(font: imgui.ImFont, label: str, label_w: float = ...)` and
`label_row(font, label, item_width, label_w=...)` — the font comes first. A first draft of this
wave's census scored argument 0 and reported 21 sites all carrying `app.font_12`, which is
unmeasurable by construction rather than a finding. `test_the_label_helpers_are_read_at_the_right_argument`
(§ Tests) pins it against the real signatures via `inspect.signature`, so a future reordering turns
the gate red rather than blind.

The tooltip budget of 5 needs its own justification, since D1 states it as "the control's name"
rather than a number. A name is not always one word (`Revert this turn's changes` is four,
`Click to open + copy` is four), and a number is what a test can assert. Five is the smallest bound
admitting every tooltip that reads as a name and excluding every one that reads as a passage. With
the keyword sites in scope the gap narrows from the first draft's claim: the longest passing tooltip
is 5 words (`Click to open + copy`, in `draw_link`) and the shortest failing one is **7**
(`Stop the whole script (freeze all uniforms)`, a keyword site), not 10. The threshold still sits in
empty space, but the numbers behind "not load-bearing" are the corrected ones.

`separator_text` gets 2 rather than § 2's loosely-stated "1-2 words": `Runs per frame` is three and
is cut to `Runs` (§ The census, and open question 6 states the trade).

### 3. Scoring: a count AND a clause check, over `Constant`, `JoinedStr`, `IfExp` and sequences

```
score(node) ->
    Constant with a str value  -> len(value.split())
    JoinedStr                  -> sum: Constant str -> its words; FormattedValue -> 1
    IfExp                      -> WORST(score(body), score(orelse)), recursively
    List | Tuple               -> WORST over the elements
    anything else              -> UNMEASURABLE

clause_ok(text) -> ";" not in text and " — " not in text and " -- " not in text
```

A measurable site passes only when **both** hold. D1 and § 2 say `help_marker` is "one clause", and
§ 2 states the mechanical test itself: "A string with an em-dash-joined second clause is over budget
by construction." A semicolon-joined second clause is the same shape with different punctuation, and
a word count cannot see it. Round 1 caught the first draft shipping four semicolon-joined
replacements under a gate that scored words only — D1's clause half unenforced by the very test
whose purpose is enforcing D1. All four are rewritten to one clause (§ The census), and the check is
now asserted beside the count.

The three separators are the ones this codebase actually uses to join a second clause; the ASCII
`--` is included because `pass_settings.py`'s longest marker uses it. A comma-joined continuation is
not caught and is not meant to be: `redraws per frame, each reading the last` is one clause with a
trailing participial phrase, which is what the budget is for.

Four properties of the scoring rule matter.

**A `FormattedValue` counts one word.** The parent fixes this. It is the right count because the
interpolated value occupies roughly one token's worth of reading, and because it makes
`f"size ({w}, {h})"` score 6 against a 2-word label budget — finding #7 exactly.

**A `JoinedStr` whose parts are adjacent string literals is invisible as a split, and that is
correct.** Python concatenates adjacent literals at parse time, so the six-line `help_marker` in
`_draw_repeat` is a single `ast.Constant` of 53 words, not six of nine. The reverse case is the one
the parent names: the Reads tooltip is three adjacent literals of which the middle is an f-string,
so the whole expression is one `JoinedStr` scoring 31.

**An `IfExp` scores its worst branch, recursively.** The first draft treated a conditional as
UNMEASURABLE, which put three real authored tooltips outside the gate — the gear's size marker (41
words on its output branch), `tabs/document.py`'s three-branch `open_tooltip`, and
`widgets/uniform.py`'s three-branch play/stop tooltip. All three are authored copy chosen at
runtime, and the branch a user sees is not knowable statically, so the honest bound is the longest
one. Recursion reaches every leaf of a chained `a if p else b if q else c`.

**A `List` or `Tuple` scores its worst element**, which is what makes `preview_cell(sublines=[...])`
measurable rather than a hole one long entry can hide in.

### 4. A `Name` argument is resolved to a string assignment in the enclosing function; otherwise the site is allowlisted

The parent: "A `Name` argument is resolved to its assignment in the enclosing function when that is
a string; otherwise the site is listed in a pinned allowlist in the test."

The resolution algorithm:

```
resolve(arg, enclosing_fn) ->
    if arg is not ast.Name: return arg
    # A name the function ever appends to is longer than any one binding shows.
    if any(isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name)
           and n.target.id == arg.id for n in ast.walk(enclosing_fn)):
        return UNMEASURABLE
    candidates = [ n.value for n in ast.walk(enclosing_fn)
                   if isinstance(n, ast.Assign)
                   and len(n.targets) == 1
                   and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id == arg.id
                   and isinstance(n.value, (ast.Constant, ast.JoinedStr)) ]
    if len(candidates) == 1: return candidates[0]
    if len(candidates) > 1:  return WORST(candidates)   # the highest word count
    return UNMEASURABLE
```

**The `AugAssign` branch is not hypothetical.** `widgets/details.py` binds
`text = "No output file selected"` and then, one line later, `text += f" ({', '.join(extensions)})"`.
A resolver that stopped at the bare `Assign` would score that site 4 words for a string that renders
7. The site is outside the gate's call set (`caption_text`), so it does not fail today — but it is
the only `Name`-to-string binding anywhere in the census, which means the naive rule's single
opportunity to be exercised is also its counterexample. Treating any augmented name as unmeasurable
is the conservative direction: it sends the site to the allowlist, where a human writes down why,
rather than reporting a number that is wrong in the permissive direction.

The enclosing function is found by walking a parent map built once per module (`ast.iter_child_nodes`
over the tree), taking the nearest `FunctionDef` / `AsyncFunctionDef` ancestor. A `Name` bound at
module scope, by a `for` target, by a tuple unpack, or by an `ast.IfExp` / `ast.BinOp` value does not
resolve.

**Multiple assignments score the worst, not the first.** A site like `tabs/document.py`'s
`open_tooltip` is bound once, by a three-branch `IfExp`, so it does not resolve at all — but a
future site bound by two straight-line assignments on different branches would resolve
ambiguously, and taking the longest is the only choice that cannot let an over-budget branch through.
This is a rule about the gate's soundness, not about any string in the tree today.

**The `IfExp` case moved to § 3.** The first draft listed the codebase's three conditional-bound
strings here as unresolvable and allowlisted them; round 1 showed all three are authored copy, so
§ 3 now scores an `IfExp`'s worst branch and all three become cuts. The resolver still stops at
straight-line assignments — a `Name` bound BY an `IfExp` resolves to that `IfExp`, which § 3 then
scores; a `Name` bound by a `Call`, a `BinOp` or a comprehension does not resolve.

### 5. The allowlists are data in the test module, keyed by symbol not line, and neither can rot

**Both dicts' initial contents are the collector's own output pasted in, never a hand transcription,
and the same is true of every census table below.** Round 1 found eight of the thirteen allowlist
keys the first draft spelled out named functions that do not exist (`usage_bar` for the real
`gauge_bar`, `copyable_label` and `copy_path_label` for `draw_copyable_text` and `draw_link`,
`_draw_copilot` for `_draw_copilot_config`, `emoji_picker.draw` and `_draw_grid` for `_draw_body`,
`_draw_inline_input` for `_draw_inline_new_input`, `code.py::draw` for `draw_chrome`), while two
real sites (`preview_cell`'s two `text_colored` calls) were missing entirely. The aggregates still
matched because the invented rows and the missed rows cancelled — which is the strongest possible
argument for generating the tables from a run. `test_every_allowlist_entry_still_names_a_real_site`
would have gone red on its first run for reasons having nothing to do with any UI string.

The parent's round-3 review left the placement open ("whether the allowlist lives in the test or a
data file"). It lives **in the test module**, as a module-level dict. A separate JSON file would put
the reason one file away from the assertion that reads it, and this repo already pays for a rotting
sibling elsewhere (`shader_lib_api_lock.json` is a generated snapshot, which is a different thing —
nobody writes a reason into it by hand).

**There are two kinds of exemption and they are two dicts, not one.** Conflating them would let a
single entry mean either "this string cannot be read" or "this string was read and we accept it",
which are different claims with different anti-rot conditions.

```python
# Sites the gate CANNOT measure, each with why. A `ui_primitives` entry is a shared helper
# forwarding a caller's text -- the CALLERS are the measured sites.
_UNMEASURABLE: dict[tuple[str, str], str] = {
    ("shaderbox/ui_primitives.py", "play_stop_toggle"): "forwards the caller's tooltip",
    ...
}

# Sites the gate CAN measure, that are over budget, and that stay. Each entry carries the
# measured word count, so a rewrite that changes the string changes this line too.
_OVER_BUDGET: dict[tuple[str, str, int], str] = {
    ("shaderbox/popups/help.py", "_draw_body", 15): "disabled-state reason; a name cannot carry it",
    ...
}
```

`_OVER_BUDGET`'s key carries the **count**, which is what makes it self-invalidating: edit the
string and the measured count no longer matches the key, so the site falls back into the budget
assertion and goes red until someone updates the entry deliberately. An exemption that survives its
own subject changing is not an exemption, it is a hole.

**The key is `(module path, enclosing function name)`, never a line number.** `conventions.md`'s
code rule is explicit: "No raw line numbers OR file-length counts in docs … Cite the **symbol**
instead of a line". A line-keyed allowlist would need re-syncing on every edit above it, which is
how an allowlist stops being read. A function-keyed one moves with its code.

Two functions carrying two unmeasurable sites collapse to one entry, which is right: the reason is
the same ("this helper forwards its caller's text"), and the caller is where the measurement
happens.

**The anti-rot half is a test, not a convention** (§ Tests,
`test_every_allowlist_entry_still_names_a_real_site`): every key in either dict must correspond to a
site the walk actually found — an unmeasurable one for `_UNMEASURABLE`, a measurable one at exactly
the recorded word count for `_OVER_BUDGET`. Delete the code, and the stale entry turns the suite red
on the next run rather than sitting there forever describing a function that no longer exists. This
is the mutation the parent asks for by name: "an allowlist entry that no longer exists in the code →
red, so the allowlist cannot rot".

### 6. A `label_row` / `row_label` label carries no `FormattedValue` at all

Separate from the word count, and stricter. The parent: "asserts a `label_row` / `row_label` label
carries no `FormattedValue` at all (a label is fixed text)". So `f"size ({w}, {h})"` fails on two
counts — 6 words against a 2-word budget, and an interpolation in a fixed-width column — and a
hypothetical `f"{n}"` in a label would fail on the second alone despite scoring 1 word.

The reason it is a separate assertion rather than a consequence of the count: a label column is
fixed-width by design (§ 2, "A label column is fixed-width by design … so nothing variable-length
ever goes in a label"), and "variable-length" is a property of the expression, not of any one
rendering of it. The count is about reading; this is about layout.

### 7. The gear sizes to its content: `always_auto_resize`, and `SIZE.PASS_SETTINGS_H` goes

`modal_window` already forwards window flags ("`flags` passes window flags through (e.g.
`no_scrollbar` for a modal that sizes its own content)"), so the change is at the call site:

```python
    with modal_window(
        _LABEL,
        (float(SIZE.PASS_SETTINGS_W), 0.0),
        flags=imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_scrollbar,
    ) as visible:
```

`always_auto_resize` is "Resize every window to its content every frame" (verified in the installed
stub, `imgui_bundle/imgui/__init__.pyi::WindowFlags_`). With it set, imgui recomputes the window
rect from the content each frame and the height passed to `set_next_window_size` is ignored, so
`0.0` is the honest value to pass rather than a number that no longer means anything. The width is
still honoured, which is what keeps the control column stable.

`no_scrollbar` rides along as belt-and-braces, per § 3's own advice ("Add `WindowFlags_.no_scrollbar`
to a pure display box as belt-and-suspenders"). With auto-resize on, a scrollbar cannot appear; with
the flag, a future content change that somehow could produce one clips instead of silently
reintroducing #7's overlap.

`SIZE.PASS_SETTINGS_H: int = 400` is **deleted** from `theme.py`. It has exactly one reader, the
call site above, and a token nothing reads is the kind of leftover `/sanitize` deletes. `theme.py`
keeps `PASS_SETTINGS_W: int = 440`.

**This is a chosen alternative, and the rejected one is worth recording.** The parent allows either
("`modal_window` with auto-resize or a computed height"). `popups/examples.py` computes its height
by hand (`body_h = grid_h + _DESC_SLOT_H + frame_h + 2.0 * style.item_spacing.y`, then adds title
bar and padding) and passes `fixed_size=True`. That is correct there, because its content is a grid
whose size the caller already knows exactly. The gear's content is a variable number of input rows
(one per sampler uniform the pass declares), so a computed height would have to model row heights,
`same_line` help markers and separators — arithmetic that is wrong the first time anyone adds a row.
Auto-resize is the shape whose correctness does not depend on the content staying the shape it is
today.

One consequence to accept knowingly: `always_auto_resize` also removes the user's ability to resize
the modal by dragging, and `modal_window`'s `Cond_.first_use_ever` persistence of a user drag stops
applying to this popup. That is the intent — a settings popup with a right size has nothing to
resize, and finding #7 is a report that the wrong size was reachable at all.

### 8. The size row: label `size`, slider format `%.0f%% · %dx%d` resolved per frame

Today:

```python
    label_row(app.font_12, f"size ({w}, {h})", _CTRL_W, _ROW_LABEL_W)
```

with a comment above it reading "The derived resolution lives in the LABEL — inside the slider only
the percent fits." That comment is the defect written down, and it goes with the line.

After:

```python
    label_row(app.font_12, "size", _CTRL_W, _ROW_LABEL_W)
    imgui.begin_disabled(is_output)
    scale_changed, percent = imgui.slider_float(
        f"##scale_{name}", target.scale * 100.0, 5.0, 100.0, f"%.0f%% · {w}x{h}"
    )
    imgui.end_disabled()
```

The parent fixes the format as `%.0f%% · WxH`. `w` and `h` are already computed two lines above from
`document.canvas_size` and `target.scale`, so the f-string interpolates them into the format string
that imgui then applies `%.0f%%` against. **The `x` is an ASCII `x`, not `×`**, matching
`util.get_resolution_str` and W-A's `Canvas: {w}x{h}` notification, so the app spells a resolution
one way everywhere.

The middle dot `·` is the separator the parent and § 2 both write, and it already appears in
`exporters/telegram.py`'s artifact line, so it is the codebase's existing separator rather than a
new character.

Two things this does not break. imgui applies a `printf` format to the slider's float value, and a
format string may carry literal text around the conversion — `exporters/telegram.py` and this same
file's `"%.0f%%"` are the existing proof that `%%` reaches the widget as a literal percent. And the
row is inside `begin_disabled` for the output pass (W-A), where the slider still renders its format
string, so the output pass shows `100% · 512x512` greyed rather than showing nothing.

### 9. The Reads section: the header tooltip goes, the empty state becomes four words

The parent: "The gear's Reads header tooltip goes; the empty state is 'no sampler2D uniforms'."

`separator_text("Reads (?)")` becomes `separator_text("Reads")` and the whole `is_item_hovered` /
`set_tooltip` block below it is deleted. The `(?)` in the label was a marker saying "this heading
hosts a tooltip"; with no tooltip there is nothing to mark, and a heading reading `Reads (?)` with
no hover behind it is worse than either end state.

The comment above it goes with the code it explains. Per `conventions.md`, a comment "states what's
non-obvious about the code as it is NOW"; with the tooltip gone it narrates a mechanism that is not
there.

What the deleted tooltip taught is not lost: "declare another sampler2D in this pass's shader" is
what the four-word empty state says in place, and the per-pass wiring rule is W-D's and the Help
panel's.

### 9a. The three deleted gear facts land in `help_content.py`, in this wave

Round 1's finding 10 refuted the first draft's justification for the deletions. It said the facts
"go nowhere in the app; the Help panel and the tutorial already own them". The Help panel does not:
`help_sections()` returns five sections (`shader_skeleton`, `engine_uniforms`, `your_uniforms`,
`shader_library`, `shortcuts`) and **none is about passes at all**; grepping `help_content.py` for
`smooth|filter|repeat|wrap|clamp|tiling|edges|sampling` returns zero matches. So deleting the
`sampling` and `edges` markers would remove the only place in the app where texture filtering and
wrap mode are explained, on a justification asserting a coverage that does not exist.

The ruling: **W-B adds the facts in this wave rather than deferring to W-H.** Because there is no
pass section to add to, the smallest honest shape is a new `HelpSection`, not an edit:

```python
        HelpSection(
            key="pass_settings",
            title="Passes",
            body=(
                "`smooth` blends between pixels when another pass reads this one — the right "
                "choice for a blur or an upscale; off gives hard pixel edges.\n"
                "\n"
                "`repeat` wraps a read past the edge to the far side for tiling; off clamps to "
                "the edge pixel, which is what a feedback trail wants.\n"
                "\n"
                "**Runs** draws the pass more than once per frame, each run reading what the one "
                "before it wrote — `u_pass_iteration` says which run this is and "
                "`u_pass_iterations` how many there are.\n"
                "\n"
                "**size** is the pass's share of the canvas. Half size is a quarter of the "
                "pixels, which is the usual choice for a blur; the output pass always draws full."
            ),
        ),
```

Four facts, not two. The reviewer drafted the `smooth` and `repeat` sentences; reading the surviving
markers' current text shows two more facts that the cuts drop and nothing else carries — the
`u_pass_iteration` / `u_pass_iterations` naming from the runs marker (the surviving 7-word
replacement cannot hold it) and the "half size is a quarter of the pixels" reasoning from the size
marker. `ENGINE_UNIFORM_DOCS` documents the two uniform names but not what sets their count, so the
runs sentence is the only place the connection is made.

`body` is markdown-lite (`**bold**`, `` `code` ``), which is what `ui_primitives.markdown_text`
understands; `snippet` defaults to `""` and is omitted, since there is no GLSL to insert.

**The key is `pass_settings`, not `passes`.** `tests/test_document_dir_layout.py` forbids the bare
string `"passes"` anywhere under `shaderbox/` outside `paths.py` — it is `PASSES_DIR_NAME`, and a
second spelling makes a rename break some readers silently. A help-section key is not that
directory, so importing the constant would be a false coupling; `pass_settings` names the panel the
section documents and collides with nothing. The title the reader sees stays `Passes`.

**`tests/test_help_content.py` needs no change, and here is why.** Its five tests are: the engine
uniform set matches `ENGINE_UNIFORM_DOCS` (untouched), the `engine_uniforms` section's snippet lists
each uniform (untouched), `test_sections_are_well_formed` (`len(sections) >= 5`, unique keys,
non-empty `key`/`title`/`body` — a sixth section with all three satisfies every clause), and two
`shortcuts` assertions (untouched). Nothing asserts the section count exactly, and nothing asserts
prose. That is also why round 1's finding 4 exists: the file's cross-reference is unpinned too.

### 9b. `help_content.py`'s cross-reference follows the heading rename

`ENGINE_UNIFORM_DOCS["u_pass_iteration"]` reads
`("float", "which run this is, 0-based (see Runs per frame)")`. That parenthetical names the gear
heading this wave renames to `Runs`, so after the cut it points at a section that does not exist. It
becomes `"which run this is, 0-based (see Runs)"`. Round 1's finding 4; `tests/test_help_content.py`
asserts coverage of the uniform set, not the wording, so nothing catches it and the fix is a
same-wave edit rather than a test.

### 10. The engine-driven uniforms become a vertical block under the sort row

`_draw_auto_row` is renamed `_draw_auto_block` and loses its inter-item `same_line`; its call site
loses the `same_line(spacing=float(SPACE.XL))` that put it beside the sort combo. Finding #32 names
both halves.

The row shape, per the finding ("`label_row`-style fixed name column, dim value"):

```python
def _draw_auto_block(app: App, uniforms: list[UIUniform]) -> None:
    # Engine-driven uniforms: one row each under the sort row, outside the sorted list.
    # The names keep the code<->panel hover/jump bridge; values are read-only.
    panel_pass = app.panel_pass(app.current_document_id)
    imgui.push_font(app.font_12, app.font_12.legacy_size)
    for u in uniforms:
        uniform_name_label(
            app, u.name, float(SIZE.AUTO_NAME_W),
            text_color=COLOR.STATE_INFO, accent=COLOR.STATE_INFO,
        )
        imgui.same_line(float(SIZE.AUTO_NAME_W) + float(SPACE.MD))
        imgui.text_colored(COLOR.FG_DIM, format_auto_value(panel_pass.uniform_values.get(u.name)))
    imgui.pop_font()
```

Three details, each a constraint rather than a preference:

**The name column is fixed, not `calc_text_size(u.name).x` per row.** Today's per-name width is what
makes the row a ragged inline list; a fixed column is what makes five rows read as a block. It is
**`SIZE.AUTO_NAME_W: int = 128` in `theme.py`, beside `CANVAS_FIELD_W`** — not a module constant.
The first draft argued for a module constant citing "W-A's precedent" and W-A's open question 3;
round 1 pointed out that W-A's fix-up (`3910900`) deletes `_CANVAS_FIELD_W` / `_CANVAS_PRESETS_W`
and the comment stating that rule, promoting both to `SIZE`. The cited precedent will not exist, and
the fix-up settled the question the other way, so this follows it.

128 fits the longest engine name, `u_pass_iterations` at 17 characters, in the 12px font, and the
column is measured against `ENGINE_DRIVEN_UNIFORMS` by a test rather than by eye
(`test_the_auto_name_column_fits_every_engine_uniform`).

**`uniform_name_label` stays, with its `same_line` moved outside it.** It is the shared row primitive
the conventions require ("One shared row primitive per row-KIND, not per-kind special-case rows …
feature 008 special-cased engine uniforms into a dim caption row, which left them out of the
code↔panel hover/jump bridge until it was generalized"). Drawing these rows with a bare
`text_colored` would re-commit exactly that mistake. The `same_line(SIZE.AUTO_NAME_W + SPACE.MD)` after
it is `row_label`'s own idiom, inlined here because `row_label` takes a font and pushes no colour,
and these rows need `STATE_INFO` on the name.

**The block does not go through `label_row`.** `label_row` would work and would put these rows
inside the gate's `label_row` census, which sounds like a bonus. It is not: `label_row` calls
`set_next_item_width` for a widget that is about to be drawn, and there is no widget here, only
read-only text. Using it would leave a dangling item-width push and mean something the code does not.

**The move, stated exactly** (round 1, finding 12: the first draft said "after the sort row's
`dummy`" while the call is currently *before* it, and its snippet silently dropped the guard). Today
the order is `if auto_hashes:` / `same_line(spacing=XL)` / `_draw_auto_row(...)`, then
`imgui.dummy((0, SPACE.MD))`. After: **the `if auto_hashes:` guard stays** (it is what prevents a
bare `same_line` with nothing after it, and with the block it prevents an empty gap), **the
`same_line(spacing=XL)` inside it goes**, and **the whole guarded call moves below the existing
`imgui.dummy((0, SPACE.MD))`**, with its own `dummy` after it before the `ui_uniforms` child.

It stays "outside the sorting" as the finding requires: `auto_hashes` is already partitioned away
from `active_uniform_hashes` before `sort_uniform_hashes` runs, and this wave does not touch that
partition.

### 11. Every cut is a rewrite to budget, not a deletion, except where the label already says it

The census below gives the exact replacement for every over-budget site. Three rules produced them,
stated here so a reviewer can check each cell against a rule rather than against taste:

1. **A `help_marker` whose label is unambiguous is deleted, not shortened.** D1: "only where the
   label is ambiguous". `edges` beside a checkbox labelled `repeat` is not ambiguous; neither is
   `sampling` beside `smooth`. Both markers go entirely.
2. **A `help_marker` that survives keeps the clause that says what the control does to the
   picture**, and drops the clause that says why one would want it. § 2: "It names what the control
   does, never why one would want it, never the alternative, never the history."
3. **A tooltip becomes the control's name.** Not a shortened sentence about the control.

---

## The census

**Generated, not transcribed.** Every table below is the throwaway collector's output pasted in, one
row per (call, scored parameter), sorted by file and line. The collector implements §§ 2-4 exactly
and is the same algorithm the gate ships. Round 1 found eight invented and two missing rows in the
hand-written first draft, whose aggregates nonetheless matched because the errors cancelled; nothing
here is typed by hand.

Run at `3910900`. **Totals: 106 sites. 71 measurable, 35 unmeasurable.
17 over the word budget, 9 failing the clause check, 18 distinct sites
violating one or both.** Of those 18, **13 are cut here** and **5 get an `_OVER_BUDGET` entry** with
a reason. One further site is cut without being a violation: `separator_text("Reads (?)")` at 2
words is in budget, and its `(?)` marker goes because the tooltip it advertises does (§ Design
decisions 9).

The W column carries `+clause` when the string joins a second clause with `;`, ` — ` or ` -- `, and
`+interp` when a `label_row` / `row_label` label carries a `FormattedValue`. A `--` in W means the
scorer could not read the argument.

Line numbers are the committed tree's; a `tabs/document.py` edit was in the working tree while this
was written and shifts four of them by three to five lines. The row SET is identical either way
(verified by diffing the collector's output across both), and the implementation cites symbols, so
nothing here depends on the numbers.

### `popups/pass_settings.py` — the gear (findings #5, #7)

| Site | Call | Current text | W | Replacement / dict | Verdict |
|---|---|---|---|---|---|
| `_draw_body` :79 | `separator_text` `label` (pos) | Pass | 1 | unchanged | 1 |
| `_draw_name` :95 | `label_row` `label` (pos) | name | 1 | unchanged | 1 |
| `_draw_name` :107 | `help_marker` `text` (pos) | The pass's name: its shader file under passes/ and what other passes' Reads call it. Enter applies; a rename re-points every wire and open tab. | 25 +clause | `names its shader file and its wires` | 25 -> 7 |
| `_draw_inputs` :140 | `separator_text` `label` (pos) | Reads (?) | 2 | `Reads` | 2 -> 1 |
| `_draw_inputs` :142 | `set_tooltip` `text` (pos) | Every sampler2D uniform this pass declares gets a row here; pick which pass fills it ({} leaves it black). To read something new, declare another samp | 31 +clause | `DELETED (D9)` | 31 -> -- |
| `_draw_inputs` :149 | `text_colored` `text` (pos) | nothing — declare a sampler2D uniform to read another pass | 10 +clause | `no sampler2D uniforms` | 10 -> 3 |
| `_draw_inputs` :159 | `label_row` `label` (pos) | uniform | -- | `_UNMEASURABLE` | unmeasurable |
| `_draw_target` :177 | `separator_text` `label` (pos) | Draws into | 2 | unchanged | 2 |
| `_draw_target` :179 | `label_row` `label` (pos) | format | 1 | unchanged | 1 |
| `_draw_target` :186 | `help_marker` `text` (pos) | _FORMATS[_FORMAT_CODES.index(target.dtype)][2] | -- | `_UNMEASURABLE` | unmeasurable |
| `_draw_target` :192 | `label_row` `label` (pos) | size ({}, {}) | 6 +interp | `size` | 6 -> 1 |
| `_draw_target` :207 | `help_marker` `text` (pos) | How big this pass's own image is, relative to the canvas. Half size is a quarter of the pixels — the usual choice for a blur, which looks the same and | 41 +clause | `share of the canvas, output always full` | 41 -> 7 |
| `_draw_target` :216 | `label_row` `label` (pos) | sampling | 1 | unchanged | 1 |
| `_draw_target` :221 | `help_marker` `text` (pos) | How another pass reads this one BETWEEN pixels: smooth blends neighbours (right for a blur or an upscale), off gives hard pixel edges. | 23 | `DELETED (label reads 'smooth')` | 23 -> -- |
| `_draw_target` :226 | `label_row` `label` (pos) | edges | 1 | unchanged | 1 |
| `_draw_target` :231 | `help_marker` `text` (pos) | What a read PAST this pass's edge returns: repeat wraps to the far side (tiling), off clamps to the edge pixel — which is what a feedback trail wants. | 29 +clause | `DELETED (label reads 'repeat')` | 29 -> -- |
| `_draw_repeat` :253 | `separator_text` `label` (pos) | Runs per frame | 3 | `Runs` | 3 -> 1 |
| `_draw_repeat` :255 | `label_row` `label` (pos) | runs | 1 | unchanged | 1 |
| `_draw_repeat` :260 | `help_marker` `text` (pos) | How many times this pass draws each frame, each run reading what the one before it wrote. One is an ordinary pass. More builds a chain inside a single | 53 +clause | `redraws per frame, each reading the last` | 53 -> 7 |

Gear total at its call sites: **223 words in, 27 out, 196 net cut.** Three markers are deleted
outright and one section tooltip with them; the arithmetic is over the rows above, summing the W
column for every row whose Verdict shows a change and subtracting the replacement's count.

The gear's `_FORMATS` tooltips are data reached through the `Subscript` at `:186`, so no call-site
walk can read them. `test_the_format_tooltips_are_within_the_help_budget` asserts against the
imported table directly:

| Format | Current tooltip | W | Replacement | W |
|---|---|---|---|---|
| `f1` | `Smallest. Values clamp to 0-1 — the right choice for a final image.` | 13 | `clamps to 0-1, the smallest` | 5 |
| `f2` | `Holds values above 1 (bright highlights, accumulated light). The default, and what bloom and feedback need.` | 16 | `holds values above 1, the default` | 6 |
| `f4` | `Full precision. Rarely needed; costs twice the memory of 16-bit.` | 10 | `full precision, twice the memory` | 5 |

All three shipped tooltips carry a second clause (`—`, `.`, `;`); all three replacements are one
clause and in budget, verified by running the clause check over them. 39 words in, 16 out, **23 net
cut** — which with the call sites' 196 is the gear's 219.

### `widgets/pass_list.py` — the strip (finding #10)

| Site | Call | Current text | W | Replacement / dict | Verdict |
|---|---|---|---|---|---|
| `_settings_overlay` :98 | `set_tooltip` `text` (pos) | Pass settings — what it reads, what it draws into | 10 +clause | `Pass settings` | 10 -> 2 |
| `_draw_pass_tile` :108 | `preview_cell` `footer` (kw) | name | -- | `_UNMEASURABLE` | unmeasurable |
| `_draw_pass_tile` :108 | `preview_cell` `sublines` (kw) | sublines | -- | `_UNMEASURABLE` | unmeasurable |

The maintainer's words are the replacement verbatim: "'Pass settings' is enough."

### `tabs/document.py` — the panel (finding #32)

| Site | Call | Current text | W | Replacement / dict | Verdict |
|---|---|---|---|---|---|
| `_draw_auto_row` :109 | `text_colored` `text` (pos) | format_auto_value(value) | -- | `_UNMEASURABLE` | unmeasurable |
| `draw` :156 | `text_colored` `text` (pos) | x | 1 | unchanged | 1 |
| `_entry_row_label` :287 | `text_colored` `text` (pos) | label | -- | `_UNMEASURABLE` | unmeasurable |
| `_draw_entry_points` :329 | `set_tooltip` `text` (pos) | Document script error -- click to open and fix | 9 +clause | `Open the document script / Create the document script` | 9 -> 4 |
| `_draw_entry_points` :333 | `play_stop_toggle` `tooltip` (kw) | Stop the whole script (freeze all uniforms) | 7 | `Stop the whole script` | 7 -> 4 |

Two cuts here, both keyword or conditional sites the first draft could not see. `:329`'s
`open_tooltip` is a three-branch `IfExp` whose longest branch is 9 words and carries a ` -- `; its
error branch's text folds into the button's existing `STATE_ERROR` colour, which already says it,
leaving `Open the document script` and `Create the document script`. `:333`'s
`play_stop_toggle(tooltip=...)` is a two-branch `IfExp` reaching the helper by KEYWORD — the exact
shape round 1's finding 1 is about — and becomes `Stop the whole script` / `Resume the whole script`.

### `widgets/uniform.py` — the uniform rows (round 1, finding 1)

| Site | Call | Current text | W | Replacement / dict | Verdict |
|---|---|---|---|---|---|
| `uniform_name_label` :52 | `clickable_label` `label` (pos) | name | -- | `_UNMEASURABLE` | unmeasurable |
| `uniform_name_label` :52 | `clickable_label` `tooltip` (kw) | Jump to declaration | 3 | unchanged | 3 |
| `_draw_play_stop` :168 | `play_stop_toggle` `tooltip` (kw) | Can't play a single uniform while the whole script is stopped | 11 | `Whole script is stopped / Stop this uniform / Resume this uniform` | 11 -> 4 |
| `draw_ui_uniform` :194 | `clipped_caption` `text` (pos) | format_auto_value(current_value) | -- | `_UNMEASURABLE` | unmeasurable |
| `draw_ui_uniform` :220 | `clipped_caption` `text` (pos) | format_auto_value(current_value) | -- | `_UNMEASURABLE` | unmeasurable |

**This file contributed zero rows to the first census.** All five of its sites reach their helper by
keyword or positionally through a helper the first call set did not carry. `:168` is the worst
string in the codebase's tooltips at 11 words, and it was invisible. The three branches become
`Whole script is stopped`, `Stop this uniform` and `Resume this uniform` — the first keeps the fact
that the document-wide stop is what blocks the button, which is the only branch a name cannot carry
on its own.

### Every other file

**`shaderbox/exporters/telegram.py`**\n\n| Site | Call | Current text | W | Replacement / dict | Verdict |
|---|---|---|---|---|---|
| `emoji_button` :315 | `set_tooltip` `text` (pos) | Click to change emoji | 4 | unchanged | 4 |
| `_draw_pack_row` :446 | `draw_link` `label` (pos) | t.me/addstickers/{} | 2 | unchanged | 2 |
| `_draw_pack_row` :452 | `text_colored` `text` (pos) | no packs yet | 3 | unchanged | 3 |
| `_draw_status_slot` :761 | `text_colored` `text` (pos) | {}x{} · {}s · {} KB | 9 | `_OVER_BUDGET`: derived stat line; 5 interpolations + 4 authored words | 9 (budget 4) |
\n**`shaderbox/exporters/youtube.py`**\n\n| Site | Call | Current text | W | Replacement / dict | Verdict |
|---|---|---|---|---|---|
| `_draw_controls` :503 | `draw_link` `label` (pos) | Open in YouTube Studio | 4 | `_OVER_BUDGET`: a link's destination name, not a control label | 4 (budget 2) |
\n**`shaderbox/popups/emoji_picker.py`**\n\n| Site | Call | Current text | W | Replacement / dict | Verdict |
|---|---|---|---|---|---|
| `_draw_body` :54 | `separator_text` `label` (pos) | group.name | -- | `_UNMEASURABLE` | unmeasurable |
| `_draw_body` :65 | `set_tooltip` `text` (pos) | entry.name | -- | `_UNMEASURABLE` | unmeasurable |
| `_draw_body` :72 | `text_colored` `text` (pos) | (no matches) | 2 | unchanged | 2 |
\n**`shaderbox/popups/help.py`**\n\n| Site | Call | Current text | W | Replacement / dict | Verdict |
|---|---|---|---|---|---|
| `_draw_body` :84 | `set_tooltip` `text` (pos) | Open a document's shader and click into the editor first (so the caret is positioned) | 15 | `_OVER_BUDGET`: disabled-state reason; a name cannot carry it | 15 (budget 5) |
\n**`shaderbox/popups/lib_picker/__init__.py`**\n\n| Site | Call | Current text | W | Replacement / dict | Verdict |
|---|---|---|---|---|---|
| `_draw_body` :116 | `text_colored` `text` (pos) | Right-click for actions | 3 | unchanged | 3 |
| `_draw_body` :140 | `set_tooltip` `text` (pos) | Click into the code editor first (so the caret is positioned) | 11 | `_OVER_BUDGET`: same disabled state, picker's Insert button | 11 (budget 5) |
\n**`shaderbox/popups/lib_picker/preview.py`**\n\n| Site | Call | Current text | W | Replacement / dict | Verdict |
|---|---|---|---|---|---|
| `draw_preview` :20 | `text_colored` `text` (pos) | (no function selected) | 3 | unchanged | 3 |
| `draw_preview` :26 | `draw_copyable_text` `label` (pos) | str(rel) | -- | `_UNMEASURABLE` | unmeasurable |
| `draw_preview` :26 | `draw_copyable_text` `tooltip` (kw) | Click to copy file path | 5 | unchanged | 5 |
| `_draw_function_tag_editor` :92 | `text_colored` `text` (pos) | Existing: | 1 | unchanged | 1 |
\n**`shaderbox/popups/lib_picker/search.py`**\n\n| Site | Call | Current text | W | Replacement / dict | Verdict |
|---|---|---|---|---|---|
| `draw_search_row` :60 | `text_colored` `text` (pos) | {} / {} | 3 | unchanged | 3 |
| `draw_search_row` :69 | `text_colored` `text` (pos) | 'matching: ' + ' '.join((f'#{t}' for t in matched)) | -- | `_UNMEASURABLE` | unmeasurable |
\n**`shaderbox/popups/lib_picker/tree.py`**\n\n| Site | Call | Current text | W | Replacement / dict | Verdict |
|---|---|---|---|---|---|
| `_draw_inline_new_input` :211 | `text_colored` `text` (pos) | label | -- | `_UNMEASURABLE` | unmeasurable |
| `_draw_function_leaf` :350 | `text_colored` `text` (pos) | sep + _ellipsize(first_doc_line, avail) | -- | `_UNMEASURABLE` | unmeasurable |
\n**`shaderbox/popups/settings.py`**\n\n| Site | Call | Current text | W | Replacement / dict | Verdict |
|---|---|---|---|---|---|
| `_draw_body` :72 | `separator_text` `label` (pos) | General | 1 | unchanged | 1 |
| `_draw_body` :73 | `label_row` `label` (pos) | Target FPS | 2 | unchanged | 2 |
| `_draw_body` :86 | `separator_text` `label` (pos) | Editor | 1 | unchanged | 1 |
| `_draw_body` :100 | `label_row` `label` (pos) | Font size | 2 | unchanged | 2 |
| `_draw_body` :108 | `label_row` `label` (pos) | Tab size | 2 | unchanged | 2 |
| `_draw_body` :116 | `label_row` `label` (pos) | Line spacing | 2 | unchanged | 2 |
| `_draw_body` :127 | `separator_text` `label` (pos) | Integrations | 1 | unchanged | 1 |
| `_draw_body` :142 | `text_colored` `text` (pos) | {} — {} | 3 +clause | `_OVER_BUDGET`: derived: exporter name + its unavailable reason | 3 (budget 4) |
| `_draw_body` :158 | `separator_text` `label` (pos) | Library | 1 | unchanged | 1 |
| `_draw_body` :162 | `separator_text` `label` (pos) | Keyboard | 1 | unchanged | 1 |
| `_draw_copilot_config` :298 | `text_colored` `text` (pos) | label | -- | `_UNMEASURABLE` | unmeasurable |
| `_draw_copilot_config` :307 | `help_marker` `text` (pos) | hint | -- | `_UNMEASURABLE` | unmeasurable |
\n**`shaderbox/tabs/code.py`**\n\n| Site | Call | Current text | W | Replacement / dict | Verdict |
|---|---|---|---|---|---|
| `_draw_error_strip` :206 | `text_colored` `text` (pos) | {} errors  (F8: next) | 4 | unchanged | 4 |
| `draw_chrome` :381 | `text_colored` `text` (pos) | No file open | 3 | unchanged | 3 |
| `draw_chrome` :384 | `text_colored` `text` (pos) | No document selected | 3 | unchanged | 3 |
| `draw_chrome` :393 | `text_colored` `text` (pos) | {}:{} | 3 | unchanged | 3 |
| `draw_chrome` :405 | `text_colored` `text` (pos) | message | -- | `_UNMEASURABLE` | unmeasurable |
| `draw_chrome` :421 | `draw_copyable_text` `label` (pos) | str(local_file_path) | -- | `_UNMEASURABLE` | unmeasurable |
| `draw_chrome` :433 | `text_colored` `text` (pos) | tab_label(app, tab) | -- | `_UNMEASURABLE` | unmeasurable |
| `draw` :676 | `set_tooltip` `text` (pos) | {}: {} | 3 | unchanged | 3 |
\n**`shaderbox/ui_primitives.py`**\n\n| Site | Call | Current text | W | Replacement / dict | Verdict |
|---|---|---|---|---|---|
| `play_stop_toggle` :211 | `set_tooltip` `text` (pos) | tooltip | -- | `_UNMEASURABLE` | unmeasurable |
| `clipped_caption` :447 | `set_tooltip` `text` (pos) | text | -- | `_UNMEASURABLE` | unmeasurable |
| `help_marker` :458 | `text_colored` `text` (pos) | (?) | 1 | unchanged | 1 |
| `setup_steps` :564 | `draw_link` `label` (pos) | url | -- | `_UNMEASURABLE` | unmeasurable |
| `small_caption` :699 | `text_colored` `text` (pos) | text | -- | `_UNMEASURABLE` | unmeasurable |
| `gauge_bar` :842 | `set_tooltip` `text` (pos) | tooltip | -- | `_UNMEASURABLE` | unmeasurable |
| `preview_cell` :1045 | `text_colored` `text` (pos) | label | -- | `_UNMEASURABLE` | unmeasurable |
| `preview_cell` :1053 | `text_colored` `text` (pos) | text | -- | `_UNMEASURABLE` | unmeasurable |
| `preview_cell` :1074 | `set_tooltip` `text` (pos) | Delete | 1 | unchanged | 1 |
| `label_row` :1096 | `row_label` `label` (pos) | label | -- | `_UNMEASURABLE` | unmeasurable |
| `draw_copyable_text` :1161 | `set_tooltip` `text` (pos) | tooltip | -- | `_UNMEASURABLE` | unmeasurable |
| `draw_link` :1189 | `set_tooltip` `text` (pos) | Click to open + copy | 5 | unchanged | 5 |
| `clickable_label` :1252 | `set_tooltip` `text` (pos) | tooltip | -- | `_UNMEASURABLE` | unmeasurable |
\n**`shaderbox/widgets/copilot_chat.py`**\n\n| Site | Call | Current text | W | Replacement / dict | Verdict |
|---|---|---|---|---|---|
| `_tooltip_stat_row` :394 | `text_colored` `text` (pos) | label | -- | `_UNMEASURABLE` | unmeasurable |
| `_draw_snippet_tooltip` :407 | `text_colored` `text` (pos) | No tool calls. | 3 | unchanged | 3 |
| `_draw_snippet_tooltip` :421 | `text_colored` `text` (pos) | nums | -- | `_UNMEASURABLE` | unmeasurable |
| `_draw_bubble` :535 | `set_tooltip` `text` (pos) | Copy | 1 | unchanged | 1 |
| `_draw_bubble` :545 | `set_tooltip` `text` (pos) | Revert this turn's changes | 4 | unchanged | 4 |
| `_draw_pending_action` :571 | `text_colored` `text` (pos) | You provided: {} | 3 | unchanged | 3 |
| `_draw_pending_action` :575 | `text_colored` `text` (pos) | You chose: No | 3 | unchanged | 3 |
| `_draw_pending_action` :577 | `text_colored` `text` (pos) | ({}) | 3 | unchanged | 3 |
| `_draw_pending_action` :615 | `set_tooltip` `text` (pos) | Recover from trash | 3 | unchanged | 3 |
| `_draw_top_bar` :679 | `set_tooltip` `text` (pos) | Layout: {} | 2 | unchanged | 2 |
| `_draw_top_bar` :685 | `gauge_bar` `tooltip` (pos) | tooltip | -- | `_UNMEASURABLE` | unmeasurable |
\n**`shaderbox/widgets/details.py`**\n\n| Site | Call | Current text | W | Replacement / dict | Verdict |
|---|---|---|---|---|---|
| `draw_file_details` :44 | `draw_copyable_text` `label` (pos) | details.path | -- | `_UNMEASURABLE` | unmeasurable |
| `draw_resolution_details` :65 | `label_row` `label` (pos) | Width | 1 | unchanged | 1 |
| `draw_resolution_details` :69 | `label_row` `label` (pos) | Height | 1 | unchanged | 1 |
| `draw_resolution_details` :88 | `row_label` `label` (pos) | Presets | 1 | unchanged | 1 |
| `draw_media_details` :110 | `label_row` `label` (pos) | Output | 1 | unchanged | 1 |
| `draw_media_details` :119 | `label_row` `label` (pos) | Quality | 1 | unchanged | 1 |
| `draw_media_details` :125 | `label_row` `label` (pos) | FPS | 1 | unchanged | 1 |
| `draw_media_details` :127 | `label_row` `label` (pos) | Duration | 1 | unchanged | 1 |
\n**`shaderbox/widgets/document_grid.py`**\n\n| Site | Call | Current text | W | Replacement / dict | Verdict |
|---|---|---|---|---|---|
| `draw_document_preview_button` :24 | `preview_cell` `footer` (kw) | ui_document.ui_state.ui_name | -- | `_UNMEASURABLE` | unmeasurable |
\n**`shaderbox/widgets/media_ops.py`**\n\n| Site | Call | Current text | W | Replacement / dict | Verdict |
|---|---|---|---|---|---|
| `draw_video_filters` :19 | `row_label` `label` (pos) | Window | 1 | unchanged | 1 |
| `draw_video_filters` :27 | `row_label` `label` (pos) | Sigma | 1 | unchanged | 1 |
\n
### Census reconciliation with the parent spec and with round 1

| Figure | Parent | Draft 1 | This run | Verdict |
|---|---|---|---|---|
| `help_marker` | 7 | 7 | 7 | matches |
| `set_tooltip` | 19 | 19 | 19 | matches |
| `separator_text` | 10 | 10 | 10 | matches |
| `text_colored` + `FG_DIM` | 31 | 32 | 32 | +1 vs the parent, W-A's `"x"` |
| `label_row` / `row_label` | not stated | 17 / 4 | 17 / 4 | matches draft 1 |
| the seven keyword-taking helpers | not in scope | **not walked** | 17 | round 1, finding 1 |
| total sites | 62 (round 3's throwaway) | 89 | **106** | see below |
| unmeasurable | 26 | 26 | **35** | see below |
| over budget | 11 | 12 | **17** (+ 9 clause = 18 distinct) | see below |

**The parent's four call-name counts all reproduce exactly**, at three commits, and the +1 on
`text_colored` is W-A's canvas `"x"` separator (31 at `faccf0e` / `a246a19` / `73e65ac`, 32 from
`78bd1bf` on). The parent's round-3 correction from a grep count to an AST count was right and its
number was right when written.

**89 → 106 is round 1's finding 1**, and it is the only change that is a domain fix rather than a
recount. Seventeen rows come from the seven `ui_primitives` helpers now in the call set, of which
five reach their helper by keyword (`play_stop_toggle` x2, `clickable_label`, `preview_cell` x2).
Four of the seventeen are over budget, and three of those four are authored tooltips in
`widgets/uniform.py` and `tabs/document.py` — the strings finding #10 is literally about, sitting
outside a gate written to enforce finding #10.

**26 → 35 unmeasurable** is the same expansion: the new helpers' own definition sites forward a
parameter (9 of them), and the new caller sites include seven whose argument is a `Call`, an
`Attribute` or a name bound to one.

**12 → 17 over budget, plus 9 clause failures.** Five of the increase are the new keyword and
conditional sites; the `IfExp` scoring rule (§ 3) accounts for three more that draft 1 recorded as
unmeasurable. The clause check is new in this round and is what catches the four `;`-joined strings
draft 1 was about to ship, plus five shipped ones the word count alone passed.

**62 vs anything is not attributable.** Round 3's throwaway was not kept and this wave will not
guess at its internals; the parent's own bullet names `label_row` and `row_label` as gate inputs
without counting them, which is most of the gap.

The parent's headline claim survives all of it: all four ledger strings are flagged — the gear's
Reads tooltip (31), its empty state (10), its size label (6) and the strip's gear tooltip (10).

## Files touched

| File | Change |
|---|---|
| `shaderbox/popups/pass_settings.py` | `modal_window` gains `always_auto_resize \| no_scrollbar` and passes `0.0` for the height; six `help_marker` strings cut or deleted; the Reads section tooltip and its `(?)` marker deleted; the empty state cut; the size label cut to `size` and the slider's format string carries `%.0f%% · WxH`; two `separator_text` labels cut; `_FORMATS`' third column rewritten; the three comments that narrate the removed mechanisms deleted with them. |
| `shaderbox/widgets/pass_list.py` | the gear tooltip cut to `Pass settings`. |
| `shaderbox/tabs/document.py` | `_draw_auto_row` → `_draw_auto_block`, vertical rows against `SIZE.AUTO_NAME_W`; inside the surviving `if auto_hashes:` guard the `same_line(spacing=SPACE.XL)` goes and the guarded call moves below the sort row's existing `imgui.dummy((0, SPACE.MD))`; `open_tooltip` cut to two branches; the whole-script `play_stop_toggle(tooltip=...)` cut. |
| `shaderbox/widgets/uniform.py` | the three-branch `play_stop_toggle(tooltip=...)` cut to `Whole script is stopped` / `Stop this uniform` / `Resume this uniform`. |
| `shaderbox/help_content.py` | `ENGINE_UNIFORM_DOCS["u_pass_iteration"]`'s text becomes `"which run this is, 0-based (see Runs)"`; `help_sections()` gains a `key="pass_settings"` section carrying the two facts the deleted `sampling` / `edges` markers held. |
| `shaderbox/theme.py` | `SIZE.PASS_SETTINGS_H` deleted; `SIZE.AUTO_NAME_W: int = 128` added beside `CANVAS_FIELD_W`. `PASS_SETTINGS_W` stays. |
| `tests/test_ui_prose_budget.py` | new: the gate. |
| `.claude/skills/imgui-ui/SKILL.md` | § 2's budget table gains a pointer line naming `tests/test_ui_prose_budget.py` as the enforcer, so the skill's reader learns the rule is checked. No budget changes; § 2 already carries D1. |

`ui_primitives.py` is **not** touched. The parent's Files line lists it for "`modal_window`
auto-size", but `modal_window` already forwards flags and already takes the size the caller gives
it, so the auto-size is a call-site change. Verified by reading its body: `flags` reaches
`begin_popup_modal` unchanged, and `set_next_window_size` is called before it with whatever tuple
the caller passed. Adding an `auto_size: bool` parameter that only sets a flag the caller can
already pass would be surface with no consumer, which `conventions.md`'s speculative-machinery rule
cuts at design review.

`help_content.py` IS touched, by § Design decisions 9a and 9b — but it stays outside the GATE:
verified to contain zero calls to any of the six gate call names, so no exemption has to be
remembered for it.

---

## Tests

Each named with its falsifier: the bug that makes it go red. The four mutations the parent requires
are covered by tests 2, 3, 6 and 7 respectively.

All live in `tests/test_ui_prose_budget.py`. No GL context, no imgui context, no `app` fixture — the
module reads source text and never imports `shaderbox` beyond `theme` for nothing at all. It is pure
`ast` over files, so it runs in `make test` on any box.

### `test_the_walk_finds_the_known_call_sites`

The collector's own falsifier, copied in spirit from
`test_worker_daemon_contract.py::test_the_enumeration_actually_finds_the_known_workers`. Asserts the
walk returns at least one site in each of `shaderbox/popups/pass_settings.py`,
`shaderbox/widgets/pass_list.py`, `shaderbox/tabs/document.py` and `shaderbox/ui_primitives.py`, and
that the total site count is at least 60.

**Falsifier:** rename `help_marker` to `_help_marker` in the collector's target set, or break the
`Attribute` branch that matches `imgui.set_tooltip`, and the walk returns a short list; the
assertions go red. Without this test, every parametrized assertion below would pass vacuously on an
empty collection — the "checker that quietly narrows its own domain" family, which
`~/.claude/CLAUDE.md` names as the most expensive bug family and which a floor assertion is the
cheapest guard against.

The floor of 60 is deliberately well below today's 89: a number that tracked the census exactly
would go red on every legitimate new string, which trains the next reader to bump it without
looking. 60 goes red only on a collector that broke.

### `test_every_measured_site_is_within_budget`

Parametrized over every measurable site not keyed in `_OVER_BUDGET`. Asserts
`word_count <= BUDGET[call_name]`, with a failure message naming the file, line, call, the string
and both numbers.

**Falsifier — the parent's mutation 1, "an over-budget literal added at a fresh site → red".** Add
`help_marker("How big this pass's own image is relative to the canvas, which matters for blurs")`
(14 words) anywhere in `shaderbox/`, and this test goes red naming that line. Verified the other
direction too: at the pre-cut tree this test fails on 12 sites, and at the post-cut tree on 0 — the
9 cut here, plus the 3 that gain an `_OVER_BUDGET` entry at their measured count, which is what
turns their exemption into a written decision rather than a green gate that never saw them.

### `test_a_keyword_supplied_argument_is_scored`

A unit test of the collector. Parses a fixture containing `help_marker(text="a b c d e f g h i")`
(9 words) and asserts the collector returns one row scoring 9, not zero rows.

**Falsifier — round 1's finding 1, the gate's whole domain hole.** Remove the keyword lookup from
§ 2's rule and this goes red. Under the first specification the site produced no row at all: no
score, no exemption, no parametrized id — it fell out of both dicts' assertions silently. Four live
over-budget authored tooltips sat in that hole, three of them in `widgets/uniform.py`, a file that
contributed zero rows to the first census. A tree-level assertion cannot falsify this after the cut
(every keyword site will be in budget), so the fixture is the falsifier.

### `test_an_argument_supplied_by_neither_position_nor_keyword_is_unmeasurable`

Parses a fixture calling a gate function through `**kwargs` and asserts the site is reported
UNMEASURABLE rather than skipped.

**Falsifier:** restore the `continue` the first draft's rule implied for a missing argument and this
goes red. The distinction it pins: a call that omits a genuinely optional parameter contributes no
row (there is no string), while a call that supplies it through a form the walk cannot read is an
exemption someone must write down. Collapsing the two is how a domain narrows silently.

### `test_no_scored_string_joins_a_second_clause`

Parametrized over every measurable site not keyed in `_OVER_BUDGET`. Asserts the string contains
none of `;`, ` — `, ` -- `.

**Falsifier — round 1's finding 5.** Restore `share of the canvas; output is always full` (8 words,
in budget, two clauses) and this goes red while `test_every_measured_site_is_within_budget` stays
green. That pair is the point: D1 has a count half and a clause half, § 2 states the clause half as
a mechanical test ("A string with an em-dash-joined second clause is over budget by construction"),
and a word count cannot see it. The first draft shipped four semicolon-joined replacements under a
gate that scored words only — D1's clause half unenforced by the test whose purpose is enforcing D1.
Nine shipped strings fail it today.

### `test_an_ifexp_argument_scores_its_worst_branch`

A unit test. Parses `help_marker("a" if p else "b c d e f g h i j")` and asserts the score is 9.

**Falsifier:** restore the first draft's treatment of a conditional as UNMEASURABLE and this goes
red. Three live authored tooltips are `IfExp`s — the gear's size marker at 41 words on its output
branch, `tabs/document.py`'s `open_tooltip`, and `widgets/uniform.py`'s play/stop tooltip — so
"unmeasurable" meant "exempt" for exactly the strings this wave exists to cut. Scoring the worst
branch is the only bound that is honest when the branch a user sees is not knowable statically.

### `test_no_label_carries_an_interpolation`

Parametrized over every `label_row` / `row_label` site whose scored argument is a `JoinedStr`.
Asserts the list is empty. Separate from the count assertion per § Design decisions 6.

**Falsifier — the parent's mutation 2, "a `FormattedValue` in a `label_row` label → red".** Restore
`label_row(app.font_12, f"size ({w}, {h})", ...)` and this goes red. It also goes red under a
one-word interpolation like `label_row(app.font_12, f"{name}", ...)` that the count assertion would
pass, which is the whole reason it is its own test: finding #7's defect is variable LENGTH in a
fixed column, and length is a property of the expression rather than of one rendering.

### `test_the_label_helpers_are_read_at_the_right_argument`

Asserts, via `inspect.signature` on the real `ui_primitives.label_row`, `row_label` and
`small_caption`, that each one's parameter at the index the collector reads is named `label` or
`text`, and that parameter 0 is named `font`.

**Falsifier:** reorder `row_label`'s parameters to `(label, font, label_w)` and this goes red. Under
today's gate without it, that reorder would silently move every label out of the measured position
and every `label_row` site would become an unmeasurable `Attribute` (`app.font_12`) — the gate would
go green while measuring nothing, which is precisely how this wave's own first census produced 17
false unmeasurables. The bug happened during authorship; the test exists because it happened.

### `test_a_name_rebound_by_augassign_is_not_resolved`

A unit test of the resolver itself, not a walk over the package. Parses a two-line fixture source
(`text = "a"` / `text += " b c"` / `help_marker(text)`) with `ast.parse` and asserts the resolver
returns UNMEASURABLE rather than a 1-word score.

**Falsifier:** delete the `AugAssign` branch from § 4's algorithm and this goes red. It is a unit
test rather than a tree assertion because the shape it guards has exactly one instance in the
codebase today and that instance is outside the gate's call set — so a tree-level assertion would
be green whether or not the branch exists, which verifies nothing. The fixture is the falsifier the
tree cannot provide.

### `test_no_site_is_both_measured_and_unmeasurable_listed`

Asserts `_UNMEASURABLE`'s keys are disjoint from the set of `(module, function)` pairs whose every
site the scorer read successfully.

**Falsifier:** add `("shaderbox/popups/pass_settings.py", "_draw_repeat")` to `_UNMEASURABLE`, whose
sites are all measurable after the cut, and this goes red. Without it, an entry could silently
suppress a function's measurement — the escape hatch would be exactly as wide as whoever last edited
the dict wanted it, with no signal. `_OVER_BUDGET` needs no equivalent, because its keys carry a
count that only a measurable site can produce.

### `test_every_unmeasurable_site_is_listed`

Parametrized over every site the scorer returned UNMEASURABLE for. Asserts its `(module, enclosing
function)` key is present in `_UNMEASURABLE`. The failure message prints the key and the unparsed
argument expression, so adding an entry is a copy-paste with a reason to write.

**Falsifier — the parent's mutation 3, "a site removed from the allowlist without being measured →
red".** Delete the `("shaderbox/ui_primitives.py", "play_stop_toggle")` entry and this goes red
naming that function. It also fires the other way, which is the point: introduce a new
`set_tooltip(some_variable)` in a fresh function and the site is neither measurable nor listed, so
the gate refuses it until someone either makes it a literal or writes down why it cannot be one. An
unmeasured site is then a deliberate entry, never a blind spot.

### `test_every_allowlist_entry_still_names_a_real_site`

Parametrized over both dicts' keys. For `_UNMEASURABLE`, asserts the key appears among the
unmeasurable sites the walk found. For `_OVER_BUDGET`, asserts the `(module, function)` half appears
among the measurable sites **and** that some site there scores exactly the recorded count.

**Falsifier — the parent's mutation 4, "an allowlist entry that no longer exists in the code → red,
so the allowlist cannot rot".** Delete `gauge_bar` from `ui_primitives.py`, or turn its forwarded
`tooltip` parameter into a literal, and its `_UNMEASURABLE` entry goes stale; this test names it.
The `_OVER_BUDGET` half is the sharper one: shorten `popups/help.py`'s tooltip from 15 words to 6
and the entry's count no longer matches, so the exemption for a string that no longer needs one goes
red rather than quietly covering whatever replaced it. This is what makes the lists a ledger rather
than a graveyard, and it is the reason the key is `(module, function)` and not a line number — a
line-keyed entry would go stale on every unrelated edit above it and this test would cry wolf until
someone disabled it.

### `test_the_format_tooltips_are_within_the_help_budget`

Imports `_FORMATS` from `popups.pass_settings` and asserts each entry's third element is at most 8
words, and each second element (the menu label) at most 2.

**Falsifier:** restore any of the three original tooltips (13, 16 and 10 words) and this goes red.
It exists because `_FORMATS` is reached through a `Subscript`, so the AST walk sees an unmeasurable
argument and the table's contents would otherwise be entirely outside the gate — a data table is
exactly where over-budget prose can hide from a call-site checker. The parent asks for "`_FORMATS`
tooltips one clause each"; this is what makes that mechanical.

The same pattern generalizes: any future table of UI strings reached by subscript needs its own
direct assertion, because no call-site walk can follow it. The allowlist entry for
`_draw_target`'s `help_marker` names this test as the reason it is safe to allowlist.

### `test_the_auto_name_column_fits_every_engine_uniform`

Imports `ENGINE_DRIVEN_UNIFORMS` from `shaderbox.core` and `SIZE.AUTO_NAME_W` from `theme`, and
asserts every name in the set (minus `TABLE_UNIFORMS`, which never gets a row) is at most **19**
characters.

The bound is `floor(SIZE.AUTO_NAME_W / 6.5508)` for the 12px monospace face: the app loads
`AnonymousPro-Regular.ttf`, whose advance is 1118/2048 em, so one character at 12px is 6.5508px and
`128 / 6.5508 = 19.54`. 19 characters is 124.5px and fits; 20 is 131.0px and does not. (The font
metric is round 1's measurement, taken with fontTools against the shipped file; this session has no
fontTools installed and did not re-derive it. It is recorded as a relayed number, and the test
carries the division rather than a bare 19 so the arithmetic is visible at the assertion.)

**Falsifier:** add a 20-character engine uniform to `ENGINE_DRIVEN_UNIFORMS` and this goes red. The
first draft set the bound at 20, which admits a name that does not fit, and named the wrong failure
mode: it claimed the name would overlap the value column. It would not. `uniform_name_label` draws
through `clickable_label`, whose body calls `_ellipsize(label, width)`, so an over-wide name is
**silently truncated to `u_pass_iteratio...`** — data loss rather than overlap, invisible to every
manual step in this spec, and exactly the defect the test is supposed to prevent. A checker whose
own bound admits its defect is the family this wave's § 1 exists to guard against, and this one had
it.

This is the check that keeps finding #32 from recurring: the finding exists because 068 added two
names to a row sized for three and nothing complained. A character bound rather than a pixel
measurement, because a pixel width needs a font and a font needs a context; the character count is
the property the layout depends on and is assertable headlessly.

### The gear's auto-resize and the size row's format: manual

Neither is testable headlessly. A window's resolved rect exists only inside a live imgui frame, and
a slider's format string is applied by the C++ widget at draw time — no return value carries the
rendered text. Both go to § Manual verification, and the review pass reads the diff for the flag and
the format string. This is stated rather than papered over: `dev_flow.md` requires each check to
fail for exactly one reason, and a test that constructed an imgui context to assert a window height
would be testing imgui.

---

## Manual verification

The parent's W-B line is: "no popup scrolls; every `(?)` reads in one glance; the engine-uniform
block fits at the narrowest panel width." Expanded into steps that fail for exactly one reason each,
run with `make run` on the six-pass Radiance Cascades example.

1. **The gear does not scroll, at its largest content.** Open the gear on `cascade`, the pass with
   the most wired inputs (three, `u_scene` / `u_df` / `u_prev`, so the Reads section is at its
   tallest). Look at the right edge of the popup. **Pass:** no scrollbar track is drawn. **Fail:** a
   track appears, meaning the auto-resize flag did not reach `begin_popup_modal`. Falsifies the
   `always_auto_resize` half of § Design decisions 7 specifically; the height token's removal is
   verified by the tree compiling at all, since a deleted token with a live reader is a pyright
   error.

2. **The gear's height follows its content, on a fresh ini AND on a stale one.** `modal_window`
   seeds the size through `set_next_window_size(..., Cond_.first_use_ever)` and `app.py` points
   imgui's ini at `app_data_dir()/imgui.ini`, which on any developer box already carries a saved
   440x400 rect for the `Pass settings` popup. Whether `always_auto_resize` overrides an
   ini-restored rect is the one claim in § Design decisions 7 not traced to the installed stub (the
   quoted text is about the resize, not about precedence), so the step is run twice. **First**
   delete `imgui.ini` from the app-data dir, launch, open the gear on `cascade`, note the bottom
   edge, close, open it on `seed` (one input, `u_scene`). **Then** quit and repeat with the ini in
   place. **Pass:** in both runs the `seed` popup is visibly shorter than the `cascade` one.
   **Fail:** equal heights in either run — the size is still winning, which is what `fixed_size=True`,
   a stale non-zero height, or a restored rect beating the flag would each do. Round 1's finding 11:
   without the two-run split this step could pass or fail for the wrong reason on the only box it
   will ever be run on.

3. **The size row fits its column and reads the resolution.** In the same gear, look at the size
   row. **Pass:** the label column reads `size` and nothing else; the slider's track carries
   `100% · 512x512` (or the pass's own percent and dims); no text crosses the label column's right
   edge. **Fail:** anything numeric in the label column — finding #7 unfixed — or a bare `100%` in
   the slider, meaning the format string lost its suffix.

4. **The output pass's size row still reads.** Click the `composite` tile so it is the output, open
   its gear. **Pass:** the slider is greyed and still shows `100% · 512x512`. **Fail:** the greyed
   slider shows nothing, meaning `begin_disabled` suppressed the format string, in which case the
   dims move to a dim caption after the widget (§ 2 allows either placement).

5. **Every `(?)` reads in one glance.** Open the gear and hover each remaining `(?)`. There are
   three: name, format, size. **Pass:** each tooltip is one line, no wrapping at the tooltip's own
   24-em wrap width. **Fail:** any of them wraps to a second line, which at that width means well
   over eight words.

6. **The markers that should be gone are gone.** In the same gear, confirm there is no `(?)` beside
   `sampling`, none beside `edges`, and that the `Reads` heading is not followed by `(?)` and shows
   no tooltip on hover. **Pass:** three absences. **Fail:** any survivor, meaning a deletion was
   missed.

7. **The empty state fits.** Open the gear on a pass declaring no `sampler2D` (add a pass; its stub
   declares none). **Pass:** under `Reads`, the line reads `no sampler2D uniforms` and ends well
   inside the popup's width. **Fail:** the old sentence, or any text reaching the right edge.

8. **The strip's gear tooltip is the control's name.** Hover the gear icon on any tile. **Pass:**
   `Pass settings`, two words, one line. **Fail:** the em-dash passage.

9. **The engine uniforms are a block, and they fit the narrowest panel.** On the Document tab, drag
   the main splitter left until the right panel is at its minimum usable width. Look under the sort
   combo. **Pass:** five rows (`u_time`, `u_aspect`, `u_resolution`, `u_pass_iteration`,
   `u_pass_iterations`), one per line, names left-aligned in one column and values starting at the
   same x on every row; nothing is clipped at the panel's right edge; the sort combo and its
   direction button are alone on their row. **Fail:** any of — the names still sharing a line with
   the sort combo, ragged value starts, or a clipped name. This is finding #32's exact report,
   inverted.

10. **The hover bridge survives the reshape.** With a shader tab open in the editor, hover
    `u_resolution` in the block. **Pass:** the declaration highlights in the editor, exactly as
    before this wave. **Fail:** no highlight, meaning `uniform_name_label` was replaced by a plain
    `text_colored` and the block re-committed feature 008's special-case-row mistake. The reader
    that makes this pass is `uniform_name_label`'s `is_item_hovered` branch calling
    `_locate_uniform_declaration`; the wire is cut if the call is gone.

11. **The sorted list is unchanged.** Change the sort key and direction. **Pass:** only the
    non-engine uniforms reorder; the engine block stays put, in `ENGINE_DRIVEN_UNIFORMS`' own order.
    **Fail:** the block reorders, meaning `auto_hashes` leaked into `sort_uniform_hashes`.

12. **The script entry-point tooltip.** Hover the `open` button on the Script row, in all three
    states: no script, script present, script with an error. **Pass:** `Create the document script`
    / `Open the document script` / `Open the document script`, and in the error state the button
    itself is `STATE_ERROR` red. **Fail:** the error state's tooltip still names the error in words,
    meaning the fold into the colour did not happen.

---

## Verified / corrected premises

Every citation and claim the parent spec's W-B section, findings #5 #7 #10 #32, and D1 make, checked
against the tree at `78bd1bf`. The parent cites symbols where it can; the line numbers below are the
real ones at that commit and are given only to show the check was run.

| Parent-spec or finding citation | Verdict |
|---|---|
| Census by AST: `help_marker` 7 sites | **Confirmed, at three commits.** The collector returns 7 at `faccf0e`, `a246a19`, `73e65ac` and `78bd1bf`. |
| Census by AST: `set_tooltip` 19 sites | **Confirmed**, 19 at all four commits. |
| Census by AST: `separator_text` 10 sites | **Confirmed**, 10 at all four commits. |
| Census by AST: `text_colored` + `FG_DIM` 31 sites, corrected from a grep count in round 3 | **Confirmed at the commit it was written against, corrected for today.** 31 at `faccf0e`, `a246a19` and `73e65ac`; **32** at `78bd1bf`. The new site is `tabs/document.py::draw`'s `"x"` between the canvas width and height fields, added by W-A, one word, in budget. The round-3 correction from grep to AST was right, and the number was right when written. |
| "round 3's throwaway implementation measured 62 sites, 26 in the allowlist, 11 flagged including all four ledger strings" | **Confirmed on the allowlist and the ledger strings, corrected on the total and on the flag count.** This census finds **26** unmeasurable, identical, and all four ledger strings among the flagged. The total is **89**, not 62. The gap is not attributable without round 3's throwaway, which was not kept; what is verifiable is that the four call names the bullet gives numbers for total 67 at the commit it measured, and `label_row` + `row_label` add 21 that the same sentence names as gate inputs without counting. Flagged is **12**, not 11: the twelfth is `separator_text("Runs per frame")` at 3 words, which round 3 had no `separator_text` budget to score against (§ 2 states that one as prose, "1-2 words", not a number). Two independent implementations agreeing on 26 is the check that matters. |
| `pass_settings.py:128`'s "(?)" tooltip "is an implicit concatenation with an f-string and parses as `JoinedStr`" | **Confirmed as the mechanism, re-pointed.** At `78bd1bf` the `set_tooltip` is `pass_settings.py:142`, inside `_draw_inputs`; it is three adjacent literals of which the second is an f-string carrying `{_UNWIRED}`, so the whole expression is one `ast.JoinedStr`. Scored under § 3 it is **31 words**. The parent's `:128` was the pre-W-C location. |
| `pass_settings.py:178`'s `f"size ({w}, {h})"` "is the overflow #7 reported and the gate must refuse it" | **Confirmed as the string, re-pointed.** At `78bd1bf` it is `pass_settings.py:192`, a `label_row` second argument, `ast.JoinedStr` scoring **6 words** against a 2-word label budget, and carrying two `FormattedValue`s against § Design decisions 6's zero. It fails both assertions, which is the parent's requirement. |
| The gear popup is a fixed 440x400 and its content is taller (#7) | **Confirmed.** `theme.py:269-270` are `PASS_SETTINGS_W: int = 440` and `PASS_SETTINGS_H: int = 400`; `pass_settings.py:61-63` passes both to `modal_window` with no flags, so `begin_popup_modal` gets `flags=0` and imgui's default scrollbar applies. |
| `SIZE.PASS_SETTINGS_H` has no other reader, so deleting it is safe | **Confirmed.** `grep -rn PASS_SETTINGS_H shaderbox/ tests/` returns the definition and the one call site. `PASS_SETTINGS_W` has the same two and stays. |
| `modal_window` can pass window flags through, so auto-resize needs no primitive change | **Confirmed.** Its signature is `(label, size, flags=0, fixed_size=False)` and its body passes `flags` straight to `imgui_ctx.begin_popup_modal`; its own docstring names `no_scrollbar` as the example. The parent's Files line lists `ui_primitives.py` for "`modal_window` auto-size"; **corrected** — no edit is needed there, the change is at the call site. |
| `WindowFlags_.always_auto_resize` exists in this imgui-bundle build | **Confirmed.** `imgui_bundle/imgui/__init__.pyi::WindowFlags_.always_auto_resize`, `= 1 << 6`, documented "Resize every window to its content every frame". `no_scrollbar` is in the same enum. Both are already used in this codebase: `ui_primitives.py::rendering_overlay` sets `always_auto_resize`, `popups/examples.py` sets `no_scrollbar`. |
| The empty-Reads line is 57 chars and overflows (#5) | **Confirmed on both.** `pass_settings.py:149`'s literal is `nothing — declare a sampler2D uniform to read another pass`, 57 characters, **10 words**. The replacement `no sampler2D uniforms` is 21 characters and 3 words, under the finding's own target ("half the current length, e.g. 'no sampler2D uniforms'") and under D1's 4. |
| "eight help markers / tooltips, each two to four sentences" in the gear (#5) | **Confirmed as eight, with the composition corrected.** The gear has **six** `help_marker` calls (name, format, size, sampling, edges, runs) plus **one** section `set_tooltip` (Reads) plus **one** `separator_text` label carrying a `(?)` marker — eight prose surfaces, of which six are `help_marker`. The finding's per-site line numbers (`:108` name, `:128` reads, `:172` format, `:191` size, `:205` sampling, `:215` edges, `:244` runs) are the pre-W-C locations of seven of the eight; all seven were found at their post-W-C lines. |
| The size row is a fixed 110px label column + 168px control (#7) | **Confirmed.** `pass_settings.py:29-30`: `_ROW_LABEL_W = 110.0`, `_CTRL_W = 168.0`. |
| `row_label` "places the control at `label_w + MD` unconditionally, so a label wider than 110px is overdrawn by its control" (#7) | **Confirmed.** `ui_primitives.py::row_label`'s body is `align_text_to_frame_padding()`, `small_caption(font, label)`, `same_line(label_w + SPACE.MD)`. `imgui.same_line` with an explicit x moves the cursor to that x whether or not the previous item ended past it, so a long label is drawn over. This is the mechanism behind #7's "overlaps with the scroll bar", one layer down from the scrollbar itself. |
| `row_label` is at `ui_primitives.py:1078` (#7) | **Confirmed.** `def row_label(` is `ui_primitives.py:1078` at `78bd1bf`. |
| The strip's gear tooltip is `Pass settings — what it reads, what it draws into` at `widgets/pass_list.py:98` (#10) | **Confirmed, exactly.** `pass_list.py:98`, inside `_settings_overlay`, **10 words**. |
| `_draw_auto_row` draws every engine uniform on ONE line with `same_line` between them (#32) | **Confirmed.** `tabs/document.py:98`'s `_draw_auto_row` loops with `if i > 0: imgui.same_line(spacing=float(SPACE.LG))`, then `uniform_name_label` at a per-name `calc_text_size` width, then `same_line(spacing=SPACE.MD)`, then the value. |
| It is placed on the SAME line as the sort combo, `tabs/document.py:157`, `same_line(spacing=XL)` (#32) | **Confirmed as the mechanism, re-pointed.** At `78bd1bf` the call is `tabs/document.py:226-228`: `if auto_hashes: imgui.same_line(spacing=float(SPACE.XL)); _draw_auto_row(...)`. The finding's `:157` predates W-A's rewrite of the canvas row above it. |
| 068 added `u_pass_iteration` / `u_pass_iterations` to the set, making five (#32) | **Confirmed.** `core.py:36-39`: `ENGINE_DRIVEN_UNIFORMS = frozenset({"u_time", "u_aspect", "u_resolution", "u_pass_iteration", "u_pass_iterations"} \| TABLE_UNIFORMS.keys())`. The glyph-table names are filtered out of the panel separately (`tabs/document.py`'s `if uniform.name in TABLE_UNIFORMS: continue`), so exactly five reach the block. |
| The engine uniforms are "still outside the sorted list and still read-only" and the hover bridge must survive (#32) | **Confirmed as the current structure and preserved by design.** `auto_hashes` and `active_uniform_hashes` are partitioned before `sort_uniform_hashes` is called, and only the latter reaches it; this wave does not touch the partition. The bridge is `uniform_name_label`'s `is_item_hovered` branch (`widgets/uniform.py:61-71`), which the block keeps calling. |
| "One shared row primitive per row-KIND" makes a hand-rolled engine row a known mistake | **Confirmed, and it is the same row kind.** `conventions.md`'s bullet names this exact history: "feature 008 special-cased engine uniforms into a dim caption row, which left them out of the code↔panel hover/jump bridge until it was generalized." Manual step 10 is the check. |
| `help_content.py` is where over-budget prose might also live (#5's "likely in Help panel content") | **Confirmed as a gate exemption, REFUTED as a reason to skip the file.** It contains **zero** calls to any gate call name, so the exemption is by construction. But round 1 checked the other direction and found the file has **no pass section at all** and zero matches for `smooth\|filter\|repeat\|wrap\|clamp\|tiling\|edges\|sampling`, so it does not carry the facts this wave's cuts remove. W-B therefore writes to it (a new `key="passes"` section plus the `u_pass_iteration` cross-reference) while measuring none of it. |
| D1's example strings are drawn from this codebase | **Confirmed on three of four.** § 2's table cites `no sampler2D uniforms` (this wave's replacement, so it is prescriptive rather than descriptive), `Pass settings` (this wave's replacement for #10), `Canvas: 1080x1080` (W-A's live notification, `tabs/document.py:47`, `f"Canvas: {w}x{h}"`, in budget as written) and `values above 1; default` (a `_FORMATS` tooltip that does not exist in that form — the shipped `f2` text is 16 words). The last is the skill showing the target shape, not quoting the code; this wave's `f2` replacement, `holds values above 1; the default`, is the closest the code gets. |
| W-A "left the gear's size help text untouched by design" for this wave | **Confirmed from the commit.** `git show 78bd1bf -- shaderbox/popups/pass_settings.py` is a two-line diff adding `begin_disabled` / `end_disabled` around the slider, and its message says so: "its help text is untouched, W-B owns that." |
| W-A's `label_row` / `small_caption` strings and the `Canvas: WxH` notification are in the census | **Confirmed, all three, all in budget.** `small_caption(app.font_12, "Document name")` (2) and `small_caption(app.font_12, "Canvas")` (1) at `tabs/document.py:129`/`:131`; `text_colored(COLOR.FG_DIM, "x")` (1) at `:155`, which is the 32nd `text_colored` site; and the notification at `:47`. W-A added no `label_row`. |
| The `ui_primitives` shared helpers forward a caller's text, so their callers are the measured sites | **Confirmed as the mechanism, REFUTED as a reason to exempt them, and its names were wrong.** Nine definition sites forward a parameter (`play_stop_toggle`, `clipped_caption`, `gauge_bar`, `draw_copyable_text`, `clickable_label`, `small_caption`, `label_row`, `setup_steps`, and `preview_cell` twice) and stay `_UNMEASURABLE`. But the callers were **not** measured while the gate read positional arguments only, so the stated reason was false when written — the four forwarder exemptions are deleted now that the callers are in scope. Two `ui_primitives` sites are measured and pass: `preview_cell`'s `"Delete"` (1) and `draw_link`'s `"Click to open + copy"` (5). The first draft named the last function `copy_path_label`, which does not exist. |
| `tests/` already contains an `ast.walk` test whose shape can be reused | **Confirmed, four of them, and one is the right shape.** `tests/test_worker_daemon_contract.py`, `test_script_api_doc.py`, `test_motion_verdict.py`, `test_document_dir_layout.py`. The first is the model: it walks `shaderbox/**/*.py`, parametrizes over the found sites, and carries an explicit "the enumeration actually finds the known workers" guard against a vacuous pass. This wave copies that three-part shape. |
| The gate's domain is the parent's six call names | **Refuted by round 1, and it is the wave's largest change.** Every `ui_primitives` helper taking authored copy takes it by KEYWORD, and a positional-only walk sees neither the helper (a parameter) nor the caller (a keyword slot). Four over-budget authored tooltips sat in the hole, three of them in `widgets/uniform.py`, which contributed zero rows to the first census. A probe `help_marker(text="…")` produced no row at all — neither a score nor an exemption, the silent-narrowing family. The call set is now the six plus seven helpers, read positionally or by keyword; 89 sites became 106. |
| D1 is enforceable by a word count alone | **Refuted.** D1 has a count half and a clause half, and § 2 states the clause half as a mechanical test. Nine shipped strings join a second clause with `;`, ` — ` or ` -- ` while passing any count; the first draft's own four replacements did too. § 3 now asserts both. |
| A conditional-bound string is unmeasurable | **Refuted.** Three live authored tooltips are `IfExp`s (the gear's size marker at 41 words on its output branch, `open_tooltip`, the play/stop tooltip), so "unmeasurable" meant "exempt" for exactly the strings this wave cuts. § 3 scores the worst branch, recursively. |
| The allowlist keys in the first draft name real functions | **Refuted, eight of thirteen.** `usage_bar` / `copyable_label` / `copy_path_label` / `_draw_copilot` / `emoji_picker.draw` / `_draw_grid` / `_draw_inline_input` / `code.py::draw` do not exist; the real names are `gauge_bar`, `draw_copyable_text`, `draw_link`, `_draw_copilot_config`, `_draw_body` (twice), `_draw_inline_new_input`, `draw_chrome`. Two real sites (`preview_cell`'s two `text_colored` calls) were missing. The aggregates matched anyway because the errors cancelled — which is why every table is now generated. |
| `_entry_row_label` has two callers | **Refuted, one.** The definition and a single call passing `"Script"`; the Shader row does not use the helper. Its literal reaches a non-gate function, so nothing measures it either way; at 1 word it is verified by eye. |
| The Help panel already explains texture filtering and wrap mode | **Refuted.** `help_sections()` returns five sections and none is about passes; the concepts return zero grep matches. The first draft's justification for deleting the `sampling` and `edges` markers asserted a coverage that does not exist. § 9a adds it. |
| The 20-character bound in `test_the_auto_name_column_fits_every_engine_uniform` is the width the column was chosen against | **Refuted, off by one, and the failure mode was wrong too.** 20 chars is 131.0px against 128.0px; the bound is 19 (`floor(128 / 6.5508)`). And an over-wide name is not overlapped — `uniform_name_label` draws through `clickable_label`, which calls `_ellipsize`, so it is silently truncated. A checker whose own bound admits its defect, named by the wrong mechanism. |
| `_AUTO_NAME_W` belongs beside W-A's module constants | **Refuted by W-A's fix-up (`3910900`)**, which deletes `_CANVAS_FIELD_W` / `_CANVAS_PRESETS_W` and the comment stating that rule, promoting both to `SIZE`. It becomes `SIZE.AUTO_NAME_W: int = 128`. |
| The engine block "sits after the sort row's `dummy`" today | **Refuted.** The call is currently *before* it, inside an `if auto_hashes:` guard the first draft's snippet silently dropped. § 10 now states the move exactly, guard included. |
| The parent's "`label_row` / `row_label`" gate inputs are readable at argument 0 | **Refuted, and it changed the collector.** Both take `font` first and `label` second (`ui_primitives.py:1078`, `:1088`), as does `small_caption` (`:693`). A collector reading argument 0 reports every one of the 21 sites as an unmeasurable `app.font_12` — a gate that is green because it measures nothing. This wave's first census did exactly that; `test_the_label_helpers_are_read_at_the_right_argument` exists so the next reorder cannot repeat it. |
| A `Name` argument can generally be resolved to a local assignment | **Refuted for today's tree: it resolves nothing in the gate's scope, and the one near-miss shows why the rule needs a third branch.** Every one of the 26 unmeasurable sites in scope is a parameter, a `Call`, a `Subscript`, an `IfExp`, an `Attribute` or a `BinOp` — none of which the rule reaches. The only `Name`-to-string binding anywhere in the census is `widgets/details.py`'s `caption_text(text)` (outside the gate's call set), and it is **augmented** on the next line (`text += f" ({...})"`), so a resolver that stopped at the bare `Assign` would report 4 words for a string that can render 7. § 4's algorithm therefore treats a `Name` that is the target of any `AugAssign` in the function as UNMEASURABLE rather than resolving it, and `test_a_name_rebound_by_augassign_is_not_resolved` pins that. The rule stays in the gate for the sites it will catch later; the allowlist carries the real weight today, and the anti-rot test is what keeps it honest. |
| The gear's size `help_marker` is a single string | **Refuted.** It is an `ast.IfExp` over two full variants (40 words for the output case, 32 otherwise), so the AST walk scores it UNMEASURABLE and it would slip the gate entirely. This wave replaces it with one 8-word `Constant` covering both cases, which is both the cut D1 asks for and the removal of a hole in the gate's coverage. |
| `_FORMATS` tooltips are reachable by the AST walk | **Refuted, and it added a test.** The call is `help_marker(_FORMATS[_FORMAT_CODES.index(target.dtype)][2])`, an `ast.Subscript`, so the walk sees nothing and the three tooltips (13, 16, 10 words) would sit permanently outside the gate. `test_the_format_tooltips_are_within_the_help_budget` asserts against the imported table directly. This is the general shape: a table of UI strings reached by subscript needs its own assertion, because no call-site walk can follow it. |
| The census can be taken once and trusted | **Refuted by the run, and it is why the reconciliation table exists.** The count changed under this session's own feet: `text_colored` moved 31 → 32 when W-A landed mid-wave. The gate is what makes this a non-issue going forward — it pins the allowlist and the budgets, not a site count, exactly as the parent specifies ("The test pins the allowlist, not a site count"). |
| `imgui.slider_float`'s format string may carry literal text around the conversion | **Confirmed from the existing code.** `pass_settings.py` already passes `"%.0f%%"`, and imgui applies the format with `printf` semantics, so `"%.0f%% · 512x512"` renders the percent and the literal suffix. No API change, no new flag. |
| A disabled slider still renders its format string | **Confirmed by inspection of the idiom, flagged for manual check.** `begin_disabled` pushes an alpha style and blocks interaction; it does not suppress the widget's text. `popups/settings.py` and `tabs/render.py` both draw labelled widgets inside disabled blocks and show their text. Manual step 4 is the falsifier, and it names the fallback (a dim caption after the widget) so the wave is not blocked if the rendering surprises. |
| `popups/help.py` and `popups/lib_picker/__init__.py` carry over-budget tooltips this wave should cut | **Corrected: they are over budget and are allowlisted rather than cut.** Both (15 and 11 words) explain a DISABLED state — why the Insert button cannot act right now — which a control's name structurally cannot carry. Cutting them would leave a greyed button with no explanation, which is a worse UX than a long tooltip. Their shared cause is the editor-focus model that W-E and W-F rework; the allowlist entries say so, and the anti-rot test will surface them when that work removes the disabled state. |

Corrected or refuted: **23** of 45 rows (7 corrections, 16 refutations). Ten of the refutations are
round 1's; nine of those changed the wave's design rather than its prose. The largest is the gate's
domain: the first draft measured a subset of the surface D1 governs and reported exact,
independently-reproducible aggregates about the wrong set — the reviewer's own collector matched
every number in it. Correct arithmetic over an incomplete domain is the failure mode this wave's
§ 1 claims four self-checks against, and it shipped in the draft that made the claim.

Three refutations are worth keeping visible because each is a checker admitting its own defect. The
argument-index one (draft 1's own, caught before review) would have measured no labels at all. The
20-character bound admitted a name that does not fit, and named the wrong mechanism for what
happens when one does. And the word-count-only scoring passed four semicolon-joined replacements
written to satisfy a rule whose text says "one clause".

## Open questions

Each carries a robust default, marked as such; none blocks implementation.

1. **Should the `set_tooltip` budget be 5 words or strictly "the control's name"?** Default, taken:
   **5 words**, asserted as a number. D1 states a shape, and a shape is not assertable; the census
   shows a wide gap between the longest passing tooltip (5 words, `Click to open + copy` and
   `Revert this turn's changes`) and the shortest failing one (10 words, the two this wave cuts), so
   the threshold sits in empty space and its exact value is not load-bearing. Revisit if a legitimate
   control name lands at 6 words, which would mean the name is doing a sentence's job and the fix is
   probably the control, not the bound.

2. **Should `caption_text` and `small_caption` join the gate's call set?** Default, taken: **no**,
   this wave keeps the parent's six. The census measured them anyway (20 and 5 sites) and found two
   over budget: `popups/examples.py`'s 13-word grid caption and `widgets/copilot_chat.py`'s 20-word
   revert explanation. Both are body copy in a modal rather than a label or a tooltip, which § 2's
   table has no row for, so adding the calls would mean inventing a budget the skill does not
   define. Revisit when § 2 grows a "modal body line" row; the collector already walks both, so the
   change would be one entry in the budget dict.

3. **Should the gear keep a width token at all, or auto-size in both axes?** Default, taken: **keep
   `PASS_SETTINGS_W = 440`.** With `always_auto_resize` the width would otherwise follow the widest
   row, so the popup would change width as the user picks a different pass in the Reads combos —
   the jitter § 3 spends a whole section forbidding. A fixed width with a following height is the
   shape that is stable in the axis the user reads across. Revisit only if 440 proves too narrow for
   a real pass name, which the rename field would show first.

4. **Where do the deleted help markers' facts go?** ~~Default: nowhere in the app; the Help panel
   already owns them.~~ **Closed by round 1, not an open question any more.** The premise was false:
   `help_sections()` has no pass section and zero matches for any of the concepts, so the deletions
   would have removed the app's only explanation of texture filtering and wrap mode. They go into a
   new `key="passes"` `HelpSection` **in this wave** (§ Design decisions 9a), carrying four facts,
   not the two the reviewer drafted — reading the surviving markers showed the runs marker's
   `u_pass_iteration` naming and the size marker's "quarter of the pixels" reasoning are also
   dropped by the cuts and carried nowhere else. § 2's "goes to the Help panel, where a reader chose
   to read" is honoured by putting it there, not by assuming it is already there.

5. **Should an allowlist value carry a revisit trigger as well as a reason?** Default, taken:
   **the reason only, with a trigger written into it as prose where one exists** (the two
   disabled-state tooltips name W-E/W-F, the telegram stat line names "a third stat line"). A
   structured trigger field would be a second thing to keep in sync with no test reading it, which
   is the shape `conventions.md`'s parallel-dict bullet warns about. Revisit if the lists grow past
   roughly a dozen deliberate entries; `_OVER_BUDGET` has 3 today and `_UNMEASURABLE` has 4
   deliberate ones beyond the mechanical forwarders.

6. **Should `separator_text`'s budget be 2 words or 3?** Default, taken: **2**, which cuts `Runs per
   frame` to `Runs` and is the only cut in this wave not traceable to a maintainer complaint. § 2's
   own examples (`Reads`, `Draws into`) are 1 and 2, and "per frame" restates the `runs` label
   directly below it. At 3 the heading survives and the flagged count is 11 rather than 12, matching
   the parent's number exactly — which is an argument for 3 that this wave deliberately does not
   take, because matching a prior count is not a reason to keep a word. Cheap to overturn: one
   number in the budget dict and one string.

---

## Review history

**Round 1, pre-implementation review** (`reviews/wave_b_pre.md`, one reviewer, correctness & design
plus verification & blast radius; opus, because the deliverable was judgement about a gate's domain
rather than retrieval). Verdicts: parent coverage **PASS**; D1 fidelity **PARTIAL**; gate domain
**FAIL**; census accuracy **PARTIAL**; test falsifiability **PARTIAL**.

The reviewer re-ran the census with an independent collector implementing §§ 2-4 verbatim and
reproduced every aggregate exactly — 89 sites, 63/26, 12 over budget, the same twelve sites. The
finding was not arithmetic, it was domain: the aggregates were right about a set that was the wrong
set. Twelve findings, **all accepted as written**.

| # | Finding | Resolution |
|---|---|---|
| 1 | The gate never reads a keyword argument, and the surviving over-budget authored tooltips live there. `widgets/uniform.py` contributed **zero** rows; a probe `help_marker(text=...)` produced no row at all — neither a score nor an exemption. | § Design decisions 2 rewritten: the argument is read positionally **or** from the keyword named for that parameter, and a call supplying it by neither is UNMEASURABLE, never skipped. The call set grows to seven `ui_primitives` helpers taking authored copy (`play_stop_toggle`, `clickable_label`, `clipped_caption`, `gauge_bar`, `draw_copyable_text`, `draw_link`, `preview_cell`'s `footer` / `sublines`), each at its budget. The four forwarder exemptions are **deleted** — with the callers measured they are unnecessary. Census 89 → 106; four new cuts, three in `widgets/uniform.py`. Two new tests. |
| 2 | Eight of thirteen named allowlist keys name functions that do not exist; two real sites missing; the aggregates matched only because the errors cancelled. | Every census table is now the collector's output pasted in, generated per file and sorted by line, never transcribed. § 5 says so explicitly and names the eight wrong keys as the reason. |
| 3 | `test_the_auto_name_column_fits_every_engine_uniform` passes a name that does not fit (20 chars = 131.0px against a 128.0px column), and its stated failure mode is wrong — `clickable_label` ellipsizes, so the real defect is silent truncation. | Bound changed to **19** (`floor(128 / 6.5508)`), the arithmetic carried at the assertion, and the falsifier restated as the ellipsized tail. The font metric is recorded as the reviewer's measurement, not re-derived here (no fontTools in this session). |
| 4 | Renaming `Runs per frame` → `Runs` breaks `help_content.py`'s live cross-reference `"(see Runs per frame)"`; the Files table does not list the file. | § Design decisions 9b added; `shaderbox/help_content.py` added to Files; the text becomes `"which run this is, 0-based (see Runs)"`. |
| 5 | Four of the shipped replacements are two clauses joined by a semicolon; D1's "one clause" half is unenforced by the test whose purpose is enforcing D1. | § 3 gains the clause check (`;`, ` — `, ` -- ` fail) asserted beside the count, plus `test_no_scored_string_joins_a_second_clause`. All four replacements rewritten to one clause and re-verified by running the check over them. Nine shipped strings fail it today. |
| 6 | The matching rule is never stated, so an alias and an `imgui.`-qualified call have undefined answers. | § 2 states it: `ast.Name.id` or `ast.Attribute.attr` in the set, module qualifier unchecked, so an aliased import is out of the walk **by construction** and the floor test is what catches an import style the walk cannot see. Exact-name matching also keeps `_entry_row_label` from being swept in. |
| 7 | `_entry_row_label` has one caller, not two, so its allowlist reason does not hold. | The row is regenerated from the collector with the real shape; its one caller passes `"Script"` positionally to a helper outside the gate's call set, so the string is verified by eye at 1 word. |
| 8 | § 2's heading promises a keyword surface the section never defines; a dangling "§ 12" reference. | The heading's promise is now real (finding 1's rule); the `_FORMATS` sentence points at `test_the_format_tooltips_are_within_the_help_budget` in § Tests. |
| 9 | `_AUTO_NAME_W`'s "W-A's precedent" argument is invalidated by W-A's fix-up, which promotes both cited constants to `SIZE`. | `SIZE.AUTO_NAME_W: int = 128` in `theme.py` beside `CANVAS_FIELD_W`; the stale citation dropped. The reviewer confirmed the fix-up moves no census row. |
| 10 | Open question 4's premise is false — `help_sections()` has no pass section and zero matches for the concepts, so the deletions remove the app's only explanation of them. | A new `HelpSection(key="passes")` ships **in this wave** (§ 9a). Reading the surviving markers found **four** facts dropped, not the two drafted: the `u_pass_iteration` naming and the "quarter of the pixels" reasoning go too. `tests/test_help_content.py` needs no change, and § 9a says why clause by clause. Open question 4 is closed rather than re-defaulted. |
| 11 | `always_auto_resize` versus a persisted `imgui.ini` rect is the one § 7 claim not traced to the stub, and manual step 2 would be run on a box already carrying the old rect. | Manual step 2 becomes a two-run check: delete `imgui.ini`, run it, then repeat with the ini in place; both must show the height following the content. |
| 12 | The block's stated placement contradicts the file's current order, and the snippet drops the `if auto_hashes:` guard. | § 10 states the move exactly: the guard **stays**, the `same_line(spacing=XL)` inside it goes, and the guarded call moves below the existing `imgui.dummy((0, SPACE.MD))`. |

**What the round changed about the wave, in one line each.** The gate went from measuring a subset
of its domain to measuring the domain (1, 6); the tables went from prose to output (2); D1 went from
half-enforced to enforced (5); three cuts that would have deleted facts with nowhere to land now
land them (10); and two tests that were green under their own named bug are now red under it (3, 5).

**The reviewer's own mutation table records three misses in the first specification**, all on one
axis: the gate matched a *shape of call*, and every shape not putting the string in a positional
slot under a bare or `imgui.`-qualified name was outside it. `g1` (keyword) had live instances;
`g2` (alias) and `g3` (qualifier) were latent and are now written down rather than left to the
implementer.

**Not verified by either party**, and carried as manual steps with named fallbacks: whether
`always_auto_resize` overrides an ini-restored rect (manual step 2, two runs), and whether
`begin_disabled` suppresses a slider's format string (manual step 4, fallback named).
