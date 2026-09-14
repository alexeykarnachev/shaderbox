# 093 wave 1 — post-implementation review: architecture and conventions

Commit `f012074`. Read-only pass over `CLAUDE.md`, `conventions.md ## Code rules` + the graph /
`EditorTab` / theme / `ui_primitives` / `PopupState` design bullets, `.claude/skills/imgui-ui/SKILL.md`
§1-§4, §6, §8, §9, `dev_flow.md ### Module map` + `## Documentation discipline`, the contract
`01_spec.md` (T1-T6, S1-S15), every changed source, test and doc.

## Findings

**4 findings: 1 VIOLATION, 3 SMELL.** Nothing here blocks the landing.

### F1 — VIOLATION. `00_findings.md`'s "Landed in" column carries an unsubstituted template literal

The column the spec's Waves section requires filled in this commit reads, for all four rows:

```
| 1 | UX | ... | wave 1, commit <sha> |
| 2 | UX | ... | wave 1, commit <sha> |
| 3 | UX | ... | wave 1, commit <sha> |
| 5 | DEFECT? (mine) | ... | wave 1, commit <sha> |
```

`<sha>` is the roadmap's own row-shape pin's placeholder (`roadmap.md`: `Spec: <ai_docs/features/NNN_*.md
| commit <sha>>`), copied rather than substituted. The intended value is `f012074`. This is the exact
failure `dev_flow.md ### Roadmap rows index` guards: a pointer a cold reader cannot resolve. Finding 4
is correct — `— (delegated to its own feature)`.

Fix: replace the four `<sha>` with `f012074`.

### F2 — SMELL. Two comments narrate the change rather than the code as it is

`conventions.md ## Code rules` bans "the why-we-changed-it backstory". Two added comments carry it:

`shaderbox/widgets/pass_graph.py::_draw_node`:
```
    # The feedback mark replaces the self-loop wire the maintainer objected to (093 G8): flush
    # at the picture's top-right, or left of an `xN` badge when one is drawn this frame.
```
The first clause is the deleted code's story; the second is the live placement rule. Trimming to
`# The feedback mark sits flush at the picture's top-right, or left of an \`xN\` badge when one is
drawn this frame (093 G8).` loses nothing about the code as it is.

`shaderbox/theme.py`, over `GRAPH_DRAG_LOCK_PX`:
```
    # click. Not zoom-scaled; imgui's own 6px default is tuned for buttons.
```
is fine; the borderline sibling is `pass_graph.py`'s release-click block, which spends four lines on
what a press-time click *would* do. It names a live ordering constraint (the drag objects survive into
the release frame) and earns its length; not flagged.

Everything else in the comment audit is class (a): a live invariant, an ordering constraint, or a
≤1-line skill/spec pointer. Explicitly checked and kept: `graph_state.py`'s `WireId` and `press_blocked`
blocks (identity and latch timing), `editor_types.py`'s "`graph` tab has NO EditorSession" (the fact
every path-keyed pass-through depends on), `theme.py`'s `GRAPH_WIRE_HIT_FLOOR` pair comment (names the
6-under-7 relationship a transposition would break), `tabs/code.py`'s two focus-latch comments (the
second mirrors a pre-existing sibling verbatim in intent), every `# ---- section ----` banner (sanctioned
by the code rule). No class (c) restatement-of-the-obvious comment found in the diff.

### F3 — SMELL. ~30 new British spellings, under a gate whose word list does not reach them

`conventions.md`: "**American spelling wherever a reader sees words: `color`.** Comments, docstrings, UI
strings, prompts, examples, docs, test names. `tests/test_prose_spelling.py` is the gate."

`uv run pytest tests/test_prose_spelling.py -q` → `2 passed`. It passes because its roster is eleven
words:

```
_BRITISH_WORDS: tuple[str, ...] = (
    "colour", "quantise", "quantisation", "optimise", "optimisation",
    "initialise", "initialisation", "serialise", "serialisation",
    "behaviour", "favourite",
)
```

`centre` and `neighbour` are not among them. Counted over the diff's added lines: 30 `centre`,
3 `centres`, 1 `centred`, 2 `neighbour`, 1 `neighbours`. Per-file, before → after:

```
f012074^:shaderbox/widgets/pass_graph.py:2   →  f012074:shaderbox/widgets/pass_graph.py:19
                                             →  f012074:tests/test_graph_view.py:12
                                             →  f012074:tests/test_graph_state.py:1
                                             →  f012074:tests/test_graph_tab.py:1
```

The class predates this commit (six files carried it), so this is drift the wave amplified rather than
introduced — the shape `~/.claude/CLAUDE.md` names as "same prose, opposite outcomes": the rule is
written, the gate does not enforce it. Two honest responses: add `centre` / `neighbour` to
`_BRITISH_WORDS` and sweep, or narrow the written rule to the words the gate actually holds. Either
belongs in this wave per "ship the check that prevents the recurrence in the same commit as the sweep".

### F4 — SMELL. `_draw_wire` changes the stroke's WIDTH on hover, against §3's first sentence

`.claude/skills/imgui-ui/SKILL.md` §3: "A selection highlight must change *color*, never *size*."

```python
    thickness = max(1.0, SIZE.GRAPH_WIRE_W * zoom)
    if halo_col is not None:
        dl.channels_set_current(_CH_HALO)
        dl.add_bezier_cubic(*points, halo_col, max(1.0, SIZE.GRAPH_WIRE_W * 3.0 * zoom))
        thickness = max(1.0, SIZE.GRAPH_WIRE_W * 1.4 * zoom)
```

A hovered or selected wire's crisp stroke is 1.4× the resting one. **Not a defect**: the design record
prescribes it explicitly (`03_graph_design.md` G6's cue table, "Wire | stroke in `COLOR.GRAPH_HOVER` at
`GRAPH_WIRE_W * 1.4 * z`, over a halo"), and the record's own "No size change anywhere" paragraph scopes
itself to dots, nodes and cards. Filed as a SMELL only so the one place the widget departs from §3's
literal wording is on the record with its authority named. Everywhere §3 governs unambiguously, it holds:
the node halo is drawn INSET (`inset = SIZE.GRAPH_WIRE_W * z`, growing by `halo_w` per layer, so it
cannot bleed into the inter-card gap), the port dot's radius is untouched on hover
(`# A hovered dot swaps its COLOR; its radius never moves (093 G6)`), and `node_size(port_count, box)`
takes no hover parameter (pinned by `test_nothing_grows_on_hover`).

---

## 1. Code rules, enumerated over the diff — CLEAN

Greps over `git show f012074 -- shaderbox/ scripts/ tests/`:

- **Suppressions.** `grep '^+' | grep -E 'type: ?ignore|noqa|pyright: ?ignore'` → no output. Zero added.
- **`if TYPE_CHECKING`** → no output. **`@staticmethod` / `@classmethod`** → no output.
  **`from __future__ import annotations`** → no output.
- **`Any` on a real-typed parameter.** Every `Any` in the diff is `app: Any` / `view: Any` in a test
  module — the repo-wide fixture idiom (323 sites across 39 test files today; `test_graph_view.py`
  already carried 11 before this commit). No production signature takes `Any`.
- **Imports at module top.** `tabs/code.py` gains `from shaderbox.widgets import pass_graph` at module
  scope. No function-body import added anywhere in the diff.
- **Annotations.** `make check` (ruff + pyright, basic, 0 errors) passed inside the gate. Spot-read of
  every new public function in `graph_state.py` (`wire_points`, `bezier_point`, `wire_hit_threshold`,
  `wire_hit`, `wire_state`, `revalidated_wire`, `delete_allowed`) and `pass_graph.py` (`_draw_wire`,
  `_draw_wire_x`, `_draw_feedback_glyph`), `App.open_graph_for`, `App.choose_output`,
  `tabs/code.py::_draw_graph_tab`, `tabs/document.py::_entry_tab_active` / `_entry_row_label`: all params
  and returns annotated.
- **Color/size literals in `theme.py`.** The widget's only `push_style_color` is
  `imgui.push_style_color(imgui.Col_.child_bg, COLOR.BG_APP)` — a theme token on the child background,
  not the banned `Col_.button` hand-roll. Every new size and alpha is a `SIZE.GRAPH_*` / `COLOR.GRAPH_*`
  token (S12); `_MIN_DIRECT_DX` and `_BEZIER_BOW` were deleted in favour of `GRAPH_WIRE_MIN_OFF` and
  `GRAPH_WIRE_BOW`. The surviving module-local constants are shape fractions and channel indices
  (`_NONE_CORE`, `_PREV_INNER`, `_MEDIA_HALF`, `_FB_RING_R`, `_FB_GAP_ARC`, `_FB_ARC_SEGS`,
  `_CH_HALO`..`_CH_OVERLAY`) plus the pre-existing `_BADGE_PAD` / `_BADGE_H` / `_BADGE_INSET`.
- **Shared draw helpers through `ui_primitives`.** S11 landed: `_ellipsize` → `ellipsize`, all four
  in-module readers plus `popups/lib_picker/tree.py` and the two test modules follow the rename. No
  second copy of an ellipsis helper exists.
- **American spelling** — see F3.

## 2. The comment rule — see F2

Every added comment and docstring was read and classified. One class (b) clause (F2), zero class (c).

## 3. Module boundaries and layering — CLEAN

- **`graph_state.py` is imgui-free and `App`-free.** Its whole import block:
  `math`, `collections.abc`, `dataclasses`, `enum`, `shaderbox.document`, `shaderbox.pass_graph`,
  `shaderbox.theme`. No `imgui_bundle`, no `shaderbox.app`.
- **`pass_graph.py` makes no session write.** `grep -n 'app\.session\.\|session\.' shaderbox/widgets/pass_graph.py`
  → no output. `test_the_widget_makes_no_session_write_of_its_own` pins the four verb names by source grep
  and now holds `pass_list` to the same rule.
- **`tabs/code.py`'s graph branch delegates only.** `_draw_graph_tab` does the focus latch, `editor_focused`,
  `editor_errors = []`, the defocus consume, then `pass_graph.draw(app, tab.document_id)`. No geometry, no
  drawing of its own. The branch sits above the `ui_document is None` guard, as T2 requires.
- **`App` gained no drawing.** `git show f012074 -- shaderbox/app.py | grep '^+' | grep 'imgui\.'` → no output.
  The two added glfw cursors are `create_standard_cursor` in `__init__`, not draw code.
- **`theme.py` gained no logic beyond tokens and invariants.** The added lines are three `COLOR` fields,
  eleven `SIZE` fields, one `_GROUP_TINT_EXCLUSIONS` entry and two import-time `assert`s. No `def`, `if`,
  `for` or `while` added.
- **No leaf imports `app.py` newly.** The one new cross-module import is `tabs/code.py → widgets.pass_graph`,
  downward.
- **S9's helpers live where the spec says, once.** AST diff of function names:
  `graph_state ∩ pass_graph = []`, `graph_state ∩ ui_primitives = []`, `pass_graph ∩ ui_primitives = []`.
  `_entry_tab_active` lives in `tabs/document.py` and both entry rows call it (T5's "one home").

## 4. The imgui-ui skill's rules in the new draw code — CLEAN

- **§3 highlight = color.** See F4 for the one sanctioned exception; the node halo, port dot and card are
  all color-only, the halo inset.
- **§3 the ✕ is two `add_line`s.** `_draw_wire_x`: `add_circle_filled` + `add_circle` + two `add_line`,
  docstring `"the mark itself as two lines -- never a font glyph (/imgui-ui §3)"`. No glyph.
- **§4 `set_cursor_screen_pos`.** Four sites, each immediately followed by an `invisible_button` covering
  the moved-to position (`##graph_bg`, `##gnode_*`, `##gport_*`, `##gout_*`), all inside the canvas
  `begin_child`. Nothing can extend the parent.
- **§8 `push_font`.** Both sites pass `font.legacy_size * z`:
  `imgui.push_font(font, max(4.0, font.legacy_size * z))` and the `app.font_12` sibling. The three
  `get_font_size()` reads in the repo are vertical-centring measurements, never a `push_font` argument.
- **§8 one `want_cursor` owner.** Three `app.want_cursor = ...` requests in `_draw_canvas`;
  `grep 'glfw.set_cursor' shaderbox/widgets/pass_graph.py` → no output, pinned by
  `test_the_widget_never_pokes_glfw_itself`.
- **§1 button tiers.** `standard_button("open##entry_graph")` with a 4-word tooltip `"Open the pass graph"`;
  the canvas's own two are `primary_button("Create")` / `standard_button("Cancel")`. No raw `imgui.button`
  in the widget.
- **§8 `begin_popup_context_item(None)`** on the shared child: `if imgui.begin_popup_context_item(None):`,
  the one call in the module.
- **The allow-overlap chain's order is intact and documented.** Background declares
  `set_next_item_allow_overlap()`, then each node declares it, then the ports declare nothing and are
  submitted last. The module docstring states the rule and the reason ("the flag makes an item
  overlappable by a LATER one and costs it its own hover, so on the last rung it only forfeits the drop
  target"), together with the hover model, the wire pass, the hand-tested ✕, the five channels and the
  `begin_popup_context_item(None)` fact.

## 5. Tests as artifacts — CLEAN

- **Falsifiers.** Every new test in `test_graph_tab.py` (11), `test_graph_state.py` (18 total) and
  `test_graph_view.py` (23 total) carries a one-line reason naming what it guards — either an explicit
  `Falsifier:` / `Break to try:` clause or a sentence naming the clause under test
  (`"S5's \`not blocked\` clause: a turn freezes every canvas write, and a key is no exception."`).
- **No number that says nothing.** The only bare counts in the new tests are
  `source.count("channels_split(5)") == 1`, `source.count("channels_merge()") == 1` (G12's one-split-one-merge
  claim) and `[t.path for t in app.editor_tabs].count(graph_path) == 1` (T1's dedupe claim). Each decides
  something.
- **`xdist_group` on both frame-driving graph modules.**
  `tests/test_graph_view.py:28: pytestmark = pytest.mark.xdist_group("gl_frames_graph_view")`,
  `tests/test_graph_tab.py:32: pytestmark = pytest.mark.xdist_group("gl_frames_graph_tab")`.
- **The three new `_UNMEASURABLE` rows are true of the code.** They appeared because the `_ellipsize` →
  `ellipsize` rename made those sites visible to the reflection-derived domain. Verified:
  `anchored_note` → `value = ellipsize(value, wrap)` on a caller-supplied value string;
  `preview_cell` → `label: str = ellipsize(footer, avail.x)` on the caller's footer;
  `pass_graph::_draw_node` → `ellipsize(node.name, ...)` and `ellipsize(node.labels[slot], label_budget)`,
  i.e. a pass's own name and its shader's sampler names. Each reason matches.
- **The deleted `tests/test_ui_regions.py` has no surviving importers.**
  `grep -rn 'test_ui_regions' pyproject.toml Makefile scripts/ tests/ shaderbox/` → no output.
  `PassesView` / `PASSES_VIEW_LABELS` / `passes_view` survive only in 092's and 093's feature records,
  which is correct (a spec narrates what shipped).

## 6. Docs discipline — VIOLATION (F1), otherwise CLEAN

- **The roadmap row** is unchanged, one markdown line, status `in progress`, one descriptive sentence,
  `Spec:` pointer — matches the shape pin.
- **The banner** is 196 words (under 200), rewritten in full (no "carry-over", no archived block),
  date-stamped `<!-- As of 2026-09-14, 093 wave 1 is implemented and awaits its post-implementation review. -->`.
- **No line numbers or file lengths** in any touched doc:
  `grep -nE '\.(py|md|glsl|json):[0-9]+'` over `conventions.md`, `dev_flow.md`, `roadmap.md`,
  `00_findings.md`, `01_spec.md`, `092/03_spec.md` → no output. The two `N L`-style hits in the repo are
  pre-existing roadmap rows and `conventions.md`'s own statement of the rule.
- **No TODO / deferred / later marker in the diff.** `grep -iE '\bTODO\b|\bFIXME\b|\bdeferred\b|\bfor now\b|\bXXX\b'`
  over added lines → no output. The four `later` hits are the ordinary word describing imgui's submission
  order ("an earlier item ... beats a later one"), not a deferral.
- **`conventions.md`'s rewritten graph bullet describes what the thing IS.** It opens
  "**The graph view is a second picture of the same wiring, it lives in the editor pane, and it stores one
  thing (features 092, 093).**" and states the wire's construction, the hover model, the click rule and the
  port/edge two-sources rule in the present tense. The one backward glance is parenthesised and one line
  ("092 put it behind a `strip | graph` toggle in half the Document tab; 093 reversed that, and `PassesView`
  went with it."), which is the sanctioned form for a reversal a reader may otherwise re-derive. The
  "revisit the canvas's home" clause is resolved, and the editor-tab bullet's trigger gained the
  non-editable-kind clause the spec required.
- **`092/03_spec.md`'s reversal pointers.** D2 and D10 each carry an inline pointer at the overruled
  passage naming the reverser (`**D2. Where it lives.** REVERSED by 093 T1-T5: ... What follows describes
  the shipped 092 shape.` / `(REVERSED by 093 S15 for the CLICK only: ...)`), plus a Review-history entry
  `**Reversed by 093 wave 1 (2026-09-14): D2 and D10's click half.**`. 083 D5's pointer for 085 uses a
  blockquote (`> **REVERSED for DENY by feature 085** (...)`) rather than inline text — same substance
  (overruled passage marked in place + reverser named + what still stands), different typography. Not
  flagged; if the maintainer wants the shape literally uniform, the fix is one blockquote.
- **`dev_flow.md`'s module-map entries describe the modules as they now are.** `pass_list.py`
  ("the caption row and the `add pass` / `import...` row are the Document tab's"), `pass_graph.py`
  ("a second picture of the same passes in its own editor tab (`draw(app, document_id)`, filling whatever
  child it is handed)", five channels, the wire's distance pass, the hand-tested badge), `graph_state.py`
  (the pure wire geometry enumerated), `tabs/` (`code.py` as the pane's whole dispatch with the graph
  branch; `document.py`'s two entry-point rows). All match the code.
- **`00_findings.md`'s "Landed in"** — F1.

## 7. The gate — CLEAN

```
TMPDIR=.../scratchpad make gates > .../gates_review.log 2>&1; echo EXIT=$?
EXIT=0
```

Log tail:
```
== gates: check passed ==
== gates: test passed ==
== gates: smoke passed ==
== gates: GREEN -- check passed, test passed, smoke passed ==
```

Smoke **passed**, not skipped. Run once, at the end of the review.

---

## What I skipped

- **Spec fidelity row by row (T1-T6, S1-S15 against the Verification table).** That is the parallel
  spec-fidelity reviewer's assignment; I checked the T/S items only where an architecture or convention
  claim rested on them (T1's session-less pass-throughs, T2's branch position, S6's two orderings,
  S9's helper homes, S11, S12, S13, S15).
- **Rendered appearance.** No WM on this box; the halo at 0.25 / 2.5 zoom, the feedback glyph's shape,
  the 136 card under his own names and the 4px lock under his hand are the maintainer's eyes, as the
  spec says.
- **Correctness of the wire geometry and the frame-timing behaviour.** The parallel correctness
  reviewer's; I confirmed the gates the commit body claims exist as named tests and that the suite is
  green, not that each break was actually tried.
- **`conventions.md` bullets outside the greps named in the brief** (`graph`, `EditorTab`, `theme`,
  `ui_primitives`, `PopupState`) — read the matches, not the whole 1548-line file.

## False trails

- **`Any` in the new tests.** Looked like a bare-`Any` violation until measured: `app: Any` is the
  repo-wide fixture idiom (323 occurrences, 39 files; 11 in `test_graph_view.py` before this commit).
  The alternative is importing `App` into every test module, which the conftest fixture's shape does not
  offer. Not a finding.
- **`push_style_color` in `pass_graph.py`.** The code rule names `push_style_color(Col_.button, …)`;
  the one call is `Col_.child_bg` with `COLOR.BG_APP`. Not a hand-rolled button.
- **`get_font_size()` in `pass_graph.py`.** Two calls, both vertical-centring arithmetic
  (`(_BADGE_H * z - imgui.get_font_size()) / 2`), neither a `push_font` argument. §8's ban is specific to
  `push_font`.
- **`PassesView` surviving in the tree.** The grep hits are all inside `ai_docs/features/09{1,2,3}/`
  records, where a spec narrating what shipped is supposed to name it. The live code and the test roster
  are clean.
- **Tests apparently missing a falsifier.** A regex over the 8 lines above each `def test_` flagged
  eleven; reading them, nine are pre-existing 092 tests and the two new ones
  (`test_the_loop_the_bus_and_the_module_constants_are_gone`,
  `test_delete_is_refused_during_a_copilot_turn`) each carry a reason sentence my pattern did not match.
  Not a finding.
- **`test_prose_spelling.py` passing.** A green gate here says nothing about the rule it names — its word
  list is eleven entries and does not include the two words the diff actually adds. F3 is the finding the
  green gate hid.

---

VERDICT: PARTIAL
