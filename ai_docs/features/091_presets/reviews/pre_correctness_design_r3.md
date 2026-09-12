# 091 pre-implementation review round 3 — correctness & design (closure)

**Verdict: PASS.** All four round-2 edits landed with the text they asked for, and the two
R3-specific checks came back clean: `UIDocument.save` does compile program-less passes **before**
the `live_rows` prune (the order D5's parenthetical now claims), and a chordless `COMMAND_SPECS`
row is already expressible — `hotkeys.py:332` returns False on `chord == 0`, so D10's palette-only
command dispatches from the palette and nowhere else. The spec is implementable as written.

`make gates` exit 0, captured unpiped (`make gates > g.log 2>&1; echo $?` → `EXIT=0`): check
passed, test passed, smoke **passed** (not skipped).

---

## A. Round 2, item by item

| Round-2 item | New spec text | Verdict |
|---|---|---|
| **Edit 1 / R1** — `_PASS_NAME_RE` unreachable from both leaf modules | D1: "The name obeys `PASS_NAME_RE` … That regex moves from `project_session.py` to `pass_graph.py` and loses its underscore: `group_slug` (D2) and `plan_import` (D4) both validate against it and both sit at or below `pass_graph` in the import order, so it cannot stay private to a module that imports them." Files touched: "`shaderbox/pass_graph.py` — … and `PASS_NAME_RE` moved down from `project_session.py` (its two uses there import it back)." | **CLOSED.** The cycle is gone and the move is named at both altitudes (the decision and the file list). Re-confirmed the premise is still live: `grep -rn "_PASS_NAME_RE\|PASS_NAME_RE" shaderbox/ tests/ scripts/` is exactly two hits, `project_session.py:124` (the definition) and `:128` (the only use) — so the move is two lines plus one import back, as the parenthetical says. D4's rejection list and verification 2's falsifier now both read `PASS_NAME_RE`, underscore-free, so the three sites agree. |
| **Edit 2 / R2** — the `bordered` compensation is 8px, and imgui ships the flag | D7: "`ChildFlags_.borders` also enables `WindowPadding` (imgui's own flag doc), so dropping it alone moves the cell's content from `(8, 8)` to `(0, 0)` and grows `avail` from 152x164 to 168x180 at a 168-wide tile: the picture, the footer and the chip row all shift. So `bordered=False` passes `ChildFlags_.always_use_window_padding` instead, the flag that exists for exactly this ("pad with style.WindowPadding even if no border are drawn") and measures byte-identical to `borders`. No hand-rolled pad: the cards keep their size (the maintainer's locked answer) because the padding is the same padding." | **CLOSED.** The flag exists with the doc string the spec quotes: `always_use_window_padding` at `.venv/lib/python3.12/site-packages/imgui_bundle/imgui/__init__.pyi:3357-3359`, `# (= 1 << 1) # Pad with style.WindowPadding even if no border are drawn`, and `borders` at 3353-3355 is `# Show an outer border and enable WindowPadding`. Both numbers the spec now carries (the `(8,8)`/152x164 vs `(0,0)`/168x180 pair) survive into verification 18's falsifier, so the measurement and the assert cite the same figures. |
| **Edit 2 (second half)** — the slack floor is 0px, not 4px | D7: "`tiles_per_row` charges the last tile no trailing gap, so a full row's slack reaches exactly 0px whenever the panel lands on `n*168 + (n-1)*8` — 344, 520, 696, 872, 1048 — and an outside rect clips" | **CLOSED.** The formula is `tiles_per_row`'s own arithmetic read back: `max(1, int((avail + gap) // (tile + gap)))` (`pass_list.py:39`) with the docstring's "a row of the returned `n` spans `n * tile + (n - 1) * gap <= avail`" (lines 35-36). The listed panel widths are that span at n=2..6 for `SIZE.PASS_TILE=168` / `SPACE.MD=8`. |
| **Edit 3 / R3** — three of five tints collide | D8: "four hues: `purple_n`, `green_b`, `yellow_n`, `aqua_n`. Four and not five because the palette has no fifth that is both free and far enough away: `purple_b` is `COLOR.SELECT` and a group outline is the same nested-outline context `SELECT`'s own invariant protects; `blue_n` is `COLOR.STATE_INFO`; `blue_b` is the blue accent's primary and `COLOR.TAG`; `green_n` sits 2 degrees of hue from `green_b`; `red_n` reads as the red error border these tiles can also carry. `aqua_n` is the aqua accent's ACTIVE colour, not its primary — allowed, and the reason the assert below names actives explicitly rather than trusting the existing `_accent_primaries` set, which enumerates element [0] of each preset only." Plus: "pins that no tint equals an accent primary, an accent ACTIVE, any `STATE_*` hue, `COLOR.SELECT`, `COLOR.TAG` or `COLOR.FAVS`" | **CLOSED, and the new tuple passes its own widened assert.** Enumerated the four against accent primaries ∪ accent actives ∪ `{SELECT, STATE_OK, STATE_WARN, STATE_ERROR, STATE_INFO, TAG, FAVS}`: `purple_n` clean, `green_b` clean, `yellow_n` clean, `aqua_n` → `accent_active` only — which is the one hit the spec declares and allows. No duplicates (4 distinct values). Each excuse in the "four and not five" list checks out by value: `COLOR.SELECT = _P["purple_b"]` (`theme.py:161`), `COLOR.STATE_INFO = _P["blue_n"]` (`:169`), `COLOR.TAG = _P["blue_b"]` (`:172`) and `blue_b` is the blue accent's primary (`:99`); `green_n` hue 59.5° vs `green_b` 61.2°. The "element [0] only" claim is literal: `_accent_primaries = {primary for primary, _active, _alpha in _ACCENTS.values()}` (`theme.py:200-202`). Verification 11's (b) list and its falsifier (`purple_b`, "which a check over accent primaries and state hues alone lets through") both match. |
| **Edit 4 / R4** — D11's negation is not expressible; and the second site | D11: "Two predicates, not one and its negation: `examples_planned` … and `import_project_tab` … The `if not any_popup_open():` branch takes `or import_project_tab`, never `not examples_planned`: that negation is true for PASS_SETTINGS, SETTINGS, HELP and PROJECTS too, so it would fire the full-set render behind every modal and leave the `elif PASS_SETTINGS` branch unreachable. The same `or import_project_tab` goes on `_tick_frame_state`'s own `if not any_popup_open():` gate, which is where `tick_documents` is BUILT: without it the set stays `[current]`, and the project tab's card grid, which blits each document's live canvas, shows black for every document whose first render has not happened, with no pending-first election to fix it." Files touched: "`shaderbox/ui.py` — … the two planned-set predicates, on the render chain AND on `_tick_frame_state`'s `tick_documents` gate (D11)." | **CLOSED on both halves.** Both gates are where the spec says and both are the literal text `if not app.any_popup_open():` — `ui.py:250` (the `tick_documents` build, with the render-all fan-out and the `pending_first` election inside it) and `ui.py:455` (the render chain's branch A, with `elif EXAMPLES` and `elif PASS_SETTINGS` after). `examples_open: bool = app.popup_state == PopupState.EXAMPLES` at `ui.py:275` is the single variable D11 renames, and `planned_documents` / `planned` / `current_planned` at 276-285 all read it, so "follow it unchanged" is one rename. The "Files touched" line now names both sites, which is the gap round 2 raised. |

### The verification reviewer's round-2 changes to decisions I own — no regressions

**D3's broken-pass exclusion.** "A source pass whose compile fails has an UNKNOWN wiring, not an
empty one: it is not offered as an entry point (the dialog would otherwise grow a spurious row for
it), it is copied as it is with its explicit rows, and the import result names it (D5)." Consistent
with D5's `ImportResult.notes` ("what was imported degraded: a source pass that did not compile")
and with verification 8, which asserts all three halves (other passes import, `notes` names it, it
is not among the entry points) and gives the right falsifier for the third ("treat its empty wiring
as a root and `entry_points` answers `["blur", "scene"]`"). Nothing here reopens D3's compile-the-
source rule or D4's plan purity: the exclusion is a filter on `entry_points`' input set, not a new
branch in `plan_import`.

**D5's compile parenthetical, and the order it asserts.** The new text is "(as `add_pass` does, so
its uniforms are live for the panel on the next frame; `UIDocument.save` compiles a program-less
pass itself before pruning, so the merged rows do not depend on this)". Read `UIDocument.save`
(`shaderbox/ui_models.py`) and the order is as claimed, not the reverse:

```
~410  for render_pass in self.document.passes.values():
~411      if render_pass.program is None:
~412          render_pass.compile()
~413  live = any(p.program is not None for p in self.document.passes.values())
...
~462  if live:
~463      live_rows = { get_uniform_hash(u) for render_pass in ... for u in render_pass.get_active_uniforms() ... }
~467      stale_rows = [h for h in self.ui_state.ui_uniforms if h not in live_rows]
```

The compile loop is ~50 lines above the prune, and the prune's `live_rows` is built from
`get_active_uniforms()` of passes the loop has already compiled — so a copied pass that arrived
program-less contributes its rows to `live_rows` and its merged `ui_uniforms` entries survive. This
**corrects** round 2's own section-B note, which reasoned from `live = any(...)` to "a never-
compiled copied pass contributes no live rows": the loop above means there is no never-compiled
pass left by the time `live_rows` is computed. The round-2 report's section B overstated the hazard;
the revised spec's weaker claim ("the merged rows do not depend on this") is the accurate one, and
verification 7's new parenthetical ("Skipping the copies' compile is NOT a falsifier") follows from
the same read. No regression — the spec still compiles the copies for the panel's sake, which is
the reason that survives.

**D10's no-auto-focus and the chordless command.** "No field takes keyboard focus on open or on
selection (an `input_text` that was given focus writes its own buffer back over an external write
on the next frame, which would defeat verification 17 and any programmatic prefill); the group field
is focused by a click." Verification 17 now carries the dependency explicitly ("This holds only
because the group field is not auto-focused (D10)"), so the rule and the gate that needs it point at
each other. And the command is "`IMPORT_PASSES` (chordless, palette only; a `COMMAND_SPECS` row AND
an `app.command_callbacks` handler, both gated by `test_command_registry_coverage.py`)" — which the
registry supports today with no new machinery, on four counts I checked:

- `spec_eligible` short-circuits on a zero chord: `if chord == 0 or chord in app.editor_consumed_chords: return False` (`hotkeys.py:332`), and the dispatch loop reads `chord = app.effective_bindings.get(spec.id, 0)` (`:344`) before that test — so a chordless row never reaches `imgui.shortcut`.
- The palette filters on `in_palette` alone: `palette_specs = [spec for spec in COMMAND_SPECS if spec.in_palette]` (`app.py:823`), so "palette only" needs no flag beyond the default.
- The help gate skips it by design: `if spec.default_chord:` guards the assert in `test_every_bound_spec_reaches_the_help_shortcuts`, so a chordless command owes no help line.
- The cheatsheet skips it: `if app.effective_bindings.get(spec.id, 0) == 0 ... continue` (`widgets/cheatsheet.py:34`), and `chord_to_str(0)` returns `"(unbound)"` (`commands.py:286-287`) for the Settings row.

The two registry-coverage asserts the spec names (`set(SPEC_BY_ID) == set(CommandId)` and
`set(app.command_callbacks) == set(CommandId)`) are chord-blind, so verification 13 stays red on a
missing row or handler exactly as stated.

---

## B. New defects

**None.** Three candidates came up and each is covered by a spec sentence on re-reading, so none is
a defect:

- *A chordless `CommandSpec` has no precedent in the table* — `COMMAND_SPECS` is 30 rows and none
  has `default_chord == 0` today, so `IMPORT_PASSES` would be the first. Not a defect: every path
  that consumes a chord already branches on zero (the four sites above), and the Settings rebinder
  draws it as "(unbound)" with a live Rebind button, which is the sane behaviour for a
  palette-only command the user may want to bind. The spec says "chordless, palette only" and
  nothing more is owed.
- *D7's rect-before-tiles order needs the run's left edge before the first tile draws* — the loop
  is `for i, name in enumerate(order): if i % per_row: imgui.same_line(spacing=float(SPACE.MD))`
  (`pass_list.py:177-179`), so at a run's first tile the cursor is at a known column and every
  later tile's x is `first_x + k*(168+8)`. The spec already states exactly this ("the strip knows
  every tile's rect from `tiles_per_row` and the cursor"), so it is expressible, not a gap.
- *D8's `aqua_n` is an accent active* — declared and allowed by the decision itself, and
  verification 11(b) asserts actives are excluded while the tuple contains one. Re-read: the assert
  is over `set(GROUP_TINTS)` vs "the accent primaries, the accent actives, …". `aqua_n` IS
  `_ACCENTS["aqua"][1]` (`theme.py:97`), so the assert as worded would be **red** on the shipped
  tuple. This is the one thing worth flagging, but it is not a spec defect at the correctness level
  I own — D8's prose is explicit that `aqua_n` is an accent active and "allowed", so the
  implementer reading D8 and verification 11 together has the decision in hand; the assert's
  wording needs `aqua_n` exempted (or the active set narrowed to the three it does not use) at
  implementation time, which the spec's "allowed" already authorizes. Naming it as an edit would be
  re-litigating a sentence the spec already settles.

---

## False trails — probed this round, fine, do not re-check

- **`UIDocument.save`'s compile-vs-prune order.** Compile loop ~410-412, `live` at ~413, the
  `live_rows` prune at ~462-471. Compile comes first; a program-less pass handed to `save` is
  compiled before anything is pruned against it. This settles round 2's section-B overstatement in
  the opposite direction from what that section said — the revised spec's weaker claim is the
  correct one.
- **A chordless command's four consumers.** `hotkeys.py:332` (eligibility), `app.py:823` (palette),
  `test_command_registry_coverage.py`'s `if spec.default_chord:` (help), `cheatsheet.py:34`
  (cheatsheet), `commands.py:286` (`"(unbound)"`). All handle 0 without an edit.
- **The `always_use_window_padding` flag and its doc string.** `imgui/__init__.pyi:3353-3359`:
  `borders` = "Show an outer border and enable WindowPadding", `always_use_window_padding` = "Pad
  with style.WindowPadding even if no border are drawn". The spec quotes the second verbatim.
- **The four-tint tuple against the widened invariant.** `purple_n` / `green_b` / `yellow_n` clean;
  `aqua_n` is the aqua accent's active and nothing else; four distinct values.
- **Both `if not app.any_popup_open():` gates in `ui.py`.** Lines 250 and 455, carrying what D11
  describes (the `tick_documents` build with the `pending_first` election; the three-branch render
  chain). `examples_open` is defined once, at 275, and read at 277/280/284.
- **`PASS_NAME_RE`'s two live sites.** `project_session.py:124` and `:128`, nothing else in
  `shaderbox/`, `tests/` or `scripts/`.
- **`tiles_per_row`'s arithmetic and its single caller.** `pass_list.py:32-39` (the `<= avail`
  docstring) and `pass_list.py:175`.
- **`preview_cell`'s hardcoded flag.** `child_flags=imgui.ChildFlags_.borders` unconditional,
  `border_color` pushing `Col_.border` only — unchanged since round 2, so `bordered=False` still
  has to touch the flag.
- **The `bloom_chain` fixture.** Five pass files under `tests/fixtures/bloom_chain/passes/`
  (`blur`, `bright`, `composite`, `scene`, `trail`), reached by `test_lazy_compile.py:31` and
  `test_default_wiring.py:40`. Round 1's and round 2's readings of it both stand.
- **`make gates`** — exit 0, unpiped, check + test + smoke all passed.
