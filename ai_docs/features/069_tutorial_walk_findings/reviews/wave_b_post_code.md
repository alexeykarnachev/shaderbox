# Wave B post-implementation review: code correctness

Commit under review: `ccd446b` ("069 W-B: cut UI prose to budget and gate it").
Role: bugs, imgui frame-order hazards, the gate as a checker, error handling.

## Verdict

| Area | Verdict |
|---|---|
| Gate domain | **FAIL** — a new copy-bearing `ui_primitives` helper escapes the walk entirely, and nothing asserts `_SCORED` covers the helper set. Live over-budget copy is already sitting in that hole. |
| Gear layout | **PARTIAL** — `always_auto_resize` overrides the seeded width, so `PASS_SETTINGS_W = 440` is dead (measured 323/294). Height clamps at the viewport past 20 Reads rows, and `no_scrollbar` removes the only visual cue that content continues. |
| Size row | **PASS** — `WxH` is composed in the control per D1, `·` (U+00B7) renders in the loaded face, and the disabled slider still shows the full format string. |
| Engine block | **PARTIAL** — layout is correct (no assert, columns align at exactly 136.0), but the `AUTO_NAME_W` metric is off by one in the unsafe direction: the test's budget says 19 characters, `_ellipsize` truncates at 19. |
| Cuts | **PARTIAL** — the `format` control's choosing criterion was deleted and is documented nowhere. Everything else survived or moved to Help. |
| Help | **PASS** — the new section renders through the existing loop, `test_help_content.py` still pins engine-uniform coverage, and `(see Runs)` matches the renamed separator. |
| Conventions | **FAIL** — the commit ships a RED `make gates`. `tests/test_ui_prose_budget.py` fails ruff SIM102 and `ruff format`. |

**Overall: FAIL.** F1 alone is disqualifying — the wave whose thesis is "a budget with no
check is a wish" committed a checker that does not pass the repo's own gate.

## The mutation table

Every mutation was a temporary edit to a real source file, run, then restored and the
restore verified with `git diff --quiet` before the next one.

| # | Mutation | Outcome |
|---|---|---|
| a | Over-budget `Constant` at a fresh `help_marker` site in `pass_settings.py` | **RED** — `test_every_measured_site_is_within_budget[...::_draw_repeat:219 help_marker.text]` |
| b | `help_marker(text="...12 words...")` (keyword-supplied) | **RED** — same test, `_draw_repeat:218` |
| c | `label_row(app.font_12, f"size {w}", ...)` (FormattedValue in a label) | **RED** — `test_no_label_carries_an_interpolation[...::_draw_target:167 label_row.label]` |
| d | Semicolon-joined 6-word `help_marker` (in budget on count) | **RED** — `test_no_scored_string_joins_a_second_clause`, count test green. The clause half earns its place. |
| e | `_UNMEASURABLE` entry naming `shaderbox/tabs/nonexistent.py::no_such_function` | **RED** — `test_every_unmeasurable_entry_still_names_a_real_site` |
| f | Over-budget string through `play_stop_toggle(..., tooltip=...)` in `tabs/document.py` | **RED** — `test_every_measured_site_is_within_budget[...::_draw_entry_points:328 play_stop_toggle.tooltip]` |
| g | `imgui.set_tooltip` with an 8-word string inside a lambda | **RED** — `_enclosing_function` walks past the lambda to the named function, so `_settings_overlay:98` is reported |
| h | New `ui_primitives` helper with `tooltip=`/`label=`, NOT in `_SCORED` | **GREEN — a miss.** See F1. |

Mutation (h) was run in two variants:

- **h1** — the helper forwards to `imgui.set_tooltip` internally. Red, but on the *wrong
  thing*: `test_every_unmeasurable_site_is_listed[shaderbox/ui_primitives.py::badge_chip:202
  set_tooltip.text]`, i.e. the helper's own forwarding call. The caller's 9-word `tooltip=`
  and 5-word `label=` were never collected — dumping `_SITES` for `pass_list.py` under the
  mutation shows three rows, none of them `badge_chip`, and
  `[s for s in _SITES if s.call=='badge_chip'] == []`. The red is incidental: the fix an
  author would apply is to add `badge_chip` to `_UNMEASURABLE`, which closes the red and
  leaves the caller's copy permanently unmeasured.
- **h2** — the helper draws its own tooltip via `begin_tooltip`/`text_unformatted`/`end_tooltip`
  instead of a scored call. **222 passed. Fully green.** Nothing anywhere notices.

## Findings

### F1 (blocker). The commit ships a red `make gates`; the new gate file fails ruff twice

**Claim.** `tests/test_ui_prose_budget.py` as committed violates ruff SIM102 and is not
`ruff format`-clean, so `make gates` exits 2 at the first step and never reaches test or smoke.

**Evidence.** Against the committed blob, extracted with
`git show ccd446b:tests/test_ui_prose_budget.py`, on a clean tree:

```
== gates: check ==
ruff (legacy alias)......................................................Failed
SIM102 Use a single `if` statement instead of nested `if` statements
   --> tests/test_ui_prose_budget.py:342:21
...
== gates: FAILED at check (exit 2); test and smoke not run ==
== gates: RED (exit 2) ==
EXIT=2
```

`ruff format --check` on the same blob reports `1 file would be reformatted` (the
`_UNMEASURABLE` dict entries exceed the 88-char line length and want the exploded tuple form).
These are two independent failures. `pyproject.toml` puts `tests/` fully in scope: `extend-exclude`
lists only `ai_docs/`, and `per-file-ignores` covers only `__init__.py` and one script.

The offending construct:

```python
                    if arg is None and (index is None or len(node.args) <= index):
                        # A call that omits an optional parameter carries no string.
                        if not _supplies(node, parameter, index):
                            continue
```

Collapsing the two `if`s into one `and` chain makes ruff pass; `ruff format` then rewrites the
long dict entries. With both applied I re-ran the full gate: **exit 0, check + test + smoke all
passed**, and the gate's own 222 tests stay green. So SIM102 plus the format drift is the whole
blocker; nothing else in the wave is red.

This is the finding the commit message's own thesis makes most costly. The message says "a
budget with no check is a wish, and this repo already has the receipts for prose rules that only
stated themselves", and its Falsified paragraph lists sixteen mutation runs. Those were run
against `pytest`, never against `make gates` — the project rule is to judge by the gate's exit
code captured unpiped, and that step was skipped.

**Fix.** Collapse the nested `if` at `_collect`'s keyword-fallback branch into one condition,
run `ruff format` over the file, and re-run `make gates > /tmp/g.log 2>&1; echo $?` confirming
exit 0 before the wave is called done.

### F2 (major). The gate narrows its own domain: a new copy-bearing `ui_primitives` helper is invisible, and live over-budget strings are already in the hole

**Claim.** `_SCORED` is a hand-maintained list of thirteen call names. Nothing asserts it covers
the set of `ui_primitives` functions that take authored copy, so a helper added outside the table
takes its callers' strings out of the gate's domain silently. This is the exact
"checker that quietly narrows its own domain" family, and it is not hypothetical.

**Evidence.** Mutation h2 above: a `badge_chip(id_, *, tooltip="", label="")` helper that renders
its own tooltip, plus a caller passing a 9-word `tooltip=` and a 5-word `label=`, leaves the
suite at **222 passed**.

Enumerating the real signatures shows the hole is already occupied. Reflecting over
`shaderbox.ui_primitives` for functions with a `tooltip`/`label`/`text`/`hint`/`footer`/`message`
parameter gives 34 functions, of which **11 are in `_SCORED` and 23 are not**. Among the unscored:
`caption_text(text)`, `wrapped_caption(text)`, `small_caption(text)`, `unconnected_gate(hint)`,
`connection_status(message)`, `labeled_text_input(label)`, `labeled_combo(label)`,
`labeled_drag_float(label)`, `labeled_multiline_input(label)`, and every button tier
(`primary_button`/`button`/`ghost_button`/`toggle_button`/`pill_button`/`chip_button`).

Measured live strings passing through unscored helpers today:

- `shaderbox/popups/examples.py:117` — `caption_text("Pick an example to read about it; open a copy to dig in.")`.
  **13 words and semicolon-joined**: it fails both halves of the gate's own scoring, and is
  invisible to it.
- `shaderbox/tabs/render.py:74` — `caption_text(f"Select an output file to render the {media_type}")`, 8 words.
- `shaderbox/exporters/youtube.py:389` — `unconnected_gate("Not connected to YouTube.", "Connect your channel in Settings to upload.", "Set up credentials", ...)`.
- `shaderbox/exporters/youtube.py:431` — `labeled_text_input("Tags (comma-separated)", ...)`, a 2-word label at the boundary.

`small_caption` is the sharpest illustration: the gate imports it and asserts its argument
index in `test_the_label_helpers_are_read_at_the_right_argument`, and lists it in
`_UNMEASURABLE` as "forwards the caller's text" — but it is absent from `_SCORED`, so no
caller of it is ever scored. The signature test creates the appearance of coverage the walk
does not deliver.

The wave's own commit message diagnoses this family correctly for the keyword half ("a
positional-only walk saw neither the helper nor the caller... Four over-budget authored
tooltips sat in that hole") and then stops one level short: it fixed the *argument-reading*
narrowing and left the *call-table* narrowing, which is the same defect one rung up.

**Fix.** Add a test that enumerates `ui_primitives`' public functions by reflection, selects
those whose signature carries a copy-bearing parameter name, and asserts each such
`(function, parameter)` pair is either in `_SCORED` or in a written, reasoned exemption list —
so a new helper defaults INTO the gate rather than out of it, the way
`test_worker_daemon_contract.py` already does for worker spawn sites. Then score the helpers
the enumeration turns up and cut the strings it exposes, starting with `examples.py:117`.

### F3 (moderate). `always_auto_resize` discards the seeded width, so `PASS_SETTINGS_W` no longer controls anything

**Claim.** The gear popup's width is no longer 440. `WindowFlags_.always_auto_resize` makes imgui
size the window to its content every frame and ignore `set_next_window_size`, so
`SIZE.PASS_SETTINGS_W` is passed in and discarded. The spec's jitter argument ("the width stays
fixed at `SIZE.PASS_SETTINGS_W = 440` and only the height follows content") does not hold.

**Evidence.** Headless probe replicating `modal_window`'s exact call sequence
(`set_next_window_size((440, 0), first_use_ever)` then
`begin_popup_modal(flags=always_auto_resize|no_scrollbar)`) over a body shaped like
`_draw_body`, reading `imgui.get_window_size()` on a settled frame:

```
 samplers    width   height    pos.y  fits-768?
        0    323.0    320.0    224.0  yes
        4    323.0    412.0    178.0  yes
       16    323.0    688.0     40.0  yes
       20    323.0    762.0      3.0  yes
       40    323.0    762.0      3.0  yes
```

Width is 323 at every content size, never 440. A second probe with a much longer row label
(`u_a_very_long_sampler_name` vs `u_tex`) also returned 294 both times — because the rows use
`same_line(absolute x)`, the label length does not extend content width, so the width happens
to be stable in practice. The jitter outcome the spec wanted holds, but by accident of the
row layout rather than by the token, and the token is now inert.

**Fix.** Either drop `always_auto_resize` and keep `set_next_window_size` with an explicit
height (contradicting the wave's intent), or keep auto-resize and delete `SIZE.PASS_SETTINGS_W`
along with `PASS_SETTINGS_H`, since neither is read any more. The half-state — passing a width
token the flag ignores — is the misleading one, because the next reader will change 440 and see
nothing move.

### F4 (moderate). Past 20 Reads rows the gear's content leaves the screen with no scrollbar to say so

**Claim.** With `always_auto_resize | no_scrollbar`, imgui clamps the window at the viewport and
suppresses the scrollbar. Content past the clamp is still wheel-scrollable but has no visual cue,
so on a 768px display a pass with 20 or more sampler uniforms shows a Close button the user
cannot see and no indication the panel continues.

**Evidence.** Probe over the real body shape (name row, N Reads rows, four Draws-into rows, Runs
row, Close), on a 1280x768 viewport, reading `get_window_size().y`,
`get_item_rect_max().y` for the Close button, and `get_scroll_max_y()`:

```
 samplers   win_h  close_bottom  scroll_max  Close reachable on 768px?
       18   734.0         743.0         0.0  yes
       19   757.0         754.0         0.0  yes
       20   762.0         775.0        18.0  NO
       24   762.0         867.0       110.0  NO
       40   762.0        1429.0       672.0  NO
```

The threshold is exactly 20 Reads rows. Severity is bounded by imgui's own documented
semantics — the `imgui.pyi` stub for `no_scrollbar` reads "Disable scrollbars (window can still
scroll with mouse or programmatically)" — so the content is reachable by wheel, just with no
affordance saying it exists. Sampler count is uncapped: `grep` finds no limit on sampler2D
uniforms per pass, so this is user-reachable, if unusual.

**Fix.** Drop `no_scrollbar` and keep `always_auto_resize`. Auto-resize already gives the
"sizes to its content" behaviour the finding asked for; the scrollbar then appears only in the
clamped case, which is exactly when the user needs to know there is more. Removing the bar
does not prevent the overflow, it only hides it.

### F5 (moderate). The `AUTO_NAME_W` character budget is off by one in the unsafe direction

**Claim.** `test_the_auto_name_column_fits_every_engine_uniform` computes its budget from an
assumed advance of `12.0 * 1118.0 / 2048.0 = 6.5508` px per character, giving
`floor(128 / 6.5508) = 19`. The rasterized 12px face advances **7.0 px**, so only 18 characters
fit. A 19-character engine uniform passes the test and is silently ellipsized — the precise
failure the test's own comment says it prevents.

**Evidence.** Measured in a headless frame with the real font
(`shaderbox/resources/fonts/Anonymous_Pro/AnonymousPro-Regular.ttf` at `size_pixels=12`):

```
advance 'A' = 7.0           test assumes char_w = 6.55078125
'u_pass_iterations' width = 119.0  vs SIZE.AUTO_NAME_W=128
real fit = 18 chars         test's floor(128/6.5508) = 19
```

Driving `ui_primitives._ellipsize` directly at `width=128.0`:

```
'u_pass_iterationss'  len=18 width=126.0 -> 'u_pass_iterationss'   fits
'u_pass_iterations_x' len=19 width=133.0 -> 'u_pass_iteratio...'   TRUNCATED
```

Adding `"u_pass_iterations_x"` to `ENGINE_DRIVEN_UNIFORMS` and running the test:
`1 passed`. Green on a name that visibly truncates.

Today's longest name is `u_pass_iterations` at 17 characters, so nothing is broken now; the
margin is one character narrower than the test reports, and the test would not catch the name
that crosses it.

**Fix.** Replace the hard-coded em-ratio with a measured value, or set the budget to 18. The
robust form is to assert against `_ellipsize` itself in a headless frame
(`assert _ellipsize(name, SIZE.AUTO_NAME_W) == name`), which cannot drift from the renderer
because it *is* the renderer, rather than re-deriving a font metric the test hard-codes.

### F6 (minor). The `format` control's choosing criterion was deleted and is documented nowhere

**Claim.** `_FORMATS`' old tooltips carried *when to pick each format*; the new ones carry only
what each is. The dropped criterion did not move to the Help panel, so the one control whose
correct answer a user cannot guess now has no explanation anywhere.

**Evidence.** Before and after:

```
f2 old: "Holds values above 1 (bright highlights, accumulated light). The default, and what bloom and feedback need."
f2 new: "holds values above 1, the default"
f1 old: "Smallest. Values clamp to 0-1 — the right choice for a final image."
f1 new: "clamps to 0-1, the smallest"
```

The new Help section (`key="pass_settings"`) documents `smooth`, `repeat`, `Runs` and `size`.
`grep -in "bit|format|precision|bloom" shaderbox/help_content.py` returns nothing — `format` is
absent from it. The commit message says the section carries "the four facts the deleted markers
held", and that is accurate for the four it names; `format` was cut from `_FORMATS` in the same
commit and was not among them.

Note the deleted Reads tooltip's actionable half ("To read something new, declare another
sampler2D in this pass's shader") is partly covered: the Help panel's Uniforms section already
says "an image slot for a `sampler2D`". Weaker, but present. The `format` criterion has no such
cover.

**Fix.** Add one line to the `pass_settings` Help section body naming when each format is
wanted — bloom and feedback need 16-bit float, a final image is fine at 8-bit — the same shape
as the four facts already there.

## False trails

- **`SIZE.PASS_SETTINGS_H` left behind a reader.** No. `grep -rn PASS_SETTINGS_H` over the repo returns only spec and review prose; the token is gone from `theme.py` and `pass_settings.py` cleanly.
- **The `·` (U+00B7) glyph does not render in the loaded face.** It renders. Rasterized to a framebuffer and counted, the separator cell has 2 ink pixels for `·` against 0 for spaces and 31 for an out-of-range control glyph (U+2318, the fallback box) — a 12px middle dot is a 1-2px mark. `is_glyph_in_font` is not a usable check here: it returns True for U+2318 too, so it queries the TTF cmap, not the built atlas.
- **The disabled output-pass slider hides its format string.** It does not; `begin_disabled` dims the widget and keeps `%.0f%% · WxH` rendering, so the output pass still shows its dims.
- **`_draw_auto_block` trips the SetCursorPos assert (skill § 4).** It does not. It uses `same_line(offset_from_start_x)`, which is normal flow, not absolute positioning. Driven three frames headless: no assert, and the value column lands at exactly 136.0 (`AUTO_NAME_W + SPACE.MD`) for all five names.
- **`same_line(float(SIZE.AUTO_NAME_W) + float(SPACE.MD))` passes the offset as `spacing`.** No. The signature is `same_line(offset_from_start_x=0.0, spacing=-1.0)`; the positional argument is the offset, which is what was intended. The old code's `same_line(spacing=...)` was the keyword form for a different purpose.
- **The `if auto_hashes:` guard leaves a stray `dummy` when empty.** It does not; both the block and its trailing `imgui.dummy` are inside the guard.
- **The new Help section needs Help-panel plumbing.** It does not; `popups/help.py` iterates `help_sections()` generically, and `test_sections_are_well_formed` covers key/title/body for every section including this one.
- **`test_help_content.py` lost its engine-uniform pin in the `(see Runs)` rename.** It did not; `test_engine_uniform_docs_cover_every_user_facing_builtin` and `test_engine_uniform_section_lists_each_uniform` are untouched and green, and the rename is inside a doc string neither asserts on.
- **The `(see Runs)` cross-reference dangles.** It matches: `separator_text("Runs")` in `pass_settings.py` and `**Runs**` in the Help body.
- **Conventions violations in the changed files.** None found: no `noqa`/`type: ignore`/`pyright: ignore`, no function-body imports, no `Any`, no literal colours or sizes at call sites (all through `COLOR`/`SIZE`/`SPACE`), and no history-narrating comments — the surviving comments state present-tense non-obvious facts.

## Coverage statement

Read end-to-end: `tests/test_ui_prose_budget.py`, `shaderbox/popups/pass_settings.py`,
`shaderbox/tabs/document.py` (the changed regions plus `draw`), `shaderbox/theme.py` (SIZE),
`shaderbox/help_content.py`, `shaderbox/widgets/pass_list.py`, `shaderbox/widgets/uniform.py`,
`.claude/skills/imgui-ui/SKILL.md` §§ 0-4, `ai_docs/conventions.md` (Code rules and the design
laws), the project `CLAUDE.md`, `tests/test_help_content.py`, and the supporting primitives
`modal_window`, `clickable_label`, `_ellipsize`, `uniform_name_label`.

Mutation battery (a)-(h) run as temporary edits to real source files, each restored and the
restore verified with `git diff --quiet` before the next ran. Five headless probes measured the
modal's size and clamp behaviour, the glyph rasterization, the font advance and `_ellipsize`
threshold, and the auto-block column alignment.

Not covered: the interactive appearance of any of this in the running app — every visual claim
here is a headless measurement, and per skill § 0 that is verification for geometry and
crash-freedom, not for aesthetics.

`git status --short` at completion shows only this review plus two other agents' untracked
spec files:

```
?? ai_docs/features/069_tutorial_walk_findings/02_keybindings.md
?? ai_docs/features/069_tutorial_walk_findings/40_wave_f_editor_chrome.md
?? ai_docs/features/069_tutorial_walk_findings/reviews/wave_b_post_code.md
```

`git diff --quiet` over tracked files passes: every probe edit was restored.

---

# Round 2 (closure)

Narrow closure round against `0ce84f8` ("069 W-B fixes: derive the gate's domain, pin the
gear"). Every measurement below was taken in an isolated `git worktree` detached at
`0ce84f8`, so nothing in this round touched the main working tree (other waves were
editing it concurrently).

## Per-finding verdicts

| # | Verdict | The line or measurement |
|---|---|---|
| F1 | **CLOSED** | `uv run ruff check .` -> `All checks passed!`; `ruff format --check .` -> `234 files already formatted`; `xvfb-run -a make gates` unpiped -> `EXIT=0`, `gates: GREEN -- check passed, test passed, smoke passed` |
| F2 | **CLOSED** | Mutation h2, fully green at `ccd446b`, now fails on the CALLER: `badge_chip.tooltip` and `badge_chip.label` at `pass_list.py::_settings_overlay:100` |
| F3 | **CLOSED** | Settled width `440.0` at 4, 20 and 40 rows with the constraint; `323.0`/`337.0` without it |
| F4 | **CLOSED** | At 60 rows the content region narrows 440.0 -> 410.0 (bar drawn); with `no_scrollbar` it stays 424.0 (bar suppressed) |
| F5 | **CLOSED** | Green at 18 characters, **red at 19** and at 21 — the real `_ellipsize` boundary. The old check passed at 19. |
| F6 | **CLOSED** | The `pass_settings` Help body now carries "`16-bit float` holds values above 1, which is what bloom and feedback need, and is the default" |

**Round 2 overall: PASS.**

### F1 — the gate is green, and the environmental caveat is real

On a clean checkout of `0ce84f8`, both ruff halves pass and the unpiped gate exits 0 with
all three stages green. The coordinator's note about this box checks out independently:
`glfw.get_monitors()` returns `monitors: 0`, and without a virtual display `make gates`
dies at `tests/test_canvas_fields.py Fatal Python error: Segmentation fault` -> `gates: RED
(exit 2)`. I confirmed that is not this diff by checking out `6a85564` — the commit BEFORE
either W-B commit — where the same test segfaults identically. Environmental, pre-existing,
correctly diagnosed.

The fix-up's commit message also names the mechanism behind the original miss: the
`make check` that was read had been auto-fixed by pre-commit mid-run. That is the
conventions.md "run it twice, judge by exit code" rule catching itself, and naming it is
the right resolution — the mechanism changed, not just the symptom.

### F2 — the domain is derived, and the hole I demonstrated is shut

`_SCORED` is now `_IMGUI_ROWS + _derived_rows()`, where `_derived_rows` reflects over
`vars(ui_primitives)`, selects any parameter named in `_PARAMETER_BUDGETS`, and reads the
positional index from `inspect.signature` so a positional caller is scored too. Census:
**239 sites / 173 measurable / 66 unmeasurable**, against 106/71/35 at `ccd446b` — the
shipped table was covering 44% of its own domain, as the commit message says.

Both variants of mutation (h) re-run:

| Variant | At `ccd446b` | At `0ce84f8` |
|---|---|---|
| h1 — helper forwards to `set_tooltip` | RED, but only on the helper's own forwarding call; caller's copy uncollected | RED on `badge_chip.tooltip` AND `badge_chip.label` at the caller, plus the forwarding site |
| h2 — helper draws its own tooltip via `begin_tooltip` | **222 passed, fully green** | **RED** on `badge_chip.tooltip` and `badge_chip.label` at the caller |

h2 is the one that matters: the shape that was completely invisible now fails at the
caller, which is where the authored copy lives.

I also probed the new escape hatch, since a derived domain with a by-name exemption list
can be widened by writing a name into it. Adding `"clickable_label"` to `_NOT_UI_COPY` does
not go quietly green — it goes red on
`test_every_unmeasurable_entry_still_names_a_real_site[...::uniform_name_label]`, because
suppressing the helper strands an allowlist entry that no longer names a readable site. The
exemption list cannot be used to hide a live helper without tripping something.

`test_a_helper_drawing_its_own_tooltip_is_still_measured_at_its_callers` anchors the
regression to `help_marker`, which genuinely IS that shape (it draws via
`begin_tooltip`/`text_unformatted`), and asserts it stays that shape. That is an anchor to a
real artifact rather than a fixture, which is the right construction.

### F3 — the width token is load-bearing again

Independent probe replicating the shipped call sequence
(`set_next_window_size_constraints((W,0),(W,display_h-MARGIN))` then
`begin_popup_modal(flags=always_auto_resize)`), reading `get_window_size()` on a settled
frame:

```
   n  constraint   width  height  scroll_max
   4       False   323.0   412.0         0.0
   4        True   440.0   412.0         0.0
  20       False   337.0   762.0        18.0
  20        True   440.0   744.0        36.0
  40       False   337.0   762.0       478.0
  40        True   440.0   744.0       496.0
```

Width is exactly the token at every content size, and the height still follows content
(412 at 4 rows) rather than being pinned. The stronger check: changing
`SIZE.PASS_SETTINGS_W` to 520 settles the window at **520.0px**, so the token now moves the
window — which was the whole complaint in F3, since at `ccd446b` a reader would have
changed 440 and seen nothing happen.

`tests/test_pass_settings_layout.py` pins this with a real app-fixture frame, and its
falsifier holds: deleting the `set_next_window_size_constraints` call fails with
`AssertionError: the gear settled at 323.0px wide, not the 440 token` — the same 323 my own
Round 1 probe measured.

### F4 — the scrollbar appears exactly in the clamped case

`no_scrollbar` is dropped. `scroll_max_y` alone does not distinguish the two flag settings
(the window scrolls either way — imgui's own stub says "window can still scroll with mouse
or programmatically"), so I measured whether the bar is DRAWN, via the content region:

```
 rows  scrollbar_flag   win_w  content_avail  scroll_max  bar drawn?
    5            True   440.0          424.0         0.0  no
    5           False   440.0          424.0         0.0  no
   60            True   440.0          410.0       307.0  YES
   60           False   440.0          424.0       307.0  no
```

With the shipped flags the content region narrows by 14px once content exceeds the display,
and is full-width when it does not. So the bar appears only in the clamped case, which is
what the code comment claims and what F4 asked for. The height also now stops at 744
(`display_h - PASS_SETTINGS_MARGIN`) rather than jamming against the viewport at 762.

### F5 — the bound is measured, and off-by-one in the SAFE direction now

The arithmetic bound is gone, replaced by `calc_text_size` plus an assertion against
`_ellipsize` itself inside a headless frame. Boundary, by injecting names into
`ENGINE_DRIVEN_UNIFORMS` and running the check:

```
  len=18 u_pass_iterationss   -> 1 passed
  len=19 u_pass_iterations_x  -> 1 failed
  len=20 u_pass_iterations_xx -> 1 failed
```

19 is the character count that passed the old test while `_ellipsize` truncated it; it is
now red. Asserting against `_ellipsize` cannot drift from the renderer, because it is the
renderer — the right shape, not just the right number.

### F6 — the criterion is back

The `pass_settings` Help body gains: "**format** is how much each pixel can hold. `8-bit`
clamps to 0-1 and is right for a final image; `16-bit float` holds values above 1, which is
what bloom and feedback need, and is the default; `32-bit float` costs twice the memory and
is rarely worth it." That restores all three choosing criteria, including the "what bloom
and feedback need" phrase F6 named as lost. The tooltips stay cut, so the budget holds and
the documentation moved to where a reader chose to read it — which is § 2's own prescription.

## The two implementer calls

### A 3-word budget for button labels: SOUND

The skill's § 2 table genuinely has no row for a button label, so a number had to be chosen
rather than looked up. Three is defensible on the evidence, not on taste.

Distribution over the 72 measured button-label sites: `{0: 2, 1: 45, 2: 18, 3: 7}`. Nothing
reaches 4, so the budget is not accommodating a long tail — it is drawing the line exactly
one word above the mass of the data.

Setting `_BUTTON_LABEL_BUDGET = 2` and re-running fails 7 sites. Three are real authored
labels that would have to be cut: `"Add to pack"`, `"Open a copy"`, `"Insert at caret"`
(twice). Those are § 1's own tier examples in form — an action phrase with a verb and its
object — and shortening them costs meaning ("Insert" does not say where). The other four
are f-string artifacts of the scorer, not prose: `f'{full_w}x{full_h}'` renders as
`1080x1080`, one visible token, but scores 3 because each `FormattedValue` counts as a
word. A budget of 2 would therefore reject four dimension labels that contain no authored
copy at all.

So 2 is not viable without either damaging working labels or special-casing the scorer, and
3 admits nothing a reader would call a sentence (longest button label overall is 26
characters, `"Load client_secret.json..."`). The reasoning is also written at the constant
with § 1's examples cited, which is where the next reader will look.

The one honest wrinkle, and it is not a blocker: the f-string scoring means a button label
of three interpolations sits exactly at budget while carrying no authored words. That is a
property of the whole scorer (`FormattedValue` = 1 word), not of this budget, and it errs
toward rejecting rather than admitting prose.

### Keeping `always_auto_resize` with a width constraint, rather than deleting `PASS_SETTINGS_W`: SOUND

My Round 1 finding offered two exits and said the half-state was the bad one: keep the flag
and delete the inert token, or drop the flag. The implementer took a third, and it satisfies
the actual complaint better than either.

The complaint was that the token was inert — passed in and discarded, so a reader changing
it would see nothing move. That is now measurably false: 440 -> 440.0, 520 -> 520.0. The
token is load-bearing, and the mechanism by which it binds is written at the call site.

Deleting `PASS_SETTINGS_W` (my suggestion) would have let the width free-float at 323 and
follow content. That is worse against the skill's § 3 jitter rule: a width that tracks
content varies with the widest row, and at `ccd446b` the width only happened to be stable
because the rows use `same_line(absolute x)` — an accident of the row layout, not a
guarantee. Pinning the width and letting only the height follow content is the combination
that gives finding #7 what it asked for (no scrolling in the ordinary case) while keeping
the axis the user reads across fixed.

The call is also pinned by a test with a working falsifier, so it cannot silently regress to
the inert state it came from. Judging by outcome rather than by which of my two options was
taken: this is the better design.

## Round 2 coverage and tree state

Measured, not read: ruff both halves and the unpiped gate on a clean checkout; the
pre-existing segfault reproduced at `6a85564`; mutations h1 and h2 re-run; a bogus
`_NOT_UI_COPY` entry probed; the census counted from the shipped collector; gear width at 3
content sizes x constraint on/off; `PASS_SETTINGS_W` retuned to 520; the constraint deleted;
scrollbar visibility via content-region width at 2 content sizes x flag on/off; the
`AUTO_NAME_W` boundary at 18/19/21 characters; the button-label distribution and a budget of
2; the Help body rendered through `help_sections()`.

Not covered: the appearance of any of this in the running app. Every claim here is a
headless measurement, which per skill § 0 settles geometry and crash-freedom, not aesthetics.

All Round 2 work ran in a throwaway worktree at `0ce84f8`; every probe edit inside it was
restored and verified with `git diff --quiet` before the next ran, and the worktree ended
clean. The main working tree carries another wave's in-flight edits (the editor FFI and
`theme.py` editor tokens, wave F) plus other agents' untracked spec files — none of them
files this round touched.
