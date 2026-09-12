# 091 pre-implementation review round 2 — correctness & design

**Verdict: PARTIAL.** Four spec edits. Round 1's seven edits all landed; the residue is new, and
three of the four are things round 1 got wrong in the same sentence the revision copied. Ranked by
cost to discover during implementation:

1. **R1 — `_PASS_NAME_RE` is private to `project_session.py` and both new leaf modules need it.**
   D2 puts `group_slug` in `pass_graph.py`, D4 puts `plan_import` in `pass_import.py`
   ("importing `pass_graph` only") and tells it to reject "a group name failing `_PASS_NAME_RE`".
   `project_session` imports `pass_graph` (`project_session.py:39-46`), so the back-import is a
   cycle, and the project bans `if TYPE_CHECKING` and inline imports. The regex has exactly one
   home and two uses, both in `project_session.py:124-130`; it has to move down to `pass_graph.py`.
2. **R2 — D7's `bordered` compensation is 8px, not "the border size".** Measured in a real frame:
   a bordered child's content starts at `(8, 8)` with `avail` 152×164; an unbordered one at
   `(0, 0)` with 168×180. `ChildFlags_.borders` means "show an outer border **and enable
   WindowPadding**" (imgui's own doc string) — and imgui already ships the one-flag answer,
   `ChildFlags_.always_use_window_padding`, which measures byte-identical to `borders`.
3. **R3 — three of D8's five tints collide, and one fails D8's own assert.** `purple_b` IS
   `COLOR.SELECT`; `blue_n` IS `COLOR.STATE_INFO` (so the tuple is red on the assert the spec
   proposes, exactly as `blue_b` was in round 1); `aqua_n` / `orange_n` / `blue_n` are the three
   accent presets' **active** colours, which the existing invariant does not look at because it
   enumerates element [0] only.
4. **R4 — D11's "the `if not any_popup_open():` branch takes the predicate's negation" is not
   expressible.** The negation is true for PASS_SETTINGS, SETTINGS, HELP, PROJECTS and every other
   modal, so branch A would fire for all of them and branch C would be dead. Two predicates, and
   `_tick_frame_state`'s own popup gate needs the second one too.

`make gates` exit 0 before reviewing, captured unpiped (`make gates > log 2>&1; echo $?`): check
passed, test passed, smoke **passed** (not skipped). So every failure below is about the spec.

---

## A. Round 1, item by item

| Round-1 item | New spec text | Verdict |
|---|---|---|
| **E1 / F1** — "Bloom Chain" is not a shipped example | Goal: "It is not a shipped example: it is the test fixture `tests/fixtures/bloom_chain/` (and a document in the maintainer's own project). The only multi-pass shipped example is Radiance Cascades, whose one entry point `paint` has no samplers." | **CLOSED** — and round 1 was wrong. `tests/fixtures/bloom_chain/` exists with all five passes; `test_lazy_compile.py:31` and `test_default_wiring.py:40` already load it. Probed the real fixture: entry points `['scene']`, output `composite`, wiring `bright.u_scene→scene`, `blur.u_bright→bright`, `composite.{u_blur,u_scene,u_trail}`, `trail.{u_prev→trail,u_scene→scene}`. |
| **E2 / F2(a)** — "`tiles_per_row` is untouched" false; the mock's 4px gap | D7: "The mock's 4px in-group gap is dropped: a per-run gap would break `tiles_per_row`'s single-gap arithmetic… `tiles_per_row` is then genuinely untouched (one caller)." | **CLOSED.** One caller confirmed (`pass_list.py:175`, the only hit). |
| **E2 / F2(b)** — the 2px-outside outline clips | D7: "a rounded rect **inset by 1px** from the run's outer tile edges… (`tiles_per_row` charges the last tile no trailing gap, so a full row's slack is as little as 4px at a 700px panel and an outside rect clips)" | **CLOSED on the remedy, the number is wrong.** The inset fixes it. But the worst case is **0.0px**, not 4px: slack is exactly zero at `avail` 344, 520, 696, 872, 1048, 1224 (swept 200–1400 with the real `SIZE.PASS_TILE=168` / `SPACE.MD=8`). Round 1 said 4px at 700 and the spec copied it. Cosmetic — an inset rect is safe at 0 slack — so it rides Edit 2 rather than its own. |
| **E3 / F2(c)** — `bordered` cannot be a colour | D7: "`preview_cell` gains `bordered: bool = True`, which drops `ChildFlags_.borders` from the child's flags — the tile's border is imgui's own child border, and `border_color` only tints it, so a colour cannot turn it off." | **CLOSED.** Verified: `child_flags=imgui.ChildFlags_.borders` hardcoded at `ui_primitives.py:1268`, `border_color` pushes `Col_.border` at 1259-1261. The follow-on sentence about the compensation is **NOT CLOSED** → R2 / Edit 2. |
| **E4 / D5 roster** — `NoSource`/`AutoSource` omitted, `Image(value.texture)` | D5: "over the whole of `core.UniformValue` plus the three `SamplerSource` members, with no default branch… `PassSource` / `NoSource` / `AutoSource` by reference (frozen dataclasses, the reference is the value); … an `Image` re-opened from its file under the source's `media/<source pass>/` (never `Image(value.texture)`…)" | **CLOSED.** `UniformValue` at `core.py:167-174` is the domain named; `SamplerSource = PassSource \| NoSource \| AutoSource` at `pass_graph.py:211`, all three frozen dataclasses. |
| **E4 / `source_dir` redundant** | D5: "The source directory is `document_dir_of(source)`, the one place that knows the `passes/` depth." Signature has no `source_dir`. | **CLOSED.** Probed: `document_dir_of(d)` on the fixture loaded into a tmp dir returns that dir exactly. |
| **E5 / F4** — D11 needs a fourth branch | D11: "The render chain gets NO new branch: `ui.py`'s `elif EXAMPLES:` block takes that same predicate… With the project tab active the dialog renders the ordinary set, which is what the `if not any_popup_open():` branch does once it takes the predicate's negation." | **NOT CLOSED** → R4 / Edit 4. The no-fourth-branch goal is reachable; the stated mechanism is not. |
| **E6** — the graph-view bullet | Out of scope: "**A second, opt-in graph view of the passes.** 070 closed the graph view as the strip's replacement for a six-pass document at 480px. In this design round the maintainer reopened it as an opt-in second view… hand-drawn on the draw list (no `imgui_node_editor`, for a future port)… 070's decision gets its revisit pointer when that feature lands." | **CLOSED / consistent** — see the note below the table. |
| **F3** — D6's handover has no release | D6: "so `import_passes` releases it first (`try_to_release`, the same call `set_sampler_source` makes) and writes `uniform_values` directly, saving once at the end rather than calling `set_sampler_source` per row, which saves per call." | **CLOSED.** Verified `project_session.py:1009-1012`: `try_to_release(values.get(uniform))`, then the write, then `self.save_ui_document(ui_document)` — one save per call, exactly as stated. |
| **F5** — `GROUP_TINTS` fails D8's assert | D8: five tints "`purple_b`, `green_n`, `aqua_n`, `orange_n`, `blue_n` (`blue_b` is the blue accent's primary and `COLOR.TAG`, so it is out)" | **NOT CLOSED.** `blue_b` is gone, but the replacement tuple has two fresh collisions of the same kind → R3 / Edit 3. |

**E6 — the rewritten bullet is consistent with both anchors.** 070's decision reads "no graph
view. The strip is the one view of the passes; each tile names what it reads. The
`imgui_node_editor` question is closed with it" (`070_pass_reads/01_spec.md:27-28`), and its
status line says it was "closed as a row of chips under each tile, after a brainstorm that
rejected the graph view" — i.e. closed as *the strip's replacement*, which is the scope the new
bullet ascribes to it ("070 closed the graph view as the strip's replacement for a six-pass
document at 480px"). That is a faithful narrowing, not a misquote: 070's brainstorm rejected the
arc layouts for being "messy" and for degrading at 480 where the strip wraps, which is an argument
about replacing the strip, and says nothing about a second opt-in surface. The bullet also matches
Review history's own words — the maintainer "reopened it in this session as an opt-in second view
and chose hand-drawn rendering" — and it carries both halves that answer round 1's real
complaint: the feature is no longer asserted as numbered ("That is its own feature, numbered when
it is specced"), and the fold rejection no longer leans on it (the fold bullet's reason is now
"a group whose members are not contiguous simply draws as two runs (D7)", self-contained in 091).
The one loose thread is 070's own file, which still reads as closed with no pointer; the bullet
says that pointer lands with the new feature rather than now, which is a deliberate choice the
spec states.

---

## B. The decisions that changed, re-verified against the code

### D4 — the plan signature, "sources carries only the WIRED subset", the AutoSource paragraph

**CONSISTENT, one blocking gap (R1).**

`host_wiring` / `host_output` / `handovers` are all GL-free: `Wiring = Mapping[str, Mapping[str,
str]]` (`pass_graph.py:214`) and `handovers` is a `Collection[tuple[str, str]]`, so `plan_import`
stays pure as claimed.

The "sources carries only the WIRED subset" sentence is now the single mechanism round 1 asked
for, and its justification measures out. Probed `wired_pass` directly:

```
wired_pass(AutoSource(), 'u_bright', 'bloom_blur', {'bloom_bright','bloom_blur'})  -> None
wired_pass(AutoSource(), 'u_prev',   'bloom_trail', {'bloom_trail'})               -> 'bloom_trail'
```

So the spec's parenthetical "measured: `wired_pass(AutoSource(), "u_bright", "bloom_blur", …)` is
`None`" is correct and materializing is mandatory. The self-read case is the one place the spec
overstates: `u_prev` **does** resolve to the renamed self by the `_FEEDBACK_UNIFORM` branch
(`pass_graph.py:230-232`), so writing it explicitly is belt-and-braces rather than load-bearing —
but D4 gives the right reason for doing it anyway ("would break the day a group is renamed"), so
no edit.

The AutoSource paragraph's claim is demonstrated, not asserted:

```
wired_pass(AutoSource(), 'u_paint', 'bloom_x', {'bloom_x','paint'})  -> 'paint'
```

An undecided `u_paint` on a copied pass does catch a host `paint`. D4 now states this is intended
and says why, which is what round 1 asked for; the rejection of the materialize-to-black
alternative is recorded in Review history.

**The gap:** the rejection set includes "a group name failing `_PASS_NAME_RE`", and
`pass_import.py` imports `pass_graph` only. `_PASS_NAME_RE` is at `project_session.py:124` with its
only two uses at 124 and 128, and `project_session` imports from `pass_graph` at 39-46 — so
`pass_graph`/`pass_import` cannot reach it. D2 has the same problem: `group_slug` lives in
`pass_graph.py` and D2 says its output must satisfy that regex, and verification 2's falsifier is
stated in its terms. `pass_graph.py` is the GL-free leaf (`dev_flow.md:219`) and its own docstring
already owns "a pass name is a FILENAME and a graph key", so the regex belongs there.

### D5 — `ImportResult`, compile-before-save, the copy roster, `document_dir_of`

**CONSISTENT.** Nothing to add.

`ImportResult(error: str = "", notes: tuple[str, ...] = ())` is a plain frozen dataclass with
defaults; verification 8 reads `notes`, so the shape is exercised.

The compile-before-save rule is load-bearing for a reason sharper than the spec states, and the
spec's version is still correct. `UIDocument.save` computes `live = any(p.program is not None for
p in self.document.passes.values())` (`ui_models.py:417`) — **any**, not all. The host's own
compiled pass therefore makes `live` True, and the prune at 462-471 builds `live_rows` from
`get_uniform_hash` over `get_active_uniforms()` of **every** pass, which for an uncompiled copy is
empty. So a never-compiled copied pass contributes no live rows and its merged `ui_uniforms` rows
are dropped in the same save that wrote them — precisely what D5 says.

The copy roster's domain matches the code: `UniformValue` is `int | float | Sequence[int] |
Sequence[float] | MediaWithTexture | moderngl.Texture | moderngl.Buffer` (`core.py:167-174`), plus
the three `SamplerSource` members (`pass_graph.py:203-211`, each `@dataclass(frozen=True)`).
"With no default branch" over that union is decidable.

`document_dir_of` verified by probe — `document_dir_of(d) == tmp` for the fixture copied into a
tmp dir (`document.py:1092-1099`, `.parent.parent` off a pass file). Taking no `source_dir`
parameter is right.

The `get_uniform_hash` precedence note is accurate: the key is `f"{u.name}_{u.array_length}_
{u.dimension}_{u.gl_type}"` (`util.py:78-82`) with no pass in it.

### D6 — the host compile, the handover rejection, the release

**CONSISTENT.** All three claims check out against the code, two by measurement.

*The host compile.* `effective_wiring` answers over `sampler_names(render_pass)` when the program
exists and falls back to "explicit `PassSource` rows only" otherwise (`document.py:689-714`, the
`if declared else` on 705-709). Measured on the bloom fixture: before compiling,
`{'blur': {}, 'bright': {}, 'composite': {}, 'scene': {}, 'trail': {}}` — every pass a root,
exactly the spec's "the uncompiled bloom fixture's wiring is `{'blur': {}, …}`, every pass a
root". After compiling each program-less pass, the full five-pass DAG. So compiling the host is
necessary for the reader checkboxes, as D6 says.

*The rejection.* `UIDocument.save`'s per-pass loop has the branch D6 names verbatim: `if
render_pass.program is None: existing = _existing_rows(dir, pass_name); … continue`
(`ui_models.py:484-489`) — a handover written onto a program-less host pass is overwritten by its
disk rows at save time and silently lost. Rejecting such a handover with a message naming the pass
is the right call and is now stated.

*The release.* `set_sampler_source` is `try_to_release(values.get(uniform))` → write →
`save_ui_document` (`project_session.py:1009-1012`), and `try_to_release` calls `.release()` on
anything that has one (`util.py:100-104`). So "the same call `set_sampler_source` makes" is
literal, and "saving once at the end rather than calling `set_sampler_source` per row, which saves
per call" is accurate about the cost.

### D7 — inset outline, draw order, `bordered` as the child flag, `bordered=border is not None`

**INCONSISTENT on the inset compensation (R2); the rest CONSISTENT.**

*The flag.* Correct. `preview_cell` passes `child_flags=imgui.ChildFlags_.borders` unconditionally
(`ui_primitives.py:1268`) and `border_color` only pushes `Col_.border` (1259-1261), so
`bordered=False` must drop the flag.

*`bordered=border is not None`.* Correct as a predicate — it keeps the flag for the accent output
border and the red error border, which is what "those still win" needs.

*Draw order.* Correct and reachable. `pass_list.draw` computes `per_row` and walks `order` with
`imgui.same_line(spacing=SPACE.MD)` (`pass_list.py:175-187`), so every tile's rect is predictable
before it is drawn, and the parent draw list is reachable from the strip's scope. Emitting the
rect before the run's tiles and the label's fill after is expressible.

*The inset number.* The remedy holds but the figure is wrong. Swept the real tokens over
`avail` 200–1400:

```
min slack = 0.0 px, at avail = 344, 520, 696, 872, 1048, 1224
avail=480  -> n=2 span=344 slack=136
avail=700  -> n=4 span=696 slack=4
avail=1040 -> n=5 span=872 slack=168
```

The spec says "as little as 4px at a 700px panel"; the true floor is **0.0px**, hit whenever
`avail` lands exactly on `n*168 + (n-1)*8`. A 1px inset is still safe at 0 slack, so the decision
does not change — only the justification's number, which a future reader would otherwise trust.

*The compensation — the defect.* D7 says dropping the flag "removes imgui's border inset, so an
unbordered cell pads its content by the border size". The border size is 1px
(`theme.py:422` `child_border_size = 1.0`); the real shift is **8px in both axes**. Measured in a
real imgui frame through the `app` fixture, same `size=(168, 180)` both times:

```
bordered   content offset from window pos (8.0, 8.0)   avail 152.0 x 164.0
plain      content offset from window pos (0.0, 0.0)   avail 168.0 x 180.0
child_border_size 1.0   window_padding 8.0 8.0
```

imgui's own flag doc says why: `ChildFlags_.borders` = "Show an outer border **and enable
WindowPadding**", and `ChildFlags_.always_use_window_padding` = "Pad with style.WindowPadding even
if no border are drawn (no padding by default for non-bordered child windows…)"
(`imgui/__init__.pyi:3352-3357`). So the whole cell's layout — `avail`, the image rect, the
footer's `origin.y + img_h`, the chip row's inset — shifts by 8px and shrinks by 16, not by 1 and
2. Hand-rolling a pad would mean recreating `WindowPadding` inside the child; the one-flag answer
measures identical:

```
bordered   (8.0, 8.0, 152.0, 164.0)
pad_only   (8.0, 8.0, 152.0, 164.0)      # ChildFlags_.always_use_window_padding
bordered == pad_only: True
```

That is a flag swap, not a compensation, and it keeps verification 18 (the same-screen-position
assert) honest rather than asserting against a pad the implementer tuned by eye.

### D9 — the create-mode group row

**CONSISTENT; the "the way the draft's target and runs are applied" phrasing is accurate, and its
accuracy is the thing to notice.**

Read `App.create_pass_from_draft` (`app.py:1075-1102`). After `add_pass`, the draft's fields are
applied **conditionally, each against `PassEntry()`'s default**:

```python
entry = draft.entry
if entry.target != PassEntry().target:
    error = self.session.set_pass_target(document_id, name, entry.target)
if entry.iterations != PassEntry().iterations:
    error = self.session.set_pass_iterations(document_id, name, entry.iterations)
```

So "applies it through `set_pass_group` after `add_pass`, the way the draft's target and runs are
applied" resolves to `if entry.group != PassEntry().group:` — i.e. `if entry.group:` — then
`self.session.set_pass_group(document_id, name, entry.group)`. That is correct and unambiguous,
and it fits D1's warning about the two sites that compare against field defaults: these are
per-field comparisons, not an "is this entry default" test, so `group` joins them without
disturbing either. The one ordering constraint the spec leaves implicit is harmless — the call
must sit before `self.pass_draft = None` / `pick_pass` at 1098-1101, which is the only slot
"after `add_pass`" leaves.

Edit mode is equally clean: `_apply_entry` (`pass_settings.py:130-142`) is already a before/after
per-field diff, so a `group` clause is one more `if after.group != before.group:`. Create mode's
draft body (`_draw_draft`, 67-100) has no session-writing name row, so editing `draft.entry.group`
there is the matching shape.

Round 1's positional-append finding survives re-checking: `capabilities.py:458-470` is
positional-only (the `/`), and all five test call sites pass nine positional arguments
(`tests/test_copilot_pass_tools.py:49,55,76,80,85`). `_pass_table`'s format string
(`backend.py:1289-1292`) is the seam for the `, group <name>` suffix.

### D10 — the busy guard, the Escape branch, `rejection`, the per-frame plan

**CONSISTENT.** Every named seam exists in the shape the spec assumes.

`_copilot_busy_blocked(action)` pushes `f"{action} is locked while the assistant is working"` and
returns True while `copilot_turn_active` (`app.py:919-927`) — a guard that both refuses and
notifies, which is what verification 16 asserts.

The Escape branch: `_handle_escape` dismisses one thing most-modal-first, and inside the
`elif app.any_popup_open():` arm the `PASS_SETTINGS` case calls `app.close_pass_settings()` while
everything else falls through to the bare `app.popup_state = PopupState.CLOSED`
(`hotkeys.py:376-387`). So "Escape reaches `close_import_passes` through its own branch in
`hotkeys.py`, the way `PASS_SETTINGS` does; the bare `popup_state = CLOSED` fallthrough would
leave the draft populated" is literally right, and verification 15's falsifier is real.

The `PopupState` gate is as the spec states: `test_every_popup_state_has_a_draw_call`
(`tests/test_project_management.py:636-667`) ASTs `ui.py` and asserts `len(called) ==
len(PopupState) - 1`; today `len(PopupState)` is 8 (`CLOSED, EXAMPLES, HELP, SETTINGS,
PASS_SETTINGS, EMOJI_PICKER, SHADER_LIB_PICKER, PROJECTS`), so verification 14's "7 called" is
accurate.

`rejection: str` on the draft plus "the plan is recomputed each frame the dialog draws (never
cached on selection)" is a decidable contract, and verification 17's pumped-frame shape has a
precedent rig (`tests/test_pass_settings_layout.py`'s `_gear_sizes`, which monkeypatches
`pass_settings._draw_body` and pumps `imgui.new_frame()` / `end_frame()` itself).

### D11 — one predicate, or does the project-tab case need something else?

**NOT expressible as written. It needs two predicates, and a third edit site the spec does not
list.**

`ui.py:453-515` is three mutually exclusive branches:

```
A: if not app.any_popup_open():                                        -> the tick_documents set
B: elif app.popup_state == PopupState.EXAMPLES:                        -> the examples set
C: elif app.popup_state == PopupState.PASS_SETTINGS and renders_this_frame(...): -> current only
```

Let `P = EXAMPLES or (IMPORT_PASSES and the Examples tab is active)`. B taking `P` is exactly
right and costs nothing. A taking `not P` is wrong in two ways at once:

- `P` is False for PASS_SETTINGS, so A fires and **C becomes dead code** — the gear would lose the
  behind-the-modal render that D11 is borrowing the pattern from.
- `P` is False for SETTINGS, HELP, PROJECTS, EMOJI_PICKER and SHADER_LIB_PICKER too, so A fires
  for every one of them and the full-set render resumes behind every modal — the thing 090 D10
  deliberately paused ("while the Examples popup is open its documents REPLACE the normal set…
  so no `tick_documents` document renders behind it", `ui.py:269-273`).

What the project-tab case actually needs is its own positive predicate:

```
P_examples = EXAMPLES or (IMPORT_PASSES and tab is Examples)
P_project  = IMPORT_PASSES and tab is Project
A: if (not app.any_popup_open()) or P_project:
B: elif P_examples:
```

That still adds no branch — the goal D11 states is met — but it is two predicates, not one and its
negation.

**And there is a second site.** `_tick_frame_state` gates the *construction* of `tick_documents`
on the same condition (`ui.py:250`): with any popup open, `tick_documents` is just
`[current_document_id]` and neither the render-all fan-out nor the one-pending-first-render
election runs. The dialog's project tab draws a card grid through `draw_document_preview_button`,
which blits each document's live canvas texture (`widgets/document_grid.py:22-32`) — so without
the same `or P_project` on line 250, every project document that has not yet had a first render
shows black behind the dialog, and the first-render sweep never elects one. The spec's
"`planned` / `planned_documents` / `current_planned` follow it unchanged" covers the planned set
at 270-284, not this gate above it; `ui.py`'s "Files touched" line needs to name both.

---

## C. New defects

**R1 — `_PASS_NAME_RE` cannot be reached from either new leaf module.** Evidence:
`project_session.py:124` `_PASS_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")`, used only at
124 and 128 (`grep -rn "_PASS_NAME_RE" shaderbox/ tests/` → three hits, all in that file);
`project_session.py:39-46` imports `MAX_ITERATIONS, PassEntry, PassGraph, PassSource,
SamplerSource, TargetConfig` from `shaderbox.pass_graph`, and `pass_graph.py`'s imports are
stdlib + pydantic only (lines 32-36). D2 needs it in `pass_graph.py`, D4 needs it in
`pass_import.py` ("importing `pass_graph` only"), and the project bans the two escapes
(`if TYPE_CHECKING`, inline imports). → Edit 1.

**R2 — D7's inset compensation is off by 8×, and imgui ships the fix as a flag.** Evidence: the
two frame measurements above (`(8,8)`/152×164 bordered vs `(0,0)`/168×180 plain, `child_border_size
1.0`, `window_padding 8.0 8.0`), the library doc strings at
`.venv/.../imgui_bundle/imgui/__init__.pyi:3352-3357`, and the third measurement showing
`always_use_window_padding` identical to `borders`. As written an implementer pads by 1px, and
every grouped tile's picture sits 7px off its neighbours with 16px more room than it should have —
which verification 18 would then be written to bless, since its author would tune the pad until
the assert passed. → Edit 2.

**R3 — three of D8's five tints collide; one fails D8's own proposed assert.** Evidence, by value:

```
purple_b   -> ['SELECT']
green_n    -> clean
aqua_n     -> ['ACCENT_ACTIVE']
orange_n   -> ['ACCENT_ACTIVE']
blue_n     -> ['ACCENT_ACTIVE', 'STATE_INFO']
```

`_ACCENTS` is `{name: (primary, active, alpha_fill)}` (`theme.py:87-100`) and the existing
invariant enumerates element [0] only — `_accent_primaries = {primary for primary, _active,
_alpha in _ACCENTS.values()}` (`theme.py:201-203`) — which is why three accent *active* colours
pass a check whose comment promises "may not equal any accent preset's primary". Two of the five
are hard failures:

- `blue_n` is `COLOR.STATE_INFO` (`theme.py:169`), so the tuple is **red on D8's own assert**
  ("no tint equals an accent primary **or a state hue**") — the identical shape as round 1's
  `blue_b`, one token over.
- `purple_b` is `COLOR.SELECT` (`theme.py:161`). D8's assert does not cover `SELECT`, so this one
  lands silently: the group outline becomes the same colour as the selection outline, on the same
  tiles. `SELECT`'s own comment states the context rule this violates — "SELECT outlines a tile
  inside an accent-chromed panel, so it must differ from every accent primary AND every state
  hue" (`theme.py:204-206`) — and a group outline around a run of tiles is the same kind of nested
  outline, so the fixed-hue invariant has to include `SELECT` on the group side.

The palette cannot supply five well-separated clean hues. Enumerating every chromatic entry
against accent primaries ∪ accent actives ∪ `{SELECT, STATE_*, TAG, FAVS}` leaves exactly five:
`red_n`, `green_n`, `green_b`, `yellow_n`, `purple_n` — and two of those are unusable as a pair
(`green_n` hue 59.5° vs `green_b` 61.2°, indistinguishable as two group tints) while `red_n`
(hue 2.4°) sits next to `STATE_ERROR`'s red on tiles that also carry the red error border. So the
honest tuple is **four**: `purple_n`, `green_b`, `yellow_n`, `aqua_n`. `aqua_n` is an accent
*active*, not a primary — acceptable, and the spec should say so rather than leave the reader to
discover the invariant does not look there. → Edit 3.

**R4 — D11's negation is not expressible.** Evidence: the three-branch chain at `ui.py:453-515`,
the eight `PopupState` members, and the `tick_documents` gate at `ui.py:250`. Worked through in
section B. → Edit 4.

---

## Spec edits proposed

**Edit 1 — move `_PASS_NAME_RE` down, and say so.** In "Files touched", change the `pass_graph.py`
line to:

> - `shaderbox/pass_graph.py` — `PassEntry.group`, `PassGraph.with_group(name, group)`,
>   `entry_points(wiring)`, `group_slug(name)`, `group_runs(order, groups)`, and `PASS_NAME_RE`
>   moved down from `project_session.py` (its two uses there import it back).

and append to D1's sentence "The name obeys `_PASS_NAME_RE` (it is drawn on a border and prefixes
filenames)":

> That regex moves from `project_session.py` to `pass_graph.py` and loses its underscore:
> `group_slug` (D2) and `plan_import` (D4) both validate against it and both sit at or below
> `pass_graph` in the import order, so it cannot stay private to a module that imports them.

**Edit 2 — D7, the compensation is a flag, and the slack floor is 0.** Replace "Dropping the flag
also removes imgui's border inset, so an unbordered cell pads its content by the border size to
keep the picture and footer on the same pixel as its bordered neighbours (the maintainer's locked
answer: the cards keep their size)." with:

> `ChildFlags_.borders` also enables `WindowPadding` (imgui's own flag doc), so dropping it alone
> moves the cell's content from `(8, 8)` to `(0, 0)` and grows `avail` from 152x164 to 168x180 at
> a 168-wide tile — the picture, the footer and the chip row all shift. So `bordered=False` passes
> `ChildFlags_.always_use_window_padding` instead of `none`, which is the flag that exists for
> exactly this ("pad with style.WindowPadding even if no border are drawn") and measures
> byte-identical to `borders`. No hand-rolled pad: the cards keep their size (the maintainer's
> locked answer) because the padding is the same padding, not a recreated one.

and in the inset paragraph replace "so a full row's slack is as little as 4px at a 700px panel and
an outside rect clips" with:

> so a full row's slack reaches exactly 0px whenever the panel lands on `n*168 + (n-1)*8` (344,
> 520, 696, 872, 1048…) and an outside rect clips

**Edit 3 — D8, four tints and a wider invariant.** Replace the first two sentences of D8 with:

> **D8 — group tints are theme tokens, picked by a stable hash.** `COLOR.GROUP_TINTS`, four hues:
> `purple_n`, `green_b`, `yellow_n`, `aqua_n`. Four and not five because the palette has no fifth
> that is both free and far enough away: `purple_b` is `COLOR.SELECT` and a group outline is the
> same nested-outline context `SELECT`'s own invariant protects; `blue_n` is `COLOR.STATE_INFO`;
> `blue_b` is the blue accent's primary and `COLOR.TAG`; `green_n` sits 2 degrees of hue from
> `green_b`; `red_n` reads as the red error border these tiles can also carry. `aqua_n` is the
> aqua accent's ACTIVE colour, not its primary — allowed, and the reason the assert below names
> actives explicitly rather than trusting the existing `_accent_primaries` set, which enumerates
> element [0] of each preset only.

and in the same decision replace "pins that no tint equals an accent primary or a state hue" with:

> pins that no tint equals an accent primary, an accent ACTIVE, any `STATE_*` hue, `COLOR.SELECT`,
> `COLOR.TAG` or `COLOR.FAVS`

Verification 11 then reads `% 4`, and its (b) falsifier should name one of the real collisions
(`purple_b`, which the narrower check would let through) rather than `yellow_b`.

**Edit 4 — D11, two predicates and the second edit site.** Replace D11's body from "The render
chain gets NO new branch" to the end with:

> The render chain gets NO new branch, but it takes two predicates rather than one and its
> negation: `examples_planned` is "the Examples popup, or the import dialog with its Examples tab
> active" and `import_project_tab` is "the import dialog with its project tab active".
> `ui.py`'s `elif EXAMPLES:` block takes `examples_planned`, so the one-example-per-frame
> first-render election, the `renders_this_frame` gate and the profiler span keep one home —
> `ui.py` already carries two copies of that budget rule and the funnel bullet names the second
> sibling as the trigger, so a third copy is out. The `if not any_popup_open():` branch takes
> `or import_project_tab`, NOT `not examples_planned`: that negation is true for PASS_SETTINGS,
> SETTINGS, HELP and PROJECTS too, so it would fire the full-set render behind every modal and
> leave the `elif PASS_SETTINGS` branch unreachable. The same `or import_project_tab` goes on
> `_tick_frame_state`'s own `if not any_popup_open():` gate, which is where `tick_documents` is
> BUILT: without it the set stays `[current]`, and the project tab's card grid — which blits each
> document's live canvas — shows black for every document whose first render has not happened,
> with no pending-first election to fix it.

and in "Files touched" change the `ui.py` line to:

> - `shaderbox/ui.py` — the modal's draw call in the popup chain; the two planned-set predicates,
>   on the render chain AND on `_tick_frame_state`'s `tick_documents` gate (D11).

---

## False trails — probed this round, fine, do not re-check

- **The `bloom_chain` fixture.** It exists (`tests/fixtures/bloom_chain/` with
  `scene/bright/blur/trail/composite`), is already loaded by `test_lazy_compile.py:31` and
  `test_default_wiring.py:40`, and its real shape matches every claim the revised spec makes of
  it: one entry point `scene`, output `composite`, the full DAG after compiling. Round 1's E1/F1
  was a false negative; the revision's correction is right and needs no further change.
- **Verification 5's counts and asserted rows.** `scene` fed leaves exactly four copied passes,
  and all three asserted rows (`bloom_bright.u_scene → main`, `bloom_composite.u_blur →
  bloom_blur`, `bloom_trail.u_prev → bloom_trail`) match the measured wiring. The fixture also
  wires `composite.u_scene` and `trail.u_scene` to `scene`, which the item does not assert —
  under-assertion, not a wrong assertion, so no edit.
- **`document_dir_of` as the `source_dir` replacement.** Probed on the fixture loaded into a tmp
  dir: returns that dir exactly. D5 dropping the parameter is right.
- **D5's compile-before-save rule.** Confirmed from `ui_models.py:417` (`live = any(...)`) and the
  prune at 462-471: the host's own compiled pass makes `live` True, so an uncompiled copy's
  merged rows are pruned in the same save. Verification 7's falsifier is real.
- **D6's release and save-per-call claims.** `project_session.py:1009-1012` is literally
  `try_to_release` → write → `save_ui_document`. Both halves of the sentence are accurate.
- **D6's uncompiled-host hole and the `program is None` branch.** Measured (the fixture's
  pre-compile wiring is all-empty) and read (`ui_models.py:484-489` carries disk rows forward).
  Both justifications hold.
- **D4's `wired_pass` measurements.** All three probed: the prefix kills the name rule
  (`u_bright`/`bloom_blur` → `None`), an undecided `u_paint` catches a host `paint`, and `u_prev`
  resolves to the renamed self.
- **The `PopupState` draw-call gate and the `set_pass` positional append.** 8 members / 7 calls as
  verification 14 states; five nine-argument positional `backend.set_pass` call sites and a
  positional-only protocol signature, so append-never-insert still holds.
- **`tiles_per_row`'s single caller.** `pass_list.py:175` is the only one, so D7's "genuinely
  untouched (one caller)" is checkable and true.
- **Verification 19's smoke plan.** The strip draws from `tabs/document.py:444`, which runs every
  frame regardless of popup state, and Radiance Cascades has 6 passes — so stamping a group on two
  non-adjacent members at the frame-42 multi-pass block does execute the outline, the label and
  the split-run path. Implementable as written.
- **`make gates`** — exit 0, unpiped, check + test + smoke all passed.
