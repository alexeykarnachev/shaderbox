# 094 — spec review log

The convergence loop for `01_spec.md`. One section per round: what each reviewer returned, what was
triaged as real, what was rejected and why. Kept so the loop survives a context reset.

---

## Round 1 — pre-implementation

Two reviewers, both opus, launched in parallel against the plan-draft.

- **R1 correctness & design** — anchored to the live tree, enumerating D1–D18, the Files-touched
  table, the 10 correctness checks and `conventions.md ## Design decisions`.
- **R2 blast radius & completeness** — anchored to the maintainer's verbatim requirements and to
  the deleted surfaces' full capability inventory.

### Findings the main agent established independently (before the reviewers returned)

- **Deleting persisted fields is load-safe by construction.** `model_salvage.drop_unknown`
  (`shaderbox/model_salvage.py:23`) prunes keys no field claims, logging a warning — so a removed
  field costs a log line, not a failed load. This is what makes D18 a deletion rather than a
  migration.
- **The dev sandbox carries exactly two of the deleted keys**, both in
  `projects/dev/app_state.json`: `is_render_all_documents` and `active_document_tab`. The two
  `documents/*/document.json` files carry none of `panel_pass` / `uniform_sort_*`. So the hand-fix
  the hard rule demands is two key deletions in one file, committed in the same wave.

### Verdicts

(filled when the round closes)

| Reviewer | Verdict | Most important gap |
|---|---|---|
| R1 | pending | |
| R2 | pending | |

### R1 (correctness & design) — PARTIAL

Verified independently by the main agent against the primary sources before triage; every
structural claim below was re-read in the file, not taken on report.

**Real, and structural:**

1. **`App.panel_pass` cannot be deleted as D4 assumes.** `draw_ui_uniform(app, ui_uniform)` takes no
   pass and resolves one internally (`widgets/uniform.py:221`), reading it at ~14 sites. It is also
   the resolver for `Open shader` (`app.py:660` → `open_shader_for_panel_pass`, `app.py:1181`) and
   `Pass settings` (`app.py:669` → `open_pass_settings_for_panel_pass`, `app.py:1176`), neither of
   which this feature deletes. And it is a LOCKED convention (`conventions.md:590`).
   **Remedy is the one that convention's own revisit clause names:** "Revisit if a surface needs a
   pass the user did NOT pick, which wants its own argument rather than a fourth tier." So
   `draw_ui_uniform` gains an explicit pass parameter, the two command resolvers take a pass, and
   only then does the state have no readers. This is the feature's real cost and the draft priced
   it as one table row.

2. **D12's per-pass Render/Share is not buildable as written.** Document-scoped by signature at
   three layers: `document.render_media(details, preset)` (`document.py:1193`),
   `render_job.render_for(document, ...)` (`render_job.py:67`), and the exporter ABC's
   `draw_target_panel(current_document, ...)` (`exporters/base.py:188`). Neither `document.py` nor
   `render_job.py` is in the Files-touched table.

3. **Correctness item 7 cannot enumerate its domain.** `valid_input_types()` is an INSTANCE method
   returning the types valid for one uniform — at most two (`ui_models.py:97`). The domain is
   `get_args(UIUniformInputType)`, which has SEVEN members (`ui_models.py:51`). The draft also said
   "eight", counting a driven uniform, which is a state and not an input type. A test written to
   the draft's letter would cover two of seven — the exact failure the item cites itself as
   guarding against.

4. **`scripts/smoke.py` breaks and is absent from the table.** It iterates `DocumentTab`
   (`smoke.py:134`), asserts `active_document_tab` every frame (`:154`), asserts the focused tab
   drew (`:356`) and calls `open_graph_for` (`:285`). It is part of `make gates`.

5. **`tests/test_region_system_is_gone.py` is built on the deleted path** (`_NODE_TABS` → the tabs),
   and `tests/test_pass_strip_layout.py:10` imports from the deleted `pass_list`. Nine further test
   files read deleted symbols.

6. **`is_render_all_documents` has a reader D15 deletes but D16 still needs** — the grid tile's
   `stale` flag (`document_grid.py:116`). The dropdown's live thumbnails need that same staleness
   signal for a document awaiting its turn.

7. **`SIZE.PANEL_CTRL_MINH` is viewer geometry, not a panel minimum** (`ui.py:964` → `:830`). It
   caps the rendering canvas's height. NOTE: this finding is against the pre-D1a draft; D1a now
   replaces that derivation with a dragged fraction, which resolves it differently than R1 assumed
   — the token does become dead, but because the splitter supersedes it, not because the panel left.

8. **`menus.py` needs no change** — the bar iterates `COMMAND_SPECS` (`menus.py:99`), so deleting
   the command entries suffices. The table row is wrong.

9. **`pass_graph.draw` wraps the WHOLE widget including `_tab_row` in
   `begin_disabled(copilot_turn_active)`** (`pass_graph.py:908`). Hosted permanently, that disables
   the documents dropdown for every copilot turn — and D16 makes it the only way to switch
   documents. The grid it replaces brackets more narrowly (`document_grid.py:78`).

10. **Correctness items 1 and 9 can pass while broken.** Item 1's grep cannot be evaluated
    mechanically (doc hits persist); item 9's assertion passes trivially whenever no shader tab is
    open, since `editor_focused` is written False on that path too (`tabs/code.py:909`).

11. **`pass_menu_items`' home is a locked convention** (`conventions.md:1066`) naming
    `pass_list.pass_menu_items` by path; D17 does not say where it lands, and the convention must be
    updated in the same wave.

**False trails recorded (do not re-litigate):** persistence is safe under `drop_unknown`; the
copilot is genuinely clean of every deleted symbol; the draw channels exist exactly as D10 names
them; `COLOR.GRAPH_DIM_ALPHA` exists (though today it fades `GRAPH_EDGE` rather than filling a
scrim); the `frozen` gesture-refusal precedent is real; `_canvas_menu` does carry Add/Import; the
two width numbers are right; `pass_graph.draw`'s contract claim is accurate; `canvas_choice_*` is
cleanly liftable; `GraphViewState` is a plain per-document dataclass.

### R2 (blast radius & completeness) — PARTIAL

Anchored to the maintainer's verbatim requirements. Converged independently with R1 on the same
root cause, and found four things R1 did not.

**Real, and new (not in R1):**

12. **Esc will NOT close the focus mode.** `App.escape_has_job` (`app.py:616`) returns True only for
    an open popup, the palette, a focused editor or a focused chat; `app.py:605` swallows the key at
    the GLFW layer when it returns False, so imgui never sees the press. A focused node is none of
    the four. The maintainer named Esc as the exit. One-line fix (a fifth clause plus a branch in
    `hotkeys._handle_escape`, in precedence order), but nothing would have found it until use.

13. **The uniform-name jump/hover bridge dies with `panel_pass`, unnamed.** `uniform.py:82-100`
    `_locate_uniform_declaration` — clicking a uniform's name jumps to its declaration, hovering
    marks the line in the editor. It walks `app.panel_pass(...).compile_unit.sources`. The draft's
    node-row table describes rows as "the value control alone" and never says this is preserved or
    deleted, against the spec's own success measure.

14. **The auto-uniform block has no home.** `tabs/uniforms.py:17` `_draw_auto_block` shows every
    engine-driven uniform with its LIVE value, and those hashes are routed AROUND `draw_ui_uniform`
    (`uniforms.py:70-83`). So "reuse the tab's body" does not carry it. The draft also claimed auto
    values are "identical on every pass" — false: `format_auto_value` reads the PASS's own slot
    (`uniforms.py:32`).

15. **"Declaration order" is not free.** `get_active_uniforms()` yields GL's driver-defined order;
    `sort_uniform_hashes(..., "code", ...)` is what PRODUCES declaration order
    (`tabs/uniforms.py:106`). Deleting the sort state without keeping that call leaves rows in an
    order that reshuffles between recompiles.

16. **Per-pass Share has no state key.** `TabState.outlets` is keyed by exporter_id alone
    (`share_state.py:237`), one `OutletRenderState` for the whole app. Focus Share on node A, render,
    focus node B, Publish — and A's artifact publishes while `artifact_is_fresh` reads True.

17. **A deferred render captures the document but not the pass.** `share.py:127` captures
    `current_document_id` precisely because a delete can release the GL program between submit and
    run. Per-pass render adds a second thing that can vanish, and the copilot can delete passes.

18. **No revalidation for the focused pass.** `GraphViewState` has the idiom
    (`pass_graph.py:906` `view.selection &= set(document.passes)`), and `app.py:708` `_on_pass_renamed`
    rewrites editor tabs but touches no graph view. Deleting or renaming the focused pass strands
    the focus: a scrim over an empty canvas, with no working Esc (finding 12).

19. **`dl.channels_split(5)` is a hard-coded count** (`pass_graph.py:1002`). Mechanical, but every
    `channels_set_current` moves and an off-by-one paints the scrim OVER the focused node.

20. **The fps chip is a divergence from the maintainer's words, not a satisfaction.** He said "merge
    everything under a single toolbelt icon"; D13 keeps fps as a separate chip. The spec's reason is
    sound but it must be PUT TO HIM as a divergence.

21. **Alt+G's per-document use is lost.** `document_grid.py` uses `OPEN_GRAPH` for ANOTHER document's
    menu ("open that document's graph"). The dropdown selects a document; it does not offer that.

22. **Check #5 does not reach imgui widgets.** Refusing canvas GESTURES does not stop the uniform
    drags on an unfocused node behind the scrim — those are imgui items, not `invisible_button`
    hit-tests.

23. **`menus.menu_enabled` regression** (`menus.py:37`, `return app.active_tab is not None`): a graph
    tab used to satisfy it, so with no shader open every EDITOR-scoped menu item now greys.

**REJECTED — verified false by the main agent:**

- **R2's finding 13 claim that `extra: "forbid"` fails the whole `app_state.json`.** `load_model`
  calls `drop_unknown` BEFORE validating (`model_salvage.py:115`), so retired keys are pruned and
  `forbid` never sees them. R1's false-trail note and the main agent's own pre-review check both had
  this right. The hand-edit of `projects/dev/app_state.json` is still required by the repo's hard
  rule, but it is hygiene, not a load-breaking bug.

**Accepted as a framing correction, not a defect:**

- **R2's finding 14, "unrequested scope".** The LOD ladder is the spec's invention, and the third
  (`Uniforms…`) focus mode is too — the maintainer listed uniforms among the buttons no longer
  needed. But findings 1-4 make the uniforms mode LOAD-BEARING: it is the only home for the texture
  combos, the multiline text box, `Randomize`, the input-type chip and the jump/hover bridge. It
  must be presented to him as "this is where the full rows had to go", never as a third peer mode
  he asked for.

---

## Round 2 — four reviewers, all opus

Launched against the round-1-patched spec. Verdicts: all four PARTIAL.

### The failure round 2 existed to catch

**R2a and R2c independently found that round 1's two structural fixes never landed.** The patch that
was to write D4a and D12a reported success and did not: five references pointed at decision bodies
that did not exist, D4 still said "delete `panel_pass`" while D18 said "keep it", and the
Files-touched table said both. R1's findings 1 and 2 — the ones the log itself called "the feature's
real cost" — were therefore OPEN, recorded as closed. Both bodies are now written.

The lesson is about the loop, not the spec: a finding recorded as accepted is not a finding fixed,
and only re-reading the artifact proves which. Round 2's value was almost entirely this.

### New findings, triaged real

24. **`Document.render` ALREADY takes `target: str | None`** (documented for exactly this), so
    per-pass render is a parameter threaded through four signatures — NOT a temporary output switch.
    R2a demonstrated the switch is unsound three ways: things read the output mid-encode
    (`restart_video_uniforms`, the scratch's dtype/filter/wrap), `set_output_pass` resizes by design
    (`conform_canvases` + resample, twice per export, destroying the off-output pass's scale), and it
    collides with the canvas-ownership decision. D12 rewritten; the rejected approach is recorded in
    it so it is not re-proposed.

25. **D16's live previews are unimplementable as written.** `_tick_frame_state` — the whole render-set
    computation — runs before `imgui.new_frame()`, so nothing in the planning path can read an imgui
    popup's open state. The spec cited the Examples popup as precedent and misread why it works: it
    keys off `app.modal`, an App field set outside the frame. Needs an App-side latch, one frame
    stale in both directions.

26. **`tick_documents` and `planned_documents` are different lists with different consumers.** D15's
    sentence did not say which the dropdown joins. Joining only `planned_documents` gives those
    documents an interval and a `begin_frame` advance with no render — a feedback document would
    advance history without drawing into it.

27. **The uniform rows need an explicit place in the `set_next_item_allow_overlap` chain**, and every
    row must declare it or the ports beneath become dead — and the ports are every wire's drop target.

28. **Rows change `node_size`**, which is the single source of truth for layout, `arrange_graph`,
    wire endpoints and hit rects — and `_port_point` / `_out_point` re-derive independently rather
    than reading `node.size`. Under the LOD ladder this makes the canvas footprint zoom-dependent,
    which contradicts D7's own principle applied to ordinary nodes.

29. **`_fit` clamps `zoom = min(1.0, ...)` — it never zooms in.** So the "near" LOD where rows live is
    unreachable by `Frame all` or any scope change; only deliberate wheeling gets there. With
    `GRAPH_NODE_W` 136 and `ZOOM_MAX` 2.5, max node width is 340px against `UNIFORM_CTRL_W` 320.

30. **The wheel is unconditionally the canvas zoom** and the canvas child sets `no_scroll_with_mouse`,
    so D6's row scrolling has no wheel to scroll with.

31. **Entering focus mid-drag** leaves `view.node_drag` live; the release writes a pass position from
    a drag the user cannot see behind the scrim.

32. **The channel split's write order is not its channel order** — `_CH_INFLIGHT` and `_CH_OVERLAY`
    are written after the node loop, so the scrim's insertion must re-place both.

33. **`tabs/render.py` has no try/except** (unlike `tabs/share.py`), and hosting it inside the split
    draw list means an exception unwinds past `channels_merge()` and `end_child()`.

34. **`pfd_block` is a main-thread spin loop** called from the Render body — inside the channel split
    once hosted on a node.

35. **The fps chip would show the TARGET, not the measured rate.** `document_fps` holds
    `target_fps / interval`; at interval 1 that is the setting. Today's two-number label hid this by
    falling back to `app.global_fps`. Fixed: interval 1 reads `app.global_fps`.

36. **A third `has_script` gate**: `App.toggle_current_document_play` refuses when absent, making
    `TOGGLE_DOCUMENT_PLAY` a silent no-op — the third state D14a says does not exist.

37. **`_locate_uniform_declaration` and `uniform_name_label` need the pass argument too**, or the
    preserved jump bridge resolves against a different pass than the rows it sits on.

38. **A THIRD convention needs amending** — the viewer-box decision, whose height derivation,
    `PANEL_CTRL_MINH` cap and "only the splitter moves that boundary" clause D1a all replace.

39. **29 raw line-number citations** violate the code rule that specs cite symbols, not lines. All
    stripped.

40. **`_reset_out_of_range_values`'s `uniform_sort_key` clause** and possibly `UniformSortKey` go
    dead with the field — a second site the deletion strands.

41. **`menus.py` has no Files-touched row** though check 15 names a regression whose fix lands there.

42. Stale "eight input types" in the prose against check 7's corrected seven; two items numbered 12.

### Maintainer decision recorded mid-round

- **The uniform sort control returns as a small icon on the node's card** (his call), rather than
  being deleted with the Uniforms tab. Open question 4 is answered.

---

## Round 3 — four reviewers, all opus

Closure audit, implementation readiness, adversarial fresh read, test design. The readiness reviewer
returned **FAIL** on two blocking findings; the others PARTIAL.

### The blocking finding

43. **D12 as written produces BLACK exports.** `Document.render`'s blit is gated on
    `name == output` — the GRAPH OUTPUT — not on `resolved`. So `render(canvas=scratch,
    target="some_pass")` draws that pass's chain into the passes' own canvases and blits nothing
    into the export scratch. Round 2's finding 24 established that `target` exists and is documented
    for this; it did not check that `target` and `canvas` COMPOSE. They do not.

    The fix is one condition — `name == resolved` — but that is an edit to the core render loop,
    which `document.py`'s Files-touched row does not cover. **Verified by the main agent against
    `Document.render`.**

    Second, independent: `render_media` sizes its scratch from `self.render_pass.canvas`'s
    dtype/filter/wrap and `export_source_size()`, both OUTPUT properties. A non-output pass renders
    at `canvas_size_for(name)` — generally smaller. The spec does not say which size the export
    takes.

44. **The uniform rows are specified as real imgui widgets submitted inside a live
    `channels_split`, and the spec never says which channel is current.** Real widgets emit into
    `get_window_draw_list()` — the same `dl` that is split — so their output lands in whatever
    channel is current and is re-ordered by the merge. Torn widgets, and the symptom is visual,
    which an unattended agent cannot see.

### Also real, from the same reviewer

45. **D4b's overlap reasoning is backwards.** The ports deliberately declare NOTHING, and the code
    comment says why ("the ports must keep declaring nothing or the drop target dies"). A row
    submitted before a port and declaring overlap means the PORT wins — so the row's drag is dead
    wherever a port overlaps it. Whether rows must be laid out clear of ports is undecided.

46. **D4c (the sort icon) contradicts D18**, which still deletes `uniform_sort_key`/`_desc`. A
    control cycling those values needs somewhere to store them; `GraphViewState` is explicitly
    transient. Three incompatible readings.

47. **`_tab_row` cannot host what D14/D14b/D16 hang off it.** `text_tab_row(id_, names, active)`
    takes flat label strings and returns the clicked one — no caret, no right-click return, no way
    to distinguish the first crumb, no slot for a sibling chip, no tightness measurement.

48. **`_draw_auto_block` also resolves through `panel_pass`** and is not in D4a's list of functions
    gaining the parameter. Hosted on a focused node it would show the PANEL pass's auto values.

49. **Nobody builds `ui_uniforms`.** `tabs/uniforms.py`'s loop is the only site that calls
    `UIUniform.from_uniform` and `snap_input_type()` — two writes to persisted document state. Delete
    the tab and a freshly-compiled pass has no rows at all.

50. **`ui_primitives.py` has no Files-touched row** though it hosts the single largest deletion.

51. **Three unenumerated readers**: `tests/test_model_salvage.py` (its retired-enum fixture),
    `tests/test_render_decoupling_loop.py`, and `App`'s project-switch restore of
    `active_document_tab`.

52. **Three ordering constraints** the spec states as hazards but not as sequence: D4a before D4;
    D2a before D3; D16 before D15.

53. **D6's "ceiling" contradicts D4b.** If rows draw outside `node_size`, the node box never grows,
    so the wall problem D6 solves does not exist — or rows do enter the box, contradicting D4b.

54. **The focused-node size is one token for three very different bodies**, presented as decided.

### From the test-design reviewer

55. **D4a — the feature's largest change — had NO check.** Its failure mode (a node silently writing
    another pass's values) is invisible on screen and in the gates. Added as check 22.

56. **Eight of 21 falsifiers did not go red**, four because the check named a value nothing can read.
    Fixed by publishing four observation seams (D19) and rewriting checks 3, 5b, 12, 14, 16.

57. **Check 20 had no assert** (it described work, not a check) and check 8's restore half guarded a
    mechanism D12 forbids. Both rewritten.

58. Checks added for D4b, D11a, D14a, D16a (23-26); check 19's falsifier narrowed from "delete
    profiling.py" to dropping ONE `profiler=` argument, per the mutation-fidelity rule.

### The pattern, named

Round 2 diagnosed "a finding recorded as accepted is not a finding fixed". Round 3 found the same
failure one level down: **a finding's downstream consequences were not re-derived.** Finding 24
established `target` exists without checking it composes with `canvas`; finding 27 established the
rows need the overlap chain without establishing they need a draw CHANNEL; the maintainer's sort
decision was recorded without reconciling D18. The lesson for round 4: when a decision is rewritten,
re-read every decision and check that cites it.

### Round 3 triage — decisions taken by the main agent

Every finding verified against the primary source before acting. Three were design calls, taken
rather than escalated:

- **43 (black exports): FIXED.** `render()`'s blit condition becomes `name == resolved`; `document.py`
  joins the Files-touched table for the render loop, not just `render_media`. The export reads
  `canvas_size_for(target)` and the target pass's own canvas properties, since `canvas_size_for` is
  the documented single home for that rule.
- **The 136px node: DECIDED — `GRAPH_NODE_W` 136 → 240 (D4e).** At 136 a vec4 got 22px per component
  against ~38px of text; at 240 it gets 46px. `GRAPH_THUMB` follows to 220 to keep the theme's
  width/inset assertion true. This also answers accepted-cost (1): a 220px node picture against the
  strip's 168px tile means comparing two passes is no longer a downgrade. The reviewer's alternative
  (read-only rows, edit only in the focused mode) was rejected — the maintainer asked for uniforms on
  the node, and the arithmetic problem was the node's size, not the idea.
- **44 (torn widgets): FIXED.** The rows are submitted AFTER `channels_merge()`, with the existing
  hit-test buttons, where every other imgui item on this canvas already lives. Only the pictures stay
  inside the split.
- **45 (overlap backwards): FIXED.** The rows are laid out CLEAR of the port column rather than
  relying on the chain — the ports must keep declaring nothing, so a row that overlapped one would
  lose its own drag there.
- **D2a (nested `begin_disabled`): FIXED.** Verified against imgui's own binding comment. Both
  brackets move; `_draw_app_panel` brackets per region.
- **New, from the fresh read: `get_uniform_hash` has no pass in the key (D4d).** Two passes declaring
  the same uniform share one `UIUniform`, and `input_type` is user config. The values were namespaced
  by pass long ago for exactly this reason; the UI config was not, because nothing drew two passes at
  once. Hand-fix the dev sandbox's `ui_uniforms` keys — a fourth file.
- **The splitter's damping (D1a): FIXED.** `_past_dead_band` applies immediately past 5%, so only
  noisy change is damped, not continuous. The applied size is latched to the drag and applied on
  release.
- **The graph cropping itself (D1c): FIXED.** `fitted` is a one-shot; a dragged region needs it
  cleared when `avail` moves.
- **Esc vs imgui popups (D8a): FIXED.** `any_popup_open` knows only about modals, so a focused node
  under an open imgui popup would have one Esc dismiss both. The focus branch gates on
  `not imgui.is_any_popup_open()`.
- **49 (`ui_uniforms` lifecycle): FIXED as D10a.** The deleted tab's loop is the only site that
  creates `UIUniform` rows; it moves to a shared helper both row paths call.
- **52 (ordering): FIXED as D20**, stated as sequence rather than as hazards.
- **Accepted costs recorded** for the three things the strip made easy and the graph makes merely
  possible (comparing pictures, the click-vs-drag output switch, reading inputs as names), each with
  a trigger.

**Rejected:** the reviewer's proposal to make the focused-node mode a real modal. The machinery it
would delete is real, but the maintainer's stated shape is "a contextual modal without the actual
modal", and D9's camera write plus D10's scrim are what deliver it. Recorded so it is not
re-proposed.


---

## Implementation notes (things found while building, not in any review round)

- **C8a: the thumb is SQUARE, so widening the card doubled its height.** `_thumb_rect` makes the
  picture `GRAPH_THUMB` on both axes and letterboxes the document inside it, which was harmless at
  116px and costly at 220: the card went 150 -> 283px tall, and that height is what drives the
  fitted zoom (a 6-pass chain fits at 0.50). A 16:9 thumb would bring the card to 187px and is the
  obvious follow-up, but it changes what the node LOOKS like beyond what the spec settled, and not
  every document is 16:9. **Left as-is and flagged for the maintainer**, because the LOD thresholds
  are keyed on screen WIDTH, so the tall card costs vertical space rather than hiding the rows.
  *Trigger:* he reports the graph feels vertically cramped, or a document set that is mostly wide.

- **D4b holds under measurement:** rows draw outside `node_size`, so a pass gaining uniforms never
  changes the layout box and never shrinks the fit. Checked at 0, 3 and 6 rows.
