# 071 — the second walk: findings into a plan

The maintainer walked the 068 tutorial again after 069 landed and filed 11 findings and 6
decisions (`00_findings.md`, each finding verified against the code before filing). This spec
groups them into seven waves, fixes the order, and carries the decisions as constraints. Where a
finding needs a redesign, the redesign is the fix (069's rule, unchanged).

Size: **high-blast-radius** — it changes the editor library and re-vendors it, renames a command,
adds a per-document clock, sweeps every prose surface for one spelling, and rewrites part of the
tutorial. Seven waves in the order of § Order, each its own commit series with its own pre/post
review per `dev_flow.md`. One spec, because the tutorial wave is written against three of the
others.

Two repos: W-A lands in the editor repo (`~/src/editor`, `alexeykarnachev/editor`) under that
repo's own flow (`CLAUDE.md` there: a `docs/features/NNN_*/feature.md`, the test corpus,
`make ffi`), then re-vendors into this one per `conventions.md ## Known quirks` (rebuild from a
committed sha, copy the vendored set, delete the mitigations the new sha makes dead).

---

## Goal

After this feature: the editor deletes, shifts and searches the way vim does for the cases the
walk hit, and shows what it is searching for; the code panel never hides the caret and cycles
tabs in the order the eye sees; a document has one Reset that restarts it whole; the uniforms
panel shows what a sampler really reads; every dormant tile is legible; every user-facing word is
spelled one way; and the tutorial tells the cascade story without interruption, then teaches
drawing properly at the end.

## Out of scope (each with a trigger)

- **The graph view of the pass strip.** Feature 070, direction fixed in `069/01_spec.md § Out of
  scope`. D2 here keeps the linear strip as the default view and makes it legible on its own.
  Trigger: unchanged — 070 opens after this feature.
- **A downstream-aware Clear.** #7's open point (show a clear on a pass whose upstream
  accumulates) is moot under D4/D5: Reset is document-wide and always visible.
- **`g*` / `g#` (substring search under the cursor).** W-A ships `*` / `#` whole-word. Trigger:
  the maintainer reaches for the substring form.
- **Windows `libeditor.dll`.** Still owed from 067; W-A's re-vendor is the Linux `.so` only.
  Trigger: next `/ship` on a Windows host.
- **The shipped example gaining a script.** D1: the example ships as it is (no script, no
  feedback wire on `paint`); the drawing chapter is paste-on-top. Trigger: a second mouse-driven
  example (069 W-G's trigger for a builtin `u_mouse`, unchanged).

## Locked decisions (from the walk — constraints, not options)

- **D1. The mouse leaves the cascade narrative; a full drawing chapter closes the tutorial.**
  Steps 1-6 are the six shipped passes and nothing else. A final chapter after "What you built"
  teaches the whole mechanism: the document script and `ctx.mouse`, the pass block by name,
  `paint` reading itself so the texture is the memory, the capsule stamp, the re-entry reset,
  Reset (D4/D5) as the wipe, and why the shipped example ships without a script.
- **D2. The linear strip stays the default view and must read on its own.** Dormant tiles are
  drawn darkened (`COLOR.STALE_TINT`, multiplicative), never washed grey. Landed during the walk;
  the value is tuned by eye in W-D's manual check.
- **D3. American spelling: `color`, never `colour`**, on every surface a user or the copilot
  reads, with a gate. Identifiers stay as they are (they are already `color`).
- **D4 + D5. One document-wide Reset, always visible, meaning "as if just opened": feedback
  histories, the document clock, the script instance, and bound videos.** Replaces the
  feedback-gated Clear. Maintainer's constraint: the machinery is encapsulated — one funnel, no
  reset logic scattered across the engine. The clock is the one primitive that makes that
  possible: a bound video is already a function of `u_time` (`core.py:400` calls
  `MediaWithTexture.update(render_time)` each draw), so rewinding the document clock rewinds
  every video with no video-specific code.
- **D6. The "Verifying it" section leaves the tutorial.** The per-step reference image is the
  reader's check; `oracle.py` stays as the repo's gate and the tutorial stops mentioning it.
- **D7 (from #8). Say what the thing is; the excluded thing gets its own sentence only when the
  reader needs it.** The "X, not Y" / "rather than" construction is out of the tutorial and the
  help, with a gate.
- **D8 (from #9). Ctrl+Tab is global; the first press on an unfocused editor only focuses it;
  every press while focused cycles, in display order.**
- **D9 (from #10). A sampler row shows what the sampler reads.** A pass source shows that pass's
  live thumbnail and name, no Load; an unresolved source shows black, captioned unwired; only a
  sampler with no pass source keeps the media row.
- **D10 (from #11). Search highlight is on by default**, follows the `/` line as it is typed,
  persists after the jump, and clears on normal-mode Esc with the pattern kept.

## Workstreams

### W-A — Editor library: last line, shift, star, search highlight (#1 #2 #6 #11 + D10)

In the editor repo, one feature dir `docs/features/NNN_walk_two/feature.md`, four items, each
with corpus cases before the code:

1. **`dd` / `Vd` on an empty last line** (#1). `src/keymap.odin::apply_operator` returns at its
   `start >= end` guard before its own last-line rule. Fix: a linewise DELETE whose range is
   empty at the buffer end and has a line before it takes the preceding newline — the rule moves
   above the guard, or the guard exempts that case. Corpus: `jdd` and `jVd` on `"a\n"` give
   `"a"`, cursor (0, 0); `dd` on `""` stays `""`; `jcc` on `"a\n"` keeps the empty line (change
   is unchanged).
2. **`>>` / `<<`** (#2). `INDENT` (`src/keymap.odin`) is the step. `>>` prepends it to each line
   of the range, `<<` strips up to one `INDENT` of leading spaces; count applies (`3>>`); both are
   operators over a motion (`>j`, `>ip`) and over a visual selection (charwise and linewise both
   shift whole lines); the cursor lands on the first non-blank of the first line; one undo step;
   the register is untouched. Tick both `docs/vim_coverage.md` boxes.
3. **`*` / `#`** (#6). The word under the cursor via the `iw` object (a cursor on a non-word byte
   takes the next keyword on the line, as vim); the pattern is stored so `n` / `N` repeat it in
   `*`'s direction; wraps; "pattern not found" on a line with no keyword. Whole-word: a
   `whole_word` flag on `Search`, honoured by `matches_at` (both neighbours non-word bytes), set
   by `*` / `#`, cleared by `/` / `?`. Tick the box.
4. **Search highlight** (#11, D10). A `Search_Match` primitive kind and a `SEARCH_MATCH` theme
   slot, emitted over every visible hit of the stored pattern while a `hlsearch` view flag is on
   (default on); while the `/` or `?` line is open the hits follow the text typed so far;
   normal-mode `Esc` clears the highlight and keeps the pattern; the next `/`, `n`, `*` brings it
   back. Corpus through the primitive list, the way `Bracket_Match` is tested.

The editor side is done by the editor session (`editor-d2`), which received the four items
as one request from this session on 2026-09-03 and pings back with the sha when the corpus is
green. Then the re-vendor into shaderbox: rebuild from the committed sha, copy the vendored set,
`VERSION`, `nm -D` diff recorded in the wave file, and the host draws the new primitive kind in
`editor/render.py` in the same pass as `Bracket_Match`, with one colour token in `theme.py` fed
through the slot setter. `tests/test_editor_ffi.py` gains one case per item, through
`Editor.feed`, so the vendored binary is pinned here too.

Files: editor repo `src/keymap.odin`, `src/keymap_normal.odin`, `src/keymap_visual.odin`,
`src/operator.odin`, `src/search.odin`, `src/view*.odin`, `ffi/ffi.odin`, `ffi/README.md`,
`docs/vim_coverage.md`, the corpus; shaderbox `shaderbox/resources/editor/*`,
`shaderbox/editor/ffi.py`, `shaderbox/editor/render.py`, `shaderbox/theme.py`,
`tests/test_editor_ffi.py`.

### W-B — Code panel: cursor follow, Ctrl+Tab (#3 #9 + D8)

1. **Cursor follow after an edit at the bottom** (#3). `tabs/code.py` follows the cursor before
   this frame's layout, through `scroll_to_line`, which the library answers against the previous
   layout. Fix: lay out first, then follow against the fresh layout, and lay out again only when
   the follow changed the scroll (the second `ed_layout` is microseconds; it runs only on the
   frames a follow fires). Test: the headless sequence from the ledger (ten lines, five rows, `G`,
   `o`) ends with the cursor row inside `[first, first + rows)`; the same for Enter at the end of
   the last line and `p` of a linewise register there.
2. **Ctrl+Tab in display order** (#9a). Inside `_draw_tab_row`, after the item loop, read
   imgui's order back (`imgui.internal.get_current_tab_bar`, `tab_bar_find_tab_by_order`,
   `tab_bar_get_tab_name`; landed with the tab's PATH as the id suffix, since an index-keyed id
   made every moved tab a new tab to imgui) and permute
   `app.editor_tabs` + `active_tab_index` to match, so the model order IS the display order and
   every index-based verb agrees with the eye. Test: a permuted read-back reorders the list and
   keeps the active tab's identity.
3. **Ctrl+Tab focus** (#9b, D8). **Repro first.** The maintainer's case: the editor unfocused,
   working in the Document tab (the `app_panel` child, a SIBLING of `code_editor` in `ui.py`),
   Ctrl+Tab switched the code tab. By the code that cannot happen: `focused` is
   `is_window_focused(child_windows)` evaluated inside `code_editor` (`code.py:562`), so the
   Document tab's focus makes it False, and `spec_eligible` refuses an EDITOR chord on that.
   So a gate is bypassed somewhere, and the wave starts by finding it: a routing test that
   sets `editor_focused = False` and dispatches the Ctrl+Tab chord (if it dispatches, the gate
   is broken); if it does not, the switch came from another path — imgui's own Ctrl+Tab
   windowing (`NavUpdateWindowing`, check whether the `no_nav_inputs` flag and the off keyboard
   nav really keep it out at this imgui version), or a stale `editor_focused` written by a
   frame the panel did not draw. The fix follows the repro. Then D8: `CYCLE_CODE_TAB` leaves
   `CommandScope.EDITOR` for a scope the editor gate does not refuse; `App.cycle_code_tab`
   becomes: if the editor does not hold focus, `editor_focus_requested = True` and return; else
   cycle. "Focus" is the code-panel child focus, which is what already routes keys; the
   unfocused dimming (`EDITOR_UNFOCUSED_ALPHA`) tracks exactly that flag.

Files: `shaderbox/tabs/code.py`, `shaderbox/app.py`, `shaderbox/commands.py`,
`shaderbox/hotkeys.py`, `tests/test_editor_panel*.py` / `tests/test_command_routing.py`.

### W-C — Document Reset (#7 + D4 D5)

- `CommandId.RESET_FEEDBACK` → `RESET_DOCUMENT`, label "Reset document", chord F6 unchanged,
  scope DOCUMENT. `key_bindings` persists by `CommandId` name; if a `projects/dev` app state holds
  an override under the old name, hand-fix it in the same wave (no migration code).
- **One funnel: `ProjectSession.reset_document(document_id, now)`.** It calls
  `Document.reset(now)` (feedback histories dropped, `time_origin = now`; the videos follow the
  clock) and `ScriptEngine.reinstantiate(document_id)`. `App.reset_current_document` forwards
  to it; the command, the button and any copilot tool call the App method and nothing else.
  Nothing outside `document.py` / `engine.py` knows what a reset consists of.
- **Document clock.** `Document.time_origin: float` (0 on load). The live loop passes
  `u_time = now - origin` to `render` and `t = now - origin` to `session.tick` for that document
  (today both are app-global; `core.py:384` and `ui.py:208`). Export and the probe pass their
  own `u_time` and are untouched. `dt` is unaffected.
- **Script re-instantiate.** `ScriptEngine.reinstantiate(document_id)`: drop the behavior the
  `_drop_script` way and recompile from the unchanged source (`scripts.source`), so `self.*`
  restarts; errors re-derived; the mouse `prev` restarts by the re-entry rule on its own.
- The button over the viewer is always drawn (the `has_feedback` gate goes), reads "Reset",
  tooltip "Reset document  F6".
- Tests: after Reset, `u_time` seen by the pass is < the frame's dt; the script's `update` sees
  `ctx.t` restart at 0 and a fresh instance (a counter in `self` reads 1); feedback histories are
  empty; export's `u_time` is unchanged by a prior live Reset.

Files: `shaderbox/commands.py`, `shaderbox/app.py`, `shaderbox/ui.py`, `shaderbox/document.py`,
`shaderbox/project_session.py`, `shaderbox/scripting/engine.py`, `shaderbox/help_content.py`
(the shortcuts table is generated), tests.

### W-D — Uniforms panel sampler rows, dormant tiles (#10 #5 + D2 D9)

- **Sampler rows** (#10, D9). `tabs/document.py` resolves the panel pass's effective inputs once
  per draw (`effective_inputs` on the pass's entry, the same call the gear makes) and hands
  `widgets/uniform.py::draw_ui_uniform` a `source: str | None` per sampler. The texture branch:
  source is a pass → that pass's live canvas thumbnail (`preview_cell`'s blit, `SIZE.THUMB_SM`
  high) captioned with the pass name, no Load, no resolution line; source is `""` / a stale name →
  a black swatch captioned "unwired"; no source → today's media row. Test: for the shipped
  example's `composite`, both sampler rows report their pass source and no Load is drawn; a
  headless probe of the row model, not pixels.
- **Dormant tiles** (#5, D2). Already landed (`COLOR.STALE_TINT = 0.45`). This wave only tunes
  the value from the maintainer's live look, in the manual check.

Files: `shaderbox/tabs/document.py`, `shaderbox/widgets/uniform.py`, `shaderbox/theme.py`,
`tests/test_uniform_panel*.py`.

### W-E — Spelling sweep and gate (D3)

- One sweep, `colour` → `color` (case-preserving), over: `shaderbox/` (prose in docstrings,
  comments, `help_content.py`, `resources/document_examples/**`, `resources/editor/abi_probe.py`),
  `tests/` (including the one test name), `ai_docs/` (all), `.claude/skills/imgui-ui`,
  `.claude/skills/shader-lab`, `README.md`, the tutorial body. Dogfood run transcripts under
  `scripts/dogfood/runs/` are records and stay.
- The rule in `conventions.md ## Code rules`: "American spelling in every string, comment and
  doc: `color`." Positive form, one line.
- The gate: `tests/test_prose_spelling.py` greps the same roster for the British form and fails
  on any hit; shipped in the same commit as the sweep.

Files: many, mechanical; `ai_docs/conventions.md`; one new test.

### W-F — Tutorial (#4 #8 + D1 D6 D7)

In `tutorial_body.html`, then `build_tutorial.py` regenerates:

1. **Cut** (D1, D6): the intro's mouse clause; the whole "Paint it with the mouse" subsection;
   the "Things to try" `u_prev` item; the "Verifying it" section, its contents entry, the intro's
   `oracle.py` clause.
2. **Rewrite** (D7): every "X, not Y" / "rather than" site in the body (`00_findings.md` #8 lists
   them; the step-2 heading and its contents echo included) as a statement, with the excluded
   thing in its own sentence only where the reader needs it.
3. **Add** (D1): the final chapter "Draw into it", after "What you built". It builds on the
   shipped example with paste-on-top steps: the three uniforms and the capsule stamp in
   `paint.frag.glsl`; wiring `u_prev` to `paint` in the gear (or by name, 069 D9) and reading it
   so strokes persist; the document script with `ctx.mouse` and the pass block; Reset (F6) to
   wipe; the re-entry rule; why the shipped example ships without a script (deterministic export,
   `EXPORT_MOUSE`). Each step self-sufficient (069 #8's lesson), every in-page string under the
   word bar.
4. **Gate** (D7): `build_tutorial.py` (or the existing tutorial test) counts `, not `, `-- not`,
   `— not` and `rather than` in the body and fails above an allowance of 2; `help_content.py`
   is counted too. Shipped with the rewrite.

Files: `ai_docs/features/068_radiance_cascades/tutorial_body.html`, `build_tutorial.py`,
`tutorial.html` (generated), `tests/test_radiance_cascades_example.py` or the tutorial test.

### W-G — Sanitize

The `/sanitize` sweep: roadmap row + banner, conventions entries (the spelling rule lands in W-E;
the re-vendor entry gets W-A's deletions; the Reset funnel as a design decision), cold-context
check, gates green under a display.

## Order

Forced: W-C before W-F (the chapter names Reset); W-E before W-F (the chapter is written under
D3, and the tutorial's own `colour`s are swept once, in W-E); W-A's editor work before its
re-vendor (obviously) and independent of everything else.

1. **W-C** Reset — small, self-contained, defines what the tutorial chapter names.
2. **W-B** Code panel — small host-side fixes; needs the answer to open question 1.
3. **W-D** Panel + tiles — UI, independent.
4. **W-E** Spelling — mechanical, with its gate; before the tutorial wave.
5. **W-A** Editor library + re-vendor — the editor session owns the library side; this
   session re-vendors when it pings back, so the wave floats: it lands whenever the sha arrives,
   between whichever waves are then in flight.
6. **W-F** Tutorial — last among the changes; written against W-C, W-E; the maintainer walks the
   new chapter.
7. **W-G** Sanitize.

Each wave: pre-impl review 1 (correctness & design vs this spec); post-impl review 2 (code
correctness; spec fidelity). W-A and W-F add a third: for W-A a corpus auditor in the editor repo
(does each case fail under the old binary), for W-F a prose auditor anchored to the maintainer's
verbatim words in `00_findings.md`.

## Files touched

Per workstream. Cross-cutting: `conventions.md` (spelling rule, Reset funnel, re-vendor entry),
`roadmap.md` (071 row + banner), `help_content.py` (generated shortcuts pick up the rename),
`todo.md` untouched (frozen).

## Manual verification (the maintainer, in the app)

- W-A: on a file ending in an empty line, `dd` on it removes it; `>>` indents the line by four,
  `<<` undoes it, `Vj>` shifts two; `*` on a word jumps to its next whole-word occurrence and every
  occurrence lights up; typing `/foo` lights matches as you type; Esc clears them, `n` still works.
- W-B: `G` then `o` on a long file — the new line is visible above the status bar; drag a tab to
  the front, Ctrl+Tab cycles in the new order; click the viewer, Ctrl+Tab once — the editor is
  focused and the tab did not change; Ctrl+Tab again — it cycles.
- W-C: the Reset button is there on every pass; F6 restarts the drifting light at its t=0
  position and a script counter from 1; an accumulated canvas empties.
- W-D: `composite`'s `u_cascade` / `u_paint` rows show the cascade and paint pictures and their
  names, no Load; a fresh pass's unbound sampler still shows the media row; dormant tiles read as
  dimmed pictures at the current `STALE_TINT` — the maintainer names a direction if not.
- W-E: `grep -ri colour` over the roster is empty; the gate fails when one is reintroduced.
- W-F: steps 1-6 mention no mouse; there is no Verifying section; the last chapter, followed
  literally on the shipped example, ends with persistent strokes, continuous under a fast drag,
  wiped by F6.

## Open questions for the maintainer

1. **#9b — what had focus.** Closed: the editor was unfocused and the maintainer was working
   in the Document tab. That is a sibling child, so the gate should have refused the chord; W-B
   step 3 starts with the repro.
2. **Who runs W-A's editor side.** Closed: the editor session (`editor-d2`), one request with
   all four items, pinging back when ready for re-vendoring.
3. **Reset also restarts bound videos?** Closed: yes, and encapsulated — the videos follow the
   document clock, so no video-specific reset code exists (D4/D5).

## Review history

(empty — filled by the pre-impl reviews per wave)
