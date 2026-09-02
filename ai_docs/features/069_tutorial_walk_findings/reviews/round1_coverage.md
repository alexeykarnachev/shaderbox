# Round 1 — coverage, citation truth, internal consistency

Reviewer: read-only pass over `01_spec.md` (as of the mid-review edit that moved the graph view to
070), `00_findings.md`, `01_design_scripting.md`, and the code each cite rests on. Every `file:line`
below was opened; nothing here is asserted from memory.

**Verdicts: Task 1 PARTIAL · Task 2 PASS · Task 3 PARTIAL.**

---

## Task 1 — coverage, finding by finding

Column "redesign vs patch" is folded into the note: where the maintainer's words say the design is
wrong, the note says whether the spec redesigns or patches.

| # | workstream | covered | note |
|---|---|---|---|
| 1 | W-A + W-H | yes | "there is no 512×512 option, where did you find it?" — W-A adds square presets 256/512/1024/2048 **and** a free `W × H` entry; W-H rewrites the "Before you start" line to use the new preset. Redesign of the control, not a one-off 512 entry. Correct call: the maintainer's complaint is that the combo cannot express a square canvas at all, and a fixed list plus free entry answers the class. |
| 2 | W-A | yes | Agent-found engine bug. Fix = route the combo through `Document.set_canvas_size`, the existing funnel. Verified: `tabs/document.py:111` really calls `render_pass.canvas.set_size` and `document.py:284` is the funnel that also writes `canvas_size`. Funnel fix, not a patch — matches the conventions law "a cross-cutting guarantee is enforced at the single FUNNEL". |
| 3 | W-A | yes | "I clicked the settings of the first pass and it still has 1280x960 -- wtf?" Same root as #2; one fix covers both. D2 pins "no manual JSON editing is a workaround", which is the maintainer's explicit rejection in the ledger. |
| 4 | W-A | yes | "is it even possible to have the first pass (or the last pass actually) at a resolution different from the canvas?" — W-A disables the output pass's size slider. Note this answers only half the question: the *design* question ("is the scale-only model right?") is answered by an Out-of-scope entry (absolute per-pass sizes, trigger = supersampled intermediate). That is the right shape — the maintainer asked a question, the spec gives a decision with a revisit trigger rather than a silent no. |
| 5 | W-B | yes | "Strip this text 2x. Pure textual noise, stop writing long passages everywhere... Compact this crap." Redesign, correctly: not a per-string cut but a word budget (D1) **plus** an `ast` gate that measures every `help_marker` / `set_tooltip` / `separator_text` / empty-state literal, shipped in the same commit as the sweep. That is exactly the maintainer's own standing rule ("a rule with no gate is a wish", filed mid-walk in the ledger). |
| 6 | W-H | **partial** | Two halves. Half A ("the pass settings panel warns" is a stale comment in `jfa.frag.glsl:17`) — covered, W-H removes it (verified: the comment is there, and `pass_settings.py` documents the warning as deliberately removed). Half B — the ledger asks the spec to **decide** whether the run count should be derivable in-shader from the canvas (`ceil(log2(max side))`, 9 at 512 vs 11 at 1080) "rather than a hand-set number". **The spec never decides this.** It is not in W-H, not a locked decision, not an Out-of-scope entry with a trigger. It matters concretely: W-A now lets a user pick 1024 or 2048, at which the shipped example's 9 runs is silently wrong. |
| 7 | W-B | yes | "text 'size (1080, 1080)' doesn't fit... it overlaps with the scroll bar" — W-B does both halves the ledger named: the derived value moves into the slider format (`%.0f%% · WxH`), and the popup sizes to content so there is no scrollbar at all. The ledger's diagnosis was "a settings popup that needs scrolling is itself the defect"; the spec adopts it and drops `PASS_SETTINGS_H`. Redesign. |
| 8 | W-H | yes | "WE ALREADY HAVE THE DEFAULT FIRST PASS. Should I delete it or what?" — W-H's text fixes say rename `main` → `paint`, no "Add". The ledger's broader diagnosis ("an instruction that depends on a note four paragraphs earlier is the failure shape — each step should be self-sufficient") is answered structurally by D8's template, not just by the one sentence. |
| 9 | W-C | yes | "add a keybinding to open settings for the currently selected pass (Alt+P?)" — `OPEN_PASS_SETTINGS` on Alt+P, dispatching on `panel_pass(...)`. Verified Alt+P is free (`commands.py` binds `K.p` only as Ctrl+P and Ctrl+Shift+P). |
| 10 | W-B | yes | "'Pass settings' is enough." Verified the tooltip at `pass_list.py:98` is the long one. Cut by the D1 sweep + gate, i.e. as a class, which is what the ledger asked ("grep every `set_tooltip(` / `help_marker(`"). |
| 11 | W-F | yes | "Make the vim symbols part of the editor itself, first-class editor elements — we have the custom editor exactly for this." D6 + W-F draw the status line INSIDE the editor rect on the cell grid with `STATUS_BG`, and the bottom bar keeps only host things. That is the redesign the words demand, not a re-layout of the imgui row. |
| 12 | W-F | yes | "We want relative line numbers, with `~` for the empty rows past the end... The editor lib must provide this — if it doesn't, file the issue there when doing this." Both obligations discharged: host-side sets `ChromeFlag.RELATIVE_NUMBERS` + replicates the picture `behavior_test.odin:338` pins, AND issue (2) is filed upstream. |
| 13 | W-E | yes | "The switch goes in the global Settings — a one-time setting, don't pollute the main UI with it." D5 + `EditorSettings.keymap` through the existing `_apply_editor_settings_to` funnel. Verified that funnel exists and is the single per-editor-settings path, and that `ffi.py` binds neither `ed_set_style` nor `ed_style` today. |
| 14 | W-F | yes | "red keyword on a red line is unreadable. Real vim flips the colours". Host fallback (gutter mark + 2px left bar, no whole-line fill) plus upstream issue (3). The ledger recommended exactly this pair. |
| 15 | W-F | yes | "Handle it properly — probably reset all errors when the file changes." Spec: fingerprint includes `is_tab_dirty`, no markers while dirty, strip keeps a dim "(stale until save)". Matches the maintainer's own suggested shape, and issue (4) carries the "proper" alternative upstream. |
| 16 | W-F | yes | Pure lib bug; ShaderBox-side nothing. Spec files it as issue (1) and picks the fix up at the re-vendor. Verified upstream HEAD is `68def59`, 32 commits past the vendored `e7db554`, and `gh issue list` is empty. |
| 17 | W-C | yes | The crash. Fix = `_draw_name` returns `renamed: bool`, `_draw_body` returns early. Verified the stale-local mechanism exactly as the ledger describes: `_draw_body` reads `name = app.pass_settings_name` at the top, guards `name not in ... passes` **before** `_draw_name` renames, then indexes `document.passes[name]`. Test through the headless imgui rig is specified — the ledger's own "a test that renames through the popup's body would have caught it". |
| 18 | W-C | yes | "The name should auto-apply without Enter." D11 + commit on `is_item_deactivated_after_edit()`, explicitly NOT per keystroke (rename is transactional — verified `rename_pass` moves the file, re-keys the graph, drops feedback history). Filed up into the imgui skill § 7.5 as a rule for every inline input, which is the right altitude. |
| 19 | W-D (reduced) + Out-of-scope → 070 | **partial** | "We need a better representation: visualise the passes as a directed graph... Or at least tune the current visuals, it is awful." The maintainer's own fallback ("or at least tune") is what ships: sublines go, name + thumbnail stay. The graph view is deferred to 070 with a fixed direction. This is defensible **as the maintainer's own call** (the spec says so twice), and the ledger listed exactly this as option (C) "B now, A as the feature". Marked partial only because the primary request is not in this feature; a later round should not re-litigate it. |
| 20 | W-H | **partial** | "we don't add this manually. We just hit Ctrl+R." The sentence was never located (verified: the ledger's repo-wide grep found nothing, and the spec repeats that). The spec's answer is a build-time check that the tutorial and Help contain no "add … script" instruction. That closes the *class* without ever having found the instance — reasonable, but the check as described greps for a phrase nobody has seen, so it may pin nothing. The maintainer's words describe a real mechanism (`Ctrl+R` creates the script — verified at `app.py:1248` / `commands.py:124`), so the positive half ("every mention says Ctrl+R creates it") is the load-bearing part; the negative grep is decoration. |
| 21 | W-A | yes | "we can't see the canvas boundary — it blends with the background." Checkerboard + 1px border, greys from `theme.py`, no literal colours at the call site (which is the repo's own colour rule). |
| 22 | W-G | yes | Both halves. (b) `MouseState` gains `down: bool`; (a) gains `prev_x`/`prev_y` so a shader can draw the capsule. Verified `MouseState` is position-only today and `ui.py:612` is the single hit-test that would fill them. **Not covered, and correctly so:** the ledger's open design question "whether a builtin `u_mouse` + `u_mouse_down` belongs in the engine directly" is not answered anywhere in the spec — see Task 1 gaps below. |
| 23 | W-G | yes | "we need a 'clear all' key or similar. Probably just handle this in the script." Spec takes option A (`CommandId.RESET_FEEDBACK` → `Document.reset_feedback`) and puts option B (`ctx.keys`) Out of scope with a named trigger. Verified `reset_feedback` exists with export as its only caller, so the command is genuinely new reach, not a rename. |
| 24 | W-E | yes | "Remove the highlight, the need for active areas, and the arrow-key rotation. Everything context-independent... Rework properly." D4 removes the subsystem whole, including the App state machine, `CYCLE_REGION`, and the conventions rule. This is the removal the words asked for, not a flag that hides the outline. |
| 25 | W-C | yes | "add an 'add pass' hotkey as well." `ADD_PASS`, Alt+A provisional, resolved by W-E's audit. |
| 26 | W-E | yes | "review ALL keybindings so nothing conflicts... Clean and conflict-less." D7 + a dedicated `02_keybindings.md` table with a generic ownership rule and a test asserting app chords disjoint from both keymaps' lists, **loaded from the vendored docs rather than retyped**. That last clause is what makes it a design and not an audit-once. |
| 27 | W-H | yes | "for the seed pass I don't see which resolution it should be... And sampling?" D8's pass card carries name · reads · size · format · smooth · repeat · runs in a fixed order on every pass step, generated from `graph.json` so it cannot drift. Redesign of the tutorial's structure, which is what the ledger prescribed. |
| 28 | W-C | yes | "when creating a pass, auto-activate it: open its code and render it." D10; after `add_pass`, run the tile-click path then open the gear. The ledger's "check `copilot/tools/document_ops.py`" is resolved correctly — verified the copilot has no add/rename/set-output pass tools at all. |
| 29 | W-G | yes | "Why one script for all passes? I asked for the correct approach to be researched and nothing was done." Superseded by #30's decision; W-G implements it, and the docs bullet gives 065 D12 and 068 D7 "superseded by 069" lines so the unlanded-decision failure is recorded rather than repeated. The ledger's separate ask — "the `orphan key` warning itself should also become a visible error in the script's error strip, not a console line" — is explicitly in W-G. |
| 30 | W-G + D3 | yes | "Must be a good UX and a robust design, no work-arounds, nothing temporary." D3 states B1 + broadcast, sourced to the design note, and W-G lists the concrete engine changes (`tick` takes the `Document`, `stopped` keyed `(pass, name)`, `EngineNode` retired, stub nested). See Task 3 for one gap the spec inherits from the note. |
| 31 | W-H | yes | "It mimics a structure but has no real flow... The attention blocks mimic order but it's theater." D8's template + generation from `graph.json` + concept sections demoted to unnumbered interludes so numbers 1–6 are the six passes. This is the structural redesign the words demand. |
| 32 | W-B | yes | "don't fit the panel. Align them vertically, under the sort picker (outside the sorting)." Verified `_draw_auto_row` uses `same_line` between every uniform and is itself placed with `same_line(spacing=XL)` after the sort row. W-B does both: drop the outer `same_line`, one uniform per line, still outside the sorted list. |
| 33 | W-H | yes | "'Replace the naive inner loop's fixed step' — replace WHERE?" W-H picks the ledger's first (honest) option: 2 and 6 become interludes that say "nothing to build here". |
| 34 | W-H | yes | "this is not the whole shader code???" W-H splices the COMPLETE `cascade.frag.glsl` from the shipped example via `build_tutorial.py`, so it cannot drift. Verified `build_tutorial.py` today only substitutes `{{IMG:x}}` and never reads `graph.json` or the pass sources — so this is real new machinery, correctly scoped. |
| 35 | W-H | yes | "you call the step 'The merge', but the shader is called 'cascade'". D8: heading = pass name, concept as subtitle. |
| 36 | W-C | yes | "I need to click each pass manually to trigger its redraw after I close/open the app." Option A (first-render every pass once, one per frame) with the branch rule kept for the steady state. The ledger's open question ("which pass was the output when you reopened?") is closed in the spec's Open-questions section with an explicit assumption and a "that would be a new finding" escape. Good shape. |
| 37 | W-D + D9 | yes | "sometimes the input uniform matches the pass name, sometimes not (u_scene ← paint)." One rule (`u_<pass>`, feedback `u_prev`), applied to both examples, the Help panel and the copilot prompt, plus the pay-off the ledger identified (default-wire by name). Verified the mixed conventions in both `graph.json` files and every shader's sampler declarations. |

**Task 1 verdict: PARTIAL.** 34 of 37 fully covered. Three gaps, in order of consequence:

1. **#6 half B is dropped without a decision or a trigger.** The ledger explicitly hands the spec a
   design question — should the JFA run count be derived from the canvas rather than hand-set? — and
   the spec answers neither yes, no, nor "later, when X". It becomes acute *because of this same
   spec*: W-A ships 1024 and 2048 presets, and the shipped example hard-codes 9 runs, correct only
   at 512. A reader who follows W-H's tutorial and then picks 1024 gets a silently wrong picture. It
   needs either a decision in W-D/W-G or an Out-of-scope entry with a trigger.
2. **#22's builtin-uniform question is unanswered.** The ledger asks whether `u_mouse` (vec4) +
   `u_mouse_down` should be engine-driven directly, noting "it would let the tutorial's paint step
   drop the script entirely". The spec's W-G adds the fields to `MouseState` only — the script route
   — and says "Tutorial's paint step may then be mouse-driven". That is a decision by omission, and
   it interacts with W-H: whether the tutorial's paint step needs a script at all is exactly what
   the answer decides, and W-H is the last wave. State it as a locked decision either way.
3. **#20 is closed by a check that may pin nothing** (see the row). Low consequence; noted so a
   later round does not treat it as covered without reading the check.

No finding is folded into the wrong workstream, and no finding is patched where the maintainer said
the design was wrong. The three highest-blast-radius "redesign, don't patch" demands — #24 (remove
019 whole), #26 (generic ownership rule + a test reading the vendored keymap docs), #31 (generate the
tutorial from `graph.json`) — are all answered structurally, and each ships its own gate.

---

## Task 2 — citation truth

Every `file:line` and every named symbol in the spec, opened and checked.

### Line citations (9)

| cite | verdict | what is actually there |
|---|---|---|
| `tabs/document.py:111` | **true** | `ui_document.document.render_pass.canvas.set_size((w, h))` inside the resolution-combo branch. Bypasses the funnel exactly as claimed. |
| `copilot/backend.py:135` | **true** | The comment line of the clamp block; `_MIN_CANVAS_PX = 16` / `_MAX_CANVAS_PX = 4096` follow at 136–137, used at 1079–1080. "The copilot's existing range" is accurate. |
| `copilot_chat.py:46` | **true** | The comment block; line 46 reads "stops it collapsing into an unrecoverable title-bar strip. no_nav_inputs stops the nav", continuing "outline on the programmatically-focused input (still typeable; /imgui-ui §8)". The spec's instruction — "the copilot window's stays if it guards something else — check this comment" — is exactly right: the comment says the flag guards a nav outline on a focused input, which is **not** region confinement. So that flag stays. |
| `tests/test_pass_editor_wiring.py:173` | **true** | `test_a_summon_from_a_non_editor_region_yields_the_editor_back`, which sets `app.active_region = ActiveRegion.PANEL` and asserts `editor_defocus_requested and region_focus_pending`. It is the one test that names the region system; "rewritten" is the right verb (it also tests the summon behaviour, which survives). |
| `conventions.md:328` | **true** | The bullet "**App-wide keyboard nav is region-confined (`nav_enable_keyboard` ON).**" — the entry D4 deletes. |
| `theme.py:193` | **true** | Inside the `SELECT`-vs-accent assertion block; 193 is `primary for primary, _active, _alpha in _ACCENTS.values()`. The surrounding comment states the outline-nesting rationale the spec says to re-examine. Off by ~2 lines from the assertion itself but unambiguously the right block. |
| `tabs/code.py:130` | **true** | `_script_errors_for` — "the document script shows its sentinel + every homeless soft-key error". The spec's claim that "the strip already shows soft errors on the script tab" and that only the script tab shows them is accurate. |
| `conventions.md:292–300` | **true** | The scripting bullet's error-as-data / freeze-granularity paragraph, including "an orphan/typo/sampler key records a soft `(document_id, name)` error + skip" — the sentence D3 changes. Note the keying is `(document_id, name)` today, which is precisely why W-G's `(pass, name)` re-key is needed. |
| `behavior_test.odin:338` | **true** | `test_relative_numbers_show_distance_and_an_absolute_cursor_line`, whose comment block pins nvim's `number relativenumber` picture with the cursor row absolute and LEFT-aligned. Exactly what the spec says it pins. |

### Symbol and factual citations

| cite | verdict | what is actually there |
|---|---|---|
| `Document.set_canvas_size` is "the funnel" | **true** | `document.py:284`; sets `self.canvas_size` then `render_pass.canvas.set_size`. Its docstring names the copilot's old bypass as the bug this exists to prevent. |
| `target_size(document.canvas_size)` follows for non-output passes | **true** | `document.py` render loop resizes each non-output pass from `entry.target.target_size(self.canvas_size)` per render. The W-A test as written is checkable. |
| `evaluation_order(graph, pass)` exists | **true** | `pass_graph.py:367`; it calls `assert_plan_invariants` itself, so the W-C claim "the draw-once invariant assert in `pass_graph.py` stays the guard" holds. |
| "per-pass `first_render_done` on `Pass`" is new | **true** | `first_render_done` exists today only on `Document` (`document.py:238`, set at 393, read in `ui.py`, `document_grid.py`, `popups/examples.py`). Nothing on `Pass`. The spec lists `core.py` in W-C's files, which is where `Pass` lives. Correct. |
| `ui.py`'s frame gate admits "at most one not-yet-rendered document" per frame | **true** | `ui.py:230–238`, the `pending_first` block, with the 066 D2 comment explaining the bounded-cost reasoning. W-C's pass-level gate is a faithful analogue. |
| `EngineNode` protocol | **true** | `scripting/engine.py:95`, `uniform_values` + `get_active_uniforms`. `tick`/`tick_export`/`dry_run`/`reload` all take `document: EngineNode` and `project_session.tick` passes `ui_document.document.render_pass` — i.e. the OUTPUT pass. The spec's "retire it for a `Document`-shaped one" is the right change and the right list of call sites. |
| `stopped_uniforms` is a flat set of names | **true** | `UIDocumentState.stopped_uniforms` (a list, coerced per-frame), built by `_stopped_for` into a `frozenset[str]`. The design note's "would freeze a name on every pass" is correct. |
| `MouseState` is position-only; `EXPORT_MOUSE` | **true** | `scripting/context.py:7`, fields `x`/`y` only; `EXPORT_MOUSE = MouseState(0.5, 0.5)` is the export default injected via the isolation seam. |
| `ui.py` hit-test fills `app.script_mouse` | **true** | `ui.py:612–618`; `item_normalized_mouse` over the captured preview rect, `hit[2]` the in-bounds flag. `is_mouse_down(0)` fits there. |
| `Document.reset_feedback`, export-only caller | **true** | `document.py:327`, sole call site `document.py:650`. No command, chord, or button. |
| `_apply_editor_settings_to` is the funnel | **true** | `app.py:1306`; five settings; `apply_editor_settings` loops every open session. D5's "applied to every open editor session" is deliverable through it. |
| `editor/ffi.py` lacks `ed_set_style` / `ed_style` / `ed_filler_glyph` | **true** | Grepped: absent. `ChromeFlag.RELATIVE_NUMBERS = 1` and `STATUS_BG/TEXT/ACCENT` (slots 8/9/10) **are** present, as W-F assumes. |
| Vendored `VERSION` = `e7db554`; upstream HEAD `68def59`; 32 commits between | **true** | `resources/editor/VERSION` is `e7db554ddfc…`; `git rev-parse --short HEAD` in the editor repo is `68def59`; `git rev-list --count e7db554..HEAD` = 32. (The ledger's #16 said "29 commits" — stale by three, harmless, and the spec does not repeat the number.) |
| `docs/standard_keymap.md` exists upstream | **true** | Present, with `STANDARD_BINDINGS` cross-checked by an upstream test. |
| "F-keys and Alt are untouched by both keymaps" | **true** | No F-key appears in either doc. `standard_keymap.md` line 64: "Every other key with Ctrl, Alt or Super held … returns false from `ed_key` and is the host's." `vim_coverage.md`'s only Ctrl chords are `<C-d> <C-u> <C-f> <C-b> <C-e> <C-y>` (scrolling); no Alt. The chord proposals in Open question 4 rest on a verified premise. |
| Standard keymap claims Ctrl+N and Ctrl+Space | **true** | Both appear in `standard_keymap.md`'s chord set. The two "known moves" (`NEW_DOCUMENT` off Ctrl+N, `TOGGLE_DOCUMENT_PLAY` off Ctrl+Space) are real collisions, verified against `commands.py`. |
| `gh issue list` on `alexeykarnachev/editor` is empty | **true** | Ran it; exit 0, no rows. |
| `chrome_emit_gutter` is core-only, not exported through the ABI | **true** | Defined `src/chrome_emit.odin:50`, referenced only by upstream tests. Not in `ffi/`. |
| `build_tutorial.py` substitutes only `{{IMG:x}}` today | **true** | Its `build()` loops the six pass names replacing `{{IMG:<name>}}` with a data URI and writes `tutorial.html`. It reads no `graph.json` and no pass source. W-H's `{{CARD:x}}` / `{{CODE:x}}` is genuinely new. |
| `jfa.frag.glsl` carries the stale "panel warns" comment | **true** | Line 17: "Resize note: 9 runs spans 512px. A 1024px canvas needs 10, and the pass settings panel warns". `pass_settings.py` (~256) documents that the warning was deliberately removed. |
| The copilot has no pass tools | **true** | No `add_pass` / `rename_pass` / `set_output_pass` anywhere under `shaderbox/copilot/`. W-C's "Copilot has no pass tools — nothing to mirror" is right, and it correctly resolves the ledger's speculative "check `copilot/tools/document_ops.py`" (no such file). |
| Bloom Chain samplers `u_src` / `u_lit` / `u_glow` / `u_trail` | **true** | Verified in all five Bloom shaders and its `graph.json`. |
| "Bloom's `bright` and `trail` both read `scene`" | **true** | `bright.inputs = {u_src: scene}`, `trail.inputs = {u_src: scene, u_prev: trail}`. |
| "inside ONE pass two inputs from the same source would collide, and none exists" | **true** | Enumerated both examples: every pass's inputs have distinct sources. Under D9, Bloom becomes `bright: u_scene`, `blur: u_bright`, `trail: u_scene + u_prev`, `composite: u_scene + u_blur + u_trail`; RC becomes `seed: u_paint`, `jfa: u_seed + u_prev`, `df: u_jfa`, `cascade: u_paint + u_df + u_prev`, `composite: u_cascade + u_paint`. No collision anywhere. |
| The W-D naming arrow `u_src/u_lit/u_glow/u_trail → u_scene/u_scene…` | **stale/garbled** | The *conclusion* is right and the *mapping as written* is wrong. `u_glow` reads `blur`, so it becomes `u_blur`, not a second `u_scene`; `u_src` maps to two different names depending on the pass (`u_scene` in `bright` and `trail`, `u_bright` in `blur`). The ellipsis hides that. It is the one place in the spec an implementer could follow the text literally and produce the wrong rename. Replace with the per-pass table above. |

**Task 2 verdict: PASS.** All 9 line citations and all 26 symbol/factual citations are true. One
entry is stale-as-written (the W-D Bloom rename arrow), and it is a wording defect inside a correct
conclusion rather than a false claim about the code.

**One convention violation worth naming, not a citation error:** `conventions.md ## Code rules`
says *"No raw line numbers OR file-length counts in docs (`todo.md`, specs, conventions, comments)
… Cite the **symbol** instead of a line."* The spec carries nine `file:line` cites. Every one is
currently accurate, and every one rots on the next edit to those files — `conventions.md:292–300`
and `conventions.md:328` will move the moment W-E and W-G edit `conventions.md`, which this very
spec instructs. Convert to symbol cites (`Document.set_canvas_size`, the region-confinement bullet,
`_script_errors_for`, `_ACCENTS`, `test_a_summon_from_a_non_editor_region_yields_the_editor_back`).

---

## Task 3 — internal consistency

### D1–D12 against each other

Checked pairwise for contradiction; the following are the live interactions.

1. **D1 × W-F's status line — consistent, and the spec applies it.** W-F's "(stale until save)"
   suffix is called out as "one clause, D1". Good. But two W-F strings are *not* measured against
   D1's gate: the mode badge and the ruler are drawn on the editor's own cell grid, not through
   `help_marker` / `set_tooltip` / `separator_text` / the `FG_DIM` empty-state idiom, so W-B's `ast`
   gate cannot see them. That is fine (they are vim's own furniture, not prose), but the spec should
   say so, or the next round will read it as an escape hatch.
2. **D4 × D6 — consistent and correctly ordered.** D4 removes the region system; D6 moves the vim
   chrome into the editor rect. Both touch focus, and the Order section sequences W-E before W-F for
   this reason. See the dependency check below, where the stated reason is only half right.
3. **D5 × D6 — consistent.** D5 makes the keymap a global setting; D6's furniture is keymap-
   conditional (relative numbers + `~` under vim, absolute + caret readout under standard), and the
   Out-of-scope entry pins that the standard keymap gets no design of its own yet, with a trigger.
4. **D7 × D4 — consistent, and D4 shrinks D7's work.** Removing 019 deletes `CYCLE_REGION` (Ctrl+`),
   one row out of the audit table, and leaves editor-focus as the only focus notion the ownership
   rule must consider. The spec says this in W-E.
5. **D8 × D9 — consistent and correctly ordered.** D8 generates pass cards from `graph.json`; D9
   rewrites the input names inside those same `graph.json` files. Generation must run after the
   rename, which is why W-D precedes W-H. Stated.
6. **D9 × D10 — consistent, and there is a real interaction the spec gets right.** D10 activates a
   new pass (opens tab + sets output) and D9's default-wiring pre-fills `u_<x>` edges on `add_pass`.
   Both fire in the same moment. W-C owns D10, W-D owns D9, and W-C lands first — so during waves
   1–6 a new pass activates without auto-wiring, and from wave 7 it does both. No conflict, but the
   W-C test for D10 must not assert "the new pass has no inputs", or W-D breaks it.
7. **D10 × D12 — consistent.** D10 reuses "the tile-click path"; D12 only removes the tile's
   sublines, not the click behaviour. W-D says the error subline becomes the border it already has.
8. **D11 × D2 — consistent.** Neither touches the other.
9. **D12 as edited — consistent with the Out-of-scope entry.** The mid-review edit moved the graph
   view to 070 in three places (Out of scope, D12, W-D, Open question 3, and the W-D manual
   verification line) and I found no leftover claiming the graph view ships here. The edit is clean.
   The only residue: the spec's opening paragraph still says the feature "touches `conventions.md`"
   and lists eight waves — both still true.

**One decision is missing from D1–D12 that the workstreams rely on:** W-G's broadcast semantics are
attributed to D3, and D3 states them, but D3 does not carry the design note's *rationale* clause
that makes the rule safe — "no uniform value is a dict, whatever the pass is named". That premise is
what makes value-type dispatch unambiguous, and it is an invariant the implementation must preserve
(a future `mat4`-as-dict or a struct-valued uniform would silently become a pass block). It belongs
in D3 as a stated invariant, not only in the note.

### The spec against the design note

| claim | consistent? |
|---|---|
| D3 = "B1 + bare-key broadcast" | **Yes.** The note's `## Decision: B1` and `### Bare keys broadcast` say exactly this, both attributed to the maintainer. |
| "Supersedes 065 D12 (per-pass files)" | **Yes.** Verified 065 D12 reads "One script per PASS, keyed `(document, pass)`" and was reached during 065's review. The note records the maintainer rejecting option A on use case 2 ("the context must be shareable"), which is the reason the supersession is a decision and not a reversal by neglect. |
| "lifts 068 D7's retraction" | **Partly.** 068 D7 retracted script-driven drawing for **two** reasons, verified in `068/01_spec.md:87` and echoed in `tutorial_body.html:183`: (a) the engine binds to the OUTPUT pass so a uniform on another pass is dropped, and (b) `ctx.mouse` carries position only, no buttons. D3 lifts (a); W-G's `MouseState.down` lifts (b). So the lift is real, but it takes **both** W-G items, and it is D3 alone that the spec credits. Minor: state that D3 + the mouse change together lift D7, so a wave that ships one without the other does not claim the lift. |
| `stopped` re-keyed `(pass, name)` | **Yes**, and the note flags it as the first thing the spec must pin. W-G pins it. |
| Stub emits nested skeleton with a commented block per pass | **Yes**, matching the note's `## Decision: B1` stub snippet. Note `script_stub_for` today takes `Iterable[moderngl.Uniform]` and `_scriptable_uniforms_for` reads only `document.render_pass` — both must change; W-G names `script_stub_for(document)` and lists `project_session.py`. Consistent. |
| Rename does NOT rewrite the script | **Yes.** The note's B1 section explicitly retracts the `ast` rewrite ("no rewrite of user code… the old key becomes the strip error"). W-G does not contradict it. Worth noting for a later round: the note's `## What the spec must pin` section still contains **B2-era** bullets (the `ast` insertion for `Ctrl+R`, per-pass method jump, "Delete pass: the method stays"), and the `## Decision: B1` section says it is "replacing the B2-specific items above". The spec correctly follows the B1 section. A reader of the note alone could take the earlier bullets as live — the note, not the spec, is where that ambiguity lives. |
| Orphan → strip error, not `logger.warning` | **Yes**, in both. |

### Order section — dependency claims, checked against the code

The section is titled "dependencies, not preference", so each claim is a testable assertion.

1. **"W-C first — small, unblocks the walk itself."** Not a dependency, a priority. Fine, and W-C
   genuinely depends on nothing else (the crash fix, the commit rule, D10, the two chords, and the
   first-render change touch no file another wave must land first). One real coupling the spec does
   not name: W-C adds two commands, and W-E's audit **moves** two others. If W-C picks Alt+P/Alt+A
   before the audit runs, the audit inherits them as fixed. The spec half-acknowledges this ("chord
   from W-E's audit; Alt+A candidate") and Open question 4 makes them provisional — consistent, but
   the W-C chord-uniqueness test will need re-running after W-E.
2. **"W-A second — the tutorial's first step depends on it."** True in substance, though the
   dependency is W-H's, not W-A's: nothing in W-A needs W-C. Ordering W-A second is preference.
   Harmless.
3. **"W-B third — independent; early so every later UI wave is written under D1."** Verified
   independent. Note W-B's gate will then run against code W-D, W-F and W-G add — which is the
   point, and it means those later waves must not introduce an over-budget string. Worth calling out
   in each wave's post-review, since the gate lands in wave 3 and the violations would land in 5–7.
4. **"W-E before W-F, because the status line and the host's key routing both hang on which focus
   notions remain."** **Half true, and the half that is false is the status line.** I read the code:
   - *Key routing* — true. `hotkeys.py::_VIM_RESERVED_CHORDS` resolves conflicts by "editor focused
     → vim wins, unfocused → app wins", and `app.editor_focused` is the surviving focus notion after
     D4. W-E's per-keymap rework genuinely depends on knowing that.
   - *Status line* — **false as stated.** The status line is drawn by `tabs/code.py` inside the
     editor image, from `editor.get_mode()` / caret position. Its only current entanglement with the
     region system is that it lives on the row `_draw_copilot_bar` hosts, and that child's
     `no_nav_inputs` flag is unrelated to it. Nothing about drawing a `STATUS_BG` strip on the cell
     grid depends on `ActiveRegion` being gone. The real W-E→W-F dependency is the **keymap
     setting**, and the spec states that one correctly and separately ("the keymap setting's ABI half
     waits for W-F's re-vendor"). Note that dependency runs W-F→W-E, the opposite direction, which
     is why the spec has to split the wave ("land the setting last within W-E or first within
     W-F") — an honest treatment of a genuine cycle. Recommendation: keep the order, fix the stated
     reason. The status line does not belong in it.
5. **"W-G before W-H so the paint step can be written against it."** True, and it is the strongest
   dependency in the list: W-H's paint step can only be mouse-driven if `MouseState.down` exists and
   D3 lets a script drive a non-output pass. Verified both are absent today.
6. **"W-D before W-H (the tutorial's wiring lines change under D9)."** True and load-bearing —
   stronger than the spec says. W-H generates pass cards **from `graph.json`**, and W-D **rewrites
   those `graph.json` files**. So W-H does not merely reference D9's names, it reads the files D9
   edits. If the order inverted, every generated card would carry the old names.
7. **"Does W-H depend on W-D or only on D9?"** — **On W-D itself, not only on D9**, for the reason
   in 6: the generation reads the renamed files, so it needs the rename *landed*, not just decided.
   It does **not** depend on W-D's strip tune or default-wiring halves. So the dependency is real but
   narrower than "W-D": W-H needs W-D's naming bullet only. That distinction matters because it means
   the strip tune could ship in any wave without moving W-H.
8. **W-H last as "the verification of everything above."** Consistent with the manual-verification
   section, where W-H's row is the maintainer's end-to-end walk.

**One order gap.** W-A adds 1024/2048 canvas presets; the shipped example's `jfa` hard-codes 9 runs
(correct at 512) and W-H generates the tutorial's pass cards from that example. If #6's half-B is
resolved by making the run count canvas-derived, the fix lands in the example's shader and therefore
must precede W-H — i.e. the missing decision from Task 1 also has an ordering consequence.

**Task 3 verdict: PARTIAL.** D1–D12 are mutually consistent and faithful to the design note; the
Order section's dependency claims are correct except one stated reason (the status line does not
depend on W-E) and one under-stated dependency (W-H reads the files W-D rewrites, and needs only
W-D's naming half). Two items to add: the "no uniform value is a dict" invariant into D3, and the
note that lifting 068 D7 takes D3 *and* the mouse change.

---

## False trails — probed, fine, do not re-check

- **Every one of the spec's nine `file:line` citations resolves to what the spec says.** They are
  accurate today; the only issue is the convention against raw line numbers, and the rot risk on
  `conventions.md:292–300` / `:328` once W-E and W-G edit that file. No fact-checking needed again.
- **`Document.first_render_done` already existing does NOT collide with W-C's plan.** The existing
  field is document-level (`document.py:238`, read by `ui.py`, `document_grid.py`,
  `popups/examples.py`); W-C adds a per-pass one on `Pass` in `core.py`. Different scopes, no clash.
- **The D9 rename cannot produce a sampler-name collision in either shipped example.** I enumerated
  every pass's inputs in both `graph.json` files and every sampler declaration in all eleven shaders.
  No pass reads two inputs from one source. The spec's conclusion is right; only its Bloom arrow is
  written badly.
- **The copilot really has no pass tools.** Grepped all of `shaderbox/copilot/` for `add_pass`,
  `rename_pass`, `set_output_pass` — nothing. `copilot/tools/document_ops.py` (named speculatively in
  the ledger's #28) does not exist. W-C's "nothing to mirror" needs no further check.
- **`ChromeFlag.RELATIVE_NUMBERS` and the `STATUS_*` theme slots are already bound** in
  `editor/ffi.py` (flag value 1; slots 8/9/10). Only `ed_set_style` / `ed_style` / `ed_filler_glyph`
  are missing, exactly as W-F says. Do not re-audit the ffi surface.
- **The F-key and Alt claim underpinning the chord proposals is sound.** Neither
  `docs/standard_keymap.md` nor `docs/vim_coverage.md` mentions any F-key; the standard keymap
  explicitly returns every Ctrl/Alt/Super chord it does not list to the host; the vim doc's only
  Ctrl chords are the six scrolling ones. Ctrl+Shift+N and F5/F6 are genuinely free.
- **The upstream editor facts hold**: HEAD `68def59`, vendored `e7db554`, 32 commits between,
  `standard_keymap.md` present, `gh issue list` empty, `chrome_emit_gutter` core-only.
  (The ledger's "29 commits" in #16 is stale; the spec does not repeat it, so nothing to fix.)
- **`build_tutorial.py` genuinely lacks card/code generation today** — it substitutes `{{IMG:x}}`
  only. W-H's machinery is new work, correctly scoped, not a claim about existing behaviour.
- **The mid-review edit that moved the graph view to 070 is internally clean.** I checked all five
  places it touches (Out of scope, D12, W-D heading + body, Open question 3, manual verification);
  no leftover asserts the graph view ships in 069.
- **W-B's `ast` gate can actually see the strings it targets.** The four idioms it names
  (`help_marker`, `set_tooltip`, `separator_text`, `text_colored` with `COLOR.FG_DIM`) are the real
  call shapes in `pass_settings.py`, `pass_list.py` and `tabs/document.py` — including the
  overflowing empty-state line and the long gear tooltips. The gate is implementable as described.
