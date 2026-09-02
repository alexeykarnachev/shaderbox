# Round 2 — closing round 1, fresh coverage, citation truth on what was added

Reviewer: read-only pass over the PATCHED `01_spec.md`, `00_findings.md` (the maintainer's verbatim
"Reported" column as the external anchor), `01_design_scripting.md`, and both round-1 reports. Every
file:line and symbol ADDED since round 1 was opened; the round-1 reports' "False trails" sections
were taken as verified and not re-checked.

**Verdicts: Task 1 PASS · Task 2 PARTIAL · Task 3 PASS.**

---

## Task 1 — round 1, item by item

A row is closed only by quoted spec text. "Closed differently" means the spec resolves the concern
by another route than the reviewer proposed — which is legitimate, and named so.

### Round 1 coverage reviewer

| Round-1 item | Verdict | The new spec text that closes it |
|---|---|---|
| **Gap #6** — "the spec never decides" whether the JFA run count should be canvas-derived; acute because W-A ships 1024/2048 while the example hard-codes 9 | **closed** | W-H: "**JFA run count (#6):** the example's `jfa.frag.glsl` derives its offset from the canvas — `offset = exp2(ceil(log2(max(u_resolution.x, u_resolution.y))) - 1.0 - u_pass_iteration)` and returns its input unchanged when `offset < 1.0` — so any run count >= `ceil(log2(max side))` is correct at every preset; the shipped `iterations` becomes 11 (correct through 2048, the largest W-A preset) and the 'panel warns' comment goes. The tutorial's card shows runs 11 and the interlude explains the formula; the 'resize changes the answer' paragraph becomes 'resize past 2048 and add a run'. The engine stays out of it (it cannot know a JFA from a cascade)." — a decision, not a deferral, and it answers the ledger's own framing ("derivable from the canvas in-shader … rather than a hand-set number"). The ordering consequence round 1 flagged is also closed: the change lands in W-H itself, which already runs last, so nothing must precede it. |
| **Gap #22** — the builtin `u_mouse` half is unanswered, "a decision by omission" | **closed** | W-G: "**No builtin `u_mouse` uniform** (#22's open half, default picked): the mouse reaches shaders through the script only, so the tutorial's paint step is written against D3 — which restores 068's lost 'stress the scripting path' goal (068 D7 lifted). Trigger to add a builtin: a second mouse-driven example that needs no other script state." Decision + trigger, and it resolves the round-1 note that this interacts with W-H (the paint step is now stated as script-driven, not "may be"). |
| **Gap #20** — closed by a check "that may pin nothing"; the positive half is the load-bearing one | **closed, restated positively** | W-H: "every sentence in the tutorial and the Help panel that mentions the document script says that `Ctrl+R` (or Script → open) creates it; the build test asserts each occurrence of 'script' in `tutorial_body.html` is within a sentence carrying `Ctrl+R` or is inside a code block." The negative grep for a phrase nobody has seen is gone; the check now measures the positive property over every occurrence. Open question 1 keeps the escape hatch ("if the maintainer meets the sentence again, its location goes to the W-H commit"). |
| **Bloom rename mapping** — "the one place an implementer could follow the text literally and produce the wrong rename"; `u_glow` → `u_blur` not `u_scene`, `u_src` maps to two names | **closed** | W-D now carries the per-pass table inline: "Bloom Chain, by source pass: `bright.u_src`→`u_scene`, `blur.u_src`→`u_bright`, `trail.u_src`→`u_scene`, `composite.u_lit`→`u_scene`, `composite.u_glow`→`u_blur`, `composite.u_trail` stays." Matches the round-1 corrected table row for row. The ellipsis is gone. |
| **Order claim** — "W-E before W-F because the status line … hangs on which focus notions remain" is false for the status line; the real edge runs W-F→W-E | **closed** | Order is inverted and the reason restated: item 4 is now "**W-F** (re-vendor, editor chrome, lib issues filed) — the re-vendor brings the standard keymap and the keymap docs W-E's audit and keymap setting read from. The status line itself depends on nothing in W-E (verified: it reads the editor's focus stop, which stays)." Item 5: "**W-E** … after the re-vendor. Independent of W-F's chrome work; the two may run in parallel once the `.so` is vendored." The false reason is not merely deleted — it is explicitly negated. |
| **Order claim** — W-H depends on W-D's *naming half only*, and reads the files W-D rewrites | **closed** | Order item 8: "It generates cards and code from the example files W-D rewrites (naming) and W-G's script contract, and the resolution preset from W-A." The parenthetical "(naming)" is the narrowing round 1 asked for. |
| **Order heading over-claims "dependencies, not preference"** | **closed** | Heading is now "**## Order (three forced edges, the rest a solo-maintainer scheduling choice)**", body: "Forced: W-F's re-vendor before W-E's keymap setting and audit; W-D's naming before W-H; W-A's preset before W-H. Everything else could run in parallel … serial waves are chosen so one review pair and one commit series own each change and the three shared files never merge." Exactly the round-1 finding, adopted. |
| **W-C's chord depends backwards on W-E** (the one ordering inconsistency inside the spec) | **closed** | W-C: "Commands: `OPEN_PASS_SETTINGS` (Alt+P) and `ADD_PASS` (Alt+A) — provisional per the closed questions; W-E's audit may move them, W-C does not wait for it." Open question 4 pins the provisional values. The backwards edge is dissolved by making W-C's chords concrete-but-provisional rather than pending. |

**Not carried into the patched spec (round-1 items the spec chose not to fold):**

- The **`file:line` convention violation** (round 1 Task 2's closing note: `conventions.md ## Code
  rules` bans raw line numbers in docs; convert to symbol cites). The patched spec still carries
  line cites — `tabs/document.py:111`, `copilot/backend.py:135`, `ui.py:553`, `copilot_chat.py:53`,
  `popups/examples.py:101`, `tests/test_pass_editor_wiring.py:173`, `conventions.md:328`,
  `theme.py:193`, `tabs/code.py:130`, `conventions.md:292–300`, `behavior_test.odin:338`,
  `8d454b7b…:56-60`, `0b0d16bb…:12-13` — and it *added* five. All are true today (Task 3), and two
  (`conventions.md:328` and `:292–300`) rot in this very feature since W-E and W-G edit that file.
  **Still open.** It is a doc-convention defect, not a correctness one.
- The **"no uniform value is a dict" invariant** round 1 asked to lift from the design note into D3.
  D3 reads: "a non-dict value under a bare key broadcasts to every pass declaring that uniform; a
  pass block wins over a broadcast; unknown pass / uniform = strip error." The dispatch rule is
  stated; the premise that makes it unambiguous is not. **Still open**, low stakes.
- **D1 × the vim status-line strings** (round 1 Task 3 item 1: the mode badge and ruler are drawn on
  the cell grid and W-B's `ast` gate cannot see them; say so or the next round reads it as an escape
  hatch). Nothing in the patched spec addresses it. **Still open**, cosmetic.
- **"lifting 068 D7 takes D3 *and* the mouse change"** — round 1 asked the spec to say so. It now
  does, in W-G's builtin-uniform bullet: "the tutorial's paint step is written against D3 — which
  restores 068's lost 'stress the scripting path' goal (068 D7 lifted)", with `MouseState.down` in
  the same bullet. **Closed differently** — by co-locating both halves in one bullet rather than by
  an explicit sentence in D3.

### Round 1 design reviewer

| Round-1 item | Verdict | The new spec text that closes it |
|---|---|---|
| **C1** — D5 fires the code-editor convention entry's own revisit trigger ("a non-modal keymap ships editor-side"); the entry's "vim-modal library" becomes false and the spec does not name the edit | **closed** | W-E: "the `conventions.md` code-editor entry ('Inline editor state lives on `App` … one libeditor instance per opened FILE') has its revisit trigger fired by D5 ('a non-modal keymap ships editor-side') — rewrite its 'vim-modal library' sentence to 'vim-modal or standard, per `EditorSettings.keymap`'." Verified the entry exists at `conventions.md:360` with the trigger at `:375–378`, worded as quoted. W-E's file list now reads "`conventions.md` (two entries)". |
| **C2** — D12 supersedes 065's "a canvas UI is a separate feature" without a doc line; two different 065 items superseded, one line | **closed differently** | The spec does not add a second 065 doc line — it removes the need for one. Out of scope: "**The graph view of the pass strip (#19 option A).** Maintainer's call: a separate feature (070), not this one." Under that framing 065's graph-view exclusion is *upheld*, not superseded: 070 is exactly the "separate feature" 065 named. Only 065 D12-the-script-decision is superseded, and it has its line ("065 D12 and 068 D7 get 'superseded by 069' lines", W-G). The `Files touched` line "065 → done with the D12 supersession noted" is therefore now unambiguous. C2's ordering worry ("the doc line cannot be written until Open question 3 is answered") is moot — Open question 3 is answered. |
| **C3** — W-F re-vendors the binary but does not name the `## Known quirks` entry documenting it | **closed** | W-F: "rebuild from a clean tree per 067 § 13, update `resources/editor/VERSION`, **and the `conventions.md ## Known quirks` entry that owns the vendored version + rebuild procedure**". Verified `## Known quirks` exists (`conventions.md:744`) and the code-editor entry points to it ("The vendored binary + rebuild procedure live in `## Known quirks`", `:374–375`). |
| **MISS E1** — `popups/examples.py` passes `nav_flatten=True`, not in W-E's file list | **closed** | W-E: "its call sites (`widgets/document_grid.py` ×2, `popups/examples.py:101`)"; `popups/examples.py` is in W-E's Files line. |
| **MISS E2** — `tabs/document.py`'s `nav_flattened` child flag + its comment becomes a false statement | **closed** | W-E: "`tabs/document.py`'s `ChildFlags_.nav_flattened` plus its now-false comment"; `tabs/document.py` is in W-E's Files line. |
| **MISS E3** — `preview_cell`'s `nav_flatten` parameter and docstring paragraph | **closed** | W-E: "The `nav_flatten` machinery goes with it: `preview_cell`'s `nav_flatten` parameter and docstring paragraph (`ui_primitives.py`)". The deliberate decision round 1 asked for (goes vs stays inert) is made: it goes. |
| **MISS E4** — there are **four** `no_nav_inputs` sites, not three; `ui.py:553` is unaccounted for and is a different case (unconditional, not a ternary) | **closed** | W-E: "and all FOUR `no_nav_inputs` sites (`ui.py`: editor child, the copilot bar child at `:553`, the panel child; `copilot_chat.py:53`) — with nav off the flag is inert everywhere, so all four go." Verified by grep: exactly four sites, at `ui.py:379`, `ui.py:553`, `ui.py:709`, `copilot_chat.py:53`. Note this **overrides** round 1 coverage's separate read that the copilot-chat flag should stay (it cited the `copilot_chat.py:46` comment about a nav outline on a focused input). The spec's reasoning is sound and stronger: with `nav_enable_keyboard` off there is no nav outline to suppress, so the flag is inert. Verified the comment at `:45–47` describes exactly a nav-outline suppression, which nav-off makes moot. |
| **MISS E5** — `067_custom_editor.md` documents `_VIM_RESERVED_CHORDS` and the editor child's `no_nav_inputs`; not in any file list | **closed** | W-E: "`067_custom_editor.md` gets a 'keymap routing superseded by 069' note where it states the vim-only `_VIM_RESERVED_CHORDS` routing and the editor child's `no_nav_inputs`". |
| **MISS G1** — `scripting/api_doc.py` produces the SCRIPT API block from `MouseState.__dataclass_fields__`; spec named the test, not the module | **closed** | W-G: "The SCRIPT API prompt block: `scripting/api_doc.py` generates it from `MouseState`'s dataclass fields and the stub"; `scripting/api_doc.py` is in W-G's Files line. Verified `api_doc.py:61` is `_MOUSE_FIELDS: str = ", ".join(f"`{n}`" for n in MouseState.__dataclass_fields__)`. |
| **MISS G2** — `copilot/prompt_context.py` is the sole importer of the generated block | **closed** | W-G: "`copilot/prompt_context.py` is its importer — both change"; in the Files line. Verified `prompt_context.py:9` imports `script_api_summary` and calls it at `:99`. |
| **MISS G3** — `scripting/__init__.py` re-exports `EngineNode` | **closed** | W-G: "`scripting/__init__.py` re-exports follow"; in the Files line. Verified `EngineNode` at `__init__.py:13` and in `__all__` at `:30`. |
| **MISS G4** — `tick_export` never named | **closed** | W-G: "`ScriptEngine.tick` / `tick_export` / `dry_run` take the `Document` (every caller in `project_session.py` — `tick`, the export pre-render closure, `write_script_source`'s reload + dry_run — stops passing `.render_pass`)". Verified all four methods exist (`engine.py:281 reload`, `:370 tick`, `:397 tick_export`, `:421 dry_run`). **`reload` is still unnamed** — see Task 2. |
| **MISS G5** — the `stopped` re-key's UI consumers: `widgets/uniform.py` and the five example `document.json`s | **closed** | W-G: "`UIDocumentState.stopped_uniforms` becomes a list of `[pass, name]` pairs (the five shipped examples' `document.json` files that persist `"stopped_uniforms": []` are hand-edited, NO migration code); … `widgets/uniform.py`'s play/stop button passes the panel pass". Both in the Files line ("`widgets/uniform.py`", "the five example `document.json`s"). Verified exactly five example `document.json`s carry `stopped_uniforms`, and `widgets/uniform.py:169,302` call `app.set_uniform_stopped` with a bare name while `:177` already computes `panel_pass`. |
| **MISS G6** — `copilot/backend.py`'s `_get_script_driven_uniforms` gate on `set_uniform` becomes ambiguous under D3 | **closed** | W-G: "`copilot/backend.py`'s `_get_script_driven_uniforms` gate on `set_uniform` tests `(pass, name)` — today's name-only test would reject `composite.u_x` because `paint.u_x` is driven"; `copilot/backend.py` in the Files line. Verified the gate at `backend.py:881` (`if name in self._get_script_driven_uniforms(document_id)`) plus three other call sites at `:640`, `:723`, `:740`. |
| **MISS G7** — `_scriptable_uniforms_for` reads `.render_pass` only; it feeds the stub | **closed** | W-G: "`script_stub_for(document)` emits the nested skeleton with one commented block per pass, fed by `_scriptable_uniforms_for` reshaped to per-pass (today it reads `.render_pass` only — the sibling of the 068 D7 defect)". Verified `project_session.py:632–640`: `_scriptable_uniforms_for` returns `….document.render_pass.get_active_uniforms()`, consumed by `script_stub_for` at `:652` and `:669`. |
| **MISS D1** — `tests/test_pass_verbs.py` imports `_strip_order`, which W-D was deleting | **closed differently** | W-D no longer deletes `pass_list.py`; the strip survives with the graph view gone to 070. Files line: "`widgets/pass_list.py` (`_strip_order` stays; `tests/test_pass_verbs.py` imports it)". Verified `_strip_order` is defined at `pass_list.py:28`, used at `:165`, imported at `test_pass_verbs.py:28` and exercised at `:293`. The miss is dissolved by the reduced D12 rather than answered. |
| **MISS D2** — `tabs/document.py` is the only caller of `pass_list.draw`; not in W-D's list | **closed differently** | Same dissolution: D12 no longer replaces the strip, so `pass_list.draw`'s signature and call site are untouched. W-D's change is "`_draw_pass_tile` passes no `sublines`" — internal to `pass_list.py`. Verified `_draw_pass_tile` builds `sublines` locally at `pass_list.py:104–106` and hands them to `preview_cell` at `:117`; `draw()` at `:139` is unchanged by that. Correctly no longer a miss. |
| **MISS D3** — the `u_light_*` / `u_glow_*` prefix collision in unrelated single-pass examples | **closed** | W-D: "The rename is per-file and exact-token, limited to the two multi-pass examples: unrelated single-pass examples declare `u_light_*` (`8d454b7b…/passes/main.frag.glsl:56-60`) and `u_glow_*` (`0b0d16bb…:12-13`) uniforms whose values persist in their `document.json`; a prefix-blind replace corrupts them. Verify by loading every example headlessly after the rename (the resolve-clean example test)." Both the scoping rule and a verification step. |
| **`Document.render` signature** — the claim is not expressible today; `target` must substitute in the planner **and** in the two `output == name` comparisons | **closed** | W-C: "`Document.render` gains `target: str \| None = None` — it substitutes for `graph.output_pass` in `plan_for_output` AND in the two `name == output` comparisons (the full-size exemption and the external-canvas rule: a first-render target sizes by its own scale like an intermediate and never receives an external canvas)." Both comparisons named, with the intended semantics for each. `document.py` is in W-C's Files line. |
| **Ctrl+A / Ctrl+Z / Ctrl+Y note** — the standard keymap's Ctrl surface is larger than the spec's two "known moves"; the table must carry those cells even though no app chord collides | **closed** | W-E: "the standard keymap also owns Ctrl+A / Ctrl+Z / Ctrl+Y / Ctrl+Shift+Z, which no app chord uses today — the table records the editor as owner so nothing lands on them later." Exactly the round-1 recommendation, including the "no app collision" framing. |

**Task 1 verdict: PASS.** All 22 enumerated round-1 findings are closed by quoted spec text —
seventeen as proposed, five closed differently (C2 by upholding 065's exclusion rather than
superseding it; MISS D1/D2 dissolved by the reduced D12; the 068-D7-lift note by co-location;
#20 restated positively). Four round-1 asides were not folded and remain open, all named above:
the raw-`file:line` convention violation (now larger, five cites added), the "no uniform value is a
dict" invariant absent from D3, D1's silence on the vim furniture strings, and — new — `reload` as
the fourth `EngineNode`-taking method still unnamed.

---

## Task 2 — fresh coverage on the patched spec

| # | workstream | covered |
|---|---|---|
| 1 | W-A + W-H | yes |
| 2 | W-A | yes |
| 3 | W-A | yes |
| 4 | W-A + Out of scope (absolute per-pass sizes, trigger named) | yes |
| 5 | W-B | yes |
| 6 | W-H | yes |
| 7 | W-B | yes |
| 8 | W-H | yes |
| 9 | W-C | yes |
| 10 | W-B | yes |
| 11 | W-F | yes |
| 12 | W-F | yes |
| 13 | W-E | yes |
| 14 | W-F | yes |
| 15 | W-F | yes |
| 16 | W-F | yes |
| 17 | W-C | yes |
| 18 | W-C | yes |
| 19 | W-D (strip tune) + Out of scope → 070 | **partial** |
| 20 | W-H | yes |
| 21 | W-A | yes |
| 22 | W-G | yes |
| 23 | W-G + Out of scope (`ctx.keys`, trigger named) | yes |
| 24 | W-E | yes |
| 25 | W-C | yes |
| 26 | W-E | yes |
| 27 | W-H | yes |
| 28 | W-C | yes |
| 29 | W-G | yes |
| 30 | W-G + D3 | yes |
| 31 | W-H | yes |
| 32 | W-B | yes |
| 33 | W-H | yes |
| 34 | W-H | yes |
| 35 | W-H | yes |
| 36 | W-C | yes |
| 37 | W-D + D9 | yes |

**36 of 37 fully covered; one partial (#19).** The three round-1 partials — #6, #20, #22 — are now
yes, by the text quoted in Task 1.

### #19 under the reduced D12 — honestly covered as partial

The maintainer's words: *"The input mappings don't fit the card ('u_prev <- …' gets cut). The `<-`
reads as a cheap workaround. We need a better representation: visualise the passes as a directed
graph — smaller, clean square previews with a small pre-render, connected with arrows. **Or at
least tune the current visuals, it is awful.**"*

That is two asks with an explicit fallback. What the spec ships against each:

1. **The truncation** — closed. W-D: "`_draw_pass_tile` passes no `sublines`; the 'has compile
   errors' subline becomes the error border it already has." Verified the mechanism: `pass_list.py`
   builds `sublines = [f"{uniform} <- {src}" …]` at `:104`, appends `"has compile errors"` at `:106`,
   and hands them to `preview_cell(… sublines=sublines …)` at `:117`; `preview_cell` ellipsizes each
   to the cell width. Dropping the argument removes the truncation *and* the `<-` the maintainer
   called a cheap workaround, in one edit. The error subline's replacement already exists — the
   `border_color` passed at `:113` from `errors`. So this half is a real fix, not a deletion that
   loses information.
2. **The graph view** — deferred to 070, with the direction fixed rather than left open: "thumbnails
   as nodes in evaluation order, edges labelled by pass name under D9, feedback as a loop mark,
   read-only, draw-list only — not `imgui_node_editor`." That is the ledger's option A verbatim,
   preserved so 070 does not re-derive it.
3. **"Or at least tune the current visuals"** — this is the half the reduction rests on, and it is
   **thinner than the word "tune" implies.** What W-D ships is exactly one change: sublines removed.
   `SIZE.PASS_THUMB` is explicitly "unchanged". Nothing in W-D touches the tile's footer, spacing,
   selection, stale wash, or the 112px wrapping strip that the ledger's #19 row describes as the
   thing being complained about. The maintainer said the current visuals are "awful"; the spec's
   answer to "tune them" is to delete the one element that was truncating. That is defensible —
   the truncation *is* what he pointed at first, and D12 is stamped "Maintainer's call" twice — but
   a reader of the spec alone cannot tell whether "at least tune" was satisfied or merely
   intersected. **This is why #19 stays partial**, and it is the row a later round should not treat
   as fully closed.

Two further checks under the reduced D12, both clean:

- **No leftover claims the graph view ships in 069.** Grepped the spec: `pass_graph_view.py` appears
  nowhere; the four places D12 touches (Out of scope, D12 itself, W-D's heading + body, Open
  question 3, the W-D manual-verification line "the strip shows name + thumbnail, nothing
  truncated") all agree.
- **The reduction does not orphan anything.** Round 1's MISS D1 and D2 both existed only because
  W-D was replacing the strip; with `pass_list.py` surviving, `_strip_order` and `pass_list.draw`'s
  single call site in `tabs/document.py` are untouched. Verified both.

**Task 2 verdict: PARTIAL** — 36/37 yes, #19 partial for the reason above (the deferral is the
maintainer's own call and correctly recorded; what is thin is the "tune" half, which reduces to a
single deletion).

---

## Task 3 — citation truth for what was added since round 1

Round 1 verified the pre-existing cites (its False-trails section covers them). Below is every
file:line and named symbol the patch ADDED. Each was opened.

| cite (added) | verdict | what is actually there |
|---|---|---|
| `ui.py:553` | **true** | `window_flags=imgui.WindowFlags_.no_nav_inputs,` inside `begin_child("copilot_bar", …)`. Unconditional, not a ternary — the spec's "the copilot bar child at `:553`" names the right site and the right kind. |
| `copilot_chat.py:53` | **true** | `\| imgui.WindowFlags_.no_nav_inputs` inside `_WINDOW_FLAGS`. Round 1 cited `:46` (the explaining comment); `:53` is the flag itself, which is the more precise cite. The comment at `:45–47` says the flag "stops the nav outline on the programmatically-focused input", consistent with the spec's "with nav off the flag is inert everywhere". |
| "all FOUR `no_nav_inputs` sites" | **true** | Grep returns exactly four flag uses: `ui.py:379`, `ui.py:553`, `ui.py:709`, `copilot_chat.py:53`. (Two further hits are prose: `app.py:184`'s comment and `copilot_chat.py:46`'s comment.) The count is right. |
| `popups/examples.py:101` | **true** | `nav_flatten=True,` in the `draw_document_preview_button(…)` call inside the example-grid loop. |
| "`nav_flatten` … `widgets/document_grid.py` ×2" | **true** | `document_grid.py:21` (the parameter), `:33` (the forward), `:102` (`nav_flatten=True`). The "×2" counts the two places the module names it as an argument (`:33` forward, `:102` literal); the parameter declaration at `:21` is the third mention. Slightly loose as a count, unambiguous as an instruction. |
| "`preview_cell`'s `nav_flatten` parameter and docstring paragraph (`ui_primitives.py`)" | **true** | `ui_primitives.py:953` `nav_flatten: bool = False,`; `:965` the docstring paragraph; `:989–990` the `ChildFlags_.nav_flattened` application. |
| "`tabs/document.py`'s `ChildFlags_.nav_flattened` plus its now-false comment" | **true** | `tabs/document.py:168` the comment "nav_flattened: Tab/arrows reach the sliders without an Enter/Esc window boundary."; `:173` `child_flags=imgui.ChildFlags_.nav_flattened \| imgui.ChildFlags_.auto_resize_y`. The comment does become false. |
| `8d454b7b…/passes/main.frag.glsl:56-60` (`u_light_*`) | **true** | Lines 56–60 exactly: `u_light_ambient`, `u_light_sky_key`, `u_light_moon_key`, `u_light_cool_color`, `u_light_warm_color`. Five uniforms, none a sampler, none an input edge. |
| `0b0d16bb…:12-13` (`u_glow_*`) | **true** | Lines 12–13: `uniform float u_glow_strength = 0.79;` and `uniform float u_glow_radius = 1.73;`. |
| `scripting/api_doc.py` | **true** | Exists. `:18` imports `EXPORT_MOUSE, EngineContext, MouseState`; `:61` derives `_MOUSE_FIELDS` from `MouseState.__dataclass_fields__`; `:62` `_EXPORT_MOUSE_AT`; `:72` splices both into the prompt prose. Adding `down`/`prev_x`/`prev_y` changes the generated block, as the spec says. |
| `copilot/prompt_context.py` | **true** | `:9` `from shaderbox.scripting.api_doc import script_api_summary`; `:45` the `script_api` field; `:99` `script_api=script_api_summary()`. Sole importer, as claimed. |
| `scripting/__init__.py` re-exports `EngineNode` | **true** | `:13` in the import list, `:30` in `__all__`. |
| `widgets/uniform.py` play/stop button | **true** | `:169` `app.set_uniform_stopped(document_id, name, playing)` and `:302` the same with `True` — both name-only. `:177` already computes `panel_pass = app.panel_pass(app.current_document_id)`, so "passes the panel pass" is a small local change, exactly as scoped. `:188` `app.session.is_uniform_stopped(document_id, name)` is the third name-only site. |
| `Document.first_render_done` (as the thing `Pass.first_render_done` mirrors) | **true** | `document.py:238` `self.first_render_done: bool = False`, set at `:393`. Nothing named `first_render_done` in `core.py` — so `Pass.first_render_done` is genuinely new, and W-C listing `core.py` is right. |
| "the five shipped examples' `document.json` files that persist `"stopped_uniforms": []`" | **true** | Exactly five, and they are the five round 1 enumerated: `53724dbd…`, `73ea2431…`, `8d454b7b…`, `0b0d16bb…`, `f90f5ff9…`. Count and set both correct. |
| `_scriptable_uniforms_for` "reads `.render_pass` only" | **true** | `project_session.py:632` the def, `:640` `].document.render_pass.get_active_uniforms()`. Consumed by `script_stub_for` at `:652` and `:669`. |
| `copilot/backend.py`'s `_get_script_driven_uniforms` gate on `set_uniform` | **true** | `:408` the injected callable; the `set_uniform` gate at `:881` (`if name in self._get_script_driven_uniforms(document_id)`), name-only as the spec says. Three other call sites at `:640`, `:723`, `:740`. |
| `ScriptEngine.tick` / `tick_export` / `dry_run` | **true** | `engine.py:370`, `:397`, `:421`. All three exist with the `EngineNode` shape. |
| `_strip_order` "stays; `tests/test_pass_verbs.py` imports it" | **true** | Defined `pass_list.py:28`, used `:165`; imported `test_pass_verbs.py:28`, exercised at `:293` inside `test_the_strip_order_is_topological_and_independent_of_the_output`. |
| "`_draw_pass_tile` passes no `sublines`" is a real change | **true** | `pass_list.py:104` builds them, `:106` appends the error one, `:117` `sublines=sublines`. `preview_cell`'s parameter is defaulted, so dropping the argument is the whole edit. |
| "the 'has compile errors' subline becomes the error border it already has" | **true** | `pass_list.py:113` already passes `border_color=border`, computed from `errors`. The border genuinely pre-exists. |
| `conventions.md` code-editor entry + its revisit trigger | **true** | `:360` "opened FILE.** The code editor is the maintainer's own vim-modal library (feature 067):"; `:374–375` "The vendored binary + rebuild procedure live in `## Known quirks`."; `:375–378` "Revisit if … a 4th editable `kind` lands, or a non-modal keymap ships editor-side." The trigger is worded as the spec quotes it, and `## Known quirks` exists at `:744`. |
| "the shipped `iterations` becomes 11" (today 9) | **true** | The RC example's `graph.json` has `"jfa": { … "iterations": 9 }`; `cascade` has 6, everything else 1. The spec's premise (9 today, 11 needed through 2048) matches the file. |
| "the largest W-A preset" = 2048 | **true** | W-A's preset list is "256, 512, 1024, 2048"; `ceil(log2(2048)) = 11`. The arithmetic behind "11 (correct through 2048)" checks out. |

**Task 3 verdict: PASS.** Every citation added since round 1 — 24 of them, spanning 13 files —
resolves to what the spec says. Nothing is wrongly cited. One cite is loose rather than wrong
(`document_grid.py` "×2" counts argument uses, not the three mentions of the name); it cannot
mislead an implementer.

The standing objection is not truth but form: the patch **added five more raw `file:line` cites**
(`ui.py:553`, `copilot_chat.py:53`, `popups/examples.py:101`, `8d454b7b…:56-60`, `0b0d16bb…:12-13`)
to a spec already carrying eight, against `conventions.md ## Code rules`' "No raw line numbers …
Cite the **symbol** instead of a line." Every one is accurate today. Two of them
(`conventions.md:328`, `conventions.md:292–300`, both pre-existing) rot inside this feature.

---

## Still open after round 2

1. **#19's "at least tune the current visuals" half** is answered by one deletion (sublines) with
   `SIZE.PASS_THUMB` explicitly unchanged. Defensible as the maintainer's own call; thin as a
   response to "it is awful". Not a blocker — a row a later round should not read as fully closed.
2. **Raw `file:line` cites** — thirteen now, five added by this patch, against the repo's own rule.
   Convert to symbol cites, or the spec instructs edits (W-E and W-G both edit `conventions.md`)
   that invalidate its own citations.
3. **D3 lacks the "no uniform value is a dict" invariant** that makes value-type dispatch
   unambiguous. Round 1 asked for it; the patch did not add it.
4. **`ScriptEngine.reload` is the fourth `EngineNode`-taking method** and is still unnamed in W-G
   (verified: `engine.py:281 def reload(self, document_id, scripts_dir, document: EngineNode)`).
   W-G names `tick` / `tick_export` / `dry_run` and lists the `project_session.py` callers, which
   includes `write_script_source`'s "reload + dry_run" — so `reload` is reachable through the caller
   list but not through the method list. Smallest of the four.
5. **D1's gate cannot see the vim status-line strings** (mode badge, ruler, drawn on the editor's
   cell grid, not through `help_marker` / `set_tooltip` / `separator_text` / the `FG_DIM` idiom).
   Fine in substance — they are furniture, not prose — but unsaid, so it reads as an escape hatch.

None of the five blocks implementation. Items 3 and 4 are one sentence each in W-G.

---

## False trails — probed this round, fine, do not re-check

- **The four `no_nav_inputs` sites are genuinely all removable.** Round 1 coverage read
  `copilot_chat.py`'s flag as one that "stays if it guards something else" and concluded it stays;
  the patched spec removes all four. I opened the comment at `copilot_chat.py:45–47`: the flag
  suppresses a nav *outline* on a programmatically-focused input, which only exists while
  `nav_enable_keyboard` is on. With nav off it is inert. The spec's reading is correct and
  supersedes round 1's on this point. Settled — do not re-open.
- **MISS D1 and D2 are dissolved, not skipped.** Both existed only because W-D was replacing the
  strip with `pass_graph_view.py`. With the reduced D12, `pass_list.py` survives intact:
  `_strip_order` (defined `:28`, used `:165`, imported by `test_pass_verbs.py:28`) and
  `pass_list.draw`'s single call site in `tabs/document.py` are untouched. Verified both. No file
  list needs them.
- **`pass_graph_view.py` appears nowhere in the patched spec.** Grepped. The D12 reduction left no
  residue asserting the graph view ships in 069.
- **The #6 formula is self-consistent and its ordering worry is gone.** `ceil(log2(2048)) = 11`
  matches the spec's "11 (correct through 2048, the largest W-A preset)", and the shader change
  lands in W-H, which already runs last — so round 1's "the fix must precede W-H" concern cannot
  fire. No ordering edit needed.
- **C2 needs no second 065 doc line.** 065 excluded a canvas graph UI as "a separate feature"; 070
  *is* that separate feature. The exclusion is upheld, not superseded, so only 065 D12-the-script-
  decision needs its supersession line — which W-G gives it. Do not re-file C2.
- **The five example `document.json`s carrying `stopped_uniforms` are exactly the five round 1
  named.** Re-enumerated by grep: `53724dbd…`, `73ea2431…`, `8d454b7b…`, `0b0d16bb…`, `f90f5ff9…`.
  Count and identity both hold; the hand-edit scope is right.
- **`widgets/uniform.py` already has `panel_pass` in scope** at `:177`, so W-G's "passes the panel
  pass" is a local change at three call sites (`:169`, `:188`, `:302`), not new plumbing.
- **The `u_light_*` / `u_glow_*` prefix trap is real and exactly as scoped.** Opened both shaders:
  five `u_light_*` floats/vec3s at `8d454b7b…:56–60`, two `u_glow_*` floats at `0b0d16bb…:12–13`.
  None is a `sampler2D`; none is an input edge. W-D's "per-file and exact-token, limited to the two
  multi-pass examples" is the correct scoping and needs no further probing.
