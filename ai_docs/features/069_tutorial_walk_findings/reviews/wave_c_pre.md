# W-C pre-implementation review (round 1)

Reviewer role per `dev_flow.md` step 4: correctness & design, verification & blast-radius.
Artifact: `10_wave_c_pass_verbs.md`. Anchors: `01_spec.md § W-C / § Locked decisions / § Order /
§ Manual verification / § Open questions / § Review history`, `00_findings.md` #9 #17 #18 #25 #28
#36, and the tree at `faccf0e`.

## Verdicts

- **Parent-bullet coverage: PASS.** Every W-C bullet has a numbered decision that implements it.
- **Locked-decision fidelity: PASS.** D10 and D11 are implemented as written; D11's stated scope
  ("every § 7.5 inline input") is honoured by a reasoned in/out list, not by silence.
- **Sweep-algorithm correctness: PASS.** Hand-traced all three scenarios; no double-draw, no skip,
  no extra feedback advance. Termination proved.
- **Test falsifiability: PARTIAL.** Six of nine named tests are falsifiable as described. Two are
  not (findings 3 and 4), and one names a falsifier its own rig cannot produce (finding 5).

Two findings are correctness defects (1, 2). The rest are gaps in the spec's own bookkeeping.

---

## Findings

### 1. The commit-on-deactivate scope list omits seven live `input_text` call sites (MEDIUM)

**Claim.** § Design decisions item 6 says it enumerates "Every `input_text` /
`input_text_multiline` call site in `shaderbox/`, from `grep -rn 'input_text' shaderbox/
--include=*.py`". It does not. It correctly identifies `ui_primitives.py`'s two forwarders and
says "their CALLERS are the measured sites" — but then measures only two of the seven callers,
and names them by fields that are not the forwarded ones.

**Evidence.** My own enumeration, run fresh:

```
$ grep -rn "labeled_text_input(\|labeled_multiline_input(" shaderbox/ --include=*.py \
    | grep -v ^shaderbox/ui_primitives.py | grep -v import
shaderbox/popups/settings.py:282:    new_key = labeled_text_input(
shaderbox/popups/settings.py:285:    new_model = labeled_text_input("Model", cfg.model, field_w)
shaderbox/exporters/youtube.py:427:        rs.title = labeled_text_input("Title", rs.title, field_w)
shaderbox/exporters/youtube.py:428:        rs.description = labeled_multiline_input(
shaderbox/exporters/youtube.py:431:        rs.tags_raw = labeled_text_input("Tags (comma-separated)", rs.tags_raw, field_w)
shaderbox/widgets/copilot_chat.py:639:    msg.gate_input = labeled_text_input(
shaderbox/exporters/telegram.py:371:            self._tg.bot_token = labeled_text_input(
```

Cross-checked against the spec's out-of-scope table:

- `exporters/youtube.py` row says "(paste box)" — that is `youtube.py:324`
  (`imgui.input_text_multiline`), a different widget from `:427/:428/:431`.
- `exporters/telegram.py` row says "(new pack title)" — that is `telegram.py:470`, not `:371`
  (`bot_token`).
- `popups/settings.py:282` (the Anthropic API key) and `:285` (the model) appear **nowhere** in
  the spec, in scope or out.
- `widgets/copilot_chat.py:639` (`msg.gate_input`, the mid-turn secret gate) appears nowhere. The
  spec's `copilot_chat.py` row is about `:297`, the chat input, and its stated reason ("Enter
  SENDS") does not apply to `:639` at all.

Direct `input_text` sites total 15 (16 grep hits minus `app.py:179`, which is
`config_input_text_cursor_blink`, not a call). Add the seven derived sites and the real domain is
22. The spec's header says "10 sites" out of scope and then lists 11 rows. A checker that quietly
narrows its own domain is exactly the shape `conventions.md` warns about; the same applies to a
spec that claims an exhaustive census and delivers a partial one.

The two unnamed sites are also the two that matter most under D11: `settings.py:282` is a
credential field whose Close path is `settings.py:64`, and `copilot_chat.py:639` gates a live
turn. Neither is a "live filter" or a "per-keystroke value", so neither is covered by the closing
paragraph's two named exemptions.

**Fix (paste into item 6).** Replace the census sentence with the real one and add the missing
rows:

> Every `input_text` / `input_text_multiline` call site in `shaderbox/` — 15 direct calls plus the
> 7 callers of `ui_primitives.labeled_text_input` / `labeled_multiline_input`, which are the
> measured sites for those two forwarders. Out of scope, additionally:
> `popups/settings.py::_draw_copilot_section` (API key, model) — a settings field whose value is
> applied at the Settings modal's own close funnel (`popups/settings.py`'s close branch +
> `app.apply_editor_settings`), so a deactivate commit would duplicate an existing funnel rather
> than add one; `widgets/copilot_chat.py::_draw_gate` (`msg.gate_input`) — the gate has its own
> explicit submit button and its value is a secret the user may still be typing, so committing on
> click-away would send a partial credential; `exporters/youtube.py` Title / Description / Tags and
> `exporters/telegram.py` bot token — export-panel fields, same reason as the two rows already
> listed.

### 2. The Close button now needs two clicks after an un-entered rename (LOW, but it is the walk's own step)

**Claim.** Items 1 and 2 interact: clicking Close moves focus off the name field, so on that same
frame `_draw_name`'s `is_item_deactivated_after_edit()` is True, `_commit_pass_name` renames,
`_draw_name` returns True, and `_draw_body` returns True **before** `ghost_button("Close")` is
ever submitted. The modal stays open. The user must click Close a second time.

**Evidence.** `pass_settings.py:89` is `return not ghost_button("Close")` — the last statement of
the body, after `_draw_inputs` (`:84`) and `_draw_target` (`:86`). The spec's item 1 places the
early return at `:81`, above all of them. So on the commit frame the Close button is not drawn,
and imgui cannot report a click on a widget that was not submitted.

The same interaction has a second, worse face: the click that lands on the Close button's screen
position on frame N is consumed by the deactivate, and on frame N+1 the button is drawn under the
cursor with the mouse already released — so nothing fires until the user clicks again. This is the
exact manual-verification step 1 shape ("type, click elsewhere") aimed at the Close button.

**Fix (paste into item 1).** Add after the "legal imgui frame" paragraph:

> One interaction with item 3: clicking Close while the name field holds an un-entered edit fires
> the deactivate commit first, so the body returns before the Close button is submitted and the
> modal survives one extra frame. That is acceptable (the rename is what the user asked for and it
> landed), but the Close click itself is lost. The implementation keeps the Close row reachable by
> making the early return skip only the two sections that index `document.passes` by the dead
> name: `_draw_inputs` and `_draw_target`. Concretely, `_draw_body` becomes
> `renamed = _draw_name(...)`, then `if not renamed:` guards the two `imgui.dummy` + section
> pairs, and the `return not ghost_button("Close")` line runs on every path. The frame then emits
> `separator_text` + name row + Close, which is still balanced, and one Close click still closes.

(If the spec prefers the plain early return, say so and drop manual step 1's "the modal stays
open" from a pass condition to an observation — but then add a step that clicks Close after an
un-entered edit and expects two clicks, so the behaviour is pinned rather than discovered.)

### 3. `test_the_gear_body_survives_a_rename_mid_frame` is falsifiable, but its stated rig is not what it describes (LOW)

**Claim + evidence.** I ran the drafter's rig and the crash reproduces verbatim:

```
File ".../shaderbox/popups/pass_settings.py", line 84, in _draw_body
    _draw_inputs(app, document_id, name, document.passes[name])
KeyError: 'main'
```

So the falsifier is real and the test goes red today. Confirmed also that the frame closes
cleanly after the exception, and that the proposed early-return body (`separator_text` +
`label_row` + `input_text` + `same_line` + `help_marker`, then return) runs three consecutive
frames with no assert — so item 1's "legal imgui frame" claim is verified by execution, not by
argument.

The gap is elsewhere: the test as described monkeypatches `_draw_name` to a stub that calls
`rename_pass`. **A stub that replaces `_draw_name` cannot observe whether `_draw_name` returns
`bool` at all.** After the fix, a `_draw_name` that renames and returns `None` (falsy) would leave
the body continuing — and the test would still be red, so it does catch that. But a `_draw_name`
that returns `True` unconditionally (never renaming) also passes, because the stub supplies the
rename. The test pins `_draw_body`'s early-return branch and nothing about `_draw_name`.

**Fix (paste into the test's section).** Add:

> The stub deliberately supplies the rename, so this test pins `_draw_body`'s branch only.
> `_draw_name`'s own return value is pinned separately by
> `test_a_rejected_rename_snaps_the_buffer_back` (False on reject) and by a third assertion in
> that same test: `_commit_pass_name` returns True for an accepted rename.

### 4. `test_the_steady_state_draws_only_the_output_chain` cannot see its first stated falsifier (MEDIUM)

**Claim.** The test's falsifier list says "an unscoped skip makes the count zero for the own-canvas
render (the thumbnail-blanking bug)". A test that runs `document.render()` and
`document.render(target=...)` by hand does **not** reproduce the two-output-renders-per-frame
shape that the unscoped skip breaks — the blanking needs the preview render at `ui.py:265` AND
the own-canvas render at `ui.py:296` in the same frame.

**Evidence.** `ui.py:265` is `ui_document.document.render(canvas=app.preview_canvas)`, run
unconditionally for the current document; `ui.py:296` is `ui_document.document.render()`, inside
`if not app.any_popup_open():`. Both draw the same chain each frame. Under an unscoped skip
(`drawn_frame == self._frame` with no `target is not None` conjunct), the second call finds every
pass stamped and draws nothing — every thumbnail freezes on frame 0's picture. A test whose frame
loop calls `render()` once cannot observe that: with one output render per frame there is nothing
for the skip to suppress.

The test's second falsifier (a missing skip making an ancestor draw twice) **is** observable, so
the test is not worthless — it is over-claimed.

**Fix (paste into the test's section).** Replace the falsifier paragraph with:

> **Falsifier:** a missing skip makes an ancestor of the elected pass draw twice in one frame,
> which the per-pass counter sees directly. The unscoped-skip regression (every thumbnail blanking)
> is NOT observable here — it needs the two output renders per frame that only `ui.py` issues
> (`:265` preview + `:296` own canvas). A separate test pins it: call
> `document.render(canvas=foreign)`, then `document.render()`, then `document.render(target=x)`
> inside one `begin_frame`, and assert the second call drew the full chain. Name it
> `test_two_output_renders_in_one_frame_both_draw`; without the `target is not None` conjunct the
> second call draws nothing and the counter assertion goes red. This is manual step 9's headless
> half, and it is the single most likely way this wave breaks something.

### 5. `test_every_pass_renders_once_within_n_frames` states a falsifier its rig cannot produce (LOW)

**Claim.** The falsifier reads "or with the sweep electing the same pass repeatedly because the
stamp is written only on a successful compile". The test as described writes the sweep loop **by
hand** in the test body, electing `next(p for p in passes if not p.first_render_done)`. Where the
stamp is written is a property of `Document.render`, so the "stamp only on success" mutation IS
observable — but only if the elected pass actually fails to compile. `_BLOOM`'s five passes all
compile.

**Evidence.** `tests/test_lazy_compile.py::test_load_compiles_no_pass` asserts
`len(document.passes) == 5` and that each has `program is None` at load — but they compile on
first render (`core.py:355-359`: `if not self.program ...: self.compile()`, then `if not
self.program or not self.vao: return`). A pass that compiles cleanly is stamped either way, so the
mutation is invisible on `_BLOOM`.

**Fix (paste into the test's section).** Add a second case:

> A second case pins the stamp-on-attempt posture: after loading `_BLOOM`, call
> `release_program("this is not glsl")` on one off-chain pass (the idiom
> `test_a_broken_source_is_attempted_once` already uses), then run the sweep loop. Assert the
> broken pass's `first_render_done` is True within `len(document.passes)` frames and that the
> sweep terminates (no pass is elected twice). Stamping inside `Pass.render` instead would leave it
> at False forever and the loop would re-elect it every frame — the assertion goes red on the
> frame budget.

### 6. The `project_session.py` rename-pop line is off by one (TRIVIAL)

**Claim + evidence.** § Verified premises row says "the `document.passes.pop(old)` line is `:822`".
It is `:821`; `:822` is `old_path = render_pass.source.path`.

```
821:         render_pass = document.passes.pop(old)
822:         old_path = render_pass.source.path
```

Immaterial to the design, and `conventions.md ## Code rules` bans raw line numbers in docs anyway —
but the row exists precisely to correct a line number, so it should be right.

**Fix.** Change `:822` to `:821`, or (better, per the convention) drop the number and cite
`ProjectSession.rename_pass` alone, which the row already does in its second sentence.

### 7. `close_pass_settings` does not honour a name `rename_pass` rejects, and the docstring history (TRIVIAL)

**Claim.** Two small things in item 3's snippet.

**Evidence.** (a) `ProjectSession.rename_pass` returns `""` for `new == old` (`project_session.py:813`),
so the `buf != name` guard is redundant with the callee's own no-op — harmless, keep it as a
cheap short-circuit, but the spec's "idempotent" sentence should say the callee is idempotent too.
(b) The docstring `"""Close the gear, committing a pending rename first (069 D11)."""` names a
feature number in a source docstring. `conventions.md ## Code rules` says a warranted comment
"states what's non-obvious about the code as it is NOW — never narrates development history", and
"if the rationale already lives in its canonical home ... the comment shrinks to a ≤1-line
pointer". A `(069 D11)` tag is a pointer to a spec that will be closed; the same rule fired on the
two `Pass` field comments in item 7, which carry "(069 W-C)" and a three-line story about `-1`.

**Fix.** Trim the docstring to `"""Close the gear, committing a pending rename first."""`, and cut
the `Pass.drawn_frame` comment to one line stating the invariant as it is: `# The document frame
this pass last drew in; -1 means never.` The `first_render_done` field needs no comment at all —
the name says it. The reason the sweep tests `>= 0` belongs in the wave spec (where it already is),
not in `core.py`.

---

## Coverage, item by item

### Parent-spec W-C bullets

| Parent bullet | Verdict | Wave-spec decision |
|---|---|---|
| Rename crash: `_draw_name` returns `renamed: bool`; `_draw_body` returns early; test drives it headlessly through the § 0 rig | **Covered** | Item 1 + `test_the_gear_body_survives_a_rename_mid_frame`. Rig verified by execution. |
| Commit on `is_item_deactivated_after_edit()` + Enter | **Covered** | Item 2. |
| ... and on Close/Escape with a pending edit | **Covered** | Item 3, at the `App.close_pass_settings` funnel, with the `ui.py:336` vs `:429` ordering argument that makes a body-side commit impossible. Verified. |
| Never per keystroke | **Covered** | Item 2's second property, with `_pass_name_error` as the reason. |
| Same rule applied to `_draw_add_input` | **Covered** | Item 6's in-scope table + open question 3. |
| ... and every § 7.5 inline input | **Partially covered** | Item 6 defers the four picker sites by wave boundary with a stated reason (acceptable), but omits seven derived sites entirely — **finding 1**. |
| The skill's § 7.5 rewritten | **Covered** | Item 8. |
| Add pass activates (D10): `add_pass` → tile-click path → `open_pass_settings` | **Covered** | Item 4. |
| Copilot has no pass tools — nothing to mirror | **Covered** | Item 4's grep; I re-ran it, no hits. |
| `OPEN_PASS_SETTINGS` Alt+P, `ADD_PASS` Alt+A, dispatch, chord test + Help table pick them up | **Covered** | Item 5. |
| First render (#36 A), decision (1): `target` or graph output feeds guard / planner / cycle fallback — all three | **Covered** | Item 7's `resolved`. All three sites confirmed at `document.py:395`, `:398`, `:400-403`. |
| Decision (2): a target draws its WHOLE ancestor chain | **Covered** | Item 7, resting on `pass_graph.py::_order_for`'s transitive stack walk (`:384-392`). Confirmed. |
| Decision (2): ONLY a target render skips passes already drawn this frame | **Covered** | Item 7's `canvas is None and target is not None` conjunct pair. This is round 3's fix and the wave spec does not regress it — see § Round-3 fidelity below. |
| `Pass.drawn_frame: int`, set by every render against `_frame` | **Covered, corrected** | Item 7 moves the setter from `Pass.render` to `Document.render` because `Pass.render` returns early at `core.py:358-359` when compilation fails. Confirmed at that line; the correction is right and the spec states why. |
| Output renders never skip | **Covered** | Item 7. The parent's `ui.py:301` cite is corrected to `:296`; I confirmed `:296` is `ui_document.document.render()` and `:301` is inside the examples generator. |
| Target renders issued only where `_frame` is defined | **Covered** | Item 7 + the `self._frame >= 0` conjunct. Confirmed `begin_frame` call sites are `ui.py:246`, `document.py:527`, `document.py:619` only. |
| Decision (3): target pass sizes by its own scale, never gets the external canvas | **Covered** | Item 7's `output` binding kept separate from `resolved`, feeding `document.py:410` and `:425`. |
| Decision (4): `Pass.first_render_done` set by render on every pass it draws | **Covered** | Item 7's second stamp. |
| Decision (4): `Document.first_render_done` = `canvas is None and target is None` | **Covered** | Item 7's narrowing; strictly narrower than today's `canvas is None` at `document.py:392-393`, so `test_a_foreign_canvas_render_leaves_first_render_pending` is unaffected. Confirmed by reading that test. |
| `ui.py`'s frame gate draws at most one pending pass per frame | **Covered** | Item 7's `next(...)` scan. |
| Export untouched | **Covered** | Item 7's last paragraph. Confirmed `_render_image` (`:520`) and `_render_media_into` (`:668`) pass `canvas=` only, and `render_media` calls `reset_feedback()` which sets `_frame = -1`. |
| The stale wash then means "was live, is no longer" | **Covered** | Item 7's "stale wash keeps its meaning, narrowed". |
| Test: every pass drawn exactly once across the first N frames | **Covered** | `test_every_pass_renders_once_within_n_frames` (see finding 5). |
| Test: steady state draws only the output chain; the `pass_graph.py` assert stays the guard | **Covered, over-claimed** | `test_the_steady_state_draws_only_the_output_chain` — finding 4. `assert_plan_invariants` runs inside `plan_for_output` (`pass_graph.py:363`) which every render path calls, so the target render is asserted on the same terms. Confirmed. |
| Files: `pass_settings.py`, `pass_list.py`, `commands.py`, `app.py`, `ui.py`, `core.py`, `document.py`, tests | **Covered** | The wave spec's Files table is a superset (adds `hotkeys.py` for the Escape funnel and the skill file, both required by the parent's own "commit on Close/Escape" and "§ 7.5 rewritten" bullets). |

### Locked decisions

**D10 — "Add pass activates the pass (opens its tab, makes it the output), then opens the gear (#28)."**
Implemented as written by item 4: `open_pass(name)` → `set_output_pass` → `open_pass_settings(name)`,
in that order, with the order's reason stated. I read the tile-click path it copies:

```
123:     if result.clicked:
126:         open_pass(name)
127:         if not is_output:
128:             error = app.session.set_output_pass(document_id, name)
```

The spec's "minus the `if not is_output` guard, which is vacuous for a pass created this frame" is
correct: `add_pass` (`project_session.py:762-785`) inserts into `document.passes` and
`graph.passes` but never touches `graph.output`, so a fresh pass is never already the output.
**Verdict: implemented as written.**

**D11 — "Inline inputs commit on deactivate-after-edit, Enter is a shortcut (#18) — rule filed into
the imgui skill § 7.5 with the wave."**
Implemented by items 2, 3, 6 and 8. Two mechanical points I checked rather than assumed:

- With `enter_returns_true`, imgui deactivates the item on Enter, so on the Enter frame both
  `committed` and `is_item_deactivated_after_edit()` are True. The spec's single
  `if committed or ...:` handles that with no double-commit. Correct.
- The spec's "read the deactivate query on the line immediately after `input_text`, before the
  `same_line` / `help_marker`" is right and load-bearing: the item-scoped queries read the last
  submitted item.

§ 7.5's Pattern bullet today reads exactly what the spec quotes
(`"imgui.input_text(..., flags=enter_returns_true)" — Enter commits, Esc cancels.`). Item 8's
rewrite keeps Focus / Outer-keyboard-suppression / Auto-expand untouched, which is right — none of
them is about commit. **Verdict: implemented as written**, with the scope gap of finding 1.

### Round-3 fidelity: the target-only skip

`01_spec.md § Review history` records round 3 folding "the once-per-frame skip is scoped to target
renders only (unscoped it blanked every thumbnail)". The wave spec keeps that scoping and makes it
the FIRST of its four conjuncts, with the blanking mechanism spelled out (`ui.py:265` +
`ui.py:296` both draw the chain in one frame). **No regression.** The one thing round 3's fix
lacked and this wave adds — `self._frame >= 0` — is a genuine strengthening, not a change of
meaning.

---

## The sweep, traced by hand

I simulated the wave spec's algorithm exactly (skip conjuncts, stamp placement, the `ui.py` scan
placed after `document.render()`), including the preview render at `ui.py:265` which runs for the
current document before the block the sweep lives in. Frame sequence per frame:
`begin_frame(f)` → `render(canvas=preview)` → `render()` → scan → `render(target=pending)`.

**(a) Six-pass chain a→b→c→d→e→f, output `f` (the LAST pass).**

| Frame | Elected | Draws, in order |
|---|---|---|
| 0 | none | preview: a b c d e f; own: a b c d e f |
| 1 | none | preview: a b c d e f; own: a b c d e f |
| 2 | none | preview: a b c d e f; own: a b c d e f |

Every pass is stamped `first_render_done = True` by the preview render on frame 0, so the scan
returns `None` on frame 0 and forever after. **The sweep draws nothing extra, ever** — which is
the required behaviour and also the answer to `01_spec.md § Open questions` item 2 ("if black tiles
persist with the LAST pass as output, that is a new finding"): under this algorithm they cannot.

**(b) Output is the FIRST pass `a`; five isolated passes b–f beside it.**

| Frame | Elected | Draws |
|---|---|---|
| 0 | b | preview: a; own: a; sweep: b |
| 1 | c | preview: a; own: a; sweep: c |
| 2 | d | preview: a; own: a; sweep: d |
| 3 | e | preview: a; own: a; sweep: e |
| 4 | f | preview: a; own: a; sweep: f |
| 5 | none | preview: a; own: a |
| 6+ | none | preview: a; own: a |

One pass per frame, each exactly once, steady state at frame 5 draws only the output chain.
Matches the spec's claim exactly. (With ancestors, the sweep also draws the elected pass's chain —
minus everything the output render already stamped, which is what makes the cost bounded.)

**(c) An iterated feedback pass on the output chain that is also swept in the same frame.**
Graph: `fb` (iterations 4, reads itself) → `comp` (output); `z` off-chain.

| Frame | Elected | Draws | `fb` advances |
|---|---|---|---|
| 0 | z | preview: fb×4, comp; own: fb×4, comp; sweep: z | 8 |
| 1 | none | preview: fb×4, comp; own: fb×4, comp | 8 |
| 2+ | none | preview: fb×4, comp; own: fb×4, comp | 8 |

**`fb` is never elected**, because the output render stamps `first_render_done = True` on it
before the `next(...)` scan runs — the scenario as posed is unreachable under the spec's ordering,
which is itself the guarantee. The 8 iterations per frame are pre-existing (two output renders per
frame, each running the iteration loop) and the sweep adds zero. Confirmed also the sibling case
where the iterated pass is OFF-chain and IS elected: it advances 4 times on the election frame and
never again, which is exactly once.

**Termination.** `first_render_done` is set on attempt, never cleared by the sweep, so each pass is
elected at most once and the scan drains. A pass on a cycle is still stamped: `_order_for` returns
`[]` for it (`pass_graph.py:380-383`), `document.py:400-403`'s fallback sets `order = [resolved]`,
and the loop stamps it. A pass whose compile fails is stamped because `Document.render` stamps
before calling `Pass.render`. **No non-terminating case found.**

**Verdict: no double-draw, no skip, no extra advance in any of the three scenarios. PASS.**

---

## The `draw_into` rename

**The shadow exists.** `document.py:425`:

```
425:                 target = canvas if (name == output and last) else None
```

inside `for iteration in range(entry.iterations):` (`:423`), inside `for name in order:` (`:404`).
A parameter named `target` on `render` would be rebound on the first iteration of the first pass
and every later read would silently be the wrong value — including, critically, the skip check at
the top of the loop body on the SECOND and later passes.

**Verdict: legitimate, and not a contradiction of the parent.** The parent's decisions are about
semantics: what `target` MEANS (the requested pass), which sites read it, and what stays on the
graph output. Renaming the local that shadows it changes no semantics, and the wave spec preserves
the parameter's name `target` — which is what the parent, the tests and the call sites speak. The
name `draw_into` is also the better one for what the local holds (the external canvas or None) and
matches the gear's own "Draws into" section (`pass_settings.py:163`). No finding.

---

## The rename crash: plumbing verified

**The `KeyError` is prevented.** The return-value plumbing (item 1) makes `_draw_body` exit before
line 84. I verified the crash exists first — reproduced with the drafter's rig against the current
tree:

```
File ".../shaderbox/popups/pass_settings.py", line 84, in _draw_body
    _draw_inputs(app, document_id, name, document.passes[name])
KeyError: 'main'
```

**No other frame-level index of `document.passes` by the stale name survives.** I grepped the whole
popup body:

```
$ grep -n "document\.passes\[\|passes\.get(\|\.passes\b" shaderbox/popups/pass_settings.py
76:    if ui_document is None or name not in ui_document.document.passes:
84:    _draw_inputs(app, document_id, name, document.passes[name])
140:    entry = document.graph.passes.get(name, PassEntry())
141:    choices = [_UNWIRED, *sorted(document.passes)]
158:    entry = document.graph.passes.get(name, PassEntry())
```

- `:76` is the top guard, before `_draw_name` — reads the live `app.pass_settings_name`, correct.
- `:84` is the crash site, killed by the early return.
- `:140` and `:158` are `document.graph.passes.get(name, PassEntry())` — a `.get` with a default,
  inside `_draw_inputs` and `_draw_target`, both unreachable after the early return. Even if they
  were reached with a dead name they would return an empty `PassEntry()` rather than raise, so
  they are not a second crash. Under the finding-2 fix (guarding the two sections instead of
  returning) they stay unreachable on a rename frame.
- `:141` is `sorted(document.passes)` — the wiring combo's choice list, not an index. Safe either
  way.

**Verdict: no surviving stale-name index. PASS.**

---

## Commands

**Chords are free.** I grepped `commands.py` myself:

```
$ grep -n "mod_alt" shaderbox/commands.py
175:    CommandSpec(CommandId.OPEN_SETTINGS, "Settings", _chord(K.s, K.mod_alt), C.TOOLS),
176:    CommandSpec(CommandId.EXAMPLES, "Examples", _chord(K.e, K.mod_alt), C.TOOLS),
183:        _chord(K.slash, K.mod_alt),
```

Alt+S, Alt+E, Alt+/ — no Alt+A, no Alt+P. `K.p` appears only at `:167` (Ctrl+P) and `:172`
(Ctrl+Shift+P); those are different chord ints because `_chord` ORs the mod bits. Alt+A is free
outright. **Matches `01_spec.md § Open questions` item 4's provisional assignment
(`ADD_PASS` = Alt+A, `OPEN_PASS_SETTINGS` = Alt+P).**

**The chord-uniqueness test picks them up with no edit.** `test_command_routing.py::
test_no_two_specs_share_a_chord_in_overlapping_scopes` is a double loop over `COMMAND_SPECS`
asserting `not scopes_overlap` for any equal `default_chord`. Both new specs default to
`CommandScope.GLOBAL` (`commands.py:78`), which `scopes_overlap` treats as clashing with
everything — so a duplicate chord fails immediately. Confirmed by reading both.

**The Help shortcuts table picks them up with no edit.** `help_content.py:72-84`:

```python
for category in CATEGORY_ORDER:
    specs = [s for s in COMMAND_SPECS if s.category == category and s.default_chord]
```

Both new specs carry `C.TOOLS` and a non-zero chord, so they land in the Tools block with no code
change. **The wave spec's assessment is right and its new pin is justified**: the existing
`test_shortcuts_section_covers_every_populated_category` asserts only that each populated category
name appears and that the Help command's chord appears, so a new command inside an already-
populated category ships undocumented with nothing red. Confirmed by reading the test.

**One thing the spec should say and does not.** `route_flag` returns `route_always` for any Alt
chord (`commands.py:234-235`), so Alt+A does reach the dispatcher while a text input is active —
the spec's claim is correct. But `popup_suppresses` returns `True` for every scope
(`commands.py:246-250`), so **Alt+A and Alt+P are both dead while any modal is open** — including
the gear that item 4 opens immediately after an add. That is the right behaviour, but manual step 6
says "Press Alt+A from anywhere in the app with no modal open", which already encodes it; manual
step 5 does not. Suggested addition to manual step 5: "with the gear already open, Alt+P does
nothing (a modal owns the frame) — that is correct, not a dead chord."

---

## The four open questions the drafter closed

**1. One elected pass per document, or one across all documents? Default taken: per document.
AGREE.** Evidence: `tick_documents` is built at `ui.py:219-240` and can hold more than one
document only under "Render all documents" plus the one `pending_first` admission. The per-document
sweep converges N documents in `max(passes)` frames rather than `sum`, and every document in the
set is already paying a full `render()` that frame, so the marginal cost is one extra pass draw per
document. The document-level first-render budget (`ui.py:230-240`) already bounds how many
documents enter the set at all, so the two budgets compose.

**2. Re-arm the sweep after a hot reload of a not-live pass? Default taken: no. AGREE, with one
correction to the reasoning.** The spec says `watch.py::_reload_pass_if_changed` calls
`release_program`, which calls `invalidate()`, and neither clears `first_render_done`. Confirmed at
`core.py:187-190` (`release_program` → `self.invalidate()`). The stale picture under the stale wash
is the honest state, and finding #36's own text says the wash means "was live, is no longer". The
one-line alternative (clear `first_render_done` in `invalidate()`) is correctly named as reversible.
The correction: the spec should say the alternative is a **behaviour** change to the sweep's
termination proof, not a cosmetic one — clearing the flag on every invalidate means an off-chain
pass being actively edited re-enters the sweep on every save, which is bounded (one per save) but
no longer "each pass at most once". Worth one clause so a later wave does not take it as free.

**3. Should `_draw_add_input`'s deactivate-commit fire on the `x` cancel button? Default taken: no.
AGREE with the answer, DISAGREE with one clause of the reasoning.** The reasoning ("the button is
drawn after the input, so the deactivate would fire on the same frame") is right — I read the
order:

```
188:    committed, app.pass_add.buf = imgui.input_text(
202:    if imgui.is_key_pressed(imgui.Key.escape, repeat=False):
203:        app.pass_add.close()
204:    imgui.same_line()
205:    if ghost_button("x##cancel_add_pass"):
206:        app.pass_add.close()
```

So the deactivate must be captured at `:189`-equivalent and applied after `:206`. The clause I
disagree with: the spec names only the cancel branch as the thing that must suppress the commit.
The **Escape** branch at `:202` also runs after the input and also calls `close()`. Escape is in
fact safe (the item is still active on the Escape frame, so the deactivate is False, and on the
next frame `_draw_add_input` is not drawn at all because `pass_add.is_open` is False at
`pass_list.py:175`) — but the spec should say so rather than leave the reader to derive it, because
the same reader will be writing the suppression condition. **Fix:** add to open question 3: "The
Escape branch needs no suppression: on the Escape frame the input is still active so the
deactivate reads False, and on the next frame `_draw_add_input` is not drawn at all
(`pass_list.py`'s `if app.pass_add.is_open:` guard). Only the cancel button's branch, which runs in
the same frame as a deactivate caused by clicking it, needs the capture-then-apply shape."

**4. Where does `Pass.drawn_frame` reset when the canvas is reallocated? Default taken: nowhere.
AGREE.** `set_target` reallocates (`core.py:170-185`) and bumps `target_generation`, which is what
the Document's feedback cache keys on. A stale `drawn_frame` can only suppress a redraw within the
same frame the target changed — and the target only changes from the gear, which is a modal, and
while a modal is open the sweep does not run at all (`ui.py:292`'s `if not app.any_popup_open():`).
So the window the spec calls "one frame of the old picture" is in practice zero frames for the
gear path. The default is right and its justification is stronger than the spec claims. Optional
one-clause strengthening: "and the gear is a modal, so no sweep runs on the frame a target
changes."

---

## False trails (probed, fine — do not re-litigate)

- `Document.graph_errors` being overwritten by a target render: harmless. `plan_passes(graph)`
  walks every name in `graph.passes` and is target-independent; `_order_for` filters only the
  order. No production reader — `grep` finds it read only in `test_document_graph.py`,
  `test_pass_verbs.py:231` and `test_radiance_cascades_example.py:68`. The strip's error border
  reads `render_pass.compile_unit.errors` (`pass_list.py:86`), not graph errors.
- Whether `Pass.first_render_done` collides with `Document.first_render_done`: different objects,
  no shadowing, and the four existing readers (`ui.py:228/235/304/309`, `examples.py:103`,
  `document_grid.py:110`) all name `document.first_render_done` explicitly. Readability nit only.
- Whether the sweep's `render(target=...)` writes into the preview canvas: it cannot.
  `document.py:425`'s `canvas if (name == output and last) else None` is None for every pass when
  `canvas` is None, and `Pass.render` falls back to `canvas or self.canvas`. Own canvas, which is
  what the tile samples.
- Whether the size fixup being skipped for a same-frame-skipped pass matters: no. The fixup ran in
  the output render that stamped it, in the same frame.
- Whether the examples popup can issue a target render: no. `ui.py:297-310` is an `elif` branch on
  `popup_state == EXAMPLES`, and the sweep lives in the `if not app.any_popup_open():` branch. The
  `self._frame >= 0` conjunct is belt-and-braces, as the spec says, and cheap.
- Whether `add_pass` can create a pass that is already the output (making item 4's dropped
  `if not is_output` guard wrong): no. `project_session.py:762-785` never touches `graph.output`.
- `test_pass_verbs.py:28` importing `_strip_order` from `pass_list.py`: W-C changes
  `_draw_add_input`'s signature and `draw`'s call to it; `_strip_order` is untouched. Confirmed by
  reading the import and `test_the_strip_order_is_topological_and_independent_of_the_output`.
- Whether `is_item_deactivated_after_edit` double-fires with `enter_returns_true`: it can be True
  on the same frame as `committed`, and the spec's `or` makes that one commit. Not a bug.
- Whether the early-return body leaves an imbalanced imgui stack: ran three consecutive frames of
  the exact proposed shape headlessly; clean.
- The `@classmethod` on `Document.load_from_dir` (`document.py:453`): a genuine alternate
  constructor, explicitly allowed by `conventions.md ## Code rules`. Not a convention violation the
  wave inherits.

---

## Coverage statement

**Read end-to-end:** `10_wave_c_pass_verbs.md`; `01_spec.md`; `shaderbox/popups/pass_settings.py`
(the parts that matter: `draw_pass_settings`, `_draw_body`, `_draw_name`, `_draw_inputs`,
`_draw_target`); `shaderbox/widgets/pass_list.py::_draw_pass_tile` / `draw` / `_draw_add_input`;
`tests/test_lazy_compile.py`; `tests/test_help_content.py`; `tests/test_command_routing.py`;
`.claude/skills/imgui-ui/SKILL.md` § 0 and § 7.5; `conventions.md ## Code rules` and the first half
of `## Design decisions`; `CLAUDE.md`.

**Read the relevant regions of:** `shaderbox/document.py` (`render`, `begin_frame`,
`_swap_feedback`, `reset_feedback`, the export entry points); `shaderbox/core.py` (`Pass.__init__`,
`set_target`, `release_program`, `Pass.render`'s early-out); `shaderbox/ui.py`
(`_tick_frame_state`, `update_and_draw`'s render block, the dispatch/draw ordering);
`shaderbox/app.py` (`_build_command_callbacks`, `_on_pass_renamed`, `panel_pass`,
`open_pass_settings`); `shaderbox/hotkeys.py` (`dispatch_commands`, `_handle_escape`);
`shaderbox/commands.py` (`CommandSpec`, `_chord`, `route_flag`, `scopes_overlap`,
`popup_suppresses`, the Alt-chord block); `shaderbox/pass_graph.py` (`assert_plan_invariants`,
`plan_for_output`, `evaluation_order`, `_order_for`, `plan_passes`'s failure paths);
`shaderbox/project_session.py` (`_pass_name_error`, `add_pass`, `rename_pass`);
`shaderbox/help_content.py::_shortcuts_section`; `shaderbox/editor_types.py::InlineInput`;
`shaderbox/ui_primitives.py`'s two input forwarders; `00_findings.md` rows 9, 17, 18, 25, 28, 36
verbatim.

**Ran:** the headless imgui rig twice — once reproducing #17's `KeyError` against the current tree,
once proving the proposed early-return body is a balanced frame across three frames. A Python
simulation of the sweep algorithm for the three traced scenarios. Fresh greps for `input_text`
call sites, `labeled_*_input` callers, `mod_alt` chords, `PopupState.CLOSED` sites,
`first_render_done` readers, `graph_errors` readers, and the copilot's pass verbs.

**Skipped, and why:** `shaderbox/copilot/**` beyond the one grep that proves it has no pass verbs
(the wave touches nothing there); `shaderbox/tabs/code.py` and the editor FFI (W-E/W-F territory);
`shaderbox/exporters/**` beyond locating the `input_text` sites (the wave leaves export untouched
by intent, verified at the `document.py` call sites instead); the W-A / W-B / W-D / W-E / W-G / W-H
sections of the parent spec except where W-C's out-of-scope list cites them; `tests/` modules other
than the four the wave names.

**Not run:** `make gates`, per the review brief. No file under `shaderbox/` or the wave spec was
edited.

---

## The maintainer's six findings, closed or not

| # | The maintainer's words | Closed by | Verdict |
|---|---|---|---|
| #9 | "add a keybinding to open settings for the currently selected pass (Alt+P?)" | Item 5's `OPEN_PASS_SETTINGS` on Alt+P, dispatching `open_pass_settings_for_panel_pass` | **Closes it.** "Currently selected pass" is `App.panel_pass` (`app.py:571`), whose docstring is literally "the active shader tab's own pass when it belongs to this document, else the output" — the notion the maintainer means. The spec correctly catches that `panel_pass` returns a `Pass`, not a name, and converts via `pass_name_of(render_pass.source.path)`; `Pass` genuinely has no `.name` attribute (checked `Pass.__init__`, `core.py:139-168`), so the parent's `panel_pass(...).name` shorthand would have been an `AttributeError`. |
| #17 | "typed 'paint', hit Enter, the app crashed: `KeyError: 'main'` at `pass_settings.py:84`" | Item 1's early return | **Closes it.** I reproduced the crash and the return is above the only indexing site. |
| #18 | "typed 'paint' without Enter ... closed, reopened, 'main' was back. The name should auto-apply without Enter." | Items 2 and 3 | **Closes both halves.** Item 2 covers "focus leaves the field" (the click-away the maintainer describes as attempt one); item 3 covers "closed" — and the ordering argument (`ui.py:336` dispatch before `:429` draw) is why item 3 must exist at all rather than living in the body. The maintainer's "the Reads wiring still said 'main'" is also closed, by item 1's redraw-next-frame against the live name. |
| #25 | "add an 'add pass' hotkey as well." | Item 5's `ADD_PASS` on Alt+A | **Closes it.** The spec correctly routes through the same `pass_add.open(...)` the ghost button calls, so `InlineInput.open` arms `needs_focus = True` (`editor_types.py:58-61`) and the input grabs the keyboard on its first draw — which is finding #25's "then focusing the input" half from the ledger's shape line. |
| #28 | "when creating a pass, auto-activate it: open its code and render it." | Item 4 | **Closes it.** "Open its code" = `open_pass(name)`; "render it" = `set_output_pass`, which puts it on the output chain so it draws every frame. The ledger's "same for the copilot's add-pass path if it has one" is refuted by grep, which I re-ran: no hits. |
| #36 | "I need to click each pass manually to trigger its redraw after I close/open the app." | Item 7's sweep | **Closes it.** Scenario (b) above shows five off-chain passes fully drawn within five frames. The ledger's option A ("first-render every pass once ... one per frame like the document first-render budget") is implemented literally, including the "keep only the output chain live" half, which scenario (a) confirms costs nothing extra. `01_spec.md § Open questions` item 2's caveat (black tiles with the LAST pass as output) is answered by scenario (a): every pass is on the chain, so every tile is drawn on frame 0. |
