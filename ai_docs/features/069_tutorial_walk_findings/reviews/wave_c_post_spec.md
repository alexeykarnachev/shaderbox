# 069 W-C — post-implementation review: spec fidelity and architecture

Reviewer role: `dev_flow.md` step 6, spec-fidelity and architecture. Commit under review:
`a246a19`. Anchors: `10_wave_c_pass_verbs.md` (the wave spec), `01_spec.md § Locked decisions
D10 D11` and `§ W-C` (the parent), `00_findings.md` #9 #17 #18 #25 #28 #36 (the maintainer's
words), `.claude/skills/imgui-ui/SKILL.md`, `ai_docs/conventions.md`.

## Verdict

| Dimension | Verdict |
|---|---|
| Wave-spec fidelity | **PASS** — all eight design decisions landed; three test-level deviations, all reported by the implementer and all additive. |
| Parent fidelity | **PASS** — every W-C bullet and both locked decisions are satisfied by lines cited below. |
| Findings closure | **PARTIAL** — #9 #17 #18 #25 #28 closed. **#36 is closed on paper and the maintainer's complaint recurs in a different form**: the tile that was a stable black rectangle is now a rectangle strobing 64% of its pixels every frame, forever, on the very example manual step 8 names. Finding 1. |
| Architecture | **PASS** — homes, callback shapes and naming are right; two low-severity notes (a duplicated 4-line activate block by design, a spurious notification on a dead gear target). |
| Docs | **PARTIAL** — the skill § 7.5 rewrite is correct and contradiction-free, Help picks both commands up, `dev_flow.md`'s module map still describes the code. But the commit does not pass `ruff check` as committed (finding 2), and `conventions.md`'s 066 D1 entry now under-describes the compile budget (finding 4). |

## Coverage table — wave-spec design decisions

| # | Decision | Status | Evidence |
|---|---|---|---|
| 1 | `_draw_name` returns `bool`; body skips the two name-indexing sections, Close row still drawn | **landed as written** | `popups/pass_settings.py:80-85` (`renamed = _draw_name(...)`, `if not renamed:` guarding exactly `_draw_inputs` + `_draw_target`), `:87-88` (`return not ghost_button("Close")` outside the guard) |
| 2 | Name field commits on deactivate-after-edit + Enter; `_commit_pass_name` free function; rejected name snaps the buffer back | **landed as written** | `pass_settings.py:102` (`deactivated = imgui.is_item_deactivated_after_edit()`, on the line after `input_text`, above `same_line` at `:106`); `:114-127` (`_commit_pass_name`, module-level, buffer reset on both reject paths) |
| 3 | Close and Escape commit through one funnel `App.close_pass_settings` | **landed as written** | `app.py:901-916`; call sites `pass_settings.py:67` and `hotkeys.py:300-302`. Escape branch order verified: `PASS_SETTINGS` is tested first, the lib-picker `inline_input_owns_esc` guard is the `elif` and is untouched (`hotkeys.py:303-306`) |
| 4 | Add pass activates (open tab, set output) then opens the gear | **landed as written** | `widgets/pass_list.py:212-220`; `draw` threads `open_pass` at `:176` |
| 5 | `OPEN_PASS_SETTINGS` Alt+P, `ADD_PASS` Alt+A in `C.TOOLS`; two `App` methods in `_build_command_callbacks` | **landed as written** | `commands.py:40-41`, `:178-184`; `app.py:516-517`, `:918-929`. `route_flag` routes Alt always (`commands.py:243-244`) |
| 6 | The 22-site `input_text` census; six sites converted | **landed as written** | `pass_settings.py:102`, `pass_list.py:196`, `lib_picker/tree.py:224` (new file + new dir, one function), `:299`, `lib_picker/preview.py:79`. The tag site keeps `picker_tag_input_focused` in place at `preview.py:78` and reads the deactivate on the next line, both above `same_line` at `:80` |
| 7 | The first-render sweep: `render(target=)`, four-conjunct skip, stamps on attempt, `first_render_done` narrowed, one pass per frame in `ui.py` | **landed as written** | `document.py:386-424` (parameter, `resolved`/`output` split, skip, both stamps), `:401-402` (`canvas is None and target is None`), `:444` (`draw_into` rename), `ui.py:295-312` (the frame gate) |
| 8 | Skill § 7.5 rewritten to the commit rule | **landed as written** | `.claude/skills/imgui-ui/SKILL.md:366-384` |

## Coverage table — tests

| Test | Status | Evidence |
|---|---|---|
| `test_pass_verbs.py::test_the_gear_body_survives_a_rename_mid_frame` | **landed with a reported deviation** — drives the REAL focus transition (six frames, injected keystroke) instead of monkeypatching `_commit_pass_name`. Strictly stronger than the spec. | `tests/test_pass_verbs.py:353-381`; the `keep_open == [True] * 6` assertion at `:381` is the Close-row pin |
| `test_pass_verbs.py::test_a_rejected_rename_snaps_the_buffer_back` | **landed as written**, plus a fourth (empty-buffer) case | `tests/test_pass_verbs.py:384-412` |
| `test_pass_verbs.py::test_add_pass_activates_the_new_pass` | **landed as written**, driven through the real widget | `tests/test_pass_verbs.py:415-449` |
| `test_lazy_compile.py::test_every_pass_renders_once_within_n_frames` | **landed as written** | `tests/test_lazy_compile.py:164-176` |
| `test_lazy_compile.py::test_a_broken_off_chain_pass_is_stamped_on_attempt` | **landed as written** (the spec's "second case", split into its own test) | `tests/test_lazy_compile.py:179-194` |
| `test_lazy_compile.py::test_the_steady_state_draws_only_the_output_chain` | **landed with a reported deviation** — gained a target render inside a settled frame asserting only `{composite, trail}` redraw | `tests/test_lazy_compile.py:197-231` |
| `test_lazy_compile.py::test_two_output_renders_in_one_frame_both_draw` | **landed with a reported deviation** — also asserts `trail` draws on the `composite` target render | `tests/test_lazy_compile.py:234-256` |
| `test_lazy_compile.py::test_a_target_render_does_not_complete_the_document_first_render` | **landed as written** | `tests/test_lazy_compile.py:259-263` |
| `test_lazy_compile.py::test_the_skip_does_not_fire_without_a_frame_counter` | **landed as written** | `tests/test_lazy_compile.py:266-276` |
| `test_command_routing.py` chord uniqueness | **no edit needed, covered** | loops `COMMAND_SPECS`; `make gates` green |
| `test_help_content.py::test_shortcuts_section_lists_every_bound_command` | **landed as written** | `tests/test_help_content.py:43-53` |
| `test_lib_files.py::test_a_picker_inline_input_commits_on_click_away` (+ the no-edit sibling) | **landed with a reported deviation** — own `lib_app` fixture with `SHADERBOX_DATA_DIR` isolated | `tests/test_lib_files.py:31-46` (fixture), `:66-94` |

## Findings

### 1. #36 is closed on paper; the maintainer's complaint recurs as a strobe on the same example — HIGH

**Claim.** An off-chain pass that reads itself (`u_prev`) now shows a tile that changes 64% of its
pixels every frame, forever, alternating between two frozen pictures. Manual step 8 names the
Radiance Cascades example; its `cascade` pass is exactly that shape
(`inputs={'u_scene': 'paint', 'u_df': 'df', 'u_prev': 'cascade'}`). The maintainer's words in #36
are "they don't initialize automatically" — the wave answers that with a picture, then replaces
the black rectangle with a strobing one, which is a worse reading of the same tile.

**Evidence, by execution.** Driving the real `ui.py` per-frame sequence (preview render, own-canvas
render, sweep) on the shipped RC example with the output left mid-chain on `df` — #36's own
scenario — and reading back each pass canvas per frame:

```
frame 0: elected='cascade'    means={'cascade': 65.64, 'composite': 0.0,  ...}
frame 1: elected='composite'  means={'cascade': 65.72, 'composite': 105.86, ...}
frame 2: elected=None         means={'cascade': 29.79, 'composite': 105.86, ...}
frame 3: elected=None         means={'cascade': 65.72, ...}
frame 4: elected=None         means={'cascade': 29.79, ...}   (alternates forever)
```

Per-pixel, from frame 3 onward: `mean abs diff 45.34, max 255, changed px 64.0%` on every frame.

The same probe with the sweep removed (pre-W-C behaviour) reports `mean 0.00 diff-vs-prev mean
0.00 changed 0.0%` — a stable black tile. So the strobe is introduced by this wave.

**Cause.** `Document.begin_frame` swaps EVERY feedback pass's canvas each frame
(`document.py:310-311`), whether or not that pass drew. The swap is pre-existing; W-C is what makes
it visible, because before the sweep both history canvases were black. The draw counter confirms
the steady state is correct — from frame 2 on, `cascade` and `composite` draw zero times and only
the `df` chain draws — so this is not the draw-once invariant failing, it is the tile pointing at a
canvas that keeps being exchanged under it.

**Fix, one sentence.** In `Document._swap_feedback`, return without swapping when the pass did not
draw this frame (`render_pass.drawn_frame != self._frame`) — the stamp this wave just added is what
makes that decidable — and pin it with a test that renders an off-chain feedback pass once and
asserts its canvas identity is unchanged across the next three frames.

### 2. The commit does not pass `ruff check` as committed, while its message claims a green gate — MEDIUM

**Claim.** `tests/test_lib_files.py` at `a246a19` fails `ruff check` (I001, unsorted imports). The
commit message's last line reads "make gates: GREEN — check passed, test passed, smoke passed". The
gate was green only because a hook rewrote the file during the run and the rewrite was never staged.

**Evidence.** `git stash && uv run ruff check tests/test_lib_files.py` on the committed tree:

```
I001 [*] Import block is un-sorted or un-formatted
  --> tests/test_lib_files.py:9:1
Found 1 error.
RUFF_EXIT=1
```

Running `make gates` on the clean tree emits `== gates: check exited 2 on the first run;
re-running (hooks rewrite files) ==`, exits 0, and leaves `tests/test_lib_files.py` modified in the
working tree — `from conftest import seed_tmp_project` moved into the third-party block. The gate
therefore passes on a tree that is not the committed one, and a fresh clone running `make check`
goes red.

**Fix.** Apply the hook's rewrite (move `from conftest import seed_tmp_project` above
`from imgui_bundle import imgui`) and commit it; the gate's own "check exited 2 on the first run"
line is the signal that the tree needs re-staging before the commit is finished.

### 3. Closing the gear on a retired pass name pushes a spurious error notification — LOW

**Claim.** `App.close_pass_settings` calls `session.rename_pass(document_id, name, buf)` without
checking that `name` is still a live pass. When the gear's target has been retired externally while
the user had typed into the field, the callee returns `"no such pass '<name>'"` and the funnel
pushes it as a user-visible error on a close the user did not associate with a rename.

**Evidence, by execution** (headless probe on a seeded app):

```
app.open_pass_settings("ghost"); app.pass_settings_name_buf = "whatever"
app.close_pass_settings()
spurious notifications on a dead gear target: ["no such pass 'ghost'"]
```

Reachable in production: `ui.py:181` calls `session.sync_documents_from_disk()` every frame with no
popup gate, and that path re-reads a document whose `document.json` changed on disk, which can
retire a pass name while the gear is open. `_draw_body`'s first guard (`pass_settings.py:74-79`)
then returns `False`, which routes straight into `close_pass_settings`.

**Fix.** Add `name in self.ui_documents[document_id].document.passes` to the existing conjunction at
`app.py:910` so a dead target closes silently instead of notifying.

### 4. `conventions.md`'s 066 D1 entry no longer describes the compile budget the code has — LOW

**Claim.** `ai_docs/conventions.md:207-209` states the live loop's bound as "first renders are
admitted one document per frame (`Document.first_render_done`, set on attempt), so neither frame 0
nor the Examples popup pays every compile at once". After this wave the loop also admits one PASS
per document per frame (`ui.py:295-312`), each of which pays that pass's compile — a second budget
the entry does not name. Nothing there is now false, but a reader deciding where a new compile cost
may go will not learn that the per-pass sweep exists.

**Fix.** Extend that bullet with one clause naming `Pass.first_render_done` and the one-pass-per-
document-per-frame sweep beside the existing document-level budget.

### 5. The activate block is duplicated between the tile click and the add path — INFORMATIONAL

`pass_list.py:123-130` (tile click) and `:212-219` (add) are two copies of open-tab-then-set-output,
differing only by the click path's `if not is_output` guard. The spec sanctioned this explicitly
("byte-for-byte what a tile click does ... minus the click path's `if not is_output` guard, which is
vacuous for a pass created this frame"), so it landed as designed. Named only so a later wave that
changes what activation means knows there are two sites, not one.

## Architecture answers

- **`_commit_pass_name` as a free function in `popups/pass_settings.py`.** Right home. It reads and
  writes `app.pass_settings_name_buf` and calls `app.session.rename_pass`, both of which the other
  pass verbs also reach from the draw site (`pass_list.py:65` calls `session.delete_pass` directly,
  `:128` calls `set_output_pass` directly). Putting it on `App` would make it the only pass verb
  routed through the App surface. It is a module-level `def`, not a `@staticmethod`, per
  `conventions.md ## Code rules`.
- **`open_pass_settings_for_panel_pass` / `open_add_pass`.** Consistent: `_build_command_callbacks`
  (`app.py:479-517`) is bare bound methods plus lambdas, and these two carry a document-existence
  guard a lambda cannot, which is the stated reason. Both sit beside `open_pass_settings` /
  `open_settings` in the same block of the file.
- **`draw_into` / `target` / `resolved` / `output` in `Document.render`.** Readable without the
  spec: the docstring's new paragraph (`document.py:397-399`) says what `target` does and that the
  graph output still decides size and external canvas, and `resolved` / `output` are bound on
  adjacent lines (`:403-404`) so the split is visible at a glance. Two stale sentences: the
  docstring's first line still says "every pass the OUTPUT needs ... each exactly once" when a
  target render draws the target's chain, and the cycle-fallback comment (`:411-412`) still says
  "draw the output alone" where the code now draws `resolved`. Both are one-word edits, neither
  changes behaviour.
- **Skill § 7.5.** States the rule as behaviour in the skill's own voice, names no wave, no
  finding number and no commit. Grepped `Enter commits`, `Esc cancels`, `enter_returns_true`,
  `is_item_deactivated` across the skill: the only other hit is line 479, the copilot chat input's
  `enter_returns_true` + `ctrl_enter_for_new_line` note, which § 7.5 explicitly excludes as a
  per-keystroke value field. No contradiction. The Pattern bullet is now one 18-line paragraph,
  which reads long for a skill bullet; splitting it into "commit trigger", "where to read the
  query", "cancel controls", "close funnel" and "what the rule excludes" would carry the same
  content at a glance. Style only.
- **`dev_flow.md` module map.** `:204-211` still describes the code: `pass_list.py` "A tile click
  sets the graph OUTPUT and opens that pass in the editor" holds, and `pass_settings.py` "Opens
  from a tile's gear, its context menu, or automatically on `add pass`" is still true and now also
  from Alt+P. No edit required; a clause naming the two chords would be an improvement, not a fix.
- **`help_content.py`.** No code change needed, as the spec said, and verified by generating the
  section: `Pass settings  Alt+P` and `Add pass  Alt+A` both appear under Tools.

## Findings closure, one by one

| Finding | Closed? | The line that closes it |
|---|---|---|
| #9 "keybinding to open settings for the currently selected pass (Alt+P?)" | **yes** | `commands.py:178-183` + `app.py:918-922`; "currently selected" is `panel_pass` (`app.py:573-584`), the notion the finding named |
| #17 "`KeyError: 'main'` at `pass_settings.py:84`" | **yes** | `pass_settings.py:81` (`if not renamed:`); pinned by `test_the_gear_body_survives_a_rename_mid_frame`, which drives the real transition and asserts the pass renamed and the body returned `True` on all six frames |
| #18 "the name should auto-apply without Enter" | **yes** | `pass_settings.py:102-105` (deactivate commit) and `app.py:901-916` (the Escape/Close funnel). Verified by execution: gear open, buffer `"scene"`, `close_pass_settings()` → `passes after Escape-close: ['scene']` |
| #25 "add an 'add pass' hotkey as well" | **yes** | `commands.py:184` + `app.py:924-929`; `pass_add.open` arms `needs_focus` (`editor_types.py:58-61`), which is the finding's "then focusing the input" |
| #28 "when creating a pass, auto-activate it: open its code and render it" | **yes** | `pass_list.py:212-220`; copilot has no pass verbs, re-confirmed by `grep -rn 'add_pass\|rename_pass\|delete_pass\|set_output_pass' shaderbox/copilot/` returning nothing |
| #36 "I need to click each pass manually to trigger its redraw" | **on paper yes, in practice PARTIAL** | `ui.py:301-312` elects one pass per frame and `composite` fills from 0.0 to 105.86 by frame 1 in the probe above. But finding 1: the same probe shows `cascade` strobing at 64% of pixels per frame from frame 2 onward, where it was a stable black before. The maintainer will not report "it initialised"; he will report a flickering tile |

## Verified deviations (the implementer's own list)

| Claim | Verdict |
|---|---|
| (a) the rename test drives the real focus transition instead of monkeypatching `_commit_pass_name` | **true** — `tests/test_pass_verbs.py:361-373` runs six frames with `set_keyboard_focus_here(0)`, an injected `"q"` at frame 2 and a focus move at frame 3; no monkeypatch of `_commit_pass_name` anywhere in the module |
| (b) `test_two_output_renders_in_one_frame_both_draw` asserts `trail` also draws on a `composite` target render | **true** — `tests/test_lazy_compile.py:252-256` asserts the delta set is exactly `{"composite", "trail"}` |
| (c) `test_the_steady_state_draws_only_the_output_chain` gained a target render inside a settled frame | **true** — `tests/test_lazy_compile.py:222-231` |
| (d) `tests/test_lib_files.py` uses its own `lib_app` fixture with `SHADERBOX_DATA_DIR` isolated, and the shared `app` fixture does not isolate `app_data_dir()` | **true on both halves, and the second half is a real pre-existing defect.** `tests/conftest.py:47-65` sets no `SHADERBOX_DATA_DIR`; `App.__init__` calls `sync_shipped_lib(SHADER_LIB_SEED_DIR, shader_lib_root())` at `app.py:260`, and `shader_lib_root()` resolves to `/home/akarnachev/.local/share/shaderbox/shader_lib` with no override. Demonstrated: back-dating the live manifest to 2000-01-01 and running `uv run pytest tests/test_pass_verbs.py -k test_add_pass_activates` (one test, shared `app` fixture) rewrites it. The live manifest's own mtime was `2026-09-02 21:22:40`, four minutes after this commit — written by the implementer's own gate run. Ten test modules use that fixture. Out of scope for W-C, worth a fixture-level `SHADERBOX_DATA_DIR` in `tests/conftest.py::app` in whichever wave next touches the fixture |
| (e) the Escape branch order in `hotkeys.py` | **true** — `hotkeys.py:300-306`: `PASS_SETTINGS` is the `if`, the lib-picker `inline_input_owns_esc` guard is the `elif` and is byte-identical to before apart from black's reflow |

## False trails

- **`_graph_errors` overwritten by a target render.** Probed: `plan_passes(graph)` walks every name
  regardless of target, and `Document.graph_errors` has no production reader. Fine, as the spec said.
- **Alt+A pressed while the add-pass input already holds typed text.** Probed through the real
  widget: the buffer resets to empty and re-focuses, no pass is created, no notification. Acceptable.
- **A stale graph output (`output_pass is None`) with a target render.** Now draws passes into their
  own canvases where before nothing drew; the preview render still early-returns unchanged. An
  improvement, not a regression.
- **A zero-pass document crashing `panel_pass` via Alt+P.** `delete_pass` refuses the last pass
  (`project_session.py:794-795`), so the state is unreachable.
- **The `pending` scan costing something after the sweep drains.** One dict scan per document per
  frame over a handful of passes; nothing measurable.
- **`test_the_skip_does_not_fire_without_a_frame_counter` reading `document._frame`.** A private
  attribute from a test, but the test's whole subject is that counter; no better public handle.

## Coverage statement

Read end to end: `shaderbox/popups/pass_settings.py` (changed region plus the whole body/name/commit
path), `shaderbox/app.py:479-517` and `:885-935` and `:560-585`, `shaderbox/commands.py:37-45`,
`:170-190`, `:233-259`, `shaderbox/core.py:155-170`, `shaderbox/document.py:230-345` and `:383-470`,
`shaderbox/hotkeys.py:265-320`, `shaderbox/ui.py:170-345`, `shaderbox/widgets/pass_list.py:100-220`,
`shaderbox/popups/lib_picker/preview.py:55-100`, `shaderbox/popups/lib_picker/tree.py:210-310`,
`shaderbox/project_session.py:440-480` and `:785-845`, `shaderbox/shader_lib/seed.py:1-80`,
`shaderbox/editor_types.py:44-70`, `shaderbox/pass_graph.py:180-195`, all four changed test modules,
`tests/conftest.py:1-80`, `.claude/skills/imgui-ui/SKILL.md § 7.5` and every hit of the four
contradiction greps, the full wave spec, the parent spec's `§ Locked decisions` and `§ W-C`, and
findings #9 #17 #18 #25 #28 #36 verbatim.

Ran: `make gates` unpiped to a file, `echo $?` → `0` (check, test, smoke all passed, smoke NOT
skipped); `ruff check` on the stashed committed tree; four headless probes (Escape commit + dead
target, Alt+A on an open input, the real `ui.py` sweep sequence with per-pass canvas readback, and
the same with the sweep removed); the Help section generated from `COMMAND_SPECS`; a manifest-mtime
experiment isolating the shared `app` fixture's write to the real shader library.

Not read: `reviews/wave_c_pre.md` beyond its findings list as summarised in the spec's § Review
history (it is the round-1 anchor the spec already folded, and reading it would anchor this review
on another reviewer's conclusions); the 1040-line wave spec's § Verified / corrected premises table
was read in full but its individual line-number citations were spot-checked rather than each
re-verified, since the spec was written against `faccf0e` and this review checks the landed code.
Manual verification steps 1, 1b, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14 were not driven in the real
window (no synthetic input on this box); steps 2, 8 and 9 were driven headlessly and are reported
above.

---

# Round 2 (closure) — against `d3338d4`

Narrow closure round on the fix-up commit `d3338d4` ("069 W-C fixes: strobe, cancel, sweep
re-arm"). Scope: round 1's four findings, the conftest isolation defect, and a re-walk of the
amended wave spec. Late-round rule applied: a preference is a false trail, not a finding.

**Overall: PASS.** All five items closed, each demonstrated. The amended spec describes the landed
code, and on one point it is now more accurate than round 1's own reading was.

## Per-item verdicts

| Item | Verdict | What closes it |
|---|---|---|
| Finding 1 — the strobe | **CLOSED** | Re-ran my own probe unchanged: `mean abs diff 0.00, max 0, changed px 0.0%` on every frame, against round 1's `45.34 / max 255 / 64.0%`. `document.py:309-317` |
| Finding 2 — ruff on the committed tree | **CLOSED** | `uv run ruff check .` → `All checks passed!`, exit 0; `uv run ruff format --check .` → `230 files already formatted`, exit 0. Tracked tree clean before and after |
| Finding 3 — spurious notification | **CLOSED** | Probe re-run: `spurious notifications on a dead gear target: []`, against round 1's `["no such pass 'ghost'"]`. `app.py:920-931` (the `name in ui_document.document.passes` conjunct) |
| Finding 4 — `conventions.md` 066 D1 | **CLOSED** | `conventions.md:207-212` now names both axes and states stamp-on-attempt as the reason a failed compile cannot be re-admitted |
| Conftest isolation | **CLOSED** | `tests/conftest.py:48,53` sets `SHADERBOX_DATA_DIR` before the `App` import. Full suite (1015 passed) leaves the real library's manifest mtime, sha1 and file count all unchanged |

## Finding 1 — the measured result

The probe is round 1's, unedited: the shipped Radiance Cascades example, output forced to `df`
(which leaves the self-reading `cascade` off-chain), driving the real `ui.py` per-frame sequence
and reading `cascade`'s canvas back each frame.

```
frame 3: cascade tile vs previous frame -- mean abs diff 0.00, max 0, changed px 0.0%
frame 4: cascade tile vs previous frame -- mean abs diff 0.00, max 0, changed px 0.0%
...through frame 7, identical
```

Round 1 measured `mean abs diff 45.34, max 255, changed px 64.0%` on the same frames.

The tile is a picture, not a black rectangle: `cascade` settles at a stable mean of `29.82` while
`paint` sits at `0.46`, so the difference between "holds a picture" and "is black" is visible in
the numbers. `composite` still fills from `0.0` to `105.88` by frame 1. #36 is now closed in both
halves — the tiles initialise, and they hold still.

Two checks beyond the finding itself, because a guard that fixes a strobe can freeze a trail:

- **The on-chain half still advances.** With RC's shipped output (`composite`), `cascade` is
  on-chain and self-reading; its per-frame diff stays non-zero (`0.14 / 2.41 / 2.16 / 1.97 / 2.18`)
  with its history live. The guard is scoped to passes that did not draw.
- **The guard is load-bearing and its test discriminates.** Deleting the two guard lines from
  `begin_frame` turns `test_a_swept_feedback_pass_holds_still` red while
  `test_an_on_chain_feedback_pass_still_advances_every_frame` stays green
  (`1 failed, 1 passed`). Restored; tree clean.

## Finding 2 — the mechanism, not just the symptom

`make gates` on the committed tree exits 0 **and** emits zero occurrences of
`re-running (hooks rewrite files)`, against one at `a246a19`. So `make check` passes on its first
run and the committed tree is the tree the gate passed on, which is what round 1's finding was
actually about. The fix-up also caught a defect round 1 did not: `B023`, the cancel-button stub
closing over loop variables late-bound — now bound as defaults in `tests/test_lib_files.py`.

Note for the record: `git status` during this round showed `shaderbox/pass_graph.py`,
`resources/editor/VERSION` and `libeditor.so` modified. Those are concurrent W-A and W-F work by
other agents, not this commit and not the gate; the ruff runs above were made against the clean
`d3338d4` tree.

## Conftest isolation — the full-suite proof

Round 1 demonstrated the defect on one test. The closure check is stronger: hash and count the
real library, run the **entire** suite, compare.

```
before: mtime=946674000 sha=cc2c9f676d83 files=16
1015 passed in 22.14s
after:  mtime=946674000 sha=cc2c9f676d83 files=16
REAL LIB UNTOUCHED BY THE FULL SUITE
```

## Spec re-walk — the four amended sections

| Amended section | Verdict |
|---|---|
| Decision 7, the feedback-swap paragraph | **Describes the landed code.** The quoted block matches `document.py:309-317` line for line, including `previous_frame` captured before the `_frame` assignment. The stated reason the comparison works — `begin_frame` assigns the new frame before the swap loop — is the code's actual order |
| Decision 7, the `self._frame >= 0` correction | **Describes the landed code, and corrects round 1.** Verified independently: `ui.py:311` is the ONLY `render(target=)` call site in `shaderbox/`, and the examples branch at `:325` calls `render()` with no target, so `target is not None` is already False there. The amendment demoting `>= 0` from mechanism to redundant defence is right, and my round-1 report repeated the spec's original framing without checking it |
| Open question 2's reversal | **Describes the landed code, and its bounding argument holds.** `core.py:202` clears the flag. The caller enumeration is exact — `grep` finds four: `core.py:194` (`release_program`), `watch.py:50` and `:87`, `copilot/backend.py:2275`; `set_target` does not call it. Both `watch.py` sites are behind mtime-equality early returns (`watch.py:29` `if disk_mtime == src.mtime: continue`; `watch.py:79` `if current == cached: return False`), which is what keeps "once per invalidate" from becoming "once per frame" |
| The cancel-order note in Tests, and the `tree.py` Files-touched row | **Describes the landed code.** Both `tree.py` sites now capture-then-apply (`:224-240`, `:304-316`). The corrected row no longer claims the `x` branches are unchanged. The test's disclosed limitation — the click is injected at the button because headless imgui never reports hover — is honest: the stub calls the real `ghost_button` and overrides only its return, so the widget is genuinely submitted in its real position. Mutation-tested: restoring the shipped commit-above-cancel order turns `test_the_cancel_button_cancels_instead_of_committing` red and nothing else |
| The `preview.py` Files-touched row (unchanged) | **Correct as it stands.** Its only post-input control is `+ Add`, a commit rather than a cancel, so capture-then-apply does not apply. No amendment was owed |

## Round 2 false trails

- The `x`-cancel defect at the two `tree.py` sites was a real miss in round 1 — I verified the
  deactivate was read on the correct line but did not trace the button's ordering at those sites.
  It was the code reviewer's find; recorded here so the gap in this review's coverage is on record,
  not as a new finding.
- `pass_graph.py` / `libeditor.so` showing as modified mid-round: concurrent work by other agents,
  not this commit.
- The stub-driven cancel click not being a real mouse press: disclosed in the spec, and the branch
  ordering it exercises is the real one. A preference for a hover-driven test is not a finding.

## Round 2 coverage statement

Read end to end: the full `d3338d4` diff for `shaderbox/document.py`, `core.py`, `app.py`,
`popups/lib_picker/tree.py`, `tests/conftest.py`, `ai_docs/conventions.md`, and the amended
sections of `10_wave_c_pass_verbs.md` (decision 3's liveness paragraph, decision 7's two
amendments, Files touched, the four new Tests entries, open question 2). Read in the tree:
`shaderbox/watch.py:19-92`, `shaderbox/ui.py:260-332`, `shaderbox/popups/lib_picker/preview.py:70-86`,
`tests/test_lib_files.py::_run_frames` and both cancel tests.

Ran: `uv run ruff check .` and `uv run ruff format --check .` on the committed tree; `make gates`
unpiped to a file with `echo $?` → 0 and a grep for the hook-rewrite line → 0 occurrences; the full
suite with a before/after hash of the real shader library; four probes (the round-1 strobe probe
unedited, the round-1 sweep probe, the round-1 Escape/dead-target probe, and a new on-chain
feedback probe); and two mutation tests (the `begin_frame` guard removed, the `tree.py` rename site
put back to commit-above-cancel), each confirmed red on exactly its own test and restored.

Not re-checked: round 1's PASS dimensions that `d3338d4` does not touch (the eight design
decisions' landing sites, the parent bullets, findings #9 #17 #18 #25 #28, the skill § 7.5 rewrite,
the Help table). The manual verification steps needing a real window remain undriven.
