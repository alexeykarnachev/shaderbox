# Wave C — post-implementation code-correctness review

Commit under review: `a246a19` ("069 W-C: pass verbs, inline commits, first-render sweep").
Role: correctness only — bugs, races, GL-context lifecycle, resource leaks, error handling,
imgui frame-order hazards. Design and doc questions are out of scope except where a design
choice produces a wrong behaviour.

**Verdict: PARTIAL** — sweep PASS · lifecycle PARTIAL · inline commits PARTIAL ·
commands PASS · tests PASS · conventions PASS.

Two findings, both in the inline-commit half. The sweep is correct in every scenario traced,
including the three the commit message calls load-bearing; the export path is unchanged; the
new commands are correctly scoped; every new test was checked for a passing-for-the-wrong-reason
failure mode and three were re-falsified by hand.

---

## Findings

### 1. The lib picker's `x` cancel button cannot cancel — it commits (MEDIUM)

The commit added a deactivate-commit to three lib-picker inputs but did not carry over the
capture-then-apply shape it wrote for `_draw_add_input`. At both `tree.py` sites the `x` button
is submitted *after* the commit has already run, so the click that presses it — which is itself
what deactivates the input — fires the transaction the user was cancelling.

`popups/lib_picker/tree.py::_draw_file_rename_input` is the clearest case. It goes halfway: it
captures `commit` into a local and correctly drops it in the Escape branch, then leaves the `x`
branch below the `if commit:` that has already fired:

```
    commit = changed or imgui.is_item_deactivated_after_edit()
    if imgui.is_key_pressed(imgui.Key.escape, repeat=False):
        app.shader_lib_files.cancel_file_rename()
        commit = False
    if commit:
        app.shader_lib_files.rename_file(path, app.shader_lib_files.file_rename.buf)
    imgui.same_line()
    if ghost_button(f"x##cancel_ren_{path}"):      # too late — the rename already ran
        app.shader_lib_files.cancel_file_rename()
```

`_draw_inline_new_input` (the shared new-file / new-dir input) has the same ordering and no
local at all — it calls `commit()` inline, then draws `x` twenty lines later.

Evidence — a structural check run against the three sites, showing the add-pass site ordered
correctly and both lib sites not:

```
$ uv run pytest tests/test_zz_review_probe.py -q
E  AssertionError: _draw_file_rename_input: cancel at line 22 runs AFTER the commit at
   line 20, so clicking it cannot drop the commit it raised
1 failed, 1 passed        # the passing one is _draw_add_input
```

I also drove the real widget through a six-frame imgui rig (focus → keystroke → focus move) and
watched `probe.glsl` disappear on the deactivate, confirming the commit does fire from a
focus-leave at this site. (Both probe files were removed; `git status --short` is clean.)

This is not a missed edge case — the wave spec's open question 3 works the hazard out in full
and specifies the capture-then-apply shape as the taken default. It was applied to
`_draw_add_input` and not to its three siblings, while spec line 592 records "the existing
Escape-cancel and `x` branches are unchanged" as if leaving them were free. It stopped being
free the moment the deactivate commit landed beside them.

**Fix:** in `_draw_file_rename_input`, move the `imgui.same_line()` + `ghost_button` block above
the `if commit:` and set `commit = False` inside its branch, exactly as `_draw_add_input` does;
in `_draw_inline_new_input`, capture the deactivate into a local, move the `x` branch above the
commit, and drop the local when it runs.

### 2. Editing an off-chain pass's shader never redraws its tile (LOW)

`Pass.first_render_done` and `Pass.drawn_frame` are set in `Document.render` and reset nowhere —
not in `set_target`, `release_program`, `invalidate`, `rename_pass`, or the `watch.py` hot-reload
path. For an on-chain pass this is invisible, because the output render draws it every frame
regardless. For a pass *off* the output chain it means the sweep elects it exactly once per
process lifetime, so a later edit to its shader never reaches its tile.

Demonstrated headlessly on the five-pass bloom example with `output = "blur"`, which leaves
`trail` and `composite` off-chain:

```
drained, all stamped: {'blur': True, 'bright': True, 'composite': True, 'scene': True, 'trail': True}
after release_program: program = None  first_render_done = True
10 frames after the edit, composite drew: 0 times
composite program recompiled: False
```

Severity is LOW, and deliberately so: an off-chain tile already carries the dim + stale-corner
treatment (`pass_list.draw` builds `live` from `evaluation_order`), so the frozen picture is
marked as frozen rather than passed off as current — and it is strictly better than the black
tile that shipped before this commit. The compile error from a broken edit still surfaces,
because `panel_pass` calls `get_active_uniforms()`, which compiles on demand, and the user
editing that pass has its tab open. So nothing lies; the picture is just older than the source.

The wave spec's open question 4 covers `set_target` (correctly: no sweep runs on a frame a modal
is open, so zero frames are observable there) but not the hot-reload path, which has no such
protection.

**Fix:** clear `first_render_done` in `Pass.invalidate()`, accepting the termination property
moving from "each pass is elected at most once" to "at most once per invalidate" — bounded at one
election per save, which the spec's open question 2 already judged probably fine. Alternatively
leave it and let the stale wash carry the meaning, which is defensible today.

---

## What was checked and found correct

**The sweep (PASS).** Traced by hand and confirmed by probe:

- *Six-pass chain, output last, frames 0..2* — the output render (`target is None`) never skips,
  so the whole chain draws every frame; the sweep finds no pending pass after the chain is
  stamped and elects nothing.
- *Output first, five passes beside it, frames 0..6* — one off-chain pass elected per frame, each
  exactly once, drained in five frames. `test_every_pass_renders_once_within_n_frames` pins the
  no-double-election property; my own run over the bloom example agrees.
- *An iterated feedback pass off the output chain* — elected once, and it advances its ping-pong
  exactly twice for three iterations (the intra-iteration `_swap_feedback`, correct and identical
  to the on-chain path), then never draws again. Instrumented `_swap_feedback` per frame:
  `frame 0: elected=composite, trail swaps 2` then `1` per frame thereafter from `begin_frame`.
- *Examples-popup documents (`_frame == -1`)* — the skip cannot fire for them, and not because of
  `self._frame >= 0`: `ui.py`'s examples branch calls `render()` with no `target`, so the
  `target is not None` conjunct is already False. The `>= 0` conjunct is belt-and-braces, which
  is fine and is pinned by its own test.
- *The two output renders per frame* — neither is skipped. Both pass `target is None` (the preview
  one also passes `canvas`), and the skip requires `target is not None`. Confirmed by
  `test_two_output_renders_in_one_frame_both_draw` and re-derived by hand.

No scenario found where a pass draws twice in one frame, is skipped when it should draw, or
reads `first_render_done` / `drawn_frame` before it is set.

**Export (PASS).** `_render_image`, `_render_video`, `render_media` and `_render_media_into` are
byte-identical to `a246a19^` — verified by AST-unparse comparison, all four `IDENTICAL`. Export
passes `canvas=` and never `target=`, so the skip is structurally unreachable there. I also
probed the one indirect coupling worth checking: `render_media` → `reset_feedback()` sets
`_frame = -1`, and the export loop then re-counts 0..N-1 on its own clock, leaving `drawn_frame`
stamps that can collide with `app.frame_idx` when the live loop resumes. It is harmless: the
only passes an export stamps are output-chain passes, and the live loop draws the output chain
unconditionally every frame. An off-chain ancestor — the one case a stale stamp could suppress —
is never stamped by an export, because an export renders the output chain and nothing else.

**Commit-on-deactivate, the four correct sites (PASS).** At all six sites the deactivate query is
read on the line immediately after its `input_text`, with no intervening widget — checked
individually in `pass_settings.py::_draw_name`, `pass_list.py::_draw_add_input`,
`tree.py::_draw_inline_new_input`, `tree.py::_draw_file_rename_input`, and
`preview.py::_draw_function_tag_editor` (where both item-scoped queries correctly sit above the
`+ Add` button). `_draw_add_input`'s cancel handling is right: both the Escape branch and the `x`
branch set `wants_commit = False` before the commit runs.

A rejected name is terminal. `_commit_pass_name` snaps `pass_settings_name_buf` back to the live
name on every non-landing path — rejected, empty, and no-change — so a later deactivate cannot
re-fire the same rejection. Pinned by `test_a_rejected_rename_snaps_the_buffer_back`, which also
asserts the notification count.

**The gear's Close-after-edit frame (PASS).** Driven through the real `_draw_body` in a
six-frame rig: the rename lands on the deactivate frame, and `_draw_body` returns `True` on that
frame — so `ghost_button("Close")` was still submitted and a Close click on that frame registers.
The order is *rename first, Close consumed in the same frame*. The guard is correctly scoped to
the two sections that index by the dead name rather than being an early return, which would have
swallowed the click. Observed: `PASSES: ['k'] settings_name: k buf: 'k' keep_open: [True]*6`.

No path closes the gear without committing, and none commits twice. Both close paths funnel
through `App.close_pass_settings`; on the Close path the body has already renamed and
`_on_pass_renamed` re-pointed `pass_settings_name` *and* `pass_settings_name_buf`, so the funnel
sees `buf == name` and does nothing. On the Escape path the funnel is the only commit, since
`dispatch_commands` (`ui.py:351`) runs ahead of `draw_pass_settings` (`ui.py:444`) and the body
never draws that frame. Escape therefore **commits** a pending rename rather than cancelling it —
verified (`ESC RESULT: passes: ['escaped']`), and it is the specified behaviour, not a defect:
the wave spec's manual step "Rename by typing and pressing Escape" exists for exactly this.

**Add-pass activation (PASS).** Versus the tile-click path the only difference is the extra
`open_pass_settings(name)` and an unconditional rather than conditional `set_output_pass` — both
required by D10 (a fresh pass is never already the output, so the conditional would be dead), so
the two paths agree on everything they share. A rejected name is handled correctly: the error is
pushed, the function returns before `pass_add.close()`, and the input stays open with the text
intact — no half-activation, and the activate block is unreachable on that path.

**Commands (PASS).** `route_flag` returns `route_always` for any Alt chord, so Alt+A reaches the
dispatcher while the add-pass input is focused, which is what the new binding needs.
`popup_suppresses` returns `True` unconditionally, so Alt+P is suppressed behind any open modal
and cannot re-enter the gear from inside it. Both new callbacks guard on
`document_id not in self.ui_documents` and return early, so no current document is a no-op rather
than a `KeyError`; `panel_pass` is only reached past that guard, and its `render_pass` fallback
cannot see an empty `passes` dict because `delete_pass` refuses the last one.

One rough edge, not a bug: Alt+A while the add-pass input is already open and holds typed text
calls `InlineInput.open()`, which clears `buf`. The typed name is discarded silently. It reads as
"re-arm the input", which is a defensible meaning for the chord.

**Conventions (PASS).** No new `# type: ignore`, `# noqa`, `# pyright: ignore`, `TODO`, or
hand-rolled `push_style_color` anywhere in the diff. `Any` appears only on test helpers taking
the `app` fixture and on the monkeypatch-wrapper signatures, which is the existing test idiom.
The one function-body import (`from shaderbox.app import App` inside `test_lib_files.py`'s
fixture) is load-bearing — the `SHADERBOX_DATA_DIR` monkeypatch must land before `App` is
imported — and matches established precedent in `test_conversation_persistence.py` and
`test_cross_project_tools.py`. The new comments state present-tense non-obvious facts (the
item-scoped read ordering, why the cancel click drops the commit) rather than narrating history.

---

## Tests

`make test`: **1006 passed** at baseline and again after every falsification was restored.

Each new test was read for a way to pass other than the behaviour it names. Two worth recording
because the answer was not obvious:

- `test_a_picker_inline_input_commits_on_click_away` asserts `r.glsl` exists after typing a
  single `r` into a buffer pre-filled with `probe.glsl`. That is not a truncation bug in the
  test — imgui selects the buffer on a focus grab, so the keystroke replaces it. The test can
  only pass if the deactivate committed.
- `test_the_steady_state_draws_only_the_output_chain` installs its per-pass draw counter *after*
  the warmup loop, so the counts describe the settled state only, which is what the assertions
  read. Counting `Pass.render` is sound here because the stamps are written by `Document.render`,
  one level up — the counter cannot perturb what it measures.

Three falsifications re-run by hand (temporary edit, run, restore by re-applying the edit; never
`git checkout`):

| Mutation | Result |
|---|---|
| Drop the `self._frame >= 0` conjunct from the skip | RED — `test_the_skip_does_not_fire_without_a_frame_counter`, `assert 0 == 2`. 10 passed. |
| Drop the rejected-name `pass_settings_name_buf = name` reset | RED — `test_a_rejected_rename_snaps_the_buffer_back`. 26 passed. |
| Drop the add-pass activate block (`open_pass` + `set_output_pass`) | RED — `test_add_pass_activates_the_new_pass`. 26 passed. |

Each went red on exactly the intended test and nothing else, so all three are load-bearing rather
than incidentally green.

One process note on the restores. Re-applying the third edit by its own text matched a *different*
occurrence first (the context menu's `app.open_pass_settings(name)`), silently relocating the
block. `git diff` caught it and it was repaired by anchoring on unique surrounding context. This
is the mutation-restore hazard `conventions.md` already names — "a mutation test verifies its own
restore before anything else runs" — and the guard that caught it was checking `git diff` before
running anything, not the re-apply itself. Every restore here was confirmed with
`git diff --quiet <file>` before the next command.

---

## False trails

- *An export's `drawn_frame` stamps colliding with `app.frame_idx` after `reset_feedback()` drops
  `_frame` to -1* — real collision, no consequence: an export stamps only output-chain passes, and
  the live loop redraws those unconditionally.
- *A pass drawing twice when the sweep elects a pass whose ancestors are on the output chain* — the
  skip is exactly what prevents this, and it is pinned by two tests.
- *`Pass.drawn_frame` never resetting on `set_target`* — cannot be observed: the target only changes
  from the gear, and the sweep lives inside `if not app.any_popup_open():`, so no sweep runs on that
  frame. Correctly reasoned through in the spec's open question 4.
- *A one-shot sweep render allocating a `_feedback` canvas pair that then ping-pongs forever with no
  writer* — costs two texture handles per off-chain feedback pass and is picture-neutral (both sides
  hold the same last-written content). Pre-existing `_feedback` lifetime behaviour, identical to what
  an output change already produces; not introduced by this commit.
- *`_draw_add_input`'s Escape branch needing the same suppression as its `x` branch* — it does not,
  for the reason the spec gives: on the Escape frame the item is still active so the deactivate reads
  False, and on the next frame the input is not drawn at all.
- *The examples popup being saved by `self._frame >= 0`* — it is saved by `target is not None` one
  conjunct earlier; the `>= 0` guard is redundant defence, not the mechanism.

---

## Coverage

Read end-to-end: `shaderbox/document.py`, `core.py`, `ui.py`, `app.py`, `commands.py`,
`hotkeys.py`, `popups/pass_settings.py`, `popups/lib_picker/tree.py`,
`popups/lib_picker/preview.py`, `widgets/pass_list.py`, and the four test files
(`tests/test_pass_verbs.py`, `test_lazy_compile.py`, `test_lib_files.py`, `test_help_content.py`).
Also read for context: `project_session.py`'s six pass verbs, `watch.py`'s reload path,
`editor_types.py::InlineInput`, `tabs/document.py`'s `pass_list.draw` call site, the project
`CLAUDE.md`, `ai_docs/conventions.md`, `.claude/skills/imgui-ui/SKILL.md` § 7.5, and the wave
spec's open questions.

Not read: `ai_docs/features/069_tutorial_walk_findings/10_wave_c_pass_verbs.md` in full (consulted
by section — the census table, open questions 2-4, the manual-verification steps — since the spec
is the implementer's own account and this review's job is to check the code against behaviour, not
against the spec), and `reviews/wave_c_pre.md` (the pre-implementation round, deliberately left
unread so this review's findings are independently derived).

Verified before finishing: `git status --short` shows no modified files — only the two untracked
wave-A documents that predate this review.
