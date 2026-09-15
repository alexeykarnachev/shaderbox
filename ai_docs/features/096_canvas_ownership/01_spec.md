# 096 — Canvas ownership

Six canvas defects in one week, three fixed and ten found. They are one class: **a canvas's
configuration (size, dtype, filter, wrap) is decided in several places, and each place knows
about some canvases but not all.** This feature fixes the class and gates it.

The evidence is `00_findings.md` beside this file — ten findings, each verified in-session
against the source rather than taken from the agent that reported it. Read it before doubting
anything here; this spec states what to DO and that file states why.

## Status

**Nothing has landed.** No wave has run. `02_progress.md` beside this file is the record of what
happens — write it as each wave lands, and read it plus `git log --oneline` first on resume.

## Goal

One track, and it is behavior work with an oracle: every defect below has a runnable
reproduction in `00_findings.md`, so correctness is decided by a script rather than by judgment.

The deliverable is a tree where a canvas's configuration has ONE home, every canvas that must
follow a change does follow it, and a check fails when a future change forgets one.

Explicitly out of scope: the `CanvasSpec` type refactor the design round proposed
(`Canvas.of()` / `conform()` replacing loose kwargs at seven construction sites). It is the right
end state and it is not this feature — W-5 below leaves the note that earns it. Fix the defects
and install the gate first; a type refactor on top of an ungated tree is how the third bug landed
after the first two had already written their rule into the conventions.

## What is broken

Each row has its verified reproduction in `00_findings.md`. Re-run it before fixing; do not
trust the description.

| # | Defect | Severity |
|---|---|---|
| F6 | A self-reading OUTPUT pass exports a FROZEN video at every iteration count — the last iteration draws into the caller's canvas, leaving the pass's own canvas unwritten, so the swap has nothing to advance | ships to users |
| F1 | The export's `RENDER_AT_TARGET` branch allocates its canvas with no dtype/filter/wrap while the sibling branch twelve lines up copies all three; Telegram and the shared shapes take the lossy one | ships to users |
| F2 | Promoting a scaled pass to output strands it at the scaled size forever — the viewer and every export then read a half-size canvas | one tile click |
| F9 | A pass file with no graph entry is born `f1` while the graph reports `f2` | latent |
| F10 | `newest_frame` returns the OLDER canvas after a render into an external canvas | latent |
| F7 | `Document.__init__` does not clamp `canvas_size` though its comment says it does | latent |

Two structural findings behind them: the sizing rule is written four times in three syntactic
shapes plus a fifth WRONG copy in `popups/pass_settings.py` (no output exemption on the label),
and nothing anywhere reconciles a canvas to its graph entry.

## Why the suite missed all of it

Verified statically: **every `filter` and `wrap` assertion in the suite is at pass BIRTH or
inside one of the two tests written by this week's own fixes.** Nothing asserts either after a
resize, after a swap, or on an export canvas. Deleting `filter=` and `wrap=` from
`resample_canvas` — which resets every canvas in the document on every resize — passes the whole
suite.

Coverage runs size > dtype > filter > wrap: the order defects have been found in. The suite
documents history rather than the contract, and each of the three fixed bugs is caught by exactly
one test, the one its own fix wrote.

**Two trap shapes this session fell into while writing those fixes.** Both must be avoided by
every test this feature adds:

1. **Rendering before asserting.** `render`'s lazy fix-up repairs non-output passes, so a test
   that renders first passes whether or not the operation under test did anything.
   `test_a_resize_moves_every_pass_together` is vacuous exactly this way and its own comment
   concedes it.
2. **Testing the default.** `DEFAULT_FILTER_LINEAR` is True and `DEFAULT_WRAP` is False, so a
   check written with defaults cannot fail. The first test written for the feedback fix asked
   for LINEAR when LINEAR was already the default and passed with the bug present.

## The waves

One wave, one commit, `make gates` green before and after each, each independently revertable.
Order matters: W-0 first because everything else is measured against it, W-4 last because it
gates what the earlier waves fixed.

- **W-0 — the invariant checker, as a test helper. No production change.**
  One function over every canvas reachable from a Document: for each pass, the live canvas's
  size/dtype/filter/wrap against what its graph entry implies (output sized to the document, not
  to its own scale); for a pass with a history, the history against the live canvas on all four;
  for the blit, its canvas's filter against its source texture.
  Driven over the NON-DEFAULT corner — `filter_linear=False`, `wrap=True`, `dtype="f4"`,
  `scale=0.5` — and asserted immediately after each operation and BEFORE any render.
  It must FAIL on the current tree for F2 and for the `resample_canvas` mutation. A W-0 that
  passes everywhere is a W-0 that checks nothing; prove it red before proceeding.

- **W-1 — F6, the frozen export.** DECIDED, do not re-open: the output pass ALWAYS draws into
  its own canvas, and when the caller supplied one, the result is blitted out afterwards.
  `Document` already holds a `CanvasResampler` (`self._resampler`) whose `blit(source_texture,
  target_canvas)` draws a whole texture into a target of any size, which is exactly this
  operation and is already the resize path's tool. So the fix is: drop `draw_into` from the
  iteration loop (every iteration goes to `render_pass.canvas`), and after the loop, if `canvas`
  is not None and this is the output pass, blit the pass's canvas into it.

  Rejected: drawing twice (doubles the output pass's cost every export frame), and keeping the
  last iteration external while separately writing the pass's canvas (two writes of the same
  picture, and the bug's own shape — a rule applied at one site and not its sibling).

  The existing comment at that loop already argues the early iterations must hit the pass's own
  canvas so the swap can advance; it stops one step short of the last one. Rewrite it to state
  the rule once.

  Done when `repro/f6_frozen_export.py` prints a climbing sequence at N=1 and N=2, judged on the
  decoded video, and `make gates` is green.

- **W-2 — F1 and F9, the default mismatch.** Three changes, all DECIDED:
  (a) the export's fit branch copies `dtype`, `filter` and `wrap` from the output pass's canvas,
  exactly as its sibling branch twelve lines up already does;
  (b) `load_from_dir` builds a pass with no graph entry from `PassEntry().target` rather than
  `None`, so the canvas gets the same defaults the graph is about to be backfilled with;
  (c) `Canvas`'s own `dtype` default changes from `"f1"` to match `TargetConfig`'s `f2`.

  (c) is the one that makes the class hard to repeat, and it is a real behavior change: any
  remaining bare `Canvas(...)` becomes `f2`. Search for every construction site before doing it
  and state in the commit which ones changed shape. If a site genuinely wants 8-bit it now says
  so explicitly, which is the point.

  Do NOT remove the default outright — that is a larger signature change and belongs with the
  `CanvasSpec` refactor, which is out of scope.

- **W-3 — F2, F7, F10, and the sizing rule that is written out at five sites.** Four sub-fixes.
  Land them as ONE commit only if the gate is green after each; otherwise split.

  (a) **The sizing function. DECIDED, do not redesign:** a method on `Document`,

          def canvas_size_for(self, name: str) -> tuple[int, int]:

  returning `self.canvas_size` when `name` is `self.graph.output_pass`, and
  `entry.target.target_size(self.canvas_size)` otherwise, with `PassEntry()` as the fallback for a
  name the graph does not hold. A method on `Document` rather than a free function or a
  `PassGraph` method because it needs BOTH the graph and `canvas_size`, and `Document` is the only
  object holding both — the recorded rule is that the document owns the size and applies each
  pass's scale.

  Note `PassGraph.output_pass` returns `str | None` — it is None when the graph names no pass that
  exists. Handle that explicitly rather than comparing a name against None, which would silently
  make EVERY pass non-output and scale the whole document down.

  Replace all four in-document copies with a call to it: the two loops in `set_canvas_size`,
  `_seed_feedback`, and `render`'s fix-up. Afterwards a search for `target_size(` in
  `shaderbox/document.py` finds no other caller.

  (b) **The settings modal's label** calls the same method instead of computing
  `canvas * scale` itself, which is why it currently shows an output pass a size it does not
  have. It has a `Document` in scope.

  (c) **F2, promotion:** a pass becoming the output must be resized to the document size. Put it
  where the output changes so it cannot be forgotten by a future caller, not in `render`.

  (d) **F7, the clamp:** `Document.__init__` applies `clamp_canvas_size` as `set_canvas_size`
  does, so the comment claiming both writers normalize becomes true.

  (e) **F10, `newest_frame`:** it must not name the older canvas after a render into an external
  canvas. The findings file has the trace.

  Done when `repro/f2_promote_scaled.py` prints `BUG: False`, a search shows one sizing site, and
  the gate is green.

- **W-4 — the gate.** Wire W-0's checker into the suite as real tests over the operation battery:
  `set_canvas_size`, `set_pass_target`, output change, `add_pass`, `rename_pass`, `delete_pass`,
  `load_from_dir`, `begin_frame`, `render`, and an export. Each must be broken on purpose and seen
  to fail. **A gate that has not been broken is a wish** — the commit says which break was tried.

- **W-5 — the note that earns the type refactor.** Record in `conventions.md` what this feature
  established: where canvas configuration lives, which canvases must follow a change, and that a
  future `CanvasSpec` is the end state. One paragraph, no TODO list.

## How correctness is decided

`make gates` — check, then test, then smoke, one exit code. Run it unpiped and read the status
before anything else:

    make gates > /tmp/g.log 2>&1; echo $?

Plus, for this feature specifically, the reproductions in `00_findings.md`: each defect's script
must go from failing to passing, and the F6 one must be judged on the DECODED VIDEO rather than
on a canvas read — a canvas read is what made one agent report F6 as sound.

**A changed test expectation is a defect in the change, not a test to update.** The one
legitimate edit is a test whose subject genuinely moved.

**Run the gate on a box whose GL stack is fresh.** The investigation session that produced this
spec ended unable to run the suite at all: every pytest process segfaulted at context creation,
including pure-logic modules, after six agents and dozens of probe scripts had each built
standalone GL contexts. `git diff HEAD -- shaderbox/ tests/` was empty throughout, so the tree
was provably unchanged from the last green gate.

The first thing the executing session does is therefore `make gates` on the untouched tree, to
establish a real baseline. If it is red with no source diff, the box is the problem, not the
repo — restart the session or the machine rather than debugging a phantom regression. A gate that
cannot run is not a red gate, and neither is it a green one.

Related and genuinely pre-existing: `test_gl_lifetime_guards.py`'s last test creates a SECOND
standalone context and aborts when contexts are scarce. Reproduced on a pristine worktree, so it
predates this work.

## Cold start

1. **Read `00_findings.md` first**, then `02_progress.md` and `git log --oneline`.
2. **Re-measure before acting.** Every defect here was verified at a commit that is now behind
   you, and this session may have fixed adjacent code. Re-run each reproduction in
   `repro/` before fixing it, and re-run the searches the findings cite rather than trusting a
   count or a line number in prose. Do not trust a description over a run — two of the ten
   findings arrived from agents with the wrong scope, and both were caught by re-running.
3. **Work in wave order**, W-0 through W-5. W-0 is measurement and comes first.
4. **Write each wave's done-condition before starting it** as a checkable statement, and append
   the result to `02_progress.md` before starting the next.

Settled, and NOT to be re-opened:

- The output pass renders at full document size and its own `scale` is ignored. The settings
  modal disables that slider and says "output always full". F2 is not a case for honoring the
  scale; it is a case for resizing to FULL on promotion.
- Feedback swaps at the FRAME boundary via `begin_frame(frame)`, idempotent per frame number.
  Do not move it into `render()`.
- The document owns the canvas SIZE and applies each pass's `scale`; a pass never sizes itself.
- NO backward-compatibility or migration code, ever. A reshaped model just changes, and
  `projects/dev/` is hand-fixed in the same wave.
- The `CanvasSpec` type refactor is OUT of this feature (see Goal).
- These are NOT defects and must not be "fixed": the blit not carrying dtype or wrap (its canvas
  is only ever sampled by imgui, where neither can show), the Share previews showing the decoded
  artifact, and minifying tiles.

## Running this unattended

This feature is written to be executed end to end with nobody watching. The rules below are the
contract for that run; they matter more than any individual wave.

**Carry on to the end. Do not stop to report, and do not ask.** Every decision this feature needs
is already made — the two that were open (W-1's approach, W-2's default) are marked DECIDED above.
If a genuinely new question appears, pick the option that keeps the tree green and the change
smallest, write what you chose and why into `02_progress.md`, and continue. Stopping with waves
unfinished is the failure mode; a wave finished under a stated assumption is not.

**Commit each wave as it lands and push.** A local tree ahead of `origin` at the end of the night
is a failure state — the work must survive the session. One wave, one commit, `make gates` green
before and after each.

**Never weaken a test to get green.** A failing test means the change is wrong: revert the change
and record why in `02_progress.md`. The one legitimate edit is a test whose subject genuinely
moved.

**If the baseline gate is red with an empty source diff, stop and write that finding into
`02_progress.md`.** That is the one legitimate early exit: the box cannot run the suite, so no
result the night produces would be trustworthy. Do not "fix" it by skipping tests.

**If a wave turns out bigger than the spec assumed**, finish the waves that are not blocked,
and record what you left and why. Scaling the work down is fine when it is written down; silently
narrowing it is not.

**Report at the end against the milestone, not the effort.** Which waves landed, which defects
have a passing reproduction, what the gate says, and what was left. A count of tool calls or
agents says nothing about whether the defects are fixed.

**Spawning agents during this run:** allowed for a focused review of a landed wave, on opus, one
at a time. Every such agent is measurement-only unless it is an implementor working in its own
wave — three agents in the investigation round left falsifier edits in the working tree despite
being told not to, so check `git status` after any agent returns, and restore anything you did
not change yourself.

## Coverage claim

scanned: every `Canvas(` construction site, every `.canvas =` assignment, `set_size`,
`resample_canvas` and `.texture` reference across `shaderbox/`; the feedback machinery end to
end; every surface that displays or exports a pass texture; and the `filter`/`wrap`/`dtype`/size
assertions across the test suite.

not scanned: the editor's own render target (`editor/render.py`) beyond noting it is independent
of the document graph; user-bound uniform textures past observing that they drop the source's
filter and wrap on copy, with no decision record found either way; whether `EditorRenderer.atlas`
having no release site is a leak or accepted context teardown.
