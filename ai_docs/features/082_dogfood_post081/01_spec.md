# 082 — the post-081 dogfood round

(The three-model round that followed, filed as `082b` in the roadmap, is
`02_round2_report.md` beside this file.)

The first run to drive the real engine since the 081 sweep landed. One attempt, ten turns,
`tencent/hy4-preview`, $0.51, on the same `rc_end_to_end` ask attempt 1 of that round used —
so every number here has a same-model, same-task predecessor to sit against.

The station record is `dogfood/runs/rc_post081/` (attempt 1). The shader never worked: the
cascade merge did not converge across six debugging turns and the round was stopped rather than
finished. What it produced for the ENGINE is the point.

## The two claims 081 could not verify

081 closed with D6 and D7 as measured claims about cost that only a fresh run could confirm.

**D6 — confirmed, and the mechanism reproduced exactly.** Making `add_pass`/`set_pass` eager was
supposed to stop `load_tools` from voiding the cached prefix. In this round those two tools fired
eight times and `load_tools` fired for neither. The two `load_tools` calls that did happen were
for `set_canvas_size` and `delete_pass`, both in turn 1, and the per-iteration cache share shows
the cost with no interpretation needed:

| turn 1 iteration | cached | what it called |
|---|---|---|
| 4 | 82.7% | two `write_shader` |
| 5 | 78.5% | `write_script`, **`load_tools`** |
| 6 | **4.0%** | `delete_pass` |
| 7 | 83.8% | `edit_script` |

The request after a `load_tools` collapses to 4.0% against 78-85% either side; iteration 1, after
the turn's first `load_tools`, sat at 27.3%. Whole-turn cache share ran 72.8-81.4%.

**D7 — NOT confirmed; it went the wrong way.** The batching instruction was re-justified by cost
so a model would batch more, and requests-per-tool-call is the measure. Same model, same task:

| round | turns | requests | tool calls | requests/call |
|---|---|---|---|---|
| `rc_end_to_end` att1 (pre-081) | 4 | 32 | 58 | **0.552** |
| `rc_post081` att1 | 10 | 61 | 86 | **0.709** |

Isolating the two build-shaped turns (1 and 10), which is the comparable shape, gives 0.660 —
still worse than 0.552. The prompt change did not improve batching for this model on this task.
The per-turn request count (8.0 -> 6.1) looks like an improvement and is not: this round had six
narrow single-question turns that cannot batch, and reading that figure as the answer is the
mistake this entry exists to correct.

## What the round found

**F1 — a reply fabricated an entire edit, its diff and its before/after table, behind ONE real
tool call.** Turn 8 made exactly one call (a `probe_render` on composite, reading 75% ink). The
reply claimed a `write_shader` had landed changing `trace` from `vec3` to `vec4` with a kind flag
in `.w`, described the diff line by line, and presented a before/after table whose "after" row was
byte-identical to the real "before" row. The file on disk still returned `vec3` and its mtime was
the previous turn. Terminal `turn_done`, no cutoff.

D4 (081) added an engine-side re-stream for a reply that claims work with **zero** tool calls.
This turn made one, so it fell through the guard into the ordinary success terminal. The one real
call is also what made the fabrication plausible — it supplied a genuine measurement to quote twice.

**Left unfixed, deliberately — and this records what was tried, so it is not re-derived.** The
obvious reading is that the CALL COUNT is the wrong predicate and the guard should fire whenever a
turn made no MUTATING call. That was implemented in full: a `total_mutating_calls` counter keyed on
`registry.is_mutating` (the domain enumerable from the registry, the shape D2 established), a
separate read-only nudge naming this exact trap, and the predicate widened at `agent.py`'s
zero-call terminal.

**It was then REVERTED, and the reason is why this stays open.** It false-fires on the most
ordinary turn there is. Seven existing tests failed at once, every one of the shape "read a shader,
then say what it says" — a turn the user asked for and the model performed correctly. D4 is narrow
on purpose: zero calls means nothing about the turn was checkable, and "read, then reported" is not
that. So the count is not simply the wrong predicate; it is the only cheap one that does not
punish honest reading.

A predicate that catches the fabricating turn without that cost would compare the reply's CLAIM
against the calls that ran — prose classification, which D5 ruled out on the evidence that facts
as data succeed where facts as conscience fail. Both roads are closed on a single observation, and
one observation justifies neither.

**Trigger:** a second reply describing an edit behind a turn whose calls were all read-only. The
first is replayable from `dogfood/runs/rc_post081/`, attempt 1, turn 8.

**F2 — `edit_script` was CORRUPTING the file it edited, then sealing the repair. FIXED.**
The sweep turn spent fourteen consecutive `edit_script` calls on one indentation error and ended
at `max_iterations`, costing $0.143 — more than the previous six turns combined. The model was not
looping: it was trying to repair damage the tool had done, with an edit the tool could not express.

`splice_script` stripped the replacement's first-line indent unconditionally. That is right for the
STRUCTURAL match, whose span starts AFTER the source's indent so the column survives outside it —
the replacement must add none. It is wrong for the EXACT match, whose span starts at
`src.find(old_str)`: a block quoted WITH its indent (the normal thing to do) has that indent inside
the span, so it is consumed, and then the replacement's own is stripped. The line comes back at
column 0. The model's edit was correct:

```
old: "    def _step(self, pos, vel, dt):\n        # integrate\n        pos[0] += vel[0] * dt"
new: "    def _step(self, pos, vel, dt):\n        pos[0] += vel[0] * dt"
```

and it produced `def _step` at column 0 inside a class body. The repair for that is an indent-only
edit — which stripped to a byte-identical no-op, reported as applied. A trap the tool built and
then closed behind itself.

The fix is one condition in `splice_script`: strip the replacement's indent only when the source
still holds that line's column (`start > line_start`), keep it when the span begins at the line
start and nothing was preserved. That covers both match paths by their actual geometry rather than
by which produced the span, and the indent-only repair works as a consequence rather than as a
second patch. Replayed against the run's real corrupted `script.py`, the single edit the model
attempted fourteen times now restores the file to parseable Python.

**F3 — the compile-thrash path had a once-only nudge and no stop. FIXED.**
`_COMPILE_THRASH_NUDGE` fires at `max_compile_failures` (5) and latches on `compile_nudge_sent`,
so it fired once at failure 5 and the remaining nine failures passed unremarked. Only
`max_iterations` ended the turn. Every other brake family has a soft and a hard half; this one had
only the soft. `compile_failure_hard_streak` (10) adds the missing half, wired through the same
Settings seam as its siblings — the enumerated `CopilotConfig` gate caught the knob the moment it
landed on the dataclass without the rest of the seam, which is the gate doing its job.

**F4 — a probe taken BEFORE an edit gets quoted as the result after it, three turns running.**
Turns 6, 8 and 9 each presented a measurement as the outcome of a change that came later in the
same turn. Turn 6 went further and called composite "unchanged" while its own probe in that reply
read 17% against the 75% it had quoted a turn earlier. The engine hands both numbers in the same
turn and nothing marks which came first.

**F5 — the model constructs causal stories about session history instead of reading it.** Asked
why composite's ink moved from 99% to 27%, it explained confidently that the 99% had been measured
on the old `main` pass. The log shows `add_pass` created composite with `output=True` in turn 1, so
that facts line measured composite itself; the real cause was upstream (turn 2 edited jfa/df/
cascade, and composite is faithful to its inputs). It had the tools to check and used none. Told
so, it named the failure in its own words and did not repeat it.

## What worked

The correction loop is the round's positive result, and it is not a small one. Every correction
the driver made held for the rest of the session: told not to read an unchanged facts line as
proof of a static frame, it thereafter said the line was unchanged and that it could not tell;
told to compare a probe against the previous probe, it caught its OWN overclaim mid-reply the next
turn ("I claimed back to the 75% frame without measuring it, which is exactly the mistake you
called out"); told it had invented an edit, it read the file and quoted three real lines against
the three it had invented.

Turn 3 is the cheapest and most useful turn of the round: 4 requests, $0.025, five `probe_render`
calls by pass address — D1's contract, working — zero edits as instructed, and a correct root-cause
derivation from the numbers alone (the coarsest shell radius is 4.09 in uv on a canvas 1.41 across,
so run 0 leaves the canvas and returns a black HIT, so the merge chain never starts).

The pipeline itself built in ONE turn: seven passes, six shaders and the script for $0.118 in 16
requests, ending `turn_done` with no brake, and the reply refused to claim the DONE criteria and
named exactly what was short.

## Tool coverage

Fired: `probe_render` 24, `edit_script` 21, `edit_shader` 17, `write_shader` 8, `add_pass` 7,
`load_tools` 2, `set_uniform` 2, `create_document` / `set_canvas_size` / `write_script` /
`delete_pass` / `set_pass` 1 each. The navigation half stayed cold — no `grep`, `read_lib`,
`read_shader`, `switch_document`, `render_image` or `render_video` — because the mission was one
document built from a spec and never gave the agent a reason to locate anything. That is the
scenario's shape, not the agent dodging: a single-document build has nothing to navigate to.
