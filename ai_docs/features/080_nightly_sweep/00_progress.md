# 080 — progress log

What actually happened, appended after every wave, before the next one starts. A log, not a
plan — the plan is `01_spec.md`. **On resume, read this file and `git log` first.**

Entry shape:

    ## W-N <name> — DONE | SKIPPED | ABANDONED   <sha>
    done-condition: <written BEFORE the wave, as a checkable statement>
    did: <what changed>
    verification: <the gate's exit code>
    ruled out: <what was considered and rejected, and why>
    surprise: <anything the spec did not predict>

## Baseline — measured at `1439739` on `dev`

- `make gates` exit 0: check passed, test passed, smoke passed. Captured unpiped.
- 1835 tests collected; a full run is 1831 passed, 4 skipped (the 4 are prose-budget exemption
  rows, not GL skips). Test count is a snapshot — re-measure.
- `uv run ruff check --select F401,F811,F841 .` → all checks passed. Unused imports and locals
  are already absent, so the dead-code inventory is about what tools cannot see.
- Source is roughly 44k lines across shaderbox/, scripts/ and dogfood/; tests roughly 28k.
  Largest hand-written file is `shaderbox/copilot/backend.py`. Snapshot — re-measure.

## Phase 1 — spec prepared

Presence scan run as a parallel swarm, one agent per question, each asked for presence plus a
single example rather than a census. Constraints verified by hand rather than delegated: the
gate's honesty, what `make check` covers, which checks name file paths, whether an oracle
exists.

Ruled out during phase 1, with the reason, so the night does not re-open them:

- **A behaviour track.** `scripts/dogfood/judge.py`'s own docstring says it returns numbers and
  never a verdict; the judgement is a human reading the report. There is no oracle that decides
  whether a refactor preserved semantics, and unattended behaviour work without one is the thing
  not to do.
- **A dependency-direction wave.** The import-graph scan reported upward edges and then concluded
  in its own words that the modules it had labelled leaves are central hub modules — the labels
  were wrong, not the edges. `theme.py` importing `editor.ffi` is `editor_palette()`'s subject.
- **File splits.** The four largest hand-written files were read in full. Each is one cohesive
  concern that happens to be long; splitting would invent a seam rather than follow one.
- **A comment-density sweep.** Of the comment lines sampled, none restated the code. The
  convention is holding, so only verbatim duplicates and one attempt-chronology are in scope.
- **Two duplication candidates.** The publish prechecks share one `if` line and differ in every
  message; the `add_pass`/`set_pass` handlers share a shape and answer different questions.
  Similar shape is not shared meaning.

Errors made during phase 1, recorded because they are the failure mode this file exists to catch:

- A scan called `pass_graph.DTYPES` dead having omitted `tests/` from its grep. It is imported by
  three test modules. **The inventory wave greps including `tests/`.**
- An earlier sweep's inventory lists `TelegramExporter.bot_token_present` as removed at a commit
  whose message says so; `git show <sha> -- shaderbox/exporters/telegram.py` shows that commit
  never touched the file, and the method is still there. **A commit message is not evidence that
  a symbol is gone; a grep over the landed tree is.**
