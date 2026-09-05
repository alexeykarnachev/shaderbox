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

## Spec review — two adversarial rounds, three waves closed

Two reviewers ran against the first draft, one re-deriving every factual claim from the code, one
attacking the argument with the repo's own docs as its anchor.

**The evidence round confirmed all nine factual claims**, with one arithmetic correction: the
earlier sweep's "Removed" table has ten rows, nine removals plus one pointer into its KEPT
section, so eight of nine landed rather than nine of ten.

**The argument round closed three of the five proposed waves**, each because the question was
already settled and the spec had not noticed:

- **The comment wave.** Its target was the five-site EGL fixture block. The earlier sweep's
  progress log records that exact block as KEPT, because it quotes a measured segfault. The
  wave's own done-condition forbade shortening a comment that quotes a measurement, so the wave
  contradicted itself.
- **The repetition wave.** Its motivating case was the document-tab case list. `ui_regions.py`'s
  own docstring says the split is deliberate: the enum is a leaf with no imgui in it, and
  hoisting it beside the command table would drag imgui into the headless model layer.
- **The relocation wave.** Both candidates were misreadings. `intel/document.py` is named for the
  document BUFFER whose intel it caches, keyed on an editor handle, and the conventions name it
  in that role; `watch.py`'s placement is recorded in the module map. The earlier sweep had also
  scanned this dimension and found it absent.

**The surviving speculative-generality wave was rewritten rather than kept as drafted.** Its
done-condition — "no parameter remains that every call site passes identically" — was satisfied
by a change that makes a real invariant vacuous. `_ACCENTS` is the domain of an import-time
theme-portability assertion, no test touches any of it, and an assertion over an empty set
passes. The wave now removes only the two parameters with no second job, keeps the presets, and
requires the invariant to be broken on purpose and observed to fire.

Also folded in: two KEPT entries the earlier sweep recorded and this spec had dropped
(`GraphError.message`, `LookupPopup.word` — read from tests only), the section-banner protection,
and a falsifier for the inventory wave, whose done-condition previously checked that a coverage
claim was WRITTEN rather than that the scan worked.

## W-0 inventory — DONE  (no code changes)

done-condition, written before the wave: every symbol kind scanned with a coverage claim naming
what it did not cover; every row of the earlier sweep's "Removed" table verdicted against the
current tree; the scan falsified against a symbol live only through the test tree.

did: `00_inventory.md`. AST enumeration of every declaration cross-counted against a word index
of all four Python trees plus the build and CI files. Four dead symbols across the whole
repository.

falsifier: `pass_graph.DTYPES` is live only through three test modules. The scan does not report
it dead, and the inventory lists it under KEPT for the next sweep that re-finds it.

surprise: the first run of the scan indexed `ai_docs/` too, so `bot_token_present` — dead in
code, named in three markdown files — scored four references and was filtered out of the
candidate list. Docs no longer count as code references. A symbol mentioned only in prose is
dead code with a paper trail, which is what the inventory exists to find.

verification: no code changed; `make gates` not re-run for this wave.
