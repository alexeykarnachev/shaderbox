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

## W-1 deletion — DONE

done-condition, written before the wave: `grep -rnw` over the landed tree returns nothing for
each symbol; the string-reach check run separately, including JSON keys under `projects/`;
`make gates` green; no test assertion changed.

did: removed the four SAFE symbols the inventory found — `TelegramExporter.bot_token_present`,
`TOOLS_BLOCK`, `COLOR.SYN_PREPROC`, `experiment_dir`. All four were tier SAFE; nothing reached
CAREFUL or RISKY, because the inventory found nothing there.

`TOOLS_BLOCK`'s removal left a comment saying "the two request parts" above one constant, so the
comment was rewritten to describe what is actually there and where the `tools=` block is measured
instead. That is the comment tracking the code, not a comment wave.

string-reach: each of the four names grepped in quotes across all four trees and `projects/` —
nothing. No JSON key carries any of them, so nothing round-trips to disk.

verification: `make gates` exit 0, captured unpiped, smoke PASSED rather than skipped. No test
file changed at all — the diff is four source files, ten lines removed and two added.

ruled out: nothing needed the cascade rule. Removing these four orphaned no producer, unlike the
earlier sweep where deleting a field made a parameter dead.

## W-2 speculative generality — DONE

done-condition, written before the wave: `density` and `rounding` gone from `shaderbox/` by grep
over the landed tree; `_accent_primaries` still built from a NON-EMPTY `_ACCENTS`, asserted by
length against the landed tree; the theme invariant re-broken and observed to fire; `make gates`
green.

did: removed `apply_theme`'s `density` and `rounding` parameters, their two `Literal` aliases, the
two-branch density switch and the three-row rounding table. Every VALUE the taken branches set is
still set, written literally — the choice was dead, not the values. The module docstring's usage
example named both parameters and would now raise `TypeError`, so it was corrected in the same
commit.

**`accent` and `_ACCENTS` were kept**, which is the whole point of the wave being written out in
advance. The presets are the domain of the import-time theme-portability invariant, not just
`apply_theme`'s parameter values, and an assertion over an empty set passes — the naive removal
would have left a green gate over a check that no longer checks anything.

invariant re-broken, twice: before the change, `SELECT` set to the yellow accent's primary made
the import raise, proving the assertion was live to begin with; after the change, the same break
raised the same way, proving it is still live over a non-empty domain. `_ACCENTS` still has four
entries and `_accent_primaries` four members, asserted against the landed tree. `SELECT` restored
and absent from the diff.

verification: `make gates` exit 0, captured unpiped, smoke PASSED rather than skipped. One file
changed, 53 lines removed and 14 added. No test file touched — and no test covers this surface at
all, which is exactly why the re-break rather than the gate is what carries the verdict here.

## Docs — DONE

The roadmap row and the Active-context banner. `ai_docs/design/` was checked and deliberately
left alone apart from one line: it is an archived point-in-time snapshot whose own README already
says it has diverged from `theme.py`, so its quick-start snippet and `SPEC.md` keep the
density/rounding call they recorded. Rewriting evidence to match today's code would misrepresent
what that design pass produced. Only the README item that describes the LIVE module was corrected.

`ai_docs/features/006_inline_editor.md` still names `SYN_PREPROC` and was left as-is for the same
reason — a historical feature spec is a record of what that feature did.

## Behaviour questions for the maintainer

Nothing was found that needs one. The sweep changed no rendered value, and the one wave that
could have (`apply_theme`) was verified not to: both the old and new function were run against a
real `ImGuiStyle` and all thirty style fields, the five spacing tokens, the row height and the
three accent colors compared identical.

## Review of the landed sweep — one defect, fixed

An adversarial round over the three landed commits, anchored to the code and the library rather
than to this feature's own documents. It confirmed the sweep changed no rendered value by running
both versions of `apply_theme` against a real `ImGuiStyle` and diffing 127 style/spacing values
plus 63 color slots — zero differences. It re-broke the theme invariant independently and saw it
fire over a four-entry domain. It enumerated every `getattr` site in the tree and found none
reaching a removed symbol, and found no removed name in `projects/`, the resources, or the dogfood
run artifacts.

**The one defect it found was in my own doc reasoning.** The archived design README's Quick-start
block still passed `density` and `rounding`, so the snippet raised `TypeError`. My commit had
argued that the archive keeps its snapshot deliberately and corrected only the line describing the
live module — sound for `SPEC.md` and the feature checklist, which record what the 005 design pass
produced, but wrong for a block headed "Quick-start (literal Day-1 integration)". That block is
framed as instructions to run, not as a record, and the file was left saying the parameters are
gone eight lines above a call that passes them. Fixed, and the corrected snippet was executed to
confirm it runs.
