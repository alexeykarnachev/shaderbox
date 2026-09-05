# CELL 6 — F7/F8/F9 station + methodology. Moderator: opus. VERIFIED by main session.

## MY F7 WAS AN OVERCOUNT — corrected
VERIFIED: attempts 4-8 opened at 11:05:51.196, :52.134, :53.056, :54.002, :54.924 — FIVE ATTEMPTS
IN 3.7 SECONDS, all on sha=3da796abd. e2e 1-3 opened in 1.6 seconds on 134bbf02d.
These are PARALLEL FAN-OUTS. start_attempt WAS called all 11 times; commits_between correctly
returned empty because HEAD had not moved. They are CORRECTLY EMPTY, not silent failures.

THE REAL HOLE is narrower and permanent: because a round's attempts are all opened BEFORE any
run, the commits a round PRODUCES land after every attempt of that round was opened, so no later
start_attempt can sweep them — it starts from its OWN previous sha.
Unreachable forever (verified `git log --oneline --reverse 3da796abd3..134bbf02de`):
  c8960e1 Engine findings from the model comparison, round one
  60e089b Formatter pass over agent.py
  134bbf0 077: the radiance-cascades build on seven models
plus b54baad after the last attempt of the last experiment. THREE COMMITS, not five attempts.

## The gate exists, passes, and is blind to the shape that failed
tests/test_dogfood_station.py:224 test_next_attempt_records_the_commits_landed_between covers the
SEQUENTIAL case only (commit, commit, start_attempt, assert). Green, in `make test`, blind to the
parallel shape. The maintainer's own "checker that quietly narrows its own domain".
Failure is silent by construction: build.py:461 renders the fixes heading only `if fixes`;
build.py:545 renders `len(a.fixes)`. Zero real commits and three unswept commits render
IDENTICALLY. `grep commits_between` = 2 hits (definition + its single caller). Nothing cross-checks.
SKILL.md:51 says a model comparison "belongs" in the station; the whole drive procedure documents
only a SEQUENTIAL loop. grep for parallel|concurrent|fan-out over SKILL.md = ONE hit, that line.
The 077 run invented the parallel shape on the spot and the ledger degraded without a word.

## A THIRD instance of the same class — VERIFIED
station.py:189 records `"dirty": bool(git status --porcelain)` on every attempt.
ALL EIGHT rc_full_build attempts recorded dirty=True — those runs provably did NOT execute the
code their sha names. (e2e attempts are all dirty=False.)
And Attempt (log.py:214-227) has NO `dirty` FIELD — verified: written to JSONL, dropped on read,
surfaced by no page and no test. A write-only integrity signal, already firing, already ignored.

## F8: the skill has an ADD seam and no RE-CHECK seam
SKILL.md:485-490 §"Improve this skill": "This is a LIVING skill. Each run... ADD it here."
Nothing instructs re-verification. grep for verify|re-check|stale|provenance finds no self-audit.
THE MECHANISM, OPERATING: the 1.07-1.08 sentence entered in fb7a85f (feature 075), BEFORE the 077
runs. Those runs produced 475 context panels each rendering the TRUE ratio live
(build.py:219 `f"estimate/billed {total / billed_in:.2f}"`). Every one was below 1.07. The 077
commit 134bbf0 then EDITED SKILL.md — adding a new gotcha, leaving the refuted number untouched.
WORSE: that same edit PROPAGATED a claim the corpus disproves — SKILL.md:328-332 now states
fabrication "happens on resume... a long engine ledger", copied from 01_report.md:74-79 and
disproven by cell 1 (2 of 6; 19 of 21 non-fabricating). And its count is wrong: six, not five.
=> The drift is not decay; it is ACTIVE PROPAGATION OF UNVERIFIED UPSTREAM PROSE.

## The skill is ~85% sound — the rot is confined to unprovenanced numbers
Of ~73 falsifiable claims: ~50 TRUE, 2 FALSE, 2 STALE, ~20 UNCHECKABLE-but-honestly-scoped
(dated empirical incidents naming their date and box — the CORRECT shape).
 FALSE 1: the 1.07-1.08 ratio (line 390). 0 of 475 at or above 1.07; mean 0.8914.
 FALSE 2: the fabrication cause and count (lines 328-332).
 STALE: "5 of ~12 reachable tools" (line 184) — REACHABLE_TOOLS is 26 today.
 STALE: the markdown REPORT_TEMPLATE flow (429-465) — superseded by the attempt page since 075.
What SURVIVES is code-grounded: the COPILOT_CONFIG/COPILOT_ENGINE clobbering asymmetry (250-265),
the drive_until_idle bridge-pump ordering (234-241), the dump() JSON contract (123-127), the
cumulative-vs-peak token distinction (416-419), the judge.py primitive list (371-373, all seven
exact). Several match code comments near-verbatim — the skill was largely written by READING
SOURCE, which is why the rot sits only in numbers nobody re-measured.

## F9: the report had no section for METHOD
01_report.md headers are all findings ABOUT THE MODELS (The table / Per model / What the run found
in the engine / Left for the sweep / Next round / The end-to-end round). A technique the driver
invented fits none. grep cross-scene|isolat|"shipped example's scene" over both 077 docs = ZERO.
judge.py has 7 primitives, each with exactly one test; NONE computes a lit fraction or an
adjacent-row difference — the two numbers the driver quoted EVERY time. The harness's render
methods already take a document_id. Missing: a lit-fraction primitive; a cross-project source copy.

## Knowledge: 2 items genuinely lost (not 16 — my count was of report-echo, not of capture)
LOST: the cross-scene diagnostic (F9); the A,B,A,B alternating-call brake gap (named in per-model
prose at 01_report.md:51, absent from "Left for the sweep" and from any code TODO); the probe
canvas-size mismatch left "as is" in fb att2 t4 with no TODO at the code site.
CORRECTION TO A SUBAGENT FINDING THE MODERATOR CHECKED: the no-op churn brake gap from fb3-t5 is
NOT lost — config.py:58-59 states the counter is "not reset by a write" and cites the exact run
("thirteen whole-file rewrites of two files that changed nothing on screen"). That fix landed.
Scoring it lost would have sent the spec chasing a CLOSED item.

## Proposals
P1 (DO FIRST) — the station asserts its fix ledger against git.
  station.py: `ledger_gap(exp_dir, repo) -> list[dict]` — per attempt, commits between the
  previous sha and this one carrying no fix record, PLUS commits after the last attempt's sha up
  to HEAD. build.py renders a gap as a visible WARNING ROW (it already has a .warn span at :224)
  rather than an absent section. The parallel-burst case falls out: same sha, empty window, no gap.
  SUBSUMES the dirty finding — surface dirty=True in the same row; a recorded sha that never ran
  is the same lie as a missing commit.
  THE BREAK (two, and the second is the one that matters): (1) take the real 077 log, drop attempt
  4's six fix records, assert the check names exactly those six shas; restore, assert zero.
  (2) REPLAY THE att4-att8 PARALLEL BURST AND ASSERT IT REPORTS ZERO GAP — a check that flags
  correctly-empty windows would be worse than none.
P2 — the estimator constant is measured, not assumed.
  prompt.py:16 `_CHARS_PER_TOKEN: int = 4`, duplicated as a bare `// 4` at
  context_breakdown.py:128 (VERIFIED — two constants), ZERO tests.
  Measured implied divisor across 475 records: 3.57 (per model 3.39-3.85; 3.6 lands every model
  within ~7% against the current 4's systematic 11% under-count).
  Set 3.6, import it at the second site so it stops being two constants, add a band test. Then
  DELETE SKILL.md:390 rather than correcting it — build.py:219 prints the live ratio on every
  panel; a prose restatement of a rendered number can only go stale again.
  THE BREAK: set the constant to 4, assert the test fails naming the ratio; restore to 3.6.
P3 — lit_fraction joins judge.py (numbers out, never a verdict), and the skill's existing
  pixel-measurement paragraph (371-377) names the cross-scene diagnostic beside the primitives.
  CONSTRAINED BY conventions.md:909-916, a SETTLED decision: "The dogfooding station records; it
  never judges... no standing checker is wired into a run's verdict... A measurement that answers
  one question in the moment and is then deleted is fine; one that STANDS is the failure."
  A judge.py primitive the driver calls BY HAND is the sanctioned shape. WIRING a cross-scene
  check into any verdict WOULD VIOLATE the convention — not proposed, and a spec that does should
  be rejected on those grounds.
  THE BREAK: synthetic array with a known count above threshold, assert the exact fraction; flip
  one texel, assert the fraction moves by exactly 1/N.
  The cross-project source-copy half stays a MANUAL procedure until a second run needs it —
  building the rig on one round's evidence is the overfitting the convention warns about.
P4 — two lost items land as TODOs AT THEIR CODE SITES (never in todo.md, frozen drain-only): the
  probe canvas-size seam; the A,B,A,B 2-cycle the repeat guard does not see (its guard is "calls
  already made this turn with the same arguments" — sees repeats, not a 2-cycle across two sites).
NOT PROPOSED: a "did you remember to record?" checklist item in the skill — a wish with no gate,
  and the exact mechanism that produced all three findings.

## False trails
- "The driver forgot to call start_attempt" — called all 11 times.
- "Attempts 5-8 and rc_end_to_end lost their fix events" — correctly empty (unmoved HEAD).
- "The report page hides the estimate/billed ratio" — it renders it on all 475 panels.
- "The no-op churn brake gap is unfiled" — filed AND fixed at config.py:58-59.
- "All 11 fix events are future-dated" — the moderator's own timezone error, retracted (station ts
  is UTC, git --date=format-local is +0300). Zero anomalies.
- "The skill is broadly untrustworthy" — ~50 of ~73 claims verify TRUE against source.
