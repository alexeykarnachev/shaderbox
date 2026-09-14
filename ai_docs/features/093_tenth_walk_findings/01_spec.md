# 093 — Tenth walk findings

Status: **pending; research done, design record awaiting the maintainer's review.** Five
findings filed (`00_findings.md`). The maintainer's verdict on the shipped canvas ("feels very
cheap": the wire anomalies, the controls, the card size, the hover cues) sent the walk into
research first: `02_research_brief.md` is the brief, `research/` holds six area reports against
primary sources, and **`03_graph_design.md` is the design record** (eighteen decisions G1-G18,
what stays, four open questions G-Q1..Q4, a verification sketch). He also chose mock C of
`00_mock_panel.html`: the graph moves into the editor pane as a third tab kind
(`01_research_graph_tab.md`). NEXT: he reviews `03_graph_design.md` and answers G-Q1..Q4; then
this spec's Goal / Design decisions / Waves are written from the record and the tab move, and
the run is planned as one feature flow (pre-implementation review, implement, review to
closure).

Source: `../TODO` (2026-09-13) and the screenshot of the control panel he took with it, verbatim
in the ledger.

## How this walk runs

- A batch of findings he reports in one message is one **wave**: research each finding against
  the code, file it in the ledger with what the code does and why, fix the small ones together,
  run `make gates` once, commit once, update this spec's wave list and the ledger's "Landed in"
  column in that commit.
- A finding whose fix is a feature by the `dev_flow.md` size preamble (a new module, a real
  behavior change, a reshape of a persisted model) gets its own numbered feature with the usual
  flow; the ledger row points at it and this spec lists it under the wave as delegated.
- A visual call stays his: no window manager here, so a fix to how something looks is described
  (what changed, where to look) and he judges it in the next batch.
- A settled decision (092's D-list, `conventions.md`) stays settled; a finding that shows one
  wrong reverses it explicitly in that feature's Review history, never silently.

## Goal

(filled by the first wave)

## Out of scope

(filled per wave, each deferral with a trigger)

## Design decisions

(filled per wave, numbered, lock-in only)

## Waves

(none yet)

## Files touched

(filled per wave)

## Open questions for the user

None yet.
