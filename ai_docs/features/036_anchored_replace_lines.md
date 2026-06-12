# 036 — Anchored replace_lines (text locates, numbers are hints)

**Status:** implemented 2026-06-12 on the Pi (headless: backend + capabilities + tools + prompt +
pytest — `make check` clean, suite green; the 27 GL-backed tests skip there). Remaining on the dev
box: `make test` (full, incl. the GL-backed guard test), `make smoke`, and the manual
`make run-bundle` pass against codex-mini. A 3-reviewer post-impl swarm converged after one fix
round (its real findings: a GL-skipped test asserting the old batch-guard message; the
`edit_shader` comment-loss reject still steering at "addressed by line number" — both fixed).

## Goal

Kill the line-coordinate failure class in `replace_lines` by removing coordinates from the wire.
The 2026-06-12 bundle trace (codex-mini, `copilot_default_2026-06-12_09-37-56`) showed the class
is systematic, not random: 2/8 ranged calls failed, both with `end_line = correct + 1` landing on
a blank line — the Python half-open-range prior leaking into our inclusive-end schema. The model
quoted the boundary-line TEXT perfectly in both failures; only the numbers were wrong. Models
copy text reliably and count unreliably — so the tool now locates by the text and drops the
numbers entirely.

## The two trace failures (regression fixtures — the trace file is volatile `/tmp`, this is the record)

1. **Off-by-one (the systematic class).** `replace_lines(start_line=37, end_line=44,
   first_line="    vec3 base = mix(vec3(0.05, 0.2, 0.5), ...", last_line="    color *= 1.0 -
   0.25 * pow(radius, 2.0);")` — line 44 was BLANK; the quoted last_line text was the exact
   content of line 43. Engine replied "did you mean end_line=43?" and burned an LLM round-trip.
   Under 036: both anchors strip-match uniquely → resolves to 37..43 → applies, zero round-trips.
2. **Garbled anchor (must STILL reject).** `first_line="    vec2 text_center_offset = vecce of
   code?"` — the model corrupted the quote mid-string (real line: `vec2 text_center_offset =
   vec2(0.0, 0.0);`); end_line was also +1-on-blank. Under 036: first_line matches nothing →
   reject naming the anchor; intent is unverifiable, same safety as before.

Same session: 6/8 ranged calls used inclusive numbers correctly — the model is bimodal, not
consistently half-open, which is why flipping the convention (or any number-side fix) loses.

## Lineage

- 033 added the boundary checksums (numbers locate, text verifies, mismatch rejects) and rejected
  a `replace_between` NEW tool on tool-count discipline.
- 034 F04 added whole-file mode (the dominant case) and kept ranged mode pending trace evidence.
- The `todo.md` deferral "ranged replace_lines may be dead weight" triggered on that trace: ranged
  calls DID keep tripping the boundary check — but 6/8 succeeded as real targeted edits, so the
  delete-ranged-mode arm is refuted. Third path (this feature): keep ranged, re-anchor it.
  Deferral closed.

## Design

Same tool, new args. Ranged mode locates the block by its boundary-line text:

| arg | was | now |
|---|---|---|
| `start_line`/`end_line` | required locators (1-based, inclusive) | **gone** |
| `first_line`/`last_line` | verbatim checksums of the boundary lines | **the locators** (strip-matched) |
| `near_line` | — | optional 1-based hint, consulted ONLY when an anchor matches several lines |
| `new_text`, `target` | unchanged | unchanged |

Whole-file mode unchanged: omit both anchors.

Resolution rules (per anchor, strip-equality against the target's current lines):
- multi-line quote → reject ("must be exactly ONE line").
- blank/empty quote → reject (anchor must be a content line).
- 0 matches → reject naming the anchor ("does not match any line — copy it verbatim from the
  working set"). Garbled quotes (the trace's `"vecce of code?"`) stay safely rejected.
- 1 match → located.
- N matches → `near_line` picks the closest (tie rejects); without `near_line`, reject listing
  the candidate line numbers so the model fixes it in one step.
- resolved `first > last` → reject ("anchors in reverse order", naming both resolved lines).
- empty/new target source → reject pointing at whole-file mode.

The success result echoes the resolved span (`replaced lines 26-36`) so the model sees where the
edit landed — a mislocated anchor becomes visible immediately. Anchor-resolution failures set
`unresolved` (count toward the edit-retry cap), same as the old checksum rejects.

False-positive analysis (vs the interim auto-snap idea): with no numbers there is no
number-vs-text conflict to adjudicate. The residual risk is a quoted anchor that uniquely matches
a line the model did NOT mean — requires the model to copy real text from the wrong place AND
that text to be globally unique; the span echo, the compile gate, the render probe, and per-turn
Revert bound the damage. The systematic +1 class this replaces had a 25% observed hit rate.

## Implementation

- `backend.py`: `apply_anchored_edit(first_line, last_line, near_line, new_text, target)` —
  resolves anchors via `_locate_anchor`, then shares the splice/persist path with
  `apply_line_edit` (which keeps insert + whole-file modes for `insert_after` and whole-file
  replaces; its now-unused `first/last_line` checksum params and `_range_check_error` are
  deleted). The D9 one-line-edit-per-file-per-step batch guard stays (reworded — its old message
  cited shifting line numbers).
- `capabilities.py`: Protocol gains `apply_anchored_edit`; `apply_line_edit` loses the checksum
  params; `EditResult.applied_span` carries the resolved-range echo.
- `tools/shader.py`: `_ReplaceLinesArgs` reshaped; handler routes whole-file vs anchored;
  descriptions rewritten (no line-number language in ranged mode).
- `prompt.py` EDITING bullets: ranged `replace_lines` = quote the block's first + last line
  verbatim; `near_line` only on ambiguity.
- `agent.py` stale-arg redaction: `near_line` joins `_LINE_ARG_KEYS`; the `_stale` marker reworded
  to cover text anchors, not just line numbers (prior-step anchor quotes go equally stale).
- Out of scope, noted: `insert_after` still takes a raw line number (same fragility class, zero
  observed failures, and a wrong insert is compile-gated). Trigger to revisit: a trace showing a
  misplaced insert.

## Verification

- Unit: anchor resolution (unique / duplicate+near_line / duplicate without hint lists candidates
  / garbled / blank / reversed / single-line block where first==last), the two trace failures as
  regression fixtures (failure 1 now applies; failure 2 still rejects), whole-file + insert modes
  unchanged, span echo present, retry-cap counts anchor rejects.
- Manual: maintainer drives the bundle (`make run-bundle`) against codex-mini.
- The next dogfood mission measures the edit-error rate delta (analyzer wave tracks per-tool
  error rates — `todo.md`).
