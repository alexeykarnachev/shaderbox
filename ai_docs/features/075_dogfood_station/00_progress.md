# 075 — progress log

A log of what happened, not a plan. **Append after every wave, before starting the next.**

On resume: read this file and `git log --oneline -15` FIRST. They are the truth about where the
work stopped; `01_spec.md` is only the plan.

Format per wave:

```
## W-N <name> — DONE | SKIPPED | ABANDONED   <sha>
done-condition: <the checkable statement, written BEFORE the wave started>
did: <what changed>
verification: make gates <green|red>, exit code read unpiped
ruled out: <what was considered and rejected, and why>
surprise: <anything the spec did not predict>
```

---

## Baseline — measured before any wave

At `3bf54ab` on `dev`, tree clean. `make gates` exit 0, GREEN, all three stages. 1706 tests.

Verified directly while writing the spec, not assumed:

- `openai/gpt-5.6-luna` IS available on OpenRouter (2026-09-04): tool-capable, 1.05M context,
  $0.20/$1.20 per Mtok — cheaper than the codex-mini default at $0.25/$2.00 with 2.6x the context.
- `openai/gpt-5.1-codex-mini` is still present (400k ctx), so the in-tree default is not stale.
- Cached-token plumbing already exists end to end: `LLMUsage.cached_tokens` is populated from
  `prompt_tokens_details.cached_tokens` in `openrouter.py`, summed across iterations, and
  `analyze.py` already parses a `cache=` field from the trace. W-1 joins it to the breakdown
  rather than building it.
- `context_breakdown` is genuinely unimplemented: deferred in `026_copilot_dogfood_harness.md`
  and still named as deferred in `057_dogfood_axes_and_scenarios/01_spec.md`. `build_prompt`
  composes five named blocks in `prompt.py`; nothing records their sizes.
- The dogfood harness is mature — ~30 public methods including `render_strip`, `script_values`,
  `render_video_mp4`, `clear_context`, resume-by-project-dir. The station OBSERVES it; W-3 wires
  logging in rather than reshaping how turns are driven.

## Spec revised twice before any wave started

The spec was written, then reviewed against what the maintainer actually said, then corrected
again on his instruction. Both rounds are recorded because a session reading only the final text
would not know which parts were argued for.

**Round 1 (`b2c7f0f`) — seven gaps, four of them things stated in conversation that no line of
the spec carried.** The driving discipline was missing entirely (driver-as-real-user, the mode
chosen up front, and the three-way rule when the copilot is stuck: file it / sub-feature it /
stop and ask). So were videos and gifs as artifacts, the first experiment's actual target, and
the "sizes AND concrete text" halves of the context view. Two gaps were omissions rather than
errors: the spec did not cite feature 027, which was drafted as a dogfood SERVER and had that
killed by a devil's-advocate pass on the same reasoning D1 reaches independently; and it said
nothing about the seven existing dogfood reports or the six report axes.

The W-1 notes were rewritten from a reading of `prompt.py` rather than from memory, which turned
up the fact that decides where the wave's code goes: **`working_set` renders `[]` at build time
and is spliced in live per iteration by `run_turn`**, so a breakdown emitted at `build_messages`
would report the block holding the actual shader source as EMPTY. Its falsifier is now a
done-condition. Same for `dialogue` being trimmed, where the trim is itself the signal.

**Round 2 (`89fee18`) — the standing oracle was struck, by explicit instruction.** The first
draft told the RC experiment to check the copilot's output against feature 068's numerical
oracle. The maintainer's words: *"Don't rely on the scripted oracles. This never works, you are
overfitting the noise. I AM THE FINAL ORACLE."* So no fixed checker is wired into any run's
verdict; judgement is his, by looking at renders and videos ad hoc. Ad-hoc measurement answering
ONE question and then deleted is still welcome — the distinction is what a measurement CLAIMS,
not how it is computed. **Do not re-introduce this**; the "station does not judge" bullet now
warns that it is the line most likely to be mistaken for an improvement later.

## W-0 event log + writer — DONE   0de87cd
done-condition: the writer appends valid JSONL under concurrent-ish use (one writer object per
appender, open+append+close each time), a reader reconstructs a full run from the log alone, and
a test pins that an interrupted write does not corrupt the file.
did: `dogfood/report/log.py` — `EventLog.append` (one `write()` per line on a freshly opened
append handle, fsync'd; a torn predecessor is repaired by prefixing a newline so the fragment
stays its own line), `read_events` (one warning per unreadable line, never a raise), and the
typed `Experiment` / `Attempt` / `TurnRecord` / `ContextRecord` tree `reconstruct` folds records
into. Context records that arrive before their turn record are joined by turn number; ones whose
turn never landed are kept as `orphan_contexts`, so a killed process still leaves what the
copilot was sent. `load_store` lists experiments newest-activity first. Seven tests in
`tests/test_dogfood_log.py`, one of them enumerating `KINDS` so a kind added without a
reconstruct branch fails.
verification: make gates GREEN (exit 0, read unpiped), smoke ran on this box.
ruled out: a shared lock file or a single long-lived handle — the harness's turn is its own
process, so nothing can be shared; O_APPEND single-write lines are the whole concurrency story.
surprise: none.

## W-1 `context_breakdown` — DONE   9266c6f
done-condition: a test asserts every block appears in the event, enumerated from the block list;
a test asserts `working_set` is reported non-empty and LIVE (its text changes between iterations
when the splice changes); the composed prompt is byte-identical with a listener attached and
without; a trimmed dialogue reports `trimmed` and the dropped count. The billed-vs-estimate
tolerance needs a real request, so it is measured in W-3 when the recorder joins the two.
did: `prompt.py` splits `build_messages` into `build_blocks` (the tier list) + `render_blocks`
(each tier rendered ONCE, kept per tier) + `build_prompt` (the flattened view); the two tiers
other code addresses by name are constants (`DIALOGUE_BLOCK`, `WORKING_SET_BLOCK`).
`context_breakdown.py::breakdown_request` measures one request: every built tier, the
within-turn tool exchange (`turn_exchange`), the LIVE working-set splice, and the `tools=` block
(compact-JSON chars / 4), each with its full text. `run_turn` emits it as a `context_breakdown`
trace event at both places a request is assembled (the loop and the forced final reply).
`TraceLog` grew a listener list — every event also reaches `(kind, fields)` observers as the
structured objects, on the emitting thread, exceptions swallowed with a warning — and
`CopilotSession.trace_listeners` threads it through every trace rotation. The transcript renders
the breakdown as a sizes table (the texts are the `llm_request` just above it).
verification: make gates GREEN (exit 0, read unpiped).
ruled out: emitting at `build_messages` (the working set is empty there — the trap the spec
names); parsing the breakdown back out of the plain-text transcript (the listener seam exists so
nobody has to); a second `LLMClient` wrapper to observe requests (the trace already sees them).
surprise: `RUF005` on a list concatenation, otherwise none.
