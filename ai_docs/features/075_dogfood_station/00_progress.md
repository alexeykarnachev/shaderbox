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
