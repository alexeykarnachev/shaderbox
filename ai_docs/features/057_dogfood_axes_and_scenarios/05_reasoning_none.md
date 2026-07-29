# Reasoning effort: "minimal" -> "none" (wave 1 of the hub-feedback refactor)

Measured 2026-07-29 against `openai/gpt-5.1-codex-mini` via OpenRouter, plain API probes:
`effort: "minimal"` is honored on a bare request (rsn=0) but IGNORED once our system prompt +
tools ride along (rsn ~100% of output; the exam run burned 332k of 428k output tokens on hidden
reasoning). `effort: "none"` (new in the GPT-5.1 family) is honored: 88 out tokens vs 1567,
tool_call fires on the first iteration, ~2x cheaper. Knob: `COPILOT_ENGINE.llm_reasoning_effort`
(wave 2 moved it off the user config onto `CopilotEngineConfig` — it is not Settings-tunable).

Echelon gate (one run each, effort=none): 04 one-shot PASS period exactly 2.0s ($0.007);
05 one-shot PASS parabolic arcs + rest ($0.027); 08 PASS with 1 correction — grid separator
lines missing initially, all 9 figures/timings exact incl. blink 2/s verified at 0.25s sampling
($0.094). No time-budget hits. No quality-degradation class attributable to the flag.
