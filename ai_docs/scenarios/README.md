# Dogfood scenarios

Free-form markdown checklists a HUMAN (Claude) reads and drives by hand through `scripts/dogfood.py`
(feature 026). NOT auto-run — there is no parser. Each `User:` line is something you type to the
copilot via `h.send(...)`; `Expect:` is what the trace should show; `Human check:` is what YOU verify
(the trace claim vs the rendered PNG you open with Read). The judge is you, not code.

The high-value scenarios deliberately probe the `ai_docs/todo.md` copilot deferrals — the reasons the
copilot isn't shipped yet. When one fires, that's the signal the deferral is real and worth building.

## How to drive one

```
export OPENROUTER_API_KEY=...          # required — billed; the harness reads it at import
uv run python                          # interactive REPL
>>> from scripts.dogfood import DogfoodHarness
>>> h = DogfoodHarness.create()        # or create(seed_templates=False) for an empty project
>>> h.send("...")                      # a User: line from the scenario
>>> h.drive_until_idle()               # pump; prints events; STOPS on a gate
>>> h.approve()  / h.decline()         # answer a gate, then drive_until_idle() again
>>> png = h.render()                   # 400x400 PNG of the current node
# then open `png` with Read and eyeball it against the Human check
```

Read the trace transcript (under the isolated `SHADERBOX_DATA_DIR`'s `copilot_traces/`) for the
per-turn context + token breakdown — the other half of the dogfood (where are tokens going?).

## Scenarios (priority order — weak-spot probes first)

| File | Probes | todo.md deferral |
|---|---|---|
| `01_visual_blindness.md` | copilot claims a visual result it can't see | machine-readable render feedback |
| `02_wrong_node_targeting.md` | "this"/"the X" resolves to the wrong node | demonstrative → current-node |
| `03_compile_thrash.md` | applies-but-broken edits loop to max_iterations | broken-compile circuit-breaker |
| `04_circle_square_morph.md` | multi-file read + aggregate (content load-bearing) | — (core capability) |
| `05_grep_readlib_uniforms.md` | grep→read_lib two-step + set_uniform edge cases | — (core capability) |
| `06_agent_error_recovery.md` | does the agent READ a failure + self-correct, or loop? | agent-level recovery (the biggest gap) |

**Run priority:** 06 (agent-level recovery) + 03 (compile-thrash) are the highest-value UNRUN probes — the
2026-06-09 run only exercised happy paths. Run those next.

Add scenarios freely — the format is loose on purpose.
