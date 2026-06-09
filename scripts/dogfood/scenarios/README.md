# Dogfood scenarios

Goal-driven, weak-spot-hunting MISSIONS (features 026/027) a human (Claude) drives by hand through the
harness (`scripts/dogfood/harness.py`). NOT auto-run, NOT parsed — you read the whole scenario, then drive
the copilot LIVE turn by turn, composing each message from what the agent actually did. The judge is YOU
(the trace + the rendered PNGs you open with Read) — there are no code assertions.

A scenario is a **final GOAL + an iterative build-up toward it + the pressure axes it attacks**, in
minimally-structured free text — never a numbered step-script (that would just get replayed). Usually the
goal is a COMPOSITE render (many shaders combined) that can only exist if the agent built pieces and then
combined them — and the build-up deliberately routes through the copilot's weak spots (tool-use under a
context wipe, visual blindness, token growth, targeting in a multi-node project).

## How to drive

ONE blocking `uv run` per turn; state persists on disk via resume/dump; `clear_context()` wipes the
agent's memory for the cold-start half. The exact commands + the model + the `~/.bashrc` key footgun + the
single-process gate rule are all in the **`/dogfood` skill** (the operating manual) — don't re-derive them
here. All run artifacts live under `scripts/dogfood/runs/` (gitignored).

After a run, produce a FINAL REPORT (`ai_docs/features/026_dogfood_report_<run>.md`) with follow-up TODOs:
what's next + ideas to improve (a) the copilot AGENT, (b) the dogfooding FRAMEWORK, (c) the library. Send
it to the maintainer.

## Scenarios

| File | Final goal | Pressure axes |
|---|---|---|
| `01_shape_gallery.md` | a 2×2 grid of 4 simple 2D shapes (circle/square/triangle/ring), built as separate nodes then COMBINED by a memory-wiped fresh agent | tool-use under a context wipe (read-from-disk vs hallucinate) · visual honesty · token-growth observation · full reachable-tool sweep (grep/read_lib/set_uniform/switch_node/delete_node/render_image) |

**Next (harder, later):** a composite that grades CODE QUALITY + a real token-overflow provocation +
trickier scenes — once the mechanism is obkatan on 01. Keep 01 simple ON PURPOSE: you must be able to judge
the render correct/wrong at a glance, which an SDF/3D/lighting scene defeats.

Add scenarios freely — the point is the weak-spot hunt, not coverage of a fixed list.
