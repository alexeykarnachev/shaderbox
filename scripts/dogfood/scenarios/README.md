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

After a run, produce a FINAL REPORT (`ai_docs/features/NNN_dogfood_report_<run>.md`, ONE per scenario)
from `scripts/dogfood/REPORT_TEMPLATE.md` — six axes (fidelity · motion · logic · honesty · process ·
code) + the dialogue — with follow-up TODOs: what's next + ideas to improve (a) the copilot AGENT, (b) the
dogfooding FRAMEWORK, (c) the library. Send it to the maintainer. The `code` axis grades the run's FINAL
sources as a code reviewer (dead code, duplication, structure, complexity, tool choice) — the copilot's
job is EXCELLENT CODE; the visual call is the human's.

**Judge tooling** (don't hand-roll it — that's what the 043 run did three times):
`analyze.py <data_dir> --scenario <name>` fills the AUTO slots (incl. the honesty axis's limit-forced
turns);
`analyze.py <data_dir> --dialogue` prints the dialogue; `h.render_strip(times, node_id)` is the motion
contact sheet; `h.script_values(times, node_id)` the logic probe; `scripts/dogfood/judge.py` the pixel
primitives (centroids, column runs, region diffs, rotation angle).

## Scenarios

| File | Final goal | Pressure axes |
|---|---|---|
| `01_shape_gallery.md` | a 2×2 grid of 4 simple 2D shapes (circle/square/triangle/ring), built as separate nodes then COMBINED by a memory-wiped fresh agent | tool-use under a context wipe (read-from-disk vs hallucinate) · visual honesty · token-growth observation · full reachable-tool sweep (grep/read_lib/set_uniform/switch_node/delete_node/render_image) |
| `02_logo_design.md` | a polished, maintainer-accepted brand image (logo / hero / icon) refined over many turns of art direction, saved to `docs/branding/` at 1024 | visual blindness (the hard probe — judged only by eye) · precise-diagnosis dependency · palette drift · free-brief over-reach · no-op-clean spree / EDIT UNDONE cost trap · animation-invisible-on-a-still framework gap · human-in-the-loop direction via SendUserFile |

## Cornerstone scenarios (the second type — 03+)

A **cornerstone** is the opposite shape of a mission: ONE base capability, isolated, with GROUND
TRUTH — never several axes mixed (the all-axes flag run proved a mixed scene can't attribute a
failure). The contract:

- The driver sends the scenario's **opening message VERBATIM** (it is part of the fixture — don't
  paraphrase), then behaves as a terse real user: judge the result, and if it misses, send at most
  **2 correction messages** (each states WHAT is wrong, never HOW to fix it — "the ball orbits too
  fast", not "divide by 2"). Ideal outcome = one message, end-to-end.
- **Metric = user-message count** (1 = perfect, 2-3 = corrected, fail = still wrong after the
  budget), plus the usual cost/iteration numbers.
- **Report = the dialogue + the final render** (a 5-10s `h.render_video_mp4` for anything animated,
  a PNG for statics), sent to the maintainer; the dialogue is PRINTED by
  `analyze.py <data_dir> --dialogue` — never retyped from memory. Its source is the UI chat store
  (`<project>/copilot/conversation.json` `messages`), i.e. exactly what the user SAW; on a
  context-wipe run that store is re-saved EMPTY, so the printer falls back to the per-turn `dump()`
  `new_messages`; the pre-wipe text also survives as
  `<project>/copilot/archive/conversation_<stamp>.json`.
- Every checklist item is BINARY and checkable from the artifact (count it, time it, point at it);
  the ground truth (period, phase timings, trajectory) is stated in the file.

| File | Base capability | Ground truth |
|---|---|---|
| `03_static_comp.md` | pure-GLSL static composition | 5 equal circles, even row |
| `04_glsl_anim.md` | u_time animation (no script) | 2s orbit period, circular path |
| `05_bounce2d.md` | physics in `script.py` | gravity parabola + damped rest |
| `08_mixed_grid.md` | many simple constraints merged in one scene | 3x3 grid, 9 stated per-cell motions |

**Next (harder, later):** a composite that grades CODE QUALITY + a real token-overflow provocation +
trickier scenes — once the mechanism is obkatan on 01. Keep 01 simple ON PURPOSE: you must be able to judge
the render correct/wrong at a glance, which an SDF/3D/lighting scene defeats. The MIXED final
cornerstone (everything at once) comes only after the base set passes.

Add scenarios freely — the point is the weak-spot hunt, not coverage of a fixed list.
